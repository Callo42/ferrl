//! M2′ bf16 GPU gates for the `qwen3_5` path (`#[ignore]`d).
//!
//! The CPU suites validate the hybrid forward at F32, where the fp32
//! boundaries the reference mandates (delta-rule state/gates, attention
//! softmax) are same-dtype casts — op-free clones, structurally absent from
//! the graph. These gates run a **real, staged** dense qwen3.5/3.6 checkpoint
//! (0.8B / 9B / 27B …, read from its `config.json`) at **bf16 on CUDA**, where
//! every one of those boundaries becomes a real `ToDType` node and the candle
//! CUDA bf16 kernels actually execute:
//!
//! 1. **bf16 logit fidelity vs the fp32 transformers dump**
//!    (`qwen35_bf16_forward_matches_reference_on_gpu`): argmax agreement +
//!    relative magnitude across every dumped position — crossing
//!    implementation (torch vs candle), device, AND dtype.
//! 2. **the `ToDType` backward** (`qwen35_bf16_dtype_split_grads_on_gpu`):
//!    bf16-base / F32-adapter two-phase per-branch coverage; every gradient
//!    lands in the F32 master dtype.
//! 3. **bf16 merged-state fidelity**
//!    (`qwen35_merged_decoder_bf16_faithfulness_on_gpu`): the cached hybrid
//!    decoder (KV + bf16 conv state + fp32 delta-rule state) over a REAL
//!    generated continuation vs the uncached adapter-aware forward.
//! 4. **GRPO smoke** (`qwen35_policy_grpo_smoke_on_gpu`): two optimizer steps
//!    of the UNCHANGED generic `Trainer` driving `Qwen3_5Policy` end to end.
//!
//! Run on a GPU node (CUDA build; see the cluster PTX-ISA recipe):
//!
//! ```text
//! FERRL_QWEN35_WEIGHTS=/path/to/qwen3_5-0.8b-base \
//! FERRL_QWEN35_ORACLE=/path/to/qwen3_5-0.8b-base/ferrl_oracle_dumps \
//!     cargo test -p ferrl --features cuda --test qwen35_gpu_smoke -- --ignored
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use ferrl::policy::GenConfig;
use ferrl::{
    grad_coverage, varbuilder_from_pretrained, HfTokenizer, LayerType, Policy, Qwen3_5Config,
    Qwen3_5GradModel, Qwen3_5Policy, RewardFn, RunDir, TokenizerLike, Trainer, TrainerConfig,
};

/// `LoRA` rank / alpha for the smoke — a typical small adapter.
const RANK: usize = 8;
const ALPHA: f64 = 16.0;

fn weights_dir() -> PathBuf {
    PathBuf::from(std::env::var("FERRL_QWEN35_WEIGHTS").expect(
        "set FERRL_QWEN35_WEIGHTS to a staged dense qwen3.5/3.6 asset directory to run the GPU smoke",
    ))
}

fn oracle_dir() -> PathBuf {
    PathBuf::from(std::env::var("FERRL_QWEN35_ORACLE").expect(
        "set FERRL_QWEN35_ORACLE to the real_logits.safetensors directory to run the GPU smoke",
    ))
}

fn cuda() -> Device {
    Device::new_cuda(0).expect("CUDA device 0 — build with --features cuda and run on a GPU node")
}

fn load_cfg() -> Qwen3_5Config {
    Qwen3_5Config::from_json_file(weights_dir().join("config.json")).expect("parse config.json")
}

/// bf16 model on the GPU (the checkpoint's native dtype — the production
/// rollout regime) with the F32 adapter split.
fn load_bf16(device: &Device) -> Qwen3_5GradModel {
    let cfg = load_cfg();
    let vb = varbuilder_from_pretrained(weights_dir(), DType::BF16, device)
        .expect("load weights (bf16) onto the GPU");
    Qwen3_5GradModel::load_with_adapter_dtype(&cfg, &vb, RANK, ALPHA, DType::F32)
        .expect("build bf16-base/F32-adapter model")
}

fn load_oracle() -> HashMap<String, Tensor> {
    candle_core::safetensors::load(oracle_dir().join("real_logits.safetensors"), &Device::Cpu)
        .expect("load real_logits.safetensors")
}

fn oracle_ids(dump: &HashMap<String, Tensor>, i: usize) -> Vec<u32> {
    dump[&format!("p{i}_ids")]
        .to_dtype(DType::U32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Argmax token id of a 1-D `[vocab]` logits row.
fn argmax_u32(row: &Tensor) -> u32 {
    row.argmax(D::Minus1).unwrap().to_scalar::<u32>().unwrap()
}

/// Max-abs of a row, in F32.
fn abs_max(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .unwrap()
        .abs()
        .unwrap()
        .max(D::Minus1)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// Per-position argmax agreement + worst relative diff between `ours`
/// (`[1, L, vocab]`, any dtype/device) and `reference` (same shape, fp32 CPU).
fn fidelity(ours: &Tensor, reference: &Tensor) -> (usize, usize, f32, f32) {
    let l = reference.dims3().unwrap().1;
    let ours = ours
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let mut agree = 0usize;
    let mut max_abs = 0f32;
    let mut max_logit = 0f32;
    for t in 0..l {
        let our_row = ours.i((0, t)).unwrap();
        let ref_row = reference.i((0, t)).unwrap();
        if argmax_u32(&our_row) == argmax_u32(&ref_row) {
            agree += 1;
        }
        max_abs = max_abs.max(abs_max(&our_row.sub(&ref_row).unwrap()));
        max_logit = max_logit.max(abs_max(&ref_row));
    }
    (agree, l, max_abs, max_logit)
}

/// Set every `LoRA` `B` factor (odd index) to small DETERMINISTIC noise in
/// the var's OWN dtype (the F32 masters here) — a gate failure must be
/// replayable, and CUDA `randn` is not seedable through candle.
fn force_b_nonzero(vars: &[Var], device: &Device) {
    for (i, v) in vars.iter().enumerate() {
        if i % 2 == 1 {
            let dims = v.as_tensor().dims().to_vec();
            let n: usize = dims.iter().product();
            let vals: Vec<f32> = (0..n)
                .map(|e| 0.02 * ((e + i) as f32 * 0.618_034).sin())
                .collect();
            let noise = Tensor::from_vec(vals, dims, device)
                .unwrap()
                .to_dtype(v.dtype())
                .unwrap();
            v.set(&noise).unwrap();
        }
    }
}

/// Split the default-recipe vars into (attention, mlp) branches via the
/// config's layer kinds (linear → 6 MLP vars; full → 8 attention + 6 MLP).
fn branch_split(cfg: &Qwen3_5Config, vars: &[Var]) -> (Vec<Var>, Vec<Var>) {
    let mut attn = Vec::new();
    let mut mlp = Vec::new();
    let mut i = 0usize;
    for kind in cfg.text_config.resolved_layer_types() {
        if kind == LayerType::FullAttention {
            attn.extend(vars[i..i + 8].iter().cloned());
            i += 8;
        }
        mlp.extend(vars[i..i + 6].iter().cloned());
        i += 6;
    }
    assert_eq!(i, vars.len(), "branch split must consume every var");
    (attn, mlp)
}

/// The default-recipe `LoRA` var count for a *dense* member, derived from the
/// staged config: MLP (3 projections) on every layer + attention (4
/// projections) on the full-attention layers, 2 vars (A, B) per projection.
/// The config-driven generalization of the old 0.8B-hardcoded `(24*3 + 6*4)*2`
/// (192 on 0.8B, 256 on 9B), so the gate scales to 9B/27B without an edit.
fn expected_default_var_count(cfg: &Qwen3_5Config) -> usize {
    let kinds = cfg.text_config.resolved_layer_types();
    let n_full = kinds
        .iter()
        .filter(|k| **k == LayerType::FullAttention)
        .count();
    (kinds.len() * 3 + n_full * 4) * 2
}

/// A reward that spreads across distinct completions (position-weighted so
/// byte-multiset collisions don't collapse the group).
struct SpreadReward;
impl RewardFn for SpreadReward {
    fn reward(&self, _prompt: &str, completion: &str) -> f32 {
        completion
            .bytes()
            .enumerate()
            .map(|(i, b)| (i as f32 + 1.0) * f32::from(b))
            .sum::<f32>()
            / 1000.0
    }
}

/// A unique temp directory, removed on drop.
struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("ferrl-qwen35-gpu-{}-{}", std::process::id(), nanos));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "needs a staged dense qwen3.5/3.6 checkpoint + oracle dumps + a CUDA build/GPU"]
#[allow(clippy::print_stderr)] // a manual gate: the printed agreement/diff numbers are the deliverable
fn qwen35_bf16_forward_matches_reference_on_gpu() {
    // Gate 1: bf16 GPU logits vs the fp32 transformers dump — crossing
    // implementation, device, and dtype at once. The envelope is set by
    // bf16's ~2^-8 relative rounding over the model's hybrid layers; a broken fp32
    // boundary (state kept in bf16, softmax in bf16) or a CUDA-kernel-family
    // break lands comparable to the logit scale.
    let t0 = Instant::now();
    let device = cuda();
    let mut model = load_bf16(&device);
    model.set_adapter_enabled(false); // base only, like-for-like with the dump
    let dump = load_oracle();

    let mut i = 0usize;
    while dump.contains_key(&format!("p{i}_ids")) {
        let seq = oracle_ids(&dump, i);
        let input = Tensor::from_vec(seq.clone(), (1, seq.len()), &device).unwrap();
        let ours = model.forward(&input).expect("bf16 forward");
        assert_eq!(ours.dtype(), DType::BF16, "the forward runs in bf16");
        let (agree, l, max_abs, max_logit) = fidelity(&ours, &dump[&format!("p{i}_logits")]);
        let rate = agree as f64 / l as f64;
        let rel = max_abs / max_logit.max(f32::EPSILON);
        eprintln!(
            "[qwen35 bf16 GPU gate] p{i} ({l} positions): argmax {agree}/{l} = {rate:.4}; \
             max-abs {max_abs:.4} (scale {max_logit:.2}, rel {rel:.4})"
        );
        assert!(max_abs.is_finite(), "p{i}: non-finite bf16 divergence");
        // Same backstops as the M1 bf16 gates: honest bf16 noise sits at a few
        // percent of the logit scale (llama measured rel 0.0095, qwen 0.054);
        // 0.5 cleaves between noise and a real break.
        assert!(
            rel <= 0.5,
            "p{i}: bf16 diff {max_abs:.4} is {rel:.2}x the logit scale {max_logit:.2}"
        );
        assert!(
            rate >= 0.9,
            "p{i}: bf16 argmax agreement {rate:.4} below 0.9 — more than bf16 noise explains"
        );
        i += 1;
    }
    assert!(i >= 3, "expected >= 3 dumped prompts, found {i}");
    eprintln!("[qwen35 bf16 GPU gate] {:.0?} elapsed", t0.elapsed());
}

#[test]
#[ignore = "needs a staged dense qwen3.5/3.6 checkpoint + oracle dumps + a CUDA build/GPU"]
fn qwen35_bf16_dtype_split_grads_on_gpu() {
    // Gate 2: the ToDType backward. Under bf16, the delta-rule F32
    // state/gate boundaries, the softmax round-trip, AND the adapter's
    // master->bf16 downcasts are all real cast nodes; this backward
    // differentiates through every one of them on real weights.
    let device = cuda();
    let cfg = load_cfg();
    let mut model = load_bf16(&device);
    model.set_adapter_enabled(true);
    let vars = model.trainable_vars();
    assert_eq!(vars.len(), expected_default_var_count(&cfg));
    let (attn_vars, mlp_vars) = branch_split(&cfg, &vars);

    let dump = load_oracle();
    let seq = oracle_ids(&dump, 0);
    let input = Tensor::from_vec(
        seq[..8.min(seq.len())].to_vec(),
        (1, 8.min(seq.len())),
        &device,
    )
    .unwrap();

    let grads_of = |model: &Qwen3_5GradModel| -> GradStore {
        let logits = model.forward(&input).expect("bf16 forward");
        assert_eq!(logits.dtype(), DType::BF16);
        // Upcast BEFORE sqr/sum so the loss reduction cannot overflow bf16.
        logits
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .expect("bf16 backward")
    };
    let assert_grads_f32 = |grads: &GradStore| {
        for v in &vars {
            let g = grads
                .get(v.as_tensor())
                .expect("adapter var missing from grad store");
            assert_eq!(
                g.dtype(),
                DType::F32,
                "grad must land in the F32 master dtype"
            );
        }
    };

    let g1 = grads_of(&model);
    assert!(
        grad_coverage(&attn_vars, &g1).unwrap().is_ok(),
        "attention branch unhealthy at zero-B init under the bf16 split"
    );
    assert!(
        grad_coverage(&mlp_vars, &g1).unwrap().is_ok(),
        "mlp branch unhealthy at zero-B init under the bf16 split"
    );
    assert_grads_f32(&g1);

    force_b_nonzero(&vars, &device);
    let g2 = grads_of(&model);
    let ac = grad_coverage(&attn_vars, &g2).unwrap();
    let mc = grad_coverage(&mlp_vars, &g2).unwrap();
    assert!(
        ac.nonzero == ac.total && ac.nonfinite == 0,
        "attention branch: not every LoRA var live after nonzero-B at bf16: {ac:?}"
    );
    assert!(
        mc.nonzero == mc.total && mc.nonfinite == 0,
        "mlp branch: not every LoRA var live after nonzero-B at bf16: {mc:?}"
    );
    assert_grads_f32(&g2);
}

#[test]
#[ignore = "needs a staged dense qwen3.5/3.6 checkpoint + oracle dumps + a CUDA build/GPU"]
#[allow(clippy::print_stderr)] // a manual gate: the printed agreement/diff numbers are the deliverable
fn qwen35_merged_decoder_bf16_faithfulness_on_gpu() {
    // Gate 3: bf16 merged-STATE fidelity. The hybrid decoder's state is not
    // just folded weights: bf16 conv columns + fp32 delta-rule matrices roll
    // forward token by token over a REAL generated continuation, vs the
    // uncached adapter-aware forward over the same final sequence. The
    // expected small divergence is the documented bf16 merge-absorption bound
    // plus the chunked/recurrent kernel boundary.
    let device = cuda();
    let model = load_bf16(&device);
    let mut policy = Qwen3_5Policy::new(model, 1234, 1.0);
    // Arm the adapter so the merge carries a real signal.
    force_b_nonzero(&policy.trainable_vars(), &device);
    assert!(policy.adapter_enabled());

    let tok = HfTokenizer::from_file(weights_dir().join("tokenizer.json")).expect("tokenizer");
    let prompt_ids = tok.encode("The capital of France is");
    let gcfg = GenConfig {
        group_size: 1,
        max_new_tokens: 16,
        temperature: 1.0,
        eos_token_id: None,
        eval_sampling: None,
    };
    let seq = policy
        .generate(&prompt_ids, &gcfg)
        .expect("cached generate (the hybrid state path)")
        .token_ids[0]
        .clone();
    let len = seq.len();

    let input = Tensor::from_vec(seq.clone(), (1, len), &device).unwrap();
    let uncached = policy.model().forward(&input).expect("uncached forward");

    let mut dec = policy.model().merged_decoder().expect("merged decoder");
    let mut agree = 0usize;
    let mut max_abs = 0f32;
    let mut max_logit = 0f32;
    for (t, &id) in seq.iter().enumerate() {
        let tokt = Tensor::from_vec(vec![id], (1, 1), &device).unwrap();
        let cached_row = dec.forward(&tokt, t).unwrap().i((0, 0)).unwrap();
        let uncached_row = uncached.i((0, t)).unwrap();
        if argmax_u32(&cached_row.to_dtype(DType::F32).unwrap())
            == argmax_u32(&uncached_row.to_dtype(DType::F32).unwrap())
        {
            agree += 1;
        }
        let diff = cached_row
            .to_dtype(DType::F32)
            .unwrap()
            .sub(&uncached_row.to_dtype(DType::F32).unwrap())
            .unwrap();
        max_abs = max_abs.max(abs_max(&diff));
        max_logit = max_logit.max(abs_max(&uncached_row));
    }
    let rate = agree as f64 / len as f64;
    let rel = max_abs / max_logit.max(f32::EPSILON);
    eprintln!(
        "[qwen35 bf16 GPU gate] cached-vs-uncached over {len} positions: argmax \
         {agree}/{len} = {rate:.4}; max-abs {max_abs:.4} (scale {max_logit:.2}, rel {rel:.4})"
    );
    assert!(max_abs.is_finite());
    assert!(
        rel <= 0.5,
        "bf16 cached diff {max_abs:.4} is {rel:.2}x the logit scale {max_logit:.2}"
    );
    assert!(
        rate >= 0.9,
        "bf16 cached argmax agreement {rate:.4} below 0.9 — the hybrid state path is \
         shifting the rollout distribution more than bf16 absorption explains"
    );
}

#[test]
#[ignore = "needs a staged dense qwen3.5/3.6 checkpoint + oracle dumps + a CUDA build/GPU"]
fn qwen35_policy_grpo_smoke_on_gpu() {
    // Gate 4: one short GRPO run driving `Qwen3_5Policy` through the UNCHANGED
    // generic `Trainer` on CUDA — cached hybrid rollout -> reward ->
    // advantages -> backward through the bf16 chunked-GDN forward ->
    // grad-coverage canary -> FerrlAdamW. Two steps; `grad_norm > 0`
    // witnesses a real optimizer step (the M1 reusability bar, third model).
    let device = cuda();
    let model = load_bf16(&device);
    let mut policy = Qwen3_5Policy::new(model, 1234, 1.0);
    let tok = HfTokenizer::from_file(weights_dir().join("tokenizer.json")).expect("tokenizer");

    let prompts = vec!["The capital of France is".to_string()];
    let cfg_t = TrainerConfig {
        steps: 2,
        group_size: 4,
        max_new_tokens: 8,
        temperature: 1.0,
        lr: 1e-4,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new();
    let run = RunDir::create(&tmp.0, "qwen35-gpu-smoke").unwrap();
    let mut trainer = Trainer::new(cfg_t, &run).unwrap();

    let (history, _stop) = trainer
        .train(&mut policy, &SpreadReward, &tok, &prompts)
        .expect("GPU GRPO run failed");

    assert_eq!(history.len(), 2);
    for m in &history {
        assert!(
            m.grad_norm.is_finite() && m.reward_mean.is_finite(),
            "non-finite metric on GPU at step {}",
            m.step
        );
    }
    assert!(
        history.iter().any(|m| m.grad_norm > 0.0),
        "no AdamW step ran on GPU — the bf16 qwen3_5 backward path was never exercised"
    );
    assert!(policy.adapter_enabled());
}
