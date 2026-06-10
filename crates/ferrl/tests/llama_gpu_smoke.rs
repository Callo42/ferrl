//! M1 bf16 GPU gates for the dense-Llama path (`#[ignore]`d).
//!
//! The CPU suite validates [`ferrl::LlamaGradModel`] at F32, where the
//! attention F32 force-cast pair is a same-dtype `to_dtype` — an op-free
//! clone, structurally absent from the autograd graph. These gates run the
//! **real** `Llama-3.2-1B` checkpoint at **bf16 on CUDA**, where that cast
//! creates real `ToDType` nodes — closing exactly the three deferred bullets
//! in `llama.rs`'s module docs:
//!
//! 1. **bf16 logit equivalence vs shipped** (`llama_bf16_forward_matches_shipped*`):
//!    our grad-bearing forward vs candle's shipped `llama::Llama`, both at
//!    bf16 over the same weights, every position of a real prompt.
//! 2. **the `ToDType` backward through the attention cast**
//!    (`llama_bf16_dtype_split_grads*`): bf16-base / F32-adapter two-phase
//!    per-branch grad coverage — every `A` and `B` live + finite, with each
//!    gradient landing in the F32 master dtype.
//! 3. **bf16 merged-weight fidelity** (`llama_merged_decoder_bf16_faithfulness*`):
//!    the cached `LlamaMergedDecoder` rollout vs the uncached adapter-aware
//!    forward under the production dtype split.
//!
//! Like the real-weights gates this is `#[ignore]`d (needs the staged
//! checkpoint via `FERRL_LLAMA_WEIGHTS`) and additionally needs a CUDA build
//! and a GPU. Run it on a GPU node:
//!
//! ```text
//! FERRL_LLAMA_WEIGHTS=/path/to/llama-3.2-1b \
//!     cargo test -p ferrl --features cuda --test llama_gpu_smoke -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use ferrl::policy::GenConfig;
use ferrl::{
    grad_coverage, HfTokenizer, LlamaGradModel, LlamaPolicy, Policy, RewardFn, RunDir,
    TokenizerLike, Trainer, TrainerConfig,
};

/// `LoRA` rank / alpha for the smoke — a typical small adapter.
const RANK: usize = 8;
const ALPHA: f64 = 16.0;

/// The real prompt the equivalence gates run over (repeated words at distinct
/// positions stress causal-mask asymmetry and the scaled `RoPE`) — the same
/// prompt as the CPU real-weights gate, so the F32 and bf16 numbers are
/// directly comparable.
const EQ_PROMPT: &str = "The cat sat on the mat, and the cat slept.";

fn weights_dir() -> PathBuf {
    let dir = std::env::var("FERRL_LLAMA_WEIGHTS").expect(
        "set FERRL_LLAMA_WEIGHTS to the Llama-3.2-1B asset directory \
         (config.json + model.safetensors + tokenizer.json) to run the ignored \
         GPU smoke",
    );
    PathBuf::from(dir)
}

/// Parse `config.json` through candle's serde mirror (`LlamaConfig`) — the
/// same robust path as `llama_real_weights.rs` (unknown HF fields ignored,
/// `eos_token_id` via the Single/Multiple union). Non-flash.
fn load_config(dir: &Path) -> Config {
    let bytes = std::fs::read(dir.join("config.json")).expect("read config.json");
    let cfg: LlamaConfig = serde_json::from_slice(&bytes).expect("parse config.json");
    cfg.into_config(false)
}

/// bf16 [`VarBuilder`] over the real safetensors on the GPU (the checkpoint's
/// native dtype — no upcast, the production rollout regime).
fn load_vb_bf16(dir: &Path, device: &Device) -> VarBuilder<'static> {
    let buf = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    VarBuilder::from_buffered_safetensors(buf, DType::BF16, device)
        .expect("load model.safetensors (bf16) onto the GPU")
}

fn cuda() -> Device {
    Device::new_cuda(0).expect("CUDA device 0 — build with --features cuda and run on a GPU node")
}

/// Argmax token id of a 1-D `[vocab]` logits row.
fn argmax_u32(row: &Tensor) -> u32 {
    row.argmax(D::Minus1).unwrap().to_scalar::<u32>().unwrap()
}

/// Max-abs of a 1-D row, in F32.
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

/// Split the trainable vars into the (q, v) branches. Per-layer order is
/// `q_A, q_B, v_A, v_B`, so `i % 4 < 2` is the q branch.
fn branch_split(vars: &[Var]) -> (Vec<Var>, Vec<Var>) {
    let pick = |want_q: bool| -> Vec<Var> {
        vars.iter()
            .enumerate()
            .filter(|(i, _)| (i % 4 < 2) == want_q)
            .map(|(_, v)| v.clone())
            .collect()
    };
    (pick(true), pick(false))
}

/// Set every `LoRA` `B` factor (the odd index within each `[A, B]` pair) to a
/// small nonzero tensor in the var's OWN dtype (F32 masters here), so the
/// update is no longer a no-op and `dL/dA` is no longer structurally zero.
fn force_b_nonzero(vars: &[Var], device: &Device) {
    for (i, v) in vars.iter().enumerate() {
        if i % 2 == 1 {
            let dims = v.as_tensor().dims().to_vec();
            let noise = Tensor::randn(0f32, 0.02f32, dims, device)
                .unwrap()
                .to_dtype(v.dtype())
                .unwrap();
            v.set(&noise).unwrap();
        }
    }
}

/// A reward that spreads across distinct completions (so the sampled group is
/// non-degenerate and a real GRPO update fires). Position-WEIGHTED so completions
/// sharing a byte multiset don't collide to the same reward.
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
        p.push(format!("ferrl-llama-gpu-{}-{}", std::process::id(), nanos));
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
#[ignore = "needs the real Llama-3.2-1B checkpoint (FERRL_LLAMA_WEIGHTS) + a CUDA build/GPU"]
#[allow(clippy::print_stderr)] // a manual gate: the printed agreement/diff numbers are the deliverable
fn llama_bf16_forward_matches_shipped_every_position_on_gpu() {
    // Deferred bullet 1: bf16 logit equivalence vs shipped. BOTH models run
    // the SAME bf16 weights; the divergence between them is op-order / cast
    // FAMILY noise (fused vs grad-safe rms-norm / rope / softmax, reduction
    // order), not an adapter or merge effect — the adapter is disabled. Both
    // attention paths force-cast q/k/v to F32 (ours mirrors shipped), so this
    // is where that cast pair runs as a REAL bf16->F32->bf16 round trip for
    // the first time. The CPU F32 gate pins the same comparison at 1.7e-5;
    // at bf16 the envelope is set by bf16's ~2^-8 relative rounding instead.
    let t0 = Instant::now();
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = cuda();
    let vb = load_vb_bf16(&dir, &device);
    let shipped = Llama::load(vb.clone(), &cfg).expect("build shipped model");
    let mut ours = LlamaGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build our model");
    ours.set_adapter_enabled(false); // base only, like-for-like with shipped

    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let seq = tok.encode(EQ_PROMPT);
    let len = seq.len();
    let input = Tensor::from_vec(seq, (1, len), &device).unwrap();
    let ours_all = ours.forward(&input).expect("our forward"); // [1, len, vocab] bf16
    assert_eq!(ours_all.dims(), &[1, len, cfg.vocab_size]);

    // Shipped: last-position logits per growing prefix (already F32 — the
    // shipped forward upcasts its output), one shared uncached Cache (it only
    // memoizes causal masks; use_kv_cache=false keeps it stateless across
    // prefixes).
    let mut cache = Cache::new(false, DType::BF16, &cfg, &device).expect("shipped cache");
    let mut agree = 0usize;
    let mut max_abs = 0f32;
    let mut max_logit = 0f32;
    for t in 0..len {
        let prefix = input.narrow(1, 0, t + 1).unwrap();
        let shipped_row = shipped
            .forward(&prefix, 0, &mut cache)
            .expect("shipped forward")
            .i(0)
            .unwrap(); // [vocab] F32
        let ours_row = ours_all.i((0, t)).unwrap().to_dtype(DType::F32).unwrap(); // [vocab] F32
        if argmax_u32(&shipped_row) == argmax_u32(&ours_row) {
            agree += 1;
        }
        max_abs = max_abs.max(abs_max(&ours_row.sub(&shipped_row).unwrap()));
        max_logit = max_logit.max(abs_max(&shipped_row));
    }
    let rate = agree as f64 / len as f64;
    let rel = max_abs / max_logit.max(f32::EPSILON);
    let elapsed = t0.elapsed();
    eprintln!(
        "[llama bf16 GPU gate] ours-vs-shipped over {len} positions: argmax agreement \
         {agree}/{len} = {rate:.4}; max-abs logit diff {max_abs:.4} (logit scale \
         {max_logit:.2}, rel {rel:.4}); {elapsed:.0?} elapsed."
    );
    assert!(
        max_abs.is_finite(),
        "bf16 logits diverged non-finitely from shipped"
    );
    // A magnitude backstop the argmax rate can't see: same-weights op-family
    // noise must stay a small fraction of the logit scale; a broken cast path
    // lands comparable to the logits themselves. 50% cleaves between, with
    // wide margin against honest bf16 noise (the Qwen bf16 gate measured
    // rel 0.054 for a HARSHER comparison that also crossed the merge).
    assert!(
        rel <= 0.5,
        "bf16 max-abs logit diff {max_abs:.4} is {rel:.2}x the logit scale {max_logit:.2}"
    );
    assert!(
        rate >= 0.9,
        "bf16 argmax agreement {rate:.4} below 0.9 — ours and shipped disagree more than \
         same-weights bf16 op-order noise explains; investigate the F32 cast path"
    );
}

#[test]
#[ignore = "needs the real Llama-3.2-1B checkpoint (FERRL_LLAMA_WEIGHTS) + a CUDA build/GPU"]
fn llama_bf16_dtype_split_grads_on_gpu() {
    // Deferred bullet 2: the ToDType backward. Under the production
    // bf16-base / F32-adapter split, the forward runs in bf16 — so the
    // attention F32 force-cast pair and the adapter's master->bf16 downcasts
    // are REAL ToDType nodes, and this backward differentiates through them
    // (the CPU gates structurally cannot: at F32 those casts are op-free
    // clones; the CPU F32/F64 surrogate moves the adapter dtype but never
    // makes the ATTENTION cast real). Two-phase per-branch coverage at the
    // same bar as the CPU gates, plus the master-dtype routing property.
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = cuda();
    let vb = load_vb_bf16(&dir, &device);
    let mut model = LlamaGradModel::load_with_adapter_dtype(&cfg, &vb, RANK, ALPHA, DType::F32)
        .expect("build bf16-base/F32-adapter model");
    model.set_adapter_enabled(true);

    let vars = model.trainable_vars();
    assert_eq!(vars.len(), cfg.num_hidden_layers * 4);
    let (q_vars, v_vars) = branch_split(&vars);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let prompt = tok.encode("The capital of France is");
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &device).unwrap();

    let grads_of = |model: &LlamaGradModel| -> GradStore {
        let logits = model.forward(&input).expect("bf16 forward");
        assert_eq!(logits.dtype(), DType::BF16, "the forward runs in bf16");
        // Upcast BEFORE sqr/sum so the loss reduction itself cannot overflow
        // bf16; the upcast is part of the graph, like the trainer's own
        // completion-logit upcast.
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
                "master adapter must receive its grad in F32, not the bf16 compute dtype"
            );
        }
    };

    // Phase 1 — zero-B init: every var present in the grad store + each branch
    // live (via dL/dB) + finite, with every grad routed to the F32 master.
    let g1 = grads_of(&model);
    assert!(
        grad_coverage(&q_vars, &g1).unwrap().is_ok(),
        "q-branch grads unhealthy at zero-B init under the bf16 split (ToDType cut?)"
    );
    assert!(
        grad_coverage(&v_vars, &g1).unwrap().is_ok(),
        "v-branch grads unhealthy at zero-B init under the bf16 split"
    );
    assert_grads_f32(&g1);

    // Phase 2 — force every B nonzero: now EVERY A and B must carry a nonzero
    // finite F32 grad (the A-input path survives the real bf16 casts too).
    force_b_nonzero(&vars, &device);
    let g2 = grads_of(&model);
    let qc = grad_coverage(&q_vars, &g2).unwrap();
    let vc = grad_coverage(&v_vars, &g2).unwrap();
    assert!(
        qc.nonzero == qc.total && qc.nonfinite == 0,
        "q-branch: not every LoRA var is live after nonzero-B at bf16: {qc:?}"
    );
    assert!(
        vc.nonzero == vc.total && vc.nonfinite == 0,
        "v-branch: not every LoRA var is live after nonzero-B at bf16: {vc:?}"
    );
    assert_grads_f32(&g2);
}

/// Arm every `LoRA` `B` factor (odd `trainable_vars` indices) to small random values
/// so the merge carries a real, representative adapter signal (not the zero-B no-op).
fn arm_adapter(policy: &LlamaPolicy, device: &Device) {
    for (i, v) in policy.trainable_vars().iter().enumerate() {
        if i % 2 == 1 {
            let dims = v.as_tensor().dims().to_vec();
            let noise = Tensor::randn(0f32, 0.05f32, dims, device)
                .unwrap()
                .to_dtype(v.dtype())
                .unwrap();
            v.set(&noise).unwrap();
        }
    }
}

#[test]
#[ignore = "needs the real Llama-3.2-1B checkpoint (FERRL_LLAMA_WEIGHTS) + a CUDA build/GPU"]
#[allow(clippy::print_stderr)] // a manual gate: the printed agreement/diff numbers are the deliverable
fn llama_merged_decoder_bf16_faithfulness_on_gpu() {
    // Deferred bullet 3: bf16 merged-weight fidelity — the same REQUIRED gate
    // the Qwen path ran for P6-C, on the Llama twin. Cached `LlamaMergedDecoder`
    // (production rollout) vs the uncached adapter-aware forward under the
    // bf16-base / F32-adapter split — the regime CPU CI cannot reach (candle's
    // CPU backend has no bf16 matmul). The CPU gates pin the F32 family; this
    // reports the bf16 argmax-agreement rate + max-abs/relative logit diff.
    //
    // What differs, and why a small divergence is EXPECTED (not a bug): BOTH
    // paths run the adapter in bf16 (the forward casts A/B down to the base
    // dtype before the matmuls — the cast-order contract). The difference is
    // *where* the small adapter contribution lands: the cached path forms
    // `W + scale·BA` in bf16 WEIGHT space, so per-element deltas below
    // ~half-ulp(W) round into W and vanish (the documented ~0.2% absorption
    // bound); the uncached path adds the contribution in ACTIVATION space,
    // where it survives relative to the activation magnitude. The grad/scoring
    // path (always the uncached forward) is unaffected.
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = cuda();
    let vb = load_vb_bf16(&dir, &device);
    let model = LlamaGradModel::load_with_adapter_dtype(&cfg, &vb, RANK, ALPHA, DType::F32)
        .expect("build bf16-base/F32-adapter model");
    let mut policy = LlamaPolicy::new(model, 1234, 1.0);
    arm_adapter(&policy, &device);
    assert!(policy.adapter_enabled());

    // A realistic sequence: a generated continuation (cached path) of a real prompt.
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let prompt_ids = tok.encode("The capital of France is");
    let gcfg = GenConfig {
        group_size: 1,
        max_new_tokens: 16,
        temperature: 1.0,
        eos_token_id: None,
    };
    let seq = policy
        .generate(&prompt_ids, &gcfg)
        .expect("cached generate")
        .token_ids[0]
        .clone();
    let len = seq.len();

    // Uncached adapter-aware reference: full-sequence forward, all positions (bf16).
    let input = Tensor::from_vec(seq.clone(), (1, len), &device).unwrap();
    let uncached = policy.model().forward(&input).expect("uncached forward"); // [1, len, vocab]

    // Cached: prefill-free token-by-token decode, collecting every position.
    // Also track the logit *scale* (max-abs logit) so the diff can be judged
    // relatively — argmax agreement alone can stay high while logit magnitudes
    // blow up.
    let mut dec = policy.model().merged_decoder().expect("merged decoder");
    let mut agree = 0usize;
    let mut max_abs = 0f32;
    let mut max_logit = 0f32;
    for (t, &id) in seq.iter().enumerate() {
        let tokt = Tensor::from_vec(vec![id], (1, 1), &device).unwrap();
        let cached_row = dec.forward(&tokt, t).unwrap().i((0, 0)).unwrap();
        let uncached_row = uncached.i((0, t)).unwrap();
        if argmax_u32(&cached_row) == argmax_u32(&uncached_row) {
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
        "[llama bf16 GPU gate] cached-vs-uncached over {len} positions: argmax agreement \
         {agree}/{len} = {rate:.4}; max-abs logit diff {max_abs:.4} (logit scale \
         {max_logit:.2}, rel {rel:.4}). Interpret vs the ~0.2% bf16 merge-absorption bound."
    );
    assert!(
        max_abs.is_finite(),
        "bf16 cached logits diverged non-finitely from uncached"
    );
    // Same backstops as the Qwen P6-C gate (which measured 21/21, rel 0.054):
    // a correct bf16 merge sits at a few percent of the logit scale; a wrong
    // scale / dropped or corrupted delta lands comparable to the logits.
    assert!(
        rel <= 0.5,
        "bf16 cached max-abs logit diff {max_abs:.4} is {rel:.2}x the logit scale \
         {max_logit:.2} — a magnitude blow-up the argmax-agreement rate would miss"
    );
    assert!(
        rate >= 0.9,
        "bf16 cached argmax agreement {rate:.4} below 0.9 — the merge is shifting the \
         rollout distribution more than bf16 absorption alone explains; investigate"
    );
}

#[test]
#[ignore = "needs the real Llama-3.2-1B checkpoint (FERRL_LLAMA_WEIGHTS) + a CUDA build/GPU"]
fn llama_policy_grpo_smoke_on_gpu() {
    // The Llama twin of `qwen_policy_grpo_smoke_on_gpu`, run directly in the
    // production bf16-base / F32-adapter split: one short GRPO run driving
    // `LmPolicy<LlamaGradModel>` through the UNCHANGED generic `Trainer` on
    // CUDA — cached merged-decoder rollout -> reward -> advantages -> backward
    // through the bf16 Llama forward (the ToDType path under the real loss) ->
    // grad-coverage canary -> FerrlAdamW. Not a convergence test (two steps);
    // `grad_norm > 0` witnesses a real optimizer step.
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = cuda();
    let vb = load_vb_bf16(&dir, &device);
    let model = LlamaGradModel::load_with_adapter_dtype(&cfg, &vb, RANK, ALPHA, DType::F32)
        .expect("build bf16-base/F32-adapter model");
    let mut policy = LlamaPolicy::new(model, 1234, 1.0);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

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
    let run = RunDir::create(&tmp.0, "llama-gpu-smoke").unwrap();
    let mut trainer = Trainer::new(cfg_t, &run).unwrap();

    // A canary failure, a non-finite gradient, or an OOM would surface as an
    // error here; the run completing is itself most of the gate.
    let history = trainer
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
    // `grad_norm > 0` is set only when an AdamW step actually runs, so this
    // witnesses that the GPU backward through the bf16 Llama forward produced a
    // usable gradient, the canary passed, and the optimizer stepped.
    assert!(
        history.iter().any(|m| m.grad_norm > 0.0),
        "no AdamW step ran on GPU — the bf16 Llama backward path was never exercised"
    );
    // The adapter is restored enabled after the run.
    assert!(policy.adapter_enabled());
}
