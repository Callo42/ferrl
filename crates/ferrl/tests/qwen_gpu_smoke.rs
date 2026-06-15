//! P4-PR1 GPU smoke gate for [`ferrl::QwenPolicy`] (`#[ignore]`d).
//!
//! Drives the **real** `Qwen3-0.6B-Base` checkpoint through one GRPO run on a
//! CUDA device: rollout (uncached, adapter-aware sampling) -> reward -> advantages
//! -> backward through the grad-bearing Qwen forward -> grad-coverage canary ->
//! `AdamW`. It validates that the whole train path runs on a GPU without OOM and
//! produces finite metrics — the P4-PR1 gate. It is **not** a convergence test
//! (two steps); reward-rise vs a held-out eval is the later P4 gate.
//!
//! Like the P3 real-weights gates this is `#[ignore]`d (needs the staged
//! checkpoint via `FERRL_QWEN_WEIGHTS`) and additionally needs a CUDA build and a
//! GPU. Run it on a GPU node:
//!
//! ```text
//! module load nvhpc && export CC=gcc CXX=g++ CUDA_COMPUTE_CAP=80
//! FERRL_QWEN_WEIGHTS=/path/to/qwen3-0.6b-base \
//!     cargo test -p ferrl --features cuda --test qwen_gpu_smoke -- --ignored
//! ```
//!
//! Everything runs in F32 (the bf16 checkpoint is upcast on load); the bf16-base /
//! f32-adapter split is a later memory optimization (see `PLAN.md`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use ferrl::policy::{GenConfig, Rollout};
use ferrl::{
    evaluate, load_adapter, save_adapter, HfTokenizer, Policy, QwenGradModel, QwenPolicy, RewardFn,
    RunDir, TokenizerLike, Trainer, TrainerConfig,
};

/// `LoRA` rank / alpha for the smoke — a typical small adapter.
const RANK: usize = 8;
const ALPHA: f64 = 16.0;

fn weights_dir() -> PathBuf {
    let dir = std::env::var("FERRL_QWEN_WEIGHTS").expect(
        "set FERRL_QWEN_WEIGHTS to the Qwen3-0.6B-Base asset directory \
         (config.json + model.safetensors + tokenizer.json) to run the ignored \
         GPU smoke",
    );
    PathBuf::from(dir)
}

fn load_config(dir: &Path) -> Config {
    let bytes = std::fs::read(dir.join("config.json")).expect("read config.json");
    serde_json::from_slice(&bytes).expect("parse config.json into qwen3::Config")
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
        p.push(format!("ferrl-qwen-gpu-{}-{}", std::process::id(), nanos));
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
#[ignore = "needs the real Qwen3-0.6B-Base checkpoint (FERRL_QWEN_WEIGHTS) + a CUDA build/GPU"]
fn qwen_policy_grpo_smoke_on_gpu() {
    let dir = weights_dir();
    let cfg = load_config(&dir);

    // The GPU under test. Without `--features cuda` this errors at runtime, but the
    // test is `#[ignore]`d so the default CI build never reaches here.
    let device = Device::new_cuda(0)
        .expect("CUDA device 0 — build with --features cuda and run on a GPU node");
    let buf = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let vb = VarBuilder::from_buffered_safetensors(buf, DType::F32, &device)
        .expect("load model.safetensors onto the GPU");
    let model = QwenGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build QwenGradModel");
    let mut policy = QwenPolicy::new(model, 1234, 1.0);
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
    let run = RunDir::create(&tmp.0, "qwen-gpu-smoke").unwrap();
    let mut trainer = Trainer::new(cfg_t, &run).unwrap();

    // A canary failure, a non-finite gradient, or an OOM would surface as an error
    // here; the run completing is itself most of the gate.
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
    // `grad_norm > 0` is set only when an AdamW step actually runs, so this witnesses
    // that the GPU backward through the Qwen forward produced a usable gradient, the
    // canary passed, and the optimizer stepped — not just that the loop didn't crash.
    assert!(
        history.iter().any(|m| m.grad_norm > 0.0),
        "no AdamW step ran on GPU — the backward path was never exercised"
    );
    // The adapter is restored enabled after the run.
    assert!(policy.adapter_enabled());
}

#[test]
#[ignore = "needs the real Qwen3-0.6B-Base checkpoint (FERRL_QWEN_WEIGHTS) + a CUDA build/GPU"]
fn qwen_v2_resume_smoke_on_gpu() {
    // P6-B PR3 on CUDA: the momentum-faithful (v2) checkpoint save (CUDA optimizer
    // moments -> CPU safetensors) + `Trainer::resume` (load_checkpoint -> restore the
    // moments back onto the GPU via `FerrlAdamW::load_state` + restore the sampler RNG
    // -> continue). The CPU toy gate proves bit-identity; this proves the CUDA
    // save/restore path runs end-to-end on the real model without OOM / dtype / device
    // errors (GPU float determinism is not asserted here — that is the CPU gate's job).
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = Device::new_cuda(0)
        .expect("CUDA device 0 — build with --features cuda and run on a GPU node");
    let buf = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let vb = VarBuilder::from_buffered_safetensors(buf, DType::F32, &device)
        .expect("load model.safetensors onto the GPU");
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let prompts = vec!["The capital of France is".to_string()];
    let make_cfg = || TrainerConfig {
        steps: 2,
        group_size: 4,
        max_new_tokens: 8,
        temperature: 1.0,
        lr: 1e-4,
        checkpoint_every: Some(1), // a v2 checkpoint at step 1 (and the final step)
        ..TrainerConfig::default()
    };

    // Train 2 steps, checkpointing a v2 checkpoint at step 1.
    let model = QwenGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build QwenGradModel");
    let mut policy = QwenPolicy::new(model, 1234, 1.0);
    let tmp = TempDir::new();
    let run = RunDir::create(&tmp.0, "qwen-gpu-v2").unwrap();
    let mut trainer = Trainer::new(make_cfg(), &run).unwrap();
    trainer
        .train(&mut policy, &SpreadReward, &tok, &prompts)
        .expect("GPU train with v2 checkpointing failed");

    // The step-1 v2 checkpoint must carry the optimizer moments (the new CUDA save path).
    let ckpt = run.checkpoints_dir().join("step-1");
    assert!(
        ckpt.join("optimizer.safetensors").exists(),
        "a v2 checkpoint must persist optimizer.safetensors"
    );

    // Resume from it on the GPU into a FRESH model with a DIFFERENT sampler seed (so the
    // restore, not a shared seed, drives the continuation), running the remaining step.
    let model2 = QwenGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build resume model");
    let mut policy2 = QwenPolicy::new(model2, 9999, 1.0);
    let tmp2 = TempDir::new();
    let run2 = RunDir::create(&tmp2.0, "qwen-gpu-v2-resume").unwrap();
    let mut trainer2 = Trainer::new(make_cfg(), &run2).unwrap();
    let (resumed, _stop) = trainer2
        .resume(&ckpt, &mut policy2, &SpreadReward, &tok, &prompts)
        .expect("GPU resume from a v2 checkpoint failed");
    assert_eq!(
        resumed.len(),
        1,
        "resume from step 1 runs the one remaining step"
    );
    for m in &resumed {
        assert!(
            m.grad_norm.is_finite() && m.reward_mean.is_finite(),
            "non-finite metric on GPU resume at step {}",
            m.step
        );
    }
}

/// Argmax token id of a 1-D `[vocab]` logits row.
fn argmax_u32(row: &Tensor) -> u32 {
    row.argmax(D::Minus1).unwrap().to_scalar::<u32>().unwrap()
}

/// Arm every `LoRA` `B` factor (odd `trainable_vars` indices) to small random values
/// so the merge carries a real, representative adapter signal (not the zero-B no-op).
fn arm_adapter(policy: &QwenPolicy, device: &Device) {
    for (i, v) in policy.trainable_vars().iter().enumerate() {
        if i % 2 == 1 {
            let dims = v.as_tensor().dims().to_vec();
            v.set(&Tensor::randn(0f32, 0.05f32, dims, device).unwrap())
                .unwrap();
        }
    }
}

#[test]
#[ignore = "needs the real Qwen3-0.6B-Base checkpoint (FERRL_QWEN_WEIGHTS) + a CUDA build/GPU"]
#[allow(clippy::print_stderr)] // a manual gate: the printed agreement/diff numbers are the deliverable
fn merged_decoder_bf16_faithfulness_on_gpu() {
    // The P6-C **required manual bf16 GPU gate**: prove the cached `MergedDecoder`
    // rollout (the production path) is faithful to the uncached adapter-aware forward
    // in the real **bf16-base / F32-adapter** dtype split — the regime CPU CI cannot
    // reach (candle's CPU backend has no bf16 matmul). The CPU gates pin the F32 family
    // bit-for-position; this reports the bf16 argmax-agreement rate + max-abs logit diff.
    //
    // What differs, and why a small divergence is EXPECTED (not a bug): BOTH paths run
    // the adapter in bf16 — the forward casts A/B down to the base dtype before the
    // matmuls (the PR1 cast-order contract), so this is NOT "merged-bf16 vs an F32
    // adapter the uncached path keeps". The difference is *where* the small adapter
    // contribution lands. The cached path forms the merged weight `W + scale·BA` in bf16
    // (WEIGHT space), so per-element deltas below ~half-ulp(W) round into W and vanish
    // (the documented ~0.2 % absorption bound). The uncached path adds `scale·(x@Aᵀ)@Bᵀ`
    // in ACTIVATION space, where that contribution survives relative to the activation
    // magnitude. So the merged rollout can be very slightly *less* adapted — bounded and
    // expected; the grad/scoring path (always the uncached forward) is unaffected.
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = Device::new_cuda(0)
        .expect("CUDA device 0 — build with --features cuda and run on a GPU node");
    let buf = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    // Production dtype split: bf16 base, F32 adapter.
    let vb = VarBuilder::from_buffered_safetensors(buf, DType::BF16, &device)
        .expect("load model.safetensors (bf16) onto the GPU");
    let model = QwenGradModel::load_with_adapter_dtype(&cfg, &vb, RANK, ALPHA, DType::F32)
        .expect("build bf16-base/F32-adapter QwenGradModel");
    let mut policy = QwenPolicy::new(model, 1234, 1.0);
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
        eval_sampling: None,
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

    // Cached: prefill + single-token incremental decode, collecting every position.
    // Also track the logit *scale* (max-abs logit) so the diff can be judged relatively
    // — argmax agreement alone can stay high while logit magnitudes blow up.
    let mut dec = policy.model().merged_decoder().expect("merged decoder");
    let mut agree = 0usize;
    let mut max_abs = 0f32;
    let mut max_logit = 0f32;
    let abs_max = |t: &Tensor| -> f32 {
        t.to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
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
        "[bf16 GPU gate] cached-vs-uncached over {len} positions: argmax agreement \
         {agree}/{len} = {rate:.4}; max-abs logit diff {max_abs:.4} (logit scale \
         {max_logit:.2}, rel {rel:.4}). Interpret vs the ~0.2% bf16 merge-absorption bound."
    );
    assert!(
        max_abs.is_finite(),
        "bf16 cached logits diverged non-finitely from uncached"
    );
    // A magnitude backstop the argmax rate can't see: the merge-vs-uncached divergence
    // must stay a small fraction of the logit scale. A correct bf16 merge sits at a few
    // percent; a wrong scale / dropped or corrupted delta lands comparable to the logits
    // themselves. 50% cleaves between, with wide margin against honest bf16 noise.
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
#[ignore = "needs the real Qwen3-0.6B-Base checkpoint (FERRL_QWEN_WEIGHTS) + a CUDA build/GPU"]
#[allow(clippy::print_stderr)] // a manual gate: the printed timings are the deliverable
fn cached_rollout_perf_witness_on_gpu() {
    // The throughput witness the whole of P6-C exists for: the cached decoder's
    // incremental forward cost is O(L) per token vs the uncached forward's O(L²)
    // (it re-runs the full prefix every step). Times the FORWARD cost of both over a
    // real generation length on the real model; a scalar read at the end of each phase
    // forces the CUDA stream to drain so the wall-clock is honest. The asymptotic gap
    // is large at this length, so the direction (cached << uncached) is non-flaky.
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = Device::new_cuda(0)
        .expect("CUDA device 0 — build with --features cuda and run on a GPU node");
    let buf = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let vb = VarBuilder::from_buffered_safetensors(buf, DType::BF16, &device)
        .expect("load model.safetensors (bf16) onto the GPU");
    let model = QwenGradModel::load_with_adapter_dtype(&cfg, &vb, RANK, ALPHA, DType::F32)
        .expect("build QwenGradModel");
    let policy = QwenPolicy::new(model, 1234, 1.0);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let prompt_ids = tok.encode("The capital of France is");
    let n = 48usize;

    // Forces the CUDA stream to complete the work queued so far.
    let sync = |t: &Tensor| {
        let _ = t
            .to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
    };

    // Pre-grow a sequence with varying in-vocab tokens so each timed uncached forward
    // just slices a longer prefix (no per-iteration push to confound the timing).
    let mut grown = prompt_ids.clone();
    while grown.len() < prompt_ids.len() + n {
        grown.push((grown.len() % 256) as u32);
    }

    // Uncached: re-run the full-sequence forward over a growing prefix, n times.
    let t0 = Instant::now();
    let mut last = None;
    for k in 0..n {
        let plen = prompt_ids.len() + k;
        let inp = Tensor::from_vec(grown[..plen].to_vec(), (1, plen), &device).unwrap();
        last = Some(policy.model().forward(&inp).unwrap());
    }
    sync(last.as_ref().unwrap());
    let uncached_t = t0.elapsed();

    // Cached: one prefill + n single-token incremental steps (offset == cache length).
    let t1 = Instant::now();
    let mut dec = policy.model().merged_decoder().unwrap();
    let pin = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &device).unwrap();
    let mut last = dec.forward(&pin, 0).unwrap();
    for off in prompt_ids.len()..prompt_ids.len() + n {
        let tokt = Tensor::from_vec(vec![0u32], (1, 1), &device).unwrap();
        last = dec.forward(&tokt, off).unwrap();
    }
    sync(&last);
    let cached_t = t1.elapsed();

    let speedup = uncached_t.as_secs_f64() / cached_t.as_secs_f64();
    eprintln!(
        "[perf witness] {n} steps (prompt {}): uncached forward {uncached_t:?} vs cached \
         {cached_t:?} => {speedup:.1}x. (Uncached is O(L²), cached O(L).)",
        prompt_ids.len()
    );
    assert!(
        cached_t < uncached_t,
        "cached decode ({cached_t:?}) was not faster than uncached ({uncached_t:?})"
    );
}

/// Assert two `[seq][tok]` log-prob grids agree within `tol`.
fn assert_logprobs_close(a: &[Vec<f32>], b: &[Vec<f32>], tol: f32) {
    let pairs = a.iter().flatten().zip(b.iter().flatten());
    for (x, y) in pairs {
        assert!(
            (x - y).abs() <= tol,
            "GPU checkpoint round-trip diverged: {x} vs {y}"
        );
    }
}

#[test]
#[ignore = "needs the real Qwen3-0.6B-Base checkpoint (FERRL_QWEN_WEIGHTS) + a CUDA build/GPU"]
fn qwen_checkpoint_roundtrip_and_eval_on_gpu() {
    // P4-PR2 on CUDA: the adapter save/load device transfer (GPU -> CPU on save,
    // CPU -> GPU on load) and the eval harness, which the CPU tests cannot cover.
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let device = Device::new_cuda(0)
        .expect("CUDA device 0 — build with --features cuda and run on a GPU node");
    let buf = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let vb = VarBuilder::from_buffered_safetensors(buf, DType::F32, &device)
        .expect("load model.safetensors onto the GPU");
    // Two models over the same GPU base weights, independent adapters.
    let model_a = QwenGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build model A");
    let model_b = QwenGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build model B");
    let mut src = QwenPolicy::new(model_a, 1234, 1.0);
    let dst = QwenPolicy::new(model_b, 1234, 1.0);

    // Force src's adapter to a non-zero state on the GPU.
    for v in &src.trainable_vars() {
        let dims = v.as_tensor().dims().to_vec();
        v.set(&Tensor::randn(0f32, 0.1f32, dims, &device).unwrap())
            .unwrap();
    }
    let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![3, 1, 4, 1, 5]], 2);
    let logp_src = src
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    // Save from GPU (-> CPU safetensors), load back onto the GPU into dst.
    let tmp = TempDir::new();
    save_adapter(&tmp.0, &src.trainable_vars(), 0, None).expect("save adapter from GPU");
    let manifest = load_adapter(&tmp.0, &dst.trainable_vars()).expect("load adapter onto GPU");
    assert_eq!(manifest.num_vars, src.trainable_vars().len());
    let logp_dst = dst
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    assert_logprobs_close(&logp_src, &logp_dst, 1e-5);

    // The eval harness on GPU: base vs adapter, finite means, flag restored.
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let prompts = vec!["The capital of France is".to_string()];
    let gen = GenConfig {
        group_size: 4,
        max_new_tokens: 6,
        temperature: 1.0,
        eos_token_id: None,
        eval_sampling: None,
    };
    let report = evaluate(&mut src, &SpreadReward, &tok, &prompts, &gen).expect("eval on GPU");
    assert_eq!(report.n_prompts, 1);
    assert!(report.base_reward_mean.is_finite() && report.adapter_reward_mean.is_finite());
    assert!(src.adapter_enabled());
}

#[test]
#[ignore = "needs a CUDA build + a GPU (no weights required)"]
fn cuda_preflight_agrees_on_a_supported_gpu() {
    // The `cuda_compat` preflight on a node whose driver DOES support this build's PTX
    // (it must, or the other GPU tests here could not run). `guard_first_kernel` is the
    // AUTHORITY — it actually JITs a kernel — and must pass. `check_driver_compat` is a
    // warn-only heuristic; on an untabulated driver it may *conservatively* say `TooOld`
    // even though the guard passed, which is acceptable (it never blocks). So the real
    // `222` translation is exercised by the deliberately-mismatched build documented in
    // the PR (an old-driver node), which CI/this suite cannot stage.
    let device = Device::new_cuda(0)
        .expect("CUDA device 0 — build with --features cuda and run on a GPU node");
    ferrl::guard_first_kernel(&device).expect("guard_first_kernel must pass on a supported GPU");
    match ferrl::check_driver_compat(&device) {
        ferrl::CompatReport::Ok {
            built_isa,
            driver_cuda,
            ..
        } => {
            // Both were read for real (built ISA from the embedded PTX, driver from
            // the runtime query), so they are plausible version numbers.
            assert!(built_isa.0 >= 7, "implausible built PTX ISA {built_isa:?}");
            assert!(
                driver_cuda.0 >= 11,
                "implausible driver CUDA {driver_cuda:?}"
            );
        }
        // Query unavailable (Unknown) or a conservative TooOld are both fine: the guard,
        // not this heuristic, is the authority, and it passed above.
        ferrl::CompatReport::Unknown(_) | ferrl::CompatReport::TooOld { .. } => {}
    }
}
