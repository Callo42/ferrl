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

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use ferrl::policy::{GenConfig, Rollout};
use ferrl::{
    evaluate, load_adapter, save_adapter, HfTokenizer, Policy, QwenGradModel, QwenPolicy, RewardFn,
    RunDir, Trainer, TrainerConfig,
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
    let resumed = trainer2
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
    save_adapter(&tmp.0, &src.trainable_vars(), 0).expect("save adapter from GPU");
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
