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

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use ferrl::{
    HfTokenizer, Policy, QwenGradModel, QwenPolicy, RewardFn, RunDir, Trainer, TrainerConfig,
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
