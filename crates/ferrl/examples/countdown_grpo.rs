//! P4-PR3 — the real Countdown GRPO run (the P4 gate).
//!
//! Drives [`ferrl::QwenPolicy`] (real `Qwen3-0.6B-Base`, all-F32 on CUDA) through a
//! GRPO run over the verifiable [`ferrl::countdown`] reward, then checks the P4
//! gate: **the training reward rises AND the trained adapter beats base on a
//! held-out Countdown eval** (via [`ferrl::evaluate`]).
//!
//! This is a *run harness*, not a CI test: it needs the staged checkpoint and a
//! GPU, so — like the `#[ignore]`d GPU tests — it lives outside the coverage-gated
//! library (`cargo llvm-cov` skips `examples/`). The CI-tested task logic it drives
//! lives in `src/countdown.rs`.
//!
//! # Running it
//!
//! Build on the login node, run on a GPU node (see the scope card for the CUDA
//! recipe). It logs via `tracing` (no stdout prints), so pass `--nocapture`-style
//! visibility by running the binary directly:
//!
//! ```text
//! cargo build --release --features cuda --example countdown_grpo
//! FERRL_QWEN_WEIGHTS=/path/to/qwen3-0.6b-base \
//!     srun --partition=home --gres=gpu:A100_40G:1 \
//!     /tmp/ferrl-target/release/examples/countdown_grpo
//! ```
//!
//! Every knob has an `FERRL_CD_*` env override (steps, group size, max new tokens,
//! lr, temperature, dataset sizes, `LoRA` rank/alpha, …) so the run can be sized to
//! the GPU and tuned without a rebuild. Exits non-zero if the gate is not met.

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use ferrl::countdown::{
    build_prompt, generate_dataset, CountdownConfig, CountdownProblem, CountdownReward,
};
use ferrl::policy::GenConfig;
use ferrl::{
    evaluate, HfTokenizer, Metrics, QwenGradModel, QwenPolicy, RunDir, Trainer, TrainerConfig,
};
use tracing::info;

/// Read `key` from the environment, parsing it as `T`, falling back to `default`.
fn env_parse<T: FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Load the Qwen3 config from `dir/config.json`.
fn load_config(dir: &Path) -> Result<Config> {
    let bytes = std::fs::read(dir.join("config.json")).context("read config.json")?;
    serde_json::from_slice(&bytes).context("parse config.json into qwen3::Config")
}

/// Mean of an iterator of `f32` (0.0 when empty).
fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut sum = 0.0;
    let mut n = 0u32;
    for v in values {
        sum += v;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

/// Mean training reward over the first vs last quarter of the run — the trend the
/// gate reads (a single noisy step is not evidence either way).
fn reward_trend(history: &[Metrics]) -> (f32, f32) {
    let window = (history.len() / 4).max(1);
    let first = mean(history.iter().take(window).map(|m| m.reward_mean));
    let last = mean(history.iter().rev().take(window).map(|m| m.reward_mean));
    (first, last)
}

/// Build the policy over the real `Qwen3-0.6B-Base` checkpoint on CUDA (all-F32).
fn build_policy(dir: &Path, device: &Device) -> Result<(QwenPolicy, HfTokenizer)> {
    let cfg = load_config(dir)?;
    let buf = std::fs::read(dir.join("model.safetensors")).context("read model.safetensors")?;
    // bf16-base / F32-adapter split: load the frozen base in BF16 (halving the base
    // weights AND the retained activations that dominate the GRPO grad forward, so a
    // useful group size fits a 40GB GPU) while the trainable adapter stays F32.
    let vb = VarBuilder::from_buffered_safetensors(buf, DType::BF16, device)
        .context("load model.safetensors onto the GPU")?;
    let rank = env_parse("FERRL_CD_RANK", 16usize);
    let alpha = env_parse("FERRL_CD_ALPHA", 32.0f64);
    let model = QwenGradModel::load_with_adapter_dtype(&cfg, &vb, rank, alpha, DType::F32)
        .context("build QwenGradModel")?;
    let seed = env_parse("FERRL_CD_SEED", 1234u64);
    let temperature = env_parse("FERRL_CD_TEMP", 1.0f64);
    let policy = QwenPolicy::new(model, seed, temperature);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).context("load tokenizer")?;
    Ok((policy, tok))
}

/// The Countdown train / held-out splits, as ready-to-use prompts. The held-out
/// set is generated from a different stream **and** filtered so no problem also
/// appears in train — a genuine generalization gap, not memorization.
fn build_splits() -> (Vec<String>, Vec<String>) {
    let cd_cfg = CountdownConfig {
        num_count: env_parse("FERRL_CD_NUMCOUNT", 3usize),
        min_number: env_parse("FERRL_CD_MINNUM", 1u32),
        max_number: env_parse("FERRL_CD_MAXNUM", 20u32),
        max_target: env_parse("FERRL_CD_MAXTARGET", 1000u32),
    };
    let train_n = env_parse("FERRL_CD_TRAIN_N", 64usize);
    let eval_n = env_parse("FERRL_CD_EVAL_N", 32usize);
    let data_seed = env_parse("FERRL_CD_DATA_SEED", 7u64);

    let train = generate_dataset(data_seed, train_n, &cd_cfg);
    let train_set: HashSet<CountdownProblem> = train.iter().cloned().collect();
    // Draw a held-out pool from a different stream, dropping any train collision or
    // duplicate, then take up to `eval_n`.
    let pool = generate_dataset(
        data_seed.wrapping_add(1),
        eval_n.saturating_mul(3).max(1),
        &cd_cfg,
    );
    let mut eval: Vec<CountdownProblem> = Vec::new();
    for p in pool {
        if eval.len() >= eval_n {
            break;
        }
        if !train_set.contains(&p) && !eval.contains(&p) {
            eval.push(p);
        }
    }

    let train_prompts = train.iter().map(build_prompt).collect();
    let eval_prompts = eval.iter().map(build_prompt).collect();
    (train_prompts, eval_prompts)
}

/// The trainer config, every field env-overridable.
fn build_trainer_config() -> TrainerConfig {
    TrainerConfig {
        steps: env_parse("FERRL_CD_STEPS", 200u64),
        group_size: env_parse("FERRL_CD_GROUP", 8usize),
        max_new_tokens: env_parse("FERRL_CD_MAXNEW", 48usize),
        temperature: env_parse("FERRL_CD_TEMP", 1.0f64),
        lr: env_parse("FERRL_CD_LR", 1e-5f64),
        beta: env_parse("FERRL_CD_BETA", 0.0f64),
        checkpoint_every: Some(env_parse("FERRL_CD_CKPT", 50u64)),
        ..TrainerConfig::default()
    }
}

fn main() -> Result<()> {
    let _ = ferrl::init_tracing();

    let weights = env::var("FERRL_QWEN_WEIGHTS")
        .map_err(|_| anyhow!("set FERRL_QWEN_WEIGHTS to the Qwen3-0.6B-Base asset directory"))?;
    let dir = PathBuf::from(weights);
    let device = Device::new_cuda(0)
        .context("CUDA device 0 — build with --features cuda and run on a GPU node")?;

    let (mut policy, tok) = build_policy(&dir, &device)?;
    let (train_prompts, eval_prompts) = build_splits();
    let reward = CountdownReward::default();
    let tcfg = build_trainer_config();
    let gen = GenConfig {
        group_size: tcfg.group_size,
        max_new_tokens: tcfg.max_new_tokens,
        temperature: tcfg.temperature,
        eos_token_id: tcfg.eos_token_id,
    };
    info!(
        steps = tcfg.steps,
        group_size = tcfg.group_size,
        max_new_tokens = tcfg.max_new_tokens,
        lr = tcfg.lr,
        train = train_prompts.len(),
        eval = eval_prompts.len(),
        "countdown GRPO run starting"
    );

    let out = env_parse("FERRL_CD_OUT", "/tmp/ferrl-runs".to_string());
    let run = RunDir::create(Path::new(&out), "countdown-grpo").context("create run dir")?;
    let mut trainer = Trainer::new(tcfg, &run)?;
    let history = trainer.train(&mut policy, &reward, &tok, &train_prompts)?;

    // Held-out eval AFTER training: `evaluate` scores base (adapter off) vs the
    // trained adapter (adapter on) in one pass — the P4 comparison. There is no
    // pre-train eval: the adapter starts as a no-op (`B = 0`), so base == adapter,
    // and that extra sampling only fragments GPU memory ahead of the first grad step.
    let post = evaluate(&mut policy, &reward, &tok, &eval_prompts, &gen)?;
    let (first, last) = reward_trend(&history);
    let improvement = post.improvement();

    // Both conditions need a MARGIN: the reward means are Monte-Carlo (sampled),
    // so a bare `> 0` would pass on noise. Require a clear, ~tier-sized gap.
    let margin = env_parse("FERRL_CD_MARGIN", 0.05f32);
    let reward_rises = (last - first) > margin;
    let beats_base = improvement > margin;
    let gate_met = reward_rises && beats_base;

    info!(
        first_window = first,
        last_window = last,
        margin,
        reward_rises,
        "train reward trend (mean reward_mean: first quarter vs last quarter)"
    );
    info!(
        base = post.base_reward_mean,
        adapter = post.adapter_reward_mean,
        improvement,
        margin,
        beats_base,
        "post-train held-out eval"
    );
    info!(
        gate_met,
        "P4 gate: training reward rises AND the adapter beats base on held-out Countdown"
    );

    if gate_met {
        Ok(())
    } else {
        Err(anyhow!(
            "P4 gate NOT met: reward_rises={reward_rises} (first={first}, last={last}), \
             beats_base={beats_base} (improvement={improvement}, margin={margin})"
        ))
    }
}
