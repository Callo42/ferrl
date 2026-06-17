//! The real Countdown GRPO run — the P4 gate, now exercising P6-A EOS/length masking.
//!
//! Drives [`ferrl::QwenPolicy`] (real `Qwen3-0.6B-Base`, bf16 base / F32 `LoRA`
//! adapter on CUDA) through a GRPO run over the verifiable [`ferrl::countdown`]
//! reward, then checks the P4 gate: **the training reward rises AND the trained
//! adapter beats base on a held-out Countdown eval** (via [`ferrl::evaluate`]).
//!
//! This is a *run harness*, not a CI test: it needs the staged checkpoint and a
//! GPU, so — like the `#[ignore]`d GPU tests — it lives outside the coverage-gated
//! library (`cargo llvm-cov` skips `examples/`). The CI-tested task logic it drives
//! lives in `src/countdown.rs`.
//!
//! This harness deliberately pins the **pre-R1 recipe** its gate margins were
//! calibrated on (see `build_trainer_config`). The goal-gate **ladder** runs the
//! modern recipe instead — see the sibling `countdown_grpo_qwen35` harness.
//!
//! # EOS / length masking (P6-A)
//!
//! The run is EOS-aware: each sampled completion stops at the model's
//! end-of-sequence token, and only the real (EOS-inclusive) tokens feed the loss,
//! the reward, and the eval — the padding tail is masked out. The EOS id is read
//! from the checkpoint's `config.json` (candle's `qwen3::Config` does not carry it),
//! so it tracks the model rather than a baked-in literal. `FERRL_CD_EOS` overrides:
//! an integer forces that id, and `none` (or `off`) restores the legacy full-width
//! rollout for an A/B comparison. For `Qwen3-0.6B-Base` the token is `<|endoftext|>`
//! (id `151643`); the base model emits it to end a document, so a completion that
//! finishes early is no longer trained on the repeated-garbage tail. The run logs
//! the mean completion length, so a value below `max_new_tokens` witnesses EOS
//! firing.
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
//! lr, temperature, dataset sizes, `LoRA` rank/alpha, EOS id, …) so the run can be
//! sized to the GPU and tuned without a rebuild. Exits non-zero if the gate is not
//! met.

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
    evaluate, HfTokenizer, Metrics, QwenGradModel, QwenPolicy, RunDir, Sample, Trainer,
    TrainerConfig,
};
use tracing::{info, warn};

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

/// Resolve the EOS token id for the run.
///
/// `FERRL_CD_EOS` overrides the checkpoint default: unset reads the model's
/// `eos_token_id` from `config.json` via [`ferrl::eos_from_config`] (the actual EOS
/// run; the helper accepts either a top-level `eos_token_id` — the Qwen3 base shape
/// here — or a `text_config`-nested one); `none`/`off` (any case) yields `None`,
/// recovering the legacy full-width rollout for an A/B comparison; any other value
/// is parsed as an explicit id.
fn resolve_eos(dir: &Path) -> Result<Option<u32>> {
    match env::var("FERRL_CD_EOS") {
        Err(_) => Ok(ferrl::eos_from_config(dir)?),
        Ok(raw) => {
            let v = raw.trim();
            if v.eq_ignore_ascii_case("none") || v.eq_ignore_ascii_case("off") {
                Ok(None)
            } else {
                let id = v.parse::<u32>().with_context(|| {
                    format!("FERRL_CD_EOS must be an integer id or 'none', got {raw:?}")
                })?;
                Ok(Some(id))
            }
        }
    }
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

/// Build the policy over the real `Qwen3-0.6B-Base` checkpoint on CUDA (bf16 base
/// / F32 `LoRA` adapter).
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
    // Deliberately the LEGACY q/v-only LoRA recipe (`load_with_adapter_dtype`
    // delegates to `DenseLoraTargets::legacy()`): the P4 gate margins below were
    // calibrated against it. Switching to `load_with_targets(industrial)` is a
    // re-calibration, not a drop-in swap.
    let model = QwenGradModel::load_with_adapter_dtype(&cfg, &vb, rank, alpha, DType::F32)
        .context("build QwenGradModel")?;
    let seed = env_parse("FERRL_CD_SEED", 1234u64);
    let temperature = env_parse("FERRL_CD_TEMP", 1.0f64);
    let policy = QwenPolicy::new(model, seed, temperature);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).context("load tokenizer")?;
    Ok((policy, tok))
}

/// The Countdown train / held-out splits, as ready-to-use samples (each prompt
/// paired with its typed `CountdownProblem` target). The held-out set is generated
/// from a different stream **and** filtered so no problem also appears in train —
/// a genuine generalization gap, not memorization.
fn build_splits() -> (Vec<Sample<CountdownProblem>>, Vec<Sample<CountdownProblem>>) {
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

    // Each sample carries the prompt the model sees AND the typed problem the reward
    // scores against — no smuggling the answer through the prompt string.
    let train_samples = train
        .into_iter()
        .map(|p| Sample::new(build_prompt(&p), p))
        .collect();
    let eval_samples = eval
        .into_iter()
        .map(|p| Sample::new(build_prompt(&p), p))
        .collect();
    (train_samples, eval_samples)
}

/// The trainer config, every field env-overridable. `eos_token_id` is resolved by
/// the caller (see [`resolve_eos`]) and flows into both training and the held-out
/// eval so generation is EOS-aware on both paths.
fn build_trainer_config(eos_token_id: Option<u32>) -> TrainerConfig {
    TrainerConfig {
        steps: env_parse("FERRL_CD_STEPS", 200u64),
        group_size: env_parse("FERRL_CD_GROUP", 8usize),
        max_new_tokens: env_parse("FERRL_CD_MAXNEW", 48usize),
        temperature: env_parse("FERRL_CD_TEMP", 1.0f64),
        lr: env_parse("FERRL_CD_LR", 1e-5f64),
        beta: env_parse("FERRL_CD_BETA", 0.0f64),
        checkpoint_every: Some(env_parse("FERRL_CD_CKPT", 50u64)),
        eos_token_id,
        // Calibration pin: this gate's margins (FERRL_CD_MARGIN, reward-trend,
        // beats-base) were established on the pre-R1 recipe — classic Grpo
        // reduction, no clipping, no truncation masking. Keep that trajectory
        // until the margins are deliberately recalibrated on the modern
        // recipe (the 0.8B PoC ladder runs the R1 defaults instead). R2 note:
        // at the FERRL_CD_TEMP=1.0 default, scoring is bit-identical to the
        // calibrated runs; a non-1.0 temperature now also rescales scoring
        // (temperature-consistent scoring) — a deliberate recipe change, so
        // re-calibrate before leaning on the margins at another temperature.
        loss_type: ferrl::LossType::Grpo,
        max_grad_norm: None,
        truncation_masking: false,
        ..TrainerConfig::default()
    }
}

/// Open CUDA device 0 and run the driver-compatibility preflight: warn early on a
/// likely PTX/driver mismatch (proactive, warn-only), then force the first kernel JIT
/// so a real mismatch fails *here* with an actionable rebuild/upgrade message rather
/// than buried in the first training forward. Both checks need only the device, so
/// this runs before the multi-second weight load.
fn open_cuda_device() -> Result<Device> {
    let device = Device::new_cuda(0)
        .context("CUDA device 0 — build with --features cuda and run on a GPU node")?;
    if let Some(w) = ferrl::check_driver_compat(&device).warning() {
        warn!("{w}");
    }
    ferrl::guard_first_kernel(&device).context("CUDA preflight")?;
    Ok(device)
}

fn main() -> Result<()> {
    let _ = ferrl::init_tracing();

    let weights = env::var("FERRL_QWEN_WEIGHTS")
        .map_err(|_| anyhow!("set FERRL_QWEN_WEIGHTS to the Qwen3-0.6B-Base asset directory"))?;
    let dir = PathBuf::from(weights);
    let device = open_cuda_device()?;
    let (mut policy, tok) = build_policy(&dir, &device)?;
    let (train_samples, eval_samples) = build_splits();
    let reward = CountdownReward::default();
    // Read the model's EOS from config.json (env-overridable); flows into both the
    // trainer and the eval `gen` below so generation is EOS-aware on both paths.
    let eos = resolve_eos(&dir)?;
    let tcfg = build_trainer_config(eos);
    // The eval generation config is the trainer's rollout config (single source of
    // truth via `GenConfig::from(&TrainerConfig)`), so the two cannot drift.
    let gen = GenConfig::from(&tcfg);
    info!(
        steps = tcfg.steps,
        group_size = tcfg.group_size,
        max_new_tokens = tcfg.max_new_tokens,
        lr = tcfg.lr,
        eos_token_id = ?tcfg.eos_token_id,
        train = train_samples.len(),
        eval = eval_samples.len(),
        "countdown GRPO run starting"
    );

    let out = env_parse("FERRL_CD_OUT", "/tmp/ferrl-runs".to_string());
    // Unique run id per invocation: RunDir::create now fails loud on a
    // duplicate run_id (appending to a prior run's metrics stream), and this
    // example is routinely re-run after a missed gate.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let run_id = format!("countdown-grpo-{stamp}");
    let run = RunDir::create(Path::new(&out), &run_id).context("create run dir")?;
    let mut trainer = Trainer::new(tcfg, &run)?;
    // No preemption flag installed → this run always completes; ignore the stop.
    let (history, _stop) = trainer.train(&mut policy, &reward, &tok, &train_samples)?;

    // Held-out eval AFTER training: `evaluate` scores base (adapter off) vs the
    // trained adapter (adapter on) in one pass — the P4 comparison. There is no
    // pre-train eval: the adapter starts as a no-op (`B = 0`), so base == adapter,
    // and that extra sampling only fragments GPU memory ahead of the first grad step.
    let post = evaluate(&mut policy, &reward, &tok, &eval_samples, &gen)?;
    let (first, last) = reward_trend(&history);
    let improvement = post.improvement();
    // EOS witness: the length-aware mean completion length over the run. Below
    // `max_new_tokens` ⇒ EOS fired and the masked tail was kept out of the loss;
    // equal to it ⇒ the model never emitted EOS within the window (or it is `None`).
    let mean_completion_len = mean(history.iter().map(|m| m.completion_len));

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
        mean_completion_len,
        max_new_tokens = gen.max_new_tokens,
        eos_token_id = ?gen.eos_token_id,
        "EOS/length: mean completion length (< max_new_tokens ⇒ EOS fired)"
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
