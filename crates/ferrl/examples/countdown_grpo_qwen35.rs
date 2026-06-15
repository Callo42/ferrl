//! The Countdown GRPO **goal-gate ladder** harness for the qwen3.5/3.6 family —
//! rung 1: the 0.8B `PoC` on the modern (R-track) recipe.
//!
//! Drives [`ferrl::Qwen3_5Policy`] (real `Qwen3.5-0.8B-Base`, bf16 base / F32
//! `LoRA` adapter on CUDA) through a GRPO run over the verifiable
//! [`ferrl::countdown`] reward, then gates like the P4 harness: **the training
//! reward rises AND the trained adapter beats base on a held-out Countdown
//! eval** (via [`ferrl::evaluate`]).
//!
//! Rung 1 (0.8B `PoC`) → rung 2 (9B dry-run) → rung 3 (27B, the goal gate) are
//! this same harness pointed at bigger checkpoints — every knob is an env
//! override (`FERRL_CD35_*`), so a rung is a config, not a rebuild. Like
//! `countdown_grpo`, this is a *run harness*, not a CI test: it needs a staged
//! checkpoint and a GPU (`cargo llvm-cov` skips `examples/`).
//!
//! # The modern recipe (how this differs from `countdown_grpo`)
//!
//! The legacy harness deliberately pins the pre-R1 recipe its gate margins were
//! calibrated on. This harness runs the **R-track library defaults** — token-level
//! DAPO loss, global-norm grad clipping at `1.0`, truncation masking ON,
//! symmetric `0.2` clip, no TIS — plus the two ladder knobs `PLAN.md` names
//! explicitly:
//!
//! - **`warmup_steps`** — default **20** here (the library default is `0` so the
//!   toy/CI config stays deterministic; ladder run configs set it explicitly).
//! - **the eval convention** — the held-out eval samples from the eval-only
//!   distribution ([`ferrl::EvalSampling`]: temperature `0.6`, nucleus top-p
//!   `0.95`) instead of the training rollout distribution, scoring avg@k with
//!   `k = FERRL_CD35_EVAL_K` completions per prompt. `FERRL_CD35_EVAL_SAMPLING=off`
//!   recovers the trainer-distribution eval for an A/B comparison.
//!
//! **Gate margins are NOT yet calibrated on this recipe.** The first rung-1 runs
//! establish them (tune `FERRL_CD35_MARGIN`); the legacy harness's margins were
//! calibrated on a different recipe and model and do not transfer.
//!
//! # Rung-1 decision knobs
//!
//! - **`FERRL_CD35_TARGETS`** — `industrial` (default; `attn:qkvo|mlp:gud|gdn:-`)
//!   or `all-linear` (adds the `GatedDeltaNet` projections). A/B-ing these on
//!   rung 1 renders the deferred **GDN-LoRA verdict**.
//! - **`FERRL_CD35_REMAT`** — `on` turns on activation checkpointing
//!   (default off, matching the library default). A/B-ing it on one config is
//!   the deferred **real peak-memory measurement**; recompute is deterministic,
//!   so the trajectory should match the uncheckpointed run.
//!
//! # Fail-loud knobs (a typo must not burn a GPU run)
//!
//! Every `FERRL_CD35_*` knob aborts on a present-but-unparsable value instead
//! of silently running its default; eval-distribution knobs are validated
//! before training, not at the post-train eval. The EOS id is read from the
//! checkpoint's `config.json` (`text_config.eos_token_id`, falling back to the
//! top level); without one the run **refuses to start** — truncation masking
//! would be silently inert — unless `FERRL_CD35_EOS=none` deliberately opts
//! into full-width rollouts (an integer `FERRL_CD35_EOS` forces an explicit id).
//!
//! # Running it
//!
//! Build on the login node, run on a GPU node (see the scope card for the CUDA
//! recipe). It logs via `tracing` (no stdout prints):
//!
//! ```text
//! cargo build --release --features cuda --example countdown_grpo_qwen35
//! FERRL_QWEN35_WEIGHTS=/path/to/qwen3.5-0.8b-base \
//!     srun --partition=home --gres=gpu:A100_40G:1 \
//!     /tmp/ferrl-target/release/examples/countdown_grpo_qwen35
//! ```
//!
//! `FERRL_QWEN35_WEIGHTS` points at the checkpoint directory (`config.json`,
//! safetensors shard(s) + index, `tokenizer.json`) — the same variable the
//! `qwen3_5` real-weights and GPU-smoke tests read. Exits non-zero if the gate is
//! not met.

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use ferrl::countdown::{
    build_prompt, generate_dataset, CountdownConfig, CountdownProblem, CountdownReward,
};
use ferrl::policy::GenConfig;
use ferrl::{
    evaluate, EvalReport, EvalSampling, HfTokenizer, LoraTargets, Metrics, Qwen3_5Config,
    Qwen3_5GradModel, Qwen3_5Policy, RunDir, RunStop, Trainer, TrainerConfig,
};
use tracing::{info, warn};

/// Read `key` from the environment, parsing it as `T`. Unset falls back to
/// `default`; a present-but-unparsable value fails LOUD — a typo'd knob must
/// not silently run the default config on a GPU node.
fn env_parse<T: FromStr>(key: &str, default: T) -> Result<T> {
    match env::var(key) {
        Err(_) => Ok(default),
        Ok(raw) => raw
            .trim()
            .parse()
            .map_err(|_| anyhow!("{key} is set but unparsable: {raw:?}")),
    }
}

/// Read a boolean switch from `key`: `on`/`1`/`true`/`yes` or
/// `off`/`none`/`0`/`false`/`no` (any case). Unset falls back to `default`;
/// any other spelling fails loud rather than silently picking a side.
fn env_switch(key: &str, default: bool) -> Result<bool> {
    match env::var(key) {
        Err(_) => Ok(default),
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "on" | "1" | "true" | "yes" => Ok(true),
            "off" | "none" | "0" | "false" | "no" => Ok(false),
            _ => bail!("{key} must be on/off (or 1/0, true/false), got {raw:?}"),
        },
    }
}

/// The run-identity knobs, read once and logged once from `main` — the A/B
/// provenance for rung-1 verdicts (rank/alpha/seed do not reach the run dir's
/// `config.json`, so the start log is their durable record).
struct RunKnobs {
    /// `LoRA` rank (`FERRL_CD35_RANK`).
    rank: usize,
    /// `LoRA` alpha (`FERRL_CD35_ALPHA`).
    alpha: f64,
    /// Sampler seed (`FERRL_CD35_SEED`).
    seed: u64,
    /// Rollout/scoring temperature (`FERRL_CD35_TEMP`) — read in exactly one
    /// place so the policy and the trainer config cannot diverge.
    temperature: f64,
    /// Dataset stream seed (`FERRL_CD35_DATA_SEED`).
    data_seed: u64,
    /// Activation checkpointing (`FERRL_CD35_REMAT`).
    remat: bool,
}

/// Read the [`RunKnobs`] from the environment, fail-loud on typos.
fn read_knobs() -> Result<RunKnobs> {
    Ok(RunKnobs {
        rank: env_parse("FERRL_CD35_RANK", 16usize)?,
        alpha: env_parse("FERRL_CD35_ALPHA", 32.0f64)?,
        seed: env_parse("FERRL_CD35_SEED", 1234u64)?,
        temperature: env_parse("FERRL_CD35_TEMP", 1.0f64)?,
        data_seed: env_parse("FERRL_CD35_DATA_SEED", 7u64)?,
        remat: env_switch("FERRL_CD35_REMAT", false)?,
    })
}

/// The model's end-of-sequence token id, read from the checkpoint's
/// `config.json`. `qwen3_5` checkpoints nest it under the multimodal wrapper's
/// `text_config` (the real 0.8B file's shape); the top level is the fallback.
///
/// [`Qwen3_5Config`] does not carry it, so read the raw JSON. Returns `Ok(None)`
/// when the field is absent or is not a plain integer — a list-valued
/// `eos_token_id` (multi-EOS) is not handled, since [`GenConfig::eos_token_id`]
/// carries a single id.
fn config_eos(dir: &Path) -> Result<Option<u32>> {
    let bytes = std::fs::read(dir.join("config.json")).context("read config.json")?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).context("parse config.json")?;
    Ok(json
        .pointer("/text_config/eos_token_id")
        .or_else(|| json.get("eos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok()))
}

/// Resolve the EOS token id for the run.
///
/// `FERRL_CD35_EOS` overrides the checkpoint default: unset reads the model's
/// `eos_token_id` from `config.json`; `none`/`off` (any case) yields `None`,
/// recovering the legacy full-width rollout for an A/B comparison; any other
/// value is parsed as an explicit id.
fn resolve_eos(dir: &Path) -> Result<Option<u32>> {
    match env::var("FERRL_CD35_EOS") {
        Err(_) => config_eos(dir),
        Ok(raw) => {
            let v = raw.trim();
            if v.eq_ignore_ascii_case("none") || v.eq_ignore_ascii_case("off") {
                Ok(None)
            } else {
                let id = v.parse::<u32>().with_context(|| {
                    format!("FERRL_CD35_EOS must be an integer id or 'none', got {raw:?}")
                })?;
                Ok(Some(id))
            }
        }
    }
}

/// Resolve the `LoRA` recipe from `FERRL_CD35_TARGETS` (the rung-1 GDN-LoRA
/// A/B knob): `industrial` (default) or `all-linear`/`all_linear`.
fn resolve_targets() -> Result<LoraTargets> {
    match env::var("FERRL_CD35_TARGETS") {
        Err(_) => Ok(LoraTargets::industrial()),
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "industrial" => Ok(LoraTargets::industrial()),
            "all-linear" | "all_linear" => Ok(LoraTargets::all_linear()),
            other => {
                bail!("FERRL_CD35_TARGETS must be 'industrial' or 'all-linear', got {other:?}")
            }
        },
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

/// Build the policy over the real `Qwen3.5-0.8B-Base` checkpoint on CUDA (bf16
/// base / F32 `LoRA` adapter), honoring the recipe and checkpointing knobs.
fn build_policy(
    dir: &Path,
    device: &Device,
    knobs: &RunKnobs,
) -> Result<(Qwen3_5Policy, HfTokenizer)> {
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json"))
        .context("parse config.json into Qwen3_5Config")?;
    // bf16-base / F32-adapter split, as on the dense harness: the frozen base
    // (weights AND the retained activations dominating the GRPO grad forward)
    // halves, while the trainable adapter stays F32.
    let vb = ferrl::varbuilder_from_pretrained(dir, DType::BF16, device)
        .context("load checkpoint weights onto the GPU")?;
    let targets = resolve_targets()?;
    let mut model = Qwen3_5GradModel::load_with_targets(
        &cfg,
        &vb,
        knobs.rank,
        knobs.alpha,
        DType::F32,
        targets,
    )
    .context("build Qwen3_5GradModel")?;
    if knobs.remat {
        model.set_activation_checkpointing(true);
    }
    let policy = Qwen3_5Policy::new(model, knobs.seed, knobs.temperature);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).context("load tokenizer")?;
    Ok((policy, tok))
}

/// Draw the held-out problems from a different stream (`data_seed + 1`),
/// dropping train collisions and duplicates, up to `eval_n`.
fn held_out(
    train_set: &HashSet<CountdownProblem>,
    eval_n: usize,
    data_seed: u64,
    cd_cfg: &CountdownConfig,
) -> Vec<CountdownProblem> {
    let pool = generate_dataset(
        data_seed.wrapping_add(1),
        eval_n.saturating_mul(3).max(1),
        cd_cfg,
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
    eval
}

/// The Countdown train / held-out splits, as ready-to-use prompts. The held-out
/// set is generated from a different stream **and** filtered so no problem also
/// appears in train — a genuine generalization gap, not memorization. Fails
/// loud (before any GPU work) if the held-out set comes up empty.
fn build_splits(data_seed: u64) -> Result<(Vec<String>, Vec<String>)> {
    let cd_cfg = CountdownConfig {
        num_count: env_parse("FERRL_CD35_NUMCOUNT", 3usize)?,
        min_number: env_parse("FERRL_CD35_MINNUM", 1u32)?,
        max_number: env_parse("FERRL_CD35_MAXNUM", 20u32)?,
        max_target: env_parse("FERRL_CD35_MAXTARGET", 1000u32)?,
    };
    let train_n = env_parse("FERRL_CD35_TRAIN_N", 64usize)?;
    let eval_n = env_parse("FERRL_CD35_EVAL_N", 32usize)?;

    let train = generate_dataset(data_seed, train_n, &cd_cfg);
    let train_set: HashSet<CountdownProblem> = train.iter().cloned().collect();
    let eval = held_out(&train_set, eval_n, data_seed, &cd_cfg);
    if eval.is_empty() {
        bail!(
            "held-out eval set is empty (every drawn problem collided with \
             train) — widen the Countdown ranges or raise FERRL_CD35_EVAL_N"
        );
    }

    let train_prompts = train.iter().map(build_prompt).collect();
    let eval_prompts = eval.iter().map(build_prompt).collect();
    Ok((train_prompts, eval_prompts))
}

/// The trainer config — the **modern (R-track) recipe**, every field
/// env-overridable. Everything not set here is deliberately the library
/// default: token-level DAPO loss, global-norm clip `1.0`, truncation masking
/// ON, symmetric `0.2` clip, no TIS. `eos_token_id` and `temperature` are
/// resolved by the caller from a single read each, so training, generation,
/// and scoring cannot diverge.
fn build_trainer_config(eos_token_id: Option<u32>, temperature: f64) -> Result<TrainerConfig> {
    let steps = env_parse("FERRL_CD35_STEPS", 200u64)?;
    if steps == 0 {
        bail!("FERRL_CD35_STEPS must be >= 1");
    }
    Ok(TrainerConfig {
        steps,
        group_size: env_parse("FERRL_CD35_GROUP", 8usize)?,
        max_new_tokens: env_parse("FERRL_CD35_MAXNEW", 48usize)?,
        temperature,
        lr: env_parse("FERRL_CD35_LR", 1e-5f64)?,
        beta: env_parse("FERRL_CD35_BETA", 0.0f64)?,
        // The PLAN-named ladder knob: the library default is 0 (deterministic
        // toy/CI config); ladder run configs set warmup explicitly.
        warmup_steps: env_parse("FERRL_CD35_WARMUP", 20u64)?,
        checkpoint_every: Some(env_parse("FERRL_CD35_CKPT", 50u64)?),
        eos_token_id,
        ..TrainerConfig::default()
    })
}

/// The nucleus knob: `FERRL_CD35_EVAL_TOPP` — a probability in `(0, 1]`, or
/// `none`/`off` to disable nucleus filtering entirely (the one spelling the
/// plain parse cannot express).
fn eval_top_p() -> Result<Option<f64>> {
    match env::var("FERRL_CD35_EVAL_TOPP") {
        Err(_) => Ok(EvalSampling::default().top_p),
        Ok(raw) => {
            let v = raw.trim();
            if v.eq_ignore_ascii_case("none") || v.eq_ignore_ascii_case("off") {
                return Ok(None);
            }
            let p: f64 = v.parse().map_err(|_| {
                anyhow!("FERRL_CD35_EVAL_TOPP must be a probability or 'none', got {raw:?}")
            })?;
            if p <= 0.0 || p > 1.0 {
                bail!("FERRL_CD35_EVAL_TOPP must be in (0, 1], got {p}");
            }
            Ok(Some(p))
        }
    }
}

/// The eval-only sampling override (the R2 eval convention), pre-flight
/// validated HERE so a typo'd eval knob aborts before training rather than
/// after the last step. `FERRL_CD35_EVAL_SAMPLING=off` drops the override and
/// evaluates on the trainer distribution instead (the legacy comparison).
fn eval_sampling_override() -> Result<Option<EvalSampling>> {
    if !env_switch("FERRL_CD35_EVAL_SAMPLING", true)? {
        return Ok(None);
    }
    let temperature = env_parse("FERRL_CD35_EVAL_TEMP", EvalSampling::default().temperature)?;
    if !temperature.is_finite() || temperature <= 0.0 {
        bail!("FERRL_CD35_EVAL_TEMP must be finite and > 0, got {temperature}");
    }
    Ok(Some(EvalSampling {
        temperature,
        top_p: eval_top_p()?,
    }))
}

/// The held-out eval generation config — the R2 **eval convention**: avg@k with
/// `k = FERRL_CD35_EVAL_K` completions per prompt (default: the training group
/// size), sampled per [`eval_sampling_override`].
fn build_eval_gen(tcfg: &TrainerConfig) -> Result<GenConfig> {
    let k = env_parse("FERRL_CD35_EVAL_K", tcfg.group_size)?;
    if k == 0 {
        bail!("FERRL_CD35_EVAL_K must be >= 1 (completions per eval prompt)");
    }
    Ok(GenConfig {
        group_size: k,
        max_new_tokens: tcfg.max_new_tokens,
        temperature: tcfg.temperature,
        eos_token_id: tcfg.eos_token_id,
        eval_sampling: eval_sampling_override()?,
    })
}

/// Open CUDA device 0 and run the driver-compatibility preflight: warn early on a
/// likely PTX/driver mismatch (proactive, warn-only), then force the first kernel
/// JIT so a real mismatch fails *here* with an actionable message rather than
/// buried in the first training forward.
fn open_cuda_device() -> Result<Device> {
    let device = Device::new_cuda(0)
        .context("CUDA device 0 — build with --features cuda and run on a GPU node")?;
    if let Some(w) = ferrl::check_driver_compat(&device).warning() {
        warn!("{w}");
    }
    ferrl::guard_first_kernel(&device).context("CUDA preflight")?;
    Ok(device)
}

/// Resolve the EOS id (see [`resolve_eos`]) and REFUSE to run without one
/// unless that was an explicit choice: the modern recipe's truncation masking
/// is inert without an EOS id, and a ladder run must not silently degrade to
/// full-width rollouts. `FERRL_CD35_EOS=none` is the sanctioned full-width A/B.
fn resolve_eos_strict(dir: &Path) -> Result<Option<u32>> {
    let explicit = env::var("FERRL_CD35_EOS").is_ok();
    let eos = resolve_eos(dir)?;
    if eos.is_none() {
        if !explicit {
            bail!(
                "no integer eos_token_id in config.json (text_config or top level): \
                 set FERRL_CD35_EOS=<id>, or FERRL_CD35_EOS=none for a deliberate \
                 full-width A/B run"
            );
        }
        warn!("FERRL_CD35_EOS=none — full-width rollouts; truncation masking inert");
    }
    Ok(eos)
}

/// Report the run's reward trend, held-out eval, and EOS witness, then apply
/// the ladder gate: the training reward **rises** AND the trained adapter
/// **beats base** on held-out Countdown, each by `FERRL_CD35_MARGIN`. The
/// margins are NOT yet calibrated on the modern recipe — rung-1 runs establish
/// them; a bare `> 0` would pass on Monte-Carlo noise.
fn report_and_gate(history: &[Metrics], post: &EvalReport, gen: &GenConfig) -> Result<()> {
    let (first, last) = reward_trend(history);
    let improvement = post.improvement();
    // EOS witness: mean completion length below `max_new_tokens` ⇒ EOS fired
    // and the masked tail was kept out of the loss.
    let mean_completion_len = mean(history.iter().map(|m| m.completion_len));

    let margin = env_parse("FERRL_CD35_MARGIN", 0.05f32)?;
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
        "post-train held-out eval (avg@k, eval-only sampling unless off)"
    );
    info!(
        mean_completion_len,
        max_new_tokens = gen.max_new_tokens,
        eos_token_id = ?gen.eos_token_id,
        "EOS/length: mean completion length (< max_new_tokens ⇒ EOS fired)"
    );
    info!(
        gate_met,
        "ladder rung gate: training reward rises AND the adapter beats base on held-out Countdown"
    );

    if gate_met {
        Ok(())
    } else {
        Err(anyhow!(
            "ladder rung gate NOT met: reward_rises={reward_rises} (first={first}, last={last}), \
             beats_base={beats_base} (improvement={improvement}, margin={margin})"
        ))
    }
}

/// Install the cooperative preemption handler and open the run directory.
///
/// Registers `SIGTERM`/`SIGUSR1` (Slurm's preempt / timeout grace signals) to flip a
/// shared flag — the trainer polls it and checkpoints + stops at the next step
/// boundary (the library itself never touches signals). A stable
/// `FERRL_CD35_RUN_ID` makes a requeued job **continue** the same run directory
/// (resuming from its latest checkpoint via [`Trainer::resume_latest`]); unset gives
/// a fresh, unique run per invocation. Returns the run dir plus the flag to hand the
/// trainer.
fn open_run_with_preemption(out: &str) -> Result<(RunDir, Arc<AtomicBool>)> {
    let preempt = Arc::new(AtomicBool::new(false));
    for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGUSR1] {
        signal_hook::flag::register(sig, Arc::clone(&preempt))
            .context("install preemption signal handler")?;
    }
    let run_id = env::var("FERRL_CD35_RUN_ID").unwrap_or_else(|_| {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        format!("countdown-grpo35-{stamp}-{}", std::process::id())
    });
    // Reopen an existing run dir (a requeue), else create it fresh. Gate on the
    // directory existing rather than catching RunDir::open's error, so a genuine I/O
    // fault on an existing run surfaces loudly instead of masquerading as a
    // duplicate-run failure from the create fallback.
    let run = if Path::new(out).join(&run_id).is_dir() {
        info!(run_id, "resuming existing run directory");
        RunDir::open(Path::new(out), &run_id).context("open existing run dir")?
    } else {
        RunDir::create(Path::new(out), &run_id).context("create run dir")?
    };
    Ok((run, preempt))
}

fn main() -> Result<()> {
    let _ = ferrl::init_tracing();

    let weights = env::var("FERRL_QWEN35_WEIGHTS").map_err(|_| {
        anyhow!("set FERRL_QWEN35_WEIGHTS to the Qwen3.5-0.8B-Base asset directory")
    })?;
    let dir = PathBuf::from(weights);
    let knobs = read_knobs()?;
    let device = open_cuda_device()?;
    let (mut policy, tok) = build_policy(&dir, &device, &knobs)?;
    let (train_prompts, eval_prompts) = build_splits(knobs.data_seed)?;
    let reward = CountdownReward::default();
    // The EOS id flows into both the trainer and the eval `gen` below so
    // generation is EOS-aware on both paths.
    let eos = resolve_eos_strict(&dir)?;
    let tcfg = build_trainer_config(eos, knobs.temperature)?;
    let gen = build_eval_gen(&tcfg)?;
    info!(
        steps = tcfg.steps,
        group_size = tcfg.group_size,
        max_new_tokens = tcfg.max_new_tokens,
        temperature = tcfg.temperature,
        lr = tcfg.lr,
        warmup_steps = tcfg.warmup_steps,
        eos_token_id = ?tcfg.eos_token_id,
        targets = %resolve_targets()?.canonical(),
        rank = knobs.rank,
        alpha = knobs.alpha,
        seed = knobs.seed,
        data_seed = knobs.data_seed,
        remat = knobs.remat,
        eval_k = gen.group_size,
        eval_sampling = ?gen.eval_sampling,
        train = train_prompts.len(),
        eval = eval_prompts.len(),
        "countdown GRPO ladder run (qwen3_5, modern recipe) starting"
    );

    let out = env_parse("FERRL_CD35_OUT", "/tmp/ferrl-runs".to_string())?;
    let (run, preempt) = open_run_with_preemption(&out)?;
    let mut trainer = Trainer::new(tcfg, &run)?.with_preemption_flag(preempt);
    // resume_latest continues from the newest checkpoint if one exists (a requeue),
    // else trains from scratch — paired with the preemption flag above, the run
    // survives a Slurm preempt/timeout: it checkpoints on the signal and resumes here.
    let (history, stop) = trainer.resume_latest(&mut policy, &reward, &tok, &train_prompts)?;

    // A preemption stop returns a PARTIAL history with a fresh checkpoint written. Exit
    // before held-out eval / the ladder gate: running them now would burn the Slurm
    // grace window and fail the run on incomplete data. The requeue (same
    // FERRL_CD35_RUN_ID) resumes from the checkpoint and reaches the gate when training
    // actually finishes.
    if stop == RunStop::Preempted {
        warn!(
            completed_windows = history.len(),
            "preempted mid-run: final checkpoint written; exiting before eval/gate so the \
             requeue (same FERRL_CD35_RUN_ID) resumes training"
        );
        return Ok(());
    }

    // A re-launch that resumed a checkpoint ALREADY at `steps` runs zero new steps and
    // returns an empty history — but eval/gate may NOT have completed: the job could
    // have been killed DURING post-training eval and then requeued. So never skip the
    // gate (reporting success without it would be a false pass). Recover the persisted
    // training metrics from `metrics.jsonl` and gate on those; the held-out eval below
    // re-runs from the resumed adapter regardless.
    let history = if history.is_empty() {
        let recovered = ferrl::read_metrics(run.metrics_path())
            .context("recover training metrics for an already-trained resumed run")?;
        if recovered.is_empty() {
            bail!(
                "resume_latest ran zero new steps and {} holds no metrics — cannot evaluate \
                 the training-reward gate; start a fresh FERRL_CD35_RUN_ID to retrain",
                run.metrics_path().display()
            );
        }
        warn!(
            recovered_windows = recovered.len(),
            "resumed an already-trained run (0 new steps); recovered training metrics to run \
             the held-out eval + gate (the gate may not have completed before the requeue)"
        );
        recovered
    } else {
        history
    };

    // Held-out eval AFTER training: `evaluate` scores base (adapter off) vs the
    // trained adapter (adapter on) in one pass, avg@k per prompt on the eval
    // distribution. No pre-train eval: the adapter starts as a no-op (`B = 0`),
    // so base == adapter there.
    let post = evaluate(&mut policy, &reward, &tok, &eval_prompts, &gen)?;
    report_and_gate(&history, &post, &gen)
}
