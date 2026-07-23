//! Data-parallel Countdown GRPO — the P9 gate the NCCL [`Comm`](ferrl::Comm)
//! bridge unlocks: a **real multi-GPU GRPO training run**, not just a synthetic
//! all-reduce.
//!
//! Launch one process per GPU on one node, with **every rank able to see all the
//! allocated GPUs** so each binds its own by `SLURM_LOCALID` — e.g. `srun --ntasks=2
//! --gres=gpu:2` with `CUDA_VISIBLE_DEVICES=0,1` exported to every task. Do **not** pass
//! `--gpus-per-task=1`: it masks each task to a single visible device, so a non-zero
//! rank tries to open a GPU index it cannot see and fails before training. Each process
//! builds its communicator from the Slurm environment
//! ([`NcclComm::from_slurm_env`](ferrl::NcclComm)) and
//! drives [`ferrl::QwenPolicy`] (real `Qwen3-0.6B-Base`, bf16 base / F32 `LoRA`
//! adapter) through GRPO over the verifiable [`ferrl::countdown`] reward. The
//! trainer all-reduces the `LoRA` gradients each accumulation window, so — starting
//! from identical weights (same `FERRL_CDDP_SEED`) and feeding each rank its own
//! global-row-index shard of the data — every rank stays in **bitwise lockstep**.
//!
//! # What the gate proves
//!
//! Each rank, after a completed run, prints two things the launcher checks:
//!
//! - `DP_LOCKSTEP rank=R sum=0x… sumsq=0x…` — a checksum of this rank's final
//!   adapter (sum and sum-of-squares of every trainable weight, as raw `f64`
//!   bits). **Every rank's checksum must be byte-identical**: that is the
//!   data-parallel correctness claim (same reduced gradients ⇒ same optimizer
//!   step ⇒ same weights), checked across processes by the launcher.
//! - `DP_TRAIN_PASS rank=R` (vs a bailing `DP_TRAIN_FAIL`) — this rank's training
//!   reward rose over the run by `FERRL_CDDP_MARGIN`: GRPO actually **learns**
//!   under data parallelism, not just stays consistent.
//!
//! # Resume under data parallelism
//!
//! The run is launched through [`Trainer::resume_latest`] with a **shared**
//! checkpoint directory ([`Trainer::with_checkpoints_dir`]): rank 0 writes the
//! world's checkpoints there, and on a requeue (same `FERRL_CDDP_RUN_ID`) rank 0's
//! discovery is **broadcast** so every rank resumes from the identical step in
//! lockstep. Paired with the cooperative preemption flag (`SIGTERM`/`SIGUSR1`),
//! the DP run survives a Slurm preempt/timeout: it checkpoints on the signal,
//! exits before eval/gate, and the requeue continues. A `DP_PREEMPTED rank=R` line
//! marks that path. Every rank derives the same immutable checkpoint-content and
//! loader-recipe digest before model construction; frozen-model or execution-recipe
//! drift makes ordinary resume fail before live policy mutation.
//!
//! Requires `--features nccl` (a multi-GPU build); the default build prints a
//! usage note and exits non-zero. Every knob has an `FERRL_CDDP_*` override.

// A standalone gate binary whose interface *is* PASS/FAIL + checksums on stdout
// for the launcher to grep — so `println!`/`eprintln!` are the right tool here
// (the library denies them via the workspace `print_*` lints).
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(feature = "nccl")]
fn main() -> anyhow::Result<()> {
    dp::run()
}

#[cfg(feature = "nccl")]
mod dp {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::str::FromStr;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use anyhow::{anyhow, bail, Context, Result};
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_transformers::models::qwen3::Config;
    use ferrl::countdown::{
        build_prompt, generate_dataset, CountdownConfig, CountdownProblem, CountdownReward,
    };
    use ferrl::{
        checkpoint_policy_sha256, read_metrics, Comm, HfTokenizer, LoaderOpts, Metrics, NcclComm,
        Policy, QwenGradModel, QwenPolicy, RunDir, RunStop, Sample, Trainer, TrainerConfig,
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

    /// Resolve the production EOS contract before any durable run publication.
    ///
    /// Unset requires one scalar checkpoint EOS; the exact string `none` is the
    /// explicit full-width opt-out; any integer override is checked against declared
    /// multi-EOS membership plus the model and tokenizer vocabularies.
    fn resolve_eos(dir: &Path, tokenizer: &HfTokenizer) -> Result<Option<u32>> {
        let selection = match env::var("FERRL_CDDP_EOS") {
            Err(env::VarError::NotPresent) => ferrl::CheckpointEosSelection::CheckpointDefault,
            Err(error) => return Err(anyhow!("read FERRL_CDDP_EOS: {error}")),
            Ok(raw) if raw == "none" => ferrl::CheckpointEosSelection::Disabled,
            Ok(raw) => ferrl::CheckpointEosSelection::Explicit(raw.parse::<u32>().with_context(
                || {
                    format!(
                        "FERRL_CDDP_EOS must be an integer id or the exact string 'none', got {raw:?}"
                    )
                },
            )?),
        };
        ferrl::resolve_checkpoint_eos(dir, tokenizer, selection).map_err(Into::into)
    }

    /// Coordinate rank-local EOS resolution, then require identical resolved semantics.
    fn coordinate_eos_resolution(
        comm: &dyn Comm,
        local: Result<Option<u32>>,
    ) -> Result<Option<u32>> {
        let failures = comm
            .all_reduce_scalar_sum(if local.is_err() { 1.0 } else { 0.0 })
            .context("coordinate rank-local EOS resolution")?;
        if failures != 0.0 {
            return match local {
                Err(error) => Err(error),
                Ok(_) => bail!("a peer rank failed checkpoint/tokenizer EOS resolution"),
            };
        }
        let eos = local.expect("zero global EOS failures implies local success");
        ferrl::validate_resolved_eos_consensus(eos, comm)
            .context("require rank-identical resolved EOS semantics")?;
        Ok(eos)
    }

    /// The base-weight dtype (`FERRL_CDDP_BASE_DTYPE`): `bf16` (default — the
    /// production split that halves base weights + retained activations on Ampere+),
    /// `f16`, or `f32`. The adapter is **always** F32 regardless. `f32` keeps the run
    /// portable to GPUs without bf16 matmul (e.g. V100/sm_70), where data-parallel
    /// correctness — the point of this gate — holds identically.
    fn resolve_base_dtype() -> Result<DType> {
        match env::var("FERRL_CDDP_BASE_DTYPE").as_deref() {
            Err(_) | Ok("bf16") => Ok(DType::BF16),
            Ok("f16") => Ok(DType::F16),
            Ok("f32") => Ok(DType::F32),
            Ok(other) => bail!("FERRL_CDDP_BASE_DTYPE must be bf16/f16/f32, got {other:?}"),
        }
    }

    /// Build the policy over the real `Qwen3-0.6B-Base` checkpoint on `device` (the
    /// rank's local GPU), `FERRL_CDDP_BASE_DTYPE` base / F32 `LoRA` adapter, the legacy
    /// q/v recipe — the same recipe as the single-process `countdown_grpo` harness.
    fn build_policy(dir: &Path, device: &Device) -> Result<(QwenPolicy, HfTokenizer)> {
        let cfg = load_config(dir)?;
        let buf = std::fs::read(dir.join("model.safetensors")).context("read model.safetensors")?;
        let vb = VarBuilder::from_buffered_safetensors(buf, resolve_base_dtype()?, device)
            .context("load model.safetensors onto the GPU")?;
        let rank = env_parse("FERRL_CDDP_RANK", 16usize);
        let alpha = env_parse("FERRL_CDDP_ALPHA", 32.0f64);
        let model = QwenGradModel::load_with_adapter_dtype(&cfg, &vb, rank, alpha, DType::F32)
            .context("build QwenGradModel")?;
        // SAME seed on every rank ⇒ identical initial adapter ⇒ the lockstep
        // invariant holds (the trainer shards the data by global row index, so
        // identical seeds do NOT mean identical rollouts).
        let seed = env_parse("FERRL_CDDP_SEED", 1234u64);
        let temperature = env_parse("FERRL_CDDP_TEMP", 1.0f64);
        let policy = QwenPolicy::new(model, seed, temperature);
        let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).context("load tokenizer")?;
        Ok((policy, tok))
    }

    /// The Countdown training prompts (every rank generates the identical set from
    /// the same `FERRL_CDDP_DATA_SEED`; the trainer shards it per rank). No held-out
    /// split — this gate proves DP correctness + learning, not the P4 beats-base
    /// claim (that is `countdown_grpo`'s job, and a held-out *win* is downstream).
    fn build_train_samples() -> Vec<Sample<CountdownProblem>> {
        let cd_cfg = CountdownConfig {
            num_count: env_parse("FERRL_CDDP_NUMCOUNT", 3usize),
            min_number: env_parse("FERRL_CDDP_MINNUM", 1u32),
            max_number: env_parse("FERRL_CDDP_MAXNUM", 20u32),
            max_target: env_parse("FERRL_CDDP_MAXTARGET", 1000u32),
        };
        let train_n = env_parse("FERRL_CDDP_TRAIN_N", 64usize);
        let data_seed = env_parse("FERRL_CDDP_DATA_SEED", 7u64);
        generate_dataset(data_seed, train_n, &cd_cfg)
            .into_iter()
            .map(|p| Sample::new(build_prompt(&p), p))
            .collect()
    }

    /// The trainer config, every field env-overridable. `grad_accum_steps` is the
    /// **per-rank** window; the global batch is `grad_accum_steps × world_size`.
    fn build_trainer_config(eos_token_id: Option<u32>) -> TrainerConfig {
        TrainerConfig {
            steps: env_parse("FERRL_CDDP_STEPS", 100u64),
            group_size: env_parse("FERRL_CDDP_GROUP", 8usize),
            grad_accum_steps: env_parse("FERRL_CDDP_ACCUM", 2usize),
            max_new_tokens: env_parse("FERRL_CDDP_MAXNEW", 48usize),
            temperature: env_parse("FERRL_CDDP_TEMP", 1.0f64),
            lr: env_parse("FERRL_CDDP_LR", 1e-5f64),
            beta: env_parse("FERRL_CDDP_BETA", 0.0f64),
            checkpoint_every: Some(env_parse("FERRL_CDDP_CKPT", 10u64)),
            eos_token_id,
            loss_type: ferrl::LossType::Grpo,
            max_grad_norm: None,
            truncation_masking: false,
            ..TrainerConfig::default()
        }
    }

    /// Mean training reward over the first vs last quarter of the run — the trend
    /// the gate reads (a single noisy step is not evidence either way).
    fn reward_trend(history: &[Metrics]) -> (f32, f32) {
        let window = (history.len() / 4).max(1);
        let mean = |it: &mut dyn Iterator<Item = f32>| {
            let (sum, n) = it.fold((0.0f32, 0u32), |(s, n), v| (s + v, n + 1));
            if n == 0 {
                0.0
            } else {
                sum / n as f32
            }
        };
        let first = mean(&mut history.iter().take(window).map(|m| m.reward_mean));
        let last = mean(&mut history.iter().rev().take(window).map(|m| m.reward_mean));
        (first, last)
    }

    /// A checksum of the policy's trainable adapter: the sum and the sum-of-squares
    /// of every weight, in `f64`. Two moments make a divergence between ranks
    /// (which must be byte-identical under the lockstep invariant) detectable with
    /// negligible collision risk — the launcher compares the raw bits across ranks.
    fn adapter_checksum(policy: &QwenPolicy) -> Result<(f64, f64)> {
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for var in policy.trainable_vars() {
            for x in var.as_tensor().flatten_all()?.to_vec1::<f32>()? {
                let x = f64::from(x);
                sum += x;
                sumsq += x * x;
            }
        }
        Ok((sum, sumsq))
    }

    /// Open this rank's per-rank run dir (so each rank's `metrics.jsonl` is its own,
    /// never interleaved) and the **shared** checkpoint dir (rank 0 writes, all read
    /// — the auto-resume substrate), plus install the cooperative preemption flag.
    ///
    /// A stable `FERRL_CDDP_RUN_ID` (default: the Slurm job id, identical across
    /// ranks and preserved across a requeue) makes a requeued job CONTINUE the same
    /// run; the shared checkpoint dir is keyed on it too, so every rank of every
    /// launch agrees on the same directory.
    fn open_dp_run(rank: usize) -> Result<(RunDir, PathBuf, Arc<AtomicBool>)> {
        let preempt = Arc::new(AtomicBool::new(false));
        for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGUSR1] {
            signal_hook::flag::register(sig, Arc::clone(&preempt))
                .context("install preemption signal handler")?;
        }
        let out = env_parse("FERRL_CDDP_OUT", "/tmp/ferrl-runs".to_string());
        let run_id = env::var("FERRL_CDDP_RUN_ID")
            .ok()
            .or_else(|| env::var("SLURM_JOB_ID").ok())
            .unwrap_or_else(|| {
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                format!("countdown-grpo-dp-{stamp}")
            });
        // Per-rank metrics dir; reopen on a requeue so the stream continues.
        let per_rank_id = format!("{run_id}-rank{rank}");
        let run = if Path::new(&out).join(&per_rank_id).is_dir() {
            info!(run_id = %per_rank_id, "resuming existing per-rank run directory");
            RunDir::open(Path::new(&out), &per_rank_id).context("open existing run dir")?
        } else {
            RunDir::create(Path::new(&out), &per_rank_id).context("create run dir")?
        };
        // ONE shared checkpoint dir for the whole world (single node: a shared /tmp
        // path; multi-node would point this at NFS). Overridable for explicit control.
        let shared_ckpts = env::var("FERRL_CDDP_CKPT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| Path::new(&out).join(format!("ckpts-{run_id}")));
        Ok((run, shared_ckpts, preempt))
    }

    pub(crate) fn run() -> Result<()> {
        let _ = ferrl::init_tracing();

        // Build the communicator from the Slurm environment FIRST: it opens this
        // rank's local CUDA device (SLURM_LOCALID), which the policy must share.
        let comm = NcclComm::from_slurm_env().context(
            "build NcclComm from the Slurm env — launch one process per GPU under srun, \
             with SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID set",
        )?;
        let rank = comm.rank();
        let world = comm.world_size();
        let device = comm.device().clone();
        // Stamp every event this rank logs with rank/world — all ranks share one stdout
        // under srun, so the span is what makes the interleaved lines attributable. The
        // trainer enters its own (identical) run span + a nested per-step span; this one
        // covers the launcher's setup / eval / gate events.
        let _run = ferrl::run_span(rank, world).entered();
        // Driver/PTX preflight on this rank's device (proactive warn, then a forced
        // JIT so a real mismatch fails here with an actionable message).
        if let Some(w) = ferrl::check_driver_compat(&device).warning() {
            warn!("{w}");
        }
        ferrl::guard_first_kernel(&device).context("CUDA preflight")?;

        let weights = env::var("FERRL_QWEN_WEIGHTS").map_err(|_| {
            anyhow!("set FERRL_QWEN_WEIGHTS to the Qwen3-0.6B-Base asset directory")
        })?;
        let dir = PathBuf::from(weights);
        let checkpoint_policy_sha256 = checkpoint_policy_sha256(
            &dir,
            &LoaderOpts {
                lora_rank: env_parse("FERRL_CDDP_RANK", 16usize),
                lora_alpha: env_parse("FERRL_CDDP_ALPHA", 32.0f64),
                base_dtype: resolve_base_dtype()?,
                adapter_dtype: DType::F32,
                seed: env_parse("FERRL_CDDP_SEED", 1234u64),
                temperature: env_parse("FERRL_CDDP_TEMP", 1.0f64),
                ..LoaderOpts::default()
            },
        )?;
        let (mut policy, tok) = build_policy(&dir, &device)?;
        let train_samples = build_train_samples();
        let reward = CountdownReward::default();
        let eos = coordinate_eos_resolution(&comm, resolve_eos(&dir, &tok))?;
        let tcfg = build_trainer_config(eos);
        info!(
            steps = tcfg.steps,
            group_size = tcfg.group_size,
            grad_accum_steps = tcfg.grad_accum_steps,
            lr = tcfg.lr,
            train = train_samples.len(),
            "data-parallel countdown GRPO run starting"
        );

        let (run, shared_ckpts, preempt) = open_dp_run(rank)?;
        let metrics_path = run.metrics_path();
        let mut trainer = Trainer::with_comm(tcfg, &run, comm)?
            .with_checkpoint_policy_sha256(checkpoint_policy_sha256)
            .with_preemption_flag(preempt)
            .with_checkpoints_dir(shared_ckpts);
        // resume_latest auto-discovers rank 0's newest checkpoint and broadcasts it
        // so every rank resumes in lockstep (or all start fresh) — see the module
        // docs. Paired with the preemption flag, a Slurm preempt is survivable.
        let (_history, stop) = trainer.resume_latest(&mut policy, &reward, &tok, &train_samples)?;

        // A preemption stop wrote a fresh checkpoint and returned a PARTIAL history.
        // Exit before the gate so the requeue (same FERRL_CDDP_RUN_ID) resumes — the
        // grace window is short; gating now would burn it and fail on incomplete data.
        if stop == RunStop::Preempted {
            warn!("preempted mid-run: checkpoint written; exiting for the requeue");
            println!("[rank {rank}] DP_PREEMPTED");
            return Ok(());
        }

        // Gate on the FULL persisted trajectory (a resume appends to metrics.jsonl),
        // not this launch's in-memory history — which is empty for an already-trained
        // resume and only the tail for a mid-run resume.
        let history = read_metrics(&metrics_path).context("read training metrics for the gate")?;
        if history.is_empty() {
            bail!(
                "[rank {rank}] no training metrics at {} — cannot gate the reward trend",
                metrics_path.display()
            );
        }

        // Cross-rank lockstep checksum (launcher compares these bits across ranks):
        // identical weights ⇒ identical bits.
        let (sum, sumsq) = adapter_checksum(&policy)?;
        println!(
            "[rank {rank}] DP_LOCKSTEP rank={rank} world={world} sum=0x{:016x} sumsq=0x{:016x}",
            sum.to_bits(),
            sumsq.to_bits()
        );

        // Per-rank learning gate: this rank's training reward rose by the margin.
        let (first, last) = reward_trend(&history);
        let margin = env_parse("FERRL_CDDP_MARGIN", 0.05f32);
        let rises = (last - first) > margin;
        info!(first, last, margin, rises, "DP train reward trend");
        if rises {
            println!("[rank {rank}] DP_TRAIN_PASS");
            Ok(())
        } else {
            bail!(
                "[rank {rank}] DP_TRAIN_FAIL: reward did not rise by the margin \
                 (first={first}, last={last}, margin={margin})"
            )
        }
    }
}

#[cfg(not(feature = "nccl"))]
fn main() {
    eprintln!(
        "countdown_grpo_dp: build with --features nccl (a multi-GPU build) and launch one \
         process per GPU under srun, with every rank seeing all the allocated GPUs so each \
         binds its own by SLURM_LOCALID (e.g. srun --ntasks=2 --gres=gpu:2 with \
         CUDA_VISIBLE_DEVICES=0,1 — NOT --gpus-per-task=1, which masks each task to one \
         device). See .git/ for the gate launcher."
    );
    std::process::exit(2);
}
