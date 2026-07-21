//! Run telemetry: structured tracing plus a per-run on-disk layout.
//!
//! Every training run materializes a `runs/<run_id>/` directory containing
//! `config.json` (the run configuration), `metrics.jsonl` (one [`Metrics`] JSON
//! object per optimizer step), an optional `candidates.jsonl` stream (top sampled
//! completions when enabled), a `checkpoints/` subdirectory, and `run.log` (a
//! human-readable log). [`init_tracing`] wires up `tracing` once for the
//! process; [`RunDir`] owns the directory; [`MetricsWriter`] appends step
//! metrics as JSON Lines; [`CandidateWriter`] appends sampled-completion
//! provenance as JSON Lines.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

/// One instantaneous CUDA memory reading for the process' current device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuMemorySnapshot {
    /// Free device memory in bytes.
    pub free_bytes: u64,
    /// Total device memory in bytes.
    pub total_bytes: u64,
    /// Used device memory in bytes (`total - free`, saturating).
    pub used_bytes: u64,
}

impl GpuMemorySnapshot {
    #[cfg(feature = "cuda")]
    fn from_free_total(free: usize, total: usize) -> Self {
        let free_bytes = free as u64;
        let total_bytes = total as u64;
        Self {
            free_bytes,
            total_bytes,
            used_bytes: total_bytes.saturating_sub(free_bytes),
        }
    }
}

/// Read CUDA memory for the current device, when this binary was built with
/// `--features cuda` and the CUDA runtime can answer.
///
/// Returns `None` for CPU/default builds and for runtime query failures. This is
/// telemetry, not a training precondition: a failed memory probe must never turn
/// a valid run red.
#[must_use]
pub fn cuda_memory_snapshot() -> Option<GpuMemorySnapshot> {
    cuda_memory_snapshot_impl()
}

/// One CUDA memory probe at a named phase boundary within an optimizer step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuMemoryProbeEvent {
    /// Stable phase label, e.g. `rollout_prefill_end` or `item_backward_start`.
    pub phase: String,
    /// CUDA memory used by this rank at the probe, in bytes.
    pub used_bytes: u64,
    /// CUDA memory free on this rank's current device at the probe, in bytes.
    pub free_bytes: u64,
    /// Total CUDA memory reported for this rank's current device, in bytes.
    pub total_bytes: u64,
    /// Peak per-step memory increase from the first successful probe, in bytes.
    pub peak_delta_bytes: u64,
}

/// A cached decoder's sequence-retention state at a named phase boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderCacheSnapshot {
    /// Stable phase label, e.g. `rollout_prefill_end` or `rollout_decode_end`.
    pub phase: String,
    /// Zero-based decoder layer index.
    pub layer_index: usize,
    /// Architecture-owned cache kind label, e.g. `full_attention` or
    /// `sliding_attention`.
    pub kind: String,
    /// Absolute number of tokens this layer has consumed since the last reset.
    pub seen_tokens: usize,
    /// Number of tokens physically retained in this layer's cache.
    pub retained_tokens: usize,
    /// Retention cap for windowed caches. Missing means unbounded/global cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retained_tokens: Option<usize>,
}

/// Sink for optional model-path telemetry captured while a [`Policy`](crate::Policy)
/// generates a rollout.
pub trait ModelTelemetryRecorder {
    /// Record a named CUDA-memory phase boundary.
    fn record_phase(&mut self, phase: &'static str);

    /// Record decoder cache-retention snapshots. Empty slices are ignored by
    /// conforming recorders.
    fn record_decoder_cache(&mut self, snapshots: Vec<DecoderCacheSnapshot>);
}

#[cfg(feature = "cuda")]
fn cuda_memory_snapshot_impl() -> Option<GpuMemorySnapshot> {
    cudarc::runtime::result::mem_get_info()
        .ok()
        .map(|(free, total)| GpuMemorySnapshot::from_free_total(free, total))
}

#[cfg(not(feature = "cuda"))]
fn cuda_memory_snapshot_impl() -> Option<GpuMemorySnapshot> {
    None
}

/// Errors raised while setting up or writing run telemetry.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// A filesystem operation (create dir / open / write) failed.
    #[error("telemetry io error at {path}: {source}")]
    Io {
        /// Path that was being operated on when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// A [`Metrics`] record could not be serialized to JSON.
    #[error("failed to serialize metrics: {0}")]
    Serialize(#[from] serde_json::Error),
    /// A line of `metrics.jsonl` could not be parsed back into a [`Metrics`]
    /// record (see [`read_metrics`]).
    #[error("failed to parse a metrics record at {path}: {source}")]
    Deserialize {
        /// The `metrics.jsonl` being read.
        path: PathBuf,
        /// Underlying JSON parse error.
        source: serde_json::Error,
    },
    /// [`RunDir::create`] was given a `run_id` whose directory already holds an
    /// append-only telemetry stream — appending a fresh run to a prior run's
    /// stream would silently interleave two runs. Use a new `run_id`, or
    /// [`RunDir::open`] to deliberately continue the existing run (resume).
    #[error(
        "run directory already contains an append-only telemetry stream at {path} \
         (duplicate run_id? use RunDir::open to resume)"
    )]
    DuplicateRun {
        /// The pre-existing append-only telemetry stream.
        path: PathBuf,
    },
}

impl TelemetryError {
    fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Initialize the global `tracing` subscriber from the `RUST_LOG` environment
/// variable, falling back to `info`.
///
/// Uses [`try_init`](tracing_subscriber::util::SubscriberInitExt::try_init), so
/// it is safe to call more than once (and from tests): a second call returns
/// `Ok(())` without replacing the existing subscriber rather than panicking.
///
/// # Errors
///
/// Never returns `Err` in practice — a failed `try_init` (subscriber already
/// set) is treated as success — but the signature is fallible to leave room for
/// future initialization that can genuinely fail.
pub fn init_tracing() -> Result<(), TelemetryError> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Ignore the "already initialized" error so repeated/test calls are no-ops.
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true))
        .try_init();
    Ok(())
}

/// Build the per-run `rank`/`world` span that stamps **every** event emitted while it
/// is entered with this rank's identity.
///
/// Under data parallelism every rank logs to the same stdout/stderr, so without a rank
/// stamp the interleaved lines are unattributable. Enter this span once for the lifetime
/// of a run (hold the returned guard) and every `tracing` event below it — the policy's,
/// the reward's, the trainer's — carries `rank` and `world`:
///
/// ```
/// let _run = ferrl::run_span(0, 1).entered();
/// // Every `tracing` event emitted while `_run` is held carries rank=0 world=1.
/// ```
///
/// [`Trainer`](crate::trainer::Trainer) enters this span around its run loop and a
/// nested per-step `step` span inside it, so a trainer's own events are
/// `rank`/`world`/`step`-stamped automatically; a launcher wraps its setup/eval/gate
/// events by entering this span itself.
///
/// The span is created at **ERROR** level — the max severity — deliberately. A span is
/// only entered while its level passes the active filter, so an `info`-level context
/// span would silently drop out under `RUST_LOG=warn`/`error` (the very filters used to
/// quiet a long run) and the warnings/errors that matter most — e.g. the preemption
/// warn — would emit *without* rank/world/step. ERROR level keeps the context enabled
/// for any filter that emits anything at all, so emitted warnings/errors always carry it.
#[must_use]
pub fn run_span(rank: usize, world: usize) -> tracing::Span {
    tracing::error_span!("run", rank, world)
}

/// The per-step span nested under [`run_span`]: stamps `step` onto every event emitted
/// during one optimizer step.
///
/// ERROR level for the same reason as [`run_span`] — the context must survive a
/// `RUST_LOG=warn`/`error` filter so emitted warnings/errors keep their `step`. Kept a
/// function rather than an inline `error_span!` at the call site so the macro's
/// level-check branch counts against this trivial helper, not the trainer's already
/// complexity-bounded run loop.
#[must_use]
pub(crate) fn step_span(step: u64) -> tracing::Span {
    tracing::error_span!("step", step)
}

/// Owns the on-disk `runs/<run_id>/` directory for a single training run.
///
/// Construction creates the directory tree eagerly. Paths to the standard
/// artifacts are exposed via accessors so callers never hand-build them.
#[derive(Debug, Clone)]
pub struct RunDir {
    run_id: String,
    root: PathBuf,
}

impl RunDir {
    /// Standard filename for the per-step metrics stream.
    pub const METRICS_FILE: &'static str = "metrics.jsonl";
    /// Standard filename for the optional per-candidate ledger stream.
    pub const CANDIDATES_FILE: &'static str = "candidates.jsonl";
    /// Standard filename for the serialized run configuration.
    pub const CONFIG_FILE: &'static str = "config.json";
    /// Standard filename for the human-readable run log.
    pub const LOG_FILE: &'static str = "run.log";
    /// Standard subdirectory for model checkpoints.
    pub const CHECKPOINTS_DIR: &'static str = "checkpoints";

    /// Create `runs_root/<run_id>/` (and its `checkpoints/` subdir) for a
    /// **fresh** run.
    ///
    /// Fails loud if the directory already holds an append-only telemetry
    /// stream (`metrics.jsonl` or `candidates.jsonl`): those writers append, so
    /// a duplicate `run_id` would silently interleave a new run's stream into a
    /// prior run's file (the `RUNDIR-APPEND` hazard). To
    /// deliberately continue an existing run — the checkpoint-resume path — use
    /// [`open`](Self::open) instead.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::DuplicateRun`] if an append-only telemetry
    /// stream already exists under `runs_root/<run_id>/`, or [`TelemetryError::Io`]
    /// if any directory cannot be created.
    pub fn create(
        runs_root: impl AsRef<Path>,
        run_id: impl Into<String>,
    ) -> Result<Self, TelemetryError> {
        let run_id = run_id.into();
        let root = runs_root.as_ref().join(&run_id);
        for file in [Self::METRICS_FILE, Self::CANDIDATES_FILE] {
            let path = root.join(file);
            if path.exists() {
                return Err(TelemetryError::DuplicateRun { path });
            }
        }
        fs::create_dir_all(&root).map_err(|e| TelemetryError::io(&root, e))?;
        let ckpt = root.join(Self::CHECKPOINTS_DIR);
        fs::create_dir_all(&ckpt).map_err(|e| TelemetryError::io(&ckpt, e))?;
        Ok(Self { run_id, root })
    }

    /// Open the **existing** `runs_root/<run_id>/` to continue it — the
    /// checkpoint-resume path, where appending to the prior `metrics.jsonl`
    /// stream is exactly the intent. Creates any missing subdirectory (a
    /// partially-materialized run dir is tolerated) but requires the run root
    /// itself to exist, so a typo'd `run_id` fails loud instead of silently
    /// starting an empty "resume".
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Io`] if the run root does not exist or a
    /// subdirectory cannot be created.
    pub fn open(
        runs_root: impl AsRef<Path>,
        run_id: impl Into<String>,
    ) -> Result<Self, TelemetryError> {
        let run_id = run_id.into();
        let root = runs_root.as_ref().join(&run_id);
        if !root.is_dir() {
            return Err(TelemetryError::io(
                &root,
                io::Error::new(io::ErrorKind::NotFound, "run directory does not exist"),
            ));
        }
        let ckpt = root.join(Self::CHECKPOINTS_DIR);
        fs::create_dir_all(&ckpt).map_err(|e| TelemetryError::io(&ckpt, e))?;
        Ok(Self { run_id, root })
    }

    /// The run identifier this directory was created for.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The run directory root (`runs_root/<run_id>`).
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to `metrics.jsonl`.
    #[must_use]
    pub fn metrics_path(&self) -> PathBuf {
        self.root.join(Self::METRICS_FILE)
    }

    /// Path to `candidates.jsonl`.
    #[must_use]
    pub fn candidates_path(&self) -> PathBuf {
        self.root.join(Self::CANDIDATES_FILE)
    }

    /// Path to `config.json`.
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.root.join(Self::CONFIG_FILE)
    }

    /// Path to `run.log`.
    #[must_use]
    pub fn log_path(&self) -> PathBuf {
        self.root.join(Self::LOG_FILE)
    }

    /// Path to the `checkpoints/` subdirectory.
    #[must_use]
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.root.join(Self::CHECKPOINTS_DIR)
    }

    /// Serialize `config` to `config.json` (pretty-printed).
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError`] if serialization or the write fails.
    pub fn write_config<C: Serialize>(&self, config: &C) -> Result<(), TelemetryError> {
        let path = self.config_path();
        let json = serde_json::to_string_pretty(config)?;
        fs::write(&path, json).map_err(|e| TelemetryError::io(&path, e))
    }

    /// Open a [`MetricsWriter`] appending to this run's `metrics.jsonl`.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Io`] if the file cannot be opened for append.
    pub fn metrics_writer(&self) -> Result<MetricsWriter, TelemetryError> {
        MetricsWriter::open(self.metrics_path())
    }

    /// Open a [`CandidateWriter`] appending to this run's `candidates.jsonl`.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Io`] if the candidate ledger file cannot be opened.
    pub fn candidate_writer(&self) -> Result<CandidateWriter, TelemetryError> {
        CandidateWriter::open(self.candidates_path())
    }
}

/// One optimizer-step's worth of training metrics, serialized as a single JSON
/// line in `metrics.jsonl`.
///
/// All fields are plain scalars; rewards are aggregated to mean/std *before*
/// reaching this struct (rewards are never tensors in `ferrl`).
///
/// `#[non_exhaustive]`: construct via [`Metrics::at_step`] then set fields, so
/// adding a metric later is not a breaking change for downstream crates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Metrics {
    /// Global optimizer step (0-based).
    pub step: u64,
    /// Mean scalar reward over the batch (all completions in the step's window —
    /// `grad_accum_steps` prompts' groups pooled when accumulating).
    pub reward_mean: f32,
    /// Standard deviation of scalar rewards over the batch. At `grad_accum_steps > 1`
    /// this is the std over the window's pooled completions (mixing cross-prompt
    /// variance), not a single group's spread.
    pub reward_std: f32,
    /// Fraction of GRPO groups in the batch whose reward std is `0` — every
    /// completion in the group scored identically, so its advantages are all `0`
    /// and it contributes no gradient. Mirrors TRL's `frac_reward_zero_std`; a
    /// value near `1` means the batch taught the policy almost nothing (rewards
    /// saturated, or the task is too easy/hard for the current model).
    ///
    /// `#[serde(default)]` so `metrics.jsonl` records written before this field
    /// existed still deserialize (defaulting to `0.0`).
    #[serde(default)]
    pub frac_reward_zero_std: f32,
    /// Mean k3 KL to the reference policy.
    pub kl: f32,
    /// Fraction of tokens whose surrogate hit the PPO clip band.
    pub clip_ratio: f32,
    /// Fraction of the window's completions masked out by **truncation
    /// masking** (DAPO overlong filtering: ran to the full completion width
    /// without sampling EOS while `truncation_masking` is on). `0` when the
    /// knob is off or no EOS token is configured.
    ///
    /// `#[serde(default)]` so `metrics.jsonl` records written before this field
    /// existed still deserialize (defaulting to `0.0`).
    #[serde(default)]
    pub frac_truncated: f32,
    /// Mean completion length in tokens.
    pub completion_len: f32,
    /// Masked token mean of the **train/rollout importance ratio**
    /// `exp(logp_train − logp_rollout)` over the window's loss-carrying tokens,
    /// where `logp_train` is the trainer's own (detached, temperature-consistent)
    /// scoring of the rollout and `logp_rollout` the behavior log-prob the
    /// sampler captured at draw time
    /// ([`Rollout::rollout_logprobs`](crate::policy::Rollout::rollout_logprobs)).
    ///
    /// **This is a sampling-health indicator, not a drift meter**: tokens are
    /// drawn from the rollout distribution, so the ratio's *expectation* is
    /// exactly `1` under **arbitrary** drift (`E[π_train/π_rollout] = Σ π_train
    /// = 1` — both are normalized). A mean away from `1` therefore signals
    /// estimator noise / a heavy upper tail, NOT how large the gap is; read the
    /// gap off [`rollout_logratio_mean`](Self::rollout_logratio_mean),
    /// [`rollout_ratio_max`](Self::rollout_ratio_max), and
    /// [`frac_rollout_ratio_capped`](Self::frac_rollout_ratio_capped). Reported
    /// whenever the policy captures rollout log-probs — independent of whether
    /// the TIS *correction* is enabled; check
    /// [`rollout_capture_tokens`](Self::rollout_capture_tokens) to distinguish
    /// a measured `1.0` from no data at all (`1.0` is also the
    /// `#[serde(default)]` for older records).
    #[serde(default = "default_rollout_ratio")]
    pub rollout_ratio_mean: f32,
    /// Masked token mean of the **log**-ratio `logp_train − logp_rollout` — the
    /// actual drift meter: its expectation is `−KL(rollout ‖ train)`, which is
    /// `0` iff the rollout was exactly on-policy and strictly negative (in
    /// expectation) under any drift. Watch this (together with the max and the
    /// capped fraction) to decide when to flip
    /// [`TrainerConfig::tis`](crate::trainer::TrainerConfig::tis) on. `0.0`
    /// when no capture is available (also the `#[serde(default)]`).
    #[serde(default)]
    pub rollout_logratio_mean: f32,
    /// Maximum train/rollout importance ratio over the window's loss-carrying
    /// tokens — the outlier detector (a single far-off-policy token can destabilize
    /// an update long before any mean moves). `1.0` when no capture is available.
    #[serde(default = "default_rollout_ratio")]
    pub rollout_ratio_max: f32,
    /// Fraction of the window's loss-carrying tokens whose train/rollout ratio
    /// exceeded [`TrainerConfig::tis_imp_ratio_cap`](crate::trainer::TrainerConfig::tis_imp_ratio_cap)
    /// — the tokens TIS truncates when enabled, and *would* truncate when not
    /// (the "flip the correction on when the gap shows" signal). `0` when no
    /// capture is available.
    #[serde(default)]
    pub frac_rollout_ratio_capped: f32,
    /// Number of loss-carrying tokens the window's rollout-ratio telemetry was
    /// computed over. **`0` means the telemetry is dark** — the policy captured
    /// no behavior log-probs, or every captured group was skipped before its
    /// scoring snapshot existed (degenerate at `beta == 0`) — and the ratio
    /// fields then carry their neutral values rather than measurements; an
    /// operator watching the gap should treat `0` as "not monitored", not
    /// "healthy". `#[serde(default)]` (`0`) for older records.
    #[serde(default)]
    pub rollout_capture_tokens: u32,
    /// Number of all-pad completion rows (no valid tokens) that contributed `0`
    /// to the loss this step. Such rows are tolerated rather than fatal (see
    /// [`crate::grpo::masked_mean`] / [`crate::grpo::zero_mask_rows`]); recorded
    /// here so a batch that silently lost completions is observable. Normally `0`.
    ///
    /// `#[serde(default)]` for back-compat with pre-existing `metrics.jsonl`
    /// records.
    #[serde(default)]
    pub dropped_rows: u32,
    /// Global gradient norm after backward (pre-clip).
    pub grad_norm: f32,
    /// Effective learning rate for this step.
    pub lr: f32,
    /// Effective KL coefficient for this step.
    #[serde(default)]
    pub beta: f32,
    /// Wall-clock seconds this optimizer step took — the whole window (rollout +
    /// reward + the `mu` inner update epochs), measured around
    /// [`Trainer::run`](crate::trainer::Trainer)'s per-step body. Lets an operator
    /// read steps/sec and an ETA off a long run that is otherwise blind to its own
    /// pace. Under DP this is the rank's own wall time; ranks run the same
    /// collectives in lockstep so the figure is ~equal across the world.
    ///
    /// `#[serde(default)]` (`0.0`) for records written before this field existed
    /// and for a step whose duration was not measured.
    #[serde(default)]
    pub step_secs: f32,
    /// Completion tokens **this rank** generated this step divided by `step_secs`
    /// — the rollout-throughput number a long run is otherwise blind to. It counts
    /// the window's real (EOS-inclusive) completion tokens, the same total that
    /// drives the DAPO loss normalizer. **Per-rank:** the world figure is
    /// `world_size ×` this (each rank rolls out its own prompt shard); a report
    /// tool multiplies up. `0.0` when the step was not timed (`#[serde(default)]`,
    /// also the value for older records).
    #[serde(default)]
    pub tokens_per_sec: f32,
    /// CUDA memory used by this rank at the first step probe, in bytes.
    ///
    /// Written only when [`TrainerConfig::gpu_memory_probe`](crate::trainer::TrainerConfig::gpu_memory_probe)
    /// is enabled in a CUDA build; `0` means unmeasured.
    #[serde(default)]
    pub cuda_mem_start_used_bytes: u64,
    /// Peak CUDA memory used by this rank across the step's probes, in bytes.
    /// `0` means unmeasured.
    #[serde(default)]
    pub cuda_mem_peak_used_bytes: u64,
    /// CUDA memory used by this rank at the final step probe, in bytes. `0`
    /// means unmeasured.
    #[serde(default)]
    pub cuda_mem_end_used_bytes: u64,
    /// Total CUDA memory reported for this rank's current device, in bytes. `0`
    /// means unmeasured.
    #[serde(default)]
    pub cuda_mem_total_bytes: u64,
    /// Peak per-step CUDA memory increase from the first probe, in bytes. `0`
    /// means unmeasured or no increase.
    #[serde(default)]
    pub cuda_mem_peak_delta_bytes: u64,
    /// Per-phase CUDA memory probe events for this optimizer step.
    ///
    /// Written only when GPU memory probing is enabled and at least one CUDA
    /// runtime probe succeeds. Empty and absent both mean unmeasured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cuda_mem_probe_events: Vec<GpuMemoryProbeEvent>,
    /// Decoder-cache retention snapshots captured during rollout generation.
    ///
    /// Empty and absent both mean the policy/model did not provide cache
    /// snapshots, or memory probing was disabled for the run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decoder_cache_snapshots: Vec<DecoderCacheSnapshot>,
}

/// `serde` default for the rollout-ratio metrics: `1.0` (exactly on-policy —
/// what the math assumes when no behavior log-probs were captured, and what
/// records written before these fields existed implicitly assumed).
fn default_rollout_ratio() -> f32 {
    1.0
}

impl Metrics {
    /// A zeroed record at the given step — convenient for tests and for steps
    /// where a particular quantity is not yet available. (The rollout-ratio
    /// fields default to their neutral `1.0`, not `0`.)
    #[must_use]
    pub fn at_step(step: u64) -> Self {
        Self {
            step,
            reward_mean: 0.0,
            reward_std: 0.0,
            frac_reward_zero_std: 0.0,
            kl: 0.0,
            clip_ratio: 0.0,
            frac_truncated: 0.0,
            completion_len: 0.0,
            rollout_ratio_mean: 1.0,
            rollout_logratio_mean: 0.0,
            rollout_ratio_max: 1.0,
            frac_rollout_ratio_capped: 0.0,
            rollout_capture_tokens: 0,
            dropped_rows: 0,
            grad_norm: 0.0,
            lr: 0.0,
            beta: 0.0,
            step_secs: 0.0,
            tokens_per_sec: 0.0,
            cuda_mem_start_used_bytes: 0,
            cuda_mem_peak_used_bytes: 0,
            cuda_mem_end_used_bytes: 0,
            cuda_mem_total_bytes: 0,
            cuda_mem_peak_delta_bytes: 0,
            cuda_mem_probe_events: Vec::new(),
            decoder_cache_snapshots: Vec::new(),
        }
    }

    /// A copy with every non-finite `f32` field replaced by a finite value:
    /// `NaN → 0`, `+∞ → f32::MAX`, `−∞ → f32::MIN`.
    ///
    /// JSON has no literal for non-finite floats — `serde_json` emits `null`,
    /// which then fails to deserialize back into an `f32`. [`MetricsWriter::append`]
    /// applies this automatically so `metrics.jsonl` stays valid, round-trippable
    /// JSON; a value that blew up (e.g. an exploded `grad_norm`, or a `kl` from an
    /// overflowed [`crate::k3_kl`]) surfaces as a saturated magnitude rather than a
    /// parse failure.
    #[must_use]
    pub fn nan_to_num(&self) -> Self {
        fn finite(x: f32) -> f32 {
            if x.is_finite() {
                x
            } else if x.is_nan() {
                0.0
            } else if x.is_sign_positive() {
                f32::MAX
            } else {
                f32::MIN
            }
        }
        let mut m = self.clone();
        for f in [
            &mut m.reward_mean,
            &mut m.reward_std,
            &mut m.frac_reward_zero_std,
            &mut m.kl,
            &mut m.clip_ratio,
            &mut m.frac_truncated,
            &mut m.completion_len,
            &mut m.rollout_ratio_mean,
            &mut m.rollout_logratio_mean,
            &mut m.rollout_ratio_max,
            &mut m.frac_rollout_ratio_capped,
            &mut m.grad_norm,
            &mut m.lr,
            &mut m.beta,
            &mut m.step_secs,
            &mut m.tokens_per_sec,
        ] {
            *f = finite(*f);
        }
        m
    }
}

/// Appends [`Metrics`] records to a JSON Lines file, one object per line.
///
/// The file is opened in append mode, so re-opening an existing run's metrics
/// file continues the stream rather than truncating it.
#[derive(Debug)]
pub struct MetricsWriter {
    path: PathBuf,
    file: File,
    #[cfg(test)]
    fail_next_append: bool,
    #[cfg(test)]
    fail_next_truncate: bool,
    #[cfg(test)]
    panic_next_truncate: bool,
}

/// One sampled completion persisted to `candidates.jsonl`.
///
/// The ledger is intentionally raw: it records the decoded completion text exactly
/// as the reward saw it, plus enough position/provenance to feed a later
/// artifact-extraction command without reconstructing candidates from aggregate
/// metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CandidateRecord {
    /// Global optimizer step (0-based).
    pub step: u64,
    /// Data-parallel rank that sampled and scored this completion.
    pub rank: usize,
    /// Data-parallel world size for the run.
    pub world_size: usize,
    /// Global prompt ordinal used to seed this group.
    pub prompt_index: u64,
    /// Completion index within the prompt's sampled group.
    pub group_index: usize,
    /// Scalar reward assigned to this completion.
    pub reward: f32,
    /// Real completion length in tokens, EOS-inclusive.
    pub completion_len_tokens: usize,
    /// Optional reward-path diagnostic for this candidate.
    ///
    /// Verifier-backed tasks use this to make zero rewards explainable in
    /// `candidates.jsonl` without bloating aggregate step metrics. Missing means the
    /// reward implementation did not provide a diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_diagnostic: Option<String>,
    /// Optional structured reward-path metadata for this candidate.
    ///
    /// The trainer treats this as opaque task-owned JSON. Old candidate ledgers do
    /// not have the field; empty metadata is omitted when writing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_metadata: Option<serde_json::Value>,
    /// Decoded completion text, exactly as passed to the reward.
    pub completion: String,
}

impl CandidateRecord {
    /// A copy with a finite reward so `candidates.jsonl` stays re-readable JSON.
    #[must_use]
    pub fn nan_to_num(&self) -> Self {
        let mut out = self.clone();
        out.reward = if out.reward.is_finite() {
            out.reward
        } else if out.reward.is_nan() {
            0.0
        } else if out.reward.is_sign_positive() {
            f32::MAX
        } else {
            f32::MIN
        };
        out
    }
}

/// Appends [`CandidateRecord`]s to a JSON Lines file, one object per line.
#[derive(Debug)]
pub struct CandidateWriter {
    path: PathBuf,
    file: File,
}

impl CandidateWriter {
    /// Open (creating if absent) `path` for appending candidate records.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Io`] if the file cannot be opened.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| TelemetryError::io(&path, e))?;
        Ok(Self { path, file })
    }

    /// Append one candidate record as a JSON line and flush it.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError`] if serialization or the write/flush fails.
    pub fn append(&mut self, record: &CandidateRecord) -> Result<(), TelemetryError> {
        let mut line = serde_json::to_string(&record.nan_to_num())?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .map_err(|e| TelemetryError::io(&self.path, e))?;
        self.file
            .flush()
            .map_err(|e| TelemetryError::io(&self.path, e))
    }

    /// The path this writer appends to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl MetricsWriter {
    /// Open (creating if absent) `path` for appending metrics lines.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Io`] if the file cannot be opened.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| TelemetryError::io(&path, e))?;
        Ok(Self {
            path,
            file,
            #[cfg(test)]
            fail_next_append: false,
            #[cfg(test)]
            fail_next_truncate: false,
            #[cfg(test)]
            panic_next_truncate: false,
        })
    }

    /// Append one metrics record as a JSON line and flush it.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError`] if serialization or the write/flush fails.
    pub fn append(&mut self, metrics: &Metrics) -> Result<(), TelemetryError> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_append) {
            return Err(TelemetryError::io(
                &self.path,
                io::Error::other("injected metrics append failure"),
            ));
        }
        // Sanitize non-finite floats so the line stays valid, re-readable JSON.
        let mut line = serde_json::to_string(&metrics.nan_to_num())?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .map_err(|e| TelemetryError::io(&self.path, e))?;
        self.file
            .flush()
            .map_err(|e| TelemetryError::io(&self.path, e))
    }

    /// Return the current append boundary for a coordinated transactional write.
    pub(crate) fn append_boundary(&mut self) -> Result<u64, TelemetryError> {
        self.file
            .flush()
            .map_err(|e| TelemetryError::io(&self.path, e))?;
        self.file
            .metadata()
            .map(|metadata| metadata.len())
            .map_err(|e| TelemetryError::io(&self.path, e))
    }

    /// Roll this append-only stream back to a previously observed boundary.
    pub(crate) fn truncate_to(&mut self, len: u64) -> Result<(), TelemetryError> {
        #[cfg(test)]
        {
            assert!(
                !std::mem::take(&mut self.panic_next_truncate),
                "injected metrics truncate panic"
            );
            if std::mem::take(&mut self.fail_next_truncate) {
                return Err(TelemetryError::io(
                    &self.path,
                    io::Error::other("injected metrics truncate failure"),
                ));
            }
        }
        self.file
            .flush()
            .map_err(|e| TelemetryError::io(&self.path, e))?;
        self.file
            .set_len(len)
            .map_err(|e| TelemetryError::io(&self.path, e))?;
        self.file
            .flush()
            .map_err(|e| TelemetryError::io(&self.path, e))
    }

    #[cfg(test)]
    pub(crate) fn inject_append_failure_once(&mut self) {
        self.fail_next_append = true;
    }

    #[cfg(test)]
    pub(crate) fn inject_truncate_failure_once(&mut self) {
        self.fail_next_truncate = true;
    }

    #[cfg(test)]
    pub(crate) fn inject_truncate_panic_once(&mut self) {
        self.panic_next_truncate = true;
    }

    /// The path this writer appends to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Read a `metrics.jsonl` file back into its [`Metrics`] records — the read
/// counterpart of [`MetricsWriter`]. One record per non-blank line, in file
/// order; blank lines are skipped and a malformed line fails loud.
///
/// Used to recover a finished run's training metrics when a requeued launch
/// resumes a checkpoint already at `steps` (so it runs no new steps itself) and
/// must still evaluate the training-reward gate — never reporting success without
/// gating just because the relaunch produced no new metrics.
///
/// # Errors
///
/// Returns [`TelemetryError::Io`] if `path` cannot be read, or
/// [`TelemetryError::Deserialize`] if a line is not a valid [`Metrics`] record.
pub fn read_metrics(path: impl AsRef<Path>) -> Result<Vec<Metrics>, TelemetryError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| TelemetryError::io(path, e))?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<Metrics>(line).map_err(|source| TelemetryError::Deserialize {
                path: path.to_path_buf(),
                source,
            })
        })
        .collect()
}

/// `grad_norm` above this multiple of the run's median positive `grad_norm` is
/// flagged as a spike.
const GRAD_SPIKE_FACTOR: f32 = 8.0;
/// Runs shorter than this many steps are not checked for a reward stall (too
/// little signal to call one).
const STALL_MIN_STEPS: usize = 5;
/// Mean degenerate-group fraction at or above this, together with a flat reward
/// trend, marks a stall.
const STALL_ZERO_STD_FRAC: f32 = 0.9;
/// `|reward_trend|` at or below this counts as "the reward did not move".
const STALL_TREND_EPS: f32 = 1e-4;

/// A health flag raised by [`summarize`] over a run's metrics stream — something
/// worth an operator's eyes, not necessarily fatal.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub enum Anomaly {
    /// A metric saturated to the [`Metrics::nan_to_num`] sentinel (`f32::MAX` /
    /// `f32::MIN`) at this step — the underlying value overflowed or was
    /// non-finite (an exploded `grad_norm`, an overflowed `kl`, …).
    NonFinite {
        /// Step the saturated value was recorded at.
        step: u64,
        /// Which metric field saturated.
        field: &'static str,
    },
    /// `grad_norm` at this step exceeded `8×` (`GRAD_SPIKE_FACTOR`) the run's
    /// median positive `grad_norm` — an update spike worth investigating.
    GradSpike {
        /// Step the spike occurred at.
        step: u64,
        /// The spiking gradient norm.
        grad_norm: f32,
        /// The run's median positive `grad_norm`, for scale.
        median: f32,
    },
    /// Across the run the batch taught the policy almost nothing: the mean
    /// fraction of zero-std (degenerate) groups was near `1` and the reward trend
    /// was flat — rewards saturated, or the task is mis-scaled for the model.
    RewardStall {
        /// Mean [`Metrics::frac_reward_zero_std`] over the run.
        mean_frac_zero_std: f32,
        /// The reward trend (see [`RunSummary::reward_trend`]).
        reward_trend: f32,
    },
    /// Completions were silently dropped (all-pad rows) over the run.
    DroppedRows {
        /// Total [`Metrics::dropped_rows`] summed over the run.
        total: u32,
    },
    /// Off-policy drift telemetry was dark for the whole run (every step captured
    /// zero rollout log-probs), so the ratio fields are neutral placeholders
    /// rather than measurements and drift went unmonitored.
    TelemetryDark,
}

impl std::fmt::Display for Anomaly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { step, field } => {
                write!(f, "non-finite `{field}` saturated at step {step}")
            }
            Self::GradSpike {
                step,
                grad_norm,
                median,
            } => write!(
                f,
                "grad_norm spike {grad_norm:.3} at step {step} ({:.1}× median {median:.3})",
                grad_norm / median
            ),
            Self::RewardStall {
                mean_frac_zero_std,
                reward_trend,
            } => write!(
                f,
                "reward stalled: {:.0}% groups zero-std, trend {reward_trend:+.4}",
                mean_frac_zero_std * 100.0
            ),
            Self::DroppedRows { total } => write!(f, "{total} completion rows dropped"),
            Self::TelemetryDark => write!(f, "off-policy drift telemetry dark all run"),
        }
    }
}

/// A reduced, operator-facing health view over a run's `metrics.jsonl`, produced
/// by [`summarize`]. A **pure function of the stream** — no I/O and no clock, so
/// it is deterministic and unit-testable; the timing inputs were measured when
/// the metrics were written.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct RunSummary {
    /// Number of metrics records (optimizer steps) summarized.
    pub steps: usize,
    /// `step` of the first record.
    pub first_step: u64,
    /// `step` of the last record.
    pub last_step: u64,
    /// `reward_mean` of the first record.
    pub reward_first: f32,
    /// `reward_mean` of the last record.
    pub reward_last: f32,
    /// `reward_last − reward_first` — the net move over the run.
    pub reward_delta: f32,
    /// Mean reward over the last third minus the mean over the first third — a
    /// noise-robust trend (falls back to `reward_delta` for runs under 3 steps).
    pub reward_trend: f32,
    /// `kl` of the last record.
    pub final_kl: f32,
    /// `lr` of the last record.
    pub final_lr: f32,
    /// `beta` of the last record.
    pub final_beta: f32,
    /// `grad_norm` of the last record.
    pub final_grad_norm: f32,
    /// Largest `grad_norm` seen over the run.
    pub max_grad_norm: f32,
    /// Mean per-step wall-clock seconds (mean of [`Metrics::step_secs`]).
    pub mean_step_secs: f32,
    /// Mean per-rank rollout throughput (mean of [`Metrics::tokens_per_sec`]); the
    /// world figure is `world_size ×` this.
    pub mean_tokens_per_sec: f32,
    /// Total measured wall-clock seconds (Σ [`Metrics::step_secs`]).
    pub total_wall_secs: f32,
    /// Highest CUDA memory peak reported by any step, in bytes. `0` when the
    /// memory probe was disabled or unavailable.
    pub max_cuda_mem_peak_used_bytes: u64,
    /// Highest per-step CUDA memory delta from a step's first probe, in bytes.
    /// `0` when the memory probe was disabled or unavailable.
    pub max_cuda_mem_peak_delta_bytes: u64,
    /// Total dropped (all-pad) completion rows over the run.
    pub total_dropped_rows: u32,
    /// Health flags raised over the stream; empty means nothing notable.
    pub anomalies: Vec<Anomaly>,
}

impl std::fmt::Display for RunSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let verdict = if self.anomalies.is_empty() {
            "HEALTHY"
        } else {
            "WARN"
        };
        writeln!(
            f,
            "run summary — {verdict}  ({} steps, {}..{})",
            self.steps, self.first_step, self.last_step
        )?;
        writeln!(
            f,
            "  reward      {:.4} → {:.4}   (Δ {:+.4}, trend {:+.4})",
            self.reward_first, self.reward_last, self.reward_delta, self.reward_trend
        )?;
        writeln!(
            f,
            "  throughput  {:.1} tok/s/rank   step {:.2}s   wall {:.1}s",
            self.mean_tokens_per_sec, self.mean_step_secs, self.total_wall_secs
        )?;
        writeln!(
            f,
            "  grad_norm   final {:.3}   max {:.3}      kl {:.4}   beta {:.2e}   lr {:.2e}",
            self.final_grad_norm, self.max_grad_norm, self.final_kl, self.final_beta, self.final_lr
        )?;
        if self.max_cuda_mem_peak_used_bytes > 0 {
            writeln!(
                f,
                "  cuda_mem    peak {} MiB   step_delta {} MiB",
                bytes_to_mib(self.max_cuda_mem_peak_used_bytes),
                bytes_to_mib(self.max_cuda_mem_peak_delta_bytes)
            )?;
        }
        if self.total_dropped_rows > 0 {
            writeln!(f, "  dropped_rows {}", self.total_dropped_rows)?;
        }
        for a in &self.anomalies {
            writeln!(f, "  ! {a}")?;
        }
        Ok(())
    }
}

/// Resource and behavior budget for [`compare_metrics`].
///
/// Ratios are expressed as candidate / baseline. For example, a
/// `max_mean_step_secs_ratio` of `1.10` allows the candidate to be at most 10%
/// slower than the baseline after warmup.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RegressionBudget {
    /// Require both streams to contain at least [`Self::min_positive_grad_steps`]
    /// finite `grad_norm > 0` rows after warmup.
    pub require_live_update: bool,
    /// Require non-zero `step_secs` values and enforce
    /// [`Self::max_mean_step_secs_ratio`].
    pub require_timing: bool,
    /// Require non-zero CUDA memory probes and enforce
    /// [`Self::max_cuda_peak_used_ratio`].
    pub require_cuda_memory: bool,
    /// Permit candidate [`RunSummary`] health anomalies to differ from baseline.
    pub allow_health_warnings: bool,
    /// Number of leading aligned rows to ignore for timing/memory/grad checks.
    pub warmup_steps: usize,
    /// Minimum finite positive-gradient rows required when
    /// [`Self::require_live_update`] is true.
    pub min_positive_grad_steps: usize,
    /// Maximum allowed candidate/baseline mean `step_secs` ratio.
    pub max_mean_step_secs_ratio: f32,
    /// Absolute slack added to the mean `step_secs` limit.
    pub max_mean_step_secs_abs_slack: f32,
    /// Maximum allowed candidate/baseline peak CUDA used-memory ratio.
    pub max_cuda_peak_used_ratio: f64,
    /// Absolute slack added to the peak CUDA used-memory limit.
    pub max_cuda_peak_used_abs_slack_bytes: u64,
    /// Optional candidate/baseline peak CUDA step-delta ratio.
    pub max_cuda_peak_delta_ratio: Option<f64>,
    /// Absolute slack added to the peak CUDA step-delta limit when the delta
    /// check is enabled.
    pub max_cuda_peak_delta_abs_slack_bytes: u64,
    /// Optional relative drift bound for final `grad_norm`.
    pub max_final_grad_norm_rel_drift: Option<f32>,
}

impl Default for RegressionBudget {
    fn default() -> Self {
        Self {
            require_live_update: true,
            require_timing: true,
            require_cuda_memory: true,
            allow_health_warnings: false,
            warmup_steps: 0,
            min_positive_grad_steps: 1,
            max_mean_step_secs_ratio: 1.10,
            max_mean_step_secs_abs_slack: 0.0,
            max_cuda_peak_used_ratio: 1.0,
            max_cuda_peak_used_abs_slack_bytes: 64 * 1024 * 1024,
            max_cuda_peak_delta_ratio: None,
            max_cuda_peak_delta_abs_slack_bytes: 64 * 1024 * 1024,
            max_final_grad_norm_rel_drift: None,
        }
    }
}

/// One failed behavior/resource regression check.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub enum RegressionFailure {
    /// A metrics stream was empty.
    EmptyStream {
        /// Which stream was empty (`baseline` or `candidate`).
        stream: &'static str,
    },
    /// The candidate and baseline have different row counts.
    StepCountMismatch {
        /// Baseline metrics rows.
        baseline: usize,
        /// Candidate metrics rows.
        candidate: usize,
    },
    /// Distributed baseline and candidate worlds have different rank counts.
    RankCountMismatch {
        /// Baseline rank streams.
        baseline: usize,
        /// Candidate rank streams.
        candidate: usize,
    },
    /// Ranks within one distributed world have different row counts.
    RankStepCountMismatch {
        /// Which stream failed (`baseline` or `candidate`).
        stream: &'static str,
        /// Rank whose row count differed from rank 0.
        rank: usize,
        /// Rank-0 row count.
        expected: usize,
        /// This rank's row count.
        observed: usize,
    },
    /// Ranks within one distributed world disagree on the step at one row.
    RankStepIndexMismatch {
        /// Which stream failed (`baseline` or `candidate`).
        stream: &'static str,
        /// Row index in the metrics stream.
        row: usize,
        /// Rank whose step differed from rank 0.
        rank: usize,
        /// Rank-0 `step`.
        expected: u64,
        /// This rank's `step`.
        observed: u64,
    },
    /// The configured warmup leaves no comparable rows.
    WarmupElidesAll {
        /// Configured warmup row count.
        warmup_steps: usize,
        /// Available aligned row count.
        len: usize,
    },
    /// An aligned row has a different `step` value.
    StepIndexMismatch {
        /// Row index in the metrics stream.
        row: usize,
        /// Baseline `step`.
        baseline: u64,
        /// Candidate `step`.
        candidate: u64,
    },
    /// A stream had too few finite positive-gradient rows.
    LiveUpdateMissing {
        /// Which stream failed (`baseline` or `candidate`).
        stream: &'static str,
        /// Observed finite positive-gradient rows.
        observed: usize,
        /// Required finite positive-gradient rows.
        required: usize,
    },
    /// A stream was missing non-zero timing telemetry.
    TimingMissing {
        /// Which stream failed (`baseline` or `candidate`).
        stream: &'static str,
    },
    /// Candidate mean step time exceeded the budget.
    StepTimeRegression {
        /// Baseline mean `step_secs`.
        baseline_mean: f32,
        /// Candidate mean `step_secs`.
        candidate_mean: f32,
        /// Allowed candidate mean.
        limit: f32,
    },
    /// A stream was missing non-zero CUDA memory telemetry.
    CudaMemoryMissing {
        /// Which stream failed (`baseline` or `candidate`).
        stream: &'static str,
    },
    /// Candidate peak CUDA used-memory exceeded the budget.
    CudaPeakRegression {
        /// Baseline peak used bytes.
        baseline_peak: u64,
        /// Candidate peak used bytes.
        candidate_peak: u64,
        /// Allowed candidate peak bytes.
        limit: u64,
    },
    /// Candidate peak CUDA step-delta exceeded the optional budget.
    CudaDeltaRegression {
        /// Baseline peak delta bytes.
        baseline_delta: u64,
        /// Candidate peak delta bytes.
        candidate_delta: u64,
        /// Allowed candidate delta bytes.
        limit: u64,
    },
    /// Final gradient norm drift exceeded the optional budget.
    GradNormDrift {
        /// Baseline final `grad_norm`.
        baseline: f32,
        /// Candidate final `grad_norm`.
        candidate: f32,
        /// Observed relative drift.
        relative_drift: f32,
        /// Allowed relative drift.
        limit: f32,
    },
    /// Candidate health anomalies differ from the baseline while
    /// `allow_health_warnings` is false.
    HealthWarnings {
        /// Baseline anomaly count.
        baseline: usize,
        /// Candidate anomaly count.
        candidate: usize,
    },
    /// Candidate rank-local health anomalies differ from the baseline while
    /// `allow_health_warnings` is false.
    RankHealthWarnings {
        /// Rank whose anomaly set differs.
        rank: usize,
        /// Baseline anomaly count for the rank.
        baseline: usize,
        /// Candidate anomaly count for the rank.
        candidate: usize,
    },
    /// One stream has candidate ledgers while the other does not.
    CandidateLedgerMissing {
        /// Which stream is missing candidate ledgers (`baseline` or `candidate`).
        stream: &'static str,
    },
    /// Candidate reward diagnostics differ from the baseline while
    /// `allow_health_warnings` is false.
    CandidateDiagnostics {
        /// Baseline candidate rows with a reward diagnostic.
        baseline: usize,
        /// Candidate rows with a reward diagnostic.
        candidate: usize,
    },
}

impl std::fmt::Display for RegressionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyStream { stream } => write!(f, "{stream} metrics stream is empty"),
            Self::StepCountMismatch {
                baseline,
                candidate,
            } => write!(
                f,
                "step-count mismatch: baseline={baseline} candidate={candidate}"
            ),
            Self::RankCountMismatch {
                baseline,
                candidate,
            } => write!(
                f,
                "rank-count mismatch: baseline={baseline} candidate={candidate}"
            ),
            Self::RankStepCountMismatch {
                stream,
                rank,
                expected,
                observed,
            } => write!(
                f,
                "{stream} rank {rank} row-count mismatch: expected={expected} observed={observed}"
            ),
            Self::RankStepIndexMismatch {
                stream,
                row,
                rank,
                expected,
                observed,
            } => write!(
                f,
                "{stream} rank {rank} step-index mismatch at row {row}: \
                 expected={expected} observed={observed}"
            ),
            Self::WarmupElidesAll { warmup_steps, len } => {
                write!(
                    f,
                    "warmup_steps={warmup_steps} leaves no rows from len={len}"
                )
            }
            Self::StepIndexMismatch {
                row,
                baseline,
                candidate,
            } => write!(
                f,
                "step-index mismatch at row {row}: baseline={baseline} candidate={candidate}"
            ),
            Self::LiveUpdateMissing {
                stream,
                observed,
                required,
            } => write!(
                f,
                "{stream} has {observed} finite positive-grad rows, required {required}"
            ),
            Self::TimingMissing { stream } => write!(f, "{stream} timing telemetry is missing"),
            Self::StepTimeRegression {
                baseline_mean,
                candidate_mean,
                limit,
            } => write!(
                f,
                "mean step_secs regression: baseline={baseline_mean:.6} \
                 candidate={candidate_mean:.6} limit={limit:.6}"
            ),
            Self::CudaMemoryMissing { stream } => {
                write!(f, "{stream} CUDA memory telemetry is missing")
            }
            Self::CudaPeakRegression {
                baseline_peak,
                candidate_peak,
                limit,
            } => write!(
                f,
                "CUDA peak regression: baseline={baseline_peak} candidate={candidate_peak} \
                 limit={limit}"
            ),
            Self::CudaDeltaRegression {
                baseline_delta,
                candidate_delta,
                limit,
            } => write!(
                f,
                "CUDA delta regression: baseline={baseline_delta} candidate={candidate_delta} \
                 limit={limit}"
            ),
            Self::GradNormDrift {
                baseline,
                candidate,
                relative_drift,
                limit,
            } => write!(
                f,
                "final grad_norm drift: baseline={baseline:.9} candidate={candidate:.9} \
                 rel={relative_drift:.9} limit={limit}"
            ),
            Self::HealthWarnings {
                baseline,
                candidate,
            } => write!(
                f,
                "health warning regression: baseline={baseline} candidate={candidate}"
            ),
            Self::RankHealthWarnings {
                rank,
                baseline,
                candidate,
            } => write!(
                f,
                "rank {rank} health warning regression: baseline={baseline} candidate={candidate}"
            ),
            Self::CandidateLedgerMissing { stream } => {
                write!(f, "{stream} candidate ledger is missing")
            }
            Self::CandidateDiagnostics {
                baseline,
                candidate,
            } => write!(
                f,
                "candidate diagnostic regression: baseline={baseline} candidate={candidate}"
            ),
        }
    }
}

/// Structured result from [`compare_metrics`].
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct RegressionReport {
    /// Whether every configured check passed.
    pub passed: bool,
    /// Budget used for this comparison.
    pub budget: RegressionBudget,
    /// Baseline run summary, when the baseline stream was non-empty.
    pub baseline: Option<RunSummary>,
    /// Candidate run summary, when the candidate stream was non-empty.
    pub candidate: Option<RunSummary>,
    /// Failed checks. Empty means pass.
    pub failures: Vec<RegressionFailure>,
}

/// Compare baseline and candidate metrics streams under a behavior/resource budget.
///
/// The comparator fails closed for absent timing or CUDA memory fields when the
/// corresponding budget requirement is enabled. It also aligns raw rows by
/// position and `step`, so a candidate cannot pass by dropping or appending rows
/// that a summary average would hide.
#[must_use]
pub fn compare_metrics(
    baseline: &[Metrics],
    candidate: &[Metrics],
    budget: &RegressionBudget,
) -> RegressionReport {
    let mut failures = Vec::new();
    let baseline_summary = summarize(baseline);
    let candidate_summary = summarize(candidate);
    if baseline_summary.is_none() {
        failures.push(RegressionFailure::EmptyStream { stream: "baseline" });
    }
    if candidate_summary.is_none() {
        failures.push(RegressionFailure::EmptyStream {
            stream: "candidate",
        });
    }
    if let (Some(base), Some(cand)) = (&baseline_summary, &candidate_summary) {
        compare_health(base, cand, budget, &mut failures);
    }
    compare_aligned_rows(baseline, candidate, budget, &mut failures);
    RegressionReport {
        passed: failures.is_empty(),
        budget: budget.clone(),
        baseline: baseline_summary,
        candidate: candidate_summary,
        failures,
    }
}

/// Compare distributed baseline and candidate metric streams after aggregating
/// each world row-by-row.
///
/// This is the data-parallel counterpart to [`compare_metrics`]. It fails
/// closed when either world is empty, rank counts differ, rank streams are
/// misaligned, or a required rank-level timing/CUDA-memory/live-update probe is
/// missing. CUDA memory is aggregated as the per-step world maximum, step time
/// as the slowest rank, throughput as the world sum, and `grad_norm` remains
/// positive only when every rank reports a finite positive value for that step.
/// Rank-local health anomaly sets are compared before aggregation.
#[must_use]
pub fn compare_distributed_metrics(
    baseline_ranks: &[Vec<Metrics>],
    candidate_ranks: &[Vec<Metrics>],
    budget: &RegressionBudget,
) -> RegressionReport {
    let mut failures = Vec::new();
    if baseline_ranks.len() != candidate_ranks.len() {
        failures.push(RegressionFailure::RankCountMismatch {
            baseline: baseline_ranks.len(),
            candidate: candidate_ranks.len(),
        });
    }
    compare_rank_health(baseline_ranks, candidate_ranks, budget, &mut failures);
    let baseline = aggregate_distributed_metrics("baseline", baseline_ranks, budget, &mut failures);
    let candidate =
        aggregate_distributed_metrics("candidate", candidate_ranks, budget, &mut failures);
    let mut report = compare_metrics(&baseline, &candidate, budget);
    failures.append(&mut report.failures);
    RegressionReport {
        passed: failures.is_empty(),
        budget: report.budget,
        baseline: report.baseline,
        candidate: report.candidate,
        failures,
    }
}

fn compare_health(
    baseline: &RunSummary,
    candidate: &RunSummary,
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    if budget.allow_health_warnings {
        return;
    }
    if baseline.anomalies != candidate.anomalies {
        failures.push(RegressionFailure::HealthWarnings {
            baseline: baseline.anomalies.len(),
            candidate: candidate.anomalies.len(),
        });
    }
}

fn compare_rank_health(
    baseline_ranks: &[Vec<Metrics>],
    candidate_ranks: &[Vec<Metrics>],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    if budget.allow_health_warnings || baseline_ranks.len() != candidate_ranks.len() {
        return;
    }

    for (rank, (baseline, candidate)) in baseline_ranks.iter().zip(candidate_ranks).enumerate() {
        let Some(baseline) = summarize(baseline) else {
            continue;
        };
        let Some(candidate) = summarize(candidate) else {
            continue;
        };
        if baseline.anomalies != candidate.anomalies {
            failures.push(RegressionFailure::RankHealthWarnings {
                rank,
                baseline: baseline.anomalies.len(),
                candidate: candidate.anomalies.len(),
            });
        }
    }
}

fn aggregate_distributed_metrics(
    stream: &'static str,
    ranks: &[Vec<Metrics>],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) -> Vec<Metrics> {
    let Some(len) = validate_distributed_rank_shape(stream, ranks, failures) else {
        return Vec::new();
    };
    check_required_rank_telemetry(stream, ranks, budget, failures);

    let mut aggregated = Vec::with_capacity(len);
    for row in 0..len {
        validate_distributed_step_index(stream, ranks, row, failures);
        aggregated.push(aggregate_world_row(ranks, row));
    }
    aggregated
}

fn validate_distributed_rank_shape(
    stream: &'static str,
    ranks: &[Vec<Metrics>],
    failures: &mut Vec<RegressionFailure>,
) -> Option<usize> {
    let Some(first) = ranks.first() else {
        failures.push(RegressionFailure::EmptyStream { stream });
        return None;
    };
    if first.is_empty() {
        failures.push(RegressionFailure::EmptyStream { stream });
        return None;
    }
    let expected_len = first.len();
    let len = ranks
        .iter()
        .enumerate()
        .map(|(rank, rows)| {
            if rows.len() != expected_len {
                failures.push(RegressionFailure::RankStepCountMismatch {
                    stream,
                    rank,
                    expected: expected_len,
                    observed: rows.len(),
                });
            }
            rows.len()
        })
        .min()
        .unwrap_or(0);
    if len == 0 {
        failures.push(RegressionFailure::EmptyStream { stream });
        None
    } else {
        Some(len)
    }
}

fn check_required_rank_telemetry(
    stream: &'static str,
    ranks: &[Vec<Metrics>],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    if budget.require_timing
        && ranks
            .iter()
            .flat_map(|rows| rows.iter())
            .any(|m| !m.step_secs.is_finite() || m.step_secs <= 0.0)
    {
        failures.push(RegressionFailure::TimingMissing { stream });
    }
    if budget.require_cuda_memory
        && ranks
            .iter()
            .flat_map(|rows| rows.iter())
            .any(|m| m.cuda_mem_peak_used_bytes == 0)
    {
        failures.push(RegressionFailure::CudaMemoryMissing { stream });
    }
    if budget.max_cuda_peak_delta_ratio.is_some()
        && ranks
            .iter()
            .flat_map(|rows| rows.iter())
            .any(|m| m.cuda_mem_peak_delta_bytes == 0)
    {
        failures.push(RegressionFailure::CudaMemoryMissing { stream });
    }
}

fn validate_distributed_step_index(
    stream: &'static str,
    ranks: &[Vec<Metrics>],
    row: usize,
    failures: &mut Vec<RegressionFailure>,
) {
    let expected_step = ranks[0][row].step;
    for (rank, rows) in ranks.iter().enumerate().skip(1) {
        let observed = rows[row].step;
        if observed != expected_step {
            failures.push(RegressionFailure::RankStepIndexMismatch {
                stream,
                row,
                rank,
                expected: expected_step,
                observed,
            });
        }
    }
}

fn aggregate_world_row(ranks: &[Vec<Metrics>], row: usize) -> Metrics {
    let mut out = ranks[0][row].clone();
    let rank_count = ranks.len() as f32;
    let rows = ranks.iter().map(|rank| &rank[row]);

    out.reward_mean = mean_f32(rows.clone().map(|m| m.reward_mean));
    out.reward_std = max_f32(rows.clone().map(|m| m.reward_std));
    out.frac_reward_zero_std = max_f32(rows.clone().map(|m| m.frac_reward_zero_std));
    out.kl = mean_f32(rows.clone().map(|m| m.kl));
    out.clip_ratio = mean_f32(rows.clone().map(|m| m.clip_ratio));
    out.frac_truncated = mean_f32(rows.clone().map(|m| m.frac_truncated));
    out.completion_len = mean_f32(rows.clone().map(|m| m.completion_len));
    out.rollout_ratio_mean = mean_f32(rows.clone().map(|m| m.rollout_ratio_mean));
    out.rollout_logratio_mean = mean_f32(rows.clone().map(|m| m.rollout_logratio_mean));
    out.rollout_ratio_max = max_f32(rows.clone().map(|m| m.rollout_ratio_max));
    out.frac_rollout_ratio_capped = mean_f32(rows.clone().map(|m| m.frac_rollout_ratio_capped));
    out.rollout_capture_tokens = rows.clone().map(|m| m.rollout_capture_tokens).sum();
    out.dropped_rows = rows.clone().map(|m| m.dropped_rows).sum();
    out.grad_norm = if rows
        .clone()
        .all(|m| m.grad_norm.is_finite() && m.grad_norm > 0.0)
    {
        rows.clone().map(|m| m.grad_norm).sum::<f32>() / rank_count
    } else {
        0.0
    };
    out.lr = mean_f32(rows.clone().map(|m| m.lr));
    out.step_secs = max_f32(rows.clone().map(|m| m.step_secs));
    out.tokens_per_sec = rows.clone().map(|m| m.tokens_per_sec).sum();
    out.cuda_mem_start_used_bytes = rows
        .clone()
        .map(|m| m.cuda_mem_start_used_bytes)
        .max()
        .unwrap_or(0);
    out.cuda_mem_peak_used_bytes = rows
        .clone()
        .map(|m| m.cuda_mem_peak_used_bytes)
        .max()
        .unwrap_or(0);
    out.cuda_mem_end_used_bytes = rows
        .clone()
        .map(|m| m.cuda_mem_end_used_bytes)
        .max()
        .unwrap_or(0);
    out.cuda_mem_total_bytes = rows
        .clone()
        .map(|m| m.cuda_mem_total_bytes)
        .min()
        .unwrap_or(0);
    out.cuda_mem_peak_delta_bytes = rows.map(|m| m.cuda_mem_peak_delta_bytes).max().unwrap_or(0);
    out
}

fn compare_aligned_rows(
    baseline: &[Metrics],
    candidate: &[Metrics],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    if baseline.len() != candidate.len() {
        failures.push(RegressionFailure::StepCountMismatch {
            baseline: baseline.len(),
            candidate: candidate.len(),
        });
    }
    let len = baseline.len().min(candidate.len());
    if len == 0 {
        return;
    }
    for i in 0..len {
        if baseline[i].step != candidate[i].step {
            failures.push(RegressionFailure::StepIndexMismatch {
                row: i,
                baseline: baseline[i].step,
                candidate: candidate[i].step,
            });
        }
    }
    if budget.warmup_steps >= len {
        failures.push(RegressionFailure::WarmupElidesAll {
            warmup_steps: budget.warmup_steps,
            len,
        });
        return;
    }
    let base = &baseline[budget.warmup_steps..len];
    let cand = &candidate[budget.warmup_steps..len];
    compare_live_update(base, cand, budget, failures);
    compare_timing(base, cand, budget, failures);
    compare_cuda_memory(base, cand, budget, failures);
    compare_grad_norm(base, cand, budget, failures);
}

fn compare_live_update(
    baseline: &[Metrics],
    candidate: &[Metrics],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    if !budget.require_live_update {
        return;
    }
    let required = budget.min_positive_grad_steps;
    for (stream, observed) in [
        ("baseline", positive_grad_steps(baseline)),
        ("candidate", positive_grad_steps(candidate)),
    ] {
        if observed < required {
            failures.push(RegressionFailure::LiveUpdateMissing {
                stream,
                observed,
                required,
            });
        }
    }
}

fn compare_timing(
    baseline: &[Metrics],
    candidate: &[Metrics],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    if !budget.require_timing {
        return;
    }
    let Some(base_mean) = required_positive_mean(baseline, |m| m.step_secs) else {
        failures.push(RegressionFailure::TimingMissing { stream: "baseline" });
        return;
    };
    let Some(cand_mean) = required_positive_mean(candidate, |m| m.step_secs) else {
        failures.push(RegressionFailure::TimingMissing {
            stream: "candidate",
        });
        return;
    };
    let limit = base_mean * budget.max_mean_step_secs_ratio + budget.max_mean_step_secs_abs_slack;
    if cand_mean > limit {
        failures.push(RegressionFailure::StepTimeRegression {
            baseline_mean: base_mean,
            candidate_mean: cand_mean,
            limit,
        });
    }
}

fn compare_cuda_memory(
    baseline: &[Metrics],
    candidate: &[Metrics],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    if !budget.require_cuda_memory {
        return;
    }
    let Some(base_peak) = required_peak(baseline, |m| m.cuda_mem_peak_used_bytes) else {
        failures.push(RegressionFailure::CudaMemoryMissing { stream: "baseline" });
        return;
    };
    let Some(cand_peak) = required_peak(candidate, |m| m.cuda_mem_peak_used_bytes) else {
        failures.push(RegressionFailure::CudaMemoryMissing {
            stream: "candidate",
        });
        return;
    };
    let limit = allowed_bytes(
        base_peak,
        budget.max_cuda_peak_used_ratio,
        budget.max_cuda_peak_used_abs_slack_bytes,
    );
    if cand_peak > limit {
        failures.push(RegressionFailure::CudaPeakRegression {
            baseline_peak: base_peak,
            candidate_peak: cand_peak,
            limit,
        });
    }
    compare_cuda_delta(baseline, candidate, budget, failures);
}

fn compare_cuda_delta(
    baseline: &[Metrics],
    candidate: &[Metrics],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    let Some(ratio) = budget.max_cuda_peak_delta_ratio else {
        return;
    };
    let Some(base_delta) = required_peak(baseline, |m| m.cuda_mem_peak_delta_bytes) else {
        failures.push(RegressionFailure::CudaMemoryMissing { stream: "baseline" });
        return;
    };
    let Some(cand_delta) = required_peak(candidate, |m| m.cuda_mem_peak_delta_bytes) else {
        failures.push(RegressionFailure::CudaMemoryMissing {
            stream: "candidate",
        });
        return;
    };
    let limit = allowed_bytes(
        base_delta,
        ratio,
        budget.max_cuda_peak_delta_abs_slack_bytes,
    );
    if cand_delta > limit {
        failures.push(RegressionFailure::CudaDeltaRegression {
            baseline_delta: base_delta,
            candidate_delta: cand_delta,
            limit,
        });
    }
}

fn compare_grad_norm(
    baseline: &[Metrics],
    candidate: &[Metrics],
    budget: &RegressionBudget,
    failures: &mut Vec<RegressionFailure>,
) {
    let Some(limit) = budget.max_final_grad_norm_rel_drift else {
        return;
    };
    let (Some(base), Some(cand)) = (baseline.last(), candidate.last()) else {
        return;
    };
    let drift = relative_drift(base.grad_norm, cand.grad_norm);
    if !base.grad_norm.is_finite() || !cand.grad_norm.is_finite() || drift > limit {
        failures.push(RegressionFailure::GradNormDrift {
            baseline: base.grad_norm,
            candidate: cand.grad_norm,
            relative_drift: drift,
            limit,
        });
    }
}

fn positive_grad_steps(history: &[Metrics]) -> usize {
    history
        .iter()
        .filter(|m| m.grad_norm.is_finite() && m.grad_norm > 0.0)
        .count()
}

fn required_positive_mean(history: &[Metrics], f: impl Fn(&Metrics) -> f32) -> Option<f32> {
    if history.is_empty()
        || history.iter().any(|m| {
            let value = f(m);
            !value.is_finite() || value <= 0.0
        })
    {
        return None;
    }
    Some(mean_of(history, f))
}

fn required_peak(history: &[Metrics], f: impl Fn(&Metrics) -> u64) -> Option<u64> {
    if history.is_empty() || history.iter().any(|m| f(m) == 0) {
        return None;
    }
    let peak = history.iter().map(f).max()?;
    (peak > 0).then_some(peak)
}

fn allowed_bytes(baseline: u64, ratio: f64, slack: u64) -> u64 {
    ((baseline as f64) * ratio).ceil() as u64 + slack
}

fn mean_f32(values: impl IntoIterator<Item = f32>) -> f32 {
    let mut count = 0_usize;
    let mut sum = 0.0_f32;
    for value in values {
        count += 1;
        sum += value;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn max_f32(values: impl IntoIterator<Item = f32>) -> f32 {
    values.into_iter().reduce(f32::max).unwrap_or(0.0)
}

fn relative_drift(baseline: f32, candidate: f32) -> f32 {
    let denom = baseline.abs().max(1e-12);
    (candidate - baseline).abs() / denom
}

/// Reduce a run's [`Metrics`] stream into a [`RunSummary`] — reward trend,
/// throughput, gradient health, and any [`Anomaly`] flags. **Pure**: no I/O, no
/// clock (the timing was measured when the metrics were written), so it is fully
/// deterministic and unit-testable.
///
/// Returns `None` for an empty stream (nothing to summarize). Pair with
/// [`read_metrics`] to summarize a finished run's `metrics.jsonl`.
#[must_use]
pub fn summarize(history: &[Metrics]) -> Option<RunSummary> {
    let first = history.first()?;
    let last = history.last()?;
    let trend = reward_trend(history);
    let max_grad_norm = history.iter().map(|m| m.grad_norm).fold(0.0_f32, f32::max);
    Some(RunSummary {
        steps: history.len(),
        first_step: first.step,
        last_step: last.step,
        reward_first: first.reward_mean,
        reward_last: last.reward_mean,
        reward_delta: last.reward_mean - first.reward_mean,
        reward_trend: trend,
        final_kl: last.kl,
        final_lr: last.lr,
        final_beta: last.beta,
        final_grad_norm: last.grad_norm,
        max_grad_norm,
        mean_step_secs: mean_of(history, |m| m.step_secs),
        mean_tokens_per_sec: mean_of(history, |m| m.tokens_per_sec),
        total_wall_secs: history.iter().map(|m| m.step_secs).sum(),
        max_cuda_mem_peak_used_bytes: history
            .iter()
            .map(|m| m.cuda_mem_peak_used_bytes)
            .max()
            .unwrap_or(0),
        max_cuda_mem_peak_delta_bytes: history
            .iter()
            .map(|m| m.cuda_mem_peak_delta_bytes)
            .max()
            .unwrap_or(0),
        total_dropped_rows: history.iter().map(|m| m.dropped_rows).sum(),
        anomalies: detect_anomalies(history, trend),
    })
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

/// Mean of `f` over the records, or `0.0` for an empty slice.
fn mean_of(history: &[Metrics], f: impl Fn(&Metrics) -> f32) -> f32 {
    if history.is_empty() {
        return 0.0;
    }
    history.iter().map(f).sum::<f32>() / history.len() as f32
}

/// Reward trend: mean of the last third minus the first third (noise-robust),
/// falling back to `last − first` for short runs (< 3 steps).
fn reward_trend(history: &[Metrics]) -> f32 {
    let n = history.len();
    let (Some(first), Some(last)) = (history.first(), history.last()) else {
        return 0.0;
    };
    if n < 3 {
        return last.reward_mean - first.reward_mean;
    }
    let third = n / 3;
    mean_of(&history[n - third..], |m| m.reward_mean)
        - mean_of(&history[..third], |m| m.reward_mean)
}

/// Run every anomaly check over the stream, collecting the flags raised.
fn detect_anomalies(history: &[Metrics], trend: f32) -> Vec<Anomaly> {
    let mut out = Vec::new();
    push_nonfinite(history, &mut out);
    push_grad_spike(history, &mut out);
    push_reward_stall(history, trend, &mut out);
    push_dropped_rows(history, &mut out);
    push_telemetry_dark(history, &mut out);
    out
}

/// Flag any metric that saturated to the `nan_to_num` sentinel (`f32::MAX`/`MIN`).
fn push_nonfinite(history: &[Metrics], out: &mut Vec<Anomaly>) {
    for m in history {
        for (field, v) in [
            ("grad_norm", m.grad_norm),
            ("kl", m.kl),
            ("beta", m.beta),
            ("reward_mean", m.reward_mean),
        ] {
            if v == f32::MAX || v == f32::MIN {
                out.push(Anomaly::NonFinite {
                    step: m.step,
                    field,
                });
            }
        }
    }
}

/// Flag the worst step whose `grad_norm` exceeds `GRAD_SPIKE_FACTOR ×` the run's
/// median positive `grad_norm`.
fn push_grad_spike(history: &[Metrics], out: &mut Vec<Anomaly>) {
    let median = median_positive_grad_norm(history);
    if median <= 0.0 {
        return;
    }
    let Some(worst) = history
        .iter()
        .max_by(|a, b| a.grad_norm.total_cmp(&b.grad_norm))
    else {
        return;
    };
    if worst.grad_norm > GRAD_SPIKE_FACTOR * median {
        out.push(Anomaly::GradSpike {
            step: worst.step,
            grad_norm: worst.grad_norm,
            median,
        });
    }
}

/// Median of the strictly-positive `grad_norm`s (warmup zero-grad steps excluded),
/// or `0.0` if none — the scale a spike is measured against.
fn median_positive_grad_norm(history: &[Metrics]) -> f32 {
    let mut v: Vec<f32> = history
        .iter()
        .map(|m| m.grad_norm)
        .filter(|x| *x > 0.0)
        .collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Flag a run that taught ~nothing: near-all-degenerate groups and a flat reward.
fn push_reward_stall(history: &[Metrics], trend: f32, out: &mut Vec<Anomaly>) {
    if history.len() < STALL_MIN_STEPS {
        return;
    }
    let mean_zero = mean_of(history, |m| m.frac_reward_zero_std);
    if mean_zero >= STALL_ZERO_STD_FRAC && trend.abs() <= STALL_TREND_EPS {
        out.push(Anomaly::RewardStall {
            mean_frac_zero_std: mean_zero,
            reward_trend: trend,
        });
    }
}

/// Flag any silently-dropped (all-pad) completion rows over the run.
fn push_dropped_rows(history: &[Metrics], out: &mut Vec<Anomaly>) {
    let total: u32 = history.iter().map(|m| m.dropped_rows).sum();
    if total > 0 {
        out.push(Anomaly::DroppedRows { total });
    }
}

/// Flag a run with no off-policy drift telemetry at all (every step captured zero).
fn push_telemetry_dark(history: &[Metrics], out: &mut Vec<Anomaly>) {
    if !history.is_empty() && history.iter().all(|m| m.rollout_capture_tokens == 0) {
        out.push(Anomaly::TelemetryDark);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            p.push(format!("ferrl-test-{tag}-{nanos}"));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn init_tracing_is_idempotent() {
        // Should not panic even when called twice (try_init swallows the second).
        init_tracing().unwrap();
        init_tracing().unwrap();
    }

    /// A `MakeWriter` that captures formatted log output into a shared buffer, so a test
    /// can assert on what the subscriber actually rendered.
    #[derive(Clone)]
    struct CaptureWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` under a thread-local fmt subscriber (default `info` filter) that captures
    /// rendered output into a buffer (returned).
    fn capture_with_default(f: impl FnOnce()) -> std::sync::Arc<std::sync::Mutex<Vec<u8>>> {
        capture_with_filter("info", f)
    }

    /// Run `f` under a thread-local fmt subscriber whose `EnvFilter` is `directive`,
    /// capturing rendered output into a buffer (returned). Installs the global subscriber
    /// first (idempotent) so the span/event callsites register as dynamic and the
    /// thread-local capture reliably receives them regardless of test order — without a
    /// real global default, a sibling test can register a callsite against the no-op
    /// subscriber and cache it disabled.
    fn capture_with_filter(
        directive: &str,
        f: impl FnOnce(),
    ) -> std::sync::Arc<std::sync::Mutex<Vec<u8>>> {
        let _ = init_tracing();
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let sub = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::new(directive))
            .with_writer(CaptureWriter(std::sync::Arc::clone(&buf)))
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(sub, f);
        buf
    }

    #[test]
    fn run_span_stamps_rank_and_world_on_events() {
        // The contract: every event emitted while the span is entered inherits this
        // rank's rank/world. Capture the rendered output and assert both appear.
        let buf = capture_with_default(|| {
            let _run = run_span(3, 8).entered();
            tracing::info!("captured event");
        });
        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(out.contains("rank=3"), "rank not stamped: {out}");
        assert!(out.contains("world=8"), "world not stamped: {out}");
        assert!(out.contains("captured event"), "event missing: {out}");
    }

    #[test]
    fn run_and_step_spans_stamp_rank_world_and_step() {
        // The trainer's exact construction: a run span (rank/world) wrapping a nested
        // per-step span (step). An event emitted within carries all three fields.
        let buf = capture_with_default(|| {
            let _run = run_span(0, 1).entered();
            let _step = step_span(2).entered();
            tracing::info!("step-scoped event");
        });
        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(out.contains("rank=0"), "rank not stamped: {out}");
        assert!(out.contains("world=1"), "world not stamped: {out}");
        assert!(out.contains("step=2"), "step not stamped: {out}");
    }

    #[test]
    fn warn_events_stay_stamped_under_a_warn_level_filter() {
        // The regression guard for the ERROR-level span choice: an operator quieting a
        // run with RUST_LOG=warn filters out info-level spans, but the spans are ERROR
        // level so they stay enabled — a WARN event (e.g. the preemption warn) must
        // still carry rank/world/step. An info-level span would drop the fields here.
        let buf = capture_with_filter("warn", || {
            let _run = run_span(5, 9).entered();
            let _step = step_span(3).entered();
            tracing::warn!("a warning under a warn-level filter");
        });
        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(out.contains("rank=5"), "rank lost under warn filter: {out}");
        assert!(
            out.contains("world=9"),
            "world lost under warn filter: {out}"
        );
        assert!(out.contains("step=3"), "step lost under warn filter: {out}");
    }

    #[test]
    fn rundir_creates_layout_and_paths() {
        let tmp = TempDir::new("rundir");
        let rd = RunDir::create(tmp.path(), "run-001").unwrap();
        assert_eq!(rd.run_id(), "run-001");
        assert!(rd.root().is_dir());
        assert!(rd.checkpoints_dir().is_dir());
        assert!(rd.metrics_path().ends_with("metrics.jsonl"));
        assert!(rd.config_path().ends_with("config.json"));
        assert!(rd.log_path().ends_with("run.log"));
    }

    #[test]
    fn rundir_create_rejects_a_duplicate_run_id() {
        // A second create() at the same run_id would append a fresh run into the
        // prior run's metrics.jsonl — the guard turns that into a loud error.
        let tmp = TempDir::new("dup");
        let rd = RunDir::create(tmp.path(), "run-001").unwrap();
        let mut w = rd.metrics_writer().unwrap();
        w.append(&Metrics::at_step(0)).unwrap();
        drop(w);
        let err = RunDir::create(tmp.path(), "run-001").unwrap_err();
        assert!(
            matches!(err, TelemetryError::DuplicateRun { .. }),
            "got {err:?}"
        );
        // A run dir with no metrics stream yet (created, never trained) is fine
        // to re-create — only an existing stream is guarded.
        RunDir::create(tmp.path(), "run-002").unwrap();
        RunDir::create(tmp.path(), "run-002").unwrap();
    }

    #[test]
    fn rundir_create_rejects_stale_candidate_ledger_without_metrics() {
        // Candidates are written during sample collection before step metrics.
        // If the process dies in that window, a fresh run with the same run_id
        // must not append indistinguishable coordinates into the stale ledger.
        let tmp = TempDir::new("dup-candidates");
        let rd = RunDir::create(tmp.path(), "run-001").unwrap();
        let mut w = rd.candidate_writer().unwrap();
        w.append(&CandidateRecord {
            step: 0,
            rank: 0,
            world_size: 1,
            prompt_index: 0,
            group_index: 0,
            reward: 1.0,
            completion_len_tokens: 2,
            reward_diagnostic: Some("test:fixture".to_string()),
            reward_metadata: Some(serde_json::json!({ "fixture": true })),
            completion: "candidate".to_string(),
        })
        .unwrap();
        drop(w);
        assert!(
            !rd.metrics_path().exists(),
            "regression must cover crash-before-metrics"
        );

        let err = RunDir::create(tmp.path(), "run-001").unwrap_err();
        match err {
            TelemetryError::DuplicateRun { path } => {
                assert_eq!(path, rd.candidates_path());
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn read_metrics_round_trips_a_written_stream() {
        // read_metrics is the inverse of MetricsWriter::append: writing N records and
        // reading them back yields the same records, in order. append() sanitizes via
        // nan_to_num, so compare against the sanitized originals. This recovery path
        // is what lets a requeued, already-trained run still evaluate its gate.
        let tmp = TempDir::new("read-metrics");
        let rd = RunDir::create(tmp.path(), "run-rt").unwrap();
        let mut w = rd.metrics_writer().unwrap();
        let written: Vec<Metrics> = (0..3).map(Metrics::at_step).collect();
        for m in &written {
            w.append(m).unwrap();
        }
        drop(w);
        let read = read_metrics(rd.metrics_path()).unwrap();
        let expected: Vec<Metrics> = written.iter().map(Metrics::nan_to_num).collect();
        assert_eq!(
            read, expected,
            "read_metrics must reproduce the written records in order"
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // compact JSONL regression over field omission/presence
    fn candidate_writer_appends_jsonl_records() {
        let tmp = TempDir::new("candidates");
        let rd = RunDir::create(tmp.path(), "run-candidates").unwrap();
        let mut w = rd.candidate_writer().unwrap();
        let rec = CandidateRecord {
            step: 3,
            rank: 0,
            world_size: 1,
            prompt_index: 12,
            group_index: 1,
            reward: f32::NAN,
            completion_len_tokens: 42,
            reward_diagnostic: Some("trimul:no_submission".to_string()),
            reward_metadata: Some(serde_json::json!({ "task": "trimul" })),
            completion: "```python\ndef custom_kernel(data):\n    return data\n```".to_string(),
        };
        w.append(&rec).unwrap();
        drop(w);

        let raw = std::fs::read_to_string(rd.candidates_path()).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: CandidateRecord = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.step, 3);
        assert_eq!(parsed.group_index, 1);
        assert_eq!(parsed.reward, 0.0);
        assert_eq!(
            parsed.reward_diagnostic.as_deref(),
            Some("trimul:no_submission")
        );
        assert_eq!(
            parsed.reward_metadata.as_ref().and_then(|m| m.get("task")),
            Some(&serde_json::json!("trimul"))
        );
        assert!(parsed.completion.contains("custom_kernel"));
    }

    #[test]
    fn candidate_record_reward_diagnostics_are_jsonl_compatible() {
        let old_row = r#"{"step":3,"rank":0,"world_size":1,"prompt_index":12,"group_index":1,"reward":0.0,"completion_len_tokens":42,"completion":"old"}"#;
        let parsed: CandidateRecord = serde_json::from_str(old_row).unwrap();
        assert_eq!(parsed.reward_diagnostic, None);
        assert_eq!(parsed.reward_metadata, None);

        let tmp = TempDir::new("candidates-compat");
        let rd = RunDir::create(tmp.path(), "run-candidates-compat").unwrap();
        let mut w = rd.candidate_writer().unwrap();
        let rec = CandidateRecord {
            step: 3,
            rank: 0,
            world_size: 1,
            prompt_index: 12,
            group_index: 1,
            reward: 0.0,
            completion_len_tokens: 42,
            reward_diagnostic: None,
            reward_metadata: None,
            completion: "new".to_string(),
        };
        w.append(&rec).unwrap();
        drop(w);

        let raw = std::fs::read_to_string(rd.candidates_path()).unwrap();
        assert!(!raw.contains("reward_diagnostic"));
        assert!(!raw.contains("reward_metadata"));
        let written: CandidateRecord = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(written.reward_diagnostic, None);
        assert_eq!(written.reward_metadata, None);
    }

    #[test]
    fn read_metrics_skips_blank_lines_and_fails_loud_on_garbage() {
        let tmp = TempDir::new("read-metrics-bad");
        let good = serde_json::to_string(&Metrics::at_step(0)).unwrap();
        let path = tmp.path().join("m.jsonl");
        // Blank lines (e.g. a trailing newline) are skipped, not parsed.
        std::fs::write(&path, format!("{good}\n\n")).unwrap();
        assert_eq!(
            read_metrics(&path).unwrap().len(),
            1,
            "blank lines must be skipped"
        );
        // A malformed line fails loud (the Deserialize variant), never a silent skip.
        std::fs::write(&path, format!("{good}\nnot json\n")).unwrap();
        let err = read_metrics(&path).unwrap_err();
        assert!(
            matches!(err, TelemetryError::Deserialize { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn rundir_open_continues_an_existing_run_and_rejects_a_missing_one() {
        let tmp = TempDir::new("open");
        let rd = RunDir::create(tmp.path(), "run-001").unwrap();
        let mut w = rd.metrics_writer().unwrap();
        w.append(&Metrics::at_step(0)).unwrap();
        drop(w);

        // open() reopens the same layout; its writer appends to the stream.
        let reopened = RunDir::open(tmp.path(), "run-001").unwrap();
        assert_eq!(reopened.root(), rd.root());
        let mut w = reopened.metrics_writer().unwrap();
        w.append(&Metrics::at_step(1)).unwrap();
        drop(w);
        let raw = std::fs::read_to_string(reopened.metrics_path()).unwrap();
        assert_eq!(raw.lines().count(), 2, "open() must continue the stream");

        // A typo'd run_id fails loud rather than starting an empty "resume".
        let err = RunDir::open(tmp.path(), "run-nope").unwrap_err();
        assert!(matches!(err, TelemetryError::Io { .. }), "got {err:?}");
    }

    #[test]
    fn write_config_roundtrips() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct Cfg {
            lr: f32,
            group_size: usize,
        }
        let tmp = TempDir::new("config");
        let rd = RunDir::create(tmp.path(), "r").unwrap();
        let cfg = Cfg {
            lr: 1e-5,
            group_size: 8,
        };
        rd.write_config(&cfg).unwrap();
        let raw = std::fs::read_to_string(rd.config_path()).unwrap();
        let back: Cfg = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn metrics_writer_appends_jsonl() {
        let tmp = TempDir::new("metrics");
        let rd = RunDir::create(tmp.path(), "r").unwrap();
        let mut w = rd.metrics_writer().unwrap();

        let mut m0 = Metrics::at_step(0);
        m0.reward_mean = 1.5;
        m0.kl = 0.01;
        w.append(&m0).unwrap();

        let mut m1 = Metrics::at_step(1);
        m1.reward_mean = 2.0;
        w.append(&m1).unwrap();
        drop(w);

        let raw = std::fs::read_to_string(rd.metrics_path()).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 2);
        let p0: Metrics = serde_json::from_str(lines[0]).unwrap();
        let p1: Metrics = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(p0.reward_mean, 1.5);
        assert_eq!(p0.step, 0);
        assert_eq!(p1.step, 1);
    }

    #[test]
    fn metrics_writer_appends_across_reopen() {
        let tmp = TempDir::new("reopen");
        let path = tmp.path().join("metrics.jsonl");
        {
            let mut w = MetricsWriter::open(&path).unwrap();
            assert_eq!(w.path(), path);
            w.append(&Metrics::at_step(0)).unwrap();
        }
        {
            let mut w = MetricsWriter::open(&path).unwrap();
            w.append(&Metrics::at_step(1)).unwrap();
        }
        let raw = std::fs::read_to_string(&path).unwrap();
        assert_eq!(raw.lines().count(), 2);
    }

    #[test]
    fn metrics_serde_roundtrip_exact() {
        let m = Metrics {
            step: 7,
            reward_mean: 0.5,
            reward_std: 0.25,
            frac_reward_zero_std: 0.125,
            kl: 0.02,
            clip_ratio: 0.1,
            frac_truncated: 0.0625,
            completion_len: 42.0,
            rollout_ratio_mean: 1.015625,
            rollout_logratio_mean: -0.0625,
            rollout_ratio_max: 2.5,
            frac_rollout_ratio_capped: 0.03125,
            rollout_capture_tokens: 96,
            dropped_rows: 3,
            grad_norm: 1.23,
            lr: 5e-6,
            beta: 0.02,
            step_secs: 1.5,
            tokens_per_sec: 128.0,
            cuda_mem_start_used_bytes: 10,
            cuda_mem_peak_used_bytes: 30,
            cuda_mem_end_used_bytes: 20,
            cuda_mem_total_bytes: 100,
            cuda_mem_peak_delta_bytes: 20,
            cuda_mem_probe_events: vec![GpuMemoryProbeEvent {
                phase: "rollout_prefill_end".to_string(),
                used_bytes: 30,
                free_bytes: 70,
                total_bytes: 100,
                peak_delta_bytes: 20,
            }],
            decoder_cache_snapshots: vec![DecoderCacheSnapshot {
                phase: "rollout_decode_end".to_string(),
                layer_index: 1,
                kind: "sliding_attention".to_string(),
                seen_tokens: 12,
                retained_tokens: 3,
                max_retained_tokens: Some(3),
            }],
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: Metrics = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn metrics_deserializes_old_log_without_new_fields() {
        // A pre-PR#3 metrics.jsonl line lacks frac_reward_zero_std and dropped_rows;
        // #[serde(default)] must let it deserialize (defaulting both to 0). The R2
        // rollout-ratio fields default to their NEUTRAL values: ratios to 1.0
        // (on-policy — what an old record implicitly assumed), the capped
        // fraction to 0.
        let old = r#"{"step":3,"reward_mean":1.0,"reward_std":0.5,"kl":0.01,"clip_ratio":0.2,"completion_len":10.0,"grad_norm":1.0,"lr":1e-5}"#;
        let m: Metrics = serde_json::from_str(old).unwrap();
        assert_eq!(
            (m.step, m.frac_reward_zero_std, m.dropped_rows),
            (3, 0.0, 0)
        );
        // The PR-4 timing fields are absent from an old record → default to 0.0.
        assert_eq!((m.step_secs, m.tokens_per_sec), (0.0, 0.0));
        assert_eq!(
            (
                m.cuda_mem_start_used_bytes,
                m.cuda_mem_peak_used_bytes,
                m.cuda_mem_end_used_bytes,
                m.cuda_mem_total_bytes,
                m.cuda_mem_peak_delta_bytes,
            ),
            (0, 0, 0, 0, 0)
        );
        assert!(m.cuda_mem_probe_events.is_empty());
        assert!(m.decoder_cache_snapshots.is_empty());
        assert_eq!(
            (
                m.rollout_ratio_mean,
                m.rollout_logratio_mean,
                m.rollout_ratio_max,
                m.frac_rollout_ratio_capped,
                m.rollout_capture_tokens,
            ),
            (1.0, 0.0, 1.0, 0.0, 0)
        );
    }

    #[test]
    fn metrics_writer_sanitizes_nonfinite_fields() {
        // Non-finite f32 would serialize to JSON `null` and fail to re-read;
        // append() must nan_to_num them so the line stays valid + parseable.
        let tmp = TempDir::new("nonfinite");
        let rd = RunDir::create(tmp.path(), "r").unwrap();
        let mut w = rd.metrics_writer().unwrap();
        let mut m = Metrics::at_step(0);
        m.grad_norm = f32::INFINITY;
        m.reward_mean = f32::NEG_INFINITY;
        m.kl = f32::NAN;
        m.beta = f32::INFINITY;
        m.frac_truncated = f32::NAN;
        m.rollout_ratio_max = f32::INFINITY; // an overflowed exp() telemetry value
        w.append(&m).unwrap();
        drop(w);

        let raw = std::fs::read_to_string(rd.metrics_path()).unwrap();
        let back: Metrics = serde_json::from_str(raw.lines().next().unwrap()).unwrap();
        assert_eq!(back.grad_norm, f32::MAX);
        assert_eq!(back.reward_mean, f32::MIN);
        assert_eq!(back.kl, 0.0);
        assert_eq!(back.beta, f32::MAX);
        assert_eq!(back.frac_truncated, 0.0);
        assert_eq!(back.rollout_ratio_max, f32::MAX);
    }

    /// A metrics record carrying the fields [`summarize`] reads, with off-policy
    /// telemetry marked live (`rollout_capture_tokens = 1`) so a stream of these is
    /// not flagged [`Anomaly::TelemetryDark`] unless a test opts in.
    fn metric(step: u64, reward: f32, grad_norm: f32, step_secs: f32, toks: f32) -> Metrics {
        let mut m = Metrics::at_step(step);
        m.reward_mean = reward;
        m.grad_norm = grad_norm;
        m.step_secs = step_secs;
        m.tokens_per_sec = toks;
        m.rollout_capture_tokens = 1;
        m
    }

    fn perf_metric(step: u64, grad_norm: f32, step_secs: f32, peak: u64, delta: u64) -> Metrics {
        let mut m = metric(step, 0.1 * step as f32, grad_norm, step_secs, 100.0);
        m.cuda_mem_peak_used_bytes = peak;
        m.cuda_mem_peak_delta_bytes = delta;
        m
    }

    fn perf_budget() -> RegressionBudget {
        RegressionBudget {
            max_mean_step_secs_ratio: 1.10,
            max_cuda_peak_used_ratio: 1.05,
            max_cuda_peak_used_abs_slack_bytes: 0,
            max_cuda_peak_delta_ratio: Some(1.05),
            max_cuda_peak_delta_abs_slack_bytes: 0,
            max_final_grad_norm_rel_drift: Some(1e-6),
            ..RegressionBudget::default()
        }
    }

    #[test]
    fn compare_metrics_identical_streams_pass() {
        let hist = vec![
            perf_metric(0, 1.0, 2.0, 1000, 100),
            perf_metric(1, 1.0, 2.0, 1100, 120),
        ];
        let report = compare_metrics(&hist, &hist, &perf_budget());
        assert!(report.passed, "failures: {:?}", report.failures);
        assert!(report.failures.is_empty());
        assert_eq!(report.baseline.as_ref().unwrap().steps, 2);
    }

    #[test]
    fn compare_metrics_allows_unchanged_health_warnings() {
        let mut hist = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        hist[0].dropped_rows = 2;

        let report = compare_metrics(&hist, &hist, &perf_budget());
        assert!(report.passed, "failures: {:?}", report.failures);
        assert_eq!(
            report.baseline.as_ref().unwrap().anomalies,
            vec![Anomaly::DroppedRows { total: 2 }]
        );
    }

    #[test]
    fn compare_metrics_fails_health_warning_regressions() {
        let mut base = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        base[0].dropped_rows = 1;
        let mut candidate = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        candidate[0].dropped_rows = 2;

        let report = compare_metrics(&base, &candidate, &perf_budget());
        assert!(
            report
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::HealthWarnings { .. })),
            "failures: {:?}",
            report.failures
        );
    }

    #[test]
    fn compare_metrics_fails_empty_and_misaligned_streams() {
        let hist = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        let empty = compare_metrics(&[], &hist, &perf_budget());
        assert!(
            empty
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::EmptyStream { stream: "baseline" })),
            "failures: {:?}",
            empty.failures
        );

        let mut shifted = hist.clone();
        shifted[0].step = 7;
        let mismatch = compare_metrics(&hist, &shifted, &perf_budget());
        assert!(
            mismatch
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::StepIndexMismatch { .. })),
            "failures: {:?}",
            mismatch.failures
        );
    }

    #[test]
    fn compare_metrics_requires_live_update() {
        let base = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        let cand = vec![perf_metric(0, 0.0, 2.0, 1000, 100)];
        let report = compare_metrics(&base, &cand, &perf_budget());
        assert!(
            report.failures.iter().any(|f| matches!(
                f,
                RegressionFailure::LiveUpdateMissing {
                    stream: "candidate",
                    ..
                }
            )),
            "failures: {:?}",
            report.failures
        );
    }

    #[test]
    fn compare_metrics_fails_missing_required_telemetry() {
        let base = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        let mut cand = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        cand[0].step_secs = 0.0;
        let timing = compare_metrics(&base, &cand, &perf_budget());
        assert!(
            timing.failures.iter().any(|f| matches!(
                f,
                RegressionFailure::TimingMissing {
                    stream: "candidate"
                }
            )),
            "failures: {:?}",
            timing.failures
        );

        cand[0].step_secs = 2.0;
        cand[0].cuda_mem_peak_used_bytes = 0;
        let memory = compare_metrics(&base, &cand, &perf_budget());
        assert!(
            memory.failures.iter().any(|f| matches!(
                f,
                RegressionFailure::CudaMemoryMissing {
                    stream: "candidate"
                }
            )),
            "failures: {:?}",
            memory.failures
        );
    }

    #[test]
    fn compare_metrics_fails_partial_missing_cuda_memory_telemetry() {
        let base = vec![
            perf_metric(0, 1.0, 2.0, 1000, 100),
            perf_metric(1, 1.0, 2.0, 1000, 100),
            perf_metric(2, 1.0, 2.0, 1000, 100),
        ];
        let mut cand = base.clone();
        cand[1].cuda_mem_peak_used_bytes = 0;
        let report = compare_metrics(&base, &cand, &perf_budget());
        assert!(
            report.failures.iter().any(|f| matches!(
                f,
                RegressionFailure::CudaMemoryMissing {
                    stream: "candidate"
                }
            )),
            "failures: {:?}",
            report.failures
        );
    }

    #[test]
    fn compare_distributed_metrics_uses_world_max_cuda_peak() {
        let base_rank0 = vec![perf_metric(0, 1.0, 2.0, 1000, 100)];
        let base_rank1 = vec![perf_metric(0, 1.0, 2.0, 2000, 200)];
        let cand_rank0 = vec![perf_metric(0, 1.0, 2.0, 2000, 200)];
        let cand_rank1 = vec![perf_metric(0, 1.0, 2.0, 2000, 200)];
        let budget = RegressionBudget {
            max_cuda_peak_used_ratio: 1.0,
            max_cuda_peak_used_abs_slack_bytes: 0,
            max_cuda_peak_delta_ratio: Some(1.0),
            max_cuda_peak_delta_abs_slack_bytes: 0,
            ..perf_budget()
        };

        let rankwise = compare_metrics(&base_rank0, &cand_rank0, &budget);
        assert!(
            rankwise
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::CudaPeakRegression { .. })),
            "rank-wise comparison should fail: {:?}",
            rankwise.failures
        );

        let distributed = compare_distributed_metrics(
            &[base_rank0, base_rank1],
            &[cand_rank0, cand_rank1],
            &budget,
        );
        assert!(
            distributed.passed,
            "world-max comparison should pass: {:?}",
            distributed.failures
        );
        assert_eq!(
            distributed
                .baseline
                .as_ref()
                .unwrap()
                .max_cuda_mem_peak_used_bytes,
            2000
        );
        assert_eq!(
            distributed
                .candidate
                .as_ref()
                .unwrap()
                .max_cuda_mem_peak_used_bytes,
            2000
        );
    }

    #[test]
    fn compare_distributed_metrics_fails_missing_rank_cuda_memory() {
        let base = vec![
            vec![perf_metric(0, 1.0, 2.0, 1000, 100)],
            vec![perf_metric(0, 1.0, 2.0, 2000, 200)],
        ];
        let candidate = vec![
            vec![perf_metric(0, 1.0, 2.0, 1000, 100)],
            vec![perf_metric(0, 1.0, 2.0, 0, 200)],
        ];

        let report = compare_distributed_metrics(&base, &candidate, &perf_budget());
        assert!(
            report.failures.iter().any(|f| matches!(
                f,
                RegressionFailure::CudaMemoryMissing {
                    stream: "candidate"
                }
            )),
            "failures: {:?}",
            report.failures
        );
    }

    #[test]
    fn compare_distributed_metrics_fails_rank_local_health_regressions() {
        let base = vec![
            vec![perf_metric(0, 1.0, 2.0, 1000, 100)],
            vec![perf_metric(0, 1.0, 2.0, 1000, 100)],
        ];
        let mut candidate = base.clone();
        candidate[0][0].rollout_capture_tokens = 0;

        let report = compare_distributed_metrics(&base, &candidate, &perf_budget());
        assert!(
            report.failures.iter().any(|f| matches!(
                f,
                RegressionFailure::RankHealthWarnings {
                    rank: 0,
                    baseline: 0,
                    candidate: 1,
                }
            )),
            "failures: {:?}",
            report.failures
        );
        assert!(
            !report
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::HealthWarnings { .. })),
            "aggregate health should not be the failing signal: {:?}",
            report.failures
        );

        let permissive = RegressionBudget {
            allow_health_warnings: true,
            ..perf_budget()
        };
        let allowed = compare_distributed_metrics(&base, &candidate, &permissive);
        assert!(allowed.passed, "failures: {:?}", allowed.failures);
    }

    #[test]
    fn compare_distributed_metrics_fails_rank_shape_mismatches() {
        let base = vec![
            vec![perf_metric(0, 1.0, 2.0, 1000, 100)],
            vec![
                perf_metric(0, 1.0, 2.0, 1000, 100),
                perf_metric(1, 1.0, 2.0, 1000, 100),
            ],
        ];
        let candidate = vec![
            vec![perf_metric(0, 1.0, 2.0, 1000, 100)],
            vec![perf_metric(7, 1.0, 2.0, 1000, 100)],
            vec![perf_metric(0, 1.0, 2.0, 1000, 100)],
        ];

        let report = compare_distributed_metrics(&base, &candidate, &perf_budget());
        assert!(
            report
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::RankCountMismatch { .. })),
            "failures: {:?}",
            report.failures
        );
        assert!(
            report
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::RankStepCountMismatch { .. })),
            "failures: {:?}",
            report.failures
        );
        assert!(
            report
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::RankStepIndexMismatch { .. })),
            "failures: {:?}",
            report.failures
        );
    }

    #[test]
    fn compare_metrics_fails_resource_regressions() {
        let base = vec![
            perf_metric(0, 1.0, 2.0, 1000, 100),
            perf_metric(1, 1.0, 2.0, 1000, 100),
        ];
        let slow = vec![
            perf_metric(0, 1.0, 2.3, 1000, 100),
            perf_metric(1, 1.0, 2.3, 1000, 100),
        ];
        let slow_report = compare_metrics(&base, &slow, &perf_budget());
        assert!(
            slow_report
                .failures
                .iter()
                .any(|f| matches!(f, RegressionFailure::StepTimeRegression { .. })),
            "failures: {:?}",
            slow_report.failures
        );

        let high_mem = vec![
            perf_metric(0, 1.0, 2.0, 1200, 130),
            perf_metric(1, 1.0, 2.0, 1200, 130),
        ];
        let mem_report = compare_metrics(&base, &high_mem, &perf_budget());
        assert!(
            mem_report.failures.iter().any(|f| matches!(
                f,
                RegressionFailure::CudaPeakRegression { .. }
                    | RegressionFailure::CudaDeltaRegression { .. }
            )),
            "failures: {:?}",
            mem_report.failures
        );
    }

    #[test]
    fn summarize_empty_stream_is_none() {
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn summarize_reports_reward_trend() {
        // Reward rises 0.0 → 0.5 over 6 steps.
        let hist: Vec<Metrics> = (0..6)
            .map(|i| metric(i, 0.1 * i as f32, 1.0, 2.0, 100.0))
            .collect();
        let mut hist = hist;
        hist[5].beta = 0.02;
        let s = summarize(&hist).unwrap();
        assert_eq!((s.steps, s.first_step, s.last_step), (6, 0, 5));
        assert!((s.reward_last - 0.5).abs() < 1e-6, "last {}", s.reward_last);
        assert!(s.reward_trend > 0.0, "trend {}", s.reward_trend);
        assert_eq!(s.final_beta, 0.02);
    }

    #[test]
    fn summarize_reports_throughput_and_grad_health() {
        // Steady throughput (2 s/step, 100 tok/s) and flat grad ≈ 1.0 → no flags.
        // Values are identical per step, so the means are exact.
        let hist: Vec<Metrics> = (0..6)
            .map(|i| metric(i, 0.1 * i as f32, 1.0, 2.0, 100.0))
            .collect();
        let s = summarize(&hist).unwrap();
        assert_eq!(s.mean_step_secs, 2.0);
        assert_eq!(s.mean_tokens_per_sec, 100.0);
        assert_eq!(s.max_grad_norm, 1.0);
        assert!(s.anomalies.is_empty(), "healthy run: {:?}", s.anomalies);
    }

    #[test]
    fn summarize_reports_cuda_memory_peak_when_present() {
        let mut hist: Vec<Metrics> = (0..3)
            .map(|i| metric(i, 0.1 * i as f32, 1.0, 2.0, 100.0))
            .collect();
        hist[1].cuda_mem_peak_used_bytes = 16 * 1024 * 1024;
        hist[1].cuda_mem_peak_delta_bytes = 4 * 1024 * 1024;

        let s = summarize(&hist).unwrap();
        assert_eq!(s.max_cuda_mem_peak_used_bytes, 16 * 1024 * 1024);
        assert_eq!(s.max_cuda_mem_peak_delta_bytes, 4 * 1024 * 1024);
        assert!(s.to_string().contains("cuda_mem"));
    }

    #[test]
    fn summarize_flags_nonfinite_saturation() {
        let mut hist: Vec<Metrics> = (0..3).map(|i| metric(i, 0.5, 1.0, 1.0, 50.0)).collect();
        hist[1].grad_norm = f32::MAX; // a nan_to_num'd blow-up
        let s = summarize(&hist).unwrap();
        assert!(
            s.anomalies
                .iter()
                .any(|a| matches!(a, Anomaly::NonFinite { field, .. } if *field == "grad_norm")),
            "expected a NonFinite(grad_norm) flag, got {:?}",
            s.anomalies
        );
    }

    #[test]
    fn summarize_flags_grad_spike() {
        // Steady grad_norm ≈ 1.0 with one 20× spike (no saturation).
        let mut hist: Vec<Metrics> = (0..6).map(|i| metric(i, 0.5, 1.0, 1.0, 50.0)).collect();
        hist[4].grad_norm = 20.0;
        let s = summarize(&hist).unwrap();
        assert!(
            s.anomalies
                .iter()
                .any(|a| matches!(a, Anomaly::GradSpike { step: 4, .. })),
            "expected a GradSpike at step 4, got {:?}",
            s.anomalies
        );
    }

    #[test]
    fn summarize_flags_reward_stall() {
        // Flat reward + every group degenerate over ≥ STALL_MIN_STEPS steps.
        let hist: Vec<Metrics> = (0..6)
            .map(|i| {
                let mut m = metric(i, 0.5, 1.0, 1.0, 50.0);
                m.frac_reward_zero_std = 1.0;
                m
            })
            .collect();
        let s = summarize(&hist).unwrap();
        assert!(
            s.anomalies
                .iter()
                .any(|a| matches!(a, Anomaly::RewardStall { .. })),
            "expected a RewardStall flag, got {:?}",
            s.anomalies
        );
    }

    #[test]
    fn summarize_flags_dropped_rows_and_dark_telemetry() {
        // at_step leaves rollout_capture_tokens = 0 → the whole run is telemetry-dark.
        let hist: Vec<Metrics> = (0..3)
            .map(|i| {
                let mut m = Metrics::at_step(i);
                m.dropped_rows = 2;
                m
            })
            .collect();
        let s = summarize(&hist).unwrap();
        assert_eq!(s.total_dropped_rows, 6);
        let flags = format!("{:?}", s.anomalies);
        assert!(flags.contains("DroppedRows"), "flags: {flags}");
        assert!(flags.contains("TelemetryDark"), "flags: {flags}");
    }

    #[test]
    fn run_summary_healthy_renders_text_and_json() {
        let hist: Vec<Metrics> = (0..4)
            .map(|i| metric(i, 0.1 * i as f32, 1.0, 1.0, 50.0))
            .collect();
        let s = summarize(&hist).unwrap();
        let text = s.to_string();
        assert!(text.contains("HEALTHY"), "text:\n{text}");
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("reward_trend"), "json: {json}");
        assert!(json.contains("mean_tokens_per_sec"), "json: {json}");
    }

    #[test]
    fn run_summary_warn_renders_flags_and_dropped_rows() {
        // Dark + dropped rows → WARN verdict, a dropped_rows line, and a flag bullet.
        let hist: Vec<Metrics> = (0..2)
            .map(|i| {
                let mut m = Metrics::at_step(i);
                m.dropped_rows = 1;
                m
            })
            .collect();
        let s = summarize(&hist).unwrap();
        let text = s.to_string();
        assert!(text.contains("WARN"), "text:\n{text}");
        assert!(text.contains("dropped_rows"), "text:\n{text}");
        assert!(text.contains("! "), "text:\n{text}");
    }

    #[test]
    fn anomaly_variants_all_render_non_empty() {
        // Exercises every Anomaly Display arm (one per variant).
        let flags = [
            Anomaly::NonFinite {
                step: 1,
                field: "grad_norm",
            },
            Anomaly::GradSpike {
                step: 2,
                grad_norm: 20.0,
                median: 1.0,
            },
            Anomaly::RewardStall {
                mean_frac_zero_std: 1.0,
                reward_trend: 0.0,
            },
            Anomaly::DroppedRows { total: 3 },
            Anomaly::TelemetryDark,
        ];
        for a in &flags {
            assert!(!a.to_string().is_empty(), "empty render: {a:?}");
        }
    }

    #[test]
    fn telemetry_error_io_is_displayable() {
        let err = TelemetryError::io(PathBuf::from("/nope"), io::Error::other("boom"));
        let s = format!("{err}");
        assert!(s.contains("/nope"));
        assert!(s.contains("boom"));
    }
}
