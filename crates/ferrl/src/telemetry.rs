//! Run telemetry: structured tracing plus a per-run on-disk layout.
//!
//! Every training run materializes a `runs/<run_id>/` directory containing
//! `config.json` (the run configuration), `metrics.jsonl` (one [`Metrics`] JSON
//! object per optimizer step), a `checkpoints/` subdirectory, and `run.log` (a
//! human-readable log). [`init_tracing`] wires up `tracing` once for the
//! process; [`RunDir`] owns the directory; [`MetricsWriter`] appends step
//! metrics as JSON Lines.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

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
    /// [`RunDir::create`] was given a `run_id` whose directory already holds a
    /// metrics stream — appending a fresh run to a prior run's `metrics.jsonl`
    /// would silently interleave two runs. Use a new `run_id`, or
    /// [`RunDir::open`] to deliberately continue the existing run (resume).
    #[error(
        "run directory already contains a metrics stream at {path} \
         (duplicate run_id? use RunDir::open to resume)"
    )]
    DuplicateRun {
        /// The pre-existing `metrics.jsonl`.
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
    /// Standard filename for the serialized run configuration.
    pub const CONFIG_FILE: &'static str = "config.json";
    /// Standard filename for the human-readable run log.
    pub const LOG_FILE: &'static str = "run.log";
    /// Standard subdirectory for model checkpoints.
    pub const CHECKPOINTS_DIR: &'static str = "checkpoints";

    /// Create `runs_root/<run_id>/` (and its `checkpoints/` subdir) for a
    /// **fresh** run.
    ///
    /// Fails loud if the directory already holds a `metrics.jsonl`: the metrics
    /// writer appends, so a duplicate `run_id` would silently interleave a new
    /// run's stream into a prior run's file (the `RUNDIR-APPEND` hazard). To
    /// deliberately continue an existing run — the checkpoint-resume path — use
    /// [`open`](Self::open) instead.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::DuplicateRun`] if `runs_root/<run_id>/metrics.jsonl`
    /// already exists, or [`TelemetryError::Io`] if any directory cannot be created.
    pub fn create(
        runs_root: impl AsRef<Path>,
        run_id: impl Into<String>,
    ) -> Result<Self, TelemetryError> {
        let run_id = run_id.into();
        let root = runs_root.as_ref().join(&run_id);
        let metrics = root.join(Self::METRICS_FILE);
        if metrics.exists() {
            return Err(TelemetryError::DuplicateRun { path: metrics });
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
}

impl Metrics {
    /// A zeroed record at the given step — convenient for tests and for steps
    /// where a particular quantity is not yet available.
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
            dropped_rows: 0,
            grad_norm: 0.0,
            lr: 0.0,
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
            &mut m.grad_norm,
            &mut m.lr,
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
        Ok(Self { path, file })
    }

    /// Append one metrics record as a JSON line and flush it.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError`] if serialization or the write/flush fails.
    pub fn append(&mut self, metrics: &Metrics) -> Result<(), TelemetryError> {
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

    /// The path this writer appends to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
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
            dropped_rows: 3,
            grad_norm: 1.23,
            lr: 5e-6,
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: Metrics = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn metrics_deserializes_old_log_without_new_fields() {
        // A pre-PR#3 metrics.jsonl line lacks frac_reward_zero_std and dropped_rows;
        // #[serde(default)] must let it deserialize (defaulting both to 0).
        let old = r#"{"step":3,"reward_mean":1.0,"reward_std":0.5,"kl":0.01,"clip_ratio":0.2,"completion_len":10.0,"grad_norm":1.0,"lr":1e-5}"#;
        let m: Metrics = serde_json::from_str(old).unwrap();
        assert_eq!(m.step, 3);
        assert_eq!(m.frac_reward_zero_std, 0.0);
        assert_eq!(m.dropped_rows, 0);
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
        m.frac_truncated = f32::NAN;
        w.append(&m).unwrap();
        drop(w);

        let raw = std::fs::read_to_string(rd.metrics_path()).unwrap();
        let back: Metrics = serde_json::from_str(raw.lines().next().unwrap()).unwrap();
        assert_eq!(back.grad_norm, f32::MAX);
        assert_eq!(back.reward_mean, f32::MIN);
        assert_eq!(back.kl, 0.0);
        assert_eq!(back.frac_truncated, 0.0);
    }

    #[test]
    fn telemetry_error_io_is_displayable() {
        let err = TelemetryError::io(PathBuf::from("/nope"), io::Error::other("boom"));
        let s = format!("{err}");
        assert!(s.contains("/nope"));
        assert!(s.contains("boom"));
    }
}
