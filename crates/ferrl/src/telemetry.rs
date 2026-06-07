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

    /// Create `runs_root/<run_id>/` (and its `checkpoints/` subdir).
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Io`] if any directory cannot be created.
    pub fn create(
        runs_root: impl AsRef<Path>,
        run_id: impl Into<String>,
    ) -> Result<Self, TelemetryError> {
        let run_id = run_id.into();
        let root = runs_root.as_ref().join(&run_id);
        fs::create_dir_all(&root).map_err(|e| TelemetryError::io(&root, e))?;
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Global optimizer step (0-based).
    pub step: u64,
    /// Mean scalar reward over the batch.
    pub reward_mean: f32,
    /// Standard deviation of scalar rewards over the batch.
    pub reward_std: f32,
    /// Mean k3 KL to the reference policy.
    pub kl: f32,
    /// Fraction of tokens whose surrogate hit the PPO clip band.
    pub clip_ratio: f32,
    /// Mean completion length in tokens.
    pub completion_len: f32,
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
            kl: 0.0,
            clip_ratio: 0.0,
            completion_len: 0.0,
            grad_norm: 0.0,
            lr: 0.0,
        }
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
        let mut line = serde_json::to_string(metrics)?;
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
            kl: 0.02,
            clip_ratio: 0.1,
            completion_len: 42.0,
            grad_norm: 1.23,
            lr: 5e-6,
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: Metrics = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn telemetry_error_io_is_displayable() {
        let err = TelemetryError::io(PathBuf::from("/nope"), io::Error::other("boom"));
        let s = format!("{err}");
        assert!(s.contains("/nope"));
        assert!(s.contains("boom"));
    }
}
