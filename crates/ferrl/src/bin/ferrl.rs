//! `ferrl` — the single-binary front door: train a built-in task end-to-end from a
//! JSON run config, and report on a finished run.
//!
//! ```text
//! ferrl train --config run.json     # GRPO-train a built-in task (countdown | math)
//! ferrl runreport <run-dir> [--json] [--strict]   # one-glance run health summary
//! ```
//!
//! `train` reads a `RunConfig` (a serialized [`TrainerConfig`](ferrl::TrainerConfig)
//! plus a model directory, a device, and a task selector), loads a Qwen policy via
//! [`ferrl::load_qwen_policy`], builds the named task's train/eval splits, and runs
//! the GRPO [`Trainer`](ferrl::Trainer). The task registry is closed (the two worked
//! examples, `countdown` and `math`); a *custom* task is wired in Rust against the
//! library — see `examples/minimal_task.rs` and the README's "Wire your own task".
//!
//! `runreport` folds in the standalone run-summary tool: it reads a run's
//! `metrics.jsonl` and prints (or emits as JSON) a [`RunSummary`](ferrl::RunSummary),
//! optionally failing (`--strict`, exit code 2) when any health anomaly is flagged.

// A CLI whose interface *is* its stdout/stderr; the library logs via `tracing`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use candle_core::{DType, Device};
use clap::{Args, Parser, Subcommand};
use serde::Deserialize;
use tracing::info;

use ferrl::countdown::{build_prompt, generate_dataset, CountdownConfig, CountdownProblem};
use ferrl::policy::GenConfig;
use ferrl::{
    evaluate, load_qwen_policy, read_jsonl, summarize, train_eval_split, CountdownReward,
    LoaderOpts, MathProblem, MathReward, RewardFn, RunDir, Sample, Trainer, TrainerConfig,
};

/// A task's train/eval split: `(train, eval)` samples of the task's target type.
type Splits<T> = (Vec<Sample<T>>, Vec<Sample<T>>);

/// The `ferrl` command-line interface.
#[derive(Debug, Parser)]
#[command(
    name = "ferrl",
    version,
    about = "candle-native GRPO trainer — single-binary ops"
)]
struct Cli {
    /// The subcommand to run.
    #[command(subcommand)]
    cmd: Command,
}

/// Top-level `ferrl` subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// GRPO-train a built-in task end-to-end from a JSON run config.
    Train(TrainArgs),
    /// Print a one-glance health summary for a finished run.
    Runreport(RunreportArgs),
}

/// Arguments for `ferrl train`.
#[derive(Debug, Args)]
struct TrainArgs {
    /// Path to the JSON run config (see `RunConfig`).
    #[arg(long)]
    config: PathBuf,
}

/// Arguments for `ferrl runreport`.
#[derive(Debug, Args)]
struct RunreportArgs {
    /// A run directory (its `metrics.jsonl` is used) or a `metrics.jsonl` file.
    path: PathBuf,
    /// Emit the summary as JSON instead of the human report.
    #[arg(long)]
    json: bool,
    /// Exit non-zero (code 2) if any health anomaly is flagged.
    #[arg(long)]
    strict: bool,
}

/// Errors surfaced by the `ferrl` CLI.
#[derive(Debug, thiserror::Error)]
enum CliError {
    /// A CLI-level usage or contract error, described by a message.
    #[error("{0}")]
    Msg(String),
    /// A file could not be read.
    #[error("read {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying IO error.
        source: std::io::Error,
    },
    /// The run config could not be parsed.
    #[error("parse run config {path}: {source}")]
    Config {
        /// The config file that failed to parse.
        path: PathBuf,
        /// The underlying deserialization error.
        source: serde_json::Error,
    },
    /// The policy/tokenizer could not be loaded.
    #[error(transparent)]
    Loader(#[from] ferrl::LoaderError),
    /// A dataset could not be read.
    #[error(transparent)]
    Data(#[from] ferrl::DataError),
    /// The trainer failed.
    #[error(transparent)]
    Trainer(#[from] ferrl::TrainerError),
    /// The held-out eval failed.
    #[error(transparent)]
    Eval(#[from] ferrl::EvalError),
    /// A run-directory / metrics IO error.
    #[error(transparent)]
    Telemetry(#[from] ferrl::telemetry::TelemetryError),
    /// A CUDA device error (only on a `--features cuda` build).
    #[cfg(feature = "cuda")]
    #[error("{0}")]
    Candle(#[from] candle_core::Error),
}

impl CliError {
    /// Construct a message-only CLI error.
    fn msg(msg: impl Into<String>) -> Self {
        Self::Msg(msg.into())
    }
}

/// Which device to run on.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DeviceSel {
    /// The CPU (the default; the only device a `--features cuda`-less build supports).
    #[default]
    Cpu,
    /// CUDA device 0 (requires a `--features cuda` build).
    Cuda,
}

impl DeviceSel {
    /// Open the selected device, running the CUDA preflight when applicable.
    fn open(self) -> Result<Device, CliError> {
        match self {
            DeviceSel::Cpu => Ok(Device::Cpu),
            DeviceSel::Cuda => open_cuda(),
        }
    }
}

/// Open CUDA device 0 with the driver-compat preflight (a `--features cuda` build).
#[cfg(feature = "cuda")]
fn open_cuda() -> Result<Device, CliError> {
    let device = Device::new_cuda(0)?;
    if let Some(w) = ferrl::check_driver_compat(&device).warning() {
        tracing::warn!("{w}");
    }
    ferrl::guard_first_kernel(&device)?;
    Ok(device)
}

/// Without the `cuda` feature there is no CUDA backend to open.
#[cfg(not(feature = "cuda"))]
fn open_cuda() -> Result<Device, CliError> {
    Err(CliError::msg(
        "device \"cuda\" requires building ferrl with --features cuda; use device \"cpu\" otherwise",
    ))
}

/// The dtype the frozen base weights load in.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DtypeSel {
    /// 32-bit float — the natural CPU dtype (the default).
    #[default]
    F32,
    /// bfloat16 — halves the frozen base's memory on a GPU run.
    Bf16,
}

impl DtypeSel {
    /// The candle [`DType`] this selector denotes.
    fn as_dtype(self) -> DType {
        match self {
            DtypeSel::F32 => DType::F32,
            DtypeSel::Bf16 => DType::BF16,
        }
    }
}

/// Policy-load knobs (the `LoRA` shape, base dtype, seed). The rollout temperature
/// is taken from the trainer config so the two cannot disagree.
#[derive(Debug, Deserialize)]
#[serde(default)]
struct PolicyCfg {
    /// `LoRA` rank.
    lora_rank: usize,
    /// `LoRA` alpha.
    lora_alpha: f64,
    /// Dtype the frozen base loads in.
    base_dtype: DtypeSel,
    /// Rollout sampler seed.
    seed: u64,
}

impl Default for PolicyCfg {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            base_dtype: DtypeSel::F32,
            seed: 1234,
        }
    }
}

/// Dataset knobs: a JSONL `path` for file-backed tasks (`math`), or the generated
/// `train_n` for procedural tasks (`countdown`), plus the held-out `eval_n` and the
/// `seed` for the deterministic dedup-aware split.
#[derive(Debug, Deserialize)]
#[serde(default)]
struct DataCfg {
    /// JSONL dataset path (required for `math`; ignored by `countdown`).
    path: Option<PathBuf>,
    /// Number of generated train problems (procedural tasks only).
    train_n: usize,
    /// Held-out eval count (`0` skips the post-train eval).
    eval_n: usize,
    /// Seed for dataset generation and the train/eval split.
    seed: u64,
}

impl Default for DataCfg {
    fn default() -> Self {
        Self {
            path: None,
            train_n: 64,
            eval_n: 0,
            seed: 7,
        }
    }
}

/// A `ferrl train` run, deserialized from JSON.
///
/// The wire shape is a flat object: a `task` selector, the `model_dir` checkpoint,
/// an optional `device` / `out_dir` / `policy` / `data` block, and the full
/// [`TrainerConfig`] under `trainer`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfig {
    /// Which built-in task to train: `"countdown"` or `"math"`.
    task: String,
    /// Checkpoint directory (`config.json` + `model.safetensors` + `tokenizer.json`).
    model_dir: PathBuf,
    /// Device to run on (default `cpu`).
    #[serde(default)]
    device: DeviceSel,
    /// Where run directories are written (default `runs/`).
    #[serde(default = "default_out_dir")]
    out_dir: PathBuf,
    /// Policy-load knobs.
    #[serde(default)]
    policy: PolicyCfg,
    /// Dataset knobs.
    #[serde(default)]
    data: DataCfg,
    /// The GRPO trainer config.
    trainer: TrainerConfig,
}

/// `serde` default for [`RunConfig::out_dir`]: `runs/`.
fn default_out_dir() -> PathBuf {
    PathBuf::from("runs")
}

impl RunConfig {
    /// Read and parse a run config from `path`.
    fn load(path: &Path) -> Result<Self, CliError> {
        let bytes = std::fs::read(path).map_err(|source| CliError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| CliError::Config {
            path: path.to_path_buf(),
            source,
        })
    }

    /// The loader options for this run (rollout temperature mirrors the trainer's).
    fn loader_opts(&self) -> LoaderOpts {
        LoaderOpts {
            lora_rank: self.policy.lora_rank,
            lora_alpha: self.policy.lora_alpha,
            base_dtype: self.policy.base_dtype.as_dtype(),
            adapter_dtype: DType::F32,
            seed: self.policy.seed,
            temperature: self.trainer.temperature,
        }
    }

    /// A unique run id for this invocation: `<task>-<unix-seconds>`.
    fn run_id(&self) -> String {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        format!("{}-{stamp}", self.task)
    }

    /// Build the Countdown train/eval splits: generate `train_n + eval_n` problems
    /// and hold out `eval_n` via the dedup-aware [`train_eval_split`].
    fn countdown_splits(&self) -> Splits<CountdownProblem> {
        let cd = CountdownConfig::default();
        let n = (self.data.train_n + self.data.eval_n).max(1);
        let samples: Vec<Sample<CountdownProblem>> = generate_dataset(self.data.seed, n, &cd)
            .into_iter()
            .map(|p| Sample::new(build_prompt(&p), p))
            .collect();
        train_eval_split(samples, self.data.eval_n, self.data.seed)
    }

    /// Build the math train/eval splits from the configured JSONL `data.path`.
    fn math_splits(&self) -> Result<Splits<MathProblem>, CliError> {
        let path = self.data.path.as_ref().ok_or_else(|| {
            CliError::msg("task \"math\" requires data.path (a JSONL dataset of {prompt, target})")
        })?;
        let samples = read_jsonl::<MathProblem, _>(path)?;
        Ok(train_eval_split(samples, self.data.eval_n, self.data.seed))
    }
}

/// Dispatch `ferrl train`: parse the config, open the device, build the named task's
/// data, and run training.
fn train(args: &TrainArgs) -> Result<(), CliError> {
    let _ = ferrl::init_tracing();
    let cfg = RunConfig::load(&args.config)?;
    let device = cfg.device.open()?;
    match cfg.task.as_str() {
        "countdown" => {
            let (train, eval) = cfg.countdown_splits();
            run_training(&cfg, &device, &CountdownReward::default(), &train, &eval)
        }
        "math" => {
            let (train, eval) = cfg.math_splits()?;
            run_training(&cfg, &device, &MathReward::default(), &train, &eval)
        }
        other => Err(CliError::msg(format!(
            "unknown task {other:?}; built-in tasks are \"countdown\" and \"math\""
        ))),
    }
}

/// Run GRPO training (and, when `eval` is non-empty, a held-out eval) for any task.
///
/// Monomorphized per task by the [`train`] dispatch — the one place the concrete
/// reward and its typed target are known.
fn run_training<R: RewardFn>(
    cfg: &RunConfig,
    device: &Device,
    reward: &R,
    train: &[Sample<R::Target>],
    eval: &[Sample<R::Target>],
) -> Result<(), CliError> {
    let (mut policy, tok) = load_qwen_policy(&cfg.model_dir, device, &cfg.loader_opts())?;
    let tcfg = cfg.trainer.clone();
    let gen = GenConfig::from(&tcfg);
    info!(
        task = %cfg.task,
        steps = tcfg.steps,
        group_size = tcfg.group_size,
        train = train.len(),
        eval = eval.len(),
        "ferrl train: starting"
    );

    let run = RunDir::create(&cfg.out_dir, cfg.run_id())?;
    let mut trainer = Trainer::new(tcfg, &run)?;
    let (history, _stop) = trainer.train(&mut policy, reward, &tok, train)?;
    if let Some(summary) = summarize(&history) {
        info!(steps = summary.steps, "ferrl train: complete");
    }

    if !eval.is_empty() {
        let report = evaluate(&mut policy, reward, &tok, eval, &gen)?;
        info!(
            base = report.base_reward_mean,
            adapter = report.adapter_reward_mean,
            improvement = report.improvement(),
            "ferrl train: held-out eval (adapter vs base)"
        );
    }

    println!("ferrl: run complete -> {}", run.root().display());
    println!(
        "ferrl: inspect with `ferrl runreport {}`",
        run.root().display()
    );
    Ok(())
}

/// Dispatch `ferrl runreport`: read the run's metrics, summarize, and emit.
fn runreport(args: &RunreportArgs) -> Result<ExitCode, CliError> {
    let metrics_path = resolve_metrics_path(&args.path);
    let history = ferrl::read_metrics(&metrics_path)?;
    let summary = summarize(&history).ok_or_else(|| {
        CliError::msg(format!(
            "{} has no metrics records yet",
            metrics_path.display()
        ))
    })?;
    if args.json {
        let s = serde_json::to_string_pretty(&summary)
            .map_err(|e| CliError::msg(format!("serialize summary: {e}")))?;
        println!("{s}");
    } else {
        // `RunSummary`'s Display already terminates each line with a newline.
        print!("{summary}");
    }
    if args.strict && !summary.anomalies.is_empty() {
        return Ok(ExitCode::from(2));
    }
    Ok(ExitCode::SUCCESS)
}

/// If `arg` is a directory, append the run's `metrics.jsonl`; otherwise treat it as
/// the metrics file path directly.
fn resolve_metrics_path(arg: &Path) -> PathBuf {
    if arg.is_dir() {
        arg.join(RunDir::METRICS_FILE)
    } else {
        arg.to_path_buf()
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.cmd {
        Command::Train(args) => train(args).map(|()| ExitCode::SUCCESS),
        Command::Runreport(args) => runreport(args),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("ferrl: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal countdown run config parses with sensible defaults.
    #[test]
    #[allow(clippy::cognitive_complexity)] // assertion-heavy test: many small checks, no real branching
    fn parses_a_countdown_config_with_defaults() {
        let json = r#"{
            "task": "countdown",
            "model_dir": "/models/qwen3-0.6b",
            "trainer": { "steps": 5, "group_size": 8, "max_new_tokens": 48,
                         "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                         "lr": 1e-5, "weight_decay": 0.0,
                         "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.task, "countdown");
        assert!(matches!(cfg.device, DeviceSel::Cpu));
        assert_eq!(cfg.out_dir, PathBuf::from("runs"));
        assert_eq!(cfg.policy.lora_rank, 16);
        assert_eq!(cfg.data.train_n, 64);
        // The loader temperature mirrors the trainer's (cannot drift).
        assert!((cfg.loader_opts().temperature - cfg.trainer.temperature).abs() < f64::EPSILON);
    }

    /// `device` and `base_dtype` selectors deserialize from lowercase strings.
    #[test]
    fn device_and_dtype_selectors_parse() {
        let json = r#"{
            "task": "math",
            "model_dir": "/m",
            "device": "cuda",
            "policy": { "base_dtype": "bf16" },
            "data": { "path": "data.jsonl", "eval_n": 4 },
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                         "temperature": 0.7, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                         "lr": 1e-5, "weight_decay": 0.0,
                         "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(cfg.device, DeviceSel::Cuda));
        assert_eq!(cfg.loader_opts().base_dtype, DType::BF16);
        assert_eq!(cfg.data.path.as_deref(), Some(Path::new("data.jsonl")));
    }

    /// An unknown top-level key is rejected (typo guard).
    #[test]
    fn unknown_field_is_rejected() {
        let json = r#"{ "task": "countdown", "model_dir": "/m", "stpes": 5,
                        "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                          "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                          "lr": 1e-5, "weight_decay": 0.0,
                          "loss_type": "grpo", "scale_rewards": "group" } }"#;
        assert!(serde_json::from_str::<RunConfig>(json).is_err());
    }

    /// `math` without `data.path` is a clear contract error, not a panic.
    #[test]
    fn math_without_data_path_errors() {
        let json = r#"{ "task": "math", "model_dir": "/m",
                        "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                          "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                          "lr": 1e-5, "weight_decay": 0.0,
                          "loss_type": "grpo", "scale_rewards": "group" } }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.math_splits().is_err());
    }

    /// `runreport` resolves a directory to its `metrics.jsonl` but leaves a file path.
    #[test]
    fn metrics_path_resolution() {
        // A path that is not an existing directory is taken verbatim.
        assert_eq!(
            resolve_metrics_path(Path::new("some/metrics.jsonl")),
            PathBuf::from("some/metrics.jsonl")
        );
    }

    /// The clap surface parses both subcommands.
    #[test]
    fn clap_parses_subcommands() {
        let c = Cli::try_parse_from(["ferrl", "train", "--config", "run.json"]).unwrap();
        assert!(matches!(c.cmd, Command::Train(_)));
        let r =
            Cli::try_parse_from(["ferrl", "runreport", "runs/x", "--json", "--strict"]).unwrap();
        match r.cmd {
            Command::Runreport(a) => {
                assert!(a.json && a.strict);
            }
            Command::Train(_) => panic!("expected runreport"),
        }
    }
}
