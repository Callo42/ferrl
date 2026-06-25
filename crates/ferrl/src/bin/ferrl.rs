//! `ferrl` — the single-binary front door: train a built-in task end-to-end from a
//! JSON run config, and report on a finished run.
//!
//! ```text
//! ferrl train --config run.json     # GRPO-train a built-in task (countdown | math | trimul)
//! ferrl trimul-baseline --config run.json   # measure the TriMul reference baseline (ns) on this GPU
//! ferrl trimul-artifact --config run.json --completion raw.txt --out artifact/ ...
//! ferrl runreport <run-dir> [--json] [--strict]   # one-glance run health summary
//! ferrl perf-gate --baseline <run-dir> --candidate <run-dir>   # resource regression check
//! ```
//!
//! `train` reads a `RunConfig` (a serialized [`TrainerConfig`](ferrl::TrainerConfig)
//! plus a model directory, a device, and a task selector), loads a Qwen-family policy via
//! [`ferrl::load_auto_policy`], builds the named task's train/eval splits, and runs
//! the GRPO [`Trainer`](ferrl::Trainer). The task registry is closed (the worked
//! examples `countdown` and `math`, plus the `trimul` kernel-discovery task — which
//! runs a sandboxed GPU eval as its reward); a *custom* task is wired in Rust against
//! the library — see `examples/minimal_task.rs` and the README's "Wire your own task".
//!
//! `trimul-baseline` runs the bundled reference kernel through the same sandboxed eval
//! to measure its geometric-mean runtime on *this* node's GPU, and prints `{ns, gpu}`
//! to paste into the run config's `trimul.baseline` (the guarded-pin baseline — a
//! `train` run refuses a baseline measured on a different GPU than it is running on).
//!
//! `runreport` folds in the standalone run-summary tool: it reads a run's
//! `metrics.jsonl` and prints (or emits as JSON) a [`RunSummary`](ferrl::RunSummary),
//! optionally failing (`--strict`, exit code 2) when any health anomaly is flagged.
//!
//! `perf-gate` compares a baseline and candidate metrics stream, failing when
//! the update path goes dark or peak memory / step time exceed configured
//! regression thresholds.

// A CLI whose interface *is* its stdout/stderr; the library logs via `tracing`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use candle_core::{DType, Device};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use ferrl::countdown::{build_prompt, generate_dataset, CountdownConfig, CountdownProblem};
use ferrl::policy::GenConfig;
use ferrl::{
    compare_distributed_metrics, compare_metrics, evaluate, load_auto_policy, read_jsonl,
    summarize, train_eval_split, CountdownReward, LoaderOpts, MathProblem, MathReward,
    RegressionBudget, RegressionReport, RewardFn, RunDir, Sample, Trainer, TrainerConfig,
    TrimulReward,
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
    /// Measure the TriMul reference baseline (ns) on this node's GPU for the guarded pin.
    TrimulBaseline(TrimulBaselineArgs),
    /// Extract and verify a TriMul artifact bundle from a raw model completion.
    TrimulArtifact(Box<TrimulArtifactArgs>),
    /// Print a one-glance health summary for a finished run.
    Runreport(RunreportArgs),
    /// Compare two finished runs and fail on behavior/resource regression.
    PerfGate(PerfGateArgs),
}

/// Arguments for `ferrl train`.
#[derive(Debug, Args)]
struct TrainArgs {
    /// Path to the JSON run config (see `RunConfig`).
    #[arg(long)]
    config: PathBuf,
}

/// Arguments for `ferrl trimul-baseline`.
#[derive(Debug, Args)]
struct TrimulBaselineArgs {
    /// Path to the JSON run config (the same `trimul` block `ferrl train` reads).
    #[arg(long)]
    config: PathBuf,
}

/// Arguments for `ferrl trimul-artifact`.
#[derive(Debug, Args)]
struct TrimulArtifactArgs {
    /// Path to the JSON run config used for the discovery run.
    #[arg(long)]
    config: PathBuf,
    /// Raw model completion to extract `custom_kernel` from.
    #[arg(long)]
    completion: PathBuf,
    /// Output artifact directory. Fails if `manifest.json` already exists.
    #[arg(long)]
    out: PathBuf,
    /// Training run id or run directory name.
    #[arg(long)]
    run_id: String,
    /// Candidate optimizer step, when known from the run notes.
    #[arg(long, default_value_t = 0)]
    step: u64,
    /// Global prompt ordinal from `candidates.jsonl`.
    #[arg(long, default_value_t = 0)]
    prompt_index: u64,
    /// Candidate index within the sampled group, when known from the run notes.
    #[arg(long, default_value_t = 0)]
    group_index: u64,
    /// Data-parallel rank from `candidates.jsonl`.
    #[arg(long, default_value_t = 0)]
    rank: usize,
    /// Data-parallel world size from `candidates.jsonl`.
    #[arg(long, default_value_t = 1)]
    world_size: usize,
    /// Candidate training reward recorded when this candidate was selected.
    #[arg(long)]
    training_reward: f64,
    /// Audit seed for clean held-out re-verification. Must differ from training seed.
    #[arg(long)]
    audit_secret_seed: u64,
    /// Raw guarded-baseline measurements in ns; pass at least three values.
    #[arg(long = "baseline-ns", required = true)]
    baseline_measurements_ns: Vec<f64>,
    /// Exact baseline command used. Defaults to `ferrl trimul-baseline --config <config>`.
    #[arg(long)]
    baseline_command: Option<String>,
    /// Number of clean candidate verification re-runs.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Full ferrl git commit SHA for the training run.
    #[arg(long)]
    ferrl_commit: String,
    /// Training run health summary copied from `runreport` or run notes.
    #[arg(long)]
    run_health: String,
    /// Source inspection result for process/file/environment/network/path probes.
    #[arg(long, value_enum)]
    source_inspection: SourceInspectionResult,
    /// Human source-inspection notes covering process state, file descriptors,
    /// environment variables, network sockets, and paths outside kernel inputs.
    #[arg(long)]
    source_inspection_notes: String,
    /// Model family label for the artifact manifest.
    #[arg(long, default_value = "qwen3.x")]
    model_family: String,
    /// Operator-supplied checkpoint identity. Defaults to `model_dir`.
    #[arg(long)]
    checkpoint: Option<String>,
    /// Operator-supplied tokenizer identity. Defaults to `model_dir/tokenizer.json`.
    #[arg(long)]
    tokenizer: Option<String>,
    /// Immutable identity of the GPUMODE eval bundle. Defaults to `trimul.eval_dir`.
    #[arg(long)]
    eval_bundle: Option<String>,
    /// Immutable identity of the Apptainer image. Defaults to `trimul.image`.
    #[arg(long)]
    sandbox_image: Option<String>,
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

/// Arguments for `ferrl perf-gate`.
#[derive(Debug, Args)]
struct PerfGateArgs {
    /// Baseline run directory or `metrics.jsonl`. Repeat once per rank with
    /// `--distributed-world-max`.
    #[arg(long)]
    baseline: Vec<PathBuf>,
    /// Candidate run directory or `metrics.jsonl`. Repeat once per rank with
    /// `--distributed-world-max`.
    #[arg(long)]
    candidate: Vec<PathBuf>,
    /// Aggregate repeated baseline/candidate rank streams as one distributed world.
    #[arg(long)]
    distributed_world_max: bool,
    /// Required expected rank count when `--distributed-world-max` is set.
    #[arg(long)]
    distributed_world_size: Option<usize>,
    /// Maximum allowed candidate peak-memory regression versus baseline.
    #[arg(long, default_value_t = 0.0)]
    max_peak_mem_regression_pct: f64,
    /// Absolute peak-memory slack in bytes, added after the percent threshold.
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    peak_mem_slack_bytes: u64,
    /// Maximum allowed candidate mean-step-time regression versus baseline.
    #[arg(long, default_value_t = 10.0)]
    max_step_secs_regression_pct: f64,
    /// Absolute mean-step-time slack in seconds, added after the percent threshold.
    #[arg(long, default_value_t = 0.0)]
    step_secs_slack: f64,
    /// Minimum number of finite positive-grad steps required in each stream.
    #[arg(long, default_value_t = 1)]
    min_positive_grad_steps: usize,
    /// Optional bound for final grad-norm drift, relative to the baseline final grad norm.
    #[arg(long)]
    max_final_grad_norm_rel_drift: Option<f64>,
    /// Do not require CUDA memory telemetry to be present and within threshold.
    #[arg(long)]
    skip_memory_check: bool,
    /// Do not require step timing telemetry to be present and within threshold.
    #[arg(long)]
    skip_step_time_check: bool,
    /// Permit candidate health warnings to differ from the baseline.
    #[arg(long)]
    allow_health_warnings: bool,
    /// Emit the gate report as JSON.
    #[arg(long)]
    json: bool,
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
    /// The TriMul `task.yml` case list could not be loaded.
    #[error(transparent)]
    Trimul(#[from] ferrl::TrimulError),
    /// The trainer failed.
    #[error(transparent)]
    Trainer(#[from] ferrl::TrainerError),
    /// The held-out eval failed.
    #[error(transparent)]
    Eval(#[from] ferrl::EvalError),
    /// A run-directory / metrics IO error.
    #[error(transparent)]
    Telemetry(#[from] ferrl::telemetry::TelemetryError),
    /// A data-parallel collective or launch-configuration error.
    #[error(transparent)]
    Comm(#[from] ferrl::CommError),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
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

    /// Stable manifest spelling for this dtype.
    fn as_str(self) -> &'static str {
        match self {
            DtypeSel::F32 => "f32",
            DtypeSel::Bf16 => "bf16",
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
    /// Enable layer-boundary activation checkpointing for the update forward.
    ///
    /// This trades extra recompute for a lower activation peak and is the main
    /// CLI-accessible memory lever for long Qwen-family GPU training runs.
    activation_checkpointing: bool,
}

impl Default for PolicyCfg {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            base_dtype: DtypeSel::F32,
            seed: 1234,
            activation_checkpointing: false,
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

/// The reference baseline pin for the TriMul speedup reward: the reference
/// geometric-mean runtime (`ns`) and the GPU it was measured on (`gpu`). A *guarded
/// pin* — `gpu` must appear in this node's `nvidia-smi` product name, so a speedup is
/// never scored against a baseline taken on different hardware. Produce it with
/// `ferrl trimul-baseline`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineCfg {
    /// Reference geometric-mean runtime in nanoseconds (the speedup denominator).
    ns: f64,
    /// A label identifying the GPU the baseline was measured on. The intended value is
    /// the full product name `ferrl trimul-baseline` prints (e.g. `"NVIDIA H100 80GB
    /// HBM3"`); a shorter token like `"H100"` also works as long as it isn't a substring
    /// of a different card's name. Unknown keys are rejected so a typo can't silently
    /// disable the guard.
    gpu: String,
}

/// The prompt envelope used for the single TriMul discovery prompt.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TrimulPromptFormat {
    /// ferrl's original raw instruction prompt.
    #[default]
    Raw,
    /// Qwen3.5's native chat template, with thinking generation enabled.
    Qwen3_5ChatThinking,
    /// Qwen3.5 thinking prompt with a concise reasoning budget.
    Qwen3_5ChatThinkingConcise,
}

impl TrimulPromptFormat {
    /// Build the configured TriMul prompt.
    fn build_prompt(self, task_prompt: &str) -> String {
        match self {
            Self::Raw => ferrl::trimul::build_raw_prompt(task_prompt),
            Self::Qwen3_5ChatThinking => {
                ferrl::trimul::build_qwen3_5_chat_thinking_prompt(task_prompt)
            }
            Self::Qwen3_5ChatThinkingConcise => {
                ferrl::trimul::build_qwen3_5_chat_concise_thinking_prompt(task_prompt)
            }
        }
    }

    /// The extraction contract paired with this prompt format.
    fn submission_extract_mode(self) -> ferrl::trimul::SubmissionExtractMode {
        match self {
            Self::Raw => ferrl::trimul::SubmissionExtractMode::FinalFence,
            Self::Qwen3_5ChatThinking | Self::Qwen3_5ChatThinkingConcise => {
                ferrl::trimul::SubmissionExtractMode::ThinkingAfterThink
            }
        }
    }
}

/// TriMul task knobs (read only when `task == "trimul"`): the sandboxed eval image and
/// the pinned GPU Mode bundle, bounded scratch, the held-out secret seed, the
/// per-candidate wall budget, and the optional baseline pin. The concrete case list is
/// loaded at run time from `<eval_dir>/task.yml` (GPU Mode's, not vendored into this repo).
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TrimulCfg {
    /// Prompt wrapper for the single TriMul instruction.
    prompt_format: TrimulPromptFormat,
    /// Optional UTF-8 text appended to the TriMul instruction before chat wrapping.
    /// This lets private runs provide a complete reference/baseline example without
    /// vendoring third-party task materials into ferrl.
    prompt_suffix_path: Option<PathBuf>,
    /// The eval image — the pinned PyTorch+Triton `.sif`.
    image: PathBuf,
    /// The pinned GPU Mode eval bundle (`eval.py`/`reference.py`/`task.py`/`utils.py` +
    /// `task.yml`), bound read-only into the sandbox.
    eval_dir: PathBuf,
    /// Node-local scratch root for per-candidate dirs; prefer a tmpfs root such as
    /// `/dev/shm/ferrl`.
    scratch_root: PathBuf,
    /// Host-supervised total byte cap for one candidate's writable scratch tree
    /// (`0` -> the reward default, 1 GiB).
    scratch_max_bytes: u64,
    /// The held-out secret seed (`POPCORN_SEED`), combined with each case's public seed.
    secret_seed: u64,
    /// Per-candidate wall-clock budget in seconds (`0` → the reward default, 600 s).
    wall_secs: u64,
    /// The reference baseline pin (omit to fall back to an inverse-time reward).
    baseline: Option<BaselineCfg>,
}

/// Data-parallel launch knobs for `ferrl train`.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DistributedCfg {
    /// When true, run this process as one rank of a Slurm/NCCL data-parallel
    /// world. Requires `--features nccl`, `device = "cuda"`, and the Slurm
    /// variables plus `FERRL_NCCL_RENDEZVOUS` expected by `NcclConfig`.
    /// Run directories are rank-suffixed to keep per-rank telemetry separate.
    enabled: bool,
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
    /// Data-parallel launch knobs.
    #[serde(default)]
    distributed: DistributedCfg,
    /// TriMul task knobs (only read when `task == "trimul"`).
    #[serde(default)]
    trimul: TrimulCfg,
    /// The GRPO trainer config.
    trainer: TrainerConfig,
}

/// `serde` default for [`RunConfig::out_dir`]: `runs/`.
fn default_out_dir() -> PathBuf {
    PathBuf::from("runs")
}

impl RunConfig {
    fn open_device(&self) -> Result<Device, CliError> {
        if self.distributed.enabled {
            if self.device != DeviceSel::Cuda {
                return Err(CliError::msg(
                    "distributed.enabled requires device = \"cuda\"",
                ));
            }
            open_distributed_device()
        } else {
            self.device.open()
        }
    }

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

    /// A unique run id for this invocation: `<task>-<unix-seconds>` plus rank suffix under DP.
    fn run_id(&self) -> String {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let base = format!("{}-{stamp}", self.task);
        if self.distributed.enabled {
            let rank = std::env::var("SLURM_PROCID").unwrap_or_else(|_| "unknown".to_owned());
            format!("{base}-rank{rank}")
        } else {
            base
        }
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

    /// Build the TriMul train/eval splits: the single discovery prompt, repeated.
    ///
    /// Unlike countdown/math this does **not** use [`train_eval_split`]: that helper
    /// deduplicates whole samples, so a unit-target dataset of one repeated prompt would
    /// collapse to a single row. TriMul is one task — the generalization held out is over
    /// the *cases* (the secret seed inside the reward), not the prompt — and the trainer
    /// cycles prompts mod the train length, so a one-prompt train set *is* the
    /// single-task regime. `eval` (held-out) runs the same prompt through the reward, so a
    /// non-zero `data.eval_n` gives an adapter-vs-base reward comparison.
    fn trimul_splits(&self) -> Result<Splits<()>, CliError> {
        let mut task_prompt = ferrl::trimul::build_prompt();
        if let Some(path) = &self.trimul.prompt_suffix_path {
            let suffix = std::fs::read_to_string(path).map_err(|source| CliError::Io {
                path: path.clone(),
                source,
            })?;
            if !suffix.trim().is_empty() {
                task_prompt.push_str("\n\n");
                task_prompt.push_str(suffix.trim());
                task_prompt.push('\n');
            }
        }
        let prompt = self.trimul.prompt_format.build_prompt(&task_prompt);
        let train = std::iter::repeat_with(|| Sample::new(prompt.clone(), ()))
            .take(self.data.train_n.max(1))
            .collect();
        let eval = std::iter::repeat_with(|| Sample::new(prompt.clone(), ()))
            .take(self.data.eval_n)
            .collect();
        Ok((train, eval))
    }

    /// Build the TriMul reward *without* a baseline: load the case list from
    /// `<eval_dir>/task.yml`, and set the image, bundle, scratch, secret seed, and wall
    /// budget. This is the form `trimul-baseline` measures against; `train` layers the
    /// guarded baseline on top via [`build_trimul_reward`](Self::build_trimul_reward).
    fn build_trimul_reward_base(&self) -> Result<TrimulReward, CliError> {
        let t = &self.trimul;
        let (tests, benches) = ferrl::trimul::load_task_yml(t.eval_dir.join("task.yml"))?;
        let wall = Duration::from_secs(if t.wall_secs == 0 { 600 } else { t.wall_secs });
        let mut reward = TrimulReward::new(&t.image, &t.eval_dir, &t.scratch_root)
            .with_cases(tests, benches)
            .with_secret_seed(t.secret_seed)
            .with_wall(wall)
            .with_submission_extract_mode(t.prompt_format.submission_extract_mode());
        if t.scratch_max_bytes != 0 {
            reward = reward.with_scratch_max_bytes(t.scratch_max_bytes);
        }
        Ok(reward)
    }

    /// Build the TriMul reward for a `train` run: the base reward plus, when a baseline
    /// is pinned, the speedup denominator — guarded so the run is refused unless this
    /// node's GPU matches the GPU the baseline was measured on. With no baseline the
    /// reward falls back to an inverse-time signal (faster still scores higher).
    fn build_trimul_reward(&self) -> Result<TrimulReward, CliError> {
        let mut reward = self.build_trimul_reward_base()?;
        if let Some(b) = &self.trimul.baseline {
            guard_baseline_gpu(&b.gpu)?;
            reward = reward.with_baseline_ns(b.ns);
        }
        Ok(reward)
    }
}

/// Dispatch `ferrl train`: parse the config, open the device, build the named task's
/// data, and run training.
fn train(args: &TrainArgs) -> Result<(), CliError> {
    let _ = ferrl::init_tracing();
    let cfg = RunConfig::load(&args.config)?;
    let device = cfg.open_device()?;
    match cfg.task.as_str() {
        "countdown" => {
            let (train, eval) = cfg.countdown_splits();
            run_training(&cfg, &device, &CountdownReward::default(), &train, &eval)
        }
        "math" => {
            let (train, eval) = cfg.math_splits()?;
            run_training(&cfg, &device, &MathReward::default(), &train, &eval)
        }
        "trimul" => {
            let (train, eval) = cfg.trimul_splits()?;
            let reward = cfg.build_trimul_reward()?;
            run_training(&cfg, &device, &reward, &train, &eval)
        }
        other => Err(CliError::msg(format!(
            "unknown task {other:?}; built-in tasks are \"countdown\", \"math\", and \"trimul\""
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
    let (mut policy, tok) = load_auto_policy(&cfg.model_dir, device, &cfg.loader_opts())?;
    policy.set_activation_checkpointing(cfg.policy.activation_checkpointing);
    let tcfg = cfg.trainer.clone();
    let gen = GenConfig::from(&tcfg);
    info!(
        task = %cfg.task,
        steps = tcfg.steps,
        group_size = tcfg.group_size,
        activation_checkpointing = policy.activation_checkpointing(),
        train = train.len(),
        eval = eval.len(),
        "ferrl train: starting"
    );

    let run = RunDir::create(&cfg.out_dir, cfg.run_id())?;
    let mut trainer = open_trainer(tcfg, &run, cfg.distributed.enabled)?;
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

fn open_trainer(
    config: TrainerConfig,
    run: &RunDir,
    distributed: bool,
) -> Result<Trainer, CliError> {
    if distributed {
        open_distributed_trainer(config, run)
    } else {
        Ok(Trainer::new(config, run)?)
    }
}

#[cfg(feature = "nccl")]
fn open_distributed_device() -> Result<Device, CliError> {
    let nccl = ferrl::NcclConfig::from_env()?;
    let device = Device::new_cuda(nccl.local_rank())?;
    if let Some(w) = ferrl::check_driver_compat(&device).warning() {
        tracing::warn!("{w}");
    }
    ferrl::guard_first_kernel(&device)?;
    Ok(device)
}

#[cfg(not(feature = "nccl"))]
fn open_distributed_device() -> Result<Device, CliError> {
    Err(CliError::msg(
        "distributed.enabled requires building ferrl with --features nccl",
    ))
}

#[cfg(feature = "nccl")]
fn open_distributed_trainer(config: TrainerConfig, run: &RunDir) -> Result<Trainer, CliError> {
    Ok(Trainer::with_comm(
        config,
        run,
        ferrl::NcclComm::from_slurm_env()?,
    )?)
}

#[cfg(not(feature = "nccl"))]
fn open_distributed_trainer(_config: TrainerConfig, _run: &RunDir) -> Result<Trainer, CliError> {
    Err(CliError::msg(
        "distributed.enabled requires building ferrl with --features nccl",
    ))
}

/// This node's first GPU product name, read from `nvidia-smi`, or `None` if it cannot
/// be read (no `nvidia-smi`, a non-GPU node, or a query failure).
fn detect_gpu_name() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(ToString::to_string)
}

/// Whether `needle` appears in `haystack` as a whole token — bounded by a string edge
/// or a non-alphanumeric character on both sides — rather than a raw substring. Both
/// inputs must already be lowercased. This is stricter than `str::contains` on purpose:
/// `"a100"` matches `"nvidia a100 80gb"` and `"nvidia a100-sxm4"` but NOT `"a1000"`, and
/// `"l40"` does NOT match `"l40s"` — so a short GPU label can't false-match a different,
/// longer part number. An empty needle never matches.
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = haystack.as_bytes();
    haystack.match_indices(needle).any(|(i, m)| {
        let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        let after = i + m.len();
        let after_ok = after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
        before_ok && after_ok
    })
}

/// The guarded-pin check: the configured baseline GPU label must appear as a whole token
/// in this node's detected GPU name (case-insensitive — so the full `ferrl trimul-baseline`
/// product name matches exactly, and a short label like `"A100"` matches `"NVIDIA A100…"`
/// but not a different card like `"A1000"`). **Fails closed**: an empty label or an
/// unreadable GPU is an error, never a pass — so a speedup is never scored against a
/// baseline taken on different hardware.
fn baseline_gpu_matches(configured: &str, detected: Option<&str>) -> Result<(), String> {
    let want = configured.trim();
    if want.is_empty() {
        return Err(
            "trimul.baseline.gpu is empty; set it to the GPU label the baseline was \
             measured on (the full product name `ferrl trimul-baseline` prints)"
                .to_string(),
        );
    }
    let want_lc = want.to_lowercase();
    match detected {
        Some(name) if contains_word(&name.to_lowercase(), &want_lc) => Ok(()),
        Some(name) => Err(format!(
            "baseline was measured on GPU {want:?} but this node's GPU is {name:?}; \
             re-measure on this GPU (`ferrl trimul-baseline`) or fix trimul.baseline.gpu"
        )),
        None => Err(format!(
            "cannot read this node's GPU (nvidia-smi unavailable) to verify the baseline \
             was measured on GPU {want:?}; run on the target GPU node"
        )),
    }
}

/// Apply [`baseline_gpu_matches`] against the live `nvidia-smi` reading.
fn guard_baseline_gpu(configured: &str) -> Result<(), CliError> {
    baseline_gpu_matches(configured, detect_gpu_name().as_deref()).map_err(CliError::Msg)
}

/// Dispatch `ferrl trimul-baseline`: run the bundled reference kernel through the
/// sandboxed eval on this node's GPU, and print `{ "ns", "gpu" }` to paste into the run
/// config's `trimul.baseline` (the guarded pin).
fn trimul_baseline(args: &TrimulBaselineArgs) -> Result<(), CliError> {
    let _ = ferrl::init_tracing();
    let cfg = RunConfig::load(&args.config)?;
    // Measure against the un-pinned reward (we are producing the baseline, not using one).
    let reward = cfg.build_trimul_reward_base()?;
    let gpu = detect_gpu_name().ok_or_else(|| {
        CliError::msg(
            "cannot read this node's GPU (nvidia-smi unavailable); run on the target GPU node",
        )
    })?;
    let ns = reward
        .measure_reference_geomean_ns()
        .map_err(|e| CliError::msg(format!("baseline eval failed: {e}")))?
        .ok_or_else(|| {
            CliError::msg("the reference kernel produced no plausible benchmark time")
        })?;
    let pin = serde_json::json!({ "ns": ns, "gpu": gpu });
    println!(
        "{}",
        serde_json::to_string_pretty(&pin).unwrap_or_else(|_| pin.to_string())
    );
    eprintln!("ferrl: paste the above into your run config's trimul.baseline");
    Ok(())
}

/// One clean artifact-verification run written under `verification/`.
#[derive(Debug, Clone, Serialize)]
struct ArtifactVerificationRun {
    /// Whether the candidate passed every correctness case.
    correct: bool,
    /// Per-benchmark means, in ns.
    benchmark_means_ns: Vec<f64>,
    /// Geometric mean runtime, in ns.
    geomean_ns: Option<f64>,
    /// Speedup over the baseline median.
    speedup: Option<f64>,
}

/// Result of the reviewer-facing source inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum SourceInspectionResult {
    /// No process-state, file-descriptor, environment, network, or out-of-input
    /// path inspection was found.
    Clean,
    /// Source inspection found suspicious process-state, file-descriptor,
    /// environment, network, or out-of-input path access.
    Suspicious,
}

/// Reviewer-facing source-inspection record.
#[derive(Debug, Clone, Serialize)]
struct SourceInspectionManifest {
    /// Machine-readable source-inspection result.
    result: SourceInspectionResult,
    /// Human notes covering the inspected surfaces.
    notes: String,
}

/// Contract manifest written to `manifest.json`.
#[derive(Debug, Serialize)]
struct ArtifactManifest {
    /// Manifest schema version.
    contract_version: u32,
    /// The task this artifact targets.
    task: &'static str,
    /// Full ferrl commit SHA.
    ferrl_commit: String,
    /// Training run id.
    run_id: String,
    /// Candidate provenance.
    candidate: CandidateManifest,
    /// Model provenance.
    model: ModelManifest,
    /// Run-config provenance.
    config: ArtifactConfigManifest,
    /// Eval harness provenance.
    eval: EvalManifest,
    /// Same-GPU baseline record.
    baseline: BaselineManifest,
    /// Clean re-verification record.
    verification: VerificationManifest,
}

/// Candidate provenance fields.
#[derive(Debug, Serialize)]
struct CandidateManifest {
    /// Optimizer step where this candidate was sampled.
    step: u64,
    /// Global prompt ordinal where this candidate was sampled.
    prompt_index: u64,
    /// Candidate group index where this candidate was sampled.
    group_index: u64,
    /// Data-parallel rank that sampled this candidate.
    rank: usize,
    /// Data-parallel world size for the training run.
    world_size: usize,
    /// Training reward recorded when this candidate was selected.
    training_reward: f64,
    /// SHA-256 of the raw completion text.
    completion_sha256: String,
    /// SHA-256 of `submission.py`.
    source_sha256: String,
    /// Reviewer-facing source-inspection evidence.
    source_inspection: SourceInspectionManifest,
}

/// Model provenance fields.
#[derive(Debug, Serialize)]
struct ModelManifest {
    /// Model family label.
    family: String,
    /// Operator-supplied checkpoint identity.
    checkpoint: String,
    /// Operator-supplied tokenizer identity.
    tokenizer: String,
    /// `LoRA` rank.
    lora_rank: usize,
    /// `LoRA` alpha.
    lora_alpha: f64,
    /// Frozen base dtype.
    base_dtype: &'static str,
}

/// Run-config provenance fields.
#[derive(Debug, Serialize)]
struct ArtifactConfigManifest {
    /// SHA-256 of the run config bytes passed to this command.
    run_config_sha256: String,
    /// Trainer step budget.
    trainer_steps: u64,
    /// GRPO group size.
    group_size: usize,
    /// Training run health summary copied from `runreport` or run notes.
    run_health: String,
    /// Policy rollout seed.
    policy_seed: u64,
    /// Data seed.
    data_seed: u64,
    /// Secret seed used during training.
    training_secret_seed: u64,
    /// Secret seed used for artifact audit verification.
    audit_secret_seed: u64,
    /// Candidate scratch cap in bytes.
    scratch_max_bytes: u64,
}

/// Eval harness provenance fields.
#[derive(Debug, Serialize)]
struct EvalManifest {
    /// Immutable eval bundle identity.
    bundle: String,
    /// Immutable sandbox image identity.
    sandbox_image: String,
    /// Number of correctness cases loaded from `task.yml`.
    test_cases: usize,
    /// Number of benchmark cases loaded from `task.yml`.
    benchmark_cases: usize,
}

/// Same-GPU baseline fields.
#[derive(Debug, Serialize)]
struct BaselineManifest {
    /// GPU product name seen during extraction.
    gpu: String,
    /// Raw baseline measurements, in ns.
    measurements_ns: Vec<f64>,
    /// Median baseline runtime, in ns.
    median_ns: f64,
    /// Exact baseline command used for these measurements.
    command: String,
}

/// Verification summary fields.
#[derive(Debug, Serialize)]
struct VerificationManifest {
    /// GPU product name seen during extraction.
    gpu: String,
    /// Clean re-verification runs.
    runs: Vec<ArtifactVerificationRun>,
    /// Whether this bundle satisfies the mechanical artifact acceptance checks.
    accepted: bool,
}

/// Dispatch `ferrl trimul-artifact`: extract `custom_kernel` from a model completion,
/// re-verify it with an audit seed, and write the contract artifact bundle.
fn trimul_artifact(args: &TrimulArtifactArgs) -> Result<(), CliError> {
    let _ = ferrl::init_tracing();
    if args.repeats < 3 {
        return Err(CliError::msg(
            "trimul-artifact requires --repeats >= 3 for the first-run contract",
        ));
    }
    let config_bytes = read_bytes(&args.config)?;
    let cfg = parse_run_config(&args.config, &config_bytes)?;
    if cfg.task != "trimul" {
        return Err(CliError::msg(
            "trimul-artifact requires a config with task \"trimul\"",
        ));
    }
    if args.audit_secret_seed == cfg.trimul.secret_seed {
        return Err(CliError::msg(
            "audit secret seed must differ from trimul.secret_seed used during training",
        ));
    }
    let baseline = cfg.trimul.baseline.as_ref().ok_or_else(|| {
        CliError::msg("trimul-artifact requires trimul.baseline in the run config")
    })?;
    let baseline_median = median_checked(&args.baseline_measurements_ns, "baseline-ns")?;
    require_baseline_matches_config(baseline_median, baseline.ns)?;
    let gpu = detect_gpu_name().ok_or_else(|| {
        CliError::msg(
            "cannot read this node's GPU (nvidia-smi unavailable); run on the target GPU node",
        )
    })?;
    baseline_gpu_matches(&baseline.gpu, Some(&gpu)).map_err(CliError::Msg)?;

    let completion_bytes = read_bytes(&args.completion)?;
    let completion = String::from_utf8(completion_bytes.clone()).map_err(|e| {
        CliError::msg(format!(
            "completion file {} is not valid UTF-8: {e}",
            args.completion.display()
        ))
    })?;
    let mut reward = cfg.build_trimul_reward_base()?;
    let submission = reward.extract_submission(&completion).ok_or_else(|| {
        CliError::msg("completion does not contain a closed non-empty fenced code block")
    })?;

    reward = reward
        .with_secret_seed(args.audit_secret_seed)
        .with_baseline_ns(baseline_median);
    let (test_cases, benchmark_cases) =
        ferrl::trimul::load_task_yml(cfg.trimul.eval_dir.join("task.yml"))?;
    let runs = verify_submission_repeated(&reward, &submission, args.repeats)?;
    let accepted = accepted_artifact(&runs, baseline_median)
        && args.source_inspection == SourceInspectionResult::Clean;
    write_artifact_bundle(
        args,
        &cfg,
        &ArtifactInputs {
            gpu,
            completion: &completion,
            completion_bytes: &completion_bytes,
            config_bytes: &config_bytes,
            submission: &submission,
            baseline_median,
            test_cases: test_cases.len(),
            benchmark_cases: benchmark_cases.len(),
            runs,
            accepted,
        },
    )?;
    println!(
        "ferrl: wrote TriMul artifact bundle -> {}",
        args.out.display()
    );
    Ok(())
}

/// Values needed to write the artifact bundle.
struct ArtifactInputs<'a> {
    /// GPU product name.
    gpu: String,
    /// Raw completion string.
    completion: &'a str,
    /// Raw completion bytes.
    completion_bytes: &'a [u8],
    /// Raw config bytes.
    config_bytes: &'a [u8],
    /// Extracted source.
    submission: &'a str,
    /// Median baseline runtime, in ns.
    baseline_median: f64,
    /// Loaded correctness case count.
    test_cases: usize,
    /// Loaded benchmark case count.
    benchmark_cases: usize,
    /// Verification runs.
    runs: Vec<ArtifactVerificationRun>,
    /// Mechanical acceptance decision.
    accepted: bool,
}

/// Read `path` into bytes with CLI-shaped IO errors.
fn read_bytes(path: &Path) -> Result<Vec<u8>, CliError> {
    std::fs::read(path).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Parse a [`RunConfig`] from already-read bytes.
fn parse_run_config(path: &Path, bytes: &[u8]) -> Result<RunConfig, CliError> {
    serde_json::from_slice(bytes).map_err(|source| CliError::Config {
        path: path.to_path_buf(),
        source,
    })
}

/// Run clean verification `repeats` times.
fn verify_submission_repeated(
    reward: &TrimulReward,
    submission: &str,
    repeats: usize,
) -> Result<Vec<ArtifactVerificationRun>, CliError> {
    (0..repeats)
        .map(|_| {
            let v = reward
                .verify_submission(submission)
                .map_err(|e| CliError::msg(format!("artifact verification failed: {e}")))?;
            Ok(ArtifactVerificationRun {
                correct: v.correct,
                benchmark_means_ns: v.benchmark_means_ns,
                geomean_ns: v.geomean_ns,
                speedup: v.speedup,
            })
        })
        .collect()
}

/// Mechanical artifact acceptance: every re-run correct and timed, and the median
/// candidate runtime beats the median baseline.
fn accepted_artifact(runs: &[ArtifactVerificationRun], baseline_median: f64) -> bool {
    let geos: Vec<f64> = runs.iter().filter_map(|r| r.geomean_ns).collect();
    geos.len() == runs.len()
        && runs.iter().all(|r| r.correct)
        && median_checked(&geos, "candidate geomean")
            .is_ok_and(|candidate| candidate < baseline_median)
}

/// Write the full contract artifact bundle.
fn write_artifact_bundle(
    args: &TrimulArtifactArgs,
    cfg: &RunConfig,
    inputs: &ArtifactInputs<'_>,
) -> Result<(), CliError> {
    let manifest_path = args.out.join("manifest.json");
    if manifest_path.exists() {
        return Err(CliError::msg(format!(
            "{} already exists; refusing to overwrite an artifact",
            manifest_path.display()
        )));
    }
    std::fs::create_dir_all(args.out.join("verification")).map_err(|source| CliError::Io {
        path: args.out.clone(),
        source,
    })?;
    write_text(&args.out.join("submission.py"), inputs.submission)?;
    write_text(&args.out.join("completion.txt"), inputs.completion)?;
    for (i, run) in inputs.runs.iter().enumerate() {
        write_json(&args.out.join(format!("verification/run-{i:03}.json")), run)?;
    }
    let manifest = build_manifest(args, cfg, inputs);
    let manifest_json = json_pretty(&manifest_path, &manifest)?;
    write_text(&manifest_path, &manifest_json)?;
    let manifest_sha256 = sha256_hex(manifest_json.as_bytes());
    write_text(
        &args.out.join("report.md"),
        &artifact_report(&manifest, &args.out, &manifest_sha256),
    )?;
    Ok(())
}

/// Build the artifact manifest.
fn build_manifest(
    args: &TrimulArtifactArgs,
    cfg: &RunConfig,
    inputs: &ArtifactInputs<'_>,
) -> ArtifactManifest {
    ArtifactManifest {
        contract_version: 1,
        task: "trimul",
        ferrl_commit: args.ferrl_commit.clone(),
        run_id: args.run_id.clone(),
        candidate: CandidateManifest {
            step: args.step,
            prompt_index: args.prompt_index,
            group_index: args.group_index,
            rank: args.rank,
            world_size: args.world_size,
            training_reward: args.training_reward,
            completion_sha256: sha256_hex(inputs.completion_bytes),
            source_sha256: sha256_hex(inputs.submission.as_bytes()),
            source_inspection: SourceInspectionManifest {
                result: args.source_inspection,
                notes: args.source_inspection_notes.clone(),
            },
        },
        model: ModelManifest {
            family: args.model_family.clone(),
            checkpoint: args
                .checkpoint
                .clone()
                .unwrap_or_else(|| cfg.model_dir.display().to_string()),
            tokenizer: args
                .tokenizer
                .clone()
                .unwrap_or_else(|| cfg.model_dir.join("tokenizer.json").display().to_string()),
            lora_rank: cfg.policy.lora_rank,
            lora_alpha: cfg.policy.lora_alpha,
            base_dtype: cfg.policy.base_dtype.as_str(),
        },
        config: ArtifactConfigManifest {
            run_config_sha256: sha256_hex(inputs.config_bytes),
            trainer_steps: cfg.trainer.steps,
            group_size: cfg.trainer.group_size,
            run_health: args.run_health.clone(),
            policy_seed: cfg.policy.seed,
            data_seed: cfg.data.seed,
            training_secret_seed: cfg.trimul.secret_seed,
            audit_secret_seed: args.audit_secret_seed,
            scratch_max_bytes: trimul_scratch_cap(cfg),
        },
        eval: EvalManifest {
            bundle: args
                .eval_bundle
                .clone()
                .unwrap_or_else(|| cfg.trimul.eval_dir.display().to_string()),
            sandbox_image: args
                .sandbox_image
                .clone()
                .unwrap_or_else(|| cfg.trimul.image.display().to_string()),
            test_cases: inputs.test_cases,
            benchmark_cases: inputs.benchmark_cases,
        },
        baseline: BaselineManifest {
            gpu: inputs.gpu.clone(),
            measurements_ns: args.baseline_measurements_ns.clone(),
            median_ns: inputs.baseline_median,
            command: args.baseline_command.clone().unwrap_or_else(|| {
                format!("ferrl trimul-baseline --config {}", args.config.display())
            }),
        },
        verification: VerificationManifest {
            gpu: inputs.gpu.clone(),
            runs: inputs.runs.clone(),
            accepted: inputs.accepted,
        },
    }
}

/// The effective TriMul scratch cap in bytes.
fn trimul_scratch_cap(cfg: &RunConfig) -> u64 {
    if cfg.trimul.scratch_max_bytes == 0 {
        1 << 30
    } else {
        cfg.trimul.scratch_max_bytes
    }
}

/// Write UTF-8 text to `path`.
fn write_text(path: &Path, text: &str) -> Result<(), CliError> {
    std::fs::write(path, text).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Pretty-write JSON to `path`.
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), CliError> {
    let json = json_pretty(path, value)?;
    write_text(path, &json)
}

/// Render pretty JSON for `path` so callers can hash the exact bytes they write.
fn json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<String, CliError> {
    serde_json::to_string_pretty(value)
        .map_err(|e| CliError::msg(format!("serialize {}: {e}", path.display())))
}

/// A contract-shaped human report next to the machine manifest.
fn artifact_report(
    manifest: &ArtifactManifest,
    artifact_dir: &Path,
    manifest_sha256: &str,
) -> String {
    let median_candidate = median_checked(
        &manifest
            .verification
            .runs
            .iter()
            .filter_map(|r| r.geomean_ns)
            .collect::<Vec<_>>(),
        "candidate geomean",
    )
    .ok();
    let speedup = median_candidate.map(|c| manifest.baseline.median_ns / c);
    let verdict = if manifest.verification.accepted {
        "accepted_artifact"
    } else {
        "invalid_run"
    };
    let clean_correct = manifest
        .verification
        .runs
        .iter()
        .filter(|r| r.correct)
        .count();
    let clean_total = manifest.verification.runs.len();
    let decision = artifact_accept_reason(manifest, median_candidate);
    let baseline_measurements = manifest
        .baseline
        .measurements_ns
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    let candidate_median =
        median_candidate.map_or_else(|| "none".to_string(), |v| format!("{v:.6}"));
    let speedup = speedup.map_or_else(|| "none".to_string(), |v| format!("{v:.6}"));
    let source_inspection = source_inspection_label(manifest.candidate.source_inspection.result);

    let mut out = String::new();
    writeln!(&mut out, "# TriMul Artifact Report\n").expect("writing to String cannot fail");
    writeln!(&mut out, "## 1. Verdict\n").expect("writing to String cannot fail");
    writeln!(&mut out, "{verdict}\n").expect("writing to String cannot fail");

    writeln!(&mut out, "## 2. Baseline\n").expect("writing to String cannot fail");
    writeln!(&mut out, "- GPU: {}", manifest.baseline.gpu).expect("writing to String cannot fail");
    writeln!(&mut out, "- Raw measurements ns: {baseline_measurements}")
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Median runtime ns: {:.6}",
        manifest.baseline.median_ns
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Command used: `{}`\n",
        manifest.baseline.command
    )
    .expect("writing to String cannot fail");

    writeln!(&mut out, "## 3. Training\n").expect("writing to String cannot fail");
    writeln!(&mut out, "- ferrl commit: {}", manifest.ferrl_commit)
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Config hash: {}",
        manifest.config.run_config_sha256
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Model: family={}, checkpoint={}, tokenizer={}, lora_rank={}, lora_alpha={}, base_dtype={}",
        manifest.model.family,
        manifest.model.checkpoint,
        manifest.model.tokenizer,
        manifest.model.lora_rank,
        manifest.model.lora_alpha,
        manifest.model.base_dtype
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Seeds: data={}, policy={}, training_secret={}, audit_secret={}",
        manifest.config.data_seed,
        manifest.config.policy_seed,
        manifest.config.training_secret_seed,
        manifest.config.audit_secret_seed
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Budget: trainer_steps={}, group_size={}, scratch_max_bytes={}",
        manifest.config.trainer_steps,
        manifest.config.group_size,
        manifest.config.scratch_max_bytes
    )
    .expect("writing to String cannot fail");
    writeln!(&mut out, "- Run health: {}\n", manifest.config.run_health)
        .expect("writing to String cannot fail");

    writeln!(&mut out, "## 4. Candidate Table\n").expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "| source hash | training reward | source inspection | clean correctness | median runtime ns | speedup | accept/reject reason |"
    )
    .expect("writing to String cannot fail");
    writeln!(&mut out, "|---|---:|---|---:|---:|---:|---|").expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "| {} | {:.6} | {} | {}/{} | {} | {} | {} |\n",
        manifest.candidate.source_sha256,
        manifest.candidate.training_reward,
        source_inspection,
        clean_correct,
        clean_total,
        candidate_median,
        speedup,
        decision
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "Source inspection notes: {}\n",
        manifest.candidate.source_inspection.notes
    )
    .expect("writing to String cannot fail");

    writeln!(&mut out, "## 5. Artifact Bundle\n").expect("writing to String cannot fail");
    writeln!(&mut out, "- Path: {}", artifact_dir.display())
        .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Manifest path: {}/manifest.json",
        artifact_dir.display()
    )
    .expect("writing to String cannot fail");
    writeln!(&mut out, "- Manifest SHA-256: {manifest_sha256}\n")
        .expect("writing to String cannot fail");

    writeln!(&mut out, "## 6. Reviewer Checklist\n").expect("writing to String cannot fail");
    push_check(&mut out, manifest.task == "trimul", "task is trimul");
    push_check(
        &mut out,
        !manifest.ferrl_commit.trim().is_empty(),
        "ferrl commit recorded",
    );
    push_check(
        &mut out,
        !manifest.config.run_config_sha256.is_empty(),
        "config hash recorded",
    );
    push_check(
        &mut out,
        manifest.baseline.measurements_ns.len() >= 3,
        "raw baseline has at least three measurements",
    );
    push_check(
        &mut out,
        manifest.baseline.gpu == manifest.verification.gpu,
        "baseline and verification GPU match",
    );
    push_check(
        &mut out,
        manifest.config.audit_secret_seed != manifest.config.training_secret_seed,
        "audit seed differs from training seed",
    );
    push_check(
        &mut out,
        clean_total >= 3,
        "at least three clean verification runs",
    );
    push_check(
        &mut out,
        clean_correct == clean_total,
        "every verification run is correct",
    );
    push_check(
        &mut out,
        manifest
            .verification
            .runs
            .iter()
            .all(|r| r.geomean_ns.is_some()),
        "every verification run is timed",
    );
    push_check(
        &mut out,
        median_candidate.is_some_and(|v| v < manifest.baseline.median_ns),
        "candidate median beats baseline median",
    );
    push_check(
        &mut out,
        manifest.candidate.source_inspection.result == SourceInspectionResult::Clean,
        "source inspection found no process/file/env/network/path probing",
    );
    push_check(
        &mut out,
        !manifest.candidate.source_inspection.notes.trim().is_empty(),
        "source inspection notes recorded",
    );
    push_check(
        &mut out,
        !manifest.eval.bundle.trim().is_empty(),
        "eval bundle identity recorded",
    );
    push_check(
        &mut out,
        !manifest.eval.sandbox_image.trim().is_empty(),
        "sandbox image identity recorded",
    );
    push_check(
        &mut out,
        manifest.config.scratch_max_bytes > 0,
        "scratch cap recorded",
    );
    push_check(
        &mut out,
        !manifest_sha256.trim().is_empty(),
        "manifest hash recorded",
    );
    out
}

/// Human-readable accept/reject reason for the candidate table.
fn artifact_accept_reason(
    manifest: &ArtifactManifest,
    median_candidate: Option<f64>,
) -> &'static str {
    if manifest.verification.accepted {
        "accepted: all clean runs correct and median runtime beats baseline"
    } else if manifest.candidate.source_inspection.result == SourceInspectionResult::Suspicious {
        "rejected: source inspection found process/file/env/network/path probing"
    } else if manifest.verification.runs.iter().any(|r| !r.correct) {
        "rejected: at least one clean verification run failed correctness"
    } else if manifest
        .verification
        .runs
        .iter()
        .any(|r| r.geomean_ns.is_none())
    {
        "rejected: at least one clean verification run did not produce timing"
    } else if median_candidate.is_some_and(|v| v >= manifest.baseline.median_ns) {
        "rejected: candidate median runtime does not beat baseline"
    } else {
        "rejected: insufficient clean verification evidence"
    }
}

/// Stable report label for source inspection results.
fn source_inspection_label(result: SourceInspectionResult) -> &'static str {
    match result {
        SourceInspectionResult::Clean => "clean",
        SourceInspectionResult::Suspicious => "suspicious",
    }
}

/// Append a reviewer checklist row.
fn push_check(out: &mut String, pass: bool, label: &str) {
    writeln!(out, "- [{}] {label}", if pass { "pass" } else { "fail" })
        .expect("writing to String cannot fail");
}

/// SHA-256 hex digest of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        write!(&mut out, "{b:02x}").expect("writing to String cannot fail");
    }
    out
}

/// Median of positive finite values. Requires at least three values for first-run
/// timing discipline.
fn median_checked(values: &[f64], label: &str) -> Result<f64, CliError> {
    if values.len() < 3 {
        return Err(CliError::msg(format!(
            "{label} requires at least three measurements"
        )));
    }
    if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(CliError::msg(format!(
            "{label} measurements must be positive finite values"
        )));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(sorted[sorted.len() / 2])
}

/// Require the raw baseline median to match the config's guarded baseline pin.
fn require_baseline_matches_config(median: f64, pinned: f64) -> Result<(), CliError> {
    let tol = (pinned.abs().max(median.abs()) * 1e-9).max(1e-6);
    if (median - pinned).abs() <= tol {
        Ok(())
    } else {
        Err(CliError::msg(format!(
            "median --baseline-ns ({median}) does not match trimul.baseline.ns ({pinned})"
        )))
    }
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

/// Dispatch `ferrl perf-gate`: compare baseline and candidate metrics streams.
fn perf_gate(args: &PerfGateArgs) -> Result<ExitCode, CliError> {
    let budget = perf_budget(args)?;
    let report = if args.distributed_world_max {
        if args.baseline.is_empty() || args.candidate.is_empty() {
            return Err(CliError::msg(
                "--distributed-world-max requires at least one --baseline and one --candidate",
            ));
        }
        if args.baseline.len() != args.candidate.len() {
            return Err(CliError::msg(format!(
                "--distributed-world-max requires matching rank counts: baseline={} candidate={}",
                args.baseline.len(),
                args.candidate.len()
            )));
        }
        let Some(expected) = args.distributed_world_size else {
            return Err(CliError::msg(
                "--distributed-world-max requires --distributed-world-size",
            ));
        };
        if expected == 0 {
            return Err(CliError::msg("--distributed-world-size must be positive"));
        }
        if args.baseline.len() != expected {
            return Err(CliError::msg(format!(
                "--distributed-world-size {expected} does not match supplied ranks: \
                 baseline={} candidate={}",
                args.baseline.len(),
                args.candidate.len()
            )));
        }
        let baseline = read_metrics_inputs(&args.baseline)?;
        let candidate = read_metrics_inputs(&args.candidate)?;
        compare_distributed_metrics(&baseline, &candidate, &budget)
    } else {
        if args.baseline.len() != 1 || args.candidate.len() != 1 {
            return Err(CliError::msg(
                "perf-gate requires exactly one --baseline and one --candidate unless \
                 --distributed-world-max is set",
            ));
        }
        let baseline = ferrl::read_metrics(resolve_metrics_path(&args.baseline[0]))?;
        let candidate = ferrl::read_metrics(resolve_metrics_path(&args.candidate[0]))?;
        compare_metrics(&baseline, &candidate, &budget)
    };
    if args.json {
        let s = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::msg(format!("serialize perf gate: {e}")))?;
        println!("{s}");
    } else {
        print_perf_gate_report(&report);
    }
    if report.passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(2))
    }
}

fn read_metrics_inputs(paths: &[PathBuf]) -> Result<Vec<Vec<ferrl::Metrics>>, CliError> {
    Ok(paths
        .iter()
        .map(|path| ferrl::read_metrics(resolve_metrics_path(path)))
        .collect::<Result<Vec<_>, _>>()?)
}

fn perf_budget(args: &PerfGateArgs) -> Result<RegressionBudget, CliError> {
    for (label, value) in [
        (
            "--max-peak-mem-regression-pct",
            args.max_peak_mem_regression_pct,
        ),
        (
            "--max-step-secs-regression-pct",
            args.max_step_secs_regression_pct,
        ),
        ("--step-secs-slack", args.step_secs_slack),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(CliError::msg(format!("{label} must be finite and >= 0")));
        }
    }
    if let Some(value) = args.max_final_grad_norm_rel_drift {
        if !value.is_finite() || value < 0.0 {
            return Err(CliError::msg(
                "--max-final-grad-norm-rel-drift must be finite and >= 0",
            ));
        }
    }
    if args.min_positive_grad_steps == 0 {
        return Err(CliError::msg(
            "--min-positive-grad-steps must be >= 1 for the strict perf gate",
        ));
    }
    Ok(RegressionBudget {
        require_live_update: true,
        require_timing: !args.skip_step_time_check,
        require_cuda_memory: !args.skip_memory_check,
        allow_health_warnings: args.allow_health_warnings,
        warmup_steps: 0,
        min_positive_grad_steps: args.min_positive_grad_steps,
        max_mean_step_secs_ratio: 1.0 + (args.max_step_secs_regression_pct as f32 / 100.0),
        max_mean_step_secs_abs_slack: args.step_secs_slack as f32,
        max_cuda_peak_used_ratio: 1.0 + args.max_peak_mem_regression_pct / 100.0,
        max_cuda_peak_used_abs_slack_bytes: args.peak_mem_slack_bytes,
        max_cuda_peak_delta_ratio: None,
        max_cuda_peak_delta_abs_slack_bytes: args.peak_mem_slack_bytes,
        max_final_grad_norm_rel_drift: args.max_final_grad_norm_rel_drift.map(|v| v as f32),
    })
}

fn print_perf_gate_report(report: &RegressionReport) {
    let verdict = if report.passed { "PASS" } else { "FAIL" };
    println!("perf gate — {verdict}");
    print_summary_line("baseline", report.baseline.as_ref());
    print_summary_line("candidate", report.candidate.as_ref());
    for failure in &report.failures {
        println!("  FAIL {failure}");
    }
}

fn print_summary_line(label: &str, summary: Option<&ferrl::RunSummary>) {
    let Some(summary) = summary else {
        println!("  {label:<9} <no metrics>");
        return;
    };
    println!(
        "  {label:<9} steps={} peak={}MiB delta={}MiB step={:.3}s grad={:.6}",
        summary.steps,
        summary.max_cuda_mem_peak_used_bytes / (1024 * 1024),
        summary.max_cuda_mem_peak_delta_bytes / (1024 * 1024),
        summary.mean_step_secs,
        summary.final_grad_norm
    );
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
        Command::TrimulBaseline(args) => trimul_baseline(args).map(|()| ExitCode::SUCCESS),
        Command::TrimulArtifact(args) => trimul_artifact(args).map(|()| ExitCode::SUCCESS),
        Command::Runreport(args) => runreport(args),
        Command::PerfGate(args) => perf_gate(args),
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
        assert!(!cfg.policy.activation_checkpointing);
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
            "policy": { "base_dtype": "bf16", "activation_checkpointing": true },
            "data": { "path": "data.jsonl", "eval_n": 4 },
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                         "temperature": 0.7, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                         "lr": 1e-5, "weight_decay": 0.0,
                         "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(cfg.device, DeviceSel::Cuda));
        assert_eq!(cfg.loader_opts().base_dtype, DType::BF16);
        assert!(cfg.policy.activation_checkpointing);
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

    /// The clap surface parses every subcommand.
    #[test]
    fn clap_parses_subcommands() {
        let c = Cli::try_parse_from(["ferrl", "train", "--config", "run.json"]).unwrap();
        assert!(matches!(c.cmd, Command::Train(_)));
        // The `TrimulBaseline` variant renders as the `trimul-baseline` subcommand.
        let b = Cli::try_parse_from(["ferrl", "trimul-baseline", "--config", "run.json"]).unwrap();
        assert!(matches!(b.cmd, Command::TrimulBaseline(_)));
        let r =
            Cli::try_parse_from(["ferrl", "runreport", "runs/x", "--json", "--strict"]).unwrap();
        match r.cmd {
            Command::Runreport(a) => {
                assert!(a.json && a.strict);
            }
            _ => panic!("expected runreport"),
        }
    }

    /// The clap surface parses the performance-regression gate.
    #[test]
    fn clap_parses_perf_gate() {
        let p = Cli::try_parse_from([
            "ferrl",
            "perf-gate",
            "--baseline",
            "main/rank0",
            "--candidate",
            "pr/rank0",
            "--max-peak-mem-regression-pct",
            "1.5",
            "--max-step-secs-regression-pct",
            "5",
            "--max-final-grad-norm-rel-drift",
            "0.001",
            "--json",
        ])
        .unwrap();
        let a = expect_perf_gate(p.cmd);
        assert_eq!(a.baseline, vec![PathBuf::from("main/rank0")]);
        assert_eq!(a.candidate, vec![PathBuf::from("pr/rank0")]);
        assert!(!a.distributed_world_max);
        assert!(a.json);
    }

    #[test]
    fn perf_gate_budget_reflects_cli_thresholds() {
        let args = PerfGateArgs {
            max_peak_mem_regression_pct: 1.5,
            max_step_secs_regression_pct: 5.0,
            max_final_grad_norm_rel_drift: Some(0.001),
            json: true,
            ..perf_gate_test_args()
        };
        let budget = perf_budget(&args).unwrap();
        assert!(budget.require_cuda_memory);
        assert!(budget.require_timing);
        assert_eq!(budget.max_cuda_peak_used_ratio, 1.015);
        assert_eq!(budget.max_mean_step_secs_ratio, 1.05);
        assert_eq!(budget.max_final_grad_norm_rel_drift, Some(0.001));
    }

    #[test]
    fn clap_parses_distributed_perf_gate() {
        let p = Cli::try_parse_from([
            "ferrl",
            "perf-gate",
            "--distributed-world-max",
            "--baseline",
            "main/rank0",
            "--baseline",
            "main/rank1",
            "--candidate",
            "pr/rank0",
            "--candidate",
            "pr/rank1",
        ])
        .unwrap();
        let a = expect_perf_gate(p.cmd);
        assert_eq!(
            a.baseline,
            vec![PathBuf::from("main/rank0"), PathBuf::from("main/rank1")]
        );
        assert_eq!(
            a.candidate,
            vec![PathBuf::from("pr/rank0"), PathBuf::from("pr/rank1")]
        );
        assert!(a.distributed_world_max);
        assert_eq!(a.distributed_world_size, None);
    }

    #[test]
    fn clap_parses_distributed_world_size() {
        let p = Cli::try_parse_from([
            "ferrl",
            "perf-gate",
            "--distributed-world-max",
            "--distributed-world-size",
            "2",
            "--baseline",
            "main/rank0",
            "--baseline",
            "main/rank1",
            "--candidate",
            "pr/rank0",
            "--candidate",
            "pr/rank1",
        ])
        .unwrap();
        let a = expect_perf_gate(p.cmd);
        assert_eq!(a.distributed_world_size, Some(2));
    }

    #[test]
    fn perf_gate_rejects_repeated_rank_paths_without_distributed_mode() {
        let mut args = perf_gate_test_args();
        args.baseline.push(PathBuf::from("main/rank1"));
        args.candidate.push(PathBuf::from("pr/rank1"));

        let err = perf_gate(&args).unwrap_err().to_string();
        assert!(
            err.contains("exactly one --baseline"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn perf_gate_rejects_distributed_mode_without_world_size() {
        let mut args = perf_gate_test_args();
        args.distributed_world_max = true;

        let err = perf_gate(&args).unwrap_err().to_string();
        assert!(
            err.contains("--distributed-world-size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn perf_gate_rejects_missing_expected_distributed_rank() {
        let mut args = perf_gate_test_args();
        args.distributed_world_max = true;
        args.distributed_world_size = Some(2);

        let err = perf_gate(&args).unwrap_err().to_string();
        assert!(
            err.contains("--distributed-world-size 2"),
            "unexpected error: {err}"
        );
    }

    fn expect_perf_gate(cmd: Command) -> PerfGateArgs {
        match cmd {
            Command::PerfGate(a) => a,
            _ => panic!("expected perf-gate"),
        }
    }

    #[test]
    fn perf_gate_rejects_zero_positive_grad_requirement() {
        let mut args = perf_gate_test_args();
        args.min_positive_grad_steps = 0;
        let err = perf_budget(&args).unwrap_err().to_string();
        assert!(
            err.contains("--min-positive-grad-steps"),
            "unexpected error: {err}"
        );
    }

    fn perf_gate_test_args() -> PerfGateArgs {
        PerfGateArgs {
            baseline: vec![PathBuf::from("main/rank0")],
            candidate: vec![PathBuf::from("pr/rank0")],
            distributed_world_max: false,
            distributed_world_size: None,
            max_peak_mem_regression_pct: 0.0,
            peak_mem_slack_bytes: 0,
            max_step_secs_regression_pct: 10.0,
            step_secs_slack: 0.0,
            min_positive_grad_steps: 1,
            max_final_grad_norm_rel_drift: None,
            skip_memory_check: false,
            skip_step_time_check: false,
            allow_health_warnings: false,
            json: false,
        }
    }

    /// The clap surface parses the artifact subcommand.
    #[test]
    fn clap_parses_trimul_artifact() {
        let a = Cli::try_parse_from([
            "ferrl",
            "trimul-artifact",
            "--config",
            "run.json",
            "--completion",
            "completion.txt",
            "--out",
            "artifact",
            "--run-id",
            "trimul-1",
            "--prompt-index",
            "5",
            "--group-index",
            "1",
            "--rank",
            "0",
            "--world-size",
            "1",
            "--training-reward",
            "1.25",
            "--run-health",
            "healthy",
            "--source-inspection",
            "clean",
            "--source-inspection-notes",
            "no process, file descriptor, environment, network, or out-of-input path probes",
            "--audit-secret-seed",
            "99",
            "--baseline-ns",
            "10",
            "--baseline-ns",
            "11",
            "--baseline-ns",
            "12",
            "--ferrl-commit",
            "abc123",
        ])
        .unwrap();
        match a.cmd {
            Command::TrimulArtifact(a) => {
                assert_eq!(
                    (a.prompt_index, a.group_index, a.rank, a.world_size),
                    (5, 1, 0, 1)
                );
            }
            _ => panic!("expected trimul-artifact"),
        }
    }

    /// A `trimul` run config parses, with its task block and a baseline pin.
    #[test]
    fn parses_a_trimul_config() {
        let json = r#"{ "task": "trimul", "model_dir": "/m",
                        "device": "cuda",
                        "data": { "train_n": 8, "eval_n": 2 },
                        "trimul": { "image": "/img.sif", "eval_dir": "/eval",
                          "scratch_root": "/tmp", "scratch_max_bytes": 1048576,
                          "secret_seed": 123, "wall_secs": 300,
                          "baseline": { "ns": 5200000.0, "gpu": "H100" } },
                        "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                          "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                          "lr": 1e-5, "weight_decay": 0.0,
                          "loss_type": "grpo", "scale_rewards": "group" } }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.task, "trimul");
        assert_eq!((cfg.trimul.secret_seed, cfg.trimul.wall_secs), (123, 300));
        assert_eq!(cfg.trimul.scratch_max_bytes, 1_048_576);
        let b = cfg.trimul.baseline.as_ref().expect("baseline present");
        assert_eq!((b.ns, b.gpu.as_str()), (5_200_000.0, "H100"));
        // The single-prompt splits honour train_n / eval_n without deduping to one row.
        let (train, eval) = cfg.trimul_splits().unwrap();
        assert_eq!((train.len(), eval.len()), (8, 2));
        assert!(train[0].prompt.contains("custom_kernel"));
    }

    /// The concise Qwen3.5 thinking prompt is opt-in and keeps the thinking extractor.
    #[test]
    #[allow(clippy::cognitive_complexity)] // assertion-heavy regression over config, prompt, and extractor
    fn trimul_config_selects_concise_qwen35_thinking_prompt() {
        let json = r#"{ "task": "trimul", "model_dir": "/m",
                        "trimul": { "prompt_format": "qwen3_5_chat_thinking_concise" },
                        "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                          "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                          "lr": 1e-5, "weight_decay": 0.0,
                          "loss_type": "grpo", "scale_rewards": "group" } }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        let (train, eval) = cfg.trimul_splits().unwrap();
        assert_eq!((train.len(), eval.len()), (64, 0));
        assert!(matches!(
            cfg.trimul.prompt_format.submission_extract_mode(),
            ferrl::trimul::SubmissionExtractMode::ThinkingAfterThink
        ));
        assert!(train[0].prompt.starts_with("<|im_start|>system\n"));
        assert!(train[0]
            .prompt
            .ends_with("<|im_start|>assistant\n<think>\n"));
        for needle in [
            "Use at most 8 short reasoning lines inside <think>.",
            "Prefer a complete valid implementation over further analysis.",
            "custom_kernel(data)",
        ] {
            assert!(train[0].prompt.contains(needle), "missing {needle:?}");
        }
    }

    /// A `trimul` config with no `trimul` block still parses (the defaults), and the
    /// other tasks parse without a `trimul` block at all.
    #[test]
    fn trimul_block_defaults_when_omitted() {
        let json = r#"{ "task": "countdown", "model_dir": "/m",
                        "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                          "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                          "lr": 1e-5, "weight_decay": 0.0,
                          "loss_type": "grpo", "scale_rewards": "group" } }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.trimul.baseline.is_none());
        assert_eq!(cfg.trimul.wall_secs, 0);
    }

    /// The guarded-pin GPU check: a label that is a substring of the detected name
    /// passes; a different GPU or an unreadable GPU fails closed.
    #[test]
    fn baseline_gpu_guard_matches_and_fails_closed() {
        // A label matches as a whole token of the full product name.
        assert!(baseline_gpu_matches("H100", Some("NVIDIA H100 80GB HBM3")).is_ok());
        assert!(baseline_gpu_matches("l40s", Some("NVIDIA L40S")).is_ok());
        // A different GPU is refused.
        assert!(baseline_gpu_matches("H100", Some("NVIDIA L40S")).is_err());
        // An unreadable GPU fails closed (never silently passes).
        assert!(baseline_gpu_matches("H100", None).is_err());
    }

    /// The guard is token-bounded (not a raw substring) and rejects an empty label, so a
    /// short or blank `baseline.gpu` cannot silently match the wrong card or disable the
    /// check.
    #[test]
    fn baseline_gpu_guard_rejects_lookalikes_and_empty() {
        // A short label must not match a longer, different part number.
        assert!(baseline_gpu_matches("A100", Some("NVIDIA A1000")).is_err());
        assert!(baseline_gpu_matches("L40", Some("NVIDIA L40S")).is_err());
        // …but still matches its real card (token bounded by space/hyphen).
        assert!(baseline_gpu_matches("A100", Some("NVIDIA A100-SXM4-80GB")).is_ok());
        // An empty / whitespace label fails closed.
        assert!(baseline_gpu_matches("", Some("NVIDIA L40S")).is_err());
        assert!(baseline_gpu_matches("   ", Some("NVIDIA L40S")).is_err());
    }

    #[test]
    fn sha256_hex_is_stable() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn median_checked_requires_three_positive_values() {
        assert_eq!(median_checked(&[3.0, 1.0, 2.0], "x").unwrap(), 2.0);
        assert!(median_checked(&[1.0, 2.0], "x").is_err());
        assert!(median_checked(&[1.0, f64::NAN, 3.0], "x").is_err());
        assert!(median_checked(&[1.0, 0.0, 3.0], "x").is_err());
    }

    #[test]
    fn baseline_median_must_match_config_pin() {
        assert!(require_baseline_matches_config(10.0, 10.0).is_ok());
        assert!(require_baseline_matches_config(10.0, 11.0).is_err());
    }

    #[test]
    fn artifact_report_matches_the_contract_outline() {
        let manifest = ArtifactManifest {
            contract_version: 1,
            task: "trimul",
            ferrl_commit: "abc123".to_string(),
            run_id: "trimul-1".to_string(),
            candidate: CandidateManifest {
                step: 7,
                prompt_index: 12,
                group_index: 2,
                rank: 0,
                world_size: 1,
                training_reward: 1.5,
                completion_sha256: "completion-hash".to_string(),
                source_sha256: "source-hash".to_string(),
                source_inspection: SourceInspectionManifest {
                    result: SourceInspectionResult::Clean,
                    notes: "no process, file descriptor, environment, network, or out-of-input path probes"
                        .to_string(),
                },
            },
            model: ModelManifest {
                family: "qwen3.x".to_string(),
                checkpoint: "checkpoint".to_string(),
                tokenizer: "tokenizer".to_string(),
                lora_rank: 8,
                lora_alpha: 16.0,
                base_dtype: "bf16",
            },
            config: ArtifactConfigManifest {
                run_config_sha256: "config-hash".to_string(),
                trainer_steps: 100,
                group_size: 4,
                run_health: "healthy".to_string(),
                policy_seed: 11,
                data_seed: 22,
                training_secret_seed: 33,
                audit_secret_seed: 44,
                scratch_max_bytes: 1024,
            },
            eval: EvalManifest {
                bundle: "eval-bundle".to_string(),
                sandbox_image: "sandbox-image".to_string(),
                test_cases: 3,
                benchmark_cases: 2,
            },
            baseline: BaselineManifest {
                gpu: "H100".to_string(),
                measurements_ns: vec![10.0, 11.0, 12.0],
                median_ns: 11.0,
                command: "ferrl trimul-baseline --config run.json".to_string(),
            },
            verification: VerificationManifest {
                gpu: "H100".to_string(),
                runs: vec![
                    ArtifactVerificationRun {
                        correct: true,
                        benchmark_means_ns: vec![8.0],
                        geomean_ns: Some(8.0),
                        speedup: Some(1.375),
                    },
                    ArtifactVerificationRun {
                        correct: true,
                        benchmark_means_ns: vec![9.0],
                        geomean_ns: Some(9.0),
                        speedup: Some(1.222),
                    },
                    ArtifactVerificationRun {
                        correct: true,
                        benchmark_means_ns: vec![10.0],
                        geomean_ns: Some(10.0),
                        speedup: Some(1.1),
                    },
                ],
                accepted: true,
            },
        };
        let report = artifact_report(&manifest, Path::new("artifact"), "manifest-hash");
        for required in [
            "## 1. Verdict",
            "Raw measurements ns: 10.000000, 11.000000, 12.000000",
            "Command used: `ferrl trimul-baseline --config run.json`",
            "ferrl commit: abc123",
            "Config hash: config-hash",
            "Run health: healthy",
            "| source hash | training reward | source inspection | clean correctness | median runtime ns | speedup | accept/reject reason |",
            "| source-hash | 1.500000 | clean | 3/3 | 9.000000 | 1.222222 | accepted: all clean runs correct and median runtime beats baseline |",
            "Source inspection notes: no process, file descriptor, environment, network, or out-of-input path probes",
            "Path: artifact",
            "Manifest SHA-256: manifest-hash",
            "## 6. Reviewer Checklist",
            "[pass] audit seed differs from training seed",
            "[pass] source inspection found no process/file/env/network/path probing",
        ] {
            assert!(report.contains(required), "missing report field: {required}");
        }
    }
}
