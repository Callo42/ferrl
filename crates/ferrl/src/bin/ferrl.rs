//! `ferrl` — the single-binary front door: train a built-in task end-to-end from a
//! JSON run config, and report on a finished run.
//!
//! ```text
//! ferrl train --config run.json     # GRPO-train a built-in task (countdown | math | trimul)
//! ferrl trimul-baseline --config run.json   # measure the TriMul reference baseline (ns) on this GPU
//! ferrl trimul-score --config run.json --prompt-copy prompt.txt --completion raw.txt --out scores.jsonl
//! ferrl trimul-score --config run.json --prompt-copy prompt.txt --completion raw.txt --completion-normalization llama-cpp --out scores.jsonl
//! ferrl trimul-artifact --config run.json --prompt-copy runs/trimul-1/prompt.txt --completion raw.txt --out artifact/ ...
//! ferrl runreport <run-dir> [--config run.json] [--json] [--strict]   # one-glance run health summary
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
//! `trimul-score` scores raw external completions with the same shaped TriMul reward
//! used during training and writes external-score JSONL. It is for rollout diagnostics;
//! `trimul-artifact` remains the strict repeated audit gate.
//!
//! `runreport` folds in the standalone run-summary tool: it reads a run's
//! `metrics.jsonl` and prints (or emits as JSON) a [`RunSummary`](ferrl::RunSummary).
//! With `--config`, it also applies the run config's post-run `run_health` policy.
//! It exits with code 2 when a `fail` policy finding is raised, or when `--strict`
//! sees any summary anomaly or policy finding.
//!
//! `perf-gate` compares a baseline and candidate metrics stream, failing when
//! the update path goes dark or peak memory / step time exceed configured
//! regression thresholds.

// A CLI whose interface *is* its stdout/stderr; the library logs via `tracing`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
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
use ferrl::telemetry::{CandidateRecord, RegressionFailure};
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
    /// Score external TriMul completions once with the shaped reward.
    TrimulScore(Box<TrimulScoreArgs>),
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

/// Arguments for `ferrl trimul-score`.
#[derive(Debug, Args)]
struct TrimulScoreArgs {
    /// Path to the JSON run config used to configure the TriMul reward.
    #[arg(long)]
    config: PathBuf,
    /// Immutable prompt copy used for generation; verifies adjacent `prompt.sha256`.
    #[arg(long)]
    prompt_copy: PathBuf,
    /// Raw completion file to score. May be passed multiple times.
    #[arg(long)]
    completion: Vec<PathBuf>,
    /// JSONL file containing objects with at least `{ "completion": "..." }`.
    ///
    /// Optional fields are `step`, `prompt_index`, `group_index`, `rank`, `world_size`,
    /// `completion_len_tokens`, `source_id`, `metadata`, and `reward_metadata`.
    #[arg(long)]
    completions_jsonl: Vec<PathBuf>,
    /// Normalize known external-runtime transport text before TriMul extraction.
    ///
    /// The default is strict: score the completion bytes exactly as supplied.
    /// Use `llama-cpp` for GGUF rollouts whose stdout appends llama.cpp's
    /// trailing `[end of text]` sentinel after the model response.
    #[arg(long, value_enum, default_value = "none")]
    completion_normalization: CompletionNormalization,
    /// Output external-score JSONL file. Fails if it already exists.
    #[arg(long)]
    out: PathBuf,
    /// Secret seed for diagnostic scoring. Must differ from `trimul.secret_seed`.
    #[arg(long)]
    score_secret_seed: u64,
    /// External rollout id recorded in score metadata.
    #[arg(long, default_value = "external-rollout")]
    run_id: String,
    /// Public-safe label used to form opaque source ids for input files.
    #[arg(long, default_value = "external")]
    source_label: String,
    /// Default candidate step for raw completion files.
    #[arg(long, default_value_t = 0)]
    step: u64,
    /// Default prompt ordinal for raw completion files.
    #[arg(long, default_value_t = 0)]
    prompt_index: u64,
    /// Default data-parallel rank for raw completion files.
    #[arg(long, default_value_t = 0)]
    rank: usize,
    /// Default data-parallel world size for raw completion files.
    #[arg(long, default_value_t = 1)]
    world_size: usize,
    /// Model family label recorded in score metadata.
    #[arg(long, default_value = "external")]
    model_family: String,
    /// Operator-supplied checkpoint identity recorded in score metadata.
    #[arg(long)]
    checkpoint: Option<String>,
    /// Operator-supplied tokenizer identity recorded in score metadata.
    #[arg(long)]
    tokenizer: Option<String>,
}

/// Arguments for `ferrl trimul-artifact`.
#[derive(Debug, Args)]
struct TrimulArtifactArgs {
    /// Path to the JSON run config used for the discovery run.
    #[arg(long)]
    config: PathBuf,
    /// Immutable prompt copy frozen at training launch, usually `<run-dir>/prompt.txt`.
    #[arg(long)]
    prompt_copy: PathBuf,
    /// Raw model completion to extract `custom_kernel` from.
    #[arg(long)]
    completion: PathBuf,
    /// Normalize known external-runtime transport text before TriMul extraction.
    ///
    /// The raw completion is still copied into the artifact bundle. When this is
    /// not `none`, the normalized text used for extraction is also copied as
    /// `completion.normalized.txt` and recorded in `manifest.json`.
    #[arg(long, value_enum, default_value = "none")]
    completion_normalization: CompletionNormalization,
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

#[derive(Debug, Deserialize)]
struct TrimulScoreJsonlRecord {
    completion: String,
    #[serde(default)]
    step: Option<u64>,
    #[serde(default)]
    prompt_index: Option<u64>,
    #[serde(default)]
    group_index: Option<usize>,
    #[serde(default)]
    rank: Option<usize>,
    #[serde(default)]
    world_size: Option<usize>,
    #[serde(default)]
    completion_len_tokens: Option<usize>,
    #[serde(default)]
    source_id: Option<String>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    reward_metadata: Option<serde_json::Value>,
}

#[derive(Debug)]
struct TrimulScoreInput {
    completion: String,
    source_id: String,
    source_index: usize,
    step: u64,
    prompt_index: u64,
    group_index: usize,
    rank: usize,
    world_size: usize,
    completion_len_tokens: Option<usize>,
    metadata: Option<serde_json::Value>,
    reward_metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct TrimulScoreRecord {
    task: &'static str,
    score_scheme: &'static str,
    run_id: String,
    step: u64,
    rank: usize,
    world_size: usize,
    prompt_index: u64,
    group_index: usize,
    reward: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    reward_diagnostic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reward_metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_reward_metadata: Option<serde_json::Value>,
    completion_len_tokens: Option<usize>,
    completion_len_bytes: usize,
    completion_sha256: String,
    completion: String,
    external_score: TrimulExternalScoreMetadata,
}

#[derive(Debug, Serialize)]
struct TrimulExternalScoreMetadata {
    model_family: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer: Option<String>,
    prompt_sha256: String,
    run_config_sha256: String,
    source_id: String,
    source_index: usize,
    score_secret_seed: u64,
    used_training_secret_seed: bool,
}

/// Optional completion normalization before TriMul submission extraction.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, ValueEnum)]
enum CompletionNormalization {
    /// Strict mode: use completion bytes exactly as supplied.
    #[default]
    None,
    /// Strip llama.cpp's trailing stdout transport sentinel.
    LlamaCpp,
}

impl CompletionNormalization {
    /// Stable spelling for metadata and docs.
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LlamaCpp => "llama_cpp",
        }
    }
}

/// A completion after optional public-runtime normalization.
#[derive(Debug)]
struct NormalizedCompletion {
    /// Completion text used for extraction/scoring.
    text: String,
    /// SHA-256 of the raw completion bytes before normalization.
    raw_sha256: String,
    /// Length of the raw completion bytes before normalization.
    raw_len_bytes: usize,
    /// Whether normalization changed the completion text.
    changed: bool,
}

/// Arguments for `ferrl runreport`.
#[derive(Debug, Args)]
struct RunreportArgs {
    /// A run directory (its `metrics.jsonl` is used) or a `metrics.jsonl` file.
    path: PathBuf,
    /// Top-level `ferrl train --config` JSON whose `run_health` policy is applied.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Emit the summary as JSON instead of the human report.
    #[arg(long)]
    json: bool,
    /// Exit with code 2 if summary anomalies or configured policy findings are flagged.
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
    /// Enable the experimental grouped cached-GQA rollout memory path for Qwen3.5.
    memory_efficient_cached_gqa: bool,
}

impl Default for PolicyCfg {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            base_dtype: DtypeSel::F32,
            seed: 1234,
            activation_checkpointing: false,
            memory_efficient_cached_gqa: false,
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

/// TriMul task knobs (read only when `task == "trimul"`): the sandboxed eval image and
/// the pinned GPU Mode bundle, bounded scratch, the held-out secret seed, the
/// per-candidate wall budget, and the optional baseline pin. The concrete case list is
/// loaded at run time from `<eval_dir>/task.yml` (GPU Mode's, not vendored into this repo).
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TrimulCfg {
    /// UTF-8 file used as the exact rendered model prompt.
    ///
    /// The CLI intentionally has one prompt owner: this file. ferrl does not prepend,
    /// append, trim, or wrap prompt text for TriMul training runs.
    prompt_path: Option<PathBuf>,
    /// Completion parser used by the reward. This never constructs prompt text.
    submission_extract_mode: Option<ferrl::trimul::SubmissionExtractMode>,
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
    /// Optional CUDA-visible device list for every sandboxed verifier process.
    verifier_cuda_visible_devices: Option<String>,
    /// Optional per-worker CUDA-visible device lists for concurrent verifier processes.
    verifier_cuda_device_pool: Vec<String>,
    /// Maximum number of candidates in one GRPO group to verify concurrently (`0` -> 1).
    verifier_parallelism: usize,
    /// Process cap applied to each verifier sandbox (`0` -> TriMul default).
    verifier_max_procs: u64,
    /// The reference baseline pin (omit to fall back to an inverse-time reward).
    baseline: Option<BaselineCfg>,
    /// Versioned shaped training-reward profile.
    reward: ferrl::trimul::TrimulRewardProfile,
}

/// Discovery-health policy schema.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RunHealthCfg {
    /// Detect mean reward collapse over a trailing window.
    reward_collapse: Option<WindowThresholdCfg>,
    /// Detect task correctness collapse over a trailing window, when task metadata supports it.
    correctness_collapse: Option<WindowThresholdCfg>,
    /// Detect dropped/all-pad completion rows.
    dropped_rows: Option<CountThresholdCfg>,
    /// Detect large gradient spikes relative to a run-local baseline.
    grad_spike: Option<FactorThresholdCfg>,
    /// Detect missing off-policy drift telemetry.
    telemetry_dark: Option<HealthActionCfg>,
    /// Detect source-hash dominance in candidate ledgers.
    source_dominance: Option<FractionThresholdCfg>,
}

impl RunHealthCfg {
    fn validate_current_support(&self, trainer: &TrainerConfig) -> Result<(), CliError> {
        if let Some(rule) = &self.reward_collapse {
            rule.validate("run_health.reward_collapse")?;
            validate_health_window("run_health.reward_collapse", rule.window, trainer.steps)?;
        }
        if let Some(rule) = &self.correctness_collapse {
            rule.validate_fraction_min("run_health.correctness_collapse")?;
            validate_health_window(
                "run_health.correctness_collapse",
                rule.window,
                trainer.steps,
            )?;
        }
        if let Some(rule) = &self.dropped_rows {
            rule.validate("run_health.dropped_rows")?;
        }
        if let Some(rule) = &self.grad_spike {
            rule.validate("run_health.grad_spike")?;
        }
        if let Some(action) = self.telemetry_dark {
            validate_post_run_health_action("run_health.telemetry_dark", action)?;
        }
        if let Some(rule) = &self.source_dominance {
            rule.validate("run_health.source_dominance")?;
        }
        if self.needs_candidate_ledger() && trainer.candidate_log_top_k < trainer.group_size {
            return Err(CliError::msg(format!(
                "run_health correctness/source policies require \
                 trainer.candidate_log_top_k >= trainer.group_size for full candidate coverage \
                 (candidate_log_top_k={}, group_size={})",
                trainer.candidate_log_top_k, trainer.group_size
            )));
        }
        Ok(())
    }

    fn is_default(&self) -> bool {
        self == &Self::default()
    }

    fn needs_candidate_ledger(&self) -> bool {
        self.correctness_collapse.is_some() || self.source_dominance.is_some()
    }

    fn evaluate(
        &self,
        history: &[ferrl::Metrics],
        summary: &ferrl::RunSummary,
        ctx: RunHealthEvalCtx,
        candidates: Option<&CandidateHealth>,
    ) -> RunHealthReport {
        let mut report = RunHealthReport::default();
        self.evaluate_metric_rules(history, summary, &mut report);
        self.evaluate_candidate_rules(history, ctx, candidates, &mut report);
        report
    }

    fn evaluate_metric_rules(
        &self,
        history: &[ferrl::Metrics],
        summary: &ferrl::RunSummary,
        report: &mut RunHealthReport,
    ) {
        if let Some(rule) = &self.reward_collapse {
            push_reward_collapse_finding(history, rule, report);
        }
        if let Some(rule) = &self.dropped_rows {
            if u64::from(summary.total_dropped_rows) > rule.max {
                report.push(
                    "dropped_rows",
                    rule.action,
                    format!(
                        "dropped rows {} exceeded max {}",
                        summary.total_dropped_rows, rule.max
                    ),
                );
            }
        }
        if let Some(rule) = &self.grad_spike {
            push_grad_spike_finding(history, rule, report);
        }
        if let Some(action) = self.telemetry_dark {
            if !history.is_empty() && history.iter().all(|m| m.rollout_capture_tokens == 0) {
                report.push(
                    "telemetry_dark",
                    action,
                    "off-policy drift telemetry was dark for every step".to_string(),
                );
            }
        }
    }

    fn evaluate_candidate_rules(
        &self,
        history: &[ferrl::Metrics],
        ctx: RunHealthEvalCtx,
        candidates: Option<&CandidateHealth>,
        report: &mut RunHealthReport,
    ) {
        if let Some(rule) = &self.correctness_collapse {
            push_correctness_collapse_finding(history, ctx, candidates, rule, report);
        }
        if let Some(rule) = &self.source_dominance {
            push_source_dominance_finding(history, ctx, candidates, rule, report);
        }
    }
}

fn validate_health_window(label: &str, window: usize, trainer_steps: u64) -> Result<(), CliError> {
    if window as u64 > trainer_steps {
        return Err(CliError::msg(format!(
            "{label}.window ({window}) must be <= trainer.steps ({trainer_steps})"
        )));
    }
    Ok(())
}

/// Action a post-run health policy may take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum HealthActionCfg {
    /// Report but do not fail.
    Warn,
    /// Fail the post-run health gate.
    Fail,
    /// Reserved for a future in-run gate; rejected by the post-run policy.
    Stop,
}

impl HealthActionCfg {
    fn label(self) -> &'static str {
        match self {
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
            Self::Stop => "STOP",
        }
    }
}

fn validate_post_run_health_action(label: &str, action: HealthActionCfg) -> Result<(), CliError> {
    match action {
        HealthActionCfg::Warn | HealthActionCfg::Fail => Ok(()),
        HealthActionCfg::Stop => Err(CliError::msg(format!(
            "{label}.action = \"stop\" is reserved for future in-run gating; use \"warn\" or \
             \"fail\" for the post-run policy"
        ))),
    }
}

/// Windowed threshold policy.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct WindowThresholdCfg {
    /// Trailing window size in optimizer steps.
    window: usize,
    /// Minimum allowed value.
    min: f64,
    /// Policy action.
    action: HealthActionCfg,
}

impl WindowThresholdCfg {
    fn validate(&self, label: &str) -> Result<(), CliError> {
        if self.window == 0 {
            return Err(CliError::msg(format!("{label}.window must be >= 1")));
        }
        if !self.min.is_finite() {
            return Err(CliError::msg(format!("{label}.min must be finite")));
        }
        validate_post_run_health_action(label, self.action)
    }

    fn validate_fraction_min(&self, label: &str) -> Result<(), CliError> {
        self.validate(label)?;
        if !(0.0..=1.0).contains(&self.min) {
            return Err(CliError::msg(format!("{label}.min must be in [0, 1]")));
        }
        Ok(())
    }
}

/// Count threshold policy.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct CountThresholdCfg {
    /// Maximum allowed count.
    max: u64,
    /// Policy action.
    action: HealthActionCfg,
}

impl CountThresholdCfg {
    fn validate(&self, label: &str) -> Result<(), CliError> {
        validate_post_run_health_action(label, self.action)
    }
}

/// Factor threshold policy.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct FactorThresholdCfg {
    /// Maximum allowed multiplicative factor.
    factor: f64,
    /// Policy action.
    action: HealthActionCfg,
}

impl FactorThresholdCfg {
    fn validate(&self, label: &str) -> Result<(), CliError> {
        if !self.factor.is_finite() || self.factor <= 0.0 {
            return Err(CliError::msg(format!(
                "{label}.factor must be finite and > 0"
            )));
        }
        validate_post_run_health_action(label, self.action)
    }
}

/// Fraction threshold policy.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct FractionThresholdCfg {
    /// Maximum allowed fraction.
    max_fraction: f64,
    /// Policy action.
    action: HealthActionCfg,
}

impl FractionThresholdCfg {
    fn validate(&self, label: &str) -> Result<(), CliError> {
        if !self.max_fraction.is_finite() || !(0.0..=1.0).contains(&self.max_fraction) {
            return Err(CliError::msg(format!(
                "{label}.max_fraction must be finite and in [0, 1]"
            )));
        }
        validate_post_run_health_action(label, self.action)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RunHealthVerdict {
    #[default]
    Healthy,
    Warn,
    Fail,
}

impl RunHealthVerdict {
    fn label(self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }

    fn observe(&mut self, action: HealthActionCfg) {
        match action {
            HealthActionCfg::Warn if *self == Self::Healthy => *self = Self::Warn,
            HealthActionCfg::Fail => *self = Self::Fail,
            HealthActionCfg::Warn | HealthActionCfg::Stop => {}
        }
    }

    fn is_fail(self) -> bool {
        self == Self::Fail
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct RunHealthFinding {
    rule: &'static str,
    action: HealthActionCfg,
    message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
struct RunHealthReport {
    verdict: RunHealthVerdict,
    findings: Vec<RunHealthFinding>,
}

impl RunHealthReport {
    fn push(&mut self, rule: &'static str, action: HealthActionCfg, message: String) {
        self.verdict.observe(action);
        self.findings.push(RunHealthFinding {
            rule,
            action,
            message,
        });
    }

    fn has_findings(&self) -> bool {
        !self.findings.is_empty()
    }

    fn is_fail(&self) -> bool {
        self.verdict.is_fail()
    }
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
    /// Discovery health policy applied after training and by `runreport --config`.
    #[serde(default)]
    run_health: RunHealthCfg,
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
        let cfg: Self = serde_json::from_slice(&bytes).map_err(|source| CliError::Config {
            path: path.to_path_buf(),
            source,
        })?;
        cfg.validate_current_config_support()?;
        Ok(cfg)
    }

    fn validate_current_config_support(&self) -> Result<(), CliError> {
        self.trimul.reward.validate().map_err(CliError::msg)?;
        self.run_health.validate_current_support(&self.trainer)
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
            memory_efficient_cached_gqa: self.policy.memory_efficient_cached_gqa,
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
    #[cfg(test)]
    fn trimul_splits(&self) -> Result<Splits<()>, CliError> {
        let prompt_file_bytes = self.trimul_prompt_file_bytes()?;
        let prompt = self.trimul_prompt_text(&prompt_file_bytes)?;
        Ok(self.trimul_splits_from_prompt(&prompt))
    }

    /// Read the complete rendered TriMul model prompt file bytes.
    fn trimul_prompt_file_bytes(&self) -> Result<Vec<u8>, CliError> {
        let Some(path) = &self.trimul.prompt_path else {
            return Err(CliError::msg(
                "task \"trimul\" requires trimul.prompt_path (the complete rendered model prompt file)",
            ));
        };
        read_bytes(path)
    }

    /// Decode the exact TriMul prompt text fed to the model from launch-file bytes.
    fn trimul_prompt_text(&self, prompt_file_bytes: &[u8]) -> Result<String, CliError> {
        let prompt = std::str::from_utf8(prompt_file_bytes)
            .map_err(|e| CliError::msg(format!("trimul prompt is not valid UTF-8: {e}")))?;
        if prompt.is_empty() {
            return Err(CliError::msg("trimul prompt is empty"));
        }
        Ok(prompt.to_owned())
    }

    /// Build the repeated TriMul train/eval splits from the exact model prompt.
    fn trimul_splits_from_prompt(&self, prompt: &str) -> Splits<()> {
        let train = std::iter::repeat_with(|| Sample::new(prompt.to_owned(), ()))
            .take(self.data.train_n.max(1))
            .collect();
        let eval = std::iter::repeat_with(|| Sample::new(prompt.to_owned(), ()))
            .take(self.data.eval_n)
            .collect();
        (train, eval)
    }

    /// Completion extraction mode for TriMul rewards.
    fn trimul_submission_extract_mode(
        &self,
    ) -> Result<ferrl::trimul::SubmissionExtractMode, CliError> {
        self.trimul.submission_extract_mode.ok_or_else(|| {
            CliError::msg(
                "task \"trimul\" requires trimul.submission_extract_mode \
                 (\"final_fence\" or \"thinking_after_think\")",
            )
        })
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
            .with_wall(wall);
        reward = reward
            .with_reward_profile(t.reward)
            .map_err(CliError::msg)?;
        if let Some(devices) = &t.verifier_cuda_visible_devices {
            reward = reward.with_verifier_cuda_visible_devices(devices.clone());
        }
        if !t.verifier_cuda_device_pool.is_empty() {
            reward = reward.with_verifier_cuda_device_pool(t.verifier_cuda_device_pool.clone());
        }
        if t.verifier_parallelism != 0 {
            reward = reward.with_verifier_parallelism(t.verifier_parallelism);
        }
        if t.verifier_max_procs != 0 {
            reward = reward.with_verifier_max_procs(t.verifier_max_procs);
        }
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
        let mode = self.trimul_submission_extract_mode()?;
        let mut reward = self
            .build_trimul_reward_base()?
            .with_submission_extract_mode(mode);
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
            run_training(
                &cfg,
                &device,
                &CountdownReward::default(),
                &train,
                &eval,
                None,
            )
        }
        "math" => {
            let (train, eval) = cfg.math_splits()?;
            run_training(&cfg, &device, &MathReward::default(), &train, &eval, None)
        }
        "trimul" => {
            let prompt_file_bytes = cfg.trimul_prompt_file_bytes()?;
            let prompt = cfg.trimul_prompt_text(&prompt_file_bytes)?;
            let (train, eval) = cfg.trimul_splits_from_prompt(&prompt);
            let reward = cfg.build_trimul_reward()?;
            run_training(
                &cfg,
                &device,
                &reward,
                &train,
                &eval,
                Some(&prompt_file_bytes),
            )
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
    rendered_prompt_bytes: Option<&[u8]>,
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
    if let Some(prompt_bytes) = rendered_prompt_bytes {
        write_bytes(&run.root().join("prompt.txt"), prompt_bytes)?;
        write_text(
            &run.root().join("prompt.sha256"),
            &format!("{}\n", sha256_hex(prompt_bytes)),
        )?;
    }
    let mut trainer = open_trainer(tcfg, &run, cfg.distributed.enabled)?;
    let (history, _stop) = trainer.train(&mut policy, reward, &tok, train)?;
    if let Some(summary) = summarize(&history) {
        info!(steps = summary.steps, "ferrl train: complete");
        apply_train_run_health_policy(cfg, &history, &summary, &run)?;
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

fn apply_train_run_health_policy(
    cfg: &RunConfig,
    history: &[ferrl::Metrics],
    summary: &ferrl::RunSummary,
    run: &RunDir,
) -> Result<(), CliError> {
    let health_report = evaluate_run_health_policy(
        &cfg.run_health,
        history,
        summary,
        RunHealthEvalCtx::from_trainer(&cfg.trainer),
        run.root(),
    )?;
    if !cfg.run_health.is_default() {
        print_run_health_report(&health_report);
    }
    if health_report.is_fail() {
        return Err(CliError::msg("run_health policy failed"));
    }
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

/// Dispatch `ferrl trimul-score`: score raw external completions with TriMul's
/// shaped reward and persist external-score JSONL for rollout diagnostics.
fn trimul_score(args: &TrimulScoreArgs) -> Result<(), CliError> {
    let _ = ferrl::init_tracing();
    let config_bytes = read_bytes(&args.config)?;
    let cfg = parse_run_config(&args.config, &config_bytes)?;
    if cfg.task != "trimul" {
        return Err(CliError::msg(
            "trimul-score requires a config with task \"trimul\"",
        ));
    }
    if args.score_secret_seed == cfg.trimul.secret_seed {
        return Err(CliError::msg(
            "trimul-score requires --score-secret-seed to differ from trimul.secret_seed",
        ));
    }
    let prompt_bytes = read_verified_prompt_copy(&args.prompt_copy)?;
    let prompt_sha256 = sha256_hex(&prompt_bytes);
    let config_sha256 = sha256_hex(&config_bytes);
    let inputs = read_trimul_score_inputs(args)?;
    if inputs.is_empty() {
        return Err(CliError::msg(
            "trimul-score requires at least one --completion or --completions-jsonl row",
        ));
    }
    validate_trimul_score_inputs(&inputs)?;

    let reward = cfg
        .build_trimul_reward()?
        .with_secret_seed(args.score_secret_seed);
    let sample = Sample::new(String::new(), ());
    let completions: Vec<String> = inputs.iter().map(|i| i.completion.clone()).collect();
    let outcomes = reward
        .reward_group_detailed(&sample, &completions)
        .map_err(|e| CliError::msg(format!("trimul scoring failed: {e}")))?;
    if outcomes.len() != inputs.len() {
        return Err(CliError::msg(format!(
            "trimul scoring returned {} outcomes for {} completions",
            outcomes.len(),
            inputs.len()
        )));
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)
        .map_err(|source| CliError::Io {
            path: args.out.clone(),
            source,
        })?;

    let mut diagnostics = BTreeMap::<String, usize>::new();
    let mut positive = 0usize;
    let mut max_reward = f32::NEG_INFINITY;
    for (input, outcome) in inputs.iter().zip(outcomes.iter()) {
        if outcome.reward > 0.0 {
            positive += 1;
        }
        max_reward = max_reward.max(outcome.reward);
        if let Some(diagnostic) = &outcome.diagnostic {
            *diagnostics.entry(diagnostic.clone()).or_default() += 1;
        }

        let record = trimul_score_record(
            args,
            input,
            outcome.reward,
            outcome.diagnostic.clone(),
            outcome.metadata.clone(),
            &prompt_sha256,
            &config_sha256,
        );
        let line = serde_json::to_string(&record)
            .map_err(|e| CliError::msg(format!("serialize trimul score row: {e}")))?;
        file.write_all(line.as_bytes())
            .and_then(|()| file.write_all(b"\n"))
            .map_err(|source| CliError::Io {
                path: args.out.clone(),
                source,
            })?;
    }
    file.flush().map_err(|source| CliError::Io {
        path: args.out.clone(),
        source,
    })?;

    println!(
        "ferrl: scored {} TriMul completions -> {}",
        inputs.len(),
        args.out.display()
    );
    println!("ferrl: positives {positive}/{}", inputs.len());
    if max_reward.is_finite() {
        println!("ferrl: max_reward {max_reward}");
    }
    if !diagnostics.is_empty() {
        println!(
            "ferrl: diagnostics {}",
            serde_json::to_string(&diagnostics).unwrap_or_else(|_| "<unserializable>".to_string())
        );
    }
    Ok(())
}

fn read_trimul_score_inputs(args: &TrimulScoreArgs) -> Result<Vec<TrimulScoreInput>, CliError> {
    validate_public_source_id("--source-label", &args.source_label)?;
    let mut inputs = Vec::new();
    for path in &args.completion {
        let bytes = read_bytes(path)?;
        let completion = String::from_utf8(bytes).map_err(|e| {
            CliError::msg(format!(
                "completion file {} is not valid UTF-8: {e}",
                path.display()
            ))
        })?;
        let completion = normalize_completion(&completion, args.completion_normalization);
        let source_index = inputs.len();
        inputs.push(TrimulScoreInput {
            metadata: completion_normalization_metadata(
                None,
                args.completion_normalization,
                &completion,
            ),
            completion: completion.text,
            source_id: default_trimul_score_source_id(
                &args.source_label,
                "completion",
                source_index,
            ),
            source_index,
            step: args.step,
            prompt_index: args.prompt_index,
            group_index: source_index,
            rank: args.rank,
            world_size: args.world_size,
            completion_len_tokens: None,
            reward_metadata: None,
        });
    }
    for (jsonl_index, path) in args.completions_jsonl.iter().enumerate() {
        let raw = std::fs::read_to_string(path).map_err(|source| CliError::Io {
            path: path.clone(),
            source,
        })?;
        for (line_index, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: TrimulScoreJsonlRecord = serde_json::from_str(line).map_err(|e| {
                CliError::msg(format!(
                    "parse {} line {} as trimul-score JSONL: {e}",
                    path.display(),
                    line_index + 1
                ))
            })?;
            let source_index = inputs.len();
            let source_id = match record.source_id {
                Some(source_id) => {
                    validate_public_source_id("trimul-score JSONL source_id", &source_id)?;
                    source_id
                }
                None => default_trimul_score_jsonl_source_id(
                    &args.source_label,
                    jsonl_index,
                    line_index + 1,
                ),
            };
            let completion =
                normalize_completion(&record.completion, args.completion_normalization);
            inputs.push(TrimulScoreInput {
                metadata: completion_normalization_metadata(
                    record.metadata,
                    args.completion_normalization,
                    &completion,
                ),
                completion: completion.text,
                source_id,
                source_index,
                step: record.step.unwrap_or(args.step),
                prompt_index: record.prompt_index.unwrap_or(args.prompt_index),
                group_index: record.group_index.unwrap_or(source_index),
                rank: record.rank.unwrap_or(args.rank),
                world_size: record.world_size.unwrap_or(args.world_size),
                completion_len_tokens: record.completion_len_tokens,
                reward_metadata: record.reward_metadata,
            });
        }
    }
    Ok(inputs)
}

fn normalize_completion(raw: &str, mode: CompletionNormalization) -> NormalizedCompletion {
    let raw_len_bytes = raw.len();
    let raw_sha256 = sha256_hex(raw.as_bytes());
    let text = match mode {
        CompletionNormalization::None => raw.to_string(),
        CompletionNormalization::LlamaCpp => strip_llama_cpp_end_of_text(raw),
    };
    let changed = text != raw;
    NormalizedCompletion {
        text,
        raw_sha256,
        raw_len_bytes,
        changed,
    }
}

fn strip_llama_cpp_end_of_text(raw: &str) -> String {
    const LLAMA_CPP_EOT_SENTINEL: &str = "[end of text]";
    let stripped = raw.trim_end();
    if let Some(prefix) = stripped.strip_suffix(LLAMA_CPP_EOT_SENTINEL) {
        let normalized = prefix.trim_end();
        let mut out = String::with_capacity(normalized.len() + 1);
        out.push_str(normalized);
        out.push('\n');
        out
    } else {
        raw.to_owned()
    }
}

fn completion_normalization_metadata(
    metadata: Option<serde_json::Value>,
    mode: CompletionNormalization,
    completion: &NormalizedCompletion,
) -> Option<serde_json::Value> {
    if mode == CompletionNormalization::None {
        return metadata;
    }
    let normalization = serde_json::json!({
        "mode": mode.as_str(),
        "changed": completion.changed,
        "raw_completion_sha256": completion.raw_sha256,
        "raw_completion_len_bytes": completion.raw_len_bytes,
        "normalized_completion_sha256": sha256_hex(completion.text.as_bytes()),
        "normalized_completion_len_bytes": completion.text.len(),
    });
    match metadata {
        None => Some(serde_json::json!({
            "ferrl_completion_normalization": normalization,
        })),
        Some(serde_json::Value::Object(mut object)) => {
            object.insert("ferrl_completion_normalization".to_string(), normalization);
            Some(serde_json::Value::Object(object))
        }
        Some(other) => Some(serde_json::json!({
            "ferrl_completion_normalization": normalization,
            "operator_metadata": other,
        })),
    }
}

fn validate_trimul_score_inputs(inputs: &[TrimulScoreInput]) -> Result<(), CliError> {
    for input in inputs {
        if input.world_size == 0 {
            return Err(CliError::msg(format!(
                "trimul-score input {} has world_size = 0",
                input.source_id
            )));
        }
        if input.rank >= input.world_size {
            return Err(CliError::msg(format!(
                "trimul-score input {} has rank {} outside world_size {}",
                input.source_id, input.rank, input.world_size
            )));
        }
    }
    Ok(())
}

fn validate_public_source_id(label: &str, value: &str) -> Result<(), CliError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(CliError::msg(format!("{label} must not be empty")));
    }
    if trimmed != value {
        return Err(CliError::msg(format!(
            "{label} must not have leading or trailing whitespace"
        )));
    }
    if value.len() > 128 {
        return Err(CliError::msg(format!("{label} must be at most 128 bytes")));
    }
    if value.contains('/') || value.contains('\\') || value.contains("..") {
        return Err(CliError::msg(format!(
            "{label} must be a public-safe id, not a filesystem path"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(CliError::msg(format!(
            "{label} must not contain control characters"
        )));
    }
    Ok(())
}

fn default_trimul_score_source_id(label: &str, kind: &str, index: usize) -> String {
    format!("{label}:{kind}:{index}")
}

fn default_trimul_score_jsonl_source_id(label: &str, file_index: usize, line: usize) -> String {
    format!("{label}:jsonl:{file_index}:line:{line}")
}

fn finite_score_reward(reward: f32) -> f32 {
    if reward.is_finite() {
        reward
    } else if reward.is_nan() {
        0.0
    } else if reward.is_sign_positive() {
        f32::MAX
    } else {
        f32::MIN
    }
}

fn trimul_score_record(
    args: &TrimulScoreArgs,
    input: &TrimulScoreInput,
    reward: f32,
    reward_diagnostic: Option<String>,
    reward_metadata: Option<serde_json::Value>,
    prompt_sha256: &str,
    config_sha256: &str,
) -> TrimulScoreRecord {
    let completion_sha256 = sha256_hex(input.completion.as_bytes());
    TrimulScoreRecord {
        task: "trimul",
        score_scheme: "trimul_external_score_v1",
        run_id: args.run_id.clone(),
        step: input.step,
        rank: input.rank,
        world_size: input.world_size,
        prompt_index: input.prompt_index,
        group_index: input.group_index,
        reward: finite_score_reward(reward),
        reward_diagnostic,
        reward_metadata,
        input_metadata: input.metadata.clone(),
        input_reward_metadata: input.reward_metadata.clone(),
        completion_len_tokens: input.completion_len_tokens,
        completion_len_bytes: input.completion.len(),
        completion_sha256,
        completion: input.completion.clone(),
        external_score: TrimulExternalScoreMetadata {
            model_family: args.model_family.clone(),
            checkpoint: args.checkpoint.clone(),
            tokenizer: args.tokenizer.clone(),
            prompt_sha256: prompt_sha256.to_string(),
            run_config_sha256: config_sha256.to_string(),
            source_id: input.source_id.clone(),
            source_index: input.source_index,
            score_secret_seed: args.score_secret_seed,
            used_training_secret_seed: false,
        },
    }
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

/// Result of the operator-facing source inspection.
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

/// Operator-facing source-inspection record.
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
    /// Optional normalization applied before extracting `submission.py`.
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_normalization: Option<ArtifactCompletionNormalization>,
    /// SHA-256 of `submission.py`.
    source_sha256: String,
    /// Operator-facing source-inspection evidence.
    source_inspection: SourceInspectionManifest,
}

/// Completion normalization provenance for artifact extraction.
#[derive(Debug, Serialize)]
struct ArtifactCompletionNormalization {
    /// Normalization mode requested by the operator.
    mode: &'static str,
    /// Whether normalization changed the raw completion text.
    changed: bool,
    /// Raw completion length in bytes.
    raw_completion_len_bytes: usize,
    /// Normalized completion length in bytes.
    normalized_completion_len_bytes: usize,
    /// SHA-256 of the normalized completion text used for extraction.
    normalized_completion_sha256: String,
    /// Artifact-relative normalized completion file, present only when changed.
    #[serde(skip_serializing_if = "Option::is_none")]
    normalized_completion_file: Option<&'static str>,
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
    /// SHA-256 of the exact rendered TriMul model prompt bytes.
    prompt_sha256: String,
    /// Artifact-relative prompt copy used for audit.
    prompt_file: &'static str,
    /// Effective shaped training-reward profile.
    reward_profile: ferrl::trimul::TrimulRewardProfile,
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
    /// Maximum number of candidates in one GRPO group verified concurrently.
    verifier_parallelism: usize,
    /// Process cap applied to each verifier sandbox.
    verifier_max_procs: u64,
    /// Per-worker verifier CUDA visibility pool used during training.
    verifier_cuda_device_pool: Vec<String>,
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
    let prompt_bytes = read_verified_prompt_copy(&args.prompt_copy)?;
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
    let raw_completion = String::from_utf8(completion_bytes.clone()).map_err(|e| {
        CliError::msg(format!(
            "completion file {} is not valid UTF-8: {e}",
            args.completion.display()
        ))
    })?;
    let completion = normalize_completion(&raw_completion, args.completion_normalization);
    let extract_mode = cfg.trimul_submission_extract_mode()?;
    let mut reward = cfg
        .build_trimul_reward_base()?
        .with_submission_extract_mode(extract_mode);
    let submission = reward.extract_submission(&completion.text).ok_or_else(|| {
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
            raw_completion: &raw_completion,
            normalized_completion: &completion.text,
            completion_normalization: args.completion_normalization,
            completion_normalization_changed: completion.changed,
            completion_bytes: &completion_bytes,
            config_bytes: &config_bytes,
            prompt_bytes: &prompt_bytes,
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
    /// Raw completion string exactly as read from the operator-provided file.
    raw_completion: &'a str,
    /// Completion string used for extraction after optional normalization.
    normalized_completion: &'a str,
    /// Completion normalization mode used before extraction.
    completion_normalization: CompletionNormalization,
    /// Whether completion normalization changed the raw text.
    completion_normalization_changed: bool,
    /// Raw completion bytes.
    completion_bytes: &'a [u8],
    /// Raw config bytes.
    config_bytes: &'a [u8],
    /// Rendered TriMul model prompt bytes.
    prompt_bytes: &'a [u8],
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

/// Read a frozen run prompt copy and verify it against the adjacent launch hash.
fn read_verified_prompt_copy(path: &Path) -> Result<Vec<u8>, CliError> {
    let bytes = read_bytes(path)?;
    let actual = sha256_hex(&bytes);
    let hash_path = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("prompt.sha256");
    let raw_expected = std::fs::read_to_string(&hash_path).map_err(|source| CliError::Io {
        path: hash_path.clone(),
        source,
    })?;
    let expected = raw_expected.split_whitespace().next().unwrap_or_default();
    if expected != actual {
        return Err(CliError::msg(format!(
            "prompt copy hash mismatch: {} records {}, but {} hashes to {}",
            hash_path.display(),
            if expected.is_empty() {
                "<empty>"
            } else {
                expected
            },
            path.display(),
            actual
        )));
    }
    Ok(bytes)
}

/// Parse a [`RunConfig`] from already-read bytes.
fn parse_run_config(path: &Path, bytes: &[u8]) -> Result<RunConfig, CliError> {
    let cfg: RunConfig = serde_json::from_slice(bytes).map_err(|source| CliError::Config {
        path: path.to_path_buf(),
        source,
    })?;
    cfg.validate_current_config_support()?;
    Ok(cfg)
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
    write_text(&args.out.join("completion.txt"), inputs.raw_completion)?;
    if inputs.completion_normalization_changed {
        write_text(
            &args.out.join("completion.normalized.txt"),
            inputs.normalized_completion,
        )?;
    }
    write_bytes(&args.out.join("prompt.txt"), inputs.prompt_bytes)?;
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
            completion_normalization: artifact_completion_normalization(inputs),
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
            prompt_sha256: sha256_hex(inputs.prompt_bytes),
            prompt_file: "prompt.txt",
            reward_profile: cfg.trimul.reward,
            trainer_steps: cfg.trainer.steps,
            group_size: cfg.trainer.group_size,
            run_health: args.run_health.clone(),
            policy_seed: cfg.policy.seed,
            data_seed: cfg.data.seed,
            training_secret_seed: cfg.trimul.secret_seed,
            audit_secret_seed: args.audit_secret_seed,
            scratch_max_bytes: trimul_scratch_cap(cfg),
            verifier_parallelism: cfg.trimul.verifier_parallelism.max(1),
            verifier_max_procs: trimul_verifier_max_procs(cfg),
            verifier_cuda_device_pool: cfg.trimul.verifier_cuda_device_pool.clone(),
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

fn artifact_completion_normalization(
    inputs: &ArtifactInputs<'_>,
) -> Option<ArtifactCompletionNormalization> {
    if inputs.completion_normalization == CompletionNormalization::None {
        return None;
    }
    Some(ArtifactCompletionNormalization {
        mode: inputs.completion_normalization.as_str(),
        changed: inputs.completion_normalization_changed,
        raw_completion_len_bytes: inputs.completion_bytes.len(),
        normalized_completion_len_bytes: inputs.normalized_completion.len(),
        normalized_completion_sha256: sha256_hex(inputs.normalized_completion.as_bytes()),
        normalized_completion_file: inputs
            .completion_normalization_changed
            .then_some("completion.normalized.txt"),
    })
}

/// The effective TriMul scratch cap in bytes.
fn trimul_scratch_cap(cfg: &RunConfig) -> u64 {
    if cfg.trimul.scratch_max_bytes == 0 {
        1 << 30
    } else {
        cfg.trimul.scratch_max_bytes
    }
}

fn trimul_verifier_max_procs(cfg: &RunConfig) -> u64 {
    if cfg.trimul.verifier_max_procs == 0 {
        ferrl::trimul::DEFAULT_VERIFIER_MAX_PROCS
    } else {
        cfg.trimul.verifier_max_procs
    }
}

/// Write UTF-8 text to `path`.
fn write_text(path: &Path, text: &str) -> Result<(), CliError> {
    std::fs::write(path, text).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Write bytes to `path`.
fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    std::fs::write(path, bytes).map_err(|source| CliError::Io {
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
        "- Prompt copy: {} ({})",
        manifest.config.prompt_file, manifest.config.prompt_sha256
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Reward profile: `{}`",
        serde_json::to_string(&manifest.config.reward_profile)
            .expect("reward profile serializes to JSON")
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
        "- Budget: trainer_steps={}, group_size={}, scratch_max_bytes={}, verifier_max_procs={}",
        manifest.config.trainer_steps,
        manifest.config.group_size,
        manifest.config.scratch_max_bytes,
        manifest.config.verifier_max_procs
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

    writeln!(&mut out, "## 6. Operator Checklist\n").expect("writing to String cannot fail");
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
        !manifest.config.prompt_sha256.is_empty() && manifest.config.prompt_file == "prompt.txt",
        "prompt copy and hash recorded",
    );
    push_check(
        &mut out,
        manifest.config.reward_profile.validate().is_ok(),
        "reward profile recorded and valid",
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
        manifest.config.verifier_max_procs > 0,
        "verifier process cap recorded",
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

/// Append an operator checklist row.
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
    let health_report = if let Some(config_path) = &args.config {
        let cfg = RunConfig::load(config_path)?;
        Some(evaluate_run_health_policy(
            &cfg.run_health,
            &history,
            &summary,
            RunHealthEvalCtx::from_trainer(&cfg.trainer),
            &args.path,
        )?)
    } else {
        None
    };
    if args.json {
        let s = if let Some(report) = &health_report {
            serde_json::to_string_pretty(&RunreportJson {
                summary: &summary,
                run_health: report,
            })
        } else {
            serde_json::to_string_pretty(&summary)
        }
        .map_err(|e| CliError::msg(format!("serialize summary: {e}")))?;
        println!("{s}");
    } else {
        // `RunSummary`'s Display already terminates each line with a newline.
        print!("{summary}");
        if let Some(report) = &health_report {
            print_run_health_report(report);
        }
    }
    let policy_failed = health_report.as_ref().is_some_and(RunHealthReport::is_fail);
    let strict_failed = args.strict
        && (!summary.anomalies.is_empty()
            || health_report
                .as_ref()
                .is_some_and(RunHealthReport::has_findings));
    if policy_failed || strict_failed {
        return Ok(ExitCode::from(2));
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Serialize)]
struct RunreportJson<'a> {
    summary: &'a ferrl::RunSummary,
    run_health: &'a RunHealthReport,
}

fn evaluate_run_health_policy(
    policy: &RunHealthCfg,
    history: &[ferrl::Metrics],
    summary: &ferrl::RunSummary,
    ctx: RunHealthEvalCtx,
    run_path: &Path,
) -> Result<RunHealthReport, CliError> {
    let candidates = if policy.needs_candidate_ledger() {
        read_candidate_health_inputs(&[run_path.to_path_buf()])?
    } else {
        None
    };
    Ok(policy.evaluate(history, summary, ctx, candidates.as_ref()))
}

fn print_run_health_report(report: &RunHealthReport) {
    println!("run health policy — {}", report.verdict.label());
    for finding in &report.findings {
        println!(
            "  {} {}: {}",
            finding.action.label(),
            finding.rule,
            finding.message
        );
    }
}

/// Dispatch `ferrl perf-gate`: compare baseline and candidate metrics streams.
fn perf_gate(args: &PerfGateArgs) -> Result<ExitCode, CliError> {
    let budget = perf_budget(args)?;
    let mut report = if args.distributed_world_max {
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
    apply_candidate_health_gate(&mut report, args)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunHealthEvalCtx {
    group_size: usize,
    prompt_groups_per_step: usize,
}

impl RunHealthEvalCtx {
    fn from_trainer(trainer: &TrainerConfig) -> Self {
        Self {
            group_size: trainer.group_size,
            prompt_groups_per_step: trainer.grad_accum_steps,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CandidateHealth {
    total: usize,
    diagnostics: usize,
    source_buckets: BTreeMap<String, usize>,
    steps: BTreeMap<u64, CandidateStepHealth>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CandidateStepHealth {
    total: usize,
    correctness_supported: usize,
    correct: usize,
    prompt_groups: BTreeMap<u64, CandidatePromptGroupHealth>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CandidatePromptGroupHealth {
    group_indices: BTreeSet<usize>,
}

fn apply_candidate_health_gate(
    report: &mut RegressionReport,
    args: &PerfGateArgs,
) -> Result<(), CliError> {
    if args.allow_health_warnings {
        return Ok(());
    }
    let baseline = read_candidate_health_inputs(&args.baseline)?;
    let candidate = read_candidate_health_inputs(&args.candidate)?;
    compare_candidate_health(baseline, candidate, &mut report.failures);
    report.passed = report.failures.is_empty();
    Ok(())
}

fn read_candidate_health_inputs(paths: &[PathBuf]) -> Result<Option<CandidateHealth>, CliError> {
    let mut health = CandidateHealth::default();
    let mut found = false;
    for path in paths {
        let path = resolve_candidates_path(path);
        if !path.exists() {
            continue;
        }
        found = true;
        let raw = std::fs::read_to_string(&path).map_err(|source| CliError::Io {
            path: path.clone(),
            source,
        })?;
        for (idx, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: CandidateRecord = serde_json::from_str(line).map_err(|e| {
                CliError::msg(format!("parse {} line {}: {e}", path.display(), idx + 1))
            })?;
            health.total += 1;
            health.diagnostics += usize::from(record.reward_diagnostic.is_some());
            *health
                .source_buckets
                .entry(candidate_source_bucket(&record))
                .or_default() += 1;
            let step = health.steps.entry(record.step).or_default();
            step.total += 1;
            step.prompt_groups
                .entry(record.prompt_index)
                .or_default()
                .group_indices
                .insert(record.group_index);
            if let Some(correct) = candidate_correctness(&record) {
                step.correctness_supported += 1;
                step.correct += usize::from(correct);
            }
        }
    }
    Ok(found.then_some(health))
}

fn candidate_correctness(record: &CandidateRecord) -> Option<bool> {
    let metadata = record.reward_metadata.as_ref()?;
    if let Some(correct) = metadata.get("correct").and_then(serde_json::Value::as_bool) {
        return Some(correct);
    }
    let task_is_trimul = metadata.get("task").and_then(serde_json::Value::as_str) == Some("trimul");
    let no_submission = metadata
        .get("submission_extracted")
        .and_then(serde_json::Value::as_bool)
        == Some(false);
    if task_is_trimul && (no_submission || record.reward_diagnostic.is_some()) {
        return Some(false);
    }
    None
}

fn candidate_source_bucket(record: &CandidateRecord) -> String {
    record
        .reward_metadata
        .as_ref()
        .and_then(|metadata| metadata.get("source_sha256"))
        .and_then(serde_json::Value::as_str)
        .filter(|source| !source.trim().is_empty())
        .unwrap_or("__unknown_source__")
        .to_string()
}

fn push_reward_collapse_finding(
    history: &[ferrl::Metrics],
    rule: &WindowThresholdCfg,
    report: &mut RunHealthReport,
) {
    if history.len() < rule.window {
        report.push(
            "reward_collapse",
            rule.action,
            format!(
                "only {} metric rows available for {}-step reward window",
                history.len(),
                rule.window
            ),
        );
        return;
    }
    let tail = &history[history.len() - rule.window..];
    let mean = tail.iter().map(|m| f64::from(m.reward_mean)).sum::<f64>() / tail.len() as f64;
    if mean < rule.min {
        report.push(
            "reward_collapse",
            rule.action,
            format!(
                "trailing {}-step mean reward {mean:.6} fell below min {:.6}",
                rule.window, rule.min
            ),
        );
    }
}

fn push_correctness_collapse_finding(
    history: &[ferrl::Metrics],
    ctx: RunHealthEvalCtx,
    candidates: Option<&CandidateHealth>,
    rule: &WindowThresholdCfg,
    report: &mut RunHealthReport,
) {
    let Some(tail_steps) = trailing_metric_steps(history, rule.window) else {
        report.push(
            "correctness_collapse",
            rule.action,
            format!(
                "only {} metric rows available for {}-step correctness window",
                history.len(),
                rule.window
            ),
        );
        return;
    };
    let Some(candidates) = candidates else {
        report.push(
            "correctness_collapse",
            rule.action,
            "candidate ledger unavailable; cannot evaluate correctness policy".to_string(),
        );
        return;
    };
    if candidates.total == 0 {
        report.push(
            "correctness_collapse",
            rule.action,
            "candidate ledger is empty; cannot evaluate correctness policy".to_string(),
        );
        return;
    }
    let missing_steps = missing_candidate_steps(candidates, &tail_steps);
    if !missing_steps.is_empty() {
        report.push(
            "correctness_collapse",
            rule.action,
            format!(
                "candidate ledger missing rows for trailing metric steps {}",
                format_steps(&missing_steps)
            ),
        );
        return;
    }
    let partial_steps = partial_candidate_coverage_steps(candidates, &tail_steps, ctx);
    if !partial_steps.is_empty() {
        report.push(
            "correctness_collapse",
            rule.action,
            format!(
                "candidate ledger lacks full group coverage for trailing metric steps {}",
                format_steps(&partial_steps)
            ),
        );
        return;
    }
    let unsupported_steps = unsupported_correctness_steps(candidates, &tail_steps);
    if !unsupported_steps.is_empty() {
        report.push(
            "correctness_collapse",
            rule.action,
            format!(
                "candidate correctness metadata unavailable for trailing metric steps {}",
                format_steps(&unsupported_steps)
            ),
        );
        return;
    }
    let supported = tail_steps
        .iter()
        .filter_map(|step| candidates.steps.get(step))
        .map(|step| step.correctness_supported)
        .sum::<usize>();
    if supported == 0 {
        report.push(
            "correctness_collapse",
            rule.action,
            format!(
                "no candidate correctness metadata in trailing {} steps",
                rule.window
            ),
        );
        return;
    }
    let correct = tail_steps
        .iter()
        .filter_map(|step| candidates.steps.get(step))
        .map(|step| step.correct)
        .sum::<usize>();
    let fraction = correct as f64 / supported as f64;
    if fraction < rule.min {
        report.push(
            "correctness_collapse",
            rule.action,
            format!(
                "trailing {}-step candidate correctness {correct}/{supported} = {fraction:.3} \
                 fell below min {:.3}",
                rule.window, rule.min
            ),
        );
    }
}

fn trailing_metric_steps(history: &[ferrl::Metrics], window: usize) -> Option<Vec<u64>> {
    if history.len() < window {
        return None;
    }
    Some(
        history[history.len() - window..]
            .iter()
            .map(|m| m.step)
            .collect(),
    )
}

fn missing_candidate_steps(candidates: &CandidateHealth, steps: &[u64]) -> Vec<u64> {
    steps
        .iter()
        .copied()
        .filter(|step| {
            candidates
                .steps
                .get(step)
                .is_none_or(|health| health.total == 0)
        })
        .collect()
}

fn partial_candidate_coverage_steps(
    candidates: &CandidateHealth,
    steps: &[u64],
    ctx: RunHealthEvalCtx,
) -> Vec<u64> {
    steps
        .iter()
        .copied()
        .filter(|step| {
            candidates
                .steps
                .get(step)
                .is_some_and(|health| !candidate_step_has_full_coverage(health, ctx))
        })
        .collect()
}

fn candidate_step_has_full_coverage(health: &CandidateStepHealth, ctx: RunHealthEvalCtx) -> bool {
    health.prompt_groups.len() == ctx.prompt_groups_per_step
        && health
            .prompt_groups
            .values()
            .all(|group| prompt_group_has_full_coverage(group, ctx.group_size))
}

fn prompt_group_has_full_coverage(group: &CandidatePromptGroupHealth, group_size: usize) -> bool {
    group.group_indices.len() == group_size
        && (0..group_size).all(|idx| group.group_indices.contains(&idx))
}

fn unsupported_correctness_steps(candidates: &CandidateHealth, steps: &[u64]) -> Vec<u64> {
    steps
        .iter()
        .copied()
        .filter(|step| {
            candidates
                .steps
                .get(step)
                .is_some_and(|health| health.correctness_supported == 0)
        })
        .collect()
}

fn format_steps(steps: &[u64]) -> String {
    steps
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn push_grad_spike_finding(
    history: &[ferrl::Metrics],
    rule: &FactorThresholdCfg,
    report: &mut RunHealthReport,
) {
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
    let factor = f64::from(worst.grad_norm) / f64::from(median);
    if factor > rule.factor {
        report.push(
            "grad_spike",
            rule.action,
            format!(
                "grad_norm {:.6} at step {} was {factor:.2}x median {:.6}, above factor {:.2}",
                worst.grad_norm, worst.step, median, rule.factor
            ),
        );
    }
}

fn median_positive_grad_norm(history: &[ferrl::Metrics]) -> f32 {
    let mut values: Vec<f32> = history
        .iter()
        .map(|m| m.grad_norm)
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

fn push_source_dominance_finding(
    history: &[ferrl::Metrics],
    ctx: RunHealthEvalCtx,
    candidates: Option<&CandidateHealth>,
    rule: &FractionThresholdCfg,
    report: &mut RunHealthReport,
) {
    let Some(candidates) = candidates else {
        report.push(
            "source_dominance",
            rule.action,
            "candidate ledger unavailable; cannot evaluate source-dominance policy".to_string(),
        );
        return;
    };
    if candidates.total == 0 {
        report.push(
            "source_dominance",
            rule.action,
            "candidate ledger is empty; cannot evaluate source-dominance policy".to_string(),
        );
        return;
    }
    let steps: Vec<u64> = history.iter().map(|metrics| metrics.step).collect();
    let missing_steps = missing_candidate_steps(candidates, &steps);
    if !missing_steps.is_empty() {
        report.push(
            "source_dominance",
            rule.action,
            format!(
                "candidate ledger missing rows for metric steps {}",
                format_steps(&missing_steps)
            ),
        );
        return;
    }
    let partial_steps = partial_candidate_coverage_steps(candidates, &steps, ctx);
    if !partial_steps.is_empty() {
        report.push(
            "source_dominance",
            rule.action,
            format!(
                "candidate ledger lacks full group coverage for metric steps {}",
                format_steps(&partial_steps)
            ),
        );
        return;
    }
    let Some((source, count)) = candidates
        .source_buckets
        .iter()
        .max_by(|(_, a), (_, b)| a.cmp(b))
    else {
        return;
    };
    let fraction = *count as f64 / candidates.total as f64;
    if fraction > rule.max_fraction {
        report.push(
            "source_dominance",
            rule.action,
            format!(
                "dominant candidate source {source} covered {count}/{} = {fraction:.3}, above \
                 max_fraction {:.3}",
                candidates.total, rule.max_fraction
            ),
        );
    }
}

fn resolve_candidates_path(input: &Path) -> PathBuf {
    if input.file_name().and_then(|name| name.to_str()) == Some("candidates.jsonl") {
        return input.to_path_buf();
    }
    if input.is_dir() {
        return input.join("candidates.jsonl");
    }
    input.with_file_name("candidates.jsonl")
}

fn compare_candidate_health(
    baseline: Option<CandidateHealth>,
    candidate: Option<CandidateHealth>,
    failures: &mut Vec<RegressionFailure>,
) {
    match (baseline, candidate) {
        (None, None) => {}
        (None, Some(_)) => {
            failures.push(RegressionFailure::CandidateLedgerMissing { stream: "baseline" });
        }
        (Some(_), None) => failures.push(RegressionFailure::CandidateLedgerMissing {
            stream: "candidate",
        }),
        (Some(baseline), Some(candidate)) => {
            if baseline.diagnostics != candidate.diagnostics {
                failures.push(RegressionFailure::CandidateDiagnostics {
                    baseline: baseline.diagnostics,
                    candidate: candidate.diagnostics,
                });
            }
        }
    }
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
        Command::TrimulScore(args) => trimul_score(args).map(|()| ExitCode::SUCCESS),
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

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(tag: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("ferrl-{tag}-{}-{nonce}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn trimul_score_test_config(secret_seed: u64) -> String {
        format!(
            r#"{{
                "task": "trimul",
                "model_dir": "/m",
                "trimul": {{
                  "prompt_path": "/prompt.txt",
                  "submission_extract_mode": "final_fence",
                  "image": "/image.sif",
                  "eval_dir": "/eval",
                  "scratch_root": "/scratch",
                  "secret_seed": {secret_seed}
                }},
                "trainer": {{ "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }}
            }}"#
        )
    }

    fn trimul_invalid_reward_test_config(secret_seed: u64) -> String {
        format!(
            r#"{{
                "task": "trimul",
                "model_dir": "/m",
                "trimul": {{
                  "prompt_path": "/prompt.txt",
                  "submission_extract_mode": "final_fence",
                  "image": "/image.sif",
                  "eval_dir": "/eval",
                  "scratch_root": "/scratch",
                  "secret_seed": {secret_seed},
                  "reward": {{ "runnable": 0.40 }}
                }},
                "trainer": {{ "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }}
            }}"#
        )
    }

    fn trimul_score_args_for_test(dir: &Path) -> TrimulScoreArgs {
        TrimulScoreArgs {
            config: dir.join("run.json"),
            prompt_copy: dir.join("prompt.txt"),
            completion: Vec::new(),
            completions_jsonl: Vec::new(),
            completion_normalization: CompletionNormalization::None,
            out: dir.join("scores.jsonl"),
            score_secret_seed: 999,
            run_id: "test-run".to_string(),
            source_label: "public-batch".to_string(),
            step: 9,
            prompt_index: 8,
            rank: 2,
            world_size: 3,
            model_family: "gemma4".to_string(),
            checkpoint: None,
            tokenizer: None,
        }
    }

    fn trimul_artifact_args_for_test(dir: &Path) -> TrimulArtifactArgs {
        TrimulArtifactArgs {
            config: dir.join("run.json"),
            prompt_copy: dir.join("prompt.txt"),
            completion: dir.join("completion.txt"),
            completion_normalization: CompletionNormalization::None,
            out: dir.join("artifact"),
            run_id: "test-run".to_string(),
            step: 0,
            prompt_index: 0,
            group_index: 0,
            rank: 0,
            world_size: 1,
            training_reward: 0.0,
            audit_secret_seed: 999,
            baseline_measurements_ns: vec![1.0, 1.0, 1.0],
            baseline_command: None,
            repeats: 3,
            ferrl_commit: "test-commit".to_string(),
            run_health: "test".to_string(),
            source_inspection: SourceInspectionResult::Clean,
            source_inspection_notes: "clean".to_string(),
            model_family: "gemma4".to_string(),
            checkpoint: None,
            tokenizer: None,
            eval_bundle: None,
            sandbox_image: None,
        }
    }

    fn trimul_score_input_for_test(
        source_id: &str,
        rank: usize,
        world_size: usize,
    ) -> TrimulScoreInput {
        TrimulScoreInput {
            completion: "completion".to_string(),
            source_id: source_id.to_string(),
            source_index: 0,
            step: 0,
            prompt_index: 0,
            group_index: 0,
            rank,
            world_size,
            completion_len_tokens: None,
            metadata: None,
            reward_metadata: None,
        }
    }

    fn write_prompt_copy(dir: &Path, prompt: &[u8], hash: &str) -> PathBuf {
        let prompt_path = dir.join("prompt.txt");
        std::fs::write(&prompt_path, prompt).unwrap();
        std::fs::write(dir.join("prompt.sha256"), format!("{hash}\n")).unwrap();
        prompt_path
    }

    fn run_health_test_metric(step: u64, reward: f32, grad_norm: f32) -> ferrl::Metrics {
        let mut m = ferrl::Metrics::at_step(step);
        m.reward_mean = reward;
        m.grad_norm = grad_norm;
        m.rollout_capture_tokens = 8;
        m.step_secs = 1.0;
        m.tokens_per_sec = 16.0;
        m
    }

    fn write_metrics_jsonl(path: &Path, history: &[ferrl::Metrics]) {
        let mut raw = String::new();
        for metrics in history {
            raw.push_str(&serde_json::to_string(metrics).unwrap());
            raw.push('\n');
        }
        std::fs::write(path, raw).unwrap();
    }

    fn write_candidate_jsonl(
        path: &Path,
        rows: impl IntoIterator<Item = (u64, usize, bool, String)>,
    ) {
        write_candidate_jsonl_with_prompts(
            path,
            rows.into_iter()
                .map(|(step, group_index, correct, source_sha256)| {
                    (step, step, group_index, correct, source_sha256)
                }),
        );
    }

    fn write_candidate_jsonl_with_prompts(
        path: &Path,
        rows: impl IntoIterator<Item = (u64, u64, usize, bool, String)>,
    ) {
        let mut raw = String::new();
        for (step, prompt_index, group_index, correct, source_sha256) in rows {
            let row = serde_json::json!({
                "step": step,
                "rank": 0,
                "world_size": 1,
                "prompt_index": prompt_index,
                "group_index": group_index,
                "reward": if correct { 2.0 } else { 0.05 },
                "completion_len_tokens": 16,
                "reward_metadata": {
                    "task": "trimul",
                    "source_sha256": source_sha256,
                    "correct": correct
                },
                "completion": "candidate"
            });
            raw.push_str(&serde_json::to_string(&row).unwrap());
            raw.push('\n');
        }
        std::fs::write(path, raw).unwrap();
    }

    fn run_health_eval_ctx(group_size: usize) -> RunHealthEvalCtx {
        RunHealthEvalCtx {
            group_size,
            prompt_groups_per_step: 1,
        }
    }

    fn run_health_s50_history() -> Vec<ferrl::Metrics> {
        (0..50).map(run_health_s50_metric).collect()
    }

    fn run_health_s50_metric(step: u64) -> ferrl::Metrics {
        let mut m = run_health_test_metric(step, s50_reward(step), s50_grad_norm(step));
        m.dropped_rows = s50_dropped_rows(step);
        m
    }

    fn s50_reward(step: u64) -> f32 {
        if step < 25 {
            2.0
        } else {
            0.05
        }
    }

    fn s50_grad_norm(step: u64) -> f32 {
        if step == 30 {
            20.0
        } else {
            1.0
        }
    }

    fn s50_dropped_rows(step: u64) -> u32 {
        if step == 10 {
            1
        } else {
            0
        }
    }

    fn run_health_s50_candidate_rows() -> Vec<(u64, usize, bool, String)> {
        (0..50)
            .flat_map(|step| {
                (0..4).map(move |group| {
                    (
                        step,
                        group,
                        s50_candidate_correct(step, group),
                        s50_candidate_source(step, group),
                    )
                })
            })
            .collect()
    }

    fn s50_candidate_correct(step: u64, group: usize) -> bool {
        step < 24 || (step == 24 && group < 3)
    }

    fn s50_candidate_source(step: u64, group: usize) -> String {
        if step < 30 {
            "dominant-source".to_string()
        } else {
            format!("source-{step}-{group}")
        }
    }

    fn s50_run_health_policy() -> RunHealthCfg {
        RunHealthCfg {
            reward_collapse: Some(WindowThresholdCfg {
                window: 10,
                min: 1.0,
                action: HealthActionCfg::Fail,
            }),
            correctness_collapse: Some(WindowThresholdCfg {
                window: 10,
                min: 0.5,
                action: HealthActionCfg::Fail,
            }),
            dropped_rows: Some(CountThresholdCfg {
                max: 0,
                action: HealthActionCfg::Warn,
            }),
            grad_spike: Some(FactorThresholdCfg {
                factor: 8.0,
                action: HealthActionCfg::Warn,
            }),
            telemetry_dark: None,
            source_dominance: Some(FractionThresholdCfg {
                max_fraction: 0.5,
                action: HealthActionCfg::Warn,
            }),
        }
    }

    fn assert_run_health_rules(report: &RunHealthReport, expected: &[&str]) {
        let rules: Vec<_> = report.findings.iter().map(|f| f.rule).collect();
        for rule in expected {
            assert!(rules.contains(rule), "{rules:?}");
        }
    }

    fn correctness_collapse_policy() -> RunHealthCfg {
        RunHealthCfg {
            correctness_collapse: Some(WindowThresholdCfg {
                window: 2,
                min: 0.5,
                action: HealthActionCfg::Fail,
            }),
            ..RunHealthCfg::default()
        }
    }

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
        assert!(!cfg.policy.memory_efficient_cached_gqa);
        assert_eq!(cfg.data.train_n, 64);
        // The loader temperature mirrors the trainer's (cannot drift).
        assert!((cfg.loader_opts().temperature - cfg.trainer.temperature).abs() < f64::EPSILON);
    }

    #[test]
    fn discovery_control_default_schema_is_accepted() {
        let tmp = TestDir::new("discovery-control-default");
        let json = r#"{
            "task": "trimul",
            "model_dir": "/m",
            "trimul": {
              "prompt_path": "/prompt.txt",
              "submission_extract_mode": "final_fence",
              "reward": {
                "scheme": "trimul_shaped_v1",
                "format_extracted": 0.02,
                "runnable": 0.05,
                "partial_correctness": 0.75,
                "correctness": 1.0,
                "speed_cap": 2.0,
                "implausible_benchmark": "zero"
              }
            },
            "run_health": {},
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
              "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
              "lr": 1e-5, "weight_decay": 0.0,
              "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let path = tmp.path().join("run.json");
        std::fs::write(&path, json).unwrap();

        let cfg = RunConfig::load(&path).unwrap();

        assert_eq!(
            cfg.trimul.reward,
            ferrl::trimul::TrimulRewardProfile::default()
        );
        assert_eq!(cfg.run_health, RunHealthCfg::default());
    }

    #[test]
    fn discovery_control_custom_reward_values_are_accepted_when_ladder_is_valid() {
        let tmp = TestDir::new("discovery-control-custom");
        let reward_json = r#"{
            "task": "trimul",
            "model_dir": "/m",
            "trimul": {
              "prompt_path": "/prompt.txt",
              "submission_extract_mode": "final_fence",
              "reward": { "format_extracted": 0.03, "runnable": 0.07, "partial_correctness": 0.70 }
            },
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
              "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
              "lr": 1e-5, "weight_decay": 0.0,
              "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let reward_path = tmp.path().join("reward.json");
        std::fs::write(&reward_path, reward_json).unwrap();

        let cfg = RunConfig::load(&reward_path).unwrap();

        assert_eq!(cfg.trimul.reward.format_extracted, 0.03);
        assert_eq!(cfg.trimul.reward.runnable, 0.07);
        assert_eq!(cfg.trimul.reward.partial_correctness, 0.70);
    }

    #[test]
    fn discovery_control_custom_run_health_policy_is_accepted() {
        let tmp = TestDir::new("discovery-control-health");
        let health_json = r#"{
            "task": "countdown",
            "model_dir": "/m",
            "run_health": {
              "reward_collapse": { "window": 10, "min": 1.0, "action": "fail" },
              "correctness_collapse": { "window": 10, "min": 0.8, "action": "fail" },
              "dropped_rows": { "max": 0, "action": "warn" },
              "grad_spike": { "factor": 6.0, "action": "warn" },
              "telemetry_dark": "warn",
              "source_dominance": { "max_fraction": 0.6, "action": "warn" }
            },
            "trainer": { "steps": 10, "group_size": 2, "candidate_log_top_k": 2,
              "max_new_tokens": 8,
              "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
              "lr": 1e-5, "weight_decay": 0.0,
              "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let health_path = tmp.path().join("health.json");
        std::fs::write(&health_path, health_json).unwrap();

        let cfg = RunConfig::load(&health_path).unwrap();

        assert!(cfg.run_health.reward_collapse.is_some());
        assert!(cfg.run_health.correctness_collapse.is_some());
        assert!(cfg.run_health.dropped_rows.is_some());
        assert!(cfg.run_health.grad_spike.is_some());
        assert_eq!(cfg.run_health.telemetry_dark, Some(HealthActionCfg::Warn));
        assert!(cfg.run_health.source_dominance.is_some());
    }

    #[test]
    fn discovery_control_candidate_health_requires_full_candidate_logging() {
        let tmp = TestDir::new("discovery-control-health-topk");
        let health_json = r#"{
            "task": "countdown",
            "model_dir": "/m",
            "run_health": {
              "correctness_collapse": { "window": 2, "min": 0.8, "action": "fail" }
            },
            "trainer": { "steps": 2, "group_size": 4, "candidate_log_top_k": 1,
              "max_new_tokens": 8,
              "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
              "lr": 1e-5, "weight_decay": 0.0,
              "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let health_path = tmp.path().join("health.json");
        std::fs::write(&health_path, health_json).unwrap();

        let err = RunConfig::load(&health_path).unwrap_err().to_string();

        assert!(err.contains("candidate_log_top_k >= trainer.group_size"));
    }

    #[test]
    fn discovery_control_windowed_run_health_requires_enough_steps() {
        let tmp = TestDir::new("discovery-control-health-window");
        let health_json = r#"{
            "task": "countdown",
            "model_dir": "/m",
            "run_health": {
              "reward_collapse": { "window": 5, "min": 1.0, "action": "fail" }
            },
            "trainer": { "steps": 2, "group_size": 2, "max_new_tokens": 8,
              "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
              "lr": 1e-5, "weight_decay": 0.0,
              "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let health_path = tmp.path().join("health.json");
        std::fs::write(&health_path, health_json).unwrap();

        let err = RunConfig::load(&health_path).unwrap_err().to_string();

        assert!(err.contains("window (5) must be <= trainer.steps (2)"));
    }

    #[test]
    fn discovery_control_invalid_reward_ladders_and_run_health_stop_are_rejected() {
        let tmp = TestDir::new("discovery-control-invalid");
        let reward_json = r#"{
            "task": "trimul",
            "model_dir": "/m",
            "trimul": {
              "prompt_path": "/prompt.txt",
              "submission_extract_mode": "final_fence",
              "reward": { "runnable": 0.40 }
            },
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
              "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
              "lr": 1e-5, "weight_decay": 0.0,
              "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let reward_path = tmp.path().join("reward.json");
        std::fs::write(&reward_path, reward_json).unwrap();

        let reward_err = RunConfig::load(&reward_path).unwrap_err().to_string();

        assert!(reward_err.contains("runnable + trimul.reward.partial_correctness"));

        let health_json = r#"{
            "task": "countdown",
            "model_dir": "/m",
            "run_health": {
              "reward_collapse": { "window": 5, "min": 1.0, "action": "stop" }
            },
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
              "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
              "lr": 1e-5, "weight_decay": 0.0,
              "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let health_path = tmp.path().join("health.json");
        std::fs::write(&health_path, health_json).unwrap();

        let health_err = RunConfig::load(&health_path).unwrap_err().to_string();

        assert!(health_err.contains("reserved for future in-run gating"));
    }

    #[test]
    fn discovery_control_validation_reaches_score_and_artifact_paths() {
        let tmp = TestDir::new("discovery-control-cli-paths");
        std::fs::write(
            tmp.path().join("run.json"),
            trimul_invalid_reward_test_config(4242),
        )
        .unwrap();

        let score_err = trimul_score(&trimul_score_args_for_test(tmp.path()))
            .unwrap_err()
            .to_string();
        let artifact_err = trimul_artifact(&trimul_artifact_args_for_test(tmp.path()))
            .unwrap_err()
            .to_string();

        assert!(score_err.contains("runnable + trimul.reward.partial_correctness"));
        assert!(artifact_err.contains("runnable + trimul.reward.partial_correctness"));
    }

    /// `device` and `base_dtype` selectors deserialize from lowercase strings.
    #[test]
    fn device_and_dtype_selectors_parse() {
        let json = r#"{
            "task": "math",
            "model_dir": "/m",
            "device": "cuda",
            "policy": {
                "base_dtype": "bf16",
                "activation_checkpointing": true,
                "memory_efficient_cached_gqa": true
            },
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
        assert!(cfg.loader_opts().memory_efficient_cached_gqa);
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

    /// The clap surface parses the train and TriMul baseline subcommands.
    #[test]
    fn clap_parses_train_and_trimul_baseline() {
        let c = Cli::try_parse_from(["ferrl", "train", "--config", "run.json"]).unwrap();
        assert!(matches!(c.cmd, Command::Train(_)));
        // The `TrimulBaseline` variant renders as the `trimul-baseline` subcommand.
        let b = Cli::try_parse_from(["ferrl", "trimul-baseline", "--config", "run.json"]).unwrap();
        assert!(matches!(b.cmd, Command::TrimulBaseline(_)));
    }

    /// The clap surface parses the TriMul external scoring subcommand.
    #[test]
    fn clap_parses_trimul_score() {
        let s = Cli::try_parse_from([
            "ferrl",
            "trimul-score",
            "--config",
            "run.json",
            "--prompt-copy",
            "runs/trimul-1/prompt.txt",
            "--completion",
            "raw.txt",
            "--out",
            "scores.jsonl",
            "--score-secret-seed",
            "424399",
            "--run-id",
            "gemma4-rollout",
            "--model-family",
            "gemma4",
            "--source-label",
            "gemma4-batch",
            "--completion-normalization",
            "llama-cpp",
        ])
        .unwrap();
        let Command::TrimulScore(a) = s.cmd else {
            panic!("expected trimul-score");
        };
        let a = *a;
        assert_eq!(
            (
                a.config,
                a.prompt_copy,
                a.completion,
                a.completion_normalization,
                a.out,
                a.score_secret_seed,
                a.run_id,
                a.model_family,
                a.source_label,
            ),
            (
                PathBuf::from("run.json"),
                PathBuf::from("runs/trimul-1/prompt.txt"),
                vec![PathBuf::from("raw.txt")],
                CompletionNormalization::LlamaCpp,
                PathBuf::from("scores.jsonl"),
                424399,
                "gemma4-rollout".to_string(),
                "gemma4".to_string(),
                "gemma4-batch".to_string(),
            )
        );
    }

    #[test]
    fn trimul_score_rejects_training_secret_seed_before_prompt_io() {
        let tmp = TestDir::new("trimul-score-seed");
        std::fs::write(tmp.path().join("run.json"), trimul_score_test_config(4242)).unwrap();
        let mut args = trimul_score_args_for_test(tmp.path());
        args.score_secret_seed = 4242;

        let err = trimul_score(&args).unwrap_err().to_string();

        assert!(err.contains("requires --score-secret-seed to differ"));
    }

    #[test]
    fn trimul_score_verifies_prompt_copy_before_reading_inputs() {
        let tmp = TestDir::new("trimul-score-prompt");
        std::fs::write(tmp.path().join("run.json"), trimul_score_test_config(4242)).unwrap();
        write_prompt_copy(tmp.path(), b"prompt", "0000");
        let mut args = trimul_score_args_for_test(tmp.path());
        args.score_secret_seed = 4243;

        let err = trimul_score(&args).unwrap_err().to_string();

        assert!(err.contains("prompt copy hash mismatch"));
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // compact table-style coverage for parsing defaults
    fn trimul_score_jsonl_defaults_and_source_ids_are_public_safe() {
        let tmp = TestDir::new("trimul-score-jsonl");
        let raw_path = tmp.path().join("private-raw-completion.txt");
        let jsonl_path = tmp.path().join("private-inputs.jsonl");
        std::fs::write(&raw_path, "raw-completion").unwrap();
        std::fs::write(
            &jsonl_path,
            concat!(
                r#"{"completion":"row-one","completion_len_tokens":13,"metadata":{"kind":"defaulted"}}"#,
                "\n",
                r#"{"completion":"row-two","step":22,"prompt_index":3,"group_index":5,"rank":1,"world_size":2,"source_id":"public-row-2","reward_metadata":{"raw":true}}"#,
                "\n",
            ),
        )
        .unwrap();
        let mut args = trimul_score_args_for_test(tmp.path());
        args.completion = vec![raw_path.clone()];
        args.completions_jsonl = vec![jsonl_path];
        args.source_label = "gemma4-public".to_string();

        let inputs = read_trimul_score_inputs(&args).unwrap();
        let observed: Vec<_> = inputs
            .iter()
            .map(|i| {
                (
                    i.source_id.as_str(),
                    i.step,
                    i.prompt_index,
                    i.group_index,
                    i.rank,
                    i.world_size,
                    i.completion_len_tokens,
                )
            })
            .collect();

        assert_eq!(
            observed,
            vec![
                ("gemma4-public:completion:0", 9, 8, 0, 2, 3, None),
                ("gemma4-public:jsonl:0:line:1", 9, 8, 1, 2, 3, Some(13)),
                ("public-row-2", 22, 3, 5, 1, 2, None),
            ]
        );
        assert!(!inputs[0]
            .source_id
            .contains(raw_path.to_string_lossy().as_ref()));
        assert_eq!(inputs[1].metadata.as_ref().unwrap()["kind"], "defaulted");
        assert_eq!(inputs[2].reward_metadata.as_ref().unwrap()["raw"], true);
    }

    #[test]
    fn trimul_score_normalizes_llama_cpp_completion_sentinel() {
        let tmp = TestDir::new("trimul-score-normalization");
        let raw_path = tmp.path().join("candidate.txt");
        std::fs::write(
            &raw_path,
            "prefix\n```python\ndef custom_kernel(data):\n    return data\n``` [end of text]\n\n",
        )
        .unwrap();
        let mut args = trimul_score_args_for_test(tmp.path());
        args.completion = vec![raw_path];
        args.completion_normalization = CompletionNormalization::LlamaCpp;

        let inputs = read_trimul_score_inputs(&args).unwrap();

        assert_eq!(
            inputs[0].completion,
            "prefix\n```python\ndef custom_kernel(data):\n    return data\n```\n"
        );
        let metadata = inputs[0].metadata.as_ref().unwrap();
        assert_eq!(
            metadata["ferrl_completion_normalization"]["mode"],
            "llama_cpp"
        );
        assert_eq!(
            metadata["ferrl_completion_normalization"]["normalized_completion_sha256"],
            sha256_hex(inputs[0].completion.as_bytes())
        );
    }

    #[test]
    fn trimul_score_records_llama_cpp_mode_even_when_unchanged() {
        let raw = "```python\ndef custom_kernel(data):\n    return data\n```\n".to_string();
        let completion = normalize_completion(&raw, CompletionNormalization::LlamaCpp);

        assert_eq!(completion.text, raw);
        assert!(!completion.changed);
        let metadata =
            completion_normalization_metadata(None, CompletionNormalization::LlamaCpp, &completion)
                .unwrap();
        assert_eq!(
            metadata["ferrl_completion_normalization"]["mode"],
            "llama_cpp"
        );
        assert_eq!(metadata["ferrl_completion_normalization"]["changed"], false);
    }

    #[test]
    fn trimul_score_rejects_path_like_source_ids() {
        let tmp = TestDir::new("trimul-score-source-id");
        let jsonl_path = tmp.path().join("inputs.jsonl");
        std::fs::write(
            &jsonl_path,
            r#"{"completion":"row","source_id":"/private/path/completion.txt"}"#,
        )
        .unwrap();
        let mut args = trimul_score_args_for_test(tmp.path());
        args.completions_jsonl = vec![jsonl_path];

        let err = read_trimul_score_inputs(&args).unwrap_err().to_string();

        assert!(err.contains("public-safe id"));
    }

    #[test]
    fn trimul_score_validates_rank_world_coordinates() {
        let zero_world = vec![trimul_score_input_for_test("candidate-0", 0, 0)];
        let bad_rank = vec![trimul_score_input_for_test("candidate-1", 2, 2)];

        let err_zero = validate_trimul_score_inputs(&zero_world)
            .unwrap_err()
            .to_string();
        let err_rank = validate_trimul_score_inputs(&bad_rank)
            .unwrap_err()
            .to_string();

        assert!(
            err_zero.contains("world_size = 0") && err_rank.contains("rank 2 outside world_size 2")
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // validates the public JSON row shape in one place
    fn trimul_score_record_serializes_external_provenance_without_paths() {
        let tmp = TestDir::new("trimul-score-record");
        let args = trimul_score_args_for_test(tmp.path());
        let mut input = trimul_score_input_for_test("public-source-7", 1, 4);
        input.source_index = 7;
        input.completion = "abc".to_string();
        input.completion_len_tokens = Some(3);
        input.metadata = Some(serde_json::json!({"input": "meta"}));
        let record = trimul_score_record(
            &args,
            &input,
            f32::NAN,
            Some("trimul:no_code".to_string()),
            Some(serde_json::json!({"reward_scheme": "trimul_shaped_v1"})),
            "prompt-hash",
            "config-hash",
        );

        let row = serde_json::to_value(record).unwrap();

        assert_eq!(row["reward"], 0.0);
        assert_eq!(row["reward_metadata"]["reward_scheme"], "trimul_shaped_v1");
        assert_eq!(row["input_metadata"]["input"], "meta");
        assert_eq!(row["completion_sha256"], sha256_hex(b"abc"));
        assert_eq!(row["external_score"]["source_id"], "public-source-7");
        assert_eq!(row["external_score"]["source_index"], 7);
        assert!(row["external_score"].get("source").is_none());
    }

    /// The clap surface parses the run-report subcommand.
    #[test]
    fn clap_parses_runreport() {
        let r = Cli::try_parse_from([
            "ferrl",
            "runreport",
            "runs/x",
            "--config",
            "run.json",
            "--json",
            "--strict",
        ])
        .unwrap();
        match r.cmd {
            Command::Runreport(a) => {
                assert!(a.json && a.strict);
                assert_eq!(a.config, Some(PathBuf::from("run.json")));
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
    fn run_health_policy_flags_s50_collapse_shape() {
        let tmp = TestDir::new("run-health-s50");
        let candidate_path = tmp.path().join("candidates.jsonl");
        let history = run_health_s50_history();
        write_candidate_jsonl(&candidate_path, run_health_s50_candidate_rows());
        let candidates = read_candidate_health_inputs(&[candidate_path])
            .unwrap()
            .unwrap();
        let summary = summarize(&history).unwrap();
        let policy = s50_run_health_policy();
        let report = policy.evaluate(
            &history,
            &summary,
            run_health_eval_ctx(4),
            Some(&candidates),
        );

        assert_eq!(report.verdict, RunHealthVerdict::Fail);
        assert_run_health_rules(
            &report,
            &[
                "reward_collapse",
                "correctness_collapse",
                "dropped_rows",
                "grad_spike",
                "source_dominance",
            ],
        );
    }

    #[test]
    fn run_health_correctness_collapse_rejects_stale_candidate_ledger() {
        let history = vec![
            run_health_test_metric(0, 2.0, 1.0),
            run_health_test_metric(1, 2.0, 1.0),
            run_health_test_metric(2, 2.0, 1.0),
            run_health_test_metric(3, 2.0, 1.0),
        ];
        let summary = summarize(&history).unwrap();
        let tmp = TestDir::new("run-health-stale-candidates");
        let candidate_path = tmp.path().join("candidates.jsonl");
        write_candidate_jsonl(
            &candidate_path,
            [
                (0, 0, true, "source-0".to_string()),
                (1, 0, true, "source-1".to_string()),
            ],
        );
        let candidates = read_candidate_health_inputs(&[candidate_path])
            .unwrap()
            .unwrap();

        let report = correctness_collapse_policy().evaluate(
            &history,
            &summary,
            run_health_eval_ctx(1),
            Some(&candidates),
        );

        assert_eq!(report.verdict, RunHealthVerdict::Fail);
        assert_run_health_rules(&report, &["correctness_collapse"]);
        assert!(report.findings[0].message.contains("2,3"));
    }

    #[test]
    fn run_health_candidate_rules_reject_empty_required_ledger() {
        let history = vec![
            run_health_test_metric(0, 2.0, 1.0),
            run_health_test_metric(1, 2.0, 1.0),
        ];
        let summary = summarize(&history).unwrap();
        let policy = RunHealthCfg {
            source_dominance: Some(FractionThresholdCfg {
                max_fraction: 0.8,
                action: HealthActionCfg::Fail,
            }),
            ..correctness_collapse_policy()
        };

        let report = policy.evaluate(
            &history,
            &summary,
            run_health_eval_ctx(1),
            Some(&CandidateHealth::default()),
        );

        assert_eq!(report.verdict, RunHealthVerdict::Fail);
        assert_run_health_rules(&report, &["correctness_collapse", "source_dominance"]);
    }

    #[test]
    fn run_health_correctness_collapse_rejects_unsupported_metadata() {
        let history = vec![
            run_health_test_metric(0, 2.0, 1.0),
            run_health_test_metric(1, 2.0, 1.0),
        ];
        let summary = summarize(&history).unwrap();
        let mut candidates = CandidateHealth {
            total: 2,
            ..CandidateHealth::default()
        };
        for step in 0..=1 {
            let mut step_health = CandidateStepHealth {
                total: 1,
                ..CandidateStepHealth::default()
            };
            step_health
                .prompt_groups
                .entry(step)
                .or_default()
                .group_indices
                .insert(0);
            candidates.steps.insert(step, step_health);
        }

        let report = correctness_collapse_policy().evaluate(
            &history,
            &summary,
            run_health_eval_ctx(1),
            Some(&candidates),
        );

        assert_eq!(report.verdict, RunHealthVerdict::Fail);
        assert!(report.findings[0].message.contains("metadata unavailable"));
    }

    #[test]
    fn run_health_candidate_rules_reject_partial_topk_coverage() {
        let tmp = TestDir::new("run-health-partial-topk");
        let candidate_path = tmp.path().join("candidates.jsonl");
        let history = vec![
            run_health_test_metric(0, 2.0, 1.0),
            run_health_test_metric(1, 2.0, 1.0),
        ];
        write_candidate_jsonl(
            &candidate_path,
            [
                (0, 0, true, "dominant".to_string()),
                (1, 0, true, "dominant".to_string()),
            ],
        );
        let candidates = read_candidate_health_inputs(&[candidate_path])
            .unwrap()
            .unwrap();
        let summary = summarize(&history).unwrap();
        let policy = RunHealthCfg {
            source_dominance: Some(FractionThresholdCfg {
                max_fraction: 0.8,
                action: HealthActionCfg::Fail,
            }),
            ..correctness_collapse_policy()
        };

        let report = policy.evaluate(
            &history,
            &summary,
            run_health_eval_ctx(2),
            Some(&candidates),
        );

        assert_eq!(report.verdict, RunHealthVerdict::Fail);
        assert_run_health_rules(&report, &["correctness_collapse", "source_dominance"]);
        assert!(report
            .findings
            .iter()
            .all(|finding| finding.message.contains("full group coverage")));
    }

    #[test]
    fn run_health_windowed_rules_reject_insufficient_history() {
        let history = vec![run_health_test_metric(0, 2.0, 1.0)];
        let summary = summarize(&history).unwrap();
        let policy = RunHealthCfg {
            reward_collapse: Some(WindowThresholdCfg {
                window: 2,
                min: 1.0,
                action: HealthActionCfg::Fail,
            }),
            ..correctness_collapse_policy()
        };

        let report = policy.evaluate(&history, &summary, run_health_eval_ctx(1), None);

        assert_eq!(report.verdict, RunHealthVerdict::Fail);
        assert_run_health_rules(&report, &["reward_collapse", "correctness_collapse"]);
        assert!(report
            .findings
            .iter()
            .all(|finding| finding.message.contains("only 1 metric rows")));
    }

    #[test]
    fn runreport_config_policy_exits_two_on_fail() {
        let tmp = TestDir::new("runreport-policy");
        let run = tmp.path().join("run-001");
        std::fs::create_dir_all(&run).unwrap();
        let history = vec![
            run_health_test_metric(0, 2.0, 1.0),
            run_health_test_metric(1, 0.05, 1.0),
            run_health_test_metric(2, 0.05, 1.0),
        ];
        write_metrics_jsonl(&run.join("metrics.jsonl"), &history);
        std::fs::write(
            tmp.path().join("run.json"),
            r#"{
                "task": "countdown",
                "model_dir": "/m",
                "run_health": {
                  "reward_collapse": { "window": 2, "min": 1.0, "action": "fail" }
                },
                "trainer": { "steps": 3, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }
            }"#,
        )
        .unwrap();

        let code = runreport(&RunreportArgs {
            path: run,
            config: Some(tmp.path().join("run.json")),
            json: false,
            strict: false,
        })
        .unwrap();

        assert_eq!(code, ExitCode::from(2));
    }

    #[test]
    fn runreport_config_policy_reads_candidate_sibling_for_metrics_file() {
        let tmp = TestDir::new("runreport-policy-metrics-file");
        let run = tmp.path().join("run-001");
        std::fs::create_dir_all(&run).unwrap();
        let history = vec![
            run_health_test_metric(0, 2.0, 1.0),
            run_health_test_metric(1, 2.0, 1.0),
        ];
        write_metrics_jsonl(&run.join("metrics.jsonl"), &history);
        write_candidate_jsonl(
            &run.join("candidates.jsonl"),
            [
                (0, 0, false, "source-0".to_string()),
                (0, 1, false, "source-0".to_string()),
                (1, 0, false, "source-1".to_string()),
                (1, 1, false, "source-1".to_string()),
            ],
        );
        std::fs::write(
            tmp.path().join("run.json"),
            r#"{
                "task": "countdown",
                "model_dir": "/m",
                "run_health": {
                  "correctness_collapse": { "window": 2, "min": 0.5, "action": "fail" }
                },
                "trainer": { "steps": 2, "group_size": 2, "candidate_log_top_k": 2,
                  "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }
            }"#,
        )
        .unwrap();

        let code = runreport(&RunreportArgs {
            path: run.join("metrics.jsonl"),
            config: Some(tmp.path().join("run.json")),
            json: false,
            strict: false,
        })
        .unwrap();

        assert_eq!(code, ExitCode::from(2));
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

    #[test]
    fn candidate_health_gate_fails_diagnostic_regressions() {
        let mut failures = Vec::new();
        compare_candidate_health(
            Some(CandidateHealth {
                diagnostics: 0,
                ..CandidateHealth::default()
            }),
            Some(CandidateHealth {
                diagnostics: 1,
                ..CandidateHealth::default()
            }),
            &mut failures,
        );

        assert_eq!(
            failures,
            vec![RegressionFailure::CandidateDiagnostics {
                baseline: 0,
                candidate: 1,
            }]
        );
    }

    #[test]
    fn candidate_health_gate_is_inert_without_ledgers() {
        let mut failures = Vec::new();
        compare_candidate_health(None, None, &mut failures);
        assert!(failures.is_empty());
    }

    #[test]
    fn candidate_health_buckets_missing_and_null_source_hashes() {
        let tmp = TestDir::new("candidate-health-source");
        let candidate_path = tmp.path().join("candidates.jsonl");
        std::fs::write(
            &candidate_path,
            concat!(
                r#"{"step":0,"rank":0,"world_size":1,"prompt_index":0,"group_index":0,"reward":0.0,"completion_len_tokens":8,"completion":"old"}"#,
                "\n",
                r#"{"step":0,"rank":0,"world_size":1,"prompt_index":0,"group_index":1,"reward":0.05,"completion_len_tokens":9,"reward_metadata":{"source_sha256":null},"completion":"null"}"#,
                "\n",
                r#"{"step":0,"rank":0,"world_size":1,"prompt_index":0,"group_index":2,"reward":2.0,"completion_len_tokens":10,"reward_metadata":{"source_sha256":"abc123"},"completion":"ok"}"#,
                "\n",
            ),
        )
        .unwrap();

        let health = read_candidate_health_inputs(&[candidate_path])
            .unwrap()
            .unwrap();

        assert_eq!(health.source_buckets["__unknown_source__"], 2);
        assert_eq!(health.source_buckets["abc123"], 1);
    }

    /// The clap surface parses the artifact subcommand.
    #[test]
    fn clap_parses_trimul_artifact() {
        let a = Cli::try_parse_from([
            "ferrl",
            "trimul-artifact",
            "--config",
            "run.json",
            "--prompt-copy",
            "runs/trimul-1/prompt.txt",
            "--completion",
            "completion.txt",
            "--completion-normalization",
            "llama-cpp",
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
                assert_eq!(a.prompt_copy, PathBuf::from("runs/trimul-1/prompt.txt"));
                assert_eq!(
                    a.completion_normalization,
                    CompletionNormalization::LlamaCpp
                );
            }
            _ => panic!("expected trimul-artifact"),
        }
    }

    /// A `trimul` run config parses, with its task block and a baseline pin.
    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn parses_a_trimul_config() {
        let prompt_path = std::env::temp_dir().join(format!(
            "ferrl-trimul-prompt-parse-{}.txt",
            std::process::id()
        ));
        std::fs::write(&prompt_path, "Parse-test custom_kernel(data) prompt.\n").unwrap();
        let json = r#"{ "task": "trimul", "model_dir": "/m",
                        "device": "cuda",
                        "data": { "train_n": 8, "eval_n": 2 },
                        "trimul": { "image": "/img.sif", "eval_dir": "/eval",
                          "prompt_path": "__PROMPT_PATH__",
                          "submission_extract_mode": "thinking_after_think",
                          "scratch_root": "/tmp", "scratch_max_bytes": 1048576,
                          "secret_seed": 123, "wall_secs": 300,
                          "verifier_cuda_visible_devices": "1",
                          "verifier_cuda_device_pool": ["1", "2"],
                          "verifier_parallelism": 2,
                          "verifier_max_procs": 2048,
                          "baseline": { "ns": 5200000.0, "gpu": "H100" } },
                        "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                          "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                          "lr": 1e-5, "weight_decay": 0.0,
                          "loss_type": "grpo", "scale_rewards": "group" } }"#
            .replace("__PROMPT_PATH__", &prompt_path.display().to_string());
        let cfg: RunConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.task, "trimul");
        assert_eq!((cfg.trimul.secret_seed, cfg.trimul.wall_secs), (123, 300));
        assert_eq!(cfg.trimul.scratch_max_bytes, 1_048_576);
        assert_eq!(
            cfg.trimul.verifier_cuda_visible_devices.as_deref(),
            Some("1")
        );
        assert_eq!(cfg.trimul.verifier_cuda_device_pool, ["1", "2"]);
        assert_eq!(cfg.trimul.verifier_parallelism, 2);
        assert_eq!(cfg.trimul.verifier_max_procs, 2048);
        let b = cfg.trimul.baseline.as_ref().expect("baseline present");
        assert_eq!((b.ns, b.gpu.as_str()), (5_200_000.0, "H100"));
        // The single-prompt splits honour train_n / eval_n without deduping to one row.
        let (train, eval) = cfg.trimul_splits().unwrap();
        assert_eq!((train.len(), eval.len()), (8, 2));
        assert_eq!(train[0].prompt, "Parse-test custom_kernel(data) prompt.\n");
        std::fs::remove_file(prompt_path).unwrap();
    }

    /// The verifier sandbox settings are not just parsed: they reach the run spec.
    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn trimul_config_wires_verifier_sandbox_settings_in_reward() {
        let eval_dir =
            std::env::temp_dir().join(format!("ferrl-trimul-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(
            eval_dir.join("task.yml"),
            r#"
tests:
  - {"seqlen": 8, "bs": 1, "dim": 16, "hiddendim": 16, "seed": 100, "nomask": True, "distribution": "normal"}
benchmarks:
  - {"seqlen": 16, "bs": 1, "dim": 32, "hiddendim": 16, "seed": 200, "nomask": True, "distribution": "normal"}
"#,
        )
        .unwrap();
        let config_json = |verifier_max_procs_field: &str| {
            format!(
                r#"{{
                "task": "trimul",
                "model_dir": "/m",
                "trimul": {{
                  "image": "/img.sif",
                  "eval_dir": "{}",
                  "scratch_root": "/tmp",
                  "verifier_cuda_visible_devices": "1",
                  {}
                  "reward": {{ "format_extracted": 0.03, "runnable": 0.07, "partial_correctness": 0.70 }}
                }},
                "trainer": {{ "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }}
            }}"#,
                eval_dir.display(),
                verifier_max_procs_field
            )
        };
        let json = config_json(r#""verifier_max_procs": 2048,"#);
        let cfg: RunConfig = serde_json::from_str(&json).unwrap();
        let reward = cfg.build_trimul_reward_base().unwrap();
        let spec = reward.build_run_spec(std::path::Path::new("/tmp/scratch"));

        assert_eq!(reward.reward_profile().format_extracted, 0.03);
        assert_eq!(reward.reward_profile().runnable, 0.07);
        assert_eq!(reward.reward_profile().partial_correctness, 0.70);
        assert!(spec
            .env
            .iter()
            .any(|(k, v)| k == "CUDA_VISIBLE_DEVICES" && v == "1"));
        assert_eq!(spec.limits.max_procs, Some(2048));

        let omitted_cfg: RunConfig = serde_json::from_str(&config_json("")).unwrap();
        let omitted_spec = omitted_cfg
            .build_trimul_reward_base()
            .unwrap()
            .build_run_spec(std::path::Path::new("/tmp/scratch"));
        assert_eq!(
            omitted_spec.limits.max_procs,
            Some(ferrl::trimul::DEFAULT_VERIFIER_MAX_PROCS)
        );

        let zero_json = config_json(r#""verifier_max_procs": 0,"#);
        let zero_cfg: RunConfig = serde_json::from_str(&zero_json).unwrap();
        let zero_spec = zero_cfg
            .build_trimul_reward_base()
            .unwrap()
            .build_run_spec(std::path::Path::new("/tmp/scratch"));
        assert_eq!(
            zero_spec.limits.max_procs,
            Some(ferrl::trimul::DEFAULT_VERIFIER_MAX_PROCS)
        );
    }

    /// TriMul prompt loading is exact; extraction mode is parser-only and does not wrap text.
    #[test]
    fn trimul_prompt_path_is_exact_and_extraction_mode_is_parser_only() {
        let prompt_path = std::env::temp_dir().join(format!(
            "ferrl-trimul-prompt-exact-{}.txt",
            std::process::id()
        ));
        let prompt = "<|im_start|>system\nManaged system prompt.<|im_end|>\n\
<|im_start|>user\nManaged custom_kernel(data) task.\n<|im_end|>\n\
<|im_start|>assistant\n<think>\n";
        std::fs::write(&prompt_path, prompt).unwrap();
        let json = r#"{ "task": "trimul", "model_dir": "/m",
                        "trimul": {
                          "prompt_path": "__PROMPT_PATH__",
                          "submission_extract_mode": "thinking_after_think"
                        },
                        "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                          "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                          "lr": 1e-5, "weight_decay": 0.0,
                          "loss_type": "grpo", "scale_rewards": "group" } }"#
            .replace("__PROMPT_PATH__", &prompt_path.display().to_string());
        let cfg: RunConfig = serde_json::from_str(&json).unwrap();
        let (train, eval) = cfg.trimul_splits().unwrap();
        assert_eq!((train.len(), eval.len()), (64, 0));
        assert!(matches!(
            cfg.trimul_submission_extract_mode().unwrap(),
            ferrl::trimul::SubmissionExtractMode::ThinkingAfterThink
        ));
        assert_eq!(train[0].prompt, prompt);
        assert!(!train[0]
            .prompt
            .contains("Use at most 8 short reasoning lines"));
        assert!(!train[0].prompt.contains("Output contract:"));
        std::fs::remove_file(prompt_path).unwrap();
    }

    /// `prompt_path` owns the whole rendered model prompt; ferrl must not trim or wrap it.
    #[test]
    fn trimul_prompt_path_replaces_all_prompt_construction() {
        let prompt_path = std::env::temp_dir().join(format!(
            "ferrl-trimul-prompt-replace-{}.txt",
            std::process::id()
        ));
        let prompt = "\n  Invent a fast custom_kernel(data). Return correct values.  \n";
        std::fs::write(&prompt_path, prompt).unwrap();
        let json = format!(
            r#"{{
                "task": "trimul",
                "model_dir": "/m",
                "trimul": {{
                  "prompt_path": "{}",
                  "submission_extract_mode": "final_fence"
                }},
                "trainer": {{ "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }}
            }}"#,
            prompt_path.display()
        );
        let cfg: RunConfig = serde_json::from_str(&json).unwrap();
        let (train, eval) = cfg.trimul_splits().unwrap();

        assert_eq!((train.len(), eval.len()), (64, 0));
        assert_eq!(train[0].prompt, prompt);
        assert!(!train[0]
            .prompt
            .contains("Input contract: `data` is a tuple"));
        assert!(!train[0].prompt.contains("Shape-safety rules:"));
        assert!(!train[0].prompt.starts_with("<|im_start|>system"));

        std::fs::remove_file(prompt_path).unwrap();
    }

    /// TriMul training has a single prompt owner, so `prompt_path` is required.
    #[test]
    fn trimul_prompt_path_is_required() {
        let json = r#"{
                "task": "trimul",
                "model_dir": "/m",
                "trimul": {},
                "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }
            }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        let err = cfg.trimul_splits().unwrap_err().to_string();

        assert!(err.contains("requires trimul.prompt_path"));
    }

    /// TriMul train/artifact rewards need an explicit parser because prompt text is no
    /// longer allowed to imply extraction behavior.
    #[test]
    fn trimul_submission_extract_mode_is_required_for_train_reward() {
        let json = r#"{
                "task": "trimul",
                "model_dir": "/m",
                "trimul": {},
                "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }
            }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        let err = cfg.build_trimul_reward().unwrap_err().to_string();

        assert!(err.contains("requires trimul.submission_extract_mode"));
    }

    /// Wrapper-based TriMul configs are intentionally rejected; prompt text is
    /// owned byte-for-byte by `prompt_path` now.
    #[test]
    fn trimul_prompt_format_config_is_rejected() {
        let json = r#"{
                "task": "trimul",
                "model_dir": "/m",
                "trimul": {
                  "prompt_format": "qwen3_5_chat_thinking_concise",
                  "prompt_path": "/prompt.txt",
                  "submission_extract_mode": "thinking_after_think"
                },
                "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }
            }"#;
        let err = serde_json::from_str::<RunConfig>(json).unwrap_err();

        assert!(err.to_string().contains("unknown field `prompt_format`"));
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
    fn prompt_copy_must_match_adjacent_launch_hash() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ferrl-prompt-copy-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prompt_path = dir.join("prompt.txt");
        let hash_path = dir.join("prompt.sha256");
        let prompt = b"<|im_start|>user\nrendered prompt<|im_end|>\n";

        std::fs::write(&prompt_path, prompt).unwrap();
        std::fs::write(&hash_path, format!("{}\n", sha256_hex(prompt))).unwrap();
        assert_eq!(
            read_verified_prompt_copy(&prompt_path).unwrap(),
            prompt.to_vec()
        );

        std::fs::write(&hash_path, "0000\n").unwrap();
        let err = read_verified_prompt_copy(&prompt_path)
            .unwrap_err()
            .to_string();
        assert!(err.contains("prompt copy hash mismatch"));

        std::fs::remove_dir_all(dir).unwrap();
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
    fn trimul_artifact_completion_normalization_records_llama_cpp_manifest_metadata() {
        let raw = b"```python\npass\n``` [end of text]\n\n";
        let normalized = "```python\npass\n```\n";
        let inputs = ArtifactInputs {
            gpu: "H100".to_string(),
            raw_completion: std::str::from_utf8(raw).unwrap(),
            normalized_completion: normalized,
            completion_normalization: CompletionNormalization::LlamaCpp,
            completion_normalization_changed: true,
            completion_bytes: raw,
            config_bytes: b"{}",
            prompt_bytes: b"prompt",
            submission: "pass\n",
            baseline_median: 1.0,
            test_cases: 1,
            benchmark_cases: 1,
            runs: Vec::new(),
            accepted: false,
        };

        let metadata = artifact_completion_normalization(&inputs).unwrap();

        assert_eq!(metadata.mode, "llama_cpp");
        assert!(metadata.changed);
        assert_eq!(metadata.raw_completion_len_bytes, raw.len());
        assert_eq!(metadata.normalized_completion_len_bytes, normalized.len());
        assert_eq!(
            metadata.normalized_completion_sha256,
            sha256_hex(normalized.as_bytes())
        );
        assert_eq!(
            metadata.normalized_completion_file,
            Some("completion.normalized.txt")
        );
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
                completion_normalization: None,
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
                prompt_sha256: "prompt-hash".to_string(),
                prompt_file: "prompt.txt",
                reward_profile: ferrl::trimul::TrimulRewardProfile::default(),
                trainer_steps: 100,
                group_size: 4,
                run_health: "healthy".to_string(),
                policy_seed: 11,
                data_seed: 22,
                training_secret_seed: 33,
                audit_secret_seed: 44,
                scratch_max_bytes: 1024,
                verifier_parallelism: 1,
                verifier_max_procs: ferrl::trimul::DEFAULT_VERIFIER_MAX_PROCS,
                verifier_cuda_device_pool: Vec::new(),
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
            "Prompt copy: prompt.txt (prompt-hash)",
            "Reward profile: `{\"scheme\":\"trimul_shaped_v1\"",
            "Budget: trainer_steps=100, group_size=4, scratch_max_bytes=1024, verifier_max_procs=1024",
            "Run health: healthy",
            "| source hash | training reward | source inspection | clean correctness | median runtime ns | speedup | accept/reject reason |",
            "| source-hash | 1.500000 | clean | 3/3 | 9.000000 | 1.222222 | accepted: all clean runs correct and median runtime beats baseline |",
            "Source inspection notes: no process, file descriptor, environment, network, or out-of-input path probes",
            "Path: artifact",
            "Manifest SHA-256: manifest-hash",
            "## 6. Operator Checklist",
            "[pass] audit seed differs from training seed",
            "[pass] reward profile recorded and valid",
            "[pass] verifier process cap recorded",
            "[pass] source inspection found no process/file/env/network/path probing",
        ] {
            assert!(report.contains(required), "missing report field: {required}");
        }
    }
}
