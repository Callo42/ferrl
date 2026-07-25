//! `ferrl` — the single-binary front door: train a built-in task end-to-end from a
//! JSON run config, and report on a finished run.
//!
//! ```text
//! ferrl train --config run.json                    # train a built-in task
//! ferrl trimul-baseline --config run.json   # measure the TriMul reference baseline (ns) on this GPU
//! ferrl trimul-score --config run.json --prompt-copy prompt.txt --completion raw.txt --out scores.jsonl
//! ferrl trimul-score --config run.json --prompt-copy prompt.txt --completion raw.txt --completion-normalization llama-cpp --out scores.jsonl
//! ferrl trimul-artifact --run-dir runs/trimul-1 --candidate-sha256 <record-sha256> --out artifact/ ...
//! ferrl runreport <run-dir> [--config run.json] [--json] [--strict]   # one-glance run health summary
//! ferrl perf-gate --baseline <run-dir> --candidate <run-dir>   # resource regression check
//! ```
//!
//! `train` reads a `RunConfig` (a serialized [`TrainerConfig`](ferrl::TrainerConfig)
//! plus a model directory, a device, and a task selector), loads a supported policy via
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
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use candle_core::{DType, Device};
use clap::{Args, Parser, Subcommand, ValueEnum};
use ring::signature::{UnparsedPublicKey, ED25519};
use serde::{
    de::Error as _,
    ser::{Error as _, SerializeStruct},
    Deserialize, Serialize,
};
use sha2::{Digest, Sha256};
use tracing::info;

use ferrl::countdown::{build_prompt, generate_dataset, CountdownConfig, CountdownProblem};
use ferrl::policy::{GenConfig, Policy, TensorParallelPolicy};
use ferrl::telemetry::{CandidateRecord, CandidateSigner, RegressionFailure};
use ferrl::{
    compare_distributed_metrics, compare_metrics, evaluate, read_jsonl, summarize,
    train_eval_split, BaseQuantization, CountdownReward, LoaderOpts, MathProblem, MathReward,
    RegressionBudget, RegressionReport, RewardFn, RunDir, RunStop, Sample, TensorParallelPlan,
    TokenizerLike, Trainer, TrainerConfig, TrimulReward,
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
    /// Extract and verify a TriMul artifact bundle from one launch-bound candidate row.
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
    /// Immutable run directory containing `launch.json` and `candidates.jsonl`.
    #[arg(long)]
    run_dir: PathBuf,
    /// Exact `record_sha256` of one immutable row in `candidates.jsonl`.
    #[arg(long)]
    candidate_sha256: String,
    /// Output artifact directory. Fails if `manifest.json` already exists.
    #[arg(long)]
    out: PathBuf,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
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
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
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

/// Optional quantization for frozen base projection weights.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum BaseQuantizationSel {
    /// Store frozen base projections as ordinary tensors.
    #[default]
    None,
    /// Store frozen base projections as GGML `Q8_0`.
    Q8_0,
}

impl BaseQuantizationSel {
    fn as_base_quantization(self) -> BaseQuantization {
        match self {
            Self::None => BaseQuantization::None,
            Self::Q8_0 => BaseQuantization::Q8_0,
        }
    }

    /// Stable manifest spelling for this frozen-base quantization mode.
    fn as_str(self) -> &'static str {
        self.as_base_quantization().as_str()
    }
}

/// Policy-load knobs (the `LoRA` shape, base dtype, seed). The rollout temperature
/// is taken from the trainer config so the two cannot disagree.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct PolicyCfg {
    /// `LoRA` rank.
    lora_rank: usize,
    /// `LoRA` alpha.
    lora_alpha: f64,
    /// Dtype the frozen base loads in.
    base_dtype: DtypeSel,
    /// Optional frozen-base projection quantization.
    base_quantization: BaseQuantizationSel,
    /// Rollout sampler seed.
    seed: u64,
    /// Enable layer-boundary activation checkpointing for the update forward.
    ///
    /// This trades extra recompute for a lower activation peak and is the main
    /// CLI-accessible memory lever for long supported-model GPU training runs.
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
            base_quantization: BaseQuantizationSel::None,
            seed: 1234,
            activation_checkpointing: false,
            memory_efficient_cached_gqa: false,
        }
    }
}

/// Dataset knobs: a JSONL `path` for file-backed tasks (`math`), or the generated
/// `train_n` for procedural tasks (`countdown`), plus the held-out `eval_n` and the
/// `seed` for the deterministic dedup-aware split.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
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
#[derive(Debug, Default, Deserialize, Serialize)]
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
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct DistributedCfg {
    /// When true, run this process as one rank of a Slurm/NCCL data-parallel
    /// world. Requires `--features nccl`, `device = "cuda"`, and the Slurm
    /// variables plus `FERRL_NCCL_RENDEZVOUS` expected by `NcclConfig`.
    /// Run directories are rank-suffixed to keep per-rank telemetry separate.
    enabled: bool,
}

/// Tensor-parallel launch knobs for sharded model execution.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct TensorParallelCfg {
    /// When true, bind this process to one rank of a live tensor-parallel world.
    /// Model projections execute in shards. Gemma 4 also streams rank-local
    /// projection weights; Qwen3 currently keeps replicated checkpoint weights.
    enabled: bool,
    /// Tensor-parallel rank in `0..world_size`.
    rank: usize,
    /// Tensor-parallel world size. The identity/default is `1`.
    world_size: usize,
}

impl Default for TensorParallelCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            rank: 0,
            world_size: 1,
        }
    }
}

impl TensorParallelCfg {
    fn plan(self) -> Result<TensorParallelPlan, CliError> {
        if !self.enabled {
            if self.rank != 0 || self.world_size != 1 {
                return Err(CliError::msg(
                    "tensor_parallel disabled requires rank = 0 and world_size = 1",
                ));
            }
            return Ok(TensorParallelPlan::single());
        }
        TensorParallelPlan::new(self.rank, self.world_size)
            .map_err(|e| CliError::msg(e.to_string()))
    }

    fn validate_current_support(self) -> Result<TensorParallelPlan, CliError> {
        self.plan()
    }
}

/// A `ferrl train` run, deserialized from JSON.
///
/// The wire shape is a flat object: a `task` selector, the `model_dir` checkpoint,
/// an optional `device` / `out_dir` / `policy` / `data` block, and the full
/// [`TrainerConfig`] under `trainer`.
#[derive(Debug)]
struct RunConfig {
    /// Which built-in task to train: `"countdown"` or `"math"`.
    task: String,
    /// Checkpoint directory (`config.json` + `model.safetensors` + `tokenizer.json`).
    model_dir: PathBuf,
    /// Device to run on (default `cpu`).
    device: DeviceSel,
    /// Where run directories are written (default `runs/`).
    out_dir: PathBuf,
    /// Policy-load knobs.
    policy: PolicyCfg,
    /// Dataset knobs.
    data: DataCfg,
    /// Data-parallel launch knobs.
    distributed: DistributedCfg,
    /// Tensor-parallel launch knobs.
    tensor_parallel: TensorParallelCfg,
    /// TriMul task knobs (only read when `task == "trimul"`).
    trimul: TrimulCfg,
    /// Discovery health policy applied after training and by `runreport --config`.
    run_health: RunHealthCfg,
    /// The GRPO trainer config.
    trainer: TrainerConfig,
    /// CLI-only interpretation of the `trainer.eos_token_id` wire value.
    eos_selection: EosSelection,
}

impl Serialize for RunConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut trainer = serde_json::to_value(&self.trainer).map_err(S::Error::custom)?;
        let trainer_object = trainer
            .as_object_mut()
            .ok_or_else(|| S::Error::custom("serialized trainer config is not an object"))?;
        match self.eos_selection {
            EosSelection::Checkpoint => {
                trainer_object.remove("eos_token_id");
            }
            EosSelection::Explicit => {
                if self.trainer.eos_token_id.is_none() {
                    return Err(S::Error::custom(
                        "explicit EOS selector lost its numeric token id",
                    ));
                }
            }
            EosSelection::Disabled => {
                trainer_object.insert(
                    "eos_token_id".into(),
                    serde_json::Value::String("none".into()),
                );
            }
        }

        let mut state = serializer.serialize_struct("RunConfig", 11)?;
        state.serialize_field("task", &self.task)?;
        state.serialize_field("model_dir", &self.model_dir)?;
        state.serialize_field("device", &self.device)?;
        state.serialize_field("out_dir", &self.out_dir)?;
        state.serialize_field("policy", &self.policy)?;
        state.serialize_field("data", &self.data)?;
        state.serialize_field("distributed", &self.distributed)?;
        state.serialize_field("tensor_parallel", &self.tensor_parallel)?;
        state.serialize_field("trimul", &self.trimul)?;
        state.serialize_field("run_health", &self.run_health)?;
        state.serialize_field("trainer", &trainer)?;
        state.end()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum EosSelection {
    /// The field was omitted: resolve the checkpoint's scalar EOS.
    #[default]
    Checkpoint,
    /// A numeric token id was supplied in the run config.
    Explicit,
    /// The exact string `"none"` was supplied in the run config.
    Disabled,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfigWire {
    task: String,
    model_dir: PathBuf,
    #[serde(default)]
    device: DeviceSel,
    #[serde(default = "default_out_dir")]
    out_dir: PathBuf,
    #[serde(default)]
    policy: PolicyCfg,
    #[serde(default)]
    data: DataCfg,
    #[serde(default)]
    distributed: DistributedCfg,
    #[serde(default)]
    tensor_parallel: TensorParallelCfg,
    #[serde(default)]
    trimul: TrimulCfg,
    #[serde(default)]
    run_health: RunHealthCfg,
    trainer: TrainerConfig,
}

impl<'de> Deserialize<'de> for RunConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut value = serde_json::Value::deserialize(deserializer)?;
        let eos_selection = value
            .get_mut("trainer")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|trainer| trainer.get_mut("eos_token_id"))
            .map_or(Ok(EosSelection::Checkpoint), |raw| match raw {
                serde_json::Value::Number(number)
                    if number
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .is_some() =>
                {
                    Ok(EosSelection::Explicit)
                }
                serde_json::Value::String(mode) if mode == "none" => {
                    *raw = serde_json::Value::Null;
                    Ok(EosSelection::Disabled)
                }
                serde_json::Value::Null => Err(D::Error::custom(
                    "trainer.eos_token_id must be omitted for checkpoint auto-resolution, an \
                     integer override, or the string \"none\"; null is not an explicit mode",
                )),
                _ => Err(D::Error::custom(
                    "trainer.eos_token_id must be an integer override or the string \"none\"",
                )),
            })?;
        let wire = RunConfigWire::deserialize(value).map_err(D::Error::custom)?;
        Ok(Self {
            task: wire.task,
            model_dir: wire.model_dir,
            device: wire.device,
            out_dir: wire.out_dir,
            policy: wire.policy,
            data: wire.data,
            distributed: wire.distributed,
            tensor_parallel: wire.tensor_parallel,
            trimul: wire.trimul,
            run_health: wire.run_health,
            trainer: wire.trainer,
            eos_selection,
        })
    }
}

/// `serde` default for [`RunConfig::out_dir`]: `runs/`.
fn default_out_dir() -> PathBuf {
    PathBuf::from("runs")
}

impl RunConfig {
    fn open_device(&self) -> Result<Device, CliError> {
        self.device.open()
    }

    /// Read and parse a run config from `path`.
    fn load(path: &Path) -> Result<Self, CliError> {
        Self::load_for_launch(path).map(|loaded| loaded.config)
    }

    /// Read, validate, and fingerprint a run config for distributed launch.
    fn load_for_launch(path: &Path) -> Result<LoadedRunConfig, CliError> {
        let bytes = std::fs::read(path).map_err(|source| CliError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let cfg: Self = serde_json::from_slice(&bytes).map_err(|source| CliError::Config {
            path: path.to_path_buf(),
            source,
        })?;
        cfg.validate_current_config_support()?;
        let resolved_config = canonicalize_json(cfg.canonical_wire_value()?);
        let resolved_bytes = serde_json::to_vec(&resolved_config)
            .map_err(|err| CliError::msg(format!("failed to canonicalize run config: {err}")))?;
        let mut consensus_config = resolved_config.clone();
        if let Some(tensor_parallel) = consensus_config
            .get_mut("tensor_parallel")
            .and_then(serde_json::Value::as_object_mut)
        {
            tensor_parallel.remove("rank");
        }
        let consensus_config = canonicalize_json(consensus_config);
        let consensus_bytes = serde_json::to_vec(&consensus_config)
            .map_err(|err| CliError::msg(format!("failed to canonicalize run config: {err}")))?;
        Ok(LoadedRunConfig {
            config: cfg,
            launch_config: LaunchConfigSnapshot {
                source_sha256: sha256_hex(&bytes),
                resolved_sha256: sha256_hex(&resolved_bytes),
                resolved: resolved_config,
            },
            consensus_digest: Sha256::digest(consensus_bytes).into(),
        })
    }

    fn validate_current_config_support(&self) -> Result<(), CliError> {
        self.trainer
            .validate()
            .map_err(|err| CliError::msg(err.to_string()))?;
        if !self.distributed.enabled
            && self.trainer.reward_group_scope == ferrl::RewardGroupScope::DistributedSamePrompt
            && self.trainer.group_size < 2
        {
            return Err(CliError::msg(
                "distributed_same_prompt group_size = 1 requires a live data-parallel world of \
                 at least two ranks",
            ));
        }
        ferrl::lora::validated_lora_scale(
            self.policy.lora_alpha,
            self.policy.lora_rank,
            self.policy.base_dtype.as_dtype(),
        )
        .map_err(|error| CliError::msg(format!("invalid policy LoRA scale: {error}")))?;
        if matches!(self.task.as_str(), "countdown" | "trimul") && self.data.train_n == 0 {
            return Err(CliError::msg(format!(
                "task {:?} requires data.train_n >= 1",
                self.task
            )));
        }
        if self.task == "countdown" {
            self.data
                .train_n
                .checked_add(self.data.eval_n)
                .ok_or_else(|| {
                    CliError::msg(
                        "countdown data.train_n + data.eval_n exceeds the supported dataset size",
                    )
                })?;
        }
        self.trimul.reward.validate().map_err(CliError::msg)?;
        self.run_health.validate_current_support(&self.trainer)?;
        let plan = self.tensor_parallel.validate_current_support()?;
        if self.tensor_parallel.enabled && self.distributed.enabled {
            return Err(CliError::msg(
                "simultaneous distributed data parallelism and tensor_parallel execution is not \
                 wired yet",
            ));
        }
        if self.tensor_parallel.enabled && self.device != DeviceSel::Cuda {
            return Err(CliError::msg(
                "tensor_parallel.enabled requires device = \"cuda\"",
            ));
        }
        if self.tensor_parallel.enabled
            && matches!(self.policy.base_quantization, BaseQuantizationSel::Q8_0)
        {
            return Err(CliError::msg(
                "tensor_parallel execution does not support \
                 policy.base_quantization = \"q8_0\" until rank-local quantized shards \
                 are implemented; disable tensor_parallel to use world-one Q8_0",
            ));
        }
        if plan.is_sharded() {
            if self.data.eval_n > 0 {
                return Err(CliError::msg(
                    "sharded tensor_parallel execution does not support held-out eval yet; \
                    set data.eval_n = 0",
                ));
            }
            if !self.policy.activation_checkpointing {
                return Err(CliError::msg(
                    "sharded tensor_parallel training requires \
                     policy.activation_checkpointing = true so replicated-boundary \
                     cotangents are reduced during backward",
                ));
            }
        }
        Ok(())
    }

    fn canonical_wire_value(&self) -> Result<serde_json::Value, CliError> {
        let mut value = serde_json::to_value(self)
            .map_err(|err| CliError::msg(format!("failed to canonicalize run config: {err}")))?;
        let trainer = value
            .get_mut("trainer")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| CliError::msg("serialized run config has no trainer object"))?;
        match self.eos_selection {
            EosSelection::Checkpoint => {
                trainer.remove("eos_token_id");
            }
            EosSelection::Explicit => {
                if self.trainer.eos_token_id.is_none() {
                    return Err(CliError::msg(
                        "explicit EOS selector lost its resolved token id",
                    ));
                }
            }
            EosSelection::Disabled => {
                trainer.insert(
                    "eos_token_id".into(),
                    serde_json::Value::String("none".into()),
                );
            }
        }
        Ok(value)
    }

    fn tensor_parallel_plan(&self) -> TensorParallelPlan {
        self.tensor_parallel
            .plan()
            .expect("RunConfig validation must validate tensor_parallel")
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
            base_quantization: self.policy.base_quantization.as_base_quantization(),
            activation_checkpointing: self.policy.activation_checkpointing,
            tensor_parallel: self.tensor_parallel_plan(),
        }
    }

    fn resolve_eos_token_id(
        &self,
        tokenizer: &ferrl::HfTokenizer,
    ) -> Result<Option<u32>, CliError> {
        let selection = match self.eos_selection {
            EosSelection::Checkpoint => ferrl::CheckpointEosSelection::CheckpointDefault,
            EosSelection::Explicit => {
                ferrl::CheckpointEosSelection::Explicit(self.trainer.eos_token_id.ok_or_else(
                    || CliError::msg("explicit EOS selector has no numeric token id"),
                )?)
            }
            EosSelection::Disabled => ferrl::CheckpointEosSelection::Disabled,
        };
        ferrl::resolve_checkpoint_eos(&self.model_dir, tokenizer, selection)
            .map_err(|error| CliError::msg(format!("checkpoint EOS resolution failed: {error}")))
    }

    fn resolved_trainer_config(
        &self,
        tokenizer: &ferrl::HfTokenizer,
    ) -> Result<TrainerConfig, CliError> {
        let mut trainer = self.trainer.clone();
        trainer.eos_token_id = self.resolve_eos_token_id(tokenizer)?;
        Ok(trainer)
    }

    /// Build the Countdown train/eval splits: generate `train_n + eval_n` problems
    /// and hold out `eval_n` via the dedup-aware [`train_eval_split`].
    fn countdown_splits(&self) -> Splits<CountdownProblem> {
        let cd = CountdownConfig::default();
        let n = self.data.train_n + self.data.eval_n;
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
            .take(self.data.train_n)
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
        let assets = self.capture_trimul_verifier_assets()?;
        self.build_trimul_reward_base_with_assets(assets)
    }

    fn capture_trimul_verifier_assets(
        &self,
    ) -> Result<ferrl::trimul::TrimulVerifierAssets, CliError> {
        let t = &self.trimul;
        ferrl::trimul::TrimulVerifierAssets::capture(&t.image, &t.eval_dir, &t.scratch_root)
            .map_err(|error| CliError::msg(error.to_string()))
    }

    fn build_trimul_reward_base_with_assets(
        &self,
        assets: ferrl::trimul::TrimulVerifierAssets,
    ) -> Result<TrimulReward, CliError> {
        let t = &self.trimul;
        let (tests, benches) = ferrl::trimul::parse_task_yml(assets.task_yml())?;
        let wall = Duration::from_secs(if t.wall_secs == 0 { 600 } else { t.wall_secs });
        let mut reward = TrimulReward::new(&t.image, &t.eval_dir, &t.scratch_root)
            .with_cases(tests, benches)
            .with_secret_seed(t.secret_seed)
            .with_wall(wall)
            .with_verifier_assets(assets);
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
        self.trimul_submission_extract_mode()?;
        let assets = self.capture_trimul_verifier_assets()?;
        self.build_trimul_reward_with_assets(assets)
    }

    fn build_trimul_reward_with_assets(
        &self,
        assets: ferrl::trimul::TrimulVerifierAssets,
    ) -> Result<TrimulReward, CliError> {
        let mode = self.trimul_submission_extract_mode()?;
        let mut reward = self
            .build_trimul_reward_base_with_assets(assets)?
            .with_submission_extract_mode(mode);
        if let Some(b) = &self.trimul.baseline {
            guard_baseline_gpu(&b.gpu)?;
            reward = reward.with_baseline_ns(b.ns);
        }
        Ok(reward)
    }
}

struct LoadedRunConfig {
    config: RunConfig,
    launch_config: LaunchConfigSnapshot,
    consensus_digest: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchConfigSnapshot {
    source_sha256: String,
    resolved_sha256: String,
    resolved: serde_json::Value,
}

const LAUNCH_CONTRACT_VERSION: u32 = 1;
const LAUNCH_KIND: &str = "ferrl.run-launch";
const CANDIDATE_RECORD_DOMAIN: &str = CandidateRecord::DIGEST_DOMAIN;
const LAUNCH_ATTESTATION_CONTRACT_VERSION: u32 = 1;
const LAUNCH_ATTESTATION_KIND: &str = "ferrl.run-launch-attestation";
const LAUNCH_ATTESTATION_ALGORITHM: &str = "ed25519";
const LAUNCH_ATTESTATION_DOMAIN: &str = "ferrl.run-launch-attestation.v1";
const LAUNCH_ATTESTATION_REQUEST_KIND: &str = "ferrl.run-launch-attestation-request";
const LAUNCH_TRUST_POLICY_KIND: &str = "ferrl.run-launch-trust-policy";
const LAUNCH_ATTESTOR_SOCKET: &str = "/run/ferrl/launch-attestor.sock";
const LAUNCH_TRUST_POLICY: &str = "/etc/ferrl/launch-trust.json";
const MAX_ATTESTATION_RESPONSE_BYTES: u64 = 16 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchManifest {
    contract_version: u32,
    kind: String,
    payload_sha256: String,
    payload: LaunchPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    attestation: Option<LaunchAttestation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchPayload {
    task: String,
    ferrl_commit: String,
    run: LaunchRunIdentity,
    config: LaunchConfigSnapshot,
    model: LaunchModelIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<LaunchPromptIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verifier: Option<ferrl::trimul::TrimulVerifierIdentity>,
    candidate_ledger: LaunchCandidateLedger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchRunIdentity {
    group_id: String,
    run_id: String,
    data_parallel_rank: usize,
    data_parallel_world_size: usize,
    tensor_parallel_rank: usize,
    tensor_parallel_world_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchModelIdentity {
    family: String,
    checkpoint_policy_sha256: String,
    tokenizer_sha256: String,
    resolved_eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchPromptIdentity {
    file: String,
    sha256: String,
    len_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchCandidateLedger {
    file: String,
    format_version: u32,
    row_digest_domain: String,
    row_signature_algorithm: String,
    signing_public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchAttestation {
    contract_version: u32,
    kind: String,
    algorithm: String,
    key_id: String,
    launch_payload_sha256: String,
    signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchAttestationRequest {
    contract_version: u32,
    kind: String,
    algorithm: String,
    launch_payload_sha256: String,
    launch_payload_json_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchTrustPolicy {
    contract_version: u32,
    kind: String,
    keys: Vec<LaunchTrustKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchTrustKey {
    key_id: String,
    algorithm: String,
    public_key: String,
}

trait LaunchAttestor {
    fn attest(&self, manifest: &LaunchManifest) -> Result<LaunchAttestation, CliError>;
}

struct SystemLaunchAttestor;

#[derive(Debug, Clone)]
struct LaunchContext {
    ferrl_commit: String,
    run: LaunchRunIdentity,
    config: LaunchConfigSnapshot,
}

#[derive(Debug, Clone)]
struct BuildSourceIdentity {
    commit: String,
    dirty: bool,
}

fn embedded_build_source_identity() -> Result<BuildSourceIdentity, CliError> {
    validated_build_source_identity(
        env!("FERRL_BUILD_GIT_COMMIT"),
        env!("FERRL_BUILD_GIT_DIRTY") == "true",
    )
}

fn validated_build_source_identity(
    commit: &str,
    dirty: bool,
) -> Result<BuildSourceIdentity, CliError> {
    let commit = validate_full_git_commit(commit).map_err(|_| {
        CliError::msg("ferrl train requires a Git-built binary with an embedded full source commit")
    })?;
    if dirty {
        return Err(CliError::msg(
            "ferrl train refuses a binary built from a dirty source tree; commit the exact source before building",
        ));
    }
    Ok(BuildSourceIdentity {
        commit,
        dirty: false,
    })
}

impl LaunchManifest {
    fn new(payload: LaunchPayload) -> Result<Self, CliError> {
        let payload_bytes = serde_json::to_vec(&payload)
            .map_err(|error| CliError::msg(format!("serialize launch payload: {error}")))?;
        Ok(Self {
            contract_version: LAUNCH_CONTRACT_VERSION,
            kind: LAUNCH_KIND.to_owned(),
            payload_sha256: domain_sha256("ferrl.run-launch.payload.v1", &[&payload_bytes]),
            payload,
            attestation: None,
        })
    }

    fn attest(mut self, attestor: &dyn LaunchAttestor) -> Result<Self, CliError> {
        if self.attestation.is_some() {
            return Err(CliError::msg("launch manifest is already attested"));
        }
        self.attestation = Some(attestor.attest(&self)?);
        Ok(self)
    }

    fn to_pretty_bytes(&self) -> Result<Vec<u8>, CliError> {
        serde_json::to_vec_pretty(self)
            .map_err(|error| CliError::msg(format!("serialize launch manifest: {error}")))
    }
}

impl LaunchAttestor for SystemLaunchAttestor {
    fn attest(&self, manifest: &LaunchManifest) -> Result<LaunchAttestation, CliError> {
        let trust_policy = load_system_launch_trust_policy()?;
        request_launch_attestation(manifest, &trust_policy)
    }
}

fn launch_attestation_message(launch_payload_sha256: &str) -> String {
    domain_sha256(
        LAUNCH_ATTESTATION_DOMAIN,
        &[launch_payload_sha256.as_bytes()],
    )
}

fn valid_attestation_key_id(key_id: &str) -> bool {
    !key_id.is_empty()
        && key_id.len() <= 128
        && key_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':'))
}

fn decode_lower_hex(label: &str, value: &str, expected_bytes: usize) -> Result<Vec<u8>, CliError> {
    validate_lower_hex(label, value, expected_bytes)?;
    Ok(value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let nibble = |byte| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => unreachable!("validate_lower_hex checked every byte"),
            };
            (nibble(pair[0]) << 4) | nibble(pair[1])
        })
        .collect())
}

fn lower_hex_bytes(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn validate_launch_trust_policy(policy: &LaunchTrustPolicy) -> Result<(), CliError> {
    if policy.contract_version != LAUNCH_ATTESTATION_CONTRACT_VERSION
        || policy.kind != LAUNCH_TRUST_POLICY_KIND
        || policy.keys.is_empty()
    {
        return Err(CliError::msg(
            "launch trust policy has an unsupported contract or no keys",
        ));
    }
    let mut key_ids = BTreeSet::new();
    for key in &policy.keys {
        if !valid_attestation_key_id(&key.key_id) || key.algorithm != LAUNCH_ATTESTATION_ALGORITHM {
            return Err(CliError::msg(format!(
                "launch trust policy key {:?} has an invalid id or algorithm",
                key.key_id
            )));
        }
        validate_lower_hex("launch trust public_key", &key.public_key, 32)?;
        if !key_ids.insert(key.key_id.as_str()) {
            return Err(CliError::msg(format!(
                "launch trust policy repeats key id {:?}",
                key.key_id
            )));
        }
    }
    Ok(())
}

fn verify_launch_attestation(
    manifest: &LaunchManifest,
    trust_policy: &LaunchTrustPolicy,
) -> Result<(), CliError> {
    validate_launch_trust_policy(trust_policy)?;
    let attestation = manifest
        .attestation
        .as_ref()
        .ok_or_else(|| CliError::msg("launch manifest has no trusted external attestation"))?;
    if attestation.contract_version != LAUNCH_ATTESTATION_CONTRACT_VERSION
        || attestation.kind != LAUNCH_ATTESTATION_KIND
        || attestation.algorithm != LAUNCH_ATTESTATION_ALGORITHM
        || attestation.launch_payload_sha256 != manifest.payload_sha256
        || !valid_attestation_key_id(&attestation.key_id)
    {
        return Err(CliError::msg(
            "launch manifest has an invalid external attestation envelope",
        ));
    }
    let key = trust_policy
        .keys
        .iter()
        .find(|key| key.key_id == attestation.key_id)
        .ok_or_else(|| {
            CliError::msg(format!(
                "launch attestation key {:?} is not trusted",
                attestation.key_id
            ))
        })?;
    let public_key = decode_lower_hex("launch trust public_key", &key.public_key, 32)?;
    let signature = decode_lower_hex("launch attestation signature", &attestation.signature, 64)?;
    let message = launch_attestation_message(&manifest.payload_sha256);
    UnparsedPublicKey::new(&ED25519, public_key)
        .verify(message.as_bytes(), &signature)
        .map_err(|_| CliError::msg("launch attestation signature is invalid"))
}

#[cfg(unix)]
#[allow(clippy::cognitive_complexity)] // one linear ownership/type/parent-chain validation
fn require_root_owned_protected_path(path: &Path, expect_socket: bool) -> Result<(), CliError> {
    use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};

    let metadata = std::fs::symlink_metadata(path).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let expected_type = if expect_socket {
        metadata.file_type().is_socket()
    } else {
        metadata.file_type().is_file()
    };
    let unsafe_mode = if expect_socket {
        metadata.mode() & 0o002 != 0
    } else {
        metadata.mode() & 0o022 != 0
    };
    if !expected_type || metadata.uid() != 0 || unsafe_mode {
        return Err(CliError::msg(format!(
            "external launch trust path {} must be a protected root-owned {}",
            path.display(),
            if expect_socket {
                "non-world-writable Unix socket"
            } else {
                "non-group/world-writable regular file"
            }
        )));
    }
    for parent in path.ancestors().skip(1) {
        let parent_metadata = std::fs::symlink_metadata(parent).map_err(|source| CliError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        if !parent_metadata.file_type().is_dir()
            || parent_metadata.uid() != 0
            || parent_metadata.mode() & 0o022 != 0
        {
            return Err(CliError::msg(format!(
                "external launch trust parent {} is not root-owned and protected",
                parent.display()
            )));
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn require_root_owned_protected_path(_path: &Path, _expect_socket: bool) -> Result<(), CliError> {
    Err(CliError::msg(
        "external launch attestation currently requires a Unix platform",
    ))
}

fn load_system_launch_trust_policy() -> Result<LaunchTrustPolicy, CliError> {
    let path = Path::new(LAUNCH_TRUST_POLICY);
    require_root_owned_protected_path(path, false)?;
    let bytes = read_regular_bytes(path)?;
    let policy: LaunchTrustPolicy =
        serde_json::from_slice(&bytes).map_err(|source| CliError::Config {
            path: path.to_path_buf(),
            source,
        })?;
    validate_launch_trust_policy(&policy)?;
    Ok(policy)
}

fn exchange_launch_attestation<S: IoRead + IoWrite>(
    stream: &mut S,
    manifest: &LaunchManifest,
    trust_policy: &LaunchTrustPolicy,
) -> Result<LaunchAttestation, CliError> {
    verify_launch_manifest_payload(manifest)?;
    let request = LaunchAttestationRequest {
        contract_version: LAUNCH_ATTESTATION_CONTRACT_VERSION,
        kind: LAUNCH_ATTESTATION_REQUEST_KIND.to_owned(),
        algorithm: LAUNCH_ATTESTATION_ALGORITHM.to_owned(),
        launch_payload_sha256: manifest.payload_sha256.clone(),
        launch_payload_json_hex: lower_hex_bytes(&serde_json::to_vec(&manifest.payload).map_err(
            |error| CliError::msg(format!("serialize launch payload for attestation: {error}")),
        )?),
    };
    serde_json::to_writer(&mut *stream, &request)
        .map_err(|error| CliError::msg(format!("serialize launch attestation request: {error}")))?;
    stream
        .write_all(b"\n")
        .map_err(|error| CliError::msg(format!("write launch attestation request: {error}")))?;
    stream
        .flush()
        .map_err(|error| CliError::msg(format!("flush launch attestation request: {error}")))?;
    let mut response = Vec::new();
    stream
        .take(MAX_ATTESTATION_RESPONSE_BYTES + 1)
        .read_to_end(&mut response)
        .map_err(|error| CliError::msg(format!("read launch attestor response: {error}")))?;
    if response.len() as u64 > MAX_ATTESTATION_RESPONSE_BYTES {
        return Err(CliError::msg("launch attestor response exceeds 16 KiB"));
    }
    let attestation: LaunchAttestation = serde_json::from_slice(&response)
        .map_err(|error| CliError::msg(format!("parse launch attestor response: {error}")))?;
    let mut attested = manifest.clone();
    attested.attestation = Some(attestation.clone());
    verify_launch_attestation(&attested, trust_policy)?;
    Ok(attestation)
}

#[cfg(unix)]
fn request_launch_attestation(
    manifest: &LaunchManifest,
    trust_policy: &LaunchTrustPolicy,
) -> Result<LaunchAttestation, CliError> {
    use std::os::unix::net::UnixStream;

    let socket = Path::new(LAUNCH_ATTESTOR_SOCKET);
    require_root_owned_protected_path(socket, true)?;
    let mut stream = UnixStream::connect(socket).map_err(|source| CliError::Io {
        path: socket.to_path_buf(),
        source,
    })?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|source| CliError::Io {
            path: socket.to_path_buf(),
            source,
        })?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|source| CliError::Io {
            path: socket.to_path_buf(),
            source,
        })?;
    exchange_launch_attestation(&mut stream, manifest, trust_policy)
}

#[cfg(not(unix))]
fn request_launch_attestation(
    _manifest: &LaunchManifest,
    _trust_policy: &LaunchTrustPolicy,
) -> Result<LaunchAttestation, CliError> {
    Err(CliError::msg(
        "external launch attestation currently requires a Unix platform",
    ))
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize_json).collect())
        }
        serde_json::Value::Object(values) => {
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize_json(value)))
                    .collect(),
            )
        }
        scalar => scalar,
    }
}

/// Dispatch `ferrl train`: parse the config, open the device, build the named task's
/// data, and run training.
fn train(args: &TrainArgs) -> Result<(), CliError> {
    let _ = ferrl::init_tracing();
    let launch_runtime = open_launch_runtime()?;
    train_with_launch_runtime_and_source_result(
        args,
        launch_runtime,
        embedded_build_source_identity(),
        prepare_launch_device,
    )
}

#[cfg(test)]
fn train_with_launch_runtime(
    args: &TrainArgs,
    launch_runtime: Option<LaunchRuntime>,
    build_source: BuildSourceIdentity,
    prepare_device: impl FnOnce(&RunConfig, Option<&LaunchRuntime>) -> Result<Device, CliError>,
) -> Result<(), CliError> {
    train_with_launch_runtime_and_source_result(
        args,
        launch_runtime,
        Ok(build_source),
        prepare_device,
    )
}

fn train_with_launch_runtime_and_source_result(
    args: &TrainArgs,
    launch_runtime: Option<LaunchRuntime>,
    local_build_source: Result<BuildSourceIdentity, CliError>,
    prepare_device: impl FnOnce(&RunConfig, Option<&LaunchRuntime>) -> Result<Device, CliError>,
) -> Result<(), CliError> {
    let launch_comm = launch_runtime.as_ref().map(|runtime| runtime.comm.as_ref());
    let build_source = coordinate_distributed_result(
        launch_comm,
        "embedded build source validation",
        local_build_source,
    )?;
    let loaded = coordinate_distributed_result(
        launch_comm,
        "run config load",
        RunConfig::load_for_launch(&args.config),
    )?;
    let cfg = loaded.config;
    validate_launch_runtime(&cfg, launch_runtime.as_ref())?;
    validate_launch_config_consensus(&loaded.consensus_digest, launch_comm)?;
    debug_assert!(!build_source.dirty);
    let ferrl_commit = build_source.commit;
    validate_launch_value_consensus("training commit", ferrl_commit.as_bytes(), launch_comm)?;
    let launch = LaunchContext {
        ferrl_commit,
        run: synchronized_run_identity(&cfg, launch_comm)?,
        config: loaded.launch_config,
    };
    let data_parallel_world = if cfg.distributed.enabled {
        launch_comm
            .ok_or_else(|| {
                CliError::msg(
                    "distributed execution has no live communicator after launch validation",
                )
            })?
            .world_size()
    } else {
        1
    };
    cfg.trainer
        .validate_reward_group_world(data_parallel_world)
        .map_err(|error| CliError::msg(error.to_string()))?;
    let local_device = prepare_device(&cfg, launch_runtime.as_ref());
    let device = coordinate_distributed_result(launch_comm, "device setup", local_device)?;
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
                None,
                &launch,
                launch_runtime,
            )
        }
        "math" => {
            let (train, eval) = coordinate_distributed_result(
                launch_comm,
                "math dataset setup",
                cfg.math_splits(),
            )?;
            run_training(
                &cfg,
                &device,
                &MathReward::default(),
                &train,
                &eval,
                None,
                None,
                &launch,
                launch_runtime,
            )
        }
        "trimul" => {
            let (prompt_file_bytes, train, eval, reward, verifier_assets) =
                coordinate_distributed_result(
                    launch_comm,
                    "TriMul reward and dataset setup",
                    (|| {
                        let prompt_file_bytes = cfg.trimul_prompt_file_bytes()?;
                        let prompt = cfg.trimul_prompt_text(&prompt_file_bytes)?;
                        let (train, eval) = cfg.trimul_splits_from_prompt(&prompt);
                        let verifier_assets = cfg.capture_trimul_verifier_assets()?;
                        let reward =
                            cfg.build_trimul_reward_with_assets(verifier_assets.clone())?;
                        Ok((prompt_file_bytes, train, eval, reward, verifier_assets))
                    })(),
                )?;
            run_training(
                &cfg,
                &device,
                &reward,
                &train,
                &eval,
                Some(&prompt_file_bytes),
                Some(&verifier_assets),
                &launch,
                launch_runtime,
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
#[allow(clippy::too_many_arguments)] // one typed task launch plus immutable launch context
fn run_training<R: RewardFn>(
    cfg: &RunConfig,
    device: &Device,
    reward: &R,
    train: &[Sample<R::Target>],
    eval: &[Sample<R::Target>],
    rendered_prompt_bytes: Option<&[u8]>,
    verifier_assets: Option<&ferrl::trimul::TrimulVerifierAssets>,
    launch: &LaunchContext,
    launch_runtime: Option<LaunchRuntime>,
) -> Result<(), CliError> {
    let launch_attestor = SystemLaunchAttestor;
    run_training_with_loader(
        cfg,
        device,
        reward,
        train,
        eval,
        rendered_prompt_bytes,
        verifier_assets,
        launch,
        launch_runtime,
        Some(&launch_attestor),
        |model_dir, device, opts| {
            ferrl::load_auto_policy_with_identity(model_dir, device, opts).map_err(CliError::from)
        },
    )
}

/// CLI-only policy capabilities that are intentionally inherent on [`ferrl::AutoPolicy`].
///
/// Keeping this narrow adapter separate from the public [`Policy`] contract lets the
/// production loader seam remain mutation-sensitive in tests without widening the library API.
trait CliTrainingPolicy: Policy + TensorParallelPolicy {
    fn supports_cli_tensor_parallel(&self) -> bool;
}

impl CliTrainingPolicy for ferrl::AutoPolicy {
    fn supports_cli_tensor_parallel(&self) -> bool {
        self.supports_tensor_parallel()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_training_with_loader<P, R>(
    cfg: &RunConfig,
    device: &Device,
    reward: &R,
    train: &[Sample<R::Target>],
    eval: &[Sample<R::Target>],
    rendered_prompt_bytes: Option<&[u8]>,
    verifier_assets: Option<&ferrl::trimul::TrimulVerifierAssets>,
    launch: &LaunchContext,
    launch_runtime: Option<LaunchRuntime>,
    launch_attestor: Option<&dyn LaunchAttestor>,
    load_policy: impl FnOnce(
        &Path,
        &Device,
        &LoaderOpts,
    )
        -> Result<(P, ferrl::HfTokenizer, ferrl::PolicyLoadIdentity), CliError>,
) -> Result<(), CliError>
where
    P: CliTrainingPolicy,
    R: RewardFn,
{
    let tensor_parallel_plan = cfg.tensor_parallel_plan();
    let (tensor_parallel_runtime, distributed_launch_comm, distributed_comm) =
        if cfg.tensor_parallel.enabled {
            (launch_runtime, None, None)
        } else if cfg.distributed.enabled {
            let runtime = launch_runtime
                .ok_or_else(|| CliError::msg("distributed launch runtime is missing"))?;
            let comm = SharedComm::from_box(runtime.comm);
            (None, Some(comm.clone()), Some(comm))
        } else {
            (None, None, None)
        };
    let tensor_parallel_comm = tensor_parallel_runtime
        .as_ref()
        .map(|runtime| runtime.comm.as_ref());
    let launch_comm = tensor_parallel_comm.or_else(|| {
        distributed_launch_comm
            .as_ref()
            .map(|comm| comm as &dyn ferrl::Comm)
    });
    info!(
        task = %cfg.task,
        steps = cfg.trainer.steps,
        group_size = cfg.trainer.group_size,
        activation_checkpointing = cfg.policy.activation_checkpointing,
        train = train.len(),
        eval = eval.len(),
        tensor_parallel_rank = tensor_parallel_plan.rank(),
        tensor_parallel_world = tensor_parallel_plan.world_size(),
        "ferrl train: starting"
    );

    let model_setup = (|| {
        let loader_opts = cfg.loader_opts();
        let (policy, tok, identity) = load_policy(&cfg.model_dir, device, &loader_opts)?;
        let checkpoint_policy_sha256 = cfg
            .trainer
            .checkpoint_every
            .map(|_| identity.policy_sha256.clone());
        let tcfg = cfg.resolved_trainer_config(&tok)?;
        if cfg.tensor_parallel.enabled && !policy.supports_cli_tensor_parallel() {
            return Err(CliError::msg(
                "loaded checkpoint family does not support tensor_parallel execution; supported \
                 families are qwen3 (including legacy configs without model_type) and dense \
                 gemma4/gemma4_unified; qwen3_5/qwen3_5_moe (Qwen3.5/3.6) are unsupported",
            ));
        }
        if tensor_parallel_plan.is_sharded() && !policy.supports_sharded_tensor_parallel_backward()
        {
            return Err(CliError::msg(
                "sharded tensor_parallel training is supported only for dense \
                 gemma4/gemma4_unified policies with activation checkpointing; the loaded \
                 policy does not provide cross-rank backward semantics",
            ));
        }
        Ok((policy, tok, tcfg, identity, checkpoint_policy_sha256))
    })();
    let (mut policy, tok, tcfg, policy_identity, checkpoint_policy_sha256) =
        coordinate_distributed_result(launch_comm, "model and EOS setup", model_setup)?;
    validate_resolved_eos_consensus(tcfg.eos_token_id, launch_comm)?;
    let prompt_sha256 = rendered_prompt_bytes.map(sha256_hex);
    let verifier_identity = verifier_assets.map(|assets| assets.identity().clone());
    let common_provenance = serde_json::to_vec(&(
        &launch.ferrl_commit,
        &launch.run.group_id,
        &policy_identity.policy_sha256,
        &policy_identity.tokenizer_sha256,
        policy_identity.model_family,
        &prompt_sha256,
        &verifier_identity,
    ))
    .map_err(|error| CliError::msg(format!("serialize launch provenance: {error}")))?;
    validate_launch_value_consensus(
        "model/checkpoint/tokenizer/prompt provenance",
        &common_provenance,
        launch_comm,
    )?;
    let gen = GenConfig::from(&tcfg);

    let attestation_setup = (|| {
        let candidate_signer = CandidateSigner::generate()?;
        let signing_public_key = candidate_signer.public_key_hex();
        let manifest = LaunchManifest::new(LaunchPayload {
            task: cfg.task.clone(),
            ferrl_commit: launch.ferrl_commit.clone(),
            run: launch.run.clone(),
            config: launch.config.clone(),
            model: LaunchModelIdentity {
                family: policy_identity.model_family.to_owned(),
                checkpoint_policy_sha256: policy_identity.policy_sha256.clone(),
                tokenizer_sha256: policy_identity.tokenizer_sha256.clone(),
                resolved_eos_token_id: tcfg.eos_token_id,
            },
            prompt: rendered_prompt_bytes.map(|bytes| LaunchPromptIdentity {
                file: RunDir::PROMPT_FILE.to_owned(),
                sha256: sha256_hex(bytes),
                len_bytes: bytes.len(),
            }),
            verifier: verifier_identity,
            candidate_ledger: LaunchCandidateLedger {
                file: RunDir::CANDIDATES_FILE.to_owned(),
                format_version: 1,
                row_digest_domain: CANDIDATE_RECORD_DOMAIN.to_owned(),
                row_signature_algorithm: "ed25519".to_owned(),
                signing_public_key,
            },
        })?;
        let manifest = if cfg.trainer.candidate_log_top_k > 0 {
            let attestor = launch_attestor.ok_or_else(|| {
                CliError::msg(
                    "launch-authenticated candidate logging requires an external attestor",
                )
            })?;
            manifest.attest(attestor)?
        } else {
            manifest
        };
        Ok((candidate_signer, manifest))
    })();
    let (candidate_signer, manifest) = coordinate_distributed_result(
        launch_comm,
        "external launch attestation",
        attestation_setup,
    )?;
    coordinate_distributed_result(
        launch_comm,
        "attested verifier asset revalidation",
        verifier_assets.map_or(Ok(()), |assets| {
            assets
                .verify_current()
                .map_err(|error| CliError::msg(error.to_string()))
        }),
    )?;

    let publication_setup = (|| {
        let launch_sha256 = manifest.payload_sha256.clone();
        let manifest_bytes = manifest.to_pretty_bytes()?;
        let run = RunDir::create(&cfg.out_dir, launch.run.run_id.clone())?;
        run.write_immutable_launch(&manifest_bytes, rendered_prompt_bytes)?;
        let trainer = open_trainer(
            tcfg,
            &run,
            distributed_comm,
            checkpoint_policy_sha256.as_deref(),
            &launch_sha256,
            candidate_signer,
        )?;
        Ok((run, trainer))
    })();
    let (run, mut trainer) = coordinate_distributed_result(
        launch_comm,
        "run directory and trainer setup",
        publication_setup,
    )?;
    let (history, _stop) = train_with_optional_tensor_parallel(
        &mut trainer,
        &mut policy,
        reward,
        &tok,
        train,
        tensor_parallel_comm,
    )?;
    run_on_tensor_parallel_primary(tensor_parallel_comm, "post-run health", || {
        if let Some(summary) = summarize(&history) {
            info!(steps = summary.steps, "ferrl train: complete");
            apply_train_run_health_policy(cfg, &history, &summary, &run)?;
        }
        Ok(())
    })?;

    if !eval.is_empty() {
        let report = evaluate(&mut policy, reward, &tok, eval, &gen)?;
        info!(
            base = report.base_reward_mean,
            adapter = report.adapter_reward_mean,
            improvement = report.improvement(),
            "ferrl train: held-out eval (adapter vs base)"
        );
    }

    run_on_tensor_parallel_primary(tensor_parallel_comm, "run completion output", || {
        println!("ferrl: run complete -> {}", run.root().display());
        println!(
            "ferrl: inspect with `ferrl runreport {}`",
            run.root().display()
        );
        Ok(())
    })
}

fn train_with_optional_tensor_parallel<P, R>(
    trainer: &mut Trainer,
    policy: &mut P,
    reward: &R,
    tokenizer: &dyn TokenizerLike,
    train: &[Sample<R::Target>],
    tensor_parallel_comm: Option<&dyn ferrl::Comm>,
) -> Result<(Vec<ferrl::Metrics>, RunStop), CliError>
where
    P: Policy + TensorParallelPolicy,
    R: RewardFn,
{
    match tensor_parallel_comm {
        Some(comm) => Ok(trainer.train_tensor_parallel(policy, reward, tokenizer, train, comm)?),
        None => Ok(trainer.train(policy, reward, tokenizer, train)?),
    }
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
    distributed_comm: Option<SharedComm>,
    checkpoint_policy_sha256: Option<&str>,
    candidate_launch_sha256: &str,
    candidate_signer: CandidateSigner,
) -> Result<Trainer, CliError> {
    let trainer = if let Some(comm) = distributed_comm {
        Trainer::with_comm(config, run, comm)?
    } else {
        Trainer::new(config, run)?
    };
    let trainer = if let Some(digest) = checkpoint_policy_sha256 {
        trainer.with_checkpoint_policy_sha256(digest)
    } else {
        trainer
    };
    Ok(trainer.with_candidate_provenance(candidate_launch_sha256, candidate_signer)?)
}

#[derive(Clone)]
struct SharedComm {
    inner: std::sync::Arc<std::sync::Mutex<Box<dyn ferrl::Comm>>>,
    rank: usize,
    world_size: usize,
}

impl SharedComm {
    fn from_box(comm: Box<dyn ferrl::Comm>) -> Self {
        let rank = comm.rank();
        let world_size = comm.world_size();
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(comm)),
            rank,
            world_size,
        }
    }

    fn with_comm<T>(
        &self,
        op: impl FnOnce(&dyn ferrl::Comm) -> Result<T, ferrl::CommError>,
    ) -> Result<T, ferrl::CommError> {
        let comm = self.inner.lock().map_err(|_| {
            ferrl::CommError::Poisoned("shared launch communicator mutex was poisoned".into())
        })?;
        op(comm.as_ref())
    }
}

impl std::fmt::Debug for SharedComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedComm")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish_non_exhaustive()
    }
}

impl ferrl::Comm for SharedComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce_sum(
        &self,
        tensors: &mut Vec<candle_core::Tensor>,
    ) -> Result<(), ferrl::CommError> {
        self.with_comm(|comm| comm.all_reduce_sum(tensors))
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, ferrl::CommError> {
        self.with_comm(|comm| comm.all_reduce_scalar_sum(value))
    }
}

struct LaunchRuntime {
    #[cfg_attr(not(any(feature = "nccl", test)), allow(dead_code))]
    device: Device,
    comm: Box<dyn ferrl::Comm>,
}

fn open_launch_runtime() -> Result<Option<LaunchRuntime>, CliError> {
    if std::env::var_os("FERRL_NCCL_RENDEZVOUS").is_none() {
        return Ok(None);
    }
    open_nccl_launch_runtime().map(Some)
}

fn validate_launch_runtime(
    cfg: &RunConfig,
    runtime: Option<&LaunchRuntime>,
) -> Result<(), CliError> {
    let Some(runtime) = runtime else {
        if cfg.tensor_parallel.enabled || cfg.distributed.enabled {
            return Err(CliError::msg(
                "distributed or tensor_parallel execution requires \
                 FERRL_NCCL_RENDEZVOUS and a matching Slurm launch",
            ));
        }
        return Ok(());
    };
    let comm = runtime.comm.as_ref();
    let tp_count = comm.all_reduce_scalar_sum(if cfg.tensor_parallel.enabled {
        1.0
    } else {
        0.0
    })?;
    let dp_count = comm.all_reduce_scalar_sum(if cfg.distributed.enabled { 1.0 } else { 0.0 })?;
    let world = comm.world_size() as f64;
    if (tp_count, dp_count) == (world, 0.0) {
        validate_tensor_parallel_runtime(cfg.tensor_parallel_plan(), comm)
    } else if (tp_count, dp_count) == (0.0, world) {
        Ok(())
    } else {
        Err(CliError::msg(format!(
            "launch ranks disagree on execution mode: tensor_parallel enabled on {tp_count:.0}/{world:.0} \
             ranks and distributed enabled on {dp_count:.0}/{world:.0} ranks"
        )))
    }
}

fn validate_tensor_parallel_runtime(
    plan: TensorParallelPlan,
    comm: &dyn ferrl::Comm,
) -> Result<(), CliError> {
    let local = ferrl::validate_comm_plan(plan, comm).map_err(|err| {
        CliError::msg(format!(
            "tensor_parallel config does not match the live communicator: {err}"
        ))
    });
    coordinate_distributed_result(Some(comm), "tensor_parallel config validation", local)
}

fn validate_full_git_commit(value: &str) -> Result<String, CliError> {
    let valid_len = matches!(value.len(), 40 | 64);
    let valid_hex = value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid_len && valid_hex {
        Ok(value.to_owned())
    } else {
        Err(CliError::msg(
            "git commit must be a full 40- or 64-character lowercase SHA",
        ))
    }
}

fn synchronized_run_identity(
    cfg: &RunConfig,
    comm: Option<&dyn ferrl::Comm>,
) -> Result<LaunchRunIdentity, CliError> {
    let local_stamp = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .map_err(|error| CliError::msg(format!("system clock precedes Unix epoch: {error}")))
    };
    let stamp = match comm.filter(|comm| comm.world_size() > 1) {
        Some(comm) => {
            let local = if comm.rank() == 0 {
                local_stamp()
            } else {
                Ok(0)
            };
            let local = coordinate_distributed_result(Some(comm), "run timestamp", local)?;
            let reduced = comm.all_reduce_scalar_sum(local as f64)?;
            if !reduced.is_finite()
                || reduced < 0.0
                || reduced.fract() != 0.0
                || reduced > (1_u64 << 53) as f64
            {
                return Err(CliError::msg(format!(
                    "distributed run timestamp is not an exact u64: {reduced:?}"
                )));
            }
            reduced as u64
        }
        None => local_stamp()?,
    };
    let group_id = format!("{}-{stamp}", cfg.task);
    let tensor_parallel = cfg.tensor_parallel_plan();
    let (data_parallel_rank, data_parallel_world_size) = if cfg.distributed.enabled {
        let comm = comm.ok_or_else(|| {
            CliError::msg("distributed run identity requires a live communicator")
        })?;
        (comm.rank(), comm.world_size())
    } else {
        (0, 1)
    };
    let run_id = if cfg.distributed.enabled {
        format!("{group_id}-rank{data_parallel_rank}")
    } else if tensor_parallel.is_sharded() {
        format!("{group_id}-rank{}", tensor_parallel.rank())
    } else {
        group_id.clone()
    };
    Ok(LaunchRunIdentity {
        group_id,
        run_id,
        data_parallel_rank,
        data_parallel_world_size,
        tensor_parallel_rank: tensor_parallel.rank(),
        tensor_parallel_world_size: tensor_parallel.world_size(),
    })
}

fn validate_launch_value_consensus(
    label: &'static str,
    value: &[u8],
    comm: Option<&dyn ferrl::Comm>,
) -> Result<(), CliError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return Ok(());
    };
    let digest: [u8; 32] = Sha256::digest(value).into();
    let world = comm.world_size() as f64;
    let mut mismatch = false;
    for byte in digest {
        let scalar = f64::from(byte);
        mismatch |= comm.all_reduce_scalar_sum(scalar)? != world * scalar;
    }
    let local = if mismatch {
        Err(CliError::msg(format!(
            "launch ranks disagree on {label}; all ranks must bind identical bytes"
        )))
    } else {
        Ok(())
    };
    coordinate_distributed_result(Some(comm), "launch provenance consensus", local)
}

fn validate_launch_config_consensus(
    digest: &[u8; 32],
    comm: Option<&dyn ferrl::Comm>,
) -> Result<(), CliError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return Ok(());
    };
    let world = comm.world_size() as f64;
    let mut mismatch = false;
    for byte in digest {
        let value = f64::from(*byte);
        let sum = comm.all_reduce_scalar_sum(value)?;
        mismatch |= sum != world * value;
    }
    let local = if mismatch {
        Err(CliError::msg(
            "launch ranks disagree on run config outside tensor_parallel.rank; configs must \
             otherwise be identical",
        ))
    } else {
        Ok(())
    };
    coordinate_distributed_result(Some(comm), "run config consensus", local)
}

fn validate_resolved_eos_consensus(
    eos_token_id: Option<u32>,
    comm: Option<&dyn ferrl::Comm>,
) -> Result<(), CliError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return Ok(());
    };
    ferrl::validate_resolved_eos_consensus(eos_token_id, comm)
        .map_err(|error| CliError::msg(error.to_string()))
}

fn coordinate_distributed_result<T>(
    comm: Option<&dyn ferrl::Comm>,
    label: &'static str,
    local: Result<T, CliError>,
) -> Result<T, CliError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return local;
    };
    let failed_local = if local.is_err() { 1.0 } else { 0.0 };
    let failed_global = comm.all_reduce_scalar_sum(failed_local);
    match (local, failed_global) {
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err.into()),
        (Ok(_), Ok(failed)) if failed > 0.0 => Err(CliError::msg(format!(
            "{label} failed on a peer distributed rank; aborting in lockstep"
        ))),
        (Ok(value), Ok(_)) => Ok(value),
    }
}

fn run_on_tensor_parallel_primary(
    comm: Option<&dyn ferrl::Comm>,
    label: &'static str,
    op: impl FnOnce() -> Result<(), CliError>,
) -> Result<(), CliError> {
    let local = if comm.is_none_or(|comm| comm.world_size() <= 1 || comm.rank() == 0) {
        op()
    } else {
        Ok(())
    };
    coordinate_distributed_result(comm, label, local)
}

#[cfg(feature = "nccl")]
fn open_nccl_launch_runtime() -> Result<LaunchRuntime, CliError> {
    let comm = ferrl::NcclComm::from_slurm_env()?;
    let device = comm.device().clone();
    Ok(LaunchRuntime {
        device,
        comm: Box::new(comm),
    })
}

#[cfg(not(feature = "nccl"))]
fn open_nccl_launch_runtime() -> Result<LaunchRuntime, CliError> {
    Err(CliError::msg(
        "distributed or tensor_parallel execution requires building ferrl with --features nccl",
    ))
}

#[cfg(feature = "nccl")]
fn prepare_launch_device(
    cfg: &RunConfig,
    runtime: Option<&LaunchRuntime>,
) -> Result<Device, CliError> {
    let Some(runtime) = runtime else {
        return cfg.open_device();
    };
    if cfg.device != DeviceSel::Cuda {
        return Err(CliError::msg(
            "distributed or tensor_parallel execution requires device = \"cuda\"",
        ));
    }
    let device = &runtime.device;
    if let Some(w) = ferrl::check_driver_compat(device).warning() {
        tracing::warn!("{w}");
    }
    ferrl::guard_first_kernel(device)?;
    Ok(device.clone())
}

#[cfg(not(feature = "nccl"))]
fn prepare_launch_device(
    cfg: &RunConfig,
    runtime: Option<&LaunchRuntime>,
) -> Result<Device, CliError> {
    if runtime.is_some() {
        Err(CliError::msg(
            "distributed or tensor_parallel execution requires building ferrl with --features nccl",
        ))
    } else {
        cfg.open_device()
    }
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
    let rewards: Vec<f32> = outcomes.iter().map(|outcome| outcome.reward).collect();
    validate_trimul_score_rewards(&rewards)?;

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
    println!("ferrl: max_reward {max_reward}");
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

fn validate_trimul_score_rewards(rewards: &[f32]) -> Result<(), CliError> {
    if let Some((index, reward)) = rewards
        .iter()
        .enumerate()
        .find(|(_, reward)| !reward.is_finite())
    {
        return Err(CliError::msg(format!(
            "trimul scoring returned non-finite reward {reward:?} at group index {index}"
        )));
    }
    Ok(())
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
    debug_assert!(reward.is_finite());
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
        reward,
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
    /// Digest of the immutable launch payload that owns this candidate.
    launch_sha256: String,
    /// SHA-256 of the exact `launch.json` copied into the artifact bundle.
    launch_file_sha256: String,
    /// Trusted external key that attested the launch before rollout.
    launch_attestation_key_id: String,
    /// Signature algorithm used by the trusted launch attestor.
    launch_attestation_algorithm: String,
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
    /// Domain-separated digest stored on the selected candidate row.
    record_sha256: String,
    /// Ed25519 authentication made by the externally attested per-run key.
    record_signature: String,
    /// SHA-256 of the exact JSONL row bytes copied to `candidate.json`.
    ledger_row_sha256: String,
    /// Optimizer step where this candidate was sampled.
    step: u64,
    /// Global prompt ordinal where this candidate was sampled.
    prompt_index: u64,
    /// Candidate group index where this candidate was sampled.
    group_index: usize,
    /// Data-parallel rank that sampled this candidate.
    rank: usize,
    /// Data-parallel world size for the training run.
    world_size: usize,
    /// Training reward recorded when this candidate was selected.
    training_reward: f32,
    /// SHA-256 of the raw completion text.
    completion_sha256: String,
    /// SHA-256 of `submission.py`.
    source_sha256: String,
    /// Operator-facing source-inspection evidence.
    source_inspection: SourceInspectionManifest,
}

/// Model provenance fields.
#[derive(Debug, Serialize)]
struct ModelManifest {
    /// Loader-derived model family.
    family: String,
    /// Exact model/checkpoint bytes plus loader execution semantics.
    checkpoint_policy_sha256: String,
    /// Exact tokenizer bytes used by the training process.
    tokenizer_sha256: String,
    /// `LoRA` rank.
    lora_rank: usize,
    /// `LoRA` alpha.
    lora_alpha: f64,
    /// Frozen base dtype.
    base_dtype: &'static str,
    /// Frozen base projection quantization.
    base_quantization: &'static str,
}

/// Run-config provenance fields.
#[derive(Debug, Serialize)]
struct ArtifactConfigManifest {
    /// SHA-256 of the original run-config file bytes seen at launch.
    run_config_source_sha256: String,
    /// SHA-256 of the complete canonical resolved launch config.
    run_config_resolved_sha256: String,
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
    /// Configured eval bundle path (informational; the digest is authoritative).
    bundle_path: String,
    /// SHA-256 of every ordered relative file name and byte in the eval bundle.
    bundle_sha256: String,
    /// Number of regular files bound by `bundle_sha256`.
    bundle_file_count: usize,
    /// Configured sandbox image path (informational; the digest is authoritative).
    sandbox_image_path: String,
    /// SHA-256 of the exact sandbox image bytes.
    sandbox_image_sha256: String,
    /// Exact sandbox image length.
    sandbox_image_len_bytes: u64,
    /// SHA-256 of the exact case-selecting `task.yml` bytes.
    task_yml_sha256: String,
    /// Exact `task.yml` length.
    task_yml_len_bytes: usize,
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

#[derive(Debug)]
struct BoundRunCandidate {
    launch: LaunchManifest,
    launch_bytes: Vec<u8>,
    config: RunConfig,
    prompt_bytes: Vec<u8>,
    candidate: CandidateRecord,
    candidate_row_bytes: Vec<u8>,
}

#[allow(clippy::cognitive_complexity)] // linear fail-closed validation of every provenance layer
fn load_bound_run_candidate_with_trust(
    run_dir: &Path,
    candidate_sha256: &str,
    trust_policy: &LaunchTrustPolicy,
) -> Result<BoundRunCandidate, CliError> {
    validate_lower_sha256("--candidate-sha256", candidate_sha256)?;
    let launch_path = run_dir.join(RunDir::LAUNCH_FILE);
    let launch_bytes = read_regular_bytes(&launch_path)?;
    let launch = parse_exact_launch_manifest(&launch_path, &launch_bytes)?;
    verify_launch_manifest_payload(&launch)?;
    verify_launch_attestation(&launch, trust_policy)?;
    let run_name = run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            CliError::msg(format!(
                "run directory {} has no UTF-8 final component",
                run_dir.display()
            ))
        })?;
    if run_name != launch.payload.run.run_id {
        return Err(CliError::msg(format!(
            "run directory name {run_name:?} does not match launch run_id {:?}",
            launch.payload.run.run_id
        )));
    }

    let resolved_bytes =
        serde_json::to_vec(&canonicalize_json(launch.payload.config.resolved.clone()))
            .map_err(|error| CliError::msg(format!("serialize resolved launch config: {error}")))?;
    let resolved_sha256 = sha256_hex(&resolved_bytes);
    if resolved_sha256 != launch.payload.config.resolved_sha256 {
        return Err(CliError::msg(format!(
            "resolved launch config hash mismatch: recorded {}, computed {resolved_sha256}",
            launch.payload.config.resolved_sha256
        )));
    }
    let config = parse_run_config(&launch_path, &resolved_bytes)?;
    if config.task != "trimul" || launch.payload.task != "trimul" {
        return Err(CliError::msg(
            "trimul-artifact requires a launch whose task is exactly \"trimul\"",
        ));
    }
    verify_launch_config_identity(&launch, &config)?;
    let prompt = launch
        .payload
        .prompt
        .as_ref()
        .ok_or_else(|| CliError::msg("TriMul launch manifest is missing prompt provenance"))?;
    if prompt.file != RunDir::PROMPT_FILE {
        return Err(CliError::msg(format!(
            "unsupported launch prompt file {:?}",
            prompt.file
        )));
    }
    let prompt_bytes = read_regular_bytes(&run_dir.join(&prompt.file))?;
    if prompt_bytes.len() != prompt.len_bytes || sha256_hex(&prompt_bytes) != prompt.sha256 {
        return Err(CliError::msg(
            "launch-bound prompt bytes do not match launch.json",
        ));
    }
    let ledger = &launch.payload.candidate_ledger;
    if ledger.file != RunDir::CANDIDATES_FILE
        || ledger.format_version != 1
        || ledger.row_digest_domain != CANDIDATE_RECORD_DOMAIN
        || ledger.row_signature_algorithm != "ed25519"
    {
        return Err(CliError::msg(
            "unsupported candidate-ledger contract in launch.json",
        ));
    }
    let ledger_path = run_dir.join(&ledger.file);
    let ledger_bytes = read_regular_bytes(&ledger_path)?;
    if !ledger_bytes.is_empty() && !ledger_bytes.ends_with(b"\n") {
        return Err(CliError::msg(format!(
            "candidate ledger {} has an unterminated final row",
            ledger_path.display()
        )));
    }
    let ledger_text = std::str::from_utf8(&ledger_bytes).map_err(|error| {
        CliError::msg(format!(
            "candidate ledger {} is not UTF-8: {error}",
            ledger_path.display()
        ))
    })?;
    let mut selected = None;
    for (index, raw_line) in ledger_text.split_terminator('\n').enumerate() {
        if raw_line.trim().is_empty() {
            return Err(CliError::msg(format!(
                "candidate ledger {} contains blank row {}",
                ledger_path.display(),
                index + 1
            )));
        }
        let record = parse_strict_candidate_row(&ledger_path, index + 1, raw_line)?;
        record.verify_signed_provenance(&ledger.signing_public_key)?;
        if record.launch_sha256.as_deref() != Some(launch.payload_sha256.as_str()) {
            return Err(CliError::msg(format!(
                "candidate ledger {} row {} belongs to a different launch",
                ledger_path.display(),
                index + 1
            )));
        }
        if record.rank != launch.payload.run.data_parallel_rank
            || record.world_size != launch.payload.run.data_parallel_world_size
        {
            return Err(CliError::msg(format!(
                "candidate ledger {} row {} rank/world disagree with launch.json",
                ledger_path.display(),
                index + 1
            )));
        }
        if record.step >= config.trainer.steps
            || record.group_index >= config.trainer.group_size
            || record.completion_len_tokens > config.trainer.max_new_tokens
        {
            return Err(CliError::msg(format!(
                "candidate ledger {} row {} coordinates exceed the launch config",
                ledger_path.display(),
                index + 1
            )));
        }
        if record.record_sha256.as_deref() == Some(candidate_sha256) {
            if selected.is_some() {
                return Err(CliError::msg(format!(
                    "candidate digest {candidate_sha256} occurs more than once in {}",
                    ledger_path.display()
                )));
            }
            selected = Some((record, raw_line.as_bytes().to_vec()));
        }
    }
    let (candidate, candidate_row_bytes) = selected.ok_or_else(|| {
        CliError::msg(format!(
            "candidate digest {candidate_sha256} was not found in {}",
            ledger_path.display()
        ))
    })?;
    Ok(BoundRunCandidate {
        launch,
        launch_bytes,
        config,
        prompt_bytes,
        candidate,
        candidate_row_bytes,
    })
}

fn parse_exact_launch_manifest(
    launch_path: &Path,
    launch_bytes: &[u8],
) -> Result<LaunchManifest, CliError> {
    let launch: LaunchManifest =
        serde_json::from_slice(launch_bytes).map_err(|source| CliError::Config {
            path: launch_path.to_path_buf(),
            source,
        })?;
    let canonical = launch.to_pretty_bytes()?;
    if launch_bytes != canonical {
        return Err(CliError::msg(format!(
            "launch manifest {} is not in the exact canonical production encoding",
            launch_path.display()
        )));
    }
    Ok(launch)
}

fn parse_strict_candidate_row(
    ledger_path: &Path,
    row_number: usize,
    raw_line: &str,
) -> Result<CandidateRecord, CliError> {
    const FIELDS: &[&str] = &[
        "launch_sha256",
        "record_sha256",
        "record_signature",
        "step",
        "rank",
        "world_size",
        "prompt_index",
        "group_index",
        "reward",
        "completion_len_tokens",
        "reward_diagnostic",
        "reward_metadata",
        "completion",
    ];
    let value: serde_json::Value = serde_json::from_str(raw_line).map_err(|error| {
        CliError::msg(format!(
            "parse candidate ledger {} row {row_number}: {error}",
            ledger_path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        CliError::msg(format!(
            "candidate ledger {} row {row_number} is not a JSON object",
            ledger_path.display()
        ))
    })?;
    if let Some(field) = object
        .keys()
        .find(|field| !FIELDS.contains(&field.as_str()))
    {
        return Err(CliError::msg(format!(
            "candidate ledger {} row {row_number} contains unknown field {field:?}",
            ledger_path.display()
        )));
    }
    let record: CandidateRecord = serde_json::from_value(value).map_err(|error| {
        CliError::msg(format!(
            "parse candidate ledger {} row {row_number}: {error}",
            ledger_path.display()
        ))
    })?;
    let canonical = serde_json::to_string(&record).map_err(|error| {
        CliError::msg(format!(
            "serialize candidate ledger {} row {row_number}: {error}",
            ledger_path.display()
        ))
    })?;
    if canonical != raw_line {
        return Err(CliError::msg(format!(
            "candidate ledger {} row {row_number} is not in the exact production encoding",
            ledger_path.display()
        )));
    }
    Ok(record)
}

fn verify_launch_config_identity(
    launch: &LaunchManifest,
    config: &RunConfig,
) -> Result<(), CliError> {
    let run = &launch.payload.run;
    let tensor_parallel = config.tensor_parallel_plan();
    if run.tensor_parallel_rank != tensor_parallel.rank()
        || run.tensor_parallel_world_size != tensor_parallel.world_size()
    {
        return Err(CliError::msg(
            "launch run identity disagrees with resolved tensor_parallel config",
        ));
    }
    if config.distributed.enabled {
        if run.data_parallel_world_size == 0
            || run.data_parallel_rank >= run.data_parallel_world_size
        {
            return Err(CliError::msg(
                "launch run identity has invalid data-parallel coordinates",
            ));
        }
    } else if run.data_parallel_rank != 0 || run.data_parallel_world_size != 1 {
        return Err(CliError::msg(
            "world-one launch has non-world-one data-parallel coordinates",
        ));
    }
    let prefix = format!("{}-", launch.payload.task);
    let stamp = run
        .group_id
        .strip_prefix(&prefix)
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| CliError::msg("launch group_id is not a generated task timestamp"))?;
    if run.group_id != format!("{}-{stamp}", launch.payload.task) {
        return Err(CliError::msg(
            "launch group_id is not in canonical generated form",
        ));
    }
    let expected_run_id = if config.distributed.enabled {
        format!("{}-rank{}", run.group_id, run.data_parallel_rank)
    } else if tensor_parallel.is_sharded() {
        format!("{}-rank{}", run.group_id, run.tensor_parallel_rank)
    } else {
        run.group_id.clone()
    };
    if run.run_id != expected_run_id {
        return Err(CliError::msg(format!(
            "launch run_id {:?} does not match generated identity {expected_run_id:?}",
            run.run_id
        )));
    }
    Ok(())
}

fn verify_launch_manifest_payload(manifest: &LaunchManifest) -> Result<(), CliError> {
    if manifest.contract_version != LAUNCH_CONTRACT_VERSION || manifest.kind != LAUNCH_KIND {
        return Err(CliError::msg(format!(
            "unsupported launch manifest contract {} / {:?}",
            manifest.contract_version, manifest.kind
        )));
    }
    validate_lower_sha256("launch payload_sha256", &manifest.payload_sha256)?;
    validate_lower_sha256(
        "launch config source_sha256",
        &manifest.payload.config.source_sha256,
    )?;
    validate_lower_sha256(
        "launch config resolved_sha256",
        &manifest.payload.config.resolved_sha256,
    )?;
    validate_lower_sha256(
        "launch checkpoint_policy_sha256",
        &manifest.payload.model.checkpoint_policy_sha256,
    )?;
    validate_lower_sha256(
        "launch tokenizer_sha256",
        &manifest.payload.model.tokenizer_sha256,
    )?;
    validate_lower_hex(
        "launch candidate signing_public_key",
        &manifest.payload.candidate_ledger.signing_public_key,
        32,
    )?;
    if let Some(prompt) = &manifest.payload.prompt {
        validate_lower_sha256("launch prompt sha256", &prompt.sha256)?;
    }
    match (&manifest.payload.verifier, manifest.payload.task.as_str()) {
        (Some(verifier), "trimul") => {
            validate_lower_sha256("launch verifier image_sha256", &verifier.image_sha256)?;
            validate_lower_sha256(
                "launch verifier eval_bundle_sha256",
                &verifier.eval_bundle_sha256,
            )?;
            validate_lower_sha256("launch verifier task_yml_sha256", &verifier.task_yml_sha256)?;
            if verifier.image_len_bytes == 0
                || verifier.eval_file_count == 0
                || verifier.task_yml_len_bytes == 0
            {
                return Err(CliError::msg(
                    "TriMul launch verifier identity has an empty required asset",
                ));
            }
        }
        (None, "trimul") => {
            return Err(CliError::msg(
                "TriMul launch manifest is missing verifier asset identity",
            ));
        }
        (Some(_), _) => {
            return Err(CliError::msg(
                "non-TriMul launch manifest unexpectedly carries verifier assets",
            ));
        }
        (None, _) => {}
    }
    validate_full_git_commit(&manifest.payload.ferrl_commit)?;
    if !matches!(
        manifest.payload.model.family.as_str(),
        "qwen3" | "qwen3_5" | "gemma4"
    ) {
        return Err(CliError::msg(format!(
            "launch manifest has unsupported model family {:?}",
            manifest.payload.model.family
        )));
    }
    if manifest.payload.run.run_id.is_empty()
        || manifest.payload.run.group_id.is_empty()
        || manifest.payload.run.data_parallel_world_size == 0
        || manifest.payload.run.data_parallel_rank >= manifest.payload.run.data_parallel_world_size
        || manifest.payload.run.tensor_parallel_world_size == 0
        || manifest.payload.run.tensor_parallel_rank
            >= manifest.payload.run.tensor_parallel_world_size
    {
        return Err(CliError::msg("launch manifest has an invalid run identity"));
    }
    let payload_bytes = serde_json::to_vec(&manifest.payload)
        .map_err(|error| CliError::msg(format!("serialize launch payload: {error}")))?;
    let expected = domain_sha256("ferrl.run-launch.payload.v1", &[&payload_bytes]);
    if manifest.payload_sha256 != expected {
        return Err(CliError::msg(format!(
            "launch payload hash mismatch: recorded {}, computed {expected}",
            manifest.payload_sha256
        )));
    }
    Ok(())
}

fn read_regular_bytes(path: &Path) -> Result<Vec<u8>, CliError> {
    let path_metadata = std::fs::symlink_metadata(path).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !path_metadata.file_type().is_file() {
        return Err(CliError::msg(format!(
            "provenance input {} is not a regular file",
            path.display()
        )));
    }
    let mut file = std::fs::File::open(path).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_metadata = file.metadata().map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if path_metadata.dev() != file_metadata.dev() || path_metadata.ino() != file_metadata.ino()
        {
            return Err(CliError::msg(format!(
                "provenance input {} changed while it was opened",
                path.display()
            )));
        }
    }
    let expected_len = file_metadata.len();
    let mut bytes = Vec::with_capacity(usize::try_from(expected_len).unwrap_or(0));
    file.read_to_end(&mut bytes)
        .map_err(|source| CliError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 != expected_len {
        return Err(CliError::msg(format!(
            "provenance input {} changed length while it was captured",
            path.display()
        )));
    }
    Ok(bytes)
}

fn validate_lower_sha256(label: &str, digest: &str) -> Result<(), CliError> {
    validate_lower_hex(label, digest, 32)
}

fn validate_lower_hex(label: &str, value: &str, bytes: usize) -> Result<(), CliError> {
    let expected_len = bytes.saturating_mul(2);
    if value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(CliError::msg(format!(
            "{label} must be {expected_len} lowercase hexadecimal characters"
        )))
    }
}

/// Dispatch `ferrl trimul-artifact`: extract `custom_kernel` from a model completion,
/// re-verify it with an audit seed, and write the contract artifact bundle.
fn trimul_artifact(args: &TrimulArtifactArgs) -> Result<(), CliError> {
    let trust_policy = load_system_launch_trust_policy()?;
    trimul_artifact_with_trust(args, &trust_policy)
}

fn trimul_artifact_with_trust(
    args: &TrimulArtifactArgs,
    trust_policy: &LaunchTrustPolicy,
) -> Result<(), CliError> {
    let _ = ferrl::init_tracing();
    if args.repeats < 3 {
        return Err(CliError::msg(
            "trimul-artifact requires --repeats >= 3 for the first-run contract",
        ));
    }
    let bound =
        load_bound_run_candidate_with_trust(&args.run_dir, &args.candidate_sha256, trust_policy)?;
    let cfg = &bound.config;
    if args.audit_secret_seed == cfg.trimul.secret_seed {
        return Err(CliError::msg(
            "audit secret seed must differ from trimul.secret_seed used during training",
        ));
    }
    let baseline = cfg.trimul.baseline.as_ref().ok_or_else(|| {
        CliError::msg("trimul-artifact requires trimul.baseline in the run config")
    })?;
    let verifier_assets = capture_attested_trimul_verifier_assets(cfg, &bound.launch)?;
    let baseline_median = median_checked(&args.baseline_measurements_ns, "baseline-ns")?;
    require_baseline_matches_config(baseline_median, baseline.ns)?;
    let gpu = detect_gpu_name().ok_or_else(|| {
        CliError::msg(
            "cannot read this node's GPU (nvidia-smi unavailable); run on the target GPU node",
        )
    })?;
    baseline_gpu_matches(&baseline.gpu, Some(&gpu)).map_err(CliError::Msg)?;

    let raw_completion = &bound.candidate.completion;
    let extract_mode = cfg.trimul_submission_extract_mode()?;
    let mut reward = cfg
        .build_trimul_reward_base_with_assets(verifier_assets.clone())?
        .with_submission_extract_mode(extract_mode);
    let submission = reward.extract_submission(raw_completion).ok_or_else(|| {
        CliError::msg("completion does not contain a closed non-empty fenced code block")
    })?;

    reward = reward
        .with_secret_seed(args.audit_secret_seed)
        .with_baseline_ns(baseline_median);
    let (test_cases, benchmark_cases) = ferrl::trimul::parse_task_yml(verifier_assets.task_yml())?;
    let runs = verify_submission_repeated(&reward, &submission, args.repeats)?;
    let accepted = accepted_artifact(&runs, baseline_median)
        && args.source_inspection == SourceInspectionResult::Clean;
    write_artifact_bundle(
        args,
        cfg,
        &ArtifactInputs {
            gpu,
            launch: &bound.launch,
            launch_bytes: &bound.launch_bytes,
            candidate: &bound.candidate,
            candidate_row_bytes: &bound.candidate_row_bytes,
            raw_completion,
            prompt_bytes: &bound.prompt_bytes,
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

fn capture_attested_trimul_verifier_assets(
    cfg: &RunConfig,
    launch: &LaunchManifest,
) -> Result<ferrl::trimul::TrimulVerifierAssets, CliError> {
    let expected = launch.payload.verifier.as_ref().ok_or_else(|| {
        CliError::msg("TriMul launch manifest is missing verifier asset identity")
    })?;
    let assets = cfg.capture_trimul_verifier_assets()?;
    if assets.identity() != expected {
        return Err(CliError::msg(
            "live TriMul verifier assets do not match the attested launch identity",
        ));
    }
    assets
        .verify_current()
        .map_err(|error| CliError::msg(error.to_string()))?;
    Ok(assets)
}

/// Values needed to write the artifact bundle.
struct ArtifactInputs<'a> {
    /// GPU product name.
    gpu: String,
    /// Verified immutable launch manifest.
    launch: &'a LaunchManifest,
    /// Exact `launch.json` bytes captured from the run directory.
    launch_bytes: &'a [u8],
    /// Verified exact candidate row.
    candidate: &'a CandidateRecord,
    /// Exact source JSONL row bytes.
    candidate_row_bytes: &'a [u8],
    /// Raw completion string exactly as stored in the candidate row.
    raw_completion: &'a str,
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
    write_bytes(&args.out.join(RunDir::LAUNCH_FILE), inputs.launch_bytes)?;
    write_bytes(&args.out.join("candidate.json"), inputs.candidate_row_bytes)?;
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
    let launch = &inputs.launch.payload;
    let verifier = launch
        .verifier
        .as_ref()
        .expect("verified TriMul launch must bind verifier assets");
    let candidate = inputs.candidate;
    ArtifactManifest {
        contract_version: 2,
        task: "trimul",
        ferrl_commit: launch.ferrl_commit.clone(),
        run_id: launch.run.run_id.clone(),
        launch_sha256: inputs.launch.payload_sha256.clone(),
        launch_file_sha256: sha256_hex(inputs.launch_bytes),
        launch_attestation_key_id: inputs
            .launch
            .attestation
            .as_ref()
            .expect("verified candidate launch must have an attestation")
            .key_id
            .clone(),
        launch_attestation_algorithm: LAUNCH_ATTESTATION_ALGORITHM.to_owned(),
        candidate: CandidateManifest {
            record_sha256: candidate
                .record_sha256
                .clone()
                .expect("verified candidate must have record_sha256"),
            record_signature: candidate
                .record_signature
                .clone()
                .expect("verified candidate must have record_signature"),
            ledger_row_sha256: sha256_hex(inputs.candidate_row_bytes),
            step: candidate.step,
            prompt_index: candidate.prompt_index,
            group_index: candidate.group_index,
            rank: candidate.rank,
            world_size: candidate.world_size,
            training_reward: candidate.reward,
            completion_sha256: sha256_hex(inputs.raw_completion.as_bytes()),
            source_sha256: sha256_hex(inputs.submission.as_bytes()),
            source_inspection: SourceInspectionManifest {
                result: args.source_inspection,
                notes: args.source_inspection_notes.clone(),
            },
        },
        model: ModelManifest {
            family: launch.model.family.clone(),
            checkpoint_policy_sha256: launch.model.checkpoint_policy_sha256.clone(),
            tokenizer_sha256: launch.model.tokenizer_sha256.clone(),
            lora_rank: cfg.policy.lora_rank,
            lora_alpha: cfg.policy.lora_alpha,
            base_dtype: cfg.policy.base_dtype.as_str(),
            base_quantization: cfg.policy.base_quantization.as_str(),
        },
        config: ArtifactConfigManifest {
            run_config_source_sha256: launch.config.source_sha256.clone(),
            run_config_resolved_sha256: launch.config.resolved_sha256.clone(),
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
            bundle_path: cfg.trimul.eval_dir.display().to_string(),
            bundle_sha256: verifier.eval_bundle_sha256.clone(),
            bundle_file_count: verifier.eval_file_count,
            sandbox_image_path: cfg.trimul.image.display().to_string(),
            sandbox_image_sha256: verifier.image_sha256.clone(),
            sandbox_image_len_bytes: verifier.image_len_bytes,
            task_yml_sha256: verifier.task_yml_sha256.clone(),
            task_yml_len_bytes: verifier.task_yml_len_bytes,
            test_cases: inputs.test_cases,
            benchmark_cases: inputs.benchmark_cases,
        },
        baseline: BaselineManifest {
            gpu: inputs.gpu.clone(),
            measurements_ns: args.baseline_measurements_ns.clone(),
            median_ns: inputs.baseline_median,
            command: args
                .baseline_command
                .clone()
                .unwrap_or_else(|| "ferrl trimul-baseline --config <launch-config>".to_owned()),
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
        "- Launch/config hashes: payload={}, file={}, source={}, resolved={}",
        manifest.launch_sha256,
        manifest.launch_file_sha256,
        manifest.config.run_config_source_sha256,
        manifest.config.run_config_resolved_sha256
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "- Launch attestation: {} ({})",
        manifest.launch_attestation_key_id, manifest.launch_attestation_algorithm
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
        "- Model: family={}, checkpoint_policy_sha256={}, tokenizer_sha256={}, lora_rank={}, lora_alpha={}, base_dtype={}, base_quantization={}",
        manifest.model.family,
        manifest.model.checkpoint_policy_sha256,
        manifest.model.tokenizer_sha256,
        manifest.model.lora_rank,
        manifest.model.lora_alpha,
        manifest.model.base_dtype,
        manifest.model.base_quantization
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
        !manifest.launch_sha256.is_empty()
            && !manifest.launch_file_sha256.is_empty()
            && !manifest.launch_attestation_key_id.is_empty()
            && manifest.launch_attestation_algorithm == LAUNCH_ATTESTATION_ALGORITHM
            && !manifest.config.run_config_source_sha256.is_empty()
            && !manifest.config.run_config_resolved_sha256.is_empty(),
        "launch attestation and config hashes recorded",
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
        !manifest.eval.bundle_path.trim().is_empty()
            && validate_lower_sha256("artifact eval bundle", &manifest.eval.bundle_sha256).is_ok()
            && manifest.eval.bundle_file_count > 0,
        "eval bundle identity recorded",
    );
    push_check(
        &mut out,
        !manifest.eval.sandbox_image_path.trim().is_empty()
            && validate_lower_sha256(
                "artifact sandbox image",
                &manifest.eval.sandbox_image_sha256,
            )
            .is_ok()
            && manifest.eval.sandbox_image_len_bytes > 0
            && validate_lower_sha256("artifact task.yml", &manifest.eval.task_yml_sha256).is_ok()
            && manifest.eval.task_yml_len_bytes > 0,
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

fn domain_sha256(domain: &str, fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain.as_bytes());
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    format!("{:x}", hasher.finalize())
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
    use candle_core::{Result as CandleResult, Tensor, Var};
    use ferrl::Comm as _;
    use ring::rand::SystemRandom;
    use ring::signature::{Ed25519KeyPair, KeyPair as _};
    use std::sync::{Arc, Mutex, OnceLock};

    const TEST_ATTESTATION_KEY_ID: &str = "test-root-1";
    const TEST_LAUNCH_ATTESTOR: TestLaunchAttestor = TestLaunchAttestor;
    const REJECTING_LAUNCH_ATTESTOR: RejectingLaunchAttestor = RejectingLaunchAttestor;
    const RANK_ONE_REJECTING_ATTESTOR: RankOneRejectingAttestor = RankOneRejectingAttestor;

    struct TestLaunchAttestor;
    struct RejectingLaunchAttestor;
    struct RankOneRejectingAttestor;
    struct MutatingLaunchAttestor {
        path: PathBuf,
        replacement: Vec<u8>,
        rank: Option<usize>,
    }

    fn test_attestation_pkcs8() -> &'static [u8] {
        static KEY: OnceLock<Vec<u8>> = OnceLock::new();
        KEY.get_or_init(|| {
            Ed25519KeyPair::generate_pkcs8(&SystemRandom::new())
                .unwrap()
                .as_ref()
                .to_vec()
        })
    }

    fn test_attestation_key_pair() -> Ed25519KeyPair {
        Ed25519KeyPair::from_pkcs8(test_attestation_pkcs8()).unwrap()
    }

    fn test_launch_trust_policy() -> LaunchTrustPolicy {
        let key_pair = test_attestation_key_pair();
        LaunchTrustPolicy {
            contract_version: LAUNCH_ATTESTATION_CONTRACT_VERSION,
            kind: LAUNCH_TRUST_POLICY_KIND.to_owned(),
            keys: vec![LaunchTrustKey {
                key_id: TEST_ATTESTATION_KEY_ID.to_owned(),
                algorithm: LAUNCH_ATTESTATION_ALGORITHM.to_owned(),
                public_key: lower_hex_bytes(key_pair.public_key().as_ref()),
            }],
        }
    }

    impl LaunchAttestor for TestLaunchAttestor {
        fn attest(&self, manifest: &LaunchManifest) -> Result<LaunchAttestation, CliError> {
            verify_launch_manifest_payload(manifest)?;
            let message = launch_attestation_message(&manifest.payload_sha256);
            Ok(LaunchAttestation {
                contract_version: LAUNCH_ATTESTATION_CONTRACT_VERSION,
                kind: LAUNCH_ATTESTATION_KIND.to_owned(),
                algorithm: LAUNCH_ATTESTATION_ALGORITHM.to_owned(),
                key_id: TEST_ATTESTATION_KEY_ID.to_owned(),
                launch_payload_sha256: manifest.payload_sha256.clone(),
                signature: lower_hex_bytes(
                    test_attestation_key_pair()
                        .sign(message.as_bytes())
                        .as_ref(),
                ),
            })
        }
    }

    impl LaunchAttestor for RejectingLaunchAttestor {
        fn attest(&self, _manifest: &LaunchManifest) -> Result<LaunchAttestation, CliError> {
            Err(CliError::msg("test launch attestor rejected request"))
        }
    }

    impl LaunchAttestor for RankOneRejectingAttestor {
        fn attest(&self, manifest: &LaunchManifest) -> Result<LaunchAttestation, CliError> {
            if manifest.payload.run.data_parallel_rank == 1 {
                Err(CliError::msg("test rank-one attestor rejection"))
            } else {
                TEST_LAUNCH_ATTESTOR.attest(manifest)
            }
        }
    }

    impl LaunchAttestor for MutatingLaunchAttestor {
        fn attest(&self, manifest: &LaunchManifest) -> Result<LaunchAttestation, CliError> {
            let attestation = TEST_LAUNCH_ATTESTOR.attest(manifest)?;
            if self
                .rank
                .is_none_or(|rank| rank == manifest.payload.run.data_parallel_rank)
            {
                let replacement_path = self.path.with_extension(format!(
                    "ferrl-replacement-{}",
                    manifest.payload.run.data_parallel_rank
                ));
                std::fs::write(&replacement_path, &self.replacement).map_err(|source| {
                    CliError::Io {
                        path: replacement_path.clone(),
                        source,
                    }
                })?;
                std::fs::rename(&replacement_path, &self.path).map_err(|source| CliError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            }
            Ok(attestation)
        }
    }

    fn attest_launch_for_test(manifest: LaunchManifest) -> LaunchManifest {
        manifest.attest(&TEST_LAUNCH_ATTESTOR).unwrap()
    }

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

    fn fixture_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    fn write_tiny_tokenizer(model_dir: &Path) {
        std::fs::copy(
            fixture_path("tiny_tokenizer.json"),
            model_dir.join("tokenizer.json"),
        )
        .unwrap();
    }

    fn write_generation_metadata_fixture(
        model_dir: &Path,
        eos_token_id: Option<serde_json::Value>,
        vocab_size: &serde_json::Value,
    ) {
        std::fs::create_dir_all(model_dir).unwrap();
        let mut config = serde_json::json!({ "vocab_size": vocab_size });
        if let Some(eos_token_id) = eos_token_id {
            config["eos_token_id"] = eos_token_id;
        }
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        write_tiny_tokenizer(model_dir);
    }

    fn countdown_run_config_with_eos(
        model_dir: &Path,
        eos_wire: Option<serde_json::Value>,
    ) -> RunConfig {
        let mut value: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        value["model_dir"] = serde_json::json!(model_dir);
        if let Some(eos_wire) = eos_wire {
            value["trainer"]["eos_token_id"] = eos_wire;
        }
        serde_json::from_value(value).unwrap()
    }

    fn move_tiny_tokenizer_special_id(tokenizer_path: &Path, id: u32) {
        let mut tokenizer_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(tokenizer_path).unwrap()).unwrap();
        tokenizer_json["model"]["vocab"]["<|special|>"] = serde_json::json!(id);
        tokenizer_json["added_tokens"][0]["id"] = serde_json::json!(id);
        std::fs::write(
            tokenizer_path,
            serde_json::to_vec_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
    }

    fn deterministic_tensor(dims: &[usize], offset: &mut usize) -> Tensor {
        let len = dims.iter().product();
        let values: Vec<f32> = (0..len)
            .map(|index| {
                let value = ((*offset + index) % 97) as f32;
                (value - 48.0) * 0.002
            })
            .collect();
        *offset += len;
        Tensor::from_vec(values, dims.to_vec(), &Device::Cpu).unwrap()
    }

    fn write_tp2_qwen3_fixture(model_dir: &Path) {
        std::fs::create_dir_all(model_dir).unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3",
            "vocab_size": 16,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "head_dim": 4,
            "attention_bias": false,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "sliding_window": null,
            "max_window_layers": 0,
            "tie_word_embeddings": true,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "use_sliding_window": false,
            "hidden_act": "silu"
        });
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();

        let mut offset = 0;
        let mut weights = std::collections::HashMap::new();
        let mut put = |name: &str, dims: &[usize]| {
            weights.insert(name.to_string(), deterministic_tensor(dims, &mut offset));
        };
        put("model.embed_tokens.weight", &[16, 16]);
        put("model.norm.weight", &[16]);
        put("model.layers.0.input_layernorm.weight", &[16]);
        put("model.layers.0.post_attention_layernorm.weight", &[16]);
        put("model.layers.0.self_attn.q_proj.weight", &[16, 16]);
        put("model.layers.0.self_attn.k_proj.weight", &[8, 16]);
        put("model.layers.0.self_attn.v_proj.weight", &[8, 16]);
        put("model.layers.0.self_attn.o_proj.weight", &[16, 16]);
        put("model.layers.0.self_attn.q_norm.weight", &[4]);
        put("model.layers.0.self_attn.k_norm.weight", &[4]);
        put("model.layers.0.mlp.gate_proj.weight", &[16, 16]);
        put("model.layers.0.mlp.up_proj.weight", &[16, 16]);
        put("model.layers.0.mlp.down_proj.weight", &[16, 16]);
        candle_core::safetensors::save(&weights, model_dir.join("model.safetensors")).unwrap();
        write_tiny_tokenizer(model_dir);
    }

    fn write_tp2_gemma4_fixture(model_dir: &Path) {
        std::fs::create_dir_all(model_dir).unwrap();
        let source = fixture_path("tiny_gemma4");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap()).unwrap();
        config["text_config"]["num_key_value_heads"] = serde_json::json!(2);
        config["text_config"]["num_global_key_value_heads"] = serde_json::json!(2);
        config["text_config"]["max_position_embeddings"] = serde_json::json!(128);
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();

        let source_weights =
            candle_core::safetensors::load(source.join("model.safetensors"), &Device::Cpu).unwrap();
        let weights: std::collections::HashMap<_, _> = source_weights
            .into_iter()
            .map(|(name, tensor)| {
                if name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight") {
                    let doubled = Tensor::cat(&[&tensor, &tensor], 0).unwrap();
                    (name, doubled)
                } else {
                    (name, tensor)
                }
            })
            .collect();
        candle_core::safetensors::save(&weights, model_dir.join("model.safetensors")).unwrap();
        write_tiny_tokenizer(model_dir);
    }

    fn copy_fixture_dir(source_name: &str, model_dir: &Path) {
        std::fs::create_dir_all(model_dir).unwrap();
        for entry in std::fs::read_dir(fixture_path(source_name)).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), model_dir.join(entry.file_name())).unwrap();
            }
        }
        write_tiny_tokenizer(model_dir);
    }

    fn countdown_train_config(extra_fields: &str) -> String {
        let extra = if extra_fields.trim().is_empty() {
            String::new()
        } else {
            format!("{extra_fields},")
        };
        format!(
            r#"{{
                "task": "countdown",
                "model_dir": "/models/qwen3-0.6b",
                {extra}
                "trainer": {{ "steps": 1, "group_size": 2, "max_new_tokens": 8,
                    "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                    "lr": 1e-5, "weight_decay": 0.0,
                    "loss_type": "grpo", "scale_rewards": "group" }}
            }}"#
        )
    }

    fn write_countdown_train_config(tag: &str, extra_fields: &str) -> (TestDir, PathBuf) {
        let tmp = TestDir::new(tag);
        let path = tmp.path().join("run.json");
        std::fs::write(&path, countdown_train_config(extra_fields)).unwrap();
        (tmp, path)
    }

    fn test_policy_identity() -> ferrl::PolicyLoadIdentity {
        ferrl::PolicyLoadIdentity {
            policy_sha256: "00".repeat(32),
            tokenizer_sha256: "11".repeat(32),
            model_family: "qwen3",
        }
    }

    fn test_build_source_identity() -> BuildSourceIdentity {
        validated_build_source_identity(&"01".repeat(20), false).unwrap()
    }

    fn test_trimul_verifier_identity() -> ferrl::trimul::TrimulVerifierIdentity {
        ferrl::trimul::TrimulVerifierIdentity {
            image_sha256: "55".repeat(32),
            image_len_bytes: 17,
            eval_bundle_sha256: "66".repeat(32),
            eval_file_count: 5,
            task_yml_sha256: "77".repeat(32),
            task_yml_len_bytes: 19,
        }
    }

    fn launch_context_for_test(
        cfg: &RunConfig,
        run_id: String,
        data_parallel_rank: usize,
        data_parallel_world_size: usize,
    ) -> LaunchContext {
        let resolved = canonicalize_json(cfg.canonical_wire_value().unwrap());
        let resolved_bytes = serde_json::to_vec(&resolved).unwrap();
        LaunchContext {
            ferrl_commit: "01".repeat(20),
            run: LaunchRunIdentity {
                group_id: "test-group".to_owned(),
                run_id,
                data_parallel_rank,
                data_parallel_world_size,
                tensor_parallel_rank: 0,
                tensor_parallel_world_size: 1,
            },
            config: LaunchConfigSnapshot {
                source_sha256: "22".repeat(32),
                resolved_sha256: sha256_hex(&resolved_bytes),
                resolved,
            },
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

    fn write_trimul_verifier_fixture(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let image = root.join("image.sif");
        let eval_dir = root.join("eval");
        let scratch = root.join("scratch");
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(&image, b"exact-test-sif-image").unwrap();
        for name in ["eval.py", "reference.py", "task.py", "utils.py"] {
            std::fs::write(eval_dir.join(name), format!("# exact {name}\n")).unwrap();
        }
        std::fs::write(
            eval_dir.join("task.yml"),
            b"tests:\n  - {\"seqlen\": 8, \"bs\": 1, \"dim\": 4, \"hiddendim\": 4, \"seed\": 1, \"nomask\": true, \"distribution\": \"normal\"}\nbenchmarks:\n  - {\"seqlen\": 16, \"bs\": 1, \"dim\": 4, \"hiddendim\": 4, \"seed\": 2, \"nomask\": false, \"distribution\": \"cauchy\"}\n",
        )
        .unwrap();
        (image, eval_dir, scratch)
    }

    fn trimul_config_with_verifier_fixture(
        root: &Path,
        model_dir: &Path,
        out_dir: &Path,
    ) -> RunConfig {
        let (image, eval_dir, scratch) = write_trimul_verifier_fixture(root);
        let prompt = root.join("prompt.txt");
        std::fs::write(&prompt, b"exact prompt").unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&trimul_score_test_config(4242)).unwrap();
        value["model_dir"] = serde_json::json!(model_dir);
        value["out_dir"] = serde_json::json!(out_dir);
        value["trimul"]["prompt_path"] = serde_json::json!(prompt);
        value["trimul"]["image"] = serde_json::json!(image);
        value["trimul"]["eval_dir"] = serde_json::json!(eval_dir);
        value["trimul"]["scratch_root"] = serde_json::json!(scratch);
        value["trainer"]["max_new_tokens"] = serde_json::json!(2);
        value["trainer"]["candidate_log_top_k"] = serde_json::json!(1);
        serde_json::from_value(value).unwrap()
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
            run_dir: dir.join("test-run"),
            candidate_sha256: "11".repeat(32),
            out: dir.join("artifact"),
            audit_secret_seed: 999,
            baseline_measurements_ns: vec![1.0, 1.0, 1.0],
            baseline_command: None,
            repeats: 3,
            run_health: "test".to_string(),
            source_inspection: SourceInspectionResult::Clean,
            source_inspection_notes: "clean".to_string(),
        }
    }

    fn launch_manifest_for_test(
        cfg: &RunConfig,
        run_id: &str,
        prompt: &[u8],
    ) -> (LaunchManifest, CandidateSigner) {
        let context = launch_context_for_test(cfg, run_id.to_owned(), 0, 1);
        let signer = CandidateSigner::generate().unwrap();
        let manifest = attest_launch_for_test(
            LaunchManifest::new(LaunchPayload {
                task: cfg.task.clone(),
                ferrl_commit: context.ferrl_commit,
                run: context.run,
                config: context.config,
                model: LaunchModelIdentity {
                    family: "gemma4".to_owned(),
                    checkpoint_policy_sha256: "33".repeat(32),
                    tokenizer_sha256: "44".repeat(32),
                    resolved_eos_token_id: None,
                },
                prompt: Some(LaunchPromptIdentity {
                    file: RunDir::PROMPT_FILE.to_owned(),
                    sha256: sha256_hex(prompt),
                    len_bytes: prompt.len(),
                }),
                verifier: (cfg.task == "trimul").then(test_trimul_verifier_identity),
                candidate_ledger: LaunchCandidateLedger {
                    file: RunDir::CANDIDATES_FILE.to_owned(),
                    format_version: 1,
                    row_digest_domain: CANDIDATE_RECORD_DOMAIN.to_owned(),
                    row_signature_algorithm: "ed25519".to_owned(),
                    signing_public_key: signer.public_key_hex(),
                },
            })
            .unwrap(),
        );
        (manifest, signer)
    }

    fn candidate_for_test(
        launch: &LaunchManifest,
        signer: &CandidateSigner,
        completion: &str,
    ) -> CandidateRecord {
        let mut candidate = CandidateRecord::new(0, 0, 1, 12, 1, 1.5, 3, completion.to_owned());
        candidate.reward_metadata = Some(serde_json::json!({ "correct": true }));
        signer
            .sign_candidate(&candidate, &launch.payload_sha256)
            .unwrap()
    }

    fn write_bound_candidate_run(
        tag: &str,
        launch_rank: usize,
        launch_world: usize,
        candidate_rank: usize,
        candidate_world: usize,
    ) -> (TestDir, PathBuf, String) {
        let tmp = TestDir::new(tag);
        let run_id = if launch_world == 1 {
            "trimul-1".to_owned()
        } else {
            format!("trimul-1-rank{launch_rank}")
        };
        let mut config_value: serde_json::Value =
            serde_json::from_str(&trimul_score_test_config(4242)).unwrap();
        if launch_world > 1 {
            config_value["distributed"] = serde_json::json!({ "enabled": true });
        }
        let cfg: RunConfig = serde_json::from_value(config_value).unwrap();
        let (mut launch, signer) = launch_manifest_for_test(&cfg, &run_id, b"prompt");
        launch.payload.run.group_id = "trimul-1".to_owned();
        launch.payload.run.data_parallel_rank = launch_rank;
        launch.payload.run.data_parallel_world_size = launch_world;
        launch = attest_launch_for_test(LaunchManifest::new(launch.payload).unwrap());
        let mut candidate = candidate_for_test(&launch, &signer, "```python\npass\n```\n");
        candidate.rank = candidate_rank;
        candidate.world_size = candidate_world;
        candidate = signer
            .sign_candidate(&candidate, &launch.payload_sha256)
            .unwrap();
        let candidate_sha256 = candidate.record_sha256.clone().unwrap();
        let run = RunDir::create(tmp.path(), &run_id).unwrap();
        run.write_immutable_launch(&launch.to_pretty_bytes().unwrap(), Some(b"prompt"))
            .unwrap();
        let mut row = serde_json::to_vec(&candidate).unwrap();
        row.push(b'\n');
        std::fs::write(run.candidates_path(), row).unwrap();
        (tmp, run.root().to_path_buf(), candidate_sha256)
    }

    fn load_bound_candidate_for_test(
        run_dir: &Path,
        candidate_sha256: &str,
    ) -> Result<BoundRunCandidate, CliError> {
        load_bound_run_candidate_with_trust(run_dir, candidate_sha256, &test_launch_trust_policy())
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
        assert_eq!(
            cfg.policy.base_quantization.as_base_quantization(),
            BaseQuantization::None
        );
        assert_eq!(
            cfg.tensor_parallel.plan().unwrap(),
            TensorParallelPlan::single()
        );
        assert_eq!(cfg.data.train_n, 64);
        // The loader temperature mirrors the trainer's (cannot drift).
        assert!((cfg.loader_opts().temperature - cfg.trainer.temperature).abs() < f64::EPSILON);
        assert_eq!(cfg.loader_opts().base_quantization, BaseQuantization::None);
        assert_eq!(
            cfg.loader_opts().tensor_parallel,
            TensorParallelPlan::single()
        );
    }

    #[test]
    fn tensor_parallel_config_fails_closed_on_rank_world_shape() {
        let tmp = TestDir::new("tensor-parallel-rank-world");
        let config = r#"{
            "task": "countdown",
            "model_dir": "/models/qwen3-0.6b",
            "tensor_parallel": { "enabled": true, "rank": 2, "world_size": 2 },
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                         "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                         "lr": 1e-5, "weight_decay": 0.0,
                         "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let path = tmp.path().join("run.json");
        std::fs::write(&path, config).unwrap();

        let err = RunConfig::load(&path).unwrap_err().to_string();

        assert!(err.contains("tensor_parallel.rank 2 outside world_size 2"));
    }

    #[test]
    fn tensor_parallel_disabled_rejects_stale_rank_world_fields() {
        let tmp = TestDir::new("tensor-parallel-disabled-stale-fields");
        let config = r#"{
            "task": "countdown",
            "model_dir": "/models/qwen3-0.6b",
            "tensor_parallel": { "rank": 1, "world_size": 2 },
            "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                         "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                         "lr": 1e-5, "weight_decay": 0.0,
                         "loss_type": "grpo", "scale_rewards": "group" }
        }"#;
        let path = tmp.path().join("run.json");
        std::fs::write(&path, config).unwrap();

        let err = RunConfig::load(&path).unwrap_err().to_string();

        assert!(err.contains("tensor_parallel disabled requires rank = 0"));
    }

    #[test]
    fn tensor_parallel_multi_rank_config_passes_plan_to_loader_and_execution() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-public-execution-plan",
            r#""device": "cuda",
               "policy": { "activation_checkpointing": true },
               "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }"#,
        );

        let cfg = RunConfig::load(&path).unwrap();

        assert_eq!(
            cfg.tensor_parallel_plan(),
            TensorParallelPlan::new(0, 2).unwrap()
        );
        assert_eq!(
            cfg.loader_opts().tensor_parallel,
            TensorParallelPlan::new(0, 2).unwrap()
        );
        let run = synchronized_run_identity(&cfg, None).unwrap();
        assert!(run.run_id.starts_with("countdown-"), "{}", run.run_id);
        assert!(run.run_id.ends_with("-rank0"), "{}", run.run_id);
    }

    #[test]
    fn tensor_parallel_multi_rank_rejects_q8_0_before_dispatch() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-q8-rejected",
            r#""device": "cuda",
               "policy": { "base_quantization": "q8_0" },
               "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }"#,
        );

        let err = RunConfig::load(&path).unwrap_err().to_string();

        assert!(err.contains("does not support policy.base_quantization = \"q8_0\""));
    }

    #[test]
    fn tensor_parallel_world_one_rejects_q8_0_before_dispatch() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-world-one-q8-rejected",
            r#""device": "cuda",
               "policy": { "base_quantization": "q8_0" },
               "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 1 }"#,
        );

        let err = RunConfig::load(&path).unwrap_err().to_string();

        assert!(err.contains("does not support policy.base_quantization = \"q8_0\""));
        assert!(err.contains("disable tensor_parallel to use world-one Q8_0"));
    }

    fn validate_local_tp_plans(plans: [TensorParallelPlan; 2]) -> Vec<Result<(), String>> {
        std::thread::scope(|scope| {
            let handles: Vec<_> = ferrl::LocalComm::world(2)
                .into_iter()
                .zip(plans)
                .map(|(comm, plan)| {
                    scope.spawn(move || {
                        validate_tensor_parallel_runtime(plan, &comm).map_err(|err| err.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
    }

    #[test]
    fn tensor_parallel_comm_plan_accepts_live_world_two() {
        let results = validate_local_tp_plans([
            TensorParallelPlan::new(0, 2).unwrap(),
            TensorParallelPlan::new(1, 2).unwrap(),
        ]);
        assert!(results.into_iter().all(|result| result.is_ok()));
    }

    #[test]
    fn tensor_parallel_comm_plan_mismatch_aborts_world_in_lockstep() {
        let rank_mismatch = validate_local_tp_plans([
            TensorParallelPlan::new(0, 2).unwrap(),
            TensorParallelPlan::new(0, 2).unwrap(),
        ]);
        assert!(rank_mismatch[0]
            .as_ref()
            .unwrap_err()
            .contains("failed on a peer distributed rank"));
        assert!(rank_mismatch[1]
            .as_ref()
            .unwrap_err()
            .contains("plan rank/world (0, 2) does not match communicator (1, 2)"));

        let world_mismatch = validate_local_tp_plans([
            TensorParallelPlan::new(0, 3).unwrap(),
            TensorParallelPlan::new(1, 3).unwrap(),
        ]);
        for result in world_mismatch {
            assert!(result.unwrap_err().contains("does not match communicator"));
        }
    }

    #[test]
    fn distributed_rank_setup_failure_aborts_world_in_lockstep() {
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = ferrl::LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    scope.spawn(move || {
                        let local = if comm.rank() == 1 {
                            Err(CliError::msg("rank-local model setup failed"))
                        } else {
                            Ok(())
                        };
                        coordinate_distributed_result(Some(&comm), "model setup", local)
                            .map_err(|err| err.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results[0]
            .as_ref()
            .unwrap_err()
            .contains("model setup failed on a peer distributed rank"));
        assert_eq!(
            results[1].as_ref().unwrap_err(),
            "rank-local model setup failed"
        );
    }

    #[test]
    fn data_parallel_retains_launch_coordination_through_trainer_setup() {
        let tmp = TestDir::new("dp-launch-setup-coordination");
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> =
                ferrl::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(1))
                    .into_iter()
                    .map(|comm| {
                        let root = tmp.path().to_path_buf();
                        scope.spawn(move || {
                            let rank = comm.rank();
                            let launch_comm = SharedComm::from_box(Box::new(comm));
                            let trainer_comm = launch_comm.clone();
                            let local = if rank == 1 {
                                Err(CliError::msg("rank-local DP trainer setup failed"))
                            } else {
                                (|| {
                                    let run =
                                        RunDir::create(&root, format!("dp-setup-rank-{rank}"))?;
                                    open_trainer(
                                        TrainerConfig::default(),
                                        &run,
                                        Some(trainer_comm),
                                        None,
                                        &"11".repeat(32),
                                        CandidateSigner::generate()?,
                                    )
                                })()
                            };
                            coordinate_distributed_result(
                                Some(&launch_comm),
                                "data-parallel model and trainer setup",
                                local,
                            )
                            .map(|_| ())
                            .map_err(|err| err.to_string())
                        })
                    })
                    .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results[0]
            .as_ref()
            .unwrap_err()
            .contains("failed on a peer distributed rank"));
        assert_eq!(
            results[1].as_ref().unwrap_err(),
            "rank-local DP trainer setup failed"
        );
    }

    fn candidate_health_run_config() -> RunConfig {
        let mut cfg: RunConfig = serde_json::from_str(&countdown_train_config("")).unwrap();
        cfg.trainer.candidate_log_top_k = 2;
        cfg.run_health = correctness_collapse_policy();
        cfg
    }

    fn healthy_candidate_history() -> Vec<ferrl::Metrics> {
        vec![
            run_health_test_metric(0, 2.0, 1.0),
            run_health_test_metric(1, 2.0, 1.0),
        ]
    }

    fn write_healthy_candidate_ledger(run: &RunDir) {
        write_candidate_jsonl(
            &run.candidates_path(),
            [
                (0, 0, true, "source-0-0".to_string()),
                (0, 1, true, "source-0-1".to_string()),
                (1, 0, true, "source-1-0".to_string()),
                (1, 1, true, "source-1-1".to_string()),
            ],
        );
    }

    fn run_coordinated_candidate_health(with_primary_ledger: bool) -> Vec<(usize, usize, String)> {
        let tmp = TestDir::new("tp-primary-candidate-health");
        std::thread::scope(|scope| {
            let handles: Vec<_> = ferrl::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = tmp.path().to_path_buf();
                    scope.spawn(move || {
                        let cfg = candidate_health_run_config();
                        let history = healthy_candidate_history();
                        let summary = summarize(&history).unwrap();
                        let run =
                            RunDir::create(&root, format!("candidate-health-rank-{rank}")).unwrap();
                        if rank == 0 && with_primary_ledger {
                            write_healthy_candidate_ledger(&run);
                        }
                        let mut calls = 0;
                        let result =
                            run_on_tensor_parallel_primary(Some(&comm), "post-run health", || {
                                calls += 1;
                                apply_train_run_health_policy(&cfg, &history, &summary, &run)
                            })
                            .map_or_else(|err| err.to_string(), |()| String::new());
                        (rank, calls, result)
                    })
                })
                .collect();
            let mut results: Vec<_> = handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();
            results.sort_by_key(|(rank, _, _)| *rank);
            results
        })
    }

    #[test]
    fn tensor_parallel_postprocess_uses_primary_candidate_ledger_only() {
        let tmp = TestDir::new("tp-empty-peer-health");
        let cfg = candidate_health_run_config();
        let history = healthy_candidate_history();
        let summary = summarize(&history).unwrap();
        let empty_peer = RunDir::create(tmp.path(), "empty-peer").unwrap();
        assert!(apply_train_run_health_policy(&cfg, &history, &summary, &empty_peer).is_err());

        let results = run_coordinated_candidate_health(true);
        assert_eq!(results[0], (0, 1, String::new()));
        assert_eq!(results[1], (1, 0, String::new()));
    }

    #[test]
    fn tensor_parallel_postprocess_primary_health_failure_reaches_all_ranks() {
        let results = run_coordinated_candidate_health(false);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, 1);
        assert_eq!(results[0].2, "run_health policy failed");
        assert_eq!(results[1].0, 1);
        assert_eq!(results[1].1, 0);
        assert!(results[1]
            .2
            .contains("post-run health failed on a peer distributed rank"));
    }

    #[test]
    fn tensor_parallel_multi_rank_requires_cuda_device() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-requires-cuda",
            r#""tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }"#,
        );

        let err = RunConfig::load(&path).unwrap_err().to_string();

        assert!(err.contains("requires device = \"cuda\""));
    }

    #[test]
    fn tensor_parallel_multi_rank_rejects_dp_combo() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-rejects-sharded-dp",
            r#""device": "cuda",
               "distributed": { "enabled": true },
               "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }"#,
        );

        let err = RunConfig::load(&path).unwrap_err().to_string();

        assert!(err.contains("simultaneous distributed data parallelism"));
    }

    #[test]
    fn tensor_parallel_multi_rank_defers_activation_checkpointing_to_model_capability() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-activation-checkpointing-capability",
            r#""device": "cuda",
               "policy": { "activation_checkpointing": true },
               "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }"#,
        );

        let cfg = RunConfig::load(&path).unwrap();
        assert!(cfg.policy.activation_checkpointing);
        assert!(cfg.tensor_parallel_plan().is_sharded());
    }

    #[test]
    fn tensor_parallel_multi_rank_requires_activation_checkpointing() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-requires-activation-checkpointing",
            r#""device": "cuda",
               "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }"#,
        );

        let err = RunConfig::load(&path).unwrap_err().to_string();
        assert!(err.contains("requires policy.activation_checkpointing = true"));
    }

    #[test]
    fn tensor_parallel_multi_rank_rejects_held_out_eval() {
        let (_tmp, path) = write_countdown_train_config(
            "tensor-parallel-rejects-held-out-eval",
            r#""device": "cuda",
               "data": { "eval_n": 1 },
               "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }"#,
        );

        let err = RunConfig::load(&path).unwrap_err().to_string();

        assert!(err.contains("held-out eval"));
    }

    #[derive(Clone, Default)]
    struct CliTpCalls {
        generate: usize,
        live_logp: usize,
        detached_logp: usize,
        comms: Vec<(usize, usize)>,
    }

    struct CliTpPolicy {
        logp: Var,
        enabled: bool,
        calls: Arc<Mutex<CliTpCalls>>,
    }

    impl Policy for CliTpPolicy {
        fn generate(
            &mut self,
            _prompt: &[u32],
            _cfg: &GenConfig,
        ) -> CandleResult<ferrl::policy::Rollout> {
            panic!("CLI tensor_parallel helper must not call Policy::generate")
        }

        fn token_logprobs(&self, _rollout: &ferrl::policy::Rollout) -> CandleResult<Tensor> {
            panic!("CLI tensor_parallel helper must not call Policy::token_logprobs")
        }

        fn token_logprobs_detached(
            &self,
            _rollout: &ferrl::policy::Rollout,
        ) -> CandleResult<Tensor> {
            panic!("CLI tensor_parallel helper must not call Policy::token_logprobs_detached")
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        fn adapter_enabled(&self) -> bool {
            self.enabled
        }

        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    impl TensorParallelPolicy for CliTpPolicy {
        fn supports_sharded_tensor_parallel_backward(&self) -> bool {
            true
        }

        fn generate_at_tensor_parallel_instrumented(
            &mut self,
            prompt: &[u32],
            _cfg: &GenConfig,
            _global_row_base: u64,
            comm: &dyn ferrl::Comm,
            _telemetry: Option<&mut dyn ferrl::ModelTelemetryRecorder>,
        ) -> CandleResult<ferrl::policy::Rollout> {
            let mut calls = self.calls.lock().unwrap();
            calls.generate += 1;
            calls.comms.push((comm.rank(), comm.world_size()));
            Ok(ferrl::policy::Rollout {
                token_ids: vec![vec![prompt[0], 1], vec![prompt[0], 2]],
                prompt_len: prompt.len(),
                completion_lens: vec![1, 1],
                rollout_logprobs: Some(vec![vec![-0.5], vec![-0.5]]),
            })
        }

        fn token_logprobs_tensor_parallel(
            &self,
            _rollout: &ferrl::policy::Rollout,
            comm: &dyn ferrl::Comm,
        ) -> CandleResult<Tensor> {
            let mut calls = self.calls.lock().unwrap();
            calls.live_logp += 1;
            calls.comms.push((comm.rank(), comm.world_size()));
            Ok(self.logp.as_tensor().clone())
        }

        fn token_logprobs_tensor_parallel_detached(
            &self,
            _rollout: &ferrl::policy::Rollout,
            comm: &dyn ferrl::Comm,
        ) -> CandleResult<Tensor> {
            let mut calls = self.calls.lock().unwrap();
            calls.detached_logp += 1;
            calls.comms.push((comm.rank(), comm.world_size()));
            Ok(self.logp.as_tensor().detach())
        }

        fn backward_tensor_parallel(
            &self,
            loss: &Tensor,
            _comm: &dyn ferrl::Comm,
        ) -> CandleResult<candle_core::backprop::GradStore> {
            loss.backward()
        }
    }

    struct CliTpCodec;

    impl TokenizerLike for CliTpCodec {
        fn encode(&self, _text: &str) -> Vec<u32> {
            vec![42]
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    struct CliTpReward;

    impl RewardFn for CliTpReward {
        type Target = ();

        fn reward(
            &self,
            _sample: &Sample<()>,
            completion: &str,
        ) -> Result<f32, ferrl::RewardError> {
            Ok(match completion {
                "1" => 0.0,
                "2" => 2.0,
                other => panic!("unexpected completion {other}"),
            })
        }
    }

    fn cli_tp_policy() -> (CliTpPolicy, Arc<Mutex<CliTpCalls>>) {
        let calls = Arc::new(Mutex::new(CliTpCalls::default()));
        let logp =
            Var::from_tensor(&Tensor::from_vec(vec![-0.4f32, -0.6], (2, 1), &Device::Cpu).unwrap())
                .unwrap();
        (
            CliTpPolicy {
                logp,
                enabled: true,
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }

    #[test]
    fn train_helper_routes_tensor_parallel_comm_through_public_trainer_hook() {
        let tmp = TestDir::new("tensor-parallel-train-helper-dispatch");
        let run = RunDir::create(tmp.path(), "tp-train-helper-dispatch").unwrap();
        let cfg = TrainerConfig {
            steps: 1,
            group_size: 2,
            max_new_tokens: 1,
            lr: 0.0,
            beta: 0.1,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let (mut policy, calls) = cli_tp_policy();
        let comm = ferrl::LocalComm::world(1).pop().unwrap();

        train_with_optional_tensor_parallel(
            &mut trainer,
            &mut policy,
            &CliTpReward,
            &CliTpCodec,
            &[Sample::new("prompt", ())],
            Some(&comm),
        )
        .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls.generate, 1);
        assert!(calls.live_logp >= 1, "live TP scoring was not used");
        assert!(
            calls.detached_logp >= 2,
            "old/reference TP detached scoring was not used"
        );
        assert!(
            calls
                .comms
                .iter()
                .all(|&(rank, world)| (rank, world) == (0, 1)),
            "trainer did not pass the explicit TP communicator: {:?}",
            calls.comms
        );
    }

    #[test]
    fn train_helper_routes_live_world_two_through_public_tp_hooks() {
        let tmp = TestDir::new("tensor-parallel-train-helper-world-two");
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = ferrl::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = tmp.path().to_path_buf();
                    scope.spawn(move || {
                        let run = RunDir::create(&root, format!("tp-helper-rank-{rank}")).unwrap();
                        let cfg = TrainerConfig {
                            steps: 1,
                            group_size: 2,
                            max_new_tokens: 1,
                            lr: 0.0,
                            beta: 0.1,
                            ..TrainerConfig::default()
                        };
                        let mut trainer = Trainer::new(cfg, &run).unwrap();
                        let (mut policy, calls) = cli_tp_policy();
                        train_with_optional_tensor_parallel(
                            &mut trainer,
                            &mut policy,
                            &CliTpReward,
                            &CliTpCodec,
                            &[Sample::new("prompt", ())],
                            Some(&comm),
                        )
                        .unwrap();
                        let calls = calls.lock().unwrap().clone();
                        (rank, calls)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (rank, calls) in results {
            assert_eq!(calls.generate, 1, "rank {rank} skipped TP rollout");
            assert!(calls.live_logp > 0, "rank {rank} skipped TP scoring");
            assert!(calls
                .comms
                .iter()
                .all(|&(seen_rank, world)| (seen_rank, world) == (rank, 2)));
        }
    }

    fn write_tp_auto_policy_config(
        root: &Path,
        model_dir: &Path,
        out_dir: &Path,
        rank: usize,
        world_size: usize,
    ) -> PathBuf {
        let config = serde_json::json!({
            "task": "countdown",
            "model_dir": model_dir,
            "device": "cuda",
            "out_dir": out_dir,
            "policy": {
                "lora_rank": 2,
                "lora_alpha": 4.0,
                "seed": 7,
                "activation_checkpointing": true
            },
            "data": { "train_n": 1, "eval_n": 0, "seed": 11 },
            "tensor_parallel": {
                "enabled": true,
                "rank": rank,
                "world_size": world_size
            },
            "trainer": {
                "steps": 1,
                "group_size": 2,
                "max_new_tokens": 1,
                "temperature": 1.0,
                "mu": 1,
                "beta": 0.0,
                "clip_eps": 0.2,
                "lr": 0.0,
                "weight_decay": 0.0,
                "loss_type": "grpo",
                "scale_rewards": "group",
                "eos_token_id": "none"
            }
        });
        let path = root.join(format!("rank-{rank}.json"));
        std::fs::write(&path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
        path
    }

    fn prepare_test_launch_device(
        _cfg: &RunConfig,
        runtime: Option<&LaunchRuntime>,
    ) -> Result<Device, CliError> {
        Ok(runtime
            .ok_or_else(|| CliError::msg("test launch runtime is missing"))?
            .device
            .clone())
    }

    fn run_train_configs_world_two(configs: [PathBuf; 2]) -> Vec<Result<(), String>> {
        std::thread::scope(|scope| {
            let handles: Vec<_> = ferrl::LocalComm::world(2)
                .into_iter()
                .zip(configs)
                .map(|(comm, config)| {
                    scope.spawn(move || {
                        let args = TrainArgs { config };
                        let runtime = LaunchRuntime {
                            device: Device::Cpu,
                            comm: Box::new(comm),
                        };
                        train_with_launch_runtime(
                            &args,
                            Some(runtime),
                            test_build_source_identity(),
                            prepare_test_launch_device,
                        )
                        .map_err(|err| err.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
    }

    #[test]
    fn tensor_parallel_config_consensus_allows_only_rank_difference() {
        let tmp = TestDir::new("tp-config-consensus-rank-only");
        let configs = [0, 1]
            .map(|rank| write_tp_auto_policy_config(tmp.path(), tmp.path(), tmp.path(), rank, 2));
        let digests =
            configs.map(|path| RunConfig::load_for_launch(&path).unwrap().consensus_digest);
        assert_eq!(digests[0], digests[1]);
    }

    #[test]
    fn distributed_run_identity_uses_one_rank_zero_timestamp() {
        let mut value: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        value["distributed"] = serde_json::json!({ "enabled": true });
        let identities = std::thread::scope(|scope| {
            ferrl::LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    let cfg: RunConfig = serde_json::from_value(value.clone()).unwrap();
                    scope.spawn(move || synchronized_run_identity(&cfg, Some(&comm)).unwrap())
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert_eq!(identities[0].group_id, identities[1].group_id);
        assert_eq!(
            identities[0].run_id,
            format!("{}-rank0", identities[0].group_id)
        );
        assert_eq!(
            identities[1].run_id,
            format!("{}-rank1", identities[1].group_id)
        );
        assert_eq!(identities[0].data_parallel_world_size, 2);
        assert_eq!(identities[1].data_parallel_rank, 1);
    }

    #[test]
    fn distributed_launch_rejects_tokenizer_identity_drift() {
        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    scope.spawn(move || {
                        validate_launch_value_consensus(
                            "model/checkpoint/tokenizer/prompt provenance",
                            if rank == 0 {
                                b"tokenizer-a"
                            } else {
                                b"tokenizer-b"
                            },
                            Some(&comm),
                        )
                        .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert!(results.iter().all(|result| result
            .as_ref()
            .unwrap_err()
            .contains("launch ranks disagree")));
    }

    #[test]
    fn tensor_parallel_config_consensus_rejects_valid_trainer_mismatch() {
        let tmp = TestDir::new("tp-config-consensus-mismatch");
        let configs = [0, 1]
            .map(|rank| write_tp_auto_policy_config(tmp.path(), tmp.path(), tmp.path(), rank, 2));
        let mut rank_one: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&configs[1]).unwrap()).unwrap();
        rank_one["trainer"]["steps"] = serde_json::Value::from(2);
        std::fs::write(&configs[1], serde_json::to_vec_pretty(&rank_one).unwrap()).unwrap();

        let results = run_train_configs_world_two(configs);
        for result in results {
            assert!(result
                .unwrap_err()
                .contains("launch ranks disagree on run config outside tensor_parallel.rank"));
        }
    }

    fn run_auto_policy_world_two(model_dir: &Path) -> Vec<Result<(), String>> {
        let output = TestDir::new("tp-auto-policy-output");
        let configs = [0, 1].map(|rank| {
            write_tp_auto_policy_config(output.path(), model_dir, output.path(), rank, 2)
        });
        run_train_configs_world_two(configs)
    }

    #[test]
    fn tensor_parallel_enabled_world_one_rejects_a_live_world_two() {
        let tmp = TestDir::new("tp-world-one-live-world-two");
        let config =
            write_tp_auto_policy_config(tmp.path(), Path::new("/unused"), tmp.path(), 0, 1);

        let results = run_train_configs_world_two([config.clone(), config]);

        for result in results {
            assert!(result
                .unwrap_err()
                .contains("does not match the live communicator"));
        }
    }

    #[test]
    fn tensor_parallel_config_load_failure_aborts_live_world_in_lockstep() {
        let tmp = TestDir::new("tp-config-load-failure");
        let valid = write_tp_auto_policy_config(tmp.path(), Path::new("/unused"), tmp.path(), 0, 2);
        let malformed = tmp.path().join("rank-1-malformed.json");
        std::fs::write(&malformed, b"{").unwrap();

        let results = run_train_configs_world_two([valid, malformed]);

        assert!(results[0]
            .as_ref()
            .unwrap_err()
            .contains("run config load failed on a peer"));
        assert!(results[1]
            .as_ref()
            .unwrap_err()
            .contains("parse run config"));
    }

    #[test]
    fn public_tp_auto_policy_trains_gemma4_and_rejects_forward_only_qwen3() {
        let fixtures = TestDir::new("tp-auto-policy-fixtures");
        let qwen = fixtures.path().join("qwen3");
        write_tp2_qwen3_fixture(&qwen);
        let qwen_results = run_auto_policy_world_two(&qwen);
        for result in qwen_results {
            assert!(result
                .unwrap_err()
                .contains("does not provide cross-rank backward semantics"));
        }

        let gemma = fixtures.path().join("gemma4");
        write_tp2_gemma4_fixture(&gemma);
        let gemma_results = run_auto_policy_world_two(&gemma);
        assert!(
            gemma_results.iter().all(Result::is_ok),
            "Gemma 4 AutoPolicy TP composition failed: {gemma_results:?}"
        );
    }

    #[test]
    fn public_tp_auto_policy_rejects_qwen35_on_world_two() {
        let fixtures = TestDir::new("tp-qwen35-unsupported");
        let qwen35 = fixtures.path().join("qwen35");
        copy_fixture_dir("tiny_qwen35", &qwen35);
        let results = run_auto_policy_world_two(&qwen35);
        for result in results {
            let err = result.unwrap_err();
            assert!(err.contains("not supported for qwen3_5"), "{err}");
            assert!(err.contains("Qwen3"), "{err}");
            assert!(err.contains("Gemma 4"), "{err}");
        }
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
        let cfg: RunConfig =
            serde_json::from_str(&trimul_invalid_reward_test_config(4242)).unwrap();
        let (launch, signer) = launch_manifest_for_test(&cfg, "test-run", b"prompt");
        let candidate = candidate_for_test(&launch, &signer, "```python\npass\n```\n");
        let run = RunDir::create(tmp.path(), "test-run").unwrap();
        run.write_immutable_launch(&launch.to_pretty_bytes().unwrap(), Some(b"prompt"))
            .unwrap();
        let mut row = serde_json::to_vec(&candidate).unwrap();
        row.push(b'\n');
        std::fs::write(run.candidates_path(), row).unwrap();
        let mut artifact_args = trimul_artifact_args_for_test(tmp.path());
        artifact_args.candidate_sha256 = candidate.record_sha256.clone().unwrap();
        let artifact_err = trimul_artifact_with_trust(&artifact_args, &test_launch_trust_policy())
            .unwrap_err()
            .to_string();

        assert!(score_err.contains("runnable + trimul.reward.partial_correctness"));
        assert!(artifact_err.contains("runnable + trimul.reward.partial_correctness"));
    }

    /// `device` and `base_dtype` selectors deserialize from lowercase strings.
    #[test]
    #[allow(clippy::cognitive_complexity)] // assertion-heavy config parse coverage
    fn device_and_dtype_selectors_parse() {
        let json = r#"{
            "task": "math",
            "model_dir": "/m",
            "device": "cuda",
            "policy": {
                "base_dtype": "bf16",
                "base_quantization": "q8_0",
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
        assert_eq!(cfg.loader_opts().base_quantization, BaseQuantization::Q8_0);
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

    #[test]
    fn unknown_nested_config_fields_are_rejected() {
        let cases = [
            ("trainer", "grad_acum_steps"),
            ("policy", "lora_aplha"),
            ("data", "trian_n"),
        ];
        for (section, typo) in cases {
            let mut value: serde_json::Value =
                serde_json::from_str(&countdown_train_config("")).unwrap();
            value[section][typo] = serde_json::json!(1);

            let err = serde_json::from_value::<RunConfig>(value).unwrap_err();

            assert!(
                err.to_string().contains(&format!("unknown field `{typo}`")),
                "{section}.{typo} was not rejected as an unknown field: {err}"
            );
        }
    }

    #[test]
    fn eos_wire_distinguishes_auto_override_and_explicit_none() {
        let omitted: RunConfig =
            serde_json::from_str(&countdown_train_config("")).expect("omitted EOS means auto");
        assert_eq!(omitted.trainer.eos_token_id, None);

        let mut explicit: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        explicit["trainer"]["eos_token_id"] = serde_json::json!(3);
        let explicit: RunConfig = serde_json::from_value(explicit).unwrap();
        assert_eq!(explicit.trainer.eos_token_id, Some(3));

        let mut disabled: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        disabled["trainer"]["eos_token_id"] = serde_json::json!("none");
        let disabled: RunConfig = serde_json::from_value(disabled).unwrap();
        assert_eq!(disabled.trainer.eos_token_id, None);

        let mut null: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        null["trainer"]["eos_token_id"] = serde_json::Value::Null;
        let err = serde_json::from_value::<RunConfig>(null).unwrap_err();
        assert!(err.to_string().contains("eos_token_id"));

        for invalid in [
            serde_json::json!("off"),
            serde_json::json!([2, 3]),
            serde_json::json!(true),
            serde_json::json!(-1),
        ] {
            let mut value: serde_json::Value =
                serde_json::from_str(&countdown_train_config("")).unwrap();
            value["trainer"]["eos_token_id"] = invalid;
            assert!(serde_json::from_value::<RunConfig>(value).is_err());
        }
    }

    #[test]
    fn eos_wire_round_trips_without_collapsing_auto_or_disabled() {
        let auto: RunConfig = serde_json::from_str(&countdown_train_config("")).unwrap();
        let auto_wire = serde_json::to_value(&auto).unwrap();
        assert!(auto_wire["trainer"].get("eos_token_id").is_none());
        let auto_round_trip: RunConfig = serde_json::from_value(auto_wire).unwrap();
        assert_eq!(auto_round_trip.eos_selection, EosSelection::Checkpoint);

        let mut disabled_wire: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        disabled_wire["trainer"]["eos_token_id"] = serde_json::json!("none");
        let disabled: RunConfig = serde_json::from_value(disabled_wire).unwrap();
        let disabled_wire = serde_json::to_value(&disabled).unwrap();
        assert_eq!(
            disabled_wire["trainer"]["eos_token_id"],
            serde_json::json!("none")
        );
        let disabled_round_trip: RunConfig = serde_json::from_value(disabled_wire).unwrap();
        assert_eq!(disabled_round_trip.eos_selection, EosSelection::Disabled);
    }

    #[test]
    fn checkpoint_eos_is_resolved_before_generation_config_construction() {
        let tmp = TestDir::new("checkpoint-eos-resolution");
        let model_dir = tmp.path().join("model");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let mut value: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        value["model_dir"] = serde_json::json!(model_dir);
        let cfg: RunConfig = serde_json::from_value(value).unwrap();
        let tokenizer = ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

        let resolved = cfg.resolved_trainer_config(&tokenizer).unwrap();
        let gen = GenConfig::from(&resolved);

        assert_eq!(resolved.eos_token_id, Some(3));
        assert_eq!(gen.eos_token_id, Some(3));
    }

    struct EosRecordingPolicy {
        logp: Var,
        enabled: bool,
        seen: Arc<Mutex<Vec<Option<u32>>>>,
    }

    impl EosRecordingPolicy {
        fn new(seen: Arc<Mutex<Vec<Option<u32>>>>) -> Self {
            let logp = Var::from_tensor(&Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap())
                .unwrap();
            Self {
                logp,
                enabled: true,
                seen,
            }
        }
    }

    impl Policy for EosRecordingPolicy {
        fn generate(
            &mut self,
            prompt: &[u32],
            cfg: &GenConfig,
        ) -> CandleResult<ferrl::policy::Rollout> {
            self.seen.lock().unwrap().push(cfg.eos_token_id);
            let eos = cfg
                .eos_token_id
                .ok_or_else(|| candle_core::Error::msg("production setup lost resolved EOS"))?;
            let rows = (0..cfg.group_size)
                .map(|_| {
                    let mut row = prompt.to_vec();
                    row.push(0);
                    row.push(eos);
                    row.resize(prompt.len() + cfg.max_new_tokens, eos);
                    row
                })
                .collect();
            Ok(ferrl::policy::Rollout::new(
                rows,
                prompt.len(),
                vec![2; cfg.group_size],
                None,
            ))
        }

        fn token_logprobs(&self, _rollout: &ferrl::policy::Rollout) -> CandleResult<Tensor> {
            Ok(self.logp.as_tensor().clone())
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        fn adapter_enabled(&self) -> bool {
            self.enabled
        }

        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    impl TensorParallelPolicy for EosRecordingPolicy {
        fn generate_at_tensor_parallel_instrumented(
            &mut self,
            prompt: &[u32],
            cfg: &GenConfig,
            _global_row_base: u64,
            _comm: &dyn ferrl::Comm,
            _telemetry: Option<&mut dyn ferrl::ModelTelemetryRecorder>,
        ) -> CandleResult<ferrl::policy::Rollout> {
            self.generate(prompt, cfg)
        }

        fn token_logprobs_tensor_parallel(
            &self,
            rollout: &ferrl::policy::Rollout,
            _comm: &dyn ferrl::Comm,
        ) -> CandleResult<Tensor> {
            self.token_logprobs(rollout)
        }

        fn token_logprobs_tensor_parallel_detached(
            &self,
            rollout: &ferrl::policy::Rollout,
            _comm: &dyn ferrl::Comm,
        ) -> CandleResult<Tensor> {
            self.token_logprobs_detached(rollout)
        }
    }

    impl CliTrainingPolicy for EosRecordingPolicy {
        fn supports_cli_tensor_parallel(&self) -> bool {
            false
        }
    }

    struct EosSetupReward;

    impl RewardFn for EosSetupReward {
        type Target = ();

        fn reward(
            &self,
            _sample: &Sample<()>,
            _completion: &str,
        ) -> Result<f32, ferrl::RewardError> {
            Ok(0.0)
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // one production seam with exact persisted assertions
    fn production_training_setup_threads_resolved_eos_to_trainer_eval_and_persistence() {
        let tmp = TestDir::new("production-checkpoint-eos-resolution");
        let model_dir = tmp.path().join("model");
        let out_dir = tmp.path().join("runs");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let mut value: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        value["model_dir"] = serde_json::json!(model_dir);
        value["out_dir"] = serde_json::json!(out_dir);
        value["trainer"]["max_new_tokens"] = serde_json::json!(2);
        value["trainer"]["candidate_log_top_k"] = serde_json::json!(1);
        let cfg: RunConfig = serde_json::from_value(value).unwrap();
        let launch = launch_context_for_test(&cfg, "test-run".to_owned(), 0, 1);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let loader_seen = Arc::clone(&seen);

        run_training_with_loader(
            &cfg,
            &Device::Cpu,
            &EosSetupReward,
            &[Sample::new("hello", ())],
            &[Sample::new("hello", ())],
            None,
            None,
            &launch,
            None,
            Some(&TEST_LAUNCH_ATTESTOR),
            move |model_dir, _device, _opts| {
                let tokenizer = ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                    .map_err(|error| CliError::msg(error.to_string()))?;
                Ok((
                    EosRecordingPolicy::new(loader_seen),
                    tokenizer,
                    test_policy_identity(),
                ))
            },
        )
        .unwrap();

        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.len(),
            3,
            "expected train plus base/adapter eval generation"
        );
        assert!(seen.iter().all(|value| *value == Some(3)), "{seen:?}");
        drop(seen);

        let run_root = std::fs::read_dir(&out_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let persisted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_root.join("config.json")).unwrap()).unwrap();
        assert_eq!(persisted["eos_token_id"], serde_json::json!(3));
        let launch: LaunchManifest =
            serde_json::from_slice(&std::fs::read(run_root.join(RunDir::LAUNCH_FILE)).unwrap())
                .unwrap();
        verify_launch_manifest_payload(&launch).unwrap();
        verify_launch_attestation(&launch, &test_launch_trust_policy()).unwrap();
        assert_eq!(launch.payload.ferrl_commit, "01".repeat(20));
        assert_eq!(launch.payload.model.resolved_eos_token_id, Some(3));
        assert_eq!(launch.payload.model.tokenizer_sha256, "11".repeat(32));
        assert_eq!(launch.payload.config.resolved["task"], "countdown");
        let candidate: CandidateRecord = serde_json::from_str(
            std::fs::read_to_string(run_root.join(RunDir::CANDIDATES_FILE))
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        candidate
            .verify_signed_provenance(&launch.payload.candidate_ledger.signing_public_key)
            .unwrap();
        assert_eq!(
            candidate.launch_sha256.as_deref(),
            Some(launch.payload_sha256.as_str())
        );
    }

    #[test]
    fn production_training_rejects_attestation_failure_before_rollout_or_run_publication() {
        let tmp = TestDir::new("production-attestation-rejection");
        let model_dir = tmp.path().join("model");
        let out_dir = tmp.path().join("runs-must-not-exist");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let mut value: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        value["model_dir"] = serde_json::json!(model_dir);
        value["out_dir"] = serde_json::json!(&out_dir);
        value["trainer"]["max_new_tokens"] = serde_json::json!(2);
        value["trainer"]["candidate_log_top_k"] = serde_json::json!(1);
        let cfg: RunConfig = serde_json::from_value(value).unwrap();
        let launch = launch_context_for_test(&cfg, "test-run".to_owned(), 0, 1);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let loader_seen = Arc::clone(&seen);

        let error = run_training_with_loader(
            &cfg,
            &Device::Cpu,
            &EosSetupReward,
            &[Sample::new("hello", ())],
            &[],
            None,
            None,
            &launch,
            None,
            Some(&REJECTING_LAUNCH_ATTESTOR),
            move |model_dir, _device, _opts| {
                let tokenizer = ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                    .map_err(|error| CliError::msg(error.to_string()))?;
                Ok((
                    EosRecordingPolicy::new(loader_seen),
                    tokenizer,
                    test_policy_identity(),
                ))
            },
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("launch attestor rejected request"),
            "{error}"
        );
        assert!(
            seen.lock().unwrap().is_empty(),
            "attestation failure reached rollout"
        );
        assert!(
            !out_dir.exists(),
            "attestation failure created a run directory"
        );
    }

    #[test]
    fn distributed_training_coordinates_attestation_failure_before_rollout_or_publication() {
        let tmp = TestDir::new("distributed-attestation-rejection");
        let model_dir = tmp.path().join("model");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(5))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let model_dir = model_dir.clone();
                    let out_dir = tmp.path().join(format!("rank-{rank}-runs"));
                    scope.spawn(move || {
                        let mut value: serde_json::Value =
                            serde_json::from_str(&countdown_train_config("")).unwrap();
                        value["model_dir"] = serde_json::json!(model_dir);
                        value["out_dir"] = serde_json::json!(&out_dir);
                        value["distributed"] = serde_json::json!({ "enabled": true });
                        value["trainer"]["max_new_tokens"] = serde_json::json!(2);
                        value["trainer"]["candidate_log_top_k"] = serde_json::json!(1);
                        let cfg: RunConfig = serde_json::from_value(value).unwrap();
                        let launch = launch_context_for_test(
                            &cfg,
                            format!("test-group-rank{rank}"),
                            rank,
                            2,
                        );
                        let seen = Arc::new(Mutex::new(Vec::new()));
                        let loader_seen = Arc::clone(&seen);
                        let result = run_training_with_loader(
                            &cfg,
                            &Device::Cpu,
                            &EosSetupReward,
                            &[Sample::new("hello", ())],
                            &[],
                            None,
                            None,
                            &launch,
                            Some(LaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            Some(&RANK_ONE_REJECTING_ATTESTOR),
                            move |model_dir, _device, _opts| {
                                let tokenizer =
                                    ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                                        .map_err(|error| CliError::msg(error.to_string()))?;
                                Ok((
                                    EosRecordingPolicy::new(loader_seen),
                                    tokenizer,
                                    test_policy_identity(),
                                ))
                            },
                        );
                        let rollout_calls = seen.lock().unwrap().len();
                        (
                            rank,
                            result.map_err(|error| error.to_string()),
                            rollout_calls,
                            out_dir,
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, result, rollout_calls, out_dir) in results {
            let error = result.unwrap_err();
            if rank == 1 {
                assert!(
                    error.contains("test rank-one attestor rejection"),
                    "rank {rank}: {error}"
                );
            } else {
                assert!(
                    error.contains("external launch attestation"),
                    "rank {rank}: {error}"
                );
            }
            assert_eq!(rollout_calls, 0, "rank {rank} reached rollout");
            assert!(!out_dir.exists(), "rank {rank} published a run directory");
        }
    }

    #[test]
    fn post_attestation_trimul_asset_substitutions_stop_before_rollout_or_publication() {
        for target in ["image.sif", "eval/eval.py", "eval/task.yml"] {
            let tmp = TestDir::new(&format!("trimul-attested-substitution-{target}"));
            let model_dir = tmp.path().join("model");
            let out_dir = tmp.path().join("runs-must-not-exist");
            write_generation_metadata_fixture(
                &model_dir,
                Some(serde_json::json!(3)),
                &serde_json::json!(4),
            );
            let cfg = trimul_config_with_verifier_fixture(tmp.path(), &model_dir, &out_dir);
            let verifier_assets = cfg.capture_trimul_verifier_assets().unwrap();
            let launch = launch_context_for_test(&cfg, "test-run".to_owned(), 0, 1);
            let seen = Arc::new(Mutex::new(Vec::new()));
            let loader_seen = Arc::clone(&seen);
            let attestor = MutatingLaunchAttestor {
                path: tmp.path().join(target),
                replacement: b"post-attestation replacement".to_vec(),
                rank: None,
            };

            let error = run_training_with_loader(
                &cfg,
                &Device::Cpu,
                &EosSetupReward,
                &[Sample::new("hello", ())],
                &[],
                Some(b"exact prompt"),
                Some(&verifier_assets),
                &launch,
                None,
                Some(&attestor),
                move |model_dir, _device, _opts| {
                    let tokenizer = ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                        .map_err(|error| CliError::msg(error.to_string()))?;
                    Ok((
                        EosRecordingPolicy::new(loader_seen),
                        tokenizer,
                        test_policy_identity(),
                    ))
                },
            )
            .unwrap_err()
            .to_string();

            assert!(
                error.contains("changed after verifier attestation"),
                "{target}: {error}"
            );
            assert!(seen.lock().unwrap().is_empty(), "{target} reached rollout");
            assert!(!out_dir.exists(), "{target} reached run publication");
        }
    }

    #[test]
    fn distributed_post_attestation_trimul_substitution_returns_all_ranks_before_rollout() {
        let tmp = TestDir::new("distributed-trimul-attested-substitution");
        let model_dir = tmp.path().join("model");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(5))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let root = tmp.path().join(format!("rank-{rank}"));
                    std::fs::create_dir_all(&root).unwrap();
                    let model_dir = model_dir.clone();
                    scope.spawn(move || {
                        let out_dir = root.join("runs-must-not-exist");
                        let mut cfg =
                            trimul_config_with_verifier_fixture(&root, &model_dir, &out_dir);
                        cfg.distributed.enabled = true;
                        let verifier_assets = cfg.capture_trimul_verifier_assets().unwrap();
                        let launch = launch_context_for_test(
                            &cfg,
                            format!("test-group-rank{rank}"),
                            rank,
                            2,
                        );
                        let seen = Arc::new(Mutex::new(Vec::new()));
                        let loader_seen = Arc::clone(&seen);
                        let attestor = MutatingLaunchAttestor {
                            path: root.join("eval/task.yml"),
                            replacement: b"rank-local post-attestation replacement".to_vec(),
                            rank: Some(1),
                        };
                        let result = run_training_with_loader(
                            &cfg,
                            &Device::Cpu,
                            &EosSetupReward,
                            &[Sample::new("hello", ())],
                            &[],
                            Some(b"exact prompt"),
                            Some(&verifier_assets),
                            &launch,
                            Some(LaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            Some(&attestor),
                            move |model_dir, _device, _opts| {
                                let tokenizer =
                                    ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                                        .map_err(|error| CliError::msg(error.to_string()))?;
                                Ok((
                                    EosRecordingPolicy::new(loader_seen),
                                    tokenizer,
                                    test_policy_identity(),
                                ))
                            },
                        );
                        let rollout_calls = seen.lock().unwrap().len();
                        (
                            rank,
                            result.map_err(|error| error.to_string()),
                            rollout_calls,
                            out_dir,
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, result, rollout_calls, out_dir) in results {
            let error = result.unwrap_err();
            if rank == 1 {
                assert!(
                    error.contains("changed after verifier attestation"),
                    "{error}"
                );
            } else {
                assert!(
                    error.contains("attested verifier asset revalidation"),
                    "{error}"
                );
            }
            assert_eq!(rollout_calls, 0, "rank {rank} reached rollout");
            assert!(!out_dir.exists(), "rank {rank} reached publication");
        }
    }

    #[test]
    fn production_training_rejects_existing_launch_before_rollout() {
        let tmp = TestDir::new("production-launch-create-new");
        let model_dir = tmp.path().join("model");
        let out_dir = tmp.path().join("runs");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let mut value: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        value["model_dir"] = serde_json::json!(model_dir);
        value["out_dir"] = serde_json::json!(out_dir);
        value["trainer"]["max_new_tokens"] = serde_json::json!(2);
        let cfg: RunConfig = serde_json::from_value(value).unwrap();
        let launch = launch_context_for_test(&cfg, "test-run".to_owned(), 0, 1);
        let existing = RunDir::create(&cfg.out_dir, "test-run").unwrap();
        existing.write_immutable_launch(b"{}", None).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let loader_seen = Arc::clone(&seen);

        let error = run_training_with_loader(
            &cfg,
            &Device::Cpu,
            &EosSetupReward,
            &[Sample::new("hello", ())],
            &[],
            None,
            None,
            &launch,
            None,
            None,
            move |model_dir, _device, _opts| {
                let tokenizer = ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                    .map_err(|error| CliError::msg(error.to_string()))?;
                Ok((
                    EosRecordingPolicy::new(loader_seen),
                    tokenizer,
                    test_policy_identity(),
                ))
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("duplicate run_id"), "{error}");
        assert!(
            seen.lock().unwrap().is_empty(),
            "rollout reached after launch rejection"
        );
    }

    #[test]
    fn distributed_production_setup_rejects_resolved_eos_drift_before_run_publication() {
        let tmp = TestDir::new("production-resolved-eos-consensus");
        let model_dir = tmp.path().join("model");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(0)),
            &serde_json::json!(4),
        );
        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(5))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let model_dir = model_dir.clone();
                    let out_dir = tmp.path().join(format!("rank-{rank}-runs"));
                    std::fs::create_dir_all(&out_dir).unwrap();
                    let sentinel = out_dir.join("sentinel");
                    std::fs::write(&sentinel, format!("rank-{rank}")).unwrap();
                    scope.spawn(move || {
                        let mut value: serde_json::Value =
                            serde_json::from_str(&countdown_train_config("")).unwrap();
                        value["model_dir"] = serde_json::json!(model_dir);
                        value["out_dir"] = serde_json::json!(out_dir);
                        value["distributed"] = serde_json::json!({ "enabled": true });
                        value["trainer"]["max_new_tokens"] = serde_json::json!(2);
                        if rank == 1 {
                            value["trainer"]["eos_token_id"] = serde_json::json!("none");
                        }
                        let cfg: RunConfig = serde_json::from_value(value).unwrap();
                        let launch = launch_context_for_test(
                            &cfg,
                            format!("test-group-rank{rank}"),
                            rank,
                            2,
                        );
                        let seen = Arc::new(Mutex::new(Vec::new()));
                        let loader_seen = Arc::clone(&seen);
                        let result = run_training_with_loader(
                            &cfg,
                            &Device::Cpu,
                            &EosSetupReward,
                            &[Sample::new("hello", ())],
                            &[],
                            None,
                            None,
                            &launch,
                            Some(LaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            None,
                            move |model_dir, _device, _opts| {
                                let tokenizer =
                                    ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                                        .map_err(|error| CliError::msg(error.to_string()))?;
                                Ok((
                                    EosRecordingPolicy::new(loader_seen),
                                    tokenizer,
                                    test_policy_identity(),
                                ))
                            },
                        );
                        let entries = std::fs::read_dir(&cfg.out_dir)
                            .unwrap()
                            .map(|entry| entry.unwrap().file_name())
                            .collect::<Vec<_>>();
                        (
                            rank,
                            result.map_err(|error| error.to_string()),
                            entries,
                            sentinel,
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, result, entries, sentinel) in results {
            let error = result.unwrap_err();
            assert!(
                error.contains("resolved EOS consensus"),
                "rank {rank}: {error}"
            );
            assert_eq!(entries, vec![sentinel.file_name().unwrap().to_os_string()]);
        }
    }

    #[test]
    fn distributed_production_rejects_tokenizer_identity_drift_before_rollout_or_publication() {
        let tmp = TestDir::new("production-tokenizer-identity-consensus");
        let model_dir = tmp.path().join("model");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(5))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let model_dir = model_dir.clone();
                    let out_dir = tmp.path().join(format!("tokenizer-rank-{rank}-runs"));
                    std::fs::create_dir_all(&out_dir).unwrap();
                    let sentinel = out_dir.join("sentinel");
                    std::fs::write(&sentinel, format!("rank-{rank}")).unwrap();
                    scope.spawn(move || {
                        let mut value: serde_json::Value =
                            serde_json::from_str(&countdown_train_config("")).unwrap();
                        value["model_dir"] = serde_json::json!(model_dir);
                        value["out_dir"] = serde_json::json!(out_dir);
                        value["distributed"] = serde_json::json!({ "enabled": true });
                        value["trainer"]["max_new_tokens"] = serde_json::json!(2);
                        let cfg: RunConfig = serde_json::from_value(value).unwrap();
                        let launch = launch_context_for_test(
                            &cfg,
                            format!("test-group-rank{rank}"),
                            rank,
                            2,
                        );
                        let seen = Arc::new(Mutex::new(Vec::new()));
                        let loader_seen = Arc::clone(&seen);
                        let result = run_training_with_loader(
                            &cfg,
                            &Device::Cpu,
                            &EosSetupReward,
                            &[Sample::new("hello", ())],
                            &[],
                            None,
                            None,
                            &launch,
                            Some(LaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            None,
                            move |model_dir, _device, _opts| {
                                let tokenizer =
                                    ferrl::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
                                        .map_err(|error| CliError::msg(error.to_string()))?;
                                let mut identity = test_policy_identity();
                                identity.tokenizer_sha256 = format!("{rank:02x}").repeat(32);
                                Ok((EosRecordingPolicy::new(loader_seen), tokenizer, identity))
                            },
                        );
                        let entries = std::fs::read_dir(&cfg.out_dir)
                            .unwrap()
                            .map(|entry| entry.unwrap().file_name())
                            .collect::<Vec<_>>();
                        let rollout_calls = seen.lock().unwrap().len();
                        (
                            rank,
                            result.map_err(|error| error.to_string()),
                            rollout_calls,
                            entries,
                            sentinel,
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, result, rollout_calls, entries, sentinel) in results {
            let error = result.unwrap_err();
            assert!(
                error.contains("tokenizer/prompt provenance"),
                "rank {rank}: {error}"
            );
            assert_eq!(rollout_calls, 0, "rank {rank} reached rollout");
            assert_eq!(entries, vec![sentinel.file_name().unwrap().to_os_string()]);
        }
    }

    fn assert_missing_checkpoint_eos_requires_explicit_mode(
        model_dir: &Path,
        tokenizer_path: &Path,
    ) {
        write_generation_metadata_fixture(model_dir, None, &serde_json::json!(4));
        let tokenizer = ferrl::HfTokenizer::from_file(tokenizer_path).unwrap();
        assert!(countdown_run_config_with_eos(model_dir, None)
            .resolve_eos_token_id(&tokenizer)
            .is_err());
        assert_eq!(
            countdown_run_config_with_eos(model_dir, Some(serde_json::json!("none")))
                .resolve_eos_token_id(&tokenizer)
                .unwrap(),
            None
        );
    }

    fn assert_multi_checkpoint_eos_requires_declared_override(
        model_dir: &Path,
        tokenizer_path: &Path,
    ) {
        write_generation_metadata_fixture(
            model_dir,
            Some(serde_json::json!([2, 3])),
            &serde_json::json!(4),
        );
        let tokenizer = ferrl::HfTokenizer::from_file(tokenizer_path).unwrap();
        assert!(countdown_run_config_with_eos(model_dir, None)
            .resolve_eos_token_id(&tokenizer)
            .is_err());
        assert_eq!(
            countdown_run_config_with_eos(model_dir, Some(serde_json::json!(3)))
                .resolve_eos_token_id(&tokenizer)
                .unwrap(),
            Some(3)
        );
        assert!(
            countdown_run_config_with_eos(model_dir, Some(serde_json::json!(1)))
                .resolve_eos_token_id(&tokenizer)
                .is_err()
        );
    }

    fn assert_explicit_eos_respects_model_and_tokenizer_bounds(
        model_dir: &Path,
        tokenizer_path: &Path,
    ) {
        write_generation_metadata_fixture(
            model_dir,
            Some(serde_json::json!(3)),
            &serde_json::json!(4),
        );
        let tokenizer = ferrl::HfTokenizer::from_file(tokenizer_path).unwrap();
        assert!(
            countdown_run_config_with_eos(model_dir, Some(serde_json::json!(4)))
                .resolve_eos_token_id(&tokenizer)
                .is_err()
        );

        write_generation_metadata_fixture(
            model_dir,
            Some(serde_json::json!(4)),
            &serde_json::json!(4),
        );
        move_tiny_tokenizer_special_id(tokenizer_path, 4);
        let tokenizer = ferrl::HfTokenizer::from_file(tokenizer_path).unwrap();
        assert!(tokenizer.contains_id(4));
        let err = countdown_run_config_with_eos(model_dir, None)
            .resolve_eos_token_id(&tokenizer)
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside model vocab_size 4"), "{err}");
    }

    #[test]
    fn checkpoint_eos_resolution_fails_closed_and_validates_overrides() {
        let tmp = TestDir::new("checkpoint-eos-negatives");
        let model_dir = tmp.path().join("model");
        let tokenizer_path = model_dir.join("tokenizer.json");

        assert_missing_checkpoint_eos_requires_explicit_mode(&model_dir, &tokenizer_path);
        assert_multi_checkpoint_eos_requires_declared_override(&model_dir, &tokenizer_path);
        assert_explicit_eos_respects_model_and_tokenizer_bounds(&model_dir, &tokenizer_path);
    }

    #[test]
    fn checkpoint_eos_accepts_a_sparse_but_present_tokenizer_id() {
        let tmp = TestDir::new("checkpoint-eos-sparse-tokenizer");
        let model_dir = tmp.path().join("model");
        write_generation_metadata_fixture(
            &model_dir,
            Some(serde_json::json!(4)),
            &serde_json::json!(5),
        );
        let tokenizer_path = model_dir.join("tokenizer.json");
        move_tiny_tokenizer_special_id(&tokenizer_path, 4);

        let mut value: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        value["model_dir"] = serde_json::json!(model_dir);
        let cfg: RunConfig = serde_json::from_value(value).unwrap();
        let tokenizer = ferrl::HfTokenizer::from_file(tokenizer_path).unwrap();

        assert_eq!(tokenizer.vocab_size(), 4);
        assert!(tokenizer.contains_id(4));
        assert_eq!(cfg.resolve_eos_token_id(&tokenizer).unwrap(), Some(4));
    }

    #[test]
    fn distributed_config_digest_distinguishes_eos_auto_from_disabled() {
        let tmp = TestDir::new("eos-selector-consensus");
        let auto_path = tmp.path().join("auto.json");
        let disabled_path = tmp.path().join("disabled.json");
        std::fs::write(&auto_path, countdown_train_config("")).unwrap();
        let mut disabled: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        disabled["trainer"]["eos_token_id"] = serde_json::json!("none");
        std::fs::write(&disabled_path, serde_json::to_vec(&disabled).unwrap()).unwrap();

        let auto = RunConfig::load_for_launch(&auto_path).unwrap();
        let disabled = RunConfig::load_for_launch(&disabled_path).unwrap();

        assert_ne!(auto.consensus_digest, disabled.consensus_digest);
    }

    #[test]
    fn distributed_resolved_eos_consensus_rejects_rank_local_metadata_drift() {
        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    scope.spawn(move || {
                        validate_resolved_eos_consensus(
                            Some(3 + u32::try_from(rank).unwrap()),
                            Some(&comm),
                        )
                        .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for result in results {
            assert!(result
                .unwrap_err()
                .contains("resolved different EOS token semantics"));
        }
    }

    #[test]
    fn distributed_resolved_eos_consensus_accepts_equal_some_and_none() {
        for eos in [Some(3), None] {
            let results = std::thread::scope(|scope| {
                ferrl::LocalComm::world(2)
                    .into_iter()
                    .map(|comm| {
                        scope.spawn(move || {
                            validate_resolved_eos_consensus(eos, Some(&comm))
                                .map_err(|error| error.to_string())
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert!(results.iter().all(Result::is_ok), "{eos:?}: {results:?}");
        }
    }

    fn assert_no_update_configs_rejected() {
        let cases = [
            ("zero-steps", "", "steps", serde_json::json!(0)),
            ("local-group-one", "", "group_size", serde_json::json!(1)),
        ];
        for (tag, extra, field, value) in cases {
            let tmp = TestDir::new(tag);
            let path = tmp.path().join("run.json");
            let mut json: serde_json::Value =
                serde_json::from_str(&countdown_train_config(extra)).unwrap();
            json["trainer"][field] = value;
            std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();

            assert!(RunConfig::load(&path).is_err(), "{tag} unexpectedly loaded");
        }
    }

    fn assert_f32_lora_scale_configs_rejected() {
        for (tag, alpha) in [
            ("zero-alpha", 0.0),
            ("negative-alpha", -1.0),
            ("underflow-alpha", f64::MIN_POSITIVE),
            ("overflow-alpha", f64::MAX),
        ] {
            let tmp = TestDir::new(tag);
            let path = tmp.path().join("run.json");
            let json = countdown_train_config(&format!(r#""policy": {{ "lora_alpha": {alpha} }}"#));
            std::fs::write(&path, json).unwrap();

            assert!(RunConfig::load(&path).is_err(), "{tag} unexpectedly loaded");
        }
    }

    fn assert_bf16_lora_scale_configs_rejected() {
        for (tag, alpha) in [
            ("bf16-underflow-alpha", f64::MIN_POSITIVE),
            ("bf16-overflow-alpha", f64::MAX),
        ] {
            let tmp = TestDir::new(tag);
            let path = tmp.path().join("run.json");
            let json = countdown_train_config(&format!(
                r#""policy": {{ "lora_alpha": {alpha}, "base_dtype": "bf16" }}"#
            ));
            std::fs::write(&path, json).unwrap();

            assert!(RunConfig::load(&path).is_err(), "{tag} unexpectedly loaded");
        }
    }

    fn assert_lora_scale_validation_uses_base_compute_dtype() {
        let alpha = 1e-42;
        let f32 = countdown_train_config(&format!(
            r#""policy": {{ "lora_rank": 1, "lora_alpha": {alpha}, "base_dtype": "f32" }}"#
        ));
        let bf16 = countdown_train_config(&format!(
            r#""policy": {{ "lora_rank": 1, "lora_alpha": {alpha}, "base_dtype": "bf16" }}"#
        ));
        assert!(serde_json::from_str::<RunConfig>(&f32)
            .unwrap()
            .validate_current_config_support()
            .is_ok());
        assert!(serde_json::from_str::<RunConfig>(&bf16)
            .unwrap()
            .validate_current_config_support()
            .is_err());
    }

    fn assert_nonfinite_in_memory_lora_alpha_rejected() {
        let mut cfg: RunConfig = serde_json::from_str(&countdown_train_config("")).unwrap();
        for alpha in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            cfg.policy.lora_alpha = alpha;
            assert!(cfg.validate_current_config_support().is_err());
        }
    }

    #[test]
    fn run_config_rejects_no_update_and_invalid_policy_paths() {
        assert_no_update_configs_rejected();
        assert_f32_lora_scale_configs_rejected();
        assert_bf16_lora_scale_configs_rejected();
        assert_lora_scale_validation_uses_base_compute_dtype();
        assert_nonfinite_in_memory_lora_alpha_rejected();
    }

    #[test]
    fn invalid_no_update_configs_stop_before_device_or_model_setup() {
        let cases = [
            ("steps", serde_json::json!(0), None),
            ("group_size", serde_json::json!(1), None),
            ("lora_alpha", serde_json::json!(0.0), Some("policy")),
            ("train_n", serde_json::json!(0), Some("data")),
        ];
        for (field, value, section) in cases {
            let tmp = TestDir::new(&format!("pre-device-{field}"));
            let path = tmp.path().join("run.json");
            let mut json: serde_json::Value =
                serde_json::from_str(&countdown_train_config("")).unwrap();
            match section {
                Some(section) => json[section][field] = value,
                None => json["trainer"][field] = value,
            }
            std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
            let prepared = std::cell::Cell::new(false);

            let result = train_with_launch_runtime(
                &TrainArgs { config: path },
                None,
                test_build_source_identity(),
                |_, _| {
                    prepared.set(true);
                    Ok(Device::Cpu)
                },
            );

            assert!(result.is_err(), "{field} unexpectedly reached training");
            assert!(!prepared.get(), "{field} reached device/model setup");
        }
    }

    #[test]
    fn asymmetric_invalid_build_source_returns_all_ranks_before_device_or_publication() {
        let tmp = TestDir::new("asymmetric-invalid-build-source");
        let config_path = tmp.path().join("run.json");
        let out_dir = tmp.path().join("runs-must-not-exist");
        let mut json: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        json["out_dir"] = serde_json::json!(&out_dir);
        json["distributed"] = serde_json::json!({ "enabled": true });
        std::fs::write(&config_path, serde_json::to_vec(&json).unwrap()).unwrap();
        let prepared = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(5))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let config_path = config_path.clone();
                    let prepared = Arc::clone(&prepared);
                    scope.spawn(move || {
                        let local_source = if rank == 1 {
                            Err(CliError::msg("test dirty source on rank one"))
                        } else {
                            Ok(test_build_source_identity())
                        };
                        (
                            rank,
                            train_with_launch_runtime_and_source_result(
                                &TrainArgs {
                                    config: config_path,
                                },
                                Some(LaunchRuntime {
                                    device: Device::Cpu,
                                    comm: Box::new(comm),
                                }),
                                local_source,
                                move |_, _| {
                                    prepared.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                    Ok(Device::Cpu)
                                },
                            )
                            .map_err(|error| error.to_string()),
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, result) in results {
            let error = result.unwrap_err();
            if rank == 1 {
                assert!(error.contains("test dirty source on rank one"), "{error}");
            } else {
                assert!(
                    error.contains("embedded build source validation"),
                    "{error}"
                );
            }
        }
        assert_eq!(prepared.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(!out_dir.exists());
    }

    #[test]
    fn live_dp_world_one_group_one_rejects_before_device_model_or_run_creation() {
        let tmp = TestDir::new("live-dp-world-one-group-one");
        let config_path = tmp.path().join("run.json");
        let out_dir = tmp.path().join("runs-must-not-exist");
        let mut json: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        json["out_dir"] = serde_json::json!(&out_dir);
        json["distributed"] = serde_json::json!({ "enabled": true });
        json["trainer"]["group_size"] = serde_json::json!(1);
        json["trainer"]["reward_group_scope"] = serde_json::json!("distributed_same_prompt");
        std::fs::write(&config_path, serde_json::to_vec(&json).unwrap()).unwrap();

        let prepared = std::cell::Cell::new(false);
        let result = train_with_launch_runtime(
            &TrainArgs {
                config: config_path,
            },
            Some(LaunchRuntime {
                device: Device::Cpu,
                comm: Box::new(ferrl::SoloComm),
            }),
            test_build_source_identity(),
            |_, _| {
                prepared.set(true);
                Err(CliError::msg(
                    "prepare-device sentinel: ineffective live DP group reached device setup",
                ))
            },
        );

        let error = result.unwrap_err().to_string();
        assert!(error.contains("effective reward-group size"), "{error}");
        assert!(!prepared.get(), "ineffective group reached device setup");
        assert!(!out_dir.exists(), "ineffective group created its run root");
    }

    #[test]
    fn live_dp_reward_group_overflow_rejects_before_device_model_or_run_creation() {
        let tmp = TestDir::new("live-dp-reward-group-overflow");
        let config_path = tmp.path().join("run.json");
        let out_dir = tmp.path().join("runs-must-not-exist");
        let mut json: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        json["out_dir"] = serde_json::json!(&out_dir);
        json["distributed"] = serde_json::json!({ "enabled": true });
        json["trainer"]["group_size"] = serde_json::json!(usize::MAX);
        json["trainer"]["reward_group_scope"] = serde_json::json!("distributed_same_prompt");
        std::fs::write(&config_path, serde_json::to_vec(&json).unwrap()).unwrap();

        let prepared = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let results = std::thread::scope(|scope| {
            ferrl::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(5))
                .into_iter()
                .map(|comm| {
                    let config_path = config_path.clone();
                    let prepared = Arc::clone(&prepared);
                    scope.spawn(move || {
                        train_with_launch_runtime(
                            &TrainArgs {
                                config: config_path,
                            },
                            Some(LaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            test_build_source_identity(),
                            move |_, _| {
                                prepared.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                Err(CliError::msg(
                                    "prepare-device sentinel: overflowing live DP group reached device setup",
                                ))
                            },
                        )
                        .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for result in results {
            let error = result.unwrap_err();
            assert!(
                error.contains("effective distributed reward-group size overflows usize"),
                "{error}"
            );
        }
        assert_eq!(
            prepared.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "overflowing group reached device setup"
        );
        assert!(!out_dir.exists(), "overflowing group created its run root");
    }

    #[test]
    fn procedural_tasks_reject_zero_training_rows() {
        for task in ["countdown", "trimul"] {
            let tmp = TestDir::new(&format!("{task}-zero-train-rows"));
            let path = tmp.path().join("run.json");
            let mut json: serde_json::Value =
                serde_json::from_str(&countdown_train_config("")).unwrap();
            json["task"] = serde_json::json!(task);
            json["data"] = serde_json::json!({ "train_n": 0, "eval_n": 0 });
            std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();

            assert!(
                RunConfig::load(&path).is_err(),
                "{task} unexpectedly loaded"
            );
        }

        let tmp = TestDir::new("math-zero-procedural-count-ignored");
        let path = tmp.path().join("run.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        json["task"] = serde_json::json!("math");
        json["data"] = serde_json::json!({ "path": "math.jsonl", "train_n": 0 });
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(RunConfig::load(&path).is_ok());
    }

    #[test]
    fn countdown_rejects_an_unrepresentable_requested_dataset_size() {
        let tmp = TestDir::new("countdown-dataset-size-overflow");
        let path = tmp.path().join("run.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&countdown_train_config("")).unwrap();
        json["data"] = serde_json::json!({
            "train_n": usize::MAX,
            "eval_n": 1,
        });
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();

        assert!(RunConfig::load(&path).is_err());
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
        assert!(Cli::try_parse_from([
            "ferrl",
            "train",
            "--config",
            "run.json",
            "--ferrl-commit",
            "0123456789012345678901234567890123456789",
        ])
        .is_err());
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
            1.25,
            Some("trimul:no_code".to_string()),
            Some(serde_json::json!({"reward_scheme": "trimul_shaped_v1"})),
            "prompt-hash",
            "config-hash",
        );

        let row = serde_json::to_value(record).unwrap();

        assert_eq!(row["reward"], 1.25);
        assert_eq!(row["reward_metadata"]["reward_scheme"], "trimul_shaped_v1");
        assert_eq!(row["input_metadata"]["input"], "meta");
        assert_eq!(row["completion_sha256"], sha256_hex(b"abc"));
        assert_eq!(row["external_score"]["source_id"], "public-source-7");
        assert_eq!(row["external_score"]["source_index"], 7);
        assert!(row["external_score"].get("source").is_none());
    }

    #[test]
    fn trimul_score_rejects_nonfinite_rewards_before_record_construction() {
        validate_trimul_score_rewards(&[0.0, 1.0]).unwrap();
        for rewards in [vec![0.0, f32::NAN], vec![f32::NEG_INFINITY, f32::INFINITY]] {
            let error = validate_trimul_score_rewards(&rewards).unwrap_err();
            assert!(error.to_string().contains("non-finite reward"));
        }
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
            "--run-dir",
            "runs/trimul-1",
            "--candidate-sha256",
            "1111111111111111111111111111111111111111111111111111111111111111",
            "--out",
            "artifact",
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
        ])
        .unwrap();
        match a.cmd {
            Command::TrimulArtifact(a) => {
                assert_eq!(a.run_dir, PathBuf::from("runs/trimul-1"));
                assert_eq!(a.candidate_sha256, "11".repeat(32));
            }
            _ => panic!("expected trimul-artifact"),
        }
    }

    #[test]
    fn clap_rejects_operator_authored_artifact_candidate_provenance() {
        let error = Cli::try_parse_from([
            "ferrl",
            "trimul-artifact",
            "--run-dir",
            "runs/trimul-1",
            "--candidate-sha256",
            "1111111111111111111111111111111111111111111111111111111111111111",
            "--completion",
            "replacement.txt",
        ])
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("unexpected argument '--completion'"),
            "{error}"
        );
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
        let image = eval_dir.join("image.sif");
        std::fs::write(&image, b"test image").unwrap();
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
                  "image": "{}",
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
                image.display(),
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
    fn launch_requires_a_full_lowercase_training_commit() {
        assert_eq!(
            validate_full_git_commit(&"ab".repeat(20)).unwrap(),
            "ab".repeat(20)
        );
        assert!(validate_full_git_commit("abc123").is_err());
        assert!(validate_full_git_commit(&"AB".repeat(20)).is_err());
        assert!(validated_build_source_identity("unknown", false).is_err());
        assert!(validated_build_source_identity(&"ab".repeat(20), true).is_err());
        assert_eq!(
            validated_build_source_identity(&"ab".repeat(20), false)
                .unwrap()
                .commit,
            "ab".repeat(20)
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
    fn launch_bound_candidate_rejects_completion_and_coordinate_mutation() {
        let cfg: RunConfig = serde_json::from_str(&trimul_score_test_config(4242)).unwrap();
        let (launch, signer) = launch_manifest_for_test(&cfg, "test-run", b"prompt");
        let candidate = candidate_for_test(&launch, &signer, "```python\npass\n```\n");
        candidate
            .verify_signed_provenance(&launch.payload.candidate_ledger.signing_public_key)
            .unwrap();

        let mut completion_mutation = candidate.clone();
        completion_mutation.completion.push_str("# changed");
        assert!(completion_mutation.verify_provenance().is_err());

        let mut coordinate_mutation = candidate;
        coordinate_mutation.group_index += 1;
        assert!(coordinate_mutation.verify_provenance().is_err());
    }

    #[test]
    fn launch_trust_policy_rejects_untrusted_shapes() {
        let mut policy = test_launch_trust_policy();
        policy.keys.push(policy.keys[0].clone());
        assert!(validate_launch_trust_policy(&policy)
            .unwrap_err()
            .to_string()
            .contains("repeats key id"));

        let mut policy = test_launch_trust_policy();
        policy.keys[0].key_id = "operator/path".to_owned();
        assert!(validate_launch_trust_policy(&policy)
            .unwrap_err()
            .to_string()
            .contains("invalid id or algorithm"));

        let mut policy = test_launch_trust_policy();
        policy.keys[0].public_key = "00".repeat(31);
        assert!(validate_launch_trust_policy(&policy)
            .unwrap_err()
            .to_string()
            .contains("64 lowercase hexadecimal"));
    }

    #[cfg(unix)]
    #[test]
    fn launch_attestation_protocol_binds_the_exact_payload() {
        use std::io::BufRead as _;
        use std::os::unix::net::UnixStream;

        let cfg: RunConfig = serde_json::from_str(&trimul_score_test_config(4242)).unwrap();
        let (mut manifest, _candidate_signer) =
            launch_manifest_for_test(&cfg, "test-run", b"prompt");
        manifest.attestation = None;
        let expected_digest = manifest.payload_sha256.clone();
        let (mut client, server) = UnixStream::pair().unwrap();
        let server = std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(server);
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            let request: LaunchAttestationRequest = serde_json::from_str(&request).unwrap();
            assert_eq!(
                request.contract_version,
                LAUNCH_ATTESTATION_CONTRACT_VERSION
            );
            assert_eq!(request.kind, LAUNCH_ATTESTATION_REQUEST_KIND);
            assert_eq!(request.algorithm, LAUNCH_ATTESTATION_ALGORITHM);
            let payload_bytes = decode_lower_hex(
                "test launch payload",
                &request.launch_payload_json_hex,
                request.launch_payload_json_hex.len() / 2,
            )
            .unwrap();
            let payload: LaunchPayload = serde_json::from_slice(&payload_bytes).unwrap();
            assert_eq!(serde_json::to_vec(&payload).unwrap(), payload_bytes);
            let reconstructed = LaunchManifest::new(payload).unwrap();
            assert_eq!(request.launch_payload_sha256, reconstructed.payload_sha256);
            assert_eq!(request.launch_payload_sha256, expected_digest);
            let attestation = TEST_LAUNCH_ATTESTOR.attest(&reconstructed).unwrap();
            serde_json::to_writer(reader.get_mut(), &attestation).unwrap();
        });

        let attestation =
            exchange_launch_attestation(&mut client, &manifest, &test_launch_trust_policy())
                .unwrap();
        server.join().unwrap();
        manifest.attestation = Some(attestation);
        verify_launch_attestation(&manifest, &test_launch_trust_policy()).unwrap();
    }

    #[test]
    fn trimul_artifact_ingest_selects_one_exact_launch_bound_row() {
        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-bound-row", 0, 1, 0, 1);

        let bound = load_bound_candidate_for_test(&run_dir, &candidate_sha256).unwrap();

        assert_eq!(bound.launch.payload.run.run_id, "trimul-1");
        assert_eq!(
            bound.candidate.record_sha256.as_deref(),
            Some(candidate_sha256.as_str())
        );
        assert_eq!(bound.prompt_bytes, b"prompt");
        assert_eq!(
            bound.candidate_row_bytes,
            serde_json::to_vec(&bound.candidate).unwrap()
        );
    }

    #[test]
    fn trimul_artifact_ingest_rejects_noncanonical_launch_encoding() {
        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-launch-encoding", 0, 1, 0, 1);
        let launch_path = run_dir.join(RunDir::LAUNCH_FILE);
        let launch_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&launch_path).unwrap()).unwrap();
        std::fs::write(&launch_path, serde_json::to_vec(&launch_value).unwrap()).unwrap();

        let error = load_bound_candidate_for_test(&run_dir, &candidate_sha256)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("exact canonical production encoding"),
            "{error}"
        );
    }

    #[test]
    fn trimul_artifact_rejects_post_attestation_verifier_substitution() {
        for target in ["image.sif", "eval/utils.py", "eval/task.yml"] {
            let tmp = TestDir::new(&format!("artifact-verifier-substitution-{target}"));
            let cfg = trimul_config_with_verifier_fixture(
                tmp.path(),
                &tmp.path().join("model"),
                &tmp.path().join("runs"),
            );
            let assets = cfg.capture_trimul_verifier_assets().unwrap();
            let (launch, _signer) = launch_manifest_for_test(&cfg, "test-run", b"exact prompt");
            let mut payload = launch.payload;
            payload.verifier = Some(assets.identity().clone());
            let launch = attest_launch_for_test(LaunchManifest::new(payload).unwrap());
            std::fs::write(tmp.path().join(target), b"post-attestation replacement").unwrap();

            let error = capture_attested_trimul_verifier_assets(&cfg, &launch)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("do not match the attested launch identity"),
                "{target}: {error}"
            );
        }
    }

    #[test]
    fn trimul_artifact_rejects_an_appended_row_signed_by_an_operator_key() {
        let (_tmp, run_dir, _native_candidate_sha256) =
            write_bound_candidate_run("artifact-operator-row", 0, 1, 0, 1);
        let launch: LaunchManifest =
            serde_json::from_slice(&std::fs::read(run_dir.join(RunDir::LAUNCH_FILE)).unwrap())
                .unwrap();
        let attacker = CandidateSigner::generate().unwrap();
        let forged = attacker
            .sign_candidate(
                &CandidateRecord::new(
                    0,
                    0,
                    1,
                    12,
                    0,
                    9.0,
                    3,
                    "```python\n# externally authored fast kernel\n```\n".to_owned(),
                ),
                &launch.payload_sha256,
            )
            .unwrap();
        let forged_sha256 = forged.record_sha256.clone().unwrap();
        let mut writer =
            ferrl::telemetry::CandidateWriter::open(run_dir.join(RunDir::CANDIDATES_FILE)).unwrap();
        writer.append(&forged).unwrap();
        drop(writer);

        let error = load_bound_candidate_for_test(&run_dir, &forged_sha256)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("record_signature was not made by the launch signing key"),
            "{error}"
        );
    }

    #[test]
    fn trimul_artifact_rejects_operator_rekeyed_whole_launch() {
        let (_tmp, run_dir, _native_candidate_sha256) =
            write_bound_candidate_run("artifact-operator-launch", 0, 1, 0, 1);
        let launch_path = run_dir.join(RunDir::LAUNCH_FILE);
        let original: LaunchManifest =
            serde_json::from_slice(&std::fs::read(&launch_path).unwrap()).unwrap();

        let candidate_signer = CandidateSigner::generate().unwrap();
        let mut payload = original.payload;
        payload.candidate_ledger.signing_public_key = candidate_signer.public_key_hex();
        let mut forged_launch = LaunchManifest::new(payload).unwrap();
        let attacker_root = Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).unwrap();
        let attacker_root = Ed25519KeyPair::from_pkcs8(attacker_root.as_ref()).unwrap();
        let message = launch_attestation_message(&forged_launch.payload_sha256);
        forged_launch.attestation = Some(LaunchAttestation {
            contract_version: LAUNCH_ATTESTATION_CONTRACT_VERSION,
            kind: LAUNCH_ATTESTATION_KIND.to_owned(),
            algorithm: LAUNCH_ATTESTATION_ALGORITHM.to_owned(),
            key_id: "operator-root".to_owned(),
            launch_payload_sha256: forged_launch.payload_sha256.clone(),
            signature: lower_hex_bytes(attacker_root.sign(message.as_bytes()).as_ref()),
        });
        let forged_candidate = candidate_signer
            .sign_candidate(
                &CandidateRecord::new(
                    0,
                    0,
                    1,
                    12,
                    0,
                    9.0,
                    3,
                    "```python\n# operator-authored replacement\n```\n".to_owned(),
                ),
                &forged_launch.payload_sha256,
            )
            .unwrap();
        let forged_sha256 = forged_candidate.record_sha256.clone().unwrap();
        std::fs::write(&launch_path, forged_launch.to_pretty_bytes().unwrap()).unwrap();
        let mut row = serde_json::to_vec(&forged_candidate).unwrap();
        row.push(b'\n');
        std::fs::write(run_dir.join(RunDir::CANDIDATES_FILE), row).unwrap();

        let error = load_bound_candidate_for_test(&run_dir, &forged_sha256)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("launch attestation key \"operator-root\" is not trusted"),
            "{error}"
        );

        forged_launch.attestation.as_mut().unwrap().key_id = TEST_ATTESTATION_KEY_ID.to_owned();
        std::fs::write(&launch_path, forged_launch.to_pretty_bytes().unwrap()).unwrap();
        let error = load_bound_candidate_for_test(&run_dir, &forged_sha256)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("launch attestation signature is invalid"),
            "{error}"
        );
    }

    #[test]
    fn trimul_artifact_ingest_rejects_mutated_row_before_verification() {
        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-mutated-row", 0, 1, 0, 1);
        let ledger_path = run_dir.join(RunDir::CANDIDATES_FILE);
        let mut row: CandidateRecord = serde_json::from_slice(
            std::fs::read(&ledger_path)
                .unwrap()
                .strip_suffix(b"\n")
                .unwrap(),
        )
        .unwrap();
        row.completion = "```python\nchanged\n```\n".to_owned();
        let mut bytes = serde_json::to_vec(&row).unwrap();
        bytes.push(b'\n');
        std::fs::write(&ledger_path, bytes).unwrap();

        let error = load_bound_candidate_for_test(&run_dir, &candidate_sha256)
            .unwrap_err()
            .to_string();

        assert!(error.contains("record_sha256 mismatch"), "{error}");
    }

    #[test]
    fn trimul_artifact_ingest_rejects_operator_added_row_fields() {
        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-extra-row-field", 0, 1, 0, 1);
        let ledger_path = run_dir.join(RunDir::CANDIDATES_FILE);
        let mut row: serde_json::Value = serde_json::from_slice(
            std::fs::read(&ledger_path)
                .unwrap()
                .strip_suffix(b"\n")
                .unwrap(),
        )
        .unwrap();
        row["operator_provenance"] = serde_json::json!("not launch bound");
        let mut bytes = serde_json::to_vec(&row).unwrap();
        bytes.push(b'\n');
        std::fs::write(&ledger_path, bytes).unwrap();

        let error = load_bound_candidate_for_test(&run_dir, &candidate_sha256)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("unknown field \"operator_provenance\""),
            "{error}"
        );
    }

    #[test]
    fn trimul_artifact_ingest_rejects_reencoded_candidate_rows() {
        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-reencoded-row", 0, 1, 0, 1);
        let ledger_path = run_dir.join(RunDir::CANDIDATES_FILE);
        let row: serde_json::Value = serde_json::from_slice(
            std::fs::read(&ledger_path)
                .unwrap()
                .strip_suffix(b"\n")
                .unwrap(),
        )
        .unwrap();
        let bytes = format!(" {}\n", serde_json::to_string(&row).unwrap()).into_bytes();
        std::fs::write(&ledger_path, bytes).unwrap();

        let error = load_bound_candidate_for_test(&run_dir, &candidate_sha256)
            .unwrap_err()
            .to_string();

        assert!(error.contains("exact production encoding"), "{error}");
    }

    #[test]
    fn trimul_artifact_ingest_rejects_distributed_rank_rebinding() {
        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-rank-rebinding", 1, 2, 0, 2);

        let error = load_bound_candidate_for_test(&run_dir, &candidate_sha256)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("rank/world disagree with launch.json"),
            "{error}"
        );
    }

    #[test]
    fn trimul_artifact_ingest_rejects_prompt_and_launch_mutation() {
        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-prompt-mutation", 0, 1, 0, 1);
        std::fs::write(run_dir.join(RunDir::PROMPT_FILE), b"changed").unwrap();
        let prompt_error = load_bound_candidate_for_test(&run_dir, &candidate_sha256)
            .unwrap_err()
            .to_string();
        assert!(prompt_error.contains("prompt bytes"), "{prompt_error}");

        let (_tmp, run_dir, candidate_sha256) =
            write_bound_candidate_run("artifact-launch-mutation", 0, 1, 0, 1);
        let launch_path = run_dir.join(RunDir::LAUNCH_FILE);
        let mut launch: LaunchManifest =
            serde_json::from_slice(&std::fs::read(&launch_path).unwrap()).unwrap();
        launch.payload.model.tokenizer_sha256 = "55".repeat(32);
        std::fs::write(&launch_path, launch.to_pretty_bytes().unwrap()).unwrap();
        let launch_error = load_bound_candidate_for_test(&run_dir, &candidate_sha256)
            .unwrap_err()
            .to_string();
        assert!(
            launch_error.contains("launch payload hash mismatch"),
            "{launch_error}"
        );
    }

    #[test]
    fn artifact_manifest_records_base_quantization() {
        let cfg: RunConfig = serde_json::from_str(
            r#"{
                "task": "trimul",
                "model_dir": "/m",
                "policy": {
                    "base_dtype": "bf16",
                    "base_quantization": "q8_0"
                },
                "trimul": {
                  "prompt_path": "/prompt.txt",
                  "submission_extract_mode": "final_fence",
                  "image": "/image.sif",
                  "eval_dir": "/eval",
                  "scratch_root": "/scratch",
                  "secret_seed": 4242
                },
                "trainer": { "steps": 1, "group_size": 2, "max_new_tokens": 8,
                  "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
                  "lr": 1e-5, "weight_decay": 0.0,
                  "loss_type": "grpo", "scale_rewards": "group" }
            }"#,
        )
        .unwrap();
        let args = trimul_artifact_args_for_test(Path::new("artifact-provenance"));
        let (launch, signer) = launch_manifest_for_test(&cfg, "test-run", b"prompt");
        let candidate = candidate_for_test(&launch, &signer, "```python\npass\n```\n");
        let candidate_row = serde_json::to_vec(&candidate).unwrap();
        let launch_bytes = launch.to_pretty_bytes().unwrap();
        let inputs = ArtifactInputs {
            gpu: "H100".to_string(),
            launch: &launch,
            launch_bytes: &launch_bytes,
            candidate: &candidate,
            candidate_row_bytes: &candidate_row,
            raw_completion: &candidate.completion,
            prompt_bytes: b"prompt",
            submission: "pass\n",
            baseline_median: 1.0,
            test_cases: 1,
            benchmark_cases: 1,
            runs: Vec::new(),
            accepted: false,
        };

        let manifest = build_manifest(&args, &cfg, &inputs);
        let json = serde_json::to_string(&manifest).unwrap();

        assert_eq!(manifest.model.base_dtype, "bf16");
        assert_eq!(manifest.model.base_quantization, "q8_0");
        assert_eq!(manifest.launch_file_sha256, sha256_hex(&launch_bytes));
        assert!(json.contains(r#""base_quantization":"q8_0""#));
    }

    #[test]
    fn artifact_report_matches_the_contract_outline() {
        let manifest = ArtifactManifest {
            contract_version: 2,
            task: "trimul",
            ferrl_commit: "01".repeat(20),
            run_id: "trimul-1".to_string(),
            launch_sha256: "02".repeat(32),
            launch_file_sha256: "03".repeat(32),
            launch_attestation_key_id: "test-root-1".to_owned(),
            launch_attestation_algorithm: LAUNCH_ATTESTATION_ALGORITHM.to_owned(),
            candidate: CandidateManifest {
                record_sha256: "04".repeat(32),
                record_signature: "06".repeat(64),
                ledger_row_sha256: "05".repeat(32),
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
                checkpoint_policy_sha256: "06".repeat(32),
                tokenizer_sha256: "07".repeat(32),
                lora_rank: 8,
                lora_alpha: 16.0,
                base_dtype: "bf16",
                base_quantization: "q8_0",
            },
            config: ArtifactConfigManifest {
                run_config_source_sha256: "07".repeat(32),
                run_config_resolved_sha256: "08".repeat(32),
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
                bundle_path: "eval-bundle".to_string(),
                bundle_sha256: "09".repeat(32),
                bundle_file_count: 5,
                sandbox_image_path: "sandbox-image".to_string(),
                sandbox_image_sha256: "0a".repeat(32),
                sandbox_image_len_bytes: 1024,
                task_yml_sha256: "0b".repeat(32),
                task_yml_len_bytes: 512,
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
        assert!(report.contains(&format!("ferrl commit: {}", "01".repeat(20))));
        assert!(report.contains(&format!(
            "Launch/config hashes: payload={}, file={}, source={}, resolved={}",
            "02".repeat(32),
            "03".repeat(32),
            "07".repeat(32),
            "08".repeat(32)
        )));
        for required in [
            "## 1. Verdict",
            "Raw measurements ns: 10.000000, 11.000000, 12.000000",
            "Command used: `ferrl trimul-baseline --config run.json`",
            "Launch attestation: test-root-1 (ed25519)",
            "Prompt copy: prompt.txt (prompt-hash)",
            "Reward profile: `{\"scheme\":\"trimul_shaped_v1\"",
            "base_quantization=q8_0",
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
