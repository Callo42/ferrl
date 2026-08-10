//! Library-owned discovery and CLI training orchestration.
//!
//! The stable SDK is [`crate::discovery`].  This module is deliberately hidden:
//! its concrete CLI request types exist only because the package binary is a
//! separate Rust crate.  They are input records, not algorithm, plugin, or
//! extension points.  The trusted execution engine itself is private
//! to the library.
//!
//! Downstream callers cannot inject a policy, tokenizer, loader identity, or
//! tensor-parallel capability callback:
//!
//! ```compile_fail
//! use ferrl::orchestration::run_cli_training_with_test_loader;
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_core::Device;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::comm::{Comm, CommError};
use crate::eval::evaluate;
use crate::hf::{resolve_checkpoint_eos, validate_resolved_eos_consensus, CheckpointEosSelection};
use crate::loader::{load_auto_policy_with_identity, AutoPolicy, LoaderOpts, PolicyLoadIdentity};
use crate::policy::{EvalSampling, GenConfig, Policy, TensorParallelPolicy};
use crate::reward::RewardFn;
use crate::sample::Sample;
use crate::telemetry::{summarize, CandidateRecord, CandidateSigner, Metrics, RunDir, RunSummary};
use crate::tensor_parallel::TensorParallelPlan;
use crate::tokenizer::HfTokenizer;
use crate::trainer::{RunStop, TokenizerLike, Trainer, TrainerConfig};

/// Error returned by the hidden concrete CLI adapter.
#[doc(hidden)]
#[derive(Debug)]
pub enum CliOrchestrationError {
    /// A fail-closed operational or contract error.
    Message(String),
    /// The post-run health gate failed after producing its report.
    RunHealth(CliRunHealthReport),
}

impl CliOrchestrationError {
    /// Construct a message-only engine error.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    /// Return the health report when that gate rejected a completed training run.
    #[must_use]
    pub fn health_report(&self) -> Option<&CliRunHealthReport> {
        match self {
            Self::Message(_) => None,
            Self::RunHealth(report) => Some(report),
        }
    }
}

impl std::fmt::Display for CliOrchestrationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(message) => formatter.write_str(message),
            Self::RunHealth(_) => formatter.write_str("run_health policy failed"),
        }
    }
}

impl std::error::Error for CliOrchestrationError {}

/// How an immutable CLI launch is authenticated.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchAuthenticationMode {
    /// Process-local signed-candidate binding.
    #[default]
    LocalEphemeralV1,
    /// Root-attested immutable launch binding.
    ExternalAttestedV1,
}

/// Canonical resolved CLI configuration captured in a launch manifest.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchConfigSnapshot {
    /// Digest of the source config bytes.
    pub source_sha256: String,
    /// Digest of the canonical resolved config bytes.
    pub resolved_sha256: String,
    /// Canonical resolved config value.
    pub resolved: serde_json::Value,
}

/// Launch identity for data- and tensor-parallel execution.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchRunIdentity {
    /// Cross-rank launch group identity.
    pub group_id: String,
    /// Rank-local run directory identity.
    pub run_id: String,
    /// Data-parallel rank.
    pub data_parallel_rank: usize,
    /// Data-parallel world size.
    pub data_parallel_world_size: usize,
    /// Tensor-parallel rank.
    pub tensor_parallel_rank: usize,
    /// Tensor-parallel world size.
    pub tensor_parallel_world_size: usize,
}

/// Loader-derived model identity persisted in a CLI launch.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchModelIdentity {
    /// Loader-derived supported model family.
    pub family: String,
    /// Frozen checkpoint policy identity.
    pub checkpoint_policy_sha256: String,
    /// Loaded tokenizer identity.
    pub tokenizer_sha256: String,
    /// Resolved EOS selection.
    pub resolved_eos_token_id: Option<u32>,
}

/// Exact rendered prompt identity persisted in a CLI launch.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchPromptIdentity {
    /// Immutable prompt file name.
    pub file: String,
    /// Prompt byte digest.
    pub sha256: String,
    /// Prompt byte length.
    pub len_bytes: usize,
}

/// Exact ordered sample identity persisted in a CLI launch.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchSampleIdentity {
    /// SHA-256 of the exact ordered serialized samples executed by the engine.
    pub sha256: String,
    /// Number of samples in the ordered slice.
    pub count: usize,
}

/// Candidate-ledger authentication contract persisted in a CLI launch.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchCandidateLedger {
    /// Immutable ledger file name.
    pub file: String,
    /// Candidate row format version.
    pub format_version: u32,
    /// Candidate row digest domain.
    pub row_digest_domain: String,
    /// Candidate row signature algorithm.
    pub row_signature_algorithm: String,
    /// Launch-local Ed25519 public key.
    pub signing_public_key: String,
}

/// Launch-bound TriMul assets and verifier-isolation evidence.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchVerifierIdentity {
    /// Immutable verifier assets.
    pub assets: crate::trimul::TrimulVerifierIdentity,
    /// Selected isolation evidence.
    pub isolation: crate::VerifierIsolationEvidence,
    /// Digest of the isolation evidence.
    pub isolation_evidence_sha256: String,
    /// Tier-specific timing metric.
    pub timing_metric: String,
    /// Runtime hardening contract name.
    pub runtime_hardening_contract: String,
    /// Captured runtime-control preflight evidence.
    pub runtime_preflight: crate::trimul::TrimulRuntimePreflightEvidence,
    /// Digest of the runtime-control preflight evidence.
    pub runtime_preflight_evidence_sha256: String,
}

/// External attestation envelope for a CLI launch.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchAttestation {
    /// Attestation contract version.
    pub contract_version: u32,
    /// Attestation kind.
    pub kind: String,
    /// Signature algorithm.
    pub algorithm: String,
    /// Trusted attestation key id.
    pub key_id: String,
    /// Attested launch payload digest.
    pub launch_payload_sha256: String,
    /// Lowercase hexadecimal signature.
    pub signature: String,
}

/// Protected launch-attestor request envelope.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchAttestationRequest {
    /// Request contract version.
    pub contract_version: u32,
    /// Request kind.
    pub kind: String,
    /// Signature algorithm.
    pub algorithm: String,
    /// Launch payload digest to attest.
    pub launch_payload_sha256: String,
    /// Canonical launch payload JSON encoded as lowercase hexadecimal bytes.
    pub launch_payload_json_hex: String,
}

/// Trusted launch-attestation key policy.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchTrustPolicy {
    /// Trust-policy contract version.
    pub contract_version: u32,
    /// Trust-policy kind.
    pub kind: String,
    /// Trusted attestation keys.
    pub keys: Vec<LaunchTrustKey>,
}

/// One trusted launch-attestation key.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchTrustKey {
    /// Stable key identifier.
    pub key_id: String,
    /// Signature algorithm.
    pub algorithm: String,
    /// Lowercase hexadecimal Ed25519 public key.
    pub public_key: String,
}

/// Canonical immutable CLI launch payload.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchPayload {
    /// Built-in task name.
    pub task: String,
    /// Embedded Ferrl source commit.
    pub ferrl_commit: String,
    /// Launch authentication mode.
    pub authentication: LaunchAuthenticationMode,
    /// Execution identity.
    pub run: LaunchRunIdentity,
    /// Canonical resolved configuration.
    pub config: LaunchConfigSnapshot,
    /// Loader-derived model identity.
    pub model: LaunchModelIdentity,
    /// Exact rendered prompt identity where applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<LaunchPromptIdentity>,
    /// Exact ordered training sample identity (required by launch v3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_samples: Option<LaunchSampleIdentity>,
    /// Exact ordered task-semantic held-out sample identity (required by launch v3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_samples: Option<LaunchSampleIdentity>,
    /// TriMul verifier identity where applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier: Option<LaunchVerifierIdentity>,
    /// Candidate-ledger contract.
    pub candidate_ledger: LaunchCandidateLedger,
}

/// Canonical immutable CLI launch manifest.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchManifest {
    /// Launch contract version.
    pub contract_version: u32,
    /// Launch manifest kind.
    pub kind: String,
    /// Domain-separated payload digest.
    pub payload_sha256: String,
    /// Bound launch payload.
    pub payload: LaunchPayload,
    /// Optional protected external attestation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation: Option<LaunchAttestation>,
}

/// Narrow attestation primitive used by the concrete CLI engine.
///
/// This is not a phase callback: Ferrl constructs the complete immutable
/// manifest and chooses when it is invoked.  The binary supplies its protected
/// system transport as the only production implementation.
#[doc(hidden)]
pub trait CliLaunchAttestor {
    /// Attest one fully constructed launch manifest.
    fn attest(&self, manifest: &LaunchManifest)
        -> Result<LaunchAttestation, CliOrchestrationError>;
}

/// CLI launch-manifest contract version.
#[doc(hidden)]
pub const LAUNCH_CONTRACT_VERSION: u32 = 3;
/// Previous CLI launch-manifest contract retained for artifact compatibility.
#[doc(hidden)]
pub const LEGACY_LAUNCH_CONTRACT_VERSION: u32 = 2;
/// CLI launch-manifest kind.
#[doc(hidden)]
pub const LAUNCH_KIND: &str = "ferrl.run-launch";
/// Domain for CLI launch payload digests.
#[doc(hidden)]
pub const LAUNCH_PAYLOAD_DOMAIN: &str = "ferrl.run-launch.payload.v3";
/// Previous payload-digest domain retained for launch-v2 artifact ingestion.
#[doc(hidden)]
pub const LEGACY_LAUNCH_PAYLOAD_DOMAIN: &str = "ferrl.run-launch.payload.v2";
/// Candidate-row digest domain committed by CLI launches.
#[doc(hidden)]
pub const CANDIDATE_RECORD_DOMAIN: &str = CandidateRecord::DIGEST_DOMAIN;
/// Launch-attestation contract version.
#[doc(hidden)]
pub const LAUNCH_ATTESTATION_CONTRACT_VERSION: u32 = 1;
/// Launch-attestation kind.
#[doc(hidden)]
pub const LAUNCH_ATTESTATION_KIND: &str = "ferrl.run-launch-attestation";
/// Launch-attestation signature algorithm.
#[doc(hidden)]
pub const LAUNCH_ATTESTATION_ALGORITHM: &str = "ed25519";
/// Domain for launch-attestation signatures.
#[doc(hidden)]
pub const LAUNCH_ATTESTATION_DOMAIN: &str = "ferrl.run-launch-attestation.v1";
/// Protected launch-attestor request kind.
#[doc(hidden)]
pub const LAUNCH_ATTESTATION_REQUEST_KIND: &str = "ferrl.run-launch-attestation-request";
/// Protected launch-attestor trust-policy kind.
#[doc(hidden)]
pub const LAUNCH_TRUST_POLICY_KIND: &str = "ferrl.run-launch-trust-policy";

impl LaunchManifest {
    /// Construct the canonical immutable launch envelope around `payload`.
    pub fn new(payload: LaunchPayload) -> Result<Self, CliOrchestrationError> {
        if payload.training_samples.is_none() || payload.held_out_samples.is_none() {
            return Err(CliOrchestrationError::msg(
                "launch v3 payload requires ordered training and held-out sample identities",
            ));
        }
        let payload_bytes = serde_json::to_vec(&payload).map_err(|error| {
            CliOrchestrationError::msg(format!("serialize launch payload: {error}"))
        })?;
        Ok(Self {
            contract_version: LAUNCH_CONTRACT_VERSION,
            kind: LAUNCH_KIND.to_owned(),
            payload_sha256: domain_sha256(LAUNCH_PAYLOAD_DOMAIN, &[&payload_bytes]),
            payload,
            attestation: None,
        })
    }

    /// Attach one protected external attestation to a fresh manifest.
    pub fn attest(
        mut self,
        attestor: &dyn CliLaunchAttestor,
    ) -> Result<Self, CliOrchestrationError> {
        if self.attestation.is_some() {
            return Err(CliOrchestrationError::msg(
                "launch manifest is already attested",
            ));
        }
        self.attestation = Some(attestor.attest(&self)?);
        Ok(self)
    }

    /// Serialize the exact production launch encoding.
    pub fn to_pretty_bytes(&self) -> Result<Vec<u8>, CliOrchestrationError> {
        serde_json::to_vec_pretty(self).map_err(|error| {
            CliOrchestrationError::msg(format!("serialize launch manifest: {error}"))
        })
    }
}

/// Error returned by the private concrete execution engine.
#[derive(Debug)]
pub(crate) enum EngineError {
    /// Configuration, topology, or invariant failure.
    Configuration(String),
    /// Production policy/tokenizer loading failure.
    ModelLoad(Box<crate::loader::LoaderError>),
    /// Checkpoint/tokenizer EOS resolution failure.
    GenerationEnd(Box<crate::hf::HfError>),
    /// Immutable launch or run-directory failure.
    Launch(Box<crate::telemetry::TelemetryError>),
    /// Training or trainer construction failure.
    Training(Box<crate::trainer::TrainerError>),
    /// Held-out evaluation failure.
    Evaluation(Box<crate::eval::EvalError>),
    /// Preemption checkpoint scan failure.
    PreemptionCheckpointScan(Box<crate::checkpoint::CheckpointError>),
    /// Preemption checkpoint invariant failure.
    PreemptionCheckpoint(String),
    /// Candidate ledger I/O failure.
    CandidateIo {
        /// Candidate ledger path.
        path: PathBuf,
        /// Underlying I/O source.
        source: std::io::Error,
    },
    /// Candidate JSON decoding failure.
    CandidateJson {
        /// Candidate ledger path.
        path: PathBuf,
        /// One-based row number.
        line: usize,
        /// Underlying JSON source.
        source: serde_json::Error,
    },
    /// Candidate provenance or coordinate failure.
    InvalidCandidateEvidence(String),
    /// Held-out report read-back failure.
    HeldOutReportIo {
        /// Published report path.
        path: PathBuf,
        /// Underlying I/O source.
        source: std::io::Error,
    },
    /// Serialization failure with its contract label.
    Serialization {
        /// Contract component being serialized.
        kind: &'static str,
        /// Underlying JSON source.
        source: serde_json::Error,
    },
    /// A configurable CLI health policy rejected the run.
    Health(CliRunHealthReport),
    /// Message-only failure used by the CLI adapter and test controls.
    Message(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configuration(message)
            | Self::Message(message)
            | Self::PreemptionCheckpoint(message)
            | Self::InvalidCandidateEvidence(message) => formatter.write_str(message),
            Self::ModelLoad(error) => write!(formatter, "model load failed: {error}"),
            Self::GenerationEnd(error) => {
                write!(formatter, "generation-end resolution failed: {error}")
            }
            Self::Launch(error) => write!(formatter, "launch setup failed: {error}"),
            Self::Training(error) => write!(formatter, "training failed: {error}"),
            Self::Evaluation(error) => write!(formatter, "held-out evaluation failed: {error}"),
            Self::PreemptionCheckpointScan(error) => {
                write!(formatter, "preemption checkpoint scan failed: {error}")
            }
            Self::CandidateIo { path, source } => {
                write!(
                    formatter,
                    "read candidate ledger {}: {source}",
                    path.display()
                )
            }
            Self::CandidateJson { path, line, source } => write!(
                formatter,
                "invalid candidate JSON at {} line {line}: {source}",
                path.display()
            ),
            Self::HeldOutReportIo { path, source } => {
                write!(
                    formatter,
                    "read held-out report {}: {source}",
                    path.display()
                )
            }
            Self::Serialization { kind, source } => {
                write!(formatter, "serialize {kind}: {source}")
            }
            Self::Health(_) => formatter.write_str("run_health policy failed"),
        }
    }
}

/// A closed EOS choice consumed by the concrete engine.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum EngineEosSelection {
    /// Resolve one checkpoint-declared EOS id.
    CheckpointDefault,
    /// Validate and use an explicit EOS id.
    Explicit(u32),
    /// Disable EOS stopping.
    Disabled,
}

/// Serialize an execution dataset and immediately deserialize it back before use.
pub(crate) fn exact_execution_samples<T>(
    samples: &[Sample<T>],
    _kind: &'static str,
) -> Result<(Vec<Sample<T>>, Vec<u8>), serde_json::Error>
where
    T: Serialize + DeserializeOwned,
{
    let bytes = serde_json::to_vec(samples)?;
    let reconstructed = serde_json::from_slice(&bytes)?;
    Ok((reconstructed, bytes))
}

/// Reject a prompt which would silently encode to zero input tokens.
pub(crate) fn preflight_prompt_tokenization<T>(
    samples: &[Sample<T>],
    kind: &'static str,
    tokenizer: &dyn TokenizerLike,
) -> Result<(), String> {
    for (index, sample) in samples.iter().enumerate() {
        if tokenizer.encode(&sample.prompt).is_empty() {
            return Err(format!(
                "{kind} prompt at index {index} encoded to zero tokens with the loaded tokenizer"
            ));
        }
    }
    Ok(())
}

/// Hidden normalized EOS selection for the concrete CLI engine.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub enum CliEosSelection {
    /// Resolve exactly one checkpoint-declared EOS id.
    CheckpointDefault,
    /// Validate and use this explicit EOS id.
    Explicit(u32),
    /// Deliberately disable EOS stopping.
    Disabled,
}

/// Hidden, already-validated CLI execution topology and live communicator.
#[doc(hidden)]
pub enum CliExecution {
    /// Ordinary world-one execution.
    WorldOne,
    /// Data-parallel execution over this communicator.
    DataParallel(Box<dyn Comm>),
    /// Tensor-parallel execution over this communicator and plan.
    TensorParallel {
        /// Validated local tensor-parallel plan.
        plan: TensorParallelPlan,
        /// Live tensor-parallel communicator.
        comm: Box<dyn Comm>,
    },
}

impl CliExecution {
    fn comm(&self) -> Option<&dyn Comm> {
        match self {
            Self::WorldOne => None,
            Self::DataParallel(comm) | Self::TensorParallel { comm, .. } => Some(comm.as_ref()),
        }
    }
}

/// Device selected by the parsed CLI configuration.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliDeviceSelection {
    /// CPU execution.
    Cpu,
    /// CUDA device zero.
    Cuda,
}

/// Live distributed transport opened by the binary adapter.
#[doc(hidden)]
pub struct CliLaunchRuntime {
    /// Rank-local CUDA device owned by the NCCL transport.
    pub device: Device,
    /// Live launch communicator.
    pub comm: Box<dyn Comm>,
}

/// Parsed built-in task inputs; the library owns their authoritative construction.
#[doc(hidden)]
pub enum CliBuiltinTask {
    /// Procedural Countdown data.
    Countdown {
        train_n: usize,
        eval_n: usize,
        seed: u64,
    },
    /// JSONL Math data.
    Math {
        path: PathBuf,
        eval_n: usize,
        seed: u64,
    },
    /// TriMul prompt, reward, and verifier setup.
    Trimul(Box<CliTrimulTask>),
}

impl CliBuiltinTask {
    fn name(&self) -> &'static str {
        match self {
            Self::Countdown { .. } => "countdown",
            Self::Math { .. } => "math",
            Self::Trimul(_) => "trimul",
        }
    }

    fn data_seed(&self) -> u64 {
        match self {
            Self::Countdown { seed, .. } | Self::Math { seed, .. } => *seed,
            Self::Trimul(task) => task.data_seed,
        }
    }

    fn trimul_held_out_secret_seed(&self) -> Option<u64> {
        match self {
            Self::Trimul(task) => task.held_out_secret_seed,
            Self::Countdown { .. } | Self::Math { .. } => None,
        }
    }
}

/// Parsed TriMul baseline pin.
#[doc(hidden)]
pub struct CliTrimulBaseline {
    pub ns: f64,
    pub gpu: String,
    pub metric: String,
    pub isolation_tier: crate::VerifierIsolationTier,
    pub isolation_evidence_sha256: String,
}

/// Parsed TriMul task setup; construction and preflight remain library-owned.
#[doc(hidden)]
pub struct CliTrimulTask {
    pub prompt_path: PathBuf,
    pub submission_extract_mode: crate::trimul::SubmissionExtractMode,
    pub image: PathBuf,
    pub eval_dir: PathBuf,
    pub scratch_root: PathBuf,
    pub verifier_isolation_tier: crate::VerifierIsolationTier,
    pub verifier_apptainer_bin: Option<PathBuf>,
    pub verifier_executor_socket: Option<PathBuf>,
    pub scratch_max_bytes: u64,
    pub secret_seed: u64,
    pub held_out_secret_seed: Option<u64>,
    pub wall_secs: u64,
    pub verifier_cuda_visible_devices: Option<String>,
    pub verifier_cuda_device_pool: Vec<String>,
    pub verifier_parallelism: usize,
    pub verifier_max_procs: u64,
    pub baseline: Option<CliTrimulBaseline>,
    pub reward_profile: crate::trimul::TrimulRewardProfile,
    pub train_n: usize,
    pub eval_n: usize,
    pub data_seed: u64,
}

/// Complete parsed `ferrl train` adapter input. The library owns setup, identity,
/// task construction, request assembly, and execution.
#[doc(hidden)]
pub struct CliTrainSetup {
    pub task: CliBuiltinTask,
    pub ferrl_commit: String,
    pub authentication: LaunchAuthenticationMode,
    pub launch_config: LaunchConfigSnapshot,
    pub config_consensus_digest: [u8; 32],
    pub model_dir: PathBuf,
    pub output_root: PathBuf,
    pub device: CliDeviceSelection,
    pub loader_opts: LoaderOpts,
    pub activation_checkpointing: bool,
    pub eos_selection: CliEosSelection,
    pub trainer_config: TrainerConfig,
    pub data_parallel: bool,
    pub tensor_parallel_plan: TensorParallelPlan,
    pub health_policy: CliRunHealthPolicy,
    pub health_policy_is_default: bool,
}

/// Immutable CLI launch inputs adapted from parsed command configuration.
#[doc(hidden)]
pub struct CliLaunchInput {
    /// Selected built-in task.
    pub task: String,
    /// Embedded clean Ferrl source commit.
    pub ferrl_commit: String,
    /// Selected launch authentication boundary.
    pub authentication: LaunchAuthenticationMode,
    /// Synchronized rank-local execution identity.
    pub run: LaunchRunIdentity,
    /// Canonical launch-bound configuration snapshot.
    pub config: LaunchConfigSnapshot,
    /// Root containing rank-local run directories.
    pub output_root: PathBuf,
}

/// Concrete CLI engine input.  It is intentionally not a stable SDK surface.
#[doc(hidden)]
pub struct CliTrainingRequest<'a, R>
where
    R: RewardFn,
{
    /// Immutable launch inputs.
    pub launch: CliLaunchInput,
    /// Loader-owned checkpoint directory.
    pub model_dir: &'a Path,
    /// Opened execution device.
    pub device: &'a Device,
    /// Validated loader controls.
    pub loader_opts: LoaderOpts,
    /// Configured activation-checkpointing state for launch telemetry.
    pub activation_checkpointing: bool,
    /// EOS selection to resolve after tokenizer loading.
    pub eos_selection: CliEosSelection,
    /// Validated trainer controls before EOS resolution.
    pub trainer_config: TrainerConfig,
    /// Typed task training inputs.
    pub training_samples: &'a [Sample<R::Target>],
    /// Typed task-semantic held-out inputs.
    pub evaluation_samples: &'a [Sample<R::Target>],
    /// Typed search reward.
    pub reward: &'a R,
    /// Typed held-out reward.
    pub evaluation_reward: &'a R,
    /// Exact rendered prompt bytes where the task owns one.
    pub rendered_prompt_bytes: Option<&'a [u8]>,
    /// Captured TriMul verifier assets where applicable.
    pub verifier_assets: Option<&'a crate::trimul::TrimulVerifierAssets>,
    /// Captured TriMul verifier identity where applicable.
    pub verifier_identity: Option<LaunchVerifierIdentity>,
    /// Active world-one, DP, or TP execution topology.
    pub execution: CliExecution,
    /// Normalized run-health policy.
    pub health_policy: CliRunHealthPolicy,
    /// Whether the original run-health policy was the wire default.
    pub health_policy_is_default: bool,
    /// Dataset split seed bound into a held-out report.
    pub data_seed: u64,
    /// Distinct TriMul held-out secret seed bound into a held-out report.
    pub trimul_held_out_secret_seed: Option<u64>,
}

/// Completed concrete CLI run information for binary presentation.
#[doc(hidden)]
#[derive(Debug)]
pub struct CliCompletedRun {
    run_dir: PathBuf,
    health_report: Option<CliRunHealthReport>,
    presentation_rank: bool,
}

impl CliCompletedRun {
    /// Completed run directory.
    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Non-default health report produced by the post-run phase.
    #[must_use]
    pub fn health_report(&self) -> Option<&CliRunHealthReport> {
        self.health_report.as_ref()
    }

    /// Whether this rank owns CLI presentation for the completed run.
    #[must_use]
    pub fn should_present(&self) -> bool {
        self.presentation_rank
    }
}

/// Preempted concrete CLI run information for binary presentation.
#[doc(hidden)]
#[derive(Debug)]
pub struct CliPreemptedRun {
    run_dir: PathBuf,
    presentation_rank: bool,
}

impl CliPreemptedRun {
    /// Preempted run directory.
    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Whether this rank owns CLI presentation for the preempted run.
    #[must_use]
    pub fn should_present(&self) -> bool {
        self.presentation_rank
    }
}

/// Terminal concrete CLI engine result.
#[doc(hidden)]
#[derive(Debug)]
pub enum CliRunOutcome {
    /// Training, post-run health, held-out evaluation, and candidate selection completed.
    Completed(CliCompletedRun),
    /// Training preempted before post-training phases.
    Preempted(CliPreemptedRun),
}

/// Closed SDK inputs adapted into the private concrete engine.
pub(crate) struct DiscoveryLaunchInput<'a> {
    pub(crate) task: &'a crate::discovery::TaskIdentity,
    pub(crate) metric_contract: crate::discovery::MetricContract,
    pub(crate) ferrl_source: crate::discovery::BuildSourceIdentity,
    pub(crate) execution_device: crate::discovery::ExecutionDevice,
    pub(crate) runs_root: &'a Path,
    pub(crate) steps: u64,
    pub(crate) group_size: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) eval_group_size: usize,
    pub(crate) temperature: f64,
    pub(crate) learning_rate: f64,
    pub(crate) seed: u64,
    pub(crate) preemption_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
}

/// Closed SDK training inputs.  The engine, not the adapter, loads the model.
pub(crate) struct DiscoveryTrainingRequest<'a, R>
where
    R: RewardFn,
{
    pub(crate) model_dir: &'a Path,
    pub(crate) device: &'a Device,
    pub(crate) loader_opts: LoaderOpts,
    pub(crate) eos_selection: EngineEosSelection,
    pub(crate) trainer_config: TrainerConfig,
    pub(crate) training_samples: &'a [Sample<R::Target>],
    pub(crate) evaluation_samples: &'a [Sample<R::Target>],
    pub(crate) reward: &'a R,
    pub(crate) evaluation_reward: &'a R,
    pub(crate) launch: DiscoveryLaunchInput<'a>,
}

struct CliEngineLaunch<'a> {
    launch: CliLaunchInput,
    attestor: Option<&'a dyn CliLaunchAttestor>,
    health_policy: CliRunHealthPolicy,
    health_policy_is_default: bool,
    activation_checkpointing: bool,
    data_seed: u64,
    trimul_held_out_secret_seed: Option<u64>,
}

enum EngineMode<'a> {
    Discovery(DiscoveryLaunchInput<'a>),
    Cli(CliEngineLaunch<'a>),
}

struct EnginePlan<'a, R>
where
    R: RewardFn,
{
    mode: EngineMode<'a>,
    model_dir: &'a Path,
    device: &'a Device,
    loader_opts: LoaderOpts,
    eos_selection: EngineEosSelection,
    trainer_config: TrainerConfig,
    training_samples: &'a [Sample<R::Target>],
    evaluation_samples: &'a [Sample<R::Target>],
    reward: &'a R,
    evaluation_reward: &'a R,
    rendered_prompt_bytes: Option<&'a [u8]>,
    verifier_assets: Option<&'a crate::trimul::TrimulVerifierAssets>,
    verifier_identity: Option<LaunchVerifierIdentity>,
    execution: CliExecution,
}

impl<'a, R> EnginePlan<'a, R>
where
    R: RewardFn,
{
    fn from_discovery(request: DiscoveryTrainingRequest<'a, R>) -> Self {
        Self {
            mode: EngineMode::Discovery(request.launch),
            model_dir: request.model_dir,
            device: request.device,
            loader_opts: request.loader_opts,
            eos_selection: request.eos_selection,
            trainer_config: request.trainer_config,
            training_samples: request.training_samples,
            evaluation_samples: request.evaluation_samples,
            reward: request.reward,
            evaluation_reward: request.evaluation_reward,
            rendered_prompt_bytes: None,
            verifier_assets: None,
            verifier_identity: None,
            execution: CliExecution::WorldOne,
        }
    }

    fn from_cli(
        request: CliTrainingRequest<'a, R>,
        attestor: Option<&'a dyn CliLaunchAttestor>,
    ) -> Self {
        Self {
            mode: EngineMode::Cli(CliEngineLaunch {
                launch: request.launch,
                attestor,
                health_policy: request.health_policy,
                health_policy_is_default: request.health_policy_is_default,
                activation_checkpointing: request.activation_checkpointing,
                data_seed: request.data_seed,
                trimul_held_out_secret_seed: request.trimul_held_out_secret_seed,
            }),
            model_dir: request.model_dir,
            device: request.device,
            loader_opts: request.loader_opts,
            eos_selection: match request.eos_selection {
                CliEosSelection::CheckpointDefault => EngineEosSelection::CheckpointDefault,
                CliEosSelection::Explicit(id) => EngineEosSelection::Explicit(id),
                CliEosSelection::Disabled => EngineEosSelection::Disabled,
            },
            trainer_config: request.trainer_config,
            training_samples: request.training_samples,
            evaluation_samples: request.evaluation_samples,
            reward: request.reward,
            evaluation_reward: request.evaluation_reward,
            rendered_prompt_bytes: request.rendered_prompt_bytes,
            verifier_assets: request.verifier_assets,
            verifier_identity: request.verifier_identity,
            execution: request.execution,
        }
    }
}

/// Evaluation bytes and typed report owned by the engine.
pub(crate) struct EngineEvaluation {
    pub(crate) report: crate::eval::EvalReport,
    pub(crate) bytes: Vec<u8>,
}

/// Exact authenticated candidate view returned by the engine.
#[derive(Debug)]
pub(crate) struct AuthenticatedCandidate {
    pub(crate) record: CandidateRecord,
    pub(crate) exact_row_bytes: Vec<u8>,
    pub(crate) provenance_sha256: String,
}

/// Completed result from the shared engine.
pub(crate) struct EngineCompletedRun {
    pub(crate) run: RunDir,
    pub(crate) launch_bytes: Vec<u8>,
    pub(crate) launch_sha256: String,
    pub(crate) signing_public_key: String,
    pub(crate) model_identity: Option<crate::discovery::ModelIdentity>,
    pub(crate) source_identity: Option<crate::discovery::BuildSourceIdentity>,
    pub(crate) metric_contract: Option<crate::discovery::MetricContract>,
    pub(crate) evaluation: Option<EngineEvaluation>,
    pub(crate) candidates: Vec<AuthenticatedCandidate>,
    pub(crate) health_report: Option<CliRunHealthReport>,
    pub(crate) presentation_rank: bool,
}

/// Preempted result from the shared engine.
pub(crate) struct EnginePreemptedRun {
    pub(crate) run_dir: PathBuf,
    pub(crate) completed_steps: Option<u64>,
    pub(crate) checkpoint_path: Option<PathBuf>,
    pub(crate) presentation_rank: bool,
}

/// Terminal result from the shared concrete engine.
pub(crate) enum EngineOutcome {
    /// Training completed and all engine-owned evidence phases ran.
    Completed(Box<EngineCompletedRun>),
    /// Training stopped before health, evaluation, and candidate phases.
    Preempted(EnginePreemptedRun),
}

/// Run production CLI training through the one library-owned engine.
#[doc(hidden)]
pub fn run_cli_training<'a, R>(
    request: CliTrainingRequest<'a, R>,
    attestor: Option<&'a dyn CliLaunchAttestor>,
) -> Result<CliRunOutcome, CliOrchestrationError>
where
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
{
    let plan = EnginePlan::from_cli(request, attestor);
    match run_production_engine(plan).map_err(cli_engine_error)? {
        EngineOutcome::Completed(completed) => Ok(CliRunOutcome::Completed(CliCompletedRun {
            run_dir: completed.run.root().to_path_buf(),
            health_report: completed.health_report,
            presentation_rank: completed.presentation_rank,
        })),
        EngineOutcome::Preempted(preempted) => Ok(CliRunOutcome::Preempted(CliPreemptedRun {
            run_dir: preempted.run_dir,
            presentation_rank: preempted.presentation_rank,
        })),
    }
}

/// Run the complete production `ferrl train` path after binary parsing.
#[doc(hidden)]
pub fn run_cli_train_setup(
    setup: &CliTrainSetup,
    runtime: Option<CliLaunchRuntime>,
    attestor: Option<&dyn CliLaunchAttestor>,
) -> Result<CliRunOutcome, CliOrchestrationError> {
    let launch_comm = runtime.as_ref().map(|runtime| runtime.comm.as_ref());
    validate_engine_value_consensus(
        "run config outside tensor_parallel.rank",
        &setup.config_consensus_digest,
        launch_comm,
    )
    .map_err(cli_engine_error)?;
    let ferrl_commit = coordinate_engine_result(
        launch_comm,
        "training commit validation",
        validate_cli_git_commit(&setup.ferrl_commit)
            .map_err(|error| EngineError::Configuration(error.to_string())),
    )
    .map_err(cli_engine_error)?;
    validate_engine_value_consensus("training commit", ferrl_commit.as_bytes(), launch_comm)
        .map_err(cli_engine_error)?;
    let run = synchronized_cli_run_identity(setup, launch_comm)?;
    let data_parallel_world = if setup.data_parallel {
        launch_comm
            .ok_or_else(|| {
                CliOrchestrationError::msg(
                    "distributed execution has no live communicator after launch validation",
                )
            })?
            .world_size()
    } else {
        1
    };
    coordinate_engine_result(
        launch_comm,
        "trainer reward-group validation",
        setup
            .trainer_config
            .validate_reward_group_world(data_parallel_world)
            .map_err(|error| EngineError::Configuration(error.to_string())),
    )
    .map_err(cli_engine_error)?;
    let topology_check = match (
        setup.data_parallel,
        setup.tensor_parallel_plan.is_sharded(),
        runtime.is_some(),
    ) {
        (true, false, true) | (false, true, true) | (false, false, false) => Ok(()),
        (true, true, _) => Err(EngineError::Configuration(
            "combined data-parallel and tensor-parallel execution is unsupported".into(),
        )),
        (_, _, true) => Err(EngineError::Configuration(
            "world-one execution received an unexpected distributed launch runtime".into(),
        )),
        (_, _, false) => Err(EngineError::Configuration(
            "distributed or tensor-parallel execution requires a live launch runtime".into(),
        )),
    };
    coordinate_engine_result(launch_comm, "CLI execution topology", topology_check)
        .map_err(cli_engine_error)?;
    let device = coordinate_engine_result(
        launch_comm,
        "CLI device setup",
        prepare_cli_device(setup.device, runtime.as_ref())
            .map_err(|error| EngineError::Message(error.to_string())),
    )
    .map_err(cli_engine_error)?;
    let execution = match (
        setup.data_parallel,
        setup.tensor_parallel_plan.is_sharded(),
        runtime,
    ) {
        (true, false, Some(runtime)) => CliExecution::DataParallel(runtime.comm),
        (false, true, Some(runtime)) => CliExecution::TensorParallel {
            plan: setup.tensor_parallel_plan,
            comm: runtime.comm,
        },
        (false, false, None) => CliExecution::WorldOne,
        (true, true, _) => {
            return Err(CliOrchestrationError::msg(
                "combined data-parallel and tensor-parallel execution is unsupported",
            ));
        }
        (_, _, Some(_)) => {
            return Err(CliOrchestrationError::msg(
                "world-one execution received an unexpected distributed launch runtime",
            ));
        }
        (_, _, None) => {
            return Err(CliOrchestrationError::msg(
                "distributed or tensor-parallel execution requires a live launch runtime",
            ));
        }
    };
    let launch = CliLaunchInput {
        task: setup.task.name().to_owned(),
        ferrl_commit,
        authentication: setup.authentication,
        run,
        config: setup.launch_config.clone(),
        output_root: setup.output_root.clone(),
    };
    match &setup.task {
        CliBuiltinTask::Countdown {
            train_n,
            eval_n,
            seed,
        } => {
            let local = (|| {
                let config = crate::countdown::CountdownConfig::default();
                let count = train_n.checked_add(*eval_n).ok_or_else(|| {
                    EngineError::Configuration("countdown dataset size overflowed usize".into())
                })?;
                let samples = crate::countdown::generate_dataset(*seed, count, &config)
                    .into_iter()
                    .map(|problem| Sample::new(crate::countdown::build_prompt(&problem), problem))
                    .collect::<Vec<_>>();
                Ok(crate::data::train_eval_split_by_key(
                    samples,
                    *eval_n,
                    *seed,
                    |sample| sample.target.split_key(),
                ))
            })();
            let (train, eval) =
                coordinate_engine_result(execution.comm(), "Countdown task setup", local)
                    .map_err(cli_engine_error)?;
            let reward = crate::countdown::CountdownReward::default();
            run_cli_built_task(
                setup, &device, &train, &eval, &reward, &reward, None, None, None, launch,
                execution, attestor,
            )
        }
        CliBuiltinTask::Math { path, eval_n, seed } => {
            let local = crate::data::read_jsonl::<crate::math::MathProblem, _>(path)
                .map(|samples| {
                    crate::data::train_eval_split_by_key(
                        samples,
                        *eval_n,
                        *seed,
                        crate::math::math_split_key,
                    )
                })
                .map_err(|error| EngineError::Message(error.to_string()));
            let (train, eval) =
                coordinate_engine_result(execution.comm(), "Math task setup", local)
                    .map_err(cli_engine_error)?;
            let reward = crate::math::MathReward::default();
            run_cli_built_task(
                setup, &device, &train, &eval, &reward, &reward, None, None, None, launch,
                execution, attestor,
            )
        }
        CliBuiltinTask::Trimul(task) => {
            let local = (|| {
                let prompt_bytes = fs::read(&task.prompt_path).map_err(|error| {
                    EngineError::Message(format!("read {}: {error}", task.prompt_path.display()))
                })?;
                let prompt = std::str::from_utf8(&prompt_bytes).map_err(|error| {
                    EngineError::Configuration(format!("trimul prompt is not valid UTF-8: {error}"))
                })?;
                if prompt.is_empty() {
                    return Err(EngineError::Configuration("trimul prompt is empty".into()));
                }
                let train = std::iter::repeat_with(|| Sample::new(prompt.to_owned(), ()))
                    .take(task.train_n)
                    .collect::<Vec<_>>();
                let eval = std::iter::repeat_with(|| Sample::new(prompt.to_owned(), ()))
                    .take(task.eval_n)
                    .collect::<Vec<_>>();
                let assets = crate::trimul::TrimulVerifierAssets::capture(
                    &task.image,
                    &task.eval_dir,
                    &task.scratch_root,
                )
                .map_err(|error| EngineError::Message(error.to_string()))?;
                let reward = build_cli_trimul_reward(task, assets.clone(), task.secret_seed, true)
                    .map_err(|error| EngineError::Message(error.to_string()))?;
                let eval_reward = if eval.is_empty() {
                    None
                } else {
                    let seed = task.held_out_secret_seed.ok_or_else(|| {
                        EngineError::Configuration(
                            "TriMul held-out eval requires trimul.held_out_secret_seed".into(),
                        )
                    })?;
                    Some(
                        build_cli_trimul_reward(task, assets.clone(), seed, false)
                            .map_err(|error| EngineError::Message(error.to_string()))?,
                    )
                };
                let verifier_identity = cli_launch_verifier_identity(&reward, &assets)
                    .map_err(|error| EngineError::Message(error.to_string()))?;
                Ok((
                    prompt_bytes,
                    train,
                    eval,
                    assets,
                    reward,
                    eval_reward,
                    verifier_identity,
                ))
            })();
            let (prompt_bytes, train, eval, assets, reward, eval_reward, verifier_identity) =
                coordinate_engine_result(execution.comm(), "TriMul task setup", local)
                    .map_err(cli_engine_error)?;
            run_cli_built_task(
                setup,
                &device,
                &train,
                &eval,
                &reward,
                eval_reward.as_ref().unwrap_or(&reward),
                Some(&prompt_bytes),
                Some(&assets),
                Some(verifier_identity),
                launch,
                execution,
                attestor,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_cli_built_task<R: RewardFn>(
    setup: &CliTrainSetup,
    device: &Device,
    train: &[Sample<R::Target>],
    eval: &[Sample<R::Target>],
    reward: &R,
    evaluation_reward: &R,
    rendered_prompt_bytes: Option<&[u8]>,
    verifier_assets: Option<&crate::trimul::TrimulVerifierAssets>,
    verifier_identity: Option<LaunchVerifierIdentity>,
    launch: CliLaunchInput,
    execution: CliExecution,
    attestor: Option<&dyn CliLaunchAttestor>,
) -> Result<CliRunOutcome, CliOrchestrationError>
where
    R::Target: Serialize + DeserializeOwned,
{
    run_cli_training(
        CliTrainingRequest {
            launch,
            model_dir: &setup.model_dir,
            device,
            loader_opts: setup.loader_opts.clone(),
            activation_checkpointing: setup.activation_checkpointing,
            eos_selection: setup.eos_selection,
            trainer_config: setup.trainer_config.clone(),
            training_samples: train,
            evaluation_samples: eval,
            reward,
            evaluation_reward,
            rendered_prompt_bytes,
            verifier_assets,
            verifier_identity,
            execution,
            health_policy: setup.health_policy.clone(),
            health_policy_is_default: setup.health_policy_is_default,
            data_seed: setup.task.data_seed(),
            trimul_held_out_secret_seed: setup.task.trimul_held_out_secret_seed(),
        },
        attestor,
    )
}

fn validate_cli_git_commit(value: &str) -> Result<String, CliOrchestrationError> {
    let valid_len = matches!(value.len(), 40 | 64);
    let valid_hex = value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid_len && valid_hex {
        Ok(value.to_owned())
    } else {
        Err(CliOrchestrationError::msg(
            "git commit must be a full 40- or 64-character lowercase SHA",
        ))
    }
}

fn synchronized_cli_run_identity(
    setup: &CliTrainSetup,
    comm: Option<&dyn Comm>,
) -> Result<LaunchRunIdentity, CliOrchestrationError> {
    let local_stamp = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .map_err(|error| {
                CliOrchestrationError::msg(format!("system clock precedes Unix epoch: {error}"))
            })
    };
    let stamp = match comm.filter(|comm| comm.world_size() > 1) {
        Some(comm) => {
            let local = if comm.rank() == 0 {
                local_stamp()
            } else {
                Ok(0)
            };
            let local = coordinate_engine_result(
                Some(comm),
                "run timestamp",
                local.map_err(|error| EngineError::Message(error.to_string())),
            )
            .map_err(cli_engine_error)?;
            let reduced = comm
                .all_reduce_scalar_sum(local as f64)
                .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
            if !reduced.is_finite()
                || reduced < 0.0
                || reduced.fract() != 0.0
                || reduced > (1_u64 << 53) as f64
            {
                return Err(CliOrchestrationError::msg(format!(
                    "distributed run timestamp is not an exact u64: {reduced:?}"
                )));
            }
            reduced as u64
        }
        None => local_stamp()?,
    };
    let group_id = format!("{}-{stamp}", setup.task.name());
    let (data_parallel_rank, data_parallel_world_size) = if setup.data_parallel {
        let comm = comm.ok_or_else(|| {
            CliOrchestrationError::msg("distributed run identity requires a live communicator")
        })?;
        (comm.rank(), comm.world_size())
    } else {
        (0, 1)
    };
    let run_id = if setup.data_parallel {
        format!("{group_id}-rank{data_parallel_rank}")
    } else if setup.tensor_parallel_plan.is_sharded() {
        format!("{group_id}-rank{}", setup.tensor_parallel_plan.rank())
    } else {
        group_id.clone()
    };
    Ok(LaunchRunIdentity {
        group_id,
        run_id,
        data_parallel_rank,
        data_parallel_world_size,
        tensor_parallel_rank: setup.tensor_parallel_plan.rank(),
        tensor_parallel_world_size: setup.tensor_parallel_plan.world_size(),
    })
}

fn prepare_cli_device(
    selected: CliDeviceSelection,
    runtime: Option<&CliLaunchRuntime>,
) -> Result<Device, CliOrchestrationError> {
    if let Some(runtime) = runtime {
        if selected != CliDeviceSelection::Cuda {
            return Err(CliOrchestrationError::msg(
                "distributed or tensor_parallel execution requires device = \"cuda\"",
            ));
        }
        let device = runtime.device.clone();
        if let Some(warning) = crate::check_driver_compat(&device).warning() {
            tracing::warn!("{warning}");
        }
        crate::guard_first_kernel(&device)
            .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
        return Ok(device);
    }
    match selected {
        CliDeviceSelection::Cpu => Ok(Device::Cpu),
        CliDeviceSelection::Cuda => open_cli_cuda(),
    }
}

#[cfg(feature = "cuda")]
fn open_cli_cuda() -> Result<Device, CliOrchestrationError> {
    let device =
        Device::new_cuda(0).map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
    if let Some(warning) = crate::check_driver_compat(&device).warning() {
        tracing::warn!("{warning}");
    }
    crate::guard_first_kernel(&device)
        .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
    Ok(device)
}

#[cfg(not(feature = "cuda"))]
fn open_cli_cuda() -> Result<Device, CliOrchestrationError> {
    Err(CliOrchestrationError::msg(
        "device \"cuda\" requires building ferrl with --features cuda; use device \"cpu\" otherwise",
    ))
}

#[allow(clippy::cognitive_complexity)] // one fail-closed TriMul reward/preflight transaction
fn build_cli_trimul_reward(
    task: &CliTrimulTask,
    assets: crate::trimul::TrimulVerifierAssets,
    secret_seed: u64,
    apply_baseline: bool,
) -> Result<crate::trimul::TrimulReward, CliOrchestrationError> {
    let (tests, benches) = crate::trimul::parse_task_yml(assets.task_yml())
        .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
    let wall = Duration::from_secs(if task.wall_secs == 0 {
        600
    } else {
        task.wall_secs
    });
    let reward = crate::trimul::TrimulReward::new(assets, &task.scratch_root)
        .with_cases(tests, benches)
        .with_secret_seed(secret_seed)
        .with_wall(wall);
    let reward = match task.verifier_isolation_tier {
        crate::VerifierIsolationTier::SameUidApptainerV1 => reward.with_same_uid_apptainer(
            task.scratch_root.join(".ferrl-verifier"),
            task.verifier_apptainer_bin
                .as_deref()
                .unwrap_or_else(|| Path::new("/usr/bin/apptainer")),
        ),
        crate::VerifierIsolationTier::DedicatedUidServiceV1 => reward
            .with_verifier_executor_socket(
                task.verifier_executor_socket
                    .as_deref()
                    .unwrap_or_else(|| Path::new(crate::DEFAULT_VERIFIER_EXECUTOR_SOCKET)),
            ),
    };
    let mut reward = reward
        .with_reward_profile(task.reward_profile)
        .map_err(CliOrchestrationError::msg)?
        .with_submission_extract_mode(task.submission_extract_mode);
    if let Some(devices) = &task.verifier_cuda_visible_devices {
        reward = reward.with_verifier_cuda_visible_devices(devices.clone());
    }
    if !task.verifier_cuda_device_pool.is_empty() {
        reward = reward.with_verifier_cuda_device_pool(task.verifier_cuda_device_pool.clone());
    }
    if task.verifier_parallelism != 0 {
        reward = reward.with_verifier_parallelism(task.verifier_parallelism);
    }
    if task.verifier_max_procs != 0 {
        reward = reward.with_verifier_max_procs(task.verifier_max_procs);
    }
    if task.scratch_max_bytes != 0 {
        reward = reward.with_scratch_max_bytes(task.scratch_max_bytes);
    }
    if apply_baseline {
        if let Some(baseline) = &task.baseline {
            let expected = crate::trimul::timing_metric_for_tier(task.verifier_isolation_tier);
            if baseline.metric != expected {
                return Err(CliOrchestrationError::msg(format!(
                    "trimul.baseline.metric must be {expected:?}; old or unversioned baselines must be re-measured"
                )));
            }
            if baseline.isolation_tier != task.verifier_isolation_tier {
                return Err(CliOrchestrationError::msg(
                    "trimul.baseline.isolation_tier does not match trimul.verifier_isolation_tier",
                ));
            }
            validate_lower_digest(
                "trimul.baseline.isolation_evidence_sha256",
                &baseline.isolation_evidence_sha256,
            )?;
            guard_cli_baseline_gpu(&baseline.gpu)?;
            reward = reward.with_baseline_ns(baseline.ns);
        }
    }
    let reward = reward.with_verified_isolation().map_err(|error| {
        CliOrchestrationError::msg(format!("verifier isolation preflight failed: {error}"))
    })?;
    let reward = reward.with_verified_runtime().map_err(|error| {
        CliOrchestrationError::msg(format!("verifier runtime preflight failed: {error}"))
    })?;
    if apply_baseline {
        if let Some(baseline) = &task.baseline {
            let isolation = reward
                .verifier_isolation_evidence()
                .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
            if baseline.isolation_tier != isolation.tier
                || baseline.isolation_evidence_sha256
                    != crate::trimul::verifier_isolation_evidence_sha256(&isolation)
            {
                return Err(CliOrchestrationError::msg(
                    "trimul.baseline isolation tier/evidence does not match the active verifier; re-measure the baseline through this exact backend",
                ));
            }
        }
    }
    Ok(reward)
}

fn cli_launch_verifier_identity(
    reward: &crate::trimul::TrimulReward,
    assets: &crate::trimul::TrimulVerifierAssets,
) -> Result<LaunchVerifierIdentity, CliOrchestrationError> {
    let isolation = reward.verifier_isolation_evidence().map_err(|error| {
        CliOrchestrationError::msg(format!("verifier preflight revalidation failed: {error}"))
    })?;
    let runtime_preflight = reward.runtime_preflight_evidence().map_err(|error| {
        CliOrchestrationError::msg(format!("runtime control preflight failed: {error}"))
    })?;
    Ok(LaunchVerifierIdentity {
        assets: assets.identity().clone(),
        isolation_evidence_sha256: crate::trimul::verifier_isolation_evidence_sha256(&isolation),
        timing_metric: crate::trimul::timing_metric_for_tier(isolation.tier).to_string(),
        runtime_hardening_contract: crate::trimul::TRIMUL_RUNTIME_HARDENING_CONTRACT.to_string(),
        runtime_preflight_evidence_sha256: crate::trimul::runtime_preflight_evidence_sha256(
            &runtime_preflight,
        ),
        runtime_preflight,
        isolation,
    })
}

fn validate_lower_digest(label: &str, digest: &str) -> Result<(), CliOrchestrationError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(CliOrchestrationError::msg(format!(
            "{label} must be a 64-character lowercase SHA-256"
        )))
    }
}

fn guard_cli_baseline_gpu(configured: &str) -> Result<(), CliOrchestrationError> {
    let want = configured.trim();
    if want.is_empty() {
        return Err(CliOrchestrationError::msg("trimul.baseline.gpu is empty"));
    }
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|output| output.status.success());
    let detected = output.as_ref().and_then(|output| {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(ToOwned::to_owned)
    });
    let want = want.to_lowercase();
    let matches = detected.as_deref().is_some_and(|name| {
        let name = name.to_lowercase();
        let bytes = name.as_bytes();
        name.match_indices(&want).any(|(index, matched)| {
            let before = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
            let after_index = index + matched.len();
            let after = after_index >= bytes.len() || !bytes[after_index].is_ascii_alphanumeric();
            before && after
        })
    });
    if matches {
        Ok(())
    } else {
        Err(CliOrchestrationError::msg(format!(
            "baseline was measured on GPU {configured:?} but this node's GPU is {:?}; re-measure on this GPU",
            detected.as_deref().unwrap_or("unavailable")
        )))
    }
}

pub(crate) fn run_discovery_training<R>(
    request: DiscoveryTrainingRequest<'_, R>,
) -> Result<EngineOutcome, EngineError>
where
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
{
    run_production_engine(EnginePlan::from_discovery(request))
}

fn cli_engine_error(error: EngineError) -> CliOrchestrationError {
    match error {
        EngineError::Health(report) => CliOrchestrationError::RunHealth(report),
        other => CliOrchestrationError::msg(other.to_string()),
    }
}

fn run_production_engine<'a, R>(plan: EnginePlan<'a, R>) -> Result<EngineOutcome, EngineError>
where
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
{
    ConcreteEngine::new(
        plan,
        Box::new(|model_dir, device, options| {
            load_auto_policy_with_identity(model_dir, device, options)
                .map_err(|error| EngineError::ModelLoad(Box::new(error)))
        }),
        AutoPolicy::supports_tensor_parallel,
        resolve_production_eos,
    )
    .run()
}

fn resolve_production_eos(
    model_dir: &Path,
    tokenizer: &HfTokenizer,
    selection: EngineEosSelection,
) -> Result<Option<u32>, EngineError> {
    resolve_checkpoint_eos(
        model_dir,
        tokenizer,
        engine_checkpoint_eos_selection(selection),
    )
    .map_err(|error| EngineError::GenerationEnd(Box::new(error)))
}

/// Test-only generic loader seam.  It is deliberately compiled only with the
/// library's own unit tests; no package or downstream caller can reach it.
#[cfg(test)]
pub(crate) fn run_discovery_with_test_loader<'a, P, K, R, F>(
    request: DiscoveryTrainingRequest<'a, R>,
    loader: F,
    supports_tensor_parallel: fn(&P) -> bool,
) -> Result<(EngineOutcome, P), EngineError>
where
    P: Policy + TensorParallelPolicy,
    K: TokenizerLike,
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
    F: FnOnce(&Path, &Device, &LoaderOpts) -> Result<(P, K, PolicyLoadIdentity), EngineError> + 'a,
{
    ConcreteEngine::new(
        EnginePlan::from_discovery(request),
        Box::new(loader),
        supports_tensor_parallel,
        resolve_test_eos::<K>,
    )
    .run_with_policy()
}

#[cfg(test)]
fn resolve_test_eos<K: TokenizerLike>(
    _model_dir: &Path,
    _tokenizer: &K,
    selection: EngineEosSelection,
) -> Result<Option<u32>, EngineError> {
    match selection {
        EngineEosSelection::Disabled => Ok(None),
        EngineEosSelection::CheckpointDefault | EngineEosSelection::Explicit(_) => {
            Err(EngineError::Configuration(
                "test-only injected loader requires EOS stopping to be disabled".into(),
            ))
        }
    }
}

struct EngineLaunchArtifacts {
    candidate_signer: CandidateSigner,
    manifest: Option<LaunchManifest>,
    launch_bytes: Vec<u8>,
    launch_sha256: String,
    signing_public_key: String,
    run_id: String,
}

#[derive(Serialize)]
struct DiscoveryEngineLaunchPayload<'a> {
    contract: &'static str,
    contract_version: u32,
    launch_authentication: &'static str,
    launch_trust_boundary: &'static str,
    ferrl_source: &'a crate::discovery::BuildSourceIdentity,
    task: &'a crate::discovery::TaskIdentity,
    model: &'a crate::discovery::ModelIdentity,
    metric_contract: &'a crate::discovery::MetricContract,
    execution: crate::discovery::ExecutionDevice,
    resolved_eos_token_id: Option<u32>,
    training_samples_sha256: &'a str,
    training_samples_count: usize,
    held_out_samples_sha256: &'a str,
    held_out_samples_count: usize,
    steps: u64,
    group_size: usize,
    max_new_tokens: usize,
    eval_group_size: usize,
    temperature: f64,
    learning_rate: f64,
    seed: u64,
    candidate_signing_public_key: &'a str,
}

#[derive(Serialize)]
struct DiscoveryEngineLaunchManifest<'a> {
    contract: &'static str,
    launch_authentication: &'static str,
    launch_trust_boundary: &'static str,
    payload_sha256: &'a str,
    payload: &'a DiscoveryEngineLaunchPayload<'a>,
}

type EngineLoader<'a, P, K> = Box<
    dyn FnOnce(&Path, &Device, &LoaderOpts) -> Result<(P, K, PolicyLoadIdentity), EngineError> + 'a,
>;

struct ConcreteEngine<'a, R, P, K>
where
    R: RewardFn,
    P: Policy + TensorParallelPolicy,
    K: TokenizerLike,
{
    plan: EnginePlan<'a, R>,
    loader: Option<EngineLoader<'a, P, K>>,
    supports_tensor_parallel: fn(&P) -> bool,
    resolve_eos: fn(&Path, &K, EngineEosSelection) -> Result<Option<u32>, EngineError>,
    runtime: Option<CliRuntime>,
    policy: Option<P>,
    tokenizer: Option<K>,
    trainer_config: TrainerConfig,
    policy_identity: Option<PolicyLoadIdentity>,
    model_identity: Option<crate::discovery::ModelIdentity>,
    generation_config: Option<GenConfig>,
    training_samples: Vec<Sample<R::Target>>,
    evaluation_samples: Vec<Sample<R::Target>>,
    verifier_assets_identity: Option<crate::trimul::TrimulVerifierIdentity>,
    verifier_identity: Option<LaunchVerifierIdentity>,
    manifest: Option<LaunchManifest>,
    run: Option<RunDir>,
    trainer: Option<Trainer>,
    launch_bytes: Option<Vec<u8>>,
    launch_sha256: Option<String>,
    signing_public_key: Option<String>,
    history: Option<Vec<Metrics>>,
    evaluation: Option<EngineEvaluation>,
    candidates: Vec<AuthenticatedCandidate>,
    health_report: Option<CliRunHealthReport>,
    training_samples_sha256: String,
    evaluation_samples_sha256: String,
}

impl<'a, R, P, K> ConcreteEngine<'a, R, P, K>
where
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
    P: Policy + TensorParallelPolicy,
    K: TokenizerLike,
{
    fn new(
        plan: EnginePlan<'a, R>,
        loader: EngineLoader<'a, P, K>,
        supports_tensor_parallel: fn(&P) -> bool,
        resolve_eos: fn(&Path, &K, EngineEosSelection) -> Result<Option<u32>, EngineError>,
    ) -> Self {
        Self {
            trainer_config: plan.trainer_config.clone(),
            plan,
            loader: Some(loader),
            supports_tensor_parallel,
            resolve_eos,
            runtime: None,
            policy: None,
            tokenizer: None,
            policy_identity: None,
            model_identity: None,
            generation_config: None,
            training_samples: Vec::new(),
            evaluation_samples: Vec::new(),
            verifier_assets_identity: None,
            verifier_identity: None,
            manifest: None,
            run: None,
            trainer: None,
            launch_bytes: None,
            launch_sha256: None,
            signing_public_key: None,
            history: None,
            evaluation: None,
            candidates: Vec::new(),
            health_report: None,
            training_samples_sha256: String::new(),
            evaluation_samples_sha256: String::new(),
        }
    }

    fn run(mut self) -> Result<EngineOutcome, EngineError> {
        self.run_inner()
    }

    #[cfg(test)]
    fn run_with_policy(mut self) -> Result<(EngineOutcome, P), EngineError> {
        let outcome = self.run_inner()?;
        let policy = self
            .policy
            .take()
            .ok_or_else(|| EngineError::Configuration("test engine lost loaded policy".into()))?;
        Ok((outcome, policy))
    }

    fn run_inner(&mut self) -> Result<EngineOutcome, EngineError> {
        self.setup()?;
        self.preflight()?;
        self.launch_and_build_trainer()?;
        if let Some(preempted) = self.train()? {
            return Ok(preempted);
        }
        self.post_run_health()?;
        self.evaluate_and_publish()?;
        self.load_and_rank_candidates()?;
        self.completed()
    }

    #[allow(clippy::cognitive_complexity)]
    fn setup(&mut self) -> Result<(), EngineError> {
        if let EngineMode::Cli(cli) = &self.plan.mode {
            cli.health_policy
                .validate(&self.plan.trainer_config)
                .map_err(|error| EngineError::Configuration(error.to_string()))?;
        }
        let execution = std::mem::replace(&mut self.plan.execution, CliExecution::WorldOne);
        let runtime = CliRuntime::from_execution(execution);
        validate_engine_topology(&self.plan.mode, &runtime)?;
        tracing::info!(
            task = %engine_task_name(&self.plan.mode),
            steps = self.plan.trainer_config.steps,
            group_size = self.plan.trainer_config.group_size,
            train = self.plan.training_samples.len(),
            eval = self.plan.evaluation_samples.len(),
            activation_checkpointing = engine_activation_checkpointing(&self.plan.mode),
            tensor_parallel_rank = runtime.tensor_parallel_plan.rank(),
            tensor_parallel_world = runtime.tensor_parallel_plan.world_size(),
            "ferrl shared orchestration: starting"
        );
        self.runtime = Some(runtime);

        let runtime = self.runtime.as_ref().expect("runtime was just installed");
        let launch_comm = runtime.launch.as_ref().map(|comm| comm as &dyn Comm);
        let distributed = runtime.distributed.clone();
        let tensor_parallel_plan = runtime.tensor_parallel_plan;
        let loader = self.loader.take().ok_or_else(|| {
            EngineError::Configuration("policy loader was entered more than once".into())
        })?;
        let model_setup = (|| {
            let (policy, tokenizer, identity) = loader(
                self.plan.model_dir,
                self.plan.device,
                &self.plan.loader_opts,
            )?;
            let mut trainer_config = self.plan.trainer_config.clone();
            trainer_config.eos_token_id =
                (self.resolve_eos)(self.plan.model_dir, &tokenizer, self.plan.eos_selection)?;
            if runtime.tensor_parallel.is_some() && !(self.supports_tensor_parallel)(&policy) {
                return Err(EngineError::Configuration(
                    "loaded checkpoint family does not support tensor_parallel execution; supported families are qwen3 (including legacy configs without model_type) and dense gemma4/gemma4_unified; qwen3_5/qwen3_5_moe (Qwen3.5/3.6) are unsupported".into(),
                ));
            }
            if tensor_parallel_plan.is_sharded()
                && !policy.supports_sharded_tensor_parallel_backward()
            {
                return Err(EngineError::Configuration(
                    "sharded tensor_parallel training is supported only for dense gemma4/gemma4_unified policies with activation checkpointing; the loaded policy does not provide cross-rank backward semantics".into(),
                ));
            }
            Ok((policy, tokenizer, trainer_config, identity))
        })();
        let (policy, tokenizer, trainer_config, policy_identity) =
            coordinate_engine_result(launch_comm, "model and EOS setup", model_setup)?;
        if let Some(comm) = launch_comm {
            validate_resolved_eos_consensus(trainer_config.eos_token_id, comm)
                .map_err(|error| EngineError::Configuration(error.to_string()))?;
        }
        validate_data_parallel_policy_preflight(
            &policy,
            &policy_identity.policy_sha256,
            distributed.as_ref().map(|comm| comm as &dyn Comm),
        )
        .map_err(|error| EngineError::Configuration(error.to_string()))?;
        if let Some(comm) = runtime.tensor_parallel.as_ref() {
            validate_tensor_parallel_policy_preflight(&policy, comm, launch_comm)?;
        }

        let verifier_assets_identity = self
            .plan
            .verifier_assets
            .map(|assets| assets.identity().clone());
        if let EngineMode::Cli(cli) = &self.plan.mode {
            let is_trimul = cli.launch.task == "trimul";
            if self.plan.verifier_assets.is_some() != is_trimul
                || self.plan.verifier_identity.is_some() != is_trimul
                || self.plan.verifier_assets.is_some() != self.plan.verifier_identity.is_some()
            {
                return Err(EngineError::Configuration(
                    "TriMul launch requires both verifier assets and isolation evidence, while non-TriMul launches require neither".into(),
                ));
            }
            let prompt_sha256 = self.plan.rendered_prompt_bytes.map(sha256_hex);
            let portable_verifier = self
                .plan
                .verifier_identity
                .as_ref()
                .map(portable_verifier_consensus)
                .transpose()
                .map_err(|error| EngineError::Message(error.to_string()))?;
            let common_provenance = serde_json::to_vec(&(
                &cli.launch.ferrl_commit,
                &cli.launch.run.group_id,
                &policy_identity.policy_sha256,
                &policy_identity.tokenizer_sha256,
                policy_identity.model_family,
                &prompt_sha256,
                &verifier_assets_identity,
                &portable_verifier,
            ))
            .map_err(|error| EngineError::Serialization {
                kind: "launch provenance",
                source: error,
            })?;
            validate_engine_value_consensus(
                "model/checkpoint/tokenizer/prompt provenance",
                &common_provenance,
                launch_comm,
            )?;
        }

        self.model_identity = match &self.plan.mode {
            EngineMode::Discovery(spec) => Some(crate::discovery::ModelIdentity::from_loader(
                policy_identity.clone(),
                spec.execution_device,
            )),
            EngineMode::Cli(_) => None,
        };
        self.policy = Some(policy);
        self.tokenizer = Some(tokenizer);
        self.trainer_config = trainer_config.clone();
        self.policy_identity = Some(policy_identity);
        self.generation_config = Some(GenConfig::from(&trainer_config));
        self.verifier_assets_identity = verifier_assets_identity;
        self.verifier_identity = self.plan.verifier_identity.clone();
        Ok(())
    }

    fn preflight(&mut self) -> Result<(), EngineError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Configuration("tokenizer was not loaded".into()))?;
        let local = (|| {
            let (training_samples, training_bytes) =
                exact_execution_samples(self.plan.training_samples, "ordered training samples")
                    .map_err(|source| EngineError::Serialization {
                        kind: "ordered training samples",
                        source,
                    })?;
            let (evaluation_samples, evaluation_bytes) =
                exact_execution_samples(self.plan.evaluation_samples, "ordered held-out samples")
                    .map_err(|source| EngineError::Serialization {
                    kind: "ordered held-out samples",
                    source,
                })?;
            preflight_prompt_tokenization(&training_samples, "ordered training samples", tokenizer)
                .map_err(EngineError::Configuration)?;
            preflight_prompt_tokenization(
                &evaluation_samples,
                "ordered held-out samples",
                tokenizer,
            )
            .map_err(EngineError::Configuration)?;
            Ok((
                training_samples,
                evaluation_samples,
                sha256_hex(&training_bytes),
                sha256_hex(&evaluation_bytes),
            ))
        })();
        let launch_comm = self.engine_comm();
        let (training_samples, evaluation_samples, training_sha256, evaluation_sha256) =
            coordinate_engine_result(
                launch_comm,
                "exact sample reconstruction and tokenizer preflight",
                local,
            )?;
        validate_engine_value_consensus(
            "ordered training samples",
            training_sha256.as_bytes(),
            launch_comm,
        )?;
        validate_engine_value_consensus(
            "ordered held-out samples",
            evaluation_sha256.as_bytes(),
            launch_comm,
        )?;
        self.training_samples = training_samples;
        self.evaluation_samples = evaluation_samples;
        self.training_samples_sha256 = training_sha256;
        self.evaluation_samples_sha256 = evaluation_sha256;
        Ok(())
    }

    #[allow(clippy::cognitive_complexity)]
    fn launch_and_build_trainer(&mut self) -> Result<(), EngineError> {
        if matches!(&self.plan.mode, EngineMode::Discovery(_))
            && self.trainer_config.candidate_log_top_k != self.trainer_config.group_size
        {
            return Err(EngineError::Configuration(
                "discovery requires complete candidate logging".into(),
            ));
        }
        let launch = self.build_launch_artifacts();
        let launch_comm = self.engine_comm();
        let artifacts = coordinate_engine_result(launch_comm, "launch authentication", launch)?;
        let verifier_check = self.plan.verifier_assets.map_or(Ok(()), |assets| {
            assets
                .verify_current()
                .map_err(|error| EngineError::Message(error.to_string()))
        });
        coordinate_engine_result(
            launch_comm,
            "launch-bound verifier revalidation",
            verifier_check,
        )?;
        let publication = (|| {
            let output_root = match &self.plan.mode {
                EngineMode::Discovery(spec) => spec.runs_root,
                EngineMode::Cli(cli) => cli.launch.output_root.as_path(),
            };
            let run = RunDir::create(output_root, artifacts.run_id.clone())
                .map_err(|error| EngineError::Launch(Box::new(error)))?;
            run.write_immutable_launch(&artifacts.launch_bytes, self.plan.rendered_prompt_bytes)
                .map_err(|error| EngineError::Launch(Box::new(error)))?;
            let policy_sha256 = self
                .policy_identity
                .as_ref()
                .ok_or_else(|| {
                    EngineError::Configuration("launch requires policy identity".into())
                })?
                .policy_sha256
                .clone();
            let mut trainer = open_engine_trainer(
                self.trainer_config.clone(),
                &run,
                self.runtime
                    .as_ref()
                    .and_then(|runtime| runtime.distributed.clone()),
                &policy_sha256,
                &artifacts.launch_sha256,
                artifacts.candidate_signer,
            )
            .map_err(|error| EngineError::Training(Box::new(error)))?;
            if let EngineMode::Discovery(spec) = &self.plan.mode {
                if let Some(flag) = spec.preemption_flag.clone() {
                    trainer = trainer.with_preemption_flag(flag);
                }
            }
            Ok((run, trainer))
        })();
        let (run, trainer) =
            coordinate_engine_result(launch_comm, "run directory and trainer setup", publication)?;
        self.launch_bytes = Some(artifacts.launch_bytes);
        self.launch_sha256 = Some(artifacts.launch_sha256);
        self.signing_public_key = Some(artifacts.signing_public_key);
        self.manifest = artifacts.manifest;
        self.run = Some(run);
        self.trainer = Some(trainer);
        Ok(())
    }

    fn build_launch_artifacts(&self) -> Result<EngineLaunchArtifacts, EngineError> {
        let signer =
            CandidateSigner::generate().map_err(|error| EngineError::Message(error.to_string()))?;
        let signing_public_key = signer.public_key_hex();
        match &self.plan.mode {
            EngineMode::Cli(cli) => {
                let manifest = LaunchManifest::new(LaunchPayload {
                    task: cli.launch.task.clone(),
                    ferrl_commit: cli.launch.ferrl_commit.clone(),
                    authentication: cli.launch.authentication,
                    run: cli.launch.run.clone(),
                    config: cli.launch.config.clone(),
                    model: LaunchModelIdentity {
                        family: self
                            .policy_identity
                            .as_ref()
                            .ok_or_else(|| {
                                EngineError::Configuration("launch requires policy identity".into())
                            })?
                            .model_family
                            .to_owned(),
                        checkpoint_policy_sha256: self
                            .policy_identity
                            .as_ref()
                            .expect("policy identity checked above")
                            .policy_sha256
                            .clone(),
                        tokenizer_sha256: self
                            .policy_identity
                            .as_ref()
                            .expect("policy identity checked above")
                            .tokenizer_sha256
                            .clone(),
                        resolved_eos_token_id: self.trainer_config.eos_token_id,
                    },
                    prompt: self
                        .plan
                        .rendered_prompt_bytes
                        .map(|bytes| LaunchPromptIdentity {
                            file: RunDir::PROMPT_FILE.to_owned(),
                            sha256: sha256_hex(bytes),
                            len_bytes: bytes.len(),
                        }),
                    training_samples: Some(LaunchSampleIdentity {
                        sha256: self.training_samples_sha256.clone(),
                        count: self.training_samples.len(),
                    }),
                    held_out_samples: Some(LaunchSampleIdentity {
                        sha256: self.evaluation_samples_sha256.clone(),
                        count: self.evaluation_samples.len(),
                    }),
                    verifier: self.verifier_identity.clone(),
                    candidate_ledger: LaunchCandidateLedger {
                        file: RunDir::CANDIDATES_FILE.to_owned(),
                        format_version: 1,
                        row_digest_domain: CANDIDATE_RECORD_DOMAIN.to_owned(),
                        row_signature_algorithm: "ed25519".to_owned(),
                        signing_public_key,
                    },
                })
                .map_err(|error| EngineError::Message(error.to_string()))?;
                let manifest = match cli.launch.authentication {
                    LaunchAuthenticationMode::LocalEphemeralV1 => manifest,
                    LaunchAuthenticationMode::ExternalAttestedV1 => {
                        let attestor = cli.attestor.ok_or_else(|| {
                            EngineError::Message(
                                "launch_authentication = \"external_attested_v1\" requires the protected external launch attestor".into(),
                            )
                        })?;
                        manifest
                            .attest(attestor)
                            .map_err(|error| EngineError::Message(error.to_string()))?
                    }
                };
                let launch_bytes = manifest
                    .to_pretty_bytes()
                    .map_err(|error| EngineError::Message(error.to_string()))?;
                Ok(EngineLaunchArtifacts {
                    candidate_signer: signer,
                    launch_sha256: manifest.payload_sha256.clone(),
                    launch_bytes,
                    run_id: cli.launch.run.run_id.clone(),
                    signing_public_key: manifest
                        .payload
                        .candidate_ledger
                        .signing_public_key
                        .clone(),
                    manifest: Some(manifest),
                })
            }
            EngineMode::Discovery(spec) => {
                let model = self.model_identity.as_ref().ok_or_else(|| {
                    EngineError::Configuration("discovery launch requires model identity".into())
                })?;
                let payload = DiscoveryEngineLaunchPayload {
                    contract: "ferrl.discovery-launch.v1",
                    contract_version: 1,
                    launch_authentication: "local_ephemeral_v1",
                    launch_trust_boundary: concat!(
                        "same-process signed candidate binding; not resistant to another process ",
                        "controlling the same UID"
                    ),
                    ferrl_source: &spec.ferrl_source,
                    task: spec.task,
                    model,
                    metric_contract: &spec.metric_contract,
                    execution: spec.execution_device,
                    resolved_eos_token_id: self.trainer_config.eos_token_id,
                    training_samples_sha256: &self.training_samples_sha256,
                    training_samples_count: self.training_samples.len(),
                    held_out_samples_sha256: &self.evaluation_samples_sha256,
                    held_out_samples_count: self.evaluation_samples.len(),
                    steps: spec.steps,
                    group_size: spec.group_size,
                    max_new_tokens: spec.max_new_tokens,
                    eval_group_size: spec.eval_group_size,
                    temperature: spec.temperature,
                    learning_rate: spec.learning_rate,
                    seed: spec.seed,
                    candidate_signing_public_key: &signing_public_key,
                };
                let payload_bytes =
                    serde_json::to_vec(&payload).map_err(|source| EngineError::Serialization {
                        kind: "discovery launch payload",
                        source,
                    })?;
                let launch_sha256 = sha256_hex(&payload_bytes);
                let run_id = format!("discovery-{}", &launch_sha256[..20]);
                let manifest = DiscoveryEngineLaunchManifest {
                    contract: "ferrl.discovery-launch.v1",
                    launch_authentication: "local_ephemeral_v1",
                    launch_trust_boundary: payload.launch_trust_boundary,
                    payload_sha256: &launch_sha256,
                    payload: &payload,
                };
                let mut launch_bytes = serde_json::to_vec_pretty(&manifest).map_err(|source| {
                    EngineError::Serialization {
                        kind: "discovery launch manifest",
                        source,
                    }
                })?;
                launch_bytes.push(b'\n');
                Ok(EngineLaunchArtifacts {
                    candidate_signer: signer,
                    manifest: None,
                    launch_bytes,
                    signing_public_key,
                    launch_sha256,
                    run_id,
                })
            }
        }
    }

    fn train(&mut self) -> Result<Option<EngineOutcome>, EngineError> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| EngineError::Configuration("training runtime is missing".into()))?;
        let tensor_parallel = runtime.tensor_parallel.clone();
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Configuration("training tokenizer is missing".into()))?;
        let samples = &self.training_samples;
        let reward = self.plan.reward;
        let (history, stop) =
            {
                let policy = self.policy.as_mut().ok_or_else(|| {
                    EngineError::Configuration("training policy is missing".into())
                })?;
                let trainer = self.trainer.as_mut().ok_or_else(|| {
                    EngineError::Configuration("training trainer is missing".into())
                })?;
                match tensor_parallel.as_ref() {
                    Some(comm) => trainer
                        .train_tensor_parallel(policy, reward, tokenizer, samples, comm)
                        .map_err(|error| EngineError::Training(Box::new(error)))?,
                    None => trainer
                        .train(policy, reward, tokenizer, samples)
                        .map_err(|error| EngineError::Training(Box::new(error)))?,
                }
            };
        self.history = Some(history);
        if stop != RunStop::Preempted {
            return Ok(None);
        }
        let run = self.run.as_ref().ok_or_else(|| {
            EngineError::Configuration("preemption requires a published run".into())
        })?;
        let (completed_steps, checkpoint_path) =
            if matches!(&self.plan.mode, EngineMode::Discovery(_)) {
                let history_len = self.history.as_ref().map_or(0, Vec::len);
                let completed_steps = u64::try_from(history_len).map_err(|_| {
                    EngineError::PreemptionCheckpoint(
                        "completed history length does not fit the checkpoint step domain".into(),
                    )
                })?;
                let latest = crate::latest_checkpoint(run.checkpoints_dir())
                    .map_err(|error| EngineError::PreemptionCheckpointScan(Box::new(error)))?;
                let latest = validate_preemption_checkpoint(completed_steps, latest)?;
                (Some(latest.step), Some(latest.dir))
            } else {
                (None, None)
            };
        Ok(Some(EngineOutcome::Preempted(EnginePreemptedRun {
            run_dir: run.root().to_path_buf(),
            completed_steps,
            checkpoint_path,
            presentation_rank: self.presentation_rank()?,
        })))
    }

    fn post_run_health(&mut self) -> Result<(), EngineError> {
        let history = self.history.as_ref().ok_or_else(|| {
            EngineError::Configuration("post-run health requires completed training".into())
        })?;
        let run = self.run.clone().ok_or_else(|| {
            EngineError::Configuration("post-run health requires a published run".into())
        })?;
        match &self.plan.mode {
            EngineMode::Discovery(_) => {
                if summarize(history).is_none() {
                    return Err(EngineError::Configuration(
                        "post-run health requires at least one completed training metric".into(),
                    ));
                }
            }
            EngineMode::Cli(cli) => {
                let policy = cli.health_policy.clone();
                let trainer = self.trainer_config.clone();
                let tensor_parallel = self
                    .runtime
                    .as_ref()
                    .and_then(|runtime| runtime.tensor_parallel.clone());
                let local = run_on_tensor_parallel_primary_engine(
                    tensor_parallel.as_ref(),
                    "tensor-parallel post-run health",
                    || {
                        let Some(summary) = summarize(history) else {
                            return Ok(None);
                        };
                        tracing::info!(steps = summary.steps, "ferrl train: complete");
                        let report =
                            evaluate_cli_run_health(&policy, history, &summary, &run, &trainer)
                                .map_err(|error| EngineError::Message(error.to_string()))?;
                        if report.is_fail() {
                            return Err(EngineError::Health(report));
                        }
                        Ok((!cli.health_policy_is_default).then_some(report))
                    },
                );
                let report =
                    coordinate_engine_result(self.engine_comm(), "post-run health", local)?;
                self.health_report = report.flatten();
            }
        }
        Ok(())
    }

    fn evaluation_generation_config(&self) -> GenConfig {
        match &self.plan.mode {
            EngineMode::Discovery(spec) => GenConfig {
                group_size: spec.eval_group_size,
                max_new_tokens: spec.max_new_tokens,
                temperature: spec.temperature,
                eos_token_id: self.trainer_config.eos_token_id,
                eval_sampling: Some(EvalSampling::default()),
            },
            EngineMode::Cli(_) => self
                .generation_config
                .expect("CLI evaluation generation config is installed during setup"),
        }
    }

    fn evaluate_and_publish(&mut self) -> Result<(), EngineError> {
        if self.evaluation_samples.is_empty() {
            return Ok(());
        }
        let run = self.run.clone().ok_or_else(|| {
            EngineError::Configuration("evaluation requires a published run".into())
        })?;
        let launch_sha256 = self.launch_sha256.clone().ok_or_else(|| {
            EngineError::Configuration("evaluation requires launch identity".into())
        })?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Configuration("evaluation tokenizer is missing".into()))?;
        let generation_config = self.evaluation_generation_config();
        let local = {
            let policy = self
                .policy
                .as_mut()
                .ok_or_else(|| EngineError::Configuration("evaluation policy is missing".into()))?;
            evaluate(
                policy,
                self.plan.evaluation_reward,
                tokenizer,
                &self.evaluation_samples,
                &generation_config,
            )
            .map_err(|error| EngineError::Evaluation(Box::new(error)))
        };
        let report = coordinate_engine_result(self.engine_comm(), "held-out evaluation", local)?;
        let bytes = match &self.plan.mode {
            EngineMode::Discovery(spec) => publish_discovery_eval_report(
                spec,
                &self.evaluation_samples,
                &self.evaluation_samples_sha256,
                &report,
                &run,
                &launch_sha256,
            )?,
            EngineMode::Cli(cli) => {
                publish_cli_eval_report(
                    &cli.launch.task,
                    cli.data_seed,
                    cli.trimul_held_out_secret_seed,
                    &self.evaluation_samples,
                    &report,
                    &run,
                    &launch_sha256,
                    self.verifier_assets_identity.as_ref(),
                    self.engine_comm(),
                )
                .map_err(|error| EngineError::Message(error.to_string()))?;
                Vec::new()
            }
        };
        if matches!(&self.plan.mode, EngineMode::Cli(_)) {
            tracing::info!(
                base = report.base_reward_mean,
                adapter = report.adapter_reward_mean,
                improvement = report.improvement(),
                "ferrl train: held-out eval (adapter vs base)"
            );
        }
        self.evaluation = Some(EngineEvaluation { report, bytes });
        Ok(())
    }

    fn load_and_rank_candidates(&mut self) -> Result<(), EngineError> {
        let run = self.run.clone().ok_or_else(|| {
            EngineError::Configuration("candidate loading requires a published run".into())
        })?;
        let launch_sha256 = self.launch_sha256.as_deref().ok_or_else(|| {
            EngineError::Configuration("candidate loading requires launch identity".into())
        })?;
        let signing_public_key = self.signing_public_key.as_deref().ok_or_else(|| {
            EngineError::Configuration("candidate loading requires signing identity".into())
        })?;
        let local = match &self.plan.mode {
            EngineMode::Discovery(spec) => load_authenticated_candidates(
                &run.candidates_path(),
                launch_sha256,
                signing_public_key,
                EngineCandidateValidation::Discovery {
                    steps: spec.steps,
                    group_size: spec.group_size,
                    max_new_tokens: spec.max_new_tokens,
                },
            ),
            EngineMode::Cli(cli) => {
                let Some(manifest) = self.manifest.as_ref() else {
                    return Err(EngineError::Configuration(
                        "candidate loading requires an authenticated launch".into(),
                    ));
                };
                load_authenticated_candidates(
                    &run.candidates_path(),
                    launch_sha256,
                    signing_public_key,
                    EngineCandidateValidation::Cli {
                        task: &cli.launch.task,
                        manifest,
                        topology: self
                            .runtime
                            .as_ref()
                            .expect("runtime installed")
                            .candidate_topology(manifest),
                        steps: self.trainer_config.steps,
                        group_size: self.trainer_config.group_size,
                        max_new_tokens: self.trainer_config.max_new_tokens,
                    },
                )
            }
        };
        let candidates = coordinate_engine_result(
            self.engine_comm(),
            "candidate loading and authentication",
            local,
        )?;
        self.candidates = candidates;
        Ok(())
    }

    fn completed(&mut self) -> Result<EngineOutcome, EngineError> {
        let presentation_rank = self.presentation_rank()?;
        let run = self
            .run
            .take()
            .ok_or_else(|| EngineError::Configuration("completed run is missing".into()))?;
        Ok(EngineOutcome::Completed(Box::new(EngineCompletedRun {
            run,
            launch_bytes: self.launch_bytes.take().ok_or_else(|| {
                EngineError::Configuration("completed launch bytes are missing".into())
            })?,
            launch_sha256: self.launch_sha256.take().ok_or_else(|| {
                EngineError::Configuration("completed launch identity is missing".into())
            })?,
            signing_public_key: self.signing_public_key.take().ok_or_else(|| {
                EngineError::Configuration("completed signing identity is missing".into())
            })?,
            model_identity: self.model_identity.take(),
            source_identity: match &self.plan.mode {
                EngineMode::Discovery(spec) => Some(spec.ferrl_source.clone()),
                EngineMode::Cli(_) => None,
            },
            metric_contract: match &self.plan.mode {
                EngineMode::Discovery(spec) => Some(spec.metric_contract.clone()),
                EngineMode::Cli(_) => None,
            },
            evaluation: self.evaluation.take(),
            candidates: std::mem::take(&mut self.candidates),
            health_report: self.health_report.take(),
            presentation_rank,
        })))
    }

    fn engine_comm(&self) -> Option<&dyn Comm> {
        self.runtime
            .as_ref()
            .and_then(|runtime| runtime.launch.as_ref())
            .map(|comm| comm as &dyn Comm)
    }

    fn presentation_rank(&self) -> Result<bool, EngineError> {
        let comm = self
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.tensor_parallel.as_ref());
        Ok(
            run_on_tensor_parallel_primary_engine(comm, "run completion output", || Ok(()))?
                .is_some(),
        )
    }
}

fn validate_preemption_checkpoint(
    completed_steps: u64,
    latest: Option<crate::LatestCheckpoint>,
) -> Result<crate::LatestCheckpoint, EngineError> {
    let latest = latest.ok_or_else(|| {
        EngineError::PreemptionCheckpoint(
            "trainer returned Preempted without a complete checkpoint".into(),
        )
    })?;
    if latest.step != completed_steps {
        return Err(EngineError::PreemptionCheckpoint(format!(
            "newest complete checkpoint step {} does not match completed history length {completed_steps}",
            latest.step
        )));
    }
    Ok(latest)
}

fn engine_checkpoint_eos_selection(selection: EngineEosSelection) -> CheckpointEosSelection {
    match selection {
        EngineEosSelection::CheckpointDefault => CheckpointEosSelection::CheckpointDefault,
        EngineEosSelection::Explicit(token_id) => CheckpointEosSelection::Explicit(token_id),
        EngineEosSelection::Disabled => CheckpointEosSelection::Disabled,
    }
}

fn engine_task_name<'a>(mode: &'a EngineMode<'a>) -> &'a str {
    match mode {
        EngineMode::Discovery(spec) => spec.task.name(),
        EngineMode::Cli(cli) => &cli.launch.task,
    }
}

fn engine_activation_checkpointing(mode: &EngineMode<'_>) -> bool {
    match mode {
        EngineMode::Discovery(_) => false,
        EngineMode::Cli(cli) => cli.activation_checkpointing,
    }
}

fn validate_engine_topology(
    mode: &EngineMode<'_>,
    runtime: &CliRuntime,
) -> Result<(), EngineError> {
    match mode {
        EngineMode::Discovery(_) => {
            if runtime.launch.is_some()
                || runtime.distributed.is_some()
                || runtime.tensor_parallel.is_some()
                || runtime.tensor_parallel_plan.is_sharded()
            {
                return Err(EngineError::Configuration(
                    "SDK discovery supports only world-one execution".into(),
                ));
            }
        }
        EngineMode::Cli(cli) => {
            let run = &cli.launch.run;
            match (
                runtime.distributed.as_ref(),
                runtime.tensor_parallel.as_ref(),
            ) {
                (None, None) => {
                    if run.data_parallel_rank != 0
                        || run.data_parallel_world_size != 1
                        || run.tensor_parallel_rank != 0
                        || run.tensor_parallel_world_size != 1
                    {
                        return Err(EngineError::Configuration(
                            "world-one launch identity disagrees with the active topology".into(),
                        ));
                    }
                }
                (Some(comm), None) => {
                    if run.data_parallel_rank != comm.rank()
                        || run.data_parallel_world_size != comm.world_size()
                        || run.tensor_parallel_rank != 0
                        || run.tensor_parallel_world_size != 1
                    {
                        return Err(EngineError::Configuration(
                            "data-parallel launch identity disagrees with the active topology"
                                .into(),
                        ));
                    }
                }
                (None, Some(comm)) => {
                    crate::validate_comm_plan(runtime.tensor_parallel_plan, comm)
                        .map_err(|error| EngineError::Configuration(error.to_string()))?;
                    if run.tensor_parallel_rank != comm.rank()
                        || run.tensor_parallel_world_size != comm.world_size()
                        || run.data_parallel_rank != 0
                        || run.data_parallel_world_size != 1
                    {
                        return Err(EngineError::Configuration(
                            "tensor-parallel launch identity disagrees with the active topology"
                                .into(),
                        ));
                    }
                }
                (Some(_), Some(_)) => {
                    return Err(EngineError::Configuration(
                        "combined data-parallel and tensor-parallel execution is unsupported"
                            .into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn engine_panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "non-string panic payload"
    }
}

fn coordinate_engine_result<T>(
    comm: Option<&dyn Comm>,
    label: &'static str,
    local: Result<T, EngineError>,
) -> Result<T, EngineError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return local;
    };
    let failed_local = if local.is_err() { 1.0 } else { 0.0 };
    let failed_global = comm.all_reduce_scalar_sum(failed_local);
    match (local, failed_global) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(EngineError::Message(error.to_string())),
        (Ok(_), Ok(failed)) if failed > 0.0 => Err(EngineError::Message(format!(
            "{label} failed on a peer distributed rank; aborting in lockstep"
        ))),
        (Ok(value), Ok(_)) => Ok(value),
    }
}

fn validate_engine_value_consensus(
    label: &'static str,
    value: &[u8],
    comm: Option<&dyn Comm>,
) -> Result<(), EngineError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return Ok(());
    };
    let digest: [u8; 32] = Sha256::digest(value).into();
    let world = comm.world_size() as f64;
    let mut mismatch = false;
    for byte in digest {
        let scalar = f64::from(byte);
        let sum = comm
            .all_reduce_scalar_sum(scalar)
            .map_err(|error| EngineError::Message(error.to_string()))?;
        mismatch |= sum != world * scalar;
    }
    coordinate_engine_result(
        Some(comm),
        "launch provenance consensus",
        if mismatch {
            Err(EngineError::Message(format!(
                "launch ranks disagree on {label}; all ranks must bind identical bytes"
            )))
        } else {
            Ok(())
        },
    )
}

fn run_on_tensor_parallel_primary_engine<T>(
    comm: Option<&SharedComm>,
    label: &'static str,
    operation: impl FnOnce() -> Result<T, EngineError>,
) -> Result<Option<T>, EngineError> {
    let local = if comm.is_none_or(|comm| comm.world_size() <= 1 || comm.rank() == 0) {
        operation().map(Some)
    } else {
        Ok(None)
    };
    coordinate_engine_result(comm.map(|comm| comm as &dyn Comm), label, local)
}

#[derive(Serialize)]
struct DiscoveryEngineHeldOutReport<'a> {
    contract: &'static str,
    contract_version: u32,
    launch_sha256: &'a str,
    task: &'a crate::discovery::TaskIdentity,
    held_out_samples_sha256: &'a str,
    held_out_samples_count: usize,
    report: &'a crate::eval::EvalReport,
}

fn publish_discovery_eval_report<T>(
    spec: &DiscoveryLaunchInput<'_>,
    evaluation_samples: &[Sample<T>],
    frozen_eval_samples_sha256: &str,
    report: &crate::eval::EvalReport,
    run: &RunDir,
    launch_sha256: &str,
) -> Result<Vec<u8>, EngineError> {
    let durable = DiscoveryEngineHeldOutReport {
        contract: "ferrl.discovery-held-out-report.v1",
        contract_version: 1,
        launch_sha256,
        task: spec.task,
        held_out_samples_sha256: frozen_eval_samples_sha256,
        held_out_samples_count: evaluation_samples.len(),
        report,
    };
    let mut expected =
        serde_json::to_vec_pretty(&durable).map_err(|source| EngineError::Serialization {
            kind: "discovery held-out report",
            source,
        })?;
    expected.push(b'\n');
    run.write_eval_report(&durable)
        .map_err(|error| EngineError::Launch(Box::new(error)))?;
    let actual =
        fs::read(run.eval_report_path()).map_err(|source| EngineError::HeldOutReportIo {
            path: run.eval_report_path(),
            source,
        })?;
    if actual != expected {
        return Err(EngineError::InvalidCandidateEvidence(
            "published held-out report bytes differ from the launch-bound report".into(),
        ));
    }
    Ok(actual)
}

#[derive(Clone)]
struct SharedComm {
    inner: Arc<Mutex<Box<dyn Comm>>>,
    rank: usize,
    world_size: usize,
}

impl SharedComm {
    fn from_box(comm: Box<dyn Comm>) -> Self {
        let rank = comm.rank();
        let world_size = comm.world_size();
        Self {
            inner: Arc::new(Mutex::new(comm)),
            rank,
            world_size,
        }
    }

    fn with_comm<T>(
        &self,
        operation: impl FnOnce(&dyn Comm) -> Result<T, CommError>,
    ) -> Result<T, CommError> {
        let comm = self.inner.lock().map_err(|_| {
            CommError::Poisoned("shared launch communicator mutex was poisoned".into())
        })?;
        operation(comm.as_ref())
    }
}

impl std::fmt::Debug for SharedComm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedComm")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish_non_exhaustive()
    }
}

impl Comm for SharedComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn validate_all_reduce_sum(&self, tensors: &[candle_core::Tensor]) -> Result<(), CommError> {
        let comm = self.inner.lock().map_err(|_| {
            CommError::Poisoned("shared launch communicator mutex was poisoned".into())
        })?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            comm.validate_all_reduce_sum(tensors)
        }))
        .unwrap_or_else(|_| {
            Err(CommError::Mismatch(
                "backend tensor capability validator panicked".into(),
            ))
        })
    }

    fn all_reduce_sum(&self, tensors: &mut Vec<candle_core::Tensor>) -> Result<(), CommError> {
        self.with_comm(|comm| comm.all_reduce_sum(tensors))
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
        self.with_comm(|comm| comm.all_reduce_scalar_sum(value))
    }
}

struct CliRuntime {
    launch: Option<SharedComm>,
    distributed: Option<SharedComm>,
    tensor_parallel: Option<SharedComm>,
    tensor_parallel_plan: TensorParallelPlan,
}

impl CliRuntime {
    fn from_execution(execution: CliExecution) -> Self {
        match execution {
            CliExecution::WorldOne => Self {
                launch: None,
                distributed: None,
                tensor_parallel: None,
                tensor_parallel_plan: TensorParallelPlan::single(),
            },
            CliExecution::DataParallel(comm) => {
                let comm = SharedComm::from_box(comm);
                Self {
                    launch: Some(comm.clone()),
                    distributed: Some(comm),
                    tensor_parallel: None,
                    tensor_parallel_plan: TensorParallelPlan::single(),
                }
            }
            CliExecution::TensorParallel { plan, comm } => {
                let comm = SharedComm::from_box(comm);
                Self {
                    launch: Some(comm.clone()),
                    distributed: None,
                    tensor_parallel: Some(comm),
                    tensor_parallel_plan: plan,
                }
            }
        }
    }

    fn candidate_topology(&self, manifest: &LaunchManifest) -> CandidateTopology {
        match self.tensor_parallel.as_ref() {
            Some(comm) if comm.world_size() > 1 => CandidateTopology {
                rank: comm.rank(),
                world_size: comm.world_size(),
            },
            Some(_) | None => {
                let (rank, world_size) = launch_candidate_topology(&manifest.payload.run);
                CandidateTopology { rank, world_size }
            }
        }
    }
}

/// Return the candidate rank/world contract bound by an immutable CLI launch.
///
/// Sharded tensor-parallel launches authenticate records against their TP
/// coordinates; all other supported launches authenticate against DP.
#[doc(hidden)]
#[must_use]
pub fn launch_candidate_topology(run: &LaunchRunIdentity) -> (usize, usize) {
    if run.tensor_parallel_world_size > 1 {
        (run.tensor_parallel_rank, run.tensor_parallel_world_size)
    } else {
        (run.data_parallel_rank, run.data_parallel_world_size)
    }
}

fn open_engine_trainer(
    config: TrainerConfig,
    run: &RunDir,
    distributed_comm: Option<SharedComm>,
    frozen_policy_sha256: &str,
    candidate_launch_sha256: &str,
    candidate_signer: CandidateSigner,
) -> Result<Trainer, crate::trainer::TrainerError> {
    let trainer = match distributed_comm {
        Some(comm) => Trainer::with_comm(config, run, comm),
        None => Trainer::new(config, run),
    }?;
    trainer
        .with_frozen_policy_sha256(frozen_policy_sha256)
        .with_candidate_provenance(candidate_launch_sha256, candidate_signer)
}

fn coordinate_cli_result<T>(
    comm: Option<&dyn Comm>,
    label: &'static str,
    local: Result<T, CliOrchestrationError>,
) -> Result<T, CliOrchestrationError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return local;
    };
    let failed_local = if local.is_err() { 1.0 } else { 0.0 };
    let failed_global = comm.all_reduce_scalar_sum(failed_local);
    match (local, failed_global) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(CliOrchestrationError::msg(error.to_string())),
        (Ok(_), Ok(failed)) if failed > 0.0 => Err(CliOrchestrationError::msg(format!(
            "{label} failed on a peer distributed rank; aborting in lockstep"
        ))),
        (Ok(value), Ok(_)) => Ok(value),
    }
}

#[cfg(test)]
fn run_on_tensor_parallel_primary<T>(
    comm: Option<&SharedComm>,
    label: &'static str,
    operation: impl FnOnce() -> Result<T, CliOrchestrationError>,
) -> Result<Option<T>, CliOrchestrationError> {
    let local = if comm.is_none_or(|comm| comm.world_size() <= 1 || comm.rank() == 0) {
        operation().map(Some)
    } else {
        Ok(None)
    };
    coordinate_cli_result(comm.map(|comm| comm as &dyn Comm), label, local)
}

fn validate_launch_value_consensus(
    label: &'static str,
    value: &[u8],
    comm: Option<&dyn Comm>,
) -> Result<(), CliOrchestrationError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return Ok(());
    };
    let digest: [u8; 32] = Sha256::digest(value).into();
    let world = comm.world_size() as f64;
    let mut mismatch = false;
    for byte in digest {
        let scalar = f64::from(byte);
        mismatch |= comm
            .all_reduce_scalar_sum(scalar)
            .map_err(|error| CliOrchestrationError::msg(error.to_string()))?
            != world * scalar;
    }
    let local = if mismatch {
        Err(CliOrchestrationError::msg(format!(
            "launch ranks disagree on {label}; all ranks must bind identical bytes"
        )))
    } else {
        Ok(())
    };
    coordinate_cli_result(Some(comm), "launch provenance consensus", local)
}

fn validate_data_parallel_policy_preflight<P: Policy>(
    policy: &P,
    policy_sha256: &str,
    comm: Option<&dyn Comm>,
) -> Result<(), CliOrchestrationError> {
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        return Ok(());
    };
    Trainer::validate_data_parallel_policy_preflight(policy, policy_sha256, comm)
        .map_err(|error| CliOrchestrationError::msg(error.to_string()))
}

fn validate_tensor_parallel_policy_preflight<P: TensorParallelPolicy>(
    policy: &P,
    comm: &dyn Comm,
    launch_comm: Option<&dyn Comm>,
) -> Result<(), EngineError> {
    let local = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy
            .validate_tensor_parallel_execution(comm)
            .map_err(|error| EngineError::Configuration(error.to_string()))
    }))
    .unwrap_or_else(|payload| {
        Err(EngineError::Configuration(format!(
            "tensor-parallel policy capability preflight panicked: {}",
            engine_panic_payload_message(payload.as_ref())
        )))
    });
    coordinate_engine_result(
        launch_comm,
        "tensor-parallel policy capability preflight",
        local,
    )
}

#[derive(Serialize)]
struct DurableCliEvalReport<'a> {
    contract: &'static str,
    publishing_launch_sha256: String,
    launch_group_sha256: String,
    task: &'a str,
    split_key_contract: &'static str,
    eval_samples_sha256: String,
    evaluation_boundary_sha256: String,
    report: &'a crate::eval::EvalReport,
}

#[allow(clippy::too_many_arguments)]
fn publish_cli_eval_report<T: Serialize>(
    task: &str,
    data_seed: u64,
    trimul_held_out_secret_seed: Option<u64>,
    evaluation_samples: &[Sample<T>],
    report: &crate::eval::EvalReport,
    run: &RunDir,
    launch_sha256: &str,
    verifier_assets: Option<&crate::trimul::TrimulVerifierIdentity>,
    launch_comm: Option<&dyn Comm>,
) -> Result<(), CliOrchestrationError> {
    let (publishing_launch_sha256, launch_group_sha256) =
        distributed_launch_binding(launch_sha256, launch_comm)?;
    let eval_samples = serde_json::to_vec(evaluation_samples).map_err(|error| {
        CliOrchestrationError::msg(format!("serialize held-out samples: {error}"))
    })?;
    let split_key_contract = match task {
        "countdown" => "ferrl.countdown-split-key.sorted-multiset-target.v1",
        "math" => "ferrl.math-split-key.normalized-prompt-answer.v1",
        "trimul" => "ferrl.trimul-held-out-boundary.v1",
        _ => "ferrl.unknown-split-key.v1",
    };
    let boundary = serde_json::to_vec(&(
        "ferrl.eval-boundary.v1",
        task,
        data_seed,
        trimul_held_out_secret_seed,
        &eval_samples,
        verifier_assets,
    ))
    .map_err(|error| CliOrchestrationError::msg(format!("serialize held-out boundary: {error}")))?;
    let durable = DurableCliEvalReport {
        contract: "ferrl.eval-report.v2",
        publishing_launch_sha256,
        launch_group_sha256,
        task,
        split_key_contract,
        eval_samples_sha256: sha256_hex(&eval_samples),
        evaluation_boundary_sha256: sha256_hex(&boundary),
        report,
    };
    let consensus = serde_json::to_vec(&durable).map_err(|error| {
        CliOrchestrationError::msg(format!("serialize held-out report: {error}"))
    })?;
    validate_launch_value_consensus("held-out evaluation report", &consensus, launch_comm)?;
    let publication = if launch_comm.is_none_or(|comm| comm.rank() == 0) {
        run.write_eval_report(&durable)
            .map_err(|error| CliOrchestrationError::msg(error.to_string()))
    } else {
        Ok(())
    };
    coordinate_cli_result(
        launch_comm,
        "held-out evaluation report publication",
        publication,
    )
}

fn distributed_launch_binding(
    local_launch_sha256: &str,
    comm: Option<&dyn Comm>,
) -> Result<(String, String), CliOrchestrationError> {
    let local = decode_lower_hex("launch payload SHA-256", local_launch_sha256, 32)?;
    let Some(comm) = comm.filter(|comm| comm.world_size() > 1) else {
        let group = serde_json::to_vec(&("ferrl.launch-group.v1", [local_launch_sha256])).map_err(
            |error| CliOrchestrationError::msg(format!("serialize launch group: {error}")),
        )?;
        return Ok((local_launch_sha256.to_owned(), sha256_hex(&group)));
    };
    let mut launches = Vec::with_capacity(comm.world_size());
    for source_rank in 0..comm.world_size() {
        let mut bytes = Vec::with_capacity(local.len());
        for byte in &local {
            let contribution = if comm.rank() == source_rank {
                f64::from(*byte)
            } else {
                0.0
            };
            let value = comm
                .all_reduce_scalar_sum(contribution)
                .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
            if !value.is_finite() || value.fract() != 0.0 || !(0.0..=255.0).contains(&value) {
                return Err(CliOrchestrationError::msg(
                    "distributed launch digest byte is invalid",
                ));
            }
            bytes.push(value as u8);
        }
        launches.push(
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        );
    }
    let publishing = launches[0].clone();
    let group = serde_json::to_vec(&("ferrl.launch-group.v1", &launches))
        .map_err(|error| CliOrchestrationError::msg(format!("serialize launch group: {error}")))?;
    Ok((publishing, sha256_hex(&group)))
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[doc(hidden)]
pub struct CliRunHealthPolicy {
    reward_collapse: Option<CliWindowThreshold>,
    correctness_collapse: Option<CliWindowThreshold>,
    dropped_rows: Option<CliCountThreshold>,
    grad_spike: Option<CliFactorThreshold>,
    telemetry_dark: Option<CliHealthAction>,
    source_dominance: Option<CliFractionThreshold>,
}

impl CliRunHealthPolicy {
    /// Decode the already schema-validated CLI health policy into the concrete engine form.
    pub fn from_json_value(value: serde_json::Value) -> Result<Self, CliOrchestrationError> {
        serde_json::from_value(value).map_err(|error| {
            CliOrchestrationError::msg(format!("deserialize run_health for shared engine: {error}"))
        })
    }

    fn validate(&self, trainer: &TrainerConfig) -> Result<(), CliOrchestrationError> {
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
            return Err(CliOrchestrationError::msg(format!(
                "run_health correctness/source policies require \
                 trainer.candidate_log_top_k >= trainer.group_size for full candidate coverage \
                 (candidate_log_top_k={}, group_size={})",
                trainer.candidate_log_top_k, trainer.group_size
            )));
        }
        Ok(())
    }

    fn needs_candidate_ledger(&self) -> bool {
        self.correctness_collapse.is_some() || self.source_dominance.is_some()
    }

    #[allow(clippy::cognitive_complexity)]
    fn evaluate(
        &self,
        history: &[Metrics],
        summary: &RunSummary,
        context: CliRunHealthContext,
        candidates: Option<&CliCandidateHealth>,
    ) -> CliRunHealthReport {
        let mut report = CliRunHealthReport::default();
        if let Some(rule) = &self.reward_collapse {
            push_reward_collapse_finding(history, rule, &mut report);
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
            push_grad_spike_finding(history, rule, &mut report);
        }
        if let Some(action) = self.telemetry_dark {
            if !history.is_empty()
                && history
                    .iter()
                    .all(|metrics| metrics.rollout_capture_tokens == 0)
            {
                report.push(
                    "telemetry_dark",
                    action,
                    "off-policy drift telemetry was dark for every step".to_owned(),
                );
            }
        }
        if let Some(rule) = &self.correctness_collapse {
            push_correctness_collapse_finding(history, context, candidates, rule, &mut report);
        }
        if let Some(rule) = &self.source_dominance {
            push_source_dominance_finding(history, context, candidates, rule, &mut report);
        }
        report
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum CliHealthAction {
    Warn,
    Fail,
    Stop,
}

impl CliHealthAction {
    fn label(self) -> &'static str {
        match self {
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
            Self::Stop => "STOP",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct CliWindowThreshold {
    window: usize,
    min: f64,
    action: CliHealthAction,
}

impl CliWindowThreshold {
    fn validate(&self, label: &str) -> Result<(), CliOrchestrationError> {
        if self.window == 0 {
            return Err(CliOrchestrationError::msg(format!(
                "{label}.window must be >= 1"
            )));
        }
        if !self.min.is_finite() {
            return Err(CliOrchestrationError::msg(format!(
                "{label}.min must be finite"
            )));
        }
        validate_post_run_health_action(label, self.action)
    }

    fn validate_fraction_min(&self, label: &str) -> Result<(), CliOrchestrationError> {
        self.validate(label)?;
        if !(0.0..=1.0).contains(&self.min) {
            return Err(CliOrchestrationError::msg(format!(
                "{label}.min must be in [0, 1]"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct CliCountThreshold {
    max: u64,
    action: CliHealthAction,
}

impl CliCountThreshold {
    fn validate(&self, label: &str) -> Result<(), CliOrchestrationError> {
        validate_post_run_health_action(label, self.action)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct CliFactorThreshold {
    factor: f64,
    action: CliHealthAction,
}

impl CliFactorThreshold {
    fn validate(&self, label: &str) -> Result<(), CliOrchestrationError> {
        if !self.factor.is_finite() || self.factor <= 0.0 {
            return Err(CliOrchestrationError::msg(format!(
                "{label}.factor must be finite and > 0"
            )));
        }
        validate_post_run_health_action(label, self.action)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct CliFractionThreshold {
    max_fraction: f64,
    action: CliHealthAction,
}

impl CliFractionThreshold {
    fn validate(&self, label: &str) -> Result<(), CliOrchestrationError> {
        if !self.max_fraction.is_finite() || !(0.0..=1.0).contains(&self.max_fraction) {
            return Err(CliOrchestrationError::msg(format!(
                "{label}.max_fraction must be finite and in [0, 1]"
            )));
        }
        validate_post_run_health_action(label, self.action)
    }
}

fn validate_health_window(
    label: &str,
    window: usize,
    trainer_steps: u64,
) -> Result<(), CliOrchestrationError> {
    if window as u64 > trainer_steps {
        return Err(CliOrchestrationError::msg(format!(
            "{label}.window ({window}) must be <= trainer.steps ({trainer_steps})"
        )));
    }
    Ok(())
}

fn validate_post_run_health_action(
    label: &str,
    action: CliHealthAction,
) -> Result<(), CliOrchestrationError> {
    match action {
        CliHealthAction::Warn | CliHealthAction::Fail => Ok(()),
        CliHealthAction::Stop => Err(CliOrchestrationError::msg(format!(
            "{label}.action = \"stop\" is reserved for future in-run gating; use \"warn\" or \
             \"fail\" for the post-run policy"
        ))),
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CliRunHealthVerdict {
    /// No configured rule found a problem.
    #[default]
    Healthy,
    /// At least one warning was emitted.
    Warn,
    /// At least one failing rule was emitted.
    Fail,
}

impl CliRunHealthVerdict {
    fn label(self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }

    fn observe(&mut self, action: CliHealthAction) {
        match action {
            CliHealthAction::Warn if *self == Self::Healthy => *self = Self::Warn,
            CliHealthAction::Fail => *self = Self::Fail,
            CliHealthAction::Warn | CliHealthAction::Stop => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct CliRunHealthFinding {
    rule: &'static str,
    action: CliHealthAction,
    message: String,
}

/// Concrete CLI post-run health report for binary presentation.
#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CliRunHealthReport {
    verdict: CliRunHealthVerdict,
    findings: Vec<CliRunHealthFinding>,
}

impl CliRunHealthReport {
    fn push(&mut self, rule: &'static str, action: CliHealthAction, message: String) {
        self.verdict.observe(action);
        self.findings.push(CliRunHealthFinding {
            rule,
            action,
            message,
        });
    }

    /// Whether a configured rule failed the run.
    #[must_use]
    pub fn is_fail(&self) -> bool {
        self.verdict == CliRunHealthVerdict::Fail
    }

    /// Render the preserved CLI health presentation.
    #[must_use]
    pub fn render(&self) -> String {
        let mut output = format!("run health policy — {}\n", self.verdict.label());
        for finding in &self.findings {
            writeln!(
                output,
                "  {} {}: {}",
                finding.action.label(),
                finding.rule,
                finding.message
            )
            .expect("writing to String cannot fail");
        }
        output
    }
}

#[derive(Debug, Clone, Copy)]
struct CliRunHealthContext {
    group_size: usize,
    prompt_groups_per_step: usize,
}

impl CliRunHealthContext {
    fn from_trainer(trainer: &TrainerConfig) -> Self {
        Self {
            group_size: trainer.group_size,
            prompt_groups_per_step: trainer.grad_accum_steps,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CliCandidateHealth {
    total: usize,
    source_buckets: BTreeMap<String, usize>,
    steps: BTreeMap<u64, CliCandidateStepHealth>,
}

#[derive(Debug, Clone, Default)]
struct CliCandidateStepHealth {
    total: usize,
    correctness_supported: usize,
    correct: usize,
    prompt_groups: BTreeMap<u64, CliCandidatePromptGroupHealth>,
}

#[derive(Debug, Clone, Default)]
struct CliCandidatePromptGroupHealth {
    group_indices: BTreeSet<usize>,
}

fn evaluate_cli_run_health(
    policy: &CliRunHealthPolicy,
    history: &[Metrics],
    summary: &RunSummary,
    run: &RunDir,
    trainer: &TrainerConfig,
) -> Result<CliRunHealthReport, CliOrchestrationError> {
    let candidates = if policy.needs_candidate_ledger() {
        read_cli_candidate_health_inputs(&[run.root().to_path_buf()])?
    } else {
        None
    };
    Ok(policy.evaluate(
        history,
        summary,
        CliRunHealthContext::from_trainer(trainer),
        candidates.as_ref(),
    ))
}

fn read_cli_candidate_health_inputs(
    paths: &[PathBuf],
) -> Result<Option<CliCandidateHealth>, CliOrchestrationError> {
    let mut health = CliCandidateHealth::default();
    let mut found = false;
    for input in paths {
        let path = resolve_candidates_path(input);
        if !path.exists() {
            continue;
        }
        found = true;
        let raw = fs::read_to_string(&path).map_err(|error| {
            CliOrchestrationError::msg(format!("read {}: {error}", path.display()))
        })?;
        for (index, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: CandidateRecord = serde_json::from_str(line).map_err(|error| {
                CliOrchestrationError::msg(format!(
                    "parse {} line {}: {error}",
                    path.display(),
                    index + 1
                ))
            })?;
            health.total += 1;
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
        .to_owned()
}

fn push_reward_collapse_finding(
    history: &[Metrics],
    rule: &CliWindowThreshold,
    report: &mut CliRunHealthReport,
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
    let mean = tail
        .iter()
        .map(|metrics| f64::from(metrics.reward_mean))
        .sum::<f64>()
        / tail.len() as f64;
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
    history: &[Metrics],
    context: CliRunHealthContext,
    candidates: Option<&CliCandidateHealth>,
    rule: &CliWindowThreshold,
    report: &mut CliRunHealthReport,
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
            "candidate ledger unavailable; cannot evaluate correctness policy".to_owned(),
        );
        return;
    };
    if candidates.total == 0 {
        report.push(
            "correctness_collapse",
            rule.action,
            "candidate ledger is empty; cannot evaluate correctness policy".to_owned(),
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
    let partial_steps = partial_candidate_coverage_steps(candidates, &tail_steps, context);
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

fn trailing_metric_steps(history: &[Metrics], window: usize) -> Option<Vec<u64>> {
    (history.len() >= window).then(|| {
        history[history.len() - window..]
            .iter()
            .map(|metrics| metrics.step)
            .collect()
    })
}

fn missing_candidate_steps(candidates: &CliCandidateHealth, steps: &[u64]) -> Vec<u64> {
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
    candidates: &CliCandidateHealth,
    steps: &[u64],
    context: CliRunHealthContext,
) -> Vec<u64> {
    steps
        .iter()
        .copied()
        .filter(|step| {
            candidates
                .steps
                .get(step)
                .is_some_and(|health| !candidate_step_has_full_coverage(health, context))
        })
        .collect()
}

fn candidate_step_has_full_coverage(
    health: &CliCandidateStepHealth,
    context: CliRunHealthContext,
) -> bool {
    health.prompt_groups.len() == context.prompt_groups_per_step
        && health
            .prompt_groups
            .values()
            .all(|group| prompt_group_has_full_coverage(group, context.group_size))
}

fn prompt_group_has_full_coverage(
    group: &CliCandidatePromptGroupHealth,
    group_size: usize,
) -> bool {
    group.group_indices.len() == group_size
        && (0..group_size).all(|index| group.group_indices.contains(&index))
}

fn unsupported_correctness_steps(candidates: &CliCandidateHealth, steps: &[u64]) -> Vec<u64> {
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
    history: &[Metrics],
    rule: &CliFactorThreshold,
    report: &mut CliRunHealthReport,
) {
    let median = median_positive_grad_norm(history);
    if median <= 0.0 {
        return;
    }
    let Some(worst) = history
        .iter()
        .max_by(|left, right| left.grad_norm.total_cmp(&right.grad_norm))
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

fn median_positive_grad_norm(history: &[Metrics]) -> f32 {
    let mut values: Vec<f32> = history
        .iter()
        .map(|metrics| metrics.grad_norm)
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

fn push_source_dominance_finding(
    history: &[Metrics],
    context: CliRunHealthContext,
    candidates: Option<&CliCandidateHealth>,
    rule: &CliFractionThreshold,
    report: &mut CliRunHealthReport,
) {
    let Some(candidates) = candidates else {
        report.push(
            "source_dominance",
            rule.action,
            "candidate ledger unavailable; cannot evaluate source-dominance policy".to_owned(),
        );
        return;
    };
    if candidates.total == 0 {
        report.push(
            "source_dominance",
            rule.action,
            "candidate ledger is empty; cannot evaluate source-dominance policy".to_owned(),
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
    let partial_steps = partial_candidate_coverage_steps(candidates, &steps, context);
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
        .max_by(|(_, left), (_, right)| left.cmp(right))
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
    if input.file_name().and_then(|name| name.to_str()) == Some(RunDir::CANDIDATES_FILE) {
        return input.to_path_buf();
    }
    if input.is_dir() {
        return input.join(RunDir::CANDIDATES_FILE);
    }
    input.with_file_name(RunDir::CANDIDATES_FILE)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CandidateTopology {
    rank: usize,
    world_size: usize,
}

/// Closed candidate-ledger validation modes used by the shared engine.
#[derive(Clone, Copy)]
pub(crate) enum EngineCandidateValidation<'a> {
    /// Complete world-one SDK ledger with task-independent provenance.
    Discovery {
        /// Trainer step budget.
        steps: u64,
        /// Candidate group size.
        group_size: usize,
        /// Maximum generated width.
        max_new_tokens: usize,
    },
    /// CLI ledger bound to its immutable launch and active topology.
    Cli {
        /// Built-in task name.
        task: &'a str,
        /// Immutable launch manifest for task-specific provenance.
        manifest: &'a LaunchManifest,
        /// Active rank/world coordinates.
        topology: CandidateTopology,
        /// Trainer step budget.
        steps: u64,
        /// Candidate group size.
        group_size: usize,
        /// Maximum generated width.
        max_new_tokens: usize,
    },
}

#[allow(clippy::cognitive_complexity)]
pub(crate) fn load_authenticated_candidates(
    path: &Path,
    launch_sha256: &str,
    signing_public_key: &str,
    validation: EngineCandidateValidation<'_>,
) -> Result<Vec<AuthenticatedCandidate>, EngineError> {
    if !path.exists() {
        if matches!(validation, EngineCandidateValidation::Discovery { .. }) {
            return Err(EngineError::InvalidCandidateEvidence(
                "candidate ledger is missing".into(),
            ));
        }
        return Ok(Vec::new());
    }
    let bytes = read_engine_regular_bytes(path)?;
    if !bytes.is_empty() && !bytes.ends_with(b"\n") {
        return Err(EngineError::InvalidCandidateEvidence(format!(
            "candidate ledger {} has an unterminated final row",
            path.display()
        )));
    }
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        EngineError::InvalidCandidateEvidence(format!(
            "candidate ledger {} is not UTF-8: {error}",
            path.display()
        ))
    })?;
    let mut records = Vec::new();
    let mut positions = BTreeSet::new();
    for (index, raw_line) in text.split_terminator('\n').enumerate() {
        let row_number = index + 1;
        if raw_line.trim().is_empty() {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} contains blank row {row_number}",
                path.display()
            )));
        }
        let record = parse_engine_candidate_row(path, row_number, raw_line)?;
        crate::telemetry::verify_signed_candidate_row(
            raw_line.as_bytes(),
            signing_public_key,
            launch_sha256,
            &record,
        )
        .map_err(|error| {
            EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} row {row_number} failed canonical launch authentication: {error}",
                path.display()
            ))
        })?;
        let (steps, group_size, max_new_tokens, topology, task, manifest) = match validation {
            EngineCandidateValidation::Discovery {
                steps,
                group_size,
                max_new_tokens,
            } => (
                steps,
                group_size,
                max_new_tokens,
                CandidateTopology {
                    rank: 0,
                    world_size: 1,
                },
                None,
                None,
            ),
            EngineCandidateValidation::Cli {
                task,
                manifest,
                topology,
                steps,
                group_size,
                max_new_tokens,
            } => (
                steps,
                group_size,
                max_new_tokens,
                topology,
                Some(task),
                Some(manifest),
            ),
        };
        if let (Some("trimul"), Some(manifest)) = (task, manifest) {
            verify_cli_candidate_verifier_provenance(&record, manifest, row_number)
                .map_err(|error| EngineError::InvalidCandidateEvidence(error.to_string()))?;
        }
        if record.rank != topology.rank || record.world_size != topology.world_size {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} row {row_number} rank/world disagree with active execution topology",
                path.display()
            )));
        }
        if record.step >= steps || record.group_index >= group_size {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} row {row_number} coordinates exceed the launch config",
                path.display()
            )));
        }
        if record.completion_len_tokens == 0 || record.completion_len_tokens > max_new_tokens {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} row {row_number} completion length {} is outside the launch-bound range 1..={max_new_tokens}",
                path.display(),
                record.completion_len_tokens
            )));
        }
        if matches!(validation, EngineCandidateValidation::Discovery { .. })
            && record.prompt_index != record.step
        {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} row {row_number} has an impossible training position",
                path.display()
            )));
        }
        if !positions.insert((record.step, record.prompt_index, record.group_index)) {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} row {row_number} duplicates a training position",
                path.display()
            )));
        }
        let provenance_sha256 = record.record_sha256.clone().ok_or_else(|| {
            EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger {} row {row_number} has no validated provenance digest",
                path.display()
            ))
        })?;
        let mut exact_row_bytes = raw_line.as_bytes().to_vec();
        exact_row_bytes.push(b'\n');
        records.push(AuthenticatedCandidate {
            record,
            exact_row_bytes,
            provenance_sha256,
        });
    }
    if let EngineCandidateValidation::Discovery {
        steps, group_size, ..
    } = validation
    {
        let steps_usize = usize::try_from(steps).map_err(|_| {
            EngineError::InvalidCandidateEvidence(
                "configured step count does not fit candidate coverage arithmetic".into(),
            )
        })?;
        let expected = steps_usize.checked_mul(group_size).ok_or_else(|| {
            EngineError::InvalidCandidateEvidence("candidate coverage count overflows usize".into())
        })?;
        if records.len() != expected {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "candidate ledger has {} rows, expected complete logging of {expected}",
                records.len()
            )));
        }
    }
    records.sort_by(|left, right| {
        right
            .record
            .reward
            .total_cmp(&left.record.reward)
            .then_with(|| left.record.step.cmp(&right.record.step))
            .then_with(|| left.record.prompt_index.cmp(&right.record.prompt_index))
            .then_with(|| left.record.group_index.cmp(&right.record.group_index))
    });
    Ok(records)
}

#[cfg(test)]
fn load_cli_candidate_selection(
    run: &RunDir,
    manifest: &LaunchManifest,
    trainer_config: &TrainerConfig,
    topology: CandidateTopology,
) -> Result<Option<CandidateRecord>, CliOrchestrationError> {
    let candidates = load_authenticated_candidates(
        &run.candidates_path(),
        &manifest.payload_sha256,
        &manifest.payload.candidate_ledger.signing_public_key,
        EngineCandidateValidation::Cli {
            task: &manifest.payload.task,
            manifest,
            topology,
            steps: trainer_config.steps,
            group_size: trainer_config.group_size,
            max_new_tokens: trainer_config.max_new_tokens,
        },
    )
    .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
    Ok(candidates
        .into_iter()
        .next()
        .map(|candidate| candidate.record))
}

/// Exact authenticated candidate returned to the separate Phase-2.3 artifact adapter.
#[doc(hidden)]
#[derive(Debug)]
pub struct CliAuthenticatedCandidate {
    candidate: CandidateRecord,
    row_bytes: Vec<u8>,
}

impl CliAuthenticatedCandidate {
    /// Borrow the canonical authenticated candidate row.
    #[must_use]
    pub fn candidate(&self) -> &CandidateRecord {
        &self.candidate
    }

    /// Borrow the exact canonical JSON row bytes authenticated by the launch.
    #[must_use]
    pub fn row_bytes(&self) -> &[u8] {
        &self.row_bytes
    }
}

/// Authenticate one CLI launch's candidate ledger for the artifact adapter.
#[doc(hidden)]
pub fn load_cli_authenticated_candidate(
    run_dir: &Path,
    manifest: &LaunchManifest,
    candidate_sha256: &str,
    trainer_config: &TrainerConfig,
) -> Result<CliAuthenticatedCandidate, CliOrchestrationError> {
    decode_lower_hex("candidate record SHA-256", candidate_sha256, 32)?;
    let ledger = &manifest.payload.candidate_ledger;
    if ledger.file != RunDir::CANDIDATES_FILE
        || ledger.format_version != 1
        || ledger.row_digest_domain != CANDIDATE_RECORD_DOMAIN
        || ledger.row_signature_algorithm != "ed25519"
    {
        return Err(CliOrchestrationError::msg(
            "unsupported candidate-ledger contract in launch.json",
        ));
    }
    let candidates = load_authenticated_candidates(
        &run_dir.join(&ledger.file),
        &manifest.payload_sha256,
        &ledger.signing_public_key,
        EngineCandidateValidation::Cli {
            task: &manifest.payload.task,
            manifest,
            topology: {
                let (rank, world_size) = launch_candidate_topology(&manifest.payload.run);
                CandidateTopology { rank, world_size }
            },
            steps: trainer_config.steps,
            group_size: trainer_config.group_size,
            max_new_tokens: trainer_config.max_new_tokens,
        },
    )
    .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
    let mut selected = None;
    for candidate in candidates {
        if candidate.record.record_sha256.as_deref() == Some(candidate_sha256) {
            if selected.is_some() {
                return Err(CliOrchestrationError::msg(format!(
                    "candidate digest {candidate_sha256} occurs more than once in {}",
                    run_dir.join(&ledger.file).display()
                )));
            }
            selected = Some(candidate);
        }
    }
    let candidate = selected.ok_or_else(|| {
        CliOrchestrationError::msg(format!(
            "candidate digest {candidate_sha256} was not found in {}",
            run_dir.join(&ledger.file).display()
        ))
    })?;
    let row_bytes = candidate
        .exact_row_bytes
        .strip_suffix(b"\n")
        .ok_or_else(|| {
            CliOrchestrationError::msg(
                "authenticated candidate row lost its required JSONL terminator",
            )
        })?
        .to_vec();
    Ok(CliAuthenticatedCandidate {
        candidate: candidate.record,
        row_bytes,
    })
}

/// Verify task-specific TriMul verifier provenance on an authenticated CLI row.
///
/// This hidden helper is shared by concrete CLI training selection and the
/// artifact command's launch-bound candidate ingestion.
#[doc(hidden)]
pub fn verify_cli_candidate_verifier_provenance(
    record: &CandidateRecord,
    manifest: &LaunchManifest,
    row_number: usize,
) -> Result<(), CliOrchestrationError> {
    let verifier = manifest.payload.verifier.as_ref().ok_or_else(|| {
        CliOrchestrationError::msg("TriMul launch manifest is missing verifier isolation evidence")
    })?;
    let metadata = record
        .reward_metadata
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CliOrchestrationError::msg(format!(
                "candidate ledger row {row_number} is missing structured verifier reward metadata"
            ))
        })?;
    let expected_tier = verifier.isolation.tier.as_str();
    let expected_metric = crate::trimul::timing_metric_for_tier(verifier.isolation.tier);
    if metadata
        .get("verifier_isolation_tier")
        .and_then(serde_json::Value::as_str)
        != Some(expected_tier)
        || metadata
            .get("verifier_isolation_evidence_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(verifier.isolation_evidence_sha256.as_str())
        || metadata
            .get("timing_metric")
            .and_then(serde_json::Value::as_str)
            != Some(expected_metric)
        || metadata
            .get("runtime_preflight_evidence_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(verifier.runtime_preflight_evidence_sha256.as_str())
    {
        return Err(CliOrchestrationError::msg(format!(
            "candidate ledger row {row_number} verifier tier/evidence does not match launch.json"
        )));
    }
    let extracted = metadata
        .get("submission_extracted")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            CliOrchestrationError::msg(format!(
                "candidate ledger row {row_number} omits submission extraction evidence"
            ))
        })?;
    let executed = metadata
        .get("verification_executed")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            CliOrchestrationError::msg(format!(
                "candidate ledger row {row_number} omits verification execution evidence"
            ))
        })?;
    if extracted != executed {
        return Err(CliOrchestrationError::msg(format!(
            "candidate ledger row {row_number} has inconsistent extraction/execution evidence"
        )));
    }
    if executed {
        let runtime_hardening = metadata
            .get("runtime_hardening")
            .and_then(serde_json::Value::as_array);
        let runtime_digest = runtime_hardening
            .map(|records| runtime_hardening_evidence_sha256(records))
            .transpose()?;
        let expected_runtime = verifier.runtime_preflight.runtime_hardening.first();
        let isolation = serde_json::to_value(&verifier.isolation).map_err(|error| {
            CliOrchestrationError::msg(format!("serialize launch verifier evidence: {error}"))
        })?;
        if metadata.get("verifier_isolation_evidence") != Some(&isolation)
            || runtime_hardening.is_none_or(Vec::is_empty)
            || runtime_hardening.is_some_and(|records| {
                records
                    .iter()
                    .any(|record| Some(record) != expected_runtime)
            })
            || metadata
                .get("runtime_hardening_evidence_sha256")
                .and_then(serde_json::Value::as_str)
                != runtime_digest.as_deref()
        {
            return Err(CliOrchestrationError::msg(format!(
                "candidate ledger row {row_number} omits or changes protected verifier run evidence"
            )));
        }
    }
    Ok(())
}

fn runtime_hardening_evidence_sha256(
    records: &[serde_json::Value],
) -> Result<String, CliOrchestrationError> {
    let encoded = records
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            CliOrchestrationError::msg(format!(
                "serialize candidate runtime hardening evidence: {error}"
            ))
        })?;
    let fields = encoded.iter().map(String::as_bytes).collect::<Vec<_>>();
    Ok(domain_sha256(
        "ferrl.trimul-runtime-hardening-evidence.v1",
        &fields,
    ))
}

fn parse_engine_candidate_row(
    ledger_path: &Path,
    row_number: usize,
    raw_line: &str,
) -> Result<CandidateRecord, EngineError> {
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
    let value: serde_json::Value =
        serde_json::from_str(raw_line).map_err(|source| EngineError::CandidateJson {
            path: ledger_path.to_path_buf(),
            line: row_number,
            source,
        })?;
    let object = value.as_object().ok_or_else(|| {
        EngineError::InvalidCandidateEvidence(format!(
            "candidate ledger {} row {row_number} is not a JSON object",
            ledger_path.display()
        ))
    })?;
    if let Some(field) = object
        .keys()
        .find(|field| !FIELDS.contains(&field.as_str()))
    {
        return Err(EngineError::InvalidCandidateEvidence(format!(
            "candidate ledger {} row {row_number} contains unknown field {field:?}",
            ledger_path.display()
        )));
    }
    serde_json::from_value(value).map_err(|source| EngineError::CandidateJson {
        path: ledger_path.to_path_buf(),
        line: row_number,
        source,
    })
}

fn read_engine_regular_bytes(path: &Path) -> Result<Vec<u8>, EngineError> {
    let path_metadata = fs::symlink_metadata(path).map_err(|source| EngineError::CandidateIo {
        path: path.to_path_buf(),
        source,
    })?;
    if !path_metadata.file_type().is_file() {
        return Err(EngineError::InvalidCandidateEvidence(format!(
            "provenance input {} is not a regular file",
            path.display()
        )));
    }
    let mut file = File::open(path).map_err(|source| EngineError::CandidateIo {
        path: path.to_path_buf(),
        source,
    })?;
    let file_metadata = file.metadata().map_err(|source| EngineError::CandidateIo {
        path: path.to_path_buf(),
        source,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if path_metadata.dev() != file_metadata.dev() || path_metadata.ino() != file_metadata.ino()
        {
            return Err(EngineError::InvalidCandidateEvidence(format!(
                "provenance input {} changed while it was opened",
                path.display()
            )));
        }
    }
    let expected_len = file_metadata.len();
    let mut bytes = Vec::with_capacity(usize::try_from(expected_len).unwrap_or(0));
    file.read_to_end(&mut bytes)
        .map_err(|source| EngineError::CandidateIo {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 != expected_len {
        return Err(EngineError::InvalidCandidateEvidence(format!(
            "provenance input {} changed length while it was captured",
            path.display()
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
fn parse_strict_candidate_row(
    ledger_path: &Path,
    row_number: usize,
    raw_line: &str,
) -> Result<CandidateRecord, CliOrchestrationError> {
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
        CliOrchestrationError::msg(format!(
            "parse candidate ledger {} row {row_number}: {error}",
            ledger_path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        CliOrchestrationError::msg(format!(
            "candidate ledger {} row {row_number} is not a JSON object",
            ledger_path.display()
        ))
    })?;
    if let Some(field) = object
        .keys()
        .find(|field| !FIELDS.contains(&field.as_str()))
    {
        return Err(CliOrchestrationError::msg(format!(
            "candidate ledger {} row {row_number} contains unknown field {field:?}",
            ledger_path.display()
        )));
    }
    serde_json::from_value(value).map_err(|error| {
        CliOrchestrationError::msg(format!(
            "parse candidate ledger {} row {row_number}: {error}",
            ledger_path.display()
        ))
    })
}

fn portable_verifier_consensus(
    identity: &LaunchVerifierIdentity,
) -> Result<Vec<u8>, CliOrchestrationError> {
    serde_json::to_vec(&(
        identity.isolation.contract_version,
        identity.isolation.tier,
        identity.isolation.uid_boundary,
        identity.isolation.asset_transport,
        &identity.isolation.apptainer_sha256,
        identity.isolation.apptainer_len_bytes,
        &identity.isolation.apptainer_version,
        &identity.timing_metric,
        &identity.runtime_hardening_contract,
        identity.runtime_preflight.contract_version,
        &identity.runtime_preflight.probe_submission_sha256,
        &identity.runtime_preflight.runtime_hardening,
    ))
    .map_err(|error| {
        CliOrchestrationError::msg(format!("serialize verifier consensus evidence: {error}"))
    })
}

fn decode_lower_hex(
    label: &str,
    value: &str,
    expected_bytes: usize,
) -> Result<Vec<u8>, CliOrchestrationError> {
    let expected_len = expected_bytes.saturating_mul(2);
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CliOrchestrationError::msg(format!(
            "{label} must be {expected_len} lowercase hexadecimal characters"
        )));
    }
    Ok(value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let nibble = |byte| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => unreachable!("validated lowercase hexadecimal input"),
            };
            (nibble(pair[0]) << 4) | nibble(pair[1])
        })
        .collect())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use candle_core::{Result as CandleResult, Tensor, Var};
    use serde_json::json;

    use crate::policy::Rollout;

    use super::*;

    struct ProbeTokenizer {
        reject: bool,
    }

    impl TokenizerLike for ProbeTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            if self.reject {
                Vec::new()
            } else {
                vec![1]
            }
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    struct ProbePolicy {
        logp: Var,
        adapter_enabled: bool,
        panic_on_tp_preflight_rank_one: bool,
    }

    impl ProbePolicy {
        fn new() -> Self {
            Self {
                logp: Var::from_tensor(
                    &Tensor::zeros((2, 1), candle_core::DType::F32, &Device::Cpu).unwrap(),
                )
                .unwrap(),
                adapter_enabled: true,
                panic_on_tp_preflight_rank_one: false,
            }
        }
    }

    impl Policy for ProbePolicy {
        fn generate(&mut self, prompt: &[u32], config: &GenConfig) -> CandleResult<Rollout> {
            let rows = (0..config.group_size)
                .map(|index| {
                    let mut row = prompt.to_vec();
                    row.push(u32::try_from(index + 1).unwrap());
                    row
                })
                .collect();
            Ok(Rollout::new(
                rows,
                prompt.len(),
                vec![config.max_new_tokens; config.group_size],
                None,
            ))
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            Ok(self.logp.as_tensor().clone())
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.adapter_enabled = enabled;
        }

        fn adapter_enabled(&self) -> bool {
            self.adapter_enabled
        }

        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
            if state.is_empty() {
                Ok(())
            } else {
                candle_core::bail!("probe policy has no sampler state")
            }
        }
    }

    impl TensorParallelPolicy for ProbePolicy {
        fn supports_sharded_tensor_parallel_backward(&self) -> bool {
            true
        }

        fn validate_tensor_parallel_execution(&self, comm: &dyn Comm) -> CandleResult<()> {
            assert!(
                !(self.panic_on_tp_preflight_rank_one && comm.rank() == 1),
                "probe TP preflight panic"
            );
            Ok(())
        }

        fn generate_at_tensor_parallel_instrumented(
            &mut self,
            prompt: &[u32],
            config: &GenConfig,
            _global_row_base: u64,
            _comm: &dyn Comm,
            _telemetry: Option<&mut dyn crate::telemetry::ModelTelemetryRecorder>,
        ) -> CandleResult<Rollout> {
            self.generate(prompt, config)
        }

        fn token_logprobs_tensor_parallel(
            &self,
            rollout: &Rollout,
            _comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            self.token_logprobs(rollout)
        }

        fn token_logprobs_tensor_parallel_detached(
            &self,
            rollout: &Rollout,
            _comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            self.token_logprobs_detached(rollout)
        }
    }

    struct ProbeReward;

    #[derive(Debug, Serialize)]
    struct NonIdempotentTarget(String);

    impl<'de> Deserialize<'de> for NonIdempotentTarget {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = String::deserialize(deserializer)?;
            Ok(Self(format!("normalized:{value}")))
        }
    }

    impl RewardFn for ProbeReward {
        type Target = ();

        fn reward(
            &self,
            _sample: &Sample<Self::Target>,
            completion: &str,
        ) -> Result<f32, crate::reward::RewardError> {
            Ok(if completion == "2" { 1.0 } else { 0.0 })
        }
    }

    fn cli_probe_request<'a>(
        root: &'a Path,
        device: &'a Device,
        training_samples: &'a [Sample<()>],
        reward: &'a ProbeReward,
    ) -> CliTrainingRequest<'a, ProbeReward> {
        CliTrainingRequest {
            launch: CliLaunchInput {
                task: "countdown".into(),
                ferrl_commit: "ab".repeat(20),
                authentication: LaunchAuthenticationMode::LocalEphemeralV1,
                run: LaunchRunIdentity {
                    group_id: "countdown-probe".into(),
                    run_id: "countdown-probe".into(),
                    data_parallel_rank: 0,
                    data_parallel_world_size: 1,
                    tensor_parallel_rank: 0,
                    tensor_parallel_world_size: 1,
                },
                config: LaunchConfigSnapshot {
                    source_sha256: "01".repeat(32),
                    resolved_sha256: "02".repeat(32),
                    resolved: json!({"task": "countdown"}),
                },
                output_root: root.to_path_buf(),
            },
            model_dir: Path::new("probe-model"),
            device,
            loader_opts: LoaderOpts::default(),
            activation_checkpointing: false,
            eos_selection: CliEosSelection::Disabled,
            trainer_config: TrainerConfig::builder()
                .steps(1)
                .group_size(2)
                .max_new_tokens(1)
                .candidate_log_top_k(2)
                .build(),
            training_samples,
            evaluation_samples: &[],
            reward,
            evaluation_reward: reward,
            rendered_prompt_bytes: None,
            verifier_assets: None,
            verifier_identity: None,
            execution: CliExecution::WorldOne,
            health_policy: CliRunHealthPolicy::default(),
            health_policy_is_default: true,
            data_seed: 1,
            trimul_held_out_secret_seed: None,
        }
    }

    fn run_cli_probe<'a>(
        request: CliTrainingRequest<'a, ProbeReward>,
        tokenizer_rejects: bool,
    ) -> Result<(EngineOutcome, ProbePolicy), EngineError> {
        fn supports_tensor_parallel(_: &ProbePolicy) -> bool {
            true
        }
        let plan = EnginePlan::from_cli(request, None);
        ConcreteEngine::new(
            plan,
            Box::new(move |_model_dir, _device, _options| {
                Ok((
                    ProbePolicy::new(),
                    ProbeTokenizer {
                        reject: tokenizer_rejects,
                    },
                    PolicyLoadIdentity {
                        policy_sha256: "11".repeat(32),
                        tokenizer_sha256: "22".repeat(32),
                        model_family: "qwen3",
                    },
                ))
            }),
            supports_tensor_parallel,
            resolve_test_eos::<ProbeTokenizer>,
        )
        .run_with_policy()
    }

    #[test]
    fn tensor_parallel_policy_preflight_panic_is_coordinated_in_lockstep() {
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    scope.spawn(move || {
                        let mut policy = ProbePolicy::new();
                        policy.panic_on_tp_preflight_rank_one = true;
                        validate_tensor_parallel_policy_preflight(&policy, &comm, Some(&comm))
                            .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results[0]
            .as_ref()
            .unwrap_err()
            .contains("failed on a peer distributed rank"));
        assert!(results[1]
            .as_ref()
            .unwrap_err()
            .contains("probe TP preflight panic"));
    }

    #[test]
    fn data_parallel_eval_publishes_on_primary_without_nonprimary_readback() {
        let temporary = TestDir::new("dp-eval-publication");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let output_root = temporary.0.clone();
                    scope.spawn(move || {
                        let samples = vec![Sample::new("probe", ())];
                        let reward = ProbeReward;
                        let device = Device::Cpu;
                        let request = CliTrainingRequest {
                            launch: CliLaunchInput {
                                task: "countdown".into(),
                                ferrl_commit: "ab".repeat(20),
                                authentication: LaunchAuthenticationMode::LocalEphemeralV1,
                                run: LaunchRunIdentity {
                                    group_id: "countdown-dp-probe".into(),
                                    run_id: format!("countdown-dp-probe-rank{rank}"),
                                    data_parallel_rank: rank,
                                    data_parallel_world_size: 2,
                                    tensor_parallel_rank: 0,
                                    tensor_parallel_world_size: 1,
                                },
                                config: LaunchConfigSnapshot {
                                    source_sha256: "01".repeat(32),
                                    resolved_sha256: "02".repeat(32),
                                    resolved: json!({"task": "countdown"}),
                                },
                                output_root,
                            },
                            model_dir: Path::new("probe-model"),
                            device: &device,
                            loader_opts: LoaderOpts::default(),
                            activation_checkpointing: false,
                            eos_selection: CliEosSelection::Disabled,
                            trainer_config: TrainerConfig::builder()
                                .steps(1)
                                .group_size(2)
                                .max_new_tokens(1)
                                .candidate_log_top_k(2)
                                .build(),
                            training_samples: &samples,
                            evaluation_samples: &samples,
                            reward: &reward,
                            evaluation_reward: &reward,
                            rendered_prompt_bytes: None,
                            verifier_assets: None,
                            verifier_identity: None,
                            execution: CliExecution::DataParallel(Box::new(comm)),
                            health_policy: CliRunHealthPolicy::default(),
                            health_policy_is_default: true,
                            data_seed: 1,
                            trimul_held_out_secret_seed: None,
                        };
                        run_cli_probe(request, false)
                            .and_then(|(outcome, _policy)| match outcome {
                                EngineOutcome::Completed(completed) => {
                                    Ok(completed.run.root().to_path_buf())
                                }
                                EngineOutcome::Preempted(_) => Err(EngineError::Configuration(
                                    "DP evaluation probe unexpectedly preempted".into(),
                                )),
                            })
                            .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let rank_zero = results[0].as_ref().unwrap();
        let rank_one = results[1].as_ref().unwrap();
        assert!(rank_zero.join(RunDir::EVAL_REPORT_FILE).is_file());
        assert!(!rank_one.join(RunDir::EVAL_REPORT_FILE).exists());
    }

    #[test]
    fn data_parallel_health_failure_aborts_every_rank_in_lockstep() {
        let temporary = TestDir::new("dp-health-lockstep");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let output_root = temporary.0.clone();
                    scope.spawn(move || {
                        let samples = vec![Sample::new("probe", ())];
                        let reward = ProbeReward;
                        let device = Device::Cpu;
                        let health_policy = if rank == 1 {
                            CliRunHealthPolicy::from_json_value(json!({
                                "reward_collapse": {
                                    "window": 1,
                                    "min": 2.0,
                                    "action": "fail"
                                }
                            }))
                            .unwrap()
                        } else {
                            CliRunHealthPolicy::default()
                        };
                        let request = CliTrainingRequest {
                            launch: CliLaunchInput {
                                task: "countdown".into(),
                                ferrl_commit: "ab".repeat(20),
                                authentication: LaunchAuthenticationMode::LocalEphemeralV1,
                                run: LaunchRunIdentity {
                                    group_id: "countdown-dp-health-probe".into(),
                                    run_id: format!("countdown-dp-health-probe-rank{rank}"),
                                    data_parallel_rank: rank,
                                    data_parallel_world_size: 2,
                                    tensor_parallel_rank: 0,
                                    tensor_parallel_world_size: 1,
                                },
                                config: LaunchConfigSnapshot {
                                    source_sha256: "01".repeat(32),
                                    resolved_sha256: "02".repeat(32),
                                    resolved: json!({"task": "countdown"}),
                                },
                                output_root,
                            },
                            model_dir: Path::new("probe-model"),
                            device: &device,
                            loader_opts: LoaderOpts::default(),
                            activation_checkpointing: false,
                            eos_selection: CliEosSelection::Disabled,
                            trainer_config: TrainerConfig::builder()
                                .steps(1)
                                .group_size(2)
                                .max_new_tokens(1)
                                .candidate_log_top_k(2)
                                .build(),
                            training_samples: &samples,
                            evaluation_samples: &samples,
                            reward: &reward,
                            evaluation_reward: &reward,
                            rendered_prompt_bytes: None,
                            verifier_assets: None,
                            verifier_identity: None,
                            execution: CliExecution::DataParallel(Box::new(comm)),
                            health_policy,
                            health_policy_is_default: rank == 0,
                            data_seed: 1,
                            trimul_held_out_secret_seed: None,
                        };
                        run_cli_probe(request, false).map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        match &results[0] {
            Err(error) => assert!(error.contains("failed on a peer distributed rank")),
            Ok(_) => panic!("rank zero unexpectedly passed asymmetric DP health"),
        }
        match &results[1] {
            Err(error) => assert!(error.contains("run_health policy failed")),
            Ok(_) => panic!("rank one unexpectedly passed failing DP health"),
        }
    }

    #[test]
    fn cli_adapter_traverses_the_same_concrete_engine_and_short_circuits_preflight() {
        let temporary = TestDir::new("cli-shared-engine");
        let samples = vec![Sample::new("probe", ())];
        let reward = ProbeReward;
        let device = Device::Cpu;
        let (outcome, _policy) = run_cli_probe(
            cli_probe_request(temporary.0.as_path(), &device, &samples, &reward),
            false,
        )
        .unwrap();
        let EngineOutcome::Completed(completed) = outcome else {
            panic!("probe CLI run unexpectedly preempted");
        };
        assert!(completed.run.root().join(RunDir::LAUNCH_FILE).is_file());
        assert!(!completed.candidates.is_empty());

        let rejected_root = temporary.0.join("rejected");
        let Err(error) = run_cli_probe(
            cli_probe_request(&rejected_root, &device, &samples, &reward),
            true,
        ) else {
            panic!("rejected tokenizer unexpectedly passed preflight");
        };
        assert!(error.to_string().contains("encoded to zero tokens"));
        assert!(!rejected_root.exists());
    }

    #[test]
    fn sdk_and_cli_adapters_share_preflight_and_fail_before_mutation() {
        let temporary = TestDir::new("adapter-preflight-parity");
        let samples = vec![Sample::new("probe", ())];
        let reward = ProbeReward;
        let device = Device::Cpu;

        let cli_root = temporary.0.join("cli-rejected");
        let Err(cli_error) = run_cli_probe(
            cli_probe_request(&cli_root, &device, &samples, &reward),
            true,
        ) else {
            panic!("CLI adapter unexpectedly bypassed shared tokenizer preflight");
        };

        let discovery_root = temporary.0.join("sdk-rejected");
        let task = crate::discovery::TaskIdentity::new("probe", 1).unwrap();
        let request = DiscoveryTrainingRequest {
            model_dir: Path::new("probe-model"),
            device: &device,
            loader_opts: LoaderOpts::default(),
            eos_selection: EngineEosSelection::Disabled,
            trainer_config: TrainerConfig::builder()
                .steps(1)
                .group_size(2)
                .max_new_tokens(1)
                .candidate_log_top_k(2)
                .build(),
            training_samples: &samples,
            evaluation_samples: &[],
            reward: &reward,
            evaluation_reward: &reward,
            launch: DiscoveryLaunchInput {
                task: &task,
                metric_contract: crate::discovery::MetricContract::new(
                    "probe-score",
                    "points",
                    crate::discovery::MetricDirection::HigherIsBetter,
                    0.0,
                    0.0,
                ),
                ferrl_source: crate::discovery::BuildSourceIdentity::for_orchestration_test(),
                execution_device: crate::discovery::ExecutionDevice::Cpu,
                runs_root: &discovery_root,
                steps: 1,
                group_size: 2,
                max_new_tokens: 1,
                eval_group_size: 1,
                temperature: 1.0,
                learning_rate: 1e-3,
                seed: 1,
                preemption_flag: None,
            },
        };
        let Err(discovery_error) = run_discovery_with_test_loader(
            request,
            |_model_dir, _device, _options| {
                Ok((
                    ProbePolicy::new(),
                    ProbeTokenizer { reject: true },
                    PolicyLoadIdentity {
                        policy_sha256: "11".repeat(32),
                        tokenizer_sha256: "22".repeat(32),
                        model_family: "qwen3",
                    },
                ))
            },
            |_| true,
        ) else {
            panic!("SDK adapter unexpectedly bypassed shared tokenizer preflight");
        };

        assert!(cli_error.to_string().contains("encoded to zero tokens"));
        assert!(discovery_error
            .to_string()
            .contains("encoded to zero tokens"));
        assert!(!cli_root.exists());
        assert!(!discovery_root.exists());
    }

    fn cli_train_setup_for_test(root: &Path) -> CliTrainSetup {
        CliTrainSetup {
            task: CliBuiltinTask::Countdown {
                train_n: 1,
                eval_n: 0,
                seed: 7,
            },
            ferrl_commit: "ab".repeat(20),
            authentication: LaunchAuthenticationMode::LocalEphemeralV1,
            launch_config: LaunchConfigSnapshot {
                source_sha256: "01".repeat(32),
                resolved_sha256: "02".repeat(32),
                resolved: json!({"task": "countdown"}),
            },
            config_consensus_digest: [3; 32],
            model_dir: root.join("missing-model"),
            output_root: root.join("runs"),
            device: CliDeviceSelection::Cpu,
            loader_opts: LoaderOpts::default(),
            activation_checkpointing: false,
            eos_selection: CliEosSelection::Disabled,
            trainer_config: TrainerConfig::builder()
                .steps(1)
                .group_size(2)
                .max_new_tokens(1)
                .candidate_log_top_k(2)
                .build(),
            data_parallel: false,
            tensor_parallel_plan: TensorParallelPlan::single(),
            health_policy: CliRunHealthPolicy::default(),
            health_policy_is_default: true,
        }
    }

    #[test]
    fn complete_cli_train_setup_owns_identity_task_and_request_assembly() {
        let temporary = TestDir::new("complete-cli-setup");
        let setup = cli_train_setup_for_test(&temporary.0);
        let error = run_cli_train_setup(&setup, None, None).unwrap_err();
        assert!(error.to_string().contains("model load"));
        assert!(!setup.output_root.exists());

        let mut invalid = cli_train_setup_for_test(&temporary.0);
        invalid.ferrl_commit = "not-a-commit".into();
        assert!(run_cli_train_setup(&invalid, None, None)
            .unwrap_err()
            .to_string()
            .contains("full 40- or 64-character"));

        let mut math = cli_train_setup_for_test(&temporary.0);
        math.task = CliBuiltinTask::Math {
            path: temporary.0.join("missing.jsonl"),
            eval_n: 0,
            seed: 7,
        };
        assert!(run_cli_train_setup(&math, None, None)
            .unwrap_err()
            .to_string()
            .contains("missing.jsonl"));

        let mut trimul = cli_train_setup_for_test(&temporary.0);
        trimul.task = CliBuiltinTask::Trimul(Box::new(CliTrimulTask {
            prompt_path: temporary.0.join("missing-prompt.txt"),
            submission_extract_mode: crate::trimul::SubmissionExtractMode::FinalFence,
            image: temporary.0.join("image.sif"),
            eval_dir: temporary.0.join("eval"),
            scratch_root: temporary.0.join("scratch"),
            verifier_isolation_tier: crate::VerifierIsolationTier::SameUidApptainerV1,
            verifier_apptainer_bin: None,
            verifier_executor_socket: None,
            scratch_max_bytes: 0,
            secret_seed: 1,
            held_out_secret_seed: None,
            wall_secs: 0,
            verifier_cuda_visible_devices: None,
            verifier_cuda_device_pool: Vec::new(),
            verifier_parallelism: 0,
            verifier_max_procs: 0,
            baseline: None,
            reward_profile: crate::trimul::TrimulRewardProfile::default(),
            train_n: 1,
            eval_n: 0,
            data_seed: 7,
        }));
        assert!(run_cli_train_setup(&trimul, None, None)
            .unwrap_err()
            .to_string()
            .contains("missing-prompt.txt"));
    }

    #[test]
    fn asymmetric_cli_training_commit_validation_fails_before_mutation() {
        let temporary = TestDir::new("distributed-cli-commit-consensus");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let mut setup = cli_train_setup_for_test(&temporary.0);
                    setup.data_parallel = true;
                    setup.device = CliDeviceSelection::Cuda;
                    if rank == 1 {
                        setup.ferrl_commit = "not-a-commit".into();
                    }
                    scope.spawn(move || {
                        run_cli_train_setup(
                            &setup,
                            Some(CliLaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            None,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().any(|result| matches!(
            result,
            Err(error) if error.contains("full 40- or 64-character")
        )));
        assert!(results.iter().any(|result| matches!(
            result,
            Err(error) if error.contains("training commit validation failed on a peer")
        )));
        assert!(!temporary.0.join("runs").exists());
    }

    #[test]
    fn asymmetric_cli_config_digest_fails_before_device_or_model_setup() {
        let temporary = TestDir::new("distributed-cli-config-consensus");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let mut setup = cli_train_setup_for_test(&temporary.0);
                    setup.data_parallel = true;
                    setup.device = CliDeviceSelection::Cuda;
                    setup.config_consensus_digest = [rank as u8; 32];
                    scope.spawn(move || {
                        run_cli_train_setup(
                            &setup,
                            Some(CliLaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            None,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(|result| matches!(
            result,
            Err(error) if error.contains("run config outside tensor_parallel.rank")
        )));
        assert!(!temporary.0.join("runs").exists());
    }

    #[test]
    fn asymmetric_cli_device_setup_fails_in_lockstep_before_mutation() {
        let temporary = TestDir::new("distributed-cli-device-setup");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let mut setup = cli_train_setup_for_test(&temporary.0);
                    setup.data_parallel = true;
                    setup.device = if rank == 0 {
                        CliDeviceSelection::Cuda
                    } else {
                        CliDeviceSelection::Cpu
                    };
                    scope.spawn(move || {
                        run_cli_train_setup(
                            &setup,
                            Some(CliLaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            None,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().any(|result| matches!(
            result,
            Err(error) if error.contains("requires device = \"cuda\"")
        )));
        assert!(results.iter().any(|result| matches!(
            result,
            Err(error) if error.contains("CLI device setup failed on a peer")
        )));
        assert!(!temporary.0.join("runs").exists());
    }

    #[test]
    fn asymmetric_cli_task_setup_fails_in_lockstep_before_mutation() {
        let temporary = TestDir::new("distributed-cli-task-setup");
        let math_path = temporary.0.join("math.jsonl");
        std::fs::write(
            &math_path,
            br#"{"prompt":"1 + 1 = ?","target":{"answer":"2"}}
"#,
        )
        .unwrap();
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let mut setup = cli_train_setup_for_test(&temporary.0);
                    setup.data_parallel = true;
                    setup.device = CliDeviceSelection::Cuda;
                    setup.task = CliBuiltinTask::Math {
                        path: if rank == 0 {
                            math_path.clone()
                        } else {
                            temporary.0.join("missing-math.jsonl")
                        },
                        eval_n: 0,
                        seed: 7,
                    };
                    scope.spawn(move || {
                        run_cli_train_setup(
                            &setup,
                            Some(CliLaunchRuntime {
                                device: Device::Cpu,
                                comm: Box::new(comm),
                            }),
                            None,
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().any(|result| matches!(
            result,
            Err(error) if error.contains("missing-math.jsonl")
        )));
        assert!(results.iter().any(|result| matches!(
            result,
            Err(error) if error.contains("Math task setup failed on a peer")
        )));
        assert!(!temporary.0.join("runs").exists());
    }

    #[test]
    fn synchronized_cli_identity_uses_one_distributed_timestamp() {
        let temporary = TestDir::new("distributed-cli-identity");
        let identities = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .map(|comm| {
                    let mut setup = cli_train_setup_for_test(&temporary.0);
                    setup.data_parallel = true;
                    scope.spawn(move || synchronized_cli_run_identity(&setup, Some(&comm)).unwrap())
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(identities[0].group_id, identities[1].group_id);
        assert_eq!(identities[0].data_parallel_rank, 0);
        assert_eq!(identities[1].data_parallel_rank, 1);
        assert_eq!(identities[0].data_parallel_world_size, 2);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // one compact contract matrix
    fn cli_setup_identity_and_device_helpers_cover_world_one_and_distributed_contracts() {
        let temporary = TestDir::new("cli-setup-helpers");
        let setup = cli_train_setup_for_test(&temporary.0);
        let identity = synchronized_cli_run_identity(&setup, None).unwrap();
        assert_eq!(identity.data_parallel_world_size, 1);
        assert_eq!(identity.tensor_parallel_world_size, 1);
        assert!(matches!(
            prepare_cli_device(CliDeviceSelection::Cpu, None).unwrap(),
            Device::Cpu
        ));
        assert!(prepare_cli_device(
            CliDeviceSelection::Cpu,
            Some(&CliLaunchRuntime {
                device: Device::Cpu,
                comm: Box::new(crate::SoloComm),
            })
        )
        .unwrap_err()
        .to_string()
        .contains("requires device = \"cuda\""));
        assert!(matches!(
            prepare_cli_device(
                CliDeviceSelection::Cuda,
                Some(&CliLaunchRuntime {
                    device: Device::Cpu,
                    comm: Box::new(crate::SoloComm),
                })
            )
            .unwrap(),
            Device::Cpu
        ));
        let mut unexpected_runtime = cli_train_setup_for_test(&temporary.0);
        unexpected_runtime.device = CliDeviceSelection::Cuda;
        assert!(run_cli_train_setup(
            &unexpected_runtime,
            Some(CliLaunchRuntime {
                device: Device::Cpu,
                comm: Box::new(crate::SoloComm),
            }),
            None,
        )
        .unwrap_err()
        .to_string()
        .contains("unexpected distributed launch runtime"));
        let mut missing_runtime = cli_train_setup_for_test(&temporary.0);
        missing_runtime.tensor_parallel_plan = TensorParallelPlan::new(0, 2).unwrap();
        assert!(run_cli_train_setup(&missing_runtime, None, None)
            .unwrap_err()
            .to_string()
            .contains("requires a live launch runtime"));
        assert!(validate_lower_digest("digest", &"ab".repeat(32)).is_ok());
        assert!(validate_lower_digest("digest", "AB").is_err());
        assert!(guard_cli_baseline_gpu("").is_err());
    }

    #[test]
    fn shared_engine_phase_helpers_are_fail_closed_before_mutation() {
        let samples = vec![Sample::new("prompt", ())];
        let (_, bytes) = exact_execution_samples(&samples, "test samples").unwrap();
        assert!(!bytes.is_empty());
        assert!(
            preflight_prompt_tokenization(&samples, "test samples", &OneTokenTokenizer,).is_ok()
        );
        assert!(preflight_prompt_tokenization(&samples, "test samples", &EmptyTokenizer,).is_err());
    }

    #[test]
    fn asymmetric_data_parallel_training_samples_fail_before_mutation() {
        let temporary = TestDir::new("dp-sample-consensus");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = temporary.0.clone();
                    scope.spawn(move || {
                        let samples = vec![Sample::new(format!("rank-{rank}"), ())];
                        let reward = ProbeReward;
                        let device = Device::Cpu;
                        let mut request = cli_probe_request(&root, &device, &samples, &reward);
                        request.launch.run = LaunchRunIdentity {
                            group_id: "dp-sample-consensus".into(),
                            run_id: format!("dp-sample-consensus-rank{rank}"),
                            data_parallel_rank: rank,
                            data_parallel_world_size: 2,
                            tensor_parallel_rank: 0,
                            tensor_parallel_world_size: 1,
                        };
                        request.execution = CliExecution::DataParallel(Box::new(comm));
                        run_cli_probe(request, false)
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(|result| matches!(
            result,
            Err(error) if error.contains("launch ranks disagree on ordered training samples")
        )));
        assert!(!temporary.0.join("dp-sample-consensus-rank0").exists());
        assert!(!temporary.0.join("dp-sample-consensus-rank1").exists());
    }

    #[test]
    fn asymmetric_tensor_parallel_held_out_samples_fail_before_mutation() {
        let temporary = TestDir::new("tp-sample-consensus");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(3))
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = temporary.0.clone();
                    scope.spawn(move || {
                        let training = vec![Sample::new("same", ())];
                        let evaluation = vec![Sample::new(format!("held-rank-{rank}"), ())];
                        let reward = ProbeReward;
                        let device = Device::Cpu;
                        let mut request = cli_probe_request(&root, &device, &training, &reward);
                        request.evaluation_samples = &evaluation;
                        request.launch.run = LaunchRunIdentity {
                            group_id: "tp-sample-consensus".into(),
                            run_id: format!("tp-sample-consensus-rank{rank}"),
                            data_parallel_rank: 0,
                            data_parallel_world_size: 1,
                            tensor_parallel_rank: rank,
                            tensor_parallel_world_size: 2,
                        };
                        request.execution = CliExecution::TensorParallel {
                            plan: TensorParallelPlan::new(rank, 2).unwrap(),
                            comm: Box::new(comm),
                        };
                        run_cli_probe(request, false)
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(
            results.iter().all(|result| matches!(
                result,
                Err(error) if error.contains("launch ranks disagree on ordered held-out samples")
            )),
            "{results:?}"
        );
        assert!(!temporary.0.join("tp-sample-consensus-rank0").exists());
        assert!(!temporary.0.join("tp-sample-consensus-rank1").exists());
    }

    #[test]
    fn discovery_report_reuses_frozen_digest_for_non_idempotent_targets() {
        let temporary = TestDir::new("non-idempotent-held-out");
        let original = vec![Sample::new("prompt", NonIdempotentTarget("value".into()))];
        let (reconstructed, frozen_bytes) =
            exact_execution_samples(&original, "non-idempotent held-out samples").unwrap();
        let frozen_sha256 = sha256_hex(&frozen_bytes);
        assert_ne!(
            frozen_sha256,
            sha256_hex(&serde_json::to_vec(&reconstructed).unwrap())
        );
        let run = RunDir::create(&temporary.0, "report").unwrap();
        let task = crate::discovery::TaskIdentity::new("non-idempotent", 1).unwrap();
        let spec = DiscoveryLaunchInput {
            task: &task,
            metric_contract: crate::discovery::MetricContract::new(
                "score",
                "points",
                crate::discovery::MetricDirection::HigherIsBetter,
                0.0,
                0.0,
            ),
            ferrl_source: crate::discovery::BuildSourceIdentity::for_orchestration_test(),
            execution_device: crate::discovery::ExecutionDevice::Cpu,
            runs_root: &temporary.0,
            steps: 1,
            group_size: 2,
            max_new_tokens: 1,
            eval_group_size: 1,
            temperature: 1.0,
            learning_rate: 1e-3,
            seed: 1,
            preemption_flag: None,
        };
        let report = crate::eval::EvalReport {
            n_prompts: 1,
            group_size: 1,
            base_reward_mean: 0.0,
            adapter_reward_mean: 0.0,
            per_prompt: Vec::new(),
        };
        let bytes = publish_discovery_eval_report(
            &spec,
            &reconstructed,
            &frozen_sha256,
            &report,
            &run,
            &"ab".repeat(32),
        )
        .unwrap();
        let durable: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(durable["held_out_samples_sha256"], frozen_sha256);
    }

    #[test]
    fn preemption_checkpoint_errors_preserve_dedicated_classification() {
        let missing = validate_preemption_checkpoint(2, None).unwrap_err();
        assert!(matches!(missing, EngineError::PreemptionCheckpoint(_)));
        let mismatched = validate_preemption_checkpoint(
            2,
            Some(crate::LatestCheckpoint {
                dir: PathBuf::from("step-1"),
                step: 1,
            }),
        )
        .unwrap_err();
        assert!(matches!(mismatched, EngineError::PreemptionCheckpoint(_)));
        assert!(matches!(
            crate::discovery::DiscoveryError::from(mismatched),
            crate::discovery::DiscoveryError::PreemptionCheckpoint(_)
        ));
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "ferrl-orchestration-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn candidate_test_manifest(
        rank: usize,
        world_size: usize,
        signing_public_key: String,
    ) -> LaunchManifest {
        LaunchManifest::new(LaunchPayload {
            task: "countdown".to_owned(),
            ferrl_commit: "01".repeat(20),
            authentication: LaunchAuthenticationMode::LocalEphemeralV1,
            run: LaunchRunIdentity {
                group_id: "candidate-test".to_owned(),
                run_id: format!("candidate-test-rank{rank}"),
                data_parallel_rank: 0,
                data_parallel_world_size: 1,
                tensor_parallel_rank: rank,
                tensor_parallel_world_size: world_size,
            },
            config: LaunchConfigSnapshot {
                source_sha256: "02".repeat(32),
                resolved_sha256: "03".repeat(32),
                resolved: serde_json::json!({"task": "countdown"}),
            },
            model: LaunchModelIdentity {
                family: "qwen3".to_owned(),
                checkpoint_policy_sha256: "04".repeat(32),
                tokenizer_sha256: "05".repeat(32),
                resolved_eos_token_id: None,
            },
            prompt: None,
            training_samples: Some(LaunchSampleIdentity {
                sha256: "06".repeat(32),
                count: 1,
            }),
            held_out_samples: Some(LaunchSampleIdentity {
                sha256: "07".repeat(32),
                count: 0,
            }),
            verifier: None,
            candidate_ledger: LaunchCandidateLedger {
                file: RunDir::CANDIDATES_FILE.to_owned(),
                format_version: 1,
                row_digest_domain: CANDIDATE_RECORD_DOMAIN.to_owned(),
                row_signature_algorithm: "ed25519".to_owned(),
                signing_public_key,
            },
        })
        .unwrap()
    }

    #[test]
    fn current_launch_manifest_requires_sample_identities() {
        let signer = CandidateSigner::generate().unwrap();
        let mut payload = candidate_test_manifest(0, 1, signer.public_key_hex()).payload;
        payload.training_samples = None;
        let error = LaunchManifest::new(payload).unwrap_err().to_string();
        assert!(
            error.contains("requires ordered training and held-out sample identities"),
            "{error}"
        );
    }

    #[test]
    fn launch_sample_identity_has_a_strict_stable_wire_shape() {
        let identity = LaunchSampleIdentity {
            sha256: "ab".repeat(32),
            count: 3,
        };
        let value = serde_json::to_value(&identity).unwrap();
        assert_eq!(value["sha256"], "ab".repeat(32));
        assert_eq!(value["count"], 3);
        let decoded: LaunchSampleIdentity = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.sha256, identity.sha256);
        assert_eq!(decoded.count, identity.count);
        assert!(serde_json::from_value::<LaunchSampleIdentity>(json!({
            "sha256": "ab".repeat(32),
            "count": 3,
            "unexpected": true
        }))
        .is_err());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit launch-field mutation matrix
    fn cli_launch_identity_binds_both_ordered_sample_digests() {
        let signer = CandidateSigner::generate().unwrap();
        let original = candidate_test_manifest(0, 1, signer.public_key_hex());
        let training = original.payload.training_samples.as_ref().unwrap();
        assert_eq!(training.sha256, "06".repeat(32));
        assert_eq!(training.count, 1);
        let held_out = original.payload.held_out_samples.as_ref().unwrap();
        assert_eq!(held_out.sha256, "07".repeat(32));
        assert_eq!(held_out.count, 0);

        let mut changed_training = candidate_test_manifest(0, 1, signer.public_key_hex());
        changed_training
            .payload
            .training_samples
            .as_mut()
            .unwrap()
            .sha256 = "08".repeat(32);
        let changed_training = LaunchManifest::new(changed_training.payload).unwrap();
        assert_ne!(original.payload_sha256, changed_training.payload_sha256);

        let mut changed_held_out = candidate_test_manifest(0, 1, signer.public_key_hex());
        let held_out = changed_held_out.payload.held_out_samples.as_mut().unwrap();
        held_out.sha256 = "09".repeat(32);
        held_out.count = 1;
        let changed_held_out = LaunchManifest::new(changed_held_out.payload).unwrap();
        assert_ne!(original.payload_sha256, changed_held_out.payload_sha256);

        let bytes = original.to_pretty_bytes().unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["payload"]["training_samples"]["count"], 1);
        assert_eq!(value["payload"]["held_out_samples"]["count"], 0);
    }

    fn candidate_test_config() -> TrainerConfig {
        TrainerConfig::builder()
            .steps(1)
            .group_size(2)
            .max_new_tokens(1)
            .build()
    }

    #[derive(Debug)]
    struct EmptyTokenizer;

    impl TokenizerLike for EmptyTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            Vec::new()
        }

        fn decode(&self, _ids: &[u32]) -> String {
            String::new()
        }
    }

    #[derive(Debug)]
    struct OneTokenTokenizer;

    impl TokenizerLike for OneTokenTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            vec![42]
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        }
    }

    fn trimul_verifier_identity() -> LaunchVerifierIdentity {
        let isolation = crate::VerifierIsolationEvidence {
            contract_version: crate::verifier_executor::VERIFIER_ISOLATION_EVIDENCE_VERSION,
            tier: crate::VerifierIsolationTier::SameUidApptainerV1,
            requester_uid: 1001,
            launcher_uid: 1001,
            uid_boundary: crate::VerifierUidBoundary::SameHostUid,
            asset_transport: crate::VerifierAssetTransport::InProcessSealedCopy,
            apptainer_path: "/usr/bin/apptainer".into(),
            apptainer_sha256: "11".repeat(32),
            apptainer_len_bytes: 2048,
            apptainer_version: "apptainer v1".to_owned(),
            work_root: std::env::temp_dir().join("ferrl-trimul-work-root"),
            work_root_uid: 1001,
            work_root_device: 7,
            work_root_inode: 42,
            work_root_mode: 0o700,
        };
        let mut runtime_preflight = crate::trimul::TrimulRuntimePreflightEvidence {
            contract_version: 1,
            isolation_tier: crate::VerifierIsolationTier::SameUidApptainerV1,
            isolation_evidence_sha256: crate::trimul::verifier_isolation_evidence_sha256(
                &isolation,
            ),
            probe_submission_sha256: "22".repeat(32),
            runtime_hardening: Vec::new(),
            runtime_hardening_evidence_sha256: String::new(),
        };
        let runtime_preflight_evidence_sha256 =
            crate::trimul::runtime_preflight_evidence_sha256(&runtime_preflight);
        runtime_preflight.runtime_hardening_evidence_sha256 =
            runtime_preflight_evidence_sha256.clone();
        LaunchVerifierIdentity {
            assets: crate::trimul::TrimulVerifierIdentity {
                image_sha256: "33".repeat(32),
                image_len_bytes: 4096,
                eval_bundle_sha256: "44".repeat(32),
                eval_file_count: 2,
                task_yml_sha256: "55".repeat(32),
                task_yml_len_bytes: 128,
            },
            isolation: isolation.clone(),
            isolation_evidence_sha256: crate::trimul::verifier_isolation_evidence_sha256(
                &isolation,
            ),
            timing_metric: crate::trimul::timing_metric_for_tier(isolation.tier).to_owned(),
            runtime_hardening_contract: crate::trimul::TRIMUL_RUNTIME_HARDENING_CONTRACT.to_owned(),
            runtime_preflight,
            runtime_preflight_evidence_sha256,
        }
    }

    fn trimul_candidate_test_manifest(
        rank: usize,
        world_size: usize,
        signing_public_key: String,
        include_verifier: bool,
    ) -> LaunchManifest {
        let mut payload = candidate_test_manifest(rank, world_size, signing_public_key).payload;
        payload.task = "trimul".to_owned();
        payload.run.data_parallel_rank = rank;
        payload.run.data_parallel_world_size = world_size;
        payload.run.tensor_parallel_rank = 0;
        payload.run.tensor_parallel_world_size = 1;
        payload.verifier = include_verifier.then_some(trimul_verifier_identity());
        LaunchManifest::new(payload).unwrap()
    }

    fn trimul_candidate_metadata(
        manifest: &LaunchManifest,
        extracted: bool,
        executed: bool,
    ) -> serde_json::Value {
        let verifier = manifest
            .payload
            .verifier
            .as_ref()
            .expect("verifier identity is present");
        serde_json::json!({
            "verifier_isolation_tier": verifier.isolation.tier.as_str(),
            "verifier_isolation_evidence_sha256": verifier.isolation_evidence_sha256,
            "timing_metric": verifier.timing_metric,
            "runtime_preflight_evidence_sha256": verifier.runtime_preflight_evidence_sha256,
            "submission_extracted": extracted,
            "verification_executed": executed,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn signed_candidate_row(
        signer: &CandidateSigner,
        manifest: &LaunchManifest,
        step: u64,
        rank: usize,
        world_size: usize,
        reward: f32,
        completion_len_tokens: usize,
        completion: &str,
        reward_metadata: Option<serde_json::Value>,
    ) -> Vec<u8> {
        let mut record = CandidateRecord::new(
            step,
            rank,
            world_size,
            0,
            0,
            reward,
            completion_len_tokens,
            completion.to_owned(),
        );
        record.reward_metadata = reward_metadata;
        let record = signer
            .sign_candidate(&record, &manifest.payload_sha256)
            .unwrap();
        let mut row = serde_json::to_vec(&record).unwrap();
        row.push(b'\n');
        row
    }

    #[test]
    fn immutable_launch_candidate_topology_prefers_sharded_tensor_parallel() {
        let manifest = candidate_test_manifest(1, 2, "00".repeat(32));
        assert_eq!(launch_candidate_topology(&manifest.payload.run), (1, 2));
    }

    #[test]
    fn tp_candidate_enabled_success_uses_tensor_parallel_rank_and_world() {
        let temporary = TestDir::new("tp-candidate-enabled");
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    let root = temporary.0.clone();
                    scope.spawn(move || {
                        let rank = comm.rank();
                        let signer = CandidateSigner::generate().unwrap();
                        let manifest = candidate_test_manifest(rank, 2, signer.public_key_hex());
                        let run = RunDir::create(&root, format!("rank-{rank}")).unwrap();
                        let record = signer
                            .sign_candidate(
                                &CandidateRecord::new(
                                    0,
                                    rank,
                                    2,
                                    0,
                                    0,
                                    1.0,
                                    1,
                                    "candidate".to_owned(),
                                ),
                                &manifest.payload_sha256,
                            )
                            .unwrap();
                        let mut row = serde_json::to_vec(&record).unwrap();
                        row.push(b'\n');
                        fs::write(run.candidates_path(), row).unwrap();
                        let local = load_cli_candidate_selection(
                            &run,
                            &manifest,
                            &candidate_test_config(),
                            CandidateTopology {
                                rank,
                                world_size: 2,
                            },
                        );
                        let selected = coordinate_cli_result(
                            Some(&comm),
                            "candidate loading and selection",
                            local,
                        );
                        (rank, selected.map(|value| value.is_some()))
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0, "{results:?}");
        assert_eq!(results[1].0, 1, "{results:?}");
        assert!(
            results[0].1.as_ref().is_ok_and(|selected| *selected),
            "{results:?}"
        );
        assert!(
            results[1].1.as_ref().is_ok_and(|selected| *selected),
            "{results:?}"
        );
    }

    #[test]
    fn launch_candidate_topology_prefers_data_parallel_when_tensor_parallel_is_not_sharded() {
        let mut manifest = candidate_test_manifest(3, 1, "00".repeat(32));
        manifest.payload.run.tensor_parallel_rank = 3;
        manifest.payload.run.tensor_parallel_world_size = 1;
        manifest.payload.run.data_parallel_rank = 3;
        manifest.payload.run.data_parallel_world_size = 7;
        assert_eq!(launch_candidate_topology(&manifest.payload.run), (3, 7));
    }

    #[test]
    fn launch_manifest_attest_can_be_applied_once() {
        #[derive(Debug)]
        struct FixedAttestor;

        impl CliLaunchAttestor for FixedAttestor {
            fn attest(
                &self,
                manifest: &LaunchManifest,
            ) -> Result<LaunchAttestation, CliOrchestrationError> {
                Ok(LaunchAttestation {
                    contract_version: LAUNCH_ATTESTATION_CONTRACT_VERSION,
                    kind: LAUNCH_ATTESTATION_KIND.to_owned(),
                    algorithm: LAUNCH_ATTESTATION_ALGORITHM.to_owned(),
                    key_id: "test-attestor".to_owned(),
                    launch_payload_sha256: manifest.payload_sha256.clone(),
                    signature: "aa".repeat(64),
                })
            }
        }

        let manifest = candidate_test_manifest(0, 1, "11".repeat(32)).attest(&FixedAttestor);
        assert!(manifest.is_ok());
        let manifest = manifest.unwrap();
        assert!(manifest.attestation.is_some());

        let duplicate = manifest.attest(&FixedAttestor);
        assert!(duplicate.is_err());
        assert!(duplicate
            .unwrap_err()
            .to_string()
            .contains("already attested"));
    }

    #[test]
    fn launch_manifest_attest_bubbles_attestor_failures() {
        #[derive(Debug)]
        struct RejectingAttestor;

        impl CliLaunchAttestor for RejectingAttestor {
            fn attest(
                &self,
                _manifest: &LaunchManifest,
            ) -> Result<LaunchAttestation, CliOrchestrationError> {
                Err(CliOrchestrationError::msg("attestor rejected launch"))
            }
        }

        let error = candidate_test_manifest(0, 1, "22".repeat(32)).attest(&RejectingAttestor);
        assert!(error.is_err());
        assert!(error
            .unwrap_err()
            .to_string()
            .contains("attestor rejected launch"));
    }

    #[test]
    fn exact_execution_samples_round_trips_through_json_bytes() {
        let samples = vec![
            Sample::new("alpha", (1u8, 2u8)),
            Sample::new("beta", (3u8, 4u8)),
        ];
        let (reconstructed, bytes) =
            exact_execution_samples(&samples, "ordered training samples").unwrap();
        assert_eq!(reconstructed, samples);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn preflight_prompt_tokenization_rejects_empty_encoded_prompt() {
        let samples = vec![Sample::new("", ())];
        let error =
            preflight_prompt_tokenization(&samples, "ordered training samples", &EmptyTokenizer)
                .expect_err("tokenizer produced no tokens");
        assert!(error.contains("encoded to zero tokens"));
    }

    #[test]
    fn preflight_prompt_tokenization_accepts_non_empty_encoding() {
        let samples = vec![Sample::new("", ()), Sample::new("", ())];
        assert!(preflight_prompt_tokenization(
            &samples,
            "ordered held-out samples",
            &OneTokenTokenizer,
        )
        .is_ok());
    }

    #[test]
    fn distributed_launch_value_consensus_detects_disagreement() {
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    scope.spawn(move || {
                        validate_launch_value_consensus(
                            "launch payload",
                            if comm.rank() == 0 { b"aaaa" } else { b"bbbb" },
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
        assert!(results.iter().all(std::result::Result::is_err));
        assert!(results.iter().all(|result| result
            .as_ref()
            .unwrap_err()
            .contains("all ranks must bind identical bytes")));
    }

    #[test]
    fn distributed_launch_binding_collects_launches_across_world() {
        let hash_a = "aa".repeat(32);
        let hash_b = "bb".repeat(32);
        let outcomes = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    let hash_a = hash_a.clone();
                    let hash_b = hash_b.clone();
                    scope.spawn(move || {
                        distributed_launch_binding(
                            if comm.rank() == 0 {
                                hash_a.as_str()
                            } else {
                                hash_b.as_str()
                            },
                            Some(&comm),
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0], outcomes[1]);
        assert_eq!(outcomes[0].0, hash_a);
    }

    #[test]
    fn run_on_tensor_parallel_primary_executes_once() {
        let executions = Arc::new(AtomicUsize::new(0));
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    let executions = Arc::clone(&executions);
                    scope.spawn(move || {
                        let shared = SharedComm::from_box(Box::new(comm));
                        run_on_tensor_parallel_primary(Some(&shared), "parallel launch", || {
                            executions.fetch_add(1, Ordering::SeqCst);
                            Ok::<usize, CliOrchestrationError>(shared.rank())
                        })
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let results: Vec<_> = results.into_iter().map(|result| result.unwrap()).collect();
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        assert_eq!(results[0], Some(0));
        assert_eq!(results[1], None);
    }

    #[test]
    fn cli_run_health_policy_validate_rejects_stop_action() {
        let policy =
            CliRunHealthPolicy::from_json_value(json!({"telemetry_dark": "stop"})).unwrap();
        let config = TrainerConfig::builder()
            .steps(10)
            .group_size(1)
            .max_new_tokens(1)
            .build();
        let err = policy.validate(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("reserved for future in-run gating"));
    }

    #[test]
    fn cli_run_health_policy_validate_rejects_window_older_than_steps() {
        let policy = CliRunHealthPolicy::from_json_value(json!({
            "reward_collapse": {
                "window": 10,
                "min": 0.5,
                "action": "warn"
            }
        }))
        .unwrap();
        let config = TrainerConfig::builder()
            .steps(3)
            .group_size(1)
            .max_new_tokens(1)
            .build();
        let err = policy.validate(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("window (10) must be <= trainer.steps (3)"));
    }

    #[test]
    fn cli_health_policy_evaluate_reports_reward_and_source_findings() {
        let policy = CliRunHealthPolicy::from_json_value(json!({
            "reward_collapse": {
                "window": 2,
                "min": 0.4,
                "action": "warn"
            },
            "source_dominance": {
                "max_fraction": 0.4,
                "action": "warn"
            }
        }))
        .unwrap();

        let history = vec![
            Metrics {
                reward_mean: 0.1,
                ..Metrics::at_step(0)
            },
            Metrics {
                reward_mean: 0.2,
                ..Metrics::at_step(1)
            },
            Metrics {
                reward_mean: 0.0,
                ..Metrics::at_step(2)
            },
        ];
        let summary = crate::telemetry::summarize(&history).unwrap();
        let config = TrainerConfig::builder()
            .steps(3)
            .group_size(1)
            .max_new_tokens(1)
            .build();
        let mut candidates = CliCandidateHealth {
            total: 4,
            source_buckets: BTreeMap::from([
                ("dominant-source".to_owned(), 3),
                ("other".to_owned(), 1),
            ]),
            steps: BTreeMap::new(),
        };
        for step in 0..3 {
            let mut step_health = CliCandidateStepHealth {
                total: 2,
                correctness_supported: 2,
                correct: 2,
                ..CliCandidateStepHealth::default()
            };
            step_health.prompt_groups.insert(
                0,
                CliCandidatePromptGroupHealth {
                    group_indices: BTreeSet::from([0]),
                },
            );
            candidates.steps.insert(step, step_health);
        }

        let report = policy.evaluate(
            &history,
            &summary,
            CliRunHealthContext::from_trainer(&config),
            Some(&candidates),
        );

        let rules: Vec<_> = report.findings.iter().map(|finding| finding.rule).collect();
        assert!(rules.contains(&"reward_collapse"));
        assert!(rules.contains(&"source_dominance"));
    }

    #[test]
    fn cli_candidate_health_reader_preserves_correctness_and_source_buckets() {
        let temporary = TestDir::new("candidate-health-inputs");
        let first = RunDir::create(&temporary.0, "first").unwrap();
        let second = RunDir::create(&temporary.0, "second").unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let manifest = candidate_test_manifest(0, 1, signer.public_key_hex());
        fs::write(
            first.candidates_path(),
            signed_candidate_row(
                &signer,
                &manifest,
                0,
                0,
                1,
                1.0,
                1,
                "correct",
                Some(json!({"correct": true, "source_sha256": "source-a"})),
            ),
        )
        .unwrap();
        fs::write(
            second.candidates_path(),
            signed_candidate_row(
                &signer,
                &manifest,
                1,
                0,
                1,
                0.0,
                1,
                "unknown",
                Some(json!({})),
            ),
        )
        .unwrap();

        let health = read_cli_candidate_health_inputs(&[
            first.root().to_path_buf(),
            second.candidates_path(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(health.total, 2);
        assert_eq!(health.source_buckets["source-a"], 1);
        assert_eq!(health.source_buckets["__unknown_source__"], 1);
        assert_eq!(health.steps[&0].correctness_supported, 1);
        assert_eq!(health.steps[&0].correct, 1);
        assert_eq!(health.steps[&1].correctness_supported, 0);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn correctness_health_rejects_each_incomplete_evidence_shape() {
        let history = vec![Metrics::at_step(0), Metrics::at_step(1)];
        let rule = CliWindowThreshold {
            window: 2,
            min: 0.75,
            action: CliHealthAction::Warn,
        };
        let context = CliRunHealthContext {
            group_size: 2,
            prompt_groups_per_step: 1,
        };
        let finding = |history: &[Metrics], candidates: Option<&CliCandidateHealth>| {
            let mut report = CliRunHealthReport::default();
            push_correctness_collapse_finding(history, context, candidates, &rule, &mut report);
            report.findings.into_iter().next().unwrap().message
        };

        assert!(finding(&history[..1], None).contains("only 1 metric rows"));
        assert!(finding(&history, None).contains("ledger unavailable"));
        assert!(finding(&history, Some(&CliCandidateHealth::default())).contains("ledger is empty"));

        let mut missing = CliCandidateHealth {
            total: 1,
            ..CliCandidateHealth::default()
        };
        missing.steps.insert(0, CliCandidateStepHealth::default());
        assert!(finding(&history, Some(&missing)).contains("missing rows"));

        let step = |indices: BTreeSet<usize>, supported: usize, correct: usize| {
            let mut step = CliCandidateStepHealth {
                total: 2,
                correctness_supported: supported,
                correct,
                ..CliCandidateStepHealth::default()
            };
            step.prompt_groups.insert(
                0,
                CliCandidatePromptGroupHealth {
                    group_indices: indices,
                },
            );
            step
        };
        let mut partial = CliCandidateHealth {
            total: 2,
            ..CliCandidateHealth::default()
        };
        partial.steps.insert(0, step(BTreeSet::from([0]), 1, 1));
        partial.steps.insert(1, step(BTreeSet::from([0]), 1, 1));
        assert!(finding(&history, Some(&partial)).contains("lacks full group coverage"));

        let mut unsupported = CliCandidateHealth {
            total: 4,
            ..CliCandidateHealth::default()
        };
        unsupported
            .steps
            .insert(0, step(BTreeSet::from([0, 1]), 0, 0));
        unsupported
            .steps
            .insert(1, step(BTreeSet::from([0, 1]), 0, 0));
        assert!(finding(&history, Some(&unsupported)).contains("metadata unavailable"));

        let mut low_fraction = CliCandidateHealth {
            total: 4,
            ..CliCandidateHealth::default()
        };
        low_fraction
            .steps
            .insert(0, step(BTreeSet::from([0, 1]), 2, 1));
        low_fraction
            .steps
            .insert(1, step(BTreeSet::from([0, 1]), 2, 0));
        assert!(finding(&history, Some(&low_fraction)).contains("1/4 = 0.250"));
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn grad_health_and_report_render_cover_warn_and_fail_verdicts() {
        let history = vec![
            Metrics {
                grad_norm: 1.0,
                ..Metrics::at_step(1)
            },
            Metrics {
                grad_norm: 3.0,
                ..Metrics::at_step(2)
            },
            Metrics {
                grad_norm: 9.0,
                ..Metrics::at_step(3)
            },
        ];
        let noisy_history = vec![
            Metrics {
                grad_norm: f32::NAN,
                ..Metrics::at_step(0)
            },
            Metrics {
                grad_norm: -1.0,
                ..Metrics::at_step(1)
            },
            Metrics {
                grad_norm: 3.0,
                ..Metrics::at_step(2)
            },
        ];
        assert_eq!(median_positive_grad_norm(&[]), 0.0);
        assert_eq!(median_positive_grad_norm(&noisy_history), 3.0);
        assert_eq!(median_positive_grad_norm(&history), 3.0);

        let mut report = CliRunHealthReport::default();
        push_grad_spike_finding(
            &history,
            &CliFactorThreshold {
                factor: 2.0,
                action: CliHealthAction::Warn,
            },
            &mut report,
        );
        assert!(report.render().contains("WARN grad_spike"));
        report.push("forced", CliHealthAction::Fail, "failed".into());
        assert!(report.is_fail());
        let rendered = report.render();
        assert!(rendered.contains("run health policy — FAIL"));
        assert!(rendered.contains("FAIL forced: failed"));
    }

    #[test]
    fn parse_strict_candidate_row_rejects_unknown_field() {
        let raw = r#"{"launch_sha256":"11","record_sha256":"22","record_signature":"33","step":0,"rank":0,"world_size":1,"prompt_index":0,"group_index":0,"reward":1.0,"completion_len_tokens":1,"completion":"x","extra":7}"#;
        let err =
            parse_strict_candidate_row(Path::new("run/candidates.jsonl"), 1, raw).unwrap_err();
        assert!(err.to_string().contains("contains unknown field \"extra\""));
    }

    #[test]
    fn asymmetric_malformed_tp_ledger_fails_in_lockstep_before_completion_mapping() {
        let temporary = TestDir::new("tp-candidate-malformed");
        let completion_entries = AtomicUsize::new(0);
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    let root = temporary.0.clone();
                    let completion_entries = &completion_entries;
                    scope.spawn(move || {
                        let rank = comm.rank();
                        let signer = CandidateSigner::generate().unwrap();
                        let manifest = candidate_test_manifest(rank, 2, signer.public_key_hex());
                        let run = RunDir::create(&root, format!("rank-{rank}")).unwrap();
                        if rank == 1 {
                            fs::write(run.candidates_path(), b"{malformed}\n").unwrap();
                        }
                        let local = load_cli_candidate_selection(
                            &run,
                            &manifest,
                            &candidate_test_config(),
                            CandidateTopology {
                                rank,
                                world_size: 2,
                            },
                        );
                        let result = coordinate_cli_result(
                            Some(&comm),
                            "candidate loading and selection",
                            local,
                        );
                        if result.is_ok() {
                            completion_entries.fetch_add(1, Ordering::SeqCst);
                        }
                        (rank, result.map_err(|error| error.to_string()))
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(
            results.iter().all(|(_, result)| result.is_err()),
            "{results:?}"
        );
        assert!(
            results.iter().any(|(rank, result)| *rank == 0
                && result
                    .as_ref()
                    .unwrap_err()
                    .contains("candidate loading and selection failed on a peer")),
            "{results:?}"
        );
        assert_eq!(completion_entries.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn load_cli_candidate_selection_detects_blank_and_unterminated_rows() {
        let temporary = TestDir::new("candidate-row-shapes");
        let run = RunDir::create(&temporary.0, "topology").unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let manifest = candidate_test_manifest(0, 1, signer.public_key_hex());

        let valid = signed_candidate_row(&signer, &manifest, 0, 0, 1, 1.0, 1, "candidate", None);
        let mut blank_lines = Vec::new();
        blank_lines.extend_from_slice(&valid);
        blank_lines.push(b'\n');
        fs::write(run.candidates_path(), blank_lines).unwrap();

        let blank_result = load_cli_candidate_selection(
            &run,
            &manifest,
            &candidate_test_config(),
            CandidateTopology {
                rank: 0,
                world_size: 1,
            },
        );
        assert!(blank_result
            .unwrap_err()
            .to_string()
            .contains("contains blank row 2"));

        let mut unterminated = valid;
        unterminated.pop();
        fs::write(run.candidates_path(), &unterminated).unwrap();
        let unterminated_result = load_cli_candidate_selection(
            &run,
            &manifest,
            &candidate_test_config(),
            CandidateTopology {
                rank: 0,
                world_size: 1,
            },
        );
        assert!(unterminated_result
            .unwrap_err()
            .to_string()
            .contains("unterminated final row"));
    }

    #[test]
    fn load_cli_candidate_selection_rejects_rank_world_mismatch() {
        let temporary = TestDir::new("candidate-rank-world-mismatch");
        let run = RunDir::create(&temporary.0, "topology").unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let manifest = candidate_test_manifest(0, 1, signer.public_key_hex());
        let row = signed_candidate_row(&signer, &manifest, 0, 1, 1, 1.0, 1, "candidate", None);
        fs::write(run.candidates_path(), &row).unwrap();

        let result = load_cli_candidate_selection(
            &run,
            &manifest,
            &candidate_test_config(),
            CandidateTopology {
                rank: 0,
                world_size: 1,
            },
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("rank/world disagree with active execution topology"));
    }

    #[test]
    fn load_cli_candidate_selection_rejects_coordinates_outside_launch_config() {
        let temporary = TestDir::new("candidate-coord-overflow");
        let run = RunDir::create(&temporary.0, "topology").unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let manifest = candidate_test_manifest(0, 1, signer.public_key_hex());
        let row = signed_candidate_row(&signer, &manifest, 9, 0, 1, 1.0, 1, "candidate", None);
        fs::write(run.candidates_path(), row).unwrap();

        let result = load_cli_candidate_selection(
            &run,
            &manifest,
            &candidate_test_config(),
            CandidateTopology {
                rank: 0,
                world_size: 1,
            },
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("coordinates exceed the launch config"));
    }

    #[test]
    fn trimul_candidate_selection_rejects_missing_verifier_identity() {
        let temporary = TestDir::new("trimul-missing-verifier");
        let run = RunDir::create(&temporary.0, "topology").unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let manifest = trimul_candidate_test_manifest(0, 1, signer.public_key_hex(), false);
        let row = signed_candidate_row(&signer, &manifest, 0, 0, 1, 1.0, 1, "candidate", None);
        fs::write(run.candidates_path(), row).unwrap();

        let result = load_cli_candidate_selection(
            &run,
            &manifest,
            &candidate_test_config(),
            CandidateTopology {
                rank: 0,
                world_size: 1,
            },
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing verifier isolation evidence"));
    }

    #[test]
    fn trimul_candidate_selection_rejects_verifier_metadata_mismatch() {
        let temporary = TestDir::new("trimul-verifier-mismatch");
        let run = RunDir::create(&temporary.0, "topology").unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let manifest = trimul_candidate_test_manifest(0, 1, signer.public_key_hex(), true);
        let mut metadata = trimul_candidate_metadata(&manifest, false, false);
        metadata.as_object_mut().unwrap().insert(
            "timing_metric".to_owned(),
            serde_json::Value::String("bad-metric".to_owned()),
        );
        let row = signed_candidate_row(
            &signer,
            &manifest,
            0,
            0,
            1,
            1.0,
            1,
            "candidate",
            Some(metadata),
        );
        fs::write(run.candidates_path(), row).unwrap();

        let result = load_cli_candidate_selection(
            &run,
            &manifest,
            &candidate_test_config(),
            CandidateTopology {
                rank: 0,
                world_size: 1,
            },
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("verifier tier/evidence does not match launch.json"));
    }

    #[test]
    fn trimul_candidate_selection_rejects_inconsistent_extraction_and_execution_provenance() {
        let temporary = TestDir::new("trimul-exec-consistency");
        let run = RunDir::create(&temporary.0, "topology").unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let manifest = trimul_candidate_test_manifest(0, 1, signer.public_key_hex(), true);
        let metadata = trimul_candidate_metadata(&manifest, true, false);
        let row = signed_candidate_row(
            &signer,
            &manifest,
            0,
            0,
            1,
            1.0,
            1,
            "candidate",
            Some(metadata),
        );
        fs::write(run.candidates_path(), row).unwrap();

        let result = load_cli_candidate_selection(
            &run,
            &manifest,
            &candidate_test_config(),
            CandidateTopology {
                rank: 0,
                world_size: 1,
            },
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("inconsistent extraction/execution evidence"));
    }
}
