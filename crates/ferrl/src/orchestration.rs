//! Library-owned discovery and CLI training orchestration.
//!
//! The stable SDK is [`crate::discovery`].  This module is deliberately hidden:
//! its concrete CLI request types exist only because the package binary is a
//! separate Rust crate.  They are input records, not algorithm, plugin, or
//! lifecycle extension points.  The lifecycle itself is private to the library.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::Device;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::comm::{Comm, CommError};
use crate::eval::evaluate;
use crate::hf::{resolve_checkpoint_eos, validate_resolved_eos_consensus, CheckpointEosSelection};
use crate::loader::{load_auto_policy_with_identity, AutoPolicy, LoaderOpts, PolicyLoadIdentity};
use crate::policy::{GenConfig, Policy, TensorParallelPolicy};
use crate::reward::RewardFn;
use crate::sample::Sample;
use crate::telemetry::{summarize, CandidateRecord, CandidateSigner, Metrics, RunDir, RunSummary};
use crate::tensor_parallel::TensorParallelPlan;
use crate::tokenizer::HfTokenizer;
use crate::trainer::{RunStop, TokenizerLike, Trainer, TrainerConfig};

/// Error returned by the hidden concrete CLI engine.
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
/// This is not a lifecycle callback: Ferrl constructs the complete immutable
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
pub const LAUNCH_CONTRACT_VERSION: u32 = 2;
/// CLI launch-manifest kind.
#[doc(hidden)]
pub const LAUNCH_KIND: &str = "ferrl.run-launch";
/// Domain for CLI launch payload digests.
#[doc(hidden)]
pub const LAUNCH_PAYLOAD_DOMAIN: &str = "ferrl.run-launch.payload.v2";
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

/// Result returned by the library-owned training boundary.
pub(crate) struct TrainOutcome<P> {
    stop: RunStop,
    preempted: Option<P>,
}

impl<P> TrainOutcome<P> {
    pub(crate) fn completed() -> Self {
        Self {
            stop: RunStop::Completed,
            preempted: None,
        }
    }

    pub(crate) fn preempted(result: P) -> Self {
        Self {
            stop: RunStop::Preempted,
            preempted: Some(result),
        }
    }

    fn stop(&self) -> RunStop {
        self.stop
    }

    fn into_preempted(self) -> P {
        self.preempted
            .expect("preempted training outcome must carry a terminal result")
    }
}

/// Terminal result of the private common lifecycle engine.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LifecycleOutcome<C, P> {
    Completed(C),
    Preempted(P),
}

/// Sealed-to-the-library implementation detail for the two concrete run modes.
///
/// This is `pub(crate)` solely so `discovery.rs` can supply the SDK mode.  No
/// downstream crate, including the package binary, can implement it.
pub(crate) trait Lifecycle {
    type Error;
    type State;
    type Completed;
    type Preempted;

    fn setup(&mut self) -> Result<Self::State, Self::Error>;
    fn preflight(&mut self, state: &mut Self::State) -> Result<(), Self::Error>;
    fn launch_and_build_trainer(&mut self, state: &mut Self::State) -> Result<(), Self::Error>;
    fn train(
        &mut self,
        state: &mut Self::State,
    ) -> Result<TrainOutcome<Self::Preempted>, Self::Error>;
    fn post_run_health(&mut self, state: &mut Self::State) -> Result<(), Self::Error>;
    fn evaluate_and_publish(&mut self, state: &mut Self::State) -> Result<(), Self::Error>;
    fn load_and_select_candidates(&mut self, state: &mut Self::State) -> Result<(), Self::Error>;
    fn map_completed(self, state: Self::State) -> Result<Self::Completed, Self::Error>;
}

/// Execute the only trusted lifecycle ordering used by SDK discovery and CLI training.
pub(crate) fn run_lifecycle<L>(
    mut lifecycle: L,
) -> Result<LifecycleOutcome<L::Completed, L::Preempted>, L::Error>
where
    L: Lifecycle,
{
    let mut state = lifecycle.setup()?;
    lifecycle.preflight(&mut state)?;
    lifecycle.launch_and_build_trainer(&mut state)?;
    let training = lifecycle.train(&mut state)?;
    if training.stop() == RunStop::Preempted {
        return Ok(LifecycleOutcome::Preempted(training.into_preempted()));
    }
    lifecycle.post_run_health(&mut state)?;
    lifecycle.evaluate_and_publish(&mut state)?;
    lifecycle.load_and_select_candidates(&mut state)?;
    Ok(LifecycleOutcome::Completed(lifecycle.map_completed(state)?))
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

/// Run production CLI training through the concrete library engine.
#[doc(hidden)]
pub fn run_cli_training<R>(
    request: CliTrainingRequest<'_, R>,
    attestor: Option<&dyn CliLaunchAttestor>,
) -> Result<CliRunOutcome, CliOrchestrationError>
where
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
{
    run_cli_training_inner(
        request,
        attestor,
        AutoPolicy::supports_tensor_parallel,
        |model_dir, device, options| {
            load_auto_policy_with_identity(model_dir, device, options)
                .map_err(|error| CliOrchestrationError::msg(error.to_string()))
        },
    )
}

/// Test-only policy-loading entry for binary unit controls.
///
/// Production callers use [`run_cli_training`].  This narrow loader injection
/// exists solely so the package binary can retain mutation-sensitive unit
/// controls without giving callers any lifecycle phase authority.
#[doc(hidden)]
pub fn run_cli_training_with_test_loader<P, R, F, C>(
    request: CliTrainingRequest<'_, R>,
    attestor: Option<&dyn CliLaunchAttestor>,
    supports_tensor_parallel: C,
    loader: F,
) -> Result<CliRunOutcome, CliOrchestrationError>
where
    P: Policy + TensorParallelPolicy,
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
    F: FnOnce(
        &Path,
        &Device,
        &LoaderOpts,
    ) -> Result<(P, HfTokenizer, PolicyLoadIdentity), CliOrchestrationError>,
    C: Fn(&P) -> bool,
{
    run_cli_training_inner(request, attestor, supports_tensor_parallel, loader)
}

fn run_cli_training_inner<P, R, F, C>(
    request: CliTrainingRequest<'_, R>,
    attestor: Option<&dyn CliLaunchAttestor>,
    supports_tensor_parallel: C,
    loader: F,
) -> Result<CliRunOutcome, CliOrchestrationError>
where
    P: Policy + TensorParallelPolicy,
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
    F: FnOnce(
        &Path,
        &Device,
        &LoaderOpts,
    ) -> Result<(P, HfTokenizer, PolicyLoadIdentity), CliOrchestrationError>,
    C: Fn(&P) -> bool,
{
    let lifecycle = CliLifecycle {
        request,
        attestor,
        loader: Some(loader),
        supports_tensor_parallel,
        _policy: std::marker::PhantomData,
    };
    match run_lifecycle(lifecycle)? {
        LifecycleOutcome::Completed(completed) => Ok(CliRunOutcome::Completed(completed)),
        LifecycleOutcome::Preempted(preempted) => Ok(CliRunOutcome::Preempted(preempted)),
    }
}

struct CliLifecycle<'request, 'attestor, P, R, F, C>
where
    R: RewardFn,
{
    request: CliTrainingRequest<'request, R>,
    attestor: Option<&'attestor dyn CliLaunchAttestor>,
    loader: Option<F>,
    supports_tensor_parallel: C,
    _policy: std::marker::PhantomData<fn() -> P>,
}

struct CliRunState<P, T> {
    runtime: CliRuntime,
    policy: P,
    tokenizer: HfTokenizer,
    trainer_config: TrainerConfig,
    policy_identity: PolicyLoadIdentity,
    frozen_policy_sha256: String,
    generation_config: GenConfig,
    training_samples: Vec<Sample<T>>,
    evaluation_samples: Vec<Sample<T>>,
    verifier_assets_identity: Option<crate::trimul::TrimulVerifierIdentity>,
    verifier_identity: Option<LaunchVerifierIdentity>,
    manifest: Option<LaunchManifest>,
    run: Option<RunDir>,
    trainer: Option<Trainer>,
    launch_sha256: Option<String>,
    history: Option<Vec<Metrics>>,
    health_report: Option<CliRunHealthReport>,
    selected_candidate: Option<CandidateRecord>,
}

impl<'request, 'attestor, P, R, F, C> Lifecycle for CliLifecycle<'request, 'attestor, P, R, F, C>
where
    P: Policy + TensorParallelPolicy,
    R: RewardFn,
    R::Target: Serialize + DeserializeOwned,
    F: FnOnce(
        &Path,
        &Device,
        &LoaderOpts,
    ) -> Result<(P, HfTokenizer, PolicyLoadIdentity), CliOrchestrationError>,
    C: Fn(&P) -> bool,
{
    type Error = CliOrchestrationError;
    type State = CliRunState<P, R::Target>;
    type Completed = CliCompletedRun;
    type Preempted = CliPreemptedRun;

    #[allow(clippy::cognitive_complexity)]
    fn setup(&mut self) -> Result<Self::State, Self::Error> {
        self.request
            .health_policy
            .validate(&self.request.trainer_config)?;
        let runtime = CliRuntime::from_execution(std::mem::replace(
            &mut self.request.execution,
            CliExecution::WorldOne,
        ));
        let launch = runtime.launch.clone();
        let launch_comm = launch.as_ref().map(|comm| comm as &dyn Comm);
        let distributed = runtime.distributed.clone();
        let tensor_parallel_plan = runtime.tensor_parallel_plan;
        tracing::info!(
            task = %self.request.launch.task,
            steps = self.request.trainer_config.steps,
            group_size = self.request.trainer_config.group_size,
            activation_checkpointing = self.request.activation_checkpointing,
            train = self.request.training_samples.len(),
            eval = self.request.evaluation_samples.len(),
            tensor_parallel_rank = tensor_parallel_plan.rank(),
            tensor_parallel_world = tensor_parallel_plan.world_size(),
            "ferrl train: starting"
        );

        let loader = self.loader.take().ok_or_else(|| {
            CliOrchestrationError::msg("CLI policy loader was entered more than once")
        })?;
        let model_setup = (|| {
            let (policy, tokenizer, identity) = loader(
                self.request.model_dir,
                self.request.device,
                &self.request.loader_opts,
            )?;
            let frozen_policy_sha256 = identity.policy_sha256.clone();
            let mut trainer_config = self.request.trainer_config.clone();
            trainer_config.eos_token_id = resolve_checkpoint_eos(
                self.request.model_dir,
                &tokenizer,
                cli_checkpoint_eos_selection(self.request.eos_selection),
            )
            .map_err(|error| {
                CliOrchestrationError::msg(format!("checkpoint EOS resolution failed: {error}"))
            })?;
            if runtime.tensor_parallel.is_some() && !(self.supports_tensor_parallel)(&policy) {
                return Err(CliOrchestrationError::msg(
                    "loaded checkpoint family does not support tensor_parallel execution; supported \
                     families are qwen3 (including legacy configs without model_type) and dense \
                     gemma4/gemma4_unified; qwen3_5/qwen3_5_moe (Qwen3.5/3.6) are unsupported",
                ));
            }
            if tensor_parallel_plan.is_sharded()
                && !policy.supports_sharded_tensor_parallel_backward()
            {
                return Err(CliOrchestrationError::msg(
                    "sharded tensor_parallel training is supported only for dense \
                     gemma4/gemma4_unified policies with activation checkpointing; the loaded \
                     policy does not provide cross-rank backward semantics",
                ));
            }
            Ok((
                policy,
                tokenizer,
                trainer_config,
                identity,
                frozen_policy_sha256,
            ))
        })();
        let (policy, tokenizer, trainer_config, policy_identity, frozen_policy_sha256) =
            coordinate_cli_result(launch_comm, "model and EOS setup", model_setup)?;
        if let Some(comm) = launch_comm {
            validate_resolved_eos_consensus(trainer_config.eos_token_id, comm)
                .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
        }
        validate_data_parallel_policy_preflight(
            &policy,
            &policy_identity.policy_sha256,
            distributed.as_ref().map(|comm| comm as &dyn Comm),
        )?;

        let prompt_sha256 = self.request.rendered_prompt_bytes.map(sha256_hex);
        let verifier_assets_identity = self
            .request
            .verifier_assets
            .map(|assets| assets.identity().clone());
        if (self.request.verifier_assets.is_some() || self.request.verifier_identity.is_some())
            != (self.request.launch.task == "trimul")
            || self.request.verifier_assets.is_some() != self.request.verifier_identity.is_some()
        {
            return Err(CliOrchestrationError::msg(
                "TriMul launch requires both verifier assets and isolation evidence, while non-TriMul launches require neither",
            ));
        }
        let portable_verifier = self
            .request
            .verifier_identity
            .as_ref()
            .map(portable_verifier_consensus)
            .transpose()?;
        let common_provenance = serde_json::to_vec(&(
            &self.request.launch.ferrl_commit,
            &self.request.launch.run.group_id,
            &policy_identity.policy_sha256,
            &policy_identity.tokenizer_sha256,
            policy_identity.model_family,
            &prompt_sha256,
            &verifier_assets_identity,
            &portable_verifier,
        ))
        .map_err(|error| {
            CliOrchestrationError::msg(format!("serialize launch provenance: {error}"))
        })?;
        validate_launch_value_consensus(
            "model/checkpoint/tokenizer/prompt provenance",
            &common_provenance,
            launch_comm,
        )?;

        Ok(CliRunState {
            runtime,
            policy,
            tokenizer,
            trainer_config: trainer_config.clone(),
            policy_identity,
            frozen_policy_sha256,
            generation_config: GenConfig::from(&trainer_config),
            training_samples: Vec::new(),
            evaluation_samples: Vec::new(),
            verifier_assets_identity,
            verifier_identity: self.request.verifier_identity.clone(),
            manifest: None,
            run: None,
            trainer: None,
            launch_sha256: None,
            history: None,
            health_report: None,
            selected_candidate: None,
        })
    }

    fn preflight(&mut self, state: &mut Self::State) -> Result<(), Self::Error> {
        let local = (|| {
            let (training_samples, _) =
                exact_execution_samples(self.request.training_samples, "ordered training samples")
                    .map_err(|error| {
                        CliOrchestrationError::msg(format!(
                            "serialize ordered training samples: {error}"
                        ))
                    })?;
            let (evaluation_samples, _) = exact_execution_samples(
                self.request.evaluation_samples,
                "ordered held-out samples",
            )
            .map_err(|error| {
                CliOrchestrationError::msg(format!("serialize ordered held-out samples: {error}"))
            })?;
            preflight_prompt_tokenization(
                &training_samples,
                "ordered training samples",
                &state.tokenizer,
            )
            .map_err(CliOrchestrationError::msg)?;
            preflight_prompt_tokenization(
                &evaluation_samples,
                "ordered held-out samples",
                &state.tokenizer,
            )
            .map_err(CliOrchestrationError::msg)?;
            Ok((training_samples, evaluation_samples))
        })();
        let launch = state.runtime.launch.clone();
        let (training_samples, evaluation_samples) = coordinate_cli_result(
            launch.as_ref().map(|comm| comm as &dyn Comm),
            "exact sample reconstruction and tokenizer preflight",
            local,
        )?;
        state.training_samples = training_samples;
        state.evaluation_samples = evaluation_samples;
        Ok(())
    }

    #[allow(clippy::cognitive_complexity)]
    fn launch_and_build_trainer(&mut self, state: &mut Self::State) -> Result<(), Self::Error> {
        let launch = state.runtime.launch.clone();
        let launch_comm = launch.as_ref().map(|comm| comm as &dyn Comm);
        let attestation_setup = (|| {
            let candidate_signer = CandidateSigner::generate()
                .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
            let signing_public_key = candidate_signer.public_key_hex();
            let manifest = LaunchManifest::new(LaunchPayload {
                task: self.request.launch.task.clone(),
                ferrl_commit: self.request.launch.ferrl_commit.clone(),
                authentication: self.request.launch.authentication,
                run: self.request.launch.run.clone(),
                config: self.request.launch.config.clone(),
                model: LaunchModelIdentity {
                    family: state.policy_identity.model_family.to_owned(),
                    checkpoint_policy_sha256: state.policy_identity.policy_sha256.clone(),
                    tokenizer_sha256: state.policy_identity.tokenizer_sha256.clone(),
                    resolved_eos_token_id: state.trainer_config.eos_token_id,
                },
                prompt: self
                    .request
                    .rendered_prompt_bytes
                    .map(|bytes| LaunchPromptIdentity {
                        file: RunDir::PROMPT_FILE.to_owned(),
                        sha256: sha256_hex(bytes),
                        len_bytes: bytes.len(),
                    }),
                verifier: state.verifier_identity.clone(),
                candidate_ledger: LaunchCandidateLedger {
                    file: RunDir::CANDIDATES_FILE.to_owned(),
                    format_version: 1,
                    row_digest_domain: CANDIDATE_RECORD_DOMAIN.to_owned(),
                    row_signature_algorithm: "ed25519".to_owned(),
                    signing_public_key,
                },
            })?;
            let manifest = match self.request.launch.authentication {
                LaunchAuthenticationMode::LocalEphemeralV1 => manifest,
                LaunchAuthenticationMode::ExternalAttestedV1 => {
                    let attestor = self.attestor.ok_or_else(|| {
                        CliOrchestrationError::msg(
                            "launch_authentication = \"external_attested_v1\" requires the protected external launch attestor",
                        )
                    })?;
                    manifest.attest(attestor)?
                }
            };
            Ok((candidate_signer, manifest))
        })();
        let (candidate_signer, manifest) =
            coordinate_cli_result(launch_comm, "launch authentication", attestation_setup)?;
        coordinate_cli_result(
            launch_comm,
            "launch-bound verifier revalidation",
            self.request.verifier_assets.map_or(Ok(()), |assets| {
                assets
                    .verify_current()
                    .map_err(|error| CliOrchestrationError::msg(error.to_string()))
            }),
        )?;

        let publication_setup = (|| {
            let launch_sha256 = manifest.payload_sha256.clone();
            let manifest_bytes = manifest.to_pretty_bytes()?;
            let run = RunDir::create(
                &self.request.launch.output_root,
                self.request.launch.run.run_id.clone(),
            )
            .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
            run.write_immutable_launch(&manifest_bytes, self.request.rendered_prompt_bytes)
                .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
            let trainer = open_cli_trainer(
                state.trainer_config.clone(),
                &run,
                state.runtime.distributed.clone(),
                &state.frozen_policy_sha256,
                &launch_sha256,
                candidate_signer,
            )?;
            Ok((run, trainer, launch_sha256))
        })();
        let (run, trainer, launch_sha256) = coordinate_cli_result(
            launch_comm,
            "run directory and trainer setup",
            publication_setup,
        )?;
        state.manifest = Some(manifest);
        state.run = Some(run);
        state.trainer = Some(trainer);
        state.launch_sha256 = Some(launch_sha256);
        Ok(())
    }

    fn train(
        &mut self,
        state: &mut Self::State,
    ) -> Result<TrainOutcome<Self::Preempted>, Self::Error> {
        let tensor_parallel = state.runtime.tensor_parallel.clone();
        let (history, stop) = {
            let trainer = state.trainer.as_mut().ok_or_else(|| {
                CliOrchestrationError::msg("training requires a constructed trainer")
            })?;
            match tensor_parallel.as_ref() {
                Some(comm) => trainer
                    .train_tensor_parallel(
                        &mut state.policy,
                        self.request.reward,
                        &state.tokenizer,
                        &state.training_samples,
                        comm,
                    )
                    .map_err(|error| CliOrchestrationError::msg(error.to_string()))?,
                None => trainer
                    .train(
                        &mut state.policy,
                        self.request.reward,
                        &state.tokenizer,
                        &state.training_samples,
                    )
                    .map_err(|error| CliOrchestrationError::msg(error.to_string()))?,
            }
        };
        let run = state
            .run
            .clone()
            .ok_or_else(|| CliOrchestrationError::msg("training requires a published run"))?;
        state.history = Some(history);
        if stop == RunStop::Preempted {
            let presentation_rank = run_on_tensor_parallel_primary(
                tensor_parallel.as_ref(),
                "run completion output",
                || Ok(()),
            )?
            .is_some();
            return Ok(TrainOutcome::preempted(CliPreemptedRun {
                run_dir: run.root().to_path_buf(),
                presentation_rank,
            }));
        }
        Ok(TrainOutcome::completed())
    }

    fn post_run_health(&mut self, state: &mut Self::State) -> Result<(), Self::Error> {
        let history = state.history.as_ref().ok_or_else(|| {
            CliOrchestrationError::msg("post-run health requires completed training")
        })?;
        let run = state.run.clone().ok_or_else(|| {
            CliOrchestrationError::msg("post-run health requires a published run")
        })?;
        let tensor_parallel = state.runtime.tensor_parallel.clone();
        let report =
            run_on_tensor_parallel_primary(tensor_parallel.as_ref(), "post-run health", || {
                let Some(summary) = summarize(history) else {
                    return Ok(None);
                };
                tracing::info!(steps = summary.steps, "ferrl train: complete");
                let report = evaluate_cli_run_health(
                    &self.request.health_policy,
                    history,
                    &summary,
                    &run,
                    &state.trainer_config,
                )?;
                if report.is_fail() {
                    return Err(CliOrchestrationError::RunHealth(report));
                }
                Ok((!self.request.health_policy_is_default).then_some(report))
            })?;
        if let Some(report) = report.flatten() {
            state.health_report = Some(report);
        }
        Ok(())
    }

    fn evaluate_and_publish(&mut self, state: &mut Self::State) -> Result<(), Self::Error> {
        if state.evaluation_samples.is_empty() {
            return Ok(());
        }
        let run = state
            .run
            .clone()
            .ok_or_else(|| CliOrchestrationError::msg("evaluation requires a published run"))?;
        let launch_sha256 = state
            .launch_sha256
            .clone()
            .ok_or_else(|| CliOrchestrationError::msg("evaluation requires launch identity"))?;
        let launch_comm = state.runtime.launch.clone();
        let local = evaluate(
            &mut state.policy,
            self.request.evaluation_reward,
            &state.tokenizer,
            &state.evaluation_samples,
            &state.generation_config,
        )
        .map_err(|error| CliOrchestrationError::msg(error.to_string()));
        let report = coordinate_cli_result(
            launch_comm.as_ref().map(|comm| comm as &dyn Comm),
            "held-out evaluation",
            local,
        )?;
        publish_cli_eval_report(
            &self.request.launch.task,
            self.request.data_seed,
            self.request.trimul_held_out_secret_seed,
            &state.evaluation_samples,
            &report,
            &run,
            &launch_sha256,
            state.verifier_assets_identity.as_ref(),
            launch_comm.as_ref().map(|comm| comm as &dyn Comm),
        )?;
        tracing::info!(
            base = report.base_reward_mean,
            adapter = report.adapter_reward_mean,
            improvement = report.improvement(),
            "ferrl train: held-out eval (adapter vs base)"
        );
        Ok(())
    }

    fn load_and_select_candidates(&mut self, state: &mut Self::State) -> Result<(), Self::Error> {
        let run = state.run.clone().ok_or_else(|| {
            CliOrchestrationError::msg("candidate loading requires a published run")
        })?;
        let manifest = state.manifest.clone().ok_or_else(|| {
            CliOrchestrationError::msg("candidate loading requires an authenticated launch")
        })?;
        let topology = state.runtime.candidate_topology(&manifest);
        let local = load_cli_candidate_selection(&run, &manifest, &state.trainer_config, topology);
        let launch = state.runtime.launch.clone();
        let selected = coordinate_cli_result(
            launch.as_ref().map(|comm| comm as &dyn Comm),
            "candidate loading and selection",
            local,
        )?;
        state.selected_candidate = selected;
        Ok(())
    }

    fn map_completed(self, mut state: Self::State) -> Result<Self::Completed, Self::Error> {
        let _selected_candidate = state.selected_candidate.take();
        let tensor_parallel = state.runtime.tensor_parallel.clone();
        let presentation_rank = run_on_tensor_parallel_primary(
            tensor_parallel.as_ref(),
            "run completion output",
            || Ok(()),
        )?
        .is_some();
        let run = state.run.take().ok_or_else(|| {
            CliOrchestrationError::msg("completed mapping requires a published run")
        })?;
        Ok(CliCompletedRun {
            run_dir: run.root().to_path_buf(),
            health_report: state.health_report.take(),
            presentation_rank,
        })
    }
}

fn cli_checkpoint_eos_selection(selection: CliEosSelection) -> CheckpointEosSelection {
    match selection {
        CliEosSelection::CheckpointDefault => CheckpointEosSelection::CheckpointDefault,
        CliEosSelection::Explicit(token_id) => CheckpointEosSelection::Explicit(token_id),
        CliEosSelection::Disabled => CheckpointEosSelection::Disabled,
    }
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

fn open_cli_trainer(
    config: TrainerConfig,
    run: &RunDir,
    distributed_comm: Option<SharedComm>,
    frozen_policy_sha256: &str,
    candidate_launch_sha256: &str,
    candidate_signer: CandidateSigner,
) -> Result<Trainer, CliOrchestrationError> {
    let trainer = match distributed_comm {
        Some(comm) => Trainer::with_comm(config, run, comm),
        None => Trainer::new(config, run),
    }
    .map_err(|error| CliOrchestrationError::msg(error.to_string()))?;
    trainer
        .with_frozen_policy_sha256(frozen_policy_sha256)
        .with_candidate_provenance(candidate_launch_sha256, candidate_signer)
        .map_err(|error| CliOrchestrationError::msg(error.to_string()))
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
struct CandidateTopology {
    rank: usize,
    world_size: usize,
}

#[allow(clippy::cognitive_complexity)]
fn load_cli_candidate_selection(
    run: &RunDir,
    manifest: &LaunchManifest,
    trainer_config: &TrainerConfig,
    topology: CandidateTopology,
) -> Result<Option<CandidateRecord>, CliOrchestrationError> {
    let path = run.candidates_path();
    if !path.exists() {
        return Ok(None);
    }
    let bytes = read_regular_bytes(&path)?;
    if !bytes.is_empty() && !bytes.ends_with(b"\n") {
        return Err(CliOrchestrationError::msg(format!(
            "candidate ledger {} has an unterminated final row",
            path.display()
        )));
    }
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        CliOrchestrationError::msg(format!(
            "candidate ledger {} is not UTF-8: {error}",
            path.display()
        ))
    })?;
    let mut selected = None;
    for (index, raw_line) in text.split_terminator('\n').enumerate() {
        if raw_line.trim().is_empty() {
            return Err(CliOrchestrationError::msg(format!(
                "candidate ledger {} contains blank row {}",
                path.display(),
                index + 1
            )));
        }
        let record = parse_strict_candidate_row(&path, index + 1, raw_line)?;
        crate::telemetry::verify_signed_candidate_row(
            raw_line.as_bytes(),
            &manifest.payload.candidate_ledger.signing_public_key,
            &manifest.payload_sha256,
            &record,
        )
        .map_err(|error| {
            CliOrchestrationError::msg(format!(
                "candidate ledger {} row {} failed canonical launch authentication: {error}",
                path.display(),
                index + 1
            ))
        })?;
        if manifest.payload.task == "trimul" {
            verify_cli_candidate_verifier_provenance(&record, manifest, index + 1)?;
        }
        if record.rank != topology.rank || record.world_size != topology.world_size {
            return Err(CliOrchestrationError::msg(format!(
                "candidate ledger {} row {} rank/world disagree with active execution topology",
                path.display(),
                index + 1
            )));
        }
        if record.step >= trainer_config.steps
            || record.group_index >= trainer_config.group_size
            || record.completion_len_tokens > trainer_config.max_new_tokens
        {
            return Err(CliOrchestrationError::msg(format!(
                "candidate ledger {} row {} coordinates exceed the launch config",
                path.display(),
                index + 1
            )));
        }
        if selected
            .as_ref()
            .is_none_or(|current: &CandidateRecord| cli_candidate_is_better(&record, current))
        {
            selected = Some(record);
        }
    }
    Ok(selected)
}

fn cli_candidate_is_better(candidate: &CandidateRecord, current: &CandidateRecord) -> bool {
    candidate
        .reward
        .total_cmp(&current.reward)
        .then_with(|| current.step.cmp(&candidate.step))
        .then_with(|| current.prompt_index.cmp(&candidate.prompt_index))
        .then_with(|| current.group_index.cmp(&candidate.group_index))
        .is_gt()
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

fn read_regular_bytes(path: &Path) -> Result<Vec<u8>, CliOrchestrationError> {
    let path_metadata = fs::symlink_metadata(path)
        .map_err(|error| CliOrchestrationError::msg(format!("read {}: {error}", path.display())))?;
    if !path_metadata.file_type().is_file() {
        return Err(CliOrchestrationError::msg(format!(
            "provenance input {} is not a regular file",
            path.display()
        )));
    }
    let mut file = File::open(path)
        .map_err(|error| CliOrchestrationError::msg(format!("read {}: {error}", path.display())))?;
    let file_metadata = file
        .metadata()
        .map_err(|error| CliOrchestrationError::msg(format!("read {}: {error}", path.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if path_metadata.dev() != file_metadata.dev() || path_metadata.ino() != file_metadata.ino()
        {
            return Err(CliOrchestrationError::msg(format!(
                "provenance input {} changed while it was opened",
                path.display()
            )));
        }
    }
    let expected_len = file_metadata.len();
    let mut bytes = Vec::with_capacity(usize::try_from(expected_len).unwrap_or(0));
    file.read_to_end(&mut bytes)
        .map_err(|error| CliOrchestrationError::msg(format!("read {}: {error}", path.display())))?;
    if bytes.len() as u64 != expected_len {
        return Err(CliOrchestrationError::msg(format!(
            "provenance input {} changed length while it was captured",
            path.display()
        )));
    }
    Ok(bytes)
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
    use std::cell::RefCell;
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::Path;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use serde_json::json;

    use super::*;

    #[derive(Clone)]
    struct ProbeLifecycle {
        events: Rc<RefCell<Vec<&'static str>>>,
        failure: Option<&'static str>,
        stop: RunStop,
    }

    impl ProbeLifecycle {
        fn new() -> Self {
            Self {
                events: Rc::new(RefCell::new(Vec::new())),
                failure: None,
                stop: RunStop::Completed,
            }
        }

        fn hit(&mut self, phase: &'static str) -> Result<(), &'static str> {
            self.events.borrow_mut().push(phase);
            if self.failure == Some(phase) {
                Err(phase)
            } else {
                Ok(())
            }
        }
    }

    impl Lifecycle for ProbeLifecycle {
        type Error = &'static str;
        type State = ();
        type Completed = &'static str;
        type Preempted = &'static str;

        fn setup(&mut self) -> Result<Self::State, Self::Error> {
            self.hit("setup")?;
            Ok(())
        }

        fn preflight(&mut self, _state: &mut Self::State) -> Result<(), Self::Error> {
            self.hit("preflight")
        }

        fn launch_and_build_trainer(
            &mut self,
            _state: &mut Self::State,
        ) -> Result<(), Self::Error> {
            self.hit("launch")
        }

        fn train(
            &mut self,
            _state: &mut Self::State,
        ) -> Result<TrainOutcome<Self::Preempted>, Self::Error> {
            self.hit("train")?;
            Ok(match self.stop {
                RunStop::Completed => TrainOutcome::completed(),
                RunStop::Preempted => TrainOutcome::preempted("checkpoint"),
            })
        }

        fn post_run_health(&mut self, _state: &mut Self::State) -> Result<(), Self::Error> {
            self.hit("health")
        }

        fn evaluate_and_publish(&mut self, _state: &mut Self::State) -> Result<(), Self::Error> {
            self.hit("eval")
        }

        fn load_and_select_candidates(
            &mut self,
            _state: &mut Self::State,
        ) -> Result<(), Self::Error> {
            self.hit("candidates")
        }

        fn map_completed(self, _state: Self::State) -> Result<Self::Completed, Self::Error> {
            self.events.borrow_mut().push("complete");
            if self.failure == Some("complete") {
                return Err("complete");
            }
            Ok("result")
        }
    }

    #[test]
    fn library_lifecycle_runs_every_trusted_phase_in_exact_order() {
        let lifecycle = ProbeLifecycle::new();
        let events = Rc::clone(&lifecycle.events);
        assert_eq!(
            run_lifecycle(lifecycle).unwrap(),
            LifecycleOutcome::Completed("result")
        );
        assert_eq!(
            *events.borrow(),
            vec![
                "setup",
                "preflight",
                "launch",
                "train",
                "health",
                "eval",
                "candidates",
                "complete"
            ]
        );
    }

    #[test]
    fn library_lifecycle_short_circuits_each_failed_phase() {
        for failed in [
            "setup",
            "preflight",
            "launch",
            "train",
            "health",
            "eval",
            "candidates",
            "complete",
        ] {
            let mut lifecycle = ProbeLifecycle::new();
            lifecycle.failure = Some(failed);
            let events = Rc::clone(&lifecycle.events);
            assert_eq!(run_lifecycle(lifecycle).unwrap_err(), failed);
            let events = events.borrow();
            assert_eq!(events.last(), Some(&failed), "{failed}: {events:?}");
        }
    }

    #[test]
    fn library_lifecycle_preemption_skips_every_post_training_phase() {
        let mut lifecycle = ProbeLifecycle::new();
        lifecycle.stop = RunStop::Preempted;
        let events = Rc::clone(&lifecycle.events);
        assert_eq!(
            run_lifecycle(lifecycle).unwrap(),
            LifecycleOutcome::Preempted("checkpoint")
        );
        assert_eq!(
            *events.borrow(),
            vec!["setup", "preflight", "launch", "train"]
        );
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
