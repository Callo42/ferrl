//! Stable, world-one discovery SDK.
//!
//! This module is the product-level boundary for downstream callers: define a
//! versioned [`DiscoveryTask`], select a supported checkpoint with
//! [`ModelSelection`], validate a [`DiscoveryConfig`], and consume an honest
//! [`DiscoveryOutcome`]. The fixed GRPO + `LoRA` recipe and the lower-level
//! trainer assembly remain implementation details.
//!
//! Candidate acceptance is capability-shaped. A task's versioned verifier is
//! the semantic trust root: it returns typed evidence and an artifact payload.
//! Ferrl validates and persists that decision, binds it to the exact task data,
//! build/model identity, and authenticated candidate row, and alone publishes
//! the manifest-last bundle represented by [`VerifiedArtifact`]. Ferrl does not
//! independently prove arbitrary task semantics.
//!
//! Launches are labeled `local_ephemeral_v1`: the SDK signs candidates with a
//! process-local ephemeral key and binds that public key into the immutable
//! launch. This detects later row mutation and cross-launch substitution, but
//! it does not resist another process already controlling the same host UID.
//!
//! # Example
//!
//! ```no_run
//! use ferrl::discovery::{
//!     Candidate, CandidateVerification, Discovery, DiscoveryConfig, DiscoveryOutcome,
//!     DiscoveryTask, FinalEvidence, MetricContract, MetricDirection, MetricReport, ModelSelection,
//!     RewardError, RewardFn, Sample, TaskIdentity, TaskVerificationError,
//! };
//!
//! struct SearchReward;
//!
//! impl RewardFn for SearchReward {
//!     type Target = String;
//!
//!     fn reward(
//!         &self,
//!         sample: &Sample<Self::Target>,
//!         completion: &str,
//!     ) -> Result<f32, RewardError> {
//!         Ok(if completion.contains(&sample.target) { 1.0 } else { 0.0 })
//!     }
//! }
//!
//! struct TextTask {
//!     identity: TaskIdentity,
//!     train: Vec<Sample<String>>,
//!     held_out: Vec<Sample<String>>,
//!     reward: SearchReward,
//! }
//!
//! impl DiscoveryTask for TextTask {
//!     type Target = String;
//!     type SearchReward = SearchReward;
//!     type Artifact = String;
//!     type VerificationEvidence = String;
//!
//!     fn identity(&self) -> &TaskIdentity { &self.identity }
//!     fn training_samples(&self) -> &[Sample<Self::Target>] { &self.train }
//!     fn held_out_samples(&self) -> &[Sample<Self::Target>] { &self.held_out }
//!     fn search_reward(&self) -> &Self::SearchReward { &self.reward }
//!     fn metric_contract(&self) -> MetricContract {
//!         MetricContract::new(
//!             "throughput", "items/s", MetricDirection::HigherIsBetter, 10.0, 1.0,
//!         )
//!     }
//!
//!     fn verify_candidate(
//!         &self,
//!         candidate: Candidate<'_>,
//!     ) -> Result<
//!         CandidateVerification<Self::Artifact, Self::VerificationEvidence>,
//!         TaskVerificationError,
//!     > {
//!         let metric = MetricReport::new(
//!             "throughput",
//!             "items/s",
//!             MetricDirection::HigherIsBetter,
//!             10.0,
//!             12.0,
//!             1.0,
//!         );
//!         Ok(CandidateVerification::measured(FinalEvidence::new(
//!             candidate.completion().to_owned(),
//!             "task verifier accepted the candidate".to_owned(),
//!             true,
//!             metric,
//!         )))
//!     }
//! }
//!
//! # fn prepare() -> Result<(), ferrl::discovery::DiscoveryError> {
//! let task = TextTask {
//!     identity: TaskIdentity::new("example.text-search", 1)?,
//!     train: vec![Sample::new("say alpha", "alpha".to_owned())],
//!     held_out: vec![Sample::new("say beta", "beta".to_owned())],
//!     reward: SearchReward,
//! };
//! let config = DiscoveryConfig::builder("runs", "artifacts/winner")
//!     .steps(20)
//!     .group_size(8)
//!     .build()?;
//! let discovery = Discovery::new(task, ModelSelection::cpu("checkpoint"), config);
//! // `discovery.run()?` loads the checkpoint and performs the world-one run.
//! let _ = discovery;
//! # Ok(())
//! # }
//!
//! fn handle(outcome: DiscoveryOutcome) {
//!     match outcome {
//!         DiscoveryOutcome::Verified(artifact) => {
//!             println!("{}", artifact.manifest_path().display());
//!         }
//!         DiscoveryOutcome::NoWin(report) => {
//!             println!("no accepted win after {} candidates", report.candidates_checked());
//!         }
//!         DiscoveryOutcome::Preempted(report) => {
//!             println!("checkpoint saved at {}", report.checkpoint_path().display());
//!         }
//!     }
//! }
//! ```
//!
//! Accepted handles cannot be forged by tasks or callers:
//!
//! ```compile_fail
//! use ferrl::discovery::VerifiedArtifact;
//! use std::path::PathBuf;
//!
//! let forged = VerifiedArtifact {
//!     output: PathBuf::from("not-verified"),
//! };
//! ```

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use candle_core::{DType, Device, DeviceLocation};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eval::{evaluate, EvalError, EvalReport};
use crate::hf::{resolve_checkpoint_eos, CheckpointEosSelection, HfError};
use crate::loader::{load_auto_policy_with_identity, LoaderError, LoaderOpts, PolicyLoadIdentity};
use crate::policy::{EvalSampling, GenConfig, Policy};
use crate::telemetry::{CandidateRecord, CandidateSigner, RunDir, TelemetryError};
use crate::trainer::{RunStop, TokenizerLike, Trainer, TrainerConfig, TrainerError};

/// The search-reward error and extension trait used by [`DiscoveryTask`].
pub use crate::reward::{RewardError, RewardFn};
/// The typed sample used for discovery training and held-out evaluation.
pub use crate::sample::Sample;

const LAUNCH_CONTRACT: &str = "ferrl.discovery-launch.v1";
const ARTIFACT_CONTRACT: &str = "ferrl.discovery-artifact.v1";
const ARTIFACT_PAYLOAD_FILE: &str = "artifact.json";
const ARTIFACT_MANIFEST_FILE: &str = "manifest.json";
const ARTIFACT_LAUNCH_FILE: &str = "launch.json";
const ARTIFACT_HELD_OUT_FILE: &str = "eval-report.json";
const ARTIFACT_CANDIDATE_FILE: &str = "candidate.json";
const ARTIFACT_VERIFIER_EVIDENCE_FILE: &str = "verifier-evidence.json";
const HELD_OUT_REPORT_CONTRACT: &str = "ferrl.discovery-held-out-report.v1";
const VERIFIER_EVIDENCE_CONTRACT: &str = "ferrl.discovery-verifier-evidence.v1";
const LOCAL_EPHEMERAL_AUTHENTICATION: &str = "local_ephemeral_v1";
const LOCAL_EPHEMERAL_TRUST_BOUNDARY: &str = concat!(
    "same-process signed candidate binding; not resistant to another process ",
    "controlling the same UID"
);
const METRIC_LABEL_MAX_BYTES: usize = 128;

/// Stable identity of a task contract.
///
/// `name` identifies the task family and `version` identifies the exact sample,
/// reward, verifier, metric, and payload semantics implemented by that family.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TaskIdentity {
    name: String,
    version: u32,
}

impl TaskIdentity {
    /// Construct a validated task identity.
    ///
    /// Names contain only ASCII alphanumerics, `.`, `_`, and `-`, are at most
    /// 128 bytes, and versions start at one.
    ///
    /// # Errors
    ///
    /// Returns [`DiscoveryError::InvalidConfiguration`] for an invalid name or
    /// zero version.
    pub fn new(name: impl Into<String>, version: u32) -> Result<Self, DiscoveryError> {
        let name = name.into();
        let valid_name = !name.is_empty()
            && name.len() <= 128
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
        if !valid_name {
            return Err(DiscoveryError::InvalidConfiguration(
                "task name must be 1..=128 ASCII alphanumeric, '.', '_', or '-' bytes".into(),
            ));
        }
        if version == 0 {
            return Err(DiscoveryError::InvalidConfiguration(
                "task version must be >= 1".into(),
            ));
        }
        Ok(Self { name, version })
    }

    /// Task-family name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Version of the task's complete semantic contract.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }
}

/// World-one execution device selected for a supported checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDevice {
    /// Host CPU execution.
    Cpu,
    /// One CUDA device in the current process.
    Cuda {
        /// Zero-based CUDA ordinal.
        ordinal: usize,
    },
}

/// End-of-sequence selection for checkpoint-backed generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GenerationEnd {
    /// Require the checkpoint to declare exactly one valid EOS id.
    CheckpointDefault,
    /// Select one explicit EOS id, validated against checkpoint and tokenizer.
    Explicit(u32),
    /// Deliberately generate to the configured token width without EOS stopping.
    Disabled,
}

/// A supported checkpoint plus its world-one execution selection.
///
/// This type contains no model-family or provenance fields. Those values are
/// derived only by [`load_auto_policy_with_identity`] during [`Discovery::run`].
#[derive(Debug, Clone)]
pub struct ModelSelection {
    checkpoint_dir: PathBuf,
    device: ExecutionDevice,
    generation_end: GenerationEnd,
}

impl ModelSelection {
    /// Select a supported checkpoint for CPU execution.
    #[must_use]
    pub fn cpu(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            device: ExecutionDevice::Cpu,
            generation_end: GenerationEnd::CheckpointDefault,
        }
    }

    /// Select a supported checkpoint for one CUDA device.
    #[must_use]
    pub fn cuda(checkpoint_dir: impl Into<PathBuf>, ordinal: usize) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            device: ExecutionDevice::Cuda { ordinal },
            generation_end: GenerationEnd::CheckpointDefault,
        }
    }

    /// Override checkpoint-default EOS selection.
    #[must_use]
    pub fn generation_end(mut self, generation_end: GenerationEnd) -> Self {
        self.generation_end = generation_end;
        self
    }

    /// Selected checkpoint directory.
    #[must_use]
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Selected world-one execution device.
    #[must_use]
    pub fn device(&self) -> ExecutionDevice {
        self.device
    }
}

/// Loader-derived identity of the exact model and tokenizer used by a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelIdentity {
    policy_sha256: String,
    tokenizer_sha256: String,
    model_family: String,
    execution_device: ExecutionDevice,
}

impl ModelIdentity {
    fn from_loader(identity: PolicyLoadIdentity, execution_device: ExecutionDevice) -> Self {
        Self {
            policy_sha256: identity.policy_sha256,
            tokenizer_sha256: identity.tokenizer_sha256,
            model_family: identity.model_family.to_owned(),
            execution_device,
        }
    }

    /// Digest of exact checkpoint bytes and loader execution semantics.
    #[must_use]
    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }

    /// Digest of the exact tokenizer bytes loaded for the run.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// Model family derived from checkpoint configuration by the loader.
    #[must_use]
    pub fn model_family(&self) -> &str {
        &self.model_family
    }

    /// World-one device on which the model executed.
    #[must_use]
    pub fn execution_device(&self) -> ExecutionDevice {
        self.execution_device
    }
}

/// Direction in which a task metric improves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// A larger candidate measurement is better.
    HigherIsBetter,
    /// A smaller candidate measurement is better.
    LowerIsBetter,
}

/// Task-level final metric contract frozen into the launch before training.
///
/// This is a versioned [`DiscoveryTask`] extension point, not a selectable
/// discovery algorithm. Every measured candidate must report this exact name,
/// unit, direction, baseline, and materiality margin; only its candidate value
/// may vary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricContract {
    name: String,
    unit: String,
    direction: MetricDirection,
    baseline: f64,
    minimum_material_improvement: f64,
}

impl MetricContract {
    /// Construct the final metric contract owned by a task version.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        direction: MetricDirection,
        baseline: f64,
        minimum_material_improvement: f64,
    ) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            direction,
            baseline,
            minimum_material_improvement,
        }
    }

    /// Metric name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Metric unit.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Direction in which the metric improves.
    #[must_use]
    pub fn direction(&self) -> MetricDirection {
        self.direction
    }

    /// Frozen baseline measurement.
    #[must_use]
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// Directional improvement required before a result is material.
    #[must_use]
    pub fn minimum_material_improvement(&self) -> f64 {
        self.minimum_material_improvement
    }

    fn validate(&self) -> Result<(), DiscoveryError> {
        validate_metric_label("metric name", &self.name)?;
        validate_metric_label("metric unit", &self.unit)?;
        if !self.baseline.is_finite()
            || !self.minimum_material_improvement.is_finite()
            || self.minimum_material_improvement < 0.0
        {
            return Err(DiscoveryError::InvalidFinalEvidence(
                "metric baseline and non-negative materiality margin must be finite".into(),
            ));
        }
        Ok(())
    }
}

/// Final baseline/candidate measurement and its predeclared materiality margin.
///
/// Ferrl accepts only when the directional improvement is *strictly greater*
/// than `minimum_material_improvement`; equality is a no-win decision.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricReport {
    name: String,
    unit: String,
    direction: MetricDirection,
    baseline: f64,
    candidate: f64,
    minimum_material_improvement: f64,
}

impl MetricReport {
    /// Construct a task measurement for SDK validation.
    ///
    /// Construction is intentionally lossless. [`Discovery::run`] rejects empty
    /// labels, labels longer than 128 bytes, control characters, non-finite
    /// values, or a negative materiality margin before it can mint an accepted
    /// handle.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        direction: MetricDirection,
        baseline: f64,
        candidate: f64,
        minimum_material_improvement: f64,
    ) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            direction,
            baseline,
            candidate,
            minimum_material_improvement,
        }
    }

    /// Metric name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Metric unit.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Direction in which the metric improves.
    #[must_use]
    pub fn direction(&self) -> MetricDirection {
        self.direction
    }

    /// Baseline measurement.
    #[must_use]
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// Candidate measurement.
    #[must_use]
    pub fn candidate(&self) -> f64 {
        self.candidate
    }

    /// Directional improvement required before a result is material.
    #[must_use]
    pub fn minimum_material_improvement(&self) -> f64 {
        self.minimum_material_improvement
    }

    #[allow(clippy::cognitive_complexity)]
    fn validate(&self) -> Result<(), DiscoveryError> {
        validate_metric_label("metric name", &self.name)?;
        validate_metric_label("metric unit", &self.unit)?;
        if !self.baseline.is_finite()
            || !self.candidate.is_finite()
            || !self.minimum_material_improvement.is_finite()
            || self.minimum_material_improvement < 0.0
        {
            return Err(DiscoveryError::InvalidFinalEvidence(
                "baseline, candidate, and non-negative materiality margin must be finite".into(),
            ));
        }
        let improvement = self.directional_improvement();
        if !improvement.is_finite() {
            return Err(DiscoveryError::InvalidFinalEvidence(
                "directional metric improvement is outside the finite f64 domain".into(),
            ));
        }
        Ok(())
    }

    fn directional_improvement(&self) -> f64 {
        match self.direction {
            MetricDirection::HigherIsBetter => self.candidate - self.baseline,
            MetricDirection::LowerIsBetter => self.baseline - self.candidate,
        }
    }

    fn is_material_win(&self) -> bool {
        self.directional_improvement() > self.minimum_material_improvement
    }

    fn validate_against(&self, contract: &MetricContract) -> Result<(), DiscoveryError> {
        self.validate()?;
        let matches = self.name == contract.name
            && self.unit == contract.unit
            && self.direction == contract.direction
            && self.baseline.to_bits() == contract.baseline.to_bits()
            && self.minimum_material_improvement.to_bits()
                == contract.minimum_material_improvement.to_bits();
        if !matches {
            return Err(DiscoveryError::InvalidFinalEvidence(
                "candidate metric does not exactly match the launch-frozen metric contract".into(),
            ));
        }
        Ok(())
    }
}

fn validate_metric_label(label: &str, value: &str) -> Result<(), DiscoveryError> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.len() > METRIC_LABEL_MAX_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(DiscoveryError::InvalidFinalEvidence(format!(
            "{label} must be 1..={METRIC_LABEL_MAX_BYTES} bytes without surrounding whitespace \
             or control characters"
        )));
    }
    Ok(())
}

fn lexically_normalize_absolute(path: &Path) -> Result<PathBuf, DiscoveryError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| {
                DiscoveryError::InvalidConfiguration(format!(
                    "could not resolve relative discovery paths: {error}"
                ))
            })?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                normalized.push(component.as_os_str());
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    Ok(normalized)
}

/// Task-owned payload plus typed verifier, correctness, and metric evidence.
///
/// The task verifier is the semantic trust root. This is its claim, not an
/// accepted result: Ferrl validates the structural and launch-bound contract,
/// persists the typed evidence, and retains the sole constructor for
/// [`VerifiedArtifact`].
#[derive(Debug, Clone)]
pub struct FinalEvidence<A, V> {
    artifact: A,
    verification_evidence: V,
    held_out_correct: bool,
    metric: MetricReport,
}

impl<A, V> FinalEvidence<A, V> {
    /// Construct final task evidence for a measured candidate.
    #[must_use]
    pub fn new(
        artifact: A,
        verification_evidence: V,
        held_out_correct: bool,
        metric: MetricReport,
    ) -> Self {
        Self {
            artifact,
            verification_evidence,
            held_out_correct,
            metric,
        }
    }

    /// Whether the final verifier proved correctness on task-semantic held-out cases.
    #[must_use]
    pub fn held_out_correct(&self) -> bool {
        self.held_out_correct
    }

    /// Final baseline/candidate metric report.
    #[must_use]
    pub fn metric(&self) -> &MetricReport {
        &self.metric
    }

    /// Borrow the task artifact payload.
    #[must_use]
    pub fn artifact(&self) -> &A {
        &self.artifact
    }

    /// Borrow the task verifier's typed semantic evidence.
    #[must_use]
    pub fn verification_evidence(&self) -> &V {
        &self.verification_evidence
    }
}

/// Final verifier decision for one authenticated candidate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CandidateVerification<A, V> {
    /// The candidate was not valid enough to measure or accept.
    Rejected {
        /// Task-facing rejection reason retained in a no-win report.
        reason: String,
    },
    /// The verifier produced complete correctness, metric, and payload evidence.
    Measured(FinalEvidence<A, V>),
}

impl<A, V> CandidateVerification<A, V> {
    /// Construct a task-semantic rejection.
    #[must_use]
    pub fn rejected(reason: impl Into<String>) -> Self {
        Self::Rejected {
            reason: reason.into(),
        }
    }

    /// Construct a measured decision for SDK validation.
    #[must_use]
    pub fn measured(evidence: FinalEvidence<A, V>) -> Self {
        Self::Measured(evidence)
    }
}

/// Read-only view of an exact authenticated candidate row.
///
/// No constructor is exposed: instances come only from the SDK after ledger
/// provenance and world-one position checks succeed.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    record: &'a CandidateRecord,
    provenance_sha256: &'a str,
}

impl Candidate<'_> {
    /// Completion text exactly as scored during search.
    #[must_use]
    pub fn completion(&self) -> &str {
        &self.record.completion
    }

    /// Finite search reward recorded for this completion.
    #[must_use]
    pub fn search_reward(&self) -> f32 {
        self.record.reward
    }

    /// Zero-based optimizer step that sampled the candidate.
    #[must_use]
    pub fn step(&self) -> u64 {
        self.record.step
    }

    /// Global prompt ordinal that sampled the candidate.
    #[must_use]
    pub fn prompt_index(&self) -> u64 {
        self.record.prompt_index
    }

    /// Completion index within its sampled group.
    #[must_use]
    pub fn group_index(&self) -> usize {
        self.record.group_index
    }

    /// Digest binding every candidate field to its immutable launch.
    #[must_use]
    pub fn provenance_sha256(&self) -> &str {
        self.provenance_sha256
    }
}

/// Error returned by a task's distinct final candidate verifier.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TaskVerificationError {
    /// Message-only verifier failure.
    #[error("{0}")]
    Message(String),
    /// Failure from an underlying verifier, sandbox, benchmark, or I/O boundary.
    #[error("final task verifier failed: {0}")]
    Verifier(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl TaskVerificationError {
    /// Construct a message-only final-verifier error.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    /// Wrap an underlying final-verifier error.
    #[must_use]
    pub fn verifier(source: impl Into<Box<dyn std::error::Error + Send + Sync + 'static>>) -> Self {
        Self::Verifier(source.into())
    }
}

/// A versioned task extension for the stable discovery facade.
///
/// Training and held-out samples share one typed target and one search reward.
/// The versioned [`TaskIdentity`] binds the complete reward, target,
/// [`metric_contract`](Self::metric_contract), verifier, and artifact semantics.
/// [`verify_candidate`](Self::verify_candidate) is the task-semantic trust root
/// over an authenticated candidate; Ferrl does not independently prove those
/// arbitrary semantics.
pub trait DiscoveryTask {
    /// Typed ground-truth target carried by all samples.
    ///
    /// Ferrl serializes the exact ordered train and held-out slices, hashes
    /// those bytes into the immutable launch, then deserializes those same bytes
    /// into the owned samples actually consumed by reward execution.
    type Target: Serialize + DeserializeOwned;
    /// Scalar search reward optimized by the fixed internal recipe.
    type SearchReward: RewardFn<Target = Self::Target>;
    /// Generic artifact payload serialized only after acceptance.
    type Artifact: Serialize;
    /// Typed task-verifier evidence persisted separately from the artifact.
    type VerificationEvidence: Serialize;

    /// Exact task family and semantic version.
    fn identity(&self) -> &TaskIdentity;

    /// Non-empty typed training set seen by the search reward.
    fn training_samples(&self) -> &[Sample<Self::Target>];

    /// Non-empty, task-semantic held-out set used only after completed training.
    fn held_out_samples(&self) -> &[Sample<Self::Target>];

    /// Search reward bound to this task.
    fn search_reward(&self) -> &Self::SearchReward;

    /// Final metric contract frozen into the launch before training starts.
    fn metric_contract(&self) -> MetricContract;

    /// Independently verify and measure one exact authenticated candidate.
    ///
    /// A normal invalid candidate returns [`CandidateVerification::Rejected`].
    /// Infrastructure, sandbox, benchmark, or evidence-production failures must
    /// return [`TaskVerificationError`] and abort the run as operational errors.
    /// Ferrl checks every authenticated candidate and selects the strongest
    /// material final metric. Exact metric ties retain the candidate ranked
    /// first by search reward, then step, prompt ordinal, and group ordinal.
    ///
    /// # Errors
    ///
    /// Returns [`TaskVerificationError`] when final verification could not
    /// produce an honest decision.
    fn verify_candidate(
        &self,
        candidate: Candidate<'_>,
    ) -> Result<
        CandidateVerification<Self::Artifact, Self::VerificationEvidence>,
        TaskVerificationError,
    >;
}

/// Validated high-level controls for one fixed-recipe discovery run.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    runs_root: PathBuf,
    artifact_output: PathBuf,
    steps: u64,
    group_size: usize,
    max_new_tokens: usize,
    eval_group_size: usize,
    temperature: f64,
    learning_rate: f64,
    seed: u64,
    preemption_flag: Option<Arc<AtomicBool>>,
}

impl DiscoveryConfig {
    /// Start a builder with caller-selected run and accepted-artifact locations.
    #[must_use]
    pub fn builder(
        runs_root: impl Into<PathBuf>,
        artifact_output: impl Into<PathBuf>,
    ) -> DiscoveryConfigBuilder {
        DiscoveryConfigBuilder::new(runs_root, artifact_output)
    }

    /// Root under which the SDK creates its owned run identity.
    #[must_use]
    pub fn runs_root(&self) -> &Path {
        &self.runs_root
    }

    /// Exclusive destination created only for an accepted artifact.
    #[must_use]
    pub fn artifact_output(&self) -> &Path {
        &self.artifact_output
    }
}

/// Builder for [`DiscoveryConfig`].
#[derive(Debug, Clone)]
pub struct DiscoveryConfigBuilder {
    config: DiscoveryConfig,
}

impl DiscoveryConfigBuilder {
    /// Construct a builder with the fixed recipe's conservative defaults.
    #[must_use]
    pub fn new(runs_root: impl Into<PathBuf>, artifact_output: impl Into<PathBuf>) -> Self {
        Self {
            config: DiscoveryConfig {
                runs_root: runs_root.into(),
                artifact_output: artifact_output.into(),
                steps: 100,
                group_size: 8,
                max_new_tokens: 256,
                eval_group_size: 8,
                temperature: 1.0,
                learning_rate: 1e-3,
                seed: 1234,
                preemption_flag: None,
            },
        }
    }

    /// Set the number of GRPO optimizer steps.
    #[must_use]
    pub fn steps(mut self, steps: u64) -> Self {
        self.config.steps = steps;
        self
    }

    /// Set completions sampled per training prompt.
    #[must_use]
    pub fn group_size(mut self, group_size: usize) -> Self {
        self.config.group_size = group_size;
        self
    }

    /// Set the maximum generated completion width.
    #[must_use]
    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.config.max_new_tokens = max_new_tokens;
        self
    }

    /// Set completions sampled per held-out prompt for both base and adapter.
    #[must_use]
    pub fn eval_group_size(mut self, eval_group_size: usize) -> Self {
        self.config.eval_group_size = eval_group_size;
        self
    }

    /// Set the training rollout temperature.
    #[must_use]
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set the fixed recipe's `AdamW` learning rate.
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the loader-owned rollout sampler seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Install a cooperative preemption flag polled by the trainer.
    #[must_use]
    pub fn preemption_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.config.preemption_flag = Some(flag);
        self
    }

    /// Validate and finish the high-level configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DiscoveryError::InvalidConfiguration`] for an empty or
    /// colliding path, zero budget, group smaller than two, or a
    /// non-finite/non-positive scalar.
    pub fn build(self) -> Result<DiscoveryConfig, DiscoveryError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl DiscoveryConfig {
    #[allow(clippy::cognitive_complexity)]
    fn validate(&self) -> Result<(), DiscoveryError> {
        if self.runs_root.as_os_str().is_empty() {
            return Err(DiscoveryError::InvalidConfiguration(
                "runs_root must not be empty".into(),
            ));
        }
        if self.artifact_output.as_os_str().is_empty() {
            return Err(DiscoveryError::InvalidConfiguration(
                "artifact_output must not be empty".into(),
            ));
        }
        let normalized_runs = lexically_normalize_absolute(&self.runs_root)?;
        let normalized_artifact = lexically_normalize_absolute(&self.artifact_output)?;
        if normalized_runs.starts_with(&normalized_artifact) {
            return Err(DiscoveryError::InvalidConfiguration(format!(
                "artifact_output {} must not equal or contain runs_root {}",
                self.artifact_output.display(),
                self.runs_root.display()
            )));
        }
        if self.steps == 0 {
            return Err(DiscoveryError::InvalidConfiguration(
                "steps must be >= 1".into(),
            ));
        }
        if self.group_size < 2 {
            return Err(DiscoveryError::InvalidConfiguration(
                "group_size must be >= 2 for world-one GRPO".into(),
            ));
        }
        if self.max_new_tokens == 0 || self.eval_group_size == 0 {
            return Err(DiscoveryError::InvalidConfiguration(
                "max_new_tokens and eval_group_size must be >= 1".into(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(DiscoveryError::InvalidConfiguration(
                "temperature must be finite and > 0".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(DiscoveryError::InvalidConfiguration(
                "learning_rate must be finite and > 0".into(),
            ));
        }
        Ok(())
    }

    fn trainer_config(&self, eos_token_id: Option<u32>) -> TrainerConfig {
        TrainerConfig::builder()
            .steps(self.steps)
            .group_size(self.group_size)
            .max_new_tokens(self.max_new_tokens)
            .temperature(self.temperature)
            .lr(self.learning_rate)
            .checkpoint_every(Some(self.steps))
            .candidate_log_top_k(self.group_size)
            .eos_token_id(eos_token_id)
            .build()
    }

    fn eval_config(&self, eos_token_id: Option<u32>) -> GenConfig {
        GenConfig {
            group_size: self.eval_group_size,
            max_new_tokens: self.max_new_tokens,
            temperature: self.temperature,
            eos_token_id,
            eval_sampling: Some(EvalSampling::default()),
        }
    }
}

/// High-level world-one discovery runner.
#[derive(Debug)]
pub struct Discovery<T> {
    task: T,
    model: ModelSelection,
    config: DiscoveryConfig,
}

impl<T> Discovery<T> {
    /// Bind one task, model selection, and validated configuration.
    #[must_use]
    pub fn new(task: T, model: ModelSelection, config: DiscoveryConfig) -> Self {
        Self {
            task,
            model,
            config,
        }
    }
}

impl<T: DiscoveryTask> Discovery<T> {
    /// Load the selected supported checkpoint and run world-one discovery.
    ///
    /// GRPO + `LoRA`, candidate logging, checkpointing, loader options, and the
    /// world-one communicator are fixed inside the facade. A preemption stop is
    /// returned before held-out evaluation, candidate verification, or artifact
    /// publication.
    ///
    /// # Errors
    ///
    /// Returns [`DiscoveryError`] for invalid task/configuration data, device or
    /// model loading, training/evaluation, malformed or tampered candidates,
    /// final-verifier infrastructure failures, malformed final evidence, or
    /// exclusive artifact publication failure.
    pub fn run(self) -> Result<DiscoveryOutcome, DiscoveryError> {
        self.validate_before_load()?;
        let ferrl_source = BuildSourceIdentity::current()?;
        let metric_contract = self.task.metric_contract();
        metric_contract.validate()?;
        let device = open_device(self.model.device)?;
        let execution_device = execution_device_from_opened(&device, self.model.device)?;
        if matches!(self.model.device, ExecutionDevice::Cuda { .. }) {
            crate::guard_first_kernel(&device)
                .map_err(|error| DiscoveryError::Device(Box::new(error)))?;
        }
        let mut loader = LoaderOpts {
            seed: self.config.seed,
            temperature: self.config.temperature,
            ..LoaderOpts::default()
        };
        if matches!(self.model.device, ExecutionDevice::Cuda { .. }) {
            loader.base_dtype = DType::BF16;
        }
        let (mut policy, tokenizer, identity) =
            load_auto_policy_with_identity(&self.model.checkpoint_dir, &device, &loader)
                .map_err(|error| DiscoveryError::ModelLoad(Box::new(error)))?;
        let eos = resolve_checkpoint_eos(
            &self.model.checkpoint_dir,
            &tokenizer,
            checkpoint_eos_selection(self.model.generation_end),
        )
        .map_err(|error| DiscoveryError::GenerationEnd(Box::new(error)))?;
        let identity = ModelIdentity::from_loader(identity, execution_device);
        let context = LoadedDiscoveryContext {
            task: &self.task,
            config: &self.config,
            model: &identity,
            ferrl_source: &ferrl_source,
            metric_contract: &metric_contract,
            tokenizer: &tokenizer,
            eos_token_id: eos,
        };
        run_with_loaded_policy(&context, &mut policy)
    }

    #[allow(clippy::cognitive_complexity)]
    fn validate_before_load(&self) -> Result<(), DiscoveryError> {
        self.config.validate()?;
        if self.model.checkpoint_dir.as_os_str().is_empty() {
            return Err(DiscoveryError::InvalidConfiguration(
                "checkpoint_dir must not be empty".into(),
            ));
        }
        if self.task.training_samples().is_empty() {
            return Err(DiscoveryError::InvalidConfiguration(
                "discovery requires at least one training sample".into(),
            ));
        }
        if self.task.held_out_samples().is_empty() {
            return Err(DiscoveryError::InvalidConfiguration(
                "discovery requires at least one task-semantic held-out sample".into(),
            ));
        }
        if self.config.artifact_output.exists() {
            return Err(DiscoveryError::InvalidConfiguration(format!(
                "artifact output already exists: {}",
                self.config.artifact_output.display()
            )));
        }
        Ok(())
    }
}

/// Honest terminal state of a discovery run.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum DiscoveryOutcome {
    /// A held-out-correct, materially winning artifact was published.
    Verified(VerifiedArtifact),
    /// Training completed, but no candidate satisfied acceptance.
    NoWin(NoWinReport),
    /// Training stopped cooperatively before evaluation or acceptance.
    Preempted(PreemptedReport),
}

/// SDK-published accepted-artifact capability.
///
/// All fields and construction remain private. Possessing this value therefore
/// proves the SDK completed candidate binding, evidence validation, and
/// manifest-last publication during this process.
#[derive(Debug, Clone)]
pub struct VerifiedArtifact {
    output: PathBuf,
    manifest_path: PathBuf,
    payload_path: PathBuf,
    candidate_path: PathBuf,
    verification_evidence_path: PathBuf,
    task_identity: TaskIdentity,
    model_identity: ModelIdentity,
    candidate_sha256: String,
    metric: MetricReport,
}

impl VerifiedArtifact {
    /// Root of the exclusively created artifact bundle.
    #[must_use]
    pub fn output(&self) -> &Path {
        &self.output
    }

    /// Manifest-last commit marker for the accepted bundle.
    #[must_use]
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Serialized task artifact payload bound by the manifest.
    #[must_use]
    pub fn payload_path(&self) -> &Path {
        &self.payload_path
    }

    /// Exact canonical authenticated candidate JSON row copied into the bundle.
    #[must_use]
    pub fn candidate_path(&self) -> &Path {
        &self.candidate_path
    }

    /// Launch-bound typed evidence emitted by the task-semantic verifier.
    #[must_use]
    pub fn verification_evidence_path(&self) -> &Path {
        &self.verification_evidence_path
    }

    /// Versioned task identity bound into the manifest.
    #[must_use]
    pub fn task_identity(&self) -> &TaskIdentity {
        &self.task_identity
    }

    /// Loader-derived model identity bound into the manifest.
    #[must_use]
    pub fn model_identity(&self) -> &ModelIdentity {
        &self.model_identity
    }

    /// Signed provenance digest carried by the accepted candidate record.
    #[must_use]
    pub fn candidate_sha256(&self) -> &str {
        &self.candidate_sha256
    }

    /// Final metric decision that passed the strict material-win rule.
    #[must_use]
    pub fn metric(&self) -> &MetricReport {
        &self.metric
    }
}

/// Why a completed run produced no accepted artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NoWinReason {
    /// Every authenticated candidate was task-semantically rejected.
    CandidatesRejected,
    /// Measured candidates did not prove held-out correctness.
    HeldOutIncorrect,
    /// Correct candidates did not strictly clear their materiality margin.
    NoMaterialMetricWin,
}

/// Held-out base-versus-trained search-reward report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HeldOutReport {
    sample_count: usize,
    group_size: usize,
    base_reward_mean: f32,
    trained_reward_mean: f32,
}

impl HeldOutReport {
    fn from_eval(report: &EvalReport) -> Self {
        Self {
            sample_count: report.n_prompts,
            group_size: report.group_size,
            base_reward_mean: report.base_reward_mean,
            trained_reward_mean: report.adapter_reward_mean,
        }
    }

    /// Number of task-semantic held-out prompts.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Samples per prompt for each of base and trained policy.
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Base-policy mean search reward.
    #[must_use]
    pub fn base_reward_mean(&self) -> f32 {
        self.base_reward_mean
    }

    /// Trained-policy mean search reward.
    #[must_use]
    pub fn trained_reward_mean(&self) -> f32 {
        self.trained_reward_mean
    }
}

/// Completed run report with no accepted win.
#[derive(Debug, Clone)]
pub struct NoWinReport {
    run_dir: PathBuf,
    held_out: HeldOutReport,
    candidates_checked: usize,
    reason: NoWinReason,
    detail: String,
}

impl NoWinReport {
    /// SDK-owned run directory containing training and candidate evidence.
    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Held-out search-reward comparison completed before final verification.
    #[must_use]
    pub fn held_out(&self) -> &HeldOutReport {
        &self.held_out
    }

    /// Number of authenticated candidates passed to the final verifier.
    #[must_use]
    pub fn candidates_checked(&self) -> usize {
        self.candidates_checked
    }

    /// Machine-readable no-win category.
    #[must_use]
    pub fn reason(&self) -> &NoWinReason {
        &self.reason
    }

    /// Task/evidence detail for the final no-win decision.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Cooperative preemption report returned before evaluation or acceptance.
#[derive(Debug, Clone)]
pub struct PreemptedReport {
    run_dir: PathBuf,
    completed_steps: u64,
    checkpoint_path: PathBuf,
}

impl PreemptedReport {
    /// SDK-owned run directory containing partial metrics and candidates.
    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Number of optimizer steps completed before the stop.
    #[must_use]
    pub fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    /// Complete checkpoint saved at the stop.
    ///
    /// The discovery facade does not currently expose a high-level resume API.
    #[must_use]
    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }
}

/// Operational failure from the discovery facade.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DiscoveryError {
    /// Invalid high-level configuration, task data, or pre-existing output.
    #[error("invalid discovery configuration: {0}")]
    InvalidConfiguration(String),
    /// CPU/CUDA device creation or first-kernel preflight failed.
    #[error("discovery device error: {0}")]
    Device(#[source] Box<candle_core::Error>),
    /// Supported-checkpoint loading failed.
    #[error("discovery model load failed: {0}")]
    ModelLoad(#[source] Box<LoaderError>),
    /// Checkpoint/tokenizer EOS semantics could not be resolved.
    #[error("discovery generation-end resolution failed: {0}")]
    GenerationEnd(#[source] Box<HfError>),
    /// Immutable launch or run-directory setup failed.
    #[error("discovery launch setup failed: {0}")]
    Launch(#[source] Box<TelemetryError>),
    /// Fixed-recipe training failed.
    #[error("discovery training failed: {0}")]
    Training(#[source] Box<TrainerError>),
    /// Held-out evaluation failed.
    #[error("discovery held-out evaluation failed: {0}")]
    Evaluation(#[source] Box<EvalError>),
    /// Scanning complete ordinary checkpoints after preemption failed.
    #[error("discovery preemption checkpoint scan failed: {0}")]
    PreemptionCheckpointScan(#[source] Box<crate::checkpoint::CheckpointError>),
    /// Preemption checkpoint discovery or validation failed.
    #[error("discovery preemption checkpoint failed: {0}")]
    PreemptionCheckpoint(String),
    /// Reading the authenticated candidate ledger failed.
    #[error("failed to read candidate ledger {path}: {source}")]
    CandidateIo {
        /// Candidate ledger path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A candidate row was not valid JSON.
    #[error("invalid candidate JSON at {path} line {line}: {source}")]
    CandidateJson {
        /// Candidate ledger path.
        path: PathBuf,
        /// One-based JSONL line number.
        line: usize,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// Reading back the exact durable held-out report failed.
    #[error("failed to read held-out report {path}: {source}")]
    HeldOutReportIo {
        /// Held-out report path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Candidate coverage, world-one position, or launch provenance was invalid.
    #[error("invalid candidate evidence: {0}")]
    InvalidCandidateEvidence(String),
    /// The distinct task verifier failed operationally.
    #[error("final candidate verification failed: {0}")]
    TaskVerification(#[source] TaskVerificationError),
    /// Final evidence was malformed or non-finite.
    #[error("invalid final task evidence: {0}")]
    InvalidFinalEvidence(String),
    /// Generic payload or manifest serialization failed.
    #[error("failed to serialize {kind}: {source}")]
    Serialization {
        /// Artifact component being serialized.
        kind: &'static str,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// Exclusive manifest-last publication failed.
    #[error("failed to publish accepted artifact at {path}: {source}")]
    Publication {
        /// Path whose exclusive creation, write, or synchronization failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct BuildSourceIdentity {
    package_version: String,
    git_commit: String,
    git_dirty: bool,
}

impl BuildSourceIdentity {
    fn current() -> Result<Self, DiscoveryError> {
        let git_dirty = match env!("FERRL_BUILD_GIT_DIRTY") {
            "true" => true,
            "false" => false,
            value => {
                return Err(DiscoveryError::InvalidConfiguration(format!(
                    "FERRL_BUILD_GIT_DIRTY has invalid build value {value:?}"
                )))
            }
        };
        let identity = Self {
            package_version: env!("CARGO_PKG_VERSION").to_owned(),
            git_commit: env!("FERRL_BUILD_GIT_COMMIT").to_owned(),
            git_dirty,
        };
        identity.validate()?;
        Ok(identity)
    }

    fn validate(&self) -> Result<(), DiscoveryError> {
        let commit_is_valid = matches!(self.git_commit.len(), 40 | 64)
            && self
                .git_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'));
        if !commit_is_valid {
            return Err(DiscoveryError::InvalidConfiguration(
                "discovery requires a known lowercase 40- or 64-hex build commit".into(),
            ));
        }
        if self.git_dirty {
            return Err(DiscoveryError::InvalidConfiguration(
                "discovery refuses a dirty Ferrl build source identity".into(),
            ));
        }
        if self.package_version.trim().is_empty() {
            return Err(DiscoveryError::InvalidConfiguration(
                "discovery requires a non-empty Ferrl package version".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct LaunchPayload<'a> {
    contract: &'static str,
    contract_version: u32,
    launch_authentication: &'static str,
    launch_trust_boundary: &'static str,
    ferrl_source: &'a BuildSourceIdentity,
    task: &'a TaskIdentity,
    model: &'a ModelIdentity,
    metric_contract: &'a MetricContract,
    execution: ExecutionDevice,
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
struct LaunchManifest<'a> {
    contract: &'static str,
    launch_authentication: &'static str,
    launch_trust_boundary: &'static str,
    payload_sha256: &'a str,
    payload: &'a LaunchPayload<'a>,
}

#[derive(Serialize)]
struct PublishedPayload<'a, A> {
    contract: &'static str,
    contract_version: u32,
    task: &'a TaskIdentity,
    artifact: &'a A,
}

#[derive(Serialize)]
struct PublishedHeldOutReport<'a> {
    contract: &'static str,
    contract_version: u32,
    launch_sha256: &'a str,
    task: &'a TaskIdentity,
    held_out_samples_sha256: &'a str,
    held_out_samples_count: usize,
    report: &'a EvalReport,
}

#[derive(Serialize)]
struct ArtifactManifest<'a> {
    contract: &'static str,
    contract_version: u32,
    launch_authentication: &'static str,
    launch_trust_boundary: &'static str,
    ferrl_source: &'a BuildSourceIdentity,
    task: &'a TaskIdentity,
    model: &'a ModelIdentity,
    run_id: &'a str,
    launch_sha256: &'a str,
    candidate_signing_public_key: &'a str,
    candidate: &'a CandidateRecord,
    candidate_file: &'static str,
    candidate_file_sha256: &'a str,
    held_out_correct: bool,
    metric: &'a MetricReport,
    material_win: bool,
    artifact_payload_file: &'static str,
    artifact_payload_sha256: &'a str,
    launch_file: &'static str,
    launch_file_sha256: &'a str,
    held_out_report_file: &'static str,
    held_out_report_sha256: &'a str,
    verifier_evidence_file: &'static str,
    verifier_evidence_sha256: &'a str,
}

#[derive(Serialize)]
struct PublishedVerifierEvidence<'a, V> {
    contract: &'static str,
    contract_version: u32,
    launch_sha256: &'a str,
    task: &'a TaskIdentity,
    candidate_record_sha256: &'a str,
    candidate_file_sha256: &'a str,
    held_out_correct: bool,
    metric: &'a MetricReport,
    evidence: &'a V,
}

fn checkpoint_eos_selection(selection: GenerationEnd) -> CheckpointEosSelection {
    match selection {
        GenerationEnd::CheckpointDefault => CheckpointEosSelection::CheckpointDefault,
        GenerationEnd::Explicit(id) => CheckpointEosSelection::Explicit(id),
        GenerationEnd::Disabled => CheckpointEosSelection::Disabled,
    }
}

fn open_device(selection: ExecutionDevice) -> Result<Device, DiscoveryError> {
    match selection {
        ExecutionDevice::Cpu => Ok(Device::Cpu),
        ExecutionDevice::Cuda { ordinal } => {
            Device::new_cuda(ordinal).map_err(|error| DiscoveryError::Device(Box::new(error)))
        }
    }
}

fn execution_device_from_opened(
    device: &Device,
    selected: ExecutionDevice,
) -> Result<ExecutionDevice, DiscoveryError> {
    let opened = match device.location() {
        DeviceLocation::Cpu => ExecutionDevice::Cpu,
        DeviceLocation::Cuda { gpu_id } => ExecutionDevice::Cuda { ordinal: gpu_id },
        DeviceLocation::Metal { .. } => {
            return Err(DiscoveryError::InvalidConfiguration(
                "discovery does not support a Metal execution device".into(),
            ))
        }
    };
    if opened != selected {
        return Err(DiscoveryError::InvalidConfiguration(format!(
            "opened model device {opened:?} does not match selected device {selected:?}"
        )));
    }
    Ok(opened)
}

fn exact_execution_samples<T: Serialize + DeserializeOwned>(
    samples: &[Sample<T>],
    kind: &'static str,
) -> Result<(Vec<Sample<T>>, Vec<u8>), DiscoveryError> {
    let bytes = serde_json::to_vec(samples)
        .map_err(|source| DiscoveryError::Serialization { kind, source })?;
    let reconstructed = serde_json::from_slice(&bytes)
        .map_err(|source| DiscoveryError::Serialization { kind, source })?;
    Ok((reconstructed, bytes))
}

fn preflight_prompt_tokenization<T>(
    samples: &[Sample<T>],
    kind: &'static str,
    tokenizer: &dyn TokenizerLike,
) -> Result<(), DiscoveryError> {
    for (index, sample) in samples.iter().enumerate() {
        if tokenizer.encode(&sample.prompt).is_empty() {
            return Err(DiscoveryError::InvalidConfiguration(format!(
                "{kind} prompt at index {index} encoded to zero tokens with the loaded tokenizer"
            )));
        }
    }
    Ok(())
}

struct LoadedDiscoveryContext<'a, T, K> {
    task: &'a T,
    config: &'a DiscoveryConfig,
    model: &'a ModelIdentity,
    ferrl_source: &'a BuildSourceIdentity,
    metric_contract: &'a MetricContract,
    tokenizer: &'a K,
    eos_token_id: Option<u32>,
}

struct RankedFinalEvidence<A, V> {
    candidate_index: usize,
    evidence: FinalEvidence<A, V>,
}

#[allow(clippy::cognitive_complexity)]
fn run_with_loaded_policy<T, P, K>(
    context: &LoadedDiscoveryContext<'_, T, K>,
    policy: &mut P,
) -> Result<DiscoveryOutcome, DiscoveryError>
where
    T: DiscoveryTask,
    P: Policy,
    K: TokenizerLike,
{
    let task = context.task;
    let config = context.config;
    let model = context.model;
    let ferrl_source = context.ferrl_source;
    let metric_contract = context.metric_contract;
    let tokenizer = context.tokenizer;
    let eos_token_id = context.eos_token_id;
    ferrl_source.validate()?;
    metric_contract.validate()?;
    if task.training_samples().is_empty() || task.held_out_samples().is_empty() {
        return Err(DiscoveryError::InvalidConfiguration(
            "training and task-semantic held-out samples must both be non-empty".into(),
        ));
    }
    let (training_samples, training_samples_bytes) =
        exact_execution_samples(task.training_samples(), "ordered training samples")?;
    let (held_out_samples, held_out_samples_bytes) =
        exact_execution_samples(task.held_out_samples(), "ordered held-out samples")?;
    preflight_prompt_tokenization(&training_samples, "ordered training samples", tokenizer)?;
    preflight_prompt_tokenization(&held_out_samples, "ordered held-out samples", tokenizer)?;
    let training_samples_sha256 = sha256_hex(&training_samples_bytes);
    let held_out_samples_sha256 = sha256_hex(&held_out_samples_bytes);
    let signer =
        CandidateSigner::generate().map_err(|error| DiscoveryError::Launch(Box::new(error)))?;
    let signing_public_key = signer.public_key_hex();
    let launch_payload = LaunchPayload {
        contract: LAUNCH_CONTRACT,
        contract_version: 1,
        launch_authentication: LOCAL_EPHEMERAL_AUTHENTICATION,
        launch_trust_boundary: LOCAL_EPHEMERAL_TRUST_BOUNDARY,
        ferrl_source,
        task: task.identity(),
        model,
        metric_contract,
        execution: model.execution_device,
        resolved_eos_token_id: eos_token_id,
        training_samples_sha256: &training_samples_sha256,
        training_samples_count: training_samples.len(),
        held_out_samples_sha256: &held_out_samples_sha256,
        held_out_samples_count: held_out_samples.len(),
        steps: config.steps,
        group_size: config.group_size,
        max_new_tokens: config.max_new_tokens,
        eval_group_size: config.eval_group_size,
        temperature: config.temperature,
        learning_rate: config.learning_rate,
        seed: config.seed,
        candidate_signing_public_key: &signing_public_key,
    };
    let payload_bytes =
        serde_json::to_vec(&launch_payload).map_err(|source| DiscoveryError::Serialization {
            kind: "discovery launch payload",
            source,
        })?;
    let launch_sha256 = sha256_hex(&payload_bytes);
    let run_id = format!("discovery-{}", &launch_sha256[..20]);
    let launch = LaunchManifest {
        contract: LAUNCH_CONTRACT,
        launch_authentication: LOCAL_EPHEMERAL_AUTHENTICATION,
        launch_trust_boundary: LOCAL_EPHEMERAL_TRUST_BOUNDARY,
        payload_sha256: &launch_sha256,
        payload: &launch_payload,
    };
    let mut launch_bytes =
        serde_json::to_vec_pretty(&launch).map_err(|source| DiscoveryError::Serialization {
            kind: "discovery launch manifest",
            source,
        })?;
    launch_bytes.push(b'\n');

    let run = RunDir::create(&config.runs_root, &run_id)
        .map_err(|error| DiscoveryError::Launch(Box::new(error)))?;
    run.write_immutable_launch(&launch_bytes, None)
        .map_err(|error| DiscoveryError::Launch(Box::new(error)))?;
    let trainer_config = config.trainer_config(eos_token_id);
    if trainer_config.candidate_log_top_k != trainer_config.group_size {
        return Err(DiscoveryError::InvalidConfiguration(
            "discovery requires complete candidate logging".into(),
        ));
    }
    let mut trainer = Trainer::new(trainer_config, &run)
        .map_err(|error| DiscoveryError::Training(Box::new(error)))?
        .with_frozen_policy_sha256(model.policy_sha256.clone())
        .with_candidate_provenance(&launch_sha256, signer)
        .map_err(|error| DiscoveryError::Training(Box::new(error)))?;
    if let Some(flag) = config.preemption_flag.clone() {
        trainer = trainer.with_preemption_flag(flag);
    }
    let (history, stop) = trainer
        .train(policy, task.search_reward(), tokenizer, &training_samples)
        .map_err(|error| DiscoveryError::Training(Box::new(error)))?;
    if stop == RunStop::Preempted {
        let completed_steps = u64::try_from(history.len()).map_err(|_| {
            DiscoveryError::PreemptionCheckpoint(
                "completed history length does not fit the checkpoint step domain".into(),
            )
        })?;
        let latest = crate::latest_checkpoint(run.checkpoints_dir())
            .map_err(|error| DiscoveryError::PreemptionCheckpointScan(Box::new(error)))?
            .ok_or_else(|| {
                DiscoveryError::PreemptionCheckpoint(
                    "trainer returned Preempted without a complete checkpoint".into(),
                )
            })?;
        if latest.step != completed_steps {
            return Err(DiscoveryError::PreemptionCheckpoint(format!(
                "newest complete checkpoint step {} does not match completed history length \
                 {completed_steps}",
                latest.step
            )));
        }
        return Ok(DiscoveryOutcome::Preempted(PreemptedReport {
            run_dir: run.root().to_path_buf(),
            completed_steps: latest.step,
            checkpoint_path: latest.dir,
        }));
    }

    let eval = evaluate(
        policy,
        task.search_reward(),
        tokenizer,
        &held_out_samples,
        &config.eval_config(eos_token_id),
    )
    .map_err(|error| DiscoveryError::Evaluation(Box::new(error)))?;
    let held_out = HeldOutReport::from_eval(&eval);
    let published_held_out = PublishedHeldOutReport {
        contract: HELD_OUT_REPORT_CONTRACT,
        contract_version: 1,
        launch_sha256: &launch_sha256,
        task: task.identity(),
        held_out_samples_sha256: &held_out_samples_sha256,
        held_out_samples_count: held_out_samples.len(),
        report: &eval,
    };
    let mut expected_held_out_bytes =
        serde_json::to_vec_pretty(&published_held_out).map_err(|source| {
            DiscoveryError::Serialization {
                kind: "launch-bound held-out report",
                source,
            }
        })?;
    expected_held_out_bytes.push(b'\n');
    run.write_eval_report(&published_held_out)
        .map_err(|error| DiscoveryError::Launch(Box::new(error)))?;
    let held_out_report_bytes =
        fs::read(run.eval_report_path()).map_err(|source| DiscoveryError::HeldOutReportIo {
            path: run.eval_report_path(),
            source,
        })?;
    if held_out_report_bytes != expected_held_out_bytes {
        return Err(DiscoveryError::InvalidCandidateEvidence(
            "published held-out report bytes differ from the launch-bound report".into(),
        ));
    }
    let mut candidates = load_and_validate_candidates(
        &run.candidates_path(),
        &launch_sha256,
        &signing_public_key,
        config.steps,
        config.group_size,
        config.max_new_tokens,
    )?;
    candidates.sort_by(|left, right| {
        right
            .record
            .reward
            .total_cmp(&left.record.reward)
            .then_with(|| left.record.step.cmp(&right.record.step))
            .then_with(|| left.record.prompt_index.cmp(&right.record.prompt_index))
            .then_with(|| left.record.group_index.cmp(&right.record.group_index))
    });

    let mut checked = 0;
    let mut saw_measured = false;
    let mut last_reason = NoWinReason::CandidatesRejected;
    let mut last_detail = "every candidate was rejected by the final verifier".to_owned();
    let mut best: Option<RankedFinalEvidence<T::Artifact, T::VerificationEvidence>> = None;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        checked += 1;
        let decision = task
            .verify_candidate(Candidate {
                record: &candidate.record,
                provenance_sha256: candidate.provenance_sha256.as_str(),
            })
            .map_err(DiscoveryError::TaskVerification)?;
        let evidence = match decision {
            CandidateVerification::Rejected { reason } => {
                if !saw_measured {
                    last_reason = NoWinReason::CandidatesRejected;
                    last_detail = if reason.trim().is_empty() {
                        "candidate rejected without task detail".into()
                    } else {
                        reason
                    };
                }
                continue;
            }
            CandidateVerification::Measured(evidence) => evidence,
        };
        saw_measured = true;
        evidence.metric.validate_against(metric_contract)?;
        if !evidence.held_out_correct {
            last_reason = NoWinReason::HeldOutIncorrect;
            last_detail = "final verifier did not prove task-semantic held-out correctness".into();
            continue;
        }
        if !evidence.metric.is_material_win() {
            last_reason = NoWinReason::NoMaterialMetricWin;
            last_detail = format!(
                "directional improvement {} did not strictly exceed materiality margin {}",
                evidence.metric.directional_improvement(),
                evidence.metric.minimum_material_improvement
            );
            continue;
        }
        let replace = match best.as_ref() {
            None => true,
            Some(current) => metric_is_stronger(
                &evidence.metric,
                &current.evidence.metric,
                metric_contract.direction,
            ),
        };
        if replace {
            best = Some(RankedFinalEvidence {
                candidate_index,
                evidence,
            });
        }
    }
    if let Some(best) = best {
        let artifact = publish_verified_artifact(
            &config.artifact_output,
            task.identity(),
            model,
            ferrl_source,
            &run_id,
            &launch_sha256,
            &signing_public_key,
            &candidates[best.candidate_index],
            &launch_bytes,
            &held_out_report_bytes,
            best.evidence,
        )?;
        return Ok(DiscoveryOutcome::Verified(artifact));
    }
    Ok(DiscoveryOutcome::NoWin(NoWinReport {
        run_dir: run.root().to_path_buf(),
        held_out,
        candidates_checked: checked,
        reason: last_reason,
        detail: last_detail,
    }))
}

fn metric_is_stronger(
    candidate: &MetricReport,
    current: &MetricReport,
    direction: MetricDirection,
) -> bool {
    match direction {
        MetricDirection::HigherIsBetter => candidate.candidate > current.candidate,
        MetricDirection::LowerIsBetter => candidate.candidate < current.candidate,
    }
}

#[derive(Debug)]
struct AuthenticatedCandidate {
    record: CandidateRecord,
    exact_row_bytes: Vec<u8>,
    provenance_sha256: String,
}

#[allow(clippy::cognitive_complexity)]
fn load_and_validate_candidates(
    path: &Path,
    launch_sha256: &str,
    signing_public_key: &str,
    steps: u64,
    group_size: usize,
    max_new_tokens: usize,
) -> Result<Vec<AuthenticatedCandidate>, DiscoveryError> {
    let file = File::open(path).map_err(|source| DiscoveryError::CandidateIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut records = Vec::new();
    let mut positions = BTreeSet::new();
    let mut reader = BufReader::new(file);
    let mut line_number = 0_usize;
    loop {
        let mut exact_row_bytes = Vec::new();
        let read = reader
            .read_until(b'\n', &mut exact_row_bytes)
            .map_err(|source| DiscoveryError::CandidateIo {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        line_number += 1;
        if exact_row_bytes.last() != Some(&b'\n') {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} is missing its canonical JSONL newline"
            )));
        }
        let json_bytes = &exact_row_bytes[..exact_row_bytes.len() - 1];
        if json_bytes.iter().all(u8::is_ascii_whitespace) {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "blank candidate row at line {line_number}"
            )));
        }
        let record: CandidateRecord =
            serde_json::from_slice(json_bytes).map_err(|source| DiscoveryError::CandidateJson {
                path: path.to_path_buf(),
                line: line_number,
                source,
            })?;
        let mut canonical =
            serde_json::to_vec(&record).map_err(|source| DiscoveryError::Serialization {
                kind: "canonical authenticated candidate row",
                source,
            })?;
        canonical.push(b'\n');
        if canonical != exact_row_bytes {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} is not the canonical CandidateWriter JSONL encoding"
            )));
        }
        record
            .verify_signed_provenance(signing_public_key)
            .map_err(|source| {
                DiscoveryError::InvalidCandidateEvidence(format!(
                    "candidate line {line_number} failed signed provenance: {source}"
                ))
            })?;
        if record.launch_sha256.as_deref() != Some(launch_sha256) {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} is bound to a different launch"
            )));
        }
        if record.rank != 0 || record.world_size != 1 {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} is not world-one rank 0"
            )));
        }
        if record.step >= steps
            || record.prompt_index != record.step
            || record.group_index >= group_size
        {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} has an impossible training position"
            )));
        }
        if record.completion_len_tokens == 0 || record.completion_len_tokens > max_new_tokens {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} completion length {} is outside the launch-bound \
                 range 1..={max_new_tokens}",
                record.completion_len_tokens
            )));
        }
        if !positions.insert((record.step, record.prompt_index, record.group_index)) {
            return Err(DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} duplicates a training position"
            )));
        }
        let provenance_sha256 = record.record_sha256.clone().ok_or_else(|| {
            DiscoveryError::InvalidCandidateEvidence(format!(
                "candidate line {line_number} has no validated provenance digest"
            ))
        })?;
        records.push(AuthenticatedCandidate {
            record,
            exact_row_bytes,
            provenance_sha256,
        });
    }
    let steps_usize = usize::try_from(steps).map_err(|_| {
        DiscoveryError::InvalidCandidateEvidence(
            "configured step count does not fit candidate coverage arithmetic".into(),
        )
    })?;
    let expected = steps_usize.checked_mul(group_size).ok_or_else(|| {
        DiscoveryError::InvalidCandidateEvidence("candidate coverage count overflows usize".into())
    })?;
    if records.len() != expected {
        return Err(DiscoveryError::InvalidCandidateEvidence(format!(
            "candidate ledger has {} rows, expected complete logging of {expected}",
            records.len()
        )));
    }
    Ok(records)
}

#[allow(clippy::too_many_arguments)]
fn publish_verified_artifact<A: Serialize, V: Serialize>(
    output: &Path,
    task: &TaskIdentity,
    model: &ModelIdentity,
    ferrl_source: &BuildSourceIdentity,
    run_id: &str,
    launch_sha256: &str,
    signing_public_key: &str,
    candidate: &AuthenticatedCandidate,
    launch_bytes: &[u8],
    held_out_report_bytes: &[u8],
    evidence: FinalEvidence<A, V>,
) -> Result<VerifiedArtifact, DiscoveryError> {
    publish_verified_artifact_with_fault(
        output,
        task,
        model,
        ferrl_source,
        run_id,
        launch_sha256,
        signing_public_key,
        candidate,
        launch_bytes,
        held_out_report_bytes,
        evidence,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn publish_verified_artifact_with_fault<A: Serialize, V: Serialize>(
    output: &Path,
    task: &TaskIdentity,
    model: &ModelIdentity,
    ferrl_source: &BuildSourceIdentity,
    run_id: &str,
    launch_sha256: &str,
    signing_public_key: &str,
    candidate: &AuthenticatedCandidate,
    launch_bytes: &[u8],
    held_out_report_bytes: &[u8],
    evidence: FinalEvidence<A, V>,
    fail_after_payload_links: Option<usize>,
) -> Result<VerifiedArtifact, DiscoveryError> {
    let FinalEvidence {
        artifact,
        verification_evidence,
        held_out_correct,
        metric,
    } = evidence;
    let payload = PublishedPayload {
        contract: ARTIFACT_CONTRACT,
        contract_version: 1,
        task,
        artifact: &artifact,
    };
    let mut payload_bytes =
        serde_json::to_vec_pretty(&payload).map_err(|source| DiscoveryError::Serialization {
            kind: "accepted artifact payload",
            source,
        })?;
    payload_bytes.push(b'\n');
    let payload_sha256 = sha256_hex(&payload_bytes);
    let launch_file_sha256 = sha256_hex(launch_bytes);
    let held_out_report_sha256 = sha256_hex(held_out_report_bytes);
    let candidate_file_sha256 = sha256_hex(&candidate.exact_row_bytes);
    let candidate_record_sha256 = candidate.provenance_sha256.as_str();
    let published_verifier_evidence = PublishedVerifierEvidence {
        contract: VERIFIER_EVIDENCE_CONTRACT,
        contract_version: 1,
        launch_sha256,
        task,
        candidate_record_sha256,
        candidate_file_sha256: &candidate_file_sha256,
        held_out_correct,
        metric: &metric,
        evidence: &verification_evidence,
    };
    let mut verifier_evidence_bytes = serde_json::to_vec_pretty(&published_verifier_evidence)
        .map_err(|source| DiscoveryError::Serialization {
            kind: "launch-bound verifier evidence",
            source,
        })?;
    verifier_evidence_bytes.push(b'\n');
    let verifier_evidence_sha256 = sha256_hex(&verifier_evidence_bytes);
    let manifest = ArtifactManifest {
        contract: ARTIFACT_CONTRACT,
        contract_version: 1,
        launch_authentication: LOCAL_EPHEMERAL_AUTHENTICATION,
        launch_trust_boundary: LOCAL_EPHEMERAL_TRUST_BOUNDARY,
        ferrl_source,
        task,
        model,
        run_id,
        launch_sha256,
        candidate_signing_public_key: signing_public_key,
        candidate: &candidate.record,
        candidate_file: ARTIFACT_CANDIDATE_FILE,
        candidate_file_sha256: &candidate_file_sha256,
        held_out_correct,
        metric: &metric,
        material_win: true,
        artifact_payload_file: ARTIFACT_PAYLOAD_FILE,
        artifact_payload_sha256: &payload_sha256,
        launch_file: ARTIFACT_LAUNCH_FILE,
        launch_file_sha256: &launch_file_sha256,
        held_out_report_file: ARTIFACT_HELD_OUT_FILE,
        held_out_report_sha256: &held_out_report_sha256,
        verifier_evidence_file: ARTIFACT_VERIFIER_EVIDENCE_FILE,
        verifier_evidence_sha256: &verifier_evidence_sha256,
    };
    let mut manifest_bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|source| DiscoveryError::Serialization {
            kind: "accepted artifact manifest",
            source,
        })?;
    manifest_bytes.push(b'\n');

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| DiscoveryError::Publication {
        path: parent.to_path_buf(),
        source,
    })?;
    sync_directory(parent)?;
    fs::create_dir(output).map_err(|source| DiscoveryError::Publication {
        path: output.to_path_buf(),
        source,
    })?;
    let stage_dir = artifact_staging_path(output)?;
    create_private_directory(&stage_dir)?;
    sync_directory(parent)?;

    let staged_files: [(&str, &[u8]); 5] = [
        (ARTIFACT_PAYLOAD_FILE, &payload_bytes),
        (ARTIFACT_LAUNCH_FILE, launch_bytes),
        (ARTIFACT_HELD_OUT_FILE, held_out_report_bytes),
        (ARTIFACT_CANDIDATE_FILE, &candidate.exact_row_bytes),
        (ARTIFACT_VERIFIER_EVIDENCE_FILE, &verifier_evidence_bytes),
    ];
    for &(name, bytes) in &staged_files {
        write_new_synced(&stage_dir.join(name), bytes)?;
    }
    write_new_synced(&stage_dir.join(ARTIFACT_MANIFEST_FILE), &manifest_bytes)?;
    sync_directory(&stage_dir)?;

    let mut linked = 0_usize;
    for &(name, _) in &staged_files {
        let destination = output.join(name);
        fs::hard_link(stage_dir.join(name), &destination).map_err(|source| {
            DiscoveryError::Publication {
                path: destination,
                source,
            }
        })?;
        linked += 1;
        if fail_after_payload_links == Some(linked) {
            return Err(DiscoveryError::Publication {
                path: output.to_path_buf(),
                source: std::io::Error::other("injected artifact mid-publication failure"),
            });
        }
    }
    sync_directory(output)?;
    let payload_path = output.join(ARTIFACT_PAYLOAD_FILE);
    let candidate_path = output.join(ARTIFACT_CANDIDATE_FILE);
    let verification_evidence_path = output.join(ARTIFACT_VERIFIER_EVIDENCE_FILE);
    let manifest_path = output.join(ARTIFACT_MANIFEST_FILE);
    fs::hard_link(stage_dir.join(ARTIFACT_MANIFEST_FILE), &manifest_path).map_err(|source| {
        DiscoveryError::Publication {
            path: manifest_path.clone(),
            source,
        }
    })?;
    sync_directory(output)?;
    sync_directory(parent)?;

    Ok(VerifiedArtifact {
        output: output.to_path_buf(),
        manifest_path,
        payload_path,
        candidate_path,
        verification_evidence_path,
        task_identity: task.clone(),
        model_identity: model.clone(),
        candidate_sha256: candidate.provenance_sha256.clone(),
        metric,
    })
}

fn artifact_staging_path(output: &Path) -> Result<PathBuf, DiscoveryError> {
    let name = output.file_name().ok_or_else(|| {
        DiscoveryError::InvalidConfiguration(format!(
            "artifact_output has no final directory name: {}",
            output.display()
        ))
    })?;
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut stage_name = OsString::from(".");
    stage_name.push(name);
    stage_name.push(".ferrl-discovery-stage");
    Ok(parent.join(stage_name))
}

fn create_private_directory(path: &Path) -> Result<(), DiscoveryError> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .map_err(|source| DiscoveryError::Publication {
            path: path.to_path_buf(),
            source,
        })
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), DiscoveryError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| DiscoveryError::Publication {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| DiscoveryError::Publication {
            path: path.to_path_buf(),
            source,
        })
}

fn sync_directory(path: &Path) -> Result<(), DiscoveryError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| DiscoveryError::Publication {
            path: path.to_path_buf(),
            source,
        })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use candle_core::{Result as CandleResult, Tensor, Var};

    use super::*;
    use crate::policy::Rollout;

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let serial = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("test clock must follow the Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "ferrl-discovery-{label}-{}-{nanos}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct TestTokenizer;

    impl TokenizerLike for TestTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            vec![1]
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    struct FailingTokenizer {
        fail_prompt: &'static str,
    }

    impl TokenizerLike for FailingTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            if text == self.fail_prompt {
                Vec::new()
            } else {
                vec![1]
            }
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    struct TestPolicy {
        logprobs: Var,
        adapter_enabled: bool,
        generate_calls: usize,
        token_logprobs_calls: Cell<usize>,
    }

    impl TestPolicy {
        fn new() -> Self {
            let logprobs =
                Var::from_tensor(&Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap())
                    .unwrap();
            Self {
                logprobs,
                adapter_enabled: true,
                generate_calls: 0,
                token_logprobs_calls: Cell::new(0),
            }
        }

        fn generate_calls(&self) -> usize {
            self.generate_calls
        }

        fn token_logprobs_calls(&self) -> usize {
            self.token_logprobs_calls.get()
        }
    }

    impl Policy for TestPolicy {
        fn generate(&mut self, prompt: &[u32], config: &GenConfig) -> CandleResult<Rollout> {
            self.generate_calls += 1;
            let rows = (0..config.group_size)
                .map(|index| {
                    let mut row = prompt.to_vec();
                    let token = if self.adapter_enabled && index % 2 == 1 {
                        9
                    } else {
                        7
                    };
                    row.extend(std::iter::repeat_n(token, config.max_new_tokens));
                    row
                })
                .collect::<Vec<_>>();
            Ok(Rollout::new(
                rows,
                prompt.len(),
                vec![config.max_new_tokens; config.group_size],
                None,
            ))
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            self.token_logprobs_calls
                .set(self.token_logprobs_calls.get().saturating_add(1));
            Ok(self.logprobs.as_tensor().clone())
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.adapter_enabled = enabled;
        }

        fn adapter_enabled(&self) -> bool {
            self.adapter_enabled
        }

        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logprobs.clone()]
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
            if !state.is_empty() {
                candle_core::bail!("test policy has no sampler state");
            }
            Ok(())
        }

        fn lora_recipe(&self) -> Option<String> {
            Some("test-lora-v1".into())
        }
    }

    #[derive(Default)]
    struct TestReward {
        calls: Cell<usize>,
    }

    impl TestReward {
        fn calls(&self) -> usize {
            self.calls.get()
        }
    }

    impl RewardFn for TestReward {
        type Target = ();

        fn reward(
            &self,
            _sample: &Sample<Self::Target>,
            completion: &str,
        ) -> Result<f32, RewardError> {
            self.calls.set(self.calls.get().saturating_add(1));
            Ok(if completion.starts_with('9') {
                1.0
            } else {
                0.0
            })
        }
    }

    #[derive(Clone, Copy)]
    enum VerifyMode {
        Win,
        BetterFinalMetricBeatsSearchReward,
        NoWin,
        Incorrect,
        NonFinite,
        MalformedLabel,
        MismatchedMetricContract,
        OperationalFailure,
    }

    impl VerifyMode {
        fn measurement(
            self,
            completion: &str,
        ) -> Result<Option<(bool, f64, f64)>, TaskVerificationError> {
            match (self, completion.starts_with('7')) {
                (Self::OperationalFailure, _) => {
                    Err(TaskVerificationError::msg("final verifier unavailable"))
                }
                (Self::BetterFinalMetricBeatsSearchReward, true) => Ok(Some((true, 15.0, 1.0))),
                (
                    Self::BetterFinalMetricBeatsSearchReward
                    | Self::Win
                    | Self::MalformedLabel
                    | Self::MismatchedMetricContract,
                    false,
                ) => Ok(Some((true, 12.0, 1.0))),
                (_, true) => Ok(None),
                (Self::NoWin, false) => Ok(Some((true, 11.0, 1.0))),
                (Self::Incorrect, false) => Ok(Some((false, 12.0, 1.0))),
                (Self::NonFinite, false) => Ok(Some((true, f64::NAN, 1.0))),
            }
        }

        fn metric_name(self) -> &'static str {
            match self {
                Self::MalformedLabel => "throughput\nforged",
                _ => "throughput",
            }
        }

        fn metric_baseline(self) -> f64 {
            match self {
                Self::MismatchedMetricContract => 9.0,
                _ => 10.0,
            }
        }
    }

    struct TestTask {
        identity: TaskIdentity,
        train: Vec<Sample<()>>,
        held_out: Vec<Sample<()>>,
        reward: TestReward,
        mode: VerifyMode,
    }

    impl TestTask {
        fn new(mode: VerifyMode) -> Self {
            Self::with_samples(
                mode,
                vec![Sample::new("train", ())],
                vec![Sample::new("held-out", ())],
            )
        }

        fn with_samples(
            mode: VerifyMode,
            train: Vec<Sample<()>>,
            held_out: Vec<Sample<()>>,
        ) -> Self {
            Self {
                identity: TaskIdentity::new("test.discovery", 1).unwrap(),
                train,
                held_out,
                reward: TestReward::default(),
                mode,
            }
        }

        fn reward_calls(&self) -> usize {
            self.reward.calls()
        }
    }

    impl DiscoveryTask for TestTask {
        type Target = ();
        type SearchReward = TestReward;
        type Artifact = String;
        type VerificationEvidence = String;

        fn identity(&self) -> &TaskIdentity {
            &self.identity
        }

        fn training_samples(&self) -> &[Sample<Self::Target>] {
            &self.train
        }

        fn held_out_samples(&self) -> &[Sample<Self::Target>] {
            &self.held_out
        }

        fn search_reward(&self) -> &Self::SearchReward {
            &self.reward
        }

        fn metric_contract(&self) -> MetricContract {
            MetricContract::new(
                "throughput",
                "items/s",
                MetricDirection::HigherIsBetter,
                10.0,
                1.0,
            )
        }

        fn verify_candidate(
            &self,
            candidate: Candidate<'_>,
        ) -> Result<
            CandidateVerification<Self::Artifact, Self::VerificationEvidence>,
            TaskVerificationError,
        > {
            let Some((correct, measured, margin)) =
                self.mode.measurement(candidate.completion())?
            else {
                return Ok(CandidateVerification::rejected("not the target token"));
            };
            let metric = MetricReport::new(
                self.mode.metric_name(),
                "items/s",
                MetricDirection::HigherIsBetter,
                self.mode.metric_baseline(),
                measured,
                margin,
            );
            Ok(CandidateVerification::measured(FinalEvidence::new(
                candidate.completion().to_owned(),
                format!("verified completion {}", candidate.completion()),
                correct,
                metric,
            )))
        }
    }

    fn test_model() -> ModelIdentity {
        ModelIdentity {
            policy_sha256: "11".repeat(32),
            tokenizer_sha256: "22".repeat(32),
            model_family: "test".into(),
            execution_device: ExecutionDevice::Cpu,
        }
    }

    fn clean_test_source() -> BuildSourceIdentity {
        BuildSourceIdentity {
            package_version: env!("CARGO_PKG_VERSION").into(),
            git_commit: "ab".repeat(20),
            git_dirty: false,
        }
    }

    fn test_launch_sha(
        task: &TestTask,
        training_samples_sha256: &str,
        held_out_samples_sha256: &str,
        resolved_eos_token_id: Option<u32>,
    ) -> String {
        test_launch_sha_with_source(
            task,
            training_samples_sha256,
            held_out_samples_sha256,
            resolved_eos_token_id,
            &clean_test_source(),
        )
    }

    fn test_launch_sha_with_source(
        task: &TestTask,
        training_samples_sha256: &str,
        held_out_samples_sha256: &str,
        resolved_eos_token_id: Option<u32>,
        ferrl_source: &BuildSourceIdentity,
    ) -> String {
        let model = test_model();
        let metric_contract = task.metric_contract();
        let signing_key = "33".repeat(32);
        let payload = LaunchPayload {
            contract: LAUNCH_CONTRACT,
            contract_version: 1,
            launch_authentication: LOCAL_EPHEMERAL_AUTHENTICATION,
            launch_trust_boundary: LOCAL_EPHEMERAL_TRUST_BOUNDARY,
            ferrl_source,
            task: task.identity(),
            model: &model,
            metric_contract: &metric_contract,
            execution: ExecutionDevice::Cpu,
            resolved_eos_token_id,
            training_samples_sha256,
            training_samples_count: 1,
            held_out_samples_sha256,
            held_out_samples_count: 1,
            steps: 1,
            group_size: 2,
            max_new_tokens: 1,
            eval_group_size: 2,
            temperature: 1.0,
            learning_rate: 1e-3,
            seed: 1234,
            candidate_signing_public_key: &signing_key,
        };
        sha256_hex(&serde_json::to_vec(&payload).unwrap())
    }

    fn test_config(root: &Path, steps: u64) -> DiscoveryConfig {
        DiscoveryConfig::builder(root.join("runs"), root.join("artifact"))
            .steps(steps)
            .group_size(2)
            .max_new_tokens(1)
            .eval_group_size(2)
            .build()
            .unwrap()
    }

    fn run_injected(
        mode: VerifyMode,
        config: &DiscoveryConfig,
    ) -> Result<DiscoveryOutcome, DiscoveryError> {
        run_injected_with_tokenizer(TestTask::new(mode), config, &TestTokenizer).0
    }

    fn run_injected_with_task(
        task: TestTask,
        config: &DiscoveryConfig,
    ) -> (
        Result<DiscoveryOutcome, DiscoveryError>,
        TestTask,
        TestPolicy,
    ) {
        run_injected_with_tokenizer(task, config, &TestTokenizer)
    }

    fn run_injected_with_tokenizer<T: TokenizerLike>(
        task: TestTask,
        config: &DiscoveryConfig,
        tokenizer: &T,
    ) -> (
        Result<DiscoveryOutcome, DiscoveryError>,
        TestTask,
        TestPolicy,
    ) {
        let model = test_model();
        let ferrl_source = clean_test_source();
        let metric_contract = task.metric_contract();
        let mut policy = TestPolicy::new();
        let context = LoadedDiscoveryContext {
            task: &task,
            config,
            model: &model,
            ferrl_source: &ferrl_source,
            metric_contract: &metric_contract,
            tokenizer,
            eos_token_id: None,
        };
        let outcome = run_with_loaded_policy(&context, &mut policy);
        (outcome, task, policy)
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn injected_policy_publishes_only_a_verified_material_win() {
        let temp = TestDir::new("win");
        let mut config = test_config(&temp.0, 1);
        config.artifact_output = temp.0.join("new-parent").join("artifact");
        let outcome = run_injected(VerifyMode::Win, &config).unwrap();
        let DiscoveryOutcome::Verified(artifact) = outcome else {
            panic!("expected verified outcome");
        };
        assert!(artifact.manifest_path().is_file());
        assert!(artifact.payload_path().is_file());
        assert!(artifact.candidate_path().is_file());
        assert!(artifact.verification_evidence_path().is_file());
        assert_eq!(artifact.task_identity().name(), "test.discovery");
        assert_eq!(artifact.model_identity().policy_sha256(), "11".repeat(32));
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(artifact.manifest_path()).unwrap()).unwrap();
        assert!(manifest["material_win"].as_bool().unwrap());
        assert!(manifest["held_out_correct"].as_bool().unwrap());
        assert_eq!(
            manifest["artifact_payload_file"],
            serde_json::Value::String(ARTIFACT_PAYLOAD_FILE.into())
        );
        assert_eq!(
            manifest["launch_authentication"],
            LOCAL_EPHEMERAL_AUTHENTICATION
        );
        assert_eq!(
            manifest["launch_trust_boundary"],
            LOCAL_EPHEMERAL_TRUST_BOUNDARY
        );
        assert_eq!(manifest["ferrl_source"]["git_commit"], "ab".repeat(20));
        assert!(!manifest["ferrl_source"]["git_dirty"].as_bool().unwrap());
        let launch_copy = fs::read(artifact.output().join(ARTIFACT_LAUNCH_FILE)).unwrap();
        let held_out_copy = fs::read(artifact.output().join(ARTIFACT_HELD_OUT_FILE)).unwrap();
        let candidate_copy = fs::read(artifact.candidate_path()).unwrap();
        let verifier_copy = fs::read(artifact.verification_evidence_path()).unwrap();
        assert_eq!(manifest["launch_file_sha256"], sha256_hex(&launch_copy));
        assert_eq!(
            manifest["held_out_report_sha256"],
            sha256_hex(&held_out_copy)
        );
        assert_eq!(
            manifest["candidate_file_sha256"],
            sha256_hex(&candidate_copy)
        );
        assert_eq!(
            manifest["verifier_evidence_sha256"],
            sha256_hex(&verifier_copy)
        );
        let run_id = manifest["run_id"].as_str().unwrap();
        assert_eq!(
            launch_copy,
            fs::read(temp.0.join("runs").join(run_id).join(RunDir::LAUNCH_FILE)).unwrap()
        );
        assert_eq!(
            held_out_copy,
            fs::read(
                temp.0
                    .join("runs")
                    .join(run_id)
                    .join(RunDir::EVAL_REPORT_FILE)
            )
            .unwrap()
        );
        let launch: serde_json::Value = serde_json::from_slice(&launch_copy).unwrap();
        assert_eq!(
            launch["launch_authentication"],
            LOCAL_EPHEMERAL_AUTHENTICATION
        );
        assert_eq!(
            launch["payload"]["launch_authentication"],
            LOCAL_EPHEMERAL_AUTHENTICATION
        );
        assert_eq!(
            launch["payload"]["ferrl_source"]["git_commit"],
            "ab".repeat(20)
        );
        assert_eq!(launch["payload"]["metric_contract"]["baseline"], 10.0);
        let held_out: serde_json::Value = serde_json::from_slice(&held_out_copy).unwrap();
        assert_eq!(held_out["contract"], HELD_OUT_REPORT_CONTRACT);
        assert_eq!(held_out["launch_sha256"], manifest["launch_sha256"]);
        let verifier: serde_json::Value = serde_json::from_slice(&verifier_copy).unwrap();
        assert_eq!(verifier["contract"], VERIFIER_EVIDENCE_CONTRACT);
        assert_eq!(verifier["launch_sha256"], manifest["launch_sha256"]);
        assert_eq!(verifier["metric"], manifest["metric"]);
        assert_eq!(
            verifier["candidate_file_sha256"],
            sha256_hex(&candidate_copy)
        );
        let run_candidate = fs::read(
            temp.0
                .join("runs")
                .join(run_id)
                .join(RunDir::CANDIDATES_FILE),
        )
        .unwrap();
        assert!(run_candidate
            .split_inclusive(|byte| *byte == b'\n')
            .any(|row| row == candidate_copy));
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn injected_policy_rejects_later_training_prompt_with_empty_tokenization() {
        let temp = TestDir::new("preflight-training-token");
        let config = test_config(&temp.0, 1);
        let task = TestTask::with_samples(
            VerifyMode::Win,
            vec![Sample::new("train-valid", ()), Sample::new("bad-train", ())],
            vec![Sample::new("held-out", ())],
        );
        let tokenizer = FailingTokenizer {
            fail_prompt: "bad-train",
        };
        let (outcome, task, policy) = run_injected_with_tokenizer(task, &config, &tokenizer);
        let err = outcome.unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, DiscoveryError::InvalidConfiguration(_)));
        assert!(message.contains("ordered training samples"));
        assert!(message.contains("index 1"));
        assert_eq!(task.reward_calls(), 0);
        assert_eq!(policy.generate_calls(), 0);
        assert_eq!(policy.token_logprobs_calls(), 0);
        assert!(!temp.0.join("runs").exists());
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn injected_policy_rejects_later_held_out_prompt_with_empty_tokenization() {
        let temp = TestDir::new("preflight-heldout-token");
        let config = test_config(&temp.0, 1);
        let task = TestTask::with_samples(
            VerifyMode::Win,
            vec![Sample::new("train", ()), Sample::new("train-2", ())],
            vec![Sample::new("held-out", ()), Sample::new("bad-held-out", ())],
        );
        let tokenizer = FailingTokenizer {
            fail_prompt: "bad-held-out",
        };
        let (outcome, task, policy) = run_injected_with_tokenizer(task, &config, &tokenizer);
        let err = outcome.unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, DiscoveryError::InvalidConfiguration(_)));
        assert!(message.contains("ordered held-out samples"));
        assert!(message.contains("index 1"));
        assert_eq!(task.reward_calls(), 0);
        assert_eq!(policy.generate_calls(), 0);
        assert_eq!(policy.token_logprobs_calls(), 0);
        assert!(!temp.0.join("runs").exists());
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn injected_policy_runs_policy_activity_on_normal_prompts() {
        let temp = TestDir::new("preflight-control");
        let mut config = test_config(&temp.0, 1);
        config.artifact_output = temp.0.join("new-parent").join("artifact");
        let (outcome, task, policy) =
            run_injected_with_task(TestTask::new(VerifyMode::Win), &config);
        let outcome = outcome.unwrap();
        let DiscoveryOutcome::Verified(artifact) = outcome else {
            panic!("expected verified outcome");
        };
        assert!(artifact.manifest_path().is_file());
        assert!(artifact.payload_path().is_file());
        assert!(artifact.candidate_path().is_file());
        assert!(artifact.verification_evidence_path().is_file());
        assert!(policy.generate_calls() > 0);
        assert!(policy.token_logprobs_calls() > 0);
        assert!(task.reward_calls() > 0);
        assert!(temp.0.join("runs").exists());
        assert!(config.artifact_output.exists());
    }
    #[test]
    fn injected_policy_returns_completed_no_win_on_threshold_equality() {
        let temp = TestDir::new("no-win");
        let outcome = run_injected(VerifyMode::NoWin, &test_config(&temp.0, 1)).unwrap();
        let DiscoveryOutcome::NoWin(report) = outcome else {
            panic!("expected no-win outcome");
        };
        assert_eq!(report.candidates_checked(), 2);
        assert_eq!(report.reason(), &NoWinReason::NoMaterialMetricWin);
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn strongest_final_metric_wins_over_higher_search_reward() {
        let temp = TestDir::new("final-metric-ranking");
        let outcome = run_injected(
            VerifyMode::BetterFinalMetricBeatsSearchReward,
            &test_config(&temp.0, 1),
        )
        .unwrap();
        let DiscoveryOutcome::Verified(artifact) = outcome else {
            panic!("expected verified outcome");
        };
        assert_eq!(artifact.metric().candidate(), 15.0);
        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(artifact.payload_path()).unwrap()).unwrap();
        assert!(payload["artifact"].as_str().unwrap().starts_with('7'));
        let candidate: CandidateRecord =
            serde_json::from_slice(&fs::read(artifact.candidate_path()).unwrap()).unwrap();
        assert_eq!(candidate.reward, 0.0);
    }

    #[test]
    fn injected_policy_returns_no_win_without_held_out_correctness() {
        let temp = TestDir::new("incorrect");
        let outcome = run_injected(VerifyMode::Incorrect, &test_config(&temp.0, 1)).unwrap();
        let DiscoveryOutcome::NoWin(report) = outcome else {
            panic!("expected no-win outcome");
        };
        assert_eq!(report.reason(), &NoWinReason::HeldOutIncorrect);
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn injected_policy_returns_preempted_before_eval_or_acceptance() {
        let temp = TestDir::new("preempted");
        let flag = Arc::new(AtomicBool::new(true));
        let config = DiscoveryConfig::builder(temp.0.join("runs"), temp.0.join("artifact"))
            .steps(2)
            .group_size(2)
            .max_new_tokens(1)
            .eval_group_size(2)
            .preemption_flag(flag)
            .build()
            .unwrap();
        let outcome = run_injected(VerifyMode::Win, &config).unwrap();
        let DiscoveryOutcome::Preempted(report) = outcome else {
            panic!("expected preempted outcome");
        };
        assert_eq!(report.completed_steps(), 1);
        assert!(report.checkpoint_path().join("manifest.json").is_file());
        let discovered = crate::latest_checkpoint(report.run_dir().join(RunDir::CHECKPOINTS_DIR))
            .unwrap()
            .unwrap();
        assert_eq!(report.checkpoint_path(), discovered.dir);
        assert_eq!(report.completed_steps(), discovered.step);
        assert!(!report.run_dir().join(RunDir::EVAL_REPORT_FILE).exists());
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn injected_policy_propagates_final_verifier_operational_failure() {
        let temp = TestDir::new("verifier-failure");
        let error =
            run_injected(VerifyMode::OperationalFailure, &test_config(&temp.0, 1)).unwrap_err();
        assert!(matches!(error, DiscoveryError::TaskVerification(_)));
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn injected_policy_rejects_nonfinite_final_evidence() {
        let temp = TestDir::new("nonfinite");
        let error = run_injected(VerifyMode::NonFinite, &test_config(&temp.0, 1)).unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidFinalEvidence(_)));
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn injected_policy_rejects_control_bearing_metric_labels_before_publication() {
        let temp = TestDir::new("metric-label");
        let error = run_injected(VerifyMode::MalformedLabel, &test_config(&temp.0, 1)).unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidFinalEvidence(_)));
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn candidate_metric_must_match_launch_frozen_contract() {
        let temp = TestDir::new("metric-contract-mismatch");
        let error = run_injected(
            VerifyMode::MismatchedMetricContract,
            &test_config(&temp.0, 1),
        )
        .unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidFinalEvidence(_)));
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn accepted_publication_is_exclusive_and_failure_is_operational() {
        let temp = TestDir::new("publication-failure");
        fs::create_dir(temp.0.join("artifact")).unwrap();
        let error = run_injected(VerifyMode::Win, &test_config(&temp.0, 1)).unwrap_err();
        assert!(matches!(error, DiscoveryError::Publication { .. }));
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn mid_publication_failure_never_links_manifest_and_claim_blocks_retry() {
        let temp = TestDir::new("publication-mid-fault");
        let output = temp.0.join("artifact");
        let task = TaskIdentity::new("test.discovery", 1).unwrap();
        let signer = CandidateSigner::generate().unwrap();
        let public_key = signer.public_key_hex();
        let launch_sha256 = "cd".repeat(32);
        let record = signer
            .sign_candidate(
                &CandidateRecord::new(0, 0, 1, 0, 0, 1.0, 1, "9".into()),
                &launch_sha256,
            )
            .unwrap();
        let mut exact_row_bytes = serde_json::to_vec(&record).unwrap();
        exact_row_bytes.push(b'\n');
        let provenance_sha256 = record.record_sha256.clone().unwrap();
        let candidate = AuthenticatedCandidate {
            record,
            exact_row_bytes,
            provenance_sha256,
        };
        let evidence = || {
            FinalEvidence::new(
                "artifact".to_owned(),
                "typed verifier evidence".to_owned(),
                true,
                MetricReport::new(
                    "throughput",
                    "items/s",
                    MetricDirection::HigherIsBetter,
                    10.0,
                    12.0,
                    1.0,
                ),
            )
        };
        let error = publish_verified_artifact_with_fault(
            &output,
            &task,
            &test_model(),
            &clean_test_source(),
            "run-id",
            &launch_sha256,
            &public_key,
            &candidate,
            b"launch\n",
            b"held-out\n",
            evidence(),
            Some(2),
        )
        .unwrap_err();
        assert!(matches!(error, DiscoveryError::Publication { .. }));
        assert!(output.is_dir());
        assert!(!output.join(ARTIFACT_MANIFEST_FILE).exists());

        let retry = publish_verified_artifact_with_fault(
            &output,
            &task,
            &test_model(),
            &clean_test_source(),
            "run-id",
            &launch_sha256,
            &public_key,
            &candidate,
            b"launch\n",
            b"held-out\n",
            evidence(),
            None,
        )
        .unwrap_err();
        assert!(matches!(retry, DiscoveryError::Publication { .. }));
        assert!(!output.join(ARTIFACT_MANIFEST_FILE).exists());
    }

    #[test]
    fn candidate_ledger_rejects_tampering_and_missing_coverage() {
        let temp = TestDir::new("candidate-negative");
        let signer = CandidateSigner::generate().unwrap();
        let public_key = signer.public_key_hex();
        let launch = "ab".repeat(32);
        let record = signer
            .sign_candidate(
                &CandidateRecord::new(0, 0, 1, 0, 0, 1.0, 1, "9".into()),
                &launch,
            )
            .unwrap();
        let path = temp.0.join("candidates.jsonl");
        let mut value = serde_json::to_value(record).unwrap();
        value["completion"] = serde_json::Value::String("tampered".into());
        fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&value).unwrap()),
        )
        .unwrap();
        let error = load_and_validate_candidates(&path, &launch, &public_key, 1, 1, 1).unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidCandidateEvidence(_)));

        fs::write(&path, b"").unwrap();
        let error = load_and_validate_candidates(&path, &launch, &public_key, 1, 1, 1).unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidCandidateEvidence(_)));
    }

    #[test]
    fn reencoded_or_unknown_signed_rows_fail_before_verification_or_publication() {
        let temp = TestDir::new("candidate-canonical");
        let signer = CandidateSigner::generate().unwrap();
        let public_key = signer.public_key_hex();
        let launch = "cd".repeat(32);
        let record = signer
            .sign_candidate(
                &CandidateRecord::new(0, 0, 1, 0, 0, 1.0, 1, "9".into()),
                &launch,
            )
            .unwrap();
        let path = temp.0.join("candidates.jsonl");

        let mut reencoded = serde_json::to_vec(&record).unwrap();
        reencoded.insert(0, b' ');
        reencoded.push(b'\n');
        fs::write(&path, reencoded).unwrap();
        let error = load_and_validate_candidates(&path, &launch, &public_key, 1, 1, 1).unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidCandidateEvidence(_)));
        assert!(!temp.0.join("artifact").exists());

        let mut with_unknown = serde_json::to_value(&record).unwrap();
        with_unknown["signed_but_unknown"] = serde_json::Value::Bool(true);
        let mut unknown_bytes = serde_json::to_vec(&with_unknown).unwrap();
        unknown_bytes.push(b'\n');
        fs::write(&path, unknown_bytes).unwrap();
        let error = load_and_validate_candidates(&path, &launch, &public_key, 1, 1, 1).unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidCandidateEvidence(_)));
        assert!(!temp.0.join("artifact").exists());
    }

    #[test]
    fn canonical_signed_candidate_cannot_exceed_launch_token_width() {
        let temp = TestDir::new("candidate-token-width");
        let signer = CandidateSigner::generate().unwrap();
        let public_key = signer.public_key_hex();
        let launch = "de".repeat(32);
        let record = signer
            .sign_candidate(
                &CandidateRecord::new(0, 0, 1, 0, 0, 1.0, 2, "9,9".into()),
                &launch,
            )
            .unwrap();
        let path = temp.0.join("candidates.jsonl");
        let mut canonical = serde_json::to_vec(&record).unwrap();
        canonical.push(b'\n');
        fs::write(&path, canonical).unwrap();

        let accepted_at_bound =
            load_and_validate_candidates(&path, &launch, &public_key, 1, 1, 2).unwrap();
        assert_eq!(accepted_at_bound.len(), 1);

        let error = load_and_validate_candidates(&path, &launch, &public_key, 1, 1, 1).unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidCandidateEvidence(_)));
        assert!(error.to_string().contains("launch-bound range 1..=1"));
    }

    #[test]
    fn sdk_config_forces_complete_candidate_logging() {
        let temp = TestDir::new("candidate-config");
        let config = test_config(&temp.0, 3);
        let trainer = config.trainer_config(None);
        assert_eq!(trainer.candidate_log_top_k, trainer.group_size);
        assert_eq!(trainer.checkpoint_every, Some(trainer.steps));
    }

    #[test]
    fn artifact_output_must_not_equal_or_contain_runs_root_after_lexical_normalization() {
        let temp = TestDir::new("path-collision");
        let equal = DiscoveryConfig::builder(temp.0.join("same"), temp.0.join("same")).build();
        assert!(matches!(
            equal,
            Err(DiscoveryError::InvalidConfiguration(_))
        ));

        let ancestor = DiscoveryConfig::builder(
            temp.0.join("claim").join("runs"),
            temp.0.join("nested").join("..").join("claim"),
        )
        .build();
        assert!(matches!(
            ancestor,
            Err(DiscoveryError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn opened_device_identity_is_derived_and_selection_mismatch_fails() {
        assert_eq!(
            execution_device_from_opened(&Device::Cpu, ExecutionDevice::Cpu).unwrap(),
            ExecutionDevice::Cpu
        );
        let error =
            execution_device_from_opened(&Device::Cpu, ExecutionDevice::Cuda { ordinal: 0 })
                .unwrap_err();
        assert!(matches!(error, DiscoveryError::InvalidConfiguration(_)));
    }

    #[test]
    fn source_identity_rejects_unknown_uppercase_and_dirty_builds() {
        for git_commit in ["unknown".to_owned(), "AB".repeat(20)] {
            let source = BuildSourceIdentity {
                package_version: env!("CARGO_PKG_VERSION").into(),
                git_commit,
                git_dirty: false,
            };
            assert!(matches!(
                source.validate(),
                Err(DiscoveryError::InvalidConfiguration(_))
            ));
        }
        let mut dirty = clean_test_source();
        dirty.git_dirty = true;
        assert!(matches!(
            dirty.validate(),
            Err(DiscoveryError::InvalidConfiguration(_))
        ));
        clean_test_source().validate().unwrap();
        let mut sha256_source = clean_test_source();
        sha256_source.git_commit = "cd".repeat(32);
        sha256_source.validate().unwrap();
    }

    #[test]
    fn exact_serialized_sample_bytes_are_reconstructed_for_execution() {
        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct TargetWithSkippedField {
            visible: u32,
            #[serde(skip, default)]
            omitted_from_contract: String,
        }

        let samples = vec![Sample::new(
            "prompt",
            TargetWithSkippedField {
                visible: 7,
                omitted_from_contract: "must-not-reach-reward".into(),
            },
        )];
        let (execution_samples, exact_bytes) =
            exact_execution_samples(&samples, "test samples").unwrap();
        assert_eq!(execution_samples[0].target.visible, 7);
        assert!(execution_samples[0].target.omitted_from_contract.is_empty());
        assert!(!String::from_utf8(exact_bytes)
            .unwrap()
            .contains("must-not-reach-reward"));
    }

    #[test]
    fn exact_ordered_sample_changes_alter_the_launch_identity() {
        let task = TestTask::new(VerifyMode::Win);
        let changed_train = vec![Sample::new("different train", ())];
        let changed_held_out = vec![Sample::new("different held-out", ())];
        let held_out_bytes = serde_json::to_vec(task.held_out_samples()).unwrap();
        let changed_held_out_bytes = serde_json::to_vec(&changed_held_out).unwrap();
        let first_train_bytes = serde_json::to_vec(task.training_samples()).unwrap();
        let changed_train_bytes = serde_json::to_vec(&changed_train).unwrap();
        let first_train_sha = sha256_hex(&first_train_bytes);
        let changed_train_sha = sha256_hex(&changed_train_bytes);
        let held_out_sha = sha256_hex(&held_out_bytes);
        let changed_held_out_sha = sha256_hex(&changed_held_out_bytes);
        let first = test_launch_sha(&task, &first_train_sha, &held_out_sha, Some(2));
        let changed_train = test_launch_sha(&task, &changed_train_sha, &held_out_sha, Some(2));
        let changed_held_out =
            test_launch_sha(&task, &first_train_sha, &changed_held_out_sha, Some(2));
        let changed_eos = test_launch_sha(&task, &first_train_sha, &held_out_sha, None);
        let mut changed_source = clean_test_source();
        changed_source.git_commit = "ef".repeat(20);
        let changed_source = test_launch_sha_with_source(
            &task,
            &first_train_sha,
            &held_out_sha,
            Some(2),
            &changed_source,
        );
        assert_ne!(first_train_sha, changed_train_sha);
        assert_ne!(held_out_sha, changed_held_out_sha);
        assert_ne!(first, changed_train);
        assert_ne!(first, changed_held_out);
        assert_ne!(first, changed_eos);
        assert_ne!(first, changed_source);
    }
}
