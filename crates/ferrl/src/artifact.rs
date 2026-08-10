//! Library-owned accepted-artifact audit, decision, and publication boundary.
//!
//! The stable discovery facade returns [`crate::discovery::VerifiedArtifact`].
//! This module also owns the production TriMul audit bundle used by the CLI so
//! the binary remains an input/output adapter rather than a second trusted
//! artifact implementation.

use crate::orchestration::{
    LaunchAuthenticationMode, LaunchManifest, LaunchVerifierIdentity, LAUNCH_ATTESTATION_ALGORITHM,
};
use crate::telemetry::CandidateRecord;
use crate::trimul::{
    timing_metric_for_tier, validate_artifact_verification_evidence,
    verifier_isolation_evidence_sha256, EvidencedTrimulVerification,
    TrimulArtifactVerificationEvidence, TrimulExecutingDevice, TrimulReward, TrimulRewardProfile,
    TrimulRuntimePreflightEvidence, TrimulVerification, TrimulVerifierIdentity,
    DEFAULT_VERIFIER_MAX_PROCS, TRIMUL_RUNTIME_HARDENING_CONTRACT,
};
use crate::{
    RunStatus, VerifierAssetTransport, VerifierIsolationEvidence, VerifierIsolationTier,
    VerifierUidBoundary, VERIFIER_ISOLATION_EVIDENCE_VERSION,
};
use ring::rand::{SecureRandom as _, SystemRandom};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};

const ARTIFACT_CONTRACT_VERSION: u32 = 4;
const ARTIFACT_AUDIT_BLOCKS: usize = 11;
const ARTIFACT_MATERIAL_SPEEDUP: f64 = 1.02;
const ARTIFACT_REQUIRED_MATERIAL_WINS: usize = 9;
const ARTIFACT_ACCEPTANCE_METHOD: &str = "paired_material_wins_v1";
const ARTIFACT_AUDIT_SEED_DERIVATION: &str = "sha256_contract_prefix_u32_be_v1";
const ARTIFACT_ATTEMPT_SELECTION_ASSURANCE: &str = "operator_attested_v1";
const ARTIFACT_OWNER_FILE: &str = ".ferrl-artifact-owner";
const ARTIFACT_ATTEMPT_FILE: &str = "audit-attempt.json";
const TRIMUL_CASE_SEED_MAX: u64 = u32::MAX as u64;

/// Failure while validating, auditing, or publishing an accepted artifact.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ArtifactError {
    /// Contract or evidence validation failed.
    #[error("{0}")]
    Message(String),
    /// An artifact file operation failed.
    #[error("artifact I/O failed for {path}: {source}")]
    Io {
        /// Affected path.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// Artifact JSON serialization failed.
    #[error("serialize {path}: {source}")]
    Serialization {
        /// Logical output path.
        path: PathBuf,
        /// Underlying JSON failure.
        #[source]
        source: serde_json::Error,
    },
}

impl ArtifactError {
    fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

/// Operator source-inspection decision included in the TriMul artifact gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceInspection {
    /// No prohibited process, file-descriptor, environment, network, or path access was found.
    Clean,
    /// Suspicious process, file-descriptor, environment, network, or path access was found.
    Suspicious,
}

/// Resolved configuration fields preserved by the TriMul v4 artifact manifest.
#[derive(Debug, Clone)]
pub struct TrimulArtifactConfig {
    /// `LoRA` rank used during discovery.
    pub lora_rank: usize,
    /// `LoRA` alpha used during discovery.
    pub lora_alpha: f64,
    /// Stable frozen-base dtype spelling.
    pub base_dtype: &'static str,
    /// Stable frozen-base quantization spelling.
    pub base_quantization: &'static str,
    /// Effective training reward profile.
    pub reward_profile: TrimulRewardProfile,
    /// Trainer step budget.
    pub trainer_steps: u64,
    /// GRPO group size.
    pub group_size: usize,
    /// Operator run-health note.
    pub run_health: String,
    /// Policy rollout seed.
    pub policy_seed: u64,
    /// Data seed.
    pub data_seed: u64,
    /// Secret seed used during discovery.
    pub training_secret_seed: u64,
    /// Effective scratch cap.
    pub scratch_max_bytes: u64,
    /// Effective verifier parallelism.
    pub verifier_parallelism: usize,
    /// Effective verifier process cap.
    pub verifier_max_procs: u64,
    /// Training verifier CUDA visibility pool.
    pub verifier_cuda_device_pool: Vec<String>,
}

/// Opaque deterministic identity for one launch-bound TriMul artifact audit.
#[derive(Debug, Clone)]
pub struct TrimulArtifactAuditIdentity {
    contract_sha256: String,
    audit_id: String,
    secret_seed: u64,
}

impl TrimulArtifactAuditIdentity {
    /// Deterministically derived case-generation seed, distinct from training.
    #[must_use]
    pub fn secret_seed(&self) -> u64 {
        self.secret_seed
    }
}

/// Derive the immutable audit identity before constructing the audit reward.
#[must_use]
pub fn trimul_artifact_audit_identity(
    launch: &LaunchManifest,
    candidate: &CandidateRecord,
    submission: &str,
    verifier_assets: &TrimulVerifierIdentity,
    training_secret_seed: u64,
) -> TrimulArtifactAuditIdentity {
    let contract_sha256 =
        artifact_audit_contract_sha256(launch, candidate, submission, verifier_assets);
    let secret_seed = artifact_audit_secret_seed(&contract_sha256, training_secret_seed);
    let audit_id = artifact_audit_id(&contract_sha256, secret_seed);
    TrimulArtifactAuditIdentity {
        contract_sha256,
        audit_id,
        secret_seed,
    }
}

/// Opaque validated request for one production TriMul artifact audit.
///
/// Callers can obtain this value only through [`Self::bind`], which proves that
/// launch bytes, candidate bytes, completion, source, prompt, verifier assets,
/// reward controls, and manifest configuration all describe one immutable run.
#[doc(hidden)]
pub struct TrimulArtifactRequest<'a> {
    output: &'a Path,
    launch: &'a LaunchManifest,
    launch_bytes: &'a [u8],
    candidate: &'a CandidateRecord,
    candidate_row_bytes: &'a [u8],
    raw_completion: &'a str,
    prompt_bytes: &'a [u8],
    submission: &'a str,
    audit_identity: &'a TrimulArtifactAuditIdentity,
    reward: &'a TrimulReward,
    audit_verifier: &'a LaunchVerifierIdentity,
    test_cases: usize,
    benchmark_cases: usize,
    audit_cuda_visible_device: &'a str,
    source_inspection: SourceInspection,
    source_inspection_notes: &'a str,
    config: TrimulArtifactConfig,
}

impl<'a> TrimulArtifactRequest<'a> {
    /// Validate and bind every decomposed CLI input before artifact measurement.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] when any byte encoding, provenance link,
    /// verifier/reward identity, or resolved configuration field disagrees.
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        output: &'a Path,
        launch: &'a LaunchManifest,
        launch_bytes: &'a [u8],
        candidate: &'a CandidateRecord,
        candidate_row_bytes: &'a [u8],
        raw_completion: &'a str,
        prompt_bytes: &'a [u8],
        submission: &'a str,
        audit_identity: &'a TrimulArtifactAuditIdentity,
        reward: &'a TrimulReward,
        audit_verifier: &'a LaunchVerifierIdentity,
        test_cases: usize,
        benchmark_cases: usize,
        audit_cuda_visible_device: &'a str,
        source_inspection: SourceInspection,
        source_inspection_notes: &'a str,
        config: TrimulArtifactConfig,
    ) -> Result<Self, ArtifactError> {
        validate_trimul_artifact_binding(
            launch,
            launch_bytes,
            candidate,
            candidate_row_bytes,
            raw_completion,
            prompt_bytes,
            submission,
            audit_identity,
            reward,
            audit_verifier,
            test_cases,
            benchmark_cases,
            &config,
        )?;
        Ok(Self {
            output,
            launch,
            launch_bytes,
            candidate,
            candidate_row_bytes,
            raw_completion,
            prompt_bytes,
            submission,
            audit_identity,
            reward,
            audit_verifier,
            test_cases,
            benchmark_cases,
            audit_cuda_visible_device,
            source_inspection,
            source_inspection_notes,
            config,
        })
    }
}

struct TrimulArtifactRequestView<'a> {
    output: &'a Path,
    launch: &'a LaunchManifest,
    launch_bytes: &'a [u8],
    candidate: &'a CandidateRecord,
    candidate_row_bytes: &'a [u8],
    raw_completion: &'a str,
    prompt_bytes: &'a [u8],
    submission: &'a str,
    audit_identity: &'a TrimulArtifactAuditIdentity,
    audit_verifier: &'a LaunchVerifierIdentity,
    test_cases: usize,
    benchmark_cases: usize,
    audit_cuda_visible_device: &'a str,
    source_inspection: SourceInspection,
    source_inspection_notes: &'a str,
    config: &'a TrimulArtifactConfig,
}

impl<'a> From<&'a TrimulArtifactRequest<'a>> for TrimulArtifactRequestView<'a> {
    fn from(request: &'a TrimulArtifactRequest<'a>) -> Self {
        Self {
            output: request.output,
            launch: request.launch,
            launch_bytes: request.launch_bytes,
            candidate: request.candidate,
            candidate_row_bytes: request.candidate_row_bytes,
            raw_completion: request.raw_completion,
            prompt_bytes: request.prompt_bytes,
            submission: request.submission,
            audit_identity: request.audit_identity,
            audit_verifier: request.audit_verifier,
            test_cases: request.test_cases,
            benchmark_cases: request.benchmark_cases,
            audit_cuda_visible_device: request.audit_cuda_visible_device,
            source_inspection: request.source_inspection,
            source_inspection_notes: request.source_inspection_notes,
            config: &request.config,
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
fn validate_trimul_artifact_binding(
    launch: &LaunchManifest,
    launch_bytes: &[u8],
    candidate: &CandidateRecord,
    candidate_row_bytes: &[u8],
    raw_completion: &str,
    prompt_bytes: &[u8],
    submission: &str,
    audit_identity: &TrimulArtifactAuditIdentity,
    reward: &TrimulReward,
    audit_verifier: &LaunchVerifierIdentity,
    test_cases: usize,
    benchmark_cases: usize,
    config: &TrimulArtifactConfig,
) -> Result<(), ArtifactError> {
    let canonical_launch = launch
        .to_pretty_bytes()
        .map_err(|error| ArtifactError::msg(error.to_string()))?;
    let reconstructed = LaunchManifest::new(launch.payload.clone())
        .map_err(|error| ArtifactError::msg(error.to_string()))?;
    if canonical_launch != launch_bytes
        || reconstructed.contract_version != launch.contract_version
        || reconstructed.kind != launch.kind
        || reconstructed.payload_sha256 != launch.payload_sha256
    {
        return Err(ArtifactError::msg(
            "TriMul artifact launch bytes or payload identity are not canonical",
        ));
    }
    candidate
        .verify_signed_provenance(&launch.payload.candidate_ledger.signing_public_key)
        .map_err(|error| ArtifactError::msg(error.to_string()))?;
    let canonical_candidate =
        serde_json::to_vec(candidate).map_err(|source| ArtifactError::Serialization {
            path: PathBuf::from("candidate.json"),
            source,
        })?;
    if canonical_candidate != candidate_row_bytes
        || candidate.launch_sha256.as_deref() != Some(&launch.payload_sha256)
        || candidate.completion != raw_completion
    {
        return Err(ArtifactError::msg(
            "TriMul artifact candidate bytes, launch, or completion disagree",
        ));
    }
    let source_sha256 = candidate
        .reward_metadata
        .as_ref()
        .and_then(|metadata| metadata.get("source_sha256"))
        .and_then(serde_json::Value::as_str);
    let expected_source_sha256 = sha256_hex(submission.as_bytes());
    if source_sha256 != Some(expected_source_sha256.as_str()) {
        return Err(ArtifactError::msg(
            "TriMul artifact submission does not match candidate reward provenance",
        ));
    }
    let prompt = launch
        .payload
        .prompt
        .as_ref()
        .ok_or_else(|| ArtifactError::msg("verified TriMul launch has no prompt identity"))?;
    if prompt.file != "prompt.txt"
        || prompt.len_bytes != prompt_bytes.len()
        || prompt.sha256 != sha256_hex(prompt_bytes)
    {
        return Err(ArtifactError::msg(
            "TriMul artifact prompt bytes do not match the immutable launch",
        ));
    }
    let discovery_verifier = launch
        .payload
        .verifier
        .as_ref()
        .ok_or_else(|| ArtifactError::msg("verified TriMul launch has no verifier identity"))?;
    let (reward_assets, reward_seed, reward_tests, reward_benchmarks) = reward.artifact_binding();
    if &discovery_verifier.assets != reward_assets
        || &audit_verifier.assets != reward_assets
        || reward_seed != audit_identity.secret_seed
        || reward_tests != test_cases
        || reward_benchmarks != benchmark_cases
        || test_cases == 0
        || benchmark_cases == 0
        || reward.reward_profile() != config.reward_profile
    {
        return Err(ArtifactError::msg(
            "TriMul artifact reward, cases, verifier assets, or audit seed disagree",
        ));
    }
    let expected_identity = trimul_artifact_audit_identity(
        launch,
        candidate,
        submission,
        reward_assets,
        config.training_secret_seed,
    );
    if audit_identity.contract_sha256 != expected_identity.contract_sha256
        || audit_identity.audit_id != expected_identity.audit_id
        || audit_identity.secret_seed != expected_identity.secret_seed
    {
        return Err(ArtifactError::msg(
            "TriMul artifact audit identity does not match the immutable request",
        ));
    }
    validate_resolved_artifact_config(&launch.payload.config.resolved, config)
}

fn validate_resolved_artifact_config(
    resolved: &serde_json::Value,
    config: &TrimulArtifactConfig,
) -> Result<(), ArtifactError> {
    let trimul = resolved
        .get("trimul")
        .ok_or_else(|| ArtifactError::msg("resolved launch config has no trimul block"))?;
    let raw_scratch = trimul
        .get("scratch_max_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let raw_parallelism = trimul
        .get("verifier_parallelism")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let raw_max_procs = trimul
        .get("verifier_max_procs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let expected = [
        ("/policy/lora_rank", serde_json::json!(config.lora_rank)),
        ("/policy/lora_alpha", serde_json::json!(config.lora_alpha)),
        ("/policy/base_dtype", serde_json::json!(config.base_dtype)),
        (
            "/policy/base_quantization",
            serde_json::json!(config.base_quantization),
        ),
        ("/policy/seed", serde_json::json!(config.policy_seed)),
        ("/data/seed", serde_json::json!(config.data_seed)),
        ("/trainer/steps", serde_json::json!(config.trainer_steps)),
        ("/trainer/group_size", serde_json::json!(config.group_size)),
        (
            "/trimul/secret_seed",
            serde_json::json!(config.training_secret_seed),
        ),
        (
            "/trimul/reward",
            serde_json::to_value(config.reward_profile).expect("reward profile is serializable"),
        ),
        (
            "/trimul/verifier_cuda_device_pool",
            serde_json::json!(&config.verifier_cuda_device_pool),
        ),
    ];
    if expected
        .iter()
        .any(|(pointer, value)| resolved.pointer(pointer) != Some(value))
        || config.scratch_max_bytes
            != if raw_scratch == 0 {
                1 << 30
            } else {
                raw_scratch
            }
        || config.verifier_parallelism
            != usize::try_from(raw_parallelism)
                .unwrap_or(usize::MAX)
                .max(1)
        || config.verifier_max_procs
            != if raw_max_procs == 0 {
                DEFAULT_VERIFIER_MAX_PROCS
            } else {
                raw_max_procs
            }
    {
        return Err(ArtifactError::msg(
            "TriMul artifact manifest configuration differs from resolved launch config",
        ));
    }
    Ok(())
}

/// Library-issued result of a completed TriMul artifact publication.
#[derive(Debug, Clone)]
pub struct PublishedArtifact {
    output: PathBuf,
    manifest_path: PathBuf,
    accepted: bool,
}

impl PublishedArtifact {
    /// Published bundle root.
    #[must_use]
    pub fn output(&self) -> &Path {
        &self.output
    }

    /// Manifest-last commit marker.
    #[must_use]
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Whether the audit and source-inspection gates accepted the artifact.
    #[must_use]
    pub fn accepted(&self) -> bool {
        self.accepted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ArtifactAuditRole {
    Reference,
    Candidate,
}

impl ArtifactAuditRole {
    const fn other(self) -> Self {
        match self {
            Self::Reference => Self::Candidate,
            Self::Candidate => Self::Reference,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactAuditExecution {
    role: ArtifactAuditRole,
    isolation_tier: VerifierIsolationTier,
    isolation: VerifierIsolationEvidence,
    isolation_evidence_sha256: String,
    runtime_hardening_evidence_sha256: String,
    runtime_hardening: Vec<serde_json::Value>,
    timing_metric: String,
    verification: TrimulVerification,
    exact: TrimulArtifactVerificationEvidence,
    protected_output: String,
    sandbox_diagnostics: String,
}

#[derive(Debug)]
struct ArtifactAuditBlock {
    index: usize,
    first: ArtifactAuditRole,
    reference: ArtifactAuditExecution,
    candidate: ArtifactAuditExecution,
    paired_speedup: f64,
    material_win: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactAuditExecutionManifest {
    role: ArtifactAuditRole,
    evidence_file: String,
    evidence_sha256: String,
    isolation_tier: VerifierIsolationTier,
    isolation: VerifierIsolationEvidence,
    isolation_evidence_sha256: String,
    runtime_hardening_evidence_sha256: String,
    runtime_hardening: Vec<serde_json::Value>,
    timing_metric: String,
    verification: TrimulVerification,
    exact: TrimulArtifactVerificationEvidence,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactAuditBlockManifest {
    index: usize,
    first: ArtifactAuditRole,
    reference: ArtifactAuditExecutionManifest,
    candidate: ArtifactAuditExecutionManifest,
    paired_speedup: f64,
    material_win: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactAcceptanceDecision {
    method: &'static str,
    paired_blocks: usize,
    material_speedup: f64,
    threshold_comparison: &'static str,
    required_material_wins: usize,
    observed_material_wins: usize,
    accepted: bool,
}

#[derive(Debug, Serialize)]
struct DiscoveryVerifierManifest {
    isolation_tier: VerifierIsolationTier,
    isolation_evidence_sha256: String,
    timing_metric: String,
    runtime_preflight_evidence_sha256: String,
}

#[derive(Debug, Serialize)]
struct ArtifactAuditManifest {
    contract: &'static str,
    audit_contract_sha256: String,
    audit_id: String,
    audit_secret_seed: u64,
    audit_seed_derivation: &'static str,
    attempt_selection_assurance: &'static str,
    durable_once_only: bool,
    artifact_wide_false_positive_guarantee: bool,
    requested_cuda_visible_device: String,
    isolation_tier: VerifierIsolationTier,
    isolation: VerifierIsolationEvidence,
    isolation_evidence_sha256: String,
    runtime_preflight: TrimulRuntimePreflightEvidence,
    runtime_preflight_evidence_sha256: String,
    timing_metric: String,
    executing_device: TrimulExecutingDevice,
    blocks: Vec<ArtifactAuditBlockManifest>,
    decision: ArtifactAcceptanceDecision,
}

#[derive(Debug, Clone, Serialize)]
struct SourceInspectionManifest {
    result: SourceInspection,
    notes: String,
}

#[derive(Debug, Serialize)]
struct ArtifactManifest {
    contract_version: u32,
    task: &'static str,
    ferrl_commit: String,
    run_id: String,
    launch_sha256: String,
    launch_file_sha256: String,
    launch_authentication: LaunchAuthenticationMode,
    launch_attestation_key_id: Option<String>,
    launch_attestation_algorithm: Option<String>,
    discovery_verifier: DiscoveryVerifierManifest,
    candidate: CandidateManifest,
    model: ModelManifest,
    config: ArtifactConfigManifest,
    eval: EvalManifest,
    audit: ArtifactAuditManifest,
    accepted: bool,
}

#[derive(Debug, Serialize)]
struct CandidateManifest {
    record_sha256: String,
    record_signature: String,
    ledger_row_sha256: String,
    step: u64,
    prompt_index: u64,
    group_index: usize,
    rank: usize,
    world_size: usize,
    training_reward: f32,
    completion_sha256: String,
    source_sha256: String,
    source_inspection: SourceInspectionManifest,
}

#[derive(Debug, Serialize)]
struct ModelManifest {
    family: String,
    checkpoint_policy_sha256: String,
    tokenizer_sha256: String,
    lora_rank: usize,
    lora_alpha: f64,
    base_dtype: &'static str,
    base_quantization: &'static str,
}

#[derive(Debug, Serialize)]
struct ArtifactConfigManifest {
    run_config_source_sha256: String,
    run_config_resolved_sha256: String,
    prompt_sha256: String,
    prompt_file: &'static str,
    reward_profile: TrimulRewardProfile,
    trainer_steps: u64,
    group_size: usize,
    run_health: String,
    policy_seed: u64,
    data_seed: u64,
    training_secret_seed: u64,
    audit_secret_seed: u64,
    scratch_max_bytes: u64,
    verifier_parallelism: usize,
    verifier_max_procs: u64,
    verifier_cuda_device_pool: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EvalManifest {
    bundle_sha256: String,
    bundle_file_count: usize,
    sandbox_image_sha256: String,
    sandbox_image_len_bytes: u64,
    task_yml_sha256: String,
    task_yml_len_bytes: usize,
    test_cases: usize,
    benchmark_cases: usize,
}

#[derive(Debug, Serialize)]
struct ArtifactAttemptRecord<'a> {
    contract_version: u32,
    audit_contract_sha256: &'a str,
    attempt_selection_assurance: &'static str,
    durable_once_only: bool,
    state: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArtifactDirectoryIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl ArtifactDirectoryIdentity {
    fn capture(path: &Path) -> Result<Self, ArtifactError> {
        let metadata = std::fs::symlink_metadata(path).map_err(|source| ArtifactError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(ArtifactError::msg(format!(
                "artifact publication path {} is not a non-symlink directory",
                path.display()
            )));
        }
        Ok(Self {
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        })
    }
}

#[derive(Debug)]
struct ArtifactPublication {
    final_dir: PathBuf,
    stage_dir: PathBuf,
    parent_dir: PathBuf,
    owner: String,
    final_identity: ArtifactDirectoryIdentity,
    stage_identity: ArtifactDirectoryIdentity,
    staged_files: BTreeSet<PathBuf>,
}

impl ArtifactPublication {
    fn claim(final_dir: &Path, audit_contract_sha256: &str) -> Result<Self, ArtifactError> {
        let file_name = final_dir.file_name().ok_or_else(|| {
            ArtifactError::msg(
                "artifact output must name a new directory beneath an existing parent",
            )
        })?;
        if file_name.is_empty() {
            return Err(ArtifactError::msg(
                "artifact output directory name must not be empty",
            ));
        }
        let parent_dir = final_dir
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        std::fs::create_dir_all(&parent_dir).map_err(|source| ArtifactError::Io {
            path: parent_dir.clone(),
            source,
        })?;
        sync_directory(&parent_dir)?;
        std::fs::create_dir(final_dir).map_err(|source| {
            if source.kind() == std::io::ErrorKind::AlreadyExists {
                ArtifactError::msg(format!(
                    "{} already exists; artifact output claims are exclusive",
                    final_dir.display()
                ))
            } else {
                ArtifactError::Io {
                    path: final_dir.to_path_buf(),
                    source,
                }
            }
        })?;
        let final_identity = ArtifactDirectoryIdentity::capture(final_dir)?;
        let mut owner_bytes = [0_u8; 32];
        SystemRandom::new().fill(&mut owner_bytes).map_err(|_| {
            ArtifactError::msg(
                "operating-system randomness failed after artifact output was claimed",
            )
        })?;
        let mut owner = String::with_capacity(64);
        for byte in owner_bytes {
            write!(&mut owner, "{byte:02x}").expect("writing hexadecimal cannot fail");
        }
        write_new_synced(&final_dir.join(ARTIFACT_OWNER_FILE), owner.as_bytes())?;
        let stage_dir = parent_dir.join(format!(".ferrl-artifact-{owner}.stage"));
        std::fs::create_dir(&stage_dir).map_err(|source| ArtifactError::Io {
            path: stage_dir.clone(),
            source,
        })?;
        let stage_identity = ArtifactDirectoryIdentity::capture(&stage_dir)?;
        write_new_synced(&stage_dir.join(ARTIFACT_OWNER_FILE), owner.as_bytes())?;
        let verification = stage_dir.join("verification");
        std::fs::create_dir(&verification).map_err(|source| ArtifactError::Io {
            path: verification.clone(),
            source,
        })?;
        let attempt_json = json_pretty(
            &final_dir.join(ARTIFACT_ATTEMPT_FILE),
            &ArtifactAttemptRecord {
                contract_version: ARTIFACT_CONTRACT_VERSION,
                audit_contract_sha256,
                attempt_selection_assurance: ARTIFACT_ATTEMPT_SELECTION_ASSURANCE,
                durable_once_only: false,
                state: "output_claimed_before_measurement",
            },
        )?;
        write_new_synced(
            &final_dir.join(ARTIFACT_ATTEMPT_FILE),
            attempt_json.as_bytes(),
        )?;
        write_new_synced(
            &stage_dir.join(ARTIFACT_ATTEMPT_FILE),
            attempt_json.as_bytes(),
        )?;
        sync_directory(&verification)?;
        sync_directory(&stage_dir)?;
        sync_directory(final_dir)?;
        sync_directory(&parent_dir)?;
        Ok(Self {
            final_dir: final_dir.to_path_buf(),
            stage_dir,
            parent_dir,
            owner,
            final_identity,
            stage_identity,
            staged_files: BTreeSet::new(),
        })
    }

    fn require_owner(&self) -> Result<(), ArtifactError> {
        if ArtifactDirectoryIdentity::capture(&self.final_dir)? != self.final_identity
            || ArtifactDirectoryIdentity::capture(&self.stage_dir)? != self.stage_identity
            || read_bytes(&self.final_dir.join(ARTIFACT_OWNER_FILE))? != self.owner.as_bytes()
            || read_bytes(&self.stage_dir.join(ARTIFACT_OWNER_FILE))? != self.owner.as_bytes()
        {
            return Err(ArtifactError::msg(
                "artifact publication ownership changed after its exclusive claim",
            ));
        }
        Ok(())
    }

    fn stage_bytes(&mut self, relative: &Path, bytes: &[u8]) -> Result<(), ArtifactError> {
        self.require_owner()?;
        if relative.as_os_str().is_empty()
            || relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
            || relative.parent().is_some_and(|parent| {
                !parent.as_os_str().is_empty() && parent != Path::new("verification")
            })
        {
            return Err(ArtifactError::msg(format!(
                "invalid artifact staging path {}",
                relative.display()
            )));
        }
        if !self.staged_files.insert(relative.to_path_buf()) {
            return Err(ArtifactError::msg(format!(
                "artifact staging attempted to replace {}",
                relative.display()
            )));
        }
        write_new_synced(&self.stage_dir.join(relative), bytes)
    }

    fn stage_text(&mut self, relative: &Path, text: &str) -> Result<(), ArtifactError> {
        self.stage_bytes(relative, text.as_bytes())
    }

    fn publish_manifest_last(&self) -> Result<(), ArtifactError> {
        self.require_owner()?;
        let manifest = Path::new("manifest.json");
        if !self.staged_files.contains(manifest) {
            return Err(ArtifactError::msg(
                "artifact publication has no staged manifest commit marker",
            ));
        }
        link_staged_manifest_last(
            &self.final_dir,
            &self.stage_dir,
            &self.parent_dir,
            &self.staged_files,
            manifest,
            None,
            || self.require_owner(),
        )
    }
}

pub(crate) fn publish_simple_manifest_last(
    output: &Path,
    stage_dir: &Path,
    files: &[(&str, &[u8])],
    manifest_name: &str,
    manifest_bytes: &[u8],
    fail_after_links: Option<usize>,
) -> Result<(), ArtifactError> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|source| ArtifactError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    sync_directory(parent)?;
    std::fs::create_dir(output).map_err(|source| ArtifactError::Io {
        path: output.to_path_buf(),
        source,
    })?;
    create_private_directory(stage_dir)?;
    sync_directory(parent)?;
    let mut staged = BTreeSet::new();
    for &(name, bytes) in files {
        write_new_synced(&stage_dir.join(name), bytes)?;
        staged.insert(PathBuf::from(name));
    }
    write_new_synced(&stage_dir.join(manifest_name), manifest_bytes)?;
    staged.insert(PathBuf::from(manifest_name));
    sync_directory(stage_dir)?;
    link_staged_manifest_last(
        output,
        stage_dir,
        parent,
        &staged,
        Path::new(manifest_name),
        fail_after_links,
        || Ok(()),
    )
}

#[allow(clippy::cognitive_complexity)] // durability ordering keeps nested syncs and commit link explicit
fn link_staged_manifest_last<F>(
    final_dir: &Path,
    stage_dir: &Path,
    parent_dir: &Path,
    staged_files: &BTreeSet<PathBuf>,
    manifest: &Path,
    fail_after_links: Option<usize>,
    require_owner: F,
) -> Result<(), ArtifactError>
where
    F: Fn() -> Result<(), ArtifactError>,
{
    let mut final_parent_dirs = BTreeSet::new();
    for relative in staged_files {
        if let Some(parent) = relative
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            let staged_parent = stage_dir.join(parent);
            if staged_parent.is_dir() {
                sync_directory(&staged_parent)?;
            }
            let mut relative_parent = PathBuf::new();
            for component in parent.components() {
                relative_parent.push(component);
                let final_parent = final_dir.join(&relative_parent);
                std::fs::create_dir_all(&final_parent).map_err(|source| ArtifactError::Io {
                    path: final_parent.clone(),
                    source,
                })?;
                final_parent_dirs.insert(final_parent);
            }
        }
    }
    sync_directory(stage_dir)?;
    let mut linked = 0_usize;
    for relative in staged_files {
        if relative == manifest {
            continue;
        }
        require_owner()?;
        let destination = final_dir.join(relative);
        std::fs::hard_link(stage_dir.join(relative), &destination).map_err(|source| {
            ArtifactError::Io {
                path: destination,
                source,
            }
        })?;
        linked += 1;
        if fail_after_links == Some(linked) {
            return Err(ArtifactError::msg(
                "injected artifact mid-publication failure",
            ));
        }
    }
    let mut final_parent_dirs = final_parent_dirs.into_iter().collect::<Vec<_>>();
    final_parent_dirs.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for directory in final_parent_dirs {
        sync_directory(&directory)?;
    }
    sync_directory(final_dir)?;
    require_owner()?;
    let destination_manifest = final_dir.join(manifest);
    std::fs::hard_link(stage_dir.join(manifest), &destination_manifest).map_err(|source| {
        ArtifactError::Io {
            path: destination_manifest,
            source,
        }
    })?;
    sync_directory(final_dir)?;
    sync_directory(parent_dir)
}

fn create_private_directory(path: &Path) -> Result<(), ArtifactError> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder.create(path).map_err(|source| ArtifactError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Audit and publish one exact launch-bound TriMul candidate.
///
/// The output directory is exclusively claimed before the first measurement.
/// The library validates every execution, computes the fixed 9-of-11 decision,
/// constructs the unchanged v4 manifest/report, and links the manifest last.
///
/// # Errors
///
/// Returns [`ArtifactError`] when evidence is invalid or publication fails.
pub fn publish_trimul_artifact(
    request: &TrimulArtifactRequest<'_>,
) -> Result<PublishedArtifact, ArtifactError> {
    let view = TrimulArtifactRequestView::from(request);
    publish_trimul_artifact_with(&view, |audit_id, publication| {
        verify_submission_paired(
            request.reward,
            request.submission,
            request.test_cases,
            request.benchmark_cases,
            request.audit_verifier,
            audit_id,
            publication,
        )
    })
}

fn publish_trimul_artifact_with<F>(
    request: &TrimulArtifactRequestView<'_>,
    verify: F,
) -> Result<PublishedArtifact, ArtifactError>
where
    F: FnOnce(&str, &mut ArtifactPublication) -> Result<Vec<ArtifactAuditBlock>, ArtifactError>,
{
    validate_device_token(request.audit_cuda_visible_device)?;
    validate_note("run health", &request.config.run_health)?;
    validate_note("source inspection notes", request.source_inspection_notes)?;
    let expected_identity = trimul_artifact_audit_identity(
        request.launch,
        request.candidate,
        request.submission,
        &request.audit_verifier.assets,
        request.config.training_secret_seed,
    );
    if request.audit_identity.contract_sha256 != expected_identity.contract_sha256
        || request.audit_identity.audit_id != expected_identity.audit_id
        || request.audit_identity.secret_seed != expected_identity.secret_seed
    {
        return Err(ArtifactError::msg(
            "TriMul artifact audit identity does not match the immutable request",
        ));
    }
    let audit_contract_sha256 = request.audit_identity.contract_sha256.clone();
    let audit_secret_seed = request.audit_identity.secret_seed;
    let audit_id = request.audit_identity.audit_id.clone();
    let mut publication = ArtifactPublication::claim(request.output, &audit_contract_sha256)?;
    publication.stage_text(Path::new("submission.py"), request.submission)?;
    publication.stage_text(Path::new("completion.txt"), request.raw_completion)?;
    publication.stage_bytes(Path::new("launch.json"), request.launch_bytes)?;
    publication.stage_bytes(Path::new("candidate.json"), request.candidate_row_bytes)?;
    publication.stage_bytes(Path::new("prompt.txt"), request.prompt_bytes)?;
    let blocks = verify(&audit_id, &mut publication)?;
    let decision = artifact_acceptance_decision(&blocks);
    let accepted = decision.accepted && request.source_inspection == SourceInspection::Clean;
    let manifest = build_manifest(
        request,
        audit_contract_sha256,
        audit_id,
        audit_secret_seed,
        &blocks,
        decision,
        accepted,
    )?;
    let manifest_path = request.output.join("manifest.json");
    let manifest_json = json_pretty(&manifest_path, &manifest)?;
    let manifest_sha256 = sha256_hex(manifest_json.as_bytes());
    publication.stage_text(
        Path::new("report.md"),
        &artifact_report(&manifest, &manifest_sha256),
    )?;
    publication.stage_text(Path::new("manifest.json"), &manifest_json)?;
    publication.publish_manifest_last()?;
    Ok(PublishedArtifact {
        output: request.output.to_path_buf(),
        manifest_path,
        accepted,
    })
}

fn verify_submission_paired(
    reward: &TrimulReward,
    submission: &str,
    expected_test_cases: usize,
    expected_benchmark_cases: usize,
    audit_verifier: &LaunchVerifierIdentity,
    audit_id: &str,
    publication: &mut ArtifactPublication,
) -> Result<Vec<ArtifactAuditBlock>, ArtifactError> {
    verify_submission_paired_with(
        expected_test_cases,
        expected_benchmark_cases,
        audit_verifier,
        audit_id,
        publication,
        |role| verify_artifact_role(reward, submission, role),
    )
}

fn verify_submission_paired_with<F>(
    expected_test_cases: usize,
    expected_benchmark_cases: usize,
    audit_verifier: &LaunchVerifierIdentity,
    audit_id: &str,
    publication: &mut ArtifactPublication,
    mut verify_role: F,
) -> Result<Vec<ArtifactAuditBlock>, ArtifactError>
where
    F: FnMut(ArtifactAuditRole) -> Result<EvidencedTrimulVerification, ArtifactError>,
{
    let schedule = artifact_audit_schedule(audit_id);
    let mut executing_device = None;
    let mut blocks = Vec::with_capacity(schedule.len());
    for (index, first) in schedule.into_iter().enumerate() {
        let second = first.other();
        let first_execution = artifact_execution(
            first,
            verify_role(first)?,
            expected_test_cases,
            expected_benchmark_cases,
            audit_verifier,
        )?;
        stage_artifact_execution(publication, index, &first_execution)?;
        let second_execution = artifact_execution(
            second,
            verify_role(second)?,
            expected_test_cases,
            expected_benchmark_cases,
            audit_verifier,
        )?;
        stage_artifact_execution(publication, index, &second_execution)?;
        for execution in [&first_execution, &second_execution] {
            if executing_device
                .as_ref()
                .is_some_and(|expected| expected != &execution.exact.executing_device)
            {
                return Err(ArtifactError::msg(
                    "paired artifact audit changed physical CUDA devices",
                ));
            }
            executing_device.get_or_insert_with(|| execution.exact.executing_device.clone());
        }
        let (reference, candidate) = if first == ArtifactAuditRole::Reference {
            (first_execution, second_execution)
        } else {
            (second_execution, first_execution)
        };
        let reference_geomean = reference
            .verification
            .geomean_ns
            .expect("strict reference evidence requires a finite geomean");
        let candidate_geomean = candidate
            .verification
            .geomean_ns
            .expect("strict candidate evidence requires a finite geomean");
        let paired_speedup = reference_geomean / candidate_geomean;
        if !paired_speedup.is_finite() || paired_speedup <= 0.0 {
            return Err(ArtifactError::msg(
                "paired artifact audit produced an invalid speedup ratio",
            ));
        }
        blocks.push(ArtifactAuditBlock {
            index,
            first,
            reference,
            candidate,
            paired_speedup,
            material_win: paired_speedup > ARTIFACT_MATERIAL_SPEEDUP,
        });
    }
    Ok(blocks)
}

fn verify_artifact_role(
    reward: &TrimulReward,
    submission: &str,
    role: ArtifactAuditRole,
) -> Result<EvidencedTrimulVerification, ArtifactError> {
    let result = match role {
        ArtifactAuditRole::Reference => reward.verify_reference_with_evidence(),
        ArtifactAuditRole::Candidate => reward.verify_submission_with_evidence(submission),
    };
    result.map_err(|error| {
        ArtifactError::msg(format!(
            "artifact {} verification failed: {error}",
            role.label()
        ))
    })
}

fn artifact_execution(
    role: ArtifactAuditRole,
    result: EvidencedTrimulVerification,
    expected_test_cases: usize,
    expected_benchmark_cases: usize,
    audit_verifier: &LaunchVerifierIdentity,
) -> Result<ArtifactAuditExecution, ArtifactError> {
    if result.isolation != audit_verifier.isolation
        || result.isolation_evidence_sha256 != audit_verifier.isolation_evidence_sha256
        || audit_verifier.runtime_preflight.runtime_hardening.len() != 1
        || result.runtime_hardening.len() != 2
        || result.runtime_hardening.iter().any(|record| {
            audit_verifier.runtime_preflight.runtime_hardening.first() != Some(record)
        })
    {
        return Err(ArtifactError::msg(format!(
            "artifact {} execution evidence differs from the selected audit preflight",
            role.label()
        )));
    }
    let exact = validate_artifact_verification_evidence(
        &result,
        expected_test_cases,
        expected_benchmark_cases,
    )
    .map_err(|error| {
        ArtifactError::msg(format!(
            "artifact {} evidence is not publication-grade: {error}",
            role.label()
        ))
    })?;
    let isolation_tier = result.isolation.tier;
    Ok(ArtifactAuditExecution {
        role,
        isolation_tier,
        isolation: result.isolation,
        isolation_evidence_sha256: result.isolation_evidence_sha256,
        runtime_hardening_evidence_sha256: result.runtime_hardening_evidence_sha256,
        runtime_hardening: result.runtime_hardening,
        timing_metric: timing_metric_for_tier(isolation_tier).to_owned(),
        verification: result.verification,
        exact,
        protected_output: result.protected_output,
        sandbox_diagnostics: result.sandbox_diagnostics,
    })
}

fn stage_artifact_execution(
    publication: &mut ArtifactPublication,
    block_index: usize,
    execution: &ArtifactAuditExecution,
) -> Result<(), ArtifactError> {
    let evidence_file = format!(
        "verification/block-{block_index:03}-{}.json",
        execution.role.label()
    );
    let evidence_json = json_pretty(&publication.stage_dir.join(&evidence_file), execution)?;
    publication.stage_text(Path::new(&evidence_file), &evidence_json)
}

fn artifact_execution_manifest(
    block_index: usize,
    execution: &ArtifactAuditExecution,
) -> Result<ArtifactAuditExecutionManifest, ArtifactError> {
    let evidence_file = format!(
        "verification/block-{block_index:03}-{}.json",
        execution.role.label()
    );
    let evidence_json = json_pretty(Path::new(&evidence_file), execution)?;
    Ok(ArtifactAuditExecutionManifest {
        role: execution.role,
        evidence_file,
        evidence_sha256: sha256_hex(evidence_json.as_bytes()),
        isolation_tier: execution.isolation_tier,
        isolation: execution.isolation.clone(),
        isolation_evidence_sha256: execution.isolation_evidence_sha256.clone(),
        runtime_hardening_evidence_sha256: execution.runtime_hardening_evidence_sha256.clone(),
        runtime_hardening: execution.runtime_hardening.clone(),
        timing_metric: execution.timing_metric.clone(),
        verification: execution.verification.clone(),
        exact: execution.exact.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_manifest(
    request: &TrimulArtifactRequestView<'_>,
    audit_contract_sha256: String,
    audit_id: String,
    audit_secret_seed: u64,
    blocks: &[ArtifactAuditBlock],
    decision: ArtifactAcceptanceDecision,
    accepted: bool,
) -> Result<ArtifactManifest, ArtifactError> {
    let launch = &request.launch.payload;
    let discovery_verifier = launch
        .verifier
        .as_ref()
        .ok_or_else(|| ArtifactError::msg("verified TriMul launch must bind verifier assets"))?;
    let audit_verifier = request.audit_verifier;
    let candidate = request.candidate;
    let mut block_manifests = Vec::with_capacity(blocks.len());
    for block in blocks {
        block_manifests.push(ArtifactAuditBlockManifest {
            index: block.index,
            first: block.first,
            reference: artifact_execution_manifest(block.index, &block.reference)?,
            candidate: artifact_execution_manifest(block.index, &block.candidate)?,
            paired_speedup: block.paired_speedup,
            material_win: block.material_win,
        });
    }
    let executing_device = block_manifests
        .first()
        .ok_or_else(|| ArtifactError::msg("fixed artifact audit contains no blocks"))?
        .reference
        .exact
        .executing_device
        .clone();
    let config = &request.config;
    Ok(ArtifactManifest {
        contract_version: ARTIFACT_CONTRACT_VERSION,
        task: "trimul",
        ferrl_commit: launch.ferrl_commit.clone(),
        run_id: launch.run.run_id.clone(),
        launch_sha256: request.launch.payload_sha256.clone(),
        launch_file_sha256: sha256_hex(request.launch_bytes),
        launch_authentication: launch.authentication,
        launch_attestation_key_id: request
            .launch
            .attestation
            .as_ref()
            .map(|attestation| attestation.key_id.clone()),
        launch_attestation_algorithm: request
            .launch
            .attestation
            .as_ref()
            .map(|_| LAUNCH_ATTESTATION_ALGORITHM.to_owned()),
        discovery_verifier: DiscoveryVerifierManifest {
            isolation_tier: discovery_verifier.isolation.tier,
            isolation_evidence_sha256: discovery_verifier.isolation_evidence_sha256.clone(),
            timing_metric: discovery_verifier.timing_metric.clone(),
            runtime_preflight_evidence_sha256: discovery_verifier
                .runtime_preflight_evidence_sha256
                .clone(),
        },
        candidate: CandidateManifest {
            record_sha256: candidate
                .record_sha256
                .clone()
                .ok_or_else(|| ArtifactError::msg("verified candidate has no record_sha256"))?,
            record_signature: candidate
                .record_signature
                .clone()
                .ok_or_else(|| ArtifactError::msg("verified candidate has no record_signature"))?,
            ledger_row_sha256: sha256_hex(request.candidate_row_bytes),
            step: candidate.step,
            prompt_index: candidate.prompt_index,
            group_index: candidate.group_index,
            rank: candidate.rank,
            world_size: candidate.world_size,
            training_reward: candidate.reward,
            completion_sha256: sha256_hex(request.raw_completion.as_bytes()),
            source_sha256: sha256_hex(request.submission.as_bytes()),
            source_inspection: SourceInspectionManifest {
                result: request.source_inspection,
                notes: request.source_inspection_notes.to_owned(),
            },
        },
        model: ModelManifest {
            family: launch.model.family.clone(),
            checkpoint_policy_sha256: launch.model.checkpoint_policy_sha256.clone(),
            tokenizer_sha256: launch.model.tokenizer_sha256.clone(),
            lora_rank: config.lora_rank,
            lora_alpha: config.lora_alpha,
            base_dtype: config.base_dtype,
            base_quantization: config.base_quantization,
        },
        config: ArtifactConfigManifest {
            run_config_source_sha256: launch.config.source_sha256.clone(),
            run_config_resolved_sha256: launch.config.resolved_sha256.clone(),
            prompt_sha256: sha256_hex(request.prompt_bytes),
            prompt_file: "prompt.txt",
            reward_profile: config.reward_profile,
            trainer_steps: config.trainer_steps,
            group_size: config.group_size,
            run_health: config.run_health.clone(),
            policy_seed: config.policy_seed,
            data_seed: config.data_seed,
            training_secret_seed: config.training_secret_seed,
            audit_secret_seed,
            scratch_max_bytes: config.scratch_max_bytes,
            verifier_parallelism: config.verifier_parallelism,
            verifier_max_procs: config.verifier_max_procs,
            verifier_cuda_device_pool: config.verifier_cuda_device_pool.clone(),
        },
        eval: EvalManifest {
            bundle_sha256: audit_verifier.assets.eval_bundle_sha256.clone(),
            bundle_file_count: audit_verifier.assets.eval_file_count,
            sandbox_image_sha256: audit_verifier.assets.image_sha256.clone(),
            sandbox_image_len_bytes: audit_verifier.assets.image_len_bytes,
            task_yml_sha256: audit_verifier.assets.task_yml_sha256.clone(),
            task_yml_len_bytes: audit_verifier.assets.task_yml_len_bytes,
            test_cases: request.test_cases,
            benchmark_cases: request.benchmark_cases,
        },
        audit: ArtifactAuditManifest {
            contract: "ferrl.trimul-artifact-audit.v2",
            audit_contract_sha256,
            audit_id,
            audit_secret_seed,
            audit_seed_derivation: ARTIFACT_AUDIT_SEED_DERIVATION,
            attempt_selection_assurance: ARTIFACT_ATTEMPT_SELECTION_ASSURANCE,
            durable_once_only: false,
            artifact_wide_false_positive_guarantee: false,
            requested_cuda_visible_device: request.audit_cuda_visible_device.to_owned(),
            isolation_tier: audit_verifier.isolation.tier,
            isolation: audit_verifier.isolation.clone(),
            isolation_evidence_sha256: audit_verifier.isolation_evidence_sha256.clone(),
            runtime_preflight: audit_verifier.runtime_preflight.clone(),
            runtime_preflight_evidence_sha256: audit_verifier
                .runtime_preflight_evidence_sha256
                .clone(),
            timing_metric: audit_verifier.timing_metric.clone(),
            executing_device,
            blocks: block_manifests,
            decision,
        },
        accepted,
    })
}

fn artifact_audit_contract_sha256(
    launch: &LaunchManifest,
    candidate: &CandidateRecord,
    submission: &str,
    verifier_assets: &TrimulVerifierIdentity,
) -> String {
    artifact_audit_contract_from_identity(
        &launch.payload_sha256,
        candidate
            .record_sha256
            .as_deref()
            .expect("verified candidate must carry record_sha256"),
        &sha256_hex(submission.as_bytes()),
        &verifier_assets.eval_bundle_sha256,
        &verifier_assets.image_sha256,
        &verifier_assets.task_yml_sha256,
    )
}

fn artifact_audit_contract_from_identity(
    launch_sha256: &str,
    candidate_sha256: &str,
    submission_sha256: &str,
    eval_bundle_sha256: &str,
    image_sha256: &str,
    task_yml_sha256: &str,
) -> String {
    let paired_blocks = (ARTIFACT_AUDIT_BLOCKS as u64).to_le_bytes();
    let required_wins = (ARTIFACT_REQUIRED_MATERIAL_WINS as u64).to_le_bytes();
    let material_speedup = ARTIFACT_MATERIAL_SPEEDUP.to_bits().to_le_bytes();
    domain_sha256(
        "ferrl.trimul-artifact-audit.v2",
        &[
            launch_sha256.as_bytes(),
            candidate_sha256.as_bytes(),
            submission_sha256.as_bytes(),
            eval_bundle_sha256.as_bytes(),
            image_sha256.as_bytes(),
            task_yml_sha256.as_bytes(),
            ARTIFACT_ACCEPTANCE_METHOD.as_bytes(),
            ARTIFACT_AUDIT_SEED_DERIVATION.as_bytes(),
            ARTIFACT_ATTEMPT_SELECTION_ASSURANCE.as_bytes(),
            b"strict_greater_than",
            &paired_blocks,
            &required_wins,
            &material_speedup,
        ],
    )
}

fn artifact_audit_secret_seed(audit_contract_sha256: &str, training_secret_seed: u64) -> u64 {
    let digest = domain_sha256(
        "ferrl.trimul-artifact-audit-seed.v1",
        &[audit_contract_sha256.as_bytes()],
    );
    let mut seed = u64::from(
        u32::from_str_radix(&digest[..8], 16)
            .expect("domain SHA-256 prefix is lowercase hexadecimal"),
    );
    if seed == training_secret_seed {
        seed ^= 1;
    }
    seed
}

fn artifact_audit_id(audit_contract_sha256: &str, audit_secret_seed: u64) -> String {
    domain_sha256(
        "ferrl.trimul-artifact-audit-attempt.v2",
        &[
            audit_contract_sha256.as_bytes(),
            &audit_secret_seed.to_le_bytes(),
        ],
    )
}

fn artifact_audit_schedule(audit_id: &str) -> Vec<ArtifactAuditRole> {
    let starting_byte =
        u8::from_str_radix(&audit_id[..2], 16).expect("domain SHA-256 is lowercase hexadecimal");
    let starts_with_reference = starting_byte & 1 == 0;
    (0..ARTIFACT_AUDIT_BLOCKS)
        .map(|index| {
            if (index % 2 == 0) == starts_with_reference {
                ArtifactAuditRole::Reference
            } else {
                ArtifactAuditRole::Candidate
            }
        })
        .collect()
}

fn artifact_acceptance_from_speedups(speedups: &[f64]) -> ArtifactAcceptanceDecision {
    let valid_speedups = speedups
        .iter()
        .filter(|speedup| speedup.is_finite() && **speedup > 0.0)
        .count();
    let material_wins = speedups
        .iter()
        .filter(|speedup| speedup.is_finite() && **speedup > ARTIFACT_MATERIAL_SPEEDUP)
        .count();
    ArtifactAcceptanceDecision {
        method: ARTIFACT_ACCEPTANCE_METHOD,
        paired_blocks: speedups.len(),
        material_speedup: ARTIFACT_MATERIAL_SPEEDUP,
        threshold_comparison: "strict_greater_than",
        required_material_wins: ARTIFACT_REQUIRED_MATERIAL_WINS,
        observed_material_wins: material_wins,
        accepted: speedups.len() == ARTIFACT_AUDIT_BLOCKS
            && valid_speedups == ARTIFACT_AUDIT_BLOCKS
            && material_wins >= ARTIFACT_REQUIRED_MATERIAL_WINS,
    }
}

fn artifact_acceptance_decision(blocks: &[ArtifactAuditBlock]) -> ArtifactAcceptanceDecision {
    artifact_acceptance_from_speedups(
        &blocks
            .iter()
            .map(|block| block.paired_speedup)
            .collect::<Vec<_>>(),
    )
}

#[allow(clippy::cognitive_complexity)]
fn artifact_report(manifest: &ArtifactManifest, manifest_sha256: &str) -> String {
    let verdict = if manifest.accepted {
        "accepted_artifact"
    } else {
        "rejected_candidate"
    };
    let attestation = manifest.launch_attestation_key_id.as_ref().map_or_else(
        || "none (local discovery authentication)".to_owned(),
        |key_id| {
            format!(
                "{} ({})",
                key_id,
                manifest
                    .launch_attestation_algorithm
                    .as_deref()
                    .unwrap_or("missing")
            )
        },
    );
    let source_inspection = match manifest.candidate.source_inspection.result {
        SourceInspection::Clean => "clean",
        SourceInspection::Suspicious => "suspicious",
    };
    let ratios = manifest
        .audit
        .blocks
        .iter()
        .map(|block| format!("{:.6}", block.paired_speedup))
        .collect::<Vec<_>>()
        .join(", ");
    let reason = if manifest.accepted {
        "accepted: at least nine of eleven same-device pairs exceed the 2% material margin"
    } else if manifest.candidate.source_inspection.result == SourceInspection::Suspicious {
        "rejected: source inspection found process/file/env/network/path probing"
    } else if !manifest.audit.decision.accepted {
        "rejected: paired audit did not establish nine material wins out of eleven"
    } else {
        "rejected: incomplete artifact evidence"
    };
    let mut out = String::new();
    writeln!(&mut out, "# TriMul Artifact Report\n").expect("String write");
    writeln!(&mut out, "## 1. Verdict\n\n{verdict}\n").expect("String write");
    writeln!(&mut out, "## 2. Discovery Provenance\n").expect("String write");
    writeln!(&mut out, "- ferrl commit: {}", manifest.ferrl_commit).expect("String write");
    writeln!(
        &mut out,
        "- Launch/config hashes: payload={}, file={}, source={}, resolved={}",
        manifest.launch_sha256,
        manifest.launch_file_sha256,
        manifest.config.run_config_source_sha256,
        manifest.config.run_config_resolved_sha256
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Launch authentication: {:?}",
        manifest.launch_authentication
    )
    .expect("String write");
    writeln!(&mut out, "- Launch attestation: {attestation}").expect("String write");
    writeln!(
        &mut out,
        "- Discovery verifier: tier={:?}, evidence={}, metric={}",
        manifest.discovery_verifier.isolation_tier,
        manifest.discovery_verifier.isolation_evidence_sha256,
        manifest.discovery_verifier.timing_metric
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Candidate: record={}, source={}, training_reward={:.6}",
        manifest.candidate.record_sha256,
        manifest.candidate.source_sha256,
        manifest.candidate.training_reward
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Prompt copy: {} ({})",
        manifest.config.prompt_file, manifest.config.prompt_sha256
    )
    .expect("String write");
    writeln!(&mut out, "- Model: family={}, checkpoint_policy_sha256={}, tokenizer_sha256={}, lora_rank={}, lora_alpha={}, base_dtype={}, base_quantization={}", manifest.model.family, manifest.model.checkpoint_policy_sha256, manifest.model.tokenizer_sha256, manifest.model.lora_rank, manifest.model.lora_alpha, manifest.model.base_dtype, manifest.model.base_quantization).expect("String write");
    writeln!(
        &mut out,
        "- Reward profile: `{}`",
        serde_json::to_string(&manifest.config.reward_profile).expect("reward profile JSON")
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Seeds: data={}, policy={}, training_secret={}, audit_secret={}",
        manifest.config.data_seed,
        manifest.config.policy_seed,
        manifest.config.training_secret_seed,
        manifest.config.audit_secret_seed
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Budget: trainer_steps={}, group_size={}, scratch_max_bytes={}, verifier_max_procs={}",
        manifest.config.trainer_steps,
        manifest.config.group_size,
        manifest.config.scratch_max_bytes,
        manifest.config.verifier_max_procs
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Eval: bundle={}, image={}, task_yml={}, tests={}, benchmarks={}",
        manifest.eval.bundle_sha256,
        manifest.eval.sandbox_image_sha256,
        manifest.eval.task_yml_sha256,
        manifest.eval.test_cases,
        manifest.eval.benchmark_cases
    )
    .expect("String write");
    writeln!(&mut out, "- Run health: {}", manifest.config.run_health).expect("String write");
    writeln!(&mut out, "- Source inspection: {source_inspection}").expect("String write");
    writeln!(
        &mut out,
        "- Source inspection notes: {}\n",
        manifest.candidate.source_inspection.notes
    )
    .expect("String write");
    writeln!(&mut out, "## 3. Launch-Bound Paired Audit\n").expect("String write");
    writeln!(&mut out, "- Audit id: {}", manifest.audit.audit_id).expect("String write");
    writeln!(
        &mut out,
        "- Audit contract: {}",
        manifest.audit.audit_contract_sha256
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Deterministic audit seed: {} ({})",
        manifest.audit.audit_secret_seed, manifest.audit.audit_seed_derivation
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Attempt selection: {}; durable_once_only={}; artifact_wide_false_positive_guarantee={}",
        manifest.audit.attempt_selection_assurance,
        manifest.audit.durable_once_only,
        manifest.audit.artifact_wide_false_positive_guarantee
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Audit verifier: tier={:?}, evidence={}, metric={}",
        manifest.audit.isolation_tier,
        manifest.audit.isolation_evidence_sha256,
        manifest.audit.timing_metric
    )
    .expect("String write");
    writeln!(
        &mut out,
        "- Executing device: {} uuid={} pci={} logical_ordinal={}",
        manifest.audit.executing_device.name,
        manifest.audit.executing_device.uuid,
        manifest.audit.executing_device.pci_bus_id,
        manifest.audit.executing_device.cuda_logical_ordinal
    )
    .expect("String write");
    writeln!(&mut out, "- Paired speedups: {ratios}").expect("String write");
    writeln!(
        &mut out,
        "- Decision: method={}, threshold={} {:.6}x, material_wins={}/{}",
        manifest.audit.decision.method,
        manifest.audit.decision.threshold_comparison,
        manifest.audit.decision.material_speedup,
        manifest.audit.decision.observed_material_wins,
        manifest.audit.decision.paired_blocks
    )
    .expect("String write");
    writeln!(&mut out, "- Acceptance reason: {reason}\n").expect("String write");
    writeln!(
        &mut out,
        "## 4. Artifact Bundle\n\n- Bundle root: .\n- Manifest SHA-256: {manifest_sha256}\n"
    )
    .expect("String write");
    writeln!(&mut out, "## 5. Operator Checklist\n").expect("String write");
    let checks = artifact_checks(manifest, manifest_sha256);
    for (pass, label) in checks {
        writeln!(
            &mut out,
            "- [{}] {label}",
            if pass { "pass" } else { "fail" }
        )
        .expect("String write");
    }
    out
}

#[allow(clippy::too_many_lines)]
fn artifact_checks(
    manifest: &ArtifactManifest,
    manifest_sha256: &str,
) -> Vec<(bool, &'static str)> {
    let speedups = manifest
        .audit
        .blocks
        .iter()
        .map(|block| block.paired_speedup)
        .collect::<Vec<_>>();
    let recomputed = artifact_acceptance_from_speedups(&speedups);
    let exact_device = &manifest.audit.executing_device;
    vec![
        (
            manifest.contract_version == ARTIFACT_CONTRACT_VERSION,
            "artifact contract is v4",
        ),
        (manifest.task == "trimul", "task is trimul"),
        (
            validate_full_git_commit(&manifest.ferrl_commit)
                && valid_sha256(&manifest.launch_sha256)
                && valid_sha256(&manifest.launch_file_sha256)
                && valid_sha256(&manifest.config.run_config_source_sha256)
                && valid_sha256(&manifest.config.run_config_resolved_sha256),
            "launch commit and config hashes are canonical",
        ),
        (
            match manifest.launch_authentication {
                LaunchAuthenticationMode::LocalEphemeralV1 => {
                    manifest.launch_attestation_key_id.is_none()
                        && manifest.launch_attestation_algorithm.is_none()
                }
                LaunchAuthenticationMode::ExternalAttestedV1 => {
                    manifest
                        .launch_attestation_key_id
                        .as_ref()
                        .is_some_and(|value| !value.is_empty())
                        && manifest.launch_attestation_algorithm.as_deref()
                            == Some(LAUNCH_ATTESTATION_ALGORITHM)
                }
            },
            "discovery authentication is recorded without relabeling",
        ),
        (
            valid_sha256(&manifest.discovery_verifier.isolation_evidence_sha256)
                && valid_sha256(
                    &manifest
                        .discovery_verifier
                        .runtime_preflight_evidence_sha256,
                )
                && manifest.discovery_verifier.timing_metric
                    == timing_metric_for_tier(manifest.discovery_verifier.isolation_tier),
            "discovery verifier provenance is complete and tier-correct",
        ),
        (
            valid_sha256(&manifest.candidate.record_sha256)
                && valid_lower_hex(&manifest.candidate.record_signature, 64)
                && valid_sha256(&manifest.candidate.ledger_row_sha256)
                && valid_sha256(&manifest.candidate.completion_sha256)
                && valid_sha256(&manifest.candidate.source_sha256),
            "candidate row, signature, completion, and source hashes are canonical",
        ),
        (
            manifest.config.prompt_file == "prompt.txt"
                && valid_sha256(&manifest.config.prompt_sha256)
                && manifest.config.reward_profile.validate().is_ok()
                && manifest.config.audit_secret_seed != manifest.config.training_secret_seed
                && manifest.config.audit_secret_seed <= TRIMUL_CASE_SEED_MAX
                && manifest.config.scratch_max_bytes > 0
                && manifest.config.verifier_max_procs > 0,
            "prompt, reward, audit seed, and verifier budgets are valid",
        ),
        (
            valid_sha256(&manifest.eval.bundle_sha256)
                && manifest.eval.bundle_file_count > 0
                && valid_sha256(&manifest.eval.sandbox_image_sha256)
                && manifest.eval.sandbox_image_len_bytes > 0
                && valid_sha256(&manifest.eval.task_yml_sha256)
                && manifest.eval.task_yml_len_bytes > 0
                && manifest.eval.test_cases > 0
                && manifest.eval.benchmark_cases > 0,
            "eval bundle, image, task, and non-empty case counts are bound",
        ),
        (
            manifest.audit.contract == "ferrl.trimul-artifact-audit.v2"
                && valid_sha256(&manifest.audit.audit_contract_sha256)
                && manifest.audit.audit_contract_sha256
                    == artifact_audit_contract_from_identity(
                        &manifest.launch_sha256,
                        &manifest.candidate.record_sha256,
                        &manifest.candidate.source_sha256,
                        &manifest.eval.bundle_sha256,
                        &manifest.eval.sandbox_image_sha256,
                        &manifest.eval.task_yml_sha256,
                    )
                && manifest.audit.audit_secret_seed
                    == artifact_audit_secret_seed(
                        &manifest.audit.audit_contract_sha256,
                        manifest.config.training_secret_seed,
                    )
                && manifest.audit.audit_seed_derivation == ARTIFACT_AUDIT_SEED_DERIVATION
                && manifest.audit.attempt_selection_assurance
                    == ARTIFACT_ATTEMPT_SELECTION_ASSURANCE
                && !manifest.audit.durable_once_only
                && !manifest.audit.artifact_wide_false_positive_guarantee
                && manifest.audit.audit_id
                    == artifact_audit_id(
                        &manifest.audit.audit_contract_sha256,
                        manifest.audit.audit_secret_seed,
                    )
                && manifest.audit.isolation.tier == manifest.audit.isolation_tier
                && manifest.audit.isolation.contract_version == VERIFIER_ISOLATION_EVIDENCE_VERSION
                && matches!(
                    (
                        manifest.audit.isolation_tier,
                        manifest.audit.isolation.uid_boundary,
                        manifest.audit.isolation.asset_transport
                    ),
                    (
                        VerifierIsolationTier::SameUidApptainerV1,
                        VerifierUidBoundary::SameHostUid,
                        VerifierAssetTransport::InProcessSealedCopy
                    ) | (
                        VerifierIsolationTier::DedicatedUidServiceV1,
                        VerifierUidBoundary::DistinctHostUid,
                        VerifierAssetTransport::ScmRightsSealedCopy
                    )
                )
                && manifest.audit.isolation.requester_uid != 0
                && manifest.audit.isolation.launcher_uid != 0
                && manifest.audit.timing_metric
                    == timing_metric_for_tier(manifest.audit.isolation_tier)
                && valid_sha256(&manifest.audit.audit_id)
                && validate_device_token(&manifest.audit.requested_cuda_visible_device).is_ok()
                && manifest.audit.isolation_evidence_sha256
                    == verifier_isolation_evidence_sha256(&manifest.audit.isolation)
                && manifest.audit.runtime_preflight.contract_version == 1
                && manifest.audit.runtime_preflight.isolation_tier == manifest.audit.isolation_tier
                && manifest.audit.runtime_preflight.isolation_evidence_sha256
                    == manifest.audit.isolation_evidence_sha256
                && manifest.audit.runtime_preflight.runtime_hardening.len() == 1
                && manifest
                    .audit
                    .runtime_preflight
                    .runtime_hardening
                    .iter()
                    .all(|record| {
                        record.get("contract").and_then(serde_json::Value::as_str)
                            == Some(TRIMUL_RUNTIME_HARDENING_CONTRACT)
                    })
                && manifest.audit.runtime_preflight_evidence_sha256
                    == crate::trimul::runtime_preflight_evidence_sha256(
                        &manifest.audit.runtime_preflight,
                    ),
            "audit uses the selected tier and records operator-trusted attempt selection",
        ),
        (
            manifest.audit.blocks.len() == ARTIFACT_AUDIT_BLOCKS
                && manifest
                    .audit
                    .blocks
                    .iter()
                    .zip(artifact_audit_schedule(&manifest.audit.audit_id))
                    .enumerate()
                    .all(|(index, (block, expected))| {
                        block.index == index && block.first == expected
                    }),
            "audit contains the fixed eleven ordered pairs",
        ),
        (
            exact_device.contract == "ferrl.executing-device.v1"
                && exact_device.cuda_logical_ordinal == 0
                && !exact_device.name.trim().is_empty()
                && valid_lower_hex(&exact_device.uuid, 16)
                && exact_device.pci_bus_id.contains(':')
                && exact_device.pci_bus_id.contains('.'),
            "audit records one canonical executing CUDA device",
        ),
        (
            manifest
                .audit
                .blocks
                .iter()
                .all(|block| artifact_block_is_valid(block, manifest, exact_device)),
            "all raw executions bind exact cases, verifier evidence, and one physical GPU",
        ),
        (
            manifest.audit.decision.method == ARTIFACT_ACCEPTANCE_METHOD
                && manifest.audit.decision.paired_blocks == ARTIFACT_AUDIT_BLOCKS
                && manifest.audit.decision.material_speedup == ARTIFACT_MATERIAL_SPEEDUP
                && manifest.audit.decision.threshold_comparison == "strict_greater_than"
                && manifest.audit.decision.required_material_wins
                    == ARTIFACT_REQUIRED_MATERIAL_WINS
                && manifest.audit.decision.observed_material_wins
                    == recomputed.observed_material_wins
                && manifest.audit.decision.accepted == recomputed.accepted,
            "predeclared empirical 9-of-11 material-win rule is recorded",
        ),
        (
            manifest.candidate.source_inspection.result == SourceInspection::Clean,
            "source inspection found no process/file/env/network/path probing",
        ),
        (
            !manifest.candidate.source_inspection.notes.trim().is_empty()
                && !manifest.config.run_health.trim().is_empty(),
            "source-inspection notes and run health are recorded",
        ),
        (
            manifest.accepted
                == (manifest.audit.decision.accepted
                    && manifest.candidate.source_inspection.result == SourceInspection::Clean),
            "final verdict follows the audit and source-inspection gates",
        ),
        (valid_sha256(manifest_sha256), "manifest hash recorded"),
    ]
}

fn artifact_block_is_valid(
    block: &ArtifactAuditBlockManifest,
    manifest: &ArtifactManifest,
    exact_device: &TrimulExecutingDevice,
) -> bool {
    let role_binding = block.reference.role == ArtifactAuditRole::Reference
        && block.candidate.role == ArtifactAuditRole::Candidate;
    let executions = [&block.reference, &block.candidate]
        .iter()
        .all(|execution| {
            let expected_file = format!(
                "verification/block-{:03}-{}.json",
                block.index,
                execution.role.label()
            );
            execution.evidence_file == expected_file
                && execution.isolation_tier == manifest.audit.isolation_tier
                && execution.isolation == manifest.audit.isolation
                && execution.isolation_evidence_sha256 == manifest.audit.isolation_evidence_sha256
                && execution.isolation_evidence_sha256
                    == verifier_isolation_evidence_sha256(&execution.isolation)
                && execution.runtime_hardening.len() == 2
                && execution.runtime_hardening.iter().all(|record| {
                    manifest.audit.runtime_preflight.runtime_hardening.first() == Some(record)
                })
                && runtime_hardening_evidence_sha256(&execution.runtime_hardening)
                    == execution.runtime_hardening_evidence_sha256
                && execution.timing_metric == manifest.audit.timing_metric
                && &execution.exact.executing_device == exact_device
                && execution.verification.correct
                && execution.verification.geomean_ns.is_some()
                && execution.exact.sandbox_status == RunStatus::Exited(0)
                && execution.exact.test_exit == 0
                && execution.exact.benchmark_exit == 0
                && execution.exact.test_cases.len() == manifest.eval.test_cases
                && execution.exact.benchmark_cases.len() == manifest.eval.benchmark_cases
                && execution
                    .exact
                    .test_cases
                    .iter()
                    .enumerate()
                    .all(|(index, case)| case.index == index && case.passed)
                && execution
                    .exact
                    .benchmark_cases
                    .iter()
                    .enumerate()
                    .all(|(index, case)| {
                        case.index == index
                            && case.runs >= 3
                            && case.mean_ns.is_finite()
                            && case.mean_ns > 0.0
                    })
                && valid_sha256(&execution.evidence_sha256)
                && valid_sha256(&execution.runtime_hardening_evidence_sha256)
                && valid_sha256(&execution.exact.protected_output_sha256)
                && valid_sha256(&execution.exact.sandbox_diagnostics_sha256)
        });
    let ratio_matches = block
        .reference
        .verification
        .geomean_ns
        .zip(block.candidate.verification.geomean_ns)
        .is_some_and(|(reference, candidate)| {
            let ratio = reference / candidate;
            ratio.is_finite()
                && ratio > 0.0
                && ratio.to_bits() == block.paired_speedup.to_bits()
                && block.material_win == (ratio > ARTIFACT_MATERIAL_SPEEDUP)
        });
    role_binding && executions && ratio_matches
}

fn runtime_hardening_evidence_sha256(records: &[serde_json::Value]) -> String {
    let encoded = records
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .expect("runtime hardening evidence contains serializable JSON values");
    let fields = encoded.iter().map(String::as_bytes).collect::<Vec<_>>();
    domain_sha256("ferrl.trimul-runtime-hardening-evidence.v1", &fields)
}

fn validate_note(label: &str, value: &str) -> Result<(), ArtifactError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(ArtifactError::msg(format!(
            "{label} must be non-empty without leading or trailing whitespace"
        )));
    }
    if value.len() > 4096 || value.chars().any(char::is_control) {
        return Err(ArtifactError::msg(format!(
            "{label} must be at most 4096 bytes and contain no control characters"
        )));
    }
    Ok(())
}

fn validate_device_token(value: &str) -> Result<(), ArtifactError> {
    if value.is_empty()
        || value.trim() != value
        || value.contains(',')
        || value.chars().any(char::is_whitespace)
        || value.bytes().any(|byte| byte == 0)
    {
        return Err(ArtifactError::msg(
            "audit CUDA visible device must be one non-empty CUDA device token",
        ));
    }
    Ok(())
}

fn validate_full_git_commit(value: &str) -> bool {
    valid_lower_hex(value, 20)
}

fn valid_sha256(value: &str) -> bool {
    valid_lower_hex(value, 32)
}

fn valid_lower_hex(value: &str, bytes: usize) -> bool {
    value.len() == bytes * 2
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<String, ArtifactError> {
    serde_json::to_string_pretty(value).map_err(|source| ArtifactError::Serialization {
        path: path.to_path_buf(),
        source,
    })
}

fn read_bytes(path: &Path) -> Result<Vec<u8>, ArtifactError> {
    std::fs::read(path).map_err(|source| ArtifactError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), ArtifactError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| ArtifactError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(bytes).map_err(|source| ArtifactError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    file.sync_all().map_err(|source| ArtifactError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn sync_directory(path: &Path) -> Result<(), ArtifactError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| ArtifactError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing hexadecimal cannot fail");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let serial = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "ferrl-artifact-{label}-{}-{nanos}-{serial}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn acceptance_requires_nine_strict_material_wins() {
        let mut speedups = vec![1.03; 9];
        speedups.extend([1.02; 2]);
        let accepted = artifact_acceptance_from_speedups(&speedups);
        assert!(accepted.accepted);
        assert_eq!(accepted.observed_material_wins, 9);
        speedups[0] = 1.02;
        assert!(!artifact_acceptance_from_speedups(&speedups).accepted);
        assert!(!artifact_acceptance_from_speedups(&[1.03; 8]).accepted);
        assert!(!artifact_acceptance_from_speedups(&[f64::NAN; 11]).accepted);
    }

    #[test]
    fn audit_schedule_is_deterministic_and_alternating() {
        let reference_first = artifact_audit_schedule(&format!("00{}", "11".repeat(31)));
        let candidate_first = artifact_audit_schedule(&format!("01{}", "11".repeat(31)));
        assert_eq!(reference_first.len(), ARTIFACT_AUDIT_BLOCKS);
        assert_eq!(reference_first[0], ArtifactAuditRole::Reference);
        assert_eq!(candidate_first[0], ArtifactAuditRole::Candidate);
        assert!(reference_first.windows(2).all(|pair| pair[0] != pair[1]));
        assert_eq!(
            reference_first,
            artifact_audit_schedule(&format!("00{}", "11".repeat(31)))
        );
    }

    #[test]
    fn shared_manifest_last_transaction_links_manifest_last() {
        let tmp = TestDir::new("manifest-last");
        let output = tmp.0.join("artifact");
        let stage = tmp.0.join(".stage");
        publish_simple_manifest_last(
            &output,
            &stage,
            &[("payload.json", b"payload")],
            "manifest.json",
            b"manifest",
            None,
        )
        .unwrap();
        assert_eq!(
            std::fs::read(output.join("payload.json")).unwrap(),
            b"payload"
        );
        assert_eq!(
            std::fs::read(output.join("manifest.json")).unwrap(),
            b"manifest"
        );
    }

    #[test]
    fn shared_manifest_last_failure_never_links_manifest_and_claim_blocks_retry() {
        let tmp = TestDir::new("manifest-fault");
        let output = tmp.0.join("artifact");
        let stage = tmp.0.join(".stage");
        let error = publish_simple_manifest_last(
            &output,
            &stage,
            &[("one", b"1"), ("two", b"2")],
            "manifest.json",
            b"manifest",
            Some(1),
        )
        .unwrap_err();
        assert!(error.to_string().contains("mid-publication"));
        assert!(!output.join("manifest.json").exists());
        let retry = publish_simple_manifest_last(
            &output,
            &tmp.0.join(".retry"),
            &[("one", b"1")],
            "manifest.json",
            b"manifest",
            None,
        )
        .unwrap_err();
        match retry {
            ArtifactError::Io { path, source } => {
                assert_eq!(path, output);
                assert_eq!(source.kind(), std::io::ErrorKind::AlreadyExists);
            }
            other => panic!("unexpected retry error: {other}"),
        }
    }

    fn isolation_for_test() -> VerifierIsolationEvidence {
        VerifierIsolationEvidence {
            contract_version: VERIFIER_ISOLATION_EVIDENCE_VERSION,
            tier: VerifierIsolationTier::SameUidApptainerV1,
            requester_uid: 1000,
            launcher_uid: 1000,
            uid_boundary: VerifierUidBoundary::SameHostUid,
            asset_transport: VerifierAssetTransport::InProcessSealedCopy,
            apptainer_path: PathBuf::from("/usr/bin/apptainer"),
            apptainer_sha256: "11".repeat(32),
            apptainer_len_bytes: 1,
            apptainer_version: "apptainer version 1.4.0".to_owned(),
            work_root: PathBuf::from("/tmp/ferrl-verifier-test"),
            work_root_uid: 1000,
            work_root_device: 1,
            work_root_inode: 2,
            work_root_mode: 0o700,
        }
    }

    fn hardening_for_test() -> serde_json::Value {
        serde_json::json!({
            "arch": "x86_64",
            "cap_amb": "0000000000000000",
            "cap_bnd": "00000000a80425fb",
            "cap_eff": "0000000000000000",
            "cap_inh": "0000000000000000",
            "cap_prm": "0000000000000000",
            "cgroup": false,
            "contract": TRIMUL_RUNTIME_HARDENING_CONTRACT,
            "denial_probes": ["bpf", "io_uring", "namespace", "network", "parent_proc", "pidfd_getfd", "process_vm", "ptrace"],
            "dumpable": 0,
            "landlock": false,
            "network_socket_policy": "af_unix_only",
            "no_new_privs": 1,
            "physical_gpu_isolation": false,
            "seccomp_filters": 1,
            "seccomp_mode": 2,
            "seccomp_policy": "x86_64-tsync-af-unix-v1",
            "seccomp_tsync": true,
            "unix_socket_probe": true,
        })
    }

    fn device_for_test() -> TrimulExecutingDevice {
        TrimulExecutingDevice {
            contract: "ferrl.executing-device.v1".to_owned(),
            cuda_logical_ordinal: 0,
            name: "NVIDIA H100 80GB HBM3".to_owned(),
            pci_bus_id: "0000:01:00.0".to_owned(),
            uuid: "ab".repeat(16),
        }
    }

    fn preflight_for_test(isolation: &VerifierIsolationEvidence) -> TrimulRuntimePreflightEvidence {
        let hardening = vec![hardening_for_test()];
        TrimulRuntimePreflightEvidence {
            contract_version: 1,
            isolation_tier: isolation.tier,
            isolation_evidence_sha256: verifier_isolation_evidence_sha256(isolation),
            probe_submission_sha256: "22".repeat(32),
            runtime_hardening_evidence_sha256: runtime_hardening_evidence_sha256(&hardening),
            runtime_hardening: hardening,
        }
    }

    fn verifier_for_test() -> LaunchVerifierIdentity {
        let isolation = isolation_for_test();
        let runtime_preflight = preflight_for_test(&isolation);
        LaunchVerifierIdentity {
            assets: TrimulVerifierIdentity {
                image_sha256: "31".repeat(32),
                image_len_bytes: 1,
                eval_bundle_sha256: "32".repeat(32),
                eval_file_count: 1,
                task_yml_sha256: "33".repeat(32),
                task_yml_len_bytes: 1,
            },
            isolation_evidence_sha256: verifier_isolation_evidence_sha256(&isolation),
            isolation,
            timing_metric: timing_metric_for_tier(VerifierIsolationTier::SameUidApptainerV1)
                .to_owned(),
            runtime_hardening_contract: TRIMUL_RUNTIME_HARDENING_CONTRACT.to_owned(),
            runtime_preflight_evidence_sha256: crate::trimul::runtime_preflight_evidence_sha256(
                &runtime_preflight,
            ),
            runtime_preflight,
        }
    }

    fn protected_grade_for_test() -> String {
        let hardening = serde_json::to_string(&hardening_for_test()).unwrap();
        let device = serde_json::to_string(&device_for_test()).unwrap();
        let prelude = format!(
            "ferrl-verifier-isolation-tier: {}\nferrl-timing-metric: {}\nferrl-candidate-hardening: {hardening}\n",
            VerifierIsolationTier::SameUidApptainerV1.as_str(),
            timing_metric_for_tier(VerifierIsolationTier::SameUidApptainerV1),
        );
        format!(
            "{prelude}ferrl-entry: test-v4\nferrl-executing-device: {device}\ntest-count: 1\ntest.0.spec: seqlen: 8; bs: 1\ntest.0.status: pass\ncheck: pass\ntest-exit: 0\n===FERRL-BENCH===\n{prelude}ferrl-entry: benchmark-v4\nferrl-executing-device: {device}\nbenchmark-count: 1\nbenchmark.0.spec: seqlen: 16; bs: 1\nbenchmark.0.runs: 100\nbenchmark.0.mean: 10\nbenchmark.0.std: 0.5\nbenchmark.0.err: 0.05\nbenchmark.0.best: 9\nbenchmark.0.worst: 11\ncheck: pass\nbenchmark-exit: 0\n"
        )
    }

    fn evidenced_verification_for_test() -> EvidencedTrimulVerification {
        let isolation = isolation_for_test();
        let hardening = vec![hardening_for_test(), hardening_for_test()];
        let protected_output = protected_grade_for_test();
        EvidencedTrimulVerification {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![10.0],
                geomean_ns: crate::trimul::geomean(&[10.0]),
                speedup: None,
            },
            isolation_evidence_sha256: verifier_isolation_evidence_sha256(&isolation),
            isolation,
            runtime_hardening_evidence_sha256: runtime_hardening_evidence_sha256(&hardening),
            runtime_hardening: hardening,
            sandbox_status: RunStatus::Exited(0),
            protected_output_sha256: sha256_hex(protected_output.as_bytes()),
            protected_output,
            sandbox_diagnostics_sha256: sha256_hex(b""),
            sandbox_diagnostics: String::new(),
        }
    }

    fn execution_for_test(
        role: ArtifactAuditRole,
        verifier: &LaunchVerifierIdentity,
    ) -> ArtifactAuditExecution {
        artifact_execution(role, evidenced_verification_for_test(), 1, 1, verifier).unwrap()
    }

    fn blocks_for_test(
        verifier: &LaunchVerifierIdentity,
        audit_id: &str,
    ) -> Vec<ArtifactAuditBlock> {
        artifact_audit_schedule(audit_id)
            .into_iter()
            .enumerate()
            .map(|(index, first)| {
                let mut reference = execution_for_test(ArtifactAuditRole::Reference, verifier);
                reference.verification.benchmark_means_ns = vec![10.3];
                reference.verification.geomean_ns = Some(10.3);
                reference.exact.benchmark_cases[0].mean_ns = 10.3;
                let candidate = execution_for_test(ArtifactAuditRole::Candidate, verifier);
                let paired_speedup = reference.verification.geomean_ns.unwrap()
                    / candidate.verification.geomean_ns.unwrap();
                ArtifactAuditBlock {
                    index,
                    first,
                    reference,
                    candidate,
                    paired_speedup,
                    material_win: true,
                }
            })
            .collect()
    }

    fn manifest_for_test() -> ArtifactManifest {
        let verifier = verifier_for_test();
        let launch_sha256 = "41".repeat(32);
        let record_sha256 = "42".repeat(32);
        let source_sha256 = "43".repeat(32);
        let audit_contract_sha256 = artifact_audit_contract_from_identity(
            &launch_sha256,
            &record_sha256,
            &source_sha256,
            &verifier.assets.eval_bundle_sha256,
            &verifier.assets.image_sha256,
            &verifier.assets.task_yml_sha256,
        );
        let training_secret_seed = 17;
        let audit_secret_seed =
            artifact_audit_secret_seed(&audit_contract_sha256, training_secret_seed);
        let audit_id = artifact_audit_id(&audit_contract_sha256, audit_secret_seed);
        let blocks = blocks_for_test(&verifier, &audit_id);
        let decision = artifact_acceptance_decision(&blocks);
        let block_manifests = blocks
            .iter()
            .map(|block| ArtifactAuditBlockManifest {
                index: block.index,
                first: block.first,
                reference: artifact_execution_manifest(block.index, &block.reference).unwrap(),
                candidate: artifact_execution_manifest(block.index, &block.candidate).unwrap(),
                paired_speedup: block.paired_speedup,
                material_win: block.material_win,
            })
            .collect();
        ArtifactManifest {
            contract_version: ARTIFACT_CONTRACT_VERSION,
            task: "trimul",
            ferrl_commit: "51".repeat(20),
            run_id: "test-run".to_owned(),
            launch_sha256,
            launch_file_sha256: "52".repeat(32),
            launch_authentication: LaunchAuthenticationMode::LocalEphemeralV1,
            launch_attestation_key_id: None,
            launch_attestation_algorithm: None,
            discovery_verifier: DiscoveryVerifierManifest {
                isolation_tier: verifier.isolation.tier,
                isolation_evidence_sha256: verifier.isolation_evidence_sha256.clone(),
                timing_metric: verifier.timing_metric.clone(),
                runtime_preflight_evidence_sha256: verifier
                    .runtime_preflight_evidence_sha256
                    .clone(),
            },
            candidate: CandidateManifest {
                record_sha256,
                record_signature: "53".repeat(64),
                ledger_row_sha256: "54".repeat(32),
                step: 1,
                prompt_index: 2,
                group_index: 3,
                rank: 0,
                world_size: 1,
                training_reward: 1.5,
                completion_sha256: "55".repeat(32),
                source_sha256,
                source_inspection: SourceInspectionManifest {
                    result: SourceInspection::Clean,
                    notes: "inspected process, file, environment, network, and path access"
                        .to_owned(),
                },
            },
            model: ModelManifest {
                family: "gemma4".to_owned(),
                checkpoint_policy_sha256: "56".repeat(32),
                tokenizer_sha256: "57".repeat(32),
                lora_rank: 8,
                lora_alpha: 16.0,
                base_dtype: "bf16",
                base_quantization: "none",
            },
            config: ArtifactConfigManifest {
                run_config_source_sha256: "58".repeat(32),
                run_config_resolved_sha256: "59".repeat(32),
                prompt_sha256: "60".repeat(32),
                prompt_file: "prompt.txt",
                reward_profile: TrimulRewardProfile::default(),
                trainer_steps: 10,
                group_size: 4,
                run_health: "healthy".to_owned(),
                policy_seed: 1,
                data_seed: 2,
                training_secret_seed,
                audit_secret_seed,
                scratch_max_bytes: 1024,
                verifier_parallelism: 1,
                verifier_max_procs: 32,
                verifier_cuda_device_pool: vec!["0".to_owned()],
            },
            eval: EvalManifest {
                bundle_sha256: verifier.assets.eval_bundle_sha256.clone(),
                bundle_file_count: verifier.assets.eval_file_count,
                sandbox_image_sha256: verifier.assets.image_sha256.clone(),
                sandbox_image_len_bytes: verifier.assets.image_len_bytes,
                task_yml_sha256: verifier.assets.task_yml_sha256.clone(),
                task_yml_len_bytes: verifier.assets.task_yml_len_bytes,
                test_cases: 1,
                benchmark_cases: 1,
            },
            audit: ArtifactAuditManifest {
                contract: "ferrl.trimul-artifact-audit.v2",
                audit_contract_sha256,
                audit_id,
                audit_secret_seed,
                audit_seed_derivation: ARTIFACT_AUDIT_SEED_DERIVATION,
                attempt_selection_assurance: ARTIFACT_ATTEMPT_SELECTION_ASSURANCE,
                durable_once_only: false,
                artifact_wide_false_positive_guarantee: false,
                requested_cuda_visible_device: "0".to_owned(),
                isolation_tier: verifier.isolation.tier,
                isolation: verifier.isolation.clone(),
                isolation_evidence_sha256: verifier.isolation_evidence_sha256.clone(),
                runtime_preflight: verifier.runtime_preflight.clone(),
                runtime_preflight_evidence_sha256: verifier
                    .runtime_preflight_evidence_sha256
                    .clone(),
                timing_metric: verifier.timing_metric.clone(),
                executing_device: device_for_test(),
                blocks: block_manifests,
                decision,
            },
            accepted: true,
        }
    }

    #[test]
    fn production_manifest_report_and_checks_follow_the_fixed_contract() {
        let mut manifest = manifest_for_test();
        let manifest_sha256 = "61".repeat(32);
        let checks = artifact_checks(&manifest, &manifest_sha256);
        assert!(checks.iter().all(|(pass, _)| *pass), "{checks:?}");
        let report = artifact_report(&manifest, &manifest_sha256);
        assert!(report.contains("accepted_artifact"), "{report}");
        assert!(!report.contains("[fail]"), "{report}");
        assert!(serde_json::to_string_pretty(&manifest)
            .unwrap()
            .contains("\"contract_version\": 4"));

        manifest.candidate.source_inspection.result = SourceInspection::Suspicious;
        manifest.accepted = false;
        let rejected = artifact_report(&manifest, &manifest_sha256);
        assert!(rejected.contains("rejected_candidate"), "{rejected}");
        assert!(rejected.contains("[fail] source inspection"), "{rejected}");
    }

    fn artifact_config_for_test() -> TrimulArtifactConfig {
        TrimulArtifactConfig {
            lora_rank: 8,
            lora_alpha: 16.0,
            base_dtype: "bf16",
            base_quantization: "none",
            reward_profile: TrimulRewardProfile::default(),
            trainer_steps: 10,
            group_size: 4,
            run_health: "healthy".to_owned(),
            policy_seed: 1,
            data_seed: 2,
            training_secret_seed: 17,
            scratch_max_bytes: 1024,
            verifier_parallelism: 1,
            verifier_max_procs: 32,
            verifier_cuda_device_pool: vec!["0".to_owned()],
        }
    }

    fn launch_and_candidate_for_test(
        verifier: &LaunchVerifierIdentity,
        config: &TrimulArtifactConfig,
    ) -> (LaunchManifest, CandidateRecord) {
        use crate::orchestration::{
            LaunchCandidateLedger, LaunchConfigSnapshot, LaunchModelIdentity, LaunchPayload,
            LaunchPromptIdentity, LaunchRunIdentity, LaunchSampleIdentity,
        };

        let signer = crate::telemetry::CandidateSigner::generate().unwrap();
        let payload = LaunchPayload {
            task: "trimul".to_owned(),
            ferrl_commit: "71".repeat(20),
            authentication: LaunchAuthenticationMode::LocalEphemeralV1,
            run: LaunchRunIdentity {
                group_id: "test-run".to_owned(),
                run_id: "test-run".to_owned(),
                data_parallel_rank: 0,
                data_parallel_world_size: 1,
                tensor_parallel_rank: 0,
                tensor_parallel_world_size: 1,
            },
            config: LaunchConfigSnapshot {
                source_sha256: "72".repeat(32),
                resolved_sha256: "73".repeat(32),
                resolved: serde_json::json!({
                    "task": "trimul",
                    "policy": {
                        "lora_rank": config.lora_rank,
                        "lora_alpha": config.lora_alpha,
                        "base_dtype": config.base_dtype,
                        "base_quantization": config.base_quantization,
                        "seed": config.policy_seed,
                    },
                    "data": { "seed": config.data_seed },
                    "trainer": {
                        "steps": config.trainer_steps,
                        "group_size": config.group_size,
                    },
                    "trimul": {
                        "secret_seed": config.training_secret_seed,
                        "reward": config.reward_profile,
                        "scratch_max_bytes": config.scratch_max_bytes,
                        "verifier_parallelism": config.verifier_parallelism,
                        "verifier_max_procs": config.verifier_max_procs,
                        "verifier_cuda_device_pool": &config.verifier_cuda_device_pool,
                    },
                }),
            },
            model: LaunchModelIdentity {
                family: "gemma4".to_owned(),
                checkpoint_policy_sha256: "74".repeat(32),
                tokenizer_sha256: "75".repeat(32),
                resolved_eos_token_id: None,
            },
            prompt: Some(LaunchPromptIdentity {
                file: "prompt.txt".to_owned(),
                sha256: sha256_hex(b"prompt"),
                len_bytes: 6,
            }),
            training_samples: Some(LaunchSampleIdentity {
                sha256: "76".repeat(32),
                count: 1,
            }),
            held_out_samples: Some(LaunchSampleIdentity {
                sha256: "77".repeat(32),
                count: 1,
            }),
            verifier: Some(verifier.clone()),
            candidate_ledger: LaunchCandidateLedger {
                file: "candidates.jsonl".to_owned(),
                format_version: 1,
                row_digest_domain: CandidateRecord::DIGEST_DOMAIN.to_owned(),
                row_signature_algorithm: "ed25519".to_owned(),
                signing_public_key: signer.public_key_hex(),
            },
        };
        let launch = LaunchManifest::new(payload).unwrap();
        let mut candidate =
            CandidateRecord::new(1, 0, 1, 2, 3, 1.5, 4, "```python\npass\n```\n".to_owned());
        candidate.reward_metadata = Some(serde_json::json!({
            "source_sha256": sha256_hex(b"pass\n"),
        }));
        let candidate = signer
            .sign_candidate(&candidate, &launch.payload_sha256)
            .unwrap();
        (launch, candidate)
    }

    #[cfg(unix)]
    #[test]
    #[allow(clippy::cognitive_complexity)] // one binding control checks every provenance preimage
    fn opaque_request_binding_rejects_decomposed_provenance_substitution() {
        let tmp = TestDir::new("request-binding");
        let image = tmp.0.join("image.sif");
        let eval = tmp.0.join("eval");
        let scratch = tmp.0.join("scratch");
        std::fs::create_dir(&eval).unwrap();
        std::fs::create_dir(&scratch).unwrap();
        std::fs::write(&image, b"image").unwrap();
        for file in ["eval.py", "reference.py", "task.py", "utils.py"] {
            std::fs::write(eval.join(file), b"# fixture\n").unwrap();
        }
        std::fs::write(
            eval.join("task.yml"),
            b"tests:\n  - {\"seqlen\": 8, \"bs\": 1, \"dim\": 4, \"hiddendim\": 4, \"seed\": 1, \"nomask\": true, \"distribution\": \"normal\"}\nbenchmarks:\n  - {\"seqlen\": 16, \"bs\": 1, \"dim\": 4, \"hiddendim\": 4, \"seed\": 2, \"nomask\": false, \"distribution\": \"cauchy\"}\n",
        )
        .unwrap();
        let assets = crate::trimul::TrimulVerifierAssets::capture(&image, &eval, &scratch).unwrap();
        let (test_cases, benchmark_cases) =
            crate::trimul::parse_task_yml(assets.task_yml()).unwrap();
        let mut verifier = verifier_for_test();
        verifier.assets = assets.identity().clone();
        let config = artifact_config_for_test();
        let (launch, candidate) = launch_and_candidate_for_test(&verifier, &config);
        let launch_bytes = launch.to_pretty_bytes().unwrap();
        let candidate_row = serde_json::to_vec(&candidate).unwrap();
        let identity = trimul_artifact_audit_identity(
            &launch,
            &candidate,
            "pass\n",
            assets.identity(),
            config.training_secret_seed,
        );
        let reward = TrimulReward::new(assets, &scratch)
            .with_cases(test_cases, benchmark_cases)
            .with_secret_seed(identity.secret_seed())
            .with_reward_profile(config.reward_profile)
            .unwrap();
        let output = tmp.0.join("artifact");
        let request = TrimulArtifactRequest::bind(
            &output,
            &launch,
            &launch_bytes,
            &candidate,
            &candidate_row,
            &candidate.completion,
            b"prompt",
            "pass\n",
            &identity,
            &reward,
            &verifier,
            1,
            1,
            "0",
            SourceInspection::Clean,
            "clean source inspection",
            config.clone(),
        )
        .unwrap();
        let view = TrimulArtifactRequestView::from(&request);
        assert_eq!(view.submission, "pass\n");
        assert_eq!(view.candidate_row_bytes, candidate_row);

        let mut substituted_row = candidate_row.clone();
        substituted_row.push(b' ');
        let error = TrimulArtifactRequest::bind(
            &output,
            &launch,
            &launch_bytes,
            &candidate,
            &substituted_row,
            &candidate.completion,
            b"prompt",
            "pass\n",
            &identity,
            &reward,
            &verifier,
            1,
            1,
            "0",
            SourceInspection::Clean,
            "clean source inspection",
            config,
        )
        .err()
        .expect("substituted candidate row must be rejected")
        .to_string();
        assert!(error.contains("candidate bytes"), "{error}");
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // end-to-end control asserts each retained bundle layer
    fn full_library_owned_publication_builds_and_commits_the_v4_bundle() {
        let tmp = TestDir::new("full-publication");
        let verifier = verifier_for_test();
        let config = artifact_config_for_test();
        let (launch, candidate) = launch_and_candidate_for_test(&verifier, &config);
        let launch_bytes = launch.to_pretty_bytes().unwrap();
        let candidate_row = serde_json::to_vec(&candidate).unwrap();
        let identity = trimul_artifact_audit_identity(
            &launch,
            &candidate,
            "pass\n",
            &verifier.assets,
            config.training_secret_seed,
        );
        assert_ne!(identity.secret_seed(), config.training_secret_seed);
        let output = tmp.0.join("artifact");
        let view = TrimulArtifactRequestView {
            output: &output,
            launch: &launch,
            launch_bytes: &launch_bytes,
            candidate: &candidate,
            candidate_row_bytes: &candidate_row,
            raw_completion: &candidate.completion,
            prompt_bytes: b"prompt",
            submission: "pass\n",
            audit_identity: &identity,
            audit_verifier: &verifier,
            test_cases: 1,
            benchmark_cases: 1,
            audit_cuda_visible_device: "0",
            source_inspection: SourceInspection::Clean,
            source_inspection_notes: "clean source inspection",
            config: &config,
        };
        let published = publish_trimul_artifact_with(&view, |audit_id, publication| {
            let blocks = blocks_for_test(&verifier, audit_id);
            for block in &blocks {
                stage_artifact_execution(publication, block.index, &block.reference)?;
                stage_artifact_execution(publication, block.index, &block.candidate)?;
            }
            Ok(blocks)
        })
        .unwrap();
        assert!(published.accepted());
        assert_eq!(published.output(), output);
        assert!(published.manifest_path().is_file());
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(published.manifest_path()).unwrap()).unwrap();
        assert_eq!(manifest["contract_version"], serde_json::json!(4));
        assert_eq!(manifest["accepted"], serde_json::json!(true));
        assert!(output.join("report.md").is_file());
        assert_eq!(
            std::fs::read_dir(output.join("verification"))
                .unwrap()
                .count(),
            2 * ARTIFACT_AUDIT_BLOCKS
        );

        let rejected_output = tmp.0.join("rejected-artifact");
        let rejected_view = TrimulArtifactRequestView {
            output: &rejected_output,
            launch: &launch,
            launch_bytes: &launch_bytes,
            candidate: &candidate,
            candidate_row_bytes: &candidate_row,
            raw_completion: &candidate.completion,
            prompt_bytes: b"prompt",
            submission: "pass\n",
            audit_identity: &identity,
            audit_verifier: &verifier,
            test_cases: 1,
            benchmark_cases: 1,
            audit_cuda_visible_device: "0",
            source_inspection: SourceInspection::Clean,
            source_inspection_notes: "clean source inspection",
            config: &config,
        };
        let rejected = publish_trimul_artifact_with(&rejected_view, |audit_id, publication| {
            verify_submission_paired_with(1, 1, &verifier, audit_id, publication, |_| {
                Ok(evidenced_verification_for_test())
            })
        })
        .unwrap();
        assert!(!rejected.accepted());
        let rejected_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(rejected.manifest_path()).unwrap()).unwrap();
        assert_eq!(rejected_manifest["audit"]["decision"]["accepted"], false);
        assert_eq!(rejected_manifest["accepted"], false);
        let rejected_report = std::fs::read_to_string(rejected_output.join("report.md")).unwrap();
        assert!(rejected_report.contains("rejected_candidate"));
        assert_eq!(
            std::fs::read_dir(rejected_output.join("verification"))
                .unwrap()
                .count(),
            2 * ARTIFACT_AUDIT_BLOCKS
        );
    }

    #[test]
    fn paired_production_audit_rejects_complete_but_nonwinning_evidence() {
        let tmp = TestDir::new("rejected-audit");
        let verifier = verifier_for_test();
        let output = tmp.0.join("artifact");
        let mut publication = ArtifactPublication::claim(&output, &"7b".repeat(32)).unwrap();
        let blocks = verify_submission_paired_with(
            1,
            1,
            &verifier,
            &format!("00{}", "7c".repeat(31)),
            &mut publication,
            |_| Ok(evidenced_verification_for_test()),
        )
        .unwrap();
        assert_eq!(blocks.len(), ARTIFACT_AUDIT_BLOCKS);
        assert!(blocks.iter().all(|block| !block.material_win));
        assert!(!artifact_acceptance_decision(&blocks).accepted);
        assert_eq!(
            std::fs::read_dir(publication.stage_dir.join("verification"))
                .unwrap()
                .count(),
            2 * ARTIFACT_AUDIT_BLOCKS
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // one control spans evidence conversion and commit order
    fn production_execution_evidence_and_manifest_last_publication_are_exercised() {
        let verifier = verifier_for_test();
        let execution = execution_for_test(ArtifactAuditRole::Candidate, &verifier);
        let mut bad = evidenced_verification_for_test();
        bad.isolation.work_root_inode += 1;
        assert!(artifact_execution(ArtifactAuditRole::Candidate, bad, 1, 1, &verifier).is_err());
        assert!(artifact_execution(
            ArtifactAuditRole::Candidate,
            evidenced_verification_for_test(),
            2,
            1,
            &verifier,
        )
        .is_err());

        let tmp = TestDir::new("production-publication");
        let output = tmp.0.join("artifact");
        let mut publication = ArtifactPublication::claim(&output, &"62".repeat(32)).unwrap();
        publication
            .stage_text(Path::new("submission.py"), "candidate")
            .unwrap();
        stage_artifact_execution(&mut publication, 0, &execution).unwrap();
        publication
            .stage_text(Path::new("manifest.json"), "manifest")
            .unwrap();
        publication.publish_manifest_last().unwrap();
        assert_eq!(
            std::fs::read_to_string(output.join("submission.py")).unwrap(),
            "candidate"
        );
        assert_eq!(
            std::fs::read_to_string(output.join("manifest.json")).unwrap(),
            "manifest"
        );
        assert!(output
            .join("verification/block-000-candidate.json")
            .is_file());

        let issued = PublishedArtifact {
            output: output.clone(),
            manifest_path: output.join("manifest.json"),
            accepted: true,
        };
        assert_eq!(issued.output(), output);
        assert_eq!(issued.manifest_path(), output.join("manifest.json"));
        assert!(issued.accepted());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // table-like negative controls share one setup
    fn artifact_input_validation_and_io_fail_closed() {
        assert!(validate_note("note", "clean").is_ok());
        for value in ["", " leading", "trailing ", "line\nbreak"] {
            assert!(validate_note("note", value).is_err());
        }
        assert!(validate_device_token("GPU-abcd").is_ok());
        for value in ["", "0,1", " 0", "0 ", "GPU 0"] {
            assert!(validate_device_token(value).is_err());
        }
        assert!(validate_full_git_commit(&"aa".repeat(20)));
        assert!(!validate_full_git_commit("short"));
        assert!(valid_sha256(&"bb".repeat(32)));
        assert!(!valid_sha256("BB"));
        assert!(valid_lower_hex(&"cc".repeat(8), 8));

        let tmp = TestDir::new("io");
        let path = tmp.0.join("new");
        write_new_synced(&path, b"bytes").unwrap();
        assert_eq!(read_bytes(&path).unwrap(), b"bytes");
        assert!(write_new_synced(&path, b"replacement").is_err());
        assert!(read_bytes(&tmp.0.join("missing")).is_err());
        assert!(ArtifactDirectoryIdentity::capture(&path).is_err());
    }
}
