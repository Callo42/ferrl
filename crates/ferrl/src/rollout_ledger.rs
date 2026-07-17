//! Atomic rollout-window artifacts for separated collection and learning.
//!
//! A rollout ledger is deliberately different from the human-facing candidate
//! ledger: one published artifact contains every group required for one optimizer
//! window. The learner can therefore reject a torn, stale, or mismatched window
//! before it scores tokens or mutates policy/optimizer state.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::grpo::{group_advantages, ScaleRewards};

/// The only rollout-ledger format this release accepts.
pub const ROLLOUT_LEDGER_FORMAT_VERSION: u32 = 3;

const PAYLOAD_FILE: &str = "window.json";
const MANIFEST_FILE: &str = "manifest.json";

#[cfg(test)]
thread_local! {
    static POST_MANIFEST_TEST_FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static SYNCED_DIRECTORIES: std::cell::RefCell<Vec<PathBuf>> = const { std::cell::RefCell::new(Vec::new()) };
    static FAIL_SYNC_DIRECTORY_ONCE: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn inject_post_manifest_failure_once() {
    POST_MANIFEST_TEST_FAULT.with(|fault| fault.set(1));
}

#[cfg(test)]
pub(crate) fn inject_persistent_post_manifest_sync_failure_once() {
    POST_MANIFEST_TEST_FAULT.with(|fault| fault.set(2));
}

#[cfg(test)]
pub(crate) fn inject_post_manifest_disappearance_once() {
    POST_MANIFEST_TEST_FAULT.with(|fault| fault.set(3));
}

#[cfg(test)]
fn inject_post_manifest_in_place_mutation_once() {
    POST_MANIFEST_TEST_FAULT.with(|fault| fault.set(4));
}

/// A rollout-ledger read, write, identity, or semantic validation failure.
#[derive(Debug, thiserror::Error)]
pub enum RolloutLedgerError {
    /// A filesystem operation failed.
    #[error("rollout ledger I/O error at {path}: {source}")]
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// Strict JSON serialization or deserialization failed.
    #[error("rollout ledger JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// The artifact exists already; rollout windows are immutable.
    #[error("rollout ledger window already exists: {0}")]
    AlreadyExists(PathBuf),
    /// Publication crossed the reader-visible manifest boundary, but the exact
    /// visible bytes could not be reconciled after a later durability failure.
    #[error("rollout ledger publication is visible but ambiguous at {path}: {detail}")]
    PublicationAmbiguous {
        /// Reader-visible destination that must not be treated as uncommitted.
        path: PathBuf,
        /// Original publication/reconciliation failure.
        detail: String,
    },
    /// The manifest belongs to a different learner pre-state.
    #[error("rollout ledger identity mismatch")]
    IdentityMismatch,
    /// The payload declares controls different from the learner's resolved values.
    #[error("rollout ledger learner controls mismatch")]
    LearnerControlsMismatch,
    /// The artifact is corrupt, unsupported, or semantically inconsistent.
    #[error("invalid rollout ledger: {0}")]
    Invalid(String),
}

impl RolloutLedgerError {
    fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Whether the manifest commit marker may already be reader-visible.
    #[must_use]
    pub fn may_be_visible(&self) -> bool {
        matches!(self, Self::PublicationAmbiguous { .. })
    }
}

/// Exact learner pre-state that a rollout window is allowed to update.
///
/// Every digest is lowercase SHA-256 hex. `policy_sha256` identifies the frozen
/// base/model execution recipe; `adapter_sha256` identifies the current trainable
/// values; `tensor_schema_sha256` binds their ordered names/shapes/dtypes; and
/// `optimizer_sha256` binds Adam moments independently of `optimizer_step`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutLedgerIdentity {
    /// Canonical learner-semantic `TrainerConfig` projection digest. Operational
    /// run horizon, checkpoint cadence, candidate logging, and GPU probing are
    /// excluded because they do not change one validated learner update.
    pub trainer_config_sha256: String,
    /// Frozen base-policy content/configuration digest.
    pub policy_sha256: String,
    /// Ordered trainable tensor schema digest.
    pub tensor_schema_sha256: String,
    /// Exact pre-update adapter-value digest.
    pub adapter_sha256: String,
    /// Exact pre-update optimizer-state digest.
    pub optimizer_sha256: String,
    /// Exact pre-collection opaque sampler-state digest.
    pub sampler_sha256: String,
    /// Exact chain lineage represented by the pre-step continuation.
    pub lineage_sha256: String,
    /// Outer trainer step that produced this window.
    pub source_step: u64,
    /// Adam update counter before consuming this window.
    pub optimizer_step: u64,
}

/// Exact learner controls that must agree with a rollout window.
///
/// The opaque trainer-configuration digest binds controls not represented by the
/// ledger. These structured values separately prevent a checksum-valid payload
/// from declaring different rollout/update semantics under the same digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutLedgerControls {
    /// Number of ordered prompt groups in the optimizer window.
    pub grad_accum_steps: u32,
    /// Number of completions in each prompt group.
    pub group_size: u32,
    /// Rectangular completion width in every group.
    pub completion_width: u32,
    /// Reward-to-advantage scaling rule.
    pub scale_rewards: ScaleRewards,
    /// EOS token used for completion and truncation semantics.
    pub eos_token_id: Option<u32>,
    /// Whether full-width non-EOS completions are wholly masked.
    pub truncation_masking: bool,
    /// Effective TIS cap, or `None` when TIS is disabled.
    pub tis_imp_ratio_cap_bits: Option<u64>,
    /// Resolved learning rate as exact f64 bits.
    pub effective_lr_bits: u64,
    /// Resolved KL coefficient as exact f64 bits.
    pub effective_beta_bits: u64,
}

/// Mandatory learner pre-state and structured controls for reading a window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutLedgerExpectations {
    /// Exact learner/model/optimizer identity expected by the consumer.
    pub identity: RolloutLedgerIdentity,
    /// Exact structured rollout and update controls expected by the consumer.
    pub controls: RolloutLedgerControls,
}

/// Detached scoring operation the learner must perform before its first update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LedgerScoreRequirement {
    /// Score once with the trainable adapter enabled and detach the result.
    AdapterEnabledDetached,
    /// Score once with the adapter disabled and detach the result.
    AdapterDisabledDetached,
    /// No score of this kind is permitted or required.
    NotRequired,
}

/// One prompt group's host-side inputs inside an optimizer window.
///
/// Float fields use IEEE-754 bit patterns so the JSON wire is exact and cannot
/// sanitize `NaN`/infinity silently. Validation rejects non-finite logprobs and
/// advantages; rewards retain the trainer's explicit non-finite hardening.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutLedgerGroup {
    /// Zero-based position inside the accumulation window.
    pub accum_index: u32,
    /// Global prompt ordinal: `step * grad_accum_steps + accum_index` in v2.
    pub prompt_index: u64,
    /// Rectangular rows: prompt tokens followed by padded completion tokens.
    pub token_ids: Vec<Vec<u32>>,
    /// Number of leading prompt tokens shared by this group's rows.
    pub prompt_len: u32,
    /// Real EOS-inclusive completion length for every row.
    pub completion_lens: Vec<u32>,
    /// Optional draw-time behavior log-probability bits, ragged to real lengths.
    pub behavior_logprob_bits: Option<Vec<Vec<u32>>>,
    /// One exact reward bit pattern per completion.
    ///
    /// Non-finite rewards are preserved because the trainer deliberately excludes
    /// them from group statistics and assigns them zero advantage.
    pub reward_bits: Vec<u32>,
    /// Learner constants derived from rewards, stored as exact finite f32 bits.
    pub advantage_bits: Vec<u32>,
    /// Exact final loss mask (`0` or `1`) with shape `[group, completion_width]`.
    pub loss_mask: Vec<Vec<u8>>,
}

/// One complete world-1 optimizer window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutLedgerStep {
    /// Outer trainer step represented by this artifact.
    pub step: u64,
    /// Execution rank; v2 requires `0`.
    pub rank: u32,
    /// Execution world size; v2 requires `1`.
    pub world_size: u32,
    /// Expected number of ordered prompt groups.
    pub grad_accum_steps: u32,
    /// Expected number of completions per group.
    pub group_size: u32,
    /// Rectangular completion width in every group.
    pub completion_width: u32,
    /// Reward-to-advantage scaling rule used by the collector.
    pub scale_rewards: ScaleRewards,
    /// EOS token used to derive truncation masking, if any.
    pub eos_token_id: Option<u32>,
    /// Whether full-width non-EOS completions are wholly masked.
    pub truncation_masking: bool,
    /// Effective TIS cap as exact finite f64 bits, or `None` when TIS is disabled.
    pub tis_imp_ratio_cap_bits: Option<u64>,
    /// Resolved learning rate as exact finite f64 bits.
    pub effective_lr_bits: u64,
    /// Resolved KL coefficient as exact finite f64 bits.
    pub effective_beta_bits: u64,
    /// DAPO denominator: the sum of real completion lengths, clamped to at least `1`.
    pub window_tokens: u64,
    /// Number of groups that must enter the learner update.
    pub live_items: u32,
    /// Required detached old-policy scoring contract.
    pub old_logprobs: LedgerScoreRequirement,
    /// Required detached reference-policy scoring contract.
    pub reference_logprobs: LedgerScoreRequirement,
    /// Exact opaque sampler state after the collector produced this window.
    pub post_rollout_sampler_state: Vec<u8>,
    /// Every prompt group in accumulation order.
    pub groups: Vec<RolloutLedgerGroup>,
}

/// A rollout window whose bytes, identity, version, and semantics were verified.
///
/// Construct this only through [`RolloutLedgerReader::read_step`]. Learner entry
/// points should accept this wrapper rather than a raw [`RolloutLedgerStep`].
#[derive(Debug, Clone)]
pub struct ValidatedRolloutLedgerStep {
    identity: RolloutLedgerIdentity,
    step: RolloutLedgerStep,
}

impl ValidatedRolloutLedgerStep {
    /// Borrow the exact learner pre-state identity validated with this window.
    #[must_use]
    pub fn identity(&self) -> &RolloutLedgerIdentity {
        &self.identity
    }

    /// Borrow the validated window payload.
    #[must_use]
    pub fn as_step(&self) -> &RolloutLedgerStep {
        &self.step
    }

    /// Consume the validation wrapper and return its window payload.
    #[must_use]
    pub fn into_step(self) -> RolloutLedgerStep {
        self.step
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutLedgerManifest {
    format_version: u32,
    identity: RolloutLedgerIdentity,
    payload_file: String,
    payload_len: u64,
    payload_sha256: String,
}

/// Atomically publishes immutable rollout-window packages beneath one root.
#[derive(Debug, Clone)]
pub struct RolloutLedgerWriter {
    root: PathBuf,
    identity: RolloutLedgerIdentity,
    #[cfg(test)]
    fail_after_manifest_link: bool,
}

impl RolloutLedgerWriter {
    /// Create a writer bound to one exact learner pre-state identity.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] if an identity digest is malformed or the
    /// ledger root cannot be created.
    pub fn create(
        root: impl Into<PathBuf>,
        identity: RolloutLedgerIdentity,
    ) -> Result<Self, RolloutLedgerError> {
        validate_identity(&identity)?;
        let root = root.into();
        create_dir_all_durable(&root)?;
        Ok(Self {
            root,
            identity,
            #[cfg(test)]
            fail_after_manifest_link: false,
        })
    }

    /// Stage and atomically publish one complete optimizer window.
    ///
    /// The payload is synced first and the manifest is written/synced last inside
    /// a sibling staging directory. An atomic final-directory claim prevents
    /// replacement; the complete staged manifest is then hard-linked last as the
    /// reader-visible commit marker. Once that link succeeds, every required
    /// directory sync is retried before success can be reported; a persistent
    /// durability failure remains ambiguous and must not rewind collector state.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] for invalid payloads, serialization/I/O
    /// failures, or an attempt to overwrite an existing window.
    pub fn write_step(&self, step: &RolloutLedgerStep) -> Result<PathBuf, RolloutLedgerError> {
        validate_step(step)?;
        if step.step != self.identity.source_step {
            return Err(RolloutLedgerError::Invalid(format!(
                "payload step {} does not match identity source_step {}",
                step.step, self.identity.source_step
            )));
        }
        let final_dir = self.root.join(step_dir_name(step.step));
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let staging = self.root.join(format!(
            ".tmp-{}-{}-{nonce}",
            step_dir_name(step.step),
            std::process::id()
        ));
        fs::create_dir(&staging).map_err(|e| RolloutLedgerError::io(&staging, e))?;
        let result = self
            .write_staged(step, &staging)
            .and_then(|()| self.publish_staged(&staging, &final_dir));
        if staging.exists() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    fn write_staged(
        &self,
        step: &RolloutLedgerStep,
        staging: &Path,
    ) -> Result<(), RolloutLedgerError> {
        let payload = serde_json::to_vec(step)?;
        let payload_path = staging.join(PAYLOAD_FILE);
        write_new_synced(&payload_path, &payload)?;
        let manifest = RolloutLedgerManifest {
            format_version: ROLLOUT_LEDGER_FORMAT_VERSION,
            identity: self.identity.clone(),
            payload_file: PAYLOAD_FILE.to_string(),
            payload_len: u64::try_from(payload.len()).map_err(|_| {
                RolloutLedgerError::Invalid("payload length does not fit u64".into())
            })?,
            payload_sha256: sha256_hex(&payload),
        };
        let mut manifest_bytes = serde_json::to_vec(&manifest)?;
        manifest_bytes.push(b'\n');
        write_new_synced(&staging.join(MANIFEST_FILE), &manifest_bytes)?;
        sync_dir(staging)
    }

    fn publish_staged(
        &self,
        staging: &Path,
        final_dir: &Path,
    ) -> Result<PathBuf, RolloutLedgerError> {
        let expected_payload = fs::read(staging.join(PAYLOAD_FILE))
            .map_err(|error| RolloutLedgerError::io(staging.join(PAYLOAD_FILE), error))?;
        let expected_manifest = fs::read(staging.join(MANIFEST_FILE))
            .map_err(|error| RolloutLedgerError::io(staging.join(MANIFEST_FILE), error))?;
        match fs::create_dir(final_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(RolloutLedgerError::AlreadyExists(final_dir.to_path_buf()));
            }
            Err(error) => return Err(RolloutLedgerError::io(final_dir, error)),
        }
        let mut manifest_linked = false;
        let result = (|| {
            let payload = final_dir.join(PAYLOAD_FILE);
            fs::hard_link(staging.join(PAYLOAD_FILE), &payload)
                .map_err(|error| RolloutLedgerError::io(&payload, error))?;
            sync_dir(final_dir)?;

            let manifest = final_dir.join(MANIFEST_FILE);
            fs::hard_link(staging.join(MANIFEST_FILE), &manifest)
                .map_err(|error| RolloutLedgerError::io(&manifest, error))?;
            manifest_linked = true;
            #[cfg(test)]
            if self.fail_after_manifest_link {
                return Err(RolloutLedgerError::io(
                    final_dir,
                    std::io::Error::other("injected post-manifest publication failure"),
                ));
            }
            #[cfg(test)]
            POST_MANIFEST_TEST_FAULT.with(|fault| match fault.get() {
                0 => Ok(()),
                1 => {
                    fault.set(0);
                    Err(RolloutLedgerError::io(
                        final_dir,
                        std::io::Error::other("injected transient post-manifest failure"),
                    ))
                }
                2 => Err(RolloutLedgerError::io(
                    final_dir,
                    std::io::Error::other("injected persistent post-manifest sync failure"),
                )),
                3 => {
                    fault.set(0);
                    let _ = fs::remove_file(&manifest);
                    Err(RolloutLedgerError::io(
                        final_dir,
                        std::io::Error::other("injected post-link manifest disappearance"),
                    ))
                }
                4 => {
                    fault.set(0);
                    let mut file = OpenOptions::new()
                        .write(true)
                        .open(&payload)
                        .map_err(|error| RolloutLedgerError::io(&payload, error))?;
                    file.write_all(b"x")
                        .map_err(|error| RolloutLedgerError::io(&payload, error))?;
                    file.sync_all()
                        .map_err(|error| RolloutLedgerError::io(&payload, error))?;
                    Err(RolloutLedgerError::io(
                        final_dir,
                        std::io::Error::other("injected in-place post-link mutation"),
                    ))
                }
                other => panic!("unknown post-manifest test fault {other}"),
            })?;
            sync_dir(final_dir)?;
            sync_dir(&self.root)?;
            Ok(final_dir.to_path_buf())
        })();
        if let Err(error) = result {
            if !manifest_linked {
                let _ = fs::remove_dir_all(final_dir);
                return Err(error);
            }
            let durability = (|| {
                #[cfg(test)]
                POST_MANIFEST_TEST_FAULT.with(|fault| {
                    if fault.replace(0) == 2 {
                        return Err(RolloutLedgerError::io(
                            final_dir,
                            std::io::Error::other("injected persistent post-manifest sync failure"),
                        ));
                    }
                    Ok(())
                })?;
                sync_dir(final_dir)?;
                sync_dir(&self.root)
            })();
            let exact_visible = [
                (PAYLOAD_FILE, expected_payload.as_slice()),
                (MANIFEST_FILE, expected_manifest.as_slice()),
            ]
            .into_iter()
            .all(|(name, expected)| {
                fs::read(final_dir.join(name)).is_ok_and(|actual| actual == expected)
            });
            if durability.is_ok() && exact_visible {
                // The manifest boundary was crossed, and all required directory
                // syncs have now completed. Only this establishes success.
                return Ok(final_dir.to_path_buf());
            }
            return Err(RolloutLedgerError::PublicationAmbiguous {
                path: final_dir.to_path_buf(),
                detail: match durability {
                    Ok(()) => format!("{error}; visible package no longer matches staged bytes"),
                    Err(sync_error) => format!("{error}; durability retry failed: {sync_error}"),
                },
            });
        }
        Ok(final_dir.to_path_buf())
    }

    /// The ledger root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Reads only rollout windows matching one exact learner pre-state identity.
#[derive(Debug, Clone)]
pub struct RolloutLedgerReader {
    root: PathBuf,
    expected: RolloutLedgerExpectations,
}

impl RolloutLedgerReader {
    /// Open a reader bound to the identity and resolved controls the learner holds.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError::Invalid`] if an expected digest or control is
    /// malformed.
    pub fn open(
        root: impl Into<PathBuf>,
        expected: RolloutLedgerExpectations,
    ) -> Result<Self, RolloutLedgerError> {
        validate_identity(&expected.identity)?;
        validate_controls(&expected.controls)?;
        Ok(Self {
            root: root.into(),
            expected,
        })
    }

    /// Read and fully validate one published optimizer window.
    ///
    /// This performs all file, version, checksum, identity, and semantic checks
    /// before returning a value a learner is allowed to consume.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] if the package is absent, partial, corrupt,
    /// unsupported, mismatched, or semantically malformed.
    pub fn read_step(&self, step: u64) -> Result<ValidatedRolloutLedgerStep, RolloutLedgerError> {
        let dir = self.root.join(step_dir_name(step));
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_bytes =
            fs::read(&manifest_path).map_err(|e| RolloutLedgerError::io(&manifest_path, e))?;
        let manifest: RolloutLedgerManifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.format_version != ROLLOUT_LEDGER_FORMAT_VERSION {
            return Err(RolloutLedgerError::Invalid(format!(
                "unsupported format version {} (expected {})",
                manifest.format_version, ROLLOUT_LEDGER_FORMAT_VERSION
            )));
        }
        if manifest.identity != self.expected.identity {
            return Err(RolloutLedgerError::IdentityMismatch);
        }
        if manifest.payload_file != PAYLOAD_FILE {
            return Err(RolloutLedgerError::Invalid(format!(
                "unexpected payload file {:?}",
                manifest.payload_file
            )));
        }
        let payload_path = dir.join(PAYLOAD_FILE);
        let payload =
            fs::read(&payload_path).map_err(|e| RolloutLedgerError::io(&payload_path, e))?;
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| RolloutLedgerError::Invalid("payload length does not fit u64".into()))?;
        if payload_len != manifest.payload_len {
            return Err(RolloutLedgerError::Invalid(format!(
                "payload length {} does not match manifest {}",
                payload.len(),
                manifest.payload_len
            )));
        }
        let actual = sha256_hex(&payload);
        if actual != manifest.payload_sha256 {
            return Err(RolloutLedgerError::Invalid(
                "payload checksum mismatch".into(),
            ));
        }
        let payload: RolloutLedgerStep = serde_json::from_slice(&payload)?;
        if payload.step != step || payload.step != manifest.identity.source_step {
            return Err(RolloutLedgerError::Invalid(format!(
                "payload step {} does not match requested/source step {step}",
                payload.step
            )));
        }
        validate_step(&payload)?;
        let actual_controls = controls_from_step(&payload);
        if actual_controls != self.expected.controls {
            return Err(RolloutLedgerError::LearnerControlsMismatch);
        }
        Ok(ValidatedRolloutLedgerStep {
            identity: manifest.identity,
            step: payload,
        })
    }

    /// The ledger root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn validate_identity(identity: &RolloutLedgerIdentity) -> Result<(), RolloutLedgerError> {
    for (label, digest) in [
        ("trainer_config_sha256", &identity.trainer_config_sha256),
        ("policy_sha256", &identity.policy_sha256),
        ("tensor_schema_sha256", &identity.tensor_schema_sha256),
        ("adapter_sha256", &identity.adapter_sha256),
        ("optimizer_sha256", &identity.optimizer_sha256),
        ("sampler_sha256", &identity.sampler_sha256),
        ("lineage_sha256", &identity.lineage_sha256),
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(RolloutLedgerError::Invalid(format!(
                "{label} must be 64 lowercase hexadecimal characters"
            )));
        }
    }
    Ok(())
}

fn validate_controls(controls: &RolloutLedgerControls) -> Result<(), RolloutLedgerError> {
    if controls.grad_accum_steps == 0 || controls.group_size == 0 || controls.completion_width == 0
    {
        return Err(RolloutLedgerError::Invalid(
            "grad_accum_steps, group_size, and completion_width must be positive".into(),
        ));
    }
    let lr = f64::from_bits(controls.effective_lr_bits);
    let beta = f64::from_bits(controls.effective_beta_bits);
    if !lr.is_finite() || lr < 0.0 || !beta.is_finite() || beta < 0.0 {
        return Err(RolloutLedgerError::Invalid(
            "effective learning rate and beta must be finite and nonnegative".into(),
        ));
    }
    if let Some(bits) = controls.tis_imp_ratio_cap_bits {
        let cap = f64::from_bits(bits);
        if !cap.is_finite() || cap < 1.0 {
            return Err(RolloutLedgerError::Invalid(
                "enabled TIS requires a finite importance-ratio cap >= 1".into(),
            ));
        }
    }
    Ok(())
}

fn controls_from_step(step: &RolloutLedgerStep) -> RolloutLedgerControls {
    RolloutLedgerControls {
        grad_accum_steps: step.grad_accum_steps,
        group_size: step.group_size,
        completion_width: step.completion_width,
        scale_rewards: step.scale_rewards,
        eos_token_id: step.eos_token_id,
        truncation_masking: step.truncation_masking,
        tis_imp_ratio_cap_bits: step.tis_imp_ratio_cap_bits,
        effective_lr_bits: step.effective_lr_bits,
        effective_beta_bits: step.effective_beta_bits,
    }
}

#[allow(clippy::cognitive_complexity)] // ordered whole-window preflight is clearest in one pass
fn validate_step(step: &RolloutLedgerStep) -> Result<(), RolloutLedgerError> {
    if step.world_size != 1 || step.rank != 0 {
        return Err(RolloutLedgerError::Invalid(format!(
            "format v3 is world-1 only (got rank {}/world {})",
            step.rank, step.world_size
        )));
    }
    let controls = controls_from_step(step);
    validate_controls(&controls)?;
    let beta = f64::from_bits(step.effective_beta_bits);
    let expected_ref = if beta > 0.0 {
        LedgerScoreRequirement::AdapterDisabledDetached
    } else {
        LedgerScoreRequirement::NotRequired
    };
    if step.reference_logprobs != expected_ref {
        return Err(RolloutLedgerError::Invalid(
            "reference scoring requirement does not match effective beta".into(),
        ));
    }
    let accum = usize::try_from(step.grad_accum_steps)
        .map_err(|_| RolloutLedgerError::Invalid("grad_accum_steps overflows usize".into()))?;
    let group_size = usize::try_from(step.group_size)
        .map_err(|_| RolloutLedgerError::Invalid("group_size overflows usize".into()))?;
    let width = usize::try_from(step.completion_width)
        .map_err(|_| RolloutLedgerError::Invalid("completion_width overflows usize".into()))?;
    if step.groups.len() != accum {
        return Err(RolloutLedgerError::Invalid(format!(
            "expected {accum} groups, found {}",
            step.groups.len()
        )));
    }
    let prompt_base = step
        .step
        .checked_mul(u64::from(step.grad_accum_steps))
        .ok_or_else(|| RolloutLedgerError::Invalid("prompt ordinal overflow".into()))?;
    let mut token_total = 0_u64;
    let mut live = 0_u32;
    for (index, group) in step.groups.iter().enumerate() {
        let expected_index = u32::try_from(index)
            .map_err(|_| RolloutLedgerError::Invalid("group index overflows u32".into()))?;
        if group.accum_index != expected_index {
            return Err(RolloutLedgerError::Invalid(format!(
                "group {index} has accum_index {}",
                group.accum_index
            )));
        }
        let expected_prompt = prompt_base
            .checked_add(u64::from(group.accum_index))
            .ok_or_else(|| RolloutLedgerError::Invalid("prompt ordinal overflow".into()))?;
        if group.prompt_index != expected_prompt {
            return Err(RolloutLedgerError::Invalid(format!(
                "group {index} has prompt_index {}, expected {expected_prompt}",
                group.prompt_index
            )));
        }
        let surrogate_live = validate_group(step, group, group_size, width)?;
        token_total = group
            .completion_lens
            .iter()
            .try_fold(token_total, |acc, &len| {
                acc.checked_add(u64::from(len)).ok_or_else(|| {
                    RolloutLedgerError::Invalid("window token count overflow".into())
                })
            })?;
        if beta > 0.0 || surrogate_live {
            live = live
                .checked_add(1)
                .ok_or_else(|| RolloutLedgerError::Invalid("live item count overflow".into()))?;
        }
    }
    let expected_window_tokens = token_total.max(1);
    if expected_window_tokens != step.window_tokens {
        return Err(RolloutLedgerError::Invalid(format!(
            "window_tokens {} does not match clamped completion total {expected_window_tokens}",
            step.window_tokens
        )));
    }
    if live != step.live_items {
        return Err(RolloutLedgerError::Invalid(format!(
            "live_items {} does not match derived count {live}",
            step.live_items
        )));
    }
    let expected_old = if live > 0 {
        LedgerScoreRequirement::AdapterEnabledDetached
    } else {
        LedgerScoreRequirement::NotRequired
    };
    if step.old_logprobs != expected_old {
        return Err(RolloutLedgerError::Invalid(
            "old scoring requirement does not match the derived live-item count".into(),
        ));
    }
    Ok(())
}

fn validate_group(
    step: &RolloutLedgerStep,
    group: &RolloutLedgerGroup,
    group_size: usize,
    width: usize,
) -> Result<bool, RolloutLedgerError> {
    let prompt_len = usize::try_from(group.prompt_len)
        .map_err(|_| RolloutLedgerError::Invalid("prompt_len overflows usize".into()))?;
    if prompt_len == 0 {
        return Err(RolloutLedgerError::Invalid(
            "prompt_len must be positive".into(),
        ));
    }
    for (label, len) in [
        ("token rows", group.token_ids.len()),
        ("completion_lens", group.completion_lens.len()),
        ("rewards", group.reward_bits.len()),
        ("advantages", group.advantage_bits.len()),
        ("mask rows", group.loss_mask.len()),
    ] {
        if len != group_size {
            return Err(RolloutLedgerError::Invalid(format!(
                "{label} has {len} entries, expected {group_size}"
            )));
        }
    }
    let seq_len = prompt_len
        .checked_add(width)
        .ok_or_else(|| RolloutLedgerError::Invalid("sequence length overflow".into()))?;
    for (row, tokens) in group.token_ids.iter().enumerate() {
        if tokens.len() != seq_len {
            return Err(RolloutLedgerError::Invalid(format!(
                "token row {row} has length {}, expected {seq_len}",
                tokens.len()
            )));
        }
    }
    let prompt_prefix = &group.token_ids[0][..prompt_len];
    let rewards: Vec<f64> = group
        .reward_bits
        .iter()
        .map(|&bits| f64::from(f32::from_bits(bits)))
        .collect();
    let expected_advantages = group_advantages(&rewards, step.scale_rewards);
    for (row, &expected_advantage) in expected_advantages.iter().enumerate() {
        validate_group_row(
            step,
            group,
            row,
            prompt_len,
            width,
            prompt_prefix,
            expected_advantage,
        )?;
    }
    validate_behavior_capture(step, group, group_size)?;
    Ok(expected_advantages
        .iter()
        .any(|&advantage| advantage != 0.0))
}

#[allow(clippy::too_many_arguments)] // the row contract is clearer with named source fields
fn validate_group_row(
    step: &RolloutLedgerStep,
    group: &RolloutLedgerGroup,
    row: usize,
    prompt_len: usize,
    width: usize,
    prompt_prefix: &[u32],
    expected_advantage: f64,
) -> Result<(), RolloutLedgerError> {
    let tokens = &group.token_ids[row];
    if &tokens[..prompt_len] != prompt_prefix {
        return Err(RolloutLedgerError::Invalid(format!(
            "token row {row} does not share the group's prompt prefix"
        )));
    }
    let real = usize::try_from(group.completion_lens[row]).map_err(|_| {
        RolloutLedgerError::Invalid(format!("completion length row {row} overflows usize"))
    })?;
    if real > width {
        return Err(RolloutLedgerError::Invalid(format!(
            "completion length row {row} exceeds width {width}"
        )));
    }
    validate_completion_tokens(step, row, &tokens[prompt_len..], real)?;
    let advantage = finite_f32(group.advantage_bits[row], &format!("advantage row {row}"))?;
    if advantage.to_bits() != (expected_advantage as f32).to_bits() {
        return Err(RolloutLedgerError::Invalid(format!(
            "advantage row {row} does not match rewards"
        )));
    }
    let mask = &group.loss_mask[row];
    if mask.len() != width || mask.iter().any(|&value| value > 1) {
        return Err(RolloutLedgerError::Invalid(format!(
            "mask row {row} must contain {width} binary entries"
        )));
    }
    let truncated = step.truncation_masking
        && real == width
        && step
            .eos_token_id
            .is_some_and(|eos| tokens[tokens.len() - 1] != eos);
    for (column, &value) in mask.iter().enumerate() {
        let expected = u8::from(!truncated && column < real);
        if value != expected {
            return Err(RolloutLedgerError::Invalid(format!(
                "mask row {row} column {column} does not match length/EOS contract"
            )));
        }
    }
    Ok(())
}

fn validate_completion_tokens(
    step: &RolloutLedgerStep,
    row: usize,
    completion: &[u32],
    real: usize,
) -> Result<(), RolloutLedgerError> {
    let Some(eos) = step.eos_token_id else {
        if real != completion.len() {
            return Err(RolloutLedgerError::Invalid(format!(
                "completion row {row} is shortened without an EOS token"
            )));
        }
        return Ok(());
    };
    if real == 0 {
        if completion.iter().any(|&token| token != eos) {
            return Err(RolloutLedgerError::Invalid(format!(
                "zero-length completion row {row} is not entirely EOS padding"
            )));
        }
        return Ok(());
    }
    let first_eos = completion.iter().position(|&token| token == eos);
    match first_eos {
        Some(index) if index + 1 != real => Err(RolloutLedgerError::Invalid(format!(
            "completion row {row} length does not end at its first EOS token"
        ))),
        Some(_) if completion[real..].iter().any(|&token| token != eos) => {
            Err(RolloutLedgerError::Invalid(format!(
                "completion row {row} padding after EOS is not EOS-filled"
            )))
        }
        None if real != completion.len() => Err(RolloutLedgerError::Invalid(format!(
            "completion row {row} is shortened without sampling EOS"
        ))),
        Some(_) | None => Ok(()),
    }
}

fn validate_behavior_capture(
    step: &RolloutLedgerStep,
    group: &RolloutLedgerGroup,
    group_size: usize,
) -> Result<(), RolloutLedgerError> {
    match &group.behavior_logprob_bits {
        None if step.tis_imp_ratio_cap_bits.is_some() => {
            return Err(RolloutLedgerError::Invalid(
                "TIS requires behavior logprobs for every group".into(),
            ));
        }
        None => {}
        Some(rows) => {
            if rows.len() != group_size {
                return Err(RolloutLedgerError::Invalid(format!(
                    "behavior logprobs have {} rows, expected {group_size}",
                    rows.len()
                )));
            }
            for (row, values) in rows.iter().enumerate() {
                let expected = usize::try_from(group.completion_lens[row]).map_err(|_| {
                    RolloutLedgerError::Invalid(format!(
                        "behavior completion length row {row} overflows usize"
                    ))
                })?;
                if values.len() != expected {
                    return Err(RolloutLedgerError::Invalid(format!(
                        "behavior row {row} has {} entries, expected {expected}",
                        values.len()
                    )));
                }
                for (column, &bits) in values.iter().enumerate() {
                    let value = finite_f32(bits, &format!("behavior row {row} column {column}"))?;
                    if value > 0.0 {
                        return Err(RolloutLedgerError::Invalid(format!(
                            "behavior row {row} column {column} is a positive logprob"
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

fn finite_f32(bits: u32, label: &str) -> Result<f32, RolloutLedgerError> {
    let value = f32::from_bits(bits);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RolloutLedgerError::Invalid(format!(
            "{label} must be finite"
        )))
    }
}

fn step_dir_name(step: u64) -> String {
    format!("step-{step:020}")
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), RolloutLedgerError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| RolloutLedgerError::io(path, e))?;
    file.write_all(bytes)
        .map_err(|e| RolloutLedgerError::io(path, e))?;
    file.sync_all().map_err(|e| RolloutLedgerError::io(path, e))
}

fn sync_dir(path: &Path) -> Result<(), RolloutLedgerError> {
    // `Path::parent()` represents the ancestor of a one-component relative
    // path as `""`. Opening that path is not portable; its durable ancestor is
    // the current directory.
    let path = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    #[cfg(test)]
    FAIL_SYNC_DIRECTORY_ONCE.with(|failure| {
        if failure.borrow().as_deref() == Some(path) {
            failure.borrow_mut().take();
            return Err(RolloutLedgerError::io(
                path,
                std::io::Error::other("injected directory sync failure"),
            ));
        }
        Ok(())
    })?;
    let dir = File::open(path).map_err(|e| RolloutLedgerError::io(path, e))?;
    dir.sync_all()
        .map_err(|e| RolloutLedgerError::io(path, e))?;
    #[cfg(test)]
    SYNCED_DIRECTORIES.with(|paths| paths.borrow_mut().push(path.to_path_buf()));
    Ok(())
}

fn create_dir_all_durable(path: &Path) -> Result<(), RolloutLedgerError> {
    if path.as_os_str().is_empty() {
        return sync_dir(Path::new("."));
    }
    let Some(parent) = path.parent() else {
        return match fs::metadata(path) {
            Ok(metadata) if metadata.is_dir() => sync_dir(path),
            Ok(_) => Err(RolloutLedgerError::io(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "ledger ancestor exists but is not a directory",
                ),
            )),
            Err(error) => Err(RolloutLedgerError::io(path, error)),
        };
    };
    create_dir_all_durable(parent)?;
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => {
            return Err(RolloutLedgerError::io(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "ledger root exists but is not a directory",
                ),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => match fs::create_dir(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && path.is_dir() => {}
            Err(error) => return Err(RolloutLedgerError::io(path, error)),
        },
        Err(error) => return Err(RolloutLedgerError::io(path, error)),
    }
    sync_dir(path)?;
    sync_dir(parent)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!(
                "ferrl-rollout-ledger-{tag}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn relative(tag: &str, create: bool) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = PathBuf::from(format!(
                ".ferrl-rollout-ledger-{tag}-{}-{nonce}",
                std::process::id()
            ));
            assert!(path.is_relative());
            assert!(!path.exists());
            if create {
                fs::create_dir(&path).unwrap();
            }
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn identity() -> RolloutLedgerIdentity {
        RolloutLedgerIdentity {
            trainer_config_sha256: digest('a'),
            policy_sha256: digest('b'),
            tensor_schema_sha256: digest('c'),
            adapter_sha256: digest('d'),
            optimizer_sha256: digest('e'),
            sampler_sha256: digest('f'),
            lineage_sha256: digest('1'),
            source_step: 7,
            optimizer_step: 3,
        }
    }

    fn expectations() -> RolloutLedgerExpectations {
        RolloutLedgerExpectations {
            identity: identity(),
            controls: controls_from_step(&step()),
        }
    }

    fn group() -> RolloutLedgerGroup {
        let rewards = vec![1.0_f32, 3.0];
        let advantages = group_advantages(
            &rewards.iter().copied().map(f64::from).collect::<Vec<_>>(),
            ScaleRewards::Group,
        );
        RolloutLedgerGroup {
            accum_index: 0,
            prompt_index: 7,
            token_ids: vec![vec![5, 6, 9, 9], vec![5, 7, 8, 3]],
            prompt_len: 1,
            completion_lens: vec![2, 3],
            behavior_logprob_bits: Some(vec![
                vec![(-0.2_f32).to_bits(), (-0.3_f32).to_bits()],
                vec![
                    (-0.4_f32).to_bits(),
                    (-0.5_f32).to_bits(),
                    (-0.6_f32).to_bits(),
                ],
            ]),
            reward_bits: rewards.into_iter().map(f32::to_bits).collect(),
            advantage_bits: advantages
                .into_iter()
                .map(|value| (value as f32).to_bits())
                .collect(),
            loss_mask: vec![vec![1, 1, 0], vec![0, 0, 0]],
        }
    }

    fn step() -> RolloutLedgerStep {
        RolloutLedgerStep {
            step: 7,
            rank: 0,
            world_size: 1,
            grad_accum_steps: 1,
            group_size: 2,
            completion_width: 3,
            scale_rewards: ScaleRewards::Group,
            eos_token_id: Some(9),
            truncation_masking: true,
            tis_imp_ratio_cap_bits: Some(2.0_f64.to_bits()),
            effective_lr_bits: 1e-5_f64.to_bits(),
            effective_beta_bits: 0.1_f64.to_bits(),
            window_tokens: 5,
            live_items: 1,
            old_logprobs: LedgerScoreRequirement::AdapterEnabledDetached,
            reference_logprobs: LedgerScoreRequirement::AdapterDisabledDetached,
            post_rollout_sampler_state: vec![1, 2, 3, 4],
            groups: vec![group()],
        }
    }

    fn degenerate_zero_token_step() -> RolloutLedgerStep {
        let mut value = step();
        value.effective_beta_bits = 0.0_f64.to_bits();
        value.old_logprobs = LedgerScoreRequirement::NotRequired;
        value.reference_logprobs = LedgerScoreRequirement::NotRequired;
        value.window_tokens = 1;
        value.live_items = 0;
        value.groups[0].completion_lens = vec![0, 0];
        value.groups[0].token_ids = vec![vec![5, 9, 9, 9]; 2];
        value.groups[0].behavior_logprob_bits = Some(vec![Vec::new(), Vec::new()]);
        value.groups[0].reward_bits = vec![1.0_f32.to_bits(); 2];
        value.groups[0].advantage_bits = vec![0.0_f32.to_bits(); 2];
        value.groups[0].loss_mask = vec![vec![0; 3]; 2];
        value
    }

    fn rewrite_payload(root: &Path, mutate: impl FnOnce(&mut RolloutLedgerStep)) {
        let dir = root.join(step_dir_name(7));
        let payload_path = dir.join(PAYLOAD_FILE);
        let mut payload: RolloutLedgerStep =
            serde_json::from_slice(&fs::read(&payload_path).unwrap()).unwrap();
        mutate(&mut payload);
        let bytes = serde_json::to_vec(&payload).unwrap();
        fs::write(&payload_path, &bytes).unwrap();
        let manifest_path = dir.join(MANIFEST_FILE);
        let mut manifest: RolloutLedgerManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest.payload_len = u64::try_from(bytes.len()).unwrap();
        manifest.payload_sha256 = sha256_hex(&bytes);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    #[test]
    fn atomic_round_trip_preserves_exact_window() {
        let tmp = TempDir::new("roundtrip");
        let expected = step();
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let published = writer.write_step(&expected).unwrap();
        assert_eq!(published.file_name().unwrap(), step_dir_name(7).as_str());
        assert!(published.join(MANIFEST_FILE).is_file());
        let reader = RolloutLedgerReader::open(&tmp.0, expectations()).unwrap();
        assert_eq!(reader.read_step(7).unwrap().as_step(), &expected);
    }

    #[test]
    fn absent_ledger_root_is_durably_anchored_in_its_parent() {
        let tmp = TempDir::new("absent-ledger-root");
        let root = tmp.0.join("ledger");
        SYNCED_DIRECTORIES.with(|paths| paths.borrow_mut().clear());

        RolloutLedgerWriter::create(&root, identity()).unwrap();

        let synced = SYNCED_DIRECTORIES.with(|paths| paths.borrow().clone());
        assert!(
            synced.contains(&root),
            "ledger root was not synced: {synced:?}"
        );
        assert!(
            synced.contains(&tmp.0),
            "ledger-root entry was not synced in its parent: {synced:?}"
        );
    }

    #[test]
    fn absent_relative_ledger_root_uses_current_directory_as_durable_base() {
        let root = TempDir::relative("absent-relative-root", false);
        SYNCED_DIRECTORIES.with(|paths| paths.borrow_mut().clear());

        let writer = RolloutLedgerWriter::create(&root.0, identity()).unwrap();
        writer.write_step(&step()).unwrap();

        let synced = SYNCED_DIRECTORIES.with(|paths| paths.borrow().clone());
        assert!(synced.contains(&PathBuf::from(".")), "synced={synced:?}");
        assert!(synced.contains(&root.0), "synced={synced:?}");
        RolloutLedgerReader::open(&root.0, expectations())
            .unwrap()
            .read_step(7)
            .unwrap();
    }

    #[test]
    fn existing_relative_ledger_root_uses_current_directory_as_durable_base() {
        let root = TempDir::relative("existing-relative-root", true);
        SYNCED_DIRECTORIES.with(|paths| paths.borrow_mut().clear());

        let writer = RolloutLedgerWriter::create(&root.0, identity()).unwrap();
        writer.write_step(&step()).unwrap();

        let synced = SYNCED_DIRECTORIES.with(|paths| paths.borrow().clone());
        assert!(synced.contains(&PathBuf::from(".")), "synced={synced:?}");
        assert!(synced.contains(&root.0), "synced={synced:?}");
        RolloutLedgerReader::open(&root.0, expectations())
            .unwrap()
            .read_step(7)
            .unwrap();
    }

    #[test]
    fn ledger_root_creation_retries_a_failed_existing_directory_sync() {
        let tmp = TempDir::new("ledger-root-sync-retry");
        let root = tmp.0.join("ledger");
        FAIL_SYNC_DIRECTORY_ONCE.with(|failure| {
            *failure.borrow_mut() = Some(root.clone());
        });

        assert!(matches!(
            RolloutLedgerWriter::create(&root, identity()),
            Err(RolloutLedgerError::Io { .. })
        ));
        assert!(root.is_dir());

        RolloutLedgerWriter::create(&root, identity()).unwrap();
        RolloutLedgerWriter::create(&root, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
    }

    #[test]
    fn post_manifest_failure_reconciles_the_visible_exact_window_as_committed() {
        let tmp = TempDir::new("post-manifest-reconcile");
        let expected = step();
        let mut writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        writer.fail_after_manifest_link = true;

        let published = writer.write_step(&expected).unwrap();
        assert!(published.join(MANIFEST_FILE).is_file());
        assert_eq!(
            RolloutLedgerReader::open(&tmp.0, expectations())
                .unwrap()
                .read_step(7)
                .unwrap()
                .as_step(),
            &expected
        );
    }

    #[test]
    fn persistent_post_manifest_sync_failure_is_ambiguous_not_success() {
        let tmp = TempDir::new("persistent-post-manifest-sync");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        inject_persistent_post_manifest_sync_failure_once();

        assert!(matches!(
            writer.write_step(&step()),
            Err(RolloutLedgerError::PublicationAmbiguous { path, detail })
                if path == tmp.0.join(step_dir_name(7))
                    && detail.contains("durability retry failed")
        ));
        // The complete package remains visible, so the caller must preserve its
        // post-collection state despite the ambiguous durability outcome.
        RolloutLedgerReader::open(&tmp.0, expectations())
            .unwrap()
            .read_step(7)
            .unwrap();
    }

    #[test]
    fn post_link_manifest_disappearance_remains_ambiguous() {
        let tmp = TempDir::new("post-link-disappearance");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        inject_post_manifest_disappearance_once();

        assert!(matches!(
            writer.write_step(&step()),
            Err(RolloutLedgerError::PublicationAmbiguous { path, detail })
                if path == tmp.0.join(step_dir_name(7))
                    && detail.contains("no longer matches")
        ));
        assert!(!tmp.0.join(step_dir_name(7)).join(MANIFEST_FILE).exists());
        assert!(tmp.0.join(step_dir_name(7)).join(PAYLOAD_FILE).exists());
    }

    #[test]
    fn post_link_in_place_mutation_cannot_reconcile_against_hard_link_aliases() {
        let tmp = TempDir::new("post-link-in-place-mutation");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        inject_post_manifest_in_place_mutation_once();

        assert!(matches!(
            writer.write_step(&step()),
            Err(RolloutLedgerError::PublicationAmbiguous { path, detail })
                if path == tmp.0.join(step_dir_name(7))
                    && detail.contains("no longer matches")
        ));
    }

    #[test]
    fn reader_rejects_self_consistent_controls_that_mismatch_the_learner() {
        type ExpectationMutation = fn(&mut RolloutLedgerExpectations);
        let cases: Vec<ExpectationMutation> = vec![
            |expected| expected.controls.grad_accum_steps = 2,
            |expected| expected.controls.group_size = 3,
            |expected| expected.controls.completion_width = 4,
            |expected| expected.controls.scale_rewards = ScaleRewards::None,
            |expected| expected.controls.eos_token_id = None,
            |expected| expected.controls.truncation_masking = false,
            |expected| expected.controls.tis_imp_ratio_cap_bits = None,
            |expected| expected.controls.tis_imp_ratio_cap_bits = Some(3.0_f64.to_bits()),
            |expected| expected.controls.effective_lr_bits = 2e-5_f64.to_bits(),
            |expected| expected.controls.effective_beta_bits = 0.2_f64.to_bits(),
        ];
        let tmp = TempDir::new("learner-controls");
        RolloutLedgerWriter::create(&tmp.0, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
        for mutate in cases {
            let mut expected = expectations();
            mutate(&mut expected);
            assert!(matches!(
                RolloutLedgerReader::open(&tmp.0, expected)
                    .unwrap()
                    .read_step(7),
                Err(RolloutLedgerError::LearnerControlsMismatch)
            ));
        }
    }

    #[test]
    fn duplicate_window_is_rejected_without_overwrite() {
        let tmp = TempDir::new("duplicate");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let published = writer.write_step(&step()).unwrap();
        let manifest_before = fs::read(published.join(MANIFEST_FILE)).unwrap();
        let payload_before = fs::read(published.join(PAYLOAD_FILE)).unwrap();
        assert!(matches!(
            writer.write_step(&step()),
            Err(RolloutLedgerError::AlreadyExists(_))
        ));
        assert_eq!(
            fs::read(published.join(MANIFEST_FILE)).unwrap(),
            manifest_before
        );
        assert_eq!(
            fs::read(published.join(PAYLOAD_FILE)).unwrap(),
            payload_before
        );
    }

    #[test]
    fn empty_destination_claim_is_never_replaced() {
        let tmp = TempDir::new("empty-destination");
        let destination = tmp.0.join(step_dir_name(7));
        fs::create_dir(&destination).unwrap();
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        assert!(matches!(
            writer.write_step(&step()),
            Err(RolloutLedgerError::AlreadyExists(path)) if path == destination
        ));
        assert_eq!(fs::read_dir(&destination).unwrap().count(), 0);
        assert!(fs::read_dir(&tmp.0).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tmp-")
        }));
    }

    #[test]
    fn empty_destination_race_never_replaces_the_winner() {
        let tmp = TempDir::new("empty-destination-race");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let staging = tmp.0.join(".manual-staging");
        fs::create_dir(&staging).unwrap();
        writer.write_staged(&step(), &staging).unwrap();
        let destination = tmp.0.join(step_dir_name(7));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));

        let publish_barrier = barrier.clone();
        let publish_writer = writer.clone();
        let publish_staging = staging.clone();
        let publish_destination = destination.clone();
        let publisher = std::thread::spawn(move || {
            publish_barrier.wait();
            publish_writer.publish_staged(&publish_staging, &publish_destination)
        });
        let claim_barrier = barrier;
        let claim_destination = destination.clone();
        let claimer = std::thread::spawn(move || {
            claim_barrier.wait();
            fs::create_dir(claim_destination)
        });

        match (publisher.join().unwrap(), claimer.join().unwrap()) {
            (Ok(path), Err(error)) => {
                assert_eq!(path, destination);
                assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
                RolloutLedgerReader::open(&tmp.0, expectations())
                    .unwrap()
                    .read_step(7)
                    .unwrap();
            }
            (Err(RolloutLedgerError::AlreadyExists(path)), Ok(())) => {
                assert_eq!(path, destination);
                assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
            }
            (publish, claim) => panic!("unexpected publish/claim race: {publish:?} / {claim:?}"),
        }
    }

    #[test]
    fn same_length_payload_corruption_hits_the_checksum_gate() {
        let tmp = TempDir::new("checksum");
        let published = RolloutLedgerWriter::create(&tmp.0, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
        let payload_path = published.join(PAYLOAD_FILE);
        let mut payload = fs::read(&payload_path).unwrap();
        payload[0] ^= 1;
        fs::write(payload_path, payload).unwrap();
        assert!(matches!(
            RolloutLedgerReader::open(&tmp.0, expectations())
                .unwrap()
                .read_step(7),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("checksum")
        ));
    }

    #[test]
    fn invalid_window_is_rejected_before_any_staging_artifact() {
        let tmp = TempDir::new("preflight");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let mut invalid = step();
        invalid.world_size = 2;
        assert!(matches!(
            writer.write_step(&invalid),
            Err(RolloutLedgerError::Invalid(_))
        ));
        assert_eq!(fs::read_dir(&tmp.0).unwrap().count(), 0);
    }

    #[test]
    fn racing_publishers_leave_one_complete_window() {
        let tmp = TempDir::new("race");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let left = writer.clone();
        let right = writer;
        let left_step = step();
        let right_step = left_step.clone();
        let a = std::thread::spawn(move || left.write_step(&left_step));
        let b = std::thread::spawn(move || right.write_step(&right_step));
        let outcomes = [a.join().unwrap(), b.join().unwrap()];
        assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
        RolloutLedgerReader::open(&tmp.0, expectations())
            .unwrap()
            .read_step(7)
            .unwrap();
        assert!(fs::read_dir(&tmp.0).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tmp-")
        }));
    }

    #[test]
    fn zero_token_degenerate_window_uses_clamped_denominator_and_no_scoring() {
        let value = degenerate_zero_token_step();
        validate_step(&value).unwrap();

        let mut raw_zero = value.clone();
        raw_zero.window_tokens = 0;
        assert!(matches!(
            validate_step(&raw_zero),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("clamped")
        ));

        let mut unnecessary_old = value;
        unnecessary_old.old_logprobs = LedgerScoreRequirement::AdapterEnabledDetached;
        assert!(matches!(
            validate_step(&unnecessary_old),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("old scoring")
        ));
    }

    #[test]
    fn f64_nonzero_advantage_keeps_group_live_after_f32_rounding() {
        let mut value = step();
        value.scale_rewards = ScaleRewards::None;
        value.effective_beta_bits = 0.0_f64.to_bits();
        value.reference_logprobs = LedgerScoreRequirement::NotRequired;
        value.groups[0].reward_bits = vec![0.0_f32.to_bits(), f32::from_bits(1).to_bits()];
        let rewards = [0.0, f64::from(f32::from_bits(1))];
        let advantages = group_advantages(&rewards, ScaleRewards::None);
        assert!(advantages.iter().any(|&advantage| advantage != 0.0));
        value.groups[0].advantage_bits = advantages
            .iter()
            .map(|&advantage| (advantage as f32).to_bits())
            .collect();
        assert!(value.groups[0]
            .advantage_bits
            .iter()
            .all(|&bits| f32::from_bits(bits) == 0.0));
        value.live_items = 1;
        value.old_logprobs = LedgerScoreRequirement::AdapterEnabledDetached;
        validate_step(&value).unwrap();

        value.live_items = 0;
        value.old_logprobs = LedgerScoreRequirement::NotRequired;
        assert!(matches!(
            validate_step(&value),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("live_items")
        ));
    }

    #[test]
    fn prompt_ordinals_follow_world_one_window_order() {
        let mut value = step();
        value.grad_accum_steps = 2;
        let mut second = value.groups[0].clone();
        value.groups[0].accum_index = 0;
        value.groups[0].prompt_index = 14;
        second.accum_index = 1;
        second.prompt_index = 15;
        value.groups.push(second);
        value.window_tokens = 10;
        value.live_items = 2;
        validate_step(&value).unwrap();

        for ordinals in [[14, 14], [15, 14], [14, 16]] {
            let mut invalid = value.clone();
            invalid.groups[0].prompt_index = ordinals[0];
            invalid.groups[1].prompt_index = ordinals[1];
            assert!(matches!(
                validate_step(&invalid),
                Err(RolloutLedgerError::Invalid(message)) if message.contains("prompt_index")
            ));
        }
    }

    #[test]
    fn nonfinite_rewards_keep_the_trainer_zero_advantage_semantics() {
        for reward_bits in [
            vec![f32::NAN.to_bits(), 3.0_f32.to_bits()],
            vec![f32::NEG_INFINITY.to_bits(), f32::INFINITY.to_bits()],
        ] {
            let mut value = step();
            let rewards: Vec<f64> = reward_bits
                .iter()
                .map(|&bits| f64::from(f32::from_bits(bits)))
                .collect();
            value.groups[0].reward_bits = reward_bits;
            value.groups[0].advantage_bits = group_advantages(&rewards, value.scale_rewards)
                .into_iter()
                .map(|advantage| (advantage as f32).to_bits())
                .collect();
            validate_step(&value).unwrap();
        }
    }

    #[test]
    fn reader_rejects_wrong_identity() {
        let tmp = TempDir::new("wrong-identity");
        RolloutLedgerWriter::create(&tmp.0, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
        let mut wrong = expectations();
        wrong.identity.sampler_sha256 = digest('0');
        assert!(matches!(
            RolloutLedgerReader::open(&tmp.0, wrong)
                .unwrap()
                .read_step(7),
            Err(RolloutLedgerError::IdentityMismatch)
        ));
    }

    #[test]
    fn reader_rejects_wrong_format_version() {
        let tmp = TempDir::new("wrong-version");
        RolloutLedgerWriter::create(&tmp.0, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
        let manifest_path = tmp.0.join(step_dir_name(7)).join(MANIFEST_FILE);
        let mut manifest: RolloutLedgerManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest.format_version = 1;
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(matches!(
            RolloutLedgerReader::open(&tmp.0, expectations())
                .unwrap()
                .read_step(7),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("version")
        ));
    }

    #[test]
    fn reader_rejects_torn_payload() {
        let tmp = TempDir::new("torn-payload");
        RolloutLedgerWriter::create(&tmp.0, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
        let payload_path = tmp.0.join(step_dir_name(7)).join(PAYLOAD_FILE);
        let mut payload = fs::read(&payload_path).unwrap();
        payload.truncate(payload.len() / 2);
        fs::write(payload_path, payload).unwrap();
        assert!(matches!(
            RolloutLedgerReader::open(&tmp.0, expectations())
                .unwrap()
                .read_step(7),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("length")
        ));
    }

    #[test]
    fn semantic_validation_rejects_shape_mask_nonfinite_and_world_drift() {
        type StepMutation = fn(&mut RolloutLedgerStep);
        let cases: Vec<StepMutation> = vec![
            |value| value.world_size = 2,
            |value| {
                let _ = value.groups[0].token_ids[0].pop();
            },
            |value| value.groups[0].loss_mask[0][2] = 1,
            |value| value.groups[0].token_ids[1][0] = 6,
            |value| {
                value.groups[0].completion_lens[0] = 1;
                value.groups[0].behavior_logprob_bits.as_mut().unwrap()[0].truncate(1);
                value.groups[0].loss_mask[0] = vec![1, 0, 0];
            },
            |value| value.eos_token_id = None,
            |value| value.groups[0].token_ids[0][3] = 8,
            |value| {
                value.groups[0].behavior_logprob_bits.as_mut().unwrap()[0][0] = 0.1_f32.to_bits();
            },
            |value| {
                value.groups[0].behavior_logprob_bits.as_mut().unwrap()[0][0] = f32::NAN.to_bits();
            },
            |value| value.groups[0].advantage_bits[0] = 0.0_f32.to_bits(),
        ];
        for (index, mutate) in cases.into_iter().enumerate() {
            let tmp = TempDir::new(&format!("semantic-{index}"));
            RolloutLedgerWriter::create(&tmp.0, identity())
                .unwrap()
                .write_step(&step())
                .unwrap();
            rewrite_payload(&tmp.0, mutate);
            assert!(matches!(
                RolloutLedgerReader::open(&tmp.0, expectations())
                    .unwrap()
                    .read_step(7),
                Err(RolloutLedgerError::Invalid(_))
            ));
        }
    }

    #[test]
    fn strict_json_rejects_unknown_payload_fields() {
        let tmp = TempDir::new("unknown");
        RolloutLedgerWriter::create(&tmp.0, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
        let dir = tmp.0.join(step_dir_name(7));
        let payload_path = dir.join(PAYLOAD_FILE);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&payload_path).unwrap()).unwrap();
        value["future_semantics"] = serde_json::json!(true);
        let bytes = serde_json::to_vec(&value).unwrap();
        fs::write(&payload_path, &bytes).unwrap();
        let manifest_path = dir.join(MANIFEST_FILE);
        let mut manifest: RolloutLedgerManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest.payload_len = u64::try_from(bytes.len()).unwrap();
        manifest.payload_sha256 = sha256_hex(&bytes);
        fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(matches!(
            RolloutLedgerReader::open(&tmp.0, expectations())
                .unwrap()
                .read_step(7),
            Err(RolloutLedgerError::Json(_))
        ));
    }
}
