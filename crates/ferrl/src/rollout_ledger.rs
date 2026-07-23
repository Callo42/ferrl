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

use crate::grpo::{finite_moments, group_advantages, ScaleRewards, GROUP_STD_EPS};
use crate::policy::validate_completion_semantics;

/// The only rollout-ledger format this release accepts.
pub const ROLLOUT_LEDGER_FORMAT_VERSION: u32 = 6;

const PAYLOAD_FILE: &str = "window.json";
const MANIFEST_FILE: &str = "manifest.json";
const DISTRIBUTED_MANIFEST_KIND: &str = "distributed_rollout_ledger";
const DISTRIBUTED_OWNER_FILE: &str = ".ferrl-ledger-owner";

#[cfg(test)]
thread_local! {
    static POST_MANIFEST_TEST_FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static SYNCED_DIRECTORIES: std::cell::RefCell<Vec<PathBuf>> = const { std::cell::RefCell::new(Vec::new()) };
    static FAIL_SYNC_DIRECTORY_ONCE: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
    static FAIL_DISTRIBUTED_STAGE_CLEANUP_ONCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DISTRIBUTED_STAGE_CLEANUP_FAULTS_CONSUMED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static FAIL_DISTRIBUTED_RECONCILIATION_READ_ONCE: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
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

#[cfg(test)]
pub(crate) fn inject_post_manifest_panic_once() {
    POST_MANIFEST_TEST_FAULT.with(|fault| fault.set(5));
}

#[cfg(test)]
pub(crate) fn inject_directory_sync_failure_once(path: impl Into<PathBuf>) {
    FAIL_SYNC_DIRECTORY_ONCE.with(|failure| {
        *failure.borrow_mut() = Some(path.into());
    });
}

#[cfg(test)]
pub(crate) fn inject_distributed_stage_cleanup_failure_once() {
    DISTRIBUTED_STAGE_CLEANUP_FAULTS_CONSUMED.with(|count| count.set(0));
    FAIL_DISTRIBUTED_STAGE_CLEANUP_ONCE.with(|failure| failure.set(true));
}

#[cfg(test)]
pub(crate) fn distributed_stage_cleanup_faults_consumed() -> u32 {
    DISTRIBUTED_STAGE_CLEANUP_FAULTS_CONSUMED.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn inject_distributed_reconciliation_read_failure_once(path: impl Into<PathBuf>) {
    FAIL_DISTRIBUTED_RECONCILIATION_READ_ONCE.with(|failure| {
        *failure.borrow_mut() = Some(path.into());
    });
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
    /// Publication did not cross the reader-visible manifest boundary, but an
    /// owned staging/final claim could not be durably cleaned. Sampler rollback
    /// remains safe; an operator must reconcile the leftover before retry.
    #[error("rollout ledger unpublished claim requires reconciliation at {path}: {detail}")]
    UnpublishedClaimAmbiguous {
        /// Hidden stage or manifest-less destination blocking retry.
        path: PathBuf,
        /// Original failure plus the failed cleanup/ownership action.
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
    /// Whether reward normalization is rank-local or spans same-prompt shards.
    pub reward_group_scope: RolloutLedgerGroupScope,
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

/// How a distributed ledger forms each reward-normalization group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutLedgerGroupScope {
    /// Each rank's completions form an independent reward group.
    Local,
    /// The same accumulation position across every rank forms one reward group.
    DistributedSamePrompt,
}

/// Exact distributed reward statistics used to derive one shard's advantages.
///
/// Scalar collectives may associate floating-point additions differently from a
/// host-side replay. The ledger therefore records the rank-identical values the
/// collector actually received and validates every advantage against these bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutLedgerRewardStats {
    /// Number of rewards across the whole same-prompt group.
    pub count: u64,
    /// Exact f64 bits of the canonical rank-major reward mean.
    pub mean_bits: u64,
    /// Exact f64 bits of the canonical rank-major sample standard deviation.
    pub sample_std_bits: u64,
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
/// sanitize `NaN`/infinity silently. Validation rejects non-finite rewards,
/// logprobs, and advantages before learner scoring or mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutLedgerGroup {
    /// Zero-based position inside the accumulation window.
    pub accum_index: u32,
    /// Global prompt ordinal under the configured local/same-prompt DP topology.
    pub prompt_index: u64,
    /// Global first rollout-row ordinal used to derive this shard's RNG substreams.
    pub rollout_global_row_base: u64,
    /// Exact tokenizer encoding of the selected sample prompt, captured before
    /// policy generation and independently bound to every rollout-row prefix.
    pub prompt_token_ids: Vec<u32>,
    /// Rectangular rows: prompt tokens followed by padded completion tokens.
    pub token_ids: Vec<Vec<u32>>,
    /// Number of leading prompt tokens shared by this group's rows.
    pub prompt_len: u32,
    /// Real EOS-inclusive completion length for every row.
    pub completion_lens: Vec<u32>,
    /// Optional draw-time behavior log-probability bits, ragged to real lengths.
    pub behavior_logprob_bits: Option<Vec<Vec<u32>>>,
    /// One exact finite reward bit pattern per completion.
    pub reward_bits: Vec<u32>,
    /// Canonical cross-rank reward statistics in distributed same-prompt mode.
    /// Rank-local groups and every world-1 group carry `None`.
    pub distributed_reward_stats: Option<RolloutLedgerRewardStats>,
    /// Learner constants derived from rewards, stored as exact finite f32 bits.
    pub advantage_bits: Vec<u32>,
    /// Exact final loss mask (`0` or `1`) with shape `[group, completion_width]`.
    pub loss_mask: Vec<Vec<u8>>,
}

/// One rank's complete shard of an optimizer window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutLedgerStep {
    /// Outer trainer step represented by this artifact.
    pub step: u64,
    /// Execution rank in `0..world_size`.
    pub rank: u32,
    /// Data-parallel execution world size.
    pub world_size: u32,
    /// Expected number of ordered prompt groups.
    pub grad_accum_steps: u32,
    /// Expected number of completions per group.
    pub group_size: u32,
    /// Rectangular completion width in every group.
    pub completion_width: u32,
    /// Reward-normalization topology used by the collector.
    pub reward_group_scope: RolloutLedgerGroupScope,
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
    /// Global DAPO denominator: every rank's real completion lengths, clamped to `1`.
    pub window_tokens: u64,
    /// Global number of groups that must enter the learner update.
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
    consumed_ledger_sha256: String,
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

    /// Exact world package digest used to advance separated-training lineage.
    #[must_use]
    pub fn consumed_ledger_sha256(&self) -> &str {
        &self.consumed_ledger_sha256
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DistributedShardManifest {
    rank: u32,
    payload_file: String,
    payload_len: u64,
    payload_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DistributedRolloutLedgerManifest {
    format_version: u32,
    kind: String,
    identity: RolloutLedgerIdentity,
    step: u64,
    world_size: u32,
    controls: RolloutLedgerControls,
    window_tokens: u64,
    live_items: u32,
    shards: Vec<DistributedShardManifest>,
}

/// Rank-0-owned staging claim for one distributed optimizer window.
#[derive(Debug, Clone)]
pub struct DistributedRolloutLedgerStage {
    path: PathBuf,
    final_dir: PathBuf,
    owner: Vec<u8>,
}

impl DistributedRolloutLedgerStage {
    /// Private staging directory shared by the ranks until publication commits.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
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
    pub(crate) fn distributed_stage_path(&self, step: u64, nonce: u64) -> PathBuf {
        self.root.join(format!(
            ".tmp-{}-distributed-{nonce:016x}",
            step_dir_name(step)
        ))
    }

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
        if step.rank != 0 || step.world_size != 1 {
            return Err(RolloutLedgerError::Invalid(format!(
                "write_step requires a world-1 payload (got rank {}/world {})",
                step.rank, step.world_size
            )));
        }
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
                5 => {
                    fault.set(0);
                    panic!("injected post-manifest rollout-ledger publication panic")
                }
                other => panic!("unknown post-manifest test fault {other}"),
            })?;
            sync_dir(final_dir)?;
            sync_dir(&self.root)?;
            if !exact_files_visible(
                final_dir,
                &[
                    (PAYLOAD_FILE, expected_payload.as_slice()),
                    (MANIFEST_FILE, expected_manifest.as_slice()),
                ],
            ) {
                return Err(RolloutLedgerError::PublicationAmbiguous {
                    path: final_dir.to_path_buf(),
                    detail: "visible package no longer matches staged bytes after the initial durability fence"
                        .into(),
                });
            }
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
            let exact_visible = exact_files_visible(
                final_dir,
                &[
                    (PAYLOAD_FILE, expected_payload.as_slice()),
                    (MANIFEST_FILE, expected_manifest.as_slice()),
                ],
            );
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

    /// Create rank 0's private staging directory for one distributed window.
    ///
    /// Every rank must wait for the caller to coordinate this successful claim
    /// before writing its shard through [`write_distributed_shard`](Self::write_distributed_shard).
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] for an invalid topology, step mismatch, or
    /// failure to durably create the staging claim.
    pub fn create_distributed_stage(
        &self,
        step: u64,
        world_size: u32,
        nonce: u64,
    ) -> Result<DistributedRolloutLedgerStage, RolloutLedgerError> {
        if world_size <= 1 {
            return Err(RolloutLedgerError::Invalid(
                "distributed ledger staging requires world_size > 1".into(),
            ));
        }
        if step != self.identity.source_step {
            return Err(RolloutLedgerError::Invalid(format!(
                "distributed stage step {step} does not match identity source_step {}",
                self.identity.source_step
            )));
        }
        let path = self.distributed_stage_path(step, nonce);
        fs::create_dir(&path).map_err(|error| RolloutLedgerError::io(&path, error))?;
        let owner = format!(
            "{}:{:?}:{nonce:016x}",
            std::process::id(),
            std::thread::current().id()
        )
        .into_bytes();
        if let Err(error) = write_new_synced(&path.join(DISTRIBUTED_OWNER_FILE), &owner) {
            return Err(RolloutLedgerError::UnpublishedClaimAmbiguous {
                path,
                detail: format!(
                    "{error}; owner marker was not established, so the hidden stage was preserved"
                ),
            });
        }
        if let Err(error) = sync_dir(&path).and_then(|()| sync_dir(&self.root)) {
            return Err(cleanup_unpublished_stage(&self.root, &path, &owner, error));
        }
        Ok(DistributedRolloutLedgerStage {
            path,
            final_dir: self.root.join(step_dir_name(step)),
            owner,
        })
    }

    /// Write and sync this rank's immutable payload into a coordinated stage.
    ///
    /// The caller must globalize every rank's result before rank 0 either commits
    /// or cleans the stage; cleanup while a peer may still be writing is forbidden.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] for an invalid shard or an I/O failure.
    pub fn write_distributed_shard(
        &self,
        stage: impl AsRef<Path>,
        step: &RolloutLedgerStep,
    ) -> Result<PathBuf, RolloutLedgerError> {
        validate_step(step)?;
        if step.world_size <= 1 || step.rank >= step.world_size {
            return Err(RolloutLedgerError::Invalid(format!(
                "distributed shard has invalid rank {}/world {}",
                step.rank, step.world_size
            )));
        }
        if step.step != self.identity.source_step {
            return Err(RolloutLedgerError::Invalid(format!(
                "payload step {} does not match identity source_step {}",
                step.step, self.identity.source_step
            )));
        }
        let stage = stage.as_ref();
        let payload_path = stage.join(distributed_payload_name(step.rank));
        let payload = serde_json::to_vec(step)?;
        write_new_synced(&payload_path, &payload)?;
        Ok(payload_path)
    }

    /// Remove an uncommitted distributed stage after every rank has reported
    /// quiescence through the communicator.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] if ownership changed or cleanup cannot be
    /// durably anchored in the ledger root.
    pub fn abort_distributed_stage(
        &self,
        stage: &DistributedRolloutLedgerStage,
    ) -> Result<(), RolloutLedgerError> {
        #[cfg(test)]
        FAIL_DISTRIBUTED_STAGE_CLEANUP_ONCE.with(|failure| {
            if failure.replace(false) {
                DISTRIBUTED_STAGE_CLEANUP_FAULTS_CONSUMED
                    .with(|count| count.set(count.get().saturating_add(1)));
                return Err(RolloutLedgerError::io(
                    &stage.path,
                    std::io::Error::other("injected distributed hidden-stage cleanup failure"),
                ));
            }
            Ok(())
        })?;
        require_exact_owner(&stage.path, &stage.owner)?;
        fs::remove_dir_all(&stage.path)
            .map_err(|error| RolloutLedgerError::io(&stage.path, error))?;
        sync_dir(&self.root)
    }

    /// Classify an existing destination marker-first and reconcile it when the
    /// intended commit marker may be reader-visible. Hidden-stage cleanup cannot
    /// downgrade a durably verified publication, while payload uncertainty or an
    /// unresolved fence is publication ambiguity regardless of cleanup outcome.
    fn reconcile_existing_distributed_package(
        &self,
        stage: &DistributedRolloutLedgerStage,
        expected_files: &[(String, Vec<u8>)],
    ) -> Option<Result<PathBuf, RolloutLedgerError>> {
        match probe_existing_distributed_package(&stage.final_dir, expected_files) {
            ExistingDistributedPackageProbe::RetryableMarker => return None,
            ExistingDistributedPackageProbe::Exact => {}
            ExistingDistributedPackageProbe::Ambiguous(mut detail) => {
                if let Err(cleanup_error) = self.abort_distributed_stage(stage) {
                    detail.push_str(&format!(
                        "; hidden-stage cleanup also failed: {cleanup_error}"
                    ));
                }
                return Some(Err(RolloutLedgerError::PublicationAmbiguous {
                    path: stage.final_dir.clone(),
                    detail,
                }));
            }
        }

        let fences = sync_dir(&stage.final_dir).and_then(|()| sync_dir(&self.root));
        let post_fence = probe_existing_distributed_package(&stage.final_dir, expected_files);
        let cleanup = self.abort_distributed_stage(stage);
        if fences.is_ok() && matches!(&post_fence, ExistingDistributedPackageProbe::Exact) {
            return Some(Ok(stage.final_dir.clone()));
        }

        let mut detail = match (fences, post_fence) {
            (Ok(()), ExistingDistributedPackageProbe::RetryableMarker) => {
                "the exact distributed manifest changed or disappeared during post-fence verification"
                    .into()
            }
            (Ok(()), ExistingDistributedPackageProbe::Ambiguous(detail)) => detail,
            (Err(error), ExistingDistributedPackageProbe::Exact) => format!(
                "exact distributed package was already visible, but durability reconciliation failed: {error}"
            ),
            (Err(error), ExistingDistributedPackageProbe::RetryableMarker) => format!(
                "exact distributed package was already visible, but durability reconciliation failed: {error}; its manifest also changed or disappeared during post-fence verification"
            ),
            (Err(error), ExistingDistributedPackageProbe::Ambiguous(detail)) => format!(
                "exact distributed package was already visible, but durability reconciliation failed: {error}; post-fence verification was also ambiguous: {detail}"
            ),
            (Ok(()), ExistingDistributedPackageProbe::Exact) => {
                unreachable!("successful reconciliation returned above")
            }
        };
        if let Err(cleanup_error) = cleanup {
            detail.push_str(&format!(
                "; hidden-stage cleanup also failed: {cleanup_error}"
            ));
        }
        Some(Err(RolloutLedgerError::PublicationAmbiguous {
            path: stage.final_dir.clone(),
            detail,
        }))
    }

    /// Validate every staged shard and publish one manifest-last distributed step.
    ///
    /// The caller must invoke this only on rank 0 after a successful shard-status
    /// collective proves every peer has stopped writing the stage.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] for a missing/mismatched shard, conflicting
    /// existing step, failed durability fence, or ambiguous post-manifest state.
    #[cfg_attr(test, allow(clippy::missing_panics_doc))]
    pub fn commit_distributed_stage(
        &self,
        stage: &DistributedRolloutLedgerStage,
        step: u64,
        world_size: u32,
        controls: &RolloutLedgerControls,
    ) -> Result<PathBuf, RolloutLedgerError> {
        let prepared = (|| {
            validate_controls(controls)?;
            if world_size <= 1 || step != self.identity.source_step {
                return Err(RolloutLedgerError::Invalid(
                    "distributed commit topology does not match the writer identity".into(),
                ));
            }
            require_exact_owner(&stage.path, &stage.owner)?;
            let mut packages = Vec::with_capacity(world_size as usize);
            let mut shards = Vec::with_capacity(world_size as usize);
            for rank in 0..world_size {
                let name = distributed_payload_name(rank);
                let path = stage.path.join(&name);
                let bytes =
                    fs::read(&path).map_err(|error| RolloutLedgerError::io(&path, error))?;
                let payload: RolloutLedgerStep = serde_json::from_slice(&bytes)?;
                if payload.step != step || payload.rank != rank || payload.world_size != world_size
                {
                    return Err(RolloutLedgerError::Invalid(format!(
                        "distributed shard {rank} declares step/rank/world {}/{}/{}",
                        payload.step, payload.rank, payload.world_size
                    )));
                }
                if controls_from_step(&payload) != *controls {
                    return Err(RolloutLedgerError::LearnerControlsMismatch);
                }
                let validation = validate_step(&payload)?;
                let payload_len = u64::try_from(bytes.len()).map_err(|_| {
                    RolloutLedgerError::Invalid(
                        "distributed payload length does not fit u64".into(),
                    )
                })?;
                shards.push(DistributedShardManifest {
                    rank,
                    payload_file: name,
                    payload_len,
                    payload_sha256: sha256_hex(&bytes),
                });
                packages.push((payload, bytes, validation));
            }
            validate_distributed_packages(&packages, controls)?;
            let first = &packages[0].0;
            let manifest = DistributedRolloutLedgerManifest {
                format_version: ROLLOUT_LEDGER_FORMAT_VERSION,
                kind: DISTRIBUTED_MANIFEST_KIND.to_owned(),
                identity: self.identity.clone(),
                step,
                world_size,
                controls: controls.clone(),
                window_tokens: first.window_tokens,
                live_items: first.live_items,
                shards,
            };
            let mut manifest_bytes = serde_json::to_vec(&manifest)?;
            manifest_bytes.push(b'\n');
            let staged_manifest = stage.path.join(MANIFEST_FILE);
            write_new_synced(&staged_manifest, &manifest_bytes)?;
            sync_dir(&stage.path)?;
            let expected_files: Vec<(String, Vec<u8>)> = packages
                .iter()
                .map(|(payload, bytes, _)| (distributed_payload_name(payload.rank), bytes.clone()))
                .chain(std::iter::once((MANIFEST_FILE.to_owned(), manifest_bytes)))
                .collect();
            Ok((expected_files, staged_manifest))
        })();
        let (expected_files, staged_manifest) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                return Err(cleanup_unpublished_stage(
                    &self.root,
                    &stage.path,
                    &stage.owner,
                    error,
                ));
            }
        };
        if stage.final_dir.exists() {
            if let Some(reconciled) =
                self.reconcile_existing_distributed_package(stage, &expected_files)
            {
                return reconciled;
            }
            return Err(cleanup_unpublished_stage(
                &self.root,
                &stage.path,
                &stage.owner,
                RolloutLedgerError::AlreadyExists(stage.final_dir.clone()),
            ));
        }

        if let Err(source) = fs::create_dir(&stage.final_dir) {
            if let Some(reconciled) =
                self.reconcile_existing_distributed_package(stage, &expected_files)
            {
                return reconciled;
            }
            let error = if source.kind() == std::io::ErrorKind::AlreadyExists {
                RolloutLedgerError::AlreadyExists(stage.final_dir.clone())
            } else {
                RolloutLedgerError::io(&stage.final_dir, source)
            };
            return Err(cleanup_unpublished_stage(
                &self.root,
                &stage.path,
                &stage.owner,
                error,
            ));
        }
        if let Err(error) =
            write_new_synced(&stage.final_dir.join(DISTRIBUTED_OWNER_FILE), &stage.owner)
        {
            return Err(cleanup_distributed_claim(self, stage, error));
        }
        if let Err(error) = sync_dir(&stage.final_dir).and_then(|()| sync_dir(&self.root)) {
            return Err(cleanup_distributed_claim(self, stage, error));
        }

        let mut manifest_linked = false;
        let publication = (|| {
            for (name, _) in &expected_files {
                if name == MANIFEST_FILE {
                    continue;
                }
                let destination = stage.final_dir.join(name);
                fs::hard_link(stage.path.join(name), &destination)
                    .map_err(|error| RolloutLedgerError::io(&destination, error))?;
            }
            sync_dir(&stage.final_dir)?;
            let destination_manifest = stage.final_dir.join(MANIFEST_FILE);
            fs::hard_link(&staged_manifest, &destination_manifest)
                .map_err(|error| RolloutLedgerError::io(&destination_manifest, error))?;
            manifest_linked = true;
            #[cfg(test)]
            POST_MANIFEST_TEST_FAULT.with(|fault| match fault.get() {
                0 => Ok(()),
                1 => {
                    fault.set(0);
                    Err(RolloutLedgerError::io(
                        &stage.final_dir,
                        std::io::Error::other(
                            "injected transient distributed post-manifest failure",
                        ),
                    ))
                }
                2 => Err(RolloutLedgerError::io(
                    &stage.final_dir,
                    std::io::Error::other(
                        "injected persistent distributed post-manifest sync failure",
                    ),
                )),
                3 => {
                    fault.set(0);
                    let _ = fs::remove_file(&destination_manifest);
                    Err(RolloutLedgerError::io(
                        &stage.final_dir,
                        std::io::Error::other(
                            "injected distributed post-link manifest disappearance",
                        ),
                    ))
                }
                4 => {
                    fault.set(0);
                    let payload = stage.final_dir.join(distributed_payload_name(0));
                    let mut file = OpenOptions::new()
                        .write(true)
                        .open(&payload)
                        .map_err(|error| RolloutLedgerError::io(&payload, error))?;
                    file.write_all(b"x")
                        .map_err(|error| RolloutLedgerError::io(&payload, error))?;
                    file.sync_all()
                        .map_err(|error| RolloutLedgerError::io(&payload, error))?;
                    Err(RolloutLedgerError::io(
                        &stage.final_dir,
                        std::io::Error::other("injected distributed in-place post-link mutation"),
                    ))
                }
                5 => {
                    fault.set(0);
                    panic!("injected distributed post-manifest publication panic")
                }
                other => Err(RolloutLedgerError::Invalid(format!(
                    "unknown distributed post-manifest test fault {other}"
                ))),
            })?;
            sync_dir(&stage.final_dir)?;
            sync_dir(&self.root)?;
            if !exact_owned_distributed_package(
                &stage.final_dir,
                &expected_files,
                Some(&stage.owner),
            ) {
                return Err(RolloutLedgerError::PublicationAmbiguous {
                    path: stage.final_dir.clone(),
                    detail: "visible distributed package does not match the staged rank shards"
                        .into(),
                });
            }
            Ok(stage.final_dir.clone())
        })();

        match publication {
            Ok(path) => {
                let _ = fs::remove_dir_all(&stage.path);
                Ok(path)
            }
            Err(error) if !manifest_linked => Err(cleanup_distributed_claim(self, stage, error)),
            Err(error) => {
                let retry = (|| {
                    #[cfg(test)]
                    POST_MANIFEST_TEST_FAULT.with(|fault| {
                        if fault.replace(0) == 2 {
                            return Err(RolloutLedgerError::io(
                                &stage.final_dir,
                                std::io::Error::other(
                                    "injected persistent distributed post-manifest sync failure",
                                ),
                            ));
                        }
                        Ok(())
                    })?;
                    sync_dir(&stage.final_dir)?;
                    sync_dir(&self.root)
                })();
                let exact = exact_owned_distributed_package(
                    &stage.final_dir,
                    &expected_files,
                    Some(&stage.owner),
                );
                if retry.is_ok() && exact {
                    let _ = fs::remove_dir_all(&stage.path);
                    Ok(stage.final_dir.clone())
                } else {
                    Err(RolloutLedgerError::PublicationAmbiguous {
                        path: stage.final_dir.clone(),
                        detail: match retry {
                            Ok(()) => format!(
                                "{error}; visible distributed package no longer matches staged bytes"
                            ),
                            Err(sync_error) => {
                                format!("{error}; durability retry failed: {sync_error}")
                            }
                        },
                    })
                }
            }
        }
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
        if serde_json::from_slice::<DistributedRolloutLedgerManifest>(&manifest_bytes).is_ok() {
            return Err(RolloutLedgerError::Invalid(
                "distributed ledger requires read_distributed_step with explicit rank/world".into(),
            ));
        }
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
        let payload_bytes =
            fs::read(&payload_path).map_err(|e| RolloutLedgerError::io(&payload_path, e))?;
        let payload_len = u64::try_from(payload_bytes.len())
            .map_err(|_| RolloutLedgerError::Invalid("payload length does not fit u64".into()))?;
        if payload_len != manifest.payload_len {
            return Err(RolloutLedgerError::Invalid(format!(
                "payload length {} does not match manifest {}",
                payload_bytes.len(),
                manifest.payload_len
            )));
        }
        let actual = sha256_hex(&payload_bytes);
        if actual != manifest.payload_sha256 {
            return Err(RolloutLedgerError::Invalid(
                "payload checksum mismatch".into(),
            ));
        }
        let payload: RolloutLedgerStep = serde_json::from_slice(&payload_bytes)?;
        if payload.step != step || payload.step != manifest.identity.source_step {
            return Err(RolloutLedgerError::Invalid(format!(
                "payload step {} does not match requested/source step {step}",
                payload.step
            )));
        }
        validate_step(&payload)?;
        if payload.rank != 0 || payload.world_size != 1 {
            return Err(RolloutLedgerError::Invalid(format!(
                "world-1 package declares rank {}/world {}",
                payload.rank, payload.world_size
            )));
        }
        let actual_controls = controls_from_step(&payload);
        if actual_controls != self.expected.controls {
            return Err(RolloutLedgerError::LearnerControlsMismatch);
        }
        let consumed_ledger_sha256 = consumed_world_one_sha256(&manifest_bytes, &payload_bytes);
        Ok(ValidatedRolloutLedgerStep {
            identity: manifest.identity,
            step: payload,
            consumed_ledger_sha256,
        })
    }

    /// Read and validate every shard of one committed distributed optimizer window.
    ///
    /// The root manifest is the only commit marker. It must bind exactly one
    /// canonical shard for every rank, all global counts and reward statistics
    /// are checked before the caller receives this rank's payload, and every rank
    /// derives the same consumed-package digest from the exact manifest bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RolloutLedgerError`] for an absent/partial/corrupt package,
    /// topology or identity mismatch, or invalid cross-rank semantics.
    pub fn read_distributed_step(
        &self,
        step: u64,
        rank: u32,
        world_size: u32,
    ) -> Result<ValidatedRolloutLedgerStep, RolloutLedgerError> {
        if world_size <= 1 || rank >= world_size {
            return Err(RolloutLedgerError::Invalid(format!(
                "invalid distributed reader rank {rank}/world {world_size}"
            )));
        }
        let dir = self.root.join(step_dir_name(step));
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path)
            .map_err(|error| RolloutLedgerError::io(&manifest_path, error))?;
        let manifest: DistributedRolloutLedgerManifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.format_version != ROLLOUT_LEDGER_FORMAT_VERSION
            || manifest.kind != DISTRIBUTED_MANIFEST_KIND
        {
            return Err(RolloutLedgerError::Invalid(format!(
                "unsupported distributed rollout-ledger format {}/{}",
                manifest.kind, manifest.format_version
            )));
        }
        if manifest.identity != self.expected.identity {
            return Err(RolloutLedgerError::IdentityMismatch);
        }
        if manifest.controls != self.expected.controls {
            return Err(RolloutLedgerError::LearnerControlsMismatch);
        }
        if manifest.step != step
            || manifest.identity.source_step != step
            || manifest.world_size != world_size
        {
            return Err(RolloutLedgerError::Invalid(format!(
                "distributed manifest step/world {}/{} does not match requested {step}/{world_size}",
                manifest.step, manifest.world_size
            )));
        }
        if manifest.shards.len() != world_size as usize {
            return Err(RolloutLedgerError::Invalid(format!(
                "distributed manifest has {} shards, expected {world_size}",
                manifest.shards.len()
            )));
        }

        let mut packages = Vec::with_capacity(world_size as usize);
        for (index, shard) in manifest.shards.iter().enumerate() {
            let expected_rank = u32::try_from(index)
                .map_err(|_| RolloutLedgerError::Invalid("shard rank overflows u32".into()))?;
            let expected_name = distributed_payload_name(expected_rank);
            if shard.rank != expected_rank || shard.payload_file != expected_name {
                return Err(RolloutLedgerError::Invalid(format!(
                    "distributed shard {index} is not the canonical rank/file entry"
                )));
            }
            let path = dir.join(&shard.payload_file);
            let bytes = fs::read(&path).map_err(|error| RolloutLedgerError::io(&path, error))?;
            let len = u64::try_from(bytes.len()).map_err(|_| {
                RolloutLedgerError::Invalid("distributed payload length does not fit u64".into())
            })?;
            if len != shard.payload_len || sha256_hex(&bytes) != shard.payload_sha256 {
                return Err(RolloutLedgerError::Invalid(format!(
                    "distributed shard {expected_rank} length/checksum mismatch"
                )));
            }
            let payload: RolloutLedgerStep = serde_json::from_slice(&bytes)?;
            if payload.step != step
                || payload.rank != expected_rank
                || payload.world_size != world_size
                || controls_from_step(&payload) != manifest.controls
            {
                return Err(RolloutLedgerError::Invalid(format!(
                    "distributed shard {expected_rank} topology/controls mismatch"
                )));
            }
            let validation = validate_step(&payload)?;
            packages.push((payload, bytes, validation));
        }
        validate_distributed_packages(&packages, &manifest.controls)?;
        if packages[0].0.window_tokens != manifest.window_tokens
            || packages[0].0.live_items != manifest.live_items
        {
            return Err(RolloutLedgerError::Invalid(
                "distributed manifest global counts do not match its shards".into(),
            ));
        }
        let consumed_ledger_sha256 = consumed_distributed_sha256(
            &manifest_bytes,
            packages.iter().map(|(_, bytes, _)| bytes.as_slice()),
        );
        let payload = packages
            .into_iter()
            .nth(rank as usize)
            .ok_or_else(|| {
                RolloutLedgerError::Invalid(
                    "requested distributed rank is absent after package validation".into(),
                )
            })?
            .0;
        Ok(ValidatedRolloutLedgerStep {
            identity: manifest.identity,
            step: payload,
            consumed_ledger_sha256,
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
        reward_group_scope: step.reward_group_scope,
        scale_rewards: step.scale_rewards,
        eos_token_id: step.eos_token_id,
        truncation_masking: step.truncation_masking,
        tis_imp_ratio_cap_bits: step.tis_imp_ratio_cap_bits,
        effective_lr_bits: step.effective_lr_bits,
        effective_beta_bits: step.effective_beta_bits,
    }
}

#[derive(Debug, Clone, Copy)]
struct StepValidation {
    completion_tokens: u64,
    live_items: u32,
}

/// Run the exact reader/writer local-shard semantic validator before candidate
/// telemetry becomes visible during collection. The returned counts let the
/// distributed collector reproduce the package-level global-count check in
/// memory, before any ledger or candidate file is published.
pub(crate) fn validate_collected_step(
    step: &RolloutLedgerStep,
) -> Result<(u64, u32), RolloutLedgerError> {
    let validation = validate_step(step)?;
    Ok((validation.completion_tokens, validation.live_items))
}

#[allow(clippy::cognitive_complexity)] // ordered whole-window preflight is clearest in one pass
fn validate_step(step: &RolloutLedgerStep) -> Result<StepValidation, RolloutLedgerError> {
    if step.world_size == 0 || step.rank >= step.world_size {
        return Err(RolloutLedgerError::Invalid(format!(
            "invalid rollout-ledger rank {}/world {}",
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
    let accum_u64 = u64::from(step.grad_accum_steps);
    let world_u64 = u64::from(step.world_size);
    let rank_u64 = u64::from(step.rank);
    let prompt_base = match step.reward_group_scope {
        RolloutLedgerGroupScope::Local => step
            .step
            .checked_mul(accum_u64)
            .and_then(|value| value.checked_mul(world_u64))
            .and_then(|value| value.checked_add(rank_u64.checked_mul(accum_u64)?)),
        RolloutLedgerGroupScope::DistributedSamePrompt => step.step.checked_mul(accum_u64),
    }
    .ok_or_else(|| RolloutLedgerError::Invalid("prompt ordinal overflow".into()))?;
    let shard_base = step
        .step
        .checked_mul(accum_u64)
        .and_then(|value| value.checked_mul(world_u64))
        .and_then(|value| value.checked_add(rank_u64.checked_mul(accum_u64)?))
        .ok_or_else(|| RolloutLedgerError::Invalid("rollout row ordinal overflow".into()))?;
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
        let expected_row_base = shard_base
            .checked_add(u64::from(group.accum_index))
            .and_then(|value| value.checked_mul(u64::from(step.group_size)))
            .ok_or_else(|| RolloutLedgerError::Invalid("rollout row ordinal overflow".into()))?;
        if group.rollout_global_row_base != expected_row_base {
            return Err(RolloutLedgerError::Invalid(format!(
                "group {index} has rollout_global_row_base {}, expected {expected_row_base}",
                group.rollout_global_row_base
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
    if step.world_size == 1 {
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
    Ok(StepValidation {
        completion_tokens: token_total,
        live_items: live,
    })
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
    if group.prompt_token_ids.is_empty() {
        return Err(RolloutLedgerError::Invalid(
            "selected prompt token ids must be nonempty".into(),
        ));
    }
    if group.prompt_token_ids.len() != prompt_len {
        return Err(RolloutLedgerError::Invalid(format!(
            "prompt_len {prompt_len} does not match {} persisted selected prompt tokens",
            group.prompt_token_ids.len()
        )));
    }
    let prompt_prefix = group.prompt_token_ids.as_slice();
    let rewards: Vec<f64> = group
        .reward_bits
        .iter()
        .enumerate()
        .map(|(row, &bits)| finite_f32(bits, &format!("reward row {row}")).map(f64::from))
        .collect::<Result<_, _>>()?;
    let distributed_stats_required = step.world_size > 1
        && step.reward_group_scope == RolloutLedgerGroupScope::DistributedSamePrompt;
    let expected_advantages = match (distributed_stats_required, group.distributed_reward_stats) {
        (false, None) => group_advantages(&rewards, step.scale_rewards),
        (true, Some(stats)) => {
            advantages_from_distributed_stats(&rewards, stats, step.scale_rewards)?
        }
        (false, Some(_)) => {
            return Err(RolloutLedgerError::Invalid(
                "rank-local/world-1 group carries distributed reward statistics".into(),
            ));
        }
        (true, None) => {
            return Err(RolloutLedgerError::Invalid(
                "distributed same-prompt group is missing canonical reward statistics".into(),
            ));
        }
    };
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

fn advantages_from_distributed_stats(
    rewards: &[f64],
    stats: RolloutLedgerRewardStats,
    scale: ScaleRewards,
) -> Result<Vec<f64>, RolloutLedgerError> {
    let mean = f64::from_bits(stats.mean_bits);
    let sample_std = f64::from_bits(stats.sample_std_bits);
    if stats.count == 0 || !mean.is_finite() || !sample_std.is_finite() || sample_std < 0.0 {
        return Err(RolloutLedgerError::Invalid(
            "distributed reward moments require a positive count, finite mean, and finite nonnegative sample std"
                .into(),
        ));
    }
    let denominator = match scale {
        ScaleRewards::None => 1.0,
        ScaleRewards::Group => sample_std + GROUP_STD_EPS,
    };
    rewards
        .iter()
        .enumerate()
        .map(|(row, &reward)| {
            let advantage = (reward - mean) / denominator;
            if advantage.is_finite() {
                Ok(advantage)
            } else {
                Err(RolloutLedgerError::Invalid(format!(
                    "derived advantage row {row} must be finite"
                )))
            }
        })
        .collect()
}

fn validate_distributed_packages(
    packages: &[(RolloutLedgerStep, Vec<u8>, StepValidation)],
    controls: &RolloutLedgerControls,
) -> Result<(), RolloutLedgerError> {
    let first = packages.first().ok_or_else(|| {
        RolloutLedgerError::Invalid("distributed package contains no shards".into())
    })?;
    let world_size = first.0.world_size;
    if world_size <= 1 || packages.len() != world_size as usize {
        return Err(RolloutLedgerError::Invalid(
            "distributed shard count does not match world_size".into(),
        ));
    }
    let mut completion_tokens = 0_u64;
    let mut live_items = 0_u32;
    for (index, (payload, _, validation)) in packages.iter().enumerate() {
        let rank = u32::try_from(index)
            .map_err(|_| RolloutLedgerError::Invalid("distributed rank overflows u32".into()))?;
        if payload.rank != rank
            || payload.world_size != world_size
            || payload.step != first.0.step
            || controls_from_step(payload) != *controls
            || payload.window_tokens != first.0.window_tokens
            || payload.live_items != first.0.live_items
            || payload.post_rollout_sampler_state != first.0.post_rollout_sampler_state
        {
            return Err(RolloutLedgerError::Invalid(format!(
                "distributed shard {rank} disagrees on topology, controls, global counts, or sampler state"
            )));
        }
        completion_tokens = completion_tokens
            .checked_add(validation.completion_tokens)
            .ok_or_else(|| {
                RolloutLedgerError::Invalid("distributed completion-token total overflow".into())
            })?;
        live_items = live_items
            .checked_add(validation.live_items)
            .ok_or_else(|| {
                RolloutLedgerError::Invalid("distributed live-item total overflow".into())
            })?;
    }
    if first.0.window_tokens != completion_tokens.max(1) || first.0.live_items != live_items {
        return Err(RolloutLedgerError::Invalid(format!(
            "distributed global counts {}/{} do not match derived {}/{}",
            first.0.window_tokens,
            first.0.live_items,
            completion_tokens.max(1),
            live_items
        )));
    }

    if controls.reward_group_scope == RolloutLedgerGroupScope::DistributedSamePrompt {
        for group_index in 0..controls.grad_accum_steps as usize {
            let canonical = packages[0].0.groups[group_index]
                .distributed_reward_stats
                .ok_or_else(|| {
                    RolloutLedgerError::Invalid(
                        "distributed same-prompt group is missing reward statistics".into(),
                    )
                })?;
            let mut global_rewards = Vec::new();
            let canonical_prompt = &packages[0].0.groups[group_index].prompt_token_ids;
            for (payload, _, _) in packages {
                let group = &payload.groups[group_index];
                if group.distributed_reward_stats != Some(canonical) {
                    return Err(RolloutLedgerError::Invalid(format!(
                        "distributed reward statistics disagree at accumulation group {group_index}"
                    )));
                }
                if &group.prompt_token_ids != canonical_prompt {
                    return Err(RolloutLedgerError::Invalid(format!(
                        "distributed same-prompt prefixes disagree at accumulation group {group_index}"
                    )));
                }
                global_rewards.extend(
                    group
                        .reward_bits
                        .iter()
                        .map(|&bits| f64::from(f32::from_bits(bits))),
                );
            }
            let moments = finite_moments(&global_rewards);
            let expected_count = u64::try_from(moments.count()).map_err(|_| {
                RolloutLedgerError::Invalid("distributed reward count overflows u64".into())
            })?;
            if canonical.count != expected_count
                || canonical.mean_bits != moments.mean().to_bits()
                || canonical.sample_std_bits != moments.sample_std().to_bits()
            {
                return Err(RolloutLedgerError::Invalid(format!(
                    "distributed reward statistics do not match shard rewards at accumulation group {group_index}"
                )));
            }
        }
    }
    Ok(())
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
    let expected_narrowed = expected_advantage as f32;
    if !expected_advantage.is_finite()
        || !expected_narrowed.is_finite()
        || (expected_advantage != 0.0 && expected_narrowed == 0.0)
    {
        return Err(RolloutLedgerError::Invalid(format!(
            "derived advantage row {row} is not representable as a finite nonzero-preserving f32"
        )));
    }
    let advantage = finite_f32(group.advantage_bits[row], &format!("advantage row {row}"))?;
    if advantage.to_bits() != expected_narrowed.to_bits() {
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
    validate_completion_semantics(completion, real, step.eos_token_id)
        .map_err(|detail| RolloutLedgerError::Invalid(format!("completion row {row} {detail}")))
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

fn distributed_payload_name(rank: u32) -> String {
    format!("rank-{rank:05}.window.json")
}

fn exact_files_visible(dir: &Path, expected: &[(&str, &[u8])]) -> bool {
    expected.iter().all(|(name, bytes)| {
        fs::read(dir.join(name)).is_ok_and(|actual| actual.as_slice() == *bytes)
    })
}

enum ExistingDistributedPackageProbe {
    /// The marker is definitely absent or contains different readable bytes, so
    /// the destination cannot be the intended reader-visible publication.
    RetryableMarker,
    /// The intended marker and every named payload are exactly readable.
    Exact,
    /// The intended marker may be visible, or is exact while a payload is not
    /// exactly readable. Sampler rollback would split state from publication.
    Ambiguous(String),
}

fn read_distributed_reconciliation_file(path: &Path) -> std::io::Result<Vec<u8>> {
    #[cfg(test)]
    {
        let injected = FAIL_DISTRIBUTED_RECONCILIATION_READ_ONCE.with(|failure| {
            let matches = failure.borrow().as_deref() == Some(path);
            if matches {
                failure.borrow_mut().take();
            }
            matches
        });
        if injected {
            return Err(std::io::Error::other(
                "injected distributed reconciliation read failure",
            ));
        }
    }
    fs::read(path)
}

/// Probe the commit marker before any payload. Only a definitely absent or
/// different readable marker is retryable; once the exact intended marker is
/// visible, every payload mismatch, absence, or read error is publication
/// ambiguity rather than evidence that nothing was committed.
fn probe_existing_distributed_package(
    dir: &Path,
    expected: &[(String, Vec<u8>)],
) -> ExistingDistributedPackageProbe {
    let Some((_, expected_manifest)) = expected.iter().find(|(name, _)| name == MANIFEST_FILE)
    else {
        return ExistingDistributedPackageProbe::Ambiguous(
            "staged distributed package has no expected manifest bytes".into(),
        );
    };
    let manifest_path = dir.join(MANIFEST_FILE);
    match read_distributed_reconciliation_file(&manifest_path) {
        Ok(actual) if actual.as_slice() != expected_manifest.as_slice() => {
            return ExistingDistributedPackageProbe::RetryableMarker;
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return ExistingDistributedPackageProbe::RetryableMarker;
        }
        Err(error) => {
            return ExistingDistributedPackageProbe::Ambiguous(format!(
                "distributed manifest could not be classified as absent or different: {error}"
            ));
        }
    }

    for (name, expected_bytes) in expected {
        if name == MANIFEST_FILE {
            continue;
        }
        let path = dir.join(name);
        match read_distributed_reconciliation_file(&path) {
            Ok(actual) if actual.as_slice() == expected_bytes.as_slice() => {}
            Ok(_) => {
                return ExistingDistributedPackageProbe::Ambiguous(format!(
                    "the exact distributed manifest is visible, but payload {name} differs from the intended bytes"
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return ExistingDistributedPackageProbe::Ambiguous(format!(
                    "the exact distributed manifest is visible, but payload {name} is missing"
                ));
            }
            Err(error) => {
                return ExistingDistributedPackageProbe::Ambiguous(format!(
                    "the exact distributed manifest is visible, but payload {name} could not be read: {error}"
                ));
            }
        }
    }
    ExistingDistributedPackageProbe::Exact
}

fn exact_owned_distributed_package(
    dir: &Path,
    expected: &[(String, Vec<u8>)],
    owner: Option<&[u8]>,
) -> bool {
    owner.is_none_or(|owner| {
        fs::read(dir.join(DISTRIBUTED_OWNER_FILE)).is_ok_and(|actual| actual == owner)
    }) && expected.iter().all(|(name, bytes)| {
        fs::read(dir.join(name)).is_ok_and(|actual| actual.as_slice() == bytes.as_slice())
    })
}

fn require_exact_owner(dir: &Path, owner: &[u8]) -> Result<(), RolloutLedgerError> {
    match fs::read(dir.join(DISTRIBUTED_OWNER_FILE)) {
        Ok(actual) if actual == owner => Ok(()),
        Ok(_) => Err(RolloutLedgerError::UnpublishedClaimAmbiguous {
            path: dir.to_path_buf(),
            detail: "distributed staging ownership changed; the directory was preserved".into(),
        }),
        Err(error) => Err(RolloutLedgerError::UnpublishedClaimAmbiguous {
            path: dir.to_path_buf(),
            detail: format!(
                "distributed staging ownership could not be verified ({error}); the directory was preserved"
            ),
        }),
    }
}

fn cleanup_unpublished_stage(
    root: &Path,
    stage: &Path,
    owner: &[u8],
    original: RolloutLedgerError,
) -> RolloutLedgerError {
    let cleanup = (|| {
        require_exact_owner(stage, owner)?;
        fs::remove_dir_all(stage).map_err(|error| RolloutLedgerError::io(stage, error))?;
        sync_dir(root)
    })();
    match cleanup {
        Ok(()) => original,
        Err(cleanup_error) => RolloutLedgerError::UnpublishedClaimAmbiguous {
            path: stage.to_path_buf(),
            detail: format!("{original}; hidden-stage cleanup failed: {cleanup_error}"),
        },
    }
}

fn cleanup_distributed_claim(
    writer: &RolloutLedgerWriter,
    stage: &DistributedRolloutLedgerStage,
    original: RolloutLedgerError,
) -> RolloutLedgerError {
    let cleanup = (|| {
        require_exact_owner(&stage.final_dir, &stage.owner)?;
        fs::remove_dir_all(&stage.final_dir)
            .map_err(|error| RolloutLedgerError::io(&stage.final_dir, error))?;
        sync_dir(&writer.root)?;
        require_exact_owner(&stage.path, &stage.owner)?;
        fs::remove_dir_all(&stage.path)
            .map_err(|error| RolloutLedgerError::io(&stage.path, error))?;
        sync_dir(&writer.root)
    })();
    match cleanup {
        Ok(()) => original,
        Err(cleanup_error) => RolloutLedgerError::UnpublishedClaimAmbiguous {
            path: stage.final_dir.clone(),
            detail: format!("{original}; owned pre-manifest claim cleanup failed: {cleanup_error}"),
        },
    }
}

fn consumed_world_one_sha256(manifest: &[u8], payload: &[u8]) -> String {
    hash_ordered_parts("ferrl.rollout-ledger.consumed-step.v1", [manifest, payload])
}

fn consumed_distributed_sha256<'a>(
    manifest: &'a [u8],
    shards: impl IntoIterator<Item = &'a [u8]>,
) -> String {
    hash_ordered_parts(
        "ferrl.rollout-ledger.consumed-distributed-step.v1",
        std::iter::once(manifest).chain(shards),
    )
}

fn hash_ordered_parts<'a>(domain: &str, parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
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
            rollout_global_row_base: 14,
            prompt_token_ids: vec![5],
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
            distributed_reward_stats: None,
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
            reward_group_scope: RolloutLedgerGroupScope::Local,
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

    fn distributed_step(rank: u32, scope: RolloutLedgerGroupScope) -> RolloutLedgerStep {
        let mut value = step();
        value.rank = rank;
        value.world_size = 2;
        value.reward_group_scope = scope;
        value.window_tokens = 10;
        value.live_items = 2;
        value.groups[0].prompt_index = match scope {
            RolloutLedgerGroupScope::Local => 14 + u64::from(rank),
            RolloutLedgerGroupScope::DistributedSamePrompt => 7,
        };
        value.groups[0].rollout_global_row_base = (14 + u64::from(rank)) * 2;
        if scope == RolloutLedgerGroupScope::DistributedSamePrompt {
            let rewards = if rank == 0 {
                vec![1.0_f32, 2.0]
            } else {
                vec![3.0_f32, 4.0]
            };
            let global_rewards = [1.0_f64, 2.0, 3.0, 4.0];
            let moments = finite_moments(&global_rewards);
            let stats = RolloutLedgerRewardStats {
                count: 4,
                mean_bits: moments.mean().to_bits(),
                sample_std_bits: moments.sample_std().to_bits(),
            };
            value.groups[0].reward_bits = rewards.iter().copied().map(f32::to_bits).collect();
            value.groups[0].distributed_reward_stats = Some(stats);
            value.groups[0].advantage_bits = advantages_from_distributed_stats(
                &rewards.iter().copied().map(f64::from).collect::<Vec<_>>(),
                stats,
                value.scale_rewards,
            )
            .unwrap()
            .into_iter()
            .map(|advantage| (advantage as f32).to_bits())
            .collect();
        }
        value
    }

    fn publish_distributed(
        root: &Path,
        scope: RolloutLedgerGroupScope,
        nonce: u64,
    ) -> (PathBuf, [RolloutLedgerStep; 2]) {
        let writer = RolloutLedgerWriter::create(root, identity()).unwrap();
        let shards = [distributed_step(0, scope), distributed_step(1, scope)];
        let stage = writer.create_distributed_stage(7, 2, nonce).unwrap();
        for shard in &shards {
            writer.write_distributed_shard(stage.path(), shard).unwrap();
        }
        let path = writer
            .commit_distributed_stage(&stage, 7, 2, &controls_from_step(&shards[0]))
            .unwrap();
        (path, shards)
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

    fn rewrite_distributed_shard(
        root: &Path,
        rank: u32,
        mutate: impl FnOnce(&mut RolloutLedgerStep),
    ) {
        let dir = root.join(step_dir_name(7));
        let manifest_path = dir.join(MANIFEST_FILE);
        let mut manifest: DistributedRolloutLedgerManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        let shard = &mut manifest.shards[rank as usize];
        let payload_path = dir.join(&shard.payload_file);
        let mut payload: RolloutLedgerStep =
            serde_json::from_slice(&fs::read(&payload_path).unwrap()).unwrap();
        mutate(&mut payload);
        let payload_bytes = serde_json::to_vec(&payload).unwrap();
        fs::write(&payload_path, &payload_bytes).unwrap();
        shard.payload_len = payload_bytes.len() as u64;
        shard.payload_sha256 = sha256_hex(&payload_bytes);
        let mut manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        manifest_bytes.push(b'\n');
        fs::write(manifest_path, manifest_bytes).unwrap();
    }

    fn forge_distributed_reward_stats(root: &Path, stats: RolloutLedgerRewardStats) {
        for rank in 0..2 {
            rewrite_distributed_shard(root, rank, |value| {
                let scale = value.scale_rewards;
                let group = &mut value.groups[0];
                let rewards = group
                    .reward_bits
                    .iter()
                    .map(|&bits| f64::from(f32::from_bits(bits)))
                    .collect::<Vec<_>>();
                group.distributed_reward_stats = Some(stats);
                group.advantage_bits = advantages_from_distributed_stats(&rewards, stats, scale)
                    .unwrap()
                    .into_iter()
                    .map(|advantage| (advantage as f32).to_bits())
                    .collect();
            });
        }
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
            |expected| {
                expected.controls.reward_group_scope =
                    RolloutLedgerGroupScope::DistributedSamePrompt;
            },
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
    #[allow(clippy::cognitive_complexity)] // assertion-heavy two-scope package oracle
    fn distributed_manifest_roundtrip_validates_every_rank_shard() {
        for (index, scope) in [
            RolloutLedgerGroupScope::Local,
            RolloutLedgerGroupScope::DistributedSamePrompt,
        ]
        .into_iter()
        .enumerate()
        {
            let tmp = TempDir::new(&format!("distributed-roundtrip-{index}"));
            let (published, shards) = publish_distributed(&tmp.0, scope, index as u64 + 1);
            let expected = RolloutLedgerExpectations {
                identity: identity(),
                controls: controls_from_step(&shards[0]),
            };
            let rank0 = RolloutLedgerReader::open(&tmp.0, expected.clone())
                .unwrap()
                .read_distributed_step(7, 0, 2)
                .unwrap();
            let rank1 = RolloutLedgerReader::open(&tmp.0, expected)
                .unwrap()
                .read_distributed_step(7, 1, 2)
                .unwrap();
            assert_eq!(rank0.as_step(), &shards[0]);
            assert_eq!(rank1.as_step(), &shards[1]);
            assert_eq!(
                rank0.consumed_ledger_sha256(),
                rank1.consumed_ledger_sha256()
            );
            assert!(published.join(MANIFEST_FILE).is_file());
            assert!(published.join(distributed_payload_name(0)).is_file());
            assert!(published.join(distributed_payload_name(1)).is_file());
        }
    }

    #[test]
    fn distributed_high_dynamic_moments_roundtrip_in_rank_major_order() {
        let tmp = TempDir::new("distributed-high-dynamic-moments");
        let magnitude = f32::MAX;
        let reward_shards = [[magnitude, 1.0_f32], [-magnitude, 0.0_f32]];
        let global_rewards: Vec<f64> = reward_shards
            .iter()
            .flatten()
            .copied()
            .map(f64::from)
            .collect();
        let moments = finite_moments(&global_rewards);
        assert_eq!(moments.mean(), 0.25);
        let stats = RolloutLedgerRewardStats {
            count: 4,
            mean_bits: moments.mean().to_bits(),
            sample_std_bits: moments.sample_std().to_bits(),
        };
        let mut shards = [
            distributed_step(0, RolloutLedgerGroupScope::DistributedSamePrompt),
            distributed_step(1, RolloutLedgerGroupScope::DistributedSamePrompt),
        ];
        for (shard, rewards) in shards.iter_mut().zip(reward_shards) {
            shard.scale_rewards = ScaleRewards::None;
            shard.groups[0].reward_bits = rewards.into_iter().map(f32::to_bits).collect();
            shard.groups[0].distributed_reward_stats = Some(stats);
            shard.groups[0].advantage_bits = rewards
                .into_iter()
                .map(|reward| (f64::from(reward) - moments.mean()) as f32)
                .map(f32::to_bits)
                .collect();
        }
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let stage = writer.create_distributed_stage(7, 2, 120).unwrap();
        for shard in &shards {
            writer.write_distributed_shard(stage.path(), shard).unwrap();
        }
        writer
            .commit_distributed_stage(&stage, 7, 2, &controls_from_step(&shards[0]))
            .unwrap();
        for rank in 0..2 {
            RolloutLedgerReader::open(
                &tmp.0,
                RolloutLedgerExpectations {
                    identity: identity(),
                    controls: controls_from_step(&shards[0]),
                },
            )
            .unwrap()
            .read_distributed_step(7, rank, 2)
            .unwrap();
        }
    }

    #[test]
    fn distributed_staged_manifest_fence_failure_cleans_and_retries() {
        let tmp = TempDir::new("distributed-staged-manifest-fence");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let shards = [
            distributed_step(0, RolloutLedgerGroupScope::Local),
            distributed_step(1, RolloutLedgerGroupScope::Local),
        ];
        let stage = writer.create_distributed_stage(7, 2, 41).unwrap();
        for shard in &shards {
            writer.write_distributed_shard(stage.path(), shard).unwrap();
        }
        FAIL_SYNC_DIRECTORY_ONCE.with(|failure| {
            *failure.borrow_mut() = Some(stage.path().to_path_buf());
        });
        assert!(matches!(
            writer.commit_distributed_stage(&stage, 7, 2, &controls_from_step(&shards[0]),),
            Err(RolloutLedgerError::Io { .. })
        ));
        assert!(!stage.path().exists());
        assert!(!tmp.0.join(step_dir_name(7)).exists());

        let (_, retry_shards) = publish_distributed(&tmp.0, RolloutLedgerGroupScope::Local, 42);
        RolloutLedgerReader::open(
            &tmp.0,
            RolloutLedgerExpectations {
                identity: identity(),
                controls: controls_from_step(&retry_shards[0]),
            },
        )
        .unwrap()
        .read_distributed_step(7, 0, 2)
        .unwrap();
    }

    #[test]
    fn distributed_shard_write_failure_cleans_and_retries() {
        let tmp = TempDir::new("distributed-shard-write-retry");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let shards = [
            distributed_step(0, RolloutLedgerGroupScope::Local),
            distributed_step(1, RolloutLedgerGroupScope::Local),
        ];
        let stage = writer.create_distributed_stage(7, 2, 45).unwrap();
        writer
            .write_distributed_shard(stage.path(), &shards[0])
            .unwrap();
        fs::write(stage.path().join(distributed_payload_name(1)), b"occupied").unwrap();
        assert!(matches!(
            writer.write_distributed_shard(stage.path(), &shards[1]),
            Err(RolloutLedgerError::Io { .. })
        ));
        writer.abort_distributed_stage(&stage).unwrap();
        assert!(!stage.path().exists());

        let (_, retry_shards) = publish_distributed(&tmp.0, RolloutLedgerGroupScope::Local, 46);
        RolloutLedgerReader::open(
            &tmp.0,
            RolloutLedgerExpectations {
                identity: identity(),
                controls: controls_from_step(&retry_shards[0]),
            },
        )
        .unwrap()
        .read_distributed_step(7, 1, 2)
        .unwrap();
    }

    #[test]
    fn distributed_conflict_cleans_only_the_hidden_owned_stage() {
        let tmp = TempDir::new("distributed-conflict-cleanup");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let shards = [
            distributed_step(0, RolloutLedgerGroupScope::Local),
            distributed_step(1, RolloutLedgerGroupScope::Local),
        ];
        let stage = writer.create_distributed_stage(7, 2, 51).unwrap();
        for shard in &shards {
            writer.write_distributed_shard(stage.path(), shard).unwrap();
        }
        let winner = tmp.0.join(step_dir_name(7));
        fs::create_dir(&winner).unwrap();
        assert!(matches!(
            writer.commit_distributed_stage(
                &stage,
                7,
                2,
                &controls_from_step(&shards[0]),
            ),
            Err(RolloutLedgerError::AlreadyExists(path)) if path == winner
        ));
        assert!(!stage.path().exists());
        assert_eq!(fs::read_dir(winner).unwrap().count(), 0);
    }

    #[test]
    fn distributed_persistent_post_manifest_failure_is_visible_and_ambiguous() {
        let tmp = TempDir::new("distributed-post-manifest-ambiguous");
        let writer = RolloutLedgerWriter::create(&tmp.0, identity()).unwrap();
        let shards = [
            distributed_step(0, RolloutLedgerGroupScope::Local),
            distributed_step(1, RolloutLedgerGroupScope::Local),
        ];
        let stage = writer.create_distributed_stage(7, 2, 61).unwrap();
        for shard in &shards {
            writer.write_distributed_shard(stage.path(), shard).unwrap();
        }
        inject_persistent_post_manifest_sync_failure_once();
        assert!(matches!(
            writer.commit_distributed_stage(
                &stage,
                7,
                2,
                &controls_from_step(&shards[0]),
            ),
            Err(RolloutLedgerError::PublicationAmbiguous { path, .. })
                if path == tmp.0.join(step_dir_name(7))
        ));
        RolloutLedgerReader::open(
            &tmp.0,
            RolloutLedgerExpectations {
                identity: identity(),
                controls: controls_from_step(&shards[0]),
            },
        )
        .unwrap()
        .read_distributed_step(7, 1, 2)
        .unwrap();
    }

    #[test]
    fn distributed_reader_rejects_a_mutated_nonlocal_shard() {
        let tmp = TempDir::new("distributed-nonlocal-mutation");
        let (published, shards) = publish_distributed(&tmp.0, RolloutLedgerGroupScope::Local, 71);
        let nonlocal = published.join(distributed_payload_name(1));
        let mut bytes = fs::read(&nonlocal).unwrap();
        bytes[0] ^= 1;
        fs::write(nonlocal, bytes).unwrap();
        assert!(matches!(
            RolloutLedgerReader::open(
                &tmp.0,
                RolloutLedgerExpectations {
                    identity: identity(),
                    controls: controls_from_step(&shards[0]),
                },
            )
            .unwrap()
            .read_distributed_step(7, 0, 2),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("checksum")
        ));
    }

    #[test]
    fn distributed_reader_rejects_cross_shard_semantic_mutations() {
        type Mutation = fn(&mut RolloutLedgerStep);
        let cases: Vec<Mutation> = vec![
            |value| {
                for row in &mut value.groups[0].token_ids {
                    row[0] = 6;
                }
            },
            |value| value.groups[0].rollout_global_row_base += 1,
            |value| value.post_rollout_sampler_state.push(9),
            |value| value.window_tokens += 1,
            |value| value.world_size = 3,
            |value| value.rank = 0,
            |value| {
                value.groups[0]
                    .distributed_reward_stats
                    .as_mut()
                    .unwrap()
                    .mean_bits ^= 1;
            },
        ];
        for (index, mutate) in cases.into_iter().enumerate() {
            let tmp = TempDir::new(&format!("distributed-semantic-{index}"));
            let (_, shards) = publish_distributed(
                &tmp.0,
                RolloutLedgerGroupScope::DistributedSamePrompt,
                80 + index as u64,
            );
            rewrite_distributed_shard(&tmp.0, 1, mutate);
            assert!(matches!(
                RolloutLedgerReader::open(
                    &tmp.0,
                    RolloutLedgerExpectations {
                        identity: identity(),
                        controls: controls_from_step(&shards[0]),
                    },
                )
                .unwrap()
                .read_distributed_step(7, 0, 2),
                Err(RolloutLedgerError::Invalid(_))
            ));
        }
    }

    #[test]
    fn distributed_reader_recomputes_identically_forged_reward_moments() {
        let canonical = distributed_step(0, RolloutLedgerGroupScope::DistributedSamePrompt).groups
            [0]
        .distributed_reward_stats
        .unwrap();
        let forgeries = [
            (
                "mean",
                RolloutLedgerRewardStats {
                    mean_bits: 2.25_f64.to_bits(),
                    ..canonical
                },
            ),
            (
                "sample-std",
                RolloutLedgerRewardStats {
                    sample_std_bits: 1.0_f64.to_bits(),
                    ..canonical
                },
            ),
        ];
        for (index, (label, forged)) in forgeries.into_iter().enumerate() {
            let tmp = TempDir::new(&format!("distributed-forged-{label}"));
            let (_, shards) = publish_distributed(
                &tmp.0,
                RolloutLedgerGroupScope::DistributedSamePrompt,
                120 + index as u64,
            );
            forge_distributed_reward_stats(&tmp.0, forged);

            let dir = tmp.0.join(step_dir_name(7));
            for rank in 0..2 {
                let payload: RolloutLedgerStep = serde_json::from_slice(
                    &fs::read(dir.join(distributed_payload_name(rank))).unwrap(),
                )
                .unwrap();
                assert_eq!(payload.groups[0].distributed_reward_stats, Some(forged));
                validate_step(&payload).unwrap();
            }

            assert!(matches!(
                RolloutLedgerReader::open(
                    &tmp.0,
                    RolloutLedgerExpectations {
                        identity: identity(),
                        controls: controls_from_step(&shards[0]),
                    },
                )
                .unwrap()
                .read_distributed_step(7, 0, 2),
                Err(RolloutLedgerError::Invalid(message))
                    if message.contains("reward statistics do not match shard rewards")
            ));
        }
    }

    #[test]
    fn distributed_reader_rejects_missing_shards_and_noncanonical_order() {
        let missing = TempDir::new("distributed-missing-shard");
        let (published, shards) =
            publish_distributed(&missing.0, RolloutLedgerGroupScope::Local, 91);
        fs::remove_file(published.join(distributed_payload_name(1))).unwrap();
        assert!(matches!(
            RolloutLedgerReader::open(
                &missing.0,
                RolloutLedgerExpectations {
                    identity: identity(),
                    controls: controls_from_step(&shards[0]),
                },
            )
            .unwrap()
            .read_distributed_step(7, 0, 2),
            Err(RolloutLedgerError::Io { .. })
        ));

        let reordered = TempDir::new("distributed-reordered-shards");
        let (published, shards) =
            publish_distributed(&reordered.0, RolloutLedgerGroupScope::Local, 92);
        let manifest_path = published.join(MANIFEST_FILE);
        let mut manifest: DistributedRolloutLedgerManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest.shards.swap(0, 1);
        fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(matches!(
            RolloutLedgerReader::open(
                &reordered.0,
                RolloutLedgerExpectations {
                    identity: identity(),
                    controls: controls_from_step(&shards[0]),
                },
            )
            .unwrap()
            .read_distributed_step(7, 0, 2),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("canonical")
        ));
    }

    #[test]
    fn zero_length_eos_filled_window_is_not_first_eos_inclusive() {
        let value = degenerate_zero_token_step();
        assert!(matches!(
            validate_step(&value),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("first EOS token")
        ));
    }

    #[test]
    fn f64_nonzero_advantage_rejects_f32_zero_rounding() {
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
        assert!(matches!(
            validate_step(&value),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("nonzero-preserving f32")
        ));
    }

    #[test]
    fn prompt_ordinals_follow_world_one_window_order() {
        let mut value = step();
        value.grad_accum_steps = 2;
        let mut second = value.groups[0].clone();
        value.groups[0].accum_index = 0;
        value.groups[0].prompt_index = 14;
        value.groups[0].rollout_global_row_base = 28;
        second.accum_index = 1;
        second.prompt_index = 15;
        second.rollout_global_row_base = 30;
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
    fn nonfinite_rewards_are_rejected_before_advantage_validation() {
        for reward_bits in [
            vec![f32::NAN.to_bits(), 3.0_f32.to_bits()],
            vec![f32::NEG_INFINITY.to_bits(), f32::INFINITY.to_bits()],
        ] {
            let mut value = step();
            value.groups[0].reward_bits = reward_bits;
            value.groups[0].advantage_bits = vec![0.0_f32.to_bits(); 2];
            assert!(matches!(
                validate_step(&value),
                Err(RolloutLedgerError::Invalid(message)) if message.contains("reward")
            ));
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
    fn reader_rejects_uniformly_wrong_prefixes_against_persisted_selected_prompt() {
        let tmp = TempDir::new("uniform-wrong-persisted-prompt");
        RolloutLedgerWriter::create(&tmp.0, identity())
            .unwrap()
            .write_step(&step())
            .unwrap();
        rewrite_payload(&tmp.0, |value| {
            for row in &mut value.groups[0].token_ids {
                row[0] = 6;
            }
        });

        assert!(matches!(
            RolloutLedgerReader::open(&tmp.0, expectations())
                .unwrap()
                .read_step(7),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("prompt prefix")
        ));
    }

    #[test]
    fn distributed_reader_rejects_identically_forged_prefixes_on_every_shard() {
        let tmp = TempDir::new("distributed-uniform-wrong-persisted-prompt");
        let (_, shards) =
            publish_distributed(&tmp.0, RolloutLedgerGroupScope::DistributedSamePrompt, 86);
        for rank in 0..2 {
            rewrite_distributed_shard(&tmp.0, rank, |value| {
                for row in &mut value.groups[0].token_ids {
                    row[0] = 6;
                }
            });
        }

        assert!(matches!(
            RolloutLedgerReader::open(
                &tmp.0,
                RolloutLedgerExpectations {
                    identity: identity(),
                    controls: controls_from_step(&shards[0]),
                },
            )
            .unwrap()
            .read_distributed_step(7, 0, 2),
            Err(RolloutLedgerError::Invalid(message)) if message.contains("prompt prefix")
        ));
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
