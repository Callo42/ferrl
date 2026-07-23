//! Adapter checkpointing: persist and restore the trainable `LoRA` parameters.
//!
//! A training run's learnable state is exactly the set of [`Var`]s the optimizer
//! updates — the `LoRA` `A`/`B` factors exposed by [`crate::Policy::trainable_vars`].
//! [`save_adapter`] writes them to `adapter.safetensors` (plus a small
//! `manifest.json`) under a checkpoint directory; [`load_adapter`] reads them back
//! into a live model's `Var`s.
//!
//! ## Generic over the policy
//!
//! Save/load operate on a `&[Var]` slice, not a concrete model. A cloned [`Var`]
//! shares its inner storage with the original (see
//! [`crate::Policy::trainable_vars`]), so [`load_adapter`] calls [`Var::set`] on
//! the slice and the change is reflected in whatever model those `Var`s came from.
//! The *same* code therefore checkpoints the toy policy and the Qwen policy. The
//! slice order is the model's stable `trainable_vars()` order; load validates the
//! tensor count and each tensor's shape and dtype against the live `Var`s, so a
//! checkpoint from a mismatched architecture fails loud rather than corrupting the
//! model.
//!
//! ## Two checkpoint flavors
//!
//! - [`save_adapter`] / [`load_adapter`] — **adapter only** (the eval / inference
//!   path): just the trainable weights + the step count. This is the legacy
//!   **format version 1** layout, and is what [`crate::eval`] loads.
//! - [`save_checkpoint`] / [`load_checkpoint`] — an identity-bound,
//!   **momentum-faithful format-v4** checkpoint: adapter/trainable weights, Adam
//!   `m`/`v`/`step_t`, sampler state, exact recipe, immutable frozen-policy digest,
//!   canonical learner semantics, ordered tensor schema, and exact payload hashes.
//!   One domain-separated state-envelope root binds those leaves to the completed
//!   step and optimizer/schema relationship. Raw typed decoding rejects duplicate
//!   manifest keys before generic JSON inspection. [`load_checkpoint`] validates the
//!   complete binding and every payload before its first live [`Var::set`]. This is
//!   the only ordinary format accepted by [`crate::Trainer::resume`].
//!
//! Legacy v1 remains explicitly readable through [`load_adapter`]. Structurally
//! complete v2/v3 manifests are parsed with a strict version-specific field matrix,
//! but are not trusted ordinary-resume inputs because they cannot prove policy/config
//! identity or payload integrity. Separated rollout-ledger continuations retain their
//! v3 outer envelope and their stronger nested identity contract. The manifest is
//! always written **last** within a checkpoint directory, as the commit marker.
//!
//! ## Crash atomicity
//!
//! Both writers coordinate through one persistent per-destination advisory lock,
//! so an ordinary replacement cannot overlap continuation publication. Ordinary
//! writers then stage the whole checkpoint into a sibling temp directory
//! (`<dir>.tmp-<pid>`) and publish it with a single `rename` — so the published
//! path never holds a *partial* checkpoint (the manifest-last ordering inside
//! the stage is belt-and-braces on top). Replacing an existing checkpoint
//! renames the old directory aside first and removes it only after the new one
//! is published, so at every instant the prior **or** the new complete
//! checkpoint exists on disk (a crash can at worst leave the prior one under
//! `<dir>.old-<pid>`, recoverable by hand). Stale `.tmp-*`/`.old-*` siblings
//! from crashed processes are swept by the next write to the same path.
//! Separated rollout continuations use an internal no-replace variant instead:
//! an atomic destination-directory claim prevents forks, a synced private owner
//! marker binds cleanup to that exact claim, and a temporary manifest is renamed
//! into place only after both payload files and the manifest have been synced and
//! their directory entries durably anchored. The checkpoint directory is synced
//! again after the rename, followed by its parent so the directory claim itself is
//! durable. Any missing parent chain is created one component at a time, syncing
//! each new directory entry in its already durable ancestor before proceeding.
//! Successful publication thus requires a filesystem/platform that supports
//! advisory file locks plus opening and syncing directories; unsupported behavior
//! returns an I/O error instead of claiming durability. A failed pre-manifest fence
//! is cleaned only while the marker proves ownership, and the removal is synced
//! before returning. Every post-manifest success path verifies that the owner marker
//! and all three visible files still match the intended package; a failed fence
//! retries the complete durability boundary before the same check. If ownership,
//! cleanup, durability, or the exact-package check cannot be confirmed, the writer
//! returns [`CheckpointError::PublicationAmbiguous`] for explicit operator
//! reconciliation. A crash may still strand an incomplete claim, but it cannot
//! expose or overwrite a completed continuation.

use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{Device, Tensor, Var};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::optim::OptimizerState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoReplaceSyncPoint {
    ClaimParent,
    PreManifestDirectory,
    ManifestDirectory,
    ManifestParent,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoReplaceHookPoint {
    BeforeFingerprint,
    BeforeManifestRename,
    BeforeFirstFence,
    AfterFirstFenceFailure,
}

#[cfg(test)]
type NoReplaceHook = Box<dyn FnOnce(&Path)>;

#[cfg(test)]
thread_local! {
    static SYNCED_DIRECTORIES: std::cell::RefCell<Vec<PathBuf>> = const { std::cell::RefCell::new(Vec::new()) };
    static FAIL_SYNC_DIRECTORY_ONCE: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
    static FAIL_NO_REPLACE_SYNCS: std::cell::RefCell<Vec<NoReplaceSyncPoint>> = const { std::cell::RefCell::new(Vec::new()) };
    static NO_REPLACE_HOOK: std::cell::RefCell<Option<(NoReplaceHookPoint, NoReplaceHook)>> = const { std::cell::RefCell::new(None) };
    static PANIC_NO_REPLACE_AFTER_MANIFEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn inject_continuation_pre_manifest_sync_failure_once() {
    FAIL_NO_REPLACE_SYNCS.with(|failures| {
        *failures.borrow_mut() = vec![NoReplaceSyncPoint::PreManifestDirectory];
    });
}

#[cfg(test)]
pub(crate) fn inject_persistent_continuation_post_manifest_sync_failure_once() {
    FAIL_NO_REPLACE_SYNCS.with(|failures| {
        *failures.borrow_mut() = vec![
            NoReplaceSyncPoint::ManifestDirectory,
            NoReplaceSyncPoint::ManifestDirectory,
        ];
    });
}

#[cfg(test)]
pub(crate) fn inject_continuation_post_manifest_panic_once() {
    PANIC_NO_REPLACE_AFTER_MANIFEST.with(|panic| panic.set(true));
}

static CLAIM_TOKEN_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct CheckpointWriterLock {
    _file: std::fs::File,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClaimOwnership {
    token: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileFingerprint {
    length: u64,
    sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CheckpointPackageFingerprint {
    adapter: FileFingerprint,
    optimizer: FileFingerprint,
    manifest: FileFingerprint,
}

/// Filename of the serialized adapter tensors within a checkpoint directory.
const ADAPTER_FILE: &str = "adapter.safetensors";
/// Filename of the serialized optimizer moment tensors in momentum-bearing formats.
const OPTIMIZER_FILE: &str = "optimizer.safetensors";
/// Filename of the checkpoint manifest within a checkpoint directory.
const MANIFEST_FILE: &str = "manifest.json";
/// Persistent sibling file used for the shared advisory destination lock.
const WRITER_LOCK_SUFFIX: &str = ".writer-lock";
/// Private file binding no-replace cleanup and publication to one exact claim.
const CLAIM_OWNER_FILE: &str = ".ferrl-continuation-owner";
/// On-disk checkpoint layout version; bumped on an incompatible format change. v1 =
/// adapter only; v2 = adapter + optimizer moments + sampler RNG (momentum-faithful);
/// v3 = v2 with the sampler blob now carrying the run `base_seed` (global-index
/// substream seeding — see [`crate::sampler::GrpoSampler`]); v4 adds the required
/// ordinary-checkpoint identity and exact payload digests. New ordinary trainer
/// checkpoints are v4. Separated rollout-ledger continuations retain their existing
/// v3 outer envelope because their nested manifest already binds the complete state.
const FORMAT_VERSION: u32 = 4;
const LEGACY_MOMENTUM_FORMAT_VERSION: u32 = 3;
/// Lowest on-disk format version this build can read. Older (v1, adapter-only)
/// checkpoints remain available to the explicit adapter/eval path.
const MIN_FORMAT_VERSION: u32 = 1;
const ORDINARY_CHECKPOINT_KIND: &str = "ordinary";
/// On-disk schema version for separated rollout-ledger continuations.
pub(crate) const ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION: u32 = 3;
pub(crate) const MIN_ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION: u32 = 1;
pub(crate) const ROLLOUT_LEDGER_CONTINUATION_KIND: &str = "rollout_ledger";
pub(crate) const ROLLOUT_LEDGER_CONTINUATION_LAYOUT: &str =
    "tensor_parallel.communicator_rank_ascending.v1";

/// Provenance that distinguishes a separated rollout-ledger continuation from
/// an ordinary trainer checkpoint and binds it to one exact ledger chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RolloutLedgerContinuationManifest {
    pub(crate) format_version: u32,
    pub(crate) kind: String,
    /// Absent only in legacy v1 manifests, which are defined as world one.
    #[serde(default)]
    pub(crate) world_size: Option<u32>,
    /// Absent in v1/v2 manifests, which are defined as tensor-parallel world one.
    #[serde(default)]
    pub(crate) tensor_parallel_world_size: Option<u32>,
    /// Canonical TP shard/communicator ordering. Absent only in legacy v1/v2.
    #[serde(default)]
    pub(crate) tensor_parallel_layout: Option<String>,
    pub(crate) completed_step: u64,
    pub(crate) policy_sha256: String,
    pub(crate) trainer_config_sha256: String,
    pub(crate) tensor_schema_sha256: String,
    pub(crate) adapter_sha256: String,
    pub(crate) optimizer_sha256: String,
    pub(crate) sampler_sha256: String,
    pub(crate) parent_lineage_sha256: String,
    pub(crate) consumed_ledger_sha256: String,
    pub(crate) lineage_sha256: String,
}

/// Caller-supplied immutable provenance required to write or restore an ordinary
/// checkpoint.
///
/// The generic checkpoint layer can derive the ordered trainable schema and every
/// mutable payload digest itself. It cannot derive the frozen model content or the
/// learner-semantic trainer projection from a bare `&[Var]`, so those two verified
/// digests are supplied by [`crate::Trainer`]. The adapter recipe is carried here so
/// save and restore compare its exact presence/value and include it in the schema
/// identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckpointBinding {
    frozen_policy_sha256: String,
    trainer_config_sha256: String,
    lora_recipe: Option<String>,
}

impl CheckpointBinding {
    /// Construct a validated ordinary-checkpoint binding.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Mismatch`] unless both digests are exactly 64
    /// lowercase hexadecimal characters.
    pub fn new(
        frozen_policy_sha256: impl Into<String>,
        trainer_config_sha256: impl Into<String>,
        lora_recipe: Option<String>,
    ) -> Result<Self, CheckpointError> {
        let binding = Self {
            frozen_policy_sha256: frozen_policy_sha256.into(),
            trainer_config_sha256: trainer_config_sha256.into(),
            lora_recipe,
        };
        validate_sha256("frozen_policy_sha256", &binding.frozen_policy_sha256)?;
        validate_sha256("trainer_config_sha256", &binding.trainer_config_sha256)?;
        Ok(binding)
    }

    /// Verified digest of the immutable frozen policy and execution recipe.
    #[must_use]
    pub fn frozen_policy_sha256(&self) -> &str {
        &self.frozen_policy_sha256
    }

    /// Digest of the canonical learner-semantic trainer projection and topology.
    #[must_use]
    pub fn trainer_config_sha256(&self) -> &str {
        &self.trainer_config_sha256
    }

    /// Exact adapter/full-FT recipe identity of the live policy.
    #[must_use]
    pub fn lora_recipe(&self) -> Option<&str> {
        self.lora_recipe.as_deref()
    }
}

/// Required identity and integrity block for an ordinary format-v4 checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrdinaryCheckpointIdentity {
    /// Checkpoint flavor discriminator; exactly `"ordinary"` for v4.
    pub kind: String,
    /// Immutable frozen policy plus execution-recipe digest.
    pub frozen_policy_sha256: String,
    /// Canonical learner-semantic trainer configuration plus topology digest.
    pub trainer_config_sha256: String,
    /// Ordered trainable key/shape/dtype schema plus exact recipe digest.
    pub tensor_schema_sha256: String,
    /// Exact serialized `adapter.safetensors` payload digest.
    pub adapter_sha256: String,
    /// Exact serialized Adam payload plus bias-correction counter digest.
    pub optimizer_sha256: String,
    /// Exact opaque sampler-state payload digest.
    pub sampler_sha256: String,
    /// Canonical root binding the complete ordinary training-state envelope:
    /// version/kind, completed step, tensor/optimizer counts, exact recipe,
    /// immutable identities, and every mutable payload digest.
    pub state_envelope_sha256: String,
}

/// Errors raised while saving or loading an adapter checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// A filesystem operation (create dir / read / write) failed.
    #[error("checkpoint io error at {path}: {source}")]
    Io {
        /// Path being operated on when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// A no-replace continuation may hold an incomplete claim or a reader-visible
    /// package, but cleanup, durability, or exact package identity could not be
    /// confirmed.
    #[error("checkpoint publication requires reconciliation at {path}: {detail}")]
    PublicationAmbiguous {
        /// Claimed checkpoint destination that must be inspected before repair.
        path: PathBuf,
        /// Original publication failure plus the failed reconciliation action.
        detail: String,
    },
    /// A candle tensor or safetensors operation failed.
    #[error("checkpoint tensor error: {0}")]
    Candle(#[from] candle_core::Error),
    /// The manifest could not be serialized or deserialized.
    #[error("checkpoint manifest error: {0}")]
    Manifest(#[from] serde_json::Error),
    /// The on-disk checkpoint violates its version matrix, expected identity,
    /// payload digest, exact tensor-key/schema contract, or live trainable [`Var`]
    /// shape/dtype contract.
    #[error("checkpoint mismatch: {0}")]
    Mismatch(String),
}

/// Self-describing metadata stored alongside checkpoint tensors.
///
/// Legacy fields remain optional at the serde layer so v1 can be read, while
/// Manifest decoding enforces the exact required/forbidden matrix for every version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointManifest {
    /// On-disk format version (validated against the supported range by readers).
    pub format_version: u32,
    /// Number of optimizer steps completed before this checkpoint was written —
    /// the step index a resumed run should continue from
    /// (see [`crate::Trainer::resume`]).
    pub step: u64,
    /// Number of trainable tensors persisted (the `trainable_vars()` count).
    pub num_vars: usize,
    /// v2: the optimizer's global step counter `t` (bias-correction state), if the
    /// optimizer moments were persisted.
    #[serde(default)]
    pub optimizer_step_t: Option<usize>,
    /// v2: the number of optimizer moment pairs persisted (the float-filtered parameter
    /// count), if the optimizer moments were persisted.
    #[serde(default)]
    pub optimizer_num_vars: Option<usize>,
    /// v2: the opaque rollout-sampler RNG blob ([`crate::Policy::sampler_state`]), if it
    /// was persisted.
    #[serde(default)]
    pub sampler_state: Option<Vec<u8>>,
    /// The `LoRA` recipe the adapter was trained with, as a stable canonical
    /// string (e.g. `attn:qkvo|mlp:gud` — see
    /// [`crate::lora::DenseLoraTargets::canonical`] /
    /// [`crate::qwen35::LoraTargets::canonical`]) — recorded so a checkpoint is
    /// self-describing about *which* projections its positional tensor list covers.
    /// Format v4 requires explicit presence (a string or `null`), binds the value
    /// into its ordered schema identity, and compares it exactly with the restoring
    /// [`CheckpointBinding`] before mutation; count/shape/dtype alone cannot
    /// distinguish **shape-aliased** recipes such as `attn:qk` and `attn:qv`.
    /// `#[serde(default)]` retains legacy v1-v3 parsing.
    #[serde(default)]
    pub lora_recipe: Option<String>,
    /// Present only for a versioned separated rollout-ledger continuation.
    /// Ordinary cadence/eval checkpoints omit it and are never eligible for
    /// separated-continuation discovery.
    #[serde(default)]
    pub(crate) rollout_ledger_continuation: Option<RolloutLedgerContinuationManifest>,
    /// Required only for an ordinary format-v4 checkpoint. Legacy adapter and
    /// separated-continuation formats must omit it (or serialize it as `null`).
    #[serde(default)]
    pub ordinary_checkpoint: Option<OrdinaryCheckpointIdentity>,
}

/// Tensor key for the `i`-th trainable var, zero-padded so lexical order matches
/// numeric order.
fn var_key(i: usize) -> String {
    format!("lora.{i:05}")
}

/// Tensor key for the `i`-th optimizer moment of kind `kind` (`"m"` first moment,
/// `"v"` second moment), zero-padded like [`var_key`].
fn moment_key(kind: &str, i: usize) -> String {
    format!("{kind}.{i:05}")
}

/// Build a [`CheckpointError::Io`] for `path`.
fn io(path: impl Into<PathBuf>, source: std::io::Error) -> CheckpointError {
    CheckpointError::Io {
        path: path.into(),
        source,
    }
}

fn validate_sha256(label: &str, digest: &str) -> Result<(), CheckpointError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(CheckpointError::Mismatch(format!(
        "{label} must be 64 lowercase hexadecimal characters"
    )))
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

fn tensor_schema_sha256(
    vars: &[Var],
    lora_recipe: Option<&str>,
) -> Result<String, CheckpointError> {
    let schema = vars
        .iter()
        .enumerate()
        .map(|(index, var)| {
            (
                var_key(index),
                var.as_tensor().dims().to_vec(),
                var.as_tensor().dtype().as_str().to_owned(),
            )
        })
        .collect::<Vec<_>>();
    let encoded = serde_json::to_vec(&(lora_recipe, schema))?;
    Ok(domain_sha256(
        "ferrl.ordinary-checkpoint.tensor-schema.v1",
        &[&encoded],
    ))
}

fn adapter_payload_sha256(bytes: &[u8]) -> String {
    domain_sha256("ferrl.ordinary-checkpoint.adapter.v1", &[bytes])
}

fn optimizer_payload_sha256(step_t: usize, bytes: &[u8]) -> Result<String, CheckpointError> {
    let step_t = u64::try_from(step_t).map_err(|_| {
        CheckpointError::Mismatch(
            "optimizer step_t does not fit the ordinary checkpoint u64 identity".into(),
        )
    })?;
    Ok(domain_sha256(
        "ferrl.ordinary-checkpoint.optimizer.v1",
        &[&step_t.to_le_bytes(), bytes],
    ))
}

fn sampler_payload_sha256(bytes: &[u8]) -> String {
    domain_sha256("ferrl.ordinary-checkpoint.sampler.v1", &[bytes])
}

#[derive(Serialize)]
struct OrdinaryCheckpointEnvelope<'a> {
    format_version: u32,
    kind: &'a str,
    step: u64,
    num_vars: u64,
    lora_recipe: Option<&'a str>,
    optimizer_step_t: u64,
    optimizer_num_vars: u64,
    frozen_policy_sha256: &'a str,
    trainer_config_sha256: &'a str,
    tensor_schema_sha256: &'a str,
    adapter_sha256: &'a str,
    optimizer_sha256: &'a str,
    sampler_sha256: &'a str,
}

#[allow(clippy::too_many_arguments)] // fixed canonical checkpoint-state envelope
fn state_envelope_sha256(
    format_version: u32,
    kind: &str,
    step: u64,
    num_vars: usize,
    lora_recipe: Option<&str>,
    optimizer_step_t: usize,
    optimizer_num_vars: usize,
    frozen_policy_sha256: &str,
    trainer_config_sha256: &str,
    tensor_schema_sha256: &str,
    adapter_sha256: &str,
    optimizer_sha256: &str,
    sampler_sha256: &str,
) -> Result<String, CheckpointError> {
    let num_vars = u64::try_from(num_vars).map_err(|_| {
        CheckpointError::Mismatch(
            "trainable tensor count does not fit the ordinary checkpoint u64 envelope".into(),
        )
    })?;
    let optimizer_step_t = u64::try_from(optimizer_step_t).map_err(|_| {
        CheckpointError::Mismatch(
            "optimizer step_t does not fit the ordinary checkpoint u64 envelope".into(),
        )
    })?;
    let optimizer_num_vars = u64::try_from(optimizer_num_vars).map_err(|_| {
        CheckpointError::Mismatch(
            "optimizer tensor count does not fit the ordinary checkpoint u64 envelope".into(),
        )
    })?;
    let envelope = OrdinaryCheckpointEnvelope {
        format_version,
        kind,
        step,
        num_vars,
        lora_recipe,
        optimizer_step_t,
        optimizer_num_vars,
        frozen_policy_sha256,
        trainer_config_sha256,
        tensor_schema_sha256,
        adapter_sha256,
        optimizer_sha256,
        sampler_sha256,
    };
    let encoded = serde_json::to_vec(&envelope)?;
    Ok(domain_sha256(
        "ferrl.ordinary-checkpoint.state-envelope.v1",
        &[&encoded],
    ))
}

fn validate_state_envelope(
    manifest: &CheckpointManifest,
    identity: &OrdinaryCheckpointIdentity,
) -> Result<(), CheckpointError> {
    let optimizer_step_t = manifest.optimizer_step_t.ok_or_else(|| {
        CheckpointError::Mismatch("ordinary checkpoint is missing optimizer_step_t".into())
    })?;
    let optimizer_num_vars = manifest.optimizer_num_vars.ok_or_else(|| {
        CheckpointError::Mismatch("ordinary checkpoint is missing optimizer_num_vars".into())
    })?;
    let expected = state_envelope_sha256(
        manifest.format_version,
        &identity.kind,
        manifest.step,
        manifest.num_vars,
        manifest.lora_recipe.as_deref(),
        optimizer_step_t,
        optimizer_num_vars,
        &identity.frozen_policy_sha256,
        &identity.trainer_config_sha256,
        &identity.tensor_schema_sha256,
        &identity.adapter_sha256,
        &identity.optimizer_sha256,
        &identity.sampler_sha256,
    )?;
    if identity.state_envelope_sha256 != expected {
        return Err(CheckpointError::Mismatch(
            "ordinary checkpoint state-envelope digest mismatch".into(),
        ));
    }
    Ok(())
}

fn sibling_path_with_suffix(dir: &Path, suffix: &str) -> PathBuf {
    let mut name = dir.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    dir.with_file_name(name)
}

fn acquire_checkpoint_writer_lock(dir: &Path) -> Result<CheckpointWriterLock, CheckpointError> {
    let parent = dir.parent().ok_or_else(|| {
        io(
            dir,
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "checkpoint path has no parent directory",
            ),
        )
    })?;
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).map_err(|error| io(parent, error))?;
    }
    let lock_path = sibling_path_with_suffix(dir, WRITER_LOCK_SUFFIX);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|error| io(&lock_path, error))?;
    FileExt::try_lock_exclusive(&file).map_err(|error| {
        let error = if error.kind() == std::io::ErrorKind::WouldBlock {
            std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "another checkpoint writer owns this destination",
            )
        } else {
            error
        };
        io(&lock_path, error)
    })?;
    Ok(CheckpointWriterLock { _file: file })
}

impl ClaimOwnership {
    fn create(dir: &Path) -> Result<Self, CheckpointError> {
        let sequence = CLAIM_TOKEN_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let token = format!(
            "{}:{:?}:{now}:{sequence}",
            std::process::id(),
            std::thread::current().id()
        )
        .into_bytes();
        let path = dir.join(CLAIM_OWNER_FILE);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| io(&path, error))?;
        file.write_all(&token).map_err(|error| io(&path, error))?;
        file.sync_all().map_err(|error| io(&path, error))?;
        Ok(Self { token })
    }

    fn matches(&self, dir: &Path) -> std::io::Result<bool> {
        std::fs::read(dir.join(CLAIM_OWNER_FILE)).map(|actual| actual == self.token)
    }
}

/// The sibling staging directory for an atomic checkpoint write:
/// `<dir>.tmp-<pid>` (pid-suffixed so a stale stage from a dead process can
/// never be confused with this one's).
fn stage_path(dir: &Path) -> PathBuf {
    sibling_path_with_suffix(dir, &format!(".tmp-{}", std::process::id()))
}

/// Prepare an empty staging directory for `dir`, sweeping any stale `.tmp-*` /
/// `.old-*` sibling left behind by a crashed process (the pid suffix makes a
/// live collision impossible, so anything matching is garbage).
fn prepare_stage(dir: &Path) -> Result<PathBuf, CheckpointError> {
    sweep_stale_siblings(dir)?;
    let stage = stage_path(dir);
    std::fs::create_dir_all(&stage).map_err(|e| io(&stage, e))?;
    Ok(stage)
}

/// Remove every `<name>.tmp-*` / `<name>.old-*` sibling of `dir` — leftovers
/// from interrupted writes by this or any dead process. Best-effort per entry
/// is NOT enough here (a stale dir at this pid's own stage path must go), so
/// failures surface.
fn sweep_stale_siblings(dir: &Path) -> Result<(), CheckpointError> {
    let Some(parent) = dir.parent() else {
        return Ok(());
    };
    let Some(name) = dir.file_name().and_then(|n| n.to_str()) else {
        return Ok(());
    };
    if !parent.exists() {
        return Ok(());
    }
    let entries = std::fs::read_dir(parent).map_err(|e| io(parent, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io(parent, e))?;
        let fname = entry.file_name();
        let Some(fname) = fname.to_str() else {
            continue;
        };
        let stale = fname
            .strip_prefix(name)
            .is_some_and(|rest| rest.starts_with(".tmp-") || rest.starts_with(".old-"));
        if stale {
            std::fs::remove_dir_all(entry.path()).map_err(|e| io(entry.path(), e))?;
        }
    }
    Ok(())
}

/// Publish a fully-written `stage` at `dir`. A prior checkpoint is renamed
/// aside (never deleted before the new one is in place): old -> `.old-<pid>`,
/// stage -> `dir`, then the aside copy is removed. At every instant the path
/// set holds the prior or the new complete checkpoint, so a crash anywhere in
/// this sequence loses nothing (at worst the prior survives under the aside
/// name, swept by the next write).
fn commit_stage(stage: &Path, dir: &Path) -> Result<(), CheckpointError> {
    let mut aside_name = dir.file_name().unwrap_or_default().to_os_string();
    aside_name.push(format!(".old-{}", std::process::id()));
    let aside = dir.with_file_name(aside_name);
    let had_prior = dir.exists();
    if had_prior {
        std::fs::rename(dir, &aside).map_err(|e| io(dir, e))?;
    }
    std::fs::rename(stage, dir).map_err(|e| io(stage, e))?;
    if had_prior {
        std::fs::remove_dir_all(&aside).map_err(|e| io(&aside, e))?;
    }
    Ok(())
}

/// Run `write` against a staging directory for `dir`, committing on success and
/// best-effort cleaning the stage on failure (so an aborted write does not
/// strand a half-built directory next to the real checkpoints).
fn write_staged(
    dir: &Path,
    write: impl FnOnce(&Path) -> Result<(), CheckpointError>,
) -> Result<(), CheckpointError> {
    // Every public writer for this destination takes the same OS-released lock.
    // Keep the guard alive through stale-stage cleanup, commit, and rollback.
    let _writer_lock = acquire_checkpoint_writer_lock(dir)?;
    let stage = prepare_stage(dir)?;
    match write(&stage) {
        Ok(()) => commit_stage(&stage, dir),
        Err(e) => {
            let _ = std::fs::remove_dir_all(&stage);
            Err(e)
        }
    }
}

/// Persist `vars` (the trainable `LoRA` factors) and a manifest to `dir`.
///
/// Writes `dir/adapter.safetensors` and `dir/manifest.json`, creating `dir` (and
/// parents) if needed. `step` is recorded as the number of completed optimizer
/// steps — the index a resume should continue from; `lora_recipe` (if given) is
/// recorded verbatim so the checkpoint is self-describing about its adapter
/// recipe (see [`CheckpointManifest::lora_recipe`]). Each tensor is moved to the
/// CPU and made contiguous before serialization, so this works for vars living on
/// any device.
///
/// The write is **crash-atomic**: everything is staged into a sibling temp
/// directory and published at `dir` with a single `rename` (see the module
/// docs), with the manifest written last inside the stage as belt-and-braces.
/// Re-writing an existing `dir` replaces it as a unit.
///
/// # Errors
///
/// Returns [`CheckpointError`] if the staging directory cannot be created, a
/// tensor cannot be moved to CPU / serialized, the manifest cannot be written,
/// or the final rename fails.
pub fn save_adapter(
    dir: impl AsRef<Path>,
    vars: &[Var],
    step: u64,
    lora_recipe: Option<&str>,
) -> Result<(), CheckpointError> {
    let recipe = lora_recipe.map(str::to_owned);
    write_staged(dir.as_ref(), |stage| {
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(vars.len());
        for (i, v) in vars.iter().enumerate() {
            // CPU + contiguous so a CUDA-resident adapter serializes the same way.
            let t = v.as_tensor().to_device(&Device::Cpu)?.contiguous()?;
            tensors.insert(var_key(i), t);
        }
        candle_core::safetensors::save(&tensors, stage.join(ADAPTER_FILE))?;

        let manifest = CheckpointManifest {
            format_version: 1,
            step,
            num_vars: vars.len(),
            optimizer_step_t: None,
            optimizer_num_vars: None,
            sampler_state: None,
            lora_recipe: recipe,
            rollout_ledger_continuation: None,
            ordinary_checkpoint: None,
        };
        let manifest_path = stage.join(MANIFEST_FILE);
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, json).map_err(|e| io(&manifest_path, e))?;
        Ok(())
    })
}

/// Restore an explicit legacy format-v1 adapter package from `dir` into `vars`.
///
/// Reads `dir/manifest.json` and `dir/adapter.safetensors`, validates that the
/// checkpoint matches `vars` (format version, tensor count, and each tensor's
/// shape and dtype), then calls [`Var::set`] on each — updating the model those
/// `Var`s belong to (cloned vars share storage). Returns the manifest; its `step`
/// is where a resumed run should continue.
///
/// **All-or-nothing:** every tensor is validated and device-prepared *before* any
/// `Var::set` runs, so a mismatched or missing tensor leaves the model **entirely
/// unmodified** rather than partially overwritten.
///
/// # Errors
///
/// Returns [`CheckpointError::Mismatch`] if the format is not v1 (v4 requires
/// [`load_checkpoint`], while v2/v3 require deliberate migration), the tensor count
/// differs, a tensor is missing/extra, or any tensor's shape/dtype does not match the
/// corresponding live `Var`; or [`CheckpointError::Io`] /
/// [`CheckpointError::Candle`] / [`CheckpointError::Manifest`] on read failures.
pub fn load_adapter(
    dir: impl AsRef<Path>,
    vars: &[Var],
) -> Result<CheckpointManifest, CheckpointError> {
    let dir = dir.as_ref();

    let manifest = read_manifest(dir)?;
    if manifest.format_version != 1 {
        return Err(CheckpointError::Mismatch(format!(
            "load_adapter accepts only explicit legacy format-v1 adapter packages; format v{} requires its identity-aware restore or migration path",
            manifest.format_version
        )));
    }
    if manifest.num_vars != vars.len() {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has {} tensors but the model exposes {} trainable vars",
            manifest.num_vars,
            vars.len()
        )));
    }

    let adapter_path = dir.join(ADAPTER_FILE);
    let bytes = std::fs::read(&adapter_path).map_err(|error| io(&adapter_path, error))?;
    let prepared = prepare_adapter_tensors(&bytes, vars)?;
    apply_adapter_tensors(vars, &prepared)?;
    Ok(manifest)
}

/// Read and version/field-matrix-validate `manifest.json` from `dir` without
/// touching model state. Ordinary resume uses the result as the first stage of
/// its complete identity, schema, and payload preflight.
pub(crate) fn read_manifest(dir: &Path) -> Result<CheckpointManifest, CheckpointError> {
    let manifest_path = dir.join(MANIFEST_FILE);
    let raw = std::fs::read_to_string(&manifest_path).map_err(|e| io(&manifest_path, e))?;
    // Parse the typed structures directly from the raw object first. Serde's
    // derived map visitors reject duplicate known fields at every nested level;
    // parsing through `Value` first would silently retain only the last value.
    let manifest: CheckpointManifest = serde_json::from_str(&raw)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    if manifest.format_version < MIN_FORMAT_VERSION || manifest.format_version > FORMAT_VERSION {
        return Err(CheckpointError::Mismatch(format!(
            "unsupported checkpoint format_version {} (this build reads {MIN_FORMAT_VERSION}..={FORMAT_VERSION})",
            manifest.format_version
        )));
    }
    validate_manifest_field_matrix(&value, &manifest)?;
    Ok(manifest)
}

fn validate_manifest_field_matrix(
    value: &serde_json::Value,
    manifest: &CheckpointManifest,
) -> Result<(), CheckpointError> {
    let object = value.as_object().ok_or_else(|| {
        CheckpointError::Mismatch("checkpoint manifest must be a JSON object".into())
    })?;
    let present = |key: &str| object.get(key).is_some_and(|value| !value.is_null());
    let require = |key: &str| {
        if present(key) {
            Ok(())
        } else {
            Err(CheckpointError::Mismatch(format!(
                "checkpoint format v{} requires non-null manifest field {key}",
                manifest.format_version
            )))
        }
    };
    let forbid = |key: &str| {
        if present(key) {
            Err(CheckpointError::Mismatch(format!(
                "checkpoint format v{} forbids manifest field {key}",
                manifest.format_version
            )))
        } else {
            Ok(())
        }
    };

    match manifest.format_version {
        1 => {
            forbid("optimizer_step_t")?;
            forbid("optimizer_num_vars")?;
            forbid("sampler_state")?;
            forbid("rollout_ledger_continuation")?;
            forbid("ordinary_checkpoint")?;
        }
        2 => {
            require("optimizer_step_t")?;
            require("optimizer_num_vars")?;
            require("sampler_state")?;
            forbid("rollout_ledger_continuation")?;
            forbid("ordinary_checkpoint")?;
        }
        LEGACY_MOMENTUM_FORMAT_VERSION => {
            require("optimizer_step_t")?;
            require("optimizer_num_vars")?;
            require("sampler_state")?;
            forbid("ordinary_checkpoint")?;
        }
        FORMAT_VERSION => {
            require("optimizer_step_t")?;
            require("optimizer_num_vars")?;
            require("sampler_state")?;
            require("ordinary_checkpoint")?;
            forbid("rollout_ledger_continuation")?;
            if !object.contains_key("lora_recipe") {
                return Err(CheckpointError::Mismatch(
                    "checkpoint format v4 requires explicit lora_recipe (string or null)".into(),
                ));
            }
            let identity = manifest.ordinary_checkpoint.as_ref().ok_or_else(|| {
                CheckpointError::Mismatch(
                    "checkpoint format v4 is missing its ordinary identity".into(),
                )
            })?;
            if identity.kind != ORDINARY_CHECKPOINT_KIND {
                return Err(CheckpointError::Mismatch(format!(
                    "checkpoint format v4 kind {:?} is not {ORDINARY_CHECKPOINT_KIND:?}",
                    identity.kind
                )));
            }
            for (label, digest) in [
                (
                    "frozen_policy_sha256",
                    identity.frozen_policy_sha256.as_str(),
                ),
                (
                    "trainer_config_sha256",
                    identity.trainer_config_sha256.as_str(),
                ),
                (
                    "tensor_schema_sha256",
                    identity.tensor_schema_sha256.as_str(),
                ),
                ("adapter_sha256", identity.adapter_sha256.as_str()),
                ("optimizer_sha256", identity.optimizer_sha256.as_str()),
                ("sampler_sha256", identity.sampler_sha256.as_str()),
                (
                    "state_envelope_sha256",
                    identity.state_envelope_sha256.as_str(),
                ),
            ] {
                validate_sha256(label, digest)?;
            }
            validate_state_envelope(manifest, identity)?;
        }
        _ => unreachable!("version range was checked before the field matrix"),
    }
    Ok(())
}

fn prepare_adapter_tensors(bytes: &[u8], vars: &[Var]) -> Result<Vec<Tensor>, CheckpointError> {
    let loaded = candle_core::safetensors::load_buffer(bytes, &Device::Cpu)?;
    let expected = (0..vars.len()).map(var_key).collect::<BTreeSet<_>>();
    let actual = loaded.keys().cloned().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(CheckpointError::Mismatch(format!(
            "adapter tensor key set mismatch: expected {expected:?}, found {actual:?}"
        )));
    }

    // Pass 1 — validate and device-prepare EVERY tensor before mutating anything,
    // so a missing/mis-shaped/mis-typed tensor aborts with the model untouched
    // (a partial overwrite would silently corrupt the model).
    let mut prepared: Vec<Tensor> = Vec::with_capacity(vars.len());
    for (i, v) in vars.iter().enumerate() {
        let key = var_key(i);
        let t = loaded.get(&key).ok_or_else(|| {
            CheckpointError::Mismatch(format!("checkpoint is missing tensor {key}"))
        })?;
        let want = v.as_tensor();
        if t.dims() != want.dims() {
            return Err(CheckpointError::Mismatch(format!(
                "tensor {key}: checkpoint shape {:?} != model shape {:?}",
                t.dims(),
                want.dims()
            )));
        }
        if t.dtype() != want.dtype() {
            return Err(CheckpointError::Mismatch(format!(
                "tensor {key}: checkpoint dtype {:?} != model dtype {:?}",
                t.dtype(),
                want.dtype()
            )));
        }
        // `Var::set` requires the *destination* var be contiguous (freshly
        // allocated LoRA factors are) and copies from a same-device source. Move the
        // CPU-loaded tensor onto the var's device; shape and dtype are already
        // validated equal. (`contiguous()` on the source is belt-and-suspenders.)
        prepared.push(t.to_device(want.device())?.contiguous()?);
    }

    Ok(prepared)
}

fn apply_adapter_tensors(vars: &[Var], prepared: &[Tensor]) -> Result<(), CheckpointError> {
    for (v, t) in vars.iter().zip(prepared.iter()) {
        v.set(t)?;
    }
    Ok(())
}

/// The result of [`load_checkpoint`]: the resume step plus the validated optimizer,
/// sampler, and recipe state.
///
#[derive(Debug)]
pub struct LoadedCheckpoint {
    /// Completed optimizer steps — the `start_step` a resume continues from.
    pub step: u64,
    /// The optimizer moments + step counter. Public v4 loads always return `Some`;
    /// the option remains for the shared internal continuation representation.
    pub optimizer_state: Option<OptimizerState>,
    /// The opaque rollout-sampler RNG blob. Public v4 loads always return `Some`.
    pub sampler_state: Option<Vec<u8>>,
    /// The writing policy's canonical adapter-recipe string, if recorded (see
    /// [`CheckpointManifest::lora_recipe`]). Public v4 loading has already compared
    /// its exact presence/value with the caller's [`CheckpointBinding`] and included
    /// it in the ordered schema identity before this value is returned.
    pub lora_recipe: Option<String>,
}

/// Persist an identity-bound, momentum-faithful ordinary checkpoint (format v4).
///
/// Writes `adapter.safetensors`, `optimizer.safetensors`, and (last, as the commit
/// marker) `manifest.json`. The optimizer moments are keyed by parameter index; the
/// optimizer step counter, the `sampler_state` blob, and the `lora_recipe` string live
/// in the manifest. Each tensor is moved to the CPU and made contiguous first, so a
/// CUDA-resident run checkpoints the same way. Restored as a unit by
/// [`load_checkpoint`] + [`crate::Trainer::resume`].
///
/// `sampler_state` is the opaque blob from [`crate::Policy::sampler_state`]; it is stored
/// verbatim and only the policy interprets it on restore. `binding` supplies the
/// externally verified frozen-policy and canonical trainer identities plus the exact
/// recipe; this function derives and records the ordered schema and exact payload hashes.
///
/// The write is **crash-atomic** via the same stage-then-rename as
/// [`save_adapter`] (see the module docs).
///
/// # Errors
///
/// Returns [`CheckpointError`] if the staging directory cannot be created, a tensor
/// cannot be moved to CPU / serialized, the manifest cannot be written, or the final
/// rename fails.
pub fn save_checkpoint(
    dir: impl AsRef<Path>,
    vars: &[Var],
    opt_state: &OptimizerState,
    sampler_state: &[u8],
    step: u64,
    binding: &CheckpointBinding,
) -> Result<(), CheckpointError> {
    write_staged(dir.as_ref(), |stage| {
        write_checkpoint_contents(
            stage,
            &stage.join(MANIFEST_FILE),
            vars,
            opt_state,
            sampler_state,
            step,
            FORMAT_VERSION,
            binding.lora_recipe.clone(),
            Some(binding),
            None,
        )
    })
}

/// Persist a momentum-faithful checkpoint while atomically claiming a previously
/// absent destination directory.
///
/// Unlike [`save_checkpoint`], this never renames an existing destination aside.
/// Both writer flavors first claim the same persistent sibling advisory lock;
/// `create_dir` is then the no-replace claim, a synced private marker proves which
/// directory this writer owns, and an atomically renamed manifest is the
/// reader-visible commit marker. A process crash can leave an incomplete claimed
/// directory, which discovery skips and a later writer refuses to replace;
/// recovery must explicitly inspect and remove that abandoned claim. Ordinary
/// operation cleans and durably removes a failed pre-manifest claim only while its
/// marker still proves ownership. The checkpoint directory is synced after all
/// payload and temporary-manifest entries are created but before the manifest rename;
/// every successful post-manifest durability fence then verifies both ownership and
/// the visible adapter, optimizer, and manifest against the exact package
/// fingerprinted before publication. An unreconciled failure, ownership loss, or
/// package mismatch is returned as
/// [`CheckpointError::PublicationAmbiguous`] and must not be blindly overwritten.
pub(crate) fn save_checkpoint_no_replace(
    dir: impl AsRef<Path>,
    vars: &[Var],
    opt_state: &OptimizerState,
    sampler_state: &[u8],
    lora_recipe: Option<&str>,
    continuation: RolloutLedgerContinuationManifest,
) -> Result<(), CheckpointError> {
    let dir = dir.as_ref();
    let parent = dir.parent().ok_or_else(|| {
        io(
            dir,
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "checkpoint path has no parent directory",
            ),
        )
    })?;
    create_directory_all_durable(parent)?;
    // Ordinary adapter/checkpoint writers take this same destination lock. Keep
    // it alive through claim cleanup and every post-manifest verification.
    let _writer_lock = acquire_checkpoint_writer_lock(dir)?;
    std::fs::create_dir(dir).map_err(|error| io(dir, error))?;
    let ownership = ClaimOwnership::create(dir).map_err(|error| {
        CheckpointError::PublicationAmbiguous {
            path: dir.to_path_buf(),
            detail: format!(
                "{error}; claim owner marker could not be established, so the destination was preserved"
            ),
        }
    })?;
    if let Err(error) = sync_directory(dir) {
        return Err(cleanup_uncommitted_claim(dir, parent, &ownership, error));
    }
    // Anchor the no-replace claim before writing behind it. A crash may retain
    // an incomplete claim, but a successful return can never depend on an
    // uncommitted directory entry.
    if let Err(error) = sync_no_replace_boundary(parent, NoReplaceSyncPoint::ClaimParent) {
        return Err(cleanup_uncommitted_claim(dir, parent, &ownership, error));
    }
    let manifest_stage = dir.join(format!("{MANIFEST_FILE}.tmp-{}", std::process::id()));
    if let Err(error) = write_checkpoint_contents(
        dir,
        &manifest_stage,
        vars,
        opt_state,
        sampler_state,
        continuation.completed_step,
        LEGACY_MOMENTUM_FORMAT_VERSION,
        lora_recipe.map(str::to_owned),
        None,
        Some(continuation),
    ) {
        return Err(cleanup_uncommitted_claim(dir, parent, &ownership, error));
    }
    #[cfg(test)]
    run_no_replace_hook(dir, NoReplaceHookPoint::BeforeFingerprint);
    let expected_package = match checkpoint_package_fingerprint(dir, &manifest_stage) {
        Ok(fingerprint) => fingerprint,
        Err(error) => {
            return Err(cleanup_uncommitted_claim(dir, parent, &ownership, error));
        }
    };
    // Anchor the payload and temporary-manifest names before the manifest rename
    // can make this package discoverable. File syncs alone do not persist new
    // directory entries across a crash.
    if let Err(error) = sync_no_replace_boundary(dir, NoReplaceSyncPoint::PreManifestDirectory) {
        return Err(cleanup_uncommitted_claim(dir, parent, &ownership, error));
    }
    #[cfg(test)]
    run_no_replace_hook(dir, NoReplaceHookPoint::BeforeManifestRename);
    let manifest = dir.join(MANIFEST_FILE);
    if let Err(error) =
        std::fs::rename(&manifest_stage, &manifest).map_err(|error| io(&manifest_stage, error))
    {
        return Err(cleanup_uncommitted_claim(dir, parent, &ownership, error));
    }
    #[cfg(test)]
    PANIC_NO_REPLACE_AFTER_MANIFEST.with(|panic| {
        if panic.replace(false) {
            panic!("injected post-manifest continuation publication panic");
        }
    });
    #[cfg(test)]
    run_no_replace_hook(dir, NoReplaceHookPoint::BeforeFirstFence);
    match sync_no_replace_publication(dir, parent) {
        Ok(()) => verify_visible_owned_package(
            dir,
            &manifest,
            &ownership,
            &expected_package,
            "initial durability fence completed",
        ),
        Err(error) => {
            #[cfg(test)]
            run_no_replace_hook(dir, NoReplaceHookPoint::AfterFirstFenceFailure);
            match sync_no_replace_publication(dir, parent) {
                Ok(()) => verify_visible_owned_package(
                    dir,
                    &manifest,
                    &ownership,
                    &expected_package,
                    &format!("{error}; durability retry completed"),
                ),
                Err(retry_error) => Err(CheckpointError::PublicationAmbiguous {
                    path: dir.to_path_buf(),
                    detail: format!("{error}; durability retry failed: {retry_error}"),
                }),
            }
        }
    }
}

fn verify_visible_owned_package(
    dir: &Path,
    manifest_path: &Path,
    ownership: &ClaimOwnership,
    expected_package: &CheckpointPackageFingerprint,
    completed_fence: &str,
) -> Result<(), CheckpointError> {
    match ownership.matches(dir) {
        Ok(true) => {}
        Ok(false) => {
            return Err(CheckpointError::PublicationAmbiguous {
                path: dir.to_path_buf(),
                detail: format!(
                    "{completed_fence}, but destination ownership no longer matches the intended claim"
                ),
            });
        }
        Err(error) => {
            return Err(CheckpointError::PublicationAmbiguous {
                path: dir.to_path_buf(),
                detail: format!(
                    "{completed_fence}, but destination ownership verification failed: {error}"
                ),
            });
        }
    }
    match checkpoint_package_fingerprint(dir, manifest_path) {
        Ok(visible_package) if visible_package == *expected_package => Ok(()),
        Ok(_) => Err(CheckpointError::PublicationAmbiguous {
            path: dir.to_path_buf(),
            detail: format!(
                "{completed_fence}, but the visible package no longer matches the intended checkpoint"
            ),
        }),
        Err(error) => Err(CheckpointError::PublicationAmbiguous {
            path: dir.to_path_buf(),
            detail: format!(
                "{completed_fence}, but visible package verification failed: {error}"
            ),
        }),
    }
}

fn checkpoint_package_fingerprint(
    dir: &Path,
    manifest_path: &Path,
) -> Result<CheckpointPackageFingerprint, CheckpointError> {
    Ok(CheckpointPackageFingerprint {
        adapter: fingerprint_file(&dir.join(ADAPTER_FILE))?,
        optimizer: fingerprint_file(&dir.join(OPTIMIZER_FILE))?,
        manifest: fingerprint_file(manifest_path)?,
    })
}

fn fingerprint_file(path: &Path) -> Result<FileFingerprint, CheckpointError> {
    let mut file = std::fs::File::open(path).map_err(|error| io(path, error))?;
    let mut hasher = Sha256::new();
    let mut length = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| io(path, error))?;
        if read == 0 {
            break;
        }
        length = length.checked_add(read as u64).ok_or_else(|| {
            io(
                path,
                std::io::Error::other("checkpoint file length overflow while fingerprinting"),
            )
        })?;
        hasher.update(&buffer[..read]);
    }
    Ok(FileFingerprint {
        length,
        sha256: hasher.finalize().into(),
    })
}

#[cfg(test)]
fn run_no_replace_hook(dir: &Path, point: NoReplaceHookPoint) {
    let hook = NO_REPLACE_HOOK.with(|hook| {
        let mut hook = hook.borrow_mut();
        if hook
            .as_ref()
            .is_some_and(|(hook_point, _)| *hook_point == point)
        {
            hook.take().map(|(_, hook)| hook)
        } else {
            None
        }
    });
    if let Some(hook) = hook {
        hook(dir);
    }
}

fn cleanup_uncommitted_claim(
    dir: &Path,
    parent: &Path,
    ownership: &ClaimOwnership,
    original: CheckpointError,
) -> CheckpointError {
    match ownership.matches(dir) {
        Ok(true) => {}
        Ok(false) => {
            return CheckpointError::PublicationAmbiguous {
                path: dir.to_path_buf(),
                detail: format!(
                    "{original}; claim ownership was lost, so the current destination was preserved"
                ),
            };
        }
        Err(error) => {
            return CheckpointError::PublicationAmbiguous {
                path: dir.to_path_buf(),
                detail: format!(
                    "{original}; claim ownership could not be verified ({error}), so the current destination was preserved"
                ),
            };
        }
    }
    match std::fs::remove_dir_all(dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return CheckpointError::PublicationAmbiguous {
                path: dir.to_path_buf(),
                detail: format!("{original}; incomplete claim cleanup failed: {error}"),
            };
        }
    }
    match sync_directory(parent) {
        Ok(()) => original,
        Err(cleanup_error) => CheckpointError::PublicationAmbiguous {
            path: dir.to_path_buf(),
            detail: format!(
                "{original}; claim was removed but removal durability could not be confirmed: {cleanup_error}"
            ),
        },
    }
}

fn sync_no_replace_publication(dir: &Path, parent: &Path) -> Result<(), CheckpointError> {
    sync_no_replace_boundary(dir, NoReplaceSyncPoint::ManifestDirectory)?;
    sync_no_replace_boundary(parent, NoReplaceSyncPoint::ManifestParent)
}

fn sync_no_replace_boundary(path: &Path, point: NoReplaceSyncPoint) -> Result<(), CheckpointError> {
    let _ = point;
    #[cfg(test)]
    {
        let fail = FAIL_NO_REPLACE_SYNCS.with(|failures| {
            let mut failures = failures.borrow_mut();
            if failures.first() == Some(&point) {
                failures.remove(0);
                true
            } else {
                false
            }
        });
        if fail {
            return Err(io(
                path,
                std::io::Error::other(format!("injected {point:?} sync failure")),
            ));
        }
    }
    sync_directory(path)
}

#[allow(clippy::too_many_arguments)] // one explicit durable checkpoint payload + provenance tuple
fn write_checkpoint_contents(
    dir: &Path,
    manifest_path: &Path,
    vars: &[Var],
    opt_state: &OptimizerState,
    sampler_state: &[u8],
    step: u64,
    format_version: u32,
    lora_recipe: Option<String>,
    ordinary_binding: Option<&CheckpointBinding>,
    rollout_ledger_continuation: Option<RolloutLedgerContinuationManifest>,
) -> Result<(), CheckpointError> {
    // Adapter weights (identical to the v1 layout).
    let mut adapter: HashMap<String, Tensor> = HashMap::with_capacity(vars.len());
    for (i, v) in vars.iter().enumerate() {
        let t = v.as_tensor().to_device(&Device::Cpu)?.contiguous()?;
        adapter.insert(var_key(i), t);
    }
    let adapter_path = dir.join(ADAPTER_FILE);
    candle_core::safetensors::save(&adapter, &adapter_path)?;
    sync_file(&adapter_path)?;

    // Optimizer moments: `m.<i>` / `v.<i>`, CPU + contiguous.
    let n = opt_state.first_moments.len();
    if opt_state.second_moments.len() != n {
        return Err(CheckpointError::Mismatch(format!(
            "optimizer state has {n} first moments but {} second moments",
            opt_state.second_moments.len()
        )));
    }
    let mut moments: HashMap<String, Tensor> = HashMap::with_capacity(n * 2);
    for i in 0..n {
        let m = opt_state.first_moments[i]
            .to_device(&Device::Cpu)?
            .contiguous()?;
        let v = opt_state.second_moments[i]
            .to_device(&Device::Cpu)?
            .contiguous()?;
        moments.insert(moment_key("m", i), m);
        moments.insert(moment_key("v", i), v);
    }
    let optimizer_path = dir.join(OPTIMIZER_FILE);
    candle_core::safetensors::save(&moments, &optimizer_path)?;
    sync_file(&optimizer_path)?;

    let ordinary_checkpoint = if let Some(binding) = ordinary_binding {
        if format_version != FORMAT_VERSION || rollout_ledger_continuation.is_some() {
            return Err(CheckpointError::Mismatch(
                "ordinary checkpoint identity can only be written on a format-v4 ordinary package"
                    .into(),
            ));
        }
        if lora_recipe.as_deref() != binding.lora_recipe() {
            return Err(CheckpointError::Mismatch(
                "ordinary checkpoint recipe does not match its binding".into(),
            ));
        }
        let adapter_bytes =
            std::fs::read(&adapter_path).map_err(|error| io(&adapter_path, error))?;
        let optimizer_bytes =
            std::fs::read(&optimizer_path).map_err(|error| io(&optimizer_path, error))?;
        let tensor_schema_sha256 = tensor_schema_sha256(vars, binding.lora_recipe())?;
        let adapter_sha256 = adapter_payload_sha256(&adapter_bytes);
        let optimizer_sha256 = optimizer_payload_sha256(opt_state.step_t, &optimizer_bytes)?;
        let sampler_sha256 = sampler_payload_sha256(sampler_state);
        let state_envelope_sha256 = state_envelope_sha256(
            format_version,
            ORDINARY_CHECKPOINT_KIND,
            step,
            vars.len(),
            binding.lora_recipe(),
            opt_state.step_t,
            n,
            &binding.frozen_policy_sha256,
            &binding.trainer_config_sha256,
            &tensor_schema_sha256,
            &adapter_sha256,
            &optimizer_sha256,
            &sampler_sha256,
        )?;
        Some(OrdinaryCheckpointIdentity {
            kind: ORDINARY_CHECKPOINT_KIND.into(),
            frozen_policy_sha256: binding.frozen_policy_sha256.clone(),
            trainer_config_sha256: binding.trainer_config_sha256.clone(),
            tensor_schema_sha256,
            adapter_sha256,
            optimizer_sha256,
            sampler_sha256,
            state_envelope_sha256,
        })
    } else {
        if format_version == FORMAT_VERSION {
            return Err(CheckpointError::Mismatch(
                "format-v4 ordinary checkpoint write requires an identity binding".into(),
            ));
        }
        None
    };

    // Manifest LAST — either inside a hidden staging directory or under a
    // temporary name that the no-replace publisher atomically commits.
    let manifest = CheckpointManifest {
        format_version,
        step,
        num_vars: vars.len(),
        optimizer_step_t: Some(opt_state.step_t),
        optimizer_num_vars: Some(n),
        sampler_state: Some(sampler_state.to_vec()),
        lora_recipe,
        rollout_ledger_continuation,
        ordinary_checkpoint,
    };
    let json = serde_json::to_string_pretty(&manifest)?;
    let mut manifest_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(manifest_path)
        .map_err(|error| io(manifest_path, error))?;
    manifest_file
        .write_all(json.as_bytes())
        .map_err(|error| io(manifest_path, error))?;
    manifest_file
        .sync_all()
        .map_err(|error| io(manifest_path, error))?;
    Ok(())
}

fn sync_file(path: &Path) -> Result<(), CheckpointError> {
    std::fs::File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| io(path, error))
}

fn sync_directory(path: &Path) -> Result<(), CheckpointError> {
    // `Path::parent()` represents the ancestor of a one-component relative
    // path as `""`. Its durable filesystem directory is the current directory.
    let path = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    #[cfg(test)]
    FAIL_SYNC_DIRECTORY_ONCE.with(|failure| {
        if failure.borrow().as_deref() == Some(path) {
            failure.borrow_mut().take();
            return Err(io(
                path,
                std::io::Error::other("injected directory sync failure"),
            ));
        }
        Ok(())
    })?;
    std::fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| io(path, error))?;
    #[cfg(test)]
    SYNCED_DIRECTORIES.with(|paths| paths.borrow_mut().push(path.to_path_buf()));
    Ok(())
}

fn create_directory_all_durable(path: &Path) -> Result<(), CheckpointError> {
    if path.as_os_str().is_empty() {
        return sync_directory(Path::new("."));
    }
    let Some(parent) = path.parent() else {
        return match std::fs::metadata(path) {
            Ok(metadata) if metadata.is_dir() => sync_directory(path),
            Ok(_) => Err(io(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "checkpoint parent exists but is not a directory",
                ),
            )),
            Err(error) => Err(io(path, error)),
        };
    };
    create_directory_all_durable(parent)?;
    let mut missing = false;
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => {
            return Err(io(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "checkpoint parent exists but is not a directory",
                ),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => missing = true,
        Err(error) => return Err(io(path, error)),
    }
    if missing {
        match std::fs::create_dir(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && path.is_dir() => {}
            Err(error) => return Err(io(path, error)),
        }
    }
    sync_directory(path)?;
    sync_directory(parent)
}

/// A completely parsed and validated ordinary checkpoint whose live adapter
/// application has deliberately not happened yet.
///
/// [`crate::Trainer`] uses this split to coordinate every rank's identity, payload,
/// sampler, and temporary-Adam preflight before any rank mutates its policy.
#[derive(Debug)]
pub(crate) struct PreparedCheckpoint {
    step: u64,
    optimizer_state: OptimizerState,
    sampler_state: Vec<u8>,
    lora_recipe: Option<String>,
    adapter_tensors: Vec<Tensor>,
    manifest_consensus_sha256: String,
}

impl PreparedCheckpoint {
    pub(crate) fn step(&self) -> u64 {
        self.step
    }

    pub(crate) fn optimizer_state(&self) -> &OptimizerState {
        &self.optimizer_state
    }

    pub(crate) fn sampler_state(&self) -> &[u8] {
        &self.sampler_state
    }

    pub(crate) fn manifest_consensus_sha256(&self) -> &str {
        &self.manifest_consensus_sha256
    }

    pub(crate) fn apply(self, vars: &[Var]) -> Result<LoadedCheckpoint, CheckpointError> {
        apply_adapter_tensors(vars, &self.adapter_tensors)?;
        Ok(LoadedCheckpoint {
            step: self.step,
            optimizer_state: Some(self.optimizer_state),
            sampler_state: Some(self.sampler_state),
            lora_recipe: self.lora_recipe,
        })
    }
}

fn prepare_optimizer_state(
    bytes: &[u8],
    step_t: usize,
    num: usize,
    vars: &[Var],
) -> Result<OptimizerState, CheckpointError> {
    let expected_vars = vars
        .iter()
        .filter(|var| var.as_tensor().dtype().is_float())
        .collect::<Vec<_>>();
    if num != expected_vars.len() {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has {num} optimizer parameters but the model exposes {} float trainable vars",
            expected_vars.len()
        )));
    }
    let loaded = candle_core::safetensors::load_buffer(bytes, &Device::Cpu)?;
    let expected = (0..num)
        .flat_map(|index| [moment_key("m", index), moment_key("v", index)])
        .collect::<BTreeSet<_>>();
    let actual = loaded.keys().cloned().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(CheckpointError::Mismatch(format!(
            "optimizer tensor key set mismatch: expected {expected:?}, found {actual:?}"
        )));
    }
    let mut first_moments = Vec::with_capacity(num);
    let mut second_moments = Vec::with_capacity(num);
    for (index, var) in expected_vars.into_iter().enumerate() {
        let want = var.as_tensor();
        let mk = moment_key("m", index);
        let vk = moment_key("v", index);
        let m = loaded
            .get(&mk)
            .ok_or_else(|| CheckpointError::Mismatch(format!("missing optimizer tensor {mk}")))?;
        let v = loaded
            .get(&vk)
            .ok_or_else(|| CheckpointError::Mismatch(format!("missing optimizer tensor {vk}")))?;
        for (label, tensor) in [(mk.as_str(), m), (vk.as_str(), v)] {
            if tensor.dims() != want.dims() {
                return Err(CheckpointError::Mismatch(format!(
                    "optimizer tensor {label}: checkpoint shape {:?} != parameter shape {:?}",
                    tensor.dims(),
                    want.dims()
                )));
            }
            if tensor.dtype() != want.dtype() {
                return Err(CheckpointError::Mismatch(format!(
                    "optimizer tensor {label}: checkpoint dtype {:?} != parameter dtype {:?}",
                    tensor.dtype(),
                    want.dtype()
                )));
            }
        }
        first_moments.push(m.clone());
        second_moments.push(v.clone());
    }
    Ok(OptimizerState {
        step_t,
        first_moments,
        second_moments,
    })
}

fn read_payload(path: &Path) -> Result<Vec<u8>, CheckpointError> {
    std::fs::read(path).map_err(|error| io(path, error))
}

/// Parse and validate a format-v4 ordinary checkpoint without mutating `vars`.
pub(crate) fn prepare_checkpoint(
    dir: &Path,
    vars: &[Var],
    binding: &CheckpointBinding,
) -> Result<PreparedCheckpoint, CheckpointError> {
    let manifest = read_manifest(dir)?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(CheckpointError::Mismatch(format!(
            "Trainer resume requires an integrity-bound ordinary format-v4 checkpoint; found legacy format v{} (v1 remains adapter-only readable, while v2/v3 may be inspected for migration but are not trusted resume inputs)",
            manifest.format_version
        )));
    }
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let manifest_consensus_sha256 = domain_sha256(
        "ferrl.ordinary-checkpoint.manifest-consensus.v1",
        &[&manifest_bytes],
    );
    if manifest.lora_recipe.as_deref() != binding.lora_recipe() {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint adapter recipe {:?} does not match the policy's {:?}",
            manifest.lora_recipe,
            binding.lora_recipe()
        )));
    }
    let identity = manifest.ordinary_checkpoint.as_ref().ok_or_else(|| {
        CheckpointError::Mismatch("ordinary checkpoint identity is missing".into())
    })?;
    if identity.frozen_policy_sha256 != binding.frozen_policy_sha256 {
        return Err(CheckpointError::Mismatch(
            "ordinary checkpoint frozen-policy identity mismatch".into(),
        ));
    }
    if identity.trainer_config_sha256 != binding.trainer_config_sha256 {
        return Err(CheckpointError::Mismatch(
            "ordinary checkpoint learner-semantic trainer identity mismatch".into(),
        ));
    }
    if manifest.num_vars != vars.len() {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has {} tensors but the model exposes {} trainable vars",
            manifest.num_vars,
            vars.len()
        )));
    }
    let schema = tensor_schema_sha256(vars, binding.lora_recipe())?;
    if identity.tensor_schema_sha256 != schema {
        return Err(CheckpointError::Mismatch(
            "ordinary checkpoint ordered tensor-schema or recipe identity mismatch".into(),
        ));
    }
    let step_t = manifest.optimizer_step_t.ok_or_else(|| {
        CheckpointError::Mismatch("ordinary checkpoint is missing optimizer_step_t".into())
    })?;
    let optimizer_num_vars = manifest.optimizer_num_vars.ok_or_else(|| {
        CheckpointError::Mismatch("ordinary checkpoint is missing optimizer_num_vars".into())
    })?;
    let sampler_state = manifest.sampler_state.clone().ok_or_else(|| {
        CheckpointError::Mismatch("ordinary checkpoint is missing sampler_state".into())
    })?;
    validate_state_envelope(&manifest, identity)?;

    let adapter_path = dir.join(ADAPTER_FILE);
    let optimizer_path = dir.join(OPTIMIZER_FILE);
    let adapter_bytes = read_payload(&adapter_path)?;
    let optimizer_bytes = read_payload(&optimizer_path)?;
    if identity.adapter_sha256 != adapter_payload_sha256(&adapter_bytes) {
        return Err(CheckpointError::Mismatch(
            "ordinary checkpoint adapter payload digest mismatch".into(),
        ));
    }
    if identity.optimizer_sha256 != optimizer_payload_sha256(step_t, &optimizer_bytes)? {
        return Err(CheckpointError::Mismatch(
            "ordinary checkpoint Adam payload digest mismatch".into(),
        ));
    }
    if identity.sampler_sha256 != sampler_payload_sha256(&sampler_state) {
        return Err(CheckpointError::Mismatch(
            "ordinary checkpoint sampler payload digest mismatch".into(),
        ));
    }

    let adapter_tensors = prepare_adapter_tensors(&adapter_bytes, vars)?;
    let optimizer_state =
        prepare_optimizer_state(&optimizer_bytes, step_t, optimizer_num_vars, vars)?;
    Ok(PreparedCheckpoint {
        step: manifest.step,
        optimizer_state,
        sampler_state,
        lora_recipe: manifest.lora_recipe,
        adapter_tensors,
        manifest_consensus_sha256,
    })
}

/// Restore a format-v4 ordinary checkpoint after validating its complete binding
/// and all payloads before the first live adapter mutation.
///
/// # Errors
///
/// Returns [`CheckpointError::Mismatch`] if the format is not v4, the caller's
/// binding differs, a required field/key/schema is missing or extra, or any exact
/// payload digest differs. Every such rejection happens before live adapter mutation.
pub fn load_checkpoint(
    dir: impl AsRef<Path>,
    vars: &[Var],
    binding: &CheckpointBinding,
) -> Result<LoadedCheckpoint, CheckpointError> {
    prepare_checkpoint(dir.as_ref(), vars, binding)?.apply(vars)
}

/// Internal legacy-v3 restore used only by the separately identity-bound
/// rollout-ledger continuation path.
pub(crate) fn load_rollout_ledger_checkpoint(
    dir: &Path,
    vars: &[Var],
) -> Result<LoadedCheckpoint, CheckpointError> {
    let manifest = read_manifest(dir)?;
    if manifest.format_version != LEGACY_MOMENTUM_FORMAT_VERSION
        || manifest.rollout_ledger_continuation.is_none()
    {
        return Err(CheckpointError::Mismatch(
            "expected a format-v3 rollout-ledger continuation checkpoint".into(),
        ));
    }
    if manifest.num_vars != vars.len() {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has {} tensors but the model exposes {} trainable vars",
            manifest.num_vars,
            vars.len()
        )));
    }
    let adapter_bytes = read_payload(&dir.join(ADAPTER_FILE))?;
    let optimizer_bytes = read_payload(&dir.join(OPTIMIZER_FILE))?;
    let step_t = manifest.optimizer_step_t.ok_or_else(|| {
        CheckpointError::Mismatch("rollout-ledger checkpoint is missing optimizer_step_t".into())
    })?;
    let num = manifest.optimizer_num_vars.ok_or_else(|| {
        CheckpointError::Mismatch("rollout-ledger checkpoint is missing optimizer_num_vars".into())
    })?;
    let sampler_state = manifest.sampler_state.ok_or_else(|| {
        CheckpointError::Mismatch("rollout-ledger checkpoint is missing sampler_state".into())
    })?;
    let optimizer_state = prepare_optimizer_state(&optimizer_bytes, step_t, num, vars)?;
    let adapter_tensors = prepare_adapter_tensors(&adapter_bytes, vars)?;
    apply_adapter_tensors(vars, &adapter_tensors)?;
    Ok(LoadedCheckpoint {
        step: manifest.step,
        optimizer_state: Some(optimizer_state),
        sampler_state: Some(sampler_state),
        lora_recipe: manifest.lora_recipe,
    })
}

/// The newest complete checkpoint discovered under a `checkpoints/` directory:
/// its directory and the completed-step count its manifest records (the
/// `start_step` a resume continues from — see [`crate::Trainer::resume`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatestCheckpoint {
    /// The checkpoint directory (`<checkpoints_dir>/step-<n>`).
    pub dir: PathBuf,
    /// The manifest `step`: completed optimizer steps, the resume `start_step`.
    pub step: u64,
}

/// Find the newest **complete** checkpoint under `checkpoints_dir`, or `None` if
/// there is none — the discovery half of restart-on-preemption (a requeued job
/// calls this to learn where to resume; see [`crate::Trainer::resume_latest`]).
///
/// Scans only the immediate `step-<n>` subdirectories (the layout
/// [`crate::Trainer`] writes), reads each one's manifest, and returns the one
/// with the greatest completed-step count. Only integrity-bound ordinary v4 packages
/// are eligible; legacy formats and separated continuations are excluded. A readable
/// eligible manifest must agree exactly with its `step-<n>` directory.
///
/// A malformed exact `step-<digits>` final name newer than every readable ordinary
/// package is fail-closed: a non-directory entry, numeric suffix overflow, or
/// missing/unreadable/malformed manifest is ambiguous published-state corruption
/// and cannot silently fall back to an older step. A readable legacy ordinary
/// package or a deliberately different continuation kind remains ineligible but is
/// not corruption; a malformed older sibling cannot cause replay and is ignored.
/// Crash-leftover `.tmp-*` / `.old-*` siblings do not match the exact final-name
/// shape and are ignored, as is any unrelated entry.
///
/// # Errors
///
/// Returns [`CheckpointError`] if the directory listing fails, a numeric suffix
/// overflows, or the newest exact final-name candidate cannot be validated. A
/// **missing** `checkpoints_dir` is not an error — it returns `None` (a run that has
/// not checkpointed yet).
#[allow(clippy::cognitive_complexity)] // fail-closed classification of every exact final-name state
pub fn latest_checkpoint(
    checkpoints_dir: impl AsRef<Path>,
) -> Result<Option<LatestCheckpoint>, CheckpointError> {
    let dir = checkpoints_dir.as_ref();
    if !dir.exists() {
        return Ok(None);
    }
    let mut best: Option<LatestCheckpoint> = None;
    let mut malformed = Vec::<(u64, CheckpointError)>::new();
    for entry in std::fs::read_dir(dir).map_err(|e| io(dir, e))? {
        let entry = entry.map_err(|e| io(dir, e))?;
        let path = entry.path();
        // Match exactly `step-<digits>`: this excludes `.tmp-*` / `.old-*`
        // crash siblings (their names carry a dot-suffix after the digits) and
        // any foreign directory.
        let Some(step_digits) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("step-"))
            .filter(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
        else {
            continue;
        };
        let directory_step = step_digits.parse::<u64>().map_err(|_| {
            CheckpointError::Mismatch(format!(
                "ordinary checkpoint directory name step-{step_digits} exceeds the u64 step range"
            ))
        })?;
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => {
                malformed.push((directory_step, io(&path, error)));
                continue;
            }
        };
        if !file_type.is_dir() {
            malformed.push((
                directory_step,
                CheckpointError::Mismatch(format!(
                    "ordinary checkpoint final name step-{directory_step} is not a directory"
                )),
            ));
            continue;
        }
        // Atomic ordinary publication exposes only complete final directories.
        // Any unreadable or malformed exact final name is therefore ambiguous
        // corruption, never a valid partial-write stage.
        let manifest = match read_manifest(&path) {
            Ok(manifest) => manifest,
            Err(error) => {
                malformed.push((directory_step, error));
                continue;
            }
        };
        if manifest.format_version != FORMAT_VERSION
            || manifest.ordinary_checkpoint.is_none()
            || manifest.rollout_ledger_continuation.is_some()
        {
            continue;
        }
        if manifest.step != directory_step {
            malformed.push((
                directory_step,
                CheckpointError::Mismatch(format!(
                    "ordinary checkpoint directory step-{directory_step} contains manifest step {}",
                    manifest.step
                )),
            ));
            continue;
        }
        let candidate = LatestCheckpoint {
            dir: path,
            step: manifest.step,
        };
        if best.as_ref().is_none_or(|b| candidate.step > b.step) {
            best = Some(candidate);
        }
    }
    if let Some((malformed_step, error)) = malformed
        .into_iter()
        .max_by_key(|(directory_step, _)| *directory_step)
    {
        if best
            .as_ref()
            .is_none_or(|checkpoint| malformed_step > checkpoint.step)
        {
            return Err(error);
        }
    }
    Ok(best)
}

/// Find the newest complete *separated rollout-ledger continuation*.
///
/// Ordinary cadence/eval checkpoints and incomplete or unsupported continuation
/// claims are ignored. This prevents the separated roles from treating a generic
/// checkpoint with coincidentally present Adam/sampler fields as chain state.
pub(crate) fn latest_rollout_ledger_continuation(
    checkpoints_dir: impl AsRef<Path>,
) -> Result<Option<LatestCheckpoint>, CheckpointError> {
    let dir = checkpoints_dir.as_ref();
    if !dir.exists() {
        return Ok(None);
    }
    let mut best: Option<LatestCheckpoint> = None;
    for entry in std::fs::read_dir(dir).map_err(|error| io(dir, error))? {
        let entry = entry.map_err(|error| io(dir, error))?;
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let path = entry.path();
        let Some(directory_step) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("step-"))
            .filter(|rest| !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|rest| rest.parse::<u64>().ok())
        else {
            continue;
        };
        let Ok(manifest) = read_manifest(&path) else {
            continue;
        };
        let Some(continuation) = manifest.rollout_ledger_continuation else {
            continue;
        };
        let supported_topology = match continuation.format_version {
            1 => {
                continuation.world_size.is_none()
                    && continuation.tensor_parallel_world_size.is_none()
                    && continuation.tensor_parallel_layout.is_none()
            }
            2 => {
                continuation
                    .world_size
                    .is_some_and(|world_size| world_size > 0)
                    && continuation.tensor_parallel_world_size.is_none()
                    && continuation.tensor_parallel_layout.is_none()
            }
            ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION => {
                continuation
                    .world_size
                    .is_some_and(|world_size| world_size > 0)
                    && continuation
                        .tensor_parallel_world_size
                        .is_some_and(|world_size| world_size > 0)
                    && continuation.tensor_parallel_layout.as_deref()
                        == Some(ROLLOUT_LEDGER_CONTINUATION_LAYOUT)
            }
            _ => false,
        };
        if continuation.format_version < MIN_ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION
            || continuation.format_version > ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION
            || !supported_topology
            || continuation.kind != ROLLOUT_LEDGER_CONTINUATION_KIND
            || continuation.completed_step != manifest.step
            || directory_step != manifest.step
        {
            continue;
        }
        let candidate = LatestCheckpoint {
            dir: path,
            step: manifest.step,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.step > current.step)
        {
            best = Some(candidate);
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::RunDir;
    use candle_core::{DType, Tensor};

    /// A unique temp directory, removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            p.push(format!("ferrl-ckpt-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
            let _ = std::fs::remove_file(sibling_path_with_suffix(&self.0, WRITER_LOCK_SUFFIX));
        }
    }

    /// Two distinct vars of different shapes (mirrors a `LoRA` A `[rank,in]` and
    /// B `[out,rank]`), filled deterministically.
    fn make_vars() -> Vec<Var> {
        let a = Tensor::from_vec(
            (0..6).map(|i| i as f32).collect::<Vec<_>>(),
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let b = Tensor::from_vec(
            (0..8).map(|i| i as f32 * -0.5).collect::<Vec<_>>(),
            (4, 2),
            &Device::Cpu,
        )
        .unwrap();
        vec![Var::from_tensor(&a).unwrap(), Var::from_tensor(&b).unwrap()]
    }

    fn snapshot(vars: &[Var]) -> Vec<Vec<f32>> {
        vars.iter()
            .map(|v| {
                v.as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn save_then_load_round_trips_bit_exactly() {
        let tmp = TempDir::new("roundtrip");
        let vars = make_vars();
        let original = snapshot(&vars);
        save_adapter(tmp.path(), &vars, 7, None).unwrap();

        // Clobber the vars, then load them back.
        for v in &vars {
            let z = Tensor::zeros(v.as_tensor().dims(), DType::F32, &Device::Cpu).unwrap();
            v.set(&z).unwrap();
        }
        assert_ne!(snapshot(&vars), original, "clobber did not change the vars");

        let manifest = load_adapter(tmp.path(), &vars).unwrap();
        assert_eq!(manifest.step, 7);
        assert_eq!(manifest.num_vars, 2);
        assert_eq!(
            snapshot(&vars),
            original,
            "loaded adapter must equal the saved one bit-for-bit"
        );
    }

    #[test]
    fn load_writes_through_to_a_fresh_set_of_aliasing_vars() {
        // The save and load var slices are different Var instances of the same
        // shapes — load must populate the second slice (this is the resume/eval
        // path: a fresh model's trainable_vars()).
        let tmp = TempDir::new("alias");
        let src = make_vars();
        let saved = snapshot(&src);
        save_adapter(tmp.path(), &src, 3, None).unwrap();

        let dst = make_vars();
        for v in &dst {
            v.set(&Tensor::ones(v.as_tensor().dims(), DType::F32, &Device::Cpu).unwrap())
                .unwrap();
        }
        load_adapter(tmp.path(), &dst).unwrap();
        assert_eq!(snapshot(&dst), saved);
    }

    #[test]
    fn load_rejects_wrong_var_count() {
        let tmp = TempDir::new("count");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 1, None).unwrap();
        let just_one = vec![vars[0].clone()];
        let err = load_adapter(tmp.path(), &just_one).unwrap_err();
        assert!(matches!(err, CheckpointError::Mismatch(_)), "got {err:?}");
    }

    #[test]
    fn load_rejects_shape_mismatch() {
        let tmp = TempDir::new("shape");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 1, None).unwrap();
        // Same count, but the first var has a different shape.
        let bad = vec![
            Var::from_tensor(&Tensor::zeros((3, 3), DType::F32, &Device::Cpu).unwrap()).unwrap(),
            vars[1].clone(),
        ];
        let err = load_adapter(tmp.path(), &bad).unwrap_err();
        match err {
            CheckpointError::Mismatch(m) => assert!(m.contains("shape"), "{m}"),
            other => panic!("expected shape mismatch, got {other:?}"),
        }
    }

    #[test]
    fn load_does_not_partially_mutate_on_a_later_mismatch() {
        // var 0 is valid, var 1 is mis-shaped: load must reject WITHOUT having
        // overwritten var 0 (all-or-nothing — validate every tensor before any set).
        let tmp = TempDir::new("partial");
        let vars = make_vars(); // shapes [2,3] and [4,2]
        save_adapter(tmp.path(), &vars, 1, None).unwrap();

        let ones0 = Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap();
        let dst0 = Var::from_tensor(&ones0).unwrap();
        let dst1_wrong =
            Var::from_tensor(&Tensor::zeros((9, 9), DType::F32, &Device::Cpu).unwrap()).unwrap();
        let before0 = dst0
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let dst = vec![dst0.clone(), dst1_wrong];
        let err = load_adapter(tmp.path(), &dst).unwrap_err();
        assert!(matches!(err, CheckpointError::Mismatch(_)), "got {err:?}");

        let after0 = dst0
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            before0, after0,
            "var 0 was mutated despite a later mismatch"
        );
    }

    #[test]
    fn load_rejects_dtype_mismatch() {
        let tmp = TempDir::new("dtype");
        let vars = make_vars(); // F32
        save_adapter(tmp.path(), &vars, 1, None).unwrap();
        // Same count and shapes, but the first target var is F64.
        let a64 = Tensor::zeros((2, 3), DType::F64, &Device::Cpu).unwrap();
        let bad = vec![Var::from_tensor(&a64).unwrap(), vars[1].clone()];
        let err = load_adapter(tmp.path(), &bad).unwrap_err();
        match err {
            CheckpointError::Mismatch(m) => assert!(m.contains("dtype"), "{m}"),
            other => panic!("expected dtype mismatch, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_unknown_format_version() {
        let tmp = TempDir::new("version");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 1, None).unwrap();
        // Rewrite the manifest with a future format version.
        let bumped = CheckpointManifest {
            format_version: FORMAT_VERSION + 1,
            step: 1,
            num_vars: vars.len(),
            optimizer_step_t: None,
            optimizer_num_vars: None,
            sampler_state: None,
            lora_recipe: None,
            rollout_ledger_continuation: None,
            ordinary_checkpoint: None,
        };
        std::fs::write(
            tmp.path().join(MANIFEST_FILE),
            serde_json::to_string(&bumped).unwrap(),
        )
        .unwrap();
        let err = load_adapter(tmp.path(), &vars).unwrap_err();
        match err {
            CheckpointError::Mismatch(m) => assert!(m.contains("format_version"), "{m}"),
            other => panic!("expected version mismatch, got {other:?}"),
        }
    }

    #[test]
    fn load_missing_dir_is_an_io_error() {
        let tmp = TempDir::new("missing");
        let vars = make_vars();
        let err = load_adapter(tmp.path().join("nope"), &vars).unwrap_err();
        assert!(matches!(err, CheckpointError::Io { .. }), "got {err:?}");
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = CheckpointManifest {
            format_version: FORMAT_VERSION,
            step: 42,
            num_vars: 8,
            optimizer_step_t: Some(40),
            optimizer_num_vars: Some(8),
            sampler_state: Some(vec![1, 2, 3, 4]),
            lora_recipe: Some("attn:qv|mlp:-".to_string()),
            rollout_ledger_continuation: None,
            ordinary_checkpoint: Some(OrdinaryCheckpointIdentity {
                kind: ORDINARY_CHECKPOINT_KIND.into(),
                frozen_policy_sha256: "1".repeat(64),
                trainer_config_sha256: "2".repeat(64),
                tensor_schema_sha256: "3".repeat(64),
                adapter_sha256: "4".repeat(64),
                optimizer_sha256: "5".repeat(64),
                sampler_sha256: "6".repeat(64),
                state_envelope_sha256: "7".repeat(64),
            }),
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: CheckpointManifest = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
    }

    /// A v1 manifest (no optimizer/sampler fields on disk) still deserializes for
    /// the explicit adapter-only compatibility path.
    #[test]
    fn v1_manifest_without_v2_fields_deserializes() {
        let j = r#"{"format_version":1,"step":7,"num_vars":2}"#;
        let m: CheckpointManifest = serde_json::from_str(j).unwrap();
        assert_eq!(m.format_version, 1);
        assert_eq!(m.step, 7);
        assert_eq!(m.num_vars, 2);
        assert_eq!(m.optimizer_step_t, None);
        assert_eq!(m.optimizer_num_vars, None);
        assert_eq!(m.sampler_state, None);
    }

    /// Build an [`OptimizerState`] of two moment pairs matching `make_vars`' shapes,
    /// filled deterministically so a round-trip is checkable bit-for-bit.
    fn make_opt_state() -> OptimizerState {
        let m0 = Tensor::from_vec(
            (0..6).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let v0 = Tensor::from_vec(
            (0..6).map(|i| i as f32 * 0.2).collect::<Vec<_>>(),
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let m1 = Tensor::from_vec(
            (0..8).map(|i| i as f32 * 0.3).collect::<Vec<_>>(),
            (4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let v1 = Tensor::from_vec(
            (0..8).map(|i| i as f32 * 0.4).collect::<Vec<_>>(),
            (4, 2),
            &Device::Cpu,
        )
        .unwrap();
        OptimizerState {
            step_t: 11,
            first_moments: vec![m0, m1],
            second_moments: vec![v0, v1],
        }
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn ordinary_binding(recipe: Option<&str>) -> CheckpointBinding {
        CheckpointBinding::new("a".repeat(64), "b".repeat(64), recipe.map(str::to_owned)).unwrap()
    }

    fn continuation_manifest(step: u64) -> RolloutLedgerContinuationManifest {
        RolloutLedgerContinuationManifest {
            format_version: ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION,
            kind: ROLLOUT_LEDGER_CONTINUATION_KIND.to_owned(),
            world_size: Some(1),
            tensor_parallel_world_size: Some(1),
            tensor_parallel_layout: Some(ROLLOUT_LEDGER_CONTINUATION_LAYOUT.to_owned()),
            completed_step: step,
            policy_sha256: "a".repeat(64),
            trainer_config_sha256: "b".repeat(64),
            tensor_schema_sha256: "c".repeat(64),
            adapter_sha256: "d".repeat(64),
            optimizer_sha256: "e".repeat(64),
            sampler_sha256: "f".repeat(64),
            parent_lineage_sha256: "1".repeat(64),
            consumed_ledger_sha256: "3".repeat(64),
            lineage_sha256: "2".repeat(64),
        }
    }

    /// Zero every var (so a subsequent load has something to overwrite).
    fn clobber(vars: &[Var]) {
        for v in vars {
            v.set(&Tensor::zeros(v.as_tensor().dims(), DType::F32, &Device::Cpu).unwrap())
                .unwrap();
        }
    }

    fn set_no_replace_hook(point: NoReplaceHookPoint, hook: impl FnOnce(&Path) + 'static) {
        NO_REPLACE_HOOK.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(
                slot.is_none(),
                "a no-replace test hook is already installed"
            );
            *slot = Some((point, Box::new(hook)));
        });
    }

    fn stage_ordinary_replacement(checkpoints: &Path, label: &str) -> PathBuf {
        let replacement = checkpoints.join(format!("ordinary-{label}"));
        save_checkpoint(
            &replacement,
            &make_vars(),
            &make_opt_state(),
            &[9, 8, 7],
            1,
            &ordinary_binding(Some("ordinary-replacement")),
        )
        .unwrap();
        replacement
    }

    fn install_uncoordinated_replacement_hook(point: NoReplaceHookPoint, replacement: PathBuf) {
        set_no_replace_hook(point, move |dir| {
            // Bypass the new public-writer lock only at this deterministic test
            // seam. `commit_stage` is the ordinary writer's exact atomic
            // replacement primitive, so this models ownership loss after claim.
            commit_stage(&replacement, dir).unwrap();
        });
    }

    fn assert_ordinary_replacement_visible(checkpoints: &Path, destination: &Path) {
        assert_eq!(
            latest_rollout_ledger_continuation(checkpoints).unwrap(),
            None,
            "ordinary replacement must not be discovered as a continuation"
        );
        assert_eq!(
            latest_checkpoint(checkpoints).unwrap().unwrap(),
            LatestCheckpoint {
                dir: destination.to_path_buf(),
                step: 1,
            }
        );
        let loaded = load_checkpoint(
            destination,
            &make_vars(),
            &ordinary_binding(Some("ordinary-replacement")),
        )
        .unwrap();
        assert_eq!(loaded.step, 1);
        assert_eq!(loaded.sampler_state.as_deref(), Some([9, 8, 7].as_slice()));
        assert_eq!(loaded.lora_recipe.as_deref(), Some("ordinary-replacement"));
    }

    fn assert_replacement_preserved_at(point: NoReplaceHookPoint, label: &str) {
        let tmp = TempDir::new(label);
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        let replacement = stage_ordinary_replacement(&checkpoints, label);
        install_uncoordinated_replacement_hook(point, replacement);

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::PublicationAmbiguous { path, detail })
                if path == destination && detail.contains("preserved")
        ));
        assert_ordinary_replacement_visible(&checkpoints, &destination);
    }

    #[test]
    fn save_checkpoint_round_trips_the_adapter_and_step() {
        let tmp = TempDir::new("v4-adapter");
        let vars = make_vars();
        let adapter = snapshot(&vars);
        save_checkpoint(
            tmp.path(),
            &vars,
            &make_opt_state(),
            &[9u8, 8, 7],
            13,
            &ordinary_binding(None),
        )
        .unwrap();
        clobber(&vars);
        let loaded = load_checkpoint(tmp.path(), &vars, &ordinary_binding(None)).unwrap();
        assert_eq!(loaded.step, 13);
        assert_eq!(
            snapshot(&vars),
            adapter,
            "adapter must round-trip bit-for-bit"
        );
    }

    #[test]
    fn no_replace_checkpoint_has_exactly_one_race_winner() {
        let tmp = TempDir::new("no-replace-race");
        let destination = tmp.path().join("step-1");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let first_destination = destination.clone();
        let first_barrier = barrier.clone();
        let second_destination = destination.clone();
        let second_barrier = barrier.clone();
        let (first, second) = std::thread::scope(|scope| {
            let first = scope.spawn(move || {
                let vars = make_vars();
                first_barrier.wait();
                save_checkpoint_no_replace(
                    &first_destination,
                    &vars,
                    &make_opt_state(),
                    &[1],
                    None,
                    continuation_manifest(1),
                )
            });
            let second = scope.spawn(move || {
                let vars = make_vars();
                second_barrier.wait();
                save_checkpoint_no_replace(
                    &second_destination,
                    &vars,
                    &make_opt_state(),
                    &[2],
                    None,
                    continuation_manifest(1),
                )
            });
            (first.join().unwrap(), second.join().unwrap())
        });
        assert_ne!(
            first.is_ok(),
            second.is_ok(),
            "results: {first:?}, {second:?}"
        );

        let vars = make_vars();
        let loaded = load_rollout_ledger_checkpoint(&destination, &vars).unwrap();
        assert_eq!(loaded.step, 1);
        assert!(matches!(loaded.sampler_state.as_deref(), Some([1 | 2])));
    }

    #[test]
    fn ordinary_writer_cannot_replace_an_active_continuation_claim() {
        let tmp = TempDir::new("shared-writer-lock");
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        let blocked = std::rc::Rc::new(std::cell::Cell::new(false));
        let hook_blocked = blocked.clone();
        set_no_replace_hook(NoReplaceHookPoint::BeforeFingerprint, move |dir| {
            let error = save_checkpoint(
                dir,
                &make_vars(),
                &make_opt_state(),
                &[9, 8, 7],
                99,
                &ordinary_binding(Some("ordinary-replacement")),
            )
            .unwrap_err();
            hook_blocked.set(matches!(
                error,
                CheckpointError::Io { source, .. }
                    if source.kind() == std::io::ErrorKind::WouldBlock
            ));
        });

        save_checkpoint_no_replace(
            &destination,
            &make_vars(),
            &make_opt_state(),
            &[1],
            None,
            continuation_manifest(1),
        )
        .unwrap();

        assert!(
            blocked.get(),
            "ordinary writer did not share the destination lock"
        );
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints)
                .unwrap()
                .unwrap(),
            LatestCheckpoint {
                dir: destination,
                step: 1,
            }
        );
    }

    #[test]
    fn pre_fingerprint_ownership_loss_preserves_ordinary_replacement() {
        assert_replacement_preserved_at(
            NoReplaceHookPoint::BeforeFingerprint,
            "pre-fingerprint-replacement",
        );
    }

    #[test]
    fn pre_manifest_rename_ownership_loss_preserves_ordinary_replacement() {
        assert_replacement_preserved_at(
            NoReplaceHookPoint::BeforeManifestRename,
            "pre-manifest-rename-replacement",
        );
    }

    #[test]
    fn no_replace_checkpoint_durably_anchors_new_parent_chain() {
        let tmp = TempDir::new("no-replace-new-parent-chain");
        let first_parent = tmp.path().join("new-run");
        let checkpoints = first_parent.join("checkpoints");
        let destination = checkpoints.join("step-1");
        assert!(!first_parent.exists());
        SYNCED_DIRECTORIES.with(|paths| paths.borrow_mut().clear());

        save_checkpoint_no_replace(
            &destination,
            &make_vars(),
            &make_opt_state(),
            &[1],
            None,
            continuation_manifest(1),
        )
        .unwrap();

        let synced = SYNCED_DIRECTORIES.with(|paths| paths.borrow().clone());
        assert!(
            synced.contains(&tmp.path().to_path_buf()),
            "new-run entry was not anchored in its existing ancestor: {synced:?}"
        );
        assert!(
            synced.contains(&first_parent),
            "checkpoints entry was not anchored in new-run: {synced:?}"
        );
        assert!(
            synced.contains(&checkpoints),
            "step-1 claim was not anchored in checkpoints: {synced:?}"
        );
        load_rollout_ledger_checkpoint(&destination, &make_vars()).unwrap();
    }

    #[test]
    fn no_replace_checkpoint_durably_anchors_fresh_run_dir_chain() {
        let tmp = TempDir::new("no-replace-fresh-run-dir");
        let runs_root = tmp.path().join("runs");
        let run = RunDir::create(&runs_root, "fresh").unwrap();
        let checkpoints = run.checkpoints_dir();
        let destination = checkpoints.join("step-1");
        SYNCED_DIRECTORIES.with(|paths| paths.borrow_mut().clear());

        save_checkpoint_no_replace(
            &destination,
            &make_vars(),
            &make_opt_state(),
            &[1],
            None,
            continuation_manifest(1),
        )
        .unwrap();

        let synced = SYNCED_DIRECTORIES.with(|paths| paths.borrow().clone());
        for expected in [
            tmp.path(),
            runs_root.as_path(),
            run.root(),
            checkpoints.as_path(),
        ] {
            assert!(
                synced.iter().any(|path| path == expected),
                "fresh RunDir ancestor was not durably re-established: {expected:?}; synced={synced:?}"
            );
        }
    }

    #[test]
    fn no_replace_checkpoint_retries_sync_for_existing_ancestor_chain() {
        let tmp = TempDir::new("no-replace-existing-retry");
        let run = RunDir::create(tmp.path().join("runs"), "fresh").unwrap();
        let destination = run.checkpoints_dir().join("step-1");
        FAIL_SYNC_DIRECTORY_ONCE.with(|failure| {
            *failure.borrow_mut() = Some(run.root().to_path_buf());
        });

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::Io { .. })
        ));
        assert!(!destination.exists());

        save_checkpoint_no_replace(
            &destination,
            &make_vars(),
            &make_opt_state(),
            &[1],
            None,
            continuation_manifest(1),
        )
        .unwrap();
        load_rollout_ledger_checkpoint(&destination, &make_vars()).unwrap();
    }

    #[test]
    fn no_replace_checkpoint_post_claim_sync_failure_removes_claim_for_retry() {
        let tmp = TempDir::new("no-replace-post-claim-sync");
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        FAIL_NO_REPLACE_SYNCS.with(|failures| {
            *failures.borrow_mut() = vec![NoReplaceSyncPoint::ClaimParent];
        });

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::Io { .. })
        ));
        assert!(
            !destination.exists(),
            "failed claim sync left a retry-blocking directory"
        );
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints).unwrap(),
            None
        );

        save_checkpoint_no_replace(
            &destination,
            &make_vars(),
            &make_opt_state(),
            &[1],
            None,
            continuation_manifest(1),
        )
        .unwrap();
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints)
                .unwrap()
                .unwrap(),
            LatestCheckpoint {
                dir: destination,
                step: 1,
            }
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // pins cleanup, retry, discovery, and restore outcomes
    fn no_replace_checkpoint_pre_manifest_sync_failure_removes_claim_for_retry() {
        let tmp = TempDir::new("no-replace-pre-manifest-sync");
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        FAIL_NO_REPLACE_SYNCS.with(|failures| {
            *failures.borrow_mut() = vec![NoReplaceSyncPoint::PreManifestDirectory];
        });

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::Io { .. })
        ));
        assert!(
            !destination.exists(),
            "failed pre-manifest fence left a retry-blocking claim"
        );
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints).unwrap(),
            None
        );

        save_checkpoint_no_replace(
            &destination,
            &make_vars(),
            &make_opt_state(),
            &[1],
            None,
            continuation_manifest(1),
        )
        .unwrap();
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints)
                .unwrap()
                .unwrap(),
            LatestCheckpoint {
                dir: destination.clone(),
                step: 1,
            }
        );
        let loaded = load_rollout_ledger_checkpoint(&destination, &make_vars()).unwrap();
        assert_eq!(loaded.step, 1);
        assert_eq!(loaded.sampler_state.as_deref(), Some([1].as_slice()));
    }

    #[test]
    fn no_replace_checkpoint_reconciles_transient_post_manifest_sync_failures() {
        let tmp = TempDir::new("no-replace-transient-post-manifest");
        for (label, point) in [
            ("directory", NoReplaceSyncPoint::ManifestDirectory),
            ("parent", NoReplaceSyncPoint::ManifestParent),
        ] {
            let checkpoints = tmp.path().join(label);
            let destination = checkpoints.join("step-1");
            FAIL_NO_REPLACE_SYNCS.with(|failures| {
                *failures.borrow_mut() = vec![point];
            });

            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            )
            .unwrap();
            assert_eq!(
                latest_rollout_ledger_continuation(&checkpoints)
                    .unwrap()
                    .unwrap(),
                LatestCheckpoint {
                    dir: destination.clone(),
                    step: 1,
                },
                "{label} sync reconciliation did not publish a discoverable continuation"
            );
            assert!(matches!(
                save_checkpoint_no_replace(
                    &destination,
                    &make_vars(),
                    &make_opt_state(),
                    &[1],
                    None,
                    continuation_manifest(1),
                ),
                Err(CheckpointError::Io { source, .. })
                    if source.kind() == std::io::ErrorKind::AlreadyExists
            ));
        }
    }

    #[test]
    fn no_replace_checkpoint_first_fence_rejects_ordinary_replacement() {
        let tmp = TempDir::new("no-replace-first-fence-replacement");
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        let replacement = stage_ordinary_replacement(&checkpoints, "first-fence");
        install_uncoordinated_replacement_hook(NoReplaceHookPoint::BeforeFirstFence, replacement);

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::PublicationAmbiguous { path, detail })
                if path == destination && detail.contains("ownership")
        ));
        assert_ordinary_replacement_visible(&checkpoints, &destination);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // pins first-fence discovery, restore, and retry behavior
    fn no_replace_checkpoint_first_fence_rejects_manifest_disappearance() {
        let tmp = TempDir::new("no-replace-first-fence-missing-manifest");
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        set_no_replace_hook(NoReplaceHookPoint::BeforeFirstFence, |dir| {
            std::fs::rename(
                dir.join(MANIFEST_FILE),
                dir.join("manifest.json.disappeared"),
            )
            .unwrap();
        });

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::PublicationAmbiguous { path, detail })
                if path == destination && detail.contains("visible package verification failed")
        ));
        assert!(matches!(
            latest_checkpoint(&checkpoints),
            Err(CheckpointError::Io { path, .. })
                if path == destination.join(MANIFEST_FILE)
        ));
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints).unwrap(),
            None
        );
        assert!(matches!(
            load_rollout_ledger_checkpoint(&destination, &make_vars()),
            Err(CheckpointError::Io { .. })
        ));
        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::AlreadyExists
        ));
    }

    #[test]
    fn no_replace_checkpoint_recovery_rejects_ordinary_replacement() {
        let tmp = TempDir::new("no-replace-recovery-replacement");
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        let replacement = stage_ordinary_replacement(&checkpoints, "recovery");
        FAIL_NO_REPLACE_SYNCS.with(|failures| {
            *failures.borrow_mut() = vec![NoReplaceSyncPoint::ManifestDirectory];
        });
        install_uncoordinated_replacement_hook(
            NoReplaceHookPoint::AfterFirstFenceFailure,
            replacement,
        );

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::PublicationAmbiguous { path, detail })
                if path == destination && detail.contains("ownership")
        ));
        assert_ordinary_replacement_visible(&checkpoints, &destination);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // pins ambiguous discovery, restore, and retry behavior
    fn no_replace_checkpoint_recovery_marks_manifest_disappearance_ambiguous() {
        let tmp = TempDir::new("no-replace-recovery-missing-manifest");
        let checkpoints = tmp.path().join("checkpoints");
        let destination = checkpoints.join("step-1");
        FAIL_NO_REPLACE_SYNCS.with(|failures| {
            *failures.borrow_mut() = vec![NoReplaceSyncPoint::ManifestDirectory];
        });
        set_no_replace_hook(NoReplaceHookPoint::AfterFirstFenceFailure, |dir| {
            std::fs::rename(
                dir.join(MANIFEST_FILE),
                dir.join("manifest.json.disappeared"),
            )
            .unwrap();
        });

        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::PublicationAmbiguous { path, detail })
                if path == destination && detail.contains("visible package verification failed")
        ));
        assert!(matches!(
            latest_checkpoint(&checkpoints),
            Err(CheckpointError::Io { path, .. })
                if path == destination.join(MANIFEST_FILE)
        ));
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints).unwrap(),
            None
        );
        assert!(matches!(
            load_rollout_ledger_checkpoint(&destination, &make_vars()),
            Err(CheckpointError::Io { .. })
        ));
        assert!(matches!(
            save_checkpoint_no_replace(
                &destination,
                &make_vars(),
                &make_opt_state(),
                &[1],
                None,
                continuation_manifest(1),
            ),
            Err(CheckpointError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::AlreadyExists
        ));
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // table pins both persistent sync boundaries
    fn no_replace_checkpoint_marks_persistent_post_manifest_sync_ambiguous() {
        let tmp = TempDir::new("no-replace-persistent-post-manifest");
        for (label, point) in [
            ("directory", NoReplaceSyncPoint::ManifestDirectory),
            ("parent", NoReplaceSyncPoint::ManifestParent),
        ] {
            let checkpoints = tmp.path().join(label);
            let destination = checkpoints.join("step-1");
            FAIL_NO_REPLACE_SYNCS.with(|failures| {
                *failures.borrow_mut() = vec![point, point];
            });

            assert!(matches!(
                save_checkpoint_no_replace(
                    &destination,
                    &make_vars(),
                    &make_opt_state(),
                    &[1],
                    None,
                    continuation_manifest(1),
                ),
                Err(CheckpointError::PublicationAmbiguous { path, detail })
                    if path == destination && detail.contains("durability retry failed")
            ));
            assert_eq!(
                latest_rollout_ledger_continuation(&checkpoints)
                    .unwrap()
                    .unwrap(),
                LatestCheckpoint {
                    dir: destination.clone(),
                    step: 1,
                },
                "{label} persistent sync failure hid a committed continuation"
            );
            assert!(matches!(
                save_checkpoint_no_replace(
                    &destination,
                    &make_vars(),
                    &make_opt_state(),
                    &[1],
                    None,
                    continuation_manifest(1),
                ),
                Err(CheckpointError::Io { source, .. })
                    if source.kind() == std::io::ErrorKind::AlreadyExists
            ));
        }
    }

    #[test]
    fn save_checkpoint_round_trips_the_optimizer_and_sampler() {
        let tmp = TempDir::new("v4-opt-sampler");
        let vars = make_vars();
        let opt = make_opt_state();
        let sampler_blob = vec![9u8, 8, 7, 6, 5];
        save_checkpoint(
            tmp.path(),
            &vars,
            &opt,
            &sampler_blob,
            13,
            &ordinary_binding(Some("attn:qkvo|mlp:gud|gdn:-")),
        )
        .unwrap();
        let loaded = load_checkpoint(
            tmp.path(),
            &vars,
            &ordinary_binding(Some("attn:qkvo|mlp:gud|gdn:-")),
        )
        .unwrap();
        let os = loaded
            .optimizer_state
            .expect("v4 must carry optimizer state");
        assert_eq!(os.step_t, 11);
        assert_eq!(flat(&os.first_moments[0]), flat(&opt.first_moments[0]));
        assert_eq!(flat(&os.second_moments[1]), flat(&opt.second_moments[1]));
        assert_eq!(
            loaded.sampler_state,
            Some(sampler_blob),
            "sampler blob must round-trip verbatim"
        );
    }

    #[test]
    fn save_is_staged_and_replaces_a_prior_checkpoint_as_a_unit() {
        let tmp = TempDir::new("atomic");
        let dir = tmp.path().join("step-5");
        let vars = make_vars();

        // Seed the published path with junk simulating a stale/partial write.
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("garbage.bin"), b"partial").unwrap();
        // And a stale stage from a "dead" prior attempt of this same pid.
        let stage = dir.with_file_name(format!("step-5.tmp-{}", std::process::id()));
        std::fs::create_dir_all(&stage).unwrap();
        std::fs::write(stage.join("leftover"), b"x").unwrap();

        save_checkpoint(
            &dir,
            &vars,
            &make_opt_state(),
            &[1u8],
            5,
            &ordinary_binding(None),
        )
        .unwrap();

        // The publish replaced the prior directory as a unit: the junk is gone,
        // the checkpoint is complete and loadable, and no stage dir remains.
        assert!(!dir.join("garbage.bin").exists(), "stale content survived");
        assert!(!stage.exists(), "stage dir must not survive a commit");
        let loaded = load_checkpoint(&dir, &vars, &ordinary_binding(None)).unwrap();
        assert_eq!(loaded.step, 5);

        // Re-writing the same path succeeds and stays complete (the in-place
        // overwrite path a periodic checkpointer hits on a re-run).
        save_checkpoint(
            &dir,
            &vars,
            &make_opt_state(),
            &[2u8],
            5,
            &ordinary_binding(None),
        )
        .unwrap();
        let again = load_checkpoint(&dir, &vars, &ordinary_binding(None)).unwrap();
        assert_eq!(again.sampler_state, Some(vec![2u8]));
    }

    #[test]
    fn failed_staged_write_leaves_the_prior_checkpoint_intact() {
        // The Err branch of write_staged: a mid-write failure must leave the
        // previously published checkpoint untouched and clean up its stage.
        let tmp = TempDir::new("staged-err");
        let dir = tmp.path().join("step-3");
        let vars = make_vars();
        save_checkpoint(
            &dir,
            &vars,
            &make_opt_state(),
            &[7u8],
            3,
            &ordinary_binding(None),
        )
        .unwrap();
        let before = snapshot(&vars);

        let err = write_staged(&dir, |stage| {
            // Leave a partial artifact in the stage, then fail.
            std::fs::write(stage.join("partial.bin"), b"half").unwrap();
            Err(CheckpointError::Mismatch("synthetic failure".into()))
        })
        .unwrap_err();
        assert!(matches!(err, CheckpointError::Mismatch(_)), "got {err:?}");

        // Prior checkpoint still loads bit-identically; no stage litter.
        clobber(&vars);
        let loaded = load_checkpoint(&dir, &vars, &ordinary_binding(None)).unwrap();
        assert_eq!(loaded.step, 3);
        assert_eq!(loaded.sampler_state, Some(vec![7u8]));
        assert_eq!(snapshot(&vars), before);
        let stage = dir.with_file_name(format!("step-3.tmp-{}", std::process::id()));
        assert!(!stage.exists(), "failed stage must be cleaned up");
    }

    #[test]
    fn lora_recipe_round_trips_and_defaults_to_none() {
        let tmp = TempDir::new("recipe");
        let vars = make_vars();
        let dir_v2 = tmp.path().join("v2");
        save_checkpoint(
            &dir_v2,
            &vars,
            &make_opt_state(),
            &[0u8],
            3,
            &ordinary_binding(Some("attn:qkvo|mlp:gud|gdn:-")),
        )
        .unwrap();
        let manifest = read_manifest(&dir_v2).unwrap();
        assert_eq!(
            manifest.lora_recipe.as_deref(),
            Some("attn:qkvo|mlp:gud|gdn:-")
        );

        // v1 adapter path records it too when given...
        let dir_v1 = tmp.path().join("v1");
        save_adapter(&dir_v1, &vars, 1, Some("attn:qv|mlp:-")).unwrap();
        let m1 = load_adapter(&dir_v1, &vars).unwrap();
        assert_eq!(m1.lora_recipe.as_deref(), Some("attn:qv|mlp:-"));

        // ...and a manifest without the field (pre-R1 checkpoint) loads as None.
        let j = r#"{"format_version":1,"step":7,"num_vars":2}"#;
        let m: CheckpointManifest = serde_json::from_str(j).unwrap();
        assert_eq!(m.lora_recipe, None);
    }

    #[test]
    fn load_checkpoint_rejects_v1_adapter_only_without_mutation() {
        let tmp = TempDir::new("v1-via-checkpoint");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 4, None).unwrap();
        let before = snapshot(&vars);
        let error = load_checkpoint(tmp.path(), &vars, &ordinary_binding(None)).unwrap_err();
        assert!(error.to_string().contains("format-v4"), "{error}");
        assert_eq!(snapshot(&vars), before);
    }

    fn write_manifest_value(dir: &Path, value: &serde_json::Value) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(value).unwrap(),
        )
        .unwrap();
    }

    fn write_manifest_raw(dir: &Path, raw: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(MANIFEST_FILE), raw).unwrap();
    }

    fn refresh_v4_state_envelope(manifest: &mut serde_json::Value) {
        let identity = &manifest["ordinary_checkpoint"];
        let root = state_envelope_sha256(
            u32::try_from(manifest["format_version"].as_u64().unwrap()).unwrap(),
            identity["kind"].as_str().unwrap(),
            manifest["step"].as_u64().unwrap(),
            usize::try_from(manifest["num_vars"].as_u64().unwrap()).unwrap(),
            manifest["lora_recipe"].as_str(),
            usize::try_from(manifest["optimizer_step_t"].as_u64().unwrap()).unwrap(),
            usize::try_from(manifest["optimizer_num_vars"].as_u64().unwrap()).unwrap(),
            identity["frozen_policy_sha256"].as_str().unwrap(),
            identity["trainer_config_sha256"].as_str().unwrap(),
            identity["tensor_schema_sha256"].as_str().unwrap(),
            identity["adapter_sha256"].as_str().unwrap(),
            identity["optimizer_sha256"].as_str().unwrap(),
            identity["sampler_sha256"].as_str().unwrap(),
        )
        .unwrap();
        manifest["ordinary_checkpoint"]["state_envelope_sha256"] = serde_json::json!(root);
    }

    fn assert_manifest_rejected(dir: &Path, value: &serde_json::Value, label: &str) {
        write_manifest_value(dir, value);
        assert!(
            read_manifest(dir).is_err(),
            "invalid manifest case {label} was accepted: {value}"
        );
    }

    fn assert_v1_manifest_matrix(dir: &Path) -> serde_json::Value {
        let v1 = serde_json::json!({
            "format_version": 1,
            "step": 1,
            "num_vars": 2,
        });
        write_manifest_value(dir, &v1);
        assert_eq!(read_manifest(dir).unwrap().format_version, 1);

        for (field, value) in [
            ("optimizer_step_t", serde_json::json!(2)),
            ("optimizer_num_vars", serde_json::json!(2)),
            ("sampler_state", serde_json::json!([1, 2, 3])),
        ] {
            let mut smuggled = v1.clone();
            smuggled[field] = value;
            assert_manifest_rejected(dir, &smuggled, &format!("v1 smuggled {field}"));
        }
        v1
    }

    fn assert_legacy_momentum_manifest_matrix(dir: &Path) {
        for version in [2, LEGACY_MOMENTUM_FORMAT_VERSION] {
            let complete = serde_json::json!({
                "format_version": version,
                "step": 1,
                "num_vars": 2,
                "optimizer_step_t": 3,
                "optimizer_num_vars": 2,
                "sampler_state": [1, 2, 3],
            });
            write_manifest_value(dir, &complete);
            assert_eq!(read_manifest(dir).unwrap().format_version, version);
            for field in ["optimizer_step_t", "optimizer_num_vars", "sampler_state"] {
                let mut missing = complete.clone();
                missing.as_object_mut().unwrap().remove(field);
                assert_manifest_rejected(dir, &missing, &format!("v{version} missing {field}"));
                let mut null = complete.clone();
                null[field] = serde_json::Value::Null;
                assert_manifest_rejected(dir, &null, &format!("v{version} null {field}"));
            }
        }
    }

    fn save_v4_manifest_fixture(dir: &Path) -> serde_json::Value {
        let vars = make_vars();
        save_checkpoint(
            dir,
            &vars,
            &make_opt_state(),
            &[1, 2, 3],
            1,
            &ordinary_binding(Some("recipe")),
        )
        .unwrap();
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap()
    }

    fn assert_v4_required_fields(dir: &Path, v4: &serde_json::Value) {
        for field in [
            "optimizer_step_t",
            "optimizer_num_vars",
            "sampler_state",
            "ordinary_checkpoint",
            "lora_recipe",
        ] {
            let mut missing = v4.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert_manifest_rejected(dir, &missing, &format!("v4 missing {field}"));
        }
        for field in [
            "kind",
            "frozen_policy_sha256",
            "trainer_config_sha256",
            "tensor_schema_sha256",
            "adapter_sha256",
            "optimizer_sha256",
            "sampler_sha256",
            "state_envelope_sha256",
        ] {
            let mut missing = v4.clone();
            missing["ordinary_checkpoint"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            assert_manifest_rejected(dir, &missing, &format!("v4 identity missing {field}"));
        }
    }

    fn assert_v4_identity_and_kind_controls(dir: &Path, v4: &serde_json::Value) {
        let mut malformed_digest = v4.clone();
        malformed_digest["ordinary_checkpoint"]["adapter_sha256"] =
            serde_json::json!("A".repeat(64));
        assert_manifest_rejected(dir, &malformed_digest, "v4 uppercase digest");
        let mut wrong_kind = v4.clone();
        wrong_kind["ordinary_checkpoint"]["kind"] = serde_json::json!("rollout_ledger");
        assert_manifest_rejected(dir, &wrong_kind, "v4 wrong kind");
        let mut continuation = v4.clone();
        continuation["rollout_ledger_continuation"] =
            serde_json::to_value(continuation_manifest(1)).unwrap();
        assert_manifest_rejected(dir, &continuation, "v4 continuation smuggling");
        let mut nested_unknown = v4.clone();
        nested_unknown["ordinary_checkpoint"]["typo_field"] = serde_json::json!(true);
        assert_manifest_rejected(dir, &nested_unknown, "unknown ordinary identity field");
        let mut unknown = v4.clone();
        unknown["typo_field"] = serde_json::json!(true);
        assert_manifest_rejected(dir, &unknown, "unknown top-level field");
    }

    fn assert_cross_version_identity_smuggling_rejected(
        dir: &Path,
        v1: &serde_json::Value,
        v4: &serde_json::Value,
    ) {
        let mut v1_ordinary = v1.clone();
        v1_ordinary["ordinary_checkpoint"] = v4["ordinary_checkpoint"].clone();
        assert_manifest_rejected(dir, &v1_ordinary, "v1 ordinary identity smuggling");
        let mut v1_continuation = v1.clone();
        v1_continuation["rollout_ledger_continuation"] =
            serde_json::to_value(continuation_manifest(1)).unwrap();
        assert_manifest_rejected(dir, &v1_continuation, "v1 continuation smuggling");

        let mut v2_continuation = serde_json::json!({
            "format_version": 2,
            "step": 1,
            "num_vars": 2,
            "optimizer_step_t": 3,
            "optimizer_num_vars": 2,
            "sampler_state": [1, 2, 3],
        });
        v2_continuation["rollout_ledger_continuation"] =
            serde_json::to_value(continuation_manifest(1)).unwrap();
        assert_manifest_rejected(dir, &v2_continuation, "v2 continuation smuggling");

        let mut v3_ordinary = serde_json::json!({
            "format_version": LEGACY_MOMENTUM_FORMAT_VERSION,
            "step": 1,
            "num_vars": 2,
            "optimizer_step_t": 3,
            "optimizer_num_vars": 2,
            "sampler_state": [1, 2, 3],
        });
        v3_ordinary["ordinary_checkpoint"] = v4["ordinary_checkpoint"].clone();
        assert_manifest_rejected(dir, &v3_ordinary, "v3 ordinary identity smuggling");
    }

    #[test]
    fn ordinary_manifest_required_field_matrix_is_version_exact() {
        let tmp = TempDir::new("field-matrix");
        let dir = tmp.path().join("case");
        let v1 = assert_v1_manifest_matrix(&dir);
        assert_legacy_momentum_manifest_matrix(&dir);
        let v4 = save_v4_manifest_fixture(&dir);
        assert_v4_required_fields(&dir, &v4);
        assert_v4_identity_and_kind_controls(&dir, &v4);
        assert_cross_version_identity_smuggling_rejected(&dir, &v1, &v4);
    }

    #[test]
    fn ordinary_manifest_rejects_raw_duplicate_security_fields() {
        let tmp = TempDir::new("duplicate-manifest-fields");
        let dir = tmp.path().join("case");
        let manifest = save_v4_manifest_fixture(&dir);
        let raw = serde_json::to_string_pretty(&manifest).unwrap();
        let zero_digest = "0".repeat(64);
        let mut cases = vec![
            (
                "top-level format_version".to_string(),
                raw.replacen(
                    "\"format_version\": 4,",
                    "\"format_version\": 4,\n  \"format_version\": 4,",
                    1,
                ),
            ),
            (
                "top-level step".to_string(),
                raw.replacen("\"step\": 1\n", "\"step\": 1,\n  \"step\": 1\n", 1),
            ),
            (
                "top-level identity".to_string(),
                raw.replacen(
                    "\"ordinary_checkpoint\": {",
                    "\"ordinary_checkpoint\": null,\n  \"ordinary_checkpoint\": {",
                    1,
                ),
            ),
            (
                "nested kind".to_string(),
                raw.replacen(
                    "\"kind\": \"ordinary\",",
                    "\"kind\": \"ordinary\",\n    \"kind\": \"ordinary\",",
                    1,
                ),
            ),
        ];
        for field in [
            "frozen_policy_sha256",
            "trainer_config_sha256",
            "tensor_schema_sha256",
            "adapter_sha256",
            "optimizer_sha256",
            "sampler_sha256",
            "state_envelope_sha256",
        ] {
            cases.push((
                format!("nested digest {field}"),
                raw.replacen(
                    &format!("\"{field}\":"),
                    &format!("\"{field}\": \"{zero_digest}\",\n    \"{field}\":"),
                    1,
                ),
            ));
        }
        for (label, duplicate) in cases {
            assert_ne!(duplicate, raw, "duplicate control did not alter {label}");
            write_manifest_raw(&dir, &duplicate);
            let error = read_manifest(&dir).unwrap_err();
            assert!(
                error.to_string().contains("duplicate field"),
                "{label}: wrong rejection: {error}"
            );
        }
    }

    fn save_integrity_fixture(dir: &Path) {
        save_checkpoint(
            dir,
            &make_vars(),
            &make_opt_state(),
            &[9, 8, 7, 6],
            13,
            &ordinary_binding(Some("recipe")),
        )
        .unwrap();
    }

    fn sentinel_vars() -> Vec<Var> {
        let vars = make_vars();
        for var in &vars {
            var.set(
                &Tensor::ones(var.as_tensor().dims(), DType::F32, &Device::Cpu)
                    .unwrap()
                    .affine(17.0, 0.0)
                    .unwrap(),
            )
            .unwrap();
        }
        vars
    }

    fn rewrite_first_tensor(path: &Path, key: &str) {
        let bytes = std::fs::read(path).unwrap();
        let mut tensors = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu).unwrap();
        let changed = tensors.get(key).unwrap().affine(1.0, 1.0).unwrap();
        tensors.insert(key.to_owned(), changed);
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    #[test]
    fn ordinary_checkpoint_state_envelope_rejects_step_relabelling_before_mutation() {
        let tmp = TempDir::new("step-relabel");
        save_integrity_fixture(tmp.path());
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(tmp.path().join(MANIFEST_FILE)).unwrap())
                .unwrap();
        manifest["step"] = serde_json::json!(14);
        write_manifest_value(tmp.path(), &manifest);

        let destination = sentinel_vars();
        let before = snapshot(&destination);
        let error = load_checkpoint(tmp.path(), &destination, &ordinary_binding(Some("recipe")))
            .unwrap_err();
        assert!(error.to_string().contains("state-envelope"), "{error}");
        assert_eq!(snapshot(&destination), before);
    }

    #[test]
    fn ordinary_checkpoint_state_envelope_rejects_coherent_leaf_transplant() {
        let tmp = TempDir::new("coherent-leaf-transplant");
        let recipient = tmp.path().join("recipient");
        let donor = tmp.path().join("donor");
        save_integrity_fixture(&recipient);

        let donor_vars = make_vars();
        donor_vars[0]
            .set(&donor_vars[0].as_tensor().affine(1.0, 11.0).unwrap())
            .unwrap();
        let mut donor_optimizer = make_opt_state();
        donor_optimizer.first_moments[0] =
            donor_optimizer.first_moments[0].affine(1.0, 5.0).unwrap();
        save_checkpoint(
            &donor,
            &donor_vars,
            &donor_optimizer,
            &[4, 3, 2, 1],
            13,
            &ordinary_binding(Some("recipe")),
        )
        .unwrap();

        std::fs::copy(donor.join(ADAPTER_FILE), recipient.join(ADAPTER_FILE)).unwrap();
        let donor_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(donor.join(MANIFEST_FILE)).unwrap()).unwrap();
        let mut recipient_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(recipient.join(MANIFEST_FILE)).unwrap()).unwrap();
        assert_ne!(
            recipient_manifest["ordinary_checkpoint"]["optimizer_sha256"],
            donor_manifest["ordinary_checkpoint"]["optimizer_sha256"],
            "the transplant control requires the recipient Adam state to remain distinct"
        );
        assert_ne!(
            recipient_manifest["ordinary_checkpoint"]["sampler_sha256"],
            donor_manifest["ordinary_checkpoint"]["sampler_sha256"],
            "the transplant control requires the recipient sampler state to remain distinct"
        );
        recipient_manifest["ordinary_checkpoint"]["adapter_sha256"] =
            donor_manifest["ordinary_checkpoint"]["adapter_sha256"].clone();
        let transplanted_bytes = std::fs::read(recipient.join(ADAPTER_FILE)).unwrap();
        assert_eq!(
            recipient_manifest["ordinary_checkpoint"]["adapter_sha256"]
                .as_str()
                .unwrap(),
            adapter_payload_sha256(&transplanted_bytes),
            "the transplanted payload and its leaf digest must be internally coherent"
        );
        write_manifest_value(&recipient, &recipient_manifest);

        let destination = sentinel_vars();
        let before = snapshot(&destination);
        let error = load_checkpoint(&recipient, &destination, &ordinary_binding(Some("recipe")))
            .unwrap_err();
        assert!(error.to_string().contains("state-envelope"), "{error}");
        assert_eq!(snapshot(&destination), before);
    }

    #[test]
    fn ordinary_checkpoint_rejects_valid_payload_tampering_before_adapter_mutation() {
        let tmp = TempDir::new("payload-tamper");
        for (label, tamper) in [
            ("adapter", 0_u8),
            ("optimizer", 1_u8),
            ("sampler", 2_u8),
            ("optimizer-step", 3_u8),
        ] {
            let dir = tmp.path().join(label);
            save_integrity_fixture(&dir);
            match tamper {
                0 => rewrite_first_tensor(&dir.join(ADAPTER_FILE), &var_key(0)),
                1 => rewrite_first_tensor(&dir.join(OPTIMIZER_FILE), &moment_key("m", 0)),
                2 => {
                    let mut manifest: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap())
                            .unwrap();
                    manifest["sampler_state"] = serde_json::json!([9, 8, 7, 5]);
                    write_manifest_value(&dir, &manifest);
                }
                3 => {
                    let mut manifest: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap())
                            .unwrap();
                    manifest["optimizer_step_t"] = serde_json::json!(12);
                    write_manifest_value(&dir, &manifest);
                }
                _ => unreachable!(),
            }
            let destination = sentinel_vars();
            let before = snapshot(&destination);
            let error =
                load_checkpoint(&dir, &destination, &ordinary_binding(Some("recipe"))).unwrap_err();
            assert!(
                error.to_string().contains("digest mismatch"),
                "{label}: wrong rejection: {error}"
            );
            assert_eq!(
                snapshot(&destination),
                before,
                "{label}: live adapter changed before tamper rejection"
            );
        }
    }

    #[test]
    fn ordinary_checkpoint_rejects_binding_drift_before_adapter_mutation() {
        let tmp = TempDir::new("binding-drift");
        save_integrity_fixture(tmp.path());
        let bindings = [
            CheckpointBinding::new("c".repeat(64), "b".repeat(64), Some("recipe".into())).unwrap(),
            CheckpointBinding::new("a".repeat(64), "c".repeat(64), Some("recipe".into())).unwrap(),
            CheckpointBinding::new("a".repeat(64), "b".repeat(64), Some("other".into())).unwrap(),
            CheckpointBinding::new("a".repeat(64), "b".repeat(64), None).unwrap(),
        ];
        for binding in bindings {
            let destination = sentinel_vars();
            let before = snapshot(&destination);
            assert!(load_checkpoint(tmp.path(), &destination, &binding).is_err());
            assert_eq!(snapshot(&destination), before);
        }
    }

    #[test]
    fn ordinary_checkpoint_rejects_extra_tensor_keys_even_with_refreshed_hashes() {
        let tmp = TempDir::new("extra-keys");
        for (label, file, key, identity_field) in [
            (
                "adapter",
                ADAPTER_FILE,
                "unexpected.adapter",
                "adapter_sha256",
            ),
            (
                "optimizer",
                OPTIMIZER_FILE,
                "unexpected.optimizer",
                "optimizer_sha256",
            ),
        ] {
            let dir = tmp.path().join(label);
            save_integrity_fixture(&dir);
            let path = dir.join(file);
            let bytes = std::fs::read(&path).unwrap();
            let mut tensors = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu).unwrap();
            tensors.insert(
                key.to_owned(),
                Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap(),
            );
            candle_core::safetensors::save(&tensors, &path).unwrap();

            // Refresh the corresponding digest so the exact-key-set comparison,
            // rather than the outer content hash, is the load-bearing rejection.
            let bytes = std::fs::read(&path).unwrap();
            let mut manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
            let digest = if file == ADAPTER_FILE {
                adapter_payload_sha256(&bytes)
            } else {
                optimizer_payload_sha256(
                    usize::try_from(manifest["optimizer_step_t"].as_u64().unwrap()).unwrap(),
                    &bytes,
                )
                .unwrap()
            };
            manifest["ordinary_checkpoint"][identity_field] = serde_json::json!(digest);
            refresh_v4_state_envelope(&mut manifest);
            write_manifest_value(&dir, &manifest);

            let destination = sentinel_vars();
            let before = snapshot(&destination);
            let error =
                load_checkpoint(&dir, &destination, &ordinary_binding(Some("recipe"))).unwrap_err();
            assert!(error.to_string().contains("key set mismatch"), "{error}");
            assert_eq!(snapshot(&destination), before);
        }
    }

    #[test]
    fn ordinary_checkpoint_rejects_same_schema_payload_swaps_before_mutation() {
        let tmp = TempDir::new("payload-swap");
        let first = tmp.path().join("first");
        save_integrity_fixture(&first);

        for file in [ADAPTER_FILE, OPTIMIZER_FILE] {
            let second = tmp.path().join(format!("second-{file}"));
            let vars = make_vars();
            vars[0]
                .set(&vars[0].as_tensor().affine(1.0, 3.0).unwrap())
                .unwrap();
            let mut optimizer = make_opt_state();
            optimizer.first_moments[0] = optimizer.first_moments[0].affine(1.0, 2.0).unwrap();
            save_checkpoint(
                &second,
                &vars,
                &optimizer,
                &[4, 5, 6],
                13,
                &ordinary_binding(Some("recipe")),
            )
            .unwrap();
            std::fs::copy(first.join(file), second.join(file)).unwrap();

            let destination = sentinel_vars();
            let before = snapshot(&destination);
            let error = load_checkpoint(&second, &destination, &ordinary_binding(Some("recipe")))
                .unwrap_err();
            assert!(error.to_string().contains("digest mismatch"), "{error}");
            assert_eq!(snapshot(&destination), before);
        }
    }

    /// Write a real (loadable) checkpoint at `root/step-<n>` with manifest `step = n`.
    fn write_step(root: &Path, n: u64) {
        save_checkpoint(
            root.join(format!("step-{n}")),
            &make_vars(),
            &make_opt_state(),
            &[1u8],
            n,
            &ordinary_binding(None),
        )
        .unwrap();
    }

    #[test]
    fn latest_checkpoint_is_none_when_missing_or_empty() {
        let tmp = TempDir::new("latest-none");
        // A path that does not exist at all → None (not an error).
        assert_eq!(latest_checkpoint(tmp.path().join("nope")).unwrap(), None);
        // An existing but empty checkpoints dir → None.
        assert_eq!(latest_checkpoint(tmp.path()).unwrap(), None);
    }

    #[test]
    fn latest_checkpoint_picks_highest_step_numerically_not_lexically() {
        // step-10 must beat step-2 even though "step-10" < "step-2" lexically —
        // a string sort would pick the wrong one.
        let tmp = TempDir::new("latest-order");
        write_step(tmp.path(), 2);
        write_step(tmp.path(), 10);
        let got = latest_checkpoint(tmp.path()).unwrap().unwrap();
        assert_eq!(got.step, 10);
        assert_eq!(got.dir, tmp.path().join("step-10"));
        // And the discovered checkpoint actually loads from there.
        let vars = make_vars();
        assert_eq!(
            load_checkpoint(&got.dir, &vars, &ordinary_binding(None))
                .unwrap()
                .step,
            10
        );
    }

    #[test]
    fn latest_checkpoint_rejects_newer_exact_dir_without_a_manifest() {
        // Ordinary publication exposes only atomically renamed final directories;
        // silently falling back here would replay step 5 despite visible step 99.
        let tmp = TempDir::new("latest-partial");
        write_step(tmp.path(), 5);
        std::fs::create_dir_all(tmp.path().join("step-99")).unwrap();
        assert!(matches!(
            latest_checkpoint(tmp.path()).unwrap_err(),
            CheckpointError::Io { .. }
        ));
    }

    #[test]
    fn latest_checkpoint_ignores_malformed_older_exact_dir_without_replay() {
        let tmp = TempDir::new("latest-older-malformed");
        std::fs::create_dir_all(tmp.path().join("step-1")).unwrap();
        write_step(tmp.path(), 5);
        assert_eq!(latest_checkpoint(tmp.path()).unwrap().unwrap().step, 5);
    }

    #[test]
    fn latest_checkpoint_rejects_newer_malformed_manifest_and_suffix_overflow() {
        let malformed = TempDir::new("latest-malformed");
        write_step(malformed.path(), 5);
        let bad_dir = malformed.path().join("step-99");
        std::fs::create_dir_all(&bad_dir).unwrap();
        std::fs::write(bad_dir.join(MANIFEST_FILE), b"{not-json").unwrap();
        assert!(matches!(
            latest_checkpoint(malformed.path()).unwrap_err(),
            CheckpointError::Manifest(_)
        ));

        let overflow = TempDir::new("latest-overflow");
        write_step(overflow.path(), 5);
        std::fs::create_dir_all(overflow.path().join("step-18446744073709551616")).unwrap();
        let error = latest_checkpoint(overflow.path()).unwrap_err();
        assert!(error.to_string().contains("u64 step range"), "{error}");
    }

    #[test]
    fn latest_checkpoint_rejects_newer_exact_final_name_that_is_not_a_directory() {
        let tmp = TempDir::new("latest-final-file");
        write_step(tmp.path(), 5);
        std::fs::write(
            tmp.path().join("step-99"),
            b"not an atomic checkpoint directory",
        )
        .unwrap();
        let error = latest_checkpoint(tmp.path()).unwrap_err();
        assert!(error.to_string().contains("is not a directory"), "{error}");
    }

    #[test]
    fn latest_checkpoint_ignores_tmp_old_and_foreign_siblings() {
        let tmp = TempDir::new("latest-foreign");
        write_step(tmp.path(), 3);
        // Crash-leftover stage / aside dirs (name has a dot-suffix after digits).
        std::fs::create_dir_all(tmp.path().join("step-7.tmp-12345")).unwrap();
        std::fs::create_dir_all(tmp.path().join("step-7.old-12345")).unwrap();
        // Foreign entries: non-numeric suffix, empty suffix, unrelated name, a file.
        std::fs::create_dir_all(tmp.path().join("step-abc")).unwrap();
        std::fs::create_dir_all(tmp.path().join("step-")).unwrap();
        std::fs::create_dir_all(tmp.path().join("latest")).unwrap();
        std::fs::write(tmp.path().join("step-100.foreign"), b"a foreign file").unwrap();
        let got = latest_checkpoint(tmp.path()).unwrap().unwrap();
        assert_eq!(
            got.step, 3,
            "only the real step-3 checkpoint is a candidate"
        );
        assert_eq!(got.dir, tmp.path().join("step-3"));
    }

    #[test]
    fn ordinary_and_continuation_discovery_keep_checkpoint_kinds_disjoint() {
        let tmp = TempDir::new("discovery-kinds");
        write_step(tmp.path(), 3);
        let continuation = tmp.path().join("step-9");
        save_checkpoint_no_replace(
            &continuation,
            &make_vars(),
            &make_opt_state(),
            &[1],
            None,
            continuation_manifest(9),
        )
        .unwrap();

        assert_eq!(
            latest_checkpoint(tmp.path()).unwrap().unwrap(),
            LatestCheckpoint {
                dir: tmp.path().join("step-3"),
                step: 3,
            }
        );
        assert_eq!(
            latest_rollout_ledger_continuation(tmp.path())
                .unwrap()
                .unwrap(),
            LatestCheckpoint {
                dir: continuation,
                step: 9,
            }
        );
    }
}
