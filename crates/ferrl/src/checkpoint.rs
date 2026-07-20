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
//! - [`save_checkpoint`] / [`load_checkpoint`] — a **momentum-faithful** checkpoint
//!   (**format version 2/3**): the adapter weights *plus* the optimizer moments
//!   ([`crate::optim::FerrlAdamW`]'s `m`/`v`/`step_t`) *plus* the rollout sampler RNG
//!   blob ([`crate::sampler::GrpoSampler`], whose state is `serde`-serializable). This
//!   is what lets [`crate::Trainer::resume`] continue an interrupted run **bit-exactly**
//!   — the same machine produces the identical post-resume trajectory. **v3** adds the
//!   run `base_seed` to the sampler blob (global-index substream seeding), making a
//!   resume self-contained; a pre-v3 sampler blob has no `base_seed` and is rejected on
//!   restore (a v2 momentum-faithful checkpoint is not resumable — fail-loud, not silent).
//!
//! A v1 checkpoint still loads (both readers accept format versions `1..=3`); when only
//! the adapter was persisted, [`load_checkpoint`] returns no optimizer/sampler state and
//! a resume falls back to fresh momentum + the policy's current RNG. The manifest is
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
//! into place only after both payload files and the manifest have been synced. The
//! checkpoint directory is then synced, followed by its parent so the directory
//! claim itself is durable. Any missing parent chain is created one component at a
//! time, syncing each new directory entry in its already durable ancestor before
//! proceeding. Successful publication thus requires a filesystem/platform that
//! supports advisory file locks plus opening and syncing directories; unsupported
//! behavior returns an I/O error instead of claiming durability. A failed claim
//! sync is removed only while the marker proves ownership, and the removal is
//! synced before returning. Every post-manifest success path verifies that the
//! owner marker and all three visible files still match the intended package; a
//! failed fence retries the complete durability boundary before the same check. If
//! ownership, cleanup, durability, or the exact-package check cannot be confirmed,
//! the writer returns [`CheckpointError::PublicationAmbiguous`] for explicit
//! operator reconciliation. A crash may still strand an incomplete claim, but it
//! cannot expose or overwrite a completed continuation.

use std::collections::HashMap;
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
/// Filename of the serialized optimizer moment tensors (format version 2).
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
/// substream seeding — see [`crate::sampler::GrpoSampler`]). v3 checkpoints are
/// self-contained: the seed that re-derives the rollout travels in the blob, so a
/// resume is bit-exact regardless of how the policy is reconstructed. A pre-v3
/// sampler blob lacks `base_seed`, so it cannot be resumed bit-exactly; restoring
/// one fails loud (see [`crate::sampler::GrpoSampler::from_state_bytes`]) rather
/// than silently re-seeding. (v1/v2 adapter weights still load for eval.)
const FORMAT_VERSION: u32 = 3;
/// Lowest on-disk format version this build can read. Older (v1, adapter-only)
/// checkpoints still load — a resume then falls back to fresh momentum.
const MIN_FORMAT_VERSION: u32 = 1;
/// On-disk schema version for separated rollout-ledger continuations.
pub(crate) const ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION: u32 = 1;
pub(crate) const ROLLOUT_LEDGER_CONTINUATION_KIND: &str = "rollout_ledger";

/// Provenance that distinguishes a separated rollout-ledger continuation from
/// an ordinary trainer checkpoint and binds it to one exact ledger chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RolloutLedgerContinuationManifest {
    pub(crate) format_version: u32,
    pub(crate) kind: String,
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
    /// The on-disk checkpoint does not match the model it is being loaded into:
    /// an unknown format version, a differing tensor count, a missing tensor, or a
    /// shape/dtype mismatch against the live trainable [`Var`]s.
    #[error("checkpoint mismatch: {0}")]
    Mismatch(String),
}

/// Self-describing metadata stored alongside the adapter tensors.
///
/// The three `Option` fields are the **format-version-2** additions; they are
/// `#[serde(default)]` so a v1 manifest (which lacks them) still deserializes, with each
/// defaulting to `None` (the fresh-momentum-on-resume fallback).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointManifest {
    /// On-disk format version (validated against `1..=3` by [`load_adapter`]).
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
    /// self-describing about *which* projections its positional tensor list
    /// covers. The load contract stays positional (count/shape/dtype validation
    /// against the live model), but [`crate::Trainer::resume`] additionally
    /// cross-checks this string against the restoring policy and fails loud on
    /// a mismatch — count/shape/dtype cannot distinguish **shape-aliased**
    /// recipes (e.g. `attn:qk` vs `attn:qv`). `None` for checkpoints written
    /// before this field existed, or by a policy that does not report a recipe.
    /// `#[serde(default)]` for back-compat.
    #[serde(default)]
    pub lora_recipe: Option<String>,
    /// Present only for a versioned separated rollout-ledger continuation.
    /// Ordinary cadence/eval checkpoints omit it and are never eligible for
    /// separated-continuation discovery.
    #[serde(default)]
    pub(crate) rollout_ledger_continuation: Option<RolloutLedgerContinuationManifest>,
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
        };
        let manifest_path = stage.join(MANIFEST_FILE);
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, json).map_err(|e| io(&manifest_path, e))?;
        Ok(())
    })
}

/// Restore a checkpoint from `dir` into `vars`, in place.
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
/// Returns [`CheckpointError::Mismatch`] if the format version is unknown, the
/// tensor count differs, a tensor is missing, or any tensor's shape/dtype does not
/// match the corresponding live `Var`; or [`CheckpointError::Io`] /
/// [`CheckpointError::Candle`] / [`CheckpointError::Manifest`] on read failures.
pub fn load_adapter(
    dir: impl AsRef<Path>,
    vars: &[Var],
) -> Result<CheckpointManifest, CheckpointError> {
    let dir = dir.as_ref();

    let manifest = read_manifest(dir)?;
    if manifest.num_vars != vars.len() {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has {} tensors but the model exposes {} trainable vars",
            manifest.num_vars,
            vars.len()
        )));
    }

    load_adapter_tensors(dir, vars)?;
    Ok(manifest)
}

/// Read and version-validate `manifest.json` from `dir` **without touching any
/// model state** — so a caller (e.g. [`crate::Trainer::resume`]) can run
/// pre-flight checks (the adapter-recipe cross-check) before the positional
/// load mutates the live `Var`s.
pub(crate) fn read_manifest(dir: &Path) -> Result<CheckpointManifest, CheckpointError> {
    let manifest_path = dir.join(MANIFEST_FILE);
    let raw = std::fs::read_to_string(&manifest_path).map_err(|e| io(&manifest_path, e))?;
    let manifest: CheckpointManifest = serde_json::from_str(&raw)?;
    if manifest.format_version < MIN_FORMAT_VERSION || manifest.format_version > FORMAT_VERSION {
        return Err(CheckpointError::Mismatch(format!(
            "unsupported checkpoint format_version {} (this build reads {MIN_FORMAT_VERSION}..={FORMAT_VERSION})",
            manifest.format_version
        )));
    }
    Ok(manifest)
}

/// The tensor half of [`load_adapter`]: validate-then-apply `adapter.safetensors`
/// into `vars` (all-or-nothing; see [`load_adapter`]).
fn load_adapter_tensors(dir: &Path, vars: &[Var]) -> Result<(), CheckpointError> {
    let adapter_path = dir.join(ADAPTER_FILE);
    let loaded = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;

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

    // Pass 2 — every tensor validated; apply. `set` cannot fail here on shape (it
    // matches), a self-set (the source is a fresh load), or a non-contiguous
    // destination (LoRA factors are contiguous).
    for (v, t) in vars.iter().zip(prepared.iter()) {
        v.set(t)?;
    }
    Ok(())
}

/// The result of [`load_checkpoint`]: the resume step plus any persisted optimizer and
/// sampler state.
///
/// For a v1 (adapter-only) checkpoint, `optimizer_state` and `sampler_state` are `None`
/// — [`crate::Trainer::resume`] then falls back to fresh momentum and the policy's
/// current sampler.
#[derive(Debug)]
pub struct LoadedCheckpoint {
    /// Completed optimizer steps — the `start_step` a resume continues from.
    pub step: u64,
    /// The optimizer moments + step counter, if the checkpoint persisted them (v2).
    pub optimizer_state: Option<OptimizerState>,
    /// The opaque rollout-sampler RNG blob, if the checkpoint persisted it (v2).
    pub sampler_state: Option<Vec<u8>>,
    /// The writing policy's canonical adapter-recipe string, if recorded (see
    /// [`CheckpointManifest::lora_recipe`]). Surfaced so a caller can
    /// cross-check it against the restoring policy: the positional load
    /// validates only count/shape/dtype, which cannot distinguish
    /// **shape-aliased** recipes (e.g. `attn:qk` vs `attn:qv` — the k and v
    /// projections are shape-identical), so a recipe swap would otherwise
    /// restore adapters onto the wrong projections silently.
    /// [`crate::Trainer::resume`] fails loud on a mismatch.
    pub lora_recipe: Option<String>,
}

/// Persist a **momentum-faithful** checkpoint (format version 2): the adapter weights,
/// the optimizer moments, and the rollout-sampler RNG blob.
///
/// Writes `adapter.safetensors`, `optimizer.safetensors`, and (last, as the commit
/// marker) `manifest.json`. The optimizer moments are keyed by parameter index; the
/// optimizer step counter, the `sampler_state` blob, and the `lora_recipe` string live
/// in the manifest. Each tensor is moved to the CPU and made contiguous first, so a
/// CUDA-resident run checkpoints the same way. Restored as a unit by
/// [`load_checkpoint`] + [`crate::Trainer::resume`].
///
/// `sampler_state` is the opaque blob from [`crate::Policy::sampler_state`]; it is stored
/// verbatim and only the policy interprets it on restore. `lora_recipe` is the policy's
/// canonical adapter-recipe string (see [`CheckpointManifest::lora_recipe`]).
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
    lora_recipe: Option<&str>,
) -> Result<(), CheckpointError> {
    let recipe = lora_recipe.map(str::to_owned);
    write_staged(dir.as_ref(), |stage| {
        write_checkpoint_contents(
            stage,
            &stage.join(MANIFEST_FILE),
            vars,
            opt_state,
            sampler_state,
            step,
            recipe,
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
/// marker still proves ownership. Every successful post-manifest durability fence
/// verifies both ownership and the visible adapter, optimizer, and manifest against
/// the exact package fingerprinted before publication. An unreconciled failure,
/// ownership loss, or package mismatch is returned as
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
        lora_recipe.map(str::to_owned),
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
    #[cfg(test)]
    run_no_replace_hook(dir, NoReplaceHookPoint::BeforeManifestRename);
    let manifest = dir.join(MANIFEST_FILE);
    if let Err(error) =
        std::fs::rename(&manifest_stage, &manifest).map_err(|error| io(&manifest_stage, error))
    {
        return Err(cleanup_uncommitted_claim(dir, parent, &ownership, error));
    }
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
    lora_recipe: Option<String>,
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

    // Manifest LAST — either inside a hidden staging directory or under a
    // temporary name that the no-replace publisher atomically commits.
    let manifest = CheckpointManifest {
        format_version: FORMAT_VERSION,
        step,
        num_vars: vars.len(),
        optimizer_step_t: Some(opt_state.step_t),
        optimizer_num_vars: Some(n),
        sampler_state: Some(sampler_state.to_vec()),
        lora_recipe,
        rollout_ledger_continuation,
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

/// Restore a checkpoint from `dir` into `vars`, returning the resume step and any
/// persisted optimizer / sampler state for a momentum-faithful resume.
///
/// Loads the adapter weights into `vars` exactly as [`load_adapter`] (same
/// all-or-nothing shape/dtype/count validation and `1..=3` version check). For a **v2
/// or v3** checkpoint it additionally reads the optimizer moments (the manifest records
/// how many) and the sampler RNG blob; for a **v1** checkpoint both come back `None`.
/// Restoring a pre-v3 sampler blob fails loud at
/// [`crate::sampler::GrpoSampler::from_state_bytes`] (no `base_seed`) rather than
/// silently re-seeding — a v2 momentum-faithful checkpoint is therefore not resumable.
///
/// The optimizer moments are **not** validated against `vars` here — the optimizer
/// filters to float parameters, so [`crate::optim::FerrlAdamW::load_state`] validates
/// them against its own parameter set (count + shape + dtype) when they are applied.
///
/// # Errors
///
/// As [`load_adapter`], plus [`CheckpointError::Mismatch`] if a v2 manifest references
/// optimizer moments that are missing from `optimizer.safetensors`.
pub fn load_checkpoint(
    dir: impl AsRef<Path>,
    vars: &[Var],
) -> Result<LoadedCheckpoint, CheckpointError> {
    let dir = dir.as_ref();
    // Reuses the adapter load + all-or-nothing validation + version-range check.
    let manifest = load_adapter(dir, vars)?;

    let optimizer_state = match (manifest.optimizer_step_t, manifest.optimizer_num_vars) {
        (Some(step_t), Some(num)) => {
            let opt_path = dir.join(OPTIMIZER_FILE);
            let loaded = candle_core::safetensors::load(&opt_path, &Device::Cpu)?;
            let mut first_moments = Vec::with_capacity(num);
            let mut second_moments = Vec::with_capacity(num);
            for i in 0..num {
                let mk = moment_key("m", i);
                let vk = moment_key("v", i);
                let m = loaded.get(&mk).ok_or_else(|| {
                    CheckpointError::Mismatch(format!(
                        "checkpoint is missing optimizer tensor {mk}"
                    ))
                })?;
                let v = loaded.get(&vk).ok_or_else(|| {
                    CheckpointError::Mismatch(format!(
                        "checkpoint is missing optimizer tensor {vk}"
                    ))
                })?;
                first_moments.push(m.clone());
                second_moments.push(v.clone());
            }
            Some(OptimizerState {
                step_t,
                first_moments,
                second_moments,
            })
        }
        _ => None,
    };

    Ok(LoadedCheckpoint {
        step: manifest.step,
        optimizer_state,
        sampler_state: manifest.sampler_state,
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
/// with the greatest completed-step count. Ordering is on the manifest's numeric
/// `step`, **not** the directory name, so `step-10` correctly outranks `step-2`
/// (a lexical sort would invert them).
///
/// A subdirectory whose manifest is missing, unreadable, or an unsupported
/// format version is **skipped, not an error**: the writer publishes a checkpoint
/// only once its manifest is committed by an atomic rename (see the module docs),
/// so a `step-<n>` directory without a readable manifest is an interrupted or
/// foreign write, never a usable checkpoint. A present, readable manifest is a
/// sufficient completeness marker: either every sibling landed in the same
/// directory rename, or the no-replace writer committed the manifest only after
/// every sibling. Crash-leftover `.tmp-*` / `.old-*` siblings do not match the
/// `step-<n>` shape and are ignored, as is any unrelated entry.
///
/// # Errors
///
/// Returns [`CheckpointError::Io`] only if `checkpoints_dir` exists but its
/// listing cannot be read (a permissions / IO fault on the directory itself). A
/// **missing** `checkpoints_dir` is not an error — it returns `None` (a run that
/// has not checkpointed yet).
pub fn latest_checkpoint(
    checkpoints_dir: impl AsRef<Path>,
) -> Result<Option<LatestCheckpoint>, CheckpointError> {
    let dir = checkpoints_dir.as_ref();
    if !dir.exists() {
        return Ok(None);
    }
    let mut best: Option<LatestCheckpoint> = None;
    for entry in std::fs::read_dir(dir).map_err(|e| io(dir, e))? {
        let entry = entry.map_err(|e| io(dir, e))?;
        // A per-entry file-type fault (e.g. a sibling being swept right now) is
        // not a candidate, not a reason to fail the whole discovery.
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let path = entry.path();
        // Match exactly `step-<digits>`: this excludes `.tmp-*` / `.old-*`
        // crash siblings (their names carry a dot-suffix after the digits) and
        // any foreign directory.
        let is_step_dir = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("step-"))
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()));
        if !is_step_dir {
            continue;
        }
        // The manifest is both the completeness marker and the source of truth
        // for `step`. A partial / foreign / future-version directory simply is
        // not a candidate (skip, do not fail).
        let Ok(manifest) = read_manifest(&path) else {
            continue;
        };
        let candidate = LatestCheckpoint {
            dir: path,
            step: manifest.step,
        };
        if best.as_ref().is_none_or(|b| candidate.step > b.step) {
            best = Some(candidate);
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
        if continuation.format_version != ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION
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
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: CheckpointManifest = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
    }

    /// A v1 manifest (no optimizer/sampler fields on disk) still deserializes — the
    /// `#[serde(default)]` fields come back `None` (the fresh-momentum fallback).
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

    fn continuation_manifest(step: u64) -> RolloutLedgerContinuationManifest {
        RolloutLedgerContinuationManifest {
            format_version: ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION,
            kind: ROLLOUT_LEDGER_CONTINUATION_KIND.to_owned(),
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
            99,
            Some("ordinary-replacement"),
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
                step: 99,
            }
        );
        let loaded = load_checkpoint(destination, &make_vars()).unwrap();
        assert_eq!(loaded.step, 99);
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
        let tmp = TempDir::new("v2-adapter");
        let vars = make_vars();
        let adapter = snapshot(&vars);
        save_checkpoint(tmp.path(), &vars, &make_opt_state(), &[9u8, 8, 7], 13, None).unwrap();
        clobber(&vars);
        let loaded = load_checkpoint(tmp.path(), &vars).unwrap();
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
        let loaded = load_checkpoint(&destination, &vars).unwrap();
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
                Some("ordinary-replacement"),
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
        load_checkpoint(&destination, &make_vars()).unwrap();
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
        load_checkpoint(&destination, &make_vars()).unwrap();
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
        assert_eq!(latest_checkpoint(&checkpoints).unwrap(), None);
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints).unwrap(),
            None
        );
        assert!(matches!(
            load_checkpoint(&destination, &make_vars()),
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
        assert_eq!(latest_checkpoint(&checkpoints).unwrap(), None);
        assert_eq!(
            latest_rollout_ledger_continuation(&checkpoints).unwrap(),
            None
        );
        assert!(matches!(
            load_checkpoint(&destination, &make_vars()),
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
        let tmp = TempDir::new("v2-opt-sampler");
        let vars = make_vars();
        let opt = make_opt_state();
        let sampler_blob = vec![9u8, 8, 7, 6, 5];
        save_checkpoint(
            tmp.path(),
            &vars,
            &opt,
            &sampler_blob,
            13,
            Some("attn:qkvo|mlp:gud|gdn:-"),
        )
        .unwrap();
        let loaded = load_checkpoint(tmp.path(), &vars).unwrap();
        let os = loaded
            .optimizer_state
            .expect("v2 must carry optimizer state");
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

        save_checkpoint(&dir, &vars, &make_opt_state(), &[1u8], 5, None).unwrap();

        // The publish replaced the prior directory as a unit: the junk is gone,
        // the checkpoint is complete and loadable, and no stage dir remains.
        assert!(!dir.join("garbage.bin").exists(), "stale content survived");
        assert!(!stage.exists(), "stage dir must not survive a commit");
        let loaded = load_checkpoint(&dir, &vars).unwrap();
        assert_eq!(loaded.step, 5);

        // Re-writing the same path succeeds and stays complete (the in-place
        // overwrite path a periodic checkpointer hits on a re-run).
        save_checkpoint(&dir, &vars, &make_opt_state(), &[2u8], 5, None).unwrap();
        let again = load_checkpoint(&dir, &vars).unwrap();
        assert_eq!(again.sampler_state, Some(vec![2u8]));
    }

    #[test]
    fn failed_staged_write_leaves_the_prior_checkpoint_intact() {
        // The Err branch of write_staged: a mid-write failure must leave the
        // previously published checkpoint untouched and clean up its stage.
        let tmp = TempDir::new("staged-err");
        let dir = tmp.path().join("step-3");
        let vars = make_vars();
        save_checkpoint(&dir, &vars, &make_opt_state(), &[7u8], 3, None).unwrap();
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
        let loaded = load_checkpoint(&dir, &vars).unwrap();
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
            Some("attn:qkvo|mlp:gud|gdn:-"),
        )
        .unwrap();
        let manifest = load_adapter(&dir_v2, &vars).unwrap();
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
    fn load_checkpoint_on_a_v1_adapter_only_yields_no_optimizer_or_sampler() {
        // A v1 checkpoint (save_adapter) loads through load_checkpoint with no optimizer
        // or sampler state — the documented fresh-momentum fallback.
        let tmp = TempDir::new("v1-via-checkpoint");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 4, None).unwrap();
        let loaded = load_checkpoint(tmp.path(), &vars).unwrap();
        assert_eq!(loaded.step, 4);
        assert!(
            loaded.optimizer_state.is_none(),
            "v1 has no optimizer state"
        );
        assert!(loaded.sampler_state.is_none(), "v1 has no sampler state");
    }

    /// Write a real (loadable) checkpoint at `root/step-<n>` with manifest `step = n`.
    fn write_step(root: &Path, n: u64) {
        save_checkpoint(
            root.join(format!("step-{n}")),
            &make_vars(),
            &make_opt_state(),
            &[1u8],
            n,
            None,
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
        assert_eq!(load_checkpoint(&got.dir, &vars).unwrap().step, 10);
    }

    #[test]
    fn latest_checkpoint_skips_dirs_without_a_committed_manifest() {
        // A higher-numbered directory with no manifest (an interrupted write)
        // is not a candidate — discovery falls back to the newest complete one.
        let tmp = TempDir::new("latest-partial");
        write_step(tmp.path(), 5);
        std::fs::create_dir_all(tmp.path().join("step-99")).unwrap(); // no manifest
        let got = latest_checkpoint(tmp.path()).unwrap().unwrap();
        assert_eq!(got.step, 5, "the manifest-less step-99 must be skipped");
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
        std::fs::write(tmp.path().join("step-100"), b"a file, not a dir").unwrap();
        let got = latest_checkpoint(tmp.path()).unwrap().unwrap();
        assert_eq!(
            got.step, 3,
            "only the real step-3 checkpoint is a candidate"
        );
        assert_eq!(got.dir, tmp.path().join("step-3"));
    }
}
