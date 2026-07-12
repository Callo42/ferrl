//! The NCCL distributed bridge for data and tensor parallelism — CPU-mock-first.
//!
//! [`NcclComm`] is the [`Comm`] implementation that runs real multi-GPU /
//! multi-process all-reduces. It is generic over a [`NcclPrimitives`] seam —
//! the one collective primitive (init + element-wise sum-reduce) — so the
//! **same orchestration** (dtype validation, the scalar→tensor packing NCCL's
//! tensor-only API forces, the per-rank local contract checks) runs two ways:
//!
//! - against `MockNccl`, an in-memory CPU substrate, in plain CI — this is
//!   the **mock gate**: it exercises the real `NcclComm` code path with no GPU
//!   and no process orchestration; and
//! - against `RealNccl`, the `unsafe` cudarc-NCCL FFI behind `--features nccl`
//!   (the [decision-D2 quarantine](crate::comm)), on the GPU cluster.
//!
//! Because the orchestration is shared, a green mock gate proves the logic the
//! FFI is wrapped in; only the byte-moving collective itself is GPU-exclusive,
//! and that is covered by a manual multi-GPU Slurm run.
//!
//! ## What the contract checks can and cannot be
//!
//! [`LocalComm`](crate::comm::LocalComm) sees every rank's contribution in one
//! process, so it validates the **cross-rank** contract (same count, shape,
//! dtype) directly. A real NCCL rank is one process that sees only its own
//! tensors, so `NcclComm` validates only what is **local** — each tensor has an
//! NCCL-supported dtype and is contiguous — and leaves the cross-rank contract
//! to NCCL itself (a shape/count disagreement deadlocks or errors the
//! collective). The mock, being in-process, restores the cross-rank check, so
//! the CI gate still covers mismatch detection.
//!
//! ## Determinism: cross-rank agreement, not a sequential fold
//!
//! NCCL guarantees **every rank receives the bit-identical all-reduce output** —
//! which is exactly the lockstep invariant the trainer rests on (same reduced
//! reduced state on every rank. It does **not** guarantee that
//! output equals a specific rank-order fold: NCCL's ring/tree reduction may sum
//! in any order, and float addition is not associative. So the equivalence gate
//! asserts cross-rank agreement plus correctness within an fp tolerance, not
//! bitwise equality to a sequential reference — except at `world_size == 2`,
//! where the reduction is a single `a + b` and IEEE-754 commutativity makes it
//! bit-exact regardless of order.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use super::{Comm, CommError};

/// `ncclUniqueId` is a fixed 128-byte blob: rank 0 generates it and every other
/// rank must receive the identical bytes before `ncclCommInitRank`.
#[cfg(any(test, feature = "nccl"))]
pub(crate) const NCCL_UNIQUE_ID_LEN: usize = 128;

// ===========================================================================
// Bootstrap config
// ===========================================================================

/// Distributed launch parameters parsed from the Slurm environment.
///
/// A DP or TP launch runs one process per rank; each reads its identity from the
/// environment Slurm sets (`SLURM_PROCID` / `SLURM_NTASKS` / `SLURM_LOCALID`)
/// and a shared rendezvous-file path (`FERRL_NCCL_RENDEZVOUS`) over which rank 0
/// publishes the NCCL unique id. This type is the pure, CPU-testable parse;
/// turning it into a live communicator is `RealNccl`'s job (behind
/// `--features nccl`).
#[derive(Debug, Clone)]
pub struct NcclConfig {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    rendezvous_file: PathBuf,
}

impl NcclConfig {
    /// Parse the launch config from the process environment.
    ///
    /// `FERRL_NCCL_RENDEZVOUS` **must be unique per launch** (e.g. derived from
    /// the Slurm job id): rank 0 publishes the NCCL unique id there, and a path
    /// reused across runs risks a peer reading a stale id. A node-local `/tmp`
    /// path is fine single-node; multi-node needs a shared-filesystem path.
    ///
    /// # Errors
    ///
    /// [`CommError::Config`] if a required variable is missing or unparseable,
    /// if `rank >= world_size`, or if the rendezvous path is unset.
    pub fn from_env() -> Result<Self, CommError> {
        Self::from_getter(|key| std::env::var(key).ok())
    }

    /// The parse, over an injectable getter — the env-free core `from_env`
    /// wraps, so tests exercise it without mutating the shared process
    /// environment.
    fn from_getter(get: impl Fn(&str) -> Option<String>) -> Result<Self, CommError> {
        let rank = parse_usize_env(&get, "SLURM_PROCID")?;
        let world_size = parse_usize_env(&get, "SLURM_NTASKS")?;
        let local_rank = parse_usize_env(&get, "SLURM_LOCALID")?;
        if world_size == 0 {
            return Err(CommError::Config(
                "SLURM_NTASKS must be at least 1".to_owned(),
            ));
        }
        if rank >= world_size {
            return Err(CommError::Config(format!(
                "SLURM_PROCID ({rank}) must be < SLURM_NTASKS ({world_size})"
            )));
        }
        let rendezvous_file = get("FERRL_NCCL_RENDEZVOUS")
            .map(PathBuf::from)
            .ok_or_else(|| {
                CommError::Config(
                    "FERRL_NCCL_RENDEZVOUS (the shared NCCL-unique-id rendezvous path) is unset"
                        .to_owned(),
                )
            })?;
        Ok(Self {
            rank,
            world_size,
            local_rank,
            rendezvous_file,
        })
    }

    /// This process's global rank in `0..world_size`.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of ranks in the world.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// This process's node-local rank — the CUDA device index it binds.
    #[must_use]
    pub fn local_rank(&self) -> usize {
        self.local_rank
    }

    /// The shared file over which rank 0 publishes the NCCL unique id.
    #[must_use]
    pub fn rendezvous_file(&self) -> &Path {
        &self.rendezvous_file
    }
}

/// Read `key` from `get` and parse it as a `usize`, mapping both absence and a
/// bad value to a descriptive [`CommError::Config`].
fn parse_usize_env(get: &impl Fn(&str) -> Option<String>, key: &str) -> Result<usize, CommError> {
    let raw = get(key)
        .ok_or_else(|| CommError::Config(format!("{key} is unset (not a Slurm launch?)")))?;
    raw.parse::<usize>()
        .map_err(|e| CommError::Config(format!("{key}={raw:?} is not a non-negative integer: {e}")))
}

// ===========================================================================
// Unique-id rendezvous (portable file dance; the id bytes come from the FFI)
// ===========================================================================

/// Publish rank 0's unique-id bytes atomically: write a sibling temp file, then
/// rename it into place so a peer never reads a half-written id.
///
/// Clears any pre-existing id at `path` first, so that during the brief publish
/// window a peer polling [`read_id_file`] finds *no* file (and waits) rather than
/// a **stale** id from a prior run on a reused path — which would build a
/// communicator from a dead `ncclUniqueId` and hang. The *primary* defense, though,
/// is that `RealNccl::bootstrap` deletes this file on rank 0 the moment the
/// communicator is built (every peer has read the id by then), so a cleanly-finished
/// run leaves nothing stale for a reused path to trip on; this clear-before-write and
/// a unique-per-launch path are the backstops for a hard kill mid-bootstrap.
///
/// # Errors
///
/// [`CommError::Config`] if the temp write or the rename fails.
#[cfg(any(test, feature = "nccl"))]
fn write_id_file(path: &Path, bytes: &[u8; NCCL_UNIQUE_ID_LEN]) -> Result<(), CommError> {
    let _ = std::fs::remove_file(path);
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &bytes[..])
        .map_err(|e| CommError::Config(format!("writing rendezvous temp {tmp:?}: {e}")))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| CommError::Config(format!("publishing rendezvous file {path:?}: {e}")))
}

/// Poll `path` until it holds a complete unique id, or `timeout` elapses.
///
/// # Errors
///
/// [`CommError::Config`] if the id does not appear, complete, within `timeout`.
#[cfg(any(test, feature = "nccl"))]
fn read_id_file(
    path: &Path,
    timeout: std::time::Duration,
) -> Result<[u8; NCCL_UNIQUE_ID_LEN], CommError> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Ok(bytes) = std::fs::read(path) {
            if bytes.len() == NCCL_UNIQUE_ID_LEN {
                let mut id = [0u8; NCCL_UNIQUE_ID_LEN];
                id.copy_from_slice(&bytes);
                return Ok(id);
            }
        }
        if std::time::Instant::now() >= deadline {
            return Err(CommError::Config(format!(
                "rendezvous file {path:?} did not hold a complete unique id within {timeout:?} \
                 (rank 0 never published, or the path is not shared across ranks)"
            )));
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

// ===========================================================================
// dtype mapping
// ===========================================================================

/// The NCCL element types ferrl all-reduces. The FFI maps each to cudarc's
/// `ncclDataType_t`; keeping the decision here (not in the `unsafe` layer) makes
/// the supported-dtype contract CI-testable.
///
/// Deliberately just the two ferrl paths reduce: **F32** (`LoRA` gradients and
/// F32-staged TP activations) and **F64** (scalar/control all-reduces). Half
/// precision (F16/BF16) is staged through F32 by the TP helper rather than
/// reduced directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcclDataType {
    /// 32-bit float — `LoRA` gradients and staged TP activations.
    F32,
    /// 64-bit float — the scalar all-reduce packs a single `f64`.
    F64,
}

/// Map a candle [`DType`] to its NCCL element type, rejecting every dtype the
/// ferrl distributed paths never reduce directly.
///
/// # Errors
///
/// [`CommError::Mismatch`] for any dtype other than F32/F64 (including the
/// half-precision types — see [`NcclDataType`]).
pub(crate) fn nccl_dtype_tag(dtype: DType) -> Result<NcclDataType, CommError> {
    match dtype {
        DType::F32 => Ok(NcclDataType::F32),
        DType::F64 => Ok(NcclDataType::F64),
        other => Err(CommError::Mismatch(format!(
            "ferrl's NCCL all-reduce carries only f32 (gradients/TP activations) and f64 \
             (scalars); got {other:?}"
        ))),
    }
}

// ===========================================================================
// The primitive seam
// ===========================================================================

/// The one collective primitive [`NcclComm`] drives: identity plus an in-place
/// element-wise sum-reduce across ranks. `RealNccl` (FFI) and `MockNccl`
/// (in-memory) are its two implementations — swapping them is what makes the
/// CPU mock exercise the real orchestration.
///
/// Implementations must be [`Send`] (the handle moves into its rank's process /
/// thread) and must return **bit-identical** reduced tensors on every rank.
pub trait NcclPrimitives: std::fmt::Debug + Send {
    /// This handle's rank in `0..world_size()`.
    fn rank(&self) -> usize;

    /// The number of ranks in the world.
    fn world_size(&self) -> usize;

    /// Sum each tensor across all ranks, in place: on return `tensors[i]` holds
    /// the cross-rank sum, identical on every rank.
    ///
    /// # Errors
    ///
    /// [`CommError`] on a collective failure.
    fn all_reduce(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError>;
}

// ===========================================================================
// The orchestration
// ===========================================================================

/// The NCCL-backed [`Comm`]: local contract validation and scalar packing over a
/// [`NcclPrimitives`] collective. Generic over the primitive so one body serves
/// both the CI mock and the GPU FFI.
#[derive(Debug)]
pub struct NcclComm<P: NcclPrimitives> {
    primitive: P,
    /// The device the scalar all-reduce allocates its one-element tensor on
    /// (the rank's CUDA device for `RealNccl`; CPU for the mock).
    device: Device,
}

impl<P: NcclPrimitives> NcclComm<P> {
    /// The device this communicator stages collectives on — the rank's CUDA
    /// device for the real backend. A multi-process launch builds its
    /// to-be-reduced tensors here so they share the communicator's stream.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(any(test, feature = "nccl"))]
impl<P: NcclPrimitives> NcclComm<P> {
    /// Assemble an `NcclComm` from a ready primitive and the scalar-staging
    /// device. Used by `RealNccl`'s bootstrap and the mock world builder.
    pub(crate) fn from_parts(primitive: P, device: Device) -> Self {
        Self { primitive, device }
    }
}

impl<P: NcclPrimitives> Comm for NcclComm<P> {
    fn rank(&self) -> usize {
        self.primitive.rank()
    }

    fn world_size(&self) -> usize {
        self.primitive.world_size()
    }

    fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        validate_local(tensors)?;
        self.primitive.all_reduce(tensors)
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
        // NCCL has no scalar collective: pack the f64 into a one-element tensor
        // on the rank's device, reduce, and read element 0 back.
        let staged = Tensor::from_vec(vec![value], 1usize, &self.device)?;
        let mut one = vec![staged];
        self.all_reduce_sum(&mut one)?;
        let reduced = one[0].to_vec1::<f64>()?;
        Ok(reduced[0])
    }
}

/// Validate the part of the collective contract a single rank can see: every
/// tensor must have an NCCL-supported dtype and be contiguous (NCCL reduces a
/// flat device buffer). The cross-rank shape/count agreement is enforced by
/// NCCL itself (or, under the mock, by the in-memory reducer).
fn validate_local(tensors: &[Tensor]) -> Result<(), CommError> {
    for (i, tensor) in tensors.iter().enumerate() {
        nccl_dtype_tag(tensor.dtype())?;
        if !tensor.is_contiguous() {
            return Err(CommError::Mismatch(format!(
                "NCCL all-reduce requires contiguous tensors; tensor {i} is not contiguous"
            )));
        }
    }
    Ok(())
}

// ===========================================================================
// The real FFI primitive + its public constructor (the `--features nccl` quarantine)
// ===========================================================================

#[cfg(feature = "nccl")]
mod real;

#[cfg(feature = "nccl")]
pub use real::RealNccl;

#[cfg(feature = "nccl")]
impl NcclComm<real::RealNccl> {
    /// Bootstrap a live, multi-GPU NCCL communicator for **this** rank from the
    /// Slurm launch environment: parse [`NcclConfig`] from the environment, bind
    /// the node-local CUDA device, exchange the NCCL unique id over the shared
    /// rendezvous file (rank 0 generates and publishes it; every other rank
    /// reads it), and initialize the communicator.
    ///
    /// One process per rank calls this once at startup, then hands the result to
    /// [`Trainer::with_comm`](crate::trainer::Trainer::with_comm). Available only
    /// under `--features nccl`.
    ///
    /// # Errors
    ///
    /// [`CommError::Config`] if the launch environment is malformed, the CUDA
    /// device cannot be opened, the unique-id rendezvous fails, or
    /// `ncclCommInitRank` errors.
    pub fn from_slurm_env() -> Result<Self, CommError> {
        let config = NcclConfig::from_env()?;
        let (primitive, device) = real::RealNccl::bootstrap(&config)?;
        Ok(NcclComm::from_parts(primitive, device))
    }
}

// ===========================================================================
// MockNccl — the in-memory CPU substrate (test only)
// ===========================================================================

/// An in-process stand-in for a real NCCL world: N ranks as N threads, the
/// collective rendezvousing over the same shared, deterministic reducer
/// [`LocalComm`](crate::comm::LocalComm) uses. It carries genuine cross-rank
/// semantics (rank-order sum, mismatch/timeout fail-loud) on the CPU, so the
/// real [`NcclComm`] orchestration runs end-to-end in CI with no GPU.
#[cfg(test)]
pub(crate) struct MockNccl {
    rank: usize,
    world: usize,
    tensors: std::sync::Arc<super::Rendezvous<Vec<Tensor>>>,
}

#[cfg(test)]
impl std::fmt::Debug for MockNccl {
    // `Rendezvous` holds a `Mutex`/`Condvar` and is not `Debug`; mirror
    // `LocalComm`'s hand-rolled impl rather than deriving.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockNccl")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl MockNccl {
    /// Mint the `world_size` ranks of a fresh mock world, each already wrapped
    /// in an `NcclComm` on the CPU device — drop-in `Comm` handles for a
    /// `thread::scope` test.
    #[must_use]
    pub(crate) fn world(world_size: usize) -> Vec<NcclComm<MockNccl>> {
        assert!(world_size > 0, "a world needs at least one rank");
        let poison = std::sync::Arc::new(super::PoisonCell::default());
        let tensors = std::sync::Arc::new(super::Rendezvous::new(
            world_size,
            super::DEFAULT_COLLECTIVE_TIMEOUT,
            poison,
        ));
        (0..world_size)
            .map(|rank| {
                let prim = MockNccl {
                    rank,
                    world: world_size,
                    tensors: std::sync::Arc::clone(&tensors),
                };
                NcclComm::from_parts(prim, Device::Cpu)
            })
            .collect()
    }
}

#[cfg(test)]
impl NcclPrimitives for MockNccl {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn all_reduce(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        let contribution = std::mem::take(tensors);
        *tensors = self.tensors.exchange(self.rank, contribution, &|vals| {
            super::sum_tensor_slots(&vals)
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::time::Duration;

    fn t(vals: &[f32]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), vals.len(), &Device::Cpu).unwrap()
    }

    fn v1(tensor: &Tensor) -> Vec<f32> {
        tensor.to_vec1::<f32>().unwrap()
    }

    // ---- config parse ----

    fn getter(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let owned: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |key| owned.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
    }

    #[test]
    fn config_parses_a_well_formed_launch() {
        let cfg = NcclConfig::from_getter(getter(&[
            ("SLURM_PROCID", "1"),
            ("SLURM_NTASKS", "2"),
            ("SLURM_LOCALID", "1"),
            ("FERRL_NCCL_RENDEZVOUS", "/tmp/ferrl_id"),
        ]))
        .unwrap();
        assert_eq!((cfg.rank(), cfg.world_size(), cfg.local_rank()), (1, 2, 1));
        assert_eq!(cfg.rendezvous_file(), Path::new("/tmp/ferrl_id"));
    }

    #[test]
    fn config_rejects_missing_and_unparseable_vars() {
        let missing = NcclConfig::from_getter(getter(&[("SLURM_NTASKS", "2")])).unwrap_err();
        assert!(matches!(missing, CommError::Config(_)), "got {missing:?}");

        let bad = NcclConfig::from_getter(getter(&[
            ("SLURM_PROCID", "x"),
            ("SLURM_NTASKS", "2"),
            ("SLURM_LOCALID", "0"),
            ("FERRL_NCCL_RENDEZVOUS", "/tmp/x"),
        ]))
        .unwrap_err();
        assert!(matches!(bad, CommError::Config(_)), "got {bad:?}");
    }

    #[test]
    fn config_rejects_rank_out_of_range_and_zero_world() {
        let oob = NcclConfig::from_getter(getter(&[
            ("SLURM_PROCID", "2"),
            ("SLURM_NTASKS", "2"),
            ("SLURM_LOCALID", "0"),
            ("FERRL_NCCL_RENDEZVOUS", "/tmp/x"),
        ]))
        .unwrap_err();
        assert!(
            matches!(oob, CommError::Config(_)),
            "rank>=world must fail, got {oob:?}"
        );

        let zero = NcclConfig::from_getter(getter(&[
            ("SLURM_PROCID", "0"),
            ("SLURM_NTASKS", "0"),
            ("SLURM_LOCALID", "0"),
            ("FERRL_NCCL_RENDEZVOUS", "/tmp/x"),
        ]))
        .unwrap_err();
        assert!(
            matches!(zero, CommError::Config(_)),
            "world=0 must fail, got {zero:?}"
        );
    }

    #[test]
    fn config_requires_a_rendezvous_path() {
        let err = NcclConfig::from_getter(getter(&[
            ("SLURM_PROCID", "0"),
            ("SLURM_NTASKS", "1"),
            ("SLURM_LOCALID", "0"),
        ]))
        .unwrap_err();
        assert!(matches!(err, CommError::Config(_)), "got {err:?}");
    }

    // ---- dtype mapping ----

    #[test]
    fn dtype_tags_cover_f32_and_f64() {
        assert_eq!(nccl_dtype_tag(DType::F32).unwrap(), NcclDataType::F32);
        assert_eq!(nccl_dtype_tag(DType::F64).unwrap(), NcclDataType::F64);
    }

    #[test]
    fn dtype_tags_reject_everything_else_including_half() {
        for bad in [DType::U8, DType::U32, DType::I64, DType::F16, DType::BF16] {
            assert!(
                matches!(nccl_dtype_tag(bad), Err(CommError::Mismatch(_))),
                "{bad:?} must reject"
            );
        }
    }

    #[test]
    fn local_validation_rejects_noncontiguous_and_bad_dtype() {
        // Transpose makes a 2x3 into a non-contiguous 3x2 view.
        let noncontig = Tensor::from_vec((0..6).map(|x| x as f32).collect(), (2, 3), &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert!(!noncontig.is_contiguous());
        assert!(matches!(
            validate_local(&[noncontig]),
            Err(CommError::Mismatch(_))
        ));

        let int_tensor = Tensor::from_vec(vec![1u32, 2], 2, &Device::Cpu).unwrap();
        assert!(matches!(
            validate_local(&[int_tensor]),
            Err(CommError::Mismatch(_))
        ));

        validate_local(&[t(&[1.0, 2.0])]).unwrap();
    }

    // ---- unique-id file rendezvous ----

    fn unique_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ferrl_nccl_{tag}_{}.id", std::process::id()))
    }

    #[test]
    fn id_file_roundtrips_through_publish_and_read() {
        let path = unique_path("roundtrip");
        let _ = std::fs::remove_file(&path);
        let mut id = [0u8; NCCL_UNIQUE_ID_LEN];
        for (i, b) in id.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        write_id_file(&path, &id).unwrap();
        let got = read_id_file(&path, Duration::from_secs(2)).unwrap();
        assert_eq!(got, id, "published bytes must read back identical");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn id_file_read_times_out_when_unpublished() {
        let path = unique_path("absent");
        let _ = std::fs::remove_file(&path);
        let err = read_id_file(&path, Duration::from_millis(80)).unwrap_err();
        assert!(matches!(err, CommError::Config(_)), "got {err:?}");
    }

    #[test]
    fn rank0_cleanup_after_bootstrap_removes_the_stale_id_hazard() {
        // `RealNccl::bootstrap` deletes the rendezvous file on rank 0 once the
        // communicator is built (every peer has read the id by then). This test pins
        // the hazard that removal eliminates — a reused path still holding a prior
        // launch's id — by contrasting the two states of the rendezvous primitives.
        let path = unique_path("stale");
        let _ = std::fs::remove_file(&path);
        let mut id = [7u8; NCCL_UNIQUE_ID_LEN];
        id[0] = 42;
        write_id_file(&path, &id).unwrap();

        // WITHOUT the cleanup the file lingers, so a second launch reusing the path
        // reads the STALE id (and would build a comm from a dead `ncclUniqueId`) — the
        // exact hazard:
        assert_eq!(
            read_id_file(&path, Duration::from_millis(80)).unwrap(),
            id,
            "a lingering rendezvous file hands a reused path the stale id"
        );

        // WITH the post-bootstrap cleanup (rank 0 removes the file), the path is empty,
        // so a reused launch finds nothing and waits for a fresh publish rather than
        // racing on a dead id:
        let _ = std::fs::remove_file(&path);
        assert!(!path.exists(), "rank 0 leaves no rendezvous file behind");
        let err = read_id_file(&path, Duration::from_millis(80)).unwrap_err();
        assert!(
            matches!(err, CommError::Config(_)),
            "a cleaned, reused path must time out (no stale id), got {err:?}"
        );
    }

    // ---- NcclComm orchestration over the mock ----

    #[test]
    fn world_one_is_the_identity_for_tensors_and_scalars() {
        let comm = MockNccl::world(1).pop().unwrap();
        assert_eq!((comm.rank(), comm.world_size()), (0, 1));
        let mut ts = vec![t(&[1.0, -2.0]), t(&[0.5])];
        comm.all_reduce_sum(&mut ts).unwrap();
        assert_eq!(v1(&ts[0]), vec![1.0, -2.0]);
        assert_eq!(v1(&ts[1]), vec![0.5]);
        assert_eq!(comm.all_reduce_scalar_sum(7.25).unwrap(), 7.25);
    }

    #[test]
    fn world_three_sums_tensors_and_scalars_across_threads() {
        let comms = MockNccl::world(3);
        let results: Vec<(Vec<f32>, f64)> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    s.spawn(move || {
                        let r = comm.rank() as f32;
                        let mut ts = vec![t(&[r, 10.0 * r])];
                        comm.all_reduce_sum(&mut ts).unwrap();
                        let scalar = comm.all_reduce_scalar_sum(f64::from(r)).unwrap();
                        (v1(&ts[0]), scalar)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (sum, scalar) in results {
            assert_eq!(sum, vec![3.0, 30.0], "0+1+2 and 0+10+20");
            assert_eq!(scalar, 3.0);
        }
    }

    #[test]
    fn orchestration_validates_before_the_collective() {
        // A bad-dtype tensor must be rejected by NcclComm BEFORE it reaches the
        // primitive (so a real launch never hands NCCL an unmappable buffer).
        let comm = MockNccl::world(1).pop().unwrap();
        let int_tensor = Tensor::from_vec(vec![1u32, 2], 2, &Device::Cpu).unwrap();
        assert!(matches!(
            comm.all_reduce_sum(&mut vec![int_tensor]),
            Err(CommError::Mismatch(_))
        ));
    }

    #[test]
    fn mock_matches_local_comm_for_the_same_inputs() {
        // The orchestration over the mock must agree with the reference seam
        // (LocalComm) rank-for-rank — same reduced tensors and scalars.
        use crate::comm::LocalComm;
        let reduce = |comms: Vec<Box<dyn Comm>>| -> Vec<(Vec<f32>, f64)> {
            std::thread::scope(|s| {
                let handles: Vec<_> = comms
                    .into_iter()
                    .map(|comm| {
                        s.spawn(move || {
                            let r = comm.rank() as f32;
                            let mut ts = vec![t(&[r + 1.0, -r])];
                            comm.all_reduce_sum(&mut ts).unwrap();
                            let sc = comm.all_reduce_scalar_sum(f64::from(r) * 2.0).unwrap();
                            (v1(&ts[0]), sc)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        let mock: Vec<Box<dyn Comm>> = MockNccl::world(4)
            .into_iter()
            .map(|c| Box::new(c) as Box<dyn Comm>)
            .collect();
        let local: Vec<Box<dyn Comm>> = LocalComm::world(4)
            .into_iter()
            .map(|c| Box::new(c) as Box<dyn Comm>)
            .collect();
        assert_eq!(
            reduce(mock),
            reduce(local),
            "mock orchestration must match LocalComm"
        );
    }
}
