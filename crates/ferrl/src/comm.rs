//! The distributed communication seam for data and tensor parallelism.
//!
//! GRPO data parallelism in ferrl uses one narrow part of this seam: after
//! each rank folds its local shard's per-item gradients, the per-var sums are
//! **all-reduce-summed** across ranks, and everything downstream of the reduce
//! (the grad-coverage canary, the global norm, clipping, the `AdamW` step) runs
//! on the identical reduced gradient on every rank. Ranks therefore stay in
//! **bitwise lockstep** — same initial weights + same reduced gradients + same
//! optimizer arithmetic — without ever broadcasting parameters. [`Comm`] is
//! that seam's whole surface: rank identity plus two sum-reductions. Tensor
//! parallelism reuses the same surface for projection activations, replicated
//! adapter gradients, canonical rewards, and lockstep control decisions.
//!
//! The surface is deliberately minimal. There is no average-reduce — a mean is
//! a sum plus a global divisor, and the trainer keeps every normalizer in one
//! place (`1 / (grad_accum_steps · world_size)` for the per-item loss scale,
//! the all-reduced window token count for the DAPO normalizer) so the scale of
//! the update lives in exactly one expression per loss type.
//!
//! ## The collective contract
//!
//! Collectives are *rendezvous* operations: **every** rank of the world must
//! call the **same sequence** of collectives, in the same order, with the same
//! tensor count, shapes, and dtypes. A rank that skips a collective the others
//! entered deadlocks the world (the others wait for it forever) — which is why
//! the trainer globalizes every decision that feeds a collective (the
//! degenerate-window skip all-reduces the live count *first*, so all ranks
//! skip together or participate together, a rank with no local items
//! contributing zeros). [`LocalComm`] converts tensor metadata disagreements
//! into loud errors, while callers with rank-local/dynamic payloads use
//! [`Comm::validate_all_reduce_sum`] and globalize its result before entering
//! the payload reduction. `LocalComm` also turns a peer that never arrives into
//! a timeout instead of a silent hang. Rounds carry no operation identity,
//! though (NCCL's collectives
//! are equally untagged): if ranks ever disagree on *which* collective a round
//! is — same types, different call sites — the values are summed together and
//! consumed until the eventual desync trips a timeout, so the same-sequence
//! discipline is the caller's load-bearing obligation, not something the
//! communicator can fully check.
//!
//! ## Implementations
//!
//! - [`SoloComm`] — world 1, every operation is the identity. The default for
//!   [`Trainer::new`](crate::trainer::Trainer::new); the single-rank path
//!   stays bit-identical to the pre-DP trainer.
//! - [`LocalComm`] — N ranks as N **threads of one process**, rendezvousing
//!   over a shared in-memory slot table. This is the CPU-testable oracle
//!   substrate: it has real collective semantics (barrier, deterministic
//!   rank-order reduction, mismatch/timeout fail-loud) with no GPU and no
//!   process orchestration, so the DP equivalence gates run in plain CI.
//! - [`NcclComm`] — the real multi-GPU / multi-process implementation, whose
//!   `unsafe` cudarc-NCCL collective is quarantined behind `--features nccl`
//!   (decision D2: the crate's only `unsafe`, in one gated module). Its
//!   GPU-independent logic (config, dtype/contract validation, scalar packing,
//!   orchestration) is in the default build and CI-tested against an in-memory
//!   mock (the `nccl` submodule).
//!
//! The invariant the trait guarantees — and all the trainer needs — is that
//! every rank receives **rank-identical** reduced values (same reduced gradient
//! on every rank ⇒ same optimizer step). The *fold order* is per-implementation:
//! the in-process reducers ([`LocalComm`] and the mock) combine in **rank order**
//! (slot 0 + slot 1 + …), independent of thread arrival order, so their result is
//! a pure function of the contributions and bit-reproducible against a sequential
//! reference; [`NcclComm`] returns NCCL's rank-identical output, whose internal
//! reduction order is its own (ring/tree, not necessarily rank order). Note the
//! reduced tensors returned to different ranks may share storage — treat them as
//! read-only (the trainer only ever reads gradients out of them).

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use candle_core::Tensor;

mod nccl;

#[cfg(feature = "nccl")]
pub use nccl::RealNccl;
pub use nccl::{NcclComm, NcclConfig, NcclPrimitives};

/// How long a [`LocalComm`] collective waits for its peers before declaring
/// the world dead. Generous: the longest legitimate skew between ranks is one
/// rank's full accumulation window (rollout + backward), seconds at test
/// scale; a peer that aborted before entering the collective never arrives at
/// all, and hitting this bound converts that silent hang into a loud
/// [`CommError::Timeout`].
pub const DEFAULT_COLLECTIVE_TIMEOUT: Duration = Duration::from_secs(300);

/// A communication error from a collective operation.
#[derive(Debug, thiserror::Error)]
pub enum CommError {
    /// The ranks' contributions to a collective disagree (tensor count,
    /// shape, dtype, layout, or device) — a programming error in the caller.
    /// Dynamic callers coordinate the non-collective preflight so this is
    /// surfaced on every rank before payload reduction.
    #[error("collective contract violation: {0}")]
    Mismatch(String),
    /// A peer rank failed to arrive at a collective within the timeout —
    /// it most likely aborted (errored or panicked) before entering it.
    #[error("collective timeout: {0}")]
    Timeout(String),
    /// A previous collective on this world failed (mismatch or timeout);
    /// the world is dead and every subsequent collective fails fast.
    #[error("communicator poisoned by an earlier failure: {0}")]
    Poisoned(String),
    /// A tensor operation inside the reduction failed.
    #[error("candle error inside a collective: {0}")]
    Candle(#[from] candle_core::Error),
    /// The distributed launch environment is malformed — a missing or
    /// unparseable rank/world variable, or an unusable rendezvous path. Raised
    /// during [`NcclComm`] bootstrap, before any collective runs.
    #[error("distributed configuration error: {0}")]
    Config(String),
}

/// The distributed collective seam: rank identity plus sum-reductions.
///
/// Implementations must be [`Send`] (each rank's handle moves into that rank's
/// thread) and every rank must receive **rank-identical** reduced values — the
/// lockstep invariant distributed trainer execution rests on. The fold *order* is
/// implementation-defined: [`LocalComm`] / [`SoloComm`] combine in rank order
/// (bit-reproducible against a sequential reference), whereas [`NcclComm`]
/// returns NCCL's rank-identical output, which need not equal a rank-order fold
/// (NCCL may sum in ring/tree order, and float addition is not associative).
///
/// See the [module docs](self) for the collective contract every caller must
/// uphold (same sequence, same shapes, every rank).
pub trait Comm: std::fmt::Debug + Send {
    /// This handle's rank in `0..world_size()`.
    fn rank(&self) -> usize;

    /// The number of ranks in the world.
    fn world_size(&self) -> usize;

    /// Validate the part of an upcoming tensor sum-reduction contract this
    /// rank can check without entering a collective.
    ///
    /// The default accepts every payload. Implementations with dtype, layout,
    /// or device restrictions override it. A caller whose payload metadata or
    /// validity can differ by rank must coordinate this result and tensor
    /// count/shapes/dtypes through fixed-shape controls before calling
    /// [`all_reduce_sum`](Self::all_reduce_sum).
    ///
    /// # Errors
    ///
    /// Returns [`CommError`] when this rank's payload cannot be reduced by the
    /// implementation. This method itself performs no collective.
    fn validate_all_reduce_sum(&self, _tensors: &[Tensor]) -> Result<(), CommError> {
        Ok(())
    }

    /// Element-wise sum of each tensor across all ranks, in place: on return,
    /// `tensors[i]` holds the sum of every rank's `tensors[i]`, **rank-identical
    /// on every rank**. Every rank must pass the same tensor count, shapes, and
    /// dtypes. This is a caller obligation: implementations such as NCCL cannot
    /// safely discover incompatible payload metadata inside the payload
    /// collective itself.
    ///
    /// # Errors
    ///
    /// Returns [`CommError`] on a contribution mismatch, unsupported local
    /// payload, peer timeout, poisoned world, or failed tensor op. Callers must
    /// not rely on this payload collective to coordinate rank-local validation
    /// failures; use [`validate_all_reduce_sum`](Self::validate_all_reduce_sum)
    /// first when the payload can differ by rank. On error the contents of
    /// `tensors` are unspecified (an implementation may have consumed them) —
    /// the world is dead at that point and the gradients in flight with it.
    fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError>;

    /// Sum of `value` across all ranks, returned **rank-identically** to every rank.
    ///
    /// # Errors
    ///
    /// As [`all_reduce_sum`](Self::all_reduce_sum).
    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError>;
}

/// The world-1 communicator: rank 0 of 1, every reduction is the identity.
///
/// This is [`Trainer::new`](crate::trainer::Trainer::new)'s default. The
/// trainer additionally guards every collective call site on
/// `world_size() > 1`, so the single-rank training path is **byte-for-byte**
/// the pre-DP path (no zero materialization, no extra float operations).
#[derive(Debug, Clone, Copy, Default)]
pub struct SoloComm;

impl Comm for SoloComm {
    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn all_reduce_sum(&self, _tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        Ok(())
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
        Ok(value)
    }
}

/// The world-level kill switch: set on the FIRST collective failure of any
/// kind (mismatch, timeout, a peer panicking mid-protocol) and checked by
/// every collective on entry — shared by the tensor and scalar rendezvous of
/// a world, so a dead world is dead for **every** operation type, not just
/// the one that failed.
#[derive(Default)]
pub(crate) struct PoisonCell(Mutex<Option<String>>);

impl PoisonCell {
    /// Record the first failure (later failures keep the original message).
    fn set(&self, msg: &str) {
        let mut slot = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(msg.to_owned());
        }
    }

    fn get(&self) -> Option<String> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn check(&self) -> Result<(), CommError> {
        match self.get() {
            Some(msg) => Err(CommError::Poisoned(msg)),
            None => Ok(()),
        }
    }
}

/// One round of a rendezvous: the slot table plus the barrier bookkeeping.
struct Round<T> {
    /// One contribution slot per rank, drained by the combiner.
    slots: Vec<Option<T>>,
    /// Ranks that have deposited this round.
    arrived: usize,
    /// Ranks that have collected this round's outcome.
    departed: usize,
    /// The combined result (or the combine failure), present from the last
    /// arrival until the last departure — its presence is also what makes a
    /// fast rank re-entering for the next round wait out the laggards of the
    /// previous one (see `wait_round_open`).
    outcome: Option<Result<T, String>>,
}

/// A reusable N-thread rendezvous: every participant deposits a value, the
/// last arrival combines all of them in rank order, and every participant
/// collects the combined result. Failures poison the whole world permanently
/// (via the shared [`PoisonCell`]). Shared by [`LocalComm`] and the in-memory
/// `MockNccl` test substrate (see the `nccl` submodule).
pub(crate) struct Rendezvous<T> {
    world: usize,
    timeout: Duration,
    poison: Arc<PoisonCell>,
    state: Mutex<Round<T>>,
    cv: Condvar,
}

impl<T: Clone> Rendezvous<T> {
    pub(crate) fn new(world: usize, timeout: Duration, poison: Arc<PoisonCell>) -> Self {
        Self {
            world,
            timeout,
            poison,
            state: Mutex::new(Round {
                slots: (0..world).map(|_| None).collect(),
                arrived: 0,
                departed: 0,
                outcome: None,
            }),
            cv: Condvar::new(),
        }
    }

    /// Lock the round state, converting a poisoned mutex — a peer rank
    /// PANICKED while holding it, leaving the protocol state unreliable —
    /// into a world poison + loud error instead of a panic storm.
    fn lock_round(&self) -> Result<MutexGuard<'_, Round<T>>, CommError> {
        self.state.lock().map_err(|_| {
            let msg = "a peer rank panicked inside a collective on this world";
            self.poison.set(msg);
            CommError::Poisoned(msg.to_owned())
        })
    }

    /// Deposit `value` for `rank`, rendezvous with the other ranks, and return
    /// the combined outcome (every rank receives a clone of the same result).
    pub(crate) fn exchange(
        &self,
        rank: usize,
        value: T,
        combine: &dyn Fn(Vec<T>) -> Result<T, String>,
    ) -> Result<T, CommError> {
        self.poison.check()?;
        let deadline = Instant::now() + self.timeout;
        let mut s = self.lock_round()?;
        s = self.wait_round_open(s, deadline)?;
        debug_assert!(s.slots[rank].is_none(), "rank deposited twice in a round");
        s.slots[rank] = Some(value);
        s.arrived += 1;
        if s.arrived == self.world {
            let vals: Vec<T> = s
                .slots
                .iter_mut()
                .map(|slot| slot.take().expect("every rank deposited"))
                .collect();
            let outcome = combine(vals);
            if let Err(msg) = &outcome {
                self.poison.set(msg);
            }
            s.outcome = Some(outcome);
            self.cv.notify_all();
        } else {
            s = self.wait_outcome(s, deadline)?;
        }
        let out = s
            .outcome
            .as_ref()
            .expect("outcome set for this round")
            .clone();
        s.departed += 1;
        if s.departed == self.world {
            s.arrived = 0;
            s.departed = 0;
            s.outcome = None;
            self.cv.notify_all();
        }
        drop(s);
        out.map_err(CommError::Mismatch)
    }

    /// Wait until no previous round is still draining (its outcome cleared),
    /// so this caller can deposit into a fresh slot table.
    fn wait_round_open<'a>(
        &'a self,
        mut s: MutexGuard<'a, Round<T>>,
        deadline: Instant,
    ) -> Result<MutexGuard<'a, Round<T>>, CommError> {
        loop {
            if let Some(msg) = self.poison.get() {
                return Err(CommError::Poisoned(msg));
            }
            if s.outcome.is_none() {
                return Ok(s);
            }
            s = self.wait_or_poison(s, deadline)?;
        }
    }

    /// Wait (having deposited) until the last arrival publishes the outcome.
    fn wait_outcome<'a>(
        &'a self,
        mut s: MutexGuard<'a, Round<T>>,
        deadline: Instant,
    ) -> Result<MutexGuard<'a, Round<T>>, CommError> {
        loop {
            if s.outcome.is_some() {
                return Ok(s);
            }
            if let Some(msg) = self.poison.get() {
                return Err(CommError::Poisoned(msg));
            }
            s = self.wait_or_poison(s, deadline)?;
        }
    }

    /// One bounded condvar wait; on deadline expiry, poison the world (waking
    /// every peer into a loud failure) and return the timeout error.
    fn wait_or_poison<'a>(
        &'a self,
        s: MutexGuard<'a, Round<T>>,
        deadline: Instant,
    ) -> Result<MutexGuard<'a, Round<T>>, CommError> {
        let now = Instant::now();
        if now >= deadline {
            let msg = format!(
                "a peer rank did not reach the collective within {:?} — it most \
                 likely aborted before entering it",
                self.timeout
            );
            self.poison.set(&msg);
            self.cv.notify_all();
            return Err(CommError::Timeout(msg));
        }
        let (s, _) = self.cv.wait_timeout(s, deadline - now).map_err(|_| {
            let msg = "a peer rank panicked inside a collective on this world";
            self.poison.set(msg);
            CommError::Poisoned(msg.to_owned())
        })?;
        Ok(s)
    }
}

/// A single-process, N-thread world: each rank is a thread holding one
/// [`LocalComm`] handle, and collectives rendezvous over shared memory.
///
/// Mint a world with [`LocalComm::world`] and move each handle into its
/// rank's thread (`std::thread::scope` works well — policies only need to be
/// `Send`). Handles are deliberately not `Clone`: a rank identity must not be
/// shared between threads.
///
/// Any collective failure (contribution mismatch, peer timeout) **poisons**
/// the world: the failing collective errors on every rank, and every
/// subsequent collective fails fast with [`CommError::Poisoned`] — a dead
/// world is never silently half-alive.
pub struct LocalComm {
    rank: usize,
    world: usize,
    tensors: Arc<Rendezvous<Vec<Tensor>>>,
    scalars: Arc<Rendezvous<f64>>,
}

impl std::fmt::Debug for LocalComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalComm")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .finish_non_exhaustive()
    }
}

// Rendezvous<T> is Send + Sync for T: Send (Mutex/Condvar over T), and both
// Vec<Tensor> and f64 are Send, so the derive-free impls below are automatic;
// this assertion just pins the property the thread::scope usage relies on.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<LocalComm>();
};

impl LocalComm {
    /// Mint the `world_size` handles of a fresh world, with the default
    /// collective timeout ([`DEFAULT_COLLECTIVE_TIMEOUT`]). Handle `i` of the
    /// returned vec is rank `i`.
    ///
    /// # Panics
    ///
    /// Panics if `world_size` is zero — a world with no ranks is a caller bug.
    #[must_use]
    pub fn world(world_size: usize) -> Vec<Self> {
        Self::world_with_timeout(world_size, DEFAULT_COLLECTIVE_TIMEOUT)
    }

    /// As [`world`](Self::world), with an explicit collective timeout (how
    /// long a collective waits for its peers before poisoning the world).
    ///
    /// # Panics
    ///
    /// Panics if `world_size` is zero.
    #[must_use]
    pub fn world_with_timeout(world_size: usize, timeout: Duration) -> Vec<Self> {
        assert!(world_size > 0, "a world needs at least one rank");
        // ONE poison cell for the whole world, shared by both rendezvous:
        // a failure in either operation type kills the other too (the
        // "dead world" contract is per WORLD, not per collective kind).
        let poison = Arc::new(PoisonCell::default());
        let tensors = Arc::new(Rendezvous::new(world_size, timeout, Arc::clone(&poison)));
        let scalars = Arc::new(Rendezvous::new(world_size, timeout, poison));
        (0..world_size)
            .map(|rank| Self {
                rank,
                world: world_size,
                tensors: Arc::clone(&tensors),
                scalars: Arc::clone(&scalars),
            })
            .collect()
    }
}

impl Comm for LocalComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        let contribution = std::mem::take(tensors);
        *tensors = self
            .tensors
            .exchange(self.rank, contribution, &|vals| sum_tensor_slots(&vals))?;
        Ok(())
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
        self.scalars
            .exchange(self.rank, value, &|vals| Ok(vals.iter().sum()))
    }
}

/// Combine the ranks' contribution vecs into their element-wise sums, in rank
/// order, validating the collective contract (same count, shape, dtype). Shared
/// with the `MockNccl` test substrate (the `nccl` submodule) so the mock reduces
/// with the identical rank-order semantics [`LocalComm`] uses.
pub(crate) fn sum_tensor_slots(vals: &[Vec<Tensor>]) -> Result<Vec<Tensor>, String> {
    let n = vals[0].len();
    if let Some((r, v)) = vals.iter().enumerate().find(|(_, v)| v.len() != n) {
        return Err(format!(
            "all_reduce_sum: rank 0 contributed {n} tensors but rank {r} \
             contributed {}",
            v.len()
        ));
    }
    (0..n).map(|i| sum_one_slot(vals, i)).collect()
}

/// Sum slot `i` across the ranks (in rank order), validating shape and dtype
/// against rank 0's contribution.
fn sum_one_slot(vals: &[Vec<Tensor>], i: usize) -> Result<Tensor, String> {
    let spec = (vals[0][i].shape().clone(), vals[0][i].dtype());
    let mut acc = vals[0][i].clone();
    for (r, v) in vals.iter().enumerate().skip(1) {
        let t = &v[i];
        if (t.shape().clone(), t.dtype()) != spec {
            return Err(format!(
                "all_reduce_sum: tensor {i} mismatch — rank 0 contributed \
                 {:?}/{:?}, rank {r} contributed {:?}/{:?}",
                spec.0,
                spec.1,
                t.shape(),
                t.dtype()
            ));
        }
        acc = acc
            .add(t)
            .map_err(|e| format!("all_reduce_sum: summing tensor {i} failed: {e}"))?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn t(vals: &[f32]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), vals.len(), &Device::Cpu).unwrap()
    }

    fn v1(tensor: &Tensor) -> Vec<f32> {
        tensor.to_vec1::<f32>().unwrap()
    }

    #[test]
    fn solo_comm_is_the_identity() {
        let comm = SoloComm;
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
        let mut ts = vec![t(&[1.0, 2.0])];
        comm.all_reduce_sum(&mut ts).unwrap();
        assert_eq!(v1(&ts[0]), vec![1.0, 2.0]);
        assert_eq!(comm.all_reduce_scalar_sum(3.5).unwrap(), 3.5);
    }

    #[test]
    fn world_one_local_comm_sums_to_itself() {
        let comm = LocalComm::world(1).pop().unwrap();
        assert_eq!((comm.rank(), comm.world_size()), (0, 1));
        let mut ts = vec![t(&[1.0, -2.0]), t(&[0.5])];
        comm.all_reduce_sum(&mut ts).unwrap();
        assert_eq!(v1(&ts[0]), vec![1.0, -2.0]);
        assert_eq!(v1(&ts[1]), vec![0.5]);
        assert_eq!(comm.all_reduce_scalar_sum(7.0).unwrap(), 7.0);
    }

    #[test]
    fn world_three_sums_tensors_and_scalars_across_threads() {
        let comms = LocalComm::world(3);
        let results: Vec<(Vec<f32>, f64)> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    s.spawn(move || {
                        let r = comm.rank() as f32;
                        let mut ts = vec![t(&[r, 10.0 * r]), t(&[1.0])];
                        comm.all_reduce_sum(&mut ts).unwrap();
                        let scalar = comm.all_reduce_scalar_sum(f64::from(r)).unwrap();
                        assert_eq!(v1(&ts[1]), vec![3.0], "second slot sums too");
                        (v1(&ts[0]), scalar)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (sum, scalar) in results {
            assert_eq!(sum, vec![3.0, 30.0], "0 + 1 + 2 and 0 + 10 + 20");
            assert_eq!(scalar, 3.0);
        }
    }

    #[test]
    fn rendezvous_is_reusable_across_many_rounds() {
        let comms = LocalComm::world(2);
        std::thread::scope(|s| {
            for comm in comms {
                s.spawn(move || {
                    for round in 0..50u32 {
                        let x = f64::from(round) + f64::from(comm.rank() as u32);
                        let got = comm.all_reduce_scalar_sum(x).unwrap();
                        assert_eq!(got, 2.0 * f64::from(round) + 1.0);
                    }
                });
            }
        });
    }

    #[test]
    fn shape_mismatch_fails_on_every_rank_and_poisons_the_world() {
        let comms = LocalComm::world(2);
        let errs: Vec<(CommError, CommError)> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    s.spawn(move || {
                        // Rank 0 contributes a 2-element tensor, rank 1 a
                        // 3-element one: the collective must fail LOUD on both.
                        let mut ts = if comm.rank() == 0 {
                            vec![t(&[1.0, 2.0])]
                        } else {
                            vec![t(&[1.0, 2.0, 3.0])]
                        };
                        let first = comm.all_reduce_sum(&mut ts).unwrap_err();
                        // The world is now dead: every later collective fails fast.
                        let second = comm.all_reduce_sum(&mut vec![t(&[0.0])]).unwrap_err();
                        (first, second)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (first, second) in errs {
            assert!(
                matches!(first, CommError::Mismatch(_)),
                "want Mismatch, got {first:?}"
            );
            assert!(
                matches!(second, CommError::Poisoned(_)),
                "want Poisoned after a failed round, got {second:?}"
            );
        }
    }

    #[test]
    fn tensor_count_mismatch_is_rejected() {
        let comms = LocalComm::world(2);
        let errs: Vec<CommError> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    s.spawn(move || {
                        let mut ts = if comm.rank() == 0 {
                            vec![t(&[1.0])]
                        } else {
                            vec![t(&[1.0]), t(&[2.0])]
                        };
                        comm.all_reduce_sum(&mut ts).unwrap_err()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for e in errs {
            assert!(matches!(e, CommError::Mismatch(_)), "got {e:?}");
        }
    }

    #[test]
    fn dtype_mismatch_is_rejected() {
        let comms = LocalComm::world(2);
        let errs: Vec<CommError> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    s.spawn(move || {
                        let tensor = if comm.rank() == 0 {
                            t(&[1.0])
                        } else {
                            t(&[1.0]).to_dtype(DType::F64).unwrap()
                        };
                        comm.all_reduce_sum(&mut vec![tensor]).unwrap_err()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for e in errs {
            assert!(matches!(e, CommError::Mismatch(_)), "got {e:?}");
        }
    }

    #[test]
    fn an_absent_peer_times_out_loudly_instead_of_hanging() {
        let mut comms = LocalComm::world_with_timeout(2, Duration::from_millis(50));
        let _absent_rank_1 = comms.pop().unwrap();
        let rank0 = comms.pop().unwrap();
        // Rank 1 never calls the collective: rank 0 must get a Timeout (not
        // hang), and the world must be poisoned afterwards.
        let err = rank0.all_reduce_scalar_sum(1.0).unwrap_err();
        assert!(matches!(err, CommError::Timeout(_)), "got {err:?}");
        let err = rank0.all_reduce_scalar_sum(1.0).unwrap_err();
        assert!(matches!(err, CommError::Poisoned(_)), "got {err:?}");
    }

    #[test]
    fn reduction_is_deterministic_in_rank_order_not_arrival_order() {
        // Values chosen so f64 summation order matters at the ulp level if it
        // ever stopped being rank-ordered: (big + tiny) + tiny != big + (tiny
        // + tiny) in floating point. Stagger arrivals both ways and pin the
        // identical rank-order result.
        let big = 1.0e16;
        let tiny = 1.0;
        let run = |delay_rank: usize| -> Vec<f64> {
            let comms = LocalComm::world(3);
            std::thread::scope(|s| {
                let handles: Vec<_> = comms
                    .into_iter()
                    .map(|comm| {
                        s.spawn(move || {
                            if comm.rank() == delay_rank {
                                std::thread::sleep(Duration::from_millis(30));
                            }
                            let x = if comm.rank() == 0 { big } else { tiny };
                            comm.all_reduce_scalar_sum(x).unwrap()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        let first = run(0);
        let second = run(2);
        assert_eq!(first, second, "arrival order must not change the sum");
        assert!(first.iter().all(|&x| x == first[0]));
        // Pin the CANONICAL rank-order left fold, not just any deterministic
        // order: (big + tiny) + tiny stays `big` (each tiny is below the ulp),
        // while a reversed fold yields big + 2 — a reverse-order combine is
        // equally arrival-independent and only this catches it.
        assert_eq!(
            first[0],
            (big + tiny) + tiny,
            "the reduction must fold in ascending rank order"
        );
    }

    #[test]
    fn poison_crosses_collective_types() {
        // The poison cell is per WORLD, not per rendezvous: after a TENSOR
        // collective fails, a SCALAR collective on the same world must fail
        // fast as Poisoned — not silently succeed on a half-dead world.
        let comms = LocalComm::world(2);
        let errs: Vec<CommError> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    s.spawn(move || {
                        let mut ts = if comm.rank() == 0 {
                            vec![t(&[1.0, 2.0])]
                        } else {
                            vec![t(&[1.0, 2.0, 3.0])]
                        };
                        comm.all_reduce_sum(&mut ts).unwrap_err();
                        comm.all_reduce_scalar_sum(1.0).unwrap_err()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for e in errs {
            assert!(
                matches!(e, CommError::Poisoned(_)),
                "a scalar collective after a tensor failure must be Poisoned, got {e:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "at least one rank")]
    fn world_of_zero_ranks_panics() {
        let _ = LocalComm::world(0);
    }
}
