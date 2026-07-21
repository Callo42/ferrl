//! Tensor-parallel planning primitives.
//!
//! This module is the contract layer beneath the distributed model execution
//! paths. It centralizes the rank/world validation and projection-axis slicing
//! rules shared by Dense/Qwen and Gemma 4 tensor-parallel implementations.
//! Keeping that policy separate from Dense/Gemma loader code gives
//! the CPU tests a small deterministic oracle: one shard must be the identity,
//! and N contiguous shards must reassemble the same projection/log-prob values
//! as the unsharded path.

use candle_core::{DType, Error as CandleError, Result as CandleResult, Tensor};

use crate::blocks::frozen_linear;
use crate::comm::Comm;

fn plan_to_candle<T>(result: Result<T, TensorParallelError>) -> CandleResult<T> {
    result.map_err(|e| candle_core::Error::Msg(e.to_string()))
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct TensorParallelCommFailure(String);

pub(crate) fn comm_to_candle<T>(result: Result<T, crate::comm::CommError>) -> CandleResult<T> {
    result.map_err(|error| CandleError::WrappedContext {
        wrapped: Box::new(TensorParallelCommFailure(error.to_string())),
        context: "tensor-parallel collective failed".to_owned(),
    })
}

pub(crate) fn coordinate_local_candle_call<T>(
    comm: &dyn Comm,
    label: &str,
    call: impl FnOnce() -> CandleResult<T>,
) -> CandleResult<T> {
    let local = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
        Ok(result) => result,
        Err(payload) => {
            let detail = if let Some(message) = payload.downcast_ref::<&str>() {
                *message
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.as_str()
            } else {
                "panic payload was not a string"
            };
            Err(CandleError::Msg(format!("{label} panicked: {detail}")))
        }
    };
    if comm.world_size() <= 1 {
        return local;
    }
    let failed =
        comm_to_candle(comm.all_reduce_scalar_sum(if local.is_err() { 1.0 } else { 0.0 }))?;
    match local {
        Err(error) => Err(error),
        Ok(_) if failed > 0.0 => Err(CandleError::Msg(format!(
            "{label} failed on a peer tensor-parallel rank; aborting before the next collective"
        ))),
        Ok(value) => Ok(value),
    }
}

pub(crate) fn is_comm_failure(error: &CandleError) -> bool {
    fn wrapped_is_comm_failure(error: &(dyn std::error::Error + Send + Sync + 'static)) -> bool {
        error.downcast_ref::<TensorParallelCommFailure>().is_some()
            || error
                .downcast_ref::<CandleError>()
                .is_some_and(is_comm_failure)
    }

    match error {
        CandleError::WithBacktrace { inner, .. }
        | CandleError::WithPath { inner, .. }
        | CandleError::Context { inner, .. } => is_comm_failure(inner),
        CandleError::WrappedContext { wrapped, .. } => wrapped_is_comm_failure(wrapped.as_ref()),
        _ => false,
    }
}

/// A tensor-parallel rank/world assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorParallelPlan {
    rank: usize,
    world_size: usize,
}

impl Default for TensorParallelPlan {
    fn default() -> Self {
        Self::single()
    }
}

impl TensorParallelPlan {
    /// The world-1 identity plan.
    #[must_use]
    pub const fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }

    /// Build a plan, failing closed on malformed rank/world coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`TensorParallelError`] if `world_size == 0` or `rank` is outside
    /// `0..world_size`.
    pub fn new(rank: usize, world_size: usize) -> Result<Self, TensorParallelError> {
        if world_size == 0 {
            return Err(TensorParallelError::InvalidWorldSize { world_size });
        }
        if rank >= world_size {
            return Err(TensorParallelError::RankOutsideWorld { rank, world_size });
        }
        Ok(Self { rank, world_size })
    }

    /// This rank in `0..world_size`.
    #[must_use]
    pub const fn rank(self) -> usize {
        self.rank
    }

    /// The tensor-parallel world size.
    #[must_use]
    pub const fn world_size(self) -> usize {
        self.world_size
    }

    /// Whether this plan is a real multi-rank shard rather than the identity.
    #[must_use]
    pub const fn is_sharded(self) -> bool {
        self.world_size > 1
    }

    /// Return this rank's contiguous shard of an axis.
    ///
    /// The first production contract is intentionally strict: an axis must be
    /// evenly divisible by `world_size`. Uneven shards can be added later with a
    /// separate explicit proof; silently rounding here would make rank layouts
    /// ambiguous.
    ///
    /// # Errors
    ///
    /// Returns [`TensorParallelError`] if `axis_len == 0` or it is not divisible
    /// by `world_size`.
    pub fn shard_axis(
        self,
        label: &'static str,
        axis_len: usize,
    ) -> Result<ShardRange, TensorParallelError> {
        if axis_len == 0 {
            return Err(TensorParallelError::EmptyAxis { label });
        }
        if !axis_len.is_multiple_of(self.world_size) {
            return Err(TensorParallelError::UnevenAxis {
                label,
                axis_len,
                world_size: self.world_size,
            });
        }
        let len = axis_len / self.world_size;
        Ok(ShardRange {
            start: self.rank * len,
            len,
            full_len: axis_len,
        })
    }

    /// Validate the dense-transformer dimensions that current TP layouts need
    /// to split exactly across ranks.
    ///
    /// This is model-agnostic on purpose: Dense Qwen/Llama and dense Gemma 4 use
    /// the same attention-head, KV-head, MLP-intermediate, hidden, and vocab
    /// divisibility constraints for the conservative contiguous-shard layout.
    ///
    /// # Errors
    ///
    /// Returns [`TensorParallelError`] on the first zero or non-divisible axis.
    pub fn validate_transformer_dims(
        self,
        dims: TensorParallelDims,
    ) -> Result<(), TensorParallelError> {
        self.shard_axis("hidden_size", dims.hidden_size)?;
        self.shard_axis("intermediate_size", dims.intermediate_size)?;
        self.shard_axis("vocab_size", dims.vocab_size)?;
        self.shard_axis("num_attention_heads", dims.num_attention_heads)?;
        self.shard_axis("num_key_value_heads", dims.num_key_value_heads)?;
        Ok(())
    }
}

/// Build a [`TensorParallelPlan`] from a communicator's rank identity.
///
/// # Errors
///
/// Returns a candle error if the communicator reports an invalid rank/world
/// coordinate.
pub fn plan_from_comm(comm: &dyn Comm) -> CandleResult<TensorParallelPlan> {
    plan_to_candle(TensorParallelPlan::new(comm.rank(), comm.world_size()))
}

/// Fail loud if `comm` and `plan` disagree about rank identity.
///
/// # Errors
///
/// Returns a candle error if either the rank or world size differs.
pub fn validate_comm_plan(plan: TensorParallelPlan, comm: &dyn Comm) -> CandleResult<()> {
    if plan.rank() != comm.rank() || plan.world_size() != comm.world_size() {
        candle_core::bail!(
            "tensor_parallel plan rank/world ({}, {}) does not match communicator ({}, {})",
            plan.rank(),
            plan.world_size(),
            comm.rank(),
            comm.world_size()
        );
    }
    Ok(())
}

/// A contiguous rank-local slice of a full axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardRange {
    /// Inclusive start index in the full axis.
    pub start: usize,
    /// Number of elements owned by this rank.
    pub len: usize,
    /// Full unsharded axis length.
    pub full_len: usize,
}

impl ShardRange {
    /// Exclusive end index in the full axis.
    #[must_use]
    pub const fn end(self) -> usize {
        self.start + self.len
    }
}

/// Dense-transformer dimensions a conservative tensor-parallel layout must
/// split without remainder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorParallelDims {
    /// Model hidden width.
    pub hidden_size: usize,
    /// MLP intermediate width.
    pub intermediate_size: usize,
    /// Vocabulary size, for vocab-parallel or tied-head policies.
    pub vocab_size: usize,
    /// Query/output attention head count.
    pub num_attention_heads: usize,
    /// Key/value head count before GQA repetition.
    pub num_key_value_heads: usize,
}

/// Tensor-parallel planning failures.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TensorParallelError {
    /// A tensor-parallel world cannot be empty.
    #[error("tensor_parallel.world_size must be positive, got {world_size}")]
    InvalidWorldSize {
        /// Supplied world size.
        world_size: usize,
    },
    /// The rank must be inside the world.
    #[error("tensor_parallel.rank {rank} outside world_size {world_size}")]
    RankOutsideWorld {
        /// Supplied rank.
        rank: usize,
        /// Supplied world size.
        world_size: usize,
    },
    /// A sharded axis cannot be empty.
    #[error("tensor_parallel axis {label} must be nonzero")]
    EmptyAxis {
        /// Axis name.
        label: &'static str,
    },
    /// Current TP policy requires evenly divisible axes.
    #[error(
        "tensor_parallel axis {label} length {axis_len} is not divisible by world_size {world_size}"
    )]
    UnevenAxis {
        /// Axis name.
        label: &'static str,
        /// Full axis length.
        axis_len: usize,
        /// Tensor-parallel world size.
        world_size: usize,
    },
}

/// Output-column-parallel linear shard: `y_r = x @ W_r^T`, where `W_r` is this
/// rank's contiguous slice of the output axis (`dim 0`) of `weight`.
///
/// # Errors
///
/// Returns a candle error if `weight` is not rank-2, if the output axis is not
/// evenly shardable by `plan`, or if the linear op fails.
pub fn column_parallel_linear(
    x: &Tensor,
    weight: &Tensor,
    plan: TensorParallelPlan,
    label: &'static str,
) -> CandleResult<Tensor> {
    let (out, _in) = weight.dims2()?;
    let shard = plan_to_candle(plan.shard_axis(label, out))?;
    let weight = weight.narrow(0, shard.start, shard.len)?;
    frozen_linear(x, &weight)
}

/// Row/input-parallel linear partial from a full input tensor. The caller must
/// sum the returned partials from every rank in rank order to recover the
/// unsharded linear result.
///
/// # Errors
///
/// Returns a candle error if `weight` is not rank-2, if the input axis is not
/// evenly shardable by `plan`, if `x` has no trailing feature dimension, or if
/// the linear op fails.
pub fn row_parallel_linear_partial(
    x: &Tensor,
    weight: &Tensor,
    plan: TensorParallelPlan,
    label: &'static str,
) -> CandleResult<Tensor> {
    let (_out, in_) = weight.dims2()?;
    let shard = plan_to_candle(plan.shard_axis(label, in_))?;
    let dim = x
        .rank()
        .checked_sub(1)
        .ok_or_else(|| candle_core::Error::Msg("row-parallel input must have rank >= 1".into()))?;
    let x = x.narrow(dim, shard.start, shard.len)?;
    row_parallel_linear_partial_from_shard(&x, weight, plan, label)
}

/// Row/input-parallel linear partial from an already-sharded input tensor.
///
/// # Errors
///
/// Returns a candle error if `weight` is not rank-2, if the input axis is not
/// evenly shardable by `plan`, or if the linear op fails.
pub fn row_parallel_linear_partial_from_shard(
    x_shard: &Tensor,
    weight: &Tensor,
    plan: TensorParallelPlan,
    label: &'static str,
) -> CandleResult<Tensor> {
    let (_out, in_) = weight.dims2()?;
    let shard = plan_to_candle(plan.shard_axis(label, in_))?;
    let weight = weight.narrow(1, shard.start, shard.len)?;
    frozen_linear(x_shard, &weight)
}

/// Concatenate column-parallel output shards in rank order.
///
/// # Errors
///
/// Returns a candle error if the shard list is empty or if concatenation fails.
pub fn concat_column_shards(shards: &[Tensor]) -> CandleResult<Tensor> {
    if shards.is_empty() {
        candle_core::bail!("concat_column_shards: no shards");
    }
    let axis = shards[0]
        .rank()
        .checked_sub(1)
        .ok_or_else(|| candle_core::Error::Msg("tensor must have rank >= 1".into()))?;
    Tensor::cat(shards, axis)
}

/// Sum row-parallel partial outputs in rank order.
///
/// # Errors
///
/// Returns a candle error if the partial list is empty, if shapes differ, or if
/// any tensor addition fails.
pub fn sum_row_parallel_partials(partials: &[Tensor]) -> CandleResult<Tensor> {
    let Some((first, rest)) = partials.split_first() else {
        candle_core::bail!("sum_row_parallel_partials: no partials");
    };
    let dims = first.dims().to_vec();
    let mut out = first.clone();
    for part in rest {
        if part.dims() != dims.as_slice() {
            candle_core::bail!(
                "sum_row_parallel_partials: shape mismatch {:?} vs {:?}",
                part.dims(),
                dims
            );
        }
        out = (&out + part)?;
    }
    Ok(out)
}

/// Row-parallel activation all-reduce with a local straight-through gradient.
///
/// A tensor-parallel row projection computes a rank-local partial `p_r` and the
/// forward value must be `sum_r p_r` on every rank. The existing [`Comm`] seam
/// moves values and does not define a distributed autograd op, so this helper
/// rebuilds the returned tensor as:
///
/// `p_r + detach(all_reduce_sum(p_r) - p_r)`
///
/// The forward value is exactly the reduced sum, while the local backward
/// derivative is the identity into `p_r`, which is the adjoint needed for the
/// row-parallel partial on a rank that computes the same downstream loss.
///
/// Half-precision activations are staged through F32 for the collective because
/// ferrl's current NCCL bridge only supports F32/F64 reductions.
///
/// # Errors
///
/// Returns a candle error if the plan and communicator disagree, if the
/// communicator fails, or if any tensor op fails.
pub fn all_reduce_sum_straight_through(
    partial: &Tensor,
    plan: TensorParallelPlan,
    comm: &dyn Comm,
) -> CandleResult<Tensor> {
    let staged = coordinate_local_candle_call(
        comm,
        "tensor-parallel activation all-reduce staging",
        || {
            validate_comm_plan(plan, comm)?;
            if !plan.is_sharded() {
                return Ok(None);
            }
            let original_dtype = partial.dtype();
            let staged = match original_dtype {
                DType::F32 | DType::F64 => partial.contiguous()?,
                _ => partial.to_dtype(DType::F32)?.contiguous()?,
            };
            let reduced = vec![staged];
            comm.validate_all_reduce_sum(&reduced).map_err(|error| {
                CandleError::Msg(format!(
                    "tensor-parallel activation all-reduce payload is invalid: {error}"
                ))
            })?;
            Ok(Some((original_dtype, reduced)))
        },
    )?;
    let Some((original_dtype, mut reduced)) = staged else {
        return Ok(partial.clone());
    };
    comm_to_candle(comm.all_reduce_sum(&mut reduced))?;
    coordinate_local_candle_call(
        comm,
        "tensor-parallel activation all-reduce readback",
        || {
            let Some(mut reduced) = reduced.pop() else {
                candle_core::bail!("tensor_parallel all-reduce returned no tensors");
            };
            if reduced.dtype() != original_dtype {
                reduced = reduced.to_dtype(original_dtype)?;
            }
            let correction = reduced.broadcast_sub(partial)?.detach();
            partial.broadcast_add(&correction)
        },
    )
}

/// Sum a detached tensor value across TP ranks.
///
/// Explicit rematerialized backward uses this for cotangents, which are values
/// rather than graph edges and must be rank-identical before replaying the
/// preceding segment.
pub(crate) fn all_reduce_sum_value(
    value: &Tensor,
    plan: TensorParallelPlan,
    comm: &dyn Comm,
) -> CandleResult<Tensor> {
    let staged =
        coordinate_local_candle_call(comm, "tensor-parallel cotangent all-reduce staging", || {
            validate_comm_plan(plan, comm)?;
            if !plan.is_sharded() {
                return Ok(None);
            }
            let original_dtype = value.dtype();
            let staged = match original_dtype {
                DType::F32 | DType::F64 => value.detach().contiguous()?,
                _ => value.to_dtype(DType::F32)?.detach().contiguous()?,
            };
            let reduced = vec![staged];
            comm.validate_all_reduce_sum(&reduced).map_err(|error| {
                CandleError::Msg(format!(
                    "tensor-parallel cotangent all-reduce payload is invalid: {error}"
                ))
            })?;
            Ok(Some((original_dtype, reduced)))
        })?;
    let Some((original_dtype, mut reduced)) = staged else {
        return Ok(value.detach());
    };
    comm_to_candle(comm.all_reduce_sum(&mut reduced))?;
    coordinate_local_candle_call(
        comm,
        "tensor-parallel cotangent all-reduce readback",
        || {
            let Some(mut reduced) = reduced.pop() else {
                candle_core::bail!("tensor_parallel cotangent all-reduce returned no tensors");
            };
            if reduced.dtype() != original_dtype {
                reduced = reduced.to_dtype(original_dtype)?;
            }
            Ok(reduced.detach())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::{Comm, CommError, LocalComm};
    use candle_core::{Device, Var, D};
    use candle_nn::ops::log_softmax;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct CountAndCorruptReadbackComm<C> {
        inner: C,
        tensor_calls: Arc<AtomicUsize>,
        clear_after_reduce: bool,
    }

    impl<C: Comm> Comm for CountAndCorruptReadbackComm<C> {
        fn rank(&self) -> usize {
            self.inner.rank()
        }

        fn world_size(&self) -> usize {
            self.inner.world_size()
        }

        fn validate_all_reduce_sum(&self, tensors: &[Tensor]) -> Result<(), CommError> {
            self.inner.validate_all_reduce_sum(tensors)
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
            self.tensor_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.all_reduce_sum(tensors)?;
            if self.clear_after_reduce {
                tensors.clear();
            }
            Ok(())
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
            self.inner.all_reduce_scalar_sum(value)
        }
    }

    fn tensor(data: &[f32], shape: impl Into<candle_core::Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()
    }

    fn all_plans(world_size: usize) -> Vec<TensorParallelPlan> {
        (0..world_size)
            .map(|rank| TensorParallelPlan::new(rank, world_size).unwrap())
            .collect()
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32) {
        assert_eq!(a.dims(), b.dims());
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max = av
            .iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max <= tol,
            "max diff {max} > {tol}\nleft={av:?}\nright={bv:?}"
        );
    }

    fn target_logprobs(logits: &Tensor, targets: &Tensor) -> CandleResult<Tensor> {
        let logp = log_softmax(logits, D::Minus1)?;
        logp.gather(&targets.unsqueeze(D::Minus1)?, D::Minus1)?
            .squeeze(D::Minus1)
    }

    #[test]
    fn rank_world_validation_fails_closed() {
        assert_eq!(
            TensorParallelPlan::new(0, 0).unwrap_err(),
            TensorParallelError::InvalidWorldSize { world_size: 0 }
        );
        assert_eq!(
            TensorParallelPlan::new(2, 2).unwrap_err(),
            TensorParallelError::RankOutsideWorld {
                rank: 2,
                world_size: 2
            }
        );
    }

    #[test]
    fn contiguous_shards_cover_an_axis_in_rank_order() {
        let ranges: Vec<_> = all_plans(4)
            .into_iter()
            .map(|plan| plan.shard_axis("hidden", 16).unwrap())
            .collect();
        assert_eq!(
            ranges,
            vec![
                ShardRange {
                    start: 0,
                    len: 4,
                    full_len: 16
                },
                ShardRange {
                    start: 4,
                    len: 4,
                    full_len: 16
                },
                ShardRange {
                    start: 8,
                    len: 4,
                    full_len: 16
                },
                ShardRange {
                    start: 12,
                    len: 4,
                    full_len: 16
                }
            ]
        );
    }

    #[test]
    fn uneven_model_axes_are_rejected() {
        let plan = TensorParallelPlan::new(1, 4).unwrap();
        let err = plan
            .validate_transformer_dims(TensorParallelDims {
                hidden_size: 8,
                intermediate_size: 12,
                vocab_size: 16,
                num_attention_heads: 6,
                num_key_value_heads: 4,
            })
            .unwrap_err();
        assert_eq!(
            err,
            TensorParallelError::UnevenAxis {
                label: "num_attention_heads",
                axis_len: 6,
                world_size: 4
            }
        );
    }

    #[test]
    fn one_shard_and_n_shard_projection_logits_and_logprobs_match() {
        let x = tensor(
            &[
                0.10, -0.20, 0.30, 0.40, -0.50, 0.60, -0.70, 0.80, 0.90, -1.00, 1.10, -1.20, 1.30,
                1.40, -1.50, 1.60, -1.70, 1.80, 1.90, -2.00, 2.10, -2.20, 2.30, 2.40,
            ],
            (2, 3, 4),
        );
        let gate = tensor(
            &[
                0.01, 0.02, -0.03, 0.04, -0.05, 0.06, 0.07, -0.08, 0.09, -0.10, 0.11, 0.12, -0.13,
                0.14, -0.15, 0.16, 0.17, 0.18, -0.19, 0.20, -0.21, 0.22, 0.23, -0.24,
            ],
            (6, 4),
        );
        let up = tensor(
            &[
                -0.03, 0.05, 0.07, -0.09, 0.11, 0.13, -0.15, 0.17, -0.19, 0.21, 0.23, -0.25, 0.27,
                -0.29, 0.31, 0.33, -0.35, 0.37, -0.39, 0.41, 0.43, -0.45, 0.47, 0.49,
            ],
            (6, 4),
        );
        let down = tensor(
            &[
                0.02, -0.04, 0.06, -0.08, 0.10, -0.12, -0.14, 0.16, -0.18, 0.20, -0.22, 0.24, 0.26,
                -0.28, 0.30, -0.32, 0.34, -0.36, -0.38, 0.40, -0.42, 0.44, -0.46, 0.48,
            ],
            (4, 6),
        );
        let head = tensor(
            &[
                0.09, -0.08, 0.07, -0.06, -0.05, 0.04, -0.03, 0.02, 0.01, 0.03, -0.05, 0.07, -0.09,
                0.11, 0.13, -0.15, 0.17, -0.19, 0.21, 0.23,
            ],
            (5, 4),
        );
        let targets = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 0], (2, 3), &Device::Cpu).unwrap();

        let gate_full = frozen_linear(&x, &gate).unwrap();
        let up_full = frozen_linear(&x, &up).unwrap();
        let full_hidden = gate_full.broadcast_mul(&up_full).unwrap();
        let full_down = frozen_linear(&full_hidden, &down).unwrap();
        let full_logits = frozen_linear(&full_down, &head).unwrap();
        let full_logp = target_logprobs(&full_logits, &targets).unwrap();

        let one = TensorParallelPlan::single();
        let one_gate = column_parallel_linear(&x, &gate, one, "gate").unwrap();
        let one_up = column_parallel_linear(&x, &up, one, "up").unwrap();
        let one_hidden = one_gate.broadcast_mul(&one_up).unwrap();
        let one_down = sum_row_parallel_partials(&[row_parallel_linear_partial_from_shard(
            &one_hidden,
            &down,
            one,
            "down",
        )
        .unwrap()])
        .unwrap();
        let one_logits = frozen_linear(&one_down, &head).unwrap();

        assert_close(&one_logits, &full_logits, 1e-6);

        let mut partials = Vec::new();
        for plan in all_plans(3) {
            let gate_shard = column_parallel_linear(&x, &gate, plan, "gate").unwrap();
            let up_shard = column_parallel_linear(&x, &up, plan, "up").unwrap();
            let hidden_shard = gate_shard.broadcast_mul(&up_shard).unwrap();
            partials.push(
                row_parallel_linear_partial_from_shard(&hidden_shard, &down, plan, "down").unwrap(),
            );
        }
        let sharded_down = sum_row_parallel_partials(&partials).unwrap();
        let sharded_logits = frozen_linear(&sharded_down, &head).unwrap();
        let sharded_logp = target_logprobs(&sharded_logits, &targets).unwrap();

        assert_close(&sharded_logits, &full_logits, 1e-6);
        assert_close(&sharded_logp, &full_logp, 1e-6);
    }

    #[test]
    fn activation_all_reduce_uses_reduced_value_but_local_gradient() {
        let comms = LocalComm::world(2);
        let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    s.spawn(move || {
                        let plan = plan_from_comm(&comm).unwrap();
                        let rank = comm.rank();
                        let base = match rank {
                            0 => 1.0f32,
                            1 => 2.0f32,
                            other => panic!("unexpected test rank {other}"),
                        };
                        let scale = match rank {
                            0 => 1.0,
                            1 => 2.0,
                            other => panic!("unexpected test rank {other}"),
                        };
                        let v = Var::from_tensor(
                            &Tensor::from_vec(vec![base, base + 1.0], 2, &Device::Cpu).unwrap(),
                        )
                        .unwrap();
                        let partial = (v.as_tensor() * scale).unwrap();
                        let reduced =
                            all_reduce_sum_straight_through(&partial, plan, &comm).unwrap();
                        let grads = reduced.sum_all().unwrap().backward().unwrap();
                        let grad = grads.get(v.as_tensor()).unwrap();
                        (
                            partial.to_vec1::<f32>().unwrap(),
                            reduced.to_vec1::<f32>().unwrap(),
                            grad.to_vec1::<f32>().unwrap(),
                        )
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert_eq!(results[0].0, vec![1.0, 2.0]);
        assert_eq!(results[1].0, vec![4.0, 6.0]);
        assert_eq!(results[0].1, vec![5.0, 8.0]);
        assert_eq!(results[1].1, vec![5.0, 8.0]);
        assert_eq!(results[0].2, vec![1.0, 1.0]);
        assert_eq!(results[1].2, vec![2.0, 2.0]);
    }

    #[test]
    fn activation_all_reduce_coordinates_asymmetric_preflight_before_payload() {
        let tensor_calls = Arc::new(AtomicUsize::new(0));
        let results = std::thread::scope(|scope| {
            let handles = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|inner| {
                    let tensor_calls = Arc::clone(&tensor_calls);
                    scope.spawn(move || {
                        let rank = inner.rank();
                        let comm = CountAndCorruptReadbackComm {
                            inner,
                            tensor_calls,
                            clear_after_reduce: false,
                        };
                        let plan = TensorParallelPlan::new(0, 2).unwrap();
                        let partial = tensor(&[rank as f32 + 1.0], 1);
                        all_reduce_sum_straight_through(&partial, plan, &comm)
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(Result::is_err), "{results:?}");
        assert!(results[1]
            .as_ref()
            .unwrap_err()
            .contains("does not match communicator"));
        assert!(results[0]
            .as_ref()
            .unwrap_err()
            .contains("failed on a peer"));
        assert_eq!(tensor_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cotangent_all_reduce_coordinates_asymmetric_readback_before_return() {
        let tensor_calls = Arc::new(AtomicUsize::new(0));
        let results = std::thread::scope(|scope| {
            let handles = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                .into_iter()
                .map(|inner| {
                    let tensor_calls = Arc::clone(&tensor_calls);
                    scope.spawn(move || {
                        let rank = inner.rank();
                        let plan = plan_from_comm(&inner).unwrap();
                        let comm = CountAndCorruptReadbackComm {
                            inner,
                            tensor_calls,
                            clear_after_reduce: rank == 1,
                        };
                        let value = tensor(&[rank as f32 + 1.0], 1);
                        all_reduce_sum_value(&value, plan, &comm)
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(Result::is_err), "{results:?}");
        assert!(results[1]
            .as_ref()
            .unwrap_err()
            .contains("returned no tensors"));
        assert!(results[0]
            .as_ref()
            .unwrap_err()
            .contains("failed on a peer"));
        assert_eq!(tensor_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn sharded_adapter_projection_keeps_nonzero_gradients() {
        let x = tensor(
            &[
                0.20, -0.30, 0.50, -0.70, 0.90, 1.10, -1.30, 1.50, -1.70, 1.90, 2.10, -2.30,
            ],
            (2, 2, 3),
        );
        let head = tensor(
            &[
                0.11, -0.13, 0.17, -0.19, 0.23, 0.29, -0.31, 0.37, 0.41, -0.43, 0.47, -0.53,
            ],
            (3, 4),
        );
        let mut vars = Vec::new();
        let mut shards = Vec::new();
        for rank in 0..2 {
            let plan = TensorParallelPlan::new(rank, 2).unwrap();
            let out = plan.shard_axis("adapter_out", 4).unwrap().len;
            let a = Var::from_tensor(
                &tensor(
                    &[
                        0.07 + rank as f32 * 0.01,
                        -0.05,
                        0.03,
                        -0.02,
                        0.04 + rank as f32 * 0.02,
                        0.06,
                    ],
                    (2, 3),
                )
                .detach(),
            )
            .unwrap();
            let b = Var::from_tensor(
                &tensor(
                    &[
                        0.09,
                        -0.08 + rank as f32 * 0.01,
                        0.07,
                        0.05 + rank as f32 * 0.02,
                    ][..out * 2],
                    (out, 2),
                )
                .detach(),
            )
            .unwrap();
            let xa = x.broadcast_matmul(&a.as_tensor().t().unwrap()).unwrap();
            let shard = xa.broadcast_matmul(&b.as_tensor().t().unwrap()).unwrap();
            vars.push((a, b));
            shards.push(shard);
        }
        let h = concat_column_shards(&shards).unwrap();
        let logits = frozen_linear(&h, &head).unwrap();
        let grads = logits.sum_all().unwrap().backward().unwrap();

        for (idx, (a, b)) in vars.iter().enumerate() {
            let ga = grads.get(a).unwrap();
            let gb = grads.get(b).unwrap();
            let ga_sum = ga
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let gb_sum = gb
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(ga_sum > 0.0, "rank {idx} A gradient is zero");
            assert!(gb_sum > 0.0, "rank {idx} B gradient is zero");
        }
    }
}
