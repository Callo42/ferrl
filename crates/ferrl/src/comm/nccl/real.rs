//! The `unsafe` NCCL FFI — the crate's sole `unsafe`, quarantined behind
//! `--features nccl` (decision D2).
//!
//! Everything correctness-bearing about the bridge — config parse, dtype and
//! contract validation, scalar packing, the [`NcclComm`](super::NcclComm)
//! orchestration — lives in the parent module and is CI-tested against the
//! in-memory mock. This file is only the thin adapter that (a) bootstraps a
//! cudarc NCCL [`Comm`] for one rank and (b) runs a single-tensor sum-all-reduce
//! as a candle custom op (candle's own `llama_multiprocess` pattern). It compiles
//! only on the GPU cluster (cudarc's NCCL bindings link `libnccl`) and is
//! runtime-gated by a manual multi-GPU Slurm run, never by CI.
#![allow(unsafe_code)]

use std::ffi::c_char;

use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::nccl::{Comm, Id, ReduceOp};
use candle_core::cuda::CudaStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Tensor};

use super::{read_id_file, write_id_file, NcclConfig, NcclPrimitives, NCCL_UNIQUE_ID_LEN};
use crate::comm::CommError;

/// How long a non-zero rank waits for rank 0 to publish the unique id before
/// giving up — generous, since rank 0 may still be opening its own device when
/// the peers reach the rendezvous.
const ID_RENDEZVOUS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// The real NCCL collective primitive: one rank's cudarc [`Comm`]. The crate's
/// only `unsafe` lives in this module.
#[derive(Debug)]
pub struct RealNccl {
    comm: Comm,
    rank: usize,
    world_size: usize,
}

// SAFETY: a data-parallel launch runs exactly one process per rank, and that
// process drives its single `RealNccl` from one thread (the trainer loop). The
// cudarc `Comm` wraps a raw `ncclComm_t` that is therefore never shared across
// threads; the `Send` bound exists only so the handle can move into the rank's
// owning context (`Box<dyn Comm>`), never to share it between threads.
unsafe impl Send for RealNccl {}

impl RealNccl {
    /// Bootstrap this rank's communicator from `config`: bind the node-local CUDA
    /// device, exchange the NCCL unique id over the rendezvous file, and init the
    /// comm. Returns the primitive plus the bound CUDA [`Device`] the
    /// orchestration stages its scalar all-reduces on.
    pub(crate) fn bootstrap(config: &NcclConfig) -> Result<(Self, Device), CommError> {
        let device = Device::new_cuda(config.local_rank()).map_err(|e| {
            CommError::Config(format!("opening CUDA device {}: {e}", config.local_rank()))
        })?;
        let stream = device
            .as_cuda_device()
            .map_err(|e| {
                CommError::Config(format!("device {} is not CUDA: {e}", config.local_rank()))
            })?
            .cuda_stream();
        let id = exchange_unique_id(config)?;
        let comm = Comm::from_rank(stream, config.rank(), config.world_size(), id)
            .map_err(|e| CommError::Config(format!("ncclCommInitRank failed: {e:?}")))?;
        Ok((
            Self {
                comm,
                rank: config.rank(),
                world_size: config.world_size(),
            },
            device,
        ))
    }
}

/// Rank 0 generates the unique id and publishes it to the rendezvous file; every
/// other rank reads it back. The same 128-byte blob on every rank is what makes
/// `ncclCommInitRank` join them into one communicator.
fn exchange_unique_id(config: &NcclConfig) -> Result<Id, CommError> {
    let path = config.rendezvous_file();
    if config.rank() == 0 {
        let id =
            Id::new().map_err(|e| CommError::Config(format!("ncclGetUniqueId failed: {e:?}")))?;
        write_id_file(path, &id_to_bytes(*id.internal()))?;
        Ok(id)
    } else {
        let bytes = read_id_file(path, ID_RENDEZVOUS_TIMEOUT)?;
        Ok(Id::uninit(bytes_to_id(bytes)))
    }
}

/// `ncclUniqueId` is `c_char` (signed) internally; the rendezvous file carries
/// plain `u8`. These two reinterpret between the representations.
fn id_to_bytes(internal: [c_char; NCCL_UNIQUE_ID_LEN]) -> [u8; NCCL_UNIQUE_ID_LEN] {
    internal.map(|c| c as u8)
}

fn bytes_to_id(bytes: [u8; NCCL_UNIQUE_ID_LEN]) -> [c_char; NCCL_UNIQUE_ID_LEN] {
    bytes.map(|b| b as c_char)
}

impl NcclPrimitives for RealNccl {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        let op = AllReduceSum { comm: &self.comm };
        for tensor in tensors.iter_mut() {
            *tensor = tensor.apply_op1_no_bwd(&op)?;
        }
        Ok(())
    }
}

/// One tensor's NCCL sum-all-reduce, expressed as a candle custom op so the
/// reduced device buffer returns as a normal [`Tensor`]. Borrows the rank's
/// [`Comm`] for the call.
struct AllReduceSum<'a> {
    comm: &'a Comm,
}

impl CustomOp1 for AllReduceSum<'_> {
    fn name(&self) -> &'static str {
        "nccl-all-reduce-sum"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("nccl-all-reduce-sum runs only on CUDA tensors")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let device = storage.device().clone();
        let shape = layout.shape().clone();
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("nccl all-reduce needs a contiguous tensor".to_string())
        })?;
        let reduced = match storage.dtype() {
            DType::F32 => {
                let src = storage.as_cuda_slice::<f32>()?.slice(start..end);
                let mut dst = device.alloc_zeros::<f32>(end - start)?;
                self.comm
                    .all_reduce(&src, &mut dst, &ReduceOp::Sum)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("ncclAllReduce f32 failed: {e:?}"))
                    })?;
                CudaStorage::wrap_cuda_slice(dst, device)
            }
            DType::F64 => {
                let src = storage.as_cuda_slice::<f64>()?.slice(start..end);
                let mut dst = device.alloc_zeros::<f64>(end - start)?;
                self.comm
                    .all_reduce(&src, &mut dst, &ReduceOp::Sum)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("ncclAllReduce f64 failed: {e:?}"))
                    })?;
                CudaStorage::wrap_cuda_slice(dst, device)
            }
            other => {
                candle_core::bail!("ferrl nccl all-reduce supports only f32/f64; got {other:?}")
            }
        };
        Ok((reduced, shape))
    }
}
