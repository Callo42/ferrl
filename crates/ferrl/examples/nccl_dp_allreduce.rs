//! Multi-process NCCL all-reduce equivalence check — the GPU gate for the
//! [`NcclComm`](ferrl::NcclComm) data-parallel bridge (P8).
//!
//! Launch one process per GPU (single node, e.g. `srun --ntasks=2
//! --gpus-per-task=1 --gres=gpu:2`). Each process builds its communicator from
//! the Slurm environment ([`NcclComm::from_slurm_env`](ferrl::NcclComm)), then
//! runs a multi-tensor and a scalar sum-all-reduce with **known** per-rank
//! inputs: rank `r` contributes `r + 1`. The cross-rank sum is therefore the
//! analytic triangular number `world·(world+1)/2`, which every rank checks its
//! reduced result against.
//!
//! Every rank printing `DP_ALLREDUCE_PASS` proves both properties the trainer's
//! DP correctness rests on: **correctness** (the reduced value equals the
//! analytic sum) and **cross-rank agreement** (all ranks converge on the same
//! value — a disagreeing rank would miss the analytic target and fail). A wrong
//! or inconsistent collective makes at least one rank print `DP_ALLREDUCE_FAIL`.
//!
//! Requires `--features nccl` (a multi-GPU build); the default build prints a
//! usage note and exits non-zero.

// A standalone gate binary whose interface *is* PASS/FAIL on stdout for the
// launcher to grep — so `println!`/`eprintln!` are the right tool here (the
// library denies them via the workspace `print_*` lints).
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(feature = "nccl")]
fn main() -> anyhow::Result<()> {
    use candle_core::Tensor;
    use ferrl::{Comm, NcclComm};

    /// Element-wise closeness — avoids a bare float `==` while still catching any
    /// real (i.e. non-ulp) disagreement in the reduced values.
    fn close(got: &[f32], want: &[f32]) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-4)
    }

    let comm = NcclComm::from_slurm_env()?;
    let rank = comm.rank();
    let world = comm.world_size();
    let device = comm.device().clone();

    // Rank r contributes (r + 1). Two tensors exercise the multi-tensor loop.
    let c = (rank + 1) as f32;
    let mut tensors = vec![
        Tensor::from_vec(vec![c, 10.0 * c], 2, &device)?,
        Tensor::from_vec(vec![-c], 1, &device)?,
    ];
    comm.all_reduce_sum(&mut tensors)?;
    let got_a = tensors[0].to_vec1::<f32>()?;
    let got_b = tensors[1].to_vec1::<f32>()?;

    // Scalar path (packed into a one-element device tensor internally).
    let got_scalar = comm.all_reduce_scalar_sum((rank + 1) as f64)?;

    // Analytic cross-rank sum of (r + 1) over r in 0..world  ==  world·(world+1)/2.
    let tri = (world * (world + 1) / 2) as f32;
    let want_a = [tri, 10.0 * tri];
    let want_b = [-tri];
    let want_scalar = (world * (world + 1) / 2) as f64;

    println!("[rank {rank}/{world}] a={got_a:?} b={got_b:?} scalar={got_scalar}");

    let scalar_ok = (got_scalar - want_scalar).abs() <= 1e-9 * want_scalar.max(1.0);
    if close(&got_a, &want_a) && close(&got_b, &want_b) && scalar_ok {
        println!("[rank {rank}] DP_ALLREDUCE_PASS");
        Ok(())
    } else {
        anyhow::bail!(
            "[rank {rank}] DP_ALLREDUCE_FAIL: a={got_a:?} want {want_a:?}, b={got_b:?} \
             want {want_b:?}, scalar={got_scalar} want {want_scalar}"
        )
    }
}

#[cfg(not(feature = "nccl"))]
fn main() {
    eprintln!(
        "nccl_dp_allreduce: build with --features nccl (a multi-GPU build) and launch one \
         process per GPU under srun (e.g. srun --ntasks=2 --gpus-per-task=1)."
    );
    std::process::exit(2);
}
