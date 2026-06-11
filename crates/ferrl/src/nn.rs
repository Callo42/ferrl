//! Grad-safe neural-net building blocks and the runtime grad-coverage canary.
//!
//! This module holds the small pieces `ferrl` must own because candle's *fused*
//! kernels would otherwise silently break training:
//!
//! - [`RmsNorm`] — root-mean-square normalization built on the **grad-bearing**
//!   [`candle_nn::ops::rms_norm_slow`]. candle's fused
//!   [`candle_nn::ops::rms_norm`] dispatches through `apply_op2_no_bwd` and
//!   defines no backward, so a gradient cannot cross it; placing it in a training
//!   forward strands every upstream parameter (the silent-skip landmine).
//!   [`RmsNorm`] deliberately uses the slow path so gradients reach the upstream
//!   [`crate::lora::LoraLinear`] factors.
//! - [`grad_coverage`] / [`GradCoverage`] — the runtime grad-coverage canary.
//!   candle optimizers *silently skip* any [`candle_core::Var`] missing from the
//!   grad store after `backward`, so a single mis-wired forward can quietly train
//!   nothing. The canary asserts every trainable var is present (coverage) and at
//!   least one carries a nonzero gradient (liveness) — the init-safe contract.

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
use candle_nn::ops::rms_norm_slow;

/// Root-mean-square layer norm with a **frozen** scale, on the grad-bearing path.
///
/// The forward normalizes the input over its last dimension via
/// [`candle_nn::ops::rms_norm_slow`] and applies a frozen per-feature scale
/// `gamma`. The slow op is built from differentiable tensor ops, so it
/// propagates `dL/dx` to whatever produced `x` — in `ferrl`, the
/// [`crate::lora::LoraLinear`] factors placed **upstream** of the norm. candle's
/// fused [`candle_nn::ops::rms_norm`] has no backward and must never appear in a
/// training forward.
///
/// The scale `gamma` is a frozen [`Tensor`], not a [`candle_core::Var`]: `ferrl`
/// trains only the `LoRA` adapter, so the norm contributes a backward path but no
/// trainable parameters of its own.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Frozen per-feature scale, broadcastable against the last dim of the
    /// input. Conventionally shape `[hidden]` or `[1, hidden]`.
    gamma: Tensor,
    /// Numerical-stability epsilon added to the mean-square before the rsqrt.
    eps: f32,
}

impl RmsNorm {
    /// Build an [`RmsNorm`] from a frozen scale `gamma` and stabilizer `eps`.
    ///
    /// `gamma` must broadcast against the trailing (feature) dimension of the
    /// inputs passed to [`forward`](Self::forward) — typically `[hidden]` or
    /// `[1, hidden]`. It is stored as a frozen [`Tensor`]; it is **not** a
    /// trainable [`candle_core::Var`].
    #[must_use]
    pub fn new(gamma: Tensor, eps: f32) -> Self {
        Self { gamma, eps }
    }

    /// A unit-scale [`RmsNorm`] (`gamma = ones([hidden])`) on the given device
    /// and dtype — the standard default for tests and from-scratch init.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the `gamma` allocation fails.
    pub fn ones(hidden: usize, eps: f32, dtype: DType, device: &Device) -> CandleResult<Self> {
        Ok(Self::new(Tensor::ones(hidden, dtype, device)?, eps))
    }

    /// The frozen scale tensor.
    #[must_use]
    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }

    /// Normalize `x` over its last dimension via the grad-bearing
    /// [`candle_nn::ops::rms_norm_slow`], then apply the frozen scale.
    ///
    /// `x` has shape `[.., hidden]`; the result has the same shape. Gradients
    /// flow back through this op to whatever produced `x`. `gamma` must share
    /// `x`'s dtype as well as broadcast against its last dimension: the scale is
    /// applied in `x`'s dtype, so a dtype mismatch surfaces as a candle error
    /// (ferrl does not silently cast, which would mask a precision-config bug).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `gamma` does not broadcast against `x`'s last
    /// dimension or does not share its dtype, or if any underlying tensor op
    /// fails.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        rms_norm_slow(x, &self.gamma, self.eps)
    }
}

/// Root-mean-square norm with **zero-centered** weights: effective scale
/// `(1 + w)`, multiplied in F32 *before* the downcast.
///
/// The `qwen3_5` family stores its decoder / q-k / final norm weights
/// zero-centered (checkpoint holds `w`, the applied scale is `1 + w`), and the
/// reference multiplies the F32-normalized activations by `(1 + w)` **in F32**
/// and only then casts back to the input dtype (transformers
/// `Qwen3_5RMSNorm`; order pinned there citing PR #29402 — Llama downcasts
/// before the scale, this family after). One model, two conventions: the
/// `DeltaNet` gated norm is plain-`w` ([`RmsNormGated`]) — confusing them loads
/// real checkpoints into a silently-wrong forward, so the two are distinct
/// types. Built from differentiable composites only (grad-safe; candle's
/// fused norm kernels have no backward).
#[derive(Debug, Clone)]
pub struct RmsNormZeroCentered {
    /// Frozen zero-centered scale, shape `[hidden]` (checkpoint layout; the
    /// applied scale is `1 + weight`).
    weight: Tensor,
    /// Stabilizer added to the mean-square before the rsqrt.
    eps: f64,
}

impl RmsNormZeroCentered {
    /// Build from the checkpoint's zero-centered `weight` and `eps`.
    #[must_use]
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// The frozen zero-centered weight tensor (as stored, **without** the +1).
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Normalize `x` (shape `[.., hidden]`) over its last dim and scale by
    /// `(1 + weight)`: F32 throughout, downcast to `x`'s dtype only at exit.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `weight` does not broadcast against `x`'s
    /// last dimension or an underlying tensor op fails.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xn = xf.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let w1 = (self.weight.to_dtype(DType::F32)? + 1.0)?;
        xn.broadcast_mul(&w1)?.to_dtype(x.dtype())
    }
}

/// The `DeltaNet` **gated** RMS norm: per-head `rms(x) * w * silu(gate)`,
/// norm-before-gate, with **plain** weights.
///
/// Mirrors transformers `Qwen3_5RMSNormGated` exactly, including the cast
/// choreography: normalize in F32, multiply by the plain `w` in the **input
/// dtype** (the reference downcasts before the weight), then gate with
/// `silu(gate)` computed in F32, and return in the input dtype. Contrast
/// [`RmsNormZeroCentered`]: that convention is `(1 + w)` with the scale
/// applied **before** the downcast — the `qwen3_5` checkpoint uses both, and
/// swapping them is a silent-wrong-forward bug (pinned by the fixture gate).
#[derive(Debug, Clone)]
pub struct RmsNormGated {
    /// Frozen plain scale, shape `[head_v_dim]`.
    weight: Tensor,
    /// Stabilizer added to the mean-square before the rsqrt.
    eps: f64,
}

impl RmsNormGated {
    /// Build from the checkpoint's plain `weight` and `eps`.
    #[must_use]
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// The frozen plain weight tensor.
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// `rms_normalize(x) * weight * silu(gate)` over the last dim. `x` and
    /// `gate` share shape `[.., head_v_dim]`; output is in `x`'s dtype.
    ///
    /// # Errors
    ///
    /// Returns a candle error on a shape/broadcast mismatch or if an
    /// underlying tensor op fails.
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> CandleResult<Tensor> {
        let in_dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xn = xf.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        // Reference order: downcast, THEN the plain weight in the input dtype.
        let scaled = xn
            .to_dtype(in_dtype)?
            .broadcast_mul(&self.weight.to_dtype(in_dtype)?)?;
        // Gate in F32 (silu computed on the F32 gate), output back in input dtype.
        let gated = scaled
            .to_dtype(DType::F32)?
            .mul(&gate.to_dtype(DType::F32)?.silu()?)?;
        gated.to_dtype(in_dtype)
    }
}

/// Outcome of a grad-coverage check over trainable [`Var`]s after a backward pass.
///
/// candle optimizers silently skip any `Var` missing from the [`GradStore`], so
/// *coverage* (every var present) is the landmine detector, while *liveness* (at
/// least one finite nonzero gradient) distinguishes a healthy graph from a fully
/// autograd-cut one. At standard zero-`B` `LoRA` init the `A` gradient is
/// legitimately zero, so this type never requires *all* gradients to be nonzero.
/// A present var whose gradient is **non-finite** (`NaN`/`±∞`, a numerical
/// blowup) is counted separately in `nonfinite`, never as nonzero, so an
/// explosion is diagnosed distinctly rather than mislabeled dead or passed green.
/// See [`grad_coverage`].
#[derive(Debug, Clone)]
pub struct GradCoverage {
    /// Number of trainable vars inspected.
    pub total: usize,
    /// Number of inspected vars present in the grad store.
    pub present: usize,
    /// Number of inspected vars present with a finite, strictly-positive L1
    /// gradient magnitude.
    pub nonzero: usize,
    /// Number of inspected vars present with a **non-finite** (`NaN`/`±∞`)
    /// gradient — a numerical blowup, distinct from a zero/dead gradient.
    pub nonfinite: usize,
    /// Index (into the inspected slice) of the first var missing from the grad
    /// store, if any. `Some` means the coverage landmine fired.
    pub first_missing: Option<usize>,
}

impl GradCoverage {
    /// `true` iff at least one var was inspected and every inspected var is
    /// present in the grad store (an empty set is **not** vacuously covered).
    #[must_use]
    pub fn is_covered(&self) -> bool {
        self.total > 0 && self.present == self.total
    }

    /// `true` iff at least one inspected var carries a finite, nonzero gradient.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.nonzero > 0
    }

    /// `true` iff the full health contract holds: covered, live, and free of
    /// non-finite gradients.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.is_covered() && self.is_live() && self.nonfinite == 0
    }

    /// Collapse to a `Result`, with a descriptive error naming the failure mode
    /// (and the offending var index for a coverage miss).
    ///
    /// # Errors
    ///
    /// Returns a candle error if no vars were inspected (empty set), if coverage
    /// fails (a var is missing — the silent-skip landmine), if any gradient is
    /// non-finite (`NaN`/`±∞`, a numerical blowup), or if no inspected var has a
    /// nonzero gradient (a dead / autograd-cut forward).
    pub fn into_result(self) -> CandleResult<()> {
        let (present, total, nonfinite) = (self.present, self.total, self.nonfinite);
        if total == 0 {
            return Err(candle_core::Error::Msg(
                "grad-coverage canary: no trainable vars were inspected (empty parameter set)"
                    .into(),
            ));
        }
        if let Some(idx) = self.first_missing {
            return Err(candle_core::Error::Msg(format!(
                "grad-coverage canary: trainable var #{idx} is absent from the grad store \
                 ({present}/{total} present) — candle silently skips such params \
                 (fused-norm / detached-graph landmine)"
            )));
        }
        if nonfinite > 0 {
            return Err(candle_core::Error::Msg(format!(
                "grad-coverage canary: {nonfinite}/{total} trainable vars have a non-finite \
                 (NaN/Inf) gradient — a numerical blowup, not a dead or autograd-cut forward"
            )));
        }
        if !self.is_live() {
            return Err(candle_core::Error::Msg(format!(
                "grad-coverage canary: all {total} trainable vars present but every gradient \
                 is zero — the forward is autograd-cut or fully dead"
            )));
        }
        Ok(())
    }
}

/// Compute [`GradCoverage`] for `vars` against the grad store `grads` produced by
/// [`candle_core::Tensor::backward`].
///
/// A var is *present* iff `grads.get(v.as_tensor())` is `Some`. Lookup is by
/// [`Tensor`] id, and a cloned [`Var`] keys the same slot as its original, so
/// passing [`crate::lora::LoraLinear::trainable_vars`] clones is safe — provided
/// they are clones of the *same* `Var` instances that built the loss (a freshly
/// reconstructed `Var` mints a new id and reads as missing). A present var is
/// *nonzero* iff the L1 magnitude of its gradient is finite and strictly
/// positive, and *non-finite* iff that magnitude is `NaN`/`±∞`; the magnitude is
/// read as `f32` regardless of the gradient's dtype, so `bf16`/`f16` grads work.
///
/// # Errors
///
/// Returns a candle error if reducing a gradient tensor to its scalar L1
/// magnitude fails (`abs` / `sum_all` / `to_dtype` / `to_scalar`).
pub fn grad_coverage(vars: &[Var], grads: &GradStore) -> CandleResult<GradCoverage> {
    let total = vars.len();
    let mut present = 0usize;
    let mut nonzero = 0usize;
    let mut nonfinite = 0usize;
    let mut first_missing = None;
    for (idx, v) in vars.iter().enumerate() {
        match classify_var(v, grads)? {
            VarGrad::Missing => {
                first_missing.get_or_insert(idx);
            }
            VarGrad::Zero => present += 1,
            VarGrad::Nonzero => {
                present += 1;
                nonzero += 1;
            }
            VarGrad::NonFinite => {
                present += 1;
                nonfinite += 1;
            }
        }
    }
    Ok(GradCoverage {
        total,
        present,
        nonzero,
        nonfinite,
        first_missing,
    })
}

/// Per-var gradient state, as classified by [`classify_var`].
enum VarGrad {
    /// Absent from the grad store — the candle silent-skip landmine.
    Missing,
    /// Present with an all-zero gradient (legitimate at zero-`B` `LoRA` init).
    Zero,
    /// Present with a finite, strictly-positive gradient magnitude.
    Nonzero,
    /// Present with a non-finite (`NaN`/`±∞`) gradient — a numerical blowup.
    NonFinite,
}

/// Classify a single var's gradient in `grads`. Split out to keep
/// [`grad_coverage`] under the cognitive-complexity bound.
fn classify_var(v: &Var, grads: &GradStore) -> CandleResult<VarGrad> {
    let Some(g) = grads.get(v.as_tensor()) else {
        return Ok(VarGrad::Missing);
    };
    let mag = grad_l1(g)?;
    if !mag.is_finite() {
        Ok(VarGrad::NonFinite)
    } else if mag > 0.0 {
        Ok(VarGrad::Nonzero)
    } else {
        Ok(VarGrad::Zero)
    }
}

/// The L1 magnitude of a gradient tensor, read as `f32` regardless of the
/// gradient's own dtype (`bf16`/`f16` grads are upcast before the scalar read,
/// which a bare `to_scalar::<f32>()` would reject with a dtype error). A
/// non-finite gradient yields `NaN`/`∞` verbatim so the caller can tell a
/// numerical blowup apart from a zero.
fn grad_l1(g: &Tensor) -> CandleResult<f32> {
    g.abs()?.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::LoraLinear;
    use candle_nn::{AdamW, Optimizer};

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.1 - 0.5).collect()
    }

    /// The P1 gate: a `LoRA` adapter trains under `AdamW` when the gradient must
    /// traverse a `rms_norm_slow` backward (norm placed downstream of the `LoRA`),
    /// and the grad-coverage canary is green on the first backward.
    #[test]
    fn lora_trains_through_rms_norm_slow_under_adamw() {
        let dev = Device::Cpu;
        let (out, in_, batch) = (4usize, 3usize, 2usize);
        let w = Tensor::from_vec(ramp(out * in_), (out, in_), &dev).unwrap();
        let lora = LoraLinear::new(w, None, 2, 8.0).unwrap();
        let norm = RmsNorm::ones(out, 1e-6, DType::F32, &dev).unwrap();

        let x =
            Tensor::from_vec(vec![0.5f32, -0.3, 0.2, -0.1, 0.4, 0.6], (batch, in_), &dev).unwrap();
        let target = Tensor::from_vec(
            vec![1.0f32, -1.0, 0.5, 0.2, -0.5, 1.0, -0.2, 0.3],
            (batch, out),
            &dev,
        )
        .unwrap();

        // TOPOLOGY (load-bearing): x -> lora -> rms_norm_slow -> MSE loss. The
        // norm sits downstream so dL/dA, dL/dB must cross its backward.
        let forward_loss = |lora: &LoraLinear, norm: &RmsNorm| -> Tensor {
            let y = lora.forward(&x).unwrap();
            let n = norm.forward(&y).unwrap();
            (n - &target).unwrap().sqr().unwrap().mean_all().unwrap()
        };

        let vars = lora.trainable_vars();
        let mut opt = AdamW::new_lr(vars.clone(), 0.05).unwrap();
        let initial: f32 = forward_loss(&lora, &norm).to_scalar().unwrap();

        // First backward: assert the canary is green THROUGH the norm.
        let loss0 = forward_loss(&lora, &norm);
        let grads0 = loss0.backward().unwrap();
        let cov = grad_coverage(&vars, &grads0).unwrap();
        assert!(cov.is_covered(), "A and B must both be present: {cov:?}");
        assert!(cov.is_live(), "at least one grad must be nonzero: {cov:?}");
        cov.into_result().unwrap();
        opt.step(&grads0).unwrap();

        for _ in 0..120 {
            let loss = forward_loss(&lora, &norm);
            opt.backward_step(&loss).unwrap();
        }
        let final_loss: f32 = forward_loss(&lora, &norm).to_scalar().unwrap();
        assert!(
            final_loss < initial,
            "loss did not decrease: {initial} -> {final_loss}"
        );
    }

    /// Negative control: same topology with candle's FUSED `rms_norm` severs the
    /// `LoRA` gradients (no backward), so the canary's coverage check fails. This
    /// pins the exact value `RmsNorm` adds — swap the op and the gate breaks.
    #[test]
    fn fused_rms_norm_severs_lora_grads_negative_control() {
        use candle_nn::ops::rms_norm;

        let dev = Device::Cpu;
        let (out, in_, batch) = (4usize, 3usize, 2usize);
        let w = Tensor::from_vec(ramp(out * in_), (out, in_), &dev).unwrap();
        let lora = LoraLinear::new(w, None, 2, 8.0).unwrap();
        let gamma = Tensor::ones(out, DType::F32, &dev).unwrap(); // fused requires 1-D
        let x =
            Tensor::from_vec(vec![0.5f32, -0.3, 0.2, -0.1, 0.4, 0.6], (batch, in_), &dev).unwrap();
        let target = Tensor::from_vec(
            vec![1.0f32, -1.0, 0.5, 0.2, -0.5, 1.0, -0.2, 0.3],
            (batch, out),
            &dev,
        )
        .unwrap();

        // SAME topology, ONLY the norm op differs: x -> lora -> FUSED rms_norm.
        let y = lora.forward(&x).unwrap();
        let n = rms_norm(&y, &gamma, 1e-6).unwrap(); // autograd cut here
        let loss = (n - &target).unwrap().sqr().unwrap().mean_all().unwrap();
        let grads = loss.backward().unwrap();

        let vars = lora.trainable_vars();
        let cov = grad_coverage(&vars, &grads).unwrap();
        assert!(
            !cov.is_ok(),
            "fused rms_norm must sever LoRA grads, but canary passed: {cov:?}"
        );
        assert!(
            grads.get(vars[1].as_tensor()).is_none(),
            "B must be absent from the grad store under the fused norm"
        );
    }

    /// Init-safety: at standard zero-`B` `LoRA` init, `A`'s gradient is
    /// legitimately zero, so the canary must still pass on coverage + liveness.
    #[test]
    fn canary_passes_through_zero_b_lora_init() {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(ramp(12), (4, 3), &dev).unwrap();
        let lora = LoraLinear::new(w, None, 2, 8.0).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &dev).unwrap();
        let loss = lora.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let vars = lora.trainable_vars();
        let cov = grad_coverage(&vars, &grads).unwrap();
        assert!(cov.is_covered(), "A and B present at zero-B init: {cov:?}");
        assert!(
            cov.is_live(),
            "B has a nonzero grad at zero-B init: {cov:?}"
        );
        cov.into_result().unwrap();
    }

    /// A var that never reached the loss is reported as the first missing index,
    /// and `into_result` names it.
    #[test]
    fn canary_reports_first_missing_var() {
        let dev = Device::Cpu;
        let used = Var::ones((2, 2), DType::F32, &dev).unwrap();
        let unused = Var::ones((2, 2), DType::F32, &dev).unwrap();
        let loss = used.as_tensor().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let cov = grad_coverage(&[used.clone(), unused.clone()], &grads).unwrap();
        assert_eq!(cov.first_missing, Some(1));
        assert_eq!(cov.present, 1);
        assert!(!cov.is_ok());
        let err = cov.into_result().unwrap_err().to_string();
        assert!(err.contains("#1"), "error must name the missing var: {err}");
    }

    /// A dead forward (every var present, every gradient zero) is reported
    /// distinctly from a coverage miss. Such an all-present-but-all-zero state is
    /// in fact producible by a real backward — a zero-grad var reached through a
    /// matmul is kept present-with-zero (see
    /// `canary_passes_through_zero_b_lora_init`, where `A` is present with a zero
    /// grad); candle only *drops* a zero-grad var reached solely via an `affine`
    /// with `mul == 0`. It is pinned here at the struct level because that is the
    /// cheapest deterministic way to fix the inputs, and the struct is exactly
    /// what the liveness guard inspects.
    #[test]
    fn coverage_all_present_but_zero_is_reported_dead() {
        let cov = GradCoverage {
            total: 2,
            present: 2,
            nonzero: 0,
            nonfinite: 0,
            first_missing: None,
        };
        assert!(cov.is_covered(), "all vars present: {cov:?}");
        assert!(!cov.is_live(), "no var has a nonzero grad: {cov:?}");
        assert!(!cov.is_ok());
        let err = cov.into_result().unwrap_err().to_string();
        assert!(
            err.contains("zero"),
            "error must flag the dead forward: {err}"
        );
    }

    /// Characterize [`RmsNorm::forward`]: it scales each row to unit RMS.
    #[test]
    fn rms_norm_normalizes_rows_to_unit_rms() {
        use approx::assert_relative_eq;

        let dev = Device::Cpu;
        let gamma = Tensor::ones(3, DType::F32, &dev).unwrap();
        let norm = RmsNorm::new(gamma, 1e-6);
        assert_eq!(norm.gamma().dims(), &[3]);

        let x = Tensor::from_vec(vec![1.0f32, 2.0, 2.0], (1, 3), &dev).unwrap();
        let y = norm.forward(&x).unwrap();
        let yv: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        let ms = yv.iter().map(|v| v * v).sum::<f32>() / 3.0;
        assert_relative_eq!(ms.sqrt(), 1.0, epsilon = 1e-4);
    }

    /// `grad_l1` reads a non-f32 gradient (the bf16/f16 the canary must tolerate)
    /// without the dtype error a bare `to_scalar::<f32>()` would raise.
    #[test]
    fn grad_l1_reads_bf16_without_dtype_error() {
        let dev = Device::Cpu;
        let g = Tensor::from_vec(vec![1.0f32, -2.0, 3.0], 3, &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let mag = grad_l1(&g).unwrap();
        assert!(
            mag > 0.0,
            "bf16 L1 magnitude must read back positive: {mag}"
        );
    }

    /// A non-finite gradient (here `+∞` from the backward of `sqrt(0)`) is
    /// classified as non-finite and reported as a blowup, not as a dead forward.
    #[test]
    fn canary_flags_nonfinite_gradient() {
        let dev = Device::Cpu;
        let v0 = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 1.0], (2, 2), &dev).unwrap();
        let v = Var::from_tensor(&v0).unwrap();
        // d/dx sqrt(x) = 0.5 / sqrt(x); at x = 0 the gradient is +inf.
        let loss = v.as_tensor().sqrt().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let cov = grad_coverage(std::slice::from_ref(&v), &grads).unwrap();
        assert_eq!(cov.present, 1);
        assert_eq!(cov.nonfinite, 1);
        assert!(!cov.is_ok(), "non-finite grad must not be ok: {cov:?}");
        let err = cov.into_result().unwrap_err().to_string();
        assert!(
            err.contains("non-finite"),
            "error must flag the blowup: {err}"
        );
    }

    /// An empty trainable set is reported distinctly, not as a liveness failure.
    #[test]
    fn coverage_empty_var_set_is_reported_distinctly() {
        let dev = Device::Cpu;
        let probe = Var::ones((2, 2), DType::F32, &dev).unwrap();
        let grads = probe
            .as_tensor()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();

        let cov = grad_coverage(&[], &grads).unwrap();
        assert_eq!(cov.total, 0);
        assert!(!cov.is_covered());
        assert!(!cov.is_ok());
        let err = cov.into_result().unwrap_err().to_string();
        assert!(
            err.contains("empty") || err.contains("no trainable"),
            "empty-set message: {err}"
        );
    }

    /// Oracle dumps from transformers v5.11.0 `qwen3_5` (see `src/gdn.rs` tests;
    /// same fixture file, the norm cases).
    const GDN_GOLDEN: &str = include_str!("../tests/fixtures/gdn_golden.json");

    fn gdn_golden() -> serde_json::Value {
        serde_json::from_str(GDN_GOLDEN).expect("gdn golden fixture parses")
    }

    fn fixture_tensor(c: &serde_json::Value, key: &str, shape: &[usize]) -> Tensor {
        let v: Vec<f32> = c[key]
            .as_array()
            .expect("fixture array")
            .iter()
            .map(|x| x.as_f64().expect("fixture float") as f32)
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let bv: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(av.len(), bv.len());
        av.iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The zero-centered norm matches transformers `Qwen3_5RMSNorm` exactly,
    /// and the convention is load-bearing: applying the same weights through
    /// the plain-`w` path (the OTHER convention in this model) is visibly
    /// wrong. One model, two norm conventions — the `qwen3_5` trap.
    #[test]
    fn rms_norm_zero_centered_matches_oracle_and_convention_matters() {
        let g = gdn_golden();
        let c = &g["cases"]["rmsnorm_zero_centered"];
        let x = fixture_tensor(c, "x", &[2, 3, 8]);
        let w = fixture_tensor(c, "weight", &[8]);
        let want = fixture_tensor(c, "out", &[2, 3, 8]);

        let norm = RmsNormZeroCentered::new(w.clone(), 1e-6);
        let got = norm.forward(&x).unwrap();
        let d = max_abs_diff(&got, &want);
        assert!(d <= 1e-6, "zero-centered norm diff {d}");
        assert_eq!(norm.weight().dims(), &[8]);

        // Planted convention swap: plain-w on zero-centered weights (w ~ 0.1,
        // so the scale is ~10x off) must be unmistakably wrong.
        let plain = RmsNorm::new(w, 1e-6);
        let wrong = plain.forward(&x).unwrap();
        let d_wrong = max_abs_diff(&wrong, &want);
        assert!(d_wrong > 1e-2, "convention swap must be visible: {d_wrong}");
    }

    /// The gated norm matches transformers `Qwen3_5RMSNormGated` exactly
    /// (plain `w`, norm-before-gate, silu on the F32 gate), and the reverse
    /// convention swap (treating its plain weights as zero-centered) is
    /// caught.
    #[test]
    fn rms_norm_gated_matches_oracle_and_convention_matters() {
        let g = gdn_golden();
        let c = &g["cases"]["rmsnorm_gated"];
        let x = fixture_tensor(c, "x", &[4, 8]);
        let gate = fixture_tensor(c, "gate", &[4, 8]);
        let w = fixture_tensor(c, "weight", &[8]);
        let want = fixture_tensor(c, "out", &[4, 8]);

        let norm = RmsNormGated::new(w.clone(), 1e-6);
        let got = norm.forward(&x, &gate).unwrap();
        let d = max_abs_diff(&got, &want);
        assert!(d <= 1e-6, "gated norm diff {d}");
        assert_eq!(norm.weight().dims(), &[8]);

        // Planted convention swap: (1 + w) on plain weights (w ~ 1.0 here, so
        // the scale is ~2x off).
        let zc = RmsNormZeroCentered::new(w, 1e-6);
        let wrong = zc.forward(&x).unwrap().mul(&gate.silu().unwrap()).unwrap();
        let d_wrong = max_abs_diff(&wrong, &want);
        assert!(d_wrong > 1e-2, "convention swap must be visible: {d_wrong}");
    }

    /// Both new norms are tape-bearing: gradients cross them to an upstream
    /// `LoRA` adapter (the same guarantee `RmsNorm` pins for the fused-kernel
    /// landmine).
    #[test]
    fn new_norms_carry_gradients_upstream() {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(ramp(12), (4, 3), &dev).unwrap();
        let x = Tensor::from_vec(vec![0.5f32, -0.3, 0.2], (1, 3), &dev).unwrap();
        let weight = Tensor::from_vec(vec![0.1f32, -0.2, 0.3, 0.0], 4, &dev).unwrap();
        let gate = Tensor::from_vec(vec![0.4f32, -0.1, 0.7, 0.2], (1, 4), &dev).unwrap();

        for gated in [false, true] {
            let lora = LoraLinear::new(w.clone(), None, 2, 8.0).unwrap();
            let y = lora.forward(&x).unwrap();
            let n = if gated {
                RmsNormGated::new(weight.clone(), 1e-6)
                    .forward(&y, &gate)
                    .unwrap()
            } else {
                RmsNormZeroCentered::new(weight.clone(), 1e-6)
                    .forward(&y)
                    .unwrap()
            };
            let loss = n.sqr().unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let cov = grad_coverage(&lora.trainable_vars(), &grads).unwrap();
            assert!(cov.is_covered(), "gated={gated}: {cov:?}");
            assert!(cov.is_live(), "gated={gated}: {cov:?}");
        }
    }
}
