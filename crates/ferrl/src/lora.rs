//! A manual `LoRA` (low-rank adaptation) linear layer.
//!
//! [`LoraLinear`] wraps a **frozen** base `candle_nn::Linear`-style weight (and
//! optional bias) and adds a trainable low-rank update `B @ A` scaled by
//! `alpha / rank`. Only the two factors `A` and `B` are [`candle_core::Var`]s
//! (i.e. on the autograd tape); the base weight and bias are plain frozen
//! [`Tensor`]s. This is the entire set of trainable parameters GRPO optimizes,
//! which is exactly what the grad-coverage canary asserts over.
//!
//! Forward (for input `x` of shape `[.., in]`):
//!
//! ```text
//! y = x @ Wᵀ (+ b)              // frozen base
//!   + enabled ? (alpha/rank) · (x @ Aᵀ) @ Bᵀ : 0   // trainable update
//! ```
//!
//! with `A` shaped `[rank, in]` and `B` shaped `[out, rank]`. Conventionally `A`
//! is initialized small/random and `B` to zero, so the adapter starts as a no-op
//! and training departs smoothly from the base model.
//!
//! Disabling the adapter ([`LoraLinear::set_enabled`]) drops the update term,
//! yielding the frozen base distribution — the GRPO reference policy.

use candle_core::{Result as CandleResult, Tensor, Var};

/// A linear layer with a frozen base weight and a trainable low-rank adapter.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Frozen base weight, shape `[out, in]`.
    base_weight: Tensor,
    /// Frozen optional bias, shape `[out]`.
    base_bias: Option<Tensor>,
    /// Trainable low-rank factor `A`, shape `[rank, in]`.
    a: Var,
    /// Trainable low-rank factor `B`, shape `[out, rank]`.
    b: Var,
    /// Scaling applied to the low-rank update: `alpha / rank`.
    scale: f64,
    /// Whether the adapter currently contributes to the forward pass.
    enabled: bool,
}

impl LoraLinear {
    /// Build a `LoRA` layer from a frozen base weight/bias and freshly created,
    /// zero-`B` adapter factors of the given `rank`.
    ///
    /// `base_weight` must be `[out, in]`; `base_bias`, if present, must be
    /// `[out]`. `A` is sampled `N(0, 0.02)` and `B` is zero-initialized, so the
    /// adapter starts as an identity (no-op) on top of the base model. The
    /// update is scaled by `alpha / rank`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `rank == 0` or if the adapter factors cannot be
    /// allocated on the base weight's device/dtype.
    pub fn new(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
    ) -> CandleResult<Self> {
        let (out, in_) = base_weight.dims2()?;
        if rank == 0 {
            return Err(candle_core::Error::Msg("lora rank must be > 0".into()));
        }
        let device = base_weight.device();
        let dtype = base_weight.dtype();
        let a = Var::randn(0.0, 0.02, (rank, in_), device)?.to_dtype(dtype)?;
        let a = Var::from_tensor(&a)?;
        let b = Var::zeros((out, rank), dtype, device)?;
        let scale = alpha / rank as f64;
        Ok(Self {
            base_weight,
            base_bias,
            a,
            b,
            scale,
            enabled: true,
        })
    }

    /// The trainable adapter [`Var`]s `(A, B)`, in the order the optimizer and
    /// the grad-coverage canary should iterate them.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        vec![self.a.clone(), self.b.clone()]
    }

    /// Whether the adapter currently contributes to the forward pass.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable the adapter contribution.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// The low-rank update scale `alpha / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Forward pass `y = x Wᵀ (+ b) [+ scale · (x Aᵀ) Bᵀ]`.
    ///
    /// `x` has shape `[.., in]`; the result has shape `[.., out]`. The adapter
    /// term is included only when the layer is [enabled](Self::is_enabled).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `x`'s trailing dimension does not match the
    /// base weight, or if any matmul/broadcast fails.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let base = x.broadcast_matmul(&self.base_weight.t()?)?;
        let base = match &self.base_bias {
            Some(bias) => base.broadcast_add(bias)?,
            None => base,
        };
        if !self.enabled {
            return Ok(base);
        }
        // (x Aᵀ) Bᵀ, scaled.
        let xa = x.broadcast_matmul(&self.a.as_tensor().t()?)?;
        let xab = xa.broadcast_matmul(&self.b.as_tensor().t()?)?;
        let update = (xab * self.scale)?;
        base.broadcast_add(&update)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn base(out: usize, in_: usize) -> Tensor {
        // Deterministic ramp weight so forward outputs are predictable.
        let data: Vec<f32> = (0..out * in_).map(|i| i as f32 * 0.1).collect();
        Tensor::from_vec(data, (out, in_), &Device::Cpu).unwrap()
    }

    #[test]
    fn new_rejects_zero_rank() {
        let w = base(2, 3);
        assert!(LoraLinear::new(w, None, 0, 8.0).is_err());
    }

    #[test]
    fn scale_is_alpha_over_rank() {
        let w = base(4, 3);
        let l = LoraLinear::new(w, None, 2, 8.0).unwrap();
        assert_eq!(l.scale(), 4.0);
    }

    #[test]
    fn exposes_two_trainable_vars() {
        let w = base(4, 3);
        let l = LoraLinear::new(w, None, 2, 8.0).unwrap();
        let vars = l.trainable_vars();
        assert_eq!(vars.len(), 2);
        // A is [rank, in] = [2, 3]; B is [out, rank] = [4, 2].
        assert_eq!(vars[0].as_tensor().dims(), &[2, 3]);
        assert_eq!(vars[1].as_tensor().dims(), &[4, 2]);
    }

    #[test]
    fn zero_init_b_makes_adapter_a_noop_at_start() {
        // With B = 0, the enabled forward must equal the pure base forward.
        let w = base(4, 3);
        let l = LoraLinear::new(w.clone(), None, 2, 8.0).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();

        let y = l.forward(&x).unwrap();
        let base_y = x.broadcast_matmul(&w.t().unwrap()).unwrap();
        let diff = (y - base_y).unwrap().abs().unwrap().sum_all().unwrap();
        let diff: f32 = diff.to_scalar().unwrap();
        assert!(
            diff < 1e-6,
            "adapter with B=0 should be a no-op, diff={diff}"
        );
    }

    #[test]
    fn disabled_equals_base_even_with_nonzero_b() {
        let w = base(4, 3);
        let mut l = LoraLinear::new(w.clone(), None, 2, 8.0).unwrap();
        // Force B nonzero so the adapter would change the output if enabled.
        l.b.set(&Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();

        let enabled_y = l.forward(&x).unwrap();
        l.set_enabled(false);
        assert!(!l.is_enabled());
        let disabled_y = l.forward(&x).unwrap();

        let base_y = x.broadcast_matmul(&w.t().unwrap()).unwrap();
        // disabled == base
        let d_base: f32 = (disabled_y - base_y.clone())
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(d_base < 1e-6);
        // enabled != base (the nonzero-B adapter moved the output)
        let d_en: f32 = (enabled_y - base_y)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(d_en > 1e-3);
    }

    #[test]
    fn bias_is_added() {
        let w = base(2, 2);
        let bias = Tensor::from_vec(vec![10.0f32, 20.0], 2, &Device::Cpu).unwrap();
        let l = LoraLinear::new(w.clone(), Some(bias.clone()), 1, 1.0).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &Device::Cpu).unwrap();
        let y = l.forward(&x).unwrap();
        let no_bias = x.broadcast_matmul(&w.t().unwrap()).unwrap();
        let expect = no_bias.broadcast_add(&bias).unwrap();
        let diff: f32 = (y - expect)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6);
    }

    #[test]
    fn grad_coverage_canary_every_var_gets_nonzero_grad() {
        // The core oracle: after backward on a loss that depends on the adapter,
        // BOTH A and B must appear in the grad store with a non-zero gradient.
        // (candle optimizers silently skip params missing from the GradStore.)
        let w = base(4, 3);
        let l = LoraLinear::new(w, None, 2, 8.0).unwrap();
        // Make B nonzero so dL/dA is also nonzero (with B=0, A's grad is 0).
        l.b.set(&Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap();

        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let y = l.forward(&x).unwrap();
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        for v in l.trainable_vars() {
            let g = grads
                .get(v.as_tensor())
                .expect("var missing from grad store");
            let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
            assert!(mag > 0.0, "trainable var received a zero gradient");
        }
    }
}
