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
//!
//! For the cached rollout path, [`LoraLinear::merged_weight`] folds the live
//! adapter into one frozen, tape-detached weight `W + (alpha/rank) · B @ A`
//! (or just `W` when disabled), so a KV-cached decoder can apply the adapter
//! as a single plain matmul per step. The grad/scoring path never uses it.

use candle_core::{DType, Result as CandleResult, Tensor, Var};

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
    /// Build a `LoRA` layer with adapter factors in the **base weight's** dtype.
    ///
    /// See [`with_adapter_dtype`](Self::with_adapter_dtype) for the details; this
    /// is the common case (toy / all-F32) where the adapter and base share a dtype.
    ///
    /// # Errors
    ///
    /// As [`with_adapter_dtype`](Self::with_adapter_dtype).
    pub fn new(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
    ) -> CandleResult<Self> {
        let dtype = base_weight.dtype();
        Self::with_adapter_dtype(base_weight, base_bias, rank, alpha, dtype)
    }

    /// Build a `LoRA` layer whose **trainable adapter** (`A`/`B`) is held in
    /// `adapter_dtype`, independent of the frozen base weight's dtype.
    ///
    /// `base_weight` must be `[out, in]`; `base_bias`, if present, must be `[out]`.
    /// `A` is sampled `N(0, 0.02)` and `B` is zero-initialized, so the adapter
    /// starts as an identity (no-op) on top of the base model. The update is scaled
    /// by `alpha / rank`.
    ///
    /// # The dtype split (bf16 base / F32 adapter)
    ///
    /// For a real model the frozen base is loaded in BF16 (halving weight **and**
    /// activation memory), but a BF16 *adapter* would lose the GRPO update: a tiny
    /// `lr · grad` step rounds away in BF16's ~3 significant digits — the precision
    /// collapse P3 flagged. Keeping `A`/`B` (and therefore their gradients and the
    /// optimizer's moment estimates) in F32 preserves the update; the [`forward`](Self::forward)
    /// casts them down to the activation dtype only for the matmul, so the big
    /// activations stay BF16. This is standard mixed-precision: an F32 master copy,
    /// a BF16 compute path.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `rank == 0` or if the adapter factors cannot be
    /// allocated on the base weight's device.
    pub fn with_adapter_dtype(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        let (out, in_) = base_weight.dims2()?;
        if rank == 0 {
            return Err(candle_core::Error::Msg("lora rank must be > 0".into()));
        }
        let device = base_weight.device();
        let a = Var::randn(0.0, 0.02, (rank, in_), device)?.to_dtype(adapter_dtype)?;
        let a = Var::from_tensor(&a)?;
        let b = Var::zeros((out, rank), adapter_dtype, device)?;
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

    /// The frozen base bias, if any — unchanged by the adapter (`LoRA` updates
    /// only the weight), exposed so a merged-weight forward can apply the same
    /// bias the unmerged [`forward`](Self::forward) applies.
    #[must_use]
    pub fn base_bias(&self) -> Option<&Tensor> {
        self.base_bias.as_ref()
    }

    /// The single effective weight the **current** forward applies: `W` when the
    /// adapter is disabled, `W + scale · (B @ A)` when enabled — so
    /// `x @ merged_weight()ᵀ (+ bias)` reproduces [`forward`](Self::forward)`(x)`
    /// up to matmul-associativity rounding, at every toggle state.
    ///
    /// The merge uses the **same cast order as the forward**: the (possibly
    /// higher-precision master) factors are cast down to the base dtype *before*
    /// the matmul, so the merged weight sees exactly the adapter the activations
    /// would see — not a higher-precision variant of it.
    ///
    /// The result is a plain frozen [`Tensor`], **detached** from the autograd
    /// tape: it is an op-free non-variable leaf (the factors are detached and the
    /// result is re-detached), so a forward built from it can never carry
    /// gradient into (or out of) the adapter. It is a *snapshot* of the live
    /// factor values — recompute it after any optimizer step **or any
    /// [`set_enabled`](Self::set_enabled) change** (the cached-rollout decoder
    /// rebuilds it per `generate()` call, which covers both and makes staleness
    /// structurally impossible).
    ///
    /// At BF16, merge fidelity is bounded by half-ulp of `W` per element: an
    /// adapter contribution below ~0.2 % of the corresponding base-weight element
    /// is absorbed into `W` by rounding (very early in training the merged
    /// rollout policy can equal the base policy exactly while the scoring
    /// forward sees a slightly adapted one). This is inherent reassociation
    /// rounding — any merge has it — and the grad/scoring path is unaffected.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the dtype cast, matmul, or add fails.
    pub fn merged_weight(&self) -> CandleResult<Tensor> {
        if !self.enabled {
            // detach() so the result is an op-free leaf even if a caller ever
            // constructed the layer from a tape-tracked base tensor.
            return Ok(self.base_weight.detach());
        }
        let dtype = self.base_weight.dtype();
        // The detached/cast factors may still ALIAS the live Var storage (candle's
        // detach shares storage; a same-dtype to_dtype is a clone). The snapshot
        // guarantee holds because matmul/affine/add below all allocate fresh
        // storage — no fast path may ever return a cast/detached factor directly.
        let a = self.a.as_tensor().detach().to_dtype(dtype)?;
        let b = self.b.as_tensor().detach().to_dtype(dtype)?;
        let delta = (b.matmul(&a)? * self.scale)?;
        Ok((&self.base_weight + &delta)?.detach())
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
        // The adapter factors may be held in a higher-precision dtype than the
        // activations (the bf16-base / F32-adapter split — see `with_adapter_dtype`).
        // Cast them down to the activation dtype for the matmul (a no-op when they
        // already match); the F32 master copy is what the optimizer updates, so its
        // precision is preserved regardless of this compute-dtype cast.
        let dtype = self.base_weight.dtype();
        let a = self.a.as_tensor().to_dtype(dtype)?;
        let b = self.b.as_tensor().to_dtype(dtype)?;
        // (x Aᵀ) Bᵀ, scaled.
        let xa = x.broadcast_matmul(&a.t()?)?;
        let xab = xa.broadcast_matmul(&b.t()?)?;
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
    fn adapter_dtype_is_independent_of_the_base() {
        // The dtype split: the trainable adapter is held in its own precision,
        // independent of the frozen base. The real instance is bf16-base / F32-adapter,
        // but candle's CPU backend has no bf16 matmul, so we test the mechanism on CPU
        // with F32-base / F64-adapter (same direction: adapter higher precision than
        // the compute path).
        let w = base(4, 3); // F32 base
        let l = LoraLinear::with_adapter_dtype(w, None, 2, 8.0, DType::F64).unwrap();
        assert_eq!(l.a.as_tensor().dtype(), DType::F64);
        assert_eq!(l.b.as_tensor().dtype(), DType::F64);
        // The forward runs in the activation/base dtype (F32), not the adapter's.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let y = l.forward(&x).unwrap();
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn master_adapter_receives_its_own_dtype_grad() {
        // The anti-collapse property: even though the matmul runs in the lower compute
        // dtype, the gradient (and so the optimizer's update) lands on the adapter in
        // its master dtype — a tiny step that would round away in the compute dtype
        // survives in the master. (CPU stand-in for bf16-base / F32-adapter; see above.)
        //
        // This proves the dtype *routing* (the grad reaches the master in the master's
        // dtype, via the cast's upcast adjoint). It does NOT stress bf16's in-gradient
        // precision: the analog computes the grad in F32 and upcasts to F64, whereas the
        // bf16 instance computes it in bf16 and upcasts to F32. That the bf16 rounding is
        // tolerable is established empirically by the GPU Countdown run, not here.
        let w = base(4, 3); // F32 base
        let l = LoraLinear::with_adapter_dtype(w, None, 2, 8.0, DType::F64).unwrap();
        // Force B nonzero so A also carries a live gradient.
        l.b.set(&Tensor::ones((4, 2), DType::F64, &Device::Cpu).unwrap())
            .unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let loss = l.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for v in l.trainable_vars() {
            let g = grads
                .get(v.as_tensor())
                .expect("var missing from grad store");
            assert_eq!(
                g.dtype(),
                DType::F64,
                "master adapter must receive a grad in its own dtype"
            );
            let mag: f64 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
            assert!(mag > 0.0, "master adapter received a zero gradient");
        }
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

    /// Max-abs elementwise difference between two tensors, as f32.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    /// A `LoRA` layer with both factors forced to non-trivial values, so the
    /// adapter genuinely moves the output (zero-B init would make every merge
    /// test vacuously pass on `delta == 0`).
    fn nonzero_lora(out: usize, in_: usize, rank: usize, alpha: f64) -> LoraLinear {
        let l = LoraLinear::new(base(out, in_), None, rank, alpha).unwrap();
        let a_data: Vec<f32> = (0..rank * in_).map(|i| 0.3 - i as f32 * 0.07).collect();
        l.a.set(&Tensor::from_vec(a_data, (rank, in_), &Device::Cpu).unwrap())
            .unwrap();
        let b_data: Vec<f32> = (0..out * rank).map(|i| 0.1 + i as f32 * 0.05).collect();
        l.b.set(&Tensor::from_vec(b_data, (out, rank), &Device::Cpu).unwrap())
            .unwrap();
        l
    }

    #[test]
    fn merged_weight_equals_base_at_zero_b_init() {
        // B = 0 ⇒ delta = scale·B@A = 0 ⇒ the merged weight IS the base weight.
        let w = base(4, 3);
        let l = LoraLinear::new(w.clone(), None, 2, 8.0).unwrap();
        let m = l.merged_weight().unwrap();
        assert_eq!(m.dims(), w.dims());
        assert_eq!(max_abs_diff(&m, &w), 0.0);
    }

    #[test]
    fn merged_forward_matches_lora_forward_with_nonzero_factors() {
        // THE faithfulness property: x @ merged_weight()ᵀ == forward(x) for a
        // genuinely non-trivial adapter, up to matmul-associativity rounding.
        let l = nonzero_lora(4, 3, 2, 8.0);
        let x = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 1.5, 0.0, -0.5],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let want = l.forward(&x).unwrap();
        let got = x
            .broadcast_matmul(&l.merged_weight().unwrap().t().unwrap())
            .unwrap();
        let md = max_abs_diff(&got, &want);
        assert!(
            md <= 1e-5,
            "merged forward diverged from LoRA forward: {md}"
        );
        // And the adapter is genuinely non-trivial (the property is not vacuous).
        let base_y = x.broadcast_matmul(&base(4, 3).t().unwrap()).unwrap();
        assert!(max_abs_diff(&want, &base_y) > 1e-3, "adapter was a no-op");
    }

    #[test]
    fn merged_weight_respects_the_adapter_toggle() {
        // Disabled ⇒ the merged weight is the pure base (the eval/base-rollout
        // case); enabled ⇒ it differs. `x @ mergedᵀ == forward(x)` must hold at
        // BOTH toggle states — that is what lets a per-generate rebuild serve the
        // adapter-on policy rollout and the adapter-off eval rollout unchanged.
        let mut l = nonzero_lora(4, 3, 2, 8.0);
        let w = base(4, 3);
        let on = l.merged_weight().unwrap();
        assert!(
            max_abs_diff(&on, &w) > 1e-3,
            "enabled merge ignored the adapter"
        );

        l.set_enabled(false);
        let off = l.merged_weight().unwrap();
        assert_eq!(max_abs_diff(&off, &w), 0.0);
        let x = Tensor::from_vec(vec![1.0f32, -2.0, 0.5], (1, 3), &Device::Cpu).unwrap();
        let want = l.forward(&x).unwrap();
        let got = x.broadcast_matmul(&off.t().unwrap()).unwrap();
        assert!(max_abs_diff(&got, &want) <= 1e-6);
    }

    #[test]
    fn merged_weight_with_bias_reproduces_forward() {
        // The merge covers only the weight; the caller applies the (unchanged)
        // frozen bias via `base_bias()`. Together they reproduce forward().
        let bias = Tensor::from_vec(vec![10.0f32, -3.0, 0.25, 7.5], 4, &Device::Cpu).unwrap();
        let l = LoraLinear::new(base(4, 3), Some(bias.clone()), 2, 8.0).unwrap();
        l.b.set(&Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        let x = Tensor::from_vec(vec![0.5f32, 1.0, -1.5], (1, 3), &Device::Cpu).unwrap();
        let want = l.forward(&x).unwrap();
        let got = x
            .broadcast_matmul(&l.merged_weight().unwrap().t().unwrap())
            .unwrap()
            .broadcast_add(l.base_bias().unwrap())
            .unwrap();
        let md = max_abs_diff(&got, &want);
        assert!(md <= 1e-5, "merged + bias diverged from forward: {md}");
    }

    #[test]
    fn merged_weight_casts_factors_down_before_the_matmul() {
        // Pin the cast ORDER under the dtype split, not just the result dtype.
        // F32 base / F64 adapter, with A built for catastrophic cancellation:
        // delta = B@A = 1·(1e8 + 0.25) + 1·(−1e8). Cast-to-F32-then-matmul (the
        // forward's order) rounds 1e8 + 0.25 → 1e8 ⇒ delta == 0 exactly; a
        // matmul-in-F64-then-cast merge would keep delta == 0.25 — far above any
        // rounding tolerance, so the wrong order cannot sneak through.
        let w = base(1, 1);
        let l = LoraLinear::with_adapter_dtype(w.clone(), None, 2, 2.0, DType::F64).unwrap();
        l.a.set(&Tensor::from_vec(vec![1.0e8f64 + 0.25, -1.0e8], (2, 1), &Device::Cpu).unwrap())
            .unwrap();
        l.b.set(&Tensor::from_vec(vec![1.0f64, 1.0], (1, 2), &Device::Cpu).unwrap())
            .unwrap();
        let m = l.merged_weight().unwrap();
        assert_eq!(m.dtype(), DType::F32, "the merge runs in the base dtype");
        assert_eq!(
            max_abs_diff(&m, &w),
            0.0,
            "merge multiplied the F64 masters before casting down — it must cast \
             to the base dtype first, exactly as the forward does"
        );
    }

    #[test]
    fn merged_weight_is_detached_from_the_tape() {
        // Structural guarantee: a loss built from the merged weight reaches
        // NEITHER LoRA factor — the rollout forward can never leak gradient into
        // the adapter (or drag the rollout graph onto the tape).
        let l = nonzero_lora(4, 3, 2, 8.0);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let loss = x
            .broadcast_matmul(&l.merged_weight().unwrap().t().unwrap())
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        for v in l.trainable_vars() {
            assert!(
                grads.get(v.as_tensor()).is_none(),
                "merged_weight leaked a grad path into a LoRA factor"
            );
        }
    }

    #[test]
    fn merged_weight_reflects_live_var_values() {
        // The merge is a snapshot of the LIVE factors: after an (optimizer-style)
        // in-place Var update, a fresh merge differs from the stale one — the
        // per-generate rebuild is what keeps the cached rollout current.
        let l = nonzero_lora(4, 3, 2, 8.0);
        let before = l.merged_weight().unwrap();
        let bumped = ((l.b.as_tensor() + 0.5).unwrap()).detach();
        l.b.set(&bumped).unwrap();
        let after = l.merged_weight().unwrap();
        assert!(
            max_abs_diff(&after, &before) > 1e-3,
            "merged_weight did not track an in-place Var update"
        );
    }

    #[test]
    fn grad_coverage_canary_contract() {
        // candle optimizers silently skip params missing from the GradStore, so
        // the canary asserts grad COVERAGE — but the contract is init-dependent:
        //
        //  (1) standard init (B = 0): the update (x@Aᵀ)@Bᵀ is 0, so dL/dA is
        //      legitimately 0; only B is guaranteed present with a non-zero grad.
        //  (2) after a non-zero-B step: BOTH A and B must be present + non-zero.
        //
        // A runtime canary must therefore not require A != 0 at step 0, or it
        // false-fails on standard zero-B LoRA init.
        let w = base(4, 3);
        let l = LoraLinear::new(w, None, 2, 8.0).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();

        // (1) standard init: B = 0 -> B has a non-zero grad (A's grad is ~0).
        let loss = l.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let gb = grads
            .get(l.b.as_tensor())
            .expect("B missing from grad store");
        let mag_b: f32 = gb.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(mag_b > 0.0, "B received a zero gradient at standard init");

        // (2) after a non-zero-B update, every trainable var has a non-zero grad.
        l.b.set(&Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        let loss = l.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for v in l.trainable_vars() {
            let g = grads
                .get(v.as_tensor())
                .expect("var missing from grad store");
            let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
            assert!(
                mag > 0.0,
                "trainable var received a zero gradient after a non-zero-B step"
            );
        }
    }
}
