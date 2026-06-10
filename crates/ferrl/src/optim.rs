//! A candle-bit-identical `AdamW` that ferrl owns: [`FerrlAdamW`].
//!
//! ## Why clone candle's optimizer
//!
//! candle-nn's [`AdamW`](candle_nn::optim::AdamW) keeps its per-parameter moment
//! buffers (`first_moment`, `second_moment`) and its global step counter `step_t`
//! private, and exposes no accessor for them. That is fine for a from-scratch run,
//! but it makes a *momentum-faithful* checkpoint/resume impossible: when a run is
//! interrupted and resumed, a fresh candle `AdamW` restarts every moment and the
//! bias-correction counter at zero, so the resumed trajectory diverges from an
//! uninterrupted one (see [`crate::trainer::Trainer::train_from`]).
//!
//! [`FerrlAdamW`] clones candle-nn 0.10.2's `AdamW` update line-for-line and ferrl
//! owns it, so a later phase can read, serialize, and restore that state. This first
//! step is a pure drop-in: it changes *nothing* about the update — it computes the
//! identical step candle does, which a permanent equivalence canary
//! (`ferrl_adamw_step_is_bit_identical_to_candle`) pins bit-for-bit. (The clone is
//! candle's `Optimizer` impl and its moment-bearing struct; candle's unused inherent
//! convenience constructors — `new_lr`/`params`/`set_params` — are omitted.)
//!
//! ## The update (decoupled `AdamW`)
//!
//! Per step the global counter `t` increments first; the bias-correction scales
//! are `1 / (1 - βᵗ)`; each parameter with a gradient `g` updates as
//!
//! ```text
//! m ← β₁·m + (1−β₁)·g
//! v ← β₂·v + (1−β₂)·g²
//! θ ← θ·(1 − lr·λ) − lr · m̂ / (√v̂ + ε)
//! ```
//!
//! where `m̂ = m / (1−β₁ᵗ)`, `v̂ = v / (1−β₂ᵗ)`, `λ` is the (decoupled) weight
//! decay, and `ε` is added *outside* the square root. The raw (uncorrected) `m`
//! and `v` are what persist in state. Parameters absent from the gradient store
//! are skipped entirely — no moment update, no decay — matching candle.
//!
//! ## Staying in sync with candle
//!
//! Because this is a deliberate clone, the canary doubles as a drift alarm: if a
//! future candle-nn changes the `AdamW` algorithm, the bit-for-bit assertion goes
//! red and we re-sync intentionally rather than silently diverging.

use candle_core::backprop::GradStore;
use candle_core::{Result as CandleResult, Tensor, Var};
use candle_nn::optim::ParamsAdamW;
use candle_nn::Optimizer;

/// Per-parameter optimizer state: the parameter plus its two moment buffers.
///
/// The moments share the parameter's shape, dtype, and device and start at zero.
#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

/// `AdamW`'s `Optimizer` impl, cloned line-for-line from candle-nn 0.10.2 so ferrl
/// owns the optimizer state needed for momentum-faithful checkpoint resume.
///
/// It implements [`candle_nn::Optimizer`] with the same [`ParamsAdamW`] config, so
/// it is a drop-in replacement for [`candle_nn::optim::AdamW`]. The step is
/// bit-identical to candle's — pinned by an equivalence canary in this module's
/// tests. The reuse of candle's [`ParamsAdamW`] (rather than a duplicate config)
/// keeps that canary a fair, apples-to-apples comparison.
#[derive(Debug)]
pub struct FerrlAdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

/// A snapshot of [`FerrlAdamW`]'s state for momentum-faithful checkpoint persistence:
/// the global step counter plus the raw (uncorrected) first/second moment buffers, one
/// pair per optimized parameter, in the optimizer's parameter order (the float-filtered
/// `trainable_vars()` order [`new`](Optimizer::new) established).
///
/// Captured by [`FerrlAdamW::state`] — which **deep-copies** the moments, so the
/// snapshot is independent of any later [`step`](FerrlAdamW::step) — and restored by
/// [`FerrlAdamW::load_state`]. Each moment buffer carries the same shape, dtype, and
/// device as its parameter.
#[derive(Debug)]
pub struct OptimizerState {
    /// The global `AdamW` step counter `t` (drives bias correction `1 / (1 - βᵗ)`).
    pub step_t: usize,
    /// Raw first-moment buffers `m`, one per optimized var, in optimizer order.
    pub first_moments: Vec<Tensor>,
    /// Raw second-moment buffers `v`, one per optimized var, in optimizer order.
    pub second_moments: Vec<Tensor>,
}

impl FerrlAdamW {
    /// The global `AdamW` step counter `t` — the number of [`step`](Self::step)s taken,
    /// which drives the bias correction. Exposed for checkpoint persistence.
    #[must_use]
    pub fn step_t(&self) -> usize {
        self.step_t
    }

    /// Snapshot the optimizer state (step counter + **deep-copied** moment buffers) for
    /// checkpoint persistence.
    ///
    /// The moments are copied (via [`Tensor::copy`]) into independent storage, so the
    /// returned [`OptimizerState`] is unaffected by any later [`step`](Self::step) and
    /// is safe to hold and serialize. Moments are returned in the optimizer's parameter
    /// order (the float-filtered order established at construction).
    ///
    /// # Errors
    ///
    /// Returns a candle error if a moment tensor cannot be copied.
    pub fn state(&self) -> CandleResult<OptimizerState> {
        let mut first_moments = Vec::with_capacity(self.vars.len());
        let mut second_moments = Vec::with_capacity(self.vars.len());
        for v in &self.vars {
            first_moments.push(v.first_moment.as_tensor().copy()?);
            second_moments.push(v.second_moment.as_tensor().copy()?);
        }
        Ok(OptimizerState {
            step_t: self.step_t,
            first_moments,
            second_moments,
        })
    }

    /// Restore optimizer state captured by [`state`](Self::state) — the heart of
    /// momentum-faithful resume.
    ///
    /// Validates the moment count and each moment's shape and dtype against this
    /// optimizer's parameters (which must have been constructed from the same model's
    /// `trainable_vars()`), moves each moment onto its parameter's device, writes it in
    /// place, and restores the global step counter. **All-or-nothing:** every moment is
    /// validated and device-prepared before any is written, so a mismatched checkpoint
    /// leaves the optimizer untouched. Fails loud on any count/shape/dtype disagreement
    /// (e.g. a checkpoint from a different architecture).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the moment count differs from this optimizer's
    /// parameter count, or any moment's shape or dtype does not match its parameter.
    pub fn load_state(&mut self, state: &OptimizerState) -> CandleResult<()> {
        let n = self.vars.len();
        if state.first_moments.len() != n || state.second_moments.len() != n {
            return Err(candle_core::Error::Msg(format!(
                "optimizer state has {} first / {} second moments but the optimizer has {n} params",
                state.first_moments.len(),
                state.second_moments.len()
            )));
        }
        // Pass 1 — validate + device-prepare every moment before mutating anything, so a
        // mismatched state leaves the optimizer entirely untouched.
        let mut prepared: Vec<(Tensor, Tensor)> = Vec::with_capacity(n);
        for (i, vw) in self.vars.iter().enumerate() {
            let want = vw.var.as_tensor();
            let m = &state.first_moments[i];
            let v = &state.second_moments[i];
            for (which, t) in [("first", m), ("second", v)] {
                if t.dims() != want.dims() {
                    return Err(candle_core::Error::Msg(format!(
                        "optimizer {which} moment {i}: state shape {:?} != param shape {:?}",
                        t.dims(),
                        want.dims()
                    )));
                }
                if t.dtype() != want.dtype() {
                    return Err(candle_core::Error::Msg(format!(
                        "optimizer {which} moment {i}: state dtype {:?} != param dtype {:?}",
                        t.dtype(),
                        want.dtype()
                    )));
                }
            }
            let dev = want.device();
            prepared.push((
                m.to_device(dev)?.contiguous()?,
                v.to_device(dev)?.contiguous()?,
            ));
        }
        // Pass 2 — every moment validated; apply. `set` cannot fail on shape (matches),
        // self-set (sources are independent copies/loads), or contiguity (moments are).
        for (vw, (m, v)) in self.vars.iter().zip(prepared.iter()) {
            vw.first_moment.set(m)?;
            vw.second_moment.set(v)?;
        }
        self.step_t = state.step_t;
        Ok(())
    }
}

impl Optimizer for FerrlAdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> CandleResult<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &GradStore) -> CandleResult<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        // `&self.vars` is the clippy-preferred idiom for candle's `self.vars.iter()`
        // — identical iteration; the only cosmetic deviation from the upstream clone.
        for var in &self.vars {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::optim::AdamW as CandleAdamW;

    fn vec1(t: &candle_core::Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Run `n` `AdamW` steps whose gradient is the constant `c`: for `L(w) = sum(w · c)`,
    /// `dL/dw = c` regardless of `w`, so the moment evolution depends only on the
    /// optimizer state — letting the resume test isolate momentum from weight-dependent
    /// gradients.
    fn step_const_grad(opt: &mut FerrlAdamW, w: &Var, c: &Tensor, n: usize) {
        for _ in 0..n {
            let g = (w.as_tensor() * c)
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap();
            opt.step(&g).unwrap();
        }
    }

    /// THE canary: ferrl's `AdamW` step is bit-for-bit candle's, across enough
    /// steps that bias correction has evolved, with a non-zero (decoupled) weight
    /// decay engaged and a gradient-less parameter present (to pin the skip path).
    ///
    /// candle hides the moment buffers, so we cannot compare them directly; but the
    /// recurrence is deterministic, so bit-identical *parameters* after every step
    /// imply bit-identical moments. If this ever reddens, candle changed `AdamW` and
    /// the clone must be re-synced — deliberately.
    #[test]
    fn ferrl_adamw_step_is_bit_identical_to_candle() {
        let dev = Device::Cpu;
        // A non-default, fully-specified config: a real weight decay (exercises the
        // decoupled `θ·(1 − lr·λ)` term) and the canonical betas/eps.
        let params = ParamsAdamW {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
        };

        // Two parameters of different shapes + one parameter that never receives a
        // gradient (the skip path: no moment update, no decay).
        let w0 = Tensor::from_vec(vec![0.5f32, -0.3, 0.2, 0.7, -0.1, 0.4], (2, 3), &dev).unwrap();
        let b0 = Tensor::from_vec(vec![0.1f32, -0.2], (2,), &dev).unwrap();
        let dead0 = Tensor::from_vec(vec![1.5f32], (1,), &dev).unwrap();
        let tw = Tensor::from_vec(vec![1.0f32, 0.0, -0.5, 0.3, 0.2, -0.1], (2, 3), &dev).unwrap();
        let tb = Tensor::from_vec(vec![-0.4f32, 0.6], (2,), &dev).unwrap();

        // Two independent, bit-identical Var sets — one driven by FerrlAdamW, one
        // by candle. `w0`/`b0`/`dead0` are plain (non-variable) tensors, so each
        // `Var::from_tensor` deep-copies (via `make_var`) into fresh storage: the
        // two sets do NOT alias, so the per-step bit-equality below is a real
        // comparison, not a vacuous read of shared memory.
        let (wf, bf, df) = (
            Var::from_tensor(&w0).unwrap(),
            Var::from_tensor(&b0).unwrap(),
            Var::from_tensor(&dead0).unwrap(),
        );
        let (wc, bc, dc) = (
            Var::from_tensor(&w0).unwrap(),
            Var::from_tensor(&b0).unwrap(),
            Var::from_tensor(&dead0).unwrap(),
        );

        let mut of =
            FerrlAdamW::new(vec![wf.clone(), bf.clone(), df.clone()], params.clone()).unwrap();
        let mut oc =
            CandleAdamW::new(vec![wc.clone(), bc.clone(), dc.clone()], params.clone()).unwrap();

        // L(w, b) = ‖w − tw‖² + ‖b − tb‖²; the dead var is absent from the graph.
        let loss = |w: &Var, b: &Var| -> CandleResult<Tensor> {
            let lw = (w.as_tensor() - &tw)?.sqr()?.sum_all()?;
            let lb = (b.as_tensor() - &tb)?.sqr()?.sum_all()?;
            lw + lb
        };

        for step in 1..=12 {
            let gf = loss(&wf, &bf).unwrap().backward().unwrap();
            of.step(&gf).unwrap();
            let gc = loss(&wc, &bc).unwrap().backward().unwrap();
            oc.step(&gc).unwrap();

            assert_eq!(
                vec1(wf.as_tensor()),
                vec1(wc.as_tensor()),
                "w diverged from candle at step {step}"
            );
            assert_eq!(
                vec1(bf.as_tensor()),
                vec1(bc.as_tensor()),
                "b diverged from candle at step {step}"
            );
            // The gradient-less parameter is untouched by both optimizers.
            assert_eq!(
                vec1(df.as_tensor()),
                vec1(&dead0),
                "ferrl touched a grad-less param"
            );
            assert_eq!(
                vec1(dc.as_tensor()),
                vec1(&dead0),
                "candle touched a grad-less param"
            );
        }
    }

    /// The learning-rate accessors round-trip (the trainer reads `learning_rate`
    /// for telemetry; `set_learning_rate` completes the [`Optimizer`] contract).
    #[test]
    fn learning_rate_accessors_round_trip() {
        let dev = Device::Cpu;
        let v =
            Var::from_tensor(&Tensor::zeros((2,), candle_core::DType::F32, &dev).unwrap()).unwrap();
        let mut opt = FerrlAdamW::new(vec![v], ParamsAdamW::default()).unwrap();
        assert_eq!(opt.learning_rate(), ParamsAdamW::default().lr);
        opt.set_learning_rate(0.01);
        assert_eq!(opt.learning_rate(), 0.01);
    }

    /// `load_state` restores the moments + step counter so a resumed optimizer steps
    /// bit-identically to one that never paused — while a fresh optimizer that does NOT
    /// load the state diverges (the negative control: momentum restoration is exactly
    /// what makes resume faithful). Also pins that [`FerrlAdamW::state`] **deep-copies**:
    /// the snapshot is taken mid-run and survives the reference's continued steps.
    #[test]
    fn load_state_restores_momentum_for_a_faithful_continuation() {
        let dev = Device::Cpu;
        let params = ParamsAdamW {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
        };
        // L(w) = sum(w * c) ⇒ dL/dw = c (see `step_const_grad`): the same constant
        // gradient for every optimizer each step regardless of w's value, so any
        // divergence is due to optimizer STATE alone, not a weight-dependent gradient.
        let c = Tensor::from_vec(vec![0.3f32, -0.7, 0.5, 0.1], (4,), &dev).unwrap();
        let new_var = |v: &[f32]| {
            Var::from_tensor(&Tensor::from_vec(v.to_vec(), (4,), &dev).unwrap()).unwrap()
        };

        // Reference: an uninterrupted optimizer. Run 5 steps, snapshot, run 5 more.
        let w_ref = new_var(&[1.0, 2.0, 3.0, 4.0]);
        let mut opt_ref = FerrlAdamW::new(vec![w_ref.clone()], params.clone()).unwrap();
        step_const_grad(&mut opt_ref, &w_ref, &c, 5);
        let snap = opt_ref.state().unwrap();
        assert_eq!(snap.step_t, 5);
        let w_at_snap = vec1(w_ref.as_tensor());
        step_const_grad(&mut opt_ref, &w_ref, &c, 5);
        let w_ref_final = vec1(w_ref.as_tensor());

        // Faithful resume: a fresh optimizer on a var re-initialized to the snapshot
        // weights, with the optimizer state LOADED, reproduces the final weights exactly.
        // (`snap` was taken before the reference's last 5 steps and must be intact — the
        // deep-copy guarantee; an aliasing snapshot would have been corrupted.)
        let w_resume = new_var(&w_at_snap);
        let mut opt_resume = FerrlAdamW::new(vec![w_resume.clone()], params.clone()).unwrap();
        opt_resume.load_state(&snap).unwrap();
        assert_eq!(opt_resume.step_t(), 5, "step counter must restore");
        step_const_grad(&mut opt_resume, &w_resume, &c, 5);
        assert_eq!(
            vec1(w_resume.as_tensor()),
            w_ref_final,
            "loaded optimizer state must continue bit-identically"
        );

        // Negative control: same re-init weights but momentum NOT restored (fresh moments
        // + step_t = 0) diverges — so the assertion above is non-vacuous.
        let w_fresh = new_var(&w_at_snap);
        let mut opt_fresh = FerrlAdamW::new(vec![w_fresh.clone()], params).unwrap();
        step_const_grad(&mut opt_fresh, &w_fresh, &c, 5);
        assert_ne!(
            vec1(w_fresh.as_tensor()),
            w_ref_final,
            "a momentum-reset resume must diverge (else the faithful-resume gate is vacuous)"
        );
    }

    /// `load_state` fails loud on a moment-count or shape mismatch (a checkpoint from a
    /// model with different parameters), leaving the optimizer untouched.
    #[test]
    fn load_state_rejects_mismatched_moments() {
        let dev = Device::Cpu;
        let f32z = |dims: &[usize]| Tensor::zeros(dims, candle_core::DType::F32, &dev).unwrap();
        let w = Var::from_tensor(&f32z(&[4])).unwrap();
        let mut opt = FerrlAdamW::new(vec![w], ParamsAdamW::default()).unwrap();

        // Too many moments for a single-param optimizer.
        let wrong_count = OptimizerState {
            step_t: 3,
            first_moments: vec![f32z(&[4]), f32z(&[4])],
            second_moments: vec![f32z(&[4]), f32z(&[4])],
        };
        assert!(
            opt.load_state(&wrong_count).is_err(),
            "moment-count mismatch must fail loud"
        );

        // Right count, wrong shape.
        let wrong_shape = OptimizerState {
            step_t: 3,
            first_moments: vec![f32z(&[3])],
            second_moments: vec![f32z(&[4])],
        };
        assert!(
            opt.load_state(&wrong_shape).is_err(),
            "moment-shape mismatch must fail loud"
        );
    }
}
