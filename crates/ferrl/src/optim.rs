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
use candle_core::{Result as CandleResult, Var};
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
}
