//! Activation checkpointing (rematerialization) for segmented forwards.
//!
//! candle ships no checkpoint primitive, so ferrl orchestrates one at the
//! layer-boundary seam: the binding memory wall of the uncached grad forward
//! is the **retained activation graph** (params fit on one card long before
//! activations do — see the P7 plan), and every intermediate tensor stays
//! alive exactly as long as something references the op graph that produced
//! it. Cutting the graph at layer boundaries and re-running one layer at a
//! time inside backward trades ~one extra forward (~33 % recompute) for an
//! activation footprint of **one segment plus the boundaries** instead of the
//! whole stack.
//!
//! ## How a forward is checkpointed
//!
//! The model runs its layer loop normally but, before each segment, copies the
//! running hidden state into a fresh leaf [`Var`] ([`RematTape::capture`]) and
//! feeds the *var* to the segment. Reassigning the running state then drops
//! the only reference to the previous segment's op graph, freeing its
//! intermediates on the spot. After the last layer one more boundary is
//! captured (the tail input), so the loss tape spans only the tail
//! (final norm + head + the caller's loss ops) with that var as its leaf.
//!
//! The boundary **must** be a [`Var`], not a plain detached tensor: candle's
//! backward only tracks nodes that are variables or reach one
//! (`sorted_nodes`), so a non-var boundary would sever every grad path that
//! runs purely through frozen base weights — backward would silently return
//! adapter-path-only boundary gradients. (The equivalence gates pin this.)
//!
//! ## How backward is stitched
//!
//! [`stitched_backward`] first runs the plain `loss.backward()` — that covers
//! the tail segment and deposits the gradient of the tail boundary var in the
//! store. Walking segments in reverse, it re-runs each segment's forward from
//! its captured boundary (rebuilding exactly the graph the forward dropped)
//! and back-propagates the **vector–Jacobian product** through it via a
//! surrogate scalar `sum(y ⊙ cot)`: the surrogate's gradient w.r.t. the
//! segment is exactly `cot ⊙ ∂y/∂·` — the chain rule continued. Each
//! segment's backward yields (a) gradients for the trainable [`Var`]s used in
//! that segment, folded into the main store, and (b) the gradient of the
//! segment's own boundary var — the cotangent for the next segment down.
//!
//! The cotangent is **detached** before entering the surrogate. candle
//! detaches gradients by default, but under `CANDLE_GRAD_DO_NOT_DETACH` they
//! carry the producing graph, and an attached cotangent would add a spurious
//! `y ⊙ ∂cot/∂θ` term to every segment gradient (a silent double-count).
//!
//! ## Contract
//!
//! - **One tape per forward, one backward per tape.** A tape describes the
//!   weights *as they were* at capture time; consuming models take the tape
//!   out before stitching, so a second backward (or a backward with no
//!   pending checkpointed forward) fails loud rather than stitching stale
//!   segments.
//! - **The loss must come from the forward that produced the tape.** This is
//!   structurally enforced: a foreign loss's backward store cannot contain
//!   the tape's tail boundary var, and [`stitched_backward`] fails loud on
//!   that absence instead of returning tail-only gradients.
//! - **Nothing may move between forward and backward** — no optimizer step,
//!   no adapter toggle: the recompute must rebuild the *same* values the
//!   forward produced. Models pin the adapter half of this by recording the
//!   toggle state on the tape and failing loud on a mismatch; the
//!   no-step half holds by construction in the trainer (backward always
//!   precedes the optimizer step).
//! - **Recompute must be deterministic.** ferrl's grad forwards are pure
//!   tensor ops (no dropout, no RNG), so re-running a segment reproduces its
//!   values exactly.

use candle_core::backprop::GradStore;
use candle_core::{Result as CandleResult, Tensor, Var};

/// The boundary record of one checkpointed forward: one leaf [`Var`] per
/// segment input, plus the tail input, in execution order.
///
/// Built incrementally by the model's checkpointed forward via
/// [`capture`](Self::capture); consumed (by value) by exactly one
/// [`stitched_backward`]. The recorded adapter state lets the model fail loud
/// if the toggle flipped between forward and backward (the recompute would
/// silently rebuild different values).
#[derive(Debug)]
pub struct RematTape {
    /// `inputs[i]` feeds segment `i`; the last entry feeds the tail.
    inputs: Vec<Var>,
    adapter_enabled: bool,
}

impl RematTape {
    /// Start an empty tape, recording the adapter toggle state the forward
    /// runs under.
    #[must_use]
    pub fn new(adapter_enabled: bool) -> Self {
        Self {
            inputs: Vec::new(),
            adapter_enabled,
        }
    }

    /// Copy the running hidden state into a fresh leaf [`Var`], record it as
    /// the next boundary, and return it as the tensor the next segment must
    /// consume.
    ///
    /// The copy is the cut: once the caller reassigns its running state to
    /// the segment's output, nothing references the previous segment's op
    /// graph and its intermediates are freed.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the boundary copy fails.
    pub fn capture(&mut self, h: &Tensor) -> CandleResult<Tensor> {
        // `Var::from_tensor` returns an already-variable tensor as-is (no
        // copy, shared storage); layer outputs are never variables, but a
        // fresh copy is part of this method's contract, so force it.
        let var = Var::from_tensor(&h.detach())?;
        let out = var.as_tensor().clone();
        self.inputs.push(var);
        Ok(out)
    }

    /// The adapter toggle state recorded at capture time.
    #[must_use]
    pub fn adapter_enabled(&self) -> bool {
        self.adapter_enabled
    }

    /// The dims of the first captured boundary (`None` on an empty tape) — a
    /// model re-derives per-sequence context (e.g. the causal mask's length)
    /// from these when it rebuilds segments at backward.
    #[must_use]
    pub fn first_boundary_dims(&self) -> Option<&[usize]> {
        self.inputs.first().map(|v| v.dims())
    }

    /// The number of segments this tape covers (boundaries minus the tail).
    #[must_use]
    pub fn segments(&self) -> usize {
        self.inputs.len().saturating_sub(1)
    }
}

/// Back-propagate `loss` through a checkpointed forward described by `tape`,
/// re-running each segment via `run_segment` (which must rebuild segment `i`'s
/// forward from its boundary input — same weights, same shapes).
///
/// Returns the loss's own [`GradStore`] extended with the gradients of every
/// `trainable` var encountered in the re-run segments, exactly as a plain
/// `loss.backward()` over an uncut graph would have produced them (up to
/// f32 reassociation of the per-segment accumulation order).
///
/// # Errors
///
/// Fails loud if the tape is empty, if the loss's backward does not reach the
/// tape's tail boundary (the loss was not built from the forward that
/// produced this tape), if a re-run segment's output shape does not match the
/// incoming cotangent, or if a segment does not consume its own boundary var.
pub fn stitched_backward<F>(
    loss: &Tensor,
    tape: &RematTape,
    trainable: &[Var],
    run_segment: F,
) -> CandleResult<GradStore>
where
    F: Fn(usize, &Tensor) -> CandleResult<Tensor>,
{
    stitched_backward_with_cotangent(loss, tape, trainable, run_segment, |_, cot| Ok(cot.clone()))
}

/// As [`stitched_backward`], applying `transform_cotangent` to each internal
/// boundary cotangent before it is fed to the preceding segment.
///
/// Tensor-parallel rematerialization uses this seam to sum replicated-boundary
/// cotangents across ranks. The transform is not called for segment zero,
/// because no earlier segment consumes its input cotangent.
pub(crate) fn stitched_backward_with_cotangent<F, C>(
    loss: &Tensor,
    tape: &RematTape,
    trainable: &[Var],
    run_segment: F,
    transform_cotangent: C,
) -> CandleResult<GradStore>
where
    F: Fn(usize, &Tensor) -> CandleResult<Tensor>,
    C: Fn(usize, &Tensor) -> CandleResult<Tensor>,
{
    let Some(segments) = tape.inputs.len().checked_sub(1) else {
        candle_core::bail!("stitched_backward: the tape is empty (no boundaries were captured)")
    };
    let mut store = loss.backward()?;
    let mut cot = store.remove(&tape.inputs[segments]).ok_or_else(|| {
        candle_core::Error::Msg(
            "stitched_backward: the loss does not reach the tape's tail boundary; \
             the loss must be built from the same forward that captured this tape"
                .to_string(),
        )
    })?;
    for i in (0..segments).rev() {
        cot = stitch_segment(
            &mut store,
            &tape.inputs[i],
            &cot,
            trainable,
            &run_segment,
            i,
        )?;
        if i > 0 {
            cot = transform_cotangent(i, &cot)?;
        }
    }
    Ok(store)
}

/// Re-run segment `i` from its boundary var, back-propagate the incoming
/// cotangent through it (the VJP surrogate), fold the segment's trainable-var
/// gradients into `store`, and return the boundary var's own gradient — the
/// cotangent for segment `i - 1`.
fn stitch_segment<F>(
    store: &mut GradStore,
    input: &Var,
    cot: &Tensor,
    trainable: &[Var],
    run_segment: &F,
    i: usize,
) -> CandleResult<Tensor>
where
    F: Fn(usize, &Tensor) -> CandleResult<Tensor>,
{
    let y = run_segment(i, input.as_tensor())?;
    if y.dims() != cot.dims() {
        candle_core::bail!(
            "stitched_backward: segment {i} rebuilt shape {:?} but the cotangent is {:?}; \
             run_segment does not match the forward that captured the tape",
            y.dims(),
            cot.dims()
        )
    }
    // The surrogate's gradient w.r.t. everything inside the segment is exactly
    // cot ⊙ ∂y/∂· — the VJP. Detaching the cotangent keeps the surrogate's
    // backward out of whatever graph produced it (see the module docs).
    let surrogate = y.mul(&cot.detach())?.sum_all()?;
    let mut seg = surrogate.backward()?;
    for var in trainable {
        if let Some(grad) = seg.remove(var) {
            // Accumulate rather than replace: correct even for a var that
            // appears in several segments (none of ferrl's models do today,
            // but replacing would make that future bug silent).
            let grad = match store.remove(var) {
                Some(prev) => (prev + grad)?,
                None => grad,
            };
            store.insert(var, grad);
        }
    }
    seg.remove(input).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "stitched_backward: segment {i} did not consume its boundary var; \
             run_segment must feed the given input through the segment"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Element-wise compare two var gradients out of two stores.
    fn assert_grad_matches(want: &GradStore, got: &GradStore, var: &Var) {
        let want = want.get(var).unwrap().to_vec2::<f32>().unwrap();
        let got = got.get(var).unwrap().to_vec2::<f32>().unwrap();
        for (rw, rg) in want.iter().zip(&got) {
            for (a, b) in rw.iter().zip(rg) {
                assert!((a - b).abs() <= 1e-6, "stitched grad {b} != plain grad {a}");
            }
        }
    }

    /// A tiny two-segment "model": segment i is
    /// `y = relu(x @ w_i + frozen_i)` — one trainable matmul plus a frozen
    /// add, so boundary gradients flow through BOTH a var path and a
    /// frozen-only path (the case a non-var boundary would silently sever).
    struct TwoSegments {
        w: Vec<Var>,
        frozen: Vec<Tensor>,
    }

    impl TwoSegments {
        fn new(dev: &Device) -> Self {
            let mk = |k: usize| {
                let data: Vec<f32> = (0..9).map(|i| ((i + k) as f32) * 0.1 - 0.4).collect();
                Tensor::from_vec(data, (3, 3), dev).unwrap()
            };
            Self {
                w: vec![
                    Var::from_tensor(&mk(1)).unwrap(),
                    Var::from_tensor(&mk(5)).unwrap(),
                ],
                frozen: vec![mk(2), mk(7)],
            }
        }

        fn segment(&self, i: usize, x: &Tensor) -> CandleResult<Tensor> {
            x.matmul(&self.w[i])?.add(&self.frozen[i])?.relu()
        }

        /// Uncut reference forward: the whole graph on one tape.
        fn forward_plain(&self, x: &Tensor) -> CandleResult<Tensor> {
            let h = self.segment(0, x)?;
            self.segment(1, &h)
        }

        /// Checkpointed forward: capture a boundary before each segment plus
        /// the tail; the "tail" here is the loss itself (no tail ops).
        fn forward_remat(&self, x: &Tensor) -> CandleResult<(Tensor, RematTape)> {
            let mut tape = RematTape::new(true);
            let mut h = x.clone();
            for i in 0..2 {
                let b = tape.capture(&h)?;
                h = self.segment(i, &b)?;
            }
            let tail = tape.capture(&h)?;
            Ok((tail, tape))
        }
    }

    fn probe_loss(y: &Tensor) -> Tensor {
        // A fixed non-uniform probe so no gradient cancels by symmetry.
        let w: Vec<f32> = (0..9).map(|i| (i as f32) * 0.3 - 1.1).collect();
        let w = Tensor::from_vec(w, (3, 3), y.device()).unwrap();
        y.mul(&w).unwrap().sum_all().unwrap()
    }

    fn input(dev: &Device) -> Tensor {
        let data: Vec<f32> = (0..9).map(|i| (i as f32) * 0.2 - 0.7).collect();
        Tensor::from_vec(data, (3, 3), dev).unwrap()
    }

    #[test]
    fn stitched_gradients_match_an_uncut_backward_exactly() {
        let dev = Device::Cpu;
        let m = TwoSegments::new(&dev);
        let x = input(&dev);

        let plain = probe_loss(&m.forward_plain(&x).unwrap())
            .backward()
            .unwrap();
        let (tail, tape) = m.forward_remat(&x).unwrap();
        let loss = probe_loss(&tail);
        let stitched = stitched_backward(&loss, &tape, &m.w, |i, b| m.segment(i, b)).unwrap();

        for var in &m.w {
            assert_grad_matches(&plain, &stitched, var);
        }
    }

    #[test]
    fn a_var_reused_across_segments_accumulates_rather_than_replaces() {
        let dev = Device::Cpu;
        let m = TwoSegments::new(&dev);
        let x = input(&dev);
        // Same var drives both segments: grads must SUM across segments.
        let shared = TwoSegments {
            w: vec![m.w[0].clone(), m.w[0].clone()],
            frozen: m.frozen.clone(),
        };

        let plain = probe_loss(&shared.forward_plain(&x).unwrap())
            .backward()
            .unwrap();
        let (tail, tape) = shared.forward_remat(&x).unwrap();
        let loss = probe_loss(&tail);
        let stitched =
            stitched_backward(&loss, &tape, &shared.w[..1], |i, b| shared.segment(i, b)).unwrap();

        assert_grad_matches(&plain, &stitched, &shared.w[0]);
    }

    #[test]
    fn a_foreign_loss_fails_loud_instead_of_stitching() {
        let dev = Device::Cpu;
        let m = TwoSegments::new(&dev);
        let x = input(&dev);
        let (_tail_a, tape_a) = m.forward_remat(&x).unwrap();
        let (tail_b, _tape_b) = m.forward_remat(&x).unwrap();
        // Loss from forward B, tape from forward A: the tail var of A is not
        // on B's loss tape.
        let err = stitched_backward(&probe_loss(&tail_b), &tape_a, &m.w, |i, b| m.segment(i, b))
            .unwrap_err();
        assert!(
            err.to_string().contains("tail boundary"),
            "want the tape/loss-mismatch error, got: {err}"
        );
    }

    #[test]
    fn an_empty_tape_and_a_shape_mismatch_fail_loud() {
        let dev = Device::Cpu;
        let m = TwoSegments::new(&dev);
        let x = input(&dev);

        let empty = RematTape::new(true);
        let (tail, tape) = m.forward_remat(&x).unwrap();
        let loss = probe_loss(&tail);
        assert!(stitched_backward(&loss, &empty, &m.w, |i, b| m.segment(i, b)).is_err());

        // A run_segment that rebuilds the wrong shape must be rejected.
        let err = stitched_backward(&loss, &tape, &m.w, |i, b| m.segment(i, b)?.narrow(0, 0, 2))
            .unwrap_err();
        assert!(
            err.to_string().contains("cotangent"),
            "want the shape-mismatch error, got: {err}"
        );
    }

    #[test]
    fn capture_copies_the_boundary_and_counts_segments() {
        let dev = Device::Cpu;
        let x = input(&dev);
        let mut tape = RematTape::new(false);
        let b = tape.capture(&x).unwrap();
        assert_eq!(
            b.to_vec2::<f32>().unwrap(),
            x.to_vec2::<f32>().unwrap(),
            "the boundary must carry the same values"
        );
        assert!(!tape.adapter_enabled());
        assert_eq!(tape.segments(), 0, "one boundary = a tail with no segments");
        tape.capture(&x).unwrap();
        assert_eq!(tape.segments(), 1);
    }
}
