//! A ferrl-owned rollout sampler with serializable RNG state: [`GrpoSampler`].
//!
//! ## Why ferrl owns the sampler
//!
//! candle's `LogitsProcessor` hides its `StdRng` behind no public accessor, so a
//! checkpoint cannot capture the rollout RNG: a resumed run re-seeds sampling and
//! its trajectory diverges from an uninterrupted one. [`GrpoSampler`] reproduces
//! candle's temperature multinomial sampling on a ferrl-owned
//! [`Xoshiro256PlusPlus`] whose full state is
//! `serde`-serializable, so a later phase (P6-B checkpoint v2) can snapshot and
//! restore it for bit-exact resume.
//!
//! ## What it reproduces (and what it deliberately changes)
//!
//! The sampling *math* is candle's `Sampling::All { temperature }`, verbatim: cast
//! the logits to F32, divide by the temperature, `softmax_last_dim`, then draw a
//! category from the resulting probabilities with rand's `WeightedIndex` — the same
//! draw candle's `LogitsProcessor::sample` performs. The one deliberate change is
//! the RNG: candle uses `StdRng` (a `ChaCha` CSPRNG with no accessor); ferrl uses a
//! serializable `Xoshiro256PlusPlus`. ferrl never needed candle-stream parity —
//! float non-associativity already makes sampled trajectories platform-dependent
//! (see the P2 build note), so the swap costs nothing, and it is exactly what makes
//! a capturable, restorable RNG possible.
//!
//! The design pass named `ChaCha12Rng`. rand 0.10's own `StdRng` *is* `ChaCha12` (now
//! sourced from the `chacha20` crate), but — like candle's `StdRng` — derives no
//! `serde`. A *serializable* `ChaCha` would cost a dependency or coupling for nothing:
//! `rand_chacha 0.9` pins an incompatible `rand_core` (so it can't drive rand 0.10's
//! `WeightedIndex`), and `chacha20`'s manual state (de)serialization is extra
//! plumbing. `Xoshiro256PlusPlus` — rand-native, serde-serializable behind rand's
//! `serde` feature, no new dependency — is the realized choice. The RNG algorithm is
//! a free parameter here (only serializability + statistical quality matter for
//! rollout sampling).

use candle_core::{DType, Result as CandleResult, Tensor};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Temperature multinomial sampling on a ferrl-owned, serializable RNG.
///
/// A drop-in for candle's `LogitsProcessor` at the one call site that matters:
/// [`sample`](GrpoSampler::sample) takes the same `[vocab]` logits tensor and
/// returns the same `u32` token id. Unlike `LogitsProcessor`, its
/// [`Xoshiro256PlusPlus`] state round-trips through
/// `serde`, so it can be persisted in a checkpoint and restored on resume.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrpoSampler {
    rng: Xoshiro256PlusPlus,
    temperature: f64,
}

impl GrpoSampler {
    /// Seed a sampler at a fixed `temperature`.
    ///
    /// The temperature is baked in (matching how candle's `LogitsProcessor` fixes
    /// it at construction): a [`Policy`](crate::Policy) built from this sampler
    /// fails loud on a `GenConfig` temperature that disagrees, rather than silently
    /// sampling at the wrong one.
    #[must_use]
    pub fn new(seed: u64, temperature: f64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            temperature,
        }
    }

    /// The sampling temperature this sampler was built with.
    #[must_use]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Serialize the full sampler state (RNG state + temperature) to an opaque byte
    /// blob, for momentum-faithful checkpoint persistence. Round-trips with
    /// [`from_state_bytes`](Self::from_state_bytes) — a sampler restored from the blob
    /// reproduces the exact remaining token stream (the property
    /// `serde_snapshot_restores_the_exact_stream` pins).
    ///
    /// # Errors
    ///
    /// Returns a candle error if serialization fails (it does not for this type, whose
    /// fields are all plain data — the `Result` keeps the seam uniform with the rest of
    /// the checkpoint path).
    pub fn to_state_bytes(&self) -> CandleResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(candle_core::Error::wrap)
    }

    /// Reconstruct a sampler from a blob produced by
    /// [`to_state_bytes`](Self::to_state_bytes).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `bytes` is not a valid serialized [`GrpoSampler`]
    /// (fail-loud: a malformed or mismatched checkpoint RNG blob aborts the restore
    /// rather than silently re-seeding).
    pub fn from_state_bytes(bytes: &[u8]) -> CandleResult<Self> {
        serde_json::from_slice(bytes).map_err(candle_core::Error::wrap)
    }

    /// Sample one token id from a 1-D `[vocab]` `logits` tensor.
    ///
    /// Reproduces candle's `Sampling::All { temperature }`: cast to F32, divide by
    /// the temperature, `softmax_last_dim`, then draw a category from the resulting
    /// probabilities with rand's `WeightedIndex` — the identical math candle's
    /// `LogitsProcessor` runs, on ferrl's own (serializable) RNG. Delegates to
    /// [`sample_with`](Self::sample_with) at the baked temperature with no nucleus
    /// filtering, so the token stream (and RNG consumption) is bit-identical to the
    /// pre-R2 sampler.
    ///
    /// # Errors
    ///
    /// As [`sample_with`](Self::sample_with).
    pub fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        Ok(self.sample_with(logits, self.temperature, None)?.0)
    }

    /// Sample one token id from a 1-D `[vocab]` `logits` tensor at an explicit
    /// `temperature` and optional nucleus (`top_p`) filter, returning the token
    /// **and its log-probability under the distribution it was actually drawn
    /// from** (temperature-scaled softmax, renormalized over the nucleus when
    /// `top_p` is active) — the *behavior-policy* log-prob that
    /// [`Rollout::rollout_logprobs`](crate::policy::Rollout::rollout_logprobs)
    /// records for off-policy diagnostics and TIS.
    ///
    /// `top_p = Some(p)` keeps the smallest set of highest-probability tokens
    /// whose cumulative probability reaches `p` (the token that crosses the
    /// threshold is **included** — the HF/vLLM convention) and zeroes the rest;
    /// the draw and the returned log-prob both use the renormalized nucleus.
    /// `None` (and `Some(1.0)`, which keeps every token) leave the distribution
    /// untouched. The training rollout path never passes `top_p` — nucleus
    /// filtering enters only through the eval-only
    /// [`EvalSampling`](crate::policy::EvalSampling) override.
    ///
    /// Exactly one `WeightedIndex` draw advances the RNG regardless of the
    /// parameters, so swapping `sample`/`sample_with` never desyncs a captured
    /// RNG stream.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a tensor op fails, if `temperature` is not
    /// finite and `> 0`, if `top_p` is not in `(0, 1]`, or if the (filtered)
    /// probabilities are not a valid categorical distribution (e.g. all-zero or
    /// non-finite weights, which `WeightedIndex` rejects) — the same failure
    /// modes as candle's sampler.
    pub fn sample_with(
        &mut self,
        logits: &Tensor,
        temperature: f64,
        top_p: Option<f64>,
    ) -> CandleResult<(u32, f32)> {
        // Fail loud on a malformed temperature, mirroring the top_p validation: a
        // negative one would silently sample the INVERTED distribution, and 0/NaN
        // only surface as an opaque WeightedIndex error downstream.
        if !temperature.is_finite() || temperature <= 0.0 {
            candle_core::bail!("temperature must be finite and > 0, got {temperature}");
        }
        let logits = logits.to_dtype(DType::F32)?;
        let logits = (&logits / temperature)?;
        let mut prs: Vec<f32> = candle_nn::ops::softmax_last_dim(&logits)?.to_vec1()?;
        if let Some(p) = top_p {
            nucleus_filter(&mut prs, p)?;
        }
        let distr = WeightedIndex::new(&prs).map_err(candle_core::Error::wrap)?;
        let token = distr.sample(&mut self.rng) as u32;
        // The behavior distribution is the kept weights renormalized — which is
        // also exactly what WeightedIndex draws from. Dividing by the kept mass
        // (~1.0 with no filter, the nucleus mass with one) makes the log-prob
        // the true probability of the draw, not the raw softmax weight.
        let total: f32 = prs.iter().sum();
        let logprob = (prs[token as usize] / total).ln();
        Ok((token, logprob))
    }
}

/// Zero out every probability outside the nucleus: sort indices by probability
/// (descending) and keep the smallest prefix whose cumulative probability reaches
/// `top_p`, **including** the token that crosses the threshold (the HF/vLLM
/// convention — the nucleus is never empty). The caller renormalizes implicitly
/// (`WeightedIndex` draws from relative weights; the returned log-prob divides by
/// the kept mass). Cumulation is in `f64` so a long low-probability tail cannot
/// lose the crossing point to `f32` rounding.
fn nucleus_filter(prs: &mut [f32], top_p: f64) -> CandleResult<()> {
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        candle_core::bail!("top_p must be in (0, 1], got {top_p}");
    }
    // `1.0` means keep everything — exactly, not "until the cumulative crosses
    // 1": an f64 cum of f32 probs can exceed 1 before the last token (rounding),
    // which would zero a genuinely sampleable tail and falsify the documented
    // `Some(1.0) == None` equivalence.
    if top_p >= 1.0 {
        return Ok(());
    }
    let mut idx: Vec<usize> = (0..prs.len()).collect();
    // `total_cmp`, not `partial_cmp`: a NaN probability (e.g. all `-inf` logits)
    // must not hand `sort_by` a broken total order (which may PANIC since Rust
    // 1.81); under the total order NaN just sorts to one end, the cumulative
    // goes NaN, the keep-all fallback engages, and `WeightedIndex` then rejects
    // the weights with the same error the unfiltered path reports.
    idx.sort_by(|&a, &b| prs[b].total_cmp(&prs[a]));
    let mut cum = 0.0_f64;
    // Fallback: keep everything (defensive — reachable only via NaN, see above).
    let mut keep = idx.len();
    for (rank, &i) in idx.iter().enumerate() {
        cum += f64::from(prs[i]);
        if cum >= top_p {
            keep = rank + 1;
            break;
        }
    }
    for &i in &idx[keep..] {
        prs[i] = 0.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// A logits vector with a sharp peak so the most-probable token dominates —
    /// lets us assert sampling lands on the peak without flakiness, and that two
    /// seeds/streams agree or differ as expected.
    fn peaked_logits(vocab: usize, peak: usize) -> Tensor {
        let mut v = vec![0f32; vocab];
        v[peak] = 20.0; // softmax ≈ 1 at `peak`
        Tensor::from_vec(v, vocab, &Device::Cpu).unwrap()
    }

    /// Same seed ⇒ identical token stream (determinism), and the temperature-softmax
    /// math routes an overwhelming peak to that token.
    #[test]
    fn same_seed_is_deterministic_and_follows_the_peak() {
        let logits = peaked_logits(32, 7);
        let mut a = GrpoSampler::new(42, 1.0);
        let mut b = GrpoSampler::new(42, 1.0);
        for _ in 0..16 {
            let ta = a.sample(&logits).unwrap();
            let tb = b.sample(&logits).unwrap();
            assert_eq!(ta, tb, "same seed must give the same stream");
            assert_eq!(ta, 7, "an overwhelming peak must be sampled");
        }
    }

    /// THE capture/restore property P6-B's resume needs: serialize the sampler
    /// mid-stream, keep drawing, then restore the snapshot and redraw — the restored
    /// stream is bit-identical to the original continuation. This is what lets a
    /// checkpoint resume the rollout RNG exactly (PR3 persists this serde blob).
    #[test]
    fn serde_snapshot_restores_the_exact_stream() {
        // A flat distribution so draws actually exercise the RNG (not a forced peak).
        let logits = Tensor::from_vec(vec![0f32; 64], 64, &Device::Cpu).unwrap();
        let mut s = GrpoSampler::new(2024, 1.0);
        // Advance past a few draws, then snapshot.
        for _ in 0..5 {
            let _ = s.sample(&logits).unwrap();
        }
        let snapshot = serde_json::to_vec(&s).unwrap();
        // The snapshot must capture the *advanced* state, not the initial seed —
        // otherwise "restore" would reset to the start and the test could pass
        // vacuously. Pin that it differs from a freshly-seeded sampler.
        let fresh = serde_json::to_vec(&GrpoSampler::new(2024, 1.0)).unwrap();
        assert_ne!(snapshot, fresh, "snapshot must capture advanced RNG state");

        // The "uninterrupted" continuation.
        let mut cont = s.clone();
        let expected: Vec<u32> = (0..10).map(|_| cont.sample(&logits).unwrap()).collect();

        // Restore from the snapshot and redraw — must match bit-for-bit.
        let mut restored: GrpoSampler = serde_json::from_slice(&snapshot).unwrap();
        let got: Vec<u32> = (0..10).map(|_| restored.sample(&logits).unwrap()).collect();

        assert_eq!(
            expected, got,
            "restored RNG state must reproduce the exact continuation"
        );
        assert_eq!(
            restored.temperature(),
            1.0,
            "temperature must round-trip too"
        );
    }

    /// The opaque byte-blob seam used by the checkpoint: `to_state_bytes` /
    /// `from_state_bytes` round-trip the advanced RNG state and the temperature, so a
    /// restored sampler reproduces the exact continuation; malformed bytes fail loud.
    #[test]
    fn state_bytes_round_trip_and_reject_garbage() {
        let logits = Tensor::from_vec(vec![0f32; 64], 64, &Device::Cpu).unwrap();
        let mut s = GrpoSampler::new(2024, 0.7);
        for _ in 0..5 {
            let _ = s.sample(&logits).unwrap();
        }
        let blob = s.to_state_bytes().unwrap();

        let mut cont = s.clone();
        let expected: Vec<u32> = (0..8).map(|_| cont.sample(&logits).unwrap()).collect();

        let mut restored = GrpoSampler::from_state_bytes(&blob).unwrap();
        let got: Vec<u32> = (0..8).map(|_| restored.sample(&logits).unwrap()).collect();
        assert_eq!(
            expected, got,
            "byte blob must reproduce the exact continuation"
        );
        assert_eq!(restored.temperature(), 0.7, "temperature must round-trip");

        assert!(
            GrpoSampler::from_state_bytes(b"not a sampler").is_err(),
            "a malformed blob must fail loud, not silently re-seed"
        );
    }

    /// `sample` must stay a pure delegation: same seed, one sampler driven by
    /// `sample` and one by `sample_with(baked_temperature, None)` produce the
    /// identical token stream (and so identical RNG consumption).
    #[test]
    fn sample_with_at_baked_params_matches_sample_stream() {
        let logits = Tensor::from_vec(vec![0f32; 64], 64, &Device::Cpu).unwrap();
        let mut a = GrpoSampler::new(11, 0.7);
        let mut b = GrpoSampler::new(11, 0.7);
        for _ in 0..16 {
            let ta = a.sample(&logits).unwrap();
            let (tb, _) = b.sample_with(&logits, 0.7, None).unwrap();
            assert_eq!(ta, tb, "sample_with diverged from sample");
        }
    }

    /// The returned log-prob is the log of the temperature-scaled softmax
    /// probability of the *sampled* token — pinned against an independent
    /// `log_softmax` recomputation, at temperature 1 and at an override.
    #[test]
    fn sampled_logprob_matches_an_independent_log_softmax() {
        let logits = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.0, 1.25], 5, &Device::Cpu).unwrap();
        for temperature in [1.0, 0.6] {
            let scaled = (&logits.to_dtype(DType::F32).unwrap() / temperature).unwrap();
            let want = candle_nn::ops::log_softmax(&scaled, 0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut s = GrpoSampler::new(3, 1.0);
            for _ in 0..8 {
                let (tok, lp) = s.sample_with(&logits, temperature, None).unwrap();
                assert!(
                    (lp - want[tok as usize]).abs() <= 1e-5,
                    "T={temperature}: logprob {lp} != log_softmax {} for token {tok}",
                    want[tok as usize]
                );
            }
        }
    }

    /// Nucleus filtering keeps exactly the smallest top-probability set whose
    /// cumulative mass reaches `top_p` — including the crossing token — and the
    /// returned log-prob is renormalized over that nucleus.
    #[test]
    fn top_p_samples_only_the_nucleus_and_renormalizes_the_logprob() {
        // p = [0.5, 0.3, 0.15, 0.05] via logits = ln(p): the cumulative after 2
        // tokens is 0.8, so top_p = 0.75 — strictly BETWEEN the rank-1 (0.5) and
        // rank-2 (0.8) cumulatives, not on the float boundary — keeps exactly
        // {0, 1} (the crossing token included), regardless of libm ln/exp
        // rounding in the round-trip.
        let p = [0.5f32, 0.3, 0.15, 0.05];
        let logits = Tensor::from_vec(
            p.iter().map(|x| x.ln()).collect::<Vec<f32>>(),
            4,
            &Device::Cpu,
        )
        .unwrap();
        let mut s = GrpoSampler::new(5, 1.0);
        for _ in 0..64 {
            let (tok, lp) = s.sample_with(&logits, 1.0, Some(0.75)).unwrap();
            assert!(tok < 2, "token {tok} sampled outside the nucleus {{0, 1}}");
            // The renormalization mass is the KEPT mass (0.5 + 0.3), not top_p.
            let want = (p[tok as usize] / 0.8).ln();
            assert!(
                (lp - want).abs() <= 1e-4,
                "logprob {lp} not renormalized over the nucleus (want {want})"
            );
        }
    }

    /// `top_p = 1.0` keeps every token — exactly (the `>= 1.0` early return),
    /// so the draw stream is bit-identical to the unfiltered path from the same
    /// seed. Skewed logits on purpose: a long low-probability tail is the case
    /// where an f64 cumulative of f32 probs can cross 1.0 *early* and a naive
    /// filter would zero genuinely sampleable tokens.
    #[test]
    fn top_p_one_is_identical_to_no_filter() {
        let skewed: Vec<f32> = (0..64).map(|i| -(i as f32) * 0.25).collect();
        let logits = Tensor::from_vec(skewed, 64, &Device::Cpu).unwrap();
        let mut a = GrpoSampler::new(7, 1.0);
        let mut b = GrpoSampler::new(7, 1.0);
        for _ in 0..32 {
            let (ta, la) = a.sample_with(&logits, 1.0, None).unwrap();
            let (tb, lb) = b.sample_with(&logits, 1.0, Some(1.0)).unwrap();
            assert_eq!(ta, tb, "top_p = 1.0 changed the draw stream");
            assert_eq!(la, lb, "top_p = 1.0 changed the logprob");
        }
    }

    /// An out-of-range `top_p` fails loud rather than silently sampling the
    /// full distribution.
    #[test]
    fn top_p_out_of_range_is_rejected() {
        let logits = Tensor::from_vec(vec![0f32; 8], 8, &Device::Cpu).unwrap();
        let mut s = GrpoSampler::new(1, 1.0);
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            assert!(
                s.sample_with(&logits, 1.0, Some(bad)).is_err(),
                "top_p = {bad} must be rejected"
            );
        }
    }

    /// A malformed temperature fails loud: a negative one would silently sample
    /// the inverted distribution, 0/NaN an opaque downstream error.
    #[test]
    fn malformed_temperature_is_rejected() {
        let logits = Tensor::from_vec(vec![0f32; 8], 8, &Device::Cpu).unwrap();
        let mut s = GrpoSampler::new(1, 1.0);
        for bad in [0.0, -0.7, f64::NAN, f64::INFINITY] {
            assert!(
                s.sample_with(&logits, bad, None).is_err(),
                "temperature = {bad} must be rejected"
            );
        }
    }

    /// Non-finite probabilities on the NUCLEUS path must return an error like
    /// the unfiltered path does — not hand `sort_by` a broken total order
    /// (which may panic): all `-inf` logits softmax to NaN; `total_cmp` keeps
    /// the sort lawful and `WeightedIndex` then rejects the weights.
    #[test]
    fn non_finite_probs_error_rather_than_panic_under_top_p() {
        let logits = Tensor::from_vec(vec![f32::NEG_INFINITY; 8], 8, &Device::Cpu).unwrap();
        let mut s = GrpoSampler::new(1, 1.0);
        assert!(s.sample_with(&logits, 1.0, None).is_err());
        assert!(s.sample_with(&logits, 1.0, Some(0.5)).is_err());
    }

    /// Different seeds diverge (the RNG is actually seeded, not constant).
    #[test]
    fn different_seeds_diverge() {
        let logits = Tensor::from_vec(vec![0f32; 256], 256, &Device::Cpu).unwrap();
        let mut a = GrpoSampler::new(1, 1.0);
        let mut b = GrpoSampler::new(2, 1.0);
        let sa: Vec<u32> = (0..32).map(|_| a.sample(&logits).unwrap()).collect();
        let sb: Vec<u32> = (0..32).map(|_| b.sample(&logits).unwrap()).collect();
        assert_ne!(sa, sb, "distinct seeds should produce distinct streams");
    }
}
