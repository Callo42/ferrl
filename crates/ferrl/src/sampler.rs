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
    /// `LogitsProcessor` runs, on ferrl's own (serializable) RNG.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a tensor op fails or the probabilities are not a
    /// valid categorical distribution (e.g. all-zero or non-finite weights, which
    /// `WeightedIndex` rejects) — the same failure modes as candle's sampler.
    pub fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let logits = (&logits / self.temperature)?;
        let prs: Vec<f32> = candle_nn::ops::softmax_last_dim(&logits)?.to_vec1()?;
        let distr = WeightedIndex::new(&prs).map_err(candle_core::Error::wrap)?;
        let token = distr.sample(&mut self.rng) as u32;
        Ok(token)
    }
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
