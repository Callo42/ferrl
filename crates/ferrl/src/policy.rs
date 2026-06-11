//! The policy abstraction.
//!
//! A [`Policy`] is the trainable language model viewed through the narrow
//! interface GRPO needs: sample completions, score tokens under the current
//! parameters, and toggle the `LoRA` adapter so the same weights can serve as both
//! the trainable policy (adapter on) and the frozen reference (adapter off).
//!
//! The concrete `candle-transformers` model implementation lives behind this
//! trait so the trainer is model-agnostic and so the pure-math layer never
//! depends on a particular architecture. Generation returns token ids; log-prob
//! scoring returns a [`candle_core::Tensor`] (it lives on the autograd tape so
//! the surrogate can be differentiated), whereas rewards stay scalar (see
//! [`crate::reward`]).

use candle_core::{Result as CandleResult, Tensor, Var};

/// A batch of sampled completions: token ids plus the prompt length that
/// produced them, so callers can slice prompt from completion.
#[derive(Debug, Clone)]
pub struct Rollout {
    /// Token ids for each sequence, prompt tokens followed by completion tokens.
    pub token_ids: Vec<Vec<u32>>,
    /// Number of leading prompt tokens shared by every sequence in this rollout.
    pub prompt_len: usize,
    /// Number of *real* completion tokens in each sequence — the count of
    /// generated tokens up to and including the first EOS (EOS-inclusive), or the
    /// full completion width if no EOS was sampled. Positions at or beyond this
    /// index within the (rectangular) completion are padding and are masked out of
    /// the GRPO loss. For a fixed-length, no-early-stop rollout this equals the
    /// full completion width for every sequence (see [`Rollout::rectangular`]); a
    /// per-element value is in `0..=comp_len`.
    pub completion_lens: Vec<usize>,
    /// Per-token **behavior-policy** log-probabilities of the sampled completion
    /// tokens — the probability each token was *actually drawn with* (the rollout
    /// path's logits at the sampling temperature, nucleus-renormalized when top-p
    /// is active), captured by the sampler at draw time. Row `i` has exactly
    /// [`completion_lens`](Self::completion_lens)`[i]` entries (one per real
    /// draw; EOS padding was never sampled, so it carries no log-prob).
    ///
    /// `None` when the policy does not capture them (toy/test policies;
    /// [`Rollout::rectangular`] always sets `None`). When present, the trainer
    /// compares them against its own scoring forward to surface the
    /// rollout-vs-train mismatch (the off-policy gap a cached/merged — possibly
    /// bf16 — decode path opens against the f32 uncached scoring forward) as
    /// ratio telemetry, and optionally corrects the surrogate with truncated
    /// importance sampling (TIS).
    pub rollout_logprobs: Option<Vec<Vec<f32>>>,
}

impl Rollout {
    /// Construct a **rectangular** rollout in which every sequence is a real
    /// completion of the full width — the legacy, no-EOS-early-stop behavior.
    ///
    /// Each `completion_lens[i]` is `token_ids[i].len() - prompt_len` (saturating,
    /// so a degenerate `prompt_len >= row length` yields `0` rather than
    /// panicking). EOS-aware generation, which stops sequences early, instead sets
    /// `completion_lens` to the true per-sequence lengths and right-pads
    /// `token_ids` to a common width.
    #[must_use]
    pub fn rectangular(token_ids: Vec<Vec<u32>>, prompt_len: usize) -> Self {
        let completion_lens = token_ids
            .iter()
            .map(|ids| ids.len().saturating_sub(prompt_len))
            .collect();
        Self {
            token_ids,
            prompt_len,
            completion_lens,
            rollout_logprobs: None,
        }
    }

    /// Number of sequences in the rollout.
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Whether the rollout contains no sequences.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// **Eval-only** sampling parameters, the deliberate override channel for
/// [`GenConfig::eval_sampling`].
///
/// Training rollouts must sample at the policy's own (baked, scoring-consistent)
/// temperature — [`crate::LmPolicy`] fails loud on a mismatched
/// [`GenConfig::temperature`] precisely to catch a drifted config. Held-out
/// evaluation legitimately wants *different* sampling (the 2026 convention:
/// temperature ≈ 0.6 with nucleus top-p 0.95 for avg@k sampled pass@1), so the
/// override is a separate, explicit field rather than a relaxation of that
/// check: setting it says "I know this is not the training distribution."
/// Nucleus (top-p) filtering can **only** enter generation through here —
/// training rollouts structurally stay untruncated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalSampling {
    /// Eval softmax temperature (must be finite and `> 0`).
    pub temperature: f64,
    /// Optional nucleus filter: keep the smallest top-probability set whose
    /// cumulative mass reaches this value (in `(0, 1]`; the crossing token is
    /// included), renormalize, and sample from it. `None` disables filtering.
    pub top_p: Option<f64>,
}

impl Default for EvalSampling {
    /// The 2026 eval convention: temperature `0.6`, nucleus top-p `0.95`.
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: Some(0.95),
        }
    }
}

/// Sampling controls for [`Policy::generate`].
#[derive(Debug, Clone, Copy)]
pub struct GenConfig {
    /// Number of completions to sample per prompt (the GRPO group size).
    pub group_size: usize,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Softmax temperature; `1.0` is unscaled. Ignored when
    /// [`eval_sampling`](Self::eval_sampling) is set (the override carries its
    /// own temperature).
    pub temperature: f64,
    /// End-of-sequence token. When `Some(id)`, a sampled `id` ends that sequence's
    /// completion early (the EOS token is **kept** — the recorded length is
    /// EOS-*inclusive*) and the sequence is right-padded back to `max_new_tokens`
    /// so the group stays rectangular at a fixed width; [`Rollout::completion_lens`]
    /// records the true per-sequence length. When `None` (the default) no sequence
    /// stops early, every completion is the full `max_new_tokens`, and the rollout
    /// is bit-identical to the legacy no-early-stop behavior. A [`Policy`] backed by
    /// a model with no EOS token (e.g. a base model) leaves this `None`.
    pub eos_token_id: Option<u32>,
    /// Eval-only sampling override (see [`EvalSampling`]). `None` (the default,
    /// and what the trainer always passes) samples at
    /// [`temperature`](Self::temperature) with no nucleus filtering, exactly the
    /// legacy behavior; `Some` deliberately samples the eval distribution
    /// instead. The held-out eval harness is the intended setter.
    pub eval_sampling: Option<EvalSampling>,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            max_new_tokens: 256,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        }
    }
}

/// The trainable policy GRPO drives.
///
/// # The three log-probability roles
///
/// A GRPO update needs three per-token log-probabilities of the *same* sampled
/// completions. All three are obtained from this one trait — no extra method is
/// required, so adding the inner optimization loop (`μ > 1`) later is **not** a
/// breaking change:
///
/// - **current** `logp` — [`token_logprobs`](Policy::token_logprobs) with the
///   adapter [enabled](Policy::set_adapter_enabled). Lives on the autograd tape;
///   it is the numerator of the importance ratio and what `backward` flows
///   through.
/// - **old** `logp_old` — a *frozen snapshot* of the current log-probs, taken
///   once when the rollout is generated (adapter enabled) and then **detached**
///   from the tape and stored by the trainer. It is the ratio denominator in
///   `exp(logp - logp_old)`. With a single inner step (`μ = 1`) it equals `logp`
///   and the ratio is exactly `1`. The trait deliberately does *not* own this
///   snapshot, keeping the policy stateless with respect to the optimizer step.
/// - **reference** `logp_ref` — [`token_logprobs`](Policy::token_logprobs) with
///   the adapter [disabled](Policy::set_adapter_enabled): the frozen base model.
///   Feeds the k3 KL ([`crate::grpo::k3_kl`]) and is likewise detached (the
///   reference is never trained).
///
/// The adapter toggle is what lets one set of weights serve all three: the same
/// parameters are the policy (adapter on) and their own reference (adapter off);
/// "old" is just an adapter-on snapshot frozen in time.
pub trait Policy {
    /// Sample `cfg.group_size` completions for `prompt` under `cfg`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the forward pass or sampling fails.
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout>;

    /// Per-token log-probabilities of `rollout`'s completion tokens under the
    /// current parameters, as a differentiable [`Tensor`].
    ///
    /// The returned tensor has shape `[num_sequences, completion_len]` and is
    /// connected to the trainable [`candle_core::Var`]s so the GRPO surrogate
    /// built from it can be back-propagated.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the forward pass fails.
    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor>;

    /// Enable (`true`) or disable (`false`) the `LoRA` adapter contribution.
    ///
    /// With the adapter disabled the policy computes the frozen base-model
    /// distribution, i.e. the GRPO reference policy.
    fn set_adapter_enabled(&mut self, enabled: bool);

    /// Whether the `LoRA` adapter is currently contributing to the forward pass.
    fn adapter_enabled(&self) -> bool;

    /// The trainable parameters the optimizer updates and the grad-coverage
    /// canary checks each step (e.g. the `LoRA` `A`/`B` factors).
    ///
    /// A cloned [`Var`] shares its inner storage (and tensor id) with the
    /// original, so the returned vars *alias* the parameters used inside
    /// [`token_logprobs`](Policy::token_logprobs): the trainer registers exactly
    /// these with the optimizer and looks them up in the grad store after
    /// `backward`. Implementors typically forward to their adapter's
    /// `trainable_vars()`.
    fn trainable_vars(&self) -> Vec<Var>;

    /// Serialize the policy's rollout-sampler RNG state to an opaque byte blob, for
    /// momentum-faithful checkpoint persistence.
    ///
    /// A faithful resume must continue the *same* rollout token stream an uninterrupted
    /// run would have produced, which requires capturing the sampler's RNG state.
    /// candle's `LogitsProcessor` hides its `StdRng` behind no accessor — which is why
    /// ferrl owns [`GrpoSampler`](crate::sampler::GrpoSampler), whose state *is*
    /// `serde`-serializable. The returned blob is opaque to the checkpoint; only
    /// [`restore_sampler_state`](Self::restore_sampler_state) interprets it. A policy
    /// with no rollout RNG returns an empty blob.
    ///
    /// This is a **required** method (not defaulted) so that giving a policy a sampler
    /// can never silently skip RNG capture — the resume footgun a faithful checkpoint
    /// must avoid.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the sampler state cannot be serialized.
    fn sampler_state(&self) -> CandleResult<Vec<u8>>;

    /// Restore the rollout-sampler RNG state from a blob produced by
    /// [`sampler_state`](Self::sampler_state), so a resumed run continues the exact
    /// token stream. **Fails loud** if the blob is malformed or does not match this
    /// policy's sampler, rather than silently re-seeding.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `state` is not a valid blob for this policy's sampler.
    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()>;

    /// The policy's `LoRA` recipe as a stable canonical string, recorded into
    /// checkpoint manifests (see
    /// [`crate::checkpoint::CheckpointManifest::lora_recipe`]). Informational —
    /// the checkpoint load contract stays positional. Defaulted (`None`) so toy
    /// / test policies need not implement it; model-backed policies forward
    /// their [`crate::GradModel::lora_recipe`].
    fn lora_recipe(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_len_and_empty() {
        let r = Rollout::rectangular(vec![vec![1, 2, 3], vec![1, 4, 5]], 1);
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());

        let e = Rollout::rectangular(vec![], 0);
        assert_eq!(e.len(), 0);
        assert!(e.is_empty());
    }

    #[test]
    fn rectangular_fills_completion_lens_to_full_width() {
        // Every sequence is a full-width real completion: prompt_len 2, comp 3.
        let r = Rollout::rectangular(vec![vec![1, 2, 3, 4, 5], vec![1, 2, 6, 7, 8]], 2);
        assert_eq!(r.completion_lens, vec![3, 3]);
        // An empty rollout has no per-sequence lengths.
        let e = Rollout::rectangular(vec![], 0);
        assert!(e.completion_lens.is_empty());
        // Saturating: a row no longer than the prompt yields a zero completion.
        let z = Rollout::rectangular(vec![vec![1, 2]], 2);
        assert_eq!(z.completion_lens, vec![0]);
    }

    #[test]
    fn gen_config_default_group_size() {
        let c = GenConfig::default();
        assert_eq!(c.group_size, 8);
        assert_eq!(c.max_new_tokens, 256);
        assert_eq!(c.temperature, 1.0);
        // The default disables EOS early-stop, preserving legacy full-width rollouts.
        assert_eq!(c.eos_token_id, None);
        // No eval override by default: training-path sampling, no nucleus filter.
        assert_eq!(c.eval_sampling, None);
    }

    #[test]
    fn eval_sampling_default_is_the_2026_convention() {
        let e = EvalSampling::default();
        assert_eq!(e.temperature, 0.6);
        assert_eq!(e.top_p, Some(0.95));
    }

    #[test]
    fn rectangular_captures_no_rollout_logprobs() {
        // The convenience constructor is the no-capture path by contract: toy
        // policies built through it must never claim behavior log-probs.
        let r = Rollout::rectangular(vec![vec![1, 2, 3]], 1);
        assert_eq!(r.rollout_logprobs, None);
    }
}
