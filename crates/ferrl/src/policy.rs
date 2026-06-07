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

use candle_core::{Result as CandleResult, Tensor};

/// A batch of sampled completions: token ids plus the prompt length that
/// produced them, so callers can slice prompt from completion.
#[derive(Debug, Clone)]
pub struct Rollout {
    /// Token ids for each sequence, prompt tokens followed by completion tokens.
    pub token_ids: Vec<Vec<u32>>,
    /// Number of leading prompt tokens shared by every sequence in this rollout.
    pub prompt_len: usize,
}

impl Rollout {
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

/// Sampling controls for [`Policy::generate`].
#[derive(Debug, Clone, Copy)]
pub struct GenConfig {
    /// Number of completions to sample per prompt (the GRPO group size).
    pub group_size: usize,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Softmax temperature; `1.0` is unscaled.
    pub temperature: f64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            max_new_tokens: 256,
            temperature: 1.0,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_len_and_empty() {
        let r = Rollout {
            token_ids: vec![vec![1, 2, 3], vec![1, 4, 5]],
            prompt_len: 1,
        };
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());

        let e = Rollout {
            token_ids: vec![],
            prompt_len: 0,
        };
        assert_eq!(e.len(), 0);
        assert!(e.is_empty());
    }

    #[test]
    fn gen_config_default_group_size() {
        let c = GenConfig::default();
        assert_eq!(c.group_size, 8);
        assert_eq!(c.max_new_tokens, 256);
        assert_eq!(c.temperature, 1.0);
    }
}
