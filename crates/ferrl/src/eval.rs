//! Held-out evaluation: the base model vs. the trained adapter.
//!
//! [`evaluate`] scores a [`Policy`] on a set of held-out prompts twice — once with
//! the `LoRA` adapter **on** (the trained policy) and once with it **off** (the
//! frozen base) — and reports the mean reward of each. The gap
//! ([`EvalReport::improvement`]) is the headline the P4 gate turns on: the adapter
//! must beat the base on a held-out set.
//!
//! It is model-agnostic. It drives any [`Policy`] through the same
//! [`GenConfig`]-shaped generation the trainer uses, so the toy policy and the
//! Qwen policy are evaluated by identical code, and the adapter toggle is the same
//! seam the trainer uses for the KL reference.
//!
//! ## Sampling, not greedy
//!
//! Generation goes through [`Policy::generate`] — the policy's own (seeded)
//! sampler — so the reported means are Monte-Carlo estimates of `E[reward]` under
//! each policy at the rollout temperature, averaged over `group_size` samples per
//! prompt. Base and adapter draw from different points of the sampler's RNG
//! stream, so with a small `group_size` the comparison carries sampling variance;
//! pass a `group_size` large enough to resolve the gap you care about. `evaluate`
//! advances the policy's sampler, and restores the adapter-enabled flag to its
//! prior state before returning — on success, on a returned error, or on a panic
//! (an RAII guard).

use crate::policy::{GenConfig, Policy, Rollout};
use crate::reward::RewardFn;
use crate::trainer::TokenizerLike;

/// Per-prompt mean reward under the base model and under the adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromptEval {
    /// Mean reward of `group_size` samples with the adapter disabled (base model).
    pub base_mean: f32,
    /// Mean reward of `group_size` samples with the adapter enabled.
    pub adapter_mean: f32,
}

/// Aggregate result of an [`evaluate`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    /// Number of held-out prompts evaluated.
    pub n_prompts: usize,
    /// Completions sampled per prompt, for each of base and adapter.
    pub group_size: usize,
    /// Mean over prompts of the per-prompt base-model mean reward.
    pub base_reward_mean: f32,
    /// Mean over prompts of the per-prompt adapter mean reward.
    pub adapter_reward_mean: f32,
    /// Per-prompt detail, in `prompts` order.
    pub per_prompt: Vec<PromptEval>,
}

impl EvalReport {
    /// `adapter_reward_mean - base_reward_mean` — positive iff the adapter helped
    /// on the held-out set (the quantity the P4 gate checks).
    ///
    /// This is the difference of two **sampled** Monte-Carlo means (see the module
    /// docs): on a small `group_size` its sign carries sampling noise, so resolve
    /// the gate with a `group_size` large enough for the effect you expect.
    #[must_use]
    pub fn improvement(&self) -> f32 {
        self.adapter_reward_mean - self.base_reward_mean
    }
}

/// Errors raised during [`evaluate`].
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// The policy forward or sampling failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// The caller passed no prompts, or a prompt encoded to zero tokens (a policy
    /// reading the last prompt position would underflow).
    #[error("eval contract violation: {0}")]
    Contract(String),
}

/// Score `policy` on `prompts` with the adapter off (base) and on, returning the
/// mean reward of each.
///
/// For each prompt, `group_size` completions are sampled with the adapter
/// disabled and another `group_size` with it enabled; the mean of the **finite**
/// rewards of each group is the per-prompt score (a group with no finite reward
/// scores `0`, so a prompt whose generation wholly failed counts as a `0` rather
/// than a dropped prompt). The aggregate `*_reward_mean` fields are the unweighted
/// mean over prompts of those per-prompt scores. The adapter-enabled flag is
/// restored to its entry value before returning — on success, on a returned error,
/// and on a panic (an RAII guard).
///
/// `gen` drives [`Policy::generate`]. For a [`crate::QwenPolicy`], `gen.temperature`
/// **must** equal the temperature the policy was built with — that policy bakes its
/// sampler temperature and rejects a mismatch — otherwise generation returns an
/// error surfaced here as [`EvalError::Candle`].
///
/// # Errors
///
/// Returns [`EvalError::Contract`] if `prompts` is empty, a prompt encodes to zero
/// tokens, a policy returns a malformed rollout (wrong completion count, no prompt
/// context, or a sequence shorter than the prompt), or a [`RewardFn`] returns a
/// reward count that does not match the number of completions; or
/// [`EvalError::Candle`] if generation fails (including a `QwenPolicy` temperature
/// mismatch).
pub fn evaluate<P: Policy, R: RewardFn>(
    policy: &mut P,
    reward_fn: &R,
    tokenizer: &dyn TokenizerLike,
    prompts: &[String],
    gen: &GenConfig,
) -> Result<EvalReport, EvalError> {
    if prompts.is_empty() {
        return Err(EvalError::Contract("no eval prompts".into()));
    }
    // Restore the adapter flag on the way out — on success, on a `?` early return,
    // and on a panic — via the guard's Drop.
    let prior = policy.adapter_enabled();
    let guard = AdapterRestore { policy, prior };
    let per_prompt = score_all(guard.policy, reward_fn, tokenizer, prompts, gen)?;

    let n = per_prompt.len();
    let base_reward_mean = per_prompt.iter().map(|p| p.base_mean).sum::<f32>() / n as f32;
    let adapter_reward_mean = per_prompt.iter().map(|p| p.adapter_mean).sum::<f32>() / n as f32;
    Ok(EvalReport {
        n_prompts: n,
        group_size: gen.group_size,
        base_reward_mean,
        adapter_reward_mean,
        per_prompt,
    })
}

/// Restores a policy's adapter-enabled flag when dropped, so [`evaluate`] leaves
/// the policy as it found it even if scoring returns early or panics.
struct AdapterRestore<'a, P: Policy> {
    policy: &'a mut P,
    prior: bool,
}

impl<P: Policy> Drop for AdapterRestore<'_, P> {
    fn drop(&mut self) {
        self.policy.set_adapter_enabled(self.prior);
    }
}

/// Score each prompt base-then-adapter; the caller's guard restores the flag.
fn score_all<P: Policy, R: RewardFn>(
    policy: &mut P,
    reward_fn: &R,
    tokenizer: &dyn TokenizerLike,
    prompts: &[String],
    gen: &GenConfig,
) -> Result<Vec<PromptEval>, EvalError> {
    let mut per_prompt = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let ids = tokenizer.encode(prompt);
        if ids.is_empty() {
            return Err(EvalError::Contract(format!(
                "prompt encoded to zero tokens: {prompt:?}"
            )));
        }
        policy.set_adapter_enabled(false);
        let base_mean = mean_reward(policy, reward_fn, tokenizer, prompt, &ids, gen)?;
        policy.set_adapter_enabled(true);
        let adapter_mean = mean_reward(policy, reward_fn, tokenizer, prompt, &ids, gen)?;
        per_prompt.push(PromptEval {
            base_mean,
            adapter_mean,
        });
    }
    Ok(per_prompt)
}

/// Sample one group for `prompt` and return the mean of its finite rewards.
fn mean_reward<P: Policy, R: RewardFn>(
    policy: &mut P,
    reward_fn: &R,
    tokenizer: &dyn TokenizerLike,
    prompt: &str,
    prompt_ids: &[u32],
    gen: &GenConfig,
) -> Result<f32, EvalError> {
    let rollout = policy.generate(prompt_ids, gen)?;
    validate_rollout(&rollout, gen.group_size)?;
    let completions: Vec<String> = rollout
        .token_ids
        .iter()
        // `validate_rollout` guarantees every sequence >= prompt_len; slice
        // defensively anyway so a future change can never turn into a panic.
        .map(|ids| tokenizer.decode(ids.get(rollout.prompt_len..).unwrap_or(&[])))
        .collect();
    let rewards = reward_fn.reward_group(prompt, &completions);
    // Enforce the RewardFn contract (one reward per completion), exactly as the
    // trainer does — otherwise eval would average over a different sample count
    // than it generated and skew the base-vs-adapter comparison.
    if rewards.len() != completions.len() {
        return Err(EvalError::Contract(format!(
            "reward_group returned {} rewards for {} completions",
            rewards.len(),
            completions.len()
        )));
    }
    Ok(finite_mean(&rewards))
}

/// Reject a malformed rollout the same way the trainer's `completion_dims` does, so
/// eval and train agree on what a valid `Policy::generate` returns: exactly
/// `group_size` completions, a non-empty prompt context, and every sequence at
/// least as long as the prompt (so the completion slice is well-defined). Unlike
/// the trainer it does **not** require rectangular completions — eval decodes and
/// scores each sequence independently, so ragged (early-stopped) completions are
/// fine.
fn validate_rollout(rollout: &Rollout, group_size: usize) -> Result<(), EvalError> {
    if rollout.len() != group_size {
        return Err(EvalError::Contract(format!(
            "Policy::generate returned {} completions for group_size {group_size}",
            rollout.len()
        )));
    }
    if rollout.prompt_len == 0 {
        return Err(EvalError::Contract(
            "rollout has no prompt context (prompt_len == 0)".into(),
        ));
    }
    if let Some(short) = rollout
        .token_ids
        .iter()
        .find(|ids| ids.len() < rollout.prompt_len)
    {
        return Err(EvalError::Contract(format!(
            "rollout sequence length {} is shorter than prompt_len {}",
            short.len(),
            rollout.prompt_len
        )));
    }
    Ok(())
}

/// Mean of the finite rewards (non-finite rewards are dropped, mirroring the
/// trainer's group statistics); an all-non-finite group contributes `0`.
fn finite_mean(rewards: &[f32]) -> f32 {
    let finite: Vec<f32> = rewards.iter().copied().filter(|r| r.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.iter().sum::<f32>() / finite.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
    use std::cell::RefCell;

    /// A deterministic [`Policy`] for testing the harness in isolation (no model,
    /// no sampling): every completion token is `base_tok` when the adapter is off
    /// and `adapter_tok` when it is on, so base and adapter are cleanly separable
    /// and the per-prompt means are exact. It records every adapter toggle so the
    /// test can assert the base-then-adapter ordering and the final restore.
    struct ScriptedPolicy {
        enabled: bool,
        base_tok: u32,
        adapter_tok: u32,
        toggles: RefCell<Vec<bool>>,
    }

    impl ScriptedPolicy {
        fn new(base_tok: u32, adapter_tok: u32) -> Self {
            Self {
                enabled: true,
                base_tok,
                adapter_tok,
                toggles: RefCell::new(Vec::new()),
            }
        }
    }

    impl Policy for ScriptedPolicy {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            let tok = if self.enabled {
                self.adapter_tok
            } else {
                self.base_tok
            };
            let mut token_ids = Vec::with_capacity(cfg.group_size);
            for _ in 0..cfg.group_size {
                let mut ids = prompt.to_vec();
                ids.extend(std::iter::repeat_n(tok, cfg.max_new_tokens));
                token_ids.push(ids);
            }
            Ok(Rollout::rectangular(token_ids, prompt.len()))
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            // Unused by the eval harness; return a trivial tensor.
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
            self.toggles.borrow_mut().push(enabled);
        }

        fn adapter_enabled(&self) -> bool {
            self.enabled
        }

        fn trainable_vars(&self) -> Vec<Var> {
            vec![]
        }
    }

    /// Codec over single decimal digits: char `'d'` <-> token `d`.
    struct DigitCodec;
    impl TokenizerLike for DigitCodec {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.chars().filter_map(|c| c.to_digit(10)).collect()
        }
        fn decode(&self, ids: &[u32]) -> String {
            ids.iter()
                .filter_map(|&i| char::from_digit(i % 10, 10))
                .collect()
        }
    }

    /// Reward = sum of the completion's decimal digits.
    struct DigitSumReward;
    impl RewardFn for DigitSumReward {
        fn reward(&self, _prompt: &str, completion: &str) -> f32 {
            completion
                .chars()
                .filter_map(|c| c.to_digit(10))
                .sum::<u32>() as f32
        }
    }

    fn gen(group_size: usize, max_new_tokens: usize) -> GenConfig {
        GenConfig {
            group_size,
            max_new_tokens,
            temperature: 1.0,
            eos_token_id: None,
        }
    }

    #[test]
    fn separates_base_from_adapter_with_exact_means() {
        // base token = 0 -> completion "000" -> reward 0.
        // adapter token = 2 -> completion "222" -> reward 6 (3 digits * 2).
        let mut policy = ScriptedPolicy::new(0, 2);
        let prompts = vec!["1".to_string(), "9".to_string()];
        let report = evaluate(
            &mut policy,
            &DigitSumReward,
            &DigitCodec,
            &prompts,
            &gen(4, 3),
        )
        .unwrap();

        // Exact, deterministic values (0.0 and 6.0 are representable): compare the
        // whole report at once.
        let expected = EvalReport {
            n_prompts: 2,
            group_size: 4,
            base_reward_mean: 0.0,
            adapter_reward_mean: 6.0,
            per_prompt: vec![
                PromptEval {
                    base_mean: 0.0,
                    adapter_mean: 6.0
                };
                2
            ],
        };
        assert_eq!(report, expected);
        assert_eq!(report.improvement(), 6.0);
    }

    #[test]
    fn toggles_base_then_adapter_per_prompt_and_restores_prior_state() {
        let mut policy = ScriptedPolicy::new(0, 1);
        assert!(policy.adapter_enabled()); // prior == true
        let prompts = vec!["5".to_string()];
        evaluate(
            &mut policy,
            &DigitSumReward,
            &DigitCodec,
            &prompts,
            &gen(2, 2),
        )
        .unwrap();

        // One prompt: off (base), on (adapter), then restore to prior (true).
        assert_eq!(*policy.toggles.borrow(), vec![false, true, true]);
        assert!(policy.adapter_enabled(), "adapter flag not restored");
    }

    #[test]
    fn restores_prior_disabled_state() {
        let mut policy = ScriptedPolicy::new(0, 1);
        policy.set_adapter_enabled(false); // prior == false
        policy.toggles.borrow_mut().clear();
        let prompts = vec!["5".to_string()];
        evaluate(
            &mut policy,
            &DigitSumReward,
            &DigitCodec,
            &prompts,
            &gen(2, 2),
        )
        .unwrap();
        assert!(
            !policy.adapter_enabled(),
            "prior disabled state not restored"
        );
    }

    #[test]
    fn empty_prompts_is_a_contract_error() {
        let mut policy = ScriptedPolicy::new(0, 1);
        let err = evaluate(&mut policy, &DigitSumReward, &DigitCodec, &[], &gen(2, 2)).unwrap_err();
        assert!(matches!(err, EvalError::Contract(_)), "got {err:?}");
    }

    #[test]
    fn zero_token_prompt_is_a_contract_error() {
        // "abc" has no decimal digits, so DigitCodec encodes it to []; the harness
        // must reject it (and still restore the adapter flag).
        let mut policy = ScriptedPolicy::new(0, 1);
        let prompts = vec!["abc".to_string()];
        let err = evaluate(
            &mut policy,
            &DigitSumReward,
            &DigitCodec,
            &prompts,
            &gen(2, 2),
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::Contract(_)), "got {err:?}");
        assert!(policy.adapter_enabled(), "flag not restored after error");
    }

    #[test]
    fn finite_mean_drops_non_finite_rewards() {
        assert_eq!(finite_mean(&[1.0, 3.0]), 2.0);
        assert_eq!(finite_mean(&[1.0, f32::NAN, 3.0]), 2.0);
        assert_eq!(finite_mean(&[f32::INFINITY, f32::NAN]), 0.0);
        assert_eq!(finite_mean(&[]), 0.0);
    }

    /// The malformed-rollout shapes the harness must reject — one per
    /// `validate_rollout` branch.
    #[derive(Debug, Clone, Copy)]
    enum Malformed {
        /// One too many completions (`len != group_size`).
        Overfill,
        /// The first sequence is shorter than `prompt_len`.
        ShortSeq,
        /// `prompt_len == 0` (no prompt context).
        ZeroPromptLen,
    }

    /// A policy that emits a chosen malformed rollout.
    struct MalformedPolicy(Malformed);
    impl Policy for MalformedPolicy {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            let body = |extra: u32| {
                let mut ids = prompt.to_vec();
                ids.push(extra);
                ids
            };
            let (token_ids, prompt_len) = match self.0 {
                Malformed::Overfill => {
                    let v = (0..cfg.group_size + 1).map(|_| body(0)).collect();
                    (v, prompt.len())
                }
                Malformed::ShortSeq => {
                    let mut v: Vec<Vec<u32>> = (0..cfg.group_size).map(|_| body(0)).collect();
                    v[0] = Vec::new(); // shorter than prompt_len (>= 1)
                    (v, prompt.len())
                }
                Malformed::ZeroPromptLen => {
                    let v = (0..cfg.group_size).map(|_| vec![0u32, 1]).collect();
                    (v, 0)
                }
            };
            Ok(Rollout::rectangular(token_ids, prompt_len))
        }
        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)
        }
        fn set_adapter_enabled(&mut self, _enabled: bool) {}
        fn adapter_enabled(&self) -> bool {
            true
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![]
        }
    }

    #[test]
    fn rejects_malformed_rollouts() {
        let prompts = vec!["5".to_string()];
        for mode in [
            Malformed::Overfill,
            Malformed::ShortSeq,
            Malformed::ZeroPromptLen,
        ] {
            let mut policy = MalformedPolicy(mode);
            let err = evaluate(
                &mut policy,
                &DigitSumReward,
                &DigitCodec,
                &prompts,
                &gen(2, 2),
            )
            .unwrap_err();
            assert!(
                matches!(err, EvalError::Contract(_)),
                "{mode:?}: got {err:?}"
            );
        }
    }

    /// A reward that violates the one-reward-per-completion contract.
    struct WrongCountReward;
    impl RewardFn for WrongCountReward {
        fn reward(&self, _prompt: &str, _completion: &str) -> f32 {
            0.0
        }
        fn reward_group(&self, _prompt: &str, _completions: &[String]) -> Vec<f32> {
            vec![1.0] // always one reward, regardless of the group size
        }
    }

    #[test]
    fn rejects_reward_count_mismatch() {
        // group_size 3 -> 3 completions, but reward_group returns 1 reward.
        let mut policy = ScriptedPolicy::new(0, 1);
        let prompts = vec!["5".to_string()];
        let err = evaluate(
            &mut policy,
            &WrongCountReward,
            &DigitCodec,
            &prompts,
            &gen(3, 2),
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::Contract(_)), "got {err:?}");
    }
}
