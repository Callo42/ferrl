//! The reward abstraction.
//!
//! In `ferrl`, rewards are **plain `f32` scalars, never tensors**. A reward
//! function scores a completion (given its prompt) on the host; the resulting
//! scalars feed [`crate::grpo::group_advantages`] to form the group-normalized
//! advantages that drive the policy update. Keeping rewards off the autograd
//! tape is deliberate: GRPO treats the reward as a black-box signal, so there is
//! nothing to differentiate through and no reason to pay tensor overhead.

/// Scores model completions against their prompts.
///
/// Implementors return one `f32` per `(prompt, completion)` pair. Rewards may be
/// any real value (they need not be normalized or bounded); GRPO normalizes them
/// per group downstream. A typical implementation parses the completion, checks
/// it against a ground-truth or a verifier, and returns e.g. `1.0` / `0.0`.
pub trait RewardFn {
    /// Score a single completion for a given prompt.
    fn reward(&self, prompt: &str, completion: &str) -> f32;

    /// Score a batch of completions sharing one `prompt` (a GRPO group).
    ///
    /// The default scores each completion independently via [`Self::reward`];
    /// override when batch context (e.g. relative ranking) is cheaper or more
    /// meaningful to compute together.
    fn reward_group(&self, prompt: &str, completions: &[String]) -> Vec<f32> {
        completions.iter().map(|c| self.reward(prompt, c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Length-based toy reward: longer completions score higher (capped).
    struct LenReward;

    impl RewardFn for LenReward {
        fn reward(&self, _prompt: &str, completion: &str) -> f32 {
            (completion.len() as f32).min(10.0)
        }
    }

    #[test]
    fn single_reward_is_scored() {
        let r = LenReward;
        assert_eq!(r.reward("p", "abc"), 3.0);
        assert_eq!(r.reward("p", "0123456789ABCDEF"), 10.0);
    }

    #[test]
    fn default_group_maps_over_completions() {
        let r = LenReward;
        let got = r.reward_group("p", &["a".to_string(), "abcd".to_string()]);
        assert_eq!(got, vec![1.0, 4.0]);
    }

    /// Verifies the override path is reachable and used in place of the default.
    struct ConstGroup;

    impl RewardFn for ConstGroup {
        fn reward(&self, _prompt: &str, _completion: &str) -> f32 {
            0.0
        }
        fn reward_group(&self, _prompt: &str, completions: &[String]) -> Vec<f32> {
            vec![1.0; completions.len()]
        }
    }

    #[test]
    fn override_group_takes_precedence() {
        let r = ConstGroup;
        let got = r.reward_group("p", &["x".to_string(), "y".to_string()]);
        assert_eq!(got, vec![1.0, 1.0]);
    }
}
