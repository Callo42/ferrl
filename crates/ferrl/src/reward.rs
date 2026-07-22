//! The reward abstraction.
//!
//! In `ferrl`, rewards are **plain `f32` scalars, never tensors**. A reward
//! function scores a completion against a typed [`Sample`] on the
//! host; the resulting scalars feed [`crate::grpo::group_advantages`] to form the
//! group-normalized advantages that drive the policy update. Keeping rewards off
//! the autograd tape is deliberate: GRPO treats the reward as a black-box signal,
//! so there is nothing to differentiate through and no reason to pay tensor
//! overhead.
//!
//! Rewards are **fallible**: scoring may invoke an external verifier (a sandboxed
//! code runner, a remote LLM-judge, a benchmark harness) that can fail for reasons
//! unrelated to completion quality. Such failures surface as [`RewardError`] and
//! propagate out of training/eval rather than being silently scored as zero.

use crate::sample::Sample;

use serde_json::Value;

/// Error returned by a [`RewardFn`] when a reward cannot be computed.
///
/// A failure here is distinct from a *low* reward: it means the verifier could not
/// produce a score at all (a process spawn, a timeout, an IO or network error).
/// The error **propagates out of training/eval** — never silently scored as zero,
/// which would corrupt the GRPO advantage signal.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RewardError {
    /// A reward function returned `NaN` or infinity instead of a real score.
    #[error("reward at group index {index} is non-finite: {value:?}")]
    NonFinite {
        /// Zero-based reward position inside the scored completion group.
        index: usize,
        /// The rejected IEEE-754 value.
        value: f32,
    },
    /// The reward could not be computed, described by a message.
    #[error("{0}")]
    Message(String),
    /// The reward could not be computed because an underlying verifier failed
    /// (sandbox / IO / network / judge).
    #[error("reward verifier failed: {0}")]
    Verifier(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
}

/// Reject a scored reward group unless every value is finite.
///
/// Training and evaluation call this immediately after checking the one-reward-
/// per-completion contract, before any value can reach publication, statistics,
/// advantage construction, metrics, or a distributed continuation decision.
pub(crate) fn validate_reward_values(rewards: &[f32]) -> Result<(), RewardError> {
    if let Some((index, &value)) = rewards
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RewardError::NonFinite { index, value });
    }
    Ok(())
}

impl RewardError {
    /// Construct a message-only reward error.
    pub fn msg(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }

    /// Wrap an underlying verifier error (sandbox / IO / network / judge).
    pub fn verifier(source: impl Into<Box<dyn std::error::Error + Send + Sync + 'static>>) -> Self {
        Self::Verifier(source.into())
    }
}

/// Scalar reward plus optional operator-facing diagnostics.
///
/// `diagnostic` is intentionally plain text: it is persisted in candidate ledgers to
/// explain low/zero rewards from external verifiers without making the core trainer
/// depend on task-specific status enums.
#[derive(Debug, Clone, PartialEq)]
pub struct RewardOutcome {
    /// Scalar reward assigned to one completion.
    pub reward: f32,
    /// Optional low-cardinality reason when the reward path can explain the score.
    pub diagnostic: Option<String>,
    /// Optional task-specific diagnostic payload for candidate ledgers.
    ///
    /// This is deliberately JSON-shaped and optional so verifier-backed tasks can
    /// preserve replay/debug evidence without forcing the trainer to understand
    /// task-specific enums or process statuses.
    pub metadata: Option<Value>,
}

impl RewardOutcome {
    /// Construct an outcome with no diagnostic.
    #[must_use]
    pub fn reward(reward: f32) -> Self {
        Self {
            reward,
            diagnostic: None,
            metadata: None,
        }
    }
}

/// Scores model completions against a typed ground-truth [`Sample`].
///
/// Implementors return one finite `f32` per completion. Rewards may be any real
/// value (they need not be normalized or bounded), but `NaN` and infinity are
/// invalid verifier outputs and training/evaluation reject them. GRPO normalizes
/// valid rewards per group downstream. A typical implementation reads the typed
/// [`Sample::target`], checks the completion against it (or a verifier), and
/// returns e.g. `1.0` / `0.0`.
pub trait RewardFn {
    /// The typed ground-truth target this reward scores against, carried by
    /// [`Sample::target`].
    ///
    /// A run is monomorphic in its target: [`crate::Trainer`] and
    /// [`crate::evaluate`] derive the sample type from this associated type, so a
    /// reward and its data are kept in sync by the compiler. Choose
    /// `Target = ()` for a reward that needs no ground truth, or e.g.
    /// `serde_json::Value` for a dynamically-shaped target.
    type Target;

    /// Score a single `completion` for `sample`.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] if the (possibly external) verifier could not
    /// produce a finite score (a sandbox, IO, network, judge, or non-finite output
    /// failure) — distinct from a merely low reward.
    fn reward(&self, sample: &Sample<Self::Target>, completion: &str) -> Result<f32, RewardError>;

    /// Score a batch of completions sharing one `sample` (a GRPO group).
    ///
    /// The default scores each completion independently via [`Self::reward`],
    /// short-circuiting on the first error. Override this as the **concurrency
    /// seam**: fan a group's completions out over rayon / threads / blocking-IO
    /// for sandboxed code execution or remote judge / benchmark verifiers (the
    /// core training loop stays synchronous).
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] if scoring any completion fails or produces a
    /// non-finite value; the default implementation short-circuits on the first
    /// error and validates the completed group.
    fn reward_group(
        &self,
        sample: &Sample<Self::Target>,
        completions: &[String],
    ) -> Result<Vec<f32>, RewardError> {
        let rewards = completions
            .iter()
            .map(|completion| self.reward(sample, completion))
            .collect::<Result<Vec<_>, _>>()?;
        validate_reward_values(&rewards)?;
        Ok(rewards)
    }

    /// Score a batch and optionally explain each score.
    ///
    /// The default preserves the historical API by delegating to
    /// [`Self::reward_group`] and attaching no diagnostics. Verifier-backed rewards
    /// can override this to make zero rewards fail-visible without running the
    /// verifier twice.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] under the same conditions as [`Self::reward_group`].
    fn reward_group_detailed(
        &self,
        sample: &Sample<Self::Target>,
        completions: &[String],
    ) -> Result<Vec<RewardOutcome>, RewardError> {
        let rewards = self.reward_group(sample, completions)?;
        validate_reward_values(&rewards)?;
        Ok(rewards.into_iter().map(RewardOutcome::reward).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Length-based toy reward: longer completions score higher (capped).
    struct LenReward;

    impl RewardFn for LenReward {
        type Target = ();
        fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
            Ok((completion.len() as f32).min(10.0))
        }
    }

    #[test]
    fn single_reward_is_scored() {
        let r = LenReward;
        let s = Sample::new("p", ());
        assert_eq!(r.reward(&s, "abc").unwrap(), 3.0);
        assert_eq!(r.reward(&s, "0123456789ABCDEF").unwrap(), 10.0);
    }

    #[test]
    fn default_group_maps_over_completions() {
        let r = LenReward;
        let s = Sample::new("p", ());
        let got = r
            .reward_group(&s, &["a".to_string(), "abcd".to_string()])
            .unwrap();
        assert_eq!(got, vec![1.0, 4.0]);
    }

    #[test]
    fn detailed_reward_group_defaults_to_rewards_without_diagnostics() {
        let r = LenReward;
        let s = Sample::new("p", ());
        let got = r
            .reward_group_detailed(&s, &["a".to_string(), "abcd".to_string()])
            .unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], RewardOutcome::reward(1.0));
        assert_eq!(got[1], RewardOutcome::reward(4.0));
    }

    /// Verifies the override path is reachable and used in place of the default.
    struct ConstGroup;

    impl RewardFn for ConstGroup {
        type Target = ();
        fn reward(&self, _sample: &Sample<()>, _completion: &str) -> Result<f32, RewardError> {
            Ok(0.0)
        }
        fn reward_group(
            &self,
            _sample: &Sample<()>,
            completions: &[String],
        ) -> Result<Vec<f32>, RewardError> {
            Ok(vec![1.0; completions.len()])
        }
    }

    #[test]
    fn override_group_takes_precedence() {
        let r = ConstGroup;
        let s = Sample::new("p", ());
        let got = r
            .reward_group(&s, &["x".to_string(), "y".to_string()])
            .unwrap();
        assert_eq!(got, vec![1.0, 1.0]);
    }

    /// A reward that always fails — used to verify error propagation.
    struct FailingReward;

    impl RewardFn for FailingReward {
        type Target = ();
        fn reward(&self, _sample: &Sample<()>, _completion: &str) -> Result<f32, RewardError> {
            Err(RewardError::msg("verifier exploded"))
        }
    }

    #[test]
    fn default_group_short_circuits_on_error() {
        let r = FailingReward;
        let s = Sample::new("p", ());
        let err = r
            .reward_group(&s, &["a".to_string(), "b".to_string()])
            .unwrap_err();
        assert!(matches!(err, RewardError::Message(m) if m == "verifier exploded"));
    }

    #[test]
    fn verifier_error_wraps_a_source() {
        let io = std::io::Error::new(std::io::ErrorKind::TimedOut, "judge timed out");
        let err = RewardError::verifier(io);
        assert!(matches!(err, RewardError::Verifier(_)));
        assert!(err.to_string().contains("judge timed out"));
    }

    struct NonFiniteReward;

    impl RewardFn for NonFiniteReward {
        type Target = ();

        fn reward(&self, _sample: &Sample<()>, _completion: &str) -> Result<f32, RewardError> {
            Ok(f32::NAN)
        }
    }

    #[test]
    fn default_group_rejects_nonfinite_single_reward_output() {
        let error = NonFiniteReward
            .reward_group(&Sample::new("p", ()), &["x".into()])
            .unwrap_err();
        assert!(matches!(error, RewardError::NonFinite { index: 0, .. }));
    }

    #[test]
    fn reward_value_validation_rejects_every_nonfinite_class() {
        validate_reward_values(&[-1.0, 0.0, 2.0]).unwrap();
        for value in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let error = validate_reward_values(&[1.0, value]).unwrap_err();
            assert!(matches!(
                error,
                RewardError::NonFinite { index: 1, value: rejected }
                    if rejected.to_bits() == value.to_bits()
            ));
        }
    }
}
