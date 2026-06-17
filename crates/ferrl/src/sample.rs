//! The training datum: a prompt paired with its typed ground-truth target.
//!
//! A [`Sample<T>`] is what a `ferrl` run consumes — one per training/eval item.
//! The `target` carries whatever a verifier needs to score a completion (a parsed
//! problem, an expected answer, a test suite) in a **structured, typed** form,
//! replacing the fragile practice of smuggling the ground truth into the prompt
//! string and re-parsing it at scoring time.
//!
//! `Sample<T>` is `serde`-(de)serializable whenever `T` is, which is what lets
//! [`crate::data::read_jsonl`] ingest a dataset from disk: each JSONL line is one
//! `{"prompt": ..., "target": ...}` object.

use serde::{Deserialize, Serialize};

/// A single training/eval datum: a `prompt` to roll out, paired with the typed
/// ground-truth `target` the reward scores the completion against.
///
/// `T` is the reward's [`crate::reward::RewardFn::Target`]. A run is monomorphic
/// in its target: [`crate::Trainer`] and [`crate::evaluate`] derive the sample
/// type from the reward, so there is no separate target knob to keep in sync.
///
/// The derived [`Serialize`]/[`Deserialize`] impls are bounded on `T` (they apply
/// only when `T` is itself (de)serializable), so a sample round-trips through JSON
/// — the wire shape is a flat object `{"prompt": "...", "target": <T>}`.
///
/// ```
/// use ferrl::Sample;
///
/// // A typed target can be anything — here a tuple of (operands, expected sum).
/// let s = Sample::new("2 + 3 = ?", (vec![2u32, 3], 5u32));
/// assert_eq!(s.prompt, "2 + 3 = ?");
/// assert_eq!(s.target, (vec![2, 3], 5));
/// // When `T` is serializable, the sample round-trips through JSON as a flat
/// // object `{"prompt": ..., "target": ...}` (see `ferrl::data::read_jsonl`).
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sample<T> {
    /// The prompt fed to the policy for rollout.
    pub prompt: String,
    /// The typed ground truth the reward scores the completion against.
    pub target: T,
}

impl<T> Sample<T> {
    /// Construct a sample from a prompt and its typed target.
    pub fn new(prompt: impl Into<String>, target: T) -> Self {
        Self {
            prompt: prompt.into(),
            target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Target {
        numbers: Vec<u32>,
        answer: i64,
    }

    #[test]
    fn round_trips_a_struct_target_through_json() {
        let s = Sample::new(
            "reach 6",
            Target {
                numbers: vec![1, 2, 3],
                answer: 6,
            },
        );
        let json = serde_json::to_string(&s).unwrap();
        // The wire shape is a flat object — prompt + the target's own fields nested
        // under `target`.
        assert_eq!(
            json,
            r#"{"prompt":"reach 6","target":{"numbers":[1,2,3],"answer":6}}"#
        );
        let back: Sample<Target> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn deserializes_a_unit_target() {
        let s: Sample<()> = serde_json::from_str(r#"{"prompt":"hi","target":null}"#).unwrap();
        assert_eq!(s, Sample::new("hi", ()));
    }
}
