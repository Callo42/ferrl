//! Exact-match math: a second worked [`RewardFn`] beyond Countdown.
//!
//! Where [`crate::countdown`] *evaluates* a candidate expression against a target,
//! this task is the simplest verifiable shape there is: the model writes a final
//! answer and the reward checks it for an **exact match** against a typed expected
//! answer. It is the second reference task wired into the `ferrl` CLI and the model
//! for the "wire your own task" template — a concrete pattern for any short-answer
//! benchmark (GSM8K-style arithmetic, multiple choice, span extraction).
//!
//! The expected answer rides in the typed [`Sample::target`](crate::Sample) as a
//! [`MathProblem`] — never smuggled through the prompt string and re-parsed. A
//! dataset is a `Vec<Sample<MathProblem>>` and loads from JSONL via
//! [`read_jsonl`](crate::read_jsonl); each line is
//! `{"prompt": "...", "target": {"answer": "..."}}`.

use serde::{Deserialize, Serialize};

use crate::reward::{RewardError, RewardFn};
use crate::sample::Sample;

/// Open marker of the answer span the model is asked to emit.
const ANSWER_OPEN: &str = "<answer>";
/// Close marker of the answer span.
const ANSWER_CLOSE: &str = "</answer>";

/// The typed ground truth for an exact-match math item: the canonical expected
/// `answer` the completion is checked against (after [normalization](MathReward#normalization)).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MathProblem {
    /// The expected answer, compared to the model's extracted answer after both are
    /// normalized (trimmed; ASCII whitespace and `,` thousands separators dropped).
    pub answer: String,
}

impl MathProblem {
    /// Construct a problem from its expected `answer`.
    pub fn new(answer: impl Into<String>) -> Self {
        Self {
            answer: answer.into(),
        }
    }
}

/// Wrap a `question` into a prompt that instructs the model to put its final answer
/// inside an `<answer></answer>` span — the span [`MathReward`] reads back.
///
/// Datasets loaded from disk usually carry fully-built prompts already; this helper
/// is for building them in code (tests, the minimal-task template).
pub fn math_prompt(question: &str) -> String {
    format!("{question}\nPut your final answer inside <answer></answer> tags.\n")
}

/// Scores a completion by exact match against the expected [`MathProblem::answer`].
///
/// The model is asked (see [`math_prompt`]) to emit its final answer inside an
/// `<answer></answer>` span; the reward extracts that span, normalizes it and the
/// expected answer, and returns [`correct_reward`](Self::correct_reward) on an exact
/// match, else `0.0`. A completion with no closed span scores `0.0`.
///
/// # Normalization
///
/// Both sides are trimmed, and ASCII whitespace and `,` thousands separators are
/// removed, so `"1, 234"` matches `"1234"`. Comparison is otherwise exact
/// (case-sensitive): exact-match is the point — a task needing fuzzier grading
/// writes its own [`RewardFn`].
#[derive(Debug, Clone, Copy)]
pub struct MathReward {
    /// Reward for an exact match (default `1.0`); a miss always scores `0.0`.
    pub correct_reward: f32,
}

impl Default for MathReward {
    fn default() -> Self {
        Self {
            correct_reward: 1.0,
        }
    }
}

impl MathReward {
    /// Score one `completion` against the `expected` answer.
    ///
    /// The host-only core, kept free of the [`RewardFn`] plumbing so it is directly
    /// testable: returns [`correct_reward`](Self::correct_reward) on an exact
    /// (normalized) match, else `0.0`.
    #[must_use]
    pub fn score(&self, expected: &str, completion: &str) -> f32 {
        match extract_answer(completion) {
            Some(got) if normalize(got) == normalize(expected) => self.correct_reward,
            _ => 0.0,
        }
    }
}

impl RewardFn for MathReward {
    type Target = MathProblem;

    fn reward(&self, sample: &Sample<MathProblem>, completion: &str) -> Result<f32, RewardError> {
        Ok(self.score(&sample.target.answer, completion))
    }
    // No `reward_group` override: exact-match needs no per-group amortization, so the
    // default (map `reward` over the group) is ideal.
}

/// Extract the inner text of the **first** closed `<answer></answer>` span, trimmed;
/// `None` if there is no closed span.
///
/// The *first* span is the model's genuine attempt: a base model has no end-of-text
/// stop and tends to parrot the prompt's format after answering, so a later span is
/// usually a hallucinated repeat (the same rule [`crate::countdown`] applies).
fn extract_answer(text: &str) -> Option<&str> {
    let open = text.find(ANSWER_OPEN)?;
    let after = &text[open + ANSWER_OPEN.len()..];
    let close = after.find(ANSWER_CLOSE)?;
    Some(after[..close].trim())
}

/// Normalize an answer for comparison: drop ASCII whitespace and `,` separators.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_ascii_whitespace() && *c != ',')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scores_full_reward() {
        let r = MathReward::default();
        let s = Sample::new("2 + 3 = ?", MathProblem::new("5"));
        assert_eq!(r.reward(&s, "the sum is <answer>5</answer>").unwrap(), 1.0);
    }

    #[test]
    fn mismatch_scores_zero() {
        let r = MathReward::default();
        let s = Sample::new("2 + 3 = ?", MathProblem::new("5"));
        assert_eq!(r.reward(&s, "<answer>6</answer>").unwrap(), 0.0);
    }

    #[test]
    fn no_closed_span_scores_zero() {
        let r = MathReward::default();
        // An open tag with no close, and a bare answer with no tags at all.
        assert_eq!(r.score("5", "the answer is <answer>5"), 0.0);
        assert_eq!(r.score("5", "the answer is 5"), 0.0);
    }

    #[test]
    fn normalization_ignores_whitespace_and_thousands_separators() {
        let r = MathReward::default();
        assert_eq!(r.score("1234", "<answer>1, 234</answer>"), 1.0);
        assert_eq!(r.score("1234", "<answer> 1234 </answer>"), 1.0);
    }

    #[test]
    fn first_span_is_the_scored_one() {
        let r = MathReward::default();
        // The genuine first answer is wrong; a later parroted span is right. The
        // first span (wrong) is what scores — guarding against rewarding a repeat.
        assert_eq!(
            r.score("5", "<answer>4</answer> ... <answer>5</answer>"),
            0.0
        );
    }

    #[test]
    fn custom_correct_reward_is_honored() {
        let r = MathReward {
            correct_reward: 2.5,
        };
        assert_eq!(r.score("42", "<answer>42</answer>"), 2.5);
    }

    #[test]
    fn math_prompt_embeds_the_question_and_the_answer_instruction() {
        let p = math_prompt("What is 6 * 7?");
        assert!(p.contains("What is 6 * 7?"));
        assert!(p.contains("<answer></answer>"));
    }

    #[test]
    fn target_round_trips_through_json() {
        let s = Sample::new("q", MathProblem::new("99"));
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#"{"prompt":"q","target":{"answer":"99"}}"#);
        let back: Sample<MathProblem> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn loads_the_math_fixture_dataset() {
        // The committed demo dataset parses into typed samples via the PR-③ loader,
        // and a model that answered each one correctly would score full reward.
        let samples =
            crate::data::read_jsonl::<MathProblem, _>("tests/fixtures/math_dataset.jsonl").unwrap();
        assert_eq!(samples.len(), 4);
        let r = MathReward::default();
        let s = &samples[0];
        let completion = format!("<answer>{}</answer>", s.target.answer);
        assert_eq!(r.reward(s, &completion).unwrap(), 1.0);
    }
}
