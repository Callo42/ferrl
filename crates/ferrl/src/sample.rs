//! The training datum: a prompt paired with its typed ground-truth target.
//!
//! A [`Sample<T>`] is what a `ferrl` run consumes — one per training/eval item.
//! The `target` carries whatever a verifier needs to score a completion (a parsed
//! problem, an expected answer, a test suite) in a **structured, typed** form,
//! replacing the fragile practice of smuggling the ground truth into the prompt
//! string and re-parsing it at scoring time.

/// A single training/eval datum: a `prompt` to roll out, paired with the typed
/// ground-truth `target` the reward scores the completion against.
///
/// `T` is the reward's [`crate::reward::RewardFn::Target`]. A run is monomorphic
/// in its target: [`crate::Trainer`] and [`crate::evaluate`] derive the sample
/// type from the reward, so there is no separate target knob to keep in sync.
///
/// ```
/// use ferrl::Sample;
///
/// // A typed target can be anything — here a tuple of (operands, expected sum).
/// let s = Sample::new("2 + 3 = ?", (vec![2u32, 3], 5u32));
/// assert_eq!(s.prompt, "2 + 3 = ?");
/// assert_eq!(s.target, (vec![2, 3], 5));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
