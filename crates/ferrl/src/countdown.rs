//! The Countdown task — a verifiable arithmetic reward for GRPO.
//!
//! Countdown is the P4 task: given a small multiset of `numbers` and a `target`,
//! write an arithmetic expression that uses **each number exactly once** with the
//! operators `+ - * /` (and parentheses) and evaluates to the target. It is an
//! exact, deterministic, verifiable reward — the kind GRPO is built for — and is
//! known to elicit clear learning on sub-1B bases (the `TinyZero` / `Jiayi-Pan`
//! Countdown line of work).
//!
//! This module owns three pieces, all pure CPU logic so they are gated in CI:
//!
//! - **the problem + generator** ([`CountdownProblem`], [`generate_dataset`]) —
//!   problems are generated *solvable by construction* (the target is produced by
//!   folding the numbers under constrained ops, so at least one solution always
//!   exists), using a self-contained `SplitMix64` PRNG so a `(seed, config)` pair
//!   is byte-stable across `rand` releases;
//! - **the prompt** ([`build_prompt`]) — a few-shot, base-model-friendly prompt
//!   that ends right before the model's turn and presents the problem in a
//!   `Numbers:` / `Target:` block for the model to read;
//! - **the reward** ([`CountdownReward`]) — a shaped [`RewardFn`]: it reads the
//!   model's `` `<answer>…</answer>` `` expression and scores it in tiers, so a
//!   GRPO group has reward *spread* even before any completion is fully correct.
//!
//! # Reward shape
//!
//! Rewards are additive tiers (default weights in parentheses), so a group of
//! completions spans a useful range even early in training:
//!
//! | outcome | reward |
//! |---|---|
//! | no closed `` `<answer>…</answer>` `` tag | `0.0` |
//! | a tag, but the expression does not parse | `format` (`0.1`) |
//! | parses, but illegal numbers (not each given number exactly once) | `format` (`0.1`) |
//! | parses, legal numbers, wrong value | `format + legal` (`0.2`) |
//! | parses, legal numbers, equals the target | `format + legal + correct` (`1.0`) |
//!
//! Correctness requires *legal numbers* — hitting the target with the wrong
//! multiset is not a Countdown solution.
//!
//! # Where the run lives
//!
//! The real GPU training run that drives [`crate::QwenPolicy`] over this reward —
//! the P4 gate (reward rises **and** the trained adapter beats base on a held-out
//! Countdown eval, via [`crate::evaluate`]) — lives in
//! `examples/countdown_grpo.rs`, outside the coverage-gated library.

use crate::reward::{RewardError, RewardFn};
use crate::sample::Sample;

/// The `<answer>` open tag the reward parses.
const ANSWER_OPEN: &str = "<answer>";
/// The `</answer>` close tag the reward parses.
const ANSWER_CLOSE: &str = "</answer>";
/// The prompt key carrying the problem's numbers.
const NUMBERS_KEY: &str = "Numbers:";
/// The prompt key carrying the problem's target.
const TARGET_KEY: &str = "Target:";
/// Bounded attempts to land a generated target inside `max_target` before
/// accepting whatever the last (always solvable) fold produced.
const MAX_GEN_ATTEMPTS: usize = 64;

/// A Countdown problem: reach `target` from `numbers`, using each number exactly
/// once with `+ - * /` and parentheses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CountdownProblem {
    /// The numbers available to the expression (each must be used exactly once).
    pub numbers: Vec<u32>,
    /// The integer the expression must evaluate to.
    pub target: i64,
}

impl CountdownProblem {
    /// Whether `expression` is a valid Countdown solution: it parses, uses each of
    /// [`numbers`](Self::numbers) exactly once (and nothing else), and evaluates
    /// exactly to [`target`](Self::target).
    #[must_use]
    pub fn check(&self, expression: &str) -> bool {
        match eval_expression(expression) {
            Some((value, leaves)) => {
                is_legal(&leaves, &self.numbers) && value.equals_int(self.target)
            }
            None => false,
        }
    }
}

/// Knobs for [`generate_dataset`].
#[derive(Debug, Clone, Copy)]
pub struct CountdownConfig {
    /// How many numbers each problem has.
    pub num_count: usize,
    /// Smallest number that can appear (inclusive).
    pub min_number: u32,
    /// Largest number that can appear (inclusive).
    pub max_number: u32,
    /// Preferred upper bound on the target; generation re-rolls a problem whose
    /// target exceeds it (up to a bounded number of attempts) to keep targets
    /// short to write.
    ///
    /// Numbers are expected to be modest (the defaults sit far inside `i64`): the
    /// generator folds them in `i64`, so `num_count * max_number` must stay well
    /// below `i64::MAX` for the "solvable by construction" guarantee to hold.
    pub max_target: u32,
}

impl Default for CountdownConfig {
    fn default() -> Self {
        Self {
            num_count: 3,
            min_number: 1,
            max_number: 20,
            max_target: 1000,
        }
    }
}

/// Generate `count` solvable Countdown problems from `seed` under `cfg`.
///
/// Deterministic: the same `(seed, count, cfg)` always yields the same problems
/// (the generator is a self-contained `SplitMix64`, independent of `rand`). Use
/// different seeds for disjoint train / held-out splits.
#[must_use]
pub fn generate_dataset(seed: u64, count: usize, cfg: &CountdownConfig) -> Vec<CountdownProblem> {
    let mut rng = SplitMix64::new(seed);
    (0..count)
        .map(|_| generate_problem(&mut rng, cfg))
        .collect()
}

/// Build the few-shot prompt for `problem`.
///
/// The prompt ends right after the problem's `Target:` line (no open tag), so the
/// model emits its own `` `<answer>…</answer>` ``. The two worked examples teach a
/// base model the format; the trailing `Numbers:` / `Target:` block presents the
/// problem to the model. (The reward scores against the typed
/// [`Sample::target`](crate::Sample::target), not by re-parsing this text.)
#[must_use]
pub fn build_prompt(problem: &CountdownProblem) -> String {
    let nums = problem
        .numbers
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let target = problem.target;
    format!("{INSTRUCTIONS}\n\n{FEWSHOT}\n{NUMBERS_KEY} {nums}\n{TARGET_KEY} {target}\n")
}

/// The instruction preamble shared by every prompt.
const INSTRUCTIONS: &str = "Solve the countdown puzzle. Use each number exactly once with the \
operators + - * / and parentheses to write an arithmetic expression that equals the target. Give \
only the final expression between <answer> and </answer> tags.";

/// Two fixed worked examples (correct solutions) demonstrating the format.
const FEWSHOT: &str = "Numbers: 2, 3, 4\nTarget: 14\n<answer>2 * (3 + 4)</answer>\n\n\
Numbers: 5, 5, 2\nTarget: 12\n<answer>5 + 5 + 2</answer>\n";

/// A shaped, verifiable reward for the Countdown task.
///
/// `CountdownReward` is stateless — it scores any completion against the typed
/// [`Sample::target`](crate::Sample::target) (`CountdownProblem`) it is given,
/// not by re-parsing the prompt. See the [module docs](self) for the reward tiers.
#[derive(Debug, Clone, Copy)]
pub struct CountdownReward {
    /// Reward for emitting a closed `` `<answer>…</answer>` `` tag at all.
    pub format_reward: f32,
    /// Additional reward for an expression that uses each given number exactly
    /// once (a legal Countdown attempt), regardless of its value.
    pub legal_reward: f32,
    /// Additional reward for a legal expression that evaluates to the target.
    pub correct_reward: f32,
}

impl Default for CountdownReward {
    fn default() -> Self {
        Self {
            format_reward: 0.1,
            legal_reward: 0.1,
            correct_reward: 0.8,
        }
    }
}

impl CountdownReward {
    /// Score one completion against an already-parsed problem.
    fn score(&self, problem: &CountdownProblem, completion: &str) -> f32 {
        let Some(answer) = extract_answer(completion) else {
            return 0.0;
        };
        let mut reward = self.format_reward;
        let Some((value, leaves)) = eval_expression(answer) else {
            return reward;
        };
        let legal = is_legal(&leaves, &problem.numbers);
        if legal {
            reward += self.legal_reward;
            if value.equals_int(problem.target) {
                reward += self.correct_reward;
            }
        }
        reward
    }
}

impl RewardFn for CountdownReward {
    type Target = CountdownProblem;

    fn reward(
        &self,
        sample: &Sample<CountdownProblem>,
        completion: &str,
    ) -> Result<f32, RewardError> {
        Ok(self.score(&sample.target, completion))
    }
    // No `reward_group` override: the default (map `reward` over the group) is
    // ideal now that the typed target needs no per-group parsing to amortize.
}

/// Extract the inner text of the **first** closed `` `<answer>…</answer>` `` tag in
/// `text`, trimmed. `None` if there is no such closed tag.
///
/// The *first* tag is the model's genuine attempt: a base model has no end-of-text
/// stop and tends to parrot the prompt's format after answering, so a *later* tag
/// is usually a hallucinated repeat that would be scored against the wrong problem.
fn extract_answer(text: &str) -> Option<&str> {
    let open = text.find(ANSWER_OPEN)?;
    let after = &text[open + ANSWER_OPEN.len()..];
    let close = after.find(ANSWER_CLOSE)?;
    Some(after[..close].trim())
}

/// Whether `leaves` (the literals an expression used) is exactly the multiset
/// `numbers` — each given number used exactly once and nothing else.
fn is_legal(leaves: &[u32], numbers: &[u32]) -> bool {
    if leaves.len() != numbers.len() {
        return false;
    }
    let mut a = leaves.to_vec();
    let mut b = numbers.to_vec();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

// ---------------------------------------------------------------------------
// Exact rational arithmetic
// ---------------------------------------------------------------------------

/// An exact rational `num / den`, kept reduced with `den > 0`. Expressions over a
/// handful of small integers never approach `i128` range, so exactness is free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ratio {
    num: i128,
    den: i128,
}

/// Greatest common divisor of `|a|` and `|b|` (`gcd(0, 0) == 0`).
///
/// Uses `unsigned_abs` so `i128::MIN` cannot overflow the negation (`(-2^127).abs()`
/// panics); the result divides a positive denominator, so it always fits back in `i128`.
fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a as i128
}

impl Ratio {
    /// The integer `n` as a rational.
    fn from_int(n: i64) -> Self {
        Self {
            num: i128::from(n),
            den: 1,
        }
    }

    /// Reduce `num / den` to lowest terms with `den > 0`. `None` on a zero
    /// denominator (division by zero).
    fn new(num: i128, den: i128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let sign = if den < 0 { -1 } else { 1 };
        let num = num.checked_mul(sign)?;
        let den = den.checked_mul(sign)?;
        let g = gcd(num, den).max(1);
        Some(Self {
            num: num / g,
            den: den / g,
        })
    }

    fn add(self, o: Self) -> Option<Self> {
        let num = self
            .num
            .checked_mul(o.den)?
            .checked_add(o.num.checked_mul(self.den)?)?;
        Self::new(num, self.den.checked_mul(o.den)?)
    }

    fn sub(self, o: Self) -> Option<Self> {
        let num = self
            .num
            .checked_mul(o.den)?
            .checked_sub(o.num.checked_mul(self.den)?)?;
        Self::new(num, self.den.checked_mul(o.den)?)
    }

    fn mul(self, o: Self) -> Option<Self> {
        Self::new(self.num.checked_mul(o.num)?, self.den.checked_mul(o.den)?)
    }

    fn div(self, o: Self) -> Option<Self> {
        // a/b / (c/d) = (a*d) / (b*c); a zero divisor makes the new denominator 0,
        // which `new` rejects.
        Self::new(self.num.checked_mul(o.den)?, self.den.checked_mul(o.num)?)
    }

    /// Whether this rational equals the integer `n` exactly.
    fn equals_int(self, n: i64) -> bool {
        self.den == 1 && self.num == i128::from(n)
    }
}

// ---------------------------------------------------------------------------
// Expression parser / evaluator
// ---------------------------------------------------------------------------

/// A lexical token of a Countdown expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tok {
    Num(u32),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

/// Lex a decimal literal starting at `i`; returns `(value, index_after)`. `None`
/// on `u32` overflow.
fn lex_number(bytes: &[u8], i: usize) -> Option<(u32, usize)> {
    let start = i;
    let mut j = i;
    while j < bytes.len() && bytes[j].is_ascii_digit() {
        j += 1;
    }
    let lit = std::str::from_utf8(&bytes[start..j]).ok()?;
    let value = lit.parse::<u32>().ok()?;
    Some((value, j))
}

/// Tokenize `s`. `None` on any character outside digits, `+ - * /`, parentheses,
/// and ASCII whitespace, or on a literal that overflows `u32`.
fn tokenize(s: &str) -> Option<Vec<Tok>> {
    let bytes = s.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_digit() {
            let (value, next) = lex_number(bytes, i)?;
            toks.push(Tok::Num(value));
            i = next;
            continue;
        }
        let tok = match c {
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'*' => Tok::Star,
            b'/' => Tok::Slash,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b' ' | b'\t' | b'\n' | b'\r' => {
                i += 1;
                continue;
            }
            _ => return None,
        };
        toks.push(tok);
        i += 1;
    }
    Some(toks)
}

/// A recursive-descent cursor over a token slice.
struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<Tok> {
        self.toks.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.peek();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// `expr := term (('+' | '-') term)*`
    fn expr(&mut self, leaves: &mut Vec<u32>) -> Option<Ratio> {
        let mut acc = self.term(leaves)?;
        while let Some(op) = self.peek() {
            match op {
                Tok::Plus => {
                    self.pos += 1;
                    acc = acc.add(self.term(leaves)?)?;
                }
                Tok::Minus => {
                    self.pos += 1;
                    acc = acc.sub(self.term(leaves)?)?;
                }
                _ => break,
            }
        }
        Some(acc)
    }

    /// `term := factor (('*' | '/') factor)*`
    fn term(&mut self, leaves: &mut Vec<u32>) -> Option<Ratio> {
        let mut acc = self.factor(leaves)?;
        while let Some(op) = self.peek() {
            match op {
                Tok::Star => {
                    self.pos += 1;
                    acc = acc.mul(self.factor(leaves)?)?;
                }
                Tok::Slash => {
                    self.pos += 1;
                    acc = acc.div(self.factor(leaves)?)?;
                }
                _ => break,
            }
        }
        Some(acc)
    }

    /// `factor := number | '(' expr ')'`
    fn factor(&mut self, leaves: &mut Vec<u32>) -> Option<Ratio> {
        match self.bump()? {
            Tok::Num(n) => {
                leaves.push(n);
                Some(Ratio::from_int(i64::from(n)))
            }
            Tok::LParen => {
                let v = self.expr(leaves)?;
                match self.bump()? {
                    Tok::RParen => Some(v),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Parse and evaluate `s` exactly, also collecting the integer literals it used.
/// `None` if `s` is not a fully-consumed, well-formed expression (bad char, syntax
/// error, trailing tokens) or if a division by zero occurs.
fn eval_expression(s: &str) -> Option<(Ratio, Vec<u32>)> {
    let toks = tokenize(s)?;
    if toks.is_empty() {
        return None;
    }
    let mut parser = Parser {
        toks: &toks,
        pos: 0,
    };
    let mut leaves = Vec::new();
    let value = parser.expr(&mut leaves)?;
    if parser.pos != toks.len() {
        return None; // trailing, unparsed tokens
    }
    Some((value, leaves))
}

// ---------------------------------------------------------------------------
// Problem generation (solvable by construction)
// ---------------------------------------------------------------------------

/// A self-contained `SplitMix64` PRNG (Steele, Lea & Flood 2014; public domain).
///
/// Used so a `(seed, config)` pair maps to a fixed problem set independent of any
/// external RNG's algorithm — the generated dataset is a reproducible oracle, and
/// must not shift when a dependency bumps its sampler. Not cryptographic; we only
/// need a well-distributed, deterministic stream.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform integer in the inclusive range bounded by `a` and `b` (order
    /// independent), so a misordered config cannot underflow.
    fn range(&mut self, a: u32, b: u32) -> u32 {
        let lo = a.min(b);
        let hi = a.max(b);
        let span = u64::from(hi - lo) + 1;
        lo + (self.next_u64() % span) as u32
    }
}

/// An arithmetic operator used by the generator's solvable fold.
#[derive(Debug, Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// A `Fisher-Yates` shuffle of `items` under `rng`.
fn shuffled<T: Copy>(rng: &mut SplitMix64, items: &[T]) -> Vec<T> {
    let mut v = items.to_vec();
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// Apply `op` to `acc` and `x` if the result stays a positive integer and does
/// not overflow `i64` (`+`/`*` only when exact, i.e. no overflow; `-` only when
/// `> 0`; `/` only when exact and non-zero divisor). Checked — never saturates —
/// so the fold stays *exact*, which is what keeps the generated target genuinely
/// reachable (a saturated `+`/`*` would clamp the target to a value no exact
/// expression can hit).
fn try_op(op: Op, acc: i64, x: i64) -> Option<i64> {
    match op {
        Op::Add => acc.checked_add(x),
        Op::Mul => acc.checked_mul(x),
        Op::Sub => (acc - x > 0).then(|| acc - x),
        Op::Div => (x != 0 && acc % x == 0).then(|| acc / x),
    }
}

/// Fold `x` into the running `acc` with a randomly chosen *valid* (exact,
/// positive) operator (see [`try_op`]). For the modest number ranges this
/// generator targets, addition never overflows, so a valid op always exists; the
/// fallback covers only a pathologically large [`CountdownConfig`].
fn fold_op(rng: &mut SplitMix64, acc: i64, x: i64) -> i64 {
    for op in shuffled(rng, &[Op::Add, Op::Sub, Op::Mul, Op::Div]) {
        if let Some(v) = try_op(op, acc, x) {
            return v;
        }
    }
    acc.saturating_add(x) // unreachable for supported configs (see CountdownConfig)
}

/// Build one solvable problem: pick the numbers, then fold them (in a random
/// order, under [`fold_op`]) into a reachable positive-integer target.
fn build_one(rng: &mut SplitMix64, cfg: &CountdownConfig) -> CountdownProblem {
    let count = cfg.num_count.max(1);
    let numbers: Vec<u32> = (0..count)
        .map(|_| rng.range(cfg.min_number, cfg.max_number))
        .collect();
    let order = shuffled(rng, &numbers);
    let mut acc = i64::from(order[0]);
    for &x in &order[1..] {
        acc = fold_op(rng, acc, i64::from(x));
    }
    CountdownProblem {
        numbers,
        target: acc,
    }
}

/// Generate one problem, preferring a target within `cfg.max_target` but always
/// returning a solvable problem (the final fold is accepted regardless).
fn generate_problem(rng: &mut SplitMix64, cfg: &CountdownConfig) -> CountdownProblem {
    let max_target = i64::from(cfg.max_target);
    for _ in 0..MAX_GEN_ATTEMPTS {
        let problem = build_one(rng, cfg);
        if (1..=max_target).contains(&problem.target) {
            return problem;
        }
    }
    build_one(rng, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- rational arithmetic ---

    #[test]
    fn ratio_reduces_and_compares() {
        let half = Ratio::new(2, 4).unwrap();
        assert_eq!(half, Ratio::new(1, 2).unwrap());
        // Negative denominator normalizes the sign onto the numerator.
        let neg = Ratio::new(1, -2).unwrap();
        assert_eq!(neg, Ratio { num: -1, den: 2 });
        assert!(Ratio::from_int(5).equals_int(5));
        assert!(!half.equals_int(0));
    }

    #[test]
    fn ratio_ops_are_exact() {
        let a = Ratio::from_int(1);
        let b = Ratio::from_int(3);
        let third = a.div(b).unwrap();
        // 1/3 + 1/3 + 1/3 == 1 exactly (no float drift).
        let sum = third.add(third).unwrap().add(third).unwrap();
        assert!(sum.equals_int(1));
        assert!(a.sub(b).unwrap().equals_int(-2));
        assert!(b.mul(Ratio::from_int(4)).unwrap().equals_int(12));
    }

    #[test]
    fn ratio_div_by_zero_is_none() {
        assert!(Ratio::from_int(1).div(Ratio::from_int(0)).is_none());
        assert!(Ratio::new(1, 0).is_none());
    }

    #[test]
    fn gcd_basics() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(-12, 18), 6);
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(7, 0), 7);
    }

    // --- expression evaluation ---

    fn eval(s: &str) -> Option<(Ratio, Vec<u32>)> {
        eval_expression(s)
    }

    #[test]
    fn precedence_and_parens() {
        assert!(eval("2 + 3 * 4").unwrap().0.equals_int(14));
        assert!(eval("(2 + 3) * 4").unwrap().0.equals_int(20));
        assert!(eval("2 * (3 + 4)").unwrap().0.equals_int(14));
        assert!(eval("10 - 2 - 3").unwrap().0.equals_int(5)); // left-assoc
    }

    #[test]
    fn division_is_exact_rational() {
        // Fractional intermediates are fine as long as the final value matches.
        assert!(eval("6 / 4 * 2").unwrap().0.equals_int(3));
        assert!(!eval("1 / 3").unwrap().0.equals_int(0));
        assert!(eval("8 / 2 / 2").unwrap().0.equals_int(2));
        assert!(eval("5 / 0").is_none());
    }

    #[test]
    fn collects_leaves_in_order() {
        let (_, leaves) = eval("2 * (3 + 4)").unwrap();
        assert_eq!(leaves, vec![2, 3, 4]);
    }

    #[test]
    fn rejects_malformed_expressions() {
        let bad = [
            "",            // empty
            "   ",         // whitespace only
            "2 +",         // trailing operator
            "2 3",         // two numbers, no operator
            "(2 + 3",      // unbalanced
            "2 + 3)",      // unbalanced
            "2 ^ 3",       // illegal char
            "2 + a",       // illegal char
            "99999999999", // u32 overflow
        ];
        for s in bad {
            assert!(eval(s).is_none(), "expected None for {s:?}");
        }
    }

    // --- answer extraction ---

    #[test]
    fn extracts_first_closed_answer() {
        assert_eq!(
            extract_answer("noise <answer> 1 + 2 </answer>"),
            Some("1 + 2")
        );
        // The model's FIRST answer wins; a base model with no EOS parrots the
        // prompt format after answering, so a later tag is a hallucinated repeat.
        assert_eq!(
            extract_answer("<answer>first</answer> then <answer>second</answer>"),
            Some("first")
        );
    }

    #[test]
    fn no_closed_tag_is_none() {
        assert_eq!(extract_answer("no tags here"), None);
        assert_eq!(extract_answer("<answer>unclosed forever"), None);
    }

    #[test]
    fn scores_the_models_answer_not_its_parroted_repeat() {
        // Realistic no-EOS completion: the model answers, then parrots a fresh
        // problem + answer. The reward must score the FIRST answer against the
        // sample's typed target, ignoring the parroted block.
        let r = CountdownReward::default();
        let s = sample_for(&[2, 3, 4], 14);
        let completion =
            "<answer>2 * (3 + 4)</answer>\n\nNumbers: 1, 1, 1\nTarget: 3\n<answer>1 + 1 + 1</answer>";
        assert!((r.reward(&s, completion).unwrap() - 1.0).abs() < 1e-6);
    }

    // --- legality ---

    #[test]
    fn legal_is_multiset_equality() {
        assert!(is_legal(&[2, 3, 4], &[4, 2, 3]));
        assert!(is_legal(&[5, 5, 2], &[5, 2, 5]));
        assert!(!is_legal(&[2, 3], &[2, 3, 4])); // too few
        assert!(!is_legal(&[2, 3, 3], &[2, 3, 4])); // wrong multiset
    }

    // --- problem checking ---

    #[test]
    fn check_accepts_correct_rejects_otherwise() {
        let p = CountdownProblem {
            numbers: vec![2, 3, 4],
            target: 14,
        };
        assert!(p.check("2 * (3 + 4)"));
        assert!(!p.check("2 + 3 + 4")); // legal numbers, wrong value (9)
        assert!(!p.check("2 * 7")); // right value, illegal numbers
        assert!(!p.check("2 * (3 + 4")); // unparseable
        assert!(!p.check("2 * 3 * 4 * 1")); // extra number
    }

    // --- prompt ---

    #[test]
    fn prompt_embeds_the_problem() {
        let p = CountdownProblem {
            numbers: vec![6, 3, 7],
            target: 16,
        };
        let prompt = build_prompt(&p);
        assert!(prompt.contains("Numbers: 6, 3, 7"));
        assert!(prompt.contains("Target: 16"));
    }

    // --- reward tiers ---

    /// A sample whose prompt is built from the problem and whose typed target is
    /// that same problem — the reward scores against the target, not the prompt.
    fn sample_for(numbers: &[u32], target: i64) -> Sample<CountdownProblem> {
        let problem = CountdownProblem {
            numbers: numbers.to_vec(),
            target,
        };
        Sample::new(build_prompt(&problem), problem)
    }

    #[test]
    fn reward_tiers_span_the_range() {
        let r = CountdownReward::default();
        let s = sample_for(&[2, 3, 4], 14);
        // No tag -> 0.0.
        assert_eq!(r.reward(&s, "I think it is 2 * 7").unwrap(), 0.0);
        // Tag, unparseable expression -> format only.
        assert_eq!(r.reward(&s, "<answer>2 * </answer>").unwrap(), 0.1);
        // Tag, parses, illegal numbers -> format only.
        assert_eq!(r.reward(&s, "<answer>2 * 7</answer>").unwrap(), 0.1);
        // Tag, legal numbers, wrong value -> format + legal.
        assert!((r.reward(&s, "<answer>2 + 3 + 4</answer>").unwrap() - 0.2).abs() < 1e-6);
        // Tag, legal numbers, equals target -> full.
        assert!(
            (r.reward(&s, "reason... <answer>2 * (3 + 4)</answer>")
                .unwrap()
                - 1.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn reward_group_scores_each() {
        let r = CountdownReward::default();
        let s = sample_for(&[5, 5, 2], 12);
        let completions = vec![
            "<answer>5 + 5 + 2</answer>".to_string(), // correct -> 1.0
            "<answer>5 + 5</answer>".to_string(),     // illegal (too few) -> 0.1
            "no answer".to_string(),                  // -> 0.0
        ];
        let got = r.reward_group(&s, &completions).unwrap();
        assert_eq!(got.len(), 3);
        assert!((got[0] - 1.0).abs() < 1e-6);
        assert!((got[1] - 0.1).abs() < 1e-6);
        assert_eq!(got[2], 0.0);
    }

    #[test]
    fn reward_weights_are_configurable() {
        let r = CountdownReward {
            format_reward: 0.05,
            legal_reward: 0.25,
            correct_reward: 0.7,
        };
        let s = sample_for(&[2, 3, 4], 14);
        assert!((r.reward(&s, "<answer>2 + 3 + 4</answer>").unwrap() - 0.30).abs() < 1e-6);
        assert!((r.reward(&s, "<answer>2 * (3 + 4)</answer>").unwrap() - 1.0).abs() < 1e-6);
    }

    // --- generator ---

    /// Independent solver: the set of values reachable by combining the whole
    /// multiset down to one number with `+ - * /` (exact rationals). Confirms the
    /// generator's targets really are solvable, without trusting its own fold.
    fn reachable(nums: &[Ratio]) -> Vec<Ratio> {
        if nums.len() == 1 {
            return vec![nums[0]];
        }
        let mut out = Vec::new();
        for i in 0..nums.len() {
            for j in 0..nums.len() {
                if i == j {
                    continue;
                }
                collect_pair(nums, i, j, &mut out);
            }
        }
        out
    }

    fn collect_pair(nums: &[Ratio], i: usize, j: usize, out: &mut Vec<Ratio>) {
        let (a, b) = (nums[i], nums[j]);
        let rest: Vec<Ratio> = nums
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != i && *k != j)
            .map(|(_, v)| *v)
            .collect();
        for c in [a.add(b), a.sub(b), a.mul(b), a.div(b)]
            .into_iter()
            .flatten()
        {
            let mut next = rest.clone();
            next.push(c);
            out.extend(reachable(&next));
        }
    }

    fn is_solvable(p: &CountdownProblem) -> bool {
        let nums: Vec<Ratio> = p
            .numbers
            .iter()
            .map(|&n| Ratio::from_int(i64::from(n)))
            .collect();
        reachable(&nums).iter().any(|r| r.equals_int(p.target))
    }

    #[test]
    fn generator_is_deterministic() {
        let cfg = CountdownConfig::default();
        assert_eq!(generate_dataset(42, 8, &cfg), generate_dataset(42, 8, &cfg));
        // A different seed gives a different stream.
        assert_ne!(generate_dataset(42, 8, &cfg), generate_dataset(43, 8, &cfg));
    }

    #[test]
    fn generated_problems_are_solvable_and_in_range() {
        let cfg = CountdownConfig {
            num_count: 4,
            min_number: 1,
            max_number: 12,
            max_target: 200,
        };
        let data = generate_dataset(7, 40, &cfg);
        assert_eq!(data.len(), 40);
        for p in &data {
            assert_eq!(p.numbers.len(), 4);
            assert!(p.numbers.iter().all(|&n| (1..=12).contains(&n)));
            assert!(p.target > 0);
            assert!(is_solvable(p), "unsolvable problem generated: {p:?}");
        }
    }

    #[test]
    fn generator_stays_solvable_with_larger_numbers() {
        // Exercises try_op's CHECKED arithmetic over a wider range (bigger products):
        // a saturated fold would clamp the target to an unreachable value, which the
        // independent solver would catch as unsolvable.
        let cfg = CountdownConfig {
            num_count: 4,
            min_number: 10,
            max_number: 80,
            max_target: 100_000,
        };
        for p in &generate_dataset(99, 30, &cfg) {
            assert!(p.target > 0);
            assert!(is_solvable(p), "unsolvable problem generated: {p:?}");
        }
    }

    #[test]
    fn generator_handles_degenerate_config() {
        // num_count 1, a single fixed value: the only target is the number itself.
        let cfg = CountdownConfig {
            num_count: 1,
            min_number: 5,
            max_number: 5,
            max_target: 1000,
        };
        let data = generate_dataset(1, 3, &cfg);
        for p in &data {
            assert_eq!(p.numbers, vec![5]);
            assert_eq!(p.target, 5);
        }
    }
}
