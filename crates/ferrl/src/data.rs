//! Dataset plumbing: ingest typed [`Sample`]s from JSONL, and split them into a
//! train / held-out eval set.
//!
//! A dataset is just a `Vec<Sample<T>>` — there is no `Dataset` wrapper type to
//! learn; the vector *is* the dataset, and the standard slice/iterator API applies
//! directly. [`read_jsonl`] (and its in-memory sibling [`parse_jsonl`]) ingest one
//! [`Sample`] per JSONL line, and [`train_eval_split`] partitions a dataset into
//! disjoint train / eval halves, deduplicating first so no example can leak across
//! the split.
//!
//! The on-disk shape is **JSON Lines**: one JSON object per line, each the wire
//! form of a [`Sample<T>`] — `{"prompt": "...", "target": <T>}` (see [`Sample`]).
//! Blank lines are skipped; any other parse failure is reported with its 1-based
//! line number.

use std::collections::HashSet;
use std::fs;
use std::hash::Hash;
use std::path::{Path, PathBuf};

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use serde::de::DeserializeOwned;

use crate::sample::Sample;

/// An error reading or parsing a JSONL dataset.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DataError {
    /// The dataset file could not be read.
    #[error("failed to read JSONL dataset from {path}")]
    Io {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// A line could not be parsed into a [`Sample`].
    #[error("JSONL parse error at line {line}")]
    Parse {
        /// The 1-based line number that failed to parse.
        line: usize,
        /// The underlying deserialization error.
        #[source]
        source: serde_json::Error,
    },
}

/// Read a JSONL file into typed samples.
///
/// Each non-blank line is parsed as one [`Sample<T>`] (`{"prompt": ..., "target":
/// ...}`); blank lines are skipped. The whole file is read into memory — datasets
/// here are host-side prompt/target pairs, not token tensors, so they stay small.
///
/// # Errors
///
/// Returns [`DataError::Io`] if `path` cannot be read, or [`DataError::Parse`]
/// (carrying the 1-based line number) if any line is not a valid `Sample<T>`.
pub fn read_jsonl<T, P>(path: P) -> Result<Vec<Sample<T>>, DataError>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| DataError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_jsonl(&text)
}

/// Parse in-memory JSONL text into typed samples.
///
/// The same line discipline as [`read_jsonl`] (one [`Sample<T>`] per non-blank
/// line), but from a string already in memory — useful for embedded fixtures or a
/// dataset assembled at runtime.
///
/// ```
/// use ferrl::{data::parse_jsonl, Sample};
///
/// let data: Vec<Sample<i64>> =
///     parse_jsonl("{\"prompt\":\"a\",\"target\":1}\n{\"prompt\":\"b\",\"target\":2}").unwrap();
/// assert_eq!(data.len(), 2);
/// assert_eq!(data[1], Sample::new("b", 2));
/// ```
///
/// # Errors
///
/// Returns [`DataError::Parse`] (carrying the 1-based line number) if any non-blank
/// line is not a valid `Sample<T>`.
pub fn parse_jsonl<T: DeserializeOwned>(text: &str) -> Result<Vec<Sample<T>>, DataError> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let sample = serde_json::from_str(line).map_err(|source| DataError::Parse {
            line: i + 1,
            source,
        })?;
        out.push(sample);
    }
    Ok(out)
}

/// Partition a dataset into disjoint `(train, eval)` halves, holding out `eval_n`
/// samples as the eval set.
///
/// Generalizes the hand-rolled Countdown split: it first **deduplicates** the
/// input (keeping the first occurrence of each distinct sample), then shuffles
/// deterministically under `seed` and holds out the last `eval_n` of the shuffled,
/// deduplicated samples as eval — the rest are train. The two halves are a
/// partition of the *deduplicated* set, so **no identical sample appears in both**.
/// When `eval_n` exceeds the deduplicated size it is capped: every sample becomes
/// eval and train is empty.
///
/// **Scope of the dedup / no-leakage guarantee.** Dedup compares the **whole
/// sample** — prompt *and* target (hence the `Eq + Hash` bound on `T`; `Clone`
/// records seen samples while preserving the originals). So it removes exact
/// duplicate rows, and no *identical* row lands in both splits. It does **not**
/// dedup on the target alone, so it does **not** by itself guarantee a problem
/// never recurs across the split: a dataset that pairs the same `target` with two
/// different prompt phrasings keeps both, and the shuffle may place one in each
/// half. (Whole-sample dedup is the safe general default precisely because target
/// dedup would collapse a dataset of distinct prompts over a repeated or unit
/// target — e.g. `Sample<()>` — down to a single row.) For Countdown the prompt is
/// a pure function of the problem, so whole-sample dedup *is* problem-level dedup;
/// a task where it isn't should dedup on its target before calling this if it needs
/// a strict problem-level generalization gap.
///
/// Determinism: the same `(input, eval_n, seed)` always yields the same split, so a
/// run is reproducible and a resume re-derives the identical sets.
///
/// ```
/// use ferrl::{data::train_eval_split, Sample};
///
/// let data = vec![Sample::new("a", 1i64), Sample::new("b", 2), Sample::new("c", 3)];
/// let (train, eval) = train_eval_split(data, 1, 0);
/// assert_eq!(train.len(), 2);
/// assert_eq!(eval.len(), 1);
/// ```
#[must_use]
pub fn train_eval_split<T>(
    samples: Vec<Sample<T>>,
    eval_n: usize,
    seed: u64,
) -> (Vec<Sample<T>>, Vec<Sample<T>>)
where
    T: Clone + Eq + Hash,
{
    // Deduplicate, preserving first-seen order.
    let mut seen: HashSet<Sample<T>> = HashSet::with_capacity(samples.len());
    let mut unique: Vec<Sample<T>> = Vec::with_capacity(samples.len());
    for s in samples {
        if seen.insert(s.clone()) {
            unique.push(s);
        }
    }

    // Deterministic Fisher-Yates shuffle (the same algorithm as `countdown::shuffled`,
    // here over a seeded `Xoshiro256PlusPlus` from the `crate::sampler` RNG family),
    // so the held-out set is an unbiased sample rather than the input's tail.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    for i in (1..unique.len()).rev() {
        let j = (rng.random::<u64>() % (i as u64 + 1)) as usize;
        unique.swap(i, j);
    }

    // Hold out the last `eval_n` (capped at the deduplicated size); the rest train.
    let eval_n = eval_n.min(unique.len());
    let eval = unique.split_off(unique.len() - eval_n);
    (unique, eval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::countdown::CountdownProblem;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
    struct Problem {
        numbers: Vec<u32>,
        answer: i64,
    }

    #[test]
    fn parses_typed_samples_and_skips_blank_lines() {
        let text = concat!(
            "{\"prompt\":\"a\",\"target\":{\"numbers\":[1,2],\"answer\":3}}\n",
            "\n",
            "   \n",
            "{\"prompt\":\"b\",\"target\":{\"numbers\":[4],\"answer\":4}}\n",
        );
        let got: Vec<Sample<Problem>> = parse_jsonl(text).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(
            got[0],
            Sample::new(
                "a",
                Problem {
                    numbers: vec![1, 2],
                    answer: 3
                }
            )
        );
        assert_eq!(got[1].target.answer, 4);
    }

    #[test]
    fn parse_reports_the_one_based_line_number() {
        // Line 2 (1-based) is malformed; the blank line 1 is skipped but still counts.
        let text = "\n{not json}\n";
        let err = parse_jsonl::<Problem>(text).unwrap_err();
        assert!(
            matches!(err, DataError::Parse { line: 2, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn jsonl_round_trips_through_serialize() {
        let original = vec![
            Sample::new(
                "p0",
                Problem {
                    numbers: vec![1, 2, 3],
                    answer: 6,
                },
            ),
            Sample::new(
                "p1",
                Problem {
                    numbers: vec![10],
                    answer: 10,
                },
            ),
        ];
        let text: String = original
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let back: Vec<Sample<Problem>> = parse_jsonl(&text).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn reads_a_committed_jsonl_fixture_into_typed_countdown_samples() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/countdown_dataset.jsonl"
        );
        let got: Vec<Sample<CountdownProblem>> = read_jsonl(path).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(
            got[0].target,
            CountdownProblem {
                numbers: vec![3, 7, 11],
                target: 100
            }
        );
        // The prompt is carried verbatim; the typed target is the ground truth.
        assert!(got[0].prompt.contains("100"));
    }

    #[test]
    fn read_jsonl_reports_io_error_for_a_missing_file() {
        let err = read_jsonl::<CountdownProblem, _>("/no/such/dataset.jsonl").unwrap_err();
        assert!(matches!(err, DataError::Io { .. }), "got {err:?}");
    }

    /// Build `n` distinct unit-target samples `p0..p{n-1}`.
    fn samples(n: usize) -> Vec<Sample<()>> {
        (0..n).map(|i| Sample::new(format!("p{i}"), ())).collect()
    }

    #[test]
    fn split_holds_out_eval_n_with_no_overlap() {
        let (train, eval) = train_eval_split(samples(10), 3, 42);
        assert_eq!(train.len(), 7);
        assert_eq!(eval.len(), 3);
        let train_set: HashSet<_> = train.iter().collect();
        assert!(
            eval.iter().all(|s| !train_set.contains(s)),
            "no eval sample may also be in train"
        );
        // The union is exactly the input (every distinct sample is placed once).
        assert_eq!(train.len() + eval.len(), 10);
    }

    #[test]
    fn split_deduplicates_before_partitioning() {
        let mut data = samples(4);
        data.extend(samples(4)); // every sample duplicated
        let (train, eval) = train_eval_split(data, 1, 7);
        // 4 distinct samples survive dedup: 3 train + 1 eval.
        assert_eq!(train.len() + eval.len(), 4);
        assert_eq!(eval.len(), 1);
        let mut all: Vec<_> = train.iter().chain(eval.iter()).cloned().collect();
        all.sort_by(|a, b| a.prompt.cmp(&b.prompt));
        all.dedup();
        assert_eq!(all.len(), 4, "no duplicates may survive across the split");
    }

    #[test]
    fn split_is_deterministic_in_the_seed() {
        let a = train_eval_split(samples(20), 5, 1234);
        let b = train_eval_split(samples(20), 5, 1234);
        assert_eq!(a, b);
        // A different seed shuffles differently (membership identical, order/split not).
        let c = train_eval_split(samples(20), 5, 9999);
        assert_ne!(
            a.1, c.1,
            "a different seed should hold out a different eval set"
        );
    }

    #[test]
    fn split_caps_eval_n_at_the_dataset_size() {
        let (train, eval) = train_eval_split(samples(3), 10, 0);
        assert_eq!(eval.len(), 3);
        assert!(train.is_empty());
    }
}
