//! TriMul kernel-discovery task — ferrl's first *discovery* [`RewardFn`].
//!
//! The policy is asked to write a faster GPU kernel for the **Triangle
//! Multiplicative Update** (the GPUMODE `bioml/trimul` task — a core AlphaFold-family
//! operator). Each completion is expected to contain a Python `custom_kernel`
//! implementation; this reward **runs it** and scores it on **correctness** and
//! **speed**.
//!
//! ## Flow (per candidate)
//!
//! 1. Extract the `custom_kernel` source from the completion according to the
//!    configured [`SubmissionExtractMode`] — the final fenced Python code block, or
//!    for thinking prompts, the final fenced block after `</think>`.
//! 2. Stage a node-local scratch dir: the candidate as `submission.py`, plus a
//!    generated test-spec and benchmark-spec file ([`render_spec`]).
//! 3. Run the eval in the sandbox ([`crate::sandbox::ApptainerSandbox`]): the pinned
//!    GPUMODE eval files are bound **read-only**, the scratch **read-write**, the GPU
//!    exposed (`--nv`), and the **network denied**. Inside, the GPUMODE `eval.py`
//!    runs `test` (correctness) then `benchmark` (variance-aware CUDA-event timing).
//!    Its grade is written to `POPCORN_FD`, which we route to the **captured stdout
//!    pipe** while the candidate's own stdout goes to `/dev/null` — so the grade rides
//!    a channel the untrusted candidate cannot reach (its worker neither inherits the
//!    fd nor can target it by path), foreclosing a forged pass.
//! 4. Map the captured grade to a reward: **`0` if the candidate is missing, crashes,
//!    fails any correctness case, or reports an implausibly fast time (below the
//!    kernel-launch floor — a glitch or forged grade); otherwise the geometric-mean
//!    speedup over a reference baseline.** GRPO normalizes rewards within a group, so a
//!    monotone-in-speed signal is what the search needs.
//!
//! ## What lives where
//!
//! This module is ferrl's own code: the case type, the spec rendering, the result
//! parsing, and the reward. The **GPUMODE task materials** (`reference.py`,
//! `eval.py`, `utils.py`, `task.py`, and the concrete case list in `task.yml`) are
//! **not** vendored here — they carry GPU Mode's Researcher Reciprocity License and
//! live only in the pinned eval bundle on the cluster (bound in at run time). The
//! case list is therefore *configuration* ([`TrimulReward::with_cases`]); the tests
//! here use generic, made-up sizes.
//!
//! ## Reward integrity
//!
//! The grade rides a channel the candidate cannot reach by file or by print, and an
//! implausibly fast time is rejected — so the trivial gaming routes (forge a `/work`
//! result file, print a fake pass, report a 0 ns kernel) all score zero, gated by the
//! negative-control suite. **Known residual (PoC):** a candidate that scans `/proc` for
//! the grader's grade fd *and* reports a physically plausible fake time could still
//! forge a pass — its worker shares the grader's PID namespace and uid, so only
//! per-candidate PID-namespace isolation closes it (earned when untrusted external
//! submissions arrive, not this PoC, whose kernel-writing policy is extremely unlikely
//! to emit such an exploit). The held-out `POPCORN_SEED` is likewise candidate-readable;
//! both are moot against an attacker who can already forge the grade and close together
//! with that isolation. The dynamic guard — watching the discovery run for implausible
//! wins and re-verifying top candidates — is the spec's Phase-1 instrumentation, done
//! in the run, not the reward.
//!
//! ## Testing split (as in [`crate::sandbox`])
//!
//! The pure pieces — submission extraction, spec rendering, result parsing, the
//! run-spec builder, and the reward math — are unit-tested in CI. The real GPU eval
//! is a `gate`-feature integration test (`tests/trimul_gate.rs`), run on an `sm_80`
//! node against the eval image; like the GPU tests it is never compiled in CI.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::reward::{RewardError, RewardFn, RewardOutcome};
use crate::sample::Sample;
use crate::sandbox::{ApptainerSandbox, Bind, ResourceLimits, RunSpec, RunStatus, Sandbox};

/// The input distribution for a TriMul case (mirrors the GPUMODE task's `distribution`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    /// Standard normal inputs.
    Normal,
    /// Heavy-tailed Cauchy inputs.
    Cauchy,
}

impl Distribution {
    /// The token the GPUMODE input generator expects.
    fn as_str(self) -> &'static str {
        match self {
            Distribution::Normal => "normal",
            Distribution::Cauchy => "cauchy",
        }
    }
}

/// One TriMul problem-size case — the columns the GPUMODE harness reads from a spec
/// line. The concrete case list is GPU Mode's (loaded from the pinned `task.yml`);
/// these fields are ferrl's neutral description of a case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimulCase {
    /// Sequence length `N` (the operator is over an `N×N` pair grid).
    pub seqlen: u32,
    /// Batch size.
    pub bs: u32,
    /// Channel dimension of the input/output.
    pub dim: u32,
    /// Hidden dimension of the projections.
    pub hiddendim: u32,
    /// Input-generation seed (public; the harness combines it with the secret seed).
    pub seed: u64,
    /// Whether the mask is all-ones (`true`) or random binary (`false`).
    pub nomask: bool,
    /// Input distribution.
    pub distribution: Distribution,
}

impl TrimulCase {
    /// Render to the `key: value; …` spec line the GPUMODE `eval.py` parses. `nomask`
    /// is emitted as `1`/`0` (an integer) — the harness int-parses values, and a
    /// non-empty string like `False` would parse as truthy.
    fn render(&self) -> String {
        format!(
            "seqlen: {}; bs: {}; dim: {}; hiddendim: {}; seed: {}; nomask: {}; distribution: {}",
            self.seqlen,
            self.bs,
            self.dim,
            self.hiddendim,
            self.seed,
            u8::from(self.nomask),
            self.distribution.as_str(),
        )
    }
}

/// Render a list of cases into a spec-file body (one line per case).
#[must_use]
pub fn render_spec(cases: &[TrimulCase]) -> String {
    cases
        .iter()
        .map(TrimulCase::render)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Which completion region is eligible for TriMul submission extraction.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SubmissionExtractMode {
    /// Extract the final fenced code block from the whole completion.
    FinalFence,
    /// Require a closing `</think>` marker, then extract from the final-answer region.
    ThinkingAfterThink,
}

/// Extract the candidate `custom_kernel` source from a completion.
///
/// This is the raw/non-thinking extractor: the whole completion is the answer region.
/// The extracted candidate is the body of the final fenced code block in that region,
/// and the block must be the last non-whitespace content in the region. Returns
/// `None` if there is no closed, final, non-empty block.
#[must_use]
pub fn extract_submission(completion: &str) -> Option<String> {
    extract_submission_with_mode(completion, SubmissionExtractMode::FinalFence)
}

/// Extract a candidate according to the configured prompt/extraction contract.
///
/// `ThinkingAfterThink` fails closed when the completion never exits the thinking
/// region with `</think>`.
#[must_use]
pub fn extract_submission_with_mode(
    completion: &str,
    mode: SubmissionExtractMode,
) -> Option<String> {
    let answer = match mode {
        SubmissionExtractMode::FinalFence => completion,
        SubmissionExtractMode::ThinkingAfterThink => completion.rsplit_once("</think>")?.1,
    };
    extract_final_fenced_block(answer)
}

/// Extract the final closed fenced block from `answer`.
fn extract_final_fenced_block(answer: &str) -> Option<String> {
    let close = answer.rfind("```")?;
    let trailing = &answer[close + 3..];
    if !trailing.trim().is_empty() {
        return None;
    }

    let before_close = &answer[..close];
    let open = before_close.rfind("```")?;
    // Skip the optional language tag up to the end of the fence's opening line.
    let after_fence = &before_close[open + 3..];
    let body_start = after_fence.find('\n')? + 1;
    let code = after_fence[body_start..].trim_end();
    if code.trim().is_empty() {
        None
    } else {
        Some(code.to_string())
    }
}

/// The value of the first `key: value` line for `key` in a POPCORN result log.
fn log_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (k, v) = line.split_once(": ")?;
        (k.trim() == key).then_some(v.trim())
    })
}

/// Whether a `test`-mode result log reports overall `check: pass`.
#[must_use]
pub fn test_passed(test_log: &str) -> bool {
    log_value(test_log, "check") == Some("pass")
}

/// The per-case mean runtimes (nanoseconds) from a `benchmark`-mode result log: every
/// `benchmark.<i>.mean` value.
fn benchmark_means_ns(bench_log: &str) -> Vec<f64> {
    bench_log
        .lines()
        .filter_map(|line| {
            let (key, val) = line.split_once(": ")?;
            let key = key.trim();
            if key.starts_with("benchmark.") && key.ends_with(".mean") {
                val.trim().parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect()
}

/// The geometric mean of `xs`, or `None` if empty or any value is non-positive (the
/// GPUMODE leaderboard ranks by the geometric mean of per-case runtimes).
#[must_use]
pub fn geomean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|&x| x <= 0.0 || x.is_nan()) {
        return None;
    }
    let log_sum: f64 = xs.iter().map(|&x| x.ln()).sum();
    Some((log_sum / xs.len() as f64).exp())
}

/// An error loading or parsing the GPU Mode `task.yml` that carries the concrete
/// TriMul case list.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrimulError {
    /// The `task.yml` file could not be read.
    #[error("failed to read task.yml from {path}")]
    Io {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// The `task.yml` body could not be parsed into case lists.
    #[error("task.yml parse error: {0}")]
    Parse(String),
}

/// Which case section of `task.yml` we are reading.
#[derive(Debug, Clone, Copy)]
enum Section {
    /// The `tests:` (correctness) cases.
    Tests,
    /// The `benchmarks:` (timing) cases.
    Benchmarks,
}

/// Load the TriMul case lists from a GPU Mode `task.yml` at `path`, returning the
/// `(tests, benchmarks)` cases — the correctness set and the timing set.
///
/// The `task.yml` carries GPU Mode's Researcher Reciprocity License and is **not**
/// vendored into this repo; it is read at run time from the pinned eval bundle on the
/// cluster (the same place [`TrimulReward`]'s `eval_dir` points at). See the module docs.
///
/// # Errors
///
/// [`TrimulError::Io`] if `path` cannot be read, or [`TrimulError::Parse`] if the body
/// has no `tests`/`benchmarks` cases or a case line is malformed.
pub fn load_task_yml(
    path: impl AsRef<Path>,
) -> Result<(Vec<TrimulCase>, Vec<TrimulCase>), TrimulError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| TrimulError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_task_yml(&text)
}

/// Parse the `tests:` and `benchmarks:` case lists out of a GPU Mode `task.yml` body.
///
/// The format each section uses is a list of one-line flow mappings, e.g.
/// `- {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask":
/// True, "distribution": "normal"}`. Python-style `True`/`False` booleans are accepted
/// (the GPU Mode file uses them). Only the `tests`/`benchmarks` top-level sections are
/// read; every other section (`files`, `description`, …) is ignored.
///
/// # Errors
///
/// [`TrimulError::Parse`] if either section is empty or a case line is malformed.
pub fn parse_task_yml(text: &str) -> Result<(Vec<TrimulCase>, Vec<TrimulCase>), TrimulError> {
    let mut tests = Vec::new();
    let mut benches = Vec::new();
    let mut section: Option<Section> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // A non-indented line is a top-level key: it switches into (or ends) a section.
        if !line.starts_with(char::is_whitespace) {
            section = match trimmed {
                "tests:" => Some(Section::Tests),
                "benchmarks:" => Some(Section::Benchmarks),
                _ => None,
            };
            continue;
        }
        // Inside a case section, an indented `- { … }` line is one case; anything else
        // (or any line while not in a case section) is skipped.
        let Some(sec) = section else { continue };
        let item = trimmed.strip_prefix('-').map_or(trimmed, str::trim);
        if !item.starts_with('{') {
            continue;
        }
        let case = parse_case(item)?;
        match sec {
            Section::Tests => tests.push(case),
            Section::Benchmarks => benches.push(case),
        }
    }
    if tests.is_empty() {
        return Err(TrimulError::Parse("no `tests:` cases found".into()));
    }
    if benches.is_empty() {
        return Err(TrimulError::Parse("no `benchmarks:` cases found".into()));
    }
    Ok((tests, benches))
}

/// Parse one flow-mapping case line (the body between `{` and `}`) into a [`TrimulCase`].
/// Values may be quoted; the mapping has no nested commas, so a flat split on `,` is safe.
fn parse_case(mapping: &str) -> Result<TrimulCase, TrimulError> {
    let inner = mapping.trim().trim_start_matches('{').trim_end_matches('}');
    let mut fields: HashMap<String, String> = HashMap::new();
    for pair in inner.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair
            .split_once(':')
            .ok_or_else(|| TrimulError::Parse(format!("malformed field {pair:?}")))?;
        let key = k.trim().trim_matches(['"', '\'']).to_string();
        let val = v.trim().trim_matches(['"', '\'']).to_string();
        fields.insert(key, val);
    }
    let int_field = |name: &str| -> Result<u32, TrimulError> {
        let raw = fields
            .get(name)
            .ok_or_else(|| TrimulError::Parse(format!("missing field {name:?}")))?;
        raw.parse()
            .map_err(|_| TrimulError::Parse(format!("field {name:?} is not an integer: {raw:?}")))
    };
    let seed = {
        let raw = fields
            .get("seed")
            .ok_or_else(|| TrimulError::Parse("missing field \"seed\"".to_string()))?;
        raw.parse::<u64>()
            .map_err(|_| TrimulError::Parse(format!("field \"seed\" is not an integer: {raw:?}")))?
    };
    Ok(TrimulCase {
        seqlen: int_field("seqlen")?,
        bs: int_field("bs")?,
        dim: int_field("dim")?,
        hiddendim: int_field("hiddendim")?,
        seed,
        nomask: parse_bool(fields.get("nomask"))?,
        distribution: parse_distribution(fields.get("distribution"))?,
    })
}

/// Parse a case's `nomask` value, accepting Python (`True`/`False`), YAML
/// (`true`/`false`/`yes`/`no`), and integer (`1`/`0`) spellings.
fn parse_bool(raw: Option<&String>) -> Result<bool, TrimulError> {
    let raw = raw.ok_or_else(|| TrimulError::Parse("missing field \"nomask\"".to_string()))?;
    match raw.to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => Ok(true),
        "false" | "no" | "0" => Ok(false),
        other => Err(TrimulError::Parse(format!(
            "field \"nomask\" is not a boolean: {other:?}"
        ))),
    }
}

/// Parse a case's `distribution` value into a [`Distribution`].
fn parse_distribution(raw: Option<&String>) -> Result<Distribution, TrimulError> {
    let raw =
        raw.ok_or_else(|| TrimulError::Parse("missing field \"distribution\"".to_string()))?;
    match raw.to_ascii_lowercase().as_str() {
        "normal" => Ok(Distribution::Normal),
        "cauchy" => Ok(Distribution::Cauchy),
        other => Err(TrimulError::Parse(format!(
            "field \"distribution\" is not normal|cauchy: {other:?}"
        ))),
    }
}

/// The discovery task prompt: describe the tensor program and evaluator contract.
///
/// This is ferrl's own wording — the GPU Mode task description is **not** vendored. It
/// states the exact call contract the eval harness expects (the `(input, mask, weights,
/// config)` tuple). Chat-format wrappers add the output-format contract separately.
#[must_use]
pub fn build_prompt() -> String {
    // A self-contained task description; kept deliberately small and stable. Prompt
    // refinement for the discovery run is a later, separate concern.
    "Implement `custom_kernel(data)` for this tensor program.\n\
     \n\
     Define exactly one Python function with this signature:\n\
     \n\
     \x20   def custom_kernel(data):\n\
     \x20       ...\n\
     \x20       return output\n\
     \n\
     Input contract: `data` is a tuple `(input, mask, weights, config)`:\n\
     \x20 - input:   a float tensor of shape [batch, seq_len, seq_len, dim]\n\
     \x20 - mask:    a tensor of shape [batch, seq_len, seq_len]\n\
     \x20 - weights: a dict of the module's parameter tensors\n\
     \x20 - config:  a dict of configuration values\n\
     \n\
     Return a tensor of shape [batch, seq_len, seq_len, dim].\n\
     Produce numerically the same result as the baseline for every test case, then make\n\
     the implementation as fast as possible.\n\
     \n\
     Allowed weight keys:\n\
     \x20 - norm.weight\n\
     \x20 - norm.bias\n\
     \x20 - left_proj.weight\n\
     \x20 - right_proj.weight\n\
     \x20 - left_gate.weight\n\
     \x20 - right_gate.weight\n\
     \x20 - out_gate.weight\n\
     \x20 - to_out_norm.weight\n\
     \x20 - to_out_norm.bias\n\
     \x20 - to_out.weight\n\
     \n"
    .to_string()
}

/// Build ferrl's raw prompt: task description plus the extraction/output contract.
#[must_use]
pub fn build_raw_prompt(task: &str) -> String {
    format!(
        "{}\n\n{}",
        task.trim(),
        "Output contract:\n\
         - Output exactly one closed fenced Python code block.\n\
         - The code block must contain only the complete custom_kernel(data) implementation.\n\
         - Do not include prose, comments, docstrings, or Markdown outside the code block.\n\
         - Stop after the closing code fence."
    )
}

fn qwen3_5_thinking_system_prompt() -> &'static str {
    "You generate Python code for a strict evaluator.\n\
     Output contract:\n\
     - After finishing the reasoning, close </think>.\n\
     - Immediately after </think>, output exactly one closed fenced Python code block.\n\
     - The code block must contain only the complete custom_kernel(data) implementation.\n\
     - Do not include prose, comments, docstrings, or Markdown outside the final code block.\n\
     - Stop after the closing code fence."
}

/// Build the Qwen3.5 native thinking chat-template prompt for a TriMul instruction.
///
/// This mirrors the released Qwen3.5 `chat_template.jinja` for system + user messages with
/// `add_generation_prompt = true` and thinking mode (`enable_thinking = true`): the
/// assistant prefix opens `<think>` and lets the model generate its reasoning. The
/// system message carries the output/extraction contract.
#[must_use]
pub fn build_qwen3_5_chat_thinking_prompt(task: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        qwen3_5_thinking_system_prompt(),
        task.trim()
    )
}

/// The TriMul discovery reward: runs a candidate kernel in the sandboxed eval image
/// and scores it on correctness + speed. Construct with [`TrimulReward::new`].
#[derive(Debug, Clone)]
pub struct TrimulReward {
    /// The eval image (the pinned PyTorch+Triton `.sif`).
    image: PathBuf,
    /// The pinned GPUMODE eval bundle (`eval.py`/`reference.py`/`task.py`/`utils.py`),
    /// bound **read-only**.
    eval_dir: PathBuf,
    /// Where per-candidate scratch dirs are created — node-local tmpfs is preferred
    /// (e.g. `/dev/shm/ferrl`) so overflow cannot fill persistent host storage.
    scratch_root: PathBuf,
    /// Host-supervised total byte cap for the candidate-writable `/work` tree.
    scratch_max_bytes: u64,
    /// Correctness cases (GPU Mode's, loaded from the pinned `task.yml`).
    test_cases: Vec<TrimulCase>,
    /// Timing cases.
    benchmark_cases: Vec<TrimulCase>,
    /// The secret seed Cantor-combined with each case's public seed for held-out
    /// inputs (passed as `POPCORN_SEED`).
    secret_seed: u64,
    /// Reference geometric-mean runtime (ns) on the target GPU; the speedup
    /// denominator. `None` falls back to an inverse-time signal.
    baseline_ns: Option<f64>,
    /// Wall-clock budget for one candidate's full eval.
    wall: Duration,
    /// Floor (ns) on each benchmark mean: a real GPU kernel cannot run faster than the
    /// kernel-launch overhead, so a sub-floor time is a measurement glitch or a forged
    /// grade — the candidate scores zero. Defence-in-depth against absurd reward
    /// gaming, on top of the off-filesystem grade channel.
    min_plausible_ns: f64,
    /// Which completion region may contain the final submitted code block.
    submission_extract_mode: SubmissionExtractMode,
    /// The sandbox backend.
    sandbox: ApptainerSandbox,
}

/// The parsed result of one sandboxed TriMul eval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrimulVerification {
    /// Whether the eval reported `check: pass`.
    pub correct: bool,
    /// Per-benchmark mean runtimes in nanoseconds, after parsing the grade stream.
    pub benchmark_means_ns: Vec<f64>,
    /// Plausibility-checked geometric-mean benchmark time (ns), if any.
    pub geomean_ns: Option<f64>,
    /// Speedup over the configured baseline, when both baseline and timing are present.
    pub speedup: Option<f64>,
}

/// A `custom_kernel` that delegates to the bundled reference implementation. Used to
/// **measure the speedup baseline**: the reference is correct by definition, so it
/// passes correctness and its benchmark time *is* the reference runtime. The
/// `reference` module is copied next to the submission inside the image (see
/// [`TrimulReward::in_container_command`]). This is the extracted code, not a fenced
/// block — it is fed straight to the eval path, bypassing [`extract_submission`].
const REFERENCE_SUBMISSION: &str =
    "def custom_kernel(data):\n    from reference import ref_kernel\n    return ref_kernel(data)\n";

impl TrimulReward {
    /// Construct with the eval `image`, the pinned `eval_dir` bundle, and a
    /// node-local `scratch_root`. Cases default to empty — set them with
    /// [`with_cases`](Self::with_cases) (they are GPU Mode's, kept out of this repo).
    #[must_use]
    pub fn new(
        image: impl Into<PathBuf>,
        eval_dir: impl Into<PathBuf>,
        scratch_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            image: image.into(),
            eval_dir: eval_dir.into(),
            scratch_root: scratch_root.into(),
            scratch_max_bytes: 1 << 30,
            test_cases: Vec::new(),
            benchmark_cases: Vec::new(),
            secret_seed: 0,
            baseline_ns: None,
            wall: Duration::from_secs(600),
            min_plausible_ns: 1_000.0,
            submission_extract_mode: SubmissionExtractMode::FinalFence,
            sandbox: ApptainerSandbox::default(),
        }
    }

    /// Set the correctness and timing case lists.
    #[must_use]
    pub fn with_cases(
        mut self,
        test_cases: Vec<TrimulCase>,
        benchmark_cases: Vec<TrimulCase>,
    ) -> Self {
        self.test_cases = test_cases;
        self.benchmark_cases = benchmark_cases;
        self
    }

    /// Set the secret held-out seed (`POPCORN_SEED`).
    #[must_use]
    pub fn with_secret_seed(mut self, seed: u64) -> Self {
        self.secret_seed = seed;
        self
    }

    /// Set the reference baseline (geometric-mean ns) the speedup is measured against.
    #[must_use]
    pub fn with_baseline_ns(mut self, baseline_ns: f64) -> Self {
        self.baseline_ns = Some(baseline_ns);
        self
    }

    /// Set the per-candidate wall-clock budget.
    #[must_use]
    pub fn with_wall(mut self, wall: Duration) -> Self {
        self.wall = wall;
        self
    }

    /// Set the total byte cap for the candidate-writable `/work` tree.
    #[must_use]
    pub fn with_scratch_max_bytes(mut self, bytes: u64) -> Self {
        self.scratch_max_bytes = bytes;
        self
    }

    /// Set the per-case timing floor (ns); a benchmark mean below it is implausible (a
    /// glitch or a forged grade) and scores the candidate zero.
    #[must_use]
    pub fn with_min_plausible_ns(mut self, min_plausible_ns: f64) -> Self {
        self.min_plausible_ns = min_plausible_ns;
        self
    }

    /// Set the completion extraction contract.
    #[must_use]
    pub fn with_submission_extract_mode(mut self, mode: SubmissionExtractMode) -> Self {
        self.submission_extract_mode = mode;
        self
    }

    /// The geometric mean of the benchmark `means`, or `None` if any is implausibly
    /// fast (below the configured floor) — a measurement glitch or a forged grade, which
    /// must not earn a reward.
    #[must_use]
    pub fn plausible_geomean(&self, means: &[f64]) -> Option<f64> {
        if means.iter().any(|&m| m < self.min_plausible_ns) {
            return None;
        }
        geomean(means)
    }

    /// Map a parsed `(correct, geom-mean ns)` outcome to a scalar reward: `0` unless
    /// the candidate is correct and produced a positive runtime; otherwise the
    /// speedup over the baseline (or an inverse-time proxy when no baseline is set).
    #[must_use]
    pub fn reward_value(&self, correct: bool, geomean_ns: Option<f64>) -> f32 {
        if !correct {
            return 0.0;
        }
        let Some(geo) = geomean_ns.filter(|&g| g > 0.0) else {
            return 0.0;
        };
        let value = match self.baseline_ns {
            Some(base) => base / geo,
            // No baseline: a normalized inverse-time so faster still scores higher.
            None => 1e9 / geo,
        };
        value as f32
    }

    /// Verify an extracted `submission` exactly as the reward does, returning the parsed
    /// correctness/timing record instead of a scalar reward.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] if the eval could not be carried out (scratch I/O or
    /// sandbox launch/supervision failure).
    pub fn verify_submission(&self, submission: &str) -> Result<TrimulVerification, RewardError> {
        self.run_eval(submission)
    }

    /// Extract a completion using this reward's configured prompt/extraction contract.
    #[must_use]
    pub fn extract_submission(&self, completion: &str) -> Option<String> {
        extract_submission_with_mode(completion, self.submission_extract_mode)
    }

    /// Resource ceilings for one eval. `address_space` is left unset — a CUDA process
    /// reserves a huge virtual range an address-space cap would wrongly kill.
    fn limits(&self) -> ResourceLimits {
        ResourceLimits {
            wall: self.wall,
            // CUDA reserves a huge virtual range and Triton JIT compilation is
            // CPU-heavy; an address-space or CPU-seconds cap would false-fail a
            // legitimate compile/eval and inject noise into the reward. The wall budget
            // (and the still-capped process / file-size limits) is the bound here.
            cpu: None,
            address_space: None,
            ..ResourceLimits::default()
        }
    }

    /// The `bash -c` program run inside the image: copy the read-only eval files next
    /// to the staged `submission.py`, run `test`, and — only if it passes — `benchmark`,
    /// each writing its result via fd 3 (`POPCORN_FD`). A trailing `true` keeps the
    /// shell's exit status clean; ferrl reads the result files, not the exit code.
    fn in_container_command() -> String {
        // Route the grade (POPCORN fd 3) to the *captured stdout pipe* (`3>&1`) and
        // send the eval's — and the candidate's — own stdout to `/dev/null`
        // (`1>/dev/null`). The grade therefore arrives on a channel the untrusted
        // candidate cannot reach: its spawn-worker does not inherit fd 3 (eval.py marks
        // it non-inheritable) and its stdout is discarded, so it cannot forge a pass by
        // writing files or printing. A separator splits the two sections; benchmark runs
        // only if `test` passed.
        "cp /eval/eval.py /eval/reference.py /eval/task.py /eval/utils.py . && \
         { POPCORN_FD=3 python eval.py test test_spec.txt 3>&1 1>/dev/null && \
           echo '===FERRL-BENCH===' && \
           POPCORN_FD=3 python eval.py benchmark bench_spec.txt 3>&1 1>/dev/null; }; \
         true"
            .to_string()
    }

    /// Build the [`RunSpec`] for a candidate whose scratch dir is `scratch`: the eval
    /// image with the GPU exposed, the eval bundle bound read-only, the scratch
    /// read-write, the network denied (the default), and only the env the eval needs.
    #[must_use]
    pub fn build_run_spec(&self, scratch: &Path) -> RunSpec {
        RunSpec::new(
            &self.image,
            vec!["bash".into(), "-c".into(), Self::in_container_command()],
        )
        .with_gpu(true)
        .with_binds(vec![
            Bind::ro(&self.eval_dir, "/eval"),
            Bind::rw(scratch, "/work").with_total_limit(self.scratch_max_bytes),
        ])
        .with_workdir("/work")
        .with_env(vec![
            ("HOME".into(), "/work/cache".into()),
            ("TRITON_CACHE_DIR".into(), "/work/cache/triton".into()),
            ("POPCORN_SEED".into(), self.secret_seed.to_string()),
        ])
        .with_limits(self.limits())
    }

    /// Stage `submission`, run the eval in the sandbox, and score the result files.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] only if the eval could not be *carried out* (scratch
    /// I/O or the sandbox failing to launch) — a crashing or wrong candidate is a
    /// `0.0` reward, not an error.
    fn run_eval(&self, submission: &str) -> Result<TrimulVerification, RewardError> {
        let scratch = self.make_scratch()?;
        let result = self
            .eval_in(&scratch, submission)
            .map(|outcome| outcome.verification);
        // Best-effort cleanup; the scratch is node-local and disposable.
        let _ = std::fs::remove_dir_all(&scratch);
        result
    }

    /// Measure the reference kernel's geometric-mean runtime (ns) on this node's GPU,
    /// by running the bundled reference as the candidate over the configured
    /// `benchmark_cases`. This is the value to pin as the speedup baseline
    /// ([`with_baseline_ns`](Self::with_baseline_ns)) — a *guarded pin*: measure it once
    /// on the target GPU, record it, and re-use it for runs on that same GPU.
    ///
    /// Returns `None` if the reference somehow did not pass correctness or produced no
    /// plausible timing (it should always pass — it *is* the reference).
    ///
    /// # Errors
    ///
    /// As [`reward`](RewardFn::reward): [`RewardError`] only if the eval could not be
    /// *carried out* (scratch I/O or the sandbox failing to launch).
    pub fn measure_reference_geomean_ns(&self) -> Result<Option<f64>, RewardError> {
        let outcome = self.run_eval(REFERENCE_SUBMISSION)?;
        Ok(outcome.correct.then_some(outcome.geomean_ns).flatten())
    }

    /// The body of [`run_eval`](Self::run_eval), split out so the scratch is always
    /// cleaned up.
    fn eval_in(&self, scratch: &Path, submission: &str) -> Result<TrimulEval, RewardError> {
        std::fs::create_dir_all(scratch.join("cache")).map_err(RewardError::verifier)?;
        write(scratch, "submission.py", submission)?;
        write(scratch, "test_spec.txt", &render_spec(&self.test_cases))?;
        write(
            scratch,
            "bench_spec.txt",
            &render_spec(&self.benchmark_cases),
        )?;

        // The grade arrives on the captured stdout (fd 3 → the pipe; see
        // `in_container_command`), NOT on a candidate-writable file — so a forged
        // `/work` file or a printed `check: pass` cannot influence the score. A
        // crashing candidate yields no grade lines, scored a failure.
        let outcome = self
            .sandbox
            .run(&self.build_run_spec(scratch))
            .map_err(RewardError::verifier)?;

        let has_benchmark_section = outcome.stdout.contains(RESULT_SPLIT);
        let (test_log, bench_log) = split_result(&outcome.stdout);
        let test_check = log_value(test_log, "check").map(str::to_string);
        let correct = test_check.as_deref() == Some("pass");
        let benchmark_means_ns = if correct {
            benchmark_means_ns(bench_log)
        } else {
            Vec::new()
        };
        let geomean_ns = if correct {
            self.plausible_geomean(&benchmark_means_ns)
        } else {
            None
        };
        let speedup = self
            .baseline_ns
            .zip(geomean_ns)
            .map(|(baseline, geo)| baseline / geo);
        Ok(TrimulEval {
            verification: TrimulVerification {
                correct,
                benchmark_means_ns,
                geomean_ns,
                speedup,
            },
            status: outcome.status,
            test_check,
            has_benchmark_section,
        })
    }

    fn reward_outcome(&self, completion: &str) -> Result<RewardOutcome, RewardError> {
        let Some(code) = self.extract_submission(completion) else {
            return Ok(RewardOutcome {
                reward: 0.0,
                diagnostic: Some("trimul:no_submission".to_string()),
            });
        };
        let scratch = self.make_scratch()?;
        let result = self.eval_in(&scratch, &code);
        // Best-effort cleanup; the scratch is node-local and disposable.
        let _ = std::fs::remove_dir_all(&scratch);
        let eval = result?;
        let reward = self.reward_from_eval(&eval);
        let diagnostic = self.reward_diagnostic(&eval);
        Ok(RewardOutcome { reward, diagnostic })
    }

    fn reward_from_eval(&self, eval: &TrimulEval) -> f32 {
        if !eval.status.is_success() {
            return 0.0;
        }
        self.reward_value(eval.verification.correct, eval.verification.geomean_ns)
    }

    fn reward_diagnostic(&self, eval: &TrimulEval) -> Option<String> {
        if !eval.status.is_success() {
            return Some(format!("trimul:sandbox_{}", run_status_label(eval.status)));
        }
        if eval.verification.correct && eval.verification.geomean_ns.is_some() {
            return None;
        }
        if !eval.verification.correct {
            return Some(if eval.test_check.is_some() {
                "trimul:test_failed".to_string()
            } else {
                "trimul:no_pass_grade".to_string()
            });
        }
        if !eval.has_benchmark_section {
            return Some("trimul:no_benchmark_section".to_string());
        }
        if eval.verification.benchmark_means_ns.is_empty() {
            return Some("trimul:no_benchmark_means".to_string());
        }
        Some("trimul:implausible_benchmark".to_string())
    }

    /// Create a fresh, uniquely-named scratch dir under `scratch_root`. The
    /// process-wide counter keeps names distinct across calls (and any concurrent
    /// callers), so two candidates never share a scratch.
    fn make_scratch(&self) -> Result<PathBuf, RewardError> {
        let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = self
            .scratch_root
            .join(format!("ferrl-trimul-{}-{seq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(RewardError::verifier)?;
        Ok(dir)
    }
}

/// Process-wide counter for unique scratch-dir names.
static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

/// Write `contents` to `dir/name`.
fn write(dir: &Path, name: &str, contents: &str) -> Result<(), RewardError> {
    std::fs::write(dir.join(name), contents).map_err(RewardError::verifier)
}

#[derive(Debug, Clone)]
struct TrimulEval {
    verification: TrimulVerification,
    status: RunStatus,
    test_check: Option<String>,
    has_benchmark_section: bool,
}

fn run_status_label(status: RunStatus) -> String {
    match status {
        RunStatus::Exited(code) => format!("exited_{code}"),
        RunStatus::TimedOut => "timed_out".to_string(),
        RunStatus::Signaled(signal) => format!("signaled_{signal}"),
        RunStatus::ScratchExceeded => "scratch_exceeded".to_string(),
    }
}

/// The marker the in-container command echoes between the `test` and `benchmark`
/// result sections on the grade channel.
const RESULT_SPLIT: &str = "===FERRL-BENCH===";

/// Split the captured grade stream into its `(test, benchmark)` sections. If the
/// separator is absent (the `test` run failed, so `benchmark` never ran), the whole
/// stream is the test section and the benchmark section is empty.
fn split_result(stdout: &str) -> (&str, &str) {
    stdout.split_once(RESULT_SPLIT).unwrap_or((stdout, ""))
}

impl RewardFn for TrimulReward {
    type Target = ();

    fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
        Ok(self.reward_outcome(completion)?.reward)
    }

    fn reward_group_detailed(
        &self,
        _sample: &Sample<()>,
        completions: &[String],
    ) -> Result<Vec<RewardOutcome>, RewardError> {
        completions
            .iter()
            .map(|completion| self.reward_outcome(completion))
            .collect()
    }
    // No `reward_group` override: a shared GPU scores candidates one at a time, so the
    // detailed path maps over the group while preserving per-candidate diagnostics.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(seqlen: u32, nomask: bool, distribution: Distribution) -> TrimulCase {
        TrimulCase {
            seqlen,
            bs: 1,
            dim: 64,
            hiddendim: 64,
            seed: 7,
            nomask,
            distribution,
        }
    }

    fn reward() -> TrimulReward {
        TrimulReward::new("/img.sif", "/eval", "/tmp")
    }

    #[test]
    fn case_renders_nomask_as_an_integer_and_distribution_as_a_token() {
        let line = case(32, true, Distribution::Normal).render();
        assert_eq!(
            line,
            "seqlen: 32; bs: 1; dim: 64; hiddendim: 64; seed: 7; nomask: 1; distribution: normal"
        );
        let masked = case(16, false, Distribution::Cauchy).render();
        assert!(masked.contains("nomask: 0"));
        assert!(masked.contains("distribution: cauchy"));
    }

    #[test]
    fn render_spec_is_one_line_per_case() {
        let body = render_spec(&[
            case(32, true, Distribution::Normal),
            case(64, false, Distribution::Cauchy),
        ]);
        assert_eq!(body.lines().count(), 2);
    }

    #[test]
    fn extract_submission_takes_the_final_fenced_block() {
        let completion = "draft:\n```python\nx = 1\n```\nfinal:\n```python\ndef custom_kernel(data):\n    return data\n```\n";
        assert_eq!(
            extract_submission(completion).as_deref(),
            Some("def custom_kernel(data):\n    return data")
        );
    }

    #[test]
    fn extract_submission_handles_a_bare_fence() {
        let completion = "```\nx = 1\n```";
        assert_eq!(extract_submission(completion).as_deref(), Some("x = 1"));
    }

    #[test]
    fn extract_submission_ignores_thinking_and_uses_final_answer_region() {
        let completion = "reasoning:\n```python\nx = 1\n```\n</think>\n\n```python\ndef custom_kernel(data):\n    return data\n```\n";
        assert_eq!(
            extract_submission_with_mode(completion, SubmissionExtractMode::ThinkingAfterThink)
                .as_deref(),
            Some("def custom_kernel(data):\n    return data")
        );
    }

    #[test]
    fn extract_submission_thinking_mode_requires_think_close() {
        let completion =
            "reasoning only:\n```python\ndef custom_kernel(data):\n    return data\n```\n";
        assert_eq!(
            extract_submission_with_mode(completion, SubmissionExtractMode::ThinkingAfterThink),
            None
        );
        assert_eq!(
            extract_submission(completion).as_deref(),
            Some("def custom_kernel(data):\n    return data")
        );
    }

    #[test]
    fn extract_submission_rejects_non_final_fence() {
        assert_eq!(
            extract_submission("```python\ndef custom_kernel(data):\n    return data\n```\nextra"),
            None
        );
    }

    #[test]
    fn extract_submission_is_none_without_a_closed_final_block() {
        assert_eq!(extract_submission("no code here"), None);
        assert_eq!(extract_submission("```python\nunterminated"), None);
        assert_eq!(extract_submission("```\n\n```"), None); // empty body
    }

    #[test]
    fn test_passed_reads_the_check_line() {
        assert!(test_passed(
            "test-count: 2\ntest.0.status: pass\ncheck: pass"
        ));
        assert!(!test_passed("test.0.status: fail\ncheck: fail"));
        assert!(!test_passed("benchmark-count: 1")); // no check line
    }

    #[test]
    fn benchmark_geomean_collects_the_means() {
        let log = "benchmark-count: 2\nbenchmark.0.mean: 100.0\nbenchmark.0.std: 5\nbenchmark.1.mean: 400.0\ncheck: pass";
        let means = benchmark_means_ns(log);
        assert_eq!(means, vec![100.0, 400.0]);
        // geometric mean of 100 and 400 is 200.
        assert!((geomean(&means).unwrap() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn geomean_rejects_empty_or_nonpositive() {
        assert_eq!(geomean(&[]), None);
        assert_eq!(geomean(&[1.0, 0.0]), None);
        assert_eq!(geomean(&[-1.0]), None);
    }

    #[test]
    fn plausible_geomean_rejects_sub_floor_timings() {
        let r = reward().with_min_plausible_ns(1000.0);
        // A real-looking set passes; an implausibly fast mean (a forged 0.001 ns or a
        // measurement glitch) makes the whole thing `None`, so it cannot score.
        assert!(r.plausible_geomean(&[2000.0, 8000.0]).is_some());
        assert_eq!(r.plausible_geomean(&[2000.0, 0.001]), None);
        assert_eq!(r.plausible_geomean(&[]), None);
    }

    #[test]
    fn reward_is_zero_when_incorrect() {
        assert_eq!(reward().reward_value(false, Some(100.0)), 0.0);
    }

    #[test]
    fn reward_is_zero_when_correct_but_no_timing() {
        assert_eq!(reward().reward_value(true, None), 0.0);
        assert_eq!(reward().reward_value(true, Some(0.0)), 0.0);
    }

    #[test]
    fn reward_is_speedup_over_baseline_when_set() {
        let r = reward().with_baseline_ns(1000.0);
        // Twice as fast as baseline -> reward 2.0; half as fast -> 0.5.
        assert!((r.reward_value(true, Some(500.0)) - 2.0).abs() < 1e-5);
        assert!((r.reward_value(true, Some(2000.0)) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn sandbox_failure_cannot_keep_positive_parsed_reward() {
        let r = reward().with_baseline_ns(1000.0);
        let eval = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![500.0],
                geomean_ns: Some(500.0),
                speedup: Some(2.0),
            },
            status: RunStatus::TimedOut,
            test_check: Some("pass".to_string()),
            has_benchmark_section: true,
        };

        assert_eq!(r.reward_from_eval(&eval), 0.0);
        assert_eq!(
            r.reward_diagnostic(&eval).as_deref(),
            Some("trimul:sandbox_timed_out")
        );
    }

    #[test]
    fn reward_falls_back_to_inverse_time_without_a_baseline() {
        // 1e9 / geo: a faster (smaller) geo yields a larger reward.
        let r = reward();
        assert!(r.reward_value(true, Some(1e6)) < r.reward_value(true, Some(1e5)));
    }

    #[test]
    fn trimul_verification_serializes_for_artifact_manifests() {
        let v = TrimulVerification {
            correct: true,
            benchmark_means_ns: vec![100.0, 400.0],
            geomean_ns: Some(200.0),
            speedup: Some(2.0),
        };
        let raw = serde_json::to_string(&v).unwrap();
        assert!(raw.contains("\"correct\":true"));
        let back: TrimulVerification = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn build_run_spec_exposes_gpu_and_denies_network() {
        let spec = reward().build_run_spec(Path::new("/tmp/scratch"));
        assert!(spec.gpu, "the eval needs the GPU");
        assert!(matches!(spec.network, crate::sandbox::NetworkPolicy::None));
        assert_eq!(spec.workdir, Path::new("/work"));
        assert!(spec
            .env
            .iter()
            .any(|(k, v)| k == "POPCORN_SEED" && v == "0"));
        assert!(
            spec.limits.address_space.is_none(),
            "an address-space cap is hostile to CUDA"
        );
    }

    #[test]
    fn build_run_spec_binds_eval_readonly_and_scratch_readwrite() {
        let spec = reward().build_run_spec(Path::new("/tmp/scratch"));
        let eval = spec
            .binds
            .iter()
            .find(|b| b.dst == Path::new("/eval"))
            .expect("eval bundle is bound");
        assert_eq!(eval.mode, crate::sandbox::BindMode::ReadOnly);
        let work = spec
            .binds
            .iter()
            .find(|b| b.dst == Path::new("/work"))
            .expect("scratch is bound");
        assert_eq!(work.mode, crate::sandbox::BindMode::ReadWrite);
        assert_eq!(work.total_limit, Some(1 << 30));
    }

    #[test]
    fn in_container_command_routes_the_grade_to_stdout_and_gates_benchmark() {
        let cmd = TrimulReward::in_container_command();
        // Grade -> fd 3 -> the captured stdout pipe; the eval's/candidate's own stdout
        // is discarded, so a forged file or print cannot influence the score.
        assert!(cmd.contains("eval.py test test_spec.txt 3>&1 1>/dev/null"));
        assert!(cmd.contains("eval.py benchmark bench_spec.txt 3>&1 1>/dev/null"));
        // The separator runs, and benchmark after it, only if `test` passed.
        assert!(cmd.contains("1>/dev/null && echo '===FERRL-BENCH===' &&"));
    }

    #[test]
    fn split_result_separates_test_and_benchmark_sections() {
        let (test, bench) = split_result("check: pass\n===FERRL-BENCH===\nbenchmark.0.mean: 5.0\n");
        assert!(test.contains("check: pass"));
        assert!(bench.contains("benchmark.0.mean: 5.0"));
        // No separator (test failed, benchmark never ran) -> all test, empty bench.
        let (test2, bench2) = split_result("check: fail\n");
        assert_eq!(test2, "check: fail\n");
        assert_eq!(bench2, "");
    }

    #[test]
    fn reward_fn_scores_zero_without_a_code_block() {
        // A completion with no fenced code block has nothing to run — the RewardFn
        // returns 0.0 without touching the sandbox.
        let s = Sample::new("write a faster TriMul kernel", ());
        assert_eq!(
            reward()
                .reward(&s, "Sorry, I can't help with that.")
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn detailed_reward_reports_missing_submission_without_sandbox() {
        let s = Sample::new("write a faster TriMul kernel", ());
        let outcomes = reward()
            .reward_group_detailed(&s, &["Sorry, no code.".to_string()])
            .unwrap();
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].reward, 0.0);
        assert_eq!(
            outcomes[0].diagnostic.as_deref(),
            Some("trimul:no_submission")
        );
    }

    #[test]
    fn reward_diagnostic_classifies_zero_eval_outcomes() {
        let r = reward();
        let test_failed = TrimulEval {
            verification: TrimulVerification {
                correct: false,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            test_check: None,
            has_benchmark_section: false,
        };
        assert_eq!(
            r.reward_diagnostic(&test_failed).as_deref(),
            Some("trimul:no_pass_grade")
        );

        let graded_failure = TrimulEval {
            test_check: Some("fail".to_string()),
            ..test_failed.clone()
        };
        assert_eq!(
            r.reward_diagnostic(&graded_failure).as_deref(),
            Some("trimul:test_failed")
        );

        let timed_out = TrimulEval {
            status: RunStatus::TimedOut,
            ..test_failed.clone()
        };
        assert_eq!(
            r.reward_diagnostic(&timed_out).as_deref(),
            Some("trimul:sandbox_timed_out")
        );

        let no_benchmark_section = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![],
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            test_check: Some("pass".to_string()),
            has_benchmark_section: false,
        };
        assert_eq!(
            r.reward_diagnostic(&no_benchmark_section).as_deref(),
            Some("trimul:no_benchmark_section")
        );

        let no_benchmark_means = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![],
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            test_check: Some("pass".to_string()),
            has_benchmark_section: true,
        };
        assert_eq!(
            r.reward_diagnostic(&no_benchmark_means).as_deref(),
            Some("trimul:no_benchmark_means")
        );

        let implausible_benchmark = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![0.001],
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            test_check: Some("pass".to_string()),
            has_benchmark_section: true,
        };
        assert_eq!(
            r.reward_diagnostic(&implausible_benchmark).as_deref(),
            Some("trimul:implausible_benchmark")
        );
    }

    // --- task.yml case-list parsing. The fixture uses made-up sizes (the real GPU Mode
    //     case list is not vendored); it mirrors the file's *shape*: surrounding sections
    //     to skip, Python `True`/`False`, and quoted keys/values.

    const SAMPLE_TASK_YML: &str = r#"
# name: trimul
files:
  - {"name": "submission.py", "source": "@SUBMISSION@"}
description: |
  A made-up description for the fixture.
  - this dash line is inside a skipped section and must be ignored
config:
  main: "eval.py"
tests:
  - {"seqlen": 8, "bs": 1, "dim": 16, "hiddendim": 16, "seed": 100, "nomask": True, "distribution": "normal"}
  - {"seqlen": 12, "bs": 2, "dim": 16, "hiddendim": 16, "seed": 101, "nomask": False, "distribution": "cauchy"}
benchmarks:
  - {"seqlen": 16, "bs": 1, "dim": 32, "hiddendim": 16, "seed": 200, "nomask": True, "distribution": "normal"}
"#;

    #[test]
    fn parse_task_yml_reads_both_sections_and_skips_the_rest() {
        let (tests, benches) = parse_task_yml(SAMPLE_TASK_YML).unwrap();
        assert_eq!((tests.len(), benches.len()), (2, 1));
        // Whole-case equality exercises every field at once — including Python
        // `True`/`False` and the distribution token — and confirms the surrounding
        // `files`/`description` sections were skipped (else the counts would be off).
        let want = TrimulCase {
            seqlen: 8,
            bs: 1,
            dim: 16,
            hiddendim: 16,
            seed: 100,
            nomask: true,
            distribution: Distribution::Normal,
        };
        assert_eq!(tests[0], want);
        assert_eq!(tests[1].distribution, Distribution::Cauchy);
        assert!(!tests[1].nomask); // "False"
        assert_eq!(benches[0].seqlen, 16);
    }

    #[test]
    fn parse_task_yml_round_trips_through_render_spec() {
        // A parsed case renders back to the spec line the eval harness reads.
        let (tests, _) = parse_task_yml(SAMPLE_TASK_YML).unwrap();
        let line = tests[0].render();
        assert!(line.contains("seqlen: 8"));
        assert!(line.contains("nomask: 1")); // rendered as an integer
        assert!(line.contains("distribution: normal"));
    }

    #[test]
    fn parse_task_yml_errors_on_missing_sections() {
        // No `tests:` / `benchmarks:` at all.
        assert!(matches!(
            parse_task_yml("files:\n  - {}\n"),
            Err(TrimulError::Parse(_))
        ));
        // A `tests:` section but no `benchmarks:`.
        let only_tests = "tests:\n  - {\"seqlen\": 8, \"bs\": 1, \"dim\": 16, \"hiddendim\": 16, \"seed\": 1, \"nomask\": True, \"distribution\": \"normal\"}\n";
        assert!(matches!(
            parse_task_yml(only_tests),
            Err(TrimulError::Parse(_))
        ));
    }

    #[test]
    fn parse_task_yml_errors_on_a_malformed_case() {
        // Missing the `distribution` field.
        let bad = "tests:\n  - {\"seqlen\": 8, \"bs\": 1, \"dim\": 16, \"hiddendim\": 16, \"seed\": 1, \"nomask\": True}\nbenchmarks:\n  - {}\n";
        assert!(matches!(parse_task_yml(bad), Err(TrimulError::Parse(_))));
        // A non-integer seqlen.
        let bad2 = "tests:\n  - {\"seqlen\": big, \"bs\": 1, \"dim\": 16, \"hiddendim\": 16, \"seed\": 1, \"nomask\": True, \"distribution\": \"normal\"}\nbenchmarks:\n  - {}\n";
        assert!(matches!(parse_task_yml(bad2), Err(TrimulError::Parse(_))));
    }

    #[test]
    fn parse_bool_accepts_python_yaml_and_int_spellings() {
        for t in ["True", "true", "yes", "1"] {
            assert!(parse_bool(Some(&t.to_string())).unwrap());
        }
        for f in ["False", "false", "no", "0"] {
            assert!(!parse_bool(Some(&f.to_string())).unwrap());
        }
        assert!(parse_bool(Some(&"maybe".to_string())).is_err());
        assert!(parse_bool(None).is_err());
    }

    #[test]
    fn build_prompt_states_the_function_contract() {
        let p = build_prompt();
        assert!(p.contains("custom_kernel(data)"));
        assert!(p.contains("(input, mask, weights, config)"));
        assert!(p.contains("[batch, seq_len, seq_len, dim]"));
    }

    #[test]
    fn build_prompt_lists_weight_constraints_without_domain_framing() {
        let p = build_prompt();
        assert!(p.contains(" - norm.weight\n"));
        assert!(p.contains(" - norm.bias\n"));
        assert!(p.contains(" - to_out.weight\n"));
        assert!(!p.contains("does not contain `norm.bias`"));
        assert!(!p.contains("AlphaFold"));
        assert!(!p.contains("```python"));
    }

    #[test]
    fn build_raw_prompt_restores_the_output_contract() {
        let p = build_raw_prompt(&build_prompt());
        assert!(p.contains("Output contract:"));
        assert!(p.contains("closed fenced Python code block"));
        assert!(p.contains("complete custom_kernel(data) implementation"));
        assert!(p.contains("Stop after the closing code fence."));
    }

    #[test]
    fn build_qwen3_5_chat_thinking_prompt_uses_system_user_roles() {
        let task = build_prompt();
        let p = build_qwen3_5_chat_thinking_prompt(&task);
        assert!(p.starts_with("<|im_start|>system\n"));
        assert!(p.contains("<|im_end|>\n<|im_start|>user\n"));
        assert!(p.contains("custom_kernel(data)"));
    }

    #[test]
    fn build_qwen3_5_chat_thinking_prompt_opens_thinking_prefix() {
        let task = build_prompt();
        let p = build_qwen3_5_chat_thinking_prompt(&task);
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(!p.ends_with("</think>\n\n"));
    }

    #[test]
    fn build_qwen3_5_chat_thinking_prompt_states_final_answer_contract() {
        let task = build_prompt();
        let p = build_qwen3_5_chat_thinking_prompt(&task);
        assert!(p.contains("After finishing the reasoning, close </think>."));
        assert!(p.contains("Immediately after </think>"));
        assert!(p.contains("Stop after the closing code fence."));
        assert!(p.contains(" - norm.bias\n"));
        assert!(!p.contains("does not contain `norm.bias`"));
    }
}
