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
//! 4. Map the captured grade to a shaped training reward: missing submissions score
//!    `0`, extracted-but-broken submissions get only a tiny format reward, runnable
//!    candidates get a small floor, partially correct candidates scale below the
//!    correctness floor, and test-passing candidates whose eval reaches a benchmark
//!    exit marker score the correctness floor plus any capped speed component.
//!    Implausibly fast timings (below the kernel-launch floor — a glitch or forged
//!    grade) still score `0`. The final artifact gate remains stricter than the
//!    training reward: held-out correctness plus repeated speedup audit.
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
//! implausibly fast time is rejected — so trivial fake-pass routes (forge a `/work`
//! result file, print a fake pass, report a 0 ns kernel) cannot reach the correctness
//! floor; the absurd-time path still scores zero. The negative-control suite gates
//! those cases. **Known residual (PoC):** a candidate that scans `/proc` for the
//! grader's grade fd *and* reports a physically plausible fake time could still forge
//! a pass — its worker shares the grader's PID namespace and uid, so only per-candidate
//! PID-namespace isolation closes it (earned when untrusted external submissions
//! arrive, not this PoC, whose kernel-writing policy is extremely unlikely to emit such
//! an exploit). The held-out `POPCORN_SEED` is likewise candidate-readable; both are
//! moot against an attacker who can already forge the grade and close together with
//! that isolation. The dynamic guard — watching the discovery run for implausible wins
//! and re-verifying top candidates — is the spec's Phase-1 instrumentation, done in the
//! run, not the reward.
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
use sha2::{Digest, Sha256};

use crate::reward::{RewardError, RewardFn, RewardOutcome};
use crate::sample::Sample;
use crate::sandbox::{ApptainerSandbox, Bind, ResourceLimits, RunSpec, RunStatus, Sandbox};

/// Versioned TriMul training-reward scheme identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrimulRewardScheme {
    /// The shaped training reward: format/runnable/partial/correctness/speed.
    TrimulShapedV1,
}

impl TrimulRewardScheme {
    fn as_str(self) -> &'static str {
        match self {
            Self::TrimulShapedV1 => "trimul_shaped_v1",
        }
    }
}

/// Handling for implausibly fast benchmark timings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplausibleBenchmarkPolicy {
    /// Score the candidate zero.
    Zero,
}

impl ImplausibleBenchmarkPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Zero => "zero",
        }
    }
}

/// Tunable TriMul training-reward profile.
///
/// The default exactly matches ferrl's original `trimul_shaped_v1` ladder. Custom
/// values are allowed when they preserve the core ordering: format-only no higher
/// than runnable, and all partial-progress rewards no higher than the fully correct
/// floor. Implausible benchmark timings remain fail-closed at zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrimulRewardProfile {
    /// Reward scheme identifier.
    pub scheme: TrimulRewardScheme,
    /// Tiny reward for an extractable final submission.
    pub format_extracted: f32,
    /// Reward for reaching the test harness.
    pub runnable: f32,
    /// Maximum sub-correctness reward for partial test progress.
    pub partial_correctness: f32,
    /// Fully correct floor before speed is considered.
    pub correctness: f32,
    /// Cap on the speed reward component.
    pub speed_cap: f32,
    /// Policy for implausibly fast benchmark timings.
    pub implausible_benchmark: ImplausibleBenchmarkPolicy,
}

impl Default for TrimulRewardProfile {
    fn default() -> Self {
        Self {
            scheme: TrimulRewardScheme::TrimulShapedV1,
            format_extracted: FORMAT_EXTRACTED_REWARD,
            runnable: RUNNABLE_REWARD,
            partial_correctness: PARTIAL_CORRECTNESS_REWARD,
            correctness: CORRECTNESS_REWARD,
            speed_cap: SPEED_REWARD_CAP,
            implausible_benchmark: ImplausibleBenchmarkPolicy::Zero,
        }
    }
}

impl TrimulRewardProfile {
    /// Validate that the profile is finite, non-negative, and preserves the reward ladder.
    ///
    /// # Errors
    ///
    /// Returns a human-readable config error if any value is non-finite, negative, or
    /// would let format-only/runnable/partial candidates outrank fully correct ones.
    pub fn validate(&self) -> Result<(), String> {
        match self.scheme {
            TrimulRewardScheme::TrimulShapedV1 => {}
        }
        match self.implausible_benchmark {
            ImplausibleBenchmarkPolicy::Zero => {}
        }
        for (label, value) in [
            ("trimul.reward.format_extracted", self.format_extracted),
            ("trimul.reward.runnable", self.runnable),
            (
                "trimul.reward.partial_correctness",
                self.partial_correctness,
            ),
            ("trimul.reward.correctness", self.correctness),
            ("trimul.reward.speed_cap", self.speed_cap),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{label} must be finite and >= 0"));
            }
        }
        if self.format_extracted > self.runnable {
            return Err(
                "trimul.reward.format_extracted must be <= trimul.reward.runnable".to_string(),
            );
        }
        if self.runnable + self.partial_correctness > self.correctness {
            return Err(
                "trimul.reward.runnable + trimul.reward.partial_correctness must be <= \
                 trimul.reward.correctness"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn metadata(self) -> serde_json::Value {
        serde_json::json!({
            "scheme": self.scheme.as_str(),
            "format_extracted": self.format_extracted,
            "runnable": self.runnable,
            "partial_correctness": self.partial_correctness,
            "correctness": self.correctness,
            "speed_cap": self.speed_cap,
            "implausible_benchmark": self.implausible_benchmark.as_str(),
        })
    }
}

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
#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
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

fn log_last_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().rev().find_map(|line| {
        let (k, v) = line.split_once(": ")?;
        (k.trim() == key).then_some(v.trim())
    })
}

fn log_i32_value(text: &str, key: &str) -> Option<i32> {
    log_last_value(text, key)?.parse().ok()
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
    /// denominator for the shaped reward's speed component. `None` falls back to an
    /// inverse-time signal.
    baseline_ns: Option<f64>,
    /// Tunable training-reward profile.
    reward_profile: TrimulRewardProfile,
    /// Wall-clock budget for one candidate's full eval.
    wall: Duration,
    /// Floor (ns) on each benchmark mean: a real GPU kernel cannot run faster than the
    /// kernel-launch overhead, so a sub-floor time is a measurement glitch or a forged
    /// grade — the candidate scores zero. Defence-in-depth against absurd reward
    /// gaming, on top of the off-filesystem grade channel.
    min_plausible_ns: f64,
    /// Which completion region may contain the final submitted code block.
    submission_extract_mode: SubmissionExtractMode,
    /// Optional CUDA device visibility override for every sandboxed verifier.
    verifier_cuda_visible_devices: Option<String>,
    /// Optional per-worker CUDA device visibility pool for concurrent verifiers.
    verifier_cuda_device_pool: Vec<String>,
    /// Maximum number of candidates from one GRPO group to verify concurrently.
    verifier_parallelism: usize,
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
            reward_profile: TrimulRewardProfile::default(),
            wall: Duration::from_secs(600),
            min_plausible_ns: 1_000.0,
            submission_extract_mode: SubmissionExtractMode::FinalFence,
            verifier_cuda_visible_devices: None,
            verifier_cuda_device_pool: Vec::new(),
            verifier_parallelism: 1,
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

    /// Set the shaped training-reward profile.
    ///
    /// # Errors
    ///
    /// Returns a config error if `profile` is non-finite, negative, or breaks the
    /// reward ladder enforced by [`TrimulRewardProfile::validate`].
    pub fn with_reward_profile(mut self, profile: TrimulRewardProfile) -> Result<Self, String> {
        profile.validate()?;
        self.reward_profile = profile;
        Ok(self)
    }

    /// The active shaped training-reward profile.
    #[must_use]
    pub fn reward_profile(&self) -> TrimulRewardProfile {
        self.reward_profile
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

    /// Set the CUDA-visible device list for the sandboxed verifier process.
    ///
    /// This is intentionally scoped to the verifier only: the trainer keeps its
    /// own device choice, while the eval image can be pointed at a separate
    /// Slurm-visible GPU when verifier memory would otherwise contend with the
    /// resident policy.
    #[must_use]
    pub fn with_verifier_cuda_visible_devices(mut self, devices: impl Into<String>) -> Self {
        let devices = devices.into();
        self.verifier_cuda_visible_devices = (!devices.trim().is_empty()).then_some(devices);
        self
    }

    /// Set per-worker CUDA-visible device lists for concurrent verifier workers.
    ///
    /// When non-empty, worker `i` receives `devices[i % devices.len()]` instead of the
    /// global verifier CUDA visibility override. Use one physical GPU per concurrent
    /// worker for memory-heavy evals; an empty list keeps the global setting.
    #[must_use]
    pub fn with_verifier_cuda_device_pool(mut self, devices: Vec<String>) -> Self {
        self.verifier_cuda_device_pool = devices
            .into_iter()
            .map(|devices| devices.trim().to_string())
            .filter(|devices| !devices.is_empty())
            .collect();
        self
    }

    /// Set the maximum number of candidates in one GRPO group to verify concurrently.
    ///
    /// The default is `1`, preserving the historical sequential verifier behavior. A
    /// higher value is useful only when the verifier has isolated GPU capacity; the
    /// implementation still returns outcomes in input order and propagates verifier
    /// errors fail-closed.
    #[must_use]
    pub fn with_verifier_parallelism(mut self, parallelism: usize) -> Self {
        self.verifier_parallelism = parallelism.max(1);
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

    /// Map a parsed `(correct, geom-mean ns)` outcome to the speed component of the
    /// training reward: `0` unless the candidate is correct and produced a positive
    /// runtime; otherwise the speedup over the baseline (or an inverse-time proxy when
    /// no baseline is set).
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
    /// to the staged `submission.py`, run `test`, and — only if it exits cleanly —
    /// `benchmark`, each writing its result via fd 3 (`POPCORN_FD`). A trailing `true` keeps the
    /// shell's exit status clean; ferrl reads the result files, not the exit code.
    fn in_container_command() -> String {
        // Route the grade (POPCORN fd 3) to the *captured stdout pipe* (`3>&1`) and
        // send the eval's — and the candidate's — own stdout to `/dev/null`
        // (`1>/dev/null`). The grade therefore arrives on a channel the untrusted
        // candidate cannot reach: its spawn-worker does not inherit fd 3 (eval.py marks
        // it non-inheritable) and its stdout is discarded, so it cannot forge a pass by
        // writing files or printing. A separator splits the two sections; benchmark runs
        // only if `test` exits cleanly.
        "cp /eval/eval.py /eval/reference.py /eval/task.py /eval/utils.py . && \
         { POPCORN_FD=3 python eval.py test test_spec.txt 3>&1 1>/dev/null; \
           test_rc=$?; \
           echo \"test-exit: $test_rc\"; \
           if [ \"$test_rc\" -eq 0 ]; then \
             echo '===FERRL-BENCH==='; \
             POPCORN_FD=3 python eval.py benchmark bench_spec.txt 3>&1 1>/dev/null; \
             bench_rc=$?; \
             echo \"benchmark-exit: $bench_rc\"; \
           fi; }; \
         true"
            .to_string()
    }

    /// Build the [`RunSpec`] for a candidate whose scratch dir is `scratch`: the eval
    /// image with the GPU exposed, the eval bundle bound read-only, the scratch
    /// read-write, the network denied (the default), and only the env the eval needs.
    #[must_use]
    pub fn build_run_spec(&self, scratch: &Path) -> RunSpec {
        self.build_run_spec_with_devices(scratch, self.verifier_cuda_visible_devices.as_deref())
    }

    fn verifier_devices_for_worker(&self, worker_index: usize) -> Option<&str> {
        self.verifier_cuda_device_pool
            .get(worker_index % self.verifier_cuda_device_pool.len().max(1))
            .map(String::as_str)
            .or(self.verifier_cuda_visible_devices.as_deref())
    }

    fn build_run_spec_for_worker(&self, scratch: &Path, worker_index: usize) -> RunSpec {
        self.build_run_spec_with_devices(scratch, self.verifier_devices_for_worker(worker_index))
    }

    fn build_run_spec_with_devices(&self, scratch: &Path, devices: Option<&str>) -> RunSpec {
        let mut env = vec![
            ("HOME".into(), "/work/cache".into()),
            ("TRITON_CACHE_DIR".into(), "/work/cache/triton".into()),
            ("POPCORN_SEED".into(), self.secret_seed.to_string()),
        ];
        if let Some(devices) = devices {
            env.push(("CUDA_VISIBLE_DEVICES".into(), devices.to_string()));
        }

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
        .with_env(env)
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
        self.eval_in_with_spec(scratch, submission, &self.build_run_spec(scratch))
    }

    fn eval_in_for_worker(
        &self,
        scratch: &Path,
        submission: &str,
        worker_index: usize,
    ) -> Result<TrimulEval, RewardError> {
        self.eval_in_with_spec(
            scratch,
            submission,
            &self.build_run_spec_for_worker(scratch, worker_index),
        )
    }

    fn eval_in_with_spec(
        &self,
        scratch: &Path,
        submission: &str,
        spec: &RunSpec,
    ) -> Result<TrimulEval, RewardError> {
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
        let outcome = self.sandbox.run(spec).map_err(RewardError::verifier)?;

        let has_benchmark_section = outcome.stdout.contains(RESULT_SPLIT);
        let (test_log, bench_log) = split_result(&outcome.stdout);
        let test_check = log_value(test_log, "check").map(str::to_string);
        let test_exit = log_i32_value(test_log, "test-exit");
        let benchmark_exit = log_i32_value(bench_log, "benchmark-exit");
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
            output: TrimulEvalOutput {
                stdout: outcome.stdout,
                stderr: outcome.stderr,
            },
            test_check,
            test_exit,
            benchmark_exit,
            has_benchmark_section,
        })
    }

    fn reward_outcome(&self, completion: &str) -> Result<RewardOutcome, RewardError> {
        self.reward_outcome_for_worker(completion, 0)
    }

    fn reward_outcome_for_worker(
        &self,
        completion: &str,
        worker_index: usize,
    ) -> Result<RewardOutcome, RewardError> {
        let Some(code) = self.extract_submission(completion) else {
            return Ok(RewardOutcome {
                reward: 0.0,
                diagnostic: Some("trimul:no_submission".to_string()),
                metadata: Some(serde_json::json!({
                    "task": "trimul",
                    "reward_scheme": self.reward_profile.scheme.as_str(),
                    "reward_profile": self.reward_profile.metadata(),
                    "submission_extracted": false,
                })),
            });
        };
        let scratch = self.make_scratch()?;
        let result = self.eval_in_for_worker(&scratch, &code, worker_index);
        // Best-effort cleanup; the scratch is node-local and disposable.
        let _ = std::fs::remove_dir_all(&scratch);
        let eval = result?;
        let reward = self.reward_from_extracted_eval(&eval);
        let diagnostic = self.reward_diagnostic(&eval);
        let metadata = Some(self.reward_metadata(&code, &eval, reward));
        Ok(RewardOutcome {
            reward,
            diagnostic,
            metadata,
        })
    }

    fn reward_metadata(
        &self,
        submission: &str,
        eval: &TrimulEval,
        training_reward: f32,
    ) -> serde_json::Value {
        let test_progress = eval.test_progress();
        let speed_component = if eval.verification.correct && eval.benchmark_exit == Some(0) {
            self.speed_reward_component(eval.verification.geomean_ns)
        } else {
            0.0
        };
        let mut metadata = serde_json::json!({
            "task": "trimul",
            "reward_scheme": self.reward_profile.scheme.as_str(),
            "reward_profile": self.reward_profile.metadata(),
            "submission_extracted": true,
            "source_sha256": sha256_hex(submission.as_bytes()),
            "source_len_bytes": submission.len(),
            "training_reward": training_reward,
            "sandbox_status": run_status_label(eval.status),
            "sandbox_success": eval.status.is_success(),
            "sandbox_stdout_len_bytes": eval.output.stdout.len(),
            "sandbox_stderr_len_bytes": eval.output.stderr.len(),
            "test_check": eval.test_check.as_deref(),
            "test_exit": eval.test_exit,
            "test_pass_count": test_progress.pass_count,
            "test_case_count": test_progress.case_count,
            "test_pass_fraction": test_progress.fraction(),
            "benchmark_exit": eval.benchmark_exit,
            "has_benchmark_section": eval.has_benchmark_section,
            "correct": eval.verification.correct,
            "benchmark_mean_count": eval.verification.benchmark_means_ns.len(),
            "geomean_ns": eval.verification.geomean_ns,
            "speedup": eval.verification.speedup,
            "speed_reward_component": speed_component,
        });

        if eval.should_preserve_output_tail() {
            let object = metadata
                .as_object_mut()
                .expect("TriMul reward metadata is a JSON object");
            if let Some(stdout_tail) =
                bounded_tail(&eval.output.stdout, EVAL_OUTPUT_TAIL_LIMIT_BYTES)
            {
                object.insert("sandbox_stdout_tail".to_string(), stdout_tail.into());
                object.insert(
                    "sandbox_stdout_tail_truncated".to_string(),
                    (eval.output.stdout.len() > EVAL_OUTPUT_TAIL_LIMIT_BYTES).into(),
                );
            }
            if let Some(stderr_tail) =
                bounded_tail(&eval.output.stderr, EVAL_OUTPUT_TAIL_LIMIT_BYTES)
            {
                object.insert("sandbox_stderr_tail".to_string(), stderr_tail.into());
                object.insert(
                    "sandbox_stderr_tail_truncated".to_string(),
                    (eval.output.stderr.len() > EVAL_OUTPUT_TAIL_LIMIT_BYTES).into(),
                );
            }
        }

        metadata
    }

    fn reward_from_extracted_eval(&self, eval: &TrimulEval) -> f32 {
        if eval_has_implausible_benchmark(eval) {
            return 0.0;
        }
        self.reward_from_eval(eval)
            .max(self.reward_profile.format_extracted)
    }

    fn reward_from_eval(&self, eval: &TrimulEval) -> f32 {
        if !eval.status.is_success() {
            return 0.0;
        }
        if eval.test_exit.is_none() {
            return 0.0;
        }
        if eval_has_implausible_benchmark(eval) {
            // A candidate with sub-floor benchmark timings is suspicious (or a
            // measurement glitch). Keep this fail-closed at zero instead of giving
            // the extraction, runnable, or correctness floors.
            return 0.0;
        }
        if eval.test_exit != Some(0) {
            return self.runnable_progress_reward(eval);
        }
        if eval.verification.correct && eval.has_benchmark_section && eval.benchmark_exit.is_some()
        {
            if eval.benchmark_exit == Some(0) {
                return self.reward_profile.correctness
                    + self.speed_reward_component(eval.verification.geomean_ns);
            }
            return self.reward_profile.correctness;
        }
        self.runnable_progress_reward(eval)
    }

    fn speed_reward_component(&self, geomean_ns: Option<f64>) -> f32 {
        self.reward_value(true, geomean_ns)
            .clamp(0.0, self.reward_profile.speed_cap)
    }

    fn runnable_progress_reward(&self, eval: &TrimulEval) -> f32 {
        let progress = eval.test_progress();
        self.reward_profile.runnable
            + self.reward_profile.partial_correctness * progress.fraction() as f32
    }

    fn reward_diagnostic(&self, eval: &TrimulEval) -> Option<String> {
        if !eval.status.is_success() {
            return Some(format!("trimul:sandbox_{}", run_status_label(eval.status)));
        }
        match eval.test_exit {
            Some(0) => {}
            Some(_) if eval_has_shape_failure(eval) => {
                return Some("trimul:test_shape_mismatch".to_string());
            }
            Some(_) => return Some("trimul:test_process_failed".to_string()),
            None => return Some("trimul:missing_test_exit".to_string()),
        }
        if eval.benchmark_exit.is_some_and(|code| code != 0) {
            return Some("trimul:benchmark_process_failed".to_string());
        }
        if eval.verification.correct && eval.verification.geomean_ns.is_some() {
            return if eval.benchmark_exit == Some(0) {
                None
            } else {
                Some("trimul:missing_benchmark_exit".to_string())
            };
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
    output: TrimulEvalOutput,
    test_check: Option<String>,
    test_exit: Option<i32>,
    benchmark_exit: Option<i32>,
    has_benchmark_section: bool,
}

impl TrimulEval {
    fn should_preserve_output_tail(&self) -> bool {
        !self.status.is_success()
            || self.test_exit != Some(0)
            || (self.verification.correct && self.benchmark_exit != Some(0))
    }

    fn test_progress(&self) -> TestProgress {
        let (test_log, _) = split_result(&self.output.stdout);
        test_progress(test_log)
    }
}

#[derive(Debug, Clone, Default)]
struct TrimulEvalOutput {
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TestProgress {
    pass_count: usize,
    case_count: usize,
}

impl TestProgress {
    fn fraction(self) -> f64 {
        if self.case_count == 0 {
            0.0
        } else {
            (self.pass_count.min(self.case_count) as f64 / self.case_count as f64).clamp(0.0, 1.0)
        }
    }
}

fn test_progress(test_log: &str) -> TestProgress {
    let declared_count = log_value(test_log, "test-count").and_then(|value| value.parse().ok());
    let mut statuses: HashMap<usize, bool> = HashMap::new();
    for line in test_log.lines() {
        let Some((key, value)) = line.split_once(": ") else {
            continue;
        };
        let key = key.trim();
        let Some(index) = key
            .strip_prefix("test.")
            .and_then(|key| key.strip_suffix(".status"))
            .and_then(|index| index.parse::<usize>().ok())
        else {
            continue;
        };
        let passed = value.trim() == "pass";
        if let Some(seen) = statuses.get_mut(&index) {
            *seen &= passed;
        } else {
            statuses.insert(index, passed);
        }
    }
    if declared_count == Some(0) && test_passed(test_log) {
        return TestProgress {
            pass_count: 1,
            case_count: 1,
        };
    }
    let pass_count = statuses
        .iter()
        .filter(|(index, passed)| {
            **passed && declared_count.is_none_or(|case_count| **index < case_count)
        })
        .count();
    TestProgress {
        pass_count,
        case_count: declared_count.unwrap_or(statuses.len()),
    }
}

fn eval_has_shape_failure(eval: &TrimulEval) -> bool {
    text_has_shape_failure(&eval.output.stderr) || text_has_shape_failure(&eval.output.stdout)
}

fn eval_has_implausible_benchmark(eval: &TrimulEval) -> bool {
    !eval.verification.benchmark_means_ns.is_empty() && eval.verification.geomean_ns.is_none()
}

fn text_has_shape_failure(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    text.contains("shapes cannot be multiplied")
        || text.contains("shape mismatch")
        || text.contains("size mismatch")
        || text.contains("invalid shape")
        || text.contains("normalized_shape")
        || text.contains("same shape as normalized_shape")
        || (text.contains("size of tensor")
            && (text.contains("must match") || text.contains("mismatch")))
}

fn run_status_label(status: RunStatus) -> String {
    match status {
        RunStatus::Exited(code) => format!("exited_{code}"),
        RunStatus::TimedOut => "timed_out".to_string(),
        RunStatus::Signaled(signal) => format!("signaled_{signal}"),
        RunStatus::ScratchExceeded => "scratch_exceeded".to_string(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

/// The marker the in-container command echoes between the `test` and `benchmark`
/// result sections on the grade channel.
const RESULT_SPLIT: &str = "===FERRL-BENCH===";

/// Maximum captured eval output text stored in candidate metadata.
const EVAL_OUTPUT_TAIL_LIMIT_BYTES: usize = 4096;

/// Tiny credit for emitting an extractable final submission. This separates
/// truncation/parser failures from candidates worth running, without letting format-only
/// completions compete with runnable or correct code.
const FORMAT_EXTRACTED_REWARD: f32 = 0.02;
/// Credit for reaching the test harness and producing a test-exit marker.
const RUNNABLE_REWARD: f32 = 0.05;
/// Maximum sub-correctness credit. Kept below [`CORRECTNESS_REWARD`] so any fully
/// correct candidate outranks every partial candidate.
const PARTIAL_CORRECTNESS_REWARD: f32 = 0.75;
/// Fully correct candidates get this floor before speed is considered.
const CORRECTNESS_REWARD: f32 = 1.0;
/// Cap the speed component so one lucky timing run cannot swamp correctness progress.
const SPEED_REWARD_CAP: f32 = 2.0;

/// Split the captured grade stream into its `(test, benchmark)` sections. If the
/// separator is absent (the `test` run failed, so `benchmark` never ran), the whole
/// stream is the test section and the benchmark section is empty.
fn split_result(stdout: &str) -> (&str, &str) {
    stdout.rsplit_once(RESULT_SPLIT).unwrap_or((stdout, ""))
}

fn bounded_tail(text: &str, limit: usize) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    if text.len() <= limit {
        return Some(text.to_string());
    }
    let mut start = text.len() - limit;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    Some(text[start..].to_string())
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
        if self.verifier_parallelism <= 1 || completions.len() <= 1 {
            return completions
                .iter()
                .enumerate()
                .map(|(index, completion)| self.reward_outcome_for_worker(completion, index))
                .collect();
        }
        map_bounded_reward_outcomes(
            completions,
            self.verifier_parallelism,
            |index, completion| self.reward_outcome_for_worker(completion, index),
        )
    }
    // No `reward_group` override: the detailed path preserves per-candidate diagnostics.
}

fn map_bounded_reward_outcomes<T, F>(
    items: &[T],
    parallelism: usize,
    f: F,
) -> Result<Vec<RewardOutcome>, RewardError>
where
    T: Sync,
    F: Fn(usize, &T) -> Result<RewardOutcome, RewardError> + Sync,
{
    let width = parallelism.max(1);
    let mut out = Vec::with_capacity(items.len());
    for (chunk_index, chunk) in items.chunks(width).enumerate() {
        let base = chunk_index * width;
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(chunk.len());
            for (offset, item) in chunk.iter().enumerate() {
                let f = &f;
                handles.push(scope.spawn(move || f(base + offset, item)));
            }
            handles
                .into_iter()
                .map(std::thread::ScopedJoinHandle::join)
                .collect::<Vec<_>>()
        });
        for result in results {
            match result {
                Ok(Ok(outcome)) => out.push(outcome),
                Ok(Err(err)) => return Err(err),
                Err(_) => return Err(RewardError::msg("trimul reward worker panicked")),
            }
        }
    }
    Ok(out)
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
    fn reward_profile_default_matches_original_ladder_and_validates() {
        let profile = TrimulRewardProfile::default();

        assert_eq!(profile.format_extracted, FORMAT_EXTRACTED_REWARD);
        assert_eq!(profile.runnable, RUNNABLE_REWARD);
        assert_eq!(profile.partial_correctness, PARTIAL_CORRECTNESS_REWARD);
        assert_eq!(profile.correctness, CORRECTNESS_REWARD);
        assert_eq!(profile.speed_cap, SPEED_REWARD_CAP);
        profile.validate().unwrap();
    }

    #[test]
    fn reward_profile_rejects_nonfinite_negative_and_inverted_ladders() {
        let negative = TrimulRewardProfile {
            runnable: -0.01,
            ..TrimulRewardProfile::default()
        };
        assert!(negative.validate().unwrap_err().contains("finite and >= 0"));

        let nonfinite = TrimulRewardProfile {
            speed_cap: f32::NAN,
            ..TrimulRewardProfile::default()
        };
        assert!(nonfinite
            .validate()
            .unwrap_err()
            .contains("finite and >= 0"));

        let format_above_runnable = TrimulRewardProfile {
            format_extracted: 0.10,
            ..TrimulRewardProfile::default()
        };
        assert!(format_above_runnable
            .validate()
            .unwrap_err()
            .contains("format_extracted"));

        let partial_above_correctness = TrimulRewardProfile {
            runnable: 0.40,
            ..TrimulRewardProfile::default()
        };
        assert!(partial_above_correctness
            .validate()
            .unwrap_err()
            .contains("partial_correctness"));
    }

    #[test]
    fn reward_rejects_invalid_profile_at_builder_boundary() {
        let invalid = TrimulRewardProfile {
            runnable: 0.40,
            ..TrimulRewardProfile::default()
        };

        assert!(reward()
            .with_reward_profile(invalid)
            .unwrap_err()
            .contains("partial_correctness"));
    }

    fn custom_reward_profile() -> TrimulRewardProfile {
        TrimulRewardProfile {
            format_extracted: 0.03,
            runnable: 0.10,
            partial_correctness: 0.20,
            correctness: 0.50,
            speed_cap: 0.25,
            ..TrimulRewardProfile::default()
        }
    }

    fn format_only_eval() -> TrimulEval {
        TrimulEval {
            verification: TrimulVerification {
                correct: false,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::TimedOut,
            output: TrimulEvalOutput::default(),
            test_check: None,
            test_exit: None,
            benchmark_exit: None,
            has_benchmark_section: false,
        }
    }

    fn partial_progress_eval() -> TrimulEval {
        TrimulEval {
            verification: TrimulVerification {
                correct: false,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput {
                stdout: "test-count: 4\ntest.0.status: pass\ntest.1.status: pass\ntest-exit: 1\n"
                    .to_string(),
                stderr: String::new(),
            },
            test_check: Some("fail".to_string()),
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        }
    }

    fn correct_fast_eval() -> TrimulEval {
        TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![250.0],
                geomean_ns: Some(250.0),
                speedup: Some(4.0),
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        }
    }

    fn assert_profile_number(profile_metadata: &serde_json::Value, key: &str, expected: f64) {
        assert!((profile_metadata[key].as_f64().unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn custom_reward_profile_controls_ladder() {
        let profile = custom_reward_profile();
        profile.validate().unwrap();
        let r = reward()
            .with_reward_profile(profile)
            .unwrap()
            .with_baseline_ns(1000.0);

        assert_eq!(r.reward_from_extracted_eval(&format_only_eval()), 0.03);
        assert!((r.reward_from_eval(&partial_progress_eval()) - 0.20).abs() < 1e-6);
        assert_eq!(r.reward_from_extracted_eval(&correct_fast_eval()), 0.75);
    }

    #[test]
    fn custom_reward_profile_records_metadata() {
        let r = reward()
            .with_reward_profile(custom_reward_profile())
            .unwrap()
            .with_baseline_ns(1000.0);
        let fast = correct_fast_eval();
        let training_reward = r.reward_from_extracted_eval(&fast);
        let metadata = r.reward_metadata(
            "def custom_kernel(data): return data",
            &fast,
            training_reward,
        );
        let profile_metadata = &metadata["reward_profile"];

        assert_profile_number(profile_metadata, "format_extracted", 0.03);
        assert_profile_number(profile_metadata, "runnable", 0.10);
        assert_profile_number(profile_metadata, "partial_correctness", 0.20);
        assert_profile_number(profile_metadata, "correctness", 0.50);
        assert_profile_number(profile_metadata, "speed_cap", 0.25);
        assert_eq!(
            profile_metadata["implausible_benchmark"],
            serde_json::json!("zero")
        );
    }

    #[test]
    fn verifier_parallelism_defaults_to_one_and_clamps_zero() {
        assert_eq!(reward().verifier_parallelism, 1);
        assert_eq!(
            reward().with_verifier_parallelism(0).verifier_parallelism,
            1
        );
        assert_eq!(
            reward().with_verifier_parallelism(3).verifier_parallelism,
            3
        );
    }

    #[test]
    fn bounded_reward_map_preserves_input_order() {
        let items = [3_i32, 1, 2, 0];
        let got = map_bounded_reward_outcomes(&items, 3, |index, item| {
            std::thread::sleep(Duration::from_millis((3 - index.min(3)) as u64));
            Ok(RewardOutcome {
                reward: (*item * 10 + index as i32) as f32,
                diagnostic: Some(format!("{index}:{item}")),
                metadata: None,
            })
        })
        .unwrap();

        assert_eq!(
            got.iter().map(|outcome| outcome.reward).collect::<Vec<_>>(),
            vec![30.0, 11.0, 22.0, 3.0]
        );
        assert_eq!(got[2].diagnostic.as_deref(), Some("2:2"));
    }

    #[test]
    fn bounded_reward_map_returns_first_error_in_input_order() {
        let items = [0_i32, 1, 2, 3];
        let err = map_bounded_reward_outcomes(&items, 4, |index, _| {
            if index >= 2 {
                return Err(RewardError::msg(format!("boom-{index}")));
            }
            Ok(RewardOutcome::reward(index as f32))
        })
        .unwrap_err();

        assert_eq!(err.to_string(), "boom-2");
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
    fn test_progress_counts_declared_case_passes() {
        let progress = test_progress(
            "test-count: 4\n\
             test.0.status: pass\n\
             test.0.status: pass\n\
             test.1.status: fail\n\
             test.2.status: pass\n\
             test.99.status: pass\n",
        );
        assert_eq!(
            progress,
            TestProgress {
                pass_count: 2,
                case_count: 4,
            }
        );
        assert!((progress.fraction() - 0.5).abs() < f64::EPSILON);
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
    #[allow(clippy::cognitive_complexity)] // one table-like ladder check is clearer than scattered cases
    fn shaped_reward_orders_format_runnable_correctness_and_speed() {
        let r = reward().with_baseline_ns(1000.0);
        let partial = TrimulEval {
            verification: TrimulVerification {
                correct: false,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput {
                stdout: "test-count: 4\ntest.0.status: pass\ntest.1.status: pass\ntest-exit: 1\n"
                    .to_string(),
                stderr: String::new(),
            },
            test_check: Some("fail".to_string()),
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        };
        let partial_reward = r.reward_from_eval(&partial);
        assert!(
            (partial_reward - (RUNNABLE_REWARD + PARTIAL_CORRECTNESS_REWARD * 0.5)).abs() < 1e-6
        );

        let correct_benchmark_failed = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(1),
            has_benchmark_section: true,
        };
        let correct_reward = r.reward_from_eval(&correct_benchmark_failed);
        assert_eq!(correct_reward, CORRECTNESS_REWARD);

        let slow = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![2000.0],
                geomean_ns: Some(2000.0),
                speedup: Some(0.5),
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };
        let slow_reward = r.reward_from_eval(&slow);
        assert!((slow_reward - 1.5).abs() < 1e-6);

        let fast = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![250.0],
                geomean_ns: Some(250.0),
                speedup: Some(4.0),
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };
        let fast_reward = r.reward_from_eval(&fast);
        assert_eq!(fast_reward, CORRECTNESS_REWARD + SPEED_REWARD_CAP);

        assert!(partial_reward < correct_reward);
        assert!(correct_reward < slow_reward);
        assert!(slow_reward < fast_reward);
    }

    #[test]
    fn implausible_benchmark_scores_zero_even_after_extraction() {
        let r = reward().with_baseline_ns(1000.0);
        let eval = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![0.001],
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };

        assert_eq!(r.reward_from_eval(&eval), 0.0);
        assert_eq!(r.reward_from_extracted_eval(&eval), 0.0);
    }

    #[test]
    fn extracted_submission_gets_format_floor_for_eval_failure() {
        let r = reward();
        let eval = TrimulEval {
            verification: TrimulVerification {
                correct: false,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::TimedOut,
            output: TrimulEvalOutput::default(),
            test_check: None,
            test_exit: None,
            benchmark_exit: None,
            has_benchmark_section: false,
        };

        assert_eq!(r.reward_from_eval(&eval), 0.0);
        assert_eq!(r.reward_from_extracted_eval(&eval), FORMAT_EXTRACTED_REWARD);
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
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };

        assert_eq!(r.reward_from_eval(&eval), 0.0);
        assert_eq!(
            r.reward_diagnostic(&eval).as_deref(),
            Some("trimul:sandbox_timed_out")
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // metadata regression intentionally checks each preserved marker
    fn reward_metadata_preserves_source_hash_and_eval_markers() {
        let r = reward().with_baseline_ns(1000.0);
        let source = "def custom_kernel(data):\n    return data\n";
        let eval = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![500.0, 800.0],
                geomean_ns: Some(632.455_532_033_675_9),
                speedup: Some(1.581_138_830_084_189_8),
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput {
                stdout: "check: pass\ntest-exit: 1\n".to_string(),
                stderr: format!(
                    "Traceback: candidate crashed\n{}",
                    "x".repeat(EVAL_OUTPUT_TAIL_LIMIT_BYTES + 8)
                ),
            },
            test_check: Some("pass".to_string()),
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        };

        let training_reward = r.reward_from_extracted_eval(&eval);
        let metadata = r.reward_metadata(source, &eval, training_reward);
        assert_eq!(metadata["task"], serde_json::json!("trimul"));
        assert_eq!(
            metadata["reward_scheme"],
            serde_json::json!("trimul_shaped_v1")
        );
        assert_eq!(metadata["submission_extracted"], serde_json::json!(true));
        assert_eq!(
            metadata["source_sha256"],
            serde_json::json!(sha256_hex(source.as_bytes()))
        );
        assert_eq!(
            metadata["source_len_bytes"],
            serde_json::json!(source.len())
        );
        assert_eq!(metadata["sandbox_status"], serde_json::json!("exited_0"));
        assert_eq!(metadata["sandbox_success"], serde_json::json!(true));
        assert_eq!(
            metadata["sandbox_stdout_len_bytes"],
            serde_json::json!("check: pass\ntest-exit: 1\n".len())
        );
        assert_eq!(
            metadata["sandbox_stderr_len_bytes"],
            serde_json::json!(eval.output.stderr.len())
        );
        assert_eq!(
            metadata["sandbox_stdout_tail"],
            serde_json::json!("check: pass\ntest-exit: 1\n")
        );
        assert_eq!(
            metadata["sandbox_stdout_tail_truncated"],
            serde_json::json!(false)
        );
        let stderr_tail = metadata["sandbox_stderr_tail"].as_str().unwrap();
        assert_eq!(stderr_tail.len(), EVAL_OUTPUT_TAIL_LIMIT_BYTES);
        assert!(stderr_tail.chars().all(|ch| ch == 'x'));
        assert_eq!(
            metadata["sandbox_stderr_tail_truncated"],
            serde_json::json!(true)
        );
        assert_eq!(metadata["test_check"], serde_json::json!("pass"));
        assert_eq!(metadata["test_exit"], serde_json::json!(1));
        assert_eq!(
            metadata["training_reward"],
            serde_json::json!(training_reward)
        );
        assert_eq!(metadata["test_pass_count"], serde_json::json!(0));
        assert_eq!(metadata["test_case_count"], serde_json::json!(0));
        assert_eq!(metadata["test_pass_fraction"], serde_json::json!(0.0));
        assert_eq!(metadata["benchmark_exit"], serde_json::Value::Null);
        assert_eq!(metadata["has_benchmark_section"], serde_json::json!(false));
        assert_eq!(metadata["correct"], serde_json::json!(true));
        assert_eq!(metadata["benchmark_mean_count"], serde_json::json!(2));
        assert_eq!(metadata["speed_reward_component"], serde_json::json!(0.0));
    }

    #[test]
    fn reward_metadata_omits_empty_output_tails_for_successful_eval() {
        let r = reward().with_baseline_ns(1000.0);
        let eval = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![500.0],
                geomean_ns: Some(500.0),
                speedup: Some(2.0),
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };

        let training_reward = r.reward_from_extracted_eval(&eval);
        let metadata = r.reward_metadata(
            "def custom_kernel(data): return data",
            &eval,
            training_reward,
        );
        assert_eq!(metadata["sandbox_stdout_len_bytes"], serde_json::json!(0));
        assert_eq!(metadata["sandbox_stderr_len_bytes"], serde_json::json!(0));
        assert_eq!(metadata["training_reward"], serde_json::json!(3.0));
        assert_eq!(metadata["speed_reward_component"], serde_json::json!(2.0));
        assert!(metadata.get("sandbox_stdout_tail").is_none());
        assert!(metadata.get("sandbox_stderr_tail").is_none());
    }

    #[test]
    fn reward_metadata_preserves_output_tail_for_test_process_failure() {
        let r = reward().with_baseline_ns(1000.0);
        let eval = TrimulEval {
            verification: TrimulVerification {
                correct: false,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput {
                stdout: "test-exit: 1\n".to_string(),
                stderr: "RuntimeError: candidate test failed\n".to_string(),
            },
            test_check: None,
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        };

        assert_eq!(
            r.reward_diagnostic(&eval).as_deref(),
            Some("trimul:test_process_failed")
        );
        let training_reward = r.reward_from_extracted_eval(&eval);
        let metadata = r.reward_metadata(
            "def custom_kernel(data): return data",
            &eval,
            training_reward,
        );
        assert_eq!(
            metadata["training_reward"],
            serde_json::json!(RUNNABLE_REWARD)
        );
        assert_eq!(
            metadata["sandbox_stdout_tail"],
            serde_json::json!("test-exit: 1\n")
        );
        assert_eq!(
            metadata["sandbox_stderr_tail"],
            serde_json::json!("RuntimeError: candidate test failed\n")
        );
        assert_eq!(
            metadata["sandbox_stderr_tail_truncated"],
            serde_json::json!(false)
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
    fn build_run_spec_can_pin_verifier_cuda_visibility() {
        let spec = reward()
            .with_verifier_cuda_visible_devices("1")
            .build_run_spec(Path::new("/tmp/scratch"));
        assert!(spec
            .env
            .iter()
            .any(|(k, v)| k == "CUDA_VISIBLE_DEVICES" && v == "1"));

        let default_spec = reward().build_run_spec(Path::new("/tmp/scratch"));
        assert!(!default_spec
            .env
            .iter()
            .any(|(k, _)| k == "CUDA_VISIBLE_DEVICES"));
    }

    #[test]
    fn build_run_spec_assigns_verifier_device_pool_by_worker() {
        let reward = reward()
            .with_verifier_cuda_visible_devices("9")
            .with_verifier_cuda_device_pool(vec![
                " 1 ".to_string(),
                "2".to_string(),
                "".to_string(),
            ]);

        let worker0 = reward.build_run_spec_for_worker(Path::new("/tmp/scratch0"), 0);
        let worker1 = reward.build_run_spec_for_worker(Path::new("/tmp/scratch1"), 1);
        let worker2 = reward.build_run_spec_for_worker(Path::new("/tmp/scratch2"), 2);

        assert!(worker0
            .env
            .iter()
            .any(|(k, v)| k == "CUDA_VISIBLE_DEVICES" && v == "1"));
        assert!(worker1
            .env
            .iter()
            .any(|(k, v)| k == "CUDA_VISIBLE_DEVICES" && v == "2"));
        assert!(worker2
            .env
            .iter()
            .any(|(k, v)| k == "CUDA_VISIBLE_DEVICES" && v == "1"));
    }

    #[test]
    fn build_run_spec_empty_device_pool_keeps_global_visibility() {
        let spec = reward()
            .with_verifier_cuda_visible_devices("9")
            .with_verifier_cuda_device_pool(vec!["".to_string(), "  ".to_string()])
            .build_run_spec_for_worker(Path::new("/tmp/scratch"), 3);

        assert!(spec
            .env
            .iter()
            .any(|(k, v)| k == "CUDA_VISIBLE_DEVICES" && v == "9"));
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
        assert!(cmd.contains("if [ \"$test_rc\" -eq 0 ]; then"));
    }

    #[test]
    fn in_container_command_reports_eval_exit_statuses() {
        let cmd = TrimulReward::in_container_command();
        // The shell reports eval process exits on the same controlled grade channel,
        // so missing grade output can be distinguished from eval process failure.
        assert!(cmd.contains("test_rc=$?"));
        assert!(cmd.contains("echo \"test-exit: $test_rc\""));
        assert!(cmd.contains("bench_rc=$?"));
        assert!(cmd.contains("echo \"benchmark-exit: $bench_rc\""));
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
    fn exit_markers_use_the_last_grade_value() {
        assert_eq!(
            log_i32_value("test-exit: 7\ntest-exit: 0\n", "test-exit"),
            Some(0)
        );
    }

    #[test]
    fn split_result_uses_the_last_separator() {
        let (test, bench) = split_result(
            "noise\n===FERRL-BENCH===\ncheck: pass\ntest-exit: 0\n===FERRL-BENCH===\nbenchmark.0.mean: 5.0\n",
        );
        assert!(test.contains("check: pass"));
        assert!(bench.contains("benchmark.0.mean: 5.0"));
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
        assert_eq!(
            outcomes[0]
                .metadata
                .as_ref()
                .and_then(|m| m.get("submission_extracted")),
            Some(&serde_json::json!(false))
        );
        assert_eq!(
            outcomes[0]
                .metadata
                .as_ref()
                .and_then(|m| m.get("reward_profile"))
                .and_then(|p| p.get("scheme")),
            Some(&serde_json::json!("trimul_shaped_v1"))
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
            output: TrimulEvalOutput::default(),
            test_check: None,
            test_exit: Some(0),
            benchmark_exit: None,
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
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: None,
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
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
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
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };
        assert_eq!(
            r.reward_diagnostic(&implausible_benchmark).as_deref(),
            Some("trimul:implausible_benchmark")
        );
    }

    #[test]
    fn reward_diagnostic_classifies_eval_process_failures() {
        let r = reward();
        let test_process_failed = TrimulEval {
            verification: TrimulVerification {
                correct: false,
                benchmark_means_ns: Vec::new(),
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: None,
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        };
        assert_eq!(
            r.reward_diagnostic(&test_process_failed).as_deref(),
            Some("trimul:test_process_failed")
        );

        let test_process_failed_after_pass_grade = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![],
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        };
        assert_eq!(
            r.reward_diagnostic(&test_process_failed_after_pass_grade)
                .as_deref(),
            Some("trimul:test_process_failed")
        );

        let benchmark_process_failed = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![],
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(2),
            has_benchmark_section: true,
        };
        assert_eq!(
            r.reward_diagnostic(&benchmark_process_failed).as_deref(),
            Some("trimul:benchmark_process_failed")
        );

        let plausible_benchmark_process_failed = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![500.0],
                geomean_ns: Some(500.0),
                speedup: Some(2.0),
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(1),
            has_benchmark_section: true,
        };
        assert_eq!(
            r.reward_from_eval(&plausible_benchmark_process_failed),
            CORRECTNESS_REWARD
        );
        assert_eq!(
            r.reward_diagnostic(&plausible_benchmark_process_failed)
                .as_deref(),
            Some("trimul:benchmark_process_failed")
        );
    }

    #[test]
    fn reward_diagnostic_classifies_shape_test_failures() {
        let r = reward();
        let base = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![],
                geomean_ns: None,
                speedup: None,
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        };

        let shape_mismatch = TrimulEval {
            output: TrimulEvalOutput {
                stdout: String::new(),
                stderr: "RuntimeError: mat1 and mat2 shapes cannot be multiplied".to_string(),
            },
            ..base.clone()
        };
        assert_eq!(r.reward_from_eval(&shape_mismatch), RUNNABLE_REWARD);
        assert_eq!(
            r.reward_diagnostic(&shape_mismatch).as_deref(),
            Some("trimul:test_shape_mismatch")
        );

        let norm_shape_mismatch = TrimulEval {
            output: TrimulEvalOutput {
                stdout: String::new(),
                stderr: "RuntimeError: Expected weight to be of same shape as normalized_shape"
                    .to_string(),
            },
            ..base
        };
        assert_eq!(
            r.reward_diagnostic(&norm_shape_mismatch).as_deref(),
            Some("trimul:test_shape_mismatch")
        );
    }

    #[test]
    fn reward_requires_test_exit_and_benchmark_marker_for_correctness_floor() {
        let r = reward();
        let missing_test_exit = TrimulEval {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![500.0],
                geomean_ns: Some(500.0),
                speedup: Some(2.0),
            },
            status: RunStatus::Exited(0),
            output: TrimulEvalOutput::default(),
            test_check: Some("pass".to_string()),
            test_exit: None,
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };
        assert_eq!(r.reward_from_eval(&missing_test_exit), 0.0);
        assert_eq!(
            r.reward_diagnostic(&missing_test_exit).as_deref(),
            Some("trimul:missing_test_exit")
        );

        let missing_benchmark_exit = TrimulEval {
            test_exit: Some(0),
            benchmark_exit: None,
            ..missing_test_exit.clone()
        };
        assert!(r.reward_from_eval(&missing_benchmark_exit) < CORRECTNESS_REWARD);
        assert_eq!(
            r.reward_diagnostic(&missing_benchmark_exit).as_deref(),
            Some("trimul:missing_benchmark_exit")
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
}
