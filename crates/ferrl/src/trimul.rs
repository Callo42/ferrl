//! TriMul kernel-discovery task — ferrl's first *discovery* [`RewardFn`].
//!
//! The policy is asked to write a faster GPU kernel for the **Triangle
//! Multiplicative Update** (the GPUMODE `bioml/trimul` task — a core AlphaFold-family
//! operator). Each completion is expected to contain a Python `custom_kernel`
//! implementation; this reward **runs it** and scores it on **correctness** and
//! versioned ferrl **service latency**.
//!
//! ## Flow (per candidate)
//!
//! 1. Extract the `custom_kernel` source from the completion according to the
//!    configured [`SubmissionExtractMode`] — the final fenced Python code block, or
//!    for thinking prompts, the final fenced block after `</think>`.
//! 2. Capture the candidate plus generated test/benchmark specs ([`render_spec`])
//!    in kernel-sealed descriptors. They stay read-only and identical across both
//!    phases; scratch contains only writable cache/output state.
//! 3. Send the eval through the explicitly selected verifier tier. The default
//!    `same_uid_apptainer_v1` tier stages sealed assets in a private mode-`0700`
//!    request root and launches Apptainer without administrator setup. The optional
//!    `dedicated_uid_service_v1` tier sends sealed descriptors to a distinct-UID
//!    [`crate::verifier_executor::VerifierExecutorSandbox`]. In both tiers, the pinned
//!    GPUMODE eval files, Ferrl driver, candidate, and specs are bound **read-only**,
//!    scratch is **read-write**, the GPU is exposed (`--nv`), and the **network is
//!    denied**. The sealed driver runs `test` then `benchmark`. A separate
//!    non-dumpable controller owns trusted protocol connections, while candidate Python
//!    lives in its own spawned payload process with independent CUDA allocations; only
//!    bounded CPU bytes cross process boundaries, so no parent allocator block is exported.
//!    The protected parent owns hidden-seed input generation and measures the versioned
//!    end-to-end candidate service latency from byte handoff through result receipt.
//!    Its grade travels over a post-launch Unix socket
//!    owned only by the non-dumpable parent. Launcher/init/shell stdout descriptors are
//!    untrusted diagnostics and can never become a grade.
//! 4. Map the captured grade to a shaped training reward: missing submissions score
//!    `0`, extracted-but-broken submissions get only a tiny format reward, runnable
//!    candidates get a small floor, partially correct candidates scale below the
//!    correctness floor, and test-passing candidates whose eval reaches a benchmark
//!    exit marker score the correctness floor plus any capped latency component.
//!    Implausibly fast timings (below the configured floor — a glitch or forged
//!    grade) still score `0`. The final artifact gate remains stricter than the
//!    training reward: secret-seed re-verification of the launch-bound cases plus a
//!    repeated same-metric latency audit. A distinct genuinely held-out case/reward
//!    boundary is outside this contract.
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
//! Verifier, candidate, and rendered-case bytes remain kernel-sealed and read-only
//! through correctness and benchmark execution. Candidate code never shares memory
//! with trusted checker/timer state. The grade and actual-import proof ride an
//! exclusive post-launch socket owned by a non-dumpable trusted parent. The payload
//! enters with no active capabilities, `NoNewPrivs`, a TSYNC seccomp deny policy, and
//! AF_UNIX-only socket creation. An implausibly fast service latency is rejected, so
//! forged scratch files, printed passes, and zero-time kernels cannot reach the
//! correctness floor. These controls constrain code inside the candidate process; the
//! same-UID tier does **not** resist an arbitrary malicious peer process already running
//! under the training account. Accepted same-UID artifact publication retains that
//! explicit operator-trust boundary; the dedicated tier remains an optional
//! higher-isolation backend and is not a no-administrator prerequisite.
//!
//! ## Testing split (as in [`crate::sandbox`])
//!
//! The pure pieces — submission extraction, spec rendering, result parsing, the
//! run-spec builder, and the reward math — are unit-tested in CI. The real GPU eval
//! is a `gate`-feature integration test (`tests/trimul_gate.rs`), run on an `sm_80`
//! node against the eval image. CI clippies that target and runs its non-ignored
//! source-contract test; only the ignored Apptainer/GPU controls are deployment-only.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::reward::{RewardError, RewardFn, RewardOutcome};
use crate::sample::Sample;
use crate::sandbox::{
    Bind, ProtectedOutput, ResourceLimits, RunOutcome, RunSpec, RunStatus, Sandbox, SandboxError,
};
use crate::verifier_executor::{
    SameUidApptainerSandbox, VerifierExecutorSandbox, VerifierIsolationEvidence,
    VerifierIsolationTier,
};
#[cfg(test)]
use crate::verifier_executor::{
    VerifierAssetTransport, VerifierUidBoundary, VERIFIER_ISOLATION_EVIDENCE_VERSION,
};

#[derive(Debug, Clone)]
enum TrimulVerifierBackend {
    SameUid(SameUidApptainerSandbox),
    Dedicated(VerifierExecutorSandbox),
}

impl TrimulVerifierBackend {
    const fn tier(&self) -> VerifierIsolationTier {
        match self {
            Self::SameUid(_) => VerifierIsolationTier::SameUidApptainerV1,
            Self::Dedicated(_) => VerifierIsolationTier::DedicatedUidServiceV1,
        }
    }

    fn preflight(&self) -> Result<VerifierIsolationEvidence, SandboxError> {
        match self {
            Self::SameUid(sandbox) => sandbox.preflight(),
            Self::Dedicated(sandbox) => sandbox.preflight(),
        }
    }

    fn run(&self, spec: &RunSpec) -> Result<RunOutcome, SandboxError> {
        match self {
            Self::SameUid(sandbox) => sandbox.run(spec),
            Self::Dedicated(sandbox) => sandbox.run(spec),
        }
    }
}

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

/// The per-case mean service latencies (nanoseconds) from a `benchmark`-mode result
/// log: every `benchmark.<i>.mean` value.
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

fn candidate_attempt_sentinels(log: &str) -> Vec<&str> {
    log.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(": ")?;
            key.trim()
                .ends_with(".candidate-sentinel")
                .then_some(value.trim())
        })
        .collect()
}

/// The geometric mean of `xs`, or `None` if empty or any value is non-positive.
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

/// Content identity of the immutable verifier assets used by a TriMul launch.
///
/// The image digest binds the exact container bytes. The bundle digest binds every
/// relative regular-file name and byte in the eval tree; `task.yml` is also named
/// separately because it selects the concrete correctness and benchmark cases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrimulVerifierIdentity {
    /// SHA-256 of the exact sandbox image bytes.
    pub image_sha256: String,
    /// Length of the exact sandbox image.
    pub image_len_bytes: u64,
    /// SHA-256 of the complete ordered eval bundle.
    pub eval_bundle_sha256: String,
    /// Number of regular files bound by `eval_bundle_sha256`.
    pub eval_file_count: usize,
    /// SHA-256 of the exact `task.yml` bytes used to construct cases.
    pub task_yml_sha256: String,
    /// Length of the exact `task.yml` bytes.
    pub task_yml_len_bytes: usize,
}

/// Failure while capturing or revalidating immutable TriMul verifier assets.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrimulAssetError {
    /// A verifier asset could not be read or sealed.
    #[error("TriMul verifier asset I/O failed at {path}: {source}")]
    Io {
        /// Asset path being accessed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The asset tree or its live identity violated the immutable contract.
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    ctime: i64,
    #[cfg(unix)]
    ctime_nsec: i64,
}

impl FileStamp {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt as _;

        Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            dev: metadata.dev(),
            #[cfg(unix)]
            ino: metadata.ino(),
            #[cfg(unix)]
            ctime: metadata.ctime(),
            #[cfg(unix)]
            ctime_nsec: metadata.ctime_nsec(),
        }
    }
}

#[derive(Debug)]
struct EvalBundleSnapshot {
    files: BTreeMap<PathBuf, Vec<u8>>,
    identity: String,
}

#[derive(Debug)]
struct SealedEvalFile {
    relative_path: PathBuf,
    file: File,
    len: u64,
}

impl SealedEvalFile {
    fn verify(&self) -> Result<(), TrimulAssetError> {
        verify_sealed_asset(&self.file, &self.relative_path, self.len)
    }

    fn descriptor_path(&self) -> PathBuf {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd as _;
            PathBuf::from(format!(
                "/proc/{}/fd/{}",
                std::process::id(),
                self.file.as_raw_fd()
            ))
        }
        #[cfg(not(target_os = "linux"))]
        {
            PathBuf::new()
        }
    }
}

#[derive(Debug)]
struct VerifierAssetSnapshot {
    image_path: PathBuf,
    image_file: File,
    image_stamp: FileStamp,
    eval_dir: PathBuf,
    sealed_eval_files: Vec<SealedEvalFile>,
    task_yml: String,
    identity: TrimulVerifierIdentity,
}

const SANDBOX_EVAL_FILES: [(&str, &str); 4] = [
    ("eval.py", "/opt/ferrl-verifier/eval.py"),
    ("reference.py", "/opt/ferrl-verifier/reference.py"),
    ("task.py", "/opt/ferrl-verifier/task.py"),
    ("utils.py", "/opt/ferrl-verifier/utils.py"),
];

const FERRL_EVAL_DRIVER_PATH: &str = "/opt/ferrl-verifier/ferrl_eval.py";
const SUBMISSION_PATH: &str = "/opt/ferrl-verifier/submission.py";
const TEST_SPEC_PATH: &str = "/opt/ferrl-verifier/test_spec.txt";
const BENCH_SPEC_PATH: &str = "/opt/ferrl-verifier/bench_spec.txt";

/// Ferrl-owned immutable verifier driver. The protected parent owns trusted input
/// generation, checking, timing, statistics, and the machine-grade socket. Candidate
/// Python runs only in fresh spawned children and can influence the parent solely
/// through checked CUDA buffers plus a bounded raw status protocol.
const FERRL_EVAL_DRIVER: &str = include_str!("trimul_eval_driver.py");

/// Stable verifier assets captured before launch attestation.
///
/// The image and every file in the eval tree are copied byte-for-byte into anonymous
/// Linux memfds with write/grow/shrink/further-seal operations permanently disabled
/// by the kernel. After sealing, the image descriptor and complete ordered eval
/// descriptor set are hashed again and compared with the identities captured from
/// their sources. Exact sealed descriptors, never writable snapshot pathnames, supply
/// every verifier invocation. The sandbox copies only from those handles into its
/// private tmpfs, remounts it read-only, and authenticates the final bytes before use.
/// Later checks inspect source identities and kernel seals without repeatedly
/// rehashing the potentially multi-gigabyte image.
#[derive(Debug, Clone)]
pub struct TrimulVerifierAssets {
    snapshot: Arc<VerifierAssetSnapshot>,
}

impl TrimulVerifierAssets {
    /// Capture exact image and eval-bundle content. The legacy scratch-root
    /// argument is retained for API compatibility but no verifier asset is staged
    /// there: eval files live only in anonymous kernel-sealed descriptors.
    ///
    /// # Errors
    ///
    /// Returns [`TrimulAssetError`] if an asset is missing, non-regular, changes
    /// during capture, contains a symlink/special entry, or cannot be kernel-sealed.
    pub fn capture(
        image_path: impl AsRef<Path>,
        eval_dir: impl AsRef<Path>,
        _scratch_root: impl AsRef<Path>,
    ) -> Result<Self, TrimulAssetError> {
        let image_path = image_path.as_ref().to_path_buf();
        let eval_dir = eval_dir.as_ref().to_path_buf();
        let (image_file, image_stamp, image_sha256) = capture_image(&image_path)?;
        let eval = capture_eval_bundle(&eval_dir)?;
        let task_yml_bytes = eval.files.get(Path::new("task.yml")).ok_or_else(|| {
            TrimulAssetError::Invalid(format!(
                "TriMul eval bundle {} has no regular task.yml",
                eval_dir.display()
            ))
        })?;
        let task_yml = std::str::from_utf8(task_yml_bytes)
            .map_err(|error| {
                TrimulAssetError::Invalid(format!("TriMul task.yml is not UTF-8: {error}"))
            })?
            .to_owned();
        for (relative, _) in SANDBOX_EVAL_FILES {
            if !eval.files.contains_key(Path::new(relative)) {
                return Err(TrimulAssetError::Invalid(format!(
                    "TriMul eval bundle {} has no regular {relative}",
                    eval_dir.display()
                )));
            }
        }
        let sealed_eval_files = seal_eval_bundle(&eval.files, &eval.identity)?;
        let identity = TrimulVerifierIdentity {
            image_sha256,
            image_len_bytes: image_stamp.len,
            eval_bundle_sha256: eval.identity,
            eval_file_count: eval.files.len(),
            task_yml_sha256: sha256_hex(task_yml_bytes),
            task_yml_len_bytes: task_yml_bytes.len(),
        };
        let assets = Self {
            snapshot: Arc::new(VerifierAssetSnapshot {
                image_path,
                image_file,
                image_stamp,
                eval_dir,
                sealed_eval_files,
                task_yml,
                identity,
            }),
        };
        assets.verify_current()?;
        Ok(assets)
    }

    /// Exact portable content identity bound into `launch.json`.
    #[must_use]
    pub fn identity(&self) -> &TrimulVerifierIdentity {
        &self.snapshot.identity
    }

    /// Exact captured `task.yml` text used to construct the reward cases.
    #[must_use]
    pub fn task_yml(&self) -> &str {
        &self.snapshot.task_yml
    }

    /// Revalidate configured sources and sealed assets without changing which bytes
    /// subsequent verifier invocations consume.
    ///
    /// # Errors
    ///
    /// Returns [`TrimulAssetError`] after any path substitution or in-place change.
    pub fn verify_current(&self) -> Result<(), TrimulAssetError> {
        let image_path_metadata = regular_metadata(&self.snapshot.image_path)?;
        if FileStamp::from_metadata(&image_path_metadata) != self.snapshot.image_stamp {
            return Err(TrimulAssetError::Invalid(
                "TriMul sandbox image changed after verifier attestation".to_string(),
            ));
        }
        verify_sealed_asset(
            &self.snapshot.image_file,
            &self.snapshot.image_path,
            self.snapshot.identity.image_len_bytes,
        )?;
        let current_eval = capture_eval_bundle(&self.snapshot.eval_dir)?;
        for file in &self.snapshot.sealed_eval_files {
            file.verify()?;
        }
        if current_eval.identity != self.snapshot.identity.eval_bundle_sha256
            || current_eval.files.len() != self.snapshot.identity.eval_file_count
        {
            return Err(TrimulAssetError::Invalid(
                "TriMul eval bundle changed after verifier attestation".to_string(),
            ));
        }
        Ok(())
    }

    fn image_for_sandbox(&self) -> PathBuf {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd as _;
            PathBuf::from(format!(
                "/proc/{}/fd/{}",
                std::process::id(),
                self.snapshot.image_file.as_raw_fd()
            ))
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.snapshot.image_path.clone()
        }
    }

    fn eval_binds(&self) -> Vec<Bind> {
        SANDBOX_EVAL_FILES
            .iter()
            .map(|(relative, destination)| {
                let sealed = self
                    .snapshot
                    .sealed_eval_files
                    .iter()
                    .find(|file| file.relative_path == Path::new(relative))
                    .expect("capture validates every sandbox eval file");
                Bind::ro(sealed.descriptor_path(), *destination)
            })
            .collect()
    }
}

/// Per-candidate bytes consumed by the verifier. These descriptors live until
/// [`Sandbox::run`] returns; every bind is read-only, so test and benchmark see
/// the exact same candidate and rendered case sets.
#[derive(Debug)]
struct SealedInvocationAssets {
    files: Vec<(SealedEvalFile, &'static str)>,
}

impl SealedInvocationAssets {
    #[cfg(target_os = "linux")]
    fn capture(
        submission: &str,
        test_spec: &str,
        bench_spec: &str,
    ) -> Result<Self, TrimulAssetError> {
        let files = [
            (
                "ferrl_eval.py",
                FERRL_EVAL_DRIVER.as_bytes(),
                FERRL_EVAL_DRIVER_PATH,
            ),
            ("submission.py", submission.as_bytes(), SUBMISSION_PATH),
            ("test_spec.txt", test_spec.as_bytes(), TEST_SPEC_PATH),
            ("bench_spec.txt", bench_spec.as_bytes(), BENCH_SPEC_PATH),
        ]
        .into_iter()
        .map(|(name, bytes, destination)| {
            seal_invocation_asset(Path::new(name), bytes).map(|sealed| (sealed, destination))
        })
        .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { files })
    }

    #[cfg(not(target_os = "linux"))]
    fn capture(
        _submission: &str,
        _test_spec: &str,
        _bench_spec: &str,
    ) -> Result<Self, TrimulAssetError> {
        Err(TrimulAssetError::Invalid(
            "TriMul invocation assets require Linux kernel-sealed memfd storage".to_string(),
        ))
    }

    fn binds(&self) -> Vec<Bind> {
        self.files
            .iter()
            .map(|(file, destination)| Bind::ro(file.descriptor_path(), *destination))
            .collect()
    }

    fn verify(&self) -> Result<(), TrimulAssetError> {
        self.files.iter().try_for_each(|(file, _)| file.verify())
    }
}

#[cfg(target_os = "linux")]
fn seal_invocation_asset(path: &Path, bytes: &[u8]) -> Result<SealedEvalFile, TrimulAssetError> {
    use std::io::Write as _;

    let descriptor = rustix::fs::memfd_create(
        path,
        rustix::fs::MemfdFlags::ALLOW_SEALING | rustix::fs::MemfdFlags::CLOEXEC,
    )
    .map_err(|source| TrimulAssetError::Io {
        path: path.to_path_buf(),
        source: source.into(),
    })?;
    let mut file = descriptor_file_above_stdio(File::from(descriptor), true).map_err(|source| {
        TrimulAssetError::Io {
            path: path.to_path_buf(),
            source,
        }
    })?;
    file.write_all(bytes)
        .map_err(|source| TrimulAssetError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    rustix::fs::fcntl_add_seals(&file, required_asset_seals()).map_err(|source| {
        TrimulAssetError::Io {
            path: path.to_path_buf(),
            source: source.into(),
        }
    })?;
    let sealed = SealedEvalFile {
        relative_path: path.to_path_buf(),
        file,
        len: bytes.len() as u64,
    };
    authenticate_sealed_asset(
        &sealed.file,
        &sealed.relative_path,
        sealed.len,
        &sha256_hex(bytes),
    )?;
    Ok(sealed)
}

fn regular_metadata(path: &Path) -> Result<std::fs::Metadata, TrimulAssetError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| TrimulAssetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file() {
        return Err(TrimulAssetError::Invalid(format!(
            "TriMul verifier asset {} is not a regular file",
            path.display()
        )));
    }
    Ok(metadata)
}

#[cfg(target_os = "linux")]
fn capture_image(path: &Path) -> Result<(File, FileStamp, String), TrimulAssetError> {
    capture_image_with_hook(path, |_| {})
}

#[cfg(target_os = "linux")]
fn capture_image_with_hook(
    path: &Path,
    before_seal: impl FnOnce(&File),
) -> Result<(File, FileStamp, String), TrimulAssetError> {
    use std::io::{Read as _, Write as _};

    let path_metadata = regular_metadata(path)?;
    let mut source_file = File::open(path).map_err(|source| TrimulAssetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let before = FileStamp::from_metadata(&source_file.metadata().map_err(|source| {
        TrimulAssetError::Io {
            path: path.to_path_buf(),
            source,
        }
    })?);
    if before != FileStamp::from_metadata(&path_metadata) {
        return Err(TrimulAssetError::Invalid(format!(
            "TriMul sandbox image {} changed while it was opened",
            path.display()
        )));
    }
    let descriptor = rustix::fs::memfd_create(
        "ferrl-trimul-image",
        rustix::fs::MemfdFlags::ALLOW_SEALING | rustix::fs::MemfdFlags::CLOEXEC,
    )
    .map_err(|source| TrimulAssetError::Io {
        path: path.to_path_buf(),
        source: source.into(),
    })?;
    let mut sealed_file =
        descriptor_file_above_stdio(File::from(descriptor), true).map_err(|source| {
            TrimulAssetError::Io {
                path: path.to_path_buf(),
                source,
            }
        })?;
    let mut digest = Sha256::new();
    let mut copied = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = source_file
            .read(&mut buffer)
            .map_err(|source| TrimulAssetError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        sealed_file
            .write_all(&buffer[..read])
            .map_err(|source| TrimulAssetError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        copied += read as u64;
        digest.update(&buffer[..read]);
    }
    let after = FileStamp::from_metadata(&source_file.metadata().map_err(|source| {
        TrimulAssetError::Io {
            path: path.to_path_buf(),
            source,
        }
    })?);
    if before != after || copied != before.len {
        return Err(TrimulAssetError::Invalid(format!(
            "TriMul sandbox image {} changed while it was captured",
            path.display()
        )));
    }
    let source_sha256 = format!("{:x}", digest.finalize());
    before_seal(&sealed_file);
    rustix::fs::fcntl_add_seals(&sealed_file, required_asset_seals()).map_err(|source| {
        TrimulAssetError::Io {
            path: path.to_path_buf(),
            source: source.into(),
        }
    })?;
    authenticate_sealed_asset(&sealed_file, path, copied, &source_sha256)?;
    Ok((sealed_file, before, source_sha256))
}

#[cfg(not(target_os = "linux"))]
fn capture_image(_path: &Path) -> Result<(File, FileStamp, String), TrimulAssetError> {
    Err(TrimulAssetError::Invalid(
        "TriMul verifier assets require Linux kernel-sealed memfd storage".to_string(),
    ))
}

fn capture_eval_bundle(root: &Path) -> Result<EvalBundleSnapshot, TrimulAssetError> {
    let metadata = std::fs::symlink_metadata(root).map_err(|source| TrimulAssetError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_dir() {
        return Err(TrimulAssetError::Invalid(format!(
            "TriMul eval bundle {} is not a directory",
            root.display()
        )));
    }
    let mut pending = vec![PathBuf::new()];
    let mut files = BTreeMap::new();
    while let Some(relative_dir) = pending.pop() {
        let directory = root.join(&relative_dir);
        let entries = std::fs::read_dir(&directory).map_err(|source| TrimulAssetError::Io {
            path: directory.clone(),
            source,
        })?;
        let mut entries =
            entries
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| TrimulAssetError::Io {
                    path: directory.clone(),
                    source,
                })?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let relative = relative_dir.join(entry.file_name());
            let path = entry.path();
            let metadata =
                std::fs::symlink_metadata(&path).map_err(|source| TrimulAssetError::Io {
                    path: path.clone(),
                    source,
                })?;
            if metadata.file_type().is_dir() {
                pending.push(relative);
            } else if metadata.file_type().is_file() {
                let before = FileStamp::from_metadata(&metadata);
                let bytes = std::fs::read(&path).map_err(|source| TrimulAssetError::Io {
                    path: path.clone(),
                    source,
                })?;
                let after = FileStamp::from_metadata(&regular_metadata(&path)?);
                if before != after || bytes.len() as u64 != before.len {
                    return Err(TrimulAssetError::Invalid(format!(
                        "TriMul eval asset {} changed while it was read",
                        path.display()
                    )));
                }
                files.insert(relative, bytes);
            } else {
                return Err(TrimulAssetError::Invalid(format!(
                    "TriMul eval bundle contains a symlink or special entry at {}",
                    path.display()
                )));
            }
        }
    }
    if files.is_empty() {
        return Err(TrimulAssetError::Invalid(format!(
            "TriMul eval bundle {} contains no regular files",
            root.display()
        )));
    }
    let identity = eval_bundle_sha256(&files)?;
    Ok(EvalBundleSnapshot { files, identity })
}

fn eval_bundle_sha256(files: &BTreeMap<PathBuf, Vec<u8>>) -> Result<String, TrimulAssetError> {
    let mut digest = Sha256::new();
    digest.update(b"ferrl.trimul.eval-bundle.v1\0");
    for (path, bytes) in files {
        let path = path.to_str().ok_or_else(|| {
            TrimulAssetError::Invalid("TriMul eval bundle paths must be UTF-8".to_string())
        })?;
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(target_os = "linux")]
fn required_asset_seals() -> rustix::fs::SealFlags {
    rustix::fs::SealFlags::WRITE
        | rustix::fs::SealFlags::GROW
        | rustix::fs::SealFlags::SHRINK
        | rustix::fs::SealFlags::SEAL
}

/// memfd creation follows the ordinary lowest-free-fd rule. A launcher with
/// closed stdio can therefore receive fd 0, 1, or 2; duplicate such handles
/// above the standard range while preserving whether they must cross exec.
#[cfg(target_os = "linux")]
fn descriptor_file_above_stdio(file: File, cloexec: bool) -> std::io::Result<File> {
    use std::os::fd::AsRawFd as _;

    if file.as_raw_fd() > 2 {
        return Ok(file);
    }
    let duplicated = rustix::io::fcntl_dupfd_cloexec(&file, 3).map_err(std::io::Error::from)?;
    if !cloexec {
        rustix::io::fcntl_setfd(&duplicated, rustix::io::FdFlags::empty())
            .map_err(std::io::Error::from)?;
    }
    Ok(File::from(duplicated))
}

fn verify_sealed_asset(file: &File, path: &Path, len: u64) -> Result<(), TrimulAssetError> {
    let metadata = file.metadata().map_err(|source| TrimulAssetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() != len {
        return Err(TrimulAssetError::Invalid(format!(
            "sealed TriMul verifier asset {} changed length",
            path.display()
        )));
    }
    #[cfg(target_os = "linux")]
    {
        let seals = rustix::fs::fcntl_get_seals(file).map_err(|source| TrimulAssetError::Io {
            path: path.to_path_buf(),
            source: source.into(),
        })?;
        if !seals.contains(required_asset_seals()) {
            return Err(TrimulAssetError::Invalid(format!(
                "TriMul verifier asset {} is not kernel-sealed",
                path.display()
            )));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn update_digest_from_sealed_asset(
    digest: &mut Sha256,
    file: &File,
    path: &Path,
    len: u64,
) -> Result<(), TrimulAssetError> {
    use std::os::unix::fs::FileExt as _;

    verify_sealed_asset(file, path, len)?;
    let mut offset = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    while offset < len {
        let remaining = len - offset;
        let chunk_len = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("bounded descriptor-hash chunk fits usize");
        let read = file
            .read_at(&mut buffer[..chunk_len], offset)
            .map_err(|source| TrimulAssetError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            return Err(TrimulAssetError::Invalid(format!(
                "sealed TriMul verifier asset {} ended before its authenticated length",
                path.display()
            )));
        }
        digest.update(&buffer[..read]);
        offset += read as u64;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn sealed_asset_sha256(file: &File, path: &Path, len: u64) -> Result<String, TrimulAssetError> {
    let mut digest = Sha256::new();
    update_digest_from_sealed_asset(&mut digest, file, path, len)?;
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(target_os = "linux")]
fn authenticate_sealed_asset(
    file: &File,
    path: &Path,
    len: u64,
    expected_sha256: &str,
) -> Result<(), TrimulAssetError> {
    if sealed_asset_sha256(file, path, len)? != expected_sha256 {
        return Err(TrimulAssetError::Invalid(format!(
            "sealed TriMul verifier asset {} does not match its captured identity",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn sealed_eval_bundle_sha256(files: &[SealedEvalFile]) -> Result<String, TrimulAssetError> {
    let mut digest = Sha256::new();
    digest.update(b"ferrl.trimul.eval-bundle.v1\0");
    for file in files {
        let path = file.relative_path.to_str().ok_or_else(|| {
            TrimulAssetError::Invalid("TriMul eval bundle paths must be UTF-8".to_string())
        })?;
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
        digest.update(file.len.to_le_bytes());
        update_digest_from_sealed_asset(&mut digest, &file.file, &file.relative_path, file.len)?;
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(target_os = "linux")]
fn seal_eval_bundle(
    files: &BTreeMap<PathBuf, Vec<u8>>,
    expected_sha256: &str,
) -> Result<Vec<SealedEvalFile>, TrimulAssetError> {
    seal_eval_bundle_with_hook(files, expected_sha256, |_, _| {})
}

#[cfg(target_os = "linux")]
fn seal_eval_bundle_with_hook(
    files: &BTreeMap<PathBuf, Vec<u8>>,
    expected_sha256: &str,
    mut before_seal: impl FnMut(&Path, &File),
) -> Result<Vec<SealedEvalFile>, TrimulAssetError> {
    use std::io::Write as _;

    let mut sealed = Vec::with_capacity(files.len());
    for (index, (relative_path, bytes)) in files.iter().enumerate() {
        // The protected handoff opens sandbox-consumed assets through the owner
        // process's proc-fd path, authenticates their private read-only copies, and
        // closes every source on exec. No verifier descriptor is inherited by
        // launcher, init, shell, or candidate processes.
        let descriptor = rustix::fs::memfd_create(
            format!("ferrl-trimul-eval-{index}"),
            rustix::fs::MemfdFlags::ALLOW_SEALING | rustix::fs::MemfdFlags::CLOEXEC,
        )
        .map_err(|source| TrimulAssetError::Io {
            path: relative_path.clone(),
            source: source.into(),
        })?;
        let mut file =
            descriptor_file_above_stdio(File::from(descriptor), true).map_err(|source| {
                TrimulAssetError::Io {
                    path: relative_path.clone(),
                    source,
                }
            })?;
        file.write_all(bytes)
            .map_err(|source| TrimulAssetError::Io {
                path: relative_path.clone(),
                source,
            })?;
        before_seal(relative_path, &file);
        rustix::fs::fcntl_add_seals(&file, required_asset_seals()).map_err(|source| {
            TrimulAssetError::Io {
                path: relative_path.clone(),
                source: source.into(),
            }
        })?;
        let sealed_file = SealedEvalFile {
            relative_path: relative_path.clone(),
            file,
            len: bytes.len() as u64,
        };
        sealed_file.verify()?;
        sealed.push(sealed_file);
    }
    if sealed_eval_bundle_sha256(&sealed)? != expected_sha256 {
        return Err(TrimulAssetError::Invalid(
            "sealed TriMul eval bundle does not match its captured identity".to_string(),
        ));
    }
    Ok(sealed)
}

#[cfg(not(target_os = "linux"))]
fn seal_eval_bundle(
    _files: &BTreeMap<PathBuf, Vec<u8>>,
    _expected_sha256: &str,
) -> Result<Vec<SealedEvalFile>, TrimulAssetError> {
    Err(TrimulAssetError::Invalid(
        "TriMul verifier assets require Linux kernel-sealed memfd storage".to_string(),
    ))
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
/// and scores it on correctness plus versioned service latency. Construct with
/// [`TrimulReward::new`].
#[derive(Debug, Clone)]
pub struct TrimulReward {
    /// Where per-candidate scratch dirs are created — node-local tmpfs is preferred
    /// (e.g. `/dev/shm/ferrl`) so overflow cannot fill persistent host storage.
    scratch_root: PathBuf,
    /// Host-supervised total byte cap for the candidate-writable `/work` tree.
    scratch_max_bytes: u64,
    /// Correctness cases (GPU Mode's, loaded from the pinned `task.yml`).
    test_cases: Vec<TrimulCase>,
    /// Timing cases.
    benchmark_cases: Vec<TrimulCase>,
    /// The secret seed Cantor-combined with each launch-bound case's public seed
    /// (passed as `POPCORN_SEED`).
    secret_seed: u64,
    /// Reference geometric-mean service latency (ns) on the target GPU; the
    /// same-metric ratio denominator for the shaped reward's latency component. `None`
    /// falls back to an inverse-latency signal.
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
    /// Process cap applied to the verifier sandbox (`ulimit -u`).
    verifier_max_procs: u64,
    /// Explicit verifier backend. Variants never fall back to one another.
    sandbox: TrimulVerifierBackend,
    /// Backend preflight evidence pinned before launch publication. CLI construction
    /// always sets this; direct library callers are preflighted on first execution.
    verifier_isolation_evidence: Option<VerifierIsolationEvidence>,
    /// Protected in-container control probe pinned before launch publication.
    runtime_preflight_evidence: Option<TrimulRuntimePreflightEvidence>,
    /// Immutable verifier assets captured before launch attestation. Construction
    /// cannot produce an execution-capable reward without this owner.
    verifier_assets: TrimulVerifierAssets,
}

/// The parsed result of one sandboxed TriMul eval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrimulVerification {
    /// Whether the eval reported `check: pass`.
    pub correct: bool,
    /// Per-benchmark mean service latencies in nanoseconds, after parsing the grade stream.
    pub benchmark_means_ns: Vec<f64>,
    /// Plausibility-checked geometric-mean service latency (ns), if any.
    pub geomean_ns: Option<f64>,
    /// Same-metric baseline/candidate latency ratio, when both values are present.
    pub speedup: Option<f64>,
}

/// A parsed TriMul result together with backend and protected in-container evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidencedTrimulVerification {
    /// Correctness and timing result.
    pub verification: TrimulVerification,
    /// Exact backend preflight evidence revalidated around the execution.
    pub isolation: VerifierIsolationEvidence,
    /// Domain-separated digest of `isolation`.
    pub isolation_evidence_sha256: String,
    /// Canonical protected hardening records, ordered test then benchmark.
    pub runtime_hardening: Vec<serde_json::Value>,
    /// Domain-separated digest of the ordered raw hardening records.
    pub runtime_hardening_evidence_sha256: String,
    /// Exact sandbox termination status for this verifier execution.
    pub sandbox_status: RunStatus,
    /// Exact protected machine-grade stream received from the trusted verifier.
    pub protected_output: String,
    /// SHA-256 of `protected_output`.
    pub protected_output_sha256: String,
    /// Untrusted sandbox stdout/stderr retained for diagnostics, never grading.
    pub sandbox_diagnostics: String,
    /// SHA-256 of `sandbox_diagnostics`.
    pub sandbox_diagnostics_sha256: String,
}

/// Protected identity of the physical CUDA device used by one TriMul verifier run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrimulExecutingDevice {
    /// Device-evidence schema identifier.
    pub contract: String,
    /// Logical CUDA ordinal seen inside the protected verifier.
    pub cuda_logical_ordinal: u32,
    /// CUDA driver product name.
    pub name: String,
    /// CUDA driver PCI bus identifier.
    pub pci_bus_id: String,
    /// Lowercase 16-byte CUDA UUID without presentation punctuation.
    pub uuid: String,
}

/// One exactly indexed correctness-case result from the protected grade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrimulTestCaseEvidence {
    /// Zero-based case index.
    pub index: usize,
    /// Trusted case specification emitted by the verifier controller.
    pub spec: String,
    /// Whether the trusted checker accepted the candidate output.
    pub passed: bool,
}

/// One exactly indexed benchmark-case result from the protected grade.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrimulBenchmarkCaseEvidence {
    /// Zero-based case index.
    pub index: usize,
    /// Trusted case specification emitted by the verifier controller.
    pub spec: String,
    /// Number of timed candidate invocations summarized by this record.
    pub runs: u64,
    /// Mean protected service latency in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds.
    pub std_ns: f64,
    /// Error estimate in nanoseconds.
    pub err_ns: f64,
    /// Best protected service latency in nanoseconds.
    pub best_ns: f64,
    /// Worst protected service latency in nanoseconds.
    pub worst_ns: f64,
}

/// Strict, publication-grade interpretation of one protected TriMul execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrimulArtifactVerificationEvidence {
    /// Exact successful sandbox termination status.
    pub sandbox_status: RunStatus,
    /// Trusted test-phase process exit marker.
    pub test_exit: i32,
    /// Trusted benchmark-phase process exit marker.
    pub benchmark_exit: i32,
    /// Physical CUDA device used by both phases.
    pub executing_device: TrimulExecutingDevice,
    /// Complete ordered correctness-case evidence.
    pub test_cases: Vec<TrimulTestCaseEvidence>,
    /// Complete ordered benchmark-case evidence.
    pub benchmark_cases: Vec<TrimulBenchmarkCaseEvidence>,
    /// SHA-256 of the exact protected grade retained in the artifact.
    pub protected_output_sha256: String,
    /// SHA-256 of retained untrusted sandbox diagnostics.
    pub sandbox_diagnostics_sha256: String,
}

/// Launch-time control probe produced through the exact staged verifier path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrimulRuntimePreflightEvidence {
    /// Evidence schema version.
    pub contract_version: u32,
    /// Selected verifier tier.
    pub isolation_tier: VerifierIsolationTier,
    /// Digest of the backend preflight evidence used by the probe.
    pub isolation_evidence_sha256: String,
    /// SHA-256 of Ferrl's fixed probe submission.
    pub probe_submission_sha256: String,
    /// Canonical protected hardening records produced before probe candidate entry.
    pub runtime_hardening: Vec<serde_json::Value>,
    /// Digest of the ordered protected records.
    pub runtime_hardening_evidence_sha256: String,
}

/// A `custom_kernel` that delegates to the bundled reference implementation. Used to
/// **measure the service-latency baseline**: the reference is correct by definition, so
/// it passes correctness and its benchmark record supplies the reference latency. The
/// `reference` module is bound read-only next to the sealed submission. This is the
/// extracted code, not a fenced block — it is fed straight to the eval path,
/// bypassing [`extract_submission`].
const REFERENCE_SUBMISSION: &str =
    "def custom_kernel(data):\n    from reference import ref_kernel\n    return ref_kernel(data)\n";
const HARDENING_PREFLIGHT_SUBMISSION: &str =
    "def custom_kernel(data):\n    return data[0].clone()\n";

/// Default process cap for a TriMul verifier sandbox.
///
/// The cap is finite for fork-bomb containment, but higher than the generic
/// sandbox default because `ulimit -u`/`RLIMIT_NPROC` is per UID. Multi-process
/// GPU training jobs can already exceed small caps before candidate code starts.
pub const DEFAULT_VERIFIER_MAX_PROCS: u64 = 1024;

impl TrimulReward {
    /// Construct from captured immutable verifier `assets` and a node-local
    /// `scratch_root`. Cases default to empty — set them with
    /// [`with_cases`](Self::with_cases) (they are GPU Mode's, kept out of this repo).
    ///
    /// Requiring the descriptor owner here prevents an apparently valid reward or
    /// run specification whose verifier handles were never captured or have already
    /// been dropped.
    #[must_use]
    pub fn new(assets: TrimulVerifierAssets, scratch_root: impl Into<PathBuf>) -> Self {
        let scratch_root = scratch_root.into();
        Self {
            sandbox: TrimulVerifierBackend::SameUid(SameUidApptainerSandbox::new(
                scratch_root.join(".ferrl-verifier"),
            )),
            verifier_isolation_evidence: None,
            runtime_preflight_evidence: None,
            scratch_root,
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
            verifier_max_procs: DEFAULT_VERIFIER_MAX_PROCS,
            verifier_assets: assets,
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

    /// Override the protected verifier executor socket.
    #[must_use]
    pub fn with_verifier_executor_socket(mut self, socket: impl Into<PathBuf>) -> Self {
        self.sandbox = TrimulVerifierBackend::Dedicated(VerifierExecutorSandbox::new(socket));
        self.verifier_isolation_evidence = None;
        self.runtime_preflight_evidence = None;
        self
    }

    /// Select the no-admin same-UID staged Apptainer backend explicitly.
    #[must_use]
    pub fn with_same_uid_apptainer(
        mut self,
        work_root: impl Into<PathBuf>,
        apptainer_bin: impl Into<PathBuf>,
    ) -> Self {
        self.sandbox = TrimulVerifierBackend::SameUid(
            SameUidApptainerSandbox::new(work_root).with_apptainer_bin(apptainer_bin),
        );
        self.verifier_isolation_evidence = None;
        self.runtime_preflight_evidence = None;
        self
    }

    /// Preflight and pin the exact verifier backend identity used by this reward.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] when the selected tier is unavailable or cannot prove
    /// its declared runtime identity. No alternate tier is attempted.
    pub fn with_verified_isolation(mut self) -> Result<Self, RewardError> {
        self.verifier_isolation_evidence =
            Some(self.sandbox.preflight().map_err(RewardError::verifier)?);
        self.runtime_preflight_evidence = None;
        Ok(self)
    }

    /// Run and pin the protected in-container control probe.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] when required controls cannot be proved before launch.
    pub fn with_verified_runtime(mut self) -> Result<Self, RewardError> {
        let evidence = self.execute_runtime_preflight()?;
        self.runtime_preflight_evidence = Some(evidence);
        Ok(self)
    }

    /// Selected verifier isolation tier.
    #[must_use]
    pub const fn verifier_isolation_tier(&self) -> VerifierIsolationTier {
        self.sandbox.tier()
    }

    /// Return and revalidate the launch-bound verifier preflight evidence.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] if the backend is unavailable or changed since its
    /// evidence was pinned.
    pub fn verifier_isolation_evidence(&self) -> Result<VerifierIsolationEvidence, RewardError> {
        self.current_isolation_evidence()
    }

    /// Execute Ferrl's fixed wrong-output probe through the production image and
    /// return the protected controls observed before its candidate entry.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] if staging, runtime hardening, protected evidence, or
    /// cleanup fails. Call this before publishing the immutable launch.
    pub fn runtime_preflight_evidence(
        &self,
    ) -> Result<TrimulRuntimePreflightEvidence, RewardError> {
        if let Some(evidence) = &self.runtime_preflight_evidence {
            return Ok(evidence.clone());
        }
        self.execute_runtime_preflight()
    }

    fn execute_runtime_preflight(&self) -> Result<TrimulRuntimePreflightEvidence, RewardError> {
        let evidenced = self.verify_submission_with_evidence(HARDENING_PREFLIGHT_SUBMISSION)?;
        if evidenced.verification.correct {
            return Err(RewardError::msg(
                "TriMul hardening preflight unexpectedly passed correctness",
            ));
        }
        Ok(TrimulRuntimePreflightEvidence {
            contract_version: 1,
            isolation_tier: evidenced.isolation.tier,
            isolation_evidence_sha256: evidenced.isolation_evidence_sha256,
            probe_submission_sha256: sha256_hex(HARDENING_PREFLIGHT_SUBMISSION.as_bytes()),
            runtime_hardening: evidenced.runtime_hardening,
            runtime_hardening_evidence_sha256: evidenced.runtime_hardening_evidence_sha256,
        })
    }

    /// Set the secret case-generation seed (`POPCORN_SEED`).
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

    /// Set the verifier sandbox process cap (`ulimit -u`).
    ///
    /// This is a per-UID limit, not a per-container count. Keep it comfortably above
    /// the expected ambient task count for the training allocation while preserving a
    /// finite fork-bomb guard.
    #[must_use]
    pub fn with_verifier_max_procs(mut self, max_procs: u64) -> Self {
        self.verifier_max_procs = max_procs.max(1);
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
    /// service latency; otherwise the same-metric ratio over the baseline (or an
    /// inverse-latency proxy when no baseline is set).
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
        self.validate_case_sets()?;
        self.run_eval(submission)
    }

    /// Verify an extracted submission and retain the exact isolation/hardening evidence.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] if execution fails or the protected result omits or
    /// contradicts the launch-pinned evidence.
    pub fn verify_submission_with_evidence(
        &self,
        submission: &str,
    ) -> Result<EvidencedTrimulVerification, RewardError> {
        self.validate_case_sets()?;
        let eval = self.run_eval_detailed(submission)?;
        eval.evidenced_verification()
    }

    /// Verify Ferrl's bundled reference submission through the exact same protected path.
    ///
    /// Artifact audit uses this entry point so reference and candidate executions retain
    /// identical raw evidence and can be paired without operator-supplied timing values.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] if the verifier case set, assets, sandbox execution, or
    /// protected evidence is invalid.
    pub fn verify_reference_with_evidence(
        &self,
    ) -> Result<EvidencedTrimulVerification, RewardError> {
        self.verify_submission_with_evidence(REFERENCE_SUBMISSION)
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
            max_procs: Some(self.verifier_max_procs),
            ..ResourceLimits::default()
        }
    }

    /// Build the [`RunSpec`] for a candidate whose scratch dir is `scratch`: the eval
    /// image with the GPU exposed, the captured eval bundle bound read-only, scratch
    /// cache/output storage read-write, the network denied, and only the env needed
    /// by the eval. Per-candidate sealed binds are added immediately before launch.
    ///
    /// This remains private so descriptor-backed paths cannot outlive the
    /// [`TrimulVerifierAssets`] owner held by this reward.
    fn build_run_spec(&self, scratch: &Path) -> RunSpec {
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
            (
                "FERRL_VERIFIER_ISOLATION_TIER".into(),
                self.verifier_isolation_tier().as_str().to_string(),
            ),
            (
                "FERRL_TIMING_METRIC".into(),
                timing_metric_for_tier(self.verifier_isolation_tier()).to_string(),
            ),
            (
                "FERRL_GRADE_SOCKET".into(),
                "/work/.ferrl-grade-v1.sock".into(),
            ),
        ];
        if let Some(devices) = devices {
            env.push(("CUDA_VISIBLE_DEVICES".into(), devices.to_string()));
        }

        let image = self.verifier_assets.image_for_sandbox();
        let mut binds = self.verifier_assets.eval_binds();
        binds.push(Bind::rw(scratch, "/work").with_total_limit(self.scratch_max_bytes));
        RunSpec::new(
            image,
            vec![
                "python".into(),
                "-I".into(),
                FERRL_EVAL_DRIVER_PATH.into(),
                TEST_SPEC_PATH.into(),
                BENCH_SPEC_PATH.into(),
            ],
        )
        .with_gpu(true)
        .with_binds(binds)
        .with_workdir("/work")
        .with_env(env)
        .with_limits(self.limits())
        .with_protected_output(ProtectedOutput::new(
            scratch.join(".ferrl-grade-v1.sock"),
            "/work/.ferrl-grade-v1.sock",
        ))
    }

    /// Stage `submission`, run the eval in the sandbox, and score the result files.
    ///
    /// # Errors
    ///
    /// Returns [`RewardError`] only if the eval could not be *carried out* (scratch
    /// I/O or the sandbox failing to launch) — a crashing or wrong candidate is a
    /// `0.0` reward, not an error.
    fn run_eval(&self, submission: &str) -> Result<TrimulVerification, RewardError> {
        Ok(self.run_eval_detailed(submission)?.verification)
    }

    fn run_eval_detailed(&self, submission: &str) -> Result<TrimulEval, RewardError> {
        self.validate_case_sets()?;
        self.verify_verifier_assets()?;
        let scratch = self.make_scratch()?;
        let result = self.eval_in(&scratch, submission);
        // Best-effort cleanup; the scratch is node-local and disposable.
        let _ = std::fs::remove_dir_all(&scratch);
        self.verify_verifier_assets()?;
        result
    }

    /// Measure the reference candidate's geometric-mean ferrl service latency (ns) on
    /// this node's GPU by running the bundled reference over the configured
    /// `benchmark_cases`. This is the value to pin as the same-metric baseline
    /// ([`with_baseline_ns`](Self::with_baseline_ns)) — a *guarded pin*: measure it once
    /// on the target GPU through the same backend, record it with that tier's
    /// [`timing_metric_for_tier`], and re-use it only with the exact same preflight
    /// evidence. It is not a GPUMODE CUDA-event kernel runtime.
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
        let isolation = self.current_isolation_evidence()?;
        std::fs::create_dir_all(scratch.join("cache")).map_err(RewardError::verifier)?;
        let test_spec = render_spec(&self.test_cases);
        let bench_spec = render_spec(&self.benchmark_cases);
        let invocation = SealedInvocationAssets::capture(submission, &test_spec, &bench_spec)
            .map_err(RewardError::verifier)?;
        invocation.verify().map_err(RewardError::verifier)?;
        let mut sealed_spec = spec.clone();
        sealed_spec.binds.extend(invocation.binds());

        // The machine grade arrives only through the verifier-owned Unix stream.
        // Ordinary stdout/stderr remain untrusted diagnostics. The sealed invocation
        // descriptors remain owned here until the sandbox returns.
        let outcome = self
            .sandbox
            .run(&sealed_spec)
            .map_err(RewardError::verifier)?;
        let isolation_after = self.current_isolation_evidence()?;
        if isolation_after != isolation {
            return Err(RewardError::msg(
                "verifier isolation evidence changed during candidate execution",
            ));
        }
        invocation.verify().map_err(RewardError::verifier)?;
        let outcome = require_trimul_verifier_entry_for_tier(outcome, isolation.tier)
            .map_err(RewardError::verifier)?;

        let grade = outcome.protected_output;
        let (runtime_hardening, runtime_hardening_raw) =
            protected_runtime_evidence(&grade, isolation.tier).map_err(|error| {
                RewardError::verifier(SandboxError::Infrastructure {
                    status: outcome.status,
                    stderr: format!("TriMul protected hardening evidence failed: {error}"),
                })
            })?;
        if let Some(preflight) = &self.runtime_preflight_evidence {
            let expected = preflight.runtime_hardening.first().ok_or_else(|| {
                RewardError::msg("launch runtime preflight contains no hardening record")
            })?;
            if preflight.isolation_tier != isolation.tier
                || preflight.isolation_evidence_sha256
                    != verifier_isolation_evidence_sha256(&isolation)
                || runtime_hardening.iter().any(|record| record != expected)
            {
                return Err(RewardError::msg(
                    "candidate runtime hardening evidence differs from the launch preflight",
                ));
            }
        }
        let has_benchmark_section = grade.contains(RESULT_SPLIT);
        let (test_log, bench_log) = split_result(&grade);
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
                stdout: grade,
                stderr: format!(
                    "sandbox stdout:\n{}\nsandbox stderr:\n{}",
                    outcome.stdout, outcome.stderr
                ),
                isolation: Some(isolation),
                runtime_hardening,
                runtime_hardening_raw,
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
        self.validate_case_sets()?;
        let Some(code) = self.extract_submission(completion) else {
            let isolation_evidence_sha256 = self
                .verifier_isolation_evidence
                .as_ref()
                .map(verifier_isolation_evidence_sha256);
            let runtime_preflight_evidence_sha256 = self
                .runtime_preflight_evidence
                .as_ref()
                .map(runtime_preflight_evidence_sha256);
            return Ok(RewardOutcome {
                reward: 0.0,
                diagnostic: Some("trimul:no_submission".to_string()),
                metadata: Some(serde_json::json!({
                    "task": "trimul",
                    "reward_scheme": self.reward_profile.scheme.as_str(),
                    "reward_profile": self.reward_profile.metadata(),
                    "submission_extracted": false,
                    "verification_executed": false,
                    "verifier_isolation_tier": self.verifier_isolation_tier().as_str(),
                    "verifier_isolation_evidence_sha256": isolation_evidence_sha256,
                    "runtime_preflight_evidence_sha256": runtime_preflight_evidence_sha256,
                    "timing_metric": timing_metric_for_tier(self.verifier_isolation_tier()),
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
        let metadata = Some(self.reward_metadata(&code, &eval, reward)?);
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
    ) -> Result<serde_json::Value, RewardError> {
        let test_progress = eval.test_progress();
        let evidenced = eval.evidenced_verification()?;
        let runtime_preflight_evidence_sha256 = self
            .runtime_preflight_evidence
            .as_ref()
            .map(runtime_preflight_evidence_sha256);
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
            "verification_executed": true,
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
            "verifier_isolation_tier": self.verifier_isolation_tier().as_str(),
            "verifier_isolation_evidence": &evidenced.isolation,
            "verifier_isolation_evidence_sha256": &evidenced.isolation_evidence_sha256,
            "runtime_hardening": &evidenced.runtime_hardening,
            "runtime_hardening_evidence_sha256": &evidenced.runtime_hardening_evidence_sha256,
            "runtime_preflight_evidence_sha256": runtime_preflight_evidence_sha256,
            "verification_evidence": &evidenced.verification,
            "timing_metric": timing_metric_for_tier(self.verifier_isolation_tier()),
            "candidate_attempt_sentinels": candidate_attempt_sentinels(&eval.output.stdout),
            "candidate_rejection_reason": log_value(
                &eval.output.stdout,
                "ferrl-candidate-rejection-reason",
            ),
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

        Ok(metadata)
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

    fn verify_verifier_assets(&self) -> Result<(), RewardError> {
        self.verifier_assets
            .verify_current()
            .map_err(RewardError::verifier)
    }

    fn current_isolation_evidence(&self) -> Result<VerifierIsolationEvidence, RewardError> {
        let current = self.sandbox.preflight().map_err(RewardError::verifier)?;
        if current.tier != self.sandbox.tier() {
            return Err(RewardError::msg(
                "verifier backend returned evidence for a different isolation tier",
            ));
        }
        if self
            .verifier_isolation_evidence
            .as_ref()
            .is_some_and(|expected| expected != &current)
        {
            return Err(RewardError::msg(
                "verifier isolation evidence changed after launch preflight",
            ));
        }
        Ok(current)
    }

    fn validate_case_sets(&self) -> Result<(), RewardError> {
        if self.test_cases.is_empty() {
            return Err(RewardError::msg(
                "trimul verifier requires at least one correctness case",
            ));
        }
        if self.benchmark_cases.is_empty() {
            return Err(RewardError::msg(
                "trimul verifier requires at least one benchmark case",
            ));
        }
        Ok(())
    }
}

const TEST_VERIFIER_ENTRY: &str = "test-v4";
const BENCHMARK_VERIFIER_ENTRY: &str = "benchmark-v4";
/// Versioned protected end-to-end candidate latency for the no-admin tier.
pub const SAME_UID_TRIMUL_TIMING_METRIC: &str = "same-uid-apptainer-latency-v1";
/// Versioned protected end-to-end candidate service latency for the dedicated tier.
pub const DEDICATED_TRIMUL_TIMING_METRIC: &str = "isolated-service-latency-v1";
/// Default public timing metric, matching [`VerifierIsolationTier::default`].
pub const TRIMUL_TIMING_METRIC: &str = SAME_UID_TRIMUL_TIMING_METRIC;
/// Protected candidate hardening evidence contract emitted before candidate entry.
pub const TRIMUL_RUNTIME_HARDENING_CONTRACT: &str = "ferrl.candidate-hardening.v1";

/// Exact timing metric for one verifier isolation tier.
#[must_use]
pub const fn timing_metric_for_tier(tier: VerifierIsolationTier) -> &'static str {
    match tier {
        VerifierIsolationTier::SameUidApptainerV1 => SAME_UID_TRIMUL_TIMING_METRIC,
        VerifierIsolationTier::DedicatedUidServiceV1 => DEDICATED_TRIMUL_TIMING_METRIC,
    }
}

/// Domain-separated digest of canonical backend preflight evidence.
///
/// # Panics
///
/// Panics only if `serde_json` cannot serialize the fixed evidence schema.
#[must_use]
pub fn verifier_isolation_evidence_sha256(evidence: &VerifierIsolationEvidence) -> String {
    let bytes = serde_json::to_vec(evidence)
        .expect("VerifierIsolationEvidence contains only infallible JSON values");
    domain_sha256("ferrl.verifier-isolation-evidence.v1", &[&bytes])
}

/// Domain-separated digest of canonical launch-time runtime-control evidence.
///
/// # Panics
///
/// Panics only if `serde_json` cannot serialize the fixed evidence schema.
#[must_use]
pub fn runtime_preflight_evidence_sha256(evidence: &TrimulRuntimePreflightEvidence) -> String {
    let bytes = serde_json::to_vec(evidence)
        .expect("TrimulRuntimePreflightEvidence contains only infallible JSON values");
    domain_sha256("ferrl.trimul-runtime-preflight-evidence.v1", &[&bytes])
}
const TRIMUL_INFRASTRUCTURE_MARKER: &str = "ferrl-infrastructure: v1";
const TRIMUL_INFRASTRUCTURE_EXIT: i32 = 114;
const HARDENING_LOG_KEY: &str = "ferrl-candidate-hardening";
const ISOLATION_LOG_KEY: &str = "ferrl-verifier-isolation-tier";
const DEVICE_IDENTITY_LOG_KEY: &str = "ferrl-executing-device";
const DEVICE_IDENTITY_CONTRACT: &str = "ferrl.executing-device.v1";

fn validate_runtime_hardening_record(raw: &str) -> Result<serde_json::Value, String> {
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|error| format!("malformed candidate hardening JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "candidate hardening evidence is not an object".to_string())?;
    const KEYS: [&str; 19] = [
        "arch",
        "cap_amb",
        "cap_bnd",
        "cap_eff",
        "cap_inh",
        "cap_prm",
        "cgroup",
        "contract",
        "denial_probes",
        "dumpable",
        "landlock",
        "network_socket_policy",
        "no_new_privs",
        "physical_gpu_isolation",
        "seccomp_filters",
        "seccomp_mode",
        "seccomp_policy",
        "seccomp_tsync",
        "unix_socket_probe",
    ];
    if object.len() != KEYS.len() || KEYS.iter().any(|key| !object.contains_key(*key)) {
        return Err("candidate hardening evidence has an unsupported schema".to_string());
    }
    let exact_string = |key: &str, expected: &str| {
        (object.get(key).and_then(serde_json::Value::as_str) == Some(expected))
            .then_some(())
            .ok_or_else(|| format!("candidate hardening field {key:?} is not {expected:?}"))
    };
    let exact_i64 = |key: &str, expected: i64| {
        (object.get(key).and_then(serde_json::Value::as_i64) == Some(expected))
            .then_some(())
            .ok_or_else(|| format!("candidate hardening field {key:?} is not {expected}"))
    };
    let exact_bool = |key: &str, expected: bool| {
        (object.get(key).and_then(serde_json::Value::as_bool) == Some(expected))
            .then_some(())
            .ok_or_else(|| format!("candidate hardening field {key:?} is not {expected}"))
    };
    exact_string("contract", TRIMUL_RUNTIME_HARDENING_CONTRACT)?;
    exact_string("arch", "x86_64")?;
    exact_string("seccomp_policy", "x86_64-tsync-af-unix-v1")?;
    exact_string("network_socket_policy", "af_unix_only")?;
    exact_i64("dumpable", 0)?;
    exact_i64("no_new_privs", 1)?;
    exact_i64("seccomp_mode", 2)?;
    exact_bool("seccomp_tsync", true)?;
    exact_bool("unix_socket_probe", true)?;
    exact_bool("landlock", false)?;
    exact_bool("cgroup", false)?;
    exact_bool("physical_gpu_isolation", false)?;
    let denial_probes = object
        .get("denial_probes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "candidate hardening denial probes are not an array".to_string())?;
    let expected_probes = [
        "bpf",
        "io_uring",
        "namespace",
        "network",
        "parent_proc",
        "pidfd_getfd",
        "process_vm",
        "ptrace",
    ];
    if denial_probes.len() != expected_probes.len()
        || denial_probes
            .iter()
            .zip(expected_probes)
            .any(|(actual, expected)| actual.as_str() != Some(expected))
    {
        return Err("candidate hardening denial probes are incomplete".to_string());
    }
    for key in ["cap_amb", "cap_bnd", "cap_eff", "cap_inh", "cap_prm"] {
        let capability = object
            .get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("candidate hardening field {key:?} is not a string"))?;
        if capability.len() != 16
            || !capability
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!(
                "candidate hardening field {key:?} is not canonical capability hex"
            ));
        }
        if key != "cap_bnd" && capability != "0000000000000000" {
            return Err(format!(
                "candidate hardening field {key:?} retains capabilities"
            ));
        }
    }
    let bounding = object["cap_bnd"]
        .as_str()
        .ok_or_else(|| "candidate hardening bounding set is not a string".to_string())
        .and_then(|value| {
            u64::from_str_radix(value, 16)
                .map_err(|error| format!("candidate hardening bounding set is invalid: {error}"))
        })?;
    const CAP_SYS_PTRACE: u64 = 1 << 19;
    const CAP_SYS_ADMIN: u64 = 1 << 21;
    if bounding & (CAP_SYS_PTRACE | CAP_SYS_ADMIN) != 0 {
        return Err(
            "candidate hardening bounding set retains ptrace or admin capability".to_string(),
        );
    }
    if !object["seccomp_filters"].is_null()
        && object["seccomp_filters"]
            .as_u64()
            .is_none_or(|count| count == 0)
    {
        return Err("candidate hardening seccomp filter count is invalid".to_string());
    }
    if serde_json::to_string(&value).map_err(|error| error.to_string())? != raw {
        return Err("candidate hardening evidence is not canonical JSON".to_string());
    }
    Ok(value)
}

fn protected_runtime_evidence(
    grade: &str,
    tier: VerifierIsolationTier,
) -> Result<(Vec<serde_json::Value>, Vec<String>), String> {
    let expected_metric = timing_metric_for_tier(tier);
    let (test_log, benchmark_log) = split_result(grade);
    let mut phases = vec![test_log];
    if grade.contains(RESULT_SPLIT) {
        phases.push(benchmark_log);
    }
    let mut values = Vec::with_capacity(phases.len());
    let mut raw_values = Vec::with_capacity(phases.len());
    for phase in phases {
        if log_value(phase, ISOLATION_LOG_KEY) != Some(tier.as_str()) {
            return Err("protected verifier did not authenticate its isolation tier".to_string());
        }
        if log_value(phase, "ferrl-timing-metric") != Some(expected_metric) {
            return Err("protected verifier did not authenticate its timing metric".to_string());
        }
        let raw = log_value(phase, HARDENING_LOG_KEY)
            .ok_or_else(|| "protected verifier omitted candidate hardening evidence".to_string())?;
        values.push(validate_runtime_hardening_record(raw)?);
        raw_values.push(raw.to_string());
    }
    Ok((values, raw_values))
}

fn exact_log_value<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    let mut values = text.lines().filter_map(|line| {
        let (actual, value) = line.split_once(": ")?;
        (actual.trim() == key).then_some(value.trim())
    });
    let value = values
        .next()
        .ok_or_else(|| format!("protected grade omitted {key:?}"))?;
    if values.next().is_some() {
        return Err(format!("protected grade repeated {key:?}"));
    }
    Ok(value)
}

fn exact_parsed_value<T>(text: &str, key: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    exact_log_value(text, key)?
        .parse()
        .map_err(|error| format!("protected grade field {key:?} is invalid: {error}"))
}

fn exact_index_set(
    text: &str,
    prefix: &str,
    suffix: &str,
    expected_count: usize,
) -> Result<(), String> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for line in text.lines() {
        let Some((key, _)) = line.split_once(": ") else {
            continue;
        };
        let key = key.trim();
        let Some(index_text) = key
            .strip_prefix(prefix)
            .and_then(|value| value.strip_suffix(suffix))
        else {
            continue;
        };
        let index = index_text.parse::<usize>().map_err(|error| {
            format!("protected grade has malformed indexed field {key:?}: {error}")
        })?;
        *counts.entry(index).or_default() += 1;
    }
    let expected = (0..expected_count)
        .map(|index| (index, 1usize))
        .collect::<BTreeMap<_, _>>();
    if counts != expected {
        return Err(format!(
            "protected grade {prefix}<index>{suffix} coverage is not exactly 0..{expected_count}"
        ));
    }
    Ok(())
}

fn parse_executing_device(raw: &str) -> Result<TrimulExecutingDevice, String> {
    let device: TrimulExecutingDevice = serde_json::from_str(raw)
        .map_err(|error| format!("malformed executing-device JSON: {error}"))?;
    if device.contract != DEVICE_IDENTITY_CONTRACT {
        return Err("executing-device evidence has an unsupported contract".to_string());
    }
    if device.name.trim().is_empty()
        || device
            .name
            .bytes()
            .any(|byte| byte == 0 || byte == b'\n' || byte == b'\r')
    {
        return Err("executing-device name is empty or non-canonical".to_string());
    }
    if device.uuid.len() != 32
        || !device
            .uuid
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("executing-device UUID is not 16-byte lowercase hexadecimal".to_string());
    }
    if device.pci_bus_id.is_empty()
        || !device.pci_bus_id.is_ascii()
        || !device.pci_bus_id.contains(':')
        || !device.pci_bus_id.contains('.')
        || !device.pci_bus_id.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) || byte == b':' || byte == b'.'
        })
    {
        return Err("executing-device PCI bus id is not canonical".to_string());
    }
    if serde_json::to_string(&device).map_err(|error| error.to_string())? != raw {
        return Err("executing-device evidence is not canonical JSON".to_string());
    }
    Ok(device)
}

/// Validate one protected verification result for publication-grade artifact use.
///
/// The grade must contain one successful sandbox/test/benchmark execution, complete
/// exactly indexed case evidence, finite benchmark statistics, and the same canonical
/// physical-device identity in both phases.
///
/// # Errors
///
/// Returns a descriptive error when raw evidence, hashes, isolation, runtime controls,
/// exits, case coverage, benchmark statistics, or device identity are incomplete or
/// inconsistent.
pub fn validate_artifact_verification_evidence(
    evidenced: &EvidencedTrimulVerification,
    expected_test_cases: usize,
    expected_benchmark_cases: usize,
) -> Result<TrimulArtifactVerificationEvidence, String> {
    if expected_test_cases == 0 || expected_benchmark_cases == 0 {
        return Err("artifact verification requires non-empty test and benchmark sets".to_string());
    }
    if !evidenced.sandbox_status.is_success() {
        let infrastructure = evidenced
            .protected_output
            .lines()
            .find(|line| line.starts_with("ferrl-infrastructure: "))
            .unwrap_or("ferrl-infrastructure: not reported")
            .chars()
            .take(512)
            .collect::<String>();
        return Err(format!(
            "artifact verifier sandbox did not exit successfully: {:?}; {infrastructure}",
            evidenced.sandbox_status
        ));
    }
    if sha256_hex(evidenced.protected_output.as_bytes()) != evidenced.protected_output_sha256
        || sha256_hex(evidenced.sandbox_diagnostics.as_bytes())
            != evidenced.sandbox_diagnostics_sha256
    {
        return Err("artifact verifier raw-evidence digest mismatch".to_string());
    }
    if verifier_isolation_evidence_sha256(&evidenced.isolation)
        != evidenced.isolation_evidence_sha256
    {
        return Err("artifact verifier isolation-evidence digest mismatch".to_string());
    }
    if evidenced.protected_output.matches(RESULT_SPLIT).count() != 1 {
        return Err(
            "artifact protected grade must contain exactly one phase separator".to_string(),
        );
    }
    let (test_log, benchmark_log) = split_result(&evidenced.protected_output);
    let mut hardening_values = Vec::with_capacity(2);
    let mut hardening_raw = Vec::with_capacity(2);
    for (phase, expected_entry) in [
        (test_log, TEST_VERIFIER_ENTRY),
        (benchmark_log, BENCHMARK_VERIFIER_ENTRY),
    ] {
        if exact_log_value(phase, "ferrl-entry")? != expected_entry {
            return Err(
                "artifact protected grade did not reach the trusted verifier entry".to_string(),
            );
        }
        if exact_log_value(phase, ISOLATION_LOG_KEY)? != evidenced.isolation.tier.as_str()
            || exact_log_value(phase, "ferrl-timing-metric")?
                != timing_metric_for_tier(evidenced.isolation.tier)
        {
            return Err(
                "artifact protected grade changed verifier tier or timing metric".to_string(),
            );
        }
        let raw = exact_log_value(phase, HARDENING_LOG_KEY)?;
        hardening_values.push(validate_runtime_hardening_record(raw)?);
        hardening_raw.push(raw);
    }
    let hardening_fields = hardening_raw
        .iter()
        .map(|value| value.as_bytes())
        .collect::<Vec<_>>();
    if hardening_values != evidenced.runtime_hardening
        || domain_sha256(
            "ferrl.trimul-runtime-hardening-evidence.v1",
            &hardening_fields,
        ) != evidenced.runtime_hardening_evidence_sha256
    {
        return Err("artifact verifier runtime-hardening evidence mismatch".to_string());
    }

    let test_count: usize = exact_parsed_value(test_log, "test-count")?;
    let benchmark_count: usize = exact_parsed_value(benchmark_log, "benchmark-count")?;
    if test_count != expected_test_cases || benchmark_count != expected_benchmark_cases {
        return Err("artifact protected grade case counts do not match task.yml".to_string());
    }
    if exact_log_value(test_log, "check")? != "pass"
        || exact_log_value(benchmark_log, "check")? != "pass"
        || !evidenced.verification.correct
    {
        return Err(
            "artifact candidate did not pass every protected correctness check".to_string(),
        );
    }
    let test_exit: i32 = exact_parsed_value(test_log, "test-exit")?;
    let benchmark_exit: i32 = exact_parsed_value(benchmark_log, "benchmark-exit")?;
    if test_exit != 0 || benchmark_exit != 0 {
        return Err("artifact test or benchmark exit marker is not zero".to_string());
    }

    exact_index_set(test_log, "test.", ".spec", test_count)?;
    exact_index_set(test_log, "test.", ".status", test_count)?;
    let mut test_cases = Vec::with_capacity(test_count);
    for index in 0..test_count {
        let spec = exact_log_value(test_log, &format!("test.{index}.spec"))?.to_string();
        if spec.is_empty() || exact_log_value(test_log, &format!("test.{index}.status"))? != "pass"
        {
            return Err(format!(
                "artifact test case {index} is missing or did not pass"
            ));
        }
        test_cases.push(TrimulTestCaseEvidence {
            index,
            spec,
            passed: true,
        });
    }

    for suffix in [".spec", ".runs", ".mean", ".std", ".err", ".best", ".worst"] {
        exact_index_set(benchmark_log, "benchmark.", suffix, benchmark_count)?;
    }
    let mut benchmark_cases = Vec::with_capacity(benchmark_count);
    for index in 0..benchmark_count {
        let spec = exact_log_value(benchmark_log, &format!("benchmark.{index}.spec"))?.to_string();
        let runs: u64 = exact_parsed_value(benchmark_log, &format!("benchmark.{index}.runs"))?;
        let mean_ns: f64 = exact_parsed_value(benchmark_log, &format!("benchmark.{index}.mean"))?;
        let std_ns: f64 = exact_parsed_value(benchmark_log, &format!("benchmark.{index}.std"))?;
        let err_ns: f64 = exact_parsed_value(benchmark_log, &format!("benchmark.{index}.err"))?;
        let best_ns: f64 = exact_parsed_value(benchmark_log, &format!("benchmark.{index}.best"))?;
        let worst_ns: f64 = exact_parsed_value(benchmark_log, &format!("benchmark.{index}.worst"))?;
        if spec.is_empty()
            || runs < 3
            || ![mean_ns, std_ns, err_ns, best_ns, worst_ns]
                .iter()
                .all(|value| value.is_finite())
            || mean_ns <= 0.0
            || best_ns <= 0.0
            || worst_ns <= 0.0
            || std_ns < 0.0
            || err_ns < 0.0
            || best_ns > mean_ns
            || mean_ns > worst_ns
        {
            return Err(format!(
                "artifact benchmark case {index} has invalid statistics"
            ));
        }
        benchmark_cases.push(TrimulBenchmarkCaseEvidence {
            index,
            spec,
            runs,
            mean_ns,
            std_ns,
            err_ns,
            best_ns,
            worst_ns,
        });
    }
    let parsed_means = benchmark_cases
        .iter()
        .map(|case| case.mean_ns)
        .collect::<Vec<_>>();
    if parsed_means.len() != evidenced.verification.benchmark_means_ns.len()
        || parsed_means
            .iter()
            .zip(&evidenced.verification.benchmark_means_ns)
            .any(|(parsed, summarized)| parsed.to_bits() != summarized.to_bits())
    {
        return Err("artifact benchmark means disagree with the protected summary".to_string());
    }
    let parsed_geomean = geomean(&parsed_means)
        .filter(|value| value.is_finite())
        .ok_or_else(|| "artifact benchmark geomean is invalid".to_string())?;
    if evidenced
        .verification
        .geomean_ns
        .is_none_or(|summarized| summarized.to_bits() != parsed_geomean.to_bits())
    {
        return Err("artifact benchmark geomean disagrees with the protected means".to_string());
    }

    let test_device = parse_executing_device(exact_log_value(test_log, DEVICE_IDENTITY_LOG_KEY)?)?;
    let benchmark_device =
        parse_executing_device(exact_log_value(benchmark_log, DEVICE_IDENTITY_LOG_KEY)?)?;
    if test_device != benchmark_device {
        return Err("test and benchmark phases used different physical CUDA devices".to_string());
    }
    if test_device.cuda_logical_ordinal != 0 {
        return Err(
            "artifact verifier did not use the sole visible logical CUDA device".to_string(),
        );
    }

    Ok(TrimulArtifactVerificationEvidence {
        sandbox_status: evidenced.sandbox_status,
        test_exit,
        benchmark_exit,
        executing_device: test_device,
        test_cases,
        benchmark_cases,
        protected_output_sha256: evidenced.protected_output_sha256.clone(),
        sandbox_diagnostics_sha256: evidenced.sandbox_diagnostics_sha256.clone(),
    })
}

/// Require records written by the sealed parent on the verifier-only socket after
/// exact trusted initialization and either actual candidate-frame entry or an
/// authenticated candidate-source rejection. Benchmark proof is mandatory whenever
/// the protected grade reached that phase, so a later platform failure cannot retain
/// correctness credit.
fn require_trimul_verifier_entry_for_tier(
    outcome: RunOutcome,
    tier: VerifierIsolationTier,
) -> Result<RunOutcome, SandboxError> {
    let infrastructure_record = outcome.protected_output.lines().find_map(|line| {
        let line = line.trim();
        (line == TRIMUL_INFRASTRUCTURE_MARKER || line.starts_with("ferrl-infrastructure: v1 "))
            .then_some(line)
    });
    if outcome.status == RunStatus::Exited(TRIMUL_INFRASTRUCTURE_EXIT)
        || infrastructure_record.is_some()
    {
        let record = infrastructure_record.unwrap_or("reserved exit 114");
        return Err(SandboxError::Infrastructure {
            status: outcome.status,
            stderr: format!(
                "TriMul protected parent reported authenticated infrastructure failure ({record}); sandbox stderr: {}",
                outcome.stderr
            ),
        });
    }
    let (test_log, benchmark_log) = split_result(&outcome.protected_output);
    if !phase_reached(test_log, TEST_VERIFIER_ENTRY, "test-import-v1") {
        return Err(missing_verifier_entry(&outcome, "test"));
    }
    if outcome.protected_output.contains(RESULT_SPLIT)
        && !phase_reached(
            benchmark_log,
            BENCHMARK_VERIFIER_ENTRY,
            "benchmark-import-v1",
        )
    {
        return Err(missing_verifier_entry(&outcome, "benchmark"));
    }
    if let Err(error) = protected_runtime_evidence(&outcome.protected_output, tier) {
        return Err(SandboxError::Infrastructure {
            status: outcome.status,
            stderr: format!("TriMul verifier hardening evidence failed: {error}"),
        });
    }
    Ok(outcome)
}

#[cfg(test)]
fn require_trimul_verifier_entry(outcome: RunOutcome) -> Result<RunOutcome, SandboxError> {
    require_trimul_verifier_entry_for_tier(outcome, VerifierIsolationTier::SameUidApptainerV1)
}

fn phase_reached(log: &str, entry: &str, rejected: &str) -> bool {
    log_value(log, "ferrl-entry") == Some(entry)
        || log_value(log, "ferrl-candidate-rejected") == Some(rejected)
}

fn missing_verifier_entry(outcome: &RunOutcome, phase: &str) -> SandboxError {
    let stderr = if outcome.stderr.is_empty() {
        format!("TriMul {phase} verifier did not reach trusted worker/GPU entry")
    } else {
        format!(
            "TriMul {phase} verifier did not reach trusted worker/GPU entry: {}",
            outcome.stderr
        )
    };
    SandboxError::Infrastructure {
        status: outcome.status,
        stderr,
    }
}

/// Process-wide counter for unique scratch-dir names.
static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

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

    fn evidenced_verification(&self) -> Result<EvidencedTrimulVerification, RewardError> {
        let isolation = self.output.isolation.clone().ok_or_else(|| {
            RewardError::msg("TriMul result is missing verifier isolation evidence")
        })?;
        if self.output.runtime_hardening.is_empty() {
            return Err(RewardError::msg(
                "TriMul result is missing protected runtime hardening evidence",
            ));
        }
        let runtime_bytes = self
            .output
            .runtime_hardening_raw
            .iter()
            .map(String::as_bytes)
            .collect::<Vec<_>>();
        Ok(EvidencedTrimulVerification {
            verification: self.verification.clone(),
            isolation_evidence_sha256: verifier_isolation_evidence_sha256(&isolation),
            isolation,
            runtime_hardening: self.output.runtime_hardening.clone(),
            runtime_hardening_evidence_sha256: domain_sha256(
                "ferrl.trimul-runtime-hardening-evidence.v1",
                &runtime_bytes,
            ),
            sandbox_status: self.status,
            protected_output: self.output.stdout.clone(),
            protected_output_sha256: sha256_hex(self.output.stdout.as_bytes()),
            sandbox_diagnostics: self.output.stderr.clone(),
            sandbox_diagnostics_sha256: sha256_hex(self.output.stderr.as_bytes()),
        })
    }
}

#[derive(Debug, Clone, Default)]
struct TrimulEvalOutput {
    stdout: String,
    stderr: String,
    isolation: Option<VerifierIsolationEvidence>,
    runtime_hardening: Vec<serde_json::Value>,
    runtime_hardening_raw: Vec<String>,
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

fn domain_sha256(domain: &str, fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain.as_bytes());
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    format!("{:x}", hasher.finalize())
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
        self.validate_case_sets()?;
        self.verify_verifier_assets()?;
        let outcome = self.reward_outcome(completion);
        self.verify_verifier_assets()?;
        Ok(outcome?.reward)
    }

    fn reward_group_detailed(
        &self,
        _sample: &Sample<()>,
        completions: &[String],
    ) -> Result<Vec<RewardOutcome>, RewardError> {
        self.validate_case_sets()?;
        self.verify_verifier_assets()?;
        let outcomes = if self.verifier_parallelism <= 1 || completions.len() <= 1 {
            completions
                .iter()
                .enumerate()
                .map(|(index, completion)| self.reward_outcome_for_worker(completion, index))
                .collect()
        } else {
            map_bounded_reward_outcomes(
                completions,
                self.verifier_parallelism,
                |index, completion| self.reward_outcome_for_worker(completion, index),
            )
        };
        self.verify_verifier_assets()?;
        outcomes
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
        static FIXTURE: std::sync::OnceLock<(PathBuf, PathBuf, PathBuf, PathBuf)> =
            std::sync::OnceLock::new();
        let (_, image, eval, scratch) = FIXTURE.get_or_init(|| verifier_fixture("shared-reward"));
        let assets = TrimulVerifierAssets::capture(image, eval, scratch).unwrap();
        TrimulReward::new(assets, scratch).with_cases(
            vec![case(8, true, Distribution::Normal)],
            vec![case(16, false, Distribution::Cauchy)],
        )
    }

    fn test_isolation_evidence() -> VerifierIsolationEvidence {
        VerifierIsolationEvidence {
            contract_version: VERIFIER_ISOLATION_EVIDENCE_VERSION,
            tier: VerifierIsolationTier::SameUidApptainerV1,
            requester_uid: 1000,
            launcher_uid: 1000,
            uid_boundary: VerifierUidBoundary::SameHostUid,
            asset_transport: VerifierAssetTransport::InProcessSealedCopy,
            apptainer_path: PathBuf::from("/usr/bin/apptainer"),
            apptainer_sha256: "11".repeat(32),
            apptainer_len_bytes: 1,
            apptainer_version: "apptainer version 1.4.0".to_string(),
            work_root: PathBuf::from("/tmp/ferrl-verifier-test"),
            work_root_uid: 1000,
            work_root_device: 1,
            work_root_inode: 2,
            work_root_mode: 0o700,
        }
    }

    fn test_runtime_hardening() -> serde_json::Value {
        serde_json::json!({
            "arch": "x86_64",
            "cap_amb": "0000000000000000",
            "cap_bnd": "00000000a80425fb",
            "cap_eff": "0000000000000000",
            "cap_inh": "0000000000000000",
            "cap_prm": "0000000000000000",
            "cgroup": false,
            "contract": TRIMUL_RUNTIME_HARDENING_CONTRACT,
            "denial_probes": ["bpf", "io_uring", "namespace", "network", "parent_proc", "pidfd_getfd", "process_vm", "ptrace"],
            "dumpable": 0,
            "landlock": false,
            "network_socket_policy": "af_unix_only",
            "no_new_privs": 1,
            "physical_gpu_isolation": false,
            "seccomp_filters": 1,
            "seccomp_mode": 2,
            "seccomp_policy": "x86_64-tsync-af-unix-v1",
            "seccomp_tsync": true,
            "unix_socket_probe": true,
        })
    }

    fn test_runtime_hardening_raw() -> String {
        serde_json::to_string(&test_runtime_hardening()).unwrap()
    }

    fn evidenced_output(stdout: &str, stderr: &str) -> TrimulEvalOutput {
        TrimulEvalOutput {
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            isolation: Some(test_isolation_evidence()),
            runtime_hardening: vec![test_runtime_hardening()],
            runtime_hardening_raw: vec![test_runtime_hardening_raw()],
        }
    }

    fn protected_phase_prelude() -> String {
        format!(
            "ferrl-verifier-isolation-tier: {}\nferrl-timing-metric: {}\nferrl-candidate-hardening: {}\n",
            VerifierIsolationTier::SameUidApptainerV1.as_str(),
            TRIMUL_TIMING_METRIC,
            test_runtime_hardening_raw(),
        )
    }

    fn test_device_identity(uuid: &str) -> String {
        serde_json::to_string(&TrimulExecutingDevice {
            contract: DEVICE_IDENTITY_CONTRACT.to_owned(),
            cuda_logical_ordinal: 0,
            name: "NVIDIA H100 80GB HBM3".to_owned(),
            pci_bus_id: "0000:01:00.0".to_owned(),
            uuid: uuid.to_owned(),
        })
        .unwrap()
    }

    fn artifact_grade(test_uuid: &str, benchmark_uuid: &str) -> String {
        format!(
            "{}ferrl-entry: {TEST_VERIFIER_ENTRY}\nferrl-executing-device: {}\ntest-count: 1\ntest.0.spec: seqlen: 8; bs: 1\ntest.0.status: pass\ncheck: pass\ntest-exit: 0\n{RESULT_SPLIT}\n{}ferrl-entry: {BENCHMARK_VERIFIER_ENTRY}\nferrl-executing-device: {}\nbenchmark-count: 1\nbenchmark.0.spec: seqlen: 16; bs: 1\nbenchmark.0.runs: 100\nbenchmark.0.mean: 10\nbenchmark.0.std: 0.5\nbenchmark.0.err: 0.05\nbenchmark.0.best: 9\nbenchmark.0.worst: 11\ncheck: pass\nbenchmark-exit: 0\n",
            protected_phase_prelude(),
            test_device_identity(test_uuid),
            protected_phase_prelude(),
            test_device_identity(benchmark_uuid),
        )
    }

    fn artifact_evidenced_grade(grade: String) -> EvidencedTrimulVerification {
        let isolation = test_isolation_evidence();
        let hardening_raw = test_runtime_hardening_raw();
        let hardening_fields = [hardening_raw.as_bytes(), hardening_raw.as_bytes()];
        EvidencedTrimulVerification {
            verification: TrimulVerification {
                correct: true,
                benchmark_means_ns: vec![10.0],
                geomean_ns: geomean(&[10.0]),
                speedup: None,
            },
            isolation_evidence_sha256: verifier_isolation_evidence_sha256(&isolation),
            isolation,
            runtime_hardening: vec![test_runtime_hardening(), test_runtime_hardening()],
            runtime_hardening_evidence_sha256: domain_sha256(
                "ferrl.trimul-runtime-hardening-evidence.v1",
                &hardening_fields,
            ),
            sandbox_status: RunStatus::Exited(0),
            protected_output_sha256: sha256_hex(grade.as_bytes()),
            protected_output: grade,
            sandbox_diagnostics_sha256: sha256_hex(b""),
            sandbox_diagnostics: String::new(),
        }
    }

    #[test]
    fn artifact_evidence_requires_complete_exact_same_device_grade() {
        let uuid = "ab".repeat(16);
        let evidenced = artifact_evidenced_grade(artifact_grade(&uuid, &uuid));

        let exact = validate_artifact_verification_evidence(&evidenced, 1, 1).unwrap();

        assert_eq!(exact.sandbox_status, RunStatus::Exited(0));
        assert_eq!(exact.executing_device.uuid, uuid);
        assert_eq!(exact.test_cases.len(), 1);
        assert_eq!(exact.benchmark_cases.len(), 1);
        assert_eq!(exact.benchmark_cases[0].mean_ns, 10.0);
    }

    #[test]
    fn artifact_evidence_rejects_case_device_exit_and_digest_mutations() {
        let uuid = "ab".repeat(16);
        let other_uuid = "cd".repeat(16);
        let valid = artifact_grade(&uuid, &uuid);
        let mutations = [
            valid.replacen("test.0.status: pass\n", "", 1),
            valid.replacen(
                "test.0.status: pass\n",
                "test.0.status: pass\ntest.0.status: pass\n",
                1,
            ),
            valid.replacen("benchmark.0.worst: 11\n", "", 1),
            valid.replacen("test-exit: 0\n", "test-exit: 7\n", 1),
            artifact_grade(&uuid, &other_uuid),
        ];
        for mutation in mutations {
            let evidenced = artifact_evidenced_grade(mutation);
            assert!(
                validate_artifact_verification_evidence(&evidenced, 1, 1).is_err(),
                "mutated publication evidence unexpectedly passed"
            );
        }

        let mut digest_mutation = artifact_evidenced_grade(valid);
        digest_mutation.protected_output_sha256 = "00".repeat(32);
        assert!(validate_artifact_verification_evidence(&digest_mutation, 1, 1).is_err());

        let mut timeout = artifact_evidenced_grade(
            "ferrl-infrastructure: v1 phase=test reason=\"candidate timed out\"\n".to_owned(),
        );
        timeout.sandbox_status = RunStatus::TimedOut;
        assert_eq!(
            validate_artifact_verification_evidence(&timeout, 1, 1).unwrap_err(),
            concat!(
                "artifact verifier sandbox did not exit successfully: TimedOut; ",
                "ferrl-infrastructure: v1 phase=test reason=\"candidate timed out\""
            )
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn candidate_hardening_record_rejects_weakened_or_noncanonical_controls() {
        let valid = test_runtime_hardening_raw();
        assert!(validate_runtime_hardening_record(&valid).is_ok());

        let reject_field = |key: &str, replacement: serde_json::Value| {
            let mut record = test_runtime_hardening();
            record
                .as_object_mut()
                .unwrap()
                .insert(key.to_string(), replacement);
            let raw = serde_json::to_string(&record).unwrap();
            assert!(
                validate_runtime_hardening_record(&raw).is_err(),
                "weakened {key} unexpectedly passed: {raw}"
            );
        };
        reject_field("cap_eff", serde_json::json!("0000000000000001"));
        reject_field("cap_prm", serde_json::json!("0000000000000001"));
        reject_field("cap_inh", serde_json::json!("0000000000000001"));
        reject_field("cap_amb", serde_json::json!("0000000000000001"));
        reject_field("cap_bnd", serde_json::json!("0000000000080000"));
        reject_field("cap_bnd", serde_json::json!("0000000000200000"));
        reject_field("no_new_privs", serde_json::json!(0));
        reject_field("dumpable", serde_json::json!(1));
        reject_field("seccomp_mode", serde_json::json!(1));
        reject_field("seccomp_filters", serde_json::json!(0));
        reject_field("seccomp_tsync", serde_json::json!(false));
        reject_field("unix_socket_probe", serde_json::json!(false));
        reject_field("network_socket_policy", serde_json::json!("unrestricted"));
        reject_field("contract", serde_json::json!("operator-claimed-v1"));
        reject_field(
            "denial_probes",
            serde_json::json!(["bpf", "io_uring", "namespace", "network", "ptrace"]),
        );
        let mut missing = test_runtime_hardening();
        missing.as_object_mut().unwrap().remove("no_new_privs");
        assert!(
            validate_runtime_hardening_record(&serde_json::to_string(&missing).unwrap()).is_err()
        );
        assert!(validate_runtime_hardening_record(&format!(" {valid}")).is_err());

        let protected = format!(
            "{}ferrl-entry: {TEST_VERIFIER_ENTRY}\ntest-exit: 7\n",
            protected_phase_prelude(),
        );
        assert!(protected_runtime_evidence(
            &protected,
            VerifierIsolationTier::DedicatedUidServiceV1,
        )
        .is_err());
    }

    fn verifier_fixture(tag: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ferrl-trimul-assets-{tag}-{}-{nonce}",
            std::process::id()
        ));
        let image = root.join("image.sif");
        let eval = root.join("eval");
        let scratch = root.join("scratch");
        std::fs::create_dir_all(&eval).unwrap();
        std::fs::write(&image, b"exact image").unwrap();
        std::fs::write(eval.join("eval.py"), b"# exact eval\n").unwrap();
        std::fs::write(eval.join("reference.py"), b"# exact reference\n").unwrap();
        std::fs::write(eval.join("task.py"), b"# exact task\n").unwrap();
        std::fs::write(eval.join("utils.py"), b"# exact utils\n").unwrap();
        std::fs::write(
            eval.join("task.yml"),
            b"tests:\n  - {\"seqlen\": 8, \"bs\": 1, \"dim\": 4, \"hiddendim\": 4, \"seed\": 1, \"nomask\": true, \"distribution\": \"normal\"}\nbenchmarks:\n  - {\"seqlen\": 16, \"bs\": 1, \"dim\": 4, \"hiddendim\": 4, \"seed\": 2, \"nomask\": false, \"distribution\": \"cauchy\"}\n",
        )
        .unwrap();
        (root, image, eval, scratch)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn verifier_assets_consume_stable_image_and_sealed_eval_descriptors() {
        let (root, image, eval, scratch) = verifier_fixture("stable-consumption");
        let assets = TrimulVerifierAssets::capture(&image, &eval, &scratch).unwrap();
        let reward = TrimulReward::new(assets.clone(), &scratch);
        let run_scratch = scratch.join("candidate");
        let spec = reward.build_run_spec(&run_scratch);

        #[cfg(target_os = "linux")]
        assert!(spec
            .image
            .starts_with(format!("/proc/{}/fd", std::process::id())));
        let eval_bind = spec
            .binds
            .iter()
            .find(|bind| bind.dst == Path::new("/opt/ferrl-verifier/eval.py"))
            .unwrap();
        assert_ne!(eval_bind.src, eval);
        assert!(eval_bind
            .src
            .starts_with(format!("/proc/{}/fd", std::process::id())));
        assert_eq!(std::fs::read(&eval_bind.src).unwrap(), b"# exact eval\n");
        assert_eq!(
            spec.binds
                .iter()
                .filter(|bind| {
                    SANDBOX_EVAL_FILES
                        .iter()
                        .any(|(_, destination)| bind.dst == Path::new(destination))
                })
                .count(),
            SANDBOX_EVAL_FILES.len()
        );
        assets.verify_current().unwrap();
        drop(reward);
        drop(assets);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn verifier_assets_are_never_inherited_by_launcher_or_candidate_processes() {
        use std::os::fd::AsRawFd as _;

        let (root, image, eval, scratch) = verifier_fixture("descriptor-inheritance");
        let assets = TrimulVerifierAssets::capture(&image, &eval, &scratch).unwrap();
        let image_flags = rustix::io::fcntl_getfd(&assets.snapshot.image_file).unwrap();
        assert!(image_flags.contains(rustix::io::FdFlags::CLOEXEC));
        assert!(assets.snapshot.image_file.as_raw_fd() > 2);
        for file in &assets.snapshot.sealed_eval_files {
            assert!(file.file.as_raw_fd() > 2);
            let flags = rustix::io::fcntl_getfd(&file.file).unwrap();
            assert!(
                flags.contains(rustix::io::FdFlags::CLOEXEC),
                "{} could cross a launcher exec",
                file.relative_path.display()
            );
        }

        drop(assets);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[allow(clippy::cognitive_complexity)] // one assertion per sealed invocation asset/boundary
    fn invocation_verifier_submission_and_specs_stay_sealed_between_phases() {
        use std::io::Write as _;
        use std::os::fd::AsRawFd as _;

        let submission = "def custom_kernel(data):\n    return data\n";
        let test_spec = "seqlen: 8; bs: 1";
        let bench_spec = "seqlen: 16; bs: 1";
        let assets = SealedInvocationAssets::capture(submission, test_spec, bench_spec).unwrap();
        assets.verify().unwrap();
        assert_eq!(assets.files.len(), 4);
        assert_eq!(
            std::fs::read(assets.files[1].0.descriptor_path()).unwrap(),
            submission.as_bytes()
        );
        assert_eq!(
            std::fs::read(assets.files[2].0.descriptor_path()).unwrap(),
            test_spec.as_bytes()
        );
        assert_eq!(
            std::fs::read(assets.files[3].0.descriptor_path()).unwrap(),
            bench_spec.as_bytes()
        );
        for (file, destination) in &assets.files {
            assert!(file.file.as_raw_fd() > 2);
            assert!(
                rustix::io::fcntl_getfd(&file.file)
                    .unwrap()
                    .contains(rustix::io::FdFlags::CLOEXEC),
                "{} could cross a launcher exec",
                file.relative_path.display()
            );
            assert!(destination.starts_with("/opt/ferrl-verifier/"));
            let mut attacker = std::fs::OpenOptions::new()
                .write(true)
                .open(file.descriptor_path())
                .unwrap();
            let error = attacker
                .write_all(b"between-phase substitution")
                .unwrap_err();
            assert_eq!(error.raw_os_error(), Some(1));
        }
        assert!(assets
            .binds()
            .iter()
            .all(|bind| bind.mode == crate::sandbox::BindMode::ReadOnly));
        assets.verify().unwrap();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn verifier_assets_reject_in_flight_mutation_between_sandbox_open_and_use() {
        use std::io::Write as _;

        let (root, image, eval, scratch) = verifier_fixture("in-flight-mutation");
        let assets = TrimulVerifierAssets::capture(&image, &eval, &scratch).unwrap();
        let reward = TrimulReward::new(assets.clone(), &scratch);
        let spec = reward.build_run_spec(&scratch.join("candidate"));
        assets.verify_current().unwrap();

        let eval_bind = spec
            .binds
            .iter()
            .find(|bind| bind.dst == Path::new("/opt/ferrl-verifier/eval.py"))
            .unwrap();
        for (target, expected) in [
            (spec.image.as_path(), b"exact image".as_slice()),
            (eval_bind.src.as_path(), b"# exact eval\n".as_slice()),
        ] {
            let mut attacker = std::fs::OpenOptions::new()
                .write(true)
                .open(target)
                .unwrap();
            let error = attacker.write_all(b"# forged verifier\n").unwrap_err();
            assert_eq!(error.raw_os_error(), Some(1));
            assert_eq!(std::fs::read(target).unwrap(), expected);
        }
        assets.verify_current().unwrap();

        drop(reward);
        drop(assets);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn verifier_assets_capture_authenticates_contents_after_sealing() {
        use std::os::unix::fs::FileExt as _;

        let (root, image, eval_dir, _scratch) = verifier_fixture("pre-seal-mutation");
        let image_error = capture_image_with_hook(&image, |file| {
            assert_eq!(file.write_at(b"X", 0).unwrap(), 1);
        })
        .unwrap_err()
        .to_string();
        assert!(
            image_error.contains("does not match its captured identity"),
            "{image_error}"
        );

        let eval = capture_eval_bundle(&eval_dir).unwrap();
        let eval_error =
            seal_eval_bundle_with_hook(&eval.files, &eval.identity, |relative_path, file| {
                if relative_path == Path::new("eval.py") {
                    assert_eq!(file.write_at(b"X", 0).unwrap(), 1);
                }
            })
            .unwrap_err()
            .to_string();
        assert!(
            eval_error.contains("does not match its captured identity"),
            "{eval_error}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn verifier_assets_reject_image_eval_and_task_substitution() {
        for target in ["image.sif", "eval/eval.py", "eval/task.yml"] {
            let (root, image, eval, scratch) = verifier_fixture(target);
            let assets = TrimulVerifierAssets::capture(&image, &eval, &scratch).unwrap();
            let target = root.join(target);
            std::fs::write(&target, b"replacement").unwrap();
            let error = assets.verify_current().unwrap_err().to_string();
            assert!(
                error.contains("changed after verifier attestation"),
                "{}: {error}",
                target.display()
            );
            drop(assets);
            let _ = std::fs::remove_dir_all(root);
        }
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
                ..TrimulEvalOutput::default()
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
            output: evidenced_output("", ""),
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
        let metadata = r
            .reward_metadata(
                "def custom_kernel(data): return data",
                &fast,
                training_reward,
            )
            .unwrap();
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
    fn public_verifier_apis_reject_empty_test_or_benchmark_sets_before_launch() {
        let sample = Sample::new("write a faster TriMul kernel", ());
        let completion = "```python\ndef custom_kernel(data):\n    return data[0]\n```".to_string();
        for (configured, expected) in [
            (
                reward().with_cases(Vec::new(), vec![case(16, false, Distribution::Normal)]),
                "at least one correctness case",
            ),
            (
                reward().with_cases(vec![case(8, true, Distribution::Normal)], Vec::new()),
                "at least one benchmark case",
            ),
        ] {
            let group_error = configured
                .reward_group_detailed(&sample, std::slice::from_ref(&completion))
                .unwrap_err()
                .to_string();
            assert!(group_error.contains(expected), "{group_error}");
            let verification_error = configured
                .verify_submission("def custom_kernel(data):\n    return data[0]\n")
                .unwrap_err()
                .to_string();
            assert!(
                verification_error.contains(expected),
                "{verification_error}"
            );
        }
    }

    #[test]
    fn verifier_max_procs_defaults_and_wires_to_run_spec() {
        let default_spec = reward().build_run_spec(Path::new("/tmp/scratch"));
        assert_eq!(
            default_spec.limits.max_procs,
            Some(DEFAULT_VERIFIER_MAX_PROCS)
        );

        let custom_spec = reward()
            .with_verifier_max_procs(2048)
            .build_run_spec(Path::new("/tmp/scratch"));
        assert_eq!(custom_spec.limits.max_procs, Some(2048));

        let clamped_spec = reward()
            .with_verifier_max_procs(0)
            .build_run_spec(Path::new("/tmp/scratch"));
        assert_eq!(clamped_spec.limits.max_procs, Some(1));
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
                ..TrimulEvalOutput::default()
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

    #[cfg(unix)]
    #[test]
    fn pre_verifier_infrastructure_failure_never_earns_the_format_floor() {
        let (root, image, eval, scratch) = verifier_fixture("pre-entry-infrastructure");
        let assets = TrimulVerifierAssets::capture(&image, &eval, &scratch).unwrap();
        let r = TrimulReward::new(assets, &scratch)
            .with_cases(
                vec![case(8, true, Distribution::Normal)],
                vec![case(16, false, Distribution::Cauchy)],
            )
            .with_verifier_executor_socket("/no/such/ferrl-verifier-executor.sock");
        let sample = Sample::new("write a faster TriMul kernel", ());
        let completion = "final:\n```python\ndef custom_kernel(data):\n    return data\n```\n";

        let error = r
            .reward_group_detailed(&sample, &[completion.to_string()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("staged verifier execution failed"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // phase matrix keeps infrastructure and candidate outcomes adjacent
    fn verifier_entry_requires_trusted_worker_and_gpu_proof_for_each_phase() {
        let spec = reward().build_run_spec(Path::new("/tmp/scratch"));
        assert_eq!(
            spec.command,
            vec![
                "python",
                "-I",
                FERRL_EVAL_DRIVER_PATH,
                TEST_SPEC_PATH,
                BENCH_SPEC_PATH,
            ]
        );
        assert!(spec.protected_output.is_some());
        assert!(spec.env.iter().any(
            |(key, value)| key == "FERRL_GRADE_SOCKET" && value == "/work/.ferrl-grade-v1.sock"
        ));

        let protect = FERRL_EVAL_DRIVER
            .find("_prctl(PR_SET_DUMPABLE, 0)")
            .expect("the grade parent is protected before trusted imports");
        let trusted_import = FERRL_EVAL_DRIVER
            .find("import eval as upstream")
            .expect("the sealed driver imports the attested verifier");
        let payload = FERRL_EVAL_DRIVER
            .split_once("def _candidate_payload")
            .expect("candidate code has a separate payload-process target")
            .1;
        let payload = payload
            .split_once("def _read_parent_pid")
            .expect("the payload target has a bounded source region")
            .0;
        let ready = payload
            .find("_send_payload(results, b\"READY\")")
            .expect("the payload process proves CUDA initialization");
        let ack = payload
            .find("commands.recv_bytes(MAX_STATUS_BYTES) != b\"ACK-v2\"")
            .expect("candidate import follows the controller acknowledgement");
        let traced_entry = payload
            .find("frame.f_code.co_filename == SUBMISSION_PATH")
            .expect("entry is observed from the actual submission module frame");
        let entry_event = payload
            .find("_send_payload(results, b\"ENTRY\")")
            .expect("payload announces candidate entry");
        let entry_ack = payload
            .find("commands.recv_bytes(MAX_STATUS_BYTES) != ENTRY_ACK")
            .expect("candidate execution waits for protected-parent entry acknowledgement");
        let trace_release = payload
            .find("sys.settrace(None)\n            return None")
            .expect("candidate tracing is released only after acknowledgement");
        let candidate_import = payload
            .find("importlib.import_module(\"submission\")")
            .expect("candidate import is explicit and deferred");
        assert!(protect < trusted_import);
        assert!(ready < ack && ack < traced_entry && traced_entry < candidate_import);
        assert!(traced_entry < entry_event && entry_event < entry_ack && entry_ack < trace_release);
        assert!(FERRL_EVAL_DRIVER.contains("multiprocessing.get_context(\"spawn\")"));
        assert!(FERRL_EVAL_DRIVER.contains("time.perf_counter_ns()"));
        assert!(FERRL_EVAL_DRIVER.contains("_CHECK_IMPLEMENTATION(data, output)"));
        assert!(FERRL_EVAL_DRIVER.contains("_CALCULATE_STATS(durations)"));
        let controller = FERRL_EVAL_DRIVER
            .split_once("def _candidate_worker")
            .expect("trusted candidate protocol has a separate controller process")
            .1
            .split_once("class CandidateSession")
            .expect("the controller target has a bounded source region")
            .0;
        assert!(!payload.contains("_send_status"));
        assert!(!payload.contains("outputs.send_bytes"));
        assert!(!controller.contains("importlib.import_module"));
        assert!(!controller.contains("kernel(candidate_data)"));
        assert!(controller.contains(
            "_send_status(status, b\"IMPORT_ERROR\" if entered else b\"IMPORT_REJECTED\")"
        ));
        assert!(controller.contains("if event == b\"IMPORT_REJECTED\" and entered:"));
        assert!(controller.contains("_send_status(status, b\"IMPORT_CHANNEL_CORRUPTED\")"));
        assert!(controller.contains("if import_events > MAX_STATUS_EVENTS:"));
        assert!(controller.contains("payload_commands.send_bytes(acknowledgement)"));
        assert!(FERRL_EVAL_DRIVER.contains("self.commands.send_bytes(ENTRY_ACK)"));
        assert!(FERRL_EVAL_DRIVER.contains("self.import_status_events += 1"));
        assert!(FERRL_EVAL_DRIVER.contains("candidate worker sent duplicate entry"));
        assert!(FERRL_EVAL_DRIVER.contains("payload-results-channel-v1"));
        assert!(controller.contains("_send_status(status, b\"OUTPUT_READY\")"));
        assert!(controller.contains("outputs.send_bytes(payload)"));
        assert!(payload.contains("_send_payload(results, b\"OUTPUT\", raw_output)"));
        assert!(payload.contains(
            "_send_payload(results, b\"IMPORT_ERROR\" if entry_sent else b\"IMPORT_REJECTED\")"
        ));
        assert!(FERRL_EVAL_DRIVER.contains("_cpu_clone(value)"));
        assert!(FERRL_EVAL_DRIVER.contains("torch.frombuffer("));
        assert!(FERRL_EVAL_DRIVER.contains("MAX_INPUT_BYTES = 4 * 1024 * 1024 * 1024"));
        assert!(!FERRL_EVAL_DRIVER.contains("candidate_data, shared_output"));
        assert!(!FERRL_EVAL_DRIVER.contains("torch.multiprocessing"));
        assert!(FERRL_EVAL_DRIVER.contains("trusted correctness checker failed"));
        assert!(!FERRL_EVAL_DRIVER.contains("upstream.run_testing"));
        assert!(!FERRL_EVAL_DRIVER.contains("multiprocessing.Pool"));

        let execute = FERRL_EVAL_DRIVER
            .split_once("def execute(self, candidate_data, checked_output):")
            .expect("the protected parent owns candidate handoff and output capture")
            .1;
        let timer = execute
            .find("started = time.perf_counter_ns()")
            .expect("protected timing starts explicitly");
        let handoff = execute
            .find("self.commands.send_bytes(b\"RUN-v2\\0\" + payload)")
            .expect("the CPU-only input bytes are handed off explicitly");
        let private_capture = execute
            .find("_COPY_TENSOR(checked_output, cpu_output)")
            .expect("the parent reconstructs a private result from CPU bytes");
        let elapsed = execute
            .find("elapsed = time.perf_counter_ns() - started")
            .expect("timing ends after bounded result receipt");
        assert!(timer < handoff && handoff < elapsed && elapsed < private_capture);
        assert!(FERRL_EVAL_DRIVER.contains("_wrap_check(data, checked_output)"));
        let session = FERRL_EVAL_DRIVER
            .split_once("class CandidateSession")
            .expect("the protected parent owns candidate-session classification")
            .1
            .split_once("def _wrap_check")
            .expect("the candidate session has a bounded source region")
            .0;
        assert!(session.contains("except CandidateFailure as error:"));
        assert!(session.contains("if not self.entered:"));
        assert!(session
            .contains("self.logger.log(\"ferrl-candidate-rejected\", f\"{self.mode}-import-v1\")"));

        let main = FERRL_EVAL_DRIVER
            .split_once("def main():")
            .expect("the sealed driver has one protected main boundary")
            .1;
        let grade_connect = main
            .find("logger = GradeLogger(grade_socket)")
            .expect("the protected parent opens the only grade endpoint");
        let device_identity = main
            .find("device_identity = _executing_device_identity()")
            .expect("the trusted parent authenticates the executing CUDA device");
        let candidate_start = main
            .find("if not _run_testing(")
            .expect("candidate test execution is explicit");
        assert!(grade_connect < device_identity && device_identity < candidate_start);
        assert!(main.contains("if seed is not None and not 0 <= seed <= MAX_CASE_SEED:"));
        assert!(main.contains("trusted case-generation seed is outside unsigned 32-bit range"));
        assert!(main.contains("_SET_SEED(42 if seed is None else seed)"));
        assert!(!main.contains("_SET_SEED(seed or 42)"));
        assert!(main.contains("reason = _bounded_message(f\"{type(error).__name__}: {error}\")"));
        assert_eq!(
            FERRL_EVAL_DRIVER
                .matches("logger.log(\"ferrl-executing-device\", device_identity)")
                .count(),
            2,
            "both protected phases must report the same trusted identity"
        );
        assert!(FERRL_EVAL_DRIVER.contains("\"cuDeviceGetUuid_v2\""));
        assert!(FERRL_EVAL_DRIVER.contains("driver.cuDeviceGetPCIBusId"));
        assert!(FERRL_EVAL_DRIVER.contains("self.socket.set_inheritable(False)"));

        let missing_runtime = require_trimul_verifier_entry(RunOutcome {
            status: RunStatus::Exited(0),
            stdout: "ferrl-entry: test-v4\ncheck: pass\n".to_string(),
            stderr: "python: not found".to_string(),
            protected_output: String::new(),
            wall: Duration::from_millis(1),
        })
        .unwrap_err();
        assert!(matches!(
            missing_runtime,
            SandboxError::Infrastructure {
                status: RunStatus::Exited(0),
                ref stderr,
            } if stderr.contains("test verifier did not reach trusted worker/GPU entry")
                && stderr.contains("python: not found")
        ));

        let trusted_benchmark_import_failure = require_trimul_verifier_entry(RunOutcome {
            status: RunStatus::Exited(0),
            stdout: "candidate forged stdout\n".to_string(),
            stderr: "ImportError: trusted benchmark dependency".to_string(),
            protected_output: format!(
                "ferrl-entry: {TEST_VERIFIER_ENTRY}\ncheck: pass\ntest-exit: 0\n\
                 {RESULT_SPLIT}\nbenchmark-exit: 1\n"
            ),
            wall: Duration::from_millis(1),
        })
        .unwrap_err();
        assert!(matches!(
            trusted_benchmark_import_failure,
            SandboxError::Infrastructure {
                status: RunStatus::Exited(0),
                ref stderr,
            } if stderr.contains("benchmark verifier did not reach trusted worker/GPU entry")
                && stderr.contains("trusted benchmark dependency")
        ));

        let candidate_failure = require_trimul_verifier_entry(RunOutcome {
            status: RunStatus::Exited(7),
            stdout: String::new(),
            stderr: "candidate failed\n".to_string(),
            protected_output: format!(
                "{}\
                 ferrl-entry: {TEST_VERIFIER_ENTRY}\ntest-exit: 7\n",
                protected_phase_prelude(),
            ),
            wall: Duration::from_millis(1),
        })
        .unwrap();
        assert_eq!(candidate_failure.status, RunStatus::Exited(7));

        let benchmark_candidate_failure = require_trimul_verifier_entry(RunOutcome {
            status: RunStatus::Exited(0),
            stdout: String::new(),
            stderr: "candidate benchmark failed\n".to_string(),
            protected_output: format!(
                "{}\
                 ferrl-entry: {TEST_VERIFIER_ENTRY}\ncheck: pass\ntest-exit: 0\n\
                 {RESULT_SPLIT}\n{}\
                 ferrl-entry: {BENCHMARK_VERIFIER_ENTRY}\nbenchmark-exit: 7\n",
                protected_phase_prelude(),
                protected_phase_prelude(),
            ),
            wall: Duration::from_millis(1),
        })
        .unwrap();
        assert_eq!(benchmark_candidate_failure.status, RunStatus::Exited(0));

        let rejected_candidate = require_trimul_verifier_entry(RunOutcome {
            status: RunStatus::Exited(0),
            stdout: String::new(),
            stderr: String::new(),
            protected_output: format!(
                "{}\
                 ferrl-candidate-rejected: test-import-v1\ncheck: fail\ntest-exit: 112\n",
                protected_phase_prelude(),
            ),
            wall: Duration::from_millis(1),
        })
        .unwrap();
        assert_eq!(rejected_candidate.status, RunStatus::Exited(0));

        for infrastructure in [
            RunOutcome {
                status: RunStatus::Exited(0),
                stdout: String::new(),
                stderr: "trusted checker failed".to_string(),
                protected_output: format!(
                    "ferrl-timing-metric: {TRIMUL_TIMING_METRIC}\n\
                     ferrl-entry: {TEST_VERIFIER_ENTRY}\n\
                     ferrl-infrastructure: v1 phase=test\n"
                ),
                wall: Duration::from_millis(1),
            },
            RunOutcome {
                status: RunStatus::Exited(TRIMUL_INFRASTRUCTURE_EXIT),
                stdout: String::new(),
                stderr: "trusted parent exited".to_string(),
                protected_output: String::new(),
                wall: Duration::from_millis(1),
            },
            RunOutcome {
                status: RunStatus::Exited(0),
                stdout: String::new(),
                stderr: "trusted statistics failed".to_string(),
                protected_output: format!(
                    "ferrl-timing-metric: {TRIMUL_TIMING_METRIC}\n\
                     ferrl-entry: {TEST_VERIFIER_ENTRY}\ncheck: pass\ntest-exit: 0\n\
                     {RESULT_SPLIT}\nferrl-timing-metric: {TRIMUL_TIMING_METRIC}\n\
                     ferrl-entry: {BENCHMARK_VERIFIER_ENTRY}\n\
                     ferrl-infrastructure: v1 phase=benchmark\n"
                ),
                wall: Duration::from_millis(1),
            },
        ] {
            let expected = infrastructure
                .protected_output
                .lines()
                .find(|line| line.trim_start().starts_with("ferrl-infrastructure: v1"))
                .map_or("reserved exit 114", str::trim)
                .to_string();
            assert!(matches!(
                require_trimul_verifier_entry(infrastructure),
                Err(SandboxError::Infrastructure { stderr, .. }) if stderr.contains(&expected)
            ));
        }
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
            output: evidenced_output(
                "check: pass\ntest-exit: 1\n",
                &format!(
                    "Traceback: candidate crashed\n{}",
                    "x".repeat(EVAL_OUTPUT_TAIL_LIMIT_BYTES + 8)
                ),
            ),
            test_check: Some("pass".to_string()),
            test_exit: Some(1),
            benchmark_exit: None,
            has_benchmark_section: false,
        };

        let training_reward = r.reward_from_extracted_eval(&eval);
        let metadata = r.reward_metadata(source, &eval, training_reward).unwrap();
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
            output: evidenced_output("", ""),
            test_check: Some("pass".to_string()),
            test_exit: Some(0),
            benchmark_exit: Some(0),
            has_benchmark_section: true,
        };

        let training_reward = r.reward_from_extracted_eval(&eval);
        let metadata = r
            .reward_metadata(
                "def custom_kernel(data): return data",
                &eval,
                training_reward,
            )
            .unwrap();
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
            output: evidenced_output("test-exit: 1\n", "RuntimeError: candidate test failed\n"),
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
        let metadata = r
            .reward_metadata(
                "def custom_kernel(data): return data",
                &eval,
                training_reward,
            )
            .unwrap();
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
        for (_, destination) in SANDBOX_EVAL_FILES {
            let eval = spec
                .binds
                .iter()
                .find(|bind| bind.dst == Path::new(destination))
                .expect("each verifier file is bound independently");
            assert_eq!(eval.mode, crate::sandbox::BindMode::ReadOnly);
        }
        let work = spec
            .binds
            .iter()
            .find(|b| b.dst == Path::new("/work"))
            .expect("scratch is bound");
        assert_eq!(work.mode, crate::sandbox::BindMode::ReadWrite);
        assert_eq!(work.total_limit, Some(1 << 30));
    }

    #[test]
    fn run_spec_uses_one_protected_driver_and_no_grade_fd_alias() {
        let spec = reward().build_run_spec(Path::new("/tmp/scratch"));
        assert_eq!(spec.command[0..3], ["python", "-I", FERRL_EVAL_DRIVER_PATH]);
        assert_eq!(spec.command[3], TEST_SPEC_PATH);
        assert_eq!(spec.command[4], BENCH_SPEC_PATH);
        assert!(spec.protected_output.is_some());
        assert!(!spec.command.iter().any(|value| value.contains("3>&1")));
        assert!(!spec.env.iter().any(|(key, _)| key == "POPCORN_FD"));
    }

    #[test]
    fn protected_driver_reports_phase_status_after_candidate_teardown() {
        assert!(FERRL_EVAL_DRIVER.contains("logger.log(\"test-exit\""));
        assert!(FERRL_EVAL_DRIVER.contains("logger.log(\"benchmark-exit\""));
        let preparation = FERRL_EVAL_DRIVER
            .find("phase = \"preparation\"")
            .expect("trusted preparation has an infrastructure phase");
        let trusted_preparation = FERRL_EVAL_DRIVER
            .find("_prctl(PR_SET_CHILD_SUBREAPER, 1)")
            .expect("trusted preparation is explicit");
        assert!(preparation < trusted_preparation);
        let final_kill = FERRL_EVAL_DRIVER
            .rfind("_kill_candidate_tree()")
            .expect("the driver has a final candidate-tree kill");
        let grade_close = FERRL_EVAL_DRIVER
            .rfind("logger.close()")
            .expect("the verifier closes its exclusive grade endpoint");
        assert!(final_kill < grade_close);
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
    fn empty_test_or_benchmark_sets_fail_before_candidate_scoring() {
        let sample = Sample::new("write a faster TriMul kernel", ());
        let completion = "Sorry, no code.";
        let mut no_tests = reward();
        no_tests.test_cases.clear();
        let error = no_tests
            .reward(&sample, completion)
            .unwrap_err()
            .to_string();
        assert!(error.contains("at least one correctness case"));

        let mut no_benchmarks = reward();
        no_benchmarks.benchmark_cases.clear();
        let error = no_benchmarks
            .reward_group_detailed(&sample, &[completion.to_string()])
            .unwrap_err()
            .to_string();
        assert!(error.contains("at least one benchmark case"));
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
                ..TrimulEvalOutput::default()
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
                ..TrimulEvalOutput::default()
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
