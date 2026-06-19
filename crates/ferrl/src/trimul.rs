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
//! 1. Extract the `custom_kernel` source from the completion (the first fenced
//!    Python code block) — [`extract_submission`].
//! 2. Stage a node-local scratch dir: the candidate as `submission.py`, plus a
//!    generated test-spec and benchmark-spec file ([`render_spec`]).
//! 3. Run the eval in the sandbox ([`crate::sandbox::ApptainerSandbox`]): the pinned
//!    GPUMODE eval files are bound **read-only**, the scratch **read-write**, the GPU
//!    exposed (`--nv`), and the **network denied**. Inside, the GPUMODE `eval.py`
//!    runs `test` (correctness) then `benchmark` (variance-aware CUDA-event timing),
//!    writing results to a file via its `POPCORN_FD` channel — never the (capped)
//!    stdout, so a noisy or hostile payload cannot drown the result.
//! 4. Read the result files and map to a reward: **`0` if the candidate is missing,
//!    crashes, or fails any correctness case; otherwise the geometric-mean speedup
//!    over a reference baseline.** GRPO normalizes rewards within a group, so a
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
//! ## Testing split (as in [`crate::sandbox`])
//!
//! The pure pieces — submission extraction, spec rendering, result parsing, the
//! run-spec builder, and the reward math — are unit-tested in CI. The real GPU eval
//! is a `gate`-feature integration test (`tests/trimul_gate.rs`), run on an `sm_80`
//! node against the eval image; like the GPU tests it is never compiled in CI.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::reward::{RewardError, RewardFn};
use crate::sample::Sample;
use crate::sandbox::{ApptainerSandbox, Bind, ResourceLimits, RunSpec, Sandbox};

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

/// Extract the candidate `custom_kernel` source from a completion: the body of the
/// **first** fenced code block (a triple-backtick fence, with or without a language
/// tag). Returns `None` if there is no closed block or the block is empty.
///
/// The first block is the model's genuine attempt — a base model with no stop token
/// tends to parrot the format afterwards, so a later block is usually a repeat (the
/// same rule [`crate::math`] applies to answer spans).
#[must_use]
pub fn extract_submission(completion: &str) -> Option<String> {
    let open = completion.find("```")?;
    // Skip the optional language tag up to the end of the fence's opening line.
    let after_fence = &completion[open + 3..];
    let body_start = after_fence.find('\n')? + 1;
    let body = &after_fence[body_start..];
    let close = body.find("```")?;
    let code = body[..close].trim_end();
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

/// The TriMul discovery reward: runs a candidate kernel in the sandboxed eval image
/// and scores it on correctness + speed. Construct with [`TrimulReward::new`].
#[derive(Debug, Clone)]
pub struct TrimulReward {
    /// The eval image (the pinned PyTorch+Triton `.sif`).
    image: PathBuf,
    /// The pinned GPUMODE eval bundle (`eval.py`/`reference.py`/`task.py`/`utils.py`),
    /// bound **read-only**.
    eval_dir: PathBuf,
    /// Where per-candidate scratch dirs are created — node-local (e.g. `/tmp`).
    scratch_root: PathBuf,
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
    /// The sandbox backend.
    sandbox: ApptainerSandbox,
}

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
            test_cases: Vec::new(),
            benchmark_cases: Vec::new(),
            secret_seed: 0,
            baseline_ns: None,
            wall: Duration::from_secs(600),
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

    /// Resource ceilings for one eval. `address_space` is left unset — a CUDA process
    /// reserves a huge virtual range an address-space cap would wrongly kill.
    fn limits(&self) -> ResourceLimits {
        ResourceLimits {
            wall: self.wall,
            address_space: None,
            ..ResourceLimits::default()
        }
    }

    /// The `bash -c` program run inside the image: copy the read-only eval files next
    /// to the staged `submission.py`, run `test`, and — only if it passes — `benchmark`,
    /// each writing its result via fd 3 (`POPCORN_FD`). A trailing `true` keeps the
    /// shell's exit status clean; ferrl reads the result files, not the exit code.
    fn in_container_command() -> String {
        "cp /eval/eval.py /eval/reference.py /eval/task.py /eval/utils.py . && \
         { POPCORN_FD=3 python eval.py test test_spec.txt 3>test_result.txt && \
           POPCORN_FD=3 python eval.py benchmark bench_spec.txt 3>bench_result.txt; }; \
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
            Bind::rw(scratch, "/work"),
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
    fn run_eval(&self, submission: &str) -> Result<f32, RewardError> {
        let scratch = self.make_scratch()?;
        let result = self.eval_in(&scratch, submission);
        // Best-effort cleanup; the scratch is node-local and disposable.
        let _ = std::fs::remove_dir_all(&scratch);
        result
    }

    /// The body of [`run_eval`](Self::run_eval), split out so the scratch is always
    /// cleaned up.
    fn eval_in(&self, scratch: &Path, submission: &str) -> Result<f32, RewardError> {
        std::fs::create_dir_all(scratch.join("cache")).map_err(RewardError::verifier)?;
        write(scratch, "submission.py", submission)?;
        write(scratch, "test_spec.txt", &render_spec(&self.test_cases))?;
        write(
            scratch,
            "bench_spec.txt",
            &render_spec(&self.benchmark_cases),
        )?;

        // The verdict lives in the POPCORN result files (read below), not the exit
        // status — the in-container command ends with `true`, so a crashing candidate
        // returns cleanly and is scored a failure from its empty result file.
        self.sandbox
            .run(&self.build_run_spec(scratch))
            .map_err(RewardError::verifier)?;

        let correct = test_passed(&read(scratch, "test_result.txt"));
        let geo = if correct {
            geomean(&benchmark_means_ns(&read(scratch, "bench_result.txt")))
        } else {
            None
        };
        Ok(self.reward_value(correct, geo))
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

/// Read `dir/name` UTF-8-lossily, or `""` if absent (a missing result file means the
/// eval never got that far — scored as a failure, not an error).
fn read(dir: &Path, name: &str) -> String {
    std::fs::read(dir.join(name))
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

impl RewardFn for TrimulReward {
    type Target = ();

    fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
        match extract_submission(completion) {
            Some(code) => self.run_eval(&code),
            // No code block at all: nothing to run, a zero reward.
            None => Ok(0.0),
        }
    }
    // No `reward_group` override: a shared GPU scores candidates one at a time, so the
    // default (map `reward` over the group) is what we want.
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
    fn extract_submission_takes_the_first_fenced_block() {
        let completion =
            "Here is my kernel:\n```python\ndef custom_kernel(data):\n    return data\n```\nrest";
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
    fn extract_submission_is_none_without_a_closed_block() {
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
    fn reward_falls_back_to_inverse_time_without_a_baseline() {
        // 1e9 / geo: a faster (smaller) geo yields a larger reward.
        let r = reward();
        assert!(r.reward_value(true, Some(1e6)) < r.reward_value(true, Some(1e5)));
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
    }

    #[test]
    fn in_container_command_runs_test_then_benchmark_via_fd3() {
        let cmd = TrimulReward::in_container_command();
        assert!(cmd.contains("eval.py test test_spec.txt 3>test_result.txt"));
        assert!(cmd.contains("eval.py benchmark bench_spec.txt 3>bench_result.txt"));
        // benchmark is gated on test passing.
        assert!(cmd.contains("test_result.txt && "));
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
}
