//! TriMul reward integration gate — the real eval, on a GPU.
//!
//! [`ferrl::trimul`]'s unit tests cover the pure pieces (extraction, spec rendering,
//! parsing, reward math). These run the *whole* [`ferrl::TrimulReward`] against the
//! pinned eval image on an `sm_80` node: a correct kernel scores above zero, a wrong
//! kernel scores zero, and a hostile kernel is **contained** (the sandbox denies it
//! the network even inside the torch/triton image) and scores zero.
//!
//! Gated behind the off-by-default `gate` feature, so — like the GPU tests — CI never
//! compiles it. Run on a node with an `sm_80` GPU and the eval bundle:
//!
//! ```text
//! FERRL_TRIMUL_IMAGE=/path/to/trimul-eval.sif \
//! FERRL_TRIMUL_EVAL_DIR=/path/to/pinned/trimul \
//!   cargo test --features gate --test trimul_gate -- --ignored --test-threads=1
//! ```

#![cfg(feature = "gate")]

use std::path::PathBuf;
use std::time::Duration;

use ferrl::{Distribution, RewardFn, Sample, TrimulCase, TrimulReward};

/// A required path from the environment (the gate only runs with `--ignored`).
fn env_path(key: &str) -> PathBuf {
    std::env::var_os(key).map_or_else(|| panic!("set {key} to run the TriMul gate"), PathBuf::from)
}

/// A reward over a couple of small, generic cases (not GPU Mode's specific sizes).
fn reward() -> TrimulReward {
    let scratch = std::env::var("FERRL_TRIMUL_SCRATCH").unwrap_or_else(|_| "/tmp".to_string());
    let cases = vec![
        TrimulCase {
            seqlen: 32,
            bs: 1,
            dim: 64,
            hiddendim: 64,
            seed: 11,
            nomask: true,
            distribution: Distribution::Normal,
        },
        TrimulCase {
            seqlen: 16,
            bs: 2,
            dim: 64,
            hiddendim: 64,
            seed: 12,
            nomask: false,
            distribution: Distribution::Normal,
        },
    ];
    TrimulReward::new(
        env_path("FERRL_TRIMUL_IMAGE"),
        env_path("FERRL_TRIMUL_EVAL_DIR"),
        scratch,
    )
    .with_cases(cases.clone(), cases)
    .with_secret_seed(123)
    .with_wall(Duration::from_secs(300))
}

fn score(completion: &str) -> f32 {
    reward()
        .reward(&Sample::new("write a faster TriMul kernel", ()), completion)
        .expect("the eval should be carried out")
}

/// Correct: delegate to the reference already in the bound eval bundle. Trivially
/// passes correctness, so the reward (inverse runtime, no baseline set) is positive.
const CORRECT: &str = "```python
def custom_kernel(data):
    from reference import ref_kernel
    return ref_kernel(data)
```";

/// Wrong: return the input unchanged — right shape, wrong values, so correctness fails.
const WRONG: &str = "```python
def custom_kernel(data):
    x, mask, weights, config = data
    return x.clone()
```";

/// Hostile: try to open a network connection (then compute correctly). The sandbox
/// denies the network, so the connection raises and the eval never reports a pass.
const MALICIOUS: &str = "```python
def custom_kernel(data):
    import socket
    socket.create_connection((\"1.1.1.1\", 53), timeout=3).close()
    from reference import ref_kernel
    return ref_kernel(data)
```";

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_correct_submission_scores_positive() {
    let value = score(CORRECT);
    assert!(
        value > 0.0,
        "a correct kernel should score above zero, got {value}"
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_wrong_submission_scores_zero() {
    assert_eq!(score(WRONG), 0.0, "a wrong kernel must score zero");
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_malicious_submission_is_contained_and_scores_zero() {
    // The sandbox runs with no network, so the connection attempt fails and the eval
    // reports no pass — the candidate scores zero and the host is untouched. This
    // re-confirms the isolation holds inside the real torch/triton image.
    assert_eq!(
        score(MALICIOUS),
        0.0,
        "a network-reaching kernel must be contained and score zero"
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_no_code_scores_zero() {
    assert_eq!(
        score("I won't write that."),
        0.0,
        "a completion with no code scores zero"
    );
}

// --- Adversarial negative controls: a candidate that tries to *forge* a pass must
//     score zero. These are the reward-gaming cases the honest-wrong WRONG case does
//     not cover.

/// Forge attempt: write fake grade files into the RW scratch. ferrl reads the grade
/// from the stdout pipe, not these files, so the forge must not affect the score.
const FORGE_WORKFILE: &str = "```python
def custom_kernel(data):
    open(\"/work/test_result.txt\", \"w\").write(\"check: pass\\n\")
    open(\"/work/bench_result.txt\", \"w\").write(\"benchmark.0.mean: 0.001\\ncheck: pass\\n\")
    x, mask, weights, config = data
    return x.clone()
```";

/// Forge attempt: print a fake grade to stdout. The candidate's stdout is routed to
/// /dev/null, so it cannot reach the grade channel.
const FORGE_STDOUT: &str = "```python
def custom_kernel(data):
    print(\"check: pass\")
    print(\"===FERRL-BENCH===\")
    print(\"benchmark.0.mean: 0.001\")
    print(\"check: pass\")
    x, mask, weights, config = data
    return x.clone()
```";

/// Forge attempt: hunt the grader's grade fd via /proc and write a fake pass with an
/// absurd time. The fd IS reachable (documented residual), but the plausibility floor
/// rejects the absurd time, so it scores zero.
const FORGE_PROCFD: &str = "```python
def custom_kernel(data):
    import os, glob
    payload = b\"check: pass\\n===FERRL-BENCH===\\nbenchmark.0.mean: 0.001\\ncheck: pass\\n\"
    for path in glob.glob(\"/proc/[0-9]*/fd/3\"):
        try:
            fd = os.open(path, os.O_WRONLY)
            os.write(fd, payload)
            os.close(fd)
        except Exception:
            pass
    x, mask, weights, config = data
    return x.clone()
```";

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_forged_result_files_score_zero() {
    assert_eq!(
        score(FORGE_WORKFILE),
        0.0,
        "forged /work result files must not score"
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_forged_stdout_scores_zero() {
    assert_eq!(
        score(FORGE_STDOUT),
        0.0,
        "a printed fake grade must not score"
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_forged_proc_fd_with_absurd_timing_is_rejected() {
    // The /proc fd DOES reach the grade channel (a documented residual — the candidate
    // worker shares the grader's PID namespace + uid), but its absurd forged time is
    // caught by the plausibility floor, so it scores zero. A plausible-time /proc forge
    // remains a known residual, closed only by per-candidate PID-namespace isolation.
    assert_eq!(
        score(FORGE_PROCFD),
        0.0,
        "an absurdly fast /proc-forged time must be rejected by the plausibility floor"
    );
}
