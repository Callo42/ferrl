//! TriMul reward integration gate — the real eval, on a GPU.
//!
//! [`ferrl::trimul`]'s unit tests cover the pure pieces (extraction, spec rendering,
//! parsing, reward math). These run the *whole* [`ferrl::TrimulReward`] against the
//! pinned eval image on an `sm_80` node: a correct kernel reaches the correctness
//! floor plus speed signal, a wrong kernel remains below the correctness floor, and a
//! hostile kernel is **contained** (the sandbox denies it the network even inside the
//! torch/triton image) and remains below the correctness floor.
//! A mutation-sensitive candidate also tries to replace every verifier,
//! submission, and rendered-case pathname during testing; the same exact sealed
//! bytes must remain in force through benchmarking.
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
use std::sync::OnceLock;
use std::time::Duration;

use ferrl::trimul::TrimulVerifierAssets;
use ferrl::{Distribution, RewardFn, Sample, TrimulCase, TrimulReward};

/// A required path from the environment (the gate only runs with `--ignored`).
fn env_path(key: &str) -> PathBuf {
    std::env::var_os(key).map_or_else(|| panic!("set {key} to run the TriMul gate"), PathBuf::from)
}

/// A reward over a couple of small, generic cases (not GPU Mode's specific sizes).
fn reward() -> TrimulReward {
    let scratch = std::env::var("FERRL_TRIMUL_SCRATCH").unwrap_or_else(|_| "/tmp".to_string());
    static VERIFIER_ASSETS: OnceLock<TrimulVerifierAssets> = OnceLock::new();
    let verifier_assets = VERIFIER_ASSETS
        .get_or_init(|| {
            let image = env_path("FERRL_TRIMUL_IMAGE");
            let eval_dir = env_path("FERRL_TRIMUL_EVAL_DIR");
            TrimulVerifierAssets::capture(&image, &eval_dir, &scratch)
                .expect("capture kernel-sealed verifier assets")
        })
        .clone();
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
    TrimulReward::new(verifier_assets, scratch)
        .with_cases(cases.clone(), cases)
        .with_secret_seed(123)
        .with_wall(Duration::from_secs(300))
}

fn score(completion: &str) -> f32 {
    reward()
        .reward(&Sample::new("write a faster TriMul kernel", ()), completion)
        .expect("the eval should be carried out")
}

fn assert_below_correctness_floor(value: f32, reason: &str) {
    assert!(
        (0.0..1.0).contains(&value),
        "{reason}; expected shaped failure reward in [0, 1), got {value}"
    );
}

/// Correct: delegate to the reference already in the bound eval bundle. Trivially
/// passes correctness and benchmark, so the reward reaches the correctness floor plus
/// the inverse-runtime speed component (no baseline set).
const CORRECT: &str = "```python
def custom_kernel(data):
    from reference import ref_kernel
    return ref_kernel(data)
```";

/// Wrong: return the input unchanged — right shape, wrong values, so correctness fails.
/// With the shaped reward it may still receive format/runnable/partial credit, but it
/// must stay below the correctness floor.
const WRONG: &str = "```python
def custom_kernel(data):
    x, mask, weights, config = data
    return x.clone()
```";

/// Hostile: try to open a network connection (then compute correctly). The sandbox
/// denies the network, so the connection raises and the eval never reports a pass.
/// The shaped reward may still give the runnable floor, but not correctness credit.
const MALICIOUS: &str = "```python
def custom_kernel(data):
    import socket
    socket.create_connection((\"1.1.1.1\", 53), timeout=3).close()
    from reference import ref_kernel
    return ref_kernel(data)
```";

/// Mutation attempt: replace every byte-bearing input after the sandbox has opened
/// the sealed verifier paths, while the candidate worker is executing. All paths remain
/// read-only; if any write succeeds, raising prevents an apparently correct result.
/// A successful correctness + benchmark outcome therefore proves the exact same
/// sealed verifier, submission, and case bytes survived the between-phase boundary.
const MUTATE_SEALED_ASSETS: &str = "```python
def custom_kernel(data):
    import os
    paths = (
        \"/opt/ferrl-verifier/eval.py\",
        \"/opt/ferrl-verifier/reference.py\",
        \"/opt/ferrl-verifier/task.py\",
        \"/opt/ferrl-verifier/utils.py\",
        \"/opt/ferrl-verifier/ferrl_eval.py\",
        \"/opt/ferrl-verifier/submission.py\",
        \"/opt/ferrl-verifier/test_spec.txt\",
        \"/opt/ferrl-verifier/bench_spec.txt\",
    )
    for path in paths:
        try:
            with open(path, \"wb\") as handle:
                handle.write(b\"forged verifier bytes\\n\")
        except OSError:
            continue
        raise RuntimeError(f\"sealed verifier path was writable: {path}\")
    # Cache/output is the only intentionally writable storage and survives the
    # shell's test/benchmark boundary without changing any scored input.
    os.makedirs(\"/work/cache/candidate-output\", exist_ok=True)
    with open(\"/work/cache/candidate-output/marker\", \"w\") as handle:
        handle.write(\"candidate-controlled cache bytes\\n\")
    from reference import ref_kernel
    return ref_kernel(data)
```";

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_correct_submission_scores_positive() {
    let outcomes = reward()
        .reward_group_detailed(
            &Sample::new("write a faster TriMul kernel", ()),
            &[CORRECT.to_string()],
        )
        .expect("the real eval should produce correctness and benchmark evidence");
    let outcome = &outcomes[0];
    assert!(
        outcome.reward >= 1.0,
        "a correct kernel should reach the default correctness floor, got {}: {outcome:#?}",
        outcome.reward,
    );
    let metadata = outcome
        .metadata
        .as_ref()
        .expect("the TriMul eval should return structured evidence");
    assert_eq!(metadata["correct"], serde_json::json!(true));
    assert_eq!(metadata["benchmark_exit"], serde_json::json!(0));
    assert!(
        metadata["geomean_ns"]
            .as_f64()
            .is_some_and(|value| value > 0.0),
        "the correct-candidate smoke must record a positive benchmark geomean: {metadata}"
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_wrong_submission_stays_below_correctness_floor() {
    assert_below_correctness_floor(score(WRONG), "a wrong kernel must not score as correct");
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_reference_baseline_is_measurable_and_positive() {
    // The guarded-pin baseline: run the bundled reference through the eval and read its
    // geometric-mean runtime. It must pass correctness (it *is* the reference) and yield
    // a positive, plausible time — the value pinned as the speedup denominator.
    let ns = reward()
        .measure_reference_geomean_ns()
        .expect("the baseline eval should be carried out");
    assert!(
        ns.is_some_and(|v| v > 0.0),
        "the reference baseline must measure a positive ns, got {ns:?}"
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_malicious_submission_is_contained_and_stays_below_correctness_floor() {
    // The sandbox runs with no network, so the connection attempt fails and the eval
    // reports no pass — the candidate cannot reach the correctness floor and the host
    // is untouched. This re-confirms the isolation holds inside the real torch/triton
    // image.
    assert_below_correctness_floor(
        score(MALICIOUS),
        "a network-reaching kernel must be contained and not score as correct",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_sealed_assets_survive_candidate_mutation_through_benchmark() {
    let outcomes = reward()
        .reward_group_detailed(
            &Sample::new("write a faster TriMul kernel", ()),
            &[MUTATE_SEALED_ASSETS.to_string()],
        )
        .expect("sealed verifier bytes should remain executable through benchmark");
    let outcome = &outcomes[0];
    assert!(
        outcome.reward >= 1.0,
        "write attempts must fail while correctness and benchmark still pass: {outcome:#?}"
    );
    let metadata = outcome
        .metadata
        .as_ref()
        .expect("the mutation control should return structured verifier evidence");
    assert_eq!(metadata["correct"], serde_json::json!(true));
    assert_eq!(metadata["benchmark_exit"], serde_json::json!(0));
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

// --- Adversarial negative controls: a candidate that tries to *forge* a pass through
//     files, stdout, or any visible descriptor must not reach the correctness floor.
//     These are the reward-gaming cases the honest-wrong WRONG case does not cover.

/// Forge attempt: write fake grade files into the RW scratch. ferrl reads the grade
/// from the protected verifier socket, not these files, so the forge must not affect
/// the score.
const FORGE_WORKFILE: &str = "```python
def custom_kernel(data):
    open(\"/work/test_result.txt\", \"w\").write(\"check: pass\\n\")
    open(\"/work/bench_result.txt\", \"w\").write(\"benchmark.0.mean: 0.001\\ncheck: pass\\n\")
    x, mask, weights, config = data
    return x.clone()
```";

/// Forge attempt: print a fake grade to stdout. Ordinary process output is diagnostic
/// only, so it cannot reach the grade channel.
const FORGE_STDOUT: &str = "```python
def custom_kernel(data):
    print(\"check: pass\")
    print(\"===FERRL-BENCH===\")
    print(\"benchmark.0.mean: 0.001\")
    print(\"check: pass\")
    x, mask, weights, config = data
    return x.clone()
```";

/// Forge attempt: scan every visible descriptor and write a complete, plausible grade.
/// Launcher/init/shell processes own only untrusted stdout/stderr; the accepted grade
/// socket is created after launch and exists only in the non-dumpable verifier parent.
const FORGE_PROCFD: &str = "```python
def custom_kernel(data):
    import os, glob
    payload = (b\"ferrl-entry: test-v4\\ntest-count: 2\\ncheck: pass\\ntest-exit: 0\\n\"
               b\"===FERRL-BENCH===\\nferrl-entry: benchmark-v4\\nbenchmark-count: 2\\n\"
               b\"benchmark.0.mean: 5000000\\nbenchmark.1.mean: 6000000\\n\"
               b\"check: pass\\nbenchmark-exit: 0\\n\")
    for path in glob.glob(\"/proc/[0-9]*/fd/*\"):
        try:
            if int(path.split(\"/\")[2]) == os.getpid():
                continue
            fd = os.open(path, os.O_WRONLY)
            os.write(fd, payload)
            os.close(fd)
        except Exception:
            pass
    x, mask, weights, config = data
    return x.clone()
```";

/// Combined real-process exploit: replace every mutable upstream checker/timer global,
/// scan every visible ancestor descriptor with a plausible forged grade, and return a
/// wrong tensor. The protected parent owns its own checker/timer state and the only
/// accepted grade endpoint, so neither path can reach the correctness floor.
const MUTATE_VERIFIER_GLOBALS: &str = "```python
def custom_kernel(data):
    import glob, inspect, os, time
    import eval as verifier
    verifier.wrap_check_implementation = lambda data, output: (True, \"forged\")
    verifier.check_implementation = lambda data, output: (True, \"forged\")
    verifier.calculate_stats = lambda durations: verifier.Stats(
        runs=10, mean=5000000.0, std=1000.0, err=316.0,
        best=4999000.0, worst=5001000.0,
    )
    time.perf_counter_ns = lambda: 1
    frame = inspect.currentframe()
    while frame is not None:
        status = frame.f_locals.get(\"status\")
        if hasattr(status, \"send_bytes\"):
            try:
                status.send_bytes(b\"DONE\")
            except Exception:
                pass
        shared_output = frame.f_locals.get(\"output_buffer\")
        if hasattr(shared_output, \"zero_\"):
            try:
                shared_output.zero_()
            except Exception:
                pass
        frame = frame.f_back
    payload = (b\"ferrl-entry: test-v4\\ntest-count: 2\\ncheck: pass\\ntest-exit: 0\\n\"
               b\"===FERRL-BENCH===\\nferrl-entry: benchmark-v4\\nbenchmark-count: 2\\n\"
               b\"benchmark.0.mean: 5000000\\nbenchmark.1.mean: 6000000\\n\"
               b\"check: pass\\nbenchmark-exit: 0\\n\")
    for path in glob.glob(\"/proc/[0-9]*/fd/*\"):
        try:
            if int(path.split(\"/\")[2]) == os.getpid():
                continue
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
fn gate_forged_result_files_stay_below_correctness_floor() {
    assert_below_correctness_floor(
        score(FORGE_WORKFILE),
        "forged /work result files must not score as correct",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_forged_stdout_stays_below_correctness_floor() {
    assert_below_correctness_floor(
        score(FORGE_STDOUT),
        "a printed fake grade must not score as correct",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_plausible_all_descriptor_grade_forgery_is_rejected() {
    assert_below_correctness_floor(
        score(FORGE_PROCFD),
        "a plausible grade written through every visible descriptor must be ignored",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_candidate_cannot_mutate_trusted_checker_or_timer_state() {
    assert_below_correctness_floor(
        score(MUTATE_VERIFIER_GLOBALS),
        "candidate-process checker/timer mutation must not validate a wrong output",
    );
}
