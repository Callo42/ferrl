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
//! Gated behind the off-by-default `gate` feature. CI compiles this target and runs
//! its source-contract test; real isolation evidence remains an ignored GPU test
//! suite against a deployed dedicated-UID executor:
//!
//! ```text
//! FERRL_TRIMUL_IMAGE=/path/to/trimul-eval.sif \
//! FERRL_TRIMUL_EVAL_DIR=/path/to/pinned/trimul \
//! FERRL_VERIFIER_EXECUTOR_SOCKET=/run/ferrl/verifier-executor.sock \
//!   cargo test --features gate --test trimul_gate -- --ignored --test-threads=1
//! ```

#![cfg(feature = "gate")]

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use ferrl::trimul::TrimulVerifierAssets;
use ferrl::{Distribution, RewardFn, RewardOutcome, Sample, TrimulCase, TrimulReward};

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
        .with_verifier_executor_socket(env_path("FERRL_VERIFIER_EXECUTOR_SOCKET"))
}

fn evaluate(completion: &str) -> RewardOutcome {
    reward()
        .reward_group_detailed(
            &Sample::new("write a faster TriMul kernel", ()),
            &[completion.to_string()],
        )
        .expect("the eval should be carried out")
        .into_iter()
        .next()
        .expect("one completion produces one detailed outcome")
}

fn score(completion: &str) -> f32 {
    evaluate(completion).reward
}

fn assert_below_correctness_floor(value: f32, reason: &str) {
    assert!(
        (0.0..1.0).contains(&value),
        "{reason}; expected shaped failure reward in [0, 1), got {value}"
    );
}

#[allow(clippy::cognitive_complexity)] // one evidence helper asserts every non-vacuity field
fn assert_attack_reached_wrong_output(completion: &str, sentinel: &str, reason: &str) {
    let outcome = evaluate(completion);
    assert!(
        (0.0..1.0).contains(&outcome.reward),
        "{reason}; expected shaped failure reward in [0, 1), got {outcome:#?}"
    );
    let metadata = outcome
        .metadata
        .as_ref()
        .expect("an executed attack must return detailed verifier metadata");
    assert_eq!(metadata["submission_extracted"], serde_json::json!(true));
    assert_eq!(metadata["correct"], serde_json::json!(false));
    assert_eq!(metadata["test_case_count"], serde_json::json!(2));
    assert_eq!(metadata["test_pass_count"], serde_json::json!(0));
    assert_eq!(
        metadata["timing_metric"],
        serde_json::json!(ferrl::trimul::TRIMUL_TIMING_METRIC)
    );
    let sentinels = metadata["candidate_attempt_sentinels"]
        .as_array()
        .expect("attack sentinel metadata is an array");
    assert_eq!(
        sentinels.len(),
        2,
        "every test case must execute the attack"
    );
    assert!(
        sentinels
            .iter()
            .all(|value| value.as_str() == Some(sentinel)),
        "after-all-attempts sentinel mismatch: {sentinels:?}"
    );
}

fn assert_import_attack_is_candidate_failure(
    completion: &str,
    sentinel: &str,
    rejection_reason: &str,
    reason: &str,
) {
    let outcome = evaluate(completion);
    assert!(
        (0.0..1.0).contains(&outcome.reward),
        "{reason}; expected a low candidate reward, got {outcome:#?}"
    );
    let metadata = outcome
        .metadata
        .as_ref()
        .expect("an import attack must return candidate metadata, not RewardError");
    assert_eq!(metadata["submission_extracted"], serde_json::json!(true));
    assert_eq!(metadata["correct"], serde_json::json!(false));
    assert_eq!(metadata["test_exit"], serde_json::json!(112));
    assert_eq!(
        metadata["candidate_rejection_reason"],
        serde_json::json!(rejection_reason)
    );
    assert_eq!(
        metadata["candidate_attempt_sentinels"],
        serde_json::json!([sentinel])
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

/// Correct but deliberately slower than [`CORRECT`]. The protected parent clock
/// must preserve the added latency even if the candidate mutates its own Python
/// timing module.
const SLOW_CORRECT: &str = "```python
def custom_kernel(data):
    import time
    time.sleep(0.025)
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
    import os, socket
    try:
        socket.create_connection((\"1.1.1.1\", 53), timeout=3).close()
    except OSError:
        pass
    else:
        raise RuntimeError(\"network unexpectedly reachable\")
    with open(\"/work/cache/ferrl-attack-sentinel\", \"w\") as handle:
        handle.write(\"network-denied-wrong-output-v1\")
    x, mask, weights, config = data
    return x.clone()
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
    with open(\"/work/cache/ferrl-attack-sentinel\", \"w\") as handle:
        handle.write(\"sealed-path-writes-denied-v1\")
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
fn gate_service_latency_orders_fast_and_slow_correct_candidates() {
    let fast = evaluate(CORRECT);
    let slow = evaluate(SLOW_CORRECT);
    for outcome in [&fast, &slow] {
        let metadata = outcome
            .metadata
            .as_ref()
            .expect("protected timing metadata");
        assert_eq!(metadata["correct"], serde_json::json!(true));
        assert_eq!(metadata["benchmark_exit"], serde_json::json!(0));
        assert_eq!(
            metadata["timing_metric"],
            serde_json::json!(ferrl::trimul::TRIMUL_TIMING_METRIC)
        );
    }
    let fast_ns = fast.metadata.as_ref().unwrap()["geomean_ns"]
        .as_f64()
        .expect("fast service latency");
    let slow_ns = slow.metadata.as_ref().unwrap()["geomean_ns"]
        .as_f64()
        .expect("slow service latency");
    assert!(
        slow_ns > fast_ns + 10_000_000.0,
        "protected parent timing did not preserve injected latency: fast={fast_ns} slow={slow_ns}"
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_malicious_submission_is_contained_and_stays_below_correctness_floor() {
    // The sandbox runs with no network, so the connection attempt fails and the eval
    // reports no pass — the candidate cannot reach the correctness floor and the host
    // is untouched. This re-confirms the isolation holds inside the real torch/triton
    // image.
    assert_attack_reached_wrong_output(
        MALICIOUS,
        "network-denied-wrong-output-v1",
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
    let sentinels = metadata["candidate_attempt_sentinels"]
        .as_array()
        .expect("mutation attempt metadata is an array");
    assert_eq!(sentinels.len(), 4);
    assert!(sentinels
        .iter()
        .all(|value| value.as_str() == Some("sealed-path-writes-denied-v1")));
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
    open(\"/work/cache/ferrl-attack-sentinel\", \"w\").write(\"workfile-forge-wrong-output-v1\")
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
    open(\"/work/cache/ferrl-attack-sentinel\", \"w\").write(\"stdout-forge-wrong-output-v1\")
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
    open(\"/work/cache/ferrl-attack-sentinel\", \"w\").write(\"procfd-forge-wrong-output-v1\")
    x, mask, weights, config = data
    return x.clone()
```";

/// Mutate child-process checker and timer globals, then reach a wrong output.
const MUTATE_CHECKER_TIMER: &str = "```python
def custom_kernel(data):
    import time
    import eval as verifier
    verifier.wrap_check_implementation = lambda data, output: (True, \"forged\")
    verifier.check_implementation = lambda data, output: (True, \"forged\")
    verifier.calculate_stats = lambda durations: verifier.Stats(
        runs=10, mean=5000000.0, std=1000.0, err=316.0,
        best=4999000.0, worst=5001000.0,
    )
    time.perf_counter_ns = lambda: 1
    open(\"/work/cache/ferrl-attack-sentinel\", \"w\").write(\"checker-timer-mutation-wrong-output-v1\")
    x, mask, weights, config = data
    return x.clone()
```";

/// Corrupt the real payload result channel during module import, after the sealed
/// driver has authenticated candidate entry but before `custom_kernel` exists.
const CORRUPT_IMPORT_PROTOCOL: &str = "```python
import inspect
frame = inspect.currentframe()
used_results_channel = False
while frame is not None:
    results = frame.f_locals.get(\"results\")
    if hasattr(results, \"send_bytes\"):
        results.send_bytes(b\"FERRL-PAYLOAD-v1\\0IMPORT_REJECTED\\0\")
        used_results_channel = True
        break
    frame = frame.f_back
if not used_results_channel:
    raise RuntimeError(\"payload results channel was not found\")
open(\"/work/cache/ferrl-attack-sentinel\", \"w\").write(\"import-protocol-corruption-rejected-v1\")

def custom_kernel(data):
    x, mask, weights, config = data
    return x.clone()
```";

/// Kill the trusted controller during candidate import. Once `ENTRY` was recorded,
/// loss of this candidate-reachable process must remain a low candidate outcome.
const KILL_IMPORT_CONTROLLER: &str = "```python
import os, signal
sentinel = os.open(\"/work/cache/ferrl-attack-sentinel\", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(sentinel, b\"import-controller-loss-after-ack-v2\")
os.fsync(sentinel)
os.close(sentinel)
os.kill(os.getppid(), signal.SIGKILL)
raise RuntimeError(\"controller survived SIGKILL\")

def custom_kernel(data):
    x, mask, weights, config = data
    return x.clone()
```";

/// Prove the payload has neither ptrace/admin capability nor access to sensitive
/// verifier-parent procfs endpoints after every attempted access.
const PROBE_PARENT_AND_CAPABILITIES: &str = "```python
def custom_kernel(data):
    import ctypes, os
    capabilities = {}
    no_new_privs = None
    with open(\"/proc/self/status\", encoding=\"ascii\") as handle:
        for line in handle:
            if line.startswith((\"CapEff:\", \"CapPrm:\", \"CapAmb:\", \"CapBnd:\")):
                key, value = line.split(\":\", 1)
                capabilities[key] = int(value.strip(), 16)
            elif line.startswith(\"NoNewPrivs:\"):
                no_new_privs = int(line.split(\":\", 1)[1].strip())
    if set(capabilities) != {\"CapEff\", \"CapPrm\", \"CapAmb\", \"CapBnd\"}:
        raise RuntimeError(\"candidate capability evidence was incomplete\")
    if no_new_privs != 1:
        raise RuntimeError(\"candidate did not retain NoNewPrivs\")
    forbidden = (1 << 19) | (1 << 21)
    if any(value & forbidden for value in capabilities.values()):
        raise RuntimeError(\"candidate retained CAP_SYS_PTRACE or CAP_SYS_ADMIN\")
    parent = os.getppid()
    for suffix in (\"mem\", \"fd/0\", \"ns/user\", \"ns/mnt\"):
        try:
            descriptor = os.open(f\"/proc/{parent}/{suffix}\", os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            continue
        os.close(descriptor)
        raise RuntimeError(f\"candidate opened verifier parent endpoint {suffix}\")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.ptrace.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    libc.ptrace.restype = ctypes.c_long
    if libc.ptrace(16, parent, None, None) == 0:
        libc.ptrace(17, parent, None, None)
        raise RuntimeError(\"candidate attached to verifier parent\")
    open(\"/work/cache/ferrl-attack-sentinel\", \"w\").write(\"caps-nnp-parent-access-denied-wrong-output-v2\")
    x, mask, weights, config = data
    return x.clone()
```";

/// Use the raw CUDA driver to discover and copy the complete allocation that
/// contains the child input. The protected parent's adjacent device canary must
/// be absent because no parent allocation is exported through CUDA IPC.
const PROBE_RAW_DEVICE_BOUNDARY: &str = "```python
def custom_kernel(data):
    import ctypes
    x, mask, weights, config = data
    driver = ctypes.CDLL(\"libcuda.so.1\")
    driver.cuMemGetAddressRange_v2.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_size_t), ctypes.c_ulonglong,
    ]
    driver.cuMemGetAddressRange_v2.restype = ctypes.c_int
    driver.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]
    driver.cuMemcpyDtoH_v2.restype = ctypes.c_int
    base = ctypes.c_ulonglong()
    size = ctypes.c_size_t()
    if driver.cuMemGetAddressRange_v2(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_ulonglong(x.data_ptr())
    ) != 0:
        raise RuntimeError(\"cuMemGetAddressRange failed\")
    if size.value == 0 or size.value > 512 * 1024 * 1024:
        raise RuntimeError(\"unexpected child allocation boundary\")
    copied = ctypes.create_string_buffer(size.value)
    if driver.cuMemcpyDtoH_v2(copied, base.value, size.value) != 0:
        raise RuntimeError(\"raw child allocation copy failed\")
    if b\"ferrl-parent-private-cuda-v1-7f4c3a19\" in copied.raw:
        raise RuntimeError(\"parent-private CUDA canary crossed the process boundary\")
    open(\"/work/cache/ferrl-attack-sentinel\", \"w\").write(\"raw-device-boundary-absent-wrong-output-v1\")
    return x.clone()
```";

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_forged_result_files_stay_below_correctness_floor() {
    assert_attack_reached_wrong_output(
        FORGE_WORKFILE,
        "workfile-forge-wrong-output-v1",
        "forged /work result files must not score as correct",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_forged_stdout_stays_below_correctness_floor() {
    assert_attack_reached_wrong_output(
        FORGE_STDOUT,
        "stdout-forge-wrong-output-v1",
        "a printed fake grade must not score as correct",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_plausible_all_descriptor_grade_forgery_is_rejected() {
    assert_attack_reached_wrong_output(
        FORGE_PROCFD,
        "procfd-forge-wrong-output-v1",
        "a plausible grade written through every visible descriptor must be ignored",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_candidate_cannot_mutate_trusted_checker_or_timer_state() {
    assert_attack_reached_wrong_output(
        MUTATE_CHECKER_TIMER,
        "checker-timer-mutation-wrong-output-v1",
        "candidate-process checker/timer mutation must not validate a wrong output",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_import_protocol_corruption_is_a_low_candidate_outcome() {
    assert_import_attack_is_candidate_failure(
        CORRUPT_IMPORT_PROTOCOL,
        "import-protocol-corruption-rejected-v1",
        "protocol-v1",
        "candidate import-time result-channel corruption must not become infrastructure failure",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_import_controller_loss_is_a_low_candidate_outcome() {
    assert_import_attack_is_candidate_failure(
        KILL_IMPORT_CONTROLLER,
        "import-controller-loss-after-ack-v2",
        "controller-loss-v1",
        "candidate-induced controller loss must not become infrastructure failure",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_candidate_has_no_ptrace_admin_caps_or_parent_access() {
    assert_attack_reached_wrong_output(
        PROBE_PARENT_AND_CAPABILITIES,
        "caps-nnp-parent-access-denied-wrong-output-v2",
        "candidate capabilities and verifier-parent access must be absent",
    );
}

#[test]
#[ignore = "needs an sm_80 GPU + the eval image/bundle; run with --ignored"]
fn gate_raw_cuda_allocation_boundary_contains_no_parent_private_memory() {
    assert_attack_reached_wrong_output(
        PROBE_RAW_DEVICE_BOUNDARY,
        "raw-device-boundary-absent-wrong-output-v1",
        "raw CUDA allocation discovery must not expose protected-parent memory",
    );
}

#[test]
#[ignore = "needs the deployed dedicated-UID executor; run with --ignored"]
fn gate_training_uid_cannot_reenter_executor_namespaces_or_memory() {
    use std::os::unix::fs::MetadataExt as _;
    use std::os::unix::net::UnixStream;

    let socket = env_path("FERRL_VERIFIER_EXECUTOR_SOCKET");
    let stream = UnixStream::connect(&socket).expect("connect to protected verifier executor");
    let peer = rustix::net::sockopt::socket_peercred(&stream)
        .expect("authenticate protected verifier executor peer");
    assert_ne!(
        peer.uid,
        rustix::process::geteuid(),
        "executor must run under a dedicated UID"
    );
    let socket_owner = std::fs::symlink_metadata(&socket)
        .expect("inspect executor socket")
        .uid();
    assert_eq!(socket_owner, peer.uid.as_raw());

    let pid = peer.pid.as_raw_nonzero().get();
    for suffix in ["mem", "fd/0"] {
        assert!(
            std::fs::File::open(format!("/proc/{pid}/{suffix}")).is_err(),
            "training UID unexpectedly opened executor {suffix}"
        );
    }
    let user_namespace = format!("--user=/proc/{pid}/ns/user");
    let mount_namespace = format!("--mount=/proc/{pid}/ns/mnt");
    let status = std::process::Command::new("nsenter")
        .args([
            user_namespace,
            mount_namespace,
            "--".to_string(),
            "true".to_string(),
        ])
        .status()
        .expect("execute the namespace re-entry attempt");
    assert!(
        !status.success(),
        "training UID unexpectedly entered executor user/mount namespaces"
    );
}

#[test]
fn gate_attack_sources_record_after_attempts_before_the_wrong_output() {
    for (source, final_attempt, sentinel) in [
        (
            MALICIOUS,
            "network unexpectedly reachable",
            "network-denied-wrong-output-v1",
        ),
        (
            FORGE_WORKFILE,
            "benchmark.0.mean: 0.001",
            "workfile-forge-wrong-output-v1",
        ),
        (
            FORGE_STDOUT,
            "benchmark.0.mean: 0.001",
            "stdout-forge-wrong-output-v1",
        ),
        (
            FORGE_PROCFD,
            "except Exception",
            "procfd-forge-wrong-output-v1",
        ),
        (
            MUTATE_CHECKER_TIMER,
            "time.perf_counter_ns = lambda: 1",
            "checker-timer-mutation-wrong-output-v1",
        ),
        (
            CORRUPT_IMPORT_PROTOCOL,
            "if not used_results_channel",
            "import-protocol-corruption-rejected-v1",
        ),
        (
            PROBE_PARENT_AND_CAPABILITIES,
            "libc.ptrace(16",
            "caps-nnp-parent-access-denied-wrong-output-v2",
        ),
        (
            PROBE_RAW_DEVICE_BOUNDARY,
            "parent-private CUDA canary crossed",
            "raw-device-boundary-absent-wrong-output-v1",
        ),
    ] {
        let final_attempt_position = source
            .rfind(final_attempt)
            .unwrap_or_else(|| panic!("attack {sentinel} is missing its final attempt"));
        let sentinel_position = source
            .find(sentinel)
            .unwrap_or_else(|| panic!("attack source is missing sentinel {sentinel}"));
        let wrong_output_position = source
            .rfind("return x.clone()")
            .unwrap_or_else(|| panic!("attack {sentinel} is missing its wrong-output path"));
        assert!(
            final_attempt_position < sentinel_position && sentinel_position < wrong_output_position,
            "attack {sentinel} does not record after all attempts and before wrong output"
        );
    }

    let sentinel = KILL_IMPORT_CONTROLLER
        .find("import-controller-loss-after-ack-v2")
        .expect("controller-loss control records acknowledged entry before the kill");
    let durable = KILL_IMPORT_CONTROLLER
        .find("os.fsync(sentinel)")
        .expect("controller-loss sentinel is durable before the kill");
    let kill = KILL_IMPORT_CONTROLLER
        .find("os.kill(os.getppid(), signal.SIGKILL)")
        .expect("controller-loss control sends SIGKILL");
    let survived = KILL_IMPORT_CONTROLLER
        .find("controller survived SIGKILL")
        .expect("a failed kill cannot fall through to ordinary wrong output");
    assert!(sentinel < durable && durable < kill && kill < survived);
}
