//! Apptainer sandbox malicious-probe battery — the *enforcement* proof.
//!
//! [`ferrl::sandbox`]'s unit tests prove the `apptainer exec` argv **encodes** the
//! isolation policy. These tests prove Apptainer **enforces** it: a deliberately
//! hostile payload cannot read host credentials, reach the network, fork-storm,
//! exhaust memory, write an unbounded file or unbounded scratch tree, and a
//! runaway is killed on its wall-clock budget. This is the spec's "isolation from
//! day one" invariant, shown against a real adversary rather than asserted.
//!
//! Gated behind the off-by-default `gate` feature, so — exactly like the GPU tests
//! — CI never compiles it (no run, no coverage impact). The probes are *bounded*
//! (no exponential fork bomb; memory/file attempts are size-capped), but they are
//! still hostile: run them only on a node with an **isolated allocation**.
//!
//! ```text
//! FERRL_SANDBOX_IMAGE=/path/to/image.sif \
//!   cargo test --features gate --test sandbox_gate -- --ignored --test-threads=1
//! ```

#![cfg(feature = "gate")]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use ferrl::{ApptainerSandbox, Bind, ResourceLimits, RunOutcome, RunSpec, RunStatus, Sandbox};

/// The image under test — required when the gate actually runs (`--ignored`).
fn image() -> PathBuf {
    std::env::var_os("FERRL_SANDBOX_IMAGE")
        .map(PathBuf::from)
        .expect("set FERRL_SANDBOX_IMAGE to an apptainer image to run the sandbox gate")
}

/// A fresh host scratch directory unique to this process + `tag`.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferrl-gate-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Best-effort size of a scratch tree after a hostile probe.
fn dir_size(path: &std::path::Path) -> u64 {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return 0;
    };
    if !meta.is_dir() {
        return meta.len();
    }
    let mut total = meta.len();
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            total = total.saturating_add(dir_size(&entry.path()));
        }
    }
    total
}

/// Run a spec to completion; the run itself must be *carried out* (a non-zero exit
/// or a kill is a normal [`RunOutcome`], not an error).
fn run(spec: &RunSpec) -> RunOutcome {
    ApptainerSandbox::default()
        .run(spec)
        .expect("the sandbox run should be carried out")
}

/// A short wall budget for the hostile probes (an isolated node, fast feedback).
fn short_limits() -> ResourceLimits {
    ResourceLimits {
        wall: Duration::from_secs(10),
        ..ResourceLimits::default()
    }
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_benign_payload_runs() {
    let spec = RunSpec::new(image(), vec!["echo".into(), "hello-sandbox".into()])
        .with_limits(short_limits());
    let out = run(&spec);
    assert_eq!(out.status, RunStatus::Exited(0), "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("hello-sandbox"),
        "stdout was {:?}",
        out.stdout
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_host_filesystem_secret_is_unreachable() {
    // Plant a secret on the host and do NOT bind it. Under --containall/--no-home
    // the host filesystem (and /tmp, a private tmpfs in-container) is invisible, so
    // the read must fail.
    let dir = scratch("secret");
    let secret = dir.join("secret.txt");
    fs::write(&secret, "TOPSECRET-DO-NOT-LEAK").unwrap();
    let probe = format!(
        "if cat {} 2>/dev/null; then echo LEAKED; else echo BLOCKED; fi",
        secret.display()
    );
    let spec =
        RunSpec::new(image(), vec!["bash".into(), "-c".into(), probe]).with_limits(short_limits());
    let out = run(&spec);
    let _ = fs::remove_dir_all(&dir);
    assert!(out.stdout.contains("BLOCKED"), "stdout: {:?}", out.stdout);
    assert!(!out.stdout.contains("LEAKED"), "stdout: {:?}", out.stdout);
    assert!(
        !out.stdout.contains("TOPSECRET"),
        "the host secret leaked into the sandbox: {:?}",
        out.stdout
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_network_is_denied() {
    // Attempt an outbound TCP connect. With NetworkPolicy::None (the default) the
    // sandbox has no interfaces, so the connect must fail. If this surfaces NET_OK,
    // unprivileged `--net --network none` is not enforcing here and the isolation
    // story must be hardened before any model output runs.
    let probe = "if timeout 4 bash -c 'exec 3<>/dev/tcp/1.1.1.1/53' 2>/dev/null; \
                 then echo NET_OK; else echo NET_BLOCKED; fi";
    let spec = RunSpec::new(image(), vec!["bash".into(), "-c".into(), probe.into()])
        .with_limits(short_limits());
    let out = run(&spec);
    assert!(
        out.stdout.contains("NET_BLOCKED"),
        "network was reachable from the sandbox: stdout={:?} stderr={:?}",
        out.stdout,
        out.stderr
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_infinite_loop_times_out() {
    let spec = RunSpec::new(
        image(),
        vec!["bash".into(), "-c".into(), "while true; do :; done".into()],
    )
    .with_limits(ResourceLimits {
        wall: Duration::from_secs(3),
        ..ResourceLimits::default()
    });
    let out = run(&spec);
    assert_eq!(out.status, RunStatus::TimedOut, "stderr: {}", out.stderr);
    assert!(out.wall < Duration::from_secs(20), "took {:?}", out.wall);
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_memory_cap_kills_a_bounded_hog() {
    // A *bounded* ~300 MiB allocation against a 256 MiB address-space cap: it must
    // not exit cleanly. Bounded so that even if the cap silently failed, only
    // ~300 MiB is touched (no node-wide memory storm).
    let spec = RunSpec::new(
        image(),
        vec![
            "bash".into(),
            "-c".into(),
            "x=$(head -c 300000000 /dev/zero | tr '\\0' 'A'); echo LEN=${#x}".into(),
        ],
    )
    .with_limits(ResourceLimits {
        wall: Duration::from_secs(10),
        address_space: Some(256 << 20),
        ..ResourceLimits::default()
    });
    let out = run(&spec);
    assert!(
        !out.status.is_success(),
        "the bounded memory hog should have been capped, got {:?} (stdout {:?})",
        out.status,
        out.stdout
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_file_size_cap_bounds_a_write() {
    // Try to write 10 MiB against a 1 MiB file-size cap, into a bound scratch. The
    // resulting file must stay near the cap, never the full attempt.
    let dir = scratch("fsize");
    let spec = RunSpec::new(
        image(),
        vec![
            "bash".into(),
            "-c".into(),
            "dd if=/dev/zero of=/work/big bs=1024 count=10240 2>/dev/null; true".into(),
        ],
    )
    .with_binds(vec![Bind::rw(&dir, "/work")])
    .with_workdir("/work")
    .with_limits(ResourceLimits {
        wall: Duration::from_secs(10),
        max_file: Some(1 << 20),
        ..ResourceLimits::default()
    });
    let _ = run(&spec);
    let written = fs::metadata(dir.join("big")).map_or(0, |m| m.len());
    let _ = fs::remove_dir_all(&dir);
    assert!(
        written <= 2 << 20,
        "file-size cap did not bound the write: {written} bytes"
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_scratch_total_cap_bounds_many_files() {
    // The per-file `ulimit -f` is not enough: a payload can write many small files.
    // Prove the host supervisor watches the total `/work` tree and kills the run.
    let dir = scratch("scratch-total");
    let spec = RunSpec::new(
        image(),
        vec![
            "bash".into(),
            "-c".into(),
            "i=0; while true; do dd if=/dev/zero of=/work/$i bs=1024 count=64 2>/dev/null; i=$((i+1)); sleep 0.05; done".into(),
        ],
    )
    .with_binds(vec![Bind::rw(&dir, "/work").with_total_limit(1 << 20)])
    .with_workdir("/work")
    .with_limits(ResourceLimits {
        wall: Duration::from_secs(10),
        max_file: Some(1 << 20),
        ..ResourceLimits::default()
    });
    let out = run(&spec);
    let written = dir_size(&dir);
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(
        out.status,
        RunStatus::ScratchExceeded,
        "scratch total cap did not terminate the run; stdout={:?} stderr={:?}",
        out.stdout,
        out.stderr
    );
    assert!(
        written <= 4 << 20,
        "scratch overflow guard reacted too late: {written} bytes"
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_scratch_total_cap_catches_fast_clean_exit() {
    // A finite payload may fill `/work` and exit before the next poll. The final
    // scratch check must still reject it as an overflow, not a clean success.
    let dir = scratch("scratch-exit");
    let spec = RunSpec::new(
        image(),
        vec![
            "bash".into(),
            "-c".into(),
            "dd if=/dev/zero of=/work/big bs=1024 count=2048 2>/dev/null; exit 0".into(),
        ],
    )
    .with_binds(vec![Bind::rw(&dir, "/work").with_total_limit(1 << 20)])
    .with_workdir("/work")
    .with_limits(ResourceLimits {
        wall: Duration::from_secs(10),
        ..ResourceLimits::default()
    });
    let out = run(&spec);
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(
        out.status,
        RunStatus::ScratchExceeded,
        "scratch overflow on a fast clean exit was accepted; stdout={:?} stderr={:?}",
        out.stdout,
        out.stderr
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_process_cap_is_applied() {
    // `ulimit -u` is the fork-bomb mitigation, but on a shared uid it is kernel-wide
    // (no user-namespace remap), so a cap below the uid's live task count would
    // strangle the container's own startup rather than the bomb — and detonating a
    // real fork bomb on shared infra is unsafe. So prove the *mechanism*: the
    // configured cap is actually applied inside the sandbox. Runaway containment is
    // the host wall-clock supervisor's job (see the timeout probe). 128 is well under
    // any hard nproc limit, and reading the limit needs no fork, so it cannot
    // strangle this probe's startup.
    let cap = 128_u64;
    let spec = RunSpec::new(
        image(),
        vec!["bash".into(), "-c".into(), "ulimit -u".into()],
    )
    .with_limits(ResourceLimits {
        wall: Duration::from_secs(10),
        max_procs: Some(cap),
        ..ResourceLimits::default()
    });
    let out = run(&spec);
    assert_eq!(
        out.stdout.trim(),
        cap.to_string(),
        "process cap not applied inside the sandbox; stderr: {}",
        out.stderr
    );
}

#[test]
#[ignore = "needs apptainer + $FERRL_SANDBOX_IMAGE on an isolated node"]
fn gate_multiprocess_orphan_is_reaped() {
    // A *multi-process* runaway: double-fork (setsid, a new session) a child that
    // touches STARTED immediately, then waits past the wall budget and touches
    // SURVIVED, while the main process busy-loops to trip the wall-clock kill.
    // STARTED present proves the orphan actually ran (so the probe is not vacuous);
    // SURVIVED absent proves it was reaped. If the kill reaped only the launcher PID
    // the orphan would survive (holding a GPU / scratch) and create SURVIVED; a
    // correct whole-tree teardown (the PID namespace dying with its init) prevents
    // it. This is the case the single-process timeout probe does NOT cover.
    let dir = scratch("orphan");
    let spec = RunSpec::new(
        image(),
        vec![
            "bash".into(),
            "-c".into(),
            "setsid bash -c 'touch /work/STARTED; sleep 8; touch /work/SURVIVED' & while true; do :; done".into(),
        ],
    )
    .with_binds(vec![Bind::rw(&dir, "/work")])
    .with_workdir("/work")
    .with_limits(ResourceLimits {
        wall: Duration::from_secs(3),
        ..ResourceLimits::default()
    });
    let out = run(&spec);
    assert_eq!(out.status, RunStatus::TimedOut, "stderr: {}", out.stderr);
    // Wait past the orphan's delayed touch (8 s after its start; the run was killed
    // at ~3 s), so a survivor would have created SURVIVED by now.
    std::thread::sleep(Duration::from_secs(10));
    let started = dir.join("STARTED").exists();
    let survived = dir.join("SURVIVED").exists();
    let _ = fs::remove_dir_all(&dir);
    assert!(
        started,
        "the orphan never launched (STARTED absent) — the probe would be vacuous"
    );
    assert!(
        !survived,
        "a forked orphan survived the wall-clock kill — the whole-tree teardown is incomplete"
    );
}
