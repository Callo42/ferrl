//! `runreport` — print a one-glance health summary for a run's `metrics.jsonl`.
//!
//! ```text
//! cargo run --example runreport -- <run-dir-or-metrics.jsonl> [--json] [--strict]
//! ```
//!
//! The path may be a run directory (the tool appends `metrics.jsonl`) or the
//! `metrics.jsonl` file itself. `--json` emits the `RunSummary` as JSON instead
//! of the human report; `--strict` exits non-zero (code 2) when any health flag
//! is raised, so a wrapper script can gate on a clean run.
//!
//! **Data-parallel runs** write one `metrics.jsonl` per rank (each under its own
//! run directory). Point this at any rank's directory for that rank's view: the
//! learning signals (reward, kl, `grad_norm`) are bitwise-identical across ranks by
//! the DP lockstep invariant, while `tokens_per_sec` is **per-rank** — the world
//! throughput is `world_size ×` the reported figure.

// A standalone report binary whose interface *is* its stdout/stderr output.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ferrl::{read_metrics, summarize, RunDir, RunSummary};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("runreport: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Parse args, load + summarize the run, emit the report. Returns the process
/// exit code (`2` under `--strict` when any anomaly was flagged), or a message
/// to print to stderr before exiting non-zero.
fn run(args: &[String]) -> Result<ExitCode, String> {
    let json = args.iter().any(|a| a == "--json");
    let strict = args.iter().any(|a| a == "--strict");
    let path_arg = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or("usage: runreport <run-dir-or-metrics.jsonl> [--json] [--strict]")?;
    let metrics_path = resolve_metrics_path(path_arg);
    let history = read_metrics(&metrics_path)
        .map_err(|e| format!("cannot read {}: {e}", metrics_path.display()))?;
    let summary = summarize(&history)
        .ok_or_else(|| format!("{} has no metrics records yet", metrics_path.display()))?;
    emit(&summary, json)?;
    if strict && !summary.anomalies.is_empty() {
        return Ok(ExitCode::from(2));
    }
    Ok(ExitCode::SUCCESS)
}

/// Print the summary as the human report or, with `json`, as pretty JSON.
fn emit(summary: &RunSummary, json: bool) -> Result<(), String> {
    if json {
        let s = serde_json::to_string_pretty(summary).map_err(|e| format!("serialize: {e}"))?;
        println!("{s}");
    } else {
        // `RunSummary`'s Display already terminates each line with a newline.
        print!("{summary}");
    }
    Ok(())
}

/// If `arg` is a directory, append the run's `metrics.jsonl`; otherwise treat it
/// as the metrics file path directly.
fn resolve_metrics_path(arg: &str) -> PathBuf {
    let p = Path::new(arg);
    if p.is_dir() {
        p.join(RunDir::METRICS_FILE)
    } else {
        p.to_path_buf()
    }
}
