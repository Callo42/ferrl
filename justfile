# ferrl task runner — the single entry point for the local quality bar.
#
# Every recipe mirrors a CI job (see .github/workflows/ci.yml) so that a green
# `just gate` locally is a faithful predictor of green CI. CI itself can call
# these recipes too, which keeps the contract in exactly one place.
#
# ---------------------------------------------------------------------------
# platformax / NFS-HOME note (the cargo SQLite deadlock)
# ---------------------------------------------------------------------------
# On the platformax dev cluster, $HOME lives on NFS. Cargo's global cache uses a
# SQLite-backed lock under $CARGO_HOME (`.global-cache`), and SQLite file locking
# is unreliable on NFS — so a bare `cargo` call can DEADLOCK on "Blocking waiting
# for file lock on package cache" / "database is locked" *before it compiles a
# single line*. The fix is to put CARGO_HOME (and the target dir) on node-local
# storage (/tmp).
#
# We do this WITHOUT penalising ordinary contributors or GitHub runners:
#
#   * By default this justfile changes NOTHING about cargo's environment. It runs
#     the plain `cargo` on your PATH with your normal CARGO_HOME — exactly what a
#     laptop or a GitHub runner wants.
#
#   * On platformax you opt in with ONE switch:  FERRL_LOCAL_CARGO=1
#     That reroutes CARGO_HOME and CARGO_TARGET_DIR to a per-user dir under /tmp
#     (node-local), sidestepping the NFS lock. Export it in your shell rc on the
#     cluster and forget about it:
#         export FERRL_LOCAL_CARGO=1
#     ...or scope it to a single command:
#         FERRL_LOCAL_CARGO=1 just test
#
#   * Every value is independently overridable, so you can point things wherever
#     you like without editing this file:
#         CARGO=cross just check
#         CARGO_HOME=/scratch/$USER/cargo CARGO_TARGET_DIR=/scratch/$USER/tgt just build
#         LOCAL_CARGO_ROOT=/local-ssd/$USER just gate     # base for the /tmp reroute
#
# Precedence for each var: an explicit env value always wins; otherwise, if
# FERRL_LOCAL_CARGO is truthy we derive a /tmp path; otherwise we leave cargo's
# own default in place (empty string = "do not set it").

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Expose recipe arguments as shell positional parameters ($@, $1, ...) with
# proper quoting. Without this, `{{args}}` is *text-substituted* into the recipe
# body, so an argument containing shell metacharacters — e.g. the coverage
# regex `(examples|bin)/` — would be reparsed by bash and break. With positional
# arguments, `_cargo` can forward "$@" verbatim and quoting is preserved.
set positional-arguments

# --- Tunables (all env-overridable) ----------------------------------------

# The cargo binary. Override to e.g. `cross`, `cargo +nightly`, sccache wrappers.
CARGO := env_var_or_default("CARGO", "cargo")

# Opt-in flag for the platformax /tmp reroute. "0" / empty = off (default).
FERRL_LOCAL_CARGO := env_var_or_default("FERRL_LOCAL_CARGO", "0")

# Base dir for the node-local reroute. Defaults to /tmp/ferrl-cargo-$USER.
LOCAL_CARGO_ROOT := env_var_or_default(
    "LOCAL_CARGO_ROOT",
    "/tmp/ferrl-cargo-" + env_var_or_default("USER", "shared")
)

# Effective CARGO_HOME / CARGO_TARGET_DIR.
# Resolution order: explicit env > /tmp reroute (if FERRL_LOCAL_CARGO truthy) > "".
# An empty string means "don't touch the variable" (cargo uses its own default).
CARGO_HOME_EFF := env_var_or_default(
    "CARGO_HOME",
    if FERRL_LOCAL_CARGO == "0" { "" } else { LOCAL_CARGO_ROOT + "/home" }
)
CARGO_TARGET_DIR_EFF := env_var_or_default(
    "CARGO_TARGET_DIR",
    if FERRL_LOCAL_CARGO == "0" { "" } else { LOCAL_CARGO_ROOT + "/target" }
)

# Coverage gate threshold (lines). Kept in one place so CI and local agree.
COV_MIN := env_var_or_default("FERRL_COV_MIN", "90")

# --- Internal: run cargo with the resolved environment ----------------------
# `_cargo` is the wrapper every other recipe funnels through. It exports
# CARGO_HOME / CARGO_TARGET_DIR ONLY when they resolve to a non-empty value, so
# the default contributor/runner path is byte-for-byte a plain `cargo ...`.
# `mkdir -p` is harmless when the dirs already exist and creates the /tmp tree
# on first use on platformax.
[private]
_cargo +args:
    #!/usr/bin/env bash
    set -euo pipefail
    home_dir='{{CARGO_HOME_EFF}}'
    target_dir='{{CARGO_TARGET_DIR_EFF}}'
    if [[ -n "$home_dir" ]]; then
        mkdir -p "$home_dir"
        export CARGO_HOME="$home_dir"
    fi
    if [[ -n "$target_dir" ]]; then
        mkdir -p "$target_dir"
        export CARGO_TARGET_DIR="$target_dir"
    fi
    # `set positional-arguments` makes the recipe args available as "$@" with
    # quoting intact, so metacharacter-bearing args (e.g. the llvm-cov regex)
    # survive. `$@` here is the variadic `args`.
    exec {{CARGO}} "$@"

# Default recipe: list everything.
default:
    @just --list

# --- One-time setup ---------------------------------------------------------

# Install the toolchain components, pre-commit hooks, and cargo-llvm-cov.
# Idempotent: safe to re-run. Honours the pinned toolchain in rust-toolchain.toml.
[doc("One-time setup: toolchain components, pre-commit hooks, cargo-llvm-cov")]
bootstrap:
    @echo ">> rustup: ensure pinned toolchain + components are present"
    rustup show active-toolchain || rustup toolchain install
    rustup component add rustfmt clippy llvm-tools-preview
    @echo ">> cargo-llvm-cov: install if missing"
    @just _cargo llvm-cov --version >/dev/null 2>&1 \
        || just _cargo install cargo-llvm-cov --locked
    @echo ">> pre-commit: install the framework + the commit-msg + commit hooks"
    @command -v pre-commit >/dev/null 2>&1 \
        || { echo "pre-commit not found — install it: pipx install pre-commit (or pip install --user pre-commit)"; exit 1; }
    pre-commit install --install-hooks
    pre-commit install --hook-type commit-msg
    @echo ">> bootstrap complete"

# --- Formatting -------------------------------------------------------------

# Rewrite source to the rustfmt contract (max_width=100, edition 2021).
fmt:
    @just _cargo fmt --all

# --- Linting ----------------------------------------------------------------

# The full static gate that CI's fmt+clippy jobs run: fmt is verified (not
# rewritten) and clippy is run with -D warnings against the curated lint set
# declared in Cargo.toml [workspace.lints].
[doc("Static gate: fmt --check + curated clippy -D warnings (mirrors CI)")]
lint: fmt-check clippy

# Verify formatting without modifying files (mirrors the CI `fmt` job).
fmt-check:
    @just _cargo fmt --all --check

# Curated clippy at -D warnings over all targets (mirrors CI). DEFAULT (CPU)
# features only: the `cuda` / `cudnn` / `flash-attn` features need nvcc and a
# CUDA toolkit, which CPU-only CI and laptops lack — `--all-features` here would
# fail to even build cudarc's build script. GPU linting is done manually on the
# cluster via `just clippy-cuda`.
[doc("Curated clippy at -D warnings, CPU features (mirrors CI)")]
clippy:
    @just _cargo clippy --all-targets -- -D warnings

# Same lint bar, but for the GPU feature set. Cluster-only (needs nvcc).
clippy-cuda:
    @just _cargo clippy --all-targets --features cuda -- -D warnings

# --- Build / type-check -----------------------------------------------------

# Fast type-check of the whole workspace without producing binaries (CPU).
check:
    @just _cargo check --workspace --all-targets

# --- Tests ------------------------------------------------------------------

# Unit + integration tests AND doctests across the workspace (CPU features).
test:
    @just _cargo test --workspace
    @just _cargo test --workspace --doc

# --- Coverage ---------------------------------------------------------------

# Line coverage with the HARD gate (--fail-under-lines, default 90). Note:
# llvm-cov does not instrument doctests, so this measures unit + integration
# tests only — matching the CI `test` job. examples/, bins, and `loader.rs` are
# excluded from the coverage denominator so thin, hard-to-test glue code (CLI
# entry points, demo scripts, the multi-GB checkpoint loader whose happy path
# needs a real asset) never dilutes the library's measured coverage.
[doc("Line coverage with HARD --fail-under-lines gate (default 90)")]
cov:
    @just _cargo llvm-cov --workspace --ignore-filename-regex '(examples|bin)/|loader\.rs' --fail-under-lines {{COV_MIN}}

# Emit an HTML coverage report under target/llvm-cov/html for local inspection.
cov-html:
    @just _cargo llvm-cov --workspace --ignore-filename-regex '(examples|bin)/|loader\.rs' --html
    @echo ">> open target/llvm-cov/html/index.html"

# --- Docs -------------------------------------------------------------------

# Build the API docs with rustdoc warnings (incl. broken intra-doc links) as
# errors, mirroring the CI `docs` job (RUSTDOCFLAGS=-D warnings). CPU features.
[doc("Build rustdoc with warnings as errors (mirrors CI docs job)")]
doc:
    RUSTDOCFLAGS="-D warnings" just _cargo doc --no-deps

# --- The full local bar -----------------------------------------------------

# Run the entire locked quality bar exactly as CI would, in CI order. A green
# `just gate` is the contract for "safe to push".
[doc("Full local quality bar: fmt-check + clippy + check + test + cov + doc")]
gate: fmt-check clippy check test cov doc
    @echo ">> gate: all checks passed"

# --- Oracle regeneration (maintenance) --------------------------------------

# Regenerate the committed GRPO golden fixture from the Python oracle. Run this
# only when the oracle math intentionally changes; the result is committed.
[doc("Regenerate the committed GRPO golden fixture from the Python oracle")]
gen-golden:
    python3 scripts/gen_golden.py > crates/ferrl/tests/fixtures/grpo_golden.json
    @echo ">> regenerated crates/ferrl/tests/fixtures/grpo_golden.json"
