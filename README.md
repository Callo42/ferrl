# ferrl

**ferrl** is a [candle](https://github.com/huggingface/candle)-native
[GRPO](https://arxiv.org/abs/2402.03300) reinforcement-learning library, in Rust,
for RL-fine-tuning LLMs. The first target is **Qwen3-0.6B-Base**.

ferrl **owns only the RL layer** — the GRPO loss, the reward trait, manual LoRA, the
rollout loop, the trainer, and the custom-gradient forward. It **delegates all tensor
math, autograd, GPU, and model code** to candle (`candle-core`, `candle-nn`,
`candle-transformers`). We do not reimplement autodiff or kernels; we orchestrate
candle's.

> Status: early. The first commit lands the well-tested pure GRPO math and its
> correctness oracle, plus thin documented stubs for the reward, policy, and LoRA
> layers. Model forward, rollout, and the trainer follow.

---

## Why a separate RL layer

Most of the heavy lifting in RL fine-tuning is ordinary deep-learning compute that
candle already does well. What is *specific* to GRPO — group-relative advantages, the
k3 KL estimate, the clipped surrogate, masked aggregation over variable-length
completions, and making sure every trainable LoRA parameter actually receives a
gradient — is small, subtle, and easy to get silently wrong. ferrl isolates exactly
that surface so it can be tested to a high bar.

---

## Layout

```
ferrl/
├── Cargo.toml                 # workspace: [workspace.lints] = the locked lint bar
├── rust-toolchain.toml        # pinned stable + rustfmt/clippy/llvm-tools-preview
├── rustfmt.toml  clippy.toml  # max_width=100; cognitive-complexity threshold 7
├── cog.toml                   # cocogitto (Conventional Commits + SemVer)
├── justfile                   # task runner (platformax-aware cargo wrapper)
├── scripts/gen_golden.py      # the Python GRPO oracle (regenerates the fixture)
├── .github/workflows/ci.yml   # CPU-only CI gate
└── crates/ferrl/
    ├── src/{lib,grpo,telemetry,reward,policy,lora}.rs
    └── tests/fixtures/grpo_golden.json   # committed oracle output
```

---

## Quality bar

ferrl holds a strict, CI-enforced contract (the Rust analog of a strict
Ruff + mypy + pytest Python setup). All of it runs on cloud CI on GitHub Actions
(CPU-only). GPU runs are manual, on a Slurm cluster — **never in CI**.

| Area        | Standard |
|-------------|----------|
| Formatting  | `rustfmt`, `max_width = 100`, edition 2021. |
| Linting     | `clippy` with `-D warnings`: `clippy::all` + a **curated** subset of pedantic, plus `clippy::print_stdout` / `clippy::print_stderr` (we log via `tracing`, never `println!`) and a cognitive-complexity bound (~7). Lint levels live in `[workspace.lints]` in `Cargo.toml`. The curated set is chosen so a fresh, correct scaffold compiles **clean** under `-D warnings` — we deliberately do **not** turn on full `pedantic`/`nursery` as warnings. |
| Safety/docs | Crate-level `#![forbid(unsafe_code)]`; `deny(missing_docs)`; `deny(rustdoc::broken_intra_doc_links)`. |
| Tests       | `cargo test` + doctests; `proptest` for numeric invariants. |
| Coverage    | `cargo-llvm-cov` with a hard gate `--fail-under-lines 90`. Examples/bin are excluded from coverage so the gate is achievable from commit 1 (which ships the pure GRPO math + tests). |
| Toolchain   | `rust-toolchain.toml` pins stable + `rustfmt`, `clippy`, `llvm-tools-preview`. |
| Commits     | Conventional Commits + SemVer via [`cocogitto`](https://github.com/cocogitto/cocogitto) (`cog.toml`, tag prefix `v`, changelog). |
| Hooks       | [`pre-commit`](https://pre-commit.com): trailing-whitespace, end-of-file-fixer, check-merge-conflict, check-toml, check-yaml, detect-private-key, plus local `cargo fmt` / `cargo clippy` (commit stage) and the `cog` commit-msg hook. Heavy test runs stay in CI. |
| License     | Dual **MIT OR Apache-2.0**. |

### Correctness oracles

Because ferrl no longer owns autodiff, correctness is pinned against references:

1. **GRPO math vs TRL / DeepSeekMath** — a committed golden JSON
   (`crates/ferrl/tests/fixtures/grpo_golden.json`) computed with **NumPy**
   (`std(ddof=1)`, matching TRL `nanstd` / candle `Tensor::var`) by
   `scripts/gen_golden.py`. The Rust test `grpo::tests::matches_golden_fixture`
   loads it and asserts agreement (scaled **and** unscaled advantages). See
   [Oracle](#oracle--golden-fixture).
2. **Model forward vs candle's shipped forward** — *planned*, with the model layer.
3. **Grad-coverage canary** (`lora::tests::grad_coverage_canary_contract`) — guards a
   real footgun: candle optimizers **silently skip** any parameter missing from the
   `GradStore`. The contract is init-dependent: at standard zero-`B` init only `B`
   is guaranteed a non-zero gradient (`A`'s is legitimately ~0); after a non-zero-`B`
   step **both** `A` and `B` must be present and non-zero.
4. **End-to-end finite-difference gradcheck** of the loss w.r.t. the LoRA params —
   *planned*, with the trainer.

---

## Quickstart

Everything routes through a [`just`](https://github.com/casey/just) runner.

```sh
just bootstrap   # install toolchain components + pre-commit hooks + cargo-llvm-cov
just fmt         # rustfmt (rewrite)
just lint        # fmt --check + clippy -D warnings
just check       # cargo check
just test        # cargo test + doctests
just cov         # cargo-llvm-cov, --fail-under-lines 90
just doc         # rustdoc, deny broken intra-doc links
just gate        # the full CI gate locally: fmt + clippy + check + test + cov + doc
```

### Regenerating the GRPO golden fixture

The fixture is committed and CI never regenerates it. Regenerate only when the GRPO
math itself changes. The oracle **requires NumPy** (it computes the group std via
`numpy.std(ddof=1)`, so the fixture stays independent of the Rust formula):

```sh
just gen-golden
# == python3 scripts/gen_golden.py > crates/ferrl/tests/fixtures/grpo_golden.json
```

The script emits stable, indented JSON, so a no-op regeneration produces no diff.

---

## Oracle + golden fixture

`scripts/gen_golden.py` computes, for tiny fixed examples, the GRPO advantages, the
k3 KL estimate, the clipped surrogate, and both masked-mean reductions — mirroring
TRL's `GRPOTrainer` and the DeepSeekMath objective — and writes them to
`crates/ferrl/tests/fixtures/grpo_golden.json`. The numbers are deliberately small and
round so every field is hand-checkable with a calculator.

The fixture schema (consumed verbatim by the Rust test):

```jsonc
{
  "eps_std": 1e-4,                       // must equal grpo::GROUP_STD_EPS
  "groups": [ { "rewards": [..],              // one GRPO group of completions
                "advantages": [..],           //  A_i = (r_i - mean) / (std + eps)
                "advantages_unscaled": [..] }],// A_i = r_i - mean (ScaleRewards::None)
  "k3_kl":  [ { "logp", "logp_ref", "kl" } ],          // exp(d) - d - 1
  "clipped_surrogate": [ { "ratio", "advantage",
                           "clip_eps", "value" } ],     // min(rA, clip(r)·A)
  "masked_mean": { "per_token"[S][T], "mask"[S][T],
                   "grpo", "dr_grpo" }                  // the two reductions
}
```

Formulas (per token, with `std` the *sample* / ddof=1 standard deviation):

```
advantage_i      = (r_i - mean(r)) / (std(r) + eps)          # eps = 1e-4
ratio            = exp(logp - old_logp)
clipped_surrogate= min( ratio * adv, clip(ratio, 1-e, 1+e) * adv )
k3_kl            = exp(d) - d - 1,   d = ref_logp - logp      # >= 0
masked_mean(grpo)   = mean_seq( sum_t v·m / max(sum_t m, 1) )  # all-pad row -> 0 (TRL clamp)
masked_mean(drgrpo) = sum_all(v·m) / (num_seq · max_len)       # Dr.GRPO; requires max_len >= 1
```

---

## Training run layout (`runs/`)

Each training run writes to `runs/<run_id>/`:

```
runs/<run_id>/
├── config.json       # the full resolved run config
├── metrics.jsonl     # one JSON object per step:
│                     #   step, reward_mean, reward_std, frac_reward_zero_std,
│                     #   kl, clip_ratio, completion_len, dropped_rows,
│                     #   grad_norm, lr
├── checkpoints/      # LoRA checkpoints
└── run.log           # human-readable log
```

Logging is structured via `tracing` + `tracing-subscriber`, with spans around rollout
and update. `runs/` is git-ignored. The on-disk layout is created by
[`ferrl::telemetry::RunDir`] and metrics are appended by `ferrl::telemetry::MetricsWriter`.

---

## Building on the platformax cluster

On the platformax dev cluster, `$HOME` lives on NFS. Bare `cargo` **deadlocks on its
SQLite global-cache lock** (`"database is locked"`) *before* it compiles anything,
because the cache lock does not behave over NFS. The fix: point `CARGO_HOME` and
`CARGO_TARGET_DIR` at node-local `/tmp`.

The `justfile` handles this transparently and is **env-overridable**. By default it
changes nothing about cargo's environment, so ordinary contributors and GitHub runners
get standard behavior. On platformax, opt in with a single switch:

```sh
export FERRL_LOCAL_CARGO=1     # reroute CARGO_HOME + CARGO_TARGET_DIR to /tmp
just test
```

Or override the paths explicitly (wins over everything):

```sh
export CARGO_HOME=/tmp/$USER/cargo-home
export CARGO_TARGET_DIR=/tmp/$USER/ferrl-target
just test
```

See the `justfile` header for the exact override variables and defaults.

---

## GPU builds

The `cuda`, `cudnn`, and `flash-attn` cargo features forward candle's GPU backends.
They require a CUDA toolkit (`nvcc`) and a compatible NVIDIA driver. We build them
**manually on a Slurm cluster, never in CI** — but that is our workflow, not a
requirement: any machine with a CUDA toolkit and a new-enough driver can build them.

```sh
just clippy-cuda        # lint the GPU feature set
cargo build --features cuda
```

### CUDA driver compatibility

The CUDA kernels ferrl uses are compiled to **PTX** at *build* time by the CUDA
toolkit (`nvcc`) on the build machine. At *run* time your NVIDIA **driver**
JIT-compiles that PTX for your GPU, and it enforces its own maximum supported PTX ISA
version. As NVIDIA's compatibility guide puts it:

> Applications that compile device code to PTX will not work on older drivers.

So the rule is one-directional:

> **The CUDA toolkit you build with must emit PTX no newer than your runtime driver
> understands.** Older PTX runs fine on newer drivers; newer PTX does not run on older
> drivers.

`CUDA_COMPUTE_CAP` sets the **GPU SM architecture** (e.g. `80` = Ampere), **not** the
PTX ISA version. Only the toolkit (`nvcc`) version sets the ISA — lowering
`CUDA_COMPUTE_CAP` will **not** fix a driver-too-old error.

#### Symptom

If you build with a toolkit newer than your driver supports, the **first** GPU kernel
load (the first model forward) fails at run time — not at build time — with:

```
CUDA_ERROR_UNSUPPORTED_PTX_VERSION   (CUDA driver error 222)
```

ferrl detects this and replaces the cryptic driver error with an actionable message
telling you to rebuild with an older toolkit or upgrade your driver (see "Built-in
preflight" below).

#### Fixing it for your driver

1. Find your driver's CUDA ceiling: run `nvidia-smi`. The top-right **"CUDA Version"**
   field is the maximum CUDA toolkit your driver can run (distinct from the driver
   number on the left, e.g. `550.54.14`).
2. Build ferrl with a CUDA toolkit **at or below** that ceiling (the table below maps
   driver minimums to the maximum toolkit), **or** upgrade your NVIDIA driver to the
   minimum for the toolkit you want.
3. If several CUDA toolkits are installed (e.g. a distro toolkit plus an HPC SDK), the
   kernels are compiled with whichever `nvcc` is **first on `PATH`** — put the toolkit
   you intend to build with first.

A CUDA **major-family** minimum driver (e.g. "CUDA 12.x runs on ≥ 525.60.13") does
**not** cover PTX JIT of a newer ISA. Meet the **per-toolkit** minimum in the table
below, not just the family floor.

#### Compatibility table

Driver minimums are **Linux x86_64** (Windows minimums differ — see the NVIDIA CUDA
Toolkit Release Notes). Match your driver against the first column to find the newest
toolkit you can build with.

| Your driver ≥ (Linux x86_64) | Max CUDA toolkit | PTX ISA |
| ---------------------------- | ---------------- | ------- |
| 520.61.05                    | 11.8             | 7.8     |
| 525.60.13                    | 12.0             | 8.0     |
| 530.30.02                    | 12.1             | 8.1     |
| 535.54.03                    | 12.2             | 8.2     |
| 545.23.06                    | 12.3             | 8.3     |
| 550.54.14                    | 12.4             | 8.4     |
| 555.42.02                    | 12.5             | 8.5     |
| 560.28.03                    | 12.6             | 8.5     |
| 565.57.01                    | 12.7             | 8.6     |
| 570.26                       | 12.8             | 8.7     |
| 575.51.03                    | 12.9             | 8.8     |

(CUDA 12.6 reuses ISA 8.5; CUDA 12.7 — the r565 driver generation, which reports CUDA
12.7 via the driver API — introduced ISA 8.6.)

The driver column is NVIDIA's minimum for the full toolkit. The built-in preflight
instead names the driver that can *JIT* your build's PTX ISA — the floor that error 222
actually keys on — which can be lower (e.g. `555.42.02` for ISA 8.5, even from a CUDA 12.6
build). Both are correct; error 222 only concerns PTX JIT, so follow whichever number the
preflight prints.

#### Built-in preflight

When built with `--features cuda`, ferrl turns the cryptic
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` into an actionable rebuild/upgrade message:

- A **reactive guard** — `ferrl::guard_first_kernel(&device)`, also applied
  automatically on ferrl's first GPU forward — runs one tiny kernel and, on a PTX
  mismatch, reports exactly how to fix it. This is the authoritative check.
- An optional **proactive check** — `ferrl::check_driver_compat(&device)` — compares
  your driver's reported CUDA version against the PTX ISA this binary was built with
  and **warns** early. It never blocks a working setup.

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.

`SPDX-License-Identifier: MIT OR Apache-2.0`
