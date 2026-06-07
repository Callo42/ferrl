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
masked_mean(grpo)   = mean_seq( sum_t v·m / sum_t m )         # per-seq mean, then mean
masked_mean(drgrpo) = sum_all(v·m) / (num_seq · max_len)      # Dr.GRPO, length-unbiased
```

---

## Training run layout (`runs/`)

Each training run writes to `runs/<run_id>/`:

```
runs/<run_id>/
├── config.json       # the full resolved run config
├── metrics.jsonl     # one JSON object per step:
│                     #   step, reward_mean, reward_std, kl, clip_ratio,
│                     #   completion_len, grad_norm, lr
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
They require a CUDA toolkit (`nvcc`) and are built **manually on the Slurm cluster**,
never in CI:

```sh
just clippy-cuda        # lint the GPU feature set
cargo build --features cuda
```

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.

`SPDX-License-Identifier: MIT OR Apache-2.0`
