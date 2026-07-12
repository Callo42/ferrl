# ferrl

**ferrl** is a [candle](https://github.com/huggingface/candle)-native
[GRPO](https://arxiv.org/abs/2402.03300) reinforcement-learning library, in Rust,
for RL-fine-tuning LLMs. The first target is **Qwen3-0.6B-Base**.

ferrl **owns only the RL layer** — the GRPO loss, the reward trait, manual LoRA, the
rollout loop, the trainer, and the custom-gradient forward. It **delegates all tensor
math, autograd, GPU, and model code** to candle (`candle-core`, `candle-nn`,
`candle-transformers`). We do not reimplement autodiff or kernels; we orchestrate
candle's.

> Status: the training stack runs end-to-end on real hardware, single-GPU **and**
> single-node multi-GPU. The single-GPU core — GRPO trainer (gradient accumulation,
> EOS/length masking, momentum-faithful checkpoint/resume), manual LoRA with a
> bf16-base / F32-adapter dtype split, a KV-cached merged-weight rollout, opt-in
> activation checkpointing (layer-boundary rematerialization — candle ships no
> checkpoint primitive), and a held-out eval harness — scales to a **Qwen3.6-27B
> LoRA-GRPO run end-to-end on a single H200** (forward-equivalent to the reference,
> rematerialization fits the activation footprint, training reward rises). Grad-bearing
> forwards exist for **four architecture families** — Qwen3 (dense), dense Llama-3.x, the
> hybrid `qwen3_5` family (GatedDeltaNet + gated GQA, i.e. Qwen3.5/3.6), and dense
> Gemma 4 text (`gemma4`, with `gemma4_unified` accepted as the same dense text
> loader shape) — behind the `GradModel`/`CachedDecoder` trait seam, so one generic
> policy and trainer drive all four unchanged. Gemma 4's committed external oracle
> covers the upstream dense text `gemma4` / `gemma4_text` reference shape; the unified
> alias is covered by config/loader gates.
>
> **Single-node data parallelism is in and verified on multi-GPU hardware**: an
> all-reduce of LoRA gradients over an NCCL `Comm` bridge (`--features nccl` — the
> crate's only `unsafe`, developed CPU-mock-first), DP-coordinated checkpoint resume +
> restart-on-preemption, and run observability (per-step timing, a `summarize` health
> view, the `runreport` tool, and rank/world/step-stamped `tracing` logs); verified
> bit-identical across ranks on multi-A100. **Single-node tensor-parallel execution is
> also available through `ferrl train`** for Qwen3 and dense Gemma 4 policies: model
> projections, rollout/scoring, adapter-gradient reduction, and trainer control flow use
> an NCCL TP communicator. Checkpoint weights are still loaded in full on every rank;
> sharded safetensors loading and combined sharded DP x TP remain future work.

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
├── docs/                      # run contracts and operator-facing project docs
├── rust-toolchain.toml        # pinned stable + rustfmt/clippy/llvm-tools-preview
├── rustfmt.toml  clippy.toml  # max_width=100; cognitive-complexity threshold 7
├── cog.toml                   # cocogitto (Conventional Commits + SemVer)
├── justfile                   # task runner (NFS-home-aware cargo wrapper)
├── scripts/gen_golden.py      # the Python GRPO oracle (regenerates the fixture)
├── .github/workflows/ci.yml   # CPU-only CI gate
└── crates/ferrl/
    ├── src/
    │   ├── grpo.rs                          # GRPO math (advantages, k3 KL, surrogate)
    │   ├── trainer.rs  eval.rs              # training loop + held-out eval harness
    │   ├── lora.rs  optim.rs  sampler.rs  checkpoint.rs
    │   ├── model.rs                         # the GradModel / CachedDecoder trait seam
    │   ├── qwen.rs  llama.rs  qwen35.rs  gemma4.rs
    │   │                                      # the model layer (grad forwards + cached decoders)
    │   ├── blocks.rs  gdn.rs  remat.rs      # shared blocks, GatedDeltaNet math, activation ckpt
    │   ├── moe.rs                           # qwen3.5/3.6 sparse-MoE kernels (router/experts, M3′)
    │   ├── lm_policy.rs                     # Policy over any GradModel (Qwen/Llama/Qwen3_5/Gemma4)
    │   ├── comm.rs  comm/                   # distributed Comm seam (Solo/Local + NCCL bridge)
    │   ├── full_ft.rs                       # opt-in full fine-tune (vs LoRA)
    │   └── {lib,policy,reward,nn,tokenizer,countdown,telemetry,cuda_compat}.rs
    └── tests/fixtures/grpo_golden.json      # committed oracle output
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
| Safety/docs | Crate-level `#![deny(unsafe_code)]` — the default build compiles **zero** `unsafe`; the sole `#[allow(unsafe_code)]` is the optional `--features nccl` NCCL FFI module (not built by default or in CI). `deny(missing_docs)`; `deny(rustdoc::broken_intra_doc_links)`. |
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
2. **Model forward vs model-family references** — per-position logit-equivalence
   gates load our grad-bearing forward and candle's shipped one from the **same**
   weights and compare at **every** position, for both Qwen3 and dense Llama-3.x
   (tiny seeded CPU configs in CI; a real-weights Qwen3-0.6B gate runs manually on
   GPU). `qwen3_5` / `qwen3_5_moe` and dense Gemma 4 text use committed tiny
   Transformers fixtures instead. The cached merged-weight decoders are pinned the
   same way against both the uncached forwards and reference KV-cached paths where a
   committed cache oracle is present.
3. **Grad-coverage canary** (`lora::tests::grad_coverage_canary_contract`) — guards a
   real footgun: candle optimizers **silently skip** any parameter missing from the
   `GradStore`. The contract is init-dependent: at standard zero-`B` init only `B`
   is guaranteed a non-zero gradient (`A`'s is legitimately ~0); after a non-zero-`B`
   step **both** `A` and `B` must be present and non-zero.
4. **End-to-end finite-difference gradcheck** of the exact production GRPO loss
   w.r.t. the LoRA params (both clip branches × both advantage signs, k3 KL, masked
   and ragged completion masks, both reductions), hermetic in f64.

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

## Wire your own task

ferrl is **bring your own task**: you supply a *reward* and a *dataset of typed
samples*, and the trainer does the rest. There are two paths in.

### From the CLI — a built-in task

The `ferrl` binary trains a built-in task end-to-end from a JSON config, and reports
on a finished run:

```sh
cargo build --release            # builds the `ferrl` binary (target/release/ferrl)
ferrl train --config run.json    # GRPO-train a built-in task; writes a run under runs/
ferrl runreport runs/<run-id> --config run.json
                                # one-glance health summary + configured post-run policy
```

A `run.json` selects a task, points at a supported model checkpoint, and carries the trainer
config (only `task`, `model_dir`, and `trainer` are required; everything else has a
default):

```jsonc
{
  "task": "countdown",                 // built-in: "countdown" or "math"
  "model_dir": "/path/to/qwen3-0.6b-base",
  "device": "cpu",                     // or "cuda" (needs a --features cuda build)
  "data": { "train_n": 64, "eval_n": 16 },
  "trainer": { "steps": 50, "group_size": 8, "max_new_tokens": 48,
               "temperature": 1.0, "mu": 1, "beta": 0.0, "clip_eps": 0.2,
               "lr": 1e-5, "weight_decay": 0.0,
               "loss_type": "grpo", "scale_rewards": "group" }
}
```

For deterministic trainer-owned scalar control, `trainer.beta_schedule` and
`trainer.lr_schedule` accept piecewise-linear points:

```jsonc
{
  "trainer": {
    "beta_schedule": { "points": [
      { "step": 0, "value": 0.0 },
      { "step": 10, "value": 0.02 }
    ] },
    "lr_schedule": { "points": [
      { "step": 0, "value": 0.0 },
      { "step": 2, "value": 1e-5 }
    ] }
  }
}
```

Schedules start at step `0`, have strictly increasing in-range steps, interpolate
linearly between points, and hold the last value. `lr_schedule` replaces both `lr`
and `warmup_steps`; encode warmup directly as points.

### Tensor-parallel `ferrl train`

Sharded tensor-parallel execution supports Qwen3 (`model_type: "qwen3"`, including
legacy configs without `model_type`) and dense Gemma 4 (`"gemma4"` or
`"gemma4_unified"`). Qwen3.5/3.6 (`"qwen3_5"` / `"qwen3_5_moe"`) are not supported.
Build with `--features nccl`, set `device` to `"cuda"`, and launch one Slurm task per
TP rank. Each task's JSON `tensor_parallel.rank` must equal `SLURM_PROCID`, its
`world_size` must equal `SLURM_NTASKS`, and every task must use the same launch-unique
`FERRL_NCCL_RENDEZVOUS`. Generate or template one JSON file per rank; a shared file
hard-coded to rank 0 is not a valid multi-rank launch. Configs must otherwise be
identical. The presence of `FERRL_NCCL_RENDEZVOUS` tells ferrl to bootstrap the Slurm/NCCL
runtime before reading any rank's JSON, so a rank-local config failure is coordinated before
later collectives. It then validates every enabled JSON plan, including world 1, against the
live communicator before device/model setup. The parsed, default-expanded configs are also
canonicalized and compared across ranks after normalizing only `tensor_parallel.rank`; any
other difference aborts the launch in lockstep before task, device, or model dispatch.

```jsonc
{
  "device": "cuda",
  "tensor_parallel": { "enabled": true, "rank": 0, "world_size": 2 }
}
```

TP cannot currently be combined with `distributed.enabled`; any enabled TP execution rejects
`policy.base_quantization = "q8_0"`, including `world_size = 1`, while sharded TP also rejects
activation checkpointing and held-out eval. Ordinary world-one execution with TP disabled keeps
Q8_0 quantized matmul support; explicit TP will reopen it only after projection weights are
constructed as persistent rank-local quantized shards rather than repeatedly dequantizing full
projections. `intermediate_size`, `num_attention_heads`, and every layer's
effective KV-head count must divide evenly by `world_size` (both sliding and global KV-head
counts matter for Gemma 4). Frozen checkpoint weights and trainable LoRA adapters remain fully
replicated on every rank until sharded safetensors loading lands.
TP rank 0 is authoritative for reward evaluation, metrics, candidate ledgers,
checkpoints, post-run health, and the advertised output directory.

`countdown` generates its data procedurally; `math` is file-backed — set `data.path`
to a JSONL dataset of `{"prompt": ..., "target": {"answer": ...}}` lines (see
`crates/ferrl/tests/fixtures/math_dataset.jsonl`).

### TriMul discovery runs

The built-in `trimul` task is ferrl's first discovery task: training samples are
candidate GPU kernels, and success is an emitted artifact rather than just a rising
reward curve. Before spending GPU time on a TriMul run, use the
[TriMul Discovery Run Contract](docs/trimul-discovery-run-contract.md). It defines the
artifact bundle, provenance fields, same-GPU baseline pin, held-out verification,
dynamic reward-hacking checks, and the no-win stopping report that the operator audits.
Set `trainer.candidate_log_top_k` to a positive value for discovery runs so the best
sampled completions are persisted in `candidates.jsonl`; pass that ledger row's raw
completion plus its step/prompt/group/rank coordinates to `ferrl trimul-artifact`
(see the contract for the full command) to extract `submission.py`, re-verify with
an audit seed, and write the manifest/report. Include `--prompt-copy
<run-dir>/prompt.txt` so the artifact uses the rendered model prompt frozen at
training launch; extraction verifies the adjacent `<run-dir>/prompt.sha256`.
For rollout-only diagnostics from an external inference runtime, use
`ferrl trimul-score --config <run.json> --prompt-copy <prompt.txt>
--completion <raw.txt> --out <scores.jsonl> --score-secret-seed <seed>` (or
`--completions-jsonl`) to score raw completions once with the same shaped TriMul
reward and persist external-score JSONL. The scoring seed must differ from the
training `trimul.secret_seed`. `trimul-score` records opaque `source_id` values,
not input file paths; use `--source-label <public-id>` or JSONL `source_id` values
that are safe to copy into public reports. The default completion contract is strict:
ferrl scores exactly the completion bytes supplied. For GGUF rollouts served through
llama.cpp, pass `--completion-normalization llama-cpp` to `trimul-score` and
`trimul-artifact`; this strips only llama.cpp's trailing `[end of text]` stdout
sentinel before extraction and records raw and normalized hashes in the score/artifact
metadata. Then run `ferrl trimul-artifact` only on promising extracted candidates;
`trimul-score` is diagnostic evidence, not the strict artifact gate.
Verifier-backed rewards may also attach `reward_diagnostic` to candidate rows so
low or zero rewards remain explainable without re-running the whole training step; for
reward-tail triage, set `candidate_log_top_k` at least as high as `group_size` so every
sampled completion is retained. TriMul's training reward is shaped for search density;
test-passing candidates whose eval reaches a benchmark marker get a correctness floor,
and artifact acceptance still requires clean held-out correctness plus repeated measured
speedup through `ferrl trimul-artifact`. The run-config schema accepts the explicit
reward profile below. Omit `trimul.reward` to use these `trimul_shaped_v1` defaults,
or tune the numeric values to adjust discovery density. Custom profiles must preserve
the reward ladder: `format_extracted <= runnable` and
`runnable + partial_correctness <= correctness`; implausibly fast benchmark timings
remain fail-closed at zero.

```jsonc
"trimul": {
  "reward": {
    "scheme": "trimul_shaped_v1",
    "format_extracted": 0.02,
    "runnable": 0.05,
    "partial_correctness": 0.75,
    "correctness": 1.0,
    "speed_cap": 2.0,
    "implausible_benchmark": "zero"
  }
}
```

The top-level `run_health` schema configures deterministic post-run discovery-health
policy. `warn` findings are reported; `fail` findings make `ferrl train` fail after
telemetry is written and make `ferrl runreport --config <run.json>` exit with code `2`.
The current policy covers trailing reward collapse, trailing TriMul correctness collapse
from candidate metadata, dropped rows, grad spikes, dark off-policy telemetry, and
candidate source dominance. The `stop` action is reserved for a future in-run gate and is
rejected today. Correctness and source-dominance checks require a candidate ledger, so set
`trainer.candidate_log_top_k >= trainer.group_size` when using them. Partial top-K
candidate ledgers fail closed for those checks because they cannot represent the whole
sampled group.
The optional `trimul.verifier_parallelism` knob
keeps the default sequential verifier path at `1`; raise it only with a matching
`trimul.verifier_cuda_device_pool` that gives each concurrent verifier worker its own
GPU, and record verifier settings in the artifact manifest for like-for-like comparisons.
`trimul.verifier_max_procs` controls the verifier sandbox process cap (`ulimit -u`);
`0` or omission uses the TriMul default, currently `1024`. This cap is finite for
fork-bomb containment, but it must be comfortably above the allocation's ambient
per-UID task count because `RLIMIT_NPROC` is not per container.
For prompt experiments, set `trimul.prompt_path` to a UTF-8 file containing the
complete rendered model prompt, including any system text, chat markers, assistant
prefill, and reasoning prefix required by the model. TriMul training has no
built-in prompt fallback, suffix path, or prompt wrapper: `ferrl train` sends that
file as-is and freezes the same bytes into the run directory as `prompt.txt` with
`prompt.sha256`. Select the completion parser separately with
`trimul.submission_extract_mode` (`final_fence` or `thinking_after_think`); this
setting never constructs prompt text. Artifact bundles copy the verified frozen
prompt and record `prompt_sha256`. Reports should use that copy/hash or another
stable non-private identifier, not the mutable local `trimul.prompt_path`.

For Qwen3.5/3.6 instruct checkpoints that use ChatML, the prompt file itself should
already be rendered in the tokenizer's chat format. A thinking TriMul prompt usually
has this shape:

```text
<|im_start|>system
You generate Python code for a strict evaluator.
Output contract:
- Close </think> before writing code.
- Immediately after </think>, output exactly one closed fenced Python code block.
- The code block must contain only the complete custom_kernel(data) implementation.<|im_end|>
<|im_start|>user
Implement `custom_kernel(data)` for the TriMul evaluator.
...task details...<|im_end|>
<|im_start|>assistant
<think>
```

Use `trimul.submission_extract_mode = "thinking_after_think"` for that form, because
the reward extractor waits for `</think>` and then reads the final fenced code block.
If the model/checkpoint does not use ChatML, do not copy these markers blindly; render
the complete prompt according to that checkpoint's tokenizer/chat template and point
`trimul.prompt_path` at those exact bytes.

### From Rust — a task that isn't built in

Implement `RewardFn` over your own typed target and hand the trainer a
`Vec<Sample<T>>`. The full runnable template is
[`examples/minimal_task.rs`](crates/ferrl/examples/minimal_task.rs); the core is:

```rust
use ferrl::{RewardError, RewardFn, Sample};

struct ContainsKeyword;
impl RewardFn for ContainsKeyword {
    type Target = String;                       // your typed ground truth
    fn reward(&self, sample: &Sample<String>, completion: &str) -> Result<f32, RewardError> {
        Ok(if completion.contains(sample.target.as_str()) { 1.0 } else { 0.0 })
    }
}
```

Then load a policy (`ferrl::load_qwen_policy`), build a config
(`TrainerConfig::builder()`), and call `Trainer::train`. Datasets load from JSONL via
`ferrl::read_jsonl`; the library's worked rewards (`ferrl::CountdownReward`,
`ferrl::MathReward`) are fuller references.

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
                           "eps_low", "eps_high",
                           "value" } ],                 // min(rA, clip(r)·A), asym bands
  "masked_mean": { "per_token"[S][T], "mask"[S][T],
                   "grpo", "dr_grpo", "dapo" },         // the three reductions
  "sequence_log_ratio": [ { "logp"[T], "logp_old"[T],
                            "mask"[T], "value" } ]      // GSPO per-sequence log-ratio
}
```

Formulas (per token, with `std` the *sample* / ddof=1 standard deviation):

```
advantage_i      = (r_i - mean(r)) / (std(r) + eps)          # eps = 1e-4
ratio            = exp(logp - old_logp)
clipped_surrogate= min( ratio * adv, clip(ratio, 1-e_low, 1+e_high) * adv )
k3_kl            = exp(d) - d - 1,   d = ref_logp - logp      # >= 0
masked_mean(grpo)   = mean_seq( sum_t v·m / max(sum_t m, 1) )  # all-pad row -> 0 (TRL clamp)
masked_mean(drgrpo) = sum_all(v·m) / (num_seq · max_len)       # Dr.GRPO; requires max_len >= 1
masked_mean(dapo)   = sum_all(v·m) / max(sum_all(m), 1)        # DAPO active-token normalizer
seq_log_ratio       = sum_t m·(logp-old) / max(sum_t m, 1)     # GSPO sequence-level ratio (log)
```

---

## Training run layout (`runs/`)

Each training run writes to `runs/<run_id>/`:

```
runs/<run_id>/
├── config.json       # the trainer config written by the generic Trainer
├── metrics.jsonl     # one JSON object per step:
│                     #   step, reward_mean, reward_std, frac_reward_zero_std,
│                     #   kl, clip_ratio, frac_truncated, completion_len,
│                     #   rollout_ratio_mean, rollout_logratio_mean,
│                     #   rollout_ratio_max, frac_rollout_ratio_capped,
│                     #   rollout_capture_tokens, dropped_rows, grad_norm, lr, beta,
│                     #   step_secs, tokens_per_sec, cuda_mem_* when enabled,
│                     #   cuda_mem_probe_events and decoder_cache_snapshots when present
├── checkpoints/      # LoRA checkpoints
└── run.log           # human-readable log
```

Logging is structured via `tracing` + `tracing-subscriber`: the trainer enters a
`run{rank=N world=N}` span and a per-step `step{step=N}` span, so every event carries
rank / world / step (at ERROR level, the fields survive even `RUST_LOG=warn`). `runs/`
is git-ignored. The on-disk layout is created by [`ferrl::telemetry::RunDir`] and metrics
are appended by `ferrl::telemetry::MetricsWriter`; with `gpu_memory_probe: true`,
CUDA runs also persist stable phase-level memory probe events and any model-provided
decoder cache snapshots, such as Gemma 4 per-layer seen/retained token counts. The
`ferrl runreport` subcommand reads a
run's `metrics.jsonl` and prints a health summary — reward trend, throughput, and grad-norm
anomalies (human, `--json`, or `--strict`). Pass the original top-level run config with
`--config run.json` to also apply its `run_health` post-run policy.
Under sharded TP, rank-suffixed directories are created for process-local trainer state,
but only TP rank 0 contains authoritative telemetry and is a valid `runreport` target.

For manual GPU resource gates, run the same smoke or training command on a baseline
commit and a candidate commit, then compare their per-rank metrics with `perf-gate`:

```sh
ferrl perf-gate --baseline runs/main-rank0 --candidate runs/pr-rank0 \
  --max-peak-mem-regression-pct 0 \
  --max-step-secs-regression-pct 10 \
  --max-final-grad-norm-rel-drift 0.0001
```

The gate fails closed if the streams are empty, steps are misaligned, candidate health warnings
change, `grad_norm` never goes positive, or required timing / CUDA memory probes are absent. Run
baseline and candidate on the same GPU model, world size, cargo features, and command.

Qwen3.5 runs also expose an experimental rollout-memory lever under the run config policy
block: set policy.memory_efficient_cached_gqa to true. It keeps the cached K/V store
compact during grouped-query attention decode instead of materializing repeated K/V tensors.
The default is false so existing runs keep the shipped cached decoder arithmetic; treat true
as an opt-in optimization and gate it against a same-prompt, same-GPU baseline with CUDA
memory telemetry enabled.

For data-parallel resource gates where a high-water rank can move between baseline and candidate,
repeat `--baseline` / `--candidate` once per rank and provide the expected world size explicitly:

```sh
ferrl perf-gate --distributed-world-max \
  --distributed-world-size 2 \
  --baseline runs/main-rank0 --baseline runs/main-rank1 \
  --candidate runs/pr-rank0 --candidate runs/pr-rank1 \
  --max-peak-mem-regression-pct 0 \
  --max-step-secs-regression-pct 10 \
  --max-final-grad-norm-rel-drift 0.0001
```

`--distributed-world-max` requires `--distributed-world-size`; together they fail closed for
missing supplied rank streams, rank-count mismatches, missing required telemetry, rank-local health
regressions, and rank/step misalignment, but compare CUDA memory by world maximum and step time by
the slowest rank. The stable
`NCCL_TINY_QWEN35_SMOKE` rows are still useful for humans, but the public comparator reads
`metrics.jsonl`.

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
