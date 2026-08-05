# AGENTS.md

Guidance for AI coding agents working in this repository. Human contributors:
see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Hard rules

- **Never push to `main`. Never merge a PR.** Open a pull request from a feature
  branch and **wait for human review.** A human reviews and merges.
- **Every change goes through a PR**, and the **CI gate must be green** before you
  ask for review.
- **Run the core local CPU gate before pushing** — `just gate` (fmt, clippy
  `-D warnings`, check, tests + coverage ≥ 90, docs). GitHub CI adds the
  feature-gated verifier target, supply-chain, MSRV, and commit-range checks.
  Don't push red.
- **Conventional Commits** for every commit message (`cog check` enforces it).
- **No secrets, credentials, machine-specific paths, or personal data** in commits.

## Project shape

ferrl is a candle-native, RL-driven **discovery platform**: given a verifiable
task and a base model, it searches with reinforcement learning and emits a
verified performance artifact. TriMul GPU-kernel discovery is the first target;
GRPO + LoRA is the first training recipe. We own the RL, reward-verification,
search, provenance, and artifact boundaries, including the grad-bearing model
forwards they require. We delegate tensor math, autograd, GPU primitives, and
the shipped inference stack to [candle](https://github.com/huggingface/candle).

- Library crate: `crates/ferrl`.
- Core seams: `Sample` + `RewardFn` (typed tasks and scalar verifiable rewards),
  `Policy` (generate + token-logprobs + adapter toggle), `LoraLinear` (frozen
  base + low-rank A/B), the GRPO math, and `Trainer`. Discovery adds the
  `trimul`, `sandbox`, and `verifier_executor` boundaries plus launch-bound
  candidate, evaluation, and artifact provenance in the CLI and telemetry.
- Data parallelism: a `Comm` seam (`SoloComm`/`LocalComm`, plus an NCCL bridge behind
  `--features nccl`) all-reduces LoRA gradients for single-node multi-GPU DP, with
  DP-coordinated resume. The same communicator seam drives single-node tensor-parallel
  Qwen3 and dense Gemma 4 execution through `ferrl train`; Gemma 4 streams rank-local
  frozen projection shards from safetensors while shared weights and every LoRA adapter
  remain replicated (Qwen3 currently keeps a fully replicated frozen-base fallback).
  Combined sharded DP x TP is rejected, and TP rank 0 owns rewards, telemetry,
  checkpoints, post-run health, and advertised output. Slurm/NCCL
  launches must set a launch-unique `FERRL_NCCL_RENDEZVOUS`; ferrl uses it to bootstrap
  before loading rank-local configs, then validates every enabled TP rank/world plan.
- Telemetry: `tracing` (run/step events stamped with `rank`/`world`/`step`);
  applicable files under `runs/<run_id>/` include immutable launch/config and
  prompt evidence, `metrics.jsonl`, optional `candidates.jsonl`, checkpoints,
  and `eval-report.json`. The `ferrl runreport` subcommand summarizes a finished
  run. `runs/` and `target/` are git-ignored.

## Gotchas to respect

- candle's fused `RmsNorm`/`LayerNorm` have **no backward** — use
  `candle_nn::ops::rms_norm_slow` on any gradient-bearing path.
- candle optimizers **silently skip** parameters absent from the gradient store —
  assert a grad-coverage canary after `backward()`: every trainable adapter `Var`
  must be **present with a finite gradient** (an absent entry or a non-finite
  value aborts; an all-zero gradient is a legitimate no-signal state — e.g. a
  fully clipped step or an all-masked window — and skips the optimizer step).
- The shipped Qwen forward is inference-shaped (`&mut self` KV-cache); the training
  update needs a separate uncached, full-sequence, gradient-bearing forward.
- The build toolkit's **PTX ISA** must be `<=` the runtime NVIDIA driver's maximum, or
  the **first** CUDA kernel load fails at run time with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (driver error 222). `CUDA_COMPUTE_CAP` sets the
  SM architecture, **not** the ISA — only the `nvcc` version sets the ISA. The
  `cuda_compat` preflight translates this into an actionable message
  (`guard_first_kernel` reactive + auto-applied, `check_driver_compat` proactive warn);
  see the README "GPU builds" → "CUDA driver compatibility".
