# AGENTS.md

Guidance for AI coding agents working in this repository. Human contributors:
see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Hard rules

- **Never push to `main`. Never merge a PR.** Open a pull request from a feature
  branch and **wait for human review.** A human reviews and merges.
- **Every change goes through a PR**, and the **CI gate must be green** before you
  ask for review.
- **Run the full local gate before pushing** — `just gate` (fmt, clippy
  `-D warnings`, tests + coverage ≥ 90, docs). Don't push red.
- **Conventional Commits** for every commit message (`cog check` enforces it).
- **No secrets, credentials, machine-specific paths, or personal data** in commits.

## Project shape

ferrl is a candle-native GRPO reinforcement-learning library for RL-fine-tuning
LLMs. We **own the RL layer** (GRPO loss, reward interface, LoRA adapters, rollout,
trainer, and a grad-bearing model forward) and **delegate all tensor math,
autograd, GPU, and the base model forward to [candle](https://github.com/huggingface/candle)**.

- Library crate: `crates/ferrl`.
- Core seams: `RewardFn` (user rewards, plain `f32`), `Policy` (generate +
  token-logprobs + adapter toggle), `LoraLinear` (frozen base + low-rank A/B),
  the GRPO math, and the `Trainer`.
- Telemetry: `tracing`; each run writes `runs/<run_id>/` (config + metrics.jsonl +
  checkpoints). `runs/` and `target/` are git-ignored.

## Gotchas to respect

- candle's fused `RmsNorm`/`LayerNorm` have **no backward** — use
  `candle_nn::ops::rms_norm_slow` on any gradient-bearing path.
- candle optimizers **silently skip** parameters absent from the gradient store —
  assert a grad-coverage canary (every trainable adapter `Var` receives a nonzero
  gradient) after `backward()`.
- The shipped Qwen forward is inference-shaped (`&mut self` KV-cache); the training
  update needs a separate uncached, full-sequence, gradient-bearing forward.
