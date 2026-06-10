//! # ferrl
//!
//! `ferrl` is a [candle](https://github.com/huggingface/candle)-native
//! [GRPO](https://arxiv.org/abs/2402.03300) reinforcement-learning library for
//! RL-fine-tuning large language models in Rust. The first target model is
//! `Qwen3-0.6B-Base`.
//!
//! ## Division of labor
//!
//! `ferrl` owns only the *reinforcement-learning layer*:
//!
//! - the pure GRPO math ([`grpo`]) — advantages, the k3 KL estimator, the
//!   clipped surrogate, and the masked-mean reductions;
//! - the [`reward`] abstraction (scalar rewards, never tensors);
//! - a reference verifiable task ([`countdown`]) — the Countdown arithmetic task: a
//!   solvable-by-construction problem generator, a few-shot prompt builder, and a
//!   shaped, exact [`CountdownReward`] (the P4 task);
//! - the [`policy`] abstraction over generation and per-token log-probabilities;
//! - a manual `LoRA` adapter ([`lora`]);
//! - grad-safe building blocks and the grad-coverage canary ([`nn`]);
//! - a grad-bearing, uncached `Qwen3` forward ([`qwen`]) — the trainable update
//!   path, weight-identical to candle's shipped (no-grad) forward;
//! - a [`Policy`] over the real model ([`qwen_policy`]) — [`QwenPolicy`] wraps that
//!   grad forward as the trainer's policy seam, with uncached, adapter-aware
//!   rollout, so the same `Trainer` drives Qwen3 as the P2 toy;
//! - a ferrl-owned rollout sampler ([`sampler`]) — [`GrpoSampler`] reproduces
//!   candle's temperature multinomial sampling on a `serde`-serializable
//!   `Xoshiro256PlusPlus`, so the rollout RNG can be captured and restored for
//!   momentum-faithful resume (replacing candle's accessor-less `LogitsProcessor`);
//! - a real-model tokenizer adapter ([`tokenizer`]) — [`HfTokenizer`] wraps a
//!   Hugging Face fast tokenizer behind the trainer's [`TokenizerLike`] bridge;
//! - the GRPO training loop ([`trainer`]) — the `Trainer` that drives rollout →
//!   reward → advantages → masked clipped surrogate (+ optional KL) →
//!   canary-guarded [`FerrlAdamW`] step;
//! - a candle-bit-identical `AdamW` ([`optim`]) — [`FerrlAdamW`], a line-for-line
//!   clone of candle's optimizer that ferrl owns so it can persist and restore the
//!   moment state ([`OptimizerState`]) for momentum-faithful resume, pinned to candle
//!   by a permanent equivalence canary;
//! - checkpointing ([`checkpoint`]) — adapter-only save/load for eval
//!   ([`save_adapter`]), and a **momentum-faithful** v2 checkpoint
//!   ([`save_checkpoint`]) that also persists the optimizer moments and the rollout
//!   sampler RNG, so [`Trainer::resume`] continues an interrupted run **bit-exactly**;
//! - held-out evaluation ([`eval`]) — the base model vs. the trained adapter,
//!   mean reward over a held-out set (the P4 gate's comparison);
//! - a CUDA driver-compatibility preflight ([`cuda_compat`]) — translates the cryptic
//!   `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (a build-PTX-newer-than-driver mismatch) into
//!   an actionable rebuild/upgrade message; a no-op without the `cuda` feature;
//! - run telemetry ([`telemetry`]).
//!
//! Everything below the RL layer — tensors, autograd, optimizers, devices, and
//! the model itself — is **delegated** to `candle-core`, `candle-nn`, and
//! `candle-transformers`. In particular `ferrl` does not implement its own
//! autodiff: it builds a loss [`candle_core::Tensor`] from candle ops and calls
//! [`candle_core::Tensor::backward`].
//!
//! ## Correctness oracles
//!
//! Because the autodiff is not ours, correctness is pinned by oracles rather
//! than by re-deriving gradients: the pure GRPO math is checked against a
//! committed golden JSON fixture computed with `NumPy` (`std(ddof=1)`, matching
//! TRL/candle — `scripts/gen_golden.py`); and a grad-coverage canary asserts that
//! trainable `LoRA` [`candle_core::Var`]s appear in the grad store after
//! `backward` with the expected (init-dependent) non-zero gradients — candle
//! optimizers *silently* skip params missing from the grad store (see [`lora`]).
//! The custom [`qwen`] forward is pinned to candle's shipped Qwen3 forward by a
//! per-position equivalence oracle — on a tiny config in CI and on the real
//! `Qwen3-0.6B-Base` checkpoint in `#[ignore]`d, weights-gated tests. Finally, an
//! end-to-end **finite-difference gradcheck** pins candle's analytic gradient of
//! the GRPO loss — the exact `grpo_loss` the trainer back-propagates — against
//! central differences w.r.t. the `LoRA` parameters, exercising the clipped
//! surrogate, the k3 KL penalty, and both masked reductions.
//!
//! ## Stability
//!
//! Pre-`1.0`: the public surface may change between minor versions.
#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod countdown;
pub mod cuda_compat;
pub mod eval;
pub mod grpo;
pub mod lora;
pub mod nn;
pub mod optim;
pub mod policy;
pub mod qwen;
pub mod qwen_policy;
pub mod reward;
pub mod sampler;
pub mod telemetry;
pub mod tokenizer;
pub mod trainer;

#[doc(inline)]
pub use checkpoint::{
    load_adapter, load_checkpoint, save_adapter, save_checkpoint, CheckpointError,
    CheckpointManifest, LoadedCheckpoint,
};
#[doc(inline)]
pub use countdown::{
    build_prompt, generate_dataset, parse_problem_from_prompt, CountdownConfig, CountdownProblem,
    CountdownReward,
};
#[doc(inline)]
pub use cuda_compat::{check_driver_compat, guard_first_kernel, translate_ptx_error, CompatReport};
#[doc(inline)]
pub use eval::{evaluate, EvalError, EvalReport, PromptEval};
#[doc(inline)]
pub use grpo::{
    clipped_surrogate, group_advantages, k3_kl, masked_mean, zero_mask_rows, LossType,
    ScaleRewards, GROUP_STD_EPS,
};
#[doc(inline)]
pub use nn::{grad_coverage, GradCoverage, RmsNorm};
#[doc(inline)]
pub use optim::{FerrlAdamW, OptimizerState};
#[doc(inline)]
pub use policy::Policy;
#[doc(inline)]
pub use qwen::QwenGradModel;
#[doc(inline)]
pub use qwen_policy::QwenPolicy;
#[doc(inline)]
pub use reward::RewardFn;
#[doc(inline)]
pub use sampler::GrpoSampler;
#[doc(inline)]
pub use telemetry::{init_tracing, Metrics, MetricsWriter, RunDir};
#[doc(inline)]
pub use tokenizer::HfTokenizer;
#[doc(inline)]
pub use trainer::{TokenizerLike, Trainer, TrainerConfig, TrainerError};
