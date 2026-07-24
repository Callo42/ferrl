//! # ferrl
//!
//! `ferrl` is a [candle](https://github.com/huggingface/candle)-native
//! [GRPO](https://arxiv.org/abs/2402.03300) reinforcement-learning library for
//! RL-fine-tuning large language models in Rust. The current native model
//! families are Qwen3, Qwen3.5/3.6, and dense Gemma 4 text.
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
//! - the **full fine-tuning opt-in** (`full_ft`, internal) — a `Var`-registry
//!   `VarBuilder` backend that wraps every base weight a load fetches in a
//!   trainable `Var`, in deterministic load order (the positional checkpoint
//!   contract); `LoRA` stays the default, and the mode is entered per model
//!   ([`Qwen3_5GradModel::load_full_ft`](qwen35::Qwen3_5GradModel::load_full_ft));
//! - grad-safe building blocks and the grad-coverage canary ([`nn`]);
//! - the model-generality seam ([`model`]) — the [`GradModel`] / [`CachedDecoder`]
//!   traits: the entire surface a model must provide (grad-bearing full-sequence
//!   forward, trainable `LoRA` vars, adapter toggle, merged cached decoder) for
//!   the generic policy to RL-fine-tune it;
//! - architecture-neutral decoder building blocks ([`blocks`]) — frozen linear,
//!   GQA `repeat_kv`, rotate-half `RoPE` tables, and the causal-mask builders,
//!   shared by every model implementation;
//! - a grad-bearing, uncached `Qwen3` forward ([`qwen`]) — the trainable update
//!   path, weight-identical to candle's shipped (no-grad) forward; the first
//!   [`GradModel`] implementor;
//! - a grad-bearing, uncached dense Llama-3.x forward ([`llama`]) — the second
//!   [`GradModel`] implementor (plain GQA / rotate-half `RoPE` with optional
//!   llama3 scaling / `SwiGLU`, no QK-norm, no biases), weight-identical to
//!   candle's shipped `llama::Llama` and pinned to it by the same per-position
//!   equivalence oracle — the witness that swapping the model is a bounded,
//!   gated exercise;
//! - a [`Policy`] over any [`GradModel`] ([`lm_policy`]) — [`LmPolicy`] wraps a
//!   grad forward as the trainer's policy seam, with KV-cached, adapter-aware
//!   rollout; [`QwenPolicy`] and [`LlamaPolicy`] are its instantiations, so the
//!   same `Trainer` drives Qwen3, Llama, and the P2 toy unchanged;
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
//! - checkpointing ([`checkpoint`]) — explicit legacy-v1 adapter-only save/load for
//!   eval ([`save_adapter`]), and an identity/integrity-bound, momentum-faithful
//!   ordinary format-v4 checkpoint ([`save_checkpoint`]) that binds immutable policy
//!   content, canonical learner semantics/topology, exact recipe/schema, adapter and
//!   Adam payloads, sampler state, and the completed-step relationship under one
//!   state-envelope root before [`Trainer::resume`] mutates live state;
//! - the separated rollout/learner artifact contract ([`rollout_ledger`]) — a
//!   strict, checksummed, no-replace whole-window package whose reader validates
//!   learner pre-state identity, mandatory structured controls, and every
//!   rollout/reward/mask invariant before returning [`ValidatedRolloutLedgerStep`].
//!   Format v6 also binds sampler prestate and chain lineage, the independently
//!   encoded selected prompt for every group, transfers the
//!   collector's exact post-rollout sampler blob, and returns an opaque
//!   learner-produced receipt for a versioned adapter + Adam + sampler
//!   continuation whose policy/config/schema/payload lineage is verified. Under
//!   data parallelism it publishes immutable rank shards behind one global
//!   manifest-last commit marker and validates the complete world before replay.
//!   Under tensor parallelism every model shard executes and validates the same
//!   logical world-one package, execution rank 0 alone publishes it, replicated
//!   `LoRA` gradients are sum-reduced before Adam, and continuation v3 binds both
//!   DP/TP topology plus canonical communicator-rank shard ordering;
//! - held-out evaluation ([`eval`]) — the base model vs. the trained adapter,
//!   mean reward over a held-out set (the P4 gate's comparison);
//! - activation checkpointing ([`remat`]) — candle ships no checkpoint
//!   primitive, so ferrl orchestrates layer-boundary rematerialization itself:
//!   the checkpointed forward cuts the autograd graph at every layer boundary
//!   and `backward` re-runs one layer at a time, stitching the full gradient
//!   from the boundary tape (the primary single-card memory lever — opt-in via
//!   each model's `set_activation_checkpointing`);
//! - `MoE` primitives ([`moe`]) — the qwen3.5/3.6 sparse layer's kernels
//!   (top-k router with unconditional renorm, packed-weight experts, the
//!   sigmoid-gated shared expert), grad-bearing and oracle-pinned, wired into
//!   [`qwen35`]'s feed-forward layer menu so the same `Qwen3_5GradModel`
//!   loads both the dense and the `MoE` family members (M3′);
//! - the distributed communication seam ([`comm`]) — the [`Comm`] trait
//!   (rank identity + sum-reductions) drives data-parallel accumulated-gradient
//!   reduction and tensor-parallel activations, rewards, control, and adapter
//!   gradients while keeping every rank in lockstep;
//!   [`SoloComm`] is the world-1 default (the single-rank path stays
//!   bit-identical to the pre-DP trainer), [`LocalComm`] runs an N-thread
//!   single-process world for CPU-testable DP/TP equivalence oracles, and
//!   [`NcclComm`] is the real multi-GPU implementation whose `unsafe` cudarc
//!   collective is quarantined behind `--features nccl`;
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
//
// `deny`, not `forbid`: the default build compiles **zero** `unsafe` (the lint
// errors on any), but `forbid` cannot be locally overridden, and the
// `--features nccl` NCCL FFI ([`comm`]'s decision-D2 quarantine) needs one
// `#[allow(unsafe_code)]` in its single gated module. That module is not
// compiled without the feature, so the default/CI build is exactly as
// unsafe-free as before; only a GPU-cluster `--features nccl` build carries the
// one allow.
#![deny(unsafe_code)]

pub mod blocks;
pub mod checkpoint;
pub mod comm;
pub mod countdown;
pub mod cuda_compat;
pub mod data;
pub mod dense;
pub mod eval;
mod full_ft;
pub mod gdn;
pub mod gemma4;
pub mod grpo;
pub mod hf;
pub mod llama;
pub mod lm_policy;
pub mod loader;
pub mod lora;
pub mod math;
pub mod model;
pub mod moe;
pub mod nn;
pub mod optim;
pub mod policy;
pub mod qwen;
pub mod qwen35;
pub mod remat;
pub mod reward;
pub mod rollout_ledger;
pub mod sample;
pub mod sampler;
pub mod sandbox;
mod sharded_safetensors;
pub mod telemetry;
pub mod tensor_parallel;
pub mod tokenizer;
pub mod trainer;
pub mod trimul;

#[doc(inline)]
pub use checkpoint::{
    latest_checkpoint, load_adapter, load_checkpoint, save_adapter, save_checkpoint,
    CheckpointBinding, CheckpointError, CheckpointManifest, LatestCheckpoint, LoadedCheckpoint,
    OrdinaryCheckpointIdentity,
};
#[cfg(feature = "nccl")]
pub use comm::RealNccl;
#[doc(inline)]
pub use comm::{Comm, CommError, LocalComm, NcclComm, NcclConfig, NcclPrimitives, SoloComm};
#[doc(inline)]
pub use countdown::{
    build_prompt, generate_dataset, CountdownConfig, CountdownProblem, CountdownReward,
};
#[doc(inline)]
pub use cuda_compat::{check_driver_compat, guard_first_kernel, translate_ptx_error, CompatReport};
#[doc(inline)]
pub use data::{parse_jsonl, read_jsonl, train_eval_split, DataError};
#[doc(inline)]
pub use eval::{evaluate, EvalError, EvalReport, PromptEval};

pub use gdn::{
    causal_depthwise_conv1d, gated_delta_rule_chunked, gated_delta_rule_recurrent, l2norm,
    stable_softplus,
};
#[doc(inline)]
pub use gemma4::{
    Gemma4Config, Gemma4FullAttentionRope, Gemma4GradModel, Gemma4LayerType, Gemma4MergedDecoder,
    Gemma4RopeParameters, Gemma4SlidingAttentionRope, Gemma4TextConfig,
    CKPT_PREFIX as GEMMA4_CKPT_PREFIX,
};
#[doc(inline)]
pub use grpo::{
    clipped_surrogate, group_advantages, k3_kl, masked_mean, tis_weight, zero_mask_rows, LossType,
    ScaleRewards, GROUP_STD_EPS,
};
#[doc(inline)]
pub use hf::{
    chatml, checkpoint_eos_from_config, eos_from_config, resolve_checkpoint_eos,
    validate_resolved_eos_consensus, vocab_size_from_config, CheckpointEos, CheckpointEosSelection,
    HfError,
};
#[doc(inline)]
pub use llama::{LlamaGradModel, LlamaMergedDecoder};
#[doc(inline)]
pub use lm_policy::{Gemma4Policy, LlamaPolicy, LmPolicy, Qwen3_5Policy, QwenPolicy};
#[doc(inline)]
pub use loader::{
    checkpoint_policy_sha256, load_auto_policy, load_auto_policy_bound, load_gemma4_policy,
    load_gemma4_policy_bound, load_qwen35_policy_with_targets_bound, load_qwen_policy,
    load_qwen_policy_bound, AutoPolicy, LoaderError, LoaderOpts,
};
#[doc(inline)]
pub use lora::{BaseQuantization, DenseLoraTargets};
#[doc(inline)]
pub use math::{math_prompt, MathProblem, MathReward};
#[doc(inline)]
pub use model::{CachedDecoder, GradModel};
#[doc(inline)]
pub use moe::{moe_experts, sparse_moe_block, topk_router, SparseMoeWeights};
#[doc(inline)]
pub use nn::{grad_coverage, GradCoverage, RmsNorm, RmsNormGated, RmsNormZeroCentered};
#[doc(inline)]
pub use optim::{FerrlAdamW, OptimizerState};
#[doc(inline)]
pub use policy::{EvalSampling, Policy, TensorParallelPolicy};
#[doc(inline)]
pub use qwen::{MergedDecoder, QwenGradModel};
#[doc(inline)]
pub use qwen35::{
    tensors_from_pretrained, varbuilder_from_pretrained, LayerType, LoraTargets, MoeDims,
    Qwen3_5Config, Qwen3_5GradModel, Qwen3_5MergedDecoder, Qwen3_5TextConfig, RopeParameters,
    GDN_CHUNK_SIZE,
};
#[doc(inline)]
pub use remat::{stitched_backward, RematTape};
#[doc(inline)]
pub use reward::{RewardError, RewardFn, RewardOutcome};
#[doc(inline)]
pub use rollout_ledger::{
    DistributedRolloutLedgerStage, LedgerScoreRequirement, RolloutLedgerControls,
    RolloutLedgerError, RolloutLedgerExpectations, RolloutLedgerGroup, RolloutLedgerGroupScope,
    RolloutLedgerIdentity, RolloutLedgerReader, RolloutLedgerRewardStats, RolloutLedgerStep,
    RolloutLedgerWriter, ValidatedRolloutLedgerStep, ROLLOUT_LEDGER_FORMAT_VERSION,
};
#[doc(inline)]
pub use sample::Sample;
#[doc(inline)]
pub use sampler::GrpoSampler;
#[doc(inline)]
pub use sandbox::{
    ApptainerSandbox, Bind, BindMode, NetworkPolicy, ResourceLimits, RunOutcome, RunSpec,
    RunStatus, Sandbox, SandboxError,
};
#[doc(inline)]
pub use telemetry::{
    compare_distributed_metrics, compare_metrics, cuda_memory_snapshot, init_tracing, read_metrics,
    run_span, summarize, Anomaly, DecoderCacheSnapshot, GpuMemoryProbeEvent, GpuMemorySnapshot,
    Metrics, MetricsWriter, ModelTelemetryRecorder, RegressionBudget, RegressionFailure,
    RegressionReport, RunDir, RunSummary,
};
#[doc(inline)]
pub use tensor_parallel::{
    all_reduce_sum_straight_through, column_parallel_linear, concat_column_shards, plan_from_comm,
    row_parallel_linear_partial, row_parallel_linear_partial_from_shard, sum_row_parallel_partials,
    validate_comm_plan, ShardRange, TensorParallelDims, TensorParallelError, TensorParallelPlan,
};
#[doc(inline)]
pub use tokenizer::{HfTokenizer, TokenizerError};
#[doc(inline)]
pub use trainer::{
    RewardGroupScope, RolloutLedgerContinuation, RunStop, ScalarSchedule, SchedulePoint,
    TokenizerLike, Trainer, TrainerConfig, TrainerConfigBuilder, TrainerError,
};
#[doc(inline)]
pub use trimul::{Distribution, TrimulCase, TrimulError, TrimulReward};
