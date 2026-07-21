//! The GRPO trainer — the fifth and final seam.
//!
//! [`Trainer`] drives one GRPO optimizer step at a time over a [`Policy`] and a
//! [`RewardFn`], owning the pieces candle does not provide: the rollout →
//! reward → advantage → masked clipped-surrogate (+ optional KL) → backward →
//! **grad-coverage canary** → optimizer-step pipeline, plus the inner-update loop
//! (`μ`), gradient accumulation across prompts (`grad_accum_steps`), and per-step
//! telemetry.
//!
//! It is generic over the [`Policy`] and [`RewardFn`] traits and never names a
//! concrete model: the trainable parameters reach it only through
//! [`Policy::trainable_vars`], so the *same* `Trainer` drives the toy policy in
//! the integration tests today and a real Qwen policy later, unchanged.
//!
//! # Differentiable GRPO vs the scalar oracle
//!
//! [`crate::grpo`] holds the GRPO algebra as pure `f64` scalar functions — the
//! tensor-free, golden-pinned *oracle*. The trainer re-expresses that same
//! algebra in differentiable [`candle_core::Tensor`] ops (so candle can
//! back-propagate it) in the private helpers below; in-module unit tests pin the
//! tensor forms to the scalar oracle. Advantages are the exception: they are
//! detached constants, so [`crate::grpo::group_advantages`] is called directly
//! on the scalar rewards.
//!
//! # The canary is load-bearing
//!
//! candle optimizers *silently skip* parameters that never reached the loss (so
//! they are absent from the [`candle_core::backprop::GradStore`]). The trainer
//! therefore runs [`crate::grad_coverage`] on every real update and aborts the
//! run if a trainable var is **missing** (the silent-skip landmine — a genuine
//! autograd cut shows up as an absent grad entry, not a zero one) or its gradient
//! is **non-finite** (a blowup). A covered-and-finite but all-zero gradient is
//! *not* an abort: it is a legitimate no-signal state (the PPO trust region
//! binding on every token, or mean-centered advantages cancelling), so the inner
//! step simply performs no optimizer update. The reward-trend gate independently
//! backstops a genuinely dead wiring (reward would never rise).
//!
//! A *degenerate* group — every completion scored identically, so all advantages
//! are zero (`frac_reward_zero_std == 1`) — carries no surrogate signal. With
//! `beta == 0` it is a GRPO no-op: the trainer performs no update for that step
//! (and runs no canary). With `beta > 0` it stays **live** — the KL penalty still
//! pulls the group toward the reference (TRL keeps every completion in the batch),
//! only the surrogate contribution is zero.
//!
//! # Data parallelism
//!
//! [`Trainer::with_comm`] runs the same loop as one rank of a data-parallel
//! world (see [`crate::comm`]): each rank consumes its own shard of every
//! window's prompts, folds its local gradients, and **all-reduce-sums** them
//! before the canary — everything downstream of the reduce (canary, clip,
//! optimizer step) runs on the identical global gradient on every rank, so the
//! ranks' weights stay in **bitwise lockstep**. All normalizers go global
//! (per-item scale `1 / (grad_accum_steps · world)`, the DAPO token normalizer
//! and the degenerate-window decision all-reduced), checkpoints are written by
//! rank 0 only, and the world-1 path is byte-for-byte the pre-DP trainer.
//!
//! Tensor-parallel model execution is a separate opt-in path: the
//! `*_tensor_parallel` methods take an explicit [`Comm`] and require
//! [`TensorParallelPolicy`]. The trainer's own communicator still represents the
//! data-parallel gradient-reduction world. Sharded tensor-parallel training is
//! allowed only without simultaneous sharded data parallelism: `LoRA` trainable
//! vars stay fully replicated on every TP rank, the trainer all-reduce-sums their
//! accumulated gradients over the TP communicator before coverage, clipping, and
//! optimizer step, and TP rank 0 owns shared side effects such as checkpoints.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::optim::ParamsAdamW;
use candle_nn::Optimizer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::comm::{Comm, SoloComm};
use crate::grpo::{
    group_advantages, zero_mask_rows, ImportanceSamplingLevel, LossType, ScaleRewards,
    GROUP_STD_EPS,
};
use crate::nn::grad_coverage;
use crate::optim::{FerrlAdamW, OptimizerState};
use crate::policy::{GenConfig, Policy, Rollout, TensorParallelPolicy};
use crate::reward::{validate_reward_values, RewardError, RewardFn, RewardOutcome};
use crate::rollout_ledger::{
    LedgerScoreRequirement, RolloutLedgerControls, RolloutLedgerError, RolloutLedgerExpectations,
    RolloutLedgerGroup, RolloutLedgerGroupScope, RolloutLedgerIdentity, RolloutLedgerReader,
    RolloutLedgerRewardStats, RolloutLedgerStep, RolloutLedgerWriter,
};
use crate::sample::Sample;
use crate::telemetry::{
    cuda_memory_snapshot, CandidateRecord, CandidateWriter, DecoderCacheSnapshot,
    GpuMemoryProbeEvent, GpuMemorySnapshot, Metrics, MetricsWriter, ModelTelemetryRecorder, RunDir,
    TelemetryError,
};

/// An error raised while running a GRPO training step.
#[derive(Debug, thiserror::Error)]
pub enum TrainerError {
    /// A candle tensor op, the policy forward, the optimizer, or the
    /// grad-coverage canary failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Writing run telemetry (`config.json` / `metrics.jsonl`) failed.
    #[error(transparent)]
    Telemetry(#[from] TelemetryError),
    /// A [`TrainerConfig`] field is out of range (rejected by
    /// [`TrainerConfig::validate`]).
    #[error("invalid trainer config: {0}")]
    InvalidConfig(String),
    /// A [`Policy`] or [`RewardFn`] returned output that violates the trainer's
    /// contract (a malformed rollout, or a reward count that does not match the
    /// number of completions).
    #[error("contract violation: {0}")]
    Contract(String),
    /// A [`RewardFn`] could not compute a finite reward — its (possibly external)
    /// verifier failed or returned a non-finite value.
    #[error(transparent)]
    Reward(#[from] RewardError),
    /// Writing a periodic adapter checkpoint failed.
    #[error(transparent)]
    Checkpoint(#[from] crate::checkpoint::CheckpointError),
    /// A data- or tensor-parallel collective failed (contribution mismatch, a
    /// peer rank timing out, or a poisoned world) — see [`crate::comm`].
    #[error(transparent)]
    Comm(#[from] crate::comm::CommError),
    /// An opaque tensor-parallel policy hook either observed a collective
    /// failure or panicked inside its lockstep collective region. The trainer
    /// must not probe the communicator with a later status rendezvous.
    #[error(
        "tensor-parallel {operation} entered a terminal distributed state: {detail}; no further collectives are safe; discard the tensor-parallel communicator and policy instance"
    )]
    TensorParallelExecutionTerminal {
        /// Policy hook that failed.
        operation: &'static str,
        /// Preserved communication error or panic diagnostic.
        detail: String,
    },
    /// A reader-visible publication may exist, but the status collective failed,
    /// so callers must preserve publication-side state and discard the dead world.
    #[error(
        "{artifact} publication may be visible at {path}; {detail}; communication failed ({communication}); the distributed execution world is dead and no further collectives are safe; discard the communicator"
    )]
    PublicationAmbiguousAfterComm {
        /// Human-readable artifact kind (continuation or rollout ledger).
        artifact: &'static str,
        /// Reader-visible destination whose publication is uncertain.
        path: PathBuf,
        /// Publication boundary crossed before communication failed.
        detail: String,
        /// Terminal collective failure that killed the data-parallel world.
        #[source]
        communication: Box<crate::comm::CommError>,
    },
    /// Metrics publication lost its status collective after the local append;
    /// the learner envelope must combine telemetry and model rollback outcomes.
    #[error(
        "rollout-ledger metrics append-status communication failed ({communication}); {telemetry_rollback}"
    )]
    RolloutLedgerMetricsComm {
        /// Terminal collective failure that killed the data-parallel world.
        #[source]
        communication: Box<crate::comm::CommError>,
        /// Exact success/failure detail from panic-contained local truncation.
        telemetry_rollback: String,
    },
    /// A separated rollout/learner ledger could not be published or validated.
    #[error(transparent)]
    RolloutLedger(#[from] RolloutLedgerError),
}

/// Learner-produced, chain-bound state for the next separated rollout step.
///
/// The optimizer is intentionally carried inside this opaque receipt: callers
/// may inspect it for orchestration, but only a successful ledger consumption or
/// strict continuation restore can construct a value accepted by subsequent
/// collector/learner/save calls. This prevents publishing mixed adapter, Adam,
/// sampler, or outer-step state as a completed continuation.
#[derive(Debug)]
pub struct RolloutLedgerContinuation {
    completed_step: u64,
    world_size: u32,
    tensor_parallel_world_size: u32,
    tensor_parallel_layout: String,
    optimizer_state: OptimizerState,
    policy_sha256: String,
    trainer_config_sha256: String,
    tensor_schema_sha256: String,
    adapter_sha256: String,
    optimizer_sha256: String,
    sampler_sha256: String,
    parent_lineage_sha256: String,
    consumed_ledger_sha256: String,
    lineage_sha256: String,
}

impl RolloutLedgerContinuation {
    /// Number of completed outer ledger steps; this is the next source step.
    #[must_use]
    pub fn completed_step(&self) -> u64 {
        self.completed_step
    }

    /// Data-parallel topology bound to this continuation.
    #[must_use]
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Tensor-parallel topology bound to this continuation.
    #[must_use]
    pub fn tensor_parallel_world_size(&self) -> u32 {
        self.tensor_parallel_world_size
    }

    /// Canonical ordering used to bind TP communicator ranks to model shards.
    #[must_use]
    pub fn tensor_parallel_layout(&self) -> &str {
        &self.tensor_parallel_layout
    }

    /// Exact chain lineage represented by the post-step continuation.
    #[must_use]
    pub fn lineage_sha256(&self) -> &str {
        &self.lineage_sha256
    }

    /// Borrow the exact Adam continuation for diagnostics or orchestration.
    #[must_use]
    pub fn optimizer_state(&self) -> &OptimizerState {
        &self.optimizer_state
    }
}

/// Bridges prompt / completion text and token ids for the trainer.
///
/// The trainer encodes each prompt to ids for [`Policy::generate`] and decodes
/// completion ids back to text for [`RewardFn::reward`]; neither the policy nor
/// the reward owns this mapping. The toy uses a trivial char codec; a real model
/// wraps its `tokenizers::Tokenizer`. Implementations are expected to be total
/// over valid vocab ids — a real wrapper handles its own decode failures (e.g.
/// lossy decoding) rather than surfacing them here.
pub trait TokenizerLike {
    /// Encode `text` into the token ids fed to [`Policy::generate`].
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Decode token `ids` back into text scored by [`RewardFn`].
    fn decode(&self, ids: &[u32]) -> String;
}

/// Where a prompt's reward-normalization group is formed.
///
/// `Local` is the classic GRPO behavior: each rank samples and normalizes a full
/// group independently. `DistributedSamePrompt` is the memory-saving distributed
/// mode: for each accumulation position, every rank samples the same prompt with
/// rank-distinct rollout RNG row bases, then all-reduces the group's finite
/// reward statistics before computing advantages. This is a lockstep
/// same-position grouping contract; future non-lockstep or packed distributed
/// batches need explicit group-key-aware aggregation, not silent mixing across
/// unrelated prompts or tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RewardGroupScope {
    /// Normalize rewards within each local rank's sampled group.
    #[default]
    Local,
    /// Treat every rank's local completions as the same prompt's reward group.
    DistributedSamePrompt,
}

/// One point in a deterministic scalar schedule.
///
/// Schedules are evaluated over 0-based optimizer steps. The first point must be
/// at step `0`, later points must have strictly increasing steps, and the value
/// is linearly interpolated between neighboring points. After the final point the
/// value is held constant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulePoint {
    /// Optimizer step where this point's value is exact.
    pub step: u64,
    /// Scalar value at [`step`](Self::step).
    pub value: f64,
}

/// Deterministic piecewise-linear scalar schedule for trainer-owned knobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarSchedule {
    /// Piecewise-linear control points. See [`SchedulePoint`] for validation.
    pub points: Vec<SchedulePoint>,
}

impl ScalarSchedule {
    /// A one-point schedule that always returns `value`.
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self {
            points: vec![SchedulePoint { step: 0, value }],
        }
    }

    /// A linear schedule from `start` at step `0` to `end` at `last_step`, then
    /// held constant.
    #[must_use]
    pub fn linear(start: f64, end: f64, last_step: u64) -> Self {
        if last_step == 0 {
            return Self::constant(end);
        }
        Self {
            points: vec![
                SchedulePoint {
                    step: 0,
                    value: start,
                },
                SchedulePoint {
                    step: last_step,
                    value: end,
                },
            ],
        }
    }

    /// Value at 0-based optimizer `step`.
    ///
    /// # Panics
    ///
    /// Panics if called before validation on an empty schedule.
    #[must_use]
    pub fn at(&self, step: u64) -> f64 {
        let first = self
            .points
            .first()
            .expect("schedule has at least one point");
        if step <= first.step {
            return first.value;
        }
        for pair in self.points.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if step <= b.step {
                let span = (b.step - a.step) as f64;
                let t = (step - a.step) as f64 / span;
                return a.value + (b.value - a.value) * t;
            }
        }
        self.points
            .last()
            .expect("schedule has at least one point")
            .value
    }

    fn validate_nonnegative(&self, label: &str, trainer_steps: u64) -> Result<(), TrainerError> {
        require(
            !self.points.is_empty(),
            &format!("{label}.points must not be empty"),
        )?;
        require(
            trainer_steps > 0,
            &format!("{label} requires trainer.steps >= 1"),
        )?;
        require(
            self.points[0].step == 0,
            &format!("{label}.points[0].step must be 0"),
        )?;
        let mut prev = None;
        for (idx, point) in self.points.iter().enumerate() {
            require(
                point.step < trainer_steps,
                &format!("{label}.points[{idx}].step must be < trainer.steps"),
            )?;
            require(
                point.value.is_finite() && point.value >= 0.0,
                &format!("{label}.points[{idx}].value must be finite and >= 0"),
            )?;
            if let Some(prev) = prev {
                require(
                    point.step > prev,
                    &format!("{label}.points must be strictly increasing by step"),
                )?;
            }
            prev = Some(point.step);
        }
        Ok(())
    }

    fn has_positive_value(&self) -> bool {
        self.points.iter().any(|point| point.value > 0.0)
    }
}

/// Configuration for a GRPO training run.
///
/// Serializable so [`RunDir::write_config`] persists it to `config.json`. The
/// defaults are the MVP setting: a single inner step (`mu = 1`, so the
/// importance ratio is wired but inert) and no KL penalty (`beta = 0`, so the
/// reference policy is never computed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Number of optimizer (outer GRPO) steps to run.
    pub steps: u64,
    /// GRPO group size `G` — completions sampled per prompt.
    pub group_size: usize,
    /// Maximum new tokens to generate per completion.
    pub max_new_tokens: usize,
    /// Rollout sampling temperature, threaded into [`GenConfig::temperature`].
    /// Scoring is **temperature-consistent**: an [`crate::LmPolicy`] divides its
    /// scoring logits by its own baked temperature (TRL parity) and fails loud
    /// if this value disagrees with it — build the policy with the same
    /// temperature configured here.
    pub temperature: f64,
    /// Inner optimization steps per rollout. At `1` the importance ratio is
    /// exactly `1` (current log-probs equal the frozen snapshot), so the clip is
    /// wired but inert.
    pub mu: usize,
    /// Legacy constant KL penalty coefficient. The reference (adapter-disabled)
    /// log-probs and k3 KL term are computed only when the effective beta for a
    /// step is `> 0`.
    pub beta: f64,
    /// Optional deterministic piecewise-linear KL schedule. When set, this owns
    /// the effective per-step beta; [`beta`](Self::beta) remains the legacy
    /// constant default used when this field is `None`.
    #[serde(default)]
    pub beta_schedule: Option<ScalarSchedule>,
    /// PPO clip half-width `epsilon` (e.g. `0.2`) — the **lower** band, and the
    /// upper band too unless [`clip_eps_high`](Self::clip_eps_high) overrides it.
    pub clip_eps: f64,
    /// Optional **upper** clip half-width (DAPO clip-higher, e.g. `0.28` —
    /// the standard entropy-control mechanism; TRL's `epsilon_high`). `None`
    /// (the default) keeps the clip symmetric at [`clip_eps`](Self::clip_eps),
    /// bit-identical to the single-knob behavior. `#[serde(default)]` so an
    /// older `config.json` still deserializes.
    #[serde(default)]
    pub clip_eps_high: Option<f64>,
    /// At which level the importance ratio is formed: per token (classic GRPO,
    /// the default) or per sequence (GSPO — see
    /// [`ImportanceSamplingLevel::Sequence`]). `#[serde(default)]` so an older
    /// `config.json` still deserializes (to `Token`).
    #[serde(default)]
    pub importance_sampling_level: ImportanceSamplingLevel,
    /// `AdamW` learning rate (the post-warmup constant — see
    /// [`warmup_steps`](Self::warmup_steps)).
    pub lr: f64,
    /// Optional deterministic piecewise-linear learning-rate schedule. When set,
    /// this replaces [`lr`](Self::lr) and [`warmup_steps`](Self::warmup_steps);
    /// encode warmup directly as schedule points.
    #[serde(default)]
    pub lr_schedule: Option<ScalarSchedule>,
    /// `AdamW` weight decay. Defaults to `0` (the toy trains pure policy gradient).
    pub weight_decay: f64,
    /// `AdamW` `β₁` (first-moment decay). Defaults to `0.9` — candle / TRL's
    /// default; `(0.9, 0.95)` is the common "modern" alternative. `#[serde(default)]`
    /// via `default_adam_beta1` so an older `config.json` still deserializes.
    #[serde(default = "default_adam_beta1")]
    pub adam_beta1: f64,
    /// `AdamW` `β₂` (second-moment decay). Defaults to `0.999`.
    /// `#[serde(default)]` via `default_adam_beta2`.
    #[serde(default = "default_adam_beta2")]
    pub adam_beta2: f64,
    /// Linear learning-rate warmup over the first `warmup_steps` optimizer
    /// steps, then constant at [`lr`](Self::lr) (the RL convention — DAPO warms
    /// up over ~20 rollout steps; cosine decay is deliberately not offered).
    /// Step `t` (0-based) runs at `lr · min(1, (t+1) / warmup_steps)`, so the
    /// first step uses `lr / warmup_steps`, never `0`. `0` disables warmup
    /// (every step at full `lr`, bit-identical to the pre-warmup trainer). The
    /// schedule is a pure function of the step index, so a checkpoint resume
    /// re-enters it faithfully. `#[serde(default)]` so an older `config.json`
    /// still deserializes (to `0` — those runs predate warmup).
    #[serde(default)]
    pub warmup_steps: u64,
    /// Global-norm gradient clipping (the settled standard, verl/TRL: `1.0`):
    /// when the window's accumulated gradient norm exceeds this, every
    /// gradient is scaled by `max_grad_norm / norm` before the optimizer step.
    /// `None` disables clipping. Defaults to `Some(1.0)` **including for older
    /// `config.json` files** (`#[serde(default)]` via `default_max_grad_norm`)
    /// — clipping is a safety net, not a behavior an old config opted out of.
    /// [`Metrics::grad_norm`] stays the **pre-clip** norm.
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: Option<f64>,
    /// DAPO overlong filtering (TRL's `mask_truncated_completions`): when `true`
    /// (the default, maintainer-confirmed for the ladder runs), a completion that
    /// ran to the full `max_new_tokens` width **without sampling EOS** has its
    /// loss-mask row zeroed — a truncated answer's verifier reward is
    /// semantically wrong in both directions, so its tokens carry no gradient.
    /// Matching TRL, the truncated completion still participates in the group's
    /// reward statistics / advantages and still counts in the DAPO loss
    /// normalizer; masked rows are counted in [`Metrics::dropped_rows`] and
    /// reported as [`Metrics::frac_truncated`]. **Inert when
    /// [`eos_token_id`](Self::eos_token_id) is `None`** (without an EOS no
    /// completion can terminate, so masking would zero every row).
    /// `#[serde(default)]` via `default_truncation_masking` (`true`) —
    /// **including for older `config.json` files** (like clipping, this is a
    /// correctness guard, not a behavior an old config opted out of; a pre-R1
    /// run with an EOS token resumes with masking active). Detection assumes
    /// the single-EOS rollout contract ([`crate::policy::GenConfig`]): TRL
    /// additionally treats a trailing *pad* token as terminated, a state
    /// ferrl's EOS-padded rollouts cannot produce.
    #[serde(default = "default_truncation_masking")]
    pub truncation_masking: bool,
    /// **Truncated importance sampling** (TIS) — correct the train/rollout
    /// off-policy mismatch by multiplying each token's surrogate by the
    /// detached weight `min(exp(logp_train − logp_rollout), tis_imp_ratio_cap)`
    /// (see [`crate::grpo::tis_weight`]; the KL penalty stays unweighted,
    /// matching verl). `logp_rollout` is the behavior log-prob the policy
    /// captured at draw time
    /// ([`Rollout::rollout_logprobs`](crate::policy::Rollout::rollout_logprobs));
    /// a policy that captures none **fails loud** when this is on, rather than
    /// silently training uncorrected. Default **`false`** — the metrics-first
    /// posture: the rollout-ratio telemetry is always reported when capture is
    /// available, and the correction is flipped on when the observed gap
    /// warrants it (it matters most when rollout and scoring run different
    /// numerics, e.g. a bf16 merged decode against the f32 scoring forward on
    /// a days-long run). Read the **gap** off
    /// [`Metrics::rollout_logratio_mean`] (a KL-style drift meter),
    /// [`Metrics::rollout_ratio_max`], and
    /// [`Metrics::frac_rollout_ratio_capped`] — NOT off
    /// [`Metrics::rollout_ratio_mean`], whose expectation is exactly `1` under
    /// arbitrary drift (see its docs) — and check
    /// [`Metrics::rollout_capture_tokens`] `> 0` first (the telemetry can be
    /// dark).
    /// Token-level only: rejected together with
    /// [`ImportanceSamplingLevel::Sequence`] (GSPO forms its ratio per
    /// sequence; mixing the two corrections is unstudied — pick one).
    /// `#[serde(default)]` (`false`) so an older `config.json` still
    /// deserializes.
    #[serde(default)]
    pub tis: bool,
    /// The TIS truncation cap `C` (verl's `tis_imp_ratio_cap`; `C ≈ 2` is the
    /// studied setting). Also the threshold the
    /// [`Metrics::frac_rollout_ratio_capped`] telemetry counts against **even
    /// while [`tis`](Self::tis) is off** — the "how often would the correction
    /// bind" signal that motivates flipping it on. Must be finite and `>= 1`
    /// (a cap below `1` would down-weight exactly on-policy tokens).
    /// `#[serde(default)]` via `default_tis_imp_ratio_cap` (`2.0`) so an older
    /// `config.json` still deserializes.
    #[serde(default = "default_tis_imp_ratio_cap")]
    pub tis_imp_ratio_cap: f64,
    /// Which masked reduction to apply to the per-token objective.
    pub loss_type: LossType,
    /// How to scale group-centered rewards into advantages.
    pub scale_rewards: ScaleRewards,
    /// Where reward-normalization groups are formed. Defaults to local GRPO
    /// behavior for old configs and single-rank runs. The distributed option
    /// samples the same logical prompt at each accumulation position across ranks,
    /// while keeping rollout rows rank-sharded so ranks draw distinct completions.
    /// Lockstep multi-prompt accumulation is allowed; arbitrary packed or
    /// non-lockstep distributed batches still need group-key-aware aggregation.
    #[serde(default)]
    pub reward_group_scope: RewardGroupScope,
    /// Number of prompts whose gradients are accumulated into a single optimizer
    /// step (gradient accumulation **across prompts**). Each [`steps`](Self::steps)
    /// outer step consumes this many prompts — a *window* — summing their per-prompt
    /// group gradients (each scaled by `1 / grad_accum_steps`) before one `AdamW`
    /// update, giving an effective batch of `grad_accum_steps` groups at a single
    /// group's peak memory (only one group's grad forward is held at a time). A
    /// degenerate (all-equal-reward) prompt in a window contributes no gradient but
    /// still counts toward the `1 / grad_accum_steps` scale, and a window is skipped
    /// only when *every* prompt in it is degenerate. `1` (the default) is plain
    /// per-prompt stepping, bit-identical to no accumulation. `#[serde(default)]` via
    /// the `default_grad_accum_steps` fn so an older `config.json` still deserializes.
    #[serde(default = "default_grad_accum_steps")]
    pub grad_accum_steps: usize,
    /// Optional microbatch size for the update backward within a single reward
    /// group. `0` (the default) keeps the existing full-group backward. A
    /// positive value splits a live group's rows into chunks of at most this
    /// size, accumulates their trainable-var gradients, and then hands the same
    /// logical group gradient to the outer prompt/window accumulator. This trades
    /// extra forwards for a lower activation peak on long-completion runs.
    #[serde(default)]
    pub backward_microbatch_size: usize,
    /// If set, write an adapter checkpoint to `checkpoints/step-<n>/` (with a
    /// resumable manifest) every `checkpoint_every` completed steps **and** after
    /// the final step (so a completed run always persists its final adapter, even
    /// when this does not divide `steps`). `None` (the default) disables
    /// checkpointing entirely. `#[serde(default)]` so a `config.json` written before
    /// this field existed still deserializes (to `None`).
    #[serde(default)]
    pub checkpoint_every: Option<u64>,
    /// Persist the top-K sampled completions per prompt group to
    /// `candidates.jsonl`, ordered by scalar reward. `0` disables the candidate
    /// ledger. This is intentionally opt-in because completions can be large; it
    /// is required for discovery runs that need to feed raw candidates to an
    /// artifact extractor after training.
    #[serde(default)]
    pub candidate_log_top_k: usize,
    /// Opt-in CUDA memory telemetry for GPU runs.
    ///
    /// When `true`, the trainer samples CUDA runtime free/total memory at coarse
    /// phase boundaries (rollout, reward, detached scoring, backward, gradient
    /// reduction, optimizer), persists per-step start/peak/end memory fields, and
    /// records detailed `cuda_mem_probe_events` plus model-provided
    /// `decoder_cache_snapshots` in `metrics.jsonl`. It is intentionally off by
    /// default: the default CPU build records zeros/empty lists, and ordinary
    /// training avoids runtime-query overhead.
    #[serde(default)]
    pub gpu_memory_probe: bool,
    /// End-of-sequence token id threaded into [`GenConfig::eos_token_id`] for
    /// rollout: when `Some`, a sampled EOS ends a completion early (EOS-inclusive)
    /// and the row is right-padded to `max_new_tokens`, the true length recorded in
    /// [`Rollout::completion_lens`](crate::policy::Rollout::completion_lens). The loss
    /// mask zeroes the padded positions and the reward decode stops at the true
    /// length, so the EOS padding never enters the objective or the reward. `None`
    /// (the default) keeps full-width rollouts, bit-identical to before.
    /// `#[serde(default)]` so a `config.json` written before this field existed still
    /// deserializes (to `None`).
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

/// The stable, learner-semantic projection bound into a rollout-ledger identity.
///
/// Keeping this projection typed and exhaustive makes every future `TrainerConfig`
/// field choose explicitly between learner semantics and operational orchestration.
/// The four omitted fields (`steps`, checkpoint cadence, candidate logging, and GPU
/// probing) do not change one validated learner update.
#[derive(Serialize)]
struct RolloutLedgerTrainerSemantics<'a> {
    group_size: &'a usize,
    max_new_tokens: &'a usize,
    temperature: &'a f64,
    mu: &'a usize,
    beta: &'a f64,
    beta_schedule: &'a Option<ScalarSchedule>,
    clip_eps: &'a f64,
    clip_eps_high: &'a Option<f64>,
    importance_sampling_level: &'a ImportanceSamplingLevel,
    lr: &'a f64,
    lr_schedule: &'a Option<ScalarSchedule>,
    weight_decay: &'a f64,
    adam_beta1: &'a f64,
    adam_beta2: &'a f64,
    warmup_steps: &'a u64,
    max_grad_norm: &'a Option<f64>,
    truncation_masking: &'a bool,
    tis: &'a bool,
    tis_imp_ratio_cap: &'a f64,
    loss_type: &'a LossType,
    scale_rewards: &'a ScaleRewards,
    reward_group_scope: &'a RewardGroupScope,
    grad_accum_steps: &'a usize,
    backward_microbatch_size: &'a usize,
    eos_token_id: &'a Option<u32>,
}

/// `serde` default for [`TrainerConfig::grad_accum_steps`]: `1` (no accumulation).
fn default_grad_accum_steps() -> usize {
    1
}

/// `serde` default for [`TrainerConfig::adam_beta1`]: `0.9`.
fn default_adam_beta1() -> f64 {
    0.9
}

/// `serde` default for [`TrainerConfig::adam_beta2`]: `0.999`.
fn default_adam_beta2() -> f64 {
    0.999
}

/// `serde` default for [`TrainerConfig::max_grad_norm`]: `Some(1.0)` (the
/// settled industry standard — applied to old configs too; see the field docs).
fn default_max_grad_norm() -> Option<f64> {
    Some(1.0)
}

/// `serde` default for [`TrainerConfig::truncation_masking`]: `true`
/// (maintainer-confirmed default-ON; inert without an EOS token).
fn default_truncation_masking() -> bool {
    true
}

/// `serde` default for [`TrainerConfig::tis_imp_ratio_cap`]: `2.0` (the studied
/// verl setting; drives the capped-fraction telemetry even with `tis` off).
fn default_tis_imp_ratio_cap() -> f64 {
    2.0
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            group_size: 8,
            max_new_tokens: 16,
            temperature: 1.0,
            mu: 1,
            beta: 0.0,
            beta_schedule: None,
            clip_eps: 0.2,
            clip_eps_high: None,
            importance_sampling_level: ImportanceSamplingLevel::Token,
            lr: 1e-3,
            lr_schedule: None,
            weight_decay: 0.0,
            adam_beta1: default_adam_beta1(),
            adam_beta2: default_adam_beta2(),
            warmup_steps: 0,
            max_grad_norm: default_max_grad_norm(),
            truncation_masking: default_truncation_masking(),
            tis: false,
            tis_imp_ratio_cap: default_tis_imp_ratio_cap(),
            loss_type: LossType::Dapo,
            scale_rewards: ScaleRewards::Group,
            reward_group_scope: RewardGroupScope::Local,
            grad_accum_steps: 1,
            backward_microbatch_size: 0,
            checkpoint_every: None,
            candidate_log_top_k: 0,
            gpu_memory_probe: false,
            eos_token_id: None,
        }
    }
}

impl From<&TrainerConfig> for GenConfig {
    /// Derive the rollout [`GenConfig`] from a [`TrainerConfig`], so the two cannot
    /// drift: group size, completion width, sampling temperature, and the EOS id
    /// all flow from the single trainer config. `eval_sampling` is always `None` —
    /// training rolls out at the configured temperature; the held-out eval harness
    /// sets its own [`EvalSampling`](crate::policy::EvalSampling) override.
    fn from(config: &TrainerConfig) -> Self {
        Self {
            group_size: config.group_size,
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            eos_token_id: config.eos_token_id,
            eval_sampling: None,
        }
    }
}

/// A fluent builder for [`TrainerConfig`].
///
/// Seeded with [`TrainerConfig::default`]; each setter overrides one field and
/// returns `self`, so only the fields you name change. [`build`](Self::build)
/// returns a [`TrainerConfig`] **identical to the equivalent struct literal**
/// (`TrainerConfig { ..overrides, ..Default::default() }`). It does not validate:
/// an out-of-range value surfaces at [`Trainer::new`], which calls
/// [`TrainerConfig::validate`].
///
/// ```
/// use ferrl::TrainerConfig;
///
/// let cfg = TrainerConfig::builder()
///     .steps(200)
///     .group_size(16)
///     .lr(1e-6)
///     .beta(0.04)
///     .build();
/// assert_eq!(cfg.steps, 200);
/// assert_eq!(cfg.group_size, 16);
/// // Unset fields keep their defaults.
/// assert_eq!(cfg.mu, TrainerConfig::default().mu);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TrainerConfigBuilder {
    cfg: TrainerConfig,
}

impl TrainerConfigBuilder {
    /// A builder seeded with [`TrainerConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`TrainerConfig::steps`].
    #[must_use]
    pub fn steps(mut self, steps: u64) -> Self {
        self.cfg.steps = steps;
        self
    }

    /// Set [`TrainerConfig::group_size`].
    #[must_use]
    pub fn group_size(mut self, group_size: usize) -> Self {
        self.cfg.group_size = group_size;
        self
    }

    /// Set [`TrainerConfig::max_new_tokens`].
    #[must_use]
    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.cfg.max_new_tokens = max_new_tokens;
        self
    }

    /// Set [`TrainerConfig::temperature`].
    #[must_use]
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.cfg.temperature = temperature;
        self
    }

    /// Set [`TrainerConfig::mu`].
    #[must_use]
    pub fn mu(mut self, mu: usize) -> Self {
        self.cfg.mu = mu;
        self
    }

    /// Set [`TrainerConfig::beta`].
    #[must_use]
    pub fn beta(mut self, beta: f64) -> Self {
        self.cfg.beta = beta;
        self
    }

    /// Set [`TrainerConfig::beta_schedule`].
    #[must_use]
    pub fn beta_schedule(mut self, beta_schedule: Option<ScalarSchedule>) -> Self {
        self.cfg.beta_schedule = beta_schedule;
        self
    }

    /// Set [`TrainerConfig::clip_eps`].
    #[must_use]
    pub fn clip_eps(mut self, clip_eps: f64) -> Self {
        self.cfg.clip_eps = clip_eps;
        self
    }

    /// Set [`TrainerConfig::clip_eps_high`].
    #[must_use]
    pub fn clip_eps_high(mut self, clip_eps_high: Option<f64>) -> Self {
        self.cfg.clip_eps_high = clip_eps_high;
        self
    }

    /// Set [`TrainerConfig::importance_sampling_level`].
    #[must_use]
    pub fn importance_sampling_level(mut self, level: ImportanceSamplingLevel) -> Self {
        self.cfg.importance_sampling_level = level;
        self
    }

    /// Set [`TrainerConfig::lr`].
    #[must_use]
    pub fn lr(mut self, lr: f64) -> Self {
        self.cfg.lr = lr;
        self
    }

    /// Set [`TrainerConfig::lr_schedule`].
    #[must_use]
    pub fn lr_schedule(mut self, lr_schedule: Option<ScalarSchedule>) -> Self {
        self.cfg.lr_schedule = lr_schedule;
        self
    }

    /// Set [`TrainerConfig::weight_decay`].
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.cfg.weight_decay = weight_decay;
        self
    }

    /// Set [`TrainerConfig::adam_beta1`].
    #[must_use]
    pub fn adam_beta1(mut self, adam_beta1: f64) -> Self {
        self.cfg.adam_beta1 = adam_beta1;
        self
    }

    /// Set [`TrainerConfig::adam_beta2`].
    #[must_use]
    pub fn adam_beta2(mut self, adam_beta2: f64) -> Self {
        self.cfg.adam_beta2 = adam_beta2;
        self
    }

    /// Set [`TrainerConfig::warmup_steps`].
    #[must_use]
    pub fn warmup_steps(mut self, warmup_steps: u64) -> Self {
        self.cfg.warmup_steps = warmup_steps;
        self
    }

    /// Set [`TrainerConfig::max_grad_norm`].
    #[must_use]
    pub fn max_grad_norm(mut self, max_grad_norm: Option<f64>) -> Self {
        self.cfg.max_grad_norm = max_grad_norm;
        self
    }

    /// Set [`TrainerConfig::truncation_masking`].
    #[must_use]
    pub fn truncation_masking(mut self, truncation_masking: bool) -> Self {
        self.cfg.truncation_masking = truncation_masking;
        self
    }

    /// Set [`TrainerConfig::tis`].
    #[must_use]
    pub fn tis(mut self, tis: bool) -> Self {
        self.cfg.tis = tis;
        self
    }

    /// Set [`TrainerConfig::tis_imp_ratio_cap`].
    #[must_use]
    pub fn tis_imp_ratio_cap(mut self, tis_imp_ratio_cap: f64) -> Self {
        self.cfg.tis_imp_ratio_cap = tis_imp_ratio_cap;
        self
    }

    /// Set [`TrainerConfig::loss_type`].
    #[must_use]
    pub fn loss_type(mut self, loss_type: LossType) -> Self {
        self.cfg.loss_type = loss_type;
        self
    }

    /// Set [`TrainerConfig::scale_rewards`].
    #[must_use]
    pub fn scale_rewards(mut self, scale_rewards: ScaleRewards) -> Self {
        self.cfg.scale_rewards = scale_rewards;
        self
    }

    /// Set [`TrainerConfig::reward_group_scope`].
    #[must_use]
    pub fn reward_group_scope(mut self, reward_group_scope: RewardGroupScope) -> Self {
        self.cfg.reward_group_scope = reward_group_scope;
        self
    }

    /// Set [`TrainerConfig::grad_accum_steps`].
    #[must_use]
    pub fn grad_accum_steps(mut self, grad_accum_steps: usize) -> Self {
        self.cfg.grad_accum_steps = grad_accum_steps;
        self
    }

    /// Set [`TrainerConfig::backward_microbatch_size`].
    #[must_use]
    pub fn backward_microbatch_size(mut self, backward_microbatch_size: usize) -> Self {
        self.cfg.backward_microbatch_size = backward_microbatch_size;
        self
    }

    /// Set [`TrainerConfig::checkpoint_every`].
    #[must_use]
    pub fn checkpoint_every(mut self, checkpoint_every: Option<u64>) -> Self {
        self.cfg.checkpoint_every = checkpoint_every;
        self
    }

    /// Set [`TrainerConfig::candidate_log_top_k`].
    #[must_use]
    pub fn candidate_log_top_k(mut self, candidate_log_top_k: usize) -> Self {
        self.cfg.candidate_log_top_k = candidate_log_top_k;
        self
    }

    /// Set [`TrainerConfig::gpu_memory_probe`].
    #[must_use]
    pub fn gpu_memory_probe(mut self, gpu_memory_probe: bool) -> Self {
        self.cfg.gpu_memory_probe = gpu_memory_probe;
        self
    }

    /// Set [`TrainerConfig::eos_token_id`].
    #[must_use]
    pub fn eos_token_id(mut self, eos_token_id: Option<u32>) -> Self {
        self.cfg.eos_token_id = eos_token_id;
        self
    }

    /// Finish building, returning the configured [`TrainerConfig`].
    ///
    /// Does not validate (see the type docs); [`Trainer::new`] validates.
    #[must_use]
    pub fn build(self) -> TrainerConfig {
        self.cfg
    }
}

impl TrainerConfig {
    /// Start a fluent [`TrainerConfigBuilder`] seeded with the defaults.
    #[must_use]
    pub fn builder() -> TrainerConfigBuilder {
        TrainerConfigBuilder::new()
    }

    fn rollout_ledger_semantics(&self) -> RolloutLedgerTrainerSemantics<'_> {
        let TrainerConfig {
            steps: _,
            group_size,
            max_new_tokens,
            temperature,
            mu,
            beta,
            beta_schedule,
            clip_eps,
            clip_eps_high,
            importance_sampling_level,
            lr,
            lr_schedule,
            weight_decay,
            adam_beta1,
            adam_beta2,
            warmup_steps,
            max_grad_norm,
            truncation_masking,
            tis,
            tis_imp_ratio_cap,
            loss_type,
            scale_rewards,
            reward_group_scope,
            grad_accum_steps,
            backward_microbatch_size,
            checkpoint_every: _,
            candidate_log_top_k: _,
            gpu_memory_probe: _,
            eos_token_id,
        } = self;
        RolloutLedgerTrainerSemantics {
            group_size,
            max_new_tokens,
            temperature,
            mu,
            beta,
            beta_schedule,
            clip_eps,
            clip_eps_high,
            importance_sampling_level,
            lr,
            lr_schedule,
            weight_decay,
            adam_beta1,
            adam_beta2,
            warmup_steps,
            max_grad_norm,
            truncation_masking,
            tis,
            tis_imp_ratio_cap,
            loss_type,
            scale_rewards,
            reward_group_scope,
            grad_accum_steps,
            backward_microbatch_size,
            eos_token_id,
        }
    }

    /// Reject settings that would silently do nothing or crash mid-run.
    ///
    /// In particular `mu = 0` would run **no** inner update — no backward, no
    /// canary, no optimizer step — while still emitting a metrics row, a silent
    /// no-op exactly counter to the canary's fail-loud philosophy. Called by
    /// [`Trainer::new`] before the config is persisted.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidConfig`] if `mu`, `group_size`,
    /// `max_new_tokens`, or `grad_accum_steps` is `0`; if `temperature` is not
    /// finite and `> 0`; if `lr`,
    /// `weight_decay`, or `beta` is not finite and `>= 0`; if `clip_eps` is not
    /// finite and in `[0, 1)` (`>= 1` makes the lower clip band cross `0`, which a
    /// strictly-positive importance ratio can never reach, silently disabling the
    /// lower clip); if `clip_eps_high` is `Some` but not finite and `>= 0` (only
    /// the **lower** band has the `< 1` constraint); if `adam_beta1` /
    /// `adam_beta2` is not in `[0, 1)`; if `max_grad_norm` is `Some` but not
    /// finite and `> 0` (`0` would zero every gradient — use `None` to disable
    /// clipping); if `checkpoint_every` is `Some(0)`; if `tis_imp_ratio_cap` is
    /// not finite and `>= 1`; or if `tis` is combined with
    /// [`ImportanceSamplingLevel::Sequence`] (TIS is token-level).
    pub fn validate(&self) -> Result<(), TrainerError> {
        require(
            self.mu >= 1,
            "mu must be >= 1 (mu = 0 runs no inner update)",
        )?;
        require(self.group_size >= 1, "group_size must be >= 1")?;
        require(self.max_new_tokens >= 1, "max_new_tokens must be >= 1")?;
        require(
            self.grad_accum_steps >= 1,
            "grad_accum_steps must be >= 1 (0 would never step the optimizer)",
        )?;
        require(
            self.temperature.is_finite() && self.temperature > 0.0,
            "temperature must be finite and > 0",
        )?;
        require(
            self.lr.is_finite() && self.lr >= 0.0,
            "lr must be finite and >= 0",
        )?;
        require(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "weight_decay must be finite and >= 0",
        )?;
        require(
            self.adam_beta1.is_finite() && (0.0..1.0).contains(&self.adam_beta1),
            "adam_beta1 must be finite and in [0, 1)",
        )?;
        require(
            self.adam_beta2.is_finite() && (0.0..1.0).contains(&self.adam_beta2),
            "adam_beta2 must be finite and in [0, 1)",
        )?;
        require(
            self.beta.is_finite() && self.beta >= 0.0,
            "beta must be finite and >= 0",
        )?;
        if let Some(schedule) = &self.beta_schedule {
            schedule.validate_nonnegative("beta_schedule", self.steps)?;
        }
        require(
            self.clip_eps.is_finite() && self.clip_eps >= 0.0 && self.clip_eps < 1.0,
            "clip_eps must be finite and in [0, 1) (>= 1 disables the lower clip)",
        )?;
        if let Some(schedule) = &self.lr_schedule {
            schedule.validate_nonnegative("lr_schedule", self.steps)?;
            require(
                self.warmup_steps == 0,
                "lr_schedule cannot be combined with warmup_steps; encode warmup as schedule points",
            )?;
        }
        if let Some(high) = self.clip_eps_high {
            require(
                high.is_finite() && high >= 0.0,
                "clip_eps_high must be finite and >= 0 when set",
            )?;
        }
        if let Some(max) = self.max_grad_norm {
            require(
                max.is_finite() && max > 0.0,
                "max_grad_norm must be finite and > 0 when set (use None to disable clipping)",
            )?;
        }
        if let Some(every) = self.checkpoint_every {
            require(
                every >= 1,
                "checkpoint_every must be >= 1 when set (0 would checkpoint every step)",
            )?;
        }
        require(
            self.tis_imp_ratio_cap.is_finite() && self.tis_imp_ratio_cap >= 1.0,
            "tis_imp_ratio_cap must be finite and >= 1 (a cap below 1 would down-weight \
             exactly on-policy tokens)",
        )?;
        require(
            !(self.tis && self.importance_sampling_level == ImportanceSamplingLevel::Sequence),
            "tis is a token-level correction and cannot combine with sequence-level \
             importance sampling (GSPO) — pick one",
        )?;
        Ok(())
    }

    /// The effective upper clip half-width: [`clip_eps_high`](Self::clip_eps_high)
    /// when set, else the symmetric [`clip_eps`](Self::clip_eps) (TRL's
    /// `epsilon_high if epsilon_high is not None else epsilon`).
    #[must_use]
    pub fn clip_eps_high_eff(&self) -> f64 {
        self.clip_eps_high.unwrap_or(self.clip_eps)
    }

    /// The effective KL coefficient for 0-based optimizer step `step`.
    #[must_use]
    pub fn beta_at(&self, step: u64) -> f64 {
        self.beta_schedule
            .as_ref()
            .map_or(self.beta, |schedule| schedule.at(step))
    }

    fn requires_reference_policy(&self) -> bool {
        self.beta_schedule
            .as_ref()
            .map_or(self.beta > 0.0, ScalarSchedule::has_positive_value)
    }

    /// The effective learning rate for 0-based optimizer step `step`. When
    /// [`lr_schedule`](Self::lr_schedule) is set, it owns the value; otherwise
    /// this uses the legacy linear warmup over [`warmup_steps`](Self::warmup_steps)
    /// and then the constant [`lr`](Self::lr). A pure function of the step index,
    /// so a resume re-enters the schedule faithfully.
    #[must_use]
    pub fn lr_at(&self, step: u64) -> f64 {
        if let Some(schedule) = &self.lr_schedule {
            return schedule.at(step);
        }
        if self.warmup_steps == 0 || step + 1 >= self.warmup_steps {
            self.lr
        } else {
            self.lr * ((step + 1) as f64 / self.warmup_steps as f64)
        }
    }
}

/// Return [`TrainerError::InvalidConfig`] with `msg` unless `cond` holds.
fn require(cond: bool, msg: &str) -> Result<(), TrainerError> {
    if cond {
        Ok(())
    } else {
        Err(TrainerError::InvalidConfig(msg.to_string()))
    }
}

/// Drives the GRPO training loop over a [`Policy`] and a [`RewardFn`].
#[derive(Debug)]
pub struct Trainer {
    config: TrainerConfig,
    writer: MetricsWriter,
    checkpoints_dir: PathBuf,
    /// The data-parallel collective seam ([`SoloComm`] for a single-rank run).
    /// Every call site is guarded on `world_size() > 1`, so the world-1 path
    /// is byte-for-byte the pre-DP trainer.
    comm: Arc<dyn Comm>,
    /// Optional cooperative preemption flag (see
    /// [`with_preemption_flag`](Self::with_preemption_flag)). When `Some` and set,
    /// the loop writes a final checkpoint and stops at the next step boundary. Safe
    /// to install unevenly across DP ranks — the per-step poll is install-invariant.
    preempt: Option<Arc<AtomicBool>>,
    /// Optional top-candidate ledger for discovery runs.
    candidate_writer: Option<CandidateWriter>,
}

/// Per-inner-step quantities folded into the step's [`Metrics`].
#[derive(Debug, Default, Clone, Copy)]
struct InnerAgg {
    kl: f32,
    clip_frac: f32,
    grad_norm: f32,
}

/// A non-degenerate prompt's data, captured once per accumulation window so the
/// `mu` inner epochs can re-forward it. `logp_old` / `logp_ref` are the detached
/// old / reference snapshots taken at the window's start; `advantages` is the
/// detached `[G, 1]` column and `mask` the `[G, comp_len]` length-aware loss mask
/// (`1` on each sequence's real completion tokens, `0` on its EOS padding).
/// `tis_w` is the detached `[G, comp_len]` TIS weight `min(exp(logp_old −
/// logp_rollout), C)` — present only when the correction is on **and** the
/// rollout captured behavior log-probs (`1.0` at padding positions, which the
/// mask removes anyway).
struct LiveItem {
    rollout: Rollout,
    advantages: Tensor,
    logp_old: Tensor,
    logp_ref: Option<Tensor>,
    mask: Tensor,
    tis_w: Option<Tensor>,
}

/// Host-side collector output for one prompt group. This is the exact boundary
/// between rollout/reward work and learner-owned detached scoring/tensorization.
struct CollectedGroup {
    accum_index: u32,
    prompt_index: u64,
    rollout_global_row_base: u64,
    rollout: Rollout,
    rewards: Vec<f32>,
    advantages: Vec<f64>,
    distributed_reward_stats: Option<RolloutLedgerRewardStats>,
    mask_rows: Vec<Vec<f64>>,
    stat: PromptStat,
    surrogate_live: bool,
}

/// Exact policy state owned by one direct-training rollout group. Under the
/// TP-world-one plus DP topology, detached learner scoring is coordinated over
/// DP only after collection has completed. Keep the pre-group state so a
/// rank-local hook failure cannot leave sampler, adapter mode, or adapter
/// tensors mutated while its peers return from the status rendezvous.
struct RolloutGroupPrestate {
    vars: Vec<Var>,
    adapter: Vec<Tensor>,
    adapter_enabled: bool,
    sampler: Vec<u8>,
}

fn slice_live_item(item: &LiveItem, start: usize, len: usize) -> CandleResult<LiveItem> {
    let end = start + len;
    let rollout = Rollout::new(
        item.rollout.token_ids[start..end].to_vec(),
        item.rollout.prompt_len,
        item.rollout.completion_lens[start..end].to_vec(),
        item.rollout
            .rollout_logprobs
            .as_ref()
            .map(|rows| rows[start..end].to_vec()),
    );
    Ok(LiveItem {
        rollout,
        advantages: item.advantages.narrow(0, start, len)?,
        logp_old: item.logp_old.narrow(0, start, len)?,
        logp_ref: item
            .logp_ref
            .as_ref()
            .map(|t| t.narrow(0, start, len))
            .transpose()?,
        mask: item.mask.narrow(0, start, len)?,
        tis_w: item
            .tis_w
            .as_ref()
            .map(|t| t.narrow(0, start, len))
            .transpose()?,
    })
}

/// Masked-token rollout-ratio aggregates for one live group, computed at collect
/// time from the captured behavior log-probs vs the trainer's own `logp_old`
/// scoring snapshot (the train/rollout off-policy gap). `sum`/`max` are over the
/// ratio `exp(logp_old − logp_rollout)` at the group's loss-carrying tokens;
/// `capped` counts those above the configured TIS cap; `tokens` is the
/// loss-carrying token count the sums normalize by.
struct RatioStats {
    /// Σ ratio over loss tokens (the unit-mean health statistic).
    sum: f64,
    /// Σ (`logp_old` − `logp_rollout`) over loss tokens — the drift
    /// accumulator: its token mean estimates −KL(rollout ‖ train), `0` iff
    /// on-policy.
    log_sum: f64,
    /// Max ratio, kept in f64 so a huge-but-finite outlier is not saturated to
    /// `inf` by an early f32 cast (the metric write still sanitizes).
    max: f64,
    capped: usize,
    tokens: usize,
}

#[derive(Debug, Clone, Copy)]
struct RewardStatsAcc {
    count: f64,
    sum: f64,
    sumsq: f64,
}

struct PromptSelection {
    sample_idx: usize,
    prompt_index: u64,
    rollout_global_row_base: u64,
}

#[derive(Debug, Clone, Copy)]
struct CandidateWriteCtx {
    step: u64,
    prompt_index: u64,
    rank: usize,
    world_size: usize,
    enabled: bool,
}

#[derive(Clone, Copy)]
struct SelectedSample<'a, T> {
    sample: &'a Sample<T>,
    selection: &'a PromptSelection,
    accum_index: usize,
}

#[derive(Debug, Default)]
struct StepGpuMemory {
    enabled: bool,
    start: Option<GpuMemorySnapshot>,
    peak: Option<GpuMemorySnapshot>,
    end: Option<GpuMemorySnapshot>,
    probe_events: Vec<GpuMemoryProbeEvent>,
    decoder_cache_snapshots: Vec<DecoderCacheSnapshot>,
}

impl StepGpuMemory {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            start: None,
            peak: None,
            end: None,
            probe_events: Vec::new(),
            decoder_cache_snapshots: Vec::new(),
        }
    }

    fn record(&mut self, phase: &'static str) {
        if !self.enabled {
            return;
        }
        let Some(snapshot) = cuda_memory_snapshot() else {
            tracing::warn!(phase, "cuda memory probe unavailable");
            return;
        };
        self.record_snapshot(phase, snapshot);
    }

    fn recorder(&mut self) -> Option<&mut dyn ModelTelemetryRecorder> {
        self.enabled
            .then_some(self as &mut dyn ModelTelemetryRecorder)
    }

    fn record_snapshot(&mut self, phase: &'static str, snapshot: GpuMemorySnapshot) {
        if !self.enabled {
            return;
        }
        if self.start.is_none() {
            self.start = Some(snapshot);
        }
        if self
            .peak
            .is_none_or(|peak| snapshot.used_bytes > peak.used_bytes)
        {
            self.peak = Some(snapshot);
        }
        self.end = Some(snapshot);
        let peak_delta_bytes = self.peak_delta_bytes();
        self.probe_events.push(GpuMemoryProbeEvent {
            phase: phase.to_string(),
            used_bytes: snapshot.used_bytes,
            free_bytes: snapshot.free_bytes,
            total_bytes: snapshot.total_bytes,
            peak_delta_bytes,
        });
        tracing::info!(
            phase,
            cuda_mem_used_bytes = snapshot.used_bytes,
            cuda_mem_free_bytes = snapshot.free_bytes,
            cuda_mem_total_bytes = snapshot.total_bytes,
            cuda_mem_peak_delta_bytes = peak_delta_bytes,
            "cuda memory probe"
        );
    }

    fn apply(&self, metrics: &mut Metrics) {
        metrics.cuda_mem_probe_events = self.probe_events.clone();
        metrics.decoder_cache_snapshots = self.decoder_cache_snapshots.clone();
        let (Some(start), Some(peak), Some(end)) = (self.start, self.peak, self.end) else {
            return;
        };
        metrics.cuda_mem_start_used_bytes = start.used_bytes;
        metrics.cuda_mem_peak_used_bytes = peak.used_bytes;
        metrics.cuda_mem_end_used_bytes = end.used_bytes;
        metrics.cuda_mem_total_bytes = peak.total_bytes;
        metrics.cuda_mem_peak_delta_bytes = peak.used_bytes.saturating_sub(start.used_bytes);
    }

    fn peak_delta_bytes(&self) -> u64 {
        match (self.start, self.peak) {
            (Some(start), Some(peak)) => peak.used_bytes.saturating_sub(start.used_bytes),
            _ => 0,
        }
    }
}

impl ModelTelemetryRecorder for StepGpuMemory {
    fn record_phase(&mut self, phase: &'static str) {
        self.record(phase);
    }

    fn record_decoder_cache(&mut self, snapshots: Vec<DecoderCacheSnapshot>) {
        if !self.enabled || snapshots.is_empty() {
            return;
        }
        for snapshot in &snapshots {
            tracing::info!(
                phase = snapshot.phase.as_str(),
                layer_index = snapshot.layer_index,
                cache_kind = snapshot.kind.as_str(),
                cache_seen_tokens = snapshot.seen_tokens,
                cache_retained_tokens = snapshot.retained_tokens,
                cache_max_retained_tokens = ?snapshot.max_retained_tokens,
                "decoder cache probe"
            );
        }
        self.decoder_cache_snapshots.extend(snapshots);
    }
}

/// Per-prompt quantities aggregated into a window's [`Metrics`] (the reward
/// distribution, completion length, dropped/truncated rows, the group's total
/// completion tokens — the DAPO normalizer contribution — and whether the
/// group was a degenerate all-equal-reward no-op).
struct PromptStat {
    rewards: Vec<f32>,
    completion_len: f32,
    /// Total *real* completion tokens in the group (Σ `completion_lens`,
    /// EOS-inclusive, pre-truncation-masking) — this group's contribution to
    /// the DAPO window normalizer, counted even for degenerate groups (TRL's
    /// `num_items_in_batch` counts every completion in the batch).
    completion_tokens: usize,
    dropped: usize,
    /// Completions masked out by truncation masking (ran to the full width
    /// without sampling EOS while `truncation_masking` is active).
    truncated: usize,
    degenerate: bool,
    /// Train/rollout ratio aggregates — `None` when the policy captured no
    /// behavior log-probs, or when the group was skipped before its `logp_old`
    /// scoring snapshot existed (a degenerate group at `beta == 0`).
    ratio_stats: Option<RatioStats>,
}

struct UpdateCtx<'a> {
    vars: &'a [Var],
    opt: &'a mut FerrlAdamW,
    gpu_mem: &'a mut StepGpuMemory,
}

trait PolicyExecution<P: Policy> {
    fn execution_comm<'a>(&'a self, trainer_comm: &'a dyn Comm) -> &'a dyn Comm {
        trainer_comm
    }

    fn generate_at_instrumented(
        &self,
        policy: &mut P,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> Result<Rollout, TrainerError>;

    fn token_logprobs(&self, policy: &P, rollout: &Rollout) -> Result<Tensor, TrainerError>;

    fn token_logprobs_detached(
        &self,
        policy: &P,
        rollout: &Rollout,
    ) -> Result<Tensor, TrainerError>;

    fn backward(&self, policy: &P, loss: &Tensor) -> Result<GradStore, TrainerError> {
        policy.backward(loss).map_err(TrainerError::from)
    }

    fn execution_rank(&self, trainer_comm: &dyn Comm) -> usize {
        self.execution_comm(trainer_comm).rank()
    }

    fn execution_world_size(&self, trainer_comm: &dyn Comm) -> usize {
        self.execution_comm(trainer_comm).world_size()
    }

    fn is_execution_primary(&self, trainer_comm: &dyn Comm) -> bool {
        self.execution_rank(trainer_comm) == 0
    }

    fn writes_rank_local_telemetry(&self, _trainer_comm: &dyn Comm) -> bool {
        true
    }

    fn model_parallel_world_size(&self) -> usize {
        1
    }

    fn execution_all_reduce_scalar_sum(
        &self,
        trainer_comm: &dyn Comm,
        value: f64,
    ) -> Result<f64, TrainerError> {
        Ok(self
            .execution_comm(trainer_comm)
            .all_reduce_scalar_sum(value)?)
    }

    fn reduce_model_parallel_grads(
        &self,
        _vars: &[Var],
        _acc: &mut [Option<Tensor>],
        _covered: &[bool],
    ) -> Result<f64, TrainerError> {
        Ok(0.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct UnshardedPolicyExecution;

impl<P: Policy> PolicyExecution<P> for UnshardedPolicyExecution {
    fn generate_at_instrumented(
        &self,
        policy: &mut P,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> Result<Rollout, TrainerError> {
        policy
            .generate_at_instrumented(prompt, cfg, global_row_base, telemetry)
            .map_err(TrainerError::from)
    }

    fn token_logprobs(&self, policy: &P, rollout: &Rollout) -> Result<Tensor, TrainerError> {
        policy.token_logprobs(rollout).map_err(TrainerError::from)
    }

    fn token_logprobs_detached(
        &self,
        policy: &P,
        rollout: &Rollout,
    ) -> Result<Tensor, TrainerError> {
        policy
            .token_logprobs_detached(rollout)
            .map_err(TrainerError::from)
    }
}

#[derive(Debug, Clone, Copy)]
struct TensorParallelPolicyExecution<'a> {
    comm: &'a dyn Comm,
}

impl TensorParallelPolicyExecution<'_> {
    fn active_comm<'a>(&'a self, trainer_comm: &'a dyn Comm) -> &'a dyn Comm {
        if self.comm.world_size() > 1 {
            self.comm
        } else {
            trainer_comm
        }
    }
}

fn tensor_parallel_policy_call<T>(
    operation: &'static str,
    comm: &dyn Comm,
    call: impl FnOnce() -> CandleResult<T>,
) -> Result<T, TrainerError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) if comm.world_size() > 1 => {
            Err(TrainerError::TensorParallelExecutionTerminal {
                operation,
                detail: error.to_string(),
            })
        }
        Ok(Err(error)) => Err(TrainerError::Candle(error)),
        Err(payload) if comm.world_size() > 1 => {
            Err(TrainerError::TensorParallelExecutionTerminal {
                operation,
                detail: format!(
                    "policy hook panicked: {}",
                    panic_payload_message(payload.as_ref())
                ),
            })
        }
        Err(payload) => Err(TrainerError::Contract(format!(
            "{operation} policy hook panicked: {}",
            panic_payload_message(payload.as_ref())
        ))),
    }
}

impl<P: TensorParallelPolicy> PolicyExecution<P> for TensorParallelPolicyExecution<'_> {
    fn execution_comm<'a>(&'a self, trainer_comm: &'a dyn Comm) -> &'a dyn Comm {
        self.active_comm(trainer_comm)
    }

    fn generate_at_instrumented(
        &self,
        policy: &mut P,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> Result<Rollout, TrainerError> {
        tensor_parallel_policy_call("rollout generation", self.comm, || {
            policy.generate_at_tensor_parallel_instrumented(
                prompt,
                cfg,
                global_row_base,
                self.comm,
                telemetry,
            )
        })
    }

    fn token_logprobs(&self, policy: &P, rollout: &Rollout) -> Result<Tensor, TrainerError> {
        tensor_parallel_policy_call("differentiable scoring", self.comm, || {
            policy.token_logprobs_tensor_parallel(rollout, self.comm)
        })
    }

    fn token_logprobs_detached(
        &self,
        policy: &P,
        rollout: &Rollout,
    ) -> Result<Tensor, TrainerError> {
        tensor_parallel_policy_call("detached scoring", self.comm, || {
            policy.token_logprobs_tensor_parallel_detached(rollout, self.comm)
        })
    }

    fn backward(&self, policy: &P, loss: &Tensor) -> Result<GradStore, TrainerError> {
        tensor_parallel_policy_call("backward", self.comm, || {
            policy.backward_tensor_parallel(loss, self.comm)
        })
    }

    fn execution_rank(&self, trainer_comm: &dyn Comm) -> usize {
        self.active_comm(trainer_comm).rank()
    }

    fn execution_world_size(&self, trainer_comm: &dyn Comm) -> usize {
        self.active_comm(trainer_comm).world_size()
    }

    fn is_execution_primary(&self, trainer_comm: &dyn Comm) -> bool {
        if self.comm.world_size() > 1 {
            self.comm.rank() == 0 && trainer_comm.rank() == 0
        } else {
            trainer_comm.rank() == 0
        }
    }

    fn writes_rank_local_telemetry(&self, trainer_comm: &dyn Comm) -> bool {
        if self.comm.world_size() > 1 {
            self.comm.rank() == 0 && trainer_comm.rank() == 0
        } else {
            true
        }
    }

    fn model_parallel_world_size(&self) -> usize {
        self.comm.world_size()
    }

    fn execution_all_reduce_scalar_sum(
        &self,
        trainer_comm: &dyn Comm,
        value: f64,
    ) -> Result<f64, TrainerError> {
        Ok(self
            .active_comm(trainer_comm)
            .all_reduce_scalar_sum(value)?)
    }

    fn reduce_model_parallel_grads(
        &self,
        vars: &[Var],
        acc: &mut [Option<Tensor>],
        covered: &[bool],
    ) -> Result<f64, TrainerError> {
        if self.comm.world_size() > 1 {
            reduce_accumulated_grads(self.comm, vars, acc, covered)
        } else {
            Ok(0.0)
        }
    }
}

/// Bound the number of per-var gradient tensors reduced in one DP collective.
///
/// The real NCCL path materializes a reduced destination buffer for each source tensor, so a
/// single all-vars collective briefly doubles the accumulated-gradient footprint. Chunking keeps
/// that peak bounded while preserving a deterministic collective order across ranks.
const GRAD_REDUCE_CHUNK: usize = 16;

/// Why a preemption-aware run returned — distinguishes a run that reached
/// `config.steps` from one the cooperative preemption flag stopped early.
///
/// Returned by [`Trainer::resume_latest`] (the launch entry point paired with
/// [`with_preemption_flag`](Trainer::with_preemption_flag)) so a launcher can tell
/// "checkpointed for requeue" from "finished all steps": the loop `break`s and
/// returns identically on both, so without this signal the launcher cannot avoid
/// running held-out eval / a final gate on a *partial* history during a Slurm
/// grace window — burning that window and failing the run on incomplete data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStop {
    /// The loop ran every step up to `config.steps`: the run is finished and
    /// downstream eval / gating may proceed on the full history.
    Completed,
    /// The preemption flag fired: a final momentum-faithful checkpoint was written
    /// at the last completed step and the loop stopped early. The returned history
    /// is partial — the launcher should exit promptly (skipping eval / the gate) so
    /// a requeued launch continues it via [`Trainer::resume_latest`].
    Preempted,
}

/// rank 0's [`Trainer::resume_latest`] discovery outcome, encoded as one `f64` for
/// the cross-rank broadcast sum (see `resume_latest`).
///
/// The three outcomes are kept disjoint and exact so a single
/// [`Comm::all_reduce_scalar_sum`] — with
/// rank 0 contributing its decision and every peer contributing [`Fresh`](Self::Fresh)
/// (the additive identity `0.0`) — returns rank 0's decision, rank-identical, on every
/// rank. Crucially this lets a rank-0 scan *failure* ride the same broadcast every rank
/// enters, so a transient IO fault aborts the world in lockstep instead of stranding
/// the peers in a collective rank 0 bailed before reaching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResumeDecision {
    /// No checkpoint found — start a fresh run.
    Fresh,
    /// Resume from this completed step.
    Resume(u64),
    /// rank 0's checkpoint scan itself failed (the directory exists but its listing
    /// errored) — the world must abort, not resume or start fresh.
    ScanFailed,
}

impl ResumeDecision {
    /// Encode for the broadcast sum: `Fresh → 0.0` (also the additive identity the
    /// peers contribute), `Resume(s) → s + 1.0` (always `≥ 1.0`, distinct from the
    /// `Fresh` sentinel even for `s = 0`), `ScanFailed → -1.0`. Every value is an exact
    /// integer far below 2^53, so summing rank 0's value against zero-contributing peers
    /// returns it bit-exactly.
    fn encode(self) -> f64 {
        match self {
            ResumeDecision::Fresh => 0.0,
            ResumeDecision::Resume(step) => step as f64 + 1.0,
            ResumeDecision::ScanFailed => -1.0,
        }
    }

    /// Inverse of [`encode`](Self::encode). The wire values are exact by construction;
    /// the `±0.5` thresholds are purely defensive rounding.
    fn decode(signal: f64) -> Self {
        if signal < -0.5 {
            ResumeDecision::ScanFailed
        } else if signal < 0.5 {
            ResumeDecision::Fresh
        } else {
            ResumeDecision::Resume(signal.round() as u64 - 1)
        }
    }
}

impl Trainer {
    /// Open a trainer for `config`, persisting it to `run`'s `config.json` and
    /// opening the `metrics.jsonl` writer.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidConfig`] if [`TrainerConfig::validate`]
    /// rejects `config`, or [`TrainerError::Telemetry`] if the config cannot be
    /// written or the metrics file cannot be opened.
    pub fn new(config: TrainerConfig, run: &RunDir) -> Result<Self, TrainerError> {
        Self::with_comm(config, run, SoloComm)
    }

    /// Open a trainer as **one rank of a data-parallel world** (see
    /// [`crate::comm`]). [`new`](Self::new) is exactly this with [`SoloComm`].
    ///
    /// Every rank of the world runs the same training entry point
    /// ([`train`](Self::train) / [`resume`](Self::resume)) with the **same
    /// config, the same prompt list, and identically initialized policy
    /// weights** (same checkpoint or same seed); each rank owns its **own**
    /// `run` directory (the config and metrics writes would otherwise
    /// collide). `config.grad_accum_steps` is the **per-rank** accumulation
    /// count. With [`RewardGroupScope::Local`], window `step` consumes the
    /// `grad_accum_steps × world_size` prompts starting at
    /// `step × grad_accum_steps × world_size` (mod len), rank `r` taking the
    /// `r`-th contiguous slice — the union across ranks is exactly the window a
    /// single-rank run with the global accumulation count would consume. With
    /// [`RewardGroupScope::DistributedSamePrompt`], each accumulation position
    /// selects one prompt shared by every rank, while each rank still gets a
    /// distinct rollout RNG row range for its local completions.
    ///
    /// Per-rank weights stay in bitwise lockstep (same start + all-reduced
    /// gradients + same optimizer arithmetic). Metrics: `kl`, `clip_ratio`
    /// and `grad_norm` are **global** (reduced); everything else —
    /// reward/length statistics, `frac_reward_zero_std`, `dropped_rows`,
    /// `frac_truncated`, and the `rollout_*` off-policy telemetry — describes
    /// the rank's **local shard**; under `DistributedSamePrompt`, those local
    /// shards are completions for the same prompt. Checkpoints are written by
    /// rank 0 only —
    /// to resume, every rank loads rank 0's checkpoint directory (weights and
    /// optimizer moments are rank-identical by lockstep; rank 0's rollout-sampler
    /// RNG blob is now SUFFICIENT for a bit-exact stochastic resume. Under
    /// global-index substream seeding every rank shares one `base_seed` and a
    /// row's draws are a pure function of `(base_seed, global row index)`
    /// ([`Policy::generate_at`]), so a resumed run re-derives every rank's exact
    /// rollout from the restored seed and the recomputed step — rollout diversity
    /// comes from the global index, not per-rank seeds, so no per-rank sampler
    /// state is needed (this closed the former per-rank-sampler follow-up)).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn with_comm(
        config: TrainerConfig,
        run: &RunDir,
        comm: impl Comm + 'static,
    ) -> Result<Self, TrainerError> {
        config.validate()?;
        // Not a validation error (a short smoke run of a long-run config is
        // legitimate), but loud: such a run trains entirely inside the ramp.
        if config.warmup_steps >= config.steps && config.warmup_steps > 0 {
            tracing::warn!(
                warmup_steps = config.warmup_steps,
                steps = config.steps,
                "warmup_steps >= steps: the run never reaches the configured lr"
            );
        }
        run.write_config(&config)?;
        let writer = run.metrics_writer()?;
        let candidate_writer = if config.candidate_log_top_k > 0 {
            Some(run.candidate_writer()?)
        } else {
            None
        };
        Ok(Self {
            config,
            writer,
            checkpoints_dir: run.checkpoints_dir(),
            comm: Arc::new(comm),
            preempt: None,
            candidate_writer,
        })
    }

    /// Install a cooperative **preemption flag**. When the flag flips to `true`
    /// mid-run — e.g. a `SIGTERM`/`SIGUSR1` handler the run binary installs for
    /// Slurm's preempt / timeout grace signal sets it — the training loop writes a
    /// final checkpoint at the next step boundary and stops cleanly, so a requeued
    /// run continues from it via [`resume_latest`](Self::resume_latest).
    ///
    /// The trainer only **polls** the flag; it never installs a signal handler
    /// itself (that stays in the run binary, keeping the library free of OS signal
    /// handling and `forbid(unsafe_code)`-clean). Under data parallelism the poll
    /// is globalized — all ranks stop on the same step — and **install-invariant**:
    /// a rank without a flag still joins the per-step poll (contributing "no stop"),
    /// so installing on only some ranks just means "no preemption," never a deadlock.
    #[must_use]
    pub fn with_preemption_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.preempt = Some(flag);
        self
    }

    /// Redirect checkpoint reads **and** writes to `dir` instead of this run's
    /// own `checkpoints/` subdirectory.
    ///
    /// The default ([`with_comm`](Self::with_comm)) checkpoints under the run
    /// dir, so each rank's checkpoints land in its **own** per-rank run dir —
    /// correct for per-rank telemetry (`metrics.jsonl` never interleaves), but
    /// it leaves non-zero ranks with no checkpoints of their own (the
    /// `maybe_checkpoint` write is rank-0-only). For a
    /// data-parallel run that wants **auto-resume**
    /// ([`resume_latest`](Self::resume_latest)), point **every rank** at one
    /// **shared** directory on a filesystem all ranks can read (a single node's
    /// `/tmp`, or NFS across nodes): rank 0 writes the world's checkpoints there
    /// and every rank discovers and resumes from them. Per-rank run dirs still
    /// own each rank's `metrics.jsonl`; only the checkpoint location is shared.
    ///
    /// Has no effect on the world-1 / single-rank path beyond relocating the
    /// checkpoint directory.
    #[must_use]
    pub fn with_checkpoints_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.checkpoints_dir = dir.into();
        self
    }

    /// Collect and atomically publish one global rollout window without scoring
    /// old/reference log-probabilities or mutating learner parameters.
    ///
    /// `policy_sha256` must be a verified lowercase SHA-256 digest binding the
    /// frozen model content and its execution recipe. It is the sole identity
    /// datum the generic [`Policy`] seam cannot derive from live state today.
    /// Format v4 does not publish collector performance telemetry; a future
    /// phase-specific schema can do so without mislabelling asynchronous work as
    /// an ordinary whole trainer step.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] for an invalid DP topology, an out-of-range step,
    /// malformed identity/state, collection failure, or publication failure.
    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    pub fn collect_rollout_ledger_step<P, R>(
        &mut self,
        step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        root: impl AsRef<Path>,
        policy_sha256: &str,
        continuation: Option<&RolloutLedgerContinuation>,
    ) -> Result<PathBuf, TrainerError>
    where
        P: Policy,
        R: RewardFn,
    {
        let exec = UnshardedPolicyExecution;
        self.collect_rollout_ledger_step_with_execution(
            step,
            policy,
            reward_fn,
            tokenizer,
            samples,
            root,
            policy_sha256,
            continuation,
            &exec,
        )
    }

    /// Collect one logical rollout window through an explicit tensor-parallel
    /// policy execution world. Every TP rank executes the model and validates
    /// identical logical payload bytes; execution rank 0 alone publishes the
    /// world-one ledger package.
    ///
    /// # Errors
    ///
    /// As [`collect_rollout_ledger_step`](Self::collect_rollout_ledger_step),
    /// plus invalid or simultaneous sharded DP×TP execution.
    #[allow(clippy::too_many_arguments)]
    pub fn collect_rollout_ledger_step_tensor_parallel<P, R>(
        &mut self,
        step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        root: impl AsRef<Path>,
        policy_sha256: &str,
        continuation: Option<&RolloutLedgerContinuation>,
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<PathBuf, TrainerError>
    where
        P: TensorParallelPolicy,
        R: RewardFn,
    {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.collect_rollout_ledger_step_with_execution(
            step,
            policy,
            reward_fn,
            tokenizer,
            samples,
            root,
            policy_sha256,
            continuation,
            &exec,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    fn collect_rollout_ledger_step_with_execution<P, R, E>(
        &mut self,
        step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        root: impl AsRef<Path>,
        policy_sha256: &str,
        continuation: Option<&RolloutLedgerContinuation>,
        exec: &E,
    ) -> Result<PathBuf, TrainerError>
    where
        P: Policy,
        R: RewardFn,
        E: PolicyExecution<P>,
    {
        self.require_rollout_ledger_topology()?;
        let trainer_comm = Arc::clone(&self.comm);
        let execution_comm = exec.execution_comm(trainer_comm.as_ref());
        let tensor_parallel_world_size = exec.model_parallel_world_size();
        let root = root.as_ref();
        let consensus = Self::coordinate_comm_result(
            execution_comm,
            "rollout-ledger distributed input serialization",
            serde_json::to_vec(&(
                &self.config,
                root.as_os_str().as_encoded_bytes(),
                policy_sha256,
                continuation.is_some(),
                samples
                    .iter()
                    .map(|sample| sample.prompt.as_str())
                    .collect::<Vec<_>>(),
            ))
            .map_err(|error| {
                TrainerError::Contract(format!(
                    "serialize distributed rollout-ledger inputs: {error}"
                ))
            }),
        )?;
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger root/config/prompt contract",
            &consensus,
        )?;
        let (vars, opt, step_beta, controls, sampler_prestate, lineage, identity) =
            Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger collector preflight",
                || {
                    self.require_rollout_ledger_step_in_range(step)?;
                    validate_external_policy_sha256(policy_sha256)?;
                    if samples.is_empty() {
                        return Err(TrainerError::Contract(
                            "collect_rollout_ledger_step: no samples".into(),
                        ));
                    }
                    let vars = policy.trainable_vars();
                    let mut opt = self.new_optimizer(vars.clone())?;
                    if let Some(state) = continuation {
                        opt.load_state(state.optimizer_state())?;
                    }
                    let step_lr = self.config.lr_at(step);
                    let step_beta = self.config.beta_at(step);
                    self.require_toggleable_reference_policy(
                        policy,
                        self.config.requires_reference_policy(),
                    )?;
                    opt.set_learning_rate(step_lr);
                    let controls =
                        self.rollout_ledger_controls(step, opt.learning_rate(), step_beta)?;
                    let sampler_prestate = policy.sampler_state()?;
                    let lineage = self
                        .require_rollout_ledger_continuation_state_with_model_parallel(
                            step,
                            policy,
                            policy_sha256,
                            &vars,
                            &opt,
                            &sampler_prestate,
                            continuation,
                            tensor_parallel_world_size,
                        )?;
                    let identity = self.rollout_ledger_identity_with_model_parallel(
                        step,
                        policy,
                        policy_sha256,
                        &vars,
                        &opt,
                        &sampler_prestate,
                        &lineage,
                        tensor_parallel_world_size,
                    )?;
                    Ok((
                        vars,
                        opt,
                        step_beta,
                        controls,
                        sampler_prestate,
                        lineage,
                        identity,
                    ))
                },
            )?;
        let identity_bytes = serde_json::to_vec(&(&identity, &controls)).map_err(|error| {
            TrainerError::Contract(format!("serialize rollout-ledger identity: {error}"))
        })?;
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger collector identity/controls",
            &identity_bytes,
        )?;
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger collector sampler prestate",
            &sampler_prestate,
        )?;
        let is_execution_primary = exec.is_execution_primary(self.comm.as_ref());
        let writer =
            Self::coordinate_comm_call(execution_comm, "rollout-ledger writer creation", || {
                if tensor_parallel_world_size <= 1 || is_execution_primary {
                    RolloutLedgerWriter::create(root, identity.clone())
                        .map(Some)
                        .map_err(TrainerError::from)
                } else {
                    Ok(None)
                }
            })?;
        let outcome: Result<PathBuf, TrainerError> = (|| {
            let mut gpu_mem = StepGpuMemory::new(false);
            let collected = Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger group collection",
                || {
                    let mut collected = Vec::with_capacity(self.config.grad_accum_steps);
                    for j in 0..self.config.grad_accum_steps {
                        let sel = self.select_prompt(step, j, samples.len());
                        self.record_prompt_selection(&sel);
                        let selected = SelectedSample {
                            sample: &samples[sel.sample_idx],
                            selection: &sel,
                            accum_index: j,
                        };
                        collected.push(self.collect_group(
                            step,
                            step_beta,
                            policy,
                            reward_fn,
                            tokenizer,
                            &selected,
                            &mut gpu_mem,
                            exec,
                            None,
                        )?);
                    }
                    Ok(collected)
                },
            )?;
            let sampler_poststate = Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger collector poststate",
                || {
                    let post_collection_vars = policy.trainable_vars();
                    self.require_same_rollout_ledger_vars(
                        &vars,
                        &post_collection_vars,
                        "rollout collection",
                    )?;
                    let post_collection_identity = self
                        .rollout_ledger_identity_with_model_parallel(
                            step,
                            policy,
                            policy_sha256,
                            &post_collection_vars,
                            &opt,
                            &sampler_prestate,
                            &lineage,
                            tensor_parallel_world_size,
                        )?;
                    if post_collection_identity != identity {
                        return Err(TrainerError::Contract(
                            "learner identity changed during rollout collection".into(),
                        ));
                    }
                    Ok(policy.sampler_state()?)
                },
            )?;
            Self::require_comm_consensus_bytes(
                execution_comm,
                "rollout-ledger collector sampler poststate",
                &sampler_poststate,
            )?;
            let (window_tokens, live_items) =
                self.rollout_ledger_global_counts(&collected, step_beta)?;
            let payload = Self::coordinate_comm_result(
                execution_comm,
                "rollout-ledger payload construction",
                self.rollout_ledger_payload(
                    step,
                    &controls,
                    &collected,
                    sampler_poststate,
                    window_tokens,
                    live_items,
                ),
            )?;
            if tensor_parallel_world_size > 1 {
                let payload_bytes = serde_json::to_vec(&payload).map_err(|error| {
                    TrainerError::Contract(format!(
                        "serialize tensor-parallel rollout-ledger payload: {error}"
                    ))
                })?;
                Self::require_comm_consensus_bytes(
                    execution_comm,
                    "tensor-parallel rollout-ledger logical payload",
                    &payload_bytes,
                )?;
                self.publish_tensor_parallel_rollout_ledger_step(
                    writer.as_ref(),
                    root,
                    &payload,
                    execution_comm,
                    is_execution_primary,
                )
            } else if self.comm.world_size() == 1 {
                let writer = writer.as_ref().ok_or_else(|| {
                    TrainerError::Contract(
                        "world-one rollout-ledger writer was not constructed".into(),
                    )
                })?;
                let final_dir = writer.root().join(format!("step-{:020}", payload.step));
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    writer.write_step(&payload)
                }))
                .unwrap_or_else(|panic| {
                    Err(RolloutLedgerError::PublicationAmbiguous {
                        path: final_dir,
                        detail: format!(
                            "world-one rollout-ledger publisher panicked after entry: {}",
                            panic_payload_message(panic.as_ref())
                        ),
                    })
                })
                .map_err(TrainerError::from)
            } else {
                self.publish_distributed_rollout_ledger_step(
                    writer.as_ref().ok_or_else(|| {
                        TrainerError::Contract(
                            "distributed rollout-ledger writer was not constructed".into(),
                        )
                    })?,
                    &payload,
                    &controls,
                )
            }
        })();
        match outcome {
            Ok(path) => Ok(path),
            Err(error @ TrainerError::PublicationAmbiguousAfterComm { .. }) => Err(error),
            Err(TrainerError::RolloutLedger(error)) if error.may_be_visible() => {
                // A manifest-bearing ledger may already be consumable. Rewinding
                // the sampler would split collector state from that visible L_k.
                Err(TrainerError::RolloutLedger(error))
            }
            Err(TrainerError::TensorParallelExecutionTerminal {
                operation,
                mut detail,
            }) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local tensor-parallel rollout-ledger collector rollback",
                    || Self::restore_rollout_ledger_sampler(policy, &sampler_prestate),
                );
                match rollback {
                    Ok(()) => detail.push_str("; local sampler rollback succeeded"),
                    Err(error) => detail.push_str(&format!(
                        "; local sampler rollback failed ({error}); policy state is partial"
                    )),
                }
                Err(TrainerError::TensorParallelExecutionTerminal { operation, detail })
            }
            Err(TrainerError::Comm(comm_error)) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local rollout-ledger collector rollback",
                    || Self::restore_rollout_ledger_sampler(policy, &sampler_prestate),
                );
                Err(Self::terminal_distributed_comm_failure(
                    "rollout-ledger collector",
                    &comm_error,
                    None,
                    rollback,
                    "policy instance",
                ))
            }
            Err(error) => {
                let rollback = Self::coordinate_comm_call(
                    execution_comm,
                    "rollout-ledger collector rollback",
                    || Self::restore_rollout_ledger_sampler(policy, &sampler_prestate),
                );
                if let Err(rollback) = rollback {
                    return Err(TrainerError::Contract(format!(
                        "rollout-ledger collector failed ({error}); coordinated sampler rollback also failed ({rollback}); discard the policy instance on every rank in this execution world"
                    )));
                }
                Err(error)
            }
        }
    }

    /// Validate and consume this rank's shard of one global rollout ledger window
    /// optimizer step, returning the metrics row and an opaque chain-bound
    /// continuation receipt carrying the exact post-update Adam state.
    /// This separated learner installs and verifies the collector's exact
    /// post-rollout sampler state before it returns, making its continuation
    /// checkpoint-faithful. Rollout-ledger v4 carries no collector timing or
    /// device-memory telemetry, so
    /// the returned/persisted whole-window performance fields remain explicitly
    /// unmeasured rather than relabelling learner-only work as a complete step.
    /// Exact rollback requires the policy to retain the same trainable [`Var`]
    /// binding across adapter toggling and scoring. A policy that replaces those
    /// bindings is rejected, and—because the generic [`Policy`] seam cannot
    /// reattach the original variables—the returned contract error also reports
    /// that rollback could not restore the exact pre-call state; callers must
    /// discard that policy instance.
    ///
    /// `policy_sha256` has the same externally verified frozen-model/execution
    /// meaning as on [`collect_rollout_ledger_step`](Self::collect_rollout_ledger_step).
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] for an invalid DP topology, an out-of-range step,
    /// malformed/mismatched learner state, invalid ledger data, scoring/backward
    /// failure, a grad-canary failure, or a metrics write failure.
    #[allow(clippy::cognitive_complexity)]
    pub fn train_rollout_ledger_step<P: Policy>(
        &mut self,
        step: u64,
        policy: &mut P,
        root: impl AsRef<Path>,
        policy_sha256: &str,
        continuation: Option<&RolloutLedgerContinuation>,
    ) -> Result<(Metrics, RolloutLedgerContinuation), TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.train_rollout_ledger_step_with_execution(
            step,
            policy,
            root,
            policy_sha256,
            continuation,
            &exec,
        )
    }

    /// Consume one logical ledger window through an explicit tensor-parallel
    /// policy execution world. Every TP rank validates the same package and
    /// runs scoring/backward; replicated adapter gradients are sum-reduced
    /// before the optimizer step and execution rank 0 owns metrics publication.
    ///
    /// # Errors
    ///
    /// As [`train_rollout_ledger_step`](Self::train_rollout_ledger_step), plus
    /// invalid, forward-only, or simultaneous sharded DP×TP execution.
    pub fn train_rollout_ledger_step_tensor_parallel<P: TensorParallelPolicy>(
        &mut self,
        step: u64,
        policy: &mut P,
        root: impl AsRef<Path>,
        policy_sha256: &str,
        continuation: Option<&RolloutLedgerContinuation>,
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<(Metrics, RolloutLedgerContinuation), TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        self.validate_tensor_parallel_backward(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.train_rollout_ledger_step_with_execution(
            step,
            policy,
            root,
            policy_sha256,
            continuation,
            &exec,
        )
    }

    #[allow(clippy::cognitive_complexity)]
    fn train_rollout_ledger_step_with_execution<P, E>(
        &mut self,
        step: u64,
        policy: &mut P,
        root: impl AsRef<Path>,
        policy_sha256: &str,
        continuation: Option<&RolloutLedgerContinuation>,
        exec: &E,
    ) -> Result<(Metrics, RolloutLedgerContinuation), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        self.require_rollout_ledger_topology()?;
        let trainer_comm = Arc::clone(&self.comm);
        let execution_comm = exec.execution_comm(trainer_comm.as_ref());
        let tensor_parallel_world_size = exec.model_parallel_world_size();
        let root = root.as_ref();
        let consensus = Self::coordinate_comm_result(
            execution_comm,
            "rollout-ledger learner input serialization",
            serde_json::to_vec(&(
                &self.config,
                root.as_os_str().as_encoded_bytes(),
                policy_sha256,
                continuation.is_some(),
                tensor_parallel_world_size,
            ))
            .map_err(|error| {
                TrainerError::Contract(format!(
                    "serialize distributed rollout-ledger learner inputs: {error}"
                ))
            }),
        )?;
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger learner root/config contract",
            &consensus,
        )?;
        let (vars, mut opt, step_beta, _controls, sampler_prestate, lineage, validated) =
            Self::coordinate_comm_call(execution_comm, "rollout-ledger learner preflight", || {
                self.require_rollout_ledger_step_in_range(step)?;
                validate_external_policy_sha256(policy_sha256)?;
                let vars = policy.trainable_vars();
                let mut opt = self.new_optimizer(vars.clone())?;
                if let Some(state) = continuation {
                    opt.load_state(state.optimizer_state())?;
                }
                let step_beta = self.config.beta_at(step);
                opt.set_learning_rate(self.config.lr_at(step));
                let controls =
                    self.rollout_ledger_controls(step, opt.learning_rate(), step_beta)?;
                let sampler_prestate = policy.sampler_state()?;
                let lineage = self.require_rollout_ledger_continuation_state_with_model_parallel(
                    step,
                    policy,
                    policy_sha256,
                    &vars,
                    &opt,
                    &sampler_prestate,
                    continuation,
                    tensor_parallel_world_size,
                )?;
                let identity = self.rollout_ledger_identity_with_model_parallel(
                    step,
                    policy,
                    policy_sha256,
                    &vars,
                    &opt,
                    &sampler_prestate,
                    &lineage,
                    tensor_parallel_world_size,
                )?;
                let reader = RolloutLedgerReader::open(
                    root,
                    RolloutLedgerExpectations {
                        identity,
                        controls: controls.clone(),
                    },
                )?;
                let validated = if self.comm.world_size() == 1 {
                    reader.read_step(step)?
                } else {
                    reader.read_distributed_step(
                        step,
                        u32::try_from(self.comm.rank()).map_err(|_| {
                            TrainerError::Contract("rollout-ledger rank does not fit u32".into())
                        })?,
                        u32::try_from(self.comm.world_size()).map_err(|_| {
                            TrainerError::Contract(
                                "rollout-ledger world size does not fit u32".into(),
                            )
                        })?,
                    )?
                };
                Ok((
                    vars,
                    opt,
                    step_beta,
                    controls,
                    sampler_prestate,
                    lineage,
                    validated,
                ))
            })?;
        let validated_identity = validated.identity().clone();
        let consumed_ledger_sha256 = validated.consumed_ledger_sha256().to_owned();
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger committed world manifest",
            consumed_ledger_sha256.as_bytes(),
        )?;
        let payload = validated.into_step();
        let (adapter_prestate, optimizer_prestate, adapter_enabled_prestate) =
            Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger learner rollback snapshot",
                || {
                    Ok((
                        Self::snapshot_rollout_ledger_vars(&vars)?,
                        opt.state()?,
                        policy.adapter_enabled(),
                    ))
                },
            )?;
        let outcome: Result<(Metrics, RolloutLedgerContinuation), TrainerError> = (|| {
            Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger reference-policy preflight",
                || {
                    self.require_toggleable_reference_policy(
                        policy,
                        self.config.requires_reference_policy(),
                    )?;
                    let post_toggle_vars = policy.trainable_vars();
                    self.require_same_rollout_ledger_vars(
                        &vars,
                        &post_toggle_vars,
                        "reference-policy preflight",
                    )?;
                    let post_toggle_identity = self.rollout_ledger_identity_with_model_parallel(
                        step,
                        policy,
                        policy_sha256,
                        &post_toggle_vars,
                        &opt,
                        &sampler_prestate,
                        &lineage,
                        tensor_parallel_world_size,
                    )?;
                    if post_toggle_identity != validated_identity {
                        return Err(TrainerError::Contract(
                            "learner identity changed during reference-policy preflight".into(),
                        ));
                    }
                    Ok(())
                },
            )?;

            let mut gpu_mem = StepGpuMemory::new(false);
            let (stats, live) = Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger detached scoring",
                || {
                    let mut stats = Vec::with_capacity(payload.groups.len());
                    let mut live = Vec::with_capacity(payload.groups.len());
                    for group in &payload.groups {
                        let collected = self.collected_group_from_ledger(group)?;
                        let (stat, item) = self.materialize_collected_group(
                            policy,
                            collected,
                            step_beta,
                            &mut gpu_mem,
                            exec,
                        )?;
                        stats.push(stat);
                        if let Some(item) = item {
                            live.push(item);
                        }
                    }
                    let post_scoring_vars = policy.trainable_vars();
                    self.require_same_rollout_ledger_vars(
                        &vars,
                        &post_scoring_vars,
                        "detached ledger scoring",
                    )?;
                    let post_scoring_identity = self.rollout_ledger_identity_with_model_parallel(
                        step,
                        policy,
                        policy_sha256,
                        &post_scoring_vars,
                        &opt,
                        &sampler_prestate,
                        &lineage,
                        tensor_parallel_world_size,
                    )?;
                    if post_scoring_identity != validated_identity {
                        return Err(TrainerError::Contract(
                            "learner identity changed during detached ledger scoring".into(),
                        ));
                    }
                    Ok((stats, live))
                },
            )?;
            let (local_tokens, local_live) = Self::coordinate_comm_result(
                execution_comm,
                "rollout-ledger learner local counts",
                (|| {
                    let tokens = stats.iter().try_fold(0_u64, |total, stat| {
                        total
                            .checked_add(u64::try_from(stat.completion_tokens).map_err(|_| {
                                TrainerError::Contract(
                                    "learner completion-token count overflows u64".into(),
                                )
                            })?)
                            .ok_or_else(|| {
                                TrainerError::Contract(
                                    "learner completion-token total overflow".into(),
                                )
                            })
                    })?;
                    Ok((
                        tokens,
                        u64::try_from(live.len()).map_err(|_| {
                            TrainerError::Contract("learner live-item count overflows u64".into())
                        })?,
                    ))
                })(),
            )?;
            if tensor_parallel_world_size > 1 {
                let local_count_bytes =
                    serde_json::to_vec(&(local_tokens, local_live)).map_err(|error| {
                        TrainerError::Contract(format!(
                            "serialize tensor-parallel learner counts: {error}"
                        ))
                    })?;
                Self::require_comm_consensus_bytes(
                    execution_comm,
                    "tensor-parallel rollout-ledger learner counts",
                    &local_count_bytes,
                )?;
            }
            let (actual_window_tokens, actual_live_items) = if self.comm.world_size() > 1 {
                let (local_tokens_f64, local_live_f64) = self.coordinate_data_parallel_result(
                    "rollout-ledger learner exact count conversion",
                    (|| {
                        Ok((
                            exact_u64_as_f64(
                                "rollout-ledger learner completion-token count",
                                local_tokens,
                            )?,
                            exact_u64_as_f64("rollout-ledger learner live-item count", local_live)?,
                        ))
                    })(),
                )?;
                let tokens = self.comm.all_reduce_scalar_sum(local_tokens_f64)?;
                let live = self.comm.all_reduce_scalar_sum(local_live_f64)?;
                self.coordinate_data_parallel_result(
                    "rollout-ledger learner global counts",
                    (|| {
                        Ok((
                            exact_reduced_u64(
                                "rollout-ledger learner completion-token count",
                                tokens,
                            )?
                            .max(1),
                            exact_reduced_u64("rollout-ledger learner live-item count", live)?,
                        ))
                    })(),
                )?
            } else {
                (local_tokens.max(1), local_live)
            };
            if actual_window_tokens != payload.window_tokens
                || actual_live_items != u64::from(payload.live_items)
            {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger runtime global counts {actual_window_tokens}/{actual_live_items} do not match committed {}/{}",
                    payload.window_tokens, payload.live_items
                )));
            }
            let agg = if payload.live_items == 0 {
                InnerAgg::default()
            } else {
                let mut ctx = UpdateCtx {
                    vars: &vars,
                    opt: &mut opt,
                    gpu_mem: &mut gpu_mem,
                };
                self.update_window(
                    policy,
                    &live,
                    &mut ctx,
                    payload.window_tokens as f64,
                    f64::from(payload.live_items),
                    step_beta,
                    exec,
                )?
            };
            Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger learner post-update state",
                || {
                    let post_update_vars = policy.trainable_vars();
                    self.require_same_rollout_ledger_vars(
                        &vars,
                        &post_update_vars,
                        "rollout-ledger learner update",
                    )?;
                    Self::restore_rollout_ledger_sampler(
                        policy,
                        &payload.post_rollout_sampler_state,
                    )?;
                    let post_handoff_vars = policy.trainable_vars();
                    self.require_same_rollout_ledger_vars(
                        &vars,
                        &post_handoff_vars,
                        "rollout-ledger sampler handoff",
                    )?;
                    Ok(())
                },
            )?;
            Self::require_comm_consensus_bytes(
                execution_comm,
                "rollout-ledger learner sampler handoff",
                &payload.post_rollout_sampler_state,
            )?;

            let metrics = self.build_window_metrics(step, step_beta, &stats, &agg, &opt);
            let next_lineage = domain_sha256(
                "ferrl.rollout-ledger.lineage.v1",
                &[lineage.as_bytes(), consumed_ledger_sha256.as_bytes()],
            );
            let (optimizer_state, post_identity) = Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger continuation construction",
                || {
                    let optimizer_state = opt.state()?;
                    let sampler_poststate = policy.sampler_state()?;
                    let post_identity = self.rollout_ledger_identity_with_model_parallel(
                        step + 1,
                        policy,
                        policy_sha256,
                        &vars,
                        &opt,
                        &sampler_poststate,
                        &next_lineage,
                        tensor_parallel_world_size,
                    )?;
                    Ok((optimizer_state, post_identity))
                },
            )?;
            let post_identity_bytes = serde_json::to_vec(&post_identity).map_err(|error| {
                TrainerError::Contract(format!("serialize post-ledger identity: {error}"))
            })?;
            Self::require_comm_consensus_bytes(
                execution_comm,
                "rollout-ledger continuation state",
                &post_identity_bytes,
            )?;
            let continuation = RolloutLedgerContinuation {
                completed_step: step + 1,
                world_size: u32::try_from(self.comm.world_size()).map_err(|_| {
                    TrainerError::Contract(
                        "rollout-ledger world size does not fit continuation u32".into(),
                    )
                })?,
                tensor_parallel_world_size: u32::try_from(tensor_parallel_world_size).map_err(
                    |_| {
                        TrainerError::Contract(
                            "tensor-parallel world size does not fit continuation u32".into(),
                        )
                    },
                )?,
                tensor_parallel_layout: crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_LAYOUT
                    .to_owned(),
                optimizer_state,
                policy_sha256: post_identity.policy_sha256,
                trainer_config_sha256: post_identity.trainer_config_sha256,
                tensor_schema_sha256: post_identity.tensor_schema_sha256,
                adapter_sha256: post_identity.adapter_sha256,
                optimizer_sha256: post_identity.optimizer_sha256,
                sampler_sha256: post_identity.sampler_sha256,
                parent_lineage_sha256: lineage.clone(),
                consumed_ledger_sha256: consumed_ledger_sha256.clone(),
                lineage_sha256: next_lineage,
            };
            self.append_rollout_ledger_metrics_with_execution(
                &metrics,
                execution_comm,
                exec.is_execution_primary(self.comm.as_ref()),
                tensor_parallel_world_size > 1,
            )?;
            Ok((metrics, continuation))
        })();

        match outcome {
            Ok(result) => Ok(result),
            Err(TrainerError::RolloutLedgerMetricsComm {
                communication,
                telemetry_rollback,
            }) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local rollout-ledger learner rollback",
                    || {
                        Self::restore_rollout_ledger_prestate(
                            policy,
                            &vars,
                            &adapter_prestate,
                            &mut opt,
                            &optimizer_prestate,
                            adapter_enabled_prestate,
                            &sampler_prestate,
                        )
                    },
                );
                Err(Self::terminal_distributed_comm_failure(
                    "rollout-ledger metrics publication",
                    &communication,
                    Some(&telemetry_rollback),
                    rollback,
                    "policy and optimizer state",
                ))
            }
            Err(TrainerError::Comm(comm_error)) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local rollout-ledger learner rollback",
                    || {
                        Self::restore_rollout_ledger_prestate(
                            policy,
                            &vars,
                            &adapter_prestate,
                            &mut opt,
                            &optimizer_prestate,
                            adapter_enabled_prestate,
                            &sampler_prestate,
                        )
                    },
                );
                Err(Self::terminal_distributed_comm_failure(
                    "rollout-ledger learner",
                    &comm_error,
                    None,
                    rollback,
                    "policy and optimizer state",
                ))
            }
            Err(TrainerError::TensorParallelExecutionTerminal {
                operation,
                mut detail,
            }) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local tensor-parallel rollout-ledger learner rollback",
                    || {
                        Self::restore_rollout_ledger_prestate(
                            policy,
                            &vars,
                            &adapter_prestate,
                            &mut opt,
                            &optimizer_prestate,
                            adapter_enabled_prestate,
                            &sampler_prestate,
                        )
                    },
                );
                match rollback {
                    Ok(()) => detail.push_str(
                        "; local adapter/optimizer/sampler rollback succeeded",
                    ),
                    Err(error) => detail.push_str(&format!(
                        "; local adapter/optimizer/sampler rollback failed ({error}); policy state is partial"
                    )),
                }
                Err(TrainerError::TensorParallelExecutionTerminal { operation, detail })
            }
            Err(error) => {
                let rollback = Self::coordinate_comm_call(
                    execution_comm,
                    "rollout-ledger learner rollback",
                    || {
                        Self::restore_rollout_ledger_prestate(
                            policy,
                            &vars,
                            &adapter_prestate,
                            &mut opt,
                            &optimizer_prestate,
                            adapter_enabled_prestate,
                            &sampler_prestate,
                        )
                    },
                );
                if let Err(rollback) = rollback {
                    return Err(TrainerError::Contract(format!(
                        "rollout-ledger learner failed ({error}); coordinated adapter/optimizer/sampler rollback also failed ({rollback}); discard the policy and optimizer state on every rank in this execution world"
                    )));
                }
                Err(error)
            }
        }
    }

    /// Persist one truthful separated-training continuation after a successful
    /// [`train_rollout_ledger_step`](Self::train_rollout_ledger_step).
    ///
    /// The checkpoint `C_(k+1)` combines the learner's updated adapter and Adam
    /// state with the collector's post-rollout sampler state installed by ledger
    /// v4. Both the next collector and learner restore this same checkpoint before
    /// processing step `k + 1`. Unlike ordinary cadence checkpointing, this role-
    /// handoff primitive is explicit and never replaces an existing completed-step
    /// path: a second writer would be a continuation fork, so it fails closed.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] if the DP topology is invalid, the receipt's
    /// completed step is outside `1..=config.steps`, its bound policy/config/schema
    /// or adapter/Adam/sampler state does not match the live values, the destination
    /// already exists, or durable checkpoint publication fails.
    pub fn save_rollout_ledger_continuation<P: Policy>(
        &self,
        policy: &P,
        continuation: &RolloutLedgerContinuation,
    ) -> Result<PathBuf, TrainerError> {
        self.save_rollout_ledger_continuation_to(&self.checkpoints_dir, policy, continuation)
    }

    /// Persist a tensor-parallel separated continuation below this trainer's
    /// checkpoint root. Every rank validates the same replicated adapter,
    /// optimizer, sampler, lineage, and topology; execution rank 0 alone writes.
    ///
    /// # Errors
    ///
    /// As [`save_rollout_ledger_continuation`](Self::save_rollout_ledger_continuation),
    /// plus invalid or simultaneous sharded DP×TP execution.
    pub fn save_rollout_ledger_continuation_tensor_parallel<P: TensorParallelPolicy>(
        &self,
        policy: &P,
        continuation: &RolloutLedgerContinuation,
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<PathBuf, TrainerError> {
        self.save_rollout_ledger_continuation_to_tensor_parallel(
            &self.checkpoints_dir,
            policy,
            continuation,
            tensor_parallel_comm,
        )
    }

    /// Persist a continuation below an explicit shared checkpoint root.
    ///
    /// Distributed trainers normally use distinct rank-local [`RunDir`]s for
    /// telemetry. Every rank must therefore call this variant with the same
    /// shared root; rank 0 alone publishes `step-<completed>`, while all peers
    /// validate and receive the publication outcome in lockstep.
    ///
    /// # Errors
    ///
    /// As [`save_rollout_ledger_continuation`](Self::save_rollout_ledger_continuation),
    /// plus a cross-rank destination mismatch.
    pub fn save_rollout_ledger_continuation_to<P: Policy>(
        &self,
        checkpoints_dir: impl AsRef<Path>,
        policy: &P,
        continuation: &RolloutLedgerContinuation,
    ) -> Result<PathBuf, TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.save_rollout_ledger_continuation_to_with_execution(
            checkpoints_dir,
            policy,
            continuation,
            &exec,
        )
    }

    /// Persist a tensor-parallel separated continuation below an explicit
    /// shared checkpoint root.
    ///
    /// # Errors
    ///
    /// As [`save_rollout_ledger_continuation_to`](Self::save_rollout_ledger_continuation_to),
    /// plus invalid or simultaneous sharded DP×TP execution.
    pub fn save_rollout_ledger_continuation_to_tensor_parallel<P: TensorParallelPolicy>(
        &self,
        checkpoints_dir: impl AsRef<Path>,
        policy: &P,
        continuation: &RolloutLedgerContinuation,
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<PathBuf, TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.save_rollout_ledger_continuation_to_with_execution(
            checkpoints_dir,
            policy,
            continuation,
            &exec,
        )
    }

    fn save_rollout_ledger_continuation_to_with_execution<P, E>(
        &self,
        checkpoints_dir: impl AsRef<Path>,
        policy: &P,
        continuation: &RolloutLedgerContinuation,
        exec: &E,
    ) -> Result<PathBuf, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        self.require_rollout_ledger_topology()?;
        let trainer_comm = Arc::clone(&self.comm);
        let execution_comm = exec.execution_comm(trainer_comm.as_ref());
        let tensor_parallel_world_size = exec.model_parallel_world_size();
        let is_execution_primary = exec.is_execution_primary(self.comm.as_ref());
        let checkpoints_dir = checkpoints_dir.as_ref();
        let (vars, sampler_state, dir, recipe, manifest) = Self::coordinate_comm_call(
            execution_comm,
            "rollout-ledger continuation save preflight",
            || {
                let completed_step = continuation.completed_step;
                if completed_step == 0 || completed_step > self.config.steps {
                    return Err(TrainerError::Contract(format!(
                        "rollout-ledger continuation step {completed_step} is outside 1..={}",
                        self.config.steps
                    )));
                }
                let vars = policy.trainable_vars();
                let mut opt = self.new_optimizer(vars.clone())?;
                opt.load_state(continuation.optimizer_state())?;
                let sampler_state = policy.sampler_state()?;
                self.require_rollout_ledger_continuation_state_with_model_parallel(
                    completed_step,
                    policy,
                    &continuation.policy_sha256,
                    &vars,
                    &opt,
                    &sampler_state,
                    Some(continuation),
                    tensor_parallel_world_size,
                )?;
                let dir = checkpoints_dir.join(format!("step-{completed_step}"));
                let recipe = policy.lora_recipe();
                let manifest = crate::checkpoint::RolloutLedgerContinuationManifest {
                    format_version: crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION,
                    kind: crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_KIND.to_owned(),
                    world_size: Some(continuation.world_size),
                    tensor_parallel_world_size: Some(continuation.tensor_parallel_world_size),
                    tensor_parallel_layout: Some(continuation.tensor_parallel_layout.clone()),
                    completed_step,
                    policy_sha256: continuation.policy_sha256.clone(),
                    trainer_config_sha256: continuation.trainer_config_sha256.clone(),
                    tensor_schema_sha256: continuation.tensor_schema_sha256.clone(),
                    adapter_sha256: continuation.adapter_sha256.clone(),
                    optimizer_sha256: continuation.optimizer_sha256.clone(),
                    sampler_sha256: continuation.sampler_sha256.clone(),
                    parent_lineage_sha256: continuation.parent_lineage_sha256.clone(),
                    consumed_ledger_sha256: continuation.consumed_ledger_sha256.clone(),
                    lineage_sha256: continuation.lineage_sha256.clone(),
                };
                Ok((vars, sampler_state, dir, recipe, manifest))
            },
        )?;
        let consensus = Self::coordinate_comm_result(
            execution_comm,
            "rollout-ledger continuation save serialization",
            serde_json::to_vec(&(
                dir.as_os_str().as_encoded_bytes(),
                recipe.as_deref(),
                &manifest,
            ))
            .map_err(|error| {
                TrainerError::Contract(format!(
                    "serialize rollout-ledger continuation save contract: {error}"
                ))
            }),
        )?;
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger continuation save contract",
            &consensus,
        )?;

        let save_local = if is_execution_primary {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::checkpoint::save_checkpoint_no_replace(
                    &dir,
                    &vars,
                    continuation.optimizer_state(),
                    &sampler_state,
                    recipe.as_deref(),
                    manifest,
                )
                .map_err(TrainerError::from)
            }))
            .unwrap_or_else(|payload| {
                Err(TrainerError::Checkpoint(
                    crate::checkpoint::CheckpointError::PublicationAmbiguous {
                        path: dir.clone(),
                        detail: format!(
                            "rollout-ledger continuation publication panicked after the publisher was entered: {}",
                            panic_payload_message(payload.as_ref())
                        ),
                    },
                ))
            })
        } else {
            Ok(())
        };
        let publication_signal = match &save_local {
            Ok(()) => 0.0,
            Err(TrainerError::Checkpoint(
                crate::checkpoint::CheckpointError::PublicationAmbiguous { .. },
            )) => 2.0,
            Err(_) => 1.0,
        };
        let publication_signal = execution_comm.all_reduce_scalar_sum(if is_execution_primary {
            publication_signal
        } else {
            0.0
        });
        let publication_signal = match publication_signal {
            Ok(signal) => signal,
            Err(error) => {
                return Err(TrainerError::PublicationAmbiguousAfterComm {
                    artifact: "rollout-ledger continuation",
                    path: dir,
                    detail: "execution rank 0 entered no-replace publication before the status collective".into(),
                    communication: Box::new(error),
                });
            }
        };
        if publication_signal > 1.5 {
            return match save_local {
                Err(error) => Err(error),
                Ok(()) => Err(TrainerError::Checkpoint(
                    crate::checkpoint::CheckpointError::PublicationAmbiguous {
                        path: dir,
                        detail: "execution rank 0 reported ambiguous continuation publication"
                            .into(),
                    },
                )),
            };
        }
        if publication_signal > 0.5 {
            return match save_local {
                Err(error) => Err(error),
                Ok(()) => Err(TrainerError::Contract(
                    "execution rank 0 failed before continuation publication; every rank may retry"
                        .into(),
                )),
            };
        }
        save_local?;
        Ok(dir)
    }

    /// Restore a mandatory adapter + Adam + sampler continuation for a separated
    /// collector or learner, returning an opaque receipt for the next step.
    ///
    /// Adapter-only legacy checkpoints are deliberately rejected: separated
    /// execution cannot reconstruct the collector's rollout stream or the
    /// learner's momentum from them. On any load, sampler-installation, or Adam
    /// validation failure, this method restores and verifies the policy's exact
    /// adapter and sampler prestate before returning the error.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] for an invalid DP topology, incompatible/missing
    /// continuation state, recipe or tensor mismatch, invalid outer step, sampler
    /// restoration failure, or Adam-state mismatch.
    #[allow(clippy::cognitive_complexity)]
    pub fn restore_rollout_ledger_continuation<P: Policy>(
        &self,
        checkpoint_dir: impl AsRef<Path>,
        policy: &mut P,
        policy_sha256: &str,
    ) -> Result<RolloutLedgerContinuation, TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.restore_rollout_ledger_continuation_with_execution(
            checkpoint_dir,
            policy,
            policy_sha256,
            &exec,
        )
    }

    /// Restore a tensor-parallel separated continuation on every execution
    /// rank, validating the saved DP×TP topology before policy mutation.
    ///
    /// # Errors
    ///
    /// As [`restore_rollout_ledger_continuation`](Self::restore_rollout_ledger_continuation),
    /// plus invalid or simultaneous sharded DP×TP execution.
    pub fn restore_rollout_ledger_continuation_tensor_parallel<P: TensorParallelPolicy>(
        &self,
        checkpoint_dir: impl AsRef<Path>,
        policy: &mut P,
        policy_sha256: &str,
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<RolloutLedgerContinuation, TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.restore_rollout_ledger_continuation_with_execution(
            checkpoint_dir,
            policy,
            policy_sha256,
            &exec,
        )
    }

    #[allow(clippy::cognitive_complexity)]
    fn restore_rollout_ledger_continuation_with_execution<P, E>(
        &self,
        checkpoint_dir: impl AsRef<Path>,
        policy: &mut P,
        policy_sha256: &str,
        exec: &E,
    ) -> Result<RolloutLedgerContinuation, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        self.require_rollout_ledger_topology()?;
        let trainer_comm = Arc::clone(&self.comm);
        let execution_comm = exec.execution_comm(trainer_comm.as_ref());
        let tensor_parallel_world_size = exec.model_parallel_world_size();
        let checkpoint_dir = checkpoint_dir.as_ref();
        let restore_input = Self::coordinate_comm_result(
            execution_comm,
            "rollout-ledger continuation restore input serialization",
            serde_json::to_vec(&(checkpoint_dir.as_os_str().as_encoded_bytes(), policy_sha256))
                .map_err(|error| {
                    TrainerError::Contract(format!(
                        "serialize rollout-ledger continuation restore inputs: {error}"
                    ))
                }),
        )?;
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger continuation restore path/policy",
            &restore_input,
        )?;
        let (vars, adapter_prestate, sampler_prestate) = Self::coordinate_comm_call(
            execution_comm,
            "rollout-ledger continuation restore snapshot",
            || {
                validate_external_policy_sha256(policy_sha256)?;
                let vars = policy.trainable_vars();
                let adapter_prestate = Self::snapshot_rollout_ledger_vars(&vars)?;
                let sampler_prestate = policy.sampler_state()?;
                Ok((vars, adapter_prestate, sampler_prestate))
            },
        )?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let manifest = crate::checkpoint::read_manifest(checkpoint_dir)?;
            let continuation_manifest =
                manifest
                    .rollout_ledger_continuation
                    .clone()
                    .ok_or_else(|| {
                        TrainerError::Contract(
                            "checkpoint is not a separated rollout-ledger continuation".into(),
                        )
                    })?;
            if continuation_manifest.format_version
                < crate::checkpoint::MIN_ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION
                || continuation_manifest.format_version
                    > crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION
                || continuation_manifest.kind
                    != crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_KIND
            {
                return Err(TrainerError::Contract(format!(
                    "unsupported rollout-ledger continuation format version {}",
                    continuation_manifest.format_version
                )));
            }
            let (
                manifest_world_size,
                manifest_tensor_parallel_world_size,
                manifest_tensor_parallel_layout,
            ) = match (
                continuation_manifest.format_version,
                continuation_manifest.world_size,
                continuation_manifest.tensor_parallel_world_size,
                continuation_manifest.tensor_parallel_layout.as_deref(),
            ) {
                (1, None, None, None) => (
                    1,
                    1,
                    crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_LAYOUT,
                ),
                (2, Some(world_size), None, None) if world_size > 0 => (
                    world_size,
                    1,
                    crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_LAYOUT,
                ),
                (
                    crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_FORMAT_VERSION,
                    Some(world_size),
                    Some(tensor_parallel_world_size),
                    Some(tensor_parallel_layout),
                ) if world_size > 0 && tensor_parallel_world_size > 0 => {
                    (world_size, tensor_parallel_world_size, tensor_parallel_layout)
                }
                _ => {
                    return Err(TrainerError::Contract(
                        "rollout-ledger continuation topology is malformed for its format version"
                            .into(),
                    ));
                }
            };
            let current_world_size = u32::try_from(self.comm.world_size()).map_err(|_| {
                TrainerError::Contract(
                    "rollout-ledger world size does not fit continuation u32".into(),
                )
            })?;
            if manifest_world_size != current_world_size {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation world size {manifest_world_size} does not match current world {current_world_size}"
                )));
            }
            let current_tensor_parallel_world_size =
                u32::try_from(tensor_parallel_world_size).map_err(|_| {
                    TrainerError::Contract(
                        "tensor-parallel world size does not fit continuation u32".into(),
                    )
                })?;
            if manifest_tensor_parallel_world_size != current_tensor_parallel_world_size {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation tensor-parallel world size {manifest_tensor_parallel_world_size} does not match current world {current_tensor_parallel_world_size}"
                )));
            }
            if manifest_tensor_parallel_layout
                != crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_LAYOUT
            {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation tensor-parallel layout {manifest_tensor_parallel_layout:?} is unsupported"
                )));
            }
            if continuation_manifest.completed_step != manifest.step
                || checkpoint_dir.file_name().and_then(|name| name.to_str())
                    != Some(format!("step-{}", manifest.step).as_str())
            {
                return Err(TrainerError::Contract(
                    "rollout-ledger continuation outer step does not match its manifest/path"
                        .into(),
                ));
            }
            if manifest.optimizer_step_t.is_none()
                || manifest.optimizer_num_vars.is_none()
                || manifest.sampler_state.is_none()
            {
                return Err(TrainerError::Contract(
                    "rollout-ledger continuation requires adapter, Adam, and sampler state".into(),
                ));
            }
            let current_recipe = policy.lora_recipe();
            if manifest.lora_recipe != current_recipe {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation adapter recipe {:?} does not match the policy's {current_recipe:?}",
                    manifest.lora_recipe
                )));
            }
            if continuation_manifest.policy_sha256 != policy_sha256 {
                return Err(TrainerError::Contract(
                    "rollout-ledger continuation frozen-policy identity mismatch".into(),
                ));
            }
            validate_external_policy_sha256(&continuation_manifest.parent_lineage_sha256)?;
            validate_external_policy_sha256(&continuation_manifest.consumed_ledger_sha256)?;
            validate_external_policy_sha256(&continuation_manifest.lineage_sha256)?;
            let expected_lineage = domain_sha256(
                "ferrl.rollout-ledger.lineage.v1",
                &[
                    continuation_manifest.parent_lineage_sha256.as_bytes(),
                    continuation_manifest.consumed_ledger_sha256.as_bytes(),
                ],
            );
            if expected_lineage != continuation_manifest.lineage_sha256 {
                return Err(TrainerError::Contract(
                    "rollout-ledger continuation lineage mismatch".into(),
                ));
            }
            let preflight_opt = self.new_optimizer(vars.clone())?;
            let preflight_identity = self.rollout_ledger_identity_with_model_parallel(
                manifest.step,
                policy,
                policy_sha256,
                &vars,
                &preflight_opt,
                &sampler_prestate,
                &continuation_manifest.lineage_sha256,
                tensor_parallel_world_size,
            )?;
            if continuation_manifest.trainer_config_sha256
                != preflight_identity.trainer_config_sha256
                || continuation_manifest.tensor_schema_sha256
                    != preflight_identity.tensor_schema_sha256
            {
                return Err(TrainerError::Contract(
                    "rollout-ledger continuation trainer configuration or tensor schema mismatch"
                        .into(),
                ));
            }
            let loaded = crate::checkpoint::load_checkpoint(checkpoint_dir, &vars)?;
            if loaded.step == 0 || loaded.step > self.config.steps {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation step {} is outside 1..={}",
                    loaded.step, self.config.steps
                )));
            }
            let optimizer_state = loaded.optimizer_state.ok_or_else(|| {
                TrainerError::Contract("rollout-ledger continuation is missing Adam state".into())
            })?;
            let sampler_state = loaded.sampler_state.ok_or_else(|| {
                TrainerError::Contract(
                    "rollout-ledger continuation is missing sampler state".into(),
                )
            })?;
            Self::restore_rollout_ledger_sampler(policy, &sampler_state)?;
            let active_vars = policy.trainable_vars();
            self.require_same_rollout_ledger_vars(
                &vars,
                &active_vars,
                "rollout-ledger continuation restore",
            )?;
            let mut opt = self.new_optimizer(vars.clone())?;
            opt.load_state(&optimizer_state)?;
            let actual = self.rollout_ledger_identity_with_model_parallel(
                loaded.step,
                policy,
                policy_sha256,
                &vars,
                &opt,
                &sampler_state,
                &continuation_manifest.lineage_sha256,
                tensor_parallel_world_size,
            )?;
            if continuation_manifest.adapter_sha256 != actual.adapter_sha256
                || continuation_manifest.optimizer_sha256 != actual.optimizer_sha256
                || continuation_manifest.sampler_sha256 != actual.sampler_sha256
            {
                return Err(TrainerError::Contract(
                    "rollout-ledger continuation payload does not match its bound adapter/Adam/sampler state"
                        .into(),
                ));
            }
            Ok(RolloutLedgerContinuation {
                completed_step: loaded.step,
                world_size: manifest_world_size,
                tensor_parallel_world_size: manifest_tensor_parallel_world_size,
                tensor_parallel_layout: manifest_tensor_parallel_layout.to_owned(),
                optimizer_state,
                policy_sha256: continuation_manifest.policy_sha256,
                trainer_config_sha256: continuation_manifest.trainer_config_sha256,
                tensor_schema_sha256: continuation_manifest.tensor_schema_sha256,
                adapter_sha256: continuation_manifest.adapter_sha256,
                optimizer_sha256: continuation_manifest.optimizer_sha256,
                sampler_sha256: continuation_manifest.sampler_sha256,
                parent_lineage_sha256: continuation_manifest.parent_lineage_sha256,
                consumed_ledger_sha256: continuation_manifest.consumed_ledger_sha256,
                lineage_sha256: continuation_manifest.lineage_sha256,
            })
        }))
        .unwrap_or_else(|payload| {
            Err(TrainerError::Contract(format!(
                "rollout-ledger continuation restore panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let failed_local = if outcome.is_err() { 1.0 } else { 0.0 };
        let failed_global = execution_comm.all_reduce_scalar_sum(failed_local);
        let failed_global = match failed_global {
            Ok(failed) => failed,
            Err(comm_error) => {
                let local_error = outcome.as_ref().err().map(ToString::to_string);
                return Err(Self::terminal_rollout_ledger_restore_comm_failure(
                    "rollout-ledger continuation restore-status collective",
                    &comm_error,
                    local_error.as_deref(),
                    policy,
                    &vars,
                    &adapter_prestate,
                    &sampler_prestate,
                ));
            }
        };
        if failed_global > 0.0 {
            let local_error = outcome.err();
            let rollback = Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger continuation restore rollback",
                || {
                    Self::restore_rollout_ledger_checkpoint_prestate(
                        policy,
                        &vars,
                        &adapter_prestate,
                        &sampler_prestate,
                    )
                },
            );
            return match (local_error, rollback) {
                (_, Err(rollback_error)) => Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation restore failed on at least one rank; coordinated rollback failed ({rollback_error}); discard the policy state on every rank in this execution world"
                ))),
                (Some(error), Ok(())) => Err(error),
                (None, Ok(())) => Err(TrainerError::Contract(
                    "rollout-ledger continuation restore failed on a peer rank; every rank restored its pre-state"
                        .into(),
                )),
            };
        }
        let continuation = outcome?;
        let consensus = (|| {
            let bytes = Self::coordinate_comm_result(
                execution_comm,
                "rollout-ledger restored continuation serialization",
                serde_json::to_vec(&(
                    continuation.completed_step,
                    continuation.world_size,
                    continuation.tensor_parallel_world_size,
                    &continuation.tensor_parallel_layout,
                    &continuation.policy_sha256,
                    &continuation.trainer_config_sha256,
                    &continuation.tensor_schema_sha256,
                    &continuation.adapter_sha256,
                    &continuation.optimizer_sha256,
                    &continuation.sampler_sha256,
                    &continuation.parent_lineage_sha256,
                    &continuation.consumed_ledger_sha256,
                    &continuation.lineage_sha256,
                ))
                .map_err(|error| {
                    TrainerError::Contract(format!(
                        "serialize restored rollout-ledger continuation: {error}"
                    ))
                }),
            )?;
            Self::require_comm_consensus_bytes(
                execution_comm,
                "restored rollout-ledger continuation",
                &bytes,
            )
        })();
        if let Err(error) = consensus {
            if let TrainerError::Comm(comm_error) = &error {
                return Err(Self::terminal_rollout_ledger_restore_comm_failure(
                    "restored rollout-ledger continuation consensus",
                    comm_error,
                    None,
                    policy,
                    &vars,
                    &adapter_prestate,
                    &sampler_prestate,
                ));
            }
            let rollback = Self::coordinate_comm_call(
                execution_comm,
                "rollout-ledger continuation consensus rollback",
                || {
                    Self::restore_rollout_ledger_checkpoint_prestate(
                        policy,
                        &vars,
                        &adapter_prestate,
                        &sampler_prestate,
                    )
                },
            );
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(TrainerError::Contract(format!(
                    "restored continuation consensus failed ({error}); coordinated rollback also failed ({rollback_error}); discard the policy state on every rank in this execution world"
                ))),
            };
        }
        Ok(continuation)
    }

    /// Restore the newest complete separated continuation, or return `None` when
    /// no checkpoint has been published yet.
    ///
    /// # Errors
    ///
    /// As [`restore_rollout_ledger_continuation`](Self::restore_rollout_ledger_continuation),
    /// plus checkpoint-directory discovery failures.
    pub fn restore_latest_rollout_ledger_continuation<P: Policy>(
        &self,
        policy: &mut P,
        policy_sha256: &str,
    ) -> Result<Option<RolloutLedgerContinuation>, TrainerError> {
        self.restore_latest_rollout_ledger_continuation_from(
            &self.checkpoints_dir,
            policy,
            policy_sha256,
        )
    }

    /// Restore the newest tensor-parallel separated continuation below this
    /// trainer's checkpoint root.
    ///
    /// # Errors
    ///
    /// As [`restore_latest_rollout_ledger_continuation`](Self::restore_latest_rollout_ledger_continuation),
    /// plus invalid or simultaneous sharded DP×TP execution.
    pub fn restore_latest_rollout_ledger_continuation_tensor_parallel<P: TensorParallelPolicy>(
        &self,
        policy: &mut P,
        policy_sha256: &str,
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<Option<RolloutLedgerContinuation>, TrainerError> {
        self.restore_latest_rollout_ledger_continuation_from_tensor_parallel(
            &self.checkpoints_dir,
            policy,
            policy_sha256,
            tensor_parallel_comm,
        )
    }

    /// Discover on rank 0 and restore from an explicit shared checkpoint root.
    ///
    /// # Errors
    ///
    /// As [`restore_latest_rollout_ledger_continuation`](Self::restore_latest_rollout_ledger_continuation),
    /// plus a cross-rank discovery-root mismatch.
    pub fn restore_latest_rollout_ledger_continuation_from<P: Policy>(
        &self,
        checkpoints_dir: impl AsRef<Path>,
        policy: &mut P,
        policy_sha256: &str,
    ) -> Result<Option<RolloutLedgerContinuation>, TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.restore_latest_rollout_ledger_continuation_from_with_execution(
            checkpoints_dir,
            policy,
            policy_sha256,
            &exec,
        )
    }

    /// Discover on tensor-parallel execution rank 0 and restore from an
    /// explicit shared checkpoint root on every rank.
    ///
    /// # Errors
    ///
    /// As [`restore_latest_rollout_ledger_continuation_from`](Self::restore_latest_rollout_ledger_continuation_from),
    /// plus invalid or simultaneous sharded DP×TP execution.
    pub fn restore_latest_rollout_ledger_continuation_from_tensor_parallel<
        P: TensorParallelPolicy,
    >(
        &self,
        checkpoints_dir: impl AsRef<Path>,
        policy: &mut P,
        policy_sha256: &str,
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<Option<RolloutLedgerContinuation>, TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.restore_latest_rollout_ledger_continuation_from_with_execution(
            checkpoints_dir,
            policy,
            policy_sha256,
            &exec,
        )
    }

    fn restore_latest_rollout_ledger_continuation_from_with_execution<P, E>(
        &self,
        checkpoints_dir: impl AsRef<Path>,
        policy: &mut P,
        policy_sha256: &str,
        exec: &E,
    ) -> Result<Option<RolloutLedgerContinuation>, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        self.require_rollout_ledger_topology()?;
        let execution_comm = exec.execution_comm(self.comm.as_ref());
        let is_execution_primary = exec.is_execution_primary(self.comm.as_ref());
        let checkpoints_dir = checkpoints_dir.as_ref();
        let discovery_input = Self::coordinate_comm_result(
            execution_comm,
            "rollout-ledger latest-discovery input serialization",
            serde_json::to_vec(&(
                checkpoints_dir.as_os_str().as_encoded_bytes(),
                policy_sha256,
            ))
            .map_err(|error| {
                TrainerError::Contract(format!(
                    "serialize rollout-ledger latest-discovery inputs: {error}"
                ))
            }),
        )?;
        Self::require_comm_consensus_bytes(
            execution_comm,
            "rollout-ledger latest-discovery root/policy",
            &discovery_input,
        )?;
        let latest_local = if is_execution_primary {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let latest = crate::checkpoint::latest_rollout_ledger_continuation(
                    checkpoints_dir,
                )?;
                if latest
                    .as_ref()
                    .is_some_and(|latest| latest.step > 9_007_199_254_740_991)
                {
                    return Err(TrainerError::Contract(
                        "latest continuation step cannot be represented exactly by the data-parallel discovery signal"
                            .into(),
                    ));
                }
                Ok(latest)
            }))
            .unwrap_or_else(|payload| {
                Err(TrainerError::Contract(format!(
                    "rollout-ledger latest-continuation discovery panicked: {}",
                    panic_payload_message(payload.as_ref())
                )))
            })
        } else {
            Ok(None)
        };
        let decision = match &latest_local {
            Ok(Some(latest)) => ResumeDecision::Resume(latest.step),
            Ok(None) => ResumeDecision::Fresh,
            Err(_) => ResumeDecision::ScanFailed,
        };
        let signal = execution_comm.all_reduce_scalar_sum(if is_execution_primary {
            decision.encode()
        } else {
            0.0
        })?;
        match ResumeDecision::decode(signal) {
            ResumeDecision::Fresh => Ok(None),
            ResumeDecision::ScanFailed => match latest_local {
                Err(error) => Err(error),
                Ok(_) => Err(TrainerError::Contract(
                    "rank 0 failed to discover the latest rollout-ledger continuation".into(),
                )),
            },
            ResumeDecision::Resume(step) => {
                if let Ok(Some(latest)) = &latest_local {
                    if latest.step != step {
                        return Err(TrainerError::Contract(
                            "rank-0 continuation discovery disagrees with its broadcast step"
                                .into(),
                        ));
                    }
                }
                let dir = checkpoints_dir.join(format!("step-{step}"));
                self.restore_rollout_ledger_continuation_with_execution(
                    dir,
                    policy,
                    policy_sha256,
                    exec,
                )
                .map(Some)
            }
        }
    }

    fn require_rollout_ledger_topology(&self) -> Result<(), TrainerError> {
        if self.comm.world_size() == 0 || self.comm.rank() >= self.comm.world_size() {
            return Err(TrainerError::Contract(format!(
                "invalid rollout-ledger data-parallel rank {}/world {}",
                self.comm.rank(),
                self.comm.world_size()
            )));
        }
        u32::try_from(self.comm.rank())
            .map_err(|_| TrainerError::Contract("rollout-ledger rank does not fit u32".into()))?;
        u32::try_from(self.comm.world_size()).map_err(|_| {
            TrainerError::Contract("rollout-ledger world size does not fit u32".into())
        })?;
        Ok(())
    }

    fn coordinate_data_parallel_result<T>(
        &self,
        label: &str,
        local: Result<T, TrainerError>,
    ) -> Result<T, TrainerError> {
        Self::coordinate_comm_result(self.comm.as_ref(), label, local)
    }

    fn is_terminal_distributed_error(error: &TrainerError) -> bool {
        matches!(
            error,
            TrainerError::Comm(_)
                | TrainerError::TensorParallelExecutionTerminal { .. }
                | TrainerError::PublicationAmbiguousAfterComm { .. }
                | TrainerError::RolloutLedgerMetricsComm { .. }
        )
    }

    fn coordinate_comm_result<T>(
        comm: &dyn Comm,
        label: &str,
        local: Result<T, TrainerError>,
    ) -> Result<T, TrainerError> {
        if comm.world_size() <= 1 {
            return local;
        }
        if local
            .as_ref()
            .err()
            .is_some_and(Self::is_terminal_distributed_error)
        {
            return local;
        }
        let failed_local = if local.is_err() { 1.0 } else { 0.0 };
        let failed = comm.all_reduce_scalar_sum(failed_local)?;
        match local {
            Err(error) => Err(error),
            Ok(_) if failed > 0.0 => Err(TrainerError::Contract(format!(
                "{label} failed on a peer rank; aborting in lockstep"
            ))),
            Ok(value) => Ok(value),
        }
    }

    fn coordinate_model_parallel_result<T>(
        model_parallel_world_size: usize,
        execution_comm: &dyn Comm,
        label: &str,
        local: Result<T, TrainerError>,
    ) -> Result<T, TrainerError> {
        if model_parallel_world_size > 1 {
            Self::coordinate_comm_result(execution_comm, label, local)
        } else {
            local
        }
    }

    fn coordinate_model_parallel_call<T, F>(
        model_parallel_world_size: usize,
        execution_comm: &dyn Comm,
        label: &str,
        operation: F,
    ) -> Result<T, TrainerError>
    where
        F: FnOnce() -> Result<T, TrainerError>,
    {
        if model_parallel_world_size > 1 {
            Self::coordinate_comm_call(execution_comm, label, operation)
        } else {
            operation()
        }
    }

    fn coordinate_comm_call<T, F>(
        comm: &dyn Comm,
        label: &str,
        operation: F,
    ) -> Result<T, TrainerError>
    where
        F: FnOnce() -> Result<T, TrainerError>,
    {
        let local = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation))
            .unwrap_or_else(|payload| {
                Err(TrainerError::Contract(format!(
                    "{label} panicked: {}",
                    panic_payload_message(payload.as_ref())
                )))
            });
        Self::coordinate_comm_result(comm, label, local)
    }

    fn require_comm_consensus_bytes(
        comm: &dyn Comm,
        label: &str,
        value: &[u8],
    ) -> Result<(), TrainerError> {
        if comm.world_size() <= 1 {
            return Ok(());
        }
        let digest: [u8; 32] = Sha256::digest(value).into();
        let mut mismatch = false;
        for word in digest.chunks_exact(4) {
            let local = u32::from_le_bytes(word.try_into().expect("four-byte digest word"));
            let canonical = comm.all_reduce_scalar_sum(if comm.rank() == 0 {
                f64::from(local)
            } else {
                0.0
            })?;
            mismatch |= canonical != f64::from(local);
        }
        Self::coordinate_comm_result(
            comm,
            label,
            if mismatch {
                Err(TrainerError::Contract(format!(
                    "{label} differs across execution ranks"
                )))
            } else {
                Ok(())
            },
        )
    }

    fn distributed_rollout_ledger_nonce(&self) -> Result<u64, TrainerError> {
        let local = if self.comm.rank() == 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos() as u64);
            now ^ u64::from(std::process::id())
        } else {
            0
        };
        let low = self.comm.all_reduce_scalar_sum(if self.comm.rank() == 0 {
            f64::from(local as u32)
        } else {
            0.0
        })?;
        let high = self.comm.all_reduce_scalar_sum(if self.comm.rank() == 0 {
            f64::from((local >> 32) as u32)
        } else {
            0.0
        })?;
        let (low, high) = self.coordinate_data_parallel_result(
            "distributed rollout-ledger nonce broadcast",
            (|| {
                let low = exact_reduced_u64("distributed nonce low word", low)?;
                let high = exact_reduced_u64("distributed nonce high word", high)?;
                Ok((
                    u32::try_from(low).map_err(|_| {
                        TrainerError::Contract("distributed nonce low word exceeds u32".into())
                    })?,
                    u32::try_from(high).map_err(|_| {
                        TrainerError::Contract("distributed nonce high word exceeds u32".into())
                    })?,
                ))
            })(),
        )?;
        Ok(u64::from(low) | (u64::from(high) << 32))
    }

    #[allow(clippy::cognitive_complexity)] // explicit staged-publication state machine
    fn publish_distributed_rollout_ledger_step(
        &self,
        writer: &RolloutLedgerWriter,
        payload: &RolloutLedgerStep,
        controls: &RolloutLedgerControls,
    ) -> Result<PathBuf, TrainerError> {
        let world_size = u32::try_from(self.comm.world_size()).map_err(|_| {
            TrainerError::Contract("rollout-ledger world size does not fit u32".into())
        })?;
        let nonce = self.distributed_rollout_ledger_nonce()?;
        let stage = self.coordinate_data_parallel_result(
            "distributed rollout-ledger stage claim",
            if self.comm.rank() == 0 {
                writer
                    .create_distributed_stage(payload.step, world_size, nonce)
                    .map(Some)
                    .map_err(TrainerError::from)
            } else {
                Ok(None)
            },
        )?;
        let stage_path = stage.as_ref().map_or_else(
            || writer.distributed_stage_path(payload.step, nonce),
            |stage| stage.path().to_path_buf(),
        );

        let shard_local = writer
            .write_distributed_shard(&stage_path, payload)
            .map_err(TrainerError::from);
        let shard_failed =
            self.comm
                .all_reduce_scalar_sum(if shard_local.is_err() { 1.0 } else { 0.0 })?;
        if shard_failed > 0.0 {
            let cleanup = self.coordinate_data_parallel_result(
                "distributed rollout-ledger failed-shard cleanup",
                if self.comm.rank() == 0 {
                    writer
                        .abort_distributed_stage(
                            stage
                                .as_ref()
                                .expect("rank 0 owns the coordinated distributed stage"),
                        )
                        .map_err(TrainerError::from)
                } else {
                    Ok(())
                },
            );
            cleanup?;
            return match shard_local {
                Err(error) => Err(error),
                Ok(_) => Err(TrainerError::Contract(
                    "distributed rollout-ledger shard write failed on a peer rank; the owned stage was cleaned and every rank may retry"
                        .into(),
                )),
            };
        }
        shard_local?;

        let final_dir = writer.root().join(format!("step-{:020}", payload.step));
        let commit_local = if self.comm.rank() == 0 {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                writer.commit_distributed_stage(
                    stage
                        .as_ref()
                        .expect("rank 0 owns the coordinated distributed stage"),
                    payload.step,
                    world_size,
                    controls,
                )
            }))
            .unwrap_or_else(|panic| {
                Err(RolloutLedgerError::PublicationAmbiguous {
                    path: final_dir.clone(),
                    detail: format!(
                        "distributed rollout-ledger publisher panicked after entry: {}",
                        panic_payload_message(panic.as_ref())
                    ),
                })
            })
            .map(Some)
            .map_err(TrainerError::from)
        } else {
            Ok(None)
        };
        let publication_signal = match &commit_local {
            Ok(_) => 0.0,
            Err(TrainerError::RolloutLedger(error)) if error.may_be_visible() => 2.0,
            Err(_) => 1.0,
        };
        let publication_signal = self.comm.all_reduce_scalar_sum(if self.comm.rank() == 0 {
            publication_signal
        } else {
            0.0
        });
        let publication_signal = match publication_signal {
            Ok(value) => value,
            Err(error) => {
                return Err(TrainerError::PublicationAmbiguousAfterComm {
                    artifact: "distributed rollout ledger",
                    path: final_dir,
                    detail: "rank 0 entered the manifest commit phase before the status collective"
                        .into(),
                    communication: Box::new(error),
                });
            }
        };
        if publication_signal > 1.5 {
            return match commit_local {
                Err(error) => Err(error),
                Ok(_) => Err(TrainerError::RolloutLedger(
                    RolloutLedgerError::PublicationAmbiguous {
                        path: final_dir,
                        detail: "rank 0 reported ambiguous distributed publication; sampler state was preserved on every rank"
                            .into(),
                    },
                )),
            };
        }
        if publication_signal > 0.5 {
            return match commit_local {
                Err(error) => Err(error),
                Ok(_) => Err(TrainerError::Contract(
                    "rank 0 failed before distributed rollout-ledger visibility; every rank may rewind and retry"
                        .into(),
                )),
            };
        }
        Ok(commit_local.ok().flatten().unwrap_or(final_dir))
    }

    fn publish_tensor_parallel_rollout_ledger_step(
        &self,
        writer: Option<&RolloutLedgerWriter>,
        root: &Path,
        payload: &RolloutLedgerStep,
        execution_comm: &dyn Comm,
        is_execution_primary: bool,
    ) -> Result<PathBuf, TrainerError> {
        let final_dir = root.join(format!("step-{:020}", payload.step));
        let publish_local = if is_execution_primary {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                writer
                    .ok_or_else(|| {
                        TrainerError::Contract(
                            "tensor-parallel primary has no rollout-ledger writer".into(),
                        )
                    })?
                    .write_step(payload)
                    .map(Some)
                    .map_err(TrainerError::from)
            }))
            .unwrap_or_else(|panic| {
                Err(TrainerError::RolloutLedger(
                    RolloutLedgerError::PublicationAmbiguous {
                        path: final_dir.clone(),
                        detail: format!(
                            "tensor-parallel rollout-ledger publication panicked after the publisher was entered: {}",
                            panic_payload_message(panic.as_ref())
                        ),
                    },
                ))
            })
        } else {
            Ok(None)
        };
        let publication_signal = match &publish_local {
            Ok(_) => 0.0,
            Err(TrainerError::RolloutLedger(error)) if error.may_be_visible() => 2.0,
            Err(_) => 1.0,
        };
        let publication_signal = execution_comm.all_reduce_scalar_sum(if is_execution_primary {
            publication_signal
        } else {
            0.0
        });
        let publication_signal = match publication_signal {
            Ok(signal) => signal,
            Err(error) => {
                return Err(TrainerError::PublicationAmbiguousAfterComm {
                    artifact: "tensor-parallel rollout ledger",
                    path: final_dir,
                    detail: "execution rank 0 entered publication before the status collective"
                        .into(),
                    communication: Box::new(error),
                });
            }
        };
        if publication_signal > 1.5 {
            return match publish_local {
                Err(error) => Err(error),
                Ok(_) => Err(TrainerError::RolloutLedger(
                    RolloutLedgerError::PublicationAmbiguous {
                        path: final_dir,
                        detail: "execution rank 0 reported ambiguous tensor-parallel publication; sampler state was preserved on every rank"
                            .into(),
                    },
                )),
            };
        }
        if publication_signal > 0.5 {
            return match publish_local {
                Err(error) => Err(error),
                Ok(_) => Err(TrainerError::Contract(
                    "execution rank 0 failed before tensor-parallel rollout-ledger visibility; every rank may rewind and retry"
                        .into(),
                )),
            };
        }
        Ok(publish_local.ok().flatten().unwrap_or(final_dir))
    }

    fn require_rollout_ledger_step_in_range(&self, step: u64) -> Result<(), TrainerError> {
        if step >= self.config.steps {
            return Err(TrainerError::Contract(format!(
                "rollout ledger step {step} is outside configured steps 0..{}",
                self.config.steps
            )));
        }
        Ok(())
    }

    fn require_toggleable_reference_policy<P: Policy>(
        &self,
        policy: &mut P,
        required: bool,
    ) -> Result<(), TrainerError> {
        if !required {
            return Ok(());
        }
        policy.set_adapter_enabled(false);
        let toggleable = !policy.adapter_enabled();
        policy.set_adapter_enabled(true);
        if !toggleable {
            return Err(TrainerError::Contract(
                "beta > 0 needs the adapter-disabled reference policy, but this policy cannot \
                 disable its adapter (full fine-tuning mode?) — train with beta = 0, or use a \
                 LoRA recipe for KL-regularized runs"
                    .into(),
            ));
        }
        Ok(())
    }

    fn new_optimizer(&self, vars: Vec<Var>) -> Result<FerrlAdamW, TrainerError> {
        let params = ParamsAdamW {
            lr: self.config.lr,
            weight_decay: self.config.weight_decay,
            beta1: self.config.adam_beta1,
            beta2: self.config.adam_beta2,
            ..Default::default()
        };
        Ok(FerrlAdamW::new(vars, params)?)
    }

    fn require_same_rollout_ledger_vars(
        &self,
        expected: &[Var],
        actual: &[Var],
        phase: &str,
    ) -> Result<(), TrainerError> {
        if !Self::same_rollout_ledger_vars(expected, actual) {
            return Err(TrainerError::Contract(format!(
                "policy trainable-variable set changed during {phase}"
            )));
        }
        Ok(())
    }

    fn same_rollout_ledger_vars(expected: &[Var], actual: &[Var]) -> bool {
        expected.len() == actual.len()
            && expected
                .iter()
                .zip(actual)
                .all(|(expected, actual)| expected.as_tensor().id() == actual.as_tensor().id())
    }

    fn snapshot_rollout_ledger_vars(vars: &[Var]) -> Result<Vec<Tensor>, TrainerError> {
        vars.iter()
            .map(|var| Ok(var.as_tensor().copy()?.contiguous()?))
            .collect()
    }

    fn snapshot_rollout_group_prestate<P: Policy>(
        policy: &P,
    ) -> Result<RolloutGroupPrestate, TrainerError> {
        let adapter_enabled = policy.adapter_enabled();
        let vars = policy.trainable_vars();
        let adapter = Self::snapshot_rollout_ledger_vars(&vars)?;
        let sampler = policy.sampler_state()?;
        Ok(RolloutGroupPrestate {
            vars,
            adapter,
            adapter_enabled,
            sampler,
        })
    }

    #[allow(clippy::cognitive_complexity)]
    fn restore_rollout_group_prestate<P: Policy>(
        policy: &mut P,
        prestate: &RolloutGroupPrestate,
    ) -> Result<(), TrainerError> {
        let mut failures = Vec::new();
        // Restore the flag first because an opaque policy is allowed to choose
        // its active trainable-variable binding from the adapter mode.
        policy.set_adapter_enabled(prestate.adapter_enabled);
        if let Err(error) = Self::restore_rollout_ledger_sampler(policy, &prestate.sampler) {
            failures.push(format!("restore sampler state: {error}"));
        }
        let active_vars = policy.trainable_vars();
        if !Self::same_rollout_ledger_vars(&prestate.vars, &active_vars) {
            failures.push(
                "policy trainable-variable binding changed and cannot be restored through the Policy seam"
                    .into(),
            );
        }
        if prestate.vars.len() != prestate.adapter.len() {
            failures.push(format!(
                "adapter snapshot has {} tensors for {} live variables",
                prestate.adapter.len(),
                prestate.vars.len()
            ));
        } else {
            for (index, (var, snapshot)) in prestate.vars.iter().zip(&prestate.adapter).enumerate()
            {
                if let Err(error) = var.set(snapshot) {
                    failures.push(format!("restore adapter tensor {index}: {error}"));
                }
            }
        }
        if policy.adapter_enabled() != prestate.adapter_enabled {
            failures.push("policy did not restore the exact adapter-enabled state".into());
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(TrainerError::Contract(failures.join("; ")))
        }
    }

    fn rollback_rollout_group_failure<P: Policy>(
        policy: &mut P,
        prestate: Option<&RolloutGroupPrestate>,
        execution_comm: &dyn Comm,
        error: TrainerError,
    ) -> TrainerError {
        let restore = || {
            let prestate = prestate.ok_or_else(|| {
                TrainerError::Contract(
                    "rollout-group prestate snapshot did not complete before failure".into(),
                )
            })?;
            Self::restore_rollout_group_prestate(policy, prestate)
        };

        match error {
            TrainerError::Comm(comm_error) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local direct rollout-group rollback",
                    restore,
                );
                Self::terminal_distributed_comm_failure(
                    "direct rollout group",
                    &comm_error,
                    None,
                    rollback,
                    "policy instance",
                )
            }
            TrainerError::TensorParallelExecutionTerminal {
                operation,
                mut detail,
            } => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local direct rollout-group rollback",
                    restore,
                );
                match rollback {
                    Ok(()) => detail.push_str("; local rollout-group rollback succeeded"),
                    Err(error) => detail.push_str(&format!(
                        "; local rollout-group rollback failed ({error}); policy state is partial"
                    )),
                }
                TrainerError::TensorParallelExecutionTerminal { operation, detail }
            }
            error @ (TrainerError::PublicationAmbiguousAfterComm { .. }
            | TrainerError::RolloutLedgerMetricsComm { .. }) => {
                // These variants cannot arise from ordinary direct group work,
                // but their contract says the communicator is already dead.
                // Preserve that structured classification and never rendezvous.
                let _ = Self::catch_local_distributed_recovery(
                    "best-effort local direct rollout-group rollback",
                    restore,
                );
                error
            }
            error => {
                let rollback_local = Self::catch_local_distributed_recovery(
                    "direct rollout-group rollback",
                    restore,
                );
                let local_rollback_error = rollback_local.as_ref().err().map(ToString::to_string);
                match Self::coordinate_comm_result(
                    execution_comm,
                    "direct rollout-group rollback",
                    rollback_local,
                ) {
                    Ok(()) => error,
                    Err(TrainerError::Comm(comm_error)) => {
                        let local_detail = format!(
                            "direct rollout group failed ({error}); {}",
                            local_rollback_error.as_ref().map_or(
                                "the local policy rollback completed".to_owned(),
                                |rollback| format!("the local policy rollback failed: {rollback}")
                            )
                        );
                        let rollback = local_rollback_error.map_or(Ok(()), |rollback| {
                            Err(TrainerError::Contract(rollback))
                        });
                        Self::terminal_distributed_comm_failure(
                            "direct rollout-group rollback status",
                            &comm_error,
                            Some(&local_detail),
                            rollback,
                            "policy instance",
                        )
                    }
                    Err(rollback) => TrainerError::Contract(format!(
                        "direct rollout group failed ({error}); coordinated adapter/sampler/adapter-mode rollback also failed ({rollback}); discard the policy instance on every rank in this execution world"
                    )),
                }
            }
        }
    }

    fn restore_rollout_ledger_checkpoint_prestate<P: Policy>(
        policy: &mut P,
        vars: &[Var],
        adapter_prestate: &[Tensor],
        sampler_prestate: &[u8],
    ) -> Result<(), TrainerError> {
        let mut failures = Vec::new();
        if vars.len() != adapter_prestate.len() {
            failures.push(format!(
                "adapter snapshot has {} tensors for {} live variables",
                adapter_prestate.len(),
                vars.len()
            ));
        } else {
            for (index, (var, snapshot)) in vars.iter().zip(adapter_prestate).enumerate() {
                if let Err(error) = var.set(snapshot) {
                    failures.push(format!("restore adapter tensor {index}: {error}"));
                }
            }
        }
        if let Err(error) = Self::restore_rollout_ledger_sampler(policy, sampler_prestate) {
            failures.push(format!("restore sampler state: {error}"));
        }
        let active_vars = policy.trainable_vars();
        if !Self::same_rollout_ledger_vars(vars, &active_vars) {
            failures.push(
                "policy trainable-variable binding changed during continuation restore".into(),
            );
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(TrainerError::Contract(failures.join("; ")))
        }
    }

    /// Run recovery after a communication failure without touching the dead
    /// world. Policy callbacks may panic, so every such local-only recovery is
    /// contained here before the caller returns a terminal discard result.
    fn catch_local_distributed_recovery<F>(label: &str, operation: F) -> Result<(), TrainerError>
    where
        F: FnOnce() -> Result<(), TrainerError>,
    {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or_else(
            |payload| {
                Err(TrainerError::Contract(format!(
                    "{label} panicked: {}",
                    panic_payload_message(payload.as_ref())
                )))
            },
        )
    }

    fn rollout_ledger_metrics_rollback_detail(rollback: &Result<(), TrainerError>) -> String {
        match rollback {
            Ok(()) => "best-effort local metrics rollback completed".into(),
            Err(error) => format!(
                "best-effort local metrics rollback failed: {error}; rank-local metrics telemetry may contain a row for a rolled-back learner step and must be repaired or discarded"
            ),
        }
    }

    fn terminal_distributed_comm_failure(
        phase: &str,
        comm_error: &crate::comm::CommError,
        local_detail: Option<&str>,
        rollback: Result<(), TrainerError>,
        discard: &str,
    ) -> TrainerError {
        let local_detail = local_detail.map_or_else(String::new, |detail| format!("; {detail}"));
        let rollback_detail = match rollback {
            Ok(()) => "best-effort local rollback completed".to_owned(),
            Err(error) => format!("best-effort local rollback failed: {error}"),
        };
        TrainerError::Contract(format!(
            "{phase} communication failed ({comm_error}){local_detail}; {rollback_detail}; the distributed execution world is dead and no further collectives are safe; discard the {discard} on every rank in this execution world"
        ))
    }

    fn restore_rollout_ledger_checkpoint_prestate_locally<P: Policy>(
        policy: &mut P,
        vars: &[Var],
        adapter_prestate: &[Tensor],
        sampler_prestate: &[u8],
    ) -> Result<(), TrainerError> {
        Self::catch_local_distributed_recovery("best-effort local continuation rollback", || {
            Self::restore_rollout_ledger_checkpoint_prestate(
                policy,
                vars,
                adapter_prestate,
                sampler_prestate,
            )
        })
    }

    fn terminal_rollout_ledger_restore_comm_failure<P: Policy>(
        phase: &str,
        comm_error: &crate::comm::CommError,
        local_error: Option<&str>,
        policy: &mut P,
        vars: &[Var],
        adapter_prestate: &[Tensor],
        sampler_prestate: &[u8],
    ) -> TrainerError {
        let rollback = Self::restore_rollout_ledger_checkpoint_prestate_locally(
            policy,
            vars,
            adapter_prestate,
            sampler_prestate,
        );
        let local_detail = local_error.map_or_else(
            || "the local restore had completed before communication failed".to_owned(),
            |error| format!("the local restore also failed: {error}"),
        );
        Self::terminal_distributed_comm_failure(
            phase,
            comm_error,
            Some(&local_detail),
            rollback,
            "policy state",
        )
    }

    fn restore_rollout_ledger_sampler<P: Policy>(
        policy: &mut P,
        expected: &[u8],
    ) -> Result<(), TrainerError> {
        policy.restore_sampler_state(expected)?;
        let actual = policy.sampler_state()?;
        if actual != expected {
            return Err(TrainerError::Contract(
                "policy did not install the exact rollout-ledger sampler state".into(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::cognitive_complexity)] // rollback must aggregate every independent failure
    fn restore_rollout_ledger_prestate<P: Policy>(
        policy: &mut P,
        vars: &[Var],
        adapter_prestate: &[Tensor],
        opt: &mut FerrlAdamW,
        optimizer_prestate: &OptimizerState,
        adapter_enabled_prestate: bool,
        sampler_prestate: &[u8],
    ) -> Result<(), TrainerError> {
        let mut failures = Vec::new();
        // Restore the flag first: an adversarial policy may replace its live Vars
        // from set_adapter_enabled itself. Only the binding observed after the
        // final flag transition is eligible for a successful rollback.
        policy.set_adapter_enabled(adapter_enabled_prestate);
        if let Err(error) = Self::restore_rollout_ledger_sampler(policy, sampler_prestate) {
            failures.push(format!("restore sampler state: {error}"));
        }
        let active_vars = policy.trainable_vars();
        if !Self::same_rollout_ledger_vars(vars, &active_vars) {
            failures.push(
                "policy trainable-variable binding changed and cannot be restored through the Policy seam"
                    .into(),
            );
        }
        if vars.len() != adapter_prestate.len() {
            failures.push(format!(
                "adapter snapshot has {} tensors for {} live variables",
                adapter_prestate.len(),
                vars.len()
            ));
        } else {
            for (index, (var, snapshot)) in vars.iter().zip(adapter_prestate).enumerate() {
                if let Err(error) = var.set(snapshot) {
                    failures.push(format!("restore adapter tensor {index}: {error}"));
                }
            }
        }
        if let Err(error) = opt.load_state(optimizer_prestate) {
            failures.push(format!("restore optimizer state: {error}"));
        }
        if policy.adapter_enabled() != adapter_enabled_prestate {
            failures.push("restore adapter-enabled state".into());
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(TrainerError::Contract(failures.join("; ")))
        }
    }

    fn rollout_ledger_controls(
        &self,
        step: u64,
        effective_lr: f64,
        effective_beta: f64,
    ) -> Result<RolloutLedgerControls, TrainerError> {
        let grad_accum_steps = u32::try_from(self.config.grad_accum_steps).map_err(|_| {
            TrainerError::Contract("grad_accum_steps does not fit rollout ledger u32".into())
        })?;
        let group_size = u32::try_from(self.config.group_size).map_err(|_| {
            TrainerError::Contract("group_size does not fit rollout ledger u32".into())
        })?;
        let completion_width = u32::try_from(self.config.max_new_tokens).map_err(|_| {
            TrainerError::Contract("max_new_tokens does not fit rollout ledger u32".into())
        })?;
        debug_assert_eq!(effective_lr.to_bits(), self.config.lr_at(step).to_bits());
        debug_assert_eq!(
            effective_beta.to_bits(),
            self.config.beta_at(step).to_bits()
        );
        Ok(RolloutLedgerControls {
            grad_accum_steps,
            group_size,
            completion_width,
            reward_group_scope: match self.config.reward_group_scope {
                RewardGroupScope::Local => RolloutLedgerGroupScope::Local,
                RewardGroupScope::DistributedSamePrompt => {
                    RolloutLedgerGroupScope::DistributedSamePrompt
                }
            },
            scale_rewards: self.config.scale_rewards,
            eos_token_id: self.config.eos_token_id,
            truncation_masking: self.config.truncation_masking,
            tis_imp_ratio_cap_bits: self
                .config
                .tis
                .then_some(self.config.tis_imp_ratio_cap.to_bits()),
            effective_lr_bits: effective_lr.to_bits(),
            effective_beta_bits: effective_beta.to_bits(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn require_rollout_ledger_continuation_state_with_model_parallel<P: Policy>(
        &self,
        step: u64,
        policy: &P,
        policy_sha256: &str,
        vars: &[Var],
        opt: &FerrlAdamW,
        sampler_state: &[u8],
        continuation: Option<&RolloutLedgerContinuation>,
        tensor_parallel_world_size: usize,
    ) -> Result<String, TrainerError> {
        let lineage = if let Some(continuation) = continuation {
            let current_world_size = u32::try_from(self.comm.world_size()).map_err(|_| {
                TrainerError::Contract(
                    "rollout-ledger world size does not fit continuation u32".into(),
                )
            })?;
            if continuation.world_size != current_world_size {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation world size {} cannot run in world {current_world_size}",
                    continuation.world_size
                )));
            }
            let current_tensor_parallel_world_size = u32::try_from(tensor_parallel_world_size)
                .map_err(|_| {
                    TrainerError::Contract(
                        "tensor-parallel world size does not fit continuation u32".into(),
                    )
                })?;
            if continuation.tensor_parallel_world_size != current_tensor_parallel_world_size {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation tensor-parallel world size {} cannot run in world {current_tensor_parallel_world_size}",
                    continuation.tensor_parallel_world_size
                )));
            }
            if continuation.tensor_parallel_layout
                != crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_LAYOUT
            {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation tensor-parallel layout {:?} is unsupported",
                    continuation.tensor_parallel_layout
                )));
            }
            if continuation.completed_step != step {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation completed step {} cannot start step {step}",
                    continuation.completed_step
                )));
            }
            let expected_lineage = domain_sha256(
                "ferrl.rollout-ledger.lineage.v1",
                &[
                    continuation.parent_lineage_sha256.as_bytes(),
                    continuation.consumed_ledger_sha256.as_bytes(),
                ],
            );
            if expected_lineage != continuation.lineage_sha256 {
                return Err(TrainerError::Contract(
                    "rollout-ledger continuation lineage mismatch".into(),
                ));
            }
            continuation.lineage_sha256.clone()
        } else {
            if step != 0 {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger step {step} requires a chain-bound continuation"
                )));
            }
            let provisional = self.rollout_ledger_identity_with_model_parallel(
                step,
                policy,
                policy_sha256,
                vars,
                opt,
                sampler_state,
                &"0".repeat(64),
                tensor_parallel_world_size,
            )?;
            let bytes = serde_json::to_vec(&provisional).map_err(|error| {
                TrainerError::Contract(format!("serialize ledger genesis identity: {error}"))
            })?;
            domain_sha256("ferrl.rollout-ledger.genesis.v1", &[&bytes])
        };
        let actual = self.rollout_ledger_identity_with_model_parallel(
            step,
            policy,
            policy_sha256,
            vars,
            opt,
            sampler_state,
            &lineage,
            tensor_parallel_world_size,
        )?;
        if let Some(continuation) = continuation {
            let mismatches = [
                ("policy", continuation.policy_sha256 != actual.policy_sha256),
                (
                    "trainer config",
                    continuation.trainer_config_sha256 != actual.trainer_config_sha256,
                ),
                (
                    "tensor schema",
                    continuation.tensor_schema_sha256 != actual.tensor_schema_sha256,
                ),
                (
                    "adapter",
                    continuation.adapter_sha256 != actual.adapter_sha256,
                ),
                (
                    "Adam",
                    continuation.optimizer_sha256 != actual.optimizer_sha256,
                ),
                (
                    "sampler",
                    continuation.sampler_sha256 != actual.sampler_sha256,
                ),
            ]
            .into_iter()
            .filter_map(|(label, differs)| differs.then_some(label))
            .collect::<Vec<_>>();
            if !mismatches.is_empty() {
                return Err(TrainerError::Contract(format!(
                    "rollout-ledger continuation does not match live {} state",
                    mismatches.join("/")
                )));
            }
        }
        Ok(lineage)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_ledger_identity_with_model_parallel<P: Policy>(
        &self,
        step: u64,
        policy: &P,
        policy_sha256: &str,
        vars: &[Var],
        opt: &FerrlAdamW,
        sampler_state: &[u8],
        lineage_sha256: &str,
        tensor_parallel_world_size: usize,
    ) -> Result<RolloutLedgerIdentity, TrainerError> {
        let (config, config_domain) = match (self.comm.world_size(), tensor_parallel_world_size) {
            (1, 1) => (
                serde_json::to_vec(&self.config.rollout_ledger_semantics()),
                "ferrl.rollout-ledger.trainer-config.v1",
            ),
            (_, 1) => (
                serde_json::to_vec(&(
                    self.config.rollout_ledger_semantics(),
                    self.comm.world_size(),
                )),
                "ferrl.rollout-ledger.trainer-config.v2",
            ),
            (_, tensor_parallel_world_size) => (
                serde_json::to_vec(&(
                    self.config.rollout_ledger_semantics(),
                    self.comm.world_size(),
                    tensor_parallel_world_size,
                    crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_LAYOUT,
                )),
                "ferrl.rollout-ledger.trainer-config.v3",
            ),
        };
        let config = config.map_err(|error| {
            TrainerError::Contract(format!(
                "serialize learner-semantic trainer config for ledger identity: {error}"
            ))
        })?;
        let schema: Vec<(String, Vec<usize>, String)> = vars
            .iter()
            .enumerate()
            .map(|(index, var)| {
                (
                    format!("lora.{index:05}"),
                    var.as_tensor().dims().to_vec(),
                    var.as_tensor().dtype().as_str().to_owned(),
                )
            })
            .collect();
        let schema = serde_json::to_vec(&(policy.lora_recipe(), schema)).map_err(|error| {
            TrainerError::Contract(format!(
                "serialize tensor schema for ledger identity: {error}"
            ))
        })?;
        let adapter = canonical_tensor_bytes(
            vars.iter()
                .enumerate()
                .map(|(index, var)| (format!("lora.{index:05}"), var.as_tensor())),
        )?;
        let optimizer_state = opt.state()?;
        let optimizer_step = u64::try_from(optimizer_state.step_t).map_err(|_| {
            TrainerError::Contract("optimizer step does not fit rollout ledger u64".into())
        })?;
        let optimizer_tensors = canonical_tensor_bytes(
            optimizer_state
                .first_moments
                .iter()
                .enumerate()
                .map(|(index, tensor)| (format!("m.{index:05}"), tensor))
                .chain(
                    optimizer_state
                        .second_moments
                        .iter()
                        .enumerate()
                        .map(|(index, tensor)| (format!("v.{index:05}"), tensor)),
                ),
        )?;
        Ok(RolloutLedgerIdentity {
            trainer_config_sha256: domain_sha256(config_domain, &[&config]),
            policy_sha256: policy_sha256.to_owned(),
            tensor_schema_sha256: domain_sha256(
                "ferrl.rollout-ledger.tensor-schema.v1",
                &[&schema],
            ),
            adapter_sha256: domain_sha256("ferrl.rollout-ledger.adapter.v1", &[&adapter]),
            optimizer_sha256: domain_sha256(
                "ferrl.rollout-ledger.optimizer.v2",
                &[&optimizer_step.to_le_bytes(), &optimizer_tensors],
            ),
            sampler_sha256: domain_sha256("ferrl.rollout-ledger.sampler.v1", &[sampler_state]),
            lineage_sha256: lineage_sha256.to_owned(),
            source_step: step,
            optimizer_step,
        })
    }

    fn rollout_ledger_global_counts(
        &self,
        collected: &[CollectedGroup],
        beta: f64,
    ) -> Result<(u64, u32), TrainerError> {
        let (local_tokens, local_live) = self.coordinate_data_parallel_result(
            "rollout-ledger local count derivation",
            (|| {
                let local_tokens = collected.iter().try_fold(0_u64, |total, item| {
                    item.rollout
                        .completion_lens
                        .iter()
                        .try_fold(total, |total, &len| {
                            total
                                .checked_add(u64::try_from(len).map_err(|_| {
                                    TrainerError::Contract(
                                        "completion length does not fit rollout ledger u64".into(),
                                    )
                                })?)
                                .ok_or_else(|| {
                                    TrainerError::Contract(
                                        "rollout ledger window token count overflow".into(),
                                    )
                                })
                        })
                })?;
                let local_live = u64::try_from(
                    collected
                        .iter()
                        .filter(|item| beta > 0.0 || item.surrogate_live)
                        .count(),
                )
                .map_err(|_| {
                    TrainerError::Contract("rollout ledger live-item count overflows u64".into())
                })?;
                Ok((local_tokens, local_live))
            })(),
        )?;
        if self.comm.world_size() == 1 {
            return Ok((
                local_tokens.max(1),
                u32::try_from(local_live).map_err(|_| {
                    TrainerError::Contract("rollout ledger live-item count overflows u32".into())
                })?,
            ));
        }
        let (local_tokens_f64, local_live_f64) = self.coordinate_data_parallel_result(
            "rollout-ledger collector exact count conversion",
            (|| {
                Ok((
                    exact_u64_as_f64("rollout ledger completion-token count", local_tokens)?,
                    exact_u64_as_f64("rollout ledger live-item count", local_live)?,
                ))
            })(),
        )?;
        let tokens = self.comm.all_reduce_scalar_sum(local_tokens_f64)?;
        let live = self.comm.all_reduce_scalar_sum(local_live_f64)?;
        let (global_tokens, global_live) = self.coordinate_data_parallel_result(
            "rollout-ledger collector global counts",
            (|| {
                Ok((
                    exact_reduced_u64("rollout ledger completion-token count", tokens)?.max(1),
                    exact_reduced_u64("rollout ledger live-item count", live)?,
                ))
            })(),
        )?;
        Ok((
            global_tokens,
            u32::try_from(global_live).map_err(|_| {
                TrainerError::Contract("rollout ledger global live-item count overflows u32".into())
            })?,
        ))
    }

    #[allow(clippy::cognitive_complexity)]
    fn rollout_ledger_payload(
        &self,
        step: u64,
        controls: &RolloutLedgerControls,
        collected: &[CollectedGroup],
        post_rollout_sampler_state: Vec<u8>,
        window_tokens: u64,
        live_items: u32,
    ) -> Result<RolloutLedgerStep, TrainerError> {
        let mut groups = Vec::with_capacity(collected.len());
        let mut local_live_items = 0_u32;
        let beta = f64::from_bits(controls.effective_beta_bits);
        for item in collected {
            let completion_lens: Vec<u32> = item
                .rollout
                .completion_lens
                .iter()
                .map(|&len| {
                    u32::try_from(len).map_err(|_| {
                        TrainerError::Contract(
                            "completion length does not fit rollout ledger u32".into(),
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            if beta > 0.0 || item.surrogate_live {
                local_live_items = local_live_items.checked_add(1).ok_or_else(|| {
                    TrainerError::Contract("rollout ledger live item count overflow".into())
                })?;
            }
            groups.push(RolloutLedgerGroup {
                accum_index: item.accum_index,
                prompt_index: item.prompt_index,
                rollout_global_row_base: item.rollout_global_row_base,
                token_ids: item.rollout.token_ids.clone(),
                prompt_len: u32::try_from(item.rollout.prompt_len).map_err(|_| {
                    TrainerError::Contract("prompt length does not fit rollout ledger u32".into())
                })?,
                completion_lens,
                behavior_logprob_bits: item.rollout.rollout_logprobs.as_ref().map(|rows| {
                    rows.iter()
                        .map(|row| row.iter().map(|value| value.to_bits()).collect())
                        .collect()
                }),
                reward_bits: item.rewards.iter().map(|value| value.to_bits()).collect(),
                distributed_reward_stats: item.distributed_reward_stats,
                advantage_bits: item
                    .advantages
                    .iter()
                    .map(|&value| (value as f32).to_bits())
                    .collect(),
                loss_mask: item
                    .mask_rows
                    .iter()
                    .map(|row| row.iter().map(|&value| u8::from(value > 0.0)).collect())
                    .collect(),
            });
        }
        let rank = u32::try_from(self.comm.rank()).map_err(|_| {
            TrainerError::Contract("data-parallel rank does not fit rollout ledger u32".into())
        })?;
        let world_size = u32::try_from(self.comm.world_size()).map_err(|_| {
            TrainerError::Contract(
                "data-parallel world size does not fit rollout ledger u32".into(),
            )
        })?;
        Ok(RolloutLedgerStep {
            step,
            rank,
            world_size,
            grad_accum_steps: controls.grad_accum_steps,
            group_size: controls.group_size,
            completion_width: controls.completion_width,
            reward_group_scope: controls.reward_group_scope,
            scale_rewards: controls.scale_rewards,
            eos_token_id: controls.eos_token_id,
            truncation_masking: controls.truncation_masking,
            tis_imp_ratio_cap_bits: controls.tis_imp_ratio_cap_bits,
            effective_lr_bits: controls.effective_lr_bits,
            effective_beta_bits: controls.effective_beta_bits,
            window_tokens,
            live_items,
            old_logprobs: if local_live_items > 0 {
                LedgerScoreRequirement::AdapterEnabledDetached
            } else {
                LedgerScoreRequirement::NotRequired
            },
            reference_logprobs: if beta > 0.0 {
                LedgerScoreRequirement::AdapterDisabledDetached
            } else {
                LedgerScoreRequirement::NotRequired
            },
            post_rollout_sampler_state,
            groups,
        })
    }

    #[allow(clippy::cognitive_complexity)]
    fn collected_group_from_ledger(
        &self,
        group: &RolloutLedgerGroup,
    ) -> Result<CollectedGroup, TrainerError> {
        let completion_lens: Vec<usize> = group
            .completion_lens
            .iter()
            .map(|&len| usize::try_from(len))
            .collect::<Result<_, _>>()
            .map_err(|_| {
                TrainerError::Contract("ledger completion length overflows usize".into())
            })?;
        let rollout_logprobs = group.behavior_logprob_bits.as_ref().map(|rows| {
            rows.iter()
                .map(|row| row.iter().map(|&bits| f32::from_bits(bits)).collect())
                .collect()
        });
        let rollout = Rollout::new(
            group.token_ids.clone(),
            usize::try_from(group.prompt_len).map_err(|_| {
                TrainerError::Contract("ledger prompt length overflows usize".into())
            })?,
            completion_lens,
            rollout_logprobs,
        );
        let rewards: Vec<f32> = group
            .reward_bits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect();
        let advantages: Vec<f64> = group
            .advantage_bits
            .iter()
            .map(|&bits| f64::from(f32::from_bits(bits)))
            .collect();
        let surrogate_live = advantages.iter().any(|&value| value != 0.0);
        let mask_rows: Vec<Vec<f64>> = group
            .loss_mask
            .iter()
            .map(|row| row.iter().map(|&value| f64::from(value)).collect())
            .collect();
        let comp_len = self.config.max_new_tokens;
        let truncated = if self.config.truncation_masking {
            self.config.eos_token_id.map_or(0, |eos| {
                rollout
                    .completion_lens
                    .iter()
                    .zip(&rollout.token_ids)
                    .filter(|(len, ids)| {
                        **len == comp_len && ids[rollout.prompt_len + **len - 1] != eos
                    })
                    .count()
            })
        } else {
            0
        };
        let stat = PromptStat {
            rewards: rewards.clone(),
            completion_len: mean_completion_len(&rollout),
            completion_tokens: rollout.completion_lens.iter().sum(),
            dropped: zero_mask_rows(&mask_rows),
            truncated,
            degenerate: !surrogate_live,
            ratio_stats: None,
        };
        Ok(CollectedGroup {
            accum_index: group.accum_index,
            prompt_index: group.prompt_index,
            rollout_global_row_base: group.rollout_global_row_base,
            rollout,
            rewards,
            advantages,
            distributed_reward_stats: group.distributed_reward_stats,
            mask_rows,
            stat,
            surrogate_live,
        })
    }

    /// Run `config.steps` optimizer steps — each over a window of
    /// `config.grad_accum_steps` samples — cycling through `samples`, returning one
    /// [`Metrics`] row per optimizer step (also appended to `metrics.jsonl`).
    ///
    /// Returns the per-step metrics **and a [`RunStop`]**: if a
    /// [`with_preemption_flag`](Self::with_preemption_flag) fired the loop stops early
    /// after checkpointing and the stop is [`RunStop::Preempted`] (the history is then
    /// *partial* — do not treat it as a finished run), else [`RunStop::Completed`].
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] if any forward, optimizer step, telemetry write,
    /// or the grad-coverage canary fails. A canary failure aborts the run.
    ///
    /// # Panics
    ///
    /// Panics if `samples` is empty: a run with no data is a caller bug.
    pub fn train<P: Policy, R: RewardFn>(
        &mut self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.run(0, None, policy, reward_fn, tokenizer, samples, &exec)
    }

    /// Run training through a policy's explicit tensor-parallel rollout and
    /// scoring hooks.
    ///
    /// `tensor_parallel_comm` is separate from the trainer's data-parallel
    /// communicator installed by [`with_comm`](Self::with_comm). A sharded TP
    /// communicator is supported only when the trainer's DP communicator is
    /// world-1 and the policy advertises a complete sharded backward: `LoRA`
    /// trainable vars remain fully replicated, their accumulated
    /// gradients are sum-reduced over the TP communicator before coverage,
    /// clipping, optimizer step, and checkpointing, and TP rank 0 owns the shared
    /// checkpoint/candidate/metrics side effects.
    ///
    /// # Errors
    ///
    /// As [`train`](Self::train), plus a fail-closed contract error when both
    /// data parallelism and tensor parallelism are sharded or the policy has
    /// only forward-value TP semantics.
    pub fn train_tensor_parallel<P: TensorParallelPolicy, R: RewardFn>(
        &mut self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        self.validate_tensor_parallel_backward(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.run(0, None, policy, reward_fn, tokenizer, samples, &exec)
    }

    /// Resume training from `start_step`, running steps `start_step .. config.steps`
    /// (so the total run still ends at `config.steps`). Returns the per-step
    /// [`Metrics`] for the steps actually executed (empty if `start_step >=
    /// config.steps`) **and a [`RunStop`]** (as [`train`](Self::train)); they are
    /// also **appended** to `metrics.jsonl`, continuing the prior run's stream.
    ///
    /// The caller must have loaded the checkpoint's adapter into `policy` first —
    /// [`crate::checkpoint::load_adapter`] returns the [`crate::checkpoint::CheckpointManifest`]
    /// whose `step` is the `start_step` to pass here. Prompt cycling keys off the
    /// window index — window `start_step` consumes prompts beginning at
    /// `start_step * grad_accum_steps` (mod len) — so resuming at the recorded window
    /// continues the prompt order an uninterrupted run would have seen.
    ///
    /// **Not momentum-faithful.** This continues from `start_step` with a **fresh**
    /// [`FerrlAdamW`] (its moments restart from zero, re-warming the bias correction)
    /// and whatever sampler RNG the reloaded policy carries — the caller is expected to
    /// have loaded only the adapter (e.g. via [`crate::checkpoint::load_adapter`]). The
    /// reloaded *adapter weights* are exact; the post-resume trajectory is a faithful
    /// continuation, not a bit-exact replay. For a **bit-exact** resume that also
    /// restores the optimizer moments and the sampler RNG from a momentum-faithful (v2)
    /// checkpoint, use [`resume`](Self::resume) instead.
    ///
    /// # Errors
    ///
    /// As [`train`](Self::train).
    ///
    /// # Panics
    ///
    /// Panics if `samples` is empty.
    pub fn train_from<P: Policy, R: RewardFn>(
        &mut self,
        start_step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.run(
            start_step, None, policy, reward_fn, tokenizer, samples, &exec,
        )
    }

    /// Resume from `start_step` while routing rollout and scoring through an
    /// explicit tensor-parallel communicator.
    ///
    /// # Errors
    ///
    /// As [`train_from`](Self::train_from), plus the tensor-parallel validation
    /// described on [`train_tensor_parallel`](Self::train_tensor_parallel).
    pub fn train_from_tensor_parallel<P: TensorParallelPolicy, R: RewardFn>(
        &mut self,
        start_step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        self.validate_tensor_parallel_backward(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.run(
            start_step, None, policy, reward_fn, tokenizer, samples, &exec,
        )
    }

    /// Resume an interrupted run from a checkpoint directory, **momentum-faithfully**.
    ///
    /// Loads the checkpoint's adapter into `policy`, restores the optimizer moments and
    /// the rollout-sampler RNG, and continues from the recorded step — so on the same
    /// machine the post-resume trajectory is **bit-identical** to an uninterrupted run
    /// (pinned by the toy gate `interrupted_run_resumes_bit_identically`). For a v1
    /// (adapter-only) checkpoint there is no optimizer/sampler state to restore, so this
    /// falls back to a fresh [`FerrlAdamW`] and the policy's current sampler (a faithful
    /// continuation, not a bit-exact replay — exactly like [`train_from`](Self::train_from)).
    ///
    /// Returns the per-step metrics **and a [`RunStop`]** (as [`train`](Self::train)).
    /// This is the **explicit-path** data-parallel resume — every rank calls it with
    /// rank 0's checkpoint dir (the launcher supplies the path) — so the caller MUST
    /// honor [`RunStop::Preempted`] (skip held-out eval / gating on the partial history
    /// and exit so the next requeue continues). The **auto-discovery** counterpart,
    /// [`resume_latest`](Self::resume_latest), also works under data parallelism: it
    /// coordinates rank 0's discovery across the world for you, given a shared
    /// checkpoint directory ([`with_checkpoints_dir`](Self::with_checkpoints_dir)).
    ///
    /// `policy` must be the same architecture AND adapter recipe the checkpoint was
    /// written from: the adapter load and the optimizer-moment load each validate
    /// count/shape/dtype, the manifest's `lora_recipe` string is cross-checked against
    /// [`Policy::lora_recipe`] **before** any model state is touched (count/shape/dtype
    /// cannot distinguish shape-aliased recipes — e.g. `attn:qk` vs `attn:qv`, whose
    /// k/v projections are shape-identical — so a recipe swap would otherwise restore
    /// adapters onto the wrong projections silently), and a malformed sampler blob
    /// fails loud too. A checkpoint or policy without a recorded recipe skips the
    /// recipe check (pre-recipe checkpoints stay loadable).
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] if the checkpoint cannot be read or does not match
    /// `policy` (count/shape/dtype or adapter recipe), or if a training step fails
    /// (as [`train`](Self::train)).
    ///
    /// # Panics
    ///
    /// Panics if `samples` is empty.
    pub fn resume<P: Policy, R: RewardFn>(
        &mut self,
        checkpoint_dir: impl AsRef<Path>,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        let exec = UnshardedPolicyExecution;
        let loaded = self.load_resume_point(checkpoint_dir.as_ref(), policy);
        let (start_step, opt_state) = self.coordinate_resume_load::<P, _>(&exec, loaded)?;
        self.run(
            start_step, opt_state, policy, reward_fn, tokenizer, samples, &exec,
        )
    }

    /// Momentum-faithful resume through an explicit tensor-parallel
    /// communicator.
    ///
    /// # Errors
    ///
    /// As [`resume`](Self::resume), plus the tensor-parallel validation
    /// described on [`train_tensor_parallel`](Self::train_tensor_parallel).
    pub fn resume_tensor_parallel<P: TensorParallelPolicy, R: RewardFn>(
        &mut self,
        checkpoint_dir: impl AsRef<Path>,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        self.validate_tensor_parallel_backward(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        let loaded = self.load_resume_point(checkpoint_dir.as_ref(), policy);
        let (start_step, opt_state) = self.coordinate_resume_load::<P, _>(&exec, loaded)?;
        self.run(
            start_step, opt_state, policy, reward_fn, tokenizer, samples, &exec,
        )
    }

    /// Load `checkpoint_dir` into `policy` and return the resumed loop's
    /// `(start_step, optimizer_state)` — the momentum-faithful prelude shared by
    /// [`resume`](Self::resume) and [`resume_latest`](Self::resume_latest).
    ///
    /// Recipe pre-flight BEFORE the positional load mutates the live vars:
    /// count/shape/dtype validation cannot distinguish shape-aliased recipes
    /// (k/v and gate/up projections are shape-identical), so a mismatch here
    /// would land trained adapters on the wrong projections silently. Then the
    /// positional adapter + optimizer-moment load, then the rollout-sampler RNG
    /// restore (a v1 checkpoint carries none → keep the policy's current sampler,
    /// the documented fresh-momentum fallback).
    fn load_resume_point<P: Policy>(
        &self,
        checkpoint_dir: &Path,
        policy: &mut P,
    ) -> Result<(u64, Option<OptimizerState>), TrainerError> {
        let manifest = crate::checkpoint::read_manifest(checkpoint_dir)?;
        if let (Some(saved), Some(current)) = (&manifest.lora_recipe, policy.lora_recipe()) {
            if *saved != current {
                return Err(TrainerError::Checkpoint(
                    crate::checkpoint::CheckpointError::Mismatch(format!(
                        "checkpoint adapter recipe {saved:?} does not match the policy's \
                         {current:?} (the positional load cannot catch a shape-aliased \
                         recipe swap — load with the recipe the checkpoint records)"
                    )),
                ));
            }
        }
        let vars = policy.trainable_vars();
        let loaded = crate::checkpoint::load_checkpoint(checkpoint_dir, &vars)?;
        if let Some(blob) = &loaded.sampler_state {
            policy.restore_sampler_state(blob)?;
        }
        Ok((loaded.step, loaded.optimizer_state))
    }

    /// Resume the **newest complete checkpoint** under this run's `checkpoints/`
    /// directory, or start a **fresh** run if there is none — the launch entry
    /// point for a job that may be (re)started repeatedly, e.g. a Slurm run that is
    /// preempted or times out and is requeued.
    ///
    /// On each (re)launch this scans `checkpoints/` ([`crate::latest_checkpoint`]):
    /// if a checkpoint exists it resumes **momentum-faithfully** exactly as
    /// [`resume`](Self::resume) (same recipe / architecture requirements on
    /// `policy`); otherwise it runs from scratch exactly as [`train`](Self::train).
    /// To make requeues *continue* rather than start over, pair this with a
    /// **stable `run_id`** reused across launches via
    /// [`RunDir::open`](crate::RunDir::open) — a fresh `run_id` each launch would
    /// always find an empty `checkpoints/` and start from zero. Combine with
    /// [`with_preemption_flag`](Self::with_preemption_flag) so each attempt also
    /// flushes a final checkpoint when preempted, minimizing re-done work.
    ///
    /// Returns the run's metrics **and a [`RunStop`]**: on [`RunStop::Preempted`]
    /// the history is *partial* (the flag stopped the loop early after
    /// checkpointing) and the launcher must exit before any held-out eval / final
    /// gate so the requeue resumes; on [`RunStop::Completed`] the run finished and
    /// eval / gating may proceed.
    ///
    /// **Auto-resume is coordinated through execution rank 0**, not scanned per
    /// rank. Only that rank writes checkpoints, so a naive per-rank scan would
    /// have non-zero ranks find none, start fresh, and diverge from rank 0 (then
    /// deadlock the next collective once rank 0 finishes its shorter remaining
    /// steps). Instead, under `world_size > 1`, execution rank 0 scans the
    /// checkpoint directory and **broadcasts the resume step** to every rank, so
    /// the whole world resumes from rank 0's checkpoint — or all start fresh —
    /// **in lockstep**, then each rank loads it via the same momentum-faithful
    /// path the explicit `resume(&rank0_ckpt)` requeue uses. This requires every
    /// rank to point at **one shared checkpoint directory** (a filesystem all
    /// ranks read: a single node's `/tmp`, or NFS across nodes) via
    /// [`with_checkpoints_dir`](Self::with_checkpoints_dir) — rank 0 writes it,
    /// all read it. The broadcast is a single scalar collective (rank 0's value
    /// summed against zero-contributing peers), so [`Comm`] needs no dedicated
    /// broadcast primitive.
    ///
    /// # Errors
    ///
    /// As [`resume`](Self::resume) (when a checkpoint is found) or
    /// [`train`](Self::train) (when none is), plus [`TrainerError::Comm`] if the
    /// rank-0 broadcast coordinating the decision fails (a peer aborted or the world
    /// is poisoned). If rank 0's checkpoint scan itself fails (the directory exists
    /// but cannot be listed), the failure is broadcast so every rank aborts **in
    /// lockstep** — rank 0 returns [`TrainerError::Checkpoint`] (the underlying IO
    /// error), the other ranks [`TrainerError::Contract`] — rather than stranding the
    /// peers in the collective until the timeout.
    ///
    /// # Panics
    ///
    /// Panics if `samples` is empty.
    pub fn resume_latest<P: Policy, R: RewardFn>(
        &mut self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.resume_latest_with_execution(policy, reward_fn, tokenizer, samples, &exec)
    }

    /// Auto-resume the newest checkpoint while routing rollout and scoring
    /// through an explicit tensor-parallel communicator.
    ///
    /// # Errors
    ///
    /// As [`resume_latest`](Self::resume_latest), plus the tensor-parallel
    /// validation described on
    /// [`train_tensor_parallel`](Self::train_tensor_parallel).
    pub fn resume_latest_tensor_parallel<P: TensorParallelPolicy, R: RewardFn>(
        &mut self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        tensor_parallel_comm: &dyn Comm,
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError> {
        self.validate_tensor_parallel_comm(tensor_parallel_comm)?;
        self.validate_tensor_parallel_policy_execution(policy, tensor_parallel_comm)?;
        self.validate_tensor_parallel_backward(policy, tensor_parallel_comm)?;
        let exec = TensorParallelPolicyExecution {
            comm: tensor_parallel_comm,
        };
        self.resume_latest_with_execution(policy, reward_fn, tokenizer, samples, &exec)
    }

    #[allow(clippy::too_many_arguments)]
    fn resume_latest_with_execution<P, R, E>(
        &mut self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        exec: &E,
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError>
    where
        P: Policy,
        R: RewardFn,
        E: PolicyExecution<P>,
    {
        // Discover the resume point through rank 0 and broadcast it so the whole world
        // branches identically in lockstep (see `coordinate_resume_step`).
        match self.coordinate_resume_step(exec)? {
            Some(step) => {
                // The canonical on-disk layout is `checkpoints_dir/step-<n>` (see
                // `write_checkpoint`), so every rank reconstructs the identical
                // directory from the broadcast step — rank 0's checkpoint.
                let dir = self.checkpoints_dir.join(format!("step-{step}"));
                tracing::info!(
                    rank = exec.execution_rank(self.comm.as_ref()),
                    resume_step = step,
                    dir = %dir.display(),
                    world_size = exec.execution_world_size(self.comm.as_ref()),
                    "resume_latest: continuing from the newest checkpoint"
                );
                let loaded = self.load_resume_point(&dir, policy);
                let (start_step, opt_state) = self.coordinate_resume_load::<P, _>(exec, loaded)?;
                self.run(
                    start_step, opt_state, policy, reward_fn, tokenizer, samples, exec,
                )
            }
            None => {
                tracing::info!(
                    rank = exec.execution_rank(self.comm.as_ref()),
                    world_size = exec.execution_world_size(self.comm.as_ref()),
                    "resume_latest: no checkpoint found — starting a fresh run"
                );
                self.run(0, None, policy, reward_fn, tokenizer, samples, exec)
            }
        }
    }

    fn validate_tensor_parallel_comm(&self, comm: &dyn Comm) -> Result<(), TrainerError> {
        crate::tensor_parallel::plan_from_comm(comm)?;
        if comm.world_size() > 1 && self.comm.world_size() > 1 {
            return Err(TrainerError::Contract(
                "simultaneous sharded data-parallel and tensor-parallel trainer execution is \
                 not wired yet; use a world-1 trainer communicator with sharded tensor \
                 parallelism until the combined DP×TP optimizer/checkpoint contract lands"
                    .into(),
            ));
        }
        Ok(())
    }

    fn validate_tensor_parallel_policy_execution<P: TensorParallelPolicy>(
        &self,
        policy: &P,
        comm: &dyn Comm,
    ) -> Result<(), TrainerError> {
        let local = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            policy
                .validate_tensor_parallel_execution(comm)
                .map_err(TrainerError::from)
        }))
        .unwrap_or_else(|payload| {
            Err(TrainerError::Contract(format!(
                "tensor-parallel policy execution preflight panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let execution_comm = if comm.world_size() > 1 {
            comm
        } else {
            self.comm.as_ref()
        };
        Self::coordinate_comm_result(
            execution_comm,
            "tensor-parallel policy execution preflight",
            local,
        )
    }

    fn validate_tensor_parallel_backward<P: TensorParallelPolicy>(
        &self,
        policy: &P,
        comm: &dyn Comm,
    ) -> Result<(), TrainerError> {
        if comm.world_size() <= 1 {
            return Ok(());
        }
        let supported =
            Self::coordinate_comm_call(comm, "tensor-parallel backward capability probe", || {
                Ok(policy.supports_sharded_tensor_parallel_backward())
            })?;
        let unsupported_local = if supported { 0.0 } else { 1.0 };
        let unsupported = comm.all_reduce_scalar_sum(unsupported_local)?;
        if unsupported > 0.0 {
            return Err(TrainerError::Contract(
                "sharded tensor-parallel training requires a policy with cross-rank backward \
                 semantics; forward-only TP policies are supported for rollout/scoring but \
                 must fail closed before training"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Coordinate the resume decision across the active execution world: execution
    /// rank 0 scans for the newest checkpoint and **broadcasts** the outcome so
    /// every DP or TP rank agrees in lockstep — see
    /// [`resume_latest`](Self::resume_latest). Returns the resume step (`Some`) or
    /// a fresh start (`None`).
    ///
    /// The outcome is three-way — found / none / scan-FAILED — and all three ride the
    /// one broadcast every rank enters. In particular a rank-0 scan failure must NOT
    /// `?`-return *before* that broadcast: a rank-0-only early return would strand every
    /// peer in the collective until the timeout. So the failure is broadcast and
    /// surfaced as an error on **every** rank in lockstep — rank 0 the real IO error,
    /// peers a synthesized [`TrainerError::Contract`].
    fn coordinate_resume_step<P, E>(&self, exec: &E) -> Result<Option<u64>, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let (local, rank0_scan_err) =
            self.scan_local_resume(exec.is_execution_primary(self.comm.as_ref()));
        // Execution rank 0 contributes its decision; peers contribute `Fresh`
        // (the additive identity), so the rank-identical sum decodes to rank 0's
        // decision on every rank. (`Comm` has no broadcast; this is
        // broadcast-from-rank-0 via the sum all-reduce — see `ResumeDecision`.)
        // Every rank enters this collective.
        let decision = if exec.execution_world_size(self.comm.as_ref()) > 1 {
            ResumeDecision::decode(
                exec.execution_all_reduce_scalar_sum(self.comm.as_ref(), local.encode())?,
            )
        } else {
            local
        };
        match decision {
            ResumeDecision::Fresh => Ok(None),
            ResumeDecision::Resume(step) => Ok(Some(step)),
            ResumeDecision::ScanFailed => Err(rank0_scan_err.map_or_else(
                || {
                    TrainerError::Contract(
                        "resume_latest: execution rank 0's checkpoint discovery failed; the \
                         resume aborted in lockstep on every rank"
                            .into(),
                    )
                },
                TrainerError::Checkpoint,
            )),
        }
    }

    /// Execution rank 0's local resume scan: the [`ResumeDecision`] to broadcast
    /// plus rank 0's real scan error, if any. Non-primary ranks skip the scan and
    /// contribute the additive identity ([`ResumeDecision::Fresh`]) — only rank
    /// 0's scan is authoritative.
    fn scan_local_resume(
        &self,
        is_primary: bool,
    ) -> (ResumeDecision, Option<crate::checkpoint::CheckpointError>) {
        if !is_primary {
            return (ResumeDecision::Fresh, None);
        }
        match crate::checkpoint::latest_checkpoint(&self.checkpoints_dir) {
            Ok(found) => (
                found.map_or(ResumeDecision::Fresh, |latest| {
                    ResumeDecision::Resume(latest.step)
                }),
                None,
            ),
            Err(e) => (ResumeDecision::ScanFailed, Some(e)),
        }
    }

    /// After a coordinated resume decision, every execution rank must also agree
    /// that the local checkpoint load/restore prelude succeeded before any rank
    /// enters the next training collective. Rank-local filesystem, tensor, or
    /// sampler-restore failures are therefore reduced into one lockstep abort.
    fn coordinate_resume_load<P, E>(
        &self,
        exec: &E,
        local: Result<(u64, Option<OptimizerState>), TrainerError>,
    ) -> Result<(u64, Option<OptimizerState>), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        if exec.execution_world_size(self.comm.as_ref()) <= 1 {
            return local;
        }
        if local
            .as_ref()
            .err()
            .is_some_and(Self::is_terminal_distributed_error)
        {
            return local;
        }
        let failed_local = if local.is_err() { 1.0 } else { 0.0 };
        let failed_global =
            exec.execution_all_reduce_scalar_sum(self.comm.as_ref(), failed_local)?;
        match local {
            Err(error) => Err(error),
            Ok(_) if failed_global > 0.0 => Err(TrainerError::Contract(
                "checkpoint load/restore failed on a peer rank; aborting in lockstep".into(),
            )),
            Ok(loaded) => Ok(loaded),
        }
    }

    /// Shared loop for [`train`](Self::train) / [`train_from`](Self::train_from):
    /// run optimizer steps `start_step .. config.steps`, each consuming a window of
    /// `grad_accum_steps` samples, checkpointing on the configured cadence.
    #[allow(clippy::too_many_arguments)]
    fn run<P, R, E>(
        &mut self,
        start_step: u64,
        resume_opt_state: Option<OptimizerState>,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        exec: &E,
    ) -> Result<(Vec<Metrics>, RunStop), TrainerError>
    where
        P: Policy,
        R: RewardFn,
        E: PolicyExecution<P>,
    {
        assert!(!samples.is_empty(), "train: no samples");
        // Stamp every event this run emits — the per-step events below, plus anything
        // the policy/reward logs — with this rank's rank/world. Under DP all ranks share
        // one stdout, so an unstamped line is unattributable; a nested per-step `step`
        // span is entered inside the loop. All four entry points funnel through `run`,
        // so the stamp covers train / train_from / resume / resume_latest alike.
        let _run = crate::telemetry::run_span(
            exec.execution_rank(self.comm.as_ref()),
            exec.execution_world_size(self.comm.as_ref()),
        )
        .entered();
        // The KL reference (`beta > 0`) IS the adapter-disabled policy
        // (`reference_logprobs` toggles the adapter off to score it). A policy
        // that cannot disable its adapter — full fine-tuning: the base weights
        // ARE the trained weights — would silently make `logp_ref` the live
        // policy itself: bit-identical to `logp_old`, the KL-to-base penalty
        // degenerating to a window-anchored proximal term that reports a
        // near-zero `kl` metric and pulls toward nothing. Fail loud instead:
        // full-FT runs take `beta = 0` (no frozen reference exists to pull
        // toward; a base-anchored KL needs a separately loaded base policy,
        // which this trainer does not model).
        self.require_toggleable_reference_policy(policy, self.config.requires_reference_policy())?;
        let vars = policy.trainable_vars();
        let params = ParamsAdamW {
            lr: self.config.lr,
            weight_decay: self.config.weight_decay,
            beta1: self.config.adam_beta1,
            beta2: self.config.adam_beta2,
            ..Default::default()
        };
        let mut opt = FerrlAdamW::new(vars.clone(), params)?;
        // Momentum-faithful resume: restore the optimizer moments + step counter before
        // the first step, so the bias correction and Adam state continue exactly where
        // the interrupted run left off (validated against `vars` inside `load_state`).
        if let Some(state) = resume_opt_state {
            opt.load_state(&state)?;
        }
        let total = self.config.steps;
        let remaining = total.saturating_sub(start_step) as usize;
        let mut history = Vec::with_capacity(remaining);
        for step in start_step..total {
            // Nest a per-step span under the run span so every event in this iteration
            // (the trainer's, the policy's) also carries `step` — rank/world/step on
            // every line. A helper, not an inline `info_span!`, to keep the macro's
            // level-check branch out of this loop's cognitive-complexity budget.
            let _step = crate::telemetry::step_span(step).entered();
            // Step-owned scalar controls are pure functions of the step index, so
            // a resume re-enters schedule or warmup state exactly.
            let step_lr = self.config.lr_at(step);
            let step_beta = self.config.beta_at(step);
            opt.set_learning_rate(step_lr);
            // Wall-clock the whole step (rollout + reward + the mu inner update
            // epochs) so a long run is observable: step_secs drives steps/sec and
            // an ETA, tokens_per_sec the rollout throughput. Per-rank — the world
            // figure is world_size × tokens_per_sec (each rank rolls out its own
            // prompt shard); see `telemetry::Metrics`.
            let started = std::time::Instant::now();
            let mut gpu_mem = StepGpuMemory::new(self.config.gpu_memory_probe);
            gpu_mem.record("step_start");
            let (mut m, local_tokens) = self.run_window(
                step,
                step_beta,
                policy,
                reward_fn,
                tokenizer,
                samples,
                &mut opt,
                &vars,
                &mut gpu_mem,
                exec,
            )?;
            gpu_mem.record("step_end");
            let secs = started.elapsed().as_secs_f64();
            m.step_secs = secs as f32;
            m.tokens_per_sec = step_throughput(local_tokens, secs);
            gpu_mem.apply(&mut m);
            self.append_metrics(&m, exec)?;
            history.push(m);
            self.maybe_checkpoint(step, &vars, &opt, policy, exec)?;
            // Cooperative preemption stop (globalized across the DP world): on a
            // Slurm preempt / timeout grace signal the run binary flips the flag,
            // and we save a final checkpoint at this completed step and stop
            // cleanly so a requeued run picks up here via `resume_latest`. (May
            // re-write the cadence checkpoint just written above — idempotent.)
            // The poll itself runs every step on every rank (lockstep), but the
            // stop-early decision keys on `completed`/`total`, which are identical
            // across ranks — so the whole world stops or continues together.
            if self.preempt_requested(exec)? {
                let completed = step + 1;
                // A preemption that arrives only after the FINAL step is moot: the
                // loop already ran every configured step, so the run is Completed.
                // Stopping early here (writing a step==total checkpoint and
                // returning Preempted) would make the launcher skip held-out
                // eval/gate, and a requeue would then resume_latest from that
                // checkpoint, run ZERO steps, and gate on an EMPTY history. Only
                // stop early — and report Preempted — when work actually remains.
                if completed < total {
                    self.write_checkpoint(completed, &vars, &opt, policy, exec)?;
                    tracing::warn!(
                        completed_steps = completed,
                        "preemption requested: checkpointed and stopping early"
                    );
                    return Ok((history, RunStop::Preempted));
                }
            }
        }
        Ok((history, RunStop::Completed))
    }

    fn append_metrics<P, E>(&mut self, metrics: &Metrics, exec: &E) -> Result<(), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        self.coordinate_side_effect(exec, "metrics write", |trainer| {
            if exec.writes_rank_local_telemetry(trainer.comm.as_ref()) {
                trainer.writer.append(metrics)?;
            }
            Ok(())
        })
    }

    /// Append one separated-learner metrics row transactionally across every
    /// rank-local run directory. If any append fails, all successful peers
    /// truncate back to their pre-append boundary before the learner rolls its
    /// model/Adam/sampler state back.
    fn append_rollout_ledger_metrics(&mut self, metrics: &Metrics) -> Result<(), TrainerError> {
        if self.comm.world_size() <= 1 {
            self.writer.append(metrics)?;
            return Ok(());
        }
        let rollback_len = self.writer.append_boundary().map_err(TrainerError::from);
        let rollback_len = self.coordinate_data_parallel_result(
            "rollout-ledger metrics rollback boundary",
            rollback_len,
        )?;
        let append_local = self.writer.append(metrics).map_err(TrainerError::from);
        let failed_local = if append_local.is_err() { 1.0 } else { 0.0 };
        let failed_global = self.comm.all_reduce_scalar_sum(failed_local);
        let failed_global = match failed_global {
            Ok(failed) => failed,
            Err(comm_error) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local rollout-ledger metrics rollback",
                    || {
                        self.writer.truncate_to(rollback_len)?;
                        Ok(())
                    },
                );
                return Err(TrainerError::RolloutLedgerMetricsComm {
                    communication: Box::new(comm_error),
                    telemetry_rollback: Self::rollout_ledger_metrics_rollback_detail(&rollback),
                });
            }
        };
        if failed_global == 0.0 {
            return append_local;
        }
        let rollback_local =
            Self::catch_local_distributed_recovery("rollout-ledger metrics rollback", || {
                self.writer.truncate_to(rollback_len)?;
                Ok(())
            });
        let rollback_detail = Self::rollout_ledger_metrics_rollback_detail(&rollback_local);
        let rollback =
            self.coordinate_data_parallel_result("rollout-ledger metrics rollback", rollback_local);
        match (append_local, rollback) {
            (_, Err(TrainerError::Comm(comm_error))) => {
                Err(TrainerError::RolloutLedgerMetricsComm {
                    communication: Box::new(comm_error),
                    telemetry_rollback: rollback_detail,
                })
            }
            (_, Err(rollback_error)) => Err(TrainerError::Contract(format!(
                "rollout-ledger metrics append failed on at least one rank; coordinated rollback failed: {rollback_error}"
            ))),
            (Err(append_error), Ok(())) => Err(append_error),
            (Ok(()), Ok(())) => Err(TrainerError::Contract(
                "rollout-ledger metrics append failed on a peer rank; every rank-local stream was rolled back"
                    .into(),
            )),
        }
    }

    fn append_rollout_ledger_metrics_with_execution(
        &mut self,
        metrics: &Metrics,
        execution_comm: &dyn Comm,
        is_execution_primary: bool,
        primary_only: bool,
    ) -> Result<(), TrainerError> {
        if !primary_only {
            return self.append_rollout_ledger_metrics(metrics);
        }

        let rollback_len_local = if is_execution_primary {
            self.writer.append_boundary().map_err(TrainerError::from)
        } else {
            Ok(0)
        };
        let rollback_len = Self::coordinate_comm_result(
            execution_comm,
            "tensor-parallel rollout-ledger metrics rollback boundary",
            rollback_len_local,
        )?;
        let append_local = if is_execution_primary {
            self.writer.append(metrics).map_err(TrainerError::from)
        } else {
            Ok(())
        };
        let failed_global =
            execution_comm.all_reduce_scalar_sum(if append_local.is_err() { 1.0 } else { 0.0 });
        let failed_global = match failed_global {
            Ok(failed) => failed,
            Err(comm_error) => {
                let rollback = Self::catch_local_distributed_recovery(
                    "best-effort local tensor-parallel metrics rollback",
                    || {
                        if is_execution_primary {
                            self.writer.truncate_to(rollback_len)?;
                        }
                        Ok(())
                    },
                );
                return Err(TrainerError::RolloutLedgerMetricsComm {
                    communication: Box::new(comm_error),
                    telemetry_rollback: Self::rollout_ledger_metrics_rollback_detail(&rollback),
                });
            }
        };
        if failed_global == 0.0 {
            return append_local;
        }

        let rollback_local = Self::catch_local_distributed_recovery(
            "tensor-parallel rollout-ledger metrics rollback",
            || {
                if is_execution_primary {
                    self.writer.truncate_to(rollback_len)?;
                }
                Ok(())
            },
        );
        let rollback_detail = Self::rollout_ledger_metrics_rollback_detail(&rollback_local);
        let rollback = Self::coordinate_comm_result(
            execution_comm,
            "tensor-parallel rollout-ledger metrics rollback",
            rollback_local,
        );
        match (append_local, rollback) {
            (_, Err(TrainerError::Comm(comm_error))) => {
                Err(TrainerError::RolloutLedgerMetricsComm {
                    communication: Box::new(comm_error),
                    telemetry_rollback: rollback_detail,
                })
            }
            (_, Err(rollback_error)) => Err(TrainerError::Contract(format!(
                "tensor-parallel rollout-ledger metrics append failed; coordinated rollback failed: {rollback_error}"
            ))),
            (Err(append_error), Ok(())) => Err(append_error),
            (Ok(()), Ok(())) => Err(TrainerError::Contract(
                "tensor-parallel rollout-ledger metrics append failed on execution rank 0; its stream was rolled back"
                    .into(),
            )),
        }
    }

    /// One optimizer step over a window of `grad_accum_steps` prompts: collect each
    /// prompt's group (rollout → reward → advantages, snapshotting the non-degenerate
    /// ones), then run the `mu` inner epochs that accumulate the window's gradients
    /// into a single `AdamW` update. Returns the window's aggregated [`Metrics`] and
    /// this rank's real completion-token count for the window (the throughput
    /// numerator the caller divides by the step wall-time — local, not world-summed).
    #[allow(clippy::too_many_arguments)]
    fn run_window<P, R, E>(
        &mut self,
        step: u64,
        beta: f64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        samples: &[Sample<R::Target>],
        opt: &mut FerrlAdamW,
        vars: &[Var],
        gpu_mem: &mut StepGpuMemory,
        exec: &E,
    ) -> Result<(Metrics, usize), TrainerError>
    where
        P: Policy,
        R: RewardFn,
        E: PolicyExecution<P>,
    {
        gpu_mem.record("window_start");
        let accum = self.config.grad_accum_steps;
        let world = self.comm.world_size();
        let mut stats = Vec::with_capacity(accum);
        let mut live = Vec::with_capacity(accum);
        for j in 0..accum {
            let sel = self.select_prompt(step, j, samples.len());
            self.record_prompt_selection(&sel);
            let selected = SelectedSample {
                sample: &samples[sel.sample_idx],
                selection: &sel,
                accum_index: j,
            };
            let (stat, item) = self.collect_sample(
                step, beta, policy, reward_fn, tokenizer, &selected, gpu_mem, exec,
            )?;
            stats.push(stat);
            if let Some(item) = item {
                live.push(item);
            }
        }
        // The DAPO loss normalizer: the window's total completion tokens (true
        // EOS-inclusive lengths) over EVERY prompt — degenerate groups and
        // truncation-masked completions included, exactly TRL's
        // `num_items_in_batch` (their masking zeroes the loss mask but the
        // length total is taken from the raw completions). Under DP the
        // normalizer is the GLOBAL window's total (TRL's `num_items_in_batch`
        // is batch-global), summed across ranks BEFORE the inner epochs.
        // Clamped to >= 1 so a pathological all-empty window yields 0, not 0/0
        // (raw local sums reduce first — clamping locally would overcount).
        let local_tokens = stats.iter().map(|s| s.completion_tokens).sum::<usize>();
        let window_tokens = if world > 1 {
            gpu_mem.record("token_count_all_reduce_start");
            self.comm
                .all_reduce_scalar_sum(local_tokens as f64)?
                .max(1.0)
        } else {
            local_tokens.max(1) as f64
        };
        gpu_mem.record("token_count_all_reduce_end");
        // A window with no live prompts (every group degenerate) is a GRPO no-op: no
        // update, no canary — mirroring the single-prompt degenerate skip. Under DP
        // the decision must be GLOBAL: a rank whose local shard is all-degenerate
        // still has to enter the inner epochs' collectives (contributing zeros)
        // while any peer holds live items — a local skip would deadlock the world.
        let n_live_global = if world > 1 {
            gpu_mem.record("live_count_all_reduce_start");
            self.comm.all_reduce_scalar_sum(live.len() as f64)?
        } else {
            live.len() as f64
        };
        gpu_mem.record("live_count_all_reduce_end");
        let agg = if n_live_global == 0.0 {
            InnerAgg::default()
        } else {
            let mut ctx = UpdateCtx { vars, opt, gpu_mem };
            self.update_window(
                policy,
                &live,
                &mut ctx,
                window_tokens,
                n_live_global,
                beta,
                exec,
            )?
        };
        gpu_mem.record("window_end");
        Ok((
            self.build_window_metrics(step, beta, &stats, &agg, opt),
            local_tokens,
        ))
    }

    fn select_prompt(&self, step: u64, j: usize, samples_len: usize) -> PromptSelection {
        let accum = self.config.grad_accum_steps;
        let world = self.comm.world_size();
        let rank = self.comm.rank();
        match self.config.reward_group_scope {
            RewardGroupScope::Local => {
                // Continuous prompt cycling across windows: window `step` consumes
                // the `accum × world` prompts starting at `step*accum*world` (mod
                // len), rank `r` taking the contiguous slice at offset `r*accum`.
                // At world 1 this is the legacy `step*accum + j`.
                let local = rank * accum + j;
                let prompt_index = step * (accum * world) as u64 + local as u64;
                PromptSelection {
                    sample_idx: (step as usize * (accum * world) + local) % samples_len,
                    prompt_index,
                    rollout_global_row_base: prompt_index
                        .wrapping_mul(self.config.group_size as u64),
                }
            }
            RewardGroupScope::DistributedSamePrompt => {
                // Lockstep same-group mode: every rank's accumulation position `j`
                // selects the same logical prompt, so the all-reduced reward stats
                // normalize shards of one group. The rollout row base still includes
                // rank/world so ranks sample distinct completions rather than clones.
                let prompt_index = step * accum as u64 + j as u64;
                let local = rank * accum + j;
                let shard_index = step * (accum * world) as u64 + local as u64;
                PromptSelection {
                    sample_idx: (step as usize * accum + j) % samples_len,
                    prompt_index,
                    rollout_global_row_base: shard_index
                        .wrapping_mul(self.config.group_size as u64),
                }
            }
        }
    }

    fn record_prompt_selection(&self, sel: &PromptSelection) {
        if !self.config.gpu_memory_probe {
            return;
        }
        tracing::info!(
            sample_idx = sel.sample_idx,
            prompt_index = sel.prompt_index,
            rollout_global_row_base = sel.rollout_global_row_base,
            reward_group_scope = ?self.config.reward_group_scope,
            "trainer: selected prompt"
        );
    }

    /// Persist this prompt group's top-K decoded completions when the candidate
    /// ledger is enabled. The ledger is written immediately after rewards are known,
    /// before any later degenerate-group skip, so a no-update run still preserves the
    /// evidence needed for artifact extraction or debugging.
    fn write_candidate_records<P, E>(
        &mut self,
        ctx: CandidateWriteCtx,
        completions: &[String],
        rewards: &[f32],
        reward_outcomes: &[RewardOutcome],
        rollout: &Rollout,
        exec: &E,
    ) -> Result<(), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let k = self.config.candidate_log_top_k.min(completions.len());
        if k == 0 {
            return Ok(());
        }
        self.coordinate_side_effect(exec, "candidate record write", |trainer| {
            if !ctx.enabled {
                return Ok(());
            }
            let Some(writer) = trainer.candidate_writer.as_mut() else {
                return Ok(());
            };
            let mut order: Vec<usize> = (0..completions.len()).collect();
            order.sort_by(|&a, &b| {
                candidate_reward_order(rewards[a], rewards[b]).then_with(|| a.cmp(&b))
            });
            for &group_index in order.iter().take(k) {
                writer.append(&CandidateRecord {
                    step: ctx.step,
                    rank: ctx.rank,
                    world_size: ctx.world_size,
                    prompt_index: ctx.prompt_index,
                    group_index,
                    reward: rewards[group_index],
                    completion_len_tokens: rollout.completion_lens[group_index],
                    reward_diagnostic: reward_outcomes[group_index].diagnostic.clone(),
                    reward_metadata: reward_outcomes[group_index].metadata.clone(),
                    completion: completions[group_index].clone(),
                })?;
            }
            Ok(())
        })
    }

    /// After completing step `step` (0-based), write a momentum-faithful (v2)
    /// `checkpoints/step-<n>/` when the configured cadence divides the completed-step
    /// count `n = step + 1`, **or** `n` is the final step of the run. The final-step
    /// write guarantees a completed run always persists its final state even when
    /// `checkpoint_every` does not divide `steps`. The checkpoint captures the adapter
    /// weights, the optimizer moments (`opt`), and the policy's rollout-sampler RNG —
    /// taken *after* this window's rollouts and update, so a [`resume`](Self::resume) at
    /// the recorded manifest `step` = `n` continues bit-exactly. (The optimizer's own
    /// `step_t` counts only non-degenerate windows and is captured independently of `n`.)
    fn maybe_checkpoint<P, E>(
        &mut self,
        step: u64,
        vars: &[Var],
        opt: &FerrlAdamW,
        policy: &P,
        exec: &E,
    ) -> Result<(), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let Some(every) = self.config.checkpoint_every else {
            return Ok(());
        };
        let completed = step + 1;
        let is_final = completed == self.config.steps;
        if !completed.is_multiple_of(every) && !is_final {
            return Ok(());
        }
        self.write_checkpoint(completed, vars, opt, policy, exec)
    }

    /// Write a momentum-faithful (v3) checkpoint to `checkpoints/step-<completed>/`
    /// unconditionally — the caller decides *when*: the periodic cadence
    /// ([`maybe_checkpoint`](Self::maybe_checkpoint)) or the preemption stop in
    /// [`run`](Self::run), which saves a final checkpoint before a requeue.
    /// Execution-rank-0-only under DP or TP: weights and optimizer moments are
    /// rank-identical by lockstep and the sampler blob is rank 0's, so a
    /// non-zero rank is a no-op. Re-writing an already-published
    /// `step-<completed>` is idempotent (the writer replaces it atomically), so a
    /// preemption that coincides with a cadence write is harmless.
    fn write_checkpoint<P, E>(
        &mut self,
        completed: u64,
        vars: &[Var],
        opt: &FerrlAdamW,
        policy: &P,
        exec: &E,
    ) -> Result<(), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        self.coordinate_side_effect(exec, "checkpoint write", |trainer| {
            if !exec.is_execution_primary(trainer.comm.as_ref()) {
                return Ok(());
            }
            let dir = trainer.checkpoints_dir.join(format!("step-{completed}"));
            let opt_state = opt.state()?;
            let sampler_state = policy.sampler_state()?;
            let recipe = policy.lora_recipe();
            crate::checkpoint::save_checkpoint(
                &dir,
                vars,
                &opt_state,
                &sampler_state,
                completed,
                recipe.as_deref(),
            )?;
            Ok(())
        })
    }

    fn coordinate_side_effect<P, E, F>(
        &mut self,
        exec: &E,
        label: &'static str,
        op: F,
    ) -> Result<(), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
        F: FnOnce(&mut Self) -> Result<(), TrainerError>,
    {
        let local = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(self)))
            .unwrap_or_else(|payload| {
                Err(TrainerError::Contract(format!(
                    "{label} panicked: {}",
                    panic_payload_message(payload.as_ref())
                )))
            });
        if exec.execution_world_size(self.comm.as_ref()) <= 1 {
            return local;
        }
        if local
            .as_ref()
            .err()
            .is_some_and(Self::is_terminal_distributed_error)
        {
            return local;
        }
        let failed_local = if local.is_err() { 1.0 } else { 0.0 };
        let failed_global =
            exec.execution_all_reduce_scalar_sum(self.comm.as_ref(), failed_local)?;
        match local {
            Err(error) => Err(error),
            Ok(()) if failed_global > 0.0 => Err(TrainerError::Contract(format!(
                "{label} failed on a peer rank; aborting in lockstep"
            ))),
            Ok(()) => Ok(()),
        }
    }

    /// Whether a stop has been requested via the preemption flag
    /// ([`with_preemption_flag`](Self::with_preemption_flag)), decided **globally**
    /// across the active DP or TP execution world so every rank stops on the same
    /// step (a local-only stop would deadlock — one rank breaks out while its
    /// peers enter the next window's collectives and wait forever).
    ///
    /// **Install-invariant:** at `world_size() > 1` *every* rank runs the poll's
    /// scalar reduce every step regardless of whether it holds a flag (a flag-less
    /// rank contributes `0.0`), so the collective sequence never depends on per-rank
    /// install state — an uneven install across ranks cannot deadlock the world,
    /// it just means no preemption. The cost is one cheap scalar all-reduce per
    /// step under DP or sharded TP; world 1 is a plain local read with no
    /// collective.
    fn preempt_requested<P, E>(&self, exec: &E) -> Result<bool, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let local = self
            .preempt
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if exec.execution_world_size(self.comm.as_ref()) > 1 {
            // Every rank reduces every step (flag-less ranks contribute 0.0), so the
            // collective sequence is identical across ranks no matter who holds a
            // flag — an uneven install can't desync the world.
            Ok(exec.execution_all_reduce_scalar_sum(
                self.comm.as_ref(),
                if local { 1.0 } else { 0.0 },
            )? > 0.0)
        } else {
            Ok(local)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_sample<P, R, E>(
        &mut self,
        step: u64,
        beta: f64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        selected: &SelectedSample<'_, R::Target>,
        gpu_mem: &mut StepGpuMemory,
        exec: &E,
    ) -> Result<(PromptStat, Option<LiveItem>), TrainerError>
    where
        P: Policy,
        R: RewardFn,
        E: PolicyExecution<P>,
    {
        let trainer_comm = Arc::clone(&self.comm);
        let execution_comm = exec.execution_comm(trainer_comm.as_ref());
        let coordinate_over_data_parallel =
            exec.model_parallel_world_size() <= 1 && execution_comm.world_size() > 1;
        if coordinate_over_data_parallel {
            let mut prestate = None;
            let collected = match self.collect_group(
                step,
                beta,
                policy,
                reward_fn,
                tokenizer,
                selected,
                gpu_mem,
                exec,
                Some(&mut prestate),
            ) {
                Ok(collected) => collected,
                Err(error) => {
                    return Err(Self::rollback_rollout_group_failure(
                        policy,
                        prestate.as_ref(),
                        execution_comm,
                        error,
                    ));
                }
            };
            let materialized = Self::coordinate_comm_call(
                execution_comm,
                "rollout group learner materialization",
                || self.materialize_collected_group(policy, collected, beta, gpu_mem, exec),
            );
            match materialized {
                Ok(materialized) => Ok(materialized),
                Err(error) => Err(Self::rollback_rollout_group_failure(
                    policy,
                    prestate.as_ref(),
                    execution_comm,
                    error,
                )),
            }
        } else {
            let collected = self.collect_group(
                step, beta, policy, reward_fn, tokenizer, selected, gpu_mem, exec, None,
            )?;
            self.materialize_collected_group(policy, collected, beta, gpu_mem, exec)
        }
    }

    /// Collector half of one prompt group: rollout → reward → host-side mask and
    /// F64 advantages. It deliberately performs no old/reference learner scoring.
    #[allow(clippy::too_many_arguments)]
    fn collect_group<P, R, E>(
        &mut self,
        step: u64,
        _beta: f64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        selected: &SelectedSample<'_, R::Target>,
        gpu_mem: &mut StepGpuMemory,
        exec: &E,
        rollback_prestate: Option<&mut Option<RolloutGroupPrestate>>,
    ) -> Result<CollectedGroup, TrainerError>
    where
        P: Policy,
        R: RewardFn,
        E: PolicyExecution<P>,
    {
        let trainer_comm = Arc::clone(&self.comm);
        let execution_comm = exec.execution_comm(trainer_comm.as_ref());
        // No rank may enter an opaque TP policy hook until every rank has
        // finished local tokenizer/toggle work.
        let (prompt_ids, gen) =
            Self::coordinate_comm_call(execution_comm, "rollout generation preflight", || {
                if let Some(prestate) = rollback_prestate {
                    *prestate = Some(Self::snapshot_rollout_group_prestate(policy)?);
                }
                policy.set_adapter_enabled(true);
                let prompt_ids = tokenizer.encode(&selected.sample.prompt);
                if prompt_ids.is_empty() {
                    return Err(TrainerError::Contract(format!(
                        "prompt encoded to zero tokens: {:?}",
                        selected.sample.prompt
                    )));
                }
                Ok((prompt_ids, GenConfig::from(&self.config)))
            })?;
        gpu_mem.record("rollout_start");
        let generated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let rollout = exec.generate_at_instrumented(
                policy,
                &prompt_ids,
                &gen,
                selected.selection.rollout_global_row_base,
                gpu_mem.recorder(),
            )?;
            let (_, comp_len) = completion_dims(&rollout)?;
            if comp_len != self.config.max_new_tokens {
                return Err(TrainerError::Contract(format!(
                    "Policy::generate returned completion width {comp_len}, expected max_new_tokens {}",
                    self.config.max_new_tokens
                )));
            }
            if self.config.tis && rollout.rollout_logprobs.is_none() {
                return Err(TrainerError::Contract(
                    "tis is enabled but Policy::generate captured no rollout log-probs \
                     (Rollout::rollout_logprobs is None) — this policy cannot supply the \
                     behavior probabilities the correction needs"
                        .into(),
                ));
            }
            if rollout.len() != self.config.group_size {
                return Err(TrainerError::Contract(format!(
                    "Policy::generate returned {} completions for group_size {}",
                    rollout.len(),
                    self.config.group_size
                )));
            }
            let completions = decode_completions(&rollout, tokenizer);
            Ok((rollout, comp_len, completions))
        }))
        .unwrap_or_else(|payload| {
            Err(TrainerError::Contract(format!(
                "rollout generation/result validation panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let (rollout, comp_len, completions) =
            Self::coordinate_comm_result(execution_comm, "rollout generation result", generated)?;
        gpu_mem.record("rollout_end");
        gpu_mem.record("reward_start");
        let reward_outcomes = self.coordinate_reward_group(
            reward_fn,
            selected.sample,
            &completions,
            rollout.len(),
            exec,
        )?;
        gpu_mem.record("reward_end");
        let rewards = Self::coordinate_comm_call(execution_comm, "rollout reward result", || {
            let rewards: Vec<f32> = reward_outcomes
                .iter()
                .map(|outcome| outcome.reward)
                .collect();
            if rewards.len() != rollout.len() {
                return Err(TrainerError::Contract(format!(
                    "reward_group_detailed returned {} rewards for {} completions",
                    rewards.len(),
                    rollout.len()
                )));
            }
            Ok(rewards)
        })?;
        self.write_candidate_records(
            CandidateWriteCtx {
                step,
                prompt_index: selected.selection.prompt_index,
                rank: exec.execution_rank(self.comm.as_ref()),
                world_size: exec.execution_world_size(self.comm.as_ref()),
                enabled: exec.writes_rank_local_telemetry(self.comm.as_ref()),
            },
            &completions,
            &rewards,
            &reward_outcomes,
            &rollout,
            exec,
        )?;
        let mut mask_rows = length_mask_rows(&rollout, comp_len);
        let truncated = if self.config.truncation_masking {
            mask_truncated_rows(&rollout, comp_len, self.config.eos_token_id, &mut mask_rows)
        } else {
            0
        };
        let dropped = zero_mask_rows(&mask_rows);
        let rewards_f64: Vec<f64> = rewards.iter().map(|&reward| f64::from(reward)).collect();
        let (advantages, distributed_reward_stats) =
            self.reward_group_advantages_with_stats(&rewards_f64)?;
        Self::coordinate_comm_result(
            execution_comm,
            "rollout-ledger group finalization",
            (|| {
                let degenerate = advantages.iter().all(|advantage| *advantage == 0.0);
                let stat = PromptStat {
                    completion_len: mean_completion_len(&rollout),
                    completion_tokens: rollout.completion_lens.iter().sum(),
                    dropped,
                    truncated,
                    degenerate,
                    ratio_stats: None,
                    rewards: rewards.clone(),
                };
                Ok(CollectedGroup {
                    accum_index: u32::try_from(selected.accum_index).map_err(|_| {
                        TrainerError::Contract(
                            "accumulation index does not fit rollout ledger u32".into(),
                        )
                    })?,
                    prompt_index: selected.selection.prompt_index,
                    rollout_global_row_base: selected.selection.rollout_global_row_base,
                    rollout,
                    rewards,
                    advantages,
                    distributed_reward_stats,
                    mask_rows,
                    stat,
                    surrogate_live: !degenerate,
                })
            })(),
        )
    }

    /// Learner half of one collected group: enforce the production F64 liveness
    /// branch, take detached old/reference scores, and tensorize constants.
    fn materialize_collected_group<P, E>(
        &self,
        policy: &mut P,
        collected: CollectedGroup,
        beta: f64,
        gpu_mem: &mut StepGpuMemory,
        exec: &E,
    ) -> Result<(PromptStat, Option<LiveItem>), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let CollectedGroup {
            rollout,
            advantages,
            mask_rows,
            mut stat,
            surrogate_live,
            ..
        } = collected;
        if !surrogate_live && beta <= 0.0 {
            return Ok((stat, None));
        }
        let execution_comm = exec.execution_comm(self.comm.as_ref());
        // Snapshot the old / reference log-probs once (the window's "old" policy),
        // reused across the mu inner epochs. Value-only, so the detached
        // scoring path: same values, a fraction of the activation footprint
        // on policies that override it (no tape is built or captured).
        gpu_mem.record("logp_old_start");
        let logp_old = Self::coordinate_model_parallel_call(
            exec.model_parallel_world_size(),
            execution_comm,
            "rollout-ledger detached old-policy scoring",
            || {
                policy.set_adapter_enabled(true);
                exec.token_logprobs_detached(policy, &rollout)
            },
        )?;
        gpu_mem.record("logp_old_end");
        // Train/rollout off-policy diagnostics + the optional TIS weight, both off
        // the captured behavior log-probs vs the logp_old scoring snapshot.
        let (ratio_stats, tis_w, device, mask) = Self::coordinate_model_parallel_call(
            exec.model_parallel_world_size(),
            execution_comm,
            "rollout-ledger old-policy score materialization",
            || {
                let (ratio_stats, tis_w) = rollout_ratio_and_tis(
                    &rollout,
                    &logp_old,
                    &mask_rows,
                    self.config.tis_imp_ratio_cap,
                    self.config.tis,
                )?;
                let device = logp_old.device().clone();
                let mask = mask_rows_to_tensor(&mask_rows, &device)?;
                Ok((ratio_stats, tis_w, device, mask))
            },
        )?;
        stat.ratio_stats = ratio_stats;
        gpu_mem.record("logp_ref_start");
        let logp_ref = Self::coordinate_model_parallel_call(
            exec.model_parallel_world_size(),
            execution_comm,
            "rollout-ledger detached reference-policy scoring",
            || self.reference_logprobs(policy, &rollout, beta, exec),
        )?;
        gpu_mem.record("logp_ref_end");
        Self::coordinate_model_parallel_call(
            exec.model_parallel_world_size(),
            execution_comm,
            "rollout-ledger learner item materialization",
            || {
                let advantages = advantages_tensor(&advantages, &device)?;
                let item = LiveItem {
                    rollout,
                    advantages,
                    logp_old,
                    logp_ref,
                    mask,
                    tis_w,
                };
                Ok((stat, Some(item)))
            },
        )
    }

    /// Score one TP group's rewards on execution rank 0, then broadcast those
    /// canonical scalars before any rank computes advantages or branches on a
    /// degenerate group. DP ranks still score their own prompt shards locally.
    #[allow(clippy::cognitive_complexity)] // explicit primary/status/value broadcast protocol
    fn coordinate_reward_group<P, R, E>(
        &self,
        reward_fn: &R,
        sample: &Sample<R::Target>,
        completions: &[String],
        expected_len: usize,
        exec: &E,
    ) -> Result<Vec<RewardOutcome>, TrainerError>
    where
        P: Policy,
        R: RewardFn,
        E: PolicyExecution<P>,
    {
        let score_local = || {
            let outcomes = reward_fn.reward_group_detailed(sample, completions)?;
            if outcomes.len() != expected_len {
                return Err(TrainerError::Contract(format!(
                    "reward_group_detailed returned {} rewards for {expected_len} completions",
                    outcomes.len()
                )));
            }
            let rewards: Vec<f32> = outcomes.iter().map(|outcome| outcome.reward).collect();
            validate_reward_values(&rewards)?;
            Ok(outcomes)
        };

        if exec.model_parallel_world_size() <= 1 {
            return Self::coordinate_comm_call(
                exec.execution_comm(self.comm.as_ref()),
                "rollout/reward evaluation",
                score_local,
            );
        }

        let is_primary = exec.is_execution_primary(self.comm.as_ref());
        let local = if is_primary {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(score_local)) {
                Ok(result) => result,
                Err(payload) => Err(TrainerError::Contract(format!(
                    "reward evaluation panicked: {}",
                    panic_payload_message(payload.as_ref())
                ))),
            }
        } else {
            Ok(Vec::new())
        };
        if local
            .as_ref()
            .err()
            .is_some_and(Self::is_terminal_distributed_error)
        {
            return local;
        }
        let failed_local = if local.is_err() { 1.0 } else { 0.0 };
        let failed_global =
            exec.execution_all_reduce_scalar_sum(self.comm.as_ref(), failed_local)?;
        let local = match local {
            Err(error) => return Err(error),
            Ok(_) if failed_global > 0.0 => {
                return Err(TrainerError::Contract(
                    "reward evaluation failed on tensor-parallel execution rank 0; aborting in \
                     lockstep"
                        .into(),
                ));
            }
            Ok(outcomes) => outcomes,
        };

        let local_reward_bits: Vec<f64> = if is_primary {
            local
                .iter()
                .map(|outcome| f64::from(outcome.reward.to_bits()))
                .collect()
        } else {
            vec![0.0; expected_len]
        };
        let mut canonical_rewards = Vec::with_capacity(expected_len);
        for local_bits in local_reward_bits {
            let canonical_bits =
                exec.execution_all_reduce_scalar_sum(self.comm.as_ref(), local_bits)? as u32;
            canonical_rewards.push(f32::from_bits(canonical_bits));
        }

        if is_primary {
            Ok(local
                .into_iter()
                .zip(canonical_rewards)
                .map(|(mut outcome, canonical)| {
                    outcome.reward = canonical;
                    outcome
                })
                .collect())
        } else {
            Ok(canonical_rewards
                .into_iter()
                .map(RewardOutcome::reward)
                .collect())
        }
    }

    #[cfg(test)]
    fn reward_group_advantages(&self, rewards: &[f64]) -> Result<Vec<f64>, TrainerError> {
        self.reward_group_advantages_with_stats(rewards)
            .map(|(advantages, _)| advantages)
    }

    fn reward_group_advantages_with_stats(
        &self,
        rewards: &[f64],
    ) -> Result<(Vec<f64>, Option<RolloutLedgerRewardStats>), TrainerError> {
        match self.config.reward_group_scope {
            RewardGroupScope::Local => {
                Ok((group_advantages(rewards, self.config.scale_rewards), None))
            }
            RewardGroupScope::DistributedSamePrompt if self.comm.world_size() == 1 => {
                Ok((group_advantages(rewards, self.config.scale_rewards), None))
            }
            RewardGroupScope::DistributedSamePrompt => {
                let local = RewardStatsAcc::from_rewards(rewards);
                let count = self.comm.all_reduce_scalar_sum(local.count)?;
                let sum = self.comm.all_reduce_scalar_sum(local.sum)?;
                let sumsq = self.comm.all_reduce_scalar_sum(local.sumsq)?;
                let global = self.coordinate_data_parallel_result(
                    "distributed reward-statistics reduction",
                    if count.is_finite() && sum.is_finite() && sumsq.is_finite() && sumsq >= 0.0 {
                        Ok(RewardStatsAcc { count, sum, sumsq })
                    } else {
                        Err(TrainerError::Contract(
                            "distributed reward statistics must be finite with nonnegative sumsq"
                                .into(),
                        ))
                    },
                )?;
                let count = exact_reduced_u64("distributed reward count", global.count)?;
                Ok((
                    advantages_from_stats(rewards, global, self.config.scale_rewards),
                    Some(RolloutLedgerRewardStats {
                        count,
                        sum_bits: global.sum.to_bits(),
                        sumsq_bits: global.sumsq.to_bits(),
                    }),
                ))
            }
        }
    }

    /// Run the `mu` inner epochs over a window's live items, each epoch accumulating
    /// every live prompt's gradient into one `AdamW` step. The last epoch's
    /// diagnostics land in the window's metrics. `window_tokens` is the window's
    /// total completion-token count (the DAPO normalizer, constant across epochs;
    /// global under DP) and `n_live_global` the world's live-item count (the
    /// kl/clip-mean divisor; `live.len()` at world 1). Under DP every rank runs
    /// these epochs — `live` may be empty on a rank whose shard was
    /// all-degenerate; it participates in the collectives with zeros.
    #[allow(clippy::too_many_arguments)]
    fn update_window<P, E>(
        &self,
        policy: &P,
        live: &[LiveItem],
        ctx: &mut UpdateCtx<'_>,
        window_tokens: f64,
        n_live_global: f64,
        beta: f64,
        exec: &E,
    ) -> Result<InnerAgg, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let mut agg = InnerAgg::default();
        for _ in 0..self.config.mu {
            agg =
                self.accumulate_step(policy, live, ctx, window_tokens, n_live_global, beta, exec)?;
        }
        Ok(agg)
    }

    /// One inner epoch: forward+backward each live prompt, fold its trainable-var
    /// gradients into a running sum (all-reduce-summed across ranks under DP),
    /// run the grad-coverage canary on the accumulated gradient, then take one
    /// optimizer step. Only one prompt's grad forward is held at a time (the
    /// accumulator keeps just the small per-var sums), so the window's peak
    /// memory is a single group's.
    ///
    /// Under DP the collective sequence per epoch is fixed — gradient reduce,
    /// uncovered count, kl sum, clip sum — and runs unconditionally on every
    /// rank **before** any early return, so every later decision (canary, the
    /// no-signal skip) is a pure function of global state and the ranks act in
    /// lockstep.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_step<P, E>(
        &self,
        policy: &P,
        live: &[LiveItem],
        ctx: &mut UpdateCtx<'_>,
        window_tokens: f64,
        n_live_global: f64,
        beta: f64,
        exec: &E,
    ) -> Result<InnerAgg, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let vars = ctx.vars;
        let execution_comm = exec.execution_comm(self.comm.as_ref());
        let local_accumulation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
            let mut covered = vec![true; vars.len()];
            let mut sum_kl = 0.0_f32;
            let mut sum_clip = 0.0_f32;
            for item in live {
                let local_item = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    ctx.gpu_mem.record("item_backward_start");
                    let (grads, kl, clip_frac) = self.item_backward_with_execution(
                        policy,
                        item,
                        window_tokens,
                        vars,
                        beta,
                        exec,
                    )?;
                    ctx.gpu_mem.record("item_backward_end");
                    fold_var_grads(vars, &grads, &mut acc, &mut covered)?;
                    Ok((kl, clip_frac))
                }))
                .unwrap_or_else(|payload| {
                    Err(TrainerError::Contract(format!(
                        "local backward item panicked: {}",
                        panic_payload_message(payload.as_ref())
                    )))
                });
                let (kl, clip_frac) = Self::coordinate_model_parallel_result(
                    exec.model_parallel_world_size(),
                    execution_comm,
                    "tensor-parallel backward item",
                    local_item,
                )?;
                sum_kl += kl;
                sum_clip += clip_frac;
            }
            // Materialize every zero contribution before any rank enters the
            // gradient collective. A rank-local allocation/device failure is
            // therefore globalized by the status rendezvous below rather than
            // stranding peers inside `all_reduce_sum`.
            materialize_zero_grad_slots(vars, &mut acc)?;
            Ok((acc, covered, sum_kl, sum_clip))
        }))
        .unwrap_or_else(|payload| {
            Err(TrainerError::Contract(format!(
                "local backward accumulation panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let (mut acc, covered, sum_kl, sum_clip) = Self::coordinate_comm_result(
            execution_comm,
            "local backward accumulation",
            local_accumulation,
        )?;
        let reduced = if self.comm.world_size() > 1 {
            ctx.gpu_mem.record("grad_all_reduce_start");
            self.reduce_epoch(vars, &mut acc, &covered, sum_kl, sum_clip, n_live_global)
        } else {
            Ok((
                sum_kl / n_live_global as f32,
                sum_clip / n_live_global as f32,
                0.0,
            ))
        };
        let (kl, clip_frac, mut uncovered_global) =
            Self::coordinate_comm_result(execution_comm, "gradient reduction", reduced)?;
        ctx.gpu_mem.record("grad_all_reduce_end");
        let model_parallel = if exec.model_parallel_world_size() > 1 {
            ctx.gpu_mem.record("tp_grad_all_reduce_start");
            exec.reduce_model_parallel_grads(vars, &mut acc, &covered)
        } else {
            Ok(0.0)
        };
        uncovered_global += Self::coordinate_comm_result(
            execution_comm,
            "model-parallel gradient reduction",
            model_parallel,
        )?;
        if exec.model_parallel_world_size() > 1 {
            ctx.gpu_mem.record("tp_grad_all_reduce_end");
        }
        let post_reduce = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.gpu_mem.record("grad_store_compact_start");
            // Build a fresh tiny store instead of reusing the last backward's full store. Candle's
            // backward store also contains unrelated intermediate-node gradients; keeping it alive
            // through the reduce/optimizer handoff was pure peak-memory pressure.
            let store = empty_grad_store(vars)?;
            let mut store = combine_into_store(vars, store, &mut acc, &covered);
            ctx.gpu_mem.record("grad_store_compact_end");
            let cov = grad_coverage(vars, &store)?;
            // Fatal: a missing var (candle's silent-skip landmine — an absent grad entry)
            // or a non-finite accumulated gradient (a blowup).
            if !cov.is_covered() || cov.nonfinite > 0 {
                cov.clone().into_result()?;
            }
            // The canary verdict is global under DP: a var missing from a PEER rank's
            // backward poisons the reduced sum every rank just stepped from, so a rank
            // that is locally covered must abort too — in lockstep with the rank that
            // reports the detail above.
            if uncovered_global > 0.0 {
                return Err(TrainerError::Contract(format!(
                    "grad-coverage canary failed on a peer rank ({uncovered_global} uncovered \
                     var-gradients across the world) — aborting in lockstep"
                )));
            }
            if !cov.is_live() {
                // Covered + finite + all-zero accumulated gradient: no usable signal this
                // epoch (every live prompt fully clipped, or advantages cancelling). Skip
                // the optimizer step rather than mislabel it a dead forward.
                return Ok(InnerAgg {
                    kl,
                    clip_frac,
                    grad_norm: 0.0,
                });
            }
            // Global-norm gradient clipping (the verl/TRL standard): scale every
            // trainable-var gradient by `max / norm` when the accumulated norm
            // exceeds the configured maximum. The reported `grad_norm` metric stays
            // the PRE-clip norm — the doc promise it always made.
            let grad_norm = global_grad_norm(vars, &store)?;
            if let Some(max) = self.config.max_grad_norm {
                if grad_norm > max {
                    ctx.gpu_mem.record("grad_clip_start");
                    scale_var_grads(vars, &mut store, max / grad_norm)?;
                    ctx.gpu_mem.record("grad_clip_end");
                }
            }
            ctx.gpu_mem.record("optimizer_start");
            ctx.opt.step(&store)?;
            ctx.gpu_mem.record("optimizer_end");
            Ok(InnerAgg {
                kl,
                clip_frac,
                grad_norm: grad_norm as f32,
            })
        }))
        .unwrap_or_else(|payload| {
            Err(TrainerError::Contract(format!(
                "post-reduction optimizer work panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        Self::coordinate_comm_result(execution_comm, "post-reduction optimizer work", post_reduce)
    }

    /// The per-epoch DP collective sequence (world > 1 only): all-reduce-sum
    /// the accumulated per-var gradients (a `None` slot — an empty local
    /// shard, or a var no local backward reached — contributes zeros, since
    /// the collective contract is a uniform tensor count/shape on every rank;
    /// an uncovered var that *some* local backwards did reach still reduces
    /// its partial sum, which is exactly why the globalized verdict below
    /// must abort every rank before that poisoned sum is stepped on), then
    /// globalize the coverage verdict and the kl/clip diagnostic sums. On
    /// return every `acc` slot holds the **global** gradient sum; `covered`
    /// keeps its local meaning so the canary still reports rank-local detail.
    fn reduce_epoch(
        &self,
        vars: &[Var],
        acc: &mut [Option<Tensor>],
        covered: &[bool],
        sum_kl: f32,
        sum_clip: f32,
        n_live_global: f64,
    ) -> Result<(f32, f32, f64), TrainerError> {
        let uncovered_global = reduce_accumulated_grads(self.comm.as_ref(), vars, acc, covered)?;
        let kl_global = self.comm.all_reduce_scalar_sum(f64::from(sum_kl))?;
        let clip_global = self.comm.all_reduce_scalar_sum(f64::from(sum_clip))?;
        Ok((
            (kl_global / n_live_global) as f32,
            (clip_global / n_live_global) as f32,
            uncovered_global,
        ))
    }

    /// Forward + backward one live prompt: the single `grpo_loss` plus its scalar
    /// diagnostics (clip fraction, mean k3 KL). Returns the backward's
    /// [`GradStore`] and the diagnostics.
    ///
    /// **Accumulation scaling differs by loss type.** For `Grpo` / `DrGrpo` the
    /// loss is scaled by `1 / (grad_accum_steps · world_size)` (so the reduced
    /// window gradient is the **global** per-prompt mean — TRL divides those
    /// reductions by the accumulation step count the same way, and under DP
    /// the mean runs over every rank's items); the scale is skipped when that
    /// product is `1`, keeping the single-prompt single-rank path bit-identical
    /// (no extra affine node). For `Dapo` the per-item reduction *already*
    /// divides by the **global window's** total completion tokens (TRL's
    /// `num_items_in_batch` normalizer, all-reduced in `run_window`), so
    /// summing the items across ranks is the complete normalization and no
    /// extra scale applies.
    #[cfg(test)]
    fn item_backward<P: Policy>(
        &self,
        policy: &P,
        item: &LiveItem,
        window_tokens: f64,
        vars: &[Var],
        beta: f64,
    ) -> Result<(GradStore, f32, f32), TrainerError> {
        let exec = UnshardedPolicyExecution;
        self.item_backward_with_execution(policy, item, window_tokens, vars, beta, &exec)
    }

    #[allow(clippy::too_many_arguments)]
    fn item_backward_with_execution<P, E>(
        &self,
        policy: &P,
        item: &LiveItem,
        window_tokens: f64,
        vars: &[Var],
        beta: f64,
        exec: &E,
    ) -> Result<(GradStore, f32, f32), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let g = item.rollout.len();
        let mb = self.config.backward_microbatch_size;
        if mb > 0 && mb < g {
            return self.item_backward_microbatched(
                policy,
                item,
                window_tokens,
                vars,
                mb,
                beta,
                exec,
            );
        }
        self.item_backward_uncut(policy, item, window_tokens, vars, 1.0, beta, exec)
    }

    #[allow(clippy::too_many_arguments)]
    fn item_backward_microbatched<P, E>(
        &self,
        policy: &P,
        item: &LiveItem,
        window_tokens: f64,
        vars: &[Var],
        microbatch_size: usize,
        beta: f64,
        exec: &E,
    ) -> Result<(GradStore, f32, f32), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let execution_comm = exec.execution_comm(self.comm.as_ref());
        let diagnostics = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let logp_diag = exec.token_logprobs_detached(policy, &item.rollout)?;
            self.item_diagnostics(&logp_diag, item)
                .map_err(TrainerError::from)
        }))
        .unwrap_or_else(|payload| {
            Err(TrainerError::Contract(format!(
                "tensor-parallel item diagnostics panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let (kl, clip_frac) = Self::coordinate_model_parallel_result(
            exec.model_parallel_world_size(),
            execution_comm,
            "tensor-parallel item diagnostics",
            diagnostics,
        )?;

        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut covered = vec![true; vars.len()];
        let total_rows = item.rollout.len();
        for start in (0..total_rows).step_by(microbatch_size) {
            let len = microbatch_size.min(total_rows - start);
            let slice = Self::coordinate_model_parallel_result(
                exec.model_parallel_world_size(),
                execution_comm,
                "tensor-parallel backward microbatch preflight",
                slice_live_item(item, start, len).map_err(TrainerError::from),
            )?;
            let loss_scale = match self.config.loss_type {
                LossType::Dapo => 1.0,
                LossType::Grpo | LossType::DrGrpo => len as f64 / total_rows as f64,
            };
            let (grads, _, _) = self.item_backward_uncut(
                policy,
                &slice,
                window_tokens,
                vars,
                loss_scale,
                beta,
                exec,
            )?;
            Self::coordinate_model_parallel_result(
                exec.model_parallel_world_size(),
                execution_comm,
                "tensor-parallel backward microbatch fold",
                fold_var_grads(vars, &grads, &mut acc, &mut covered).map_err(TrainerError::from),
            )?;
        }
        let store = empty_grad_store(vars)?;
        let store = combine_into_store(vars, store, &mut acc, &covered);
        Ok((store, kl, clip_frac))
    }

    #[allow(clippy::too_many_arguments)]
    fn item_backward_uncut<P, E>(
        &self,
        policy: &P,
        item: &LiveItem,
        window_tokens: f64,
        vars: &[Var],
        loss_scale: f64,
        beta: f64,
        exec: &E,
    ) -> Result<(GradStore, f32, f32), TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        let execution_comm = exec.execution_comm(self.comm.as_ref());
        let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let logp = exec.token_logprobs(policy, &item.rollout)?;
            let cfg = LossCfg {
                clip_eps_low: self.config.clip_eps,
                clip_eps_high: self.config.clip_eps_high_eff(),
                beta,
                loss_type: self.config.loss_type,
                is_level: self.config.importance_sampling_level,
                dapo_norm: Some(window_tokens),
                tis_w: item.tis_w.clone(),
            };
            let mut loss = grpo_loss(
                &logp,
                &item.logp_old,
                item.logp_ref.as_ref(),
                &item.advantages,
                &item.mask,
                &cfg,
            )?;
            if (loss_scale - 1.0).abs() > f64::EPSILON {
                loss = loss.affine(loss_scale, 0.0)?;
            }
            let global_items = self.config.grad_accum_steps * self.comm.world_size();
            if global_items > 1 && self.config.loss_type != LossType::Dapo {
                loss = loss.affine(1.0 / global_items as f64, 0.0)?;
            }
            let (kl, clip_frac) = self.item_diagnostics(&logp.detach(), item)?;
            Ok((loss, kl, clip_frac))
        }))
        .unwrap_or_else(|payload| {
            Err(TrainerError::Contract(format!(
                "tensor-parallel loss preparation panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let (loss, kl, clip_frac) = Self::coordinate_model_parallel_result(
            exec.model_parallel_world_size(),
            execution_comm,
            "tensor-parallel loss preparation",
            prepared,
        )?;
        // Through the active execution seam (default: exactly
        // `Policy::backward`): TP checkpointing can replay layer collectives
        // through its explicit communicator, while the canary downstream holds
        // either way.
        let raw = exec.backward(policy, &loss)?;
        let grads = Self::coordinate_model_parallel_result(
            exec.model_parallel_world_size(),
            execution_comm,
            "tensor-parallel backward gradient materialization",
            compact_trainable_grad_store(vars, raw).map_err(TrainerError::from),
        )?;
        Ok((grads, kl, clip_frac))
    }

    fn item_diagnostics(&self, logp_diag: &Tensor, item: &LiveItem) -> CandleResult<(f32, f32)> {
        // Scalar diagnostics, off the differentiated path. The ratio is formed at
        // the configured level over the same padding-substituted log-probs the
        // loss uses, so the clip-fraction metric reports the ratio the surrogate
        // actually clipped.
        let logp_sub = substitute_padding(logp_diag, &item.logp_old, &item.mask)?;
        let ratio = level_ratio(
            &logp_sub,
            &item.logp_old,
            &item.mask,
            self.config.importance_sampling_level,
        )?;
        let clip_frac = clip_fraction(
            &ratio,
            &item.advantages,
            self.config.clip_eps,
            self.config.clip_eps_high_eff(),
            &item.mask,
        )?;
        let kl = self.kl_metric(logp_diag, item.logp_ref.as_ref(), &item.mask)?;
        Ok((kl, clip_frac))
    }

    /// Mean masked k3 KL for the step's metrics — the diagnostic counterpart of the
    /// KL penalty [`grpo_loss`] folds into the differentiated objective. Returns `0`
    /// when no reference policy is active (`beta == 0`, so `logp_ref` is `None`).
    /// Always the masked **token mean** `Σ(kl·mask) / max(Σmask, 1)` — TRL's
    /// `masked_batch_mean` — independent of the configured loss reduction (the
    /// loss normalizer shapes the gradient, not the diagnostic).
    fn kl_metric(
        &self,
        logp: &Tensor,
        logp_ref: Option<&Tensor>,
        mask: &Tensor,
    ) -> CandleResult<f32> {
        let Some(logp_ref) = logp_ref else {
            return Ok(0.0);
        };
        let kl = k3_kl_tensor(logp, logp_ref)?;
        scalar_f32(&masked_mean_tensor(&kl, mask, LossType::Dapo, None)?)
    }

    /// Reference (adapter-disabled) log-probs, restoring the adapter before any
    /// fallible op can early-return — only computed when `beta > 0`.
    fn reference_logprobs<P, E>(
        &self,
        policy: &mut P,
        rollout: &Rollout,
        beta: f64,
        exec: &E,
    ) -> Result<Option<Tensor>, TrainerError>
    where
        P: Policy,
        E: PolicyExecution<P>,
    {
        if beta <= 0.0 {
            return Ok(None);
        }
        policy.set_adapter_enabled(false);
        // Value-only (the reference is never trained): the detached scoring path.
        let logp = exec.token_logprobs_detached(policy, rollout);
        policy.set_adapter_enabled(true); // always restore.
        Ok(Some(logp?))
    }

    /// Aggregate a window's per-prompt [`PromptStat`]s and the update's diagnostics
    /// into one [`Metrics`] row: mean/std reward over **every** completion in the
    /// window, the fraction of degenerate groups, mean completion length, and total
    /// dropped rows. At `grad_accum_steps == 1` the window is a single prompt and this
    /// is identical to the prior per-prompt metrics.
    fn build_window_metrics(
        &self,
        step: u64,
        beta: f64,
        stats: &[PromptStat],
        agg: &InnerAgg,
        opt: &FerrlAdamW,
    ) -> Metrics {
        let mut m = Metrics::at_step(step);
        let all_rewards: Vec<f32> = stats
            .iter()
            .flat_map(|s| s.rewards.iter().copied())
            .collect();
        let (mean, std) = reward_stats(&all_rewards);
        m.reward_mean = mean;
        m.reward_std = std;
        // Fraction of the window's groups that were degenerate no-ops — tied to the
        // same condition that drove each skip, so metric and optimizer never disagree.
        let degenerate = stats.iter().filter(|s| s.degenerate).count();
        m.frac_reward_zero_std = degenerate as f32 / stats.len() as f32;
        m.completion_len = stats.iter().map(|s| s.completion_len).sum::<f32>() / stats.len() as f32;
        m.dropped_rows = stats.iter().map(|s| s.dropped as u32).sum();
        // Fraction of the window's completions masked out by truncation masking
        // (ran to full width without EOS) — DAPO overlong-filtering telemetry.
        let truncated: usize = stats.iter().map(|s| s.truncated).sum();
        let completions: usize = stats.iter().map(|s| s.rewards.len()).sum();
        m.frac_truncated = if completions > 0 {
            truncated as f32 / completions as f32
        } else {
            0.0
        };
        m.kl = agg.kl;
        m.clip_ratio = agg.clip_frac;
        // Train/rollout off-policy telemetry over the window's loss tokens.
        let folded = fold_ratio_stats(stats);
        m.rollout_ratio_mean = folded.ratio_mean;
        m.rollout_logratio_mean = folded.logratio_mean;
        m.rollout_ratio_max = folded.ratio_max;
        m.frac_rollout_ratio_capped = folded.frac_capped;
        m.rollout_capture_tokens = folded.tokens;
        m.grad_norm = agg.grad_norm;
        m.lr = opt.learning_rate() as f32;
        m.beta = beta as f32;
        m
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

/// Per-rank rollout throughput for a step: this rank's real completion tokens over
/// the step wall-time, guarding the degenerate `secs == 0` (a sub-tick step) so the
/// metric is `0.0` rather than infinite.
fn step_throughput(tokens: usize, secs: f64) -> f32 {
    if secs > 0.0 {
        (tokens as f64 / secs) as f32
    } else {
        0.0
    }
}

fn validate_external_policy_sha256(digest: &str) -> Result<(), TrainerError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(TrainerError::Contract(
        "policy_sha256 must be 64 lowercase hexadecimal characters".into(),
    ))
}

fn exact_reduced_u64(label: &str, value: f64) -> Result<u64, TrainerError> {
    const MAX_EXACT_F64_INTEGER: f64 = 9_007_199_254_740_992.0;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > MAX_EXACT_F64_INTEGER {
        return Err(TrainerError::Contract(format!(
            "{label} reduction {value:?} is not an exact nonnegative integer representable in f64"
        )));
    }
    Ok(value as u64)
}

fn exact_u64_as_f64(label: &str, value: u64) -> Result<f64, TrainerError> {
    const MAX_EXACT_F64_U64: u64 = 9_007_199_254_740_992;
    if value > MAX_EXACT_F64_U64 {
        return Err(TrainerError::Contract(format!(
            "{label} {value} cannot be represented exactly for scalar reduction"
        )));
    }
    Ok(value as f64)
}

fn domain_sha256(domain: &str, fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain.as_bytes());
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    let digest = hasher.finalize();
    format!("{digest:x}")
}

fn canonical_tensor_bytes<'a>(
    tensors: impl IntoIterator<Item = (String, &'a Tensor)>,
) -> Result<Vec<u8>, TrainerError> {
    let mut ordered = BTreeMap::new();
    for (name, tensor) in tensors {
        let canonical = tensor.to_device(&Device::Cpu)?.contiguous()?;
        ordered.insert(name, canonical);
    }
    safetensors::tensor::serialize(
        ordered.iter().map(|(name, tensor)| (name.as_str(), tensor)),
        None,
    )
    .map_err(|error| {
        TrainerError::Contract(format!(
            "serialize exact tensors for rollout ledger identity: {error}"
        ))
    })
}

/// Candidate ledger ordering after reward validation: higher reward first.
fn candidate_reward_order(a: f32, b: f32) -> std::cmp::Ordering {
    debug_assert!(a.is_finite() && b.is_finite());
    b.total_cmp(&a)
}

/// A window's folded rollout-ratio telemetry (see [`fold_ratio_stats`]).
struct FoldedRatios {
    ratio_mean: f32,
    logratio_mean: f32,
    ratio_max: f32,
    frac_capped: f32,
    tokens: u32,
}

/// Fold a window's per-group [`RatioStats`] into the rollout-ratio metrics
/// (token-weighted ratio and log-ratio means, max, capped fraction, and the
/// captured-token count). A window with no captured loss tokens reports the
/// neutral values (`1.0` ratios, `0.0` log-ratio/fraction) with `tokens == 0`
/// — the count is what distinguishes "measured on-policy" from "telemetry
/// dark" (no capture, or every group degenerate at `beta == 0`).
fn fold_ratio_stats(stats: &[PromptStat]) -> FoldedRatios {
    let mut sum = 0.0_f64;
    let mut log_sum = 0.0_f64;
    let mut max = f64::NEG_INFINITY;
    let mut capped = 0_usize;
    let mut tokens = 0_usize;
    for r in stats.iter().filter_map(|s| s.ratio_stats.as_ref()) {
        sum += r.sum;
        log_sum += r.log_sum;
        max = max.max(r.max);
        capped += r.capped;
        tokens += r.tokens;
    }
    if tokens == 0 {
        return FoldedRatios {
            ratio_mean: 1.0,
            logratio_mean: 0.0,
            ratio_max: 1.0,
            frac_capped: 0.0,
            tokens: 0,
        };
    }
    FoldedRatios {
        ratio_mean: (sum / tokens as f64) as f32,
        logratio_mean: (log_sum / tokens as f64) as f32,
        ratio_max: max as f32,
        frac_capped: capped as f32 / tokens as f32,
        tokens: tokens.min(u32::MAX as usize) as u32,
    }
}

/// Decode each completion to text for the reward — the **real** completion tokens
/// only. `completion_lens[i]` is the EOS-inclusive real length (see
/// [`Rollout::completion_lens`]), so the slice stops there and the EOS padding never
/// reaches the [`RewardFn`]. With `eos_token_id == None` every length is the full
/// width and this is the entire post-prompt slice, unchanged. [`completion_dims`] has
/// already bounded every length by the completion width, so `prompt_len + len` is in
/// range; `.min(ids.len())` is a defensive belt so a future change can never panic.
fn decode_completions(rollout: &Rollout, tokenizer: &dyn TokenizerLike) -> Vec<String> {
    rollout
        .token_ids
        .iter()
        .zip(&rollout.completion_lens)
        .map(|(ids, &len)| {
            let start = rollout.prompt_len;
            let end = (start + len).min(ids.len());
            tokenizer.decode(&ids[start..end])
        })
        .collect()
}

/// Per-group train/rollout ratio telemetry and the optional TIS weight tensor,
/// from the captured behavior log-probs (`rollout.rollout_logprobs`) against the
/// trainer's own detached `logp_old` scoring snapshot.
///
/// Returns `(None, None)` when the policy captured nothing (the telemetry then
/// reports its neutral defaults and no correction applies — the caller has
/// already failed loud if `tis` demanded a capture). Otherwise the stats
/// aggregate the ratio `exp(logp_old − logp_rollout)` over the group's
/// **loss-carrying** tokens (`mask_rows > 0` — truncation-masked rows carry no
/// gradient, so their mismatch is irrelevant), and `tis_w` — built only when
/// `tis` is on — holds [`crate::grpo::tis_weight`] per token (`1.0` at padding,
/// which the mask removes anyway), as a detached `[G, comp_len]` constant.
/// A group whose mask kept no tokens yields `(None, tis_w)` rather than
/// degenerate stats.
fn rollout_ratio_and_tis(
    rollout: &Rollout,
    logp_old: &Tensor,
    mask_rows: &[Vec<f64>],
    cap: f64,
    tis: bool,
) -> Result<(Option<RatioStats>, Option<Tensor>), TrainerError> {
    let Some(captured) = &rollout.rollout_logprobs else {
        return Ok((None, None));
    };
    let train = logp_old.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    // The rollout (and its capture) is validated by `completion_dims`; the score
    // tensor is the policy's own output and is NOT — a misshapen
    // `token_logprobs` would otherwise silently zip-truncate into wrong-token
    // pairings here (and a wrong-shaped `tis_w` would surface only at the
    // broadcast, or with `tis` off never at all).
    let comp_len = mask_rows.first().map_or(0, Vec::len);
    if train.len() != mask_rows.len() || train.iter().any(|row| row.len() != comp_len) {
        return Err(TrainerError::Contract(format!(
            "Policy::token_logprobs returned a [{}, {}] tensor for a [{}, {comp_len}] rollout",
            train.len(),
            train.first().map_or(0, Vec::len),
            mask_rows.len()
        )));
    }
    let mut stats = RatioStats {
        sum: 0.0,
        log_sum: 0.0,
        max: f64::NEG_INFINITY,
        capped: 0,
        tokens: 0,
    };
    let mut weights = Vec::with_capacity(train.len() * comp_len);
    for ((cap_row, train_row), mask_row) in captured.iter().zip(&train).zip(mask_rows) {
        fold_ratio_row(cap_row, train_row, mask_row, cap, &mut stats, &mut weights);
    }
    let tis_w = tis
        .then(|| Tensor::from_vec(weights, (train.len(), comp_len), logp_old.device()))
        .transpose()?;
    Ok(((stats.tokens > 0).then_some(stats), tis_w))
}

/// Fold one sequence's captured behavior log-probs into the running
/// [`RatioStats`] and TIS-weight buffer (see [`rollout_ratio_and_tis`]). The
/// capture has one entry per real draw; positions past it are EOS padding —
/// weight `1.0`, never counted (the loss mask is `0` there by construction).
/// The log-ratio is formed once in f64 and feeds the ratio, the drift
/// accumulator, AND the TIS weight (`min(exp, cap)` —
/// [`crate::grpo::tis_weight`]'s formula with shared rounding, so the
/// capped-count telemetry and the actual truncation can never disagree by an
/// ulp at the cap).
fn fold_ratio_row(
    cap_row: &[f32],
    train_row: &[f32],
    mask_row: &[f64],
    cap: f64,
    stats: &mut RatioStats,
    weights: &mut Vec<f32>,
) {
    for (j, (&lp_train, &m)) in train_row.iter().zip(mask_row).enumerate() {
        let Some(&lp_rollout) = cap_row.get(j) else {
            weights.push(1.0);
            continue;
        };
        let log_ratio = f64::from(lp_train) - f64::from(lp_rollout);
        let ratio = log_ratio.exp();
        weights.push(ratio.min(cap) as f32);
        if m > 0.0 {
            stats.sum += ratio;
            stats.log_sum += log_ratio;
            stats.max = stats.max.max(ratio);
            stats.capped += usize::from(ratio > cap);
            stats.tokens += 1;
        }
    }
}

/// The length-aware loss mask as `[G][comp_len]` `f64` rows: `1.0` on the real
/// completion tokens (column `j < completion_lens[i]`) and `0.0` on the EOS padding.
/// Shared by the dropped-row count ([`zero_mask_rows`]) and the differentiated mask
/// tensor ([`mask_rows_to_tensor`]) so the two never disagree. [`completion_dims`]
/// has already bounded every length by `comp_len`.
fn length_mask_rows(rollout: &Rollout, comp_len: usize) -> Vec<Vec<f64>> {
    rollout
        .completion_lens
        .iter()
        .map(|&len| {
            (0..comp_len)
                .map(|j| if j < len { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Zero the mask rows of **truncated** completions and return how many were
/// masked. A completion is truncated iff it occupies the full completion width
/// (`completion_lens[i] == comp_len`) and its last real token is not EOS — the
/// EOS-inclusive length convention means a non-truncated full-width row ends
/// in EOS exactly at the boundary (TRL's test is the same: `ids[-1] not in
/// (eos, pad)`). With `eos_token_id == None` nothing can be truncated-detected
/// (no sequence can terminate), so the mask is untouched and the count is `0`
/// — the masking knob is inert, never completion-zeroing.
fn mask_truncated_rows(
    rollout: &Rollout,
    comp_len: usize,
    eos_token_id: Option<u32>,
    mask_rows: &mut [Vec<f64>],
) -> usize {
    let Some(eos) = eos_token_id else {
        return 0;
    };
    let mut truncated = 0;
    for (i, (&len, ids)) in rollout
        .completion_lens
        .iter()
        .zip(&rollout.token_ids)
        .enumerate()
    {
        // completion_dims has already bounded len <= comp_len and validated the
        // rectangular shape, so the index below is in range when len == comp_len.
        if len == comp_len && ids[rollout.prompt_len + len - 1] != eos {
            mask_rows[i].iter_mut().for_each(|m| *m = 0.0);
            truncated += 1;
        }
    }
    truncated
}

/// Build the `[G, comp_len]` F32 mask tensor from the `f64` mask rows (the same rows
/// [`zero_mask_rows`] counted), so the differentiated mask and the dropped-row
/// telemetry are the one source of truth.
fn mask_rows_to_tensor(rows: &[Vec<f64>], device: &Device) -> CandleResult<Tensor> {
    let g = rows.len();
    let comp_len = rows.first().map_or(0, Vec::len);
    let data: Vec<f32> = rows.iter().flatten().map(|&m| m as f32).collect();
    Tensor::from_vec(data, (g, comp_len), device)
}

/// Fold the `vars` gradients from one backward's `grads` into the running per-var
/// accumulator `acc` (summing across an accumulation window's prompts). A var
/// **absent** from `grads` marks `covered[i] = false` — candle's silent-skip
/// landmine — surfaced as a canary abort once the window's combined store is built.
fn fold_var_grads(
    vars: &[Var],
    grads: &GradStore,
    acc: &mut [Option<Tensor>],
    covered: &mut [bool],
) -> CandleResult<()> {
    for (i, v) in vars.iter().enumerate() {
        match grads.get(v.as_tensor()) {
            None => covered[i] = false,
            Some(g) => {
                acc[i] = Some(match acc[i].take() {
                    None => g.clone(),
                    Some(prev) => prev.add(g)?,
                });
            }
        }
    }
    Ok(())
}

/// Materialize the additive identity for every locally absent gradient before a
/// distributed reduction. Every rank must enter the tensor collective with the
/// same tensor count, shapes, dtypes, and devices even when one rank had no live
/// items (or no local backward reached a particular variable).
fn materialize_zero_grad_slots(vars: &[Var], acc: &mut [Option<Tensor>]) -> CandleResult<()> {
    for (slot, var) in acc.iter_mut().zip(vars) {
        if slot.is_none() {
            *slot = Some(var.as_tensor().zeros_like()?);
        }
    }
    Ok(())
}

fn reduce_accumulated_grads(
    comm: &dyn Comm,
    vars: &[Var],
    acc: &mut [Option<Tensor>],
    covered: &[bool],
) -> Result<f64, TrainerError> {
    for start in (0..vars.len()).step_by(GRAD_REDUCE_CHUNK) {
        let end = (start + GRAD_REDUCE_CHUNK).min(vars.len());
        let mut flat = Vec::with_capacity(end - start);
        for (i, slot) in acc.iter_mut().enumerate().take(end).skip(start) {
            flat.push(slot.take().ok_or_else(|| {
                TrainerError::Contract(format!(
                    "gradient slot {i} was not materialized before reduction"
                ))
            })?);
        }
        comm.all_reduce_sum(&mut flat)?;
        for (slot, g) in acc[start..end].iter_mut().zip(flat) {
            *slot = Some(g);
        }
    }
    let uncovered_local = covered.iter().filter(|c| !**c).count() as f64;
    Ok(comm.all_reduce_scalar_sum(uncovered_local)?)
}

/// Copy only trainable-var entries from a raw backward store into a fresh store.
///
/// Candle's raw [`GradStore`] can retain gradients for intermediate autograd
/// nodes. The trainer never consumes those entries: only trainable [`Var`]
/// gradients feed accumulation, coverage, clipping, and the optimizer. Compacting
/// immediately after a backward lets those intermediate entries drop before the
/// next memory probe phase while preserving missing-var semantics — absent
/// trainable grads stay absent so [`grad_coverage`] still fails loud.
fn compact_trainable_grad_store(vars: &[Var], mut raw: GradStore) -> CandleResult<GradStore> {
    let mut store = empty_grad_store(vars)?;
    for v in vars {
        store.remove(v.as_tensor());
        if let Some(g) = raw.remove(v.as_tensor()) {
            store.insert(v.as_tensor(), g);
        }
    }
    Ok(store)
}

/// Build the optimizer's gradient store for an accumulation window by moving the accumulated
/// per-var sums out of `acc`. A var marked uncovered (absent from some prompt's backward) is left
/// out entirely so [`grad_coverage`] flags it; its accumulator slot is still consumed because the
/// window will abort before any optimizer step.
fn combine_into_store(
    vars: &[Var],
    mut store: GradStore,
    acc: &mut [Option<Tensor>],
    covered: &[bool],
) -> GradStore {
    for (i, v) in vars.iter().enumerate() {
        store.remove(v.as_tensor());
        let grad = acc[i].take();
        if covered[i] {
            if let Some(g) = grad {
                store.insert(v.as_tensor(), g);
            }
        }
    }
    store
}

/// A [`GradStore`] with no trainable-var entries, for a rank whose local shard
/// produced no backward (DP, all-degenerate shard). candle exposes no public
/// `GradStore` constructor, so this runs a throwaway one-node backward over a
/// temporary scalar [`Var`]; the two stray entries it leaves (the temporary's
/// ids) are harmless — only trainable-var entries are read downstream.
fn empty_grad_store(vars: &[Var]) -> CandleResult<GradStore> {
    let device = vars
        .first()
        .map_or(Device::Cpu, |v| v.as_tensor().device().clone());
    let tmp = Var::zeros(1, DType::F32, &device)?;
    tmp.as_tensor().sum_all()?.backward()
}

/// The scalar group advantages as a detached `[G, 1]` tensor (broadcast over the
/// completion length in the surrogate).
fn advantages_tensor(advantages: &[f64], device: &Device) -> CandleResult<Tensor> {
    let adv: Vec<f32> = advantages.iter().map(|&a| a as f32).collect();
    Tensor::from_vec(adv, (advantages.len(), 1), device)
}

/// Validate that the rollout is rectangular with non-empty completions and a
/// well-formed `completion_lens`, and return `(num_seq, completion_len)`. The rows
/// stay a fixed rectangular width (EOS early-stop right-pads back to it); the real
/// per-sequence lengths live in [`Rollout::completion_lens`] and drive the loss mask,
/// which is why they are validated here. The rectangular shape is required because
/// [`Policy::token_logprobs`] returns a rectangular `[G, completion_len]` tensor. Run
/// this **before** decoding so a malformed rollout becomes a typed
/// [`TrainerError::Contract`] rather than a slice panic.
fn completion_dims(rollout: &Rollout) -> Result<(usize, usize), TrainerError> {
    if rollout.is_empty() {
        return Err(TrainerError::Contract(
            "empty rollout (no sequences)".into(),
        ));
    }
    if rollout.prompt_len == 0 {
        return Err(TrainerError::Contract(
            "rollout has no prompt context (prompt_len == 0); teacher-forced scoring \
             needs >= 1 prompt token to predict the first completion token"
                .into(),
        ));
    }
    let seq_len = rollout.token_ids[0].len();
    for ids in &rollout.token_ids {
        if ids.len() != seq_len {
            return Err(TrainerError::Contract(
                "ragged rollout (variable sequence length) is not supported yet".into(),
            ));
        }
    }
    let comp_len = seq_len.saturating_sub(rollout.prompt_len);
    if comp_len == 0 {
        return Err(TrainerError::Contract(format!(
            "rollout has no completion tokens (sequence length {seq_len} <= prompt_len {})",
            rollout.prompt_len
        )));
    }
    validate_completion_lens(rollout, comp_len)?;
    validate_rollout_logprobs(rollout)?;
    Ok((rollout.len(), comp_len))
}

/// Validate the optional behavior-log-prob capture: when present it must carry
/// one row per sequence with exactly `completion_lens[i]` entries in row `i`
/// (one log-prob per real draw) — a misaligned capture would silently pair
/// ratios with the wrong tokens in the rollout-ratio telemetry and the TIS
/// weights. `None` (a policy that does not capture) is always valid.
fn validate_rollout_logprobs(rollout: &Rollout) -> Result<(), TrainerError> {
    let Some(captured) = &rollout.rollout_logprobs else {
        return Ok(());
    };
    if captured.len() != rollout.len() {
        return Err(TrainerError::Contract(format!(
            "rollout has {} rollout_logprob rows for {} sequences",
            captured.len(),
            rollout.len()
        )));
    }
    for (i, (row, &len)) in captured.iter().zip(&rollout.completion_lens).enumerate() {
        if row.len() != len {
            return Err(TrainerError::Contract(format!(
                "rollout_logprobs row {i} has {} entries for completion_len {len} \
                 (one behavior log-prob per real draw)",
                row.len()
            )));
        }
    }
    Ok(())
}

/// Validate that `completion_lens` aligns with the rollout and stays within the
/// rectangular completion width. It drives the loss mask and the reward decode, so a
/// count that does not match the sequences, or a length past the width (which would
/// treat padding as real tokens and over-read the decode slice), is malformed
/// `Policy` output. A per-sequence length of `0..=comp_len` is allowed; an all-pad
/// (`0`) row is tolerated and counted by [`zero_mask_rows`]. Split out of
/// [`completion_dims`] to keep it under the cognitive-complexity bound.
fn validate_completion_lens(rollout: &Rollout, comp_len: usize) -> Result<(), TrainerError> {
    if rollout.completion_lens.len() != rollout.len() {
        return Err(TrainerError::Contract(format!(
            "rollout has {} completion_lens for {} sequences",
            rollout.completion_lens.len(),
            rollout.len()
        )));
    }
    if let Some(&bad) = rollout.completion_lens.iter().find(|&&len| len > comp_len) {
        return Err(TrainerError::Contract(format!(
            "completion_len {bad} exceeds the completion width {comp_len}"
        )));
    }
    Ok(())
}

/// Mean *real* completion length (EOS-inclusive tokens, per
/// [`Rollout::completion_lens`]) over the rollout — so the telemetry reports the
/// length the policy actually generated, not the padded width. With `eos_token_id ==
/// None` every length is the full width and this is the padded width, unchanged.
fn mean_completion_len(rollout: &Rollout) -> f32 {
    if rollout.is_empty() {
        return 0.0;
    }
    let total: usize = rollout.completion_lens.iter().sum();
    total as f32 / rollout.len() as f32
}

/// Reward `(mean, population-std)` over an already validated reward group.
fn reward_stats(rewards: &[f32]) -> (f32, f32) {
    debug_assert!(rewards.iter().all(|reward| reward.is_finite()));
    if rewards.is_empty() {
        return (0.0, 0.0);
    }
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let var = rewards
        .iter()
        .map(|&reward| (reward - mean).powi(2))
        .sum::<f32>()
        / n;
    if mean.is_finite() && var.is_finite() {
        return (mean, var.sqrt());
    }

    // Keep existing f32 arithmetic for ordinary groups and widen only when a
    // finite reward set overflowed an intermediate sum or square.
    let n = rewards.len() as f64;
    let mean = rewards.iter().map(|&reward| f64::from(reward)).sum::<f64>() / n;
    let var = rewards
        .iter()
        .map(|&reward| (f64::from(reward) - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean as f32, var.sqrt() as f32)
}

impl RewardStatsAcc {
    fn from_rewards(rewards: &[f64]) -> Self {
        debug_assert!(rewards.iter().all(|reward| reward.is_finite()));
        rewards.iter().copied().fold(
            Self {
                count: 0.0,
                sum: 0.0,
                sumsq: 0.0,
            },
            |acc, r| Self {
                count: acc.count + 1.0,
                sum: acc.sum + r,
                sumsq: acc.sumsq + r * r,
            },
        )
    }

    fn mean_std(self) -> (f64, f64) {
        if self.count <= 0.0 {
            return (0.0, 0.0);
        }
        let mean = self.sum / self.count;
        let std = if self.count < 2.0 {
            0.0
        } else {
            ((self.sumsq - (self.sum * self.sum / self.count)) / (self.count - 1.0))
                .max(0.0)
                .sqrt()
        };
        (mean, std)
    }
}

fn advantages_from_stats(rewards: &[f64], stats: RewardStatsAcc, scale: ScaleRewards) -> Vec<f64> {
    let (mean, std) = stats.mean_std();
    let denom = match scale {
        ScaleRewards::None => 1.0,
        ScaleRewards::Group => std + GROUP_STD_EPS,
    };
    rewards.iter().map(|&r| (r - mean) / denom).collect()
}

/// The raw importance ratio `exp(logp - logp_old)`. At `mu = 1` the snapshot
/// equals the current log-probs, so this is exactly `1`. Test-only reference:
/// the production path goes through [`level_ratio`], whose Token arm is this
/// plus the overflow-capping clamp (bit-identical for any sane log-ratio).
#[cfg(test)]
fn importance_ratio(logp: &Tensor, logp_old: &Tensor) -> CandleResult<Tensor> {
    logp.broadcast_sub(logp_old)?.exp()
}

/// Substitute the detached `logp_old` for `logp` at masked-out (padding)
/// positions, so every downstream `exp` argument is `0` there (see the
/// "EOS-padding gradient inertness" note on [`grpo_loss`]). Identical to
/// `logp` wherever `mask > 0`; a no-op for an all-ones mask.
fn substitute_padding(logp: &Tensor, logp_old: &Tensor, mask: &Tensor) -> CandleResult<Tensor> {
    mask.gt(0.0)?.where_cond(logp, logp_old)
}

/// The importance ratio at the configured [`ImportanceSamplingLevel`]:
///
/// - [`Token`](ImportanceSamplingLevel::Token): `exp(logp - logp_old)` per
///   token, shape `[G, comp_len]` — classic GRPO, bit-identical to the
///   pre-seam ratio.
/// - [`Sequence`](ImportanceSamplingLevel::Sequence): `exp` of the masked
///   mean per-token log-ratio, shape `[G, 1]` (GSPO; broadcasts over the
///   sequence's tokens downstream). Mirrors TRL:
///   `(log_ratio · mask).sum(-1) / mask.sum(-1).clamp(min=1)`. The
///   differentiable counterpart of [`crate::grpo::sequence_log_ratio`].
///
/// `logp` is expected padding-substituted (see [`substitute_padding`]) so a
/// divergent padding log-prob can neither overflow the token-level `exp` nor
/// poison the sequence-level masked sum.
///
/// The log-ratio is clamped to `±RATIO_LOG_CAP` **before** the `exp`: f32 `exp`
/// overflows to `+inf` near 88.7, and while every downstream consumer keeps the
/// loss *value* finite (the clip band / zero-advantage guard), `exp`'s backward
/// multiplies the upstream gradient by its own output — `0 · inf = NaN` — so an
/// overflowed ratio poisons every parameter gradient and aborts the run via the
/// canary even when the loss is finite. Clamping the *argument* keeps forward
/// values bit-identical for any `|log-ratio| < 60` (astronomically beyond the
/// `~±0.3` clip band where the surrogate already saturates) and makes the
/// gradient exactly the clip-saturated `0` beyond the cap.
fn level_ratio(
    logp: &Tensor,
    logp_old: &Tensor,
    mask: &Tensor,
    level: ImportanceSamplingLevel,
) -> CandleResult<Tensor> {
    /// See the function docs: keeps `exp` (and its backward) finite in f32.
    const RATIO_LOG_CAP: f64 = 60.0;
    match level {
        ImportanceSamplingLevel::Token => logp
            .broadcast_sub(logp_old)?
            .clamp(-RATIO_LOG_CAP, RATIO_LOG_CAP)?
            .exp(),
        ImportanceSamplingLevel::Sequence => {
            let log_ratio = logp.broadcast_sub(logp_old)?;
            let num = log_ratio.broadcast_mul(mask)?.sum_keepdim(D::Minus1)?;
            let denom_raw = mask.sum_keepdim(D::Minus1)?;
            let denom = denom_raw.maximum(&Tensor::ones_like(&denom_raw)?)?;
            num.broadcast_div(&denom)?
                .clamp(-RATIO_LOG_CAP, RATIO_LOG_CAP)?
                .exp()
        }
    }
}

/// Per-token PPO clipped surrogate `min(ratio * A, clip(ratio) * A)` with
/// asymmetric clip bands (DAPO clip-higher; symmetric when the two are equal).
/// The differentiable counterpart of [`crate::grpo::clipped_surrogate`].
fn clipped_surrogate_tensor(
    ratio: &Tensor,
    advantages: &Tensor,
    eps_low: f64,
    eps_high: f64,
) -> CandleResult<Tensor> {
    let unclipped = ratio.broadcast_mul(advantages)?;
    let clipped = ratio
        .clamp(1.0 - eps_low, 1.0 + eps_high)?
        .broadcast_mul(advantages)?;
    let surrogate = unclipped.minimum(&clipped)?;
    // A zero-advantage completion contributes exactly 0 in the scalar oracle (its
    // NaN-aware `min`), but candle's `minimum` is not NaN-aware, so a 0 advantage
    // times an overflowed (`inf`) importance ratio would leak `NaN` here. Select 0
    // for zero-advantage rows so the tensor matches the oracle (and stays finite).
    let nonzero_adv = advantages.ne(0.0)?.broadcast_as(surrogate.shape())?;
    nonzero_adv.where_cond(&surrogate, &surrogate.zeros_like()?)
}

/// Per-token Schulman k3 KL `exp(d) - d - 1`, `d = logp_ref - logp`. The
/// differentiable counterpart of [`crate::grpo::k3_kl`].
fn k3_kl_tensor(logp: &Tensor, logp_ref: &Tensor) -> CandleResult<Tensor> {
    let d = logp_ref.broadcast_sub(logp)?;
    d.exp()?.broadcast_sub(&d)?.affine(1.0, -1.0)
}

/// Mask-weighted reduction of a per-token objective, the differentiable
/// counterpart of [`crate::grpo::masked_mean`]. Returns a scalar tensor.
///
/// `values` is `[G, comp_len]`, or `[G, 1]` under sequence-level importance
/// sampling (a per-sequence objective broadcast against the mask — TRL shapes
/// its `per_token_loss` the same way). `dapo_norm` overrides the
/// [`LossType::Dapo`] denominator with the accumulation **window's** total
/// completion tokens; `None` uses this batch's active-token count (the
/// `grad_accum_steps == 1` / diagnostic form). Ignored by the other reductions.
fn masked_mean_tensor(
    values: &Tensor,
    mask: &Tensor,
    loss_type: LossType,
    dapo_norm: Option<f64>,
) -> CandleResult<Tensor> {
    // Hard-zero masked-out cells (mask <= 0) BEFORE multiplying, so a non-finite
    // value at a padding position cannot leak NaN via `0 * inf` — matching the
    // scalar oracle, which only sums `v * m` where `m > 0`. (Masks are 0/1 by
    // contract.) A [G, 1] sequence-level objective is first broadcast to the
    // mask's [G, comp_len] so the where_cond shapes agree.
    let values = if values.shape() == mask.shape() {
        values.clone()
    } else {
        values.broadcast_mul(&mask.ones_like()?)?
    };
    let keep = mask.gt(0.0)?;
    let kept = keep.where_cond(&values, &values.zeros_like()?)?;
    let masked = kept.broadcast_mul(mask)?;
    match loss_type {
        LossType::Grpo => {
            let per_seq_sum = masked.sum(D::Minus1)?;
            let mask_sums = mask.sum(D::Minus1)?;
            let denom = mask_sums.maximum(&Tensor::ones_like(&mask_sums)?)?;
            per_seq_sum.broadcast_div(&denom)?.mean(0)
        }
        LossType::DrGrpo => {
            // The denominator uses the mask WIDTH as the Dr.GRPO constant.
            // Equivalent to TRL's `B * max_completion_length` only because
            // ferrl rollouts are always right-padded to the fixed
            // `max_new_tokens` width; a future width-trimming rollout would
            // have to thread the configured maximum in here explicitly.
            let (num_seq, max_len) = mask.dims2()?;
            let total = masked.sum_all()?;
            total.affine(1.0 / (num_seq as f64 * max_len as f64), 0.0)
        }
        LossType::Dapo => {
            let norm = match dapo_norm {
                Some(n) => n.max(1.0),
                None => f64::from(scalar_f32(&mask.sum_all()?)?).max(1.0),
            };
            masked.sum_all()?.affine(1.0 / norm, 0.0)
        }
    }
}

/// The loss-shaping knobs [`grpo_loss`] reads, bundled so the signature stays
/// readable. `clip_eps_high` is the **effective** upper band (the caller has
/// already resolved the `None → symmetric` default); `dapo_norm` is the
/// accumulation window's total completion tokens (see [`masked_mean_tensor`]);
/// `tis_w` is the detached `[G, comp_len]` TIS weight (`None` when the
/// correction is off — token-level only, the `tis`+GSPO combination is rejected
/// at config validation).
struct LossCfg {
    clip_eps_low: f64,
    clip_eps_high: f64,
    beta: f64,
    loss_type: LossType,
    is_level: ImportanceSamplingLevel,
    dapo_norm: Option<f64>,
    tis_w: Option<Tensor>,
}

/// Assemble the GRPO loss for one inner step: the negative masked-mean of the
/// per-token objective `surrogate - beta * k3_kl` (the KL term only when a
/// reference policy is supplied). With `cfg.tis_w` set, the per-token surrogate
/// is first multiplied by the detached TIS weight (truncated importance
/// sampling — the behavior-policy correction; the KL penalty stays unweighted,
/// matching verl). This is the **single** differentiated loss the
/// trainer back-propagates; the trainer's inner step (`item_backward`) calls it on
/// the live policy log-probs and the in-module finite-difference gradcheck
/// (`gradcheck_*`) calls it on a tiny `f64` stand-in, so the gradcheck verifies
/// candle's analytic gradient of *exactly* this expression w.r.t. the `LoRA`
/// parameters.
///
/// `logp` is `[num_seq, comp_len]`; `logp_old` / `logp_ref` share that shape and
/// are detached constants (the only trainable path is `logp`); `advantages` is a
/// detached `[num_seq, 1]` column broadcast over the completion length; `mask` is
/// `[num_seq, comp_len]`. Under [`ImportanceSamplingLevel::Sequence`] the ratio —
/// and so the surrogate — is a `[num_seq, 1]` column shared by the sequence's
/// tokens (GSPO); the KL penalty stays per-token, broadcasting the surrogate over
/// the completion (TRL adds the two the same way). Returns a scalar loss tensor.
///
/// # EOS-padding gradient inertness
///
/// Masked-out (EOS-padding) positions are dropped from the reduction by
/// [`masked_mean_tensor`], so they cannot change the loss *value*. But they still
/// flow through the `exp` in the importance ratio and the k3 KL, and `exp` overflows
/// f32 at an argument `> ~88`: at a padding position whose log-prob diverges that far,
/// `exp` is `inf`, and its backward `grad * node = 0 * inf` is `NaN` — the upstream
/// gradient is correctly `0` (the cell is masked), but `exp`'s local derivative is
/// `inf`, poisoning the gradient of an otherwise-inert padding token (the canary would
/// then fail the whole step loud). So force the `exp` arguments to `0` at padding by
/// substituting the detached `logp_old` / `logp_ref` for `logp` there (`exp(0) = 1`,
/// finite). Real positions (`keep == 1`) are untouched — the differentiated loss is
/// identical — and with an all-ones mask (`eos_token_id == None`) this is a no-op, so
/// it is bit-identical to the pre-masking loss. This makes padding gradient-inert
/// *unconditionally* rather than only fail-loud in the overflow corner. (The
/// sequence-level masked sum excludes padding by construction, but the substitution
/// is applied uniformly so both levels share one guarantee.)
fn grpo_loss(
    logp: &Tensor,
    logp_old: &Tensor,
    logp_ref: Option<&Tensor>,
    advantages: &Tensor,
    mask: &Tensor,
    cfg: &LossCfg,
) -> CandleResult<Tensor> {
    let keep = mask.gt(0.0)?;
    // At padding, substitute logp_old so the ratio's exp argument is 0 (see the
    // "EOS-padding gradient inertness" note); identical to `logp` where keep == 1.
    let logp_ratio = substitute_padding(logp, logp_old, mask)?;
    let ratio = level_ratio(&logp_ratio, logp_old, mask, cfg.is_level)?;
    let surrogate =
        clipped_surrogate_tensor(&ratio, advantages, cfg.clip_eps_low, cfg.clip_eps_high)?;
    // Truncated importance sampling: re-weight each token's surrogate toward the
    // scoring distribution by the detached behavior-correction weight (the KL
    // penalty below stays unweighted, matching verl). A detached constant — it
    // scales the gradient, it is never differentiated through; `None` is the
    // bit-identical off path.
    let surrogate = match &cfg.tis_w {
        None => surrogate,
        Some(w) => surrogate.broadcast_mul(w)?,
    };
    let per_token = match logp_ref {
        None => surrogate,
        Some(logp_ref) => {
            // At padding, substitute logp_ref so the k3 KL's exp argument is 0.
            let logp_kl = keep.where_cond(logp, logp_ref)?;
            let penalty = k3_kl_tensor(&logp_kl, logp_ref)?.affine(cfg.beta, 0.0)?;
            surrogate.broadcast_sub(&penalty)?
        }
    };
    masked_mean_tensor(&per_token, mask, cfg.loss_type, cfg.dapo_norm)?.neg()
}

/// Fraction of the batch whose surrogate `min` selected the clipped term.
///
/// For a per-token ratio (`[G, comp_len]`) this is the masked token mean of the
/// clip indicator; for a per-sequence ratio (`[G, 1]`, the GSPO level) it is the
/// **plain mean over sequences** — mirroring TRL's `masked_batch_mean`, which
/// special-cases the `(B, 1)` shape to `x.mean()` (a clipped sequence counts
/// once, not once per token).
fn clip_fraction(
    ratio: &Tensor,
    advantages: &Tensor,
    eps_low: f64,
    eps_high: f64,
    mask: &Tensor,
) -> CandleResult<f32> {
    let unclipped = ratio.broadcast_mul(advantages)?;
    let clipped = ratio
        .clamp(1.0 - eps_low, 1.0 + eps_high)?
        .broadcast_mul(advantages)?;
    let was_clipped = clipped.lt(&unclipped)?.to_dtype(DType::F32)?;
    if ratio.dims2()?.1 == 1 {
        return scalar_f32(&was_clipped.mean_all()?);
    }
    let num = scalar_f32(&was_clipped.broadcast_mul(mask)?.sum_all()?)?;
    let den = scalar_f32(&mask.sum_all()?)?;
    Ok(if den > 0.0 { num / den } else { 0.0 })
}

/// Scale every trainable-var gradient in `store` by `scale` (global-norm
/// clipping). Vars absent from the store are skipped — the canary has already
/// guaranteed coverage by the time this runs.
fn scale_var_grads(vars: &[Var], store: &mut GradStore, scale: f64) -> CandleResult<()> {
    for v in vars {
        if let Some(g) = store.get(v.as_tensor()) {
            let scaled = g.affine(scale, 0.0)?;
            store.insert(v.as_tensor(), scaled);
        }
    }
    Ok(())
}

/// Global L2 norm over the trainable vars' gradients (pre-clip). Vars absent
/// from the store contribute `0` (the canary has already guaranteed coverage).
/// Each square-sum is accumulated in **f64**: an f32 `sqr` overflows to `inf`
/// for elements above ~1.8e19, and an `inf` norm would turn the clip factor
/// `max / norm` into a silent all-zero gradient scale.
fn global_grad_norm(vars: &[Var], grads: &GradStore) -> CandleResult<f64> {
    let mut sq = 0.0;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let g64 = g.to_dtype(DType::F64)?;
            sq += g64.sqr()?.sum_all()?.to_scalar::<f64>()?;
        }
    }
    Ok(sq.sqrt())
}

/// Read a scalar tensor as `f32`, upcasting first so a bf16/f16 tensor does not
/// error in `to_scalar`.
fn scalar_f32(t: &Tensor) -> CandleResult<f32> {
    t.to_dtype(DType::F32)?.to_scalar::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpo::{clipped_surrogate, k3_kl, masked_mean};
    use approx::assert_relative_eq;

    const TOL: f32 = 1e-6;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn gen_config_from_trainer_config_mirrors_the_rollout_fields() {
        let config = TrainerConfig {
            group_size: 12,
            max_new_tokens: 48,
            temperature: 0.7,
            eos_token_id: Some(151_645),
            ..TrainerConfig::default()
        };
        let gen = GenConfig::from(&config);
        // The exact config the rollout site used to hand-build, now in one place.
        // (`GenConfig` is not `PartialEq`, so check field-by-field.)
        assert_eq!(gen.group_size, 12);
        assert_eq!(gen.max_new_tokens, 48);
        assert_eq!(gen.temperature, 0.7);
        assert_eq!(gen.eos_token_id, Some(151_645));
        assert!(gen.eval_sampling.is_none());
    }

    #[test]
    fn builder_default_equals_trainer_config_default() {
        let built = TrainerConfig::builder().build();
        assert_eq!(
            serde_json::to_value(&built).unwrap(),
            serde_json::to_value(TrainerConfig::default()).unwrap()
        );
    }

    #[test]
    fn builder_equals_the_equivalent_struct_literal() {
        let built = TrainerConfig::builder()
            .steps(200)
            .group_size(16)
            .max_new_tokens(64)
            .temperature(0.8)
            .mu(2)
            .beta(0.04)
            .clip_eps(0.2)
            .clip_eps_high(Some(0.28))
            .importance_sampling_level(ImportanceSamplingLevel::Sequence)
            .lr(1e-6)
            .weight_decay(0.01)
            .adam_beta1(0.9)
            .adam_beta2(0.95)
            .warmup_steps(20)
            .max_grad_norm(Some(1.0))
            .truncation_masking(false)
            .tis(true)
            .tis_imp_ratio_cap(2.0)
            .loss_type(LossType::Grpo)
            .scale_rewards(ScaleRewards::None)
            .reward_group_scope(RewardGroupScope::Local)
            .grad_accum_steps(4)
            .backward_microbatch_size(2)
            .checkpoint_every(Some(50))
            .candidate_log_top_k(2)
            .gpu_memory_probe(true)
            .eos_token_id(Some(151_645))
            .build();
        let literal = TrainerConfig {
            steps: 200,
            group_size: 16,
            max_new_tokens: 64,
            temperature: 0.8,
            mu: 2,
            beta: 0.04,
            beta_schedule: None,
            clip_eps: 0.2,
            clip_eps_high: Some(0.28),
            importance_sampling_level: ImportanceSamplingLevel::Sequence,
            lr: 1e-6,
            lr_schedule: None,
            weight_decay: 0.01,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            warmup_steps: 20,
            max_grad_norm: Some(1.0),
            truncation_masking: false,
            tis: true,
            tis_imp_ratio_cap: 2.0,
            loss_type: LossType::Grpo,
            scale_rewards: ScaleRewards::None,
            reward_group_scope: RewardGroupScope::Local,
            grad_accum_steps: 4,
            backward_microbatch_size: 2,
            checkpoint_every: Some(50),
            candidate_log_top_k: 2,
            gpu_memory_probe: true,
            eos_token_id: Some(151_645),
        };
        assert_eq!(
            serde_json::to_value(&built).unwrap(),
            serde_json::to_value(&literal).unwrap()
        );
    }

    fn gpu_memory_snapshot_for_test(used_bytes: u64) -> GpuMemorySnapshot {
        GpuMemorySnapshot {
            free_bytes: 100 - used_bytes,
            total_bytes: 100,
            used_bytes,
        }
    }

    #[test]
    fn step_gpu_memory_persists_stable_phase_events() {
        let mut mem = StepGpuMemory::new(true);
        mem.record_snapshot("step_start", gpu_memory_snapshot_for_test(10));
        mem.record_snapshot("rollout_prefill_end", gpu_memory_snapshot_for_test(25));
        mem.record_snapshot("step_end", gpu_memory_snapshot_for_test(20));

        let mut metrics = Metrics::at_step(0);
        mem.apply(&mut metrics);
        assert_eq!(
            (
                metrics.cuda_mem_start_used_bytes,
                metrics.cuda_mem_peak_used_bytes,
                metrics.cuda_mem_end_used_bytes,
                metrics.cuda_mem_total_bytes,
                metrics.cuda_mem_peak_delta_bytes,
            ),
            (10, 25, 20, 100, 15)
        );
        let events: Vec<(&str, u64)> = metrics
            .cuda_mem_probe_events
            .iter()
            .map(|event| (event.phase.as_str(), event.peak_delta_bytes))
            .collect();
        assert_eq!(
            events,
            [
                ("step_start", 0),
                ("rollout_prefill_end", 15),
                ("step_end", 15),
            ]
        );
    }

    #[test]
    fn step_gpu_memory_persists_decoder_cache_snapshots_when_enabled() {
        let mut mem = StepGpuMemory::new(true);
        ModelTelemetryRecorder::record_decoder_cache(
            &mut mem,
            vec![DecoderCacheSnapshot {
                phase: "rollout_decode_end".to_string(),
                layer_index: 0,
                kind: "sliding_attention".to_string(),
                seen_tokens: 6,
                retained_tokens: 3,
                max_retained_tokens: Some(3),
            }],
        );

        let mut metrics = Metrics::at_step(0);
        mem.apply(&mut metrics);
        assert_eq!(metrics.decoder_cache_snapshots.len(), 1);
        assert_eq!(
            metrics.decoder_cache_snapshots[0].phase,
            "rollout_decode_end"
        );
        assert_eq!(metrics.decoder_cache_snapshots[0].seen_tokens, 6);
        assert_eq!(metrics.decoder_cache_snapshots[0].retained_tokens, 3);

        let mut disabled = StepGpuMemory::new(false);
        ModelTelemetryRecorder::record_decoder_cache(
            &mut disabled,
            vec![DecoderCacheSnapshot {
                phase: "rollout_decode_end".to_string(),
                layer_index: 0,
                kind: "sliding_attention".to_string(),
                seen_tokens: 6,
                retained_tokens: 3,
                max_retained_tokens: Some(3),
            }],
        );
        let mut disabled_metrics = Metrics::at_step(0);
        disabled.apply(&mut disabled_metrics);
        assert!(disabled_metrics.decoder_cache_snapshots.is_empty());
    }

    fn mat(rows: &[&[f32]]) -> Tensor {
        let r = rows.len();
        let c = rows[0].len();
        let data: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_vec(data, (r, c), &cpu()).unwrap()
    }

    #[test]
    fn importance_ratio_is_one_when_logp_equals_old() {
        // mu = 1: the snapshot equals the current log-probs -> ratio == 1.
        let x = mat(&[&[-1.0, -2.0], &[-0.5, -3.0]]);
        let r = importance_ratio(&x, &x.detach()).unwrap();
        let flat = r.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in flat {
            assert_relative_eq!(v, 1.0, epsilon = TOL);
        }
    }

    #[test]
    fn clipped_surrogate_tensor_matches_scalar_oracle() {
        // Grid over the cases the scalar oracle tests cover (incl. negative A and
        // out-of-band ratios), broadcasting a [2,1] advantage over [2,3] ratios.
        let ratio = mat(&[&[1.0, 1.5, 0.5], &[1.0, 1.5, 0.5]]);
        let adv = Tensor::from_vec(vec![0.5f32, -0.5], (2, 1), &cpu()).unwrap();
        let eps = 0.2;
        let got = clipped_surrogate_tensor(&ratio, &adv, eps, eps).unwrap();
        let got = got.to_vec2::<f32>().unwrap();
        let advs = [0.5f64, -0.5];
        let ratios = [1.0f64, 1.5, 0.5];
        for (i, &a) in advs.iter().enumerate() {
            for (j, &rt) in ratios.iter().enumerate() {
                let want = clipped_surrogate(rt, a, eps, eps) as f32;
                assert_relative_eq!(got[i][j], want, epsilon = TOL);
            }
        }
        // Clip-higher pass: ratio 1.25 sits between the symmetric edge (1.2)
        // and the widened edge (1.28), so a band swap or ignored eps_high
        // produces a different surrogate at that cell.
        let ratio = mat(&[&[1.25, 1.5, 0.5], &[1.25, 1.5, 0.5]]);
        let got = clipped_surrogate_tensor(&ratio, &adv, 0.2, 0.28).unwrap();
        let got = got.to_vec2::<f32>().unwrap();
        let ratios = [1.25f64, 1.5, 0.5];
        for (i, &a) in advs.iter().enumerate() {
            for (j, &rt) in ratios.iter().enumerate() {
                let want = clipped_surrogate(rt, a, 0.2, 0.28) as f32;
                assert_relative_eq!(got[i][j], want, epsilon = TOL);
            }
        }
    }

    fn assert_f64s_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_relative_eq!(actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn clipped_surrogate_tensor_zero_advantage_inf_ratio_matches_oracle() {
        // A zero-advantage row with an overflowed (inf) ratio: the NaN-aware scalar
        // oracle gives 0; the tensor must too (no 0*inf -> NaN leak).
        let ratio = Tensor::from_vec(vec![f32::INFINITY, 2.0], (2, 1), &cpu()).unwrap();
        let adv = Tensor::from_vec(vec![0.0f32, 0.5], (2, 1), &cpu()).unwrap();
        let got = clipped_surrogate_tensor(&ratio, &adv, 0.2, 0.2).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(got[0].is_finite(), "0*inf leaked NaN: {}", got[0]);
        assert_relative_eq!(
            got[0],
            clipped_surrogate(f64::INFINITY, 0.0, 0.2, 0.2) as f32,
            epsilon = TOL
        );
        assert_relative_eq!(
            got[1],
            clipped_surrogate(2.0, 0.5, 0.2, 0.2) as f32,
            epsilon = TOL
        );
    }

    #[test]
    fn k3_kl_tensor_matches_scalar_oracle() {
        let logp = mat(&[&[-1.0, -0.4], &[-2.0, -0.1]]);
        let logp_ref = mat(&[&[-1.0, -0.5], &[1.0, -0.3]]);
        let got = k3_kl_tensor(&logp, &logp_ref)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let lp = [[-1.0f64, -0.4], [-2.0, -0.1]];
        let lr = [[-1.0f64, -0.5], [1.0, -0.3]];
        for i in 0..2 {
            for j in 0..2 {
                let want = k3_kl(lp[i][j], lr[i][j]) as f32;
                assert_relative_eq!(got[i][j], want, epsilon = TOL);
            }
        }
    }

    #[test]
    fn masked_mean_tensor_matches_scalar_oracle_both_reductions() {
        let values = mat(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 0.0]]);
        let mask = mat(&[&[1.0, 1.0, 1.0], &[1.0, 1.0, 0.0]]);
        let v = vec![vec![1.0f64, 2.0, 3.0], vec![4.0, 5.0, 0.0]];
        let m = vec![vec![1.0f64, 1.0, 1.0], vec![1.0, 1.0, 0.0]];

        let grpo =
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::Grpo, None).unwrap()).unwrap();
        assert_relative_eq!(
            grpo,
            masked_mean(&v, &m, LossType::Grpo) as f32,
            epsilon = TOL
        );

        let dr = scalar_f32(&masked_mean_tensor(&values, &mask, LossType::DrGrpo, None).unwrap())
            .unwrap();
        assert_relative_eq!(
            dr,
            masked_mean(&v, &m, LossType::DrGrpo) as f32,
            epsilon = TOL
        );
    }

    #[test]
    fn masked_mean_tensor_ignores_nonfinite_masked_cell() {
        // A NaN/inf value at a masked-out (m == 0) position must not leak into the
        // reduction (0 * inf): the tensor must match the scalar oracle and stay
        // finite. This is the P4-padding guard.
        let values = mat(&[&[1.0, f32::NAN], &[5.0, f32::INFINITY]]);
        let mask = mat(&[&[1.0, 0.0], &[1.0, 0.0]]);
        let v = vec![vec![1.0f64, f64::NAN], vec![5.0, f64::INFINITY]];
        let m = vec![vec![1.0f64, 0.0], vec![1.0, 0.0]];
        let grpo =
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::Grpo, None).unwrap()).unwrap();
        assert!(grpo.is_finite(), "masked-out NaN/inf leaked: {grpo}");
        assert_relative_eq!(
            grpo,
            masked_mean(&v, &m, LossType::Grpo) as f32,
            epsilon = TOL
        );
        let dr = scalar_f32(&masked_mean_tensor(&values, &mask, LossType::DrGrpo, None).unwrap())
            .unwrap();
        assert!(dr.is_finite());
        assert_relative_eq!(
            dr,
            masked_mean(&v, &m, LossType::DrGrpo) as f32,
            epsilon = TOL
        );
    }

    #[test]
    fn config_validate_accepts_default() {
        assert!(TrainerConfig::default().validate().is_ok());
    }

    #[test]
    fn config_validate_rejects_bad_settings() {
        let bad = |mutate: fn(&mut TrainerConfig)| {
            let mut c = TrainerConfig::default();
            mutate(&mut c);
            assert!(
                matches!(c.validate(), Err(TrainerError::InvalidConfig(_))),
                "config should have been rejected"
            );
        };
        bad(|c| c.mu = 0);
        bad(|c| c.group_size = 0);
        bad(|c| c.max_new_tokens = 0);
        bad(|c| c.temperature = 0.0);
        bad(|c| c.temperature = f64::NAN);
        bad(|c| c.lr = -1.0);
        bad(|c| c.lr = f64::INFINITY);
        bad(|c| c.weight_decay = -0.1);
        bad(|c| c.beta = -1.0);
        bad(|c| c.beta_schedule = Some(ScalarSchedule { points: vec![] }));
        bad(|c| {
            c.beta_schedule = Some(ScalarSchedule {
                points: vec![SchedulePoint {
                    step: 1,
                    value: 0.0,
                }],
            });
        });
        bad(|c| {
            c.beta_schedule = Some(ScalarSchedule {
                points: vec![
                    SchedulePoint {
                        step: 0,
                        value: 0.0,
                    },
                    SchedulePoint {
                        step: 0,
                        value: 0.1,
                    },
                ],
            });
        });
        bad(|c| {
            c.beta_schedule = Some(ScalarSchedule {
                points: vec![SchedulePoint {
                    step: 0,
                    value: -0.1,
                }],
            });
        });
        bad(|c| {
            c.lr_schedule = Some(ScalarSchedule::constant(0.001));
            c.warmup_steps = 2;
        });
        bad(|c| c.clip_eps = f64::NAN);
        bad(|c| c.clip_eps = 1.0);
        bad(|c| c.clip_eps = 2.0);
        bad(|c| c.grad_accum_steps = 0);
        bad(|c| c.checkpoint_every = Some(0));
        bad(|c| c.clip_eps_high = Some(f64::NAN));
        bad(|c| c.clip_eps_high = Some(-0.1));
        bad(|c| c.adam_beta1 = 1.0);
        bad(|c| c.adam_beta2 = -0.1);
        bad(|c| c.adam_beta2 = f64::NAN);
        bad(|c| c.max_grad_norm = Some(0.0));
        bad(|c| c.max_grad_norm = Some(f64::INFINITY));
    }

    #[test]
    fn config_validate_accepts_eos_token_id() {
        // The loss mask + length-aware decode now honor the EOS padding, so a `Some`
        // eos_token_id is a valid run (the PR3 guard-lift; before it `validate`
        // rejected it to avoid silently scoring the padding).
        let cfg = TrainerConfig {
            eos_token_id: Some(151_643),
            ..TrainerConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_accepts_checkpoint_every() {
        let with = TrainerConfig {
            checkpoint_every: Some(5),
            ..TrainerConfig::default()
        };
        assert!(with.validate().is_ok());
        let without = TrainerConfig {
            checkpoint_every: None,
            ..TrainerConfig::default()
        };
        assert!(without.validate().is_ok());
    }

    #[test]
    fn clip_fraction_is_zero_at_ratio_one() {
        let ratio = mat(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let adv = Tensor::from_vec(vec![0.5f32, -0.5], (2, 1), &cpu()).unwrap();
        let mask = mat(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let frac = clip_fraction(&ratio, &adv, 0.2, 0.2, &mask).unwrap();
        assert_relative_eq!(frac, 0.0, epsilon = TOL);
    }

    #[test]
    fn clip_fraction_counts_binding_tokens() {
        // A>0, ratio 1.5 > 1.2 -> clipped term binds (lower). A<0 row: ratio 1.5
        // -> unclipped is lower, so the clip does NOT bind. 1 of 4 tokens clipped.
        let ratio = mat(&[&[1.5, 1.0], &[1.5, 1.0]]);
        let adv = Tensor::from_vec(vec![0.5f32, -0.5], (2, 1), &cpu()).unwrap();
        let mask = mat(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let frac = clip_fraction(&ratio, &adv, 0.2, 0.2, &mask).unwrap();
        assert_relative_eq!(frac, 0.25, epsilon = TOL);
    }

    #[test]
    fn clip_fraction_honors_the_widened_upper_band() {
        // ratio 1.25 with A>0: clipped at symmetric 0.2 (1.25 > 1.2) but inside
        // the clip-higher band 0.28 (1.25 < 1.28) — the asymmetric knob is live.
        let ratio = mat(&[&[1.25, 1.0]]);
        let adv = Tensor::from_vec(vec![0.5f32], (1, 1), &cpu()).unwrap();
        let mask = mat(&[&[1.0, 1.0]]);
        let sym = clip_fraction(&ratio, &adv, 0.2, 0.2, &mask).unwrap();
        let asym = clip_fraction(&ratio, &adv, 0.2, 0.28, &mask).unwrap();
        assert_relative_eq!(sym, 0.5, epsilon = TOL);
        assert_relative_eq!(asym, 0.0, epsilon = TOL);
    }

    #[test]
    fn clip_eps_high_defaults_to_symmetric() {
        let cfg = TrainerConfig::default();
        assert_relative_eq!(cfg.clip_eps_high_eff(), cfg.clip_eps, epsilon = 1e-12);
        let asym = TrainerConfig {
            clip_eps_high: Some(0.28),
            ..TrainerConfig::default()
        };
        assert_relative_eq!(asym.clip_eps_high_eff(), 0.28, epsilon = 1e-12);
    }

    #[test]
    fn lr_at_warms_up_linearly_then_holds() {
        let cfg = TrainerConfig {
            lr: 1.0,
            warmup_steps: 4,
            ..TrainerConfig::default()
        };
        // Steps 0..3 ramp (t+1)/4; step 3 reaches full lr exactly; constant after.
        assert_relative_eq!(cfg.lr_at(0), 0.25, epsilon = 1e-12);
        assert_relative_eq!(cfg.lr_at(1), 0.5, epsilon = 1e-12);
        assert_relative_eq!(cfg.lr_at(2), 0.75, epsilon = 1e-12);
        assert_relative_eq!(cfg.lr_at(3), 1.0, epsilon = 1e-12);
        assert_relative_eq!(cfg.lr_at(100), 1.0, epsilon = 1e-12);
        // warmup_steps == 0 disables the ramp entirely (bit-identical legacy).
        let off = TrainerConfig {
            lr: 1.0,
            warmup_steps: 0,
            ..TrainerConfig::default()
        };
        assert_relative_eq!(off.lr_at(0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn scalar_schedule_interpolates_and_holds_last_point() {
        let schedule = ScalarSchedule {
            points: vec![
                SchedulePoint {
                    step: 0,
                    value: 0.0,
                },
                SchedulePoint {
                    step: 4,
                    value: 0.04,
                },
                SchedulePoint {
                    step: 8,
                    value: 0.02,
                },
            ],
        };
        schedule.validate_nonnegative("test", 9).unwrap();

        assert_relative_eq!(schedule.at(0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(schedule.at(2), 0.02, epsilon = 1e-12);
        assert_relative_eq!(schedule.at(4), 0.04, epsilon = 1e-12);
        assert_relative_eq!(schedule.at(6), 0.03, epsilon = 1e-12);
        assert_relative_eq!(schedule.at(20), 0.02, epsilon = 1e-12);
    }

    #[test]
    fn beta_and_lr_schedules_override_legacy_constants() {
        let cfg = TrainerConfig {
            beta: 0.2,
            beta_schedule: Some(ScalarSchedule::linear(0.0, 0.04, 4)),
            lr: 1.0,
            lr_schedule: Some(ScalarSchedule::linear(0.0, 0.001, 2)),
            ..TrainerConfig::default()
        };

        assert_eq!(
            (cfg.validate().is_ok(), cfg.requires_reference_policy()),
            (true, true)
        );
        assert_f64s_close(
            &[cfg.beta_at(0), cfg.beta_at(2), cfg.beta_at(8)],
            &[0.0, 0.02, 0.04],
        );
        assert_f64s_close(
            &[cfg.lr_at(0), cfg.lr_at(1), cfg.lr_at(3)],
            &[0.0, 0.0005, 0.001],
        );
    }

    #[test]
    fn all_zero_beta_schedule_does_not_require_reference_policy() {
        let cfg = TrainerConfig {
            beta: 0.2,
            beta_schedule: Some(ScalarSchedule::constant(0.0)),
            ..TrainerConfig::default()
        };

        assert!(cfg.validate().is_ok());
        assert_relative_eq!(cfg.beta_at(0), 0.0, epsilon = 1e-12);
        assert!(!cfg.requires_reference_policy());
    }

    #[test]
    fn resume_decision_round_trips_and_keeps_outcomes_disjoint() {
        use ResumeDecision::{Fresh, Resume, ScanFailed};
        // The DP `resume_latest` broadcast encodes rank 0's three-way decision in one
        // f64. Every outcome must round-trip exactly (it is summed against the peers'
        // zero contributions) — including Resume(0), the corner the `+1` sentinel keeps
        // distinct from a fresh start.
        for d in [
            Fresh,
            ScanFailed,
            Resume(0),
            Resume(1),
            Resume(49),
            Resume(1_000_000),
        ] {
            assert_eq!(ResumeDecision::decode(d.encode()), d, "round-trip {d:?}");
        }
        // Fresh is the additive identity peers contribute, and the three outcome kinds
        // must occupy three DISTINCT wire values (so the broadcast sum is unambiguous).
        assert_eq!(Fresh.encode(), 0.0);
        let mut wires = vec![Fresh.encode(), ScanFailed.encode(), Resume(0).encode()];
        wires.sort_by(f64::total_cmp);
        wires.dedup();
        assert_eq!(
            wires.len(),
            3,
            "Fresh / ScanFailed / Resume(0) must be distinct wires"
        );
    }

    #[test]
    fn mask_truncated_rows_detects_only_full_width_non_eos_rows() {
        const EOS: u32 = 9;
        // prompt_len 1, comp width 3. Rows: (a) EOS at position 1 (len 2, padded
        // with EOS) — terminated; (b) full width, last token != EOS — TRUNCATED;
        // (c) full width, last token == EOS exactly at the boundary — terminated.
        let rollout = Rollout {
            token_ids: vec![vec![5, 1, EOS, EOS], vec![5, 1, 2, 3], vec![5, 1, 2, EOS]],
            prompt_len: 1,
            completion_lens: vec![2, 3, 3],
            rollout_logprobs: None,
        };
        let mut rows = length_mask_rows(&rollout, 3);
        let n = mask_truncated_rows(&rollout, 3, Some(EOS), &mut rows);
        assert_eq!(n, 1);
        assert_eq!(rows[0], vec![1.0, 1.0, 0.0], "terminated row untouched");
        assert_eq!(rows[1], vec![0.0, 0.0, 0.0], "truncated row fully masked");
        assert_eq!(rows[2], vec![1.0, 1.0, 1.0], "boundary-EOS row untouched");
        assert_eq!(zero_mask_rows(&rows), 1, "masked row shows up as dropped");
    }

    #[test]
    fn mask_truncated_rows_is_inert_without_an_eos_token() {
        // Without an EOS token the knob is inert — nothing can be truncated.
        let rollout = Rollout {
            token_ids: vec![vec![5, 1, 2, 3], vec![5, 1, 2, 9]],
            prompt_len: 1,
            completion_lens: vec![3, 3],
            rollout_logprobs: None,
        };
        let mut rows = length_mask_rows(&rollout, 3);
        assert_eq!(mask_truncated_rows(&rollout, 3, None, &mut rows), 0);
        assert_eq!(rows, length_mask_rows(&rollout, 3));
    }

    #[test]
    fn level_ratio_sequence_matches_scalar_oracle_and_token_is_identity() {
        use crate::grpo::sequence_log_ratio;
        let logp = mat(&[&[-1.0, -2.0, -0.4], &[-0.5, -0.25, -0.75]]);
        let old = mat(&[&[-1.2, -1.6, -1.0], &[-0.5, -0.25, -0.75]]);
        let mask = mat(&[&[1.0, 1.0, 0.0], &[1.0, 1.0, 1.0]]);
        let seq = level_ratio(&logp, &old, &mask, ImportanceSamplingLevel::Sequence).unwrap();
        assert_eq!(seq.dims(), &[2, 1], "sequence level is a [G, 1] column");
        let got = seq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rows: [(&[f64], &[f64], &[f64]); 2] = [
            (&[-1.0, -2.0, -0.4], &[-1.2, -1.6, -1.0], &[1.0, 1.0, 0.0]),
            (
                &[-0.5, -0.25, -0.75],
                &[-0.5, -0.25, -0.75],
                &[1.0, 1.0, 1.0],
            ),
        ];
        for (i, (lp, lo, m)) in rows.iter().enumerate() {
            let want = sequence_log_ratio(lp, lo, m).exp() as f32;
            assert_relative_eq!(got[i], want, epsilon = TOL);
        }
        // Token level is exactly the elementwise importance ratio (the pre-seam
        // behavior, bit-identical).
        let tok = level_ratio(&logp, &old, &mask, ImportanceSamplingLevel::Token).unwrap();
        let want = importance_ratio(&logp, &old).unwrap();
        assert_eq!(
            tok.to_vec2::<f32>().unwrap(),
            want.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn masked_mean_tensor_dapo_matches_oracle_and_window_normalizer() {
        let values = mat(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 0.0]]);
        let mask = mat(&[&[1.0, 1.0, 1.0], &[1.0, 1.0, 0.0]]);
        let v = vec![vec![1.0f64, 2.0, 3.0], vec![4.0, 5.0, 0.0]];
        let m = vec![vec![1.0f64, 1.0, 1.0], vec![1.0, 1.0, 0.0]];
        // Single-batch form (None) matches the scalar oracle.
        let one =
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::Dapo, None).unwrap()).unwrap();
        assert_relative_eq!(
            one,
            masked_mean(&v, &m, LossType::Dapo) as f32,
            epsilon = TOL
        );
        // A window normalizer overrides the denominator: total 15 over 12 tokens.
        let window =
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::Dapo, Some(12.0)).unwrap())
                .unwrap();
        assert_relative_eq!(window, 15.0 / 12.0, epsilon = TOL);
    }

    #[test]
    fn grpo_loss_dapo_equals_grpo_on_full_width_loss_and_grads() {
        // Full-width equal-length batch: the Dapo and Grpo reductions are the
        // same mathematical quantity (total / (G·len)), so the default switch
        // preserves every full-width trajectory. Pinned here on BOTH the loss
        // value and the gradient (approx — the op orders differ in float).
        let logp = Var::from_tensor(&mat(&[&[-1.0, -2.0, -0.4], &[-0.5, -0.25, -0.75]])).unwrap();
        let old = logp.as_tensor().affine(1.0, 0.1).unwrap().detach();
        let adv = Tensor::from_vec(vec![0.8f32, -0.3], (2, 1), &cpu()).unwrap();
        let mask = mat(&[&[1.0; 3], &[1.0; 3]]);
        let run = |loss_type: LossType| -> (f32, Vec<Vec<f32>>) {
            let cfg = LossCfg {
                clip_eps_low: 0.2,
                clip_eps_high: 0.2,
                beta: 0.0,
                loss_type,
                is_level: ImportanceSamplingLevel::Token,
                dapo_norm: None,
                tis_w: None,
            };
            let loss = grpo_loss(logp.as_tensor(), &old, None, &adv, &mask, &cfg).unwrap();
            let v = scalar_f32(&loss).unwrap();
            let g = loss
                .backward()
                .unwrap()
                .get(logp.as_tensor())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            (v, g)
        };
        let (lg, gg) = run(LossType::Grpo);
        let (ld, gd) = run(LossType::Dapo);
        assert_relative_eq!(lg, ld, epsilon = TOL);
        for (rg, rd) in gg.iter().zip(&gd) {
            for (a, b) in rg.iter().zip(rd) {
                assert_relative_eq!(a, b, epsilon = TOL);
            }
        }
    }

    #[test]
    fn grpo_loss_sequence_level_reduces_to_minus_advantage_mean_at_ratio_one() {
        // At ratio == 1 (logp == logp_old) the sequence-level surrogate is just
        // A_i, so the Grpo reduction gives -mean(A) — a closed-form pin that the
        // [G,1] broadcast through masked_mean_tensor is wired right.
        let logp = mat(&[&[-1.0, -2.0], &[-0.5, -0.25]]);
        let adv = Tensor::from_vec(vec![0.8f32, -0.3], (2, 1), &cpu()).unwrap();
        let mask = mat(&[&[1.0, 0.0], &[1.0, 1.0]]);
        let cfg = LossCfg {
            clip_eps_low: 0.2,
            clip_eps_high: 0.2,
            beta: 0.0,
            loss_type: LossType::Grpo,
            is_level: ImportanceSamplingLevel::Sequence,
            dapo_norm: None,
            tis_w: None,
        };
        let loss = grpo_loss(&logp, &logp.detach(), None, &adv, &mask, &cfg).unwrap();
        let got = scalar_f32(&loss).unwrap();
        assert_relative_eq!(got, -(0.8 - 0.3) / 2.0, epsilon = TOL);
    }

    #[test]
    fn scale_var_grads_scales_only_present_entries() {
        let dev = cpu();
        let x =
            Var::from_tensor(&Tensor::from_vec(vec![3.0f32, 4.0], (2,), &dev).unwrap()).unwrap();
        let dead = Var::from_tensor(&Tensor::from_vec(vec![1.0f32], (1,), &dev).unwrap()).unwrap();
        let vars = vec![x.clone(), dead.clone()];
        // loss = sum(x^2) -> grad 2x = [6, 8], norm 10.
        let mut store = x
            .as_tensor()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let norm = global_grad_norm(&vars, &store).unwrap();
        assert_relative_eq!(norm as f32, 10.0, epsilon = TOL);
        // Clip to max_norm 1.0: scale 0.1 -> [0.6, 0.8], norm 1.
        scale_var_grads(&vars, &mut store, 1.0 / norm).unwrap();
        let g = store.get(x.as_tensor()).unwrap().to_vec1::<f32>().unwrap();
        assert_relative_eq!(g[0], 0.6, epsilon = TOL);
        assert_relative_eq!(g[1], 0.8, epsilon = TOL);
        assert_relative_eq!(
            global_grad_norm(&vars, &store).unwrap() as f32,
            1.0,
            epsilon = TOL
        );
        // The absent var stays absent (no phantom entry materialized).
        assert!(store.get(dead.as_tensor()).is_none());
    }

    // ---- trainer wiring: config knobs reach the differentiated loss ---------

    /// A unique temp dir for in-module Trainer construction, removed on drop.
    struct WireTmp(std::path::PathBuf);
    impl WireTmp {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            p.push(format!("ferrl-wire-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for WireTmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A minimal Policy whose `token_logprobs` returns a fixed Var-backed
    /// tensor — `item_backward` only needs that method, so the wiring tests
    /// drive the REAL config→loss path with a fully crafted gradient surface.
    struct StubPolicy {
        logp: Var,
    }
    impl Policy for StubPolicy {
        fn generate(&mut self, _p: &[u32], _c: &GenConfig) -> CandleResult<Rollout> {
            unreachable!("wiring tests never roll out")
        }
        fn token_logprobs(&self, _r: &Rollout) -> CandleResult<Tensor> {
            Ok(self.logp.as_tensor().clone())
        }
        fn set_adapter_enabled(&mut self, _e: bool) {}
        fn adapter_enabled(&self) -> bool {
            true
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn restore_sampler_state(&mut self, _s: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    /// A policy backed by a full `[G, T]` trainable log-prob table, selecting
    /// rows by the first token id in each rollout row. This lets the
    /// microbatch test prove that row-sliced rollouts hit the same gradient
    /// coordinates as the full-group backward.
    struct RowAwarePolicy {
        logp: Var,
    }
    impl Policy for RowAwarePolicy {
        fn generate(&mut self, _p: &[u32], _c: &GenConfig) -> CandleResult<Rollout> {
            unreachable!("wiring tests never roll out")
        }
        fn token_logprobs(&self, r: &Rollout) -> CandleResult<Tensor> {
            let rows: Vec<u32> = r.token_ids.iter().map(|ids| ids[0]).collect();
            let idx = Tensor::from_vec(rows, r.len(), self.logp.as_tensor().device())?;
            self.logp.as_tensor().index_select(&idx, 0)
        }
        fn token_logprobs_detached(&self, r: &Rollout) -> CandleResult<Tensor> {
            Ok(self.token_logprobs(r)?.detach())
        }
        fn set_adapter_enabled(&mut self, _e: bool) {}
        fn adapter_enabled(&self) -> bool {
            true
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn restore_sampler_state(&mut self, _s: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    /// Run `item_backward` under `cfg` over a fixed crafted item (ratios
    /// straddling the clip bands, ragged mask) and return the flat gradient of
    /// the logp Var. This pins that each config knob actually reaches the
    /// differentiated loss — the seam a hardcoded default would sever.
    fn wire_grad(cfg: &TrainerConfig, window_tokens: f64) -> Vec<f32> {
        let logp = wire_logp();
        let policy = StubPolicy { logp: logp.clone() };
        wire_grad_via(&policy, &logp, cfg, window_tokens)
    }

    fn wire_logp() -> Var {
        Var::from_tensor(&mat(&[&[-1.0, -2.0, -0.4], &[-0.5, -0.25, -0.75]])).unwrap()
    }

    /// The fixed crafted item over `logp` (ratios straddling the clip bands,
    /// ragged mask), driven through `item_backward` with `policy`; returns the
    /// flat gradient of the logp Var out of the store `item_backward` returns.
    fn wire_grad_via<P: Policy>(
        policy: &P,
        logp: &Var,
        cfg: &TrainerConfig,
        window_tokens: f64,
    ) -> Vec<f32> {
        let grads = wire_store_via(policy, logp, cfg, window_tokens);
        grads
            .get(logp.as_tensor())
            .expect("logp var must be in the grad store")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn wire_store_via<P: Policy>(
        policy: &P,
        logp: &Var,
        cfg: &TrainerConfig,
        window_tokens: f64,
    ) -> GradStore {
        let dev = cpu();
        // Shifts 0.22 / -0.30 straddle both bands (1.246 / 0.74); ratio != 1
        // even at mu = 1 because logp_old is crafted, not snapshotted.
        let shift = mat(&[&[0.22, -0.30, 0.05], &[0.05, 0.22, -0.30]]);
        let logp_old = logp.as_tensor().sub(&shift).unwrap().detach();
        let item = LiveItem {
            rollout: Rollout {
                token_ids: vec![vec![0; 4]; 2],
                prompt_len: 1,
                completion_lens: vec![3, 2],
                rollout_logprobs: None,
            },
            advantages: Tensor::from_vec(vec![0.8f32, -0.7], (2, 1), &dev).unwrap(),
            logp_old,
            logp_ref: None,
            mask: mat(&[&[1.0, 1.0, 1.0], &[1.0, 1.0, 0.0]]),
            tis_w: None,
        };
        item_store_via(policy, &item, cfg, window_tokens)
    }

    fn item_store_via<P: Policy>(
        policy: &P,
        item: &LiveItem,
        cfg: &TrainerConfig,
        window_tokens: f64,
    ) -> GradStore {
        item_backward_via(policy, item, cfg, window_tokens).0
    }

    fn item_backward_via<P: Policy>(
        policy: &P,
        item: &LiveItem,
        cfg: &TrainerConfig,
        window_tokens: f64,
    ) -> (GradStore, f32, f32) {
        let tmp = WireTmp::new("grad");
        let run = RunDir::create(&tmp.0, "wire").unwrap();
        let trainer = Trainer::new(cfg.clone(), &run).unwrap();
        let vars = policy.trainable_vars();
        trainer
            .item_backward(policy, item, window_tokens, &vars, cfg.beta_at(0))
            .unwrap()
    }

    fn assert_grads_scaled(base: &[f32], scaled: &[f32], factor: f32, ctx: &str) {
        for (b, s) in base.iter().zip(scaled) {
            assert_relative_eq!(b * factor, s, epsilon = 1e-6, max_relative = 1e-5);
        }
        assert!(
            base.iter().any(|g| g.abs() > 1e-8),
            "{ctx}: baseline gradient is all-zero — the comparison is vacuous"
        );
    }

    fn assert_grads_differ(a: &[f32], b: &[f32], ctx: &str) {
        assert!(
            a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-7),
            "{ctx}: gradients identical — the knob never reached the loss"
        );
    }

    /// A [`StubPolicy`] whose `Policy::backward` override doubles the logp
    /// gradient after the plain backward — the probe that proves
    /// `item_backward` both *routes through* the policy seam and *returns the
    /// policy's store* (a hardcoded `loss.backward()` would sever exactly
    /// this, which is what activation checkpointing rides on).
    struct DoublingBackwardPolicy {
        inner: StubPolicy,
    }
    impl Policy for DoublingBackwardPolicy {
        fn generate(&mut self, p: &[u32], c: &GenConfig) -> CandleResult<Rollout> {
            self.inner.generate(p, c)
        }
        fn token_logprobs(&self, r: &Rollout) -> CandleResult<Tensor> {
            self.inner.token_logprobs(r)
        }
        fn set_adapter_enabled(&mut self, e: bool) {
            self.inner.set_adapter_enabled(e);
        }
        fn adapter_enabled(&self) -> bool {
            self.inner.adapter_enabled()
        }
        fn trainable_vars(&self) -> Vec<Var> {
            self.inner.trainable_vars()
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            self.inner.sampler_state()
        }
        fn restore_sampler_state(&mut self, s: &[u8]) -> CandleResult<()> {
            self.inner.restore_sampler_state(s)
        }
        fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
            let mut store = loss.backward()?;
            let v = &self.inner.logp;
            let g = store
                .remove(v)
                .expect("the logp grad must be on the loss tape");
            store.insert(v, (g * 2.0)?);
            Ok(store)
        }
    }

    struct NoisyBackwardPolicy {
        inner: StubPolicy,
        stray: Var,
    }
    impl Policy for NoisyBackwardPolicy {
        fn generate(&mut self, p: &[u32], c: &GenConfig) -> CandleResult<Rollout> {
            self.inner.generate(p, c)
        }
        fn token_logprobs(&self, r: &Rollout) -> CandleResult<Tensor> {
            self.inner.token_logprobs(r)
        }
        fn set_adapter_enabled(&mut self, e: bool) {
            self.inner.set_adapter_enabled(e);
        }
        fn adapter_enabled(&self) -> bool {
            self.inner.adapter_enabled()
        }
        fn trainable_vars(&self) -> Vec<Var> {
            self.inner.trainable_vars()
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            self.inner.sampler_state()
        }
        fn restore_sampler_state(&mut self, s: &[u8]) -> CandleResult<()> {
            self.inner.restore_sampler_state(s)
        }
        fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
            let mut store = loss.backward()?;
            store.insert(
                self.stray.as_tensor(),
                Tensor::ones_like(self.stray.as_tensor())?,
            );
            Ok(store)
        }
    }

    #[test]
    fn wiring_item_backward_runs_through_the_policy_backward_seam() {
        let cfg = TrainerConfig::default();
        let base = wire_grad(&cfg, 5.0);
        let logp = wire_logp();
        let policy = DoublingBackwardPolicy {
            inner: StubPolicy { logp: logp.clone() },
        };
        let doubled = wire_grad_via(&policy, &logp, &cfg, 5.0);
        assert_grads_scaled(&base, &doubled, 2.0, "policy backward seam");
    }

    #[test]
    fn wiring_item_backward_returns_a_compact_trainable_store() {
        let cfg = TrainerConfig::default();
        let logp = wire_logp();
        let stray = Var::zeros(1, DType::F32, &cpu()).unwrap();
        let policy = NoisyBackwardPolicy {
            inner: StubPolicy { logp: logp.clone() },
            stray: stray.clone(),
        };
        let grads = wire_store_via(&policy, &logp, &cfg, 5.0);
        assert!(
            grads.get(logp.as_tensor()).is_some(),
            "the trainable logp gradient must survive compaction"
        );
        assert!(
            grads.get(stray.as_tensor()).is_none(),
            "non-trainable raw-store entries must not survive item_backward"
        );
    }

    #[test]
    fn wiring_dapo_window_normalizer_scales_the_gradient() {
        // Doubling the window's token total must exactly halve the gradient —
        // and the trainer must pass the WINDOW total, not the item's mask sum.
        let cfg = TrainerConfig::default(); // loss_type: Dapo
        let g1 = wire_grad(&cfg, 5.0);
        let g2 = wire_grad(&cfg, 10.0);
        assert_grads_scaled(&g1, &g2, 0.5, "dapo window normalizer");
    }

    #[test]
    fn wiring_grpo_keeps_the_accum_scale_and_dapo_skips_it() {
        // Grpo at grad_accum_steps 2 halves the per-item gradient (the 1/accum
        // scale); Dapo must NOT add that scale on top of its window normalizer.
        let grpo1 = wire_grad(
            &TrainerConfig {
                loss_type: LossType::Grpo,
                ..TrainerConfig::default()
            },
            5.0,
        );
        let grpo2 = wire_grad(
            &TrainerConfig {
                loss_type: LossType::Grpo,
                grad_accum_steps: 2,
                ..TrainerConfig::default()
            },
            5.0,
        );
        assert_grads_scaled(&grpo1, &grpo2, 0.5, "grpo 1/accum scale");
        let dapo1 = wire_grad(&TrainerConfig::default(), 5.0);
        let dapo2 = wire_grad(
            &TrainerConfig {
                grad_accum_steps: 2,
                ..TrainerConfig::default()
            },
            5.0,
        );
        assert_grads_scaled(&dapo1, &dapo2, 1.0, "dapo must skip the accum scale");
    }

    fn assert_grad_mats_match(full: &[Vec<f32>], micro: &[Vec<f32>], ctx: &str) {
        for (full_row, micro_row) in full.iter().zip(micro) {
            for (f, m) in full_row.iter().zip(micro_row) {
                assert_relative_eq!(f, m, epsilon = 1e-6, max_relative = 1e-5);
            }
        }
        assert!(
            full.iter().flatten().any(|g| g.abs() > 1e-8),
            "{ctx}: baseline gradient is all-zero — the comparison is vacuous"
        );
    }

    fn assert_microbatch_matches_full(
        policy: &RowAwarePolicy,
        item: &LiveItem,
        logp: &Var,
        cfg: &TrainerConfig,
        ctx: &str,
    ) {
        let full_cfg = TrainerConfig {
            backward_microbatch_size: 0,
            ..cfg.clone()
        };
        let micro_cfg = TrainerConfig {
            backward_microbatch_size: 1,
            ..cfg.clone()
        };
        let (full_store, full_kl, full_clip) = item_backward_via(policy, item, &full_cfg, 5.0);
        let (micro_store, micro_kl, micro_clip) = item_backward_via(policy, item, &micro_cfg, 5.0);
        let full = full_store
            .get(logp.as_tensor())
            .expect("full-group trainable gradient must remain present")
            .to_vec2::<f32>()
            .unwrap();
        let micro = micro_store
            .get(logp.as_tensor())
            .expect("microbatched trainable gradient must remain present")
            .to_vec2::<f32>()
            .unwrap();
        assert_grad_mats_match(&full, &micro, ctx);
        assert_relative_eq!(full_kl, micro_kl, epsilon = 1e-6, max_relative = 1e-5);
        assert_relative_eq!(full_clip, micro_clip, epsilon = 1e-6, max_relative = 1e-5);
    }

    #[test]
    fn wiring_backward_microbatch_matches_full_group_gradients_and_diagnostics() {
        let dev = cpu();
        let logp = Var::from_tensor(&mat(&[&[-1.0, -2.0, -0.4], &[-0.5, -0.25, -0.75]])).unwrap();
        let policy = RowAwarePolicy { logp: logp.clone() };
        let shift = mat(&[&[0.22, -0.30, 0.05], &[0.05, 0.22, -0.30]]);
        let item = LiveItem {
            rollout: Rollout {
                token_ids: vec![vec![0, 7, 8, 9], vec![1, 10, 11, 12]],
                prompt_len: 1,
                completion_lens: vec![3, 0],
                rollout_logprobs: None,
            },
            advantages: Tensor::from_vec(vec![0.8f32, -0.7], (2, 1), &dev).unwrap(),
            logp_old: logp.as_tensor().sub(&shift).unwrap().detach(),
            logp_ref: Some(
                logp.as_tensor()
                    .add(&mat(&[&[0.08, -0.04, 0.03], &[-0.02, 0.06, -0.01]]))
                    .unwrap()
                    .detach(),
            ),
            mask: mat(&[&[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0]]),
            tis_w: Some(mat(&[&[1.0, 0.7, 1.3], &[0.0, 0.0, 0.0]])),
        };
        assert_microbatch_matches_full(
            &policy,
            &item,
            &logp,
            &TrainerConfig {
                loss_type: LossType::Grpo,
                ..TrainerConfig::default()
            },
            "grpo",
        );
        assert_microbatch_matches_full(
            &policy,
            &item,
            &logp,
            &TrainerConfig {
                loss_type: LossType::DrGrpo,
                beta: 0.2,
                ..TrainerConfig::default()
            },
            "dr_grpo beta",
        );
        assert_microbatch_matches_full(
            &policy,
            &item,
            &logp,
            &TrainerConfig {
                loss_type: LossType::Dapo,
                beta: 0.2,
                ..TrainerConfig::default()
            },
            "dapo beta",
        );
        assert_microbatch_matches_full(
            &policy,
            &item,
            &logp,
            &TrainerConfig {
                loss_type: LossType::Grpo,
                importance_sampling_level: ImportanceSamplingLevel::Sequence,
                beta: 0.2,
                ..TrainerConfig::default()
            },
            "sequence-level beta tis",
        );
    }

    #[test]
    fn wiring_clip_eps_high_reaches_the_loss() {
        // The crafted ratios include 1.246 cells with positive advantage:
        // clipped at symmetric 0.2, unclipped at clip-higher 0.28 — so the
        // configured upper band must change the gradient.
        let sym = wire_grad(&TrainerConfig::default(), 5.0);
        let asym = wire_grad(
            &TrainerConfig {
                clip_eps_high: Some(0.28),
                ..TrainerConfig::default()
            },
            5.0,
        );
        assert_grads_differ(&sym, &asym, "clip_eps_high");
    }

    #[test]
    fn wiring_importance_sampling_level_reaches_the_loss() {
        let token = wire_grad(&TrainerConfig::default(), 5.0);
        let seq = wire_grad(
            &TrainerConfig {
                importance_sampling_level: ImportanceSamplingLevel::Sequence,
                ..TrainerConfig::default()
            },
            5.0,
        );
        assert_grads_differ(&token, &seq, "importance_sampling_level");
    }

    // ---- masking / overflow corners ----------------------------------------

    #[test]
    fn all_masked_live_item_yields_present_zero_grads_not_canary_abort() {
        // A live item whose mask is entirely zero (every completion truncation-
        // masked) must produce PRESENT all-zero gradients — candle's where_cond
        // backward visits the unselected branch — so the trainer takes the
        // documented no-signal skip instead of a canary abort. Pinned here
        // because the guarantee rests on candle internals (a candle bump that
        // prunes zero subgraphs would turn ladder runs into 2 a.m. aborts).
        for level in [
            ImportanceSamplingLevel::Token,
            ImportanceSamplingLevel::Sequence,
        ] {
            let logp = Var::from_tensor(&mat(&[&[-1.0, -2.0], &[-0.5, -0.25]])).unwrap();
            let adv = Tensor::from_vec(vec![0.8f32, -0.7], (2, 1), &cpu()).unwrap();
            let mask = mat(&[&[0.0, 0.0], &[0.0, 0.0]]);
            let cfg = LossCfg {
                clip_eps_low: 0.2,
                clip_eps_high: 0.2,
                beta: 0.0,
                loss_type: LossType::Dapo,
                is_level: level,
                dapo_norm: Some(4.0),
                tis_w: None,
            };
            let old = logp.as_tensor().affine(1.0, 0.1).unwrap().detach();
            let loss = grpo_loss(logp.as_tensor(), &old, None, &adv, &mask, &cfg).unwrap();
            let v = scalar_f32(&loss).unwrap();
            assert_eq!(v, 0.0, "{level:?}: all-masked loss must be exactly 0");
            let grads = loss.backward().unwrap();
            let g = grads
                .get(logp.as_tensor())
                .unwrap_or_else(|| panic!("{level:?}: grad entry ABSENT — canary would abort"))
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                g.iter().all(|x| *x == 0.0),
                "{level:?}: all-masked grads must be exactly zero, got {g:?}"
            );
        }
    }

    #[test]
    fn overflowing_log_ratio_keeps_gradients_finite() {
        // A kept position whose log-ratio exceeds f32 exp's overflow point
        // (~88.7) used to poison every gradient with NaN through exp's
        // backward (grad · exp_output = 0 · inf) even though the loss value
        // stayed finite. The RATIO_LOG_CAP clamp must keep both the loss and
        // the gradient finite, at both levels.
        for level in [
            ImportanceSamplingLevel::Token,
            ImportanceSamplingLevel::Sequence,
        ] {
            let logp = Var::from_tensor(&mat(&[&[-0.5, 99.5]])).unwrap();
            let old = mat(&[&[-0.55, -0.5]]); // log-ratio [0.05, 100.0]
            let adv = Tensor::from_vec(vec![0.7f32], (1, 1), &cpu()).unwrap();
            let mask = mat(&[&[1.0, 1.0]]);
            let cfg = LossCfg {
                clip_eps_low: 0.2,
                clip_eps_high: 0.2,
                beta: 0.0,
                loss_type: LossType::Dapo,
                is_level: level,
                dapo_norm: None,
                tis_w: None,
            };
            let loss = grpo_loss(logp.as_tensor(), &old, None, &adv, &mask, &cfg).unwrap();
            let v = scalar_f32(&loss).unwrap();
            assert!(v.is_finite(), "{level:?}: loss must stay finite, got {v}");
            let grads = loss.backward().unwrap();
            let g = grads
                .get(logp.as_tensor())
                .expect("grad present")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                g.iter().all(|x| x.is_finite()),
                "{level:?}: overflowed ratio leaked NaN into the gradient: {g:?}"
            );
        }
    }

    #[test]
    fn clip_fraction_sequence_shape_counts_sequences_not_tokens() {
        // TRL's masked_batch_mean special-cases the (B, 1) sequence-level shape
        // to a plain mean over sequences: one clipped sequence of length 3 out
        // of [3, 1] valid tokens is 1/2, not 3/4.
        let ratio = Tensor::from_vec(vec![1.5f32, 1.0], (2, 1), &cpu()).unwrap();
        let adv = Tensor::from_vec(vec![0.5f32, 0.5], (2, 1), &cpu()).unwrap();
        let mask = mat(&[&[1.0, 1.0, 1.0], &[1.0, 0.0, 0.0]]);
        let frac = clip_fraction(&ratio, &adv, 0.2, 0.2, &mask).unwrap();
        assert_relative_eq!(frac, 0.5, epsilon = TOL);
    }

    // ---- REAL-TRL golden fixture (closes GOLDEN-CIRCULAR) -------------------

    #[derive(serde::Deserialize)]
    struct TrlBatch {
        logp: Vec<Vec<f64>>,
        logp_old: Vec<Vec<f64>>,
        logp_ref: Vec<Vec<f64>>,
        advantages: Vec<f64>,
        mask: Vec<Vec<f64>>,
    }

    #[derive(serde::Deserialize)]
    struct TrlCase {
        loss_type: String,
        beta: f64,
        eps_low: f64,
        eps_high: f64,
        importance_sampling_level: String,
        dapo_norm: f64,
        loss: f64,
    }

    #[derive(serde::Deserialize)]
    struct TrlGolden {
        trl_version: String,
        torch_version: String,
        transformers_version: String,
        batch: TrlBatch,
        cases: Vec<TrlCase>,
    }

    fn rows_to_tensor(rows: &[Vec<f64>]) -> Tensor {
        let r = rows.len();
        let c = rows[0].len();
        let data: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
        Tensor::from_vec(data, (r, c), &cpu()).unwrap()
    }

    /// Every case in the fixture was produced by TRL 1.5.1's OWN
    /// `GRPOTrainer._compute_loss` (not a transcription of its formulas — see
    /// `scripts/oracle/gen_grpo_golden_trl.py`), so this is the
    /// independent-implementation gate the `NumPy` golden cannot be: a shared
    /// misreading of the GRPO spec between ferrl and its same-author oracle
    /// fails here against the industry reference. 32 cases sweep
    /// `{grpo, dr_grpo, dapo} × beta {0, 0.04} × {token, sequence} ×
    /// eps_high {0.2, 0.28}` plus the explicit DAPO window normalizer, over a
    /// ragged mask (lengths 3/2/1/0) with ratios straddling both clip bands.
    #[test]
    fn matches_trl_golden_fixture() {
        let raw = include_str!("../tests/fixtures/grpo_golden_trl.json");
        let g: TrlGolden = serde_json::from_str(raw).expect("TRL golden parses");
        // The version pins make regeneration a deliberate, reviewed act: the
        // values are a numeric contract with exactly this TRL/torch pair.
        assert_eq!(
            g.trl_version, "1.5.1",
            "TRL pin drifted — regenerate deliberately"
        );
        assert!(g.torch_version.starts_with("2.12.0"), "{}", g.torch_version);
        assert_eq!(g.transformers_version, "5.11.0");

        let logp = rows_to_tensor(&g.batch.logp);
        let logp_old = rows_to_tensor(&g.batch.logp_old);
        let logp_ref = rows_to_tensor(&g.batch.logp_ref);
        let mask = rows_to_tensor(&g.batch.mask);
        let adv: Vec<f32> = g.batch.advantages.iter().map(|&a| a as f32).collect();
        let n = adv.len();
        let advantages = Tensor::from_vec(adv, (n, 1), &cpu()).unwrap();

        assert_eq!(g.cases.len(), 32, "expected the full case sweep");
        for c in &g.cases {
            check_trl_case(c, &logp, &logp_old, &logp_ref, &advantages, &mask);
        }
    }

    /// Replay one TRL golden case through the production `grpo_loss` and
    /// assert agreement. Split out of `matches_trl_golden_fixture` to stay
    /// under the cognitive-complexity bound.
    fn check_trl_case(
        c: &TrlCase,
        logp: &Tensor,
        logp_old: &Tensor,
        logp_ref: &Tensor,
        advantages: &Tensor,
        mask: &Tensor,
    ) {
        let loss_type = match c.loss_type.as_str() {
            "grpo" => LossType::Grpo,
            "dr_grpo" => LossType::DrGrpo,
            "dapo" => LossType::Dapo,
            other => panic!("unknown loss_type {other}"),
        };
        let is_level = match c.importance_sampling_level.as_str() {
            "token" => ImportanceSamplingLevel::Token,
            "sequence" => ImportanceSamplingLevel::Sequence,
            other => panic!("unknown level {other}"),
        };
        let cfg = LossCfg {
            clip_eps_low: c.eps_low,
            clip_eps_high: c.eps_high,
            beta: c.beta,
            loss_type,
            is_level,
            dapo_norm: Some(c.dapo_norm),
            tis_w: None,
        };
        let lref = (c.beta != 0.0).then_some(logp_ref);
        let got =
            scalar_f32(&grpo_loss(logp, logp_old, lref, advantages, mask, &cfg).unwrap()).unwrap();
        let want = c.loss as f32;
        assert_relative_eq!(got, want, epsilon = 2e-6, max_relative = 2e-5);
    }

    #[test]
    fn config_roundtrips_through_json() {
        let cfg = TrainerConfig::default();
        let j = serde_json::to_string(&cfg).unwrap();
        let back: TrainerConfig = serde_json::from_str(&j).unwrap();
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&cfg).unwrap()
        );
    }

    #[test]
    fn r1_config_fields_roundtrip_through_json() {
        let cfg = TrainerConfig {
            clip_eps_high: Some(0.28),
            importance_sampling_level: ImportanceSamplingLevel::Sequence,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            warmup_steps: 20,
            max_grad_norm: Some(0.5),
            truncation_masking: false,
            ..TrainerConfig::default()
        };
        let back: TrainerConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        // Whole-config JSON equality covers every R1 field in one shot.
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&cfg).unwrap()
        );
    }

    #[test]
    fn eos_token_id_round_trips_through_json() {
        // Default (None) and an explicit Some both survive a JSON round-trip; serde
        // carries Some verbatim even though `validate` rejects it until the loss path
        // honors the EOS padding.
        let dflt = TrainerConfig::default();
        let back: TrainerConfig =
            serde_json::from_str(&serde_json::to_string(&dflt).unwrap()).unwrap();
        assert_eq!(back.eos_token_id, None);
        let eos_cfg = TrainerConfig {
            eos_token_id: Some(151_643),
            ..TrainerConfig::default()
        };
        let back2: TrainerConfig =
            serde_json::from_str(&serde_json::to_string(&eos_cfg).unwrap()).unwrap();
        assert_eq!(back2.eos_token_id, Some(151_643));
    }

    /// A pre-R1 (and pre-grad-accum) `config.json`, shared by the back-compat
    /// deserialization tests below.
    const OLD_CONFIG_JSON: &str = r#"{"steps":10,"group_size":8,"max_new_tokens":16,
        "temperature":1.0,"mu":1,"beta":0.0,"clip_eps":0.2,"lr":0.001,"weight_decay":0.0,
        "loss_type":"grpo","scale_rewards":"group"}"#;

    #[test]
    fn grad_accum_steps_defaults_to_one_for_old_configs() {
        // A config.json written before grad_accum_steps existed must deserialize to 1
        // (no accumulation), not fail — the serde default keeps old runs loadable.
        let cfg: TrainerConfig = serde_json::from_str(OLD_CONFIG_JSON).unwrap();
        assert_eq!(
            (
                cfg.grad_accum_steps,
                cfg.backward_microbatch_size,
                cfg.candidate_log_top_k,
                cfg.reward_group_scope,
                cfg.beta_schedule.as_ref(),
                cfg.lr_schedule.as_ref(),
                cfg.eos_token_id,
            ),
            (1, 0, 0, RewardGroupScope::Local, None, None, None)
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn r1_config_fields_default_for_old_configs() {
        // R1 fields, absent from a pre-R1 config.json, fill from their serde
        // defaults — note loss_type stays the EXPLICIT legacy "grpo" the file
        // recorded, while clipping (a safety net) and truncation masking
        // default on.
        let cfg: TrainerConfig = serde_json::from_str(OLD_CONFIG_JSON).unwrap();
        assert_eq!(cfg.loss_type, LossType::Grpo);
        assert_eq!(
            cfg.importance_sampling_level,
            ImportanceSamplingLevel::Token
        );
        assert_eq!(
            (cfg.adam_beta1, cfg.adam_beta2, cfg.warmup_steps),
            (0.9, 0.999, 0)
        );
        assert_eq!(
            (cfg.clip_eps_high, cfg.max_grad_norm, cfg.truncation_masking),
            (None, Some(1.0), true)
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn distributed_reward_group_scope_allows_lockstep_multi_prompt_accumulation() {
        let cfg = TrainerConfig {
            reward_group_scope: RewardGroupScope::DistributedSamePrompt,
            grad_accum_steps: 2,
            ..TrainerConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn distributed_reward_group_stats_match_local_for_one_rank() {
        let rewards = [1.0, 2.0, 3.0];
        let local = group_advantages(&rewards, ScaleRewards::Group);
        let via_stats = advantages_from_stats(
            &rewards,
            RewardStatsAcc::from_rewards(&rewards),
            ScaleRewards::Group,
        );
        assert_eq!(via_stats.len(), local.len());
        for (a, b) in via_stats.iter().zip(local.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn distributed_reward_group_same_prompt_produces_cross_rank_advantages() {
        std::thread::scope(|scope| {
            let comms = crate::comm::LocalComm::world(2);
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    scope.spawn(move || {
                        let tmp = WireTmp::new(&format!("dist-adv-{rank}"));
                        let run = RunDir::create(&tmp.0, format!("rank-{rank}")).unwrap();
                        let cfg = TrainerConfig {
                            reward_group_scope: RewardGroupScope::DistributedSamePrompt,
                            scale_rewards: ScaleRewards::None,
                            ..TrainerConfig::default()
                        };
                        let trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                        let reward = if rank == 0 { 1.0 } else { 3.0 };
                        trainer.reward_group_advantages(&[reward]).unwrap()
                    })
                })
                .collect();
            let mut got = Vec::new();
            for h in handles {
                got.push(h.join().unwrap());
            }
            assert_eq!(got[0], vec![-1.0]);
            assert_eq!(got[1], vec![1.0]);
        });
    }

    #[test]
    fn distributed_reward_group_same_prompt_identical_rewards_stays_degenerate() {
        std::thread::scope(|scope| {
            let comms = crate::comm::LocalComm::world(2);
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    scope.spawn(move || {
                        let tmp = WireTmp::new(&format!("dist-degen-{rank}"));
                        let run = RunDir::create(&tmp.0, format!("rank-{rank}")).unwrap();
                        let cfg = TrainerConfig {
                            reward_group_scope: RewardGroupScope::DistributedSamePrompt,
                            scale_rewards: ScaleRewards::Group,
                            ..TrainerConfig::default()
                        };
                        let trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                        trainer.reward_group_advantages(&[2.0]).unwrap()
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), vec![0.0]);
            }
        });
    }

    #[test]
    fn local_group_one_remains_degenerate_without_distributed_scope() {
        let tmp = WireTmp::new("local-g1");
        let run = RunDir::create(&tmp.0, "local").unwrap();
        let cfg = TrainerConfig {
            reward_group_scope: RewardGroupScope::Local,
            ..TrainerConfig::default()
        };
        let trainer = Trainer::new(cfg, &run).unwrap();
        assert_eq!(trainer.reward_group_advantages(&[3.0]).unwrap(), vec![0.0]);
    }

    struct PromptSelectionCodec;
    impl TokenizerLike for PromptSelectionCodec {
        fn encode(&self, _text: &str) -> Vec<u32> {
            vec![1]
        }
        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    struct PromptSelectionPolicy {
        logp: Var,
    }
    impl Policy for PromptSelectionPolicy {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            self.generate_at(prompt, cfg, 0)
        }

        fn generate_at(
            &mut self,
            _prompt: &[u32],
            _cfg: &GenConfig,
            global_row_base: u64,
        ) -> CandleResult<Rollout> {
            Ok(Rollout {
                token_ids: vec![vec![1, 10 + global_row_base as u32]],
                prompt_len: 1,
                completion_lens: vec![1],
                rollout_logprobs: None,
            })
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            Ok(self.logp.as_tensor().clone())
        }

        fn set_adapter_enabled(&mut self, _enabled: bool) {}
        fn adapter_enabled(&self) -> bool {
            true
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    struct PromptSelectionReward {
        rank: usize,
        seen: std::sync::Arc<std::sync::Mutex<Vec<(usize, String, String)>>>,
    }
    impl RewardFn for PromptSelectionReward {
        type Target = ();

        fn reward(&self, sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
            self.seen.lock().unwrap().push((
                self.rank,
                sample.prompt.clone(),
                completion.to_owned(),
            ));
            Ok(1.0)
        }
    }

    #[test]
    fn distributed_same_prompt_train_selects_same_prompt_across_ranks() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let samples: Vec<_> = ["p0", "p1", "p2", "p3"]
            .into_iter()
            .map(|p| Sample::new(p, ()))
            .collect();
        std::thread::scope(|scope| {
            let comms = crate::comm::LocalComm::world(2);
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let seen = std::sync::Arc::clone(&seen);
                    let samples = samples.clone();
                    scope.spawn(move || {
                        let tmp = WireTmp::new(&format!("dist-select-{rank}"));
                        let run = RunDir::create(&tmp.0, format!("rank-{rank}")).unwrap();
                        let cfg = TrainerConfig {
                            steps: 2,
                            group_size: 1,
                            grad_accum_steps: 2,
                            max_new_tokens: 1,
                            lr: 0.0,
                            reward_group_scope: RewardGroupScope::DistributedSamePrompt,
                            ..TrainerConfig::default()
                        };
                        let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                        let logp = Var::from_tensor(&mat(&[&[-1.0]])).unwrap();
                        let mut policy = PromptSelectionPolicy { logp };
                        trainer
                            .train(
                                &mut policy,
                                &PromptSelectionReward { rank, seen },
                                &PromptSelectionCodec,
                                &samples,
                            )
                            .unwrap();
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });

        let mut got = seen.lock().unwrap().clone();
        got.sort();
        assert_eq!(
            got,
            vec![
                (0, "p0".to_string(), "10".to_string()),
                (0, "p1".to_string(), "11".to_string()),
                (0, "p2".to_string(), "14".to_string()),
                (0, "p3".to_string(), "15".to_string()),
                (1, "p0".to_string(), "12".to_string()),
                (1, "p1".to_string(), "13".to_string()),
                (1, "p2".to_string(), "16".to_string()),
                (1, "p3".to_string(), "17".to_string()),
            ]
        );
    }

    #[test]
    fn reward_stats_mean_and_std() {
        let (mean, std) = reward_stats(&[1.0, 1.0, 1.0]);
        assert_relative_eq!(mean, 1.0, epsilon = TOL);
        assert_relative_eq!(std, 0.0, epsilon = TOL);

        let (mean, std) = reward_stats(&[1.0, 0.0]);
        assert_relative_eq!(mean, 0.5, epsilon = TOL);
        assert_relative_eq!(std, 0.5, epsilon = TOL);
    }

    #[test]
    fn reward_stats_uses_every_validated_reward() {
        let (mean, std) = reward_stats(&[1.0, 2.0, 3.0]);
        assert_relative_eq!(mean, 2.0, epsilon = TOL);
        assert_relative_eq!(std, (2.0_f32 / 3.0).sqrt(), epsilon = TOL);
        let (mean, std) = reward_stats(&[f32::MAX, f32::MAX]);
        assert_eq!(mean, f32::MAX);
        assert_eq!(std, 0.0);
        let (mean, std) = reward_stats(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    struct CandidateCodec;
    impl TokenizerLike for CandidateCodec {
        fn encode(&self, _text: &str) -> Vec<u32> {
            vec![1]
        }
        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    struct CandidatePolicy {
        logp: Var,
    }
    impl Policy for CandidatePolicy {
        fn generate(&mut self, _prompt: &[u32], _cfg: &GenConfig) -> CandleResult<Rollout> {
            Ok(Rollout {
                token_ids: vec![vec![1, 7, 7], vec![1, 9, 9], vec![1, 2, 2]],
                prompt_len: 1,
                completion_lens: vec![2, 2, 2],
                rollout_logprobs: None,
            })
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            Ok(self.logp.as_tensor().clone())
        }

        fn set_adapter_enabled(&mut self, _enabled: bool) {}
        fn adapter_enabled(&self) -> bool {
            true
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    struct CandidateReward;
    impl RewardFn for CandidateReward {
        type Target = ();

        fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
            Ok(match completion {
                "7,7" => 0.0,
                "9,9" => 2.0,
                _ => 1.0,
            })
        }

        fn reward_group_detailed(
            &self,
            sample: &Sample<()>,
            completions: &[String],
        ) -> Result<Vec<crate::RewardOutcome>, RewardError> {
            completions
                .iter()
                .map(|completion| {
                    let reward = self.reward(sample, completion)?;
                    Ok(crate::RewardOutcome {
                        reward,
                        diagnostic: Some(format!("candidate:{completion}")),
                        metadata: Some(serde_json::json!({ "completion": completion })),
                    })
                })
                .collect()
        }
    }

    struct RankedNonFiniteReward {
        invalid: bool,
    }

    impl RewardFn for RankedNonFiniteReward {
        type Target = ();

        fn reward(&self, _sample: &Sample<()>, _completion: &str) -> Result<f32, RewardError> {
            unreachable!("the ranked non-finite reward test uses the detailed group seam")
        }

        fn reward_group_detailed(
            &self,
            _sample: &Sample<()>,
            completions: &[String],
        ) -> Result<Vec<RewardOutcome>, RewardError> {
            Ok(completions
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    RewardOutcome::reward(if self.invalid && index == 1 {
                        f32::NAN
                    } else {
                        index as f32
                    })
                })
                .collect())
        }
    }

    struct StatefulCandidatePolicy {
        inner: CandidatePolicy,
        sampler: u64,
    }

    impl Policy for StatefulCandidatePolicy {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            let rollout = self.inner.generate(prompt, cfg)?;
            self.sampler += 1;
            Ok(rollout)
        }

        fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
            self.inner.token_logprobs(rollout)
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.inner.set_adapter_enabled(enabled);
        }

        fn adapter_enabled(&self) -> bool {
            self.inner.adapter_enabled()
        }

        fn trainable_vars(&self) -> Vec<Var> {
            self.inner.trainable_vars()
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(self.sampler.to_le_bytes().to_vec())
        }

        fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
            let bytes: [u8; 8] = state.try_into().map_err(|error| {
                candle_core::Error::msg(format!("invalid test sampler state: {error}"))
            })?;
            self.sampler = u64::from_le_bytes(bytes);
            Ok(())
        }
    }

    impl TensorParallelPolicy for StatefulCandidatePolicy {
        fn generate_at_tensor_parallel_instrumented(
            &mut self,
            prompt: &[u32],
            cfg: &GenConfig,
            _global_row_base: u64,
            _comm: &dyn Comm,
            _telemetry: Option<&mut dyn ModelTelemetryRecorder>,
        ) -> CandleResult<Rollout> {
            self.generate(prompt, cfg)
        }

        fn token_logprobs_tensor_parallel(
            &self,
            rollout: &Rollout,
            _comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            self.token_logprobs(rollout)
        }

        fn token_logprobs_tensor_parallel_detached(
            &self,
            rollout: &Rollout,
            _comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            Ok(self.token_logprobs(rollout)?.detach())
        }

        fn backward_tensor_parallel(
            &self,
            loss: &Tensor,
            _comm: &dyn Comm,
        ) -> CandleResult<GradStore> {
            loss.backward()
        }

        fn supports_sharded_tensor_parallel_backward(&self) -> bool {
            true
        }
    }

    #[derive(Debug)]
    struct ArmedCollectiveFailureState {
        remaining_successes: std::sync::atomic::AtomicUsize,
        failed: std::sync::atomic::AtomicBool,
        calls_after_failure: std::sync::atomic::AtomicUsize,
    }

    impl ArmedCollectiveFailureState {
        fn new(remaining_successes: usize) -> Self {
            Self {
                remaining_successes: std::sync::atomic::AtomicUsize::new(remaining_successes),
                failed: std::sync::atomic::AtomicBool::new(false),
                calls_after_failure: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn enter(&self, armed: bool) -> Result<(), crate::comm::CommError> {
            use std::sync::atomic::Ordering;

            if self.failed.load(Ordering::SeqCst) {
                self.calls_after_failure.fetch_add(1, Ordering::SeqCst);
                return Err(crate::comm::CommError::Poisoned(
                    "collective issued after injected terminal failure".into(),
                ));
            }
            if !armed {
                return Ok(());
            }
            let remaining = self.remaining_successes.load(Ordering::SeqCst);
            if remaining > 0 {
                self.remaining_successes.fetch_sub(1, Ordering::SeqCst);
                return Ok(());
            }
            self.failed.store(true, Ordering::SeqCst);
            Err(crate::comm::CommError::Mismatch(
                "injected terminal collective-chain failure".into(),
            ))
        }
    }

    #[derive(Debug)]
    struct FailAfterArmComm<C> {
        inner: C,
        armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
        state: std::sync::Arc<ArmedCollectiveFailureState>,
    }

    impl<C: crate::comm::Comm> crate::comm::Comm for FailAfterArmComm<C> {
        fn rank(&self) -> usize {
            self.inner.rank()
        }

        fn world_size(&self) -> usize {
            self.inner.world_size()
        }

        fn validate_all_reduce_sum(
            &self,
            tensors: &[Tensor],
        ) -> Result<(), crate::comm::CommError> {
            self.inner.validate_all_reduce_sum(tensors)
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), crate::comm::CommError> {
            self.state
                .enter(self.armed.load(std::sync::atomic::Ordering::SeqCst))?;
            self.inner.all_reduce_sum(tensors)
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, crate::comm::CommError> {
            self.state
                .enter(self.armed.load(std::sync::atomic::Ordering::SeqCst))?;
            self.inner.all_reduce_scalar_sum(value)
        }
    }

    #[derive(Debug)]
    struct MetricsStatusFailureState {
        failed: std::sync::atomic::AtomicBool,
        calls_after_failure: std::sync::atomic::AtomicUsize,
    }

    impl MetricsStatusFailureState {
        fn new() -> Self {
            Self {
                failed: std::sync::atomic::AtomicBool::new(false),
                calls_after_failure: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    #[derive(Debug)]
    struct FailAtMetricsStatusComm<C> {
        inner: C,
        metrics_path: PathBuf,
        state: std::sync::Arc<MetricsStatusFailureState>,
    }

    impl<C: crate::comm::Comm> FailAtMetricsStatusComm<C> {
        fn enter(&self) -> Result<(), crate::comm::CommError> {
            use std::sync::atomic::Ordering;

            if self.state.failed.load(Ordering::SeqCst) {
                self.state
                    .calls_after_failure
                    .fetch_add(1, Ordering::SeqCst);
                return Err(crate::comm::CommError::Poisoned(
                    "collective issued after injected metrics-status failure".into(),
                ));
            }
            if std::fs::metadata(&self.metrics_path).is_ok_and(|metadata| metadata.len() > 0) {
                self.state.failed.store(true, Ordering::SeqCst);
                return Err(crate::comm::CommError::Mismatch(
                    "injected metrics append-status failure".into(),
                ));
            }
            Ok(())
        }
    }

    impl<C: crate::comm::Comm> crate::comm::Comm for FailAtMetricsStatusComm<C> {
        fn rank(&self) -> usize {
            self.inner.rank()
        }

        fn world_size(&self) -> usize {
            self.inner.world_size()
        }

        fn validate_all_reduce_sum(
            &self,
            tensors: &[Tensor],
        ) -> Result<(), crate::comm::CommError> {
            self.inner.validate_all_reduce_sum(tensors)
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), crate::comm::CommError> {
            self.enter()?;
            self.inner.all_reduce_sum(tensors)
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, crate::comm::CommError> {
            self.enter()?;
            self.inner.all_reduce_scalar_sum(value)
        }
    }

    #[derive(Clone, Copy)]
    enum CandidatePolicyArmPoint {
        DetachedScoring,
        SamplerState(usize),
    }

    struct ArmedCandidatePolicy {
        inner: StatefulCandidatePolicy,
        armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
        arm_point: CandidatePolicyArmPoint,
        sampler_state_calls: std::cell::Cell<usize>,
    }

    impl Policy for ArmedCandidatePolicy {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            self.inner.generate(prompt, cfg)
        }

        fn generate_at(
            &mut self,
            prompt: &[u32],
            cfg: &GenConfig,
            global_row_base: u64,
        ) -> CandleResult<Rollout> {
            self.inner.generate_at(prompt, cfg, global_row_base)
        }

        fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
            self.inner.token_logprobs(rollout)
        }

        fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
            let logprobs = self.inner.token_logprobs_detached(rollout)?;
            if matches!(self.arm_point, CandidatePolicyArmPoint::DetachedScoring) {
                self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(logprobs)
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.inner.set_adapter_enabled(enabled);
        }

        fn adapter_enabled(&self) -> bool {
            self.inner.adapter_enabled()
        }

        fn trainable_vars(&self) -> Vec<Var> {
            self.inner.trainable_vars()
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            let call = self.sampler_state_calls.get() + 1;
            self.sampler_state_calls.set(call);
            if matches!(self.arm_point, CandidatePolicyArmPoint::SamplerState(expected) if call == expected)
            {
                self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            self.inner.sampler_state()
        }

        fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
            self.inner.restore_sampler_state(state)
        }
    }

    struct ArmedCandidateReward {
        armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl RewardFn for ArmedCandidateReward {
        type Target = ();

        fn reward(&self, sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
            CandidateReward.reward(sample, completion)
        }

        fn reward_group_detailed(
            &self,
            sample: &Sample<()>,
            completions: &[String],
        ) -> Result<Vec<crate::RewardOutcome>, RewardError> {
            let outcomes = CandidateReward.reward_group_detailed(sample, completions)?;
            self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(outcomes)
        }
    }

    fn stateful_candidate_policy() -> StatefulCandidatePolicy {
        let logp =
            Var::from_tensor(&Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap()).unwrap();
        StatefulCandidatePolicy {
            inner: CandidatePolicy { logp },
            sampler: 0,
        }
    }

    fn candidate_ledger_config() -> TrainerConfig {
        TrainerConfig {
            steps: 1,
            group_size: 3,
            grad_accum_steps: 1,
            max_new_tokens: 2,
            beta: 0.0,
            mu: 1,
            checkpoint_every: None,
            ..TrainerConfig::default()
        }
    }

    #[test]
    fn collector_rejects_nonfinite_reward_before_candidate_or_ledger_publication() {
        let tmp = WireTmp::new("ledger-nonfinite-reward");
        let run = RunDir::create(&tmp.0, "run").unwrap();
        let ledger_root = tmp.0.join("ledger");
        let mut config = candidate_ledger_config();
        config.candidate_log_top_k = 1;
        let mut trainer = Trainer::new(config, &run).unwrap();
        let mut policy = stateful_candidate_policy();

        let error = trainer
            .collect_rollout_ledger_step(
                0,
                &mut policy,
                &RankedNonFiniteReward { invalid: true },
                &CandidateCodec,
                &[Sample::new("prompt", ())],
                &ledger_root,
                &"2".repeat(64),
                None,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            TrainerError::Reward(error) if error.to_string().contains("non-finite")
        ));
        assert_eq!(
            policy.sampler, 0,
            "failed collection must roll back sampler"
        );
        assert!(std::fs::read_to_string(run.candidates_path())
            .unwrap_or_default()
            .is_empty());
        assert!(!ledger_root.join("step-00000000000000000000").exists());
    }

    #[test]
    fn collector_keeps_poststate_when_post_manifest_failure_reconciles_as_committed() {
        let tmp = WireTmp::new("ledger-post-manifest-collector");
        let run = RunDir::create(&tmp.0, "run").unwrap();
        let config = TrainerConfig {
            steps: 1,
            group_size: 3,
            grad_accum_steps: 1,
            max_new_tokens: 2,
            beta: 0.0,
            checkpoint_every: None,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(config, &run).unwrap();
        let logp =
            Var::from_tensor(&Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap()).unwrap();
        let mut policy = StatefulCandidatePolicy {
            inner: CandidatePolicy { logp },
            sampler: 0,
        };
        crate::rollout_ledger::inject_post_manifest_failure_once();
        let published = trainer
            .collect_rollout_ledger_step(
                0,
                &mut policy,
                &CandidateReward,
                &CandidateCodec,
                &[Sample::new("p", ())],
                tmp.0.join("ledger"),
                &"a".repeat(64),
                None,
            )
            .unwrap();
        assert!(published.join("manifest.json").is_file());
        assert_eq!(policy.sampler, 1, "collector rewound a visible L_0");
    }

    #[test]
    fn collector_keeps_poststate_when_post_manifest_durability_is_ambiguous() {
        let tmp = WireTmp::new("ledger-ambiguous-post-manifest-collector");
        let run = RunDir::create(&tmp.0, "run").unwrap();
        let config = TrainerConfig {
            steps: 1,
            group_size: 3,
            grad_accum_steps: 1,
            max_new_tokens: 2,
            beta: 0.0,
            checkpoint_every: None,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(config, &run).unwrap();
        let logp =
            Var::from_tensor(&Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap()).unwrap();
        let mut policy = StatefulCandidatePolicy {
            inner: CandidatePolicy { logp },
            sampler: 0,
        };
        crate::rollout_ledger::inject_persistent_post_manifest_sync_failure_once();
        assert!(matches!(
            trainer.collect_rollout_ledger_step(
                0,
                &mut policy,
                &CandidateReward,
                &CandidateCodec,
                &[Sample::new("p", ())],
                tmp.0.join("ledger"),
                &"a".repeat(64),
                None,
            ),
            Err(TrainerError::RolloutLedger(
                RolloutLedgerError::PublicationAmbiguous { .. }
            ))
        ));
        assert_eq!(
            policy.sampler, 1,
            "collector rewound after crossing the manifest boundary"
        );
    }

    #[test]
    fn world_one_publication_panic_is_ambiguous_loadable_and_preserves_sampler() {
        let tmp = WireTmp::new("ledger-world-one-publication-panic");
        let ledger_root = tmp.0.join("ledger");
        let run = RunDir::create(&tmp.0, "run").unwrap();
        let mut trainer = Trainer::new(candidate_ledger_config(), &run).unwrap();
        let mut collector = stateful_candidate_policy();
        crate::rollout_ledger::inject_post_manifest_panic_once();
        let error = trainer
            .collect_rollout_ledger_step(
                0,
                &mut collector,
                &CandidateReward,
                &CandidateCodec,
                &[Sample::new("p", ())],
                &ledger_root,
                &"6".repeat(64),
                None,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TrainerError::RolloutLedger(RolloutLedgerError::PublicationAmbiguous { .. })
        ));
        assert_eq!(collector.sampler, 1);
        assert!(ledger_root
            .join("step-00000000000000000000/manifest.json")
            .is_file());

        let mut learner = stateful_candidate_policy();
        let (_, continuation) = trainer
            .train_rollout_ledger_step(0, &mut learner, &ledger_root, &"6".repeat(64), None)
            .unwrap();
        assert_eq!(learner.sampler, 1);
        assert_eq!(continuation.completed_step(), 1);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn distributed_publication_panic_is_ambiguous_loadable_and_preserves_every_sampler() {
        let tmp = WireTmp::new("ledger-distributed-publication-panic");
        let ledger_root = tmp.0.join("ledger");
        let outcomes = std::thread::scope(|scope| {
            let handles =
                crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10))
                    .into_iter()
                    .map(|comm| {
                        let base = tmp.0.clone();
                        let ledger_root = ledger_root.clone();
                        scope.spawn(move || {
                            let rank = comm.rank();
                            let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                            let mut trainer =
                                Trainer::with_comm(candidate_ledger_config(), &run, comm).unwrap();
                            let mut collector = stateful_candidate_policy();
                            if rank == 0 {
                                crate::rollout_ledger::inject_post_manifest_panic_once();
                            }
                            let collect_error = trainer
                                .collect_rollout_ledger_step(
                                    0,
                                    &mut collector,
                                    &CandidateReward,
                                    &CandidateCodec,
                                    &[Sample::new("p", ()), Sample::new("q", ())],
                                    &ledger_root,
                                    &"5".repeat(64),
                                    None,
                                )
                                .unwrap_err();
                            let mut learner = stateful_candidate_policy();
                            let (_, continuation) = trainer
                                .train_rollout_ledger_step(
                                    0,
                                    &mut learner,
                                    &ledger_root,
                                    &"5".repeat(64),
                                    None,
                                )
                                .unwrap();
                            (
                                rank,
                                collect_error,
                                collector.sampler,
                                learner.sampler,
                                continuation.completed_step(),
                            )
                        })
                    })
                    .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, error, collector_sampler, learner_sampler, completed_step) in outcomes {
            assert!(
                matches!(
                    error,
                    TrainerError::RolloutLedger(RolloutLedgerError::PublicationAmbiguous { .. })
                ),
                "rank {rank}: {error:?}"
            );
            assert_eq!(collector_sampler, 1, "rank {rank} collector sampler");
            assert_eq!(learner_sampler, 1, "rank {rank} learner sampler");
            assert_eq!(completed_step, 1, "rank {rank} continuation step");
        }
        assert!(ledger_root
            .join("step-00000000000000000000/manifest.json")
            .is_file());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_ledger_and_continuation_ambiguity_preserve_visible_state() {
        let tmp = WireTmp::new("tp-ledger-continuation-ambiguity");
        let ledger_root = tmp.0.join("ledger");
        let checkpoint_root = tmp.0.join("continuations");
        let outcomes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_secs(10),
            )
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let base = tmp.0.clone();
                let ledger_root = ledger_root.clone();
                let checkpoint_root = checkpoint_root.clone();
                let outcomes = std::sync::Arc::clone(&outcomes);
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                    let mut trainer = Trainer::new(candidate_ledger_config(), &run).unwrap();
                    let mut collector = stateful_candidate_policy();
                    if rank == 0 {
                        crate::rollout_ledger::inject_persistent_post_manifest_sync_failure_once();
                    }
                    let collect_error = trainer
                        .collect_rollout_ledger_step_tensor_parallel(
                            0,
                            &mut collector,
                            &CandidateReward,
                            &CandidateCodec,
                            &[Sample::new("p", ())],
                            &ledger_root,
                            &"7".repeat(64),
                            None,
                            &comm,
                        )
                        .unwrap_err();
                    let mut learner = stateful_candidate_policy();
                    let (_, continuation) = trainer
                        .train_rollout_ledger_step_tensor_parallel(
                            0,
                            &mut learner,
                            &ledger_root,
                            &"7".repeat(64),
                            None,
                            &comm,
                        )
                        .unwrap();
                    if rank == 0 {
                        crate::checkpoint::inject_persistent_continuation_post_manifest_sync_failure_once();
                    }
                    let save_error = trainer
                        .save_rollout_ledger_continuation_to_tensor_parallel(
                            &checkpoint_root,
                            &learner,
                            &continuation,
                            &comm,
                        )
                        .unwrap_err();
                    outcomes.lock().unwrap().push((
                        rank,
                        collect_error,
                        save_error,
                        collector.sampler,
                        learner.sampler,
                    ));
                })
            })
            .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });

        let mut outcomes = std::mem::take(&mut *outcomes.lock().unwrap());
        outcomes.sort_by_key(|outcome| outcome.0);
        for (rank, collect_error, save_error, collector_sampler, learner_sampler) in outcomes {
            assert!(
                matches!(
                    collect_error,
                    TrainerError::RolloutLedger(RolloutLedgerError::PublicationAmbiguous { .. })
                ),
                "rank {rank}: {collect_error:?}"
            );
            assert!(
                matches!(
                    save_error,
                    TrainerError::Checkpoint(
                        crate::checkpoint::CheckpointError::PublicationAmbiguous { .. }
                    )
                ),
                "rank {rank}: {save_error:?}"
            );
            assert_eq!(collector_sampler, 1, "rank {rank} collector sampler");
            assert_eq!(learner_sampler, 1, "rank {rank} learner sampler");
        }
        assert!(ledger_root
            .join("step-00000000000000000000/manifest.json")
            .is_file());
        assert!(checkpoint_root.join("step-1/manifest.json").is_file());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_publication_panics_are_ambiguous_and_loadable() {
        let tmp = WireTmp::new("tp-ledger-continuation-publication-panic");
        let ledger_root = tmp.0.join("ledger");
        let checkpoint_root = tmp.0.join("continuations");
        let policy_sha256 = "8".repeat(64);
        let outcomes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            let handles: Vec<_> =
                crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10))
                    .into_iter()
                    .enumerate()
                    .map(|(rank, comm)| {
                        let base = tmp.0.clone();
                        let ledger_root = ledger_root.clone();
                        let checkpoint_root = checkpoint_root.clone();
                        let policy_sha256 = policy_sha256.clone();
                        let outcomes = std::sync::Arc::clone(&outcomes);
                        scope.spawn(move || {
                            let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                            let mut trainer =
                                Trainer::new(candidate_ledger_config(), &run).unwrap();
                            let mut collector = stateful_candidate_policy();
                            if rank == 0 {
                                crate::rollout_ledger::inject_post_manifest_panic_once();
                            }
                            let collect_error = trainer
                                .collect_rollout_ledger_step_tensor_parallel(
                                    0,
                                    &mut collector,
                                    &CandidateReward,
                                    &CandidateCodec,
                                    &[Sample::new("p", ())],
                                    &ledger_root,
                                    &policy_sha256,
                                    None,
                                    &comm,
                                )
                                .unwrap_err();

                            let mut learner = stateful_candidate_policy();
                            let (_, continuation) = trainer
                                .train_rollout_ledger_step_tensor_parallel(
                                    0,
                                    &mut learner,
                                    &ledger_root,
                                    &policy_sha256,
                                    None,
                                    &comm,
                                )
                                .unwrap();
                            let learner_adapter = learner
                                .inner
                                .logp
                                .as_tensor()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                            if rank == 0 {
                                crate::checkpoint::inject_continuation_post_manifest_panic_once();
                            }
                            let save_error = trainer
                                .save_rollout_ledger_continuation_to_tensor_parallel(
                                    &checkpoint_root,
                                    &learner,
                                    &continuation,
                                    &comm,
                                )
                                .unwrap_err();

                            let mut restored_policy = stateful_candidate_policy();
                            restored_policy.sampler = 99;
                            let restored = trainer
                                .restore_latest_rollout_ledger_continuation_from_tensor_parallel(
                                    &checkpoint_root,
                                    &mut restored_policy,
                                    &policy_sha256,
                                    &comm,
                                )
                                .unwrap()
                                .expect("post-manifest checkpoint must remain discoverable");
                            let restored_adapter = restored_policy
                                .inner
                                .logp
                                .as_tensor()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                            outcomes.lock().unwrap().push((
                                rank,
                                collect_error,
                                save_error,
                                collector.sampler,
                                learner.sampler,
                                restored_policy.sampler,
                                restored.completed_step(),
                                learner_adapter == restored_adapter,
                                continuation.optimizer_sha256 == restored.optimizer_sha256,
                                continuation.lineage_sha256 == restored.lineage_sha256,
                            ));
                        })
                    })
                    .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });

        let mut outcomes = std::mem::take(&mut *outcomes.lock().unwrap());
        outcomes.sort_by_key(|outcome| outcome.0);
        for (
            rank,
            collect_error,
            save_error,
            collector_sampler,
            learner_sampler,
            restored_sampler,
            restored_step,
            adapter_matches,
            optimizer_matches,
            lineage_matches,
        ) in outcomes
        {
            assert!(
                matches!(
                    collect_error,
                    TrainerError::RolloutLedger(RolloutLedgerError::PublicationAmbiguous { .. })
                ),
                "rank {rank}: {collect_error:?}"
            );
            assert!(
                matches!(
                    save_error,
                    TrainerError::Checkpoint(
                        crate::checkpoint::CheckpointError::PublicationAmbiguous { .. }
                    )
                ),
                "rank {rank}: {save_error:?}"
            );
            assert_eq!(collector_sampler, 1, "rank {rank} collector sampler");
            assert_eq!(learner_sampler, 1, "rank {rank} learner sampler");
            assert_eq!(restored_sampler, 1, "rank {rank} restored sampler");
            assert_eq!(restored_step, 1, "rank {rank} restored step");
            assert!(adapter_matches, "rank {rank} restored adapter");
            assert!(optimizer_matches, "rank {rank} restored optimizer");
            assert!(lineage_matches, "rank {rank} restored lineage");
        }
        assert!(ledger_root
            .join("step-00000000000000000000/manifest.json")
            .is_file());
        assert!(checkpoint_root.join("step-1/manifest.json").is_file());
    }

    #[test]
    fn distributed_collector_propagates_post_manifest_ambiguity_to_every_rank() {
        let tmp = WireTmp::new("distributed-ledger-post-manifest-ambiguity");
        let ledger_root = tmp.0.join("ledger");
        let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
            2,
            std::time::Duration::from_secs(10),
        )
        .into_iter()
        .map(|comm| {
            let base = tmp.0.clone();
            let ledger_root = ledger_root.clone();
            std::thread::spawn(move || {
                let rank = comm.rank();
                let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                let config = TrainerConfig {
                    steps: 1,
                    group_size: 3,
                    grad_accum_steps: 1,
                    max_new_tokens: 2,
                    beta: 0.0,
                    checkpoint_every: None,
                    ..TrainerConfig::default()
                };
                let mut trainer = Trainer::with_comm(config, &run, comm).unwrap();
                let logp =
                    Var::from_tensor(&Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap())
                        .unwrap();
                let mut policy = StatefulCandidatePolicy {
                    inner: CandidatePolicy { logp },
                    sampler: 0,
                };
                if rank == 0 {
                    crate::rollout_ledger::inject_persistent_post_manifest_sync_failure_once();
                }
                let error = trainer
                    .collect_rollout_ledger_step(
                        0,
                        &mut policy,
                        &CandidateReward,
                        &CandidateCodec,
                        &[Sample::new("p", ()), Sample::new("q", ())],
                        &ledger_root,
                        &"a".repeat(64),
                        None,
                    )
                    .unwrap_err();
                assert!(matches!(
                    error,
                    TrainerError::RolloutLedger(RolloutLedgerError::PublicationAmbiguous { .. })
                ));
                policy.sampler
            })
        })
        .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 1);
        }
        assert!(ledger_root
            .join("step-00000000000000000000/manifest.json")
            .is_file());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // two exact-existing reconciliation boundaries
    fn distributed_exact_existing_sync_and_cleanup_faults_preserve_every_sampler() {
        for fault in ["sync", "cleanup"] {
            let tmp = WireTmp::new(&format!("distributed-exact-existing-{fault}"));
            let ledger_root = tmp.0.join("ledger");
            let final_dir = ledger_root.join("step-00000000000000000000");
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_secs(10),
            )
            .into_iter()
            .map(|comm| {
                let base = tmp.0.clone();
                let ledger_root = ledger_root.clone();
                let final_dir = final_dir.clone();
                std::thread::spawn(move || {
                    let rank = comm.rank();
                    let run = RunDir::create(&base, format!("{fault}-rank-{rank}")).unwrap();
                    let config = TrainerConfig {
                        steps: 1,
                        group_size: 3,
                        grad_accum_steps: 1,
                        max_new_tokens: 2,
                        beta: 0.0,
                        checkpoint_every: None,
                        ..TrainerConfig::default()
                    };
                    let mut trainer = Trainer::with_comm(config, &run, comm).unwrap();
                    let make_policy = || {
                        let logp = Var::from_tensor(
                            &Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap(),
                        )
                        .unwrap();
                        StatefulCandidatePolicy {
                            inner: CandidatePolicy { logp },
                            sampler: 0,
                        }
                    };
                    let mut first = make_policy();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut first,
                            &CandidateReward,
                            &CandidateCodec,
                            &[Sample::new("p", ()), Sample::new("q", ())],
                            &ledger_root,
                            &"d".repeat(64),
                            None,
                        )
                        .unwrap();
                    assert_eq!(first.sampler, 1);

                    if rank == 0 {
                        match fault {
                            "sync" => crate::rollout_ledger::inject_directory_sync_failure_once(
                                final_dir,
                            ),
                            "cleanup" => {
                                crate::rollout_ledger::inject_distributed_stage_cleanup_failure_once();
                            }
                            _ => unreachable!(),
                        }
                    }
                    let mut retry = make_policy();
                    let outcome = trainer.collect_rollout_ledger_step(
                        0,
                        &mut retry,
                        &CandidateReward,
                        &CandidateCodec,
                        &[Sample::new("p", ()), Sample::new("q", ())],
                        &ledger_root,
                        &"d".repeat(64),
                        None,
                    );
                    let cleanup_faults_consumed =
                        crate::rollout_ledger::distributed_stage_cleanup_faults_consumed();
                    (rank, outcome, retry.sampler, cleanup_faults_consumed)
                })
            })
            .collect();

            for handle in handles {
                let (rank, outcome, sampler, cleanup_faults_consumed) = handle.join().unwrap();
                match fault {
                    "sync" => assert!(
                        matches!(
                            outcome,
                            Err(TrainerError::RolloutLedger(
                                RolloutLedgerError::PublicationAmbiguous { .. }
                            ))
                        ),
                        "rank {rank}: {outcome:?}"
                    ),
                    "cleanup" => assert!(outcome.is_ok(), "rank {rank}: {outcome:?}"),
                    _ => unreachable!(),
                }
                assert_eq!(
                    cleanup_faults_consumed,
                    u32::from(fault == "cleanup" && rank == 0),
                    "{fault}: rank {rank} cleanup injector consumption"
                );
                assert_eq!(
                    sampler, 1,
                    "{fault}: rank {rank} rewound after exact visibility"
                );
            }
            assert!(final_dir.join("manifest.json").is_file());
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // three marker-exact payload uncertainty modes
    fn distributed_exact_marker_payload_uncertainty_preserves_every_sampler() {
        for fault in ["mutated", "missing", "read-failure"] {
            let tmp = WireTmp::new(&format!("distributed-exact-marker-{fault}"));
            let ledger_root = tmp.0.join("ledger");
            let final_dir = ledger_root.join("step-00000000000000000000");
            let payload_path = final_dir.join("rank-00000.window.json");
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_secs(10),
            )
            .into_iter()
            .map(|comm| {
                let base = tmp.0.clone();
                let ledger_root = ledger_root.clone();
                let payload_path = payload_path.clone();
                std::thread::spawn(move || {
                    let rank = comm.rank();
                    let run = RunDir::create(&base, format!("{fault}-rank-{rank}")).unwrap();
                    let config = TrainerConfig {
                        steps: 1,
                        group_size: 3,
                        grad_accum_steps: 1,
                        max_new_tokens: 2,
                        beta: 0.0,
                        checkpoint_every: None,
                        ..TrainerConfig::default()
                    };
                    let mut trainer = Trainer::with_comm(config, &run, comm).unwrap();
                    let make_policy = || {
                        let logp = Var::from_tensor(
                            &Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap(),
                        )
                        .unwrap();
                        StatefulCandidatePolicy {
                            inner: CandidatePolicy { logp },
                            sampler: 0,
                        }
                    };
                    let mut first = make_policy();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut first,
                            &CandidateReward,
                            &CandidateCodec,
                            &[Sample::new("p", ()), Sample::new("q", ())],
                            &ledger_root,
                            &"e".repeat(64),
                            None,
                        )
                        .unwrap();
                    assert_eq!(first.sampler, 1);
                    let manifest_path = ledger_root
                        .join("step-00000000000000000000")
                        .join("manifest.json");
                    let exact_manifest = std::fs::read(&manifest_path).unwrap();

                    if rank == 0 {
                        match fault {
                            "mutated" => std::fs::write(&payload_path, b"corrupt").unwrap(),
                            "missing" => std::fs::remove_file(&payload_path).unwrap(),
                            "read-failure" => crate::rollout_ledger::inject_distributed_reconciliation_read_failure_once(&payload_path),
                            _ => unreachable!(),
                        }
                    }
                    let mut retry = make_policy();
                    let outcome = trainer.collect_rollout_ledger_step(
                        0,
                        &mut retry,
                        &CandidateReward,
                        &CandidateCodec,
                        &[Sample::new("p", ()), Sample::new("q", ())],
                        &ledger_root,
                        &"e".repeat(64),
                        None,
                    );
                    assert_eq!(std::fs::read(manifest_path).unwrap(), exact_manifest);
                    (rank, outcome, retry.sampler)
                })
            })
            .collect();

            for handle in handles {
                let (rank, outcome, sampler) = handle.join().unwrap();
                assert!(
                    matches!(
                        outcome,
                        Err(TrainerError::RolloutLedger(
                            RolloutLedgerError::PublicationAmbiguous { .. }
                        ))
                    ),
                    "{fault}: rank {rank}: {outcome:?}"
                );
                assert_eq!(
                    sampler, 1,
                    "{fault}: rank {rank} rewound after the exact manifest became visible"
                );
            }
            assert!(final_dir.join("manifest.json").is_file());
            match fault {
                "mutated" => assert_eq!(std::fs::read(&payload_path).unwrap(), b"corrupt"),
                "missing" => assert!(!payload_path.exists()),
                "read-failure" => assert!(payload_path.is_file()),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn distributed_continuation_publication_broadcasts_retry_and_ambiguity() {
        for persistent_post_manifest in [false, true] {
            let tag = if persistent_post_manifest {
                "post-manifest"
            } else {
                "pre-manifest"
            };
            let tmp = WireTmp::new(&format!("distributed-continuation-{tag}"));
            let ledger_root = tmp.0.join("ledger");
            let checkpoint_root = tmp.0.join("continuations");
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_secs(10),
            )
            .into_iter()
            .map(|comm| {
                let base = tmp.0.clone();
                let ledger_root = ledger_root.clone();
                let checkpoint_root = checkpoint_root.clone();
                std::thread::spawn(move || {
                    let rank = comm.rank();
                    let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                    let config = TrainerConfig {
                        steps: 1,
                        group_size: 3,
                        grad_accum_steps: 1,
                        max_new_tokens: 2,
                        beta: 0.0,
                        checkpoint_every: None,
                        ..TrainerConfig::default()
                    };
                    let mut trainer = Trainer::with_comm(config, &run, comm).unwrap();
                    let make_policy = || {
                        let logp = Var::from_tensor(
                            &Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap(),
                        )
                        .unwrap();
                        StatefulCandidatePolicy {
                            inner: CandidatePolicy { logp },
                            sampler: 0,
                        }
                    };
                    let mut collector_policy = make_policy();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut collector_policy,
                            &CandidateReward,
                            &CandidateCodec,
                            &[Sample::new("p", ()), Sample::new("q", ())],
                            &ledger_root,
                            &"b".repeat(64),
                            None,
                        )
                        .unwrap();
                    let mut learner_policy = make_policy();
                    let (_, continuation) = trainer
                        .train_rollout_ledger_step(
                            0,
                            &mut learner_policy,
                            &ledger_root,
                            &"b".repeat(64),
                            None,
                        )
                        .unwrap();
                    if rank == 0 {
                        if persistent_post_manifest {
                            crate::checkpoint::inject_persistent_continuation_post_manifest_sync_failure_once();
                        } else {
                            crate::checkpoint::inject_continuation_pre_manifest_sync_failure_once();
                        }
                    }
                    let first = trainer.save_rollout_ledger_continuation_to(
                        &checkpoint_root,
                        &learner_policy,
                        &continuation,
                    );
                    if persistent_post_manifest {
                        assert!(matches!(
                            first,
                            Err(TrainerError::Checkpoint(
                                crate::checkpoint::CheckpointError::PublicationAmbiguous { .. }
                            ))
                        ));
                    } else {
                        assert!(first.is_err());
                        trainer
                            .save_rollout_ledger_continuation_to(
                                &checkpoint_root,
                                &learner_policy,
                                &continuation,
                            )
                            .unwrap();
                    }
                })
            })
            .collect();
            for handle in handles {
                handle.join().unwrap();
            }
            assert!(checkpoint_root.join("step-1/manifest.json").is_file());
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit two-rank fault protocol
    fn distributed_ledger_publication_comm_failure_preserves_visibility_and_dead_world() {
        use std::sync::atomic::Ordering;

        let tmp = WireTmp::new("distributed-ledger-publication-terminal-comm");
        let ledger_root = tmp.0.join("ledger");
        let final_dir = ledger_root.join("step-00000000000000000000");
        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Arming on the collector's second sampler snapshot leaves exactly 21
        // successful collectives through shard status; the next call is the
        // manifest-publication status reduction and must fail on both ranks.
        let states: Vec<_> = (0..2)
            .map(|_| std::sync::Arc::new(ArmedCollectiveFailureState::new(21)))
            .collect();
        let comms =
            crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .zip(states.iter().cloned())
                .map(|(inner, state)| {
                    let rank = inner.rank();
                    let base = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let final_dir = final_dir.clone();
                    let armed = std::sync::Arc::clone(&armed);
                    scope.spawn(move || {
                        let comm = FailAfterArmComm {
                            inner,
                            armed: std::sync::Arc::clone(&armed),
                            state,
                        };
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let mut trainer =
                            Trainer::with_comm(candidate_ledger_config(), &run, comm).unwrap();
                        let mut policy = ArmedCandidatePolicy {
                            inner: stateful_candidate_policy(),
                            armed,
                            arm_point: CandidatePolicyArmPoint::SamplerState(2),
                            sampler_state_calls: std::cell::Cell::new(0),
                        };
                        if rank == 0 {
                            crate::rollout_ledger::inject_post_manifest_panic_once();
                        }
                        let error = trainer
                            .collect_rollout_ledger_step(
                                0,
                                &mut policy,
                                &CandidateReward,
                                &CandidateCodec,
                                &[Sample::new("p", ()), Sample::new("q", ())],
                                &ledger_root,
                                &"f".repeat(64),
                                None,
                            )
                            .unwrap_err();
                        match &error {
                            TrainerError::PublicationAmbiguousAfterComm {
                                artifact,
                                path,
                                communication: _,
                                detail: _,
                            } => {
                                assert_eq!(*artifact, "distributed rollout ledger");
                                assert_eq!(path, &final_dir);
                            }
                            other => panic!(
                                "rank {rank}: expected publication+communication ambiguity, got {other:?}"
                            ),
                        }
                        (rank, error.to_string(), policy.inner.sampler)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        for (rank, message, sampler) in outcomes {
            assert!(
                message.contains("publication may be visible"),
                "rank {rank}: {message}"
            );
            assert!(
                message.contains("distributed execution world is dead"),
                "rank {rank}: {message}"
            );
            assert!(
                message.contains("discard the communicator"),
                "rank {rank}: {message}"
            );
            assert_eq!(
                sampler, 1,
                "rank {rank} rewound a potentially visible ledger"
            );
        }
        for (rank, state) in states.iter().enumerate() {
            assert!(
                state.failed.load(Ordering::SeqCst),
                "rank {rank}: fault not consumed"
            );
            assert_eq!(
                state.remaining_successes.load(Ordering::SeqCst),
                0,
                "rank {rank}: failed at the wrong publication boundary"
            );
            assert_eq!(
                state.calls_after_failure.load(Ordering::SeqCst),
                0,
                "rank {rank}: collective issued after publication-status failure"
            );
        }
        assert!(final_dir.join("manifest.json").is_file());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit two-rank fault protocol
    fn distributed_continuation_publication_comm_failure_preserves_visibility_and_dead_world() {
        use std::sync::atomic::Ordering;

        let tmp = WireTmp::new("distributed-continuation-publication-terminal-comm");
        let ledger_root = tmp.0.join("ledger");
        let checkpoint_root = tmp.0.join("continuations");
        let final_dir = checkpoint_root.join("step-1");
        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        // Save preflight, serialization, and the eight-word digest consensus
        // consume 11 successful reductions. Publication status is the next one.
        let states: Vec<_> = (0..2)
            .map(|_| std::sync::Arc::new(ArmedCollectiveFailureState::new(11)))
            .collect();
        let comms =
            crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .zip(states.iter().cloned())
                .map(|(inner, state)| {
                    let rank = inner.rank();
                    let base = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let checkpoint_root = checkpoint_root.clone();
                    let final_dir = final_dir.clone();
                    let armed = std::sync::Arc::clone(&armed);
                    let barrier = std::sync::Arc::clone(&barrier);
                    scope.spawn(move || {
                        let comm = FailAfterArmComm {
                            inner,
                            armed: std::sync::Arc::clone(&armed),
                            state,
                        };
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let mut trainer =
                            Trainer::with_comm(candidate_ledger_config(), &run, comm).unwrap();
                        let mut collector = stateful_candidate_policy();
                        trainer
                            .collect_rollout_ledger_step(
                                0,
                                &mut collector,
                                &CandidateReward,
                                &CandidateCodec,
                                &[Sample::new("p", ()), Sample::new("q", ())],
                                &ledger_root,
                                &"1".repeat(64),
                                None,
                            )
                            .unwrap();
                        let mut learner = stateful_candidate_policy();
                        let (_, continuation) = trainer
                            .train_rollout_ledger_step(
                                0,
                                &mut learner,
                                &ledger_root,
                                &"1".repeat(64),
                                None,
                            )
                            .unwrap();
                        barrier.wait();
                        if rank == 0 {
                            armed.store(true, Ordering::SeqCst);
                        }
                        barrier.wait();
                        let error = trainer
                            .save_rollout_ledger_continuation_to(
                                &checkpoint_root,
                                &learner,
                                &continuation,
                            )
                            .unwrap_err();
                        match &error {
                            TrainerError::PublicationAmbiguousAfterComm {
                                artifact,
                                path,
                                communication: _,
                                detail: _,
                            } => {
                                assert_eq!(*artifact, "rollout-ledger continuation");
                                assert_eq!(path, &final_dir);
                            }
                            other => panic!(
                                "rank {rank}: expected publication+communication ambiguity, got {other:?}"
                            ),
                        }
                        (rank, error.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        for (rank, message) in outcomes {
            assert!(
                message.contains("publication may be visible"),
                "rank {rank}: {message}"
            );
            assert!(
                message.contains("distributed execution world is dead"),
                "rank {rank}: {message}"
            );
            assert!(
                message.contains("discard the communicator"),
                "rank {rank}: {message}"
            );
        }
        for (rank, state) in states.iter().enumerate() {
            assert!(
                state.failed.load(Ordering::SeqCst),
                "rank {rank}: fault not consumed"
            );
            assert_eq!(state.remaining_successes.load(Ordering::SeqCst), 0);
            assert_eq!(
                state.calls_after_failure.load(Ordering::SeqCst),
                0,
                "rank {rank}: collective issued after publication-status failure"
            );
        }
        assert!(final_dir.join("manifest.json").is_file());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit two-rank fault protocol
    fn nonce_chain_stops_after_first_failed_reduction() {
        use std::sync::atomic::Ordering;

        let tmp = WireTmp::new("nonce-chain-first-terminal-comm");
        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        // Fail nonce-low immediately. Any eager nonce-high call is therefore a
        // collective after terminal communication failure and must be observed.
        let states: Vec<_> = (0..2)
            .map(|_| std::sync::Arc::new(ArmedCollectiveFailureState::new(0)))
            .collect();
        let comms =
            crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .zip(states.iter().cloned())
                .map(|(inner, state)| {
                    let rank = inner.rank();
                    let base = tmp.0.clone();
                    let armed = std::sync::Arc::clone(&armed);
                    scope.spawn(move || {
                        let comm = FailAfterArmComm {
                            inner,
                            armed,
                            state,
                        };
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let trainer =
                            Trainer::with_comm(candidate_ledger_config(), &run, comm).unwrap();
                        let error = trainer.distributed_rollout_ledger_nonce().unwrap_err();
                        (rank, error)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        for (rank, error) in outcomes {
            assert!(
                matches!(&error, TrainerError::Comm(_)),
                "rank {rank}: expected nonce-low communication failure, got {error:?}"
            );
        }
        for (rank, state) in states.iter().enumerate() {
            assert!(
                state.failed.load(Ordering::SeqCst),
                "rank {rank}: nonce-low fault not consumed"
            );
            assert_eq!(state.remaining_successes.load(Ordering::SeqCst), 0);
            assert_eq!(
                state.calls_after_failure.load(Ordering::SeqCst),
                0,
                "rank {rank}: nonce-high ran after nonce-low communication failure"
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit two-rank fault protocol
    fn collector_count_chain_stops_after_first_failed_reduction() {
        use std::sync::atomic::Ordering;

        let tmp = WireTmp::new("collector-count-chain-first-terminal-comm");
        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        // Local derivation and exact conversion each coordinate once. Fail the
        // following token-count reduction; an eager live-count call must trip
        // the post-failure counter.
        let states: Vec<_> = (0..2)
            .map(|_| std::sync::Arc::new(ArmedCollectiveFailureState::new(2)))
            .collect();
        let comms =
            crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .zip(states.iter().cloned())
                .map(|(inner, state)| {
                    let rank = inner.rank();
                    let base = tmp.0.clone();
                    let armed = std::sync::Arc::clone(&armed);
                    scope.spawn(move || {
                        let comm = FailAfterArmComm {
                            inner,
                            armed,
                            state,
                        };
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let trainer =
                            Trainer::with_comm(candidate_ledger_config(), &run, comm).unwrap();
                        let error = trainer.rollout_ledger_global_counts(&[], 0.0).unwrap_err();
                        (rank, error)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        for (rank, error) in outcomes {
            assert!(
                matches!(&error, TrainerError::Comm(_)),
                "rank {rank}: expected token-count communication failure, got {error:?}"
            );
        }
        for (rank, state) in states.iter().enumerate() {
            assert!(
                state.failed.load(Ordering::SeqCst),
                "rank {rank}: token-count fault not consumed"
            );
            assert_eq!(state.remaining_successes.load(Ordering::SeqCst), 0);
            assert_eq!(
                state.calls_after_failure.load(Ordering::SeqCst),
                0,
                "rank {rank}: live-count ran after token-count communication failure"
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit two-rank fault protocol
    fn learner_count_chain_stops_after_first_failed_reduction() {
        use std::sync::atomic::Ordering;

        let tmp = WireTmp::new("learner-count-chain-terminal-comm");
        let ledger_root = tmp.0.join("ledger");
        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        // After detached scoring: scoring status, local-count status, exact
        // conversion succeed; token-count then fails, so live-count must never
        // be attempted.
        let states: Vec<_> = (0..2)
            .map(|_| std::sync::Arc::new(ArmedCollectiveFailureState::new(3)))
            .collect();
        let comms =
            crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .zip(states.iter().cloned())
                .map(|(inner, state)| {
                    let rank = inner.rank();
                    let base = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let armed = std::sync::Arc::clone(&armed);
                    scope.spawn(move || {
                        let comm = FailAfterArmComm {
                            inner,
                            armed: std::sync::Arc::clone(&armed),
                            state,
                        };
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let mut trainer =
                            Trainer::with_comm(candidate_ledger_config(), &run, comm).unwrap();
                        let mut collector = stateful_candidate_policy();
                        trainer
                            .collect_rollout_ledger_step(
                                0,
                                &mut collector,
                                &CandidateReward,
                                &CandidateCodec,
                                &[Sample::new("p", ()), Sample::new("q", ())],
                                &ledger_root,
                                &"2".repeat(64),
                                None,
                            )
                            .unwrap();
                        let mut learner = ArmedCandidatePolicy {
                            inner: stateful_candidate_policy(),
                            armed,
                            arm_point: CandidatePolicyArmPoint::DetachedScoring,
                            sampler_state_calls: std::cell::Cell::new(0),
                        };
                        let error = trainer
                            .train_rollout_ledger_step(
                                0,
                                &mut learner,
                                &ledger_root,
                                &"2".repeat(64),
                                None,
                            )
                            .unwrap_err();
                        (rank, error.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        for (rank, message) in outcomes {
            assert!(
                message.contains("distributed execution world is dead"),
                "rank {rank}: {message}"
            );
            assert!(message.contains("discard the policy and optimizer state"));
        }
        for (rank, state) in states.iter().enumerate() {
            assert!(
                state.failed.load(Ordering::SeqCst),
                "rank {rank}: fault not consumed"
            );
            assert_eq!(state.remaining_successes.load(Ordering::SeqCst), 0);
            assert_eq!(
                state.calls_after_failure.load(Ordering::SeqCst),
                0,
                "rank {rank}: learner live-count ran after token-count communication failure"
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit two-rank fault protocol
    fn reward_statistics_chain_stops_after_first_failed_reduction() {
        use std::sync::atomic::Ordering;

        let tmp = WireTmp::new("reward-stat-chain-terminal-comm");
        let ledger_root = tmp.0.join("ledger");
        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Reward callback arming is followed by rollout/reward status and the
        // reward-count reduction. Reward-sum then fails, so sum-of-squares must
        // never be attempted.
        let states: Vec<_> = (0..2)
            .map(|_| std::sync::Arc::new(ArmedCollectiveFailureState::new(2)))
            .collect();
        let comms =
            crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .zip(states.iter().cloned())
                .map(|(inner, state)| {
                    let rank = inner.rank();
                    let base = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let armed = std::sync::Arc::clone(&armed);
                    scope.spawn(move || {
                        let comm = FailAfterArmComm {
                            inner,
                            armed: std::sync::Arc::clone(&armed),
                            state,
                        };
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let mut config = candidate_ledger_config();
                        config.reward_group_scope = RewardGroupScope::DistributedSamePrompt;
                        config.candidate_log_top_k = 0;
                        let mut trainer = Trainer::with_comm(config, &run, comm).unwrap();
                        let mut policy = stateful_candidate_policy();
                        let error = trainer
                            .collect_rollout_ledger_step(
                                0,
                                &mut policy,
                                &ArmedCandidateReward { armed },
                                &CandidateCodec,
                                &[Sample::new("p", ()), Sample::new("q", ())],
                                &ledger_root,
                                &"3".repeat(64),
                                None,
                            )
                            .unwrap_err();
                        (rank, error.to_string(), policy.sampler)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        for (rank, message, sampler) in outcomes {
            assert!(
                message.contains("distributed execution world is dead"),
                "rank {rank}: {message}"
            );
            assert_eq!(
                sampler, 0,
                "rank {rank}: local sampler rollback did not run"
            );
        }
        for (rank, state) in states.iter().enumerate() {
            assert!(
                state.failed.load(Ordering::SeqCst),
                "rank {rank}: fault not consumed"
            );
            assert_eq!(state.remaining_successes.load(Ordering::SeqCst), 0);
            assert_eq!(
                state.calls_after_failure.load(Ordering::SeqCst),
                0,
                "rank {rank}: reward sumsq reduction ran after reward-sum failure"
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit fault matrix and two-rank protocol
    fn distributed_metrics_comm_failure_reports_truncate_error_and_panic() {
        use std::sync::atomic::Ordering;

        for (fault, injected_message) in [
            ("error", "injected metrics truncate failure"),
            ("panic", "injected metrics truncate panic"),
        ] {
            let tmp = WireTmp::new(&format!("distributed-metrics-terminal-{fault}"));
            let ledger_root = tmp.0.join("ledger");
            let states: Vec<_> = (0..2)
                .map(|_| std::sync::Arc::new(MetricsStatusFailureState::new()))
                .collect();
            let comms =
                crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
            let outcomes: Vec<_> = std::thread::scope(|scope| {
                let handles: Vec<_> = comms
                    .into_iter()
                    .zip(states.iter().cloned())
                    .map(|(inner, state)| {
                        let rank = inner.rank();
                        let base = tmp.0.clone();
                        let ledger_root = ledger_root.clone();
                        scope.spawn(move || {
                            let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                            let comm = FailAtMetricsStatusComm {
                                inner,
                                metrics_path: run.metrics_path().to_path_buf(),
                                state,
                            };
                            let mut trainer =
                                Trainer::with_comm(candidate_ledger_config(), &run, comm).unwrap();
                            let mut collector = stateful_candidate_policy();
                            trainer
                                .collect_rollout_ledger_step(
                                    0,
                                    &mut collector,
                                    &CandidateReward,
                                    &CandidateCodec,
                                    &[Sample::new("p", ()), Sample::new("q", ())],
                                    &ledger_root,
                                    &"4".repeat(64),
                                    None,
                                )
                                .unwrap();

                            let mut learner = stateful_candidate_policy();
                            let adapter_before = learner
                                .inner
                                .logp
                                .as_tensor()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                            let sampler_before = learner.sampler;
                            match fault {
                                "error" => trainer.writer.inject_truncate_failure_once(),
                                "panic" => trainer.writer.inject_truncate_panic_once(),
                                other => unreachable!("unknown metrics rollback fault {other}"),
                            }
                            let error = trainer
                                .train_rollout_ledger_step(
                                    0,
                                    &mut learner,
                                    &ledger_root,
                                    &"4".repeat(64),
                                    None,
                                )
                                .unwrap_err();
                            let adapter_after = learner
                                .inner
                                .logp
                                .as_tensor()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                            (
                                rank,
                                error.to_string(),
                                adapter_before,
                                adapter_after,
                                sampler_before,
                                learner.sampler,
                                crate::telemetry::read_metrics(run.metrics_path())
                                    .unwrap()
                                    .len(),
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect()
            });

            for (
                rank,
                message,
                adapter_before,
                adapter_after,
                sampler_before,
                sampler_after,
                rows,
            ) in outcomes
            {
                assert!(
                    message.contains("distributed execution world is dead"),
                    "{fault} rank {rank}: {message}"
                );
                assert!(
                    message.contains("best-effort local metrics rollback failed"),
                    "{fault} rank {rank}: {message}"
                );
                assert!(
                    message.contains(injected_message),
                    "{fault} rank {rank}: {message}"
                );
                assert!(
                    message.contains("must be repaired or discarded"),
                    "{fault} rank {rank}: {message}"
                );
                assert!(
                    message.contains("best-effort local rollback completed"),
                    "{fault} rank {rank}: {message}"
                );
                assert!(
                    message.contains("discard the policy and optimizer state"),
                    "{fault} rank {rank}: {message}"
                );
                assert_eq!(adapter_after, adapter_before, "{fault} rank {rank} adapter");
                assert_eq!(sampler_after, sampler_before, "{fault} rank {rank} sampler");
                assert_eq!(
                    rows, 1,
                    "{fault} rank {rank}: injected truncate fault unexpectedly removed the row"
                );
            }
            for (rank, state) in states.iter().enumerate() {
                assert!(
                    state.failed.load(Ordering::SeqCst),
                    "{fault} rank {rank}: metrics status fault not consumed"
                );
                assert_eq!(
                    state.calls_after_failure.load(Ordering::SeqCst),
                    0,
                    "{fault} rank {rank}: collective issued after metrics status failure"
                );
            }
        }
    }

    #[test]
    fn distributed_metrics_failure_truncates_successful_peers_and_rolls_back_state() {
        let tmp = WireTmp::new("distributed-ledger-metrics-transaction");
        let ledger_root = tmp.0.join("ledger");
        let handles: Vec<_> =
            crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10))
                .into_iter()
                .map(|comm| {
                    let base = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    std::thread::spawn(move || {
                        let rank = comm.rank();
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let config = TrainerConfig {
                            steps: 1,
                            group_size: 3,
                            grad_accum_steps: 1,
                            max_new_tokens: 2,
                            beta: 0.0,
                            mu: 1,
                            checkpoint_every: None,
                            ..TrainerConfig::default()
                        };
                        let mut trainer = Trainer::with_comm(config, &run, comm).unwrap();
                        let make_policy = || {
                            let logp = Var::from_tensor(
                                &Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap(),
                            )
                            .unwrap();
                            StatefulCandidatePolicy {
                                inner: CandidatePolicy { logp },
                                sampler: 0,
                            }
                        };
                        let mut collector_policy = make_policy();
                        trainer
                            .collect_rollout_ledger_step(
                                0,
                                &mut collector_policy,
                                &CandidateReward,
                                &CandidateCodec,
                                &[Sample::new("p", ()), Sample::new("q", ())],
                                &ledger_root,
                                &"c".repeat(64),
                                None,
                            )
                            .unwrap();
                        let mut learner_policy = make_policy();
                        let adapter_before = learner_policy
                            .inner
                            .logp
                            .as_tensor()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        let sampler_before = learner_policy.sampler;
                        if rank == 1 {
                            trainer.writer.inject_append_failure_once();
                        }
                        let error = trainer
                            .train_rollout_ledger_step(
                                0,
                                &mut learner_policy,
                                &ledger_root,
                                &"c".repeat(64),
                                None,
                            )
                            .unwrap_err();
                        let adapter_after = learner_policy
                            .inner
                            .logp
                            .as_tensor()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        (
                            rank,
                            error.to_string(),
                            adapter_before,
                            adapter_after,
                            sampler_before,
                            learner_policy.sampler,
                            crate::telemetry::read_metrics(run.metrics_path())
                                .unwrap()
                                .len(),
                        )
                    })
                })
                .collect();
        for handle in handles {
            let (rank, error, adapter_before, adapter_after, sampler_before, sampler_after, rows) =
                handle.join().unwrap();
            assert!(!error.contains("timeout"), "rank {rank}: {error}");
            assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
            assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
            assert_eq!(rows, 0, "rank {rank} retained a metrics row");
        }
    }

    struct TelemetryProbePolicy {
        logp: Var,
        seen_telemetry: std::sync::Arc<std::sync::Mutex<Vec<bool>>>,
    }
    impl Policy for TelemetryProbePolicy {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            self.generate_at_instrumented(prompt, cfg, 0, None)
        }

        fn generate_at_instrumented(
            &mut self,
            prompt: &[u32],
            _cfg: &GenConfig,
            _global_row_base: u64,
            telemetry: Option<&mut dyn ModelTelemetryRecorder>,
        ) -> CandleResult<Rollout> {
            self.seen_telemetry
                .lock()
                .unwrap()
                .push(telemetry.is_some());
            Ok(Rollout {
                token_ids: vec![vec![prompt[0], 7]],
                prompt_len: prompt.len(),
                completion_lens: vec![1],
                rollout_logprobs: None,
            })
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            Ok(self.logp.as_tensor().clone())
        }

        fn set_adapter_enabled(&mut self, _enabled: bool) {}
        fn adapter_enabled(&self) -> bool {
            true
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    fn telemetry_probe_seen(gpu_memory_probe: bool) -> Vec<bool> {
        let tmp = WireTmp::new("telemetry-probe");
        let run = RunDir::create(&tmp.0, "telemetry-probe-run").unwrap();
        let cfg = TrainerConfig {
            steps: 1,
            group_size: 1,
            max_new_tokens: 1,
            lr: 0.0,
            gpu_memory_probe,
            ..TrainerConfig::default()
        };
        let seen_telemetry = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let logp = Var::from_tensor(&mat(&[&[-1.0]])).unwrap();
        let mut policy = TelemetryProbePolicy {
            logp,
            seen_telemetry: std::sync::Arc::clone(&seen_telemetry),
        };
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        trainer
            .train(
                &mut policy,
                &CandidateReward,
                &CandidateCodec,
                &[Sample::new("prompt", ())],
            )
            .unwrap();
        let seen = seen_telemetry.lock().unwrap().clone();
        seen
    }

    #[test]
    fn trainer_passes_model_telemetry_recorder_only_when_gpu_probe_enabled() {
        assert_eq!(telemetry_probe_seen(false), vec![false]);
        assert_eq!(telemetry_probe_seen(true), vec![true]);
    }

    #[derive(Debug, Clone, Copy)]
    struct ProbeTpComm {
        rank: usize,
        world_size: usize,
    }

    impl Comm for ProbeTpComm {
        fn rank(&self) -> usize {
            self.rank
        }

        fn world_size(&self) -> usize {
            self.world_size
        }

        fn all_reduce_sum(&self, _tensors: &mut Vec<Tensor>) -> Result<(), crate::comm::CommError> {
            panic!("ProbeTpComm should only be inspected by the trainer dispatch test")
        }

        fn all_reduce_scalar_sum(&self, _value: f64) -> Result<f64, crate::comm::CommError> {
            panic!("ProbeTpComm should only be inspected by the trainer dispatch test")
        }
    }

    #[derive(Debug)]
    struct CountScalarComm<C> {
        inner: C,
        scalar_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl<C: Comm> Comm for CountScalarComm<C> {
        fn rank(&self) -> usize {
            self.inner.rank()
        }

        fn world_size(&self) -> usize {
            self.inner.world_size()
        }

        fn validate_all_reduce_sum(
            &self,
            tensors: &[Tensor],
        ) -> Result<(), crate::comm::CommError> {
            self.inner.validate_all_reduce_sum(tensors)
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), crate::comm::CommError> {
            self.inner.all_reduce_sum(tensors)
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, crate::comm::CommError> {
            self.scalar_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.all_reduce_scalar_sum(value)
        }
    }

    #[derive(Clone, Default)]
    struct TpProbeCalls {
        generate: usize,
        live_logp: usize,
        detached_logp: usize,
        backward: usize,
        telemetry_seen: Vec<bool>,
        comms: Vec<(usize, usize)>,
    }

    struct TpProbePolicy {
        logp: Var,
        enabled: bool,
        sharded_backward: bool,
        panic_backward_capability: bool,
        calls: std::sync::Arc<std::sync::Mutex<TpProbeCalls>>,
    }

    impl Policy for TpProbePolicy {
        fn generate(&mut self, _prompt: &[u32], _cfg: &GenConfig) -> CandleResult<Rollout> {
            panic!("train_tensor_parallel must not call Policy::generate")
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            panic!("train_tensor_parallel must not call Policy::token_logprobs")
        }

        fn token_logprobs_detached(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            panic!("train_tensor_parallel must not call Policy::token_logprobs_detached")
        }

        fn backward(&self, _loss: &Tensor) -> CandleResult<GradStore> {
            panic!("train_tensor_parallel must not call Policy::backward")
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        fn adapter_enabled(&self) -> bool {
            self.enabled
        }

        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logp.clone()]
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
            Ok(())
        }
    }

    impl TensorParallelPolicy for TpProbePolicy {
        fn supports_sharded_tensor_parallel_backward(&self) -> bool {
            if self.panic_backward_capability {
                panic!("injected backward capability panic");
            }
            self.sharded_backward
        }

        fn generate_at_tensor_parallel_instrumented(
            &mut self,
            prompt: &[u32],
            _cfg: &GenConfig,
            _global_row_base: u64,
            comm: &dyn Comm,
            telemetry: Option<&mut dyn ModelTelemetryRecorder>,
        ) -> CandleResult<Rollout> {
            let mut calls = self.calls.lock().unwrap();
            calls.generate += 1;
            calls.telemetry_seen.push(telemetry.is_some());
            calls.comms.push((comm.rank(), comm.world_size()));
            Ok(Rollout {
                token_ids: vec![vec![prompt[0], 1], vec![prompt[0], 2]],
                prompt_len: prompt.len(),
                completion_lens: vec![1, 1],
                rollout_logprobs: Some(vec![vec![-0.5], vec![-0.5]]),
            })
        }

        fn token_logprobs_tensor_parallel(
            &self,
            _rollout: &Rollout,
            comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            let mut calls = self.calls.lock().unwrap();
            calls.live_logp += 1;
            calls.comms.push((comm.rank(), comm.world_size()));
            Ok(self.logp.as_tensor().clone())
        }

        fn token_logprobs_tensor_parallel_detached(
            &self,
            _rollout: &Rollout,
            comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            let mut calls = self.calls.lock().unwrap();
            calls.detached_logp += 1;
            calls.comms.push((comm.rank(), comm.world_size()));
            Ok(self.logp.as_tensor().detach())
        }

        fn backward_tensor_parallel(
            &self,
            loss: &Tensor,
            comm: &dyn Comm,
        ) -> CandleResult<GradStore> {
            let mut calls = self.calls.lock().unwrap();
            calls.backward += 1;
            calls.comms.push((comm.rank(), comm.world_size()));
            drop(calls);
            loss.backward()
        }
    }

    struct TpProbeCodec;
    impl TokenizerLike for TpProbeCodec {
        fn encode(&self, _text: &str) -> Vec<u32> {
            vec![42]
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    struct TpProbeReward;
    impl RewardFn for TpProbeReward {
        type Target = ();

        fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
            Ok(match completion {
                "1" => 0.0,
                "2" => 2.0,
                other => panic!("unexpected completion {other}"),
            })
        }
    }

    #[derive(Clone, Copy)]
    enum CoordinatedTpRewardMode {
        Scores([f32; 2]),
        Error,
        CountMismatch,
        Panic,
    }

    struct CoordinatedTpReward {
        rank: usize,
        mode: CoordinatedTpRewardMode,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl RewardFn for CoordinatedTpReward {
        type Target = ();

        fn reward(&self, _sample: &Sample<()>, _completion: &str) -> Result<f32, RewardError> {
            panic!("coordinated TP reward test uses the detailed group seam")
        }

        fn reward_group_detailed(
            &self,
            _sample: &Sample<()>,
            completions: &[String],
        ) -> Result<Vec<RewardOutcome>, RewardError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match self.mode {
                CoordinatedTpRewardMode::Scores(scores) => Ok(scores
                    .into_iter()
                    .enumerate()
                    .map(|(index, reward)| RewardOutcome {
                        reward,
                        diagnostic: Some(format!("rank{}:{index}", self.rank)),
                        metadata: Some(serde_json::json!({
                            "scoring_rank": self.rank,
                            "completion": completions[index],
                        })),
                    })
                    .collect()),
                CoordinatedTpRewardMode::Error => {
                    Err(RewardError::msg("execution-primary reward failure"))
                }
                CoordinatedTpRewardMode::CountMismatch => Ok(vec![RewardOutcome::reward(1.0)]),
                CoordinatedTpRewardMode::Panic => {
                    panic!("execution-primary reward panic")
                }
            }
        }
    }

    struct CoordinatedTpRunResult {
        rank: usize,
        result: Result<(Vec<Metrics>, RunStop), TrainerError>,
        reward_calls: usize,
        policy_calls: TpProbeCalls,
        candidates: String,
    }

    fn run_coordinated_tp_reward_case(
        modes: [CoordinatedTpRewardMode; 2],
        candidate_log_top_k: usize,
        beta: f64,
    ) -> Vec<CoordinatedTpRunResult> {
        let tmp = WireTmp::new("tp-coordinated-reward");
        let results = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .zip(modes)
                .enumerate()
                .map(|(rank, (tp_comm, mode))| {
                    let root = tmp.0.clone();
                    let results = std::sync::Arc::clone(&results);
                    scope.spawn(move || {
                        let run = RunDir::create(&root, format!("tp-reward-rank-{rank}")).unwrap();
                        let cfg = TrainerConfig {
                            steps: 1,
                            group_size: 2,
                            max_new_tokens: 1,
                            lr: 0.0,
                            beta,
                            candidate_log_top_k,
                            ..TrainerConfig::default()
                        };
                        let mut trainer = Trainer::new(cfg, &run).unwrap();
                        let (mut policy, policy_calls) = tp_probe_policy();
                        let reward_calls =
                            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
                        let reward = CoordinatedTpReward {
                            rank,
                            mode,
                            calls: std::sync::Arc::clone(&reward_calls),
                        };
                        let result = trainer.train_tensor_parallel(
                            &mut policy,
                            &reward,
                            &TpProbeCodec,
                            &[Sample::new("prompt", ())],
                            &tp_comm,
                        );
                        let policy_calls = policy_calls.lock().unwrap().clone();
                        let candidates =
                            std::fs::read_to_string(run.candidates_path()).unwrap_or_default();
                        results.lock().unwrap().push(CoordinatedTpRunResult {
                            rank,
                            result,
                            reward_calls: reward_calls.load(Ordering::Relaxed),
                            policy_calls,
                            candidates,
                        });
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
        let mut results = std::mem::take(&mut *results.lock().unwrap());
        results.sort_by_key(|result| result.rank);
        results
    }

    fn tp_probe_policy() -> (
        TpProbePolicy,
        std::sync::Arc<std::sync::Mutex<TpProbeCalls>>,
    ) {
        let calls = std::sync::Arc::new(std::sync::Mutex::new(TpProbeCalls::default()));
        let logp = Var::from_tensor(&mat(&[&[-0.4], &[-0.6]])).unwrap();
        (
            TpProbePolicy {
                logp,
                enabled: true,
                sharded_backward: true,
                panic_backward_capability: false,
                calls: std::sync::Arc::clone(&calls),
            },
            calls,
        )
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TpSeparatedHookFailure {
        Generate,
        DetachedScoring,
        LiveScoring,
        Backward,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TpSeparatedFailureBehavior {
        CommunicationError,
        Error,
        Panic,
    }

    struct TpSeparatedFailingPolicy {
        inner: TpProbePolicy,
        arm: std::sync::Arc<std::sync::atomic::AtomicBool>,
        failure: TpSeparatedHookFailure,
        behavior: TpSeparatedFailureBehavior,
        fail_this_rank: bool,
        sampler_state: std::cell::RefCell<Vec<u8>>,
    }

    impl TpSeparatedFailingPolicy {
        fn mutate_sampler(&self) {
            self.sampler_state.borrow_mut()[0] += 1;
        }

        fn mutate_learner_state(&self) -> CandleResult<()> {
            self.mutate_sampler();
            let changed = (self.inner.logp.as_tensor() + 0.75)?;
            self.inner.logp.set(&changed)
        }

        fn fail(&self, comm: &dyn Comm) -> CandleResult<()> {
            match self.behavior {
                TpSeparatedFailureBehavior::CommunicationError => {
                    self.arm.store(true, std::sync::atomic::Ordering::SeqCst);
                    crate::tensor_parallel::comm_to_candle(comm.all_reduce_scalar_sum(1.0))
                        .map(|_| ())
                }
                TpSeparatedFailureBehavior::Error => {
                    self.arm.store(true, std::sync::atomic::Ordering::SeqCst);
                    candle_core::bail!("injected opaque tensor-parallel policy error")
                }
                TpSeparatedFailureBehavior::Panic => {
                    self.arm.store(true, std::sync::atomic::Ordering::SeqCst);
                    panic!("injected opaque tensor-parallel policy panic")
                }
            }
        }
    }

    impl Policy for TpSeparatedFailingPolicy {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            self.inner.generate(prompt, cfg)
        }

        fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
            self.inner.token_logprobs(rollout)
        }

        fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
            self.inner.token_logprobs_detached(rollout)
        }

        fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
            self.inner.backward(loss)
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.inner.set_adapter_enabled(enabled);
        }

        fn adapter_enabled(&self) -> bool {
            self.inner.adapter_enabled()
        }

        fn trainable_vars(&self) -> Vec<Var> {
            self.inner.trainable_vars()
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(self.sampler_state.borrow().clone())
        }

        fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
            if state.len() != 1 {
                candle_core::bail!("invalid test sampler state")
            }
            *self.sampler_state.borrow_mut() = state.to_vec();
            Ok(())
        }
    }

    impl TensorParallelPolicy for TpSeparatedFailingPolicy {
        fn supports_sharded_tensor_parallel_backward(&self) -> bool {
            true
        }

        fn generate_at_tensor_parallel_instrumented(
            &mut self,
            prompt: &[u32],
            cfg: &GenConfig,
            global_row_base: u64,
            comm: &dyn Comm,
            telemetry: Option<&mut dyn ModelTelemetryRecorder>,
        ) -> CandleResult<Rollout> {
            let rollout = self.inner.generate_at_tensor_parallel_instrumented(
                prompt,
                cfg,
                global_row_base,
                comm,
                telemetry,
            )?;
            if self.fail_this_rank && self.failure == TpSeparatedHookFailure::Generate {
                self.mutate_sampler();
                self.fail(comm)?;
            }
            Ok(rollout)
        }

        fn token_logprobs_tensor_parallel(
            &self,
            rollout: &Rollout,
            comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            let logprobs = self.inner.token_logprobs_tensor_parallel(rollout, comm)?;
            if self.fail_this_rank && self.failure == TpSeparatedHookFailure::LiveScoring {
                self.mutate_learner_state()?;
                self.fail(comm)?;
            }
            Ok(logprobs)
        }

        fn token_logprobs_tensor_parallel_detached(
            &self,
            rollout: &Rollout,
            comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            let logprobs = self
                .inner
                .token_logprobs_tensor_parallel_detached(rollout, comm)?;
            if self.fail_this_rank && self.failure == TpSeparatedHookFailure::DetachedScoring {
                self.mutate_learner_state()?;
                self.fail(comm)?;
            }
            Ok(logprobs)
        }

        fn backward_tensor_parallel(
            &self,
            loss: &Tensor,
            comm: &dyn Comm,
        ) -> CandleResult<GradStore> {
            let grads = self.inner.backward_tensor_parallel(loss, comm)?;
            if self.fail_this_rank && self.failure == TpSeparatedHookFailure::Backward {
                self.mutate_learner_state()?;
                self.fail(comm)?;
            }
            Ok(grads)
        }
    }

    struct TpSyncPolicy {
        replicated: Var,
        sharded: Var,
        enabled: bool,
        sampler_state: Vec<u8>,
        fail_restore: bool,
    }

    impl TpSyncPolicy {
        fn new(rank: usize) -> Self {
            Self {
                replicated: Var::zeros((2, 1), DType::F32, &cpu()).unwrap(),
                sharded: Var::zeros((2, 1), DType::F32, &cpu()).unwrap(),
                enabled: true,
                sampler_state: vec![rank as u8],
                fail_restore: false,
            }
        }

        fn fail_sampler_restore_on_resume(rank: usize) -> Self {
            Self {
                fail_restore: true,
                ..Self::new(rank)
            }
        }

        fn logps(&self, comm: &dyn Comm, detached: bool) -> CandleResult<Tensor> {
            let device = self.replicated.as_tensor().device().clone();
            let coeff = if comm.rank() == 0 { 1.0_f32 } else { -3.0_f32 };
            let replicated_coeff = Tensor::from_vec(vec![coeff, coeff], (2, 1), &device)?;
            let shard_mask = if comm.rank() == 0 {
                Tensor::from_vec(vec![1.0_f32, 0.0], (2, 1), &device)?
            } else {
                Tensor::from_vec(vec![0.0_f32, 1.0], (2, 1), &device)?
            };
            let fixed = Tensor::from_vec(vec![-0.4_f32, -0.6], (2, 1), &device)?;
            let logp = fixed
                .add(
                    &self
                        .replicated
                        .as_tensor()
                        .broadcast_mul(&replicated_coeff)?,
                )?
                .add(&self.sharded.as_tensor().broadcast_mul(&shard_mask)?)?;
            if detached {
                Ok(logp.detach())
            } else {
                Ok(logp)
            }
        }

        fn snapshot(&self) -> (Vec<f32>, Vec<f32>) {
            (
                self.replicated
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                self.sharded
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            )
        }
    }

    impl Policy for TpSyncPolicy {
        fn generate(&mut self, _prompt: &[u32], _cfg: &GenConfig) -> CandleResult<Rollout> {
            panic!("train_tensor_parallel must not call Policy::generate")
        }

        fn token_logprobs(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            panic!("train_tensor_parallel must not call Policy::token_logprobs")
        }

        fn token_logprobs_detached(&self, _rollout: &Rollout) -> CandleResult<Tensor> {
            panic!("train_tensor_parallel must not call Policy::token_logprobs_detached")
        }

        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        fn adapter_enabled(&self) -> bool {
            self.enabled
        }

        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.replicated.clone(), self.sharded.clone()]
        }

        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            Ok(self.sampler_state.clone())
        }

        fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
            if self.fail_restore {
                candle_core::bail!("rank-local sampler restore failure");
            }
            self.sampler_state = state.to_vec();
            Ok(())
        }
    }

    impl TensorParallelPolicy for TpSyncPolicy {
        fn supports_sharded_tensor_parallel_backward(&self) -> bool {
            true
        }

        fn generate_at_tensor_parallel_instrumented(
            &mut self,
            prompt: &[u32],
            _cfg: &GenConfig,
            _global_row_base: u64,
            _comm: &dyn Comm,
            _telemetry: Option<&mut dyn ModelTelemetryRecorder>,
        ) -> CandleResult<Rollout> {
            Ok(Rollout {
                token_ids: vec![vec![prompt[0], 1], vec![prompt[0], 2]],
                prompt_len: prompt.len(),
                completion_lens: vec![1, 1],
                rollout_logprobs: Some(vec![vec![-0.5], vec![-0.5]]),
            })
        }

        fn token_logprobs_tensor_parallel(
            &self,
            _rollout: &Rollout,
            comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            self.logps(comm, false)
        }

        fn token_logprobs_tensor_parallel_detached(
            &self,
            _rollout: &Rollout,
            comm: &dyn Comm,
        ) -> CandleResult<Tensor> {
            self.logps(comm, true)
        }

        fn backward_tensor_parallel(
            &self,
            loss: &Tensor,
            _comm: &dyn Comm,
        ) -> CandleResult<GradStore> {
            loss.backward()
        }
    }

    #[test]
    fn train_tensor_parallel_routes_rollout_and_scoring_through_explicit_comm() {
        let tmp = WireTmp::new("tp-trainer-dispatch");
        let run = RunDir::create(&tmp.0, "tp-trainer-dispatch").unwrap();
        let cfg = TrainerConfig {
            steps: 1,
            group_size: 2,
            max_new_tokens: 1,
            lr: 0.0,
            beta: 0.1,
            gpu_memory_probe: true,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let (mut policy, calls) = tp_probe_policy();
        trainer
            .train_tensor_parallel(
                &mut policy,
                &TpProbeReward,
                &TpProbeCodec,
                &[Sample::new("prompt", ())],
                &ProbeTpComm {
                    rank: 0,
                    world_size: 1,
                },
            )
            .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls.generate, 1);
        assert!(calls.live_logp >= 1, "live TP scoring was not used");
        assert!(
            calls.detached_logp >= 2,
            "old/reference TP detached scoring was not used"
        );
        assert!(calls.backward >= 1, "TP backward hook was not used");
        assert_eq!(calls.telemetry_seen, vec![true]);
        assert!(
            calls
                .comms
                .iter()
                .all(|&(rank, world)| (rank, world) == (0, 1)),
            "trainer did not pass the explicit TP communicator to every TP hook: {:?}",
            calls.comms
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_ledger_collector_comm_failure_is_terminal_without_later_collectives() {
        let tmp = WireTmp::new("tp-ledger-collector-terminal-comm");
        let ledger_root = tmp.0.join("ledger");
        for (behavior_name, behavior) in [
            (
                "communication-error",
                TpSeparatedFailureBehavior::CommunicationError,
            ),
            ("error", TpSeparatedFailureBehavior::Error),
            ("panic", TpSeparatedFailureBehavior::Panic),
        ] {
            let outcomes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            std::thread::scope(|scope| {
                let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                    2,
                    std::time::Duration::from_millis(500),
                )
                .into_iter()
                .enumerate()
                .map(|(rank, inner_comm)| {
                    let root = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let outcomes = std::sync::Arc::clone(&outcomes);
                    scope.spawn(move || {
                        let run =
                            RunDir::create(&root, format!("{behavior_name}-rank-{rank}")).unwrap();
                        let mut trainer = Trainer::new(
                            TrainerConfig {
                                steps: 1,
                                group_size: 2,
                                max_new_tokens: 1,
                                ..TrainerConfig::default()
                            },
                            &run,
                        )
                        .unwrap();
                        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                        let state = std::sync::Arc::new(ArmedCollectiveFailureState::new(0));
                        let comm = FailAfterArmComm {
                            inner: inner_comm,
                            armed: std::sync::Arc::clone(&armed),
                            state: std::sync::Arc::clone(&state),
                        };
                        let (inner, _) = tp_probe_policy();
                        let mut policy = TpSeparatedFailingPolicy {
                            inner,
                            arm: armed,
                            failure: TpSeparatedHookFailure::Generate,
                            behavior,
                            fail_this_rank: true,
                            sampler_state: std::cell::RefCell::new(vec![17]),
                        };
                        let sampler_before = policy.sampler_state().unwrap();
                        let error = trainer
                            .collect_rollout_ledger_step_tensor_parallel(
                                0,
                                &mut policy,
                                &TpProbeReward,
                                &TpProbeCodec,
                                &[Sample::new("prompt", ())],
                                &ledger_root,
                                &format!("{:064x}", 131),
                                None,
                                &comm,
                            )
                            .unwrap_err();
                        outcomes.lock().unwrap().push((
                            rank,
                            error,
                            sampler_before,
                            policy.sampler_state().unwrap(),
                            state.failed.load(std::sync::atomic::Ordering::SeqCst),
                            state
                                .calls_after_failure
                                .load(std::sync::atomic::Ordering::SeqCst),
                        ));
                    })
                })
                .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
            });

            let mut outcomes = std::mem::take(&mut *outcomes.lock().unwrap());
            outcomes.sort_by_key(|outcome| outcome.0);
            for (rank, error, sampler_before, sampler_after, failed_comm, later_calls) in outcomes {
                let expected_detail = match behavior {
                    TpSeparatedFailureBehavior::CommunicationError => {
                        "injected terminal collective-chain failure"
                    }
                    TpSeparatedFailureBehavior::Panic => {
                        "policy hook panicked: injected opaque tensor-parallel policy panic"
                    }
                    TpSeparatedFailureBehavior::Error => {
                        "injected opaque tensor-parallel policy error"
                    }
                };
                assert!(
                    matches!(&error, TrainerError::TensorParallelExecutionTerminal {
                        operation: "rollout generation",
                        detail,
                    } if detail.contains(expected_detail)
                        && detail.contains("local sampler rollback succeeded")),
                    "{behavior_name} rank {rank}: {error:?}"
                );
                assert_eq!(
                    sampler_after, sampler_before,
                    "{behavior_name} rank {rank} sampler"
                );
                assert_eq!(
                    failed_comm,
                    behavior == TpSeparatedFailureBehavior::CommunicationError,
                    "{behavior_name} rank {rank} communicator failure state"
                );
                assert_eq!(
                    later_calls, 0,
                    "{behavior_name} rank {rank} issued a later collective"
                );
            }
        }
        assert!(ledger_root.is_dir());
        assert!(!ledger_root.join("step-00000000000000000000").exists());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_ledger_scoring_and_backward_comm_failures_rollback_locally() {
        let tmp = WireTmp::new("tp-ledger-learner-terminal-comm");
        let ledger_root = tmp.0.join("ledger");
        let policy_sha256 = format!("{:064x}", 137);
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let policy_sha256 = policy_sha256.clone();
                    scope.spawn(move || {
                        let run = RunDir::create(&root, format!("collector-rank-{rank}")).unwrap();
                        let mut trainer = Trainer::new(
                            TrainerConfig {
                                steps: 1,
                                group_size: 2,
                                max_new_tokens: 1,
                                ..TrainerConfig::default()
                            },
                            &run,
                        )
                        .unwrap();
                        let (inner, _) = tp_probe_policy();
                        let mut policy = TpSeparatedFailingPolicy {
                            inner,
                            arm: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                            failure: TpSeparatedHookFailure::Generate,
                            behavior: TpSeparatedFailureBehavior::Error,
                            fail_this_rank: false,
                            sampler_state: std::cell::RefCell::new(vec![17]),
                        };
                        trainer
                            .collect_rollout_ledger_step_tensor_parallel(
                                0,
                                &mut policy,
                                &TpProbeReward,
                                &TpProbeCodec,
                                &[Sample::new("prompt", ())],
                                ledger_root,
                                &policy_sha256,
                                None,
                                &comm,
                            )
                            .unwrap();
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });

        for (behavior_name, behavior) in [
            (
                "communication-error",
                TpSeparatedFailureBehavior::CommunicationError,
            ),
            ("error", TpSeparatedFailureBehavior::Error),
            ("panic", TpSeparatedFailureBehavior::Panic),
        ] {
            for (case, failure, operation) in [
                (
                    "detached",
                    TpSeparatedHookFailure::DetachedScoring,
                    "detached scoring",
                ),
                (
                    "live",
                    TpSeparatedHookFailure::LiveScoring,
                    "differentiable scoring",
                ),
                ("backward", TpSeparatedHookFailure::Backward, "backward"),
            ] {
                let outcomes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
                std::thread::scope(|scope| {
                    let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                        2,
                        std::time::Duration::from_millis(500),
                    )
                    .into_iter()
                    .enumerate()
                    .map(|(rank, inner_comm)| {
                        let root = tmp.0.clone();
                        let ledger_root = ledger_root.clone();
                        let policy_sha256 = policy_sha256.clone();
                        let outcomes = std::sync::Arc::clone(&outcomes);
                        scope.spawn(move || {
                            let run = RunDir::create(
                                &root,
                                format!("{behavior_name}-{case}-learner-rank-{rank}"),
                            )
                            .unwrap();
                            let mut trainer = Trainer::new(
                                TrainerConfig {
                                    steps: 1,
                                    group_size: 2,
                                    max_new_tokens: 1,
                                    ..TrainerConfig::default()
                                },
                                &run,
                            )
                            .unwrap();
                            let armed =
                                std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                            let state = std::sync::Arc::new(ArmedCollectiveFailureState::new(0));
                            let comm = FailAfterArmComm {
                                inner: inner_comm,
                                armed: std::sync::Arc::clone(&armed),
                                state: std::sync::Arc::clone(&state),
                            };
                            let (inner, _) = tp_probe_policy();
                            let mut policy = TpSeparatedFailingPolicy {
                                inner,
                                arm: armed,
                                failure,
                                behavior,
                                fail_this_rank: true,
                                sampler_state: std::cell::RefCell::new(vec![17]),
                            };
                            let vars = policy.trainable_vars();
                            let before = vars[0]
                                .as_tensor()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                            let sampler_before = policy.sampler_state().unwrap();
                            let error = trainer
                                .train_rollout_ledger_step_tensor_parallel(
                                    0,
                                    &mut policy,
                                    ledger_root,
                                    &policy_sha256,
                                    None,
                                    &comm,
                                )
                                .unwrap_err();
                            let after = vars[0]
                                .as_tensor()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                            outcomes.lock().unwrap().push((
                                rank,
                                error,
                                before,
                                after,
                                sampler_before,
                                policy.sampler_state().unwrap(),
                                policy.adapter_enabled(),
                                state.failed.load(std::sync::atomic::Ordering::SeqCst),
                                state
                                    .calls_after_failure
                                    .load(std::sync::atomic::Ordering::SeqCst),
                                crate::telemetry::read_metrics(run.metrics_path())
                                    .unwrap()
                                    .len(),
                            ));
                        })
                    })
                    .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });

                let mut outcomes = std::mem::take(&mut *outcomes.lock().unwrap());
                outcomes.sort_by_key(|outcome| outcome.0);
                for (
                    rank,
                    error,
                    before,
                    after,
                    sampler_before,
                    sampler_after,
                    adapter_enabled,
                    failed_comm,
                    later_calls,
                    metrics_rows,
                ) in outcomes
                {
                    let expected_detail = match behavior {
                        TpSeparatedFailureBehavior::CommunicationError => {
                            "injected terminal collective-chain failure"
                        }
                        TpSeparatedFailureBehavior::Panic => {
                            "policy hook panicked: injected opaque tensor-parallel policy panic"
                        }
                        TpSeparatedFailureBehavior::Error => {
                            "injected opaque tensor-parallel policy error"
                        }
                    };
                    assert!(
                        matches!(&error, TrainerError::TensorParallelExecutionTerminal {
                        operation: actual_operation,
                        detail,
                    } if *actual_operation == operation
                        && detail.contains(expected_detail)
                        && detail.contains("local adapter/optimizer/sampler rollback succeeded")),
                        "{behavior_name} {case} rank {rank}: {error:?}"
                    );
                    assert_eq!(after, before, "{behavior_name} {case} rank {rank} adapter");
                    assert_eq!(
                        sampler_after, sampler_before,
                        "{behavior_name} {case} rank {rank} sampler"
                    );
                    assert!(
                        adapter_enabled,
                        "{behavior_name} {case} rank {rank} adapter flag"
                    );
                    assert_eq!(
                        failed_comm,
                        behavior == TpSeparatedFailureBehavior::CommunicationError,
                        "{behavior_name} {case} rank {rank} communicator failure state"
                    );
                    assert_eq!(
                        later_calls, 0,
                        "{behavior_name} {case} rank {rank} later collectives"
                    );
                    assert_eq!(
                        metrics_rows, 0,
                        "{behavior_name} {case} rank {rank} metrics"
                    );
                }
            }
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_ledger_metrics_failure_rolls_back_every_rank() {
        let tmp = WireTmp::new("tp-ledger-metrics-rollback");
        let ledger_root = tmp.0.join("ledger");
        let policy_sha256 = format!("{:064x}", 139);
        let config = TrainerConfig {
            steps: 1,
            group_size: 2,
            max_new_tokens: 1,
            lr: 0.01,
            max_grad_norm: None,
            ..TrainerConfig::default()
        };
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let policy_sha256 = policy_sha256.clone();
                    let config = config.clone();
                    scope.spawn(move || {
                        let run = RunDir::create(&root, format!("collector-rank-{rank}")).unwrap();
                        let mut trainer = Trainer::new(config, &run).unwrap();
                        let (mut policy, _) = tp_probe_policy();
                        trainer
                            .collect_rollout_ledger_step_tensor_parallel(
                                0,
                                &mut policy,
                                &TpProbeReward,
                                &TpProbeCodec,
                                &[Sample::new("prompt", ())],
                                ledger_root,
                                &policy_sha256,
                                None,
                                &comm,
                            )
                            .unwrap();
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });

        let outcomes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = tmp.0.clone();
                    let ledger_root = ledger_root.clone();
                    let policy_sha256 = policy_sha256.clone();
                    let config = config.clone();
                    let outcomes = std::sync::Arc::clone(&outcomes);
                    scope.spawn(move || {
                        let run = RunDir::create(&root, format!("learner-rank-{rank}")).unwrap();
                        let mut trainer = Trainer::new(config, &run).unwrap();
                        if rank == 0 {
                            trainer.writer.inject_append_failure_once();
                        }
                        let (mut policy, _) = tp_probe_policy();
                        let vars = policy.trainable_vars();
                        let before = vars[0]
                            .as_tensor()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        let sampler_before = policy.sampler_state().unwrap();
                        let error = trainer
                            .train_rollout_ledger_step_tensor_parallel(
                                0,
                                &mut policy,
                                ledger_root,
                                &policy_sha256,
                                None,
                                &comm,
                            )
                            .unwrap_err();
                        let after = vars[0]
                            .as_tensor()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        outcomes.lock().unwrap().push((
                            rank,
                            error,
                            before,
                            after,
                            sampler_before,
                            policy.sampler_state().unwrap(),
                            policy.adapter_enabled(),
                            crate::telemetry::read_metrics(run.metrics_path())
                                .unwrap()
                                .len(),
                        ));
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });

        let mut outcomes = std::mem::take(&mut *outcomes.lock().unwrap());
        outcomes.sort_by_key(|outcome| outcome.0);
        for (
            rank,
            error,
            adapter_before,
            adapter_after,
            sampler_before,
            sampler_after,
            adapter_enabled,
            metrics_rows,
        ) in outcomes
        {
            if rank == 0 {
                assert!(
                    matches!(&error, TrainerError::Telemetry(detail)
                        if detail.to_string().contains("injected metrics append failure")),
                    "rank {rank}: {error:?}"
                );
            } else {
                assert!(
                    matches!(&error, TrainerError::Contract(detail)
                        if detail.contains("metrics append failed on execution rank 0")),
                    "rank {rank}: {error:?}"
                );
            }
            assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
            assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
            assert!(adapter_enabled, "rank {rank} adapter flag");
            assert_eq!(metrics_rows, 0, "rank {rank} metrics");
        }
    }

    #[test]
    fn train_tensor_parallel_rejects_forward_only_sharded_policy_in_lockstep() {
        let tmp = WireTmp::new("tp-trainer-forward-only-reject");
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let root = tmp.0.clone();
                    scope.spawn(move || {
                        let run = RunDir::create(&root, format!("rank-{rank}")).unwrap();
                        let mut trainer = Trainer::new(
                            TrainerConfig {
                                steps: 1,
                                group_size: 2,
                                max_new_tokens: 1,
                                ..TrainerConfig::default()
                            },
                            &run,
                        )
                        .unwrap();
                        let (mut policy, calls) = tp_probe_policy();
                        policy.sharded_backward = false;
                        let err = trainer
                            .train_tensor_parallel(
                                &mut policy,
                                &TpProbeReward,
                                &TpProbeCodec,
                                &[Sample::new("prompt", ())],
                                &comm,
                            )
                            .unwrap_err();
                        assert!(
                            matches!(&err, TrainerError::Contract(msg)
                                if msg.contains("cross-rank backward semantics")),
                            "unexpected rank {rank} error: {err:?}"
                        );
                        let calls = calls.lock().unwrap();
                        assert_eq!(calls.generate, 0);
                        assert_eq!(calls.live_logp, 0);
                        assert_eq!(calls.detached_logp, 0);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_backward_capability_panic_stops_before_unsupported_reduction() {
        let tmp = WireTmp::new("tp-backward-capability-panic");
        let scalar_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let errors = std::thread::scope(|scope| {
            let handles =
                crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                    .into_iter()
                    .map(|inner| {
                        let root = tmp.0.clone();
                        let scalar_calls = std::sync::Arc::clone(&scalar_calls);
                        scope.spawn(move || {
                            let rank = inner.rank();
                            let comm = CountScalarComm {
                                inner,
                                scalar_calls,
                            };
                            let run = RunDir::create(&root, format!("rank-{rank}")).unwrap();
                            let trainer = Trainer::new(TrainerConfig::default(), &run).unwrap();
                            let (mut policy, calls) = tp_probe_policy();
                            policy.panic_backward_capability = rank == 1;
                            let error = trainer
                                .validate_tensor_parallel_backward(&policy, &comm)
                                .unwrap_err()
                                .to_string();
                            (rank, error, calls)
                        })
                    })
                    .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, error, calls) in errors {
            if rank == 1 {
                assert!(
                    error.contains("injected backward capability panic"),
                    "{error}"
                );
            } else {
                assert!(
                    error.contains("capability probe failed on a peer"),
                    "{error}"
                );
            }
            let calls = calls.lock().unwrap();
            assert_eq!(calls.generate, 0);
            assert_eq!(calls.live_logp, 0);
            assert_eq!(calls.detached_logp, 0);
            assert_eq!(calls.backward, 0);
        }
        assert_eq!(
            scalar_calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the unsupported-count reduction ran after capability panic"
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // assertion-heavy two-rank reward/ledger contract
    fn train_tensor_parallel_broadcasts_primary_rewards_and_diagnostics() {
        let results = run_coordinated_tp_reward_case(
            [
                CoordinatedTpRewardMode::Scores([0.0, 2.0]),
                CoordinatedTpRewardMode::Scores([2.0, 0.0]),
            ],
            2,
            0.0,
        );

        assert_eq!(results[0].reward_calls, 1);
        assert_eq!(
            results[1].reward_calls, 0,
            "non-primary TP rank must not invoke RewardFn"
        );
        let root_history = results[0].result.as_ref().unwrap().0.as_slice();
        let peer_history = results[1].result.as_ref().unwrap().0.as_slice();
        assert_eq!(root_history.len(), 1);
        assert_eq!(peer_history.len(), 1);
        assert_eq!(root_history[0].reward_mean, 1.0);
        assert_eq!(root_history[0].reward_std, peer_history[0].reward_std);
        assert_eq!(root_history[0].frac_reward_zero_std, 0.0);
        assert_eq!(peer_history[0].frac_reward_zero_std, 0.0);
        assert!(results[0].policy_calls.live_logp > 0);
        assert!(results[1].policy_calls.live_logp > 0);
        assert!(results[0].candidates.contains("rank0:0"));
        assert!(results[0].candidates.contains("\"scoring_rank\":0"));
        assert!(
            results[1].candidates.is_empty(),
            "non-primary TP rank wrote candidate diagnostics"
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // two coordinated failure variants, same assertions
    fn train_tensor_parallel_coordinates_reward_error_and_count_mismatch() {
        for mode in [
            CoordinatedTpRewardMode::Error,
            CoordinatedTpRewardMode::CountMismatch,
        ] {
            let results = run_coordinated_tp_reward_case(
                [mode, CoordinatedTpRewardMode::Scores([0.0, 2.0])],
                0,
                0.0,
            );
            assert_eq!(results[0].reward_calls, 1);
            assert_eq!(results[1].reward_calls, 0);
            match (mode, results[0].result.as_ref().unwrap_err()) {
                (CoordinatedTpRewardMode::Error, TrainerError::Reward(err)) => {
                    assert!(err.to_string().contains("execution-primary reward failure"));
                }
                (CoordinatedTpRewardMode::CountMismatch, TrainerError::Contract(msg)) => {
                    assert!(msg.contains("returned 1 rewards for 2 completions"));
                }
                (_, err) => panic!("unexpected primary reward error: {err:?}"),
            }
            assert!(matches!(
                results[1].result.as_ref().unwrap_err(),
                TrainerError::Contract(msg)
                    if msg.contains("reward evaluation failed on tensor-parallel execution rank 0")
            ));
            for result in &results {
                assert_eq!(result.policy_calls.live_logp, 0);
                assert_eq!(result.policy_calls.detached_logp, 0);
            }
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // paired invalid groups and rank-specific errors
    fn train_tensor_parallel_rejects_nonfinite_rewards_before_candidates_or_kl_scoring() {
        for invalid in [[1.0, f32::NAN], [f32::NEG_INFINITY, f32::INFINITY]] {
            let results = run_coordinated_tp_reward_case(
                [
                    CoordinatedTpRewardMode::Scores(invalid),
                    CoordinatedTpRewardMode::Scores([0.0, 2.0]),
                ],
                2,
                0.1,
            );
            assert_eq!(results[0].reward_calls, 1);
            assert_eq!(results[1].reward_calls, 0);
            assert!(matches!(
                results[0].result.as_ref().unwrap_err(),
                TrainerError::Reward(error) if error.to_string().contains("non-finite")
            ));
            assert!(matches!(
                results[1].result.as_ref().unwrap_err(),
                TrainerError::Contract(message)
                    if message.contains("reward evaluation failed on tensor-parallel execution rank 0")
            ));
            for result in &results {
                assert!(
                    result.candidates.is_empty(),
                    "invalid rewards must not reach candidate publication"
                );
                assert_eq!(result.policy_calls.detached_logp, 0);
                assert_eq!(result.policy_calls.live_logp, 0);
                assert_eq!(result.policy_calls.backward, 0);
            }
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // paired rank-specific panic assertions
    fn train_tensor_parallel_coordinates_primary_reward_panic() {
        let results = run_coordinated_tp_reward_case(
            [
                CoordinatedTpRewardMode::Panic,
                CoordinatedTpRewardMode::Scores([0.0, 2.0]),
            ],
            0,
            0.0,
        );
        assert_eq!(results[0].reward_calls, 1);
        assert_eq!(results[1].reward_calls, 0);
        assert!(matches!(
            results[0].result.as_ref().unwrap_err(),
            TrainerError::Contract(msg)
                if msg.contains("reward evaluation panicked: execution-primary reward panic")
        ));
        assert!(matches!(
            results[1].result.as_ref().unwrap_err(),
            TrainerError::Contract(msg)
                if msg.contains("reward evaluation failed on tensor-parallel execution rank 0")
        ));
        for result in &results {
            assert_eq!(result.policy_calls.live_logp, 0);
            assert_eq!(result.policy_calls.detached_logp, 0);
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // paired branch-direction regression
    fn train_tensor_parallel_uses_primary_reward_for_degenerate_live_branch() {
        let degenerate = run_coordinated_tp_reward_case(
            [
                CoordinatedTpRewardMode::Scores([1.0, 1.0]),
                CoordinatedTpRewardMode::Scores([0.0, 2.0]),
            ],
            0,
            0.0,
        );
        for result in &degenerate {
            let history = &result.result.as_ref().unwrap().0;
            assert_eq!(history[0].frac_reward_zero_std, 1.0);
            assert_eq!(result.policy_calls.live_logp, 0);
            assert_eq!(result.policy_calls.detached_logp, 0);
        }

        let live = run_coordinated_tp_reward_case(
            [
                CoordinatedTpRewardMode::Scores([0.0, 2.0]),
                CoordinatedTpRewardMode::Scores([1.0, 1.0]),
            ],
            0,
            0.0,
        );
        for result in &live {
            let history = &result.result.as_ref().unwrap().0;
            assert_eq!(history[0].frac_reward_zero_std, 0.0);
            assert!(result.policy_calls.live_logp > 0);
            assert!(result.policy_calls.detached_logp > 0);
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_world_one_asymmetric_hook_failures_coordinate_over_data_parallel() {
        for behavior in [
            TpSeparatedFailureBehavior::CommunicationError,
            TpSeparatedFailureBehavior::Error,
            TpSeparatedFailureBehavior::Panic,
        ] {
            let tmp = WireTmp::new(match behavior {
                TpSeparatedFailureBehavior::CommunicationError => "tp1-dp2-hook-comm-error",
                TpSeparatedFailureBehavior::Error => "tp1-dp2-hook-error",
                TpSeparatedFailureBehavior::Panic => "tp1-dp2-hook-panic",
            });
            let outcomes = std::thread::scope(|scope| {
                let handles = crate::comm::LocalComm::world_with_timeout(
                    2,
                    std::time::Duration::from_secs(2),
                )
                .into_iter()
                .map(|dp_comm| {
                    let base = tmp.0.clone();
                    scope.spawn(move || {
                        let rank = dp_comm.rank();
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let mut trainer = Trainer::with_comm(
                            TrainerConfig {
                                steps: 1,
                                group_size: 2,
                                max_new_tokens: 1,
                                lr: 0.0,
                                ..TrainerConfig::default()
                            },
                            &run,
                            dp_comm,
                        )
                        .unwrap();
                        let (inner, _) = tp_probe_policy();
                        let armed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                        let state = std::sync::Arc::new(ArmedCollectiveFailureState::new(0));
                        let tp_comm = FailAfterArmComm {
                            inner: SoloComm,
                            armed: std::sync::Arc::clone(&armed),
                            state,
                        };
                        let mut policy = TpSeparatedFailingPolicy {
                            inner,
                            arm: armed,
                            failure: TpSeparatedHookFailure::Generate,
                            behavior,
                            fail_this_rank: rank == 1,
                            sampler_state: std::cell::RefCell::new(vec![17]),
                        };
                        (
                            rank,
                            trainer
                                .train_tensor_parallel(
                                    &mut policy,
                                    &TpProbeReward,
                                    &TpProbeCodec,
                                    &[Sample::new("prompt", ())],
                                    &tp_comm,
                                )
                                .unwrap_err()
                                .to_string(),
                        )
                    })
                })
                .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });

            for (rank, error) in outcomes {
                if rank == 0 {
                    assert!(
                        error.contains("rollout generation result failed on a peer rank"),
                        "behavior={behavior:?}: {error}"
                    );
                } else {
                    let expected = match behavior {
                        TpSeparatedFailureBehavior::CommunicationError => {
                            "injected terminal collective-chain failure"
                        }
                        TpSeparatedFailureBehavior::Error => {
                            "injected opaque tensor-parallel policy error"
                        }
                        TpSeparatedFailureBehavior::Panic => {
                            "rollout generation policy hook panicked"
                        }
                    };
                    assert!(error.contains(expected), "behavior={behavior:?}: {error}");
                }
            }
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn tensor_parallel_world_one_detached_scoring_failures_coordinate_over_data_parallel() {
        for behavior in [
            TpSeparatedFailureBehavior::Error,
            TpSeparatedFailureBehavior::Panic,
        ] {
            let tmp = WireTmp::new(match behavior {
                TpSeparatedFailureBehavior::Error => "tp1-dp2-detached-error",
                TpSeparatedFailureBehavior::Panic => "tp1-dp2-detached-panic",
                TpSeparatedFailureBehavior::CommunicationError => unreachable!(),
            });
            let outcomes = std::thread::scope(|scope| {
                let handles = crate::comm::LocalComm::world_with_timeout(
                    2,
                    std::time::Duration::from_millis(500),
                )
                .into_iter()
                .map(|dp_comm| {
                    let base = tmp.0.clone();
                    scope.spawn(move || {
                        let rank = dp_comm.rank();
                        let run = RunDir::create(&base, format!("rank-{rank}")).unwrap();
                        let mut trainer = Trainer::with_comm(
                            TrainerConfig {
                                steps: 1,
                                group_size: 2,
                                max_new_tokens: 1,
                                lr: 0.0,
                                ..TrainerConfig::default()
                            },
                            &run,
                            dp_comm,
                        )
                        .unwrap();
                        let (inner, _) = tp_probe_policy();
                        let mut policy = TpSeparatedFailingPolicy {
                            inner,
                            arm: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                            failure: TpSeparatedHookFailure::DetachedScoring,
                            behavior,
                            fail_this_rank: rank == 1,
                            sampler_state: std::cell::RefCell::new(vec![17]),
                        };
                        let before_adapter =
                            policy.inner.logp.as_tensor().to_vec2::<f32>().unwrap();
                        let before_sampler = policy.sampler_state().unwrap();
                        let error = trainer
                            .train_tensor_parallel(
                                &mut policy,
                                &TpProbeReward,
                                &TpProbeCodec,
                                &[Sample::new("prompt", ())],
                                &SoloComm,
                            )
                            .unwrap_err()
                            .to_string();
                        (
                            rank,
                            error,
                            before_adapter,
                            policy.inner.logp.as_tensor().to_vec2::<f32>().unwrap(),
                            before_sampler,
                            policy.sampler_state().unwrap(),
                            policy.adapter_enabled(),
                        )
                    })
                })
                .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });

            for (
                rank,
                error,
                adapter_before,
                adapter_after,
                sampler_before,
                sampler_after,
                adapter_enabled,
            ) in outcomes
            {
                if rank == 0 {
                    assert!(
                        error.contains("rollout group learner materialization failed on a peer"),
                        "behavior={behavior:?}: {error}"
                    );
                } else {
                    let expected = match behavior {
                        TpSeparatedFailureBehavior::Error => {
                            "injected opaque tensor-parallel policy error"
                        }
                        TpSeparatedFailureBehavior::Panic => {
                            "detached scoring policy hook panicked"
                        }
                        TpSeparatedFailureBehavior::CommunicationError => unreachable!(),
                    };
                    assert!(error.contains(expected), "behavior={behavior:?}: {error}");
                }
                assert_eq!(
                    adapter_after, adapter_before,
                    "rank {rank} adapter rollback"
                );
                assert_eq!(
                    sampler_after, sampler_before,
                    "rank {rank} sampler rollback"
                );
                assert!(adapter_enabled, "rank {rank} adapter-mode rollback");
            }
        }
    }

    #[test]
    fn train_tensor_parallel_rejects_simultaneous_sharded_dp_and_tp() {
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, dp_comm)| {
                    scope.spawn(move || {
                        let tmp = WireTmp::new(&format!("tp-trainer-dp-tp-guard-{rank}"));
                        let run = RunDir::create(&tmp.0, format!("tp-trainer-dp-tp-rank-{rank}"))
                            .unwrap();
                        let cfg = TrainerConfig {
                            steps: 1,
                            group_size: 2,
                            max_new_tokens: 1,
                            lr: 0.0,
                            ..TrainerConfig::default()
                        };
                        let mut trainer = Trainer::with_comm(cfg, &run, dp_comm).unwrap();
                        let (mut policy, calls) = tp_probe_policy();
                        let err = trainer
                            .train_tensor_parallel(
                                &mut policy,
                                &TpProbeReward,
                                &TpProbeCodec,
                                &[Sample::new("prompt", ())],
                                &ProbeTpComm {
                                    rank,
                                    world_size: 2,
                                },
                            )
                            .unwrap_err();
                        assert!(
                            matches!(&err, TrainerError::Contract(msg)
                                if msg.contains("simultaneous sharded data-parallel")),
                            "unexpected error on rank {rank}: {err:?}"
                        );
                        let calls = calls.lock().unwrap();
                        assert_eq!(calls.generate, 0, "rank {rank} reached TP rollout");
                        assert_eq!(calls.live_logp, 0, "rank {rank} reached live TP scoring");
                        assert_eq!(
                            calls.detached_logp, 0,
                            "rank {rank} reached detached TP scoring"
                        );
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // ordered two-rank side-effect + checkpoint contract
    fn train_tensor_parallel_two_rank_syncs_grads_and_writes_rank0_checkpoint() {
        let tmp = WireTmp::new("tp-trainer-grad-sync");
        let shared_checkpoints = tmp.0.join("shared-checkpoints");
        std::fs::create_dir_all(&shared_checkpoints).unwrap();
        let results = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, tp_comm)| {
                    let root = tmp.0.clone();
                    let shared_checkpoints = shared_checkpoints.clone();
                    let results = std::sync::Arc::clone(&results);
                    scope.spawn(move || {
                        let run =
                            RunDir::create(&root, format!("tp-trainer-sync-rank-{rank}")).unwrap();
                        let cfg = TrainerConfig {
                            steps: 1,
                            group_size: 2,
                            max_new_tokens: 1,
                            lr: 0.01,
                            max_grad_norm: None,
                            checkpoint_every: Some(1),
                            candidate_log_top_k: 1,
                            ..TrainerConfig::default()
                        };
                        let mut trainer = Trainer::new(cfg, &run)
                            .unwrap()
                            .with_checkpoints_dir(shared_checkpoints);
                        let mut policy = TpSyncPolicy::new(rank);
                        let (history, stop) = trainer
                            .train_tensor_parallel(
                                &mut policy,
                                &TpProbeReward,
                                &TpProbeCodec,
                                &[Sample::new("prompt", ())],
                                &tp_comm,
                            )
                            .unwrap();
                        assert_eq!(stop, RunStop::Completed);
                        assert_eq!(history.len(), 1);
                        let metrics_rows =
                            crate::telemetry::read_metrics(run.metrics_path()).unwrap();
                        let candidate_bytes =
                            std::fs::read_to_string(run.candidates_path()).unwrap_or_default();
                        results.lock().unwrap().push((
                            rank,
                            policy.snapshot(),
                            metrics_rows.len(),
                            candidate_bytes.lines().count(),
                        ));
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });

        let mut results = std::sync::Arc::try_unwrap(results)
            .unwrap()
            .into_inner()
            .unwrap();
        results.sort_by_key(|(rank, _, _, _)| *rank);
        let (_, rank0, rank0_metrics, rank0_candidates) = &results[0];
        let (_, rank1, rank1_metrics, rank1_candidates) = &results[1];
        assert_eq!(*rank0_metrics, 1);
        assert_eq!(
            *rank1_metrics, 0,
            "non-primary TP ranks must not duplicate metrics side effects"
        );
        assert_eq!(*rank0_candidates, 1);
        assert_eq!(
            *rank1_candidates, 0,
            "non-primary TP ranks must not duplicate candidate side effects"
        );
        for (a, b) in rank0.0.iter().zip(&rank1.0) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6, max_relative = 1e-5);
        }
        for (a, b) in rank0.1.iter().zip(&rank1.1) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6, max_relative = 1e-5);
        }
        assert!(
            rank0.1.iter().all(|v| v.abs() > 0.0),
            "all logical sharded adapter rows must be updated on every TP rank: {:?}",
            rank0.1
        );

        let checkpoint_vars = vec![
            Var::zeros((2, 1), DType::F32, &cpu()).unwrap(),
            Var::zeros((2, 1), DType::F32, &cpu()).unwrap(),
        ];
        let loaded = crate::checkpoint::load_checkpoint(
            tmp.0.join("shared-checkpoints/step-1"),
            &checkpoint_vars,
        )
        .unwrap();
        assert_eq!(loaded.step, 1);
        assert_eq!(
            loaded.sampler_state.as_deref(),
            Some([0_u8].as_slice()),
            "TP rank 0 must own the checkpoint side effect"
        );
        let ckpt_replicated = checkpoint_vars[0]
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let ckpt_sharded = checkpoint_vars[1]
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in ckpt_replicated.iter().zip(&rank0.0) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6, max_relative = 1e-5);
        }
        for (a, b) in ckpt_sharded.iter().zip(&rank0.1) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6, max_relative = 1e-5);
        }
    }

    fn zero_optimizer_state(vars: &[Var]) -> OptimizerState {
        OptimizerState {
            step_t: 0,
            first_moments: vars
                .iter()
                .map(|v| {
                    Tensor::zeros(v.as_tensor().dims(), v.as_tensor().dtype(), &cpu()).unwrap()
                })
                .collect(),
            second_moments: vars
                .iter()
                .map(|v| {
                    Tensor::zeros(v.as_tensor().dims(), v.as_tensor().dtype(), &cpu()).unwrap()
                })
                .collect(),
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn resume_latest_tensor_parallel_load_failure_aborts_world_before_training() {
        struct PanicReward;
        impl RewardFn for PanicReward {
            type Target = ();

            fn reward(&self, _sample: &Sample<()>, _completion: &str) -> Result<f32, RewardError> {
                panic!("resume load failure must abort before rollout reward")
            }
        }

        let tmp = WireTmp::new("tp-trainer-resume-load-failure");
        let shared_checkpoints = tmp.0.join("shared-checkpoints");
        let seed_policy = TpSyncPolicy::new(0);
        let seed_vars = seed_policy.trainable_vars();
        crate::checkpoint::save_checkpoint(
            shared_checkpoints.join("step-1"),
            &seed_vars,
            &zero_optimizer_state(&seed_vars),
            &[7_u8],
            1,
            None,
        )
        .unwrap();
        let mut results = Vec::new();

        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_millis(250),
            )
            .into_iter()
            .enumerate()
            .map(|(rank, tp_comm)| {
                let root = tmp.0.clone();
                let shared_checkpoints = shared_checkpoints.clone();
                scope.spawn(move || {
                    let run =
                        RunDir::create(&root, format!("tp-resume-load-fail-rank-{rank}")).unwrap();
                    let cfg = TrainerConfig {
                        steps: 2,
                        group_size: 2,
                        max_new_tokens: 1,
                        lr: 0.01,
                        ..TrainerConfig::default()
                    };
                    let mut trainer = Trainer::new(cfg, &run)
                        .unwrap()
                        .with_checkpoints_dir(shared_checkpoints);
                    let mut policy = if rank == 0 {
                        TpSyncPolicy::fail_sampler_restore_on_resume(rank)
                    } else {
                        TpSyncPolicy::new(rank)
                    };
                    let err = trainer
                        .resume_latest_tensor_parallel(
                            &mut policy,
                            &PanicReward,
                            &TpProbeCodec,
                            &[Sample::new("prompt", ())],
                            &tp_comm,
                        )
                        .unwrap_err();
                    (rank, err)
                })
            })
            .collect();
            for handle in handles {
                results.push(handle.join().unwrap());
            }
        });

        results.sort_by_key(|(rank, _)| *rank);
        for (rank, err) in &results {
            if *rank == 0 {
                assert!(
                    matches!(err, TrainerError::Candle(e)
                        if e.to_string().contains("rank-local sampler restore failure")),
                    "rank 0 should return the local restore error, got {err:?}"
                );
            } else {
                assert!(
                    matches!(err, TrainerError::Contract(msg)
                        if msg.contains("checkpoint load/restore failed on a peer rank")),
                    "peer rank should abort in lockstep, got {err:?}"
                );
            }
        }
    }

    fn assert_primary_telemetry_peer_contract(rank: usize, err: &TrainerError, peer_msg: &str) {
        if rank == 0 {
            assert!(
                matches!(err, TrainerError::Telemetry(_)),
                "rank 0 should return the local telemetry error, got {err:?}"
            );
        } else {
            assert!(
                matches!(err, TrainerError::Contract(msg) if msg.contains(peer_msg)),
                "peer rank should abort in lockstep, got {err:?}"
            );
        }
    }

    fn assert_primary_checkpoint_peer_contract(rank: usize, err: &TrainerError, peer_msg: &str) {
        if rank == 0 {
            assert!(
                matches!(err, TrainerError::Checkpoint(_)),
                "rank 0 should return the local checkpoint error, got {err:?}"
            );
        } else {
            assert!(
                matches!(err, TrainerError::Contract(msg) if msg.contains(peer_msg)),
                "peer rank should abort in lockstep, got {err:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn train_tensor_parallel_candidate_write_failure_aborts_world_before_scoring() {
        let tmp = WireTmp::new("tp-trainer-candidate-side-effect-failure");
        let mut results = Vec::new();

        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_millis(250),
            )
            .into_iter()
            .enumerate()
            .map(|(rank, tp_comm)| {
                let root = tmp.0.clone();
                scope.spawn(move || {
                    let run =
                        RunDir::create(&root, format!("tp-candidate-fail-rank-{rank}")).unwrap();
                    if rank == 0 {
                        std::os::unix::fs::symlink("/dev/full", run.candidates_path()).unwrap();
                    }
                    let cfg = TrainerConfig {
                        steps: 1,
                        group_size: 2,
                        max_new_tokens: 1,
                        lr: 0.01,
                        candidate_log_top_k: 1,
                        ..TrainerConfig::default()
                    };
                    let mut trainer = Trainer::new(cfg, &run).unwrap();
                    let (mut policy, calls) = tp_probe_policy();
                    let err = trainer
                        .train_tensor_parallel(
                            &mut policy,
                            &TpProbeReward,
                            &TpProbeCodec,
                            &[Sample::new("prompt", ())],
                            &tp_comm,
                        )
                        .unwrap_err();
                    let calls = calls.lock().unwrap().clone();
                    (rank, err, calls)
                })
            })
            .collect();
            for handle in handles {
                results.push(handle.join().unwrap());
            }
        });

        results.sort_by_key(|(rank, _, _)| *rank);
        for (rank, err, calls) in &results {
            assert_primary_telemetry_peer_contract(
                *rank,
                err,
                "candidate record write failed on a peer rank",
            );
            assert_eq!(calls.generate, 1, "rank {rank} should reach rollout");
            assert_eq!(
                calls.detached_logp, 0,
                "rank {rank} should stop before detached TP scoring"
            );
            assert_eq!(
                calls.live_logp, 0,
                "rank {rank} should stop before live TP scoring"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn train_tensor_parallel_metrics_write_failure_aborts_world_after_window() {
        let tmp = WireTmp::new("tp-trainer-metrics-side-effect-failure");
        let mut results = Vec::new();

        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_millis(250),
            )
            .into_iter()
            .enumerate()
            .map(|(rank, tp_comm)| {
                let root = tmp.0.clone();
                scope.spawn(move || {
                    let run =
                        RunDir::create(&root, format!("tp-metrics-fail-rank-{rank}")).unwrap();
                    if rank == 0 {
                        std::os::unix::fs::symlink("/dev/full", run.metrics_path()).unwrap();
                    }
                    let cfg = TrainerConfig {
                        steps: 1,
                        group_size: 2,
                        max_new_tokens: 1,
                        lr: 0.01,
                        max_grad_norm: None,
                        ..TrainerConfig::default()
                    };
                    let mut trainer = Trainer::new(cfg, &run).unwrap();
                    let mut policy = TpSyncPolicy::new(rank);
                    let err = trainer
                        .train_tensor_parallel(
                            &mut policy,
                            &TpProbeReward,
                            &TpProbeCodec,
                            &[Sample::new("prompt", ())],
                            &tp_comm,
                        )
                        .unwrap_err();
                    (rank, err)
                })
            })
            .collect();
            for handle in handles {
                results.push(handle.join().unwrap());
            }
        });

        results.sort_by_key(|(rank, _)| *rank);
        for (rank, err) in &results {
            assert_primary_telemetry_peer_contract(
                *rank,
                err,
                "metrics write failed on a peer rank",
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn train_tensor_parallel_checkpoint_write_failure_aborts_world_after_window() {
        let tmp = WireTmp::new("tp-trainer-checkpoint-side-effect-failure");
        let bad_checkpoints = tmp.0.join("not-a-checkpoint-dir");
        std::fs::write(&bad_checkpoints, b"not a directory").unwrap();
        let mut results = Vec::new();

        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world_with_timeout(
                2,
                std::time::Duration::from_millis(250),
            )
            .into_iter()
            .enumerate()
            .map(|(rank, tp_comm)| {
                let root = tmp.0.clone();
                let bad_checkpoints = bad_checkpoints.clone();
                scope.spawn(move || {
                    let run =
                        RunDir::create(&root, format!("tp-checkpoint-fail-rank-{rank}")).unwrap();
                    let cfg = TrainerConfig {
                        steps: 1,
                        group_size: 2,
                        max_new_tokens: 1,
                        lr: 0.01,
                        max_grad_norm: None,
                        checkpoint_every: Some(1),
                        ..TrainerConfig::default()
                    };
                    let mut trainer = Trainer::new(cfg, &run)
                        .unwrap()
                        .with_checkpoints_dir(bad_checkpoints);
                    let mut policy = TpSyncPolicy::new(rank);
                    let err = trainer
                        .train_tensor_parallel(
                            &mut policy,
                            &TpProbeReward,
                            &TpProbeCodec,
                            &[Sample::new("prompt", ())],
                            &tp_comm,
                        )
                        .unwrap_err();
                    (rank, err)
                })
            })
            .collect();
            for handle in handles {
                results.push(handle.join().unwrap());
            }
        });

        results.sort_by_key(|(rank, _)| *rank);
        for (rank, err) in &results {
            assert_primary_checkpoint_peer_contract(
                *rank,
                err,
                "checkpoint write failed on a peer rank",
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // ledger regression checks ordered fields plus metadata
    fn trainer_writes_top_candidate_records_when_enabled() {
        let tmp = WireTmp::new("candidates");
        let run = RunDir::create(&tmp.0, "candidate-run").unwrap();
        let cfg = TrainerConfig {
            steps: 1,
            group_size: 3,
            max_new_tokens: 2,
            lr: 0.0,
            candidate_log_top_k: 2,
            loss_type: LossType::Grpo,
            ..TrainerConfig::default()
        };
        let logp = Var::from_tensor(&mat(&[&[-1.0, -1.0], &[-1.0, -1.0], &[-1.0, -1.0]])).unwrap();
        let mut policy = CandidatePolicy { logp };
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        trainer
            .train(
                &mut policy,
                &CandidateReward,
                &CandidateCodec,
                &[Sample::new("prompt", ())],
            )
            .unwrap();

        let raw = std::fs::read_to_string(run.candidates_path()).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 2);
        let rec0: CandidateRecord = serde_json::from_str(lines[0]).unwrap();
        let rec1: CandidateRecord = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(
            (
                rec0.step,
                rec0.prompt_index,
                rec0.group_index,
                rec0.reward,
                rec0.completion_len_tokens,
                rec0.completion.as_str()
            ),
            (0, 0, 1, 2.0, 2, "9,9")
        );
        assert_eq!(rec0.reward_diagnostic.as_deref(), Some("candidate:9,9"));
        assert_eq!(
            rec0.reward_metadata
                .as_ref()
                .and_then(|m| m.get("completion")),
            Some(&serde_json::json!("9,9"))
        );
        assert_eq!(
            (rec1.group_index, rec1.reward, rec1.completion.as_str()),
            (2, 1.0, "2,2")
        );
        assert_eq!(rec1.reward_diagnostic.as_deref(), Some("candidate:2,2"));
        assert_eq!(
            rec1.reward_metadata
                .as_ref()
                .and_then(|m| m.get("completion")),
            Some(&serde_json::json!("2,2"))
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // explicit asymmetric two-rank abort assertions
    fn data_parallel_nonfinite_reward_aborts_every_rank_before_publication() {
        let tmp = WireTmp::new("dp-nonfinite-reward");
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> =
                crate::comm::LocalComm::world_with_timeout(2, std::time::Duration::from_secs(2))
                    .into_iter()
                    .map(|comm| {
                        let rank = comm.rank();
                        let root = tmp.0.clone();
                        scope.spawn(move || {
                            let run = RunDir::create(&root, format!("rank-{rank}")).unwrap();
                            let cfg = TrainerConfig {
                                steps: 1,
                                group_size: 3,
                                max_new_tokens: 2,
                                beta: 0.0,
                                lr: 0.0,
                                candidate_log_top_k: 1,
                                ..TrainerConfig::default()
                            };
                            let rows: [&[f32]; 3] = [&[-1.0, -1.0], &[-1.0, -1.0], &[-1.0, -1.0]];
                            let logp = Var::from_tensor(&mat(&rows)).unwrap();
                            let mut policy = CandidatePolicy { logp };
                            let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                            let result = trainer.train(
                                &mut policy,
                                &RankedNonFiniteReward { invalid: rank == 0 },
                                &CandidateCodec,
                                &[Sample::new("prompt", ())],
                            );
                            let candidates =
                                std::fs::read_to_string(run.candidates_path()).unwrap_or_default();
                            let metrics =
                                std::fs::read_to_string(run.metrics_path()).unwrap_or_default();
                            (rank, result, candidates, metrics)
                        })
                    })
                    .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, result, candidates, metrics) in results {
            let error = result.unwrap_err();
            if rank == 0 {
                assert!(
                    matches!(
                        &error,
                        TrainerError::Reward(error) if error.to_string().contains("non-finite")
                    ),
                    "unexpected rank-{rank} error: {error:?}"
                );
            } else {
                assert!(
                    matches!(
                        &error,
                        TrainerError::Contract(message)
                            if message.contains("rollout/reward evaluation failed on a peer rank")
                    ),
                    "unexpected rank-{rank} error: {error:?}"
                );
            }
            assert!(candidates.is_empty());
            assert!(metrics.is_empty());
        }
    }

    // ---- completion_lens consumption (length-aware mask / decode / metric) --

    /// A trivial codec that renders token ids as comma-joined decimals, so a decode
    /// test can see exactly which tokens reached it.
    struct JoinCodec;
    impl TokenizerLike for JoinCodec {
        fn encode(&self, _text: &str) -> Vec<u32> {
            Vec::new()
        }
        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
    }

    #[test]
    fn length_mask_rows_marks_padding_and_is_all_ones_at_full_width() {
        // Shorter-than-width lengths => 1.0 up to the length, 0.0 after (the padding).
        let r = Rollout {
            token_ids: vec![vec![0u32; 5]; 3],
            prompt_len: 2,
            completion_lens: vec![1, 3, 2],
            rollout_logprobs: None,
        };
        assert_eq!(
            length_mask_rows(&r, 3),
            vec![
                vec![1.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 0.0],
            ]
        );
        // Full-width lengths => all-ones, bit-identical to the legacy Tensor::ones mask.
        let full = Rollout {
            token_ids: vec![vec![0u32; 5]; 2],
            prompt_len: 2,
            completion_lens: vec![3, 3],
            rollout_logprobs: None,
        };
        assert_eq!(length_mask_rows(&full, 3), vec![vec![1.0; 3]; 2]);
    }

    #[test]
    fn decode_completions_stops_at_completion_len() {
        // seq 0: length 3 (full) => all three completion tokens decoded.
        // seq 1: length 1 => only the first; the trailing pad tokens (7, 7) excluded.
        let r = Rollout {
            token_ids: vec![vec![9, 9, 1, 2, 7], vec![9, 9, 3, 7, 7]],
            prompt_len: 2,
            completion_lens: vec![3, 1],
            rollout_logprobs: None,
        };
        assert_eq!(
            decode_completions(&r, &JoinCodec),
            vec!["1,2,7".to_string(), "3".to_string()]
        );
    }

    #[test]
    fn mean_completion_len_uses_real_completion_lengths() {
        // Real lengths 1 and 3 (not the padded width 3) => mean 2.0.
        let r = Rollout {
            token_ids: vec![vec![0u32; 5], vec![0u32; 5]],
            prompt_len: 2,
            completion_lens: vec![1, 3],
            rollout_logprobs: None,
        };
        assert_relative_eq!(mean_completion_len(&r), 2.0, epsilon = TOL);
        // Empty rollout => 0.0 (no divide-by-zero).
        let e = Rollout {
            token_ids: vec![],
            prompt_len: 0,
            completion_lens: vec![],
            rollout_logprobs: None,
        };
        assert_relative_eq!(mean_completion_len(&e), 0.0, epsilon = TOL);
    }

    #[test]
    fn completion_dims_rejects_misaligned_or_overlong_completion_lens() {
        // Aligned + within-width passes (comp_len here is 3).
        let ok = Rollout {
            token_ids: vec![vec![0u32; 5], vec![0u32; 5]],
            prompt_len: 2,
            completion_lens: vec![1, 3],
            rollout_logprobs: None,
        };
        assert_eq!(completion_dims(&ok).unwrap(), (2, 3));
        // Wrong number of lengths (one length for two sequences).
        let misaligned = Rollout {
            token_ids: vec![vec![0u32; 5], vec![0u32; 5]],
            prompt_len: 2,
            completion_lens: vec![3],
            rollout_logprobs: None,
        };
        assert!(matches!(
            completion_dims(&misaligned),
            Err(TrainerError::Contract(_))
        ));
        // A recorded length past the completion width (4 > 3).
        let overlong = Rollout {
            token_ids: vec![vec![0u32; 5], vec![0u32; 5]],
            prompt_len: 2,
            completion_lens: vec![3, 4],
            rollout_logprobs: None,
        };
        assert!(matches!(
            completion_dims(&overlong),
            Err(TrainerError::Contract(_))
        ));
    }

    // ---- R2: rollout-logprob capture, ratio telemetry, TIS -------------------

    #[test]
    fn completion_dims_rejects_a_misaligned_rollout_logprob_capture() {
        // Aligned capture (row i has completion_lens[i] entries) passes.
        let mut r = Rollout {
            token_ids: vec![vec![0u32; 5], vec![0u32; 5]],
            prompt_len: 2,
            completion_lens: vec![3, 1],
            rollout_logprobs: Some(vec![vec![-0.1, -0.2, -0.3], vec![-0.4]]),
        };
        assert_eq!(completion_dims(&r).unwrap(), (2, 3));
        // A row with the wrong entry count (2 for completion_len 1) is malformed:
        // it would pair ratios with the wrong tokens.
        r.rollout_logprobs = Some(vec![vec![-0.1, -0.2, -0.3], vec![-0.4, -0.5]]);
        assert!(matches!(
            completion_dims(&r),
            Err(TrainerError::Contract(_))
        ));
        // A capture with the wrong row count is malformed too.
        r.rollout_logprobs = Some(vec![vec![-0.1, -0.2, -0.3]]);
        assert!(matches!(
            completion_dims(&r),
            Err(TrainerError::Contract(_))
        ));
    }

    #[test]
    fn validate_rejects_bad_tis_settings() {
        // A sub-1 cap would down-weight exactly on-policy tokens.
        for bad in [0.5, 0.0, -1.0, f64::NAN, f64::INFINITY] {
            let cfg = TrainerConfig {
                tis_imp_ratio_cap: bad,
                ..TrainerConfig::default()
            };
            assert!(cfg.validate().is_err(), "cap {bad} must be rejected");
        }
        // TIS is token-level: combining it with GSPO sequence-level IS is rejected.
        let cfg = TrainerConfig {
            tis: true,
            importance_sampling_level: ImportanceSamplingLevel::Sequence,
            ..TrainerConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(TrainerError::InvalidConfig(_))
        ));
        // The supported combination validates.
        let ok = TrainerConfig {
            tis: true,
            tis_imp_ratio_cap: 2.0,
            ..TrainerConfig::default()
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn r2_config_fields_default_for_old_configs_and_roundtrip() {
        // A pre-R2 config.json fills tis (off) and the cap (2.0) from serde.
        let cfg: TrainerConfig = serde_json::from_str(OLD_CONFIG_JSON).unwrap();
        assert!(!cfg.tis);
        assert_eq!(cfg.tis_imp_ratio_cap, 2.0);
        // Non-default values survive a JSON round-trip.
        let on = TrainerConfig {
            tis: true,
            tis_imp_ratio_cap: 3.5,
            ..TrainerConfig::default()
        };
        let back: TrainerConfig =
            serde_json::from_str(&serde_json::to_string(&on).unwrap()).unwrap();
        assert!(back.tis);
        assert_eq!(back.tis_imp_ratio_cap, 3.5);
    }

    /// The crafted capture fixture the ratio-stats tests share: 2 sequences,
    /// completion width 3, row 1 EOS-stopped after 2 draws (so its position 2 is
    /// padding), one ratio above the cap C = 2. Returns
    /// `(rollout, logp_old, mask_rows)`; the per-token ratios
    /// `exp(logp_old − logp_rollout)` are row 0 `[1, e^0.5, e^-1]`, row 1
    /// `[e^0.7, 1, (padding)]`.
    fn ratio_fixture() -> (Rollout, Tensor, Vec<Vec<f64>>) {
        let logp_old = mat(&[&[-0.5, -1.0, -2.0], &[-0.3, -2.3, -0.4]]);
        let rollout = Rollout {
            token_ids: vec![vec![9, 9, 1, 2, 3], vec![9, 9, 4, 5, 5]],
            prompt_len: 2,
            completion_lens: vec![3, 2],
            rollout_logprobs: Some(vec![vec![-0.5, -1.5, -1.0], vec![-1.0, -2.3]]),
        };
        // Mask: row 0 fully live; row 1 live on its 2 real tokens.
        let mask_rows = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 0.0]];
        (rollout, logp_old, mask_rows)
    }

    /// Hand-computed ratio stats for the crafted capture, exercising the masked
    /// accounting, the padding skip (a short captured row), the cap count, and
    /// the log-ratio drift accumulator.
    #[test]
    fn rollout_ratio_stats_match_a_hand_computed_reference() {
        let (rollout, logp_old, mask_rows) = ratio_fixture();
        let (stats, _) = rollout_ratio_and_tis(&rollout, &logp_old, &mask_rows, 2.0, true).unwrap();
        let stats = stats.expect("capture present, masked tokens > 0");
        let r01 = 0.5f64.exp();
        let r02 = (-1.0f64).exp();
        let r10 = 0.7f64.exp(); // ~2.0138 -> above the cap C = 2.0
        assert_eq!(stats.tokens, 5);
        assert_relative_eq!(stats.sum, 1.0 + r01 + r02 + r10 + 1.0, max_relative = 1e-6);
        // The drift accumulator: Σ log-ratios = 0 + 0.5 − 1 + 0.7 + 0 = 0.2.
        assert_relative_eq!(stats.log_sum, 0.2, epsilon = 1e-6);
        assert_relative_eq!(stats.max, r10, max_relative = 1e-6);
        assert_eq!(stats.capped, 1, "exactly one ratio exceeds C = 2");
    }

    #[test]
    fn rollout_ratio_and_tis_rejects_a_misshapen_score_tensor() {
        // The score tensor is the policy's own output and is otherwise
        // unvalidated: a wrong-shaped token_logprobs must become a loud
        // Contract error, not a silent zip-truncation into wrong-token pairs.
        let (rollout, _, mask_rows) = ratio_fixture();
        let wrong_rows = mat(&[&[-0.5, -1.0, -2.0]]); // 1 row for 2 sequences
        assert!(matches!(
            rollout_ratio_and_tis(&rollout, &wrong_rows, &mask_rows, 2.0, false),
            Err(TrainerError::Contract(_))
        ));
        let wrong_cols = mat(&[&[-0.5, -1.0], &[-0.3, -2.3]]); // width 2 for 3
        assert!(matches!(
            rollout_ratio_and_tis(&rollout, &wrong_cols, &mask_rows, 2.0, false),
            Err(TrainerError::Contract(_))
        ));
    }

    /// The TIS weight tensor for the crafted capture: the over-cap ratio
    /// truncates to exactly C; the padding position is a neutral 1.0
    /// (mask-removed downstream).
    #[test]
    fn tis_weights_match_a_hand_computed_reference() {
        let (rollout, logp_old, mask_rows) = ratio_fixture();
        let (_, tis_w) = rollout_ratio_and_tis(&rollout, &logp_old, &mask_rows, 2.0, true).unwrap();
        let w = tis_w.expect("tis on").to_vec2::<f32>().unwrap();
        let want = [
            [1.0, 0.5f64.exp() as f32, (-1.0f64).exp() as f32],
            [2.0, 1.0, 1.0],
        ];
        for (w_row, want_row) in w.iter().zip(&want) {
            for (got, want) in w_row.iter().zip(want_row) {
                assert_relative_eq!(got, want, epsilon = 1e-5, max_relative = 1e-5);
            }
        }
    }

    /// Telemetry-only mode (tis off) still reports stats but builds no weight
    /// tensor; no capture reports nothing at all.
    #[test]
    fn rollout_ratio_and_tis_off_and_no_capture_modes() {
        let (rollout, logp_old, mask_rows) = ratio_fixture();
        let (stats_off, w_off) =
            rollout_ratio_and_tis(&rollout, &logp_old, &mask_rows, 2.0, false).unwrap();
        assert!(w_off.is_none());
        assert_eq!(stats_off.expect("stats still reported").tokens, 5);

        let bare = Rollout {
            rollout_logprobs: None,
            ..rollout
        };
        let (none_stats, none_w) =
            rollout_ratio_and_tis(&bare, &logp_old, &mask_rows, 2.0, false).unwrap();
        assert!(none_stats.is_none() && none_w.is_none());
    }

    #[test]
    fn fold_ratio_stats_aggregates_token_weighted_and_defaults_neutral() {
        let stat = |ratio: Option<RatioStats>| PromptStat {
            rewards: vec![0.0],
            completion_len: 1.0,
            completion_tokens: 1,
            dropped: 0,
            truncated: 0,
            degenerate: false,
            ratio_stats: ratio,
        };
        // Two captured groups (4 + 1 tokens) and one without capture: the means
        // are token-weighted across the captured ones, the max is the overall
        // max, and the capped fraction / token count share the denominator.
        let stats = vec![
            stat(Some(RatioStats {
                sum: 4.4,
                log_sum: 0.2,
                max: 1.3,
                capped: 1,
                tokens: 4,
            })),
            stat(None),
            stat(Some(RatioStats {
                sum: 0.6,
                log_sum: 0.3,
                max: 0.6,
                capped: 0,
                tokens: 1,
            })),
        ];
        let f = fold_ratio_stats(&stats);
        // (4.4 + 0.6) / 5; (0.2 + 0.3) / 5; max; 1 / 5 — exactly representable.
        assert_eq!(
            (f.ratio_mean, f.logratio_mean, f.ratio_max, f.frac_capped),
            (1.0, 0.1, 1.3, 0.2)
        );
        assert_eq!(f.tokens, 5);
        // A window with no captured tokens reports the neutral on-policy values
        // and a ZERO token count — the disambiguator between "measured
        // on-policy" and "telemetry dark".
        let f = fold_ratio_stats(&[stat(None)]);
        assert_eq!(
            (f.ratio_mean, f.logratio_mean, f.ratio_max, f.frac_capped),
            (1.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(f.tokens, 0);
    }

    #[test]
    fn tis_weight_rescales_the_surrogate_and_its_gradient() {
        // At logp == logp_old the ratio is exactly 1 (unclipped), so the
        // per-token surrogate is adv_i and the TIS weight multiplies it
        // verbatim: loss = -mean_seq(mean_tok(w_ij * adv_i)) under Grpo, and
        // d loss / d logp_ij = -w_ij * adv_i / 4 (2 seqs x 2 tokens). This pins
        // the weight into both the VALUE and the GRADIENT — a dropped
        // broadcast_mul fails both, an accidentally differentiated weight
        // fails the gradient.
        let dev = cpu();
        let v = Var::from_tensor(
            &Tensor::from_vec(vec![-0.2f32, -0.5, -0.9, -0.3], (2, 2), &dev).unwrap(),
        )
        .unwrap();
        let old = v.as_tensor().detach();
        let adv = Tensor::from_vec(vec![0.6f32, -0.4], (2, 1), &dev).unwrap();
        let mask = mat(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let w = mat(&[&[1.0, 2.0], &[3.0, 1.0]]);
        let cfg = |tis_w: Option<Tensor>| LossCfg {
            clip_eps_low: 0.2,
            clip_eps_high: 0.2,
            beta: 0.0,
            loss_type: LossType::Grpo,
            is_level: ImportanceSamplingLevel::Token,
            dapo_norm: None,
            tis_w,
        };

        // Weighted loss value: -mean(0.6*(1+2)/2, -0.4*(3+1)/2) = -(0.9 - 0.8)/2.
        let loss = grpo_loss(
            v.as_tensor(),
            &old,
            None,
            &adv,
            &mask,
            &cfg(Some(w.clone())),
        )
        .unwrap();
        assert_relative_eq!(scalar_f32(&loss).unwrap(), -0.05, epsilon = 1e-6);
        // The unweighted loss differs (knob-wiring: the weight is connected).
        let plain = grpo_loss(v.as_tensor(), &old, None, &adv, &mask, &cfg(None)).unwrap();
        assert_relative_eq!(scalar_f32(&plain).unwrap(), -0.1, epsilon = 1e-6);

        // Weighted gradient: -w_ij * adv_i / 4.
        let grads = loss.backward().unwrap();
        let g = grads.get(v.as_tensor()).unwrap().to_vec2::<f32>().unwrap();
        let want = [[-0.15f32, -0.3], [0.3, 0.1]];
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(g[i][j], want[i][j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn padding_columns_are_inert_in_the_grpo_loss() {
        // A length-aware mask whose last column is EOS padding must (a) leave a zero
        // gradient at the padded positions and (b) under the GRPO reduction give the
        // SAME gradient on the real columns as a loss with no padding column at all
        // (the per-sequence denominator is the real-token count in both). Proves the
        // EOS padding is fully inert in the differentiated objective — on the exact
        // production `grpo_loss`.
        let dev = cpu();
        let logp_data = [-0.2f32, -0.5, -0.9, -0.3, -0.7, -0.1]; // 2 rows x 3 cols
        let make = |cols: usize| {
            let mut d = Vec::new();
            for row in 0..2 {
                d.extend_from_slice(&logp_data[row * 3..row * 3 + cols]);
            }
            Var::from_tensor(&Tensor::from_vec(d, (2, cols), &dev).unwrap()).unwrap()
        };
        let adv = Tensor::from_vec(vec![0.6f32, -0.4], (2, 1), &dev).unwrap();
        let grad_of = |v: &Var, mask: &Tensor| -> Vec<Vec<f32>> {
            let old = v.as_tensor().detach();
            grpo_loss(
                v.as_tensor(),
                &old,
                None,
                &adv,
                mask,
                &LossCfg {
                    clip_eps_low: 0.2,
                    clip_eps_high: 0.2,
                    beta: 0.0,
                    loss_type: LossType::Grpo,
                    is_level: ImportanceSamplingLevel::Token,
                    dapo_norm: None,
                    tis_w: None,
                },
            )
            .unwrap()
            .backward()
            .unwrap()
            .get(v.as_tensor())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
        };

        // Padded: 3 columns, the last masked out per row (real length 2).
        let v3 = make(3);
        let grad3 = grad_of(&v3, &mat(&[&[1.0, 1.0, 0.0], &[1.0, 1.0, 0.0]]));
        // (a) the padded column carries no gradient.
        assert_eq!(grad3[0][2], 0.0);
        assert_eq!(grad3[1][2], 0.0);

        // Unpadded reference: 2 real columns, all kept.
        let v2 = make(2);
        let grad2 = grad_of(&v2, &mat(&[&[1.0, 1.0], &[1.0, 1.0]]));
        // (b) the real columns match the no-padding loss exactly.
        for row in 0..2 {
            for col in 0..2 {
                assert_relative_eq!(grad3[row][col], grad2[row][col], epsilon = TOL);
            }
        }
    }

    #[test]
    fn padding_with_an_exp_overflowing_logp_gap_stays_grad_finite() {
        // A masked (EOS-padding) position whose log-prob is wildly below the reference
        // makes the k3 KL's exp(logp_ref - logp) overflow f32 to +inf (argument ~200 >>
        // 88). Without the exp-argument masking in grpo_loss, exp's backward there is
        // `0 * inf = NaN` (the cell is masked so its upstream grad is 0, but exp's local
        // derivative is inf), which would poison the LoRA gradient of an inert padding
        // token and fail the step via the canary. The masked column's gradient must stay
        // EXACTLY zero and finite, and the real column unaffected.
        let dev = cpu();
        // One sequence, 2 columns; column 1 is padding (mask 0) with a huge -logp.
        let logp = Var::from_tensor(&mat(&[&[-0.5, -200.0]])).unwrap();
        let logp_old = logp.as_tensor().detach(); // ratio == 1 everywhere (mu = 1)
        let logp_ref = mat(&[&[-0.5, -0.5]]); // ref - logp at col 1 == 199.5 -> exp overflow
        let adv = Tensor::from_vec(vec![0.7f32], (1, 1), &dev).unwrap();
        let mask = mat(&[&[1.0, 0.0]]);
        let grads = grpo_loss(
            logp.as_tensor(),
            &logp_old,
            Some(&logp_ref),
            &adv,
            &mask,
            &LossCfg {
                clip_eps_low: 0.2,
                clip_eps_high: 0.2,
                beta: 0.1, // beta > 0: the k3 KL term is active
                loss_type: LossType::Grpo,
                is_level: ImportanceSamplingLevel::Token,
                dapo_norm: None,
                tis_w: None,
            },
        )
        .unwrap()
        .backward()
        .unwrap();
        let g = grads
            .get(logp.as_tensor())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(
            g[0][0].is_finite(),
            "real-column gradient went non-finite: {}",
            g[0][0]
        );
        assert_eq!(
            g[0][1], 0.0,
            "padding gradient is not exactly zero (NaN leaked from exp overflow?): {}",
            g[0][1]
        );
    }

    // ---- gradient accumulation (fold across backwards) ---------------------

    #[test]
    fn fold_var_grads_sums_gradients_across_backwards() {
        // Two separate backward passes on a shared Var; folding both must yield the
        // element-wise sum of their gradients — the core grad-accumulation invariant.
        let dev = cpu();
        let x =
            Var::from_tensor(&Tensor::from_vec(vec![2.0f64, 3.0], (2,), &dev).unwrap()).unwrap();
        let vars = vec![x.clone()];
        // loss1 = sum(x^2) -> grad 2x = [4, 6]; loss2 = sum(3x) -> grad [3, 3].
        let g1 = x
            .as_tensor()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let g2 = x
            .as_tensor()
            .affine(3.0, 0.0)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let mut acc: Vec<Option<Tensor>> = vec![None];
        let mut covered = vec![true];
        fold_var_grads(&vars, &g1, &mut acc, &mut covered).unwrap();
        fold_var_grads(&vars, &g2, &mut acc, &mut covered).unwrap();
        assert!(covered[0]);
        let summed = acc[0].as_ref().unwrap().to_vec1::<f64>().unwrap();
        assert_relative_eq!(summed[0], 7.0, epsilon = 1e-9); // 4 + 3
        assert_relative_eq!(summed[1], 9.0, epsilon = 1e-9); // 6 + 3
    }

    #[test]
    fn fold_var_grads_marks_var_absent_from_a_backward_uncovered() {
        // A var that never reached a given loss is absent from that backward's store:
        // fold must flag it uncovered (the silent-skip landmine) so the window's
        // canary aborts, rather than silently treating it as a zero contribution.
        let dev = cpu();
        let x = Var::from_tensor(&Tensor::from_vec(vec![1.0f64], (1,), &dev).unwrap()).unwrap();
        let y = Var::from_tensor(&Tensor::from_vec(vec![1.0f64], (1,), &dev).unwrap()).unwrap();
        let vars = vec![x.clone(), y.clone()];
        // The loss depends only on x, so y is absent from the grad store.
        let g = x
            .as_tensor()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let mut acc: Vec<Option<Tensor>> = vec![None, None];
        let mut covered = vec![true, true];
        fold_var_grads(&vars, &g, &mut acc, &mut covered).unwrap();
        assert!(covered[0] && acc[0].is_some(), "x must be covered");
        assert!(
            !covered[1] && acc[1].is_none(),
            "y must be flagged uncovered"
        );
    }

    #[test]
    fn compact_trainable_grad_store_preserves_absent_vars() {
        let dev = cpu();
        let x = Var::from_tensor(&Tensor::from_vec(vec![2.0f64], (1,), &dev).unwrap()).unwrap();
        let y = Var::from_tensor(&Tensor::from_vec(vec![3.0f64], (1,), &dev).unwrap()).unwrap();
        let raw = x
            .as_tensor()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let compact = compact_trainable_grad_store(&[x.clone(), y.clone()], raw).unwrap();
        assert_eq!(
            compact
                .get(x.as_tensor())
                .unwrap()
                .to_vec1::<f64>()
                .unwrap(),
            vec![4.0]
        );
        assert!(
            compact.get(y.as_tensor()).is_none(),
            "an absent trainable var must remain absent so the canary can fail loud"
        );
    }

    #[test]
    fn combine_into_store_consumes_accumulators_and_omits_uncovered_vars() {
        let dev = cpu();
        let x = Var::from_tensor(&Tensor::from_vec(vec![1.0f64], (1,), &dev).unwrap()).unwrap();
        let y = Var::from_tensor(&Tensor::from_vec(vec![2.0f64], (1,), &dev).unwrap()).unwrap();
        let vars = vec![x.clone(), y.clone()];
        let mut acc = vec![
            Some(Tensor::from_vec(vec![3.0f64], (1,), &dev).unwrap()),
            Some(Tensor::from_vec(vec![4.0f64], (1,), &dev).unwrap()),
        ];
        let covered = vec![true, false];

        let store = empty_grad_store(&vars).unwrap();
        let store = combine_into_store(&vars, store, &mut acc, &covered);

        assert!(
            acc.iter().all(Option::is_none),
            "all accumulator slots are moved out"
        );
        assert_eq!(
            store.get(x.as_tensor()).unwrap().to_vec1::<f64>().unwrap(),
            vec![3.0]
        );
        assert!(
            store.get(y.as_tensor()).is_none(),
            "uncovered vars must stay absent so the canary fails loud"
        );
    }

    #[test]
    fn reduce_epoch_reduces_gradients_across_chunk_boundary() {
        std::thread::scope(|scope| {
            let comms = crate::comm::LocalComm::world(2);
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    scope.spawn(move || {
                        let tmp = WireTmp::new(&format!("chunked-reduce-{rank}"));
                        let run = RunDir::create(&tmp.0, format!("rank-{rank}")).unwrap();
                        let trainer =
                            Trainer::with_comm(TrainerConfig::default(), &run, comm).unwrap();
                        let dev = cpu();
                        let vars: Vec<Var> = (0..(GRAD_REDUCE_CHUNK + 2))
                            .map(|i| {
                                Var::from_tensor(
                                    &Tensor::from_vec(vec![i as f32], (1,), &dev).unwrap(),
                                )
                                .unwrap()
                            })
                            .collect();
                        let mut acc: Vec<Option<Tensor>> = (0..vars.len())
                            .map(|i| {
                                let local = if rank == 0 {
                                    i as f32
                                } else {
                                    100.0 + i as f32
                                };
                                Some(Tensor::from_vec(vec![local], (1,), &dev).unwrap())
                            })
                            .collect();
                        let covered = vec![true; vars.len()];

                        let (kl, clip, uncovered) = trainer
                            .reduce_epoch(&vars, &mut acc, &covered, rank as f32 + 1.0, 4.0, 2.0)
                            .unwrap();
                        let got: Vec<f32> = acc
                            .iter()
                            .map(|slot| slot.as_ref().unwrap().to_vec1::<f32>().unwrap()[0])
                            .collect();
                        (kl, clip, uncovered, got)
                    })
                })
                .collect();

            for h in handles {
                let (kl, clip, uncovered, got) = h.join().unwrap();
                assert_eq!(uncovered, 0.0);
                assert_relative_eq!(kl, 1.5, epsilon = 1e-6);
                assert_relative_eq!(clip, 4.0, epsilon = 1e-6);
                for (i, value) in got.iter().enumerate() {
                    assert_relative_eq!(*value, 100.0 + 2.0 * i as f32, epsilon = 1e-6);
                }
            }
        });
    }

    #[test]
    fn reduce_epoch_reduces_materialized_empty_peer_shard_across_chunk_boundary() {
        std::thread::scope(|scope| {
            let comms = crate::comm::LocalComm::world(2);
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    scope.spawn(move || {
                        let tmp = WireTmp::new(&format!("chunked-empty-shard-{rank}"));
                        let run = RunDir::create(&tmp.0, format!("rank-{rank}")).unwrap();
                        let trainer =
                            Trainer::with_comm(TrainerConfig::default(), &run, comm).unwrap();
                        let dev = cpu();
                        let vars: Vec<Var> = (0..(GRAD_REDUCE_CHUNK + 2))
                            .map(|i| {
                                Var::from_tensor(
                                    &Tensor::from_vec(vec![i as f32], (1,), &dev).unwrap(),
                                )
                                .unwrap()
                            })
                            .collect();
                        let mut acc: Vec<Option<Tensor>> = if rank == 0 {
                            (0..vars.len())
                                .map(|i| {
                                    Some(
                                        Tensor::from_vec(vec![10.0 + i as f32], (1,), &dev)
                                            .unwrap(),
                                    )
                                })
                                .collect()
                        } else {
                            vec![None; vars.len()]
                        };
                        materialize_zero_grad_slots(&vars, &mut acc).unwrap();
                        let covered = vec![true; vars.len()];

                        let (kl, clip, uncovered) = trainer
                            .reduce_epoch(&vars, &mut acc, &covered, rank as f32 + 1.0, 4.0, 2.0)
                            .unwrap();
                        let got: Vec<f32> = acc
                            .iter()
                            .map(|slot| slot.as_ref().unwrap().to_vec1::<f32>().unwrap()[0])
                            .collect();
                        (kl, clip, uncovered, got)
                    })
                })
                .collect();

            for h in handles {
                let (kl, clip, uncovered, got) = h.join().unwrap();
                assert_eq!(uncovered, 0.0);
                assert_relative_eq!(kl, 1.5, epsilon = 1e-6);
                assert_relative_eq!(clip, 4.0, epsilon = 1e-6);
                for (i, value) in got.iter().enumerate() {
                    assert_relative_eq!(*value, 10.0 + i as f32, epsilon = 1e-6);
                }
            }
        });
    }

    #[test]
    fn tensor_parallel_grad_reduce_reconstructs_full_replicated_adapter_grads() {
        std::thread::scope(|scope| {
            let handles: Vec<_> = crate::comm::LocalComm::world(2)
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    scope.spawn(move || {
                        let dev = cpu();
                        let vars = vec![
                            Var::zeros((2,), DType::F32, &dev).unwrap(),
                            Var::zeros((2,), DType::F32, &dev).unwrap(),
                        ];
                        let mut acc = if rank == 0 {
                            vec![
                                Some(Tensor::from_vec(vec![1.0_f32, 2.0], (2,), &dev).unwrap()),
                                Some(Tensor::from_vec(vec![3.0_f32, 0.0], (2,), &dev).unwrap()),
                            ]
                        } else {
                            vec![
                                Some(Tensor::from_vec(vec![10.0_f32, 20.0], (2,), &dev).unwrap()),
                                Some(Tensor::from_vec(vec![0.0_f32, 4.0], (2,), &dev).unwrap()),
                            ]
                        };
                        let covered = vec![true; vars.len()];
                        let uncovered =
                            reduce_accumulated_grads(&comm, &vars, &mut acc, &covered).unwrap();
                        let got: Vec<Vec<f32>> = acc
                            .iter()
                            .map(|slot| {
                                slot.as_ref()
                                    .unwrap()
                                    .flatten_all()
                                    .unwrap()
                                    .to_vec1::<f32>()
                                    .unwrap()
                            })
                            .collect();
                        (uncovered, got)
                    })
                })
                .collect();

            for handle in handles {
                let (uncovered, got) = handle.join().unwrap();
                assert_eq!(uncovered, 0.0);
                assert_eq!(got[0], vec![11.0, 22.0]);
                assert_eq!(got[1], vec![3.0, 4.0]);
            }
        });
    }

    // ---- finite-difference gradcheck of the GRPO loss ----------------------
    //
    // The PLAN's last correctness oracle: numerically verify candle's analytic
    // gradient of `grpo_loss` — the exact loss the trainer back-propagates — w.r.t.
    // the LoRA parameters. We do it on a hermetic, double-precision LoRA-style map
    // (`f64` so central differences are accurate) and pin the *production* loss
    // function, so this cannot drift from what the trainer actually back-propagates.
    // The `logp_old` shifts are chosen so the importance ratios straddle the clip
    // band (both branches of the surrogate `min` are exercised), advantages are
    // mixed-sign and non-zero, one completion column is masked out, and a reference
    // policy drives the k3 KL term — so a single check covers surrogate clipping,
    // KL, and masking under either reduction.
    //
    // Three mask shapes are checked. The **uniform** scenario masks the same trailing
    // column for every row (constant per-row denominator). The **ragged** scenario uses
    // the variable-per-row mask `length_mask_rows` builds from `Rollout::completion_lens`
    // under EOS-aware generation — distinct per-row GRPO denominators, a row whose final
    // column is a real token, and staggered padding. The **all-padding** scenario adds a
    // zero-length row, exercising the GRPO denominator clamp and an entirely-masked row's
    // gradient inertness. Together they numerically pin the gradient of exactly the mask
    // the trainer back-propagates (the defense-in-depth for variable-length loss masking).

    /// Deterministic, rng-free `f64` fill in `[-0.5, 0.5)` — reproducible inputs so
    /// the gradcheck is byte-stable across platforms.
    fn gc_fill(n: usize, seed: u64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                (z % 1000) as f64 / 1000.0 - 0.5
            })
            .collect()
    }

    /// A tiny `f64` LoRA-style map `(A, B) -> logp[G, T]`, mirroring a policy's
    /// `one_hot -> linear -> log_softmax -> gather` token-logprob path:
    /// `logits = x · (W0 + scale · B·A)`, then gather the target log-probs.
    struct GcModel {
        x: Tensor,       // [G, T, V] fixed inputs
        w0: Tensor,      // [V, V] fixed frozen base
        targets: Tensor, // [G, T] fixed target ids
        a: Var,          // [R, V]
        b: Var,          // [V, R]
        scale: f64,      // alpha / rank
    }

    impl GcModel {
        fn logp(&self) -> CandleResult<Tensor> {
            let (g, t, v) = self.x.dims3()?;
            let delta = self.b.as_tensor().matmul(self.a.as_tensor())?; // [V, V]
            let w = self.w0.add(&delta.affine(self.scale, 0.0)?)?; // [V, V]
            let logits = self.x.reshape((g * t, v))?.matmul(&w)?.reshape((g, t, v))?;
            let logp_full = candle_nn::ops::log_softmax(&logits, D::Minus1)?; // [G, T, V]
            let idx = self.targets.unsqueeze(D::Minus1)?; // [G, T, 1]
            logp_full.gather(&idx, D::Minus1)?.squeeze(D::Minus1) // [G, T]
        }
    }

    /// Central finite-difference gradient of `loss_of` w.r.t. every element of
    /// `var`, restoring `var` to its original value before returning. `var` shares
    /// storage with the model's copy, so perturbing it changes what `loss_of` sees.
    fn gc_numeric_grad(var: &Var, dev: &Device, loss_of: &dyn Fn() -> f64, h: f64) -> Vec<f64> {
        let shape = var.as_tensor().shape().clone();
        let orig: Vec<f64> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
        let set = |data: Vec<f64>| {
            var.set(&Tensor::from_vec(data, shape.clone(), dev).unwrap())
                .unwrap();
        };
        let mut grad = vec![0.0; orig.len()];
        for (k, g) in grad.iter_mut().enumerate() {
            let mut up = orig.clone();
            up[k] += h;
            set(up);
            let l_plus = loss_of();
            let mut dn = orig.clone();
            dn[k] -= h;
            set(dn);
            let l_minus = loss_of();
            *g = (l_plus - l_minus) / (2.0 * h);
        }
        set(orig); // restore
        grad
    }

    /// Assert analytic and numeric gradients agree (absolute or relative tolerance).
    fn gc_assert_close(analytic: &[f64], numeric: &[f64], tol: f64, name: &str, ctx: &str) {
        assert_eq!(analytic.len(), numeric.len(), "{name}: length mismatch");
        for (k, (a, n)) in analytic.iter().zip(numeric).enumerate() {
            let diff = (a - n).abs();
            let rel = diff / a.abs().max(1.0);
            assert!(
                diff < tol || rel < tol,
                "gradcheck {name}[{k}] ({ctx}): analytic={a}, numeric={n}, diff={diff}"
            );
        }
    }

    /// Run the gradcheck for one scenario: a `(loss_type, beta)` setting under a
    /// specific per-token `shift` (which sets where importance ratios fall relative
    /// to the clip band), `advantages` column, and loss `mask`. The geometry
    /// `(G, T)` is read from `mask`; `shift` is `[G, T]` and `adv` is `[G, 1]`. Builds
    /// the tiny `f64` `LoRA` map, then asserts candle's analytic gradient of
    /// `grpo_loss` w.r.t. the `LoRA` factors `A` and `B` matches central differences.
    /// `ctx` labels the scenario in assertion messages.
    #[allow(clippy::too_many_arguments)]
    fn run_gradcheck_with(
        loss_type: LossType,
        beta: f64,
        level: ImportanceSamplingLevel,
        eps_high: f64,
        shift: &Tensor,
        adv: &Tensor,
        mask: &Tensor,
        ctx: &str,
    ) {
        let dev = cpu();
        let (g, t) = mask.dims2().unwrap();
        const V: usize = 4;
        const R: usize = 2;
        const EPS: f64 = 0.2;
        const H: f64 = 1e-6;

        let x = Tensor::from_vec(gc_fill(g * t * V, 1), (g, t, V), &dev).unwrap();
        let w0 = Tensor::from_vec(gc_fill(V * V, 2), (V, V), &dev).unwrap();
        // Targets cycle through the vocab (0,1,..,V-1,0,..) — identical to the prior
        // hard-coded `[0,1,2,3,...]` at G=4,T=3,V=4, now derived from the geometry.
        let target_ids: Vec<u32> = (0..g * t).map(|i| (i % V) as u32).collect();
        let targets = Tensor::from_vec(target_ids, (g, t), &dev).unwrap();
        // B starts non-zero so the loss depends non-trivially on *both* factors
        // (a zero-init B — the real LoRA default — would give A a zero gradient).
        let a =
            Var::from_tensor(&Tensor::from_vec(gc_fill(R * V, 3), (R, V), &dev).unwrap()).unwrap();
        let b =
            Var::from_tensor(&Tensor::from_vec(gc_fill(V * R, 4), (V, R), &dev).unwrap()).unwrap();
        let model = GcModel {
            x,
            w0,
            targets,
            a: a.clone(),
            b: b.clone(),
            scale: 1.0,
        };

        let logp0 = model.logp().unwrap().detach();
        let logp_old = logp0.broadcast_sub(shift).unwrap().detach();
        let logp_ref = (beta > 0.0).then(|| logp0.affine(1.0, 0.1).unwrap().detach());

        let cfg = LossCfg {
            clip_eps_low: EPS,
            clip_eps_high: eps_high,
            beta,
            loss_type,
            is_level: level,
            dapo_norm: None,
            tis_w: None,
        };
        let loss_of = || -> f64 {
            let logp = model.logp().unwrap();
            grpo_loss(&logp, &logp_old, logp_ref.as_ref(), adv, mask, &cfg)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        };

        // Analytic gradients (extract to Vec before perturbing anything).
        let logp = model.logp().unwrap();
        let loss = grpo_loss(&logp, &logp_old, logp_ref.as_ref(), adv, mask, &cfg).unwrap();
        let grads = loss.backward().unwrap();
        let ga: Vec<f64> = grads
            .get(a.as_tensor())
            .expect("A in grad store")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let gb: Vec<f64> = grads
            .get(b.as_tensor())
            .expect("B in grad store")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let na = gc_numeric_grad(&a, &dev, &loss_of, H);
        let nb = gc_numeric_grad(&b, &dev, &loss_of, H);

        gc_assert_close(&ga, &na, 1e-5, "A", ctx);
        gc_assert_close(&gb, &nb, 1e-5, "B", ctx);
    }

    /// Gradcheck under the **uniform** mask: the trailing completion column (`T-1`)
    /// is masked out for every row — one masked column, a constant per-row
    /// denominator. Shifts straddle the clip band on both sides; advantages are
    /// mixed-sign.
    fn run_gradcheck(loss_type: LossType, beta: f64) {
        run_gradcheck_level(loss_type, beta, ImportanceSamplingLevel::Token);
    }

    /// Like [`run_gradcheck`] but at an explicit importance-sampling level —
    /// the GSPO-seam oracle (sequence-level ratios are a different gradient).
    fn run_gradcheck_level(loss_type: LossType, beta: f64, level: ImportanceSamplingLevel) {
        let dev = cpu();
        // Per-token shifts: ratio = exp(shift) at the eval point. 0.30 -> 1.35 (above
        // band), -0.30 -> 0.74 (below band), 0.05 -> 1.05 (inside). Column 2 is masked.
        #[rustfmt::skip]
        let shift = Tensor::from_vec(
            vec![0.30, 0.05, 0.10,
                 -0.30, 0.05, 0.10,
                 0.05, 0.30, 0.10,
                 0.05, -0.30, 0.10f64],
            (4, 3), &dev,
        ).unwrap();
        let adv = Tensor::from_vec(vec![0.8, -0.7, 0.5, -0.4f64], (4, 1), &dev).unwrap();
        #[rustfmt::skip]
        let mask = Tensor::from_vec(
            vec![1.0, 1.0, 0.0,
                 1.0, 1.0, 0.0,
                 1.0, 1.0, 0.0,
                 1.0, 1.0, 0.0f64],
            (4, 3), &dev,
        ).unwrap();
        let ctx = format!("uniform-mask, loss_type={loss_type:?}, beta={beta}, level={level:?}");
        run_gradcheck_with(loss_type, beta, level, 0.2, &shift, &adv, &mask, &ctx);
    }

    /// Gradcheck with the DAPO **clip-higher** band (eps 0.2 / 0.28): shifts at
    /// `0.22` put ratios ≈ 1.246 *between* the symmetric edge (1.2) and the
    /// widened edge (1.28), so the asymmetric upper band genuinely changes
    /// which surrogate branch is differentiated — a band swap or an ignored
    /// `eps_high` moves the analytic gradient off the numeric one.
    fn run_gradcheck_clip_higher(loss_type: LossType, beta: f64) {
        let dev = cpu();
        #[rustfmt::skip]
        let shift = Tensor::from_vec(
            vec![0.22, 0.05, 0.10,
                 -0.30, 0.22, 0.10,
                 0.05, 0.30, 0.10,
                 0.22, -0.30, 0.10f64],
            (4, 3), &dev,
        ).unwrap();
        let adv = Tensor::from_vec(vec![0.8, -0.7, 0.5, -0.4f64], (4, 1), &dev).unwrap();
        #[rustfmt::skip]
        let mask = Tensor::from_vec(
            vec![1.0, 1.0, 0.0,
                 1.0, 1.0, 0.0,
                 1.0, 1.0, 0.0,
                 1.0, 1.0, 0.0f64],
            (4, 3), &dev,
        ).unwrap();
        let ctx = format!("clip-higher 0.2/0.28, loss_type={loss_type:?}, beta={beta}");
        run_gradcheck_with(
            loss_type,
            beta,
            ImportanceSamplingLevel::Token,
            0.28,
            &shift,
            &adv,
            &mask,
            &ctx,
        );
    }

    /// Shared driver for a ragged-mask gradcheck. Builds the `[G, T]` loss mask from
    /// `completion_lens` via the production [`length_mask_rows`] (so the gradcheck pins
    /// the same `j < completion_lens[i]` predicate `collect_sample` uses, at `f64` for
    /// accurate central differences), hard-asserts the mask is ragged exactly as
    /// specified (premise guard — else the check could silently degrade to a
    /// uniform/all-ones mask and stop exercising variable lengths), then runs the
    /// analytic-vs-numeric gradcheck. `T` is read from `shift`'s last dim; `shift` is
    /// `[G, T]` and `adv` is `[G, 1]`.
    fn run_gradcheck_ragged_with(
        loss_type: LossType,
        beta: f64,
        completion_lens: Vec<usize>,
        shift: &Tensor,
        adv: &Tensor,
        dev: &Device,
    ) {
        let g = completion_lens.len();
        let t = shift.dims2().unwrap().1;
        let expected: Vec<f64> = completion_lens.iter().map(|&l| l.min(t) as f64).collect();
        let ctx =
            format!("ragged-mask lens={completion_lens:?}, loss_type={loss_type:?}, beta={beta}");
        let rollout = Rollout {
            token_ids: vec![Vec::new(); g],
            prompt_len: 0,
            completion_lens,
            rollout_logprobs: None,
        };
        let mask_rows = length_mask_rows(&rollout, t);
        let mask_data: Vec<f64> = mask_rows.iter().flatten().copied().collect();
        let mask = Tensor::from_vec(mask_data, (g, t), dev).unwrap();
        let row_sums: Vec<f64> = mask.sum(D::Minus1).unwrap().to_vec1().unwrap();
        assert_eq!(row_sums, expected, "ragged mask premise");
        run_gradcheck_with(
            loss_type,
            beta,
            ImportanceSamplingLevel::Token,
            0.2,
            shift,
            adv,
            &mask,
            &ctx,
        );
    }

    /// Gradcheck under a **ragged** mask — variable real-completion lengths per row,
    /// the shape [`length_mask_rows`] builds from [`Rollout::completion_lens`] under
    /// EOS-aware generation. Lengths `[1, 2, 3, 2]` over `T = 3` give per-row GRPO
    /// denominators `1, 2, 3, 2` (the uniform case only ever exercised a constant
    /// denominator of 2), a row (`len = 3`) whose **final** column is a real
    /// gradient-bearing token (every uniform row masked it), and padding columns that
    /// differ per row — so the `keep.where_cond` substitution in `grpo_loss` is
    /// verified not to corrupt the real-position gradient at genuinely staggered
    /// padding. Shifts straddle the clip band at **kept** positions (padding ratios are
    /// forced to 1 and contribute nothing).
    fn run_gradcheck_ragged(loss_type: LossType, beta: f64) {
        let dev = cpu();
        // Shifts placed so each row's clip-band straddle lands on a KEPT column:
        // row0 keeps {0} -> above-band at col0; row1 keeps {0,1} -> below-band at col1;
        // row2 keeps {0,1,2} -> above-band at the kept final col2; row3 keeps {0,1} ->
        // below-band at col0. Padding-column shifts are irrelevant (masked) -> 0.10.
        #[rustfmt::skip]
        let shift = Tensor::from_vec(
            vec![0.30, 0.10, 0.10,
                 0.05, -0.30, 0.10,
                 0.05, 0.05, 0.30,
                 -0.30, 0.05, 0.10f64],
            (4, 3), &dev,
        ).unwrap();
        let adv = Tensor::from_vec(vec![0.8, -0.7, 0.5, -0.4f64], (4, 1), &dev).unwrap();
        run_gradcheck_ragged_with(loss_type, beta, vec![1, 2, 3, 2], &shift, &adv, &dev);
    }

    /// Gradcheck under a ragged mask containing an **all-padding row** — lengths
    /// `[0, 2, 3, 1]` over `T = 3`, so row 0 keeps no real tokens. This exercises the
    /// GRPO denominator clamp `mask.sum(-1).maximum(1)` in [`masked_mean_tensor`]
    /// (without it the zero-length row's `0 / 0` is `NaN` and poisons the whole
    /// backward) and confirms an entirely-masked row is gradient-inert: it contributes
    /// exactly zero to the loss and to every parameter gradient while the other rows
    /// stay correct. A `completion_lens` of `0` is a contract-valid value
    /// (`0..=comp_len`) the production loss deliberately defends against, so the oracle
    /// covers it.
    fn run_gradcheck_all_padding(loss_type: LossType, beta: f64) {
        let dev = cpu();
        // Row 0 is fully masked (its shifts/advantage are irrelevant). Remaining kept
        // positions straddle the clip band: row1 keeps {0,1} -> above/below; row2 keeps
        // {0,1,2} incl. the final col; row3 keeps {0} -> below-band.
        #[rustfmt::skip]
        let shift = Tensor::from_vec(
            vec![0.10, 0.10, 0.10,
                 0.30, -0.30, 0.10,
                 0.05, 0.30, -0.30,
                 -0.30, 0.10, 0.10f64],
            (4, 3), &dev,
        ).unwrap();
        let adv = Tensor::from_vec(vec![0.8, -0.7, 0.5, -0.4f64], (4, 1), &dev).unwrap();
        run_gradcheck_ragged_with(loss_type, beta, vec![0, 2, 3, 1], &shift, &adv, &dev);
    }

    #[test]
    fn gradcheck_grpo_no_kl() {
        // Classic GRPO reduction, no reference policy (beta = 0): surrogate only.
        run_gradcheck(LossType::Grpo, 0.0);
    }

    #[test]
    fn gradcheck_grpo_with_kl() {
        // Classic GRPO reduction with the k3 KL penalty active (beta > 0).
        run_gradcheck(LossType::Grpo, 0.1);
    }

    #[test]
    fn gradcheck_dr_grpo_with_kl() {
        // Dr.GRPO fixed-denominator reduction with the KL penalty active.
        run_gradcheck(LossType::DrGrpo, 0.1);
    }

    #[test]
    fn gradcheck_ragged_grpo_no_kl() {
        // Variable per-row GRPO denominator (1/len_i), surrogate only (beta = 0).
        run_gradcheck_ragged(LossType::Grpo, 0.0);
    }

    #[test]
    fn gradcheck_ragged_grpo_with_kl() {
        // Variable per-row GRPO denominator with the k3 KL penalty active.
        run_gradcheck_ragged(LossType::Grpo, 0.1);
    }

    #[test]
    fn gradcheck_ragged_dr_grpo_with_kl() {
        // Dr.GRPO fixed denominator under ragged real-token counts, KL active.
        run_gradcheck_ragged(LossType::DrGrpo, 0.1);
    }

    #[test]
    fn gradcheck_all_padding_grpo_no_kl() {
        // All-padding row (len 0) exercises the GRPO denominator clamp; surrogate only.
        run_gradcheck_all_padding(LossType::Grpo, 0.0);
    }

    #[test]
    fn gradcheck_all_padding_grpo_with_kl() {
        // GRPO denominator clamp with the k3 KL penalty active.
        run_gradcheck_all_padding(LossType::Grpo, 0.1);
    }

    #[test]
    fn gradcheck_all_padding_dr_grpo_with_kl() {
        // Dr.GRPO fixed denom: a fully-masked row stays gradient-inert, KL active.
        run_gradcheck_all_padding(LossType::DrGrpo, 0.1);
    }

    #[test]
    fn gradcheck_dapo_no_kl() {
        // DAPO token-level batch normalizer (the new default), surrogate only.
        run_gradcheck(LossType::Dapo, 0.0);
    }

    #[test]
    fn gradcheck_dapo_with_kl() {
        // DAPO reduction with the k3 KL penalty active.
        run_gradcheck(LossType::Dapo, 0.1);
    }

    #[test]
    fn gradcheck_ragged_dapo_with_kl() {
        // DAPO active-token denominator under ragged real-token counts, KL active.
        run_gradcheck_ragged(LossType::Dapo, 0.1);
    }

    #[test]
    fn gradcheck_all_padding_dapo_with_kl() {
        // DAPO: a fully-masked row adds nothing to numerator or denominator.
        run_gradcheck_all_padding(LossType::Dapo, 0.1);
    }

    #[test]
    fn gradcheck_sequence_level_grpo_no_kl() {
        // GSPO sequence-level ratio: the masked-mean log-ratio reshapes the
        // gradient (it flows through the per-sequence mean), so the analytic
        // gradient is re-derived by candle and pinned numerically here.
        run_gradcheck_level(LossType::Grpo, 0.0, ImportanceSamplingLevel::Sequence);
    }

    #[test]
    fn gradcheck_sequence_level_dapo_with_kl() {
        // Sequence-level ratio under the DAPO reduction with KL active — the
        // GSPO-recipe combination closest to a real MoE-era run.
        run_gradcheck_level(LossType::Dapo, 0.1, ImportanceSamplingLevel::Sequence);
    }

    #[test]
    fn gradcheck_clip_higher_dapo_no_kl() {
        run_gradcheck_clip_higher(LossType::Dapo, 0.0);
    }

    #[test]
    fn gradcheck_clip_higher_grpo_with_kl() {
        run_gradcheck_clip_higher(LossType::Grpo, 0.1);
    }
}
