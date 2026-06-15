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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::optim::ParamsAdamW;
use candle_nn::Optimizer;
use serde::{Deserialize, Serialize};

use crate::comm::{Comm, SoloComm};
use crate::grpo::{
    group_advantages, zero_mask_rows, ImportanceSamplingLevel, LossType, ScaleRewards,
};
use crate::nn::grad_coverage;
use crate::optim::{FerrlAdamW, OptimizerState};
use crate::policy::{GenConfig, Policy, Rollout};
use crate::reward::RewardFn;
use crate::telemetry::{Metrics, MetricsWriter, RunDir, TelemetryError};

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
    /// Writing a periodic adapter checkpoint failed.
    #[error(transparent)]
    Checkpoint(#[from] crate::checkpoint::CheckpointError),
    /// A data-parallel collective failed (contribution mismatch, a peer rank
    /// timing out, or a poisoned world) — see [`crate::comm`].
    #[error(transparent)]
    Comm(#[from] crate::comm::CommError),
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
    /// KL penalty coefficient. The reference (adapter-disabled) log-probs and
    /// the k3 KL term are computed only when this is `> 0`.
    pub beta: f64,
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
    /// If set, write an adapter checkpoint to `checkpoints/step-<n>/` (with a
    /// resumable manifest) every `checkpoint_every` completed steps **and** after
    /// the final step (so a completed run always persists its final adapter, even
    /// when this does not divide `steps`). `None` (the default) disables
    /// checkpointing entirely. `#[serde(default)]` so a `config.json` written before
    /// this field existed still deserializes (to `None`).
    #[serde(default)]
    pub checkpoint_every: Option<u64>,
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
            clip_eps: 0.2,
            clip_eps_high: None,
            importance_sampling_level: ImportanceSamplingLevel::Token,
            lr: 1e-3,
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
            grad_accum_steps: 1,
            checkpoint_every: None,
            eos_token_id: None,
        }
    }
}

impl TrainerConfig {
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
        require(
            self.clip_eps.is_finite() && self.clip_eps >= 0.0 && self.clip_eps < 1.0,
            "clip_eps must be finite and in [0, 1) (>= 1 disables the lower clip)",
        )?;
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

    /// The effective learning rate for 0-based optimizer step `step`: linear
    /// warmup `lr · (step + 1) / warmup_steps` over the first
    /// [`warmup_steps`](Self::warmup_steps) steps, then constant [`lr`](Self::lr).
    /// A pure function of the step index, so a resume re-enters the schedule
    /// faithfully.
    #[must_use]
    pub fn lr_at(&self, step: u64) -> f64 {
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
    comm: Box<dyn Comm>,
    /// Optional cooperative preemption flag (see
    /// [`with_preemption_flag`](Self::with_preemption_flag)). When `Some` and set,
    /// the loop writes a final checkpoint and stops at the next step boundary. Safe
    /// to install unevenly across DP ranks — the per-step poll is install-invariant.
    preempt: Option<Arc<AtomicBool>>,
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
    /// count: window `step` consumes the `grad_accum_steps × world_size`
    /// prompts starting at `step × grad_accum_steps × world_size` (mod len),
    /// rank `r` taking the `r`-th contiguous slice — the union across ranks
    /// is exactly the window a single-rank run with the global accumulation
    /// count would consume.
    ///
    /// Per-rank weights stay in bitwise lockstep (same start + all-reduced
    /// gradients + same optimizer arithmetic). Metrics: `kl`, `clip_ratio`
    /// and `grad_norm` are **global** (reduced); everything else —
    /// reward/length statistics, `frac_reward_zero_std`, `dropped_rows`,
    /// `frac_truncated`, and the `rollout_*` off-policy telemetry — describes
    /// the rank's **local shard**. Checkpoints are written by rank 0 only —
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
        Ok(Self {
            config,
            writer,
            checkpoints_dir: run.checkpoints_dir(),
            comm: Box::new(comm),
            preempt: None,
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

    /// Run `config.steps` optimizer steps — each over a window of
    /// `config.grad_accum_steps` prompts — cycling through `prompts`, returning one
    /// [`Metrics`] row per optimizer step (also appended to `metrics.jsonl`).
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError`] if any forward, optimizer step, telemetry write,
    /// or the grad-coverage canary fails. A canary failure aborts the run.
    ///
    /// # Panics
    ///
    /// Panics if `prompts` is empty: a run with no data is a caller bug.
    pub fn train<P: Policy, R: RewardFn>(
        &mut self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompts: &[String],
    ) -> Result<Vec<Metrics>, TrainerError> {
        self.run(0, None, policy, reward_fn, tokenizer, prompts)
    }

    /// Resume training from `start_step`, running steps `start_step .. config.steps`
    /// (so the total run still ends at `config.steps`). Returns the per-step
    /// [`Metrics`] for the steps actually executed (empty if `start_step >=
    /// config.steps`); they are also **appended** to `metrics.jsonl`, continuing
    /// the prior run's stream.
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
    /// Panics if `prompts` is empty.
    pub fn train_from<P: Policy, R: RewardFn>(
        &mut self,
        start_step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompts: &[String],
    ) -> Result<Vec<Metrics>, TrainerError> {
        self.run(start_step, None, policy, reward_fn, tokenizer, prompts)
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
    /// Panics if `prompts` is empty.
    pub fn resume<P: Policy, R: RewardFn>(
        &mut self,
        checkpoint_dir: impl AsRef<Path>,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompts: &[String],
    ) -> Result<Vec<Metrics>, TrainerError> {
        let checkpoint_dir = checkpoint_dir.as_ref();
        // Recipe pre-flight BEFORE the positional load mutates the live vars:
        // count/shape/dtype validation cannot distinguish shape-aliased recipes
        // (k/v and gate/up projections are shape-identical), so a mismatch here
        // would land trained adapters on the wrong projections silently.
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
        // Restore the sampler RNG (v2). A v1 checkpoint carries none → keep the policy's
        // current sampler (the documented fresh-momentum fallback).
        if let Some(blob) = &loaded.sampler_state {
            policy.restore_sampler_state(blob)?;
        }
        self.run(
            loaded.step,
            loaded.optimizer_state,
            policy,
            reward_fn,
            tokenizer,
            prompts,
        )
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
    /// # Errors
    ///
    /// As [`resume`](Self::resume) (when a checkpoint is found) or
    /// [`train`](Self::train) (when none is), plus [`TrainerError::Checkpoint`] if
    /// `checkpoints/` exists but cannot be listed.
    ///
    /// # Panics
    ///
    /// Panics if `prompts` is empty.
    pub fn resume_latest<P: Policy, R: RewardFn>(
        &mut self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompts: &[String],
    ) -> Result<Vec<Metrics>, TrainerError> {
        match crate::checkpoint::latest_checkpoint(&self.checkpoints_dir)? {
            Some(latest) => {
                tracing::info!(
                    resume_step = latest.step,
                    dir = %latest.dir.display(),
                    "resume_latest: continuing from the newest checkpoint"
                );
                self.resume(latest.dir, policy, reward_fn, tokenizer, prompts)
            }
            None => {
                tracing::info!("resume_latest: no checkpoint found — starting a fresh run");
                self.train(policy, reward_fn, tokenizer, prompts)
            }
        }
    }

    /// Shared loop for [`train`](Self::train) / [`train_from`](Self::train_from):
    /// run optimizer steps `start_step .. config.steps`, each consuming a window of
    /// `grad_accum_steps` prompts, checkpointing on the configured cadence.
    fn run<P: Policy, R: RewardFn>(
        &mut self,
        start_step: u64,
        resume_opt_state: Option<OptimizerState>,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompts: &[String],
    ) -> Result<Vec<Metrics>, TrainerError> {
        assert!(!prompts.is_empty(), "train: no prompts");
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
        if self.config.beta > 0.0 {
            policy.set_adapter_enabled(false);
            let toggleable = !policy.adapter_enabled();
            policy.set_adapter_enabled(true);
            if !toggleable {
                return Err(TrainerError::Contract(
                    "beta > 0 needs the adapter-disabled reference policy, but this policy \
                     cannot disable its adapter (full fine-tuning mode?) — train with \
                     beta = 0, or use a LoRA recipe for KL-regularized runs"
                        .into(),
                ));
            }
        }
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
            // Linear warmup, constant after: a pure function of the step index,
            // so a resume mid-warmup re-enters the schedule exactly.
            opt.set_learning_rate(self.config.lr_at(step));
            let m =
                self.run_window(step, policy, reward_fn, tokenizer, prompts, &mut opt, &vars)?;
            self.writer.append(&m)?;
            history.push(m);
            self.maybe_checkpoint(step, &vars, &opt, policy)?;
            // Cooperative preemption stop (globalized across the DP world): on a
            // Slurm preempt / timeout grace signal the run binary flips the flag,
            // and we save a final checkpoint at this completed step and stop
            // cleanly so a requeued run picks up here via `resume_latest`. (May
            // re-write the cadence checkpoint just written above — idempotent.)
            if self.preempt_requested()? {
                let completed = step + 1;
                self.write_checkpoint(completed, &vars, &opt, policy)?;
                tracing::warn!(
                    completed_steps = completed,
                    "preemption requested: checkpointed and stopping early"
                );
                break;
            }
        }
        Ok(history)
    }

    /// One optimizer step over a window of `grad_accum_steps` prompts: collect each
    /// prompt's group (rollout → reward → advantages, snapshotting the non-degenerate
    /// ones), then run the `mu` inner epochs that accumulate the window's gradients
    /// into a single `AdamW` update. Returns the window's aggregated [`Metrics`].
    #[allow(clippy::too_many_arguments)]
    fn run_window<P: Policy, R: RewardFn>(
        &self,
        step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompts: &[String],
        opt: &mut FerrlAdamW,
        vars: &[Var],
    ) -> Result<Metrics, TrainerError> {
        let accum = self.config.grad_accum_steps;
        let world = self.comm.world_size();
        let mut stats = Vec::with_capacity(accum);
        let mut live = Vec::with_capacity(accum);
        for j in 0..accum {
            // Continuous prompt cycling across windows: window `step` consumes the
            // `accum × world` prompts starting at `step*accum*world` (mod len),
            // rank `r` taking the contiguous slice at offset `r*accum` — so a
            // resume at window `start_step` continues the order an uninterrupted
            // run would have seen, and the union of the ranks' shards is exactly
            // the window a single-rank run at the global accumulation count
            // consumes (the DP sharding contract the equivalence oracle pins).
            // At world 1 this is the legacy `step*accum + j`.
            // The prompt's GLOBAL ordinal in the flattened rollout stream — the same
            // for a given (step, window position) regardless of world size / rank, so
            // seeding the rollout by it (in `collect_prompt`) makes a world-W run
            // reproduce the single-process draws. `idx` (mod len) selects which prompt;
            // `g` (unbounded) seeds it, so re-cycling the dataset across windows still
            // gives fresh per-window rollout RNG.
            let local = self.comm.rank() * accum + j;
            // `g` (u64, the seed base) and `idx` (usize) use the same plain
            // arithmetic; for any real run neither overflows, so they stay
            // consistent by construction.
            let g = step * (accum * world) as u64 + local as u64;
            let idx = (step as usize * (accum * world) + local) % prompts.len();
            let (stat, item) =
                self.collect_prompt(policy, reward_fn, tokenizer, &prompts[idx], g)?;
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
            self.comm
                .all_reduce_scalar_sum(local_tokens as f64)?
                .max(1.0)
        } else {
            local_tokens.max(1) as f64
        };
        // A window with no live prompts (every group degenerate) is a GRPO no-op: no
        // update, no canary — mirroring the single-prompt degenerate skip. Under DP
        // the decision must be GLOBAL: a rank whose local shard is all-degenerate
        // still has to enter the inner epochs' collectives (contributing zeros)
        // while any peer holds live items — a local skip would deadlock the world.
        let n_live_global = if world > 1 {
            self.comm.all_reduce_scalar_sum(live.len() as f64)?
        } else {
            live.len() as f64
        };
        let agg = if n_live_global == 0.0 {
            InnerAgg::default()
        } else {
            self.update_window(policy, &live, vars, opt, window_tokens, n_live_global)?
        };
        Ok(self.build_window_metrics(step, &stats, &agg, opt))
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
    fn maybe_checkpoint<P: Policy>(
        &self,
        step: u64,
        vars: &[Var],
        opt: &FerrlAdamW,
        policy: &P,
    ) -> Result<(), TrainerError> {
        // Under DP, rank 0 writes the world's checkpoint: the weights and the
        // optimizer moments are rank-identical by the lockstep invariant, so
        // N copies would be redundant (and a shared run dir would race). See
        // `with_comm` for the resume contract (every rank loads rank 0's
        // checkpoint; the sampler blob is rank 0's).
        if self.comm.rank() != 0 {
            return Ok(());
        }
        let Some(every) = self.config.checkpoint_every else {
            return Ok(());
        };
        let completed = step + 1;
        let is_final = completed == self.config.steps;
        if completed % every != 0 && !is_final {
            return Ok(());
        }
        self.write_checkpoint(completed, vars, opt, policy)
    }

    /// Write a momentum-faithful (v3) checkpoint to `checkpoints/step-<completed>/`
    /// unconditionally — the caller decides *when*: the periodic cadence
    /// ([`maybe_checkpoint`](Self::maybe_checkpoint)) or the preemption stop in
    /// [`run`](Self::run), which saves a final checkpoint before a requeue.
    /// Rank-0-only under DP: weights and optimizer moments are rank-identical by
    /// lockstep and the sampler blob is rank 0's (see
    /// [`with_comm`](Self::with_comm)), so a non-zero rank is a no-op. Re-writing
    /// an already-published `step-<completed>` is idempotent (the writer replaces
    /// it atomically), so a preemption that coincides with a cadence write is
    /// harmless.
    fn write_checkpoint<P: Policy>(
        &self,
        completed: u64,
        vars: &[Var],
        opt: &FerrlAdamW,
        policy: &P,
    ) -> Result<(), TrainerError> {
        if self.comm.rank() != 0 {
            return Ok(());
        }
        let dir = self.checkpoints_dir.join(format!("step-{completed}"));
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
    }

    /// Whether a stop has been requested via the preemption flag
    /// ([`with_preemption_flag`](Self::with_preemption_flag)), decided **globally**
    /// across the DP world so every rank stops on the same step (a local-only stop
    /// would deadlock — one rank breaks out while its peers enter the next window's
    /// collectives and wait forever).
    ///
    /// **Install-invariant:** at `world_size() > 1` *every* rank runs the poll's
    /// scalar reduce every step regardless of whether it holds a flag (a flag-less
    /// rank contributes `0.0`), so the collective sequence never depends on per-rank
    /// install state — an uneven install across ranks cannot deadlock the world, it
    /// just means no preemption. The cost is one cheap scalar all-reduce per step
    /// under DP; world 1 is a plain local read with no collective.
    fn preempt_requested(&self) -> Result<bool, TrainerError> {
        let local = self
            .preempt
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if self.comm.world_size() > 1 {
            // Every rank reduces every step (flag-less ranks contribute 0.0), so the
            // collective sequence is identical across ranks no matter who holds a
            // flag — an uneven install can't desync the world.
            Ok(self
                .comm
                .all_reduce_scalar_sum(if local { 1.0 } else { 0.0 })?
                > 0.0)
        } else {
            Ok(local)
        }
    }

    /// Collect one prompt's group for the current window: rollout (adapter on) →
    /// reward → group advantages, validating the `Policy` / `RewardFn` contract. A
    /// degenerate (all-zero-advantage) group returns `(stat, None)` — a GRPO no-op
    /// with no snapshot and no update; a live group also snapshots the old / reference
    /// log-probs (taken now, at the window's start) into a [`LiveItem`] for the inner
    /// epochs.
    fn collect_prompt<P: Policy, R: RewardFn>(
        &self,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompt: &str,
        global_prompt_index: u64,
    ) -> Result<(PromptStat, Option<LiveItem>), TrainerError> {
        // Rollout with the adapter on, then score it.
        policy.set_adapter_enabled(true);
        let prompt_ids = tokenizer.encode(prompt);
        // A prompt that encodes to zero tokens (a real tokenizer can yield `[]` for
        // empty/whitespace input) must fail HERE, before generate: teacher-forced
        // scoring needs >= 1 prompt token, and a policy that builds an empty input
        // and reads the last position (`len - 1`) underflows/panics. The rollout
        // contract (`prompt_len >= 1`) is otherwise only checked after generation.
        if prompt_ids.is_empty() {
            return Err(TrainerError::Contract(format!(
                "prompt encoded to zero tokens: {prompt:?}"
            )));
        }
        let gen = GenConfig {
            group_size: self.config.group_size,
            max_new_tokens: self.config.max_new_tokens,
            temperature: self.config.temperature,
            eos_token_id: self.config.eos_token_id,
            eval_sampling: None,
        };
        // Seed the rollout's per-row substreams from this prompt's GLOBAL row base
        // (its ordinal in the flattened world-global stream × group_size), so the
        // sampled tokens are invariant to the data-parallel shard layout and a resume
        // re-derives them — see `Policy::generate_at`.
        let global_row_base = global_prompt_index.wrapping_mul(self.config.group_size as u64);
        let rollout = policy.generate_at(&prompt_ids, &gen, global_row_base)?;
        // Validate the rollout BEFORE decoding/scoring it: completion_dims rejects
        // an empty, ragged, or shorter-than-prompt rollout (and a misaligned
        // behavior-log-prob capture), so the decode slice `ids[prompt_len..]`
        // cannot panic on malformed Policy output.
        let (_, comp_len) = completion_dims(&rollout)?;
        // The TIS weight is undefined without the behavior policy's probabilities;
        // fail loud on the FIRST prompt rather than silently training uncorrected.
        if self.config.tis && rollout.rollout_logprobs.is_none() {
            return Err(TrainerError::Contract(
                "tis is enabled but Policy::generate captured no rollout log-probs \
                 (Rollout::rollout_logprobs is None) — this policy cannot supply the \
                 behavior probabilities the correction needs"
                    .into(),
            ));
        }
        // Policy::generate is contracted to return exactly `group_size` completions.
        // An underfilled rollout would otherwise become a degenerate single-item
        // group (all-zero advantages -> silently skipped); an overfilled one would
        // silently change the effective group size. Reject either.
        if rollout.len() != self.config.group_size {
            return Err(TrainerError::Contract(format!(
                "Policy::generate returned {} completions for group_size {}",
                rollout.len(),
                self.config.group_size
            )));
        }
        let completions = decode_completions(&rollout, tokenizer);
        let rewards = reward_fn.reward_group(prompt, &completions);
        if rewards.len() != rollout.len() {
            return Err(TrainerError::Contract(format!(
                "reward_group returned {} rewards for {} completions",
                rewards.len(),
                rollout.len()
            )));
        }

        // Length-aware loss mask: column j of sequence i is a real completion token
        // (kept, 1.0) iff j < completion_lens[i]; the EOS padding at and beyond that
        // index is masked out (0.0). With eos_token_id == None every length is the
        // full width, so every row is all-ones — bit-identical to the legacy mask.
        let mut mask_rows = length_mask_rows(&rollout, comp_len);

        // DAPO overlong filtering (TRL `mask_truncated_completions`): zero the
        // whole mask row of any completion that ran to the full width without
        // sampling EOS. The completion still feeds the reward statistics /
        // advantages and the DAPO normalizer (matching TRL); only its loss
        // tokens are removed. Inert when eos_token_id is None.
        let truncated = if self.config.truncation_masking {
            mask_truncated_rows(&rollout, comp_len, self.config.eos_token_id, &mut mask_rows)
        } else {
            0
        };
        // zero_mask_rows counts any all-pad row — truncation-masked completions
        // land here too, so a batch that lost loss signal stays observable.
        let dropped = zero_mask_rows(&mask_rows);

        // Group-normalized advantages (scalar oracle). A group whose advantages are
        // all exactly zero — no reward spread, or non-finite rewards forced to a 0
        // advantage — carries no SURROGATE gradient. With `beta == 0` (no KL term)
        // it is therefore a complete no-op and is skipped (no snapshot, no update,
        // no canary). With `beta > 0` it must stay LIVE: TRL keeps every
        // completion in the batch, and the KL penalty still pulls a
        // zero-advantage group toward the reference (its surrogate contributes
        // exactly 0 — the zero-advantage guard in the clipped surrogate — but the
        // k3 term carries gradient). Skipping it would silently drop that
        // regularization, diverging from TRL whenever rewards saturate mid-run.
        let rewards_f64: Vec<f64> = rewards.iter().map(|&r| f64::from(r)).collect();
        let advantages = group_advantages(&rewards_f64, self.config.scale_rewards);
        let degenerate = advantages.iter().all(|a| *a == 0.0);
        let mut stat = PromptStat {
            completion_len: mean_completion_len(&rollout),
            completion_tokens: rollout.completion_lens.iter().sum(),
            dropped,
            truncated,
            degenerate,
            ratio_stats: None,
            rewards,
        };
        if degenerate && self.config.beta <= 0.0 {
            return Ok((stat, None));
        }

        // Snapshot the old / reference log-probs once (the window's "old" policy),
        // reused across the mu inner epochs. Value-only, so the detached
        // scoring path: same values, a fraction of the activation footprint
        // on policies that override it (no tape is built or captured).
        let logp_old = policy.token_logprobs_detached(&rollout)?;
        // Train/rollout off-policy diagnostics + the optional TIS weight, both off
        // the captured behavior log-probs vs the logp_old scoring snapshot.
        let (ratio_stats, tis_w) = rollout_ratio_and_tis(
            &rollout,
            &logp_old,
            &mask_rows,
            self.config.tis_imp_ratio_cap,
            self.config.tis,
        )?;
        stat.ratio_stats = ratio_stats;
        let device = logp_old.device().clone();
        let mask = mask_rows_to_tensor(&mask_rows, &device)?;
        let logp_ref = self.reference_logprobs(policy, &rollout)?;
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
    }

    /// Run the `mu` inner epochs over a window's live items, each epoch accumulating
    /// every live prompt's gradient into one `AdamW` step. The last epoch's
    /// diagnostics land in the window's metrics. `window_tokens` is the window's
    /// total completion-token count (the DAPO normalizer, constant across epochs;
    /// global under DP) and `n_live_global` the world's live-item count (the
    /// kl/clip-mean divisor; `live.len()` at world 1). Under DP every rank runs
    /// these epochs — `live` may be empty on a rank whose shard was
    /// all-degenerate; it participates in the collectives with zeros.
    fn update_window<P: Policy>(
        &self,
        policy: &P,
        live: &[LiveItem],
        vars: &[Var],
        opt: &mut FerrlAdamW,
        window_tokens: f64,
        n_live_global: f64,
    ) -> Result<InnerAgg, TrainerError> {
        let mut agg = InnerAgg::default();
        for _ in 0..self.config.mu {
            agg = self.accumulate_step(policy, live, vars, opt, window_tokens, n_live_global)?;
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
    fn accumulate_step<P: Policy>(
        &self,
        policy: &P,
        live: &[LiveItem],
        vars: &[Var],
        opt: &mut FerrlAdamW,
        window_tokens: f64,
        n_live_global: f64,
    ) -> Result<InnerAgg, TrainerError> {
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut covered = vec![true; vars.len()];
        let mut container: Option<GradStore> = None;
        let mut sum_kl = 0.0_f32;
        let mut sum_clip = 0.0_f32;
        for item in live {
            let (grads, kl, clip_frac) = self.item_backward(policy, item, window_tokens)?;
            sum_kl += kl;
            sum_clip += clip_frac;
            fold_var_grads(vars, &grads, &mut acc, &mut covered)?;
            container = Some(grads);
        }
        let (kl, clip_frac, uncovered_global) = if self.comm.world_size() > 1 {
            self.reduce_epoch(vars, &mut acc, &covered, sum_kl, sum_clip, n_live_global)?
        } else {
            (
                sum_kl / n_live_global as f32,
                sum_clip / n_live_global as f32,
                0.0,
            )
        };
        // Reuse the last backward's store as the optimizer container, overwriting its
        // trainable-var entries with the accumulated sums (and dropping any var absent
        // from some prompt so the canary catches the silent-skip). A rank whose local
        // shard is empty (DP, all-degenerate shard) has no backward to reuse — candle
        // exposes no public `GradStore` constructor, so mint one from a throwaway
        // one-node backward (its stray entry is harmless: only var entries are read
        // downstream).
        let store = match container {
            Some(store) => store,
            None => empty_grad_store(vars)?,
        };
        let mut store = combine_into_store(vars, store, &acc, &covered);
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
                scale_var_grads(vars, &mut store, max / grad_norm)?;
            }
        }
        opt.step(&store)?;
        Ok(InnerAgg {
            kl,
            clip_frac,
            grad_norm: grad_norm as f32,
        })
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
        let mut flat = Vec::with_capacity(vars.len());
        for (i, v) in vars.iter().enumerate() {
            flat.push(match &acc[i] {
                Some(g) => g.clone(),
                None => v.as_tensor().zeros_like()?,
            });
        }
        self.comm.all_reduce_sum(&mut flat)?;
        for (slot, g) in acc.iter_mut().zip(flat) {
            *slot = Some(g);
        }
        let uncovered_local = covered.iter().filter(|c| !**c).count() as f64;
        let uncovered_global = self.comm.all_reduce_scalar_sum(uncovered_local)?;
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
    fn item_backward<P: Policy>(
        &self,
        policy: &P,
        item: &LiveItem,
        window_tokens: f64,
    ) -> Result<(GradStore, f32, f32), TrainerError> {
        let logp = policy.token_logprobs(&item.rollout)?;
        let cfg = LossCfg {
            clip_eps_low: self.config.clip_eps,
            clip_eps_high: self.config.clip_eps_high_eff(),
            beta: self.config.beta,
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
        let global_items = self.config.grad_accum_steps * self.comm.world_size();
        if global_items > 1 && self.config.loss_type != LossType::Dapo {
            loss = loss.affine(1.0 / global_items as f64, 0.0)?;
        }
        // Scalar diagnostics, off the differentiated path. The ratio is formed at
        // the configured level over the same padding-substituted log-probs the
        // loss uses, so the clip-fraction metric reports the ratio the surrogate
        // actually clipped.
        let logp_diag = logp.detach();
        let logp_sub = substitute_padding(&logp_diag, &item.logp_old, &item.mask)?;
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
        let kl = self.kl_metric(&logp, item.logp_ref.as_ref(), &item.mask)?;
        // Through the policy seam (default: exactly `loss.backward()`): under
        // activation checkpointing the policy stitches the full gradient from
        // its boundary tape — the canary downstream holds either way.
        let grads = policy.backward(&loss)?;
        Ok((grads, kl, clip_frac))
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
    fn reference_logprobs<P: Policy>(
        &self,
        policy: &mut P,
        rollout: &Rollout,
    ) -> Result<Option<Tensor>, TrainerError> {
        if self.config.beta <= 0.0 {
            return Ok(None);
        }
        policy.set_adapter_enabled(false);
        // Value-only (the reference is never trained): the detached scoring path.
        let logp = policy.token_logprobs_detached(rollout);
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
        // same condition that drove each skip, so metric and optimizer never disagree
        // (covers all-non-finite groups too, forced to all-zero advantages).
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
        m
    }
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

/// Build the optimizer's gradient store for an accumulation window by overwriting
/// `store`'s trainable-var entries with the accumulated sums in `acc`. A var marked
/// uncovered (absent from some prompt's backward) is left out entirely so
/// [`grad_coverage`] flags it. `store` (reused from the last prompt's backward) also
/// carries unrelated intermediate-node grads; the optimizer and canary read only the
/// var entries, so those are harmless.
fn combine_into_store(
    vars: &[Var],
    mut store: GradStore,
    acc: &[Option<Tensor>],
    covered: &[bool],
) -> GradStore {
    for (i, v) in vars.iter().enumerate() {
        store.remove(v.as_tensor());
        if covered[i] {
            if let Some(g) = &acc[i] {
                store.insert(v.as_tensor(), g.clone());
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

/// Reward `(mean, population-std)` over the **finite** rewards in a group. A
/// non-finite reward is ignored — mirroring [`group_advantages`], which drops it
/// from the group statistics — so one bad completion does not collapse the
/// headline metric. A group with no finite rewards reports `(0, 0)`.
fn reward_stats(rewards: &[f32]) -> (f32, f32) {
    let finite: Vec<f32> = rewards.iter().copied().filter(|r| r.is_finite()).collect();
    if finite.is_empty() {
        return (0.0, 0.0);
    }
    let n = finite.len() as f32;
    let mean = finite.iter().sum::<f32>() / n;
    let var = finite.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
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
        let dev = cpu();
        let tmp = WireTmp::new("grad");
        let run = RunDir::create(&tmp.0, "wire").unwrap();
        let trainer = Trainer::new(cfg.clone(), &run).unwrap();
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
        let (grads, _, _) = trainer.item_backward(policy, &item, window_tokens).unwrap();
        grads
            .get(logp.as_tensor())
            .expect("logp var must be in the grad store")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
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
        assert_eq!(back.mu, cfg.mu);
        assert_eq!(back.beta, cfg.beta);
        assert_eq!(back.loss_type, cfg.loss_type);
        assert_eq!(back.scale_rewards, cfg.scale_rewards);
        assert_eq!(back.grad_accum_steps, cfg.grad_accum_steps);
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
        assert_eq!(cfg.grad_accum_steps, 1);
        // The EOS field also predates the JSON above; serde fills it from the default.
        assert_eq!(cfg.eos_token_id, None);
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
    fn reward_stats_mean_and_std() {
        let (mean, std) = reward_stats(&[1.0, 1.0, 1.0]);
        assert_relative_eq!(mean, 1.0, epsilon = TOL);
        assert_relative_eq!(std, 0.0, epsilon = TOL);

        let (mean, std) = reward_stats(&[1.0, 0.0]);
        assert_relative_eq!(mean, 0.5, epsilon = TOL);
        assert_relative_eq!(std, 0.5, epsilon = TOL);
    }

    #[test]
    fn reward_stats_ignores_non_finite_rewards() {
        // One bad completion must not collapse the headline metric: the finite
        // rewards still produce mean=2, mirroring how group_advantages isolates it.
        let (mean, std) = reward_stats(&[1.0, f32::NAN, 3.0]);
        assert_relative_eq!(mean, 2.0, epsilon = TOL);
        assert!(std.is_finite());
        let (mean, _) = reward_stats(&[1.0, f32::INFINITY, 3.0]);
        assert_relative_eq!(mean, 2.0, epsilon = TOL);
        // No finite rewards -> (0, 0), not NaN.
        let (mean, std) = reward_stats(&[f32::NAN, f32::NAN]);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
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
    /// the same `j < completion_lens[i]` predicate `collect_prompt` uses, at `f64` for
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
