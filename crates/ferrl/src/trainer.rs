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
//! are zero (`frac_reward_zero_std == 1`) — carries no learning signal and is a
//! GRPO no-op: the trainer performs no update for that step (and runs no canary).

use std::path::PathBuf;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::optim::ParamsAdamW;
use candle_nn::Optimizer;
use serde::{Deserialize, Serialize};

use crate::grpo::{group_advantages, zero_mask_rows, LossType, ScaleRewards};
use crate::nn::grad_coverage;
use crate::optim::FerrlAdamW;
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
    /// Rollout sampling temperature. Scoring is always at temperature `1`.
    pub temperature: f64,
    /// Inner optimization steps per rollout. At `1` the importance ratio is
    /// exactly `1` (current log-probs equal the frozen snapshot), so the clip is
    /// wired but inert.
    pub mu: usize,
    /// KL penalty coefficient. The reference (adapter-disabled) log-probs and
    /// the k3 KL term are computed only when this is `> 0`.
    pub beta: f64,
    /// PPO clip half-width `epsilon` (e.g. `0.2`).
    pub clip_eps: f64,
    /// `AdamW` learning rate.
    pub lr: f64,
    /// `AdamW` weight decay. Defaults to `0` (the toy trains pure policy gradient).
    pub weight_decay: f64,
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
            lr: 1e-3,
            weight_decay: 0.0,
            loss_type: LossType::Grpo,
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
    /// lower clip); or if `checkpoint_every` is `Some(0)`.
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
            self.beta.is_finite() && self.beta >= 0.0,
            "beta must be finite and >= 0",
        )?;
        require(
            self.clip_eps.is_finite() && self.clip_eps >= 0.0 && self.clip_eps < 1.0,
            "clip_eps must be finite and in [0, 1) (>= 1 disables the lower clip)",
        )?;
        if let Some(every) = self.checkpoint_every {
            require(
                every >= 1,
                "checkpoint_every must be >= 1 when set (0 would checkpoint every step)",
            )?;
        }
        Ok(())
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
struct LiveItem {
    rollout: Rollout,
    advantages: Tensor,
    logp_old: Tensor,
    logp_ref: Option<Tensor>,
    mask: Tensor,
}

/// Per-prompt quantities aggregated into a window's [`Metrics`] (the reward
/// distribution, completion length, dropped rows, and whether the group was a
/// degenerate all-equal-reward no-op).
struct PromptStat {
    rewards: Vec<f32>,
    completion_len: f32,
    dropped: usize,
    degenerate: bool,
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
        config.validate()?;
        run.write_config(&config)?;
        let writer = run.metrics_writer()?;
        Ok(Self {
            config,
            writer,
            checkpoints_dir: run.checkpoints_dir(),
        })
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
        self.run(0, policy, reward_fn, tokenizer, prompts)
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
    /// **Not bit-exact to an uninterrupted run.** A fresh [`FerrlAdamW`] is
    /// constructed (its moment estimates restart from zero, re-warming the bias
    /// correction) and the policy's sampler RNG is whatever the reloaded policy
    /// carries; neither is persisted by [`crate::checkpoint`] yet. The reloaded
    /// *adapter weights* are exact; the post-resume trajectory is a faithful
    /// continuation, not a replay. (Momentum-faithful resume — persisting and
    /// restoring both the optimizer moments and the sampler RNG — is P6-B; owning
    /// [`FerrlAdamW`] instead of candle's `AdamW` is its first step.)
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
        self.run(start_step, policy, reward_fn, tokenizer, prompts)
    }

    /// Shared loop for [`train`](Self::train) / [`train_from`](Self::train_from):
    /// run optimizer steps `start_step .. config.steps`, each consuming a window of
    /// `grad_accum_steps` prompts, checkpointing on the configured cadence.
    fn run<P: Policy, R: RewardFn>(
        &mut self,
        start_step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompts: &[String],
    ) -> Result<Vec<Metrics>, TrainerError> {
        assert!(!prompts.is_empty(), "train: no prompts");
        let vars = policy.trainable_vars();
        let params = ParamsAdamW {
            lr: self.config.lr,
            weight_decay: self.config.weight_decay,
            ..Default::default()
        };
        let mut opt = FerrlAdamW::new(vars.clone(), params)?;
        let total = self.config.steps;
        let remaining = total.saturating_sub(start_step) as usize;
        let mut history = Vec::with_capacity(remaining);
        for step in start_step..total {
            let m =
                self.run_window(step, policy, reward_fn, tokenizer, prompts, &mut opt, &vars)?;
            self.writer.append(&m)?;
            history.push(m);
            self.maybe_checkpoint(step, &vars)?;
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
        let mut stats = Vec::with_capacity(accum);
        let mut live = Vec::with_capacity(accum);
        for j in 0..accum {
            // Continuous prompt cycling across windows: window `step` consumes prompts
            // `step*accum .. step*accum + accum` (mod len), so a resume at window
            // `start_step` continues the order an uninterrupted run would have seen.
            let idx = (step as usize * accum + j) % prompts.len();
            let (stat, item) = self.collect_prompt(policy, reward_fn, tokenizer, &prompts[idx])?;
            stats.push(stat);
            if let Some(item) = item {
                live.push(item);
            }
        }
        // A window with no live prompts (every group degenerate) is a GRPO no-op: no
        // update, no canary — mirroring the single-prompt degenerate skip.
        let agg = if live.is_empty() {
            InnerAgg::default()
        } else {
            self.update_window(policy, &live, vars, opt)?
        };
        Ok(self.build_window_metrics(step, &stats, &agg, opt))
    }

    /// After completing step `step` (0-based), write `checkpoints/step-<n>/` when
    /// the configured cadence divides the completed-step count `n = step + 1`, **or**
    /// `n` is the final step of the run. The final-step write guarantees a completed
    /// run always persists its final adapter even when `checkpoint_every` does not
    /// divide `steps` (otherwise the last steps' weights would live only in memory).
    /// The recorded manifest `step` is `n`, the index a resume continues from.
    fn maybe_checkpoint(&self, step: u64, vars: &[Var]) -> Result<(), TrainerError> {
        let Some(every) = self.config.checkpoint_every else {
            return Ok(());
        };
        let completed = step + 1;
        let is_final = completed == self.config.steps;
        if completed % every != 0 && !is_final {
            return Ok(());
        }
        let dir = self.checkpoints_dir.join(format!("step-{completed}"));
        crate::checkpoint::save_adapter(&dir, vars, completed)?;
        Ok(())
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
        };
        let rollout = policy.generate(&prompt_ids, &gen)?;
        // Validate the rollout BEFORE decoding/scoring it: completion_dims rejects
        // an empty, ragged, or shorter-than-prompt rollout, so the decode slice
        // `ids[prompt_len..]` cannot panic on malformed Policy output.
        let (_, comp_len) = completion_dims(&rollout)?;
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
        // zero_mask_rows counts any all-pad row: unreachable from EOS-inclusive
        // generation (every recorded length is >= 1), but retained as the
        // dropped-rows guard against a degenerate zero-length completion.
        let mask_rows = length_mask_rows(&rollout, comp_len);
        let dropped = zero_mask_rows(&mask_rows);

        // Group-normalized advantages (scalar oracle). A group whose advantages are
        // all exactly zero — no reward spread, or non-finite rewards forced to a 0
        // advantage — carries no gradient, so it is a no-op (no snapshot, no update,
        // no canary), exactly as the single-prompt path skipped it.
        let rewards_f64: Vec<f64> = rewards.iter().map(|&r| f64::from(r)).collect();
        let advantages = group_advantages(&rewards_f64, self.config.scale_rewards);
        let degenerate = advantages.iter().all(|a| *a == 0.0);
        let stat = PromptStat {
            completion_len: mean_completion_len(&rollout),
            dropped,
            degenerate,
            rewards,
        };
        if degenerate {
            return Ok((stat, None));
        }

        // Snapshot the old / reference log-probs once (the window's "old" policy),
        // reused across the mu inner epochs.
        let logp_old = policy.token_logprobs(&rollout)?.detach();
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
        };
        Ok((stat, Some(item)))
    }

    /// Run the `mu` inner epochs over a window's live items, each epoch accumulating
    /// every live prompt's gradient into one `AdamW` step. The last epoch's
    /// diagnostics land in the window's metrics.
    fn update_window<P: Policy>(
        &self,
        policy: &P,
        live: &[LiveItem],
        vars: &[Var],
        opt: &mut FerrlAdamW,
    ) -> Result<InnerAgg, TrainerError> {
        let mut agg = InnerAgg::default();
        for _ in 0..self.config.mu {
            agg = self.accumulate_step(policy, live, vars, opt)?;
        }
        Ok(agg)
    }

    /// One inner epoch: forward+backward each live prompt, fold its trainable-var
    /// gradients into a running sum, run the grad-coverage canary on the accumulated
    /// gradient, then take one optimizer step. Only one prompt's grad forward is held
    /// at a time (the accumulator keeps just the small per-var sums), so the window's
    /// peak memory is a single group's.
    fn accumulate_step<P: Policy>(
        &self,
        policy: &P,
        live: &[LiveItem],
        vars: &[Var],
        opt: &mut FerrlAdamW,
    ) -> Result<InnerAgg, TrainerError> {
        let n_live = live.len() as f32;
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut covered = vec![true; vars.len()];
        let mut container: Option<GradStore> = None;
        let mut sum_kl = 0.0_f32;
        let mut sum_clip = 0.0_f32;
        for item in live {
            let (grads, kl, clip_frac) = self.item_backward(policy, item)?;
            sum_kl += kl;
            sum_clip += clip_frac;
            fold_var_grads(vars, &grads, &mut acc, &mut covered)?;
            container = Some(grads);
        }
        // Reuse the last backward's store as the optimizer container, overwriting its
        // trainable-var entries with the accumulated sums (and dropping any var absent
        // from some prompt so the canary catches the silent-skip).
        let store = combine_into_store(vars, container.expect("live is non-empty"), &acc, &covered);
        let cov = grad_coverage(vars, &store)?;
        let kl = sum_kl / n_live;
        let clip_frac = sum_clip / n_live;
        // Fatal: a missing var (candle's silent-skip landmine — an absent grad entry)
        // or a non-finite accumulated gradient (a blowup).
        if !cov.is_covered() || cov.nonfinite > 0 {
            cov.clone().into_result()?;
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
        let grad_norm = global_grad_norm(vars, &store)? as f32;
        opt.step(&store)?;
        Ok(InnerAgg {
            kl,
            clip_frac,
            grad_norm,
        })
    }

    /// Forward + backward one live prompt: the single `grpo_loss` (scaled by
    /// `1 / grad_accum_steps` when accumulating, so once folded the window's gradient
    /// is the sum over its live prompts divided by the full window size — the mean
    /// when every prompt is live, with degenerate prompts diluting the step) plus its
    /// scalar diagnostics (clip fraction, mean k3 KL). Returns the backward's
    /// [`GradStore`] and the diagnostics. The scale is skipped at
    /// `grad_accum_steps == 1`, keeping the no-accumulation path bit-identical (no
    /// extra affine node, identical to the prior single-step loss).
    fn item_backward<P: Policy>(
        &self,
        policy: &P,
        item: &LiveItem,
    ) -> Result<(GradStore, f32, f32), TrainerError> {
        let logp = policy.token_logprobs(&item.rollout)?;
        let mut loss = grpo_loss(
            &logp,
            &item.logp_old,
            item.logp_ref.as_ref(),
            &item.advantages,
            &item.mask,
            self.config.clip_eps,
            self.config.beta,
            self.config.loss_type,
        )?;
        if self.config.grad_accum_steps > 1 {
            loss = loss.affine(1.0 / self.config.grad_accum_steps as f64, 0.0)?;
        }
        // Scalar diagnostics, off the differentiated path.
        let ratio = importance_ratio(&logp, &item.logp_old)?;
        let clip_frac = clip_fraction(&ratio, &item.advantages, self.config.clip_eps, &item.mask)?;
        let kl = self.kl_metric(&logp, item.logp_ref.as_ref(), &item.mask)?;
        let grads = loss.backward()?;
        Ok((grads, kl, clip_frac))
    }

    /// Mean masked k3 KL for the step's metrics — the diagnostic counterpart of the
    /// KL penalty [`grpo_loss`] folds into the differentiated objective. Returns `0`
    /// when no reference policy is active (`beta == 0`, so `logp_ref` is `None`).
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
        scalar_f32(&masked_mean_tensor(&kl, mask, self.config.loss_type)?)
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
        let logp = policy.token_logprobs(rollout);
        policy.set_adapter_enabled(true); // always restore.
        Ok(Some(logp?.detach()))
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
        m.kl = agg.kl;
        m.clip_ratio = agg.clip_frac;
        m.grad_norm = agg.grad_norm;
        m.lr = opt.learning_rate() as f32;
        m
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
    Ok((rollout.len(), comp_len))
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

/// The importance ratio `exp(logp - logp_old)`. At `mu = 1` the snapshot equals
/// the current log-probs, so this is exactly `1`.
fn importance_ratio(logp: &Tensor, logp_old: &Tensor) -> CandleResult<Tensor> {
    logp.broadcast_sub(logp_old)?.exp()
}

/// Per-token PPO clipped surrogate `min(ratio * A, clip(ratio) * A)`. The
/// differentiable counterpart of [`crate::grpo::clipped_surrogate`].
fn clipped_surrogate_tensor(
    ratio: &Tensor,
    advantages: &Tensor,
    clip_eps: f64,
) -> CandleResult<Tensor> {
    let unclipped = ratio.broadcast_mul(advantages)?;
    let clipped = ratio
        .clamp(1.0 - clip_eps, 1.0 + clip_eps)?
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
fn masked_mean_tensor(values: &Tensor, mask: &Tensor, loss_type: LossType) -> CandleResult<Tensor> {
    // Hard-zero masked-out cells (mask <= 0) BEFORE multiplying, so a non-finite
    // value at a padding position cannot leak NaN via `0 * inf` — matching the
    // scalar oracle, which only sums `v * m` where `m > 0`. (Masks are 0/1 by
    // contract; today they are all-ones, but P4 padding makes this load-bearing.)
    let keep = mask.gt(0.0)?;
    let kept = keep.where_cond(values, &values.zeros_like()?)?;
    let masked = kept.broadcast_mul(mask)?;
    match loss_type {
        LossType::Grpo => {
            let per_seq_sum = masked.sum(D::Minus1)?;
            let denom = mask
                .sum(D::Minus1)?
                .maximum(&Tensor::ones_like(&mask.sum(D::Minus1)?)?)?;
            per_seq_sum.broadcast_div(&denom)?.mean(0)
        }
        LossType::DrGrpo => {
            let (num_seq, max_len) = values.dims2()?;
            let total = masked.sum_all()?;
            total.affine(1.0 / (num_seq as f64 * max_len as f64), 0.0)
        }
    }
}

/// Assemble the GRPO loss for one inner step: the negative masked-mean of the
/// per-token objective `surrogate - beta * k3_kl` (the KL term only when a
/// reference policy is supplied). This is the **single** differentiated loss the
/// trainer back-propagates; the trainer's inner step (`item_backward`) calls it on
/// the live policy log-probs and the in-module finite-difference gradcheck
/// (`gradcheck_*`) calls it on a tiny `f64` stand-in, so the gradcheck verifies
/// candle's analytic gradient of *exactly* this expression w.r.t. the `LoRA`
/// parameters.
///
/// `logp` is `[num_seq, comp_len]`; `logp_old` / `logp_ref` share that shape and
/// are detached constants (the only trainable path is `logp`); `advantages` is a
/// detached `[num_seq, 1]` column broadcast over the completion length; `mask` is
/// `[num_seq, comp_len]`. Returns a scalar loss tensor.
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
/// *unconditionally* rather than only fail-loud in the overflow corner.
#[allow(clippy::too_many_arguments)]
fn grpo_loss(
    logp: &Tensor,
    logp_old: &Tensor,
    logp_ref: Option<&Tensor>,
    advantages: &Tensor,
    mask: &Tensor,
    clip_eps: f64,
    beta: f64,
    loss_type: LossType,
) -> CandleResult<Tensor> {
    let keep = mask.gt(0.0)?;
    // At padding, substitute logp_old so the ratio's exp argument is 0 (see the
    // "EOS-padding gradient inertness" note); identical to `logp` where keep == 1.
    let logp_ratio = keep.where_cond(logp, logp_old)?;
    let ratio = importance_ratio(&logp_ratio, logp_old)?;
    let surrogate = clipped_surrogate_tensor(&ratio, advantages, clip_eps)?;
    let per_token = match logp_ref {
        None => surrogate,
        Some(logp_ref) => {
            // At padding, substitute logp_ref so the k3 KL's exp argument is 0.
            let logp_kl = keep.where_cond(logp, logp_ref)?;
            let penalty = k3_kl_tensor(&logp_kl, logp_ref)?.affine(beta, 0.0)?;
            surrogate.broadcast_sub(&penalty)?
        }
    };
    masked_mean_tensor(&per_token, mask, loss_type)?.neg()
}

/// Fraction of valid tokens whose surrogate `min` selected the clipped term.
fn clip_fraction(
    ratio: &Tensor,
    advantages: &Tensor,
    clip_eps: f64,
    mask: &Tensor,
) -> CandleResult<f32> {
    let unclipped = ratio.broadcast_mul(advantages)?;
    let clipped = ratio
        .clamp(1.0 - clip_eps, 1.0 + clip_eps)?
        .broadcast_mul(advantages)?;
    let was_clipped = clipped.lt(&unclipped)?.to_dtype(DType::F32)?;
    let num = scalar_f32(&was_clipped.broadcast_mul(mask)?.sum_all()?)?;
    let den = scalar_f32(&mask.sum_all()?)?;
    Ok(if den > 0.0 { num / den } else { 0.0 })
}

/// Global L2 norm over the trainable vars' gradients (pre-clip). Vars absent
/// from the store contribute `0` (the canary has already guaranteed coverage).
fn global_grad_norm(vars: &[Var], grads: &GradStore) -> CandleResult<f64> {
    let mut sq = 0.0;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            sq += f64::from(scalar_f32(&g.sqr()?.sum_all()?)?);
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
        let got = clipped_surrogate_tensor(&ratio, &adv, eps).unwrap();
        let got = got.to_vec2::<f32>().unwrap();
        let advs = [0.5f64, -0.5];
        let ratios = [1.0f64, 1.5, 0.5];
        for (i, &a) in advs.iter().enumerate() {
            for (j, &rt) in ratios.iter().enumerate() {
                let want = clipped_surrogate(rt, a, eps) as f32;
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
        let got = clipped_surrogate_tensor(&ratio, &adv, 0.2).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(got[0].is_finite(), "0*inf leaked NaN: {}", got[0]);
        assert_relative_eq!(
            got[0],
            clipped_surrogate(f64::INFINITY, 0.0, 0.2) as f32,
            epsilon = TOL
        );
        assert_relative_eq!(
            got[1],
            clipped_surrogate(2.0, 0.5, 0.2) as f32,
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
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::Grpo).unwrap()).unwrap();
        assert_relative_eq!(
            grpo,
            masked_mean(&v, &m, LossType::Grpo) as f32,
            epsilon = TOL
        );

        let dr =
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::DrGrpo).unwrap()).unwrap();
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
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::Grpo).unwrap()).unwrap();
        assert!(grpo.is_finite(), "masked-out NaN/inf leaked: {grpo}");
        assert_relative_eq!(
            grpo,
            masked_mean(&v, &m, LossType::Grpo) as f32,
            epsilon = TOL
        );
        let dr =
            scalar_f32(&masked_mean_tensor(&values, &mask, LossType::DrGrpo).unwrap()).unwrap();
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
        let frac = clip_fraction(&ratio, &adv, 0.2, &mask).unwrap();
        assert_relative_eq!(frac, 0.0, epsilon = TOL);
    }

    #[test]
    fn clip_fraction_counts_binding_tokens() {
        // A>0, ratio 1.5 > 1.2 -> clipped term binds (lower). A<0 row: ratio 1.5
        // -> unclipped is lower, so the clip does NOT bind. 1 of 4 tokens clipped.
        let ratio = mat(&[&[1.5, 1.0], &[1.5, 1.0]]);
        let adv = Tensor::from_vec(vec![0.5f32, -0.5], (2, 1), &cpu()).unwrap();
        let mask = mat(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let frac = clip_fraction(&ratio, &adv, 0.2, &mask).unwrap();
        assert_relative_eq!(frac, 0.25, epsilon = TOL);
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

    #[test]
    fn grad_accum_steps_defaults_to_one_for_old_configs() {
        // A config.json written before grad_accum_steps existed must deserialize to 1
        // (no accumulation), not fail — the serde default keeps old runs loadable.
        let j = r#"{"steps":10,"group_size":8,"max_new_tokens":16,"temperature":1.0,
            "mu":1,"beta":0.0,"clip_eps":0.2,"lr":0.001,"weight_decay":0.0,
            "loss_type":"grpo","scale_rewards":"group"}"#;
        let cfg: TrainerConfig = serde_json::from_str(j).unwrap();
        assert_eq!(cfg.grad_accum_steps, 1);
        // The EOS field also predates the JSON above; serde fills it from the default.
        assert_eq!(cfg.eos_token_id, None);
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
        };
        assert_relative_eq!(mean_completion_len(&r), 2.0, epsilon = TOL);
        // Empty rollout => 0.0 (no divide-by-zero).
        let e = Rollout {
            token_ids: vec![],
            prompt_len: 0,
            completion_lens: vec![],
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
        };
        assert_eq!(completion_dims(&ok).unwrap(), (2, 3));
        // Wrong number of lengths (one length for two sequences).
        let misaligned = Rollout {
            token_ids: vec![vec![0u32; 5], vec![0u32; 5]],
            prompt_len: 2,
            completion_lens: vec![3],
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
        };
        assert!(matches!(
            completion_dims(&overlong),
            Err(TrainerError::Contract(_))
        ));
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
                0.2,
                0.0,
                LossType::Grpo,
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
            0.2,
            0.1, // beta > 0: the k3 KL term is active
            LossType::Grpo,
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
    fn run_gradcheck_with(
        loss_type: LossType,
        beta: f64,
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

        let loss_of = || -> f64 {
            let logp = model.logp().unwrap();
            grpo_loss(
                &logp,
                &logp_old,
                logp_ref.as_ref(),
                adv,
                mask,
                EPS,
                beta,
                loss_type,
            )
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
        };

        // Analytic gradients (extract to Vec before perturbing anything).
        let logp = model.logp().unwrap();
        let loss = grpo_loss(
            &logp,
            &logp_old,
            logp_ref.as_ref(),
            adv,
            mask,
            EPS,
            beta,
            loss_type,
        )
        .unwrap();
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
        let ctx = format!("uniform-mask, loss_type={loss_type:?}, beta={beta}");
        run_gradcheck_with(loss_type, beta, &shift, &adv, &mask, &ctx);
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
        };
        let mask_rows = length_mask_rows(&rollout, t);
        let mask_data: Vec<f64> = mask_rows.iter().flatten().copied().collect();
        let mask = Tensor::from_vec(mask_data, (g, t), dev).unwrap();
        let row_sums: Vec<f64> = mask.sum(D::Minus1).unwrap().to_vec1().unwrap();
        assert_eq!(row_sums, expected, "ragged mask premise");
        run_gradcheck_with(loss_type, beta, shift, adv, &mask, &ctx);
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
}
