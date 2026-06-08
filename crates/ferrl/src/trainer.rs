//! The GRPO trainer — the fifth and final seam.
//!
//! [`Trainer`] drives one GRPO optimizer step at a time over a [`Policy`] and a
//! [`RewardFn`], owning the pieces candle does not provide: the rollout →
//! reward → advantage → masked clipped-surrogate (+ optional KL) → backward →
//! **grad-coverage canary** → optimizer-step pipeline, plus the inner update
//! loop (`μ`) and per-step telemetry.
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
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use serde::{Deserialize, Serialize};

use crate::grpo::{group_advantages, zero_mask_rows, LossType, ScaleRewards};
use crate::nn::grad_coverage;
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
    /// If set, write an adapter checkpoint to `checkpoints/step-<n>/` (with a
    /// resumable manifest) every `checkpoint_every` completed steps **and** after
    /// the final step (so a completed run always persists its final adapter, even
    /// when this does not divide `steps`). `None` (the default) disables
    /// checkpointing entirely. `#[serde(default)]` so a `config.json` written before
    /// this field existed still deserializes (to `None`).
    #[serde(default)]
    pub checkpoint_every: Option<u64>,
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
            checkpoint_every: None,
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
    /// Returns [`TrainerError::InvalidConfig`] if `mu`, `group_size`, or
    /// `max_new_tokens` is `0`; if `temperature` is not finite and `> 0`; if `lr`,
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

    /// Run `config.steps` GRPO steps, cycling through `prompts`, returning the
    /// per-step [`Metrics`] (also appended to `metrics.jsonl`).
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
    /// global step index, so resuming at the recorded step keeps the same
    /// prompt order an uninterrupted run would have seen.
    ///
    /// **Not bit-exact to an uninterrupted run.** A fresh `AdamW` is constructed
    /// (its moment estimates restart from zero, re-warming the bias correction) and
    /// the policy's sampler RNG is whatever the reloaded policy carries; neither is
    /// persisted by [`crate::checkpoint`] (candle exposes no accessor for them).
    /// The reloaded *adapter weights* are exact; the post-resume trajectory is a
    /// faithful continuation, not a replay. (Momentum-faithful resume is deferred
    /// to P5.)
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
    /// run steps `start_step .. config.steps`, checkpointing on the configured
    /// cadence.
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
        let mut opt = AdamW::new(vars.clone(), params)?;
        let total = self.config.steps;
        let remaining = total.saturating_sub(start_step) as usize;
        let mut history = Vec::with_capacity(remaining);
        for step in start_step..total {
            let prompt = &prompts[step as usize % prompts.len()];
            let m = self.run_step(step, policy, reward_fn, tokenizer, prompt, &mut opt, &vars)?;
            self.writer.append(&m)?;
            history.push(m);
            self.maybe_checkpoint(step, &vars)?;
        }
        Ok(history)
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

    /// One outer GRPO step over a single prompt's group.
    #[allow(clippy::too_many_arguments)]
    fn run_step<P: Policy, R: RewardFn>(
        &self,
        step: u64,
        policy: &mut P,
        reward_fn: &R,
        tokenizer: &dyn TokenizerLike,
        prompt: &str,
        opt: &mut AdamW,
        vars: &[Var],
    ) -> Result<Metrics, TrainerError> {
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

        // Rectangular toy: an all-ones mask drops no rows. zero_mask_rows is still
        // the source of truth (variable-length masking arrives with P4).
        let mask_rows = vec![vec![1.0_f64; comp_len]; rollout.len()];
        let dropped = zero_mask_rows(&mask_rows);

        // Group-normalized advantages (scalar oracle). A group whose advantages are
        // all exactly zero — no reward spread, or non-finite rewards forced to a 0
        // advantage — carries no gradient, so the update (and the canary, which
        // would correctly flag the resulting all-zero gradient) is skipped.
        let rewards_f64: Vec<f64> = rewards.iter().map(|&r| f64::from(r)).collect();
        let advantages = group_advantages(&rewards_f64, self.config.scale_rewards);
        let degenerate = advantages.iter().all(|a| *a == 0.0);
        let agg = if degenerate {
            InnerAgg::default()
        } else {
            self.update_group(policy, &rollout, &advantages, comp_len, vars, opt)?
        };

        Ok(self.build_metrics(step, &rewards, &rollout, dropped, degenerate, &agg, opt))
    }

    /// Run the `mu` inner updates for a group that has a non-zero advantage
    /// signal: snapshot the old / reference log-probs, then optimize.
    #[allow(clippy::too_many_arguments)]
    fn update_group<P: Policy>(
        &self,
        policy: &mut P,
        rollout: &Rollout,
        advantages: &[f64],
        comp_len: usize,
        vars: &[Var],
        opt: &mut AdamW,
    ) -> Result<InnerAgg, TrainerError> {
        let logp_old = policy.token_logprobs(rollout)?.detach();
        let device = logp_old.device().clone();
        let mask = Tensor::ones((rollout.len(), comp_len), DType::F32, &device)?;
        let logp_ref = self.reference_logprobs(policy, rollout)?;
        let advantages = advantages_tensor(advantages, &device)?;

        // mu inner updates; the last one's diagnostics land in the metrics.
        let mut agg = InnerAgg::default();
        for _ in 0..self.config.mu {
            agg = self.inner_update(
                policy,
                rollout,
                &logp_old,
                logp_ref.as_ref(),
                &advantages,
                &mask,
                vars,
                opt,
            )?;
        }
        Ok(agg)
    }

    /// One inner optimization step: grad forward, surrogate (+ optional KL),
    /// masked aggregation, backward, the canary, then the optimizer step.
    #[allow(clippy::too_many_arguments)]
    fn inner_update<P: Policy>(
        &self,
        policy: &P,
        rollout: &Rollout,
        logp_old: &Tensor,
        logp_ref: Option<&Tensor>,
        advantages: &Tensor,
        mask: &Tensor,
        vars: &[Var],
        opt: &mut AdamW,
    ) -> Result<InnerAgg, TrainerError> {
        let logp = policy.token_logprobs(rollout)?;
        let ratio = importance_ratio(&logp, logp_old)?;
        let surrogate = clipped_surrogate_tensor(&ratio, advantages, self.config.clip_eps)?;
        let (per_token, kl) = self.apply_kl(&surrogate, &logp, logp_ref, mask)?;
        let clip_frac = clip_fraction(&ratio, advantages, self.config.clip_eps, mask)?;

        let objective = masked_mean_tensor(&per_token, mask, self.config.loss_type)?;
        let loss = objective.neg()?;

        let grads = loss.backward()?;
        let cov = grad_coverage(vars, &grads)?;
        // Fatal: a missing var (candle's silent-skip landmine — a real autograd cut
        // is an absent grad entry) or a non-finite gradient (a blowup). into_result
        // emits the precise message for whichever fired.
        if !cov.is_covered() || cov.nonfinite > 0 {
            cov.clone().into_result()?;
        }
        if !cov.is_live() {
            // Covered + finite + all-zero gradient: no usable signal this inner step
            // (fully-clipped trust region, or mean-centered advantages cancelling).
            // Skip the optimizer step rather than mislabel it a dead forward.
            return Ok(InnerAgg {
                kl,
                clip_frac,
                grad_norm: 0.0,
            });
        }
        let grad_norm = global_grad_norm(vars, &grads)? as f32;
        opt.step(&grads)?;

        Ok(InnerAgg {
            kl,
            clip_frac,
            grad_norm,
        })
    }

    /// Subtract `beta * k3_kl` from the surrogate when a reference is present.
    fn apply_kl(
        &self,
        surrogate: &Tensor,
        logp: &Tensor,
        logp_ref: Option<&Tensor>,
        mask: &Tensor,
    ) -> CandleResult<(Tensor, f32)> {
        let Some(logp_ref) = logp_ref else {
            return Ok((surrogate.clone(), 0.0));
        };
        let kl = k3_kl_tensor(logp, logp_ref)?;
        let kl_mean = scalar_f32(&masked_mean_tensor(&kl, mask, self.config.loss_type)?)?;
        let penalty = kl.affine(self.config.beta, 0.0)?;
        let per_token = surrogate.broadcast_sub(&penalty)?;
        Ok((per_token, kl_mean))
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

    /// Populate the step's [`Metrics`] from the rewards, rollout, and inner-step
    /// diagnostics.
    #[allow(clippy::too_many_arguments)]
    fn build_metrics(
        &self,
        step: u64,
        rewards: &[f32],
        rollout: &Rollout,
        dropped: usize,
        degenerate: bool,
        agg: &InnerAgg,
        opt: &AdamW,
    ) -> Metrics {
        let mut m = Metrics::at_step(step);
        let (mean, std) = reward_stats(rewards);
        m.reward_mean = mean;
        m.reward_std = std;
        // Tie the zero-std flag to the same condition that drove the skip, so the
        // metric and the optimizer can never disagree (covers all-non-finite groups
        // too, which group_advantages forces to all-zero advantages).
        m.frac_reward_zero_std = if degenerate { 1.0 } else { 0.0 };
        m.completion_len = mean_completion_len(rollout);
        m.dropped_rows = dropped as u32;
        m.kl = agg.kl;
        m.clip_ratio = agg.clip_frac;
        m.grad_norm = agg.grad_norm;
        m.lr = opt.learning_rate() as f32;
        m
    }
}

/// Decode each completion (the tokens after the shared prompt) to text.
fn decode_completions(rollout: &Rollout, tokenizer: &dyn TokenizerLike) -> Vec<String> {
    rollout
        .token_ids
        .iter()
        .map(|ids| tokenizer.decode(&ids[rollout.prompt_len..]))
        .collect()
}

/// The scalar group advantages as a detached `[G, 1]` tensor (broadcast over the
/// completion length in the surrogate).
fn advantages_tensor(advantages: &[f64], device: &Device) -> CandleResult<Tensor> {
    let adv: Vec<f32> = advantages.iter().map(|&a| a as f32).collect();
    Tensor::from_vec(adv, (advantages.len(), 1), device)
}

/// Validate that the rollout is rectangular with non-empty completions, and
/// return `(num_seq, completion_len)`. The rectangular shape is required because
/// [`Policy::token_logprobs`] returns a rectangular `[G, completion_len]` tensor;
/// variable-length (ragged) completions arrive with P4. Run this **before**
/// decoding so a malformed rollout becomes a typed [`TrainerError::Contract`]
/// rather than a slice panic.
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
    Ok((rollout.len(), comp_len))
}

/// Mean completion length (tokens after the prompt) over the rollout.
fn mean_completion_len(rollout: &Rollout) -> f32 {
    if rollout.is_empty() {
        return 0.0;
    }
    let total: usize = rollout
        .token_ids
        .iter()
        .map(|ids| ids.len().saturating_sub(rollout.prompt_len))
        .sum();
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
        bad(|c| c.checkpoint_every = Some(0));
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
}
