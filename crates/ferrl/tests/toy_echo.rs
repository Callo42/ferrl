//! End-to-end P2 gate: a tiny CPU GRPO loop on an *echo* task.
//!
//! A `LoRA`-adapted one-layer LM over a small vocab must learn to copy its
//! prompt symbol. The model is built only from `ferrl`'s public API — a frozen
//! base weight + a `LoraLinear` adapter, a grad-safe `rms_norm_slow` `RmsNorm`,
//! and a categorical sampler — so it exercises all five seams (`RewardFn`,
//! `Policy`, `LoraLinear`, the GRPO math, and the `Trainer`) exactly as a real
//! Qwen policy will at P4. The toy lives in the test crate, not the library, so
//! `ferrl` stays a model-agnostic RL layer.
//!
//! Gates proven here:
//! 1. **reward trends up** — the loop learns the echo map (β = 0, μ = 1);
//! 2. **canary holds on every real update** — a green multi-step run with many
//!    real updates proves it; a negative control (an uncovered trainable var)
//!    shows the canary aborts on the silent-skip landmine;
//! 3. **grad forward == no-grad forward** — the on-tape and detached
//!    log-probs match to float precision (and so the μ = 1 importance ratio is
//!    exactly 1);
//! 4. **μ = 2 / β > 0** — the inner loop, clip, k3 KL, and the adapter-disabled
//!    reference all run on CPU with KL ≥ 0 and the adapter restored afterwards;
//! 5. **μ > 1 / β = 0 completes** — a saturated (fully-clipped) inner step's
//!    legitimately-zero gradient is a no-op, not a canary abort.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, Var, D};
use candle_nn::ops::log_softmax;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use ferrl::lora::LoraLinear;
use ferrl::nn::RmsNorm;
use ferrl::policy::{GenConfig, Policy, Rollout};
use ferrl::reward::RewardFn;
use ferrl::telemetry::RunDir;
use ferrl::trainer::{TokenizerLike, Trainer, TrainerConfig};
use ferrl::Metrics;

/// Toy vocabulary size; the (full-rank) `LoRA` rank equals it, so a rank-`VOCAB`
/// adapter can represent the whole echo map.
const VOCAB: usize = 5;
/// A constant `RmsNorm` gain. Kept moderate so the policy stays soft enough to
/// keep groups diverse while learning (avoids the confident-wrong collapse that a
/// very peaked policy falls into) while still letting reward approach 1.
const GAMMA: f64 = 3.0;
/// Rollout sampling temperature.
const TEMP: f64 = 1.0;

// ---- the toy policy --------------------------------------------------------

/// A one-layer `LoRA` LM over a `vocab`-symbol alphabet. The forward is
/// `one_hot(x) -> LoraLinear -> rms_norm_slow -> logits`, mirroring the P1
/// grad-flow template so the canary is meaningful and grads must cross the norm.
struct EchoPolicy {
    lora: LoraLinear,
    norm: RmsNorm,
    vocab: usize,
    sampler: LogitsProcessor,
    device: Device,
}

impl EchoPolicy {
    fn new(
        vocab: usize,
        rank: usize,
        gamma_scale: f64,
        seed: u64,
        temperature: f64,
    ) -> CandleResult<Self> {
        let device = Device::Cpu;
        // Zero base weight => uniform logits at init => reward starts at ~1/vocab.
        let base = Tensor::zeros((vocab, vocab), DType::F32, &device)?;
        // alpha = rank so the update scale (alpha / rank) is 1.
        let lora = LoraLinear::new(base, None, rank, rank as f64)?;
        // A constant gamma > 1 lifts the post-norm logit scale so the softmax can
        // become peaky enough for the reward to approach 1.
        let gamma = Tensor::ones(vocab, DType::F32, &device)?.affine(gamma_scale, 0.0)?;
        let norm = RmsNorm::new(gamma, 1e-6);
        let sampler = LogitsProcessor::from_sampling(seed, Sampling::All { temperature });
        Ok(Self {
            lora,
            norm,
            vocab,
            sampler,
            device,
        })
    }

    /// Logits `[len, vocab]` for one token sequence.
    fn logits(&self, ids: &[u32]) -> CandleResult<Tensor> {
        let oh = one_hot_batch(
            std::slice::from_ref(&ids.to_vec()),
            ids.len(),
            self.vocab,
            &self.device,
        )?;
        let h = self.lora.forward(&oh)?;
        self.norm.forward(&h)?.squeeze(0)
    }
}

impl Policy for EchoPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let mut token_ids = Vec::with_capacity(cfg.group_size);
        for _ in 0..cfg.group_size {
            let mut ids = prompt.to_vec();
            for _ in 0..cfg.max_new_tokens {
                let logits = self.logits(&ids)?;
                let last = logits.i(ids.len() - 1)?;
                let next = self.sampler.sample(&last)?;
                ids.push(next);
            }
            token_ids.push(ids);
        }
        Ok(Rollout {
            token_ids,
            prompt_len: prompt.len(),
        })
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let prompt_len = rollout.prompt_len;
        let seq_len = rollout.token_ids[0].len();
        let comp_len = seq_len - prompt_len;
        let input_len = seq_len - 1;
        // Teacher forcing: forward all but the last token, read the positions that
        // predict the completion tokens.
        let oh = one_hot_batch(&rollout.token_ids, input_len, self.vocab, &self.device)?;
        let h = self.lora.forward(&oh)?;
        let logits = self.norm.forward(&h)?;
        let pred = logits.narrow(1, prompt_len - 1, comp_len)?;
        let logp = log_softmax(&pred, D::Minus1)?;
        let targets = targets_tensor(&rollout.token_ids, prompt_len, comp_len, &self.device)?;
        let idx = targets.unsqueeze(D::Minus1)?;
        logp.gather(&idx, D::Minus1)?.squeeze(D::Minus1)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.lora.set_enabled(enabled);
    }

    fn adapter_enabled(&self) -> bool {
        self.lora.is_enabled()
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.lora.trainable_vars()
    }
}

/// One-hot encode the first `input_len` tokens of each sequence into a
/// `[num_seq, input_len, vocab]` `f32` tensor.
fn one_hot_batch(
    seqs: &[Vec<u32>],
    input_len: usize,
    vocab: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    let g = seqs.len();
    let mut data = vec![0f32; g * input_len * vocab];
    for (i, ids) in seqs.iter().enumerate() {
        for t in 0..input_len {
            data[(i * input_len + t) * vocab + ids[t] as usize] = 1.0;
        }
    }
    Tensor::from_vec(data, (g, input_len, vocab), device)
}

/// The completion target ids as a `[num_seq, comp_len]` `u32` tensor.
fn targets_tensor(
    seqs: &[Vec<u32>],
    prompt_len: usize,
    comp_len: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    let g = seqs.len();
    let mut data = vec![0u32; g * comp_len];
    for (i, ids) in seqs.iter().enumerate() {
        for j in 0..comp_len {
            data[i * comp_len + j] = ids[prompt_len + j];
        }
    }
    Tensor::from_vec(data, (g, comp_len), device)
}

// ---- tokenizer + reward ----------------------------------------------------

/// Trivial codec mapping `'a'..` to ids `0..`.
struct CharTokenizer;

impl TokenizerLike for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| u32::from(c) - u32::from('a'))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&i| char::from_u32(u32::from('a') + i))
            .collect()
    }
}

/// Verifiable echo reward: `1.0` iff the completion's first symbol equals the
/// prompt's first symbol.
struct EchoReward;

impl RewardFn for EchoReward {
    fn reward(&self, prompt: &str, completion: &str) -> f32 {
        match (prompt.chars().next(), completion.chars().next()) {
            (Some(p), Some(c)) if p == c => 1.0,
            _ => 0.0,
        }
    }
}

// ---- a negative control: an uncovered trainable var ------------------------

/// Wraps [`EchoPolicy`] but reports an extra trainable [`Var`] that never reaches
/// the loss, so the grad-coverage canary must abort the run.
struct UncoveredPolicy {
    inner: EchoPolicy,
    dangling: Var,
}

impl Policy for UncoveredPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
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
        let mut vars = self.inner.trainable_vars();
        vars.push(self.dangling.clone());
        vars
    }
}

// ---- helpers ---------------------------------------------------------------

/// `'a'..` prompts, one per vocab symbol, each a single token.
fn echo_prompts(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| char::from(b'a' + i as u8).to_string())
        .collect()
}

/// Mean reward over a slice of step metrics.
fn window_mean(ms: &[Metrics]) -> f32 {
    ms.iter().map(|m| m.reward_mean).sum::<f32>() / ms.len() as f32
}

/// A unique temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("ferrl-toy-{tag}-{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---- gates -----------------------------------------------------------------

#[test]
fn gate_reward_trends_up() {
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 7, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 300,
        group_size: 32,
        max_new_tokens: 1,
        temperature: TEMP,
        lr: 0.05,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("reward-up");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();

    // Compare an early window to a late window: the loop learns the echo map, so
    // the late-phase reward is far above the early phase and near the ceiling.
    // (Deterministic: the policy is seeded, so the trajectory is reproducible.)
    let early = window_mean(&history[..40]);
    let late = window_mean(&history[history.len() - 40..]);
    assert!(
        late > early + 0.25,
        "reward did not trend up: early-40 mean={early}, late-40 mean={late}"
    );
    assert!(late > 0.85, "final reward too low: late-40 mean={late}");
}

#[test]
fn gate_canary_holds_on_every_real_update() {
    // The canary is a hard error (missing var / non-finite gradient) on every real
    // update, so a completed run with many real updates proves it held on all of
    // them. Degenerate zero-advantage groups perform no update and run no canary —
    // canary_aborts_when_a_trainable_var_is_uncovered proves its teeth separately.
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 5, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 60,
        group_size: 32,
        max_new_tokens: 1,
        temperature: TEMP,
        lr: 0.05,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("canary");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();

    assert_eq!(history.len(), 60);
    for m in &history {
        assert!(
            m.grad_norm.is_finite(),
            "non-finite grad_norm at step {}",
            m.step
        );
        assert!(m.reward_mean.is_finite());
    }
    // The run is not vacuous: a substantial number of steps were real (non-
    // degenerate) updates that actually exercised the canary.
    let real_updates = history
        .iter()
        .filter(|m| m.frac_reward_zero_std < 0.5)
        .count();
    assert!(
        real_updates >= 15,
        "too few real updates exercised the canary: {real_updates}/60"
    );
}

#[test]
fn canary_aborts_when_a_trainable_var_is_uncovered() {
    // An extra trainable Var that never reaches the loss must be caught by the
    // canary (candle would otherwise silently skip it), aborting the run.
    let inner = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 1, TEMP).unwrap();
    let dangling = Var::zeros((1,), DType::F32, &Device::Cpu).unwrap();
    let mut policy = UncoveredPolicy { inner, dangling };
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 1,
        group_size: 8,
        max_new_tokens: 1,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("canary-neg");
    let run = RunDir::create(tmp.path(), "broken").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let res = trainer.train(&mut policy, &EchoReward, &CharTokenizer, &prompts);
    assert!(
        res.is_err(),
        "canary must abort when a trainable var never reaches the loss"
    );
}

#[test]
fn gate_grad_path_equals_nograd_path() {
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 3, TEMP).unwrap();
    let cfg = GenConfig {
        group_size: 8,
        max_new_tokens: 1,
        temperature: TEMP,
    };
    let rollout = policy.generate(&[0u32], &cfg).unwrap();

    let logp_grad = policy.token_logprobs(&rollout).unwrap();
    let logp_nograd = policy.token_logprobs(&rollout).unwrap().detach();

    let max_diff: f32 = logp_grad
        .broadcast_sub(&logp_nograd)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        max_diff <= 1e-6,
        "grad/no-grad logprobs diverged: {max_diff}"
    );

    // μ = 1: ratio = exp(logp - logp_old) with logp_old the detached snapshot, so
    // it must be exactly 1 (the clip is wired but inert).
    let ratio = logp_grad
        .broadcast_sub(&logp_nograd)
        .unwrap()
        .exp()
        .unwrap();
    let max_r_dev: f32 = ratio
        .broadcast_sub(&Tensor::ones_like(&ratio).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        max_r_dev <= 1e-6,
        "μ=1 importance ratio not 1: dev={max_r_dev}"
    );
}

#[test]
fn gate_mu2_beta_positive_run() {
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 11, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 20,
        group_size: 16,
        max_new_tokens: 1,
        temperature: TEMP,
        mu: 2,
        beta: 0.05,
        lr: 0.05,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("mu2-beta");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();

    // The reference forward toggled the adapter off then restored it.
    assert!(
        policy.adapter_enabled(),
        "adapter not restored after the reference forward"
    );

    for m in &history {
        assert!(m.kl >= 0.0, "k3 KL must be non-negative, got {}", m.kl);
        assert!(m.kl.is_finite(), "non-finite KL at step {}", m.step);
    }
    let max_kl = history.iter().map(|m| m.kl).fold(0.0f32, f32::max);
    assert!(
        max_kl > 0.0,
        "KL never became positive; the reference path may be inert"
    );
}

#[test]
fn gate_mu_gt1_beta_zero_completes() {
    // mu>1 with beta=0: once the PPO clip saturates (every token clipped) the
    // gradient is legitimately exactly zero. That must be a no-op inner step, NOT
    // a canary abort — a high lr forces saturation within a few steps. (Before the
    // liveness fix this run aborted with "every gradient is zero".)
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 13, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 40,
        group_size: 32,
        max_new_tokens: 1,
        temperature: TEMP,
        mu: 3,
        beta: 0.0,
        clip_eps: 0.2,
        lr: 0.3,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("mu3-beta0");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();
    assert_eq!(history.len(), 40);
    for m in &history {
        assert!(m.grad_norm.is_finite() && m.clip_ratio.is_finite());
    }
}
