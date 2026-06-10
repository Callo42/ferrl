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

use ferrl::lora::LoraLinear;
use ferrl::nn::RmsNorm;
use ferrl::policy::{GenConfig, Policy, Rollout};
use ferrl::reward::RewardFn;
use ferrl::sampler::GrpoSampler;
use ferrl::telemetry::RunDir;
use ferrl::trainer::{TokenizerLike, Trainer, TrainerConfig, TrainerError};
use ferrl::{LossType, Metrics, ScaleRewards};

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
    sampler: GrpoSampler,
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
        let sampler = GrpoSampler::new(seed, temperature);
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
        Ok(Rollout::rectangular(token_ids, prompt.len()))
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

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.sampler.to_state_bytes()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.sampler = GrpoSampler::from_state_bytes(state)?;
        Ok(())
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

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
    }
}

// ---- contract-violation controls (malformed Policy / RewardFn output) ------

/// Wraps [`EchoPolicy`] but overrides the rollout's `prompt_len` with a malformed
/// value, so the trainer must reject it with a typed error instead of panicking
/// downstream. `prompt_len` larger than the sequence ⇒ no completion tokens;
/// `prompt_len == 0` ⇒ no prompt context (teacher forcing reads index -1).
struct BadPromptLenPolicy {
    inner: EchoPolicy,
    prompt_len: usize,
}

impl Policy for BadPromptLenPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let r = self.inner.generate(prompt, cfg)?;
        Ok(Rollout::rectangular(r.token_ids, self.prompt_len))
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
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
    }
}

/// Wraps [`EchoPolicy`] and rewrites each completion to be **EOS-padded**: keep the
/// first `real_lens[i % len]` sampled tokens of sequence `i`, overwrite the rest with
/// `pad`, and record that length in `completion_lens` — exactly the shape EOS
/// early-stop produces, with a **variable per-row** length so the trainer's
/// length-aware mask is genuinely per-sequence (not a uniform column).
/// `token_logprobs` delegates to the inner policy (it scores the full rectangular
/// width; the trainer's length-aware mask is what excludes the padding from the
/// loss). Lets a test drive the real `Trainer` with `eos_token_id` set and confirm
/// the loss mask + length-aware decode honor `completion_lens` end-to-end.
struct EosPaddedEchoPolicy {
    inner: EchoPolicy,
    real_lens: Vec<usize>,
    pad: u32,
}

impl EosPaddedEchoPolicy {
    /// The real length recorded for sequence `i`, cycling `real_lens` and capped at
    /// the available completion width.
    fn real_len(&self, i: usize, max_real: usize) -> usize {
        self.real_lens[i % self.real_lens.len()].min(max_real)
    }
}

impl Policy for EosPaddedEchoPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let r = self.inner.generate(prompt, cfg)?;
        let prompt_len = r.prompt_len;
        let max_real = r.token_ids[0].len() - prompt_len;
        let mut completion_lens = Vec::with_capacity(r.token_ids.len());
        let token_ids: Vec<Vec<u32>> = r
            .token_ids
            .into_iter()
            .enumerate()
            .map(|(i, mut ids)| {
                let real = self.real_len(i, max_real);
                completion_lens.push(real);
                for slot in ids.iter_mut().skip(prompt_len + real) {
                    *slot = self.pad;
                }
                ids
            })
            .collect();
        Ok(Rollout {
            token_ids,
            prompt_len,
            completion_lens,
        })
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
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
    }
}

/// A reward whose `reward_group` returns the wrong number of scores.
struct BadCountReward;

impl RewardFn for BadCountReward {
    fn reward(&self, _prompt: &str, _completion: &str) -> f32 {
        0.0
    }
    fn reward_group(&self, _prompt: &str, completions: &[String]) -> Vec<f32> {
        vec![0.0; completions.len().saturating_sub(1)]
    }
}

/// Wraps [`EchoPolicy`] but underfills the rollout (one completion instead of
/// `group_size`), violating the `generate` contract.
struct UnderfilledPolicy {
    inner: EchoPolicy,
}

impl Policy for UnderfilledPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let mut r = self.inner.generate(prompt, cfg)?;
        r.token_ids.truncate(1);
        Ok(r)
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
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
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

/// Flatten each trainable var to a `Vec<f32>` for bit-exact comparison.
fn snapshot_vars(vars: &[Var]) -> Vec<Vec<f32>> {
    vars.iter()
        .map(|v| {
            v.as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .collect()
}

/// Assert two metrics streams are bit-identical on the fields a faithful resume must
/// reproduce: the step index, the reward (so the rollout RNG matched), the gradient norm
/// (so weights + momentum matched), and the per-window degeneracy pattern.
fn assert_metrics_bit_identical(got: &[Metrics], want: &[Metrics]) {
    assert_eq!(got.len(), want.len(), "post-resume metrics length mismatch");
    for (i, (r, f)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            r.step, f.step,
            "step index misaligned at post-resume step {i}"
        );
        assert_eq!(
            r.reward_mean, f.reward_mean,
            "reward_mean diverged at post-resume step {i} (rollout RNG not restored?)"
        );
        assert_eq!(
            r.grad_norm, f.grad_norm,
            "grad_norm diverged at post-resume step {i} (weights/momentum not restored?)"
        );
        assert_eq!(
            r.frac_reward_zero_std, f.frac_reward_zero_std,
            "degeneracy pattern diverged at post-resume step {i}"
        );
    }
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
    // the late-phase reward is well above the early phase and far above the ~1/V
    // (=0.2) baseline. The thresholds carry wide margins on purpose — the policy
    // is seeded, but the exact converged value varies across CPUs (float
    // non-associativity in matmul/softmax shifts the sampled trajectory: ~0.99 on
    // one host, ~0.80 on a CI runner). The upward trend itself is robust.
    let early = window_mean(&history[..40]);
    let late = window_mean(&history[history.len() - 40..]);
    assert!(
        late > early + 0.2,
        "reward did not trend up: early-40 mean={early}, late-40 mean={late}"
    );
    assert!(late > 0.5, "final reward too low: late-40 mean={late}");
}

#[test]
fn eos_padded_rollout_trains_with_length_aware_mask() {
    // The trainer consumes completion_lens end-to-end with a VARIABLE per-row length:
    // a `Some` eos_token_id no longer errors (the PR3 guard-lift), each sequence keeps a
    // different number of real tokens (lengths 1 and 2, alternating), and the run steps
    // cleanly through a genuinely per-sequence masked backward — the EOS padding (scored
    // by token_logprobs but masked out) stays out of the loss. The completion_len metric
    // reports the mean real length (1.5), not the padded width (3). The mask's per-token
    // gradient inertness — including the exp-overflow corner — is pinned unit-side by
    // `padding_columns_are_inert_in_the_grpo_loss` and
    // `padding_with_an_exp_overflowing_logp_gap_stays_grad_finite`; the dedicated
    // variable-length finite-difference gradcheck lands in PR4.
    let inner = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 7, TEMP).unwrap();
    let mut policy = EosPaddedEchoPolicy {
        inner,
        real_lens: vec![1, 2], // per-row lengths: rows alternate 1 and 2 real tokens
        pad: 0,
    };
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 5,
        group_size: 8, // even -> four rows of length 1, four of length 2 -> mean 1.5
        max_new_tokens: 3, // padded width 3
        temperature: TEMP,
        eos_token_id: Some(0), // accepted now; the mask honors completion_lens
        lr: 0.05,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("eos-padded");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();

    assert_eq!(history.len(), 5);
    for m in &history {
        assert!(
            m.reward_mean.is_finite() && m.grad_norm.is_finite(),
            "non-finite metric at step {}",
            m.step
        );
        // Mean of the per-row real lengths (1 and 2), not the padded width (3).
        assert!(
            (m.completion_len - 1.5).abs() < 1e-6,
            "completion_len {} != mean real length 1.5 at step {}",
            m.completion_len,
            m.step
        );
    }
}

#[test]
fn gate_dr_grpo_paper_config_learns() {
    // The Dr.GRPO *paper* config — the DrGrpo reduction AND ScaleRewards::None
    // (centered-only advantages) — driven end-to-end through Trainer::train for the
    // first time (both non-default variants are otherwise only tensor-/oracle-unit
    // tested). It must learn the echo map.
    //
    // Honest scope: for this toy the DrGrpo reduction is *numerically identical* to
    // classic Grpo. The two diverge only on ragged / padded masks, and this toy always
    // produces rectangular, all-ones masks (the trainer rejects ragged rollouts, and
    // this policy generates no EOS, so its length-aware mask is all-ones — the
    // EOS-padded variant is exercised by `eos_padded_rollout_trains_with_length_aware_mask`).
    // So what this gate uniquely proves is
    // (a) ScaleRewards::None — the variant that genuinely changes the trajectory —
    // learns through the real loop, and (b) the DrGrpo config path runs end-to-end
    // without error. The reductions' distinct denominators are pinned where they
    // actually differ — the tensor test `masked_mean_tensor_matches_scalar_oracle_*`
    // and the gradcheck `gradcheck_dr_grpo_with_kl`, both on ragged masks.
    //
    // Wide margins on purpose — the trajectory is seeded but platform-dependent (float
    // non-associativity), like the other reward-trend gates. lr is kept modest (low
    // overshoot risk on a numerically-different CI CPU); the un-std-scaled advantages
    // already converge near 1 on the dev host, so `late > 0.5` keeps ample slack.
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 17, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 500,
        group_size: 32,
        max_new_tokens: 1,
        temperature: TEMP,
        lr: 0.1,
        loss_type: LossType::DrGrpo,
        scale_rewards: ScaleRewards::None,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("drgrpo-paper");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();

    let early = window_mean(&history[..40]);
    let late = window_mean(&history[history.len() - 40..]);
    assert!(
        late > early + 0.2,
        "Dr.GRPO paper config did not learn: early-40 mean={early}, late-40 mean={late}"
    );
    assert!(
        late > 0.5,
        "Dr.GRPO paper config final reward too low: late-40 mean={late}"
    );
}

#[test]
fn gate_grad_accum_effective_batch_learns() {
    // Gradient accumulation across prompts: group_size 4 with grad_accum_steps 8 forms
    // an effective batch of 32 completions per optimizer step (the lever the Countdown
    // run wanted to escape degenerate group-4 windows) — each optimizer step folds eight
    // prompts' group-4 gradients into one AdamW update. The AT-SCALE accumulation learning
    // gate: the effective batch (4 * 8 = 32) matches the WIDE-MARGIN group-32 learning gates
    // (gate_reward_trends_up / gate_dr_grpo) — `late` lands in [0.80, 1.0] vs the 0.5 floor —
    // so it learns the echo map with ample margin (it shares those gates' rare, pre-existing
    // contention flakiness — a separate known issue, the P2 lesson — not flake-proof). The
    // smaller two-prompt window (effective batch 8) keeps its own non-flaky mechanism
    // coverage in `gate_grad_accum_two_prompt_window`.
    //
    // A *small* effective batch is what lets a group-4 run land in a CPU-dependent weak
    // optimum (the P2 platform-dependence lesson — float non-associativity, dev host !=
    // CI; at grad_accum_steps 2 this config plateaued ~0.59 on a CI runner under the
    // P6-B Xoshiro swap), which is why this gate is dialed up to the robust batch size.
    // lr stays at the proven-safe 0.05. The learning signal is a single fixed floor
    // `late > 0.5` (~0.3 above the ~1/VOCAB untrained baseline): at this effective batch
    // `late` lands in [0.80, 1.0] across the verification runs, so the floor carries
    // ample slack — no fragile step-0 trend (the pattern the review flagged) needed.
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 29, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 500,
        group_size: 4,
        grad_accum_steps: 8,
        max_new_tokens: 1,
        temperature: TEMP,
        lr: 0.05,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("grad-accum");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();
    assert_eq!(
        history.len(),
        500,
        "one metrics row per optimizer step (window)"
    );
    let late = window_mean(&history[history.len() - 40..]);
    assert!(
        late > 0.5,
        "grad-accum did not learn: late-40 mean={late} (untrained ~= 0.2)"
    );
}

#[test]
fn gate_grad_accum_two_prompt_window() {
    // The *small-window* accumulation contract, kept DISTINCT from the robust at-scale
    // learning gate above (which dials the effective batch up to 32). Here
    // grad_accum_steps = 2 over group_size 4 forms an effective batch of 8 — the
    // SMALLEST accumulation window — and each optimizer step folds TWO prompts' group-4
    // gradients (each scaled 1/2) into one AdamW update. This preserves the original P5
    // two-prompt coverage that raising the at-scale gate to effective batch 32 would
    // otherwise erase.
    //
    // This gate asserts the MECHANISM, not a converged-optimum learning LEVEL. At this
    // small effective batch the converged reward is float-/contention-dependent (the P2
    // platform lesson: float non-associativity; dev host != CI): across 200+ full-suite
    // samples the last-40 mean ranged ~0.39..1.0, and a fixed `late` floor flakes under
    // heavy parallel-suite load (it dipped to 0.394 once the heavier resume gate joined
    // the suite). So the learning OUTCOME is left to the wide-margin at-scale sibling
    // `gate_grad_accum_effective_batch_learns` (effective batch 32; note: no test asserts a
    // learning level at THIS effective-batch-8 regime — a deliberate trade-off, since that
    // level is the contention-fragile quantity). The numeric grad FOLD (summing gradients
    // across separate backwards) is unit-pinned by
    // `fold_var_grads_sums_gradients_across_backwards`. What this gate proves is
    // contention-ROBUST and specific to a genuine 2-prompt window: the windowing path
    // runs to completion, every metric is finite, a real half-degenerate two-prompt
    // window occurs (impossible at N=1), and real accumulated AdamW updates happen —
    // all EARLY-training quantities (diverse groups at uniform init). Measured floors
    // under the full 16-test suite (incl. 3x concurrent load): real updates >= 27,
    // half-degenerate windows >= 26, so the `>= 10` / `> 0` thresholds carry wide slack.
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 29, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 500,
        group_size: 4,
        grad_accum_steps: 2,
        max_new_tokens: 1,
        temperature: TEMP,
        lr: 0.05,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("grad-accum-2");
    let run = RunDir::create(tmp.path(), "echo").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();

    assert_eq!(
        history.len(),
        500,
        "one metrics row per optimizer step (the loop ran to completion)"
    );
    for m in &history {
        assert!(
            m.reward_mean.is_finite() && m.grad_norm.is_finite(),
            "non-finite metric at step {}",
            m.step
        );
    }
    // N=2 witness: a window whose two folded prompts split exactly one degenerate / one
    // live (frac_reward_zero_std == 0.5) can ONLY occur at window size >= 2 — at
    // grad_accum_steps = 1 that fraction is 0 or 1. So this proves the window genuinely
    // folded two prompts and did not collapse to a single-prompt step. Robustly present:
    // 26..=109 such windows per run across the measurements.
    let half_degenerate_windows = history
        .iter()
        .filter(|m| (m.frac_reward_zero_std - 0.5).abs() < 1e-6)
        .count();
    assert!(
        half_degenerate_windows > 0,
        "no half-degenerate 2-prompt window seen — accumulation may have collapsed to N=1"
    );
    // The windows actually accumulated and stepped: a real AdamW update has grad_norm > 0
    // (a window steps if >= 1 of its 2 prompts is non-degenerate). Wide margin — min
    // observed 27/500; the exact count is trajectory- (so platform-) dependent.
    let real_updates = history.iter().filter(|m| m.grad_norm > 0.0).count();
    assert!(
        real_updates >= 10,
        "too few real accumulated updates: {real_updates}/500"
    );
}

#[test]
fn interrupted_run_resumes_bit_identically() {
    // THE P6-B capstone gate: a run interrupted at INTERRUPT_AT and resumed from a
    // momentum-faithful (v2) checkpoint reproduces the uninterrupted run's post-resume
    // trajectory BIT-FOR-BIT. `Trainer::resume` restores the adapter weights, the
    // optimizer moments + step counter, AND the sampler RNG, so every post-resume window
    // samples the same tokens, takes the same gradient, and applies the same AdamW step.
    //
    // Determinism note: the *absolute* trajectory is NOT reproducible across processes at
    // a fixed seed — candle's backprop walks a `HashMap<TensorId, _>` whose per-process
    // `RandomState` seeds the f32 reduction order, and non-associative addition then
    // varies the gradient. This gate does not rely on cross-process determinism: it
    // compares the uninterrupted and resumed runs WITHIN ONE PROCESS (a single reduction
    // order), so their *relative* bit-equality is exactly what proves the restore faithful.
    //
    // group_size 32 + an EARLY interrupt (step 1) keep real AdamW updates firing
    // post-resume: without a real update the weight delta is moment-INDEPENDENT (final
    // weights == checkpoint weights regardless of the moments) and the gate would be
    // vacuous w.r.t. moment restoration. A window is degenerate per-PROMPT, so the
    // post-resume window spans 5 prompts (steps 1..6, one per echo symbol) — needing ALL
    // five groups all-same to be vacuous, which the early diverse regime never does.
    // Pre-checkpoint, step 0 at uniform init has every group diverse -> non-zero moments
    // to restore. A guard below still fails loud if the post-resume window is degenerate.
    const TOTAL: u64 = 6;
    const INTERRUPT_AT: u64 = 1;
    let prompts = echo_prompts(VOCAB);
    let make_cfg = || TrainerConfig {
        steps: TOTAL,
        group_size: 32,
        max_new_tokens: 1,
        temperature: TEMP,
        lr: 0.05,
        checkpoint_every: Some(INTERRUPT_AT), // a v2 checkpoint lands at INTERRUPT_AT (and the final step)
        ..TrainerConfig::default()
    };

    // Uninterrupted reference: train all TOTAL steps; record per-step metrics + final weights.
    let tmp = TempDir::new("resume-faithful");
    let mut policy_full = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 29, TEMP).unwrap();
    let run_full = RunDir::create(tmp.path().join("full"), "echo").unwrap();
    let mut trainer_full = Trainer::new(make_cfg(), &run_full).unwrap();
    let hist_full = trainer_full
        .train(&mut policy_full, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();
    let weights_full = snapshot_vars(&policy_full.trainable_vars());
    assert_eq!(hist_full.len(), TOTAL as usize);

    // NON-VACUITY GUARD: the post-resume window MUST contain a real AdamW update
    // (grad_norm > 0). Otherwise the weight / grad-norm bit-equality below is moment-blind
    // — a broken (e.g. zeroed) moment restore would still pass. Reliable at group 32 in
    // the early regime; fail loud rather than silently green-light a vacuous run.
    let post = &hist_full[INTERRUPT_AT as usize..];
    assert!(
        post.iter().any(|m| m.grad_norm > 0.0),
        "post-resume window had no real AdamW update — the gate cannot test moment restoration"
    );

    // FAITHFUL RESUME: a FRESH policy (seeded DIFFERENTLY — 999 — so the match is the
    // restore's doing, not a shared seed) resumes from the step-INTERRUPT_AT v2 checkpoint.
    let ckpt = run_full
        .checkpoints_dir()
        .join(format!("step-{INTERRUPT_AT}"));
    let mut policy_f = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 999, TEMP).unwrap();
    let run_f = RunDir::create(tmp.path().join("faithful"), "echo").unwrap();
    let mut trainer_f = Trainer::new(make_cfg(), &run_f).unwrap();
    let hist_f = trainer_f
        .resume(&ckpt, &mut policy_f, &EchoReward, &CharTokenizer, &prompts)
        .unwrap();
    assert_eq!(hist_f.len(), (TOTAL - INTERRUPT_AT) as usize);
    // Post-resume metrics bit-equal: reward_mean / frac_reward_zero_std being equal proves
    // the RNG was restored (they derive from the rollout draws); grad_norm + the final
    // weights being equal across a REAL update (the guard above) proves the moments were.
    assert_metrics_bit_identical(&hist_f, post);
    assert_eq!(
        snapshot_vars(&policy_f.trainable_vars()),
        weights_full,
        "final adapter weights must be bit-identical after a momentum-faithful resume"
    );

    // MOMENTUM-ONLY CONTROL (isolates the moments, non-vacuity): restore the adapter AND
    // the sampler RNG but start the optimizer with FRESH moments (load_checkpoint +
    // restore_sampler_state + train_from). The RNG is held identical to the faithful
    // resume, so any divergence is PURELY the missing momentum — proving the moment
    // restore is load-bearing, not an artifact masked by a re-seeded RNG.
    let mut policy_m = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 999, TEMP).unwrap();
    let loaded = ferrl::checkpoint::load_checkpoint(&ckpt, &policy_m.trainable_vars()).unwrap();
    policy_m
        .restore_sampler_state(loaded.sampler_state.as_ref().unwrap())
        .unwrap();
    let run_m = RunDir::create(tmp.path().join("moments-fresh"), "echo").unwrap();
    let mut trainer_m = Trainer::new(make_cfg(), &run_m).unwrap();
    trainer_m
        .train_from(
            loaded.step,
            &mut policy_m,
            &EchoReward,
            &CharTokenizer,
            &prompts,
        )
        .unwrap();
    assert_ne!(
        snapshot_vars(&policy_m.trainable_vars()),
        weights_full,
        "fresh moments (RNG restored) must diverge — momentum restoration is what makes resume faithful"
    );
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
    // The run is not vacuous: real (non-degenerate) updates actually exercised the
    // canary. A wide margin — the exact count is trajectory- (so platform-)
    // dependent; we only need to prove the run was not all degenerate skips.
    let real_updates = history
        .iter()
        .filter(|m| m.frac_reward_zero_std < 0.5)
        .count();
    assert!(
        real_updates >= 5,
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
        eos_token_id: None,
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

fn malformed_prompt_len_yields_contract_error(bad_prompt_len: usize, tag: &str) {
    // A malformed rollout prompt_len must surface a typed TrainerError::Contract —
    // never a decode-slice panic (prompt_len too long) or a usize underflow in the
    // teacher-forced narrow at prompt_len - 1 (prompt_len == 0). Validation runs
    // before decode/score, so both are caught up front.
    let mut policy = BadPromptLenPolicy {
        inner: EchoPolicy::new(VOCAB, VOCAB, GAMMA, 1, TEMP).unwrap(),
        prompt_len: bad_prompt_len,
    };
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 1,
        group_size: 8,
        max_new_tokens: 1,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new(tag);
    let run = RunDir::create(tmp.path(), "x").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let err = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap_err();
    assert!(
        matches!(err, TrainerError::Contract(_)),
        "expected a Contract error, got {err:?}"
    );
}

#[test]
fn malformed_rollout_too_long_prompt_is_a_typed_error() {
    // prompt_len longer than the sequence -> no completion tokens (would panic the
    // decode slice ids[prompt_len..]).
    malformed_prompt_len_yields_contract_error(99, "malformed-long");
}

#[test]
fn malformed_rollout_zero_prompt_is_a_typed_error() {
    // prompt_len == 0 -> teacher forcing would underflow at narrow(1, prompt_len-1, ..).
    malformed_prompt_len_yields_contract_error(0, "malformed-zero");
}

#[test]
fn reward_count_mismatch_is_a_typed_error() {
    // A RewardFn returning the wrong number of scores must surface a typed error,
    // not a later cryptic shape/broadcast failure.
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 1, TEMP).unwrap();
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 1,
        group_size: 8,
        max_new_tokens: 1,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("badreward");
    let run = RunDir::create(tmp.path(), "x").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let err = trainer
        .train(&mut policy, &BadCountReward, &CharTokenizer, &prompts)
        .unwrap_err();
    assert!(
        matches!(err, TrainerError::Contract(_)),
        "expected a Contract error, got {err:?}"
    );
}

#[test]
fn wrong_rollout_size_is_a_typed_error() {
    // A Policy returning fewer completions than group_size must surface a typed
    // error, not silently become a degenerate single-item group that skips the step.
    let mut policy = UnderfilledPolicy {
        inner: EchoPolicy::new(VOCAB, VOCAB, GAMMA, 1, TEMP).unwrap(),
    };
    let prompts = echo_prompts(VOCAB);
    let cfg = TrainerConfig {
        steps: 1,
        group_size: 8,
        max_new_tokens: 1,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("underfilled");
    let run = RunDir::create(tmp.path(), "x").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let err = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap_err();
    assert!(
        matches!(err, TrainerError::Contract(_)),
        "expected a Contract error, got {err:?}"
    );
}

#[test]
fn empty_prompt_is_a_typed_error() {
    // A prompt that encodes to zero tokens (CharTokenizer encodes "" -> []) must
    // surface a typed Contract error BEFORE generate — never an underflow panic at
    // a policy's `len - 1` last-position index, and never a malformed rollout.
    let mut policy = EchoPolicy::new(VOCAB, VOCAB, GAMMA, 1, TEMP).unwrap();
    let prompts = vec![String::new()];
    let cfg = TrainerConfig {
        steps: 1,
        group_size: 8,
        max_new_tokens: 1,
        ..TrainerConfig::default()
    };
    let tmp = TempDir::new("empty-prompt");
    let run = RunDir::create(tmp.path(), "x").unwrap();
    let mut trainer = Trainer::new(cfg, &run).unwrap();
    let err = trainer
        .train(&mut policy, &EchoReward, &CharTokenizer, &prompts)
        .unwrap_err();
    assert!(
        matches!(err, TrainerError::Contract(_)),
        "expected a Contract error, got {err:?}"
    );
}
