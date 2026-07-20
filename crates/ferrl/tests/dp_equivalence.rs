//! Data-parallel equivalence gates (PR-F, P8 CPU-side).
//!
//! The DP correctness story has two halves, and the gates split accordingly:
//!
//! 1. **Sharded ≡ single, within a measured envelope.** A world-W run at
//!    per-rank accumulation `a` consumes exactly the prompts a single-rank run
//!    at accumulation `a·W` consumes, and folds the same per-item gradients —
//!    only the **summation association** differs (per-shard then cross-rank,
//!    vs one linear fold), so f32 non-associativity makes bit-exactness
//!    impossible *by construction* and the gate is a measured reassociation
//!    envelope (the `frozen_linear` / P6-C class). The policies here are
//!    **deterministic** (scripted rollouts that are a pure function of the
//!    prompt, real `LoRA` gradients) — a deliberate simplification that isolates
//!    the envelope to gradient arithmetic alone. (A *stochastic* sampler is now
//!    shard-invariant too: substreams seed from the **global row index**, so a
//!    world-W rollout reproduces the single-process draws — gated directly in
//!    `sampler` / `lm_policy`. These gradient gates keep deterministic rollouts
//!    only to keep the arithmetic the sole variable.)
//! 2. **Ranks ≡ each other, bitwise.** Whatever the rollouts, every rank steps
//!    from the identical reduced gradient, so the world's weights stay in
//!    bitwise lockstep — gated on the REAL tiny-qwen3.5 fixture in BOTH
//!    training modes (`LoRA` and full fine-tuning; their var sets and grad
//!    paths differ end-to-end), with stochastic per-rank rollouts.
//!
//! Plus the structural gates: world-1 `LocalComm` ≡ `SoloComm` ≡ the legacy
//! constructor (bitwise — the DP plumbing must not perturb the single-rank
//! path); an all-degenerate **local** shard neither deadlocks nor diverges
//! (the rank participates with zeros — the live-count decision is global);
//! an all-degenerate **global** window is skipped by every rank in lockstep;
//! and checkpointing is rank-0-only, with a DP resume continuing bit-exactly
//! (deterministic policies here; and under global-index seeding the rank-0
//! sampler-blob is anyway sufficient — a stochastic resume re-derives every
//! rank's per-row draws from the restored run seed and the recomputed global
//! index, with no per-rank RNG state to capture). A manual ignored
//! `--features nccl` smoke at the bottom runs the same tiny qwen3.5 path over
//! real CUDA/NCCL ranks and prints memory/timing fields for resource regression
//! checks.

use candle_core::{
    CpuStorage, CustomOp1, DType, Device, Layout, Result as CandleResult, Shape, Tensor, Var, D,
};
use candle_nn::ops::log_softmax;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use ferrl::lora::LoraLinear;
use ferrl::nn::RmsNorm;
use ferrl::policy::{GenConfig, Policy, Rollout};
use ferrl::telemetry::RunDir;
use ferrl::trainer::{TokenizerLike, Trainer, TrainerConfig, TrainerError};
use ferrl::{
    tensors_from_pretrained, varbuilder_from_pretrained, Comm, CommError, LocalComm, LossType,
    Metrics, OptimizerState, Qwen3_5Config, Qwen3_5GradModel, Qwen3_5Policy, RewardError, RewardFn,
    RewardGroupScope, RolloutLedgerError, Sample, SoloComm,
};

const VOCAB: usize = 5;
const SEED: u64 = 11;

// ---- the deterministic policy ----------------------------------------------

/// A one-layer `LoRA` LM (the toy-echo scaffold) whose `generate` is
/// **scripted**: completion `g` of a prompt echoes the prompt's first symbol
/// when `g` is even and emits the next symbol when odd — a pure function of
/// the prompt, identical under any sharding, while `token_logprobs` stays the
/// real differentiable forward (one-hot → `LoraLinear` → `RmsNorm` →
/// `log_softmax`), so gradients and optimizer steps are the real thing.
struct ScriptedPolicy {
    lora: LoraLinear,
    norm: RmsNorm,
    device: Device,
}

impl ScriptedPolicy {
    fn new(seed: u64) -> CandleResult<Self> {
        let device = Device::Cpu;
        let base = Tensor::zeros((VOCAB, VOCAB), DType::F32, &device)?;
        let lora = LoraLinear::new(base, None, VOCAB, VOCAB as f64)?;
        // Deterministic adapter init (LoraLinear's A comes from OS-entropy
        // randn on CPU): a seeded hash fill at the same ~0.02 scale, B zero —
        // every rank and every run constructs the identical policy.
        let a = &lora.trainable_vars()[0];
        let (r, c) = a.as_tensor().dims2()?;
        let fill: Vec<f32> = (0..r * c)
            .map(|i| {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                ((z % 1000) as f32 / 1000.0 - 0.5) * 0.04
            })
            .collect();
        a.set(&Tensor::from_vec(fill, (r, c), &device)?)?;
        let gamma = Tensor::ones(VOCAB, DType::F32, &device)?.affine(3.0, 0.0)?;
        let norm = RmsNorm::new(gamma, 1e-6);
        Ok(Self { lora, norm, device })
    }
}

impl Policy for ScriptedPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let first = prompt[0];
        let token_ids = (0..cfg.group_size)
            .map(|g| {
                let symbol = if g % 2 == 0 {
                    first
                } else {
                    (first + 1) % VOCAB as u32
                };
                let mut ids = prompt.to_vec();
                ids.extend(std::iter::repeat_n(symbol, cfg.max_new_tokens));
                ids
            })
            .collect();
        Ok(Rollout::rectangular(token_ids, prompt.len()))
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let prompt_len = rollout.prompt_len;
        let seq_len = rollout.token_ids[0].len();
        let comp_len = seq_len - prompt_len;
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let mut oh = vec![0f32; g * input_len * VOCAB];
        let mut targets = vec![0u32; g * comp_len];
        for (i, ids) in rollout.token_ids.iter().enumerate() {
            for t in 0..input_len {
                oh[(i * input_len + t) * VOCAB + ids[t] as usize] = 1.0;
            }
            for j in 0..comp_len {
                targets[i * comp_len + j] = ids[prompt_len + j];
            }
        }
        let oh = Tensor::from_vec(oh, (g, input_len, VOCAB), &self.device)?;
        let h = self.lora.forward(&oh)?;
        let logits = self.norm.forward(&h)?;
        let pred = logits.narrow(1, prompt_len - 1, comp_len)?;
        let logp = log_softmax(&pred, D::Minus1)?;
        let idx = Tensor::from_vec(targets, (g, comp_len), &self.device)?.unsqueeze(D::Minus1)?;
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
        // Generation is scripted — there is no RNG to capture, which is what
        // makes the DP resume gate bit-exact (see the module docs).
        Ok(Vec::new())
    }

    fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
        Ok(())
    }
}

/// A deterministic but genuinely stateful rollout stream over the same real
/// LoRA/autograd scoring path as [`ScriptedPolicy`]. Each prompt group advances
/// an opaque epoch and changes how many rows echo the prompt, so sampler handoff
/// affects rewards, advantages, gradients, and the final Adam trajectory.
struct StatefulScriptedPolicy {
    inner: ScriptedPolicy,
    sampler_epoch: u64,
}

impl StatefulScriptedPolicy {
    const STATE_MAGIC: [u8; 4] = *b"FSP1";

    fn new(seed: u64, sampler_epoch: u64) -> CandleResult<Self> {
        Ok(Self {
            inner: ScriptedPolicy::new(seed)?,
            sampler_epoch,
        })
    }
}

impl Policy for StatefulScriptedPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.generate_at(prompt, cfg, 0)
    }

    fn generate_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
    ) -> CandleResult<Rollout> {
        let first = prompt[0];
        let token_ids = (0..cfg.group_size)
            .map(|row| {
                let row = global_row_base.wrapping_add(row as u64);
                let symbol = if row.wrapping_add(self.sampler_epoch).is_multiple_of(2) {
                    first
                } else {
                    (first + 1) % VOCAB as u32
                };
                let mut ids = prompt.to_vec();
                ids.extend(std::iter::repeat_n(symbol, cfg.max_new_tokens));
                ids
            })
            .collect();
        self.sampler_epoch = self.sampler_epoch.wrapping_add(1);
        Ok(Rollout::rectangular(token_ids, prompt.len()))
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
        let mut state = Self::STATE_MAGIC.to_vec();
        state.extend_from_slice(&self.sampler_epoch.to_le_bytes());
        Ok(state)
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        if state.len() != 12 || state[..4] != Self::STATE_MAGIC {
            candle_core::bail!("invalid stateful scripted sampler state")
        }
        let mut epoch = [0_u8; 8];
        epoch.copy_from_slice(&state[4..]);
        self.sampler_epoch = u64::from_le_bytes(epoch);
        Ok(())
    }
}

/// A forwarding policy with the exact same live trainable tensors as
/// [`ScriptedPolicy`] but a different declared adapter recipe. Ledger identity
/// must bind this semantic schema, not only tensor values and shapes.
struct RecipeScriptedPolicy {
    inner: ScriptedPolicy,
}

impl Policy for RecipeScriptedPolicy {
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
        self.inner.trainable_vars()
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
    }

    fn lora_recipe(&self) -> Option<String> {
        Some("test:alternate-scripted-recipe".to_string())
    }
}

/// Replaces its trainable variables with byte-identical fresh variables after
/// the first rollout. A post-collection check that only rehashes previously
/// captured `Var` handles cannot observe this policy-state replacement.
struct ReplacingGeneratePolicy {
    inner: ScriptedPolicy,
    seed: u64,
    replaced: bool,
}

impl Policy for ReplacingGeneratePolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let rollout = self.inner.generate(prompt, cfg)?;
        if !self.replaced {
            self.inner = ScriptedPolicy::new(self.seed)?;
            self.replaced = true;
        }
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
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
    }
}

/// Replaces and perturbs its live trainable variables during the learner's
/// adapter-toggle preflight. The rollback seam cannot reattach the original
/// handles through `Policy`, so it must report that exact restoration failed.
struct ReplacingTogglePolicy {
    inner: ScriptedPolicy,
    seed: u64,
    replaced: bool,
}

impl Policy for ReplacingTogglePolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        self.inner.token_logprobs(rollout)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        if !enabled && !self.replaced {
            self.inner = ScriptedPolicy::new(self.seed).unwrap();
            let live_var = self.inner.trainable_vars()[1].clone();
            let (rows, cols) = live_var.as_tensor().dims2().unwrap();
            let mut values = vec![0.0_f32; rows * cols];
            values[0] = 0.125;
            live_var
                .set(&Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap())
                .unwrap();
            self.replaced = true;
        }
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

/// Returns a live scoring graph over the original trainable variables, then
/// replaces the policy's active variables before backward. Gradient coverage
/// and Adam can still succeed on the orphaned originals, so the learner must
/// recheck the binding after the complete update.
struct ReplacingLiveScoringPolicy {
    inner: RefCell<ScriptedPolicy>,
    seed: u64,
    replaced: Cell<bool>,
}

impl Policy for ReplacingLiveScoringPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.get_mut().generate(prompt, cfg)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let logp = self.inner.borrow().token_logprobs(rollout)?;
        if !self.replaced.replace(true) {
            let replacement = ScriptedPolicy::new(self.seed)?;
            let live_var = replacement.trainable_vars()[1].clone();
            let (rows, cols) = live_var.as_tensor().dims2()?;
            let mut values = vec![0.0_f32; rows * cols];
            values[0] = 0.125;
            live_var.set(&Tensor::from_vec(values, (rows, cols), &Device::Cpu)?)?;
            *self.inner.borrow_mut() = replacement;
        }
        Ok(logp)
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        Ok(self.inner.borrow().token_logprobs(rollout)?.detach())
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.inner.get_mut().set_adapter_enabled(enabled);
    }

    fn adapter_enabled(&self) -> bool {
        self.inner.borrow().adapter_enabled()
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.inner.borrow().trainable_vars()
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.inner.borrow().sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.get_mut().restore_sampler_state(state)
    }
}

/// `'a'..` to ids `0..` (the toy-echo codec).
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

/// Echo reward (spread within every scripted group: even members echo and
/// score 1, odd members don't and score 0) — except prompts starting with
/// `'e'`, which score a CONSTANT: their groups are degenerate (all-equal
/// rewards, zero advantages), the lever the degenerate-shard gates pull.
struct EchoOrFlatReward;
impl RewardFn for EchoOrFlatReward {
    type Target = ();
    fn reward(&self, sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
        let prompt = sample.prompt.as_str();
        if prompt.starts_with('e') {
            return Ok(0.5);
        }
        Ok(match (prompt.chars().next(), completion.chars().next()) {
            (Some(p), Some(c)) if p == c => 1.0,
            _ => 0.0,
        })
    }
}

/// [`ScriptedPolicy`] plus one extra trainable var that reaches the loss only
/// when `wired` — mimicking, e.g., a never-routed expert on one rank. The var
/// COUNT matches across ranks (the collective contract holds; the unwired
/// rank's reduce slot carries zeros) but the unwired rank's backward never
/// covers it, which must abort the whole world in lockstep.
struct GatedExtraPolicy {
    inner: ScriptedPolicy,
    extra: Var,
    wired: bool,
}

impl GatedExtraPolicy {
    fn new(seed: u64, wired: bool) -> Self {
        Self {
            inner: ScriptedPolicy::new(seed).unwrap(),
            extra: Var::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            wired,
        }
    }
}

impl Policy for GatedExtraPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let logp = self.inner.token_logprobs(rollout)?;
        if self.wired {
            // A zero-valued additive touch: the scores are unchanged (the
            // ranks' losses stay identical) but the var joins the graph, so
            // only the wired rank's backward covers it.
            logp.broadcast_add(self.extra.as_tensor())
        } else {
            Ok(logp)
        }
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.inner.set_adapter_enabled(enabled);
    }

    fn adapter_enabled(&self) -> bool {
        self.inner.adapter_enabled()
    }

    fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = self.inner.trainable_vars();
        vars.push(self.extra.clone());
        vars
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
    }
}

struct FailingScoringPolicy {
    inner: ScriptedPolicy,
    fail_detached: bool,
    fail_live: bool,
}

impl Policy for FailingScoringPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        if self.fail_live {
            candle_core::bail!("injected live scoring failure")
        }
        self.inner.token_logprobs(rollout)
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        if self.fail_detached {
            candle_core::bail!("injected detached scoring failure")
        }
        Ok(self.inner.token_logprobs(rollout)?.detach())
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

struct PanickingDetachedScoringPolicy {
    inner: ScriptedPolicy,
    panic_detached: bool,
}

impl Policy for PanickingDetachedScoringPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        self.inner.token_logprobs(rollout)
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        assert!(!self.panic_detached, "injected detached scoring panic");
        Ok(self.inner.token_logprobs(rollout)?.detach())
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

struct FailingCollectionPolicy {
    inner: StatefulScriptedPolicy,
    fail_generate: bool,
}

impl Policy for FailingCollectionPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.generate_at(prompt, cfg, 0)
    }

    fn generate_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
    ) -> CandleResult<Rollout> {
        if self.fail_generate {
            candle_core::bail!("injected rank-local rollout generation failure")
        }
        self.inner.generate_at(prompt, cfg, global_row_base)
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

struct FailingCollectionReward {
    fail: bool,
}

impl RewardFn for FailingCollectionReward {
    type Target = ();

    fn reward(&self, sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
        if self.fail {
            return Err(RewardError::msg(
                "injected rank-local rollout reward failure",
            ));
        }
        EchoOrFlatReward.reward(sample, completion)
    }
}

struct FailingBackwardOp;

impl CustomOp1 for FailingBackwardOp {
    fn name(&self) -> &'static str {
        "injected-failing-backward"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        Ok((storage.clone(), layout.shape().clone()))
    }
}

struct FailingBackwardPolicy {
    inner: ScriptedPolicy,
    fail_backward: bool,
}

impl Policy for FailingBackwardPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let logp = self.inner.token_logprobs(rollout)?;
        if self.fail_backward {
            logp.contiguous()?.apply_op1(FailingBackwardOp)
        } else {
            Ok(logp)
        }
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        Ok(self.inner.token_logprobs(rollout)?.detach())
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

struct RejectingSamplerRestorePolicy {
    inner: StatefulScriptedPolicy,
    fail_next_restore: bool,
}

impl Policy for RejectingSamplerRestorePolicy {
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
        if std::mem::take(&mut self.fail_next_restore) {
            candle_core::bail!("injected rank-local continuation sampler failure")
        }
        self.inner.restore_sampler_state(state)
    }
}

struct PanickingSamplerRestorePolicy {
    inner: StatefulScriptedPolicy,
    panic_next_restore: bool,
}

impl Policy for PanickingSamplerRestorePolicy {
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
        assert!(
            !std::mem::take(&mut self.panic_next_restore),
            "injected sampler handoff panic"
        );
        self.inner.restore_sampler_state(state)
    }
}

struct PersistentRollbackFailurePolicy {
    inner: StatefulScriptedPolicy,
    fail_generate: bool,
    fail_restore: bool,
    restore_calls: usize,
}

impl Policy for PersistentRollbackFailurePolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.generate_at(prompt, cfg, 0)
    }

    fn generate_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
    ) -> CandleResult<Rollout> {
        if self.fail_generate {
            candle_core::bail!("injected collector failure before rollback")
        }
        self.inner.generate_at(prompt, cfg, global_row_base)
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
        self.restore_calls += 1;
        if self.fail_restore {
            candle_core::bail!("injected persistent rank-local sampler restore failure")
        }
        self.inner.restore_sampler_state(state)
    }
}

#[derive(Debug)]
struct InjectedCollectiveFailureState {
    armed: AtomicBool,
    remaining_successes: AtomicUsize,
    failed: AtomicBool,
    calls_after_failure: AtomicUsize,
}

impl InjectedCollectiveFailureState {
    fn new(successes_after_arm: usize) -> Self {
        Self {
            armed: AtomicBool::new(false),
            remaining_successes: AtomicUsize::new(successes_after_arm),
            failed: AtomicBool::new(false),
            calls_after_failure: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
struct InjectedFailureComm {
    state: Arc<InjectedCollectiveFailureState>,
}

impl InjectedFailureComm {
    fn enter_collective(&self) -> Result<(), CommError> {
        if self.state.failed.load(Ordering::SeqCst) {
            self.state
                .calls_after_failure
                .fetch_add(1, Ordering::SeqCst);
            return Err(CommError::Poisoned(
                "collective issued after injected terminal failure".into(),
            ));
        }
        if !self.state.armed.load(Ordering::SeqCst) {
            return Ok(());
        }
        let remaining = self.state.remaining_successes.load(Ordering::SeqCst);
        if remaining > 0 {
            self.state
                .remaining_successes
                .fetch_sub(1, Ordering::SeqCst);
            return Ok(());
        }
        self.state.failed.store(true, Ordering::SeqCst);
        Err(CommError::Mismatch(
            "injected terminal data-parallel failure".into(),
        ))
    }
}

impl Comm for InjectedFailureComm {
    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        2
    }

    fn all_reduce_sum(&self, _tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        self.enter_collective()
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
        self.enter_collective()?;
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeadCommPhase {
    CollectorGeneration,
    LearnerDetachedScoring,
}

#[derive(Clone, Copy, Debug)]
enum DeadCommRollbackFault {
    ReturnError,
    Panic,
}

/// Arms a terminal communicator failure immediately after a policy callback
/// has made local progress, then faults the local-only sampler rollback. This
/// proves the trainer classifies the communication failure first and never
/// tries to coordinate recovery through the poisoned world.
struct DeadCommRecoveryPolicy {
    inner: StatefulScriptedPolicy,
    state: Arc<InjectedCollectiveFailureState>,
    phase: DeadCommPhase,
    rollback_fault: DeadCommRollbackFault,
    post_failure_restore_calls: usize,
}

impl DeadCommRecoveryPolicy {
    fn arm_and_fail(&self, message: &'static str) -> CandleResult<()> {
        self.state.armed.store(true, Ordering::SeqCst);
        candle_core::bail!("{message}")
    }
}

impl Policy for DeadCommRecoveryPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.generate_at(prompt, cfg, 0)
    }

    fn generate_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
    ) -> CandleResult<Rollout> {
        let rollout = self.inner.generate_at(prompt, cfg, global_row_base)?;
        if self.phase == DeadCommPhase::CollectorGeneration {
            self.arm_and_fail("injected collector failure before terminal status")?;
        }
        Ok(rollout)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        self.inner.token_logprobs(rollout)
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let logprobs = self.inner.token_logprobs_detached(rollout)?;
        if self.phase == DeadCommPhase::LearnerDetachedScoring {
            self.arm_and_fail("injected learner failure before terminal status")?;
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
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        if self.state.failed.load(Ordering::SeqCst) {
            self.post_failure_restore_calls += 1;
            match self.rollback_fault {
                DeadCommRollbackFault::ReturnError => {
                    candle_core::bail!("injected post-communication local rollback failure")
                }
                DeadCommRollbackFault::Panic => {
                    panic!("injected post-communication local rollback panic")
                }
            }
        }
        self.inner.restore_sampler_state(state)
    }
}

struct ContinuationFaultPolicy {
    inner: StatefulScriptedPolicy,
    panic_next_trainable_vars: Cell<bool>,
    fail_restore_call: Option<usize>,
    panic_restore_call: Option<usize>,
    arm_collective_failure_on_restore_call: Option<(usize, Arc<InjectedCollectiveFailureState>)>,
    restore_calls: usize,
}

impl Policy for ContinuationFaultPolicy {
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

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.inner.set_adapter_enabled(enabled);
    }

    fn adapter_enabled(&self) -> bool {
        self.inner.adapter_enabled()
    }

    fn trainable_vars(&self) -> Vec<Var> {
        assert!(
            !self.panic_next_trainable_vars.replace(false),
            "injected continuation trainable-vars panic"
        );
        self.inner.trainable_vars()
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.restore_calls += 1;
        if self.fail_restore_call == Some(self.restore_calls) {
            candle_core::bail!("injected continuation restore failure")
        }
        assert!(
            self.panic_restore_call != Some(self.restore_calls),
            "injected continuation rollback panic"
        );
        let result = self.inner.restore_sampler_state(state);
        if result.is_ok() {
            if let Some((restore_call, failure)) = &self.arm_collective_failure_on_restore_call {
                if *restore_call == self.restore_calls {
                    failure.armed.store(true, Ordering::SeqCst);
                }
            }
        }
        result
    }
}

/// A world-1 spy counting collective invocations — the oracle for the
/// "world-1 issues no collectives" guard discipline.
#[derive(Debug)]
struct SpyComm {
    calls: Arc<AtomicUsize>,
}

impl Comm for SpyComm {
    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn all_reduce_sum(&self, _tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(value)
    }
}

// ---- harness ----------------------------------------------------------------

struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("ferrl-dp-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

struct RelativeTempRoot(PathBuf);
impl RelativeTempRoot {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = PathBuf::from(format!(
            ".ferrl-dp-relative-{tag}-{}-{nanos}",
            std::process::id()
        ));
        assert!(path.is_relative());
        assert!(!path.exists());
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for RelativeTempRoot {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// The policy's trainable vars as raw `f32` bit patterns (bitwise comparison —
/// `==` on floats would also pass for `-0.0` vs `0.0`).
fn var_bits<P: Policy>(policy: &P) -> Vec<Vec<u32>> {
    policy
        .trainable_vars()
        .iter()
        .map(|v| {
            v.as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect()
}

/// Raw `f32` bits for every tensor in an optimizer-moment list.
fn tensor_bits(tensors: &[Tensor]) -> Vec<Vec<u32>> {
    tensors
        .iter()
        .map(|tensor| {
            tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect()
}

/// Adam state as a bitwise-comparable value. The moments matter independently
/// of the adapter: Adam can cancel a uniform gradient-scale error in the weight
/// update while retaining the wrong first and second moments.
fn optimizer_bits(state: &OptimizerState) -> (usize, Vec<Vec<u32>>, Vec<Vec<u32>>) {
    (
        state.step_t,
        tensor_bits(&state.first_moments),
        tensor_bits(&state.second_moments),
    )
}

/// Strip fields that describe wall time or device-memory sampling rather than
/// the mathematical result of a trainer window.
fn deterministic_metrics(metrics: &Metrics) -> Metrics {
    let mut metrics = metrics.clone();
    metrics.step_secs = 0.0;
    metrics.tokens_per_sec = 0.0;
    metrics.cuda_mem_start_used_bytes = 0;
    metrics.cuda_mem_peak_used_bytes = 0;
    metrics.cuda_mem_end_used_bytes = 0;
    metrics.cuda_mem_total_bytes = 0;
    metrics.cuda_mem_peak_delta_bytes = 0;
    metrics.cuda_mem_probe_events.clear();
    metrics.decoder_cache_snapshots.clear();
    metrics
}

#[allow(clippy::cognitive_complexity)] // one assertion surface pins every ordinary sentinel
fn assert_ledger_performance_unmeasured(metrics: &Metrics, context: &str) {
    assert_eq!(metrics.step_secs, 0.0, "{context}: step_secs");
    assert_eq!(metrics.tokens_per_sec, 0.0, "{context}: tokens_per_sec");
    assert_eq!(
        metrics.cuda_mem_start_used_bytes, 0,
        "{context}: cuda start"
    );
    assert_eq!(metrics.cuda_mem_peak_used_bytes, 0, "{context}: cuda peak");
    assert_eq!(metrics.cuda_mem_end_used_bytes, 0, "{context}: cuda end");
    assert_eq!(metrics.cuda_mem_total_bytes, 0, "{context}: cuda total");
    assert_eq!(
        metrics.cuda_mem_peak_delta_bytes, 0,
        "{context}: cuda delta"
    );
    assert!(
        metrics.cuda_mem_probe_events.is_empty(),
        "{context}: cuda probe events"
    );
    assert!(
        metrics.decoder_cache_snapshots.is_empty(),
        "{context}: decoder cache snapshots"
    );
}

fn assert_ledger_identity_mismatch<T>(result: Result<T, TrainerError>, what: &str) {
    match result {
        Err(TrainerError::RolloutLedger(RolloutLedgerError::IdentityMismatch)) => {}
        Err(error) => panic!("{what}: expected ledger identity mismatch, got {error:?}"),
        Ok(_) => panic!("{what}: mismatched learner identity was accepted"),
    }
}

/// Largest |a - b| across two var-bit snapshots (for the measured envelope).
fn max_abs_diff(a: &[Vec<u32>], b: &[Vec<u32>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(va, vb)| va.iter().zip(vb))
        .map(|(&xa, &xb)| f64::from((f32::from_bits(xa) - f32::from_bits(xb)).abs()))
        .fold(0.0, f64::max)
}

/// One rank's outcome: final var bits + the metrics history.
type RankRun = (Vec<Vec<u32>>, Vec<Metrics>);

/// Train a fresh [`ScriptedPolicy`] world of `world_size` ranks (one thread
/// per rank, rank `r` under `base/rank<r>`), returning each rank's outcome.
fn run_scripted_world(
    base: &Path,
    world_size: usize,
    cfg: &TrainerConfig,
    samples: &[Sample<()>],
) -> Vec<RankRun> {
    let comms = LocalComm::world(world_size);
    std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg.clone(), &run, comm).unwrap();
                    let history = trainer
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, samples)
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Train a fresh single-rank [`ScriptedPolicy`] run via the legacy
/// constructor, returning its outcome.
fn run_scripted_single(base: &Path, cfg: &TrainerConfig, samples: &[Sample<()>]) -> RankRun {
    let mut policy = ScriptedPolicy::new(SEED).unwrap();
    let run = RunDir::create(base, "single").unwrap();
    let mut trainer = Trainer::new(cfg.clone(), &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, samples)
        .unwrap();
    (var_bits(&policy), history.0)
}

type AdamBits = (usize, Vec<Vec<u32>>, Vec<Vec<u32>>);

struct StatefulDirectRank {
    initial_adapter: Vec<Vec<u32>>,
    adapter: Vec<Vec<u32>>,
    metrics: Vec<Metrics>,
    sampler: Vec<u8>,
    optimizer: Option<AdamBits>,
}

struct StatefulSeparatedRank {
    initial_adapter: Vec<Vec<u32>>,
    adapter: Vec<Vec<u32>>,
    metrics: Vec<Metrics>,
    sampler: Vec<u8>,
    optimizer: AdamBits,
    lineage: String,
}

fn run_stateful_direct_dp(
    base: &Path,
    tag: &str,
    cfg: &TrainerConfig,
    samples: &[Sample<()>],
) -> Vec<StatefulDirectRank> {
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(20));
    std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let samples = samples.to_vec();
                let base = base.to_path_buf();
                scope.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    let initial_adapter = var_bits(&policy);
                    let run = RunDir::create(&base, format!("{tag}-direct-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg.clone(), &run, comm).unwrap();
                    let metrics = trainer
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, &samples)
                        .unwrap()
                        .0;
                    let optimizer = if rank == 0 {
                        let loaded = ferrl::load_checkpoint(
                            run.checkpoints_dir().join(format!("step-{}", cfg.steps)),
                            &policy.trainable_vars(),
                        )
                        .unwrap();
                        Some(optimizer_bits(&loaded.optimizer_state.unwrap()))
                    } else {
                        None
                    };
                    StatefulDirectRank {
                        initial_adapter,
                        adapter: var_bits(&policy),
                        metrics,
                        sampler: policy.sampler_state().unwrap(),
                        optimizer,
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    })
}

#[allow(clippy::cognitive_complexity)] // full three-step two-role restart oracle
fn run_stateful_separated_dp(
    base: &Path,
    tag: &str,
    cfg: &TrainerConfig,
    samples: &[Sample<()>],
    policy_sha256: &str,
) -> Vec<StatefulSeparatedRank> {
    let ledger_root = base.join(format!("{tag}-ledger"));
    let checkpoint_root = base.join(format!("{tag}-continuations"));
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(20));
    std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let samples = samples.to_vec();
                let base = base.to_path_buf();
                let ledger_root = ledger_root.clone();
                let checkpoint_root = checkpoint_root.clone();
                scope.spawn(move || {
                    let rank = comm.rank();
                    let run = RunDir::create(&base, format!("{tag}-separated-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg.clone(), &run, comm).unwrap();
                    let mut collector_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    let mut learner_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    let initial_adapter = var_bits(&learner_policy);
                    let mut continuation = None;
                    let mut metrics = Vec::new();

                    for step in 0..cfg.steps {
                        trainer
                            .collect_rollout_ledger_step(
                                step,
                                &mut collector_policy,
                                &EchoOrFlatReward,
                                &CharTokenizer,
                                &samples,
                                &ledger_root,
                                policy_sha256,
                                continuation.as_ref(),
                            )
                            .unwrap();
                        let (row, next) = trainer
                            .train_rollout_ledger_step(
                                step,
                                &mut learner_policy,
                                &ledger_root,
                                policy_sha256,
                                continuation.as_ref(),
                            )
                            .unwrap();
                        metrics.push(row);
                        let checkpoint = trainer
                            .save_rollout_ledger_continuation_to(
                                &checkpoint_root,
                                &learner_policy,
                                &next,
                            )
                            .unwrap();

                        // The continuation package is the role-handoff boundary:
                        // both independently hosted roles install it before the
                        // next outer step. Exercise a real process replacement
                        // after step 0, then keep using the same durable handoff
                        // on later steps so the collector never runs a newer
                        // receipt against a stale adapter.
                        if step == 0 {
                            collector_policy =
                                StatefulScriptedPolicy::new(SEED.wrapping_add(101), 777).unwrap();
                            learner_policy =
                                StatefulScriptedPolicy::new(SEED.wrapping_add(202), 888).unwrap();
                        }
                        let collector_restored = trainer
                            .restore_rollout_ledger_continuation(
                                &checkpoint,
                                &mut collector_policy,
                                policy_sha256,
                            )
                            .unwrap();
                        let learner_restored = trainer
                            .restore_rollout_ledger_continuation(
                                &checkpoint,
                                &mut learner_policy,
                                policy_sha256,
                            )
                            .unwrap();
                        assert_eq!(collector_restored.completed_step(), step + 1);
                        assert_eq!(learner_restored.completed_step(), step + 1);
                        assert_eq!(collector_restored.world_size(), 2);
                        assert_eq!(learner_restored.world_size(), 2);
                        assert_eq!(
                            optimizer_bits(collector_restored.optimizer_state()),
                            optimizer_bits(learner_restored.optimizer_state())
                        );
                        assert_eq!(var_bits(&collector_policy), var_bits(&learner_policy));
                        continuation = Some(learner_restored);
                    }

                    let expected_adapter = var_bits(&learner_policy);
                    let expected_sampler = learner_policy.sampler_state().unwrap();
                    let mut latest_policy =
                        StatefulScriptedPolicy::new(SEED.wrapping_add(303), 999).unwrap();
                    let latest = trainer
                        .restore_latest_rollout_ledger_continuation_from(
                            &checkpoint_root,
                            &mut latest_policy,
                            policy_sha256,
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(latest.completed_step(), cfg.steps);
                    assert_eq!(var_bits(&latest_policy), expected_adapter);
                    assert_eq!(latest_policy.sampler_state().unwrap(), expected_sampler);
                    assert_eq!(
                        ferrl::read_metrics(run.metrics_path()).unwrap().len(),
                        cfg.steps as usize
                    );
                    StatefulSeparatedRank {
                        initial_adapter,
                        adapter: expected_adapter,
                        metrics,
                        sampler: expected_sampler,
                        optimizer: optimizer_bits(latest.optimizer_state()),
                        lineage: latest.lineage_sha256().to_owned(),
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    })
}

fn scripted_cfg() -> TrainerConfig {
    TrainerConfig {
        steps: 3,
        group_size: 4,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.04,
        mu: 2,
        lr: 5e-3,
        loss_type: LossType::Grpo,
        ..TrainerConfig::default()
    }
}

fn live_samples() -> Vec<Sample<()>> {
    ["a", "b", "c", "d"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect()
}

fn assert_lockstep(ranks: &[RankRun], what: &str) {
    for (r, (bits, _)) in ranks.iter().enumerate().skip(1) {
        assert_eq!(
            &ranks[0].0, bits,
            "{what}: rank {r} diverged bitwise from rank 0"
        );
    }
}

/// The pre-clip global `grad_norm` must agree per step between a sharded and
/// a single run (modulo reassociation). This is the scale oracle the weight
/// envelope CANNOT be: `AdamW`'s m̂/√v̂ cancels a uniform gradient-scale error
/// and global-norm clipping is bitwise invariant to power-of-2 scales, so a
/// missed `world` divisor or a localized Dapo normalizer (both exact 2x)
/// leaves the final weights untouched — but the reported pre-clip norm sees
/// the 2x directly (found by the PR-F mutation sweep: M2/M3 survived the
/// weight envelope alone).
fn assert_grad_norms_match(world: &[Metrics], single: &[Metrics], what: &str) {
    assert_eq!(world.len(), single.len(), "{what}: step counts differ");
    for (i, (w, s)) in world.iter().zip(single).enumerate() {
        let diff = (f64::from(w.grad_norm) - f64::from(s.grad_norm)).abs();
        let scale = f64::from(s.grad_norm).abs().max(1e-12);
        assert!(
            diff / scale < 1e-3,
            "{what}: step {i} grad_norm diverged — sharded {} vs single {} (a \
             uniform gradient-scale error, e.g. a missed world divisor or a \
             local normalizer?)",
            w.grad_norm,
            s.grad_norm
        );
    }
}

// ---- preemption stop is globalized across the DP world ----------------------

/// The cooperative preemption stop must halt **every** rank at the same step even
/// when the flag is installed on only ONE rank — the un-footgunned, install-invariant
/// case. Rank 1 here has **no** flag at all; under a rank-local poll it would skip
/// the per-step reduce, run ahead into the next window's collective while rank 0
/// (flag set) broke out, and hang until the timeout. The install-invariant poll makes
/// rank 1 join the reduce anyway, so both stop after the same step. The 20 s timeout
/// converts any regression into a loud failure instead of a suite hang.
#[test]
fn preemption_flag_stops_the_whole_dp_world_in_lockstep() {
    let tmp = TempDir::new("preempt-dp");
    // More steps than we expect to run, so stopping after step 0 is visibly early.
    let cfg = TrainerConfig {
        steps: 5,
        ..scripted_cfg()
    };
    let samples = live_samples();
    // Flag installed on rank 0 ONLY; rank 1 gets none — the uneven-install case.
    let flag0 = Arc::new(AtomicBool::new(true));
    // A short collective timeout so a regression (a rank that fails to stop in
    // lockstep) fails fast as a loud timeout instead of hanging the suite.
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(20));
    let histories: Vec<Vec<Metrics>> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let flag0 = Arc::clone(&flag0);
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(&base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    if rank == 0 {
                        trainer = trainer.with_preemption_flag(flag0);
                    }
                    // Returns Ok only if the world never deadlocked — a rank stuck in a
                    // collective its peer skipped would surface as a CommError here.
                    trainer
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, &samples)
                        .unwrap()
                        .0
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(
        histories[0].len(),
        1,
        "rank 0 (flag set) should stop after step 0"
    );
    assert_eq!(
        histories[1].len(),
        histories[0].len(),
        "rank 1 (NO flag) must stop at the same step — the poll is install-invariant"
    );
}

// ---- gate 1: sharded vs single, measured envelope ---------------------------

/// World-2 at per-rank accumulation 2 vs single-rank at accumulation 4: the
/// same global windows, the same per-item gradients, different summation
/// association. Measured on the dev host 2026-06-12: Grpo 3.5e-9, Dapo
/// 4.6e-8 — the 1e-5 bound leaves ~200x headroom for the runner-pool spread
/// while staying orders below the bug signal (a missed `world` divisor is a
/// ~2x gradient-scale error, ~1e-2 on the weights after these steps).
#[test]
#[allow(clippy::print_stderr)] // the measured envelope is the calibration record
fn world2_matches_single_run_within_the_reassociation_envelope_grpo() {
    let tmp = TempDir::new("env-grpo");
    let cfg2 = TrainerConfig {
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let cfg1 = TrainerConfig {
        grad_accum_steps: 4,
        ..scripted_cfg()
    };
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &live_samples());
    let single = run_scripted_single(tmp.path(), &cfg1, &live_samples());
    assert_lockstep(&ranks, "grpo envelope");
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    eprintln!("measured grpo sharded-vs-single envelope: {envelope:.3e}");
    assert!(
        envelope < 1e-5,
        "sharded run left the reassociation envelope: {envelope:.3e}"
    );
    assert_grad_norms_match(&ranks[0].1, &single.1, "grpo envelope");
    // Vacuity guard: training actually moved the weights.
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert!(max_abs_diff(&single.0, &init) > 0.0, "no training signal");
}

/// The Dapo variant pins the GLOBAL window-token normalizer: a local (per
/// rank) normalizer would scale every gradient by ~2x and blow far past the
/// envelope.
#[test]
#[allow(clippy::print_stderr)] // the measured envelope is the calibration record
fn world2_matches_single_run_within_the_reassociation_envelope_dapo() {
    let tmp = TempDir::new("env-dapo");
    let base = TrainerConfig {
        beta: 0.0,
        mu: 1,
        loss_type: LossType::Dapo,
        ..scripted_cfg()
    };
    let cfg2 = TrainerConfig {
        grad_accum_steps: 2,
        ..base.clone()
    };
    let cfg1 = TrainerConfig {
        grad_accum_steps: 4,
        ..base
    };
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &live_samples());
    let single = run_scripted_single(tmp.path(), &cfg1, &live_samples());
    assert_lockstep(&ranks, "dapo envelope");
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    eprintln!("measured dapo sharded-vs-single envelope: {envelope:.3e}");
    assert!(
        envelope < 1e-5,
        "sharded run left the reassociation envelope: {envelope:.3e}"
    );
    // The scale oracle: a LOCAL window_tokens normalizer is an exact 2x the
    // weight envelope cannot see (AdamW cancels it) but the norm does.
    assert_grad_norms_match(&ranks[0].1, &single.1, "dapo envelope");
    // Vacuity guard: training actually moved the weights.
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert!(max_abs_diff(&single.0, &init) > 0.0, "no training signal");
}

/// The wraparound case: 5 prompts under a global window of 4, so windows wrap
/// mod len and shards straddle the boundary — the union-of-shards ≡
/// single-run-window identity must hold for ANY prompt count, not just exact
/// fits (the multiset identity under the shared `mod len` cycling).
#[test]
#[allow(clippy::print_stderr)] // the measured envelope is the calibration record
fn world2_matches_single_run_when_the_prompt_cycle_wraps() {
    let tmp = TempDir::new("env-wrap");
    let cfg2 = TrainerConfig {
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let cfg1 = TrainerConfig {
        grad_accum_steps: 4,
        ..scripted_cfg()
    };
    let samples: Vec<Sample<()>> = ["a", "b", "c", "d", "b"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &samples);
    let single = run_scripted_single(tmp.path(), &cfg1, &samples);
    assert_lockstep(&ranks, "wraparound envelope");
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    eprintln!("measured wraparound sharded-vs-single envelope: {envelope:.3e}");
    assert!(
        envelope < 1e-5,
        "sharded run left the reassociation envelope: {envelope:.3e}"
    );
    assert_grad_norms_match(&ranks[0].1, &single.1, "wraparound envelope");
}

// ---- gate 3: world-1 is byte-for-byte the legacy path ------------------------

/// `Trainer::new`, `with_comm(SoloComm)` and a world-1 `LocalComm` must be
/// BIT-identical — the DP plumbing may not perturb the single-rank path (every
/// collective call site is guarded on `world_size() > 1`; this pins the guard
/// discipline). Covered at accumulation 2 (the affine scale path) and 1 (the
/// no-affine bit-identity skip, now keyed on `accum · world == 1`).
#[test]
fn world_one_localcomm_solocomm_and_legacy_are_bit_identical() {
    for accum in [1, 2] {
        let tmp = TempDir::new("world1");
        let cfg = TrainerConfig {
            grad_accum_steps: accum,
            ..scripted_cfg()
        };
        let legacy = run_scripted_single(tmp.path(), &cfg, &live_samples());

        let mut solo_policy = ScriptedPolicy::new(SEED).unwrap();
        let run = RunDir::create(tmp.path(), "solo").unwrap();
        let mut trainer = Trainer::with_comm(cfg.clone(), &run, SoloComm).unwrap();
        trainer
            .train(
                &mut solo_policy,
                &EchoOrFlatReward,
                &CharTokenizer,
                &live_samples(),
            )
            .unwrap();
        assert_eq!(
            legacy.0,
            var_bits(&solo_policy),
            "SoloComm diverged from the legacy constructor at accum {accum}"
        );

        let world1 = run_scripted_world(tmp.path(), 1, &cfg, &live_samples());
        assert_eq!(
            legacy.0, world1[0].0,
            "world-1 LocalComm diverged from the legacy constructor at accum {accum}"
        );
    }
}

/// One separated world-1 ledger window must produce the exact update the
/// ordinary in-process trainer produces from the same learner pre-state.
///
/// The scripted fixture keeps rollouts deterministic while retaining a real
/// LoRA/autograd/Adam path. Two live prompt groups exercise whole-window
/// accumulation, and `mu = 2` makes the frozen old/reference snapshots
/// load-bearing on the second inner update. Comparing optimizer moments as well
/// as adapter values closes Adam's uniform-gradient-scale blind spot.
#[test]
#[allow(clippy::cognitive_complexity)] // one end-to-end oracle keeps every prestate rejection visible
fn world_one_rollout_ledger_update_is_bit_identical_to_direct_training() {
    let tmp = TempDir::new("world1-ledger-equivalence");
    let samples = live_samples();
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 2,
        checkpoint_every: Some(1),
        ..scripted_cfg()
    };
    let policy_sha256 = format!("{:064x}", 1);

    // Direct reference: one ordinary trainer window, including a
    // momentum-faithful checkpoint so its otherwise-private Adam state is
    // observable through the existing public checkpoint API.
    let mut direct_policy = ScriptedPolicy::new(SEED).unwrap();
    let initial_adapter = var_bits(&direct_policy);
    let direct_run = RunDir::create(tmp.path(), "direct").unwrap();
    let mut direct_trainer = Trainer::new(cfg.clone(), &direct_run).unwrap();
    let (direct_history, _) = direct_trainer
        .train(
            &mut direct_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
        )
        .unwrap();
    assert_eq!(direct_history.len(), 1);
    let direct_metrics = &direct_history[0];
    assert!(
        direct_metrics.grad_norm > 0.0,
        "direct reference performed no real optimizer update"
    );
    assert!(
        direct_metrics.kl > 0.0,
        "mu=2/beta>0 reference did not exercise the frozen reference snapshot"
    );
    assert!(
        direct_metrics.step_secs > 0.0 && direct_metrics.tokens_per_sec > 0.0,
        "ordinary direct training did not publish measured whole-window performance"
    );
    let direct_adapter = var_bits(&direct_policy);
    assert_ne!(
        direct_adapter, initial_adapter,
        "direct reference left the adapter at initialization"
    );
    let direct_probe = ScriptedPolicy::new(SEED.wrapping_add(1)).unwrap();
    let direct_checkpoint = ferrl::checkpoint::load_checkpoint(
        direct_run.checkpoints_dir().join("step-1"),
        &direct_probe.trainable_vars(),
    )
    .unwrap();
    let direct_optimizer = direct_checkpoint
        .optimizer_state
        .expect("direct checkpoint must contain Adam state");
    assert_eq!(direct_optimizer.step_t, cfg.mu);

    // Fresh, byte-identical Vars are still a different live optimizer binding.
    // The collector must re-read the policy's current variable set and reject
    // the replacement before publishing a ledger step.
    let replacing_root = tmp.path().join("replacing-rollout-ledger");
    let mut replacing_policy = ReplacingGeneratePolicy {
        inner: ScriptedPolicy::new(SEED).unwrap(),
        seed: SEED,
        replaced: false,
    };
    let replacing_run = RunDir::create(tmp.path(), "replacing-collector").unwrap();
    let mut replacing_collector = Trainer::new(cfg.clone(), &replacing_run).unwrap();
    match replacing_collector.collect_rollout_ledger_step(
        0,
        &mut replacing_policy,
        &EchoOrFlatReward,
        &CharTokenizer,
        &samples,
        &replacing_root,
        &policy_sha256,
        None,
    ) {
        Err(TrainerError::Contract(message)) => assert!(
            message.contains("trainable-variable set changed during rollout collection"),
            "unexpected replacement rejection: {message}"
        ),
        Err(error) => panic!("expected trainable-variable replacement rejection, got {error:?}"),
        Ok(path) => panic!("trainable-variable replacement published {path:?}"),
    }
    assert!(
        std::fs::read_dir(&replacing_root).unwrap().next().is_none(),
        "rejected collection left a reader-visible package"
    );

    // Separated path: collect with one policy instance and consume with an
    // independent instance holding the same learner pre-state.
    let ledger_root = tmp.path().join("rollout-ledger");
    let mut collector_policy = ScriptedPolicy::new(SEED).unwrap();
    let collector_run = RunDir::create(tmp.path(), "collector").unwrap();
    let mut collector = Trainer::new(cfg.clone(), &collector_run).unwrap();
    collector
        .collect_rollout_ledger_step(
            0,
            &mut collector_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
            &ledger_root,
            &policy_sha256,
            None,
        )
        .unwrap();

    // A learner-side replacement is detected after reference-toggle preflight,
    // but rollback cannot falsely claim success by restoring only the now-stale
    // original handles. The combined error makes the policy-discard requirement
    // explicit, while still restoring its independently controlled enable flag.
    let mut replacing_learner = ReplacingTogglePolicy {
        inner: ScriptedPolicy::new(SEED).unwrap(),
        seed: SEED,
        replaced: false,
    };
    let replacing_before = var_bits(&replacing_learner);
    let replacing_ids_before: Vec<_> = replacing_learner
        .trainable_vars()
        .iter()
        .map(|var| var.as_tensor().id())
        .collect();
    let replacing_learner_run = RunDir::create(tmp.path(), "replacing-toggle-learner").unwrap();
    let mut replacing_learner_trainer = Trainer::new(cfg.clone(), &replacing_learner_run).unwrap();
    match replacing_learner_trainer.train_rollout_ledger_step(
        0,
        &mut replacing_learner,
        &ledger_root,
        &policy_sha256,
        None,
    ) {
        Err(TrainerError::Contract(message)) => {
            assert!(
                message.contains(
                    "policy trainable-variable set changed during reference-policy preflight"
                ),
                "missing primary replacement failure: {message}"
            );
            assert!(
                message.contains("coordinated adapter/optimizer/sampler rollback also failed")
                    && message.contains("trainable-variable binding changed"),
                "rollback falsely reported success after rebinding: {message}"
            );
        }
        Err(error) => panic!("expected terminal learner rebinding error, got {error:?}"),
        Ok(_) => panic!("learner trainable-variable replacement unexpectedly succeeded"),
    }
    let replacing_ids_after: Vec<_> = replacing_learner
        .trainable_vars()
        .iter()
        .map(|var| var.as_tensor().id())
        .collect();
    assert_ne!(
        replacing_ids_after, replacing_ids_before,
        "replacement fixture did not change the live Var binding"
    );
    assert_ne!(
        var_bits(&replacing_learner),
        replacing_before,
        "replacement fixture did not leave a distinguishable live adapter"
    );
    assert!(
        replacing_learner.adapter_enabled(),
        "rollback did not restore the independently controlled adapter flag"
    );

    // A one-group, one-epoch update lets live scoring return a graph over the
    // original Vars and then replace the policy binding. Backward coverage and
    // Adam both succeed on those orphaned handles; only the post-update binding
    // barrier can prevent a false successful state and telemetry commit.
    let live_replacement_cfg = TrainerConfig {
        grad_accum_steps: 1,
        mu: 1,
        checkpoint_every: None,
        ..cfg.clone()
    };
    let live_replacement_root = tmp.path().join("live-replacement-rollout-ledger");
    let mut live_replacement_collector_policy = ScriptedPolicy::new(SEED).unwrap();
    let live_replacement_collector_run =
        RunDir::create(tmp.path(), "live-replacement-collector").unwrap();
    let mut live_replacement_collector = Trainer::new(
        live_replacement_cfg.clone(),
        &live_replacement_collector_run,
    )
    .unwrap();
    live_replacement_collector
        .collect_rollout_ledger_step(
            0,
            &mut live_replacement_collector_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
            &live_replacement_root,
            &policy_sha256,
            None,
        )
        .unwrap();

    let mut live_replacement_learner = ReplacingLiveScoringPolicy {
        inner: RefCell::new(ScriptedPolicy::new(SEED).unwrap()),
        seed: SEED,
        replaced: Cell::new(false),
    };
    let live_replacement_ids_before: Vec<_> = live_replacement_learner
        .trainable_vars()
        .iter()
        .map(|var| var.as_tensor().id())
        .collect();
    let live_replacement_run = RunDir::create(tmp.path(), "live-replacement-learner").unwrap();
    let mut live_replacement_trainer =
        Trainer::new(live_replacement_cfg, &live_replacement_run).unwrap();
    match live_replacement_trainer.train_rollout_ledger_step(
        0,
        &mut live_replacement_learner,
        &live_replacement_root,
        &policy_sha256,
        None,
    ) {
        Err(TrainerError::Contract(message)) => {
            assert!(
                message.contains(
                    "policy trainable-variable set changed during rollout-ledger learner update"
                ),
                "missing post-update replacement failure: {message}"
            );
            assert!(
                message.contains("coordinated adapter/optimizer/sampler rollback also failed")
                    && message.contains("trainable-variable binding changed"),
                "live-scoring rebinding did not reach terminal rollback failure: {message}"
            );
        }
        Err(error) => panic!("expected terminal live-scoring rebinding error, got {error:?}"),
        Ok(_) => panic!("live-scoring trainable-variable replacement unexpectedly succeeded"),
    }
    assert!(
        live_replacement_learner.replaced.get(),
        "replacement fixture never reached live scoring"
    );
    let live_replacement_ids_after: Vec<_> = live_replacement_learner
        .trainable_vars()
        .iter()
        .map(|var| var.as_tensor().id())
        .collect();
    assert_ne!(
        live_replacement_ids_after, live_replacement_ids_before,
        "live-scoring fixture did not change the active Var binding"
    );
    assert!(
        ferrl::read_metrics(live_replacement_run.metrics_path())
            .unwrap()
            .is_empty(),
        "rejected live-scoring replacement committed a metrics row"
    );

    // Run orchestration and output controls do not change a single learner
    // update. Each mutation must remain compatible with the collector's ledger
    // while preserving the exact mathematical result.
    let mut longer_run = cfg.clone();
    longer_run.steps += 4;
    let mut different_checkpoint_cadence = cfg.clone();
    different_checkpoint_cadence.checkpoint_every = None;
    let mut candidate_logging = cfg.clone();
    candidate_logging.candidate_log_top_k = 1;
    let mut gpu_probing = cfg.clone();
    gpu_probing.gpu_memory_probe = true;
    for (label, operational_cfg) in [
        ("different run horizon", longer_run),
        ("different checkpoint cadence", different_checkpoint_cadence),
        ("different candidate logging", candidate_logging),
        ("different GPU probing", gpu_probing),
    ] {
        let mut policy = ScriptedPolicy::new(SEED).unwrap();
        let run = RunDir::create(tmp.path(), format!("operational-{label}")).unwrap();
        let mut trainer = Trainer::new(operational_cfg, &run).unwrap();
        let (metrics, continuation) = trainer
            .train_rollout_ledger_step(0, &mut policy, &ledger_root, &policy_sha256, None)
            .unwrap_or_else(|error| panic!("{label} rejected: {error:?}"));
        assert_eq!(
            deterministic_metrics(&metrics),
            deterministic_metrics(direct_metrics),
            "{label}: mathematical metrics diverged"
        );
        assert_eq!(
            var_bits(&policy),
            direct_adapter,
            "{label}: adapter diverged"
        );
        assert_eq!(
            optimizer_bits(continuation.optimizer_state()),
            optimizer_bits(&direct_optimizer),
            "{label}: Adam state diverged"
        );
        assert_ledger_performance_unmeasured(&metrics, label);
    }

    // Both public identity inputs and live adapter values are checked before
    // scoring or mutation. A rejected read must leave the learner untouched.
    let mut wrong_policy = ScriptedPolicy::new(SEED).unwrap();
    let wrong_before = var_bits(&wrong_policy);
    let wrong_run = RunDir::create(tmp.path(), "wrong-policy").unwrap();
    let mut wrong_trainer = Trainer::new(cfg.clone(), &wrong_run).unwrap();
    let wrong_digest = format!("{:064x}", 2);
    assert_ledger_identity_mismatch(
        wrong_trainer.train_rollout_ledger_step(
            0,
            &mut wrong_policy,
            &ledger_root,
            &wrong_digest,
            None,
        ),
        "different policy digest",
    );
    assert_eq!(
        var_bits(&wrong_policy),
        wrong_before,
        "policy-digest rejection mutated the learner adapter"
    );

    let mut stale_adapter_policy = ScriptedPolicy::new(SEED).unwrap();
    let stale_var = stale_adapter_policy.trainable_vars()[1].clone();
    let (rows, cols) = stale_var.as_tensor().dims2().unwrap();
    let mut stale_values = vec![0.0_f32; rows * cols];
    stale_values[0] = 0.125;
    stale_var
        .set(&Tensor::from_vec(stale_values, (rows, cols), &Device::Cpu).unwrap())
        .unwrap();
    let stale_before = var_bits(&stale_adapter_policy);
    let stale_run = RunDir::create(tmp.path(), "stale-adapter").unwrap();
    let mut stale_trainer = Trainer::new(cfg.clone(), &stale_run).unwrap();
    assert_ledger_identity_mismatch(
        stale_trainer.train_rollout_ledger_step(
            0,
            &mut stale_adapter_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ),
        "different live adapter pre-state",
    );
    assert_eq!(
        var_bits(&stale_adapter_policy),
        stale_before,
        "adapter-identity rejection mutated the learner adapter"
    );

    let drift_cfg = TrainerConfig {
        clip_eps: cfg.clip_eps + 0.01,
        ..cfg.clone()
    };
    let mut drift_policy = ScriptedPolicy::new(SEED).unwrap();
    let drift_before = var_bits(&drift_policy);
    let drift_run = RunDir::create(tmp.path(), "config-drift").unwrap();
    let mut drift_trainer = Trainer::new(drift_cfg, &drift_run).unwrap();
    assert_ledger_identity_mismatch(
        drift_trainer.train_rollout_ledger_step(
            0,
            &mut drift_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ),
        "different clip_eps",
    );
    assert_eq!(
        var_bits(&drift_policy),
        drift_before,
        "trainer-config rejection mutated the learner adapter"
    );

    let mut wrong_recipe_policy = RecipeScriptedPolicy {
        inner: ScriptedPolicy::new(SEED).unwrap(),
    };
    assert_eq!(
        var_bits(&wrong_recipe_policy),
        initial_adapter,
        "recipe-drift premise changed the live adapter values"
    );
    let wrong_recipe_before = var_bits(&wrong_recipe_policy);
    let wrong_recipe_run = RunDir::create(tmp.path(), "wrong-recipe").unwrap();
    let mut wrong_recipe_trainer = Trainer::new(cfg.clone(), &wrong_recipe_run).unwrap();
    assert_ledger_identity_mismatch(
        wrong_recipe_trainer.train_rollout_ledger_step(
            0,
            &mut wrong_recipe_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ),
        "different tensor recipe/schema",
    );
    assert_eq!(
        var_bits(&wrong_recipe_policy),
        wrong_recipe_before,
        "tensor-schema rejection mutated the learner adapter"
    );

    #[cfg(target_os = "linux")]
    {
        // `/dev/full` opens successfully but rejects the final metrics write. The
        // same live mu=2 window has already performed real Adam updates when that
        // error is reached, so the learner must restore its complete pre-call state
        // and permit an exact retry from the caller's original optimizer prestate.
        let mut late_failure_policy = ScriptedPolicy::new(SEED).unwrap();
        let late_failure_before = var_bits(&late_failure_policy);
        let late_failure_run = RunDir::create(tmp.path(), "late-metrics-failure").unwrap();
        std::os::unix::fs::symlink("/dev/full", late_failure_run.metrics_path()).unwrap();
        let mut late_failure_trainer = Trainer::new(cfg.clone(), &late_failure_run).unwrap();
        match late_failure_trainer.train_rollout_ledger_step(
            0,
            &mut late_failure_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ) {
            Err(TrainerError::Telemetry(_)) => {}
            Err(error) => panic!("expected forced late metrics failure, got {error:?}"),
            Ok(_) => panic!("/dev/full unexpectedly accepted the learner metrics row"),
        }
        assert_eq!(
            var_bits(&late_failure_policy),
            late_failure_before,
            "late failure left the adapter advanced"
        );
        assert!(
            late_failure_policy.adapter_enabled(),
            "late failure did not restore the adapter-enabled flag"
        );

        let retry_run = RunDir::create(tmp.path(), "late-metrics-retry").unwrap();
        let mut retry_trainer = Trainer::new(cfg.clone(), &retry_run).unwrap();
        let (retry_metrics, retry_continuation) = retry_trainer
            .train_rollout_ledger_step(
                0,
                &mut late_failure_policy,
                &ledger_root,
                &policy_sha256,
                None,
            )
            .unwrap();
        assert_eq!(var_bits(&late_failure_policy), direct_adapter);
        assert_eq!(
            optimizer_bits(retry_continuation.optimizer_state()),
            optimizer_bits(&direct_optimizer),
            "retry after rollback did not recover the exact Adam continuation"
        );
        assert_ledger_performance_unmeasured(&retry_metrics, "retry after rollback");
    }

    let mut ledger_policy = ScriptedPolicy::new(SEED).unwrap();
    assert_eq!(var_bits(&ledger_policy), initial_adapter);
    let ledger_run = RunDir::create(tmp.path(), "learner").unwrap();
    let mut ledger_trainer = Trainer::new(cfg.clone(), &ledger_run).unwrap();
    let (ledger_metrics, ledger_continuation) = ledger_trainer
        .train_rollout_ledger_step(0, &mut ledger_policy, &ledger_root, &policy_sha256, None)
        .unwrap();

    assert_ledger_performance_unmeasured(&ledger_metrics, "returned ledger metrics");
    let persisted_ledger_metrics = ferrl::read_metrics(ledger_run.metrics_path()).unwrap();
    assert_eq!(persisted_ledger_metrics.len(), 1);
    assert_ledger_performance_unmeasured(&persisted_ledger_metrics[0], "persisted ledger metrics");

    assert_eq!(
        deterministic_metrics(&ledger_metrics),
        deterministic_metrics(direct_metrics),
        "ledger learner metrics diverged from direct training"
    );
    assert_eq!(
        var_bits(&ledger_policy),
        direct_adapter,
        "ledger learner adapter diverged from direct training"
    );
    assert_eq!(
        optimizer_bits(ledger_continuation.optimizer_state()),
        optimizer_bits(&direct_optimizer),
        "ledger learner Adam state diverged from direct training"
    );

    let (_, first_moments, second_moments) = optimizer_bits(ledger_continuation.optimizer_state());
    assert!(
        first_moments
            .iter()
            .flatten()
            .any(|&bits| bits != 0.0_f32.to_bits()),
        "first moments stayed zero — optimizer-state equivalence is vacuous"
    );
    assert!(
        second_moments
            .iter()
            .flatten()
            .any(|&bits| bits != 0.0_f32.to_bits()),
        "second moments stayed zero — optimizer-state equivalence is vacuous"
    );
}

/// Ledger v3 must carry a load-bearing sampler transition and lineage across independent
/// collector and learner policies, then persist the combined adapter + Adam +
/// sampler continuation so fresh roles resume the same multi-step trajectory.
#[test]
#[allow(clippy::cognitive_complexity)]
fn world_one_rollout_ledger_sampler_handoff_and_resume_are_bit_exact() {
    let tmp = TempDir::new("world1-ledger-sampler-resume");
    let samples = live_samples();
    let cfg = TrainerConfig {
        steps: 3,
        grad_accum_steps: 2,
        checkpoint_every: Some(1),
        ..scripted_cfg()
    };
    let policy_sha256 = format!("{:064x}", 7);

    let mut direct_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let direct_run = RunDir::create(tmp.path(), "stateful-direct").unwrap();
    let mut direct_trainer = Trainer::new(cfg.clone(), &direct_run).unwrap();
    let (direct_metrics, direct_stop) = direct_trainer
        .train(
            &mut direct_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
        )
        .unwrap();
    assert_eq!(direct_stop, ferrl::RunStop::Completed);
    assert_eq!(direct_metrics.len(), cfg.steps as usize);
    let direct_adapter = var_bits(&direct_policy);
    let direct_sampler = direct_policy.sampler_state().unwrap();
    assert_ne!(
        direct_sampler,
        StatefulScriptedPolicy::new(SEED, 0)
            .unwrap()
            .sampler_state()
            .unwrap(),
        "stateful direct fixture did not advance its sampler"
    );
    let direct_probe = StatefulScriptedPolicy::new(SEED.wrapping_add(1), 999).unwrap();
    let direct_checkpoint = ferrl::load_checkpoint(
        direct_run.checkpoints_dir().join("step-3"),
        &direct_probe.trainable_vars(),
    )
    .unwrap();
    let direct_optimizer = direct_checkpoint.optimizer_state.unwrap();
    assert_eq!(
        direct_checkpoint.sampler_state.as_deref(),
        Some(direct_sampler.as_slice())
    );

    let ledger_root = tmp.path().join("stateful-rollout-ledger");
    let mut collector_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let collector_run = RunDir::create(tmp.path(), "stateful-collector").unwrap();
    let mut collector = Trainer::new(cfg.clone(), &collector_run).unwrap();
    let learner_run = RunDir::create(tmp.path(), "stateful-learner").unwrap();
    let mut learner = Trainer::new(cfg.clone(), &learner_run).unwrap();
    let mut learner_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let mut separated_metrics = Vec::new();

    let first_path = collector
        .collect_rollout_ledger_step(
            0,
            &mut collector_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
            &ledger_root,
            &policy_sha256,
            None,
        )
        .unwrap();
    let collector_after_first = collector_policy.sampler_state().unwrap();

    // Publication is discovered after generation. A duplicate destination must
    // nevertheless restore the collector's sampler exactly for a safe retry.
    let collision_root = tmp.path().join("stateful-collision-ledger");
    std::fs::create_dir_all(collision_root.join(first_path.file_name().unwrap())).unwrap();
    let mut collision_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let collision_before = collision_policy.sampler_state().unwrap();
    let collision_run = RunDir::create(tmp.path(), "stateful-collision-collector").unwrap();
    let mut collision_collector = Trainer::new(cfg.clone(), &collision_run).unwrap();
    assert!(matches!(
        collision_collector.collect_rollout_ledger_step(
            0,
            &mut collision_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
            &collision_root,
            &policy_sha256,
            None,
        ),
        Err(TrainerError::RolloutLedger(
            RolloutLedgerError::AlreadyExists(_)
        ))
    ));
    assert_eq!(
        collision_policy.sampler_state().unwrap(),
        collision_before,
        "failed collector publication advanced the sampler"
    );

    // Adapter/Adam/config equality is insufficient: the learner must begin at
    // the collector's exact sampler prestate before it can consume this window.
    let wrong_sampler_run = RunDir::create(tmp.path(), "wrong-sampler-learner").unwrap();
    let mut wrong_sampler_trainer = Trainer::new(cfg.clone(), &wrong_sampler_run).unwrap();
    let mut wrong_sampler_policy = StatefulScriptedPolicy::new(SEED, 99).unwrap();
    let wrong_sampler_before = var_bits(&wrong_sampler_policy);
    assert!(matches!(
        wrong_sampler_trainer.train_rollout_ledger_step(
            0,
            &mut wrong_sampler_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ),
        Err(TrainerError::RolloutLedger(
            RolloutLedgerError::IdentityMismatch
        ))
    ));
    assert_eq!(var_bits(&wrong_sampler_policy), wrong_sampler_before);
    assert!(
        ferrl::read_metrics(wrong_sampler_run.metrics_path())
            .unwrap()
            .is_empty(),
        "sampler-prestate rejection wrote telemetry"
    );

    let (first_metrics, first_continuation_state) = learner
        .train_rollout_ledger_step(0, &mut learner_policy, &ledger_root, &policy_sha256, None)
        .unwrap();
    assert_eq!(
        learner_policy.sampler_state().unwrap(),
        collector_after_first,
        "ledger did not install the collector's exact post-rollout sampler"
    );
    separated_metrics.push(first_metrics);

    // Only the learner-produced receipt may publish C_1. A same-shaped policy
    // carrying the wrong adapter/sampler state cannot be mixed with its Adam
    // payload, and rejection happens before the destination is claimed.
    let mixed_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    assert!(learner
        .save_rollout_ledger_continuation(&mixed_policy, &first_continuation_state)
        .is_err());
    assert!(!learner_run.checkpoints_dir().join("step-1").exists());

    let first_continuation = learner
        .save_rollout_ledger_continuation(&learner_policy, &first_continuation_state)
        .unwrap();
    assert!(
        learner
            .save_rollout_ledger_continuation(&learner_policy, &first_continuation_state)
            .is_err(),
        "separated continuation silently replaced an existing step"
    );

    let mut unverifiable_recipe_policy = RecipeScriptedPolicy {
        inner: ScriptedPolicy::new(SEED).unwrap(),
    };
    let unverifiable_recipe_before = var_bits(&unverifiable_recipe_policy);
    match learner.restore_rollout_ledger_continuation(
        &first_continuation,
        &mut unverifiable_recipe_policy,
        &policy_sha256,
    ) {
        Err(TrainerError::Contract(message)) => {
            assert!(message.contains("adapter recipe"), "{message}");
        }
        Err(error) => panic!("expected strict continuation recipe error, got {error:?}"),
        Ok(_) => panic!("continuation accepted unverifiable adapter provenance"),
    }
    assert_eq!(
        var_bits(&unverifiable_recipe_policy),
        unverifiable_recipe_before,
        "recipe-preflight rejection mutated the policy"
    );

    // Frozen-policy identity is external to Policy and must be supplied and
    // matched before checkpoint tensors mutate the live model.
    let mut wrong_model_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let wrong_model_before = var_bits(&wrong_model_policy);
    let wrong_model_sha256 = format!("{:064x}", 8);
    match learner.restore_rollout_ledger_continuation(
        &first_continuation,
        &mut wrong_model_policy,
        &wrong_model_sha256,
    ) {
        Err(TrainerError::Contract(message)) => assert!(message.contains("frozen-policy")),
        other => panic!("expected frozen-policy continuation rejection, got {other:?}"),
    }
    assert_eq!(var_bits(&wrong_model_policy), wrong_model_before);

    // Learner-semantic configuration is part of continuation provenance even
    // when tensor shapes and the external policy digest are unchanged.
    let swapped_cfg = TrainerConfig {
        clip_eps: cfg.clip_eps + 0.01,
        ..cfg.clone()
    };
    let swapped_run = RunDir::create(tmp.path(), "swapped-config-continuation").unwrap();
    let swapped_trainer = Trainer::new(swapped_cfg, &swapped_run).unwrap();
    let mut swapped_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let swapped_before = var_bits(&swapped_policy);
    match swapped_trainer.restore_rollout_ledger_continuation(
        &first_continuation,
        &mut swapped_policy,
        &policy_sha256,
    ) {
        Err(TrainerError::Contract(message)) => assert!(message.contains("configuration")),
        other => panic!("expected config-bound continuation rejection, got {other:?}"),
    }
    assert_eq!(var_bits(&swapped_policy), swapped_before);

    // The outer step is bound redundantly by directory, generic manifest, and
    // separated manifest. A renamed/cross-wired package fails before mutation.
    let wrong_step = tmp.path().join("wrong-step").join("step-2");
    std::fs::create_dir_all(&wrong_step).unwrap();
    for entry in std::fs::read_dir(&first_continuation).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), wrong_step.join(entry.file_name())).unwrap();
    }
    let mut wrong_step_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let wrong_step_before = var_bits(&wrong_step_policy);
    match learner.restore_rollout_ledger_continuation(
        &wrong_step,
        &mut wrong_step_policy,
        &policy_sha256,
    ) {
        Err(TrainerError::Contract(message)) => assert!(message.contains("outer step")),
        other => panic!("expected wrong-step continuation rejection, got {other:?}"),
    }
    assert_eq!(var_bits(&wrong_step_policy), wrong_step_before);

    // Adam's bias-correction counter is part of optimizer provenance even when
    // every moment tensor is byte-identical.
    let wrong_adam_step = tmp.path().join("wrong-adam-step").join("step-1");
    std::fs::create_dir_all(&wrong_adam_step).unwrap();
    for entry in std::fs::read_dir(&first_continuation).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), wrong_adam_step.join(entry.file_name())).unwrap();
    }
    let wrong_adam_manifest_path = wrong_adam_step.join("manifest.json");
    let mut wrong_adam_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&wrong_adam_manifest_path).unwrap()).unwrap();
    let original_step_t = wrong_adam_manifest["optimizer_step_t"].as_u64().unwrap();
    wrong_adam_manifest["optimizer_step_t"] = serde_json::json!(original_step_t + 1);
    std::fs::write(
        &wrong_adam_manifest_path,
        serde_json::to_vec_pretty(&wrong_adam_manifest).unwrap(),
    )
    .unwrap();
    let mut wrong_adam_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let wrong_adam_before = var_bits(&wrong_adam_policy);
    match learner.restore_rollout_ledger_continuation(
        &wrong_adam_step,
        &mut wrong_adam_policy,
        &policy_sha256,
    ) {
        Err(TrainerError::Contract(message)) => assert!(message.contains("payload"), "{message}"),
        other => panic!("expected Adam step_t provenance rejection, got {other:?}"),
    }
    assert_eq!(var_bits(&wrong_adam_policy), wrong_adam_before);

    // The receipt's parent/consumed lineage pair must derive its published
    // lineage exactly. A stale or cross-wired parent fails before adapter or
    // sampler mutation.
    let wrong_lineage = tmp.path().join("wrong-lineage").join("step-1");
    std::fs::create_dir_all(&wrong_lineage).unwrap();
    for entry in std::fs::read_dir(&first_continuation).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), wrong_lineage.join(entry.file_name())).unwrap();
    }
    let wrong_lineage_manifest_path = wrong_lineage.join("manifest.json");
    let mut wrong_lineage_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&wrong_lineage_manifest_path).unwrap()).unwrap();
    wrong_lineage_manifest["rollout_ledger_continuation"]["parent_lineage_sha256"] =
        serde_json::json!("f".repeat(64));
    std::fs::write(
        &wrong_lineage_manifest_path,
        serde_json::to_vec_pretty(&wrong_lineage_manifest).unwrap(),
    )
    .unwrap();
    let mut wrong_lineage_policy = StatefulScriptedPolicy::new(SEED.wrapping_add(2), 123).unwrap();
    let wrong_lineage_adapter_before = var_bits(&wrong_lineage_policy);
    let wrong_lineage_sampler_before = wrong_lineage_policy.sampler_state().unwrap();
    match learner.restore_rollout_ledger_continuation(
        &wrong_lineage,
        &mut wrong_lineage_policy,
        &policy_sha256,
    ) {
        Err(TrainerError::Contract(message)) => assert!(message.contains("lineage"), "{message}"),
        other => panic!("expected stale-lineage continuation rejection, got {other:?}"),
    }
    assert_eq!(
        var_bits(&wrong_lineage_policy),
        wrong_lineage_adapter_before
    );
    assert_eq!(
        wrong_lineage_policy.sampler_state().unwrap(),
        wrong_lineage_sampler_before
    );

    // Generic cadence checkpoints are not separated continuations and cannot
    // outrank C_1 during continuation-specific latest discovery.
    ferrl::save_checkpoint(
        learner_run.checkpoints_dir().join("step-999"),
        &learner_policy.trainable_vars(),
        first_continuation_state.optimizer_state(),
        &learner_policy.sampler_state().unwrap(),
        999,
        learner_policy.lora_recipe().as_deref(),
    )
    .unwrap();
    let mut ordinary_discovery_probe = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let discovered = learner
        .restore_latest_rollout_ledger_continuation(&mut ordinary_discovery_probe, &policy_sha256)
        .unwrap()
        .unwrap();
    assert_eq!(discovered.completed_step(), 1);

    // Restart both roles from deliberately wrong policies. The one shared
    // continuation—not process-local memory—must restore adapter, Adam, sampler,
    // and the next outer step.
    collector_policy = StatefulScriptedPolicy::new(SEED.wrapping_add(5), 777).unwrap();
    learner_policy = StatefulScriptedPolicy::new(SEED.wrapping_add(9), 888).unwrap();
    let collector_continuation = collector
        .restore_rollout_ledger_continuation(
            &first_continuation,
            &mut collector_policy,
            &policy_sha256,
        )
        .unwrap();
    let learner_continuation = learner
        .restore_latest_rollout_ledger_continuation(&mut learner_policy, &policy_sha256)
        .unwrap()
        .unwrap();
    assert_eq!(collector_continuation.completed_step(), 1);
    assert_eq!(learner_continuation.completed_step(), 1);
    assert_eq!(
        optimizer_bits(collector_continuation.optimizer_state()),
        optimizer_bits(learner_continuation.optimizer_state())
    );
    let mut continuation = Some(learner_continuation);

    for step in 1..cfg.steps {
        collector
            .collect_rollout_ledger_step(
                step,
                &mut collector_policy,
                &EchoOrFlatReward,
                &CharTokenizer,
                &samples,
                &ledger_root,
                &policy_sha256,
                continuation.as_ref(),
            )
            .unwrap();
        let collector_sampler = collector_policy.sampler_state().unwrap();
        let (metrics, next_continuation) = learner
            .train_rollout_ledger_step(
                step,
                &mut learner_policy,
                &ledger_root,
                &policy_sha256,
                continuation.as_ref(),
            )
            .unwrap();
        assert_eq!(
            learner_policy.sampler_state().unwrap(),
            collector_sampler,
            "step {step} sampler handoff diverged"
        );
        separated_metrics.push(metrics);
        let continuation_path = learner
            .save_rollout_ledger_continuation(&learner_policy, &next_continuation)
            .unwrap();
        continuation = Some(next_continuation);
        if step + 1 < cfg.steps {
            collector_policy = StatefulScriptedPolicy::new(SEED, 1234).unwrap();
            let restored = collector
                .restore_rollout_ledger_continuation(
                    &continuation_path,
                    &mut collector_policy,
                    &policy_sha256,
                )
                .unwrap();
            assert_eq!(restored.completed_step(), step + 1);
            assert_eq!(
                optimizer_bits(restored.optimizer_state()),
                optimizer_bits(continuation.as_ref().unwrap().optimizer_state())
            );
        }
    }

    for (step, (direct, separated)) in direct_metrics.iter().zip(&separated_metrics).enumerate() {
        assert_eq!(
            deterministic_metrics(direct),
            deterministic_metrics(separated),
            "step {step} direct/separated mathematical metrics diverged"
        );
        assert_ledger_performance_unmeasured(separated, "stateful separated step");
    }
    assert_eq!(var_bits(&learner_policy), direct_adapter);
    assert_eq!(learner_policy.sampler_state().unwrap(), direct_sampler);
    assert_eq!(
        optimizer_bits(continuation.as_ref().unwrap().optimizer_state()),
        optimizer_bits(&direct_optimizer)
    );
    assert!(
        optimizer_bits(continuation.as_ref().unwrap().optimizer_state())
            .1
            .iter()
            .flatten()
            .any(|&bits| bits != 0.0_f32.to_bits()),
        "stateful continuation kept vacuous first moments"
    );
    assert_eq!(
        ferrl::read_metrics(learner_run.metrics_path())
            .unwrap()
            .len(),
        cfg.steps as usize
    );

    let mut final_probe = StatefulScriptedPolicy::new(SEED.wrapping_add(17), 4321).unwrap();
    let final_continuation = learner
        .restore_latest_rollout_ledger_continuation(&mut final_probe, &policy_sha256)
        .unwrap()
        .unwrap();
    assert_eq!(final_continuation.completed_step(), cfg.steps);
    assert_eq!(var_bits(&final_probe), direct_adapter);
    assert_eq!(final_probe.sampler_state().unwrap(), direct_sampler);
    assert_eq!(
        optimizer_bits(final_continuation.optimizer_state()),
        optimizer_bits(&direct_optimizer)
    );
}

#[test]
#[allow(clippy::cognitive_complexity)]
fn world_two_rollout_ledger_matches_direct_dp_across_restart() {
    let tmp = TempDir::new("world2-ledger-restart");
    let samples = live_samples();
    let policy_sha256 = format!("{:064x}", 29);
    for (tag, reward_group_scope, grad_accum_steps, group_size) in [
        ("local", RewardGroupScope::Local, 2, 2),
        (
            "same-prompt-singleton",
            RewardGroupScope::DistributedSamePrompt,
            1,
            1,
        ),
    ] {
        let cfg = TrainerConfig {
            steps: 3,
            grad_accum_steps,
            group_size,
            beta: 0.0,
            mu: 2,
            checkpoint_every: Some(1),
            reward_group_scope,
            ..scripted_cfg()
        };
        let direct = run_stateful_direct_dp(tmp.path(), tag, &cfg, &samples);
        let separated = run_stateful_separated_dp(tmp.path(), tag, &cfg, &samples, &policy_sha256);
        assert_eq!(direct.len(), 2);
        assert_eq!(separated.len(), 2);
        assert_eq!(direct[0].adapter, direct[1].adapter, "{tag}: direct ranks");
        assert_eq!(
            separated[0].adapter, separated[1].adapter,
            "{tag}: separated ranks"
        );
        assert_eq!(
            separated[0].optimizer, separated[1].optimizer,
            "{tag}: separated Adam"
        );
        assert_eq!(
            separated[0].sampler, separated[1].sampler,
            "{tag}: separated sampler"
        );
        assert_eq!(separated[0].lineage, separated[1].lineage, "{tag}: lineage");
        for rank in 0..2 {
            assert_eq!(
                separated[rank].adapter, direct[rank].adapter,
                "{tag}: rank {rank} adapter"
            );
            assert_eq!(
                separated[rank].sampler, direct[rank].sampler,
                "{tag}: rank {rank} sampler"
            );
            assert_eq!(
                separated[rank].metrics.len(),
                direct[rank].metrics.len(),
                "{tag}: rank {rank} metric count"
            );
            for (step, (separated_row, direct_row)) in separated[rank]
                .metrics
                .iter()
                .zip(&direct[rank].metrics)
                .enumerate()
            {
                assert_eq!(
                    deterministic_metrics(separated_row),
                    deterministic_metrics(direct_row),
                    "{tag}: rank {rank} step {step} metrics"
                );
                assert_ledger_performance_unmeasured(
                    separated_row,
                    &format!("{tag}: rank {rank} step {step}"),
                );
            }
        }
        let direct_optimizer = direct[0]
            .optimizer
            .as_ref()
            .expect("rank 0 direct checkpoint carries Adam");
        assert_eq!(&separated[0].optimizer, direct_optimizer, "{tag}: Adam");
        assert_ne!(
            separated[0].adapter, separated[0].initial_adapter,
            "{tag}: adapter update was vacuous"
        );
        assert_eq!(
            direct[0].initial_adapter, separated[0].initial_adapter,
            "{tag}: initial adapters"
        );
        assert!(
            separated[0]
                .optimizer
                .1
                .iter()
                .flatten()
                .any(|&bits| bits != 0.0_f32.to_bits()),
            "{tag}: first Adam moments stayed zero"
        );
        assert!(
            separated[0]
                .optimizer
                .2
                .iter()
                .flatten()
                .any(|&bits| bits != 0.0_f32.to_bits()),
            "{tag}: second Adam moments stayed zero"
        );
        assert_ne!(
            separated[0].sampler,
            StatefulScriptedPolicy::new(SEED, 0)
                .unwrap()
                .sampler_state()
                .unwrap(),
            "{tag}: sampler did not advance"
        );

        let step0 = tmp
            .path()
            .join(format!("{tag}-ledger"))
            .join("step-00000000000000000000");
        let shard_bytes = [0, 1]
            .map(|rank| std::fs::read(step0.join(format!("rank-{rank:05}.window.json"))).unwrap());
        assert_ne!(
            shard_bytes[0], shard_bytes[1],
            "{tag}: rank shards are equal"
        );
        for rank in 0..2_u64 {
            let shard: serde_json::Value =
                serde_json::from_slice(&shard_bytes[rank as usize]).unwrap();
            let groups = shard["groups"].as_array().unwrap();
            assert_eq!(groups.len(), grad_accum_steps);
            for (accum_index, group) in groups.iter().enumerate() {
                let expected_prompt = match reward_group_scope {
                    RewardGroupScope::Local => rank * grad_accum_steps as u64 + accum_index as u64,
                    RewardGroupScope::DistributedSamePrompt => accum_index as u64,
                };
                let expected_row =
                    (rank * grad_accum_steps as u64 + accum_index as u64) * group_size as u64;
                assert_eq!(group["prompt_index"].as_u64(), Some(expected_prompt));
                assert_eq!(
                    group["rollout_global_row_base"].as_u64(),
                    Some(expected_row)
                );
                if reward_group_scope == RewardGroupScope::DistributedSamePrompt {
                    assert!(
                        !group["distributed_reward_stats"].is_null(),
                        "same-prompt singleton omitted global reward statistics"
                    );
                    assert!(
                        group["advantage_bits"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|bits| { bits.as_u64() != Some(u64::from(0.0_f32.to_bits())) }),
                        "same-prompt singleton stayed locally degenerate"
                    );
                }
            }
        }
    }

    let world_one_run = RunDir::create(tmp.path(), "topology-mismatch-world1").unwrap();
    let world_one_trainer = Trainer::new(
        TrainerConfig {
            steps: 3,
            grad_accum_steps: 2,
            group_size: 2,
            beta: 0.0,
            mu: 2,
            checkpoint_every: Some(1),
            reward_group_scope: RewardGroupScope::Local,
            ..scripted_cfg()
        },
        &world_one_run,
    )
    .unwrap();
    let mut world_one_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let before = var_bits(&world_one_policy);
    match world_one_trainer.restore_rollout_ledger_continuation(
        tmp.path().join("local-continuations/step-1"),
        &mut world_one_policy,
        &policy_sha256,
    ) {
        Err(TrainerError::Contract(message)) => {
            assert!(message.contains("world size"), "{message}");
        }
        other => panic!("world-one trainer accepted a world-two continuation: {other:?}"),
    }
    assert_eq!(var_bits(&world_one_policy), before);
}

#[test]
#[allow(clippy::cognitive_complexity)] // assertion-heavy coordinated rollback oracle
fn distributed_continuation_restore_rolls_back_after_asymmetric_sampler_failure() {
    let tmp = TempDir::new("distributed-continuation-restore-rollback");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = live_samples();
    let policy_sha256 = format!("{:064x}", 43);
    let _seeded =
        run_stateful_separated_dp(tmp.path(), "restore-source", &cfg, &samples, &policy_sha256);
    let checkpoint = tmp.path().join("restore-source-continuations/step-1");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let base = tmp.path().to_path_buf();
                let checkpoint = checkpoint.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("restore-rank{rank}")).unwrap();
                    let trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut policy = RejectingSamplerRestorePolicy {
                        inner: StatefulScriptedPolicy::new(
                            SEED.wrapping_add(400 + rank as u64),
                            500 + rank as u64,
                        )
                        .unwrap(),
                        fail_next_restore: rank == 1,
                    };
                    let adapter_before = var_bits(&policy);
                    let sampler_before = policy.sampler_state().unwrap();
                    let error = trainer
                        .restore_rollout_ledger_continuation(
                            &checkpoint,
                            &mut policy,
                            &policy_sha256,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&policy),
                        sampler_before,
                        policy.sampler_state().unwrap(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, adapter_before, adapter_after, sampler_before, sampler_after) in outcomes {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        if rank == 0 {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
        } else {
            assert!(
                error.contains("injected rank-local continuation sampler failure"),
                "rank {rank}: {error}"
            );
        }
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // asymmetric save-preflight panic oracle
fn distributed_continuation_save_preflight_panic_aborts_every_rank() {
    let tmp = TempDir::new("distributed-continuation-save-panic");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = live_samples();
    let policy_sha256 = format!("{:064x}", 89);
    let ledger_root = tmp.path().join("ledger");
    let checkpoint_root = tmp.path().join("continuations");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let checkpoint_root = checkpoint_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("save-panic-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut collector = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    let mut learner = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut collector,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                    let (_, continuation) = trainer
                        .train_rollout_ledger_step(
                            0,
                            &mut learner,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                    let adapter_before = var_bits(&learner);
                    let sampler_before = learner.sampler_state().unwrap();
                    let policy = ContinuationFaultPolicy {
                        inner: learner,
                        panic_next_trainable_vars: Cell::new(rank == 1),
                        fail_restore_call: None,
                        panic_restore_call: None,
                        arm_collective_failure_on_restore_call: None,
                        restore_calls: 0,
                    };
                    let error = trainer
                        .save_rollout_ledger_continuation_to(
                            &checkpoint_root,
                            &policy,
                            &continuation,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&policy),
                        sampler_before,
                        policy.sampler_state().unwrap(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, adapter_before, adapter_after, sampler_before, sampler_after) in outcomes {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        if rank == 1 {
            assert!(
                error.contains(
                    "continuation save preflight panicked: injected continuation trainable-vars panic"
                ),
                "rank {rank}: {error}"
            );
        } else {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
        }
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
    }
    assert!(!checkpoint_root.join("step-1").exists());
}

#[test]
#[allow(clippy::cognitive_complexity)] // asymmetric restore-snapshot panic oracle
fn distributed_continuation_restore_snapshot_panic_aborts_every_rank() {
    let tmp = TempDir::new("distributed-continuation-snapshot-panic");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = live_samples();
    let policy_sha256 = format!("{:064x}", 97);
    let _seeded = run_stateful_separated_dp(
        tmp.path(),
        "snapshot-source",
        &cfg,
        &samples,
        &policy_sha256,
    );
    let checkpoint = tmp.path().join("snapshot-source-continuations/step-1");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let base = tmp.path().to_path_buf();
                let checkpoint = checkpoint.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("snapshot-panic-rank{rank}")).unwrap();
                    let trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let inner = StatefulScriptedPolicy::new(
                        SEED.wrapping_add(600 + rank as u64),
                        700 + rank as u64,
                    )
                    .unwrap();
                    let adapter_before = var_bits(&inner);
                    let sampler_before = inner.sampler_state().unwrap();
                    let mut policy = ContinuationFaultPolicy {
                        inner,
                        panic_next_trainable_vars: Cell::new(rank == 0),
                        fail_restore_call: None,
                        panic_restore_call: None,
                        arm_collective_failure_on_restore_call: None,
                        restore_calls: 0,
                    };
                    let error = trainer
                        .restore_rollout_ledger_continuation(
                            &checkpoint,
                            &mut policy,
                            &policy_sha256,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&policy),
                        sampler_before,
                        policy.sampler_state().unwrap(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, adapter_before, adapter_after, sampler_before, sampler_after) in outcomes {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        if rank == 0 {
            assert!(
                error.contains(
                    "continuation restore snapshot panicked: injected continuation trainable-vars panic"
                ),
                "rank {rank}: {error}"
            );
        } else {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
        }
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // asymmetric primary failure plus rollback panic
fn distributed_continuation_rollback_panic_requires_discarding_every_rank() {
    let tmp = TempDir::new("distributed-continuation-rollback-panic");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = live_samples();
    let policy_sha256 = format!("{:064x}", 101);
    let _seeded = run_stateful_separated_dp(
        tmp.path(),
        "rollback-panic-source",
        &cfg,
        &samples,
        &policy_sha256,
    );
    let checkpoint = tmp
        .path()
        .join("rollback-panic-source-continuations/step-1");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let base = tmp.path().to_path_buf();
                let checkpoint = checkpoint.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("rollback-panic-rank{rank}")).unwrap();
                    let trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let inner = StatefulScriptedPolicy::new(
                        SEED.wrapping_add(800 + rank as u64),
                        900 + rank as u64,
                    )
                    .unwrap();
                    let adapter_before = var_bits(&inner);
                    let sampler_before = inner.sampler_state().unwrap();
                    let mut policy = ContinuationFaultPolicy {
                        inner,
                        panic_next_trainable_vars: Cell::new(false),
                        fail_restore_call: (rank == 0).then_some(1),
                        panic_restore_call: (rank == 1).then_some(2),
                        arm_collective_failure_on_restore_call: None,
                        restore_calls: 0,
                    };
                    let error = trainer
                        .restore_rollout_ledger_continuation(
                            &checkpoint,
                            &mut policy,
                            &policy_sha256,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&policy),
                        sampler_before,
                        policy.sampler_state().unwrap(),
                        policy.restore_calls,
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
        error,
        adapter_before,
        adapter_after,
        sampler_before,
        sampler_after,
        restore_calls,
    ) in outcomes
    {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        assert!(
            error.contains("discard the policy state on every rank"),
            "rank {rank}: {error}"
        );
        if rank == 0 {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
            assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
        } else {
            assert!(
                error.contains("injected continuation rollback panic"),
                "rank {rank}: {error}"
            );
            assert_ne!(
                sampler_after, sampler_before,
                "rank {rank} fixture did not leave partially restored state"
            );
        }
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(restore_calls, 2, "rank {rank} restore calls");
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // terminal communicator oracle at two restore boundaries
fn distributed_continuation_comm_failure_never_enters_another_collective() {
    let tmp = TempDir::new("distributed-continuation-terminal-comm");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = live_samples();
    let policy_sha256 = format!("{:064x}", 103);
    let _seeded = run_stateful_separated_dp(
        tmp.path(),
        "terminal-comm-source",
        &cfg,
        &samples,
        &policy_sha256,
    );
    let checkpoint = tmp.path().join("terminal-comm-source-continuations/step-1");

    // Arming occurs after the checkpoint sampler is installed. With no allowed
    // successes the restore-status reduction fails; with two, restore status
    // and continuation serialization succeed before the first byte-consensus
    // word fails. Both failures poison the world immediately.
    for (phase, successes_after_arm) in [("restore-status", 0), ("consensus", 2)] {
        let failure = Arc::new(InjectedCollectiveFailureState::new(successes_after_arm));
        let comm = InjectedFailureComm {
            state: Arc::clone(&failure),
        };
        let run = RunDir::create(tmp.path(), format!("terminal-comm-{phase}")).unwrap();
        let trainer = Trainer::with_comm(cfg.clone(), &run, comm).unwrap();
        let inner = StatefulScriptedPolicy::new(
            SEED.wrapping_add(1_000 + successes_after_arm as u64),
            1_100 + successes_after_arm as u64,
        )
        .unwrap();
        let adapter_before = var_bits(&inner);
        let sampler_before = inner.sampler_state().unwrap();
        let mut policy = ContinuationFaultPolicy {
            inner,
            panic_next_trainable_vars: Cell::new(false),
            fail_restore_call: None,
            panic_restore_call: None,
            arm_collective_failure_on_restore_call: Some((1, Arc::clone(&failure))),
            restore_calls: 0,
        };

        let error = trainer
            .restore_rollout_ledger_continuation(&checkpoint, &mut policy, &policy_sha256)
            .unwrap_err();
        let TrainerError::Contract(message) = error else {
            panic!("{phase}: expected terminal discard-world classification, got {error:?}");
        };
        assert!(
            message.contains("the data-parallel world is dead"),
            "{phase}: {message}"
        );
        assert!(
            message.contains("no further collectives are safe"),
            "{phase}: {message}"
        );
        assert!(
            message.contains("discard the policy state on every rank"),
            "{phase}: {message}"
        );
        assert!(
            failure.failed.load(Ordering::SeqCst),
            "{phase}: communicator fault was not consumed"
        );
        assert_eq!(
            failure.calls_after_failure.load(Ordering::SeqCst),
            0,
            "{phase}: restore issued a collective after the world became dead"
        );
        assert_eq!(
            failure.remaining_successes.load(Ordering::SeqCst),
            0,
            "{phase}: communicator failed at the wrong restore boundary"
        );
        assert_eq!(policy.restore_calls, 2, "{phase}: local rollback calls");
        assert_eq!(
            var_bits(&policy),
            adapter_before,
            "{phase}: adapter rollback"
        );
        assert_eq!(
            policy.sampler_state().unwrap(),
            sampler_before,
            "{phase}: sampler rollback"
        );
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // collector/learner x error/panic terminal matrix
fn distributed_collector_and_learner_comm_failures_never_coordinate_recovery() {
    let tmp = TempDir::new("distributed-terminal-recovery");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = live_samples();
    let policy_sha256 = format!("{:064x}", 107);
    let ledger_root = tmp.path().join("learner-source-ledger");

    // Publish one ordinary world-two ledger so the learner cases reach
    // detached scoring before arming the injected communicator failure.
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("terminal-source-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });

    for phase in [
        DeadCommPhase::CollectorGeneration,
        DeadCommPhase::LearnerDetachedScoring,
    ] {
        for rollback_fault in [
            DeadCommRollbackFault::ReturnError,
            DeadCommRollbackFault::Panic,
        ] {
            let phase_label = match phase {
                DeadCommPhase::CollectorGeneration => "collector",
                DeadCommPhase::LearnerDetachedScoring => "learner",
            };
            let fault_label = match rollback_fault {
                DeadCommRollbackFault::ReturnError => "error",
                DeadCommRollbackFault::Panic => "panic",
            };
            let case = format!("{phase_label}-{fault_label}");
            let failure = Arc::new(InjectedCollectiveFailureState::new(0));
            let comm = InjectedFailureComm {
                state: Arc::clone(&failure),
            };
            let run = RunDir::create(tmp.path(), format!("terminal-{case}")).unwrap();
            let mut trainer = Trainer::with_comm(cfg.clone(), &run, comm).unwrap();
            let mut policy = DeadCommRecoveryPolicy {
                inner: StatefulScriptedPolicy::new(SEED, 0).unwrap(),
                state: Arc::clone(&failure),
                phase,
                rollback_fault,
                post_failure_restore_calls: 0,
            };
            let sampler_before = policy.sampler_state().unwrap();

            let error = match phase {
                DeadCommPhase::CollectorGeneration => trainer
                    .collect_rollout_ledger_step(
                        0,
                        &mut policy,
                        &EchoOrFlatReward,
                        &CharTokenizer,
                        &samples,
                        tmp.path().join(format!("terminal-{case}-ledger")),
                        &policy_sha256,
                        None,
                    )
                    .unwrap_err(),
                DeadCommPhase::LearnerDetachedScoring => trainer
                    .train_rollout_ledger_step(0, &mut policy, &ledger_root, &policy_sha256, None)
                    .unwrap_err(),
            };
            let TrainerError::Contract(message) = error else {
                panic!("{case}: expected terminal discard-world classification, got {error:?}");
            };
            assert!(
                message.contains("the data-parallel world is dead"),
                "{case}: {message}"
            );
            assert!(
                message.contains("no further collectives are safe"),
                "{case}: {message}"
            );
            let expected_discard = match phase {
                DeadCommPhase::CollectorGeneration => "discard the policy instance on every rank",
                DeadCommPhase::LearnerDetachedScoring => {
                    "discard the policy and optimizer state on every rank"
                }
            };
            assert!(message.contains(expected_discard), "{case}: {message}");
            let expected_rollback = match rollback_fault {
                DeadCommRollbackFault::ReturnError => {
                    "injected post-communication local rollback failure"
                }
                DeadCommRollbackFault::Panic => "injected post-communication local rollback panic",
            };
            assert!(message.contains(expected_rollback), "{case}: {message}");
            assert!(
                failure.failed.load(Ordering::SeqCst),
                "{case}: communicator fault was not consumed"
            );
            assert_eq!(
                failure.calls_after_failure.load(Ordering::SeqCst),
                0,
                "{case}: recovery issued a collective after the world became dead"
            );
            assert_eq!(
                failure.remaining_successes.load(Ordering::SeqCst),
                0,
                "{case}: communicator failed at the wrong boundary"
            );
            assert_eq!(
                policy.post_failure_restore_calls, 1,
                "{case}: local rollback fault was not exercised exactly once"
            );
            assert_eq!(
                ferrl::read_metrics(run.metrics_path()).unwrap().len(),
                0,
                "{case}: terminal failure published metrics"
            );
            if phase == DeadCommPhase::CollectorGeneration {
                assert_ne!(
                    policy.sampler_state().unwrap(),
                    sampler_before,
                    "{case}: collector did not make progress before rollback failed"
                );
            }
        }
    }
}

#[test]
fn distributed_latest_continuation_scan_failure_is_coordinated_without_mutation() {
    let tmp = TempDir::new("distributed-continuation-scan-failure");
    let invalid_root = tmp.path().join("not-a-directory");
    std::fs::write(&invalid_root, b"not a checkpoint root").unwrap();
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        ..scripted_cfg()
    };
    let policy_sha256 = format!("{:064x}", 47);
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let base = tmp.path().to_path_buf();
                let invalid_root = invalid_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("scan-rank{rank}")).unwrap();
                    let trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut policy =
                        StatefulScriptedPolicy::new(SEED.wrapping_add(rank as u64), rank as u64)
                            .unwrap();
                    let adapter_before = var_bits(&policy);
                    let sampler_before = policy.sampler_state().unwrap();
                    let error = trainer
                        .restore_latest_rollout_ledger_continuation_from(
                            &invalid_root,
                            &mut policy,
                            &policy_sha256,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&policy),
                        sampler_before,
                        policy.sampler_state().unwrap(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, adapter_before, adapter_after, sampler_before, sampler_after) in outcomes {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        if rank == 1 {
            assert!(error.contains("failed to discover"), "rank {rank}: {error}");
        }
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // package inspection plus two-rank equivalence assertions
fn separated_dp_empty_local_shard_still_joins_the_global_update() {
    let tmp = TempDir::new("separated-empty-shard");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        checkpoint_every: Some(1),
        reward_group_scope: RewardGroupScope::Local,
        ..scripted_cfg()
    };
    let samples = vec![Sample::new("e", ()), Sample::new("a", ())];
    let policy_sha256 = format!("{:064x}", 31);
    let ranks =
        run_stateful_separated_dp(tmp.path(), "empty-local", &cfg, &samples, &policy_sha256);
    assert_eq!(ranks[0].adapter, ranks[1].adapter);
    assert_ne!(ranks[0].adapter, ranks[0].initial_adapter);
    assert!(ranks[0].metrics[0].frac_reward_zero_std > 0.0);

    let step = tmp
        .path()
        .join("empty-local-ledger")
        .join("step-00000000000000000000");
    let rank0: serde_json::Value =
        serde_json::from_slice(&std::fs::read(step.join("rank-00000.window.json")).unwrap())
            .unwrap();
    let rank1: serde_json::Value =
        serde_json::from_slice(&std::fs::read(step.join("rank-00001.window.json")).unwrap())
            .unwrap();
    assert_eq!(rank0["old_logprobs"], "not_required");
    assert_eq!(rank1["old_logprobs"], "adapter_enabled_detached");
    assert_eq!(rank0["live_items"].as_u64(), Some(1));
    assert_eq!(rank1["live_items"].as_u64(), Some(1));
}

#[test]
#[allow(clippy::cognitive_complexity)] // paired generation/reward failure variants
fn separated_dp_asymmetric_collection_failures_rewind_every_sampler_before_stats() {
    for mode in ["generation", "reward"] {
        let tmp = TempDir::new(&format!("separated-collection-{mode}"));
        let cfg = TrainerConfig {
            steps: 1,
            grad_accum_steps: 1,
            group_size: 2,
            beta: 0.0,
            mu: 1,
            ..scripted_cfg()
        };
        let samples = live_samples();
        let policy_sha256 = format!("{:064x}", 59);
        let ledger_root = tmp.path().join("ledger");
        let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let cfg = cfg.clone();
                    let samples = samples.clone();
                    let base = tmp.path().to_path_buf();
                    let ledger_root = ledger_root.clone();
                    let policy_sha256 = policy_sha256.clone();
                    scope.spawn(move || {
                        let run = RunDir::create(&base, format!("{mode}-rank{rank}")).unwrap();
                        let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                        let mut policy = FailingCollectionPolicy {
                            inner: StatefulScriptedPolicy::new(SEED, 0).unwrap(),
                            fail_generate: mode == "generation" && rank == 0,
                        };
                        let adapter_before = var_bits(&policy);
                        let sampler_before = policy.sampler_state().unwrap();
                        let error = trainer
                            .collect_rollout_ledger_step(
                                0,
                                &mut policy,
                                &FailingCollectionReward {
                                    fail: mode == "reward" && rank == 0,
                                },
                                &CharTokenizer,
                                &samples,
                                &ledger_root,
                                &policy_sha256,
                                None,
                            )
                            .unwrap_err();
                        (
                            rank,
                            error.to_string(),
                            adapter_before,
                            var_bits(&policy),
                            sampler_before,
                            policy.sampler_state().unwrap(),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });
        for (rank, error, adapter_before, adapter_after, sampler_before, sampler_after) in outcomes
        {
            assert!(!error.contains("timeout"), "{mode}: rank {rank}: {error}");
            assert_eq!(adapter_after, adapter_before, "{mode}: rank {rank} adapter");
            assert_eq!(sampler_after, sampler_before, "{mode}: rank {rank} sampler");
        }
        if ledger_root.exists() {
            assert!(std::fs::read_dir(&ledger_root).unwrap().all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("step-")
            }));
        }
    }
}

#[test]
#[allow(clippy::type_complexity)] // explicit before/after rollback tuple
fn separated_dp_asymmetric_scoring_failures_abort_without_collective_timeout() {
    for (mode, fail_detached, fail_live) in
        [("detached", true, false), ("live-scoring", false, true)]
    {
        let tmp = TempDir::new(&format!("separated-asymmetric-{mode}"));
        let cfg = TrainerConfig {
            steps: 1,
            grad_accum_steps: 1,
            group_size: 2,
            beta: 0.0,
            mu: 1,
            ..scripted_cfg()
        };
        let samples = vec![Sample::new("a", ()), Sample::new("b", ())];
        let policy_sha256 = format!("{:064x}", 37);
        let ledger_root = tmp.path().join("ledger");
        let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<(String, Vec<Vec<u32>>, Vec<Vec<u32>>, usize)> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = comms
                    .into_iter()
                    .map(|comm| {
                        let cfg = cfg.clone();
                        let samples = samples.clone();
                        let base = tmp.path().to_path_buf();
                        let ledger_root = ledger_root.clone();
                        let policy_sha256 = policy_sha256.clone();
                        scope.spawn(move || {
                            let rank = comm.rank();
                            let run = RunDir::create(&base, format!("{mode}-rank{rank}")).unwrap();
                            let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                            let mut collector_policy = ScriptedPolicy::new(SEED).unwrap();
                            trainer
                                .collect_rollout_ledger_step(
                                    0,
                                    &mut collector_policy,
                                    &EchoOrFlatReward,
                                    &CharTokenizer,
                                    &samples,
                                    &ledger_root,
                                    &policy_sha256,
                                    None,
                                )
                                .unwrap();
                            let mut learner_policy = FailingScoringPolicy {
                                inner: ScriptedPolicy::new(SEED).unwrap(),
                                fail_detached: rank == 0 && fail_detached,
                                fail_live: rank == 0 && fail_live,
                            };
                            let before = var_bits(&learner_policy);
                            let error = trainer
                                .train_rollout_ledger_step(
                                    0,
                                    &mut learner_policy,
                                    &ledger_root,
                                    &policy_sha256,
                                    None,
                                )
                                .unwrap_err();
                            (
                                error.to_string(),
                                before,
                                var_bits(&learner_policy),
                                ferrl::read_metrics(run.metrics_path()).unwrap().len(),
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect()
            });
        for (rank, (error, before, after, metric_rows)) in outcomes.iter().enumerate() {
            assert!(
                !error.contains("timeout"),
                "{mode}: rank {rank} stalled instead of aborting: {error}"
            );
            assert_eq!(after, before, "{mode}: rank {rank} was not rolled back");
            assert_eq!(*metric_rows, 0, "{mode}: rank {rank} wrote metrics");
        }
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // asymmetric panic and rollback assertions
fn separated_dp_asymmetric_detached_scoring_panic_aborts_and_rolls_back_every_rank() {
    let tmp = TempDir::new("separated-asymmetric-detached-panic");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = vec![Sample::new("a", ()), Sample::new("b", ())];
    let policy_sha256 = format!("{:064x}", 71);
    let ledger_root = tmp.path().join("ledger");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("panic-score-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut collector_policy = ScriptedPolicy::new(SEED).unwrap();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut collector_policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                    let mut learner_policy = PanickingDetachedScoringPolicy {
                        inner: ScriptedPolicy::new(SEED).unwrap(),
                        panic_detached: rank == 0,
                    };
                    let before = var_bits(&learner_policy);
                    let error = trainer
                        .train_rollout_ledger_step(
                            0,
                            &mut learner_policy,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        before,
                        var_bits(&learner_policy),
                        ferrl::read_metrics(run.metrics_path()).unwrap().len(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, before, after, rows) in outcomes {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        if rank == 0 {
            assert!(
                error.contains("detached scoring panicked: injected detached scoring panic"),
                "rank {rank}: {error}"
            );
        } else {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
        }
        assert_eq!(after, before, "rank {rank} adapter");
        assert_eq!(rows, 0, "rank {rank} wrote metrics");
    }
}

#[test]
fn separated_dp_asymmetric_backward_failure_aborts_before_gradient_collectives() {
    let tmp = TempDir::new("separated-asymmetric-backward");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = vec![Sample::new("a", ()), Sample::new("b", ())];
    let policy_sha256 = format!("{:064x}", 61);
    let ledger_root = tmp.path().join("ledger");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("backward-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut collector_policy = ScriptedPolicy::new(SEED).unwrap();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut collector_policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                    let mut learner_policy = FailingBackwardPolicy {
                        inner: ScriptedPolicy::new(SEED).unwrap(),
                        fail_backward: rank == 0,
                    };
                    let before = var_bits(&learner_policy);
                    let error = trainer
                        .train_rollout_ledger_step(
                            0,
                            &mut learner_policy,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        before,
                        var_bits(&learner_policy),
                        ferrl::read_metrics(run.metrics_path()).unwrap().len(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, before, after, rows) in outcomes {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        assert_eq!(after, before, "rank {rank} adapter");
        assert_eq!(rows, 0, "rank {rank} wrote metrics");
    }
}

#[test]
fn separated_dp_asymmetric_sampler_handoff_failure_rolls_back_every_rank() {
    let tmp = TempDir::new("separated-asymmetric-sampler-handoff");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = vec![Sample::new("a", ()), Sample::new("b", ())];
    let policy_sha256 = format!("{:064x}", 67);
    let ledger_root = tmp.path().join("ledger");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("handoff-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut collector_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut collector_policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                    let mut learner_policy = RejectingSamplerRestorePolicy {
                        inner: StatefulScriptedPolicy::new(SEED, 0).unwrap(),
                        fail_next_restore: rank == 1,
                    };
                    let adapter_before = var_bits(&learner_policy);
                    let sampler_before = learner_policy.sampler_state().unwrap();
                    let error = trainer
                        .train_rollout_ledger_step(
                            0,
                            &mut learner_policy,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&learner_policy),
                        sampler_before,
                        learner_policy.sampler_state().unwrap(),
                        ferrl::read_metrics(run.metrics_path()).unwrap().len(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, adapter_before, adapter_after, sampler_before, sampler_after, rows) in
        outcomes
    {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
        assert_eq!(rows, 0, "rank {rank} wrote metrics");
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // asymmetric panic and rollback assertions
fn separated_dp_asymmetric_sampler_handoff_panic_aborts_and_rolls_back_every_rank() {
    let tmp = TempDir::new("separated-asymmetric-sampler-handoff-panic");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = vec![Sample::new("a", ()), Sample::new("b", ())];
    let policy_sha256 = format!("{:064x}", 73);
    let ledger_root = tmp.path().join("ledger");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("panic-handoff-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut collector_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut collector_policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                    let mut learner_policy = PanickingSamplerRestorePolicy {
                        inner: StatefulScriptedPolicy::new(SEED, 0).unwrap(),
                        panic_next_restore: rank == 1,
                    };
                    let adapter_before = var_bits(&learner_policy);
                    let sampler_before = learner_policy.sampler_state().unwrap();
                    let error = trainer
                        .train_rollout_ledger_step(
                            0,
                            &mut learner_policy,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&learner_policy),
                        sampler_before,
                        learner_policy.sampler_state().unwrap(),
                        ferrl::read_metrics(run.metrics_path()).unwrap().len(),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, adapter_before, adapter_after, sampler_before, sampler_after, rows) in
        outcomes
    {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        if rank == 1 {
            assert!(
                error.contains("post-update state panicked: injected sampler handoff panic"),
                "rank {rank}: {error}"
            );
        } else {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
        }
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
        assert_eq!(rows, 0, "rank {rank} wrote metrics");
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // asymmetric failure and terminal-state assertions
fn separated_dp_collector_rollback_failure_requires_discarding_every_rank() {
    let tmp = TempDir::new("separated-collector-terminal-rollback");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = vec![Sample::new("a", ()), Sample::new("b", ())];
    let policy_sha256 = format!("{:064x}", 79);
    let ledger_root = tmp.path().join("ledger");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("collector-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut policy = PersistentRollbackFailurePolicy {
                        inner: StatefulScriptedPolicy::new(SEED, 0).unwrap(),
                        fail_generate: rank == 0,
                        fail_restore: rank == 1,
                        restore_calls: 0,
                    };
                    let sampler_before = policy.sampler_state().unwrap();
                    let error = trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        sampler_before,
                        policy.sampler_state().unwrap(),
                        policy.restore_calls,
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    for (rank, error, sampler_before, sampler_after, restore_calls) in outcomes {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        assert!(
            error.contains("discard the policy instance on every rank"),
            "rank {rank}: {error}"
        );
        if rank == 0 {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
            assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
        } else {
            assert!(
                error.contains("persistent rank-local sampler restore failure"),
                "rank {rank}: {error}"
            );
            assert_ne!(
                sampler_after, sampler_before,
                "rank {rank} fixture did not leave unrestored sampler state"
            );
        }
        assert_eq!(restore_calls, 1, "rank {rank} rollback attempts");
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // asymmetric failure and terminal-state assertions
fn separated_dp_persistent_learner_rollback_failure_requires_discarding_every_rank() {
    let tmp = TempDir::new("separated-learner-terminal-rollback");
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let samples = vec![Sample::new("a", ()), Sample::new("b", ())];
    let policy_sha256 = format!("{:064x}", 83);
    let ledger_root = tmp.path().join("ledger");
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let rank = comm.rank();
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                let ledger_root = ledger_root.clone();
                let policy_sha256 = policy_sha256.clone();
                scope.spawn(move || {
                    let run = RunDir::create(&base, format!("learner-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let mut collector_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
                    trainer
                        .collect_rollout_ledger_step(
                            0,
                            &mut collector_policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &samples,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap();
                    let mut learner_policy = PersistentRollbackFailurePolicy {
                        inner: StatefulScriptedPolicy::new(SEED, 0).unwrap(),
                        fail_generate: false,
                        fail_restore: rank == 1,
                        restore_calls: 0,
                    };
                    let adapter_before = var_bits(&learner_policy);
                    let sampler_before = learner_policy.sampler_state().unwrap();
                    let error = trainer
                        .train_rollout_ledger_step(
                            0,
                            &mut learner_policy,
                            &ledger_root,
                            &policy_sha256,
                            None,
                        )
                        .unwrap_err();
                    (
                        rank,
                        error.to_string(),
                        adapter_before,
                        var_bits(&learner_policy),
                        sampler_before,
                        learner_policy.sampler_state().unwrap(),
                        learner_policy.restore_calls,
                        ferrl::read_metrics(run.metrics_path()).unwrap().len(),
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
        error,
        adapter_before,
        adapter_after,
        sampler_before,
        sampler_after,
        restore_calls,
        rows,
    ) in outcomes
    {
        assert!(!error.contains("timeout"), "rank {rank}: {error}");
        assert!(
            error.contains("discard the policy and optimizer state on every rank"),
            "rank {rank}: {error}"
        );
        if rank == 0 {
            assert!(error.contains("peer rank"), "rank {rank}: {error}");
        } else {
            assert!(
                error.contains("persistent rank-local sampler restore failure"),
                "rank {rank}: {error}"
            );
        }
        assert_eq!(adapter_after, adapter_before, "rank {rank} adapter");
        assert_eq!(sampler_after, sampler_before, "rank {rank} sampler");
        assert_eq!(restore_calls, 2, "rank {rank} handoff plus rollback");
        assert_eq!(rows, 0, "rank {rank} wrote metrics");
    }
}

#[test]
fn separated_dp_rejects_config_and_sampler_mismatch_before_publication() {
    for mismatch in ["reward-scope", "sampler"] {
        let tmp = TempDir::new(&format!("separated-preflight-{mismatch}"));
        let ledger_root = tmp.path().join("ledger");
        let samples = live_samples();
        let policy_sha256 = format!("{:064x}", 41);
        let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(10));
        let outcomes: Vec<(String, Vec<u8>, Vec<u8>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let rank = comm.rank();
                    let cfg = TrainerConfig {
                        steps: 1,
                        grad_accum_steps: 1,
                        beta: 0.0,
                        reward_group_scope: if mismatch == "reward-scope" && rank == 1 {
                            RewardGroupScope::DistributedSamePrompt
                        } else {
                            RewardGroupScope::Local
                        },
                        ..scripted_cfg()
                    };
                    let samples = samples.clone();
                    let base = tmp.path().to_path_buf();
                    let ledger_root = ledger_root.clone();
                    let policy_sha256 = policy_sha256.clone();
                    scope.spawn(move || {
                        let run = RunDir::create(&base, format!("{mismatch}-rank{rank}")).unwrap();
                        let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                        let mut policy = StatefulScriptedPolicy::new(
                            SEED,
                            if mismatch == "sampler" {
                                rank as u64
                            } else {
                                0
                            },
                        )
                        .unwrap();
                        let before = policy.sampler_state().unwrap();
                        let error = trainer
                            .collect_rollout_ledger_step(
                                0,
                                &mut policy,
                                &EchoOrFlatReward,
                                &CharTokenizer,
                                &samples,
                                &ledger_root,
                                &policy_sha256,
                                None,
                            )
                            .unwrap_err();
                        (error.to_string(), before, policy.sampler_state().unwrap())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });
        for (rank, (error, before, after)) in outcomes.iter().enumerate() {
            assert!(
                !error.contains("timeout"),
                "{mismatch}: rank {rank}: {error}"
            );
            assert_eq!(after, before, "{mismatch}: rank {rank} sampler advanced");
        }
        if ledger_root.exists() {
            assert!(std::fs::read_dir(&ledger_root).unwrap().all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("step-")
            }));
        }
    }
}

#[test]
fn relative_run_dir_continuation_saves_and_restores() {
    let root = RelativeTempRoot::new("continuation");
    let runs_root = root.path().join("runs");
    let ledger_root = root.path().join("ledger");
    let samples = live_samples();
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        ..scripted_cfg()
    };
    let policy_sha256 = format!("{:064x}", 17);

    let collector_run = RunDir::create(&runs_root, "collector").unwrap();
    let mut collector = Trainer::new(cfg.clone(), &collector_run).unwrap();
    let mut collector_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    collector
        .collect_rollout_ledger_step(
            0,
            &mut collector_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
            &ledger_root,
            &policy_sha256,
            None,
        )
        .unwrap();

    let learner_run = RunDir::create(&runs_root, "learner").unwrap();
    let mut learner = Trainer::new(cfg.clone(), &learner_run).unwrap();
    let mut learner_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let (_, continuation) = learner
        .train_rollout_ledger_step(0, &mut learner_policy, &ledger_root, &policy_sha256, None)
        .unwrap();
    let expected_adapter = var_bits(&learner_policy);
    let expected_sampler = learner_policy.sampler_state().unwrap();
    let checkpoint = learner
        .save_rollout_ledger_continuation(&learner_policy, &continuation)
        .unwrap();
    assert!(checkpoint.is_relative(), "checkpoint={checkpoint:?}");

    let reopened_run = RunDir::open(&runs_root, "learner").unwrap();
    let reopened = Trainer::new(cfg, &reopened_run).unwrap();
    let mut restored_policy = StatefulScriptedPolicy::new(SEED.wrapping_add(1), 99).unwrap();
    let restored = reopened
        .restore_rollout_ledger_continuation(&checkpoint, &mut restored_policy, &policy_sha256)
        .unwrap();
    assert_eq!(restored.completed_step(), 1);
    assert_eq!(var_bits(&restored_policy), expected_adapter);
    assert_eq!(restored_policy.sampler_state().unwrap(), expected_sampler);
}

#[test]
#[allow(clippy::cognitive_complexity)] // fixture conversion plus full payload comparison
fn world_one_restores_legacy_v1_rollout_ledger_continuation() {
    let tmp = TempDir::new("legacy-v1-rollout-ledger-continuation");
    let ledger_root = tmp.path().join("ledger");
    let samples = live_samples();
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 1,
        group_size: 2,
        beta: 0.0,
        mu: 1,
        ..scripted_cfg()
    };
    let policy_sha256 = format!("{:064x}", 53);

    let collector_run = RunDir::create(tmp.path(), "collector").unwrap();
    let mut collector = Trainer::new(cfg.clone(), &collector_run).unwrap();
    let mut collector_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    collector
        .collect_rollout_ledger_step(
            0,
            &mut collector_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
            &ledger_root,
            &policy_sha256,
            None,
        )
        .unwrap();

    let learner_run = RunDir::create(tmp.path(), "learner").unwrap();
    let mut learner = Trainer::new(cfg.clone(), &learner_run).unwrap();
    let mut learner_policy = StatefulScriptedPolicy::new(SEED, 0).unwrap();
    let (_, continuation) = learner
        .train_rollout_ledger_step(0, &mut learner_policy, &ledger_root, &policy_sha256, None)
        .unwrap();
    let expected_adapter = var_bits(&learner_policy);
    let expected_sampler = learner_policy.sampler_state().unwrap();
    let expected_optimizer = optimizer_bits(continuation.optimizer_state());
    let current = learner
        .save_rollout_ledger_continuation(&learner_policy, &continuation)
        .unwrap();

    let legacy = tmp.path().join("legacy/step-1");
    std::fs::create_dir_all(&legacy).unwrap();
    for entry in std::fs::read_dir(current).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        std::fs::copy(entry.path(), legacy.join(entry.file_name())).unwrap();
    }
    let manifest_path = legacy.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let continuation_manifest = manifest["rollout_ledger_continuation"]
        .as_object_mut()
        .unwrap();
    assert_eq!(continuation_manifest["world_size"].as_u64(), Some(1));
    continuation_manifest.insert("format_version".into(), serde_json::json!(1));
    continuation_manifest.remove("world_size");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let restore_run = RunDir::create(tmp.path(), "legacy-restore").unwrap();
    let restore_trainer = Trainer::new(cfg, &restore_run).unwrap();
    let mut restored_policy = StatefulScriptedPolicy::new(SEED.wrapping_add(1), 99).unwrap();
    let restored = restore_trainer
        .restore_rollout_ledger_continuation(&legacy, &mut restored_policy, &policy_sha256)
        .unwrap();
    assert_eq!(restored.completed_step(), 1);
    assert_eq!(restored.world_size(), 1);
    assert_eq!(var_bits(&restored_policy), expected_adapter);
    assert_eq!(restored_policy.sampler_state().unwrap(), expected_sampler);
    assert_eq!(
        optimizer_bits(restored.optimizer_state()),
        expected_optimizer
    );
}

// ---- gate 4: degenerate shards ------------------------------------------------

/// Rank 0's shard is all-degenerate every window ('e' prompts score a constant
/// → zero advantages → no live items at beta 0) while rank 1 holds live items:
/// the live-count decision is global, so rank 0 must enter every collective
/// with zeros — a local skip would deadlock the world (this test would hang,
/// converted to a loud timeout by `LocalComm`) — and both ranks must keep
/// stepping in lockstep on rank 1's signal alone.
#[test]
fn an_all_degenerate_local_shard_neither_deadlocks_nor_diverges() {
    let tmp = TempDir::new("deg-local");
    let cfg = TrainerConfig {
        steps: 2,
        beta: 0.0,
        ..scripted_cfg()
    };
    // accum 1, world 2: window step consumes prompts [2*step, 2*step + 1] —
    // rank 0 always draws "e" (degenerate), rank 1 always "a" (live).
    let cfg = TrainerConfig {
        grad_accum_steps: 1,
        ..cfg
    };
    let samples = ["e", "a"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg, &samples);
    assert_lockstep(&ranks, "degenerate local shard");
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert!(
        max_abs_diff(&ranks[0].0, &init) > 0.0,
        "rank 1's live signal should still move the (lockstepped) weights"
    );
    // Rank 0's local stats see only its degenerate shard.
    assert!(
        ranks[0]
            .1
            .iter()
            .all(|m| (m.frac_reward_zero_std - 1.0).abs() < f32::EPSILON),
        "rank 0's shard is degenerate every window"
    );
    // The zeros oracle: a single-rank run at accum 2 consumes the identical
    // global window ('e' degenerate + 'a' live) with the identical loss scale
    // and live count, so the empty shard's contribution must be EXACTLY zeros
    // — folding anything else in (the PR-F mutation sweep's M6 planted the
    // weights themselves) diverges from this reference immediately, while
    // lockstep alone stays green (both ranks share the corrupted sum).
    let single = run_scripted_single(
        tmp.path(),
        &TrainerConfig {
            grad_accum_steps: 2,
            ..cfg
        },
        &samples,
    );
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    assert!(
        envelope < 1e-5,
        "the empty shard contributed something non-zero to the reduce: {envelope:.3e}"
    );
    assert_grad_norms_match(&ranks[0].1, &single.1, "degenerate local shard");
}

/// A var that some rank's backward never covers must abort EVERY rank, fast
/// and in lockstep: the locally-uncovered rank via the grad-coverage canary,
/// its peers via the globalized verdict — NOT a 300s timeout (a half-stepped
/// world or a stalled collective is exactly what the global uncovered count
/// exists to prevent; the PR-F mutation sweep's M8 showed no other gate
/// exercises a partially-covering rank).
#[test]
fn an_uncovered_var_on_one_rank_aborts_every_rank_in_lockstep() {
    let tmp = TempDir::new("uncovered-peer");
    let cfg = TrainerConfig {
        steps: 1,
        mu: 1,
        beta: 0.0,
        grad_accum_steps: 1,
        ..scripted_cfg()
    };
    let samples = ["a", "b"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let comms = LocalComm::world(2);
    let errs: Vec<(usize, String)> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    // Rank 1 wires the extra var into its loss; rank 0 does
                    // not — same var COUNT on both ranks (the collective
                    // contract holds), but rank 0's backward never covers it.
                    let mut policy = GatedExtraPolicy::new(SEED, rank == 1);
                    let run = RunDir::create(base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let err = trainer
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, &samples)
                        .unwrap_err();
                    (rank, err.to_string())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (rank, msg) in &errs {
        assert!(
            !msg.contains("timeout"),
            "rank {rank} must abort promptly, not stall into a timeout: {msg}"
        );
        if *rank == 0 {
            // The locally-uncovered rank reports the canary's own detail.
            assert!(
                !msg.contains("peer rank"),
                "rank 0 should fail its OWN canary, got: {msg}"
            );
        } else {
            assert!(
                msg.contains("peer rank"),
                "rank 1 must abort on the peer's verdict, got: {msg}"
            );
        }
    }
}

/// The world-1 path must issue ZERO collective calls — the byte-for-byte
/// legacy-path promise is the `world_size() > 1` guard discipline, and the
/// weight-bits trio gate alone cannot see a guard regression (an identity
/// reduce keeps the weights bitwise; the PR-F mutation sweep's M9).
#[test]
fn world_one_training_issues_no_collective_calls() {
    let tmp = TempDir::new("spy");
    let calls = Arc::new(AtomicUsize::new(0));
    let spy = SpyComm {
        calls: Arc::clone(&calls),
    };
    let mut policy = ScriptedPolicy::new(SEED).unwrap();
    let run = RunDir::create(tmp.path(), "spy").unwrap();
    let cfg = TrainerConfig {
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let mut trainer = Trainer::with_comm(cfg, &run, spy).unwrap();
    trainer
        .train(
            &mut policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &live_samples(),
        )
        .unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "the world-1 path entered a collective — a world > 1 guard regressed"
    );
}

/// Every rank's shard degenerate → the global live count is 0 → every rank
/// skips the window in lockstep (no update, no collective inside the skipped
/// epochs) and the run completes with the weights untouched.
#[test]
fn an_all_degenerate_global_window_is_skipped_by_every_rank() {
    let tmp = TempDir::new("deg-global");
    let cfg = TrainerConfig {
        steps: 2,
        beta: 0.0,
        grad_accum_steps: 1,
        ..scripted_cfg()
    };
    let samples = ["e", "e"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg, &samples);
    assert_lockstep(&ranks, "degenerate global window");
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert_eq!(
        max_abs_diff(&ranks[0].0, &init),
        0.0,
        "a globally degenerate run must not move any weight"
    );
    for (bits, history) in &ranks {
        assert_eq!(bits, &init, "weights must stay at init bitwise");
        assert!(
            history.iter().all(|m| m.grad_norm == 0.0),
            "no update may run"
        );
    }
}

// ---- gate 6: rank-0 checkpointing + DP resume ---------------------------------

/// Rank 0 writes the world's only checkpoint, and a world-2 resume from it
/// continues BIT-exactly (deterministic policies: weights and optimizer
/// moments are rank-identical by lockstep, and there is no sampler RNG) — the
/// interrupted+resumed world ends bitwise equal to an uninterrupted one.
#[test]
fn rank_zero_writes_the_only_checkpoint_and_a_dp_resume_continues_bit_exactly() {
    let base = TrainerConfig {
        checkpoint_every: Some(1),
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let tmp_full = TempDir::new("resume-full");
    let full = run_scripted_world(
        tmp_full.path(),
        2,
        &TrainerConfig {
            steps: 4,
            ..base.clone()
        },
        &live_samples(),
    );
    assert_lockstep(&full, "uninterrupted");

    // Phase 1: run the first 2 steps, checkpointing each.
    let tmp = TempDir::new("resume-phased");
    let phase1 = run_scripted_world(
        tmp.path(),
        2,
        &TrainerConfig {
            steps: 2,
            ..base.clone()
        },
        &live_samples(),
    );
    assert_lockstep(&phase1, "phase 1");
    let rank0_ckpt = tmp.path().join("rank0").join("checkpoints").join("step-2");
    assert!(rank0_ckpt.is_dir(), "rank 0 must write the checkpoint");
    // RunDir pre-creates the (empty) checkpoints dir; rank 1 must never fill it.
    let rank1_ckpts = tmp.path().join("rank1").join("checkpoints");
    assert_eq!(
        std::fs::read_dir(&rank1_ckpts).unwrap().count(),
        0,
        "rank 1 must write NO checkpoints (rank-0-only)"
    );

    // Phase 2: every rank resumes from rank 0's checkpoint, to step 4.
    let cfg = TrainerConfig { steps: 4, ..base };
    let comms = LocalComm::world(2);
    let resumed: Vec<RankRun> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let ckpt = rank0_ckpt.clone();
                let cfg = cfg.clone();
                let base = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(base, format!("resume-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let history = trainer
                        .resume(
                            &ckpt,
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_lockstep(&resumed, "resumed");
    assert_eq!(resumed[0].1.len(), 2, "resume runs steps 2..4");
    assert_eq!(
        full[0].0, resumed[0].0,
        "interrupted + resumed must be bitwise the uninterrupted run"
    );
}

// ---- resume_latest auto-discovers rank 0's checkpoint under data parallelism --

/// Run a fresh 2-rank DP world that resumes-or-starts via `resume_latest`, with
/// every rank pointing at ONE shared checkpoint dir (`with_checkpoints_dir`) — the
/// auto-discovery substrate — while keeping per-rank run dirs for telemetry. `tag`
/// distinguishes each call's per-rank run dirs (a duplicate `run_id` is rejected).
fn run_dp_world_resume_latest(
    base: &Path,
    shared_ckpts: &Path,
    tag: u64,
    cfg: &TrainerConfig,
) -> Vec<RankRun> {
    let comms = LocalComm::world(2);
    std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let shared = shared_ckpts.to_path_buf();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(base, format!("rl{tag}-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm)
                        .unwrap()
                        .with_checkpoints_dir(shared);
                    let history = trainer
                        .resume_latest(
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// The auto-discovery counterpart of
/// `rank_zero_writes_the_only_checkpoint_and_a_dp_resume_continues_bit_exactly`:
/// instead of the launcher handing every rank rank 0's checkpoint dir for an
/// explicit `resume`, every rank points at ONE shared checkpoint dir
/// (`with_checkpoints_dir`) and calls `resume_latest`. rank 0's scan is broadcast
/// to the world (broadcast-from-rank-0 via the sum all-reduce), so all ranks resume
/// from the SAME step in lockstep — bitwise the uninterrupted run, exactly as the
/// explicit path. This is the behavior that replaced the old hard DP refusal.
#[test]
fn resume_latest_under_dp_auto_discovers_rank0_checkpoint_in_lockstep() {
    let base = TrainerConfig {
        checkpoint_every: Some(1),
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    // Oracle: an uninterrupted 4-step DP world (per-rank dirs, default checkpoints).
    let tmp_full = TempDir::new("rl-dp-full");
    let full = run_scripted_world(
        tmp_full.path(),
        2,
        &TrainerConfig {
            steps: 4,
            ..base.clone()
        },
        &live_samples(),
    );
    assert_lockstep(&full, "uninterrupted");

    // ONE shared checkpoint dir every rank reads, rank 0 writes (not pre-created —
    // the save path makes it; `latest_checkpoint` reads a missing dir as "none").
    let tmp = TempDir::new("rl-dp-shared");
    let shared_ckpts = tmp.path().join("shared-checkpoints");

    // Phase 1: an empty shared dir → all ranks start fresh, run 2 steps, rank 0
    // checkpoints step-2 into the shared dir.
    let phase1 = run_dp_world_resume_latest(
        tmp.path(),
        &shared_ckpts,
        1,
        &TrainerConfig {
            steps: 2,
            ..base.clone()
        },
    );
    assert_lockstep(&phase1, "phase 1");
    assert_eq!(phase1[0].1.len(), 2, "phase 1 runs steps 0..2 fresh");
    assert!(
        shared_ckpts.join("step-2").is_dir(),
        "rank 0 must write the checkpoint into the shared dir"
    );

    // Phase 2: resume_latest to step 4 — auto-discovers step-2 from the shared dir
    // and broadcasts it, so every rank resumes there and runs 2..4 in lockstep.
    let resumed = run_dp_world_resume_latest(
        tmp.path(),
        &shared_ckpts,
        2,
        &TrainerConfig { steps: 4, ..base },
    );
    assert_lockstep(&resumed, "resumed");
    assert_eq!(resumed[0].1.len(), 2, "resume_latest runs steps 2..4");
    assert_eq!(
        full[0].0, resumed[0].0,
        "DP resume_latest must reproduce the uninterrupted run bit-for-bit"
    );
}

/// With no checkpoint in the shared dir, `resume_latest` under DP must start every
/// rank fresh in lockstep — the broadcast carries the "start fresh" sentinel, so
/// the whole world takes the same branch (never a rank-0-resumes / peers-restart
/// split). Equivalent to an uninterrupted `train()` DP world.
#[test]
fn resume_latest_under_dp_with_no_checkpoint_starts_fresh_in_lockstep() {
    let cfg = TrainerConfig {
        steps: 3,
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let tmp_full = TempDir::new("rl-dp-fresh-oracle");
    let full = run_scripted_world(tmp_full.path(), 2, &cfg, &live_samples());
    assert_lockstep(&full, "uninterrupted fresh");

    let tmp = TempDir::new("rl-dp-fresh");
    let shared_ckpts = tmp.path().join("shared-checkpoints"); // never created
    let fresh = run_dp_world_resume_latest(tmp.path(), &shared_ckpts, 1, &cfg);
    assert_lockstep(&fresh, "no-checkpoint fresh");
    assert_eq!(fresh[0].1.len(), 3, "no checkpoint → every step runs fresh");
    assert_eq!(
        full[0].0, fresh[0].0,
        "DP resume_latest with no checkpoint must equal an uninterrupted DP train()"
    );
}

/// The divergence-prevention claim, pinned directly: with **per-rank** checkpoint dirs
/// (the misuse the shared-dir contract warns against), the broadcast still carries rank
/// 0's decision to the peers, so a peer ATTEMPTS rank 0's resume and fails **loudly**
/// (its own dir lacks the checkpoint) rather than silently starting fresh and diverging.
///
/// This is the test the shared-dir equivalence cases above cannot be: with a shared dir,
/// a buggy no-broadcast implementation (each rank scans its own dir) would *also* pass,
/// because every rank sees rank 0's checkpoint. Here, only rank 0's dir has the
/// checkpoint — so a no-broadcast implementation would have rank 1 scan its empty dir,
/// start fresh, run every step while rank 0 (resuming the final step) runs none, and
/// **deadlock** on the first mismatched collective (surfacing as a `Comm` timeout). The
/// coordinated implementation instead makes rank 1 fail at the checkpoint load
/// (`TrainerError::Checkpoint`) and makes rank 0 abort on the peer's load failure
/// (`TrainerError::Contract`) before it can proceed alone. A `Comm` error would fail
/// this test.
#[test]
#[allow(clippy::cognitive_complexity)]
fn resume_latest_under_dp_without_a_shared_dir_aborts_load_in_lockstep_not_fresh() {
    let base = TrainerConfig {
        checkpoint_every: Some(1),
        grad_accum_steps: 2,
        steps: 2,
        ..scripted_cfg()
    };
    let tmp = TempDir::new("rl-dp-perrank");
    // Phase 1: a 2-rank world with PER-RANK dirs writes rank 0's checkpoint at step 2;
    // rank 1's own checkpoint dir stays empty (rank-0-only writes).
    let phase1 = run_scripted_world(tmp.path(), 2, &base, &live_samples());
    assert_lockstep(&phase1, "phase 1");
    assert!(
        tmp.path()
            .join("rank0")
            .join("checkpoints")
            .join("step-2")
            .is_dir(),
        "rank 0 must write the only checkpoint"
    );
    assert_eq!(
        std::fs::read_dir(tmp.path().join("rank1").join("checkpoints"))
            .unwrap()
            .count(),
        0,
        "rank 1's own checkpoint dir must be empty"
    );

    // Phase 2: resume_latest with PER-RANK dirs (each rank REOPENS its own phase-1 dir),
    // steps == the checkpoint step. A short timeout so the buggy-impl divergence surfaces
    // as a fast Comm timeout rather than hanging the test.
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(20));
    let outcomes: Vec<Result<(), TrainerError>> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = base.clone();
                let basep = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::open(basep, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    trainer
                        .resume_latest(
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .map(|_| ())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // rank 0 can load its local step-2 checkpoint, but rank 1 cannot load the
    // broadcast step from its empty per-rank checkpoint dir. The load failure is
    // coordinated, so rank 0 must abort in lockstep instead of proceeding alone.
    let err0 = outcomes[0].as_ref().unwrap_err();
    assert!(
        matches!(err0, TrainerError::Contract(msg)
            if msg.contains("checkpoint load/restore failed on a peer rank")),
        "rank 0 must abort when a peer cannot load the broadcast checkpoint: got {err0:?}"
    );
    // rank 1 received rank 0's broadcast step and tried to resume from its OWN (empty)
    // dir → a loud checkpoint-load error. NOT a silent fresh start, NOT a Comm timeout
    // from a diverged collective (which is exactly what a no-broadcast impl produces).
    let err1 = outcomes[1].as_ref().unwrap_err();
    assert!(
        matches!(err1, TrainerError::Checkpoint(_)),
        "rank 1 must fail loudly at the checkpoint load (broadcast carried rank 0's step \
         to a dir that lacks it), not diverge into a Comm timeout: got {err1:?}"
    );
}

/// A rank-0 checkpoint-scan FAILURE must ride the broadcast every rank enters, so the
/// world aborts **in lockstep and promptly** — not a rank-0-only early return that
/// strands the peers in the collective until the timeout (the fresh-eyes-review
/// deadlock vector). Induced by pointing the shared checkpoint dir at a FILE, so rank
/// 0's `read_dir` of it errors (`ENOTDIR`). With a short collective timeout, a
/// regression (rank-0 `?`-return before the broadcast) would surface as a peer `Comm`
/// timeout — which this test forbids; the fix makes rank 0 return the real
/// `Checkpoint` error and the peer the synthesized `Contract` error, both at once.
#[test]
fn resume_latest_under_dp_broadcasts_a_rank0_scan_failure_instead_of_hanging() {
    let tmp = TempDir::new("rl-dp-scanfail");
    let not_a_dir = tmp.path().join("checkpoints-is-a-file");
    std::fs::write(&not_a_dir, b"not a directory").unwrap();
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(15));
    let errs: Vec<TrainerError> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let basep = tmp.path();
                let ckpt = not_a_dir.clone();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(basep, format!("sf-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(scripted_cfg(), &run, comm)
                        .unwrap()
                        .with_checkpoints_dir(ckpt);
                    trainer
                        .resume_latest(
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .map(|_| ())
                        .unwrap_err()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // No rank may surface a Comm timeout — that would mean a peer was stranded in the
    // broadcast by a rank-0-only early return (the bug the broadcast-the-failure fix
    // closes). rank 0 surfaces the real checkpoint IO error; the peer the Contract one.
    for e in &errs {
        assert!(
            !matches!(e, TrainerError::Comm(_)),
            "a rank-0 scan failure must abort in lockstep, not strand a peer in the \
             collective (Comm timeout): got {e:?}"
        );
    }
    assert!(
        errs.iter()
            .any(|e| matches!(e, TrainerError::Checkpoint(_))),
        "rank 0 must surface the real checkpoint scan error: {errs:?}"
    );
    assert!(
        errs.iter().any(|e| matches!(e, TrainerError::Contract(_))),
        "the peer must surface the synthesized lockstep-abort error: {errs:?}"
    );
}

// ---- gate 2: real tiny qwen3.5, bitwise rank lockstep, both modes -------------

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen35")
}

/// A char-level codec over the fixture's 64-token vocabulary.
struct ByteCodec;
impl TokenizerLike for ByteCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| u32::from(b) % 64).collect()
    }
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| char::from(b'a' + u8::try_from(i % 26).unwrap()))
            .collect()
    }
}

/// A deterministic reward that SPREADS over completions, so group advantages
/// are non-degenerate.
struct SpreadReward;
impl RewardFn for SpreadReward {
    type Target = ();
    fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
        Ok(completion
            .bytes()
            .enumerate()
            .map(|(i, b)| f32::from(b) * (0.3 + i as f32 * 0.17))
            .sum::<f32>()
            % 5.0)
    }
}

fn lora_policy(seed: u64) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap();
    let policy = Qwen3_5Policy::new(model, seed, 1.0);
    // The LoRA A factors come from `Var::randn`, which on CPU draws from
    // OS entropy (unseedable) — every rank would otherwise start from a
    // DIFFERENT adapter and lockstep would be broken at step 0 (the full-FT
    // twin needs no fill: its vars are the loaded weights). Overwrite every
    // adapter var with a seeded ~0.02-scale hash fill, identical on every
    // rank.
    for (k, v) in policy.trainable_vars().iter().enumerate() {
        let dims = v.as_tensor().dims().to_vec();
        let n: usize = dims.iter().product();
        let fill: Vec<f32> = (0..n)
            .map(|i| {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((k as u64 + 1).wrapping_mul(seed.wrapping_mul(40_503)));
                ((z % 1000) as f32 / 1000.0 - 0.5) * 0.04
            })
            .collect();
        v.set(&Tensor::from_vec(fill, dims, &Device::Cpu).unwrap())
            .unwrap();
    }
    policy
}

#[cfg(feature = "nccl")]
fn lora_policy_on_device(seed: u64, dtype: DType, device: &Device) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, dtype, device).unwrap();
    let mut model =
        Qwen3_5GradModel::load_with_adapter_dtype(&cfg, &vb, 2, 4.0, DType::F32).unwrap();
    model.set_activation_checkpointing(true);
    let policy = Qwen3_5Policy::new(model, seed, 1.0);
    for (k, v) in policy.trainable_vars().iter().enumerate() {
        let dims = v.as_tensor().dims().to_vec();
        let n: usize = dims.iter().product();
        let fill: Vec<f32> = (0..n)
            .map(|i| {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((k as u64 + 1).wrapping_mul(seed.wrapping_mul(40_503)));
                ((z % 1000) as f32 / 1000.0 - 0.5) * 0.04
            })
            .collect();
        v.set(&Tensor::from_vec(fill, dims, device).unwrap())
            .unwrap();
    }
    policy
}

fn full_ft_policy(seed: u64) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let tensors = tensors_from_pretrained(&dir, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load_full_ft(&cfg, tensors, DType::F32, &Device::Cpu).unwrap();
    Qwen3_5Policy::new(model, seed, 1.0)
}

/// World-2 over the real tiny fixture: per-rank rollouts are STOCHASTIC and
/// shard-different, yet every rank steps from the identical reduced gradient,
/// so the weights must stay in bitwise lockstep — the invariant the whole DP
/// design rests on, checked end-to-end through the real model's grad path.
fn assert_real_model_lockstep(tag: &str, make_policy: fn(u64) -> Qwen3_5Policy) {
    let tmp = TempDir::new(tag);
    let cfg = TrainerConfig {
        steps: 2,
        group_size: 2,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.0,
        mu: 1,
        lr: 1e-3,
        loss_type: LossType::Grpo,
        ..TrainerConfig::default()
    };
    let samples = ["abc", "bcd"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let comms = LocalComm::world(2);
    let ranks: Vec<RankRun> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = make_policy(7);
                    let run = RunDir::create(base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let history = trainer
                        .train(&mut policy, &SpreadReward, &ByteCodec, &samples)
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_lockstep(&ranks, tag);
    let init = var_bits(&make_policy(7));
    assert!(
        max_abs_diff(&ranks[0].0, &init) > 0.0,
        "{tag}: no training signal — the lockstep gate would be vacuous"
    );
    assert!(
        ranks[0]
            .1
            .iter()
            .any(|m| m.grad_norm > 0.0 && m.grad_norm.is_finite()),
        "{tag}: no real update ran"
    );
}

#[test]
fn dp_ranks_stay_in_bitwise_lockstep_on_real_qwen35_lora() {
    assert_real_model_lockstep("lockstep-lora", lora_policy);
}

#[test]
fn dp_ranks_stay_in_bitwise_lockstep_on_real_qwen35_full_ft() {
    assert_real_model_lockstep("lockstep-full-ft", full_ft_policy);
}

/// Manual resource-regression smoke for the trainer's CUDA/NCCL update path.
///
/// Launch one process per rank under Slurm, with a shared `FERRL_NCCL_RENDEZVOUS`.
/// The test uses the committed tiny qwen3.5 fixture so it is cheap, deterministic,
/// and asset-free, while still reaching rollout -> backward -> gradient all-reduce
/// -> optimizer over real CUDA tensors. The printed `NCCL_TINY_QWEN35_SMOKE` rows
/// are intentionally stable for external branch-vs-main parsers.
#[cfg(feature = "nccl")]
#[test]
#[ignore = "manual CUDA/NCCL resource gate; launch one process per rank under Slurm"]
#[allow(clippy::print_stderr)] // manual gate: the printed memory/timing rows are the deliverable
fn nccl_tiny_qwen35_lora_smoke_reaches_update_path() {
    let comm = ferrl::NcclComm::from_slurm_env().expect("bootstrap NCCL from Slurm env");
    let rank = comm.rank();
    let world = comm.world_size();
    let device = comm.device().clone();
    if let Some(warning) = ferrl::check_driver_compat(&device).warning() {
        eprintln!("{warning}");
    }
    ferrl::guard_first_kernel(&device).expect("CUDA first-kernel guard");

    let mut policy = lora_policy_on_device(7, DType::BF16, &device);
    let cfg = TrainerConfig {
        steps: 2,
        group_size: 2,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.0,
        mu: 1,
        lr: 1e-3,
        loss_type: LossType::Grpo,
        gpu_memory_probe: true,
        ..TrainerConfig::default()
    };
    let root = std::env::var_os("FERRL_NCCL_SMOKE_RUN_ROOT").map_or_else(
        || std::env::temp_dir().join(format!("ferrl-nccl-smoke-{}", std::process::id())),
        PathBuf::from,
    );
    std::fs::create_dir_all(&root).unwrap();
    let run = RunDir::create(&root, format!("rank{rank}")).unwrap();
    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
    let samples = ["abc", "bcd"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();

    let history = trainer
        .train(&mut policy, &SpreadReward, &ByteCodec, &samples)
        .expect("NCCL tiny qwen3.5 GRPO smoke failed")
        .0;

    assert_eq!(history.len(), 2);
    assert!(
        history
            .iter()
            .any(|m| m.grad_norm > 0.0 && m.grad_norm.is_finite()),
        "NCCL smoke reached no real optimizer update"
    );
    for m in history {
        assert!(
            m.cuda_mem_peak_used_bytes >= m.cuda_mem_start_used_bytes,
            "memory probe did not record a valid peak at step {}",
            m.step
        );
        eprintln!(
            "NCCL_TINY_QWEN35_SMOKE rank={rank} world={world} step={} grad_norm={} \
             step_secs={} tokens_per_sec={} cuda_start={} cuda_peak={} cuda_end={} \
             cuda_delta={}",
            m.step,
            m.grad_norm,
            m.step_secs,
            m.tokens_per_sec,
            m.cuda_mem_start_used_bytes,
            m.cuda_mem_peak_used_bytes,
            m.cuda_mem_end_used_bytes,
            m.cuda_mem_peak_delta_bytes
        );
    }
}
