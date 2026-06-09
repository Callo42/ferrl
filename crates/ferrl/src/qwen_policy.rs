//! A [`Policy`] over the real Qwen3 model.
//!
//! [`QwenPolicy`] bridges the grad-bearing [`QwenGradModel`] forward (the update
//! path) to the trainer's [`Policy`] seam, so the *same* [`Trainer`] that drives
//! the P2 echo toy drives Qwen3-0.6B-Base unchanged. It is the production
//! counterpart of the in-test `EchoPolicy`.
//!
//! ## Generation is uncached and adapter-aware
//!
//! Sampling re-runs the full-sequence [`QwenGradModel::forward`] each step (no KV
//! cache), exactly like the toy. This is deliberate: the rollout must be drawn
//! from the *current* policy (`LoRA` adapter **on**), and candle's shipped
//! `ModelForCausalLM` — the only KV-cached forward available — carries no adapter,
//! so generating from it would sample the frozen base model and the policy's
//! rollouts would never reflect training. A fast adapter-aware rollout (e.g.
//! merging `W + scale·BA` into a cached forward) is a throughput optimization for
//! later; correctness comes first.
//!
//! ## Rectangular rollouts
//!
//! [`generate`](QwenPolicy::generate) emits exactly `max_new_tokens` per
//! completion with no early stop, so every sequence in a group has the same
//! length — the rectangular shape the [`Trainer`] requires (it rejects ragged
//! rollouts; variable-length completions with an attention mask arrive with the
//! later P4 work). Scoring
//! ([`token_logprobs`](QwenPolicy::token_logprobs)) is teacher-forced: forward all
//! but the last token, read the positions that predict the completion tokens, and
//! gather their log-probabilities.
//!
//! [`Trainer`]: crate::trainer::Trainer

use candle_core::{DType, IndexOp, Result as CandleResult, Tensor, Var, D};
use candle_nn::ops::log_softmax;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::policy::{GenConfig, Policy, Rollout};
use crate::qwen::QwenGradModel;

/// A [`Policy`] backed by the grad-bearing [`QwenGradModel`].
///
/// Construct it from a loaded model with [`QwenPolicy::new`]; the device and dtype
/// follow the model's — all-F32, or the bf16-base / F32-adapter split (see
/// [`QwenGradModel::load_with_adapter_dtype`](crate::qwen::QwenGradModel::load_with_adapter_dtype)),
/// whose BF16 logits the scoring path upcasts to F32 for the surrogate.
pub struct QwenPolicy {
    model: QwenGradModel,
    sampler: LogitsProcessor,
    temperature: f64,
    enabled: bool,
}

// `LogitsProcessor` is not `Debug`; format the inspectable fields and elide it.
impl std::fmt::Debug for QwenPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenPolicy")
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

impl QwenPolicy {
    /// Wrap a loaded [`QwenGradModel`] as a policy, seeding the rollout sampler.
    ///
    /// `temperature` is the rollout sampling temperature, fixed for this policy's
    /// lifetime: candle's [`LogitsProcessor`] owns the sampling RNG and exposes no
    /// per-call temperature, so it is baked in here. [`generate`](Self::generate)
    /// **fails loud** if handed a [`GenConfig`] whose `temperature` differs (rather
    /// than silently sampling at the wrong temperature); the trainer passes this
    /// same value through. Scoring is always at temperature 1. The adapter starts
    /// enabled (the trainer toggles it off for the KL reference forward).
    #[must_use]
    pub fn new(model: QwenGradModel, seed: u64, temperature: f64) -> Self {
        let sampler = LogitsProcessor::from_sampling(seed, Sampling::All { temperature });
        Self {
            model,
            sampler,
            temperature,
            enabled: true,
        }
    }

    /// The wrapped grad-bearing model — e.g. to inspect its device or (later) save
    /// the trained adapter.
    #[must_use]
    pub fn model(&self) -> &QwenGradModel {
        &self.model
    }
}

impl Policy for QwenPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        // The sampler's temperature is fixed at construction (see `new`); fail loud
        // rather than silently sampling at a different cfg.temperature.
        if (cfg.temperature - self.temperature).abs() > f64::EPSILON {
            candle_core::bail!(
                "QwenPolicy was built with temperature {} but generate was called \
                 with cfg.temperature {}; rebuild the policy to change it",
                self.temperature,
                cfg.temperature
            );
        }
        let device = self.model.device().clone();
        let mut token_ids = Vec::with_capacity(cfg.group_size);
        for _ in 0..cfg.group_size {
            let mut ids = prompt.to_vec();
            for _ in 0..cfg.max_new_tokens {
                let len = ids.len();
                let input = Tensor::from_vec(ids.clone(), (1, len), &device)?;
                // Uncached forward at the current adapter state; sample the last pos.
                let logits = self.model.forward(&input)?;
                let last = logits.i((0, len - 1))?;
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
        // Precondition (the Trainer guarantees this via `completion_dims`): a
        // rectangular rollout with `prompt_len >= 1` and `comp_len >= 1`. Called
        // directly with `prompt_len == 0`, the `prompt_len - 1` narrow underflows.
        let prompt_len = rollout.prompt_len;
        let seq_len = rollout.token_ids[0].len();
        let comp_len = seq_len - prompt_len;
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let device = self.model.device();

        // Teacher forcing: forward all but the last token of every sequence.
        let mut input_data = Vec::with_capacity(g * input_len);
        for ids in &rollout.token_ids {
            input_data.extend_from_slice(&ids[..input_len]);
        }
        let input = Tensor::from_vec(input_data, (g, input_len), device)?;
        let logits = self.model.forward(&input)?; // [g, input_len, vocab]

        // The positions that predict the completion tokens are
        // [prompt_len - 1 .. prompt_len - 1 + comp_len].
        // Upcast just the completion-position logits (a small `[g, comp_len, vocab]`
        // slice, NOT the full sequence) to F32 before the log-softmax, so the
        // surrogate's log-probs keep F32 precision even when the model runs in BF16
        // (the dtype split); the big full-sequence logits stay BF16. A no-op when the
        // model is already F32.
        let pred = logits
            .narrow(1, prompt_len - 1, comp_len)?
            .to_dtype(DType::F32)?;
        let logp = log_softmax(&pred, D::Minus1)?;

        let mut tgt_data = Vec::with_capacity(g * comp_len);
        for ids in &rollout.token_ids {
            tgt_data.extend_from_slice(&ids[prompt_len..seq_len]);
        }
        let targets = Tensor::from_vec(tgt_data, (g, comp_len), device)?;
        let idx = targets.unsqueeze(D::Minus1)?;
        logp.gather(&idx, D::Minus1)?.squeeze(D::Minus1)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.model.set_adapter_enabled(enabled);
        self.enabled = enabled;
    }

    fn adapter_enabled(&self) -> bool {
        self.enabled
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.model.trainable_vars()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::grad_coverage;
    use candle_core::backprop::GradStore;
    use candle_core::{DType, Device};
    use candle_nn::{Activation, VarBuilder};
    use candle_transformers::models::qwen3::Config;
    use std::collections::HashMap;

    /// A tiny Qwen3 config (2 layers, 2 Q / 1 KV head, `head_dim` 4) — the same
    /// scaffold qwen.rs's tests use, at a runnable scale on CPU.
    fn tiny_cfg() -> Config {
        Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            head_dim: 4,
            attention_bias: false,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: None,
            max_window_layers: 0,
            tie_word_embeddings: true,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
        }
    }

    /// Random weights matching `cfg`'s dotted tensor names (tied head → no
    /// `lm_head.weight`).
    fn weight_map(cfg: &Config) -> HashMap<String, Tensor> {
        let d = Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, dims: &[usize]| {
            t.insert(
                name.to_string(),
                Tensor::randn(0f32, 0.2f32, dims.to_vec(), &d).unwrap(),
            );
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let qo = cfg.num_attention_heads * cfg.head_dim;
        let kvo = cfg.num_key_value_heads * cfg.head_dim;
        put("model.embed_tokens.weight", &[cfg.vocab_size, h]);
        put("model.norm.weight", &[h]);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            put(&format!("{p}.input_layernorm.weight"), &[h]);
            put(&format!("{p}.post_attention_layernorm.weight"), &[h]);
            put(&format!("{p}.self_attn.q_proj.weight"), &[qo, h]);
            put(&format!("{p}.self_attn.k_proj.weight"), &[kvo, h]);
            put(&format!("{p}.self_attn.v_proj.weight"), &[kvo, h]);
            put(&format!("{p}.self_attn.o_proj.weight"), &[h, qo]);
            put(&format!("{p}.self_attn.q_norm.weight"), &[cfg.head_dim]);
            put(&format!("{p}.self_attn.k_norm.weight"), &[cfg.head_dim]);
            put(&format!("{p}.mlp.gate_proj.weight"), &[i, h]);
            put(&format!("{p}.mlp.up_proj.weight"), &[i, h]);
            put(&format!("{p}.mlp.down_proj.weight"), &[h, i]);
        }
        t
    }

    fn tiny_policy() -> QwenPolicy {
        let cfg = tiny_cfg();
        let vb = VarBuilder::from_tensors(weight_map(&cfg), DType::F32, &Device::Cpu);
        let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        QwenPolicy::new(model, 7, 1.0)
    }

    #[test]
    fn generate_returns_rectangular_group() {
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 3,
            temperature: 1.0,
        };
        let rollout = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(rollout.len(), 4);
        assert_eq!(rollout.prompt_len, 3);
        // Every sequence has the same length (rectangular): prompt + new tokens.
        for ids in &rollout.token_ids {
            assert_eq!(ids.len(), 3 + 3);
            assert!(ids.iter().all(|&i| i < tiny_cfg().vocab_size as u32));
        }
    }

    #[test]
    fn token_logprobs_shape_and_finiteness() {
        let policy = tiny_policy();
        let rollout = Rollout {
            // Two sequences, prompt_len 2, completion_len 3 (rectangular).
            token_ids: vec![vec![1u32, 2, 3, 4, 5], vec![1, 2, 6, 7, 8]],
            prompt_len: 2,
        };
        let logp = policy.token_logprobs(&rollout).unwrap();
        assert_eq!(logp.dims(), &[2, 3]);
        // Log-probs are <= 0 and finite.
        let flat = logp.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|&x| x.is_finite() && x <= 1e-5));
    }

    #[test]
    fn token_logprobs_align_with_a_manual_per_position_reference() {
        // Shape + finiteness can't catch a teacher-forcing off-by-one (a wrong but
        // finite, correctly-shaped score makes GRPO optimize garbage). Pin the
        // alignment: each returned log-prob must equal the model's own
        // log_softmax(logits)[g, prompt_len-1+j, completion_token] recomputed
        // independently of the narrow/gather under test.
        let policy = tiny_policy();
        let rollout = Rollout {
            token_ids: vec![vec![1u32, 2, 3, 4, 5], vec![3, 1, 4, 1, 5]],
            prompt_len: 2,
        };
        let got = policy
            .token_logprobs(&rollout)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        let seq_len = rollout.token_ids[0].len();
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let mut data = Vec::new();
        for ids in &rollout.token_ids {
            data.extend_from_slice(&ids[..input_len]);
        }
        let input = Tensor::from_vec(data, (g, input_len), &Device::Cpu).unwrap();
        let logp_full = log_softmax(&policy.model().forward(&input).unwrap(), D::Minus1)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let comp_len = seq_len - rollout.prompt_len;
        for (gi, ids) in rollout.token_ids.iter().enumerate() {
            for j in 0..comp_len {
                let pos = rollout.prompt_len - 1 + j;
                let tgt = ids[rollout.prompt_len + j] as usize;
                let want = logp_full[gi][pos][tgt];
                assert!(
                    (got[gi][j] - want).abs() <= 1e-5,
                    "logp[{gi}][{j}]={} != manual {want} (pos {pos}, tgt {tgt})",
                    got[gi][j]
                );
            }
        }
    }

    /// One `token_logprobs -> sqr -> sum -> backward`, returning the grad store —
    /// the scoring path the trainer actually differentiates.
    fn grads_of(policy: &QwenPolicy, rollout: &Rollout) -> GradStore {
        let loss = policy
            .token_logprobs(rollout)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        loss.backward().unwrap()
    }

    /// Split the trainable vars into the (q, v) branches. Per-layer order is
    /// `q_A, q_B, v_A, v_B`, so `i % 4 < 2` is the q branch.
    fn branch_split(vars: &[Var]) -> (Vec<Var>, Vec<Var>) {
        let pick = |want_q: bool| -> Vec<Var> {
            vars.iter()
                .enumerate()
                .filter(|(i, _)| (i % 4 < 2) == want_q)
                .map(|(_, v)| v.clone())
                .collect()
        };
        (pick(true), pick(false))
    }

    /// Set every `B` factor (the odd index within each `[A, B]` pair) to small
    /// noise, so the update is no longer a no-op and `dL/dA` is no longer 0.
    fn force_b_nonzero(vars: &[Var]) {
        for (i, v) in vars.iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                v.set(&Tensor::randn(0f32, 0.02f32, dims, &Device::Cpu).unwrap())
                    .unwrap();
            }
        }
    }

    #[test]
    fn lora_grads_flow_through_token_logprobs_both_branches() {
        // Deterministic proof (no sampling) that gradients reach BOTH LoRA factors
        // (A and B) of q AND v THROUGH `token_logprobs` — the narrow/log_softmax/
        // gather must not detach A. At zero-B init dL/dA is structurally 0, so a
        // severed A-path is invisible to a single backward (the P3 PR-B trap); the
        // two-phase check (force B nonzero) closes it.
        let policy = tiny_policy();
        let rollout = Rollout {
            token_ids: vec![vec![1u32, 2, 3, 4, 5], vec![5, 4, 3, 2, 1]],
            prompt_len: 2,
        };
        let vars = policy.trainable_vars();
        assert_eq!(vars.len(), 2 * 4); // per layer: q_A, q_B, v_A, v_B
        let (q_vars, v_vars) = branch_split(&vars);

        // Phase 1 — zero-B: every var present + each branch live (via dL/dB) + finite.
        let g1 = grads_of(&policy, &rollout);
        assert!(
            grad_coverage(&q_vars, &g1).unwrap().is_ok(),
            "q-branch unhealthy at zero-B init"
        );
        assert!(
            grad_coverage(&v_vars, &g1).unwrap().is_ok(),
            "v-branch unhealthy at zero-B init"
        );

        // Phase 2 — force every B nonzero: now EVERY A and B must carry a nonzero
        // finite grad (proves the A-input path is wired, not just B).
        force_b_nonzero(&vars);
        let g2 = grads_of(&policy, &rollout);
        let qc = grad_coverage(&q_vars, &g2).unwrap();
        let vc = grad_coverage(&v_vars, &g2).unwrap();
        assert!(
            qc.nonzero == qc.total && qc.nonfinite == 0,
            "q-branch: not every LoRA var is live after nonzero-B (severed A?): {qc:?}"
        );
        assert!(
            vc.nonzero == vc.total && vc.nonfinite == 0,
            "v-branch: not every LoRA var is live after nonzero-B: {vc:?}"
        );
    }

    #[test]
    fn adapter_toggle_tracks_state_and_is_noop_at_zero_b() {
        let mut policy = tiny_policy();
        assert!(policy.adapter_enabled());
        let rollout = Rollout {
            token_ids: vec![vec![1u32, 2, 3, 4]],
            prompt_len: 2,
        };
        let on = policy.token_logprobs(&rollout).unwrap();
        policy.set_adapter_enabled(false);
        assert!(!policy.adapter_enabled());
        let off = policy.token_logprobs(&rollout).unwrap();
        // Zero-B init: the adapter is a no-op, so enabled == disabled log-probs.
        let diff: f32 = on
            .sub(&off)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff <= 1e-6, "zero-B adapter changed log-probs: {diff}");
        policy.set_adapter_enabled(true);
        assert!(policy.adapter_enabled());
    }

    #[test]
    fn trainable_vars_are_the_models() {
        let policy = tiny_policy();
        // 2 layers x (q_A, q_B, v_A, v_B) = 8 trainable vars.
        assert_eq!(policy.trainable_vars().len(), 2 * 4);
        // The manual Debug impl elides the non-Debug sampler.
        let dbg = format!("{policy:?}");
        assert!(dbg.contains("QwenPolicy") && dbg.contains(".."));
    }

    // ---- end-to-end: QwenPolicy through the real Trainer (CPU) --------------

    use crate::reward::RewardFn;
    use crate::telemetry::RunDir;
    use crate::trainer::{TokenizerLike, Trainer, TrainerConfig};

    /// Trivial char codec over the tiny vocab (id `i` <-> `'a' + i`); the tiny
    /// model's vocab is 16, so generated ids land in `'a'..'p'`.
    struct CharCodec;
    impl TokenizerLike for CharCodec {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.chars()
                .map(|c| (u32::from(c) - u32::from('a')) % 16)
                .collect()
        }
        fn decode(&self, ids: &[u32]) -> String {
            ids.iter()
                .filter_map(|&i| char::from_u32(u32::from('a') + (i % 16)))
                .collect()
        }
    }

    /// A reward that spreads across distinct completions (so a sampled group is
    /// non-degenerate and a real GRPO update fires). Position-WEIGHTED so that two
    /// completions sharing a byte multiset (`"ab"` vs `"ba"`) do not collide to the
    /// same reward and silently degenerate the group.
    struct SpreadReward;
    impl RewardFn for SpreadReward {
        fn reward(&self, _prompt: &str, completion: &str) -> f32 {
            completion
                .bytes()
                .enumerate()
                .map(|(i, b)| (i as f32 + 1.0) * f32::from(b))
                .sum::<f32>()
                / 1000.0
        }
    }

    /// A unique temp directory, removed on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new() -> Self {
            let mut p = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            p.push(format!(
                "ferrl-qwen-policy-{}-{}",
                std::process::id(),
                nanos
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Per-step metric sanity for the CPU GRPO run.
    fn assert_step_metrics_ok(m: &crate::telemetry::Metrics) {
        assert!(
            m.grad_norm.is_finite(),
            "non-finite grad_norm at step {}",
            m.step
        );
        assert!(m.reward_mean.is_finite());
        assert!(m.kl.is_finite() && m.kl >= 0.0, "bad KL at step {}", m.step);
    }

    #[test]
    fn drives_a_grpo_step_through_the_trainer_on_cpu() {
        // The same Trainer that drives the echo toy drives a (tiny) Qwen policy:
        // rollout -> reward -> advantages -> backward THROUGH the Qwen forward ->
        // grad-coverage canary -> AdamW. A clean multi-step run proves the canary
        // held on every real update (it aborts on a missing/non-finite grad).
        let mut policy = tiny_policy();
        let prompts = vec!["abc".to_string(), "bcd".to_string()];
        // beta > 0 so the adapter-disabled KL reference forward (and its restore)
        // actually runs through the Qwen path, not just the policy forward.
        let cfg = TrainerConfig {
            steps: 4,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            beta: 0.02,
            lr: 1e-3,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run = RunDir::create(&tmp.0, "qwen-cpu").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let history = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &prompts)
            .unwrap();

        assert_eq!(history.len(), 4);
        for m in &history {
            assert_step_metrics_ok(m);
        }
        // `grad_norm > 0` is set ONLY when an AdamW step actually runs (a real,
        // non-degenerate, non-fully-clipped update). Asserting it witnesses that the
        // Qwen backward produced a usable gradient and the optimizer stepped — far
        // stronger than `frac_reward_zero_std < 1` (which is computed from scalar
        // rewards, upstream of any backward). Deterministic A-path liveness is pinned
        // separately by `lora_grads_flow_through_token_logprobs_both_branches`.
        assert!(
            history.iter().any(|m| m.grad_norm > 0.0),
            "no AdamW step ran — the Qwen backward path was never exercised"
        );
        // The adapter is restored enabled after the (reference-toggling) run.
        assert!(policy.adapter_enabled());
    }
}
