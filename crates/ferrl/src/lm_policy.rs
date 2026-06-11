//! A [`Policy`] over any [`GradModel`].
//!
//! [`LmPolicy`] bridges a grad-bearing model forward (the update path) to the
//! trainer's [`Policy`] seam, generically over the [`GradModel`] /
//! [`CachedDecoder`] traits — so the *same*
//! [`Trainer`] that drives the P2 echo toy drives any real model unchanged.
//! [`QwenPolicy`] (= `LmPolicy<QwenGradModel>`) is the production instantiation
//! over Qwen3-0.6B-Base.
//!
//! ## Generation is KV-cached over merged weights, and adapter-aware
//!
//! [`generate`](LmPolicy::generate) snapshots the policy's **current** effective
//! weights into a cached decoder ([`GradModel::merged_decoder`]) — the `LoRA`
//! adapter folded into the base (`W + scale·BA` when enabled, plain `W` when
//! disabled, so the eval adapter-off path samples the frozen base) — and decodes
//! incrementally over a KV cache: **O(L) per token** instead of the uncached
//! forward's O(L²). The rollout is still drawn from the *current* policy at every
//! step (candle's shipped cached forwards carry no adapter, so they could only
//! sample the frozen base — the merge is what makes a cached **adapter-aware**
//! rollout possible). The merged decoder is a tape-detached value snapshot,
//! rebuilt every `generate` call, so it always reflects the latest optimizer
//! step. **Scoring is unaffected** — the grad-bearing
//! [`token_logprobs`](LmPolicy::token_logprobs) and the KL reference forward
//! still run the uncached [`GradModel::forward`] (the cache holds no tape).
//! The cached and uncached rollouts are equivalent up to F32 reassociation of the
//! merge (CI-gated: identical token stream **and** identical sampler-RNG
//! consumption on a tiny model); the bf16-merge faithfulness is a manual GPU gate.
//!
//! ## Rectangular rollouts
//!
//! [`generate`](LmPolicy::generate) always emits a **fixed** width of
//! `max_new_tokens` completion tokens per sequence, so every sequence in a group
//! has the same length — the rectangular shape the [`Trainer`] requires (it rejects
//! ragged rollouts, and a fixed width keeps Dr.GRPO's token denominator constant).
//! When [`GenConfig::eos_token_id`](crate::policy::GenConfig::eos_token_id) is set,
//! a sequence that samples the EOS token stops early (the EOS token is **kept** — the
//! length is EOS-*inclusive*) and is right-padded back to the fixed width with that
//! same EOS id; [`Rollout::completion_lens`](crate::policy::Rollout::completion_lens)
//! records each true length so the padding can be masked out of the loss downstream.
//! With `eos_token_id == None` no sequence stops early, every completion is the full
//! width, and the rollout is bit-identical to the legacy behavior. Scoring
//! ([`token_logprobs`](LmPolicy::token_logprobs)) is teacher-forced: forward all
//! but the last token, read the positions that predict the completion tokens, and
//! gather their log-probabilities — divided by the policy's rollout temperature
//! first (temperature-consistent scoring, TRL parity; a guarded no-op at the
//! `1.0` default).
//!
//! ## Behavior log-probs and the off-policy gap
//!
//! [`generate`](LmPolicy::generate) also records each drawn token's log-prob
//! under the distribution it was sampled from
//! ([`Rollout::rollout_logprobs`](crate::policy::Rollout::rollout_logprobs) —
//! the sampler computes the full distribution anyway, so the capture is free).
//! Rollout draws from the **merged cached** decoder while training scores with
//! the **uncached grad** forward; on an all-F32 model the two differ only by
//! float reassociation of the merge, but a bf16 base makes the rollout
//! genuinely off-policy relative to the f32-scored objective — exactly the
//! mismatch the trainer's rollout-ratio telemetry (and optional TIS
//! correction) measures from these captured log-probs.
//!
//! [`Trainer`]: crate::trainer::Trainer

use candle_core::{DType, IndexOp, Result as CandleResult, Tensor, Var, D};
use candle_nn::ops::log_softmax;

use crate::model::{CachedDecoder, GradModel};
use crate::policy::{GenConfig, Policy, Rollout};
use crate::qwen::QwenGradModel;
use crate::sampler::GrpoSampler;

/// A [`Policy`] backed by any grad-bearing [`GradModel`].
///
/// Construct it from a loaded model with [`LmPolicy::new`]; the device and dtype
/// follow the model's — all-F32, or a bf16-base / F32-adapter split (see
/// [`QwenGradModel::load_with_adapter_dtype`](crate::qwen::QwenGradModel::load_with_adapter_dtype)),
/// whose BF16 logits the scoring path upcasts to F32 for the surrogate.
pub struct LmPolicy<M: GradModel> {
    model: M,
    sampler: GrpoSampler,
    temperature: f64,
    enabled: bool,
}

/// The production policy over the real Qwen3 model — the first [`LmPolicy`]
/// instantiation (and the name every pre-M1 call site uses).
pub type QwenPolicy = LmPolicy<QwenGradModel>;

/// The policy over a dense Llama-3.x model — the second [`LmPolicy`]
/// instantiation, and the witness that the [`GradModel`] seam is real: the same
/// generic policy (and through it the same `Trainer`) drives
/// [`LlamaGradModel`](crate::llama::LlamaGradModel) with zero policy changes.
pub type LlamaPolicy = LmPolicy<crate::llama::LlamaGradModel>;

/// The policy over the hybrid `qwen3_5` (Qwen3.5 / Qwen3.6) model — the third
/// [`LmPolicy`] instantiation, and the first whose decoder state is not purely
/// KV-shaped (conv + delta-rule recurrent state on the linear-attention
/// layers); the generic policy drives it through the same
/// [`CachedDecoder`] contract with zero changes.
pub type Qwen3_5Policy = LmPolicy<crate::qwen35::Qwen3_5GradModel>;

// Elide the sampler's RNG state and the heavy model fields; show the inspectable
// scalars. (`GrpoSampler` is `Debug`, but the raw RNG words add only noise.)
impl<M: GradModel + std::fmt::Debug> std::fmt::Debug for LmPolicy<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmPolicy")
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

impl<M: GradModel> LmPolicy<M> {
    /// Wrap a loaded [`GradModel`] as a policy, seeding the rollout sampler.
    ///
    /// `temperature` is the rollout sampling temperature, fixed for this policy's
    /// lifetime: the [`GrpoSampler`] bakes it in (as candle's `LogitsProcessor`
    /// does). [`generate`](Self::generate) **fails loud** if handed a
    /// [`GenConfig`] whose `temperature` differs (rather than silently sampling
    /// at the wrong temperature); the trainer passes this same value through.
    /// The one exception is an explicit eval-only override
    /// ([`GenConfig::eval_sampling`](crate::policy::GenConfig::eval_sampling)),
    /// which deliberately samples the eval distribution instead. **Scoring is
    /// temperature-consistent** ([`token_logprobs`](Self::token_logprobs) divides
    /// the logits by this same temperature — TRL parity — so the distribution
    /// GRPO optimizes is the one the rollout sampled from; at the `1.0` default
    /// this is bit-identical to unscaled scoring). The adapter starts enabled
    /// (the trainer toggles it off for the KL reference forward).
    #[must_use]
    pub fn new(model: M, seed: u64, temperature: f64) -> Self {
        let sampler = GrpoSampler::new(seed, temperature);
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
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Mutable access to the wrapped model — e.g. to turn on **activation
    /// checkpointing** after construction
    /// (`policy.model_mut().set_activation_checkpointing(true)` on the models
    /// that support it; see
    /// [`QwenGradModel::set_activation_checkpointing`](crate::qwen::QwenGradModel::set_activation_checkpointing)).
    #[must_use]
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// The teacher-forcing scoring input: all but the last token of every
    /// sequence, as one `[group, seq_len - 1]` tensor on the model's device.
    ///
    /// Precondition (the `Trainer` guarantees this via `completion_dims`): a
    /// rectangular rollout with `prompt_len >= 1` and `comp_len >= 1`. Called
    /// directly with `prompt_len == 0`, the scoring narrow underflows.
    fn scoring_input(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let seq_len = rollout.token_ids[0].len();
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let mut input_data = Vec::with_capacity(g * input_len);
        for ids in &rollout.token_ids {
            input_data.extend_from_slice(&ids[..input_len]);
        }
        Tensor::from_vec(input_data, (g, input_len), self.model.device())
    }

    /// Gather the completion tokens' log-probabilities out of full-sequence
    /// `logits` — narrow to the completion-predicting positions, upcast to
    /// F32, divide by the rollout temperature (temperature-consistent
    /// scoring; a guarded no-op at the `1.0` default), `log_softmax`, gather.
    fn completion_logprobs(&self, rollout: &Rollout, logits: &Tensor) -> CandleResult<Tensor> {
        let prompt_len = rollout.prompt_len;
        let seq_len = rollout.token_ids[0].len();
        let comp_len = seq_len - prompt_len;
        let g = rollout.token_ids.len();

        // The positions that predict the completion tokens are
        // [prompt_len - 1 .. prompt_len - 1 + comp_len].
        // Upcast just the completion-position logits (a small `[g, comp_len, vocab]`
        // slice, NOT the full sequence) to F32 before the log-softmax, so the
        // surrogate's log-probs keep F32 precision even when the model runs in BF16
        // (the dtype split); the big full-sequence logits stay BF16. A no-op when the
        // model is already F32.
        let mut pred = logits
            .narrow(1, prompt_len - 1, comp_len)?
            .to_dtype(DType::F32)?;
        // Temperature-consistent scoring (TRL parity): divide the logits by the
        // policy's rollout temperature before the log-softmax, so the distribution
        // being optimized IS the one the rollout sampled from. Guarded so the
        // T = 1.0 default adds no op and stays bit-identical to the pre-R2 path.
        if (self.temperature - 1.0).abs() > f64::EPSILON {
            pred = (pred / self.temperature)?;
        }
        let logp = log_softmax(&pred, D::Minus1)?;

        let mut tgt_data = Vec::with_capacity(g * comp_len);
        for ids in &rollout.token_ids {
            tgt_data.extend_from_slice(&ids[prompt_len..seq_len]);
        }
        let targets = Tensor::from_vec(tgt_data, (g, comp_len), self.model.device())?;
        let idx = targets.unsqueeze(D::Minus1)?;
        logp.gather(&idx, D::Minus1)?.squeeze(D::Minus1)
    }

    /// The pre-P6-C **uncached** rollout: re-run the full-sequence
    /// [`GradModel::forward`] every step. Kept as the equivalence oracle for the
    /// cached [`generate`](Self::generate) — same sampler, same EOS/padding logic, so
    /// a per-token-identical cached path must reproduce its token stream and RNG
    /// consumption exactly. Test-only; the production path is `generate`.
    #[cfg(test)]
    fn generate_uncached(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        // Resolve the sampling parameters exactly as `generate` does, so the
        // oracle also covers the eval-override decode path (a cached-vs-uncached
        // gate with `eval_sampling: Some(..)` compares the same distribution).
        let (temperature, top_p) = self.resolve_sampling(cfg)?;
        let device = self.model.device().clone();
        let prompt_len = prompt.len();
        let width = prompt_len + cfg.max_new_tokens;
        let mut token_ids = Vec::with_capacity(cfg.group_size);
        let mut completion_lens = Vec::with_capacity(cfg.group_size);
        let mut rollout_logprobs = Vec::with_capacity(cfg.group_size);
        for _ in 0..cfg.group_size {
            let mut ids = prompt.to_vec();
            let mut logprobs = Vec::with_capacity(cfg.max_new_tokens);
            let mut comp_len = cfg.max_new_tokens;
            for step in 0..cfg.max_new_tokens {
                let len = ids.len();
                let input = Tensor::from_vec(ids.clone(), (1, len), &device)?;
                let logits = self.model.forward(&input)?;
                let last = logits.i((0, len - 1))?;
                let (next, logprob) = self.sampler.sample_with(&last, temperature, top_p)?;
                ids.push(next);
                logprobs.push(logprob);
                if cfg.eos_token_id == Some(next) {
                    comp_len = step + 1;
                    ids.resize(width, next);
                    break;
                }
            }
            token_ids.push(ids);
            completion_lens.push(comp_len);
            rollout_logprobs.push(logprobs);
        }
        Ok(Rollout {
            token_ids,
            prompt_len,
            completion_lens,
            rollout_logprobs: Some(rollout_logprobs),
        })
    }
}

impl<M: GradModel> LmPolicy<M> {
    /// Resolve one `generate` call's sampling parameters. The training path
    /// (no override) keeps the fail-loud temperature check: the sampler's
    /// temperature is fixed at construction (see [`new`](Self::new)) and
    /// scoring is temperature-consistent with it, so a disagreeing
    /// `cfg.temperature` is a drifted config, not a request. An explicit eval
    /// override (`cfg.eval_sampling`) deliberately samples a DIFFERENT
    /// distribution — eval-only temperature / nucleus top-p — and skips the
    /// check (`cfg.temperature` is documented as ignored then).
    fn resolve_sampling(&self, cfg: &GenConfig) -> CandleResult<(f64, Option<f64>)> {
        match cfg.eval_sampling {
            Some(eval) => {
                if !eval.temperature.is_finite() || eval.temperature <= 0.0 {
                    candle_core::bail!(
                        "eval_sampling.temperature must be finite and > 0, got {}",
                        eval.temperature
                    );
                }
                // Validate top_p HERE, not first at draw time inside the sampler:
                // by then the O(params) merged-weight build and the prompt prefill
                // have already been paid for a config that was never valid.
                if let Some(p) = eval.top_p {
                    if !p.is_finite() || p <= 0.0 || p > 1.0 {
                        candle_core::bail!("eval_sampling.top_p must be in (0, 1], got {p}");
                    }
                }
                Ok((eval.temperature, eval.top_p))
            }
            None => {
                if (cfg.temperature - self.temperature).abs() > f64::EPSILON {
                    candle_core::bail!(
                        "LmPolicy was built with temperature {} but generate was called \
                         with cfg.temperature {}; rebuild the policy to change it",
                        self.temperature,
                        cfg.temperature
                    );
                }
                Ok((self.temperature, None))
            }
        }
    }
}

/// Decode one group member on the shared cached decoder: reset the cache,
/// prefill the prompt, then sample up to `cfg.max_new_tokens` tokens at the
/// resolved `(temperature, top_p)` — EOS-inclusive early stop, right-padded
/// back to the fixed rectangular width — capturing each draw's behavior
/// log-prob. Returns `(ids, behavior_logprobs, completion_len)`; `ids` is the
/// full fixed-width row, `behavior_logprobs` has exactly `completion_len`
/// entries (one per real draw).
fn sample_one_sequence<D: CachedDecoder>(
    sampler: &mut GrpoSampler,
    decoder: &mut D,
    prompt: &[u32],
    cfg: &GenConfig,
    (temperature, top_p): (f64, Option<f64>),
    device: &candle_core::Device,
) -> CandleResult<(Vec<u32>, Vec<f32>, usize)> {
    let prompt_len = prompt.len();
    // The fixed rectangular width every sequence is padded/grown to.
    let width = prompt_len + cfg.max_new_tokens;
    decoder.reset_cache();
    let mut ids = prompt.to_vec();
    let mut logprobs = Vec::with_capacity(cfg.max_new_tokens);
    // Real completion tokens, counting up to and INCLUDING the first EOS.
    // Stays `max_new_tokens` unless an EOS early-stop overwrites it below.
    let mut comp_len = cfg.max_new_tokens;
    // Prefill the prompt (offset 0); its last position predicts token 1.
    let prompt_input = Tensor::from_vec(prompt.to_vec(), (1, prompt_len), device)?;
    let logits = decoder
        .forward(&prompt_input, 0)
        .map_err(crate::cuda_compat::translate_ptx_error)?;
    let mut last = logits.i((0, prompt_len - 1))?;
    let mut offset = prompt_len;
    for step in 0..cfg.max_new_tokens {
        let (next, logprob) = sampler.sample_with(&last, temperature, top_p)?;
        ids.push(next);
        logprobs.push(logprob);
        // EOS-inclusive early stop: keep the EOS token, record the true
        // length, and stop generating this sequence. With `eos_token_id ==
        // None` this never fires, so the loop runs the full `max_new_tokens`.
        if cfg.eos_token_id == Some(next) {
            comp_len = step + 1;
            // Right-pad the stopped sequence back to the fixed width so the
            // group stays rectangular. The pad value is the EOS token itself:
            // guaranteed in-vocab (it was just sampled) and masked out of the
            // loss / ignored by length-aware decoding.
            ids.resize(width, next);
            break;
        }
        // Feed the just-sampled token to advance the cache and get the next
        // step's logits — unless this was the final step (no further token to
        // predict), which keeps the sampler-draw count exactly `comp_len`.
        if step + 1 < cfg.max_new_tokens {
            let tok = Tensor::from_vec(vec![next], (1, 1), device)?;
            let logits = decoder
                .forward(&tok, offset)
                .map_err(crate::cuda_compat::translate_ptx_error)?;
            last = logits.i((0, 0))?;
            offset += 1;
        }
    }
    // (`logprobs.len() == comp_len` — one per real draw — is pinned by the
    // capture-alignment tests rather than a debug_assert, which would tip this
    // function over the cognitive-complexity bound.)
    debug_assert_eq!(ids.len(), width, "rollout row is not the fixed width");
    Ok((ids, logprobs, comp_len))
}

impl<M: GradModel> Policy for LmPolicy<M> {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let (temperature, top_p) = self.resolve_sampling(cfg)?;
        let device = self.model.device().clone();
        let prompt_len = prompt.len();
        // One KV-cached decoder snapshots the CURRENT merged weights (adapter folded
        // in, toggle respected) for the whole call; `reset_cache` starts each group
        // member on a fresh sequence. The first GPU kernel JIT happens building the
        // merged weights / in the first forward, so translate a driver-too-old PTX
        // mismatch (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) into an actionable
        // rebuild/upgrade message — a no-op off the `cuda` build and on the success path.
        let mut decoder = self
            .model
            .merged_decoder()
            .map_err(crate::cuda_compat::translate_ptx_error)?;
        let mut token_ids = Vec::with_capacity(cfg.group_size);
        let mut completion_lens = Vec::with_capacity(cfg.group_size);
        // Behavior-policy log-probs, one per draw: the sampler computes the full
        // sampling distribution anyway, so capturing the drawn token's log-prob
        // is free — see `Rollout::rollout_logprobs`.
        let mut rollout_logprobs = Vec::with_capacity(cfg.group_size);
        for _ in 0..cfg.group_size {
            let (ids, logprobs, comp_len) = sample_one_sequence(
                &mut self.sampler,
                &mut decoder,
                prompt,
                cfg,
                (temperature, top_p),
                &device,
            )?;
            token_ids.push(ids);
            completion_lens.push(comp_len);
            rollout_logprobs.push(logprobs);
        }
        // Built directly (not via `Rollout::rectangular`) so `completion_lens` carries
        // the true per-sequence lengths; under `eos_token_id == None` every entry is
        // `max_new_tokens` and this equals the rectangular construction exactly.
        Ok(Rollout {
            token_ids,
            prompt_len,
            completion_lens,
            rollout_logprobs: Some(rollout_logprobs),
        })
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let input = self.scoring_input(rollout)?;
        // Same CUDA-compat translation as `generate` (see there): a no-op off the
        // `cuda` build and on the success path.
        let logits = self
            .model
            .forward(&input)
            .map_err(crate::cuda_compat::translate_ptx_error)?; // [g, input_len, vocab]
        self.completion_logprobs(rollout, &logits)
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let input = self.scoring_input(rollout)?;
        // The value-only scorings (logp_old / the KL reference) route through
        // the model's detached forward: a rolling boundary detach frees each
        // layer's intermediates as the walk proceeds, and no checkpoint tape
        // is captured (so the tape of the NEXT update forward — the one
        // `backward` consumes — can never be clobbered by a value scoring).
        let logits = self
            .model
            .forward_detached(&input)
            .map_err(crate::cuda_compat::translate_ptx_error)?;
        // Already tape-free; the explicit detach states the trait contract
        // rather than trusting every model impl.
        Ok(self.completion_logprobs(rollout, &logits)?.detach())
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<candle_core::backprop::GradStore> {
        // Under activation checkpointing the model stitches the full gradient
        // from its boundary tape; otherwise this is exactly `loss.backward()`.
        self.model.backward(loss)
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

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.sampler.to_state_bytes()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        let restored = GrpoSampler::from_state_bytes(state)?;
        // The blob bakes the temperature it was checkpointed at. This policy
        // scores (and samples) at ITS OWN temperature, so a mismatched blob is a
        // cross-run restore the trait contract promises to fail loud on — the
        // restored RNG would otherwise continue a token stream the scorer
        // doesn't score (pre-R2 the blob's temperature silently won; post-R2 the
        // policy's silently would — neither is acceptable, so reject).
        if (restored.temperature() - self.temperature).abs() > f64::EPSILON {
            candle_core::bail!(
                "sampler state was checkpointed at temperature {} but this policy runs at {}; \
                 rebuild the policy with the checkpoint's temperature to resume it",
                restored.temperature(),
                self.temperature
            );
        }
        self.sampler = restored;
        Ok(())
    }

    fn lora_recipe(&self) -> Option<String> {
        self.model.lora_recipe()
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
        tiny_policy_at(1.0)
    }

    /// A tiny policy at an explicit rollout temperature (the temperature-consistent
    /// scoring tests need a non-1.0 one).
    fn tiny_policy_at(temperature: f64) -> QwenPolicy {
        let cfg = tiny_cfg();
        let vb = VarBuilder::from_tensors(weight_map(&cfg), DType::F32, &Device::Cpu);
        let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        QwenPolicy::new(model, 7, temperature)
    }

    /// Two policies sharing the SAME base weights and sampler seed, so they draw an
    /// identical token stream. `weight_map` is random and unseeded, so two
    /// independent `tiny_policy()` calls would NOT sample alike; cloning one map into
    /// both `VarBuilder`s makes them bit-identical. (The `LoRA` adapter is a no-op at
    /// its `B = 0` init, so only the shared base weights drive sampling — the
    /// per-policy random `A` factors never reach the logits.) This lets one policy
    /// observe a sampled token and the other stop on it *deterministically*, instead
    /// of relying on a cross-policy RNG coincidence.
    fn paired_policies() -> (QwenPolicy, QwenPolicy) {
        let cfg = tiny_cfg();
        let weights = weight_map(&cfg);
        let build = || {
            let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
            let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
            QwenPolicy::new(model, 7, 1.0)
        };
        (build(), build())
    }

    #[test]
    fn generate_returns_rectangular_group() {
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 3,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let rollout = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(rollout.len(), 4);
        assert_eq!(rollout.prompt_len, 3);
        // Every sequence has the same length (rectangular): prompt + new tokens.
        for ids in &rollout.token_ids {
            assert_eq!(ids.len(), 3 + 3);
            assert!(ids.iter().all(|&i| i < tiny_cfg().vocab_size as u32));
        }
        // No EOS configured: every completion is the full width, no early stop.
        assert_eq!(rollout.completion_lens, vec![3; 4]);
    }

    /// Assert the EOS-aware rollout invariants for every sequence: each row is the
    /// fixed `prompt_len + max_new` width, `completion_lens[i]` is exactly the
    /// EOS-inclusive length (first-EOS index + 1, or the full width when no EOS was
    /// sampled), and everything at/after that length is EOS padding (an empty tail
    /// for a full-width row). The `position` (first occurrence) check folds the
    /// "EOS at the boundary, none before it" invariants into one comparison.
    fn assert_eos_rollout_invariants(r: &Rollout, eos: u32, max_new: usize) {
        let width = r.prompt_len + max_new;
        for (gi, ids) in r.token_ids.iter().enumerate() {
            assert_eq!(ids.len(), width, "seq {gi} not padded to the fixed width");
            let comp = &ids[r.prompt_len..];
            let expected = comp
                .iter()
                .position(|&t| t == eos)
                .map_or(max_new, |i| i + 1);
            let cl = r.completion_lens[gi];
            assert_eq!(
                cl, expected,
                "seq {gi} completion_len {cl} != EOS-inclusive {expected}"
            );
            assert!(
                comp[cl..].iter().all(|&t| t == eos),
                "seq {gi} pad tail is not EOS-filled"
            );
        }
    }

    #[test]
    fn generate_stops_at_eos_inclusive_and_right_pads_to_fixed_width() {
        // EOS-aware generation: a sampled EOS ends the completion (EOS kept →
        // inclusive length) and the row is right-padded back to the FIXED width, so
        // the group stays rectangular and `completion_lens` carries the true lengths.
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let width = prompt.len() + max_new;
        let (mut p_ref, mut p_test) = paired_policies();

        // Reference run, no EOS: full-width rectangular, lengths all == max_new.
        let cfg_none = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let r_none = p_ref.generate(&prompt, &cfg_none).unwrap();
        assert_eq!(r_none.completion_lens, vec![max_new; 4]);
        for ids in &r_none.token_ids {
            assert_eq!(ids.len(), width);
        }

        // p_test shares p_ref's weights + seed, so it draws the SAME first token for
        // seq 0; setting that token as the EOS makes seq 0 stop at step 0 → an
        // EOS-inclusive length of exactly 1 with the rest padded.
        let eos = r_none.token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..cfg_none
        };
        let r = p_test.generate(&prompt, &cfg_eos).unwrap();

        assert_eq!(r.len(), 4);
        assert_eq!(r.prompt_len, prompt.len());
        // seq 0 stops at its first sampled token (== eos): inclusive length 1.
        assert_eq!(
            r.completion_lens[0], 1,
            "seq 0 did not stop at the first EOS"
        );
        // Every sequence: fixed width, EOS-inclusive length, EOS-filled pad tail.
        assert_eos_rollout_invariants(&r, eos, max_new);
    }

    #[test]
    fn generate_with_configured_but_unsampled_eos_is_full_width() {
        // A configured EOS that is never sampled (here an out-of-vocab id) must leave
        // generation identical to the None path: full width, every completion_len ==
        // max_new. This pins the "configured-yet-inert" branch — distinct from None —
        // deterministically: an out-of-vocab id can never equal a sampled token, so no
        // RNG coincidence is required.
        let mut policy = tiny_policy();
        let max_new = 4usize;
        let unsampled = tiny_cfg().vocab_size as u32; // == 16, never a valid sampled id
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: Some(unsampled),
            eval_sampling: None,
        };
        let r = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(r.completion_lens, vec![max_new; 4]);
        for ids in &r.token_ids {
            assert_eq!(ids.len(), 3 + max_new);
            assert!(ids.iter().all(|&t| t < unsampled));
        }
    }

    #[test]
    fn generate_eos_at_the_max_new_tokens_one_boundary() {
        // max_new_tokens == 1 with an EOS sampled at the only step: comp_len == 1 ==
        // max_new (the resize is a no-op — no double-handling) and each row is exactly
        // prompt + 1 wide. Paired policies make the single draw deterministic.
        let prompt = [2u32, 5];
        let (mut p_ref, mut p_test) = paired_policies();
        let base = GenConfig {
            group_size: 3,
            max_new_tokens: 1,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = p_ref.generate(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..base
        };
        let r = p_test.generate(&prompt, &cfg_eos).unwrap();
        assert_eq!(r.completion_lens[0], 1);
        for ids in &r.token_ids {
            assert_eq!(ids.len(), prompt.len() + 1);
        }
        assert_eos_rollout_invariants(&r, eos, 1);
    }

    /// The P6-C cached-rollout equivalence gate: the cached [`generate`] must
    /// reproduce the uncached oracle's token stream **and** consume an identical
    /// amount of sampler RNG (same draw count — which the RNG-state equality proves,
    /// since each draw advances the RNG a fixed amount regardless of the token). Runs
    /// both paths from the *same* saved sampler state on one policy. Generic
    /// over the model — the Llama gates below reuse it verbatim.
    fn assert_cached_matches_uncached<M: GradModel>(
        policy: &mut LmPolicy<M>,
        prompt: &[u32],
        cfg: &GenConfig,
    ) {
        let start = policy.sampler_state().unwrap();
        let cached = policy.generate(prompt, cfg).unwrap();
        let after_cached = policy.sampler_state().unwrap();

        policy.restore_sampler_state(&start).unwrap();
        let uncached = policy.generate_uncached(prompt, cfg).unwrap();
        let after_uncached = policy.sampler_state().unwrap();

        assert_eq!(
            cached.token_ids, uncached.token_ids,
            "cached rollout token stream diverged from the uncached oracle"
        );
        assert_eq!(
            cached.completion_lens, uncached.completion_lens,
            "cached rollout completion_lens diverged from the uncached oracle"
        );
        assert_eq!(cached.prompt_len, uncached.prompt_len);
        assert_eq!(
            after_cached, after_uncached,
            "cached path consumed a different amount of sampler RNG (draw-count mismatch)"
        );
        assert_rollout_logprobs_close(&cached, &uncached);
    }

    /// The captured behavior log-probs of two equivalent rollouts must agree —
    /// within a float tolerance, not bit-exactly: the merged (cached) and
    /// base+`LoRA` (uncached) forwards differ by F32 reassociation of the merge.
    fn assert_rollout_logprobs_close(cached: &Rollout, uncached: &Rollout) {
        let c_lp = cached.rollout_logprobs.as_ref().expect("cached capture");
        let u_lp = uncached
            .rollout_logprobs
            .as_ref()
            .expect("uncached capture");
        assert_eq!(c_lp.len(), u_lp.len());
        for (i, (c_row, u_row)) in c_lp.iter().zip(u_lp).enumerate() {
            assert_eq!(c_row.len(), u_row.len(), "seq {i} logprob count mismatch");
            for (j, (c, u)) in c_row.iter().zip(u_row).enumerate() {
                assert!(
                    (c - u).abs() <= 1e-4,
                    "seq {i} draw {j}: cached logprob {c} != uncached {u}"
                );
            }
        }
    }

    #[test]
    fn cached_generate_matches_uncached_adapter_on() {
        // Arm the adapter (B != 0) so the merge is non-trivial: the cached path must
        // reproduce the ADAPTER-AWARE uncached stream, not merely the base one.
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        assert!(policy.adapter_enabled());
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 6,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[1u32, 2, 3], &cfg);
    }

    #[test]
    fn cached_generate_matches_uncached_adapter_off() {
        // The eval path: adapter disabled => the snapshot is the pure base model.
        // Proves the toggle-respecting merge keeps eval's adapter-off rollout (and its
        // RNG consumption) identical to the uncached one.
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars()); // armed, but...
        policy.set_adapter_enabled(false); // ...disabled => base only
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[2u32, 4, 1], &cfg);
    }

    #[test]
    fn cached_generate_matches_uncached_with_eos() {
        // EOS early-stop + right-pad must be identical between paths, and the
        // sampler-RNG consumption must match — eval draws base then adapter from
        // successive RNG points, so a draw-count mismatch would desync them. A paired
        // probe picks a real first-token EOS deterministically; then compare cached vs
        // uncached on a fresh-sampler policy that draws that same first token.
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let (mut probe, mut policy) = paired_policies();
        let base = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = probe.generate_uncached(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..base
        };
        assert_cached_matches_uncached(&mut policy, &prompt, &cfg_eos);
    }

    /// THE R2 capture-alignment gate: every captured behavior log-prob must agree
    /// with the scoring path's log-prob of the same (sequence, draw) — at the
    /// policy temperature. Generation samples from `softmax(merged_logits / T)` and
    /// scoring computes `log_softmax(uncached_logits / T)` (temperature-consistent
    /// scoring), so on an all-F32 tiny model the two can differ only by float
    /// reassociation of the merge. A capture indexing bug (wrong token, shifted
    /// position) or a scoring-temperature bug shows up as a gross mismatch.
    /// Run at T = 1.0 (the bit-identical default) and a non-trivial T = 0.7.
    #[test]
    fn captured_behavior_logprobs_align_with_the_scoring_path() {
        for temperature in [1.0, 0.7] {
            let mut policy = tiny_policy_at(temperature);
            force_b_nonzero(&policy.trainable_vars()); // non-trivial merge
            let cfg = GenConfig {
                group_size: 3,
                max_new_tokens: 4,
                temperature,
                eos_token_id: None,
                eval_sampling: None,
            };
            let rollout = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
            let captured = rollout.rollout_logprobs.clone().expect("capture present");
            let scored = policy
                .token_logprobs(&rollout)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for (i, row) in captured.iter().enumerate() {
                assert_eq!(row.len(), 4, "full-width capture expected");
                for (j, &lp) in row.iter().enumerate() {
                    assert!(
                        (lp - scored[i][j]).abs() <= 1e-4,
                        "T={temperature} seq {i} draw {j}: behavior logprob {lp} != scored {}",
                        scored[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn captured_logprob_rows_match_the_true_completion_lens_under_eos() {
        // EOS early-stop: row i carries exactly completion_lens[i] log-probs (one
        // per real draw) — the EOS padding was never sampled, so it has none.
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let (mut p_ref, mut p_test) = paired_policies();
        let base = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = p_ref.generate(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let r = p_test
            .generate(
                &prompt,
                &GenConfig {
                    eos_token_id: Some(eos),
                    ..base
                },
            )
            .unwrap();
        let captured = r.rollout_logprobs.as_ref().expect("capture present");
        assert_eq!(captured.len(), r.len());
        for (row, &len) in captured.iter().zip(&r.completion_lens) {
            assert_eq!(row.len(), len, "one behavior logprob per real draw");
            assert!(row.iter().all(|lp| lp.is_finite() && *lp <= 0.0));
        }
        assert_eq!(r.completion_lens[0], 1, "seq 0 stops at the probed EOS");
    }

    #[test]
    fn eval_sampling_override_bypasses_the_temperature_check() {
        // The override is the deliberate eval channel: it samples its own
        // temperature/top-p and skips the baked-temperature equality check
        // (cfg.temperature is documented as ignored). A mismatched
        // cfg.temperature that would bail on the training path must not bail
        // here.
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 4,
            temperature: 123.0, // would bail without the override
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.5,
                top_p: Some(0.9),
            }),
        };
        let before = policy.sampler_state().unwrap();
        let r = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r.completion_lens, vec![4; 3]);
        assert!(r.rollout_logprobs.is_some(), "override path still captures");
        assert_ne!(
            before,
            policy.sampler_state().unwrap(),
            "override sampling must advance the shared RNG"
        );

        // Without the override the same mismatched temperature fails loud.
        let train_cfg = GenConfig {
            eval_sampling: None,
            ..cfg
        };
        assert!(policy.generate(&[1u32, 2, 3], &train_cfg).is_err());
    }

    #[test]
    fn malformed_eval_override_fails_before_decoding() {
        // A malformed override temperature or top_p fails loud — BEFORE the
        // merged-decoder build (resolve_sampling validates both).
        let mut policy = tiny_policy();
        let base = GenConfig {
            group_size: 2,
            max_new_tokens: 2,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let bad = GenConfig {
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.0,
                top_p: None,
            }),
            ..base
        };
        assert!(policy.generate(&[1u32, 2, 3], &bad).is_err());
        let bad_p = GenConfig {
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.6,
                top_p: Some(1.5),
            }),
            ..base
        };
        assert!(policy.generate(&[1u32, 2, 3], &bad_p).is_err());
    }

    #[test]
    fn eval_override_values_actually_reach_the_sampler() {
        // The mutation-killer for the override plumbing: a resolve_sampling that
        // validates but then samples the TRAINING parameters passes the no-bail
        // test above — this one it cannot pass. With top_p so small that only
        // the argmax survives, every draw's nucleus is a single token, so every
        // captured behavior log-prob is EXACTLY ln(p/p) = 0.0 and every group
        // member decodes the identical greedy stream. Without the override
        // plumbed, the full-softmax probabilities over a 16-token vocab make
        // every log-prob strictly negative.
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 4,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.5,
                top_p: Some(1e-6),
            }),
        };
        let r = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        let captured = r.rollout_logprobs.as_ref().expect("capture present");
        for (i, row) in captured.iter().enumerate() {
            for (j, &lp) in row.iter().enumerate() {
                assert_eq!(
                    lp, 0.0,
                    "seq {i} draw {j}: argmax-nucleus logprob must be exactly 0, got {lp} \
                     (override top_p did not reach the sampler?)"
                );
            }
        }
        assert!(
            r.token_ids.iter().all(|ids| ids == &r.token_ids[0]),
            "argmax-nucleus decoding must be greedy-deterministic across the group"
        );
    }

    #[test]
    fn cached_generate_matches_uncached_under_the_eval_override() {
        // The override decode path gets the same cached-vs-uncached equivalence
        // gate as the training path: same token stream, same RNG consumption,
        // logprobs within merge-reassociation tolerance (the uncached oracle
        // resolves the override exactly like generate does).
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.7,
                top_p: Some(0.9),
            }),
        };
        assert_cached_matches_uncached(&mut policy, &[2u32, 4, 1], &cfg);
    }

    #[test]
    fn restore_rejects_a_mismatched_sampler_temperature() {
        // The blob bakes the temperature it was checkpointed at; restoring it
        // into a policy scoring at a DIFFERENT temperature must fail loud (the
        // Policy trait's documented contract) instead of silently continuing a
        // token stream the scorer doesn't score.
        let mut policy = tiny_policy(); // T = 1.0
        let foreign = GrpoSampler::new(5, 0.7).to_state_bytes().unwrap();
        let err = policy.restore_sampler_state(&foreign).unwrap_err();
        assert!(
            err.to_string().contains("temperature"),
            "expected a temperature-mismatch error, got: {err}"
        );
        // A matching-temperature blob restores fine.
        let matching = GrpoSampler::new(5, 1.0).to_state_bytes().unwrap();
        policy.restore_sampler_state(&matching).unwrap();
    }

    #[test]
    fn token_logprobs_shape_and_finiteness() {
        let policy = tiny_policy();
        // Two sequences, prompt_len 2, completion_len 3 (rectangular).
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![1, 2, 6, 7, 8]], 2);
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
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![3, 1, 4, 1, 5]], 2);
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

    #[test]
    fn token_logprobs_at_a_non_unit_temperature_matches_a_manual_reference() {
        // Temperature-consistent scoring: at T != 1 the log-probs must equal
        // log_softmax(logits / T) gathered at the completion tokens — recomputed
        // here independently of the narrow/scale/gather under test.
        let policy = tiny_policy_at(0.7);
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![3, 1, 4, 1, 5]], 2);
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
        let scaled = (policy.model().forward(&input).unwrap() / 0.7).unwrap();
        let logp_full = log_softmax(&scaled, D::Minus1)
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
                    "T=0.7 logp[{gi}][{j}]={} != manual {want}",
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
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![5, 4, 3, 2, 1]], 2);
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
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4]], 2);
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
        assert!(dbg.contains("LmPolicy") && dbg.contains(".."));
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

    #[test]
    fn rollout_ratio_telemetry_is_near_one_for_an_f32_policy() {
        // End-to-end pipeline gate for the R2 telemetry: on an all-F32 model the
        // rollout (merged cached decode) and the scoring forward differ only by
        // float reassociation, so the train/rollout ratio must sit hard against 1
        // on every step — and nothing may approach the TIS cap. A capture
        // misalignment, a temperature inconsistency, or a wiring bug shows up as
        // a ratio visibly away from 1.
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars()); // non-trivial merge
        let prompts = vec!["abc".to_string(), "bcd".to_string()];
        let cfg = TrainerConfig {
            steps: 3,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            lr: 1e-3,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run = RunDir::create(&tmp.0, "qwen-ratio").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let history = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &prompts)
            .unwrap();
        assert_eq!(history.len(), 3);
        for m in &history {
            assert_on_policy_ratio_metrics(m);
        }
    }

    /// Per-step assertions for an all-F32 (reassociation-only) run: ratio
    /// mean/max hard against 1, the drift meter against 0, nothing capped, and
    /// the telemetry not dark.
    fn assert_on_policy_ratio_metrics(m: &crate::telemetry::Metrics) {
        assert!(
            (m.rollout_ratio_mean - 1.0).abs() <= 1e-3,
            "step {}: rollout_ratio_mean {} far from 1 on an F32 model",
            m.step,
            m.rollout_ratio_mean
        );
        assert!(
            (m.rollout_ratio_max - 1.0).abs() <= 1e-3,
            "step {}: rollout_ratio_max {} far from 1 on an F32 model",
            m.step,
            m.rollout_ratio_max
        );
        assert_eq!(
            m.frac_rollout_ratio_capped, 0.0,
            "step {}: no token can sit above the TIS cap on an F32 model",
            m.step
        );
        assert!(
            m.rollout_logratio_mean.abs() <= 1e-3,
            "step {}: drift meter {} far from 0 on an F32 model",
            m.step,
            m.rollout_logratio_mean
        );
        assert!(
            m.rollout_capture_tokens > 0,
            "step {}: telemetry must not be dark — the policy captures",
            m.step
        );
    }

    #[test]
    fn tis_enabled_run_completes_with_near_unit_weights() {
        // With TIS ON against an F32 policy the weights are ~1, so the run must
        // behave like the uncorrected one: finite metrics, real optimizer steps.
        // (The fail-loud path for a policy WITHOUT capture is pinned in
        // tests/toy_echo.rs; the weight math itself in trainer.rs unit tests.)
        let mut policy = tiny_policy();
        let prompts = vec!["abc".to_string(), "bcd".to_string()];
        let cfg = TrainerConfig {
            steps: 3,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            lr: 1e-3,
            tis: true,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run = RunDir::create(&tmp.0, "qwen-tis").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let history = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &prompts)
            .unwrap();
        assert_eq!(history.len(), 3);
        for m in &history {
            assert_step_metrics_ok(m);
        }
        assert!(
            history.iter().any(|m| m.grad_norm > 0.0),
            "no optimizer step ran under TIS"
        );
    }

    /// A [`Policy`] wrapper that shifts every captured behavior log-prob DOWN
    /// by a constant `delta` — claiming the rollout assigned `e^delta`-times
    /// less mass to each token than it actually did — so the train/rollout
    /// ratio is `e^delta` (× merge-reassociation noise) at every loss token by
    /// construction: a deterministic off-policy injection for the end-to-end
    /// TIS / telemetry gates.
    struct ShiftedCapture<P: Policy> {
        inner: P,
        delta: f32,
    }

    impl<P: Policy> Policy for ShiftedCapture<P> {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            let mut r = self.inner.generate(prompt, cfg)?;
            if let Some(rows) = &mut r.rollout_logprobs {
                for row in rows {
                    for lp in row {
                        *lp -= self.delta;
                    }
                }
            }
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
        fn lora_recipe(&self) -> Option<String> {
            self.inner.lora_recipe()
        }
    }

    /// Per-run assertions for the δ-shifted capture: ratio mean at e^δ (and
    /// NOT e^{−δ} — the direction pin), the drift meter at +δ, every loss
    /// token above the cap, telemetry not dark.
    fn assert_shifted_ratio_metrics(m: &crate::telemetry::Metrics, delta: f32) {
        assert!(
            (m.rollout_ratio_mean - 2.0).abs() <= 2e-3,
            "ratio mean {} != e^ln2 = 2 (direction/magnitude)",
            m.rollout_ratio_mean
        );
        assert!(
            (m.rollout_logratio_mean - delta).abs() <= 1e-3,
            "drift meter {} != +ln2",
            m.rollout_logratio_mean
        );
        assert_eq!(
            m.frac_rollout_ratio_capped, 1.0,
            "every token sits above the cap by construction"
        );
        assert!(m.rollout_capture_tokens > 0);
    }

    #[test]
    fn tis_and_ratio_telemetry_verified_end_to_end_with_a_shifted_capture() {
        // δ-shifted capture ⇒ every loss token's ratio is e^δ ≈ 2; with cap
        // C = 1.5 < e^δ, EVERY token caps. Through Trainer::train this pins:
        //   (1) the ratio DIRECTION (a swapped exp(b−a) would read e^{−δ} = ½);
        //   (2) the log-ratio drift meter (≈ +δ);
        //   (3) the capped-fraction wiring (≡ 1.0) and the token count;
        //   (4) the TIS weight reaching the GRADIENT: uniformly capped weights
        //       make the first step's pre-clip grad_norm exactly C × the
        //       tis-off run's (paired policies ⇒ identical rollouts/logp_old).
        // A dropped `tis_w` (the one-line mutant), a swapped ratio, or a
        // disconnected capped-fraction all redden this test.
        let delta = std::f64::consts::LN_2;
        let cap = 1.5_f64;
        let (p_a, p_b) = paired_policies();
        // paired_policies shares the BASE weights and sampler seed, but each
        // policy draws its own random LoRA A factors — invisible to the forward
        // at B = 0 (so rollouts/logp_old still match), yet dL/dB ∝ A, so the
        // grad-norm comparison below needs the adapters synced too.
        for (va, vb) in p_a.trainable_vars().iter().zip(p_b.trainable_vars()) {
            vb.set(va.as_tensor()).unwrap();
        }
        let mut off = ShiftedCapture {
            inner: p_a,
            delta: delta as f32,
        };
        let mut on = ShiftedCapture {
            inner: p_b,
            delta: delta as f32,
        };
        let prompts = vec!["abc".to_string(), "bcd".to_string()];
        let cfg = |tis: bool| TrainerConfig {
            steps: 1,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            lr: 1e-3,
            tis,
            tis_imp_ratio_cap: cap,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run_off = RunDir::create(&tmp.0, "tis-off").unwrap();
        let m_off = Trainer::new(cfg(false), &run_off)
            .unwrap()
            .train(&mut off, &SpreadReward, &CharCodec, &prompts)
            .unwrap()
            .remove(0);
        let run_on = RunDir::create(&tmp.0, "tis-on").unwrap();
        let m_on = Trainer::new(cfg(true), &run_on)
            .unwrap()
            .train(&mut on, &SpreadReward, &CharCodec, &prompts)
            .unwrap()
            .remove(0);

        for m in [&m_off, &m_on] {
            assert_shifted_ratio_metrics(m, delta as f32);
        }
        // (4) the weight scales the gradient.
        assert!(m_off.grad_norm > 0.0, "tis-off run must take a real step");
        let scale = m_on.grad_norm / m_off.grad_norm;
        assert!(
            (f64::from(scale) - cap).abs() <= 1e-3,
            "grad_norm scaled by {scale}, want exactly the cap {cap}"
        );
    }

    #[test]
    fn evaluate_honors_the_eval_sampling_override_end_to_end() {
        // The held-out eval harness over a real (tiny) policy with the eval-only
        // sampling convention: a temperature different from the baked one plus
        // nucleus top-p must generate (no temperature bail) and produce a finite
        // report, with the adapter flag restored.
        let mut policy = tiny_policy();
        let prompts = vec!["abc".to_string()];
        let gen = GenConfig {
            group_size: 4,
            max_new_tokens: 3,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling::default()), // T 0.6 / top-p 0.95
        };
        let report =
            crate::eval::evaluate(&mut policy, &SpreadReward, &CharCodec, &prompts, &gen).unwrap();
        assert_eq!(report.n_prompts, 1);
        assert_eq!(report.group_size, 4);
        assert!(report.base_reward_mean.is_finite());
        assert!(report.adapter_reward_mean.is_finite());
        assert!(policy.adapter_enabled(), "adapter flag restored");
    }

    // ---- LlamaPolicy: the M1 second-implementor gates ------------------------
    //
    // Everything below reuses the GENERIC machinery above unchanged
    // (`assert_cached_matches_uncached`, `force_b_nonzero`, the codec/reward/
    // trainer scaffold) — only the model construction is Llama-specific. That
    // reuse IS the point: it witnesses that the `GradModel` seam, not the test
    // code, carries the architecture difference.

    use crate::llama::LlamaGradModel;
    use candle_transformers::models::llama::Config as LlamaConfig;

    /// A tiny dense-Llama config (2 layers, 2 Q / 1 KV head → real GQA, derived
    /// `head_dim` 4) — the same scaffold llama.rs's tests use.
    fn llama_tiny_cfg() -> LlamaConfig {
        LlamaConfig {
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            use_flash_attn: false,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 32,
            tie_word_embeddings: true,
        }
    }

    /// Random weights matching the llama dotted tensor names (tied head → no
    /// `lm_head.weight`; no QK-norm tensors, no biases).
    fn llama_weight_map(cfg: &LlamaConfig) -> HashMap<String, Tensor> {
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
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let qo = cfg.num_attention_heads * head_dim;
        let kvo = cfg.num_key_value_heads * head_dim;
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
            put(&format!("{p}.mlp.gate_proj.weight"), &[i, h]);
            put(&format!("{p}.mlp.up_proj.weight"), &[i, h]);
            put(&format!("{p}.mlp.down_proj.weight"), &[h, i]);
        }
        t
    }

    fn llama_tiny_policy() -> LlamaPolicy {
        let cfg = llama_tiny_cfg();
        let vb = VarBuilder::from_tensors(llama_weight_map(&cfg), DType::F32, &Device::Cpu);
        let model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        LlamaPolicy::new(model, 7, 1.0)
    }

    /// Two Llama policies sharing the SAME base weights and sampler seed (the
    /// same determinism device as [`paired_policies`] — see there for why).
    fn llama_paired_policies() -> (LlamaPolicy, LlamaPolicy) {
        let cfg = llama_tiny_cfg();
        let weights = llama_weight_map(&cfg);
        let build = || {
            let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
            let model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
            LlamaPolicy::new(model, 7, 1.0)
        };
        (build(), build())
    }

    #[test]
    fn llama_cached_generate_matches_uncached_adapter_on() {
        // Armed adapter (B != 0): the cached path must reproduce the
        // ADAPTER-AWARE uncached stream, not merely the base one.
        let mut policy = llama_tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        assert!(policy.adapter_enabled());
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 6,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[1u32, 2, 3], &cfg);
    }

    #[test]
    fn llama_cached_generate_matches_uncached_adapter_off() {
        // The eval path: adapter disabled => the snapshot is the pure base model.
        let mut policy = llama_tiny_policy();
        force_b_nonzero(&policy.trainable_vars()); // armed, but...
        policy.set_adapter_enabled(false); // ...disabled => base only
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[2u32, 4, 1], &cfg);
    }

    #[test]
    fn llama_cached_generate_matches_uncached_with_eos() {
        // EOS early-stop + right-pad identical between paths, with matching
        // sampler-RNG consumption (same deterministic paired-probe pattern as
        // the Qwen gate).
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let (mut probe, mut policy) = llama_paired_policies();
        let base = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = probe.generate_uncached(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..base
        };
        assert_cached_matches_uncached(&mut policy, &prompt, &cfg_eos);
    }

    #[test]
    fn llama_drives_a_grpo_step_through_the_trainer_on_cpu() {
        // THE M1 extended reusability gate: the SAME `Trainer` (and the same
        // codec + reward scaffold) that drives the P2 echo toy and the Qwen
        // policy drives `LmPolicy<LlamaGradModel>` UNCHANGED — rollout via the
        // Llama merged decoder, reward, advantages, backward THROUGH the Llama
        // forward, grad-coverage canary, FerrlAdamW. `grad_norm > 0` witnesses
        // a real optimizer step (no learning-curve assertion — the platform-
        // dependence lesson); beta > 0 routes the adapter-disabled KL reference
        // forward through the Llama path too.
        let mut policy = llama_tiny_policy();
        let prompts = vec!["abc".to_string(), "bcd".to_string()];
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
        let run = RunDir::create(&tmp.0, "llama-cpu").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let history = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &prompts)
            .unwrap();

        assert_eq!(history.len(), 4);
        for m in &history {
            assert_step_metrics_ok(m);
        }
        assert!(
            history.iter().any(|m| m.grad_norm > 0.0),
            "no AdamW step ran — the Llama backward path was never exercised"
        );
        // The adapter is restored enabled after the (reference-toggling) run.
        assert!(policy.adapter_enabled());
    }

    // ---- detached scoring + checkpointed backward (P7) ----------------------

    /// Force every adapter `B` nonzero so the adapter path is live in the
    /// scored logits (at the `B = 0` init both scorings would trivially agree).
    fn arm_policy(policy: &QwenPolicy) {
        for v in policy.trainable_vars().iter().skip(1).step_by(2) {
            let dims = v.as_tensor().dims().to_vec();
            v.set(&Tensor::randn(0f32, 0.5f32, dims, &Device::Cpu).unwrap())
                .unwrap();
        }
    }

    fn small_rollout(policy: &mut QwenPolicy) -> Rollout {
        let cfg = GenConfig {
            group_size: 2,
            max_new_tokens: 3,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        policy.generate(&[1u32, 2, 3], &cfg).unwrap()
    }

    /// A fixed non-uniform weighted sum of the scored log-probs — the loss
    /// stand-in for the backward-seam tests.
    fn probe_loss_of(policy: &QwenPolicy, rollout: &Rollout) -> Tensor {
        let logp = policy.token_logprobs(rollout).unwrap();
        let n = logp.elem_count();
        let w: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.3 - 0.5).collect();
        let w = Tensor::from_vec(w, logp.dims().to_vec(), &Device::Cpu).unwrap();
        logp.mul(&w).unwrap().sum_all().unwrap()
    }

    #[test]
    fn detached_scoring_matches_token_logprobs_and_is_tape_free() {
        let mut policy = tiny_policy();
        arm_policy(&policy);
        let rollout = small_rollout(&mut policy);

        let live = policy.token_logprobs(&rollout).unwrap();
        let det = policy.token_logprobs_detached(&rollout).unwrap();
        assert_eq!(
            det.to_vec2::<f32>().unwrap(),
            live.to_vec2::<f32>().unwrap(),
            "the detached scoring drifted from token_logprobs"
        );

        // Tape-free: a backward through the detached scores reaches no var…
        let store = det.sum_all().unwrap().backward().unwrap();
        assert!(policy
            .trainable_vars()
            .iter()
            .all(|v| store.get(v).is_none()));
        // …while the live path IS on the tape (the comparison is non-vacuous).
        let store = live.sum_all().unwrap().backward().unwrap();
        assert!(policy
            .trainable_vars()
            .iter()
            .any(|v| store.get(v).is_some()));
    }

    /// `Policy::backward` under activation checkpointing: full var coverage,
    /// gradients matching the uncut run on the same instance — the
    /// policy-level end-to-end of the remat stitch.
    #[test]
    fn policy_backward_stitches_under_checkpointing() {
        let mut policy = tiny_policy();
        arm_policy(&policy);
        let rollout = small_rollout(&mut policy);
        let vars = policy.trainable_vars();

        let plain = policy.backward(&probe_loss_of(&policy, &rollout)).unwrap();
        policy.model_mut().set_activation_checkpointing(true);
        let stitched = policy.backward(&probe_loss_of(&policy, &rollout)).unwrap();

        for v in &vars {
            let a = plain.get(v).expect("var missing from the uncut store");
            let b = stitched
                .get(v)
                .expect("var missing from the stitched store");
            let diff: f32 = a
                .sub(b)
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
                diff <= 1e-5,
                "stitched grad diverged from the uncut backward by {diff}"
            );
        }
    }
}
