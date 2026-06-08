//! A grad-bearing, uncached Qwen3 forward — our half of the two-forward split.
//!
//! candle-transformers ships a Qwen3 forward, but it is inference-shaped
//! (`&mut self` + `ConcatKvCache`, all layer types `pub(crate)`) and built from
//! ops that have **no backward**, so it cannot be used to train. This module is the
//! *update* path: a full-sequence, uncached forward over the **same loaded
//! weights**, expressed entirely in grad-bearing ops, with a manual `LoRA` adapter
//! on `q_proj`/`v_proj`. It is **also** the (uncached) **rollout** forward used by
//! [`crate::qwen_policy`] — the only *adapter-aware* forward; candle's shipped
//! cached forward carries no adapter, so a fast merged-weight rollout is a later
//! optimization.
//!
//! It is gated against the shipped forward by an equivalence test (same weights →
//! same logits) and a LoRA-grad-coverage test (the adapter trains).
//!
//! ## The three grad landmines (all replaced here)
//!
//! candle's fast kernels for these three ops are autograd-cutting custom ops
//! (`BackpropOp::none()` / `BackwardNotSupported`); using any of them in the
//! update path would silently sever the `LoRA` gradients (the grad-coverage canary
//! would catch it). Each has a pure-tensor, grad-bearing, numerically-equal twin:
//!
//! | fused (no backward)              | grad-safe twin used here              |
//! |---------------------------------|---------------------------------------|
//! | `candle_nn::ops::rms_norm`       | [`crate::nn::RmsNorm`] (`rms_norm_slow`) |
//! | `candle_nn::rotary_emb::rope`    | `candle_nn::rotary_emb::rope_slow`    |
//! | `candle_nn::ops::softmax_last_dim`| `candle_nn::ops::softmax(_, D::Minus1)` |
//!
//! Every other op in the forward (matmul, transpose/reshape, `repeat_kv`,
//! residual add, `SwiGLU` mul, `Silu`, embedding lookup, the tied `lm_head` matmul)
//! is grad-bearing and reused verbatim.

use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::ops::softmax;
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen3::Config;

use crate::lora::LoraLinear;
use crate::nn::RmsNorm;

/// `y = x Wᵀ` for a frozen weight `w` of shape `[out, in]` (candle `Linear`
/// layout), broadcasting over the leading dims of `x`.
fn frozen_linear(x: &Tensor, w: &Tensor) -> CandleResult<Tensor> {
    x.broadcast_matmul(&w.t()?)
}

/// Grouped-query repeat: `[b, kv_heads, l, d] -> [b, kv_heads * n_rep, l, d]`,
/// each kv head repeated `n_rep` times consecutively (matching candle's
/// `repeat_kv`). Pure `expand`+`reshape`, so it carries gradient.
fn repeat_kv(x: &Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_heads, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv_heads, n_rep, l, d))?
        .contiguous()?
        .reshape((b, kv_heads * n_rep, l, d))
}

/// Precomputed `RoPE` `cos`/`sin` tables, each `[max_pos, head_dim / 2]` (the
/// half-width `rope_slow` expects). Built exactly as candle's
/// `Qwen3RotaryEmbedding::new`.
#[derive(Debug, Clone)]
struct RotaryTables {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryTables {
    fn new(cfg: &Config, dtype: DType, device: &Device) -> CandleResult<Self> {
        let dim = cfg.head_dim;
        let max_pos = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        // Match the model dtype (shipped Qwen3RotaryEmbedding casts here too) so
        // rope_slow's broadcast_mul against q/k does not dtype-mismatch at bf16.
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// `cos`/`sin` narrowed to the first `seq_len` positions.
    fn slice(&self, seq_len: usize) -> CandleResult<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(0, 0, seq_len)?,
            self.sin.narrow(0, 0, seq_len)?,
        ))
    }
}

/// One Qwen3 attention block. `q_proj`/`v_proj` carry the `LoRA` adapter; the
/// rest are frozen. Replicates candle's `Qwen3Attention::forward` with the three
/// grad-safe substitutions and no KV cache.
#[derive(Debug)]
struct QwenAttention {
    q_proj: LoraLinear,
    v_proj: LoraLinear,
    k_weight: Tensor,
    o_weight: Tensor,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
}

impl QwenAttention {
    fn load(cfg: &Config, vb: &VarBuilder, rank: usize, alpha: f64) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
        let q_weight = vb.pp("q_proj").get((q_out, h), "weight")?;
        let v_weight = vb.pp("v_proj").get((kv_out, h), "weight")?;
        let eps = cfg.rms_norm_eps as f32;
        Ok(Self {
            q_proj: LoraLinear::new(q_weight, None, rank, alpha)?,
            v_proj: LoraLinear::new(v_weight, None, rank, alpha)?,
            k_weight: vb.pp("k_proj").get((kv_out, h), "weight")?,
            o_weight: vb.pp("o_proj").get((h, q_out), "weight")?,
            q_norm: RmsNorm::new(vb.pp("q_norm").get(head_dim, "weight")?, eps),
            k_norm: RmsNorm::new(vb.pp("k_norm").get(head_dim, "weight")?, eps),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim,
            attn_hidden: q_out,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Projections (q, v carry LoRA; k frozen).
        let q = self.q_proj.forward(x)?;
        let k = frozen_linear(x, &self.k_weight)?;
        let v = self.v_proj.forward(x)?;

        // 2. (B, L, H, D) -> (B, H, L, D).
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head QK-Norm (grad-safe rms_norm_slow) BEFORE RoPE.
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // 4. RoPE (grad-safe rope_slow; half-width cos/sin).
        let (cos, sin) = rot.slice(l)?;
        let q = rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k = rope_slow(&k.contiguous()?, &cos, &sin)?;

        // 5. (no KV cache) GQA repeat.
        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v.contiguous()?, self.num_kv_groups)?.contiguous()?;

        // 6. Scaled dot-product attention with grad-safe softmax.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?;

        // 7. Output projection.
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.attn_hidden))?;
        frozen_linear(&ctx, &self.o_weight)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.q_proj.set_enabled(enabled);
        self.v_proj.set_enabled(enabled);
    }

    fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = self.q_proj.trainable_vars();
        vars.extend(self.v_proj.trainable_vars());
        vars
    }
}

/// `SwiGLU` MLP, all frozen.
#[derive(Debug)]
struct QwenMlp {
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    act: Activation,
}

impl QwenMlp {
    fn load(cfg: &Config, vb: &VarBuilder) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_weight: vb.pp("gate_proj").get((i, h), "weight")?,
            up_weight: vb.pp("up_proj").get((i, h), "weight")?,
            down_weight: vb.pp("down_proj").get((h, i), "weight")?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = frozen_linear(x, &self.gate_weight)?.apply(&self.act)?;
        let rhs = frozen_linear(x, &self.up_weight)?;
        frozen_linear(&lhs.broadcast_mul(&rhs)?, &self.down_weight)
    }
}

/// One decoder layer: pre-norm attention + pre-norm `SwiGLU`, both residual.
#[derive(Debug)]
struct QwenLayer {
    ln1: RmsNorm,
    attn: QwenAttention,
    ln2: RmsNorm,
    mlp: QwenMlp,
}

impl QwenLayer {
    fn load(cfg: &Config, vb: &VarBuilder, rank: usize, alpha: f64) -> CandleResult<Self> {
        let eps = cfg.rms_norm_eps as f32;
        let h = cfg.hidden_size;
        Ok(Self {
            ln1: RmsNorm::new(vb.pp("input_layernorm").get(h, "weight")?, eps),
            attn: QwenAttention::load(cfg, &vb.pp("self_attn"), rank, alpha)?,
            ln2: RmsNorm::new(vb.pp("post_attention_layernorm").get(h, "weight")?, eps),
            mlp: QwenMlp::load(cfg, &vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h, mask, rot)?;
        let x = x.broadcast_add(&h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x.broadcast_add(&h2)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.attn.set_adapter_enabled(enabled);
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.attn.trainable_vars()
    }
}

/// A grad-bearing, uncached Qwen3 forward with `LoRA` on `q_proj`/`v_proj`.
///
/// Built from the same [`VarBuilder`] (over the same safetensors) as candle's
/// shipped `ModelForCausalLM`, so the two are weight-identical and their logits
/// match (the P3 equivalence gate). The base weights are frozen [`Tensor`]s; only
/// the `LoRA` `A`/`B` factors are trainable [`Var`]s.
#[derive(Debug)]
pub struct QwenGradModel {
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<QwenLayer>,
    norm: RmsNorm,
    rot: RotaryTables,
    hidden: usize,
    device: Device,
    dtype: DType,
}

impl QwenGradModel {
    /// Load the model from `vb`, attaching a `LoRA` adapter of the given `rank`
    /// and `alpha` to every layer's `q_proj` and `v_proj`.
    ///
    /// `vb` must be over the Qwen3 safetensors (any dtype; F32 for the CPU
    /// equivalence gate). `cfg` is candle's own `qwen3::Config` so derived dims
    /// match the shipped model exactly. Only the bias-free, non-sliding-window
    /// configuration (as used by Qwen3-0.6B-Base) is supported; other configs are
    /// rejected (see Errors) rather than loaded as a silently non-parity model.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `cfg` requests an unsupported option
    /// (`attention_bias`, `use_sliding_window`), if a weight tensor is missing or
    /// mis-shaped, or if the `LoRA` factors cannot be allocated.
    pub fn load(cfg: &Config, vb: &VarBuilder, rank: usize, alpha: f64) -> CandleResult<Self> {
        // Fail loud on Config options this forward does not implement, rather than
        // silently loading a non-parity model. Qwen3-0.6B-Base uses neither: candle's
        // shipped loader honors `attention_bias` on all four projections and a
        // sliding-window mask, but this update path loads bias-free q/k/v/o and only
        // a full causal mask.
        if cfg.attention_bias {
            candle_core::bail!(
                "QwenGradModel: cfg.attention_bias=true is unsupported (this forward \
                 loads bias-free q/k/v/o projections to match Qwen3-0.6B-Base; loading \
                 it would silently produce non-parity logits)"
            );
        }
        if cfg.use_sliding_window {
            candle_core::bail!(
                "QwenGradModel: cfg.use_sliding_window=true is unsupported (only a full \
                 causal mask is implemented)"
            );
        }
        let h = cfg.hidden_size;
        let embed = vb
            .pp("model.embed_tokens")
            .get((cfg.vocab_size, h), "weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(vb.pp("lm_head").get((cfg.vocab_size, h), "weight")?)
        };
        let layers_vb = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QwenLayer::load(cfg, &layers_vb.pp(i), rank, alpha)?);
        }
        Ok(Self {
            embed,
            lm_head,
            layers,
            norm: RmsNorm::new(
                vb.pp("model.norm").get(h, "weight")?,
                cfg.rms_norm_eps as f32,
            ),
            rot: RotaryTables::new(cfg, vb.dtype(), vb.device())?,
            hidden: h,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Full-sequence logits `[batch, seq, vocab]` for `input_ids` (`[batch, seq]`,
    /// `u32`). Unlike candle's `ModelForCausalLM` (which narrows to the last
    /// position), this returns every position so the trainer can score whole
    /// completions.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails (e.g. a shape mismatch).
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.flatten_all()?;
        let mut h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?;
        let mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(l)?)
        };
        for layer in &self.layers {
            h = layer.forward(&h, mask.as_ref(), &self.rot)?;
        }
        let h = self.norm.forward(&h)?;
        match &self.lm_head {
            Some(w) => frozen_linear(&h, w),
            None => frozen_linear(&h, &self.embed),
        }
    }

    /// Enable/disable the `LoRA` adapter on every layer (disabled == the frozen
    /// base model == the GRPO reference policy).
    pub fn set_adapter_enabled(&mut self, enabled: bool) {
        for layer in &mut self.layers {
            layer.set_adapter_enabled(enabled);
        }
    }

    /// All trainable `LoRA` [`Var`]s (every layer's `q_proj`/`v_proj` `A`/`B`),
    /// in a stable order, for the optimizer and the grad-coverage canary.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        self.layers
            .iter()
            .flat_map(QwenLayer::trainable_vars)
            .collect()
    }

    /// The device the weights live on, so a caller (e.g. [`crate::QwenPolicy`])
    /// can build `input_ids` tensors on the same device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Additive causal mask `[1, 1, l, l]` (`0` on/below the diagonal, `-inf`
    /// above), matching candle's `Model::causal_mask` at offset 0.
    fn causal_mask(&self, l: usize) -> CandleResult<Tensor> {
        let mut data = Vec::with_capacity(l * l);
        for i in 0..l {
            for j in 0..l {
                data.push(if j <= i { 0f32 } else { f32::NEG_INFINITY });
            }
        }
        Tensor::from_vec(data, (1, 1, l, l), &self.device)?.to_dtype(self.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::grad_coverage;
    use candle_core::safetensors;
    use candle_transformers::models::qwen3::ModelForCausalLM;
    use std::collections::HashMap;

    fn dev() -> Device {
        Device::Cpu
    }

    /// A tiny Qwen3 config (2 layers, 2 Q / 1 KV head → GQA groups = 2, `head_dim`
    /// 4) for offline tests — same arithmetic as the 0.6B model at a runnable
    /// scale. `tie` toggles the tied vs separate `lm_head` branch.
    fn cfg_variant(tie: bool) -> Config {
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
            tie_word_embeddings: tie,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
        }
    }

    fn tiny_cfg() -> Config {
        cfg_variant(true)
    }

    /// Random weights matching `cfg`'s dotted tensor names (incl. `lm_head.weight`
    /// when untied). Norm weights are 1-D `[n]`, projections `[out, in]`.
    fn weight_map(cfg: &Config) -> HashMap<String, Tensor> {
        let d = dev();
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
        if !cfg.tie_word_embeddings {
            put("lm_head.weight", &[cfg.vocab_size, h]);
        }
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

    /// In-memory `VarBuilder` over `weight_map` (no shared temp file → no race
    /// under parallel test execution). The file-based load path is covered
    /// separately by `loads_from_buffered_safetensors`.
    fn tiny_vb(cfg: &Config) -> VarBuilder<'static> {
        VarBuilder::from_tensors(weight_map(cfg), DType::F32, &dev())
    }

    fn ids(seq: usize) -> Tensor {
        let v: Vec<u32> = (0..seq as u32).map(|i| i % 5).collect();
        Tensor::from_vec(v, (1, seq), &dev()).unwrap()
    }

    #[test]
    fn forward_produces_full_seq_logits() {
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        let logits = model.forward(&ids(5)).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
        // No NaN/inf.
        let s: f32 = logits
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(s.is_finite());
    }

    #[test]
    fn lora_grads_flow_through_qwen_backward() {
        // The whole grad path carries gradient to the LoRA A/B of q_proj AND
        // v_proj. We assert PER-BRANCH: the aggregate canary alone could pass with
        // a fully dead q-path (present-but-zero, not missing) that v keeps "live".
        // The q branch specifically exercises rms_norm_slow + rope_slow + the
        // grad-bearing softmax; the v branch is the always-grad-safe net.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        // q/v A+B over 2 layers = 8 Vars; per-layer order: q_A, q_B, v_A, v_B.
        assert_eq!(vars.len(), cfg.num_hidden_layers * 4);

        let loss = model
            .forward(&ids(6))
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();

        let q_vars: Vec<Var> = vars
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 < 2)
            .map(|(_, v)| v.clone())
            .collect();
        let v_vars: Vec<Var> = vars
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 >= 2)
            .map(|(_, v)| v.clone())
            .collect();
        let qc = grad_coverage(&q_vars, &grads).unwrap();
        let vc = grad_coverage(&v_vars, &grads).unwrap();
        assert!(
            qc.is_covered() && qc.is_live() && qc.nonfinite == 0,
            "q-branch LoRA grads not live (rms_norm_slow/rope_slow/softmax cut?): {qc:?}"
        );
        assert!(
            vc.is_covered() && vc.is_live() && vc.nonfinite == 0,
            "v-branch LoRA grads not live: {vc:?}"
        );
    }

    #[test]
    fn adapter_toggle_is_noop_at_zero_b_init() {
        // With the standard zero-B init the adapter is a no-op, so enabled == disabled.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        let input = ids(4);
        model.set_adapter_enabled(true);
        let on = model.forward(&input).unwrap();
        model.set_adapter_enabled(false);
        let off = model.forward(&input).unwrap();
        let diff: f32 = on
            .broadcast_sub(&off)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            diff <= 1e-6,
            "zero-B adapter should be a no-op, diff={diff}"
        );
        assert_eq!(on.dims(), off.dims());
    }

    #[test]
    fn rope_slow_equals_fused_rope() {
        // rope_slow is the grad-bearing twin of the fused (no-backward) rope.
        let (b, hh, seq, d) = (1usize, 2usize, 5usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, hh, seq, d), &dev()).unwrap();
        let half = d / 2;
        let cos = Tensor::randn(0f32, 1f32, (seq, half), &dev()).unwrap();
        let sin = Tensor::randn(0f32, 1f32, (seq, half), &dev()).unwrap();
        let slow = rope_slow(&q, &cos, &sin).unwrap();
        let fused = candle_nn::rotary_emb::rope(&q.contiguous().unwrap(), &cos, &sin).unwrap();
        let md: f32 = slow
            .broadcast_sub(&fused)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(md <= 1e-5, "rope_slow diverged from rope: {md}");
    }

    #[test]
    fn grad_softmax_equals_softmax_last_dim() {
        let s = Tensor::randn(0f32, 1f32, (1, 2, 5, 5), &dev()).unwrap();
        let grad = softmax(&s, D::Minus1).unwrap();
        let fused = candle_nn::ops::softmax_last_dim(&s).unwrap();
        let md: f32 = grad
            .broadcast_sub(&fused)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(md <= 1e-6, "softmax diverged from softmax_last_dim: {md}");
    }

    #[test]
    fn rms_norm_slow_equals_fused_rms_norm() {
        let x = Tensor::randn(0f32, 1f32, (3, 8), &dev()).unwrap();
        let gamma = Tensor::ones(8, DType::F32, &dev()).unwrap();
        let slow = RmsNorm::new(gamma.clone(), 1e-6).forward(&x).unwrap();
        let fused = candle_nn::ops::rms_norm(&x.contiguous().unwrap(), &gamma, 1e-6).unwrap();
        let md: f32 = slow
            .broadcast_sub(&fused)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(md <= 1e-5, "rms_norm_slow diverged from rms_norm: {md}");
    }

    /// Load our forward AND candle's shipped `ModelForCausalLM` from the SAME
    /// weights and assert the last-position logits match (the shipped forward
    /// narrows to the last position; ours returns all positions).
    fn assert_matches_shipped(cfg: &Config, seq: usize) {
        let vb = tiny_vb(cfg);
        let mut shipped = ModelForCausalLM::new(cfg, vb.clone()).unwrap();
        let mut ours = QwenGradModel::load(cfg, &vb, 2, 4.0).unwrap();
        ours.set_adapter_enabled(false); // base only, for a like-for-like compare
        let input = ids(seq);
        shipped.clear_kv_cache();
        let shipped_last = shipped.forward(&input, 0).unwrap();
        let ours_last = ours.forward(&input).unwrap().narrow(1, seq - 1, 1).unwrap();
        let md: f32 = shipped_last
            .broadcast_sub(&ours_last)
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
            md <= 1e-3,
            "custom forward diverged from shipped: max-abs={md}"
        );
    }

    #[test]
    fn custom_forward_matches_shipped_tied_gqa() {
        // tied lm_head + GQA (2 Q / 1 KV head) + causal mask (seq > 1).
        assert_matches_shipped(&tiny_cfg(), 5);
    }

    #[test]
    fn custom_forward_matches_shipped_untied() {
        // separate lm_head.weight branch.
        assert_matches_shipped(&cfg_variant(false), 5);
    }

    #[test]
    fn custom_forward_matches_shipped_single_token() {
        // seq == 1 exercises the mask == None branch.
        assert_matches_shipped(&tiny_cfg(), 1);
    }

    /// Stronger than the last-position gate: for EVERY position `t`, our
    /// full-sequence logits at `t` must match the shipped model's last-position
    /// logits on the prefix `[0..=t]`. The last-position-only gate cannot catch a
    /// causal-mask bug in a non-final row or a full-seq indexing error — but GRPO
    /// scores per-token log-probs across the WHOLE completion, so every position
    /// must be parity-correct.
    fn assert_matches_shipped_all_positions(cfg: &Config, seq: usize) {
        let vb = tiny_vb(cfg);
        let mut shipped = ModelForCausalLM::new(cfg, vb.clone()).unwrap();
        let mut ours = QwenGradModel::load(cfg, &vb, 2, 4.0).unwrap();
        ours.set_adapter_enabled(false); // base only, for a like-for-like compare
        let input = ids(seq);
        let ours_all = ours.forward(&input).unwrap(); // [1, seq, vocab]
        for t in 0..seq {
            let prefix = input.narrow(1, 0, t + 1).unwrap();
            shipped.clear_kv_cache();
            let shipped_t = shipped.forward(&prefix, 0).unwrap(); // [1, 1, vocab] @ pos t
            let ours_t = ours_all.narrow(1, t, 1).unwrap();
            let md: f32 = shipped_t
                .broadcast_sub(&ours_t)
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
                md <= 1e-3,
                "full-seq logits at position {t} diverged from shipped: max-abs={md}"
            );
        }
    }

    #[test]
    fn custom_forward_matches_shipped_every_position() {
        // tied + GQA, and the untied lm_head branch — both at every position.
        assert_matches_shipped_all_positions(&tiny_cfg(), 5);
        assert_matches_shipped_all_positions(&cfg_variant(false), 4);
    }

    #[test]
    fn load_rejects_attention_bias() {
        // A valid Qwen3 Config we don't implement must fail loud, not load a
        // silently non-parity (bias-free) model.
        let mut cfg = tiny_cfg();
        cfg.attention_bias = true;
        let vb = tiny_vb(&cfg);
        let err = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string().contains("attention_bias"),
            "expected an attention_bias rejection, got: {err}"
        );
    }

    #[test]
    fn load_rejects_sliding_window() {
        let mut cfg = tiny_cfg();
        cfg.use_sliding_window = true;
        let vb = tiny_vb(&cfg);
        let err = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string().contains("sliding_window"),
            "expected a sliding-window rejection, got: {err}"
        );
    }

    #[test]
    fn adapter_toggle_changes_output_with_trained_b() {
        // Force every LoRA B nonzero (B vars are the odd indices: q_B, v_B per
        // layer) so the adapter is no longer a no-op, then assert enabling it
        // changes the output — proving set_adapter_enabled fans out to every
        // layer's q_proj AND v_proj.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        for (i, v) in model.trainable_vars().iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                v.set(&Tensor::randn(0f32, 1f32, dims, &dev()).unwrap())
                    .unwrap();
            }
        }
        let input = ids(4);
        model.set_adapter_enabled(true);
        let on = model.forward(&input).unwrap();
        model.set_adapter_enabled(false);
        let off = model.forward(&input).unwrap();
        let diff: f32 = on
            .broadcast_sub(&off)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            diff > 1e-4,
            "enabling a nonzero-B adapter must change output, diff={diff}"
        );
    }

    #[test]
    fn loads_from_buffered_safetensors() {
        // Cover the real load path (from_buffered_safetensors) once, with a
        // process-unique file name (no shared-temp race).
        let cfg = tiny_cfg();
        let map = weight_map(&cfg);
        let path = std::env::temp_dir().join(format!(
            "ferrl-qwen-load-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        safetensors::save(&map, &path).unwrap();
        let buf = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let vb = VarBuilder::from_buffered_safetensors(buf, DType::F32, &dev()).unwrap();
        let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        assert_eq!(
            model.forward(&ids(3)).unwrap().dims(),
            &[1, 3, cfg.vocab_size]
        );
    }

    #[test]
    fn rope_slow_is_grad_bearing_fused_is_not() {
        // The grad-safe twin carries gradient; the fused custom op severs it.
        let (b, hh, seq, d) = (1usize, 2usize, 4usize, 4usize);
        let half = d / 2;
        let cos = Tensor::randn(0f32, 1f32, (seq, half), &dev()).unwrap();
        let sin = Tensor::randn(0f32, 1f32, (seq, half), &dev()).unwrap();

        let v =
            Var::from_tensor(&Tensor::randn(0f32, 1f32, (b, hh, seq, d), &dev()).unwrap()).unwrap();
        let loss = rope_slow(v.as_tensor(), &cos, &sin)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let g = loss.backward().unwrap();
        assert!(
            g.get(v.as_tensor()).is_some(),
            "rope_slow must carry gradient"
        );

        let v2 =
            Var::from_tensor(&Tensor::randn(0f32, 1f32, (b, hh, seq, d), &dev()).unwrap()).unwrap();
        let loss2 = candle_nn::rotary_emb::rope(v2.as_tensor(), &cos, &sin)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let g2 = loss2.backward().unwrap();
        assert!(
            g2.get(v2.as_tensor()).is_none(),
            "fused rope must sever gradient"
        );
    }

    #[test]
    fn grad_softmax_is_grad_bearing_fused_is_not() {
        let v =
            Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 2, 4, 4), &dev()).unwrap()).unwrap();
        let loss = softmax(v.as_tensor(), D::Minus1)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let g = loss.backward().unwrap();
        assert!(
            g.get(v.as_tensor()).is_some(),
            "softmax must carry gradient"
        );

        let v2 =
            Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 2, 4, 4), &dev()).unwrap()).unwrap();
        let loss2 = candle_nn::ops::softmax_last_dim(v2.as_tensor())
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let g2 = loss2.backward().unwrap();
        assert!(
            g2.get(v2.as_tensor()).is_none(),
            "softmax_last_dim must sever gradient"
        );
    }
}
