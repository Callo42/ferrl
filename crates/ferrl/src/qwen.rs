//! A grad-bearing, uncached Qwen3 forward — our half of the two-forward split.
//!
//! candle-transformers ships a Qwen3 forward, but it is inference-shaped
//! (`&mut self` + `ConcatKvCache`, all layer types `pub(crate)`) and built from
//! ops that have **no backward**, so it cannot be used to train. This module is the
//! *update* path: a full-sequence, uncached forward over the **same loaded
//! weights**, expressed entirely in grad-bearing ops, with a manual `LoRA` adapter
//! attached per a [`DenseLoraTargets`] recipe (the historical `load()` default is
//! q/v-only; see [`QwenGradModel::load_with_targets`] for the industrial
//! every-projection recipe). Rollout is handled by the KV-cached [`MergedDecoder`]
//! below (the production `generate` path since P6-C): candle's shipped cached
//! forward carries no adapter, so the merged-weight snapshot — the live adapter
//! folded into the base per `generate` call — is what makes a fast *adapter-aware*
//! rollout possible. The uncached forward here remains the only grad-bearing
//! (scoring / KL-reference) forward, and serves as the cached path's test oracle.
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

use std::cell::RefCell;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::ops::softmax;
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen3::Config;

use crate::blocks::{
    causal_mask, causal_mask_at, frozen_linear, repeat_kv, windowed, RotaryTables,
};
use crate::lora::{DenseLoraTargets, Proj};
use crate::model::{CachedDecoder, GradModel};
use crate::nn::RmsNorm;
use crate::remat::{stitched_backward, RematTape};

/// One Qwen3 attention block; each projection carries the `LoRA` adapter or
/// stays frozen per the [`DenseLoraTargets`] recipe. Replicates candle's
/// `Qwen3Attention::forward` with the three grad-safe substitutions and no KV
/// cache.
#[derive(Debug)]
struct QwenAttention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
}

impl QwenAttention {
    fn load(
        cfg: &Config,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
        let eps = cfg.rms_norm_eps as f32;
        Ok(Self {
            q_proj: Proj::load(
                vb,
                "q_proj",
                (q_out, h),
                targets.attn_q,
                rank,
                alpha,
                adapter_dtype,
            )?,
            k_proj: Proj::load(
                vb,
                "k_proj",
                (kv_out, h),
                targets.attn_k,
                rank,
                alpha,
                adapter_dtype,
            )?,
            v_proj: Proj::load(
                vb,
                "v_proj",
                (kv_out, h),
                targets.attn_v,
                rank,
                alpha,
                adapter_dtype,
            )?,
            o_proj: Proj::load(
                vb,
                "o_proj",
                (h, q_out),
                targets.attn_o,
                rank,
                alpha,
                adapter_dtype,
            )?,
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

        // 1. Projections, each adapted or frozen per the recipe.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
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
        self.o_proj.forward(&ctx)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.q_proj.set_enabled(enabled);
        self.k_proj.set_enabled(enabled);
        self.v_proj.set_enabled(enabled);
        self.o_proj.set_enabled(enabled);
    }

    /// Var order within the layer: `q_proj, k_proj, v_proj, o_proj` (adapted
    /// ones only).
    fn push_vars(&self, out: &mut Vec<Var>) {
        self.q_proj.push_vars(out);
        self.k_proj.push_vars(out);
        self.v_proj.push_vars(out);
        self.o_proj.push_vars(out);
    }
}

/// `SwiGLU` MLP; each projection may carry the adapter per the
/// [`DenseLoraTargets`] recipe.
#[derive(Debug)]
struct QwenMlp {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
    act: Activation,
}

impl QwenMlp {
    fn load(
        cfg: &Config,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: Proj::load(
                vb,
                "gate_proj",
                (i, h),
                targets.mlp_gate,
                rank,
                alpha,
                adapter_dtype,
            )?,
            up_proj: Proj::load(
                vb,
                "up_proj",
                (i, h),
                targets.mlp_up,
                rank,
                alpha,
                adapter_dtype,
            )?,
            down_proj: Proj::load(
                vb,
                "down_proj",
                (h, i),
                targets.mlp_down,
                rank,
                alpha,
                adapter_dtype,
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&lhs.broadcast_mul(&rhs)?)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.gate_proj.set_enabled(enabled);
        self.up_proj.set_enabled(enabled);
        self.down_proj.set_enabled(enabled);
    }

    /// Var order: `gate_proj, up_proj, down_proj` (adapted ones only).
    fn push_vars(&self, out: &mut Vec<Var>) {
        self.gate_proj.push_vars(out);
        self.up_proj.push_vars(out);
        self.down_proj.push_vars(out);
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
    fn load(
        cfg: &Config,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        let eps = cfg.rms_norm_eps as f32;
        let h = cfg.hidden_size;
        Ok(Self {
            ln1: RmsNorm::new(vb.pp("input_layernorm").get(h, "weight")?, eps),
            attn: QwenAttention::load(
                cfg,
                &vb.pp("self_attn"),
                targets,
                rank,
                alpha,
                adapter_dtype,
            )?,
            ln2: RmsNorm::new(vb.pp("post_attention_layernorm").get(h, "weight")?, eps),
            mlp: QwenMlp::load(cfg, &vb.pp("mlp"), targets, rank, alpha, adapter_dtype)?,
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
        self.mlp.set_adapter_enabled(enabled);
    }

    /// Var order within the layer: the attention projections first, then the
    /// MLP's.
    fn push_vars(&self, out: &mut Vec<Var>) {
        self.attn.push_vars(out);
        self.mlp.push_vars(out);
    }
}

/// A grad-bearing, uncached Qwen3 forward with `LoRA` attached per a
/// [`DenseLoraTargets`] recipe.
///
/// Built from the same [`VarBuilder`] (over the same safetensors) as candle's
/// shipped `ModelForCausalLM`, so the two are weight-identical and their logits
/// match (the P3 equivalence gate). The base weights are frozen [`Tensor`]s; only
/// the `LoRA` `A`/`B` factors are trainable [`Var`]s, in a deterministic
/// layer-major order (the positional checkpoint contract).
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
    targets: DenseLoraTargets,
    adapter_enabled: bool,
    remat: bool,
    /// The boundary tape of the most recent checkpointed [`forward`]
    /// (`RefCell`: `forward` takes `&self` by trait contract). One tape per
    /// forward, consumed by exactly one [`backward`]; note the interior
    /// mutability makes the model `!Sync` — nothing in ferrl shares a model
    /// across threads.
    ///
    /// [`forward`]: Self::forward
    /// [`backward`]: Self::backward
    tape: RefCell<Option<RematTape>>,
}

impl QwenGradModel {
    /// Load the model from `vb`, attaching a `LoRA` adapter of the given `rank`
    /// and `alpha` with the **historical q/v-only recipe**
    /// ([`DenseLoraTargets::legacy`]) — kept (rather than the industrial
    /// default) so pre-recipe adapter checkpoints stay positionally loadable
    /// through this constructor. Use
    /// [`load_with_targets`](Self::load_with_targets) for the industrial
    /// recipe.
    ///
    /// `vb` must be over the Qwen3 safetensors (any dtype; F32 for the CPU
    /// equivalence gate). `cfg` is candle's own `qwen3::Config` so derived dims
    /// match the shipped model exactly. Only the bias-free, non-sliding-window
    /// configuration (as used by Qwen3-0.6B-Base) is supported; other configs are
    /// rejected (see Errors) rather than loaded as a silently non-parity model.
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets).
    pub fn load(cfg: &Config, vb: &VarBuilder, rank: usize, alpha: f64) -> CandleResult<Self> {
        // The adapter shares the base weights' dtype (the toy / all-F32 case).
        Self::load_with_targets(cfg, vb, rank, alpha, vb.dtype(), DenseLoraTargets::legacy())
    }

    /// Like [`load`](Self::load) (the historical q/v-only recipe), but holds
    /// the trainable `LoRA` adapter in `adapter_dtype`, independent of the
    /// (frozen) base weights' dtype.
    ///
    /// This is the **bf16-base / F32-adapter** split: load `vb` in BF16 (halving the
    /// base weights *and* the retained activations that dominate the GRPO grad
    /// forward's memory) while keeping the adapter — and so its gradients and the
    /// `AdamW` moments — in F32, where a small update cannot collapse. See
    /// [`crate::lora::LoraLinear::with_adapter_dtype`].
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets).
    pub fn load_with_adapter_dtype(
        cfg: &Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        Self::load_with_targets(
            cfg,
            vb,
            rank,
            alpha,
            adapter_dtype,
            DenseLoraTargets::legacy(),
        )
    }

    /// Load the model from `vb`, attaching the `LoRA` adapter per `targets`
    /// (see [`DenseLoraTargets`]; `DenseLoraTargets::default()` is the
    /// industrial every-projection recipe).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `cfg` requests an unsupported option
    /// (`attention_bias`, `use_sliding_window`), if `targets` selects nothing
    /// (an untrainable model), if a weight tensor is missing or mis-shaped, or
    /// if the `LoRA` factors cannot be allocated.
    pub fn load_with_targets(
        cfg: &Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        targets: DenseLoraTargets,
    ) -> CandleResult<Self> {
        if !targets.any() {
            candle_core::bail!(
                "QwenGradModel: DenseLoraTargets selects no projection — the model would \
                 have no trainable parameters"
            );
        }
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
            layers.push(QwenLayer::load(
                cfg,
                &layers_vb.pp(i),
                targets,
                rank,
                alpha,
                adapter_dtype,
            )?);
        }
        Ok(Self {
            embed,
            lm_head,
            layers,
            norm: RmsNorm::new(
                vb.pp("model.norm").get(h, "weight")?,
                cfg.rms_norm_eps as f32,
            ),
            // Plain scalars, not the Config: the tables are architecture-neutral
            // (crate::blocks) and the dtype cast matches the shipped rotary embedding.
            rot: RotaryTables::new(
                cfg.head_dim,
                cfg.rope_theta,
                cfg.max_position_embeddings,
                vb.dtype(),
                vb.device(),
            )?,
            hidden: h,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            targets,
            adapter_enabled: true,
            remat: false,
            tape: RefCell::new(None),
        })
    }

    /// The [`DenseLoraTargets`] recipe this model was loaded with (for logging
    /// and checkpoint metadata — see [`DenseLoraTargets::canonical`]).
    #[must_use]
    pub fn lora_targets(&self) -> DenseLoraTargets {
        self.targets
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
        self.forward_window(input_ids, None)
    }

    /// The narrowed scoring forward: the full layer walk (attention needs the
    /// whole prefix), with the final norm + head applied to the
    /// `(start, len)` window alone — the full-width logits never materialize
    /// (see [`GradModel::forward_narrowed`]).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence or any
    /// tensor op fails.
    pub fn forward_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        self.forward_window(input_ids, Some((start, len)))
    }

    /// The shared tape-bearing walk behind [`forward`](Self::forward) and
    /// [`forward_narrowed`](Self::forward_narrowed).
    fn forward_window(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        if self.remat {
            return self.forward_remat(input_ids, window);
        }
        let (mut h, mask) = self.embed_and_mask(input_ids)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask.as_ref(), &self.rot)?;
        }
        self.norm_and_head(&h, window)
    }

    /// Shared prologue of every full-sequence walk: the token embedding plus
    /// the full causal mask (`None` at seq-len 1).
    fn embed_and_mask(&self, input_ids: &Tensor) -> CandleResult<(Tensor, Option<Tensor>)> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.flatten_all()?;
        let h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?;
        let mask = if l == 1 {
            None
        } else {
            Some(causal_mask(l, self.dtype, &self.device)?)
        };
        Ok((h, mask))
    }

    /// Shared tail of every walk: narrow to the `window` FIRST (this is the
    /// memory lever — the head must only ever see the window), then the final
    /// norm plus the (possibly tied) `lm_head` projection. The narrow lives
    /// inside this seam deliberately, so no caller can reorder it after the
    /// head and silently rematerialize full-width logits.
    fn norm_and_head(&self, h: &Tensor, window: Option<(usize, usize)>) -> CandleResult<Tensor> {
        let h = self.norm.forward(&windowed(h, window)?)?;
        match &self.lm_head {
            Some(w) => frozen_linear(&h, w),
            None => frozen_linear(&h, &self.embed),
        }
    }

    /// The checkpointed forward: capture a boundary [`Var`] before each layer
    /// (and one for the tail), so reassigning the hidden state frees each
    /// layer's intermediates as the walk proceeds and the loss tape spans only
    /// the tail. The boundaries are saved as the model's pending tape;
    /// [`backward`](Self::backward) stitches the full gradient from them.
    ///
    /// A `window` narrows the tail only — the narrow rides the LOSS tape (the
    /// tail boundary var stays full-width), so the stitched backward's tail
    /// gradient arrives pre-scattered to full width through the narrow
    /// adjoint, and the segment re-runs are untouched.
    fn forward_remat(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        let (mut h, mask) = self.embed_and_mask(input_ids)?;
        let mut tape = RematTape::new(self.adapter_enabled);
        for layer in &self.layers {
            let x = tape.capture(&h)?;
            h = layer.forward(&x, mask.as_ref(), &self.rot)?;
        }
        let x = tape.capture(&h)?;
        *self.tape.borrow_mut() = Some(tape);
        self.norm_and_head(&x, window)
    }

    /// Full-sequence logits like [`forward`](Self::forward) but **detached**,
    /// walking the stack with a rolling boundary detach: each layer's
    /// intermediates are freed as soon as the next layer has consumed its
    /// output, so the peak footprint is one layer plus one boundary —
    /// regardless of the checkpointing mode, and without capturing a tape.
    /// Same values as `forward` (detaching is the identity on values); for the
    /// value-only scorings (`logp_old`, the KL reference).
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails.
    pub fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_detached_window(input_ids, None)
    }

    /// The narrowed detached scoring forward: rolling boundary detach plus the
    /// windowed tail (see [`GradModel::forward_detached_narrowed`]).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence or any
    /// tensor op fails.
    pub fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        self.forward_detached_window(input_ids, Some((start, len)))
    }

    /// The shared detached walk behind
    /// [`forward_detached`](Self::forward_detached) and
    /// [`forward_detached_narrowed`](Self::forward_detached_narrowed).
    fn forward_detached_window(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        let (mut h, mask) = self.embed_and_mask(input_ids)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask.as_ref(), &self.rot)?.detach();
        }
        Ok(self.norm_and_head(&h, window)?.detach())
    }

    /// Back-propagate a loss built from this model's logits: plain
    /// `loss.backward()` normally; under
    /// [activation checkpointing](Self::set_activation_checkpointing) it takes
    /// the pending boundary tape and stitches the full gradient by re-running
    /// each layer in reverse (see [`crate::remat`]). Fails loud if no
    /// checkpointed forward is pending, if the loss does not pair with the
    /// pending tape, or if the adapter toggle flipped since the forward (the
    /// recompute would silently rebuild different values).
    ///
    /// # Errors
    ///
    /// Returns a candle error on any backward failure or contract violation
    /// above.
    pub fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        if !self.remat {
            return loss.backward();
        }
        let Some(tape) = self.tape.borrow_mut().take() else {
            candle_core::bail!(
                "QwenGradModel::backward: activation checkpointing is on but no checkpointed \
                 forward is pending (each forward's tape is consumed by exactly one backward)"
            )
        };
        if tape.adapter_enabled() != self.adapter_enabled {
            candle_core::bail!(
                "QwenGradModel::backward: the adapter toggle flipped between the checkpointed \
                 forward and its backward — the recompute would rebuild different values"
            )
        }
        let l = tape.first_boundary_dims().map(|d| d[1]).unwrap_or_default();
        let mask = if l <= 1 {
            None
        } else {
            Some(causal_mask(l, self.dtype, &self.device)?)
        };
        stitched_backward(loss, &tape, &self.trainable_vars(), |i, x| {
            self.layers[i].forward(x, mask.as_ref(), &self.rot)
        })
    }

    /// Turn **activation checkpointing** on or off (default: off).
    ///
    /// On, [`forward`](Self::forward) cuts the autograd graph at every layer
    /// boundary and [`backward`](Self::backward) re-runs one layer at a time —
    /// the peak activation footprint drops from the whole stack to one layer
    /// plus the boundary states, for ~one extra forward of recompute (the
    /// primary single-card memory lever; see [`crate::remat`] for the
    /// contract). Off (the default), both are the plain uncut paths,
    /// bit-identical to the pre-P7 behavior. Flipping the mode drops any
    /// pending tape.
    pub fn set_activation_checkpointing(&mut self, on: bool) {
        self.remat = on;
        if !on {
            *self.tape.borrow_mut() = None;
        }
    }

    /// Whether activation checkpointing is currently on.
    #[must_use]
    pub fn activation_checkpointing(&self) -> bool {
        self.remat
    }

    /// Enable/disable the `LoRA` adapter on every layer (disabled == the frozen
    /// base model == the GRPO reference policy).
    pub fn set_adapter_enabled(&mut self, enabled: bool) {
        for layer in &mut self.layers {
            layer.set_adapter_enabled(enabled);
        }
        self.adapter_enabled = enabled;
    }

    /// All trainable `LoRA` [`Var`]s in a **deterministic** order — layer-major;
    /// within a layer the attention projections first (`q,k,v,o`), then the
    /// MLP's (`gate,up,down`); each adapted projection contributes `[A, B]`.
    /// The order is a pure function of (config, [`DenseLoraTargets`]) — the
    /// positional checkpoint contract.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        for layer in &self.layers {
            layer.push_vars(&mut vars);
        }
        vars
    }

    /// The device the weights live on, so a caller (e.g. [`crate::QwenPolicy`])
    /// can build `input_ids` tensors on the same device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Snapshot the **current** effective weights into a KV-cached, grad-free
    /// [`MergedDecoder`] for fast incremental rollout.
    ///
    /// This is the build half of the cached-rollout optimization. It folds the
    /// live `LoRA` adapter into every adapted projection via
    /// [`crate::lora::LoraLinear::merged_weight`] (respecting the adapter toggle,
    /// so a disabled adapter snapshots the pure base model), clones the frozen
    /// rest, and hands back a decoder that walks the sequence one chunk at a time
    /// over candle's `ConcatKvCache` — O(L) per token instead of the uncached
    /// forward's O(L²).
    ///
    /// **Rebuild after every optimizer step** (and after any `set_adapter_enabled`
    /// flip): the returned decoder is a value snapshot, frozen at the `Var` values
    /// it read. The grad/scoring path ([`forward`](Self::forward)) is untouched and
    /// must keep being used for training — the merged weights are tape-detached.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any `merged_weight` build fails.
    pub fn merged_decoder(&self) -> CandleResult<MergedDecoder> {
        MergedDecoder::from_model(self)
    }
}

/// The [`GradModel`] seam over [`QwenGradModel`]: pure delegation to the
/// inherent methods above (which stay public — the trait adds a generic
/// surface, it does not replace the concrete one).
impl GradModel for QwenGradModel {
    type Decoder = MergedDecoder;

    fn device(&self) -> &Device {
        QwenGradModel::device(self)
    }

    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        QwenGradModel::forward(self, input_ids)
    }

    fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        QwenGradModel::forward_detached(self, input_ids)
    }

    fn forward_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        QwenGradModel::forward_narrowed(self, input_ids, start, len)
    }

    fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        QwenGradModel::forward_detached_narrowed(self, input_ids, start, len)
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        QwenGradModel::backward(self, loss)
    }

    fn trainable_vars(&self) -> Vec<Var> {
        QwenGradModel::trainable_vars(self)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        QwenGradModel::set_adapter_enabled(self, enabled);
    }

    fn merged_decoder(&self) -> CandleResult<MergedDecoder> {
        QwenGradModel::merged_decoder(self)
    }

    fn lora_recipe(&self) -> Option<String> {
        Some(self.targets.canonical())
    }
}

/// One Qwen3 attention block over **merged** weights with an incremental KV cache.
///
/// The grad-free mirror of [`QwenAttention`]: every projection uses its single
/// effective weight (the folded [`crate::lora::LoraLinear::merged_weight`] when
/// adapted — no `LoRA` side-path; the frozen base weight otherwise; all
/// bias-free — the supported Qwen3 configs have no projection biases), and the
/// un-repeated K/V are appended to a [`ConcatKvCache`] before `repeat_kv` — the
/// exact op order of candle's shipped `Qwen3Attention` (project →
/// reshape/transpose → per-head q/k-norm → `RoPE(offset)` → `cache.append` →
/// `repeat_kv` → SDPA → `o_proj`), with the same grad-safe op twins
/// [`QwenAttention`] uses so the cached logits equal the uncached ones.
#[derive(Debug)]
struct MergedAttention {
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
    /// Un-repeated K/V cache (`[b, kv_heads, seq, head_dim]`), concatenated on the
    /// sequence axis (dim 2). `repeat_kv` is applied to the cache's output, never
    /// to what is stored — storing the repeated KV would inflate the cache by
    /// `num_kv_groups`.
    cache: ConcatKvCache,
}

impl MergedAttention {
    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Projections over merged weights (adapted ones carry the folded
        //    adapter; the rest are the frozen base).
        let q = frozen_linear(x, &self.q_weight)?;
        let k = frozen_linear(x, &self.k_weight)?;
        let v = frozen_linear(x, &self.v_weight)?;

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

        // 4. RoPE at the absolute position `offset` (grad-safe rope_slow).
        let (cos, sin) = rot.slice_at(offset, l)?;
        let q = rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k = rope_slow(&k.contiguous()?, &cos, &sin)?;

        // 5. Append the UN-repeated K/V, then GQA-repeat the full cached K/V —
        //    repeat AFTER append (the shipped order) so the cache stays compact.
        let (k, v) = self.cache.append(&k, &v)?;
        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

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
}

/// `SwiGLU` MLP over merged weights — the grad-free mirror of [`QwenMlp`].
#[derive(Debug)]
struct MergedMlp {
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    act: Activation,
}

impl MergedMlp {
    fn from_layer(mlp: &QwenMlp) -> CandleResult<Self> {
        Ok(Self {
            gate_weight: mlp.gate_proj.merged_weight()?,
            up_weight: mlp.up_proj.merged_weight()?,
            down_weight: mlp.down_proj.merged_weight()?,
            act: mlp.act,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = frozen_linear(x, &self.gate_weight)?.apply(&self.act)?;
        let rhs = frozen_linear(x, &self.up_weight)?;
        frozen_linear(&lhs.broadcast_mul(&rhs)?, &self.down_weight)
    }
}

/// One decoder layer over merged weights: pre-norm cached attention + pre-norm
/// merged `SwiGLU`, both residual. The grad-free mirror of [`QwenLayer`].
#[derive(Debug)]
struct MergedLayer {
    ln1: RmsNorm,
    attn: MergedAttention,
    ln2: RmsNorm,
    mlp: MergedMlp,
}

impl MergedLayer {
    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h, offset, mask, rot)?;
        let x = x.broadcast_add(&h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x.broadcast_add(&h2)
    }
}

/// A KV-cached, **grad-free** Qwen3 decoder over weights with the `LoRA` adapter
/// already folded in — the fast rollout twin of [`QwenGradModel`].
///
/// Built by [`QwenGradModel::merged_decoder`], which snapshots the live merged
/// weights (so it captures whatever the adapter is at build time, toggle included).
/// [`forward`](Self::forward) consumes one chunk of new tokens at a time, advancing
/// a per-layer [`ConcatKvCache`], so generating `L` tokens costs O(L) attention work
/// instead of the uncached forward's O(L²). It holds **no** `Var` and records no
/// autograd tape — it is for inference/rollout only; training still goes through
/// [`QwenGradModel::forward`].
///
/// Faithfulness is CI-gated: cached logits equal the uncached
/// [`QwenGradModel::forward`] logits position-by-position at F32 (adapter on), and
/// the adapter-off snapshot equals candle's shipped cached forward at every position.
///
/// # Cache lifecycle
///
/// The cache grows with each [`forward`](Self::forward); positions are placed at the
/// `offset` you pass (which must equal the number of tokens already consumed). Call
/// [`reset_cache`](Self::reset_cache) to reuse one decoder for a fresh sequence, or
/// build a new decoder. Because the cache is mutable state, `forward` takes `&mut self`.
#[derive(Debug)]
pub struct MergedDecoder {
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<MergedLayer>,
    norm: RmsNorm,
    rot: RotaryTables,
    hidden: usize,
    device: Device,
    dtype: DType,
}

impl MergedDecoder {
    /// Snapshot a [`QwenGradModel`]'s current effective weights. Private — callers
    /// go through [`QwenGradModel::merged_decoder`].
    fn from_model(model: &QwenGradModel) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let a = &layer.attn;
            layers.push(MergedLayer {
                ln1: layer.ln1.clone(),
                attn: MergedAttention {
                    q_weight: a.q_proj.merged_weight()?,
                    k_weight: a.k_proj.merged_weight()?,
                    v_weight: a.v_proj.merged_weight()?,
                    o_weight: a.o_proj.merged_weight()?,
                    q_norm: a.q_norm.clone(),
                    k_norm: a.k_norm.clone(),
                    num_heads: a.num_heads,
                    num_kv_heads: a.num_kv_heads,
                    num_kv_groups: a.num_kv_groups,
                    head_dim: a.head_dim,
                    attn_hidden: a.attn_hidden,
                    cache: ConcatKvCache::new(2),
                },
                ln2: layer.ln2.clone(),
                mlp: MergedMlp::from_layer(&layer.mlp)?,
            });
        }
        Ok(Self {
            embed: model.embed.clone(),
            lm_head: model.lm_head.clone(),
            layers,
            norm: model.norm.clone(),
            rot: model.rot.clone(),
            hidden: model.hidden,
            device: model.device.clone(),
            dtype: model.dtype,
        })
    }

    /// Logits `[batch, chunk_len, vocab]` for `input_ids` (`[batch, chunk_len]`,
    /// `u32`) placed at absolute positions `[offset, offset + chunk_len)`, appending
    /// to the KV cache.
    ///
    /// Pass the whole prompt at `offset == 0` to prefill, then one token at a time
    /// at the running offset to decode. `offset` **must** equal the number of tokens
    /// already in the cache (it indexes the `RoPE` tables and sizes the causal mask);
    /// a mismatch is rejected (see Errors) rather than silently producing wrong
    /// logits. Like [`QwenGradModel::forward`], every position is returned (the
    /// caller narrows to the last for sampling).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `offset` does not equal the cached sequence length,
    /// if any tensor op fails (e.g. a shape mismatch), or if `offset + chunk_len`
    /// exceeds the `RoPE` table's `max_position_embeddings`.
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        // The caller's `offset` must equal the number of tokens already cached: it
        // positions RoPE and sizes the causal mask, but the `l == 1` decode path
        // builds no mask, so a desync would NOT trip a shape check — it would silently
        // corrupt the logits. Fail loud instead, so an offset-bookkeeping bug (the
        // exact risk in the generation/eval loop) surfaces as an error, not as quietly
        // wrong rollout. All layer caches advance in lockstep, so layer 0 is the truth.
        let cached = self
            .layers
            .first()
            .map_or(0, |layer| layer.attn.cache.current_seq_len());
        if offset != cached {
            candle_core::bail!(
                "MergedDecoder::forward: offset {offset} != cached sequence length \
                 {cached} (pass offset == tokens already decoded; 0 to prefill)"
            );
        }
        let ids = input_ids.flatten_all()?;
        let mut h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?;
        // A single new token attends to the whole cache (all past keys are causally
        // valid), matching both the uncached `l == 1` branch and the shipped model.
        let mask = if l == 1 {
            None
        } else {
            Some(causal_mask_at(offset, l, self.dtype, &self.device)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, offset, mask.as_ref(), &self.rot)?;
        }
        let h = self.norm.forward(&h)?;
        match &self.lm_head {
            Some(w) => frozen_linear(&h, w),
            None => frozen_linear(&h, &self.embed),
        }
    }

    /// Clear every layer's KV cache so the decoder can start a fresh sequence
    /// (next [`forward`](Self::forward) must use `offset == 0`).
    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.attn.cache.reset();
        }
    }
}

/// The [`CachedDecoder`] seam over [`MergedDecoder`]: pure delegation to the
/// inherent methods above (which carry the offset fail-loud guard and the
/// cache-lifecycle contract the trait requires).
impl CachedDecoder for MergedDecoder {
    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        MergedDecoder::forward(self, input_ids, offset)
    }

    fn reset_cache(&mut self) {
        MergedDecoder::reset_cache(self);
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
    fn dtype_split_forward_and_grad() {
        // The dtype-split mechanism on a tiny model: the adapter is held in a
        // different (higher) precision than the base, the forward runs in the base
        // dtype, and the adapter's gradients land in the adapter dtype. The real
        // instance is bf16-base / F32-adapter, but candle's CPU backend has no bf16
        // matmul, so the CPU gate uses F32-base / F64-adapter (the bf16 instance is
        // exercised on the GPU by the Countdown run and the `#[ignore]`d GPU gates).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg); // F32 base
        let mut model =
            QwenGradModel::load_with_adapter_dtype(&cfg, &vb, 2, 4.0, DType::F64).unwrap();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        // Force every B (odd indices: q_B, v_B per layer) nonzero so A also carries a
        // live gradient through the backward.
        for v in vars.iter().skip(1).step_by(2) {
            let dims = v.as_tensor().dims().to_vec();
            v.set(&Tensor::ones(dims, DType::F64, &dev()).unwrap())
                .unwrap();
        }

        let logits = model.forward(&ids(5)).unwrap();
        assert_eq!(
            logits.dtype(),
            DType::F32,
            "the forward runs in the base/activation dtype"
        );
        let loss = logits.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for v in &vars {
            let g = grads
                .get(v.as_tensor())
                .expect("adapter var missing from grad store");
            assert_eq!(
                g.dtype(),
                DType::F64,
                "master adapter must receive a grad in its own dtype"
            );
        }
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

    // ---- MergedDecoder: cached-rollout equivalence gates -------------------

    /// Max absolute element-wise difference between two same-shaped tensors.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        a.broadcast_sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    /// Force every `LoRA` B factor (the odd `trainable_vars` indices — each
    /// adapted projection contributes `[A, B]`) to small random values so the
    /// adapter is a genuine perturbation, not the zero-B no-op — the merge must
    /// then differ from the base. Recipe-agnostic.
    fn arm_adapter(model: &QwenGradModel) {
        for (i, v) in model.trainable_vars().iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                v.set(&Tensor::randn(0f32, 0.5f32, dims, &dev()).unwrap())
                    .unwrap();
            }
        }
    }

    /// Uncached base-only logits over the same weights `vb`, for the non-vacuity
    /// witness (the armed adapter must move the logits away from this).
    fn uncached_base_logits(cfg: &Config, vb: &VarBuilder, input: &Tensor) -> Tensor {
        let mut m = QwenGradModel::load(cfg, vb, 2, 4.0).unwrap();
        m.set_adapter_enabled(false);
        m.forward(input).unwrap()
    }

    #[test]
    fn merged_decoder_matches_uncached_token_by_token() {
        // THE core gate: cached single-token decode == uncached full-seq forward at
        // every position, adapter ON, at F32.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(true);

        let seq = 6;
        let input = ids(seq);
        let reference = model.forward(&input).unwrap(); // adapter-aware, uncached

        // Non-vacuity: the armed adapter must actually move the logits, else a
        // no-op merge would pass this gate trivially.
        assert!(
            max_abs_diff(&reference, &uncached_base_logits(&cfg, &vb, &input)) > 1e-3,
            "armed adapter must change the logits (gate would be vacuous otherwise)"
        );

        let mut dec = model.merged_decoder().unwrap();
        let mut worst = 0f32;
        for t in 0..seq {
            let tok = input.narrow(1, t, 1).unwrap();
            let logits_t = dec.forward(&tok, t).unwrap();
            assert_eq!(logits_t.dims(), &[1, 1, cfg.vocab_size]);
            worst = worst.max(max_abs_diff(&logits_t, &reference.narrow(1, t, 1).unwrap()));
        }
        assert!(
            worst <= 1e-3,
            "cached token-by-token decode diverged from uncached forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_prefill_then_incremental_matches_uncached() {
        // The realistic generate() pattern: prefill the prompt in one chunk
        // (exercises the multi-token causal mask), then decode one token at a time
        // at the running offset (exercises offset>0 incremental decode).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(true);

        let seq = 7;
        let prompt_len = 3;
        let input = ids(seq);
        let reference = model.forward(&input).unwrap();

        let mut dec = model.merged_decoder().unwrap();
        let mut worst = 0f32;

        let prefill = input.narrow(1, 0, prompt_len).unwrap();
        let pre = dec.forward(&prefill, 0).unwrap();
        assert_eq!(pre.dims(), &[1, prompt_len, cfg.vocab_size]);
        for t in 0..prompt_len {
            worst = worst.max(max_abs_diff(
                &pre.narrow(1, t, 1).unwrap(),
                &reference.narrow(1, t, 1).unwrap(),
            ));
        }
        for t in prompt_len..seq {
            let tok = input.narrow(1, t, 1).unwrap();
            let logits_t = dec.forward(&tok, t).unwrap();
            worst = worst.max(max_abs_diff(&logits_t, &reference.narrow(1, t, 1).unwrap()));
        }
        assert!(
            worst <= 1e-3,
            "cached prefill+incremental decode diverged from uncached forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_chunked_at_offset_matches_uncached() {
        // Two MULTI-token chunks: [0..3] at offset 0, then [3..7] at offset 3. The
        // second chunk has chunk_len>1 AND offset>0, the only path that builds the
        // rectangular causal mask `[1,1,chunk_len,offset+chunk_len]` — never reached
        // by prefill (offset 0) or single-token decode (l==1 => mask None). Adapter ON.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(true);

        let seq = 7;
        let split = 3;
        let input = ids(seq);
        let reference = model.forward(&input).unwrap();

        let mut dec = model.merged_decoder().unwrap();
        let first = dec.forward(&input.narrow(1, 0, split).unwrap(), 0).unwrap();
        let second = dec
            .forward(&input.narrow(1, split, seq - split).unwrap(), split)
            .unwrap();
        assert_eq!(second.dims(), &[1, seq - split, cfg.vocab_size]);

        let mut worst = 0f32;
        for t in 0..split {
            worst = worst.max(max_abs_diff(
                &first.narrow(1, t, 1).unwrap(),
                &reference.narrow(1, t, 1).unwrap(),
            ));
        }
        for t in 0..(seq - split) {
            worst = worst.max(max_abs_diff(
                &second.narrow(1, t, 1).unwrap(),
                &reference.narrow(1, split + t, 1).unwrap(),
            ));
        }
        assert!(
            worst <= 1e-3,
            "chunked decode (multi-token chunk at offset>0) diverged from uncached: {worst}"
        );
    }

    #[test]
    fn merged_decoder_base_only_matches_shipped_every_position() {
        // The second gate: the adapter-OFF snapshot == candle's shipped cached
        // forward at every position (also proves merged_weight respects the toggle —
        // the adapter is armed but disabled, so the snapshot must be pure base).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(false);
        let mut dec = model.merged_decoder().unwrap();

        let mut shipped = ModelForCausalLM::new(&cfg, vb.clone()).unwrap();
        shipped.clear_kv_cache();

        let seq = 6;
        let input = ids(seq);
        let mut worst = 0f32;
        for t in 0..seq {
            let tok = input.narrow(1, t, 1).unwrap();
            let ours_t = dec.forward(&tok, t).unwrap();
            let shipped_t = shipped.forward(&tok, t).unwrap();
            worst = worst.max(max_abs_diff(&ours_t, &shipped_t));
        }
        assert!(
            worst <= 1e-3,
            "base-only cached decode diverged from candle's shipped cached forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_base_only_matches_uncached_base() {
        // Same grad-safe ops on both sides; the ONLY difference is incremental
        // caching, so this is a tight pin on the cache/offset/mask wiring alone,
        // independent of the slow-twin vs fused-kernel gap the shipped gate tolerates.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        model.set_adapter_enabled(false);
        let input = ids(6);
        let reference = model.forward(&input).unwrap();
        let mut dec = model.merged_decoder().unwrap();
        let mut worst = 0f32;
        for t in 0..6 {
            let tok = input.narrow(1, t, 1).unwrap();
            worst = worst.max(max_abs_diff(
                &dec.forward(&tok, t).unwrap(),
                &reference.narrow(1, t, 1).unwrap(),
            ));
        }
        assert!(
            worst <= 1e-4,
            "base-only cached decode diverged from uncached base forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_reset_cache_restarts_sequence() {
        // reset_cache() lets one decoder serve a fresh sequence; a replay after
        // reset must reproduce the reference (a leftover cache would not).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(true);
        let input = ids(5);
        let reference = model.forward(&input).unwrap();

        let mut dec = model.merged_decoder().unwrap();
        for t in 0..5 {
            dec.forward(&input.narrow(1, t, 1).unwrap(), t).unwrap();
        }
        dec.reset_cache();
        let mut worst = 0f32;
        for t in 0..5 {
            let lt = dec.forward(&input.narrow(1, t, 1).unwrap(), t).unwrap();
            worst = worst.max(max_abs_diff(&lt, &reference.narrow(1, t, 1).unwrap()));
        }
        assert!(
            worst <= 1e-3,
            "decode after reset_cache diverged from the reference: {worst}"
        );
    }

    #[test]
    fn merged_decoder_rejects_offset_mismatch() {
        // The offset MUST equal the cached length. On the l==1 decode path no mask is
        // built, so a desync (e.g. a generation-loop offset-bookkeeping bug) would
        // silently corrupt the logits rather than trip a shape error — the decoder
        // guards against it and fails loud. This is the negative control with teeth:
        // a wrong offset cannot pass quietly.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        model.set_adapter_enabled(false);
        let input = ids(5);
        let mut dec = model.merged_decoder().unwrap();

        // A non-zero offset on the first (empty-cache) call is rejected.
        let err = dec.forward(&input.narrow(1, 0, 1).unwrap(), 3).unwrap_err();
        assert!(
            err.to_string().contains("offset"),
            "first-call offset!=0 should be rejected, got: {err}"
        );

        // Prime position 0, then feed token 1 at the WRONG offset 0 (should be 1).
        dec.forward(&input.narrow(1, 0, 1).unwrap(), 0).unwrap();
        let err = dec.forward(&input.narrow(1, 1, 1).unwrap(), 0).unwrap_err();
        assert!(
            err.to_string().contains("offset"),
            "stale offset should be rejected, got: {err}"
        );

        // A rejected call must not have mutated the cache: the correct offset 1 works.
        dec.forward(&input.narrow(1, 1, 1).unwrap(), 1).unwrap();
    }

    // ---- DenseLoraTargets recipe gates -------------------------------------

    /// The trainable [`Var`]s of one projection (`[A, B]` when adapted, empty
    /// when frozen) — the building block the order pins compare against.
    fn proj_vars(p: &Proj) -> Vec<Var> {
        let mut out = Vec::new();
        p.push_vars(&mut out);
        out
    }

    #[test]
    fn load_with_targets_rejects_an_empty_recipe() {
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let none = DenseLoraTargets {
            attn_q: false,
            attn_k: false,
            attn_v: false,
            attn_o: false,
            mlp_gate: false,
            mlp_up: false,
            mlp_down: false,
        };
        let err =
            QwenGradModel::load_with_targets(&cfg, &vb, 2, 4.0, DType::F32, none).unwrap_err();
        assert!(
            err.to_string().contains("no projection"),
            "expected a no-target rejection, got: {err}"
        );
    }

    /// Assert `vars[base..base+expect.len()]` are exactly `expect`, by Var
    /// identity (shape alone could not catch a same-shape swap, e.g. k/v).
    fn assert_var_ids_match(vars: &[Var], expect: &[Var], base: usize) {
        for (j, e) in expect.iter().enumerate() {
            assert_eq!(
                vars[base + j].as_tensor().id(),
                e.as_tensor().id(),
                "var {} out of positional order",
                base + j
            );
        }
    }

    /// Every projection the legacy q/v-only recipe must NOT adapt stays frozen.
    fn assert_legacy_frozen(layer: &QwenLayer) {
        for (p, name) in [
            (&layer.attn.k_proj, "k_proj"),
            (&layer.attn.o_proj, "o_proj"),
            (&layer.mlp.gate_proj, "gate_proj"),
            (&layer.mlp.up_proj, "up_proj"),
            (&layer.mlp.down_proj, "down_proj"),
        ] {
            assert!(
                matches!(p, Proj::Frozen(_)),
                "{name} must stay frozen under the legacy recipe"
            );
        }
    }

    /// One layer's vars in the DOCUMENTED industrial order
    /// `[q,k,v,o,gate,up,down]`, built directly from the layer fields — so a
    /// swap inside any `push_vars` reddens the pin that compares against this.
    fn industrial_layer_vars(layer: &QwenLayer) -> Vec<Var> {
        let mut out = Vec::new();
        for p in [
            &layer.attn.q_proj,
            &layer.attn.k_proj,
            &layer.attn.v_proj,
            &layer.attn.o_proj,
            &layer.mlp.gate_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
        ] {
            p.push_vars(&mut out);
        }
        out
    }

    #[test]
    fn legacy_load_pins_the_qv_only_var_order_and_recipe() {
        // load() = the historical q/v-only recipe: 4 vars per layer in
        // [q_A, q_B, v_A, v_B] order — THE positional back-compat contract for
        // pre-recipe adapter checkpoints — plus frozen-ness of everything the
        // legacy recipe must not adapt.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        assert_eq!(model.lora_recipe(), Some("attn:qv|mlp:-".to_string()));
        let vars = model.trainable_vars();
        assert_eq!(vars.len(), cfg.num_hidden_layers * 4);
        for (l, layer) in model.layers.iter().enumerate() {
            let mut expect = proj_vars(&layer.attn.q_proj);
            expect.extend(proj_vars(&layer.attn.v_proj));
            assert_var_ids_match(&vars, &expect, l * 4);
            assert_legacy_frozen(layer);
        }
    }

    #[test]
    fn industrial_recipe_var_order_is_layer_major_qkvo_then_mlp() {
        // The positional checkpoint contract under the industrial recipe:
        // layer-major, [q,k,v,o,gate,up,down] within a layer, [A,B] per
        // projection — pinned by Var identity against an expectation built
        // directly from the layer fields in the documented order.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let model = QwenGradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        assert_eq!(model.lora_recipe(), Some("attn:qkvo|mlp:gud".to_string()));
        let vars = model.trainable_vars();
        assert_eq!(vars.len(), cfg.num_hidden_layers * 14);
        for (l, layer) in model.layers.iter().enumerate() {
            let expect = industrial_layer_vars(layer);
            assert_eq!(expect.len(), 14);
            assert_var_ids_match(&vars, &expect, l * 14);
        }
    }

    #[test]
    fn industrial_grads_flow_to_every_projection() {
        // Per-projection liveness under the industrial recipe: each adapted
        // projection's [A, B] pair gets a present, finite, nonzero gradient.
        // An aggregate-only canary could hide one dead projection (e.g. a
        // newly-adaptable o_proj or down_proj wired around the tape).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        // Arm every B (odd indices) so every A also carries a live gradient.
        for v in vars.iter().skip(1).step_by(2) {
            let dims = v.as_tensor().dims().to_vec();
            v.set(&Tensor::randn(0f32, 0.2f32, dims, &dev()).unwrap())
                .unwrap();
        }
        let loss = model
            .forward(&ids(6))
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        for (pair_idx, pair) in vars.chunks(2).enumerate() {
            let c = grad_coverage(pair, &grads).unwrap();
            assert!(
                c.is_covered() && c.is_live() && c.nonfinite == 0,
                "projection pair {pair_idx} has dead/missing grads: {c:?}"
            );
        }
    }

    #[test]
    fn adapter_toggle_reaches_the_mlp_projections() {
        // A recipe adapting ONLY mlp_down, armed: the toggle must change the
        // output — proving set_adapter_enabled fans out past the attention
        // block into the MLP (newly adaptable in this retrofit).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let targets = DenseLoraTargets {
            attn_q: false,
            attn_k: false,
            attn_v: false,
            attn_o: false,
            mlp_gate: false,
            mlp_up: false,
            mlp_down: true,
        };
        let mut model =
            QwenGradModel::load_with_targets(&cfg, &vb, 2, 4.0, DType::F32, targets).unwrap();
        assert_eq!(model.trainable_vars().len(), cfg.num_hidden_layers * 2);
        arm_adapter(&model);
        let input = ids(4);
        model.set_adapter_enabled(true);
        let on = model.forward(&input).unwrap();
        model.set_adapter_enabled(false);
        let off = model.forward(&input).unwrap();
        assert!(
            max_abs_diff(&on, &off) > 1e-4,
            "an armed mlp_down adapter must change the output when enabled"
        );
    }

    #[test]
    fn merged_decoder_matches_uncached_under_the_industrial_recipe() {
        // The cached-equivalence gate with EVERY projection adapted and armed:
        // pins the merged fold of k/o/gate/up/down — the projections this
        // retrofit makes adaptable for the first time.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(true);

        let seq = 6;
        let input = ids(seq);
        let reference = model.forward(&input).unwrap();
        // Non-vacuity: the armed all-projection adapter must move the logits.
        assert!(
            max_abs_diff(&reference, &uncached_base_logits(&cfg, &vb, &input)) > 1e-3,
            "armed industrial adapter must change the logits (gate would be vacuous)"
        );

        let mut dec = model.merged_decoder().unwrap();
        let mut worst = 0f32;
        for t in 0..seq {
            let tok = input.narrow(1, t, 1).unwrap();
            let logits_t = dec.forward(&tok, t).unwrap();
            worst = worst.max(max_abs_diff(&logits_t, &reference.narrow(1, t, 1).unwrap()));
        }
        assert!(
            worst <= 1e-3,
            "industrial-recipe cached decode diverged from uncached forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_disabled_industrial_snapshots_pure_base() {
        // The adapter-OFF half of the industrial gate: every projection
        // adapted and ARMED but DISABLED — the merged snapshot must be the
        // pure base model (the GRPO reference policy), proving merged_weight
        // respects the toggle on all seven folds, not just legacy q/v.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = QwenGradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(false);

        let seq = 6;
        let input = ids(seq);
        let base = uncached_base_logits(&cfg, &vb, &input);
        let mut dec = model.merged_decoder().unwrap();
        let mut worst = 0f32;
        for t in 0..seq {
            let tok = input.narrow(1, t, 1).unwrap();
            worst = worst.max(max_abs_diff(
                &dec.forward(&tok, t).unwrap(),
                &base.narrow(1, t, 1).unwrap(),
            ));
        }
        assert!(
            worst <= 1e-4,
            "disabled-industrial cached decode diverged from the pure base: {worst}"
        );
    }

    // ---- activation checkpointing (P7) --------------------------------------

    /// A fixed non-uniform probe loss over the logits, in the logits' dtype —
    /// no gradient cancels by symmetry.
    fn probe_loss(logits: &Tensor) -> Tensor {
        let n = logits.elem_count();
        let w: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.21 - 0.6).collect();
        let w = Tensor::from_vec(w, logits.dims().to_vec(), logits.device())
            .unwrap()
            .to_dtype(logits.dtype())
            .unwrap();
        logits.mul(&w).unwrap().sum_all().unwrap()
    }

    /// Every trainable var must appear in BOTH stores (canary-style) with
    /// near-identical gradients (the stitched accumulation order may differ
    /// from the uncut one by float reassociation only).
    fn assert_var_grads_close(plain: &GradStore, stitched: &GradStore, vars: &[Var]) {
        for (k, v) in vars.iter().enumerate() {
            let a = plain.get(v).expect("var missing from the uncut store");
            let b = stitched
                .get(v)
                .expect("var missing from the stitched store");
            let diff = max_abs_diff(a, b);
            let scale: f32 = a
                .abs()
                .unwrap()
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar()
                .unwrap();
            assert!(
                diff <= 1e-5 * scale.max(1.0),
                "var {k}: stitched grad diverged from the uncut backward by {diff} (scale {scale})"
            );
        }
    }

    #[test]
    fn checkpointed_gradients_match_the_uncut_backward() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let input = ids(6);
        let vars = model.trainable_vars();

        let plain = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();
        model.set_activation_checkpointing(true);
        assert!(model.activation_checkpointing());
        let stitched = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();

        assert_var_grads_close(&plain, &stitched, &vars);
        // Non-vacuity: the probe produces real gradients.
        assert!(vars.iter().any(|v| max_abs_diff(
            plain.get(v).unwrap(),
            &plain.get(v).unwrap().zeros_like().unwrap()
        ) > 1e-6));
    }

    /// The structural memory claim: a checkpointed forward CUTS the loss tape
    /// at the tail boundary, so a raw `loss.backward()` (bypassing the
    /// stitching) reaches NO layer var — which is exactly why
    /// [`QwenGradModel::backward`] must stitch, and what frees the per-layer
    /// activation graph during the forward.
    #[test]
    fn a_checkpointed_forward_actually_cuts_the_loss_tape() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_activation_checkpointing(true);
        let raw = probe_loss(&model.forward(&ids(6)).unwrap())
            .backward()
            .unwrap();
        for v in model.trainable_vars() {
            assert!(
                raw.get(&v).is_none(),
                "a layer var is on the loss tape — the boundary cut is not happening"
            );
        }
    }

    // ---- narrowed scoring forward (PR-B) -------------------------------

    /// Every adapter var must appear in BOTH stores, with gradients within
    /// `tol` of each other (`0.0` = exact).
    fn assert_grads_match(a: &GradStore, b: &GradStore, vars: &[Var], tol: f32) {
        for (k, v) in vars.iter().enumerate() {
            let ga = a.get(v).expect("var missing from the first store");
            let gb = b.get(v).expect("var missing from the second store");
            let d = max_abs_diff(ga, gb);
            assert!(d <= tol, "var {k}: grads diverged by {d}");
        }
    }

    /// The narrowed forward is `forward` + narrow by another route: values
    /// exact, and adapter gradients through a window loss exact too —
    /// positions outside the window contribute exact zeros through the
    /// narrow adjoint, so the two graphs backprop identical cotangents into
    /// every layer.
    #[test]
    fn narrowed_forward_matches_values_and_adapter_grads_exactly() {
        let cfg = tiny_cfg();
        let model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let input = ids(6);
        let (start, len) = (2, 3);
        let vars = model.trainable_vars();

        let full = model
            .forward(&input)
            .unwrap()
            .narrow(1, start, len)
            .unwrap();
        // UFCS: dispatch through the TRAIT, so the `impl GradModel`
        // delegation bodies are exercised, not just the inherent methods.
        let narrowed = GradModel::forward_narrowed(&model, &input, start, len).unwrap();
        assert_eq!(full.dims(), narrowed.dims());
        assert_eq!(
            max_abs_diff(&full, &narrowed),
            0.0,
            "narrowed values diverged"
        );

        let detached = GradModel::forward_detached_narrowed(&model, &input, start, len).unwrap();
        assert_eq!(
            max_abs_diff(&full, &detached),
            0.0,
            "detached values diverged"
        );

        let g_full = model.backward(&probe_loss(&full)).unwrap();
        let g_narrow = model.backward(&probe_loss(&narrowed)).unwrap();
        assert_grads_match(&g_full, &g_narrow, &vars, 0.0);
        // Non-vacuity: the probe produces real gradients.
        assert!(vars.iter().any(|v| {
            let g = g_full.get(v).unwrap();
            max_abs_diff(g, &g.zeros_like().unwrap()) > 1e-6
        }));

        // The detached route is genuinely tape-free: a raw backward through
        // its probe reaches no adapter var.
        let raw = probe_loss(&detached).backward().unwrap();
        assert!(vars.iter().all(|v| raw.get(v).is_none()));
    }

    /// Under checkpointing the window narrow rides the LOSS tape (the tail
    /// boundary stays full-width): the stitched narrowed backward matches the
    /// uncut narrowed one, and a narrowed *detached* walk never captures a
    /// tape.
    #[test]
    fn narrowed_remat_stitches_and_detached_stays_off_the_tape() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let input = ids(6);
        let (start, len) = (2, 3);
        let vars = model.trainable_vars();

        let uncut = model
            .backward(&probe_loss(
                &model.forward_narrowed(&input, start, len).unwrap(),
            ))
            .unwrap();
        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(
                &model.forward_narrowed(&input, start, len).unwrap(),
            ))
            .unwrap();
        assert_var_grads_close(&uncut, &stitched, &vars);

        let _ = model.forward_detached_narrowed(&input, start, len).unwrap();
        assert!(
            model.tape.borrow().is_none(),
            "a detached narrowed walk captured a checkpoint tape"
        );
    }

    /// The P7 gate's finite-difference half, in f64 end-to-end (the trainer
    /// gradcheck convention) so central differences are sharp: the stitched
    /// analytic gradient matches `(L(θ+ε) − L(θ−ε)) / 2ε` on sampled entries
    /// of the first and last adapter vars (deepest and shallowest stitch).
    #[test]
    fn checkpointed_backward_passes_a_finite_difference_gradcheck() {
        let cfg = tiny_cfg();
        let map_f64: HashMap<String, Tensor> = weight_map(&cfg)
            .into_iter()
            .map(|(k, t)| (k, t.to_dtype(DType::F64).unwrap()))
            .collect();
        let vb = VarBuilder::from_tensors(map_f64, DType::F64, &dev());
        let mut model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        for (i, v) in model.trainable_vars().iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                v.set(&Tensor::randn(0f64, 0.5f64, dims, &dev()).unwrap())
                    .unwrap();
            }
        }
        model.set_activation_checkpointing(true);
        let input = ids(5);
        let vars = model.trainable_vars();

        let loss = probe_loss(&model.forward(&input).unwrap());
        let grads = model.backward(&loss).unwrap();

        let eps = 1e-5f64;
        for var in [&vars[0], vars.last().unwrap()] {
            let analytic = grads.get(var).unwrap().to_vec2::<f64>().unwrap()[0][0];
            let orig = var.as_tensor().to_vec2::<f64>().unwrap();
            let loss_at = |delta: f64, model: &QwenGradModel| -> f64 {
                let mut bent = orig.clone();
                bent[0][0] += delta;
                let rows = bent.len();
                let cols = bent[0].len();
                let flat: Vec<f64> = bent.into_iter().flatten().collect();
                var.set(&Tensor::from_vec(flat, (rows, cols), &dev()).unwrap())
                    .unwrap();
                let l = probe_loss(&model.forward(&input).unwrap())
                    .to_scalar::<f64>()
                    .unwrap();
                // Drop the tape this value-forward captured (checkpointing is
                // on), so the FD loop leaves the model clean.
                model.tape.borrow_mut().take();
                l
            };
            let numeric = (loss_at(eps, &model) - loss_at(-eps, &model)) / (2.0 * eps);
            loss_at(0.0, &model); // restore the entry
            let rel = (analytic - numeric).abs() / analytic.abs().max(1e-8);
            assert!(
                rel <= 1e-5,
                "FD gradcheck failed: analytic={analytic}, numeric={numeric}, rel={rel}"
            );
        }
    }

    #[test]
    fn backward_demands_a_pending_matching_tape() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_activation_checkpointing(true);
        let scalar = Tensor::zeros((), DType::F32, &dev()).unwrap();

        // (a) No checkpointed forward has run.
        let err = model.backward(&scalar).unwrap_err();
        assert!(err.to_string().contains("no checkpointed forward"));

        // (b) A tape is consumed by exactly one backward.
        let loss = probe_loss(&model.forward(&ids(4)).unwrap());
        model.backward(&loss).unwrap();
        let err = model.backward(&loss).unwrap_err();
        assert!(err.to_string().contains("no checkpointed forward"));

        // (c) A loss from an OLDER forward cannot consume a newer tape.
        let stale_loss = probe_loss(&model.forward(&ids(4)).unwrap());
        let _ = model.forward(&ids(4)).unwrap(); // replaces the tape
        let err = model.backward(&stale_loss).unwrap_err();
        assert!(
            err.to_string().contains("tail boundary"),
            "want the foreign-loss error, got: {err}"
        );
    }

    #[test]
    fn an_adapter_flip_between_forward_and_backward_fails_loud() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_activation_checkpointing(true);
        let loss = probe_loss(&model.forward(&ids(4)).unwrap());
        model.set_adapter_enabled(false);
        let err = model.backward(&loss).unwrap_err();
        assert!(
            err.to_string().contains("adapter toggle flipped"),
            "want the adapter-flip error, got: {err}"
        );
    }

    #[test]
    fn forward_detached_matches_forward_and_stays_off_the_tape() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let input = ids(6);

        // Identical values (a rolling detach is the identity on values)…
        let want = model.forward(&input).unwrap();
        let got = model.forward_detached(&input).unwrap();
        assert_eq!(
            got.to_vec3::<f32>().unwrap(),
            want.to_vec3::<f32>().unwrap(),
            "forward_detached drifted from forward"
        );
        // …but tape-free: a backward through them reaches no trainable var.
        let raw = probe_loss(&got).backward().unwrap();
        assert!(model.trainable_vars().iter().all(|v| raw.get(v).is_none()));

        // And under checkpointing it must NOT capture a tape (a value scoring
        // may never clobber the tape the next update backward consumes).
        model.set_activation_checkpointing(true);
        let _ = model.forward_detached(&input).unwrap();
        let scalar = Tensor::zeros((), DType::F32, &dev()).unwrap();
        let err = model.backward(&scalar).unwrap_err();
        assert!(err.to_string().contains("no checkpointed forward"));
    }

    /// Toggling checkpointing OFF drops a pending tape (a stale tape must not
    /// survive a mode flip and get stitched later).
    #[test]
    fn toggling_checkpointing_off_drops_the_pending_tape() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        model.set_activation_checkpointing(true);
        let loss = probe_loss(&model.forward(&ids(4)).unwrap()); // captures a tape
        model.set_activation_checkpointing(false);
        model.set_activation_checkpointing(true);
        let err = model.backward(&loss).unwrap_err();
        assert!(
            err.to_string().contains("no checkpointed forward"),
            "the mode flip kept a stale tape alive: {err}"
        );
    }

    /// Stitched == uncut on a batch > 1 input: the mask-length derivation
    /// from the boundary dims must read the SEQ axis (`dims[1]`), which a
    /// batch-1 fixture cannot distinguish from a batch read.
    #[test]
    fn checkpointed_gradients_match_at_batch_two() {
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let v: Vec<u32> = (0..12u32).map(|i| i % 5).collect();
        let input = Tensor::from_vec(v, (2, 6), &dev()).unwrap();
        let vars = model.trainable_vars();

        let plain = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();
        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();
        assert_var_grads_close(&plain, &stitched, &vars);
    }

    /// The minimal shapes: a single-token sequence (the `mask == None` branch
    /// of the checkpointed forward AND of the backward's mask rebuild) and a
    /// single-layer stack (one segment — the reverse loop's boundary case).
    #[test]
    fn checkpointed_gradients_match_at_minimal_shapes() {
        // seq_len 1 on the standard 2-layer fixture.
        let cfg = tiny_cfg();
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let vars = model.trainable_vars();
        let plain = model
            .backward(&probe_loss(&model.forward(&ids(1)).unwrap()))
            .unwrap();
        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(&model.forward(&ids(1)).unwrap()))
            .unwrap();
        assert_var_grads_close(&plain, &stitched, &vars);

        // A single-layer model (segments == 1).
        let mut cfg = tiny_cfg();
        cfg.num_hidden_layers = 1;
        let mut model = QwenGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let vars = model.trainable_vars();
        let plain = model
            .backward(&probe_loss(&model.forward(&ids(4)).unwrap()))
            .unwrap();
        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(&model.forward(&ids(4)).unwrap()))
            .unwrap();
        assert_var_grads_close(&plain, &stitched, &vars);
    }
}
