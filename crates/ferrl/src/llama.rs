//! A grad-bearing, uncached dense Llama-3.x forward — the second [`GradModel`].
//!
//! candle-transformers ships a Llama forward, but it is inference-shaped
//! (`&mut Cache`, last-position-only logits, all layer types private) and built
//! from ops that have **no backward**, so it cannot be used to train. This module
//! is the *update* path for the dense Llama family (Llama 3.x: plain GQA,
//! rotate-half `RoPE` with optional llama3 scaling, `SwiGLU`, `RMSNorm`, no
//! QK-norm, no biases): a full-sequence, uncached forward over the **same loaded
//! weights** as candle's shipped `llama::Llama`, expressed entirely in
//! grad-bearing ops, with a manual `LoRA` adapter attached per a
//! [`DenseLoraTargets`] recipe (the historical `load()` default is q/v-only;
//! see [`LlamaGradModel::load_with_targets`] for the industrial
//! every-projection recipe).
//!
//! It is the second implementor of the [`GradModel`] / [`CachedDecoder`] seam
//! (after [`crate::qwen`]) — the validator that the model abstraction is real:
//! the generic [`crate::lm_policy::LmPolicy`] and the `Trainer` drive it with
//! **zero** policy/trainer changes. The architecture-specific code is only this
//! module; everything neutral (frozen linear, GQA `repeat_kv`, `RoPE` tables,
//! causal masks) comes from [`crate::blocks`].
//!
//! ## The three grad landmines (all replaced here)
//!
//! The same three autograd-cutting fused ops the Qwen forward replaces
//! (see [`crate::qwen`] for the full table): `candle_nn::ops::rms_norm` →
//! [`crate::nn::RmsNorm`] (`rms_norm_slow`), `candle_nn::rotary_emb::rope` →
//! `rope_slow`, `candle_nn::ops::softmax_last_dim` → `softmax(_, D::Minus1)`.
//! `silu` has a backward and is reused verbatim.
//!
//! ## Parity notes vs the shipped forward
//!
//! - The shipped non-flash attention **force-casts q/k/v to F32** for the
//!   score/softmax computation and casts the context back to the model dtype;
//!   this forward mirrors that (a no-op at F32, the same numerics at BF16).
//!   The causal mask is therefore built in F32 here, matching the F32 scores.
//! - `head_dim` is derived as `hidden_size / num_attention_heads` (the llama
//!   `Config` has no `head_dim` field), exactly as the shipped loader does.
//! - `tie_word_embeddings == true` reuses the embedding matrix as the LM head
//!   (the shared weight map then has **no** `lm_head.weight`), mirroring shipped.
//! - `rope_scaling`: `None` (or `Some` with `rope_type: Default`) is the plain
//!   `1/theta^(2i/d)` family; `Some` with `rope_type: Llama3` applies the llama3
//!   wavelength-smoothing rescale to the inv-freqs at table-build time — both
//!   mirrored from the shipped `Cache::new`, pinned by exact-value inv-freq
//!   pins (every smoothing branch) plus the equivalence gates.
//!
//! ## Validation beyond CPU CI: real weights and the bf16 path
//!
//! Every CI test here runs on CPU at F32, where the attention force-cast pair
//! is a same-dtype `to_dtype` — an op-free clone, structurally absent from the
//! autograd graph. The gaps that leaves are closed by two `#[ignore]`d manual
//! gates (run by hand with a staged Llama-3.2-1B checkpoint; see their module
//! docs), both green as of 2026-06:
//!
//! - `tests/llama_real_weights.rs` (CPU, F32): per-position equivalence vs
//!   shipped at real scale — including the real llama3 `RoPE`-scaling regime
//!   (factor 32 over 131k positions) and the tied 128k-row head — measured
//!   worst per-position max-abs divergence 1.7e-5; plus two-phase per-branch
//!   `LoRA`-grad coverage and a tokenizer round-trip.
//! - `tests/llama_gpu_smoke.rs` (CUDA, bf16): the three behaviors previously
//!   deferred here, now validated on an `sm_80` GPU — **bf16 logit equivalence
//!   vs shipped** (argmax agreement 12/12, max rel diff 0.95% of logit scale),
//!   **a real `ToDType` backward through the attention cast** (bf16-base /
//!   F32-adapter two-phase grad coverage, every gradient landing in the F32
//!   master dtype), and **bf16 merged-weight fidelity** (cached vs uncached:
//!   argmax 21/21, max rel diff 1.9%) — plus a GRPO smoke driving
//!   `LmPolicy<LlamaGradModel>` through the unchanged generic `Trainer`.
//!
//! Honest residual: these gates are manual (CI stays offline and CPU-only), so
//! a bf16 regression surfaces at the next manual gate run, not in CI.

use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::ops::softmax;
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

use crate::blocks::{causal_mask, causal_mask_at, frozen_linear, repeat_kv, RotaryTables};
use crate::lora::{DenseLoraTargets, Proj};
use crate::model::{CachedDecoder, GradModel};
use crate::nn::RmsNorm;

/// The plain rotate-half inverse frequencies `1/theta^(2i/d)`, one per rotated
/// dimension pair — computed in f32 exactly as the shipped
/// `calculate_default_inv_freq` (f32 `powf` on the f32 `rope_theta`), so the
/// tables match shipped bit-for-bit at build time.
fn default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

/// The inv-freqs the config actually asks for: the default family, or the
/// llama3 wavelength-smoothing rescale when `rope_scaling` requests it —
/// mirrored line-for-line from the shipped `Cache::new` (a `Some` config with
/// `rope_type: Default` is treated as unscaled there too).
fn inv_freq_for(cfg: &Config) -> Vec<f32> {
    match &cfg.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => default_inv_freq(cfg),
        Some(rs) => {
            let low_freq_wavelen = rs.original_max_position_embeddings as f32 / rs.low_freq_factor;
            let high_freq_wavelen =
                rs.original_max_position_embeddings as f32 / rs.high_freq_factor;
            default_inv_freq(cfg)
                .into_iter()
                .map(|freq| {
                    let wavelen = 2. * std::f32::consts::PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / rs.factor
                    } else {
                        let smooth = (rs.original_max_position_embeddings as f32 / wavelen
                            - rs.low_freq_factor)
                            / (rs.high_freq_factor - rs.low_freq_factor);
                        (1. - smooth) * freq / rs.factor + smooth * freq
                    }
                })
                .collect()
        }
    }
}

/// One dense-Llama attention block; each projection carries the `LoRA` adapter
/// or stays frozen per the [`DenseLoraTargets`] recipe. Replicates the shipped
/// `CausalSelfAttention` (no QK-norm, no biases, F32 score path) with the
/// grad-safe substitutions and no KV cache.
#[derive(Debug)]
struct LlamaAttention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
}

impl LlamaAttention {
    fn load(
        cfg: &Config,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        // The llama Config has no head_dim field; derive it as shipped does.
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
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
        let in_dtype = x.dtype();

        // 1. Projections, each adapted or frozen per the recipe. No biases
        //    anywhere.
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

        // 3. RoPE (grad-safe rope_slow; half-width cos/sin). No QK-norm.
        let (cos, sin) = rot.slice(l)?;
        let q = rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k = rope_slow(&k.contiguous()?, &cos, &sin)?;

        // 4. (no KV cache) GQA repeat.
        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v.contiguous()?, self.num_kv_groups)?.contiguous()?;

        // 5. Scaled dot-product attention in F32 (the shipped non-flash path
        //    force-casts q/k/v to F32 — a grad-safe identity at F32, the same
        //    numerics at BF16), with the grad-safe softmax.
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
            / (self.head_dim as f64).sqrt())?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?.to_dtype(in_dtype)?;

        // 6. Output projection.
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

/// `SwiGLU` MLP, activation fixed to `silu` (the llama Config has no
/// `hidden_act` knob); each projection may carry the adapter per the
/// [`DenseLoraTargets`] recipe.
#[derive(Debug)]
struct LlamaMlp {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
}

impl LlamaMlp {
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
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&Activation::Silu)?;
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
struct LlamaLayer {
    ln1: RmsNorm,
    attn: LlamaAttention,
    ln2: RmsNorm,
    mlp: LlamaMlp,
}

impl LlamaLayer {
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
            attn: LlamaAttention::load(
                cfg,
                &vb.pp("self_attn"),
                targets,
                rank,
                alpha,
                adapter_dtype,
            )?,
            ln2: RmsNorm::new(vb.pp("post_attention_layernorm").get(h, "weight")?, eps),
            mlp: LlamaMlp::load(cfg, &vb.pp("mlp"), targets, rank, alpha, adapter_dtype)?,
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

/// A grad-bearing, uncached dense Llama-3.x forward with `LoRA` attached per a
/// [`DenseLoraTargets`] recipe — the second [`GradModel`] implementor.
///
/// Built from the same [`VarBuilder`] (over the same safetensors) as candle's
/// shipped `llama::Llama`, so the two are weight-identical and their logits
/// match (the M1 equivalence gate). The base weights are frozen [`Tensor`]s;
/// only the `LoRA` `A`/`B` factors are trainable [`Var`]s, in a deterministic
/// layer-major order (the positional checkpoint contract).
#[derive(Debug)]
pub struct LlamaGradModel {
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<LlamaLayer>,
    norm: RmsNorm,
    rot: RotaryTables,
    hidden: usize,
    device: Device,
    targets: DenseLoraTargets,
}

impl LlamaGradModel {
    /// Load the model from `vb`, attaching a `LoRA` adapter of the given `rank`
    /// and `alpha` with the **historical q/v-only recipe**
    /// ([`DenseLoraTargets::legacy`]) — kept (rather than the industrial
    /// default) so pre-recipe adapter checkpoints stay positionally loadable
    /// through this constructor. Use
    /// [`load_with_targets`](Self::load_with_targets) for the industrial
    /// recipe.
    ///
    /// `vb` must be over the Llama safetensors (any dtype; F32 for the CPU
    /// equivalence gate). `cfg` is candle's own `llama::Config` so derived dims
    /// (notably `head_dim = hidden_size / num_attention_heads`) match the
    /// shipped model exactly. Only the non-flash-attention configuration is
    /// supported; `use_flash_attn == true` is rejected (see Errors) rather than
    /// loaded as a silently non-parity model.
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
    /// This is the same **bf16-base / F32-adapter** split as
    /// [`crate::qwen::QwenGradModel::load_with_adapter_dtype`]: load `vb` in
    /// BF16 while keeping the adapter — and so its gradients and the `AdamW`
    /// moments — in F32, where a small update cannot collapse. See
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
    /// Returns a candle error if `cfg.use_flash_attn` is set, if `targets`
    /// selects nothing (an untrainable model), if a weight tensor is missing
    /// or mis-shaped, or if the `LoRA` factors cannot be allocated.
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
                "LlamaGradModel: DenseLoraTargets selects no projection — the model would \
                 have no trainable parameters"
            );
        }
        // Fail loud on Config options this forward does not implement, rather
        // than silently loading a non-parity model (the P3 pattern). Flash
        // attention is a fused GPU kernel with no backward and different
        // masking semantics; this update path implements only the non-flash
        // F32 SDPA the shipped CPU/parity path uses.
        if cfg.use_flash_attn {
            candle_core::bail!(
                "LlamaGradModel: cfg.use_flash_attn=true is unsupported (this grad-bearing \
                 forward implements only the non-flash F32 attention path; loading it \
                 would silently produce non-parity logits)"
            );
        }
        // head_dim is DERIVED as hidden_size / num_attention_heads (the llama
        // Config has no head_dim field). The shipped loader trips a reshape
        // error deep in the forward when the division truncates; we would
        // silently run a degenerate (non-parity) model — so reject it at load.
        if cfg.hidden_size % cfg.num_attention_heads != 0 {
            candle_core::bail!(
                "LlamaGradModel: hidden_size {} is not divisible by num_attention_heads {} \
                 (head_dim is derived as their quotient; such a config cannot be a real \
                 Llama and would silently load as a degenerate model)",
                cfg.hidden_size,
                cfg.num_attention_heads
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
            layers.push(LlamaLayer::load(
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
            // The inv-freqs carry the whole rope_scaling story (llama3
            // smoothing happens at table-build time); the table layout itself
            // is the architecture-neutral one in crate::blocks.
            rot: RotaryTables::with_inv_freq(
                inv_freq_for(cfg),
                cfg.max_position_embeddings,
                vb.dtype(),
                vb.device(),
            )?,
            hidden: h,
            device: vb.device().clone(),
            targets,
        })
    }

    /// The [`DenseLoraTargets`] recipe this model was loaded with (for logging
    /// and checkpoint metadata — see [`DenseLoraTargets::canonical`]).
    #[must_use]
    pub fn lora_targets(&self) -> DenseLoraTargets {
        self.targets
    }

    /// Full-sequence logits `[batch, seq, vocab]` for `input_ids` (`[batch,
    /// seq]`, `u32`). Unlike candle's shipped `Llama::forward` (which narrows
    /// to the last position and force-casts to F32), this returns every
    /// position in the model dtype so the trainer can score whole completions.
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
        // The mask is built in F32 because the attention scores it is added to
        // are always F32 (the shipped force-cast, mirrored in LlamaAttention).
        let mask = if l == 1 {
            None
        } else {
            Some(causal_mask(l, DType::F32, &self.device)?)
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

    /// The device the weights live on, so a caller (e.g.
    /// [`crate::LlamaPolicy`]) can build `input_ids` tensors on the same device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Snapshot the **current** effective weights into a KV-cached, grad-free
    /// [`LlamaMergedDecoder`] for fast incremental rollout.
    ///
    /// Same design as [`crate::qwen::QwenGradModel::merged_decoder`]: folds the
    /// live `LoRA` adapter into every adapted projection via
    /// [`crate::lora::LoraLinear::merged_weight`] (respecting the adapter
    /// toggle), clones the frozen rest, and hands back a decoder over candle's
    /// `ConcatKvCache` — O(L) per token instead of the uncached forward's
    /// O(L²). **Rebuild after every optimizer step** (and after any
    /// `set_adapter_enabled` flip): the returned decoder is a tape-detached
    /// value snapshot.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any `merged_weight` build fails.
    pub fn merged_decoder(&self) -> CandleResult<LlamaMergedDecoder> {
        LlamaMergedDecoder::from_model(self)
    }
}

/// The [`GradModel`] seam over [`LlamaGradModel`]: pure delegation to the
/// inherent methods above (which stay public — the trait adds a generic
/// surface, it does not replace the concrete one).
impl GradModel for LlamaGradModel {
    type Decoder = LlamaMergedDecoder;

    fn device(&self) -> &Device {
        LlamaGradModel::device(self)
    }

    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        LlamaGradModel::forward(self, input_ids)
    }

    fn trainable_vars(&self) -> Vec<Var> {
        LlamaGradModel::trainable_vars(self)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        LlamaGradModel::set_adapter_enabled(self, enabled);
    }

    fn merged_decoder(&self) -> CandleResult<LlamaMergedDecoder> {
        LlamaGradModel::merged_decoder(self)
    }

    fn lora_recipe(&self) -> Option<String> {
        Some(self.targets.canonical())
    }
}

/// One dense-Llama attention block over **merged** weights with an incremental
/// KV cache — the grad-free mirror of [`LlamaAttention`]. Every projection uses
/// its single effective weight (the folded
/// [`crate::lora::LoraLinear::merged_weight`] when adapted, the frozen base
/// otherwise; all bias-free — the llama family has no projection biases); the
/// un-repeated K/V are appended to a [`ConcatKvCache`] before `repeat_kv`, the
/// shipped op order (project → reshape/transpose → `RoPE(offset)` → cache
/// append → `repeat_kv` → F32 SDPA → `o_proj`).
#[derive(Debug)]
struct LlamaMergedAttention {
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
    /// Un-repeated K/V cache (`[b, kv_heads, seq, head_dim]`), concatenated on
    /// the sequence axis (dim 2); `repeat_kv` is applied to the cache's output,
    /// never to what is stored.
    cache: ConcatKvCache,
}

impl LlamaMergedAttention {
    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let in_dtype = x.dtype();

        // 1. Projections over merged weights (adapted ones carry the folded
        //    adapter; the rest are the frozen base). Bias-free by construction.
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

        // 3. RoPE at the absolute position `offset` (grad-safe rope_slow).
        let (cos, sin) = rot.slice_at(offset, l)?;
        let q = rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k = rope_slow(&k.contiguous()?, &cos, &sin)?;

        // 4. Append the UN-repeated K/V, then GQA-repeat the full cached K/V —
        //    repeat AFTER append (the shipped order) so the cache stays compact.
        let (k, v) = self.cache.append(&k.contiguous()?, &v.contiguous()?)?;
        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

        // 5. Scaled dot-product attention in F32 (mirroring the shipped
        //    force-cast and the uncached LlamaAttention) with grad-safe softmax.
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
            / (self.head_dim as f64).sqrt())?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?.to_dtype(in_dtype)?;

        // 6. Output projection.
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.attn_hidden))?;
        frozen_linear(&ctx, &self.o_weight)
    }
}

/// `SwiGLU` MLP over merged weights — the grad-free mirror of [`LlamaMlp`].
#[derive(Debug)]
struct LlamaMergedMlp {
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
}

impl LlamaMergedMlp {
    fn from_layer(mlp: &LlamaMlp) -> CandleResult<Self> {
        Ok(Self {
            gate_weight: mlp.gate_proj.merged_weight()?,
            up_weight: mlp.up_proj.merged_weight()?,
            down_weight: mlp.down_proj.merged_weight()?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = frozen_linear(x, &self.gate_weight)?.apply(&Activation::Silu)?;
        let rhs = frozen_linear(x, &self.up_weight)?;
        frozen_linear(&lhs.broadcast_mul(&rhs)?, &self.down_weight)
    }
}

/// One decoder layer over merged weights: pre-norm cached attention + pre-norm
/// merged `SwiGLU`, both residual. The grad-free mirror of [`LlamaLayer`].
#[derive(Debug)]
struct LlamaMergedLayer {
    ln1: RmsNorm,
    attn: LlamaMergedAttention,
    ln2: RmsNorm,
    mlp: LlamaMergedMlp,
}

impl LlamaMergedLayer {
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

/// A KV-cached, **grad-free** dense-Llama decoder over weights with the `LoRA`
/// adapter already folded in — the fast rollout twin of [`LlamaGradModel`],
/// and its [`CachedDecoder`].
///
/// Built by [`LlamaGradModel::merged_decoder`]; same design, contracts, and
/// cache lifecycle as [`crate::qwen::MergedDecoder`] (value snapshot —
/// rebuild after any optimizer step or adapter toggle; `offset` must equal the
/// cached length, enforced fail-loud; `reset_cache` starts a fresh sequence).
/// It holds **no** [`Var`] and records no autograd tape.
#[derive(Debug)]
pub struct LlamaMergedDecoder {
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<LlamaMergedLayer>,
    norm: RmsNorm,
    rot: RotaryTables,
    hidden: usize,
    device: Device,
}

impl LlamaMergedDecoder {
    /// Snapshot a [`LlamaGradModel`]'s current effective weights. Private —
    /// callers go through [`LlamaGradModel::merged_decoder`].
    fn from_model(model: &LlamaGradModel) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let a = &layer.attn;
            // This snapshot applies NO bias, which is sound by construction:
            // every projection is a `Proj`, and `Proj::load` always builds
            // bias-free (the llama family has no projection biases).
            layers.push(LlamaMergedLayer {
                ln1: layer.ln1.clone(),
                attn: LlamaMergedAttention {
                    q_weight: a.q_proj.merged_weight()?,
                    k_weight: a.k_proj.merged_weight()?,
                    v_weight: a.v_proj.merged_weight()?,
                    o_weight: a.o_proj.merged_weight()?,
                    num_heads: a.num_heads,
                    num_kv_heads: a.num_kv_heads,
                    num_kv_groups: a.num_kv_groups,
                    head_dim: a.head_dim,
                    attn_hidden: a.attn_hidden,
                    cache: ConcatKvCache::new(2),
                },
                ln2: layer.ln2.clone(),
                mlp: LlamaMergedMlp::from_layer(&layer.mlp)?,
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
        })
    }

    /// Logits `[batch, chunk_len, vocab]` for `input_ids` (`[batch,
    /// chunk_len]`, `u32`) placed at absolute positions `[offset, offset +
    /// chunk_len)`, appending to the KV cache.
    ///
    /// Pass the whole prompt at `offset == 0` to prefill, then one token at a
    /// time at the running offset to decode. `offset` **must** equal the number
    /// of tokens already in the cache (it indexes the `RoPE` tables and sizes
    /// the causal mask); a mismatch is rejected (see Errors) rather than
    /// silently producing wrong logits. Like [`LlamaGradModel::forward`],
    /// every position is returned (the caller narrows to the last for sampling).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `offset` does not equal the cached sequence
    /// length, if any tensor op fails, or if `offset + chunk_len` exceeds the
    /// `RoPE` table's `max_position_embeddings`.
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        // Same fail-loud guard as the Qwen MergedDecoder: the l == 1 decode
        // path builds no mask, so an offset desync would silently corrupt the
        // logits rather than trip a shape error. All layer caches advance in
        // lockstep, so layer 0 is the truth.
        let cached = self
            .layers
            .first()
            .map_or(0, |layer| layer.attn.cache.current_seq_len());
        if offset != cached {
            candle_core::bail!(
                "LlamaMergedDecoder::forward: offset {offset} != cached sequence length \
                 {cached} (pass offset == tokens already decoded; 0 to prefill)"
            );
        }
        let ids = input_ids.flatten_all()?;
        let mut h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?;
        // A single new token attends to the whole cache (all past keys are
        // causally valid). The mask is F32 to match the F32 attention scores.
        let mask = if l == 1 {
            None
        } else {
            Some(causal_mask_at(offset, l, DType::F32, &self.device)?)
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

/// The [`CachedDecoder`] seam over [`LlamaMergedDecoder`]: pure delegation to
/// the inherent methods above (which carry the offset fail-loud guard and the
/// cache-lifecycle contract the trait requires).
impl CachedDecoder for LlamaMergedDecoder {
    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        LlamaMergedDecoder::forward(self, input_ids, offset)
    }

    fn reset_cache(&mut self) {
        LlamaMergedDecoder::reset_cache(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::grad_coverage;
    use candle_transformers::models::llama::{Cache, Llama};
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::{RngExt, SeedableRng};
    use std::collections::HashMap;

    fn dev() -> Device {
        Device::Cpu
    }

    /// Seed for the deterministic test-weight RNG. The weights MUST be seeded:
    /// with unseeded `Tensor::randn` a real forward bug shows up as an
    /// *intermittent* CI failure (each run draws fresh weights), which reads as
    /// flake and gets retried into green. Seeded weights make every gate
    /// reproduce identically on every run.
    const WEIGHT_SEED: u64 = 0x4C4C_414D_4131; // "LLAMA1"

    /// Weight std for the tiny test models. At the original `N(0, 0.2)` and
    /// hidden 8, attention was near-uniform (scores ≈ 1/seq for every key), so
    /// even deleting `RoPE` moved the logits by less than the gate envelopes —
    /// the equivalence gates were close to vacuous. `N(0, 1)` makes attention
    /// decisively non-uniform at these dims (measured; see the gate envelopes).
    const WEIGHT_STD: f32 = 1.0;

    /// Deterministic `N(0, std)` tensor: seeded Xoshiro (the same RNG family as
    /// `crate::sampler`) + Box–Muller, since `rand` alone ships no Normal
    /// distribution and candle's CPU device rejects `set_seed`.
    fn seeded_randn(
        rng: &mut Xoshiro256PlusPlus,
        std: f32,
        dims: &[usize],
        device: &Device,
    ) -> Tensor {
        let n: usize = dims.iter().product();
        let mut v = Vec::with_capacity(n + 1);
        while v.len() < n {
            // Box–Muller: two uniforms -> two independent standard normals.
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random();
            let r = (-2.0f32 * u1.ln()).sqrt();
            let (sin, cos) = (2.0 * std::f32::consts::PI * u2).sin_cos();
            v.push(std * r * cos);
            v.push(std * r * sin);
        }
        v.truncate(n);
        Tensor::from_vec(v, dims.to_vec(), device).unwrap()
    }

    /// A tiny dense-Llama config (2 layers, 2 Q / 1 KV head → real GQA, derived
    /// `head_dim` 4) for offline tests — same arithmetic as a real Llama-3.x at
    /// a runnable scale. `tie` toggles the tied vs separate `lm_head` branch.
    fn cfg_variant(tie: bool) -> Config {
        Config {
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
            tie_word_embeddings: tie,
        }
    }

    fn tiny_cfg() -> Config {
        cfg_variant(true)
    }

    /// The tiny config with llama3 `RoPE` scaling. `original_max_position_embeddings`
    /// picks which smoothing branches the two inv-freqs (`head_dim` 4 → freqs
    /// 1.0 and 0.01) hit: 16 → freq0 lands in the *smooth interpolation* band
    /// and freq1 in the *scaled-down* band; 28 (with `high_freq_factor` 4 →
    /// `high_freq_wavelen` 7 > freq0's wavelength 2π) → freq0 *passes through
    /// unscaled* and freq1 is scaled — together the three llama3 branches.
    fn cfg_rope_scaled(original_max: usize) -> Config {
        Config {
            rope_scaling: Some(Llama3RopeConfig {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: original_max,
                rope_type: Llama3RopeType::Llama3,
            }),
            ..tiny_cfg()
        }
    }

    /// Deterministic (seeded — see [`WEIGHT_SEED`]) random weights matching
    /// `cfg`'s dotted tensor names (incl. `lm_head.weight` only when untied —
    /// the tied map must NOT carry it, so a shared `VarBuilder` proves neither
    /// loader requires it). Norm weights are 1-D `[n]`, projections
    /// `[out, in]`. No QK-norm tensors, no biases. Insertion order is fixed,
    /// so every call yields bit-identical tensors.
    fn weight_map(cfg: &Config) -> HashMap<String, Tensor> {
        let d = dev();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(WEIGHT_SEED);
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, dims: &[usize]| {
            t.insert(
                name.to_string(),
                seeded_randn(&mut rng, WEIGHT_STD, dims, &d),
            );
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let qo = cfg.num_attention_heads * head_dim;
        let kvo = cfg.num_key_value_heads * head_dim;
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
            put(&format!("{p}.mlp.gate_proj.weight"), &[i, h]);
            put(&format!("{p}.mlp.up_proj.weight"), &[i, h]);
            put(&format!("{p}.mlp.down_proj.weight"), &[h, i]);
        }
        t
    }

    /// In-memory `VarBuilder` over `weight_map` (no shared temp file → no race
    /// under parallel test execution).
    fn tiny_vb(cfg: &Config) -> VarBuilder<'static> {
        VarBuilder::from_tensors(weight_map(cfg), DType::F32, &dev())
    }

    fn ids(seq: usize) -> Tensor {
        let v: Vec<u32> = (0..seq as u32).map(|i| i % 5).collect();
        Tensor::from_vec(v, (1, seq), &dev()).unwrap()
    }

    // ---- gate envelopes ----------------------------------------------------
    //
    // All measured under the seeded `WEIGHT_SEED`/`WEIGHT_STD` weights (the
    // measured worst per gate is recorded next to each constant; logit scale
    // ≈ 6–7), then set with ~10–30x headroom for cross-host float
    // reassociation (CI's CPU is not the dev host's — the P2 platform
    // lesson; the floors themselves are deterministic under the seed). Tight
    // envelopes are what make these gates non-vacuous: at the old 1e-3 with
    // unseeded N(0, 0.2) weights, even removing RoPE entirely passed most
    // draws, while here the RoPE-scaling signal alone is ~1.2 in logit space
    // — four orders of magnitude above every envelope.

    /// OUR uncached forward vs candle's SHIPPED forward (slow-twin vs fused
    /// kernels over the same weights). Measured worst across the four
    /// configs: 4.8e-6 → ~10x headroom.
    const SHIPPED_TOL: f32 = 5e-5;

    /// Cached (merged-decoder) vs OUR uncached forward, adapter armed — same
    /// grad-safe ops on both sides, so only cache-chunking + merge
    /// reassociation. Measured worst across the five gates: 3.4e-6 → ~15x.
    const MERGED_TOL: f32 = 5e-5;

    /// Cached base-only decode vs candle's shipped KV-cached forward (crosses
    /// the slow-twin/fused-kernel gap AND the cache wiring). Measured worst
    /// 1.7e-6 → ~30x.
    const SHIPPED_CACHED_TOL: f32 = 5e-5;

    /// The cache-only pin: cached base-only vs uncached base — identical ops,
    /// the ONLY difference is incremental caching, so its floor is the lowest
    /// of all (measured worst 9.6e-7); kept at ≥100x headroom.
    const CACHE_PIN_TOL: f32 = 1e-4;

    /// Max absolute element-wise difference between two (broadcast-compatible)
    /// tensors.
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

    #[test]
    fn forward_produces_full_seq_logits() {
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        let logits = model.forward(&ids(5)).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
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
        // The dtype-split mechanism on a tiny model (same CPU surrogate as the
        // Qwen gate: F32 base / F64 adapter — candle's CPU backend has no bf16
        // matmul): the forward runs in the base dtype and the adapter masters
        // receive their gradients in the adapter dtype. Two-phase per-branch
        // MAGNITUDE coverage (not just presence + dtype): a cast bug yielding
        // present-but-ZERO grads must fail here, the same bar the all-F32
        // grad test holds.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg); // F32 base
        let mut model =
            LlamaGradModel::load_with_adapter_dtype(&cfg, &vb, 2, 4.0, DType::F64).unwrap();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        let (q_vars, v_vars) = branch_split(&vars);

        // The forward itself stays in the base/activation dtype.
        let logits = model.forward(&ids(5)).unwrap();
        assert_eq!(
            logits.dtype(),
            DType::F32,
            "the forward runs in the base/activation dtype"
        );

        // Phase 1 — zero-B init: every var present in the grad store, each
        // branch live (via dL/dB), all grads finite, and every grad lands in
        // the MASTER dtype (F64), not the F32 compute dtype.
        let g1 = backward_grads(&model);
        assert!(
            grad_coverage(&q_vars, &g1).unwrap().is_ok(),
            "q-branch unhealthy at zero-B init under the dtype split"
        );
        assert!(
            grad_coverage(&v_vars, &g1).unwrap().is_ok(),
            "v-branch unhealthy at zero-B init under the dtype split"
        );
        assert_grads_in_master_dtype(&g1, &vars);

        // Phase 2 — force every B nonzero (`force_b_nonzero` casts to the var
        // dtype, F64 here): now EVERY A and B must carry a nonzero finite
        // grad in the master dtype, proving the A-input path survives the
        // dtype casts too.
        force_b_nonzero(&vars);
        let g2 = backward_grads(&model);
        let qc = grad_coverage(&q_vars, &g2).unwrap();
        let vc = grad_coverage(&v_vars, &g2).unwrap();
        assert!(
            qc.nonzero == qc.total && qc.nonfinite == 0,
            "q-branch: not every dtype-split LoRA var is live after nonzero-B: {qc:?}"
        );
        assert!(
            vc.nonzero == vc.total && vc.nonfinite == 0,
            "v-branch: not every dtype-split LoRA var is live after nonzero-B: {vc:?}"
        );
        assert_grads_in_master_dtype(&g2, &vars);
    }

    /// Every var's gradient must land in the F64 MASTER dtype (the dtype-split
    /// routing property), not the F32 compute dtype.
    fn assert_grads_in_master_dtype(grads: &candle_core::backprop::GradStore, vars: &[Var]) {
        for v in vars {
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

    /// One `forward -> sqr -> sum -> backward` over a 6-token input, returning
    /// the grad store.
    fn backward_grads(model: &LlamaGradModel) -> candle_core::backprop::GradStore {
        model
            .forward(&ids(6))
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
    }

    /// Set every `B` factor (the odd index within each `[A, B]` pair) to small
    /// seeded noise, so the update is no longer a no-op and `dL/dA` is no
    /// longer 0. Cast to each var's own dtype, so it also serves the
    /// dtype-split (F64-adapter) model.
    fn force_b_nonzero(vars: &[Var]) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(WEIGHT_SEED ^ 0xB);
        for (i, v) in vars.iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                let noise = seeded_randn(&mut rng, 0.02, &dims, &dev())
                    .to_dtype(v.dtype())
                    .unwrap();
                v.set(&noise).unwrap();
            }
        }
    }

    #[test]
    fn lora_grads_flow_through_llama_backward() {
        // Two-phase, PER-BRANCH grad coverage through the full Llama backward
        // (rms_norm_slow + rope_slow + grad-bearing softmax on the q path; the
        // always-grad-safe net on the v path). NOTE: at F32 the attention
        // force-cast pair is a same-dtype `to_dtype` — an op-free clone,
        // structurally ABSENT from this graph — so this test does NOT cover
        // the bf16 `ToDType` backward (covered by the `#[ignore]`d
        // `llama_bf16_dtype_split_grads_on_gpu` gate in
        // `tests/llama_gpu_smoke.rs`; see the module docs).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        // q/v A+B over 2 layers = 8 Vars; per-layer order: q_A, q_B, v_A, v_B.
        assert_eq!(vars.len(), cfg.num_hidden_layers * 4);
        let (q_vars, v_vars) = branch_split(&vars);

        // Phase 1 — zero-B init: every var present in the grad store + each
        // branch live (via dL/dB) + finite. At zero-B, dL/dA is structurally 0,
        // so a severed A-path would be invisible here — phase 2 closes that.
        let g1 = backward_grads(&model);
        assert!(
            grad_coverage(&q_vars, &g1).unwrap().is_ok(),
            "q-branch unhealthy at zero-B init (grad-safe twin cut?)"
        );
        assert!(
            grad_coverage(&v_vars, &g1).unwrap().is_ok(),
            "v-branch unhealthy at zero-B init"
        );

        // Phase 2 — force every B nonzero: now EVERY A and B must carry a
        // nonzero finite grad (proves the A-input path is wired, not just B).
        force_b_nonzero(&vars);
        let g2 = backward_grads(&model);
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
    fn adapter_toggle_is_noop_at_zero_b_init() {
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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
    fn adapter_toggle_changes_output_with_trained_b() {
        // Force every LoRA B nonzero (seeded, via arm_adapter), then assert
        // enabling the adapter changes the output — proving set_adapter_enabled
        // fans out to every layer's q_proj AND v_proj.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        arm_adapter(&model);
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
    fn load_rejects_flash_attn() {
        // A valid llama Config we don't implement must fail loud, not load a
        // silently non-parity model (the P3 pattern).
        let mut cfg = tiny_cfg();
        cfg.use_flash_attn = true;
        let vb = tiny_vb(&cfg);
        let err = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string().contains("use_flash_attn"),
            "expected a use_flash_attn rejection, got: {err}"
        );
    }

    #[test]
    fn load_rejects_indivisible_head_dim() {
        // hidden_size not divisible by num_attention_heads: the shipped model
        // errors at a reshape mid-forward; ours derives a truncated head_dim
        // and would silently run a degenerate non-parity model — so the load
        // must fail loud instead (the P3 pattern).
        let mut cfg = tiny_cfg();
        cfg.hidden_size = 10; // 10 % 2 == 0 would pass; use 4 heads: 10 % 4 != 0
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        let vb = tiny_vb(&cfg);
        let err = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string().contains("num_attention_heads"),
            "expected a divisibility rejection, got: {err}"
        );
    }

    // ---- equivalence vs candle's shipped llama::Llama ----------------------

    /// The shipped model's logits at position `t` (i.e. on the prefix
    /// `[0..=t]`), `(1, vocab)` F32, computed with a FRESH uncached `Cache` per
    /// call (`Llama::forward` needs `&mut Cache` even uncached — causal masks
    /// are memoized in it — and returns last-position-only logits).
    fn shipped_at(shipped: &Llama, cfg: &Config, input: &Tensor, t: usize) -> Tensor {
        let prefix = input.narrow(1, 0, t + 1).unwrap();
        let mut cache = Cache::new(false, DType::F32, cfg, &dev()).unwrap();
        shipped.forward(&prefix, 0, &mut cache).unwrap()
    }

    /// Load our forward AND candle's shipped `Llama` from the SAME weights and
    /// assert that for EVERY position `t`, our full-sequence logits at `t`
    /// match the shipped model's (last-position-only, F32) logits on the prefix
    /// `[0..=t]` — the strongest CPU oracle (a last-position-only gate cannot
    /// catch a causal-mask bug in a non-final row, but GRPO scores per-token
    /// log-probs across the WHOLE completion). Returns the worst max-abs diff.
    fn assert_matches_shipped_all_positions(cfg: &Config, seq: usize) -> f32 {
        let vb = tiny_vb(cfg);
        let shipped = Llama::load(vb.clone(), cfg).unwrap();
        let mut ours = LlamaGradModel::load(cfg, &vb, 2, 4.0).unwrap();
        ours.set_adapter_enabled(false); // base only, for a like-for-like compare
        let input = ids(seq);
        let ours_all = ours.forward(&input).unwrap(); // [1, seq, vocab]
        let mut worst = 0f32;
        for t in 0..seq {
            let shipped_t = shipped_at(&shipped, cfg, &input, t); // (1, vocab)
            let ours_t = ours_all.narrow(1, t, 1).unwrap(); // (1, 1, vocab)
            let md = max_abs_diff(&shipped_t, &ours_t);
            assert!(
                md <= SHIPPED_TOL,
                "full-seq logits at position {t} diverged from shipped: max-abs={md}"
            );
            worst = worst.max(md);
        }
        worst
    }

    #[test]
    fn llama_forward_matches_shipped_tied_gqa() {
        // tied lm_head + real GQA (2 Q / 1 KV head) + causal mask, every position.
        let worst = assert_matches_shipped_all_positions(&tiny_cfg(), 5);
        assert!(worst.is_finite());
    }

    #[test]
    fn llama_forward_matches_shipped_untied() {
        // separate lm_head.weight branch, every position.
        assert_matches_shipped_all_positions(&cfg_variant(false), 4);
    }

    #[test]
    fn llama_forward_matches_shipped_single_token() {
        // seq == 1 exercises the mask == None branch on both sides.
        assert_matches_shipped_all_positions(&tiny_cfg(), 1);
    }

    #[test]
    fn llama_forward_matches_shipped_rope_scaled() {
        // The llama3 RoPE-scaling branches, each pinned against shipped (whose
        // Cache::new computes the same smoothing): original_max 16 → the smooth
        // interpolation + scaled-down branches; original_max 28 → the
        // pass-through + scaled-down branches. Together all three. (The exact
        // per-branch numerics are pinned independently of weights by
        // `inv_freq_pins_the_llama3_scaling_branches_exactly` — this gate then
        // proves the scaled tables flow through the forward as shipped's do.)
        assert_matches_shipped_all_positions(&cfg_rope_scaled(16), 5);
        assert_matches_shipped_all_positions(&cfg_rope_scaled(28), 5);

        // Non-vacuity premise: over the SAME weights, scaled-vs-unscaled
        // tables must move the logits by MORE than the gate envelope — i.e.
        // had we built unscaled tables, the gate above COULD have failed.
        let scaled_cfg = cfg_rope_scaled(16);
        let vb = tiny_vb(&scaled_cfg); // same weight shapes as tiny_cfg's
        let mut scaled = LlamaGradModel::load(&scaled_cfg, &vb, 2, 4.0).unwrap();
        scaled.set_adapter_enabled(false);
        let mut plain = LlamaGradModel::load(&tiny_cfg(), &vb, 2, 4.0).unwrap();
        plain.set_adapter_enabled(false);
        let input = ids(5);
        let premise = max_abs_diff(
            &scaled.forward(&input).unwrap(),
            &plain.forward(&input).unwrap(),
        );
        assert!(
            premise > SHIPPED_TOL,
            "rope-scaling moved the logits by only {premise} (≤ envelope {SHIPPED_TOL}) — \
             the rope-scaled equivalence gate would be vacuous at these dims"
        );
    }

    #[test]
    fn inv_freq_pins_the_llama3_scaling_branches_exactly() {
        // The weight-based equivalence gates alone CANNOT catch llama3
        // smoothing-branch bugs at the tiny dims: a planted `freq` instead of
        // `freq/factor`, or swapped smooth-interpolation weights, moved the
        // logits by only ~4e-5..9e-5 (the pre-fix adversarial sweep measured
        // 0/30 random draws failing). So pin `inv_freq_for` itself against
        // HAND-COMPUTED values from the shipped `Cache::new` formula, one pin
        // per smoothing branch — deterministic and independent of weights.
        //
        // Tiny config: head_dim 4 → two inv-freqs: freq0 = 1/10000^0 = 1.0,
        // freq1 = 1/10000^(2/4) = 0.01. Scaling params: factor 8,
        // low_freq_factor 1, high_freq_factor 4.
        //
        // The tolerance is f32-epsilon scale (≈8 ULP at these magnitudes; it
        // only absorbs platform `powf`/rounding jitter), while the branch-bug
        // deltas are ≥ 2.7e-2 (swapped smooth weights) and 8.75e-3 (unscaled
        // low-freq) — 4+ orders of magnitude above it.
        let tol = 1e-6f32;
        let check = |got: Vec<f32>, want: [f32; 2], ctx: &str| {
            assert_eq!(got.len(), want.len(), "{ctx}: inv-freq count");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= tol,
                    "{ctx}: inv_freq[{i}] = {g}, want {w} (±{tol})"
                );
            }
        };

        // original_max 16: low_freq_wavelen 16, high_freq_wavelen 4.
        //   freq0 = 1.0:  wavelen 2π ≈ 6.28319 ∈ (4, 16) → SMOOTH branch:
        //     smooth = (16/2π − 1)/(4 − 1)            = 0.51549303
        //     expect = (1 − smooth)·1.0/8 + smooth·1.0 = 0.57605640
        //   freq1 = 0.01: wavelen 2π/0.01 ≈ 628.3 > 16 → SCALED: 0.01/8.
        check(
            inv_freq_for(&cfg_rope_scaled(16)),
            [0.576_056_4, 0.001_25],
            "original_max 16 (smooth + scaled branches)",
        );

        // original_max 28: low_freq_wavelen 28, high_freq_wavelen 7.
        //   freq0 = 1.0:  wavelen ≈ 6.28319 < 7  → PASS-THROUGH: exactly 1.0.
        //   freq1 = 0.01: wavelen ≈ 628.3   > 28 → SCALED: 0.01/8.
        check(
            inv_freq_for(&cfg_rope_scaled(28)),
            [1.0, 0.001_25],
            "original_max 28 (pass-through + scaled branches)",
        );
    }

    #[test]
    fn llama_forward_matches_shipped_rope_some_default() {
        // A Some(rope_scaling) whose rope_type is Default must be treated as
        // UNSCALED (the shipped Cache::new's `None | Some(Default)` arm) — and
        // therefore produce tables identical to the plain config's.
        let cfg = Config {
            rope_scaling: Some(Llama3RopeConfig {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 16,
                rope_type: Llama3RopeType::Default,
            }),
            ..tiny_cfg()
        };
        assert_eq!(
            inv_freq_for(&cfg),
            default_inv_freq(&cfg),
            "rope_type: Default must fall back to the unscaled inv-freqs"
        );
        assert_matches_shipped_all_positions(&cfg, 4);
    }

    // ---- LlamaMergedDecoder: cached-rollout equivalence gates ---------------

    /// Force every `LoRA` B factor (odd `trainable_vars` indices) to seeded
    /// random values so the adapter is a genuine perturbation, not the zero-B
    /// no-op — the merge must then differ from the base. Seeded for the same
    /// reason as the weights: the decoder-gate floors must be reproducible.
    fn arm_adapter(model: &LlamaGradModel) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(WEIGHT_SEED ^ 0xA);
        for (i, v) in model.trainable_vars().iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                v.set(&seeded_randn(&mut rng, 0.5, &dims, &dev())).unwrap();
            }
        }
    }

    /// Uncached base-only logits over the same weights `vb`, for the
    /// non-vacuity witness (the armed adapter must move the logits away from
    /// this).
    fn uncached_base_logits(cfg: &Config, vb: &VarBuilder, input: &Tensor) -> Tensor {
        let mut m = LlamaGradModel::load(cfg, vb, 2, 4.0).unwrap();
        m.set_adapter_enabled(false);
        m.forward(input).unwrap()
    }

    #[test]
    fn merged_decoder_matches_uncached_token_by_token() {
        // THE core decoder gate: cached single-token decode == uncached
        // full-seq forward at every position, adapter ON, at F32.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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
            worst <= MERGED_TOL,
            "cached token-by-token decode diverged from uncached forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_prefill_then_incremental_matches_uncached() {
        // The realistic generate() pattern: prefill the prompt in one chunk
        // (multi-token causal mask), then decode one token at a time at the
        // running offset (offset>0 incremental decode).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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
            worst <= MERGED_TOL,
            "cached prefill+incremental decode diverged from uncached forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_chunked_at_offset_matches_uncached() {
        // Two MULTI-token chunks: [0..3] at offset 0, then [3..7] at offset 3.
        // The second chunk has chunk_len>1 AND offset>0 — the only path that
        // builds the rectangular causal mask, never reached by prefill
        // (offset 0) or single-token decode (l==1 => mask None). Adapter ON.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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
            worst <= MERGED_TOL,
            "chunked decode (multi-token chunk at offset>0) diverged from uncached: {worst}"
        );
    }

    #[test]
    fn merged_decoder_base_only_matches_shipped_every_position() {
        // The adapter-OFF snapshot == candle's shipped KV-CACHED forward fed
        // one token at a time (also proves merged_weight respects the toggle —
        // the adapter is armed but disabled, so the snapshot must be pure base).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        arm_adapter(&model);
        model.set_adapter_enabled(false);
        let mut dec = model.merged_decoder().unwrap();

        let shipped = Llama::load(vb.clone(), &cfg).unwrap();
        let mut cache = Cache::new(true, DType::F32, &cfg, &dev()).unwrap();

        let seq = 6;
        let input = ids(seq);
        let mut worst = 0f32;
        for t in 0..seq {
            let tok = input.narrow(1, t, 1).unwrap();
            let ours_t = dec.forward(&tok, t).unwrap(); // (1, 1, vocab)
            let shipped_t = shipped.forward(&tok, t, &mut cache).unwrap(); // (1, vocab)
            worst = worst.max(max_abs_diff(&shipped_t, &ours_t));
        }
        assert!(
            worst <= SHIPPED_CACHED_TOL,
            "base-only cached decode diverged from candle's shipped cached forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_base_only_matches_uncached_base() {
        // Same grad-safe ops on both sides; the ONLY difference is incremental
        // caching, so this is a tight pin on the cache/offset/mask wiring
        // alone, independent of the slow-twin vs fused-kernel gap the shipped
        // gate tolerates.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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
            worst <= CACHE_PIN_TOL,
            "base-only cached decode diverged from uncached base forward: {worst}"
        );
    }

    #[test]
    fn merged_decoder_reset_cache_restarts_sequence() {
        // reset_cache() lets one decoder serve a fresh sequence; a replay after
        // reset must reproduce the reference (a leftover cache would not).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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
            worst <= MERGED_TOL,
            "decode after reset_cache diverged from the reference: {worst}"
        );
    }

    #[test]
    fn merged_decoder_rejects_offset_mismatch() {
        // The offset MUST equal the cached length. On the l==1 decode path no
        // mask is built, so a desync would silently corrupt the logits rather
        // than trip a shape error — the decoder guards and fails loud.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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

    #[test]
    fn merged_decoder_rejects_a_biased_projection() {
        // `Proj::load` always builds bias-free (no supported config has
        // projection biases), but `LoraLinear` can carry one — and the merged
        // snapshot applies NO bias, so a future biased construction path would
        // make the cached rollout silently diverge from the uncached forward.
        // The snapshot must fail loud instead (the guard lives in
        // `Proj::merged_weight`, shared by every model's merged decoder).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        let mut model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        let h = cfg.hidden_size; // q_out == hidden at these dims
        let w = Tensor::zeros((h, h), DType::F32, &dev()).unwrap();
        let bias = Tensor::ones(h, DType::F32, &dev()).unwrap();
        model.layers[0].attn.q_proj =
            Proj::Lora(crate::lora::LoraLinear::new(w, Some(bias), 2, 4.0).unwrap());
        let err = model.merged_decoder().unwrap_err();
        assert!(
            err.to_string().contains("base bias"),
            "expected a biased-projection rejection, got: {err}"
        );
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
            LlamaGradModel::load_with_targets(&cfg, &vb, 2, 4.0, DType::F32, none).unwrap_err();
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
    fn assert_legacy_frozen(layer: &LlamaLayer) {
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
    fn industrial_layer_vars(layer: &LlamaLayer) -> Vec<Var> {
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
        let model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
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
        let model = LlamaGradModel::load_with_targets(
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
        let mut model = LlamaGradModel::load_with_targets(
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
        force_b_nonzero(&vars);
        let grads = backward_grads(&model);
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
            LlamaGradModel::load_with_targets(&cfg, &vb, 2, 4.0, DType::F32, targets).unwrap();
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
        let mut model = LlamaGradModel::load_with_targets(
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
        let mut model = LlamaGradModel::load_with_targets(
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
}
