//! A grad-bearing, uncached `qwen3_5` (Qwen3.5 / Qwen3.6) text forward — the
//! third [`GradModel`], and the first **hybrid** one.
//!
//! The `qwen3_5` family interleaves two token mixers (`layer_types`, 3:1 in the
//! shipped configs): **`GatedDeltaNet` linear attention** (a gated delta-rule
//! recurrence over a `[k_dim, v_dim]` state per value head, fed by a causal
//! depthwise conv — see [`crate::gdn`]) and **gated GQA full attention** (a
//! doubled `q_proj` whose per-head output halves are `[query | gate]`, QK-norm
//! before partial-rotary `RoPE`, and a `sigmoid(gate)` on the attention
//! output). Orthogonally, the family has a SECOND layer menu — the
//! feed-forward slot: dense members (`model_type: "qwen3_5"`) run a plain
//! `SwiGLU` MLP, `MoE` members (`"qwen3_5_moe"`, e.g. 35B-A3B) the sparse
//! block ([`crate::moe`]: top-k routed experts + a sigmoid-gated shared
//! expert) in EVERY layer — the per-layer dense/sparse knobs are deleted in
//! this family. One architecture, both menus, the same [`GradModel`]. candle-transformers 0.10.2 ships **no** `qwen3_5`/`qwen3_next`
//! support at all, so unlike [`crate::qwen`]/[`crate::llama`] there is no
//! shipped forward to pin against — the reference oracle is HF transformers
//! (pinned `v5.11.0`) instead, via committed tiny-config fixtures and staged
//! real-weights logit dumps (see the test tiers below).
//!
//! ## Architecture facts this module is built to (the 0.8B-Base config)
//!
//! - Decoder config nested under `text_config` (`model_type: "qwen3_5_text"`);
//!   checkpoint tensors under the **`model.language_model.*`** prefix (the
//!   multimodal `ForConditionalGeneration` layout every family checkpoint
//!   ships); `model.visual.*` / `mtp.*` tensors are never requested; tied
//!   embeddings mean **no** `lm_head.weight`.
//! - **Two `RMSNorm` conventions in one model**: decoder/final/q/k norms are
//!   **zero-centered** ([`RmsNormZeroCentered`], effective scale `1 + w`); the
//!   `GatedDeltaNet` output norm is **plain-`w` gated** ([`RmsNormGated`],
//!   norm-before-gate, weight kept in the activation dtype).
//! - **Partial rotary** `partial_rotary_factor 0.25`: rotate-half on the first
//!   `head_dim/4` dims only ([`crate::blocks::rope_partial`]); the interleaved
//!   M-`RoPE` of the reference is an exact no-op for text-only inputs (all
//!   three T/H/W position rows are identical), pinned by the PR-1 real-geometry
//!   rope oracle gate.
//! - **fp32 boundaries exactly as the reference**: the delta-rule state, its
//!   decay gate `g`, and the attention softmax run in F32 under a BF16 model;
//!   everything else stays in the activation dtype.
//!
//! ## The grad landmines (all replaced here)
//!
//! Same table as [`crate::qwen`]: fused `rms_norm` → the slow-path norms in
//! [`crate::nn`]; fused `rope*` → [`crate::blocks::rope_partial`] (rotate-half
//! composite); `softmax_last_dim` → `softmax(_, D::Minus1)`. Additionally for
//! this family: grouped `conv1d` → the shifted-taps composite
//! ([`crate::gdn::causal_depthwise_conv1d`]) and the missing `softplus` → the
//! stable composite ([`crate::gdn::stable_softplus`]) — both with committed
//! transformers-fallback fixtures from PR-1.
//!
//! ## Training form vs decode form
//!
//! The grad forward runs the delta rule in its **chunked WY form**
//! ([`crate::gdn::gated_delta_rule_chunked`], chunk [`GDN_CHUNK_SIZE`]) — the
//! reference's own training/prefill dispatch, and the only form whose autograd
//! tape scales per *chunk* rather than per *token*. The cached
//! [`Qwen3_5MergedDecoder`] mirrors the reference dispatch exactly: chunked
//! for multi-token inputs (prefill at 0 or continuation at an offset),
//! sequential-recurrent ([`crate::gdn::gated_delta_rule_recurrent`]) for
//! single-token decode.
//!
//! ## Inputs are unpadded
//!
//! Like every [`GradModel`], the forward takes `input_ids` only — the ferrl
//! trainer scores unpadded sequences, which is the reference's
//! `attention_mask == all-ones` regime where its padding-state masking and
//! left-padded linear-attention mask are both no-ops by construction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{bail, DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::ops::{sigmoid, softmax};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::blocks::{
    causal_mask, causal_mask_at, frozen_linear, repeat_kv, rope_partial, windowed, RotaryTables,
};
use crate::gdn::{
    causal_depthwise_conv1d, gated_delta_rule_chunked, gated_delta_rule_recurrent, stable_softplus,
};
use crate::lora::Proj;
use crate::model::{CachedDecoder, GradModel};
use crate::nn::{RmsNormGated, RmsNormZeroCentered};
use crate::remat::{stitched_backward, RematTape};

/// Delta-rule chunk length for training and multi-token prefill — the
/// reference kernel default, and what the pinned oracle executes. A pure
/// compute-scheduling constant (output is chunk-size invariant up to float
/// reassociation, pinned by the PR-1 cross-checks), not a quality knob.
pub const GDN_CHUNK_SIZE: usize = 64;

/// The checkpoint prefix the whole `qwen3_5` family ships its text decoder
/// under (the multimodal `ForConditionalGeneration` layout — even text-only
/// checkpoints like 0.8B-Base use it).
const CKPT_PREFIX: &str = "model.language_model";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// One decoder layer's token-mixer kind (`config.layer_types[i]`).
///
/// Deserialization is fail-loud: any string other than the two shipped kinds
/// is a config error, never a silently mis-built layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum LayerType {
    /// A [`GatedDeltaNet`](https://arxiv.org/abs/2412.06464) linear-attention
    /// layer (causal conv + gated delta-rule recurrence; no positional
    /// encoding, causal by construction).
    #[serde(rename = "linear_attention")]
    LinearAttention,
    /// A gated GQA full-attention layer (doubled `q_proj`, QK-norm, partial
    /// rotary `RoPE`, output gate).
    #[serde(rename = "full_attention")]
    FullAttention,
}

/// The `rope_parameters` sub-object of the text config.
///
/// Only the default rope family is supported (`rope_type: "default"`, which
/// has `attention_scaling == 1.0`); the M-`RoPE` fields are accepted and
/// validated but are an exact no-op for text-only inputs (see the module
/// docs), so they configure nothing here.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// Rope family; only `"default"` is supported (fail-loud otherwise).
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    /// Frequency base (`1e7` for the shipped family configs).
    pub rope_theta: f64,
    /// Fraction of each head's dims that rotate (`0.25` shipped). The product
    /// with `head_dim` must be a nonzero even integer.
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    /// M-`RoPE` text/height/width frequency split; validated for shape sanity
    /// only (it reduces to standard 1-D rope for text).
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    /// M-`RoPE` interleaved layout flag. `Some(false)` is rejected: the
    /// text-only no-op reduction this module relies on is proven for the
    /// interleaved layout the family ships.
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

fn default_rope_type() -> String {
    "default".to_string()
}

fn default_partial_rotary_factor() -> f64 {
    // The transformers Qwen3_5TextConfig back-compat default.
    0.25
}

fn default_full_attention_interval() -> usize {
    // The transformers __post_init__ default when layer_types is absent.
    4
}

fn default_true() -> bool {
    true
}

/// The `text_config` of a `qwen3_5` checkpoint — every field this forward
/// consumes, mirroring the HF `Qwen3_5TextConfig` names so the shipped
/// `config.json` deserializes directly.
///
/// Unknown JSON keys are tolerated (vision/MTP knobs ride along in shipped
/// configs); **known-but-unsupported values fail loud** in
/// [`validate`](Self::validate) — the P3 discipline: never silently load a
/// non-parity model.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5TextConfig {
    /// Vocabulary size (248320 for the shipped family).
    pub vocab_size: usize,
    /// Residual stream width.
    pub hidden_size: usize,
    /// `SwiGLU` MLP inner width — dense members only. The `MoE` members
    /// DELETE this field (every layer is sparse); exactly one of this and the
    /// [`num_experts`](Self::num_experts) quartet is present (validated).
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    /// Decoder layer count.
    pub num_hidden_layers: usize,
    /// Full-attention query head count.
    pub num_attention_heads: usize,
    /// Full-attention KV head count (GQA).
    pub num_key_value_heads: usize,
    /// Full-attention per-head width — an **explicit** field (256 shipped,
    /// larger than `hidden_size / num_attention_heads`: the attention width
    /// exceeds the residual width, the Qwen3 pattern).
    pub head_dim: usize,
    /// MLP activation; only `"silu"` is supported.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// `RMSNorm` epsilon (all norms in the model share it).
    pub rms_norm_eps: f64,
    /// Maximum absolute position (sizes the rope tables).
    pub max_position_embeddings: usize,
    /// Whether the LM head reuses the embedding matrix (true for 0.8B-Base;
    /// when true the checkpoint has no `lm_head.weight`).
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Projection bias flag; only the shipped `false` is supported.
    #[serde(default)]
    pub attention_bias: bool,
    /// Attention dropout; only the shipped `0.0` is supported (RL fine-tuning
    /// never trains with attention dropout here).
    #[serde(default)]
    pub attention_dropout: f64,
    /// Per-layer mixer kinds; absent means the
    /// [`full_attention_interval`](Self::full_attention_interval) pattern.
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    /// Every `interval`-th layer is full attention when
    /// [`layer_types`](Self::layer_types) is absent (the transformers
    /// `__post_init__` derivation).
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    /// Causal-conv kernel width in the `GatedDeltaNet` layers (4 shipped).
    pub linear_conv_kernel_dim: usize,
    /// Delta-rule key head width.
    pub linear_key_head_dim: usize,
    /// Delta-rule value head width.
    pub linear_value_head_dim: usize,
    /// Delta-rule key head count.
    pub linear_num_key_heads: usize,
    /// Delta-rule value head count (a multiple of the key head count; the
    /// ratio is the GVA query/key broadcast).
    pub linear_num_value_heads: usize,
    /// Rope family parameters (see [`RopeParameters`]).
    pub rope_parameters: RopeParameters,
    /// Layers whose mixer is skipped entirely in some `MoE` configs; only the
    /// shipped empty list is supported.
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    /// Whether full attention applies the output gate. Not a real knob in the
    /// reference (the gate is hardcoded in the class); shipped configs carry
    /// `true` and anything else is rejected.
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,
    /// The delta-rule state dtype. Not a real knob in the reference (fp32 is
    /// hardcoded, mirrored by [`crate::gdn`]); shipped configs carry
    /// `"float32"` and anything else is rejected.
    #[serde(default = "default_mamba_ssm_dtype")]
    pub mamba_ssm_dtype: String,
    /// Routed expert count — `MoE` members only (`qwen3_5_moe`, where EVERY
    /// layer's feed-forward is sparse). Present iff the other three `MoE`
    /// fields are (validated); dense members carry none of the four.
    #[serde(default)]
    pub num_experts: Option<usize>,
    /// Experts consulted per token (the router's top-k).
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    /// Per-routed-expert `SwiGLU` inner width.
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    /// The always-on shared expert's `SwiGLU` inner width.
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    /// Router-logit emission for the aux load-balancing loss — a
    /// pretraining-time concern outside the reference's eager forward (and
    /// outside RL fine-tuning, which never requests it). Shipped configs
    /// carry `false`; `true` is rejected rather than silently ignored.
    #[serde(default)]
    pub output_router_logits: bool,
}

/// The resolved sparse feed-forward geometry of an `MoE` member (see
/// [`Qwen3_5TextConfig::moe`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeDims {
    /// Routed expert count `E`.
    pub num_experts: usize,
    /// Experts consulted per token (top-k).
    pub top_k: usize,
    /// Per-routed-expert `SwiGLU` inner width.
    pub moe_intermediate_size: usize,
    /// The shared expert's `SwiGLU` inner width.
    pub shared_expert_intermediate_size: usize,
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_mamba_ssm_dtype() -> String {
    "float32".to_string()
}

/// A full `qwen3_5` checkpoint config (`config.json`): the text decoder under
/// `text_config`, with the vision/MTP sub-configs tolerated and ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5Config {
    /// Checkpoint model type; `"qwen3_5"` when present (fail-loud otherwise —
    /// loading a different family through this forward would be silent
    /// non-parity).
    #[serde(default)]
    pub model_type: Option<String>,
    /// The OUTER tie flag (the multimodal wrapper's own, distinct from the
    /// text config's). The loader keys the head choice on the text flag; a
    /// checkpoint where the two disagree would otherwise load silently with
    /// the wrong head, so disagreement is rejected at validation.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    /// The text decoder config this module consumes.
    pub text_config: Qwen3_5TextConfig,
}

impl Qwen3_5Config {
    /// Parse and [`validate`](Qwen3_5TextConfig::validate) a checkpoint
    /// `config.json` string.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the JSON does not parse into the supported
    /// shape or any validation rule fails.
    pub fn from_json_str(json: &str) -> CandleResult<Self> {
        let cfg: Self = serde_json::from_str(json)
            .map_err(|e| candle_core::Error::Msg(format!("qwen3_5 config parse: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read, parse, and validate a checkpoint's `config.json`.
    ///
    /// # Errors
    ///
    /// As [`from_json_str`](Self::from_json_str), plus I/O errors.
    pub fn from_json_file(path: impl AsRef<Path>) -> CandleResult<Self> {
        let json = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            candle_core::Error::Msg(format!(
                "qwen3_5 config read {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Self::from_json_str(&json)
    }

    /// Validate the composite config (model type + the text config rules).
    ///
    /// # Errors
    ///
    /// Returns a candle error on any unsupported value (see
    /// [`Qwen3_5TextConfig::validate`]).
    pub fn validate(&self) -> CandleResult<()> {
        if let Some(mt) = &self.model_type {
            if mt != "qwen3_5" && mt != "qwen3_5_moe" {
                bail!(
                    "qwen3_5 config: model_type {mt:?}, expected \"qwen3_5\" (dense) or \
                     \"qwen3_5_moe\" (sparse)"
                );
            }
            // The family tag and the feed-forward menu must agree — a dense
            // tag with MoE fields (or vice versa) is a foreign or hand-edited
            // config, not a member of either branch.
            let sparse_tag = mt == "qwen3_5_moe";
            let sparse_cfg = self.text_config.num_experts.is_some();
            if sparse_tag != sparse_cfg {
                bail!(
                    "qwen3_5 config: model_type {mt:?} disagrees with the feed-forward \
                     fields (num_experts {:?})",
                    self.text_config.num_experts
                );
            }
        }
        if let Some(outer) = self.tie_word_embeddings {
            if outer != self.text_config.tie_word_embeddings {
                bail!(
                    "qwen3_5 config: outer tie_word_embeddings {outer} disagrees with \
                     text_config.tie_word_embeddings {} — the loader keys the head on the \
                     text flag and would silently pick the wrong head",
                    self.text_config.tie_word_embeddings
                );
            }
        }
        self.text_config.validate()
    }
}

impl Qwen3_5TextConfig {
    /// The resolved per-layer mixer kinds: the explicit
    /// [`layer_types`](Self::layer_types) when present, else the
    /// `full_attention_interval` pattern (every `interval`-th layer full,
    /// 1-indexed — the transformers derivation).
    #[must_use]
    pub fn resolved_layer_types(&self) -> Vec<LayerType> {
        match &self.layer_types {
            Some(t) => t.clone(),
            None => (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % self.full_attention_interval == 0 {
                        LayerType::FullAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect(),
        }
    }

    /// The sparse feed-forward geometry when this config is an `MoE` member
    /// (`num_experts` present), `None` for dense members.
    ///
    /// Total by construction: returns `Some` only when ALL four `MoE` fields
    /// are present; [`validate`](Self::validate) enforces all-or-nothing, so
    /// on a validated config `moe().is_some() == num_experts.is_some()`.
    #[must_use]
    pub fn moe(&self) -> Option<MoeDims> {
        Some(MoeDims {
            num_experts: self.num_experts?,
            top_k: self.num_experts_per_tok?,
            moe_intermediate_size: self.moe_intermediate_size?,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size?,
        })
    }

    /// The number of rotated dims per full-attention head:
    /// `head_dim * partial_rotary_factor` (64 shipped).
    #[must_use]
    pub fn rotary_dim(&self) -> usize {
        // Validated to be an exact nonzero even integer by `validate`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rot = (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize;
        rot
    }

    /// Fail loud on any config value this forward does not implement, rather
    /// than silently loading a non-parity model (the P3 / M1 discipline).
    ///
    /// # Errors
    ///
    /// Returns a candle error naming the offending field and value.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> CandleResult<()> {
        for (name, v) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("full_attention_interval", self.full_attention_interval),
        ] {
            if v == 0 {
                bail!("qwen3_5 config: {name} must be > 0");
            }
        }
        // The feed-forward menu: dense members carry intermediate_size and
        // none of the MoE quartet; MoE members carry the complete quartet and
        // NO intermediate_size (the family deletes it — every layer sparse).
        // Anything mixed is a foreign config, not a member of either branch.
        match (self.intermediate_size, self.num_experts) {
            (Some(_), Some(_)) => bail!(
                "qwen3_5 config: both intermediate_size and num_experts present — the MoE \
                 members delete intermediate_size; this is not a member of either branch"
            ),
            (None, None) => bail!(
                "qwen3_5 config: neither intermediate_size (dense) nor num_experts (MoE) \
                 present"
            ),
            (Some(0), None) => bail!("qwen3_5 config: intermediate_size must be > 0"),
            (Some(_), None) => {
                for (name, v) in [
                    ("num_experts_per_tok", self.num_experts_per_tok),
                    ("moe_intermediate_size", self.moe_intermediate_size),
                    (
                        "shared_expert_intermediate_size",
                        self.shared_expert_intermediate_size,
                    ),
                ] {
                    if v.is_some() {
                        bail!(
                            "qwen3_5 config: dense member (intermediate_size present) carries \
                             the MoE field {name}"
                        );
                    }
                }
            }
            (None, Some(experts)) => {
                let Some(moe) = self.moe() else {
                    bail!(
                        "qwen3_5 config: num_experts present but the MoE quartet is \
                         incomplete (need num_experts_per_tok, moe_intermediate_size, \
                         shared_expert_intermediate_size)"
                    );
                };
                if experts == 0 {
                    bail!("qwen3_5 config: num_experts must be > 0");
                }
                if moe.top_k == 0 || moe.top_k > experts {
                    bail!(
                        "qwen3_5 config: num_experts_per_tok {} must be in 1..={experts}",
                        moe.top_k
                    );
                }
                if moe.moe_intermediate_size == 0 || moe.shared_expert_intermediate_size == 0 {
                    bail!("qwen3_5 config: MoE intermediate sizes must be > 0");
                }
            }
        }
        if self.output_router_logits {
            bail!(
                "qwen3_5 config: output_router_logits=true unsupported (the aux \
                 load-balancing loss is a pretraining concern; this forward is the eager \
                 path RL fine-tuning uses, which never emits router logits)"
            );
        }
        if self.hidden_act != "silu" {
            bail!(
                "qwen3_5 config: hidden_act {:?} unsupported (only \"silu\")",
                self.hidden_act
            );
        }
        if self.attention_bias {
            bail!("qwen3_5 config: attention_bias=true unsupported (the family ships bias-free)");
        }
        if self.attention_dropout != 0.0 {
            bail!(
                "qwen3_5 config: attention_dropout {} unsupported (only 0.0)",
                self.attention_dropout
            );
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            bail!(
                "qwen3_5 config: num_attention_heads {} not divisible by num_key_value_heads {}",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if self.linear_num_value_heads % self.linear_num_key_heads != 0 {
            bail!(
                "qwen3_5 config: linear_num_value_heads {} not divisible by \
                 linear_num_key_heads {}",
                self.linear_num_value_heads,
                self.linear_num_key_heads
            );
        }
        if self.linear_conv_kernel_dim == 0 {
            bail!("qwen3_5 config: linear_conv_kernel_dim must be >= 1");
        }
        if let Some(t) = &self.layer_types {
            if t.len() != self.num_hidden_layers {
                bail!(
                    "qwen3_5 config: layer_types has {} entries for {} layers",
                    t.len(),
                    self.num_hidden_layers
                );
            }
        }
        if !self.mlp_only_layers.is_empty() {
            bail!(
                "qwen3_5 config: non-empty mlp_only_layers {:?} unsupported (deleted in \
                 both family branches — the MoE members are sparse in every layer)",
                self.mlp_only_layers
            );
        }
        if !self.attn_output_gate {
            bail!(
                "qwen3_5 config: attn_output_gate=false unsupported (the gate is hardcoded in \
                 the reference and in this forward)"
            );
        }
        if self.mamba_ssm_dtype != "float32" {
            bail!(
                "qwen3_5 config: mamba_ssm_dtype {:?} unsupported (the reference hardcodes the \
                 fp32 delta-rule state this forward mirrors)",
                self.mamba_ssm_dtype
            );
        }
        let rp = &self.rope_parameters;
        if rp.rope_type != "default" {
            bail!(
                "qwen3_5 config: rope_type {:?} unsupported (only \"default\", whose \
                 attention_scaling is 1.0 — any other family would silently de-calibrate \
                 the rope tables)",
                rp.rope_type
            );
        }
        if !(rp.rope_theta.is_finite() && rp.rope_theta > 0.0) {
            bail!(
                "qwen3_5 config: rope_theta {} must be a positive number",
                rp.rope_theta
            );
        }
        let prf = rp.partial_rotary_factor;
        if !(prf > 0.0 && prf <= 1.0) {
            bail!("qwen3_5 config: partial_rotary_factor {prf} must be in (0, 1]");
        }
        let rot_exact = self.head_dim as f64 * prf;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rot = rot_exact as usize;
        #[allow(clippy::cast_precision_loss)]
        if (rot as f64 - rot_exact).abs() > 1e-9 || rot == 0 || rot % 2 != 0 {
            bail!(
                "qwen3_5 config: head_dim {} * partial_rotary_factor {prf} = {rot_exact} must \
                 be a nonzero even integer",
                self.head_dim
            );
        }
        if rp.mrope_interleaved == Some(false) {
            bail!(
                "qwen3_5 config: mrope_interleaved=false unsupported (the text-only no-op \
                 reduction this forward relies on is proven for the interleaved layout)"
            );
        }
        if let Some(sections) = &rp.mrope_section {
            let total: usize = sections.iter().sum();
            if total * 2 != rot {
                bail!(
                    "qwen3_5 config: mrope_section {sections:?} sums to {total}, expected \
                     rotary_dim/2 = {}",
                    rot / 2
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights loading
// ---------------------------------------------------------------------------

/// Build a [`VarBuilder`] over a checkpoint directory, supporting both the
/// single-file (`model.safetensors`) and multi-shard
/// (`model.safetensors.index.json` + shards) layouts — the 0.8B validator is a
/// single shard, the 9B/27B ladder is not.
///
/// The load is buffered (safe, no mmap): every shard is read into memory and
/// the tensors are converted to `dtype` on retrieval. Vision/MTP tensors ride
/// along in the map but are simply never requested by the loader.
///
/// # Errors
///
/// Returns a candle error if neither layout is present, a shard fails to
/// read/parse, or the index JSON is malformed.
pub fn varbuilder_from_pretrained(
    dir: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
) -> CandleResult<VarBuilder<'static>> {
    Ok(VarBuilder::from_tensors(
        tensors_from_pretrained(dir, device)?,
        dtype,
        device,
    ))
}

/// The raw checkpoint tensor map behind [`varbuilder_from_pretrained`] —
/// same directory layouts, same buffered load — for entry points that build
/// their own backend over it ([`Qwen3_5GradModel::load_full_ft`]).
///
/// # Errors
///
/// As [`varbuilder_from_pretrained`].
pub fn tensors_from_pretrained(
    dir: impl AsRef<Path>,
    device: &Device,
) -> CandleResult<HashMap<String, Tensor>> {
    let dir = dir.as_ref();
    let index_path = dir.join("model.safetensors.index.json");
    let single_path = dir.join("model.safetensors");
    let files: Vec<PathBuf> = if index_path.is_file() {
        #[derive(Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let raw = std::fs::read_to_string(&index_path)
            .map_err(|e| candle_core::Error::Msg(format!("read {}: {e}", index_path.display())))?;
        let index: Index = serde_json::from_str(&raw)
            .map_err(|e| candle_core::Error::Msg(format!("parse {}: {e}", index_path.display())))?;
        let mut names: Vec<String> = index.weight_map.into_values().collect();
        names.sort();
        names.dedup();
        names.into_iter().map(|n| dir.join(n)).collect()
    } else if single_path.is_file() {
        vec![single_path]
    } else {
        bail!(
            "qwen3_5 loader: neither model.safetensors.index.json nor model.safetensors \
             found in {}",
            dir.display()
        );
    };
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for file in files {
        let shard = candle_core::safetensors::load(&file, device)?;
        tensors.extend(shard);
    }
    Ok(tensors)
}

// ---------------------------------------------------------------------------
// LoRA targets
// ---------------------------------------------------------------------------

/// Which projections carry a trainable `LoRA` adapter — the configurable
/// recipe of the M2′ design. This is the **hybrid** (`qwen3_5`) recipe, with
/// `GatedDeltaNet` opt-ins; the dense models ([`crate::qwen`],
/// [`crate::llama`]) use [`crate::lora::DenseLoraTargets`] instead.
///
/// The [`Default`] is the **industrial recipe**: attention `q,k,v,o` on the
/// full-attention layers plus MLP `gate,up,down` on **every** layer
/// (the Unsloth-shipped Qwen3.5 recipe; PEFT-`"all-linear"`-minus-`GatedDeltaNet`).
/// The `GatedDeltaNet` projections are an explicit **opt-in** (default off):
/// the one hybrid-specific ablation — on Qwen3.5-0.8B itself — found adapting
/// the linear-attention backbone destructive, while literal all-linear trains
/// fine at larger scales, so the knob exists but is not the default. The conv
/// kernel, `A_log`/`dt_bias`, and every norm weight are never adapted (no
/// framework's `LoRA` recipe targets them by default).
///
/// The recipe (together with the config) **determines the trainable-var
/// order** — layer-major, fixed projection order within each layer — which is
/// the positional checkpoint contract; [`canonical`](Self::canonical) is the
/// stable string form for recording it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct LoraTargets {
    /// Full-attention `q_proj` (the doubled `[query | gate]` projection — the
    /// adapter co-modulates the gate half, matching the reference structure).
    pub attn_q: bool,
    /// Full-attention `k_proj`.
    pub attn_k: bool,
    /// Full-attention `v_proj`.
    pub attn_v: bool,
    /// Full-attention `o_proj`.
    pub attn_o: bool,
    /// MLP `gate_proj` (all layers). On `MoE` members this binds the SHARED
    /// expert's `gate_proj` — the router, the packed routed experts, and the
    /// scalar sigmoid gate are never adaptable (the locked `MoE`-`LoRA`
    /// policy: adapting the router makes routing non-stationary during
    /// training, and per-expert adapters on packed 3-D weights are
    /// out-of-recipe).
    pub mlp_gate: bool,
    /// MLP `up_proj` (all layers; shared expert on `MoE` members).
    pub mlp_up: bool,
    /// MLP `down_proj` (all layers; shared expert on `MoE` members).
    pub mlp_down: bool,
    /// `GatedDeltaNet` fused `in_proj_qkv` (opt-in; see the type docs).
    pub gdn_qkv: bool,
    /// `GatedDeltaNet` `in_proj_z` (output-gate input; opt-in).
    pub gdn_z: bool,
    /// `GatedDeltaNet` `in_proj_b` (write-strength input; opt-in).
    pub gdn_b: bool,
    /// `GatedDeltaNet` `in_proj_a` (decay input; opt-in).
    pub gdn_a: bool,
    /// `GatedDeltaNet` `out_proj` (opt-in).
    pub gdn_out: bool,
}

impl Default for LoraTargets {
    fn default() -> Self {
        Self::industrial()
    }
}

impl LoraTargets {
    /// The industrial default recipe (see the type docs).
    #[must_use]
    pub fn industrial() -> Self {
        Self {
            attn_q: true,
            attn_k: true,
            attn_v: true,
            attn_o: true,
            mlp_gate: true,
            mlp_up: true,
            mlp_down: true,
            gdn_qkv: false,
            gdn_z: false,
            gdn_b: false,
            gdn_a: false,
            gdn_out: false,
        }
    }

    /// Literal all-linear (every projection incl. the `GatedDeltaNet` ones) —
    /// the ms-swift-style opt-in.
    #[must_use]
    pub fn all_linear() -> Self {
        Self {
            gdn_qkv: true,
            gdn_z: true,
            gdn_b: true,
            gdn_a: true,
            gdn_out: true,
            ..Self::industrial()
        }
    }

    /// Whether any projection is targeted at all (a no-target model would
    /// train nothing — the loaders reject it).
    #[must_use]
    pub fn any(&self) -> bool {
        self.attn_q
            || self.attn_k
            || self.attn_v
            || self.attn_o
            || self.mlp_gate
            || self.mlp_up
            || self.mlp_down
            || self.gdn_qkv
            || self.gdn_z
            || self.gdn_b
            || self.gdn_a
            || self.gdn_out
    }

    /// A stable, human-readable encoding of the recipe (for logs and
    /// checkpoint metadata): e.g. the default is `attn:qkvo|mlp:gud|gdn:-`.
    #[must_use]
    pub fn canonical(&self) -> String {
        let pick = |pairs: &[(bool, char)]| -> String {
            let s: String = pairs
                .iter()
                .filter(|(on, _)| *on)
                .map(|(_, c)| *c)
                .collect();
            if s.is_empty() {
                "-".to_string()
            } else {
                s
            }
        };
        format!(
            "attn:{}|mlp:{}|gdn:{}",
            pick(&[
                (self.attn_q, 'q'),
                (self.attn_k, 'k'),
                (self.attn_v, 'v'),
                (self.attn_o, 'o'),
            ]),
            pick(&[
                (self.mlp_gate, 'g'),
                (self.mlp_up, 'u'),
                (self.mlp_down, 'd'),
            ]),
            pick(&[
                (self.gdn_qkv, 'q'),
                (self.gdn_z, 'z'),
                (self.gdn_b, 'b'),
                (self.gdn_a, 'a'),
                (self.gdn_out, 'o'),
            ]),
        )
    }

    /// No projection targeted (`attn:-|mlp:-|gdn:-`). The `LoRA` loaders
    /// reject it (an adapter recipe that trains nothing); it is the recipe a
    /// **full fine-tuning** model records in
    /// [`lora_targets`](Qwen3_5GradModel::lora_targets) — full-FT trains
    /// every base weight and carries no adapters at all.
    #[must_use]
    pub fn none() -> Self {
        Self {
            attn_q: false,
            attn_k: false,
            attn_v: false,
            attn_o: false,
            mlp_gate: false,
            mlp_up: false,
            mlp_down: false,
            gdn_qkv: false,
            gdn_z: false,
            gdn_b: false,
            gdn_a: false,
            gdn_out: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Load mode (LoRA vs full fine-tuning)
// ---------------------------------------------------------------------------

/// How a load builds its trainable surface: the `LoRA` recipe (the default
/// mode) or the full-fine-tuning var registry. Threaded through every layer
/// loader so the projection sites stay single-sourced.
enum LoadMode<'a> {
    /// Adapter mode: frozen base weights, `LoRA` factors per `targets`.
    Lora {
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        targets: LoraTargets,
    },
    /// Full fine-tuning: every base weight is a registry [`Var`]'s inner
    /// tensor (the `vb` is registry-backed); no adapters anywhere.
    FullFt {
        /// The registry the backend fills; the sparse loader also registers
        /// its packed expert tensors here explicitly.
        registry: &'a crate::full_ft::VarRegistry,
    },
}

impl LoadMode<'_> {
    /// Load one projection: adapter-wrapped per the recipe in `Lora` mode,
    /// plain `Frozen` in `FullFt` mode — where the "frozen" tensor is the
    /// registry var's inner tensor, trainable straight through
    /// [`frozen_linear`](crate::blocks::frozen_linear) (which never detaches
    /// its weight).
    fn proj(
        &self,
        vb: &VarBuilder,
        name: &str,
        shape: (usize, usize),
        select: fn(&LoraTargets) -> bool,
    ) -> CandleResult<Proj> {
        match self {
            Self::Lora {
                rank,
                alpha,
                adapter_dtype,
                targets,
            } => Proj::load(
                vb,
                name,
                shape,
                select(targets),
                *rank,
                *alpha,
                *adapter_dtype,
            ),
            // rank/alpha/dtype are dead on the unadapted path.
            Self::FullFt { .. } => Proj::load(vb, name, shape, false, 1, 1.0, DType::F32),
        }
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet (linear attention) layer
// ---------------------------------------------------------------------------

/// The static dims a `GatedDeltaNet` layer derives from the config.
#[derive(Debug, Clone, Copy)]
struct GdnDims {
    num_k_heads: usize,
    num_v_heads: usize,
    head_k: usize,
    head_v: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    kernel: usize,
}

impl GdnDims {
    fn from_config(cfg: &Qwen3_5TextConfig) -> Self {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        Self {
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            head_k: cfg.linear_key_head_dim,
            head_v: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim: 2 * key_dim + value_dim,
            kernel: cfg.linear_conv_kernel_dim,
        }
    }

    /// The GVA broadcast factor (value heads per key head; 1 for 0.8B-Base).
    fn gva_rep(&self) -> usize {
        self.num_v_heads / self.num_k_heads
    }
}

/// Repeat the head dim of a `[batch, seq, heads, d]` tensor `n_rep` times
/// consecutively per head (`torch.repeat_interleave(dim=2)`) — the GVA
/// query/key broadcast. Pure `broadcast_as` + `reshape`, so it carries
/// gradient (the backward sums over the repeats).
fn repeat_gva_heads(x: &Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, l, h, d) = x.dims4()?;
    x.unsqueeze(3)?
        .broadcast_as((b, l, h, n_rep, d))?
        .contiguous()?
        .reshape((b, l, h * n_rep, d))
}

/// The shared per-token-mixer math of the `GatedDeltaNet` layer: everything
/// between the input projections and the delta-rule kernel call that does not
/// depend on caching (split, head reshape, `beta`/`g` computation, GVA
/// repeat). `mixed` is the post-conv post-`SiLU` `[batch, seq, conv_dim]`.
struct GdnKernelInputs {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
}

fn gdn_kernel_inputs(
    dims: &GdnDims,
    mixed: &Tensor,
    b_in: &Tensor,
    a_in: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> CandleResult<GdnKernelInputs> {
    let (b, l, _) = mixed.dims3()?;
    let query = mixed
        .narrow(D::Minus1, 0, dims.key_dim)?
        .contiguous()?
        .reshape((b, l, dims.num_k_heads, dims.head_k))?;
    let key = mixed
        .narrow(D::Minus1, dims.key_dim, dims.key_dim)?
        .contiguous()?
        .reshape((b, l, dims.num_k_heads, dims.head_k))?;
    let value = mixed
        .narrow(D::Minus1, 2 * dims.key_dim, dims.value_dim)?
        .contiguous()?
        .reshape((b, l, dims.num_v_heads, dims.head_v))?;
    // beta in the activation dtype (the kernel upcasts); g ALWAYS in F32 —
    // the reference `.float()`s both factors (an fp16 A_log.exp() can be
    // -inf), and crate::gdn's contract requires the F32 log-decay. The casts
    // live HERE, at use time, not at load: a load-time cast of a trainable
    // (full-FT) var would freeze its load value into derived storage.
    let a_log = a_log.to_dtype(DType::F32)?;
    let dt_bias = dt_bias.to_dtype(DType::F32)?;
    let beta = sigmoid(b_in)?;
    let g = stable_softplus(&a_in.to_dtype(DType::F32)?.broadcast_add(&dt_bias)?)?
        .broadcast_mul(&a_log.exp()?)?
        .neg()?;
    let rep = dims.gva_rep();
    let query = repeat_gva_heads(&query, rep)?;
    let key = repeat_gva_heads(&key, rep)?;
    Ok(GdnKernelInputs {
        query,
        key,
        value,
        g,
        beta,
    })
}

/// One `GatedDeltaNet` layer (grad side, uncached): fused q‖k‖v projection →
/// causal depthwise conv + `SiLU` (z/b/a bypass the conv) → **chunked** gated
/// delta rule → per-head gated `RMSNorm` (plain-`w`, `silu(z)` gate) →
/// `out_proj`.
#[derive(Debug)]
struct Qwen3_5GatedDeltaNet {
    in_proj_qkv: Proj,
    in_proj_z: Proj,
    in_proj_b: Proj,
    in_proj_a: Proj,
    /// Depthwise conv taps, stored in the checkpoint's `[conv_dim, 1, kernel]`
    /// layout **as fetched** and squeezed per forward — a load-time squeeze
    /// would still share storage today (contiguous reshape), but the
    /// store-as-fetched rule is what keeps every full-FT weight var-backed by
    /// construction, not by a layout accident.
    conv_weight: Tensor,
    /// `[num_v_heads]`, stored **as loaded** (the vb dtype); cast to F32 at
    /// use ([`gdn_kernel_inputs`] — the reference computes `g` in fp32
    /// always). A load-time cast would go stale under full-FT updates.
    a_log: Tensor,
    /// `[num_v_heads]`, stored **as loaded**; cast to F32 at use (added to
    /// the fp32 `a` projection).
    dt_bias: Tensor,
    /// Plain-`w` gated norm over `head_v` (weight in the activation dtype —
    /// the PR-1 convention pin).
    norm: RmsNormGated,
    out_proj: Proj,
    dims: GdnDims,
}

impl Qwen3_5GatedDeltaNet {
    fn load(cfg: &Qwen3_5TextConfig, vb: &VarBuilder, mode: &LoadMode) -> CandleResult<Self> {
        let dims = GdnDims::from_config(cfg);
        let h = cfg.hidden_size;
        Ok(Self {
            in_proj_qkv: mode.proj(vb, "in_proj_qkv", (dims.conv_dim, h), |t| t.gdn_qkv)?,
            in_proj_z: mode.proj(vb, "in_proj_z", (dims.value_dim, h), |t| t.gdn_z)?,
            in_proj_b: mode.proj(vb, "in_proj_b", (dims.num_v_heads, h), |t| t.gdn_b)?,
            in_proj_a: mode.proj(vb, "in_proj_a", (dims.num_v_heads, h), |t| t.gdn_a)?,
            // Stored AS FETCHED (no squeeze, no cast) — the full-FT
            // storage-sharing contract; see the field docs.
            conv_weight: vb
                .pp("conv1d")
                .get((dims.conv_dim, 1, dims.kernel), "weight")?,
            a_log: vb.get(dims.num_v_heads, "A_log")?,
            dt_bias: vb.get(dims.num_v_heads, "dt_bias")?,
            norm: RmsNormGated::new(vb.pp("norm").get(dims.head_v, "weight")?, cfg.rms_norm_eps),
            out_proj: mode.proj(vb, "out_proj", (h, dims.value_dim), |t| t.gdn_out)?,
            dims,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let mixed = self.in_proj_qkv.forward(x)?;
        let z =
            self.in_proj_z
                .forward(x)?
                .reshape((b, l, self.dims.num_v_heads, self.dims.head_v))?;
        let b_in = self.in_proj_b.forward(x)?;
        let a_in = self.in_proj_a.forward(x)?;

        // Causal depthwise conv + SiLU over q‖k‖v only (z/b/a bypass it).
        // The taps squeeze [conv_dim, 1, kernel] -> [conv_dim, kernel] HERE
        // (a view; grads flow through it to the full-FT var).
        let conv_in = mixed.transpose(1, 2)?.contiguous()?;
        let conv_out = causal_depthwise_conv1d(&conv_in, &self.conv_weight.squeeze(1)?, None)?;
        let mixed = conv_out.silu()?.transpose(1, 2)?.contiguous()?;

        let inp = gdn_kernel_inputs(&self.dims, &mixed, &b_in, &a_in, &self.a_log, &self.dt_bias)?;
        // The CHUNKED form: the only delta-rule form whose autograd tape
        // scales per chunk, not per token (the design-pass memory analysis) —
        // and the reference's own training dispatch.
        let (out, _state) = gated_delta_rule_chunked(
            &inp.query,
            &inp.key,
            &inp.value,
            &inp.g,
            &inp.beta,
            GDN_CHUNK_SIZE,
            None,
        )?;
        let out = self.norm.forward(&out, &z)?;
        let out = out.reshape((b, l, self.dims.value_dim))?;
        self.out_proj.forward(&out)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.in_proj_qkv.set_enabled(enabled);
        self.in_proj_z.set_enabled(enabled);
        self.in_proj_b.set_enabled(enabled);
        self.in_proj_a.set_enabled(enabled);
        self.out_proj.set_enabled(enabled);
    }

    /// Var order within the layer: `in_proj_qkv, in_proj_z, in_proj_b,
    /// in_proj_a, out_proj` (each contributing `[A, B]` when adapted).
    fn push_vars(&self, out: &mut Vec<Var>) {
        self.in_proj_qkv.push_vars(out);
        self.in_proj_z.push_vars(out);
        self.in_proj_b.push_vars(out);
        self.in_proj_a.push_vars(out);
        self.out_proj.push_vars(out);
    }
}

// ---------------------------------------------------------------------------
// Gated GQA full-attention layer
// ---------------------------------------------------------------------------

/// The static dims a full-attention layer derives from the config.
#[derive(Debug, Clone, Copy)]
struct AttnDims {
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rot_dim: usize,
}

impl AttnDims {
    fn from_config(cfg: &Qwen3_5TextConfig) -> Self {
        Self {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rot_dim: cfg.rotary_dim(),
        }
    }
}

/// The shared attention math from the doubled-`q_proj` output onward, over
/// already-projected tensors — used by both the grad side (no cache) and the
/// merged side (KV-cached): per-head `[query | gate]` split → QK-norm →
/// partial rope at `offset` → (cache append) → GQA repeat → SDPA with fp32
/// softmax → `* sigmoid(gate)` — returning the pre-`o_proj` context.
#[allow(clippy::too_many_arguments)]
fn gated_attention_core(
    dims: &AttnDims,
    qg: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_norm: &RmsNormZeroCentered,
    k_norm: &RmsNormZeroCentered,
    mask: Option<&Tensor>,
    rot: &RotaryTables,
    offset: usize,
    cache: Option<&mut ConcatKvCache>,
) -> CandleResult<Tensor> {
    let (b, l, _) = qg.dims3()?;
    let h = dims.num_heads;
    let d = dims.head_dim;
    let in_dtype = qg.dtype();

    // Doubled q_proj: per head the 2d outputs are [query(d) | gate(d)] —
    // chunk on the last dim of the (B, L, H, 2d) view (per-head interleaved,
    // NOT block [allQ | allG]).
    let qg = qg.reshape((b, l, h, 2 * d))?;
    let q = qg.narrow(D::Minus1, 0, d)?.contiguous()?;
    let gate = qg
        .narrow(D::Minus1, d, d)?
        .contiguous()?
        .reshape((b, l, h * d))?;

    // QK-norm (zero-centered, per head) BEFORE the transpose and BEFORE rope
    // (the reference order).
    let q = q_norm.forward(&q)?.transpose(1, 2)?; // [B, H, L, D]
    let k = k_norm
        .forward(&k.reshape((b, l, dims.num_kv_heads, d))?)?
        .transpose(1, 2)?;
    let v = v.reshape((b, l, dims.num_kv_heads, d))?.transpose(1, 2)?;

    // Partial rotate-half rope on the first rot_dim dims at absolute
    // positions [offset, offset + l).
    let (cos, sin) = rot.slice_at(offset, l)?;
    let q = rope_partial(&q.contiguous()?, &cos, &sin, dims.rot_dim)?;
    let k = rope_partial(&k.contiguous()?, &cos, &sin, dims.rot_dim)?;

    // Append the UN-repeated K/V when cached, then GQA-repeat what attention
    // consumes (the cache stays compact, the shipped order).
    let (k, v) = match cache {
        Some(c) => c.append(&k.contiguous()?, &v.contiguous()?)?,
        None => (k, v.contiguous()?),
    };
    let k = repeat_kv(&k, dims.num_kv_groups)?.contiguous()?;
    let v = repeat_kv(&v, dims.num_kv_groups)?.contiguous()?;

    // SDPA in the activation dtype with the fp32 softmax round-trip — the
    // reference eager path computes scores in the model dtype, softmaxes in
    // fp32, and casts the probabilities back before the value matmul.
    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    let probs = softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(in_dtype)?;
    let ctx = probs.matmul(&v)?;

    // Back to [B, L, H*D], then the output gate.
    let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, l, h * d))?;
    ctx.mul(&sigmoid(&gate)?)
}

/// One gated GQA full-attention layer (grad side, uncached).
#[derive(Debug)]
struct Qwen3_5Attention {
    /// The DOUBLED query projection `[num_heads * head_dim * 2, hidden]`.
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    q_norm: RmsNormZeroCentered,
    k_norm: RmsNormZeroCentered,
    dims: AttnDims,
}

impl Qwen3_5Attention {
    fn load(cfg: &Qwen3_5TextConfig, vb: &VarBuilder, mode: &LoadMode) -> CandleResult<Self> {
        let dims = AttnDims::from_config(cfg);
        let h = cfg.hidden_size;
        let kv_out = dims.num_kv_heads * dims.head_dim;
        let attn_out = dims.num_heads * dims.head_dim;
        Ok(Self {
            q_proj: mode.proj(vb, "q_proj", (attn_out * 2, h), |t| t.attn_q)?,
            k_proj: mode.proj(vb, "k_proj", (kv_out, h), |t| t.attn_k)?,
            v_proj: mode.proj(vb, "v_proj", (kv_out, h), |t| t.attn_v)?,
            o_proj: mode.proj(vb, "o_proj", (h, attn_out), |t| t.attn_o)?,
            q_norm: RmsNormZeroCentered::new(
                vb.pp("q_norm").get(dims.head_dim, "weight")?,
                cfg.rms_norm_eps,
            ),
            k_norm: RmsNormZeroCentered::new(
                vb.pp("k_norm").get(dims.head_dim, "weight")?,
                cfg.rms_norm_eps,
            ),
            dims,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let qg = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let ctx = gated_attention_core(
            &self.dims,
            &qg,
            &k,
            &v,
            &self.q_norm,
            &self.k_norm,
            mask,
            rot,
            0,
            None,
        )?;
        self.o_proj.forward(&ctx)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.q_proj.set_enabled(enabled);
        self.k_proj.set_enabled(enabled);
        self.v_proj.set_enabled(enabled);
        self.o_proj.set_enabled(enabled);
    }

    /// Var order within the layer: `q_proj, k_proj, v_proj, o_proj`.
    fn push_vars(&self, out: &mut Vec<Var>) {
        self.q_proj.push_vars(out);
        self.k_proj.push_vars(out);
        self.v_proj.push_vars(out);
        self.o_proj.push_vars(out);
    }
}

// ---------------------------------------------------------------------------
// MLP, layer, model
// ---------------------------------------------------------------------------

/// `SwiGLU` MLP; each projection may carry the adapter per the recipe.
#[derive(Debug)]
struct Qwen3_5Mlp {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
}

impl Qwen3_5Mlp {
    /// `inner` is the `SwiGLU` width: `intermediate_size` for a dense layer's
    /// MLP, `shared_expert_intermediate_size` for a sparse layer's shared
    /// expert (the two callers).
    fn load(
        cfg: &Qwen3_5TextConfig,
        inner: usize,
        vb: &VarBuilder,
        mode: &LoadMode,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = inner;
        Ok(Self {
            gate_proj: mode.proj(vb, "gate_proj", (i, h), |t| t.mlp_gate)?,
            up_proj: mode.proj(vb, "up_proj", (i, h), |t| t.mlp_up)?,
            down_proj: mode.proj(vb, "down_proj", (h, i), |t| t.mlp_down)?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = self.gate_proj.forward(x)?.silu()?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&lhs.mul(&rhs)?)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.gate_proj.set_enabled(enabled);
        self.up_proj.set_enabled(enabled);
        self.down_proj.set_enabled(enabled);
    }

    /// Var order: `gate_proj, up_proj, down_proj`.
    fn push_vars(&self, out: &mut Vec<Var>) {
        self.gate_proj.push_vars(out);
        self.up_proj.push_vars(out);
        self.down_proj.push_vars(out);
    }
}

/// A layer's token mixer (grad side).
#[derive(Debug)]
enum Mixer {
    Linear(Qwen3_5GatedDeltaNet),
    Full(Qwen3_5Attention),
}

/// A layer's feed-forward slot — the SECOND layer menu (M3′): the dense
/// members run a plain `SwiGLU` MLP, the `MoE` members the sparse block, in
/// EVERY layer (the family deletes the per-layer dense/sparse knobs).
#[derive(Debug)]
enum FeedForward {
    Dense(Qwen3_5Mlp),
    Sparse(Qwen3_5SparseMoe),
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(moe) => moe.forward(x),
        }
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        match self {
            Self::Dense(mlp) => mlp.set_adapter_enabled(enabled),
            Self::Sparse(moe) => moe.shared.set_adapter_enabled(enabled),
        }
    }

    fn push_vars(&self, out: &mut Vec<Var>) {
        match self {
            Self::Dense(mlp) => mlp.push_vars(out),
            Self::Sparse(moe) => moe.shared.push_vars(out),
        }
    }
}

/// The sparse `MoE` feed-forward (grad side): the oracle-pinned [`crate::moe`]
/// kernels over FROZEN routed weights, plus an adapter-aware shared expert.
///
/// Trainability is the locked `MoE`-`LoRA` policy (GSPO lock, 2026-06-12): the
/// router, the packed routed experts, and the scalar sigmoid gate are frozen —
/// adapting the router would make routing non-stationary during training
/// (precisely the instability sequence-level importance sampling exists to
/// dampen), and per-expert adapters on the packed 3-D weights are
/// out-of-recipe. [`LoraTargets`]' `mlp_*` flags bind the SHARED expert's
/// projections (the always-on `SwiGLU`), the sparse layer's counterpart of the
/// dense MLP.
#[derive(Debug)]
struct Qwen3_5SparseMoe {
    /// Router weight `[E, hidden]`, frozen.
    router: Tensor,
    /// Packed per-expert gate+up `[E, 2·moe_inter, hidden]`, frozen.
    gate_up: Tensor,
    /// Per-expert down `[E, hidden, moe_inter]`, frozen.
    down: Tensor,
    /// The shared expert — adapter-aware, width
    /// `shared_expert_intermediate_size`.
    shared: Qwen3_5Mlp,
    /// The shared expert's sigmoid gate `[1, hidden]`, frozen.
    shared_gate: Tensor,
    top_k: usize,
}

impl Qwen3_5SparseMoe {
    fn load(
        cfg: &Qwen3_5TextConfig,
        moe: MoeDims,
        vb: &VarBuilder,
        mode: &LoadMode,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let m = moe.moe_intermediate_size;
        // The CHECKPOINT layout is per-expert split linears
        // (`experts.{i}.gate_proj/up_proj/down_proj.weight`) — transformers
        // converts to/from the packed in-memory tensors at load/save
        // (`core_model_loading`: MergeModulelist(dim=0) + Concatenate(dim=1),
        // gate first). The same packing here: [e, 0..m, :] = gate_e,
        // [e, m.., :] = up_e — the orientation the kernel's `chunk(2)`
        // semantics (and its oracle gates) pin.
        let experts = vb.pp("experts");
        let mut gate_up_rows = Vec::with_capacity(moe.num_experts);
        let mut down_rows = Vec::with_capacity(moe.num_experts);
        for e in 0..moe.num_experts {
            let evb = experts.pp(e);
            let gate = evb.pp("gate_proj").get((m, h), "weight")?;
            let up = evb.pp("up_proj").get((m, h), "weight")?;
            gate_up_rows.push(Tensor::cat(&[&gate, &up], 0)?); // [2m, h]
            down_rows.push(evb.pp("down_proj").get((h, m), "weight")?); // [h, m]
        }
        let gate_up = Tensor::stack(&gate_up_rows, 0)?; // [E, 2m, h]
        let down = Tensor::stack(&down_rows, 0)?; // [E, h, m]
                                                  // Full-FT registers each PACKED tensor as ONE var: the backend's
                                                  // exclude rule passed the per-expert fetches through raw, because
                                                  // packing from per-expert vars would leave these tensors stale
                                                  // load-time derivations (`cat`/`stack` copy storage — the `Var::set`
                                                  // contract in [`crate::full_ft`]). Hit-expert gradients land on the
                                                  // packed var through the kernel's index/narrow autograd, zeros
                                                  // elsewhere, so the grad-coverage canary stays satisfiable.
        let (gate_up, down) = match mode {
            LoadMode::Lora { .. } => (gate_up, down),
            LoadMode::FullFt { registry } => (
                registry.register(&format!("{}.gate_up_packed", experts.prefix()), &gate_up)?,
                registry.register(&format!("{}.down_packed", experts.prefix()), &down)?,
            ),
        };
        Ok(Self {
            router: vb.pp("gate").get((moe.num_experts, h), "weight")?,
            gate_up,
            down,
            shared: Qwen3_5Mlp::load(
                cfg,
                moe.shared_expert_intermediate_size,
                &vb.pp("shared_expert"),
                mode,
            )?,
            shared_gate: vb.pp("shared_expert_gate").get((1, h), "weight")?,
            top_k: moe.top_k,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // The shared expert runs adapter-aware on the layer geometry; the
        // composition itself is shared with the merged decoder.
        let shared = self.shared.forward(x)?;
        sparse_ffn_compose(
            x,
            &self.router,
            &self.gate_up,
            &self.down,
            &shared,
            &self.shared_gate,
            self.top_k,
        )
    }
}

/// The sparse feed-forward composition both decoders share: routed experts
/// (the [`crate::moe`] kernels) plus the sigmoid-gated `shared_out`, over
/// `x`/`shared_out` of shape `[batch, seq, hidden]`.
///
/// Faithful to `Qwen3_5MoeSparseMoeBlock.forward` with the shared-expert
/// `SwiGLU` factored out to the caller (it is the adapter-aware half on the
/// grad side and the merged half in the cached decoder; the reference
/// computes it on the same flattened tokens — `frozen_linear`/`LoRA` matmuls
/// flatten leading dims identically).
fn sparse_ffn_compose(
    x: &Tensor,
    router: &Tensor,
    gate_up: &Tensor,
    down: &Tensor,
    shared_out: &Tensor,
    shared_gate: &Tensor,
    top_k: usize,
) -> CandleResult<Tensor> {
    let (b, t, h) = x.dims3()?;
    let flat = x.reshape((b * t, h))?;
    let (_logits, scores, indices) = crate::moe::topk_router(&flat, router, top_k)?;
    let routed = crate::moe::moe_experts(&flat, gate_up, down, &indices, &scores)?;
    let gate = sigmoid(&flat.matmul(&shared_gate.t()?)?)?; // [n, 1]
    let shared = shared_out.reshape((b * t, h))?;
    let out = (routed + gate.broadcast_mul(&shared)?)?;
    out.reshape((b, t, h))
}

/// One decoder layer: pre-norm mixer + pre-norm feed-forward (dense `SwiGLU`
/// or the sparse `MoE` block, per the config's family branch), both residual,
/// zero-centered norms.
#[derive(Debug)]
struct Qwen3_5Layer {
    ln1: RmsNormZeroCentered,
    mixer: Mixer,
    ln2: RmsNormZeroCentered,
    mlp: FeedForward,
}

impl Qwen3_5Layer {
    fn load(
        cfg: &Qwen3_5TextConfig,
        kind: LayerType,
        vb: &VarBuilder,
        mode: &LoadMode,
    ) -> CandleResult<Self> {
        let mixer = match kind {
            LayerType::LinearAttention => Mixer::Linear(Qwen3_5GatedDeltaNet::load(
                cfg,
                &vb.pp("linear_attn"),
                mode,
            )?),
            LayerType::FullAttention => {
                Mixer::Full(Qwen3_5Attention::load(cfg, &vb.pp("self_attn"), mode)?)
            }
        };
        let mlp_vb = vb.pp("mlp");
        let mlp = match cfg.moe() {
            None => {
                let inner = cfg.intermediate_size.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "qwen3_5 layer: dense member without intermediate_size (validate \
                         the config before loading)"
                            .to_string(),
                    )
                })?;
                FeedForward::Dense(Qwen3_5Mlp::load(cfg, inner, &mlp_vb, mode)?)
            }
            Some(moe) => FeedForward::Sparse(Qwen3_5SparseMoe::load(cfg, moe, &mlp_vb, mode)?),
        };
        Ok(Self {
            ln1: RmsNormZeroCentered::new(
                vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?,
                cfg.rms_norm_eps,
            ),
            mixer,
            ln2: RmsNormZeroCentered::new(
                vb.pp("post_attention_layernorm")
                    .get(cfg.hidden_size, "weight")?,
                cfg.rms_norm_eps,
            ),
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = match &self.mixer {
            // Linear attention is causal by construction: no mask, no rope.
            Mixer::Linear(gdn) => gdn.forward(&h)?,
            Mixer::Full(attn) => attn.forward(&h, mask, rot)?,
        };
        let x = (x + &h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        match &mut self.mixer {
            Mixer::Linear(gdn) => gdn.set_adapter_enabled(enabled),
            Mixer::Full(attn) => attn.set_adapter_enabled(enabled),
        }
        self.mlp.set_adapter_enabled(enabled);
    }

    /// Var order within the layer: mixer projections first (in their fixed
    /// order), then the feed-forward's (the dense MLP's, or on sparse layers
    /// the shared expert's — same gate/up/down order).
    fn push_vars(&self, out: &mut Vec<Var>) {
        match &self.mixer {
            Mixer::Linear(gdn) => gdn.push_vars(out),
            Mixer::Full(attn) => attn.push_vars(out),
        }
        self.mlp.push_vars(out);
    }
}

/// A grad-bearing, uncached `qwen3_5` text forward with a configurable
/// [`LoraTargets`] adapter recipe — the third [`GradModel`] implementor, and
/// the first hybrid (`GatedDeltaNet` + gated GQA) one.
///
/// Loaded from the same safetensors as the HF reference (the
/// `model.language_model.*` multimodal layout every family checkpoint ships);
/// vision/MTP tensors are never requested. In the default (`LoRA`) mode the
/// base weights are frozen [`Tensor`]s and only the `LoRA` factors are
/// trainable [`Var`]s, in a deterministic layer-major order (the positional
/// checkpoint contract). In **full fine-tuning** mode
/// ([`load_full_ft`](Self::load_full_ft)) every base weight is a trainable
/// var's inner tensor and there are no adapters at all — same forward, same
/// order convention (the model's own load order).
#[derive(Debug)]
pub struct Qwen3_5GradModel {
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<Qwen3_5Layer>,
    norm: RmsNormZeroCentered,
    rot: RotaryTables,
    hidden: usize,
    max_position: usize,
    device: Device,
    targets: LoraTargets,
    adapter_enabled: bool,
    remat: bool,
    /// `Some` iff loaded via [`load_full_ft`](Self::load_full_ft): every base
    /// weight var in registry (load) order — the full-FT positional
    /// checkpoint contract. Doubles as the mode flag.
    full_ft_vars: Option<Vec<Var>>,
    /// The pending checkpointed-forward tape — see
    /// [`QwenGradModel`](crate::qwen::QwenGradModel)'s field of the same name
    /// (same contract: one tape per forward, one backward per tape, `!Sync`).
    tape: RefCell<Option<RematTape>>,
}

impl Qwen3_5GradModel {
    /// Load with the default (industrial) [`LoraTargets`] recipe and the
    /// adapter in the base weights' dtype.
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets).
    pub fn load(
        cfg: &Qwen3_5Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
    ) -> CandleResult<Self> {
        Self::load_with_targets(cfg, vb, rank, alpha, vb.dtype(), LoraTargets::default())
    }

    /// Load with the default recipe but the trainable adapter held in
    /// `adapter_dtype` — the bf16-base / F32-adapter split (see
    /// [`crate::lora::LoraLinear::with_adapter_dtype`]).
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets).
    pub fn load_with_adapter_dtype(
        cfg: &Qwen3_5Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        Self::load_with_targets(cfg, vb, rank, alpha, adapter_dtype, LoraTargets::default())
    }

    /// Load the model from `vb` (rooted at the **checkpoint root** — the
    /// loader applies the `model.language_model.*` prefix itself), attaching
    /// the `LoRA` adapter per `targets`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the config fails validation, `targets`
    /// selects nothing (an untrainable model), a tensor is missing or
    /// mis-shaped, or the adapter factors cannot be allocated.
    pub fn load_with_targets(
        cfg: &Qwen3_5Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        targets: LoraTargets,
    ) -> CandleResult<Self> {
        cfg.validate()?;
        // LoRA-mode-only bail: a full-FT load trains every base weight and
        // carries the deliberately empty recipe instead.
        if !targets.any() {
            bail!(
                "Qwen3_5GradModel: LoraTargets selects no projection — the model would have \
                 no trainable parameters"
            );
        }
        Self::load_inner(
            cfg,
            vb,
            &LoadMode::Lora {
                rank,
                alpha,
                adapter_dtype,
                targets,
            },
        )
    }

    /// Load in **full fine-tuning** mode: every base weight the load fetches
    /// becomes a trainable [`Var`] (via the internal `full_ft` var-registry
    /// `VarBuilder` backend, built over `tensors` — pass the checkpoint map
    /// from [`tensors_from_pretrained`]); there are no adapters. `LoRA` stays
    /// the default mode — this is the opt-in.
    ///
    /// Contract notes (each gated):
    /// - [`trainable_vars`](Self::trainable_vars) returns the registry in
    ///   **load order** — the positional checkpoint contract;
    ///   [`lora_recipe`](GradModel::lora_recipe) reports `"full-ft"` (plus
    ///   the feed-forward menu tag), so a resume cross-check catches a
    ///   `LoRA`↔full-FT checkpoint confusion loudly.
    /// - On `MoE` members the packed routed tensors are ONE var each and the
    ///   **router trains too** — full-FT moves routing during training, which
    ///   is exactly why GSPO (sequence-level importance sampling) is the
    ///   pinned recipe for `MoE` runs (the M3′ lock).
    /// - [`merged_decoder`](Self::merged_decoder) **deep-copies** every
    ///   weight (one full-model copy per rebuild): the vars' storage mutates
    ///   in place under optimizer steps, so a storage-sharing snapshot would
    ///   silently track them.
    /// - [`set_adapter_enabled`](Self::set_adapter_enabled) is a no-op (there
    ///   is no frozen base policy); the eval base-vs-trained comparison is
    ///   unavailable and fails loud in [`crate::eval::evaluate`].
    /// - Activation checkpointing is **not yet supported** in this mode (the
    ///   forward fails loud; see
    ///   [`set_activation_checkpointing`](Self::set_activation_checkpointing)).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the config fails validation, a tensor is
    /// missing or mis-shaped, or a var allocation fails.
    pub fn load_full_ft(
        cfg: &Qwen3_5Config,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        cfg.validate()?;
        let (vb, registry) = crate::full_ft::registry_varbuilder(
            tensors,
            dtype,
            device,
            // Per-expert checkpoint tensors pass through raw: the sparse
            // loader packs them and registers the packed tensors itself.
            |name| name.contains(".mlp.experts."),
        );
        let mut model = Self::load_inner(
            cfg,
            &vb,
            &LoadMode::FullFt {
                registry: &registry,
            },
        )?;
        let vars = registry.vars()?;
        if vars.is_empty() {
            bail!("Qwen3_5GradModel::load_full_ft: the load registered no trainable vars");
        }
        model.full_ft_vars = Some(vars);
        Ok(model)
    }

    /// Whether this model was loaded in full-fine-tuning mode.
    #[must_use]
    pub fn is_full_ft(&self) -> bool {
        self.full_ft_vars.is_some()
    }

    /// The shared load walk (both modes). Fetch order is deterministic —
    /// embed, (untied head), layers 0..N (mixer, feed-forward, then the two
    /// layer norms), final norm — and in full-FT mode that order **is** the
    /// positional checkpoint contract.
    fn load_inner(cfg: &Qwen3_5Config, vb: &VarBuilder, mode: &LoadMode) -> CandleResult<Self> {
        let t = &cfg.text_config;
        let root = vb.pp(CKPT_PREFIX);
        let embed = root
            .pp("embed_tokens")
            .get((t.vocab_size, t.hidden_size), "weight")?;
        let lm_head = if t.tie_word_embeddings {
            None
        } else {
            // Untied checkpoints keep the head OUTSIDE the language_model
            // prefix (the ForCausalLM layout).
            Some(
                vb.pp("lm_head")
                    .get((t.vocab_size, t.hidden_size), "weight")?,
            )
        };
        let kinds = t.resolved_layer_types();
        let layers_vb = root.pp("layers");
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for (i, kind) in kinds.iter().enumerate() {
            layers.push(Qwen3_5Layer::load(t, *kind, &layers_vb.pp(i), mode)?);
        }
        Ok(Self {
            embed,
            lm_head,
            layers,
            norm: RmsNormZeroCentered::new(
                root.pp("norm").get(t.hidden_size, "weight")?,
                t.rms_norm_eps,
            ),
            // Partial rotary: the table is just NARROWER (width = rot_dim/2
            // freqs over denominators of rot_dim) — the construction the PR-1
            // real-geometry rope oracle gate pinned at the 0.8B geometry.
            rot: RotaryTables::new(
                t.rotary_dim(),
                t.rope_parameters.rope_theta,
                t.max_position_embeddings,
                vb.dtype(),
                vb.device(),
            )?,
            hidden: t.hidden_size,
            max_position: t.max_position_embeddings,
            device: vb.device().clone(),
            targets: match mode {
                LoadMode::Lora { targets, .. } => *targets,
                LoadMode::FullFt { .. } => LoraTargets::none(),
            },
            adapter_enabled: true,
            remat: false,
            full_ft_vars: None,
            tape: RefCell::new(None),
        })
    }

    /// The [`LoraTargets`] recipe this model was loaded with (for logging and
    /// checkpoint metadata — see [`LoraTargets::canonical`]). A full-FT model
    /// reports [`LoraTargets::none`] here (it has no adapters); its mode
    /// string lives in [`lora_recipe`](GradModel::lora_recipe) (`"full-ft"`).
    #[must_use]
    pub fn lora_targets(&self) -> LoraTargets {
        self.targets
    }

    /// Full-sequence logits `[batch, seq, vocab]` for `input_ids`
    /// (`[batch, seq]`, `u32`): uncached, tape-bearing, every position
    /// returned. Inputs are unpadded by contract (see the module docs).
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails (e.g. a shape mismatch).
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_window(input_ids, None)
    }

    /// The narrowed scoring forward: the full layer walk with the final
    /// norm + head applied to the `(start, len)` window alone — same scheme
    /// and contract as
    /// [`QwenGradModel::forward_narrowed`](crate::qwen::QwenGradModel::forward_narrowed).
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
    /// the full causal mask (`None` at seq-len 1). The mask is built in the
    /// ACTIVATION dtype: unlike the Llama eager path (which force-casts
    /// scores to F32), the `qwen3_5` reference adds the mask to model-dtype
    /// scores and upcasts only inside the softmax.
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
            Some(causal_mask(l, h.dtype(), &self.device)?)
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

    /// The checkpointed forward — boundary [`Var`] per layer plus the tail;
    /// same scheme and contract as
    /// [`QwenGradModel`](crate::qwen::QwenGradModel)'s, including the
    /// loss-tape window narrow (the tail boundary stays full-width).
    /// Layer-type-agnostic: a segment is one [`Qwen3_5Layer`] whichever mixer
    /// (`GatedDeltaNet` or gated GQA) it holds — both are pure `x -> y` over
    /// the boundary.
    fn forward_remat(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        // Full-FT × checkpointing is not wired yet: the boundary tape starts
        // AFTER the embedding lookup, and `stitched_backward` discards the
        // first boundary's cotangent — the embedding gradient would be
        // silently dropped (and in LoRA mode never existed to drop). Fail
        // loud rather than train wrong. Flagged follow-up: return the final
        // cotangent and fold it through an embedding re-lookup.
        if self.is_full_ft() {
            bail!(
                "Qwen3_5GradModel: activation checkpointing is not supported in full \
                 fine-tuning mode yet — the boundary tape would silently drop the \
                 embedding gradient. Turn checkpointing off for full-FT runs."
            );
        }
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

    /// Detached full-sequence logits with a rolling boundary detach — same
    /// values as [`forward`](Self::forward) at a one-layer peak footprint,
    /// never capturing a tape. For the value-only scorings.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails.
    pub fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_detached_window(input_ids, None)
    }

    /// The narrowed detached scoring forward: rolling boundary detach plus
    /// the windowed tail (see [`GradModel::forward_detached_narrowed`]).
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

    /// Back-propagate a loss built from this model's logits — plain
    /// `loss.backward()` normally, tape-stitched under
    /// [activation checkpointing](Self::set_activation_checkpointing); same
    /// contract as [`QwenGradModel::backward`](crate::qwen::QwenGradModel::backward).
    ///
    /// # Errors
    ///
    /// Returns a candle error on any backward failure or tape-contract
    /// violation (no pending forward, foreign loss, adapter flipped).
    pub fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        if !self.remat {
            return loss.backward();
        }
        let Some(tape) = self.tape.borrow_mut().take() else {
            bail!(
                "Qwen3_5GradModel::backward: activation checkpointing is on but no checkpointed \
                 forward is pending (each forward's tape is consumed by exactly one backward)"
            )
        };
        if tape.adapter_enabled() != self.adapter_enabled {
            bail!(
                "Qwen3_5GradModel::backward: the adapter toggle flipped between the checkpointed \
                 forward and its backward — the recompute would rebuild different values"
            )
        }
        let l = tape.first_boundary_dims().map(|d| d[1]).unwrap_or_default();
        let mask = if l <= 1 {
            None
        } else {
            Some(causal_mask(l, self.embed.dtype(), &self.device)?)
        };
        stitched_backward(loss, &tape, &self.trainable_vars(), |i, x| {
            self.layers[i].forward(x, mask.as_ref(), &self.rot)
        })
    }

    /// Turn **activation checkpointing** on or off (default: off) — same
    /// trade and contract as
    /// [`QwenGradModel::set_activation_checkpointing`](crate::qwen::QwenGradModel::set_activation_checkpointing).
    ///
    /// **Full-FT caveat:** checkpointing is not yet supported in full
    /// fine-tuning mode — the next grad-bearing forward fails loud (see
    /// [`load_full_ft`](Self::load_full_ft)).
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

    /// Enable/disable the `LoRA` adapter on every targeted projection
    /// (disabled == the frozen base model == the GRPO reference policy).
    ///
    /// **No-op in full-FT mode** (per the [`GradModel`] contract): there are
    /// no adapters and no frozen base policy to toggle back to, so the flag
    /// stays `true` — callers that need the toggle (eval's base-vs-trained
    /// comparison) observe it did not take and fail loud.
    pub fn set_adapter_enabled(&mut self, enabled: bool) {
        if self.is_full_ft() {
            self.adapter_enabled = true;
            return;
        }
        for layer in &mut self.layers {
            layer.set_adapter_enabled(enabled);
        }
        self.adapter_enabled = enabled;
    }

    /// All trainable [`Var`]s in a **deterministic** order — the positional
    /// checkpoint contract.
    ///
    /// `LoRA` mode: layer-major; within a layer the mixer's projections first
    /// (`q,k,v,o` / `in_proj_qkv,z,b,a,out_proj`), then the MLP's
    /// (`gate,up,down`); each adapted projection contributes `[A, B]`. A pure
    /// function of (config, [`LoraTargets`]).
    ///
    /// Full-FT mode: every base weight var, in registry (= load) order — a
    /// pure function of the config alone (see
    /// [`load_full_ft`](Self::load_full_ft)).
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        if let Some(vars) = &self.full_ft_vars {
            return vars.clone();
        }
        let mut vars = Vec::new();
        for layer in &self.layers {
            layer.push_vars(&mut vars);
        }
        vars
    }

    /// The device the weights live on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Snapshot the **current** effective weights into a cached, grad-free
    /// [`Qwen3_5MergedDecoder`] for fast incremental rollout.
    ///
    /// Same contract as the Qwen/Llama merged decoders (tape-detached value
    /// snapshot; rebuild after any optimizer step or adapter toggle), but the
    /// per-layer state is hybrid: KV cache on full-attention layers, conv
    /// state + fp32 delta-rule state on linear-attention layers.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a merged-weight build fails.
    pub fn merged_decoder(&self) -> CandleResult<Qwen3_5MergedDecoder> {
        Qwen3_5MergedDecoder::from_model(self)
    }
}

/// The [`GradModel`] seam over [`Qwen3_5GradModel`]: pure delegation.
impl GradModel for Qwen3_5GradModel {
    type Decoder = Qwen3_5MergedDecoder;

    fn device(&self) -> &Device {
        Qwen3_5GradModel::device(self)
    }

    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        Qwen3_5GradModel::forward(self, input_ids)
    }

    fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        Qwen3_5GradModel::forward_detached(self, input_ids)
    }

    fn forward_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        Qwen3_5GradModel::forward_narrowed(self, input_ids, start, len)
    }

    fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        Qwen3_5GradModel::forward_detached_narrowed(self, input_ids, start, len)
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        Qwen3_5GradModel::backward(self, loss)
    }

    fn trainable_vars(&self) -> Vec<Var> {
        Qwen3_5GradModel::trainable_vars(self)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        Qwen3_5GradModel::set_adapter_enabled(self, enabled);
    }

    fn merged_decoder(&self) -> CandleResult<Qwen3_5MergedDecoder> {
        Qwen3_5GradModel::merged_decoder(self)
    }

    fn lora_recipe(&self) -> Option<String> {
        // The feed-forward menu is part of the recipe: a dense and an MoE
        // model with aliasing widths produce positionally IDENTICAL var lists
        // (the mlp flags bind the layer MLP vs the shared expert), so the
        // manifest cross-check must see the menu to catch a dense-vs-MoE
        // checkpoint confusion. Dense models keep the historical string.
        // Full-FT reports its mode instead of a targets string — the same
        // cross-check then catches a LoRA↔full-FT checkpoint confusion.
        let menu = match self.layers.first().map(|l| &l.mlp) {
            Some(FeedForward::Sparse(_)) => "|ffn:moe",
            _ => "",
        };
        if self.is_full_ft() {
            return Some(format!("full-ft{menu}"));
        }
        Some(format!("{}{menu}", self.targets.canonical()))
    }

    fn has_adapters(&self) -> bool {
        !self.is_full_ft()
    }
}

// ---------------------------------------------------------------------------
// Merged (cached, grad-free) decoder
// ---------------------------------------------------------------------------

/// Snapshot a non-projection weight for the merged decoder: a cheap
/// storage-sharing `clone()` in `LoRA` mode (the tensor is frozen — it can
/// never change under it), a **fresh-storage deep copy** in full-FT mode
/// (the tensor is a trainable var's inner storage, mutated in place by every
/// optimizer step — a share would silently track training instead of
/// snapshotting a value; the mutation gate pins this).
fn snap(w: &Tensor, full_ft: bool) -> CandleResult<Tensor> {
    if full_ft {
        Ok(w.copy()?.detach())
    } else {
        Ok(w.clone())
    }
}

/// [`snap`] for a [`Proj`]: the adapter-merging snapshot, deep in full-FT.
fn snap_proj(p: &Proj, full_ft: bool) -> CandleResult<Tensor> {
    if full_ft {
        p.merged_weight_deep()
    } else {
        p.merged_weight()
    }
}

/// A `GatedDeltaNet` layer over merged weights with its recurrent state — the
/// grad-free mirror of [`Qwen3_5GatedDeltaNet`].
///
/// State (absent until the first forward; reset clears it):
/// - `conv_state`: `[batch, conv_dim, kernel]` in the **model dtype** — the
///   last `kernel` **pre-`SiLU` conv inputs** (the projection outputs,
///   left-zero-padded when fewer tokens have been consumed), exactly what the
///   reference caches;
/// - `s_state`: `[batch, num_v_heads, head_k, head_v]` **F32** — the
///   delta-rule state matrix.
#[derive(Debug)]
struct MergedGdn {
    in_qkv_w: Tensor,
    in_z_w: Tensor,
    in_b_w: Tensor,
    in_a_w: Tensor,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: RmsNormGated,
    out_w: Tensor,
    dims: GdnDims,
    conv_state: Option<Tensor>,
    s_state: Option<Tensor>,
}

impl MergedGdn {
    fn from_layer(l: &Qwen3_5GatedDeltaNet, full_ft: bool) -> CandleResult<Self> {
        Ok(Self {
            in_qkv_w: snap_proj(&l.in_proj_qkv, full_ft)?,
            in_z_w: snap_proj(&l.in_proj_z, full_ft)?,
            in_b_w: snap_proj(&l.in_proj_b, full_ft)?,
            in_a_w: snap_proj(&l.in_proj_a, full_ft)?,
            conv_weight: snap(&l.conv_weight, full_ft)?,
            a_log: snap(&l.a_log, full_ft)?,
            dt_bias: snap(&l.dt_bias, full_ft)?,
            norm: if full_ft {
                l.norm.deep_copy()?
            } else {
                l.norm.clone()
            },
            out_w: snap_proj(&l.out_proj, full_ft)?,
            dims: l.dims,
            conv_state: None,
            s_state: None,
        })
    }

    /// One cached forward chunk. Dispatch mirrors the reference exactly:
    /// single token **with** previous state → per-step conv update + the
    /// sequential recurrent rule; anything else (prefill at 0, or a
    /// multi-token continuation at an offset — the v5.7.0-fixed reference
    /// path) → conv with the cached left-context + the **chunked** rule from
    /// the stored state.
    fn forward(&mut self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let dims = self.dims;
        let mixed = frozen_linear(x, &self.in_qkv_w)?;
        let z = frozen_linear(x, &self.in_z_w)?.reshape((b, l, dims.num_v_heads, dims.head_v))?;
        let b_in = frozen_linear(x, &self.in_b_w)?;
        let a_in = frozen_linear(x, &self.in_a_w)?;

        let conv_in = mixed.transpose(1, 2)?.contiguous()?; // [B, C, L]
        let has_state = self.s_state.is_some();
        if has_state != self.conv_state.is_some() {
            bail!("MergedGdn: conv_state and recurrent state out of sync (reset_cache to recover)");
        }
        // The conv left-context is the newest kernel-1 cached columns (the
        // oldest column was the "current token" of the previous step and is
        // outside every new window); the new cache is the last `kernel`
        // columns of [cached, new] — left-zero-padded at a fresh start.
        // Taps squeeze [conv_dim, 1, kernel] -> [conv_dim, kernel], mirroring
        // the grad side's per-forward view.
        let conv_w = self.conv_weight.squeeze(1)?;
        let (conv_out, new_conv_state) = match &self.conv_state {
            Some(cs) => {
                let ctx = cs.narrow(D::Minus1, 1, dims.kernel - 1)?.contiguous()?;
                let out = causal_depthwise_conv1d(&conv_in, &conv_w, Some(&ctx))?;
                let cat = Tensor::cat(&[cs, &conv_in], D::Minus1)?;
                let w = cat.dim(D::Minus1)?;
                let ncs = cat
                    .narrow(D::Minus1, w - dims.kernel, dims.kernel)?
                    .contiguous()?;
                (out, ncs)
            }
            None => {
                let out = causal_depthwise_conv1d(&conv_in, &conv_w, None)?;
                let ncs = if l >= dims.kernel {
                    conv_in
                        .narrow(D::Minus1, l - dims.kernel, dims.kernel)?
                        .contiguous()?
                } else {
                    conv_in.pad_with_zeros(D::Minus1, dims.kernel - l, 0)?
                };
                (out, ncs)
            }
        };
        let mixed = conv_out.silu()?.transpose(1, 2)?.contiguous()?;

        let inp = gdn_kernel_inputs(&dims, &mixed, &b_in, &a_in, &self.a_log, &self.dt_bias)?;
        let s0 = self.s_state.as_ref();
        let (out, s_new) = if l == 1 && has_state {
            gated_delta_rule_recurrent(&inp.query, &inp.key, &inp.value, &inp.g, &inp.beta, s0)?
        } else {
            gated_delta_rule_chunked(
                &inp.query,
                &inp.key,
                &inp.value,
                &inp.g,
                &inp.beta,
                GDN_CHUNK_SIZE,
                s0,
            )?
        };
        self.conv_state = Some(new_conv_state);
        self.s_state = Some(s_new);

        let out = self.norm.forward(&out, &z)?;
        let out = out.reshape((b, l, dims.value_dim))?;
        frozen_linear(&out, &self.out_w)
    }

    fn reset(&mut self) {
        self.conv_state = None;
        self.s_state = None;
    }
}

/// A full-attention layer over merged weights with an incremental KV cache —
/// the grad-free mirror of [`Qwen3_5Attention`].
#[derive(Debug)]
struct MergedAttention {
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    q_norm: RmsNormZeroCentered,
    k_norm: RmsNormZeroCentered,
    dims: AttnDims,
    /// Un-repeated K/V, concatenated on the sequence axis (dim 2).
    cache: ConcatKvCache,
}

impl MergedAttention {
    fn from_layer(l: &Qwen3_5Attention, full_ft: bool) -> CandleResult<Self> {
        Ok(Self {
            q_w: snap_proj(&l.q_proj, full_ft)?,
            k_w: snap_proj(&l.k_proj, full_ft)?,
            v_w: snap_proj(&l.v_proj, full_ft)?,
            o_w: snap_proj(&l.o_proj, full_ft)?,
            q_norm: if full_ft {
                l.q_norm.deep_copy()?
            } else {
                l.q_norm.clone()
            },
            k_norm: if full_ft {
                l.k_norm.deep_copy()?
            } else {
                l.k_norm.clone()
            },
            dims: l.dims,
            cache: ConcatKvCache::new(2),
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let qg = frozen_linear(x, &self.q_w)?;
        let k = frozen_linear(x, &self.k_w)?;
        let v = frozen_linear(x, &self.v_w)?;
        let ctx = gated_attention_core(
            &self.dims,
            &qg,
            &k,
            &v,
            &self.q_norm,
            &self.k_norm,
            mask,
            rot,
            offset,
            Some(&mut self.cache),
        )?;
        frozen_linear(&ctx, &self.o_w)
    }
}

/// A merged layer's token mixer and its state.
#[derive(Debug)]
enum MergedMixer {
    Linear(MergedGdn),
    Full(MergedAttention),
}

/// One merged decoder layer.
#[derive(Debug)]
struct MergedLayer {
    ln1: RmsNormZeroCentered,
    mixer: MergedMixer,
    ln2: RmsNormZeroCentered,
    mlp: MergedFeedForward,
}

/// `SwiGLU` MLP over merged weights.
#[derive(Debug)]
struct MergedMlp {
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
}

impl MergedMlp {
    fn from_layer(l: &Qwen3_5Mlp, full_ft: bool) -> CandleResult<Self> {
        Ok(Self {
            gate_w: snap_proj(&l.gate_proj, full_ft)?,
            up_w: snap_proj(&l.up_proj, full_ft)?,
            down_w: snap_proj(&l.down_proj, full_ft)?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = frozen_linear(x, &self.gate_w)?.silu()?;
        let rhs = frozen_linear(x, &self.up_w)?;
        frozen_linear(&lhs.mul(&rhs)?, &self.down_w)
    }
}

/// The merged feed-forward menu — the cached twin of [`FeedForward`].
#[derive(Debug)]
enum MergedFeedForward {
    Dense(MergedMlp),
    Sparse(MergedSparseMoe),
}

impl MergedFeedForward {
    fn from_layer(ff: &FeedForward, full_ft: bool) -> CandleResult<Self> {
        Ok(match ff {
            FeedForward::Dense(mlp) => Self::Dense(MergedMlp::from_layer(mlp, full_ft)?),
            FeedForward::Sparse(moe) => Self::Sparse(MergedSparseMoe {
                // In full-FT the routed side (router + packed experts) and
                // the sigmoid gate are vars too — the deep snap covers them.
                router: snap(&moe.router, full_ft)?,
                gate_up: snap(&moe.gate_up, full_ft)?,
                down: snap(&moe.down, full_ft)?,
                shared: MergedMlp::from_layer(&moe.shared, full_ft)?,
                shared_gate: snap(&moe.shared_gate, full_ft)?,
                top_k: moe.top_k,
            }),
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(moe) => moe.forward(x),
        }
    }
}

/// The sparse `MoE` feed-forward over merged weights: the routed side is
/// frozen (identical tensors to the grad side); only the shared expert
/// carries merged adapter weights. Stateless — routing is per-token, so the
/// cached decoder needs no `MoE` state.
#[derive(Debug)]
struct MergedSparseMoe {
    router: Tensor,
    gate_up: Tensor,
    down: Tensor,
    shared: MergedMlp,
    shared_gate: Tensor,
    top_k: usize,
}

impl MergedSparseMoe {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let shared = self.shared.forward(x)?;
        sparse_ffn_compose(
            x,
            &self.router,
            &self.gate_up,
            &self.down,
            &shared,
            &self.shared_gate,
            self.top_k,
        )
    }
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
        let h = match &mut self.mixer {
            MergedMixer::Linear(gdn) => gdn.forward(&h)?,
            MergedMixer::Full(attn) => attn.forward(&h, offset, mask, rot)?,
        };
        let x = (x + &h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }
}

/// A cached, **grad-free** `qwen3_5` decoder over weights with the `LoRA`
/// adapter already folded in — the fast rollout twin of [`Qwen3_5GradModel`],
/// and its [`CachedDecoder`].
///
/// The per-layer state is **hybrid**: full-attention layers hold a KV cache;
/// linear-attention layers hold the conv state (model dtype) and the fp32
/// delta-rule state matrix. Because the KV layers' cache length no longer
/// describes the whole model, the decoder keeps its own `consumed` token
/// counter as the **single source of truth** for the offset contract,
/// cross-checked against the first KV layer when one exists.
///
/// Same lifecycle contract as the Qwen/Llama merged decoders: value snapshot
/// (rebuild after any optimizer step or adapter toggle), `offset` must equal
/// the tokens already consumed (fail-loud), `reset_cache` starts a fresh
/// sequence. It holds **no** [`Var`] and records no autograd tape.
///
/// **Error contract (hybrid-state caveat):** the offset / position / KV
/// guards all run **before** any layer state advances, so a rejected call
/// leaves the decoder untouched. But layer state commits layer by layer
/// during a forward, so after any **other** mid-forward error (an allocator
/// or device failure inside a layer) the state is unspecified — call
/// [`reset_cache`](Self::reset_cache) before reusing the decoder; do not
/// retry the chunk. (The generic policy builds a fresh decoder per generate
/// call and propagates errors, so it never continues past one.)
#[derive(Debug)]
pub struct Qwen3_5MergedDecoder {
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<MergedLayer>,
    norm: RmsNormZeroCentered,
    rot: RotaryTables,
    hidden: usize,
    /// `max_position_embeddings` — enforced on `consumed + chunk_len` BEFORE
    /// any layer state advances. Without this precheck the rope-table bound
    /// would fire mid-stack at the first full-attention layer, AFTER the
    /// preceding linear layers had already committed the failed chunk into
    /// their conv/recurrent state (and an all-linear model would have no
    /// position bound at all).
    max_position: usize,
    device: Device,
    /// Tokens already consumed — the decoder-level offset truth (the hybrid
    /// state means no single layer cache can carry it).
    consumed: usize,
}

impl Qwen3_5MergedDecoder {
    /// Snapshot a [`Qwen3_5GradModel`]'s current effective weights. Private —
    /// callers go through [`Qwen3_5GradModel::merged_decoder`].
    fn from_model(model: &Qwen3_5GradModel) -> CandleResult<Self> {
        // Full-FT: every weight is a trainable var's storage — the snapshot
        // deep-copies (one full-model copy per rebuild, the documented cost)
        // so an optimizer step cannot mutate a built decoder under the
        // rollout. LoRA mode keeps the cheap storage-sharing clones of the
        // frozen tensors.
        let full_ft = model.is_full_ft();
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let mixer = match &layer.mixer {
                Mixer::Linear(gdn) => MergedMixer::Linear(MergedGdn::from_layer(gdn, full_ft)?),
                Mixer::Full(attn) => MergedMixer::Full(MergedAttention::from_layer(attn, full_ft)?),
            };
            layers.push(MergedLayer {
                ln1: if full_ft {
                    layer.ln1.deep_copy()?
                } else {
                    layer.ln1.clone()
                },
                mixer,
                ln2: if full_ft {
                    layer.ln2.deep_copy()?
                } else {
                    layer.ln2.clone()
                },
                mlp: MergedFeedForward::from_layer(&layer.mlp, full_ft)?,
            });
        }
        Ok(Self {
            embed: snap(&model.embed, full_ft)?,
            lm_head: model
                .lm_head
                .as_ref()
                .map(|w| snap(w, full_ft))
                .transpose()?,
            layers,
            norm: if full_ft {
                model.norm.deep_copy()?
            } else {
                model.norm.clone()
            },
            rot: model.rot.clone(),
            hidden: model.hidden,
            max_position: model.max_position,
            device: model.device.clone(),
            consumed: 0,
        })
    }

    /// Logits `[batch, chunk_len, vocab]` for `input_ids`
    /// (`[batch, chunk_len]`, `u32`) at absolute positions
    /// `[offset, offset + chunk_len)`, advancing the decoder state.
    ///
    /// Pass the whole prompt at `offset == 0` to prefill, then decode at the
    /// running offset — one token at a time (the sequential delta-rule step),
    /// or several at once (the chunked continuation, the reference's own
    /// multi-token cached path). `offset` **must** equal the number of tokens
    /// already consumed; a mismatch is rejected **before any state advances**.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `offset != consumed`, if
    /// `offset + chunk_len` exceeds `max_position_embeddings`, if the
    /// internal KV cross-check fails (all three rejected **before** any state
    /// advances), or if a tensor op fails (after which the state is
    /// unspecified — `reset_cache`; see the type docs).
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        if offset != self.consumed {
            bail!(
                "Qwen3_5MergedDecoder::forward: offset {offset} != consumed {} (pass offset == \
                 tokens already consumed; 0 to prefill)",
                self.consumed
            );
        }
        // Position bound, enforced BEFORE any layer mutates: inside the loop
        // the rope-table bound would only fire at the first full-attention
        // layer — after the preceding linear layers had already committed the
        // chunk into their conv/recurrent state, where a contract-correct
        // retry would silently double-consume it. (An all-linear model has no
        // rope table; this is also its only position bound.)
        if offset + l > self.max_position {
            bail!(
                "Qwen3_5MergedDecoder::forward: offset {offset} + chunk_len {l} exceeds \
                 max_position_embeddings {} (state untouched)",
                self.max_position
            );
        }
        // Internal invariant: the first KV layer (when one exists) must agree
        // with the decoder counter — a desync means corrupted state, not a
        // caller error, and decoding through it would be silent corruption.
        let kv_len = self.layers.iter().find_map(|layer| match &layer.mixer {
            MergedMixer::Full(a) => Some(a.cache.current_seq_len()),
            MergedMixer::Linear(_) => None,
        });
        if let Some(kv_len) = kv_len {
            if kv_len != self.consumed {
                bail!(
                    "Qwen3_5MergedDecoder::forward: internal desync — KV cache holds {kv_len} \
                     tokens but the decoder consumed {} (reset_cache to recover)",
                    self.consumed
                );
            }
        }
        let ids = input_ids.flatten_all()?;
        let mut h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?;
        // Activation-dtype mask (see the grad forward); a single new token
        // attends to the whole cache — no mask.
        let mask = if l == 1 {
            None
        } else {
            Some(causal_mask_at(offset, l, h.dtype(), &self.device)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, offset, mask.as_ref(), &self.rot)?;
        }
        let h = self.norm.forward(&h)?;
        self.consumed += l;
        match &self.lm_head {
            Some(w) => frozen_linear(&h, w),
            None => frozen_linear(&h, &self.embed),
        }
    }

    /// Clear every layer's state (KV caches, conv states, delta-rule states)
    /// and the consumed counter, so the decoder can start a fresh sequence
    /// (next [`forward`](Self::forward) must use `offset == 0`).
    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                MergedMixer::Linear(gdn) => gdn.reset(),
                MergedMixer::Full(attn) => attn.cache.reset(),
            }
        }
        self.consumed = 0;
    }
}

/// The [`CachedDecoder`] seam over [`Qwen3_5MergedDecoder`]: pure delegation.
impl CachedDecoder for Qwen3_5MergedDecoder {
    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        Qwen3_5MergedDecoder::forward(self, input_ids, offset)
    }

    fn reset_cache(&mut self) {
        Qwen3_5MergedDecoder::reset_cache(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::grad_coverage;
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::{RngExt, SeedableRng};
    use std::collections::HashMap;

    fn dev() -> Device {
        Device::Cpu
    }

    /// Seed for the deterministic test-weight RNG — seeded for the same reason
    /// as the Llama gates: unseeded weights turn real forward bugs into
    /// intermittent "flake" that gets retried into green.
    const WEIGHT_SEED: u64 = 0x5157_454E_3335; // "QWEN35"

    /// Weight std. At tiny dims, weak weights make attention near-uniform and
    /// the equivalence gates near-vacuous (the M1 lesson); 0.5 keeps the
    /// hybrid recurrence finite while staying decisively non-uniform.
    const WEIGHT_STD: f32 = 0.5;

    /// Deterministic `N(0, std)` tensor (seeded Xoshiro + Box–Muller, the
    /// crate-standard recipe — candle's CPU device rejects `set_seed`).
    fn seeded_randn(
        rng: &mut Xoshiro256PlusPlus,
        std: f32,
        dims: &[usize],
        device: &Device,
    ) -> Tensor {
        let n: usize = dims.iter().product();
        let mut v = Vec::with_capacity(n + 1);
        while v.len() < n {
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

    /// A tiny hybrid config: 4 layers `L,L,L,F`, real GVA (4 value / 2 key
    /// heads), explicit `head_dim` 8 (≠ `hidden/heads` — the Qwen3 pattern),
    /// partial rotary 0.25 → 2 rotated dims. Same arithmetic as the 0.8B at a
    /// runnable scale.
    fn tiny_text_cfg(tie: bool) -> Qwen3_5TextConfig {
        Qwen3_5TextConfig {
            vocab_size: 24,
            hidden_size: 8,
            intermediate_size: Some(16),
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 32,
            tie_word_embeddings: tie,
            attention_bias: false,
            attention_dropout: 0.0,
            layer_types: Some(vec![
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
            ]),
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            rope_parameters: RopeParameters {
                rope_type: "default".to_string(),
                rope_theta: 10_000.0,
                partial_rotary_factor: 0.25,
                mrope_section: None,
                mrope_interleaved: None,
            },
            mlp_only_layers: vec![],
            attn_output_gate: true,
            mamba_ssm_dtype: "float32".to_string(),
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
        }
    }

    /// The sparse twin of [`tiny_text_cfg`]: same mixers/geometry, the
    /// feed-forward slot switched to 4 routed experts top-3 + a shared expert
    /// (widths 6 ≠ 10 — swaps cannot alias). Deliberately top-3 where the
    /// committed fixture is top-2: a `top_k` hardcode or copy-omission on
    /// EITHER decoder side is observationally identical to correct code when
    /// every test geometry shares one k — diversifying k across the two
    /// suites kills that mutant class (the merged decoder is the SAMPLING
    /// path; a silent k drift there would bias every importance ratio).
    fn tiny_moe_text_cfg(tie: bool) -> Qwen3_5TextConfig {
        Qwen3_5TextConfig {
            intermediate_size: None,
            num_experts: Some(4),
            num_experts_per_tok: Some(3),
            moe_intermediate_size: Some(6),
            shared_expert_intermediate_size: Some(10),
            ..tiny_text_cfg(tie)
        }
    }

    fn tiny_cfg() -> Qwen3_5Config {
        Qwen3_5Config {
            model_type: Some("qwen3_5".to_string()),
            tie_word_embeddings: None,
            text_config: tiny_text_cfg(true),
        }
    }

    fn tiny_moe_cfg() -> Qwen3_5Config {
        Qwen3_5Config {
            model_type: Some("qwen3_5_moe".to_string()),
            tie_word_embeddings: None,
            text_config: tiny_moe_text_cfg(true),
        }
    }

    /// Deterministic seeded weights for `cfg`, under the real checkpoint
    /// prefix (`model.language_model.*`; `lm_head.weight` only when untied).
    /// Insertion order is fixed → bit-identical tensors on every call.
    fn weight_map(cfg: &Qwen3_5TextConfig) -> HashMap<String, Tensor> {
        let d = dev();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(WEIGHT_SEED);
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, dims: &[usize]| {
            t.insert(
                format!("model.language_model.{name}"),
                seeded_randn(&mut rng, WEIGHT_STD, dims, &d),
            );
        };
        let h = cfg.hidden_size;
        let gd = GdnDims::from_config(cfg);
        let ad = AttnDims::from_config(cfg);
        put("embed_tokens.weight", &[cfg.vocab_size, h]);
        put("norm.weight", &[h]);
        for (i, kind) in cfg.resolved_layer_types().iter().enumerate() {
            let p = format!("layers.{i}");
            put(&format!("{p}.input_layernorm.weight"), &[h]);
            put(&format!("{p}.post_attention_layernorm.weight"), &[h]);
            match kind {
                LayerType::LinearAttention => {
                    let q = format!("{p}.linear_attn");
                    put(&format!("{q}.in_proj_qkv.weight"), &[gd.conv_dim, h]);
                    put(&format!("{q}.in_proj_z.weight"), &[gd.value_dim, h]);
                    put(&format!("{q}.in_proj_b.weight"), &[gd.num_v_heads, h]);
                    put(&format!("{q}.in_proj_a.weight"), &[gd.num_v_heads, h]);
                    put(&format!("{q}.conv1d.weight"), &[gd.conv_dim, 1, gd.kernel]);
                    put(&format!("{q}.A_log"), &[gd.num_v_heads]);
                    put(&format!("{q}.dt_bias"), &[gd.num_v_heads]);
                    put(&format!("{q}.norm.weight"), &[gd.head_v]);
                    put(&format!("{q}.out_proj.weight"), &[h, gd.value_dim]);
                }
                LayerType::FullAttention => {
                    let q = format!("{p}.self_attn");
                    let attn_out = ad.num_heads * ad.head_dim;
                    let kv_out = ad.num_kv_heads * ad.head_dim;
                    put(&format!("{q}.q_proj.weight"), &[attn_out * 2, h]);
                    put(&format!("{q}.k_proj.weight"), &[kv_out, h]);
                    put(&format!("{q}.v_proj.weight"), &[kv_out, h]);
                    put(&format!("{q}.o_proj.weight"), &[h, attn_out]);
                    put(&format!("{q}.q_norm.weight"), &[ad.head_dim]);
                    put(&format!("{q}.k_norm.weight"), &[ad.head_dim]);
                }
            }
            match cfg.moe() {
                None => {
                    let i = cfg.intermediate_size.unwrap();
                    put(&format!("{p}.mlp.gate_proj.weight"), &[i, h]);
                    put(&format!("{p}.mlp.up_proj.weight"), &[i, h]);
                    put(&format!("{p}.mlp.down_proj.weight"), &[h, i]);
                }
                Some(moe) => {
                    let (e, m, s) = (
                        moe.num_experts,
                        moe.moe_intermediate_size,
                        moe.shared_expert_intermediate_size,
                    );
                    put(&format!("{p}.mlp.gate.weight"), &[e, h]);
                    // The checkpoint layout: per-expert split linears (the
                    // loader packs them).
                    for x in 0..e {
                        let q = format!("{p}.mlp.experts.{x}");
                        put(&format!("{q}.gate_proj.weight"), &[m, h]);
                        put(&format!("{q}.up_proj.weight"), &[m, h]);
                        put(&format!("{q}.down_proj.weight"), &[h, m]);
                    }
                    put(&format!("{p}.mlp.shared_expert.gate_proj.weight"), &[s, h]);
                    put(&format!("{p}.mlp.shared_expert.up_proj.weight"), &[s, h]);
                    put(&format!("{p}.mlp.shared_expert.down_proj.weight"), &[h, s]);
                    put(&format!("{p}.mlp.shared_expert_gate.weight"), &[1, h]);
                }
            }
        }
        if !cfg.tie_word_embeddings {
            let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(WEIGHT_SEED ^ 0x77);
            t.insert(
                "lm_head.weight".to_string(),
                seeded_randn(&mut rng2, WEIGHT_STD, &[cfg.vocab_size, h], &d),
            );
        }
        t
    }

    fn tiny_vb(cfg: &Qwen3_5TextConfig) -> VarBuilder<'static> {
        VarBuilder::from_tensors(weight_map(cfg), DType::F32, &dev())
    }

    fn tiny_model() -> Qwen3_5GradModel {
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg.text_config);
        Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap()
    }

    fn ids(seq: usize) -> Tensor {
        let v: Vec<u32> = (0..seq as u32).map(|i| (i * 7 + 3) % 24).collect();
        Tensor::from_vec(v, (1, seq), &dev()).unwrap()
    }

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

    // ---- gate envelopes ----------------------------------------------------
    //
    // Measured under the seeded WEIGHT_SEED/WEIGHT_STD weights (worst measured
    // value recorded next to each constant), then set with wide headroom for
    // cross-host float reassociation (the P2 platform lesson).

    /// Cached (merged-decoder) vs uncached forward, adapter armed. The decode
    /// path crosses the chunked/recurrent kernel boundary (independent ports,
    /// PR-1 cross-checked at 5e-5 kernel-level), so its floor is the highest.
    /// Measured worst under the seeded weights (2026-06-11): prefill 3.8e-5,
    /// chunked continuation 7.0e-5, token-by-token decode 1.2e-4 (the
    /// std-0.5 tiny weights run hotter than the real 0.8B, whose same trio
    /// measures 3.3e-5; planted bugs land at 4.9e-3–12.5e-3). GitHub's
    /// runner pool (2026-06-12) measured honest cross-host excursions of
    /// 1.14e-3 (continuation) and 1.04e-3 (decode) — different runs, same
    /// binary content, different failures: the pool mixes CPU generations
    /// whose gemm reassociation differs ~10x from the dev host on this
    /// kernel boundary. Envelope sits at the geometric midpoint between the
    /// worst honest excursion (2.2x below) and the nearest planted bug
    /// (~2x above).
    const MERGED_TOL: f32 = 2.5e-3;

    // ---- config validation -------------------------------------------------

    #[test]
    fn config_rejects_unsupported_knobs() {
        let assert_rejects = |mutate: &dyn Fn(&mut Qwen3_5TextConfig), needle: &str| {
            let mut t = tiny_text_cfg(true);
            mutate(&mut t);
            let err = t.validate().expect_err(needle).to_string();
            assert!(
                err.contains(needle),
                "error {err:?} should mention {needle:?}"
            );
        };
        assert_rejects(&|t| t.hidden_act = "gelu".into(), "hidden_act");
        assert_rejects(&|t| t.attention_bias = true, "attention_bias");
        assert_rejects(&|t| t.attention_dropout = 0.1, "attention_dropout");
        assert_rejects(&|t| t.mlp_only_layers = vec![1], "mlp_only_layers");
        assert_rejects(&|t| t.attn_output_gate = false, "attn_output_gate");
        assert_rejects(
            &|t| t.mamba_ssm_dtype = "bfloat16".into(),
            "mamba_ssm_dtype",
        );
        assert_rejects(
            &|t| t.rope_parameters.rope_type = "yarn".into(),
            "rope_type",
        );
        assert_rejects(
            &|t| t.rope_parameters.partial_rotary_factor = 0.0,
            "partial_rotary_factor",
        );
        // head_dim 8 * 0.3 = 2.4 — not an integer dim count.
        assert_rejects(
            &|t| t.rope_parameters.partial_rotary_factor = 0.3,
            "partial_rotary_factor",
        );
        assert_rejects(
            &|t| t.rope_parameters.mrope_interleaved = Some(false),
            "mrope_interleaved",
        );
        assert_rejects(
            &|t| t.rope_parameters.mrope_section = Some(vec![3, 1]),
            "mrope_section",
        );
        assert_rejects(&|t| t.num_key_value_heads = 3, "num_attention_heads");
        assert_rejects(&|t| t.linear_num_value_heads = 3, "linear_num_value_heads");
        assert_rejects(
            &|t| t.layer_types = Some(vec![LayerType::FullAttention]),
            "layer_types",
        );
        assert_rejects(&|t| t.vocab_size = 0, "vocab_size");

        // The composite config also pins the model type.
        let bad = Qwen3_5Config {
            model_type: Some("qwen3".to_string()),
            tie_word_embeddings: None,
            text_config: tiny_text_cfg(true),
        };
        assert!(bad
            .validate()
            .unwrap_err()
            .to_string()
            .contains("model_type"));

        // And the unmutated tiny config is valid.
        tiny_cfg().validate().unwrap();
    }

    #[test]
    fn layer_type_deserialization_is_fail_loud() {
        let ok: LayerType = serde_json::from_str("\"linear_attention\"").unwrap();
        assert_eq!(ok, LayerType::LinearAttention);
        let err = serde_json::from_str::<LayerType>("\"sliding_attention\"");
        assert!(err.is_err(), "unknown layer type must not deserialize");
    }

    #[test]
    fn resolved_layer_types_interval_pattern() {
        let mut t = tiny_text_cfg(true);
        t.layer_types = None;
        t.num_hidden_layers = 8;
        let kinds = t.resolved_layer_types();
        assert_eq!(kinds.len(), 8);
        for (i, k) in kinds.iter().enumerate() {
            let want = if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(*k, want, "layer {i}");
        }
    }

    /// The shipped 0.8B-Base config.json head (trimmed to the keys we consume
    /// plus the riders we must tolerate). Pins the serde mapping against the
    /// real file's shape.
    const REAL_0_8B_CONFIG: &str = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "dtype": "bfloat16",
                "eos_token_id": 248044,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 3584,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "mlp_only_layers": [],
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": true,
                "use_cache": true,
                "vocab_size": 248320,
                "mamba_ssm_dtype": "float32",
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }
            },
            "tie_word_embeddings": true,
            "vision_config": {"depth": 12, "hidden_size": 768}
        }"#;

    #[test]
    fn real_0_8b_config_shape_parses() {
        let cfg = Qwen3_5Config::from_json_str(REAL_0_8B_CONFIG).unwrap();
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 24);
        assert_eq!(t.rotary_dim(), 64);
        assert!(t.tie_word_embeddings);
    }

    #[test]
    fn real_0_8b_layer_pattern_resolves() {
        let cfg = Qwen3_5Config::from_json_str(REAL_0_8B_CONFIG).unwrap();
        let kinds = cfg.text_config.resolved_layer_types();
        assert_eq!(kinds.len(), 24);
        assert_eq!(kinds[0], LayerType::LinearAttention);
        assert_eq!(kinds[3], LayerType::FullAttention);
        let full = kinds
            .iter()
            .filter(|k| **k == LayerType::FullAttention)
            .count();
        assert_eq!(full, 6);
    }

    // ---- LoraTargets --------------------------------------------------------

    #[test]
    fn lora_targets_canonical_and_any() {
        assert_eq!(
            LoraTargets::default().canonical(),
            "attn:qkvo|mlp:gud|gdn:-"
        );
        assert_eq!(
            LoraTargets::all_linear().canonical(),
            "attn:qkvo|mlp:gud|gdn:qzbao"
        );
        let none = LoraTargets {
            attn_q: false,
            attn_k: false,
            attn_v: false,
            attn_o: false,
            mlp_gate: false,
            mlp_up: false,
            mlp_down: false,
            ..LoraTargets::industrial()
        };
        assert!(!none.any());
        assert_eq!(none.canonical(), "attn:-|mlp:-|gdn:-");

        // A no-target load is rejected (the model would train nothing).
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg.text_config);
        let err = Qwen3_5GradModel::load_with_targets(&cfg, &vb, 2, 4.0, DType::F32, none);
        assert!(err.is_err());
    }

    #[test]
    fn trainable_var_count_and_determinism() {
        // Default recipe on L,L,L,F: 3 linear layers contribute MLP only
        // (3 projs), the full layer q,k,v,o + MLP (7 projs); 2 vars per proj.
        let model = tiny_model();
        let vars = model.trainable_vars();
        assert_eq!(vars.len(), (3 * 3 + 7) * 2);
        // Deterministic across identically configured loads: same count,
        // same shapes, position by position.
        let model2 = tiny_model();
        let vars2 = model2.trainable_vars();
        assert_eq!(vars.len(), vars2.len());
        for (a, b) in vars.iter().zip(vars2.iter()) {
            assert_eq!(a.dims(), b.dims());
            assert_eq!(a.dtype(), b.dtype());
        }
    }

    #[test]
    fn trainable_var_count_all_linear() {
        // all_linear adds the 5 GDN projections on each linear layer.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg.text_config);
        let all = Qwen3_5GradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            LoraTargets::all_linear(),
        )
        .unwrap();
        assert_eq!(all.trainable_vars().len(), (3 * (5 + 3) + 7) * 2);
        assert_eq!(
            all.lora_targets().canonical(),
            "attn:qkvo|mlp:gud|gdn:qzbao"
        );
    }

    // ---- forward shape + autograd -------------------------------------------

    #[test]
    fn forward_produces_full_seq_logits() {
        let model = tiny_model();
        let logits = model.forward(&ids(9)).unwrap();
        assert_eq!(logits.dims(), &[1, 9, 24]);
        let s: f32 = logits
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(s.is_finite() && s > 0.0);
    }

    #[test]
    fn single_token_uncached_forward_works() {
        // l == 1 takes the no-mask branch in the uncached forward too.
        let model = tiny_model();
        let logits = model.forward(&ids(1)).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 24]);
    }

    #[test]
    fn untied_lm_head_branch_loads_and_runs() {
        let cfg = Qwen3_5Config {
            model_type: Some("qwen3_5".to_string()),
            tie_word_embeddings: None,
            text_config: tiny_text_cfg(false),
        };
        let vb = tiny_vb(&cfg.text_config);
        let model = Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        let tied = tiny_model();
        let logits = model.forward(&ids(5)).unwrap();
        let logits_tied = tied.forward(&ids(5)).unwrap();
        // Different heads ⇒ different logits (the untied weight is actually
        // used, not silently swapped for the embedding).
        assert!(max_abs_diff(&logits, &logits_tied) > 1e-3);
    }

    /// One `forward -> sqr -> sum -> backward`, returning the grad store.
    fn backward_grads(model: &Qwen3_5GradModel) -> candle_core::backprop::GradStore {
        model
            .forward(&ids(7))
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
    }

    /// Set every `B` factor (odd index in each `[A, B]` pair) to small seeded
    /// noise so `dL/dA` is no longer structurally zero.
    fn force_b_nonzero(vars: &[Var]) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(WEIGHT_SEED ^ 0xB);
        for (i, v) in vars.iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                let noise = seeded_randn(&mut rng, 0.05, &dims, &dev())
                    .to_dtype(v.dtype())
                    .unwrap();
                v.set(&noise).unwrap();
            }
        }
    }

    /// Split the default-recipe var list into (attn, mlp) branches. Layer
    /// order is L,L,L,F; linear layers contribute 6 MLP vars each, the full
    /// layer 8 attention vars then 6 MLP vars.
    fn branch_split_default(vars: &[Var]) -> (Vec<Var>, Vec<Var>) {
        assert_eq!(vars.len(), 32);
        let attn = vars[18..26].to_vec();
        let mlp = [&vars[..18], &vars[26..]].concat();
        (attn, mlp)
    }

    #[test]
    fn lora_grads_flow_two_phase_per_branch() {
        let mut model = tiny_model();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        let (attn_vars, mlp_vars) = branch_split_default(&vars);

        // Phase 1 — zero-B init: every var present + each branch live + finite.
        let g1 = backward_grads(&model);
        assert!(
            grad_coverage(&attn_vars, &g1).unwrap().is_ok(),
            "attention branch unhealthy at zero-B init (grad-safe twin cut?)"
        );
        assert!(
            grad_coverage(&mlp_vars, &g1).unwrap().is_ok(),
            "mlp branch unhealthy at zero-B init"
        );

        // Phase 2 — nonzero B: EVERY A and B must carry a nonzero finite grad.
        force_b_nonzero(&vars);
        let g2 = backward_grads(&model);
        let ac = grad_coverage(&attn_vars, &g2).unwrap();
        let mc = grad_coverage(&mlp_vars, &g2).unwrap();
        assert!(
            ac.nonzero == ac.total && ac.nonfinite == 0,
            "attention branch: not every LoRA var live after nonzero-B: {ac:?}"
        );
        assert!(
            mc.nonzero == mc.total && mc.nonfinite == 0,
            "mlp branch: not every LoRA var live after nonzero-B: {mc:?}"
        );
    }

    #[test]
    fn gdn_optin_lora_grads_flow() {
        // The opt-in GDN projections must be just as grad-live when enabled —
        // their forward path crosses the conv composite, the chunked delta
        // rule, and the gated norm.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg.text_config);
        let mut model = Qwen3_5GradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            LoraTargets::all_linear(),
        )
        .unwrap();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        assert_eq!(vars.len(), 62);
        // GDN vars: the first 10 of each linear layer's 16. The slice is
        // keyed to the documented mixer-first push order — pin it
        // STRUCTURALLY by shape, so a silent push_vars reorder turns this
        // test red instead of quietly rotating GDN vars out of the checked
        // set (the one place a refactor could make this gate vacuous).
        let gd = GdnDims::from_config(&tiny_text_cfg(true));
        let gdn_vars: Vec<Var> = (0..3)
            .flat_map(|l| vars[l * 16..l * 16 + 10].to_vec())
            .collect();
        for l in 0..3 {
            assert_eq!(
                vars[l * 16].dims(),
                &[2, 8],
                "layer {l}: first var must be in_proj_qkv A [rank, hidden] — push order drifted"
            );
            assert_eq!(
                vars[l * 16 + 1].dims(),
                &[gd.conv_dim, 2],
                "layer {l}: second var must be in_proj_qkv B [conv_dim, rank] — push order drifted"
            );
        }
        let g1 = backward_grads(&model);
        assert!(
            grad_coverage(&gdn_vars, &g1).unwrap().is_ok(),
            "GDN branch unhealthy at zero-B init"
        );
        force_b_nonzero(&vars);
        let g2 = backward_grads(&model);
        let gc = grad_coverage(&gdn_vars, &g2).unwrap();
        assert!(
            gc.nonzero == gc.total && gc.nonfinite == 0,
            "GDN branch: not every LoRA var live after nonzero-B: {gc:?}"
        );
    }

    #[test]
    fn dtype_split_forward_and_grad() {
        // F32 base / F64 adapter (the CPU surrogate for bf16/F32): forward in
        // the base dtype, every grad landing in the MASTER dtype.
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg.text_config);
        let mut model =
            Qwen3_5GradModel::load_with_adapter_dtype(&cfg, &vb, 2, 4.0, DType::F64).unwrap();
        model.set_adapter_enabled(true);
        let vars = model.trainable_vars();
        let logits = model.forward(&ids(5)).unwrap();
        assert_eq!(logits.dtype(), DType::F32);
        force_b_nonzero(&vars);
        let grads = backward_grads(&model);
        for v in &vars {
            let g = grads
                .get(v.as_tensor())
                .expect("adapter var missing from grad store");
            assert_eq!(g.dtype(), DType::F64, "grad must land in the master dtype");
        }
    }

    #[test]
    fn adapter_toggle_is_noop_at_zero_b_and_bites_after() {
        let mut model = tiny_model();
        let input = ids(6);
        model.set_adapter_enabled(true);
        let on = model.forward(&input).unwrap();
        model.set_adapter_enabled(false);
        let off = model.forward(&input).unwrap();
        assert!(
            max_abs_diff(&on, &off) == 0.0,
            "zero-B adapter must be an exact no-op"
        );
        // Arm the adapter: now the toggle must matter.
        force_b_nonzero(&model.trainable_vars());
        model.set_adapter_enabled(true);
        let on2 = model.forward(&input).unwrap();
        assert!(
            max_abs_diff(&on2, &off) > 1e-4,
            "armed adapter must change the logits"
        );
    }

    // ---- GVA repeat ----------------------------------------------------------

    #[test]
    fn repeat_gva_heads_matches_interleave() {
        // [1, 1, 2, 2] with distinguishable heads -> each head doubled
        // CONSECUTIVELY (torch.repeat_interleave semantics, NOT tiling).
        let x = Tensor::from_vec(vec![1f32, 2., 10., 20.], (1, 1, 2, 2), &dev()).unwrap();
        let y = repeat_gva_heads(&x, 2).unwrap();
        assert_eq!(y.dims(), &[1, 1, 4, 2]);
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![1., 2., 1., 2., 10., 20., 10., 20.]);
    }

    // ---- causality ------------------------------------------------------------

    #[test]
    fn future_token_does_not_change_past_logits() {
        let model = tiny_model();
        let a = ids(8);
        let mut v: Vec<u32> = (0..8u32).map(|i| (i * 7 + 3) % 24).collect();
        v[7] = (v[7] + 11) % 24;
        let b = Tensor::from_vec(v, (1, 8), &dev()).unwrap();
        let la = model.forward(&a).unwrap();
        let lb = model.forward(&b).unwrap();
        let past_a = la.narrow(1, 0, 7).unwrap();
        let past_b = lb.narrow(1, 0, 7).unwrap();
        assert!(
            max_abs_diff(&past_a, &past_b) == 0.0,
            "perturbing token 7 must not move logits at positions 0..7 (masked future \
             contributions are exact zeros, so causality is bit-exact, not approximate)"
        );
        let last_a = la.narrow(1, 7, 1).unwrap();
        let last_b = lb.narrow(1, 7, 1).unwrap();
        assert!(
            max_abs_diff(&last_a, &last_b) > 1e-3,
            "the perturbed position itself must move (else this gate is vacuous)"
        );
    }

    // ---- merged decoder ---------------------------------------------------------

    /// An armed (nonzero-B, adapter-on) model and its merged decoder — the
    /// nontrivial case where the merge must actually fold the adapter.
    fn armed_model() -> Qwen3_5GradModel {
        let mut model = tiny_model();
        force_b_nonzero(&model.trainable_vars());
        model.set_adapter_enabled(true);
        model
    }

    #[test]
    fn merged_prefill_matches_uncached() {
        let model = armed_model();
        let input = ids(9);
        let uncached = model.forward(&input).unwrap();
        let mut dec = model.merged_decoder().unwrap();
        let cached = dec.forward(&input, 0).unwrap();
        let d = max_abs_diff(&uncached, &cached);
        assert!(d <= MERGED_TOL, "prefill vs uncached diff {d}");
    }

    #[test]
    fn decode_equals_prefill_token_by_token() {
        let model = armed_model();
        let full = ids(9);
        let uncached = model.forward(&full).unwrap();
        let mut dec = model.merged_decoder().unwrap();
        // Prefill 5, then decode tokens 5..9 one at a time (the sequential
        // delta-rule step + per-step conv update on the linear layers).
        let prefix = full.narrow(1, 0, 5).unwrap();
        let mut got = vec![dec.forward(&prefix, 0).unwrap()];
        for t in 5..9 {
            let tok = full.narrow(1, t, 1).unwrap();
            got.push(dec.forward(&tok, t).unwrap());
        }
        let cached = Tensor::cat(&got, 1).unwrap();
        let d = max_abs_diff(&uncached, &cached);
        assert!(d <= MERGED_TOL, "decode vs uncached diff {d}");
    }

    #[test]
    fn chunked_continuation_at_offset_equals_uncached() {
        // Multi-token cached continuation — the reference path fixed in
        // transformers v5.7.0, our highest-risk decoder path: prefill p
        // tokens, then forward a MULTI-token chunk at offset p. Covers
        // prefills shorter than the conv kernel (left-zero-pad), longer, and
        // a single-token prefill.
        let model = armed_model();
        let full = ids(9);
        let uncached = model.forward(&full).unwrap();
        for p in [1usize, 2, 5] {
            let mut dec = model.merged_decoder().unwrap();
            let prefix = full.narrow(1, 0, p).unwrap();
            let rest = full.narrow(1, p, 9 - p).unwrap();
            let first = dec.forward(&prefix, 0).unwrap();
            let second = dec.forward(&rest, p).unwrap();
            let cached = Tensor::cat(&[first, second], 1).unwrap();
            let d = max_abs_diff(&uncached, &cached);
            assert!(
                d <= MERGED_TOL,
                "split at {p}: chunked continuation diff {d}"
            );
        }
    }

    #[test]
    fn offset_mismatch_is_rejected_and_state_untouched() {
        let model = armed_model();
        let mut dec = model.merged_decoder().unwrap();
        let prompt = ids(4);
        dec.forward(&prompt, 0).unwrap();
        // Wrong offsets in both directions — and offset 0 on a non-fresh
        // decoder (the llama-3 guard semantics).
        for wrong in [0usize, 3, 5] {
            let err = dec.forward(&ids(1), wrong);
            assert!(err.is_err(), "offset {wrong} must be rejected (consumed 4)");
        }
        // The failed calls must not have advanced anything: the correct
        // offset still works and matches a fresh decoder's value.
        let tok = ids(5).narrow(1, 4, 1).unwrap();
        let good = dec.forward(&tok, 4).unwrap();
        let mut fresh = model.merged_decoder().unwrap();
        fresh.forward(&prompt, 0).unwrap();
        let want = fresh.forward(&tok, 4).unwrap();
        assert!(max_abs_diff(&good, &want) == 0.0);
    }

    #[test]
    fn max_position_overflow_is_rejected_before_state_advances() {
        // The sweep-found hybrid hazard: without the decoder-level precheck,
        // the rope-table bound fires at the FIRST FULL-ATTENTION layer — after
        // the preceding linear layers already committed the failed chunk into
        // their conv/recurrent state — so a contract-correct retry would
        // silently double-consume it there. The precheck must reject the
        // overflow with the state untouched: the in-bound retry must be
        // bit-identical to a fresh decoder's. (tiny cfg max_position = 32)
        let model = armed_model();
        let mut dec = model.merged_decoder().unwrap();
        let prefix = Tensor::from_vec(
            (0..30u32).map(|i| i % 24).collect::<Vec<_>>(),
            (1, 30),
            &dev(),
        )
        .unwrap();
        dec.forward(&prefix, 0).unwrap();
        let over = Tensor::from_vec(vec![1u32; 5], (1, 5), &dev()).unwrap();
        assert!(
            dec.forward(&over, 30).is_err(),
            "30 + 5 > 32 must be rejected"
        );
        let fit = Tensor::from_vec(vec![1u32, 2], (1, 2), &dev()).unwrap();
        let good = dec.forward(&fit, 30).unwrap();
        let mut fresh = model.merged_decoder().unwrap();
        fresh.forward(&prefix, 0).unwrap();
        let want = fresh.forward(&fit, 30).unwrap();
        assert!(
            max_abs_diff(&good, &want) == 0.0,
            "the rejected overflow chunk must not have advanced any layer state"
        );
        // And the now-full decoder rejects even a single further token.
        let one = Tensor::from_vec(vec![3u32], (1, 1), &dev()).unwrap();
        assert!(
            dec.forward(&one, 32).is_err(),
            "32 + 1 > 32 must be rejected"
        );
    }

    #[test]
    fn reset_replay_is_exact() {
        let model = armed_model();
        let mut dec = model.merged_decoder().unwrap();
        let prompt = ids(6);
        let a1 = dec.forward(&prompt, 0).unwrap();
        let t = ids(7).narrow(1, 6, 1).unwrap();
        let a2 = dec.forward(&t, 6).unwrap();
        dec.reset_cache();
        let b1 = dec.forward(&prompt, 0).unwrap();
        let b2 = dec.forward(&t, 6).unwrap();
        // Same ops, same state lifecycle ⇒ bit-identical replay (any residue
        // in the conv/recurrent/KV state would show here).
        assert!(
            max_abs_diff(&a1, &b1) == 0.0,
            "prefill replay must be exact"
        );
        assert!(max_abs_diff(&a2, &b2) == 0.0, "decode replay must be exact");
    }

    #[test]
    fn merged_decoder_is_a_value_snapshot() {
        // After an "optimizer step" (mutating B), an EXISTING decoder keeps
        // sampling from the old weights; a rebuilt one sees the new ones.
        let mut model = tiny_model();
        model.set_adapter_enabled(true);
        let input = ids(5);
        let mut dec_old = model.merged_decoder().unwrap();
        let old = dec_old.forward(&input, 0).unwrap();
        force_b_nonzero(&model.trainable_vars());
        let mut dec_old_replay = dec_old;
        dec_old_replay.reset_cache();
        let old_replay = dec_old_replay.forward(&input, 0).unwrap();
        assert!(
            max_abs_diff(&old, &old_replay) == 0.0,
            "an existing snapshot must not see the weight change"
        );
        let mut dec_new = model.merged_decoder().unwrap();
        let new = dec_new.forward(&input, 0).unwrap();
        assert!(
            max_abs_diff(&old, &new) > 1e-4,
            "a rebuilt snapshot must see the weight change"
        );
    }

    // ---- loader ---------------------------------------------------------------

    #[test]
    fn varbuilder_from_pretrained_single_and_sharded() {
        let map = weight_map(&tiny_text_cfg(true));
        let base = std::env::temp_dir().join(format!("ferrl-qwen35-loader-{}", std::process::id()));

        // Single-file layout.
        let single = base.join("single");
        std::fs::create_dir_all(&single).unwrap();
        candle_core::safetensors::save(&map, single.join("model.safetensors")).unwrap();

        // Sharded layout: split the map in two + an index.json, plus a decoy
        // visual tensor that must ride along unrequested.
        let sharded = base.join("sharded");
        std::fs::create_dir_all(&sharded).unwrap();
        let mut names: Vec<String> = map.keys().cloned().collect();
        names.sort();
        let (a_names, b_names) = names.split_at(names.len() / 2);
        let mut shard_a: HashMap<String, Tensor> = a_names
            .iter()
            .map(|n| (n.clone(), map[n].clone()))
            .collect();
        shard_a.insert(
            "model.visual.patch_embed.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &dev()).unwrap(),
        );
        let shard_b: HashMap<String, Tensor> = b_names
            .iter()
            .map(|n| (n.clone(), map[n].clone()))
            .collect();
        candle_core::safetensors::save(&shard_a, sharded.join("model-00001-of-00002.safetensors"))
            .unwrap();
        candle_core::safetensors::save(&shard_b, sharded.join("model-00002-of-00002.safetensors"))
            .unwrap();
        let weight_map_json: HashMap<&String, &str> = a_names
            .iter()
            .map(|n| (n, "model-00001-of-00002.safetensors"))
            .chain(
                b_names
                    .iter()
                    .map(|n| (n, "model-00002-of-00002.safetensors")),
            )
            .collect();
        let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map_json });
        std::fs::write(
            sharded.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let cfg = tiny_cfg();
        let input = ids(5);
        let want = tiny_model().forward(&input).unwrap();
        for dir in [&single, &sharded] {
            let vb = varbuilder_from_pretrained(dir, DType::F32, &dev()).unwrap();
            let model = Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap();
            let got = model.forward(&input).unwrap();
            assert!(
                max_abs_diff(&got, &want) == 0.0,
                "{} must load the identical model",
                dir.display()
            );
        }

        // A directory with neither layout fails loud.
        let empty = base.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(varbuilder_from_pretrained(&empty, DType::F32, &dev()).is_err());

        std::fs::remove_dir_all(&base).ok();
    }

    // ---- activation checkpointing (P7) --------------------------------------

    /// A fixed non-uniform probe loss over the logits — no gradient cancels
    /// by symmetry.
    fn probe_loss(logits: &Tensor) -> Tensor {
        let n = logits.elem_count();
        let w: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.21 - 0.6).collect();
        let w = Tensor::from_vec(w, logits.dims().to_vec(), logits.device())
            .unwrap()
            .to_dtype(logits.dtype())
            .unwrap();
        logits.mul(&w).unwrap().sum_all().unwrap()
    }

    /// The hybrid stack under checkpointing: the segment closure must rebuild
    /// BOTH mixer kinds (`GatedDeltaNet` and gated GQA) — the stitched
    /// gradients must match the uncut backward on every adapter var, and a
    /// raw `loss.backward()` after a checkpointed forward must reach none
    /// (the tape really is cut).
    #[test]
    fn checkpointed_gradients_match_the_uncut_backward_across_both_mixers() {
        let mut model = armed_model();
        let input = ids(9);
        let vars = model.trainable_vars();

        let plain = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();
        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();

        let mut worst = 0f32;
        for v in &vars {
            let a = plain.get(v).expect("var missing from the uncut store");
            let b = stitched
                .get(v)
                .expect("var missing from the stitched store");
            worst = worst.max(max_abs_diff(a, b));
        }
        assert!(
            worst <= 1e-5,
            "stitched grads diverged from the uncut backward: {worst}"
        );

        // The cut: bypassing the stitching reaches no layer var.
        let raw = probe_loss(&model.forward(&input).unwrap())
            .backward()
            .unwrap();
        assert!(
            vars.iter().all(|v| raw.get(v).is_none()),
            "a layer var is on the loss tape — the boundary cut is not happening"
        );
    }

    /// Under checkpointing a value scoring must capture NO tape (it would
    /// clobber the tape the next update backward consumes).
    #[test]
    fn forward_detached_captures_no_tape_under_checkpointing() {
        let mut model = armed_model();
        model.set_activation_checkpointing(true);
        let _ = model.forward_detached(&ids(5)).unwrap();
        let scalar = Tensor::zeros((), DType::F32, &dev()).unwrap();
        let err = model.backward(&scalar).unwrap_err();
        assert!(err.to_string().contains("no checkpointed forward"));
    }

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

    /// The narrowed scoring forward on the hybrid stack — the window spans
    /// positions fed by BOTH mixer kinds (the 9-token tiny stack interleaves
    /// `GatedDeltaNet` and gated GQA layers): values and adapter gradients
    /// exactly match `forward` + narrow, plain and checkpointed, and the
    /// narrowed detached walk captures no tape.
    #[test]
    fn narrowed_forward_matches_the_full_walk_across_both_mixers() {
        let mut model = armed_model();
        let input = ids(9);
        let (start, len) = (3, 4);
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

        // Checkpointed: the narrow rides the loss tape; the stitch matches.
        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(
                &model.forward_narrowed(&input, start, len).unwrap(),
            ))
            .unwrap();
        assert_grads_match(&g_narrow, &stitched, &vars, 1e-5);

        // The narrowed detached walk captures no tape.
        let _ = model.forward_detached_narrowed(&input, start, len).unwrap();
        let scalar = Tensor::zeros((), DType::F32, &dev()).unwrap();
        let err = model.backward(&scalar).unwrap_err();
        assert!(err.to_string().contains("no checkpointed forward"));
    }

    // ---- the sparse feed-forward menu (M3' PR-2) -----------------------------

    fn tiny_moe_model() -> Qwen3_5GradModel {
        let cfg = tiny_moe_cfg();
        let vb = tiny_vb(&cfg.text_config);
        Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap()
    }

    fn armed_moe_model() -> Qwen3_5GradModel {
        let mut model = tiny_moe_model();
        force_b_nonzero(&model.trainable_vars());
        model.set_adapter_enabled(true);
        model
    }

    #[test]
    fn moe_config_menu_validation_is_fail_loud() {
        let assert_rejects = |mutate: &dyn Fn(&mut Qwen3_5Config), needle: &str| {
            let mut c = tiny_moe_cfg();
            mutate(&mut c);
            let err = c.validate().expect_err(needle).to_string();
            assert!(
                err.contains(needle),
                "error {err:?} should mention {needle:?}"
            );
        };
        // Both menus at once / neither.
        assert_rejects(
            &|c| c.text_config.intermediate_size = Some(16),
            "both intermediate_size and num_experts",
        );
        assert_rejects(
            &|c| {
                c.model_type = Some("qwen3_5".to_string());
                c.text_config.num_experts = None;
                c.text_config.num_experts_per_tok = None;
                c.text_config.moe_intermediate_size = None;
                c.text_config.shared_expert_intermediate_size = None;
            },
            "neither intermediate_size",
        );
        // An incomplete quartet — each optional field's absence individually.
        assert_rejects(&|c| c.text_config.num_experts_per_tok = None, "incomplete");
        assert_rejects(
            &|c| c.text_config.moe_intermediate_size = None,
            "incomplete",
        );
        assert_rejects(
            &|c| c.text_config.shared_expert_intermediate_size = None,
            "incomplete",
        );
        // Top-k out of range, both ends.
        assert_rejects(
            &|c| c.text_config.num_experts_per_tok = Some(5),
            "num_experts_per_tok",
        );
        assert_rejects(
            &|c| c.text_config.num_experts_per_tok = Some(0),
            "num_experts_per_tok",
        );
        // The aux-loss switch this forward does not implement.
        assert_rejects(
            &|c| c.text_config.output_router_logits = true,
            "output_router_logits",
        );
        // A dense member carrying a stray MoE field — each field individually.
        type Mutation = fn(&mut Qwen3_5Config);
        let stray: [(&str, Mutation); 3] = [
            ("num_experts_per_tok", |c| {
                c.text_config.num_experts_per_tok = Some(2);
            }),
            ("moe_intermediate_size", |c| {
                c.text_config.moe_intermediate_size = Some(6);
            }),
            ("shared_expert_intermediate_size", |c| {
                c.text_config.shared_expert_intermediate_size = Some(10);
            }),
        ];
        for (needle, mutate) in stray {
            let mut c = tiny_cfg();
            mutate(&mut c);
            let err = c.validate().expect_err(needle).to_string();
            assert!(
                err.contains(needle),
                "error {err:?} should mention {needle:?}"
            );
        }
        // The family tag and the menu must agree, both directions.
        assert_rejects(&|c| c.model_type = Some("qwen3_5".to_string()), "disagrees");
        {
            let mut c = tiny_cfg();
            c.model_type = Some("qwen3_5_moe".to_string());
            let err = c.validate().expect_err("tag/menu disagreement").to_string();
            assert!(err.contains("disagrees"), "{err:?}");
        }
        // The valid sparse config passes, and the dims resolve — all four
        // values distinct, so any cross-field mapping in `moe()` fails here.
        let c = tiny_moe_cfg();
        c.validate().unwrap();
        assert_eq!(
            c.text_config.moe().unwrap(),
            MoeDims {
                num_experts: 4,
                top_k: 3,
                moe_intermediate_size: 6,
                shared_expert_intermediate_size: 10,
            }
        );
        assert!(tiny_cfg().text_config.moe().is_none());
    }

    /// The locked MoE-LoRA policy is STRUCTURAL: the trainable-var count on
    /// the sparse model equals the dense formula (mixer projections + the
    /// shared expert's three — nothing for the router, the packed experts, or
    /// the sigmoid gate), every var is a 2-D adapter factor, gradients reach
    /// the full set, and on the linear-attention layers (where the industrial
    /// recipe adapts no mixer projection) the shared expert's adapters are
    /// the ONLY trainable surface — a nonzero gradient there proves the
    /// sigmoid-gated shared path is both adapted and grad-reached.
    #[test]
    fn moe_lora_grads_flow_and_the_routed_side_stays_frozen() {
        let model = armed_moe_model();
        let vars = model.trainable_vars();
        // Same formula as the dense tiny model: 3 linear layers x 3 (shared
        // expert) + the full layer's q,k,v,o + 3; 2 vars per projection.
        assert_eq!(vars.len(), (3 * 3 + 7) * 2);
        assert!(
            vars.iter().all(|v| v.dims().len() == 2),
            "a non-2-D trainable var — a packed routed weight leaked into the adapter set"
        );

        let logits = model.forward(&ids(9)).unwrap();
        assert_eq!(logits.dims(), &[1, 9, 24]);
        let grads = model.backward(&probe_loss(&logits)).unwrap();
        let cov = grad_coverage(&vars, &grads).unwrap();
        assert!(cov.is_ok(), "grad coverage: {cov:?}");

        // Layer 0 is linear-attention: under the industrial recipe its ONLY
        // adapters are the shared expert's (gate, up, down — vars 0..6).
        let shared_a = &vars[0];
        let g = grads.get(shared_a).expect("shared-expert A missing");
        assert!(
            max_abs_diff(g, &g.zeros_like().unwrap()) > 1e-8,
            "no gradient reached the shared expert through the sigmoid-gated path"
        );
    }

    /// The manifest recipe string carries the feed-forward menu: a dense and
    /// an `MoE` model with aliasing widths produce positionally IDENTICAL var
    /// lists (the mlp flags bind the layer MLP vs the shared expert), so the
    /// checkpoint cross-check must see the menu to catch a dense-vs-MoE
    /// confusion. Dense models keep the historical string (back-compat).
    #[test]
    fn moe_lora_recipe_string_carries_the_menu() {
        assert_eq!(
            GradModel::lora_recipe(&tiny_moe_model()).as_deref(),
            Some("attn:qkvo|mlp:gud|gdn:-|ffn:moe")
        );
        assert_eq!(
            GradModel::lora_recipe(&tiny_model()).as_deref(),
            Some("attn:qkvo|mlp:gud|gdn:-"),
            "the dense recipe string must stay historical (manifest back-compat)"
        );
    }

    /// Adapter semantics through the sparse menu: zero-B init is a no-op
    /// (enabled == disabled exactly); a trained B changes the output and the
    /// toggle restores the base.
    #[test]
    fn moe_adapter_toggle_flows_through_the_shared_expert() {
        let mut fresh = tiny_moe_model();
        let input = ids(7);
        let on = fresh.forward(&input).unwrap();
        fresh.set_adapter_enabled(false);
        let off = fresh.forward(&input).unwrap();
        assert_eq!(max_abs_diff(&on, &off), 0.0, "zero-B init must be a no-op");

        let mut armed = armed_moe_model();
        let trained = armed.forward(&input).unwrap();
        armed.set_adapter_enabled(false);
        let base = armed.forward(&input).unwrap();
        assert!(
            max_abs_diff(&trained, &base) > 1e-6,
            "a trained B must change the sparse forward"
        );
        assert_eq!(
            max_abs_diff(&base, &off),
            0.0,
            "disabling the adapter must restore the base model exactly"
        );
    }

    /// The cached merged decoder over the sparse menu: prefill, the
    /// multi-token continuation at an offset, and token-by-token decode all
    /// match the uncached forward (routing is per-token, so the `MoE` slot is
    /// stateless across the cache boundary — this gate pins exactly that).
    #[test]
    fn moe_merged_decoder_matches_uncached() {
        let model = armed_moe_model();
        let full = ids(9);
        let uncached = model.forward(&full).unwrap();

        // Vacuity guard: the armed adapters must displace the forward by
        // MORE than the envelope below (measured ~3e-1, ~120x MERGED_TOL),
        // or the merged comparison would not bind adapter FOLDING at all —
        // a merged decoder built from base weights would pass.
        let base = tiny_moe_model().forward(&full).unwrap();
        assert!(
            max_abs_diff(&uncached, &base) > MERGED_TOL,
            "armed adapters inside the merged envelope — the folding gate went vacuous"
        );

        let mut dec = model.merged_decoder().unwrap();
        let prefill = dec.forward(&full, 0).unwrap();
        let d = max_abs_diff(&uncached, &prefill);
        assert!(d <= MERGED_TOL, "sparse prefill vs uncached diff {d}");

        for p in [1usize, 4] {
            let mut dec = model.merged_decoder().unwrap();
            let head = full.narrow(1, 0, p).unwrap();
            let rest = full.narrow(1, p, 9 - p).unwrap();
            let first = dec.forward(&head, 0).unwrap();
            let second = dec.forward(&rest, p).unwrap();
            let cached = Tensor::cat(&[first, second], 1).unwrap();
            let d = max_abs_diff(&uncached, &cached);
            assert!(d <= MERGED_TOL, "sparse split at {p}: diff {d}");
        }

        let mut dec = model.merged_decoder().unwrap();
        let mut steps = Vec::new();
        for t in 0..9 {
            let tok = full.narrow(1, t, 1).unwrap();
            steps.push(dec.forward(&tok, t).unwrap());
        }
        let stepped = Tensor::cat(&steps, 1).unwrap();
        let d = max_abs_diff(&uncached, &stepped);
        assert!(d <= MERGED_TOL, "sparse token-by-token decode diff {d}");
    }

    /// P7 over the sparse menu: the checkpointed segment closure must rebuild
    /// the `MoE` feed-forward (host-side routing recomputes deterministically
    /// in the re-run); stitched gradients match the uncut backward, the
    /// boundary cut holds, and a detached walk captures no tape.
    #[test]
    fn moe_checkpointed_gradients_match_the_uncut_backward() {
        let mut model = armed_moe_model();
        let input = ids(9);
        let vars = model.trainable_vars();

        let plain = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();
        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();
        assert_grads_match(&plain, &stitched, &vars, 1e-5);
        // Non-vacuity: real gradients flow.
        assert!(vars.iter().any(|v| {
            let g = plain.get(v).unwrap();
            max_abs_diff(g, &g.zeros_like().unwrap()) > 1e-6
        }));

        // A detached walk captures no tape (the stitched backward above
        // consumed the pending one).
        let _ = model.forward_detached(&ids(5)).unwrap();
        let scalar = Tensor::zeros((), DType::F32, &dev()).unwrap();
        let err = model.backward(&scalar).unwrap_err();
        assert!(err.to_string().contains("no checkpointed forward"));

        // The cut: bypassing the stitching reaches no layer var.
        let raw = probe_loss(&model.forward(&input).unwrap())
            .backward()
            .unwrap();
        assert!(
            vars.iter().all(|v| raw.get(v).is_none()),
            "a layer var is on the loss tape — the boundary cut is not happening"
        );
    }

    /// The narrowed scoring forward (PR-B) over the sparse menu: routing runs
    /// full-width in the layer walk (the narrow folds into norm-and-head
    /// AFTER the layers), so values and adapter gradients must match
    /// `forward` + narrow exactly, plain and checkpointed.
    #[test]
    fn moe_narrowed_forward_matches_the_full_walk() {
        let mut model = armed_moe_model();
        let input = ids(9);
        let (start, len) = (3, 4);
        let vars = model.trainable_vars();

        let full = model
            .forward(&input)
            .unwrap()
            .narrow(1, start, len)
            .unwrap();
        let narrowed = GradModel::forward_narrowed(&model, &input, start, len).unwrap();
        assert_eq!(full.dims(), narrowed.dims());
        assert_eq!(
            max_abs_diff(&full, &narrowed),
            0.0,
            "narrowed values diverged"
        );

        let g_full = model.backward(&probe_loss(&full)).unwrap();
        let g_narrow = model.backward(&probe_loss(&narrowed)).unwrap();
        assert_grads_match(&g_full, &g_narrow, &vars, 0.0);

        model.set_activation_checkpointing(true);
        let stitched = model
            .backward(&probe_loss(
                &model.forward_narrowed(&input, start, len).unwrap(),
            ))
            .unwrap();
        assert_grads_match(&g_narrow, &stitched, &vars, 1e-5);
    }

    // ---- full fine-tuning (PR-E) ---------------------------------------------

    fn tiny_full_ft_model() -> Qwen3_5GradModel {
        let cfg = tiny_cfg();
        Qwen3_5GradModel::load_full_ft(&cfg, weight_map(&cfg.text_config), DType::F32, &dev())
            .unwrap()
    }

    fn tiny_full_ft_moe_model() -> Qwen3_5GradModel {
        let cfg = tiny_moe_cfg();
        Qwen3_5GradModel::load_full_ft(&cfg, weight_map(&cfg.text_config), DType::F32, &dev())
            .unwrap()
    }

    /// The full-FT positional contract: every base weight registers, in load
    /// order. Tiny dense tied: embed(1) + 3 linear layers × (`GatedDeltaNet`
    /// 9 + MLP 3 + 2 layer norms) + the full layer (attn 4 + q/k norms 2 +
    /// MLP 3 + 2 layer norms) + final norm = 55.
    #[test]
    fn full_ft_registers_every_base_weight_in_load_order() {
        let model = tiny_full_ft_model();
        assert!(model.is_full_ft());
        let vars = model.trainable_vars();
        assert_eq!(vars.len(), 55, "dense tied var count");
        assert_eq!(vars[0].dims(), &[24, 8], "vars[0] must be the embedding");
        // Layer 0 (GatedDeltaNet): conv taps and A_log at positions 5/6 —
        // stored in the on-disk [conv_dim, 1, kernel] layout, as fetched.
        assert_eq!(vars[5].dims(), &[32, 1, 4], "vars[5] must be the conv taps");
        assert_eq!(vars[6].dims(), &[4], "vars[6] must be A_log");
    }

    /// The recipe surface of the mode: `lora_recipe` reports `"full-ft"` (the
    /// manifest cross-check's mode guard) and `lora_targets` is the empty
    /// recipe (there are no adapters).
    #[test]
    fn full_ft_recipe_reports_the_mode() {
        let model = tiny_full_ft_model();
        assert_eq!(
            GradModel::lora_recipe(&model).as_deref(),
            Some("full-ft"),
            "the manifest cross-check must see the mode"
        );
        assert!(
            !model.lora_targets().any(),
            "full-FT records the empty adapter recipe"
        );
    }

    /// An untied head is one more registered var (position 1, after embed).
    #[test]
    fn full_ft_untied_head_joins_the_registry() {
        let untied_cfg = Qwen3_5Config {
            model_type: Some("qwen3_5".to_string()),
            tie_word_embeddings: None,
            text_config: tiny_text_cfg(false),
        };
        let untied = Qwen3_5GradModel::load_full_ft(
            &untied_cfg,
            weight_map(&untied_cfg.text_config),
            DType::F32,
            &dev(),
        )
        .unwrap();
        let vars = untied.trainable_vars();
        assert_eq!(vars.len(), 56, "untied adds lm_head");
        assert_eq!(vars[1].dims(), &[24, 8], "vars[1] must be the untied head");
    }

    /// `MoE` full-FT: each layer's feed-forward contributes 7 vars (packed
    /// `gate_up` + packed `down` + router + shared g/u/d + shared gate) — the
    /// packed routed tensors as ONE var each (per-expert vars would leave the
    /// packed forward tensors stale under `Var::set`) — for 71 total, and the
    /// recipe carries the menu.
    #[test]
    fn full_ft_moe_packs_each_routed_tensor_as_one_var() {
        let moe = tiny_full_ft_moe_model();
        let moe_vars = moe.trainable_vars();
        assert_eq!(moe_vars.len(), 71, "MoE tied var count");
        // Layer 0 feed-forward: the packed tensors register first (E=4, m=6,
        // h=8 — all axes distinct, so a packing transpose cannot alias).
        assert_eq!(
            moe_vars[10].dims(),
            &[4, 12, 8],
            "packed gate_up [E, 2m, h] must be one var"
        );
        assert_eq!(
            moe_vars[11].dims(),
            &[4, 8, 6],
            "packed down [E, h, m] must be one var"
        );
        assert_eq!(
            GradModel::lora_recipe(&moe).as_deref(),
            Some("full-ft|ffn:moe"),
            "the MoE menu must ride the full-FT recipe"
        );
    }

    /// Two identically configured loads must produce the same var order with
    /// the same values — the positional checkpoint contract is a pure
    /// function of the config.
    #[test]
    fn full_ft_var_order_is_reproducible_across_loads() {
        let (a, b) = (tiny_full_ft_moe_model(), tiny_full_ft_moe_model());
        let (va, vb) = (a.trainable_vars(), b.trainable_vars());
        assert_eq!(va.len(), vb.len());
        for (i, (x, y)) in va.iter().zip(&vb).enumerate() {
            assert_eq!(x.dims(), y.dims(), "var {i}: shape order drifted");
            assert_eq!(
                max_abs_diff(x.as_tensor(), y.as_tensor()),
                0.0,
                "var {i}: value order drifted — the positional contract broke"
            );
        }
    }

    /// Full-FT must not perturb the forward (same weights ⇒ logits identical
    /// to a `LoRA`-mode load of the same map — zero-B adapters are exact
    /// no-ops), and after one backward EVERY var must hold a finite, nonzero
    /// gradient: the canary against candle optimizers silently skipping
    /// grad-less params, now over the whole model — embed, conv taps,
    /// `A_log`/`dt_bias`, every norm, and on `MoE` the router, the packed
    /// experts (hit-expert rows), and the sigmoid gate.
    #[test]
    fn full_ft_forward_matches_lora_mode_and_grads_reach_every_var() {
        for (full, lora, tag) in [
            (tiny_full_ft_model(), tiny_model(), "dense"),
            (tiny_full_ft_moe_model(), tiny_moe_model(), "moe"),
        ] {
            let input = ids(9);
            let got = full.forward(&input).unwrap();
            let want = lora.forward(&input).unwrap();
            assert_eq!(
                max_abs_diff(&got, &want),
                0.0,
                "{tag}: the full-FT forward diverged from the same weights"
            );

            let vars = full.trainable_vars();
            let grads = full.backward(&probe_loss(&got)).unwrap();
            let cov = grad_coverage(&vars, &grads).unwrap();
            assert_eq!(
                cov.present, cov.total,
                "{tag}: a var is missing from the grad store"
            );
            assert_eq!(cov.nonfinite, 0, "{tag}: non-finite gradient");
            assert_eq!(
                cov.nonzero, cov.total,
                "{tag}: a var got an all-zero gradient"
            );
        }
    }

    /// Central-difference gradcheck on the full-FT hazard weights: the GDN
    /// conv taps and `A_log` (both had load-time transforms before this mode,
    /// and the conv kernel had only ever run with frozen weights) plus an
    /// embedding entry (`index_select` backward + the tied head). F64
    /// end-to-end; each probe at the var's max-|grad| entry with a
    /// non-vacuity floor (near-zero entries measure CPU-dependent
    /// cancellation, not the gradient — the runner-pool lesson).
    #[test]
    #[allow(clippy::print_stderr)] // the measured rel is the calibration record
    fn full_ft_backward_passes_a_finite_difference_gradcheck() {
        let cfg = tiny_cfg();
        let map_f64: HashMap<String, Tensor> = weight_map(&cfg.text_config)
            .into_iter()
            .map(|(k, t)| (k, t.to_dtype(DType::F64).unwrap()))
            .collect();
        let model = Qwen3_5GradModel::load_full_ft(&cfg, map_f64, DType::F64, &dev()).unwrap();
        let input = ids(5);
        let vars = model.trainable_vars();

        let grads = model
            .backward(&probe_loss(&model.forward(&input).unwrap()))
            .unwrap();

        // The FD noise floor here is NOT F64 cancellation: the qwen3_5
        // forward pins several internals to F32 by reference convention (both
        // norm flavors normalize in F32; the GDN `g` factor is always F32),
        // so even an F64-loaded model has an F32-quantized loss surface.
        // Measured on the embed probe: rel 4.1e-4 at eps 1e-5 vs 2.6e-2 at
        // eps 1e-6 — ∝ 1/eps, quantization noise, not truncation; a smaller
        // analytic gradient sits proportionally higher in it. At eps 1e-4 the
        // dev-host floor is embed 2.8e-4 / conv 4.7e-4 / A_log 4.0e-3. This
        // noise is F32 reassociation — the class the runner pool spreads
        // ~10x — so the gate is 3e-2: ~7x above the measured worst and ~17x
        // below the smallest real-bug signal (a missing tied-head/lookup
        // path or a wrong factor is rel ≳ 0.5).
        let eps = 1e-4f64;
        for (var, tag) in [
            (&vars[0], "embed"),
            (&vars[5], "conv taps"),
            (&vars[6], "A_log"),
        ] {
            let dims = var.dims().to_vec();
            let g = grads
                .get(var)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f64>()
                .unwrap();
            let k = (0..g.len())
                .max_by(|&a, &b| g[a].abs().total_cmp(&g[b].abs()))
                .unwrap();
            let analytic = g[k];
            assert!(
                analytic.abs() > 1e-6,
                "{tag}: strongest gradient entry {analytic} is inside the FD noise floor — \
                 the probe is vacuous"
            );
            let orig: Vec<f64> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            let loss_at = |delta: f64| -> f64 {
                let mut bent = orig.clone();
                bent[k] += delta;
                var.set(&Tensor::from_vec(bent, dims.clone(), &dev()).unwrap())
                    .unwrap();
                probe_loss(&model.forward(&input).unwrap())
                    .to_scalar::<f64>()
                    .unwrap()
            };
            let numeric = (loss_at(eps) - loss_at(-eps)) / (2.0 * eps);
            loss_at(0.0); // restore the entry
            let rel = (analytic - numeric).abs() / analytic.abs().max(1e-8);
            eprintln!("[full-FT FD] {tag} probe {k}: analytic={analytic:e}, rel={rel:e}");
            assert!(
                rel <= 3e-2,
                "{tag}: FD gradcheck failed: analytic={analytic}, numeric={numeric}, rel={rel}"
            );
        }
    }

    /// Full-FT merged decoders are PURE value copies (no adapter fold): the
    /// cached walk must match the uncached forward within the merged
    /// envelope, on both menus.
    #[test]
    fn full_ft_merged_decoder_matches_the_uncached_forward() {
        for (model, tag) in [
            (tiny_full_ft_model(), "dense"),
            (tiny_full_ft_moe_model(), "moe"),
        ] {
            let input = ids(7);
            let uncached = model.forward(&input).unwrap();
            let mut dec = model.merged_decoder().unwrap();
            let cached = dec.forward(&input, 0).unwrap();
            let d = max_abs_diff(&uncached, &cached);
            assert!(d <= MERGED_TOL, "{tag}: merged diverged by {d}");
        }
    }

    /// THE deep-copy mutation gate: an optimizer step after the decoder is
    /// built must NOT change the decoder's output — `Var::set` mutates
    /// storage in place, so a storage-sharing "snapshot" would silently track
    /// training mid-rollout — and a decoder REBUILT from the stepped weights
    /// must differ (the vacuity guard: the step really displaced the model).
    /// Runs on both menus: the `MoE` arm additionally pins the packed-expert
    /// / router / sigmoid-gate deep copies.
    #[test]
    fn full_ft_merged_decoder_is_a_value_snapshot_not_a_storage_alias() {
        use crate::optim::FerrlAdamW;
        use candle_nn::{Optimizer, ParamsAdamW};
        for (model, tag) in [
            (tiny_full_ft_model(), "dense"),
            (tiny_full_ft_moe_model(), "moe"),
        ] {
            let input = ids(7);
            let mut dec = model.merged_decoder().unwrap();
            let before = dec.forward(&input, 0).unwrap();

            let vars = model.trainable_vars();
            let w0 = vars[0].as_tensor().copy().unwrap();
            let grads = model
                .backward(&probe_loss(&model.forward(&input).unwrap()))
                .unwrap();
            let mut opt = FerrlAdamW::new(
                vars.clone(),
                ParamsAdamW {
                    lr: 1e-2,
                    ..Default::default()
                },
            )
            .unwrap();
            opt.step(&grads).unwrap();
            assert!(
                max_abs_diff(&w0, vars[0].as_tensor()) > 0.0,
                "{tag}: the optimizer step did not move the weights — the gate is vacuous"
            );

            dec.reset_cache();
            let after = dec.forward(&input, 0).unwrap();
            assert_eq!(
                max_abs_diff(&before, &after),
                0.0,
                "{tag}: the merged decoder tracked an optimizer step — a snapshot weight \
                 shares storage with a trainable var"
            );

            let mut rebuilt = model.merged_decoder().unwrap();
            let moved = rebuilt.forward(&input, 0).unwrap();
            assert!(
                max_abs_diff(&before, &moved) > 0.0,
                "{tag}: the rebuilt decoder shows no displacement — the step changed nothing \
                 the decoder can see"
            );
        }
    }

    /// Full-FT has no adapters: the toggle is a no-op (the flag stays true,
    /// the forward unchanged) and the trait reports it — the seam eval's
    /// fail-loud base-vs-trained check stands on.
    #[test]
    fn full_ft_has_no_adapters_and_ignores_the_toggle() {
        let mut model = tiny_full_ft_model();
        assert!(!GradModel::has_adapters(&model));
        let input = ids(5);
        let before = model.forward(&input).unwrap();
        model.set_adapter_enabled(false);
        assert!(
            model.adapter_enabled,
            "the flag must stay true (the eval detection contract)"
        );
        let after = model.forward(&input).unwrap();
        assert_eq!(max_abs_diff(&before, &after), 0.0);
    }

    /// Full-FT × activation checkpointing fails loud at the grad forward (the
    /// boundary tape would silently drop the embedding gradient — flagged
    /// follow-up), and turning checkpointing back off recovers.
    #[test]
    fn full_ft_rejects_activation_checkpointing_loudly() {
        let mut model = tiny_full_ft_model();
        model.set_activation_checkpointing(true);
        let err = model.forward(&ids(4)).unwrap_err();
        assert!(
            err.to_string().contains("not supported in full"),
            "got: {err}"
        );
        model.set_activation_checkpointing(false);
        model.forward(&ids(4)).unwrap();
    }

    /// The v2 positional checkpoint mechanism works unchanged over the
    /// full-FT registry order: save, displace every var, load restores
    /// bit-exactly, and the manifest carries the `"full-ft"` recipe the
    /// trainer's resume cross-check guards on.
    #[test]
    fn full_ft_checkpoint_round_trips_positionally() {
        let model = tiny_full_ft_model();
        let vars = model.trainable_vars();
        let dir = std::env::temp_dir().join(format!("ferrl-full-ft-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let recipe = GradModel::lora_recipe(&model);
        crate::checkpoint::save_adapter(&dir, &vars, 3, recipe.as_deref()).unwrap();

        let originals: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().copy().unwrap()).collect();
        for v in &vars {
            v.set(&(v.as_tensor() * 2.0).unwrap()).unwrap();
        }
        let manifest = crate::checkpoint::load_adapter(&dir, &vars).unwrap();
        assert_eq!(manifest.lora_recipe.as_deref(), Some("full-ft"));
        assert_eq!(manifest.num_vars, vars.len());
        for (i, (v, orig)) in vars.iter().zip(&originals).enumerate() {
            assert_eq!(
                max_abs_diff(v.as_tensor(), orig),
                0.0,
                "var {i} did not restore bit-exactly"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
