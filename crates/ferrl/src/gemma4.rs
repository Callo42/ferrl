//! Gemma 4 text-model config parsing and validation.
//!
//! This module is the native Gemma 4 support surface: it accepts the public
//! Hugging Face `config.json` shapes (`model_type: "gemma4"` or
//! `"gemma4_unified"`, decoder under `text_config`) and fails loud on text
//! variants the native ferrl forward does not yet implement. The initial target
//! is the dense Gemma 4 text decoder; vision/audio wrappers are tolerated
//! because text-only RL never requests those tensors.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{bail, DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::ops::softmax;
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{Activation, Module, VarBuilder};
use serde::Deserialize;

use crate::blocks::{frozen_linear, repeat_kv, rope_partial, windowed, RotaryTables};
use crate::lora::{
    BaseQuantization, DenseLoraTargets, FrozenLinearSnapshot, Proj, ProjLoadOptions,
};
use crate::model::{CachedDecoder, GradModel};
use crate::nn::RmsNorm;
use crate::remat::{stitched_backward, RematTape};
use crate::telemetry::DecoderCacheSnapshot;
use crate::tensor_parallel::TensorParallelPlan;

/// The checkpoint prefix used by public Gemma 4 conditional-generation
/// checkpoints for the text decoder.
pub const CKPT_PREFIX: &str = "model.language_model";

/// A Gemma 4 decoder layer's attention kind (`text_config.layer_types[i]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Gemma4LayerType {
    /// Local sliding-window attention.
    #[serde(rename = "sliding_attention")]
    SlidingAttention,
    /// Full global attention.
    #[serde(rename = "full_attention")]
    FullAttention,
}

/// Full-attention rope settings.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4FullAttentionRope {
    /// Rope family; the 31B dense text model uses proportional `RoPE`.
    pub rope_type: String,
    /// Frequency base.
    pub rope_theta: f64,
    /// Fraction of each global-head dim that rotates.
    pub partial_rotary_factor: f64,
}

/// Sliding-window attention rope settings.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4SlidingAttentionRope {
    /// Rope family; the 31B dense text model uses default `RoPE`.
    pub rope_type: String,
    /// Frequency base.
    pub rope_theta: f64,
}

/// The `rope_parameters` object in Gemma 4 text configs.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4RopeParameters {
    /// Full-attention rope parameters.
    pub full_attention: Gemma4FullAttentionRope,
    /// Sliding-window attention rope parameters.
    pub sliding_attention: Gemma4SlidingAttentionRope,
}

fn default_gemma4_text_model_type() -> Option<String> {
    Some("gemma4_text".to_string())
}

fn is_supported_top_level_model_type(model_type: &str) -> bool {
    matches!(model_type, "gemma4" | "gemma4_unified")
}

fn is_supported_text_model_type(model_type: &str) -> bool {
    matches!(model_type, "gemma4_text" | "gemma4_unified_text")
}

/// The `text_config` object consumed by ferrl's native Gemma 4 path.
///
/// Unknown keys are tolerated: shipped checkpoints include wrapper and modality
/// metadata that text-only RL does not use. Known unsupported values are
/// rejected in [`validate`](Self::validate) so a foreign Gemma-family config
/// cannot silently load through the wrong architecture.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    /// Text decoder model type (`"gemma4_text"` or `"gemma4_unified_text"`).
    #[serde(default = "default_gemma4_text_model_type")]
    pub model_type: Option<String>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Per-layer input vocab size; dense 31B has the same vocabulary and no PLE.
    pub vocab_size_per_layer_input: usize,
    /// Residual stream width.
    pub hidden_size: usize,
    /// Per-layer input hidden width. Dense 31B has no per-layer embeddings.
    pub hidden_size_per_layer_input: usize,
    /// MLP inner width.
    pub intermediate_size: usize,
    /// Decoder layer count.
    pub num_hidden_layers: usize,
    /// Query head count.
    pub num_attention_heads: usize,
    /// Sliding-attention KV head count.
    pub num_key_value_heads: usize,
    /// Full-attention global KV head count.
    pub num_global_key_value_heads: usize,
    /// Number of shared KV layers. The initial dense 31B target has none.
    pub num_kv_shared_layers: usize,
    /// Per-head width for sliding attention.
    pub head_dim: usize,
    /// Per-global-head width for full attention.
    pub global_head_dim: usize,
    /// MLP activation.
    pub hidden_activation: String,
    /// Norm epsilon.
    pub rms_norm_eps: f64,
    /// Maximum context length.
    pub max_position_embeddings: usize,
    /// Sliding attention window.
    pub sliding_window: usize,
    /// Whether token embeddings and LM head are tied.
    pub tie_word_embeddings: bool,
    /// Whether attention projections use bias.
    #[serde(default)]
    pub attention_bias: bool,
    /// Attention dropout. RL fine-tuning uses the eager deterministic path.
    #[serde(default)]
    pub attention_dropout: f64,
    /// Gemma 4 global layers unify K and V.
    pub attention_k_eq_v: bool,
    /// Final-logit softcap value.
    pub final_logit_softcapping: f64,
    /// Per-layer attention menu.
    pub layer_types: Vec<Gemma4LayerType>,
    /// Rope settings for both attention kinds.
    pub rope_parameters: Gemma4RopeParameters,
    /// Whether generation uses a cache.
    #[serde(default)]
    pub use_cache: bool,
    /// The shipped text config marks bidirectional attention as vision-only.
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
    /// Alternate MLP layout. Not in the first dense target.
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    /// `MoE` enable flag.
    #[serde(default)]
    pub enable_moe_block: bool,
    /// Routed expert count, if this is an `MoE` member.
    #[serde(default)]
    pub num_experts: Option<usize>,
    /// Top-k routed experts, if this is an `MoE` member.
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    /// Per-expert MLP width, if this is an `MoE` member.
    #[serde(default)]
    pub expert_intermediate_size: Option<usize>,
    /// Alternate upstream name for per-expert MLP width.
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
}

/// A full Gemma 4 checkpoint config (`config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    /// Top-level model type (`"gemma4"` or `"gemma4_unified"`).
    #[serde(default)]
    pub model_type: Option<String>,
    /// Top-level tied-embedding flag. It must agree with the text config when
    /// present.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    /// Text decoder config.
    pub text_config: Gemma4TextConfig,
}

impl Gemma4Config {
    /// Parse and validate a Gemma 4 `config.json` string.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the JSON does not parse into the supported
    /// shape or any validation rule fails.
    pub fn from_json_str(json: &str) -> CandleResult<Self> {
        let cfg: Self = serde_json::from_str(json)
            .map_err(|e| candle_core::Error::Msg(format!("gemma4 config parse: {e}")))?;
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
                "gemma4 config read {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Self::from_json_str(&json)
    }

    /// Validate the composite config.
    ///
    /// # Errors
    ///
    /// Returns a candle error naming the unsupported field/value.
    pub fn validate(&self) -> CandleResult<()> {
        if let Some(mt) = &self.model_type {
            if !is_supported_top_level_model_type(mt) {
                bail!(
                    "gemma4 config: model_type {mt:?}, expected \"gemma4\" or \
                     \"gemma4_unified\""
                );
            }
        }
        if let Some(outer) = self.tie_word_embeddings {
            if outer != self.text_config.tie_word_embeddings {
                bail!(
                    "gemma4 config: outer tie_word_embeddings {outer} disagrees with \
                     text_config.tie_word_embeddings {}",
                    self.text_config.tie_word_embeddings
                );
            }
        }
        self.text_config.validate()
    }
}

impl Gemma4TextConfig {
    /// The number of rotated dims per full-attention global head.
    #[must_use]
    pub fn full_attention_rotary_dim(&self) -> usize {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rot = (self.global_head_dim as f64
            * self.rope_parameters.full_attention.partial_rotary_factor) as usize;
        rot
    }

    /// Fail loud on any config value the initial native text path does not
    /// implement.
    ///
    /// # Errors
    ///
    /// Returns a candle error naming the offending field and value.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> CandleResult<()> {
        let Some(model_type) = self.model_type.as_deref() else {
            bail!(
                "gemma4 config: text_config.model_type missing, expected \"gemma4_text\" or \
                 \"gemma4_unified_text\""
            );
        };
        if !is_supported_text_model_type(model_type) {
            bail!(
                "gemma4 config: text_config.model_type {:?}, expected \"gemma4_text\" or \
                 \"gemma4_unified_text\"",
                model_type
            );
        }
        for (name, v) in [
            ("vocab_size", self.vocab_size),
            (
                "vocab_size_per_layer_input",
                self.vocab_size_per_layer_input,
            ),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            (
                "num_global_key_value_heads",
                self.num_global_key_value_heads,
            ),
            ("head_dim", self.head_dim),
            ("global_head_dim", self.global_head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("sliding_window", self.sliding_window),
        ] {
            if v == 0 {
                bail!("gemma4 config: {name} must be > 0");
            }
        }
        if self.hidden_size_per_layer_input != 0 {
            bail!(
                "gemma4 config: hidden_size_per_layer_input {} unsupported (the initial \
                 dense text path does not implement per-layer embeddings)",
                self.hidden_size_per_layer_input
            );
        }
        if self.vocab_size_per_layer_input != self.vocab_size {
            bail!(
                "gemma4 config: vocab_size_per_layer_input {} must equal vocab_size {} for \
                 the initial dense text path",
                self.vocab_size_per_layer_input,
                self.vocab_size
            );
        }
        if self.enable_moe_block
            || self.num_experts.is_some()
            || self.top_k_experts.is_some()
            || self.expert_intermediate_size.is_some()
            || self.moe_intermediate_size.is_some()
        {
            bail!(
                "gemma4 config: MoE fields unsupported in the initial native path (first \
                 target is dense text checkpoints)"
            );
        }
        if self.hidden_activation != "gelu_pytorch_tanh" {
            bail!(
                "gemma4 config: hidden_activation {:?} unsupported (only \
                 \"gelu_pytorch_tanh\")",
                self.hidden_activation
            );
        }
        if self.attention_bias {
            bail!("gemma4 config: attention_bias=true unsupported (31B ships bias-free)");
        }
        if self.attention_dropout != 0.0 {
            bail!(
                "gemma4 config: attention_dropout {} unsupported (only 0.0)",
                self.attention_dropout
            );
        }
        if !self.attention_k_eq_v {
            bail!(
                "gemma4 config: attention_k_eq_v=false unsupported (global layers unify K/V \
                 in the initial target)"
            );
        }
        if self.num_kv_shared_layers != 0 {
            bail!(
                "gemma4 config: num_kv_shared_layers {} unsupported (initial target uses 0)",
                self.num_kv_shared_layers
            );
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            bail!(
                "gemma4 config: num_attention_heads {} not divisible by num_key_value_heads {}",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_global_key_value_heads)
        {
            bail!(
                "gemma4 config: num_attention_heads {} not divisible by \
                 num_global_key_value_heads {}",
                self.num_attention_heads,
                self.num_global_key_value_heads
            );
        }
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "gemma4 config: layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if !self.layer_types.contains(&Gemma4LayerType::FullAttention) {
            bail!("gemma4 config: layer_types must contain at least one full_attention layer");
        }
        if self.layer_types.last() != Some(&Gemma4LayerType::FullAttention) {
            bail!("gemma4 config: final layer must be full_attention");
        }
        if !self.tie_word_embeddings {
            bail!(
                "gemma4 config: tie_word_embeddings=false unsupported (initial target uses \
                 tied embeddings)"
            );
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0) {
            bail!(
                "gemma4 config: rms_norm_eps {} must be a positive number",
                self.rms_norm_eps
            );
        }
        if !(self.final_logit_softcapping.is_finite() && self.final_logit_softcapping > 0.0) {
            bail!(
                "gemma4 config: final_logit_softcapping {} must be a positive number",
                self.final_logit_softcapping
            );
        }
        if self.use_bidirectional_attention.as_deref() != Some("vision") {
            bail!(
                "gemma4 config: use_bidirectional_attention {:?} unsupported (text path \
                 expects \"vision\")",
                self.use_bidirectional_attention
            );
        }
        if self.use_double_wide_mlp {
            bail!("gemma4 config: use_double_wide_mlp=true unsupported");
        }

        let full = &self.rope_parameters.full_attention;
        if full.rope_type != "proportional" {
            bail!(
                "gemma4 config: full_attention rope_type {:?} unsupported (only \
                 \"proportional\")",
                full.rope_type
            );
        }
        if !(full.rope_theta.is_finite() && full.rope_theta > 0.0) {
            bail!(
                "gemma4 config: full_attention rope_theta {} must be a positive number",
                full.rope_theta
            );
        }
        let prf = full.partial_rotary_factor;
        if !(prf > 0.0 && prf <= 1.0) {
            bail!("gemma4 config: partial_rotary_factor {prf} must be in (0, 1]");
        }
        let rot_exact = self.global_head_dim as f64 * prf;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rot = rot_exact as usize;
        #[allow(clippy::cast_precision_loss)]
        if (rot as f64 - rot_exact).abs() > 1e-9 || rot == 0 || !rot.is_multiple_of(2) {
            bail!(
                "gemma4 config: global_head_dim {} * partial_rotary_factor {prf} = \
                 {rot_exact} must be a nonzero even integer",
                self.global_head_dim
            );
        }

        let sliding = &self.rope_parameters.sliding_attention;
        if sliding.rope_type != "default" {
            bail!(
                "gemma4 config: sliding_attention rope_type {:?} unsupported (only \
                 \"default\")",
                sliding.rope_type
            );
        }
        if !(sliding.rope_theta.is_finite() && sliding.rope_theta > 0.0) {
            bail!(
                "gemma4 config: sliding_attention rope_theta {} must be a positive number",
                sliding.rope_theta
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Build a [`VarBuilder`] over a Gemma 4 checkpoint directory, supporting both
/// single-file and sharded safetensors layouts.
///
/// Vision/audio tensors may be present in the map; the native text loader only
/// requests `model.language_model.*`.
///
/// # Errors
///
/// Returns a candle error if no supported safetensors layout is found, a shard
/// fails to read, or the index JSON is malformed.
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

/// The raw checkpoint tensor map behind [`varbuilder_from_pretrained`].
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
        let mut names: Vec<String> = index
            .weight_map
            .into_iter()
            .filter_map(|(tensor, shard)| is_gemma4_text_tensor(&tensor).then_some(shard))
            .collect();
        if names.is_empty() {
            bail!(
                "gemma4 loader: no {CKPT_PREFIX}.* tensors listed in {}",
                index_path.display()
            );
        }
        names.sort();
        names.dedup();
        names.into_iter().map(|n| dir.join(n)).collect()
    } else if single_path.is_file() {
        vec![single_path]
    } else {
        bail!(
            "gemma4 loader: neither model.safetensors.index.json nor model.safetensors found \
             in {}",
            dir.display()
        );
    };
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for file in files {
        let shard = candle_core::safetensors::load(&file, device)?;
        tensors.extend(shard);
    }
    tensors.retain(|name, _| is_gemma4_text_tensor(name));
    if tensors.is_empty() {
        bail!(
            "gemma4 loader: no {CKPT_PREFIX}.* tensors loaded from {}",
            dir.display()
        );
    }
    Ok(tensors)
}

fn is_gemma4_text_tensor(name: &str) -> bool {
    name.strip_prefix(CKPT_PREFIX)
        .is_some_and(|suffix| suffix.starts_with('.'))
}

// ---------------------------------------------------------------------------
// Native dense text model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Gemma4Rotary {
    sliding: RotaryTables,
    full: RotaryTables,
    sliding_rot_dim: usize,
}

impl Gemma4Rotary {
    fn new(cfg: &Gemma4TextConfig, dtype: DType, device: &Device) -> CandleResult<Self> {
        let sliding = RotaryTables::new(
            cfg.head_dim,
            cfg.rope_parameters.sliding_attention.rope_theta,
            cfg.max_position_embeddings,
            dtype,
            device,
        )?;
        let full_rot_dim = cfg.full_attention_rotary_dim();
        let full = RotaryTables::with_inv_freq(
            proportional_inv_freq(
                full_rot_dim,
                cfg.global_head_dim,
                cfg.rope_parameters.full_attention.rope_theta,
            ),
            cfg.max_position_embeddings,
            dtype,
            device,
        )?;
        Ok(Self {
            sliding,
            full,
            sliding_rot_dim: cfg.head_dim,
        })
    }

    fn apply(
        &self,
        kind: Gemma4LayerType,
        x: &Tensor,
        offset: usize,
        seq_len: usize,
    ) -> CandleResult<Tensor> {
        match kind {
            Gemma4LayerType::SlidingAttention => {
                let (cos, sin) = self.sliding.slice_at(offset, seq_len)?;
                rope_partial(&x.contiguous()?, &cos, &sin, self.sliding_rot_dim)
            }
            Gemma4LayerType::FullAttention => {
                let (cos, sin) = self.full.slice_at(offset, seq_len)?;
                rope_slow(&x.contiguous()?, &cos, &sin)
            }
        }
    }
}

fn proportional_inv_freq(rot_dim: usize, head_dim: usize, theta: f64) -> Vec<f32> {
    let mut inv_freq: Vec<f32> = (0..rot_dim)
        .step_by(2)
        .map(|i| (1.0 / theta.powf(i as f64 / head_dim as f64)) as f32)
        .collect();
    // Gemma 4 proportional RoPE applies rotate-half over the whole global head:
    // unrotated pairs are represented as zero frequencies, not as a narrow
    // contiguous-prefix partial rotation.
    inv_freq.resize(head_dim / 2, 0.0);
    inv_freq
}

#[derive(Debug)]
struct Gemma4Masks {
    full: Option<Tensor>,
    sliding: Option<Tensor>,
}

impl Gemma4Masks {
    fn get(&self, kind: Gemma4LayerType) -> Option<&Tensor> {
        match kind {
            Gemma4LayerType::FullAttention => self.full.as_ref(),
            Gemma4LayerType::SlidingAttention => self.sliding.as_ref(),
        }
    }
}

fn masks_at(
    offset: usize,
    chunk_len: usize,
    sliding_window: usize,
    dtype: DType,
    device: &Device,
) -> CandleResult<Gemma4Masks> {
    let total = offset + chunk_len;
    let full = if chunk_len == 1 {
        None
    } else {
        Some(attention_mask_at(
            offset, chunk_len, total, None, dtype, device,
        )?)
    };
    let sliding = if chunk_len == 1 && total <= sliding_window {
        None
    } else {
        Some(attention_mask_at(
            offset,
            chunk_len,
            total,
            Some(sliding_window),
            dtype,
            device,
        )?)
    };
    Ok(Gemma4Masks { full, sliding })
}

fn merged_masks_at(
    offset: usize,
    chunk_len: usize,
    dtype: DType,
    device: &Device,
) -> CandleResult<Gemma4Masks> {
    let total = offset + chunk_len;
    let full = if chunk_len == 1 {
        None
    } else {
        Some(attention_mask_at(
            offset, chunk_len, total, None, dtype, device,
        )?)
    };
    Ok(Gemma4Masks {
        full,
        sliding: None,
    })
}

fn attention_mask_at(
    offset: usize,
    chunk_len: usize,
    total_keys: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> CandleResult<Tensor> {
    attention_mask_for_keys(
        offset,
        chunk_len,
        0,
        total_keys,
        sliding_window,
        dtype,
        device,
    )
}

fn attention_mask_for_keys(
    offset: usize,
    chunk_len: usize,
    key_start: usize,
    key_len: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> CandleResult<Tensor> {
    let mut data = Vec::with_capacity(chunk_len * key_len);
    for i in 0..chunk_len {
        let q_abs = offset + i;
        for j in 0..key_len {
            let k_abs = key_start + j;
            let causal = k_abs <= q_abs;
            let in_window = sliding_window.is_none_or(|w| q_abs.saturating_sub(k_abs) < w);
            data.push(if causal && in_window {
                0f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::from_vec(data, (1, 1, chunk_len, key_len), device)?.to_dtype(dtype)
}

/// One Gemma 4 text attention block, supporting both local sliding-window and
/// full global layers.
#[derive(Debug)]
struct Gemma4Attention {
    kind: Gemma4LayerType,
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Option<Proj>,
    o_proj: Proj,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
}

impl Gemma4Attention {
    fn load(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        proj_opts: ProjLoadOptions,
    ) -> CandleResult<Self> {
        let kind = cfg.layer_types[layer_idx];
        let full = kind == Gemma4LayerType::FullAttention;
        let head_dim = if full {
            cfg.global_head_dim
        } else {
            cfg.head_dim
        };
        let num_kv_heads = if full {
            cfg.num_global_key_value_heads
        } else {
            cfg.num_key_value_heads
        };
        let h = cfg.hidden_size;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;
        let v_proj = if full {
            None
        } else {
            Some(Proj::load_with_options(
                vb,
                "v_proj",
                (kv_out, h),
                targets.attn_v,
                proj_opts,
            )?)
        };
        Ok(Self {
            kind,
            q_proj: Proj::load_with_options(vb, "q_proj", (q_out, h), targets.attn_q, proj_opts)?,
            k_proj: Proj::load_with_options(vb, "k_proj", (kv_out, h), targets.attn_k, proj_opts)?,
            v_proj,
            o_proj: Proj::load_with_options(vb, "o_proj", (h, q_out), targets.attn_o, proj_opts)?,
            q_norm: RmsNorm::new(
                vb.pp("q_norm").get(head_dim, "weight")?,
                cfg.rms_norm_eps as f32,
            ),
            k_norm: RmsNorm::new(
                vb.pp("k_norm").get(head_dim, "weight")?,
                cfg.rms_norm_eps as f32,
            ),
            v_norm: RmsNorm::ones(head_dim, cfg.rms_norm_eps as f32, vb.dtype(), vb.device())?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads,
            num_kv_groups: cfg.num_attention_heads / num_kv_heads,
            head_dim,
            attn_hidden: q_out,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &Gemma4Rotary,
    ) -> CandleResult<Tensor> {
        self.forward_at(x, 0, mask, rot, None)
    }

    fn forward_at(
        &self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &Gemma4Rotary,
        cache: Option<&mut ConcatKvCache>,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let in_dtype = x.dtype();
        let tp = TensorParallelPlan::single();
        let q = self
            .q_proj
            .column_parallel_forward(x, tp, "attention_q_out")?;
        let k_raw = self
            .k_proj
            .column_parallel_forward(x, tp, "attention_k_out")?;
        let v_raw = match &self.v_proj {
            Some(v) => v.column_parallel_forward(x, tp, "attention_v_out")?,
            None => k_raw.clone(),
        };

        let q = q.reshape((b, l, self.num_heads, self.head_dim))?;
        let k = k_raw.reshape((b, l, self.num_kv_heads, self.head_dim))?;
        let v = v_raw.reshape((b, l, self.num_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?.transpose(1, 2)?;
        let k = rot.apply(self.kind, &k.transpose(1, 2)?, offset, l)?;
        let q = rot.apply(self.kind, &q, offset, l)?;

        let (k, v) = match cache {
            Some(c) => c.append(&k.contiguous()?, &v.contiguous()?)?,
            None => (k, v.contiguous()?),
        };
        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

        let mut scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(in_dtype)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.attn_hidden))?;
        self.o_proj
            .row_parallel_forward_partial(&ctx, tp, "attention_hidden")
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.q_proj.set_enabled(enabled);
        self.k_proj.set_enabled(enabled);
        if let Some(v) = &mut self.v_proj {
            v.set_enabled(enabled);
        }
        self.o_proj.set_enabled(enabled);
    }

    fn push_vars(&self, out: &mut Vec<Var>) {
        self.q_proj.push_vars(out);
        self.k_proj.push_vars(out);
        if let Some(v) = &self.v_proj {
            v.push_vars(out);
        }
        self.o_proj.push_vars(out);
    }
}

/// Gemma 4 text feed-forward block.
#[derive(Debug)]
struct Gemma4Mlp {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
    act: Activation,
}

impl Gemma4Mlp {
    fn load(
        cfg: &Gemma4TextConfig,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        proj_opts: ProjLoadOptions,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: Proj::load_with_options(
                vb,
                "gate_proj",
                (i, h),
                targets.mlp_gate,
                proj_opts,
            )?,
            up_proj: Proj::load_with_options(vb, "up_proj", (i, h), targets.mlp_up, proj_opts)?,
            down_proj: Proj::load_with_options(
                vb,
                "down_proj",
                (h, i),
                targets.mlp_down,
                proj_opts,
            )?,
            act: Activation::GeluPytorchTanh,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let tp = TensorParallelPlan::single();
        let g = self
            .gate_proj
            .column_parallel_forward(x, tp, "intermediate_size")?;
        let u = self
            .up_proj
            .column_parallel_forward(x, tp, "intermediate_size")?;
        let h = self.act.forward(&g)?.broadcast_mul(&u)?;
        self.down_proj
            .row_parallel_forward_partial(&h, tp, "intermediate_size")
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.gate_proj.set_enabled(enabled);
        self.up_proj.set_enabled(enabled);
        self.down_proj.set_enabled(enabled);
    }

    fn push_vars(&self, out: &mut Vec<Var>) {
        self.gate_proj.push_vars(out);
        self.up_proj.push_vars(out);
        self.down_proj.push_vars(out);
    }
}

#[derive(Debug)]
struct Gemma4Layer {
    kind: Gemma4LayerType,
    input_layernorm: RmsNorm,
    attn: Gemma4Attention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: Gemma4Mlp,
    post_feedforward_layernorm: RmsNorm,
    layer_scalar: Tensor,
}

impl Gemma4Layer {
    fn load(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        proj_opts: ProjLoadOptions,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        Ok(Self {
            kind: cfg.layer_types[layer_idx],
            input_layernorm: RmsNorm::new(vb.pp("input_layernorm").get(h, "weight")?, eps),
            attn: Gemma4Attention::load(cfg, layer_idx, &vb.pp("self_attn"), targets, proj_opts)?,
            post_attention_layernorm: RmsNorm::new(
                vb.pp("post_attention_layernorm").get(h, "weight")?,
                eps,
            ),
            pre_feedforward_layernorm: RmsNorm::new(
                vb.pp("pre_feedforward_layernorm").get(h, "weight")?,
                eps,
            ),
            mlp: Gemma4Mlp::load(cfg, &vb.pp("mlp"), targets, proj_opts)?,
            post_feedforward_layernorm: RmsNorm::new(
                vb.pp("post_feedforward_layernorm").get(h, "weight")?,
                eps,
            ),
            layer_scalar: vb.get(1, "layer_scalar")?,
        })
    }

    fn forward(&self, x: &Tensor, masks: &Gemma4Masks, rot: &Gemma4Rotary) -> CandleResult<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h, masks.get(self.kind), rot)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let x = x.broadcast_add(&h)?;
        let h2 = self.pre_feedforward_layernorm.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        let h2 = self.post_feedforward_layernorm.forward(&h2)?;
        let x = x.broadcast_add(&h2)?;
        x.broadcast_mul(&self.layer_scalar)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.attn.set_adapter_enabled(enabled);
        self.mlp.set_adapter_enabled(enabled);
    }

    fn push_vars(&self, out: &mut Vec<Var>) {
        self.attn.push_vars(out);
        self.mlp.push_vars(out);
    }
}

/// A grad-bearing, uncached dense Gemma 4 text forward with `LoRA`.
#[derive(Debug)]
pub struct Gemma4GradModel {
    embed: Tensor,
    layers: Vec<Gemma4Layer>,
    norm: RmsNorm,
    rot: Gemma4Rotary,
    hidden: usize,
    sliding_window: usize,
    embed_scale: f64,
    final_logit_softcap: f64,
    device: Device,
    dtype: DType,
    targets: DenseLoraTargets,
    adapter_enabled: bool,
    remat: bool,
    tape: RefCell<Option<RematTape>>,
}

impl Gemma4GradModel {
    /// Load the dense Gemma 4 text model with the industrial `LoRA` recipe.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the config is unsupported, weights are missing
    /// or mis-shaped, or adapter allocation fails.
    pub fn load(
        cfg: &Gemma4Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
    ) -> CandleResult<Self> {
        Self::load_with_adapter_dtype(cfg, vb, rank, alpha, vb.dtype())
    }

    /// As [`load`](Self::load), but with explicit adapter dtype.
    ///
    /// # Errors
    ///
    /// As [`load`](Self::load).
    pub fn load_with_adapter_dtype(
        cfg: &Gemma4Config,
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
            DenseLoraTargets::industrial(),
        )
    }

    /// Load with an explicit dense `LoRA` target recipe.
    ///
    /// # Errors
    ///
    /// As [`load`](Self::load), plus an error when `targets` selects no
    /// trainable projection.
    pub fn load_with_targets(
        cfg: &Gemma4Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        targets: DenseLoraTargets,
    ) -> CandleResult<Self> {
        Self::load_with_targets_and_base_quantization(
            cfg,
            vb,
            rank,
            alpha,
            adapter_dtype,
            targets,
            BaseQuantization::None,
        )
    }

    /// Load with an explicit frozen-base projection quantization mode.
    ///
    /// The trainable adapter recipe and positional checkpoint contract are
    /// unchanged; only frozen projection storage and rollout application differ.
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets), plus quantization
    /// shape/storage errors.
    pub fn load_with_targets_and_base_quantization(
        cfg: &Gemma4Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        targets: DenseLoraTargets,
        base_quantization: BaseQuantization,
    ) -> CandleResult<Self> {
        cfg.validate()?;
        if !targets.any() {
            bail!("Gemma4GradModel: DenseLoraTargets selects no projection");
        }
        let t = &cfg.text_config;
        let root = vb.pp(CKPT_PREFIX);
        let embed = root
            .pp("embed_tokens")
            .get((t.vocab_size, t.hidden_size), "weight")?;
        let layers_vb = root.pp("layers");
        let proj_opts = ProjLoadOptions::new(rank, alpha, adapter_dtype, base_quantization);
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            layers.push(Gemma4Layer::load(
                t,
                i,
                &layers_vb.pp(i),
                targets,
                proj_opts,
            )?);
        }
        Ok(Self {
            embed,
            layers,
            norm: RmsNorm::new(
                root.pp("norm").get(t.hidden_size, "weight")?,
                t.rms_norm_eps as f32,
            ),
            rot: Gemma4Rotary::new(t, vb.dtype(), vb.device())?,
            hidden: t.hidden_size,
            sliding_window: t.sliding_window,
            embed_scale: (t.hidden_size as f64).sqrt(),
            final_logit_softcap: t.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            targets,
            adapter_enabled: true,
            remat: false,
            tape: RefCell::new(None),
        })
    }

    /// Full-sequence logits `[batch, seq, vocab]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails.
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_window(input_ids, None)
    }

    /// Memory-lean forward over a final output window.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence or a tensor op
    /// fails.
    pub fn forward_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        self.forward_window(input_ids, Some((start, len)))
    }

    fn forward_window(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        if self.remat {
            return self.forward_remat(input_ids, window);
        }
        let (mut h, masks) = self.embed_and_masks(input_ids)?;
        for layer in &self.layers {
            h = layer.forward(&h, &masks, &self.rot)?;
        }
        self.norm_head_softcap(&h, window)
    }

    fn embed_and_masks(&self, input_ids: &Tensor) -> CandleResult<(Tensor, Gemma4Masks)> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.flatten_all()?;
        let h = (self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?
            * self.embed_scale)?;
        let masks = masks_at(0, l, self.sliding_window, self.dtype, &self.device)?;
        Ok((h, masks))
    }

    fn norm_head_softcap(
        &self,
        h: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        let h = self.norm.forward(&windowed(h, window)?)?;
        let logits = frozen_linear(&h, &self.embed)?;
        softcap(&logits, self.final_logit_softcap)
    }

    /// Detached full-sequence logits.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails.
    pub fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let (mut h, masks) = self.embed_and_masks(input_ids)?;
        h = h.detach();
        for layer in &self.layers {
            h = layer.forward(&h, &masks, &self.rot)?.detach();
        }
        self.norm_head_softcap(&h, None)
    }

    /// Detached windowed logits.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence or a tensor op
    /// fails.
    pub fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        let (mut h, masks) = self.embed_and_masks(input_ids)?;
        h = h.detach();
        for layer in &self.layers {
            h = layer.forward(&h, &masks, &self.rot)?.detach();
        }
        self.norm_head_softcap(&h, Some((start, len)))
    }

    fn forward_remat(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        let (mut h, masks) = self.embed_and_masks(input_ids)?;
        let mut tape = RematTape::new(self.adapter_enabled);
        for layer in &self.layers {
            let x = tape.capture(&h)?;
            h = layer.forward(&x, &masks, &self.rot)?;
        }
        let x = tape.capture(&h)?;
        *self.tape.borrow_mut() = Some(tape);
        self.norm_head_softcap(&x, window)
    }

    /// Backward through the most recent loss.
    ///
    /// # Errors
    ///
    /// Returns a candle error if activation-checkpoint recompute is inconsistent
    /// or the candle backward pass fails.
    pub fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        if !self.remat {
            return loss.backward();
        }
        let Some(tape) = self.tape.borrow_mut().take() else {
            bail!("Gemma4GradModel::backward: activation checkpointing is on but no tape exists")
        };
        if tape.segments() != self.layers.len() {
            bail!(
                "Gemma4GradModel::backward: tape has {} layer segments for {} layers",
                tape.segments(),
                self.layers.len()
            )
        }
        if tape.adapter_enabled() != self.adapter_enabled {
            bail!("Gemma4GradModel::backward: adapter toggle changed between forward and backward")
        }
        let l = tape.first_boundary_dims().map(|d| d[1]).unwrap_or_default();
        let masks = masks_at(0, l, self.sliding_window, self.dtype, &self.device)?;
        stitched_backward(loss, &tape, &self.trainable_vars(), |i, x| {
            self.layers[i].forward(x, &masks, &self.rot)
        })
    }

    /// Turn layer-boundary activation checkpointing on or off.
    pub fn set_activation_checkpointing(&mut self, on: bool) {
        self.remat = on;
        if !on {
            *self.tape.borrow_mut() = None;
        }
    }

    /// Whether activation checkpointing is currently enabled.
    #[must_use]
    pub fn activation_checkpointing(&self) -> bool {
        self.remat
    }

    /// Enable or disable every `LoRA` adapter.
    pub fn set_adapter_enabled(&mut self, enabled: bool) {
        for layer in &mut self.layers {
            layer.set_adapter_enabled(enabled);
        }
        self.adapter_enabled = enabled;
    }

    /// Trainable `LoRA` variables in deterministic layer-major order.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        let mut out = Vec::new();
        for layer in &self.layers {
            layer.push_vars(&mut out);
        }
        out
    }

    /// Device holding the model tensors.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Snapshot a merged, cached decoder.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any adapter merge fails.
    pub fn merged_decoder(&self) -> CandleResult<Gemma4MergedDecoder> {
        Gemma4MergedDecoder::from_model(self)
    }
}

impl GradModel for Gemma4GradModel {
    type Decoder = Gemma4MergedDecoder;

    fn device(&self) -> &Device {
        Gemma4GradModel::device(self)
    }

    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        Gemma4GradModel::forward(self, input_ids)
    }

    fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        Gemma4GradModel::forward_detached(self, input_ids)
    }

    fn forward_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        Gemma4GradModel::forward_narrowed(self, input_ids, start, len)
    }

    fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        Gemma4GradModel::forward_detached_narrowed(self, input_ids, start, len)
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        Gemma4GradModel::backward(self, loss)
    }

    fn trainable_vars(&self) -> Vec<Var> {
        Gemma4GradModel::trainable_vars(self)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        Gemma4GradModel::set_adapter_enabled(self, enabled);
    }

    fn merged_decoder(&self) -> CandleResult<Gemma4MergedDecoder> {
        Gemma4GradModel::merged_decoder(self)
    }

    fn lora_recipe(&self) -> Option<String> {
        Some(self.targets.canonical())
    }
}

#[derive(Debug)]
struct Gemma4KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
    max_retained: Option<usize>,
    seen_len: usize,
}

impl Gemma4KvCache {
    fn new(dim: usize, max_retained: Option<usize>) -> Self {
        Self {
            k: None,
            v: None,
            dim,
            max_retained,
            seen_len: 0,
        }
    }

    fn current_seq_len(&self) -> usize {
        self.seen_len
    }

    fn retained_seq_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[self.dim])
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<(Tensor, Tensor, usize)> {
        let chunk_len = k.dim(self.dim)?;
        let old_retained = self.retained_seq_len();
        let key_start = self.seen_len.saturating_sub(old_retained);
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn_k = match &self.k {
            Some(prev) => Tensor::cat(&[prev, &k], self.dim)?,
            None => k,
        };
        let attn_v = match &self.v {
            Some(prev) => Tensor::cat(&[prev, &v], self.dim)?,
            None => v,
        };
        self.seen_len += chunk_len;
        let retained = attn_k.dim(self.dim)?;
        let (store_k, store_v) = match self.max_retained {
            Some(max) if retained > max => {
                let start = retained - max;
                (
                    attn_k.narrow(self.dim, start, max)?.contiguous()?,
                    attn_v.narrow(self.dim, start, max)?.contiguous()?,
                )
            }
            _ => (attn_k.clone(), attn_v.clone()),
        };
        self.k = Some(store_k);
        self.v = Some(store_v);
        Ok((attn_k, attn_v, key_start))
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.seen_len = 0;
    }

    fn max_retained_seq_len(&self) -> Option<usize> {
        self.max_retained
    }
}

#[derive(Debug)]
struct Gemma4MergedAttention {
    kind: Gemma4LayerType,
    q_weight: FrozenLinearSnapshot,
    k_weight: FrozenLinearSnapshot,
    v_weight: Option<FrozenLinearSnapshot>,
    o_weight: FrozenLinearSnapshot,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
    cache: Gemma4KvCache,
}

impl Gemma4MergedAttention {
    fn from_attention(a: &Gemma4Attention, sliding_window: usize) -> CandleResult<Self> {
        let max_retained = (a.kind == Gemma4LayerType::SlidingAttention).then_some(sliding_window);
        Ok(Self {
            kind: a.kind,
            q_weight: a.q_proj.snapshot()?,
            k_weight: a.k_proj.snapshot()?,
            v_weight: a.v_proj.as_ref().map(Proj::snapshot).transpose()?,
            o_weight: a.o_proj.snapshot()?,
            q_norm: a.q_norm.clone(),
            k_norm: a.k_norm.clone(),
            v_norm: a.v_norm.clone(),
            num_heads: a.num_heads,
            num_kv_heads: a.num_kv_heads,
            num_kv_groups: a.num_kv_groups,
            head_dim: a.head_dim,
            attn_hidden: a.attn_hidden,
            cache: Gemma4KvCache::new(2, max_retained),
        })
    }

    fn current_seq_len(&self) -> usize {
        self.cache.current_seq_len()
    }

    #[cfg(test)]
    fn retained_seq_len(&self) -> usize {
        self.cache.retained_seq_len()
    }

    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &Gemma4Rotary,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let in_dtype = x.dtype();
        let q = self.q_weight.forward(x)?;
        let k_raw = self.k_weight.forward(x)?;
        let v_raw = match &self.v_weight {
            Some(w) => w.forward(x)?,
            None => k_raw.clone(),
        };
        let q = q.reshape((b, l, self.num_heads, self.head_dim))?;
        let k = k_raw.reshape((b, l, self.num_kv_heads, self.head_dim))?;
        let v = v_raw.reshape((b, l, self.num_kv_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?.transpose(1, 2)?;
        let k = rot.apply(self.kind, &k.transpose(1, 2)?, offset, l)?;
        let q = rot.apply(self.kind, &q, offset, l)?;
        let (k, v, key_start) = self.cache.append(&k, &v)?;
        let sliding_mask = if self.kind == Gemma4LayerType::SlidingAttention {
            Some(attention_mask_for_keys(
                offset,
                l,
                key_start,
                k.dim(2)?,
                self.cache.max_retained,
                in_dtype,
                x.device(),
            )?)
        } else {
            None
        };
        let mask = sliding_mask.as_ref().or(mask);
        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;
        let mut scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(in_dtype)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.attn_hidden))?;
        self.o_weight.forward(&ctx)
    }

    fn reset_cache(&mut self) {
        self.cache.reset();
    }
}

#[derive(Debug)]
struct Gemma4MergedMlp {
    gate_weight: FrozenLinearSnapshot,
    up_weight: FrozenLinearSnapshot,
    down_weight: FrozenLinearSnapshot,
    act: Activation,
}

impl Gemma4MergedMlp {
    fn from_mlp(mlp: &Gemma4Mlp) -> CandleResult<Self> {
        Ok(Self {
            gate_weight: mlp.gate_proj.snapshot()?,
            up_weight: mlp.up_proj.snapshot()?,
            down_weight: mlp.down_proj.snapshot()?,
            act: mlp.act,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let g = self.gate_weight.forward(x)?;
        let u = self.up_weight.forward(x)?;
        let h = self.act.forward(&g)?.broadcast_mul(&u)?;
        self.down_weight.forward(&h)
    }
}

#[derive(Debug)]
struct Gemma4MergedLayer {
    kind: Gemma4LayerType,
    input_layernorm: RmsNorm,
    attn: Gemma4MergedAttention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: Gemma4MergedMlp,
    post_feedforward_layernorm: RmsNorm,
    layer_scalar: Tensor,
}

impl Gemma4MergedLayer {
    fn from_layer(layer: &Gemma4Layer, sliding_window: usize) -> CandleResult<Self> {
        Ok(Self {
            kind: layer.kind,
            input_layernorm: layer.input_layernorm.clone(),
            attn: Gemma4MergedAttention::from_attention(&layer.attn, sliding_window)?,
            post_attention_layernorm: layer.post_attention_layernorm.clone(),
            pre_feedforward_layernorm: layer.pre_feedforward_layernorm.clone(),
            mlp: Gemma4MergedMlp::from_mlp(&layer.mlp)?,
            post_feedforward_layernorm: layer.post_feedforward_layernorm.clone(),
            layer_scalar: layer.layer_scalar.clone(),
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        masks: &Gemma4Masks,
        rot: &Gemma4Rotary,
    ) -> CandleResult<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h, offset, masks.get(self.kind), rot)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let x = x.broadcast_add(&h)?;
        let h2 = self.pre_feedforward_layernorm.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        let h2 = self.post_feedforward_layernorm.forward(&h2)?;
        let x = x.broadcast_add(&h2)?;
        x.broadcast_mul(&self.layer_scalar)
    }

    fn reset_cache(&mut self) {
        self.attn.reset_cache();
    }
}

/// A KV-cached, grad-free dense Gemma 4 text decoder over merged weights.
#[derive(Debug)]
pub struct Gemma4MergedDecoder {
    embed: Tensor,
    layers: Vec<Gemma4MergedLayer>,
    norm: RmsNorm,
    rot: Gemma4Rotary,
    hidden: usize,
    embed_scale: f64,
    final_logit_softcap: f64,
    device: Device,
    dtype: DType,
}

impl Gemma4MergedDecoder {
    fn from_model(model: &Gemma4GradModel) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            layers.push(Gemma4MergedLayer::from_layer(layer, model.sliding_window)?);
        }
        Ok(Self {
            embed: model.embed.clone(),
            layers,
            norm: model.norm.clone(),
            rot: model.rot.clone(),
            hidden: model.hidden,
            embed_scale: model.embed_scale,
            final_logit_softcap: model.final_logit_softcap,
            device: model.device.clone(),
            dtype: model.dtype,
        })
    }

    /// Logits for `input_ids` placed at absolute positions starting at `offset`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `offset` disagrees with the cache length or any
    /// tensor op fails.
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let cached = self
            .layers
            .first()
            .map_or(0, |layer| layer.attn.current_seq_len());
        if offset != cached {
            bail!(
                "Gemma4MergedDecoder::forward: offset {offset} != cached sequence length {cached}"
            );
        }
        let ids = input_ids.flatten_all()?;
        let mut h = (self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?
            * self.embed_scale)?;
        let masks = merged_masks_at(offset, l, self.dtype, &self.device)?;
        for layer in &mut self.layers {
            h = layer.forward(&h, offset, &masks, &self.rot)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = frozen_linear(&h, &self.embed)?;
        softcap(&logits, self.final_logit_softcap)
    }

    /// Reset all per-layer KV caches.
    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }
}

impl CachedDecoder for Gemma4MergedDecoder {
    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        Gemma4MergedDecoder::forward(self, input_ids, offset)
    }

    fn reset_cache(&mut self) {
        Gemma4MergedDecoder::reset_cache(self);
    }

    fn decoder_cache_snapshots(&self, phase: &'static str) -> Vec<DecoderCacheSnapshot> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_index, layer)| DecoderCacheSnapshot {
                phase: phase.to_string(),
                layer_index,
                kind: match layer.kind {
                    Gemma4LayerType::SlidingAttention => "sliding_attention",
                    Gemma4LayerType::FullAttention => "full_attention",
                }
                .to_string(),
                seen_tokens: layer.attn.current_seq_len(),
                retained_tokens: layer.attn.cache.retained_seq_len(),
                max_retained_tokens: layer.attn.cache.max_retained_seq_len(),
            })
            .collect()
    }
}

fn softcap(logits: &Tensor, cap: f64) -> CandleResult<Tensor> {
    (logits / cap)?.tanh()? * cap
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::nn::grad_coverage;
    use crate::tensor_parallel::{
        concat_column_shards, sum_row_parallel_partials, TensorParallelPlan,
    };

    const GEMMA4_31B_TEXT_CONFIG: &str = r#"{
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "vision_config": { "model_type": "gemma4_vision" },
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": true,
            "enable_moe_block": false,
            "expert_intermediate_size": null,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 512,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 5376,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": 21504,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ],
            "max_position_embeddings": 262144,
            "model_type": "gemma4_text",
            "num_attention_heads": 32,
            "num_experts": null,
            "num_global_key_value_heads": 4,
            "num_hidden_layers": 6,
            "num_key_value_heads": 16,
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "sliding_window": 1024,
            "tie_word_embeddings": true,
            "top_k_experts": null,
            "use_bidirectional_attention": "vision",
            "use_cache": true,
            "use_double_wide_mlp": false,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144
        }
    }"#;

    const TINY_GEMMA4_TEXT_CONFIG: &str = r#"{
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": true,
            "enable_moe_block": false,
            "expert_intermediate_size": null,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 8,
            "head_dim": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 8,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": 16,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 32,
            "model_type": "gemma4_text",
            "num_attention_heads": 2,
            "num_experts": null,
            "num_global_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "sliding_window": 3,
            "tie_word_embeddings": true,
            "top_k_experts": null,
            "use_bidirectional_attention": "vision",
            "use_cache": true,
            "use_double_wide_mlp": false,
            "vocab_size": 16,
            "vocab_size_per_layer_input": 16
        }
    }"#;

    fn dev() -> Device {
        Device::Cpu
    }

    fn tiny_cfg() -> Gemma4Config {
        Gemma4Config::from_json_str(TINY_GEMMA4_TEXT_CONFIG).unwrap()
    }

    fn quantized_tiny_cfg() -> Gemma4Config {
        let mut cfg = tiny_cfg();
        let t = &mut cfg.text_config;
        t.vocab_size = 32;
        t.vocab_size_per_layer_input = 32;
        t.hidden_size = 32;
        t.intermediate_size = 64;
        t.num_attention_heads = 2;
        t.num_key_value_heads = 1;
        t.num_global_key_value_heads = 1;
        t.head_dim = 16;
        t.global_head_dim = 16;
        t.max_position_embeddings = 32;
        cfg.validate().unwrap();
        cfg
    }

    fn gemma4_unified_12b_config_json() -> String {
        let layer_types: Vec<&str> = (0..48)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        serde_json::json!({
            "model_type": "gemma4_unified",
            "tie_word_embeddings": true,
            "audio_config": { "model_type": "gemma4_unified_audio" },
            "vision_config": { "model_type": "gemma4_unified_vision" },
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attention_k_eq_v": true,
                "enable_moe_block": false,
                "expert_intermediate_size": null,
                "final_logit_softcapping": 30.0,
                "global_head_dim": 512,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 3840,
                "hidden_size_per_layer_input": 0,
                "intermediate_size": 15360,
                "layer_types": layer_types,
                "max_position_embeddings": 262144,
                "model_type": "gemma4_unified_text",
                "moe_intermediate_size": null,
                "num_attention_heads": 16,
                "num_experts": null,
                "num_global_key_value_heads": 1,
                "num_hidden_layers": 48,
                "num_key_value_heads": 8,
                "num_kv_shared_layers": 0,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                },
                "sliding_window": 1024,
                "tie_word_embeddings": true,
                "top_k_experts": null,
                "use_bidirectional_attention": "vision",
                "use_cache": true,
                "use_double_wide_mlp": false,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144
            }
        })
        .to_string()
    }

    fn put_rand(t: &mut HashMap<String, Tensor>, name: &str, dims: &[usize]) {
        t.insert(
            name.to_string(),
            Tensor::randn(0f32, 0.05f32, dims.to_vec(), &dev()).unwrap(),
        );
    }

    fn put_ones(t: &mut HashMap<String, Tensor>, name: &str, dims: &[usize]) {
        t.insert(
            name.to_string(),
            Tensor::ones(dims.to_vec(), DType::F32, &dev()).unwrap(),
        );
    }

    fn weight_map(cfg: &Gemma4Config) -> HashMap<String, Tensor> {
        let tcfg = &cfg.text_config;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let h = tcfg.hidden_size;
        let i = tcfg.intermediate_size;

        put_rand(
            &mut t,
            &format!("{CKPT_PREFIX}.embed_tokens.weight"),
            &[tcfg.vocab_size, h],
        );
        put_ones(&mut t, &format!("{CKPT_PREFIX}.norm.weight"), &[h]);
        for l in 0..tcfg.num_hidden_layers {
            let p = format!("{CKPT_PREFIX}.layers.{l}");
            let full = tcfg.layer_types[l] == Gemma4LayerType::FullAttention;
            let head_dim = if full {
                tcfg.global_head_dim
            } else {
                tcfg.head_dim
            };
            let kv_heads = if full {
                tcfg.num_global_key_value_heads
            } else {
                tcfg.num_key_value_heads
            };
            let q_out = tcfg.num_attention_heads * head_dim;
            let kv_out = kv_heads * head_dim;

            put_ones(&mut t, &format!("{p}.input_layernorm.weight"), &[h]);
            put_ones(
                &mut t,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            );
            put_ones(
                &mut t,
                &format!("{p}.pre_feedforward_layernorm.weight"),
                &[h],
            );
            put_ones(
                &mut t,
                &format!("{p}.post_feedforward_layernorm.weight"),
                &[h],
            );
            put_ones(&mut t, &format!("{p}.layer_scalar"), &[1]);
            put_rand(&mut t, &format!("{p}.self_attn.q_proj.weight"), &[q_out, h]);
            put_rand(
                &mut t,
                &format!("{p}.self_attn.k_proj.weight"),
                &[kv_out, h],
            );
            if !full {
                put_rand(
                    &mut t,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[kv_out, h],
                );
            }
            put_rand(&mut t, &format!("{p}.self_attn.o_proj.weight"), &[h, q_out]);
            put_ones(&mut t, &format!("{p}.self_attn.q_norm.weight"), &[head_dim]);
            put_ones(&mut t, &format!("{p}.self_attn.k_norm.weight"), &[head_dim]);
            put_rand(&mut t, &format!("{p}.mlp.gate_proj.weight"), &[i, h]);
            put_rand(&mut t, &format!("{p}.mlp.up_proj.weight"), &[i, h]);
            put_rand(&mut t, &format!("{p}.mlp.down_proj.weight"), &[h, i]);
        }
        t
    }

    fn tiny_vb(cfg: &Gemma4Config) -> VarBuilder<'static> {
        VarBuilder::from_tensors(weight_map(cfg), DType::F32, &dev())
    }

    fn tiny_model() -> Gemma4GradModel {
        let cfg = tiny_cfg();
        let vb = tiny_vb(&cfg);
        Gemma4GradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap()
    }

    fn quantized_tiny_model() -> Gemma4GradModel {
        let cfg = quantized_tiny_cfg();
        let vb = tiny_vb(&cfg);
        Gemma4GradModel::load_with_targets_and_base_quantization(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
            BaseQuantization::Q8_0,
        )
        .unwrap()
    }

    fn unique_tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ferrl-gemma4-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn ids(seq: usize) -> Tensor {
        let v: Vec<u32> = (0..seq as u32).map(|i| i % 7).collect();
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

    fn assert_merged_cache_lens(
        dec: &Gemma4MergedDecoder,
        seen: usize,
        sliding_retained: usize,
        full_retained: usize,
    ) {
        assert_eq!(dec.layers[0].attn.current_seq_len(), seen);
        assert_eq!(dec.layers[0].attn.retained_seq_len(), sliding_retained);
        assert_eq!(dec.layers[1].attn.current_seq_len(), seen);
        assert_eq!(dec.layers[1].attn.retained_seq_len(), full_retained);
    }

    #[test]
    fn proportional_rope_rotates_full_head_halves_with_zero_tail() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.text_config.global_head_dim, 8);
        assert_eq!(cfg.text_config.full_attention_rotary_dim(), 4);
        let rot = Gemma4Rotary::new(&cfg.text_config, DType::F32, &dev()).unwrap();
        let x = Tensor::from_vec(
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32],
            (1, 1, 1, 8),
            &dev(),
        )
        .unwrap();
        let y = rot.apply(Gemma4LayerType::FullAttention, &x, 1, 1).unwrap();
        let got: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();

        let theta = cfg.text_config.rope_parameters.full_attention.rope_theta as f32;
        let f0 = 1f32;
        let f1 = 1f32 / theta.powf(2f32 / 8f32);
        let expected = [
            0f32 * f0.cos() - 4f32 * f0.sin(),
            1f32 * f1.cos() - 5f32 * f1.sin(),
            2f32,
            3f32,
            4f32 * f0.cos() + 0f32 * f0.sin(),
            5f32 * f1.cos() + 1f32 * f1.sin(),
            6f32,
            7f32,
        ];
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert!(
                (*g - e).abs() < 1e-5,
                "dim {i}: got {g}, expected {e}, all {got:?}"
            );
        }
    }

    fn arm_adapter(model: &Gemma4GradModel) {
        for (i, v) in model.trainable_vars().iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                v.set(&Tensor::randn(0f32, 0.1f32, dims, &dev()).unwrap())
                    .unwrap();
            }
        }
    }

    fn all_tp_plans(world_size: usize) -> Vec<TensorParallelPlan> {
        (0..world_size)
            .map(|rank| TensorParallelPlan::new(rank, world_size).unwrap())
            .collect()
    }

    #[test]
    fn gemma4_mlp_tensor_parallel_projection_shards_reassemble() {
        let model = tiny_model();
        arm_adapter(&model);
        let mlp = &model.layers[0].mlp;
        let x = Tensor::from_vec(
            (0..16).map(|i| i as f32 * 0.035 - 0.25).collect::<Vec<_>>(),
            (1, 2, model.hidden),
            &dev(),
        )
        .unwrap();

        let full = mlp.forward(&x).unwrap();
        let partials = all_tp_plans(2)
            .into_iter()
            .map(|plan| {
                let gate = mlp
                    .gate_proj
                    .column_parallel_forward(&x, plan, "intermediate_size")?;
                let up = mlp
                    .up_proj
                    .column_parallel_forward(&x, plan, "intermediate_size")?;
                let hidden = mlp.act.forward(&gate)?.broadcast_mul(&up)?;
                mlp.down_proj.row_parallel_forward_partial_from_shard(
                    &hidden,
                    plan,
                    "intermediate_size",
                )
            })
            .collect::<CandleResult<Vec<_>>>()
            .unwrap();
        let sharded = sum_row_parallel_partials(&partials).unwrap();

        assert_eq!(sharded.dims(), full.dims());
        let worst = max_abs_diff(&sharded, &full);
        assert!(
            worst <= 1e-5,
            "tensor-parallel Gemma 4 MLP reassembly diverged: {worst}"
        );
    }

    #[test]
    fn gemma4_attention_tensor_parallel_projection_shards_reassemble() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let vb = tiny_vb(&cfg);
        let model = Gemma4GradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter(&model);
        let attn = &model.layers[0].attn;
        let x = Tensor::from_vec(
            (0..16).map(|i| i as f32 * 0.02 - 0.15).collect::<Vec<_>>(),
            (1, 2, model.hidden),
            &dev(),
        )
        .unwrap();

        for (name, proj) in [
            ("q_proj", &attn.q_proj),
            ("k_proj", &attn.k_proj),
            ("v_proj", attn.v_proj.as_ref().unwrap()),
        ] {
            let full = proj.forward(&x).unwrap();
            let shards = all_tp_plans(2)
                .into_iter()
                .map(|plan| proj.column_parallel_forward(&x, plan, "attention_out"))
                .collect::<CandleResult<Vec<_>>>()
                .unwrap();
            let sharded = concat_column_shards(&shards).unwrap();
            let worst = max_abs_diff(&sharded, &full);
            assert!(
                worst <= 1e-5,
                "{name} column shards diverged from full Gemma 4 projection: {worst}"
            );
        }

        let ctx = Tensor::from_vec(
            (0..16).map(|i| i as f32 * -0.025 + 0.3).collect::<Vec<_>>(),
            (1, 2, attn.attn_hidden),
            &dev(),
        )
        .unwrap();
        let full = attn.o_proj.forward(&ctx).unwrap();
        let partials = all_tp_plans(2)
            .into_iter()
            .map(|plan| {
                attn.o_proj
                    .row_parallel_forward_partial(&ctx, plan, "attention_hidden")
            })
            .collect::<CandleResult<Vec<_>>>()
            .unwrap();
        let sharded = sum_row_parallel_partials(&partials).unwrap();
        let worst = max_abs_diff(&sharded, &full);
        assert!(
            worst <= 1e-5,
            "o_proj row partials diverged from full Gemma 4 projection: {worst}"
        );
    }

    fn overwrite_adapter(model: &Gemma4GradModel) {
        for (i, v) in model.trainable_vars().iter().enumerate() {
            let dims = v.as_tensor().dims().to_vec();
            let scale = if i % 2 == 0 { 1.25 } else { -0.75 };
            let replacement = (Tensor::ones(dims, DType::F32, &dev()).unwrap() * scale).unwrap();
            v.set(&replacement).unwrap();
        }
    }

    #[test]
    fn tiny_forward_produces_full_seq_logits() {
        let cfg = tiny_cfg();
        let model = tiny_model();
        let logits = model.forward(&ids(5)).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.text_config.vocab_size]);
        let sum: f32 = logits
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(sum.is_finite());
    }

    #[test]
    fn industrial_recipe_var_order_omits_full_attention_v_proj() {
        let model = tiny_model();
        assert_eq!(model.lora_recipe().as_deref(), Some("attn:qkvo|mlp:gud"));
        assert_eq!(
            model.trainable_vars().len(),
            14 + 12,
            "sliding layer has q/k/v/o + mlp, full layer has q/k/o + mlp"
        );
    }

    #[test]
    fn industrial_grads_flow_to_every_gemma4_projection() {
        let model = tiny_model();
        arm_adapter(&model);
        let vars = model.trainable_vars();
        let loss = model
            .forward(&ids(5))
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
    fn merged_decoder_matches_uncached_with_armed_adapters() {
        let mut model = tiny_model();
        arm_adapter(&model);
        let input = ids(5);

        model.set_adapter_enabled(false);
        let base = model.forward(&input).unwrap();
        model.set_adapter_enabled(true);
        let reference = model.forward(&input).unwrap();
        assert!(
            max_abs_diff(&reference, &base) > 1e-5,
            "armed Gemma 4 adapter must change the logits"
        );

        let mut prefill = model.merged_decoder().unwrap();
        let prefill_logits = prefill.forward(&input, 0).unwrap();
        assert!(
            max_abs_diff(&prefill_logits, &reference) <= 1e-3,
            "Gemma 4 cached prefill diverged from uncached forward"
        );

        let mut dec = model.merged_decoder().unwrap();
        let mut worst = 0f32;
        for t in 0..5 {
            let tok = input.narrow(1, t, 1).unwrap();
            let logits_t = dec.forward(&tok, t).unwrap();
            worst = worst.max(max_abs_diff(&logits_t, &reference.narrow(1, t, 1).unwrap()));
        }
        assert!(
            worst <= 1e-3,
            "Gemma 4 cached decode diverged from uncached forward: {worst}"
        );
    }

    #[test]
    fn quantized_merged_decoder_matches_uncached_with_armed_adapters() {
        let model = quantized_tiny_model();
        arm_adapter(&model);
        let input = ids(5);
        let reference = model.forward(&input).unwrap();
        let mut dec = model.merged_decoder().unwrap();
        let got = dec.forward(&input, 0).unwrap();
        assert_eq!(got.dims(), reference.dims());
        assert!(
            max_abs_diff(&got, &reference) <= 0.05,
            "quantized Gemma 4 cached prefill diverged from uncached forward"
        );
    }

    #[test]
    fn quantized_merged_decoder_snapshots_adapter_values_across_var_updates() {
        let model = quantized_tiny_model();
        arm_adapter(&model);
        let input = ids(6);
        let before_update = model.forward(&input).unwrap();
        let mut dec = model.merged_decoder().unwrap();

        let prefix_len = 3;
        let prefix = input.narrow(1, 0, prefix_len).unwrap();
        let prefix_logits = dec.forward(&prefix, 0).unwrap();
        assert!(
            max_abs_diff(
                &prefix_logits,
                &before_update.narrow(1, 0, prefix_len).unwrap()
            ) <= 0.05,
            "quantized snapshot must match the pre-update adapter before mutation"
        );

        overwrite_adapter(&model);
        let after_update = model.forward(&input).unwrap();
        assert!(
            max_abs_diff(&after_update, &before_update) > 0.05,
            "adapter overwrite must move the live model enough to make this gate non-vacuous"
        );

        let suffix_len = 2;
        let suffix = input.narrow(1, prefix_len, suffix_len).unwrap();
        let got = dec.forward(&suffix, prefix_len).unwrap();
        let want = before_update.narrow(1, prefix_len, suffix_len).unwrap();
        assert!(
            max_abs_diff(&got, &want) <= 0.05,
            "existing quantized decoder observed adapter vars mutated after construction"
        );
    }

    #[test]
    fn merged_decoder_matches_uncached_multi_token_decode_after_prefill() {
        let model = tiny_model();
        arm_adapter(&model);
        let input = ids(7);
        let reference = model.forward(&input).unwrap();
        let mut dec = model.merged_decoder().unwrap();

        let prefix = input.narrow(1, 0, 4).unwrap();
        let prefix_logits = dec.forward(&prefix, 0).unwrap();
        assert!(
            max_abs_diff(&prefix_logits, &reference.narrow(1, 0, 4).unwrap()) <= 1e-3,
            "Gemma 4 cached prefill diverged before multi-token decode"
        );
        assert_merged_cache_lens(&dec, 4, 3, 4);

        let chunk = input.narrow(1, 4, 2).unwrap();
        let chunk_logits = dec.forward(&chunk, 4).unwrap();
        assert!(
            max_abs_diff(&chunk_logits, &reference.narrow(1, 4, 2).unwrap()) <= 1e-3,
            "Gemma 4 cached multi-token decode diverged after prefill"
        );
        assert_merged_cache_lens(&dec, 6, 3, 6);
    }

    #[test]
    fn merged_decoder_bounds_sliding_cache_without_changing_offsets() {
        let model = tiny_model();
        let mut dec = model.merged_decoder().unwrap();
        let input = ids(5);
        let _ = dec.forward(&input, 0).unwrap();
        assert_merged_cache_lens(&dec, 5, 3, 5);

        let tok = Tensor::from_vec(vec![1u32], (1, 1), &dev()).unwrap();
        let _ = dec.forward(&tok, 5).unwrap();
        assert_merged_cache_lens(&dec, 6, 3, 6);
    }

    #[test]
    fn merged_decoder_reports_seen_and_retained_cache_snapshots() {
        let model = tiny_model();
        let mut dec = model.merged_decoder().unwrap();
        let input = ids(5);
        let _ = dec.forward(&input, 0).unwrap();

        let snapshots = dec.decoder_cache_snapshots("rollout_prefill_end");
        assert_eq!(
            snapshots,
            vec![
                DecoderCacheSnapshot {
                    phase: "rollout_prefill_end".to_string(),
                    layer_index: 0,
                    kind: "sliding_attention".to_string(),
                    seen_tokens: 5,
                    retained_tokens: 3,
                    max_retained_tokens: Some(3),
                },
                DecoderCacheSnapshot {
                    phase: "rollout_prefill_end".to_string(),
                    layer_index: 1,
                    kind: "full_attention".to_string(),
                    seen_tokens: 5,
                    retained_tokens: 5,
                    max_retained_tokens: None,
                },
            ]
        );
    }

    #[test]
    fn tensors_from_pretrained_ignores_non_text_shards_and_tensors() {
        let cfg = tiny_cfg();
        let mut map = weight_map(&cfg);
        map.insert(
            "model.vision.patch_embed.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &dev()).unwrap(),
        );
        let base = unique_tmp("loader");

        let single = base.join("single");
        std::fs::create_dir_all(&single).unwrap();
        candle_core::safetensors::save(&map, single.join("model.safetensors")).unwrap();
        let single_loaded = tensors_from_pretrained(&single, &dev()).unwrap();
        assert!(single_loaded.keys().all(|name| is_gemma4_text_tensor(name)));

        let sharded = base.join("sharded");
        std::fs::create_dir_all(&sharded).unwrap();
        let text_only: HashMap<String, Tensor> = map
            .iter()
            .filter(|(name, _)| is_gemma4_text_tensor(name))
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        candle_core::safetensors::save(&text_only, sharded.join("text.safetensors")).unwrap();
        let mut index_map: HashMap<String, String> = text_only
            .keys()
            .map(|name| (name.clone(), "text.safetensors".to_string()))
            .collect();
        index_map.insert(
            "model.vision.patch_embed.weight".to_string(),
            "missing-vision.safetensors".to_string(),
        );
        let index = serde_json::json!({ "metadata": {}, "weight_map": index_map });
        std::fs::write(
            sharded.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        let sharded_loaded = tensors_from_pretrained(&sharded, &dev()).unwrap();
        assert_eq!(sharded_loaded.len(), text_only.len());
        assert!(sharded_loaded
            .keys()
            .all(|name| is_gemma4_text_tensor(name)));

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn tensors_from_pretrained_rejects_non_text_only_checkpoints() {
        let base = unique_tmp("loader-empty");
        let mut non_text: HashMap<String, Tensor> = HashMap::new();
        non_text.insert(
            "model.vision.patch_embed.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &dev()).unwrap(),
        );

        let single = base.join("single");
        std::fs::create_dir_all(&single).unwrap();
        candle_core::safetensors::save(&non_text, single.join("model.safetensors")).unwrap();
        let err = tensors_from_pretrained(&single, &dev())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no model.language_model.* tensors loaded"));

        let sharded = base.join("sharded");
        std::fs::create_dir_all(&sharded).unwrap();
        candle_core::safetensors::save(&non_text, sharded.join("vision.safetensors")).unwrap();
        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.vision.patch_embed.weight": "vision.safetensors"
            }
        });
        std::fs::write(
            sharded.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        let err = tensors_from_pretrained(&sharded, &dev())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no model.language_model.* tensors listed"));

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn parses_dense_31b_text_config_shape() {
        let cfg = Gemma4Config::from_json_str(GEMMA4_31B_TEXT_CONFIG).unwrap();
        assert_eq!(cfg.model_type.as_deref(), Some("gemma4"));
        assert_eq!(cfg.text_config.num_hidden_layers, 6);
        assert_eq!(cfg.text_config.full_attention_rotary_dim(), 128);
        assert_eq!(
            cfg.text_config.layer_types.last(),
            Some(&Gemma4LayerType::FullAttention)
        );
    }

    #[test]
    fn parses_unified_12b_dense_text_model_types() {
        let json = gemma4_unified_12b_config_json();
        let cfg = Gemma4Config::from_json_str(&json).unwrap();
        assert_eq!(cfg.model_type.as_deref(), Some("gemma4_unified"));
        assert_eq!(
            cfg.text_config.model_type.as_deref(),
            Some("gemma4_unified_text")
        );
    }

    #[test]
    fn parses_unified_12b_dense_text_geometry() {
        let json = gemma4_unified_12b_config_json();
        let cfg = Gemma4Config::from_json_str(&json).unwrap();
        assert_eq!(cfg.text_config.hidden_size, 3840);
        assert_eq!(cfg.text_config.num_hidden_layers, 48);
        assert_eq!(cfg.text_config.sliding_window, 1024);
        assert_eq!(cfg.text_config.full_attention_rotary_dim(), 128);
        assert_eq!(
            cfg.text_config.layer_types.last(),
            Some(&Gemma4LayerType::FullAttention)
        );
    }

    #[test]
    fn rejects_wrong_top_level_model_type() {
        let json = GEMMA4_31B_TEXT_CONFIG.replace("\"gemma4\"", "\"gemma3\"");
        let err = Gemma4Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("model_type"));
        assert!(err.contains("gemma4"));
    }

    #[test]
    fn rejects_wrong_unified_text_model_type() {
        let mut value: serde_json::Value =
            serde_json::from_str(&gemma4_unified_12b_config_json()).unwrap();
        value["text_config"]["model_type"] = serde_json::json!("gemma4_unified_audio");
        let err = Gemma4Config::from_json_str(&value.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("text_config.model_type"));
        assert!(err.contains("gemma4_unified_text"));
    }

    #[test]
    fn rejects_moe_fields_in_initial_native_path() {
        let json = GEMMA4_31B_TEXT_CONFIG
            .replace("\"enable_moe_block\": false", "\"enable_moe_block\": true")
            .replace("\"num_experts\": null", "\"num_experts\": 128");
        let err = Gemma4Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("MoE fields unsupported"));
    }

    #[test]
    fn rejects_unified_per_layer_embeddings() {
        let mut value: serde_json::Value =
            serde_json::from_str(&gemma4_unified_12b_config_json()).unwrap();
        value["text_config"]["hidden_size_per_layer_input"] = serde_json::json!(128);
        let err = Gemma4Config::from_json_str(&value.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("hidden_size_per_layer_input"));
    }

    #[test]
    fn rejects_unified_moe_fields() {
        let mut value: serde_json::Value =
            serde_json::from_str(&gemma4_unified_12b_config_json()).unwrap();
        value["text_config"]["enable_moe_block"] = serde_json::json!(true);
        value["text_config"]["num_experts"] = serde_json::json!(128);
        value["text_config"]["top_k_experts"] = serde_json::json!(8);
        let err = Gemma4Config::from_json_str(&value.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("MoE fields unsupported"));
    }

    #[test]
    fn rejects_unified_moe_intermediate_size() {
        let mut value: serde_json::Value =
            serde_json::from_str(&gemma4_unified_12b_config_json()).unwrap();
        value["text_config"]["moe_intermediate_size"] = serde_json::json!(15360);
        let err = Gemma4Config::from_json_str(&value.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("MoE fields unsupported"));
    }

    #[test]
    fn rejects_non_global_final_layer() {
        let json = GEMMA4_31B_TEXT_CONFIG.replace(
            "\"sliding_attention\",\n                \"sliding_attention\",\n                \"sliding_attention\",\n                \"sliding_attention\",\n                \"sliding_attention\",\n                \"full_attention\"",
            "\"full_attention\",\n                \"sliding_attention\",\n                \"sliding_attention\",\n                \"sliding_attention\",\n                \"sliding_attention\",\n                \"sliding_attention\"",
        );
        let err = Gemma4Config::from_json_str(&json).unwrap_err().to_string();
        assert!(err.contains("final layer"));
    }
}
