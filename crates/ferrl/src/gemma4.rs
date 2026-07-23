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
use candle_nn::ops::softmax;
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{Activation, Module, VarBuilder};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::blocks::{frozen_linear, repeat_kv, rope_partial, windowed, RotaryTables};
use crate::comm::Comm;
use crate::lora::{
    BaseQuantization, DenseLoraTargets, FrozenLinearSnapshot, Proj, ProjLoadOptions,
};
use crate::model::{CachedDecoder, GradModel};
use crate::nn::RmsNorm;
use crate::remat::{
    finish_stitched_backward_with_cotangent, prepare_stitched_backward, stitched_backward,
    RematTape,
};
use crate::telemetry::DecoderCacheSnapshot;
use crate::tensor_parallel::{
    all_reduce_sum_straight_through, all_reduce_sum_value, comm_to_candle,
    coordinate_local_candle_call, is_comm_failure, plan_from_comm, ShardRange, TensorParallelPlan,
};

#[cfg(test)]
thread_local! {
    static ADAPTER_VALIDATION_STAGE_FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static TENSOR_PARALLEL_LOCAL_STAGE_FAULT:
        std::cell::RefCell<Option<(&'static str, bool)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn inject_adapter_validation_stage_failure_once(stage: u8) {
    ADAPTER_VALIDATION_STAGE_FAULT.with(|fault| fault.set(stage));
}

#[cfg(test)]
fn take_adapter_validation_stage_fault(stage: u8) -> bool {
    ADAPTER_VALIDATION_STAGE_FAULT.with(|fault| {
        if fault.get() == stage {
            fault.set(0);
            true
        } else {
            false
        }
    })
}

#[cfg(test)]
fn inject_tensor_parallel_local_stage_failure_once(stage: &'static str, panic: bool) {
    TENSOR_PARALLEL_LOCAL_STAGE_FAULT.with(|fault| fault.replace(Some((stage, panic))));
}

#[cfg(test)]
fn tensor_parallel_local_stage_fault_consumed() -> bool {
    TENSOR_PARALLEL_LOCAL_STAGE_FAULT.with(|fault| fault.borrow().is_none())
}

#[cfg(test)]
fn maybe_inject_tensor_parallel_local_stage_failure(stage: &str) -> CandleResult<()> {
    let behavior = TENSOR_PARALLEL_LOCAL_STAGE_FAULT.with(|fault| {
        let mut fault = fault.borrow_mut();
        if fault.as_ref().is_some_and(|(target, _)| *target == stage) {
            fault.take().map(|(_, panic)| panic)
        } else {
            None
        }
    });
    match behavior {
        Some(true) => panic!("injected {stage} panic"),
        Some(false) => bail!("injected {stage} failure"),
        None => Ok(()),
    }
}

/// The checkpoint prefix used by public Gemma 4 conditional-generation
/// checkpoints for the text decoder.
pub const CKPT_PREFIX: &str = "model.language_model";

fn tp_shard(
    plan: TensorParallelPlan,
    label: &'static str,
    axis_len: usize,
) -> CandleResult<ShardRange> {
    plan.shard_axis(label, axis_len)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}

fn reduce_row_parallel_output(
    partial: &Tensor,
    plan: TensorParallelPlan,
    comm: Option<&dyn Comm>,
) -> CandleResult<Tensor> {
    match (plan.is_sharded(), comm) {
        (false, _) => Ok(partial.clone()),
        (true, Some(comm)) => all_reduce_sum_straight_through(partial, plan, comm),
        (true, None) => candle_core::bail!(
            "tensor_parallel row-parallel output for world_size {} needs a communicator",
            plan.world_size()
        ),
    }
}

fn coordinate_tensor_parallel_local_call<T>(
    plan: TensorParallelPlan,
    comm: Option<&dyn Comm>,
    label: &str,
    call: impl FnOnce() -> CandleResult<T>,
) -> CandleResult<T> {
    match comm {
        Some(comm) => coordinate_local_candle_call(comm, label, || {
            #[cfg(test)]
            maybe_inject_tensor_parallel_local_stage_failure(label)?;
            call()
        }),
        None if plan.is_sharded() => candle_core::bail!(
            "{label} for tensor-parallel world size {} needs a communicator",
            plan.world_size()
        ),
        None => call(),
    }
}

fn validate_activation_checkpointing_consensus(
    remat: bool,
    plan: TensorParallelPlan,
    comm: &dyn Comm,
) -> CandleResult<()> {
    if !plan.is_sharded() {
        return Ok(());
    }
    let checkpointed_count =
        comm_to_candle(comm.all_reduce_scalar_sum(if remat { 1.0 } else { 0.0 }))?;
    if checkpointed_count != 0.0 && checkpointed_count != plan.world_size() as f64 {
        bail!(
            "Gemma4 tensor-parallel execution: activation-checkpointing state differs across \
             tensor-parallel ranks"
        )
    }
    Ok(())
}

fn validate_replicated_adapter_values(
    vars: &[Var],
    recipe: &str,
    device: &Device,
    plan: TensorParallelPlan,
    comm: &dyn Comm,
) -> CandleResult<()> {
    if !plan.is_sharded() {
        return Ok(());
    }
    let (mut summed, local_validity) = coordinate_local_candle_call(
        comm,
        "Gemma4 tensor-parallel adapter payload staging",
        || {
            let summed = vars
                .iter()
                .map(|var| var.as_tensor().clone())
                .collect::<Vec<_>>();
            let local_validity = comm.validate_all_reduce_sum(&summed);
            Ok((summed, local_validity))
        },
    )?;
    let invalid_global = comm_to_candle(comm.all_reduce_scalar_sum(if local_validity.is_err() {
        1.0
    } else {
        0.0
    }))?;
    if invalid_global > 0.0 {
        return match local_validity {
            Err(error) => Err(candle_core::Error::Msg(format!(
                "Gemma4GradModel::backward_tensor_parallel: local adapter payload is invalid: \
                 {error}; adapter values were not reduced"
            ))),
            Ok(()) => candle_core::bail!(
                "Gemma4GradModel::backward_tensor_parallel: adapter payload validation failed \
                 on a peer tensor-parallel rank; adapter values were not reduced"
            ),
        };
    }
    let world = plan.world_size() as f64;
    let (digest, mut reduced_manifest) = coordinate_local_candle_call(
        comm,
        "Gemma4 tensor-parallel adapter manifest staging",
        || {
            #[cfg(test)]
            if take_adapter_validation_stage_fault(1) {
                bail!("injected adapter manifest staging failure")
            }
            let digest = adapter_manifest_digest(vars, recipe);
            let manifest = Tensor::from_vec(digest.map(f64::from).to_vec(), digest.len(), device)?;
            Ok((digest, vec![manifest]))
        },
    )?;
    comm_to_candle(comm.all_reduce_sum(&mut reduced_manifest))?;
    let metadata_mismatch_local = coordinate_local_candle_call(
        comm,
        "Gemma4 tensor-parallel adapter manifest readback",
        || {
            #[cfg(test)]
            if take_adapter_validation_stage_fault(2) {
                bail!("injected adapter manifest readback failure")
            }
            let sums = reduced_manifest[0].to_vec1::<f64>()?;
            Ok(digest
                .iter()
                .zip(&sums)
                .any(|(byte, sum)| *sum != world * f64::from(*byte)))
        },
    )?;
    let metadata_mismatch_global =
        comm_to_candle(comm.all_reduce_scalar_sum(if metadata_mismatch_local {
            1.0
        } else {
            0.0
        }))?;
    if metadata_mismatch_global > 0.0 {
        bail!(
            "Gemma4GradModel::backward_tensor_parallel: adapter recipe, tensor count, shapes, or \
             dtypes differ across tensor-parallel ranks; adapter values were not reduced"
        )
    }
    if vars.is_empty() {
        return Ok(());
    }
    comm_to_candle(comm.all_reduce_sum(&mut summed))?;
    coordinate_local_candle_call(
        comm,
        "Gemma4 tensor-parallel adapter value readback",
        || {
            for (index, (var, sum)) in vars.iter().zip(&summed).enumerate() {
                let expected = (var.as_tensor() * world)?;
                let worst = sum
                    .broadcast_sub(&expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                let scale = expected
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?
                    .max(1.0);
                let tolerance = 64.0 * f32::EPSILON * plan.world_size() as f32 * scale;
                if !worst.is_finite() || worst > tolerance {
                    bail!(
                        "Gemma4GradModel::backward_tensor_parallel: replicated adapter tensor {index} \
                         differs across tensor-parallel ranks (max reduction residual {worst}, \
                         tolerance {tolerance})"
                    )
                }
            }
            Ok(())
        },
    )
}

fn adapter_manifest_digest(vars: &[Var], recipe: &str) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"ferrl-gemma4-adapter-manifest-v1\0");
    digest.update(recipe.len().to_le_bytes());
    digest.update(recipe.as_bytes());
    digest.update(vars.len().to_le_bytes());
    for var in vars {
        let tensor = var.as_tensor();
        let dtype = format!("{:?}", tensor.dtype());
        digest.update(dtype.len().to_le_bytes());
        digest.update(dtype.as_bytes());
        digest.update(tensor.dims().len().to_le_bytes());
        for dim in tensor.dims() {
            digest.update(dim.to_le_bytes());
        }
    }
    digest.finalize().into()
}

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
        weight_plan: TensorParallelPlan,
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
        let load_column = |name: &str, shape: (usize, usize), adapted: bool| {
            if weight_plan.is_sharded() {
                Proj::load_column_parallel_with_options(
                    vb,
                    name,
                    shape,
                    adapted,
                    proj_opts,
                    weight_plan,
                )
            } else {
                Proj::load_with_options(vb, name, shape, adapted, proj_opts)
            }
        };
        let v_proj = if full {
            None
        } else {
            Some(load_column("v_proj", (kv_out, h), targets.attn_v)?)
        };
        let o_proj = if weight_plan.is_sharded() {
            Proj::load_row_parallel_with_options(
                vb,
                "o_proj",
                (h, q_out),
                targets.attn_o,
                proj_opts,
                weight_plan,
            )?
        } else {
            Proj::load_with_options(vb, "o_proj", (h, q_out), targets.attn_o, proj_opts)?
        };
        Ok(Self {
            kind,
            q_proj: load_column("q_proj", (q_out, h), targets.attn_q)?,
            k_proj: load_column("k_proj", (kv_out, h), targets.attn_k)?,
            v_proj,
            o_proj,
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

    fn forward_at_tensor_parallel(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &Gemma4Rotary,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
    ) -> CandleResult<Tensor> {
        let partial = coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "Gemma4 attention payload staging",
            || {
                let (b, l, _) = x.dims3()?;
                let in_dtype = x.dtype();
                let offset = 0;
                let heads = tp_shard(plan, "num_attention_heads", self.num_heads)?.len;
                let kv_heads = tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?.len;
                let attn_hidden = heads * self.head_dim;
                let q = self
                    .q_proj
                    .column_parallel_forward(x, plan, "attention_q_out")?;
                let k_raw = self
                    .k_proj
                    .column_parallel_forward(x, plan, "attention_k_out")?;
                let v_raw = match &self.v_proj {
                    Some(v) => v.column_parallel_forward(x, plan, "attention_v_out")?,
                    None => k_raw.clone(),
                };

                let q = q.reshape((b, l, heads, self.head_dim))?;
                let k = k_raw.reshape((b, l, kv_heads, self.head_dim))?;
                let v = v_raw.reshape((b, l, kv_heads, self.head_dim))?;

                let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
                let k = self.k_norm.forward(&k)?;
                let v = self.v_norm.forward(&v)?.transpose(1, 2)?;
                let k = rot.apply(self.kind, &k.transpose(1, 2)?, offset, l)?;
                let q = rot.apply(self.kind, &q, offset, l)?;

                let v = v.contiguous()?;
                let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
                let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

                let mut scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
                if let Some(m) = mask {
                    scores = scores.broadcast_add(m)?;
                }
                let probs =
                    softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(in_dtype)?;
                let ctx = probs.matmul(&v)?;
                let ctx = ctx
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((b, l, attn_hidden))?;
                self.o_proj
                    .row_parallel_forward_partial_from_shard(&ctx, plan, "attention_hidden")
            },
        )?;
        reduce_row_parallel_output(&partial, plan, comm)
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.q_proj
            .validate_column_parallel_support(plan, "attention_q_out")?;
        self.k_proj
            .validate_column_parallel_support(plan, "attention_k_out")?;
        if let Some(projection) = &self.v_proj {
            projection.validate_column_parallel_support(plan, "attention_v_out")?;
        }
        self.o_proj
            .validate_row_parallel_support(plan, "attention_hidden")?;
        tp_shard(plan, "num_attention_heads", self.num_heads)?;
        tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?;
        Ok(())
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
        weight_plan: TensorParallelPlan,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let load_column = |name: &str, adapted: bool| {
            if weight_plan.is_sharded() {
                Proj::load_column_parallel_with_options(
                    vb,
                    name,
                    (i, h),
                    adapted,
                    proj_opts,
                    weight_plan,
                )
            } else {
                Proj::load_with_options(vb, name, (i, h), adapted, proj_opts)
            }
        };
        let down_proj = if weight_plan.is_sharded() {
            Proj::load_row_parallel_with_options(
                vb,
                "down_proj",
                (h, i),
                targets.mlp_down,
                proj_opts,
                weight_plan,
            )?
        } else {
            Proj::load_with_options(vb, "down_proj", (h, i), targets.mlp_down, proj_opts)?
        };
        Ok(Self {
            gate_proj: load_column("gate_proj", targets.mlp_gate)?,
            up_proj: load_column("up_proj", targets.mlp_up)?,
            down_proj,
            act: Activation::GeluPytorchTanh,
        })
    }

    fn forward_tensor_parallel(
        &self,
        x: &Tensor,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
    ) -> CandleResult<Tensor> {
        let partial = coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "Gemma4 MLP payload staging",
            || {
                let g = self
                    .gate_proj
                    .column_parallel_forward(x, plan, "intermediate_size")?;
                let u = self
                    .up_proj
                    .column_parallel_forward(x, plan, "intermediate_size")?;
                let h = self.act.forward(&g)?.broadcast_mul(&u)?;
                self.down_proj.row_parallel_forward_partial_from_shard(
                    &h,
                    plan,
                    "intermediate_size",
                )
            },
        )?;
        reduce_row_parallel_output(&partial, plan, comm)
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.gate_proj
            .validate_column_parallel_support(plan, "intermediate_size")?;
        self.up_proj
            .validate_column_parallel_support(plan, "intermediate_size")?;
        self.down_proj
            .validate_row_parallel_support(plan, "intermediate_size")
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
        weight_plan: TensorParallelPlan,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        Ok(Self {
            kind: cfg.layer_types[layer_idx],
            input_layernorm: RmsNorm::new(vb.pp("input_layernorm").get(h, "weight")?, eps),
            attn: Gemma4Attention::load(
                cfg,
                layer_idx,
                &vb.pp("self_attn"),
                targets,
                proj_opts,
                weight_plan,
            )?,
            post_attention_layernorm: RmsNorm::new(
                vb.pp("post_attention_layernorm").get(h, "weight")?,
                eps,
            ),
            pre_feedforward_layernorm: RmsNorm::new(
                vb.pp("pre_feedforward_layernorm").get(h, "weight")?,
                eps,
            ),
            mlp: Gemma4Mlp::load(cfg, &vb.pp("mlp"), targets, proj_opts, weight_plan)?,
            post_feedforward_layernorm: RmsNorm::new(
                vb.pp("post_feedforward_layernorm").get(h, "weight")?,
                eps,
            ),
            layer_scalar: vb.get(1, "layer_scalar")?,
        })
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.attn.validate_tensor_parallel_plan(plan)?;
        self.mlp.validate_tensor_parallel_plan(plan)
    }

    fn forward(&self, x: &Tensor, masks: &Gemma4Masks, rot: &Gemma4Rotary) -> CandleResult<Tensor> {
        self.forward_tensor_parallel(x, masks, rot, TensorParallelPlan::single(), None)
    }

    fn forward_tensor_parallel(
        &self,
        x: &Tensor,
        masks: &Gemma4Masks,
        rot: &Gemma4Rotary,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
    ) -> CandleResult<Tensor> {
        let x = self.forward_attention_tensor_parallel(x, masks, rot, plan, comm, false)?;
        self.forward_mlp_tensor_parallel(&x, plan, comm, false)
    }

    fn reduced_backward_residual(x: &Tensor, plan: TensorParallelPlan) -> CandleResult<Tensor> {
        if !plan.is_sharded() {
            return Ok(x.clone());
        }
        let scaled = (x * (1.0 / plan.world_size() as f64))?;
        let correction = x.broadcast_sub(&scaled)?.detach();
        scaled.broadcast_add(&correction)
    }

    fn forward_attention_tensor_parallel(
        &self,
        x: &Tensor,
        masks: &Gemma4Masks,
        rot: &Gemma4Rotary,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
        reduce_boundary_backward: bool,
    ) -> CandleResult<Tensor> {
        let (residual, h) = coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "Gemma4 attention boundary preparation",
            || {
                let residual = if reduce_boundary_backward {
                    Self::reduced_backward_residual(x, plan)?
                } else {
                    x.clone()
                };
                Ok((residual, self.input_layernorm.forward(x)?))
            },
        )?;
        let h = self
            .attn
            .forward_at_tensor_parallel(&h, masks.get(self.kind), rot, plan, comm)?;
        coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "Gemma4 attention boundary completion",
            || {
                let h = self.post_attention_layernorm.forward(&h)?;
                residual.broadcast_add(&h)
            },
        )
    }

    fn forward_mlp_tensor_parallel(
        &self,
        x: &Tensor,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
        reduce_boundary_backward: bool,
    ) -> CandleResult<Tensor> {
        let (residual, h2) = coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "Gemma4 MLP boundary preparation",
            || {
                let residual = if reduce_boundary_backward {
                    Self::reduced_backward_residual(x, plan)?
                } else {
                    x.clone()
                };
                Ok((residual, self.pre_feedforward_layernorm.forward(x)?))
            },
        )?;
        let h2 = self.mlp.forward_tensor_parallel(&h2, plan, comm)?;
        coordinate_tensor_parallel_local_call(plan, comm, "Gemma4 MLP boundary completion", || {
            let h2 = self.post_feedforward_layernorm.forward(&h2)?;
            let x = residual.broadcast_add(&h2)?;
            x.broadcast_mul(&self.layer_scalar)
        })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4RematExecution {
    Ordinary,
    TensorParallel(TensorParallelPlan),
}

#[derive(Debug)]
struct Gemma4RematTape {
    tape: RematTape,
    execution: Gemma4RematExecution,
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
    base_quantization: BaseQuantization,
    weight_plan: TensorParallelPlan,
    targets: DenseLoraTargets,
    adapter_enabled: bool,
    remat: bool,
    tape: RefCell<Option<Gemma4RematTape>>,
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
        Self::load_with_targets_base_quantization_and_tensor_parallel(
            cfg,
            vb,
            rank,
            alpha,
            adapter_dtype,
            targets,
            base_quantization,
            TensorParallelPlan::single(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_with_targets_base_quantization_and_tensor_parallel(
        cfg: &Gemma4Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        targets: DenseLoraTargets,
        base_quantization: BaseQuantization,
        weight_plan: TensorParallelPlan,
    ) -> CandleResult<Self> {
        cfg.validate()?;
        if !targets.any() {
            bail!("Gemma4GradModel: DenseLoraTargets selects no projection");
        }
        if weight_plan.is_sharded() && base_quantization != BaseQuantization::None {
            bail!(
                "Gemma4GradModel rank-local tensor_parallel loading does not support {} base \
                 quantization",
                base_quantization.as_str()
            );
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
                weight_plan,
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
            base_quantization,
            weight_plan,
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
    /// Returns a candle error if the model stores rank-local tensor-parallel
    /// weights (use [`forward_tensor_parallel`](Self::forward_tensor_parallel)
    /// instead) or if any tensor op fails.
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_window(input_ids, None)
    }

    /// Memory-lean forward over a final output window.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the model stores rank-local tensor-parallel
    /// weights, the window exceeds the sequence, or a tensor op fails.
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
        self.validate_ordinary_execution_support()?;
        if self.remat {
            return self.forward_remat(input_ids, window);
        }
        let (mut h, masks) = self.embed_and_masks(input_ids)?;
        for layer in &self.layers {
            h = layer.forward(&h, &masks, &self.rot)?;
        }
        self.norm_head_softcap(&h, window)
    }

    /// Full-sequence logits through the tensor-parallel projection/collective
    /// path, using `comm`'s rank/world as the plan.
    ///
    /// Public `ferrl train` uses this path with either replicated weights or
    /// persistent rank-local projection weights loaded for the matching plan.
    ///
    /// # Errors
    ///
    /// Returns a candle error if rank/world validation fails, `comm` does not
    /// match the stored rank-local weight plan, or a collective/tensor op fails.
    pub fn forward_tensor_parallel(
        &self,
        input_ids: &Tensor,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        self.forward_tensor_parallel_window(input_ids, None, comm)
    }

    /// Windowed logits through the tensor-parallel projection/collective path.
    ///
    /// # Errors
    ///
    /// As [`forward_tensor_parallel`](Self::forward_tensor_parallel), plus any
    /// windowing error.
    pub fn forward_tensor_parallel_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        self.forward_tensor_parallel_window(input_ids, Some((start, len)), comm)
    }

    fn forward_tensor_parallel_window(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let (plan, remat) =
            coordinate_local_candle_call(comm, "Gemma4 tensor-parallel forward preflight", || {
                let plan = plan_from_comm(comm)?;
                self.validate_tensor_parallel_execution_support(plan)?;
                Ok((plan, self.remat))
            })?;
        validate_activation_checkpointing_consensus(remat, plan, comm)?;
        if remat {
            return self.forward_tensor_parallel_remat(input_ids, window, plan, comm);
        }
        let (mut h, masks) = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 tensor-parallel forward input staging",
            || self.embed_and_masks(input_ids),
        )?;
        for layer in &self.layers {
            h = layer.forward_tensor_parallel(&h, &masks, &self.rot, plan, Some(comm))?;
        }
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 tensor-parallel forward output staging",
            || self.norm_head_softcap(&h, window),
        )
    }

    fn validate_tensor_parallel_execution_support(
        &self,
        execution_plan: TensorParallelPlan,
    ) -> CandleResult<()> {
        if self.base_quantization == BaseQuantization::Q8_0 {
            bail!(
                "Gemma4GradModel tensor_parallel execution does not support q8_0 base \
                 projections; disable tensor_parallel for world-one Q8_0 until rank-local \
                 quantized shards are implemented"
            );
        }
        if self.weight_plan.is_sharded() && self.weight_plan != execution_plan {
            bail!(
                "Gemma4GradModel rank-local weight plan rank {}/world {} does not match \
                 execution plan rank {}/world {}",
                self.weight_plan.rank(),
                self.weight_plan.world_size(),
                execution_plan.rank(),
                execution_plan.world_size()
            );
        }
        Ok(())
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        for layer in &self.layers {
            layer.validate_tensor_parallel_plan(plan)?;
        }
        Ok(())
    }

    fn validate_ordinary_execution_support(&self) -> CandleResult<()> {
        if self.weight_plan.is_sharded() {
            bail!(
                "ordinary Gemma4GradModel forward cannot use rank-local tensor_parallel base \
                 weights; call a tensor_parallel forward method with the matching communicator"
            );
        }
        Ok(())
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
    /// Returns a candle error if the model stores rank-local tensor-parallel
    /// weights (use
    /// [`forward_tensor_parallel_detached`](Self::forward_tensor_parallel_detached)
    /// instead) or if any tensor op fails.
    pub fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.validate_ordinary_execution_support()?;
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
    /// Returns a candle error if the model stores rank-local tensor-parallel
    /// weights, the window exceeds the sequence, or a tensor op fails.
    pub fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        self.validate_ordinary_execution_support()?;
        let (mut h, masks) = self.embed_and_masks(input_ids)?;
        h = h.detach();
        for layer in &self.layers {
            h = layer.forward(&h, &masks, &self.rot)?.detach();
        }
        self.norm_head_softcap(&h, Some((start, len)))
    }

    /// Detached full-sequence logits through the tensor-parallel
    /// projection/collective path.
    ///
    /// # Errors
    ///
    /// As [`forward_tensor_parallel`](Self::forward_tensor_parallel).
    pub fn forward_tensor_parallel_detached(
        &self,
        input_ids: &Tensor,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        self.forward_tensor_parallel_detached_window(input_ids, None, comm)
    }

    /// Detached windowed logits through the tensor-parallel path.
    ///
    /// # Errors
    ///
    /// As [`forward_tensor_parallel`](Self::forward_tensor_parallel), plus any
    /// windowing error.
    pub fn forward_tensor_parallel_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        self.forward_tensor_parallel_detached_window(input_ids, Some((start, len)), comm)
    }

    fn forward_tensor_parallel_detached_window(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let (plan, mut h, masks) = coordinate_local_candle_call(
            comm,
            "Gemma4 detached tensor-parallel input staging",
            || {
                let plan = plan_from_comm(comm)?;
                self.validate_tensor_parallel_execution_support(plan)?;
                let (h, masks) = self.embed_and_masks(input_ids)?;
                Ok((plan, h.detach(), masks))
            },
        )?;
        for layer in &self.layers {
            let next = layer.forward_tensor_parallel(&h, &masks, &self.rot, plan, Some(comm))?;
            h = coordinate_tensor_parallel_local_call(
                plan,
                Some(comm),
                "Gemma4 detached tensor-parallel layer completion",
                || Ok(next.detach()),
            )?;
        }
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 detached tensor-parallel output staging",
            || Ok(self.norm_head_softcap(&h, window)?.detach()),
        )
    }

    fn forward_remat(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        self.tape.borrow_mut().take();
        let (mut h, masks) = self.embed_and_masks(input_ids)?;
        let mut tape = RematTape::new(self.adapter_enabled);
        for layer in &self.layers {
            let x = tape.capture(&h)?;
            h = layer.forward(&x, &masks, &self.rot)?;
        }
        let x = tape.capture(&h)?;
        let logits = self.norm_head_softcap(&x, window)?;
        *self.tape.borrow_mut() = Some(Gemma4RematTape {
            tape,
            execution: Gemma4RematExecution::Ordinary,
        });
        Ok(logits)
    }

    fn forward_tensor_parallel_remat(
        &self,
        input_ids: &Tensor,
        window: Option<(usize, usize)>,
        plan: TensorParallelPlan,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let (mut h, masks, mut tape) = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 rematerialized tensor-parallel input staging",
            || {
                self.tape.borrow_mut().take();
                let (h, masks) = self.embed_and_masks(input_ids)?;
                let tape = RematTape::new(self.adapter_enabled);
                Ok((h, masks, tape))
            },
        )?;
        for layer in &self.layers {
            let x = coordinate_tensor_parallel_local_call(
                plan,
                Some(comm),
                "Gemma4 rematerialized attention capture",
                || tape.capture(&h),
            )?;
            h = layer.forward_attention_tensor_parallel(
                &x,
                &masks,
                &self.rot,
                plan,
                Some(comm),
                true,
            )?;
            let x = coordinate_tensor_parallel_local_call(
                plan,
                Some(comm),
                "Gemma4 rematerialized MLP capture",
                || tape.capture(&h),
            )?;
            h = layer.forward_mlp_tensor_parallel(&x, plan, Some(comm), true)?;
        }
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 rematerialized tensor-parallel output staging",
            || {
                let x = tape.capture(&h)?;
                let logits = self.norm_head_softcap(&x, window)?;
                *self.tape.borrow_mut() = Some(Gemma4RematTape {
                    tape,
                    execution: Gemma4RematExecution::TensorParallel(plan),
                });
                Ok(logits)
            },
        )
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
        let Some(captured) = self.tape.borrow_mut().take() else {
            bail!("Gemma4GradModel::backward: activation checkpointing is on but no tape exists")
        };
        if captured.execution != Gemma4RematExecution::Ordinary {
            bail!(
                "Gemma4GradModel::backward: the pending activation-checkpoint tape was captured \
                 by tensor-parallel execution; call backward_tensor_parallel with the matching \
                 communicator"
            )
        }
        let tape = captured.tape;
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

    /// Backward through the most recent tensor-parallel loss, replaying each
    /// checkpointed layer with the same communicator when rematerialization is
    /// enabled.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the live communicator does not match the
    /// stored rank-local plan or captured tape, the tape/loss pairing is stale,
    /// a collective fails, or candle backward fails.
    pub fn backward_tensor_parallel(
        &self,
        loss: &Tensor,
        comm: &dyn Comm,
    ) -> CandleResult<GradStore> {
        let plan = coordinate_local_candle_call(
            comm,
            "Gemma4 tensor-parallel backward preflight",
            || plan_from_comm(comm),
        )?;
        validate_activation_checkpointing_consensus(self.remat, plan, comm)?;
        if !self.remat {
            self.validate_tensor_parallel_execution_support(plan)?;
            if plan.is_sharded() {
                bail!(
                    "Gemma4GradModel sharded tensor-parallel backward requires activation \
                     checkpointing so replicated-boundary cotangents can be reduced correctly"
                )
            }
            return loss.backward();
        }
        let adapters_enabled = if plan.is_sharded() {
            comm_to_candle(comm.all_reduce_scalar_sum(if self.adapter_enabled {
                1.0
            } else {
                0.0
            }))?
        } else if self.adapter_enabled {
            1.0
        } else {
            0.0
        };
        let adapters_match =
            adapters_enabled == 0.0 || adapters_enabled == plan.world_size() as f64;
        let (trainable, adapter_recipe) = coordinate_local_candle_call(
            comm,
            "Gemma4 tensor-parallel backward adapter staging",
            || Ok((self.trainable_vars(), self.targets.canonical())),
        )?;
        let adapter_values_match = match validate_replicated_adapter_values(
            &trainable,
            &adapter_recipe,
            &self.device,
            plan,
            comm,
        ) {
            Err(error) if is_comm_failure(&error) => return Err(error),
            result => result,
        };
        let (tape, masks, prepared) = coordinate_local_candle_call(
            comm,
            "Gemma4 tensor-parallel backward readiness",
            || {
                #[cfg(test)]
                maybe_inject_tensor_parallel_local_stage_failure(
                    "Gemma4 tensor-parallel backward readiness",
                )?;
                self.validate_tensor_parallel_execution_support(plan)?;
                let Some(captured) = self.tape.borrow_mut().take() else {
                    bail!(
                        "Gemma4GradModel::backward_tensor_parallel: activation checkpointing is \
                         on but no tape exists"
                    )
                };
                if !adapters_match {
                    bail!(
                        "Gemma4GradModel::backward_tensor_parallel: adapter enabled state differs \
                         across tensor-parallel ranks"
                    )
                }
                adapter_values_match?;
                if captured.execution != Gemma4RematExecution::TensorParallel(plan) {
                    bail!(
                        "Gemma4GradModel::backward_tensor_parallel: pending tape execution {:?} \
                         does not match live tensor-parallel rank {}/world {}",
                        captured.execution,
                        plan.rank(),
                        plan.world_size()
                    )
                }
                let tape = captured.tape;
                let expected_segments = self.layers.len() * 2;
                if tape.segments() != expected_segments {
                    bail!(
                        "Gemma4GradModel::backward_tensor_parallel: tape has {} sublayer \
                         segments, expected {} for {} layers",
                        tape.segments(),
                        expected_segments,
                        self.layers.len()
                    )
                }
                if tape.adapter_enabled() != self.adapter_enabled {
                    bail!(
                        "Gemma4GradModel::backward_tensor_parallel: adapter toggle changed \
                         between forward and backward"
                    )
                }
                let l = tape.first_boundary_dims().map(|d| d[1]).unwrap_or_default();
                let masks = masks_at(0, l, self.sliding_window, self.dtype, &self.device)?;
                let prepared = prepare_stitched_backward(loss, &tape)?;
                Ok((tape, masks, prepared))
            },
        )?;
        finish_stitched_backward_with_cotangent(
            prepared,
            &tape,
            &trainable,
            |i, x| {
                let layer = &self.layers[i / 2];
                if i % 2 == 0 {
                    layer.forward_attention_tensor_parallel(
                        x,
                        &masks,
                        &self.rot,
                        plan,
                        Some(comm),
                        true,
                    )
                } else {
                    layer.forward_mlp_tensor_parallel(x, plan, Some(comm), true)
                }
            },
            |_, cot| all_reduce_sum_value(cot, plan, comm),
            |i, local_vjp| {
                coordinate_local_candle_call(
                    comm,
                    &format!("Gemma4 tensor-parallel segment {i} local VJP"),
                    || {
                        #[cfg(test)]
                        maybe_inject_tensor_parallel_local_stage_failure(
                            "Gemma4 tensor-parallel local VJP",
                        )?;
                        local_vjp
                    },
                )
            },
        )
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

    fn validate_tensor_parallel_execution(&self, comm: &dyn Comm) -> CandleResult<()> {
        let plan = plan_from_comm(comm)?;
        self.validate_tensor_parallel_execution_support(plan)?;
        self.validate_tensor_parallel_plan(plan)
    }

    fn forward_tensor_parallel(&self, input_ids: &Tensor, comm: &dyn Comm) -> CandleResult<Tensor> {
        Gemma4GradModel::forward_tensor_parallel(self, input_ids, comm)
    }

    fn forward_tensor_parallel_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        Gemma4GradModel::forward_tensor_parallel_narrowed(self, input_ids, start, len, comm)
    }

    fn forward_tensor_parallel_detached(
        &self,
        input_ids: &Tensor,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        Gemma4GradModel::forward_tensor_parallel_detached(self, input_ids, comm)
    }

    fn forward_tensor_parallel_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        Gemma4GradModel::forward_tensor_parallel_detached_narrowed(
            self, input_ids, start, len, comm,
        )
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        Gemma4GradModel::backward(self, loss)
    }

    fn backward_tensor_parallel(&self, loss: &Tensor, comm: &dyn Comm) -> CandleResult<GradStore> {
        Gemma4GradModel::backward_tensor_parallel(self, loss, comm)
    }

    fn supports_sharded_tensor_parallel_backward(&self) -> bool {
        self.remat
    }

    fn trainable_vars(&self) -> Vec<Var> {
        Gemma4GradModel::trainable_vars(self)
    }

    fn requires_rollout_tensor_snapshot(&self) -> bool {
        false
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

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.q_weight
            .validate_column_parallel_support(plan, "attention_q_out")?;
        self.k_weight
            .validate_column_parallel_support(plan, "attention_k_out")?;
        if let Some(weight) = &self.v_weight {
            weight.validate_column_parallel_support(plan, "attention_v_out")?;
        }
        self.o_weight
            .validate_row_parallel_support(plan, "attention_hidden")?;
        tp_shard(plan, "num_attention_heads", self.num_heads)?;
        tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?;
        tp_shard(plan, "attention_q_out", self.q_weight.dims2()?.0)?;
        tp_shard(plan, "attention_k_out", self.k_weight.dims2()?.0)?;
        if let Some(w) = &self.v_weight {
            tp_shard(plan, "attention_v_out", w.dims2()?.0)?;
        }
        tp_shard(plan, "attention_hidden", self.o_weight.dims2()?.1)?;
        Ok(())
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

    fn forward_tensor_parallel(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &Gemma4Rotary,
        plan: TensorParallelPlan,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let partial = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 cached attention payload staging",
            || {
                let (b, l, _) = x.dims3()?;
                let in_dtype = x.dtype();
                let heads = tp_shard(plan, "num_attention_heads", self.num_heads)?.len;
                let kv_heads = tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?.len;
                let attn_hidden = heads * self.head_dim;

                let q = self
                    .q_weight
                    .column_parallel_forward(x, plan, "attention_q_out")?;
                let k_raw = self
                    .k_weight
                    .column_parallel_forward(x, plan, "attention_k_out")?;
                let v_raw = match &self.v_weight {
                    Some(w) => w.column_parallel_forward(x, plan, "attention_v_out")?,
                    None => k_raw.clone(),
                };

                let q = q.reshape((b, l, heads, self.head_dim))?;
                let k = k_raw.reshape((b, l, kv_heads, self.head_dim))?;
                let v = v_raw.reshape((b, l, kv_heads, self.head_dim))?;
                let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
                let k = self.k_norm.forward(&k)?;
                let v = self.v_norm.forward(&v)?.transpose(1, 2)?;
                let k = rot.apply(self.kind, &k.transpose(1, 2)?, offset, l)?;
                let q = rot.apply(self.kind, &q, offset, l)?;
                let (k, v, key_start) = self.cache.append(&k, &v.contiguous()?)?;
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
                let probs =
                    softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(in_dtype)?;
                let ctx = probs.matmul(&v)?;
                let ctx = ctx
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((b, l, attn_hidden))?;
                self.o_weight.row_parallel_forward_partial_from_shard(
                    &ctx,
                    plan,
                    "attention_hidden",
                )
            },
        )?;
        reduce_row_parallel_output(&partial, plan, Some(comm))
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

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.gate_weight
            .validate_column_parallel_support(plan, "intermediate_size")?;
        self.up_weight
            .validate_column_parallel_support(plan, "intermediate_size")?;
        self.down_weight
            .validate_row_parallel_support(plan, "intermediate_size")?;
        tp_shard(plan, "intermediate_size", self.gate_weight.dims2()?.0)?;
        tp_shard(plan, "intermediate_size", self.up_weight.dims2()?.0)?;
        tp_shard(plan, "intermediate_size", self.down_weight.dims2()?.1)?;
        Ok(())
    }

    fn forward_tensor_parallel(
        &self,
        x: &Tensor,
        plan: TensorParallelPlan,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let partial = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 cached MLP payload staging",
            || {
                let g = self
                    .gate_weight
                    .column_parallel_forward(x, plan, "intermediate_size")?;
                let u = self
                    .up_weight
                    .column_parallel_forward(x, plan, "intermediate_size")?;
                let h = self.act.forward(&g)?.broadcast_mul(&u)?;
                self.down_weight.row_parallel_forward_partial_from_shard(
                    &h,
                    plan,
                    "intermediate_size",
                )
            },
        )?;
        reduce_row_parallel_output(&partial, plan, Some(comm))
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

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.attn.validate_tensor_parallel_plan(plan)?;
        self.mlp.validate_tensor_parallel_plan(plan)
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

    fn forward_tensor_parallel(
        &mut self,
        x: &Tensor,
        offset: usize,
        masks: &Gemma4Masks,
        rot: &Gemma4Rotary,
        plan: TensorParallelPlan,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let h = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 cached attention boundary preparation",
            || self.input_layernorm.forward(x),
        )?;
        let h =
            self.attn
                .forward_tensor_parallel(&h, offset, masks.get(self.kind), rot, plan, comm)?;
        let (x, h2) = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 cached attention boundary completion",
            || {
                let h = self.post_attention_layernorm.forward(&h)?;
                let x = x.broadcast_add(&h)?;
                let h2 = self.pre_feedforward_layernorm.forward(&x)?;
                Ok((x, h2))
            },
        )?;
        let h2 = self.mlp.forward_tensor_parallel(&h2, plan, comm)?;
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 cached MLP boundary completion",
            || {
                let h2 = self.post_feedforward_layernorm.forward(&h2)?;
                let x = x.broadcast_add(&h2)?;
                x.broadcast_mul(&self.layer_scalar)
            },
        )
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
    base_quantization: BaseQuantization,
    weight_plan: TensorParallelPlan,
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
            base_quantization: model.base_quantization,
            weight_plan: model.weight_plan,
        })
    }

    /// Logits for `input_ids` placed at absolute positions starting at `offset`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the decoder stores rank-local tensor-parallel
    /// weights (use
    /// [`forward_tensor_parallel`](Self::forward_tensor_parallel) instead),
    /// `offset` disagrees with the cache length, or any tensor op fails.
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        self.validate_ordinary_execution_support()?;
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

    /// Logits through the tensor-parallel cached projection/collective path,
    /// driven by the explicit communicator.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `comm` does not match the stored rank-local
    /// weight plan, rank/world or projection-shape validation fails, `offset`
    /// disagrees with the cache length, a collective fails, or any tensor op
    /// fails.
    pub fn forward_tensor_parallel(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let (plan, mut h, masks) = coordinate_local_candle_call(
            comm,
            "Gemma4 cached tensor-parallel input staging",
            || {
                let plan = plan_from_comm(comm)?;
                self.validate_tensor_parallel_execution_support(plan)?;
                let (b, l) = input_ids.dims2()?;
                let cached = self
                    .layers
                    .first()
                    .map_or(0, |layer| layer.attn.current_seq_len());
                if offset != cached {
                    bail!(
                        "Gemma4MergedDecoder::forward_tensor_parallel: offset {offset} != cached \
                         sequence length {cached}"
                    );
                }
                self.validate_tensor_parallel_plan(plan)?;
                let ids = input_ids.flatten_all()?;
                let h = (self
                    .embed
                    .index_select(&ids, 0)?
                    .reshape((b, l, self.hidden))?
                    * self.embed_scale)?;
                let masks = merged_masks_at(offset, l, self.dtype, &self.device)?;
                Ok((plan, h, masks))
            },
        )?;
        for layer in &mut self.layers {
            h = layer.forward_tensor_parallel(&h, offset, &masks, &self.rot, plan, comm)?;
        }
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "Gemma4 cached tensor-parallel output staging",
            || {
                let h = self.norm.forward(&h)?;
                let logits = frozen_linear(&h, &self.embed)?;
                softcap(&logits, self.final_logit_softcap)
            },
        )
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        for layer in &self.layers {
            layer.validate_tensor_parallel_plan(plan)?;
        }
        Ok(())
    }

    fn validate_tensor_parallel_execution_support(
        &self,
        execution_plan: TensorParallelPlan,
    ) -> CandleResult<()> {
        if self.base_quantization == BaseQuantization::Q8_0 {
            bail!(
                "Gemma4MergedDecoder tensor_parallel execution does not support q8_0 base \
                 projections; disable tensor_parallel for world-one Q8_0 until rank-local \
                 quantized shards are implemented"
            );
        }
        if self.weight_plan.is_sharded() && self.weight_plan != execution_plan {
            bail!(
                "Gemma4MergedDecoder rank-local weight plan rank {}/world {} does not match \
                 execution plan rank {}/world {}",
                self.weight_plan.rank(),
                self.weight_plan.world_size(),
                execution_plan.rank(),
                execution_plan.world_size()
            );
        }
        Ok(())
    }

    fn validate_ordinary_execution_support(&self) -> CandleResult<()> {
        if self.weight_plan.is_sharded() {
            bail!(
                "ordinary Gemma4MergedDecoder forward cannot use rank-local tensor_parallel \
                 base weights; call forward_tensor_parallel with the matching communicator"
            );
        }
        Ok(())
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

    fn forward_tensor_parallel(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        Gemma4MergedDecoder::forward_tensor_parallel(self, input_ids, offset, comm)
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

    use rand::rngs::Xoshiro256PlusPlus;
    use rand::{RngExt, SeedableRng};
    use std::time::Duration;

    use crate::comm::{CommError, LocalComm};
    use crate::nn::grad_coverage;
    use crate::sharded_safetensors::varbuilder_from_rank_local_safetensors;
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

    fn world_three_cfg() -> Gemma4Config {
        let mut cfg = tiny_cfg();
        let text = &mut cfg.text_config;
        text.vocab_size = 18;
        text.vocab_size_per_layer_input = 18;
        text.hidden_size = 12;
        text.intermediate_size = 24;
        text.num_attention_heads = 3;
        text.num_key_value_heads = 3;
        text.num_global_key_value_heads = 3;
        text.head_dim = 4;
        text.global_head_dim = 4;
        cfg.validate().unwrap();
        cfg
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

    const WEIGHT_SEED: u64 = 0x4745_4D4D_4134; // "GEMMA4"

    /// Deterministic `N(0, 0.05)` test weights. Candle's CPU device cannot be
    /// seeded, and fresh random weights made quantized equivalence coverage
    /// depend on parallel test scheduling.
    fn seeded_randn(rng: &mut Xoshiro256PlusPlus, dims: &[usize]) -> Tensor {
        let n: usize = dims.iter().product();
        let mut values = Vec::with_capacity(n + 1);
        while values.len() < n {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random();
            let radius = (-2.0f32 * u1.ln()).sqrt();
            let (sin, cos) = (2.0 * std::f32::consts::PI * u2).sin_cos();
            values.push(0.05 * radius * cos);
            values.push(0.05 * radius * sin);
        }
        values.truncate(n);
        Tensor::from_vec(values, dims.to_vec(), &dev()).unwrap()
    }

    fn put_rand(
        t: &mut HashMap<String, Tensor>,
        rng: &mut Xoshiro256PlusPlus,
        name: &str,
        dims: &[usize],
    ) {
        t.insert(name.to_string(), seeded_randn(rng, dims));
    }

    fn put_ones(t: &mut HashMap<String, Tensor>, name: &str, dims: &[usize]) {
        t.insert(
            name.to_string(),
            Tensor::ones(dims.to_vec(), DType::F32, &dev()).unwrap(),
        );
    }

    fn weight_map(cfg: &Gemma4Config) -> HashMap<String, Tensor> {
        let tcfg = &cfg.text_config;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(WEIGHT_SEED);
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let h = tcfg.hidden_size;
        let i = tcfg.intermediate_size;

        put_rand(
            &mut t,
            &mut rng,
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
            put_rand(
                &mut t,
                &mut rng,
                &format!("{p}.self_attn.q_proj.weight"),
                &[q_out, h],
            );
            put_rand(
                &mut t,
                &mut rng,
                &format!("{p}.self_attn.k_proj.weight"),
                &[kv_out, h],
            );
            if !full {
                put_rand(
                    &mut t,
                    &mut rng,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[kv_out, h],
                );
            }
            put_rand(
                &mut t,
                &mut rng,
                &format!("{p}.self_attn.o_proj.weight"),
                &[h, q_out],
            );
            put_ones(&mut t, &format!("{p}.self_attn.q_norm.weight"), &[head_dim]);
            put_ones(&mut t, &format!("{p}.self_attn.k_norm.weight"), &[head_dim]);
            put_rand(
                &mut t,
                &mut rng,
                &format!("{p}.mlp.gate_proj.weight"),
                &[i, h],
            );
            put_rand(
                &mut t,
                &mut rng,
                &format!("{p}.mlp.up_proj.weight"),
                &[i, h],
            );
            put_rand(
                &mut t,
                &mut rng,
                &format!("{p}.mlp.down_proj.weight"),
                &[h, i],
            );
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

    fn arm_adapter_deterministic(model: &Gemma4GradModel) {
        for (i, v) in model.trainable_vars().iter().enumerate() {
            let dims = v.as_tensor().dims().to_vec();
            let n = dims.iter().product::<usize>();
            let data: Vec<f32> = (0..n)
                .map(|j| 0.02 + i as f32 * 0.003 + j as f32 * 0.001)
                .collect();
            v.set(&Tensor::from_vec(data, dims, &dev()).unwrap())
                .unwrap();
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

        let full = mlp
            .forward_tensor_parallel(&x, TensorParallelPlan::single(), None)
            .unwrap();
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

    #[test]
    fn gemma4_tensor_parallel_collective_forward_matches_unsharded_logits() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let vb = tiny_vb(&cfg);
        let input = ids(5);
        let reference_model = Gemma4GradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter_deterministic(&reference_model);
        let reference = reference_model.forward(&input).unwrap();
        let reference_flat = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let comms = LocalComm::world(2);
        let outputs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let vb = vb.clone();
                    let input = input.clone();
                    s.spawn(move || {
                        let model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &vb,
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        let logits = model.forward_tensor_parallel(&input, &comm).unwrap();
                        assert_eq!(logits.dims(), &[1, 5, cfg.text_config.vocab_size]);
                        logits.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (rank, got) in outputs.iter().enumerate() {
            let worst = got
                .iter()
                .zip(&reference_flat)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= 1e-5,
                "rank {rank} TP collective logits diverged from unsharded logits: {worst}"
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // two-rank value + local/global gradient oracle
    fn gemma4_tensor_parallel_checkpointed_backward_matches_unsharded_and_rejects_uncut() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let weights = weight_map(&cfg);
        let input = ids(5);

        let reference = Gemma4GradModel::load_with_targets(
            &cfg,
            &VarBuilder::from_tensors(weights.clone(), DType::F32, &dev()),
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter_deterministic(&reference);
        let reference_logits = reference.forward(&input).unwrap();
        let reference_loss = reference_logits.sqr().unwrap().sum_all().unwrap();
        let reference_store = reference.backward(&reference_loss).unwrap();
        let reference_grads: Vec<Vec<f32>> = reference
            .trainable_vars()
            .iter()
            .map(|var| {
                reference_store
                    .get(var)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();

        let rank_results = std::thread::scope(|scope| {
            let handles: Vec<_> = LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let input = input.clone();
                    scope.spawn(move || {
                        let load = || {
                            Gemma4GradModel::load_with_targets(
                                &cfg,
                                &VarBuilder::from_tensors(weights.clone(), DType::F32, &dev()),
                                2,
                                4.0,
                                DType::F32,
                                DenseLoraTargets::industrial(),
                            )
                            .unwrap()
                        };

                        let uncut = load();
                        arm_adapter_deterministic(&uncut);
                        let uncut_logits = uncut.forward_tensor_parallel(&input, &comm).unwrap();
                        let uncut_loss = uncut_logits.sqr().unwrap().sum_all().unwrap();
                        let uncut_error = uncut
                            .backward_tensor_parallel(&uncut_loss, &comm)
                            .unwrap_err()
                            .to_string();

                        let mut checkpointed = load();
                        arm_adapter_deterministic(&checkpointed);
                        checkpointed.set_activation_checkpointing(true);
                        let detached = checkpointed
                            .forward_tensor_parallel_detached(&input, &comm)
                            .unwrap();
                        let checkpointed_logits =
                            checkpointed.forward_tensor_parallel(&input, &comm).unwrap();
                        let checkpointed_loss =
                            checkpointed_logits.sqr().unwrap().sum_all().unwrap();
                        let checkpointed_store = checkpointed
                            .backward_tensor_parallel(&checkpointed_loss, &comm)
                            .unwrap();

                        let vars = checkpointed.trainable_vars();
                        let checkpointed_grads: Vec<Vec<f32>> = vars
                            .iter()
                            .map(|var| {
                                checkpointed_store
                                    .get(var)
                                    .unwrap()
                                    .flatten_all()
                                    .unwrap()
                                    .to_vec1::<f32>()
                                    .unwrap()
                            })
                            .collect();
                        (
                            comm.rank(),
                            uncut_error,
                            detached,
                            checkpointed_logits,
                            checkpointed_grads,
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        let mut summed_checkpointed = reference_grads
            .iter()
            .map(|grad| vec![0.0_f32; grad.len()])
            .collect::<Vec<_>>();
        for (rank, uncut_error, detached, checkpointed_logits, checkpointed_grads) in rank_results {
            assert!(
                uncut_error.contains("requires activation checkpointing"),
                "rank {rank} did not fail closed for uncut TP backward: {uncut_error}"
            );
            assert!(
                max_abs_diff(&detached, &reference_logits) <= 1e-5,
                "rank {rank} detached TP logits diverged under checkpointing"
            );
            assert!(
                max_abs_diff(&checkpointed_logits, &reference_logits) <= 1e-5,
                "rank {rank} checkpointed TP logits diverged"
            );
            for (sum, rank_grad) in summed_checkpointed.iter_mut().zip(checkpointed_grads) {
                for (total, value) in sum.iter_mut().zip(rank_grad) {
                    *total += value;
                }
            }
        }
        for (var_index, (summed, reference)) in
            summed_checkpointed.iter().zip(&reference_grads).enumerate()
        {
            let (worst_index, worst) = summed
                .iter()
                .zip(reference)
                .enumerate()
                .map(|(index, (a, b))| (index, (a - b).abs()))
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();
            assert!(
                worst <= 1e-4,
                "TP-reduced var {var_index} gradient diverged from unsharded at element \
                 {worst_index}: {worst}; summed={}, reference={}",
                summed[worst_index],
                reference[worst_index]
            );
        }
    }

    #[test]
    fn gemma4_tensor_parallel_backward_coordinates_asymmetric_tail_readiness() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let weights = weight_map(&cfg);
        let input = ids(4);

        let errors = std::thread::scope(|scope| {
            let handles: Vec<_> = LocalComm::world_with_timeout(2, Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let input = input.clone();
                    scope.spawn(move || {
                        let mut model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        model.set_activation_checkpointing(true);
                        let logits = model.forward_tensor_parallel(&input, &comm).unwrap();
                        let loss = if comm.rank() == 0 {
                            logits.sqr().unwrap().sum_all().unwrap()
                        } else {
                            let unrelated =
                                Var::from_tensor(&Tensor::new(1.0_f32, &dev()).unwrap()).unwrap();
                            unrelated.as_tensor().sqr().unwrap().sum_all().unwrap()
                        };
                        (
                            comm.rank(),
                            model
                                .backward_tensor_parallel(&loss, &comm)
                                .unwrap_err()
                                .to_string(),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for (rank, error) in errors {
            if rank == 0 {
                assert!(error.contains("readiness failed on a peer"), "{error}");
                assert!(error.contains("before the next collective"), "{error}");
            } else {
                assert!(
                    error.contains("does not reach the tape's tail boundary"),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn gemma4_tensor_parallel_backward_coordinates_asymmetric_checkpointing_state() {
        let cfg = tiny_cfg();
        let weights = weight_map(&cfg);

        let errors = std::thread::scope(|scope| {
            let handles = LocalComm::world_with_timeout(2, Duration::from_secs(2))
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    scope.spawn(move || {
                        let mut model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        model.set_activation_checkpointing(comm.rank() == 0);
                        model
                            .backward_tensor_parallel(&Tensor::new(1.0_f32, &dev()).unwrap(), &comm)
                            .unwrap_err()
                            .to_string()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for error in errors {
            assert!(
                error.contains(
                    "activation-checkpointing state differs across tensor-parallel ranks"
                ),
                "{error}"
            );
        }
    }

    fn two_rank_adapter_manifest_errors(
        make: impl Fn(usize) -> (Vec<Var>, String) + Sync,
    ) -> Vec<String> {
        std::thread::scope(|scope| {
            let make = &make;
            let handles = LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    scope.spawn(move || {
                        let (vars, recipe) = make(comm.rank());
                        let plan = TensorParallelPlan::new(comm.rank(), comm.world_size()).unwrap();
                        validate_replicated_adapter_values(
                            &vars,
                            &recipe,
                            &Device::Cpu,
                            plan,
                            &comm,
                        )
                        .unwrap_err()
                        .to_string()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
    }

    #[test]
    fn gemma4_adapter_validation_coordinates_asymmetric_inter_collective_failures() {
        for (stage, injected, peer) in [
            (
                1,
                "injected adapter manifest staging failure",
                "adapter manifest staging failed on a peer tensor-parallel rank",
            ),
            (
                2,
                "injected adapter manifest readback failure",
                "adapter manifest readback failed on a peer tensor-parallel rank",
            ),
        ] {
            let errors = std::thread::scope(|scope| {
                let handles = LocalComm::world_with_timeout(2, Duration::from_secs(2))
                    .into_iter()
                    .map(|comm| {
                        scope.spawn(move || {
                            if comm.rank() == 1 {
                                inject_adapter_validation_stage_failure_once(stage);
                            }
                            let vars = vec![Var::from_tensor(
                                &Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
                            )
                            .unwrap()];
                            let plan =
                                TensorParallelPlan::new(comm.rank(), comm.world_size()).unwrap();
                            (
                                comm.rank(),
                                validate_replicated_adapter_values(
                                    &vars,
                                    "attn:q",
                                    &Device::Cpu,
                                    plan,
                                    &comm,
                                )
                                .unwrap_err()
                                .to_string(),
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });

            for (rank, error) in errors {
                if rank == 1 {
                    assert!(error.contains(injected), "stage {stage}: {error}");
                } else {
                    assert!(error.contains(peer), "stage {stage}: {error}");
                }
            }
        }
    }

    #[derive(Debug)]
    struct F32F64OnlyComm(LocalComm);

    impl Comm for F32F64OnlyComm {
        fn rank(&self) -> usize {
            self.0.rank()
        }

        fn world_size(&self) -> usize {
            self.0.world_size()
        }

        fn validate_all_reduce_sum(&self, tensors: &[Tensor]) -> Result<(), CommError> {
            if let Some((index, tensor)) = tensors
                .iter()
                .enumerate()
                .find(|(_, tensor)| !matches!(tensor.dtype(), DType::F32 | DType::F64))
            {
                return Err(CommError::Mismatch(format!(
                    "test communicator accepts only f32/f64; tensor {index} is {:?}",
                    tensor.dtype()
                )));
            }
            Ok(())
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
            self.validate_all_reduce_sum(tensors)?;
            self.0.all_reduce_sum(tensors)
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
            self.0.all_reduce_scalar_sum(value)
        }
    }

    #[test]
    fn gemma4_adapter_manifest_preflight_rejects_recipe_count_shape_and_dtype_mismatches() {
        let var = |tensor: Tensor| Var::from_tensor(&tensor).unwrap();
        let recipe = two_rank_adapter_manifest_errors(|rank| {
            (
                vec![var(Tensor::zeros(2, DType::F32, &dev()).unwrap())],
                if rank == 0 { "attn:q" } else { "attn:k" }.to_owned(),
            )
        });
        let count = two_rank_adapter_manifest_errors(|rank| {
            let mut vars = vec![var(Tensor::zeros(2, DType::F32, &dev()).unwrap())];
            if rank == 1 {
                vars.push(var(Tensor::zeros(1, DType::F32, &dev()).unwrap()));
            }
            (vars, "attn:q".to_owned())
        });
        let shape = two_rank_adapter_manifest_errors(|rank| {
            (
                vec![var(Tensor::zeros(
                    if rank == 0 { 2 } else { 3 },
                    DType::F32,
                    &dev(),
                )
                .unwrap())],
                "attn:q".to_owned(),
            )
        });
        let dtype = two_rank_adapter_manifest_errors(|rank| {
            (
                vec![var(Tensor::zeros(
                    2,
                    if rank == 0 { DType::F32 } else { DType::BF16 },
                    &dev(),
                )
                .unwrap())],
                "attn:q".to_owned(),
            )
        });

        for error in recipe.into_iter().chain(count).chain(shape).chain(dtype) {
            assert!(
                error.contains("adapter recipe, tensor count, shapes, or dtypes differ"),
                "{error}"
            );
            assert!(error.contains("adapter values were not reduced"), "{error}");
        }
    }

    #[test]
    fn gemma4_adapter_manifest_preflight_globalizes_mixed_unsupported_dtype() {
        let errors = std::thread::scope(|scope| {
            let handles = LocalComm::world(2)
                .into_iter()
                .map(F32F64OnlyComm)
                .map(|comm| {
                    scope.spawn(move || {
                        let dtype = if comm.rank() == 0 {
                            DType::F32
                        } else {
                            DType::BF16
                        };
                        let vars =
                            vec![
                                Var::from_tensor(&Tensor::zeros(2, dtype, &Device::Cpu).unwrap())
                                    .unwrap(),
                            ];
                        let plan = TensorParallelPlan::new(comm.rank(), comm.world_size()).unwrap();
                        validate_replicated_adapter_values(
                            &vars,
                            "attn:q",
                            &Device::Cpu,
                            plan,
                            &comm,
                        )
                        .unwrap_err()
                        .to_string()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert!(
            errors
                .iter()
                .any(|error| error.contains("local adapter payload is invalid")),
            "the unsupported rank should retain its local diagnostic: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("payload validation failed on a peer")),
            "the valid rank should fail before adapter values are reduced: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .all(|error| error.contains("adapter values were not reduced")),
            "{errors:?}"
        );
    }

    // Emulates the NcclComm contract when rank 1's model tensors live on a
    // different device: scalar controls are communicator-staged, while any
    // caller-supplied tensor collective re-runs the local device preflight.
    #[derive(Debug)]
    struct WrongDeviceComm {
        inner: LocalComm,
        tensor_payload_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[derive(Debug)]
    struct CountingTensorComm {
        inner: LocalComm,
        tensor_payload_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Comm for CountingTensorComm {
        fn rank(&self) -> usize {
            self.inner.rank()
        }

        fn world_size(&self) -> usize {
            self.inner.world_size()
        }

        fn validate_all_reduce_sum(&self, tensors: &[Tensor]) -> Result<(), CommError> {
            self.inner.validate_all_reduce_sum(tensors)
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
            self.tensor_payload_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.all_reduce_sum(tensors)
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
            self.inner.all_reduce_scalar_sum(value)
        }
    }

    impl Comm for WrongDeviceComm {
        fn rank(&self) -> usize {
            self.inner.rank()
        }

        fn world_size(&self) -> usize {
            self.inner.world_size()
        }

        fn validate_all_reduce_sum(&self, _tensors: &[Tensor]) -> Result<(), CommError> {
            if self.rank() == 1 {
                return Err(CommError::Mismatch(
                    "test adapter tensor is not on this communicator's device".to_owned(),
                ));
            }
            Ok(())
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
            self.tensor_payload_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.validate_all_reduce_sum(tensors)?;
            self.inner.all_reduce_sum(tensors)
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
            self.inner.all_reduce_scalar_sum(value)
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn gemma4_builtin_forward_stages_coordinate_asymmetric_errors_and_panics() {
        for (mode, stage, expected_tensor_calls) in [
            ("full", "Gemma4 attention boundary completion", 2),
            ("full", "Gemma4 MLP boundary completion", 4),
            ("cached", "Gemma4 cached attention boundary completion", 2),
            ("cached", "Gemma4 cached MLP boundary completion", 4),
            ("remat", "Gemma4 rematerialized attention capture", 0),
            ("remat", "Gemma4 rematerialized MLP capture", 2),
        ] {
            for panic in [false, true] {
                let mut cfg = tiny_cfg();
                cfg.text_config.num_key_value_heads = 2;
                cfg.text_config.num_global_key_value_heads = 2;
                cfg.validate().unwrap();
                let weights = weight_map(&cfg);
                let input = ids(4);
                let tensor_payload_calls =
                    std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let errors = std::thread::scope(|scope| {
                    let handles = LocalComm::world_with_timeout(2, Duration::from_secs(2))
                        .into_iter()
                        .map(|inner| {
                            let cfg = cfg.clone();
                            let weights = weights.clone();
                            let input = input.clone();
                            let tensor_payload_calls = std::sync::Arc::clone(&tensor_payload_calls);
                            scope.spawn(move || {
                                let rank = inner.rank();
                                let comm = CountingTensorComm {
                                    inner,
                                    tensor_payload_calls,
                                };
                                let mut model = Gemma4GradModel::load_with_targets(
                                    &cfg,
                                    &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                                    2,
                                    4.0,
                                    DType::F32,
                                    DenseLoraTargets::industrial(),
                                )
                                .unwrap();
                                if rank == 1 {
                                    inject_tensor_parallel_local_stage_failure_once(stage, panic);
                                }
                                let result = match mode {
                                    "full" => model.forward_tensor_parallel(&input, &comm),
                                    "cached" => {
                                        let mut decoder = model.merged_decoder().unwrap();
                                        decoder.forward_tensor_parallel(&input, 0, &comm)
                                    }
                                    "remat" => {
                                        model.set_activation_checkpointing(true);
                                        model.forward_tensor_parallel(&input, &comm)
                                    }
                                    other => unreachable!("unknown test mode {other}"),
                                };
                                (
                                    rank,
                                    result.unwrap_err().to_string(),
                                    tensor_parallel_local_stage_fault_consumed(),
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|handle| handle.join().unwrap())
                        .collect::<Vec<_>>()
                });

                for (rank, error, fault_consumed) in errors {
                    if rank == 1 {
                        assert!(error.contains(&format!("injected {stage}")), "{error}");
                        assert!(fault_consumed, "{stage} injector was not consumed");
                    } else {
                        assert!(error.contains("failed on a peer"), "{error}");
                    }
                }
                assert_eq!(
                    tensor_payload_calls.load(std::sync::atomic::Ordering::SeqCst),
                    expected_tensor_calls,
                    "mode={mode} stage={stage} panic={panic}: a later tensor payload was entered"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn gemma4_backward_coordinates_readiness_and_local_vjp_failures_before_payloads() {
        for (stage, expected_tensor_calls) in [
            ("Gemma4 tensor-parallel backward readiness", 4),
            ("Gemma4 tensor-parallel local VJP", 6),
        ] {
            for panic in [false, true] {
                let mut cfg = tiny_cfg();
                cfg.text_config.num_key_value_heads = 2;
                cfg.text_config.num_global_key_value_heads = 2;
                cfg.validate().unwrap();
                let weights = weight_map(&cfg);
                let input = ids(4);
                let tensor_payload_calls =
                    std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
                let errors = std::thread::scope(|scope| {
                    let handles = LocalComm::world_with_timeout(2, Duration::from_secs(2))
                        .into_iter()
                        .map(|inner| {
                            let cfg = cfg.clone();
                            let weights = weights.clone();
                            let input = input.clone();
                            let tensor_payload_calls = std::sync::Arc::clone(&tensor_payload_calls);
                            let barrier = std::sync::Arc::clone(&barrier);
                            scope.spawn(move || {
                                let rank = inner.rank();
                                let comm = CountingTensorComm {
                                    inner,
                                    tensor_payload_calls: std::sync::Arc::clone(
                                        &tensor_payload_calls,
                                    ),
                                };
                                let mut model = Gemma4GradModel::load_with_targets(
                                    &cfg,
                                    &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                                    2,
                                    4.0,
                                    DType::F32,
                                    DenseLoraTargets::industrial(),
                                )
                                .unwrap();
                                arm_adapter_deterministic(&model);
                                model.set_activation_checkpointing(true);
                                let loss = model
                                    .forward_tensor_parallel(&input, &comm)
                                    .unwrap()
                                    .sqr()
                                    .unwrap()
                                    .sum_all()
                                    .unwrap();
                                barrier.wait();
                                if rank == 0 {
                                    tensor_payload_calls
                                        .store(0, std::sync::atomic::Ordering::SeqCst);
                                }
                                barrier.wait();
                                if rank == 1 {
                                    inject_tensor_parallel_local_stage_failure_once(stage, panic);
                                }
                                (
                                    rank,
                                    model
                                        .backward_tensor_parallel(&loss, &comm)
                                        .unwrap_err()
                                        .to_string(),
                                    tensor_parallel_local_stage_fault_consumed(),
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|handle| handle.join().unwrap())
                        .collect::<Vec<_>>()
                });

                for (rank, error, fault_consumed) in errors {
                    if rank == 1 {
                        assert!(error.contains(&format!("injected {stage}")), "{error}");
                        assert!(fault_consumed, "{stage} injector was not consumed");
                    } else {
                        assert!(error.contains("failed on a peer"), "{error}");
                    }
                }
                assert_eq!(
                    tensor_payload_calls.load(std::sync::atomic::Ordering::SeqCst),
                    expected_tensor_calls,
                    "stage={stage} panic={panic}: a later backward payload was entered"
                );
            }
        }
    }

    #[derive(Debug)]
    struct FailAfterArmedComm {
        inner: LocalComm,
        armed: std::sync::atomic::AtomicBool,
        remaining_successes: std::sync::atomic::AtomicUsize,
        failed: std::sync::atomic::AtomicBool,
        failures_triggered: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        calls_after_failure: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl FailAfterArmedComm {
        fn new(
            inner: LocalComm,
            failures_triggered: std::sync::Arc<std::sync::atomic::AtomicUsize>,
            calls_after_failure: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        ) -> Self {
            Self {
                inner,
                armed: std::sync::atomic::AtomicBool::new(false),
                remaining_successes: std::sync::atomic::AtomicUsize::new(0),
                failed: std::sync::atomic::AtomicBool::new(false),
                failures_triggered,
                calls_after_failure,
            }
        }

        fn arm_after(&self, successful_collectives: usize) {
            self.remaining_successes
                .store(successful_collectives, std::sync::atomic::Ordering::SeqCst);
            self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
        }

        fn enter_collective(&self) -> Result<(), CommError> {
            if !self.armed.load(std::sync::atomic::Ordering::SeqCst) {
                return Ok(());
            }
            if self.failed.load(std::sync::atomic::Ordering::SeqCst) {
                self.calls_after_failure
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                return Err(CommError::Poisoned(
                    "collective issued after injected adapter-validation failure".into(),
                ));
            }
            let remaining = self
                .remaining_successes
                .load(std::sync::atomic::Ordering::SeqCst);
            if remaining > 0 {
                self.remaining_successes
                    .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                return Ok(());
            }
            self.failed.store(true, std::sync::atomic::Ordering::SeqCst);
            self.failures_triggered
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(CommError::Mismatch(
                "injected adapter-validation collective failure".into(),
            ))
        }
    }

    impl Comm for FailAfterArmedComm {
        fn rank(&self) -> usize {
            self.inner.rank()
        }

        fn world_size(&self) -> usize {
            self.inner.world_size()
        }

        fn validate_all_reduce_sum(&self, tensors: &[Tensor]) -> Result<(), CommError> {
            self.inner.validate_all_reduce_sum(tensors)
        }

        fn all_reduce_sum(&self, tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
            self.enter_collective()?;
            self.inner.all_reduce_sum(tensors)
        }

        fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
            self.enter_collective()?;
            self.inner.all_reduce_scalar_sum(value)
        }
    }

    #[test]
    fn gemma4_backward_stops_after_adapter_validation_communication_failure() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let weights = weight_map(&cfg);
        let input = ids(4);
        let failures_triggered = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_after_failure = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let errors = std::thread::scope(|scope| {
            let handles = LocalComm::world_with_timeout(2, Duration::from_secs(2))
                .into_iter()
                .map(|inner| {
                    FailAfterArmedComm::new(
                        inner,
                        std::sync::Arc::clone(&failures_triggered),
                        std::sync::Arc::clone(&calls_after_failure),
                    )
                })
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let input = input.clone();
                    scope.spawn(move || {
                        let mut model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        model.set_activation_checkpointing(true);
                        let loss = model
                            .forward_tensor_parallel(&input, &comm)
                            .unwrap()
                            .sqr()
                            .unwrap()
                            .sum_all()
                            .unwrap();
                        // Backward performs plan, activation-checkpointing, adapter-enabled,
                        // and local adapter-state coordination before adapter payload
                        // validation. Fail that validation's first status collective and prove
                        // readiness never performs another one.
                        comm.arm_after(4);
                        let error = model.backward_tensor_parallel(&loss, &comm).unwrap_err();
                        assert!(is_comm_failure(&error), "{error}");
                        error.to_string()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert!(errors
            .iter()
            .all(|error| error.contains("injected adapter-validation collective failure")));
        assert_eq!(
            failures_triggered.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the injected failure must be consumed once on every rank"
        );
        assert_eq!(
            calls_after_failure.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "Gemma backward issued a collective after adapter validation killed the world"
        );
    }

    #[test]
    fn gemma4_adapter_payload_preflight_globalizes_wrong_device_before_tensor_collective() {
        let tensor_payload_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let errors = std::thread::scope(|scope| {
            let handles = LocalComm::world_with_timeout(2, Duration::from_secs(2))
                .into_iter()
                .map(|inner| WrongDeviceComm {
                    inner,
                    tensor_payload_calls: std::sync::Arc::clone(&tensor_payload_calls),
                })
                .map(|comm| {
                    scope.spawn(move || {
                        let vars = vec![Var::from_tensor(
                            &Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
                        )
                        .unwrap()];
                        let plan = TensorParallelPlan::new(comm.rank(), comm.world_size()).unwrap();
                        validate_replicated_adapter_values(
                            &vars,
                            "attn:q",
                            &Device::Cpu,
                            plan,
                            &comm,
                        )
                        .unwrap_err()
                        .to_string()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert!(
            errors
                .iter()
                .any(|error| error.contains("local adapter payload is invalid")),
            "the wrong-device rank should retain its local diagnostic: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("payload validation failed on a peer")),
            "the valid rank should fail through communicator-owned control traffic: {errors:?}"
        );
        assert_eq!(
            tensor_payload_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "no manifest or adapter-value tensor collective may start after an asymmetric \
             wrong-device preflight failure"
        );
    }

    #[test]
    fn gemma4_tensor_parallel_backward_rejects_cross_rank_adapter_disagreement() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let weights = weight_map(&cfg);
        let input = ids(4);

        let errors = std::thread::scope(|scope| {
            let handles: Vec<_> = LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let input = input.clone();
                    scope.spawn(move || {
                        let mut model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        model.set_activation_checkpointing(true);
                        let loss = model
                            .forward_tensor_parallel(&input, &comm)
                            .unwrap()
                            .sqr()
                            .unwrap()
                            .sum_all()
                            .unwrap();
                        if comm.rank() == 1 {
                            model.set_adapter_enabled(false);
                        }
                        model
                            .backward_tensor_parallel(&loss, &comm)
                            .unwrap_err()
                            .to_string()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for error in errors {
            assert!(
                error.contains("adapter enabled state differs across tensor-parallel ranks"),
                "{error}"
            );
        }
    }

    #[test]
    fn gemma4_tensor_parallel_backward_rejects_cross_rank_adapter_value_disagreement() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let weights = weight_map(&cfg);
        let input = ids(4);

        let errors = std::thread::scope(|scope| {
            let handles: Vec<_> = LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let input = input.clone();
                    scope.spawn(move || {
                        let mut model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        model.set_activation_checkpointing(true);
                        let loss = model
                            .forward_tensor_parallel(&input, &comm)
                            .unwrap()
                            .sqr()
                            .unwrap()
                            .sum_all()
                            .unwrap();
                        if comm.rank() == 1 {
                            let first = model.trainable_vars().remove(0);
                            first
                                .set(&first.as_tensor().affine(1.0, 1.0).unwrap())
                                .unwrap();
                        }
                        model
                            .backward_tensor_parallel(&loss, &comm)
                            .unwrap_err()
                            .to_string()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for error in errors {
            assert!(
                error.contains("replicated adapter tensor 0 differs across tensor-parallel ranks"),
                "{error}"
            );
        }
    }

    #[test]
    fn gemma4_tensor_parallel_checkpointed_backward_supports_world_three() {
        let cfg = world_three_cfg();
        let weights = weight_map(&cfg);
        let input = ids(4);

        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = LocalComm::world(3)
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let input = input.clone();
                    scope.spawn(move || {
                        let mut model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &VarBuilder::from_tensors(weights, DType::F32, &dev()),
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        model.set_activation_checkpointing(true);
                        let logits = model.forward_tensor_parallel(&input, &comm).unwrap();
                        let store = model
                            .backward_tensor_parallel(
                                &logits.sqr().unwrap().sum_all().unwrap(),
                                &comm,
                            )
                            .unwrap();
                        assert!(grad_coverage(&model.trainable_vars(), &store)
                            .unwrap()
                            .is_ok());
                        logits.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for rank in 1..outputs.len() {
            let worst = outputs[rank]
                .iter()
                .zip(&outputs[0])
                .map(|(got, reference)| (got - reference).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                worst <= 1e-5,
                "world-three rank {rank} logits mismatch: {worst}"
            );
        }
    }

    #[test]
    fn gemma4_tensor_parallel_checkpoint_tape_requires_matching_backward_hook() {
        let comm = LocalComm::world(1).pop().unwrap();
        let mut model = tiny_model();
        arm_adapter_deterministic(&model);
        model.set_activation_checkpointing(true);
        let loss = model
            .forward_tensor_parallel(&ids(3), &comm)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();

        let error = model.backward(&loss).unwrap_err().to_string();
        assert!(
            error.contains("captured by tensor-parallel execution"),
            "{error}"
        );
        assert!(error.contains("backward_tensor_parallel"), "{error}");
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // storage, live/cached, and backward oracle
    fn gemma4_rank_local_tensor_parallel_load_matches_unsharded_live_cached_and_backward() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let weights = weight_map(&cfg);
        let dir = unique_tmp("rank-local-tp-load");
        std::fs::create_dir_all(&dir).unwrap();
        candle_core::safetensors::save(&weights, dir.join("model.safetensors")).unwrap();

        let reference_vb = VarBuilder::from_tensors(weights, DType::F32, &dev());
        let mut reference_model = Gemma4GradModel::load_with_targets(
            &cfg,
            &reference_vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter_deterministic(&reference_model);
        reference_model.set_activation_checkpointing(true);
        let input = ids(5);
        let reference_live = reference_model.forward(&input).unwrap();
        let reference_store = reference_model
            .backward(&reference_live.sqr().unwrap().sum_all().unwrap())
            .unwrap();
        let reference_grads: Vec<Vec<f32>> = reference_model
            .trainable_vars()
            .iter()
            .map(|var| {
                reference_store
                    .get(var)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();
        let reference_live_values = reference_live
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let reference_var_shapes: Vec<_> = reference_model
            .trainable_vars()
            .iter()
            .map(|var| var.as_tensor().dims().to_vec())
            .collect();
        let mut reference_decoder = reference_model.merged_decoder().unwrap();
        let reference_cached = reference_decoder.forward(&input, 0).unwrap();
        let reference_cached = reference_cached
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let input = input.clone();
                    let dir = dir.clone();
                    let reference_var_shapes = reference_var_shapes.clone();
                    scope.spawn(move || {
                        let plan = TensorParallelPlan::new(comm.rank(), comm.world_size()).unwrap();
                        let vb = varbuilder_from_rank_local_safetensors(
                            &dir,
                            DType::F32,
                            &dev(),
                            plan,
                        )
                        .unwrap();
                        let mut model = Gemma4GradModel::load_with_targets_base_quantization_and_tensor_parallel(
                            &cfg,
                            &vb,
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                            BaseQuantization::None,
                            plan,
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        model.set_activation_checkpointing(true);

                        let layer = &model.layers[0];
                        assert_eq!(layer.attn.q_proj.stored_dims2().unwrap(), (4, 8));
                        assert_eq!(layer.attn.k_proj.stored_dims2().unwrap(), (4, 8));
                        assert_eq!(layer.attn.v_proj.as_ref().unwrap().stored_dims2().unwrap(), (4, 8));
                        assert_eq!(layer.attn.o_proj.stored_dims2().unwrap(), (8, 4));
                        assert_eq!(layer.mlp.gate_proj.stored_dims2().unwrap(), (8, 8));
                        assert_eq!(layer.mlp.up_proj.stored_dims2().unwrap(), (8, 8));
                        assert_eq!(layer.mlp.down_proj.stored_dims2().unwrap(), (8, 8));
                        let var_shapes: Vec<_> = model
                            .trainable_vars()
                            .iter()
                            .map(|var| var.as_tensor().dims().to_vec())
                            .collect();
                        assert_eq!(var_shapes, reference_var_shapes);
                        assert!(model
                            .forward(&input)
                            .unwrap_err()
                            .to_string()
                            .contains("ordinary Gemma4GradModel forward cannot use rank-local"));

                        let live = model.forward_tensor_parallel(&input, &comm).unwrap();
                        let grads = model
                            .backward_tensor_parallel(
                                &live.sqr().unwrap().sum_all().unwrap(),
                                &comm,
                            )
                            .unwrap();
                        for (pair_index, pair) in model.trainable_vars().chunks(2).enumerate() {
                            let coverage = grad_coverage(pair, &grads).unwrap();
                            assert!(
                                coverage.is_ok(),
                                "rank {} projection pair {pair_index} has dead/missing grads: \
                                 {coverage:?}",
                                comm.rank()
                            );
                        }
                        let rank_grads = model
                            .trainable_vars()
                            .iter()
                            .map(|var| {
                                grads
                                    .get(var)
                                    .unwrap()
                                    .flatten_all()
                                    .unwrap()
                                    .to_vec1::<f32>()
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        let mut decoder = model.merged_decoder().unwrap();
                        assert!(decoder
                            .forward(&input, 0)
                            .unwrap_err()
                            .to_string()
                            .contains("ordinary Gemma4MergedDecoder forward cannot use rank-local"));
                        let cached = decoder.forward_tensor_parallel(&input, 0, &comm).unwrap();
                        (
                            live.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                            cached.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                            rank_grads,
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        std::fs::remove_dir_all(&dir).ok();

        let mut summed_grads = reference_grads
            .iter()
            .map(|grad| vec![0.0_f32; grad.len()])
            .collect::<Vec<_>>();
        for (rank, (live, cached, rank_grads)) in outputs.iter().enumerate() {
            let live_worst = live
                .iter()
                .zip(&reference_live_values)
                .map(|(got, want)| (got - want).abs())
                .fold(0.0_f32, f32::max);
            let cached_worst = cached
                .iter()
                .zip(&reference_cached)
                .map(|(got, want)| (got - want).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                live_worst <= 1e-5,
                "rank {rank} live mismatch: {live_worst}"
            );
            assert!(
                cached_worst <= 1e-5,
                "rank {rank} cached mismatch: {cached_worst}"
            );
            for (summed, rank_grad) in summed_grads.iter_mut().zip(rank_grads) {
                for (total, value) in summed.iter_mut().zip(rank_grad) {
                    *total += value;
                }
            }
        }
        for (var_index, (summed, reference)) in
            summed_grads.iter().zip(&reference_grads).enumerate()
        {
            let worst = summed
                .iter()
                .zip(reference)
                .map(|(got, want)| (got - want).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                worst <= 1e-4,
                "rank-local TP-reduced var {var_index} gradient mismatch: {worst}"
            );
        }
    }

    #[test]
    fn gemma4_tensor_parallel_cached_decoder_matches_unsharded_decode() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.validate().unwrap();
        let vb = tiny_vb(&cfg);
        let input = ids(5);
        let prefix_len = 3;
        let suffix_len = 2;

        let reference_model = Gemma4GradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter_deterministic(&reference_model);
        let mut reference_decoder = reference_model.merged_decoder().unwrap();
        let prefix = input.narrow(1, 0, prefix_len).unwrap();
        let suffix = input.narrow(1, prefix_len, suffix_len).unwrap();
        let reference_prefix = reference_decoder.forward(&prefix, 0).unwrap();
        let reference_suffix = reference_decoder.forward(&suffix, prefix_len).unwrap();
        let reference_prefix_flat = reference_prefix
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let reference_suffix_flat = reference_suffix
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let comms = LocalComm::world(2);
        let outputs: Vec<(Vec<f32>, Vec<f32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let vb = vb.clone();
                    let prefix = prefix.clone();
                    let suffix = suffix.clone();
                    s.spawn(move || {
                        let model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &vb,
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        let mut decoder = model.merged_decoder().unwrap();
                        let prefix_logits =
                            decoder.forward_tensor_parallel(&prefix, 0, &comm).unwrap();
                        let suffix_logits = decoder
                            .forward_tensor_parallel(&suffix, prefix_len, &comm)
                            .unwrap();
                        assert_eq!(
                            prefix_logits.dims(),
                            &[1, prefix_len, cfg.text_config.vocab_size]
                        );
                        assert_eq!(
                            suffix_logits.dims(),
                            &[1, suffix_len, cfg.text_config.vocab_size]
                        );
                        (
                            prefix_logits
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap(),
                            suffix_logits
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap(),
                        )
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (rank, (prefix, suffix)) in outputs.iter().enumerate() {
            let prefix_worst = prefix
                .iter()
                .zip(&reference_prefix_flat)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                prefix_worst <= 1e-5,
                "rank {rank} TP cached Gemma 4 prefill diverged: {prefix_worst}"
            );
            let suffix_worst = suffix
                .iter()
                .zip(&reference_suffix_flat)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                suffix_worst <= 1e-5,
                "rank {rank} TP cached Gemma 4 decode diverged: {suffix_worst}"
            );
        }
    }

    #[test]
    fn gemma4_tensor_parallel_cached_decoder_preflights_layout_before_mutating_cache() {
        let mut cfg = tiny_cfg();
        cfg.text_config.num_key_value_heads = 2;
        cfg.text_config.num_global_key_value_heads = 2;
        cfg.text_config.intermediate_size = 15;
        cfg.validate().unwrap();
        let vb = tiny_vb(&cfg);
        let input = ids(3);

        let reference_model = Gemma4GradModel::load_with_targets(
            &cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        arm_adapter_deterministic(&reference_model);
        let mut reference_decoder = reference_model.merged_decoder().unwrap();
        let reference = reference_decoder.forward(&input, 0).unwrap();
        let reference_flat = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let comms = LocalComm::world(2);
        let outputs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let vb = vb.clone();
                    let input = input.clone();
                    s.spawn(move || {
                        let model = Gemma4GradModel::load_with_targets(
                            &cfg,
                            &vb,
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        let mut decoder = model.merged_decoder().unwrap();
                        let err = decoder
                            .forward_tensor_parallel(&input, 0, &comm)
                            .unwrap_err()
                            .to_string();
                        assert!(
                            err.contains("intermediate_size"),
                            "expected MLP intermediate preflight failure, got {err}"
                        );
                        let got = decoder.forward(&input, 0).unwrap();
                        got.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (rank, got) in outputs.iter().enumerate() {
            let worst = got
                .iter()
                .zip(&reference_flat)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= 1e-5,
                "rank {rank} Gemma 4 decoder cache mutated before unsupported TP layout failed: \
                 {worst}"
            );
        }
    }

    #[test]
    fn gemma4_quantized_multi_rank_tensor_parallel_rejects_before_cached_state_mutation() {
        let input = ids(3);
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    let input = input.clone();
                    scope.spawn(move || {
                        let mut cfg = quantized_tiny_cfg();
                        cfg.text_config.num_key_value_heads = 2;
                        cfg.text_config.num_global_key_value_heads = 2;
                        cfg.validate().unwrap();
                        let vb = tiny_vb(&cfg);
                        let model = Gemma4GradModel::load_with_targets_and_base_quantization(
                            &cfg,
                            &vb,
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                            BaseQuantization::Q8_0,
                        )
                        .unwrap();
                        let live_error = model
                            .forward_tensor_parallel(&input, &comm)
                            .unwrap_err()
                            .to_string();
                        assert!(
                            live_error.contains("does not support q8_0 base projections"),
                            "unexpected live TP rejection: {live_error}"
                        );

                        let mut decoder = model.merged_decoder().unwrap();
                        let cached_error = decoder
                            .forward_tensor_parallel(&input, 0, &comm)
                            .unwrap_err()
                            .to_string();
                        assert!(
                            cached_error.contains("does not support q8_0 base projections"),
                            "unexpected cached TP rejection: {cached_error}"
                        );

                        decoder.forward(&input, 0).unwrap().dims().to_vec()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for dims in results {
            assert_eq!(dims, [1, 3, quantized_tiny_cfg().text_config.vocab_size]);
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)] // assertion-heavy live/detached/cache path contract
    fn gemma4_quantized_world_one_tensor_parallel_rejects_but_ordinary_paths_work() {
        let input = ids(3);
        let model = quantized_tiny_model();
        let comm = LocalComm::world(1).pop().unwrap();

        let ordinary = model.forward(&input).unwrap();
        assert_eq!(
            ordinary.dims(),
            &[1, 3, quantized_tiny_cfg().text_config.vocab_size]
        );

        let live_error = model
            .forward_tensor_parallel(&input, &comm)
            .unwrap_err()
            .to_string();
        assert!(
            live_error.contains("does not support q8_0 base projections"),
            "unexpected live world-one TP rejection: {live_error}"
        );
        assert!(live_error.contains("disable tensor_parallel for world-one Q8_0"));

        let detached_error = model
            .forward_tensor_parallel_detached(&input, &comm)
            .unwrap_err()
            .to_string();
        assert!(
            detached_error.contains("does not support q8_0 base projections"),
            "unexpected detached world-one TP rejection: {detached_error}"
        );

        let mut decoder = model.merged_decoder().unwrap();
        let cached_error = decoder
            .forward_tensor_parallel(&input, 0, &comm)
            .unwrap_err()
            .to_string();
        assert!(
            cached_error.contains("does not support q8_0 base projections"),
            "unexpected cached world-one TP rejection: {cached_error}"
        );
        assert!(cached_error.contains("disable tensor_parallel for world-one Q8_0"));

        let ordinary_cached = decoder.forward(&input, 0).unwrap();
        assert_eq!(
            ordinary_cached.dims(),
            &[1, 3, quantized_tiny_cfg().text_config.vocab_size]
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
        assert!(!GradModel::requires_rollout_tensor_snapshot(&model));
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
        arm_adapter_deterministic(&model);
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
        arm_adapter_deterministic(&model);
        let input = ids(6);
        let mut dec = model.merged_decoder().unwrap();
        let mut pre_update_dec = model.merged_decoder().unwrap();

        let prefix_len = 3;
        let prefix = input.narrow(1, 0, prefix_len).unwrap();
        dec.forward(&prefix, 0).unwrap();
        pre_update_dec.forward(&prefix, 0).unwrap();

        let suffix_len = 2;
        let suffix = input.narrow(1, prefix_len, suffix_len).unwrap();
        let want = pre_update_dec.forward(&suffix, prefix_len).unwrap();
        overwrite_adapter(&model);
        let mut post_update_dec = model.merged_decoder().unwrap();
        post_update_dec.forward(&prefix, 0).unwrap();
        let post_update = post_update_dec.forward(&suffix, prefix_len).unwrap();
        let mutation_delta = max_abs_diff(&post_update, &want);
        assert!(
            mutation_delta > 0.05,
            "adapter overwrite must move the quantized decoder enough to make this gate \
             non-vacuous: {mutation_delta}"
        );

        let got = dec.forward(&suffix, prefix_len).unwrap();
        let snapshot_delta = max_abs_diff(&got, &want);
        assert!(
            snapshot_delta <= 1e-5,
            "existing quantized decoder observed adapter vars mutated after construction: \
             snapshot_delta={snapshot_delta}, mutation_delta={mutation_delta}"
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
