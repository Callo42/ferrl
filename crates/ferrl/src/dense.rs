//! The shared dense-transformer backbone — the qwen ≈ llama collapse.
//!
//! [`crate::qwen`] and [`crate::llama`] are the same decoder skeleton: a
//! pre-norm `RMSNorm` + GQA-attention + `SwiGLU`-MLP residual stack, an
//! uncached grad-bearing update forward (the GRPO scoring/KL path), and a
//! KV-cached grad-free rollout twin over merged weights. They differed only in
//! a few knobs:
//!
//! | knob | Qwen3 | dense Llama-3.x |
//! |------|-------|-----------------|
//! | per-head q/k `RMSNorm` before `RoPE` | yes | no |
//! | force-cast q/k/v to F32 in SDPA | no (model dtype) | yes (shipped non-flash) |
//! | causal-mask dtype | model dtype | F32 (matches the F32 scores) |
//! | `SwiGLU` activation | `cfg.hidden_act` | fixed `Silu` |
//! | `head_dim` | explicit `cfg.head_dim` | derived `hidden / heads` |
//! | `RoPE` inv-freqs | plain `1/θ^(2i/d)` | optional llama3 smoothing |
//! | rejected configs | `attention_bias`, `sliding_window` | `use_flash_attn` |
//!
//! This module factors the skeleton out once. The variation rides as **data**
//! on the blocks — an `Option<(q_norm, k_norm)>`, an `sdpa_f32` flag, a stored
//! [`Activation`], a `mask_dtype`, and a prebuilt [`RotaryTables`] — never as
//! type-level branching, so there is exactly one attention forward, one layer
//! walk, one checkpointing path, one merged-decoder.
//!
//! Each architecture supplies a zero-sized [`DenseArch`] marker that parses its
//! candle `Config` into a uniform [`DenseSpec`] (validating it and building its
//! `RoPE` tables). [`DenseGradModel`] is generic over that marker, so
//! `pub type QwenGradModel = DenseGradModel<QwenArch>` and its llama sibling
//! each keep their own `Config`-typed constructors and identity while sharing
//! one implementation. The numeric core stays pinned by each architecture's
//! own forward-equivalence gates against candle's shipped forward (the
//! behavior-preserving contract of the extraction).
//!
//! ## The three grad landmines (all replaced here)
//!
//! As in the pre-extraction forwards, the three autograd-cutting fused ops are
//! replaced by grad-bearing twins: `rms_norm` → [`crate::nn::RmsNorm`]
//! (`rms_norm_slow`), `rotary_emb::rope` → `rope_slow`, `softmax_last_dim` →
//! `softmax(_, D::Minus1)`. Every other op is grad-bearing and reused verbatim.

use std::cell::RefCell;
use std::marker::PhantomData;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::ops::softmax;
use candle_nn::rotary_emb::rope_slow;
use candle_nn::{Activation, VarBuilder};

use crate::blocks::{
    causal_mask, causal_mask_at, frozen_linear, repeat_kv, windowed, RotaryTables,
};
use crate::comm::Comm;
use crate::lora::{
    BaseQuantization, DenseLoraTargets, FrozenLinearSnapshot, Proj, ProjLoadOptions,
};
use crate::model::{CachedDecoder, GradModel};
use crate::nn::RmsNorm;
use crate::remat::{stitched_backward, RematTape};
use crate::tensor_parallel::{
    all_reduce_sum_straight_through, coordinate_local_candle_call, plan_from_comm, ShardRange,
    TensorParallelPlan,
};

#[cfg(test)]
thread_local! {
    static DENSE_TENSOR_PARALLEL_LOCAL_STAGE_FAULT:
        std::cell::RefCell<Option<(&'static str, bool)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn inject_dense_tensor_parallel_local_stage_failure_once(
    stage: &'static str,
    panic: bool,
) {
    DENSE_TENSOR_PARALLEL_LOCAL_STAGE_FAULT.with(|fault| fault.replace(Some((stage, panic))));
}

#[cfg(test)]
pub(crate) fn dense_tensor_parallel_local_stage_fault_consumed() -> bool {
    DENSE_TENSOR_PARALLEL_LOCAL_STAGE_FAULT.with(|fault| fault.borrow().is_none())
}

#[cfg(test)]
fn maybe_inject_dense_tensor_parallel_local_stage_failure(stage: &str) -> CandleResult<()> {
    let behavior = DENSE_TENSOR_PARALLEL_LOCAL_STAGE_FAULT.with(|fault| {
        let mut fault = fault.borrow_mut();
        if fault.as_ref().is_some_and(|(target, _)| *target == stage) {
            fault.take().map(|(_, panic)| panic)
        } else {
            None
        }
    });
    match behavior {
        Some(true) => panic!("injected {stage} panic"),
        Some(false) => candle_core::bail!("injected {stage} failure"),
        None => Ok(()),
    }
}

/// The per-architecture seam of the shared dense backbone.
///
/// An implementor is a zero-sized marker (e.g. `QwenArch`, `LlamaArch`) that
/// names its candle `Config` type and distills it — validated — into the
/// uniform [`DenseSpec`] the generic [`DenseGradModel`] builds from.
/// [`DenseGradModel`] is generic over this marker purely so each architecture's
/// `pub type` alias keeps its own `Config`-typed `load*` constructors and a
/// distinct type identity, over one shared implementation.
pub trait DenseArch {
    /// The architecture's candle config type (e.g. `qwen3::Config`).
    type Config;

    /// A short label for this architecture, used in load-time diagnostics
    /// (e.g. `"QwenGradModel"`).
    const LABEL: &'static str;

    /// Validate `cfg` and distill it — together with its `RoPE` tables, built
    /// at `dtype`/`device` — into the uniform [`DenseSpec`].
    ///
    /// Fails loud (rather than loading a silently non-parity model) on any
    /// config option this backbone does not implement, and on a degenerate head
    /// configuration that would divide-by-zero or silently truncate the GQA /
    /// `head_dim` arithmetic.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `cfg` requests an unsupported option or carries
    /// a degenerate head configuration, or if a `RoPE` table cannot be built.
    fn spec(cfg: &Self::Config, dtype: DType, device: &Device) -> CandleResult<DenseSpec>;
}

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
            maybe_inject_dense_tensor_parallel_local_stage_failure(label)?;
            call()
        }),
        None if plan.is_sharded() => candle_core::bail!(
            "{label} for tensor-parallel world size {} needs a communicator",
            plan.world_size()
        ),
        None => call(),
    }
}

/// The architecture-neutral distillation of a dense model's `Config`: the dims
/// plus the handful of knobs that vary across architectures, all resolved to
/// plain values the generic backbone consumes.
///
/// Produced by [`DenseArch::spec`]; consumed once by
/// [`DenseGradModel::load_with_targets`].
pub struct DenseSpec {
    /// Vocabulary size (rows of the embedding / LM head).
    pub vocab_size: usize,
    /// Model hidden size.
    pub hidden_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA; the query head count must be a multiple).
    pub num_key_value_heads: usize,
    /// Per-head dimension (explicit for Qwen3, derived `hidden / heads` for Llama).
    pub head_dim: usize,
    /// `SwiGLU` intermediate (feed-forward) size.
    pub intermediate_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_norm_eps: f32,
    /// Whether the LM head reuses the embedding matrix (no `lm_head.weight`).
    pub tie_word_embeddings: bool,
    /// Apply per-head q/k `RMSNorm` before `RoPE` (Qwen3 yes, Llama no).
    pub qk_norm: bool,
    /// Force-cast q/k/v to F32 for the score/softmax/context (the Llama shipped
    /// non-flash path; Qwen stays in the model dtype). Also selects the
    /// causal-mask dtype: F32 when set, the model dtype otherwise.
    pub sdpa_f32: bool,
    /// The `SwiGLU` activation (`cfg.hidden_act` for Qwen, fixed `Silu` for Llama).
    pub activation: Activation,
    /// Prebuilt `RoPE` tables (the architecture chooses plain vs llama3-smoothed
    /// inv-freqs at build time; the table layout itself is neutral).
    pub rot: RotaryTables,
}

/// One dense attention block; each projection carries the `LoRA` adapter or
/// stays frozen per the [`DenseLoraTargets`] recipe. Uncached, grad-bearing,
/// with the three grad-safe op twins. The architectural variation rides as
/// data: an optional per-head q/k norm and the `sdpa_f32` flag.
#[derive(Debug)]
pub(crate) struct DenseAttention {
    pub(crate) q_proj: Proj,
    pub(crate) k_proj: Proj,
    pub(crate) v_proj: Proj,
    pub(crate) o_proj: Proj,
    /// Per-head q/k `RMSNorm` applied before `RoPE` — `Some` for Qwen3, `None`
    /// for Llama.
    qk_norm: Option<(RmsNorm, RmsNorm)>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
    /// Force-cast q/k/v to F32 for the SDPA (Llama; a no-op at F32).
    sdpa_f32: bool,
}

impl DenseAttention {
    fn load(
        spec: &DenseSpec,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        base_quantization: BaseQuantization,
    ) -> CandleResult<Self> {
        let proj_opts = ProjLoadOptions::new(rank, alpha, adapter_dtype, base_quantization);
        let h = spec.hidden_size;
        let head_dim = spec.head_dim;
        let q_out = spec.num_attention_heads * head_dim;
        let kv_out = spec.num_key_value_heads * head_dim;
        let qk_norm = if spec.qk_norm {
            let eps = spec.rms_norm_eps;
            Some((
                RmsNorm::new(vb.pp("q_norm").get(head_dim, "weight")?, eps),
                RmsNorm::new(vb.pp("k_norm").get(head_dim, "weight")?, eps),
            ))
        } else {
            None
        };
        Ok(Self {
            q_proj: Proj::load_with_options(vb, "q_proj", (q_out, h), targets.attn_q, proj_opts)?,
            k_proj: Proj::load_with_options(vb, "k_proj", (kv_out, h), targets.attn_k, proj_opts)?,
            v_proj: Proj::load_with_options(vb, "v_proj", (kv_out, h), targets.attn_v, proj_opts)?,
            o_proj: Proj::load_with_options(vb, "o_proj", (h, q_out), targets.attn_o, proj_opts)?,
            qk_norm,
            num_heads: spec.num_attention_heads,
            num_kv_heads: spec.num_key_value_heads,
            num_kv_groups: spec.num_attention_heads / spec.num_key_value_heads,
            head_dim,
            attn_hidden: q_out,
            sdpa_f32: spec.sdpa_f32,
        })
    }

    fn forward_tensor_parallel(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
    ) -> CandleResult<Tensor> {
        let partial = coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "dense attention payload staging",
            || {
                let (b, l, _) = x.dims3()?;
                let in_dtype = x.dtype();
                let heads = tp_shard(plan, "num_attention_heads", self.num_heads)?.len;
                let kv_heads = tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?.len;
                let attn_hidden = heads * self.head_dim;
                let q = self
                    .q_proj
                    .column_parallel_forward(x, plan, "attention_q_out")?;
                let k = self
                    .k_proj
                    .column_parallel_forward(x, plan, "attention_k_out")?;
                let v = self
                    .v_proj
                    .column_parallel_forward(x, plan, "attention_v_out")?;
                let q = q.reshape((b, l, heads, self.head_dim))?.transpose(1, 2)?;
                let k = k
                    .reshape((b, l, kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let v = v
                    .reshape((b, l, kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let (q, k) = match &self.qk_norm {
                    Some((qn, kn)) => {
                        (qn.forward(&q.contiguous()?)?, kn.forward(&k.contiguous()?)?)
                    }
                    None => (q, k),
                };
                let (cos, sin) = rot.slice(l)?;
                let q = rope_slow(&q.contiguous()?, &cos, &sin)?;
                let k = rope_slow(&k.contiguous()?, &cos, &sin)?;
                let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
                let v = repeat_kv(&v.contiguous()?, self.num_kv_groups)?.contiguous()?;
                let (q, k, v) = if self.sdpa_f32 {
                    (
                        q.to_dtype(DType::F32)?,
                        k.to_dtype(DType::F32)?,
                        v.to_dtype(DType::F32)?,
                    )
                } else {
                    (q, k, v)
                };
                let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
                    / (self.head_dim as f64).sqrt())?;
                if let Some(m) = mask {
                    scores = scores.broadcast_add(m)?;
                }
                let probs = softmax(&scores, D::Minus1)?;
                let ctx = probs.matmul(&v)?;
                let ctx = if self.sdpa_f32 {
                    ctx.to_dtype(in_dtype)?
                } else {
                    ctx
                };
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
        self.v_proj
            .validate_column_parallel_support(plan, "attention_v_out")?;
        self.o_proj
            .validate_row_parallel_support(plan, "attention_hidden")?;
        tp_shard(plan, "num_attention_heads", self.num_heads)?;
        tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?;
        Ok(())
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
/// [`DenseLoraTargets`] recipe. The activation is data (`cfg.hidden_act` for
/// Qwen, fixed `Silu` for Llama).
#[derive(Debug)]
pub(crate) struct DenseMlp {
    pub(crate) gate_proj: Proj,
    pub(crate) up_proj: Proj,
    pub(crate) down_proj: Proj,
    act: Activation,
}

impl DenseMlp {
    fn load(
        spec: &DenseSpec,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        base_quantization: BaseQuantization,
    ) -> CandleResult<Self> {
        let proj_opts = ProjLoadOptions::new(rank, alpha, adapter_dtype, base_quantization);
        let h = spec.hidden_size;
        let i = spec.intermediate_size;
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
            act: spec.activation,
        })
    }

    fn forward_tensor_parallel(
        &self,
        x: &Tensor,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
    ) -> CandleResult<Tensor> {
        let partial =
            coordinate_tensor_parallel_local_call(plan, comm, "dense MLP payload staging", || {
                let lhs = self
                    .gate_proj
                    .column_parallel_forward(x, plan, "intermediate_size")?
                    .apply(&self.act)?;
                let rhs = self
                    .up_proj
                    .column_parallel_forward(x, plan, "intermediate_size")?;
                let hidden = lhs.broadcast_mul(&rhs)?;
                self.down_proj.row_parallel_forward_partial_from_shard(
                    &hidden,
                    plan,
                    "intermediate_size",
                )
            })?;
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

    /// Var order: `gate_proj, up_proj, down_proj` (adapted ones only).
    fn push_vars(&self, out: &mut Vec<Var>) {
        self.gate_proj.push_vars(out);
        self.up_proj.push_vars(out);
        self.down_proj.push_vars(out);
    }
}

/// One decoder layer: pre-norm attention + pre-norm `SwiGLU`, both residual.
#[derive(Debug)]
pub(crate) struct DenseLayer {
    pub(crate) ln1: RmsNorm,
    pub(crate) attn: DenseAttention,
    pub(crate) ln2: RmsNorm,
    pub(crate) mlp: DenseMlp,
}

impl DenseLayer {
    fn load(
        spec: &DenseSpec,
        vb: &VarBuilder,
        targets: DenseLoraTargets,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        base_quantization: BaseQuantization,
    ) -> CandleResult<Self> {
        let eps = spec.rms_norm_eps;
        let h = spec.hidden_size;
        Ok(Self {
            ln1: RmsNorm::new(vb.pp("input_layernorm").get(h, "weight")?, eps),
            attn: DenseAttention::load(
                spec,
                &vb.pp("self_attn"),
                targets,
                rank,
                alpha,
                adapter_dtype,
                base_quantization,
            )?,
            ln2: RmsNorm::new(vb.pp("post_attention_layernorm").get(h, "weight")?, eps),
            mlp: DenseMlp::load(
                spec,
                &vb.pp("mlp"),
                targets,
                rank,
                alpha,
                adapter_dtype,
                base_quantization,
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        self.forward_tensor_parallel(x, mask, rot, TensorParallelPlan::single(), None)
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.attn.validate_tensor_parallel_plan(plan)?;
        self.mlp.validate_tensor_parallel_plan(plan)
    }

    fn forward_tensor_parallel(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
        plan: TensorParallelPlan,
        comm: Option<&dyn Comm>,
    ) -> CandleResult<Tensor> {
        let h = coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "dense attention boundary preparation",
            || self.ln1.forward(x),
        )?;
        let h = self
            .attn
            .forward_tensor_parallel(&h, mask, rot, plan, comm)?;
        let (x, h2) = coordinate_tensor_parallel_local_call(
            plan,
            comm,
            "dense attention boundary completion",
            || {
                let x = x.broadcast_add(&h)?;
                let h2 = self.ln2.forward(&x)?;
                Ok((x, h2))
            },
        )?;
        let h2 = self.mlp.forward_tensor_parallel(&h2, plan, comm)?;
        coordinate_tensor_parallel_local_call(plan, comm, "dense MLP boundary completion", || {
            x.broadcast_add(&h2)
        })
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

/// A grad-bearing, uncached dense-transformer forward with `LoRA` attached per a
/// [`DenseLoraTargets`] recipe — the shared core behind every dense
/// [`GradModel`].
///
/// Generic over a [`DenseArch`] marker `A`, whose `A::Config` typed
/// constructors (`load`, [`load_with_adapter_dtype`](Self::load_with_adapter_dtype),
/// [`load_with_targets`](Self::load_with_targets)) build it from candle
/// safetensors weight-identical to the architecture's shipped forward. The base
/// weights are frozen [`Tensor`]s; only the `LoRA` `A`/`B` factors are trainable
/// [`Var`]s, in a deterministic layer-major order (the positional checkpoint
/// contract).
#[derive(Debug)]
pub struct DenseGradModel<A: DenseArch> {
    embed: Tensor,
    lm_head: Option<Tensor>,
    pub(crate) layers: Vec<DenseLayer>,
    norm: RmsNorm,
    rot: RotaryTables,
    hidden: usize,
    device: Device,
    /// The dtype of the additive causal mask: F32 when the attention force-casts
    /// to F32 (Llama), the model dtype otherwise (Qwen).
    mask_dtype: DType,
    base_quantization: BaseQuantization,
    targets: DenseLoraTargets,
    adapter_enabled: bool,
    remat: bool,
    /// The boundary tape of the most recent checkpointed [`forward`](Self::forward)
    /// (`RefCell`: `forward` takes `&self` by trait contract). One tape per
    /// forward, consumed by exactly one [`backward`](Self::backward); the
    /// interior mutability makes the model `!Sync`.
    pub(crate) tape: RefCell<Option<RematTape>>,
    _arch: PhantomData<A>,
}

impl<A: DenseArch> DenseGradModel<A> {
    /// Load the model from `vb`, attaching a `LoRA` adapter of the given `rank`
    /// and `alpha` with the **historical q/v-only recipe**
    /// ([`DenseLoraTargets::legacy`]) — kept so pre-recipe adapter checkpoints
    /// stay positionally loadable. Use
    /// [`load_with_targets`](Self::load_with_targets) for the industrial recipe.
    ///
    /// `vb` must be over the architecture's safetensors (any dtype; F32 for the
    /// CPU equivalence gate). `cfg` is candle's own config so derived dims match
    /// the shipped model exactly.
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets).
    pub fn load(cfg: &A::Config, vb: &VarBuilder, rank: usize, alpha: f64) -> CandleResult<Self> {
        // The adapter shares the base weights' dtype (the toy / all-F32 case).
        Self::load_with_targets(cfg, vb, rank, alpha, vb.dtype(), DenseLoraTargets::legacy())
    }

    /// Like [`load`](Self::load) (the historical q/v-only recipe), but holds the
    /// trainable `LoRA` adapter in `adapter_dtype`, independent of the (frozen)
    /// base weights' dtype — the **bf16-base / F32-adapter** split (see
    /// [`crate::lora::LoraLinear::with_adapter_dtype`]).
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets).
    pub fn load_with_adapter_dtype(
        cfg: &A::Config,
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

    /// Load the model from `vb`, attaching the `LoRA` adapter per `targets` (see
    /// [`DenseLoraTargets`]; `DenseLoraTargets::default()` is the industrial
    /// every-projection recipe).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `targets` selects nothing (an untrainable
    /// model), if `A::spec` rejects `cfg` (an unsupported option or a degenerate
    /// head configuration — see [`DenseArch::spec`]), if a weight tensor is
    /// missing or mis-shaped, or if the `LoRA` factors cannot be allocated.
    pub fn load_with_targets(
        cfg: &A::Config,
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

    /// Load the model with an explicit frozen-base quantization mode.
    ///
    /// The adapter recipe and trainable-var order are identical to
    /// [`load_with_targets`](Self::load_with_targets); only the frozen base
    /// projection storage changes.
    ///
    /// # Errors
    ///
    /// As [`load_with_targets`](Self::load_with_targets), plus quantization
    /// shape/storage errors.
    pub fn load_with_targets_and_base_quantization(
        cfg: &A::Config,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        targets: DenseLoraTargets,
        base_quantization: BaseQuantization,
    ) -> CandleResult<Self> {
        if !targets.any() {
            candle_core::bail!(
                "{}: DenseLoraTargets selects no projection — the model would have no \
                 trainable parameters",
                A::LABEL
            );
        }
        // The architecture validates `cfg` and distills it (+ the RoPE tables)
        // into a uniform spec; every config guard lives there.
        let spec = A::spec(cfg, vb.dtype(), vb.device())?;
        let h = spec.hidden_size;
        let embed = vb
            .pp("model.embed_tokens")
            .get((spec.vocab_size, h), "weight")?;
        let lm_head = if spec.tie_word_embeddings {
            None
        } else {
            Some(vb.pp("lm_head").get((spec.vocab_size, h), "weight")?)
        };
        let layers_vb = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(spec.num_hidden_layers);
        for i in 0..spec.num_hidden_layers {
            layers.push(DenseLayer::load(
                &spec,
                &layers_vb.pp(i),
                targets,
                rank,
                alpha,
                adapter_dtype,
                base_quantization,
            )?);
        }
        // The mask dtype follows the attention's score dtype: F32 when q/k/v are
        // force-cast (Llama), the model dtype otherwise (Qwen).
        let mask_dtype = if spec.sdpa_f32 {
            DType::F32
        } else {
            vb.dtype()
        };
        Ok(Self {
            norm: RmsNorm::new(vb.pp("model.norm").get(h, "weight")?, spec.rms_norm_eps),
            embed,
            lm_head,
            layers,
            rot: spec.rot,
            hidden: h,
            device: vb.device().clone(),
            mask_dtype,
            base_quantization,
            targets,
            adapter_enabled: true,
            remat: false,
            tape: RefCell::new(None),
            _arch: PhantomData,
        })
    }

    /// The [`DenseLoraTargets`] recipe this model was loaded with (for logging
    /// and checkpoint metadata — see [`DenseLoraTargets::canonical`]).
    #[must_use]
    pub fn lora_targets(&self) -> DenseLoraTargets {
        self.targets
    }

    /// Full-sequence logits `[batch, seq, vocab]` for `input_ids` (`[batch, seq]`,
    /// `u32`) — every position, so the trainer can score whole completions.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails (e.g. a shape mismatch).
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_window(input_ids, None)
    }

    /// The narrowed scoring forward: the full layer walk (attention needs the
    /// whole prefix), with the final norm + head applied to the `(start, len)`
    /// window alone — the full-width logits never materialize (see
    /// [`GradModel::forward_narrowed`]).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence or any tensor op
    /// fails.
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

    /// Full-sequence logits through the tensor-parallel projection/collective
    /// path, using `comm`'s rank/world as the plan.
    ///
    /// This wires real row-output activation reductions for the dense blocks.
    /// Public `ferrl train` uses this path for supported families while weights
    /// remain fully loaded on every rank.
    ///
    /// # Errors
    ///
    /// Returns a candle error if rank/world validation, a collective, or any
    /// tensor op fails. Activation checkpointing on this path is not wired yet
    /// and fails loud.
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
        let plan =
            coordinate_local_candle_call(comm, "dense tensor-parallel forward preflight", || {
                if self.remat {
                    candle_core::bail!(
                        "{}::forward_tensor_parallel: activation checkpointing is not wired for \
                         tensor-parallel execution yet",
                        A::LABEL
                    );
                }
                let plan = plan_from_comm(comm)?;
                self.validate_tensor_parallel_execution_support()?;
                Ok(plan)
            })?;
        let (mut h, mask) = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense tensor-parallel input staging",
            || self.embed_and_mask(input_ids),
        )?;
        for layer in &self.layers {
            h = layer.forward_tensor_parallel(&h, mask.as_ref(), &self.rot, plan, Some(comm))?;
        }
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense tensor-parallel output staging",
            || self.norm_and_head(&h, window),
        )
    }

    fn validate_tensor_parallel_execution_support(&self) -> CandleResult<()> {
        if self.base_quantization == BaseQuantization::Q8_0 {
            candle_core::bail!(
                "{} tensor_parallel execution does not support q8_0 base projections; disable \
                 tensor_parallel for world-one Q8_0 until rank-local quantized shards are \
                 implemented",
                A::LABEL
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

    /// Shared prologue of every full-sequence walk: the token embedding plus the
    /// full causal mask (`None` at seq-len 1). The mask dtype is `self.mask_dtype`
    /// — F32 when the attention scores are F32 (Llama), the model dtype otherwise.
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
            Some(causal_mask(l, self.mask_dtype, &self.device)?)
        };
        Ok((h, mask))
    }

    /// Shared tail of every walk: narrow to the `window` FIRST (the memory lever
    /// — the head must only ever see the window), then the final norm plus the
    /// (possibly tied) `lm_head` projection.
    fn norm_and_head(&self, h: &Tensor, window: Option<(usize, usize)>) -> CandleResult<Tensor> {
        let h = self.norm.forward(&windowed(h, window)?)?;
        match &self.lm_head {
            Some(w) => frozen_linear(&h, w),
            None => frozen_linear(&h, &self.embed),
        }
    }

    /// The checkpointed forward: capture a boundary [`Var`] before each layer
    /// (and one for the tail), so reassigning the hidden state frees each layer's
    /// intermediates as the walk proceeds and the loss tape spans only the tail.
    /// [`backward`](Self::backward) stitches the full gradient from the boundaries.
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
    /// walking the stack with a rolling boundary detach (one-layer peak
    /// footprint, no tape). Same values as `forward`; for the value-only
    /// scorings (`logp_old`, the KL reference).
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
    /// Returns a candle error if the window exceeds the sequence or any tensor op
    /// fails.
    pub fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        self.forward_detached_window(input_ids, Some((start, len)))
    }

    /// The shared detached walk behind [`forward_detached`](Self::forward_detached)
    /// and [`forward_detached_narrowed`](Self::forward_detached_narrowed).
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
        let (plan, mut h, mask) = coordinate_local_candle_call(
            comm,
            "dense detached tensor-parallel input staging",
            || {
                if self.remat {
                    candle_core::bail!(
                        "{}::forward_tensor_parallel_detached: activation checkpointing is not \
                         wired for tensor-parallel execution yet",
                        A::LABEL
                    );
                }
                let plan = plan_from_comm(comm)?;
                self.validate_tensor_parallel_execution_support()?;
                let (h, mask) = self.embed_and_mask(input_ids)?;
                Ok((plan, h.detach(), mask))
            },
        )?;
        for layer in &self.layers {
            let next =
                layer.forward_tensor_parallel(&h, mask.as_ref(), &self.rot, plan, Some(comm))?;
            h = coordinate_tensor_parallel_local_call(
                plan,
                Some(comm),
                "dense detached tensor-parallel layer completion",
                || Ok(next.detach()),
            )?;
        }
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense detached tensor-parallel output staging",
            || Ok(self.norm_and_head(&h, window)?.detach()),
        )
    }

    /// Back-propagate a loss built from this model's logits: plain
    /// `loss.backward()` normally; under
    /// [activation checkpointing](Self::set_activation_checkpointing) it takes
    /// the pending boundary tape and stitches the full gradient by re-running
    /// each layer in reverse (see [`crate::remat`]). Fails loud if no
    /// checkpointed forward is pending, if the loss does not pair with the
    /// pending tape, or if the adapter toggle flipped since the forward.
    ///
    /// # Errors
    ///
    /// Returns a candle error on any backward failure or contract violation above.
    pub fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        if !self.remat {
            return loss.backward();
        }
        let Some(tape) = self.tape.borrow_mut().take() else {
            candle_core::bail!(
                "{}::backward: activation checkpointing is on but no checkpointed forward is \
                 pending (each forward's tape is consumed by exactly one backward)",
                A::LABEL
            )
        };
        if tape.adapter_enabled() != self.adapter_enabled {
            candle_core::bail!(
                "{}::backward: the adapter toggle flipped between the checkpointed forward and \
                 its backward — the recompute would rebuild different values",
                A::LABEL
            )
        }
        let l = tape.first_boundary_dims().map(|d| d[1]).unwrap_or_default();
        let mask = if l <= 1 {
            None
        } else {
            Some(causal_mask(l, self.mask_dtype, &self.device)?)
        };
        stitched_backward(loss, &tape, &self.trainable_vars(), |i, x| {
            self.layers[i].forward(x, mask.as_ref(), &self.rot)
        })
    }

    /// Turn **activation checkpointing** on or off (default: off). On, `forward`
    /// cuts the autograd graph at every layer boundary and `backward` re-runs one
    /// layer at a time — the peak activation footprint drops from the whole stack
    /// to one layer plus the boundary states, for ~one extra forward of recompute
    /// (see [`crate::remat`]). Flipping the mode drops any pending tape.
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
    /// within a layer the attention projections first (`q,k,v,o`), then the MLP's
    /// (`gate,up,down`); each adapted projection contributes `[A, B]`. A pure
    /// function of (config, [`DenseLoraTargets`]) — the positional checkpoint
    /// contract.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        for layer in &self.layers {
            layer.push_vars(&mut vars);
        }
        vars
    }

    /// The device the weights live on, so a caller can build `input_ids` tensors
    /// on the same device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Snapshot the **current** effective weights into a KV-cached, grad-free
    /// [`DenseCachedDecoder`] for fast incremental rollout — folding the live
    /// `LoRA` adapter into every adapted projection (respecting the adapter
    /// toggle), cloning the frozen rest. **Rebuild after every optimizer step**
    /// (and after any `set_adapter_enabled` flip): the returned decoder is a
    /// tape-detached value snapshot.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any `merged_weight` build fails.
    pub fn merged_decoder(&self) -> CandleResult<DenseCachedDecoder> {
        DenseCachedDecoder::from_model(self)
    }
}

/// The [`GradModel`] seam over [`DenseGradModel`]: pure delegation to the
/// inherent methods above (which stay public — the trait adds a generic surface,
/// it does not replace the concrete one).
impl<A: DenseArch> GradModel for DenseGradModel<A> {
    type Decoder = DenseCachedDecoder;

    fn device(&self) -> &Device {
        DenseGradModel::device(self)
    }

    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        DenseGradModel::forward(self, input_ids)
    }

    fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        DenseGradModel::forward_detached(self, input_ids)
    }

    fn forward_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        DenseGradModel::forward_narrowed(self, input_ids, start, len)
    }

    fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        DenseGradModel::forward_detached_narrowed(self, input_ids, start, len)
    }

    fn validate_tensor_parallel_execution(&self, comm: &dyn Comm) -> CandleResult<()> {
        let plan = plan_from_comm(comm)?;
        if self.remat {
            candle_core::bail!(
                "{} tensor-parallel execution does not support activation checkpointing",
                A::LABEL
            );
        }
        self.validate_tensor_parallel_execution_support()?;
        self.validate_tensor_parallel_plan(plan)
    }

    fn forward_tensor_parallel(&self, input_ids: &Tensor, comm: &dyn Comm) -> CandleResult<Tensor> {
        DenseGradModel::forward_tensor_parallel(self, input_ids, comm)
    }

    fn forward_tensor_parallel_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        DenseGradModel::forward_tensor_parallel_narrowed(self, input_ids, start, len, comm)
    }

    fn forward_tensor_parallel_detached(
        &self,
        input_ids: &Tensor,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        DenseGradModel::forward_tensor_parallel_detached(self, input_ids, comm)
    }

    fn forward_tensor_parallel_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        DenseGradModel::forward_tensor_parallel_detached_narrowed(self, input_ids, start, len, comm)
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        DenseGradModel::backward(self, loss)
    }

    fn trainable_vars(&self) -> Vec<Var> {
        DenseGradModel::trainable_vars(self)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        DenseGradModel::set_adapter_enabled(self, enabled);
    }

    fn merged_decoder(&self) -> CandleResult<DenseCachedDecoder> {
        DenseGradModel::merged_decoder(self)
    }

    fn lora_recipe(&self) -> Option<String> {
        Some(self.targets.canonical())
    }
}

/// One dense attention block over **merged** weights with an incremental KV
/// cache — the grad-free mirror of [`DenseAttention`]. Every projection uses its
/// single effective weight (the folded adapter when adapted, the frozen base
/// otherwise; all bias-free); the un-repeated K/V are appended to a
/// [`ConcatKvCache`] before `repeat_kv` (the shipped op order). The same
/// `Option<(q_norm, k_norm)>` + `sdpa_f32` knobs drive it as the uncached block,
/// so cached logits equal the uncached ones.
#[derive(Debug)]
struct DenseMergedAttention {
    q_weight: FrozenLinearSnapshot,
    k_weight: FrozenLinearSnapshot,
    v_weight: FrozenLinearSnapshot,
    o_weight: FrozenLinearSnapshot,
    qk_norm: Option<(RmsNorm, RmsNorm)>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_hidden: usize,
    sdpa_f32: bool,
    /// Un-repeated K/V cache (`[b, kv_heads, seq, head_dim]`), concatenated on the
    /// sequence axis (dim 2). `repeat_kv` is applied to the cache's output, never
    /// to what is stored.
    cache: ConcatKvCache,
}

impl DenseMergedAttention {
    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.q_weight
            .validate_tensor_parallel_support(plan, "attention_q_out")?;
        self.k_weight
            .validate_tensor_parallel_support(plan, "attention_k_out")?;
        self.v_weight
            .validate_tensor_parallel_support(plan, "attention_v_out")?;
        self.o_weight
            .validate_tensor_parallel_support(plan, "attention_hidden")?;
        tp_shard(plan, "num_attention_heads", self.num_heads)?;
        tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?;
        tp_shard(plan, "attention_q_out", self.q_weight.dims2()?.0)?;
        tp_shard(plan, "attention_k_out", self.k_weight.dims2()?.0)?;
        tp_shard(plan, "attention_v_out", self.v_weight.dims2()?.0)?;
        tp_shard(plan, "attention_hidden", self.o_weight.dims2()?.1)?;
        Ok(())
    }

    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let in_dtype = x.dtype();

        // 1. Projections over merged weights.
        let q = self.q_weight.forward(x)?;
        let k = self.k_weight.forward(x)?;
        let v = self.v_weight.forward(x)?;

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

        // 3. Per-head QK-Norm BEFORE RoPE — Qwen only.
        let (q, k) = match &self.qk_norm {
            Some((qn, kn)) => (qn.forward(&q.contiguous()?)?, kn.forward(&k.contiguous()?)?),
            None => (q, k),
        };

        // 4. RoPE at the absolute position `offset`.
        let (cos, sin) = rot.slice_at(offset, l)?;
        let q = rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k = rope_slow(&k.contiguous()?, &cos, &sin)?;

        // 5. Append the UN-repeated K/V, then GQA-repeat the full cached K/V —
        //    repeat AFTER append (the shipped order) so the cache stays compact.
        let (k, v) = self.cache.append(&k.contiguous()?, &v.contiguous()?)?;
        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

        // 6. Optional F32 force-cast, then scaled dot-product attention with the
        //    grad-safe softmax (scaling is a division by sqrt(head_dim)).
        let (q, k, v) = if self.sdpa_f32 {
            (
                q.to_dtype(DType::F32)?,
                k.to_dtype(DType::F32)?,
                v.to_dtype(DType::F32)?,
            )
        } else {
            (q, k, v)
        };
        let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
            / (self.head_dim as f64).sqrt())?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?;
        let ctx = if self.sdpa_f32 {
            ctx.to_dtype(in_dtype)?
        } else {
            ctx
        };

        // 7. Output projection.
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
        rot: &RotaryTables,
        plan: TensorParallelPlan,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let partial = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense cached attention payload staging",
            || {
                let (b, l, _) = x.dims3()?;
                let in_dtype = x.dtype();
                let heads = tp_shard(plan, "num_attention_heads", self.num_heads)?.len;
                let kv_heads = tp_shard(plan, "num_key_value_heads", self.num_kv_heads)?.len;
                let attn_hidden = heads * self.head_dim;
                let q = self
                    .q_weight
                    .column_parallel_forward(x, plan, "attention_q_out")?;
                let k = self
                    .k_weight
                    .column_parallel_forward(x, plan, "attention_k_out")?;
                let v = self
                    .v_weight
                    .column_parallel_forward(x, plan, "attention_v_out")?;
                let q = q.reshape((b, l, heads, self.head_dim))?.transpose(1, 2)?;
                let k = k
                    .reshape((b, l, kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let v = v
                    .reshape((b, l, kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let (q, k) = match &self.qk_norm {
                    Some((qn, kn)) => {
                        (qn.forward(&q.contiguous()?)?, kn.forward(&k.contiguous()?)?)
                    }
                    None => (q, k),
                };
                let (cos, sin) = rot.slice_at(offset, l)?;
                let q = rope_slow(&q.contiguous()?, &cos, &sin)?;
                let k = rope_slow(&k.contiguous()?, &cos, &sin)?;
                let (k, v) = self.cache.append(&k.contiguous()?, &v.contiguous()?)?;
                let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
                let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;
                let (q, k, v) = if self.sdpa_f32 {
                    (
                        q.to_dtype(DType::F32)?,
                        k.to_dtype(DType::F32)?,
                        v.to_dtype(DType::F32)?,
                    )
                } else {
                    (q, k, v)
                };
                let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
                    / (self.head_dim as f64).sqrt())?;
                if let Some(m) = mask {
                    scores = scores.broadcast_add(m)?;
                }
                let probs = softmax(&scores, D::Minus1)?;
                let ctx = probs.matmul(&v)?;
                let ctx = if self.sdpa_f32 {
                    ctx.to_dtype(in_dtype)?
                } else {
                    ctx
                };
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
}

/// `SwiGLU` MLP over merged weights — the grad-free mirror of [`DenseMlp`].
#[derive(Debug)]
struct DenseMergedMlp {
    gate_weight: FrozenLinearSnapshot,
    up_weight: FrozenLinearSnapshot,
    down_weight: FrozenLinearSnapshot,
    act: Activation,
}

impl DenseMergedMlp {
    fn from_mlp(mlp: &DenseMlp) -> CandleResult<Self> {
        Ok(Self {
            gate_weight: mlp.gate_proj.snapshot()?,
            up_weight: mlp.up_proj.snapshot()?,
            down_weight: mlp.down_proj.snapshot()?,
            act: mlp.act,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let lhs = self.gate_weight.forward(x)?.apply(&self.act)?;
        let rhs = self.up_weight.forward(x)?;
        self.down_weight.forward(&lhs.broadcast_mul(&rhs)?)
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.gate_weight
            .validate_tensor_parallel_support(plan, "intermediate_size")?;
        self.up_weight
            .validate_tensor_parallel_support(plan, "intermediate_size")?;
        self.down_weight
            .validate_tensor_parallel_support(plan, "intermediate_size")?;
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
            "dense cached MLP payload staging",
            || {
                let lhs = self
                    .gate_weight
                    .column_parallel_forward(x, plan, "intermediate_size")?
                    .apply(&self.act)?;
                let rhs = self
                    .up_weight
                    .column_parallel_forward(x, plan, "intermediate_size")?;
                let hidden = lhs.broadcast_mul(&rhs)?;
                self.down_weight.row_parallel_forward_partial_from_shard(
                    &hidden,
                    plan,
                    "intermediate_size",
                )
            },
        )?;
        reduce_row_parallel_output(&partial, plan, Some(comm))
    }
}

/// One decoder layer over merged weights — the grad-free mirror of [`DenseLayer`].
#[derive(Debug)]
struct DenseMergedLayer {
    ln1: RmsNorm,
    attn: DenseMergedAttention,
    ln2: RmsNorm,
    mlp: DenseMergedMlp,
}

impl DenseMergedLayer {
    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        self.attn.validate_tensor_parallel_plan(plan)?;
        self.mlp.validate_tensor_parallel_plan(plan)
    }

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

    fn forward_tensor_parallel(
        &mut self,
        x: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
        rot: &RotaryTables,
        plan: TensorParallelPlan,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let h = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense cached attention boundary preparation",
            || self.ln1.forward(x),
        )?;
        let h = self
            .attn
            .forward_tensor_parallel(&h, offset, mask, rot, plan, comm)?;
        let (x, h2) = coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense cached attention boundary completion",
            || {
                let x = x.broadcast_add(&h)?;
                let h2 = self.ln2.forward(&x)?;
                Ok((x, h2))
            },
        )?;
        let h2 = self.mlp.forward_tensor_parallel(&h2, plan, comm)?;
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense cached MLP boundary completion",
            || x.broadcast_add(&h2),
        )
    }
}

/// A KV-cached, **grad-free** dense-transformer decoder over weights with the
/// `LoRA` adapter already folded in — the fast rollout twin of
/// [`DenseGradModel`], and its [`CachedDecoder`].
///
/// Built by [`DenseGradModel::merged_decoder`], which snapshots the live merged
/// weights (capturing whatever the adapter is at build time, toggle included).
/// [`forward`](Self::forward) consumes one chunk of new tokens at a time,
/// advancing a per-layer [`ConcatKvCache`], so generating `L` tokens costs O(L)
/// attention work instead of the uncached forward's O(L²). It holds **no**
/// [`Var`] and records no autograd tape.
///
/// # Cache lifecycle
///
/// The cache grows with each [`forward`](Self::forward); positions are placed at
/// the `offset` you pass (which must equal the number of tokens already
/// consumed). Call [`reset_cache`](Self::reset_cache) to reuse one decoder for a
/// fresh sequence. Because the cache is mutable state, `forward` takes `&mut self`.
#[derive(Debug)]
pub struct DenseCachedDecoder {
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<DenseMergedLayer>,
    norm: RmsNorm,
    rot: RotaryTables,
    hidden: usize,
    device: Device,
    mask_dtype: DType,
    base_quantization: BaseQuantization,
}

impl DenseCachedDecoder {
    /// Snapshot a [`DenseGradModel`]'s current effective weights. Private —
    /// callers go through [`DenseGradModel::merged_decoder`].
    fn from_model<A: DenseArch>(model: &DenseGradModel<A>) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let a = &layer.attn;
            layers.push(DenseMergedLayer {
                ln1: layer.ln1.clone(),
                attn: DenseMergedAttention {
                    q_weight: a.q_proj.snapshot()?,
                    k_weight: a.k_proj.snapshot()?,
                    v_weight: a.v_proj.snapshot()?,
                    o_weight: a.o_proj.snapshot()?,
                    qk_norm: a.qk_norm.clone(),
                    num_heads: a.num_heads,
                    num_kv_heads: a.num_kv_heads,
                    num_kv_groups: a.num_kv_groups,
                    head_dim: a.head_dim,
                    attn_hidden: a.attn_hidden,
                    sdpa_f32: a.sdpa_f32,
                    cache: ConcatKvCache::new(2),
                },
                ln2: layer.ln2.clone(),
                mlp: DenseMergedMlp::from_mlp(&layer.mlp)?,
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
            mask_dtype: model.mask_dtype,
            base_quantization: model.base_quantization,
        })
    }

    /// Logits `[batch, chunk_len, vocab]` for `input_ids` (`[batch, chunk_len]`,
    /// `u32`) placed at absolute positions `[offset, offset + chunk_len)`,
    /// appending to the KV cache.
    ///
    /// Pass the whole prompt at `offset == 0` to prefill, then one token at a
    /// time at the running offset to decode. `offset` **must** equal the number
    /// of tokens already in the cache (it indexes the `RoPE` tables and sizes the
    /// causal mask); a mismatch is rejected (see Errors) rather than silently
    /// producing wrong logits. Every position is returned.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `offset` does not equal the cached sequence
    /// length, if any tensor op fails, or if `offset + chunk_len` exceeds the
    /// `RoPE` table's `max_position_embeddings`.
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let (b, l) = input_ids.dims2()?;
        // The caller's `offset` must equal the number of tokens already cached:
        // the l == 1 decode path builds no mask, so a desync would silently
        // corrupt the logits rather than trip a shape check. All layer caches
        // advance in lockstep, so layer 0 is the truth.
        let cached = self
            .layers
            .first()
            .map_or(0, |layer| layer.attn.cache.current_seq_len());
        if offset != cached {
            candle_core::bail!(
                "DenseCachedDecoder::forward: offset {offset} != cached sequence length \
                 {cached} (pass offset == tokens already decoded; 0 to prefill)"
            );
        }
        let ids = input_ids.flatten_all()?;
        let mut h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.hidden))?;
        // A single new token attends to the whole cache (all past keys are
        // causally valid), matching the uncached `l == 1` branch and the shipped
        // model.
        let mask = if l == 1 {
            None
        } else {
            Some(causal_mask_at(offset, l, self.mask_dtype, &self.device)?)
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

    /// Logits through the tensor-parallel cached projection/collective path,
    /// driven by the explicit communicator.
    ///
    /// # Errors
    ///
    /// As [`forward`](Self::forward), plus any rank/world validation or
    /// collective failure.
    pub fn forward_tensor_parallel(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let (plan, mut h, mask) = coordinate_local_candle_call(
            comm,
            "dense cached tensor-parallel input staging",
            || {
                let plan = plan_from_comm(comm)?;
                self.validate_tensor_parallel_execution_support()?;
                let (b, l) = input_ids.dims2()?;
                let cached = self
                    .layers
                    .first()
                    .map_or(0, |layer| layer.attn.cache.current_seq_len());
                if offset != cached {
                    candle_core::bail!(
                        "DenseCachedDecoder::forward_tensor_parallel: offset {offset} != cached \
                         sequence length {cached} (pass offset == tokens already decoded; 0 to \
                         prefill)"
                    );
                }
                self.validate_tensor_parallel_plan(plan)?;
                let ids = input_ids.flatten_all()?;
                let h = self
                    .embed
                    .index_select(&ids, 0)?
                    .reshape((b, l, self.hidden))?;
                let mask = if l == 1 {
                    None
                } else {
                    Some(causal_mask_at(offset, l, self.mask_dtype, &self.device)?)
                };
                Ok((plan, h, mask))
            },
        )?;
        for layer in &mut self.layers {
            h = layer.forward_tensor_parallel(&h, offset, mask.as_ref(), &self.rot, plan, comm)?;
        }
        coordinate_tensor_parallel_local_call(
            plan,
            Some(comm),
            "dense cached tensor-parallel output staging",
            || {
                let h = self.norm.forward(&h)?;
                match &self.lm_head {
                    Some(w) => frozen_linear(&h, w),
                    None => frozen_linear(&h, &self.embed),
                }
            },
        )
    }

    fn validate_tensor_parallel_plan(&self, plan: TensorParallelPlan) -> CandleResult<()> {
        for layer in &self.layers {
            layer.validate_tensor_parallel_plan(plan)?;
        }
        Ok(())
    }

    fn validate_tensor_parallel_execution_support(&self) -> CandleResult<()> {
        if self.base_quantization == BaseQuantization::Q8_0 {
            candle_core::bail!(
                "DenseCachedDecoder tensor_parallel execution does not support q8_0 base \
                 projections; disable tensor_parallel for world-one Q8_0 until rank-local \
                 quantized shards are implemented"
            );
        }
        Ok(())
    }

    /// Clear every layer's KV cache so the decoder can start a fresh sequence
    /// (next [`forward`](Self::forward) must use `offset == 0`).
    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.attn.cache.reset();
        }
    }
}

/// The [`CachedDecoder`] seam over [`DenseCachedDecoder`]: pure delegation to the
/// inherent methods above (which carry the offset fail-loud guard and the
/// cache-lifecycle contract the trait requires).
impl CachedDecoder for DenseCachedDecoder {
    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor> {
        DenseCachedDecoder::forward(self, input_ids, offset)
    }

    fn forward_tensor_parallel(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        DenseCachedDecoder::forward_tensor_parallel(self, input_ids, offset, comm)
    }

    fn reset_cache(&mut self) {
        DenseCachedDecoder::reset_cache(self);
    }
}
