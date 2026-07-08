//! A manual `LoRA` (low-rank adaptation) linear layer.
//!
//! [`LoraLinear`] wraps a **frozen** base `candle_nn::Linear`-style weight (and
//! optional bias) and adds a trainable low-rank update `B @ A` scaled by
//! `alpha / rank`. Only the two factors `A` and `B` are [`candle_core::Var`]s
//! (i.e. on the autograd tape); the base weight and bias are plain frozen
//! [`Tensor`]s. This is the entire set of trainable parameters GRPO optimizes,
//! which is exactly what the grad-coverage canary asserts over.
//!
//! Forward (for input `x` of shape `[.., in]`):
//!
//! ```text
//! y = x @ Wᵀ (+ b)              // frozen base
//!   + enabled ? (alpha/rank) · (x @ Aᵀ) @ Bᵀ : 0   // trainable update
//! ```
//!
//! with `A` shaped `[rank, in]` and `B` shaped `[out, rank]`. Conventionally `A`
//! is initialized small/random and `B` to zero, so the adapter starts as a no-op
//! and training departs smoothly from the base model.
//!
//! Disabling the adapter ([`LoraLinear::set_enabled`]) drops the update term,
//! yielding the frozen base distribution — the GRPO reference policy.
//!
//! For the cached rollout path, [`LoraLinear::merged_weight`] folds the live
//! adapter into one frozen, tape-detached weight `W + (alpha/rank) · B @ A`
//! (or just `W` when disabled), so a KV-cached decoder can apply the adapter
//! as a single plain matmul per step. The grad/scoring path never uses it.
//!
//! Which projections of a model carry the adapter is a **recipe**:
//! [`DenseLoraTargets`] for the dense (attention + `SwiGLU`-MLP) models
//! ([`crate::qwen`], [`crate::llama`]); the hybrid `qwen3_5` family has its own
//! extended recipe ([`crate::qwen35::LoraTargets`]) with the `GatedDeltaNet`
//! opt-ins. The recipe (together with the config) determines the trainable-var
//! order — the positional checkpoint contract.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
use candle_nn::VarBuilder;

use crate::blocks::frozen_linear;

/// Optional quantization for frozen base projection weights.
///
/// The trainable adapter remains ordinary `Var` storage. This knob only changes
/// how the frozen base weight is stored and applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaseQuantization {
    /// Store the frozen base as an ordinary candle tensor.
    #[default]
    None,
    /// Store the frozen base as GGML `Q8_0`.
    Q8_0,
}

impl BaseQuantization {
    /// Stable config/report spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Q8_0 => "q8_0",
        }
    }

    fn ggml_dtype(self) -> Option<GgmlDType> {
        match self {
            Self::None => None,
            Self::Q8_0 => Some(GgmlDType::Q8_0),
        }
    }
}

/// Shared projection-load knobs that are independent of a projection's name,
/// shape, and adaptation flag.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProjLoadOptions {
    rank: usize,
    alpha: f64,
    adapter_dtype: DType,
    base_quantization: BaseQuantization,
}

impl ProjLoadOptions {
    pub(crate) fn new(
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        base_quantization: BaseQuantization,
    ) -> Self {
        Self {
            rank,
            alpha,
            adapter_dtype,
            base_quantization,
        }
    }
}

/// A frozen linear weight, optionally stored in quantized form.
#[derive(Debug, Clone)]
pub(crate) enum FrozenLinearWeight {
    Dense(Tensor),
    Quantized {
        q: Arc<QTensor>,
        qmatmul: QMatMul,
        dtype: DType,
        device: Device,
        shape: (usize, usize),
    },
}

impl FrozenLinearWeight {
    pub(crate) fn from_tensor(
        weight: Tensor,
        quantization: BaseQuantization,
    ) -> CandleResult<Self> {
        let shape = weight.dims2()?;
        let dtype = weight.dtype();
        let device = weight.device().clone();
        let Some(qdtype) = quantization.ggml_dtype() else {
            return Ok(Self::Dense(weight));
        };
        if !shape.1.is_multiple_of(qdtype.block_size()) {
            candle_core::bail!(
                "base quantization {} requires projection input width to be divisible by \
                 block size {}, got {:?}",
                quantization.as_str(),
                qdtype.block_size(),
                shape
            );
        }
        let q = Arc::new(QTensor::quantize(&weight, qdtype)?);
        let qmatmul = QMatMul::from_arc(q.clone())?;
        Ok(Self::Quantized {
            q,
            qmatmul,
            dtype,
            device,
            shape,
        })
    }

    pub(crate) fn dims2(&self) -> CandleResult<(usize, usize)> {
        match self {
            Self::Dense(w) => w.dims2(),
            Self::Quantized { shape, .. } => Ok(*shape),
        }
    }

    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Dense(w) => w.dtype(),
            Self::Quantized { dtype, .. } => *dtype,
        }
    }

    pub(crate) fn device(&self) -> &Device {
        match self {
            Self::Dense(w) => w.device(),
            Self::Quantized { device, .. } => device,
        }
    }

    fn dequantized_to(&self, dtype: DType) -> CandleResult<Tensor> {
        match self {
            Self::Dense(w) => w.to_dtype(dtype),
            Self::Quantized { q, device, .. } => q.dequantize(device)?.to_dtype(dtype),
        }
    }

    pub(crate) fn dequantized(&self) -> CandleResult<Tensor> {
        self.dequantized_to(self.dtype())
    }

    /// Tape-bearing frozen linear: gradients still flow to `x`.
    pub(crate) fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(w) => frozen_linear(x, w),
            Self::Quantized { .. } => {
                let w = self.dequantized_to(x.dtype())?;
                frozen_linear(x, &w)
            }
        }
    }

    /// Tape-free rollout linear. Quantized weights use candle's quantized matmul.
    pub(crate) fn forward_tape_free(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(w) => frozen_linear(x, w),
            Self::Quantized { qmatmul, .. } => {
                let input_dtype = x.dtype();
                let compute_dtype = match input_dtype {
                    DType::F32 | DType::F16 => input_dtype,
                    _ => DType::F32,
                };
                let input = if input_dtype == compute_dtype {
                    x.clone()
                } else {
                    x.to_dtype(compute_dtype)?
                };
                let y = input.apply(qmatmul)?;
                if y.dtype() == input_dtype {
                    Ok(y)
                } else {
                    y.to_dtype(input_dtype)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum FrozenLinearSnapshot {
    Dense(Tensor),
    Base(FrozenLinearWeight),
    Lora {
        base: FrozenLinearWeight,
        base_bias: Option<Tensor>,
        a: Tensor,
        b: Tensor,
        scale: f64,
        enabled: bool,
    },
}

impl FrozenLinearSnapshot {
    pub(crate) fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(w) => frozen_linear(x, w),
            Self::Base(w) => w.forward_tape_free(x),
            Self::Lora {
                base,
                base_bias,
                a,
                b,
                scale,
                enabled,
            } => {
                let base = base.forward_tape_free(x)?;
                let base = match base_bias {
                    Some(bias) => base.broadcast_add(bias)?,
                    None => base,
                };
                if !enabled {
                    return Ok(base);
                }
                let dtype = base.dtype();
                let a = a.to_dtype(dtype)?;
                let b = b.to_dtype(dtype)?;
                let xa = x.broadcast_matmul(&a.t()?)?;
                let xab = xa.broadcast_matmul(&b.t()?)?;
                base.broadcast_add(&(xab * *scale)?)
            }
        }
    }
}

/// Which projections of a **dense** (attention + `SwiGLU`-MLP) transformer
/// carry the `LoRA` adapter — the recipe for [`crate::qwen::QwenGradModel`]
/// and [`crate::llama::LlamaGradModel`].
///
/// The default is the **industrial** recipe (every attention and MLP
/// projection — what TRL/verl/ms-swift configure for RL fine-tuning in 2026;
/// see [`industrial`](Self::industrial)). The **historical** ferrl recipe is
/// q/v-only ([`legacy`](Self::legacy)) — the dense models' `load()` /
/// `load_with_adapter_dtype()` constructors keep it so every pre-existing
/// adapter checkpoint stays positionally loadable (see the type docs'
/// var-order note below).
///
/// Norm weights, the embedding, and the (tied or untied) `lm_head` are never
/// adapted — no framework's `LoRA` recipe targets them by default (frameworks
/// that train them at all do it full-rank, peft-`modules_to_save`-style, a
/// different mechanism than an adapter).
///
/// The recipe (together with the config) **determines the trainable-var
/// order** — layer-major, fixed projection order within each layer
/// (`q,k,v,o,gate,up,down`, adapted ones only, each contributing `[A, B]`) —
/// which is the positional checkpoint contract;
/// [`canonical`](Self::canonical) is the stable string form for recording it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct DenseLoraTargets {
    /// Attention `q_proj`.
    pub attn_q: bool,
    /// Attention `k_proj`.
    pub attn_k: bool,
    /// Attention `v_proj`.
    pub attn_v: bool,
    /// Attention `o_proj`.
    pub attn_o: bool,
    /// MLP `gate_proj`.
    pub mlp_gate: bool,
    /// MLP `up_proj`.
    pub mlp_up: bool,
    /// MLP `down_proj`.
    pub mlp_down: bool,
}

impl Default for DenseLoraTargets {
    fn default() -> Self {
        Self::industrial()
    }
}

impl DenseLoraTargets {
    /// The industrial default recipe: every attention and MLP projection.
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
        }
    }

    /// The historical ferrl recipe: `q_proj`/`v_proj` only. This is what the
    /// dense models hard-wired before the recipe existed, and what their
    /// `load()` constructors still use — pre-recipe adapter checkpoints
    /// (canonical `attn:qv|mlp:-`) restore positionally under it.
    #[must_use]
    pub fn legacy() -> Self {
        Self {
            attn_q: true,
            attn_k: false,
            attn_v: true,
            attn_o: false,
            mlp_gate: false,
            mlp_up: false,
            mlp_down: false,
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
    }

    /// A stable, human-readable encoding of the recipe (for logs and
    /// checkpoint metadata): e.g. the industrial default is
    /// `attn:qkvo|mlp:gud`, the legacy recipe `attn:qv|mlp:-`.
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
            "attn:{}|mlp:{}",
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
        )
    }
}

/// A bias-free linear projection that either stays frozen or carries the
/// `LoRA` adapter, per a recipe flag — the shared building block every
/// recipe-driven model loader is assembled from.
#[derive(Debug)]
pub(crate) enum Proj {
    /// The frozen base weight, `[out, in]`.
    Frozen(FrozenLinearWeight),
    /// The same weight wrapped with a trainable adapter.
    Lora(LoraLinear),
}

impl Proj {
    /// Load `<name>.weight` of `shape` from `vb`, adapted iff `adapted`.
    pub(crate) fn load(
        vb: &VarBuilder,
        name: &str,
        shape: (usize, usize),
        adapted: bool,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        Self::load_with_options(
            vb,
            name,
            shape,
            adapted,
            ProjLoadOptions::new(rank, alpha, adapter_dtype, BaseQuantization::None),
        )
    }

    pub(crate) fn load_with_options(
        vb: &VarBuilder,
        name: &str,
        shape: (usize, usize),
        adapted: bool,
        opts: ProjLoadOptions,
    ) -> CandleResult<Self> {
        let w = vb.pp(name).get(shape, "weight")?;
        let w = FrozenLinearWeight::from_tensor(w, opts.base_quantization)?;
        if adapted {
            Ok(Self::Lora(LoraLinear::from_frozen_base(
                w,
                None,
                opts.rank,
                opts.alpha,
                opts.adapter_dtype,
            )?))
        } else {
            Ok(Self::Frozen(w))
        }
    }

    /// `y = x Wᵀ` (plus the adapter side-path when adapted and enabled).
    pub(crate) fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Frozen(w) => w.forward(x),
            Self::Lora(l) => l.forward(x),
        }
    }

    /// The single effective weight of the current forward (see
    /// [`LoraLinear::merged_weight`]); a frozen projection is its own merge.
    ///
    /// Fails loud on a biased adapted projection: [`Proj::load`] always builds
    /// bias-free, but `LoraLinear` can carry a bias — and every merged decoder
    /// applies NONE, so a future biased construction path would silently make
    /// the cached rollout diverge from the biased uncached forward.
    pub(crate) fn merged_weight(&self) -> CandleResult<Tensor> {
        match self {
            Self::Frozen(w) => Ok(w.dequantized()?.detach()),
            Self::Lora(l) => {
                if l.base_bias().is_some() {
                    candle_core::bail!(
                        "Proj::merged_weight: this projection carries a base bias, but the \
                         merged snapshot is bias-free (extend the merged decoder to apply \
                         base_bias() before using a biased adapter here)"
                    );
                }
                l.merged_weight()
            }
        }
    }

    /// [`merged_weight`](Self::merged_weight) with a **fresh-storage**
    /// guarantee — the full-fine-tuning merged-decoder variant. The ordinary
    /// `Frozen` merge arm returns a `detach()`, which SHARES storage with the
    /// base weight; when that weight is a trainable var's inner tensor
    /// (full-FT), a merged decoder built from the share would silently track
    /// optimizer updates instead of snapshotting a value. The `Lora` arm
    /// already materializes fresh storage (base + scale·B·A).
    ///
    /// # Errors
    ///
    /// As [`merged_weight`](Self::merged_weight), plus a copy failure.
    pub(crate) fn merged_weight_deep(&self) -> CandleResult<Tensor> {
        match self {
            Self::Frozen(w) => Ok(w.dequantized()?.copy()?.detach()),
            Self::Lora(_) => self.merged_weight(),
        }
    }

    pub(crate) fn snapshot(&self) -> CandleResult<FrozenLinearSnapshot> {
        match self {
            Self::Frozen(w) => Ok(FrozenLinearSnapshot::Base(w.clone())),
            Self::Lora(l) => l.snapshot(),
        }
    }

    /// Append this projection's trainable [`Var`]s (`[A, B]` when adapted,
    /// nothing when frozen) — the positional-order building block.
    pub(crate) fn push_vars(&self, out: &mut Vec<Var>) {
        if let Self::Lora(l) = self {
            out.extend(l.trainable_vars());
        }
    }

    /// Forward the adapter toggle (a no-op on a frozen projection).
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        if let Self::Lora(l) = self {
            l.set_enabled(enabled);
        }
    }
}

/// A linear layer with a frozen base weight and a trainable low-rank adapter.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Frozen base weight, shape `[out, in]`.
    base_weight: FrozenLinearWeight,
    /// Frozen optional bias, shape `[out]`.
    base_bias: Option<Tensor>,
    /// Trainable low-rank factor `A`, shape `[rank, in]`.
    a: Var,
    /// Trainable low-rank factor `B`, shape `[out, rank]`.
    b: Var,
    /// Scaling applied to the low-rank update: `alpha / rank`.
    scale: f64,
    /// Whether the adapter currently contributes to the forward pass.
    enabled: bool,
}

impl LoraLinear {
    /// Build a `LoRA` layer with adapter factors in the **base weight's** dtype.
    ///
    /// See [`with_adapter_dtype`](Self::with_adapter_dtype) for the details; this
    /// is the common case (toy / all-F32) where the adapter and base share a dtype.
    ///
    /// # Errors
    ///
    /// As [`with_adapter_dtype`](Self::with_adapter_dtype).
    pub fn new(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
    ) -> CandleResult<Self> {
        let dtype = base_weight.dtype();
        Self::with_adapter_dtype(base_weight, base_bias, rank, alpha, dtype)
    }

    /// Build a `LoRA` layer whose **trainable adapter** (`A`/`B`) is held in
    /// `adapter_dtype`, independent of the frozen base weight's dtype.
    ///
    /// `base_weight` must be `[out, in]`; `base_bias`, if present, must be `[out]`.
    /// `A` is sampled `N(0, 0.02)` and `B` is zero-initialized, so the adapter
    /// starts as an identity (no-op) on top of the base model. The update is scaled
    /// by `alpha / rank`.
    ///
    /// # The dtype split (bf16 base / F32 adapter)
    ///
    /// For a real model the frozen base is loaded in BF16 (halving weight **and**
    /// activation memory), but a BF16 *adapter* would lose the GRPO update: a tiny
    /// `lr · grad` step rounds away in BF16's ~3 significant digits — the precision
    /// collapse P3 flagged. Keeping `A`/`B` (and therefore their gradients and the
    /// optimizer's moment estimates) in F32 preserves the update; the [`forward`](Self::forward)
    /// casts them down to the activation dtype only for the matmul, so the big
    /// activations stay BF16. This is standard mixed-precision: an F32 master copy,
    /// a BF16 compute path.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `rank == 0` or if the adapter factors cannot be
    /// allocated on the base weight's device.
    pub fn with_adapter_dtype(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        Self::with_adapter_dtype_and_base_quantization(
            base_weight,
            base_bias,
            rank,
            alpha,
            adapter_dtype,
            BaseQuantization::None,
        )
    }

    pub(crate) fn with_adapter_dtype_and_base_quantization(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
        base_quantization: BaseQuantization,
    ) -> CandleResult<Self> {
        let base_weight = FrozenLinearWeight::from_tensor(base_weight, base_quantization)?;
        Self::from_frozen_base(base_weight, base_bias, rank, alpha, adapter_dtype)
    }

    fn from_frozen_base(
        base_weight: FrozenLinearWeight,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
        adapter_dtype: DType,
    ) -> CandleResult<Self> {
        let (out, in_) = base_weight.dims2()?;
        if rank == 0 {
            return Err(candle_core::Error::Msg("lora rank must be > 0".into()));
        }
        let device = base_weight.device();
        let a = Var::randn(0.0, 0.02, (rank, in_), device)?.to_dtype(adapter_dtype)?;
        let a = Var::from_tensor(&a)?;
        let b = Var::zeros((out, rank), adapter_dtype, device)?;
        let scale = alpha / rank as f64;
        Ok(Self {
            base_weight,
            base_bias,
            a,
            b,
            scale,
            enabled: true,
        })
    }

    /// The trainable adapter [`Var`]s `(A, B)`, in the order the optimizer and
    /// the grad-coverage canary should iterate them.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        vec![self.a.clone(), self.b.clone()]
    }

    /// Whether the adapter currently contributes to the forward pass.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable the adapter contribution.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// The low-rank update scale `alpha / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The frozen base bias, if any — unchanged by the adapter (`LoRA` updates
    /// only the weight), exposed so a merged-weight forward can apply the same
    /// bias the unmerged [`forward`](Self::forward) applies.
    #[must_use]
    pub fn base_bias(&self) -> Option<&Tensor> {
        self.base_bias.as_ref()
    }

    /// The single effective weight the **current** forward applies: `W` when the
    /// adapter is disabled, `W + scale · (B @ A)` when enabled — so
    /// `x @ merged_weight()ᵀ (+ bias)` reproduces [`forward`](Self::forward)`(x)`
    /// up to matmul-associativity rounding, at every toggle state.
    ///
    /// The merge uses the **same cast order as the forward**: the (possibly
    /// higher-precision master) factors are cast down to the base dtype *before*
    /// the matmul, so the merged weight sees exactly the adapter the activations
    /// would see — not a higher-precision variant of it.
    ///
    /// The result is a plain frozen [`Tensor`], **detached** from the autograd
    /// tape: it is an op-free non-variable leaf (the factors are detached and the
    /// result is re-detached), so a forward built from it can never carry
    /// gradient into (or out of) the adapter. It is a *snapshot* of the live
    /// factor values — recompute it after any optimizer step **or any
    /// [`set_enabled`](Self::set_enabled) change** (the cached-rollout decoder
    /// rebuilds it per `generate()` call, which covers both and makes staleness
    /// structurally impossible).
    ///
    /// At BF16, merge fidelity is bounded by half-ulp of `W` per element: an
    /// adapter contribution below ~0.2 % of the corresponding base-weight element
    /// is absorbed into `W` by rounding (very early in training the merged
    /// rollout policy can equal the base policy exactly while the scoring
    /// forward sees a slightly adapted one). This is inherent reassociation
    /// rounding — any merge has it — and the grad/scoring path is unaffected.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the dtype cast, matmul, or add fails.
    pub fn merged_weight(&self) -> CandleResult<Tensor> {
        if !self.enabled {
            // detach() so the result is an op-free leaf even if a caller ever
            // constructed the layer from a tape-tracked base tensor.
            return Ok(self.base_weight.dequantized()?.detach());
        }
        let dtype = self.base_weight.dtype();
        // The detached/cast factors may still ALIAS the live Var storage (candle's
        // detach shares storage; a same-dtype to_dtype is a clone). The snapshot
        // guarantee holds because matmul/affine/add below all allocate fresh
        // storage — no fast path may ever return a cast/detached factor directly.
        let a = self.a.as_tensor().detach().to_dtype(dtype)?;
        let b = self.b.as_tensor().detach().to_dtype(dtype)?;
        let delta = (b.matmul(&a)? * self.scale)?;
        Ok((&self.base_weight.dequantized()? + &delta)?.detach())
    }

    pub(crate) fn snapshot(&self) -> CandleResult<FrozenLinearSnapshot> {
        match &self.base_weight {
            FrozenLinearWeight::Dense(_) => Ok(FrozenLinearSnapshot::Dense(self.merged_weight()?)),
            FrozenLinearWeight::Quantized { .. } => Ok(FrozenLinearSnapshot::Lora {
                base: self.base_weight.clone(),
                base_bias: self.base_bias.clone(),
                a: self.a.as_tensor().detach(),
                b: self.b.as_tensor().detach(),
                scale: self.scale,
                enabled: self.enabled,
            }),
        }
    }

    /// Forward pass `y = x Wᵀ (+ b) [+ scale · (x Aᵀ) Bᵀ]`.
    ///
    /// `x` has shape `[.., in]`; the result has shape `[.., out]`. The adapter
    /// term is included only when the layer is [enabled](Self::is_enabled).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `x`'s trailing dimension does not match the
    /// base weight, or if any matmul/broadcast fails.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // The frozen base path goes through `frozen_linear` (a flattened 2-D
        // matmul) so candle's unconditional weight-gradient materialization
        // stays `[out, in]`-shaped instead of batch-shaped — see its docs.
        // The adapter matmuls below keep `broadcast_matmul`: the A/B factors
        // are live Vars whose gradient *numerics* (summation order included)
        // are pinned by the optimizer-trajectory gates.
        let base = self.base_weight.forward(x)?;
        let base = match &self.base_bias {
            Some(bias) => base.broadcast_add(bias)?,
            None => base,
        };
        if !self.enabled {
            return Ok(base);
        }
        // The adapter factors may be held in a higher-precision dtype than the
        // activations (the bf16-base / F32-adapter split — see `with_adapter_dtype`).
        // Cast them down to the activation dtype for the matmul (a no-op when they
        // already match); the F32 master copy is what the optimizer updates, so its
        // precision is preserved regardless of this compute-dtype cast.
        let dtype = self.base_weight.dtype();
        let a = self.a.as_tensor().to_dtype(dtype)?;
        let b = self.b.as_tensor().to_dtype(dtype)?;
        // (x Aᵀ) Bᵀ, scaled.
        let xa = x.broadcast_matmul(&a.t()?)?;
        let xab = xa.broadcast_matmul(&b.t()?)?;
        let update = (xab * self.scale)?;
        base.broadcast_add(&update)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn base(out: usize, in_: usize) -> Tensor {
        // Deterministic ramp weight so forward outputs are predictable.
        let data: Vec<f32> = (0..out * in_).map(|i| i as f32 * 0.1).collect();
        Tensor::from_vec(data, (out, in_), &Device::Cpu).unwrap()
    }

    #[test]
    fn dense_targets_canonical_strings() {
        assert_eq!(
            DenseLoraTargets::industrial().canonical(),
            "attn:qkvo|mlp:gud"
        );
        // The default IS the industrial recipe (mirrors the qwen3_5 LoraTargets
        // convention); the dense models' load() keeps legacy() explicitly.
        assert_eq!(DenseLoraTargets::default(), DenseLoraTargets::industrial());
        // The legacy string must equal what the pre-recipe models hard-wired
        // into checkpoint manifests ("attn:qv|mlp:-") — manifest continuity.
        assert_eq!(DenseLoraTargets::legacy().canonical(), "attn:qv|mlp:-");
    }

    #[test]
    fn dense_targets_empty_recipe_is_detectable() {
        let none = DenseLoraTargets {
            attn_q: false,
            attn_k: false,
            attn_v: false,
            attn_o: false,
            mlp_gate: false,
            mlp_up: false,
            mlp_down: false,
        };
        assert_eq!(none.canonical(), "attn:-|mlp:-");
        assert!(!none.any());
        assert!(DenseLoraTargets::legacy().any());
        assert!(DenseLoraTargets::industrial().any());
    }

    #[test]
    fn proj_frozen_forward_equals_frozen_linear_and_has_no_vars() {
        let w = base(4, 3);
        let p = Proj::Frozen(
            FrozenLinearWeight::from_tensor(w.clone(), BaseQuantization::None).unwrap(),
        );
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], (1, 1, 3), &Device::Cpu).unwrap();
        let got = p.forward(&x).unwrap();
        let want = crate::blocks::frozen_linear(&x, &w).unwrap();
        let diff: f32 = got
            .broadcast_sub(&want)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(diff, 0.0);
        let mut vars = Vec::new();
        p.push_vars(&mut vars);
        assert!(vars.is_empty());
        // A frozen projection is its own merge (same values, detached).
        let m = p.merged_weight().unwrap();
        let mdiff: f32 = m
            .broadcast_sub(&w)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(mdiff, 0.0);
    }

    #[test]
    fn quantized_frozen_projection_matches_dense_with_tolerance() {
        let w = Tensor::from_vec(
            (0..128)
                .map(|i| (i as f32 - 64.0) * 0.002)
                .collect::<Vec<_>>(),
            (4, 32),
            &Device::Cpu,
        )
        .unwrap();
        let q = FrozenLinearWeight::from_tensor(w.clone(), BaseQuantization::Q8_0).unwrap();
        match &q {
            FrozenLinearWeight::Quantized { qmatmul, .. } => {
                assert!(matches!(qmatmul, QMatMul::QTensor(_)));
            }
            FrozenLinearWeight::Dense(_) => panic!("expected quantized frozen weight"),
        }

        let x2 = Tensor::from_vec(
            (0..64).map(|i| i as f32 * 0.03 - 0.2).collect::<Vec<_>>(),
            (2, 32),
            &Device::Cpu,
        )
        .unwrap();
        let got2 = q.forward_tape_free(&x2).unwrap();
        let want2 = frozen_linear(&x2, &w).unwrap();
        assert!(
            max_abs_diff(&got2, &want2) < 0.03,
            "rank-2 quantized projection drift too large"
        );

        let x3 = Tensor::from_vec(
            (0..192).map(|i| i as f32 * 0.01 - 0.1).collect::<Vec<_>>(),
            (2, 3, 32),
            &Device::Cpu,
        )
        .unwrap();
        let got3 = q.forward_tape_free(&x3).unwrap();
        let want3 = frozen_linear(&x3, &w).unwrap();
        assert!(
            max_abs_diff(&got3, &want3) < 0.03,
            "rank-3 quantized projection drift too large"
        );
    }

    #[test]
    fn quantized_frozen_projection_rejects_unaligned_shapes() {
        let err = FrozenLinearWeight::from_tensor(base(32, 3), BaseQuantization::Q8_0).unwrap_err();
        assert!(
            err.to_string().contains("block size"),
            "expected block-size rejection, got {err}"
        );
    }

    #[test]
    fn proj_merged_weight_rejects_a_biased_adapter() {
        // Proj::load always builds bias-free, but a hand-built biased
        // LoraLinear must be rejected at merge time — every merged decoder
        // applies no bias, so silently dropping one would corrupt rollout.
        let w = base(4, 3);
        let bias = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let p = Proj::Lora(LoraLinear::new(w, Some(bias), 2, 4.0).unwrap());
        let err = p.merged_weight().unwrap_err();
        assert!(
            err.to_string().contains("base bias"),
            "expected a biased-adapter rejection, got: {err}"
        );
        // Forward still applies the bias (only the merge path is unsupported).
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], (1, 1, 3), &Device::Cpu).unwrap();
        assert!(p.forward(&x).is_ok());
    }

    #[test]
    fn new_rejects_zero_rank() {
        let w = base(2, 3);
        assert!(LoraLinear::new(w, None, 0, 8.0).is_err());
    }

    #[test]
    fn scale_is_alpha_over_rank() {
        let w = base(4, 3);
        let l = LoraLinear::new(w, None, 2, 8.0).unwrap();
        assert_eq!(l.scale(), 4.0);
    }

    #[test]
    fn exposes_two_trainable_vars() {
        let w = base(4, 3);
        let l = LoraLinear::new(w, None, 2, 8.0).unwrap();
        let vars = l.trainable_vars();
        assert_eq!(vars.len(), 2);
        // A is [rank, in] = [2, 3]; B is [out, rank] = [4, 2].
        assert_eq!(vars[0].as_tensor().dims(), &[2, 3]);
        assert_eq!(vars[1].as_tensor().dims(), &[4, 2]);
    }

    #[test]
    fn zero_init_b_makes_adapter_a_noop_at_start() {
        // With B = 0, the enabled forward must equal the pure base forward.
        let w = base(4, 3);
        let l = LoraLinear::new(w.clone(), None, 2, 8.0).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();

        let y = l.forward(&x).unwrap();
        let base_y = x.broadcast_matmul(&w.t().unwrap()).unwrap();
        let diff = (y - base_y).unwrap().abs().unwrap().sum_all().unwrap();
        let diff: f32 = diff.to_scalar().unwrap();
        assert!(
            diff < 1e-6,
            "adapter with B=0 should be a no-op, diff={diff}"
        );
    }

    #[test]
    fn disabled_equals_base_even_with_nonzero_b() {
        let w = base(4, 3);
        let mut l = LoraLinear::new(w.clone(), None, 2, 8.0).unwrap();
        // Force B nonzero so the adapter would change the output if enabled.
        l.b.set(&Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();

        let enabled_y = l.forward(&x).unwrap();
        l.set_enabled(false);
        assert!(!l.is_enabled());
        let disabled_y = l.forward(&x).unwrap();

        let base_y = x.broadcast_matmul(&w.t().unwrap()).unwrap();
        // disabled == base
        let d_base: f32 = (disabled_y - base_y.clone())
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(d_base < 1e-6);
        // enabled != base (the nonzero-B adapter moved the output)
        let d_en: f32 = (enabled_y - base_y)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(d_en > 1e-3);
    }

    #[test]
    fn quantized_lora_toggle_and_nonzero_adapter_are_non_vacuous() {
        let w = base(4, 32);
        let mut l = LoraLinear::with_adapter_dtype_and_base_quantization(
            w,
            None,
            2,
            8.0,
            DType::F32,
            BaseQuantization::Q8_0,
        )
        .unwrap();
        let x = Tensor::from_vec(
            (0..64).map(|i| i as f32 * 0.05 - 0.25).collect::<Vec<_>>(),
            (2, 32),
            &Device::Cpu,
        )
        .unwrap();

        let base_y = l.base_weight.forward(&x).unwrap();
        let zero_b_y = l.forward(&x).unwrap();
        assert_eq!(
            max_abs_diff(&zero_b_y, &base_y),
            0.0,
            "zero-B quantized LoRA should equal the quantized base"
        );

        l.a.set(
            &Tensor::from_vec(
                (0..64)
                    .map(|i| 0.2f32 - i as f32 * 0.006)
                    .collect::<Vec<_>>(),
                (2, 32),
                &Device::Cpu,
            )
            .unwrap(),
        )
        .unwrap();
        l.b.set(
            &Tensor::from_vec(
                vec![0.3f32, -0.2, 0.1, 0.25, -0.15, 0.4, 0.05, -0.3],
                (4, 2),
                &Device::Cpu,
            )
            .unwrap(),
        )
        .unwrap();
        let enabled_y = l.forward(&x).unwrap();
        assert!(
            max_abs_diff(&enabled_y, &base_y) > 1e-3,
            "nonzero adapter did not move the quantized-base output"
        );

        l.set_enabled(false);
        let disabled_y = l.forward(&x).unwrap();
        assert_eq!(
            max_abs_diff(&disabled_y, &base_y),
            0.0,
            "disabled quantized LoRA should equal the quantized base"
        );
    }

    #[test]
    fn adapter_dtype_is_independent_of_the_base() {
        // The dtype split: the trainable adapter is held in its own precision,
        // independent of the frozen base. The real instance is bf16-base / F32-adapter,
        // but candle's CPU backend has no bf16 matmul, so we test the mechanism on CPU
        // with F32-base / F64-adapter (same direction: adapter higher precision than
        // the compute path).
        let w = base(4, 3); // F32 base
        let l = LoraLinear::with_adapter_dtype(w, None, 2, 8.0, DType::F64).unwrap();
        assert_eq!(l.a.as_tensor().dtype(), DType::F64);
        assert_eq!(l.b.as_tensor().dtype(), DType::F64);
        // The forward runs in the activation/base dtype (F32), not the adapter's.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let y = l.forward(&x).unwrap();
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn master_adapter_receives_its_own_dtype_grad() {
        // The anti-collapse property: even though the matmul runs in the lower compute
        // dtype, the gradient (and so the optimizer's update) lands on the adapter in
        // its master dtype — a tiny step that would round away in the compute dtype
        // survives in the master. (CPU stand-in for bf16-base / F32-adapter; see above.)
        //
        // This proves the dtype *routing* (the grad reaches the master in the master's
        // dtype, via the cast's upcast adjoint). It does NOT stress bf16's in-gradient
        // precision: the analog computes the grad in F32 and upcasts to F64, whereas the
        // bf16 instance computes it in bf16 and upcasts to F32. That the bf16 rounding is
        // tolerable is established empirically by the GPU Countdown run, not here.
        let w = base(4, 3); // F32 base
        let l = LoraLinear::with_adapter_dtype(w, None, 2, 8.0, DType::F64).unwrap();
        // Force B nonzero so A also carries a live gradient.
        l.b.set(&Tensor::ones((4, 2), DType::F64, &Device::Cpu).unwrap())
            .unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let loss = l.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for v in l.trainable_vars() {
            let g = grads
                .get(v.as_tensor())
                .expect("var missing from grad store");
            assert_eq!(
                g.dtype(),
                DType::F64,
                "master adapter must receive a grad in its own dtype"
            );
            let mag: f64 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
            assert!(mag > 0.0, "master adapter received a zero gradient");
        }
    }

    #[test]
    fn bias_is_added() {
        let w = base(2, 2);
        let bias = Tensor::from_vec(vec![10.0f32, 20.0], 2, &Device::Cpu).unwrap();
        let l = LoraLinear::new(w.clone(), Some(bias.clone()), 1, 1.0).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &Device::Cpu).unwrap();
        let y = l.forward(&x).unwrap();
        let no_bias = x.broadcast_matmul(&w.t().unwrap()).unwrap();
        let expect = no_bias.broadcast_add(&bias).unwrap();
        let diff: f32 = (y - expect)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6);
    }

    /// Max-abs elementwise difference between two tensors, as f32.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    /// A `LoRA` layer with both factors forced to non-trivial values, so the
    /// adapter genuinely moves the output (zero-B init would make every merge
    /// test vacuously pass on `delta == 0`).
    fn nonzero_lora(out: usize, in_: usize, rank: usize, alpha: f64) -> LoraLinear {
        let l = LoraLinear::new(base(out, in_), None, rank, alpha).unwrap();
        let a_data: Vec<f32> = (0..rank * in_).map(|i| 0.3 - i as f32 * 0.07).collect();
        l.a.set(&Tensor::from_vec(a_data, (rank, in_), &Device::Cpu).unwrap())
            .unwrap();
        let b_data: Vec<f32> = (0..out * rank).map(|i| 0.1 + i as f32 * 0.05).collect();
        l.b.set(&Tensor::from_vec(b_data, (out, rank), &Device::Cpu).unwrap())
            .unwrap();
        l
    }

    #[test]
    fn merged_weight_equals_base_at_zero_b_init() {
        // B = 0 ⇒ delta = scale·B@A = 0 ⇒ the merged weight IS the base weight.
        let w = base(4, 3);
        let l = LoraLinear::new(w.clone(), None, 2, 8.0).unwrap();
        let m = l.merged_weight().unwrap();
        assert_eq!(m.dims(), w.dims());
        assert_eq!(max_abs_diff(&m, &w), 0.0);
    }

    #[test]
    fn merged_forward_matches_lora_forward_with_nonzero_factors() {
        // THE faithfulness property: x @ merged_weight()ᵀ == forward(x) for a
        // genuinely non-trivial adapter, up to matmul-associativity rounding.
        let l = nonzero_lora(4, 3, 2, 8.0);
        let x = Tensor::from_vec(
            vec![0.5f32, -1.0, 2.0, 1.5, 0.0, -0.5],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let want = l.forward(&x).unwrap();
        let got = x
            .broadcast_matmul(&l.merged_weight().unwrap().t().unwrap())
            .unwrap();
        let md = max_abs_diff(&got, &want);
        assert!(
            md <= 1e-5,
            "merged forward diverged from LoRA forward: {md}"
        );
        // And the adapter is genuinely non-trivial (the property is not vacuous).
        let base_y = x.broadcast_matmul(&base(4, 3).t().unwrap()).unwrap();
        assert!(max_abs_diff(&want, &base_y) > 1e-3, "adapter was a no-op");
    }

    #[test]
    fn merged_weight_respects_the_adapter_toggle() {
        // Disabled ⇒ the merged weight is the pure base (the eval/base-rollout
        // case); enabled ⇒ it differs. `x @ mergedᵀ == forward(x)` must hold at
        // BOTH toggle states — that is what lets a per-generate rebuild serve the
        // adapter-on policy rollout and the adapter-off eval rollout unchanged.
        let mut l = nonzero_lora(4, 3, 2, 8.0);
        let w = base(4, 3);
        let on = l.merged_weight().unwrap();
        assert!(
            max_abs_diff(&on, &w) > 1e-3,
            "enabled merge ignored the adapter"
        );

        l.set_enabled(false);
        let off = l.merged_weight().unwrap();
        assert_eq!(max_abs_diff(&off, &w), 0.0);
        let x = Tensor::from_vec(vec![1.0f32, -2.0, 0.5], (1, 3), &Device::Cpu).unwrap();
        let want = l.forward(&x).unwrap();
        let got = x.broadcast_matmul(&off.t().unwrap()).unwrap();
        assert!(max_abs_diff(&got, &want) <= 1e-6);
    }

    #[test]
    fn merged_weight_with_bias_reproduces_forward() {
        // The merge covers only the weight; the caller applies the (unchanged)
        // frozen bias via `base_bias()`. Together they reproduce forward().
        let bias = Tensor::from_vec(vec![10.0f32, -3.0, 0.25, 7.5], 4, &Device::Cpu).unwrap();
        let l = LoraLinear::new(base(4, 3), Some(bias.clone()), 2, 8.0).unwrap();
        l.b.set(&Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        let x = Tensor::from_vec(vec![0.5f32, 1.0, -1.5], (1, 3), &Device::Cpu).unwrap();
        let want = l.forward(&x).unwrap();
        let got = x
            .broadcast_matmul(&l.merged_weight().unwrap().t().unwrap())
            .unwrap()
            .broadcast_add(l.base_bias().unwrap())
            .unwrap();
        let md = max_abs_diff(&got, &want);
        assert!(md <= 1e-5, "merged + bias diverged from forward: {md}");
    }

    #[test]
    fn merged_weight_casts_factors_down_before_the_matmul() {
        // Pin the cast ORDER under the dtype split, not just the result dtype.
        // F32 base / F64 adapter, with A built for catastrophic cancellation:
        // delta = B@A = 1·(1e8 + 0.25) + 1·(−1e8). Cast-to-F32-then-matmul (the
        // forward's order) rounds 1e8 + 0.25 → 1e8 ⇒ delta == 0 exactly; a
        // matmul-in-F64-then-cast merge would keep delta == 0.25 — far above any
        // rounding tolerance, so the wrong order cannot sneak through.
        let w = base(1, 1);
        let l = LoraLinear::with_adapter_dtype(w.clone(), None, 2, 2.0, DType::F64).unwrap();
        l.a.set(&Tensor::from_vec(vec![1.0e8f64 + 0.25, -1.0e8], (2, 1), &Device::Cpu).unwrap())
            .unwrap();
        l.b.set(&Tensor::from_vec(vec![1.0f64, 1.0], (1, 2), &Device::Cpu).unwrap())
            .unwrap();
        let m = l.merged_weight().unwrap();
        assert_eq!(m.dtype(), DType::F32, "the merge runs in the base dtype");
        assert_eq!(
            max_abs_diff(&m, &w),
            0.0,
            "merge multiplied the F64 masters before casting down — it must cast \
             to the base dtype first, exactly as the forward does"
        );
    }

    #[test]
    fn merged_weight_is_detached_from_the_tape() {
        // Structural guarantee: a loss built from the merged weight reaches
        // NEITHER LoRA factor — the rollout forward can never leak gradient into
        // the adapter (or drag the rollout graph onto the tape).
        let l = nonzero_lora(4, 3, 2, 8.0);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let loss = x
            .broadcast_matmul(&l.merged_weight().unwrap().t().unwrap())
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        for v in l.trainable_vars() {
            assert!(
                grads.get(v.as_tensor()).is_none(),
                "merged_weight leaked a grad path into a LoRA factor"
            );
        }
    }

    #[test]
    fn merged_weight_reflects_live_var_values() {
        // The merge is a snapshot of the LIVE factors: after an (optimizer-style)
        // in-place Var update, a fresh merge differs from the stale one — the
        // per-generate rebuild is what keeps the cached rollout current.
        let l = nonzero_lora(4, 3, 2, 8.0);
        let before = l.merged_weight().unwrap();
        let bumped = ((l.b.as_tensor() + 0.5).unwrap()).detach();
        l.b.set(&bumped).unwrap();
        let after = l.merged_weight().unwrap();
        assert!(
            max_abs_diff(&after, &before) > 1e-3,
            "merged_weight did not track an in-place Var update"
        );
    }

    #[test]
    fn grad_coverage_canary_contract() {
        // candle optimizers silently skip params missing from the GradStore, so
        // the canary asserts grad COVERAGE — but the contract is init-dependent:
        //
        //  (1) standard init (B = 0): the update (x@Aᵀ)@Bᵀ is 0, so dL/dA is
        //      legitimately 0; only B is guaranteed present with a non-zero grad.
        //  (2) after a non-zero-B step: BOTH A and B must be present + non-zero.
        //
        // A runtime canary must therefore not require A != 0 at step 0, or it
        // false-fails on standard zero-B LoRA init.
        let w = base(4, 3);
        let l = LoraLinear::new(w, None, 2, 8.0).unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();

        // (1) standard init: B = 0 -> B has a non-zero grad (A's grad is ~0).
        let loss = l.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let gb = grads
            .get(l.b.as_tensor())
            .expect("B missing from grad store");
        let mag_b: f32 = gb.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(mag_b > 0.0, "B received a zero gradient at standard init");

        // (2) after a non-zero-B update, every trainable var has a non-zero grad.
        l.b.set(&Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        let loss = l.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for v in l.trainable_vars() {
            let g = grads
                .get(v.as_tensor())
                .expect("var missing from grad store");
            let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
            assert!(
                mag > 0.0,
                "trainable var received a zero gradient after a non-zero-B step"
            );
        }
    }

    #[test]
    fn downstream_quantized_projection_preserves_upstream_adapter_gradients() {
        // The training/scoring path may store frozen weights quantized, but it
        // must not use candle's tape-free QMatMul there. A downstream quantized
        // frozen projection still has to pass dL/dhidden back into an upstream
        // adapter; otherwise a multi-layer QLoRA model would train only its
        // final adapted projection.
        let l1 = LoraLinear::with_adapter_dtype_and_base_quantization(
            base(32, 32),
            None,
            2,
            4.0,
            DType::F32,
            BaseQuantization::Q8_0,
        )
        .unwrap();
        l1.a.set(
            &Tensor::from_vec(
                (0..64).map(|i| i as f32 * 0.003 - 0.08).collect::<Vec<_>>(),
                (2, 32),
                &Device::Cpu,
            )
            .unwrap(),
        )
        .unwrap();
        l1.b.set(
            &Tensor::from_vec(
                (0..64).map(|i| 0.04 - i as f32 * 0.001).collect::<Vec<_>>(),
                (32, 2),
                &Device::Cpu,
            )
            .unwrap(),
        )
        .unwrap();
        let downstream =
            FrozenLinearWeight::from_tensor(base(1, 32), BaseQuantization::Q8_0).unwrap();
        let x = Tensor::from_vec(
            (0..32).map(|i| i as f32 * 0.02 - 0.3).collect::<Vec<_>>(),
            (1, 32),
            &Device::Cpu,
        )
        .unwrap();

        let h = l1.forward(&x).unwrap();
        let y = downstream.forward(&h).unwrap();
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for v in l1.trainable_vars() {
            let g = grads
                .get(v.as_tensor())
                .expect("upstream adapter var missing from grad store");
            let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
            assert!(
                mag.is_finite() && mag > 0.0,
                "upstream adapter gradient was not live through quantized frozen projection"
            );
        }
    }
}
