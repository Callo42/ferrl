//! Architecture-neutral decoder building blocks.
//!
//! The pieces of a rotate-half-`RoPE` transformer decoder that do **not** depend
//! on any particular architecture's config type: frozen linear projection,
//! grouped-query KV repeat, the precomputed `RoPE` cos/sin tables, and the
//! additive causal-mask builders (full-sequence and offset-aware). They were
//! extracted verbatim from the Qwen3 forward ([`crate::qwen`]) so a second
//! architecture (e.g. a dense Llama-3.x, which uses the same rotate-half `RoPE`
//! family, GQA, and causal masking) can reuse them instead of re-deriving them.
//!
//! Everything here is grad-bearing (pure tensor ops, no autograd-cutting custom
//! kernels), so the blocks are safe in both the update (grad) forward and the
//! cached rollout decoder. Behavior is pinned by the [`crate::qwen`] equivalence
//! gates against candle's shipped forward, which exercise every function in this
//! module at every position.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

/// `y = x Wᵀ` for a frozen weight `w` of shape `[out, in]` (candle `Linear`
/// layout), broadcasting over the leading dims of `x`.
///
/// # Errors
///
/// Returns a candle error if the matmul shapes are incompatible.
pub fn frozen_linear(x: &Tensor, w: &Tensor) -> CandleResult<Tensor> {
    x.broadcast_matmul(&w.t()?)
}

/// Grouped-query repeat: `[b, kv_heads, l, d] -> [b, kv_heads * n_rep, l, d]`,
/// each kv head repeated `n_rep` times consecutively (matching candle's
/// `repeat_kv`). Pure `expand`+`reshape`, so it carries gradient.
///
/// # Errors
///
/// Returns a candle error if `x` is not a 4-D tensor or a reshape fails.
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> CandleResult<Tensor> {
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
/// `Qwen3RotaryEmbedding::new` — the same rotate-half table layout Llama-family
/// models use, so the tables are architecture-neutral.
#[derive(Debug, Clone)]
pub struct RotaryTables {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryTables {
    /// Build the tables from plain scalars (no architecture config type):
    /// `head_dim` is the per-head dimension the rotation is applied over,
    /// `rope_theta` the frequency base, and `max_position_embeddings` the number
    /// of absolute positions to precompute.
    ///
    /// The tables are cast to `dtype` (the model/activation dtype — the shipped
    /// `Qwen3RotaryEmbedding` casts here too) so `rope_slow`'s `broadcast_mul`
    /// against q/k does not dtype-mismatch at bf16.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a table tensor cannot be built on `device`.
    pub fn new(
        head_dim: usize,
        rope_theta: f64,
        max_position_embeddings: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        let dim = head_dim;
        let max_pos = max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// `cos`/`sin` narrowed to the first `seq_len` positions (offset 0).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `seq_len` exceeds `max_position_embeddings`.
    pub fn slice(&self, seq_len: usize) -> CandleResult<(Tensor, Tensor)> {
        self.slice_at(0, seq_len)
    }

    /// `cos`/`sin` narrowed to the `seq_len` positions starting at absolute
    /// position `offset` — the cached-decode generalization of [`slice`](Self::slice),
    /// matching candle's `Qwen3RotaryEmbedding::apply` (`cos.narrow(0, offset, seq_len)`).
    /// At `offset == 0` it is exactly [`slice`](Self::slice).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `offset + seq_len` exceeds
    /// `max_position_embeddings`.
    pub fn slice_at(&self, offset: usize, seq_len: usize) -> CandleResult<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(0, offset, seq_len)?,
            self.sin.narrow(0, offset, seq_len)?,
        ))
    }
}

/// Additive causal mask `[1, 1, l, l]` (`0` on/below the diagonal, `-inf`
/// above), matching candle's `Model::causal_mask` at offset 0 — the
/// full-sequence (uncached) mask.
///
/// Exactly [`causal_mask_at`] with `offset == 0` (same data, same shape).
///
/// # Errors
///
/// Returns a candle error if the mask tensor cannot be built on `device`.
pub fn causal_mask(l: usize, dtype: DType, device: &Device) -> CandleResult<Tensor> {
    causal_mask_at(0, l, dtype, device)
}

/// Additive causal mask `[1, 1, chunk_len, offset + chunk_len]` for a chunk of
/// `chunk_len` queries at absolute positions `[offset, offset + chunk_len)`
/// against `offset + chunk_len` keys: `0` where query `i` (absolute `i+offset`)
/// may attend to key `j` (`j <= i + offset`), `-inf` above. Matches candle's
/// `Model::causal_mask(b, tgt, offset, None)` — the cached-decode mask.
///
/// # Errors
///
/// Returns a candle error if the mask tensor cannot be built on `device`.
pub fn causal_mask_at(
    offset: usize,
    chunk_len: usize,
    dtype: DType,
    device: &Device,
) -> CandleResult<Tensor> {
    let total = offset + chunk_len;
    let mut data = Vec::with_capacity(chunk_len * total);
    for i in 0..chunk_len {
        for j in 0..total {
            data.push(if j <= i + offset {
                0f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::from_vec(data, (1, 1, chunk_len, total), device)?.to_dtype(dtype)
}
