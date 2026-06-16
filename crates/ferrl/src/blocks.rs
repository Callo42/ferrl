//! Architecture-neutral decoder building blocks.
//!
//! The pieces of a rotate-half-`RoPE` transformer decoder that do **not** depend
//! on any particular architecture's config type: frozen linear projection,
//! grouped-query KV repeat, the precomputed `RoPE` cos/sin tables, and the
//! additive causal-mask builders (full-sequence and offset-aware). They were
//! extracted from the Qwen3 forward ([`crate::qwen`]) — behavior-preserving,
//! though not literally verbatim: [`RotaryTables::new`] generalized its
//! signature from the Qwen config type to plain scalars (and gained
//! [`RotaryTables::with_inv_freq`] for precomputed frequencies), and
//! [`causal_mask`] became a delegation to the offset-aware [`causal_mask_at`] —
//! so a second architecture (e.g. a dense Llama-3.x, which uses the same
//! rotate-half `RoPE` family, GQA, and causal masking) can reuse them instead
//! of re-deriving them.
//!
//! Everything here is grad-bearing (pure tensor ops, no autograd-cutting custom
//! kernels), so the blocks are safe in both the update (grad) forward and the
//! cached rollout decoder. Behavior is pinned by the [`crate::qwen`] **and**
//! [`crate::llama`] equivalence gates against candle's shipped forwards, at
//! every position — [`RotaryTables::with_inv_freq`] is reached only by the
//! llama gates (plus exact inv-freq pins there); the rest are exercised by both
//! architectures.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

/// Narrow `h` (`[batch, seq, ..]`) to `window = (start, len)` along the
/// sequence dim; `None` is the identity (a shallow clone). The shared helper
/// behind every model's narrowed scoring forward.
///
/// # Errors
///
/// Returns a candle error if the window exceeds the sequence.
pub(crate) fn windowed(h: &Tensor, window: Option<(usize, usize)>) -> CandleResult<Tensor> {
    match window {
        None => Ok(h.clone()),
        Some((start, len)) => h.narrow(1, start, len),
    }
}

/// `y = x Wᵀ` for a frozen weight `w` of shape `[out, in]` (candle `Linear`
/// layout); leading dims of `x` are flattened around a single 2-D matmul.
///
/// The flatten (rather than `broadcast_matmul`) is a **memory** fix, not a
/// numeric one: candle's `Matmul` backward materializes a gradient for *both*
/// arguments unconditionally — even a frozen weight — and a grad computed
/// against a batch-broadcast weight is batch-shaped (`[b, in, out]`; at the
/// tied lm-head that is `batch × hidden × vocab`, gigabytes per prompt) and,
/// worse, *retained* in the returned `GradStore` because the broadcast node
/// under it tracks no gradient and is never visited to reduce it. With the
/// 2-D matmul the unconditional weight gradient is `[in, out]`-shaped — the
/// `batch × seq` factor is gone. Every output element is the same
/// dot-product over the same `in`-axis either way; values (and the `x`
/// gradient) agree with the broadcast path up to **f32 reassociation** of the
/// gemm's accumulation (measured ≲ 1e-7 on CPU — the same class as the
/// P6-C merged-weight reassociation; the equivalence tests pin a tight
/// envelope, and the cached/uncached identity gates stay *exact* because both
/// paths share this primitive).
///
/// # Errors
///
/// Returns a candle error if the matmul shapes are incompatible.
pub fn frozen_linear(x: &Tensor, w: &Tensor) -> CandleResult<Tensor> {
    let wt = w.t()?;
    if x.rank() <= 2 {
        return x.matmul(&wt);
    }
    let mut out_dims = x.dims().to_vec();
    if let Some(last) = out_dims.last_mut() {
        *last = w.dim(0)?;
    }
    x.contiguous()?
        .flatten_to(x.rank() - 2)?
        .matmul(&wt)?
        .reshape(out_dims)
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
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        Self::with_inv_freq(inv_freq, max_position_embeddings, dtype, device)
    }

    /// Build the tables from **precomputed** inverse frequencies (one per
    /// rotated dimension pair, i.e. `head_dim / 2` entries) — the generic core
    /// [`new`](Self::new) delegates to.
    ///
    /// This is the seam for architectures whose inv-freqs are *not* the plain
    /// `1/theta^(2i/d)` family: e.g. Llama-3.x `RoPE` scaling rescales the
    /// inv-freqs at table-build time ([`crate::llama`] computes the smoothed
    /// frequencies and passes them here), while the table layout, slicing, and
    /// dtype handling stay identical.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a table tensor cannot be built on `device`.
    pub fn with_inv_freq(
        inv_freq: Vec<f32>,
        max_position_embeddings: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        let max_pos = max_position_embeddings;
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

/// Partial rotate-half `RoPE`: rotate only the first `rot_dim` dims of each
/// head, pass dims `rot_dim..` through untouched.
///
/// `x` is `[b, heads, l, head_dim]`; `cos`/`sin` are `[l, rot_dim / 2]` (e.g.
/// a [`RotaryTables`] built with `head_dim = rot_dim` — table width is just
/// `len(inv_freq)`, so the partial case is a narrower table, sliced as usual).
/// The `qwen3_5` family uses `partial_rotary_factor 0.25` (64 of 256 head
/// dims, theta 1e7); its interleaved M-`RoPE` reduces exactly to this standard
/// 1-D rotate-half rope for text-only inputs (the reference expands text
/// positions to three identical T/H/W rows → the section interleave
/// overwrites entries with identical values, an exact no-op). At `rot_dim == head_dim`
/// this is exactly `rope_slow` (the grad-safe rope — the fused `rope*` kernels
/// have no backward and must never appear in a training forward).
///
/// # Errors
///
/// Returns a candle error if `rot_dim` is zero, odd, or exceeds `head_dim`,
/// or if an underlying tensor op fails.
pub fn rope_partial(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rot_dim: usize,
) -> CandleResult<Tensor> {
    let (_b, _h, _l, head_dim) = x.dims4()?;
    if rot_dim == 0 || !rot_dim.is_multiple_of(2) || rot_dim > head_dim {
        candle_core::bail!(
            "rope_partial: rot_dim {rot_dim} must be a nonzero even number <= head_dim {head_dim}"
        );
    }
    if rot_dim == head_dim {
        return candle_nn::rotary_emb::rope_slow(&x.contiguous()?, cos, sin);
    }
    let x_rot = x.narrow(candle_core::D::Minus1, 0, rot_dim)?.contiguous()?;
    let x_pass = x.narrow(candle_core::D::Minus1, rot_dim, head_dim - rot_dim)?;
    let rotated = candle_nn::rotary_emb::rope_slow(&x_rot, cos, sin)?;
    Tensor::cat(&[&rotated, &x_pass.contiguous()?], candle_core::D::Minus1)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    /// The flattened matmul is the same dot-product per output element as the
    /// broadcast path up to f32 reassociation of the gemm's accumulation —
    /// pin a tight envelope (measured ≲ 2.4e-7 on these O(1) values), at
    /// every rank.
    #[test]
    fn frozen_linear_matches_broadcast_matmul_tightly() {
        let dev = Device::Cpu;
        let w_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.05 - 0.4).collect();
        let w = Tensor::from_vec(w_data, (5, 4), &dev).unwrap();
        for dims in [vec![3, 4], vec![2, 3, 4], vec![2, 2, 3, 4]] {
            let n: usize = dims.iter().product();
            let xv: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 0.7).collect();
            let x = Tensor::from_vec(xv, dims, &dev).unwrap();
            let got: Vec<f32> = frozen_linear(&x, &w)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let want: Vec<f32> = x
                .broadcast_matmul(&w.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (g, want) in got.iter().zip(&want) {
                assert!(
                    (g - want).abs() <= 1e-6,
                    "rank-{} forward drifted: {g} vs {want}",
                    x.rank()
                );
            }
        }
    }

    /// The gradient that flows back into `x` is also the same dot-products
    /// either way up to gemm reassociation — pin a tight envelope so the
    /// flatten can never bend training beyond last-ulp noise.
    #[test]
    fn frozen_linear_x_gradient_matches_the_broadcast_path() {
        let dev = Device::Cpu;
        let w_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.05 - 0.4).collect();
        let w = Tensor::from_vec(w_data, (5, 4), &dev).unwrap();
        let xv: Vec<f32> = (0..24).map(|i| i as f32 * 0.1 - 0.7).collect();
        let x = Var::from_tensor(&Tensor::from_vec(xv, (2, 3, 4), &dev).unwrap()).unwrap();

        let grad_of = |y: Tensor| -> Vec<f32> {
            let store = y.sum_all().unwrap().backward().unwrap();
            store
                .get(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let got = grad_of(frozen_linear(&x, &w).unwrap());
        let want = grad_of(x.broadcast_matmul(&w.t().unwrap()).unwrap());
        for (g, want) in got.iter().zip(&want) {
            assert!(
                (g - want).abs() <= 1e-6,
                "x-gradient drifted under the flatten: {g} vs {want}"
            );
        }
    }

    /// THE point of the flatten: candle materializes (and the store retains) a
    /// gradient even for a frozen matmul argument, and against a broadcast
    /// weight that gradient is batch-shaped. After the fix no retained
    /// gradient may exceed the boundary-input size — the batch-shaped
    /// `[b, in, out]` weight gradient (40 elements here) must be gone.
    #[test]
    fn frozen_linear_retains_no_batch_shaped_weight_gradient() {
        let dev = Device::Cpu;
        let w_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.05 - 0.4).collect();
        let w = Tensor::from_vec(w_data, (5, 4), &dev).unwrap();
        let xv: Vec<f32> = (0..24).map(|i| i as f32 * 0.1 - 0.7).collect();
        let x = Var::from_tensor(&Tensor::from_vec(xv, (2, 3, 4), &dev).unwrap()).unwrap();

        let store = frozen_linear(&x, &w)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let max_retained = store
            .get_ids()
            .filter_map(|id| store.get_id(*id))
            .map(candle_core::Tensor::elem_count)
            .max()
            .unwrap();
        assert!(
            max_retained <= x.elem_count().max(w.elem_count()),
            "a retained gradient has {max_retained} elements — \
             the batch-shaped frozen-weight gradient is back"
        );
    }

    /// `rope_partial` rotates exactly the first `rot_dim` dims (matching a
    /// scalar rotate-half reference) and passes the rest through untouched.
    #[test]
    fn rope_partial_rotates_head_and_passes_tail() {
        let dev = Device::Cpu;
        let (b, h, l, d, rot) = (1usize, 1usize, 2usize, 8usize, 4usize);
        let theta = 100.0f64;
        let tables = RotaryTables::new(rot, theta, 4, DType::F32, &dev).unwrap();
        let (cos, sin) = tables.slice(l).unwrap();

        let xv: Vec<f32> = (0..b * h * l * d).map(|i| i as f32 * 0.1 - 0.7).collect();
        let x = Tensor::from_vec(xv.clone(), (b, h, l, d), &dev).unwrap();
        let y = rope_partial(&x, &cos, &sin, rot).unwrap();
        let yv: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();

        // Scalar rotate-half reference on the first `rot` dims of each position.
        let half = rot / 2;
        for pos in 0..l {
            let inv_freq: Vec<f64> = (0..rot)
                .step_by(2)
                .map(|i| 1.0 / theta.powf(i as f64 / rot as f64))
                .collect();
            for j in 0..half {
                let angle = pos as f64 * inv_freq[j];
                let (c, s) = (angle.cos() as f32, angle.sin() as f32);
                let base = pos * d;
                let x1 = xv[base + j];
                let x2 = xv[base + half + j];
                let want1 = x1 * c - x2 * s;
                let want2 = x2 * c + x1 * s;
                assert!(
                    (yv[base + j] - want1).abs() < 1e-5,
                    "pos {pos} dim {j}: {} vs {want1}",
                    yv[base + j]
                );
                assert!(
                    (yv[base + half + j] - want2).abs() < 1e-5,
                    "pos {pos} dim {}: {} vs {want2}",
                    half + j,
                    yv[base + half + j]
                );
            }
            // Pass-through tail is bit-identical.
            for j in rot..d {
                assert_eq!(yv[pos * d + j], xv[pos * d + j], "pos {pos} dim {j}");
            }
        }
    }

    /// Full-width `rope_partial` equals `rope_slow` (the degenerate case), and
    /// gradients cross the partial rope to an upstream Var on BOTH regions.
    #[test]
    fn rope_partial_full_width_and_grad_flow() {
        let dev = Device::Cpu;
        let (b, h, l, d) = (1usize, 2usize, 3usize, 4usize);
        let tables = RotaryTables::new(d, 10_000.0, 8, DType::F32, &dev).unwrap();
        let (cos, sin) = tables.slice(l).unwrap();
        let xv: Vec<f32> = (0..b * h * l * d)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let x = Tensor::from_vec(xv, (b, h, l, d), &dev).unwrap();

        let full = rope_partial(&x, &cos, &sin, d).unwrap();
        let slow = candle_nn::rotary_emb::rope_slow(&x, &cos, &sin).unwrap();
        let dv: Vec<f32> = (full - slow)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            dv.iter().all(|v| *v < 1e-6),
            "full-width must equal rope_slow"
        );

        // Grad flow through the partial case: every input position receives
        // gradient (rotated region via cos/sin, pass-through via identity).
        let tables2 = RotaryTables::new(2, 10_000.0, 8, DType::F32, &dev).unwrap();
        let (cos2, sin2) = tables2.slice(l).unwrap();
        let xv2 = Var::from_tensor(&x).unwrap();
        let y = rope_partial(xv2.as_tensor(), &cos2, &sin2, 2).unwrap();
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let gx = grads
            .get(xv2.as_tensor())
            .expect("rope_partial grad present");
        let gv: Vec<f32> = gx.flatten_all().unwrap().to_vec1().unwrap();
        assert!(gv.iter().all(|v| v.is_finite()));
        assert!(gv.iter().filter(|v| **v != 0.0).count() > gv.len() / 2);
    }

    /// Fail-loud on invalid `rot_dim`.
    #[test]
    fn rope_partial_rejects_bad_rot_dim() {
        let dev = Device::Cpu;
        let tables = RotaryTables::new(4, 10_000.0, 8, DType::F32, &dev).unwrap();
        let (cos, sin) = tables.slice(2).unwrap();
        let x = Tensor::zeros((1, 1, 2, 8), DType::F32, &dev).unwrap();
        assert!(rope_partial(&x, &cos, &sin, 0).is_err());
        assert!(rope_partial(&x, &cos, &sin, 3).is_err());
        assert!(rope_partial(&x, &cos, &sin, 10).is_err());
    }

    /// THE rope oracle gate: `RotaryTables::new` + `rope_partial` against a
    /// dump of the ACTUAL transformers `qwen3_5` rotary path
    /// (`Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb`) at the real
    /// geometry — `head_dim` 256, `rotary_dim` 64, theta 1e7, text-only 1-D
    /// positions. Unlike the scalar reference above (which shares this
    /// module's conventions and can't catch a shared misunderstanding), this
    /// pins the exponent base (over `rot_dim`, not `head_dim`), the
    /// rotate-half pairing, the partial split, AND the prose claim that the
    /// interleaved M-`RoPE` is an exact no-op for text.
    #[test]
    fn rope_partial_matches_transformers_oracle_at_qwen35_geometry() {
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/gdn_golden.json"))
                .expect("gdn golden fixture parses");
        let c = &golden["cases"]["rope_text_qwen35_geometry"];
        assert!(!c.is_null(), "rope fixture case present");
        let l = c["l"].as_u64().unwrap() as usize;
        let head_dim = c["head_dim"].as_u64().unwrap() as usize;
        let rot_dim = c["rot_dim"].as_u64().unwrap() as usize;
        let theta = c["rope_theta"].as_f64().unwrap();
        let dev = Device::Cpu;
        let load = |key: &str| -> Tensor {
            let v: Vec<f32> = c[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            Tensor::from_vec(v, (1, 1, l, head_dim), &dev).unwrap()
        };
        let tables = RotaryTables::new(rot_dim, theta, 64, DType::F32, &dev).unwrap();
        let (cos, sin) = tables.slice(l).unwrap();
        for (inp, out) in [("q", "q_out"), ("k", "k_out")] {
            let x = load(inp);
            let want = load(out);
            let got = rope_partial(&x, &cos, &sin, rot_dim).unwrap();
            let diff: Vec<f32> = (got - &want)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let max = diff.iter().fold(0.0f32, |a, b| a.max(*b));
            assert!(max <= 1e-5, "{inp}: rope oracle diff {max}");
        }
    }
}
