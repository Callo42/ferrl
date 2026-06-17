//! The dense Llama-3.x binding of the shared dense backbone ([`crate::dense`]).
//!
//! candle-transformers ships a Llama forward, but it is inference-shaped
//! (`&mut Cache`, last-position-only logits, all layer types private) and built
//! from ops that have **no backward**, so it cannot be used to train. The
//! grad-bearing *update* path for the dense Llama family (Llama 3.x: plain GQA,
//! rotate-half `RoPE` with optional llama3 scaling, `SwiGLU`, `RMSNorm`, no
//! QK-norm, no biases) lives in [`crate::dense`] now, shared with Qwen3. This
//! module is the thin Llama-specific seam over it: the [`LlamaArch`] marker,
//! which validates candle's `llama::Config` and distills it into a [`DenseSpec`].
//!
//! ## Parity notes vs the shipped forward (the Llama knobs)
//!
//! - The shipped non-flash attention **force-casts q/k/v to F32** for the
//!   score/softmax computation and casts the context back to the model dtype;
//!   the shared backbone mirrors that via `sdpa_f32 = true` (a no-op at F32, the
//!   same numerics at BF16). The causal mask is therefore F32 here.
//! - `head_dim` is derived as `hidden_size / num_attention_heads` (the llama
//!   `Config` has no `head_dim` field), exactly as the shipped loader does.
//! - No per-head QK-norm; the `SwiGLU` activation is fixed `Silu`.
//! - `rope_scaling`: `None` (or `Some` with `rope_type: Default`) is the plain
//!   `1/theta^(2i/d)` family; `Some` with `rope_type: Llama3` applies the llama3
//!   wavelength-smoothing rescale to the inv-freqs at table-build time (the
//!   private `inv_freq_for` helper) — both mirrored from the shipped `Cache::new`,
//!   pinned by exact-value inv-freq pins plus the equivalence gates below.
//!
//! ## Validation beyond CPU CI: real weights and the bf16 path
//!
//! Every CI test here runs on CPU at F32, where the attention force-cast pair is
//! a same-dtype `to_dtype` — an op-free clone, structurally absent from the
//! autograd graph. The gaps that leaves are closed by two `#[ignore]`d manual
//! gates (`tests/llama_real_weights.rs`, `tests/llama_gpu_smoke.rs`); see their
//! module docs. Honest residual: those gates are manual, so a bf16 regression
//! surfaces at the next manual gate run, not in CI.

use candle_core::{DType, Device, Result as CandleResult};
use candle_nn::Activation;
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

use crate::blocks::RotaryTables;
use crate::dense::{DenseArch, DenseCachedDecoder, DenseGradModel, DenseSpec};

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

/// The dense Llama-3.x architecture marker for the shared dense backbone (see
/// [`crate::dense::DenseArch`]). Zero-sized; only its [`DenseArch`] impl matters.
#[derive(Debug, Clone, Copy)]
pub struct LlamaArch;

impl DenseArch for LlamaArch {
    type Config = Config;

    const LABEL: &'static str = "LlamaGradModel";

    fn spec(cfg: &Config, dtype: DType, device: &Device) -> CandleResult<DenseSpec> {
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
        // num_attention_heads is the divisor for the derived head_dim. Zero must
        // be rejected explicitly: `is_multiple_of(0)` is true when hidden_size is
        // also 0, so the divisibility check below would let a degenerate `(0, 0)`
        // config slip past and fail deep in weight-load instead of loud here.
        if cfg.num_attention_heads == 0 {
            candle_core::bail!(
                "LlamaGradModel: num_attention_heads must be >= 1 (got 0); head_dim is \
                 derived as hidden_size / num_attention_heads"
            );
        }
        // num_key_value_heads is the divisor that derives num_kv_groups (the GQA
        // repeat factor, num_attention_heads / num_key_value_heads). Zero would
        // integer-divide-by-zero and panic deep in attention load; reject it loud
        // here. num_attention_heads is guarded above, so a zero divisor can only
        // come from num_key_value_heads itself.
        if cfg.num_key_value_heads == 0 {
            candle_core::bail!(
                "LlamaGradModel: num_key_value_heads must be >= 1 (got 0); num_kv_groups is \
                 derived as num_attention_heads / num_key_value_heads"
            );
        }
        // head_dim is DERIVED as hidden_size / num_attention_heads (the llama
        // Config has no head_dim field). The shipped loader trips a reshape
        // error deep in the forward when the division truncates; we would
        // silently run a degenerate (non-parity) model — so reject it at load.
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            candle_core::bail!(
                "LlamaGradModel: hidden_size {} is not divisible by num_attention_heads {} \
                 (head_dim is derived as their quotient; such a config cannot be a real \
                 Llama and would silently load as a degenerate model)",
                cfg.hidden_size,
                cfg.num_attention_heads
            );
        }
        // num_kv_groups is DERIVED as num_attention_heads / num_key_value_heads
        // (the GQA repeat factor). A non-divisible pair truncates the quotient,
        // so repeat_kv would expand the KV heads to a count that no longer matches
        // the Q heads — a degenerate (non-parity) model. Reject it at load,
        // matching the qwen3.5 loader's existing GQA-divisibility guard.
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            candle_core::bail!(
                "LlamaGradModel: num_attention_heads {} is not divisible by \
                 num_key_value_heads {} (num_kv_groups is derived as their quotient; such a \
                 config cannot be a real Llama and would silently load as a degenerate model)",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        Ok(DenseSpec {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            // The llama Config has no head_dim field; derive it as shipped does.
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            intermediate_size: cfg.intermediate_size,
            rms_norm_eps: cfg.rms_norm_eps as f32,
            tie_word_embeddings: cfg.tie_word_embeddings,
            // Llama has no per-head QK-norm; it force-casts the SDPA to F32 (the
            // shipped non-flash path) and fixes the SwiGLU activation to Silu.
            qk_norm: false,
            sdpa_f32: true,
            activation: Activation::Silu,
            // The inv-freqs carry the whole rope_scaling story (llama3 smoothing
            // happens at table-build time); the table layout is the neutral one.
            rot: RotaryTables::with_inv_freq(
                inv_freq_for(cfg),
                cfg.max_position_embeddings,
                dtype,
                device,
            )?,
        })
    }
}

/// A grad-bearing, uncached dense Llama-3.x forward with `LoRA` — the
/// [`DenseGradModel`] over the Llama [`LlamaArch`]. Built from the same
/// `VarBuilder` (over the same safetensors) as candle's shipped `llama::Llama`,
/// so the two are weight-identical and their logits match (the equivalence gate
/// below).
pub type LlamaGradModel = DenseGradModel<LlamaArch>;

/// A KV-cached, grad-free dense-Llama decoder over merged weights — the fast
/// rollout twin of [`LlamaGradModel`]. The shared [`DenseCachedDecoder`].
pub type LlamaMergedDecoder = DenseCachedDecoder;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseLayer;
    use crate::lora::{DenseLoraTargets, Proj};
    use crate::model::GradModel;
    use crate::nn::grad_coverage;
    use candle_core::backprop::GradStore;
    use candle_core::{Tensor, Var};
    use candle_nn::VarBuilder;
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

    #[test]
    fn load_rejects_zero_attention_heads() {
        // The num_attention_heads >= 1 guard closes the one gap the is_multiple_of
        // rewrite left: at num_attention_heads == 0 AND hidden_size == 0,
        // `hidden_size.is_multiple_of(0)` is true, so the divisibility check alone
        // would let a degenerate (0, 0) config slip past and fail later at
        // weight-load. The explicit guard rejects it loud, up front. (vb is built
        // from a valid cfg; the guard fires before any tensor is touched.)
        let good = tiny_cfg();
        let vb = tiny_vb(&good);
        let mut bad = tiny_cfg();
        bad.hidden_size = 0;
        bad.num_attention_heads = 0;
        let err = LlamaGradModel::load(&bad, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string().contains("num_attention_heads must be"),
            "expected a num_attention_heads>=1 rejection, got: {err}"
        );
    }

    #[test]
    fn load_rejects_zero_kv_heads() {
        // num_key_value_heads == 0 would integer-divide-by-zero deriving
        // num_kv_groups (num_attention_heads / num_key_value_heads) and panic in
        // attention load. The guard rejects it loud, up front. (vb is built from a
        // valid cfg; the guard fires before any tensor is touched — and a kv==0
        // weight_map would itself build zero-row kv projections.)
        let good = tiny_cfg();
        let vb = tiny_vb(&good);
        let mut bad = tiny_cfg();
        bad.num_key_value_heads = 0;
        let err = LlamaGradModel::load(&bad, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string().contains("num_key_value_heads must be"),
            "expected a num_key_value_heads>=1 rejection, got: {err}"
        );
    }

    #[test]
    fn load_rejects_indivisible_gqa() {
        // num_attention_heads not divisible by num_key_value_heads truncates
        // num_kv_groups, so repeat_kv expands the KV heads to a count that no
        // longer matches the Q heads — a degenerate non-parity model. The load
        // must fail loud (mirrors the head_dim divisibility guard). hidden_size 8
        // stays divisible by 4 heads (head_dim 2) so the head_dim guard passes
        // and the GQA-divisibility guard is what fires.
        let mut cfg = tiny_cfg();
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 3; // 4 % 3 != 0
        let vb = tiny_vb(&cfg);
        let err = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string()
                .contains("not divisible by num_key_value_heads"),
            "expected a GQA divisibility rejection, got: {err}"
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
    fn assert_legacy_frozen(layer: &DenseLayer) {
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
    fn industrial_layer_vars(layer: &DenseLayer) -> Vec<Var> {
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

    // ---- activation checkpointing (P7) --------------------------------------

    /// A fixed non-uniform probe loss over the logits — no gradient cancels
    /// by symmetry.
    fn probe_loss(logits: &Tensor) -> Tensor {
        let n = logits.elem_count();
        let w: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.21 - 0.6).collect();
        let w = Tensor::from_vec(w, logits.dims().to_vec(), logits.device()).unwrap();
        logits.mul(&w).unwrap().sum_all().unwrap()
    }

    /// Checkpointing on the second architecture: stitched gradients match the
    /// uncut backward on every adapter var, and a raw `loss.backward()` after
    /// a checkpointed forward reaches none (the tape really is cut).
    #[test]
    fn checkpointed_gradients_match_the_uncut_backward() {
        let cfg = tiny_cfg();
        let mut model = LlamaGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        arm_adapter(&model);
        let input = ids(6);
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
        let cfg = tiny_cfg();
        let mut model = LlamaGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
        model.set_activation_checkpointing(true);
        let _ = model.forward_detached(&ids(4)).unwrap();
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

    /// The narrowed scoring forward on the second architecture: values and
    /// adapter gradients exactly match `forward` + narrow (plain and
    /// checkpointed), and the narrowed detached walk captures no tape.
    #[test]
    fn narrowed_forward_matches_the_full_walk_exactly() {
        let cfg = tiny_cfg();
        let mut model = LlamaGradModel::load(&cfg, &tiny_vb(&cfg), 2, 4.0).unwrap();
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
}
