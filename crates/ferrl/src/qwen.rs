//! The Qwen3 binding of the shared dense backbone ([`crate::dense`]).
//!
//! candle-transformers ships a Qwen3 forward, but it is inference-shaped
//! (`&mut self` + `ConcatKvCache`, all layer types `pub(crate)`) and built from
//! ops that have **no backward**, so it cannot be used to train. The grad-bearing
//! *update* path — a full-sequence, uncached forward over the same loaded
//! weights, expressed entirely in grad-bearing ops with a manual `LoRA` adapter
//! — lives in [`crate::dense`] now, shared with the dense Llama family. This
//! module is the thin Qwen3-specific seam over it: the [`QwenArch`] marker, which
//! validates candle's `qwen3::Config` and distills it into a [`DenseSpec`] (the
//! Qwen3 knobs: per-head q/k norm before `RoPE`, model-dtype SDPA, an explicit
//! `head_dim`, `cfg.hidden_act`, plain `1/θ^(2i/d)` `RoPE`; and the rejected
//! options `attention_bias` / `use_sliding_window`).
//!
//! [`QwenGradModel`] and its cached rollout twin [`MergedDecoder`] are the
//! `Config`-typed aliases of the generic [`DenseGradModel`] /
//! [`DenseCachedDecoder`]; every constructor (`load`, `load_with_adapter_dtype`,
//! `load_with_targets`) resolves to the shared implementation. The forward is
//! gated against candle's shipped forward by the equivalence tests below (same
//! weights → same logits, every position) and a `LoRA`-grad-coverage test.

use candle_core::{DType, Device, Result as CandleResult};
use candle_transformers::models::qwen3::Config;

use crate::blocks::RotaryTables;
use crate::dense::{DenseArch, DenseCachedDecoder, DenseGradModel, DenseSpec};

/// The Qwen3 architecture marker for the shared dense backbone (see
/// [`crate::dense::DenseArch`]). Zero-sized; only its [`DenseArch`] impl matters.
#[derive(Debug, Clone, Copy)]
pub struct QwenArch;

impl DenseArch for QwenArch {
    type Config = Config;

    const LABEL: &'static str = "QwenGradModel";

    fn spec(cfg: &Config, dtype: DType, device: &Device) -> CandleResult<DenseSpec> {
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
        // The head-count invariants the GQA arithmetic relies on. Unlike the llama
        // grad model (which DERIVES head_dim = hidden_size / num_attention_heads),
        // Qwen3 carries an explicit head_dim, so the projection widths are not at
        // risk here; what is, is num_kv_groups = num_attention_heads /
        // num_key_value_heads (the GQA repeat factor). Reject the degenerate configs
        // that would divide-by-zero or silently truncate that quotient — matching the
        // qwen3.5 loader's Config::validate() (which enforces both head counts > 0
        // and GQA divisibility).
        if cfg.num_attention_heads == 0 {
            candle_core::bail!(
                "QwenGradModel: num_attention_heads must be >= 1 (got 0); it is the query \
                 head count and the numerator of num_kv_groups = num_attention_heads / \
                 num_key_value_heads"
            );
        }
        // num_key_value_heads is the divisor that derives num_kv_groups. Zero would
        // integer-divide-by-zero and panic deep in attention load; reject it loud
        // here. Checked before the divisibility guard below so that guard never
        // evaluates is_multiple_of(0) (which is true only when the dividend is 0).
        if cfg.num_key_value_heads == 0 {
            candle_core::bail!(
                "QwenGradModel: num_key_value_heads must be >= 1 (got 0); num_kv_groups is \
                 derived as num_attention_heads / num_key_value_heads"
            );
        }
        // A non-divisible pair truncates num_kv_groups, so repeat_kv would expand the
        // KV heads to a count that no longer matches the Q heads — a degenerate
        // (non-parity) model. Reject it at load, matching the qwen3.5 loader's
        // existing GQA-divisibility guard.
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            candle_core::bail!(
                "QwenGradModel: num_attention_heads {} is not divisible by \
                 num_key_value_heads {} (num_kv_groups is derived as their quotient; such a \
                 config cannot be a real Qwen3 and would silently load as a degenerate model)",
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
            head_dim: cfg.head_dim,
            intermediate_size: cfg.intermediate_size,
            rms_norm_eps: cfg.rms_norm_eps as f32,
            tie_word_embeddings: cfg.tie_word_embeddings,
            // Qwen3 applies per-head q/k RMSNorm before RoPE, keeps the SDPA in the
            // model dtype, and uses the plain 1/theta^(2i/d) RoPE family.
            qk_norm: true,
            sdpa_f32: false,
            activation: cfg.hidden_act,
            // Plain scalars: the tables are architecture-neutral (crate::blocks) and
            // the dtype cast matches the shipped rotary embedding.
            rot: RotaryTables::new(
                cfg.head_dim,
                cfg.rope_theta,
                cfg.max_position_embeddings,
                dtype,
                device,
            )?,
        })
    }
}

/// A grad-bearing, uncached Qwen3 forward with `LoRA` — the [`DenseGradModel`]
/// over the Qwen3 [`QwenArch`]. Built from the same `VarBuilder` (over the same
/// safetensors) as candle's shipped `ModelForCausalLM`, so the two are
/// weight-identical and their logits match (the equivalence gate below).
pub type QwenGradModel = DenseGradModel<QwenArch>;

/// A KV-cached, grad-free Qwen3 decoder over merged weights — the fast rollout
/// twin of [`QwenGradModel`]. The shared [`DenseCachedDecoder`].
pub type MergedDecoder = DenseCachedDecoder;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::LocalComm;
    use crate::dense::DenseLayer;
    use crate::lora::{DenseLoraTargets, Proj};
    use crate::model::GradModel;
    use crate::nn::{grad_coverage, RmsNorm};
    use crate::tensor_parallel::{
        concat_column_shards, sum_row_parallel_partials, TensorParallelPlan,
    };
    use candle_core::backprop::GradStore;
    use candle_core::safetensors;
    use candle_core::{Tensor, Var, D};
    use candle_nn::ops::softmax;
    use candle_nn::rotary_emb::rope_slow;
    use candle_nn::{Activation, VarBuilder};
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
    fn load_rejects_zero_attention_heads() {
        // num_attention_heads == 0 is the query head count and the numerator of
        // num_kv_groups; it loads a degenerate (zero-width q_proj, num_kv_groups 0)
        // model. The guard rejects it loud, up front. (vb is built from a valid
        // cfg; the guard fires before any tensor is touched.)
        let good = tiny_cfg();
        let vb = tiny_vb(&good);
        let mut bad = tiny_cfg();
        bad.num_attention_heads = 0;
        let err = QwenGradModel::load(&bad, &vb, 2, 4.0).unwrap_err();
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
        // valid cfg; the guard fires before any tensor is touched.)
        let good = tiny_cfg();
        let vb = tiny_vb(&good);
        let mut bad = tiny_cfg();
        bad.num_key_value_heads = 0;
        let err = QwenGradModel::load(&bad, &vb, 2, 4.0).unwrap_err();
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
        // must fail loud. (Qwen3 head_dim is an explicit Config field, independent
        // of num_attention_heads, so only the GQA-divisibility guard is at play.)
        let mut cfg = tiny_cfg();
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 3; // 4 % 3 != 0
        let vb = tiny_vb(&cfg);
        let err = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap_err();
        assert!(
            err.to_string()
                .contains("not divisible by num_key_value_heads"),
            "expected a GQA divisibility rejection, got: {err}"
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

    fn arm_adapter_deterministic(model: &QwenGradModel) {
        for (i, v) in model.trainable_vars().iter().enumerate() {
            let dims = v.as_tensor().dims().to_vec();
            let n = dims.iter().product::<usize>();
            let data: Vec<f32> = (0..n)
                .map(|j| 0.03 + i as f32 * 0.004 + j as f32 * 0.002)
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

    fn tiny_tp_gqa_cfg() -> Config {
        let mut cfg = tiny_cfg();
        cfg.hidden_size = 16;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 4;
        cfg.intermediate_size = 16;
        assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 2);
        cfg
    }

    fn assert_rank_tp_projection_grads_live(
        rank: usize,
        vars: &[Var],
        grads: &GradStore,
        cfg: &Config,
    ) {
        let names = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ];
        for layer in 0..cfg.num_hidden_layers {
            for (pair_idx, name) in names.iter().enumerate() {
                let a_idx = layer * 14 + pair_idx * 2;
                let b_idx = a_idx + 1;
                let c = grad_coverage(&vars[a_idx..=b_idx], grads).unwrap();
                assert!(
                    c.is_covered() && c.nonzero == c.total && c.nonfinite == 0,
                    "rank {rank} layer {layer} {name} TP backward grads not fully live: {c:?}"
                );
            }
        }
    }

    #[test]
    fn dense_mlp_tensor_parallel_projection_shards_reassemble() {
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
        arm_adapter(&model);
        let mlp = &model.layers[0].mlp;
        let x = Tensor::from_vec(
            (0..16).map(|i| i as f32 * 0.04 - 0.3).collect::<Vec<_>>(),
            (1, 2, cfg.hidden_size),
            &dev(),
        )
        .unwrap();

        let full_hidden = mlp
            .gate_proj
            .forward(&x)
            .unwrap()
            .apply(&Activation::Silu)
            .unwrap()
            .broadcast_mul(&mlp.up_proj.forward(&x).unwrap())
            .unwrap();
        let full = mlp.down_proj.forward(&full_hidden).unwrap();
        let partials = all_tp_plans(2)
            .into_iter()
            .map(|plan| {
                let gate = mlp
                    .gate_proj
                    .column_parallel_forward(&x, plan, "intermediate_size")?
                    .apply(&Activation::Silu)?;
                let up = mlp
                    .up_proj
                    .column_parallel_forward(&x, plan, "intermediate_size")?;
                let hidden = gate.broadcast_mul(&up)?;
                mlp.down_proj.row_parallel_forward_partial_from_shard(
                    &hidden,
                    plan,
                    "intermediate_size",
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()
            .unwrap();
        let sharded = sum_row_parallel_partials(&partials).unwrap();

        assert_eq!(sharded.dims(), full.dims());
        let worst = max_abs_diff(&sharded, &full);
        assert!(
            worst <= 1e-5,
            "tensor-parallel dense MLP reassembly diverged: {worst}"
        );
    }

    #[test]
    fn dense_attention_tensor_parallel_projection_shards_reassemble() {
        let mut cfg = tiny_cfg();
        cfg.num_key_value_heads = 2;
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
        arm_adapter(&model);
        let attn = &model.layers[0].attn;
        let x = Tensor::from_vec(
            (0..16).map(|i| i as f32 * 0.025 - 0.2).collect::<Vec<_>>(),
            (1, 2, cfg.hidden_size),
            &dev(),
        )
        .unwrap();

        for (name, proj) in [
            ("q_proj", &attn.q_proj),
            ("k_proj", &attn.k_proj),
            ("v_proj", &attn.v_proj),
        ] {
            let full = proj.forward(&x).unwrap();
            let shards = all_tp_plans(2)
                .into_iter()
                .map(|plan| proj.column_parallel_forward(&x, plan, "attention_out"))
                .collect::<candle_core::Result<Vec<_>>>()
                .unwrap();
            let sharded = concat_column_shards(&shards).unwrap();
            let worst = max_abs_diff(&sharded, &full);
            assert!(
                worst <= 1e-5,
                "{name} column shards diverged from full projection: {worst}"
            );
        }

        let ctx = Tensor::from_vec(
            (0..16).map(|i| i as f32 * -0.03 + 0.4).collect::<Vec<_>>(),
            (1, 2, cfg.num_attention_heads * cfg.head_dim),
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
            .collect::<candle_core::Result<Vec<_>>>()
            .unwrap();
        let sharded = sum_row_parallel_partials(&partials).unwrap();
        let worst = max_abs_diff(&sharded, &full);
        assert!(
            worst <= 1e-5,
            "o_proj row partials diverged from full projection: {worst}"
        );
    }

    #[test]
    fn dense_tensor_parallel_collective_forward_matches_unsharded_logits() {
        let mut cfg = tiny_cfg();
        cfg.num_key_value_heads = 2;
        let vb = tiny_vb(&cfg);
        let input = ids(5);
        let reference_model = QwenGradModel::load_with_targets(
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
                        let model = QwenGradModel::load_with_targets(
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
                        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
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
    fn dense_tensor_parallel_collective_gqa_forward_matches_unsharded_logits() {
        let cfg = tiny_tp_gqa_cfg();
        let vb = tiny_vb(&cfg);
        let input = ids(5);
        let reference_model = QwenGradModel::load_with_targets(
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
                        let model = QwenGradModel::load_with_targets(
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
                        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
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
                "rank {rank} GQA TP collective logits diverged from unsharded logits: {worst}"
            );
        }
    }

    #[test]
    fn dense_tensor_parallel_collective_backward_keeps_adapter_grads_live() {
        let cfg = tiny_tp_gqa_cfg();
        let vb = tiny_vb(&cfg);
        let input = ids(5);

        let comms = LocalComm::world(2);
        std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let cfg = cfg.clone();
                    let vb = vb.clone();
                    let input = input.clone();
                    s.spawn(move || {
                        let model = QwenGradModel::load_with_targets(
                            &cfg,
                            &vb,
                            2,
                            4.0,
                            DType::F32,
                            DenseLoraTargets::industrial(),
                        )
                        .unwrap();
                        arm_adapter_deterministic(&model);
                        let vars = model.trainable_vars();
                        let logits = model.forward_tensor_parallel(&input, &comm).unwrap();
                        let loss = probe_loss(&logits);
                        let grads = model.backward(&loss).unwrap();
                        assert_rank_tp_projection_grads_live(rank, &vars, &grads, &cfg);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
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
    /// analytic gradient matches `(L(θ+ε) − L(θ−ε)) / 2ε` on the
    /// strongest entry (max |gradient|) of the first and last adapter vars
    /// (deepest and shallowest stitch). The probe entry must sit far above
    /// the FD noise floor: central differences on a near-zero entry measure
    /// loss-evaluation cancellation (`|L|·εmach/2ε`, CPU-dependent — GitHub's
    /// runner pool measured rel 8.3e-5 on a 4.3e-8-magnitude entry where the
    /// dev host passes 1e-5), not the gradient. Stitching bugs — the class
    /// this gate exists for — are O(1) relative on the strongest entry.
    #[test]
    #[allow(clippy::print_stderr)] // the measured rel is the calibration record
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
            let g = grads.get(var).unwrap().to_vec2::<f64>().unwrap();
            let (r, c) = (0..g.len())
                .flat_map(|i| (0..g[i].len()).map(move |j| (i, j)))
                .max_by(|a, b| g[a.0][a.1].abs().total_cmp(&g[b.0][b.1].abs()))
                .unwrap();
            let analytic = g[r][c];
            assert!(
                analytic.abs() > 1e-6,
                "strongest gradient entry {analytic} is inside the FD noise floor — \
                 the probe is vacuous; re-seed the armed vars"
            );
            let orig = var.as_tensor().to_vec2::<f64>().unwrap();
            let loss_at = |delta: f64, model: &QwenGradModel| -> f64 {
                let mut bent = orig.clone();
                bent[r][c] += delta;
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
            eprintln!("[FD gradcheck] probe ({r},{c}): analytic={analytic:e}, rel={rel:e}");
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
