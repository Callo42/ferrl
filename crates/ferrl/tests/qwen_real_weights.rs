//! Real-weights P3 gates for the custom Qwen3 forward (`#[ignore]`d).
//!
//! These scale PR-A's tiny-config gates to the **real** `Qwen3-0.6B-Base`
//! checkpoint: per-position equivalence vs candle's shipped forward, per-branch
//! `LoRA`-grad coverage **and** liveness on a real backward, and a tokenizer
//! round-trip. The weights are not in the repo (and Hugging Face is unreachable
//! from the build cluster), so the checkpoint is pre-staged out-of-band and
//! located via the `FERRL_QWEN_WEIGHTS` environment variable. Every test here is
//! `#[ignore]`d so CI stays fully offline; run them by hand with the weights
//! present:
//!
//! ```text
//! FERRL_QWEN_WEIGHTS=/path/to/qwen3-0.6b-base \
//!     cargo test -p ferrl --test qwen_real_weights -- --ignored --test-threads=1
//! ```
//!
//! `FERRL_QWEN_WEIGHTS` points at the **directory** holding `config.json`,
//! `model.safetensors`, and `tokenizer.json`. The bf16 checkpoint is loaded
//! upcast to f32, so both forwards run in clean CPU f32 (the bf16 adapter-dtype
//! split — f32 adapter over a bf16 frozen base — is a P4/GPU concern; see
//! `PLAN.md`). `--test-threads=1` keeps at most one f32 copy of the 0.6B weights
//! resident at a time.

use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use ferrl::{grad_coverage, HfTokenizer, QwenGradModel, TokenizerLike};

/// `LoRA` rank / alpha for the gates — small, matching the tiny-config tests.
const RANK: usize = 4;
const ALPHA: f64 = 8.0;

/// Max-abs logit divergence allowed between our forward and the shipped forward,
/// per position, on the real f32 weights. The two paths share bit-identical
/// weights and differ only in numerically-equal op *implementations* (fused vs
/// grad-safe rms-norm / rope / softmax), so the gap is pure f32 rounding-order
/// noise accumulated over 28 layers and the 151 936-wide `lm_head` matmul.
/// Calibrated with headroom: the observed worst-position divergence on the real
/// 0.6B-Base checkpoint is ~2.4e-4 (8 positions), so 2e-3 (~8x) clears f32
/// matmul reduction-order noise while still catching any real parity regression
/// (a broken `RoPE` / `QK-norm` / GQA repeat would diverge by orders of magnitude;
/// the per-op and tiny-model every-position gates in `qwen.rs` pin the building
/// blocks exactly, so a subtle sub-tolerance forward bug has nowhere to hide).
const LOGIT_TOL: f32 = 2e-3;

fn weights_dir() -> PathBuf {
    let dir = std::env::var("FERRL_QWEN_WEIGHTS").expect(
        "set FERRL_QWEN_WEIGHTS to the Qwen3-0.6B-Base asset directory \
         (config.json + model.safetensors + tokenizer.json) to run the ignored \
         real-weights gates",
    );
    PathBuf::from(dir)
}

fn load_config(dir: &Path) -> Config {
    let bytes = std::fs::read(dir.join("config.json")).expect("read config.json");
    serde_json::from_slice(&bytes).expect("parse config.json into qwen3::Config")
}

/// f32 [`VarBuilder`] over the real safetensors (bf16 on disk, upcast on load).
/// Uses the safe buffered loader — `from_mmaped_safetensors` is `unsafe`, which
/// the crate forbids.
fn load_vb(dir: &Path) -> VarBuilder<'static> {
    let buf = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    VarBuilder::from_buffered_safetensors(buf, DType::F32, &Device::Cpu)
        .expect("load model.safetensors")
}

fn ids(seq: &[u32]) -> Tensor {
    Tensor::from_vec(seq.to_vec(), (1, seq.len()), &Device::Cpu).unwrap()
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    a.sub(b)
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

/// Assert `cfg` really is the 0.6B-Base shape — the parity traps PR-A pinned:
/// GQA 16Q/8KV, `head_dim` 128 (so attention width 2048 > hidden 1024), tied head.
fn assert_0p6b_shape(cfg: &Config) {
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert!(cfg.tie_word_embeddings);
}

/// Worst per-position max-abs divergence between our full-sequence logits and
/// the shipped model's last-position logits on each growing prefix `[0..=t]`.
/// Returning the single worst value (rather than asserting per position) lets a
/// calibration run surface the true maximum in one shot; a bound on the max is a
/// bound on every position.
fn worst_divergence(
    shipped: &mut ModelForCausalLM,
    ours_all: &Tensor,
    input: &Tensor,
    seq_len: usize,
) -> f32 {
    let mut worst = 0f32;
    for t in 0..seq_len {
        let prefix = input.narrow(1, 0, t + 1).unwrap();
        shipped.clear_kv_cache();
        let shipped_t = shipped.forward(&prefix, 0).expect("shipped forward"); // [1,1,vocab] @ t
        let ours_t = ours_all.narrow(1, t, 1).unwrap();
        // Exact `sub` (not broadcast) so a shape divergence fails loudly rather
        // than silently broadcasting to a misleadingly-small scalar.
        assert_eq!(
            shipped_t.dims(),
            ours_t.dims(),
            "logit shape mismatch at {t}"
        );
        worst = worst.max(max_abs(&shipped_t, &ours_t));
    }
    worst
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

/// One `sqr().sum()` forward + backward, returning the grad store.
fn grads_of(model: &QwenGradModel, input: &Tensor) -> GradStore {
    let loss = model
        .forward(input)
        .expect("forward")
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    loss.backward().expect("backward")
}

/// Set every `LoRA` `B` factor (the odd index within each `[A, B]` pair) to a
/// small nonzero tensor, so the update is no longer a no-op and `dL/dA` is no
/// longer structurally zero.
fn force_b_nonzero(vars: &[Var]) {
    for (i, v) in vars.iter().enumerate() {
        if i % 2 == 1 {
            let dims = v.as_tensor().dims().to_vec();
            v.set(&Tensor::randn(0f32, 0.02f32, dims, &Device::Cpu).unwrap())
                .unwrap();
        }
    }
}

#[test]
#[ignore = "needs the real Qwen3-0.6B-Base checkpoint via FERRL_QWEN_WEIGHTS"]
fn real_forward_matches_shipped_every_position() {
    let dir = weights_dir();
    let cfg = load_config(&dir);
    assert_0p6b_shape(&cfg);

    let vb = load_vb(&dir);
    let mut shipped = ModelForCausalLM::new(&cfg, vb.clone()).expect("build shipped model");
    let mut ours = QwenGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build our model");
    ours.set_adapter_enabled(false); // base only, like-for-like with shipped

    // Arbitrary in-vocab ids with a repeated token (11 and 1879 recur at distinct
    // positions) to stress causal-mask asymmetry and position-dependent RoPE;
    // equivalence is otherwise content-agnostic.
    let seq: Vec<u32> = vec![9707, 11, 1879, 358, 1879, 11, 1273, 13];
    let input = ids(&seq);
    let ours_all = ours.forward(&input).expect("our forward"); // [1, seq, vocab]
    assert_eq!(ours_all.dims(), &[1, seq.len(), cfg.vocab_size]);

    let worst = worst_divergence(&mut shipped, &ours_all, &input, seq.len());
    assert!(
        worst <= LOGIT_TOL,
        "real-weights logits diverged from shipped: worst max-abs {worst} > {LOGIT_TOL}"
    );
}

#[test]
#[ignore = "needs the real Qwen3-0.6B-Base checkpoint via FERRL_QWEN_WEIGHTS"]
fn real_lora_grads_flow_through_qwen_backward() {
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let vb = load_vb(&dir);
    let mut model = QwenGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build our model");
    model.set_adapter_enabled(true);

    let vars = model.trainable_vars();
    // q/v (A+B) over every layer: 4 trainable vars per layer.
    assert_eq!(vars.len(), cfg.num_hidden_layers * 4);
    let (q_vars, v_vars) = branch_split(&vars);
    let input = ids(&[9707, 11, 1879, 358]);

    // Phase 1 — natural zero-B init: every trainable var must be PRESENT in the
    // grad store (the candle silent-skip canary) and each branch live (via dL/dB)
    // with no non-finite grad. dL/dA is legitimately ~0 at B=0, so liveness of the
    // A factor is NOT asserted here — phase 2 does that.
    let g1 = grads_of(&model, &input);
    let qc = grad_coverage(&q_vars, &g1).unwrap();
    let vc = grad_coverage(&v_vars, &g1).unwrap();
    assert!(
        qc.is_ok(),
        "q-branch grads unhealthy at zero-B init (rms_norm_slow/rope_slow/softmax cut?): {qc:?}"
    );
    assert!(
        vc.is_ok(),
        "v-branch grads unhealthy at zero-B init: {vc:?}"
    );

    // Phase 2 — force every B nonzero so dL/dA is no longer structurally zero; now
    // EVERY var (A and B, across all layers) must carry a nonzero finite grad. A
    // zero-B backward cannot prove this: it catches a severed/dead A-input path or
    // a partial graph break (one mis-wired layer) that leaves A present-but-zero.
    force_b_nonzero(&vars);
    let g2 = grads_of(&model, &input);
    let qc = grad_coverage(&q_vars, &g2).unwrap();
    let vc = grad_coverage(&v_vars, &g2).unwrap();
    assert!(
        qc.nonzero == qc.total && qc.nonfinite == 0,
        "q-branch: not every LoRA var is live after nonzero-B (severed A path?): {qc:?}"
    );
    assert!(
        vc.nonzero == vc.total && vc.nonfinite == 0,
        "v-branch: not every LoRA var is live after nonzero-B: {vc:?}"
    );
}

#[test]
#[ignore = "needs the real Qwen3-0.6B-Base tokenizer via FERRL_QWEN_WEIGHTS"]
fn real_tokenizer_round_trips() {
    let dir = weights_dir();
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

    // Exact round-trip (no trim): a leading/trailing-space regression in a
    // byte-level BPE is exactly where round-trip bugs live, so compare verbatim.
    for prompt in ["The capital of France is", "café — déjà vu, naïve résumé"] {
        let token_ids = tok.encode(prompt);
        assert!(
            !token_ids.is_empty(),
            "encode produced no ids for {prompt:?}"
        );
        assert!(
            token_ids.iter().all(|&i| i < 151936),
            "token id out of Qwen3 vocab range for {prompt:?}"
        );
        assert_eq!(
            tok.decode(&token_ids),
            prompt,
            "tokenizer did not round-trip {prompt:?}"
        );
    }
}
