//! Real-weights M1 gates for the custom dense-Llama forward (`#[ignore]`d).
//!
//! These scale `llama.rs`'s tiny-config gates to the **real** `Llama-3.2-1B`
//! checkpoint: per-position equivalence vs candle's shipped `llama::Llama`,
//! per-branch `LoRA`-grad coverage **and** liveness on a real backward, and a
//! tokenizer round-trip — the same bar `qwen_real_weights.rs` holds for the
//! first [`GradModel`](ferrl::GradModel) implementor. The weights are not in
//! the repo (and Hugging Face is unreachable from the build cluster), so the
//! checkpoint is pre-staged out-of-band and located via the
//! `FERRL_LLAMA_WEIGHTS` environment variable. Every test here is `#[ignore]`d
//! so CI stays fully offline; run them by hand with the weights present:
//!
//! ```text
//! FERRL_LLAMA_WEIGHTS=/path/to/llama-3.2-1b \
//!     cargo test -p ferrl --test llama_real_weights -- --ignored --test-threads=1
//! ```
//!
//! `FERRL_LLAMA_WEIGHTS` points at the **directory** holding `config.json`,
//! `model.safetensors`, and `tokenizer.json`. The bf16 checkpoint is loaded
//! upcast to f32, so both forwards run in clean CPU f32 (the bf16 path is the
//! separate GPU gate — `llama_gpu_smoke.rs`). `--test-threads=1` keeps at most
//! one f32 copy of the 1B weights resident at a time.
//!
//! What this exercises that the tiny-config gates cannot: the REAL llama3
//! `RoPE`-scaling regime (`rope_theta` 5e5, factor 32, `original_max` 8192 —
//! all three smoothing branches populated across the 32 inv-freqs, at
//! `max_position_embeddings` 131 072), a genuinely tied 128 256-row LM head,
//! and 16 layers of real-scale GQA (32Q/8KV, derived `head_dim` 64).

use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{
    Cache, Config, Llama, Llama3RopeType, LlamaConfig, LlamaEosToks,
};
use ferrl::{grad_coverage, HfTokenizer, LlamaGradModel, TokenizerLike};

/// `LoRA` rank / alpha for the gates — small, matching the tiny-config tests.
const RANK: usize = 4;
const ALPHA: f64 = 8.0;

/// Max-abs logit divergence allowed between our forward and the shipped
/// forward, per position, on the real f32 weights. The two paths share
/// bit-identical weights and differ only in numerically-equal op
/// *implementations* (fused vs grad-safe rms-norm / rope / softmax), so the
/// gap is pure f32 rounding-order noise accumulated over 16 layers and the
/// 128 256-wide tied-head matmul. Calibrated with headroom: the measured
/// worst-position divergence on the real Llama-3.2-1B checkpoint is 1.7e-5
/// (12 positions), so 5e-4 (~30x) clears f32 matmul reduction-order noise
/// across hosts (the gate runs on whichever cluster CPU is free — the P2
/// platform lesson) while still catching any real parity regression (a
/// broken llama3 `RoPE`-scaling branch / GQA repeat / tied head would
/// diverge by orders of magnitude; the inv-freq pins and tiny-model
/// every-position gates in `llama.rs` pin the building blocks exactly, so a
/// subtle sub-tolerance forward bug has nowhere to hide). Same envelope
/// family as the Qwen real-weights gate (worst 2.4e-4, tol 2e-3, at
/// 0.6B/28 layers — Llama's floor is lower: 16 layers, not 28).
const LOGIT_TOL: f32 = 5e-4;

/// A short real prompt with repeated words ("the", "cat") at distinct
/// positions, to stress causal-mask asymmetry and position-dependent (scaled)
/// `RoPE`; equivalence is otherwise content-agnostic. Kept modest: every
/// prefix forward materializes the full 128 256-wide logit row.
const EQ_PROMPT: &str = "The cat sat on the mat, and the cat slept.";

fn weights_dir() -> PathBuf {
    let dir = std::env::var("FERRL_LLAMA_WEIGHTS").expect(
        "set FERRL_LLAMA_WEIGHTS to the Llama-3.2-1B asset directory \
         (config.json + model.safetensors + tokenizer.json) to run the ignored \
         real-weights gates",
    );
    PathBuf::from(dir)
}

/// Parse `config.json` through candle's serde mirror (`LlamaConfig`), exactly
/// as a candle consumer would: unknown HF fields (`head_dim`, `mlp_bias`,
/// `pretraining_tp`) are ignored by serde, `num_key_value_heads` defaults to
/// MHA when absent, and `eos_token_id` deserializes into the
/// `Option<LlamaEosToks>` Single/Multiple union. Non-flash (the only path the
/// grad forward implements, and the only one with CPU parity).
fn load_config(dir: &Path) -> Config {
    let bytes = std::fs::read(dir.join("config.json")).expect("read config.json");
    let cfg: LlamaConfig = serde_json::from_slice(&bytes).expect("parse config.json");
    cfg.into_config(false)
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

/// Assert `cfg` really is the Llama-3.2-1B shape — the parity-relevant traps:
/// GQA 32Q/8KV with **derived** `head_dim` 64 (the config's own `head_dim`
/// field is ignored by candle's deserializer and must agree), tied 128 256-row
/// head, llama3 `RoPE` scaling (factor 32 over `original_max` 8192), and the
/// scalar `eos_token_id` parsed into `LlamaEosToks::Single`.
fn assert_1b_shape(cfg: &Config) {
    assert_eq!(
        (
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.vocab_size,
            cfg.max_position_embeddings,
        ),
        (2048, 16, 32, 8, 128_256, 131_072),
        "not the Llama-3.2-1B geometry"
    );
    assert_eq!(cfg.hidden_size / cfg.num_attention_heads, 64); // derived head_dim
    assert!(cfg.tie_word_embeddings);
    assert_1b_rope_and_eos(cfg);
}

/// The parse-sensitive halves of [`assert_1b_shape`]: the llama3
/// `rope_scaling` block and the Single/Multiple `eos_token_id` union.
fn assert_1b_rope_and_eos(cfg: &Config) {
    let rs = cfg
        .rope_scaling
        .as_ref()
        .expect("Llama-3.2-1B carries a llama3 rope_scaling block");
    assert!(
        matches!(rs.rope_type, Llama3RopeType::Llama3),
        "rope_type must parse as Llama3"
    );
    assert!((rs.factor - 32.0).abs() < 1e-6);
    assert_eq!(rs.original_max_position_embeddings, 8192);
    match &cfg.eos_token_id {
        Some(LlamaEosToks::Single(id)) => assert_eq!(*id, 128_001),
        other => panic!("expected the scalar eos 128001 to parse as Single, got {other:?}"),
    }
}

/// Worst per-position max-abs divergence between our full-sequence logits and
/// the shipped model's last-position logits on each growing prefix `[0..=t]`.
/// Returning the single worst value (rather than asserting per position) lets a
/// calibration run surface the true maximum in one shot; a bound on the max is
/// a bound on every position. The shipped `Llama::forward` needs a `&mut
/// Cache` even uncached (causal masks are memoized in it) and returns
/// last-position-only `[1, vocab]` logits already cast to F32.
fn worst_divergence(shipped: &Llama, cache: &mut Cache, ours_all: &Tensor, input: &Tensor) -> f32 {
    let seq_len = input.dims2().unwrap().1;
    let mut worst = 0f32;
    for t in 0..seq_len {
        let prefix = input.narrow(1, 0, t + 1).unwrap();
        let shipped_t = shipped.forward(&prefix, 0, cache).expect("shipped forward");
        let ours_t = ours_all.narrow(1, t, 1).unwrap().squeeze(1).unwrap();
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
fn grads_of(model: &LlamaGradModel, input: &Tensor) -> GradStore {
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
#[ignore = "needs the real Llama-3.2-1B checkpoint via FERRL_LLAMA_WEIGHTS"]
#[allow(clippy::print_stderr)] // a manual gate: the measured worst divergence is the deliverable
fn real_forward_matches_shipped_every_position() {
    let dir = weights_dir();
    let cfg = load_config(&dir);
    assert_1b_shape(&cfg);

    let vb = load_vb(&dir);
    // The shipped loader REUSES the embedding as the tied head; ours must make
    // the same choice over the same VarBuilder (the shared weight map carries
    // no `lm_head.weight` for this checkpoint, so a wrong branch fails loud).
    let shipped = Llama::load(vb.clone(), &cfg).expect("build shipped model");
    let mut ours = LlamaGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build our model");
    ours.set_adapter_enabled(false); // base only, like-for-like with shipped

    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let seq = tok.encode(EQ_PROMPT);
    assert!(
        seq.len() >= 8,
        "expected a multi-token prompt, got {} ids",
        seq.len()
    );
    let input = ids(&seq);
    let ours_all = ours.forward(&input).expect("our forward"); // [1, seq, vocab]
    assert_eq!(ours_all.dims(), &[1, seq.len(), cfg.vocab_size]);

    let mut cache = Cache::new(false, DType::F32, &cfg, &Device::Cpu).expect("shipped cache");
    let worst = worst_divergence(&shipped, &mut cache, &ours_all, &input);
    eprintln!(
        "[llama real-weights gate] worst per-position max-abs logit divergence \
         vs shipped over {} positions: {worst:e} (tol {LOGIT_TOL:e})",
        seq.len()
    );
    assert!(
        worst <= LOGIT_TOL,
        "real-weights logits diverged from shipped: worst max-abs {worst} > {LOGIT_TOL}"
    );
}

#[test]
#[ignore = "needs the real Llama-3.2-1B checkpoint via FERRL_LLAMA_WEIGHTS"]
fn real_lora_grads_flow_through_llama_backward() {
    let dir = weights_dir();
    let cfg = load_config(&dir);
    let vb = load_vb(&dir);
    let mut model = LlamaGradModel::load(&cfg, &vb, RANK, ALPHA).expect("build our model");
    model.set_adapter_enabled(true);

    let vars = model.trainable_vars();
    // q/v (A+B) over every layer: 4 trainable vars per layer.
    assert_eq!(vars.len(), cfg.num_hidden_layers * 4);
    let (q_vars, v_vars) = branch_split(&vars);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let input = ids(&tok.encode("The capital of France is"));

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
#[ignore = "needs the real Llama-3.2-1B tokenizer via FERRL_LLAMA_WEIGHTS"]
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
            token_ids.iter().all(|&i| i < 128_256),
            "token id out of Llama-3.2 vocab range for {prompt:?}"
        );
        assert_eq!(
            tok.decode(&token_ids),
            prompt,
            "tokenizer did not round-trip {prompt:?}"
        );
    }
}
