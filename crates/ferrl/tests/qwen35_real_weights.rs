//! Real-weights M2′ gates for the `qwen3_5` forward (`#[ignore]`d).
//!
//! These scale the committed tiny-oracle gates to the **real**
//! `Qwen3.5-0.8B-Base` checkpoint. candle ships no `qwen3_5`, so the
//! reference is the pinned transformers oracle itself: fp32 per-position
//! logits dumped on the cluster by `scripts/oracle/dump_qwen35_real_logits.py`
//! (`ferrl-oracle` env — transformers 5.11.0, CPU torch 2.12.0), staged next
//! to the checkpoint, never committed (the 248 320-wide vocab rules out JSON
//! fixtures). Run by hand with both staged:
//!
//! ```text
//! FERRL_QWEN35_WEIGHTS=/path/to/qwen3_5-0.8b-base \
//! FERRL_QWEN35_ORACLE=/path/to/qwen3_5-0.8b-base/ferrl_oracle_dumps \
//!     cargo test -p ferrl --test qwen35_real_weights -- --ignored --test-threads=1
//! ```
//!
//! `--test-threads=1` keeps at most one f32 copy of the 0.8B weights resident.
//!
//! What this exercises that the tiny gates cannot: the real geometry (24
//! layers 3:1, 16==16 delta-rule heads → NO GVA broadcast, 8Q/2KV `head_dim`
//! 256, partial rotary 64/256 at theta 1e7, the 248 320-row tied embedding
//! head), the real single-shard-with-index checkpoint layout, and the real
//! tokenizer.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use ferrl::{
    grad_coverage, varbuilder_from_pretrained, HfTokenizer, LayerType, Qwen3_5Config,
    Qwen3_5GradModel, TokenizerLike,
};

/// `LoRA` rank / alpha for the gates — small, matching the tiny-config tests.
const RANK: usize = 4;
const ALPHA: f64 = 8.0;

/// Max-abs per-position logit divergence allowed between our f32 forward and
/// the transformers fp32 dump. Same envelope family as the M1 gates (Qwen3
/// 0.6B/28 layers measured worst 2.4e-4 under tol 2e-3; Llama 1B/16 layers
/// 1.7e-5 under 5e-4) — here the stack is 24 hybrid layers and the dump
/// crosses *implementations* (torch vs candle), not just op families.
/// Measured worst on the real checkpoint (2026-06-11, after the `ut_solve`
/// stability fix): 4.5e-5 across the three prompts, FLAT at every position
/// (the per-position profile this gate prints is what caught the original
/// explicit-inverse instability — error doubling per position to 1.1e-2 by
/// t=22) → 1e-3 keeps ~22x headroom for cross-host reduction-order noise
/// while staying orders of magnitude under any real parity break (the
/// tiny-oracle planted bugs land at 4.9–12.5).
const LOGIT_TOL: f32 = 1e-3;

/// Cached (merged-decoder) vs our own uncached forward on real weights — same
/// ops both sides, the chunked/recurrent kernel boundary dominates.
/// Measured worst (2026-06-11): 3.3e-5 (prefill+chunk+decode trio) → ~30x.
const MERGED_TOL: f32 = 1e-3;

fn weights_dir() -> PathBuf {
    PathBuf::from(std::env::var("FERRL_QWEN35_WEIGHTS").expect(
        "set FERRL_QWEN35_WEIGHTS to the Qwen3.5-0.8B-Base asset directory \
         (config.json + model.safetensors.index.json + shards + tokenizer.json) \
         to run the ignored real-weights gates",
    ))
}

fn oracle_dir() -> PathBuf {
    PathBuf::from(std::env::var("FERRL_QWEN35_ORACLE").expect(
        "set FERRL_QWEN35_ORACLE to the directory holding real_logits.safetensors \
         (scripts/oracle/dump_qwen35_real_logits.py output)",
    ))
}

/// The staged oracle dump: per-prompt token ids (i64 in the file) and fp32
/// per-position logits.
fn load_oracle() -> HashMap<String, Tensor> {
    candle_core::safetensors::load(oracle_dir().join("real_logits.safetensors"), &Device::Cpu)
        .expect("load real_logits.safetensors")
}

fn oracle_ids(dump: &HashMap<String, Tensor>, i: usize) -> Vec<u32> {
    dump[&format!("p{i}_ids")]
        .to_dtype(DType::U32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Assert `cfg` really is the 0.8B-Base geometry — the parity-relevant traps.
fn assert_0_8b_shape(cfg: &Qwen3_5Config) {
    let t = &cfg.text_config;
    assert_eq!(
        (
            t.hidden_size,
            t.num_hidden_layers,
            t.num_attention_heads,
            t.num_key_value_heads,
            t.head_dim,
            t.vocab_size,
        ),
        (1024, 24, 8, 2, 256, 248_320),
        "not the Qwen3.5-0.8B-Base geometry"
    );
    assert_eq!(t.rotary_dim(), 64);
    assert!(t.tie_word_embeddings);
    // 16 == 16: no GVA broadcast on the real model (the tiny fixture covers
    // the broadcast; here the ratio must resolve to 1).
    assert_eq!(t.linear_num_value_heads, 16);
    assert_eq!(t.linear_num_key_heads, 16);
    assert_0_8b_layer_pattern(cfg);
}

/// The 3:1 layer pattern half of [`assert_0_8b_shape`].
fn assert_0_8b_layer_pattern(cfg: &Qwen3_5Config) {
    let kinds = cfg.text_config.resolved_layer_types();
    assert_eq!(kinds.len(), 24);
    let full = kinds
        .iter()
        .filter(|k| **k == LayerType::FullAttention)
        .count();
    assert_eq!(full, 6);
}

fn load_model() -> (Qwen3_5Config, Qwen3_5GradModel) {
    let dir = weights_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).expect("parse config.json");
    assert_0_8b_shape(&cfg);
    // f32 upcast on load (bf16 on disk); the bf16 path is the GPU gate.
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).expect("load weights");
    let model = Qwen3_5GradModel::load(&cfg, &vb, RANK, ALPHA).expect("build model");
    (cfg, model)
}

fn ids(seq: &[u32]) -> Tensor {
    Tensor::from_vec(seq.to_vec(), (1, seq.len()), &Device::Cpu).unwrap()
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    // Exact `sub` (not broadcast): a shape divergence must fail loudly.
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

#[test]
#[ignore = "needs the staged 0.8B checkpoint + oracle dumps (FERRL_QWEN35_WEIGHTS/_ORACLE)"]
#[allow(clippy::print_stderr)] // a manual gate: the measured worst divergence is the deliverable
fn real_forward_matches_reference_every_position() {
    let (_cfg, mut model) = load_model();
    model.set_adapter_enabled(false); // base only, like-for-like with the dump
    let dump = load_oracle();
    let mut worst = 0f32;
    let mut i = 0usize;
    while dump.contains_key(&format!("p{i}_ids")) {
        let seq = oracle_ids(&dump, i);
        let want = &dump[&format!("p{i}_logits")];
        let got = model.forward(&ids(&seq)).expect("our forward");
        assert_eq!(got.dims(), want.dims(), "logit shape mismatch on p{i}");
        let d = max_abs(&got, want);
        // Per-position profile: distinguishes one outlier position (a near-tie
        // / scale artifact) from systematic position-dependent growth (a rope
        // table or state-decay break that short fixtures cannot see).
        for t in 0..seq.len() {
            let g_row = got.narrow(1, t, 1).unwrap();
            let w_row = want.narrow(1, t, 1).unwrap();
            let dt = max_abs(&g_row, &w_row);
            let scale = w_row
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!(
                "  p{i} t={t}: max-abs {dt:e} scale {scale:.2} rel {:e}",
                dt / scale
            );
        }
        eprintln!(
            "[qwen35 real-weights gate] p{i} ({} tokens): max-abs divergence {d:e}",
            seq.len()
        );
        worst = worst.max(d);
        i += 1;
    }
    assert!(i >= 3, "expected >= 3 dumped prompts, found {i}");
    eprintln!(
        "[qwen35 real-weights gate] worst per-position max-abs divergence over {i} prompts: \
         {worst:e} (tol {LOGIT_TOL:e})"
    );
    assert!(
        worst <= LOGIT_TOL,
        "real-weights logits diverged from the transformers reference: {worst} > {LOGIT_TOL}"
    );
}

#[test]
#[ignore = "needs the staged 0.8B checkpoint + oracle dumps (FERRL_QWEN35_WEIGHTS/_ORACLE)"]
#[allow(clippy::print_stderr)] // a manual gate: the measured worst divergence is the deliverable
fn real_merged_decoder_matches_uncached() {
    // Prefill -> multi-token continuation at an offset -> single-token decode,
    // on the REAL geometry (conv_dim 6144, S [1,16,128,128] f32 per layer) —
    // vs our own uncached forward. The tiny-oracle gate pins the same trio
    // against the reference's cached execution; this scales the state
    // lifecycle to real shapes.
    let (_cfg, mut model) = load_model();
    model.set_adapter_enabled(false);
    let dump = load_oracle();
    let seq = oracle_ids(&dump, 0);
    let input = ids(&seq);
    let uncached = model.forward(&input).expect("uncached forward");

    let mut dec = model.merged_decoder().expect("merged decoder");
    let p = seq.len() / 2;
    let c = seq.len() - p - 2;
    let mut parts = vec![
        dec.forward(&input.narrow(1, 0, p).unwrap(), 0).unwrap(),
        dec.forward(&input.narrow(1, p, c).unwrap(), p).unwrap(),
    ];
    for t in (p + c)..seq.len() {
        parts.push(dec.forward(&input.narrow(1, t, 1).unwrap(), t).unwrap());
    }
    let cached = Tensor::cat(&parts, 1).unwrap();
    let d = max_abs(&cached, &uncached);
    eprintln!(
        "[qwen35 real-weights gate] cached (prefill {p} + chunk {c} + 2 decodes) vs uncached: \
         max-abs {d:e} (tol {MERGED_TOL:e})"
    );
    assert!(
        d <= MERGED_TOL,
        "cached rollout diverged from uncached: {d}"
    );
}

#[test]
#[ignore = "needs the staged 0.8B checkpoint + oracle dumps (FERRL_QWEN35_WEIGHTS/_ORACLE)"]
fn real_lora_grads_flow_through_qwen35_backward() {
    let (cfg, mut model) = load_model();
    model.set_adapter_enabled(true);
    let vars = model.trainable_vars();
    // Default recipe: MLP (3 projs) on all 24 layers + attention (4 projs) on
    // the 6 full layers; 2 vars per projection.
    assert_eq!(vars.len(), (24 * 3 + 6 * 4) * 2);
    let (attn_vars, mlp_vars) = branch_split(&cfg, &vars);

    let dump = load_oracle();
    let seq = oracle_ids(&dump, 0);
    let input = ids(&seq[..seq.len().min(8)]);

    // Phase 1 — zero-B init: every var present, each branch live, all finite.
    let g1 = grads_of(&model, &input);
    assert!(
        grad_coverage(&attn_vars, &g1).unwrap().is_ok(),
        "attention branch unhealthy at zero-B init on real weights"
    );
    assert!(
        grad_coverage(&mlp_vars, &g1).unwrap().is_ok(),
        "mlp branch unhealthy at zero-B init on real weights"
    );

    // Phase 2 — nonzero B: EVERY A and B live + finite.
    force_b_nonzero(&vars);
    let g2 = grads_of(&model, &input);
    let ac = grad_coverage(&attn_vars, &g2).unwrap();
    let mc = grad_coverage(&mlp_vars, &g2).unwrap();
    assert!(
        ac.nonzero == ac.total && ac.nonfinite == 0,
        "attention branch: not every LoRA var live after nonzero-B: {ac:?}"
    );
    assert!(
        mc.nonzero == mc.total && mc.nonfinite == 0,
        "mlp branch: not every LoRA var live after nonzero-B: {mc:?}"
    );
}

#[test]
#[ignore = "needs the staged 0.8B checkpoint + oracle dumps (FERRL_QWEN35_WEIGHTS/_ORACLE)"]
fn real_tokenizer_round_trips_and_matches_dump() {
    let dir = weights_dir();
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let dump = load_oracle();
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(oracle_dir().join("meta.json")).expect("read meta.json"),
    )
    .unwrap();
    assert_eq!(meta["transformers"].as_str().unwrap(), "5.11.0");
    let prompts = meta["prompts"].as_array().unwrap();
    for (i, p) in prompts.iter().enumerate() {
        let prompt = p.as_str().unwrap();
        let our_ids = tok.encode(prompt);
        // The equivalence gates consume the DUMPED ids, so they hold without
        // this — but rollout tokenizes with OUR tokenizer, so it must agree
        // with what the reference tokenizer produced.
        assert_eq!(
            our_ids,
            oracle_ids(&dump, i),
            "tokenizer ids diverge from the reference dump for {prompt:?}"
        );
        assert!(our_ids.iter().all(|&id| id < 248_320));
        assert_eq!(tok.decode(&our_ids), prompt, "round-trip failed");
    }
}

/// Split the default-recipe vars into (attention, mlp) branches using the
/// config's layer kinds: linear layers contribute 6 MLP vars; full layers 8
/// attention vars then 6 MLP vars (the documented deterministic order).
fn branch_split(cfg: &Qwen3_5Config, vars: &[Var]) -> (Vec<Var>, Vec<Var>) {
    let mut attn = Vec::new();
    let mut mlp = Vec::new();
    let mut i = 0usize;
    for kind in cfg.text_config.resolved_layer_types() {
        if kind == LayerType::FullAttention {
            attn.extend(vars[i..i + 8].iter().cloned());
            i += 8;
        }
        mlp.extend(vars[i..i + 6].iter().cloned());
        i += 6;
    }
    assert_eq!(i, vars.len(), "branch split must consume every var");
    (attn, mlp)
}

/// One `sqr().sum()` forward + backward, returning the grad store.
fn grads_of(model: &Qwen3_5GradModel, input: &Tensor) -> GradStore {
    model
        .forward(input)
        .expect("forward")
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .expect("backward")
}

/// Set every `LoRA` `B` factor (odd index in each `[A, B]` pair) to small
/// DETERMINISTIC noise (a phase-2 grad failure must be replayable — candle's
/// CPU `randn` cannot be seeded, so build the values directly).
fn force_b_nonzero(vars: &[Var]) {
    for (i, v) in vars.iter().enumerate() {
        if i % 2 == 1 {
            let dims = v.as_tensor().dims().to_vec();
            let n: usize = dims.iter().product();
            let vals: Vec<f32> = (0..n)
                .map(|e| 0.02 * ((e + i) as f32 * 0.618_034).sin())
                .collect();
            v.set(&Tensor::from_vec(vals, dims, &Device::Cpu).unwrap())
                .unwrap();
        }
    }
}
