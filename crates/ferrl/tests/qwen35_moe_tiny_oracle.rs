//! The tiny-oracle architecture gate for the `qwen3_5_moe` forward — the
//! `MoE` twin of `qwen35_tiny_oracle.rs`.
//!
//! candle ships no `qwen3_5_moe`, so `tests/fixtures/tiny_qwen35_moe/` is a
//! COMMITTED checkpoint built and executed by the real HF reference
//! (`Qwen3_5MoeForConditionalGeneration`, transformers pinned 5.11.0, fp32
//! CPU, the EAGER packed-weight experts path — the exact code `src/moe.rs` is
//! ported from; the generator pins the dispatch and records the
//! grouped-mm-vs-eager delta in the meta). Every layer's feed-forward is the
//! sparse block: 8 routed experts top-2 + a shared expert, over both mixer
//! kinds. These gates run our loader + forward + merged decoder against the
//! reference's own per-position logits, offline, on every CI run.
//!
//! Generator: `scripts/oracle/gen_qwen35_moe_tiny_golden.py` (version-pinned;
//! regenerate only when the oracle pin moves).

use candle_core::{DType, Device, Tensor};
use ferrl::{
    tensors_from_pretrained, varbuilder_from_pretrained, MoeDims, Qwen3_5Config, Qwen3_5GradModel,
};
use std::path::PathBuf;

/// Our fp32 forward vs the reference's fp32 logits, per position. Measured
/// worst on the dev host (2026-06-12, printed by the uncached gate as the
/// calibration record): 2.84e-4 (`full_b1`), 3.5e-5 (`full_b2`) — ~9x the
/// dense twin's floor; the sparse paths amplify gemm reassociation through
/// the renormalized routing weights. Set ~10x above the measured worst:
/// GitHub's runner pool mixes CPU generations measured ~10x off the dev
/// host on this family's paths (the 2026-06-12 `MERGED_TOL` incident), and
/// a thin-headroom envelope WILL flake there. Still ~16x below
/// `PERTURBATION_FLOOR`, so the gate cannot silently go vacuous.
const ORACLE_TOL: f32 = 3e-3;

/// How far a deliberately mis-built model must land from the oracle — the
/// permanent vacuity guard, here on the MoE-specific axis: a wrong top-k.
const PERTURBATION_FLOOR: f32 = 5e-2;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen35_moe")
}

fn golden() -> serde_json::Value {
    let raw = std::fs::read_to_string(fixture_dir().join("golden.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    // The version pin is asserted IN the gate, not just in the generator (a
    // fixture regenerated under a drifted oracle silently re-baselines
    // everything).
    assert_eq!(
        v["meta"]["transformers"].as_str().unwrap(),
        "5.11.0",
        "fixture must come from the pinned oracle transformers"
    );
    assert!(
        v["meta"]["torch"].as_str().unwrap().starts_with("2.12.0"),
        "fixture must come from the pinned oracle torch"
    );
    // The fixture must pin the EAGER experts path (the one src/moe.rs ports);
    // an auto-dispatched regeneration would silently re-baseline against a
    // different kernel.
    assert_eq!(
        v["meta"]["experts_implementation"].as_str().unwrap(),
        "eager",
        "fixture must pin the eager experts path"
    );
    // Routing sharpness, re-asserted on every run (the kernel-fixture
    // lesson): a regenerated fixture whose routing went uniform would
    // silence the top-k axis and make torch-vs-candle tie-breaking
    // load-bearing.
    assert!(
        v["meta"]["routing_sharpness"].as_f64().unwrap() > 0.05,
        "degenerate fixture routing"
    );
    v
}

fn tensor_from(case: &serde_json::Value, key: &str, dims: (usize, usize, usize)) -> Tensor {
    let v: Vec<f32> = case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(v.len(), dims.0 * dims.1 * dims.2, "{key} length");
    Tensor::from_vec(v, dims, &Device::Cpu).unwrap()
}

fn input_ids(case: &serde_json::Value, key: &str) -> Tensor {
    let rows: Vec<Vec<u32>> = case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
                .collect()
        })
        .collect();
    let b = rows.len();
    let l = rows[0].len();
    let flat: Vec<u32> = rows.into_iter().flatten().collect();
    Tensor::from_vec(flat, (b, l), &Device::Cpu).unwrap()
}

fn load_model() -> (Qwen3_5Config, Qwen3_5GradModel) {
    let dir = fixture_dir();
    // Parsing the file transformers itself wrote IS part of the gate: the
    // MoE serde shape (no intermediate_size, the quartet present, the
    // qwen3_5_moe model_type) and its validation.
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load(&cfg, &vb, 4, 8.0).unwrap();
    (cfg, model)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "logit shape mismatch");
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
fn fixture_config_resolves_as_a_sparse_member() {
    let (cfg, _model) = load_model();
    let t = &cfg.text_config;
    assert_eq!(t.num_hidden_layers, 4);
    assert!(t.intermediate_size.is_none(), "the MoE family deletes it");
    assert_eq!(
        t.moe().unwrap(),
        MoeDims {
            num_experts: 8,
            top_k: 2,
            moe_intermediate_size: 6,
            shared_expert_intermediate_size: 12,
        }
    );
    assert_eq!(cfg.model_type.as_deref(), Some("qwen3_5_moe"));
}

#[test]
#[allow(clippy::print_stderr)] // the measured diffs are the calibration record
fn uncached_forward_matches_reference_logits() {
    let g = golden();
    let (_cfg, model) = load_model();
    for case_name in ["full_b1", "full_b2"] {
        let case = &g["cases"][case_name];
        let ids = input_ids(case, "input_ids");
        let (b, l) = ids.dims2().unwrap();
        let want = tensor_from(case, "logits", (b, l, 64));
        let got = model.forward(&ids).unwrap();
        let d = max_abs_diff(&got, &want);
        eprintln!("[moe tiny oracle] {case_name}: diff {d:e}");
        assert!(
            d <= ORACLE_TOL,
            "{case_name}: our forward vs transformers reference diff {d}"
        );
        // Signal floor: degenerate logits would make the tolerance above
        // meaningless.
        let scale: f32 = want
            .abs()
            .unwrap()
            .max(2)
            .unwrap()
            .min(1)
            .unwrap()
            .min(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            scale > 0.5,
            "{case_name}: degenerate oracle logits ({scale})"
        );
    }
}

/// The full-fine-tuning load over the `MoE` fixture: value-transparent (same
/// reference parity as the `LoRA`-mode load) with the per-expert checkpoint
/// tensors packed into single 3-D vars (the structural full-FT `MoE`
/// contract — per-expert vars would leave the packed forward tensors stale
/// under optimizer updates).
#[test]
fn full_ft_load_matches_reference_logits() {
    let g = golden();
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let tensors = tensors_from_pretrained(&dir, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load_full_ft(&cfg, tensors, DType::F32, &Device::Cpu).unwrap();
    assert!(model.is_full_ft());
    // E=8, m=6, h=16: one [8, 12, 16] gate_up var and one [8, 16, 6] down
    // var per layer — packed, not per-expert.
    let vars = model.trainable_vars();
    let packed_gate_up = vars.iter().filter(|v| v.dims() == [8, 12, 16]).count();
    let packed_down = vars.iter().filter(|v| v.dims() == [8, 16, 6]).count();
    assert_eq!(packed_gate_up, 4, "one packed gate_up var per layer");
    assert_eq!(packed_down, 4, "one packed down var per layer");

    let case = &g["cases"]["full_b1"];
    let ids = input_ids(case, "input_ids");
    let (b, l) = ids.dims2().unwrap();
    let want = tensor_from(case, "logits", (b, l, 64));
    let got = model.forward(&ids).unwrap();
    let d = max_abs_diff(&got, &want);
    assert!(d <= ORACLE_TOL, "full-FT forward vs reference diff {d}");
}

#[test]
fn merged_decoder_matches_reference_cached_path() {
    let g = golden();
    let (_cfg, model) = load_model();
    let case = &g["cases"]["cached_split"];
    let seq = input_ids(case, "input_ids"); // [1, 14]
    let p = case["prefill_len"].as_u64().unwrap() as usize; // 7
    let c = case["chunk_len"].as_u64().unwrap() as usize; // 5

    let want_full = tensor_from(case, "uncached_logits", (1, 14, 64));
    let got_full = model.forward(&seq).unwrap();
    let d_full = max_abs_diff(&got_full, &want_full);
    assert!(d_full <= ORACLE_TOL, "uncached 14-token diff {d_full}");

    // The cached trio against the reference's OWN cached execution — routing
    // is per-token, so the sparse block must be transparent to the cache
    // split (prefill / multi-token continuation / single decodes).
    let mut dec = model.merged_decoder().unwrap();
    let prefill = seq.narrow(1, 0, p).unwrap();
    let want_prefill = tensor_from(case, "prefill_logits", (1, p, 64));
    let got_prefill = dec.forward(&prefill, 0).unwrap();
    let d_prefill = max_abs_diff(&got_prefill, &want_prefill);
    assert!(d_prefill <= ORACLE_TOL, "cached prefill diff {d_prefill}");

    let chunk = seq.narrow(1, p, c).unwrap();
    let want_chunk = tensor_from(case, "chunk_logits", (1, c, 64));
    let got_chunk = dec.forward(&chunk, p).unwrap();
    let d_chunk = max_abs_diff(&got_chunk, &want_chunk);
    assert!(
        d_chunk <= ORACLE_TOL,
        "multi-token cached continuation diff {d_chunk}"
    );

    let decode_cases = case["decode_logits"].as_array().unwrap();
    for (i, want_step) in decode_cases.iter().enumerate() {
        let t = p + c + i;
        let tok = seq.narrow(1, t, 1).unwrap();
        let want: Vec<f32> = want_step
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let want = Tensor::from_vec(want, (1, 1, 64), &Device::Cpu).unwrap();
        let got = dec.forward(&tok, t).unwrap();
        let d = max_abs_diff(&got, &want);
        assert!(d <= ORACLE_TOL, "single-token decode at {t} diff {d}");
    }
}

#[test]
fn perturbed_top_k_is_caught_by_the_gate() {
    // The MoE-specific vacuity guard: the SAME weights consulted at top-1
    // instead of top-2 must land far outside ORACLE_TOL — if a wrong routing
    // depth still passed, the sparse path would not actually be binding the
    // gate. (The dense twin pins the rope/eps axes; they are shared
    // machinery and not re-pinned here.)
    let g = golden();
    let dir = fixture_dir();
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    let cfg = Qwen3_5Config::from_json_str(&json.to_string()).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load(&cfg, &vb, 4, 8.0).unwrap();

    let case = &g["cases"]["full_b1"];
    let ids = input_ids(case, "input_ids");
    let want = tensor_from(case, "logits", (1, 12, 64));
    let got = model.forward(&ids).unwrap();
    let d = max_abs_diff(&got, &want);
    assert!(
        d >= PERTURBATION_FLOOR,
        "top-1 routing diverged only {d} — the sparse path is not binding the oracle gate"
    );
}

#[test]
fn corrupted_moe_widths_fail_loud_at_load() {
    // A config whose expert width disagrees with the checkpoint tensors must
    // be a LOAD error (shape mismatch), never a silently misshapen model.
    let dir = fixture_dir();
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["text_config"]["moe_intermediate_size"] = serde_json::json!(7);
    let cfg = Qwen3_5Config::from_json_str(&json.to_string()).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    assert!(Qwen3_5GradModel::load(&cfg, &vb, 4, 8.0).is_err());
}
