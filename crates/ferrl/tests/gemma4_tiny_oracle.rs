//! Tiny external-reference oracle for the dense Gemma 4 text forward.
//!
//! `tests/fixtures/tiny_gemma4/` is a committed checkpoint built and executed
//! by the official Transformers reference (`Gemma4ForCausalLM`,
//! `Gemma4TextConfig`, transformers pinned 5.11.0, fp32 CPU). The reference
//! currently exposes the dense text config shape as `gemma4` / `gemma4_text`,
//! not ferrl's unified alias names; ferrl's config tests cover
//! `gemma4_unified` / `gemma4_unified_text` routing into the same dense text
//! loader.
//!
//! Generator: `scripts/oracle/gen_gemma4_tiny_golden.py` (version-pinned;
//! regenerate only when the oracle pin moves).

use candle_core::{DType, Device, Tensor};
use ferrl::gemma4::varbuilder_from_pretrained;
use ferrl::{DenseLoraTargets, Gemma4Config, Gemma4GradModel};
use std::path::PathBuf;

/// Our fp32 forward vs the reference's fp32 logits, per position.
///
/// Stated tolerance for this committed oracle. It is intentionally wider than
/// the measured floor to leave cross-host CPU reassociation room, but still
/// tight enough that a mask, `RoPE`, norm, or residual-order bug should fail.
const ORACLE_TOL: f32 = 3e-4;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_gemma4")
}

fn golden() -> serde_json::Value {
    let raw = std::fs::read_to_string(fixture_dir().join("golden.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        v["meta"]["transformers"].as_str().unwrap(),
        "5.11.0",
        "fixture must come from the pinned oracle transformers"
    );
    assert!(
        v["meta"]["torch"].as_str().unwrap().starts_with("2.12.0"),
        "fixture must come from the pinned oracle torch"
    );
    assert_eq!(
        v["meta"]["fixture_config_shape"].as_str().unwrap(),
        "gemma4/gemma4_text",
        "fixture must state the upstream dense text reference shape"
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

fn load_model() -> (Gemma4Config, Gemma4GradModel) {
    let dir = fixture_dir();
    let cfg = Gemma4Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    let mut model = Gemma4GradModel::load_with_targets(
        &cfg,
        &vb,
        2,
        4.0,
        DType::F32,
        DenseLoraTargets::industrial(),
    )
    .unwrap();
    model.set_adapter_enabled(false);
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
#[allow(clippy::cognitive_complexity)] // compact fixture-contract checklist
fn fixture_config_is_the_upstream_dense_text_shape() {
    let (cfg, _model) = load_model();
    let t = &cfg.text_config;
    assert_eq!(cfg.model_type.as_deref(), Some("gemma4"));
    assert_eq!(t.model_type.as_deref(), Some("gemma4_text"));
    assert_eq!(t.num_hidden_layers, 2);
    assert_eq!(t.layer_types.len(), 2);
    assert_eq!(t.sliding_window, 3);
    assert!(t.attention_k_eq_v);
    assert_eq!(t.hidden_size_per_layer_input, 0);
    assert_eq!(t.vocab_size_per_layer_input, t.vocab_size);
}

#[test]
fn uncached_forward_matches_transformers_reference_logits() {
    let g = golden();
    let (cfg, model) = load_model();
    for case_name in ["full_b1", "full_b2"] {
        let case = &g["cases"][case_name];
        let ids = input_ids(case, "input_ids");
        let (b, l) = ids.dims2().unwrap();
        let shape = case["shape"].as_array().unwrap();
        assert_eq!(shape[0].as_u64().unwrap() as usize, b);
        assert_eq!(shape[1].as_u64().unwrap() as usize, l);
        assert_eq!(
            shape[2].as_u64().unwrap() as usize,
            cfg.text_config.vocab_size
        );
        let want = tensor_from(case, "logits", (b, l, cfg.text_config.vocab_size));
        let got = model.forward(&ids).unwrap();
        let d = max_abs_diff(&got, &want);
        assert!(
            d <= ORACLE_TOL,
            "{case_name}: Gemma4GradModel vs transformers reference diff {d}"
        );

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
            scale > 0.1,
            "{case_name}: degenerate oracle logits ({scale})"
        );
    }
}
