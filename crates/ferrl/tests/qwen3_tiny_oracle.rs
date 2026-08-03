//! Official external-reference oracle for the native Qwen3 text forward.
//!
//! `tests/fixtures/tiny_qwen3/` is a real saved Transformers checkpoint built
//! and executed by `Qwen3ForCausalLM` under the pinned fp32 CPU oracle env.  It
//! exercises GQA, Q/K RMS normalization, an explicit `head_dim` whose attention
//! width differs from `hidden_size`, and compares every logit position for both
//! batch-1 and batch-2 forwards.
//!
//! Generator: `scripts/oracle/gen_qwen3_tiny_golden.py`.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use ferrl::QwenGradModel;
use std::path::PathBuf;

/// Calibrate remotely against the generated fp32 fixture before merging.  This
/// leaves cross-host CPU reassociation room while remaining far below the
/// planted wrong-RoPE floor.
const ORACLE_TOL: f32 = 5e-4;
const PERTURBATION_FLOOR: f32 = 1e-2;
const ROPE_BRIDGE: &str = "upstream rope_parameters.rope_theta mirrored to candle rope_theta";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3")
}

fn config_json() -> serde_json::Value {
    let raw = std::fs::read_to_string(fixture_dir().join("config.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn parse_config(value: &serde_json::Value) -> Config {
    serde_json::from_value(value.clone()).expect("parse official Qwen3 config into candle mirror")
}

#[allow(clippy::cognitive_complexity)] // provenance assertion checklist
fn golden() -> serde_json::Value {
    let raw = std::fs::read_to_string(fixture_dir().join("golden.json")).unwrap();
    let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        value["meta"]["generator"].as_str().unwrap(),
        "scripts/oracle/gen_qwen3_tiny_golden.py"
    );
    assert_eq!(
        value["meta"]["reference"].as_str().unwrap(),
        "transformers.Qwen3ForCausalLM/Qwen3Config"
    );
    assert_eq!(value["meta"]["transformers"].as_str().unwrap(), "5.11.0");
    assert!(value["meta"]["torch"]
        .as_str()
        .unwrap()
        .starts_with("2.12.0"));
    assert_eq!(value["meta"]["device"].as_str().unwrap(), "cpu");
    assert_eq!(value["meta"]["dtype"].as_str().unwrap(), "float32");
    assert_eq!(value["meta"]["seed"].as_u64().unwrap(), 31_415);
    assert_eq!(value["meta"]["weight_std"].as_f64().unwrap(), 0.45);
    assert_eq!(
        value["meta"]["rope_config_bridge"].as_str().unwrap(),
        ROPE_BRIDGE
    );
    value
}

fn input_ids(case: &serde_json::Value) -> Tensor {
    let rows: Vec<Vec<u32>> = case["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
                .collect()
        })
        .collect();
    let batch = rows.len();
    let sequence = rows[0].len();
    assert!(rows.iter().all(|row| row.len() == sequence));
    Tensor::from_vec(
        rows.into_iter().flatten().collect::<Vec<_>>(),
        (batch, sequence),
        &Device::Cpu,
    )
    .unwrap()
}

fn logits(case: &serde_json::Value, dims: (usize, usize, usize)) -> Tensor {
    let values: Vec<f32> = case["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(values.len(), dims.0 * dims.1 * dims.2);
    Tensor::from_vec(values, dims, &Device::Cpu).unwrap()
}

fn varbuilder() -> VarBuilder<'static> {
    let buffer = std::fs::read(fixture_dir().join("model.safetensors")).unwrap();
    VarBuilder::from_buffered_safetensors(buffer, DType::F32, &Device::Cpu).unwrap()
}

fn load_model(config: &Config) -> QwenGradModel {
    let mut model = QwenGradModel::load(config, &varbuilder(), 2, 4.0).unwrap();
    model.set_adapter_enabled(false);
    model
}

fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
    assert_eq!(left.dims(), right.dims(), "logit shape mismatch");
    left.sub(right)
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

fn signal_span(tensor: &Tensor) -> f32 {
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    high - low
}

#[test]
#[allow(clippy::cognitive_complexity)] // fixture provenance/geometry checklist
fn fixture_pins_qwen_family_traps_and_rope_bridge() {
    let raw = config_json();
    assert_eq!(raw["model_type"].as_str().unwrap(), "qwen3");
    assert_eq!(
        raw["rope_parameters"]["rope_type"].as_str().unwrap(),
        "default"
    );
    assert_eq!(
        raw["rope_parameters"]["rope_theta"].as_f64().unwrap(),
        raw["rope_theta"].as_f64().unwrap(),
        "candle compatibility scalar drifted from upstream rope_parameters"
    );

    let config = parse_config(&raw);
    assert_eq!(config.num_hidden_layers, 2);
    assert_eq!(config.hidden_size, 16);
    assert_eq!(config.num_attention_heads, 4);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim, 8);
    assert_eq!(config.num_attention_heads * config.head_dim, 32);
    assert!(config.tie_word_embeddings);
    assert!(!config.attention_bias);
    assert!(!config.use_sliding_window);
    assert_eq!(config.rope_theta, 1_000_000.0);
}

#[test]
fn full_position_logits_match_official_qwen3_for_both_batches() {
    let golden = golden();
    let config = parse_config(&config_json());
    let model = load_model(&config);
    for case_name in ["full_b1", "full_b2"] {
        let case = &golden["cases"][case_name];
        let ids = input_ids(case);
        let (batch, sequence) = ids.dims2().unwrap();
        let expected_shape = case["shape"].as_array().unwrap();
        assert_eq!(expected_shape[0].as_u64().unwrap() as usize, batch);
        assert_eq!(expected_shape[1].as_u64().unwrap() as usize, sequence);
        assert_eq!(
            expected_shape[2].as_u64().unwrap() as usize,
            config.vocab_size
        );
        let expected = logits(case, (batch, sequence, config.vocab_size));
        assert!(
            signal_span(&expected) > 0.5,
            "{case_name}: degenerate official Qwen3 signal"
        );
        let actual = model.forward(&ids).unwrap();
        let difference = max_abs_diff(&actual, &expected);
        assert!(
            difference <= ORACLE_TOL,
            "{case_name}: QwenGradModel vs official Qwen3 diff {difference}"
        );
    }
}

#[test]
fn wrong_qwen_rope_theta_is_well_outside_honest_tolerance() {
    let golden = golden();
    let mut raw = config_json();
    // Deliberately change only candle's load-compatible execution field.  The
    // same exact checkpoint weights are compared with the official golden.
    raw["rope_theta"] = serde_json::json!(10.0);
    let config = parse_config(&raw);
    let model = load_model(&config);
    let case = &golden["cases"]["full_b1"];
    let ids = input_ids(case);
    let expected = logits(case, (1, 12, 32));
    let difference = max_abs_diff(&model.forward(&ids).unwrap(), &expected);
    assert!(
        difference >= PERTURBATION_FLOOR,
        "wrong Qwen rope theta diverged only {difference}; oracle is vacuous"
    );
}
