//! The tiny-oracle architecture gate for the `qwen3_5` forward — the gate
//! that closes the "no candle reference" hole.
//!
//! candle ships no `qwen3_5`, so unlike the Qwen3/Llama validators there is no
//! shipped forward to pin against. Instead, `tests/fixtures/tiny_qwen35/` is a
//! COMMITTED checkpoint built and executed by the real HF reference
//! (`Qwen3_5ForConditionalGeneration`, transformers pinned 5.11.0, fp32 CPU,
//! torch fallback kernels — the exact code `src/gdn.rs` is ported from), in
//! the production `model.language_model.*` sharded layout with a decoy vision
//! tower. These gates run our loader + forward + merged decoder against the
//! reference's own per-position logits, offline, on every CI run.
//!
//! Generator: `scripts/oracle/gen_qwen35_tiny_golden.py` (version-pinned;
//! regenerate only when the oracle pin moves).

use candle_core::{DType, Device, Tensor};
use ferrl::{varbuilder_from_pretrained, Qwen3_5Config, Qwen3_5GradModel};
use std::path::PathBuf;

/// Our fp32 forward vs the reference's fp32 logits, per position.
///
/// Measured worst under the committed fixture (2026-06-11): 3.15e-5
/// (`full_b2`); every cached-path gate measured ≤ 1.19e-5; `full_b1` and the
/// 14-token case 1.18e-5. Set ~10x above the measured floor for cross-host
/// float reassociation (the P2 platform lesson) while staying ~500x below the
/// `PERTURBATION_FLOOR` teeth, so the gate cannot silently go vacuous.
const ORACLE_TOL: f32 = 3e-4;

/// How far a deliberately mis-built model (wrong rope theta) must land from
/// the oracle — the permanent vacuity guard: if a config-level wrongness
/// still passed `ORACLE_TOL`, the gate would be toothless. Measured divergence
/// of the theta-1e4 model on this fixture: 0.168 (~3x this floor, ~5300x the
/// honest-model diff).
const PERTURBATION_FLOOR: f32 = 5e-2;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen35")
}

fn golden() -> serde_json::Value {
    let raw = std::fs::read_to_string(fixture_dir().join("golden.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    // The fixture is only meaningful under the pinned oracle env: a fixture
    // regenerated under a drifted transformers/torch silently re-baselines
    // every gate, so the version is asserted IN the gate, not just in the
    // generator.
    assert_eq!(
        v["meta"]["transformers"].as_str().unwrap(),
        "5.11.0",
        "fixture must come from the pinned oracle transformers"
    );
    assert!(
        v["meta"]["torch"].as_str().unwrap().starts_with("2.12.0"),
        "fixture must come from the pinned oracle torch"
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
    // The config is the file transformers itself wrote — parsing it IS part
    // of the gate (serde shape, tolerated riders, validation).
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    // Multi-shard layout + the decoy model.visual.* tensors riding along.
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    // Zero-B LoRA init is exactly the base model (pinned by the unit gates).
    let model = Qwen3_5GradModel::load(&cfg, &vb, 4, 8.0).unwrap();
    (cfg, model)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    // Exact `sub` (not broadcast): a shape divergence must fail loudly, never
    // silently broadcast into a misleadingly small scalar.
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
fn fixture_config_resolves_as_expected() {
    let (cfg, _model) = load_model();
    let t = &cfg.text_config;
    assert_eq!(t.num_hidden_layers, 4);
    assert_eq!(t.rotary_dim(), 8);
    assert!(t.tie_word_embeddings);
    // Real GVA in the fixture (the 0.8B itself has none — 16 == 16).
    assert_eq!(t.linear_num_value_heads / t.linear_num_key_heads, 2);
}

#[test]
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
        assert!(
            d <= ORACLE_TOL,
            "{case_name}: our forward vs transformers reference diff {d}"
        );
        // Signal floor: the logits must be decisively non-degenerate, or the
        // tolerance above means nothing.
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

#[test]
fn merged_decoder_matches_reference_cached_path() {
    let g = golden();
    let (_cfg, model) = load_model();
    let case = &g["cases"]["cached_split"];
    let seq = input_ids(case, "input_ids"); // [1, 14]
    let p = case["prefill_len"].as_u64().unwrap() as usize; // 7
    let c = case["chunk_len"].as_u64().unwrap() as usize; // 5

    // First: our UNCACHED forward over the same 14 tokens vs the reference's
    // uncached logits (ties this case to the full gates).
    let want_full = tensor_from(case, "uncached_logits", (1, 14, 64));
    let got_full = model.forward(&seq).unwrap();
    let d_full = max_abs_diff(&got_full, &want_full);
    assert!(d_full <= ORACLE_TOL, "uncached 14-token diff {d_full}");

    // Then the cached trio against the reference's OWN cached execution:
    // prefill -> multi-token continuation at an offset (the transformers
    // v5.7.0-fixed GDN path, our highest-risk decoder path) -> single decodes.
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
fn perturbed_rms_eps_is_caught_by_the_gate() {
    // Second perturbation axis for the vacuity guard (the rope-theta guard
    // alone pins one axis): a wrong norm epsilon is the classic
    // quiet-constant bug class a loose tolerance could hide. Measured on this
    // fixture (2026-06-11): eps 1e-3 diverges by ~10 in logit space — 4
    // orders of magnitude above ORACLE_TOL.
    let g = golden();
    let dir = fixture_dir();
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["text_config"]["rms_norm_eps"] = serde_json::json!(1e-3);
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
        "wrong rms_norm_eps diverged only {d} — the oracle gate is going vacuous"
    );
}

#[test]
fn perturbed_rope_theta_is_caught_by_the_gate() {
    // The permanent vacuity guard (the M1 rope lesson institutionalized): a
    // model built from the SAME weights but a wrong rope base must land far
    // outside ORACLE_TOL on the same case the real gate uses. If this margin
    // ever collapses, the oracle gate has lost its teeth, not the model its
    // correctness.
    let g = golden();
    let dir = fixture_dir();
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["text_config"]["rope_parameters"]["rope_theta"] = serde_json::json!(10_000.0);
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
        "wrong rope theta diverged only {d} — the oracle gate is going vacuous"
    );
}
