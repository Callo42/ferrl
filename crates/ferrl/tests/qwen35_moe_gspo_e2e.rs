//! The GSPO × `MoE` × checkpointing end-to-end gate (M3′ PR-2).
//!
//! The locked `MoE` training recipe (GSPO adoption, 2026-06-12) is
//! sequence-level importance sampling. This gate runs the REAL `Trainer` over
//! the committed tiny `qwen3_5_moe` fixture at
//! `ImportanceSamplingLevel::Sequence` with `mu = 2` and `beta > 0` — the
//! configuration where the sequence-level ratio, the detached `logp_old`/KL
//! scorings, and the sparse forward all genuinely execute — twice: activation
//! checkpointing OFF and ON, over policies with synced adapters and a shared
//! sampler seed. The trained vars must agree within float tolerance: the P7
//! e2e pattern, instantiated on the `MoE` model and the locked recipe.

use candle_core::{DType, Device};
use ferrl::grpo::ImportanceSamplingLevel;
use ferrl::{
    varbuilder_from_pretrained, Policy, Qwen3_5Config, Qwen3_5GradModel, Qwen3_5Policy, RewardFn,
    RunDir, TokenizerLike, Trainer, TrainerConfig,
};
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen35_moe")
}

/// A char-level codec over the fixture's 64-token vocabulary.
struct ByteCodec;
impl TokenizerLike for ByteCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| u32::from(b) % 64).collect()
    }
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| char::from(b'a' + u8::try_from(i % 26).unwrap()))
            .collect()
    }
}

/// A deterministic reward that SPREADS over completions (distinct outputs get
/// distinct scores), so group advantages are non-degenerate.
struct SpreadReward;
impl RewardFn for SpreadReward {
    fn reward(&self, _prompt: &str, completion: &str) -> f32 {
        completion
            .bytes()
            .enumerate()
            .map(|(i, b)| f32::from(b) * (0.3 + i as f32 * 0.17))
            .sum::<f32>()
            % 5.0
    }
}

fn build_policy() -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap();
    Qwen3_5Policy::new(model, 7, 1.0)
}

struct TempDir(std::path::PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ferrl-moe-gspo-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

#[test]
fn gspo_training_on_the_moe_model_matches_across_checkpointing() {
    let mut off = build_policy();
    let mut on = build_policy();
    // The base weights are the same fixture; the adapter `A` factors are
    // drawn per load — sync them (invisible to the forward at `B = 0`, but
    // `dL/dB ∝ A`, the R2 lesson).
    for (va, vb) in off.trainable_vars().iter().zip(on.trainable_vars()) {
        vb.set(va.as_tensor()).unwrap();
    }
    on.model_mut().set_activation_checkpointing(true);

    let cfg = TrainerConfig {
        steps: 2,
        group_size: 4,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.02,
        mu: 2,
        lr: 1e-3,
        importance_sampling_level: ImportanceSamplingLevel::Sequence,
        ..TrainerConfig::default()
    };
    let prompts = vec!["abc".to_string(), "bcd".to_string()];

    let run_one = |policy: &mut Qwen3_5Policy, tag: &str| {
        let tmp = TempDir::new(tag);
        let run = RunDir::create(&tmp.0, tag).unwrap();
        let mut trainer = Trainer::new(cfg.clone(), &run).unwrap();
        let (history, _stop) = trainer
            .train(policy, &SpreadReward, &ByteCodec, &prompts)
            .unwrap();
        assert!(
            history.iter().any(|m| m.grad_norm > 0.0),
            "{tag}: no real update ran — the comparison would be vacuous"
        );
    };
    run_one(&mut off, "gspo-remat-off");
    run_one(&mut on, "gspo-remat-on");

    for (va, vb) in off.trainable_vars().iter().zip(on.trainable_vars()) {
        let diff: f32 = va
            .as_tensor()
            .sub(vb.as_tensor())
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
            diff <= 1e-5,
            "trained vars diverged between checkpointing on/off under GSPO: {diff}"
        );
    }
}
