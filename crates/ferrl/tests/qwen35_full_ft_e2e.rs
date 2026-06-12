//! Full fine-tuning end-to-end gates (PR-E).
//!
//! The REAL `Trainer` over the committed tiny dense fixture with EVERY base
//! weight trainable: (1) updates genuinely run and move the base weights;
//! (2) the momentum-faithful resume continues a full-FT checkpoint, and the
//! manifest's `"full-ft"` recipe makes the resume cross-check REJECT a `LoRA`
//! policy against it loudly (a positional load would land base weights on
//! adapter factors silently — count/shape checks cannot catch every aliasing);
//! (3) the eval harness fails loud on the unavailable base-vs-trained
//! comparison instead of comparing the policy against itself.

use candle_core::{DType, Device, Tensor};
use ferrl::policy::GenConfig;
use ferrl::{
    evaluate, tensors_from_pretrained, varbuilder_from_pretrained, EvalError, Policy,
    Qwen3_5Config, Qwen3_5GradModel, Qwen3_5Policy, RewardFn, RunDir, TokenizerLike, Trainer,
    TrainerConfig, TrainerError,
};
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen35")
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

/// A deterministic reward that SPREADS over completions, so group advantages
/// are non-degenerate.
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

fn full_ft_policy(seed: u64) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let tensors = tensors_from_pretrained(&dir, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load_full_ft(&cfg, tensors, DType::F32, &Device::Cpu).unwrap();
    Qwen3_5Policy::new(model, seed, 1.0)
}

fn lora_policy(seed: u64) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap();
    Qwen3_5Policy::new(model, seed, 1.0)
}

struct TempDir(std::path::PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ferrl-full-ft-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn train_cfg() -> TrainerConfig {
    TrainerConfig {
        steps: 2,
        group_size: 4,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.02,
        mu: 2,
        lr: 1e-3,
        checkpoint_every: Some(1),
        ..TrainerConfig::default()
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
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

/// How many of the policy's vars moved away from their `before` snapshots.
fn moved_count(policy: &Qwen3_5Policy, before: &[Tensor]) -> usize {
    policy
        .trainable_vars()
        .iter()
        .zip(before)
        .filter(|(v, b)| max_abs_diff(v.as_tensor(), b) > 0.0)
        .count()
}

#[test]
fn full_ft_training_moves_the_base_weights_and_resume_guards_the_recipe() {
    let mut policy = full_ft_policy(7);
    let n_vars = policy.trainable_vars().len();
    let before: Vec<_> = policy
        .trainable_vars()
        .iter()
        .map(|v| v.as_tensor().copy().unwrap())
        .collect();

    let tmp = TempDir::new("train");
    let run = RunDir::create(&tmp.0, "full-ft").unwrap();
    let mut trainer = Trainer::new(train_cfg(), &run).unwrap();
    let prompts = vec!["abc".to_string(), "bcd".to_string()];
    let history = trainer
        .train(&mut policy, &SpreadReward, &ByteCodec, &prompts)
        .unwrap();
    assert!(
        history
            .iter()
            .any(|m| m.grad_norm > 0.0 && m.grad_norm.is_finite()),
        "no real update ran — the gates below would be vacuous"
    );
    let moved = moved_count(&policy, &before);
    assert!(
        moved > n_vars / 2,
        "only {moved}/{n_vars} base weights moved — full-FT is not training the base model"
    );

    let ckpt = run.checkpoints_dir().join("step-2");
    assert!(ckpt.is_dir(), "expected the step-2 checkpoint");

    let step1 = run.checkpoints_dir().join("step-1");
    assert!(step1.is_dir(), "expected the step-1 checkpoint");
    resume_continues(&tmp, &step1, &prompts);
    lora_resume_is_rejected(&tmp, &step1, &prompts);
}

/// Resume leg: a FRESH full-FT policy continues from the step-1 checkpoint
/// (the recipe matches, vars/optimizer/sampler restore positionally).
fn resume_continues(tmp: &TempDir, step1: &std::path::Path, prompts: &[String]) {
    let mut policy = full_ft_policy(7);
    let run = RunDir::create(&tmp.0, "full-ft-resume").unwrap();
    let mut trainer = Trainer::new(train_cfg(), &run).unwrap();
    let resumed = trainer
        .resume(step1, &mut policy, &SpreadReward, &ByteCodec, prompts)
        .unwrap();
    assert_eq!(
        resumed.len(),
        1,
        "resume from step-1 should run step 2 only"
    );
}

/// The guard: a `LoRA` policy against the full-FT checkpoint is a loud recipe
/// mismatch, BEFORE any var is mutated.
fn lora_resume_is_rejected(tmp: &TempDir, step1: &std::path::Path, prompts: &[String]) {
    let mut wrong = lora_policy(7);
    let run = RunDir::create(&tmp.0, "full-ft-mismatch").unwrap();
    let mut trainer = Trainer::new(train_cfg(), &run).unwrap();
    let err = trainer
        .resume(step1, &mut wrong, &SpreadReward, &ByteCodec, prompts)
        .unwrap_err();
    assert!(
        matches!(err, TrainerError::Checkpoint(_)),
        "expected a checkpoint error, got {err:?}"
    );
    assert!(
        err.to_string().contains("does not match"),
        "expected the recipe cross-check, got: {err}"
    );
}

#[test]
fn full_ft_eval_comparison_fails_loud() {
    let mut policy = full_ft_policy(11);
    let gen = GenConfig {
        group_size: 2,
        max_new_tokens: 2,
        temperature: 1.0,
        eos_token_id: None,
        eval_sampling: None,
    };
    let err = evaluate(
        &mut policy,
        &SpreadReward,
        &ByteCodec,
        &["abc".to_string()],
        &gen,
    )
    .unwrap_err();
    assert!(matches!(err, EvalError::Contract(_)), "got {err:?}");
    assert!(
        format!("{err}").contains("cannot disable its adapter"),
        "got: {err}"
    );
}
