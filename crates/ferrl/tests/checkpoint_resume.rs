//! Checkpoint / resume / eval through the real `QwenPolicy` path (tiny CPU model).
//!
//! These are the P4-PR2 deliverables exercised end-to-end on a runnable-on-CPU
//! Qwen3 config (the same tiny scaffold `qwen.rs`/`lm_policy.rs` use in-crate):
//!
//! 1. an adapter saved from one model loads bit-exactly into a *fresh* model and
//!    changes its forward (proving [`load_adapter`] writes through the aliasing
//!    `trainable_vars()` into the real `QwenGradModel`);
//! 2. the `Trainer`'s periodic checkpoint captures the in-memory adapter exactly,
//!    and a run resumes from it via `train_from`;
//! 3. the [`evaluate`] harness drives base-vs-adapter scoring through `QwenPolicy`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::llama::Config as LlamaConfig;
use candle_transformers::models::qwen3::Config;

use ferrl::policy::{GenConfig, Rollout};
use ferrl::{
    evaluate, load_adapter, save_adapter, EvalReport, LlamaGradModel, LlamaPolicy, Policy,
    QwenGradModel, QwenPolicy, RewardFn, RunDir, TokenizerLike, Trainer, TrainerConfig,
};

const RANK: usize = 2;
const ALPHA: f64 = 4.0;
const SEED: u64 = 7;

/// A tiny Qwen3 config (2 layers, 2 Q / 1 KV head, `head_dim` 4) — runnable on CPU.
fn tiny_cfg() -> Config {
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
        tie_word_embeddings: true,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

/// Random weights matching `cfg`'s dotted tensor names (tied head → no `lm_head`).
fn weight_map(cfg: &Config) -> HashMap<String, Tensor> {
    let d = Device::Cpu;
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

/// A `VarBuilder` over a fresh weight map (`'static`: it owns the tensors). Two
/// models loaded from the *same* builder share base weights (cloned) but get
/// independent fresh `LoRA` factors.
fn tiny_vb(cfg: &Config) -> VarBuilder<'static> {
    VarBuilder::from_tensors(weight_map(cfg), DType::F32, &Device::Cpu)
}

fn policy_from(vb: &VarBuilder, cfg: &Config) -> QwenPolicy {
    QwenPolicy::new(
        QwenGradModel::load(cfg, vb, RANK, ALPHA).unwrap(),
        SEED,
        1.0,
    )
}

/// Flattened snapshot of every trainable var, for bit-exact comparison.
fn snapshot(policy: &QwenPolicy) -> Vec<Vec<f32>> {
    policy
        .trainable_vars()
        .iter()
        .map(|v| {
            v.as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .collect()
}

/// Trivial char codec over the tiny vocab (`'a' + i` <-> id `i`, vocab 16).
struct CharCodec;
impl TokenizerLike for CharCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| (u32::from(c) - u32::from('a')) % 16)
            .collect()
    }
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&i| char::from_u32(u32::from('a') + (i % 16)))
            .collect()
    }
}

/// Position-weighted reward so distinct completions rarely collide (a degenerate
/// group carries no gradient); same one the in-crate Qwen CPU test uses.
struct SpreadReward;
impl RewardFn for SpreadReward {
    fn reward(&self, _prompt: &str, completion: &str) -> f32 {
        completion
            .bytes()
            .enumerate()
            .map(|(i, b)| (i as f32 + 1.0) * f32::from(b))
            .sum::<f32>()
            / 1000.0
    }
}

/// A unique temp directory, removed on drop.
struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!(
            "ferrl-ckpt-it-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A fixed rectangular rollout for deterministic scoring (`prompt_len` 2, comp 3).
fn fixed_rollout() -> Rollout {
    Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![3, 1, 4, 1, 5]], 2)
}

#[test]
fn adapter_round_trips_into_a_fresh_model() {
    let cfg = tiny_cfg();
    let vb = tiny_vb(&cfg);
    // Two models over the SAME base weights, independent fresh adapters.
    let src = policy_from(&vb, &cfg);
    let dst = policy_from(&vb, &cfg);
    let rollout = fixed_rollout();

    // Force `src`'s adapter to a clearly non-zero state so it diverges from the
    // zero-B `dst` (and from the base distribution).
    for v in &src.trainable_vars() {
        let dims = v.as_tensor().dims().to_vec();
        v.set(&Tensor::randn(0f32, 0.1f32, dims, &Device::Cpu).unwrap())
            .unwrap();
    }
    let logp_src = src
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let logp_dst_before = dst
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    // Sanity: a non-zero adapter on `src` actually moved its scores off `dst`'s.
    let max_diff = logp_src
        .iter()
        .flatten()
        .zip(logp_dst_before.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-4,
        "forced adapter did not change scores: {max_diff}"
    );

    // Save src's adapter, load it into dst, and the forwards must now agree.
    let tmp = TempDir::new("roundtrip");
    save_adapter(tmp.path(), &src.trainable_vars(), 0).unwrap();
    let manifest = load_adapter(tmp.path(), &dst.trainable_vars()).unwrap();
    assert_eq!(manifest.num_vars, src.trainable_vars().len());

    let logp_dst_after = dst
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(
        logp_src, logp_dst_after,
        "loaded adapter must reproduce src's scores bit-for-bit"
    );
}

/// A tiny dense-Llama config (2 layers, 2 Q / 1 KV head, derived `head_dim` 4)
/// — the same scaffold `llama.rs`/`lm_policy.rs` use, at a runnable CPU scale.
fn llama_tiny_cfg() -> LlamaConfig {
    LlamaConfig {
        hidden_size: 8,
        intermediate_size: 16,
        vocab_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        use_flash_attn: false,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        max_position_embeddings: 32,
        tie_word_embeddings: true,
    }
}

/// Random weights matching the llama dotted tensor names (tied head → no
/// `lm_head.weight`; no QK-norm tensors, no biases).
fn llama_weight_map(cfg: &LlamaConfig) -> HashMap<String, Tensor> {
    let d = Device::Cpu;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut put = |name: &str, dims: &[usize]| {
        t.insert(
            name.to_string(),
            Tensor::randn(0f32, 0.2f32, dims.to_vec(), &d).unwrap(),
        );
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let qo = cfg.num_attention_heads * head_dim;
    let kvo = cfg.num_key_value_heads * head_dim;
    put("model.embed_tokens.weight", &[cfg.vocab_size, h]);
    put("model.norm.weight", &[h]);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        put(&format!("{p}.input_layernorm.weight"), &[h]);
        put(&format!("{p}.post_attention_layernorm.weight"), &[h]);
        put(&format!("{p}.self_attn.q_proj.weight"), &[qo, h]);
        put(&format!("{p}.self_attn.k_proj.weight"), &[kvo, h]);
        put(&format!("{p}.self_attn.v_proj.weight"), &[kvo, h]);
        put(&format!("{p}.self_attn.o_proj.weight"), &[h, qo]);
        put(&format!("{p}.mlp.gate_proj.weight"), &[i, h]);
        put(&format!("{p}.mlp.up_proj.weight"), &[i, h]);
        put(&format!("{p}.mlp.down_proj.weight"), &[h, i]);
    }
    t
}

#[test]
fn llama_adapter_round_trips_into_a_fresh_model() {
    // The M1 mirror of `adapter_round_trips_into_a_fresh_model`: the adapter
    // checkpoint POSITIONAL contract (`trainable_vars()` order is the schema —
    // see `checkpoint.rs`) must hold for the second `GradModel` too. Save from
    // a trained-ish LlamaPolicy, load into a fresh model over the same base
    // weights, and the forwards must agree bit-for-bit.
    let cfg = llama_tiny_cfg();
    let weights = llama_weight_map(&cfg);
    let policy_over = |w: &HashMap<String, Tensor>| -> LlamaPolicy {
        let vb = VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
        LlamaPolicy::new(
            LlamaGradModel::load(&cfg, &vb, RANK, ALPHA).unwrap(),
            SEED,
            1.0,
        )
    };
    // Two models over the SAME base weights, independent fresh adapters.
    let src = policy_over(&weights);
    let dst = policy_over(&weights);
    let rollout = fixed_rollout();

    // Force `src`'s adapter to a clearly non-zero ("trained-ish") state so it
    // diverges from the zero-B `dst`.
    for v in &src.trainable_vars() {
        let dims = v.as_tensor().dims().to_vec();
        v.set(&Tensor::randn(0f32, 0.1f32, dims, &Device::Cpu).unwrap())
            .unwrap();
    }
    let logp_src = src
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let logp_dst_before = dst
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    // Premise: the non-zero adapter actually moved src's scores off dst's
    // (otherwise the bit-equality below would be vacuous).
    let max_diff = logp_src
        .iter()
        .flatten()
        .zip(logp_dst_before.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-4,
        "forced adapter did not change scores: {max_diff}"
    );

    // Save src's adapter, load it into dst: forwards must now agree bit-for-bit.
    let tmp = TempDir::new("llama-roundtrip");
    save_adapter(tmp.path(), &src.trainable_vars(), 0).unwrap();
    let manifest = load_adapter(tmp.path(), &dst.trainable_vars()).unwrap();
    assert_eq!(manifest.num_vars, src.trainable_vars().len());

    let logp_dst_after = dst
        .token_logprobs(&rollout)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(
        logp_src, logp_dst_after,
        "loaded Llama adapter must reproduce src's scores bit-for-bit"
    );
}

/// The 4-step CPU training config the checkpoint/resume tests share.
fn ckpt_train_cfg(checkpoint_every: Option<u64>) -> TrainerConfig {
    TrainerConfig {
        steps: 4,
        group_size: 6,
        max_new_tokens: 4,
        temperature: 1.0,
        lr: 1e-3,
        checkpoint_every,
        ..TrainerConfig::default()
    }
}

#[test]
fn trainer_checkpoint_captures_final_adapter() {
    let cfg = tiny_cfg();
    let vb = tiny_vb(&cfg);
    let mut policy = policy_from(&vb, &cfg);
    let prompts = vec!["abc".to_string(), "bcd".to_string()];

    let tmp = TempDir::new("run");
    let run = RunDir::create(tmp.path(), "qwen-ckpt").unwrap();
    let ckpt_root = run.checkpoints_dir();
    let mut trainer = Trainer::new(ckpt_train_cfg(Some(2)), &run).unwrap();
    let history = trainer
        .train(&mut policy, &SpreadReward, &CharCodec, &prompts)
        .unwrap();
    assert_eq!(history.len(), 4);
    // A real AdamW step must have run (grad_norm > 0 only when the optimizer
    // stepped), so the adapter actually moved off its B=0 init — otherwise the
    // round-trip below would vacuously compare init-vs-init.
    assert!(
        history.iter().any(|m| m.grad_norm > 0.0),
        "no AdamW step ran — checkpoint round-trip would be vacuous"
    );

    // checkpoint_every = 2 over 4 steps -> step-2 and step-4 directories.
    assert!(
        ckpt_root.join("step-2").is_dir() && ckpt_root.join("step-4").is_dir(),
        "expected step-2 and step-4 checkpoints"
    );

    // The final checkpoint must equal the in-memory adapter, bit-for-bit, when
    // loaded into a fresh model (proves save captured the live weights through the
    // model's aliasing vars).
    let final_adapter = snapshot(&policy);
    let probe = policy_from(&vb, &cfg);
    let m4 = load_adapter(ckpt_root.join("step-4"), &probe.trainable_vars()).unwrap();
    assert_eq!(m4.step, 4);
    assert_eq!(
        snapshot(&probe),
        final_adapter,
        "step-4 checkpoint != final adapter"
    );
}

#[test]
fn trainer_resumes_from_a_checkpoint() {
    let cfg = tiny_cfg();
    let vb = tiny_vb(&cfg);
    let mut policy = policy_from(&vb, &cfg);
    let prompts = vec!["abc".to_string(), "bcd".to_string()];

    // First run: produce the step-2 checkpoint.
    let tmp = TempDir::new("resume");
    let run = RunDir::create(tmp.path(), "qwen-ckpt").unwrap();
    let ckpt_root = run.checkpoints_dir();
    let mut trainer = Trainer::new(ckpt_train_cfg(Some(2)), &run).unwrap();
    trainer
        .train(&mut policy, &SpreadReward, &CharCodec, &prompts)
        .unwrap();

    // Resume: load step-2 into a fresh policy and run the remaining 2 steps. We
    // assert it continues cleanly with finite metrics and the right step indices —
    // NOT that it matches the uninterrupted run (Adam momentum + sampler RNG
    // re-warm; see the checkpoint module docs).
    let mut resume_policy = policy_from(&vb, &cfg);
    let m2 = load_adapter(ckpt_root.join("step-2"), &resume_policy.trainable_vars()).unwrap();
    assert_eq!(m2.step, 2);

    let run2 = RunDir::create(tmp.path(), "qwen-resume").unwrap();
    let mut trainer2 = Trainer::new(ckpt_train_cfg(None), &run2).unwrap();
    let resumed = trainer2
        .train_from(2, &mut resume_policy, &SpreadReward, &CharCodec, &prompts)
        .unwrap();

    assert_eq!(resumed.len(), 2, "resume should run steps 2 and 3");
    assert!(resumed.iter().all(|m| m.grad_norm.is_finite()
        && m.reward_mean.is_finite()
        && m.lr.is_finite()
        && m.lr > 0.0));
    assert_eq!((resumed[0].step, resumed[1].step), (2, 3));
}

#[test]
fn evaluate_scores_base_and_adapter_through_qwen_policy() {
    let cfg = tiny_cfg();
    let vb = tiny_vb(&cfg);
    let mut policy = policy_from(&vb, &cfg);
    // Give the adapter a non-zero state so base and adapter are genuinely two
    // different distributions being evaluated.
    for v in &policy.trainable_vars() {
        let dims = v.as_tensor().dims().to_vec();
        v.set(&Tensor::randn(0f32, 0.1f32, dims, &Device::Cpu).unwrap())
            .unwrap();
    }
    let prompts = vec!["abc".to_string(), "bca".to_string()];
    let gen = GenConfig {
        group_size: 4,
        max_new_tokens: 3,
        temperature: 1.0, // must match the policy's baked temperature
        eos_token_id: None,
    };
    let report = evaluate(&mut policy, &SpreadReward, &CharCodec, &prompts, &gen).unwrap();

    assert_eq!(report.n_prompts, 2);
    assert_eq!(report.group_size, 4);
    assert_eq!(report.per_prompt.len(), 2);
    assert_finite_report(&report);
    // The harness restores the adapter-enabled flag (it entered enabled).
    assert!(policy.adapter_enabled());
}

/// Every reward field of `report` (aggregate, improvement, and per-prompt) is finite.
fn assert_finite_report(report: &EvalReport) {
    assert!(report.base_reward_mean.is_finite());
    assert!(report.adapter_reward_mean.is_finite());
    assert!(report.improvement().is_finite());
    assert!(report
        .per_prompt
        .iter()
        .all(|p| p.base_mean.is_finite() && p.adapter_mean.is_finite()));
}
