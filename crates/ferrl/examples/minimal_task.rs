//! The smallest "wire your own task" template: define a reward, pair prompts with
//! typed targets, point at a checkpoint, train. This is the *library* path — for a
//! built-in task from the CLI instead, see `ferrl train --config`.
//!
//! Build-checked in CI; to run: `FERRL_MODEL_DIR=<ckpt> cargo run --example minimal_task`.

// A runnable template whose final line reports where the run landed.
#![allow(clippy::print_stdout)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use ferrl::{
    load_qwen_policy, LoaderOpts, RewardError, RewardFn, RunDir, Sample, Trainer, TrainerConfig,
};

/// 1. Your task: implement [`RewardFn`] over your own typed `Target`. Here the
///    target is a keyword a good completion should contain (`1.0` if present).
struct ContainsKeyword;

impl RewardFn for ContainsKeyword {
    type Target = String;
    fn reward(&self, sample: &Sample<String>, completion: &str) -> Result<f32, RewardError> {
        let hit = completion.contains(sample.target.as_str());
        Ok(if hit { 1.0 } else { 0.0 })
    }
}

fn main() -> Result<()> {
    // 2. Your data: prompts paired with typed targets (load from JSONL in real use,
    //    via `ferrl::read_jsonl`).
    let train = vec![
        Sample::new("Name a primary color.", "red".to_string()),
        Sample::new("Greet the user.", "hello".to_string()),
    ];

    // 3. Load a policy from a checkpoint dir (config.json + model.safetensors +
    //    tokenizer.json). CPU here; pass a CUDA device for a GPU run.
    let dir = PathBuf::from(
        std::env::var("FERRL_MODEL_DIR")
            .context("set FERRL_MODEL_DIR to a Qwen3 checkpoint dir")?,
    );
    let cfg = TrainerConfig::builder()
        .steps(10)
        .group_size(8)
        .max_new_tokens(32)
        .build();
    let opts = LoaderOpts {
        temperature: cfg.temperature,
        ..LoaderOpts::default()
    };
    let (mut policy, tok) = load_qwen_policy(&dir, &Device::Cpu, &opts)?;

    // 4. Train: GRPO over your reward, writing metrics/checkpoints under the run dir.
    let run = RunDir::create("runs", "minimal-task")?;
    let mut trainer = Trainer::new(cfg, &run)?;
    trainer.train(&mut policy, &ContainsKeyword, &tok, &train)?;
    println!("done -> {}", run.root().display());
    Ok(())
}
