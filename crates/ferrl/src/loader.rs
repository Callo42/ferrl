//! Load a ready-to-train [`QwenPolicy`] from a Hugging Face checkpoint directory.
//!
//! Every runnable entry point — the `ferrl` CLI, the worked examples, the
//! "wire your own task" template — needs the same few steps to turn a checkpoint
//! directory into a policy: parse `config.json`, map `model.safetensors` into a
//! [`VarBuilder`], attach a `LoRA` adapter, and load the tokenizer. This module is
//! that one shared loader, so those call sites do not each hand-roll it (and drift).
//!
//! It is deliberately thin glue over [`QwenGradModel`] + [`QwenPolicy`]; the heavy,
//! correctness-bearing logic lives in those types and is covered there. Loading a
//! real multi-gigabyte checkpoint needs the asset on disk, so the happy path is
//! exercised by the runnable harnesses (the CLI's end-to-end smoke), not the unit
//! tests — this file is therefore excluded from the coverage denominator alongside
//! the binaries and examples (see `justfile` / `.github/workflows/ci.yml`).

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;

use crate::lm_policy::QwenPolicy;
use crate::qwen::QwenGradModel;
use crate::tokenizer::{HfTokenizer, TokenizerError};

/// Knobs for [`load_qwen_policy`]: the `LoRA` shape, the load dtypes, and the
/// sampling seed/temperature.
///
/// [`Default`] is the `PoC` recipe — `LoRA` rank 16 / alpha 32, base and adapter both
/// `F32` (the natural CPU dtype), seed 1234, temperature 1.0. On a GPU run, set
/// `base_dtype` to [`DType::BF16`] to halve the frozen base's memory while keeping
/// the trainable adapter in `F32`.
///
/// `temperature` **must** equal the [`TrainerConfig::temperature`](crate::TrainerConfig::temperature)
/// of the run it feeds: an [`LmPolicy`](crate::LmPolicy) scores against its own baked
/// temperature and fails loud on a mismatch.
#[derive(Debug, Clone)]
pub struct LoaderOpts {
    /// `LoRA` rank `r`.
    pub lora_rank: usize,
    /// `LoRA` scaling `alpha` (effective scale `alpha / r`).
    pub lora_alpha: f64,
    /// Dtype the frozen base weights load in (`F32` on CPU; `BF16` to save GPU memory).
    pub base_dtype: DType,
    /// Dtype the trainable `LoRA` adapter (and its gradients / moments) is held in —
    /// keep `F32` so a small update cannot collapse.
    pub adapter_dtype: DType,
    /// RNG seed for the policy's rollout sampler.
    pub seed: u64,
    /// Rollout sampling temperature; must match the trainer's configured temperature.
    pub temperature: f64,
}

impl Default for LoaderOpts {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            base_dtype: DType::F32,
            adapter_dtype: DType::F32,
            seed: 1234,
            temperature: 1.0,
        }
    }
}

/// Errors from [`load_qwen_policy`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoaderError {
    /// A required checkpoint file could not be read.
    #[error("read {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying IO error.
        source: std::io::Error,
    },
    /// `config.json` could not be parsed into a Qwen3 [`Config`].
    #[error("parse {path}: {source}")]
    Config {
        /// The config file that failed to parse.
        path: PathBuf,
        /// The underlying deserialization error.
        source: serde_json::Error,
    },
    /// The model could not be built / mapped onto the device (a candle error).
    #[error("build model: {0}")]
    Model(#[from] candle_core::Error),
    /// The tokenizer could not be loaded.
    #[error("load tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
}

/// Load a [`QwenPolicy`] and its tokenizer from a checkpoint `dir` onto `device`.
///
/// Reads `dir/config.json`, `dir/model.safetensors`, and `dir/tokenizer.json`,
/// attaches a `LoRA` adapter per `opts`, and returns a policy ready to hand to
/// [`Trainer::train`](crate::Trainer::train) (or [`evaluate`](crate::evaluate)).
///
/// Pass [`Device::Cpu`] for a CPU run or [`Device::new_cuda`] for a GPU run; the
/// caller owns any CUDA preflight (see [`crate::guard_first_kernel`]).
///
/// # Errors
///
/// Returns [`LoaderError`] if a checkpoint file is missing or unreadable
/// ([`Io`](LoaderError::Io)), `config.json` does not parse
/// ([`Config`](LoaderError::Config)), the model fails to build on the device
/// ([`Model`](LoaderError::Model)), or the tokenizer fails to load
/// ([`Tokenizer`](LoaderError::Tokenizer)).
pub fn load_qwen_policy(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(QwenPolicy, HfTokenizer), LoaderError> {
    let cfg_path = dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|source| LoaderError::Io {
        path: cfg_path.clone(),
        source,
    })?;
    let cfg: Config = serde_json::from_slice(&cfg_bytes).map_err(|source| LoaderError::Config {
        path: cfg_path,
        source,
    })?;

    let weights_path = dir.join("model.safetensors");
    let weights = std::fs::read(&weights_path).map_err(|source| LoaderError::Io {
        path: weights_path,
        source,
    })?;
    let vb = VarBuilder::from_buffered_safetensors(weights, opts.base_dtype, device)?;
    let model = QwenGradModel::load_with_adapter_dtype(
        &cfg,
        &vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
    )?;
    let policy = QwenPolicy::new(model, opts.seed, opts.temperature);

    let tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((policy, tok))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_config_is_an_io_error_naming_the_path() {
        let dir = Path::new("/definitely/not/a/real/checkpoint/dir");
        let err = load_qwen_policy(dir, &Device::Cpu, &LoaderOpts::default()).unwrap_err();
        match err {
            LoaderError::Io { path, .. } => assert!(path.ends_with("config.json")),
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    #[test]
    fn default_opts_are_the_poc_recipe() {
        let o = LoaderOpts::default();
        assert_eq!(o.lora_rank, 16);
        assert_eq!(o.base_dtype, DType::F32);
        assert_eq!(o.adapter_dtype, DType::F32);
    }
}
