//! Load a ready-to-train Qwen-family policy from a Hugging Face checkpoint directory.
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

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;

use crate::lm_policy::{Qwen3_5Policy, QwenPolicy};
use crate::policy::{GenConfig, Policy, Rollout};
use crate::qwen::QwenGradModel;
use crate::qwen35::{
    varbuilder_from_pretrained, LoraTargets as Qwen35LoraTargets, Qwen3_5Config, Qwen3_5GradModel,
};
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
    /// The checkpoint declares a model family this loader does not support.
    #[error(
        "unsupported model_type {model_type:?}; supported values are \"qwen3\" and \"qwen3_5\""
    )]
    UnsupportedModelType {
        /// The `config.json` `model_type` value, or `None` if the field was absent.
        model_type: Option<String>,
    },
}

/// A concrete policy loaded from any qwen-family checkpoint the CLI supports.
///
/// The enum keeps `load_qwen_policy` backward-compatible while giving `ferrl train`
/// one model-agnostic return type for Qwen3 and Qwen3.5/3.6 checkpoints.
pub enum AutoPolicy {
    /// A classic dense-attention Qwen3 policy.
    Qwen(Box<QwenPolicy>),
    /// A `qwen3_5` family policy (Qwen3.5 / Qwen3.6).
    Qwen3_5(Box<Qwen3_5Policy>),
}

impl Policy for AutoPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        match self {
            Self::Qwen(policy) => policy.generate(prompt, cfg),
            Self::Qwen3_5(policy) => policy.generate(prompt, cfg),
        }
    }

    fn generate_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
    ) -> CandleResult<Rollout> {
        match self {
            Self::Qwen(policy) => policy.generate_at(prompt, cfg, global_row_base),
            Self::Qwen3_5(policy) => policy.generate_at(prompt, cfg, global_row_base),
        }
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        match self {
            Self::Qwen(policy) => policy.token_logprobs(rollout),
            Self::Qwen3_5(policy) => policy.token_logprobs(rollout),
        }
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        match self {
            Self::Qwen(policy) => policy.token_logprobs_detached(rollout),
            Self::Qwen3_5(policy) => policy.token_logprobs_detached(rollout),
        }
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        match self {
            Self::Qwen(policy) => policy.backward(loss),
            Self::Qwen3_5(policy) => policy.backward(loss),
        }
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        match self {
            Self::Qwen(policy) => policy.set_adapter_enabled(enabled),
            Self::Qwen3_5(policy) => policy.set_adapter_enabled(enabled),
        }
    }

    fn adapter_enabled(&self) -> bool {
        match self {
            Self::Qwen(policy) => policy.adapter_enabled(),
            Self::Qwen3_5(policy) => policy.adapter_enabled(),
        }
    }

    fn trainable_vars(&self) -> Vec<Var> {
        match self {
            Self::Qwen(policy) => policy.trainable_vars(),
            Self::Qwen3_5(policy) => policy.trainable_vars(),
        }
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        match self {
            Self::Qwen(policy) => policy.sampler_state(),
            Self::Qwen3_5(policy) => policy.sampler_state(),
        }
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        match self {
            Self::Qwen(policy) => policy.restore_sampler_state(state),
            Self::Qwen3_5(policy) => policy.restore_sampler_state(state),
        }
    }

    fn lora_recipe(&self) -> Option<String> {
        match self {
            Self::Qwen(policy) => policy.lora_recipe(),
            Self::Qwen3_5(policy) => policy.lora_recipe(),
        }
    }
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

/// Load any qwen-family policy supported by the CLI from a checkpoint directory.
///
/// Currently dispatches `model_type == "qwen3_5"` / `"qwen3_5_moe"` to the
/// Qwen3.5/3.6 loader and `model_type == "qwen3"` (or an absent legacy field) to
/// the original Qwen3 loader.
///
/// # Errors
///
/// As [`load_qwen_policy`], plus [`LoaderError::UnsupportedModelType`] when
/// `config.json` declares a different model family.
pub fn load_auto_policy(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(AutoPolicy, HfTokenizer), LoaderError> {
    match read_model_type(dir)? {
        None | Some(ModelType::Qwen3) => {
            let (policy, tok) = load_qwen_policy(dir, device, opts)?;
            Ok((AutoPolicy::Qwen(Box::new(policy)), tok))
        }
        Some(ModelType::Qwen3_5) => {
            let (policy, tok) = load_qwen35_policy(dir, device, opts)?;
            Ok((AutoPolicy::Qwen3_5(Box::new(policy)), tok))
        }
        Some(ModelType::Unsupported(model_type)) => {
            Err(LoaderError::UnsupportedModelType { model_type })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelType {
    Qwen3,
    Qwen3_5,
    Unsupported(Option<String>),
}

fn read_model_type(dir: &Path) -> Result<Option<ModelType>, LoaderError> {
    let cfg_path = dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|source| LoaderError::Io {
        path: cfg_path.clone(),
        source,
    })?;
    let value: serde_json::Value =
        serde_json::from_slice(&cfg_bytes).map_err(|source| LoaderError::Config {
            path: cfg_path,
            source,
        })?;
    let Some(model_type) = value.get("model_type") else {
        return Ok(None);
    };
    let Some(model_type) = model_type.as_str() else {
        return Ok(Some(ModelType::Unsupported(None)));
    };
    Ok(Some(match model_type {
        "qwen3" => ModelType::Qwen3,
        "qwen3_5" | "qwen3_5_moe" => ModelType::Qwen3_5,
        other => ModelType::Unsupported(Some(other.to_string())),
    }))
}

fn load_qwen35_policy(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(Qwen3_5Policy, HfTokenizer), LoaderError> {
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json"))?;
    let vb = varbuilder_from_pretrained(dir, opts.base_dtype, device)?;
    let model = Qwen3_5GradModel::load_with_targets(
        &cfg,
        &vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
        Qwen35LoraTargets::industrial(),
    )?;
    let policy = Qwen3_5Policy::new(model, opts.seed, opts.temperature);
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

    #[test]
    fn auto_loader_rejects_unknown_model_type_before_weight_load() {
        let tmp = TempDir::new("loader-unknown-model");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "not_qwen" }"#,
        )
        .unwrap();
        match load_auto_policy(tmp.path(), &Device::Cpu, &LoaderOpts::default()) {
            Err(LoaderError::UnsupportedModelType { model_type }) => {
                assert_eq!(model_type.as_deref(), Some("not_qwen"));
            }
            Err(other) => panic!("expected UnsupportedModelType, got {other:?}"),
            Ok(_) => panic!("expected UnsupportedModelType, got loaded policy"),
        }
    }

    #[test]
    fn model_type_detection_routes_qwen35_family() {
        let tmp = TempDir::new("loader-qwen35-model-type");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "qwen3_5" }"#,
        )
        .unwrap();
        assert_eq!(
            read_model_type(tmp.path()).unwrap(),
            Some(ModelType::Qwen3_5)
        );
    }

    #[test]
    fn auto_loader_loads_the_tiny_qwen35_fixture() {
        let tmp = TempDir::new("loader-qwen35-fixture");
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        copy_dir_contents(&fixture_root.join("tiny_qwen35"), tmp.path());
        std::fs::copy(
            fixture_root.join("tiny_tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();

        let (policy, _tok) = load_auto_policy(
            tmp.path(),
            &Device::Cpu,
            &LoaderOpts {
                lora_rank: 4,
                lora_alpha: 8.0,
                ..LoaderOpts::default()
            },
        )
        .unwrap();
        let AutoPolicy::Qwen3_5(policy) = policy else {
            panic!("expected Qwen3.5 auto-loader branch");
        };
        assert!(!Policy::trainable_vars(&*policy).is_empty());
        assert_eq!(
            Policy::lora_recipe(&*policy).as_deref(),
            Some("attn:qkvo|mlp:gud|gdn:-")
        );
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "ferrl-{prefix}-{}-{}",
                std::process::id(),
                unique_suffix()
            ));
            std::fs::create_dir_all(&root).unwrap();
            Self(root)
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

    fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    fn copy_dir_contents(from: &Path, to: &Path) {
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let file_type = entry.file_type().unwrap();
            assert!(file_type.is_file(), "fixture copy only supports files");
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
}
