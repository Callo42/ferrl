//! Load a ready-to-train policy from a Hugging Face checkpoint directory.
//!
//! Every runnable entry point — the `ferrl` CLI, the worked examples, the
//! "wire your own task" template — needs the same few steps to turn a checkpoint
//! directory into a policy: parse `config.json`, map `model.safetensors` into a
//! [`VarBuilder`], attach a `LoRA` adapter, and load the tokenizer. This module is
//! that one shared loader, so those call sites do not each hand-roll it (and drift).
//!
//! It is deliberately thin glue over each model module; the heavy,
//! correctness-bearing logic lives in those types and is covered there. Loading
//! a real multi-gigabyte checkpoint needs the asset on disk, so the happy path
//! is exercised by the runnable harnesses (the CLI's end-to-end smoke), not the
//! unit tests — this file is therefore excluded from the coverage denominator
//! alongside the binaries and examples (see `justfile` / `.github/workflows/ci.yml`).

use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;

use crate::gemma4::{
    varbuilder_from_pretrained as gemma4_varbuilder_from_pretrained, Gemma4Config, Gemma4GradModel,
};
use crate::lm_policy::{Gemma4Policy, Qwen3_5Policy, QwenPolicy};
use crate::lora::{BaseQuantization, DenseLoraTargets};
use crate::policy::{GenConfig, Policy, Rollout, TensorParallelPolicy};
use crate::qwen::QwenGradModel;
use crate::qwen35::{
    varbuilder_from_pretrained, LoraTargets as Qwen35LoraTargets, Qwen3_5Config, Qwen3_5GradModel,
};
use crate::sharded_safetensors::varbuilder_from_rank_local_safetensors;
use crate::telemetry::ModelTelemetryRecorder;
use crate::tensor_parallel::TensorParallelPlan;
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
    /// Use the experimental grouped cached-GQA rollout path for Qwen3.5 models.
    pub memory_efficient_cached_gqa: bool,
    /// Optional quantization mode for frozen base projection weights.
    pub base_quantization: BaseQuantization,
    /// Tensor-parallel loading/execution plan. On Unix, dense Gemma 4 uses it to
    /// stream rank-local projection shards; indexed shards must be flat, regular,
    /// non-symlink `.safetensors` files. Qwen3 keeps replicated frozen weights,
    /// while Qwen3.5 rejects sharded plans.
    pub tensor_parallel: TensorParallelPlan,
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
            memory_efficient_cached_gqa: false,
            base_quantization: BaseQuantization::None,
            tensor_parallel: TensorParallelPlan::single(),
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
    /// `config.json` could not be parsed into the selected model config.
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
    /// The checkpoint declares a model family this loader does not recognize.
    #[error(
        "unsupported model_type {model_type:?}; recognized values are \"qwen3\", \"qwen3_5\", \
         \"qwen3_5_moe\", \"gemma4\", and \"gemma4_unified\""
    )]
    UnsupportedModelType {
        /// The `config.json` `model_type` value, or `None` if the field was absent.
        model_type: Option<String>,
    },
    /// A loader option was requested for a model family that does not implement it.
    #[error("loader option {option} is only supported for {supported}; checkpoint model is {model_type}")]
    UnsupportedLoaderOption {
        /// The unsupported run-config / loader option.
        option: String,
        /// The model family that implements the option.
        supported: String,
        /// The model family selected by the checkpoint.
        model_type: String,
    },
}

/// A concrete policy loaded from any model-family checkpoint the CLI supports.
///
/// The enum keeps `load_qwen_policy` backward-compatible while giving `ferrl train`
/// one model-agnostic return type for Qwen3, Qwen3.5/3.6, and dense Gemma 4 checkpoints.
pub enum AutoPolicy {
    /// A classic dense-attention Qwen3 policy.
    Qwen(Box<QwenPolicy>),
    /// A `qwen3_5` family policy (Qwen3.5 / Qwen3.6).
    Qwen3_5(Box<Qwen3_5Policy>),
    /// A dense Gemma 4 text policy.
    Gemma4(Box<Gemma4Policy>),
}

impl AutoPolicy {
    /// Enable or disable layer-boundary activation checkpointing when the loaded
    /// model supports it.
    ///
    /// This is the CLI-facing memory lever for long CUDA training runs: rollout
    /// remains cached and grad-free, while update forwards rematerialize layer
    /// segments during backward to reduce activation peak memory.
    pub fn set_activation_checkpointing(&mut self, on: bool) {
        match self {
            Self::Qwen(policy) => policy.model_mut().set_activation_checkpointing(on),
            Self::Qwen3_5(policy) => policy.model_mut().set_activation_checkpointing(on),
            Self::Gemma4(policy) => policy.model_mut().set_activation_checkpointing(on),
        }
    }

    /// Whether activation checkpointing is enabled on this model.
    #[must_use]
    pub fn activation_checkpointing(&self) -> bool {
        match self {
            Self::Qwen(policy) => policy.model().activation_checkpointing(),
            Self::Qwen3_5(policy) => policy.model().activation_checkpointing(),
            Self::Gemma4(policy) => policy.model().activation_checkpointing(),
        }
    }

    /// Whether the loaded model family has real tensor-parallel model execution
    /// wired behind [`TensorParallelPolicy`].
    #[must_use]
    pub fn supports_tensor_parallel(&self) -> bool {
        match self {
            Self::Qwen(_) | Self::Gemma4(_) => true,
            Self::Qwen3_5(_) => false,
        }
    }
}

impl Policy for AutoPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        match self {
            Self::Qwen(policy) => policy.generate(prompt, cfg),
            Self::Qwen3_5(policy) => policy.generate(prompt, cfg),
            Self::Gemma4(policy) => policy.generate(prompt, cfg),
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
            Self::Gemma4(policy) => policy.generate_at(prompt, cfg, global_row_base),
        }
    }

    fn generate_at_instrumented(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout> {
        match self {
            Self::Qwen(policy) => {
                policy.generate_at_instrumented(prompt, cfg, global_row_base, telemetry)
            }
            Self::Qwen3_5(policy) => {
                policy.generate_at_instrumented(prompt, cfg, global_row_base, telemetry)
            }
            Self::Gemma4(policy) => {
                policy.generate_at_instrumented(prompt, cfg, global_row_base, telemetry)
            }
        }
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        match self {
            Self::Qwen(policy) => policy.token_logprobs(rollout),
            Self::Qwen3_5(policy) => policy.token_logprobs(rollout),
            Self::Gemma4(policy) => policy.token_logprobs(rollout),
        }
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        match self {
            Self::Qwen(policy) => policy.token_logprobs_detached(rollout),
            Self::Qwen3_5(policy) => policy.token_logprobs_detached(rollout),
            Self::Gemma4(policy) => policy.token_logprobs_detached(rollout),
        }
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        match self {
            Self::Qwen(policy) => policy.backward(loss),
            Self::Qwen3_5(policy) => policy.backward(loss),
            Self::Gemma4(policy) => policy.backward(loss),
        }
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        match self {
            Self::Qwen(policy) => policy.set_adapter_enabled(enabled),
            Self::Qwen3_5(policy) => policy.set_adapter_enabled(enabled),
            Self::Gemma4(policy) => policy.set_adapter_enabled(enabled),
        }
    }

    fn adapter_enabled(&self) -> bool {
        match self {
            Self::Qwen(policy) => policy.adapter_enabled(),
            Self::Qwen3_5(policy) => policy.adapter_enabled(),
            Self::Gemma4(policy) => policy.adapter_enabled(),
        }
    }

    fn trainable_vars(&self) -> Vec<Var> {
        match self {
            Self::Qwen(policy) => policy.trainable_vars(),
            Self::Qwen3_5(policy) => policy.trainable_vars(),
            Self::Gemma4(policy) => policy.trainable_vars(),
        }
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        match self {
            Self::Qwen(policy) => policy.sampler_state(),
            Self::Qwen3_5(policy) => policy.sampler_state(),
            Self::Gemma4(policy) => policy.sampler_state(),
        }
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        match self {
            Self::Qwen(policy) => policy.restore_sampler_state(state),
            Self::Qwen3_5(policy) => policy.restore_sampler_state(state),
            Self::Gemma4(policy) => policy.restore_sampler_state(state),
        }
    }

    fn lora_recipe(&self) -> Option<String> {
        match self {
            Self::Qwen(policy) => policy.lora_recipe(),
            Self::Qwen3_5(policy) => policy.lora_recipe(),
            Self::Gemma4(policy) => policy.lora_recipe(),
        }
    }
}

impl TensorParallelPolicy for AutoPolicy {
    fn generate_at_tensor_parallel_instrumented(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        comm: &dyn crate::Comm,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout> {
        match self {
            Self::Qwen(policy) => policy.generate_at_tensor_parallel_instrumented(
                prompt,
                cfg,
                global_row_base,
                comm,
                telemetry,
            ),
            Self::Qwen3_5(policy) => policy.generate_at_tensor_parallel_instrumented(
                prompt,
                cfg,
                global_row_base,
                comm,
                telemetry,
            ),
            Self::Gemma4(policy) => policy.generate_at_tensor_parallel_instrumented(
                prompt,
                cfg,
                global_row_base,
                comm,
                telemetry,
            ),
        }
    }

    fn token_logprobs_tensor_parallel(
        &self,
        rollout: &Rollout,
        comm: &dyn crate::Comm,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Qwen(policy) => policy.token_logprobs_tensor_parallel(rollout, comm),
            Self::Qwen3_5(policy) => policy.token_logprobs_tensor_parallel(rollout, comm),
            Self::Gemma4(policy) => policy.token_logprobs_tensor_parallel(rollout, comm),
        }
    }

    fn token_logprobs_tensor_parallel_detached(
        &self,
        rollout: &Rollout,
        comm: &dyn crate::Comm,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Qwen(policy) => policy.token_logprobs_tensor_parallel_detached(rollout, comm),
            Self::Qwen3_5(policy) => policy.token_logprobs_tensor_parallel_detached(rollout, comm),
            Self::Gemma4(policy) => policy.token_logprobs_tensor_parallel_detached(rollout, comm),
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
    if opts.memory_efficient_cached_gqa {
        return Err(LoaderError::UnsupportedLoaderOption {
            option: "policy.memory_efficient_cached_gqa".to_string(),
            supported: "qwen3_5".to_string(),
            model_type: "qwen3".to_string(),
        });
    }
    let weights_path = dir.join("model.safetensors");
    let weights = std::fs::read(&weights_path).map_err(|source| LoaderError::Io {
        path: weights_path,
        source,
    })?;
    let vb = VarBuilder::from_buffered_safetensors(weights, opts.base_dtype, device)?;
    let model = QwenGradModel::load_with_targets_and_base_quantization(
        &cfg,
        &vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
        DenseLoraTargets::legacy(),
        opts.base_quantization,
    )?;
    let policy = QwenPolicy::new(model, opts.seed, opts.temperature);

    let tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((policy, tok))
}

/// Load any policy supported by the CLI from a checkpoint directory.
///
/// Dispatches `model_type == "qwen3_5"` / `"qwen3_5_moe"` to the Qwen3.5/3.6
/// loader, `model_type == "gemma4"` / `"gemma4_unified"` to the dense Gemma 4
/// text loader, and `model_type == "qwen3"` (or an absent legacy field) to the
/// original Qwen3 loader.
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
        Some(ModelType::Gemma4) => {
            let (policy, tok) = load_gemma4_policy(dir, device, opts)?;
            Ok((AutoPolicy::Gemma4(Box::new(policy)), tok))
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
    Gemma4,
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
        "gemma4" | "gemma4_unified" => ModelType::Gemma4,
        other => ModelType::Unsupported(Some(other.to_string())),
    }))
}

fn load_qwen35_policy(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(Qwen3_5Policy, HfTokenizer), LoaderError> {
    if opts.base_quantization != BaseQuantization::None {
        return Err(LoaderError::UnsupportedLoaderOption {
            option: "policy.base_quantization".to_string(),
            supported: "qwen3 or gemma4".to_string(),
            model_type: "qwen3_5".to_string(),
        });
    }
    reject_unsupported_sharded_tensor_parallel(opts, "qwen3_5")?;
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json"))?;
    let vb = varbuilder_from_pretrained(dir, opts.base_dtype, device)?;
    let mut model = Qwen3_5GradModel::load_with_targets(
        &cfg,
        &vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
        Qwen35LoraTargets::industrial(),
    )?;
    model.set_memory_efficient_cached_gqa(opts.memory_efficient_cached_gqa);
    let policy = Qwen3_5Policy::new(model, opts.seed, opts.temperature);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((policy, tok))
}

fn read_gemma4_config(dir: &Path) -> Result<Gemma4Config, LoaderError> {
    let cfg_path = dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|source| LoaderError::Io {
        path: cfg_path.clone(),
        source,
    })?;
    let cfg: Gemma4Config =
        serde_json::from_slice(&cfg_bytes).map_err(|source| LoaderError::Config {
            path: cfg_path,
            source,
        })?;
    cfg.validate()?;
    Ok(cfg)
}

fn reject_unsupported_sharded_tensor_parallel(
    opts: &LoaderOpts,
    model_type: &str,
) -> Result<(), LoaderError> {
    if opts.tensor_parallel.is_sharded() {
        return Err(LoaderError::UnsupportedLoaderOption {
            option: "tensor_parallel".to_string(),
            supported: format!(
                "sharded tensor_parallel execution is not supported for {model_type}; use \
                 Qwen3 or dense Gemma 4, or set tensor_parallel.world_size = 1"
            ),
            model_type: model_type.to_string(),
        });
    }
    Ok(())
}

/// Load a dense Gemma 4 text policy and its tokenizer from a checkpoint `dir`.
///
/// The public Gemma 4 checkpoints are conditional-generation wrappers whose
/// text decoder lives under `model.language_model.*`; the native loader validates
/// the wrapper config, maps only those text tensors, attaches the dense
/// industrial `LoRA` recipe, and returns a policy ready for the generic trainer.
/// Sharded tensor-parallel loading is supported only on Unix. Indexed shard values
/// must name flat, regular, non-symlink `.safetensors` files in `dir`; materialize
/// symlink-based cache snapshots into that layout before loading.
///
/// # Errors
///
/// Returns [`LoaderError`] if the config is unsupported, the requested loader
/// options are incompatible with Gemma 4 or the current platform, indexed shards
/// violate the supported file layout, model tensors cannot be mapped, or the
/// tokenizer cannot be loaded.
pub fn load_gemma4_policy(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(Gemma4Policy, HfTokenizer), LoaderError> {
    let cfg = read_gemma4_config(dir)?;
    if opts.memory_efficient_cached_gqa {
        return Err(LoaderError::UnsupportedLoaderOption {
            option: "policy.memory_efficient_cached_gqa".to_string(),
            supported: "qwen3_5".to_string(),
            model_type: cfg
                .model_type
                .clone()
                .unwrap_or_else(|| "gemma4".to_string()),
        });
    }
    let vb = if opts.tensor_parallel.is_sharded() {
        varbuilder_from_rank_local_safetensors(dir, opts.base_dtype, device, opts.tensor_parallel)?
    } else {
        gemma4_varbuilder_from_pretrained(dir, opts.base_dtype, device)?
    };
    let model = Gemma4GradModel::load_with_targets_base_quantization_and_tensor_parallel(
        &cfg,
        &vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
        DenseLoraTargets::industrial(),
        opts.base_quantization,
        opts.tensor_parallel,
    )?;
    let policy = Gemma4Policy::new(model, opts.seed, opts.temperature);
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
    fn gemma4_loader_missing_config_is_an_io_error_naming_the_path() {
        let tmp = TempDir::new("loader-gemma4-missing-config");
        let err = load_gemma4_policy(tmp.path(), &Device::Cpu, &LoaderOpts::default()).unwrap_err();
        match err {
            LoaderError::Io { path, .. } => assert!(path.ends_with("config.json")),
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_loader_malformed_config_is_a_config_error() {
        let tmp = TempDir::new("loader-gemma4-malformed-config");
        std::fs::write(tmp.path().join("config.json"), "{ not json").unwrap();
        let err = load_gemma4_policy(tmp.path(), &Device::Cpu, &LoaderOpts::default()).unwrap_err();
        match err {
            LoaderError::Config { path, .. } => assert!(path.ends_with("config.json")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn default_opts_are_the_poc_recipe() {
        let o = LoaderOpts::default();
        assert_eq!(o.lora_rank, 16);
        assert_eq!(o.base_dtype, DType::F32);
        assert_eq!(o.adapter_dtype, DType::F32);
        assert!(!o.memory_efficient_cached_gqa);
        assert_eq!(o.base_quantization, BaseQuantization::None);
        assert_eq!(o.tensor_parallel, TensorParallelPlan::single());
    }

    fn sharded_tensor_parallel_opts() -> LoaderOpts {
        LoaderOpts {
            tensor_parallel: TensorParallelPlan::new(0, 2).unwrap(),
            ..LoaderOpts::default()
        }
    }

    fn assert_tensor_parallel_rejected(err: LoaderError, expected_model_type: &str) {
        match err {
            LoaderError::UnsupportedLoaderOption {
                option,
                supported,
                model_type,
            } => {
                assert_eq!(option, "tensor_parallel");
                assert!(supported.contains("not supported"));
                assert_eq!(model_type, expected_model_type);
            }
            other => panic!("expected UnsupportedLoaderOption, got {other:?}"),
        }
    }

    #[test]
    fn qwen_loader_keeps_replicated_weight_fallback_under_sharded_execution_plan() {
        let tmp = TempDir::new("loader-qwen3-tensor-parallel-option");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "qwen3", "vocab_size": 16, "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "attention_bias": false, "max_position_embeddings": 32,
                "sliding_window": null, "max_window_layers": 0,
                "tie_word_embeddings": true, "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6, "use_sliding_window": false,
                "hidden_act": "silu" }"#,
        )
        .unwrap();

        let err = load_qwen_policy(tmp.path(), &Device::Cpu, &sharded_tensor_parallel_opts())
            .unwrap_err();

        match err {
            LoaderError::Io { path, .. } => assert!(path.ends_with("model.safetensors")),
            other => panic!("expected replicated Qwen weight read, got {other:?}"),
        }
    }

    #[test]
    fn auto_qwen35_loader_rejects_sharded_tensor_parallel_before_weight_load() {
        let tmp = TempDir::new("loader-qwen35-tensor-parallel-option");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "qwen3_5" }"#,
        )
        .unwrap();

        match load_auto_policy(tmp.path(), &Device::Cpu, &sharded_tensor_parallel_opts()) {
            Err(err) => assert_tensor_parallel_rejected(err, "qwen3_5"),
            Ok(_) => panic!("expected UnsupportedLoaderOption, got loaded policy"),
        }
    }

    #[test]
    fn gemma4_loader_enters_rank_local_safetensors_path() {
        let tmp = TempDir::new("loader-gemma4-tensor-parallel-option");
        std::fs::write(tmp.path().join("config.json"), gemma4_config_json()).unwrap();

        let err = load_gemma4_policy(tmp.path(), &Device::Cpu, &sharded_tensor_parallel_opts())
            .unwrap_err();

        match err {
            LoaderError::Model(error) => assert!(
                error.to_string().contains("neither model.safetensors"),
                "unexpected rank-local loader error: {error}"
            ),
            other => panic!("expected rank-local Gemma safetensors read, got {other:?}"),
        }
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
    fn qwen_loader_rejects_qwen35_cached_gqa_option() {
        let tmp = TempDir::new("loader-qwen3-gqa-option");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "qwen3", "vocab_size": 16, "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "attention_bias": false, "max_position_embeddings": 32,
                "sliding_window": null, "max_window_layers": 0,
                "tie_word_embeddings": true, "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6, "use_sliding_window": false,
                "hidden_act": "silu" }"#,
        )
        .unwrap();
        let opts = LoaderOpts {
            memory_efficient_cached_gqa: true,
            ..LoaderOpts::default()
        };
        match load_auto_policy(tmp.path(), &Device::Cpu, &opts) {
            Err(LoaderError::UnsupportedLoaderOption {
                option,
                supported,
                model_type,
            }) => {
                assert_eq!(option, "policy.memory_efficient_cached_gqa");
                assert_eq!(supported, "qwen3_5");
                assert_eq!(model_type, "qwen3");
            }
            Err(other) => panic!("expected UnsupportedLoaderOption, got {other:?}"),
            Ok(_) => panic!("expected UnsupportedLoaderOption, got loaded policy"),
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
    fn qwen35_loader_rejects_base_quantization_option_before_weight_load() {
        let tmp = TempDir::new("loader-qwen35-base-quantization-option");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "qwen3_5" }"#,
        )
        .unwrap();
        let opts = LoaderOpts {
            base_quantization: BaseQuantization::Q8_0,
            ..LoaderOpts::default()
        };
        match load_auto_policy(tmp.path(), &Device::Cpu, &opts) {
            Err(LoaderError::UnsupportedLoaderOption {
                option,
                supported,
                model_type,
            }) => {
                assert_eq!(option, "policy.base_quantization");
                assert_eq!(supported, "qwen3 or gemma4");
                assert_eq!(model_type, "qwen3_5");
            }
            Err(other) => panic!("expected UnsupportedLoaderOption, got {other:?}"),
            Ok(_) => panic!("expected UnsupportedLoaderOption, got loaded policy"),
        }
    }

    #[test]
    fn model_type_detection_routes_gemma4_family() {
        let tmp = TempDir::new("loader-gemma4-model-type");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "gemma4" }"#,
        )
        .unwrap();
        assert_eq!(
            read_model_type(tmp.path()).unwrap(),
            Some(ModelType::Gemma4)
        );
    }

    #[test]
    fn model_type_detection_routes_gemma4_unified_family() {
        let tmp = TempDir::new("loader-gemma4-unified-model-type");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{ "model_type": "gemma4_unified" }"#,
        )
        .unwrap();
        assert_eq!(
            read_model_type(tmp.path()).unwrap(),
            Some(ModelType::Gemma4)
        );
    }

    #[test]
    fn auto_loader_reaches_gemma4_weight_load_after_config_validation() {
        let tmp = TempDir::new("loader-gemma4-recognized");
        std::fs::write(tmp.path().join("config.json"), gemma4_config_json()).unwrap();
        match load_auto_policy(tmp.path(), &Device::Cpu, &LoaderOpts::default()) {
            Err(LoaderError::Model(e)) => {
                let msg = e.to_string();
                assert!(msg.contains("gemma4 loader"));
                assert!(msg.contains("model.safetensors"));
            }
            Err(other) => panic!("expected Gemma4 model-load error, got {other:?}"),
            Ok(_) => panic!("expected Gemma4 model-load error, got loaded policy"),
        }
    }

    #[test]
    fn auto_loader_reaches_gemma4_unified_weight_load_after_config_validation() {
        let tmp = TempDir::new("loader-gemma4-unified-recognized");
        std::fs::write(tmp.path().join("config.json"), gemma4_unified_config_json()).unwrap();
        match load_auto_policy(tmp.path(), &Device::Cpu, &LoaderOpts::default()) {
            Err(LoaderError::Model(e)) => {
                let msg = e.to_string();
                assert!(msg.contains("gemma4 loader"));
                assert!(msg.contains("model.safetensors"));
            }
            Err(other) => panic!("expected Gemma4 model-load error, got {other:?}"),
            Ok(_) => panic!("expected Gemma4 model-load error, got loaded policy"),
        }
    }

    #[test]
    fn gemma4_loader_rejects_qwen35_cached_gqa_option() {
        let tmp = TempDir::new("loader-gemma4-gqa-option");
        std::fs::write(tmp.path().join("config.json"), gemma4_config_json()).unwrap();
        let opts = LoaderOpts {
            memory_efficient_cached_gqa: true,
            ..LoaderOpts::default()
        };
        match load_auto_policy(tmp.path(), &Device::Cpu, &opts) {
            Err(LoaderError::UnsupportedLoaderOption {
                option,
                supported,
                model_type,
            }) => {
                assert_eq!(option, "policy.memory_efficient_cached_gqa");
                assert_eq!(supported, "qwen3_5");
                assert_eq!(model_type, "gemma4");
            }
            Err(other) => panic!("expected UnsupportedLoaderOption, got {other:?}"),
            Ok(_) => panic!("expected UnsupportedLoaderOption, got loaded policy"),
        }
    }

    #[test]
    fn gemma4_unified_loader_rejects_qwen35_cached_gqa_option_with_unified_model_type() {
        let tmp = TempDir::new("loader-gemma4-unified-gqa-option");
        std::fs::write(tmp.path().join("config.json"), gemma4_unified_config_json()).unwrap();
        let opts = LoaderOpts {
            memory_efficient_cached_gqa: true,
            ..LoaderOpts::default()
        };
        match load_auto_policy(tmp.path(), &Device::Cpu, &opts) {
            Err(LoaderError::UnsupportedLoaderOption {
                option,
                supported,
                model_type,
            }) => {
                assert_eq!(option, "policy.memory_efficient_cached_gqa");
                assert_eq!(supported, "qwen3_5");
                assert_eq!(model_type, "gemma4_unified");
            }
            Err(other) => panic!("expected UnsupportedLoaderOption, got {other:?}"),
            Ok(_) => panic!("expected UnsupportedLoaderOption, got loaded policy"),
        }
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
        assert!(!policy.supports_tensor_parallel());
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

    fn gemma4_config_json() -> &'static str {
        r#"{
            "model_type": "gemma4",
            "tie_word_embeddings": true,
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attention_k_eq_v": true,
                "enable_moe_block": false,
                "expert_intermediate_size": null,
                "final_logit_softcapping": 30.0,
                "global_head_dim": 512,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 5376,
                "hidden_size_per_layer_input": 0,
                "intermediate_size": 21504,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention"
                ],
                "max_position_embeddings": 262144,
                "model_type": "gemma4_text",
                "num_attention_heads": 32,
                "num_experts": null,
                "num_global_key_value_heads": 4,
                "num_hidden_layers": 6,
                "num_key_value_heads": 16,
                "num_kv_shared_layers": 0,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                },
                "sliding_window": 1024,
                "tie_word_embeddings": true,
                "top_k_experts": null,
                "use_bidirectional_attention": "vision",
                "use_cache": true,
                "use_double_wide_mlp": false,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144
            }
        }"#
    }

    fn gemma4_unified_config_json() -> String {
        gemma4_config_json()
            .replacen(
                "\"model_type\": \"gemma4\"",
                "\"model_type\": \"gemma4_unified\"",
                1,
            )
            .replace(
                "\"model_type\": \"gemma4_text\"",
                "\"model_type\": \"gemma4_unified_text\"",
            )
    }
}
