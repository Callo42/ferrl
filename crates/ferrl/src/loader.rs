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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use sha2::{Digest, Sha256};

use crate::gemma4::{Gemma4Config, Gemma4GradModel, CKPT_PREFIX as GEMMA4_CKPT_PREFIX};
use crate::lm_policy::{Gemma4Policy, Qwen3_5Policy, QwenPolicy};
use crate::lora::{BaseQuantization, DenseLoraTargets};
use crate::policy::{GenConfig, Policy, Rollout, TensorParallelPolicy};
use crate::qwen::QwenGradModel;
use crate::qwen35::{LoraTargets as Qwen35LoraTargets, Qwen3_5Config, Qwen3_5GradModel};
use crate::sharded_safetensors::{
    varbuilder_from_rank_local_safetensors_bound, BoundSafetensorsIdentity,
};
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
    /// Rematerialize layer activations during the trainable forward/backward.
    /// This changes execution semantics and is therefore part of the bound
    /// checkpoint policy identity.
    pub activation_checkpointing: bool,
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
            activation_checkpointing: false,
            tensor_parallel: TensorParallelPlan::single(),
        }
    }
}

#[derive(Debug)]
struct CapturedShard {
    name: String,
    bytes: Vec<u8>,
}

/// Immutable, buffered model source. These exact owned bytes feed both the
/// identity and the model builder, so path replacement after capture cannot
/// create a digest-A/policy-B load.
#[derive(Debug)]
struct CapturedPolicySource {
    config_bytes: Vec<u8>,
    weight_map: Vec<(String, String)>,
    shards: Vec<CapturedShard>,
}

impl CapturedPolicySource {
    fn capture(dir: &Path) -> Result<Self, LoaderError> {
        let config_path = dir.join("config.json");
        let config_bytes = read_regular_file(&config_path)?;
        let model_type = model_type_from_bytes(&config_bytes, &config_path)?;
        Self::capture_with_config(dir, config_bytes, model_type.as_ref())
    }

    fn capture_with_config(
        dir: &Path,
        config_bytes: Vec<u8>,
        model_type: Option<&ModelType>,
    ) -> Result<Self, LoaderError> {
        let index_path = dir.join("model.safetensors.index.json");
        let uses_index =
            !matches!(&model_type, None | Some(ModelType::Qwen3)) && index_path.is_file();
        if !uses_index {
            let name = "model.safetensors".to_string();
            let bytes = read_regular_file(&dir.join(&name))?;
            let weight_map = if matches!(&model_type, Some(ModelType::Gemma4)) {
                let tensors = safetensors::SafeTensors::deserialize(&bytes)
                    .map_err(|source| LoaderError::Model(candle_core::Error::from(source)))?;
                let mut selected = tensors
                    .names()
                    .into_iter()
                    .filter(|tensor| is_gemma4_text_tensor(tensor))
                    .map(|tensor| (tensor.to_string(), name.clone()))
                    .collect::<Vec<_>>();
                selected.sort();
                if selected.is_empty() {
                    return Err(invalid_data(
                        dir.join(&name),
                        "single-file Gemma checkpoint has no selected text tensors",
                    ));
                }
                selected
            } else {
                vec![("*".to_string(), name.clone())]
            };
            return Ok(Self {
                config_bytes,
                weight_map,
                shards: vec![CapturedShard { bytes, name }],
            });
        }

        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let index_bytes = read_regular_file(&index_path)?;
        let index: Index =
            serde_json::from_slice(&index_bytes).map_err(|source| LoaderError::Config {
                path: index_path.clone(),
                source,
            })?;
        let selected = |tensor: &str| match &model_type {
            Some(ModelType::Gemma4) => is_gemma4_text_tensor(tensor),
            _ => true,
        };
        let weight_map = index
            .weight_map
            .into_iter()
            .filter(|(tensor, _)| selected(tensor))
            .collect::<Vec<_>>();
        if weight_map.is_empty() {
            return Err(invalid_data(
                index_path,
                "safetensors index has no tensors selected by this model loader",
            ));
        }
        let shard_names = weight_map
            .iter()
            .map(|(_, shard)| shard.clone())
            .collect::<BTreeSet<_>>();
        let mut shards = Vec::with_capacity(shard_names.len());
        for name in shard_names {
            validate_shard_name(&index_path, &name)?;
            shards.push(CapturedShard {
                bytes: read_regular_file(&dir.join(&name))?,
                name,
            });
        }
        Ok(Self {
            config_bytes,
            weight_map,
            shards,
        })
    }

    fn digest(&self, opts: &LoaderOpts) -> Result<String, LoaderError> {
        let shards = self.shards.iter().map(|shard| {
            let digest: [u8; 32] = Sha256::digest(&shard.bytes).into();
            (shard.name.clone(), shard.bytes.len() as u64, digest)
        });
        checkpoint_policy_digest(&self.config_bytes, &self.weight_map, shards, opts)
    }

    fn into_single_varbuilder(
        self,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<VarBuilder<'static>> {
        let shard =
            self.shards.into_iter().next().ok_or_else(|| {
                candle_core::Error::Msg("captured checkpoint has no shard".into())
            })?;
        VarBuilder::from_buffered_safetensors(shard.bytes, dtype, device)
    }

    fn into_tensor_varbuilder(
        self,
        dtype: DType,
        device: &Device,
        keep: impl Fn(&str) -> bool,
    ) -> CandleResult<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        for shard in self.shards {
            let loaded = candle_core::safetensors::load_buffer(&shard.bytes, device)?;
            tensors.extend(loaded.into_iter().filter(|(name, _)| keep(name)));
        }
        if tensors.is_empty() {
            return Err(candle_core::Error::Msg(
                "captured checkpoint has no selected tensors".into(),
            ));
        }
        Ok(VarBuilder::from_tensors(tensors, dtype, device))
    }
}

/// Compute an offline checkpoint-directory fingerprint.
///
/// This helper is useful for inspection and tests, but production resume must
/// use one of the `*_bound` loaders so the returned identity is derived from the
/// exact captured bytes/handles used to construct the policy.
///
/// # Errors
///
/// Returns [`LoaderError`] when a selected source is missing, unreadable,
/// malformed, non-regular, or uses an unsafe shard name.
pub fn checkpoint_policy_sha256(dir: &Path, opts: &LoaderOpts) -> Result<String, LoaderError> {
    CapturedPolicySource::capture(dir)?.digest(opts)
}

fn checkpoint_policy_digest(
    config_bytes: &[u8],
    weight_map: &[(String, String)],
    shards: impl IntoIterator<Item = (String, u64, [u8; 32])>,
    opts: &LoaderOpts,
) -> Result<String, LoaderError> {
    let recipe = serde_json::to_vec(&serde_json::json!({
        "lora_rank": opts.lora_rank,
        "lora_alpha_bits": opts.lora_alpha.to_bits(),
        "base_dtype": opts.base_dtype.as_str(),
        "adapter_dtype": opts.adapter_dtype.as_str(),
        "memory_efficient_cached_gqa": opts.memory_efficient_cached_gqa,
        "base_quantization": opts.base_quantization.as_str(),
        "activation_checkpointing": opts.activation_checkpointing,
        "tensor_parallel_world_size": opts.tensor_parallel.world_size(),
        "tensor_parallel_layout": crate::checkpoint::ROLLOUT_LEDGER_CONTINUATION_LAYOUT,
    }))
    .map_err(|source| LoaderError::Config {
        path: PathBuf::from("config.json"),
        source,
    })?;
    let mut hasher = Sha256::new();
    identity_field(&mut hasher, b"ferrl.checkpoint-policy.v2");
    identity_field(&mut hasher, &recipe);
    identity_field(&mut hasher, b"config.json");
    identity_field(&mut hasher, config_bytes);
    identity_field(&mut hasher, b"selected-weight-map");
    for (tensor, shard) in weight_map {
        identity_field(&mut hasher, tensor.as_bytes());
        identity_field(&mut hasher, shard.as_bytes());
    }
    for (name, len, digest) in shards {
        identity_field(&mut hasher, b"selected-shard");
        identity_field(&mut hasher, name.as_bytes());
        hasher.update(len.to_le_bytes());
        hasher.update(digest);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, LoaderError> {
    let path_metadata = std::fs::symlink_metadata(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !path_metadata.file_type().is_file() {
        return Err(invalid_data(
            path.to_path_buf(),
            "checkpoint source is not a regular file",
        ));
    }
    let mut file = std::fs::File::open(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_metadata = file.metadata().map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if path_metadata.dev() != file_metadata.dev() || path_metadata.ino() != file_metadata.ino()
        {
            return Err(invalid_data(
                path.to_path_buf(),
                "checkpoint source changed while it was being opened",
            ));
        }
    }
    let expected_len = file_metadata.len();
    let mut bytes = Vec::with_capacity(usize::try_from(expected_len).unwrap_or(0));
    file.read_to_end(&mut bytes)
        .map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 != expected_len {
        return Err(invalid_data(
            path.to_path_buf(),
            "checkpoint source changed length while being captured",
        ));
    }
    Ok(bytes)
}

fn validate_shard_name(index_path: &Path, shard: &str) -> Result<(), LoaderError> {
    let path = Path::new(shard);
    let safe = path.components().count() == 1
        && matches!(path.components().next(), Some(Component::Normal(_)))
        && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors");
    if safe {
        Ok(())
    } else {
        Err(invalid_data(
            index_path.to_path_buf(),
            format!("unsafe safetensors shard name {shard:?}"),
        ))
    }
}

fn invalid_data(path: PathBuf, message: impl Into<String>) -> LoaderError {
    LoaderError::Io {
        path,
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()),
    }
}

fn is_gemma4_text_tensor(name: &str) -> bool {
    name.strip_prefix(GEMMA4_CKPT_PREFIX)
        .is_some_and(|suffix| suffix.starts_with('.'))
}

fn identity_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
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

    /// Whether this model family can replay activation-checkpointed update
    /// layers through the explicit tensor-parallel communicator.
    ///
    /// This is a family-static loader capability, not the live policy
    /// instance's current checkpointing setting. Use
    /// [`activation_checkpointing`](Self::activation_checkpointing) for the
    /// latter.
    #[must_use]
    pub fn model_family_supports_tensor_parallel_activation_checkpointing(&self) -> bool {
        matches!(self, Self::Gemma4(_))
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

    fn requires_rollout_tensor_snapshot(&self) -> bool {
        match self {
            Self::Qwen(policy) => policy.requires_rollout_tensor_snapshot(),
            Self::Qwen3_5(policy) => policy.requires_rollout_tensor_snapshot(),
            Self::Gemma4(policy) => policy.requires_rollout_tensor_snapshot(),
        }
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        match self {
            Self::Qwen(policy) => policy.sampler_state(),
            Self::Qwen3_5(policy) => policy.sampler_state(),
            Self::Gemma4(policy) => policy.sampler_state(),
        }
    }

    fn validate_sampler_state(&self, state: &[u8]) -> CandleResult<()> {
        match self {
            Self::Qwen(policy) => policy.validate_sampler_state(state),
            Self::Qwen3_5(policy) => policy.validate_sampler_state(state),
            Self::Gemma4(policy) => policy.validate_sampler_state(state),
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
    fn validate_tensor_parallel_execution(&self, comm: &dyn crate::Comm) -> CandleResult<()> {
        match self {
            Self::Qwen(policy) => policy.validate_tensor_parallel_execution(comm),
            Self::Qwen3_5(policy) => policy.validate_tensor_parallel_execution(comm),
            Self::Gemma4(policy) => policy.validate_tensor_parallel_execution(comm),
        }
    }

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

    fn backward_tensor_parallel(
        &self,
        loss: &Tensor,
        comm: &dyn crate::Comm,
    ) -> CandleResult<GradStore> {
        match self {
            Self::Qwen(policy) => policy.backward_tensor_parallel(loss, comm),
            Self::Qwen3_5(policy) => policy.backward_tensor_parallel(loss, comm),
            Self::Gemma4(policy) => policy.backward_tensor_parallel(loss, comm),
        }
    }

    fn supports_sharded_tensor_parallel_backward(&self) -> bool {
        match self {
            Self::Qwen(policy) => policy.supports_sharded_tensor_parallel_backward(),
            Self::Qwen3_5(policy) => policy.supports_sharded_tensor_parallel_backward(),
            Self::Gemma4(policy) => policy.supports_sharded_tensor_parallel_backward(),
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
    let (policy, tokenizer, _) = load_qwen_policy_bound(dir, device, opts)?;
    Ok((policy, tokenizer))
}

/// Load Qwen3 and return the identity derived from the same captured config and
/// weight bytes used to construct the policy.
///
/// # Errors
///
/// As [`load_qwen_policy`].
pub fn load_qwen_policy_bound(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(QwenPolicy, HfTokenizer, String), LoaderError> {
    let config_path = dir.join("config.json");
    let config_bytes = read_regular_file(&config_path)?;
    validate_qwen3_loader_options(opts)?;
    validate_qwen3_config_bytes(&config_bytes, &config_path)?;
    let model_type = model_type_from_bytes(&config_bytes, &config_path)?;
    let source = CapturedPolicySource::capture_with_config(dir, config_bytes, model_type.as_ref())?;
    let digest = source.digest(opts)?;
    load_qwen_policy_from_source(dir, device, opts, source, digest)
}

fn load_qwen_policy_from_source(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
    source: CapturedPolicySource,
    digest: String,
) -> Result<(QwenPolicy, HfTokenizer, String), LoaderError> {
    let cfg_path = dir.join("config.json");
    let cfg = parse_qwen3_config_bytes(&source.config_bytes, &cfg_path)?;
    validate_qwen3_loader_options(opts)?;
    let vb = source.into_single_varbuilder(opts.base_dtype, device)?;
    let mut model = QwenGradModel::load_with_targets_and_base_quantization(
        &cfg,
        &vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
        DenseLoraTargets::legacy(),
        opts.base_quantization,
    )?;
    model.set_activation_checkpointing(opts.activation_checkpointing);
    let policy = QwenPolicy::new(model, opts.seed, opts.temperature);

    let tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((policy, tok, digest))
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
    let (policy, tokenizer, _) = load_auto_policy_bound(dir, device, opts)?;
    Ok((policy, tokenizer))
}

/// Load any supported production policy and return its identity from the exact
/// captured bytes/handles consumed by that load.
///
/// # Errors
///
/// As [`load_auto_policy`].
pub fn load_auto_policy_bound(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(AutoPolicy, HfTokenizer, String), LoaderError> {
    load_auto_policy_bound_inner(dir, device, opts, || {})
}

fn load_auto_policy_bound_inner(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
    after_identity: impl FnOnce(),
) -> Result<(AutoPolicy, HfTokenizer, String), LoaderError> {
    let config_path = dir.join("config.json");
    let config_bytes = read_regular_file(&config_path)?;
    let model_type = model_type_from_bytes(&config_bytes, &config_path)?;
    if let Some(ModelType::Unsupported(model_type)) = &model_type {
        return Err(LoaderError::UnsupportedModelType {
            model_type: model_type.clone(),
        });
    }
    match &model_type {
        None | Some(ModelType::Qwen3) => {
            validate_qwen3_loader_options(opts)?;
            validate_qwen3_config_bytes(&config_bytes, &config_path)?;
        }
        Some(ModelType::Qwen3_5) => {
            validate_qwen35_loader_options(opts)?;
            validate_qwen35_config_bytes(&config_bytes, &config_path)?;
        }
        Some(ModelType::Gemma4) => {
            let cfg = read_gemma4_config_bytes(&config_bytes, &config_path)?;
            validate_gemma4_loader_options(opts, &cfg)?;
        }
        Some(ModelType::Unsupported(_)) => unreachable!("unsupported model type returned above"),
    }
    if matches!(&model_type, Some(ModelType::Gemma4)) && opts.tensor_parallel.is_sharded() {
        let loaded =
            load_gemma4_policy_streaming_bound(dir, device, opts, &config_bytes, after_identity)?;
        return Ok((AutoPolicy::Gemma4(Box::new(loaded.0)), loaded.1, loaded.2));
    }
    let source = CapturedPolicySource::capture_with_config(dir, config_bytes, model_type.as_ref())?;
    let digest = source.digest(opts)?;
    after_identity();
    match model_type {
        None | Some(ModelType::Qwen3) => {
            let (policy, tok, digest) =
                load_qwen_policy_from_source(dir, device, opts, source, digest)?;
            Ok((AutoPolicy::Qwen(Box::new(policy)), tok, digest))
        }
        Some(ModelType::Qwen3_5) => {
            let (policy, tok, digest) = load_qwen35_policy_from_source(
                dir,
                device,
                opts,
                source,
                Qwen35LoraTargets::industrial(),
                digest,
            )?;
            Ok((AutoPolicy::Qwen3_5(Box::new(policy)), tok, digest))
        }
        Some(ModelType::Gemma4) => {
            let (policy, tok, digest) =
                load_gemma4_policy_from_source(dir, device, opts, source, digest)?;
            Ok((AutoPolicy::Gemma4(Box::new(policy)), tok, digest))
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

#[cfg(test)]
fn read_model_type(dir: &Path) -> Result<Option<ModelType>, LoaderError> {
    let cfg_path = dir.join("config.json");
    let cfg_bytes = read_regular_file(&cfg_path)?;
    model_type_from_bytes(&cfg_bytes, &cfg_path)
}

fn model_type_from_bytes(
    cfg_bytes: &[u8],
    cfg_path: &Path,
) -> Result<Option<ModelType>, LoaderError> {
    let value: serde_json::Value =
        serde_json::from_slice(cfg_bytes).map_err(|source| LoaderError::Config {
            path: cfg_path.to_path_buf(),
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

/// Load a Qwen3.5/3.6 policy with an explicit adapter target recipe and return
/// an identity derived from the exact captured bytes used by the model.
///
/// # Errors
///
/// As [`load_auto_policy`].
pub fn load_qwen35_policy_with_targets_bound(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
    targets: Qwen35LoraTargets,
) -> Result<(Qwen3_5Policy, HfTokenizer, String), LoaderError> {
    let config_path = dir.join("config.json");
    let config_bytes = read_regular_file(&config_path)?;
    validate_qwen35_loader_options(opts)?;
    validate_qwen35_config_bytes(&config_bytes, &config_path)?;
    let model_type = model_type_from_bytes(&config_bytes, &config_path)?;
    let source = CapturedPolicySource::capture_with_config(dir, config_bytes, model_type.as_ref())?;
    let digest = source.digest(opts)?;
    load_qwen35_policy_from_source(dir, device, opts, source, targets, digest)
}

fn load_qwen35_policy_from_source(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
    source: CapturedPolicySource,
    targets: Qwen35LoraTargets,
    digest: String,
) -> Result<(Qwen3_5Policy, HfTokenizer, String), LoaderError> {
    validate_qwen35_loader_options(opts)?;
    let cfg = parse_qwen35_config_bytes(&source.config_bytes, &dir.join("config.json"))?;
    let vb = source.into_tensor_varbuilder(opts.base_dtype, device, |_| true)?;
    let mut model = Qwen3_5GradModel::load_with_targets(
        &cfg,
        &vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
        targets,
    )?;
    model.set_memory_efficient_cached_gqa(opts.memory_efficient_cached_gqa);
    model.set_activation_checkpointing(opts.activation_checkpointing);
    let policy = Qwen3_5Policy::new(model, opts.seed, opts.temperature);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((policy, tok, digest))
}

fn read_gemma4_config_bytes(
    cfg_bytes: &[u8],
    cfg_path: &Path,
) -> Result<Gemma4Config, LoaderError> {
    let cfg: Gemma4Config =
        serde_json::from_slice(cfg_bytes).map_err(|source| LoaderError::Config {
            path: cfg_path.to_path_buf(),
            source,
        })?;
    cfg.validate()?;
    Ok(cfg)
}

fn parse_qwen3_config_bytes(cfg_bytes: &[u8], cfg_path: &Path) -> Result<Config, LoaderError> {
    serde_json::from_slice(cfg_bytes).map_err(|source| LoaderError::Config {
        path: cfg_path.to_path_buf(),
        source,
    })
}

fn validate_qwen3_config_bytes(cfg_bytes: &[u8], cfg_path: &Path) -> Result<(), LoaderError> {
    parse_qwen3_config_bytes(cfg_bytes, cfg_path).map(|_| ())
}

fn parse_qwen35_config_bytes(
    cfg_bytes: &[u8],
    cfg_path: &Path,
) -> Result<Qwen3_5Config, LoaderError> {
    let cfg_json = std::str::from_utf8(cfg_bytes).map_err(|source| LoaderError::Config {
        path: cfg_path.to_path_buf(),
        source: serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, source)),
    })?;
    Qwen3_5Config::from_json_str(cfg_json).map_err(LoaderError::Model)
}

fn validate_qwen35_config_bytes(cfg_bytes: &[u8], cfg_path: &Path) -> Result<(), LoaderError> {
    parse_qwen35_config_bytes(cfg_bytes, cfg_path).map(|_| ())
}

fn validate_qwen3_loader_options(opts: &LoaderOpts) -> Result<(), LoaderError> {
    if opts.memory_efficient_cached_gqa {
        return Err(LoaderError::UnsupportedLoaderOption {
            option: "policy.memory_efficient_cached_gqa".to_string(),
            supported: "qwen3_5".to_string(),
            model_type: "qwen3".to_string(),
        });
    }
    Ok(())
}

fn validate_qwen35_loader_options(opts: &LoaderOpts) -> Result<(), LoaderError> {
    if opts.base_quantization != BaseQuantization::None {
        return Err(LoaderError::UnsupportedLoaderOption {
            option: "policy.base_quantization".to_string(),
            supported: "qwen3 or gemma4".to_string(),
            model_type: "qwen3_5".to_string(),
        });
    }
    reject_unsupported_sharded_tensor_parallel(opts, "qwen3_5")
}

fn validate_gemma4_loader_options(
    opts: &LoaderOpts,
    cfg: &Gemma4Config,
) -> Result<(), LoaderError> {
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
    Ok(())
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
    let (policy, tokenizer, _) = load_gemma4_policy_bound(dir, device, opts)?;
    Ok((policy, tokenizer))
}

/// Load a Gemma 4 text policy and return the identity derived from the exact
/// selected buffered bytes or retained streaming handles used by the model.
///
/// # Errors
///
/// As [`load_gemma4_policy`].
pub fn load_gemma4_policy_bound(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
) -> Result<(Gemma4Policy, HfTokenizer, String), LoaderError> {
    let config_path = dir.join("config.json");
    let config_bytes = read_regular_file(&config_path)?;
    let model_type = model_type_from_bytes(&config_bytes, &config_path)?;
    let cfg = read_gemma4_config_bytes(&config_bytes, &config_path)?;
    validate_gemma4_loader_options(opts, &cfg)?;
    if opts.tensor_parallel.is_sharded() {
        return load_gemma4_policy_streaming_bound(dir, device, opts, &config_bytes, || {});
    }
    let source = CapturedPolicySource::capture_with_config(dir, config_bytes, model_type.as_ref())?;
    let digest = source.digest(opts)?;
    load_gemma4_policy_from_source(dir, device, opts, source, digest)
}

fn load_gemma4_policy_from_source(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
    source: CapturedPolicySource,
    digest: String,
) -> Result<(Gemma4Policy, HfTokenizer, String), LoaderError> {
    let cfg = read_gemma4_config_bytes(&source.config_bytes, &dir.join("config.json"))?;
    validate_gemma4_loader_options(opts, &cfg)?;
    let vb = source.into_tensor_varbuilder(opts.base_dtype, device, is_gemma4_text_tensor)?;
    build_gemma4_policy(dir, device, opts, &cfg, &vb, digest)
}

fn load_gemma4_policy_streaming_bound(
    dir: &Path,
    device: &Device,
    opts: &LoaderOpts,
    config_bytes: &[u8],
    after_identity: impl FnOnce(),
) -> Result<(Gemma4Policy, HfTokenizer, String), LoaderError> {
    let cfg = read_gemma4_config_bytes(config_bytes, &dir.join("config.json"))?;
    validate_gemma4_loader_options(opts, &cfg)?;
    let (vb, identity) = varbuilder_from_rank_local_safetensors_bound(
        dir,
        opts.base_dtype,
        device,
        opts.tensor_parallel,
        is_gemma4_text_tensor,
    )?;
    let BoundSafetensorsIdentity { weight_map, shards } = identity;
    let digest = checkpoint_policy_digest(config_bytes, &weight_map, shards, opts)?;
    after_identity();
    build_gemma4_policy(dir, device, opts, &cfg, &vb, digest)
}

fn build_gemma4_policy(
    dir: &Path,
    _device: &Device,
    opts: &LoaderOpts,
    cfg: &Gemma4Config,
    vb: &VarBuilder<'static>,
    digest: String,
) -> Result<(Gemma4Policy, HfTokenizer, String), LoaderError> {
    let mut model = Gemma4GradModel::load_with_targets_base_quantization_and_tensor_parallel(
        cfg,
        vb,
        opts.lora_rank,
        opts.lora_alpha,
        opts.adapter_dtype,
        DenseLoraTargets::industrial(),
        opts.base_quantization,
        opts.tensor_parallel,
    )?;
    model.set_activation_checkpointing(opts.activation_checkpointing);
    let policy = Gemma4Policy::new(model, opts.seed, opts.temperature);
    let tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((policy, tok, digest))
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
    #[allow(clippy::cognitive_complexity)] // one assertion per identity-bound loader default
    fn default_opts_are_the_poc_recipe() {
        let o = LoaderOpts::default();
        assert_eq!(o.lora_rank, 16);
        assert_eq!(o.base_dtype, DType::F32);
        assert_eq!(o.adapter_dtype, DType::F32);
        assert!(!o.memory_efficient_cached_gqa);
        assert_eq!(o.base_quantization, BaseQuantization::None);
        assert!(!o.activation_checkpointing);
        assert_eq!(o.tensor_parallel, TensorParallelPlan::single());
    }

    fn write_identity_fixture(dir: &Path, model_bytes: &[u8]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), br#"{"model_type":"qwen3"}"#).unwrap();
        std::fs::write(dir.join("model.safetensors"), model_bytes).unwrap();
    }

    #[test]
    fn checkpoint_policy_identity_is_path_independent_and_content_sensitive() {
        let tmp = TempDir::new("checkpoint-policy-identity");
        let first = tmp.path().join("first");
        let copied = tmp.path().join("copied");
        write_identity_fixture(&first, b"same immutable model bytes");
        write_identity_fixture(&copied, b"same immutable model bytes");
        let opts = LoaderOpts::default();
        let expected = checkpoint_policy_sha256(&first, &opts).unwrap();
        assert_eq!(checkpoint_policy_sha256(&copied, &opts).unwrap(), expected);

        std::fs::write(
            copied.join("model.safetensors"),
            b"changed immutable model bytes",
        )
        .unwrap();
        assert_ne!(checkpoint_policy_sha256(&copied, &opts).unwrap(), expected);
        write_identity_fixture(&copied, b"same immutable model bytes");
        std::fs::write(
            copied.join("config.json"),
            br#"{"model_type":"qwen3","changed":true}"#,
        )
        .unwrap();
        assert_ne!(checkpoint_policy_sha256(&copied, &opts).unwrap(), expected);
    }

    #[test]
    fn checkpoint_policy_identity_binds_execution_recipe_but_not_sampler_position() {
        let tmp = TempDir::new("checkpoint-policy-options");
        write_identity_fixture(tmp.path(), b"model");
        let base = LoaderOpts::default();
        let expected = checkpoint_policy_sha256(tmp.path(), &base).unwrap();

        let execution_mutations = [
            (
                "LoRA rank",
                LoaderOpts {
                    lora_rank: base.lora_rank + 1,
                    ..base.clone()
                },
            ),
            (
                "LoRA alpha",
                LoaderOpts {
                    lora_alpha: base.lora_alpha + 1.0,
                    ..base.clone()
                },
            ),
            (
                "base dtype",
                LoaderOpts {
                    base_dtype: DType::BF16,
                    ..base.clone()
                },
            ),
            (
                "adapter dtype",
                LoaderOpts {
                    adapter_dtype: DType::BF16,
                    ..base.clone()
                },
            ),
            (
                "cached GQA recipe",
                LoaderOpts {
                    memory_efficient_cached_gqa: true,
                    ..base.clone()
                },
            ),
            (
                "base quantization",
                LoaderOpts {
                    base_quantization: BaseQuantization::Q8_0,
                    ..base.clone()
                },
            ),
            (
                "activation checkpointing",
                LoaderOpts {
                    activation_checkpointing: true,
                    ..base.clone()
                },
            ),
            (
                "tensor-parallel world",
                LoaderOpts {
                    tensor_parallel: TensorParallelPlan::new(0, 2).unwrap(),
                    ..base.clone()
                },
            ),
        ];
        for (label, changed) in execution_mutations {
            assert_ne!(
                checkpoint_policy_sha256(tmp.path(), &changed).unwrap(),
                expected,
                "{label} was omitted from the checkpoint policy identity"
            );
        }
        let sampler_only = LoaderOpts {
            seed: base.seed.wrapping_add(1),
            temperature: 0.7,
            ..base.clone()
        };
        assert_eq!(
            checkpoint_policy_sha256(tmp.path(), &sampler_only).unwrap(),
            expected
        );
        let rank_zero = LoaderOpts {
            tensor_parallel: TensorParallelPlan::new(0, 2).unwrap(),
            ..base
        };
        let other_tp_rank = LoaderOpts {
            tensor_parallel: TensorParallelPlan::new(1, 2).unwrap(),
            ..rank_zero.clone()
        };
        assert_eq!(
            checkpoint_policy_sha256(tmp.path(), &other_tp_rank).unwrap(),
            checkpoint_policy_sha256(tmp.path(), &rank_zero).unwrap(),
            "TP rank must not split one logical policy identity"
        );
    }

    #[test]
    fn checkpoint_policy_identity_binds_every_indexed_shard_and_rejects_unsafe_names() {
        let tmp = TempDir::new("checkpoint-policy-indexed");
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{"model_type":"qwen3_5"}"#,
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"layer.0":"model-00001-of-00002.safetensors","layer.1":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let first = tmp.path().join("model-00001-of-00002.safetensors");
        let second = tmp.path().join("model-00002-of-00002.safetensors");
        std::fs::write(&first, b"first shard").unwrap();
        std::fs::write(&second, b"second shard").unwrap();
        let expected = checkpoint_policy_sha256(tmp.path(), &LoaderOpts::default()).unwrap();
        for (label, path, original) in [
            ("first", &first, &b"first shard"[..]),
            ("second", &second, &b"second shard"[..]),
        ] {
            std::fs::write(path, format!("changed {label} shard")).unwrap();
            assert_ne!(
                checkpoint_policy_sha256(tmp.path(), &LoaderOpts::default()).unwrap(),
                expected,
                "{label} indexed shard was omitted from the identity"
            );
            std::fs::write(path, original).unwrap();
        }

        // The same referenced shard set with different tensor-to-shard coordinates
        // is a different checkpoint layout and must not collide.
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"layer.0":"model-00002-of-00002.safetensors","layer.1":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        assert_ne!(
            checkpoint_policy_sha256(tmp.path(), &LoaderOpts::default()).unwrap(),
            expected,
            "indexed weight-map coordinates were omitted from the identity"
        );

        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"layer.0":"../outside.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            checkpoint_policy_sha256(tmp.path(), &LoaderOpts::default()),
            Err(LoaderError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::InvalidData
        ));
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
    fn gemma4_policy_advertises_tensor_parallel_activation_checkpointing() {
        let tmp = TempDir::new("loader-gemma4-tp-remat-capability");
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        copy_dir_contents(&fixture_root.join("tiny_gemma4"), tmp.path());
        std::fs::copy(
            fixture_root.join("tiny_tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();

        let (policy, _tok) = load_auto_policy(
            tmp.path(),
            &Device::Cpu,
            &LoaderOpts {
                lora_rank: 2,
                lora_alpha: 4.0,
                ..LoaderOpts::default()
            },
        )
        .unwrap();
        assert!(policy.supports_tensor_parallel());
        assert!(policy.model_family_supports_tensor_parallel_activation_checkpointing());
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
    fn auto_loader_reaches_gemma4_weight_capture_after_config_validation() {
        let tmp = TempDir::new("loader-gemma4-recognized");
        std::fs::write(tmp.path().join("config.json"), gemma4_config_json()).unwrap();
        match load_auto_policy(tmp.path(), &Device::Cpu, &LoaderOpts::default()) {
            Err(LoaderError::Io { path, source }) => {
                assert!(path.ends_with("model.safetensors"));
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            Err(other) => panic!("expected Gemma4 weight-capture error, got {other:?}"),
            Ok(_) => panic!("expected Gemma4 weight-capture error, got loaded policy"),
        }
    }

    #[test]
    fn auto_loader_reaches_gemma4_unified_weight_capture_after_config_validation() {
        let tmp = TempDir::new("loader-gemma4-unified-recognized");
        std::fs::write(tmp.path().join("config.json"), gemma4_unified_config_json()).unwrap();
        match load_auto_policy(tmp.path(), &Device::Cpu, &LoaderOpts::default()) {
            Err(LoaderError::Io { path, source }) => {
                assert!(path.ends_with("model.safetensors"));
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            Err(other) => panic!("expected Gemma4 weight-capture error, got {other:?}"),
            Ok(_) => panic!("expected Gemma4 weight-capture error, got loaded policy"),
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
        assert!(!policy.requires_rollout_tensor_snapshot());
        let AutoPolicy::Qwen3_5(policy) = policy else {
            panic!("expected Qwen3.5 auto-loader branch");
        };
        assert!(!Policy::trainable_vars(&*policy).is_empty());
        assert_eq!(
            Policy::lora_recipe(&*policy).as_deref(),
            Some("attn:qkvo|mlp:gud|gdn:-")
        );
    }

    #[test]
    fn bound_load_uses_captured_qwen35_bytes_when_path_is_replaced_after_identity() {
        let tmp = TempDir::new("loader-qwen35-bound-replacement");
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        copy_dir_contents(&fixture_root.join("tiny_qwen35"), tmp.path());
        std::fs::copy(
            fixture_root.join("tiny_tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();
        let opts = LoaderOpts {
            lora_rank: 4,
            lora_alpha: 8.0,
            activation_checkpointing: true,
            ..LoaderOpts::default()
        };
        let expected = checkpoint_policy_sha256(tmp.path(), &opts).unwrap();
        let replaced = tmp.path().join("model-00001-of-00002.safetensors");
        let (policy, _tok, bound) =
            load_auto_policy_bound_inner(tmp.path(), &Device::Cpu, &opts, || {
                std::fs::write(&replaced, b"replacement bytes that are not safetensors").unwrap();
            })
            .unwrap();
        assert_eq!(bound, expected);
        let AutoPolicy::Qwen3_5(policy) = policy else {
            panic!("expected Qwen3.5 auto-loader branch");
        };
        assert!(policy.model().activation_checkpointing());
        assert_ne!(
            checkpoint_policy_sha256(tmp.path(), &opts).unwrap(),
            expected,
            "the offline post-replacement directory fingerprint must see model B"
        );
    }

    #[test]
    fn tiny_gemma_loader_and_identity_share_the_text_shard_selector() {
        let tmp = TempDir::new("loader-gemma4-bound-selector");
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        copy_dir_contents(&fixture_root.join("tiny_gemma4"), tmp.path());
        std::fs::copy(
            fixture_root.join("tiny_tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();

        let model_bytes = std::fs::read(tmp.path().join("model.safetensors")).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&model_bytes).unwrap();
        let mut weight_map = tensors
            .names()
            .into_iter()
            .map(|name| (name.to_string(), "text.safetensors".to_string()))
            .collect::<BTreeMap<_, _>>();
        weight_map.insert(
            "model.vision_tower.unused".to_string(),
            "vision-only.safetensors".to_string(),
        );
        std::fs::write(tmp.path().join("text.safetensors"), &model_bytes).unwrap();
        std::fs::write(
            tmp.path().join("vision-only.safetensors"),
            b"deliberately not a safetensors payload",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
        let opts = LoaderOpts {
            lora_rank: 2,
            lora_alpha: 4.0,
            ..LoaderOpts::default()
        };
        let (policy, _tok, expected) =
            load_gemma4_policy_bound(tmp.path(), &Device::Cpu, &opts).unwrap();
        assert!(!Policy::trainable_vars(&policy).is_empty());
        assert_eq!(
            checkpoint_policy_sha256(tmp.path(), &opts).unwrap(),
            expected
        );

        std::fs::write(
            tmp.path().join("vision-only.safetensors"),
            b"different ignored vision-only bytes",
        )
        .unwrap();
        let (_policy, _tok, after_vision_change) =
            load_gemma4_policy_bound(tmp.path(), &Device::Cpu, &opts).unwrap();
        assert_eq!(after_vision_change, expected);

        std::fs::write(
            tmp.path().join("text.safetensors"),
            [model_bytes.as_slice(), b"selected-text-change"].concat(),
        )
        .unwrap();
        assert_ne!(
            checkpoint_policy_sha256(tmp.path(), &opts).unwrap(),
            expected,
            "a selected text shard must remain identity-sensitive"
        );
    }

    #[test]
    fn tiny_gemma_streaming_identity_is_tp_rank_invariant_and_binds_remat() {
        let tmp = TempDir::new("loader-gemma4-streaming-identity");
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        copy_dir_contents(&fixture_root.join("tiny_gemma4"), tmp.path());
        std::fs::copy(
            fixture_root.join("tiny_tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();
        let rank_zero = LoaderOpts {
            lora_rank: 2,
            lora_alpha: 4.0,
            activation_checkpointing: true,
            tensor_parallel: TensorParallelPlan::new(0, 2).unwrap(),
            ..LoaderOpts::default()
        };
        let rank_one = LoaderOpts {
            tensor_parallel: TensorParallelPlan::new(1, 2).unwrap(),
            ..rank_zero.clone()
        };
        let (rank_zero_policy, _tok, rank_zero_digest) =
            load_gemma4_policy_bound(tmp.path(), &Device::Cpu, &rank_zero).unwrap();
        let (rank_one_policy, _tok, rank_one_digest) =
            load_gemma4_policy_bound(tmp.path(), &Device::Cpu, &rank_one).unwrap();
        assert!(rank_zero_policy.model().activation_checkpointing());
        assert!(rank_one_policy.model().activation_checkpointing());
        assert_eq!(rank_zero_digest, rank_one_digest);
        assert_eq!(
            checkpoint_policy_sha256(tmp.path(), &rank_one).unwrap(),
            rank_zero_digest,
            "offline and retained-handle identities must share one canonical selector"
        );

        let without_remat = LoaderOpts {
            activation_checkpointing: false,
            ..rank_zero
        };
        assert_ne!(
            checkpoint_policy_sha256(tmp.path(), &without_remat).unwrap(),
            rank_zero_digest
        );
    }

    #[cfg(unix)]
    #[test]
    fn tiny_gemma_streaming_bound_load_rejects_in_place_mutation_after_identity() {
        use std::io::{Seek as _, Write as _};
        use std::os::unix::fs::MetadataExt;

        let tmp = TempDir::new("loader-gemma4-streaming-in-place-mutation");
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        copy_dir_contents(&fixture_root.join("tiny_gemma4"), tmp.path());
        std::fs::copy(
            fixture_root.join("tiny_tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();
        let opts = LoaderOpts {
            lora_rank: 2,
            lora_alpha: 4.0,
            activation_checkpointing: true,
            tensor_parallel: TensorParallelPlan::new(0, 2).unwrap(),
            ..LoaderOpts::default()
        };
        let weights = tmp.path().join("model.safetensors");
        let before = std::fs::metadata(&weights).unwrap();
        let mut mutated = std::fs::read(&weights).unwrap();
        *mutated.last_mut().unwrap() ^= 1;
        let result = load_auto_policy_bound_inner(tmp.path(), &Device::Cpu, &opts, || {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&weights)
                .unwrap();
            file.seek(std::io::SeekFrom::Start(0)).unwrap();
            file.write_all(&mutated).unwrap();
            file.flush().unwrap();
        });
        let error = match result {
            Err(LoaderError::Model(error)) => error.to_string(),
            Err(other) => panic!("expected authenticated model-load failure, got {other:?}"),
            Ok(_) => panic!("streaming Gemma load accepted post-identity source mutation"),
        };
        assert!(
            error.contains("changed after its bound identity was captured"),
            "unexpected streaming authentication error: {error}"
        );
        let after = std::fs::metadata(&weights).unwrap();
        assert_eq!(after.dev(), before.dev());
        assert_eq!(after.ino(), before.ino());
        assert_eq!(after.len(), before.len());
        assert_eq!(std::fs::read(&weights).unwrap(), mutated);
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
