//! A [`Policy`] over any [`GradModel`].
//!
//! [`LmPolicy`] bridges a grad-bearing model forward (the update path) to the
//! trainer's [`Policy`] seam, generically over the [`GradModel`] /
//! [`CachedDecoder`] traits — so the *same*
//! [`Trainer`] that drives the P2 echo toy drives any real model unchanged.
//! [`QwenPolicy`] (= `LmPolicy<QwenGradModel>`) is the production instantiation
//! over Qwen3-0.6B-Base.
//!
//! ## Generation is KV-cached over merged weights, and adapter-aware
//!
//! [`generate`](LmPolicy::generate) snapshots the policy's **current** effective
//! weights into a cached decoder ([`GradModel::merged_decoder`]) — the `LoRA`
//! adapter folded into the base (`W + scale·BA` when enabled, plain `W` when
//! disabled, so the eval adapter-off path samples the frozen base) — and decodes
//! incrementally over a KV cache: **O(L) per token** instead of the uncached
//! forward's O(L²). The rollout is still drawn from the *current* policy at every
//! step (candle's shipped cached forwards carry no adapter, so they could only
//! sample the frozen base — the merge is what makes a cached **adapter-aware**
//! rollout possible). The merged decoder is a tape-detached value snapshot,
//! rebuilt every `generate` call, so it always reflects the latest optimizer
//! step. **Scoring is unaffected** — the grad-bearing
//! [`token_logprobs`](LmPolicy::token_logprobs) and the KL reference forward
//! still run the uncached [`GradModel::forward`] (the cache holds no tape).
//! The cached and uncached rollouts are equivalent up to F32 reassociation of the
//! merge (CI-gated: identical token stream **and** identical sampler-RNG
//! consumption on a tiny model); the bf16-merge faithfulness is a manual GPU gate.
//!
//! ## The group is decoded as one batch, with per-row RNG substreams
//!
//! [`generate`](LmPolicy::generate) decodes all `group_size` sequences in a
//! single **batched** pass over the shared merged decoder (prefill `[group,
//! prompt_len]`, then one `[group, 1]` step per token) rather than one sequence
//! at a time — rollout is 80–90 % of RL wall-clock, and the old per-sequence
//! loop left a ~`group_size`× throughput factor unbatched. To keep the sampled
//! stream invariant to that batching, each row draws from its **own** RNG
//! substream forked from the policy sampler by global row index
//! ([`GrpoSampler::fork_substreams_at`](crate::sampler::GrpoSampler::fork_substreams_at)):
//! a row's tokens depend only on its substream, never on whether the group is
//! decoded together or one row at a time. That invariance is what the CI
//! batch-invariance gates check, in two regimes. For the **dense-attention**
//! models (Qwen3, Llama, Gemma 4) the cached merged decode equals the uncached forward
//! bit-exactly on CPU, so the sequential, uncached `generate_uncached` oracle
//! (test-only) reproduces the batched path **token-for-token**
//! (`cached_generate_matches_uncached*`). The **Qwen3.5 `GatedDeltaNet`** family
//! decodes through a recurrent/conv cache whose cached path and the uncached
//! *chunked* forward agree only within a tolerance (an algorithm difference
//! independent of batching), so its batch-invariance is gated directly at the
//! decoder level — a batched decode equals the per-row decodes within that
//! tolerance, including with a retired row fed the EOS pad
//! (`qwen35::assert_batched_decode_matches_per_row` and the retirement test).
//!
//! A row that samples EOS retires — drawing no further tokens, so its captured
//! behavior log-probs stay one-per-real-token — but stays in the batch, fed the
//! EOS pad so every layer's cache advances in lockstep, until all rows retire or
//! the fixed width is reached.
//!
//! ## Rectangular rollouts
//!
//! [`generate`](LmPolicy::generate) always emits a **fixed** width of
//! `max_new_tokens` completion tokens per sequence, so every sequence in a group
//! has the same length — the rectangular shape the [`Trainer`] requires (it rejects
//! ragged rollouts, and a fixed width keeps Dr.GRPO's token denominator constant).
//! When [`GenConfig::eos_token_id`](crate::policy::GenConfig::eos_token_id) is set,
//! a sequence that samples the EOS token stops early (the EOS token is **kept** — the
//! length is EOS-*inclusive*) and is right-padded back to the fixed width with that
//! same EOS id; [`Rollout::completion_lens`](crate::policy::Rollout::completion_lens)
//! records each true length so the padding can be masked out of the loss downstream.
//! With `eos_token_id == None` no sequence stops early, every completion is the full
//! width, and the rollout is bit-identical to the legacy behavior. Scoring
//! ([`token_logprobs`](LmPolicy::token_logprobs)) is teacher-forced: forward all
//! but the last token, read the positions that predict the completion tokens, and
//! gather their log-probabilities — divided by the policy's rollout temperature
//! first (temperature-consistent scoring, TRL parity; a guarded no-op at the
//! `1.0` default).
//!
//! ## Behavior log-probs and the off-policy gap
//!
//! [`generate`](LmPolicy::generate) also records each drawn token's log-prob
//! under the distribution it was sampled from
//! ([`Rollout::rollout_logprobs`](crate::policy::Rollout::rollout_logprobs) —
//! the sampler computes the full distribution anyway, so the capture is free).
//! Rollout draws from the **merged cached** decoder while training scores with
//! the **uncached grad** forward; on an all-F32 model the two differ only by
//! float reassociation of the merge, but a bf16 base makes the rollout
//! genuinely off-policy relative to the f32-scored objective — exactly the
//! mismatch the trainer's rollout-ratio telemetry (and optional TIS
//! correction) measures from these captured log-probs.
//!
//! [`Trainer`]: crate::trainer::Trainer

use candle_core::{IndexOp, Result as CandleResult, Tensor, Var};

use crate::comm::Comm;
use crate::model::{chunked_logprobs_from_logits, CachedDecoder, GradModel};
use crate::policy::{GenConfig, Policy, Rollout, TensorParallelPolicy};
use crate::qwen::QwenGradModel;
use crate::sampler::GrpoSampler;
use crate::telemetry::ModelTelemetryRecorder;

/// Scoring positions processed per chunk by the log-softmax/gather stage —
/// bounds the F32 upcast + softmax buffers to `[group, SCORING_CHUNK, vocab]`
/// at a time (chunking over positions is exact: the softmax reduces over the
/// vocab axis only). 128 keeps the chunk small relative to any real
/// completion window while adding no measurable overhead on tiny CI shapes.
const SCORING_CHUNK: usize = 128;

/// A [`Policy`] backed by any grad-bearing [`GradModel`].
///
/// Construct it from a loaded model with [`LmPolicy::new`]; the device and dtype
/// follow the model's — all-F32, or a bf16-base / F32-adapter split (see
/// [`QwenGradModel::load_with_adapter_dtype`](crate::qwen::QwenGradModel::load_with_adapter_dtype)),
/// whose BF16 logits the scoring path upcasts to F32 for the surrogate.
pub struct LmPolicy<M: GradModel> {
    model: M,
    sampler: GrpoSampler,
    temperature: f64,
    enabled: bool,
}

/// The production policy over the real Qwen3 model — the first [`LmPolicy`]
/// instantiation (and the name every pre-M1 call site uses).
pub type QwenPolicy = LmPolicy<QwenGradModel>;

/// The policy over a dense Llama-3.x model — the second [`LmPolicy`]
/// instantiation, and the witness that the [`GradModel`] seam is real: the same
/// generic policy (and through it the same `Trainer`) drives
/// [`LlamaGradModel`](crate::llama::LlamaGradModel) with zero policy changes.
pub type LlamaPolicy = LmPolicy<crate::llama::LlamaGradModel>;

/// The policy over the hybrid `qwen3_5` (Qwen3.5 / Qwen3.6) model — the third
/// [`LmPolicy`] instantiation, and the first whose decoder state is not purely
/// KV-shaped (conv + delta-rule recurrent state on the linear-attention
/// layers); the generic policy drives it through the same
/// [`CachedDecoder`] contract with zero changes.
pub type Qwen3_5Policy = LmPolicy<crate::qwen35::Qwen3_5GradModel>;

/// The policy over the dense Gemma 4 text model.
pub type Gemma4Policy = LmPolicy<crate::gemma4::Gemma4GradModel>;

// Elide the sampler's RNG state and the heavy model fields; show the inspectable
// scalars. (`GrpoSampler` is `Debug`, but the raw RNG words add only noise.)
impl<M: GradModel + std::fmt::Debug> std::fmt::Debug for LmPolicy<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmPolicy")
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

impl<M: GradModel> LmPolicy<M> {
    /// Wrap a loaded [`GradModel`] as a policy, seeding the rollout sampler.
    ///
    /// `temperature` is the rollout sampling temperature, fixed for this policy's
    /// lifetime: the [`GrpoSampler`] bakes it in (as candle's `LogitsProcessor`
    /// does). [`generate`](Self::generate) **fails loud** if handed a
    /// [`GenConfig`] whose `temperature` differs (rather than silently sampling
    /// at the wrong temperature); the trainer passes this same value through.
    /// The one exception is an explicit eval-only override
    /// ([`GenConfig::eval_sampling`](crate::policy::GenConfig::eval_sampling)),
    /// which deliberately samples the eval distribution instead. **Scoring is
    /// temperature-consistent** ([`token_logprobs`](Self::token_logprobs) divides
    /// the logits by this same temperature — TRL parity — so the distribution
    /// GRPO optimizes is the one the rollout sampled from; at the `1.0` default
    /// this is bit-identical to unscaled scoring). The adapter starts enabled
    /// (the trainer toggles it off for the KL reference forward).
    #[must_use]
    pub fn new(model: M, seed: u64, temperature: f64) -> Self {
        let sampler = GrpoSampler::new(seed, temperature);
        Self {
            model,
            sampler,
            temperature,
            enabled: true,
        }
    }

    /// The wrapped grad-bearing model — e.g. to inspect its device or (later) save
    /// the trained adapter.
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Mutable access to the wrapped model — e.g. to turn on **activation
    /// checkpointing** after construction
    /// (`policy.model_mut().set_activation_checkpointing(true)` on the models
    /// that support it; see
    /// [`QwenGradModel::set_activation_checkpointing`](crate::qwen::QwenGradModel::set_activation_checkpointing)).
    #[must_use]
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Score a rollout through the model's explicit tensor-parallel forward
    /// path.
    ///
    /// This is deliberately an inherent helper rather than part of
    /// [`Policy`]: callers must supply the communicator at the scoring site,
    /// while the ordinary trainer/loader/CLI paths continue to reject sharded
    /// TP; the explicit trainer TP entry points also reject sharded
    /// communicators until gradient/optimizer/checkpoint semantics are wired
    /// end to end.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the communicator/model plan is unsupported,
    /// the rollout scoring window is malformed, or any tensor op fails.
    pub fn token_logprobs_tensor_parallel(
        &self,
        rollout: &Rollout,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let input = self.scoring_input(rollout)?;
        let (start, len) = Self::scoring_window(rollout);
        let targets = self.scoring_targets(rollout)?;
        let logits = self
            .model
            .forward_tensor_parallel_narrowed(&input, start, len, comm)
            .map_err(crate::cuda_compat::translate_ptx_error)?;
        chunked_logprobs_from_logits(&logits, &targets, self.temperature, SCORING_CHUNK)
            .map_err(crate::cuda_compat::translate_ptx_error)
    }

    /// Detached/value-only variant of
    /// [`token_logprobs_tensor_parallel`](Self::token_logprobs_tensor_parallel).
    ///
    /// # Errors
    ///
    /// As [`token_logprobs_tensor_parallel`](Self::token_logprobs_tensor_parallel).
    pub fn token_logprobs_tensor_parallel_detached(
        &self,
        rollout: &Rollout,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let input = self.scoring_input(rollout)?;
        let (start, len) = Self::scoring_window(rollout);
        let targets = self.scoring_targets(rollout)?;
        let logits = self
            .model
            .forward_tensor_parallel_detached_narrowed(&input, start, len, comm)
            .map_err(crate::cuda_compat::translate_ptx_error)?;
        Ok(
            chunked_logprobs_from_logits(&logits, &targets, self.temperature, SCORING_CHUNK)
                .map_err(crate::cuda_compat::translate_ptx_error)?
                .detach(),
        )
    }

    /// Generate a rollout through the model's explicit tensor-parallel cached
    /// decoder path.
    ///
    /// This is deliberately an inherent helper rather than part of
    /// [`Policy`]: callers must supply the communicator at the generation site,
    /// while public trainer, loader, and CLI entry points continue to reject
    /// sharded TP until the end-to-end execution contract is opened.
    ///
    /// # Errors
    ///
    /// Returns a candle error if communicator/model validation, cached decode,
    /// sampling, or rollout construction fails.
    pub fn generate_tensor_parallel(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        comm: &dyn Comm,
    ) -> CandleResult<Rollout> {
        self.generate_tensor_parallel_at(prompt, cfg, 0, comm)
    }

    /// As [`generate_tensor_parallel`](Self::generate_tensor_parallel), but
    /// forks per-row sampler streams from an explicit global row base.
    ///
    /// # Errors
    ///
    /// As [`generate_tensor_parallel`](Self::generate_tensor_parallel).
    pub fn generate_tensor_parallel_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        comm: &dyn Comm,
    ) -> CandleResult<Rollout> {
        self.generate_tensor_parallel_at_instrumented(prompt, cfg, global_row_base, comm, None)
    }

    /// As [`generate_tensor_parallel_at`](Self::generate_tensor_parallel_at),
    /// with an optional telemetry sink for model-path phase boundaries and
    /// decoder-cache snapshots.
    ///
    /// # Errors
    ///
    /// As [`generate_tensor_parallel`](Self::generate_tensor_parallel).
    pub fn generate_tensor_parallel_at_instrumented(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        comm: &dyn Comm,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout> {
        self.generate_at_instrumented_inner(prompt, cfg, global_row_base, Some(comm), telemetry)
    }

    /// The teacher-forcing scoring input: all but the last token of every
    /// sequence, as one `[group, seq_len - 1]` tensor on the model's device.
    ///
    /// Precondition (the `Trainer` guarantees this via `completion_dims`): a
    /// rectangular rollout with `prompt_len >= 1` and `comp_len >= 1`. Called
    /// directly with `prompt_len == 0`, the scoring window underflows (a
    /// debug-profile panic at the subtraction in
    /// [`scoring_window`](Self::scoring_window); in release, a loud narrow
    /// error in the model tail).
    fn scoring_input(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let seq_len = rollout.token_ids[0].len();
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let mut input_data = Vec::with_capacity(g * input_len);
        for ids in &rollout.token_ids {
            input_data.extend_from_slice(&ids[..input_len]);
        }
        Tensor::from_vec(input_data, (g, input_len), self.model.device())
    }

    /// The completion-predicting scoring window of a rectangular rollout:
    /// `(start, len)` along the scoring input's sequence axis — positions
    /// `[prompt_len - 1 .. prompt_len - 1 + comp_len]` are the ones whose
    /// next-token distributions score the completion tokens.
    fn scoring_window(rollout: &Rollout) -> (usize, usize) {
        let seq_len = rollout.token_ids[0].len();
        (rollout.prompt_len - 1, seq_len - rollout.prompt_len)
    }

    /// Gather the completion tokens' log-probabilities out of the **already
    /// narrowed** window logits `pred` (`[g, comp_len, vocab]`, the
    /// [`scoring_window`](Self::scoring_window) positions — guarded, fail-loud):
    /// upcast to F32, divide by the rollout temperature
    /// (temperature-consistent scoring; a guarded no-op at the `1.0` default),
    /// `log_softmax`, gather. Both production call sites pass
    /// [`SCORING_CHUNK`]; tests pass tiny chunks to force multi-chunk runs on
    /// small rollouts.
    ///
    /// Chunking bounds the F32 expansion: the upcast + `log_softmax` buffers
    /// exist for one `[g, chunk, vocab]` slice at a time instead of the whole
    /// window — on the detached scorings each chunk's intermediates are freed
    /// as the loop advances. `log_softmax` reduces over the vocab axis only
    /// and `gather` is positionwise, so chunking over positions is exact: the
    /// concatenated result is identical to the unchunked one.
    #[cfg(test)]
    fn completion_logprobs_chunked(
        &self,
        rollout: &Rollout,
        pred: &Tensor,
        chunk: usize,
    ) -> CandleResult<Tensor> {
        let prompt_len = rollout.prompt_len;
        let seq_len = rollout.token_ids[0].len();
        let comp_len = seq_len - prompt_len;
        let g = rollout.token_ids.len();
        // Fail-loud preconditions: the type system cannot enforce that `pred`
        // is the narrowed window (full-width logits would gather positions
        // 0..comp_len — silently wrong values), and a zero-length window would
        // otherwise surface as an obscure empty-cat error.
        if comp_len == 0 {
            candle_core::bail!("completion_logprobs: zero-length completion window");
        }
        let (pg, pw, _vocab) = pred.dims3()?;
        if pg != g || pw != comp_len {
            candle_core::bail!(
                "completion_logprobs: pred must be the narrowed scoring window \
                 [{g}, {comp_len}, vocab], got {:?}",
                pred.dims()
            );
        }

        let targets = self.scoring_targets(rollout)?;
        chunked_logprobs_from_logits(pred, &targets, self.temperature, chunk)
    }

    fn scoring_targets(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let prompt_len = rollout.prompt_len;
        let seq_len = rollout.token_ids[0].len();
        let comp_len = seq_len - prompt_len;
        let g = rollout.token_ids.len();
        let mut tgt_data = Vec::with_capacity(g * comp_len);
        for ids in &rollout.token_ids {
            tgt_data.extend_from_slice(&ids[prompt_len..seq_len]);
        }
        Tensor::from_vec(tgt_data, (g, comp_len), self.model.device())
    }

    /// The **sequential, uncached** rollout oracle: fork the SAME per-row
    /// substreams [`generate`](Self::generate) forks (at global row base 0), then
    /// decode each row independently at
    /// batch 1 by re-running the full-sequence [`GradModel::forward`] every step.
    /// Because each row draws only from its own substream, this sequential decode
    /// reproduces the batched, cached `generate` **bit-for-bit** — same token
    /// stream, same EOS/padding, same RNG consumption — so it is at once the
    /// KV-cache faithfulness oracle (cached == uncached) and the batch-invariance
    /// oracle (batched == sequential). That bit-exactness holds for the
    /// **dense-attention** models this is instantiated on (Qwen3, Llama, Gemma 4),
    /// where the cached merged decode equals the uncached forward exactly on CPU; the Qwen3.5
    /// `GatedDeltaNet` family is batch-invariance-gated at the decoder level instead
    /// (see [`crate::qwen35`]), since its cached recurrent decode and uncached
    /// chunked forward differ by an algorithm-level tolerance unrelated to batching.
    /// Test-only; the production path is `generate`.
    #[cfg(test)]
    fn generate_uncached(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        // Resolve the sampling parameters exactly as `generate` does, so the
        // oracle also covers the eval-override decode path (a cached-vs-uncached
        // gate with `eval_sampling: Some(..)` compares the same distribution).
        let (temperature, top_p) = self.resolve_sampling(cfg)?;
        let device = self.model.device().clone();
        let prompt_len = prompt.len();
        let width = prompt_len + cfg.max_new_tokens;
        // Fork the per-row substreams exactly as `generate` does at base 0 (each row
        // seeded by its global index); row r is driven solely by its own substream, so
        // decoding sequentially here vs batched in `generate` cannot change the sampled
        // stream.
        let mut substreams = self.sampler.fork_substreams_at(0, cfg.group_size);
        let mut token_ids = Vec::with_capacity(cfg.group_size);
        let mut completion_lens = Vec::with_capacity(cfg.group_size);
        let mut rollout_logprobs = Vec::with_capacity(cfg.group_size);
        for sub in &mut substreams {
            let mut ids = prompt.to_vec();
            let mut logprobs = Vec::with_capacity(cfg.max_new_tokens);
            let mut comp_len = cfg.max_new_tokens;
            for step in 0..cfg.max_new_tokens {
                let len = ids.len();
                let input = Tensor::from_vec(ids.clone(), (1, len), &device)?;
                let logits = self.model.forward(&input)?;
                let last = logits.i((0, len - 1))?;
                let (next, logprob) = sub.sample_with(&last, temperature, top_p)?;
                ids.push(next);
                logprobs.push(logprob);
                if cfg.eos_token_id == Some(next) {
                    comp_len = step + 1;
                    ids.resize(width, next);
                    break;
                }
            }
            token_ids.push(ids);
            completion_lens.push(comp_len);
            rollout_logprobs.push(logprobs);
        }
        Ok(Rollout {
            token_ids,
            prompt_len,
            completion_lens,
            rollout_logprobs: Some(rollout_logprobs),
        })
    }
}

impl<M: GradModel> LmPolicy<M> {
    /// Resolve one `generate` call's sampling parameters. The training path
    /// (no override) keeps the fail-loud temperature check: the sampler's
    /// temperature is fixed at construction (see [`new`](Self::new)) and
    /// scoring is temperature-consistent with it, so a disagreeing
    /// `cfg.temperature` is a drifted config, not a request. An explicit eval
    /// override (`cfg.eval_sampling`) deliberately samples a DIFFERENT
    /// distribution — eval-only temperature / nucleus top-p — and skips the
    /// check (`cfg.temperature` is documented as ignored then).
    fn resolve_sampling(&self, cfg: &GenConfig) -> CandleResult<(f64, Option<f64>)> {
        match cfg.eval_sampling {
            Some(eval) => {
                if !eval.temperature.is_finite() || eval.temperature <= 0.0 {
                    candle_core::bail!(
                        "eval_sampling.temperature must be finite and > 0, got {}",
                        eval.temperature
                    );
                }
                // Validate top_p HERE, not first at draw time inside the sampler:
                // by then the O(params) merged-weight build and the prompt prefill
                // have already been paid for a config that was never valid.
                if let Some(p) = eval.top_p {
                    if !p.is_finite() || p <= 0.0 || p > 1.0 {
                        candle_core::bail!("eval_sampling.top_p must be in (0, 1], got {p}");
                    }
                }
                Ok((eval.temperature, eval.top_p))
            }
            None => {
                if (cfg.temperature - self.temperature).abs() > f64::EPSILON {
                    candle_core::bail!(
                        "LmPolicy was built with temperature {} but generate was called \
                         with cfg.temperature {}; rebuild the policy to change it",
                        self.temperature,
                        cfg.temperature
                    );
                }
                Ok((self.temperature, None))
            }
        }
    }

    fn generate_at_instrumented_inner(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        tp_comm: Option<&dyn Comm>,
        mut telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout> {
        let (temperature, top_p) = self.resolve_sampling(cfg)?;
        let device = self.model.device().clone();
        let prompt_len = prompt.len();
        // One KV-cached decoder snapshots the CURRENT merged weights (adapter folded
        // in, toggle respected); the batch dimension carries all group members at
        // once. The first GPU kernel JIT happens building the merged weights / in the
        // first forward, so translate a driver-too-old PTX mismatch
        // (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) into an actionable rebuild/upgrade
        // message — a no-op off the `cuda` build and on the success path.
        if let Some(recorder) = telemetry.as_deref_mut() {
            recorder.record_phase("merged_decoder_build_start");
        }
        let mut decoder = self
            .model
            .merged_decoder()
            .map_err(crate::cuda_compat::translate_ptx_error)?;
        if let Some(recorder) = telemetry.as_deref_mut() {
            recorder.record_phase("merged_decoder_build_end");
            recorder
                .record_decoder_cache(decoder.decoder_cache_snapshots("merged_decoder_build_end"));
        }
        // Per-row independent RNG substreams seeded by GLOBAL row index
        // (`global_row_base + row`): each row draws ONLY from its own substream, so
        // the sampled stream is invariant both to decode order (this batched decode
        // and the sequential `generate_uncached` oracle produce bit-identical groups)
        // AND to data-parallel shard layout (a world-W run reproduces the
        // single-process draws). See `GrpoSampler::fork_substreams_at`.
        let mut substreams = self
            .sampler
            .fork_substreams_at(global_row_base, cfg.group_size);
        let (token_ids, completion_lens, rollout_logprobs) = batched_group_decode(
            &mut decoder,
            &mut substreams,
            prompt,
            cfg,
            (temperature, top_p),
            (&device, tp_comm),
            &mut telemetry,
        )?;
        // Built directly (not via `Rollout::rectangular`) so `completion_lens` carries
        // the true per-sequence lengths; under `eos_token_id == None` every entry is
        // `max_new_tokens` and this equals the rectangular construction exactly.
        Ok(Rollout {
            token_ids,
            prompt_len,
            completion_lens,
            rollout_logprobs: Some(rollout_logprobs),
        })
    }
}

/// The raw rollout of [`batched_group_decode`]: per-row token ids (rectangular),
/// per-row EOS-inclusive completion lengths, and per-row behavior log-probs. A named
/// alias so the trait method and helper share one signature (and clippy's
/// `type_complexity` stays quiet).
type GroupDecode = (Vec<Vec<u32>>, Vec<usize>, Vec<Vec<f32>>);

fn cached_decoder_forward<D: CachedDecoder>(
    decoder: &mut D,
    input_ids: &Tensor,
    offset: usize,
    tp_comm: Option<&dyn Comm>,
) -> CandleResult<Tensor> {
    match tp_comm {
        Some(comm) => decoder.forward_tensor_parallel(input_ids, offset, comm),
        None => decoder.forward(input_ids, offset),
    }
}

/// Decode an entire GRPO group as one batch over the shared cached decoder.
///
/// Prefill the shared prompt for all `group_size` rows, then each step sample one
/// token per still-active row from that row's own substream and feed the batch
/// forward in lockstep. A row that samples EOS retires — drawing no further tokens,
/// so its captured behavior log-probs stay one-per-real-token — but stays in the
/// batch fed the EOS pad until every row retires or the fixed width is reached.
/// Returns the rectangular `token_ids`, the per-row EOS-inclusive `completion_lens`,
/// and the per-row behavior log-probs.
///
/// `allow(cognitive_complexity)`: the prefill + per-step + per-row decode is one
/// cohesive loop; splitting the per-row body out would thread ~ten pieces of per-row
/// state and trip `too_many_arguments` for no readability gain. Kept out of
/// [`LmPolicy::generate`] so that trait method stays thin.
#[allow(clippy::cognitive_complexity)]
fn batched_group_decode<D: CachedDecoder>(
    decoder: &mut D,
    substreams: &mut [GrpoSampler],
    prompt: &[u32],
    cfg: &GenConfig,
    (temperature, top_p): (f64, Option<f64>),
    (device, tp_comm): (&candle_core::Device, Option<&dyn Comm>),
    telemetry: &mut Option<&mut dyn ModelTelemetryRecorder>,
) -> CandleResult<GroupDecode> {
    let g = cfg.group_size;
    let prompt_len = prompt.len();
    // The fixed rectangular width every sequence is padded/grown to.
    let width = prompt_len + cfg.max_new_tokens;
    // EOS pad: a stopped row is right-padded back to `width` with the EOS it sampled
    // (== cfg.eos_token_id), and once retired a row is FED this same id each later
    // step so the batch stays rectangular while the cache advances. Never used when
    // eos is None (no row stops; every row is full width), so 0 is an inert fallback.
    let pad = cfg.eos_token_id.unwrap_or(0);

    // Prefill the shared prompt for all `g` rows at offset 0 (the prompt is identical
    // across the group, so one broadcast prefill serves every row); its last position
    // predicts each row's first token.
    let mut prompt_data = Vec::with_capacity(g * prompt_len);
    for _ in 0..g {
        prompt_data.extend_from_slice(prompt);
    }
    let prompt_input = Tensor::from_vec(prompt_data, (g, prompt_len), device)?;
    if let Some(recorder) = telemetry.as_mut() {
        recorder.record_phase("rollout_prefill_start");
    }
    let logits = cached_decoder_forward(decoder, &prompt_input, 0, tp_comm)
        .map_err(crate::cuda_compat::translate_ptx_error)?;
    if let Some(recorder) = telemetry.as_mut() {
        recorder.record_phase("rollout_prefill_end");
        recorder.record_decoder_cache(decoder.decoder_cache_snapshots("rollout_prefill_end"));
    }
    let mut last = logits.i((.., prompt_len - 1))?; // [g, vocab]

    let mut token_ids: Vec<Vec<u32>> = (0..g).map(|_| prompt.to_vec()).collect();
    // Behavior-policy log-probs, one per real draw: the sampler computes the full
    // sampling distribution anyway, so capturing the drawn token's log-prob is free —
    // see `Rollout::rollout_logprobs`.
    let mut rollout_logprobs: Vec<Vec<f32>> = (0..g)
        .map(|_| Vec::with_capacity(cfg.max_new_tokens))
        .collect();
    // Real completion tokens per row, counting up to and INCLUDING the first EOS;
    // stays `max_new_tokens` unless an EOS early-stop overwrites it below.
    let mut completion_lens = vec![cfg.max_new_tokens; g];
    let mut active = vec![true; g];
    let mut offset = prompt_len;

    if let Some(recorder) = telemetry.as_mut() {
        recorder.record_phase("rollout_decode_start");
    }
    for step in 0..cfg.max_new_tokens {
        // Sample each still-active row from its OWN substream; a retired row draws
        // nothing (so its log-prob count stays == completion_lens[r], one per real
        // token) and is fed the EOS pad to keep the batch rectangular.
        let mut feed: Vec<u32> = Vec::with_capacity(g);
        for r in 0..g {
            if active[r] {
                let row_logits = last.i(r)?; // [vocab]
                let (next, logprob) = substreams[r].sample_with(&row_logits, temperature, top_p)?;
                token_ids[r].push(next);
                rollout_logprobs[r].push(logprob);
                // EOS-inclusive early stop: keep the EOS token, record the true
                // length, retire the row. With `eos_token_id == None` this never
                // fires, so every row runs the full `max_new_tokens`.
                if cfg.eos_token_id == Some(next) {
                    completion_lens[r] = step + 1;
                    active[r] = false;
                }
                feed.push(next);
            } else {
                feed.push(pad);
            }
        }
        // Every row retired (all sampled EOS): nothing left to decode.
        if active.iter().all(|&a| !a) {
            break;
        }
        // Advance the cache with the just-sampled tokens to get the next step's logits
        // — unless this was the final step (no further token to predict), which keeps
        // each substream's draw count exactly completion_lens[r].
        if step + 1 < cfg.max_new_tokens {
            let tok = Tensor::from_vec(feed, (g, 1), device)?;
            let logits = cached_decoder_forward(decoder, &tok, offset, tp_comm)
                .map_err(crate::cuda_compat::translate_ptx_error)?;
            last = logits.i((.., 0))?; // [g, vocab]
            offset += 1;
        }
    }
    if let Some(recorder) = telemetry.as_mut() {
        recorder.record_phase("rollout_decode_end");
        recorder.record_decoder_cache(decoder.decoder_cache_snapshots("rollout_decode_end"));
    }

    // Right-pad every retired row back to the fixed width with the EOS pad; full-width
    // rows (no early stop) are already `width` and unchanged.
    for row in &mut token_ids {
        row.resize(width, pad);
    }
    Ok((token_ids, completion_lens, rollout_logprobs))
}

impl<M: GradModel> Policy for LmPolicy<M> {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        // The `global_row_base = 0` convenience; the trainer drives the
        // world-invariant path through `generate_at`.
        self.generate_at(prompt, cfg, 0)
    }

    fn generate_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
    ) -> CandleResult<Rollout> {
        self.generate_at_instrumented(prompt, cfg, global_row_base, None)
    }

    fn generate_at_instrumented(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout> {
        self.generate_at_instrumented_inner(prompt, cfg, global_row_base, None, telemetry)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let input = self.scoring_input(rollout)?;
        let (start, len) = Self::scoring_window(rollout);
        // The NARROWED forward: the full-width `[g, input_len, vocab]` logits
        // never materialize — the model applies its head to the
        // completion-predicting window alone. Same CUDA-compat translation as
        // `generate` (see there): a no-op off the `cuda` build and on the
        // success path.
        let targets = self.scoring_targets(rollout)?;
        self.model
            .token_logprobs_narrowed(
                &input,
                &targets,
                start,
                len,
                self.temperature,
                SCORING_CHUNK,
            )
            .map_err(crate::cuda_compat::translate_ptx_error)
    }

    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let input = self.scoring_input(rollout)?;
        let (start, len) = Self::scoring_window(rollout);
        // The value-only scorings (logp_old / the KL reference) route through
        // the model's narrowed detached forward: a rolling boundary detach
        // frees each layer's intermediates as the walk proceeds, the head is
        // applied to the scoring window alone, and no checkpoint tape is
        // captured (so the tape of the NEXT update forward — the one
        // `backward` consumes — can never be clobbered by a value scoring).
        let targets = self.scoring_targets(rollout)?;
        self.model
            .token_logprobs_detached_narrowed(
                &input,
                &targets,
                start,
                len,
                self.temperature,
                SCORING_CHUNK,
            )
            .map_err(crate::cuda_compat::translate_ptx_error)
    }

    fn backward(&self, loss: &Tensor) -> CandleResult<candle_core::backprop::GradStore> {
        // Under activation checkpointing the model stitches the full gradient
        // from its boundary tape; otherwise this is exactly `loss.backward()`.
        self.model.backward(loss)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        // A model without adapters (full fine-tuning) cannot be toggled: the
        // flag stays true, so callers that need the toggle (eval's
        // base-vs-trained comparison) observe it did not take and fail loud
        // instead of silently comparing the policy against itself.
        if !self.model.has_adapters() {
            self.model.set_adapter_enabled(true);
            self.enabled = true;
            return;
        }
        self.model.set_adapter_enabled(enabled);
        self.enabled = enabled;
    }

    fn adapter_enabled(&self) -> bool {
        self.enabled
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.model.trainable_vars()
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.sampler.to_state_bytes()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        let restored = GrpoSampler::from_state_bytes(state)?;
        // The blob bakes the temperature it was checkpointed at. This policy
        // scores (and samples) at ITS OWN temperature, so a mismatched blob is a
        // cross-run restore the trait contract promises to fail loud on — the
        // restored RNG would otherwise continue a token stream the scorer
        // doesn't score (pre-R2 the blob's temperature silently won; post-R2 the
        // policy's silently would — neither is acceptable, so reject).
        if (restored.temperature() - self.temperature).abs() > f64::EPSILON {
            candle_core::bail!(
                "sampler state was checkpointed at temperature {} but this policy runs at {}; \
                 rebuild the policy with the checkpoint's temperature to resume it",
                restored.temperature(),
                self.temperature
            );
        }
        self.sampler = restored;
        Ok(())
    }

    fn lora_recipe(&self) -> Option<String> {
        self.model.lora_recipe()
    }
}

impl<M: GradModel> TensorParallelPolicy for LmPolicy<M> {
    fn generate_at_tensor_parallel_instrumented(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        comm: &dyn Comm,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout> {
        self.generate_tensor_parallel_at_instrumented(prompt, cfg, global_row_base, comm, telemetry)
    }

    fn token_logprobs_tensor_parallel(
        &self,
        rollout: &Rollout,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        LmPolicy::token_logprobs_tensor_parallel(self, rollout, comm)
    }

    fn token_logprobs_tensor_parallel_detached(
        &self,
        rollout: &Rollout,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        LmPolicy::token_logprobs_tensor_parallel_detached(self, rollout, comm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::LocalComm;
    use crate::gemma4::{Gemma4Config, Gemma4GradModel, Gemma4LayerType, CKPT_PREFIX};
    use crate::lora::DenseLoraTargets;
    use crate::nn::grad_coverage;
    use crate::telemetry::DecoderCacheSnapshot;
    use candle_core::backprop::GradStore;
    use candle_core::{DType, Device, D};
    use candle_nn::ops::log_softmax;
    use candle_nn::{Activation, VarBuilder};
    use candle_transformers::models::qwen3::Config;
    use std::collections::HashMap;

    /// A tiny Qwen3 config (2 layers, 2 Q / 1 KV head, `head_dim` 4) — the same
    /// scaffold qwen.rs's tests use, at a runnable scale on CPU.
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

    /// Random weights matching `cfg`'s dotted tensor names (tied head → no
    /// `lm_head.weight`).
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

    fn tiny_policy() -> QwenPolicy {
        tiny_policy_at(1.0)
    }

    /// A tiny policy at an explicit rollout temperature (the temperature-consistent
    /// scoring tests need a non-1.0 one).
    fn tiny_policy_at(temperature: f64) -> QwenPolicy {
        let cfg = tiny_cfg();
        let vb = VarBuilder::from_tensors(weight_map(&cfg), DType::F32, &Device::Cpu);
        let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        QwenPolicy::new(model, 7, temperature)
    }

    fn tiny_tp_gqa_cfg() -> Config {
        let mut cfg = tiny_cfg();
        cfg.hidden_size = 16;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 4;
        cfg.intermediate_size = 16;
        assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 2);
        cfg
    }

    fn tiny_tp_gqa_uneven_mlp_cfg() -> Config {
        let mut cfg = tiny_tp_gqa_cfg();
        cfg.intermediate_size = 15;
        cfg
    }

    fn tiny_tp_policy_from_weights(cfg: &Config, weights: HashMap<String, Tensor>) -> QwenPolicy {
        let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
        let model = QwenGradModel::load_with_targets(
            cfg,
            &vb,
            2,
            4.0,
            DType::F32,
            DenseLoraTargets::industrial(),
        )
        .unwrap();
        QwenPolicy::new(model, 7, 1.0)
    }

    fn arm_adapter_deterministic(model: &QwenGradModel) {
        for (i, v) in model.trainable_vars().iter().enumerate() {
            let dims = v.as_tensor().dims().to_vec();
            let n = dims.iter().product::<usize>();
            let data: Vec<f32> = (0..n)
                .map(|j| 0.03 + i as f32 * 0.004 + j as f32 * 0.002)
                .collect();
            v.set(&Tensor::from_vec(data, dims, &Device::Cpu).unwrap())
                .unwrap();
        }
    }

    fn tp_policy_rollout() -> Rollout {
        Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5, 6], vec![3, 1, 2, 6, 4, 5]], 3)
    }

    fn max_abs_diff_vec(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn nonuniform_logprob_loss(logp: &Tensor) -> Tensor {
        let n = logp.elem_count();
        let weights: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 + 1.0) * 0.125).collect();
        let weights = Tensor::from_vec(weights, logp.dims().to_vec(), &Device::Cpu).unwrap();
        logp.mul(&weights).unwrap().sum_all().unwrap()
    }

    fn assert_policy_tp_projection_grads_live(
        rank: usize,
        vars: &[Var],
        grads: &GradStore,
        cfg: &Config,
    ) {
        let names = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ];
        assert_eq!(vars.len(), cfg.num_hidden_layers * names.len() * 2);
        for layer in 0..cfg.num_hidden_layers {
            for (pair_idx, name) in names.iter().enumerate() {
                let a_idx = layer * names.len() * 2 + pair_idx * 2;
                let b_idx = a_idx + 1;
                let c = grad_coverage(&vars[a_idx..=b_idx], grads).unwrap();
                assert!(
                    c.is_covered() && c.nonzero == c.total && c.nonfinite == 0,
                    "rank {rank} layer {layer} {name} policy TP grads not fully live: {c:?}"
                );
            }
        }
    }

    #[test]
    fn tensor_parallel_policy_logprobs_match_unsharded_and_detached_is_tape_free() {
        let cfg = tiny_tp_gqa_cfg();
        let weights = weight_map(&cfg);
        let rollout = tp_policy_rollout();

        let reference_policy = tiny_tp_policy_from_weights(&cfg, weights.clone());
        arm_adapter_deterministic(reference_policy.model());
        let reference = reference_policy.token_logprobs(&rollout).unwrap();
        let reference_flat = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let comms = LocalComm::world(2);
        let outputs: Vec<(Vec<f32>, Vec<f32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let rollout = rollout.clone();
                    s.spawn(move || {
                        let policy = tiny_tp_policy_from_weights(&cfg, weights);
                        arm_adapter_deterministic(policy.model());
                        let live = policy
                            .token_logprobs_tensor_parallel(&rollout, &comm)
                            .unwrap();
                        let det = policy
                            .token_logprobs_tensor_parallel_detached(&rollout, &comm)
                            .unwrap();
                        assert_eq!(live.dims(), &[2, 3]);
                        assert_eq!(det.dims(), live.dims());
                        let det_store = det.sum_all().unwrap().backward().unwrap();
                        assert!(policy
                            .trainable_vars()
                            .iter()
                            .all(|v| det_store.get(v).is_none()));
                        let live_flat = live.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                        let det_flat = det.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                        let detached_worst = max_abs_diff_vec(&live_flat, &det_flat);
                        assert!(
                            detached_worst <= 1e-6,
                            "detached TP policy logprobs diverged from live TP values: \
                             {detached_worst}"
                        );
                        (live_flat, det_flat)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (rank, (live, det)) in outputs.iter().enumerate() {
            let live_worst = max_abs_diff_vec(live, &reference_flat);
            assert!(
                live_worst <= 1e-5,
                "rank {rank} TP policy logprobs diverged from unsharded scoring: {live_worst}"
            );
            let det_worst = max_abs_diff_vec(det, &reference_flat);
            assert!(
                det_worst <= 1e-5,
                "rank {rank} detached TP policy logprobs diverged from unsharded scoring: \
                 {det_worst}"
            );
        }
    }

    #[test]
    fn tensor_parallel_policy_backward_keeps_industrial_adapter_grads_live() {
        let cfg = tiny_tp_gqa_cfg();
        let weights = weight_map(&cfg);
        let rollout = tp_policy_rollout();

        let comms = LocalComm::world(2);
        std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .enumerate()
                .map(|(rank, comm)| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let rollout = rollout.clone();
                    s.spawn(move || {
                        let policy = tiny_tp_policy_from_weights(&cfg, weights);
                        arm_adapter_deterministic(policy.model());
                        let vars = policy.trainable_vars();
                        let logp = policy
                            .token_logprobs_tensor_parallel(&rollout, &comm)
                            .unwrap();
                        let loss = nonuniform_logprob_loss(&logp);
                        let grads = policy.backward(&loss).unwrap();
                        assert_policy_tp_projection_grads_live(rank, &vars, &grads, &cfg);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn tensor_parallel_cached_generate_matches_unsharded_generate() {
        let cfg = tiny_tp_gqa_cfg();
        let weights = weight_map(&cfg);
        let prompt = [1u32, 2, 3];
        let gen_cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 4,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let global_row_base = 17;

        let mut reference_policy = tiny_tp_policy_from_weights(&cfg, weights.clone());
        arm_adapter_deterministic(reference_policy.model());
        let reference = reference_policy
            .generate_at(&prompt, &gen_cfg, global_row_base)
            .unwrap();
        let reference_logprobs = reference.rollout_logprobs.clone().unwrap();

        let comms = LocalComm::world(2);
        let outputs: Vec<Rollout> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let prompt = prompt.to_vec();
                    let rank_gen_cfg = gen_cfg;
                    s.spawn(move || {
                        let mut policy = tiny_tp_policy_from_weights(&cfg, weights);
                        arm_adapter_deterministic(policy.model());
                        policy
                            .generate_tensor_parallel_at(
                                &prompt,
                                &rank_gen_cfg,
                                global_row_base,
                                &comm,
                            )
                            .unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (rank, got) in outputs.iter().enumerate() {
            assert_eq!(
                got.token_ids, reference.token_ids,
                "rank {rank} TP cached generation changed sampled tokens"
            );
            assert_eq!(
                got.completion_lens, reference.completion_lens,
                "rank {rank} TP cached generation changed completion lengths"
            );
            let got_logprobs = got.rollout_logprobs.as_ref().unwrap();
            assert_eq!(got_logprobs.len(), reference_logprobs.len());
            for (row, (got_row, want_row)) in
                got_logprobs.iter().zip(&reference_logprobs).enumerate()
            {
                let worst = max_abs_diff_vec(got_row, want_row);
                assert!(
                    worst <= 1e-5,
                    "rank {rank} row {row} TP cached generation logprobs diverged: {worst}"
                );
            }
        }
    }

    #[test]
    fn tensor_parallel_cached_decoder_preflights_layout_before_mutating_cache() {
        let cfg = tiny_tp_gqa_uneven_mlp_cfg();
        let weights = weight_map(&cfg);
        let input = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &Device::Cpu).unwrap();

        let reference_policy = tiny_tp_policy_from_weights(&cfg, weights.clone());
        arm_adapter_deterministic(reference_policy.model());
        let mut reference_decoder = reference_policy.model().merged_decoder().unwrap();
        let reference = reference_decoder.forward(&input, 0).unwrap();
        let reference_flat = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let comms = LocalComm::world(2);
        let outputs: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .into_iter()
                .map(|comm| {
                    let cfg = cfg.clone();
                    let weights = weights.clone();
                    let input = input.clone();
                    s.spawn(move || {
                        let policy = tiny_tp_policy_from_weights(&cfg, weights);
                        arm_adapter_deterministic(policy.model());
                        let mut decoder = policy.model().merged_decoder().unwrap();
                        let err = decoder
                            .forward_tensor_parallel(&input, 0, &comm)
                            .unwrap_err()
                            .to_string();
                        assert!(
                            err.contains("intermediate_size"),
                            "expected MLP intermediate preflight failure, got {err}"
                        );
                        let got = decoder.forward(&input, 0).unwrap();
                        got.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (rank, got) in outputs.iter().enumerate() {
            let worst = max_abs_diff_vec(got, &reference_flat);
            assert!(
                worst <= 1e-5,
                "rank {rank} decoder cache mutated before unsupported TP layout failed: {worst}"
            );
        }
    }

    /// Two policies sharing the SAME base weights and sampler seed, so they draw an
    /// identical token stream. `weight_map` is random and unseeded, so two
    /// independent `tiny_policy()` calls would NOT sample alike; cloning one map into
    /// both `VarBuilder`s makes them bit-identical. (The `LoRA` adapter is a no-op at
    /// its `B = 0` init, so only the shared base weights drive sampling — the
    /// per-policy random `A` factors never reach the logits.) This lets one policy
    /// observe a sampled token and the other stop on it *deterministically*, instead
    /// of relying on a cross-policy RNG coincidence.
    fn paired_policies() -> (QwenPolicy, QwenPolicy) {
        let cfg = tiny_cfg();
        let weights = weight_map(&cfg);
        let build = || {
            let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
            let model = QwenGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
            QwenPolicy::new(model, 7, 1.0)
        };
        (build(), build())
    }

    /// `generate_at` forks its substreams from the GLOBAL row base via `&self`,
    /// drawing nothing from the policy sampler, so a given global index's rollout
    /// is a pure function of `(run seed, global index)` — independent of what
    /// other rollouts (other prompts, other data-parallel ranks) run before it.
    /// That call-order independence on a shared-seed policy IS the world-size /
    /// shard-layout invariance: rank `r`'s prompt at global index `g` samples
    /// exactly what a single-process run would at `g` (the DP rollout-invariance
    /// property closed by global-index seeding).
    #[test]
    fn generate_at_rollout_is_invariant_to_call_order() {
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let prompt = [1u32, 2, 3];
        let base = 8u64 * cfg.group_size as u64; // global prompt index 8
        let first = policy.generate_at(&prompt, &cfg, base).unwrap().token_ids;
        // Interleave unrelated rollouts, as other ranks / other prompts would...
        let _ = policy.generate_at(&prompt, &cfg, 0).unwrap();
        let _ = policy
            .generate_at(&[2u32, 1], &cfg, 50 * cfg.group_size as u64)
            .unwrap();
        // ...the global-index-8 rollout is byte-for-byte unchanged.
        let again = policy.generate_at(&prompt, &cfg, base).unwrap().token_ids;
        assert_eq!(
            first, again,
            "generate_at(g) must not depend on intervening rollouts (shard-invariant)"
        );
    }

    #[test]
    fn generate_returns_rectangular_group() {
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 3,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let rollout = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(rollout.len(), 4);
        assert_eq!(rollout.prompt_len, 3);
        // Every sequence has the same length (rectangular): prompt + new tokens.
        for ids in &rollout.token_ids {
            assert_eq!(ids.len(), 3 + 3);
            assert!(ids.iter().all(|&i| i < tiny_cfg().vocab_size as u32));
        }
        // No EOS configured: every completion is the full width, no early stop.
        assert_eq!(rollout.completion_lens, vec![3; 4]);
    }

    /// Assert the EOS-aware rollout invariants for every sequence: each row is the
    /// fixed `prompt_len + max_new` width, `completion_lens[i]` is exactly the
    /// EOS-inclusive length (first-EOS index + 1, or the full width when no EOS was
    /// sampled), and everything at/after that length is EOS padding (an empty tail
    /// for a full-width row). The `position` (first occurrence) check folds the
    /// "EOS at the boundary, none before it" invariants into one comparison.
    fn assert_eos_rollout_invariants(r: &Rollout, eos: u32, max_new: usize) {
        let width = r.prompt_len + max_new;
        for (gi, ids) in r.token_ids.iter().enumerate() {
            assert_eq!(ids.len(), width, "seq {gi} not padded to the fixed width");
            let comp = &ids[r.prompt_len..];
            let expected = comp
                .iter()
                .position(|&t| t == eos)
                .map_or(max_new, |i| i + 1);
            let cl = r.completion_lens[gi];
            assert_eq!(
                cl, expected,
                "seq {gi} completion_len {cl} != EOS-inclusive {expected}"
            );
            assert!(
                comp[cl..].iter().all(|&t| t == eos),
                "seq {gi} pad tail is not EOS-filled"
            );
        }
    }

    #[test]
    fn generate_stops_at_eos_inclusive_and_right_pads_to_fixed_width() {
        // EOS-aware generation: a sampled EOS ends the completion (EOS kept →
        // inclusive length) and the row is right-padded back to the FIXED width, so
        // the group stays rectangular and `completion_lens` carries the true lengths.
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let width = prompt.len() + max_new;
        let (mut p_ref, mut p_test) = paired_policies();

        // Reference run, no EOS: full-width rectangular, lengths all == max_new.
        let cfg_none = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let r_none = p_ref.generate(&prompt, &cfg_none).unwrap();
        assert_eq!(r_none.completion_lens, vec![max_new; 4]);
        for ids in &r_none.token_ids {
            assert_eq!(ids.len(), width);
        }

        // p_test shares p_ref's weights + seed, so it draws the SAME first token for
        // seq 0; setting that token as the EOS makes seq 0 stop at step 0 → an
        // EOS-inclusive length of exactly 1 with the rest padded.
        let eos = r_none.token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..cfg_none
        };
        let r = p_test.generate(&prompt, &cfg_eos).unwrap();

        assert_eq!(r.len(), 4);
        assert_eq!(r.prompt_len, prompt.len());
        // seq 0 stops at its first sampled token (== eos): inclusive length 1.
        assert_eq!(
            r.completion_lens[0], 1,
            "seq 0 did not stop at the first EOS"
        );
        // Every sequence: fixed width, EOS-inclusive length, EOS-filled pad tail.
        assert_eos_rollout_invariants(&r, eos, max_new);
    }

    #[test]
    fn generate_with_configured_but_unsampled_eos_is_full_width() {
        // A configured EOS that is never sampled (here an out-of-vocab id) must leave
        // generation identical to the None path: full width, every completion_len ==
        // max_new. This pins the "configured-yet-inert" branch — distinct from None —
        // deterministically: an out-of-vocab id can never equal a sampled token, so no
        // RNG coincidence is required.
        let mut policy = tiny_policy();
        let max_new = 4usize;
        let unsampled = tiny_cfg().vocab_size as u32; // == 16, never a valid sampled id
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: Some(unsampled),
            eval_sampling: None,
        };
        let r = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(r.completion_lens, vec![max_new; 4]);
        for ids in &r.token_ids {
            assert_eq!(ids.len(), 3 + max_new);
            assert!(ids.iter().all(|&t| t < unsampled));
        }
    }

    #[test]
    fn generate_breaks_when_every_row_retires_before_max_new() {
        // The all-rows-retired early `break` in `batched_group_decode`: greedy
        // (argmax-nucleus) decoding is RNG-independent, so every group row samples
        // the IDENTICAL first token; configuring THAT token as the EOS retires every
        // row at step 0, firing the break before `max_new_tokens`. Pins that the
        // break leaves a correct rectangular rollout (all completion_lens == 1, every
        // row padded to the fixed width) and that the batched path still matches the
        // sequential oracle under a wholesale early stop.
        let mut policy = tiny_policy();
        let greedy = crate::policy::EvalSampling {
            temperature: 0.5,
            top_p: Some(1e-6),
        };
        let probe = GenConfig {
            group_size: 3,
            max_new_tokens: 4,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(greedy),
        };
        // The shared greedy first token every row draws at step 0.
        let first = policy.generate(&[1u32, 2, 3], &probe).unwrap().token_ids[0][3];
        let cfg = GenConfig {
            eos_token_id: Some(first),
            ..probe
        };
        // Batched == sequential under a wholesale early stop (token stream + RNG).
        assert_cached_matches_uncached(&mut policy, &[1u32, 2, 3], &cfg);
        let r = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(r.completion_lens, vec![1; 3], "every row retires at step 0");
        for ids in &r.token_ids {
            assert_eq!(
                ids.len(),
                3 + 4,
                "rectangular width preserved after the break"
            );
        }
    }

    #[test]
    fn generate_eos_at_the_max_new_tokens_one_boundary() {
        // max_new_tokens == 1 with an EOS sampled at the only step: comp_len == 1 ==
        // max_new (the resize is a no-op — no double-handling) and each row is exactly
        // prompt + 1 wide. Paired policies make the single draw deterministic.
        let prompt = [2u32, 5];
        let (mut p_ref, mut p_test) = paired_policies();
        let base = GenConfig {
            group_size: 3,
            max_new_tokens: 1,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = p_ref.generate(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..base
        };
        let r = p_test.generate(&prompt, &cfg_eos).unwrap();
        assert_eq!(r.completion_lens[0], 1);
        for ids in &r.token_ids {
            assert_eq!(ids.len(), prompt.len() + 1);
        }
        assert_eos_rollout_invariants(&r, eos, 1);
    }

    /// The P6-C cached-rollout equivalence gate: the cached [`generate`] must
    /// reproduce the uncached oracle's token stream bit-for-bit. Both paths fork
    /// per-row substreams from the same global indices (base 0), so the streams must
    /// match; and since global-index forking is a pure `&self` operation, neither
    /// path mutates the parent sampler — both leave it at the saved start state.
    /// Runs both paths from the *same* saved sampler state on one policy. Generic
    /// over the model — the Llama gates below reuse it verbatim.
    fn assert_cached_matches_uncached<M: GradModel>(
        policy: &mut LmPolicy<M>,
        prompt: &[u32],
        cfg: &GenConfig,
    ) {
        let start = policy.sampler_state().unwrap();
        let cached = policy.generate(prompt, cfg).unwrap();
        let after_cached = policy.sampler_state().unwrap();

        policy.restore_sampler_state(&start).unwrap();
        let uncached = policy.generate_uncached(prompt, cfg).unwrap();
        let after_uncached = policy.sampler_state().unwrap();

        assert_eq!(
            cached.token_ids, uncached.token_ids,
            "cached rollout token stream diverged from the uncached oracle"
        );
        assert_eq!(
            cached.completion_lens, uncached.completion_lens,
            "cached rollout completion_lens diverged from the uncached oracle"
        );
        assert_eq!(cached.prompt_len, uncached.prompt_len);
        // Under the global-index per-row-substream design, forking is a pure
        // function of the parent's fixed seed (`fork_substreams_at` takes `&self`),
        // so NEITHER path advances the policy sampler — both leave it byte-for-byte
        // at `start`. That zero-mutation is what makes resume re-derive identical
        // substreams from the run seed alone (no per-row RNG state to capture);
        // per-row draw-count parity is what the `token_ids` equality above proves.
        assert_eq!(
            after_cached, start,
            "cached generate must not mutate the policy sampler (global-index forking is pure)"
        );
        assert_eq!(
            after_uncached, start,
            "uncached generate must not mutate the policy sampler (global-index forking is pure)"
        );
        assert_rollout_logprobs_close(&cached, &uncached);
    }

    /// The captured behavior log-probs of two equivalent rollouts must agree —
    /// within a float tolerance, not bit-exactly: the merged (cached) and
    /// base+`LoRA` (uncached) forwards differ by F32 reassociation of the merge.
    fn assert_rollout_logprobs_close(cached: &Rollout, uncached: &Rollout) {
        let c_lp = cached.rollout_logprobs.as_ref().expect("cached capture");
        let u_lp = uncached
            .rollout_logprobs
            .as_ref()
            .expect("uncached capture");
        assert_eq!(c_lp.len(), u_lp.len());
        for (i, (c_row, u_row)) in c_lp.iter().zip(u_lp).enumerate() {
            assert_eq!(c_row.len(), u_row.len(), "seq {i} logprob count mismatch");
            for (j, (c, u)) in c_row.iter().zip(u_row).enumerate() {
                assert!(
                    (c - u).abs() <= 1e-4,
                    "seq {i} draw {j}: cached logprob {c} != uncached {u}"
                );
            }
        }
    }

    #[test]
    fn cached_generate_matches_uncached_adapter_on() {
        // Arm the adapter (B != 0) so the merge is non-trivial: the cached path must
        // reproduce the ADAPTER-AWARE uncached stream, not merely the base one.
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        assert!(policy.adapter_enabled());
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 6,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[1u32, 2, 3], &cfg);
    }

    #[test]
    fn cached_generate_matches_uncached_adapter_off() {
        // The eval path: adapter disabled => the snapshot is the pure base model.
        // Proves the toggle-respecting merge keeps eval's adapter-off rollout (and its
        // RNG consumption) identical to the uncached one.
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars()); // armed, but...
        policy.set_adapter_enabled(false); // ...disabled => base only
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[2u32, 4, 1], &cfg);
    }

    #[test]
    fn cached_generate_matches_uncached_with_eos() {
        // EOS early-stop + right-pad must be identical between paths, and the
        // sampler-RNG consumption must match — eval draws base then adapter from
        // successive RNG points, so a draw-count mismatch would desync them. A paired
        // probe picks a real first-token EOS deterministically; then compare cached vs
        // uncached on a fresh-sampler policy that draws that same first token.
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let (mut probe, mut policy) = paired_policies();
        let base = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = probe.generate_uncached(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..base
        };
        assert_cached_matches_uncached(&mut policy, &prompt, &cfg_eos);
    }

    /// THE R2 capture-alignment gate: every captured behavior log-prob must agree
    /// with the scoring path's log-prob of the same (sequence, draw) — at the
    /// policy temperature. Generation samples from `softmax(merged_logits / T)` and
    /// scoring computes `log_softmax(uncached_logits / T)` (temperature-consistent
    /// scoring), so on an all-F32 tiny model the two can differ only by float
    /// reassociation of the merge. A capture indexing bug (wrong token, shifted
    /// position) or a scoring-temperature bug shows up as a gross mismatch.
    /// Run at T = 1.0 (the bit-identical default) and a non-trivial T = 0.7.
    #[test]
    fn captured_behavior_logprobs_align_with_the_scoring_path() {
        for temperature in [1.0, 0.7] {
            let mut policy = tiny_policy_at(temperature);
            force_b_nonzero(&policy.trainable_vars()); // non-trivial merge
            let cfg = GenConfig {
                group_size: 3,
                max_new_tokens: 4,
                temperature,
                eos_token_id: None,
                eval_sampling: None,
            };
            let rollout = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
            let captured = rollout.rollout_logprobs.clone().expect("capture present");
            let scored = policy
                .token_logprobs(&rollout)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for (i, row) in captured.iter().enumerate() {
                assert_eq!(row.len(), 4, "full-width capture expected");
                for (j, &lp) in row.iter().enumerate() {
                    assert!(
                        (lp - scored[i][j]).abs() <= 1e-4,
                        "T={temperature} seq {i} draw {j}: behavior logprob {lp} != scored {}",
                        scored[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn captured_logprob_rows_match_the_true_completion_lens_under_eos() {
        // EOS early-stop: row i carries exactly completion_lens[i] log-probs (one
        // per real draw) — the EOS padding was never sampled, so it has none.
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let (mut p_ref, mut p_test) = paired_policies();
        let base = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = p_ref.generate(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let r = p_test
            .generate(
                &prompt,
                &GenConfig {
                    eos_token_id: Some(eos),
                    ..base
                },
            )
            .unwrap();
        let captured = r.rollout_logprobs.as_ref().expect("capture present");
        assert_eq!(captured.len(), r.len());
        for (row, &len) in captured.iter().zip(&r.completion_lens) {
            assert_eq!(row.len(), len, "one behavior logprob per real draw");
            assert!(row.iter().all(|lp| lp.is_finite() && *lp <= 0.0));
        }
        assert_eq!(r.completion_lens[0], 1, "seq 0 stops at the probed EOS");
    }

    /// THE EOS-path companion to
    /// [`captured_behavior_logprobs_align_with_the_scoring_path`]: under EOS
    /// early-stop, the per-token *value* alignment of capture vs scoring must still
    /// hold across each row's live prefix — INCLUDING the EOS draw itself (the last
    /// real token of a retired row). The no-EOS gate above proves value alignment
    /// only when every row is full width; the row-count gate
    /// ([`captured_logprob_rows_match_the_true_completion_lens_under_eos`]) checks
    /// only the *count* under EOS, never the values. But discovery runs *with* an EOS
    /// token, so a capture/scoring misalignment that bites only on the early-stop path
    /// — a shifted retired-row capture, or an EOS-draw log-prob gathered from the
    /// wrong position — would slip every existing gate. Here `scored` is the full
    /// rectangular completion width; the EOS-padded tail (`j >= completion_lens[i]`)
    /// is scored but never sampled, so only the real draws are compared. Run at
    /// T = 1.0 (bit-identical default) and a non-trivial T = 0.7.
    #[test]
    fn captured_behavior_logprobs_align_with_the_scoring_path_under_eos() {
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        for temperature in [1.0, 0.7] {
            let mut policy = tiny_policy_at(temperature);
            force_b_nonzero(&policy.trainable_vars()); // non-trivial merge
            let base = GenConfig {
                group_size: 4,
                max_new_tokens: max_new,
                temperature,
                eos_token_id: None,
                eval_sampling: None,
            };
            // Probe row 0's first token, then make THAT the EOS id. `generate` forks
            // each row's substream purely from the run seed (no sampler mutation), and
            // EOS only retires a row *after* its draw — it never changes the draw — so
            // row 0 redraws the same token at step 0 and deterministically retires at
            // length 1. At least one row therefore exercises the early-stop path.
            let eos = policy.generate(&prompt, &base).unwrap().token_ids[0][prompt.len()];
            let rollout = policy
                .generate(
                    &prompt,
                    &GenConfig {
                        eos_token_id: Some(eos),
                        ..base
                    },
                )
                .unwrap();
            assert!(
                rollout.completion_lens.iter().any(|&l| l < max_new),
                "T={temperature}: no row retired under EOS — the early-stop path went untested"
            );
            let captured = rollout.rollout_logprobs.clone().expect("capture present");
            let scored = policy
                .token_logprobs(&rollout)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for (i, (cap_row, &len)) in captured.iter().zip(&rollout.completion_lens).enumerate() {
                // Ragged capture: exactly `completion_lens[i]` real draws, EOS-inclusive.
                assert_eq!(
                    cap_row.len(),
                    len,
                    "T={temperature} seq {i}: one capture per real draw"
                );
                // Value alignment over the live prefix only (incl. the EOS draw at
                // j == len - 1); the padded tail j >= len was never sampled.
                for (j, &lp) in cap_row.iter().enumerate() {
                    assert!(
                        (lp - scored[i][j]).abs() <= 1e-4,
                        "T={temperature} seq {i} draw {j} (len {len}): behavior logprob {lp} \
                         != scored {} — capture/scoring misaligned on the EOS path",
                        scored[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn eval_sampling_override_bypasses_the_temperature_check() {
        // The override is the deliberate eval channel: it samples its own
        // temperature/top-p and skips the baked-temperature equality check
        // (cfg.temperature is documented as ignored). A mismatched
        // cfg.temperature that would bail on the training path must not bail
        // here.
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 4,
            temperature: 123.0, // would bail without the override
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.5,
                top_p: Some(0.9),
            }),
        };
        let before = policy.sampler_state().unwrap();
        let r = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r.completion_lens, vec![4; 3]);
        assert!(r.rollout_logprobs.is_some(), "override path still captures");
        // Global-index seeding forks substreams via `&self`, so generate — the
        // override path included — does NOT mutate the shared sampler; that purity
        // is the resume guarantee (the run seed alone re-derives every draw).
        assert_eq!(
            before,
            policy.sampler_state().unwrap(),
            "override sampling must not mutate the shared sampler (global-index forking is pure)"
        );

        // Without the override the same mismatched temperature fails loud.
        let train_cfg = GenConfig {
            eval_sampling: None,
            ..cfg
        };
        assert!(policy.generate(&[1u32, 2, 3], &train_cfg).is_err());
    }

    #[test]
    fn malformed_eval_override_fails_before_decoding() {
        // A malformed override temperature or top_p fails loud — BEFORE the
        // merged-decoder build (resolve_sampling validates both).
        let mut policy = tiny_policy();
        let base = GenConfig {
            group_size: 2,
            max_new_tokens: 2,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let bad = GenConfig {
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.0,
                top_p: None,
            }),
            ..base
        };
        assert!(policy.generate(&[1u32, 2, 3], &bad).is_err());
        let bad_p = GenConfig {
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.6,
                top_p: Some(1.5),
            }),
            ..base
        };
        assert!(policy.generate(&[1u32, 2, 3], &bad_p).is_err());
    }

    #[test]
    fn eval_override_values_actually_reach_the_sampler() {
        // The mutation-killer for the override plumbing: a resolve_sampling that
        // validates but then samples the TRAINING parameters passes the no-bail
        // test above — this one it cannot pass. With top_p so small that only
        // the argmax survives, every draw's nucleus is a single token, so every
        // captured behavior log-prob is EXACTLY ln(p/p) = 0.0 and every group
        // member decodes the identical greedy stream. Without the override
        // plumbed, the full-softmax probabilities over a 16-token vocab make
        // every log-prob strictly negative.
        let mut policy = tiny_policy();
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 4,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.5,
                top_p: Some(1e-6),
            }),
        };
        let r = policy.generate(&[1u32, 2, 3], &cfg).unwrap();
        let captured = r.rollout_logprobs.as_ref().expect("capture present");
        for (i, row) in captured.iter().enumerate() {
            for (j, &lp) in row.iter().enumerate() {
                assert_eq!(
                    lp, 0.0,
                    "seq {i} draw {j}: argmax-nucleus logprob must be exactly 0, got {lp} \
                     (override top_p did not reach the sampler?)"
                );
            }
        }
        assert!(
            r.token_ids.iter().all(|ids| ids == &r.token_ids[0]),
            "argmax-nucleus decoding must be greedy-deterministic across the group"
        );
    }

    #[test]
    fn cached_generate_matches_uncached_under_the_eval_override() {
        // The override decode path gets the same cached-vs-uncached equivalence
        // gate as the training path: same token stream, same RNG consumption,
        // logprobs within merge-reassociation tolerance (the uncached oracle
        // resolves the override exactly like generate does).
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.7,
                top_p: Some(0.9),
            }),
        };
        assert_cached_matches_uncached(&mut policy, &[2u32, 4, 1], &cfg);
    }

    #[test]
    fn restore_rejects_a_mismatched_sampler_temperature() {
        // The blob bakes the temperature it was checkpointed at; restoring it
        // into a policy scoring at a DIFFERENT temperature must fail loud (the
        // Policy trait's documented contract) instead of silently continuing a
        // token stream the scorer doesn't score.
        let mut policy = tiny_policy(); // T = 1.0
        let foreign = GrpoSampler::new(5, 0.7).to_state_bytes().unwrap();
        let err = policy.restore_sampler_state(&foreign).unwrap_err();
        assert!(
            err.to_string().contains("temperature"),
            "expected a temperature-mismatch error, got: {err}"
        );
        // A matching-temperature blob restores fine.
        let matching = GrpoSampler::new(5, 1.0).to_state_bytes().unwrap();
        policy.restore_sampler_state(&matching).unwrap();
    }

    #[test]
    fn token_logprobs_shape_and_finiteness() {
        let policy = tiny_policy();
        // Two sequences, prompt_len 2, completion_len 3 (rectangular).
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![1, 2, 6, 7, 8]], 2);
        let logp = policy.token_logprobs(&rollout).unwrap();
        assert_eq!(logp.dims(), &[2, 3]);
        // Log-probs are <= 0 and finite.
        let flat = logp.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|&x| x.is_finite() && x <= 1e-5));
    }

    #[test]
    fn token_logprobs_align_with_a_manual_per_position_reference() {
        // Shape + finiteness can't catch a teacher-forcing off-by-one (a wrong but
        // finite, correctly-shaped score makes GRPO optimize garbage). Pin the
        // alignment: each returned log-prob must equal the model's own
        // log_softmax(logits)[g, prompt_len-1+j, completion_token] recomputed
        // independently of the narrow/gather under test.
        let policy = tiny_policy();
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![3, 1, 4, 1, 5]], 2);
        let got = policy
            .token_logprobs(&rollout)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        let seq_len = rollout.token_ids[0].len();
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let mut data = Vec::new();
        for ids in &rollout.token_ids {
            data.extend_from_slice(&ids[..input_len]);
        }
        let input = Tensor::from_vec(data, (g, input_len), &Device::Cpu).unwrap();
        let logp_full = log_softmax(&policy.model().forward(&input).unwrap(), D::Minus1)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let comp_len = seq_len - rollout.prompt_len;
        for (gi, ids) in rollout.token_ids.iter().enumerate() {
            for j in 0..comp_len {
                let pos = rollout.prompt_len - 1 + j;
                let tgt = ids[rollout.prompt_len + j] as usize;
                let want = logp_full[gi][pos][tgt];
                assert!(
                    (got[gi][j] - want).abs() <= 1e-5,
                    "logp[{gi}][{j}]={} != manual {want} (pos {pos}, tgt {tgt})",
                    got[gi][j]
                );
            }
        }
    }

    #[test]
    fn token_logprobs_at_a_non_unit_temperature_matches_a_manual_reference() {
        // Temperature-consistent scoring: at T != 1 the log-probs must equal
        // log_softmax(logits / T) gathered at the completion tokens — recomputed
        // here independently of the narrow/scale/gather under test.
        let policy = tiny_policy_at(0.7);
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![3, 1, 4, 1, 5]], 2);
        let got = policy
            .token_logprobs(&rollout)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        let seq_len = rollout.token_ids[0].len();
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let mut data = Vec::new();
        for ids in &rollout.token_ids {
            data.extend_from_slice(&ids[..input_len]);
        }
        let input = Tensor::from_vec(data, (g, input_len), &Device::Cpu).unwrap();
        let scaled = (policy.model().forward(&input).unwrap() / 0.7).unwrap();
        let logp_full = log_softmax(&scaled, D::Minus1)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let comp_len = seq_len - rollout.prompt_len;
        for (gi, ids) in rollout.token_ids.iter().enumerate() {
            for j in 0..comp_len {
                let pos = rollout.prompt_len - 1 + j;
                let tgt = ids[rollout.prompt_len + j] as usize;
                let want = logp_full[gi][pos][tgt];
                assert!(
                    (got[gi][j] - want).abs() <= 1e-5,
                    "T=0.7 logp[{gi}][{j}]={} != manual {want}",
                    got[gi][j]
                );
            }
        }
    }

    /// One `token_logprobs -> sqr -> sum -> backward`, returning the grad store —
    /// the scoring path the trainer actually differentiates.
    fn grads_of(policy: &QwenPolicy, rollout: &Rollout) -> GradStore {
        let loss = policy
            .token_logprobs(rollout)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        loss.backward().unwrap()
    }

    /// Split the trainable vars into the (q, v) branches. Per-layer order is
    /// `q_A, q_B, v_A, v_B`, so `i % 4 < 2` is the q branch.
    fn branch_split(vars: &[Var]) -> (Vec<Var>, Vec<Var>) {
        let pick = |want_q: bool| -> Vec<Var> {
            vars.iter()
                .enumerate()
                .filter(|(i, _)| (i % 4 < 2) == want_q)
                .map(|(_, v)| v.clone())
                .collect()
        };
        (pick(true), pick(false))
    }

    /// Set every `B` factor (the odd index within each `[A, B]` pair) to small
    /// noise, so the update is no longer a no-op and `dL/dA` is no longer 0.
    fn force_b_nonzero(vars: &[Var]) {
        for (i, v) in vars.iter().enumerate() {
            if i % 2 == 1 {
                let dims = v.as_tensor().dims().to_vec();
                v.set(&Tensor::randn(0f32, 0.02f32, dims, &Device::Cpu).unwrap())
                    .unwrap();
            }
        }
    }

    #[test]
    fn lora_grads_flow_through_token_logprobs_both_branches() {
        // Deterministic proof (no sampling) that gradients reach BOTH LoRA factors
        // (A and B) of q AND v THROUGH `token_logprobs` — the narrow/log_softmax/
        // gather must not detach A. At zero-B init dL/dA is structurally 0, so a
        // severed A-path is invisible to a single backward (the P3 PR-B trap); the
        // two-phase check (force B nonzero) closes it.
        let policy = tiny_policy();
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4, 5], vec![5, 4, 3, 2, 1]], 2);
        let vars = policy.trainable_vars();
        assert_eq!(vars.len(), 2 * 4); // per layer: q_A, q_B, v_A, v_B
        let (q_vars, v_vars) = branch_split(&vars);

        // Phase 1 — zero-B: every var present + each branch live (via dL/dB) + finite.
        let g1 = grads_of(&policy, &rollout);
        assert!(
            grad_coverage(&q_vars, &g1).unwrap().is_ok(),
            "q-branch unhealthy at zero-B init"
        );
        assert!(
            grad_coverage(&v_vars, &g1).unwrap().is_ok(),
            "v-branch unhealthy at zero-B init"
        );

        // Phase 2 — force every B nonzero: now EVERY A and B must carry a nonzero
        // finite grad (proves the A-input path is wired, not just B).
        force_b_nonzero(&vars);
        let g2 = grads_of(&policy, &rollout);
        let qc = grad_coverage(&q_vars, &g2).unwrap();
        let vc = grad_coverage(&v_vars, &g2).unwrap();
        assert!(
            qc.nonzero == qc.total && qc.nonfinite == 0,
            "q-branch: not every LoRA var is live after nonzero-B (severed A?): {qc:?}"
        );
        assert!(
            vc.nonzero == vc.total && vc.nonfinite == 0,
            "v-branch: not every LoRA var is live after nonzero-B: {vc:?}"
        );
    }

    #[test]
    fn adapter_toggle_tracks_state_and_is_noop_at_zero_b() {
        let mut policy = tiny_policy();
        assert!(policy.adapter_enabled());
        let rollout = Rollout::rectangular(vec![vec![1u32, 2, 3, 4]], 2);
        let on = policy.token_logprobs(&rollout).unwrap();
        policy.set_adapter_enabled(false);
        assert!(!policy.adapter_enabled());
        let off = policy.token_logprobs(&rollout).unwrap();
        // Zero-B init: the adapter is a no-op, so enabled == disabled log-probs.
        let diff: f32 = on
            .sub(&off)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff <= 1e-6, "zero-B adapter changed log-probs: {diff}");
        policy.set_adapter_enabled(true);
        assert!(policy.adapter_enabled());
    }

    #[test]
    fn trainable_vars_are_the_models() {
        let policy = tiny_policy();
        // 2 layers x (q_A, q_B, v_A, v_B) = 8 trainable vars.
        assert_eq!(policy.trainable_vars().len(), 2 * 4);
        // The manual Debug impl elides the non-Debug sampler.
        let dbg = format!("{policy:?}");
        assert!(dbg.contains("LmPolicy") && dbg.contains(".."));
    }

    // ---- end-to-end: QwenPolicy through the real Trainer (CPU) --------------

    use crate::reward::{RewardError, RewardFn};
    use crate::sample::Sample;
    use crate::telemetry::RunDir;
    use crate::trainer::{TokenizerLike, Trainer, TrainerConfig};

    /// Trivial char codec over the tiny vocab (id `i` <-> `'a' + i`); the tiny
    /// model's vocab is 16, so generated ids land in `'a'..'p'`.
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

    /// A reward that spreads across distinct completions (so a sampled group is
    /// non-degenerate and a real GRPO update fires). Position-WEIGHTED so that two
    /// completions sharing a byte multiset (`"ab"` vs `"ba"`) do not collide to the
    /// same reward and silently degenerate the group.
    struct SpreadReward;
    impl RewardFn for SpreadReward {
        type Target = ();
        fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
            Ok(completion
                .bytes()
                .enumerate()
                .map(|(i, b)| (i as f32 + 1.0) * f32::from(b))
                .sum::<f32>()
                / 1000.0)
        }
    }

    /// A unique temp directory, removed on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new() -> Self {
            let mut p = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            p.push(format!(
                "ferrl-qwen-policy-{}-{}",
                std::process::id(),
                nanos
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Per-step metric sanity for the CPU GRPO run.
    fn assert_step_metrics_ok(m: &crate::telemetry::Metrics) {
        assert!(
            m.grad_norm.is_finite(),
            "non-finite grad_norm at step {}",
            m.step
        );
        assert!(m.reward_mean.is_finite());
        assert!(m.kl.is_finite() && m.kl >= 0.0, "bad KL at step {}", m.step);
    }

    #[test]
    fn drives_a_grpo_step_through_the_trainer_on_cpu() {
        // The same Trainer that drives the echo toy drives a (tiny) Qwen policy:
        // rollout -> reward -> advantages -> backward THROUGH the Qwen forward ->
        // grad-coverage canary -> AdamW. A clean multi-step run proves the canary
        // held on every real update (it aborts on a missing/non-finite grad).
        let mut policy = tiny_policy();
        let samples = vec![Sample::new("abc", ()), Sample::new("bcd", ())];
        // beta > 0 so the adapter-disabled KL reference forward (and its restore)
        // actually runs through the Qwen path, not just the policy forward.
        let cfg = TrainerConfig {
            steps: 4,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            beta: 0.02,
            lr: 1e-3,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run = RunDir::create(&tmp.0, "qwen-cpu").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let (history, _stop) = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &samples)
            .unwrap();

        assert_eq!(history.len(), 4);
        for m in &history {
            assert_step_metrics_ok(m);
        }
        // `grad_norm > 0` is set ONLY when an AdamW step actually runs (a real,
        // non-degenerate, non-fully-clipped update). Asserting it witnesses that the
        // Qwen backward produced a usable gradient and the optimizer stepped — far
        // stronger than `frac_reward_zero_std < 1` (which is computed from scalar
        // rewards, upstream of any backward). Deterministic A-path liveness is pinned
        // separately by `lora_grads_flow_through_token_logprobs_both_branches`.
        assert!(
            history.iter().any(|m| m.grad_norm > 0.0),
            "no AdamW step ran — the Qwen backward path was never exercised"
        );
        // The adapter is restored enabled after the (reference-toggling) run.
        assert!(policy.adapter_enabled());
    }

    #[test]
    fn rollout_ratio_telemetry_is_near_one_for_an_f32_policy() {
        // End-to-end pipeline gate for the R2 telemetry: on an all-F32 model the
        // rollout (merged cached decode) and the scoring forward differ only by
        // float reassociation, so the train/rollout ratio must sit hard against 1
        // on every step — and nothing may approach the TIS cap. A capture
        // misalignment, a temperature inconsistency, or a wiring bug shows up as
        // a ratio visibly away from 1.
        let mut policy = tiny_policy();
        force_b_nonzero(&policy.trainable_vars()); // non-trivial merge
        let samples = vec![Sample::new("abc", ()), Sample::new("bcd", ())];
        let cfg = TrainerConfig {
            steps: 3,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            lr: 1e-3,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run = RunDir::create(&tmp.0, "qwen-ratio").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let (history, _stop) = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &samples)
            .unwrap();
        assert_eq!(history.len(), 3);
        for m in &history {
            assert_on_policy_ratio_metrics(m);
        }
    }

    /// Per-step assertions for an all-F32 (reassociation-only) run: ratio
    /// mean/max hard against 1, the drift meter against 0, nothing capped, and
    /// the telemetry not dark.
    fn assert_on_policy_ratio_metrics(m: &crate::telemetry::Metrics) {
        assert!(
            (m.rollout_ratio_mean - 1.0).abs() <= 1e-3,
            "step {}: rollout_ratio_mean {} far from 1 on an F32 model",
            m.step,
            m.rollout_ratio_mean
        );
        assert!(
            (m.rollout_ratio_max - 1.0).abs() <= 1e-3,
            "step {}: rollout_ratio_max {} far from 1 on an F32 model",
            m.step,
            m.rollout_ratio_max
        );
        assert_eq!(
            m.frac_rollout_ratio_capped, 0.0,
            "step {}: no token can sit above the TIS cap on an F32 model",
            m.step
        );
        assert!(
            m.rollout_logratio_mean.abs() <= 1e-3,
            "step {}: drift meter {} far from 0 on an F32 model",
            m.step,
            m.rollout_logratio_mean
        );
        assert!(
            m.rollout_capture_tokens > 0,
            "step {}: telemetry must not be dark — the policy captures",
            m.step
        );
    }

    #[test]
    fn tis_enabled_run_completes_with_near_unit_weights() {
        // With TIS ON against an F32 policy the weights are ~1, so the run must
        // behave like the uncorrected one: finite metrics, real optimizer steps.
        // (The fail-loud path for a policy WITHOUT capture is pinned in
        // tests/toy_echo.rs; the weight math itself in trainer.rs unit tests.)
        let mut policy = tiny_policy();
        let samples = vec![Sample::new("abc", ()), Sample::new("bcd", ())];
        let cfg = TrainerConfig {
            steps: 3,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            lr: 1e-3,
            tis: true,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run = RunDir::create(&tmp.0, "qwen-tis").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let (history, _stop) = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &samples)
            .unwrap();
        assert_eq!(history.len(), 3);
        for m in &history {
            assert_step_metrics_ok(m);
        }
        assert!(
            history.iter().any(|m| m.grad_norm > 0.0),
            "no optimizer step ran under TIS"
        );
    }

    /// A [`Policy`] wrapper that shifts every captured behavior log-prob DOWN
    /// by a constant `delta` — claiming the rollout assigned `e^delta`-times
    /// less mass to each token than it actually did — so the train/rollout
    /// ratio is `e^delta` (× merge-reassociation noise) at every loss token by
    /// construction: a deterministic off-policy injection for the end-to-end
    /// TIS / telemetry gates.
    struct ShiftedCapture<P: Policy> {
        inner: P,
        delta: f32,
    }

    impl<P: Policy> Policy for ShiftedCapture<P> {
        fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
            self.generate_at(prompt, cfg, 0)
        }
        fn generate_at(
            &mut self,
            prompt: &[u32],
            cfg: &GenConfig,
            global_row_base: u64,
        ) -> CandleResult<Rollout> {
            let mut r = self.inner.generate_at(prompt, cfg, global_row_base)?;
            if let Some(rows) = &mut r.rollout_logprobs {
                for row in rows {
                    for lp in row {
                        *lp -= self.delta;
                    }
                }
            }
            Ok(r)
        }
        fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
            self.inner.token_logprobs(rollout)
        }
        fn set_adapter_enabled(&mut self, enabled: bool) {
            self.inner.set_adapter_enabled(enabled);
        }
        fn adapter_enabled(&self) -> bool {
            self.inner.adapter_enabled()
        }
        fn trainable_vars(&self) -> Vec<Var> {
            self.inner.trainable_vars()
        }
        fn sampler_state(&self) -> CandleResult<Vec<u8>> {
            self.inner.sampler_state()
        }
        fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
            self.inner.restore_sampler_state(state)
        }
        fn lora_recipe(&self) -> Option<String> {
            self.inner.lora_recipe()
        }
    }

    /// Per-run assertions for the δ-shifted capture: ratio mean at e^δ (and
    /// NOT e^{−δ} — the direction pin), the drift meter at +δ, every loss
    /// token above the cap, telemetry not dark.
    fn assert_shifted_ratio_metrics(m: &crate::telemetry::Metrics, delta: f32) {
        assert!(
            (m.rollout_ratio_mean - 2.0).abs() <= 2e-3,
            "ratio mean {} != e^ln2 = 2 (direction/magnitude)",
            m.rollout_ratio_mean
        );
        assert!(
            (m.rollout_logratio_mean - delta).abs() <= 1e-3,
            "drift meter {} != +ln2",
            m.rollout_logratio_mean
        );
        assert_eq!(
            m.frac_rollout_ratio_capped, 1.0,
            "every token sits above the cap by construction"
        );
        assert!(m.rollout_capture_tokens > 0);
    }

    #[test]
    fn tis_and_ratio_telemetry_verified_end_to_end_with_a_shifted_capture() {
        // δ-shifted capture ⇒ every loss token's ratio is e^δ ≈ 2; with cap
        // C = 1.5 < e^δ, EVERY token caps. Through Trainer::train this pins:
        //   (1) the ratio DIRECTION (a swapped exp(b−a) would read e^{−δ} = ½);
        //   (2) the log-ratio drift meter (≈ +δ);
        //   (3) the capped-fraction wiring (≡ 1.0) and the token count;
        //   (4) the TIS weight reaching the GRADIENT: uniformly capped weights
        //       make the first step's pre-clip grad_norm exactly C × the
        //       tis-off run's (paired policies ⇒ identical rollouts/logp_old).
        // A dropped `tis_w` (the one-line mutant), a swapped ratio, or a
        // disconnected capped-fraction all redden this test.
        let delta = std::f64::consts::LN_2;
        let cap = 1.5_f64;
        let (p_a, p_b) = paired_policies();
        // paired_policies shares the BASE weights and sampler seed, but each
        // policy draws its own random LoRA A factors — invisible to the forward
        // at B = 0 (so rollouts/logp_old still match), yet dL/dB ∝ A, so the
        // grad-norm comparison below needs the adapters synced too.
        for (va, vb) in p_a.trainable_vars().iter().zip(p_b.trainable_vars()) {
            vb.set(va.as_tensor()).unwrap();
        }
        let mut off = ShiftedCapture {
            inner: p_a,
            delta: delta as f32,
        };
        let mut on = ShiftedCapture {
            inner: p_b,
            delta: delta as f32,
        };
        let samples = vec![Sample::new("abc", ()), Sample::new("bcd", ())];
        let cfg = |tis: bool| TrainerConfig {
            steps: 1,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            lr: 1e-3,
            tis,
            tis_imp_ratio_cap: cap,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run_off = RunDir::create(&tmp.0, "tis-off").unwrap();
        let m_off = Trainer::new(cfg(false), &run_off)
            .unwrap()
            .train(&mut off, &SpreadReward, &CharCodec, &samples)
            .unwrap()
            .0
            .remove(0);
        let run_on = RunDir::create(&tmp.0, "tis-on").unwrap();
        let m_on = Trainer::new(cfg(true), &run_on)
            .unwrap()
            .train(&mut on, &SpreadReward, &CharCodec, &samples)
            .unwrap()
            .0
            .remove(0);

        for m in [&m_off, &m_on] {
            assert_shifted_ratio_metrics(m, delta as f32);
        }
        // (4) the weight scales the gradient.
        assert!(m_off.grad_norm > 0.0, "tis-off run must take a real step");
        let scale = m_on.grad_norm / m_off.grad_norm;
        assert!(
            (f64::from(scale) - cap).abs() <= 1e-3,
            "grad_norm scaled by {scale}, want exactly the cap {cap}"
        );
    }

    #[test]
    fn evaluate_honors_the_eval_sampling_override_end_to_end() {
        // The held-out eval harness over a real (tiny) policy with the eval-only
        // sampling convention: a temperature different from the baked one plus
        // nucleus top-p must generate (no temperature bail) and produce a finite
        // report, with the adapter flag restored.
        let mut policy = tiny_policy();
        let samples = vec![Sample::new("abc", ())];
        let gen = GenConfig {
            group_size: 4,
            max_new_tokens: 3,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling::default()), // T 0.6 / top-p 0.95
        };
        let report =
            crate::eval::evaluate(&mut policy, &SpreadReward, &CharCodec, &samples, &gen).unwrap();
        assert_eq!(report.n_prompts, 1);
        assert_eq!(report.group_size, 4);
        assert!(report.base_reward_mean.is_finite());
        assert!(report.adapter_reward_mean.is_finite());
        assert!(policy.adapter_enabled(), "adapter flag restored");
    }

    // ---- LlamaPolicy: the M1 second-implementor gates ------------------------
    //
    // Everything below reuses the GENERIC machinery above unchanged
    // (`assert_cached_matches_uncached`, `force_b_nonzero`, the codec/reward/
    // trainer scaffold) — only the model construction is Llama-specific. That
    // reuse IS the point: it witnesses that the `GradModel` seam, not the test
    // code, carries the architecture difference.

    use crate::llama::LlamaGradModel;
    use candle_transformers::models::llama::Config as LlamaConfig;

    /// A tiny dense-Llama config (2 layers, 2 Q / 1 KV head → real GQA, derived
    /// `head_dim` 4) — the same scaffold llama.rs's tests use.
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

    fn llama_tiny_policy() -> LlamaPolicy {
        let cfg = llama_tiny_cfg();
        let vb = VarBuilder::from_tensors(llama_weight_map(&cfg), DType::F32, &Device::Cpu);
        let model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
        LlamaPolicy::new(model, 7, 1.0)
    }

    /// Two Llama policies sharing the SAME base weights and sampler seed (the
    /// same determinism device as [`paired_policies`] — see there for why).
    fn llama_paired_policies() -> (LlamaPolicy, LlamaPolicy) {
        let cfg = llama_tiny_cfg();
        let weights = llama_weight_map(&cfg);
        let build = || {
            let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
            let model = LlamaGradModel::load(&cfg, &vb, 2, 4.0).unwrap();
            LlamaPolicy::new(model, 7, 1.0)
        };
        (build(), build())
    }

    #[test]
    fn llama_cached_generate_matches_uncached_adapter_on() {
        // Armed adapter (B != 0): the cached path must reproduce the
        // ADAPTER-AWARE uncached stream, not merely the base one.
        let mut policy = llama_tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        assert!(policy.adapter_enabled());
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 6,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[1u32, 2, 3], &cfg);
    }

    #[test]
    fn llama_cached_generate_matches_uncached_adapter_off() {
        // The eval path: adapter disabled => the snapshot is the pure base model.
        let mut policy = llama_tiny_policy();
        force_b_nonzero(&policy.trainable_vars()); // armed, but...
        policy.set_adapter_enabled(false); // ...disabled => base only
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[2u32, 4, 1], &cfg);
    }

    #[test]
    fn llama_cached_generate_matches_uncached_with_eos() {
        // EOS early-stop + right-pad identical between paths, with matching
        // sampler-RNG consumption (same deterministic paired-probe pattern as
        // the Qwen gate).
        let prompt = [1u32, 2, 3];
        let max_new = 5usize;
        let (mut probe, mut policy) = llama_paired_policies();
        let base = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = probe.generate_uncached(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..base
        };
        assert_cached_matches_uncached(&mut policy, &prompt, &cfg_eos);
    }

    #[test]
    fn llama_drives_a_grpo_step_through_the_trainer_on_cpu() {
        // THE M1 extended reusability gate: the SAME `Trainer` (and the same
        // codec + reward scaffold) that drives the P2 echo toy and the Qwen
        // policy drives `LmPolicy<LlamaGradModel>` UNCHANGED — rollout via the
        // Llama merged decoder, reward, advantages, backward THROUGH the Llama
        // forward, grad-coverage canary, FerrlAdamW. `grad_norm > 0` witnesses
        // a real optimizer step (no learning-curve assertion — the platform-
        // dependence lesson); beta > 0 routes the adapter-disabled KL reference
        // forward through the Llama path too.
        let mut policy = llama_tiny_policy();
        let samples = vec![Sample::new("abc", ()), Sample::new("bcd", ())];
        let cfg = TrainerConfig {
            steps: 4,
            group_size: 6,
            max_new_tokens: 4,
            temperature: 1.0,
            beta: 0.02,
            lr: 1e-3,
            ..TrainerConfig::default()
        };
        let tmp = TempDir::new();
        let run = RunDir::create(&tmp.0, "llama-cpu").unwrap();
        let mut trainer = Trainer::new(cfg, &run).unwrap();
        let (history, _stop) = trainer
            .train(&mut policy, &SpreadReward, &CharCodec, &samples)
            .unwrap();

        assert_eq!(history.len(), 4);
        for m in &history {
            assert_step_metrics_ok(m);
        }
        assert!(
            history.iter().any(|m| m.grad_norm > 0.0),
            "no AdamW step ran — the Llama backward path was never exercised"
        );
        // The adapter is restored enabled after the (reference-toggling) run.
        assert!(policy.adapter_enabled());
    }

    // ---- Gemma4Policy: production-shaped cached generation gate --------------

    const GEMMA4_TINY_TEXT_CONFIG: &str = r#"{
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": true,
            "enable_moe_block": false,
            "expert_intermediate_size": null,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 8,
            "head_dim": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 8,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": 16,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 32,
            "model_type": "gemma4_text",
            "num_attention_heads": 2,
            "num_experts": null,
            "num_global_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "sliding_window": 3,
            "tie_word_embeddings": true,
            "top_k_experts": null,
            "use_bidirectional_attention": "vision",
            "use_cache": true,
            "use_double_wide_mlp": false,
            "vocab_size": 16,
            "vocab_size_per_layer_input": 16
        }
    }"#;

    fn gemma4_tiny_cfg() -> Gemma4Config {
        Gemma4Config::from_json_str(GEMMA4_TINY_TEXT_CONFIG).unwrap()
    }

    fn gemma4_put_rand(t: &mut HashMap<String, Tensor>, name: &str, dims: &[usize]) {
        t.insert(
            name.to_string(),
            Tensor::randn(0f32, 0.05f32, dims.to_vec(), &Device::Cpu).unwrap(),
        );
    }

    fn gemma4_put_ones(t: &mut HashMap<String, Tensor>, name: &str, dims: &[usize]) {
        t.insert(
            name.to_string(),
            Tensor::ones(dims.to_vec(), DType::F32, &Device::Cpu).unwrap(),
        );
    }

    fn gemma4_weight_map(cfg: &Gemma4Config) -> HashMap<String, Tensor> {
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let tcfg = &cfg.text_config;
        let h = tcfg.hidden_size;
        let i = tcfg.intermediate_size;
        gemma4_put_rand(
            &mut t,
            &format!("{CKPT_PREFIX}.embed_tokens.weight"),
            &[tcfg.vocab_size, h],
        );
        gemma4_put_ones(&mut t, &format!("{CKPT_PREFIX}.norm.weight"), &[h]);
        for layer in 0..tcfg.num_hidden_layers {
            let p = format!("{CKPT_PREFIX}.layers.{layer}");
            let full = tcfg.layer_types[layer] == Gemma4LayerType::FullAttention;
            let head_dim = if full {
                tcfg.global_head_dim
            } else {
                tcfg.head_dim
            };
            let kv_heads = if full {
                tcfg.num_global_key_value_heads
            } else {
                tcfg.num_key_value_heads
            };
            let q_out = tcfg.num_attention_heads * head_dim;
            let kv_out = kv_heads * head_dim;

            gemma4_put_ones(&mut t, &format!("{p}.input_layernorm.weight"), &[h]);
            gemma4_put_ones(
                &mut t,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            );
            gemma4_put_ones(
                &mut t,
                &format!("{p}.pre_feedforward_layernorm.weight"),
                &[h],
            );
            gemma4_put_ones(
                &mut t,
                &format!("{p}.post_feedforward_layernorm.weight"),
                &[h],
            );
            gemma4_put_ones(&mut t, &format!("{p}.layer_scalar"), &[1]);
            gemma4_put_rand(&mut t, &format!("{p}.self_attn.q_proj.weight"), &[q_out, h]);
            gemma4_put_rand(
                &mut t,
                &format!("{p}.self_attn.k_proj.weight"),
                &[kv_out, h],
            );
            if !full {
                gemma4_put_rand(
                    &mut t,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[kv_out, h],
                );
            }
            gemma4_put_rand(&mut t, &format!("{p}.self_attn.o_proj.weight"), &[h, q_out]);
            gemma4_put_ones(&mut t, &format!("{p}.self_attn.q_norm.weight"), &[head_dim]);
            gemma4_put_ones(&mut t, &format!("{p}.self_attn.k_norm.weight"), &[head_dim]);
            gemma4_put_rand(&mut t, &format!("{p}.mlp.gate_proj.weight"), &[i, h]);
            gemma4_put_rand(&mut t, &format!("{p}.mlp.up_proj.weight"), &[i, h]);
            gemma4_put_rand(&mut t, &format!("{p}.mlp.down_proj.weight"), &[h, i]);
        }
        t
    }

    fn gemma4_tiny_policy() -> Gemma4Policy {
        let cfg = gemma4_tiny_cfg();
        let vb = VarBuilder::from_tensors(gemma4_weight_map(&cfg), DType::F32, &Device::Cpu);
        let model =
            Gemma4GradModel::load_with_adapter_dtype(&cfg, &vb, 2, 4.0, DType::F32).unwrap();
        Gemma4Policy::new(model, 7, 1.0)
    }

    #[derive(Default)]
    struct CollectingTelemetry {
        phases: Vec<&'static str>,
        cache: Vec<DecoderCacheSnapshot>,
    }

    impl ModelTelemetryRecorder for CollectingTelemetry {
        fn record_phase(&mut self, phase: &'static str) {
            self.phases.push(phase);
        }

        fn record_decoder_cache(&mut self, snapshots: Vec<DecoderCacheSnapshot>) {
            self.cache.extend(snapshots);
        }
    }

    fn snapshots_for_phase(
        snapshots: &[DecoderCacheSnapshot],
        phase: &str,
    ) -> Vec<DecoderCacheSnapshot> {
        snapshots
            .iter()
            .filter(|snapshot| snapshot.phase == phase)
            .cloned()
            .collect()
    }

    #[test]
    fn gemma4_instrumented_generate_reports_prefill_decode_and_cache_phases() {
        let mut policy = gemma4_tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        let cfg = GenConfig {
            group_size: 3,
            max_new_tokens: 4,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let prompt = [1u32, 2, 3, 4, 5];
        let mut telemetry = CollectingTelemetry::default();

        let rollout = policy
            .generate_at_instrumented(&prompt, &cfg, 0, Some(&mut telemetry))
            .unwrap();
        assert_eq!(rollout.len(), cfg.group_size);
        assert_eq!(
            telemetry.phases,
            [
                "merged_decoder_build_start",
                "merged_decoder_build_end",
                "rollout_prefill_start",
                "rollout_prefill_end",
                "rollout_decode_start",
                "rollout_decode_end",
            ]
        );
        assert_eq!(
            snapshots_for_phase(&telemetry.cache, "merged_decoder_build_end").len(),
            2
        );

        let seen_tokens = prompt.len() + cfg.max_new_tokens - 1;
        assert_eq!(
            snapshots_for_phase(&telemetry.cache, "rollout_decode_end"),
            vec![
                DecoderCacheSnapshot {
                    phase: "rollout_decode_end".to_string(),
                    layer_index: 0,
                    kind: "sliding_attention".to_string(),
                    seen_tokens,
                    retained_tokens: 3,
                    max_retained_tokens: Some(3),
                },
                DecoderCacheSnapshot {
                    phase: "rollout_decode_end".to_string(),
                    layer_index: 1,
                    kind: "full_attention".to_string(),
                    seen_tokens,
                    retained_tokens: seen_tokens,
                    max_retained_tokens: None,
                },
            ]
        );
    }

    #[test]
    fn gemma4_cached_generate_matches_uncached_batched_prefill_and_decode() {
        let mut policy = gemma4_tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        assert_cached_matches_uncached(&mut policy, &[1u32, 2, 3, 4, 5], &cfg);
    }

    #[test]
    fn gemma4_cached_generate_matches_uncached_with_eos() {
        let prompt = [1u32, 2, 3, 4, 5];
        let max_new = 5usize;
        let mut policy = gemma4_tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        let base = GenConfig {
            group_size: 4,
            max_new_tokens: max_new,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        let eos = policy.generate_uncached(&prompt, &base).unwrap().token_ids[0][prompt.len()];
        let cfg_eos = GenConfig {
            eos_token_id: Some(eos),
            ..base
        };
        assert_cached_matches_uncached(&mut policy, &prompt, &cfg_eos);
        let rollout = policy.generate(&prompt, &cfg_eos).unwrap();
        assert_eq!(rollout.completion_lens[0], 1);
        assert_eos_rollout_invariants(&rollout, eos, max_new);
    }

    #[test]
    fn gemma4_cached_generate_matches_uncached_under_eval_override() {
        let mut policy = gemma4_tiny_policy();
        force_b_nonzero(&policy.trainable_vars());
        let cfg = GenConfig {
            group_size: 4,
            max_new_tokens: 5,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: Some(crate::policy::EvalSampling {
                temperature: 0.7,
                top_p: Some(0.9),
            }),
        };
        assert_cached_matches_uncached(&mut policy, &[1u32, 2, 3, 4, 5], &cfg);
    }

    // ---- detached scoring + checkpointed backward (P7) ----------------------

    /// Force every adapter `B` nonzero so the adapter path is live in the
    /// scored logits (at the `B = 0` init both scorings would trivially agree).
    fn arm_policy(policy: &QwenPolicy) {
        for v in policy.trainable_vars().iter().skip(1).step_by(2) {
            let dims = v.as_tensor().dims().to_vec();
            v.set(&Tensor::randn(0f32, 0.5f32, dims, &Device::Cpu).unwrap())
                .unwrap();
        }
    }

    fn small_rollout(policy: &mut QwenPolicy) -> Rollout {
        let cfg = GenConfig {
            group_size: 2,
            max_new_tokens: 3,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        };
        policy.generate(&[1u32, 2, 3], &cfg).unwrap()
    }

    /// A fixed non-uniform weighted sum of the scored log-probs — the loss
    /// stand-in for the backward-seam tests.
    fn probe_loss_of(policy: &QwenPolicy, rollout: &Rollout) -> Tensor {
        let logp = policy.token_logprobs(rollout).unwrap();
        let n = logp.elem_count();
        let w: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.3 - 0.5).collect();
        let w = Tensor::from_vec(w, logp.dims().to_vec(), &Device::Cpu).unwrap();
        logp.mul(&w).unwrap().sum_all().unwrap()
    }

    #[test]
    fn detached_scoring_matches_token_logprobs_and_is_tape_free() {
        let mut policy = tiny_policy();
        arm_policy(&policy);
        let rollout = small_rollout(&mut policy);

        let live = policy.token_logprobs(&rollout).unwrap();
        let det = policy.token_logprobs_detached(&rollout).unwrap();
        assert_eq!(
            det.to_vec2::<f32>().unwrap(),
            live.to_vec2::<f32>().unwrap(),
            "the detached scoring drifted from token_logprobs"
        );

        // Tape-free: a backward through the detached scores reaches no var…
        let store = det.sum_all().unwrap().backward().unwrap();
        assert!(policy
            .trainable_vars()
            .iter()
            .all(|v| store.get(v).is_none()));
        // …while the live path IS on the tape (the comparison is non-vacuous).
        let store = live.sum_all().unwrap().backward().unwrap();
        assert!(policy
            .trainable_vars()
            .iter()
            .any(|v| store.get(v).is_some()));
    }

    /// Chunking the log-softmax/gather stage over positions is exact: the
    /// softmax reduces over the vocab axis only and the gather is
    /// positionwise, so every chunk size — degenerate ones included — matches
    /// an INDEPENDENTLY computed unchunked reference (full-window log-softmax
    /// then per-position lookup), and so does the public scoring path. The
    /// handcrafted rollout has position-distinct completion tokens, so a
    /// gather reading the wrong chunk's targets cannot pass. Also pins the
    /// new fail-loud preconditions: full-width (un-narrowed) pred and a
    /// zero-length window are rejected, not silently mis-scored.
    #[test]
    fn chunked_scoring_is_identical_across_chunk_sizes() {
        let policy = tiny_policy();
        arm_policy(&policy);
        let rollout = Rollout::rectangular(vec![vec![1, 2, 3, 4, 5, 6], vec![3, 1, 2, 6, 4, 5]], 3);
        let input = policy.scoring_input(&rollout).unwrap();
        let (start, len) = QwenPolicy::scoring_window(&rollout);
        let pred = policy.model().forward_narrowed(&input, start, len).unwrap();

        let logp = log_softmax(&pred.to_dtype(DType::F32).unwrap(), D::Minus1).unwrap();
        let mut reference: Vec<Vec<f32>> = Vec::new();
        for (gi, ids) in rollout.token_ids.iter().enumerate() {
            let row = ids[rollout.prompt_len..]
                .iter()
                .enumerate()
                .map(|(pi, &t)| {
                    logp.get(gi)
                        .unwrap()
                        .get(pi)
                        .unwrap()
                        .get(t as usize)
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                })
                .collect();
            reference.push(row);
        }

        for chunk in [0usize, 1, 2, 64, usize::MAX] {
            let got = policy
                .completion_logprobs_chunked(&rollout, &pred, chunk)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            assert_eq!(got, reference, "chunk size {chunk} diverged");
        }
        // The public scoring path agrees too (it fixes SCORING_CHUNK).
        let public = policy
            .token_logprobs(&rollout)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(public, reference);

        // Fail-loud preconditions: full-width pred is rejected…
        let full = policy.model().forward(&input).unwrap();
        let err = policy
            .completion_logprobs_chunked(&rollout, &full, 64)
            .unwrap_err();
        assert!(err.to_string().contains("narrowed scoring window"));
        // …and so is a zero-length completion window.
        let empty = Rollout::rectangular(vec![vec![1, 2, 3], vec![4, 5, 6]], 3);
        let err = policy
            .completion_logprobs_chunked(&empty, &pred, 64)
            .unwrap_err();
        assert!(err.to_string().contains("zero-length"));
    }

    /// `Policy::backward` under activation checkpointing: full var coverage,
    /// gradients matching the uncut run on the same instance — the
    /// policy-level end-to-end of the remat stitch.
    ///
    /// **Scope: F32, by necessity.** `tiny_policy` is all-F32 and candle's CPU
    /// backend has no bf16 matmul (`UnsupportedDTypeForOp`), so this CI gate can
    /// only assert the stitch bit-exactly at F32. The regime discovery actually
    /// runs in — a BF16 frozen base with an F32 adapter — is gated by the cuda-only
    /// `remat_equivalence_holds_in_the_bf16_base_regime` below (a manual GPU gate,
    /// since the recompute's bf16 reassociation cannot be exercised on CPU).
    #[test]
    fn policy_backward_stitches_under_checkpointing() {
        let mut policy = tiny_policy();
        arm_policy(&policy);
        let rollout = small_rollout(&mut policy);
        let vars = policy.trainable_vars();

        let plain = policy.backward(&probe_loss_of(&policy, &rollout)).unwrap();
        policy.model_mut().set_activation_checkpointing(true);
        let stitched = policy.backward(&probe_loss_of(&policy, &rollout)).unwrap();

        for v in &vars {
            let a = plain.get(v).expect("var missing from the uncut store");
            let b = stitched
                .get(v)
                .expect("var missing from the stitched store");
            let diff: f32 = a
                .sub(b)
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
                "stitched grad diverged from the uncut backward by {diff}"
            );
        }
    }

    /// The bf16-base companion to [`policy_backward_stitches_under_checkpointing`]:
    /// the remat stitch must reproduce the uncut backward in the regime discovery
    /// runs in — a **BF16 frozen base with an F32 adapter** — not only at F32.
    ///
    /// candle's CPU backend has no bf16 matmul, so this is a **manual GPU gate**
    /// (cuda-only; absent from the default build and CI). Run it on a GPU node:
    /// `cargo test --features cuda remat_equivalence_holds_in_the_bf16_base_regime`.
    /// The recompute is deterministic, so remat-vs-uncut can differ only by bf16
    /// reassociation of the per-segment accumulation — bounded here by an honest
    /// envelope well below the adapter-grad scale. The negative control (a boundary
    /// cotangent truncated before the stitch's VJP surrogate) blows past the
    /// envelope, so the gate is not vacuous.
    #[cfg(feature = "cuda")]
    // A manual GPU gate prints its measured envelope (max_diff / grad scale) so a
    // re-run shows the gate is exercising real grads, not vacuously passing.
    #[allow(clippy::print_stderr)]
    #[test]
    fn remat_equivalence_holds_in_the_bf16_base_regime() {
        let device = Device::new_cuda(0).expect("a cuda device for the bf16 remat gate");
        let cfg = tiny_cfg();
        // The production dtype split: BF16 frozen base, F32 adapter. weight_map
        // builds F32 CPU tensors — move them to the GPU and cast the base to BF16.
        let weights: HashMap<String, Tensor> = weight_map(&cfg)
            .into_iter()
            .map(|(k, v)| {
                let t = v.to_device(&device).unwrap().to_dtype(DType::BF16).unwrap();
                (k, t)
            })
            .collect();
        let vb = VarBuilder::from_tensors(weights, DType::BF16, &device);
        let model = QwenGradModel::load_with_adapter_dtype(&cfg, &vb, 2, 4.0, DType::F32).unwrap();
        let mut policy = QwenPolicy::new(model, 7, 1.0);

        // Arm every adapter B nonzero (F32 adapter, on the GPU) so A and B both
        // carry a live gradient (dL/dA ∝ B). Device/dtype-matched, unlike the
        // CPU `arm_policy`.
        for v in policy.trainable_vars().iter().skip(1).step_by(2) {
            let dims = v.as_tensor().dims().to_vec();
            v.set(&Tensor::randn(0f32, 0.5f32, dims, &device).unwrap())
                .unwrap();
        }
        let rollout = small_rollout(&mut policy);
        let vars = policy.trainable_vars();

        // A fixed non-uniform weighted sum of the scored log-probs, on-device.
        let probe = |p: &QwenPolicy| -> Tensor {
            let logp = p.token_logprobs(&rollout).unwrap();
            let n = logp.elem_count();
            let w: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.3 - 0.5).collect();
            let w = Tensor::from_vec(w, logp.dims().to_vec(), &device).unwrap();
            logp.mul(&w).unwrap().sum_all().unwrap()
        };

        let plain = policy.backward(&probe(&policy)).unwrap();
        policy.model_mut().set_activation_checkpointing(true);
        let stitched = policy.backward(&probe(&policy)).unwrap();

        // Honest, scale-invariant BF16 envelope: the worst per-var grad
        // disagreement must be a small *fraction* of the grad scale. The adapter
        // grads are F32 but are formed from BF16 activations the stitch recomputes,
        // so remat-vs-uncut could differ by bf16 reassociation; `rel_tol` is a few
        // bf16 ulps of headroom. In practice each adapter var lives in a single
        // segment, so there is no cross-segment reassociation and the grads come out
        // bit-identical (measured `max_diff == 0` on an Ampere-class GPU) — the envelope is
        // conservative, and the teeth are the negative control: a boundary cotangent
        // truncated to a coarse grid before the VJP surrogate drives `max_diff` to a
        // large fraction of the grad scale, far past `rel_tol`.
        let rel_tol = 2e-2_f32;
        let scalar = |t: &Tensor| -> f32 {
            t.abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar()
                .unwrap()
        };
        let mut max_diff = 0f32;
        let mut max_mag = 0f32;
        for v in &vars {
            let a = plain
                .get(v)
                .expect("uncut store missing a var")
                .to_dtype(DType::F32)
                .unwrap();
            let b = stitched
                .get(v)
                .expect("stitched store missing a var")
                .to_dtype(DType::F32)
                .unwrap();
            max_diff = max_diff.max(scalar(&a.sub(&b).unwrap()));
            max_mag = max_mag.max(scalar(&a));
        }
        // Non-vacuity: the uncut backward must carry a real gradient to compare.
        assert!(
            max_mag > 0.0,
            "every uncut grad was zero — the comparison is vacuous"
        );
        eprintln!("REMAT_BF16_EQUIV max_diff={max_diff} max_grad_mag={max_mag} rel_tol={rel_tol}");
        assert!(
            max_diff <= rel_tol * max_mag,
            "bf16-base remat grad diverged from the uncut backward by {max_diff} \
             (> {rel_tol} * {max_mag})"
        );
    }

    /// The tape-preservation invariant, in the order that would clobber it: a
    /// LIVE scoring captures the tape, a value scoring runs AFTER it, and the
    /// live loss must still stitch. Kills the mutant where
    /// `token_logprobs_detached` routes through the tape-capturing `forward`
    /// (which would silently replace the pending tape — and with it the whole
    /// memory feature): under that mutant the backward below fails loud with
    /// the foreign-loss error.
    #[test]
    fn a_value_scoring_does_not_disturb_the_pending_tape() {
        let mut policy = tiny_policy();
        arm_policy(&policy);
        let rollout = small_rollout(&mut policy);
        policy.model_mut().set_activation_checkpointing(true);

        let loss = probe_loss_of(&policy, &rollout); // live: captures the tape
        let _ = policy.token_logprobs_detached(&rollout).unwrap(); // must not touch it
        let grads = policy.backward(&loss).unwrap(); // must still stitch
        for v in policy.trainable_vars() {
            assert!(
                grads.get(&v).is_some(),
                "the stitched store lost a var — the value scoring disturbed the tape"
            );
        }
    }

    /// The full trainer loop with checkpointing ON equals the loop with it
    /// OFF: paired policies (shared base weights + sampler seed, adapter vars
    /// synced — independently drawn `A` factors are invisible to the forward
    /// at `B = 0` but `dL/dB ∝ A`, the R2 lesson), identical config with
    /// `mu = 2` inner epochs and `beta > 0` (so the detached `logp_old` + KL
    /// reference scorings and the repeated live-forward/backward tape
    /// pairings all run through the real `Trainer`), then the trained vars
    /// must agree within float tolerance.
    #[test]
    fn trainer_with_checkpointing_matches_without_end_to_end() {
        let (mut off, mut on) = paired_policies();
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
            ..TrainerConfig::default()
        };
        let samples = vec![Sample::new("abc", ()), Sample::new("bcd", ())];
        let run_one = |policy: &mut QwenPolicy, tag: &str| {
            let tmp = TempDir::new();
            let run = RunDir::create(&tmp.0, tag).unwrap();
            let mut trainer = Trainer::new(cfg.clone(), &run).unwrap();
            let (history, _stop) = trainer
                .train(policy, &SpreadReward, &CharCodec, &samples)
                .unwrap();
            assert!(
                history.iter().any(|m| m.grad_norm > 0.0),
                "{tag}: no real update ran — the comparison would be vacuous"
            );
        };
        run_one(&mut off, "remat-off");
        run_one(&mut on, "remat-on");

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
                "trained vars diverged between checkpointing on/off: {diff}"
            );
        }
    }
}
