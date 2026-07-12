//! The model-generality seam: [`GradModel`] / [`CachedDecoder`].
//!
//! These two traits are the *entire* surface a model must provide for the
//! generic policy ([`crate::lm_policy::LmPolicy`]) — and through it the
//! `Trainer`, checkpointing, and eval, which only know the [`crate::Policy`]
//! seam — to RL-fine-tune it. [`crate::qwen::QwenGradModel`] (with
//! [`crate::qwen::MergedDecoder`] as its cached decoder) is the first
//! implementor; a second architecture only needs its own grad-bearing forward
//! and merged-weight cached decoder on the shared building blocks in
//! [`crate::blocks`].
//!
//! ## The two-forward split (the contract in one paragraph)
//!
//! Training needs an **uncached, tape-bearing** full-sequence forward
//! ([`GradModel::forward`]) whose backward reaches every
//! [`trainable_vars`](GradModel::trainable_vars) entry; rollout wants a
//! **stateful, tape-free** incremental decoder
//! ([`GradModel::merged_decoder`] → [`CachedDecoder`]) over a snapshot of the
//! *same* effective weights. What the decoder's state *is* belongs to the
//! implementor — a KV cache for pure-attention models, conv + recurrent state
//! matrices for linear-attention hybrids. The two forwards are pinned equal
//! (position-by-position logit equivalence) by each implementor's gates; the
//! trait only carries the obligations.

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::ops::log_softmax;

use crate::comm::Comm;
use crate::telemetry::DecoderCacheSnapshot;
use crate::tensor_parallel::plan_from_comm;

/// A stateful, grad-free incremental decoder over a snapshot of a
/// [`GradModel`]'s effective weights.
///
/// Obtained from [`GradModel::merged_decoder`]; consumed by the generic
/// policy's `generate` loop (prefill the prompt at offset 0, then decode one
/// token at a time at the running offset).
pub trait CachedDecoder {
    /// Logits `[batch, chunk_len, vocab]` for `input_ids` (`[batch, chunk_len]`,
    /// `u32`) placed at absolute positions `[offset, offset + chunk_len)`,
    /// advancing the decoder state.
    ///
    /// CONTRACT:
    /// - **Every position is returned** (the caller narrows to the last for
    ///   sampling), matching [`GradModel::forward`]'s full-sequence shape.
    /// - `offset` **must equal the number of tokens already consumed** — it
    ///   positions `RoPE` and sizes the causal mask, and a single-token decode
    ///   builds no mask, so a desync would *silently* corrupt the logits rather
    ///   than trip a shape error. An implementation must **fail loud** on a
    ///   mismatch (return an error, leaving the state untouched), never decode at
    ///   the wrong position.
    /// - The output is **tape-free**: no autograd graph is recorded; calling
    ///   `backward` through it must be impossible by construction (the decoder
    ///   holds no [`Var`]).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `offset` does not equal the number of tokens
    /// already consumed, if any tensor op fails, or if `offset + chunk_len`
    /// exceeds the model's maximum position.
    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> CandleResult<Tensor>;

    /// Tensor-parallel variant of [`forward`](Self::forward), driven by an
    /// explicitly supplied communicator.
    ///
    /// The default preserves fail-closed behavior for unsupported decoders:
    /// single-rank communicators reuse the ordinary cached decode, while
    /// sharded worlds fail loud unless a concrete decoder overrides this method.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a sharded communicator is supplied to a decoder
    /// that has not overridden this method, or if the concrete decode fails.
    fn forward_tensor_parallel(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        let plan = plan_from_comm(comm)?;
        if !plan.is_sharded() {
            self.forward(input_ids, offset)
        } else {
            candle_core::bail!(
                "{} does not implement tensor-parallel cached decode for world_size {}",
                std::any::type_name::<Self>(),
                plan.world_size()
            )
        }
    }

    /// Clear the decoder state so the decoder can start a fresh sequence.
    ///
    /// CONTRACT: after `reset_cache`, the next [`forward`](Self::forward) must
    /// use `offset == 0`, and a replayed sequence must reproduce the same logits
    /// as a fresh decoder (no stale state may survive the reset).
    fn reset_cache(&mut self);

    /// Optional cache-retention telemetry for the current decoder state.
    ///
    /// Implementors with windowed or otherwise memory-sensitive caches can
    /// report one snapshot per layer. The default keeps non-reporting decoders
    /// compatible and makes the instrumentation opt-in.
    fn decoder_cache_snapshots(&self, phase: &'static str) -> Vec<DecoderCacheSnapshot> {
        let _ = phase;
        Vec::new()
    }
}

/// A trainable (`LoRA`-adapted) language model: the grad-bearing update forward
/// plus the adapter/rollout plumbing the generic policy needs.
///
/// One trait, not a `GradModel`/`LoraModel` split: at two implementors a split
/// buys nothing, and every current consumer needs both halves together — the
/// adapter toggle *is* how the GRPO reference policy (adapter off == frozen
/// base) is obtained, so it is not separable from training.
pub trait GradModel {
    /// The cached decoder type [`merged_decoder`](Self::merged_decoder) builds.
    type Decoder: CachedDecoder;

    /// The device the weights live on, so a caller can build `input_ids`
    /// tensors on the same device.
    fn device(&self) -> &Device;

    /// Full-sequence logits `[batch, seq, vocab]` for `input_ids`
    /// (`[batch, seq]`, `u32`).
    ///
    /// CONTRACT: **uncached and tape-bearing** — every position is returned
    /// (the trainer scores whole completions, not just the last token), the
    /// forward records the autograd tape, and a `backward` through the returned
    /// logits must reach every [`trainable_vars`](Self::trainable_vars) entry
    /// (the grad-coverage canary enforces this at runtime). The forward must
    /// respect the current [`set_adapter_enabled`](Self::set_adapter_enabled)
    /// state: adapter off is the frozen base model (the GRPO reference policy).
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails (e.g. a shape mismatch).
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor>;

    /// All trainable [`Var`]s (the `LoRA` factors), for the optimizer, the
    /// grad-coverage canary, and checkpointing.
    ///
    /// CONTRACT: the order is **stable** across calls and across identically
    /// configured loads — it is the checkpoint contract
    /// ([`crate::checkpoint`] persists and restores the adapter positionally).
    /// The returned [`Var`]s are clones that **alias the live storage** (candle
    /// `Var` clones share the underlying tensor), so `Var::set` through them
    /// writes through to the model.
    fn trainable_vars(&self) -> Vec<Var>;

    /// Enable/disable the `LoRA` adapter everywhere.
    ///
    /// CONTRACT: disabled == the frozen base model == the GRPO reference
    /// policy (no second model copy). Affects both [`forward`](Self::forward)
    /// and any *subsequently built* [`merged_decoder`](Self::merged_decoder);
    /// an already-built decoder is a snapshot and does **not** see the flip.
    /// A model without adapters ([`has_adapters`](Self::has_adapters) ==
    /// `false`) must treat this as a no-op and keep reporting enabled.
    fn set_adapter_enabled(&mut self, enabled: bool);

    /// Whether this model carries toggleable adapters (the `LoRA` modes —
    /// the default). A **full fine-tuning** model has none: the base weights
    /// ARE the trained weights, so there is no frozen base policy to toggle
    /// back to, and [`set_adapter_enabled`](Self::set_adapter_enabled) is a
    /// no-op. Callers that depend on the toggle (the eval base-vs-trained
    /// comparison) must check this — or observe that the toggle did not
    /// take — and fail loud rather than compare a policy against itself.
    fn has_adapters(&self) -> bool {
        true
    }

    /// Snapshot the **current** effective weights into a KV-cached, grad-free
    /// [`CachedDecoder`] for fast incremental rollout.
    ///
    /// CONTRACT: the snapshot is **tape-DETACHED** (no autograd reaches the
    /// model through it) and **respects the adapter toggle** at build time —
    /// adapter on folds the live `LoRA` factors into the weights
    /// (`W + scale·B@A`), adapter off snapshots the pure base. It is a *value*
    /// snapshot: it must be **rebuilt after any optimizer step or toggle flip**,
    /// or it will sample from stale weights. Its logits must equal
    /// [`forward`](Self::forward)'s position-by-position (each implementor
    /// pins this with equivalence gates).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the merged-weight snapshot cannot be built.
    fn merged_decoder(&self) -> CandleResult<Self::Decoder>;

    /// Full-sequence logits like [`forward`](Self::forward), but **detached**:
    /// the caller wants values only (the trainer's `logp_old` / KL-reference
    /// scoring), never a backward through them.
    ///
    /// The default simply detaches a plain forward. A model that supports
    /// activation checkpointing should override it with a *rolling* detached
    /// walk (detach the hidden state at every layer boundary), which frees
    /// each layer's intermediates as the walk proceeds — same values
    /// (detaching is the identity on values), a fraction of the peak memory,
    /// and no checkpoint tape is captured.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor op fails.
    fn forward_detached(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        Ok(self.forward(input_ids)?.detach())
    }

    /// Logits for a **window** of `len` positions starting at `start` — the
    /// narrowed (memory-lean) scoring forward.
    ///
    /// The default is `forward(input_ids).narrow(1, start, len)`: identical
    /// values, no memory win. A model should override it to run the full layer
    /// stack (attention needs the whole prefix) but apply the final norm + LM
    /// head to the window alone, so the full-width `[batch, seq, vocab]`
    /// logits — usually the single biggest activation of a scoring forward —
    /// never materialize. The trainer's scoring paths only ever read the
    /// completion-predicting window; everything outside it is wasted work and
    /// peak memory.
    ///
    /// CONTRACT: same tape semantics as [`forward`](Self::forward) — tape-
    /// bearing, and under activation checkpointing it captures the boundary
    /// tape exactly like `forward` (the next [`backward`](Self::backward)
    /// consumes it). On CPU, values and trainable-var gradients must equal
    /// the default's exactly (positions outside the window contribute exact
    /// zeros through the narrow adjoint; the in-crate gates pin this). On
    /// CUDA the head gemm's row count changes, and shape-dependent kernel
    /// selection may reassociate the accumulation — values can differ at ulp
    /// level from the default's (the same accepted class as the merged-weight
    /// reassociation); training math stays self-consistent because every
    /// scoring path uses the same narrowed route. Frozen-weight gradient
    /// values may reassociate on any device (retained but never consumed).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence or any
    /// tensor op fails.
    fn forward_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        self.forward(input_ids)?.narrow(1, start, len)
    }

    /// The detached variant of [`forward_narrowed`](Self::forward_narrowed),
    /// for the value-only scorings (`logp_old`, the KL reference): same window
    /// semantics, no autograd tape. An overriding implementation must never
    /// capture a checkpoint tape; the provided default inherits
    /// [`forward_detached`](Self::forward_detached)'s softness (its own
    /// default routes through the tape-capturing `forward`), so a
    /// checkpointing model must override BOTH detached methods — as every
    /// in-crate model does.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence or any
    /// tensor op fails.
    fn forward_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
    ) -> CandleResult<Tensor> {
        self.forward_detached(input_ids)?.narrow(1, start, len)
    }

    /// Full-sequence logits through an explicitly supplied tensor-parallel
    /// communicator.
    ///
    /// The default preserves fail-closed behavior for models that have not wired
    /// tensor parallelism: single-rank communicators reuse
    /// [`forward`](Self::forward), while sharded worlds fail loud instead of
    /// silently running an unsharded forward.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a sharded communicator is supplied to a model
    /// that has not overridden this method, or if the concrete forward fails.
    fn forward_tensor_parallel(&self, input_ids: &Tensor, comm: &dyn Comm) -> CandleResult<Tensor> {
        if comm.world_size() == 1 {
            self.forward(input_ids)
        } else {
            candle_core::bail!(
                "{} does not implement tensor-parallel forward for world_size {}",
                std::any::type_name::<Self>(),
                comm.world_size()
            )
        }
    }

    /// Windowed variant of
    /// [`forward_tensor_parallel`](Self::forward_tensor_parallel).
    ///
    /// The default narrows the TP full-sequence logits, so overriding models can
    /// provide the same memory win as [`forward_narrowed`](Self::forward_narrowed)
    /// by applying the head to the scoring window alone.
    ///
    /// # Errors
    ///
    /// As [`forward_tensor_parallel`](Self::forward_tensor_parallel), plus any
    /// windowing error.
    fn forward_tensor_parallel_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        self.forward_tensor_parallel(input_ids, comm)?
            .narrow(1, start, len)
    }

    /// Detached/value-only tensor-parallel logits.
    ///
    /// The default is fail-closed for sharded communicators and reuses
    /// [`forward_detached`](Self::forward_detached) only for a single-rank
    /// communicator.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a sharded communicator is supplied to a model
    /// that has not overridden this method, or if the concrete forward fails.
    fn forward_tensor_parallel_detached(
        &self,
        input_ids: &Tensor,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        if comm.world_size() == 1 {
            self.forward_detached(input_ids)
        } else {
            candle_core::bail!(
                "{} does not implement detached tensor-parallel forward for world_size {}",
                std::any::type_name::<Self>(),
                comm.world_size()
            )
        }
    }

    /// Detached/value-only windowed TP logits.
    ///
    /// # Errors
    ///
    /// As [`forward_tensor_parallel_detached`](Self::forward_tensor_parallel_detached),
    /// plus any windowing error.
    fn forward_tensor_parallel_detached_narrowed(
        &self,
        input_ids: &Tensor,
        start: usize,
        len: usize,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor> {
        self.forward_tensor_parallel_detached(input_ids, comm)?
            .narrow(1, start, len)
    }

    /// Gather temperature-scaled log-probabilities for `targets`
    /// (`[batch, len]`) from the narrowed scoring window without requiring
    /// callers to materialize or own the chunking policy.
    ///
    /// The default preserves the existing model contract: run
    /// [`forward_narrowed`](Self::forward_narrowed) to produce
    /// `[batch, len, vocab]` logits, then chunk the F32
    /// `log_softmax`/gather stage over positions. Implementors with access
    /// to the hidden-state tail can override this to chunk the final
    /// norm/head projection too, so a full `[batch, len, vocab]` tensor never
    /// exists on the update path.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence, `targets`
    /// does not match `[batch, len]`, or any tensor op fails.
    fn token_logprobs_narrowed(
        &self,
        input_ids: &Tensor,
        targets: &Tensor,
        start: usize,
        len: usize,
        temperature: f64,
        chunk: usize,
    ) -> CandleResult<Tensor> {
        let logits = self.forward_narrowed(input_ids, start, len)?;
        chunked_logprobs_from_logits(&logits, targets, temperature, chunk)
    }

    /// Detached/value-only variant of
    /// [`token_logprobs_narrowed`](Self::token_logprobs_narrowed). It must not
    /// capture checkpoint tape.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the window exceeds the sequence, `targets`
    /// does not match `[batch, len]`, or any tensor op fails.
    fn token_logprobs_detached_narrowed(
        &self,
        input_ids: &Tensor,
        targets: &Tensor,
        start: usize,
        len: usize,
        temperature: f64,
        chunk: usize,
    ) -> CandleResult<Tensor> {
        let logits = self.forward_detached_narrowed(input_ids, start, len)?;
        Ok(chunked_logprobs_from_logits(&logits, targets, temperature, chunk)?.detach())
    }

    /// Back-propagate a loss built from this model's [`forward`](Self::forward)
    /// logits.
    ///
    /// The default is exactly `loss.backward()`. A model running with
    /// **activation checkpointing** overrides this to stitch the full gradient
    /// out of its saved boundary tape (see [`crate::remat`]): re-running each
    /// layer inside backward and folding the segment gradients into the store,
    /// so the returned [`GradStore`] covers every
    /// [`trainable_vars`](Self::trainable_vars) entry exactly as an uncut
    /// backward would (the grad-coverage canary holds either way).
    ///
    /// CONTRACT (checkpointing implementations): the loss must come from this
    /// model's **most recent** checkpointed forward, with no optimizer step or
    /// adapter toggle in between, and each forward's tape is consumed by
    /// exactly one backward — violations must fail loud, never stitch stale
    /// segments.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the backward fails, or (under checkpointing)
    /// if the tape/loss pairing is violated.
    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        loss.backward()
    }

    /// Back-propagate a loss built from
    /// [`forward_tensor_parallel`](Self::forward_tensor_parallel).
    ///
    /// The default delegates to [`backward`](Self::backward) for world-one and
    /// forward-only compatibility. It does **not** advertise sharded-training
    /// support: an uncut local tensor graph cannot express the backward
    /// all-reduces required at replicated TP activation boundaries. A model
    /// that implements those semantics must override this hook and
    /// [`supports_sharded_tensor_parallel_backward`](Self::supports_sharded_tensor_parallel_backward).
    ///
    /// # Errors
    ///
    /// Returns a candle error if backward fails or the concrete model rejects
    /// the communicator/tape pairing.
    fn backward_tensor_parallel(&self, loss: &Tensor, _comm: &dyn Comm) -> CandleResult<GradStore> {
        self.backward(loss)
    }

    /// Whether this instance implements the cross-rank backward semantics
    /// required for sharded tensor-parallel training.
    ///
    /// The default is fail-closed. Forward-value equivalence alone is not
    /// sufficient: replicated activation boundaries need rank-summed
    /// cotangents during reverse execution.
    fn supports_sharded_tensor_parallel_backward(&self) -> bool {
        false
    }

    /// The model's `LoRA` recipe as a stable canonical string (e.g.
    /// `attn:qkvo|mlp:gud|gdn:-`), recorded into checkpoint manifests so an
    /// adapter is self-describing about which projections its positional
    /// tensor list covers (see
    /// [`crate::checkpoint::CheckpointManifest::lora_recipe`]). Defaults to
    /// `None` (a model that predates recipes, or has none to report). The
    /// positional checkpoint contract does not depend on it, but
    /// [`crate::Trainer::resume`] cross-checks it against the manifest and
    /// fails loud on a mismatch (a shape-aliased recipe swap is invisible to
    /// the positional count/shape/dtype validation); a `None` on either side
    /// skips that check.
    fn lora_recipe(&self) -> Option<String> {
        None
    }
}

/// Gather target log-probabilities from already-windowed logits
/// `[batch, len, vocab]`, chunking the F32 `log_softmax` over positions.
///
/// Chunking is exact because `log_softmax` reduces only over the vocab axis
/// and the gather is positionwise. The helper is shared by the generic model
/// default, [`crate::lm_policy::LmPolicy`] tests, and model overrides that
/// chunk their final head projection before calling into the same math.
///
/// # Errors
///
/// Returns a candle error if `targets` is not `[batch, len]`, if `len == 0`,
/// or if any tensor op fails.
pub(crate) fn chunked_logprobs_from_logits(
    logits: &Tensor,
    targets: &Tensor,
    temperature: f64,
    chunk: usize,
) -> CandleResult<Tensor> {
    let (b, len, _vocab) = logits.dims3()?;
    if len == 0 {
        candle_core::bail!("token_logprobs: zero-length scoring window");
    }
    let (tb, tw) = targets.dims2()?;
    if tb != b || tw != len {
        candle_core::bail!(
            "token_logprobs: targets must be [{b}, {len}], got {:?}",
            targets.dims()
        );
    }

    let idx = targets.unsqueeze(D::Minus1)?;
    let chunk = chunk.max(1);
    let mut parts = Vec::with_capacity(len.div_ceil(chunk));
    let mut pos = 0;
    while pos < len {
        let n = chunk.min(len - pos);
        let mut p = logits.narrow(1, pos, n)?.to_dtype(DType::F32)?;
        if (temperature - 1.0).abs() > f64::EPSILON {
            p = (p / temperature)?;
        }
        let logp = log_softmax(&p, D::Minus1)?;
        let part = logp
            .gather(&idx.narrow(1, pos, n)?.contiguous()?, D::Minus1)?
            .squeeze(D::Minus1)?;
        parts.push(part);
        pos += n;
    }
    Tensor::cat(&parts, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// A minimal [`GradModel`] that relies on every provided default — the
    /// witness that an external implementor gets working behavior for free:
    /// `forward_detached` is a value-identical detach and `backward` is
    /// exactly `loss.backward()`.
    struct OneVarModel {
        w: Var,
        device: Device,
    }

    struct NoopDecoder;
    impl CachedDecoder for NoopDecoder {
        fn forward(&mut self, _input_ids: &Tensor, _offset: usize) -> CandleResult<Tensor> {
            candle_core::bail!("the default-behavior test never decodes")
        }
        fn reset_cache(&mut self) {}
    }

    impl GradModel for OneVarModel {
        type Decoder = NoopDecoder;
        fn device(&self) -> &Device {
            &self.device
        }
        fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
            let x = input_ids.to_dtype(DType::F32)?.unsqueeze(2)?;
            x.broadcast_mul(self.w.as_tensor())
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.w.clone()]
        }
        fn set_adapter_enabled(&mut self, _enabled: bool) {}
        fn merged_decoder(&self) -> CandleResult<NoopDecoder> {
            Ok(NoopDecoder)
        }
    }

    #[test]
    fn the_provided_defaults_detach_values_and_backward_plainly() {
        let device = Device::Cpu;
        let w = Var::from_tensor(&Tensor::from_vec(vec![2.0f32], (1,), &device).unwrap()).unwrap();
        let m = OneVarModel {
            w: w.clone(),
            device,
        };
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (2, 2), m.device()).unwrap();

        // forward_detached: identical values, tape-free.
        let live = m.forward(&ids).unwrap();
        let det = m.forward_detached(&ids).unwrap();
        assert_eq!(
            det.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            live.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let store = det.sum_all().unwrap().backward().unwrap();
        assert!(store.get(&w).is_none(), "detached logits reached the var");

        // backward: exactly loss.backward() — the var's grad is in the store.
        let loss = live.sum_all().unwrap();
        let grads = m.backward(&loss).unwrap();
        assert!(
            grads.get(&w).is_some(),
            "default backward lost the var grad"
        );
        assert!(m.lora_recipe().is_none());
    }

    #[test]
    fn the_provided_narrowed_defaults_are_forward_plus_narrow() {
        let device = Device::Cpu;
        let w = Var::from_tensor(&Tensor::from_vec(vec![2.0f32], (1,), &device).unwrap()).unwrap();
        let m = OneVarModel {
            w: w.clone(),
            device,
        };
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (2, 3), m.device()).unwrap();

        let reference = m.forward(&ids).unwrap().narrow(1, 1, 2).unwrap();
        let narrowed = m.forward_narrowed(&ids, 1, 2).unwrap();
        assert_eq!(narrowed.dims(), reference.dims());
        assert_eq!(
            narrowed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            reference.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        // The narrowed default is tape-bearing (it routes through `forward`)…
        let grads = m
            .backward(&narrowed.sum_all().unwrap())
            .expect("backward through the narrowed default");
        assert!(grads.get(&w).is_some(), "narrowed default lost the tape");

        // …and the detached variant is value-identical but tape-free.
        let det = m.forward_detached_narrowed(&ids, 1, 2).unwrap();
        assert_eq!(
            det.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            reference.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let store = det.sum_all().unwrap().backward().unwrap();
        assert!(store.get(&w).is_none(), "detached narrowed reached the var");

        // An out-of-range window fails loud instead of clamping.
        assert!(m.forward_narrowed(&ids, 2, 2).is_err());
    }
}
