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

use candle_core::{Device, Result as CandleResult, Tensor, Var};

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

    /// Clear the decoder state so the decoder can start a fresh sequence.
    ///
    /// CONTRACT: after `reset_cache`, the next [`forward`](Self::forward) must
    /// use `offset == 0`, and a replayed sequence must reproduce the same logits
    /// as a fresh decoder (no stale state may survive the reset).
    fn reset_cache(&mut self);
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
    fn set_adapter_enabled(&mut self, enabled: bool);

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
}
