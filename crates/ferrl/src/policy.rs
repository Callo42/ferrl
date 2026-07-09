//! The policy abstraction.
//!
//! A [`Policy`] is the trainable language model viewed through the narrow
//! interface GRPO needs: sample completions, score tokens under the current
//! parameters, and toggle the `LoRA` adapter so the same weights can serve as both
//! the trainable policy (adapter on) and the frozen reference (adapter off).
//!
//! The concrete `candle-transformers` model implementation lives behind this
//! trait so the trainer is model-agnostic and so the pure-math layer never
//! depends on a particular architecture. Generation returns token ids; log-prob
//! scoring returns a [`candle_core::Tensor`] (it lives on the autograd tape so
//! the surrogate can be differentiated), whereas rewards stay scalar (see
//! [`crate::reward`]).

use candle_core::backprop::GradStore;
use candle_core::{Result as CandleResult, Tensor, Var};

use crate::comm::Comm;
use crate::telemetry::ModelTelemetryRecorder;

/// A batch of sampled completions: token ids plus the prompt length that
/// produced them, so callers can slice prompt from completion.
#[derive(Debug, Clone)]
pub struct Rollout {
    /// Token ids for each sequence, prompt tokens followed by completion tokens.
    pub token_ids: Vec<Vec<u32>>,
    /// Number of leading prompt tokens shared by every sequence in this rollout.
    pub prompt_len: usize,
    /// Number of *real* completion tokens in each sequence — the count of
    /// generated tokens up to and including the first EOS (EOS-inclusive), or the
    /// full completion width if no EOS was sampled. Positions at or beyond this
    /// index within the (rectangular) completion are padding and are masked out of
    /// the GRPO loss. For a fixed-length, no-early-stop rollout this equals the
    /// full completion width for every sequence (see [`Rollout::rectangular`]); a
    /// per-element value is in `0..=comp_len`.
    pub completion_lens: Vec<usize>,
    /// Per-token **behavior-policy** log-probabilities of the sampled completion
    /// tokens — the probability each token was *actually drawn with* (the rollout
    /// path's logits at the sampling temperature, nucleus-renormalized when top-p
    /// is active), captured by the sampler at draw time. Row `i` has exactly
    /// [`completion_lens`](Self::completion_lens)`[i]` entries (one per real
    /// draw; EOS padding was never sampled, so it carries no log-prob).
    ///
    /// `None` when the policy does not capture them (toy/test policies;
    /// [`Rollout::rectangular`] always sets `None`; capturing policies construct
    /// via [`Rollout::new`]). When present, the trainer compares them against
    /// its own scoring forward to surface the rollout-vs-train mismatch (the
    /// off-policy gap a cached/merged — possibly bf16 — decode path opens
    /// against the f32 uncached scoring forward) as ratio telemetry, and
    /// optionally corrects the surrogate with truncated importance sampling
    /// (TIS).
    ///
    /// Precision note: the captured value is `ln(p / Σp)` over the sampler's
    /// f32 probabilities — the exact distribution `WeightedIndex` drew from —
    /// which differs from an f32 `log_softmax` recomputation by the rounding of
    /// the probability sum (~1e-5 over a real vocab). Don't over-interpret
    /// sub-1e-5 ratio readings.
    pub rollout_logprobs: Option<Vec<Vec<f32>>>,
}

impl Rollout {
    /// Construct a **rectangular** rollout in which every sequence is a real
    /// completion of the full width — the legacy, no-EOS-early-stop behavior.
    ///
    /// Each `completion_lens[i]` is `token_ids[i].len() - prompt_len` (saturating,
    /// so a degenerate `prompt_len >= row length` yields `0` rather than
    /// panicking). EOS-aware generation, which stops sequences early, instead sets
    /// `completion_lens` to the true per-sequence lengths and right-pads
    /// `token_ids` to a common width.
    #[must_use]
    pub fn rectangular(token_ids: Vec<Vec<u32>>, prompt_len: usize) -> Self {
        let completion_lens = token_ids
            .iter()
            .map(|ids| ids.len().saturating_sub(prompt_len))
            .collect();
        Self {
            token_ids,
            prompt_len,
            completion_lens,
            rollout_logprobs: None,
        }
    }

    /// Construct a rollout from all of its parts — the constructor for policies
    /// that capture behavior log-probs (or stop sequences early), so an external
    /// [`Policy`] implementor never needs a struct literal and a future field
    /// addition is not automatically a source break for it. `rollout_logprobs`,
    /// when `Some`, must carry one row per sequence with exactly
    /// `completion_lens[i]` entries in row `i` (the trainer validates this and
    /// fails loud on a mismatch).
    #[must_use]
    pub fn new(
        token_ids: Vec<Vec<u32>>,
        prompt_len: usize,
        completion_lens: Vec<usize>,
        rollout_logprobs: Option<Vec<Vec<f32>>>,
    ) -> Self {
        Self {
            token_ids,
            prompt_len,
            completion_lens,
            rollout_logprobs,
        }
    }

    /// Number of sequences in the rollout.
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Whether the rollout contains no sequences.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// **Eval-only** sampling parameters, the deliberate override channel for
/// [`GenConfig::eval_sampling`].
///
/// Training rollouts must sample at the policy's own (baked, scoring-consistent)
/// temperature — [`crate::LmPolicy`] fails loud on a mismatched
/// [`GenConfig::temperature`] precisely to catch a drifted config. Held-out
/// evaluation legitimately wants *different* sampling (the 2026 convention:
/// temperature ≈ 0.6 with nucleus top-p 0.95 for avg@k sampled pass@1), so the
/// override is a separate, explicit field rather than a relaxation of that
/// check: setting it says "I know this is not the training distribution."
/// For [`crate::LmPolicy`] (and any conforming [`Policy`] — see the
/// [`Policy::generate`] contract), nucleus (top-p) filtering can **only**
/// enter generation through here — training rollouts structurally stay
/// untruncated. A policy that ignores sampling parameters samples its own
/// distribution regardless; eval reports over such a policy reflect that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalSampling {
    /// Eval softmax temperature (must be finite and `> 0`).
    pub temperature: f64,
    /// Optional nucleus filter: keep the smallest top-probability set whose
    /// cumulative mass reaches this value (in `(0, 1]`; the crossing token is
    /// included), renormalize, and sample from it. `None` disables filtering.
    pub top_p: Option<f64>,
}

impl Default for EvalSampling {
    /// The 2026 eval convention: temperature `0.6`, nucleus top-p `0.95`.
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: Some(0.95),
        }
    }
}

/// Sampling controls for [`Policy::generate`].
#[derive(Debug, Clone, Copy)]
pub struct GenConfig {
    /// Number of completions to sample per prompt (the GRPO group size).
    pub group_size: usize,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Softmax temperature; `1.0` is unscaled. Ignored when
    /// [`eval_sampling`](Self::eval_sampling) is set (the override carries its
    /// own temperature).
    pub temperature: f64,
    /// End-of-sequence token. When `Some(id)`, a sampled `id` ends that sequence's
    /// completion early (the EOS token is **kept** — the recorded length is
    /// EOS-*inclusive*) and the sequence is right-padded back to `max_new_tokens`
    /// so the group stays rectangular at a fixed width; [`Rollout::completion_lens`]
    /// records the true per-sequence length. When `None` (the default) no sequence
    /// stops early, every completion is the full `max_new_tokens`, and the rollout
    /// is bit-identical to the legacy no-early-stop behavior. A [`Policy`] backed by
    /// a model with no EOS token (e.g. a base model) leaves this `None`.
    pub eos_token_id: Option<u32>,
    /// Eval-only sampling override (see [`EvalSampling`]). `None` (the default,
    /// and what the trainer always passes) samples at
    /// [`temperature`](Self::temperature) with no nucleus filtering, exactly the
    /// legacy behavior; `Some` deliberately samples the eval distribution
    /// instead. The held-out eval harness is the intended setter.
    pub eval_sampling: Option<EvalSampling>,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            max_new_tokens: 256,
            temperature: 1.0,
            eos_token_id: None,
            eval_sampling: None,
        }
    }
}

/// The trainable policy GRPO drives.
///
/// # The three log-probability roles
///
/// A GRPO update needs three per-token log-probabilities of the *same* sampled
/// completions. All three are obtained from this one trait — no extra method is
/// required, so adding the inner optimization loop (`μ > 1`) later is **not** a
/// breaking change:
///
/// - **current** `logp` — [`token_logprobs`](Policy::token_logprobs) with the
///   adapter [enabled](Policy::set_adapter_enabled). Lives on the autograd tape;
///   it is the numerator of the importance ratio and what `backward` flows
///   through.
/// - **old** `logp_old` — a *frozen snapshot* of the current log-probs, taken
///   once when the rollout is generated (adapter enabled) and then **detached**
///   from the tape and stored by the trainer. It is the ratio denominator in
///   `exp(logp - logp_old)`. With a single inner step (`μ = 1`) it equals `logp`
///   and the ratio is exactly `1`. The trait deliberately does *not* own this
///   snapshot, keeping the policy stateless with respect to the optimizer step.
/// - **reference** `logp_ref` — [`token_logprobs`](Policy::token_logprobs) with
///   the adapter [disabled](Policy::set_adapter_enabled): the frozen base model.
///   Feeds the k3 KL ([`crate::grpo::k3_kl`]) and is likewise detached (the
///   reference is never trained).
///
/// The adapter toggle is what lets one set of weights serve all three: the same
/// parameters are the policy (adapter on) and their own reference (adapter off);
/// "old" is just an adapter-on snapshot frozen in time.
pub trait Policy {
    /// Sample `cfg.group_size` completions for `prompt` under `cfg`.
    ///
    /// Implementations that sample SHOULD honor
    /// [`cfg.eval_sampling`](GenConfig::eval_sampling) (sample the override's
    /// temperature/top-p) or return an error — silently ignoring it makes an
    /// eval caller believe the eval convention applied when it did not.
    /// [`crate::LmPolicy`] honors it; a toy/test policy that ignores sampling
    /// parameters entirely is consistent with this (it ignores `temperature`
    /// too — its "distribution" has no knobs to override). Policies that can
    /// cheaply report per-draw probabilities SHOULD also fill
    /// [`Rollout::rollout_logprobs`]; `None` is always valid.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the forward pass or sampling fails.
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout>;

    /// Like [`generate`](Self::generate), but seeds the rollout's per-row RNG
    /// substreams from an explicit **global row base** — the index of this
    /// prompt's first completion in the *flattened, world-global* rollout stream.
    /// A sampling policy seeds row `i` of the group from `global_row_base + i`,
    /// so the sampled tokens are invariant to how the global batch is sharded
    /// across a data-parallel world (a world-W run reproduces the single-process
    /// draws) and are recomputable on resume.
    ///
    /// The default **ignores** `global_row_base` and calls
    /// [`generate`](Self::generate) — correct for a deterministic / non-sampling
    /// policy, whose rollout does not depend on RNG. A policy that samples
    /// (e.g. [`crate::LmPolicy`]) overrides this to thread the base into its
    /// sampler, and its [`generate`](Self::generate) becomes the
    /// `global_row_base = 0` convenience.
    ///
    /// **Wrapper policies:** a policy that delegates [`generate`](Self::generate)
    /// to an inner policy MUST override this to delegate `generate_at` too —
    /// inheriting the default would route through the wrapper's own `generate`
    /// and silently drop the base (every prompt seeded from 0), exactly the
    /// world-size-variance the base removes.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the forward pass or sampling fails.
    fn generate_at(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
    ) -> CandleResult<Rollout> {
        let _ = global_row_base;
        self.generate(prompt, cfg)
    }

    /// Like [`generate_at`](Self::generate_at), with an optional telemetry sink
    /// for model-path phase boundaries and decoder-cache snapshots.
    ///
    /// The default delegates to [`generate_at`](Self::generate_at), so existing
    /// policies remain valid. Model-backed policies can override this to expose
    /// prefill/decode/cache evidence without changing rollout semantics.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the forward pass or sampling fails.
    fn generate_at_instrumented(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout> {
        let _ = telemetry;
        self.generate_at(prompt, cfg, global_row_base)
    }

    /// Per-token log-probabilities of `rollout`'s completion tokens under the
    /// current parameters, as a differentiable [`Tensor`].
    ///
    /// The returned tensor has shape `[num_sequences, completion_len]` and is
    /// connected to the trainable [`candle_core::Var`]s so the GRPO surrogate
    /// built from it can be back-propagated.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the forward pass fails.
    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor>;

    /// [`token_logprobs`](Self::token_logprobs), but **detached** — for the
    /// value-only scorings (the `logp_old` snapshot and the KL reference)
    /// that are never back-propagated.
    ///
    /// The default detaches a plain `token_logprobs` call — identical values,
    /// identical behavior for every existing policy. A model-backed policy
    /// overrides it to route through a memory-light detached forward
    /// ([`crate::GradModel::forward_detached`]), which never retains the full
    /// activation graph and — under activation checkpointing — never captures
    /// a boundary tape (a value scoring must not disturb the tape the next
    /// update backward will consume).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the forward pass fails.
    fn token_logprobs_detached(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        Ok(self.token_logprobs(rollout)?.detach())
    }

    /// Back-propagate a loss built from this policy's
    /// [`token_logprobs`](Self::token_logprobs).
    ///
    /// The default is exactly `loss.backward()` — what the trainer always did.
    /// A model-backed policy forwards to [`crate::GradModel::backward`], which
    /// under **activation checkpointing** stitches the full gradient out of
    /// the saved boundary tape instead (see [`crate::remat`]); the returned
    /// store covers every [`trainable_vars`](Self::trainable_vars) entry
    /// either way (the trainer's grad-coverage canary enforces it).
    ///
    /// **Wrapper policies:** a policy that delegates `token_logprobs` to an
    /// inner policy MUST delegate `backward` (and
    /// [`token_logprobs_detached`](Self::token_logprobs_detached)) too. With a
    /// checkpointing inner policy, inheriting this default would run a plain
    /// `loss.backward()` over the cut tape — every layer var absent. The
    /// trainer's grad-coverage canary aborts that run (loud, not silent), but
    /// the failure reads as "var absent from the grad store", not as the
    /// missing delegation it actually is.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the backward fails, or (under checkpointing)
    /// if the loss does not pair with the most recent checkpointed forward.
    fn backward(&self, loss: &Tensor) -> CandleResult<GradStore> {
        loss.backward()
    }

    /// Enable (`true`) or disable (`false`) the `LoRA` adapter contribution.
    ///
    /// With the adapter disabled the policy computes the frozen base-model
    /// distribution, i.e. the GRPO reference policy.
    fn set_adapter_enabled(&mut self, enabled: bool);

    /// Whether the `LoRA` adapter is currently contributing to the forward pass.
    fn adapter_enabled(&self) -> bool;

    /// The trainable parameters the optimizer updates and the grad-coverage
    /// canary checks each step (e.g. the `LoRA` `A`/`B` factors).
    ///
    /// A cloned [`Var`] shares its inner storage (and tensor id) with the
    /// original, so the returned vars *alias* the parameters used inside
    /// [`token_logprobs`](Policy::token_logprobs): the trainer registers exactly
    /// these with the optimizer and looks them up in the grad store after
    /// `backward`. Implementors typically forward to their adapter's
    /// `trainable_vars()`.
    fn trainable_vars(&self) -> Vec<Var>;

    /// Serialize the policy's rollout-sampler RNG state to an opaque byte blob, for
    /// momentum-faithful checkpoint persistence.
    ///
    /// A faithful resume must continue the *same* rollout token stream an uninterrupted
    /// run would have produced, which requires capturing the sampler's RNG state.
    /// candle's `LogitsProcessor` hides its `StdRng` behind no accessor — which is why
    /// ferrl owns [`GrpoSampler`](crate::sampler::GrpoSampler), whose state *is*
    /// `serde`-serializable. The returned blob is opaque to the checkpoint; only
    /// [`restore_sampler_state`](Self::restore_sampler_state) interprets it. A policy
    /// with no rollout RNG returns an empty blob.
    ///
    /// This is a **required** method (not defaulted) so that giving a policy a sampler
    /// can never silently skip RNG capture — the resume footgun a faithful checkpoint
    /// must avoid.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the sampler state cannot be serialized.
    fn sampler_state(&self) -> CandleResult<Vec<u8>>;

    /// Restore the rollout-sampler RNG state from a blob produced by
    /// [`sampler_state`](Self::sampler_state), so a resumed run continues the exact
    /// token stream. **Fails loud** if the blob is malformed or does not match this
    /// policy's sampler, rather than silently re-seeding.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `state` is not a valid blob for this policy's sampler.
    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()>;

    /// The policy's `LoRA` recipe as a stable canonical string, recorded into
    /// checkpoint manifests (see
    /// [`crate::checkpoint::CheckpointManifest::lora_recipe`]). Informational —
    /// the checkpoint load contract stays positional. Defaulted (`None`) so toy
    /// / test policies need not implement it; model-backed policies forward
    /// their [`crate::GradModel::lora_recipe`].
    fn lora_recipe(&self) -> Option<String> {
        None
    }
}

/// Explicit tensor-parallel execution hooks for policies that can route GRPO
/// rollout and scoring through a caller-supplied communicator.
///
/// This is intentionally separate from [`Policy`]: loader and CLI paths remain
/// fail-closed for sharded tensor parallelism. Explicit trainer
/// tensor-parallel entry points require callers to provide a tensor-parallel
/// [`Comm`] and support sharded TP only without simultaneous sharded data
/// parallelism: trainable adapter vars are fully replicated, gradients are
/// sum-reduced over the TP communicator before the optimizer step, and TP rank
/// 0 owns shared checkpoints.
pub trait TensorParallelPolicy: Policy {
    /// Generate a rollout through the policy's tensor-parallel rollout path,
    /// using `comm` as the tensor-parallel communicator.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the tensor-parallel plan is unsupported,
    /// rollout generation fails, or telemetry recording touches a failing model
    /// path.
    fn generate_at_tensor_parallel_instrumented(
        &mut self,
        prompt: &[u32],
        cfg: &GenConfig,
        global_row_base: u64,
        comm: &dyn Comm,
        telemetry: Option<&mut dyn ModelTelemetryRecorder>,
    ) -> CandleResult<Rollout>;

    /// Per-token log-probabilities through the policy's tensor-parallel scoring
    /// path, as a differentiable tensor.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the tensor-parallel plan is unsupported or the
    /// scoring forward fails.
    fn token_logprobs_tensor_parallel(
        &self,
        rollout: &Rollout,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor>;

    /// Detached/value-only variant of
    /// [`token_logprobs_tensor_parallel`](Self::token_logprobs_tensor_parallel).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the tensor-parallel plan is unsupported or the
    /// scoring forward fails.
    fn token_logprobs_tensor_parallel_detached(
        &self,
        rollout: &Rollout,
        comm: &dyn Comm,
    ) -> CandleResult<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_len_and_empty() {
        let r = Rollout::rectangular(vec![vec![1, 2, 3], vec![1, 4, 5]], 1);
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());

        let e = Rollout::rectangular(vec![], 0);
        assert_eq!(e.len(), 0);
        assert!(e.is_empty());
    }

    #[test]
    fn rectangular_fills_completion_lens_to_full_width() {
        // Every sequence is a full-width real completion: prompt_len 2, comp 3.
        let r = Rollout::rectangular(vec![vec![1, 2, 3, 4, 5], vec![1, 2, 6, 7, 8]], 2);
        assert_eq!(r.completion_lens, vec![3, 3]);
        // An empty rollout has no per-sequence lengths.
        let e = Rollout::rectangular(vec![], 0);
        assert!(e.completion_lens.is_empty());
        // Saturating: a row no longer than the prompt yields a zero completion.
        let z = Rollout::rectangular(vec![vec![1, 2]], 2);
        assert_eq!(z.completion_lens, vec![0]);
    }

    #[test]
    fn gen_config_default_group_size() {
        let c = GenConfig::default();
        assert_eq!(c.group_size, 8);
        assert_eq!(c.max_new_tokens, 256);
        assert_eq!(c.temperature, 1.0);
        // The default disables EOS early-stop, preserving legacy full-width rollouts.
        assert_eq!(c.eos_token_id, None);
        // No eval override by default: training-path sampling, no nucleus filter.
        assert_eq!(c.eval_sampling, None);
    }

    #[test]
    fn eval_sampling_default_is_the_2026_convention() {
        let e = EvalSampling::default();
        assert_eq!(e.temperature, 0.6);
        assert_eq!(e.top_p, Some(0.95));
    }

    #[test]
    fn rectangular_captures_no_rollout_logprobs() {
        // The convenience constructor is the no-capture path by contract: toy
        // policies built through it must never claim behavior log-probs.
        let r = Rollout::rectangular(vec![vec![1, 2, 3]], 1);
        assert_eq!(r.rollout_logprobs, None);
    }

    #[test]
    fn new_carries_every_part_verbatim() {
        // The full-args constructor for capturing implementors: everything
        // lands as passed, no recomputation.
        let r = Rollout::new(
            vec![vec![1, 2, 3, 4], vec![1, 2, 5, 5]],
            2,
            vec![2, 1],
            Some(vec![vec![-0.5, -1.0], vec![-0.25]]),
        );
        assert_eq!(r.prompt_len, 2);
        assert_eq!(r.completion_lens, vec![2, 1]);
        assert_eq!(
            r.rollout_logprobs,
            Some(vec![vec![-0.5, -1.0], vec![-0.25]])
        );
    }
}
