//! Data-parallel equivalence gates (PR-F, P8 CPU-side).
//!
//! The DP correctness story has two halves, and the gates split accordingly:
//!
//! 1. **Sharded ≡ single, within a measured envelope.** A world-W run at
//!    per-rank accumulation `a` consumes exactly the prompts a single-rank run
//!    at accumulation `a·W` consumes, and folds the same per-item gradients —
//!    only the **summation association** differs (per-shard then cross-rank,
//!    vs one linear fold), so f32 non-associativity makes bit-exactness
//!    impossible *by construction* and the gate is a measured reassociation
//!    envelope (the `frozen_linear` / P6-C class). The policies here are
//!    **deterministic** (scripted rollouts that are a pure function of the
//!    prompt, real `LoRA` gradients) — a deliberate simplification that isolates
//!    the envelope to gradient arithmetic alone. (A *stochastic* sampler is now
//!    shard-invariant too: substreams seed from the **global row index**, so a
//!    world-W rollout reproduces the single-process draws — gated directly in
//!    `sampler` / `lm_policy`. These gradient gates keep deterministic rollouts
//!    only to keep the arithmetic the sole variable.)
//! 2. **Ranks ≡ each other, bitwise.** Whatever the rollouts, every rank steps
//!    from the identical reduced gradient, so the world's weights stay in
//!    bitwise lockstep — gated on the REAL tiny-qwen3.5 fixture in BOTH
//!    training modes (`LoRA` and full fine-tuning; their var sets and grad
//!    paths differ end-to-end), with stochastic per-rank rollouts.
//!
//! Plus the structural gates: world-1 `LocalComm` ≡ `SoloComm` ≡ the legacy
//! constructor (bitwise — the DP plumbing must not perturb the single-rank
//! path); an all-degenerate **local** shard neither deadlocks nor diverges
//! (the rank participates with zeros — the live-count decision is global);
//! an all-degenerate **global** window is skipped by every rank in lockstep;
//! and checkpointing is rank-0-only, with a DP resume continuing bit-exactly
//! (deterministic policies here; and under global-index seeding the rank-0
//! sampler-blob is anyway sufficient — a stochastic resume re-derives every
//! rank's per-row draws from the restored run seed and the recomputed global
//! index, with no per-rank RNG state to capture). A manual ignored
//! `--features nccl` smoke at the bottom runs the same tiny qwen3.5 path over
//! real CUDA/NCCL ranks and prints memory/timing fields for resource regression
//! checks.

use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::ops::log_softmax;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use ferrl::lora::LoraLinear;
use ferrl::nn::RmsNorm;
use ferrl::policy::{GenConfig, Policy, Rollout};
use ferrl::telemetry::RunDir;
use ferrl::trainer::{TokenizerLike, Trainer, TrainerConfig, TrainerError};
use ferrl::{
    tensors_from_pretrained, varbuilder_from_pretrained, Comm, CommError, LocalComm, LossType,
    Metrics, OptimizerState, Qwen3_5Config, Qwen3_5GradModel, Qwen3_5Policy, RewardError, RewardFn,
    RolloutLedgerError, Sample, SoloComm,
};

const VOCAB: usize = 5;
const SEED: u64 = 11;

// ---- the deterministic policy ----------------------------------------------

/// A one-layer `LoRA` LM (the toy-echo scaffold) whose `generate` is
/// **scripted**: completion `g` of a prompt echoes the prompt's first symbol
/// when `g` is even and emits the next symbol when odd — a pure function of
/// the prompt, identical under any sharding, while `token_logprobs` stays the
/// real differentiable forward (one-hot → `LoraLinear` → `RmsNorm` →
/// `log_softmax`), so gradients and optimizer steps are the real thing.
struct ScriptedPolicy {
    lora: LoraLinear,
    norm: RmsNorm,
    device: Device,
}

impl ScriptedPolicy {
    fn new(seed: u64) -> CandleResult<Self> {
        let device = Device::Cpu;
        let base = Tensor::zeros((VOCAB, VOCAB), DType::F32, &device)?;
        let lora = LoraLinear::new(base, None, VOCAB, VOCAB as f64)?;
        // Deterministic adapter init (LoraLinear's A comes from OS-entropy
        // randn on CPU): a seeded hash fill at the same ~0.02 scale, B zero —
        // every rank and every run constructs the identical policy.
        let a = &lora.trainable_vars()[0];
        let (r, c) = a.as_tensor().dims2()?;
        let fill: Vec<f32> = (0..r * c)
            .map(|i| {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                ((z % 1000) as f32 / 1000.0 - 0.5) * 0.04
            })
            .collect();
        a.set(&Tensor::from_vec(fill, (r, c), &device)?)?;
        let gamma = Tensor::ones(VOCAB, DType::F32, &device)?.affine(3.0, 0.0)?;
        let norm = RmsNorm::new(gamma, 1e-6);
        Ok(Self { lora, norm, device })
    }
}

impl Policy for ScriptedPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let first = prompt[0];
        let token_ids = (0..cfg.group_size)
            .map(|g| {
                let symbol = if g % 2 == 0 {
                    first
                } else {
                    (first + 1) % VOCAB as u32
                };
                let mut ids = prompt.to_vec();
                ids.extend(std::iter::repeat_n(symbol, cfg.max_new_tokens));
                ids
            })
            .collect();
        Ok(Rollout::rectangular(token_ids, prompt.len()))
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let prompt_len = rollout.prompt_len;
        let seq_len = rollout.token_ids[0].len();
        let comp_len = seq_len - prompt_len;
        let input_len = seq_len - 1;
        let g = rollout.token_ids.len();
        let mut oh = vec![0f32; g * input_len * VOCAB];
        let mut targets = vec![0u32; g * comp_len];
        for (i, ids) in rollout.token_ids.iter().enumerate() {
            for t in 0..input_len {
                oh[(i * input_len + t) * VOCAB + ids[t] as usize] = 1.0;
            }
            for j in 0..comp_len {
                targets[i * comp_len + j] = ids[prompt_len + j];
            }
        }
        let oh = Tensor::from_vec(oh, (g, input_len, VOCAB), &self.device)?;
        let h = self.lora.forward(&oh)?;
        let logits = self.norm.forward(&h)?;
        let pred = logits.narrow(1, prompt_len - 1, comp_len)?;
        let logp = log_softmax(&pred, D::Minus1)?;
        let idx = Tensor::from_vec(targets, (g, comp_len), &self.device)?.unsqueeze(D::Minus1)?;
        logp.gather(&idx, D::Minus1)?.squeeze(D::Minus1)
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.lora.set_enabled(enabled);
    }

    fn adapter_enabled(&self) -> bool {
        self.lora.is_enabled()
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.lora.trainable_vars()
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        // Generation is scripted — there is no RNG to capture, which is what
        // makes the DP resume gate bit-exact (see the module docs).
        Ok(Vec::new())
    }

    fn restore_sampler_state(&mut self, _state: &[u8]) -> CandleResult<()> {
        Ok(())
    }
}

/// A forwarding policy with the exact same live trainable tensors as
/// [`ScriptedPolicy`] but a different declared adapter recipe. Ledger identity
/// must bind this semantic schema, not only tensor values and shapes.
struct RecipeScriptedPolicy {
    inner: ScriptedPolicy,
}

impl Policy for RecipeScriptedPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
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
        Some("test:alternate-scripted-recipe".to_string())
    }
}

/// Replaces its trainable variables with byte-identical fresh variables after
/// the first rollout. A post-collection check that only rehashes previously
/// captured `Var` handles cannot observe this policy-state replacement.
struct ReplacingGeneratePolicy {
    inner: ScriptedPolicy,
    seed: u64,
    replaced: bool,
}

impl Policy for ReplacingGeneratePolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        let rollout = self.inner.generate(prompt, cfg)?;
        if !self.replaced {
            self.inner = ScriptedPolicy::new(self.seed)?;
            self.replaced = true;
        }
        Ok(rollout)
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
}

/// `'a'..` to ids `0..` (the toy-echo codec).
struct CharTokenizer;
impl TokenizerLike for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| u32::from(c) - u32::from('a'))
            .collect()
    }
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&i| char::from_u32(u32::from('a') + i))
            .collect()
    }
}

/// Echo reward (spread within every scripted group: even members echo and
/// score 1, odd members don't and score 0) — except prompts starting with
/// `'e'`, which score a CONSTANT: their groups are degenerate (all-equal
/// rewards, zero advantages), the lever the degenerate-shard gates pull.
struct EchoOrFlatReward;
impl RewardFn for EchoOrFlatReward {
    type Target = ();
    fn reward(&self, sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
        let prompt = sample.prompt.as_str();
        if prompt.starts_with('e') {
            return Ok(0.5);
        }
        Ok(match (prompt.chars().next(), completion.chars().next()) {
            (Some(p), Some(c)) if p == c => 1.0,
            _ => 0.0,
        })
    }
}

/// [`ScriptedPolicy`] plus one extra trainable var that reaches the loss only
/// when `wired` — mimicking, e.g., a never-routed expert on one rank. The var
/// COUNT matches across ranks (the collective contract holds; the unwired
/// rank's reduce slot carries zeros) but the unwired rank's backward never
/// covers it, which must abort the whole world in lockstep.
struct GatedExtraPolicy {
    inner: ScriptedPolicy,
    extra: Var,
    wired: bool,
}

impl GatedExtraPolicy {
    fn new(seed: u64, wired: bool) -> Self {
        Self {
            inner: ScriptedPolicy::new(seed).unwrap(),
            extra: Var::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            wired,
        }
    }
}

impl Policy for GatedExtraPolicy {
    fn generate(&mut self, prompt: &[u32], cfg: &GenConfig) -> CandleResult<Rollout> {
        self.inner.generate(prompt, cfg)
    }

    fn token_logprobs(&self, rollout: &Rollout) -> CandleResult<Tensor> {
        let logp = self.inner.token_logprobs(rollout)?;
        if self.wired {
            // A zero-valued additive touch: the scores are unchanged (the
            // ranks' losses stay identical) but the var joins the graph, so
            // only the wired rank's backward covers it.
            logp.broadcast_add(self.extra.as_tensor())
        } else {
            Ok(logp)
        }
    }

    fn set_adapter_enabled(&mut self, enabled: bool) {
        self.inner.set_adapter_enabled(enabled);
    }

    fn adapter_enabled(&self) -> bool {
        self.inner.adapter_enabled()
    }

    fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = self.inner.trainable_vars();
        vars.push(self.extra.clone());
        vars
    }

    fn sampler_state(&self) -> CandleResult<Vec<u8>> {
        self.inner.sampler_state()
    }

    fn restore_sampler_state(&mut self, state: &[u8]) -> CandleResult<()> {
        self.inner.restore_sampler_state(state)
    }
}

/// A world-1 spy counting collective invocations — the oracle for the
/// "world-1 issues no collectives" guard discipline.
#[derive(Debug)]
struct SpyComm {
    calls: Arc<AtomicUsize>,
}

impl Comm for SpyComm {
    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn all_reduce_sum(&self, _tensors: &mut Vec<Tensor>) -> Result<(), CommError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn all_reduce_scalar_sum(&self, value: f64) -> Result<f64, CommError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(value)
    }
}

// ---- harness ----------------------------------------------------------------

struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("ferrl-dp-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// The policy's trainable vars as raw `f32` bit patterns (bitwise comparison —
/// `==` on floats would also pass for `-0.0` vs `0.0`).
fn var_bits<P: Policy>(policy: &P) -> Vec<Vec<u32>> {
    policy
        .trainable_vars()
        .iter()
        .map(|v| {
            v.as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect()
}

/// Raw `f32` bits for every tensor in an optimizer-moment list.
fn tensor_bits(tensors: &[Tensor]) -> Vec<Vec<u32>> {
    tensors
        .iter()
        .map(|tensor| {
            tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect()
}

/// Adam state as a bitwise-comparable value. The moments matter independently
/// of the adapter: Adam can cancel a uniform gradient-scale error in the weight
/// update while retaining the wrong first and second moments.
fn optimizer_bits(state: &OptimizerState) -> (usize, Vec<Vec<u32>>, Vec<Vec<u32>>) {
    (
        state.step_t,
        tensor_bits(&state.first_moments),
        tensor_bits(&state.second_moments),
    )
}

/// Strip fields that describe wall time or device-memory sampling rather than
/// the mathematical result of a trainer window.
fn deterministic_metrics(metrics: &Metrics) -> Metrics {
    let mut metrics = metrics.clone();
    metrics.step_secs = 0.0;
    metrics.tokens_per_sec = 0.0;
    metrics.cuda_mem_start_used_bytes = 0;
    metrics.cuda_mem_peak_used_bytes = 0;
    metrics.cuda_mem_end_used_bytes = 0;
    metrics.cuda_mem_total_bytes = 0;
    metrics.cuda_mem_peak_delta_bytes = 0;
    metrics.cuda_mem_probe_events.clear();
    metrics.decoder_cache_snapshots.clear();
    metrics
}

#[allow(clippy::cognitive_complexity)] // one assertion surface pins every ordinary sentinel
fn assert_ledger_performance_unmeasured(metrics: &Metrics, context: &str) {
    assert_eq!(metrics.step_secs, 0.0, "{context}: step_secs");
    assert_eq!(metrics.tokens_per_sec, 0.0, "{context}: tokens_per_sec");
    assert_eq!(
        metrics.cuda_mem_start_used_bytes, 0,
        "{context}: cuda start"
    );
    assert_eq!(metrics.cuda_mem_peak_used_bytes, 0, "{context}: cuda peak");
    assert_eq!(metrics.cuda_mem_end_used_bytes, 0, "{context}: cuda end");
    assert_eq!(metrics.cuda_mem_total_bytes, 0, "{context}: cuda total");
    assert_eq!(
        metrics.cuda_mem_peak_delta_bytes, 0,
        "{context}: cuda delta"
    );
    assert!(
        metrics.cuda_mem_probe_events.is_empty(),
        "{context}: cuda probe events"
    );
    assert!(
        metrics.decoder_cache_snapshots.is_empty(),
        "{context}: decoder cache snapshots"
    );
}

fn assert_ledger_identity_mismatch<T>(result: Result<T, TrainerError>, what: &str) {
    match result {
        Err(TrainerError::RolloutLedger(RolloutLedgerError::IdentityMismatch)) => {}
        Err(error) => panic!("{what}: expected ledger identity mismatch, got {error:?}"),
        Ok(_) => panic!("{what}: mismatched learner identity was accepted"),
    }
}

/// Largest |a - b| across two var-bit snapshots (for the measured envelope).
fn max_abs_diff(a: &[Vec<u32>], b: &[Vec<u32>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(va, vb)| va.iter().zip(vb))
        .map(|(&xa, &xb)| f64::from((f32::from_bits(xa) - f32::from_bits(xb)).abs()))
        .fold(0.0, f64::max)
}

/// One rank's outcome: final var bits + the metrics history.
type RankRun = (Vec<Vec<u32>>, Vec<Metrics>);

/// Train a fresh [`ScriptedPolicy`] world of `world_size` ranks (one thread
/// per rank, rank `r` under `base/rank<r>`), returning each rank's outcome.
fn run_scripted_world(
    base: &Path,
    world_size: usize,
    cfg: &TrainerConfig,
    samples: &[Sample<()>],
) -> Vec<RankRun> {
    let comms = LocalComm::world(world_size);
    std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg.clone(), &run, comm).unwrap();
                    let history = trainer
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, samples)
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Train a fresh single-rank [`ScriptedPolicy`] run via the legacy
/// constructor, returning its outcome.
fn run_scripted_single(base: &Path, cfg: &TrainerConfig, samples: &[Sample<()>]) -> RankRun {
    let mut policy = ScriptedPolicy::new(SEED).unwrap();
    let run = RunDir::create(base, "single").unwrap();
    let mut trainer = Trainer::new(cfg.clone(), &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, samples)
        .unwrap();
    (var_bits(&policy), history.0)
}

fn scripted_cfg() -> TrainerConfig {
    TrainerConfig {
        steps: 3,
        group_size: 4,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.04,
        mu: 2,
        lr: 5e-3,
        loss_type: LossType::Grpo,
        ..TrainerConfig::default()
    }
}

fn live_samples() -> Vec<Sample<()>> {
    ["a", "b", "c", "d"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect()
}

fn assert_lockstep(ranks: &[RankRun], what: &str) {
    for (r, (bits, _)) in ranks.iter().enumerate().skip(1) {
        assert_eq!(
            &ranks[0].0, bits,
            "{what}: rank {r} diverged bitwise from rank 0"
        );
    }
}

/// The pre-clip global `grad_norm` must agree per step between a sharded and
/// a single run (modulo reassociation). This is the scale oracle the weight
/// envelope CANNOT be: `AdamW`'s m̂/√v̂ cancels a uniform gradient-scale error
/// and global-norm clipping is bitwise invariant to power-of-2 scales, so a
/// missed `world` divisor or a localized Dapo normalizer (both exact 2x)
/// leaves the final weights untouched — but the reported pre-clip norm sees
/// the 2x directly (found by the PR-F mutation sweep: M2/M3 survived the
/// weight envelope alone).
fn assert_grad_norms_match(world: &[Metrics], single: &[Metrics], what: &str) {
    assert_eq!(world.len(), single.len(), "{what}: step counts differ");
    for (i, (w, s)) in world.iter().zip(single).enumerate() {
        let diff = (f64::from(w.grad_norm) - f64::from(s.grad_norm)).abs();
        let scale = f64::from(s.grad_norm).abs().max(1e-12);
        assert!(
            diff / scale < 1e-3,
            "{what}: step {i} grad_norm diverged — sharded {} vs single {} (a \
             uniform gradient-scale error, e.g. a missed world divisor or a \
             local normalizer?)",
            w.grad_norm,
            s.grad_norm
        );
    }
}

// ---- preemption stop is globalized across the DP world ----------------------

/// The cooperative preemption stop must halt **every** rank at the same step even
/// when the flag is installed on only ONE rank — the un-footgunned, install-invariant
/// case. Rank 1 here has **no** flag at all; under a rank-local poll it would skip
/// the per-step reduce, run ahead into the next window's collective while rank 0
/// (flag set) broke out, and hang until the timeout. The install-invariant poll makes
/// rank 1 join the reduce anyway, so both stop after the same step. The 20 s timeout
/// converts any regression into a loud failure instead of a suite hang.
#[test]
fn preemption_flag_stops_the_whole_dp_world_in_lockstep() {
    let tmp = TempDir::new("preempt-dp");
    // More steps than we expect to run, so stopping after step 0 is visibly early.
    let cfg = TrainerConfig {
        steps: 5,
        ..scripted_cfg()
    };
    let samples = live_samples();
    // Flag installed on rank 0 ONLY; rank 1 gets none — the uneven-install case.
    let flag0 = Arc::new(AtomicBool::new(true));
    // A short collective timeout so a regression (a rank that fails to stop in
    // lockstep) fails fast as a loud timeout instead of hanging the suite.
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(20));
    let histories: Vec<Vec<Metrics>> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let flag0 = Arc::clone(&flag0);
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path().to_path_buf();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(&base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    if rank == 0 {
                        trainer = trainer.with_preemption_flag(flag0);
                    }
                    // Returns Ok only if the world never deadlocked — a rank stuck in a
                    // collective its peer skipped would surface as a CommError here.
                    trainer
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, &samples)
                        .unwrap()
                        .0
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(
        histories[0].len(),
        1,
        "rank 0 (flag set) should stop after step 0"
    );
    assert_eq!(
        histories[1].len(),
        histories[0].len(),
        "rank 1 (NO flag) must stop at the same step — the poll is install-invariant"
    );
}

// ---- gate 1: sharded vs single, measured envelope ---------------------------

/// World-2 at per-rank accumulation 2 vs single-rank at accumulation 4: the
/// same global windows, the same per-item gradients, different summation
/// association. Measured on the dev host 2026-06-12: Grpo 3.5e-9, Dapo
/// 4.6e-8 — the 1e-5 bound leaves ~200x headroom for the runner-pool spread
/// while staying orders below the bug signal (a missed `world` divisor is a
/// ~2x gradient-scale error, ~1e-2 on the weights after these steps).
#[test]
#[allow(clippy::print_stderr)] // the measured envelope is the calibration record
fn world2_matches_single_run_within_the_reassociation_envelope_grpo() {
    let tmp = TempDir::new("env-grpo");
    let cfg2 = TrainerConfig {
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let cfg1 = TrainerConfig {
        grad_accum_steps: 4,
        ..scripted_cfg()
    };
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &live_samples());
    let single = run_scripted_single(tmp.path(), &cfg1, &live_samples());
    assert_lockstep(&ranks, "grpo envelope");
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    eprintln!("measured grpo sharded-vs-single envelope: {envelope:.3e}");
    assert!(
        envelope < 1e-5,
        "sharded run left the reassociation envelope: {envelope:.3e}"
    );
    assert_grad_norms_match(&ranks[0].1, &single.1, "grpo envelope");
    // Vacuity guard: training actually moved the weights.
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert!(max_abs_diff(&single.0, &init) > 0.0, "no training signal");
}

/// The Dapo variant pins the GLOBAL window-token normalizer: a local (per
/// rank) normalizer would scale every gradient by ~2x and blow far past the
/// envelope.
#[test]
#[allow(clippy::print_stderr)] // the measured envelope is the calibration record
fn world2_matches_single_run_within_the_reassociation_envelope_dapo() {
    let tmp = TempDir::new("env-dapo");
    let base = TrainerConfig {
        beta: 0.0,
        mu: 1,
        loss_type: LossType::Dapo,
        ..scripted_cfg()
    };
    let cfg2 = TrainerConfig {
        grad_accum_steps: 2,
        ..base.clone()
    };
    let cfg1 = TrainerConfig {
        grad_accum_steps: 4,
        ..base
    };
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &live_samples());
    let single = run_scripted_single(tmp.path(), &cfg1, &live_samples());
    assert_lockstep(&ranks, "dapo envelope");
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    eprintln!("measured dapo sharded-vs-single envelope: {envelope:.3e}");
    assert!(
        envelope < 1e-5,
        "sharded run left the reassociation envelope: {envelope:.3e}"
    );
    // The scale oracle: a LOCAL window_tokens normalizer is an exact 2x the
    // weight envelope cannot see (AdamW cancels it) but the norm does.
    assert_grad_norms_match(&ranks[0].1, &single.1, "dapo envelope");
    // Vacuity guard: training actually moved the weights.
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert!(max_abs_diff(&single.0, &init) > 0.0, "no training signal");
}

/// The wraparound case: 5 prompts under a global window of 4, so windows wrap
/// mod len and shards straddle the boundary — the union-of-shards ≡
/// single-run-window identity must hold for ANY prompt count, not just exact
/// fits (the multiset identity under the shared `mod len` cycling).
#[test]
#[allow(clippy::print_stderr)] // the measured envelope is the calibration record
fn world2_matches_single_run_when_the_prompt_cycle_wraps() {
    let tmp = TempDir::new("env-wrap");
    let cfg2 = TrainerConfig {
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let cfg1 = TrainerConfig {
        grad_accum_steps: 4,
        ..scripted_cfg()
    };
    let samples: Vec<Sample<()>> = ["a", "b", "c", "d", "b"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &samples);
    let single = run_scripted_single(tmp.path(), &cfg1, &samples);
    assert_lockstep(&ranks, "wraparound envelope");
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    eprintln!("measured wraparound sharded-vs-single envelope: {envelope:.3e}");
    assert!(
        envelope < 1e-5,
        "sharded run left the reassociation envelope: {envelope:.3e}"
    );
    assert_grad_norms_match(&ranks[0].1, &single.1, "wraparound envelope");
}

// ---- gate 3: world-1 is byte-for-byte the legacy path ------------------------

/// `Trainer::new`, `with_comm(SoloComm)` and a world-1 `LocalComm` must be
/// BIT-identical — the DP plumbing may not perturb the single-rank path (every
/// collective call site is guarded on `world_size() > 1`; this pins the guard
/// discipline). Covered at accumulation 2 (the affine scale path) and 1 (the
/// no-affine bit-identity skip, now keyed on `accum · world == 1`).
#[test]
fn world_one_localcomm_solocomm_and_legacy_are_bit_identical() {
    for accum in [1, 2] {
        let tmp = TempDir::new("world1");
        let cfg = TrainerConfig {
            grad_accum_steps: accum,
            ..scripted_cfg()
        };
        let legacy = run_scripted_single(tmp.path(), &cfg, &live_samples());

        let mut solo_policy = ScriptedPolicy::new(SEED).unwrap();
        let run = RunDir::create(tmp.path(), "solo").unwrap();
        let mut trainer = Trainer::with_comm(cfg.clone(), &run, SoloComm).unwrap();
        trainer
            .train(
                &mut solo_policy,
                &EchoOrFlatReward,
                &CharTokenizer,
                &live_samples(),
            )
            .unwrap();
        assert_eq!(
            legacy.0,
            var_bits(&solo_policy),
            "SoloComm diverged from the legacy constructor at accum {accum}"
        );

        let world1 = run_scripted_world(tmp.path(), 1, &cfg, &live_samples());
        assert_eq!(
            legacy.0, world1[0].0,
            "world-1 LocalComm diverged from the legacy constructor at accum {accum}"
        );
    }
}

/// One separated world-1 ledger window must produce the exact update the
/// ordinary in-process trainer produces from the same learner pre-state.
///
/// The scripted fixture keeps rollouts deterministic while retaining a real
/// LoRA/autograd/Adam path. Two live prompt groups exercise whole-window
/// accumulation, and `mu = 2` makes the frozen old/reference snapshots
/// load-bearing on the second inner update. Comparing optimizer moments as well
/// as adapter values closes Adam's uniform-gradient-scale blind spot.
#[test]
#[allow(clippy::cognitive_complexity)] // one end-to-end oracle keeps every prestate rejection visible
fn world_one_rollout_ledger_update_is_bit_identical_to_direct_training() {
    let tmp = TempDir::new("world1-ledger-equivalence");
    let samples = live_samples();
    let cfg = TrainerConfig {
        steps: 1,
        grad_accum_steps: 2,
        checkpoint_every: Some(1),
        ..scripted_cfg()
    };
    let policy_sha256 = format!("{:064x}", 1);

    // Direct reference: one ordinary trainer window, including a
    // momentum-faithful checkpoint so its otherwise-private Adam state is
    // observable through the existing public checkpoint API.
    let mut direct_policy = ScriptedPolicy::new(SEED).unwrap();
    let initial_adapter = var_bits(&direct_policy);
    let direct_run = RunDir::create(tmp.path(), "direct").unwrap();
    let mut direct_trainer = Trainer::new(cfg.clone(), &direct_run).unwrap();
    let (direct_history, _) = direct_trainer
        .train(
            &mut direct_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
        )
        .unwrap();
    assert_eq!(direct_history.len(), 1);
    let direct_metrics = &direct_history[0];
    assert!(
        direct_metrics.grad_norm > 0.0,
        "direct reference performed no real optimizer update"
    );
    assert!(
        direct_metrics.kl > 0.0,
        "mu=2/beta>0 reference did not exercise the frozen reference snapshot"
    );
    assert!(
        direct_metrics.step_secs > 0.0 && direct_metrics.tokens_per_sec > 0.0,
        "ordinary direct training did not publish measured whole-window performance"
    );
    let direct_adapter = var_bits(&direct_policy);
    assert_ne!(
        direct_adapter, initial_adapter,
        "direct reference left the adapter at initialization"
    );
    let direct_probe = ScriptedPolicy::new(SEED.wrapping_add(1)).unwrap();
    let direct_checkpoint = ferrl::checkpoint::load_checkpoint(
        direct_run.checkpoints_dir().join("step-1"),
        &direct_probe.trainable_vars(),
    )
    .unwrap();
    let direct_optimizer = direct_checkpoint
        .optimizer_state
        .expect("direct checkpoint must contain Adam state");
    assert_eq!(direct_optimizer.step_t, cfg.mu);

    // Fresh, byte-identical Vars are still a different live optimizer binding.
    // The collector must re-read the policy's current variable set and reject
    // the replacement before publishing a ledger step.
    let replacing_root = tmp.path().join("replacing-rollout-ledger");
    let mut replacing_policy = ReplacingGeneratePolicy {
        inner: ScriptedPolicy::new(SEED).unwrap(),
        seed: SEED,
        replaced: false,
    };
    let replacing_run = RunDir::create(tmp.path(), "replacing-collector").unwrap();
    let mut replacing_collector = Trainer::new(cfg.clone(), &replacing_run).unwrap();
    match replacing_collector.collect_rollout_ledger_step(
        0,
        &mut replacing_policy,
        &EchoOrFlatReward,
        &CharTokenizer,
        &samples,
        &replacing_root,
        &policy_sha256,
        None,
    ) {
        Err(TrainerError::Contract(message)) => assert!(
            message.contains("trainable-variable set changed during rollout collection"),
            "unexpected replacement rejection: {message}"
        ),
        Err(error) => panic!("expected trainable-variable replacement rejection, got {error:?}"),
        Ok(path) => panic!("trainable-variable replacement published {path:?}"),
    }
    assert!(
        std::fs::read_dir(&replacing_root).unwrap().next().is_none(),
        "rejected collection left a reader-visible package"
    );

    // Separated path: collect with one policy instance and consume with an
    // independent instance holding the same learner pre-state.
    let ledger_root = tmp.path().join("rollout-ledger");
    let mut collector_policy = ScriptedPolicy::new(SEED).unwrap();
    let collector_run = RunDir::create(tmp.path(), "collector").unwrap();
    let mut collector = Trainer::new(cfg.clone(), &collector_run).unwrap();
    collector
        .collect_rollout_ledger_step(
            0,
            &mut collector_policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &samples,
            &ledger_root,
            &policy_sha256,
            None,
        )
        .unwrap();

    // Run orchestration and output controls do not change a single learner
    // update. Each mutation must remain compatible with the collector's ledger
    // while preserving the exact mathematical result.
    let mut longer_run = cfg.clone();
    longer_run.steps += 4;
    let mut different_checkpoint_cadence = cfg.clone();
    different_checkpoint_cadence.checkpoint_every = None;
    let mut candidate_logging = cfg.clone();
    candidate_logging.candidate_log_top_k = 1;
    let mut gpu_probing = cfg.clone();
    gpu_probing.gpu_memory_probe = true;
    for (label, operational_cfg) in [
        ("different run horizon", longer_run),
        ("different checkpoint cadence", different_checkpoint_cadence),
        ("different candidate logging", candidate_logging),
        ("different GPU probing", gpu_probing),
    ] {
        let mut policy = ScriptedPolicy::new(SEED).unwrap();
        let run = RunDir::create(tmp.path(), format!("operational-{label}")).unwrap();
        let mut trainer = Trainer::new(operational_cfg, &run).unwrap();
        let (metrics, optimizer) = trainer
            .train_rollout_ledger_step(0, &mut policy, &ledger_root, &policy_sha256, None)
            .unwrap_or_else(|error| panic!("{label} rejected: {error:?}"));
        assert_eq!(
            deterministic_metrics(&metrics),
            deterministic_metrics(direct_metrics),
            "{label}: mathematical metrics diverged"
        );
        assert_eq!(
            var_bits(&policy),
            direct_adapter,
            "{label}: adapter diverged"
        );
        assert_eq!(
            optimizer_bits(&optimizer),
            optimizer_bits(&direct_optimizer),
            "{label}: Adam state diverged"
        );
        assert_ledger_performance_unmeasured(&metrics, label);
    }

    // Both public identity inputs and live adapter values are checked before
    // scoring or mutation. A rejected read must leave the learner untouched.
    let mut wrong_policy = ScriptedPolicy::new(SEED).unwrap();
    let wrong_before = var_bits(&wrong_policy);
    let wrong_run = RunDir::create(tmp.path(), "wrong-policy").unwrap();
    let mut wrong_trainer = Trainer::new(cfg.clone(), &wrong_run).unwrap();
    let wrong_digest = format!("{:064x}", 2);
    assert_ledger_identity_mismatch(
        wrong_trainer.train_rollout_ledger_step(
            0,
            &mut wrong_policy,
            &ledger_root,
            &wrong_digest,
            None,
        ),
        "different policy digest",
    );
    assert_eq!(
        var_bits(&wrong_policy),
        wrong_before,
        "policy-digest rejection mutated the learner adapter"
    );

    let mut stale_adapter_policy = ScriptedPolicy::new(SEED).unwrap();
    let stale_var = stale_adapter_policy.trainable_vars()[1].clone();
    let (rows, cols) = stale_var.as_tensor().dims2().unwrap();
    let mut stale_values = vec![0.0_f32; rows * cols];
    stale_values[0] = 0.125;
    stale_var
        .set(&Tensor::from_vec(stale_values, (rows, cols), &Device::Cpu).unwrap())
        .unwrap();
    let stale_before = var_bits(&stale_adapter_policy);
    let stale_run = RunDir::create(tmp.path(), "stale-adapter").unwrap();
    let mut stale_trainer = Trainer::new(cfg.clone(), &stale_run).unwrap();
    assert_ledger_identity_mismatch(
        stale_trainer.train_rollout_ledger_step(
            0,
            &mut stale_adapter_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ),
        "different live adapter pre-state",
    );
    assert_eq!(
        var_bits(&stale_adapter_policy),
        stale_before,
        "adapter-identity rejection mutated the learner adapter"
    );

    let drift_cfg = TrainerConfig {
        clip_eps: cfg.clip_eps + 0.01,
        ..cfg.clone()
    };
    let mut drift_policy = ScriptedPolicy::new(SEED).unwrap();
    let drift_before = var_bits(&drift_policy);
    let drift_run = RunDir::create(tmp.path(), "config-drift").unwrap();
    let mut drift_trainer = Trainer::new(drift_cfg, &drift_run).unwrap();
    assert_ledger_identity_mismatch(
        drift_trainer.train_rollout_ledger_step(
            0,
            &mut drift_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ),
        "different clip_eps",
    );
    assert_eq!(
        var_bits(&drift_policy),
        drift_before,
        "trainer-config rejection mutated the learner adapter"
    );

    let mut wrong_optimizer_policy = ScriptedPolicy::new(SEED).unwrap();
    let wrong_optimizer_before = var_bits(&wrong_optimizer_policy);
    let wrong_optimizer_run = RunDir::create(tmp.path(), "wrong-optimizer").unwrap();
    let mut wrong_optimizer_trainer = Trainer::new(cfg.clone(), &wrong_optimizer_run).unwrap();
    assert_ledger_identity_mismatch(
        wrong_optimizer_trainer.train_rollout_ledger_step(
            0,
            &mut wrong_optimizer_policy,
            &ledger_root,
            &policy_sha256,
            Some(&direct_optimizer),
        ),
        "different Adam moments/step counter",
    );
    assert_eq!(
        var_bits(&wrong_optimizer_policy),
        wrong_optimizer_before,
        "optimizer-identity rejection mutated the learner adapter"
    );

    let mut wrong_recipe_policy = RecipeScriptedPolicy {
        inner: ScriptedPolicy::new(SEED).unwrap(),
    };
    assert_eq!(
        var_bits(&wrong_recipe_policy),
        initial_adapter,
        "recipe-drift premise changed the live adapter values"
    );
    let wrong_recipe_before = var_bits(&wrong_recipe_policy);
    let wrong_recipe_run = RunDir::create(tmp.path(), "wrong-recipe").unwrap();
    let mut wrong_recipe_trainer = Trainer::new(cfg.clone(), &wrong_recipe_run).unwrap();
    assert_ledger_identity_mismatch(
        wrong_recipe_trainer.train_rollout_ledger_step(
            0,
            &mut wrong_recipe_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ),
        "different tensor recipe/schema",
    );
    assert_eq!(
        var_bits(&wrong_recipe_policy),
        wrong_recipe_before,
        "tensor-schema rejection mutated the learner adapter"
    );

    #[cfg(target_os = "linux")]
    {
        // `/dev/full` opens successfully but rejects the final metrics write. The
        // same live mu=2 window has already performed real Adam updates when that
        // error is reached, so the learner must restore its complete pre-call state
        // and permit an exact retry from the caller's original optimizer prestate.
        let mut late_failure_policy = ScriptedPolicy::new(SEED).unwrap();
        let late_failure_before = var_bits(&late_failure_policy);
        let late_failure_run = RunDir::create(tmp.path(), "late-metrics-failure").unwrap();
        std::os::unix::fs::symlink("/dev/full", late_failure_run.metrics_path()).unwrap();
        let mut late_failure_trainer = Trainer::new(cfg.clone(), &late_failure_run).unwrap();
        match late_failure_trainer.train_rollout_ledger_step(
            0,
            &mut late_failure_policy,
            &ledger_root,
            &policy_sha256,
            None,
        ) {
            Err(TrainerError::Telemetry(_)) => {}
            Err(error) => panic!("expected forced late metrics failure, got {error:?}"),
            Ok(_) => panic!("/dev/full unexpectedly accepted the learner metrics row"),
        }
        assert_eq!(
            var_bits(&late_failure_policy),
            late_failure_before,
            "late failure left the adapter advanced"
        );
        assert!(
            late_failure_policy.adapter_enabled(),
            "late failure did not restore the adapter-enabled flag"
        );

        let retry_run = RunDir::create(tmp.path(), "late-metrics-retry").unwrap();
        let mut retry_trainer = Trainer::new(cfg.clone(), &retry_run).unwrap();
        let (retry_metrics, retry_optimizer) = retry_trainer
            .train_rollout_ledger_step(
                0,
                &mut late_failure_policy,
                &ledger_root,
                &policy_sha256,
                None,
            )
            .unwrap();
        assert_eq!(var_bits(&late_failure_policy), direct_adapter);
        assert_eq!(
            optimizer_bits(&retry_optimizer),
            optimizer_bits(&direct_optimizer),
            "retry after rollback did not recover the exact Adam continuation"
        );
        assert_ledger_performance_unmeasured(&retry_metrics, "retry after rollback");
    }

    let mut ledger_policy = ScriptedPolicy::new(SEED).unwrap();
    assert_eq!(var_bits(&ledger_policy), initial_adapter);
    let ledger_run = RunDir::create(tmp.path(), "learner").unwrap();
    let mut ledger_trainer = Trainer::new(cfg.clone(), &ledger_run).unwrap();
    let (ledger_metrics, ledger_optimizer) = ledger_trainer
        .train_rollout_ledger_step(0, &mut ledger_policy, &ledger_root, &policy_sha256, None)
        .unwrap();

    assert_ledger_performance_unmeasured(&ledger_metrics, "returned ledger metrics");
    let persisted_ledger_metrics = ferrl::read_metrics(ledger_run.metrics_path()).unwrap();
    assert_eq!(persisted_ledger_metrics.len(), 1);
    assert_ledger_performance_unmeasured(&persisted_ledger_metrics[0], "persisted ledger metrics");

    assert_eq!(
        deterministic_metrics(&ledger_metrics),
        deterministic_metrics(direct_metrics),
        "ledger learner metrics diverged from direct training"
    );
    assert_eq!(
        var_bits(&ledger_policy),
        direct_adapter,
        "ledger learner adapter diverged from direct training"
    );
    assert_eq!(
        optimizer_bits(&ledger_optimizer),
        optimizer_bits(&direct_optimizer),
        "ledger learner Adam state diverged from direct training"
    );

    let (_, first_moments, second_moments) = optimizer_bits(&ledger_optimizer);
    assert!(
        first_moments
            .iter()
            .flatten()
            .any(|&bits| bits != 0.0_f32.to_bits()),
        "first moments stayed zero — optimizer-state equivalence is vacuous"
    );
    assert!(
        second_moments
            .iter()
            .flatten()
            .any(|&bits| bits != 0.0_f32.to_bits()),
        "second moments stayed zero — optimizer-state equivalence is vacuous"
    );
}

// ---- gate 4: degenerate shards ------------------------------------------------

/// Rank 0's shard is all-degenerate every window ('e' prompts score a constant
/// → zero advantages → no live items at beta 0) while rank 1 holds live items:
/// the live-count decision is global, so rank 0 must enter every collective
/// with zeros — a local skip would deadlock the world (this test would hang,
/// converted to a loud timeout by `LocalComm`) — and both ranks must keep
/// stepping in lockstep on rank 1's signal alone.
#[test]
fn an_all_degenerate_local_shard_neither_deadlocks_nor_diverges() {
    let tmp = TempDir::new("deg-local");
    let cfg = TrainerConfig {
        steps: 2,
        beta: 0.0,
        ..scripted_cfg()
    };
    // accum 1, world 2: window step consumes prompts [2*step, 2*step + 1] —
    // rank 0 always draws "e" (degenerate), rank 1 always "a" (live).
    let cfg = TrainerConfig {
        grad_accum_steps: 1,
        ..cfg
    };
    let samples = ["e", "a"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg, &samples);
    assert_lockstep(&ranks, "degenerate local shard");
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert!(
        max_abs_diff(&ranks[0].0, &init) > 0.0,
        "rank 1's live signal should still move the (lockstepped) weights"
    );
    // Rank 0's local stats see only its degenerate shard.
    assert!(
        ranks[0]
            .1
            .iter()
            .all(|m| (m.frac_reward_zero_std - 1.0).abs() < f32::EPSILON),
        "rank 0's shard is degenerate every window"
    );
    // The zeros oracle: a single-rank run at accum 2 consumes the identical
    // global window ('e' degenerate + 'a' live) with the identical loss scale
    // and live count, so the empty shard's contribution must be EXACTLY zeros
    // — folding anything else in (the PR-F mutation sweep's M6 planted the
    // weights themselves) diverges from this reference immediately, while
    // lockstep alone stays green (both ranks share the corrupted sum).
    let single = run_scripted_single(
        tmp.path(),
        &TrainerConfig {
            grad_accum_steps: 2,
            ..cfg
        },
        &samples,
    );
    let envelope = max_abs_diff(&ranks[0].0, &single.0);
    assert!(
        envelope < 1e-5,
        "the empty shard contributed something non-zero to the reduce: {envelope:.3e}"
    );
    assert_grad_norms_match(&ranks[0].1, &single.1, "degenerate local shard");
}

/// A var that some rank's backward never covers must abort EVERY rank, fast
/// and in lockstep: the locally-uncovered rank via the grad-coverage canary,
/// its peers via the globalized verdict — NOT a 300s timeout (a half-stepped
/// world or a stalled collective is exactly what the global uncovered count
/// exists to prevent; the PR-F mutation sweep's M8 showed no other gate
/// exercises a partially-covering rank).
#[test]
fn an_uncovered_var_on_one_rank_aborts_every_rank_in_lockstep() {
    let tmp = TempDir::new("uncovered-peer");
    let cfg = TrainerConfig {
        steps: 1,
        mu: 1,
        beta: 0.0,
        grad_accum_steps: 1,
        ..scripted_cfg()
    };
    let samples = ["a", "b"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let comms = LocalComm::world(2);
    let errs: Vec<(usize, String)> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    // Rank 1 wires the extra var into its loss; rank 0 does
                    // not — same var COUNT on both ranks (the collective
                    // contract holds), but rank 0's backward never covers it.
                    let mut policy = GatedExtraPolicy::new(SEED, rank == 1);
                    let run = RunDir::create(base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let err = trainer
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, &samples)
                        .unwrap_err();
                    (rank, err.to_string())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (rank, msg) in &errs {
        assert!(
            !msg.contains("timeout"),
            "rank {rank} must abort promptly, not stall into a timeout: {msg}"
        );
        if *rank == 0 {
            // The locally-uncovered rank reports the canary's own detail.
            assert!(
                !msg.contains("peer rank"),
                "rank 0 should fail its OWN canary, got: {msg}"
            );
        } else {
            assert!(
                msg.contains("peer rank"),
                "rank 1 must abort on the peer's verdict, got: {msg}"
            );
        }
    }
}

/// The world-1 path must issue ZERO collective calls — the byte-for-byte
/// legacy-path promise is the `world_size() > 1` guard discipline, and the
/// weight-bits trio gate alone cannot see a guard regression (an identity
/// reduce keeps the weights bitwise; the PR-F mutation sweep's M9).
#[test]
fn world_one_training_issues_no_collective_calls() {
    let tmp = TempDir::new("spy");
    let calls = Arc::new(AtomicUsize::new(0));
    let spy = SpyComm {
        calls: Arc::clone(&calls),
    };
    let mut policy = ScriptedPolicy::new(SEED).unwrap();
    let run = RunDir::create(tmp.path(), "spy").unwrap();
    let cfg = TrainerConfig {
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let mut trainer = Trainer::with_comm(cfg, &run, spy).unwrap();
    trainer
        .train(
            &mut policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &live_samples(),
        )
        .unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "the world-1 path entered a collective — a world > 1 guard regressed"
    );
}

/// Every rank's shard degenerate → the global live count is 0 → every rank
/// skips the window in lockstep (no update, no collective inside the skipped
/// epochs) and the run completes with the weights untouched.
#[test]
fn an_all_degenerate_global_window_is_skipped_by_every_rank() {
    let tmp = TempDir::new("deg-global");
    let cfg = TrainerConfig {
        steps: 2,
        beta: 0.0,
        grad_accum_steps: 1,
        ..scripted_cfg()
    };
    let samples = ["e", "e"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg, &samples);
    assert_lockstep(&ranks, "degenerate global window");
    let init = var_bits(&ScriptedPolicy::new(SEED).unwrap());
    assert_eq!(
        max_abs_diff(&ranks[0].0, &init),
        0.0,
        "a globally degenerate run must not move any weight"
    );
    for (bits, history) in &ranks {
        assert_eq!(bits, &init, "weights must stay at init bitwise");
        assert!(
            history.iter().all(|m| m.grad_norm == 0.0),
            "no update may run"
        );
    }
}

// ---- gate 6: rank-0 checkpointing + DP resume ---------------------------------

/// Rank 0 writes the world's only checkpoint, and a world-2 resume from it
/// continues BIT-exactly (deterministic policies: weights and optimizer
/// moments are rank-identical by lockstep, and there is no sampler RNG) — the
/// interrupted+resumed world ends bitwise equal to an uninterrupted one.
#[test]
fn rank_zero_writes_the_only_checkpoint_and_a_dp_resume_continues_bit_exactly() {
    let base = TrainerConfig {
        checkpoint_every: Some(1),
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let tmp_full = TempDir::new("resume-full");
    let full = run_scripted_world(
        tmp_full.path(),
        2,
        &TrainerConfig {
            steps: 4,
            ..base.clone()
        },
        &live_samples(),
    );
    assert_lockstep(&full, "uninterrupted");

    // Phase 1: run the first 2 steps, checkpointing each.
    let tmp = TempDir::new("resume-phased");
    let phase1 = run_scripted_world(
        tmp.path(),
        2,
        &TrainerConfig {
            steps: 2,
            ..base.clone()
        },
        &live_samples(),
    );
    assert_lockstep(&phase1, "phase 1");
    let rank0_ckpt = tmp.path().join("rank0").join("checkpoints").join("step-2");
    assert!(rank0_ckpt.is_dir(), "rank 0 must write the checkpoint");
    // RunDir pre-creates the (empty) checkpoints dir; rank 1 must never fill it.
    let rank1_ckpts = tmp.path().join("rank1").join("checkpoints");
    assert_eq!(
        std::fs::read_dir(&rank1_ckpts).unwrap().count(),
        0,
        "rank 1 must write NO checkpoints (rank-0-only)"
    );

    // Phase 2: every rank resumes from rank 0's checkpoint, to step 4.
    let cfg = TrainerConfig { steps: 4, ..base };
    let comms = LocalComm::world(2);
    let resumed: Vec<RankRun> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let ckpt = rank0_ckpt.clone();
                let cfg = cfg.clone();
                let base = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(base, format!("resume-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let history = trainer
                        .resume(
                            &ckpt,
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_lockstep(&resumed, "resumed");
    assert_eq!(resumed[0].1.len(), 2, "resume runs steps 2..4");
    assert_eq!(
        full[0].0, resumed[0].0,
        "interrupted + resumed must be bitwise the uninterrupted run"
    );
}

// ---- resume_latest auto-discovers rank 0's checkpoint under data parallelism --

/// Run a fresh 2-rank DP world that resumes-or-starts via `resume_latest`, with
/// every rank pointing at ONE shared checkpoint dir (`with_checkpoints_dir`) — the
/// auto-discovery substrate — while keeping per-rank run dirs for telemetry. `tag`
/// distinguishes each call's per-rank run dirs (a duplicate `run_id` is rejected).
fn run_dp_world_resume_latest(
    base: &Path,
    shared_ckpts: &Path,
    tag: u64,
    cfg: &TrainerConfig,
) -> Vec<RankRun> {
    let comms = LocalComm::world(2);
    std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let shared = shared_ckpts.to_path_buf();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(base, format!("rl{tag}-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm)
                        .unwrap()
                        .with_checkpoints_dir(shared);
                    let history = trainer
                        .resume_latest(
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// The auto-discovery counterpart of
/// `rank_zero_writes_the_only_checkpoint_and_a_dp_resume_continues_bit_exactly`:
/// instead of the launcher handing every rank rank 0's checkpoint dir for an
/// explicit `resume`, every rank points at ONE shared checkpoint dir
/// (`with_checkpoints_dir`) and calls `resume_latest`. rank 0's scan is broadcast
/// to the world (broadcast-from-rank-0 via the sum all-reduce), so all ranks resume
/// from the SAME step in lockstep — bitwise the uninterrupted run, exactly as the
/// explicit path. This is the behavior that replaced the old hard DP refusal.
#[test]
fn resume_latest_under_dp_auto_discovers_rank0_checkpoint_in_lockstep() {
    let base = TrainerConfig {
        checkpoint_every: Some(1),
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    // Oracle: an uninterrupted 4-step DP world (per-rank dirs, default checkpoints).
    let tmp_full = TempDir::new("rl-dp-full");
    let full = run_scripted_world(
        tmp_full.path(),
        2,
        &TrainerConfig {
            steps: 4,
            ..base.clone()
        },
        &live_samples(),
    );
    assert_lockstep(&full, "uninterrupted");

    // ONE shared checkpoint dir every rank reads, rank 0 writes (not pre-created —
    // the save path makes it; `latest_checkpoint` reads a missing dir as "none").
    let tmp = TempDir::new("rl-dp-shared");
    let shared_ckpts = tmp.path().join("shared-checkpoints");

    // Phase 1: an empty shared dir → all ranks start fresh, run 2 steps, rank 0
    // checkpoints step-2 into the shared dir.
    let phase1 = run_dp_world_resume_latest(
        tmp.path(),
        &shared_ckpts,
        1,
        &TrainerConfig {
            steps: 2,
            ..base.clone()
        },
    );
    assert_lockstep(&phase1, "phase 1");
    assert_eq!(phase1[0].1.len(), 2, "phase 1 runs steps 0..2 fresh");
    assert!(
        shared_ckpts.join("step-2").is_dir(),
        "rank 0 must write the checkpoint into the shared dir"
    );

    // Phase 2: resume_latest to step 4 — auto-discovers step-2 from the shared dir
    // and broadcasts it, so every rank resumes there and runs 2..4 in lockstep.
    let resumed = run_dp_world_resume_latest(
        tmp.path(),
        &shared_ckpts,
        2,
        &TrainerConfig { steps: 4, ..base },
    );
    assert_lockstep(&resumed, "resumed");
    assert_eq!(resumed[0].1.len(), 2, "resume_latest runs steps 2..4");
    assert_eq!(
        full[0].0, resumed[0].0,
        "DP resume_latest must reproduce the uninterrupted run bit-for-bit"
    );
}

/// With no checkpoint in the shared dir, `resume_latest` under DP must start every
/// rank fresh in lockstep — the broadcast carries the "start fresh" sentinel, so
/// the whole world takes the same branch (never a rank-0-resumes / peers-restart
/// split). Equivalent to an uninterrupted `train()` DP world.
#[test]
fn resume_latest_under_dp_with_no_checkpoint_starts_fresh_in_lockstep() {
    let cfg = TrainerConfig {
        steps: 3,
        grad_accum_steps: 2,
        ..scripted_cfg()
    };
    let tmp_full = TempDir::new("rl-dp-fresh-oracle");
    let full = run_scripted_world(tmp_full.path(), 2, &cfg, &live_samples());
    assert_lockstep(&full, "uninterrupted fresh");

    let tmp = TempDir::new("rl-dp-fresh");
    let shared_ckpts = tmp.path().join("shared-checkpoints"); // never created
    let fresh = run_dp_world_resume_latest(tmp.path(), &shared_ckpts, 1, &cfg);
    assert_lockstep(&fresh, "no-checkpoint fresh");
    assert_eq!(fresh[0].1.len(), 3, "no checkpoint → every step runs fresh");
    assert_eq!(
        full[0].0, fresh[0].0,
        "DP resume_latest with no checkpoint must equal an uninterrupted DP train()"
    );
}

/// The divergence-prevention claim, pinned directly: with **per-rank** checkpoint dirs
/// (the misuse the shared-dir contract warns against), the broadcast still carries rank
/// 0's decision to the peers, so a peer ATTEMPTS rank 0's resume and fails **loudly**
/// (its own dir lacks the checkpoint) rather than silently starting fresh and diverging.
///
/// This is the test the shared-dir equivalence cases above cannot be: with a shared dir,
/// a buggy no-broadcast implementation (each rank scans its own dir) would *also* pass,
/// because every rank sees rank 0's checkpoint. Here, only rank 0's dir has the
/// checkpoint — so a no-broadcast implementation would have rank 1 scan its empty dir,
/// start fresh, run every step while rank 0 (resuming the final step) runs none, and
/// **deadlock** on the first mismatched collective (surfacing as a `Comm` timeout). The
/// coordinated implementation instead makes rank 1 fail at the checkpoint load
/// (`TrainerError::Checkpoint`) and makes rank 0 abort on the peer's load failure
/// (`TrainerError::Contract`) before it can proceed alone. A `Comm` error would fail
/// this test.
#[test]
#[allow(clippy::cognitive_complexity)]
fn resume_latest_under_dp_without_a_shared_dir_aborts_load_in_lockstep_not_fresh() {
    let base = TrainerConfig {
        checkpoint_every: Some(1),
        grad_accum_steps: 2,
        steps: 2,
        ..scripted_cfg()
    };
    let tmp = TempDir::new("rl-dp-perrank");
    // Phase 1: a 2-rank world with PER-RANK dirs writes rank 0's checkpoint at step 2;
    // rank 1's own checkpoint dir stays empty (rank-0-only writes).
    let phase1 = run_scripted_world(tmp.path(), 2, &base, &live_samples());
    assert_lockstep(&phase1, "phase 1");
    assert!(
        tmp.path()
            .join("rank0")
            .join("checkpoints")
            .join("step-2")
            .is_dir(),
        "rank 0 must write the only checkpoint"
    );
    assert_eq!(
        std::fs::read_dir(tmp.path().join("rank1").join("checkpoints"))
            .unwrap()
            .count(),
        0,
        "rank 1's own checkpoint dir must be empty"
    );

    // Phase 2: resume_latest with PER-RANK dirs (each rank REOPENS its own phase-1 dir),
    // steps == the checkpoint step. A short timeout so the buggy-impl divergence surfaces
    // as a fast Comm timeout rather than hanging the test.
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(20));
    let outcomes: Vec<Result<(), TrainerError>> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = base.clone();
                let basep = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::open(basep, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    trainer
                        .resume_latest(
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .map(|_| ())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // rank 0 can load its local step-2 checkpoint, but rank 1 cannot load the
    // broadcast step from its empty per-rank checkpoint dir. The load failure is
    // coordinated, so rank 0 must abort in lockstep instead of proceeding alone.
    let err0 = outcomes[0].as_ref().unwrap_err();
    assert!(
        matches!(err0, TrainerError::Contract(msg)
            if msg.contains("checkpoint load/restore failed on a peer rank")),
        "rank 0 must abort when a peer cannot load the broadcast checkpoint: got {err0:?}"
    );
    // rank 1 received rank 0's broadcast step and tried to resume from its OWN (empty)
    // dir → a loud checkpoint-load error. NOT a silent fresh start, NOT a Comm timeout
    // from a diverged collective (which is exactly what a no-broadcast impl produces).
    let err1 = outcomes[1].as_ref().unwrap_err();
    assert!(
        matches!(err1, TrainerError::Checkpoint(_)),
        "rank 1 must fail loudly at the checkpoint load (broadcast carried rank 0's step \
         to a dir that lacks it), not diverge into a Comm timeout: got {err1:?}"
    );
}

/// A rank-0 checkpoint-scan FAILURE must ride the broadcast every rank enters, so the
/// world aborts **in lockstep and promptly** — not a rank-0-only early return that
/// strands the peers in the collective until the timeout (the fresh-eyes-review
/// deadlock vector). Induced by pointing the shared checkpoint dir at a FILE, so rank
/// 0's `read_dir` of it errors (`ENOTDIR`). With a short collective timeout, a
/// regression (rank-0 `?`-return before the broadcast) would surface as a peer `Comm`
/// timeout — which this test forbids; the fix makes rank 0 return the real
/// `Checkpoint` error and the peer the synthesized `Contract` error, both at once.
#[test]
fn resume_latest_under_dp_broadcasts_a_rank0_scan_failure_instead_of_hanging() {
    let tmp = TempDir::new("rl-dp-scanfail");
    let not_a_dir = tmp.path().join("checkpoints-is-a-file");
    std::fs::write(&not_a_dir, b"not a directory").unwrap();
    let comms = LocalComm::world_with_timeout(2, std::time::Duration::from_secs(15));
    let errs: Vec<TrainerError> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let basep = tmp.path();
                let ckpt = not_a_dir.clone();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = ScriptedPolicy::new(SEED).unwrap();
                    let run = RunDir::create(basep, format!("sf-rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(scripted_cfg(), &run, comm)
                        .unwrap()
                        .with_checkpoints_dir(ckpt);
                    trainer
                        .resume_latest(
                            &mut policy,
                            &EchoOrFlatReward,
                            &CharTokenizer,
                            &live_samples(),
                        )
                        .map(|_| ())
                        .unwrap_err()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // No rank may surface a Comm timeout — that would mean a peer was stranded in the
    // broadcast by a rank-0-only early return (the bug the broadcast-the-failure fix
    // closes). rank 0 surfaces the real checkpoint IO error; the peer the Contract one.
    for e in &errs {
        assert!(
            !matches!(e, TrainerError::Comm(_)),
            "a rank-0 scan failure must abort in lockstep, not strand a peer in the \
             collective (Comm timeout): got {e:?}"
        );
    }
    assert!(
        errs.iter()
            .any(|e| matches!(e, TrainerError::Checkpoint(_))),
        "rank 0 must surface the real checkpoint scan error: {errs:?}"
    );
    assert!(
        errs.iter().any(|e| matches!(e, TrainerError::Contract(_))),
        "the peer must surface the synthesized lockstep-abort error: {errs:?}"
    );
}

// ---- gate 2: real tiny qwen3.5, bitwise rank lockstep, both modes -------------

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen35")
}

/// A char-level codec over the fixture's 64-token vocabulary.
struct ByteCodec;
impl TokenizerLike for ByteCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| u32::from(b) % 64).collect()
    }
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| char::from(b'a' + u8::try_from(i % 26).unwrap()))
            .collect()
    }
}

/// A deterministic reward that SPREADS over completions, so group advantages
/// are non-degenerate.
struct SpreadReward;
impl RewardFn for SpreadReward {
    type Target = ();
    fn reward(&self, _sample: &Sample<()>, completion: &str) -> Result<f32, RewardError> {
        Ok(completion
            .bytes()
            .enumerate()
            .map(|(i, b)| f32::from(b) * (0.3 + i as f32 * 0.17))
            .sum::<f32>()
            % 5.0)
    }
}

fn lora_policy(seed: u64) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, DType::F32, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load(&cfg, &vb, 2, 4.0).unwrap();
    let policy = Qwen3_5Policy::new(model, seed, 1.0);
    // The LoRA A factors come from `Var::randn`, which on CPU draws from
    // OS entropy (unseedable) — every rank would otherwise start from a
    // DIFFERENT adapter and lockstep would be broken at step 0 (the full-FT
    // twin needs no fill: its vars are the loaded weights). Overwrite every
    // adapter var with a seeded ~0.02-scale hash fill, identical on every
    // rank.
    for (k, v) in policy.trainable_vars().iter().enumerate() {
        let dims = v.as_tensor().dims().to_vec();
        let n: usize = dims.iter().product();
        let fill: Vec<f32> = (0..n)
            .map(|i| {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((k as u64 + 1).wrapping_mul(seed.wrapping_mul(40_503)));
                ((z % 1000) as f32 / 1000.0 - 0.5) * 0.04
            })
            .collect();
        v.set(&Tensor::from_vec(fill, dims, &Device::Cpu).unwrap())
            .unwrap();
    }
    policy
}

#[cfg(feature = "nccl")]
fn lora_policy_on_device(seed: u64, dtype: DType, device: &Device) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let vb = varbuilder_from_pretrained(&dir, dtype, device).unwrap();
    let mut model =
        Qwen3_5GradModel::load_with_adapter_dtype(&cfg, &vb, 2, 4.0, DType::F32).unwrap();
    model.set_activation_checkpointing(true);
    let policy = Qwen3_5Policy::new(model, seed, 1.0);
    for (k, v) in policy.trainable_vars().iter().enumerate() {
        let dims = v.as_tensor().dims().to_vec();
        let n: usize = dims.iter().product();
        let fill: Vec<f32> = (0..n)
            .map(|i| {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((k as u64 + 1).wrapping_mul(seed.wrapping_mul(40_503)));
                ((z % 1000) as f32 / 1000.0 - 0.5) * 0.04
            })
            .collect();
        v.set(&Tensor::from_vec(fill, dims, device).unwrap())
            .unwrap();
    }
    policy
}

fn full_ft_policy(seed: u64) -> Qwen3_5Policy {
    let dir = fixture_dir();
    let cfg = Qwen3_5Config::from_json_file(dir.join("config.json")).unwrap();
    let tensors = tensors_from_pretrained(&dir, &Device::Cpu).unwrap();
    let model = Qwen3_5GradModel::load_full_ft(&cfg, tensors, DType::F32, &Device::Cpu).unwrap();
    Qwen3_5Policy::new(model, seed, 1.0)
}

/// World-2 over the real tiny fixture: per-rank rollouts are STOCHASTIC and
/// shard-different, yet every rank steps from the identical reduced gradient,
/// so the weights must stay in bitwise lockstep — the invariant the whole DP
/// design rests on, checked end-to-end through the real model's grad path.
fn assert_real_model_lockstep(tag: &str, make_policy: fn(u64) -> Qwen3_5Policy) {
    let tmp = TempDir::new(tag);
    let cfg = TrainerConfig {
        steps: 2,
        group_size: 2,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.0,
        mu: 1,
        lr: 1e-3,
        loss_type: LossType::Grpo,
        ..TrainerConfig::default()
    };
    let samples = ["abc", "bcd"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();
    let comms = LocalComm::world(2);
    let ranks: Vec<RankRun> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let samples = samples.clone();
                let base = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = make_policy(7);
                    let run = RunDir::create(base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let history = trainer
                        .train(&mut policy, &SpreadReward, &ByteCodec, &samples)
                        .unwrap();
                    (var_bits(&policy), history.0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_lockstep(&ranks, tag);
    let init = var_bits(&make_policy(7));
    assert!(
        max_abs_diff(&ranks[0].0, &init) > 0.0,
        "{tag}: no training signal — the lockstep gate would be vacuous"
    );
    assert!(
        ranks[0]
            .1
            .iter()
            .any(|m| m.grad_norm > 0.0 && m.grad_norm.is_finite()),
        "{tag}: no real update ran"
    );
}

#[test]
fn dp_ranks_stay_in_bitwise_lockstep_on_real_qwen35_lora() {
    assert_real_model_lockstep("lockstep-lora", lora_policy);
}

#[test]
fn dp_ranks_stay_in_bitwise_lockstep_on_real_qwen35_full_ft() {
    assert_real_model_lockstep("lockstep-full-ft", full_ft_policy);
}

/// Manual resource-regression smoke for the trainer's CUDA/NCCL update path.
///
/// Launch one process per rank under Slurm, with a shared `FERRL_NCCL_RENDEZVOUS`.
/// The test uses the committed tiny qwen3.5 fixture so it is cheap, deterministic,
/// and asset-free, while still reaching rollout -> backward -> gradient all-reduce
/// -> optimizer over real CUDA tensors. The printed `NCCL_TINY_QWEN35_SMOKE` rows
/// are intentionally stable for external branch-vs-main parsers.
#[cfg(feature = "nccl")]
#[test]
#[ignore = "manual CUDA/NCCL resource gate; launch one process per rank under Slurm"]
#[allow(clippy::print_stderr)] // manual gate: the printed memory/timing rows are the deliverable
fn nccl_tiny_qwen35_lora_smoke_reaches_update_path() {
    let comm = ferrl::NcclComm::from_slurm_env().expect("bootstrap NCCL from Slurm env");
    let rank = comm.rank();
    let world = comm.world_size();
    let device = comm.device().clone();
    if let Some(warning) = ferrl::check_driver_compat(&device).warning() {
        eprintln!("{warning}");
    }
    ferrl::guard_first_kernel(&device).expect("CUDA first-kernel guard");

    let mut policy = lora_policy_on_device(7, DType::BF16, &device);
    let cfg = TrainerConfig {
        steps: 2,
        group_size: 2,
        max_new_tokens: 3,
        temperature: 1.0,
        beta: 0.0,
        mu: 1,
        lr: 1e-3,
        loss_type: LossType::Grpo,
        gpu_memory_probe: true,
        ..TrainerConfig::default()
    };
    let root = std::env::var_os("FERRL_NCCL_SMOKE_RUN_ROOT").map_or_else(
        || std::env::temp_dir().join(format!("ferrl-nccl-smoke-{}", std::process::id())),
        PathBuf::from,
    );
    std::fs::create_dir_all(&root).unwrap();
    let run = RunDir::create(&root, format!("rank{rank}")).unwrap();
    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
    let samples = ["abc", "bcd"]
        .map(|s| Sample::new(s, ()))
        .into_iter()
        .collect::<Vec<Sample<()>>>();

    let history = trainer
        .train(&mut policy, &SpreadReward, &ByteCodec, &samples)
        .expect("NCCL tiny qwen3.5 GRPO smoke failed")
        .0;

    assert_eq!(history.len(), 2);
    assert!(
        history
            .iter()
            .any(|m| m.grad_norm > 0.0 && m.grad_norm.is_finite()),
        "NCCL smoke reached no real optimizer update"
    );
    for m in history {
        assert!(
            m.cuda_mem_peak_used_bytes >= m.cuda_mem_start_used_bytes,
            "memory probe did not record a valid peak at step {}",
            m.step
        );
        eprintln!(
            "NCCL_TINY_QWEN35_SMOKE rank={rank} world={world} step={} grad_norm={} \
             step_secs={} tokens_per_sec={} cuda_start={} cuda_peak={} cuda_end={} \
             cuda_delta={}",
            m.step,
            m.grad_norm,
            m.step_secs,
            m.tokens_per_sec,
            m.cuda_mem_start_used_bytes,
            m.cuda_mem_peak_used_bytes,
            m.cuda_mem_end_used_bytes,
            m.cuda_mem_peak_delta_bytes
        );
    }
}
