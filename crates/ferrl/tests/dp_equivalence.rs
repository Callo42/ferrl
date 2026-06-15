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
//! index, with no per-rank RNG state to capture).

use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::ops::log_softmax;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use ferrl::lora::LoraLinear;
use ferrl::nn::RmsNorm;
use ferrl::policy::{GenConfig, Policy, Rollout};
use ferrl::telemetry::RunDir;
use ferrl::trainer::{TokenizerLike, Trainer, TrainerConfig};
use ferrl::{
    tensors_from_pretrained, varbuilder_from_pretrained, Comm, CommError, LocalComm, LossType,
    Metrics, Qwen3_5Config, Qwen3_5GradModel, Qwen3_5Policy, RewardFn, SoloComm,
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
    fn reward(&self, prompt: &str, completion: &str) -> f32 {
        if prompt.starts_with('e') {
            return 0.5;
        }
        match (prompt.chars().next(), completion.chars().next()) {
            (Some(p), Some(c)) if p == c => 1.0,
            _ => 0.0,
        }
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
    prompts: &[String],
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
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, prompts)
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
fn run_scripted_single(base: &Path, cfg: &TrainerConfig, prompts: &[String]) -> RankRun {
    let mut policy = ScriptedPolicy::new(SEED).unwrap();
    let run = RunDir::create(base, "single").unwrap();
    let mut trainer = Trainer::new(cfg.clone(), &run).unwrap();
    let history = trainer
        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, prompts)
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

fn live_prompts() -> Vec<String> {
    ["a", "b", "c", "d"].map(String::from).to_vec()
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
    let prompts = live_prompts();
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
                let prompts = prompts.clone();
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
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, &prompts)
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
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &live_prompts());
    let single = run_scripted_single(tmp.path(), &cfg1, &live_prompts());
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
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &live_prompts());
    let single = run_scripted_single(tmp.path(), &cfg1, &live_prompts());
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
    let prompts: Vec<String> = ["a", "b", "c", "d", "b"].map(String::from).to_vec();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg2, &prompts);
    let single = run_scripted_single(tmp.path(), &cfg1, &prompts);
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
        let legacy = run_scripted_single(tmp.path(), &cfg, &live_prompts());

        let mut solo_policy = ScriptedPolicy::new(SEED).unwrap();
        let run = RunDir::create(tmp.path(), "solo").unwrap();
        let mut trainer = Trainer::with_comm(cfg.clone(), &run, SoloComm).unwrap();
        trainer
            .train(
                &mut solo_policy,
                &EchoOrFlatReward,
                &CharTokenizer,
                &live_prompts(),
            )
            .unwrap();
        assert_eq!(
            legacy.0,
            var_bits(&solo_policy),
            "SoloComm diverged from the legacy constructor at accum {accum}"
        );

        let world1 = run_scripted_world(tmp.path(), 1, &cfg, &live_prompts());
        assert_eq!(
            legacy.0, world1[0].0,
            "world-1 LocalComm diverged from the legacy constructor at accum {accum}"
        );
    }
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
    let prompts = ["e", "a"].map(String::from).to_vec();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg, &prompts);
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
        &prompts,
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
    let prompts = ["a", "b"].map(String::from).to_vec();
    let comms = LocalComm::world(2);
    let errs: Vec<(usize, String)> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let prompts = prompts.clone();
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
                        .train(&mut policy, &EchoOrFlatReward, &CharTokenizer, &prompts)
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
            &live_prompts(),
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
    let prompts = ["e", "e"].map(String::from).to_vec();
    let ranks = run_scripted_world(tmp.path(), 2, &cfg, &prompts);
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
        &live_prompts(),
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
        &live_prompts(),
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
                            &live_prompts(),
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

// ---- resume_latest is single-rank only (rank-0-only checkpoint contract) ------

/// `resume_latest` auto-discovers a checkpoint in THIS rank's own `checkpoints/`,
/// but the DP contract writes checkpoints on rank 0 only and resumes every rank from
/// rank 0's directory. Per-rank discovery would let non-zero ranks find nothing,
/// silently start fresh, and diverge from rank 0 — then hang the next collective once
/// rank 0 finishes its shorter remaining steps first. So under `world_size > 1` it
/// must REFUSE (a contract error pointing at the explicit `resume(&rank0_ckpt)` path),
/// not foot-gun. The guard fires before any collective, so a single rank handle of a
/// 2-world proves it — no threads, no rendezvous.
#[test]
fn resume_latest_refuses_under_data_parallel() {
    let tmp = TempDir::new("resume-latest-dp-guard");
    let comm = LocalComm::world(2)
        .into_iter()
        .next()
        .expect("a 2-rank world has a rank 0");
    let run = RunDir::create(tmp.path(), "rank0").unwrap();
    let mut policy = ScriptedPolicy::new(SEED).unwrap();
    let mut trainer = Trainer::with_comm(scripted_cfg(), &run, comm).unwrap();
    let err = trainer
        .resume_latest(
            &mut policy,
            &EchoOrFlatReward,
            &CharTokenizer,
            &live_prompts(),
        )
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("single-rank") && msg.contains("resume("),
        "DP resume_latest must fail with a contract error pointing at resume(): got {msg:?}"
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
    fn reward(&self, _prompt: &str, completion: &str) -> f32 {
        completion
            .bytes()
            .enumerate()
            .map(|(i, b)| f32::from(b) * (0.3 + i as f32 * 0.17))
            .sum::<f32>()
            % 5.0
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
    let prompts = ["abc", "bcd"].map(String::from).to_vec();
    let comms = LocalComm::world(2);
    let ranks: Vec<RankRun> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let cfg = cfg.clone();
                let prompts = prompts.clone();
                let base = tmp.path();
                s.spawn(move || {
                    let rank = comm.rank();
                    let mut policy = make_policy(7);
                    let run = RunDir::create(base, format!("rank{rank}")).unwrap();
                    let mut trainer = Trainer::with_comm(cfg, &run, comm).unwrap();
                    let history = trainer
                        .train(&mut policy, &SpreadReward, &ByteCodec, &prompts)
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
