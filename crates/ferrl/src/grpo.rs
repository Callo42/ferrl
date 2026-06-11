//! Pure GRPO math.
//!
//! These functions are the *owned* core of `ferrl`: the closed-form GRPO
//! quantities, expressed over plain Rust slices of `f64` so they are trivially
//! unit-testable and free of any tensor / device dependency. Once the training
//! loop exists it will re-express the same algebra in [`candle_core::Tensor`] ops
//! (so candle can differentiate it), pinned to these scalar versions; the scalar
//! versions are themselves checked against a `NumPy` reference oracle via the
//! committed golden fixture (`grpo_golden.json`).
//!
//! The formulas follow TRL's `GRPOTrainer` and the `DeepSeekMath` paper:
//!
//! - **Advantages** are group-normalized:
//!   `A_i = (r_i - mean_g) / (std_g + eps)` with `eps =` [`GROUP_STD_EPS`] and
//!   `std_g` the *sample* (Bessel-corrected, `ddof = 1`) standard deviation over
//!   the group — matching TRL's `nanstd` / candle's `Tensor::var` on finite
//!   rewards (non-finite handling differs; see below). Dividing by the std is the
//!   [`ScaleRewards::Group`] default; [`ScaleRewards::None`] drops it (the
//!   Dr.GRPO-recommended setting).
//! - **KL** uses Schulman's k3 estimator `exp(d) - d - 1`, `d = logp_ref - logp`,
//!   which is non-negative and unbiased.
//! - **Surrogate** is the PPO-style clipped objective
//!   `min(ratio * A, clip(ratio, 1 - e_low, 1 + e_high) * A)` — asymmetric
//!   bands per DAPO clip-higher; symmetric when `e_low == e_high`. The ratio
//!   itself is per-token or per-sequence per [`ImportanceSamplingLevel`] (the
//!   GSPO seam).
//! - **Reduction** ([`LossType`]) is length-normalization only: classic GRPO
//!   averages per sequence then over sequences; Dr.GRPO divides the total by
//!   `num_sequences * max_len`; DAPO (the default) divides the total by the
//!   batch's active-token count. The Dr.GRPO *paper* algorithm is
//!   `LossType::DrGrpo` **plus** [`ScaleRewards::None`].
//!
//! Non-finite inputs are handled defensively — a **deliberate divergence** from
//! TRL's mainline path (which `nansum`s an all-NaN reward row to `0.0` and then
//! propagates any remaining `NaN` through the plain `mean`/`std`): here a
//! `NaN`/`±∞` reward is dropped from its group's mean/std and given a `0`
//! advantage — note `±∞ → 0`, unlike `torch.nan_to_num`, which maps `±∞` to
//! finite extremes — so one bad completion cannot poison the group. In [`masked_mean`] a value in a
//! masked-out position is ignored (so `0 · ∞` cannot leak `NaN`) and an all-pad
//! row contributes `0` to either reduction. Masks must be finite and
//! non-negative and rows must have width `≥ 1`; violations are caller bugs and
//! panic.

use serde::{Deserialize, Serialize};

/// Numerical-stability epsilon added to the group standard deviation when
/// normalizing advantages: `A_i = (r_i - mean) / (std + eps)`.
///
/// Matches TRL's default (`1e-4`) and keeps the advantage finite for
/// degenerate groups where every reward is identical (`std = 0`).
///
/// **Precondition: rewards are O(1)-scaled** (the verifiable-reward convention
/// — e.g. `0/1` correctness, or a small bounded score). The constant is
/// *absolute*, so for rewards whose genuine group spread is at or below `1e-4`
/// the eps dominates the denominator and crushes every advantage toward `0` —
/// the group silently stops teaching. If a reward function must emit tiny
/// magnitudes, rescale it (or use [`ScaleRewards::None`]) rather than relying
/// on the std normalization.
pub const GROUP_STD_EPS: f64 = 1e-4;

/// Which GRPO loss reduction to use over the per-token objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossType {
    /// Classic GRPO: average over valid tokens *within* each sequence, then
    /// average those per-sequence means over the group. Short completions are
    /// weighted as heavily as long ones — the length bias Dr.GRPO/DAPO
    /// identified; TRL's docstring now recommends against it. Kept selectable.
    Grpo,
    /// Dr.GRPO (<https://arxiv.org/abs/2503.20783>): divide the summed,
    /// mask-weighted objective by `num_sequences * max_len`, removing the
    /// length bias of classic GRPO.
    DrGrpo,
    /// DAPO (<https://arxiv.org/abs/2503.14476>): divide the summed,
    /// mask-weighted objective by the **total number of active tokens** in the
    /// batch (TRL's `loss_type="dapo"`, today's default there too). Removes the
    /// length bias without Dr.GRPO's fixed-constant choice. This is the
    /// default. In [`masked_mean`] the denominator is this batch's active
    /// tokens (clamped to `>= 1`); the trainer generalizes it to the
    /// **accumulation window's** total completion tokens (see
    /// `TrainerConfig::grad_accum_steps`), matching TRL's
    /// `num_items_in_batch` normalizer across an accumulated batch.
    ///
    /// For a batch of **equal-length, fully-unmasked** sequences this is
    /// numerically identical to [`LossType::Grpo`] (both reduce to
    /// `total / (num_seq * len)`); the two differ only under variable lengths
    /// or masking.
    #[default]
    Dapo,
}

/// How to scale group-centered rewards into advantages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleRewards {
    /// Divide centered rewards by the group std `+ eps` (TRL / GRPO default).
    #[default]
    Group,
    /// No std scaling: `A_i = r_i - mean_g`. Removes the question-difficulty
    /// bias; pair with [`LossType::DrGrpo`] for the Dr.GRPO paper algorithm.
    None,
}

/// At which level the PPO importance-sampling ratio is formed — the GSPO seam.
///
/// GSPO (<https://arxiv.org/abs/2507.18071>, what Qwen used for Qwen3-MoE RL
/// stability) replaces the per-token ratio with a single per-sequence ratio:
/// the masked **mean** of the per-token log-ratios, exponentiated (the
/// length-normalized sequence ratio `s_i = (π/π_old)(y_i|x)^{1/|y_i|}`).
/// Everything downstream of the ratio (clipping, surrogate, reduction) is
/// unchanged — mirroring TRL's `importance_sampling_level`, where the
/// sequence-level log-weight is `(log_ratio · mask).sum(-1) / mask.sum(-1)`
/// broadcast back over the sequence's tokens.
///
/// [`Token`](Self::Token) (the default) is classic GRPO and is bit-identical
/// to the pre-seam behavior. Note: GSPO practice pairs `Sequence` with a much
/// tighter clip band (the paper uses ~`3e-4`–`4e-4`) — the knob does not
/// re-default `clip_eps`. Whether to adopt `Sequence` for mixture-of-experts
/// training is an M3′-era decision; this seam exists so that decision is a
/// config flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportanceSamplingLevel {
    /// One importance ratio per token: `exp(logp_t - logp_old_t)` (classic
    /// GRPO / PPO). The default.
    #[default]
    Token,
    /// One importance ratio per sequence: `exp` of the masked mean per-token
    /// log-ratio, shared by every token of that sequence (GSPO).
    Sequence,
}

/// The per-sequence GSPO log-ratio: the masked mean of the per-token
/// log-ratios `logp - logp_old` over one sequence's valid tokens.
///
/// Returns `(Σ_t mask_t · (logp_t - logp_old_t)) / max(Σ_t mask_t, 1)` — the
/// log of the length-normalized sequence importance ratio
/// (see [`ImportanceSamplingLevel::Sequence`]). The `max(·, 1)` clamp mirrors
/// TRL (`mask.sum(-1).clamp(min=1.0)`), so an all-pad row yields `0` (ratio
/// `1`) rather than `0/0`. A value in a masked-out position is ignored, so a
/// non-finite log-prob at padding cannot poison the mean.
///
/// # Panics
///
/// Panics if the three slices differ in length, or if any mask entry is
/// negative or non-finite (the contract is `0.0`/`1.0`).
#[must_use]
pub fn sequence_log_ratio(logp: &[f64], logp_old: &[f64], mask: &[f64]) -> f64 {
    assert_eq!(
        logp.len(),
        logp_old.len(),
        "sequence_log_ratio: length mismatch"
    );
    assert_eq!(
        logp.len(),
        mask.len(),
        "sequence_log_ratio: mask length mismatch"
    );
    assert_nonneg_mask(mask);
    let denom: f64 = mask.iter().sum::<f64>().max(1.0);
    let num: f64 = logp
        .iter()
        .zip(logp_old.iter())
        .zip(mask.iter())
        .map(|((lp, lo), m)| if *m > 0.0 { (lp - lo) * m } else { 0.0 })
        .sum();
    num / denom
}

/// Sample (Bessel-corrected, `ddof = 1`) mean and standard deviation over the
/// **finite** entries of a slice.
///
/// Returns `(mean, std)`. The std divides by `n - 1` where `n` counts only the
/// finite entries. For **all-finite** input this matches candle's `Tensor::var`
/// and `numpy.std(ddof=1)`. Non-finite entries (`NaN`, `±∞`) are *all* skipped so
/// one bad reward cannot poison the group — a deliberate hardening over TRL's
/// mainline reward path, whose plain `.std()` would propagate them (TRL's
/// `nanstd` runs only on its non-default multi-reward aggregation path). Fewer
/// than two finite entries give std `0` (no `0/0`), and none gives `(0, 0)`.
#[must_use]
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = finite.iter().sum::<f64>() / n as f64;
    let std = if n < 2 {
        0.0
    } else {
        let var = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        var.sqrt()
    };
    (mean, std)
}

/// Group-normalized advantages for one GRPO group.
///
/// - [`ScaleRewards::Group`]: `A_i = (r_i - mean_g) / (std_g + eps)`.
/// - [`ScaleRewards::None`]:  `A_i = r_i - mean_g` (centered only).
///
/// `rewards` is one GRPO *group* (the completions sampled for a single prompt).
/// The returned vector has the same length and order as `rewards`. A group of
/// identical rewards — or a single completion — yields all-zero advantages.
///
/// Non-finite rewards are handled defensively: a `NaN`/`±∞` reward is excluded
/// from the group mean/std and its own advantage is forced to `0` — note this
/// maps `±∞ → 0`, *stronger* than `torch.nan_to_num` — so one bad completion
/// neither crashes the step nor distorts the rest of the group.
///
/// # Panics
///
/// Panics if `rewards` is empty: a GRPO group always has at least one
/// completion, and an empty group is a caller bug, not a runtime condition.
#[must_use]
pub fn group_advantages(rewards: &[f64], scale: ScaleRewards) -> Vec<f64> {
    assert!(!rewards.is_empty(), "group_advantages: empty reward group");
    let (mean, std) = mean_std(rewards);
    let denom = match scale {
        ScaleRewards::None => 1.0,
        ScaleRewards::Group => std + GROUP_STD_EPS,
    };
    rewards
        .iter()
        .map(|&r| {
            let a = (r - mean) / denom;
            // Force any non-finite advantage (NaN or ±inf reward) to 0; this is
            // stronger than torch.nan_to_num, which maps ±inf to finite extremes.
            if a.is_finite() {
                a
            } else {
                0.0
            }
        })
        .collect()
}

/// Schulman's k3 KL estimator for a single token: `exp(d) - d - 1`, where
/// `d = logp_ref - logp`.
///
/// `logp` is the current policy's log-probability of the sampled token and
/// `logp_ref` the reference (frozen) policy's. The estimator is non-negative,
/// unbiased for the true KL, and lower-variance than the naive `logp - logp_ref`.
///
/// Following TRL it is intentionally **unclamped**: for a very large
/// `d = logp_ref - logp` (`> ~709`) `exp(d)` overflows to `+∞`, and the estimate
/// is asymmetric (a large *negative* `d` stays finite). Realistic sampled-token
/// log-probs do not reach that regime, and `+∞` is the faithful limit, so it is
/// not sanitized here; the telemetry layer sanitizes non-finite metrics on write
/// (see [`crate::Metrics::nan_to_num`]).
#[must_use]
pub fn k3_kl(logp: f64, logp_ref: f64) -> f64 {
    let d = logp_ref - logp;
    d.exp() - d - 1.0
}

/// PPO-style clipped surrogate for a single token:
/// `min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A)`.
///
/// `ratio` is `exp(logp - logp_old)` (the importance ratio), `advantage` the
/// token's group advantage, and `eps_low` / `eps_high` the trust-region
/// half-widths below and above `1`. Symmetric PPO/GRPO passes the same value
/// for both (e.g. `0.2`); DAPO's **clip-higher** (the standard entropy-control
/// mechanism) widens only the upper band (e.g. `0.2 / 0.28`), letting
/// low-probability tokens grow further before the clip binds — mirroring
/// TRL's `epsilon` / `epsilon_high` pair. The `min` makes the objective
/// pessimistic: it ignores ratio moves that would *improve* the surrogate
/// beyond the trust region.
#[must_use]
pub fn clipped_surrogate(ratio: f64, advantage: f64, eps_low: f64, eps_high: f64) -> f64 {
    let unclipped = ratio * advantage;
    let clipped = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * advantage;
    unclipped.min(clipped)
}

/// Mask-weighted reduction of a per-token objective, dispatched on [`LossType`].
///
/// `values` and `mask` are row-major `[num_sequences][max_len]` matrices of the
/// same shape; `mask[i][j]` is `1.0` for a real token and `0.0` for padding.
///
/// - [`LossType::Grpo`]: for each sequence, average `values` over its valid
///   tokens, then average those per-sequence means over the sequences.
/// - [`LossType::DrGrpo`]: sum every masked contribution and divide by
///   `num_sequences * max_len`.
/// - [`LossType::Dapo`]: sum every masked contribution and divide by the
///   total number of active tokens `max(Σ mask, 1)` — the token-level batch
///   normalizer. (The trainer generalizes the denominator to the accumulation
///   window's total completion tokens; this single-batch form is the
///   `grad_accum_steps == 1` case.)
///
/// An all-pad sequence (a zero mask row) is tolerated in **all** reductions: it
/// contributes `0` and — for `Grpo`/`DrGrpo` — is still counted in the
/// denominator, mirroring TRL's per-sequence `clamp(min=1)`. Use
/// [`zero_mask_rows`] to count such rows for telemetry, since they are
/// otherwise silent. A non-finite value sitting in a masked-out position is
/// ignored (it cannot leak `NaN` via `0 · v`).
///
/// `mask` entries must be finite and non-negative (the contract is `0.0`/`1.0`),
/// and rows must be non-empty (`max_len ≥ 1`).
///
/// # Panics
///
/// Panics if `values` and `mask` differ in shape (row count or any row length),
/// if `values` is empty, if any row is zero-width (`max_len == 0`), or if any
/// mask entry is negative or non-finite.
#[must_use]
pub fn masked_mean(values: &[Vec<f64>], mask: &[Vec<f64>], loss_type: LossType) -> f64 {
    let (num_seq, max_len) = validate_masked_inputs(values, mask);
    match loss_type {
        LossType::Grpo => masked_mean_grpo(values, mask),
        LossType::DrGrpo => masked_mean_dr_grpo(values, mask, num_seq, max_len),
        LossType::Dapo => masked_mean_dapo(values, mask),
    }
}

/// Validate the shared [`masked_mean`] preconditions and return
/// `(num_sequences, max_len)`. Split out to keep [`masked_mean`] under the
/// cognitive-complexity bound.
fn validate_masked_inputs(values: &[Vec<f64>], mask: &[Vec<f64>]) -> (usize, usize) {
    assert!(!values.is_empty(), "masked_mean: empty batch");
    assert_eq!(values.len(), mask.len(), "masked_mean: row count mismatch");
    let num_seq = values.len();
    let max_len = values[0].len();
    // A zero-width row makes the Dr.GRPO denominator (num_seq * max_len) zero.
    assert!(max_len > 0, "masked_mean: zero-width rows (max_len == 0)");
    // Require a rectangular [num_seq][max_len] shape with finite, non-negative
    // masks: a ragged row would make the Dr.GRPO denominator silently wrong, and
    // a negative/non-finite mask weight would desync the denominator from the
    // zero_mask_rows / dropped_rows telemetry.
    for (row, mrow) in values.iter().zip(mask.iter()) {
        assert_eq!(row.len(), max_len, "masked_mean: ragged values row");
        assert_eq!(mrow.len(), max_len, "masked_mean: ragged mask row");
        assert_nonneg_mask(mrow);
    }
    (num_seq, max_len)
}

/// Assert every entry of one mask row is finite and non-negative. Split out of
/// [`masked_mean`] to keep it under the cognitive-complexity bound.
fn assert_nonneg_mask(mrow: &[f64]) {
    for &m in mrow {
        assert!(
            m.is_finite() && m >= 0.0,
            "masked_mean: mask entries must be finite and non-negative"
        );
    }
}

/// Classic-GRPO reduction: per-sequence mean over valid tokens, then mean over
/// sequences. Split out to keep [`masked_mean`] under the cognitive-complexity
/// bound.
fn masked_mean_grpo(values: &[Vec<f64>], mask: &[Vec<f64>]) -> f64 {
    let num_seq = values.len();
    let mut acc = 0.0;
    for (row, mrow) in values.iter().zip(mask.iter()) {
        assert_eq!(row.len(), mrow.len(), "masked_mean: column count mismatch");
        // Clamp the per-sequence denominator to >= 1 (mirrors TRL): an all-pad
        // row then contributes 0 / 1 = 0 instead of dividing by zero.
        let denom: f64 = mrow.iter().sum::<f64>().max(1.0);
        // Hard-zero masked-out cells so a non-finite value at a padding position
        // (0 · v) cannot leak NaN into the sum.
        let s: f64 = row
            .iter()
            .zip(mrow.iter())
            .map(|(v, m)| if *m > 0.0 { v * m } else { 0.0 })
            .sum();
        acc += s / denom;
    }
    acc / num_seq as f64
}

/// Dr.GRPO reduction: total masked contribution over a fixed
/// `num_seq * max_len` denominator. Split out to mirror [`masked_mean_grpo`].
fn masked_mean_dr_grpo(
    values: &[Vec<f64>],
    mask: &[Vec<f64>],
    num_seq: usize,
    max_len: usize,
) -> f64 {
    masked_total(values, mask) / (num_seq as f64 * max_len as f64)
}

/// DAPO reduction: total masked contribution over the total active-token count
/// (clamped to `>= 1`, mirroring TRL's `mask.sum().clamp(min=1.0)` so an
/// all-pad batch yields `0`, not `0/0`). Split out to mirror
/// [`masked_mean_grpo`].
fn masked_mean_dapo(values: &[Vec<f64>], mask: &[Vec<f64>]) -> f64 {
    let active: f64 = mask.iter().flatten().sum::<f64>().max(1.0);
    masked_total(values, mask) / active
}

/// Sum of every masked contribution `v · m` over the batch, hard-zeroing
/// masked-out cells (see [`masked_mean_grpo`]) so `0 · non-finite` cannot leak
/// `NaN`. Shared by the fixed-denominator reductions.
fn masked_total(values: &[Vec<f64>], mask: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for (row, mrow) in values.iter().zip(mask.iter()) {
        assert_eq!(row.len(), mrow.len(), "masked_mean: column count mismatch");
        total += row
            .iter()
            .zip(mrow.iter())
            .map(|(v, m)| if *m > 0.0 { v * m } else { 0.0 })
            .sum::<f64>();
    }
    total
}

/// Count the all-pad sequences in `mask` — rows whose mask sums to zero (no
/// valid tokens).
///
/// [`masked_mean`] tolerates such rows (they contribute `0`), so they are
/// otherwise invisible. The trainer records this count as
/// [`crate::Metrics::dropped_rows`] so a batch that silently lost completions is
/// observable rather than a silent loss of signal.
#[must_use]
pub fn zero_mask_rows(mask: &[Vec<f64>]) -> usize {
    mask.iter()
        .filter(|row| row.iter().sum::<f64>() <= 0.0)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use serde::Deserialize;

    const TOL: f64 = 1e-9;

    // ---- hand-computed unit tests -----------------------------------------

    #[test]
    fn advantages_center_and_scale() {
        // rewards [1, 0, 0.5, 0.5]: mean = 0.5; deviations [0.5, -0.5, 0, 0].
        // SAMPLE var = (0.25 + 0.25) / (4 - 1) = 1/6; std = sqrt(1/6) ~ 0.40825.
        let adv = group_advantages(&[1.0, 0.0, 0.5, 0.5], ScaleRewards::Group);
        let std = (0.5_f64 / 3.0).sqrt();
        let denom = std + GROUP_STD_EPS;
        assert_relative_eq!(adv[0], 0.5 / denom, epsilon = TOL);
        assert_relative_eq!(adv[1], -0.5 / denom, epsilon = TOL);
        assert_relative_eq!(adv[2], 0.0, epsilon = TOL);
        assert_relative_eq!(adv[3], 0.0, epsilon = TOL);
        // Advantages of a group sum to (numerically) zero.
        assert_relative_eq!(adv.iter().sum::<f64>(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn advantages_unscaled_is_centered_only() {
        // ScaleRewards::None: advantage = r - mean, no std division.
        let adv = group_advantages(&[1.0, 0.0, 0.5, 0.5], ScaleRewards::None);
        assert_relative_eq!(adv[0], 0.5, epsilon = TOL);
        assert_relative_eq!(adv[1], -0.5, epsilon = TOL);
        assert_relative_eq!(adv[2], 0.0, epsilon = TOL);
        assert_relative_eq!(adv[3], 0.0, epsilon = TOL);
    }

    #[test]
    fn advantages_single_element_is_zero() {
        // n < 2 -> std = 0 -> advantage = 0 / eps = 0 (no NaN from /(n-1)).
        assert_relative_eq!(
            group_advantages(&[5.0], ScaleRewards::Group)[0],
            0.0,
            epsilon = TOL
        );
        assert_relative_eq!(
            group_advantages(&[5.0], ScaleRewards::None)[0],
            0.0,
            epsilon = TOL
        );
    }

    #[test]
    fn advantages_degenerate_group_is_zero() {
        // All equal -> std = 0 -> every advantage is 0 / eps = 0 (finite).
        let adv = group_advantages(&[2.0, 2.0, 2.0], ScaleRewards::Group);
        for a in adv {
            assert_relative_eq!(a, 0.0, epsilon = TOL);
        }
    }

    #[test]
    #[should_panic(expected = "empty reward group")]
    fn advantages_empty_panics() {
        let _ = group_advantages(&[], ScaleRewards::Group);
    }

    #[test]
    fn advantages_nan_reward_is_isolated() {
        // A NaN reward must not poison the group: it gets a 0 advantage, and the
        // finite rewards are normalized among themselves (mean/std over finite).
        let adv = group_advantages(&[1.0, f64::NAN, 3.0], ScaleRewards::Group);
        assert_eq!(adv[1], 0.0, "NaN reward must map to a 0 advantage");
        assert!(adv[0].is_finite() && adv[2].is_finite());
        // finite = [1, 3]: mean 2, so advantages are antisymmetric about 0.
        assert_relative_eq!(adv[0], -adv[2], epsilon = TOL);
        assert!(adv[0] < 0.0 && adv[2] > 0.0);
    }

    #[test]
    fn advantages_all_nan_group_is_all_zero() {
        // No finite entries -> (mean, std) = (0, 0) -> every advantage is 0.
        let adv = group_advantages(&[f64::NAN, f64::NAN], ScaleRewards::Group);
        assert_eq!(adv, vec![0.0, 0.0]);
    }

    #[test]
    fn advantages_infinite_reward_maps_to_zero() {
        // ±inf is excluded from the stats and its own advantage is forced to 0.
        let adv = group_advantages(&[1.0, f64::INFINITY, 3.0], ScaleRewards::None);
        assert_eq!(adv[1], 0.0);
        // finite = [1, 3], mean 2 -> centered advantages -1 and +1.
        assert_relative_eq!(adv[0], -1.0, epsilon = TOL);
        assert_relative_eq!(adv[2], 1.0, epsilon = TOL);
    }

    #[test]
    fn k3_is_zero_when_equal_and_nonnegative() {
        assert_relative_eq!(k3_kl(-1.0, -1.0), 0.0, epsilon = TOL);
        // d = 0.1: exp(0.1) - 0.1 - 1 = 0.00517091807...
        assert_relative_eq!(k3_kl(-0.5, -0.4), 0.005_170_918_075_647_624, epsilon = TOL);
        // k3 is non-negative for any inputs.
        for (lp, lr) in [(-2.0, 1.0), (3.0, -1.0), (0.0, 0.0), (0.7, 0.2)] {
            assert!(k3_kl(lp, lr) >= 0.0);
        }
    }

    #[test]
    fn surrogate_no_clip_when_ratio_one() {
        assert_relative_eq!(clipped_surrogate(1.0, 0.5, 0.2, 0.2), 0.5, epsilon = TOL);
        assert_relative_eq!(clipped_surrogate(1.0, -0.5, 0.2, 0.2), -0.5, epsilon = TOL);
    }

    #[test]
    fn surrogate_clips_positive_advantage_above_band() {
        // ratio 1.5 > 1.2, A > 0: min(1.5*0.5, 1.2*0.5) = min(0.75, 0.6) = 0.6.
        assert_relative_eq!(clipped_surrogate(1.5, 0.5, 0.2, 0.2), 0.6, epsilon = TOL);
    }

    #[test]
    fn surrogate_keeps_unclipped_for_negative_advantage_above_band() {
        // ratio 1.5, A < 0: min(1.5*-0.5, 1.2*-0.5) = min(-0.75, -0.6) = -0.75.
        assert_relative_eq!(clipped_surrogate(1.5, -0.5, 0.2, 0.2), -0.75, epsilon = TOL);
    }

    #[test]
    fn surrogate_clips_below_band() {
        // ratio 0.5 < 0.8, A < 0: min(0.5*-0.5, 0.8*-0.5) = min(-0.25, -0.4) = -0.4.
        assert_relative_eq!(clipped_surrogate(0.5, -0.5, 0.2, 0.2), -0.4, epsilon = TOL);
        // ratio 0.8 < ... actually 0.8 is the lower band edge; A > 0:
        // min(0.8*0.5, 0.8*0.5) = 0.4.
        assert_relative_eq!(clipped_surrogate(0.8, 0.5, 0.2, 0.2), 0.4, epsilon = TOL);
    }

    #[test]
    fn surrogate_clip_higher_widens_only_the_upper_band() {
        // DAPO clip-higher 0.2/0.28: ratio 1.25 is inside the widened upper band
        // (1.28) so the surrogate is unclipped — symmetric 0.2 would clip at 1.2.
        assert_relative_eq!(
            clipped_surrogate(1.25, 0.5, 0.2, 0.28),
            0.625,
            epsilon = TOL
        );
        assert_relative_eq!(clipped_surrogate(1.25, 0.5, 0.2, 0.2), 0.6, epsilon = TOL);
        // ratio 1.5 still clips, now at the 1.28 edge.
        assert_relative_eq!(clipped_surrogate(1.5, 0.5, 0.2, 0.28), 0.64, epsilon = TOL);
        // The lower band is untouched by eps_high: ratio 0.5, A < 0 clips at 0.8.
        assert_relative_eq!(clipped_surrogate(0.5, -0.5, 0.2, 0.28), -0.4, epsilon = TOL);
    }

    #[test]
    fn sequence_log_ratio_is_masked_mean_of_token_log_ratios() {
        // logp - logp_old = [0.2, -0.4, 0.6]; mask keeps the first two:
        // (0.2 - 0.4) / 2 = -0.1.
        let logp = [-1.0, -2.0, -0.4];
        let old = [-1.2, -1.6, -1.0];
        let mask = [1.0, 1.0, 0.0];
        assert_relative_eq!(sequence_log_ratio(&logp, &old, &mask), -0.1, epsilon = TOL);
        // Identical logps -> log-ratio 0 -> sequence ratio exp(0) = 1.
        assert_relative_eq!(sequence_log_ratio(&logp, &logp, &mask), 0.0, epsilon = TOL);
    }

    #[test]
    fn sequence_log_ratio_all_pad_row_is_zero_and_padding_cannot_poison() {
        // All-pad row: clamp(denom, 1) -> 0/1 = 0 (ratio 1), mirroring TRL.
        assert_relative_eq!(
            sequence_log_ratio(&[5.0, -3.0], &[0.0, 0.0], &[0.0, 0.0]),
            0.0,
            epsilon = TOL
        );
        // A non-finite logp at a masked-out position is ignored.
        assert_relative_eq!(
            sequence_log_ratio(&[-1.0, f64::NAN], &[-1.5, 0.0], &[1.0, 0.0]),
            0.5,
            epsilon = TOL
        );
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn sequence_log_ratio_rejects_ragged_inputs() {
        let _ = sequence_log_ratio(&[1.0, 2.0], &[1.0], &[1.0, 1.0]);
    }

    #[test]
    fn masked_mean_grpo_averages_per_sequence() {
        let v = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 0.0]];
        let m = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 0.0]];
        // seq0 mean = 6/3 = 2; seq1 mean = 9/2 = 4.5; group mean = 3.25.
        assert_relative_eq!(masked_mean(&v, &m, LossType::Grpo), 3.25, epsilon = TOL);
    }

    #[test]
    fn masked_mean_dr_grpo_uses_fixed_denominator() {
        let v = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 0.0]];
        let m = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 0.0]];
        // total = 1+2+3+4+5 = 15; denom = 2*3 = 6; 15/6 = 2.5.
        assert_relative_eq!(masked_mean(&v, &m, LossType::DrGrpo), 2.5, epsilon = TOL);
    }

    #[test]
    fn masked_mean_dapo_divides_by_active_tokens() {
        let v = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 0.0]];
        let m = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 0.0]];
        // total = 15; active tokens = 5; 15/5 = 3.
        assert_relative_eq!(masked_mean(&v, &m, LossType::Dapo), 3.0, epsilon = TOL);
        // All-pad batch: clamp(active, 1) -> 0/1 = 0 (mirrors TRL), no 0/0.
        let v0 = vec![vec![7.0, 7.0]];
        let m0 = vec![vec![0.0, 0.0]];
        assert_relative_eq!(masked_mean(&v0, &m0, LossType::Dapo), 0.0, epsilon = TOL);
    }

    #[test]
    fn dapo_equals_grpo_on_full_width_equal_length_batches() {
        // With every row fully unmasked at the same width, both reduce to
        // total / (num_seq * len) — the property that makes the Dapo default
        // switch bit-preserving for full-width rollouts.
        let v = vec![vec![1.0, -2.0, 3.5], vec![0.25, 5.0, -1.0]];
        let m = vec![vec![1.0; 3]; 2];
        assert_relative_eq!(
            masked_mean(&v, &m, LossType::Dapo),
            masked_mean(&v, &m, LossType::Grpo),
            epsilon = 1e-12
        );
        // And they differ once lengths vary (the length bias being removed).
        let mv = vec![vec![1.0, 1.0, 1.0], vec![1.0, 0.0, 0.0]];
        assert!(
            (masked_mean(&v, &mv, LossType::Dapo) - masked_mean(&v, &mv, LossType::Grpo)).abs()
                > 1e-6
        );
    }

    #[test]
    fn masked_mean_tolerates_zero_mask_row_in_both_reductions() {
        // An all-pad sequence contributes 0 and is still counted in the
        // denominator (TRL clamp(min=1)); neither reduction panics.
        let v = vec![vec![3.0, 9.0], vec![0.0, 0.0], vec![6.0, 6.0]];
        let m = vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![1.0, 1.0]];
        // grpo: per-seq means 3, 0, 6 -> (3 + 0 + 6) / 3 = 3.
        assert_relative_eq!(masked_mean(&v, &m, LossType::Grpo), 3.0, epsilon = TOL);
        // dr_grpo: total 3 + 0 + 12 = 15 over 3 * 2 = 6 -> 2.5.
        assert_relative_eq!(masked_mean(&v, &m, LossType::DrGrpo), 2.5, epsilon = TOL);
    }

    #[test]
    fn zero_mask_rows_counts_all_pad_sequences() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_eq!(zero_mask_rows(&m), 2);
    }

    #[test]
    #[should_panic(expected = "zero-width rows")]
    fn masked_mean_zero_width_batch_panics() {
        // max_len == 0 would make the Dr.GRPO denominator (num_seq * max_len) 0.
        let v: Vec<Vec<f64>> = vec![vec![]];
        let m: Vec<Vec<f64>> = vec![vec![]];
        let _ = masked_mean(&v, &m, LossType::DrGrpo);
    }

    #[test]
    #[should_panic(expected = "finite and non-negative")]
    fn masked_mean_rejects_negative_mask() {
        // A negative mask weight would desync the denominator from zero_mask_rows.
        let v = vec![vec![1.0, 2.0]];
        let m = vec![vec![1.0, -1.0]];
        let _ = masked_mean(&v, &m, LossType::Grpo);
    }

    #[test]
    fn masked_mean_ignores_nonfinite_value_in_masked_cell() {
        // A NaN/inf value in a masked-out (m == 0) position must not leak into
        // either reduction: the mask is the source of truth for what counts.
        let v = vec![vec![1.0, f64::NAN], vec![5.0, f64::INFINITY]];
        let m = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        // grpo: each valid token alone -> per-seq means 1 and 5 -> (1 + 5) / 2 = 3.
        assert_relative_eq!(masked_mean(&v, &m, LossType::Grpo), 3.0, epsilon = TOL);
        // dr_grpo: total 1 + 5 = 6 over 2 * 2 = 4 -> 1.5.
        assert_relative_eq!(masked_mean(&v, &m, LossType::DrGrpo), 1.5, epsilon = TOL);
    }

    #[test]
    #[should_panic(expected = "row count mismatch")]
    fn masked_mean_shape_mismatch_panics() {
        let v = vec![vec![1.0, 2.0]];
        let m = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let _ = masked_mean(&v, &m, LossType::Grpo);
    }

    #[test]
    #[should_panic(expected = "ragged")]
    fn masked_mean_ragged_row_panics() {
        // A ragged row would silently corrupt the Dr.GRPO denominator.
        let v = vec![vec![1.0], vec![100.0, 100.0]];
        let m = vec![vec![1.0], vec![1.0, 1.0]];
        let _ = masked_mean(&v, &m, LossType::DrGrpo);
    }

    #[test]
    fn loss_type_default_is_dapo() {
        // The R1 default switch (maintainer-confirmed): token-level batch
        // normalizer, matching TRL's shipped default; Grpo stays selectable.
        assert_eq!(LossType::default(), LossType::Dapo);
    }

    #[test]
    fn loss_type_serde_roundtrip() {
        let j = serde_json::to_string(&LossType::DrGrpo).unwrap();
        assert_eq!(j, "\"dr_grpo\"");
        let back: LossType = serde_json::from_str(&j).unwrap();
        assert_eq!(back, LossType::DrGrpo);
        // The wire names match TRL's loss_type strings.
        assert_eq!(serde_json::to_string(&LossType::Dapo).unwrap(), "\"dapo\"");
        assert_eq!(serde_json::to_string(&LossType::Grpo).unwrap(), "\"grpo\"");
        // An old config.json with the explicit legacy default still selects it.
        let old: LossType = serde_json::from_str("\"grpo\"").unwrap();
        assert_eq!(old, LossType::Grpo);
    }

    #[test]
    fn importance_sampling_level_default_and_serde() {
        assert_eq!(
            ImportanceSamplingLevel::default(),
            ImportanceSamplingLevel::Token
        );
        let j = serde_json::to_string(&ImportanceSamplingLevel::Sequence).unwrap();
        assert_eq!(j, "\"sequence\"");
        let back: ImportanceSamplingLevel = serde_json::from_str(&j).unwrap();
        assert_eq!(back, ImportanceSamplingLevel::Sequence);
    }

    // ---- golden-fixture oracle test ---------------------------------------

    #[derive(Deserialize)]
    struct Group {
        rewards: Vec<f64>,
        advantages: Vec<f64>,
        advantages_unscaled: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct KlCase {
        logp: f64,
        logp_ref: f64,
        kl: f64,
    }

    #[derive(Deserialize)]
    struct SurrogateCase {
        ratio: f64,
        advantage: f64,
        eps_low: f64,
        eps_high: f64,
        value: f64,
    }

    #[derive(Deserialize)]
    struct MaskedMeanCase {
        per_token: Vec<Vec<f64>>,
        mask: Vec<Vec<f64>>,
        grpo: f64,
        dr_grpo: f64,
        dapo: f64,
    }

    #[derive(Deserialize)]
    struct SeqLogRatioCase {
        logp: Vec<f64>,
        logp_old: Vec<f64>,
        mask: Vec<f64>,
        value: f64,
    }

    #[derive(Deserialize)]
    struct Golden {
        eps_std: f64,
        groups: Vec<Group>,
        k3_kl: Vec<KlCase>,
        clipped_surrogate: Vec<SurrogateCase>,
        masked_mean: MaskedMeanCase,
        sequence_log_ratio: Vec<SeqLogRatioCase>,
    }

    fn check_golden_advantages(groups: &[Group]) {
        for group in groups {
            let scaled = group_advantages(&group.rewards, ScaleRewards::Group);
            let unscaled = group_advantages(&group.rewards, ScaleRewards::None);
            assert_eq!(scaled.len(), group.advantages.len());
            for (a, b) in scaled.iter().zip(group.advantages.iter()) {
                assert_relative_eq!(a, b, epsilon = 1e-9);
            }
            for (a, b) in unscaled.iter().zip(group.advantages_unscaled.iter()) {
                assert_relative_eq!(a, b, epsilon = 1e-9);
            }
        }
    }

    fn check_golden_kl(cases: &[KlCase]) {
        for c in cases {
            assert_relative_eq!(k3_kl(c.logp, c.logp_ref), c.kl, epsilon = 1e-9);
        }
    }

    fn check_golden_surrogate(cases: &[SurrogateCase]) {
        for c in cases {
            assert_relative_eq!(
                clipped_surrogate(c.ratio, c.advantage, c.eps_low, c.eps_high),
                c.value,
                epsilon = 1e-9
            );
        }
    }

    fn check_golden_masked_mean(mm: &MaskedMeanCase) {
        assert_relative_eq!(
            masked_mean(&mm.per_token, &mm.mask, LossType::Grpo),
            mm.grpo,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            masked_mean(&mm.per_token, &mm.mask, LossType::DrGrpo),
            mm.dr_grpo,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            masked_mean(&mm.per_token, &mm.mask, LossType::Dapo),
            mm.dapo,
            epsilon = 1e-9
        );
    }

    fn check_golden_seq_log_ratio(cases: &[SeqLogRatioCase]) {
        for c in cases {
            assert_relative_eq!(
                sequence_log_ratio(&c.logp, &c.logp_old, &c.mask),
                c.value,
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn matches_golden_fixture() {
        // Committed by scripts/gen_golden.py — a NumPy (ddof=1) reference oracle.
        let raw = include_str!("../tests/fixtures/grpo_golden.json");
        let g: Golden = serde_json::from_str(raw).expect("golden fixture parses");

        // The fixture's eps must equal our constant, or the oracle drifted.
        assert_relative_eq!(g.eps_std, GROUP_STD_EPS, epsilon = 1e-18);
        check_golden_advantages(&g.groups);
        check_golden_kl(&g.k3_kl);
        check_golden_surrogate(&g.clipped_surrogate);
        check_golden_masked_mean(&g.masked_mean);
        check_golden_seq_log_ratio(&g.sequence_log_ratio);
    }

    // ---- proptest numeric invariants --------------------------------------

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_advantages_are_zero_mean(rewards in prop::collection::vec(-100.0f64..100.0, 1..16)) {
            let adv = group_advantages(&rewards, ScaleRewards::Group);
            let sum: f64 = adv.iter().sum();
            // Mean-centered, so the advantages sum to ~0 (scaled by 1/(std+eps)).
            prop_assert!(sum.abs() < 1e-6);
        }

        #[test]
        fn prop_k3_kl_nonnegative(logp in -20.0f64..20.0, logp_ref in -20.0f64..20.0) {
            prop_assert!(k3_kl(logp, logp_ref) >= 0.0);
        }

        #[test]
        fn prop_surrogate_bounded_by_unclipped_sign(
            ratio in 0.01f64..5.0,
            adv in -10.0f64..10.0,
            eps_low in 0.01f64..0.5,
            eps_high in 0.01f64..0.5,
        ) {
            let s = clipped_surrogate(ratio, adv, eps_low, eps_high);
            let unclipped = ratio * adv;
            // The pessimistic min never exceeds the unclipped objective.
            prop_assert!(s <= unclipped + 1e-9);
        }

        #[test]
        fn prop_widening_eps_high_never_decreases_the_surrogate(
            ratio in 0.01f64..5.0,
            adv in -10.0f64..10.0,
            eps in 0.01f64..0.5,
            extra in 0.0f64..0.5,
        ) {
            // Clip-higher only relaxes the upper clamp, so the pessimistic min
            // is monotonically non-decreasing in eps_high.
            let sym = clipped_surrogate(ratio, adv, eps, eps);
            let wide = clipped_surrogate(ratio, adv, eps, eps + extra);
            prop_assert!(wide >= sym - 1e-12);
        }

        #[test]
        fn prop_dapo_equals_grpo_at_full_width(
            rows in prop::collection::vec(
                prop::collection::vec(-10.0f64..10.0, 4),
                1..6,
            ),
        ) {
            // Full-width equal-length batches: the Dapo and Grpo reductions
            // coincide (total / (num_seq * len)) — the default-switch
            // bit-preservation property.
            let mask = vec![vec![1.0; 4]; rows.len()];
            let d = masked_mean(&rows, &mask, LossType::Dapo);
            let g = masked_mean(&rows, &mask, LossType::Grpo);
            prop_assert!((d - g).abs() < 1e-9);
        }
    }
}
