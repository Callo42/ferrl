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
//!   the group — matching TRL's `nanstd` and candle's `Tensor::var`. Dividing by
//!   the std is the [`ScaleRewards::Group`] default; [`ScaleRewards::None`] drops
//!   it (the Dr.GRPO-recommended setting).
//! - **KL** uses Schulman's k3 estimator `exp(d) - d - 1`, `d = logp_ref - logp`,
//!   which is non-negative and unbiased.
//! - **Surrogate** is the PPO-style clipped objective
//!   `min(ratio * A, clip(ratio, 1 - e, 1 + e) * A)`.
//! - **Reduction** ([`LossType`]) is length-normalization only: classic GRPO
//!   averages per sequence then over sequences; Dr.GRPO divides the total by
//!   `num_sequences * max_len`. The Dr.GRPO *paper* algorithm is
//!   `LossType::DrGrpo` **plus** [`ScaleRewards::None`].

use serde::{Deserialize, Serialize};

/// Numerical-stability epsilon added to the group standard deviation when
/// normalizing advantages: `A_i = (r_i - mean) / (std + eps)`.
///
/// Matches TRL's default (`1e-4`) and keeps the advantage finite for
/// degenerate groups where every reward is identical (`std = 0`).
pub const GROUP_STD_EPS: f64 = 1e-4;

/// Which GRPO loss reduction to use over the per-token objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossType {
    /// Classic GRPO: average over valid tokens *within* each sequence, then
    /// average those per-sequence means over the group. Short completions are
    /// weighted as heavily as long ones. This is the default.
    #[default]
    Grpo,
    /// Dr.GRPO (<https://arxiv.org/abs/2503.20783>): divide the summed,
    /// mask-weighted objective by `num_sequences * max_len`, removing the
    /// length bias of classic GRPO.
    DrGrpo,
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

/// Sample (Bessel-corrected, `ddof = 1`) mean and standard deviation of a slice.
///
/// Returns `(mean, std)`. The std divides by `n - 1`, matching TRL's `nanstd` and
/// candle's `Tensor::var`; a slice of length `< 2` has std `0` (no `0/0`).
#[must_use]
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let std = if n < 2 {
        0.0
    } else {
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
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
/// # Panics
///
/// Panics if `rewards` is empty: a GRPO group always has at least one
/// completion, and an empty group is a caller bug, not a runtime condition.
#[must_use]
pub fn group_advantages(rewards: &[f64], scale: ScaleRewards) -> Vec<f64> {
    assert!(!rewards.is_empty(), "group_advantages: empty reward group");
    let (mean, std) = mean_std(rewards);
    match scale {
        ScaleRewards::None => rewards.iter().map(|&r| r - mean).collect(),
        ScaleRewards::Group => {
            let denom = std + GROUP_STD_EPS;
            rewards.iter().map(|&r| (r - mean) / denom).collect()
        }
    }
}

/// Schulman's k3 KL estimator for a single token: `exp(d) - d - 1`, where
/// `d = logp_ref - logp`.
///
/// `logp` is the current policy's log-probability of the sampled token and
/// `logp_ref` the reference (frozen) policy's. The estimator is non-negative,
/// unbiased for the true KL, and lower-variance than the naive `logp - logp_ref`.
#[must_use]
pub fn k3_kl(logp: f64, logp_ref: f64) -> f64 {
    let d = logp_ref - logp;
    d.exp() - d - 1.0
}

/// PPO-style clipped surrogate for a single token:
/// `min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)`.
///
/// `ratio` is `exp(logp - logp_old)` (the importance ratio), `advantage` the
/// token's group advantage, and `clip_eps` the trust-region half-width (e.g.
/// `0.2`). The `min` makes the objective pessimistic: it ignores ratio moves
/// that would *improve* the surrogate beyond the trust region.
#[must_use]
pub fn clipped_surrogate(ratio: f64, advantage: f64, clip_eps: f64) -> f64 {
    let unclipped = ratio * advantage;
    let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage;
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
///
/// # Panics
///
/// Panics if `values` and `mask` differ in shape, if `values` is empty, or if a
/// [`LossType::Grpo`] sequence has no valid tokens (zero mask row), which would
/// otherwise divide by zero.
#[must_use]
pub fn masked_mean(values: &[Vec<f64>], mask: &[Vec<f64>], loss_type: LossType) -> f64 {
    assert!(!values.is_empty(), "masked_mean: empty batch");
    assert_eq!(values.len(), mask.len(), "masked_mean: row count mismatch");
    let num_seq = values.len();
    let max_len = values[0].len();
    // Require a rectangular [num_seq][max_len] shape: a ragged row would make the
    // Dr.GRPO denominator (num_seq * max_len) silently wrong.
    for (row, mrow) in values.iter().zip(mask.iter()) {
        assert_eq!(row.len(), max_len, "masked_mean: ragged values row");
        assert_eq!(mrow.len(), max_len, "masked_mean: ragged mask row");
    }
    match loss_type {
        LossType::Grpo => masked_mean_grpo(values, mask),
        LossType::DrGrpo => masked_mean_dr_grpo(values, mask, num_seq, max_len),
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
        let denom: f64 = mrow.iter().sum();
        assert!(
            denom > 0.0,
            "masked_mean(grpo): sequence with no valid tokens"
        );
        let s: f64 = row.iter().zip(mrow.iter()).map(|(v, m)| v * m).sum();
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
    let mut total = 0.0;
    for (row, mrow) in values.iter().zip(mask.iter()) {
        assert_eq!(row.len(), mrow.len(), "masked_mean: column count mismatch");
        total += row.iter().zip(mrow.iter()).map(|(v, m)| v * m).sum::<f64>();
    }
    total / (num_seq as f64 * max_len as f64)
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
        assert_relative_eq!(clipped_surrogate(1.0, 0.5, 0.2), 0.5, epsilon = TOL);
        assert_relative_eq!(clipped_surrogate(1.0, -0.5, 0.2), -0.5, epsilon = TOL);
    }

    #[test]
    fn surrogate_clips_positive_advantage_above_band() {
        // ratio 1.5 > 1.2, A > 0: min(1.5*0.5, 1.2*0.5) = min(0.75, 0.6) = 0.6.
        assert_relative_eq!(clipped_surrogate(1.5, 0.5, 0.2), 0.6, epsilon = TOL);
    }

    #[test]
    fn surrogate_keeps_unclipped_for_negative_advantage_above_band() {
        // ratio 1.5, A < 0: min(1.5*-0.5, 1.2*-0.5) = min(-0.75, -0.6) = -0.75.
        assert_relative_eq!(clipped_surrogate(1.5, -0.5, 0.2), -0.75, epsilon = TOL);
    }

    #[test]
    fn surrogate_clips_below_band() {
        // ratio 0.5 < 0.8, A < 0: min(0.5*-0.5, 0.8*-0.5) = min(-0.25, -0.4) = -0.4.
        assert_relative_eq!(clipped_surrogate(0.5, -0.5, 0.2), -0.4, epsilon = TOL);
        // ratio 0.8 < ... actually 0.8 is the lower band edge; A > 0:
        // min(0.8*0.5, 0.8*0.5) = 0.4.
        assert_relative_eq!(clipped_surrogate(0.8, 0.5, 0.2), 0.4, epsilon = TOL);
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
    #[should_panic(expected = "no valid tokens")]
    fn masked_mean_grpo_zero_mask_row_panics() {
        let v = vec![vec![1.0, 2.0]];
        let m = vec![vec![0.0, 0.0]];
        let _ = masked_mean(&v, &m, LossType::Grpo);
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
    fn loss_type_default_is_grpo() {
        assert_eq!(LossType::default(), LossType::Grpo);
    }

    #[test]
    fn loss_type_serde_roundtrip() {
        let j = serde_json::to_string(&LossType::DrGrpo).unwrap();
        assert_eq!(j, "\"dr_grpo\"");
        let back: LossType = serde_json::from_str(&j).unwrap();
        assert_eq!(back, LossType::DrGrpo);
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
        clip_eps: f64,
        value: f64,
    }

    #[derive(Deserialize)]
    struct MaskedMeanCase {
        per_token: Vec<Vec<f64>>,
        mask: Vec<Vec<f64>>,
        grpo: f64,
        dr_grpo: f64,
    }

    #[derive(Deserialize)]
    struct Golden {
        eps_std: f64,
        groups: Vec<Group>,
        k3_kl: Vec<KlCase>,
        clipped_surrogate: Vec<SurrogateCase>,
        masked_mean: MaskedMeanCase,
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
                clipped_surrogate(c.ratio, c.advantage, c.clip_eps),
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
            eps in 0.01f64..0.5,
        ) {
            let s = clipped_surrogate(ratio, adv, eps);
            let unclipped = ratio * adv;
            // The pessimistic min never exceeds the unclipped objective.
            prop_assert!(s <= unclipped + 1e-9);
        }
    }
}
