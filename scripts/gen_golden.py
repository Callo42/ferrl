#!/usr/bin/env python3
"""Generate the GRPO golden-math fixture consumed by the Rust unit tests.

This is the *oracle*: a reference implementation of the GRPO quantities that
``crates/ferrl/src/grpo.rs`` re-derives in Rust, with the contested group
statistic computed by **NumPy** rather than a hand-rolled copy of the Rust
formula. The advantage std uses NumPy's ``std(ddof=1)`` (sample / Bessel-
corrected), which — for the all-finite rewards in this fixture — matches TRL's
``nanstd`` and candle's ``Tensor::var``, so a population-vs-sample regression in
the Rust code would make this fixture disagree (which a same-formula
reimplementation could not catch). Non-finite rewards are handled by the Rust
code (``grpo::group_advantages`` skips NaN *and* ±inf), not exercised here.

Formulas mirror TRL's ``GRPOTrainer`` / the DeepSeekMath paper:

* group-normalized advantages  A_i = (r_i - mean_g) / (std_g + eps), eps = 1e-4
  (scale="group", default); A_i = r_i - mean_g (scale="none", Dr.GRPO-recommended)
* k3 (Schulman) KL estimator   k3 = exp(d) - d - 1  where d = logp_ref - logp
* per-token clipped surrogate  min(ratio * A, clip(ratio, 1-e_low, 1+e_high) * A)
  (asymmetric bands per DAPO clip-higher; symmetric when e_low == e_high)
* masked mean reductions        "grpo"   -> mean over max(valid tokens, 1) per-seq
                                "dr_grpo"-> sum / (num_seq * max_len)  (Dr.GRPO)
                                "dapo"   -> sum / max(total active tokens, 1)
* sequence log-ratio (GSPO)     (sum_t m_t * (logp_t - old_t)) / max(sum_t m_t, 1)

Output is committed at ``crates/ferrl/tests/fixtures/grpo_golden.json`` and loaded
verbatim by ``grpo::tests::matches_golden_fixture``. **Requires NumPy** to
regenerate (CI never runs this script — it only loads the committed JSON):

    python3 scripts/gen_golden.py > crates/ferrl/tests/fixtures/grpo_golden.json
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

EPS_STD = 1e-4


def group_advantages(rewards: list[float], scale: str = "group") -> list[float]:
    """Group-normalized advantages. ``scale`` is "group" (std-scaled) or "none"."""
    r = np.asarray(rewards, dtype=np.float64)
    centered = r - r.mean()
    if scale == "none":
        return centered.tolist()
    # Sample std (ddof=1), matching TRL nanstd / candle Tensor::var. n < 2 -> 0.
    std = float(r.std(ddof=1)) if r.size > 1 else 0.0
    return (centered / (std + EPS_STD)).tolist()


def k3_kl(logp: float, logp_ref: float) -> float:
    d = logp_ref - logp
    return math.exp(d) - d - 1.0


def clipped_surrogate(ratio: float, adv: float, eps_low: float, eps_high: float) -> float:
    unclipped = ratio * adv
    clipped = max(1.0 - eps_low, min(1.0 + eps_high, ratio)) * adv
    return min(unclipped, clipped)


def sequence_log_ratio(logp: list[float], logp_old: list[float], mask: list[float]) -> float:
    """GSPO per-sequence log-ratio: masked mean of per-token log-ratios."""
    denom = max(sum(mask), 1.0)
    num = sum((lp - lo) * m for lp, lo, m in zip(logp, logp_old, mask) if m > 0.0)
    return num / denom


def group_entry(rewards: list[float]) -> dict:
    return {
        "rewards": rewards,
        "advantages": group_advantages(rewards, "group"),
        "advantages_unscaled": group_advantages(rewards, "none"),
    }


def main() -> None:
    # A small, deterministic scenario: two groups of completions.
    rewards_a = [1.0, 0.0, 0.5, 0.5]
    rewards_b = [2.0, -1.0, 0.0]

    # logprob pairs (policy, reference) for the k3 KL estimator.
    logp_pairs = [
        (-0.5, -0.4),
        (-1.0, -1.0),
        (-2.0, -1.5),
        (0.0, -0.25),
    ]

    # ratio / advantage / clip-band scenarios for the surrogate. The symmetric
    # 0.2/0.2 rows are the classic PPO cases; the 0.2/0.28 rows pin DAPO
    # clip-higher (only the upper band widens).
    surrogate_cases = [
        (1.0, 0.5, 0.2, 0.2),    # ratio == 1, no clip
        (1.5, 0.5, 0.2, 0.2),    # ratio above 1+eps, positive adv -> clipped
        (1.5, -0.5, 0.2, 0.2),   # ratio above 1+eps, negative adv -> unclipped is smaller
        (0.5, -0.5, 0.2, 0.2),   # ratio below 1-eps, negative adv -> clipped
        (0.8, 0.5, 0.2, 0.2),    # ratio below 1-eps, positive adv -> unclipped is smaller
        (1.25, 0.5, 0.2, 0.28),  # inside the widened upper band -> unclipped
        (1.5, 0.5, 0.2, 0.28),   # above the widened upper band -> clipped at 1.28
        (0.5, -0.5, 0.2, 0.28),  # lower band untouched by eps_high -> clipped at 0.8
    ]

    # GSPO sequence-level log-ratio cases (logp, logp_old, mask).
    seq_log_ratio_cases = [
        ([-1.0, -2.0, -0.4], [-1.2, -1.6, -1.0], [1.0, 1.0, 0.0]),
        ([-0.5, -0.25], [-0.5, -0.25], [1.0, 1.0]),      # identical -> 0
        ([-3.0, -1.0], [-1.0, -2.0], [1.0, 1.0]),        # mixed signs
        ([5.0, -3.0], [0.0, 0.0], [0.0, 0.0]),           # all-pad -> 0 (clamp)
    ]

    # masked-mean scenario: 2 sequences, padded to max_len = 3.
    per_token = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 0.0],
    ]
    mask = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    valid = sum(sum(m) for m in mask)
    num_seq = len(per_token)
    max_len = len(per_token[0])

    # grpo: average over valid tokens per-sequence, then average over sequences.
    per_seq = []
    for row, mrow in zip(per_token, mask):
        denom = max(sum(mrow), 1.0)  # TRL clamp(min=1): an all-pad row contributes 0
        s = sum(v * m for v, m in zip(row, mrow))
        per_seq.append(s / denom)
    masked_mean_grpo = sum(per_seq) / num_seq

    # dr_grpo: sum of all valid contributions / (num_seq * max_len).
    total = sum(v * m for row, mrow in zip(per_token, mask) for v, m in zip(row, mrow))
    masked_mean_dr_grpo = total / (num_seq * max_len)

    # dapo: sum of all valid contributions / max(total active tokens, 1).
    masked_mean_dapo = total / max(valid, 1.0)

    fixture = {
        "_comment": "Golden GRPO math oracle (NumPy ddof=1). Regenerate via scripts/gen_golden.py.",
        "eps_std": EPS_STD,
        "groups": [group_entry(rewards_a), group_entry(rewards_b)],
        "k3_kl": [
            {"logp": lp, "logp_ref": lr, "kl": k3_kl(lp, lr)}
            for (lp, lr) in logp_pairs
        ],
        "clipped_surrogate": [
            {
                "ratio": r,
                "advantage": a,
                "eps_low": el,
                "eps_high": eh,
                "value": clipped_surrogate(r, a, el, eh),
            }
            for (r, a, el, eh) in surrogate_cases
        ],
        "masked_mean": {
            "per_token": per_token,
            "mask": mask,
            "valid_tokens": valid,
            "num_seq": num_seq,
            "max_len": max_len,
            "grpo": masked_mean_grpo,
            "dr_grpo": masked_mean_dr_grpo,
            "dapo": masked_mean_dapo,
        },
        "sequence_log_ratio": [
            {
                "logp": lp,
                "logp_old": lo,
                "mask": m,
                "value": sequence_log_ratio(lp, lo, m),
            }
            for (lp, lo, m) in seq_log_ratio_cases
        ],
    }

    json.dump(fixture, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
