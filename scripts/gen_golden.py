#!/usr/bin/env python3
"""Generate the GRPO golden-math fixture consumed by the Rust unit tests.

This is the *oracle*: an independent, dependency-free reference implementation of
the GRPO quantities that ``crates/ferrl/src/grpo.rs`` re-derives in Rust. The
formulas mirror TRL's ``GRPOTrainer`` / the DeepSeekMath paper:

* group-normalized advantages  A_i = (r_i - mean_g) / (std_g + eps), eps = 1e-4
* k3 (Schulman) KL estimator   k3 = exp(d) - d - 1  where d = logp_ref - logp
* per-token clipped surrogate  min(ratio * A, clip(ratio, 1-e, 1+e) * A)
* masked mean reductions        "grpo"   -> mean over valid tokens per-sequence
                                "dr_grpo"-> sum / (num_seq * max_len)  (Dr.GRPO)

The output is committed at ``crates/ferrl/tests/fixtures/grpo_golden.json`` and
loaded verbatim by the Rust test ``grpo::tests::matches_golden_fixture``. Keep
this script and that fixture in lockstep: regenerate with

    python3 scripts/gen_golden.py > crates/ferrl/tests/fixtures/grpo_golden.json

No numpy / torch on purpose, so it runs in any CI container.
"""

from __future__ import annotations

import json
import math
import sys

EPS_STD = 1e-4


def population_std(xs: list[float], mean: float) -> float:
    """Population (biased, ddof=0) standard deviation, matching candle's var()."""
    n = len(xs)
    var = sum((x - mean) ** 2 for x in xs) / n
    return math.sqrt(var)


def group_advantages(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards)
    std = population_std(rewards, mean)
    return [(r - mean) / (std + EPS_STD) for r in rewards]


def k3_kl(logp: float, logp_ref: float) -> float:
    d = logp_ref - logp
    return math.exp(d) - d - 1.0


def clipped_surrogate(ratio: float, adv: float, clip_eps: float) -> float:
    unclipped = ratio * adv
    clipped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio)) * adv
    return min(unclipped, clipped)


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

    # ratio / advantage / clip scenario for the surrogate.
    clip_eps = 0.2
    surrogate_cases = [
        (1.0, 0.5),    # ratio == 1, no clip
        (1.5, 0.5),    # ratio above 1+eps, positive adv -> clipped
        (1.5, -0.5),   # ratio above 1+eps, negative adv -> unclipped is smaller
        (0.5, -0.5),   # ratio below 1-eps, negative adv -> clipped
        (0.8, 0.5),    # ratio below 1-eps, positive adv -> unclipped is smaller
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
        denom = sum(mrow)
        s = sum(v * m for v, m in zip(row, mrow))
        per_seq.append(s / denom)
    masked_mean_grpo = sum(per_seq) / num_seq

    # dr_grpo: sum of all valid contributions / (num_seq * max_len).
    total = sum(v * m for row, mrow in zip(per_token, mask) for v, m in zip(row, mrow))
    masked_mean_dr_grpo = total / (num_seq * max_len)

    fixture = {
        "_comment": "Golden GRPO math oracle. Regenerate via scripts/gen_golden.py.",
        "eps_std": EPS_STD,
        "groups": [
            {"rewards": rewards_a, "advantages": group_advantages(rewards_a)},
            {"rewards": rewards_b, "advantages": group_advantages(rewards_b)},
        ],
        "k3_kl": [
            {"logp": lp, "logp_ref": lr, "kl": k3_kl(lp, lr)}
            for (lp, lr) in logp_pairs
        ],
        "clipped_surrogate": [
            {
                "ratio": r,
                "advantage": a,
                "clip_eps": clip_eps,
                "value": clipped_surrogate(r, a, clip_eps),
            }
            for (r, a) in surrogate_cases
        ],
        "masked_mean": {
            "per_token": per_token,
            "mask": mask,
            "valid_tokens": valid,
            "num_seq": num_seq,
            "max_len": max_len,
            "grpo": masked_mean_grpo,
            "dr_grpo": masked_mean_dr_grpo,
        },
    }

    json.dump(fixture, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
