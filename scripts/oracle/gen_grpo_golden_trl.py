#!/usr/bin/env python3
"""Generate the REAL-TRL GRPO loss golden fixture (closes GOLDEN-CIRCULAR).

``scripts/gen_golden.py`` is a NumPy *transcription* of the GRPO formulas —
same author as the Rust code, so a shared misreading of the spec passes both
sides (the P0 review's ``GOLDEN-CIRCULAR`` finding). This script instead drives
**TRL's own** ``GRPOTrainer._compute_loss`` (the industry reference
implementation, pinned by version) over crafted log-prob/mask/advantage
tensors, and records the losses it produces. The Rust gate
(``trainer::tests::matches_trl_golden_fixture``) replays every case through
``ferrl``'s production ``grpo_loss`` and must agree.

Scope (honest):

* Pinned against real TRL code: the clipped surrogate (asymmetric
  ``epsilon_low``/``epsilon_high``), the k3 KL penalty fold, token- vs
  sequence-level importance sampling (the GSPO seam), and the ``grpo`` /
  ``dr_grpo`` / ``dapo`` reductions including the DAPO
  ``num_items_in_batch`` window normalizer.
* NOT pinned here: group-advantage normalization (it lives inside TRL's
  generation path, which needs a full trainer+model; the formula is
  read-verified against TRL source — ``torch.std`` unbiased, ``+1e-4`` — and
  stays pinned by the NumPy fixture) and rollout/truncation mechanics
  (``mask_truncated_completions`` only zeroes the loss mask upstream of
  ``_compute_loss``; ferrl pins that semantics in its own unit tests).

The trainer instance is built via ``object.__new__`` with exactly the
attributes ``_compute_loss`` reads, and ``_get_per_token_logps_and_entropies``
is stubbed to return the crafted logps — so the *loss math* is 100% TRL's,
while no model/dataset/accelerator stack is needed. Brittle by design across
TRL versions: the fixture records the TRL/torch versions and the Rust gate
asserts the pins, so a regeneration under a different TRL is a deliberate,
reviewed act (same contract as the qwen3_5 tiny-oracle fixture).

Everything runs in float64 so the recorded values are reference-accurate
(ferrl's f32 tensor path compares at a loose-but-honest tolerance).

Run in the pinned ``ferrl-oracle`` conda env on the cluster (after
``pip install trl``):

    python3 scripts/oracle/gen_grpo_golden_trl.py \
        > crates/ferrl/tests/fixtures/grpo_golden_trl.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import torch
import transformers
import trl
from trl.trainer.grpo_trainer import GRPOTrainer


class _Args:
    """The ``self.args`` fields ``_compute_loss`` touches."""

    delta = None
    use_bias_correction_kl = False


class _Accel:
    """Single-process accelerator stub (gather is the identity)."""

    num_processes = 1

    @staticmethod
    def gather(x):
        return x


class _Model:
    """``self.model`` stub: only ``.training`` is read (selects train mode)."""

    training = True


def make_trainer(
    loss_type: str,
    beta: float,
    eps_low: float,
    eps_high: float,
    importance_sampling_level: str,
    max_completion_length: int,
    logps: torch.Tensor,
) -> GRPOTrainer:
    """A GRPOTrainer shell carrying exactly the state _compute_loss reads."""
    t = object.__new__(GRPOTrainer)
    t.loss_type = loss_type
    t.beta = beta
    t.epsilon_low = eps_low
    t.epsilon_high = eps_high
    t.importance_sampling_level = importance_sampling_level
    t.top_entropy_quantile = 1.0  # no entropy masking
    t.off_policy_mask_threshold = None
    t.use_vllm = False
    t.vllm_importance_sampling_correction = False
    t.mask_truncated_completions = False  # acts upstream of _compute_loss
    t.current_gradient_accumulation_steps = 1
    t.max_completion_length = max_completion_length
    t.args = _Args()
    t.accelerator = _Accel()
    t.model = _Model()
    t._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    # The only non-loss work _compute_loss does is computing the current
    # policy's per-token logps from the model; stub it to the crafted tensor.
    def _stub(model, input_ids, attention_mask, logits_to_keep, **kw):
        return logps, torch.zeros_like(logps)

    t._get_per_token_logps_and_entropies = _stub
    return t


def main() -> None:
    torch.set_default_dtype(torch.float64)

    # One fixed batch geometry: B=4 completions of width T=3 over a P=2
    # prompt. The mask is ragged — lengths 3, 2, 1, 0 — so per-row
    # denominators, the final-column real token, and an all-pad row are all
    # exercised (mirroring the Rust gradcheck scenarios).
    prompt_ids = torch.zeros((4, 2), dtype=torch.long)
    prompt_mask = torch.ones((4, 2), dtype=torch.long)
    completion_ids = torch.zeros((4, 3), dtype=torch.long)
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    # Current/old log-probs chosen so the per-token ratios straddle both clip
    # bands (exp(±0.30) ≈ 1.35 / 0.74, exp(0.05) ≈ 1.05) and the masked cells
    # carry junk that must stay inert.
    logp = torch.tensor(
        [
            [-1.00, -2.00, -0.40],
            [-0.50, -0.25, -9.00],
            [-1.50, -7.00, -8.00],
            [-2.00, -2.00, -2.00],
        ]
    )
    shift = torch.tensor(
        [
            [0.30, 0.05, -0.30],
            [-0.30, 0.05, 0.00],
            [0.30, 0.00, 0.00],
            [0.05, 0.05, 0.05],
        ]
    )
    logp_old = logp - shift
    logp_ref = logp + 0.10  # a slightly-different reference for the k3 KL
    advantages = torch.tensor([0.80, -0.70, 0.50, -0.40])
    total_tokens = float(mask.sum())  # 6 active tokens

    cases = []
    for loss_type in ("grpo", "dr_grpo", "dapo"):
        for beta in (0.0, 0.04):
            for level in ("token", "sequence"):
                for eps_low, eps_high in ((0.2, 0.2), (0.2, 0.28)):
                    # DAPO additionally pins the window normalizer: the
                    # single-batch value AND an explicit larger window (the
                    # accumulation case, num_items_in_batch > this batch).
                    if loss_type == "dapo":
                        norms = [total_tokens, 20.0]
                    else:
                        norms = [total_tokens]
                    for num_items in norms:
                        trainer = make_trainer(
                            loss_type, beta, eps_low, eps_high, level, 3, logp
                        )
                        inputs = {
                            "prompt_ids": prompt_ids,
                            "prompt_mask": prompt_mask,
                            "completion_ids": completion_ids,
                            "completion_mask": mask,
                            "advantages": advantages,
                            "old_per_token_logps": logp_old,
                            "num_items_in_batch": torch.tensor(num_items),
                        }
                        if beta != 0.0:
                            inputs["ref_per_token_logps"] = logp_ref
                        loss = trainer._compute_loss(trainer.model, inputs)
                        cases.append(
                            {
                                "loss_type": loss_type,
                                "beta": beta,
                                "eps_low": eps_low,
                                "eps_high": eps_high,
                                "importance_sampling_level": level,
                                "dapo_norm": num_items,
                                "loss": float(loss),
                            }
                        )

    fixture = {
        "_comment": (
            "REAL-TRL GRPO loss golden: produced by TRL's "
            "GRPOTrainer._compute_loss over the shared crafted batch below. "
            "Regenerate via scripts/oracle/gen_grpo_golden_trl.py in the "
            "pinned ferrl-oracle env."
        ),
        "trl_version": trl.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "batch": {
            "logp": logp.tolist(),
            "logp_old": logp_old.tolist(),
            "logp_ref": logp_ref.tolist(),
            "advantages": advantages.tolist(),
            "mask": mask.tolist(),
        },
        "cases": cases,
    }
    json.dump(fixture, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
