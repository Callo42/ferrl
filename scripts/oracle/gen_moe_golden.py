#!/usr/bin/env python3
"""Dump MoE-primitive golden fixtures from the transformers qwen3_5_moe reference.

Runs the EXACT reference code ferrl's MoE kernels are ported from
(`transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`, pinned env
`ferrl-oracle`: transformers 5.11.0, CPU torch) on seeded inputs and writes
`crates/ferrl/tests/fixtures/moe_golden.json` — the committed oracle for the
`src/moe.rs` unit gates. Regenerate only when the pin moves:

    conda activate ferrl-oracle && python scripts/oracle/gen_moe_golden.py

All tensors are float32; JSON floats round-trip f32 exactly. Router indices are
dumped as ints and asserted EXACTLY in the gates (continuous random inputs make
top-k ties measure-zero). The family's reference deletes `norm_topk_prob` /
`mlp_only_layers` / `decoder_sparse_step`: every layer is sparse and the top-k
probabilities are renormalized UNCONDITIONALLY — the dumps pin that shape.
"""

import json
import pathlib
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter,
)

OUT_PATH = pathlib.Path(__file__).resolve().parents[2] / "crates/ferrl/tests/fixtures/moe_golden.json"
SEED = 42


def flat(t: torch.Tensor) -> list[float]:
    return t.detach().to(torch.float32).flatten().tolist()


def flat_int(t: torch.Tensor) -> list[int]:
    return t.detach().to(torch.int64).flatten().tolist()


def tiny_cfg(hidden: int, experts: int, top_k: int, moe_inter: int, shared_inter: int):
    """The minimal attribute surface the reference MoE classes read.

    `_experts_implementation = None` routes the decorated `Qwen3_5MoeExperts.forward`
    to the original eager loop (the code ferrl ports) rather than a kernel backend.
    """
    return SimpleNamespace(
        hidden_size=hidden,
        num_experts=experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        hidden_act="silu",
        _experts_implementation=None,
    )


def seeded_router(cfg) -> Qwen3_5MoeTopKRouter:
    router = Qwen3_5MoeTopKRouter(cfg)
    # The shipped init is zeros (uniform routing, tie-broken top-k) — useless
    # as an oracle. Seeded gaussian weights make the routing decisions sharp.
    router.weight.data = torch.randn(cfg.num_experts, cfg.hidden_size) * 0.7
    return router


def seeded_experts(cfg) -> Qwen3_5MoeExperts:
    experts = Qwen3_5MoeExperts(cfg)
    experts.gate_up_proj.data = (
        torch.randn(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size) * 0.4
    )
    experts.down_proj.data = (
        torch.randn(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size) * 0.4
    )
    return experts


def router_case(cfg, tokens: int) -> dict:
    router = seeded_router(cfg)
    x = torch.randn(tokens, cfg.hidden_size)
    logits, scores, indices = router(x)
    assert logits.shape == (tokens, cfg.num_experts)
    assert scores.shape == (tokens, cfg.num_experts_per_tok)
    # The renorm is unconditional: rows of `scores` sum to 1 exactly-ish.
    assert torch.allclose(scores.sum(-1), torch.ones(tokens), atol=1e-6)
    return {
        "tokens": tokens,
        "hidden": cfg.hidden_size,
        "experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
        "x": flat(x),
        "weight": flat(router.weight),
        "logits": flat(logits),
        "scores": flat(scores),
        "indices": flat_int(indices),
    }


def experts_case(cfg, tokens: int, name: str, want_unhit: bool) -> dict:
    router = seeded_router(cfg)
    experts = seeded_experts(cfg)
    x = torch.randn(tokens, cfg.hidden_size)
    _, scores, indices = router(x)
    hit = torch.zeros(cfg.num_experts, dtype=torch.bool)
    hit[indices.flatten()] = True
    if want_unhit:
        assert not hit.all(), f"{name}: every expert was hit — shrink tokens or grow experts"
    out = experts(x, indices, scores)
    assert out.shape == x.shape
    return {
        "tokens": tokens,
        "hidden": cfg.hidden_size,
        "experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
        "moe_inter": cfg.moe_intermediate_size,
        "x": flat(x),
        "gate_up_proj": flat(experts.gate_up_proj),
        "down_proj": flat(experts.down_proj),
        "indices": flat_int(indices),
        "scores": flat(scores),
        "out": flat(out),
    }


def block_case(cfg, b: int, t: int) -> dict:
    block = Qwen3_5MoeSparseMoeBlock(cfg)
    block.gate.weight.data = torch.randn(cfg.num_experts, cfg.hidden_size) * 0.7
    block.experts.gate_up_proj.data = (
        torch.randn(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size) * 0.4
    )
    block.experts.down_proj.data = (
        torch.randn(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size) * 0.4
    )
    sh = block.shared_expert
    sh.gate_proj.weight.data = torch.randn(cfg.shared_expert_intermediate_size, cfg.hidden_size) * 0.4
    sh.up_proj.weight.data = torch.randn(cfg.shared_expert_intermediate_size, cfg.hidden_size) * 0.4
    sh.down_proj.weight.data = torch.randn(cfg.hidden_size, cfg.shared_expert_intermediate_size) * 0.4
    block.shared_expert_gate.weight.data = torch.randn(1, cfg.hidden_size) * 0.7
    x = torch.randn(b, t, cfg.hidden_size)
    out = block(x)
    assert out.shape == x.shape
    return {
        "b": b,
        "t": t,
        "hidden": cfg.hidden_size,
        "experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
        "moe_inter": cfg.moe_intermediate_size,
        "shared_inter": cfg.shared_expert_intermediate_size,
        "x": flat(x),
        "router_weight": flat(block.gate.weight),
        "gate_up_proj": flat(block.experts.gate_up_proj),
        "down_proj": flat(block.experts.down_proj),
        "shared_gate_proj": flat(sh.gate_proj.weight),
        "shared_up_proj": flat(sh.up_proj.weight),
        "shared_down_proj": flat(sh.down_proj.weight),
        "shared_expert_gate": flat(block.shared_expert_gate.weight),
        "out": flat(out),
    }


def main() -> None:
    assert transformers.__version__ == "5.11.0", transformers.__version__
    assert torch.__version__.startswith("2.12.0"), torch.__version__
    torch.manual_seed(SEED)

    cfg = tiny_cfg(hidden=16, experts=8, top_k=3, moe_inter=12, shared_inter=20)
    cases = {
        "router_t5": router_case(cfg, tokens=5),
        "router_t1": router_case(cfg, tokens=1),
        # 2 tokens x top-3 of 8 experts: at most 6 hit, >= 2 unhit — pins the
        # reference's skip-unhit-expert behavior.
        "experts_t9": experts_case(cfg, tokens=9, name="experts_t9", want_unhit=False),
        "experts_unhit_t2": experts_case(cfg, tokens=2, name="experts_unhit_t2", want_unhit=True),
        "block_b1_t6": block_case(cfg, b=1, t=6),
        "block_b2_t4": block_case(cfg, b=2, t=4),
    }
    # A second geometry (k == 1 edge + non-divisible sizes) on the block path.
    cfg_k1 = tiny_cfg(hidden=10, experts=5, top_k=1, moe_inter=7, shared_inter=9)
    cases["block_k1_b1_t3"] = block_case(cfg_k1, b=1, t=3)

    obj = {
        "meta": {
            "generator": "scripts/oracle/gen_moe_golden.py",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "seed": SEED,
        },
        "cases": cases,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(obj, f)
        f.write("\n")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
