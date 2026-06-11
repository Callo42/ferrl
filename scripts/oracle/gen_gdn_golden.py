#!/usr/bin/env python3
"""Dump GDN-primitive golden fixtures from the transformers qwen3_5 reference.

Runs the EXACT reference code ferrl's kernels are ported from
(`transformers.models.qwen3_5.modeling_qwen3_5` torch fallbacks, pinned env
`ferrl-oracle`: transformers 5.11.0, CPU torch) on seeded inputs and writes
`crates/ferrl/tests/fixtures/gdn_golden.json` — the committed oracle for the
`src/gdn.rs` (and norm-variant) unit gates. Regenerate only when the pin moves:

    conda activate ferrl-oracle && python scripts/oracle/gen_gdn_golden.py

All tensors are float32; JSON floats round-trip f32 exactly (f32 -> f64 repr).
"""

import json
import pathlib

import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

OUT_PATH = pathlib.Path(__file__).resolve().parents[2] / "crates/ferrl/tests/fixtures/gdn_golden.json"
SEED = 42


def flat(t: torch.Tensor) -> list[float]:
    return t.detach().to(torch.float32).flatten().tolist()


def gdn_inputs(b: int, t: int, h: int, k: int, v: int) -> dict[str, torch.Tensor]:
    """Realistically-ranged kernel inputs: beta in (0,1), g <= 0 in log-space."""
    return {
        "query": torch.randn(b, t, h, k),
        "key": torch.randn(b, t, h, k),
        "value": torch.randn(b, t, h, v),
        "g": -torch.exp(torch.randn(h) * 0.5) * F.softplus(torch.randn(b, t, h) + 0.5),
        "beta": torch.sigmoid(torch.randn(b, t, h)),
    }


def gdn_case(
    kind: str, b: int, t: int, h: int, k: int, v: int, with_state: bool, chunk_size: int = 64
) -> dict:
    inp = gdn_inputs(b, t, h, k, v)
    initial = torch.randn(b, h, k, v) * 0.3 if with_state else None
    if kind == "recurrent":
        out, state = torch_recurrent_gated_delta_rule(
            inp["query"], inp["key"], inp["value"], inp["g"], inp["beta"],
            initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    else:
        out, state = torch_chunk_gated_delta_rule(
            inp["query"], inp["key"], inp["value"], inp["g"], inp["beta"],
            chunk_size=chunk_size, initial_state=initial, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    case = {"b": b, "t": t, "h": h, "k": k, "v": v}
    case.update({name: flat(x) for name, x in inp.items()})
    if initial is not None:
        case["initial_state"] = flat(initial)
    case["out"] = flat(out)
    case["state"] = flat(state)
    return case


def conv_case(with_ctx: bool) -> dict:
    b, c, l, kernel = 2, 5, 9, 4
    x = torch.randn(b, c, l)
    w = torch.randn(c, kernel)
    case = {"b": b, "c": c, "l": l, "kernel": kernel, "x": flat(x), "weight": flat(w)}
    if with_ctx:
        left = torch.randn(b, c, kernel - 1)
        case["left"] = flat(left)
        full = torch.cat([left, x], dim=-1)
        out = F.conv1d(full, w.unsqueeze(1), padding=0, groups=c)
    else:
        out = F.conv1d(F.pad(x, (kernel - 1, 0)), w.unsqueeze(1), padding=0, groups=c)
    assert out.shape == (b, c, l), out.shape
    case["out"] = flat(out)
    return case


def main() -> None:
    # The fixtures are a numeric contract: refuse to regenerate under
    # anything but the pinned oracle stack (see setup_env.sh).
    assert transformers.__version__ == "5.11.0", transformers.__version__
    assert torch.__version__.startswith("2.12.0"), torch.__version__
    torch.manual_seed(SEED)
    cases = {
        # Recurrent vs chunked, with and without an initial state. t=130
        # crosses two chunk boundaries (pads 130 -> 192); t=70 pads 70 -> 128.
        "recurrent_no_state": gdn_case("recurrent", b=2, t=7, h=2, k=4, v=6, with_state=False),
        "recurrent_with_state": gdn_case("recurrent", b=2, t=5, h=2, k=4, v=6, with_state=True),
        "chunked_t130": gdn_case("chunked", b=1, t=130, h=2, k=3, v=5, with_state=False),
        "chunked_with_state": gdn_case("chunked", b=1, t=70, h=2, k=3, v=5, with_state=True),
        # Pre-activation causal depthwise conv (no SiLU - the caller's job).
        "conv_no_ctx": conv_case(with_ctx=False),
        "conv_with_ctx": conv_case(with_ctx=True),
    }

    x = torch.tensor([-100.0, -30.0, -5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 30.0, 100.0])
    cases["softplus"] = {"x": flat(x), "out": flat(F.softplus(x))}

    xl = torch.randn(3, 5)
    cases["l2norm"] = {"rows": 3, "dim": 5, "x": flat(xl), "out": flat(l2norm(xl, dim=-1, eps=1e-6))}
    # Small-magnitude rows: the eps-inside-the-sum convention only diverges
    # from eps-on-the-norm near zero, so this case is what keeps the
    # wrong-convention planted bug catchable (gate non-vacuity).
    xs = torch.randn(3, 5) * 1e-3
    cases["l2norm_small"] = {"rows": 3, "dim": 5, "x": flat(xs), "out": flat(l2norm(xs, dim=-1, eps=1e-6))}

    hidden = 8
    m = Qwen3_5RMSNorm(hidden, eps=1e-6)
    m.weight.data = torch.randn(hidden) * 0.1
    xn = torch.randn(2, 3, hidden)
    cases["rmsnorm_zero_centered"] = {
        "shape": [2, 3, hidden], "hidden": hidden,
        "x": flat(xn), "weight": flat(m.weight), "out": flat(m(xn)),
    }

    mg = Qwen3_5RMSNormGated(hidden, eps=1e-6)
    mg.weight.data = torch.randn(hidden) * 0.1 + 1.0
    xg = torch.randn(4, hidden)
    gate = torch.randn(4, hidden)
    cases["rmsnorm_gated"] = {
        "rows": 4, "hidden": hidden,
        "x": flat(xg), "gate": flat(gate), "weight": flat(mg.weight), "out": flat(mg(xg, gate)),
    }

    # Rope at REAL qwen3_5 geometry (head_dim 256, rotary_dim 64, theta 1e7),
    # through the ACTUAL transformers rotary path (Qwen3_5TextRotaryEmbedding +
    # apply_rotary_pos_emb) with text-only 1-D positions — this pins both the
    # partial rotate-half conventions AND the claim that the interleaved
    # M-RoPE is an exact no-op for text. No shared formula with the Rust side.
    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    )
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    l = 4
    q = torch.randn(1, 1, l, 256)
    k = torch.randn(1, 1, l, 256)
    position_ids = torch.arange(l).unsqueeze(0)  # 2-D text-only positions
    cos, sin = rot(q, position_ids)
    q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_emb.shape == q.shape, q_emb.shape
    cases["rope_text_qwen35_geometry"] = {
        "l": l,
        "head_dim": 256,
        "rot_dim": 64,
        "rope_theta": 10000000.0,
        "q": flat(q),
        "k": flat(k),
        "q_out": flat(q_emb),
        "k_out": flat(k_emb),
    }

    obj = {
        "meta": {
            "generator": "scripts/oracle/gen_gdn_golden.py",
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
