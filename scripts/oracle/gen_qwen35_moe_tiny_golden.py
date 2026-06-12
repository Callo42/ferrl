#!/usr/bin/env python3
"""Build + dump the committed tiny qwen3_5_moe oracle fixture (M3' PR-2).

The MoE twin of `gen_qwen35_tiny_golden.py`: constructs a tiny
`Qwen3_5MoeForConditionalGeneration` (the REAL reference class — production
`model.language_model.*` layout, EVERY layer's MLP a sparse MoE block, the
eager packed-weight experts path), seeds every parameter deterministically
(including the router, which ships zero-init — a sharp routing fixture is
load-bearing for the model gates, same as the kernel fixture), saves it
SHARDED, and dumps fp32 logits for the same case trio as the dense fixture:

  - two uncached full forwards (batch 1 and batch 2);
  - a cached run: prefill -> MULTI-token continuation at an offset -> two
    single-token decodes — plus the same tokens uncached, tying the trio to
    the uncached truth inside one fixture.

Outputs (committed under `crates/ferrl/tests/fixtures/tiny_qwen35_moe/`):
  config.json + model-*.safetensors + model.safetensors.index.json and
  golden.json (inputs, logits, version-pinned meta).

Pinned env `ferrl-oracle` (transformers 5.11.0, CPU torch 2.12.0); regenerate
only when the pin moves:

    conda activate ferrl-oracle && python scripts/oracle/gen_qwen35_moe_tiny_golden.py
"""

import json
import pathlib

import torch
import transformers
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

OUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "crates/ferrl/tests/fixtures/tiny_qwen35_moe"
)
SEED = 42
WEIGHT_STD = 0.5


def flat(t: torch.Tensor) -> list[float]:
    return t.detach().to(torch.float32).flatten().tolist()


def tiny_config() -> Qwen3_5MoeConfig:
    """A tiny hybrid MoE config: both mixers, 8 routed experts top-2, real
    GVA (4 value / 2 key heads). Every packed axis is distinct so a stack/
    transpose bug is a SHAPE error, not a silent value bug: E=8 != m=6 !=
    shared=12 != hidden=16, and 2m=12 != hidden=16 (the packed gate_up is
    [8, 12, 16], down [8, 16, 6] — no aliasing axes)."""
    text = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=12,
        num_experts=8,
        num_experts_per_tok=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        pad_token_id=0,
        eos_token_id=1,
    )
    vision = Qwen3_5MoeVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=4,
    )
    return Qwen3_5MoeConfig(
        text_config=text, vision_config=vision, tie_word_embeddings=True
    )


def seed_weights(model: torch.nn.Module) -> None:
    """Deterministic, decisively-non-uniform weights (the M1 vacuity lesson) —
    the router included: zero-init routing is uniform/tie-broken and would
    make the sparse paths near-vacuous."""
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=gen) * WEIGHT_STD)


def ids_row(length: int, stride: int, offset: int, vocab: int) -> list[int]:
    return [(i * stride + offset) % vocab for i in range(length)]


def main() -> None:
    assert transformers.__version__ == "5.11.0", transformers.__version__
    assert torch.__version__.startswith("2.12.0"), torch.__version__
    torch.manual_seed(SEED)

    cfg = tiny_config()
    model = Qwen3_5MoeForConditionalGeneration(cfg)
    # Model construction auto-dispatches the experts implementation (this env
    # picks `grouped_mm`). ferrl ports the EAGER packed-weight loop, so the
    # fixture must pin that path — the decorated forward reads the config at
    # call time, so a post-construction override is honored.
    dispatched = cfg.text_config._experts_implementation
    model.config.text_config._experts_implementation_internal = "eager"
    assert cfg.text_config._experts_implementation == "eager"
    seed_weights(model)
    model = model.eval().to(torch.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # A tiny shard cap forces the index.json layout the multi-shard loader
    # must handle.
    model.save_pretrained(OUT_DIR, max_shard_size="100KB", safe_serialization=True)

    vocab = cfg.text_config.vocab_size
    cases: dict = {}

    with torch.no_grad():
        full_b1 = torch.tensor([ids_row(12, 7, 3, vocab)])
        out = model(input_ids=full_b1, use_cache=False)
        cases["full_b1"] = {
            "input_ids": full_b1.tolist(),
            "logits": flat(out.logits),
            "shape": list(out.logits.shape),
        }

        full_b2 = torch.tensor([ids_row(10, 7, 3, vocab), ids_row(10, 5, 11, vocab)])
        out = model(input_ids=full_b2, use_cache=False)
        cases["full_b2"] = {
            "input_ids": full_b2.tolist(),
            "logits": flat(out.logits),
            "shape": list(out.logits.shape),
        }

        seq = ids_row(14, 7, 3, vocab)
        prefill = torch.tensor([seq[:7]])
        chunk = torch.tensor([seq[7:12]])
        out1 = model(input_ids=prefill, use_cache=True)
        past = out1.past_key_values
        out2 = model(input_ids=chunk, past_key_values=past, use_cache=True)
        past = out2.past_key_values
        step_logits = []
        for t in range(12, 14):
            tok = torch.tensor([[seq[t]]])
            out_t = model(input_ids=tok, past_key_values=past, use_cache=True)
            past = out_t.past_key_values
            step_logits.append(flat(out_t.logits))
        cases["cached_split"] = {
            "input_ids": [seq],
            "prefill_len": 7,
            "chunk_len": 5,
            "prefill_logits": flat(out1.logits),
            "chunk_logits": flat(out2.logits),
            "decode_logits": step_logits,
        }

        full = torch.tensor([seq])
        out = model(input_ids=full, use_cache=False)
        cases["cached_split"]["uncached_logits"] = flat(out.logits)

        # Documentation datum (best-effort): how far the env's auto-dispatched
        # experts implementation sits from the pinned eager path. Some
        # geometries are outside the fused kernel's constraints (e.g.
        # torch._grouped_mm demands 16-byte strides, which m=6 breaks) — the
        # eager path is the pin either way, so record the failure instead.
        eager_logits = out.logits
        impl_diff = None
        if dispatched not in (None, "eager"):
            model.config.text_config._experts_implementation_internal = dispatched
            try:
                alt = model(input_ids=full, use_cache=False).logits
                impl_diff = (alt - eager_logits).abs().max().item()
            except RuntimeError as e:
                impl_diff = f"unavailable at this geometry: {e}"
            model.config.text_config._experts_implementation_internal = "eager"

        # Routing sharpness: the fixture's routing must be decisively
        # non-uniform, or the top-k depth/index gates downstream go vacuous
        # (the kernel-fixture lesson). Asserted at generation time here AND
        # recorded in the meta so the Rust gate re-asserts it on every CI run
        # — a degenerate regeneration cannot silently land.
        text_model = model.model.language_model
        h = text_model.embed_tokens(full_b1).reshape(-1, cfg.text_config.hidden_size)
        _, scores, _ = text_model.layers[0].mlp.gate(h)
        k = cfg.text_config.num_experts_per_tok
        routing_sharpness = (scores - 1.0 / k).abs().max().item()
        assert routing_sharpness > 0.05, "degenerate routing"

    obj = {
        "meta": {
            "generator": "scripts/oracle/gen_qwen35_moe_tiny_golden.py",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "seed": SEED,
            "weight_std": WEIGHT_STD,
            "experts_implementation": "eager",
            "auto_dispatched_implementation": dispatched,
            "dispatched_vs_eager_max_abs_diff": impl_diff,
            "routing_sharpness": routing_sharpness,
        },
        "cases": cases,
    }
    golden = OUT_DIR / "golden.json"
    with golden.open("w") as f:
        json.dump(obj, f)
        f.write("\n")
    sizes = {p.name: p.stat().st_size for p in sorted(OUT_DIR.iterdir())}
    print(f"wrote {OUT_DIR}:")
    for name, size in sizes.items():
        print(f"  {name}: {size} bytes")


if __name__ == "__main__":
    main()
