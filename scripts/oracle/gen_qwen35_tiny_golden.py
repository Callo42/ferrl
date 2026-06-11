#!/usr/bin/env python3
"""Build + dump the committed tiny qwen3_5 oracle fixture.

Constructs a tiny `Qwen3_5ForConditionalGeneration` (the REAL reference class,
so the saved checkpoint has the production `model.language_model.*` layout, a
decoy vision tower for the loader's ignore behavior, and the real text-only
M-RoPE/cache code paths), seeds every parameter deterministically, saves it
SHARDED (forcing the `model.safetensors.index.json` layout the multi-shard
loader must handle), and dumps fp32 logits for:

  - two uncached full forwards (batch 1 and batch 2);
  - a cached run: prefill -> MULTI-token continuation at an offset (the GDN
    cached path fixed in transformers v5.7.0 — ferrl's highest-risk decoder
    path) -> two single-token decodes.

Outputs (committed under `crates/ferrl/tests/fixtures/tiny_qwen35/`):
  config.json + model-*.safetensors + model.safetensors.index.json  (the tiny
  checkpoint) and golden.json (inputs, logits, version-pinned meta).

Pinned env `ferrl-oracle` (transformers 5.11.0, CPU torch 2.12.0); regenerate
only when the pin moves:

    conda activate ferrl-oracle && python scripts/oracle/gen_qwen35_tiny_golden.py
"""

import json
import pathlib

import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

OUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "crates/ferrl/tests/fixtures/tiny_qwen35"
)
SEED = 42
WEIGHT_STD = 0.5


def flat(t: torch.Tensor) -> list[float]:
    return t.detach().to(torch.float32).flatten().tolist()


def tiny_config() -> Qwen3_5Config:
    """A tiny hybrid config exercising everything the 0.8B does — plus real
    GVA (4 value / 2 key heads), which the 0.8B itself lacks (16 == 16)."""
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
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
    vision = Qwen3_5VisionConfig(
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
    return Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True)


def seed_weights(model: torch.nn.Module) -> None:
    """Deterministic, decisively-non-uniform weights (the M1 vacuity lesson:
    transformers' own init std 0.02 makes tiny-model attention near-uniform
    and equivalence gates near-vacuous). Iteration over named_parameters is
    deterministic (module insertion order)."""
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
    model = Qwen3_5ForConditionalGeneration(cfg)
    seed_weights(model)
    model = model.eval().to(torch.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ~150KB of fp32 weights; a tiny shard cap forces the index.json layout.
    model.save_pretrained(OUT_DIR, max_shard_size="100KB", safe_serialization=True)

    vocab = cfg.text_config.vocab_size
    cases: dict = {}

    with torch.no_grad():
        # Uncached full forwards (batch 1 and 2 — the batch dim crosses the
        # conv, GVA broadcast, and mask shapes).
        full_b1 = torch.tensor([ids_row(12, 7, 3, vocab)])
        out = model(input_ids=full_b1, use_cache=False)
        cases["full_b1"] = {
            "input_ids": full_b1.tolist(),
            "logits": flat(out.logits),
            "shape": list(out.logits.shape),
        }

        full_b2 = torch.tensor(
            [ids_row(10, 7, 3, vocab), ids_row(10, 5, 11, vocab)]
        )
        out = model(input_ids=full_b2, use_cache=False)
        cases["full_b2"] = {
            "input_ids": full_b2.tolist(),
            "logits": flat(out.logits),
            "shape": list(out.logits.shape),
        }

        # Cached: prefill 7 -> 5-token continuation at offset 7 (the chunked
        # multi-token cached path, v5.7.0 semantics) -> 2 single decodes.
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

        # The same 14 tokens uncached — ties the cached trio back to the
        # uncached truth inside ONE fixture.
        full = torch.tensor([seq])
        out = model(input_ids=full, use_cache=False)
        cases["cached_split"]["uncached_logits"] = flat(out.logits)

    obj = {
        "meta": {
            "generator": "scripts/oracle/gen_qwen35_tiny_golden.py",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "seed": SEED,
            "weight_std": WEIGHT_STD,
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
