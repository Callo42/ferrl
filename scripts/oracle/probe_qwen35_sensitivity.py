#!/usr/bin/env python3
"""Measure the reference's OWN per-position float sensitivity on the real 0.8B.

Runs the pinned transformers reference twice on the same prompt: once with the
standard chunk-64 delta-rule kernel, once with chunk-32 (mathematically
identical, float-different op order). The per-position logit divergence
between the two REFERENCE runs is the intrinsic amplification floor of the
hybrid recurrence — anything a correct independent port can be expected to
accumulate. Used to calibrate the real-weights gate tolerance honestly
(measured, not hand-waved).
"""

import functools
import pathlib

import torch
import transformers
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

ASSET_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets/qwen3_5-0.8b-base"
PROMPT = "In 1969, humans first landed on the Moon; the mission was called Apollo 11."


def main() -> None:
    assert transformers.__version__ == "5.11.0", transformers.__version__
    assert torch.__version__.startswith("2.12.0"), torch.__version__

    tok = AutoTokenizer.from_pretrained(ASSET_DIR)
    model = m.Qwen3_5ForConditionalGeneration.from_pretrained(
        ASSET_DIR, dtype=torch.float32
    ).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids

    orig = m.torch_chunk_gated_delta_rule
    with torch.no_grad():
        out64 = model(input_ids=ids, use_cache=False).logits
        # Same kernel, chunk 8: exact same math, but the 23-token sequence now
        # decomposes into 3 chunks with a real inter-chunk recurrence —
        # genuinely different float op order (chunk 32 vs 64 is vacuous here:
        # both hold the whole sequence in one chunk and zero-padding is exact).
        for layer in model.model.language_model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.chunk_gated_delta_rule = functools.partial(
                    orig, chunk_size=8
                )
        out32 = model(input_ids=ids, use_cache=False).logits

    print("per-position max-abs divergence, reference(chunk64) vs reference(chunk8):")
    for t in range(out64.shape[1]):
        d = (out64[0, t] - out32[0, t]).abs().max().item()
        s = out64[0, t].abs().max().item()
        print(f"  t={t}: max-abs {d:.6e}  scale {s:.2f}  rel {d / s:.3e}")


if __name__ == "__main__":
    main()
