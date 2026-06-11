#!/usr/bin/env python3
"""Dump fp32 reference logits for the real Qwen3.5-0.8B-Base checkpoint.

Runs the pinned transformers reference (`ferrl-oracle` env: transformers
5.11.0, CPU torch 2.12.0, fp32) over fixed prompts and writes per-position
logits + token ids next to the checkpoint:

    <asset>/ferrl_oracle_dumps/real_logits.safetensors  (binary, NOT committed
    — the 248k vocab makes JSON fixtures unreasonable)
    <asset>/ferrl_oracle_dumps/meta.json                (prompts + version pins)

The `#[ignore]`d Rust gates (`tests/qwen35_real_weights.rs`,
`tests/qwen35_gpu_smoke.rs`) consume these via `FERRL_QWEN35_ORACLE`.

    conda activate ferrl-oracle && python scripts/oracle/dump_qwen35_real_logits.py
"""

import json
import pathlib

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

ASSET_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets/qwen3_5-0.8b-base"
OUT_DIR = ASSET_DIR / "ferrl_oracle_dumps"

PROMPTS = [
    "The cat sat on the mat, and the cat slept.",
    "In 1969, humans first landed on the Moon; the mission was called Apollo 11.",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
]


def main() -> None:
    assert transformers.__version__ == "5.11.0", transformers.__version__
    assert torch.__version__.startswith("2.12.0"), torch.__version__

    tok = AutoTokenizer.from_pretrained(ASSET_DIR)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        ASSET_DIR, dtype=torch.float32
    ).eval()

    tensors: dict[str, torch.Tensor] = {}
    meta: dict = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "prompts": PROMPTS,
        "cases": [],
    }
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").input_ids
            out = model(input_ids=ids, use_cache=False)
            tensors[f"p{i}_ids"] = ids.to(torch.int64).contiguous()
            tensors[f"p{i}_logits"] = out.logits.to(torch.float32).contiguous()
            meta["cases"].append(
                {"prompt": prompt, "len": int(ids.shape[1]), "vocab": int(out.logits.shape[2])}
            )
            print(f"p{i}: {ids.shape[1]} tokens, logits {tuple(out.logits.shape)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUT_DIR / "real_logits.safetensors")
    with (OUT_DIR / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
