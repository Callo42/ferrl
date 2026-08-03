#!/usr/bin/env python3
"""Build and dump the committed tiny dense-Llama forward oracle.

The checkpoint and logits come from the official Transformers implementation
(`LlamaForCausalLM`, transformers 5.11.0, torch 2.12.0, fp32 CPU).  The tiny
shape exercises real GQA plus Llama-3 wavelength-smoothing RoPE at every
sequence position.

Transformers 5.11 serializes the authoritative Llama-3 settings under
``rope_parameters``.  Candle's current Llama serde mirror consumes the legacy
``rope_scaling`` name.  After the official save, this generator retains the
upstream block and adds an exactly equal compatibility mirror.  The Rust gate
asserts both fields and the golden provenance before parsing the config.

Outputs (committed under ``crates/ferrl/tests/fixtures/tiny_llama/``):
``config.json``, ``model.safetensors``, the auxiliary ``generation_config.json``
when emitted by ``save_pretrained``, and ``golden.json``.

Pinned regeneration command::

    conda activate ferrl-oracle && python scripts/oracle/gen_llama_tiny_golden.py
"""

import json
import pathlib

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

OUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "crates/ferrl/tests/fixtures/tiny_llama"
)
SEED = 27182
WEIGHT_STD = 0.5
TRANSFORMERS_PIN = "5.11.0"
TORCH_PIN = "2.12.0"
ROPE_PARAMETERS = {
    "rope_type": "llama3",
    "rope_theta": 500_000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 16,
}
ROPE_BRIDGE = (
    "upstream rope_parameters mirrored to candle rope_scaling and rope_theta"
)


def flat(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(torch.float32).flatten().tolist()


def tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_parameters=ROPE_PARAMETERS,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def seed_weights(model: torch.nn.Module) -> None:
    """Install deterministic high-signal weights in stable parameter-name order."""
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for _, parameter in sorted(model.named_parameters()):
            values = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=torch.float32,
            )
            parameter.copy_(values * WEIGHT_STD)
    model.tie_weights()


def ids_row(length: int, stride: int, offset: int, vocab: int) -> list[int]:
    return [(index * stride + offset) % vocab for index in range(length)]


def install_candle_rope_bridge() -> None:
    config_path = OUT_DIR / "config.json"
    with config_path.open() as handle:
        config = json.load(handle)
    rope_parameters = config.get("rope_parameters")
    assert rope_parameters == ROPE_PARAMETERS, rope_parameters
    config["rope_scaling"] = rope_parameters
    config["rope_theta"] = rope_parameters["rope_theta"]
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    assert transformers.__version__ == TRANSFORMERS_PIN, transformers.__version__
    assert torch.__version__.startswith(TORCH_PIN), torch.__version__
    torch.manual_seed(SEED)

    config = tiny_config()
    model = LlamaForCausalLM(config)
    seed_weights(model)
    model = model.eval().to(torch.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    install_candle_rope_bridge()

    vocab = config.vocab_size
    cases: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        full_b1 = torch.tensor(
            [ids_row(12, 7, 3, vocab)],
            dtype=torch.long,
        )
        output = model(input_ids=full_b1, use_cache=False)
        cases["full_b1"] = {
            "input_ids": full_b1.tolist(),
            "logits": flat(output.logits),
            "shape": list(output.logits.shape),
        }

        full_b2 = torch.tensor(
            [
                ids_row(9, 5, 1, vocab),
                ids_row(9, 11, 4, vocab),
            ],
            dtype=torch.long,
        )
        output = model(input_ids=full_b2, use_cache=False)
        cases["full_b2"] = {
            "input_ids": full_b2.tolist(),
            "logits": flat(output.logits),
            "shape": list(output.logits.shape),
        }

    golden = {
        "meta": {
            "generator": "scripts/oracle/gen_llama_tiny_golden.py",
            "reference": "transformers.LlamaForCausalLM/LlamaConfig",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "seed": SEED,
            "weight_std": WEIGHT_STD,
            "rope_config_bridge": ROPE_BRIDGE,
        },
        "cases": cases,
    }
    with (OUT_DIR / "golden.json").open("w") as handle:
        json.dump(golden, handle)
        handle.write("\n")

    sizes = {path.name: path.stat().st_size for path in sorted(OUT_DIR.iterdir())}
    print(f"wrote {OUT_DIR}:")
    for name, size in sizes.items():
        print(f"  {name}: {size} bytes")


if __name__ == "__main__":
    main()
