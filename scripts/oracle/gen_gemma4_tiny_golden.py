#!/usr/bin/env python3
"""Build + dump the committed tiny Gemma 4 text oracle fixture.

Constructs a tiny dense `Gemma4ForCausalLM` from the official Transformers
Gemma 4 text implementation, seeds every trainable parameter deterministically,
renames the saved text tensors into ferrl's public conditional-generation
checkpoint prefix (`model.language_model.*`), and dumps fp32 logits at every
position for tiny full-sequence forwards.

The upstream reference class in transformers 5.11.0 exposes the equivalent
dense text config as `gemma4` / `gemma4_text`. It does not expose ferrl's
`gemma4_unified` / `gemma4_unified_text` wrapper names, so this fixture pins
the dense text math while ferrl's Rust config tests cover the unified alias
being routed into the same dense text loader.

Outputs (committed under `crates/ferrl/tests/fixtures/tiny_gemma4/`):
  config.json + model.safetensors  (the tiny checkpoint) and golden.json
  (inputs, per-position logits, version-pinned meta).

Pinned env `ferrl-oracle` (transformers 5.11.0, CPU torch 2.12.0); regenerate
only when the pin moves:

    conda activate ferrl-oracle && python scripts/oracle/gen_gemma4_tiny_golden.py
"""

import json
import pathlib

import torch
import transformers
from safetensors.torch import save_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

OUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "crates/ferrl/tests/fixtures/tiny_gemma4"
)
SEED = 42
WEIGHT_STD = 0.35
TRANSFORMERS_PIN = "5.11.0"
TORCH_PIN = "2.12.0"


def flat(t: torch.Tensor) -> list[float]:
    return t.detach().to(torch.float32).flatten().tolist()


def tiny_text_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=16,
        vocab_size_per_layer_input=16,
        hidden_size=8,
        hidden_size_per_layer_input=0,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        num_kv_shared_layers=0,
        head_dim=4,
        global_head_dim=8,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        sliding_window=3,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        },
        use_cache=True,
        use_bidirectional_attention="vision",
        use_double_wide_mlp=False,
        enable_moe_block=False,
        num_experts=None,
        top_k_experts=None,
        expert_intermediate_size=None,
        moe_intermediate_size=None,
    )


def seed_weights(model: torch.nn.Module) -> None:
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * WEIGHT_STD)
    model.tie_weights()


def ferrl_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("model."):
            out[f"model.language_model.{name[len('model.'):]}"] = tensor.detach().contiguous()
    return out


def ids_row(length: int, stride: int, offset: int, vocab: int) -> list[int]:
    return [(i * stride + offset) % vocab for i in range(length)]


def main() -> None:
    assert transformers.__version__ == TRANSFORMERS_PIN, transformers.__version__
    assert torch.__version__.startswith(TORCH_PIN), torch.__version__
    torch.manual_seed(SEED)

    cfg = tiny_text_config()
    model = Gemma4ForCausalLM(cfg)
    seed_weights(model)
    model = model.eval().to(torch.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(ferrl_state_dict(model), OUT_DIR / "model.safetensors")

    text_config = cfg.to_dict()
    text_config["model_type"] = "gemma4_text"
    config = {
        "model_type": "gemma4",
        "tie_word_embeddings": True,
        "text_config": text_config,
    }
    with (OUT_DIR / "config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    vocab = cfg.vocab_size
    cases: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        full_b1 = torch.tensor([ids_row(6, 5, 3, vocab)], dtype=torch.long)
        out = model(input_ids=full_b1, use_cache=False)
        cases["full_b1"] = {
            "input_ids": full_b1.tolist(),
            "logits": flat(out.logits),
            "shape": list(out.logits.shape),
        }

        full_b2 = torch.tensor(
            [ids_row(7, 3, 1, vocab), ids_row(7, 5, 4, vocab)],
            dtype=torch.long,
        )
        out = model(input_ids=full_b2, use_cache=False)
        cases["full_b2"] = {
            "input_ids": full_b2.tolist(),
            "logits": flat(out.logits),
            "shape": list(out.logits.shape),
        }

    obj = {
        "meta": {
            "generator": "scripts/oracle/gen_gemma4_tiny_golden.py",
            "reference": "transformers.models.gemma4.Gemma4ForCausalLM/Gemma4TextConfig",
            "fixture_config_shape": "gemma4/gemma4_text",
            "unified_note": (
                "transformers 5.11.0 exposes the dense text reference as "
                "gemma4/gemma4_text; ferrl accepts gemma4_unified aliases as "
                "the same dense text loader shape."
            ),
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
