#!/bin/bash
# Set up the pinned `ferrl-oracle` conda env on the cluster (M2' D3 decision).
#
# transformers is pinned to v5.11.0 (newest stable containing qwen3_5; hard
# floor v5.7.0 = the GDN multi-token cached-forward fix). torch is CPU-only:
# logit/fixture dumps need no GPU and the CPU path is deterministic.
#
# Run on the login node:  bash scripts/oracle/setup_env.sh
# Override CONDA_ROOT if conda is not installed at ~/miniconda3.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="ferrl-oracle"
TRANSFORMERS_PIN="5.11.0"
TORCH_PIN="2.12.0"

if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "error: no conda at CONDA_ROOT=$CONDA_ROOT (set CONDA_ROOT to your conda install)" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# conda-forge only: the Anaconda default channels are ToS-gated (and the ToS
# acceptance is an org-level licensing decision); conda-forge is open.
if conda env list | grep -q "^$ENV_NAME "; then
    echo "env $ENV_NAME already exists; activating"
else
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.12
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

# torch is pinned exactly: the fixtures are a numeric contract, and a torch
# bump can shift the reference numerics — regeneration under a different
# version must be a deliberate, reviewed act (the Rust gates assert both
# pins from the fixture metadata). Prefer the CPU-only wheel; fall back to
# the PyPI build if the PyTorch index is unreachable.
if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH_PIN"; then
    echo "download.pytorch.org unreachable; falling back to PyPI torch"
    python -m pip install "torch==$TORCH_PIN"
fi

python -m pip install "transformers==$TRANSFORMERS_PIN" safetensors modelscope

python - <<'PYEOF'
import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

print("torch", torch.__version__)
print("transformers", transformers.__version__)
assert transformers.__version__ == "5.11.0", transformers.__version__
assert torch.__version__.startswith("2.12.0"), torch.__version__
for fn in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
    assert hasattr(m, fn), f"missing {fn} in qwen3_5 modeling"
print("qwen3_5 reference kernels present")
PYEOF

echo "ORACLE-ENV-READY"
