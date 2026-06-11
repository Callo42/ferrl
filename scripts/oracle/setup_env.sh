#!/bin/bash
# Set up the pinned `ferrl-oracle` conda env on the cluster (M2' D3 decision).
#
# transformers is pinned to v5.11.0 (newest stable containing qwen3_5; hard
# floor v5.7.0 = the GDN multi-token cached-forward fix). torch is CPU-only:
# logit/fixture dumps need no GPU and the CPU path is deterministic.
#
# Run on the login node:  bash scripts/oracle/setup_env.sh
set -euo pipefail

CONDA_ROOT="$HOME/private/homefile/miniconda3"
ENV_NAME="ferrl-oracle"
TRANSFORMERS_PIN="5.11.0"

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

# Prefer the CPU-only torch wheel; fall back to the PyPI build if the PyTorch
# index is unreachable from the cluster.
if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.*"; then
    echo "download.pytorch.org unreachable; falling back to PyPI torch"
    python -m pip install "torch==2.*"
fi

python -m pip install "transformers==$TRANSFORMERS_PIN" safetensors modelscope

python - <<'PYEOF'
import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

print("torch", torch.__version__)
print("transformers", transformers.__version__)
assert transformers.__version__ == "5.11.0", transformers.__version__
for fn in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
    assert hasattr(m, fn), f"missing {fn} in qwen3_5 modeling"
print("qwen3_5 reference kernels present")
PYEOF

echo "ORACLE-ENV-READY"
