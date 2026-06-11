#!/usr/bin/env python3
"""Fetch Qwen/Qwen3.5-0.8B-Base from ModelScope into Research_Realm/assets/.

Run on the cluster in the `ferrl-oracle` env (ModelScope is the reachable
mirror there; HF is blocked). The asset lands OUTSIDE the repo
(`../assets/qwen3_5-0.8b-base`), like the other model assets. Verify shas
against the ModelScope manifest afterwards (see the M1 llama playbook).
"""

import pathlib

from modelscope import snapshot_download

ASSET_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets/qwen3_5-0.8b-base"

if __name__ == "__main__":
    p = snapshot_download("Qwen/Qwen3.5-0.8B-Base", local_dir=str(ASSET_DIR))
    print(f"downloaded to {p}")
