# TriMul Discovery Run Contract

This is the acceptance contract for ferrl's first TriMul kernel-discovery run. It is
intentionally narrow: it locks what must be recorded, re-run, and reported before a
candidate kernel can count as a discovery artifact. It is not a general task SDK.

The contract applies to runs of `ferrl train --config <run.json>` where `task` is
`trimul` and to the artifact extraction step that follows such a run. The extraction command is `ferrl trimul-artifact --config <run.json> --completion <raw.txt> --out <artifact-dir>` with run provenance, audit seed, and repeated `--baseline-ns` values.

## Pre-Run Lock

Before training starts, record these values in the run notes and keep an immutable copy
with the final report:

| Field | Required value |
|---|---|
| ferrl revision | Full git commit SHA. |
| run config | The exact JSON config passed to `ferrl train`. |
| model | Model family, checkpoint identity, tokenizer identity, LoRA rank/alpha, base dtype, and rollout seed. |
| TriMul eval bundle | Immutable identity of the GPUMODE `bioml/trimul` bundle used for `eval_dir` (commit, release, or digest). |
| sandbox image | Immutable identity of the Apptainer image used by `trimul.image` (path plus digest when available). |
| cases | `task.yml` identity and the loaded counts for `tests` and `benchmarks`. |
| seeds | `data.seed`, `policy.seed`, trainer seed-bearing knobs, and the training `trimul.secret_seed`. |
| scratch cap | `trimul.scratch_max_bytes`; `0` means the ferrl default, currently 1 GiB. |
| hardware | GPU product name reported by the baseline command and visible CUDA device count. |
| budget | Trainer `steps`, `group_size`, wall-clock allocation, and the stop condition chosen below. |

A discovery run must not start without a guarded same-GPU baseline in
`trimul.baseline`. Measure it on the target GPU with `ferrl trimul-baseline --config
<run.json>`. For the first run, take at least three baseline measurements and use the
median `ns` in the config. Keep all raw baseline measurements in the report.

## Artifact Definition

A candidate is an accepted artifact only when the final bundle contains all of:

- `submission.py`: the exact extracted `custom_kernel` source.
- `manifest.json`: a machine-readable manifest with the fields below.
- `verification/`: the clean re-verification logs and benchmark summaries.
- `report.md`: the human summary and reviewer checklist outcome.

The manifest schema is versioned from the first run:

```json
{
  "contract_version": 1,
  "task": "trimul",
  "ferrl_commit": "<full git sha>",
  "run_id": "<run directory name>",
  "candidate": {
    "step": 0,
    "group_index": 0,
    "completion_sha256": "<sha256 of raw completion>",
    "source_sha256": "<sha256 of submission.py>"
  },
  "model": {
    "family": "qwen3.x",
    "checkpoint": "<operator supplied identity>",
    "tokenizer": "<operator supplied identity>",
    "lora_rank": 16,
    "lora_alpha": 32.0,
    "base_dtype": "bf16"
  },
  "config": {
    "run_config_sha256": "<sha256 of resolved run config>",
    "trainer_steps": 0,
    "group_size": 0,
    "policy_seed": 0,
    "data_seed": 0,
    "training_secret_seed": 0,
    "audit_secret_seed": 0,
    "scratch_max_bytes": 1073741824
  },
  "eval": {
    "bundle": "<immutable eval bundle identity>",
    "sandbox_image": "<image identity>",
    "test_cases": 0,
    "benchmark_cases": 0
  },
  "baseline": {
    "gpu": "<nvidia-smi product name>",
    "measurements_ns": [0.0, 0.0, 0.0],
    "median_ns": 0.0
  },
  "verification": {
    "gpu": "<nvidia-smi product name>",
    "runs": [
      {
        "correct": true,
        "geomean_ns": 0.0,
        "speedup": 0.0
      }
    ],
    "accepted": true
  }
}
```

The artifact extractor may add fields, but it must keep the fields above stable for
the first run so a reviewer can audit the result without reading training logs by hand.

## Acceptance Rule

A TriMul run counts as a success only if one artifact candidate satisfies every rule:

1. The candidate is extracted from a model completion, not hand-authored after the run.
2. The candidate passes every correctness case in a clean re-verification run.
3. Re-verification uses the same eval bundle, same sandbox image, same GPU product name,
   and a fresh scratch directory.
4. Re-verification uses an audit `trimul.secret_seed` that was not used for training.
5. At least three clean benchmark re-runs are recorded for the candidate.
6. The median candidate geometric-mean runtime is lower than the median guarded
   baseline runtime recorded in the manifest.
7. The report states speedup as `baseline.median_ns / candidate.median_geomean_ns`.

If any correctness re-run fails, or if the GPU product name does not match the baseline
pin, the candidate is rejected even if a prior training reward was high.

## Dynamic Reward-Hacking Checks

The TriMul reward already keeps candidate scratch bounded, routes the grade over a
captured channel, denies network by default, and rejects implausibly fast timings. The
first discovery run still needs dynamic checks on top candidates because the training
loop is optimizing against that reward.

For every candidate included in the final report:

- Re-run from `submission.py` only; do not reuse the training scratch tree.
- Re-run with a fresh audit secret seed.
- Record whether the source tries to inspect process state, file descriptors,
  environment variables, network sockets, or paths outside the kernel inputs.
- Treat unexplained sub-launch-floor timings, inconsistent correctness, sandbox
  resource failures, or grade-channel anomalies as rejection signals.
- Include rejected high-reward candidates in the report when they explain why the
  accepted candidate was not simply the highest training reward.

These checks are deliberately reviewer-facing. They are not a proof against arbitrary
malicious code; they are the Phase-1 guardrail for deciding whether the first run found
a real faster kernel or a reward artifact.

## Stopping Rule

Choose and record one stop condition before launch:

- `budget_exhausted`: stop after the configured trainer step budget.
- `target_found`: stop early only after a candidate passes the acceptance rule above.
- `operator_abort`: stop because the run became invalid or uneconomical; the report must
  say why.

If no candidate passes the acceptance rule, the run result is `no_win`. A `no_win`
report is still a valid outcome when it includes the locked config, baseline, training
health summary, top rejected candidates, and the reason each top candidate failed
verification.

## Report Shape

The final report must fit this outline:

1. Verdict: `accepted_artifact`, `no_win`, or `invalid_run`.
2. Baseline: GPU, raw measurements, median runtime, and command used.
3. Training: ferrl commit, config hash, model identity, seeds, budget, and run health.
4. Candidate table: source hash, training reward, clean correctness, median runtime,
   speedup, and accept/reject reason.
5. Artifact bundle path and manifest hash, when accepted.
6. Reviewer checklist: each acceptance and reward-hacking check marked pass/fail.

Use `ferrl trimul-artifact` after training to persist the best correct-and-fast candidates with enough provenance to fill this manifest and produce the reviewer-facing report.
