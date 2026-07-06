# TriMul Discovery Run Contract

This is the acceptance contract for ferrl's first TriMul kernel-discovery run. It is
intentionally narrow: it locks what must be recorded, re-run, and reported before a
candidate kernel can count as a discovery artifact. It is not a general task SDK.

The contract applies to runs of `ferrl train --config <run.json>` where `task` is
`trimul` and to the artifact extraction step that follows such a run. Set
`trainer.candidate_log_top_k` high enough to persist the best sampled completions
in `candidates.jsonl`; that ledger is the source for the raw completion and the
`--step` / `--prompt-index` / `--group-index` / `--rank` / `--world-size`
coordinates passed to artifact extraction. TriMul candidate rows may also include
`reward_diagnostic` (for example no submission, test failure, no pass grade, sandbox
timeout, or missing/implausible benchmark data); preserve it in the run report when
explaining low- or zero-reward tails. For reward-tail triage, set `candidate_log_top_k >=
group_size` so every sampled completion is retained. At launch, `ferrl train`
freezes the exact configured model prompt to `<run-dir>/prompt.txt` and writes its
digest to `<run-dir>/prompt.sha256`. `trimul.prompt_path` is the complete rendered
model prompt: ferrl does not trim, wrap, prepend, append, or otherwise construct
prompt text. Select completion parsing separately with
`trimul.submission_extract_mode`, which must be either `final_fence` or
`thinking_after_think` and never changes prompt bytes. The extraction command is `ferrl
trimul-artifact --config <run.json> --prompt-copy <run-dir>/prompt.txt
--completion <raw.txt> --out <artifact-dir> --run-id <run-id> --step <step>
--prompt-index <prompt-index> --group-index <group-index> --rank <rank>
--world-size <world-size> --training-reward <reward> --audit-secret-seed <seed>
--baseline-ns <ns> --baseline-ns <ns> --baseline-ns <ns> --ferrl-commit <sha>
--run-health <summary> --source-inspection clean --source-inspection-notes
<notes>` with run provenance, audit seed, source-inspection evidence, the frozen
prompt copy, and repeated `--baseline-ns` values. Artifact extraction verifies
that `--prompt-copy` matches the adjacent launch-time `prompt.sha256`.

For rollout-only diagnostics from an external inference runtime, use `ferrl
trimul-score --config <run.json> --prompt-copy <prompt.txt> --completion <raw.txt>
--out <scores.jsonl> --score-secret-seed <seed>` or pass `--completions-jsonl
<jsonl>`. The scoring seed must differ from the training `trimul.secret_seed`.
The JSONL input rows must contain `completion` and may include `step`,
`prompt_index`, `group_index`, `rank`, `world_size`, `completion_len_tokens`,
`source_id`, `metadata`, and `reward_metadata`; `world_size` must be nonzero and
`rank` must be inside it. The output is external-score JSONL with the shaped
reward, reward diagnostic, top-level TriMul reward metadata, prompt/config
hashes, completion hash, opaque/public-safe `source_id`, and namespaced external
rollout provenance. Input file paths are not persisted into diagnostic evidence;
use `--source-label <public-id>` or row-level `source_id` values that are safe to
copy into public reports. The default is strict and scores the completion bytes
exactly as supplied. For GGUF rollouts generated through llama.cpp, pass
`--completion-normalization llama-cpp`; this strips only llama.cpp's trailing
`[end of text]` stdout sentinel before extraction and records raw and normalized
hash/length metadata. `trimul-score` is a search-quality diagnostic for comparing
external rollouts; it does not replace `trimul-artifact` and cannot by itself satisfy
the artifact acceptance rule.

Use the same `--completion-normalization` value when promoting an external candidate
to `ferrl trimul-artifact`. Artifact bundles always preserve the raw model output as
`completion.txt`; when normalization changes the text used for extraction, they also
write `completion.normalized.txt` and record the normalization mode plus hashes in
`manifest.json`.

For prompt-controlled runs, `trimul.prompt_path` is only the mutable launch-time
path for the complete rendered model prompt. Do not use that local path as artifact
provenance: it may change and may expose private filesystem layout. `ferrl train`
freezes the exact prompt file bytes into the run directory as `prompt.txt` and
records `prompt.sha256`; `ferrl trimul-artifact` verifies the adjacent
`prompt.sha256`, copies the immutable rendered prompt into the artifact bundle as
`prompt.txt`, and records `prompt_sha256`. Any operator-facing path in notes should
be redacted or replaced by a stable non-private identifier. TriMul training has no
built-in prompt fallback, no suffix prompt path, and no prompt wrapper, so the run
prompt is fully owned in one editable file before launch and frozen by the
run/artifact copy and hash after launch.

TriMul training reward is shaped for search density, not artifact acceptance. The
current reward scheme gives tiny credit for extractable code, small credit for
reaching the test harness, bounded partial credit for passing individual test cases,
and then the correctness floor for test-passing candidates whose eval reaches a
benchmark exit marker. A successful plausible benchmark adds a capped speed
component. Implausibly fast benchmark timings still score zero. The artifact
acceptance rule below stays strict: held-out correctness, repeated same-GPU
benchmarking, and measured speedup over the pinned baseline.

The run-config schema accepts the explicit reward profile below. Omit `trimul.reward`
to use these `trimul_shaped_v1` defaults, or tune the numeric values to adjust
discovery density. Custom profiles must preserve the reward ladder:
`format_extracted <= runnable` and
`runnable + partial_correctness <= correctness`; implausibly fast benchmark timings
remain fail-closed at zero.

```jsonc
"trimul": {
  "reward": {
    "scheme": "trimul_shaped_v1",
    "format_extracted": 0.02,
    "runnable": 0.05,
    "partial_correctness": 0.75,
    "correctness": 1.0,
    "speed_cap": 2.0,
    "implausible_benchmark": "zero"
  }
}
```

The top-level `run_health` schema is reserved for reward/correctness collapse,
dropped-row, grad-spike, dark-telemetry, and source-dominance policies. Non-empty
policies are rejected until the configurable run-health follow-up lands; for now,
record the intended stop condition and use `ferrl runreport` plus candidate-ledger
diagnostics as the operator-facing health evidence.

### Preparing a Qwen rendered prompt

For Qwen3.5/3.6 instruct checkpoints that use ChatML, `trimul.prompt_path` should
already contain the rendered chat-template bytes. ferrl will not call a chat-template
renderer for TriMul. A thinking prompt has this structure:

```text
<|im_start|>system
{system/output contract}<|im_end|>
<|im_start|>user
{TriMul task prompt}<|im_end|>
<|im_start|>assistant
<think>
```

Set `trimul.submission_extract_mode` to `thinking_after_think` for that prompt shape:
the extractor requires the model to emit `</think>`, then extracts the final fenced
Python code block from the answer region. For a non-thinking prompt whose completion
is just the final answer region, use `final_fence` instead and omit the `<think>`
assistant prefill. If a checkpoint uses a different chat template, render that
checkpoint's complete prompt format yourself and keep it in the single
`trimul.prompt_path` file.

## Pre-Run Lock

Before training starts, record these values in the run notes and keep an immutable copy
with the final report:

| Field | Required value |
|---|---|
| ferrl revision | Full git commit SHA. |
| run config | The exact JSON config passed to `ferrl train`. |
| prompt | The exact rendered model prompt bytes, frozen as `<run-dir>/prompt.txt` plus `prompt.sha256`; do not rely on a mutable local `trimul.prompt_path` for provenance. |
| submission extraction | `trimul.submission_extract_mode` (`final_fence` or `thinking_after_think`); this controls parsing only and must not construct prompt text. |
| reward profile | `trimul.reward`; defaults to `trimul_shaped_v1`, with custom ladder-preserving values allowed. |
| run-health policy | `run_health`; currently empty/default only, with the intended stop condition recorded separately below. |
| model | Model family, checkpoint identity, tokenizer identity, LoRA rank/alpha, base dtype, and rollout seed. |
| TriMul eval bundle | Immutable identity of the GPUMODE `bioml/trimul` bundle used for `eval_dir` (commit, release, or digest). |
| sandbox image | Immutable identity of the Apptainer image used by `trimul.image` (path plus digest when available). |
| cases | `task.yml` identity and the loaded counts for `tests` and `benchmarks`. |
| seeds | `data.seed`, `policy.seed`, trainer seed-bearing knobs, and the training `trimul.secret_seed`. |
| scratch cap | `trimul.scratch_max_bytes`; `0` means the ferrl default, currently 1 GiB. |
| candidate ledger | `trainer.candidate_log_top_k`; use a positive value for discovery runs, and use at least `group_size` when diagnosing low- or zero-reward tails so all completions are persisted in `candidates.jsonl`; retain any `reward_diagnostic` values in the report. |
| hardware | GPU product name reported by the baseline command and visible CUDA device count. |
| budget | Trainer `steps`, `group_size`, wall-clock allocation, and the stop condition chosen below. |

A discovery run must not start without a guarded same-GPU baseline in
`trimul.baseline`. Measure it on the target GPU with `ferrl trimul-baseline --config
<run.json>`. For the first run, take at least three baseline measurements and use the
median `ns` in the config. Keep all raw baseline measurements in the report.

## Artifact Definition

A candidate is an accepted artifact only when the final bundle contains all of:

- `submission.py`: the exact extracted `custom_kernel` source.
- `prompt.txt`: the exact rendered TriMul model prompt used for generation,
  copied from `<run-dir>/prompt.txt` after verifying `<run-dir>/prompt.sha256`.
- `manifest.json`: a machine-readable manifest with the fields below.
- `verification/`: the clean re-verification logs and benchmark summaries.
- `report.md`: the human summary and operator checklist outcome.

The manifest schema is versioned from the first run:

```json
{
  "contract_version": 1,
  "task": "trimul",
  "ferrl_commit": "<full git sha>",
  "run_id": "<run directory name>",
  "candidate": {
    "step": 0,
    "prompt_index": 0,
    "group_index": 0,
    "rank": 0,
    "world_size": 1,
    "training_reward": 0.0,
    "completion_sha256": "<sha256 of raw completion>",
    "source_sha256": "<sha256 of submission.py>",
    "source_inspection": {
      "result": "clean",
      "notes": "<process/file-descriptor/environment/network/out-of-input path inspection notes>"
    }
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
    "prompt_sha256": "<sha256 of prompt.txt>",
    "prompt_file": "prompt.txt",
    "reward_profile": {
      "scheme": "trimul_shaped_v1",
      "format_extracted": 0.02,
      "runnable": 0.05,
      "partial_correctness": 0.75,
      "correctness": 1.0,
      "speed_cap": 2.0,
      "implausible_benchmark": "zero"
    },
    "trainer_steps": 0,
    "group_size": 0,
    "run_health": "<runreport summary or run notes>",
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
    "median_ns": 0.0,
    "command": "ferrl trimul-baseline --config <run.json>"
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
the first run so an operator can audit the result without reading training logs by hand.

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

These checks are deliberately operator-facing. They are not a proof against arbitrary
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
3. Training: ferrl commit, config hash, prompt copy/hash, model identity, seeds,
   budget, and run health.
4. Candidate table: source hash, training reward, source-inspection result, clean
   correctness, median runtime, speedup, and accept/reject reason.
5. Artifact bundle path and manifest hash, when accepted.
6. Operator checklist: each acceptance and reward-hacking check marked pass/fail.

Use `ferrl trimul-artifact` after training to persist the best correct-and-fast candidates with enough provenance to fill this manifest and produce the operator-facing report.
