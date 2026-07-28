# TriMul Discovery Run Contract

This is the acceptance contract for ferrl's first TriMul kernel-discovery run. It is
intentionally narrow: it locks what must be recorded, re-run, and reported before a
candidate kernel can count as a discovery artifact. It is not a general task SDK.

The contract applies to runs of `ferrl train --config <run.json>` where `task` is
`trimul` and to the artifact extraction step that follows such a run. Set
`trainer.candidate_log_top_k` high enough to persist the best sampled completions
in `candidates.jsonl`; every production row is integrity-bound to the immutable
launch payload, carries its own `record_sha256`, and is authenticated by the
process-local Ed25519 key whose public half is committed in the launch. A later
process cannot recover that capability: launch-bound candidate logging rejects
a non-empty ledger before append, so checkpoint continuation must use a new run
identity or disable candidate logging. TriMul candidate rows may also include
`reward_diagnostic` (for example no submission, test failure, no pass grade, sandbox
timeout, or missing/implausible benchmark data); preserve it in the run report when
explaining low- or zero-reward tails. For reward-tail triage, set `candidate_log_top_k >=
group_size` so every sampled completion is retained. At launch, `ferrl train` uses the
explicit top-level `launch_authentication` mode. The default `local_ephemeral_v1`
commits an unsigned launch for no-admin discovery; `external_attested_v1` obtains a
detached signature from the protected launch attestor and verifies it against the
external trust policy. There is no fallback between them. The launch payload binds the full resolved config,
synchronized run identity, build-embedded training commit,
exact model/checkpoint loader identity, exact tokenizer bytes, prompt, and candidate
ledger contract. For TriMul it also binds the SHA-256 and byte length of the sandbox
image, a digest and file count for the complete ordered eval tree, and the exact
`task.yml` SHA-256 and byte length, plus the selected verifier isolation tier, backend
preflight evidence, timing metric, and protected runtime-control probe. Ferrl streams
and hashes the image once into a
kernel-sealed anonymous descriptor, gives every eval-tree file the same sealed
storage, binds those exact descriptors rather than writable pathnames, revalidates
source identities and kernel seals after attestation and around verifier use, and
aborts in lockstep on a rank-local substitution. The prompt
is frozen to `<run-dir>/prompt.txt`; the compatibility
digest remains at `<run-dir>/prompt.sha256`. `trimul.prompt_path` is the complete rendered
model prompt: ferrl does not trim, wrap, prepend, append, or otherwise construct
prompt text. Select completion parsing separately with
`trimul.submission_extract_mode`, which must be either `final_fence` or
`thinking_after_think` and never changes prompt bytes. The extraction command is `ferrl
trimul-artifact --run-dir <run-dir> --candidate-sha256 <record_sha256>
--out <artifact-dir> --audit-secret-seed <seed> --baseline-ns <ns>
--baseline-ns <ns> --baseline-ns <ns> --run-health <summary>
--source-inspection clean --source-inspection-notes <notes>`. Artifact extraction
requires `launch.json` to have the exact production-canonical encoding, then validates
its attestation, every candidate row, the exact selected row, the frozen prompt, and the
live verifier assets before GPU detection or audit verification. It does not accept operator-authored
completion, coordinate, reward, run, commit, model, tokenizer, eval, or sandbox
provenance fields. The current artifact command accepts only an
`external_attested_v1` launch audited under `dedicated_uid_service_v1`; a local or
same-UID discovery run remains training evidence and returns a clear higher-assurance
audit-required error.

## Launch Authentication

`local_ephemeral_v1` is the default discovery mode. It requires no system trust policy
or attestor socket and permits launch-bound, signed candidate ledgers. It protects the
run against accidental drift, cross-run substitution, and later loss of the ephemeral
candidate signing key. It does not authenticate the launch against a process already
controlling the same host account, so it cannot authorize artifact publication.

`external_attested_v1` is the optional higher-assurance launch mode used for accepted
publication. Candidate-producing training in this mode and artifact extraction have no
CLI or run-config override for the trust root. Deployment provides two protected system
surfaces:

- `/etc/ferrl/launch-trust.json`: a root-owned regular file whose parent chain and
  contents are not group/world writable. It lists the accepted Ed25519 public keys.
  Key rotation adds a new `key_id`; retain old verification keys for every artifact
  whose launch remains admissible.
- `/run/ferrl/launch-attestor.sock`: a root-owned, non-world-writable Unix socket under
  a protected parent chain. Its group may grant the live launcher access. The external
  service owns the private key and must authorize only the live
  launcher/job; the run and artifact operator must never receive the root private key
  or an unrestricted signing capability.

Socket access alone is not authorization. The attestor must bind every request to the
protected scheduler/launcher identity, approved executable measurement and source
commit, expected job/config policy, and permitted DP/TP rank set; it must reject replay
or requests beyond that launch. A generic daemon that signs any well-formed request
from the socket group does not satisfy this contract. The launcher must also isolate the
training process from the artifact operator while the delegated candidate private key
is live (including ptrace, process-memory, core-dump, and `/proc` credential exposure);
otherwise the operator could steal the delegated capability instead of replacing it.

The trust policy is strict JSON:

```json
{
  "contract_version": 1,
  "kind": "ferrl.run-launch-trust-policy",
  "keys": [
    {
      "key_id": "cluster-launch-2026-01",
      "algorithm": "ed25519",
      "public_key": "<64 lowercase hex characters>"
    }
  ]
}
```

Ferrl sends one newline-terminated
`ferrl.run-launch-attestation-request` JSON object containing contract version 1,
algorithm `ed25519`, the exact compact launch-payload JSON bytes as lowercase hex,
and their domain-separated SHA-256. The service must decode and strictly parse those
exact bytes, recompute and policy-check the payload hash, return one strict
`ferrl.run-launch-attestation` object, and close the connection. The response carries
the trusted `key_id`, the same launch digest, and a 128-character lowercase Ed25519
signature over the domain-separated attestation message. Ferrl verifies that response
against the protected policy before creating the run directory. Absence, malformed
policy, service rejection, an untrusted key, or a bad signature fails before launch,
candidate, metrics, checkpoint, rollout, reward, or optimizer publication.

The wire shapes are:

```json
{
  "contract_version": 1,
  "kind": "ferrl.run-launch-attestation-request",
  "algorithm": "ed25519",
  "launch_payload_sha256": "<64 lowercase hex characters>",
  "launch_payload_json_hex": "<lowercase hex of exact compact payload JSON>"
}
```

```json
{
  "contract_version": 1,
  "kind": "ferrl.run-launch-attestation",
  "algorithm": "ed25519",
  "key_id": "cluster-launch-2026-01",
  "launch_payload_sha256": "<same digest>",
  "signature": "<128 lowercase hex characters>"
}
```

Domain hashes use SHA-256 over `u64_le(domain_length) || domain ||
u64_le(field_length) || field` for each ordered field. The payload domain is
`ferrl.run-launch.payload.v2` with the exact compact JSON bytes as its sole field.
The attestation message is the 64-byte lowercase hexadecimal result of the same
construction under domain `ferrl.run-launch-attestation.v1`, with the ASCII launch
payload digest as its sole field. Ed25519 signs those 64 message bytes directly.

`trimul-artifact` loads the same protected policy independently. Replacing the entire
run directory, recomputing every unkeyed hash, and signing rows under an
operator-generated key therefore remains rejected: the replacement launch lacks a
signature under a key trusted outside that directory.

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

External-score rows are diagnostic evidence and are not artifact inputs. The strict
artifact path accepts only a native, launch-bound `candidates.jsonl` row and preserves
that row as `candidate.json`, its completion as `completion.txt`, and the exact verified
run manifest as `launch.json`.

For prompt-controlled runs, `trimul.prompt_path` is only the mutable launch-time
path for the complete rendered model prompt. Do not use that local path as artifact
provenance: it may change and may expose private filesystem layout. `ferrl train`
freezes the exact prompt file bytes into the run directory as `prompt.txt` and
records `prompt.sha256`; `ferrl trimul-artifact` verifies the prompt against
`launch.json`, copies the immutable rendered prompt into the artifact bundle as
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
"launch_authentication": "local_ephemeral_v1",
"trimul": {
  "verifier_isolation_tier": "same_uid_apptainer_v1",
  "verifier_apptainer_bin": "/usr/bin/apptainer",
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

Omit `verifier_apptainer_bin` to use `/usr/bin/apptainer`. For the optional stronger
backend, set `verifier_isolation_tier` to `dedicated_uid_service_v1`, omit
`verifier_apptainer_bin`, and optionally set `verifier_executor_socket`. Mixed backend
fields are rejected, and an unavailable selected backend never falls back.

The top-level `run_health` schema configures post-run reward/correctness collapse,
dropped-row, grad-spike, dark-telemetry, and source-dominance policies. `warn` reports a
finding; `fail` makes `ferrl train` fail after telemetry is written and makes
`ferrl runreport --config <run.json>` exit with code `2`. The `stop` action is reserved
for a future in-run gate and is rejected today. Correctness collapse and source dominance
depend on `candidates.jsonl`, so set
`trainer.candidate_log_top_k >= trainer.group_size` when using those rules. Partial
top-K candidate ledgers fail closed for those checks because they cannot represent the
whole sampled group.

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
| ferrl revision | Full git commit SHA embedded when the binary is built from a clean Git tree and sealed in `launch.json`; no runtime revision claim is accepted. |
| launch authentication | Exact `launch_authentication` mode. `local_ephemeral_v1` needs no external service and is discovery-only. `external_attested_v1` additionally records a trusted external `key_id`, Ed25519 algorithm, and detached launch signature verified through `/etc/ferrl/launch-trust.json`; the private key and signing authority stay outside ferrl and the operator-writable run directory. |
| run config | Original file SHA-256 plus the complete canonical resolved config stored in `launch.json`. |
| prompt | The exact rendered model prompt bytes, frozen as `<run-dir>/prompt.txt` and sealed in `launch.json`; `prompt.sha256` remains a compatibility sidecar. Do not rely on a mutable local `trimul.prompt_path` for provenance. |
| submission extraction | `trimul.submission_extract_mode` (`final_fence` or `thinking_after_think`); this controls parsing only and must not construct prompt text. |
| reward profile | `trimul.reward`; defaults to `trimul_shaped_v1`, with custom ladder-preserving values allowed. |
| run-health policy | `run_health`; post-run warn/fail policy, including the original top-level config passed to `ferrl runreport --config`. |
| model | Loader-derived family, exact model/checkpoint policy SHA-256, exact tokenizer-file SHA-256, resolved EOS, LoRA rank/alpha, base dtype, and rollout seed, all sealed at launch. |
| verifier assurance | Exact `trimul.verifier_isolation_tier`, backend preflight evidence and digest, tier-specific timing metric, and protected runtime-control evidence. `same_uid_apptainer_v1` is the default no-admin training tier: it uses a private user-owned mode-`0700` work root and a canonical root-owned Apptainer executable, but explicitly does not resist arbitrary hostile peers under the same host UID. `dedicated_uid_service_v1` uses a protected Unix socket and a distinct non-root service UID that authenticates `SO_PEERCRED`; it is the only audit tier accepted by the current artifact command. Neither tier falls back to the other. |
| TriMul eval bundle | SHA-256 over every ordered relative regular-file name and byte under `eval_dir`, plus the exact file count. Every captured file is held in a Linux kernel-sealed anonymous descriptor; after all seals are installed, the ordered descriptor contents are rehashed and required to equal the launch identity. The selected backend copies and rehashes each asset into a unique private request directory and supplies only the resulting read-only paths to Apptainer. The dedicated backend receives descriptors over `SCM_RIGHTS`; the same-UID backend stages them in process. The configured source path is informational only. |
| sandbox image | SHA-256 and byte length of the exact Apptainer image streamed into a kernel-sealed anonymous descriptor. After sealing, the descriptor is rehashed and required to equal the captured identity. The selected backend copies and rehashes it in private storage before launch. The configured source path is informational only. |
| cases | Attested `task.yml` SHA-256 and byte length plus the loaded counts for `tests` and `benchmarks`. |
| seeds | `data.seed`, `policy.seed`, trainer seed-bearing knobs, and the training `trimul.secret_seed`. |
| scratch cap | `trimul.scratch_max_bytes`; `0` means the ferrl default, currently 1 GiB. |
| verifier process cap | `trimul.verifier_max_procs`; `0` means the TriMul default, currently `1024`. This is a per-UID `RLIMIT_NPROC` cap, not a per-container task count. |
| candidate ledger | `trainer.candidate_log_top_k`; use a positive value for discovery runs, and use at least `group_size` for `run_health.correctness_collapse`, `run_health.source_dominance`, and low- or zero-reward tail diagnosis. Every persisted row must carry the launch digest, its exact `record_sha256`, and a valid Ed25519 `record_signature` under the public key frozen in `launch.json`; retain any `reward_diagnostic` values in the report. |
| trainer scalar controls | Exact `trainer.lr_schedule` and `trainer.beta_schedule` when present, otherwise the scalar `lr`, `warmup_steps`, and `beta` values. Schedules are deterministic step-index functions and must be copied with the final run config. |
| hardware | GPU product name reported by the baseline command and visible CUDA device count. |
| budget | Trainer `steps`, `group_size`, wall-clock allocation, and the stop condition chosen below. |

A discovery run must not start without a guarded same-GPU baseline in
`trimul.baseline`. Its `metric`, `isolation_tier`, and
`isolation_evidence_sha256` must exactly match the active preflight. The metric is
`same-uid-apptainer-latency-v1` for `same_uid_apptainer_v1` and
`isolated-service-latency-v1` for `dedicated_uid_service_v1`; legacy, unversioned,
upstream CUDA-event, cross-tier, or changed-backend baselines are rejected. Measure it
on the target GPU through the selected backend with `ferrl trimul-baseline --config
<run.json>`. Take at least three measurements, use the median `ns` in the config, and
keep every raw value in the report. These are ferrl end-to-end latency baselines, not
GPUMODE kernel runtimes.

## Artifact Definition

A candidate is an accepted artifact only when the final bundle contains all of:

- `submission.py`: the exact extracted `custom_kernel` source.
- `launch.json`: the exact verified immutable run manifest from the training directory.
- `candidate.json`: the exact selected `candidates.jsonl` row bytes.
- `prompt.txt`: the exact rendered TriMul model prompt used for generation,
  copied from `<run-dir>/prompt.txt` after verifying `launch.json`.
- `manifest.json`: a machine-readable manifest with the fields below.
- `verification/`: the clean re-verification logs and benchmark summaries.
- `report.md`: the human summary and operator checklist outcome.

Artifact contract v3 extends the launch- and row-derived v2 identities with the exact
verifier tier and protected preflight/run evidence used for timing and correctness:

```json
{
  "contract_version": 3,
  "task": "trimul",
  "ferrl_commit": "<full git sha>",
  "run_id": "<run directory name>",
  "launch_sha256": "<launch payload sha256>",
  "launch_file_sha256": "<sha256 of launch.json>",
  "launch_attestation_key_id": "cluster-launch-2026-01",
  "launch_attestation_algorithm": "ed25519",
  "candidate": {
    "record_sha256": "<domain-separated candidate digest>",
    "record_signature": "<Ed25519 signature from the launch key>",
    "ledger_row_sha256": "<sha256 of candidate.json>",
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
    "family": "<loader-derived family>",
    "checkpoint_policy_sha256": "<exact model/checkpoint plus loader semantics>",
    "tokenizer_sha256": "<exact tokenizer.json bytes>",
    "lora_rank": 16,
    "lora_alpha": 32.0,
    "base_dtype": "bf16",
    "base_quantization": "none"
  },
  "config": {
    "run_config_source_sha256": "<sha256 of launch input file>",
    "run_config_resolved_sha256": "<sha256 of complete canonical resolved config>",
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
    "scratch_max_bytes": 1073741824,
    "verifier_parallelism": 1,
    "verifier_max_procs": 1024,
    "verifier_cuda_device_pool": []
  },
  "eval": {
    "bundle_path": "<configured eval_dir; informational>",
    "bundle_sha256": "<ordered eval-tree sha256>",
    "bundle_file_count": 0,
    "sandbox_image_path": "<configured image path; informational>",
    "sandbox_image_sha256": "<exact image sha256>",
    "sandbox_image_len_bytes": 0,
    "task_yml_sha256": "<exact task.yml sha256>",
    "task_yml_len_bytes": 0,
    "test_cases": 0,
    "benchmark_cases": 0
  },
  "baseline": {
    "metric": "isolated-service-latency-v1",
    "isolation_tier": "dedicated_uid_service_v1",
    "isolation_evidence_sha256": "<backend preflight evidence digest>",
    "gpu": "<nvidia-smi product name>",
    "measurements_ns": [0.0, 0.0, 0.0],
    "median_ns": 0.0,
    "command": "ferrl trimul-baseline --config <run.json>"
  },
  "verification": {
    "gpu": "<nvidia-smi product name>",
    "isolation_tier": "dedicated_uid_service_v1",
    "isolation_evidence_sha256": "<backend preflight evidence digest>",
    "runs": [
      {
        "isolation_tier": "dedicated_uid_service_v1",
        "isolation_evidence_sha256": "<backend preflight evidence digest>",
        "runtime_hardening_evidence_sha256": "<protected hardening digest>",
        "runtime_hardening": [{
          "contract": "ferrl.candidate-hardening.v1",
          "no_new_privs": 1,
          "seccomp_mode": 2,
          "seccomp_tsync": true
        }],
        "timing_metric": "isolated-service-latency-v1",
        "correct": true,
        "geomean_ns": 0.0,
        "speedup": 0.0
      }
    ],
    "accepted": true
  }
}
```

The remaining reward, eval, baseline, verification, and run-health fields retain their
prior meanings. Candidate completion, coordinates, training reward, run id, commit,
model/checkpoint, tokenizer, prompt, eval path, and sandbox path are all derived from
the verified run; only audit-time measurements and inspection evidence remain command
inputs.

## Acceptance Rule

A TriMul run counts as a success only if one artifact candidate satisfies every rule:

1. The candidate is extracted from a model completion, not hand-authored after the run.
2. The immutable launch uses `external_attested_v1`; local-ephemeral discovery evidence
   cannot authorize publication.
3. The audit backend is `dedicated_uid_service_v1`, and its authenticated preflight and
   protected runtime evidence are preserved in every verification run. Same-UID evidence
   may drive training but cannot be relabeled or accepted here.
4. The candidate passes every correctness case in a clean re-verification run.
5. Re-verification matches the attested eval-bundle, sandbox-image, and `task.yml`
   content identities, uses the same GPU product name, and uses a fresh scratch directory.
6. Re-verification uses an audit `trimul.secret_seed` that was not used for training.
7. At least three clean benchmark re-runs are recorded for the candidate.
8. Every audit run and the guarded baseline use `isolated-service-latency-v1`, carry
   matching dedicated-tier preflight evidence, and the median
   candidate geometric-mean service latency is lower than the baseline median.
9. The report states speedup as `baseline.median_ns / candidate.median_geomean_ns`.

If any correctness re-run fails, or if the GPU product name does not match the baseline
pin, the candidate is rejected even if a prior training reward was high.

## Dynamic Reward-Hacking Checks

The TriMul reward already keeps candidate scratch bounded, denies network by default,
and rejects implausibly fast timings. Both tiers strip active payload capabilities,
install `NoNewPrivs`, and install a TSYNC seccomp deny policy before candidate entry.
The dedicated tier additionally places staging and launch under a distinct service UID;
the same-UID tier does not resist arbitrary malicious peers already running as the
training user. A non-dumpable protected verifier process owns input
generation, starts elapsed-time measurement before candidate input handoff, reconstructs
each exact-size result from CPU bytes into parent-only storage, and owns correctness,
statistics, and the machine grade. No CUDA tensor or allocator block crosses the process
boundary. A separate non-dumpable controller owns the trusted status/output channels;
candidate Python owns only its untrusted request/result channel and never inherits the
grade socket. Launcher/init/shell stdout remains diagnostic only.
The first discovery run still needs dynamic checks on top candidates because the
training loop is optimizing against that reward.

For every candidate included in the final report:

- Re-run from `submission.py` only; do not reuse the training scratch tree.
- Re-run with a fresh audit secret seed.
- Record whether the source tries to inspect process state, file descriptors,
  environment variables, network sockets, or paths outside the kernel inputs.
- Treat unexplained sub-launch-floor timings, inconsistent correctness, sandbox
  resource failures, or grade-channel anomalies as rejection signals.
- Include rejected high-reward candidates in the report when they explain why the
  accepted candidate was not simply the highest training reward.

These checks are deliberately operator-facing. Same-UID discovery is not proof against
an arbitrary malicious same-account peer; accepted publication requires the dedicated
audit boundary above. The checks decide whether the run found a correct candidate with
lower versioned ferrl latency or a reward artifact.

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
2. Baseline: GPU, exact timing metric, raw measurements, median service latency, and command used.
3. Training: ferrl commit, config hash, prompt copy/hash, model identity, seeds,
   budget, and run health.
4. Candidate table: source hash, training reward, source-inspection result, clean
   correctness, median service latency, same-metric ratio, and accept/reject reason.
5. Artifact bundle path and manifest hash, when accepted.
6. Operator checklist: each acceptance and reward-hacking check marked pass/fail.

Use `ferrl trimul-artifact` after training to persist the best correct-and-fast candidates with enough provenance to fill this manifest and produce the operator-facing report.
