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
--out <artifact-dir>
--audit-cuda-visible-device <one-device-token> --run-health <summary>
--source-inspection clean --source-inspection-notes <notes>
[--audit-verifier-executor-socket <dedicated-service.sock>]`. Artifact extraction
requires `launch.json` to have the exact production-canonical encoding, then validates
its authentication contract and any required external attestation, every candidate row,
the exact selected row, the frozen prompt, and the
live verifier assets before GPU detection or audit verification. It does not accept operator-authored
completion, coordinate, reward, run, commit, model, tokenizer, eval, or sandbox
provenance fields. Both launch-authentication modes may hand a launch-bound native row
to the artifact command. The discovery mode and verifier tier remain recorded without
upgrade. Artifact audit defaults to the no-administrator `same_uid_apptainer_v1` path;
the dedicated socket is an explicit optional higher-isolation selection with no fallback.

## Launch Authentication

`local_ephemeral_v1` is the default discovery mode. It requires no system trust policy
or attestor socket and permits launch-bound, signed candidate ledgers. It protects the
run against accidental drift, cross-run substitution, and later loss of the ephemeral
candidate signing key. It does not authenticate the launch against a process already
controlling the same host account. Publication is therefore operator-attested at that
same-UID boundary rather than externally authenticated.

The immutable local launch and its signed candidate row may nevertheless be inputs to
`trimul-artifact`: the command records their limited discovery boundary without
relabeling it, then produces fresh paired audit evidence under the explicitly selected
audit tier. The default audit remains operator-trusted; the optional dedicated service
strengthens execution isolation but does not by itself enforce whole-audit once-only
selection.

`external_attested_v1` is the optional higher-assurance discovery-launch mode. Candidate-producing
training in this mode and artifact extraction have no
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

For `external_attested_v1`, `trimul-artifact` loads the protected trust policy
independently. Replacing the entire run directory, recomputing every unkeyed hash, and
signing rows under an operator-generated key therefore remains rejected: the replacement
launch lacks a signature under a key trusted outside that directory. The local mode makes
no such same-UID authenticity claim. Its artifact publication is explicitly
operator-attested unless an external authority is used, while the manifest preserves the
discovery boundary and fresh audit evidence without relabeling either.

For rollout-only diagnostics from an external inference runtime, use `ferrl
trimul-score --config <run.json> --prompt-copy <prompt.txt> --completion <raw.txt>
--out <scores.jsonl> --score-secret-seed <seed>` or pass `--completions-jsonl
<jsonl>`. Both seeds must be in `0..=2^32-1`, and the scoring seed must differ
from the training `trimul.secret_seed`.
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
acceptance rule below stays strict: secret-seed re-verification of the launch-bound
cases, fixed same-device paired benchmarking, and nine of eleven strict speedups over
the freshly measured reference. This artifact audit remains separate from
training-time held-out evaluation. When `data.eval_n > 0`, `ferrl train` requires
`trimul.held_out_secret_seed` to be a different in-range case-generation seed,
constructs a separate verifier-backed reward for those cases, and publishes the
launch-bound result as `eval-report.json`. Data-parallel reports bind the ordered
launch group plus the exact rank-zero publishing launch, and coordinate result
consensus and immutable publication failure in lockstep. The artifact audit still derives its own
seed and must not be relabeled as that held-out run.

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
| launch authentication | Exact `launch_authentication` mode. `local_ephemeral_v1` needs no external service and provides discovery provenance only; its launch-bound row may be handed to the artifact command, but is not relabeled as audit evidence. `external_attested_v1` additionally records a trusted external `key_id`, Ed25519 algorithm, and detached launch signature verified through `/etc/ferrl/launch-trust.json`; the private key and signing authority stay outside ferrl and the operator-writable run directory. |
| run config | Original file SHA-256 plus the complete canonical resolved config stored in `launch.json`. |
| prompt | The exact rendered model prompt bytes, frozen as `<run-dir>/prompt.txt` and sealed in `launch.json`; `prompt.sha256` remains a compatibility sidecar. Do not rely on a mutable local `trimul.prompt_path` for provenance. |
| submission extraction | `trimul.submission_extract_mode` (`final_fence` or `thinking_after_think`); this controls parsing only and must not construct prompt text. |
| reward profile | `trimul.reward`; defaults to `trimul_shaped_v1`, with custom ladder-preserving values allowed. |
| run-health policy | `run_health`; post-run warn/fail policy, including the original top-level config passed to `ferrl runreport --config`. |
| model | Loader-derived family, exact model/checkpoint policy SHA-256, exact tokenizer-file SHA-256, resolved EOS, LoRA rank/alpha, base dtype, and rollout seed, all sealed at launch. |
| verifier assurance | Exact discovery and audit tiers, backend preflight evidence and digest, tier-specific timing metric, and protected runtime-control evidence. `same_uid_apptainer_v1` is the default no-admin training and artifact-audit tier: it uses a private user-owned mode-`0700` work root and a canonical root-owned Apptainer executable, but explicitly does not resist arbitrary hostile peers under the same host UID. `dedicated_uid_service_v1` is an optional audit tier using a protected Unix socket and a distinct non-root service UID that authenticates `SO_PEERCRED`. Neither tier falls back to the other, and dedicated execution isolation alone is not a whole-audit once-only authority. |
| TriMul eval bundle | SHA-256 over every ordered relative regular-file name and byte under `eval_dir`, plus the exact file count. Every captured file is held in a Linux kernel-sealed anonymous descriptor; after all seals are installed, the ordered descriptor contents are rehashed and required to equal the launch identity. The selected backend copies and rehashes each asset into a unique private request directory and supplies only the resulting read-only paths to Apptainer. The dedicated backend receives descriptors over `SCM_RIGHTS`; the same-UID backend stages them in process. The configured source path is informational only. |
| sandbox image | SHA-256 and byte length of the exact Apptainer image streamed into a kernel-sealed anonymous descriptor. After sealing, the descriptor is rehashed and required to equal the captured identity. The selected backend copies and rehashes it in private storage before launch. The configured source path is informational only. |
| cases | Attested `task.yml` SHA-256 and byte length plus the loaded counts for `tests` and `benchmarks`. |
| seeds | `data.seed`, `policy.seed`, trainer seed-bearing knobs, and the training `trimul.secret_seed`. |
| scratch cap | `trimul.scratch_max_bytes`; `0` means the ferrl default, currently 1 GiB. |
| verifier process cap | `trimul.verifier_max_procs`; `0` means the TriMul default, currently `1024`. This is a per-UID `RLIMIT_NPROC` cap, not a per-container task count. |
| candidate ledger | `trainer.candidate_log_top_k`; use a positive value for discovery runs, and use at least `group_size` for `run_health.correctness_collapse`, `run_health.source_dominance`, and low- or zero-reward tail diagnosis. Every persisted row must carry the launch digest, its exact `record_sha256`, and a valid Ed25519 `record_signature` under the public key frozen in `launch.json`; retain any `reward_diagnostic` values in the report. |
| trainer scalar controls | Exact `trainer.lr_schedule` and `trainer.beta_schedule` when present, otherwise the scalar `lr`, `warmup_steps`, and `beta` values. Schedules are deterministic step-index functions and must be copied with the final run config. |
| hardware | Discovery-baseline GPU product name and visible CUDA count; for artifact audit, one explicit CUDA-visible token plus the protected CUDA Driver name, logical ordinal, PCI bus id, and UUID. |
| budget | Trainer `steps`, `group_size`, wall-clock allocation, and the stop condition chosen below. |

A discovery run must not start without a guarded same-GPU baseline in
`trimul.baseline`. Its `metric`, `isolation_tier`, and
`isolation_evidence_sha256` must exactly match the active preflight. The metric is
`same-uid-apptainer-latency-v1` for `same_uid_apptainer_v1` and
`isolated-service-latency-v1` for `dedicated_uid_service_v1`; legacy, unversioned,
upstream CUDA-event, cross-tier, or changed-backend baselines are rejected. Measure it
on the target GPU through the selected backend with `ferrl trimul-baseline --config
<run.json>`. Take at least three measurements, use the median `ns` in the config, and
keep every raw value in the discovery run notes/report. Artifact acceptance remeasures
Ferrl's bundled reference inside every paired audit block instead of importing this pin.
These are ferrl end-to-end latency baselines, not
GPUMODE kernel runtimes.

## Artifact Definition

A candidate is an accepted artifact only when the final bundle contains all of:

- `submission.py`: the exact extracted `custom_kernel` source.
- `launch.json`: the exact verified immutable run manifest from the training directory.
- `candidate.json`: the exact selected `candidates.jsonl` row bytes.
- `prompt.txt`: the exact rendered TriMul model prompt used for generation,
  copied from `<run-dir>/prompt.txt` after verifying `launch.json`.
- `manifest.json`: a machine-readable manifest with the fields below.
- `verification/`: 22 raw JSON evidence records, one fresh reference and candidate
  execution for each of eleven paired blocks.
- `report.md`: the human summary and operator checklist outcome.

Artifact contract v4 replaces operator-supplied baseline samples and median-only
acceptance with a fixed independent paired audit. Discovery provenance stays distinct from
audit provenance, and every raw protected execution is retained under verification/ and
bound by SHA-256 from the manifest:

~~~json
{
  "contract_version": 4,
  "task": "trimul",
  "ferrl_commit": "<full git sha>",
  "run_id": "<run directory name>",
  "launch_sha256": "<launch payload sha256>",
  "launch_file_sha256": "<sha256 of launch.json>",
  "launch_authentication": "local_ephemeral_v1",
  "launch_attestation_key_id": null,
  "launch_attestation_algorithm": null,
  "discovery_verifier": {
    "isolation_tier": "same_uid_apptainer_v1",
    "isolation_evidence_sha256": "<discovery preflight digest>",
    "timing_metric": "same-uid-apptainer-latency-v1",
    "runtime_preflight_evidence_sha256": "<discovery runtime-preflight digest>"
  },
  "candidate": {
    "record_sha256": "<domain-separated candidate digest>",
    "record_signature": "<launch-key signature>",
    "ledger_row_sha256": "<sha256 of candidate.json>",
    "step": 0,
    "prompt_index": 0,
    "group_index": 0,
    "rank": 0,
    "world_size": 1,
    "training_reward": 0.0,
    "completion_sha256": "<sha256 of completion.txt>",
    "source_sha256": "<sha256 of submission.py>",
    "source_inspection": {"result": "clean", "notes": "<inspection notes>"}
  },
  "model": {"family": "<family>", "checkpoint_policy_sha256": "<sha256>", "tokenizer_sha256": "<sha256>", "lora_rank": 16, "lora_alpha": 32.0, "base_dtype": "bf16", "base_quantization": "none"},
  "config": {"run_config_source_sha256": "<sha256>", "run_config_resolved_sha256": "<sha256>", "prompt_sha256": "<sha256>", "prompt_file": "prompt.txt", "training_secret_seed": 0, "audit_secret_seed": 1},
  "eval": {"bundle_sha256": "<sha256>", "sandbox_image_sha256": "<sha256>", "task_yml_sha256": "<sha256>", "test_cases": 18, "benchmark_cases": 5},
  "audit": {
    "contract": "ferrl.trimul-artifact-audit.v2",
    "audit_contract_sha256": "<seed/output/device-independent candidate contract sha256>",
    "audit_id": "<domain-separated sha256>",
    "audit_secret_seed": 1,
    "audit_seed_derivation": "sha256_contract_prefix_u32_be_v1",
    "attempt_selection_assurance": "operator_attested_v1",
    "durable_once_only": false,
    "artifact_wide_false_positive_guarantee": false,
    "requested_cuda_visible_device": "0",
    "isolation_tier": "same_uid_apptainer_v1",
    "isolation": "<complete authenticated selected-tier isolation evidence>",
    "isolation_evidence_sha256": "<selected-tier preflight digest>",
    "runtime_preflight": "<complete authenticated runtime-control preflight>",
    "runtime_preflight_evidence_sha256": "<selected-tier runtime-preflight digest>",
    "timing_metric": "same-uid-apptainer-latency-v1",
    "executing_device": {"contract": "ferrl.executing-device.v1", "cuda_logical_ordinal": 0, "name": "<driver product name>", "pci_bus_id": "0000:00:00.0", "uuid": "<32 lowercase hex>"},
    "blocks": [{
      "index": 0,
      "first": "reference",
      "reference": {"role": "reference", "evidence_file": "verification/block-000-reference.json", "evidence_sha256": "<sha256>", "isolation": "<same complete selected-tier evidence>", "runtime_hardening": "<both protected phase records>", "verification": {"correct": true, "benchmark_means_ns": [0.0], "geomean_ns": 0.0, "speedup": null}, "exact": {"sandbox_status": {"Exited": 0}, "test_exit": 0, "benchmark_exit": 0, "executing_device": "<same structured identity>", "test_cases": "<complete ordered cases>", "benchmark_cases": "<complete ordered statistics>", "protected_output_sha256": "<sha256>", "sandbox_diagnostics_sha256": "<sha256>"}},
      "candidate": {"role": "candidate", "evidence_file": "verification/block-000-candidate.json", "evidence_sha256": "<sha256>", "verification": "<summary>", "exact": "<complete exact evidence>"},
      "paired_speedup": 1.03,
      "material_win": true
    }],
    "decision": {"method": "paired_material_wins_v1", "paired_blocks": 11, "material_speedup": 1.02, "threshold_comparison": "strict_greater_than", "required_material_wins": 9, "observed_material_wins": 9, "accepted": true}
  },
  "accepted": true
}
~~~

There are exactly eleven ordered blocks and exactly two fresh executions per block. Each
raw execution file retains the full protected grade, sandbox diagnostics, parsed summary,
isolation and runtime-hardening records, exact indexed correctness cases, exact benchmark
statistics, zero exit markers, and the executing CUDA device identity. Candidate completion,
coordinates, reward, run id, commit, model/checkpoint, tokenizer, prompt, and verifier assets
remain derived from the authenticated run. Audit measurements use the selected explicit tier;
the no-administrator same-UID tier is the default and the dedicated service is optional.
Before any runtime probe or measurement, the output directory is created exclusively
with a durable per-output owner/attempt record and a hidden sibling stage. The case seed
and alternating order are derived deterministically from the immutable audit contract,
independently of the output path; the CLI accepts no audit-seed override. The client
writes each raw execution to the retained stage as soon as it is validated. Publication
uses ownership-checked no-replace hard links and makes `manifest.json` visible only after
every other artifact file is durable. A failed or partial destination remains claimed and
cannot be mistaken for a published artifact because it has no manifest.

This exclusive output transaction is not a global audit ledger. Under
`operator_attested_v1`, the operator or another process controlling the same host UID can
suppress that destination or run the whole audit into another output directory. The
artifact therefore makes no durable once-only or artifact-wide false-positive guarantee.
The manifest also retains the complete selected-tier isolation and runtime-preflight objects,
not only their digests, so an offline reader can recompute every preflight binding. Host
source paths and the operator's absolute output path are intentionally excluded from the
portable manifest/report.

## Acceptance Rule

A TriMul run counts as a success only if one artifact candidate satisfies every rule:

1. The candidate is extracted from a model completion, not hand-authored after the run.
2. The exact launch-authentication mode and discovery verifier are retained without
   relabeling. If the launch is externally attested, its configured trust policy verifies it.
3. The audit backend is `same_uid_apptainer_v1` by default, or
   `dedicated_uid_service_v1` when the optional socket is supplied. Its authenticated
   preflight and protected runtime evidence are preserved in every verification run;
   discovery evidence is retained separately and never relabeled as audit evidence.
4. Every one of the 22 fresh executions (reference and candidate in each of eleven blocks)
   exits successfully and reports zero test and benchmark exits.
5. Re-verification matches the attested eval-bundle, sandbox-image, and `task.yml`
   content identities, preserves the complete exactly indexed correctness and benchmark
   cases plus raw protected output, and uses a fresh scratch directory.
6. The audit `trimul.secret_seed` is deterministically derived from the immutable
   candidate/audit contract as an unsigned 32-bit value, is independent of output path,
   differs from the training seed, and has no operator-selectable CLI input.
7. The command exposes one CUDA device token; the trusted CUDA Driver API records one
   canonical logical ordinal, PCI bus id, UUID, and product name, identical in both phases
   and all 22 executions.
8. Block order is precommitted by the audit id and alternates which role runs first.
   An accepted bundle contains exactly the 22 scheduled measurements; skipped,
   reordered, repeated, or extra executions inside that bundle are invalid.
9. Each paired speedup is `reference.geomean_ns / candidate.geomean_ns`. A material win is
   strictly greater than `1.02`; equality is a loss.
10. Acceptance requires at least nine material wins out of exactly eleven pairs. This is
    the predeclared empirical material-win rule. The `1.02` threshold is a materiality
    margin, not a confidence bound on speedup magnitude.
11. Source inspection is `clean`; otherwise the mechanical audit decision cannot become
    an accepted artifact.

If any execution is incomplete, changes device identity, omits raw/exact evidence, or
fails correctness, the audit aborts and the candidate is rejected even if its training
reward was high. There are no operator-supplied artifact baseline measurements and no
post-hoc repeat count inside one invocation. The output path is claimed before the audit,
partial evidence remains in its owned stage on failure, concurrent writers cannot share a
destination, and only the manifest-last no-replace commit is a published bundle.

The honest assurance boundary is explicit: same-UID publication trusts the operator not
to select among whole-audit attempts. Even with the optional dedicated execution backend,
this command does not enforce experiment-wide once-only selection and does not advertise
the per-attempt binomial tail `67/2048` as an artifact-wide guarantee. A stronger claim
requires a separately approved external runner or non-resettable attempt authority that
retains every attempt and failure end to end.

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

For every candidate considered before the artifact audit:

- Re-run from `submission.py` only; do not reuse the training scratch tree.
- Use discovery/diagnostic scoring for exploratory checks; the artifact command derives
  its own seed and does not accept an operator-selected audit seed.
- Record whether the source tries to inspect process state, file descriptors,
  environment variables, network sockets, or paths outside the kernel inputs.
- Treat unexplained sub-launch-floor timings, inconsistent correctness, sandbox
  resource failures, or grade-channel anomalies as rejection signals.
- Include rejected high-reward candidates in the report when they explain why the
  accepted candidate was not simply the highest training reward.

These checks are deliberately operator-facing. Same-UID discovery is not proof against
an arbitrary malicious same-account peer; same-UID publication retains that exact
operator-trusted boundary. Use the optional dedicated backend or an externally isolated
runner when a stronger candidate-execution boundary is required. The checks decide
whether the run found a correct candidate with lower versioned ferrl latency or a reward
artifact.

## Stopping Rule

Choose and record one stop condition before launch:

- `budget_exhausted`: stop after the configured trainer step budget.
- `target_found`: stop early only after a candidate passes the acceptance rule above.
- `operator_abort`: stop because the run became invalid or uneconomical; the report must
  say why.

If no candidate passes the acceptance rule, the run result is `no_win`. A `no_win`
report is still a valid outcome when it includes the locked config, discovery baseline, training
health summary, top rejected candidates, and the reason each top candidate failed
verification.

## Report Shape

The final report must fit this outline:

1. Verdict: `accepted_artifact`, `no_win`, or `invalid_run`.
2. Discovery provenance: ferrl commit, launch/config/candidate hashes, exact launch
   authentication, original verifier tier, source inspection, model, seeds, and run health.
3. Independent paired audit: audit id, dedicated preflight identity, executing-device
   identity, all eleven paired speedups, the 2% material threshold, win count,
   operator-trusted attempt boundary, and accept/reject reason.
4. Artifact bundle path and manifest hash.
5. Operator checklist: each provenance, exact-evidence, fixed-sample, decision, and
   reward-hacking check marked pass/fail.

Use `ferrl trimul-artifact` after training to persist the best correct-and-fast candidates with enough provenance to fill this manifest and produce the operator-facing report.
