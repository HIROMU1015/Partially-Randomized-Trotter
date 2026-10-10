# G10 v2 one-shot technical result and GPT handoff

G10 v2 ended as **G10_TECHNICAL_INCONCLUSIVE**. The fixed S2 runner was invoked
once from its authorization-only direct child A2. retry=0; mandatory STOP is in
effect. No further science, synthesis, matrix work, repair or next stage occurred.

The failure was `TypeError: G10 JSON keys must be strings` at
`before_result_stream` / `validate_encode_write`. It occurred before any payload
bytes were written. Peak RSS was 237.33984375 MiB, below the unchanged 512 MiB cap.
This attempt did not establish completion of full result serialization within
the cap and supplies no usable scientific comparison.

## Fixed identities and authorization

| Role | Identity |
| --- | --- |
| Frozen S2 | `a139b91f119d109430ae3154a045d0fdcf722233` |
| Execution HEAD / authorization A2 | `1a2cd261ebe0cf097e71026a9150756dc8c9acc3` |
| Branch | `track-b-g10-v2-one-shot-execution-20261010` |
| Source-review reference only | `0e6a6f115a16691813fefbaa95d68ee9cb70c6cf` |
| Contract SHA256 | `5e464aa91ff46571e28152fc10dc39377822fba7d15baf81672af72c8ee5612e` |
| Authorization SHA256 | `3d9e47348c5fa40d2a3dff26802ac74ef10fa6f4cd0047f2606e34e5d81b70f7` |
| Source manifest SHA256 | `331ecda60e0152172b766d9ad2b4c0dd8883eca9d0a3cf8735416d5297261884` |
| One-shot marker SHA256 | `64dc3cb83a47b9c2ca5bfad14aaee84f88c4bbcc5e65cdeb35e70f39fc17b5e9` |
| Runtime executable SHA256 | `b94e9b56bc1f96b18d36a0f1d14308575bb7b5960eda94ae0520f0376e95d12d` |
| Tool identity SHA256 | `803d902a18d925a9565749fbc64422dca4979bfa6c0bd8b438b28de18d3aa984` |

The contract and authorization are under
`artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/`.
[The receipt](g10_v2_execution_authorization_receipt.md) records the user's exact
instruction. A2 has exactly one parent, S2, and changes only the new authorization
JSON and receipt. The clean launch HEAD and remote SHA both matched A2; the v2
directory and marker were absent. The reference-review commit was not used as
the execution source or parent.

The unchanged runtime was the existing isolated SP05 environment, Python
3.10.12, with package/tree identity verification and `-B`, `PYTHONPATH=src`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. Runtime receipt
`synthesizer_calls=0` refers to metadata verification; actual acquisition reported
27 new synthesis calls and 19 saved G9 reuses. No sequences from the failed old
G10 result were imported as rescue cache.

## What ran and what was actually saved

The scope remained the registered synthetic three-qubit provider with
`p=(1/5,3/10,1/2)`, `x=5/7`, degrees 3/5/7, and the same finite operator
`P_m(-ix sum p_i Q_i)` within each degree. Provider:
`Q0=Z0`, `V1=R_XX01(pi/4)`,
`V2=R_XX12(pi/4) R_ZZ01(pi/4)`, `Qi=Vi†ZiVi` (right factor first).
There is no molecular geometry, basis or DF rank in this synthetic domain.
No across-degree equal-exponential-accuracy claim is made.

The telemetry records eleven new m3/m7 row-accounting stages, one saved m5
rebudget stage, and return from `collect` before output. The bounded failure
receipt reports 17 retained rows, 27 new synthesis calls and 19 reused keys.
These are diagnostic counters, not saved scientific rows.

| Raw runner output | Bytes | Meaning |
| --- | ---: | --- |
| `one_shot_consumed.json` | 1,522 | Exclusive marker; permanently consumed |
| `io_stages_v2.jsonl` | 10,218 | 41 technical stage snapshots |
| `failure_receipt_v2.json` | 2,122 | Reason, exception frames, resource/counters and provenance |
| `STOP.json` | 211 | Technical STOP, science not committed |
| `result_v1.json.partial` | 0 | Empty partial; audit confirms empty-file SHA256 |

Total raw runner output is 14,073 bytes. `result_v1.json` and `COMPLETED.v2`
are absent. There is no persisted synthesis cache, sequence inventory, native
IR, row-level error/accounting data or resource map from this run. They were
released by the frozen failure protocol. Reconstructing them would require
unauthorized new computation and has not been attempted. The old 17-row failed
result is unchanged and cannot substitute for the missing v2 payload.

## Outer-process and completion audit

The runner process exited normally with code 0, no signal and empty stderr.
Its sole stdout status is **G10_TECHNICAL_INCONCLUSIVE**. This is a handled failure;
exit code 0 is not evidence of scientific completion. The outer process and
file-completion conditions agree that this attempt is unusable scientifically.

| Measurement | Saved value |
| --- | ---: |
| Outer wall, including launch/runtime and shutdown | 15.977114358916879 s |
| Outer CPU (wait4 user + system) | 13.706302 s |
| Outer peak RSS | 243,036 KiB = 237.33984375 MiB |
| Failure-receipt guard wall | 13.455659988801926 s |
| Failure-receipt guard CPU | 13.454465 s |
| Failure-receipt guard peak RSS | 243,036 KiB |
| Raw output bytes | 14,073 |
| Runner invocations / retries | 1 / 0 |
| Scientifically usable saved rows | 0 |

The outer `wait4` peak agrees with the guard and maximum saved telemetry peak.
Observed wall/CPU/RSS/output totals are below their fixed limits. The run failed
before payload encoding, so this is not a successful production memory benchmark.
Per-key resource and strict-error records did not survive; their independent
post-run verification is unavailable. The technical failure is not interpreted
as favorable or unfavorable evidence for any algorithm.

The [stdlib saved-failure audit](../../../scripts/tracks/algorithm_codesign/audit_g10_v2_saved_failure.py)
checks the raw files, outer logs/receipt, marker/auth/contract/runtime bindings,
143 S2 paths and all 1,300 protected old paths. It reruns no scientific guards,
samplers, matrices, synthesis, budgets or proposal lower bounds. The initial
audit passed 1,654 checks; postpublication prefix checking is recorded separately.
This is local source-bound technical evidence, not immutable CI or independent
scientific replication. Post-STOP science calls are zero. Physical quantum
shots/trajectories, molecular/DF/NPZ/GPU/LP work remain outside the executed scope;
the fixed 576 interface trials are deterministic interface diagnostics.

## Static compatibility diagnosis and remaining uncertainty

Confirmed source facts:

1. `g7_generator._event` uses `Counter(reduced)`, increments `calls[child]`,
   and attaches `provider_calls=dict(calls)`. Label indices are integers.
2. The old `g10_saved.serial` converted every dictionary key with `str(k)`.
3. S2's `g10_io.validate_tree` instead rejects any non-string dictionary key.
4. New m3/m7 events are attached directly to output rows. Decoded m5 saved rows
   already have JSON string keys, so the saved-JSON IO benchmark could not expose
   this live typed-event mismatch.

This establishes a compatibility defect between the producer and S2's declared
string-key output boundary. `rows[new_degree].events[*].event.provider_calls`
is a high-confidence cause of the rejection. The bounded traceback stores only
function/line frames, not the failing key or traversal path, so the exact first
offending path was not directly recorded. No live scientific event was regenerated
to locate it. The 47 synthetic tests and saved-JSON memory profiles did not
establish compatibility with integer-key dictionaries from the live producer.

No fix has been applied. A future narrowly authorized preparation could examine
bounded, nonmutating key normalization that reproduces old `str(k)` semantics,
including collisions/order, using synthetic integer-key fixtures. It would need
a new source review, unchanged-science/schema evidence and memory/IO validation.
The alternative of changing scientific event construction would require a wider
semantic review. No RSS cap increase is warranted by this failure. These are
technical proposals for GPT review, not adopted changes or execution authority.

## Protected history, publication and STOP

Old G10 S `05c5ef23fce775a822ab5686f5da2f0d77675864`,
A `f5cd0755424d1b11e2249cc115518d84fb8bb8d3`,
R `e429c99d77b3222c5cca62750b2d111f87e4cb50`, their marker/result/STOP,
G9 evidence and the frozen S2 source are unchanged. Only declared index paths
receive post-run appendices preserving their entire S2 prefix. Authorization,
contract and the consumed v2 marker are not edited after execution.

The new result directory contains raw failure evidence, outer-process capture,
preflight, source-prefix identities, saved/provenance audits and an evidence
manifest. These materials and this report are published on the dedicated
execution branch. Raw runner bytes are immutable; post-STOP audit/publication
files are listed separately and are not retroactively counted as runner output.

**Mandatory STOP.** No rerun, source repair, synthesis rescue, m9, cap/precision
change or G11 follows this report. Any authorized repair would require a new
source commit and final source review; any subsequent production attempt would
require its own separate authorization-only direct child and fresh marker/output
namespace while retaining both consumed attempts. The necessity and scope of
that work, and Track B's scientific direction, return to GPT.

GPT should review the technical compatibility defect and information value of
another bounded preparation before authorizing work. Neither scientific success
nor algorithm adoption is established here.
