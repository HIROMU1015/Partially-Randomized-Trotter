# G10 S3 JSON-key compatibility source preparation / GPT review

**READY_FOR_S3_FINAL_SOURCE_REVIEW_NOT_EXECUTION.** S3 restores the old key
stringification/collision behavior while retaining bounded streaming output.
76 focused tests passed; 17 recorded synthetic byte/hash comparisons and both
saved/typed IO profile pairs match the unchanged legacy serializer.

No production runner, new scientific input, event generation, matrix, synthesis,
sampling or LP was executed. Authorization remains pending, A3 and the v3
production directory/marker are absent. Mandatory STOP follows this preparation.

## Scope, fixed identities and paths

| Role | Identity |
| --- | --- |
| Original G10 science S | `05c5ef23fce775a822ab5686f5da2f0d77675864` |
| Original technical result R | `e429c99d77b3222c5cca62750b2d111f87e4cb50` |
| Streaming source S2 | `a139b91f119d109430ae3154a045d0fdcf722233` |
| Consumed authorization A2 | `1a2cd261ebe0cf097e71026a9150756dc8c9acc3` |
| Consumed technical result R2 / S3 branch base | `f9d2665283c707e5a025a2c92c1a051153eaf2e1` |
| S3 preparation branch | `track-b-g10-v3-key-compatibility-source-preparation-20261010` |

S3's full commit is the fixed GitHub commit containing this report, supplied in
the final handoff; no self-referential source SHA is written into the commit.
The branch is independent of the original execution/source and Track A worktrees.

New preparation: `artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/`.
It contains the normalized user instruction and original attachment identity,
old IO source snapshot, pending authorization, contract/source manifest,
protection ledgers, type/AST/equivalence audits, test/runtime/IO receipts and
evidence manifest. Only necessary explicit paths are published.

## Exact repair and scientific boundaries

The only changed existing code is `src/trottertracks/algorithm_codesign/g10_io.py`:

- `validate_tree` examines original dictionary values without rejecting integer
  keys. Finite-value, supported-value, cycle and shared-subtree rules remain.
- `_compatible_tokens` shallow-normalizes each active dictionary using
  `str(key)`, preserving first insertion position and last value on collision;
  emits container tokens without a recursive result copy.
- `iter_json_bytes` uses these tokens with the existing bounded UTF-8 buffer,
  incremental writes/hash and guards. Scalar formatting uses the unchanged
  `FractionEncoder` / CPython JSON encoder.

All other IO classes/functions/imports/constants, including `OutputSession`,
failure receipts, write/promotion/STOP/token protocol, are AST-identical to S2.
There is no `default=str` fallback, scientific-event mutation, global result
copy, loss of collision fields, row spooling or restoration of large fallback
serialization.

New future runner: `scripts/tracks/algorithm_codesign/g10_degree_matched_native_v3.py`.
Its AST equals the unchanged v2 runner after normalizing only the preparation
directory, contract filename, marker-kind label and module docstring. `collect`
and all scientific statements are unchanged. The original v1→v2 normalized
science AST digest remains
`955867820a5fb605f61a2a795017580470df48adbd920ef3449680bf4560032e`.
The m5 deepcopy implementation and exact cleanup ordering remain unchanged.

The [payload type/key audit](g10_v3_payload_type_and_key_audit_20261010.md) covers
21 source areas and 68 dictionary AST nodes. Confirmed retained non-string keys
are integer labels in new native events' `provider_calls`; tuple-key algebraic
maps remain temporary or are explicitly converted before certificate retention.
The actual first failing v2 key/path was not recorded, so no live trace is claimed.

Admitted values are finite JSON scalars, Fraction, dict/list/tuple and shared
acyclic ordinary containers. Key equivalence assumes stable side-effect-free
`str(key)`. All original values are validated even if collision-overwritten:
legacy behavior on discarded invalid values is outside this admitted domain.
Side-effecting custom key/container behavior and concurrent mutation are not
guaranteed. These are general-object boundaries, not changes to scientific data.

## Verification

The focused command is the fixed isolated SP05 Python with `-B`, `PYTHONPATH=src`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, running
`tests/tracks/algorithm_codesign/test_g10_key_compatibility_v3.py -v`.

**76 tests passed:** 46 unchanged applicable S2 IO/guard/failure/launch/lifetime
tests plus 30 compatibility/pending-launch tests. The old blanket non-string-key
rejection test is explicitly superseded, not silently deleted or counted as
passing. The original test file is immutable. The new tests compare with the
actual old pure `g10_saved.serial` function.

Coverage includes integer/mixed keys, both collision orders, last value/first
position, reordered input, Fraction keys/values, tuple/list/Unicode/scalars,
shared subtree/no mutation, scalar subclass formatting, pure custom keys,
conversion errors, invalid/cyclic overwritten values, periodic guards and
synthetic normal/failure publication. A deterministic 128-case artificial
encoder matrix is within the test count; it is not scientific sampling.

`json_equivalence_v3.json` separately records bytes/SHA256/max-write-size for
17 artificial cases, including 2,500 typed rows. Each matches legacy bytes.
Existing failure/cap/exclusive-marker/partial/token safeguards pass without
production inputs. The pending v3 launch is rejected before Git/science access.

### IO-only memory checks

The existing isolated runtime and **unchanged entire cap dictionary** were used
in independent processes. The saved input is old v1 technical JSON, read-only:
66,842,493 bytes / SHA256
`b62695c19964a5a121b965c39142427bfa8048efad9bce05f3221494bb14dfe1`.
Artificial input is 50,000 typed rows with integer provider keys, collisions,
Fractions, tuples and Unicode; it is not regenerated scientific data.

| Input | Legacy peak RSS | S3 peak RSS | Matching bytes/hash |
| --- | ---: | ---: | --- |
| Saved technical JSON | 509.71484375 MiB | 255.5 MiB | 66,842,493 / above SHA256 |
| Artificial typed payload | 331 MiB | 79.75 MiB | 16,410,125 / `e1d3ce71482e0f846c9004859880557dbdc6e158b0625157eb98791c19ce9aa4` |

S3 measurements include encode/write/flush/fsync/close/disk-identity verification
under the existing guard. Temporary directories contain no production marker or
completion token. The original live G10 heap was not reconstructed. These peaks
do **not** prove that a complete S3 science run fits within 512 MiB.

Four initial IO measurements were retained. The first streaming saved-input
validator wrongly compared the entire identity dict against a two-field dict,
despite matching bytes/hash; the extra `path` field caused its reported failure.
Only that diagnostic assertion was corrected, and one additional saved-stream
measurement passed at 255.5 MiB with identical identity. Initial failure/source
and all five process receipts remain available. The encoder and caps were
unchanged by the diagnostic correction. Production runs/retries are both zero.

## Contract, protection and runtime

`contract_v3.json` changes only preparation/bookkeeping paths/schema and the IO
key-compatibility description. All scientific fields and the entire caps object
are equal to `contract_v2.json`: p=(1/5,3/10,1/2), x=5/7, same synthetic three-qubit
provider and m3/5/7 arms; within-degree same finite operator; native T/CX/1Q,
confidence/error/precision/seed/synthesis/proposal-lower policies unchanged.
No molecular geometry/basis/DF rank applies to this synthetic provider.

RSS512 MiB, AS1536 MiB, wall1200 s, CPU900 s, per-key wall30/CPU20 s, output128 MiB,
162 synthesis keys, 12,000 bindings, fixed runtime/tool identity, one shot,
retry0 and all-outcome STOP remain fixed. The future result directory is
`artifacts/track_b_g10_degree_result/2026-10-10/v3/` and is absent.

The existing terminal file protocol, including `COMPLETED.v2` and bounded receipt
filenames, is retained **inside the fresh v3 namespace**. File completion remains
necessary only: outer normal exit with COMPLETE status plus source/result audit
is also required. Exit0 alone and a surviving token alone remain insufficient.

The original 1,300-path S2 protection ledger passes unchanged. An extended
1,352-path ledger additionally protects complete R2 prefixes, S2 preparation
materials and all R2 result/auth/marker/STOP files. The one authorized mutable
IO path is separately checked against immutable Git source references and its
exact S2 snapshot; all other old source is unchanged. Declared indexes only
receive appendices retaining complete R2 prefixes. Shared scientific APIs and
Track A worktrees are untouched.

Fixed Python3.10.12, package/tree identity and executable SHA256
`b94e9b56bc1f96b18d36a0f1d14308575bb7b5960eda94ae0520f0376e95d12d`
are verified without importing scientific dependencies. The new S3 source
manifest binds current source/tests/contract/ledger/validation; authorization
is intentionally excluded so a future separate direct-child A3 can change it.

## Risks and the review requested

Local tests and IO profiles are preparation evidence, not immutable CI,
independent scientific reproduction or a new resource advantage. Full live typed
payload compatibility is supported by source inspection and artificial tests;
no new registered payload has been generated. Any uninspected dynamic field,
pathological container or production IO failure must still stop the attempt.
Active-map width/depth and a huge individual escaped string token are not
universally bounded. The accepted outer-process/token-unlink failure boundary
from the S2 review remains; arbitrary OOM/kill/disk failures can prevent receipts.

GPT should review key/collision semantics, the static type coverage, bounded
memory behavior, unchanged scientific/guard AST and pending launch bindings.
No performance winner, method adoption or change of research direction follows.

**Mandatory STOP.** If review passes, execution still requires a separate user
instruction, S3's single-parent authorization-only child A3 (only the new v3
authorization plus optional `g10_v3_execution_authorization_receipt.md`), clean
remote-matched HEAD and a fresh v3 exclusive marker. No A3 was created here.
Both consumed v1/v2 authorities/results/markers remain permanently protected.
No production run, retry, m9, new synthesis, rescue cache or G11 is authorized.
