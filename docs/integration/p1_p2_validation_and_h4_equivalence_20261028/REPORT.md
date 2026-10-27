# P1/P2 technical validation and limited H4 qualification (2026-10-28, LOCAL)

Technical validation is complete. DF H4 numerical equivalence is
**SOURCE_OR_INPUT_UNQUALIFIED**, with zero scientific numerical calls.
The accessible reused local archive indexes and fixed donor tree do not identify
the complete saved H4 input payload and full array/freeze identity needed for
the requested comparison. No input was regenerated, substituted or inferred.
This is not a numerical mismatch and does not invalidate historical results.

| Current execution | PASS | FAIL | SKIP |
|---|---:|---:|---:|
| P1 current pipeline | 79 | 0 | 0 |
| P1 selected legacy integer/executor slice | 13 | 0 | 0 |
| P1 added real writer/import checks | 13 | 0 | 0 |
| P2 current pipeline | 121 | 0 | 0 |
| P2 selected legacy pure/mock variants | 13 | 0 | 0 |
| P2 added real writer/import checks | 13 | 0 | 0 |

P1 total: **105 PASS / 0 FAIL / 0 SKIP**. P2 total: **147 PASS / 0 FAIL / 0 SKIP**.
These are the final task-harness runs. Earlier exploratory runs and historical
78/109/120/17 counts are not added. The P2 selected legacy tests use normal
imports and retain the selected artificial/pure/mock test bodies; they are not
a claim that all original 23 tests were run unchanged.

The missing optional import stack is resolved in a fresh task-owned Python3.11
venv. The existing eager initializer and all 413 public exports import normally
in both fixed versions; star import/reload preserve their original identities.
Real writer tests no longer skip. No alias loader, fake scientific package,
global lazy import or blanket import-error suppression was added.

Real atomic publication, JSON value semantics, collision refusal, stream/fsync/
replace failure and interruption preserve the artificial old destination and
remove temporary files. Original exceptions propagate. A failed new-directory
publication leaves an empty directory and refuses implicit reuse. This is the
existing per-file atomicity contract, not a claim of transactional directory
rollback. No pipeline/executor/kernel fix was required.

The first added assertion compared decoded JSON lists to Python tuple fields;
the test expectation was corrected to JSON serialization semantics. Those two
test-harness failures and the initial logs are retained privately; they were not
scientific mismatches or implementation fixes. Final reruns include the actual
stream-write failure case and all pass.

H4 priority qualification target was 0.90 Angstrom, linear H4/STO-3G/rank12
under the fixed geometry contract. Public metadata records its NPZ hash and
input/DF/H/state identities; that metadata does not supply the NPZ or prove its
contents. The run05 SOURCE hashes were verified for inputs/signal/circuits.
`inputs.py` is unchanged at the donor, while signal/circuits evolved later.
The metadata's input-source commit is also distinguished from run05 and donor.
No claim is made that all these versions are one scientific SOURCE.

No numerical case was selected for execution: L_D, r/K, frozen fragment/partition,
full input/state identity and the complete comparison binding remain unresolved.
The artificial P1 fixture supplies none of these scientific defaults.
All ten requested real-data comparison items are unqualified; the three excluded
resource/compile/RPE items are NOT_TESTED. Metadata/mock matches are separately
labelled and are not independent numerical equivalence evidence.

See [technical results](TECHNICAL_VALIDATION.md), [scientific scope](SCIENTIFIC_VALIDATION_SCOPE.md),
[item results](NUMERICAL_EQUIVALENCE_RESULTS.csv), and [preservation/next steps](PRESERVATION_AND_NEXT_STEPS.md).
The new harness and tests are on one independent validation branch
`codex/p1-p2-validation-h4-equivalence-20261028`, based on `2c865c66171fc9d1972b67c1257c57e5684fe26c`. Existing representative branches are unchanged;
the P2 checkout is a separate fixed read/test target, not a merged scientific source.

**Mandatory STOP after publication and public readback.**
