# Parallel integration acceptance

Future integrated schema: PENDING_SERVER_SCHEMA.

|ID|Judgment|Evidence|Unresolved|
|---|---|---|---|
|STATE_HASH_CLAIM|PENDING_EVIDENCE|INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES|None|
|P01|PASS|INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES|None|
|P03|PASS|INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES|None|
|P04|PASS|INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES|None|
|P08|PENDING_EVIDENCE|INSUFFICIENT_INDEPENDENT_EVIDENCE|Need ordered parent gate trace; statefile is not transition history.|
|P02|PASS|INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES|None|
|P05|PASS|INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES|Physical child exit is only a saved producer claim.|
|P06|PASS|INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES|None|
|P07|PENDING_EVIDENCE|COMPARED_SUMMARY_CLAIMS|Flags/state alone do not prove atomic admission. Baseline runtime artificial regressions supplied separately.|
|P10|PASS|COMPARED_SUMMARY_CLAIMS|all_workers_joined is a receipt declaration; control WORKER_END precedes join in producer.|
|P11|PASS|COMPARED_SUMMARY_CLAIMS|Runtime claim-reuse refusal is separate artificial evidence.|
|P09|PASS|COMPARED_SUMMARY_CLAIMS|Measurement boundary comes from fixed producer SOURCE; no ranking/estimand adoption.|
