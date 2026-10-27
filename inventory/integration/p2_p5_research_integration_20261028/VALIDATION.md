# P5 metadata validation (2026-10-28)

All checks below are technical navigation/preservation checks, not scientific validation.

16 PASS / 0 FAIL / 0 SKIP.

- PASS: CEM versions remain 43 and 50 references (93 total).
- PASS: Composite reference keys include version/row/study/SOURCE/result/cohort and are unique.
- PASS: All selected evidence fields literally match their original versioned row.
- PASS: Explicit cohorts preserved; missing/non-JSON metadata is left unresolved.
- PASS: Both original CEM files remain unchanged.
- PASS: Six independent development scopes, with no new scientific results.
- PASS: No evidence row is relabelled with a new integration code SOURCE.
- PASS: Development code/CLI/test paths exist at the pinned commits.
- PASS: 71 public artifact locators have fixed SHA/URL and existing Git paths; bodies not re-executed.
- PASS: Seven B2-D stages keep separate SOURCE/result/STOP entries.
- PASS: Four selected M1 descriptor blobs remain byte-identical to donors.
- PASS: Original checkout/index and fixed remote branches preserved; concurrent Research C delta explicitly qualified.
- PASS: P2/P3/P4 pushed SHA and major GitHub file readback match.
- PASS: Public additions contain no internal absolute paths, host/IP, credential or private raw content.
- PASS: Only new files in the requested integration-index directory are pending.
- PASS: B3/server A/B/runtime untouched; no scientific work or new execution controls.

Of the 93 versioned references, 81 lack a separately indexed cohort string; 6 have a non-JSON saved_fields_json cell. Their full original row/context remains the authority, and no cohort or value was inferred.

P2: 120 PASS / 0 FAIL / 1 SKIP; P3: 32 PASS / 0 FAIL / 0 SKIP; P4: 30 PASS / 0 FAIL / 0 SKIP. These were separate current runs; no old109/17 counts were combined. The P2 skip is the optional real writer (matplotlib unavailable).

No raw-wide hash, scientific result analysis, new mathematical comparison, operator evaluation, sampling, compilation, N/G/SE computation or scientific-status update was performed. Artifact path existence is not proof of scientific correctness, source custody or applicability.
