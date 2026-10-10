# G10 v3: primary-data access for GPT scientific review

This is a read-only extraction of the completed saved result, not another G10
run, scientific review, optimization, or algorithm adoption. Mandatory STOP
remains in force. Existing source, contract, authorization, marker, STOP and
result paths are unchanged. Only new files are added on the dedicated branch
`track-b-g10-v3-review-data-access-20261010`.

Original result commit: `fcd3ea6217bc00b667180cec149a70102d75f07e`.
Science S3: `b9ed01455351628c9073748f5ba5751aa794b789`.
The original [completion and GPT handoff](g10_v3_results_and_gpt_handoff_20261010.md)
continues to define the evidence boundary and frozen comparison.

## Start here

All links below are relative to this document and remain fixed when opened at a
GitHub commit. The final chat handoff supplies the full published commit SHA.

- [Access README and 17-row/event navigation](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/README.md)
- [Exact resource table, JSON](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.json)
- [Exact resource table, CSV](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.csv)
- [Row/event mapping and raw-part manifest](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/manifest.json)
- [Top-level metadata, including all 46 synthesis records and sequences](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/result_metadata.json)
- [Extraction identity verification](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/extraction_verification.json)
- [Protected-source/result audit](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/protected_provenance_verification.json)
- [All newly published file identities](../../../artifacts/track_b_g10_v3_review_access/2026-10-10/publication_manifest.json)

## Exact row values and normalization fields

The JSON resource table contains every non-event field of all 17 original rows,
in the original order. Native per-trial and two-axis T/CX/1Q, workspace,
acceptance, second moment/range, common confidence budget and sufficient shots,
tail/hard caps and T costs, prep/readout coefficients, fixed-dictionary lower,
original G9 budget and saved diagnostic errors are included. No values were
rounded, recomputed or ranked. The CSV flattens the same saved fields using
row-relative JSON-pointer column names; missing fields are blank.

In particular, `/fixed_dictionary_policy_lower/coefficient_norm_rational` is
the **saved coefficient norm**, copied without renaming it as a universal
normalizer. `/reference_m2`, `/reference_range`, `/reference_acceptance` and
the `/budget` bounds remain distinct. Their original interpretation belongs
to the frozen source and GPT scientific review, not this extraction.

Rational strings (including long numerator/denominator values) retain their
exact spelling. Original JSON decimal numbers are parsed as `Decimal` and
encoded as exact numeric values, never as binary floats. Raw parts additionally
preserve their original lexical spelling and every whitespace byte.

## Event inspection without the 66.8 MB input

The 10,936 bindings occupy 59 independently valid JSON files. Each row has its
own small metadata file and event index. The index identifies consecutive
event intervals `[start,end)`; original event `i` is `/bindings/(i-start)` in
the corresponding part. The full original binding is retained, including
event coefficient, proposal probability, weight, phase/word/provider fields,
native IR, native resource counts, strict error bound and saved diagnostic
errors. Missing provider fields on matched CTS events are not invented.

The top-level metadata retains every original top-level field except `rows`,
including the synthesis cache, inventory, interface traces, CTS certificates,
classical accounting, runtime and provenance. Inserting the indexed event
bindings into each row metadata file, then inserting these 17 rows into that
top-level object, reconstructs the complete original JSON value exactly.

## Byte-exact original access

The original JSON is also split into 136 UTF-8 text fragments. Every fragment
and every independently parseable extraction file is at most **491,520 bytes
(480 KiB)**, below 512 KiB. The complete resource JSON is 293,673 bytes; CSV is
267,192 bytes. Row metadata permits a smaller per-row reading path.

The raw fragments are not standalone JSON documents. Download the bytes in
`raw_parts_in_concatenation_order`, verify each listed size/SHA256, and concatenate
without inserting separators or normalizing line endings. The resulting bytes
must be exactly 66,842,494 bytes, with SHA256
`64a867dd6cc8f5f535880607616dea72f13af1360c7468f91eef44f79c543a2f`.
All UTF-8 cuts, offsets and this complete byte-for-byte identity passed locally.
Some raw fragments end inside the original JSON indentation. Those intentional
fragment endings are checked by byte identity rather than whitespace cleanup;
the extractor, documentation, CSV and independent JSON pass Git whitespace checks.

## Audit authority and limits

The only adopted saved scientific audits are
`saved_output_audit_corrected_v3.json` and `final_saved_output_audit_v3.json`,
at the original result commit. Their identities are recorded in the manifests.
The incorrect field in the initial `saved_output_audit_v3.json` is not used.

[The stdlib extractor/verifier](../../../scripts/tracks/algorithm_codesign/export_g10_v3_review_access.py)
uses no scientific imports or runner calls. `verify` reads the original saved
result and partitions, checks all fields/rational strings/decimal values, and
validates the flattened CSV and raw concatenation. It neither synthesizes nor
evaluates operators, budgets, proposal laws or resource-optimal alternatives.
The existing final saved-completion audit remains unchanged and passes all
57,021 checks, including 180 critical paths and 1,352 protected paths.

GitHub raw downloads of the published files are checked against their local
identities after push. This verifies delivery from GitHub; it does not emulate
every GPT connector implementation or lift any connector-specific aggregate
request limit. The small per-row/index path is available if a combined table
still exceeds a particular reader's limit.

No new science, synthesis, circuit/matrix calculation, sampling, LP, DF, GPU,
reoptimization, outcome reclassification or G11 occurs. Existing evidence is
source-bound local evidence; the export adds accessibility, not reproduction.
**Mandatory STOP; scientific interpretation returns to GPT/user.**
