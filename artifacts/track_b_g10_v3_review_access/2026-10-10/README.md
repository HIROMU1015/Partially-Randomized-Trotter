# G10 v3 saved primary-data access

Original result commit: `fcd3ea6217bc00b667180cec149a70102d75f07e`. This package only extracts saved data. **Mandatory STOP remains active.**

## Review entry points

- [Exact 17-row resource table (JSON)](resource_table_exact.json): all original non-event row fields, rational strings unchanged.
- [Exact resource table (CSV)](resource_table_exact.csv): same fields flattened using JSON-pointer columns; missing fields blank.
- [Manifest](manifest.json): row/event intervals, raw byte offsets, sizes and SHA256.
- [Top-level result metadata](result_metadata.json): all non-row fields, synthesis sequences/errors, inventory and provenance.
- [Extraction verification](extraction_verification.json) and [protected provenance](protected_provenance_verification.json).
- [Publication inventory](publication_manifest.json): all new files; excludes its own hash.
- [GitHub reacquisition receipt](github_reacquisition_verification.json): all 240 data-publication files at `12c320f948f5a93298cc9cc8e13139a9c481401c` were downloaded/hash-matched, including byte-exact reconstruction of all 136 raw fragments. The final documentation commit retains identical data partitions.
- [Full access report](../../../../docs/tracks/algorithm_codesign/g10_v3_primary_data_access_20261010.md).

## Row and event navigation

Original zero-based row order is retained. Event indices are zero-based within each row. Each event index maps `[start,end)` to a part; event `i` is `/bindings/(i-start)` in that independently valid JSON. Every binding field is present, including coefficient/proposal/weight, native IR/cost, phase and saved error records.

| Row | Degree | Arm | Events | Exact row metadata | Event index |
| ---: | ---: | --- | ---: | --- | --- |
| 0 | 5 | ordinary | 273 | [metadata](rows/row_00_m5_ordinary.json) | [index](events/row_00_m5_ordinary_index.json) |
| 1 | 5 | partial_return_tail | 264 | [metadata](rows/row_01_m5_partial_return_tail.json) | [index](events/row_01_m5_partial_return_tail_index.json) |
| 2 | 5 | closed_P3_tail | 258 | [metadata](rows/row_02_m5_closed_P3_tail.json) | [index](events/row_02_m5_closed_P3_tail_index.json) |
| 3 | 5 | full_return | 63 | [metadata](rows/row_03_m5_full_return.json) | [index](events/row_03_m5_full_return_index.json) |
| 4 | 5 | closed_P5_full | 63 | [metadata](rows/row_04_m5_closed_P5_full.json) | [index](events/row_04_m5_closed_P5_full_index.json) |
| 5 | 5 | matched_CTS | 24 | [metadata](rows/row_05_m5_matched_CTS.json) | [index](events/row_05_m5_matched_CTS_index.json) |
| 6 | 3 | ordinary | 30 | [metadata](rows/row_06_m3_ordinary.json) | [index](events/row_06_m3_ordinary_index.json) |
| 7 | 3 | partial_return_tail | 21 | [metadata](rows/row_07_m3_partial_return_tail.json) | [index](events/row_07_m3_partial_return_tail_index.json) |
| 8 | 3 | closed_P3_tail | 15 | [metadata](rows/row_08_m3_closed_P3_tail.json) | [index](events/row_08_m3_closed_P3_tail_index.json) |
| 9 | 3 | full_return | 15 | [metadata](rows/row_09_m3_full_return.json) | [index](events/row_09_m3_full_return_index.json) |
| 10 | 3 | matched_CTS | 21 | [metadata](rows/row_10_m3_matched_CTS.json) | [index](events/row_10_m3_matched_CTS_index.json) |
| 11 | 7 | ordinary | 2460 | [metadata](rows/row_11_m7_ordinary.json) | [index](events/row_11_m7_ordinary_index.json) |
| 12 | 7 | partial_return_tail | 2451 | [metadata](rows/row_12_m7_partial_return_tail.json) | [index](events/row_12_m7_partial_return_tail_index.json) |
| 13 | 7 | closed_P3_tail | 2445 | [metadata](rows/row_13_m7_closed_P3_tail.json) | [index](events/row_13_m7_closed_P3_tail_index.json) |
| 14 | 7 | full_return | 255 | [metadata](rows/row_14_m7_full_return.json) | [index](events/row_14_m7_full_return_index.json) |
| 15 | 7 | closed_P5_tail | 2250 | [metadata](rows/row_15_m7_closed_P5_tail.json) | [index](events/row_15_m7_closed_P5_tail_index.json) |
| 16 | 7 | matched_CTS | 28 | [metadata](rows/row_16_m7_matched_CTS.json) | [index](events/row_16_m7_matched_CTS_index.json) |

## Whole original JSON

There are 136 raw UTF-8 fragments in `raw_parts/`, each no larger than 491,520 bytes (480 KiB). They are byte fragments, **not** individually valid JSON. All event parts and metadata are valid independent JSON. All publication files are below 512 KiB.

Raw fragment boundaries can fall inside original indentation. That whitespace is intentional and retained for byte-exact reconstruction; raw fragments are checked by size/UTF-8/SHA256 rather than whitespace normalization. Other new files pass Git whitespace checks.

To restore the original lexical bytes, download each manifest-listed raw file as bytes and concatenate in `raw_parts_in_concatenation_order`. Verify each SHA256 before concatenation. Add no LF or other separator. Do not copy rendered GitHub text, change encoding or normalize line endings.

Expected reconstruction: **66,842,494 bytes**, SHA256 `64a867dd6cc8f5f535880607616dea72f13af1360c7468f91eef44f79c543a2f`. [First fragment](raw_parts/part_0000.txt); [last fragment](raw_parts/part_0135.txt).

To recover the complete JSON value without byte reconstruction, read `result_metadata.json`; insert the event-indexed bindings as `events` into each `rows/*.json`; insert the 17 original-order rows as `rows`. This exact semantic reconstruction passed for every original field. Decimal JSON numbers retain their exact numeric values via Decimal; rational strings retain their exact spelling.

## Authority and scope

Only corrected and final saved audits from the original result commit are authoritative: `saved_output_audit_corrected_v3.json` and `final_saved_output_audit_v3.json`. Their identities appear in the manifest. The incorrect initial audit field is never used.

Normalization-related values remain distinct: the saved coefficient norm is `/fixed_dictionary_policy_lower/coefficient_norm_rational`, with separate `/reference_m2`, `/reference_range`, `/reference_acceptance` and budget bounds. No universal normalizer, new optimum or resource ranking is inferred.

This package is saved-data accessibility only. No synthesis, operator/matrix/circuit calculation, sampling, LP, DF, GPU, scientific rerun, budget reevaluation or G11. Original evidence remains source-bound local evidence; scientific interpretation belongs to GPT/user.

For a full local clone, the stdlib verifier is `python3 -B scripts/tracks/algorithm_codesign/export_g10_v3_review_access.py verify`. It checks saved identities and extraction equality only.
