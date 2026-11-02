# 2026-10-10 Track B G10 S3 key compatibility preparation

User authorized only a JSON-key compatibility repair and S3 source preparation,
based on technical R2 `f9d2665283c707e5a025a2c92c1a051153eaf2e1`. No production run.

S2's string-key validator rejected integer-label `provider_calls`. S3 output
shallow-normalizes each active dictionary with old `str(key)` semantics:
first normalized-key position and last value. Scientific event construction
is unchanged; no full recursive result copy. Other guards/publication/failure
classes remain AST-identical. New v3 runner differs only in metadata paths/kind.

76 focused tests pass, 17 recorded synthetic byte/hash cases match legacy.
Saved JSON IO peak legacy509.71484375/S3 255.5 MiB; artificial typed payload
legacy331/S3 79.75 MiB. Initial IO diagnostic comparator defect and its one
corrected non-science measurement are recorded, not hidden. No production
512 MiB completion claim. All scientific settings/caps/runtime remain frozen.

Original1,300 and extended1,352 protected paths pass, with immutable source
references for the intentionally repaired IO file. Pending authority, no A3,
no production marker/result. Source/tests/evidence are published for GPT review.

[S3 review/handoff](../../tracks/algorithm_codesign/g10_v3_key_compatibility_source_and_gpt_review_20261010.md)
and [type/key audit](../../tracks/algorithm_codesign/g10_v3_payload_type_and_key_audit_20261010.md)
record general-object limits and unchanged scientific semantics. Mandatory STOP;
review plus separate explicit one-shot instruction is required before execution.
