# H4 B0+I pilot6 — completed result entry

**Six real compiles completed, then mandatory STOP.**
The [Japanese result report](../../../../docs/research/track_a_h4_b0i_pilot6_results_20261014.md)
contains the scope, measured circuit counts, saved-shot comparisons and limits.

- Fixed execution source: `8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b`.
- Launch grant commit: `449614525d5de74daccf9f193e3199eb931fb388`.
- [Measured six wrapper metrics and fingerprints](compiled_wrapper_metrics.json), [CSV](compiled_wrapper_metrics.csv).
- [Three measured configurations at epsilon .05](selected_epsilon_0p05.csv).
- [The same fixed configurations at 302 precisions each](measured_fixed3_precision_906.csv).
- [Result summary and cumulative budget](result_summary.json).
- [Native exit/cleanup, sampled memory, monitoring and original evidence hashes](native_resource_receipt.json).
- [Public input provenance](result_provenance.json) and [local verification receipt](result_verification.json).
- [Scientific review](scientific_review.md), [user approval evidence](user_approval_evidence.json), [plan](plan.json), [authorization](authorization.json), [final binding](binding.json).
- [Next-scope proposal, not authorized](NEXT_SCOPE_PROPOSAL.md).

The old launch README/manifest are preserved unchanged. Approved artifacts remain
archived as evidence of the consumed one-shot grant; this result is not a new
runtime grant. The retained lock prevents repeating the command.

Public-only verification, from this branch checkout:

```bash
python3 -B artifacts/resource_applicability/track_a_h4_b0i_pilot6/2026-10-14/verify_published_results.py
```

This verifies source Git blobs, artifact bytes, grant/identity binding and 906
saved-shot cost rows with Fraction arithmetic. It does not read private NPZ or
runtime, build circuits, rerun tests or compile. The collector is an optional
private-evidence projection tool; its output is already committed and no private
runtime is needed for public verification.

Six new real calls, no retry, no subsequent scientific stage. No H8/shared
configuration/private environment/GPU changes. The measurements cover three
q1 configurations, not all deterministic alternatives, all geometries or a
new 1812-point winner map. B2 SE is descriptive and does not certify population
superiority. Transpile-only wall and instantaneous peak RSS were not captured.
