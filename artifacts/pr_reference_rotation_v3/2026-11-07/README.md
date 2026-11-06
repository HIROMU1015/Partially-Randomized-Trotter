# V3 saved-data rotation accounting — 2026-11-07

[REPORT](../../../docs/research/pr_reference_rotation_v3_20261107/REPORT.md) · [SCOPE](../../../docs/research/pr_reference_rotation_v3_20261107/SCOPE.md) · [classification](../../../docs/research/pr_reference_rotation_v3_20261107/CLASSIFICATION_RULES.md)

V3 is complete and [STOP.json](STOP.json) is mandatory. No V5, new compile, simulation, sampling, synthesis, or precision allocation was run. These are author checks, not independent replication or CI.

- [CLASSIFIED_18ROUND.json](CLASSIFIED_18ROUND.json): all 18 rounds, both X/Y axes, 3 arms; exact rational lower/upper counts per shot, saved shots, whole-RPE totals and inherited CX/RZ/shot/qubit figures. Primary ledger `literal_native`; supplementary `analytic_provenance`. Resolved arbitrary subset falls 2.818200% against Bprime; total arbitrary count advantage remains **UNQUALIFIED** because unresolved counts overlap.
- [ANGLE_KEY_MULTIPLICITIES.json.xz](ANGLE_KEY_MULTIPLICITIES.json.xz): signed binary angle tokens, category/provenance, round, axis, arm, role, exact multiplicities and fusion corrections. XZ decompresses to UTF-8 JSON. Key reuse is not gate sharing.
- [LOCAL_INVENTORY.json.xz](LOCAL_INVENTORY.json.xz): 2,165 saved local record projections and D context histogram checks. XZ decompresses to UTF-8 JSON, byte-identical to the local uncompressed output.
- [COHORT_CHECKS.json](COHORT_CHECKS.json): 384 saved whole-price comparisons, 64 paths × 2 axes × 3 arms; no recompilation.
- [QPY_CHECKS.json](QPY_CHECKS.json): signed RZ histogram checks on 14 existing native QPY, no new circuit build/compile. Other paths do not acquire QPY observations by inference.
- [SYNTHETIC_TESTS.json](SYNTHETIC_TESTS.json): 15 artificial checks, not performance evidence.
- [REAL_BASIS_WITNESS.json](REAL_BASIS_WITNESS.json): cached matrices/numeric phase observations; no eigensolve and no promotion to symbolic Clifford provenance.
- [SYNTHESIS_REUSE_AUDIT.json](SYNTHESIS_REUSE_AUDIT.json): read-only inspection of 7,603 old strict synthesis records; no T price/word reused and no new guard evaluation.
- [INPUTS.json](INPUTS.json), [ENVIRONMENT.json](ENVIRONMENT.json), [SOURCE.json](SOURCE.json): fixed input, environment and scientific provenance.

The existing `common.pkl.xz` is reused from the pilot at parent commit 43bfc5918703e8dcd020be4fcf96bb8dfd99f909. The decompressed cache and uncompressed local inventory remain ignored local working copies; no old artifact is changed.

Executed entry points in `src/trottertracks/pr_reference_rotation_v3.py` were `initialize`, `real_input_witness`, `inventory`, `validate_cohort`, `validate_qpy`, `synthetic_tests`, `audit_synthesis_records`, `account_rounds`, and `finish_stop`. Technical corrections to numeric beta handling and QPY arm filtering preceded final checks/accounting. The successful QPY check reused saved cohort projections. Final serialization exposed the primary ledger and represented exact rational zero as zero; numeric fractions and scientific conditions were unchanged. The final witness annotation/comment was corrected to the actual ops.py/Qiskit Nature source. Packaging and document/link/hash validation followed STOP; scientific functions were not rerun.

These entry points are an execution record, not permission to resume. `attach()` rejects the saved STOP. Next scientific decision belongs to GPT/user; V5 or synthesis needs separate explicit approval.
