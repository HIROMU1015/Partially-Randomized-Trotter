# New A V5 — fixed native DAG depth bounds

[REPORT](../../../docs/research/pr_reference_depth_v5_20261107/REPORT.md) · [SCOPE](../../../docs/research/pr_reference_depth_v5_20261107/SCOPE.md) · [derivation](../../../docs/research/pr_reference_depth_v5_20261107/DERIVATION.md)

**MANDATORY STOP.** V5-A through E is complete. Reference total expected native DAG depth upper bound is below Bprime lower bound in the fixed model, with a guaranteed reduction of at least 40.995952643300015%. Exact expected depths were not computed. No scientific circuit build/compile, sampling, simulation, synthesis, precision optimization or new native price acquisition occurred. Six artificial circuits were used only for technical depth tests.

- [DEPTH_18ROUND.json](DEPTH_18ROUND.json): all18 per-round X/Y rational lower/upper prices, unchanged shots, lane max penalty, total rational bounds and directed floating enclosures. Inherited CX/RZ/shot/qubit results are copied unchanged.
- [ROUND_DEPTHS.tsv](ROUND_DEPTHS.tsv): 72 round/axis/arm rows, exact per-shot bounds, shots, and factorized shot-weighted contributions. Products are an exact representation, not a new numerical estimate.
- [LOCAL_FRONTIERS.json.xz](LOCAL_FRONTIERS.json.xz): local integer max-plus matrices, spans and system-only progress bounds, 5,284 context comparisons. XZ decompresses to UTF-8 JSON, byte-identical to the local uncompressed output.
- [FENCE_CHECKS.json](FENCE_CHECKS.json): 268 saved pre-final-cancellation D/frame boundary checks.
- [COHORT_DEPTH_CHECKS.json](COHORT_DEPTH_CHECKS.json): 256 saved path/axis/arm structural interval-inclusion checks; not exact topology reconstruction of every cohort circuit.
- [QPY_DEPTH_CHECKS.json](QPY_DEPTH_CHECKS.json): 8 saved whole QPY; reused frontier equals Qiskit depth and saved price, plus interval inclusion. X axis only; no new QPY acquisition.
- [SYNTHETIC_TESTS.json](SYNTHETIC_TESTS.json): 13 technical tests, not scientific performance evidence.
- [INPUTS.json](INPUTS.json), [ENVIRONMENT.json](ENVIRONMENT.json), [SOURCE.json](SOURCE.json): scientific provenance and runtime. The common cache is reused from the fixed pilot, not regenerated.
- [STOP.json](STOP.json): final V5 boundary; old STOPs remain unchanged.

Executed SOURCE entry points were initialize, local_analysis, synthetic_tests, validate_fences, validate_saved_cohort, validate_saved_qpy, account_all18, finish_stop. The first QPY attempt exposed a new metadata assumption incorrectly excluding reference reset instructions. Fixed SOURCE already adds 8 resets after native compilation. That assertion was corrected without changing any saved circuit, depth or scientific condition. A reset/initial-H parallelism test was added, then the successful QPY check, final 13 tests, all18 accounting and STOP were saved. No old scientific runner was invoked.

After STOP, publication-only work compressed the local matrix file, added factorized TSV presentation columns and the equivalent output-serialization lines in final SOURCE, and recorded hashes/documentation. No scientific function was rerun after STOP. The decompressed common.pkl and LOCAL_FRONTIERS.json remain ignored local working copies. No approval certificate or grant was issued.

The reported DAG includes measurement and, for reference, 8 initialization resets. Resets have one parallel layer and overlap the initial ancilla H's 3 native layers. Device durations, routing/noise, wall time, T/T-depth, space-time and all-resource superiority remain unqualified. Next scientific decision belongs to GPT/user.
