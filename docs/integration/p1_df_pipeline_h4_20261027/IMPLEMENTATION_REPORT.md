# P1 implementation report

Status: implementation and technical validation only. No scientific execution.

## Implemented structure

The study-specific CLI calls the thin composer, then an H4 adapter and injected
existing-core stage operations. The new entry is `scripts/run_df_research_pipeline.py`.
`compose` supports plan and synthetic modes only; real execution is refused.

The H4 input boundary maps the donor's saved `one` (already corrected), `nuclear`,
`lambdas` and `G` into `DFHamiltonian` without regenerating chemistry or states.
The pre-frozen partition/order is explicit. Preparation delegates to
`prepare_df_partial_s2(..., partition_policy="explicit_ordered_partition")` rather
than reranking or selecting L_D. The RTE boundary delegates to `make_rte_config`
and explicit event/step request constructors; it never uses the convenience
`make_df_partial_s2_step_request`, which would sample.

The composer passes stage qualification and complete study/input/SOURCE/cohort/
version/compiler/metric binding. STOP, failure, unqualified or missing stages halt
the chain without retry. No missing signal, N/G or price is inferred.

`df_core_stages` binds the real same-checkout modules and call signatures, including
the S2 builder that internally reuses `df_rte_qiskit`. Mock execution uses exactly
these wrapper functions with injected mocks. Phase-RPE's existing evaluator is a
separate callable and is rejected for this H4 fixed-signal context.

## Selection and source reuse

- Architecture: `1b649896393a4a0c5f2e3bcf41b2853090ef2053` / `inventory/integration/research_architecture_consolidation_20261027/` (P0/P1).
- Core/base: `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`.
- H4 donor: `codex/h4-tol-p-seal-20261019` @ `821cc0f6ced57168293ea166644024867b064ffb`.
- Existing H4 geometry SOURCE catalogue: [821cc0f6ced57168293ea166644024867b064ffb:source_freeze_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/source_freeze_v1.json). Its `source_commit` is `6d365257770e99022b91d6a38dbee49ee0077503`; it is not the P1 code version.
- Old geometry contract: `b662dbd72e49fa713a25c716f323843e547e973b` / `artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/zero_compute_plan_v2.json`.
- H6 reference only: `3cccfcb2051dacf29e91473e2e1152acef1574bd`; native reference only: `7e41670565d9284da9aded94063be591fdcaa3dd`. Neither study was migrated or run.


The remote all-r and three reference tips matched the plan on entry. The original
LOCAL primary HEAD was a descendant of the public fixed base and dirty; it was
not switched, committed or used as a source of private/unpublished changes.
An independent clone and dedicated branch start from the fixed public base.

Selected H4 behavior, not the whole donor history, was carried over:

|Donor location|New use|
|---|---|
|inputs.py:191-202|saved representation mapping; no generation/state solve|
|signal.py:96,151-159|frozen generated-order split, one-body/scalar policy and symbolic context|
|signal.py:200-203|fixed-signal `shots(B,bias,epsilon)` arithmetic, with equivalent input rejection|
|circuits.py:290-298|Hadamard wrapper gate sequence, only behind explicit callable|
|gates.py / old contract|distance/sector/scope and locator references; no host path, permits or old launch machinery copied|

No old H4 runtime, ledger, grant/receipt, output layout, worker/cap controls,
frozen-state validator or numerical preparation was copied into the new launcher.
Both selected numerical/wrapper callables were mocked in all P1 tests.

## Main files

- `src/trottertracks/resource_applicability/pipeline.py`: contexts, stage records,
  composer and optional new-directory result publication.
- `h4_geometry_pipeline.py`: H4 boundary and two selected legacy callables.
- `df_core_stages.py`: actual-core call delegation and explicit missing-science boundary.
- `pipeline_fixture.py`: artificial zero-array fixture / injected mock API providers.
- New CLI and `tests/test_df_research_pipeline.py`.
- `src/trotterlib/__init__.py`: dependency-safe lazy facade; 413 export targets/order retained.
- Relevant README/index additions and these reports. No CEM or validation manifest update.

## Implemented versus not connected

|Capability|P1 status|
|---|---|
|H4 adapter → each stage → result|synthetic end-to-end PASS through real wrapper call sites|
|DF/PF/RTE/request/compile API signatures|static plus mocked call compatibility|
|Real dependency binding|explicit lazy binding code, optional-dependency rejection tested; full chemistry/Qiskit environment not installed or exercised|
|Signal accounting|legacy H4 callable connected and mocked; no signal/bias/N evaluated|
|Phase-RPE|separate existing evaluator binding only; not an H4 fixed-signal capability|
|Executor mechanism|existing atomic result publication reused; job scheduling deliberately not wired|
|Scientific execution|disabled, not a completed scientific pipeline|
|q>1 / whole-RPE / real input identity checks / sampling|not connected by this entry|

Technical integration is ready for a separately scoped scientific validation
decision, not for executing a frozen study. Acceptance does not mean numerical
equivalence, new scientific GO, existing SOURCE approval or an operational real mode.
