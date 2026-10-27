# API compatibility and boundaries

- Architecture: `1b649896393a4a0c5f2e3bcf41b2853090ef2053` / `inventory/integration/research_architecture_consolidation_20261027/` (P0/P1).
- Core/base: `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`.
- H4 donor: `codex/h4-tol-p-seal-20261019` @ `821cc0f6ced57168293ea166644024867b064ffb`.
- Existing H4 geometry SOURCE catalogue: [821cc0f6ced57168293ea166644024867b064ffb:source_freeze_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/source_freeze_v1.json). Its `source_commit` is `6d365257770e99022b91d6a38dbee49ee0077503`; it is not the P1 code version.
- Old geometry contract: `b662dbd72e49fa713a25c716f323843e547e973b` / `artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/zero_compute_plan_v2.json`.
- H6 reference only: `3cccfcb2051dacf29e91473e2e1152acef1574bd`; native reference only: `7e41670565d9284da9aded94063be591fdcaa3dd`. Neither study was migrated or run.


## Stage interfaces

|Stage|Input → output|Existing implementation|Boundary retained|
|---|---|---|---|
|Input/context|FrozenH4Input + context → DFHamiltonian + fixed partition|`df_hamiltonian.DFHamiltonian`|corrected one-body + nuclear once; no factorization/solve|
|PF/PR|H/partition → DFPartialS2Preparation|`df_partial_s2.prepare_df_partial_s2`|explicit_ordered_partition, extract_identity_phase, coefficient_atol=0, descending_abs, fermionic_gaussian_jw|
|Finite RTE|symbolic tail + time/r/K/seed/tolerance → config/distribution|`rte.make_rte_config`|no new law, finite Taylor order, no automatic order choice|
|Event|caller-supplied events → DFRTEEventSequenceCircuitRequest / DFPartialS2StepRequest|existing dataclasses|original event order, ancilla after system, exact r count; no sampler|
|Signal accounting|explicit supplied normalization/bias/epsilon → N axes or missing|selected H4 `shots` callable|fixed complex signal; alpha-axis and P=0 convention unchanged, G not filled|
|Phase-RPE|separate evaluator binding|`rpe_resource_accounting.evaluate_rpe_round_candidate`|cannot receive H4 fixed-signal context; no phase budget invented|
|Circuit/price|step request → S2 circuit/wrapper → cost|`QiskitDFPartialS2CircuitBuilder.build_step`, selected H4 wrapper, `transpile_and_measure_cost`|compiler profile explicit; mocked, no build/transpile|
|Result|stage qualifications → serialized control record|new PipelineResult, existing executor `atomic_write_json`|no opaque arrays/state serialized, no retry/checkpoint/old output mutation|

`ScientificContext` explicitly carries geometry/basis, sector/state/input identities,
rank and truncation policy, LD and exact partition indices, S2 and stage order, q,
time/r/K/law/normalization/seed, precision/domain, compiler profile, estimand, metric,
cost scope, SOURCE/cohort/result version and preserved old STOP status. This is a
new-entry interface, not a mandatory change to any old scientific schema.

## Compatibility actually checked

STATIC_COMPATIBILITY: the actual existing function definitions accept the injected
wrapper positional/keyword calls. Dataclass field names and builder method names
were checked from source. All 413 original package-level export names map to the
same module/attribute, and original `__all__` order is retained. Direct RTE export
identity is tested. Unknown exports still raise AttributeError.

The facade change intentionally defers optional imports/errors until an attribute
is used. It does not preserve the old eager import timing or incidental package
initialization side effects. This dependency-loading change is the only existing
core Python-file change. Full scientific runner behavior was not rerun or certified.

All pre-existing numerical module bodies, CLI files, tests, SOURCE, CEM, manifest,
result/raw and output/cache contracts are unchanged relative to the base. Donor
objects are read from pinned Git, not imported from another checkout. No absolute
worktree path or temporary external PYTHONPATH is a runtime dependency. Test
PYTHONPATH refers only to this development checkout's `src`.

## Incompatibilities that remain explicit

- H4 old full factorization ordering differs from the core default ranking rule.
  The adapter requires an already-fixed explicit partition; no sorting fallback.
  Actual saved representation/partition compatibility needs separate scientific validation.
- The selected H4 geometry path is rank12 generated order, not H4 tol-only rank10
  or the H6 confirmation model. Other geometries/methods are not silently accepted.
- q=1 only. Repeated evolution and whole-round RPE are not connected. H4 fixed
  signal, state-preparation-free wrapper and phase-RPE accounting are distinct.
- Raw-state/sector bit order, precision, scalar/global phase/control and actual
  operator equivalence require later evaluation; no equivalence is inferred from signatures.
- Full core import requires the original optional dependencies; no fake scientific
  fallback or package installation on import is provided.
- `None` denotes missing science. Artificial token or mock outputs are never a
  scientific qualification, even when end-to-end control wiring passes.
