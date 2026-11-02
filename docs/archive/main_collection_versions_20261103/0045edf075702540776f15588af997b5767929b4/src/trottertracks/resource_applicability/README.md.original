# DF research pipeline — selected H4 integration

This new non-frozen entry is available on `all-r-coherent-opt2-reoptimization`.
It selectively adopts P1 `af54aeb7ebac042a3ca02379d61516ac635feef9` following [adoption review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/76fec27f45b64879cdf5a86a32772ba78bd0a038/inventory/integration/p1_adoption_review_20261027).
Integration base: `f29eafbf4412193e8e8660d9f051889c2cba684b`. This README's commit is the new code version;
the [old P1 implementation/test history](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af54aeb7ebac042a3ca02379d61516ac635feef9/docs/integration/p1_df_pipeline_h4_20261027) remains a separate immutable record.
No scientific execution or numerical equivalence is established.

## End-to-end responsibility

```text
CLI (default plan / explicit synthetic)
  → DF compose(context, H4GeometryAdapter, providers)
  → saved-field input boundary (one/nuclear/lambdas/G + fixed partition)
  → existing prepare_df_partial_s2 with explicit H4 policy
  → existing make_rte_config + caller-supplied event/step requests
  → fixed-signal accounting boundary, separate from phase-RPE
  → existing build_step / study wrapper / compiler API boundary
  → qualified stage records and versioned evidence locators
```

The artificial fixture uses read-only zero arrays of the expected field shapes.
All numerical preparation, event/request, shot, wrapper and compiler callables are mocks.
No physical Hamiltonian/state is generated or evaluated. Normalization, N/G and compiled price remain missing/unqualified.
`SYNTHETIC_END_TO_END_PASS` means control/data routing, with `science_executed=false` and numerical equivalence false.
Context binds SOURCE/input/state/cohort/result version, representation/rank/partition/law,
precision/estimand/compiler/metric/cost scope and preserved STOP. Different bindings fail;
STOP/FAILED/UNQUALIFIED/MISSING stop without retry. DF fields are not a mandatory schema for Pauli or other research.

## API and study policy

- `pipeline.py`: ScientificContext, StageResult, StageProviders, compose and new-directory publication.
- `h4_geometry_pipeline.py`: FrozenH4Input and H4GeometryAdapter, including fresh `preparation_policy` and explicit `bind_core(context)`.
- `df_core_stages.py`: same-checkout core delegation. `bind_existing_core(signal_shots=..., signal_wrapper=..., estimand=...)` rejects missing callbacks before imports.
- `pipeline_fixture.py`: artificial context and mock providers through those same stage wrappers.

H4 adapter validation/input logic and the selected legacy shots/wrapper bodies are unchanged from P1.
H4 B2/non-B3, rank12/STO-3G, generated-order prefix, S2/q1, sector, geometry allowlist,
fixed complex signal, RZ metric and P=0 full single Hadamard scope remain study conditions.
The five explicit preparation fields retain their original values:

|Field|H4 value|
|---|---|
|identity_policy|extract_identity_phase|
|coefficient_atol|0.0|
|diagonal_sort|descending_abs|
|basis_construction_policy|fermionic_gaussian_jw|
|partition_policy|explicit_ordered_partition|

The common stage wrapper supplies no policy defaults. It validates presence/types and forwards the adapter's policy;
it does not choose L_D, rerank, repair input or sample events.
H4 `bind_core` passes the original shots/wrapper functions explicitly, without calling them.
Fixed-signal binding explicitly requests no phase-RPE module/callback. Only explicit `phase_rpe` binding resolves
`evaluate_rpe_round_candidate`. Bound estimands cannot be silently exchanged.
The unchanged legacy eager package initializer may still import phase-RPE transitively:
this implementation removes the new binding's unnecessary request, not that existing global behavior.
No registry, alternate package loader, global lazy initialization or authorization-proof system is introduced.

## Commands and output dependencies

Python >=3.11 is required. Plan uses the standard library; synthetic needs NumPy.

```bash
python scripts/run_df_research_pipeline.py
python scripts/run_df_research_pipeline.py --mode synthetic
python scripts/run_df_research_pipeline.py --mode synthetic --output-dir NEW_OUTPUT_DIRECTORY
```

Plan prints the built-in artificial context and opens no input, calls no provider and creates no directory.
Scientific/real/resume are disabled before imports/input/output. `scientific_providers` always refuses;
callbacks are interfaces, not execution permission. No real saved-input loader or scientific provider is enabled.
Missing context, input/cohort binding, stage callback or preparation policy is rejected without a scientific fallback.

Optional output reuses `trotterlib.parallel_validation_executor.atomic_write_json` under its original import name.
The old eager initializer is byte-identical to the integration base and requires its existing optional stack.
If that import lacks a dependency, PipelineError retains the ModuleNotFoundError cause;
the CLI explains the missing module and creates no output directory. Use stdout or an already suitable environment.
It does not install dependencies, load real code under another package name or modify a shared environment.
When the writer is available, output is a newly created directory containing `pipeline_result.json`;
existing directories are refused and old outputs are not overwritten.

## Actual integration verification

**79 cases collected: 78 PASS / 0 FAIL / 1 SKIP.**
Environment: task-only Python3.11.1, pytest8.3.5, NumPy1.26.4 reused without package installation.
The original 46 study-wiring/boundary case topics are retained; three facade/output assumptions are reworked.
New checks cover H4 callback/policy propagation, missing callback/policy/stage providers,
explicit phase binding and estimand mismatch, SOURCE/compiler/metric/cohort mismatch,
all disabled CLI modes, preserved dependency cause, no-directory errors and artificial-writer collision behavior.

The real existing atomic publication/collision case is **SKIP**: the eager initializer fails on missing `matplotlib`.
The directory/collision boundary with an artificial writer passed, and is explicitly not a real atomic-writer check.
The old 13 selected cases (integer6/executor7) were **not run**: ordinary imports of both modules fail on the same dependency.
They are not counted as 13 extra SKIPs or PASSes. No alias loader/package bypass was used to make them run.
Historical P1 62 PASS and adoption-review 19 PASS are separate records, not results of this integration.

From the same checkout with pytest/NumPy available:

```bash
PYTHONPATH="$PWD/src" python -B -m pytest -q -p no:cacheprovider tests/test_df_research_pipeline.py
```

Static checks confirmed exactly the authorized14 changed files, byte-identical eager initializer,
unchanged existing core Python modules/runners/scientific artifacts, unchanged H4 validation/input/shots/wrapper bodies,
and no H4 import added to common core bindings. No old runner, science suite, numerical reanalysis,
sampling or Qiskit build/compile was executed. Technical imports stop at missing dependencies in the real-stack probes.
Original dirty checkout, P1 fixed branch, SOURCE/input/raw/result/cache/grant/ledger/marker/STOP,
CEM/validation manifest and other research environments remain outside the change.

## Limits and P2 handoff

Real input loader/authoritative identity, scientific provider/runtime, sampling, mean/bias/normalization,
qualified N/G/price, q>1/full RPE and numerical/operator/phase/compiler equivalence remain unverified or unconnected.
No performance or scientific superiority claim follows from mocks.
P2 can reuse the responsibility and explicit callback/policy/qualification boundaries described in
[the fixed P2 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/76fec27f45b64879cdf5a86a32772ba78bd0a038/inventory/integration/p1_adoption_review_20261027/P2_HANDOFF.md), together with the latest formal Pauli handoff.
It must reuse existing Pauli implementations and preserve its own frozen namespaces, laws, states/cohorts and accounting;
neither DF rank12/context nor H4 callbacks/compiler scope are Pauli defaults.
P2 implementation, real scientific validation, H6/H8 rollout and DF-native B3 are separate explicitly authorized tasks.
No B3 access, SERVER_A/B connection/transfer, other-job operation or shared environment change was performed.
Mandatory STOP after normal publication and remote verification.
