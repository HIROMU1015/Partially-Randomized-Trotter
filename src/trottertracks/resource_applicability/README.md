# P1 DF research pipeline — H4 B2/S2 wiring

This new, non-frozen path connects a study adapter to existing core API boundaries.
It does not replace any fixed runner or authorize scientific execution.

```text
CLI (plan / explicit synthetic)
  → compose(context, H4GeometryAdapter, providers)
  → frozen H4 input conversion (one/nuclear/lambdas/G + fixed partition)
  → existing prepare_df_partial_s2 (explicit ordered partition)
  → existing make_rte_config + explicit event/step request (no sampler)
  → H4 fixed-signal accounting boundary (separate from phase-RPE)
  → existing S2 builder / wrapper / compiler boundary
  → qualified stage records + versioned evidence locators
```

`df_core_stages.py` contains calls to the actual same-checkout APIs. The synthetic
fixture replaces those callables, not the composer or its stage connection code.
The fixture allocates read-only zero arrays with the saved-field shapes; no
operator, circuit, law probability, bias, shot count or price is evaluated. N, G, normalization and compiled price stay `None`/not acquired. Its PASS
means control/data wiring only.

## Commands

From this checkout, using an interpreter with Python >=3.11:

```bash
python scripts/run_df_research_pipeline.py
python scripts/run_df_research_pipeline.py --mode synthetic
python scripts/run_df_research_pipeline.py --mode synthetic --output-dir NEW_OUTPUT_DIRECTORY
```

Default mode is a plan for the built-in **artificial** H4 context: no input is opened,
no stage/provider called and no output directory created. Synthetic mode explicitly
uses mock providers. `--mode scientific` always exits with an error before imports,
input access or output creation. There is no `--approve`, grant, receipt or other
permission-proof mechanism. Old STOPs remain untouched.

The optional output creates a new directory exclusively and reuses the existing
executor's `atomic_write_json` for `pipeline_result.json`. Existing directories are
refused. It does not launch an executor, create checkpoints or inherit its worker
caps, retry/resume policy or scientific adapter selection.

## Public API

- `pipeline.py`: explicit `ScientificContext`, `CompilerProfile`, evidence,
  stage/result records; small sequential composer with injected providers.
- `h4_geometry_pipeline.py`: one H4 **B2** (non-B3), fixed rank12, STO-3G,
  frozen-generated-order prefix, S2/q=1, fixed complex-signal P=0 adapter.
- `df_core_stages.py`: lazy binding of existing APIs; separate PF preparation,
  finite-law/event request, fixed-signal accounting and compile calls.
- `pipeline_fixture.py`: explicit artificial zero arrays and mocks; no scientific data. Synthetic mode needs NumPy; plan/import do not.

H4-specific distances, sector, rank/order, wrapper, estimand and metric restrictions
stay in the study adapter. The common numerical kernels have not been changed.
The public package facade now loads its original 413 named exports on demand so
lightweight use need not load chemistry/Qiskit/plotting. All original names, target
module/attributes and `__all__` order are statically unchanged. Optional dependency
errors are deferred until the relevant export is used, not replaced by fake code.

## Limits and follow-up

Only one S2 step is connected. Repeated q>1, H4 B0/B0I or other methods, real saved
input loading/identity verification, runtime authorization, real event acquisition,
mean-signal evaluation, whole-round RPE and scientific end-to-end execution are not
implemented by this entry. A caller-supplied frozen partition must already satisfy
the existing core's explicit-partition semantics; no rank or prefix is selected here.
The old donor `signal.prepare`/`validate_frozen_state` includes diagonalization,
operator evaluation or build and is deliberately not called.

Real function bindings and request types are present, but numerical equivalence of
the generated-order H4 representation to this core path, signed/control/phase
behavior, compiler costs, shot qualification and RPE performance remain unverified.
Do not use an artificial fixture or this new code version as the SOURCE of old
results. See the five reports under `docs/integration/p1_df_pipeline_h4_20261027/`.
