# P1 technical test results

|Category|Result|Scope|
|---|---|---|
|STATIC_COMPATIBILITY|PASS|413 exports/order retained, real source signatures and request-field/method names|
|SYNTHETIC_END_TO_END_PASS|PASS|H4 adapter → PF/PR → RTE/event → fixed-signal accounting → mock compile → result|
|PURE_UNIT_PASS|PASS|6 existing integer-validator cases; type/scope/qualification tests in the new suite|
|MOCK_REGRESSION_PASS|PASS|49 new P1 cases plus 7 selected existing executor cases; categories overlap, total 62|
|REAL_SCIENTIFIC_EXECUTION_NOT_PERFORMED|confirmed|no molecular input/state/SCF/operator/sampling/build/compile or numerical N/G|
|NUMERICAL_EQUIVALENCE_NOT_VERIFIED|confirmed|no numerical claim from mocks/signatures|

Final relevant check: **62 PASS, 0 FAIL, 0 SKIP**. No whole-repository suite run.
Environment: isolated task-only Python 3.11.1, pytest 8.3.5, NumPy 1.26.4.
No shared environment was changed. PySCF, OpenFermion, SciPy and Qiskit were not
installed in the task venv. The import test also runs with `-I -S` and no site packages.

The new suite has 49 parameterized cases. Input fields use read-only zero arrays
of shapes (8,8), (12,), and 12 (8,8) blocks, plus numeric fixture constants;
this allocates artificial data without generating a physical Hamiltonian.
All preparation/event/request/circuit objects are interface-shaped mocks, not
scientifically validated concrete core objects. Their signatures/fields are
checked statically, and concrete numerical type/invariant checks remain pending.
 It exercises end-to-end call routing,
explicit compiler/scope/provenance, missing/STOP/failed stages, cross-cohort rejection,
no price/N/G imputation, no event-sampling fallback, optional-dependency refusal,
plan/scientific CLI behavior, import side effects, and atomic new-output publication.
All core scientific callables are mocks. It never calls the real legacy shots
function or wrapper, real core preparation/distribution/compile, or RPE evaluator.

## Selected unchanged tests

- `tests/test_rte_integer_counts.py::test_integer_count_rejects_bool_float_and_string` (5 cases).
- `tests/test_rte_integer_counts.py::test_integer_count_accepts_python_and_numpy_integers` (1).
- `tests/test_parallel_validation_executor.py::test_task_fingerprint_is_deterministic_and_binds_inputs`.
- `...::test_atomic_json_write_leaves_no_temporary_file`.
- `...::test_deterministic_aggregate_ignores_completion_and_runtime_order` (artificial values only).
- `...::test_resource_guard_limits_workers_memory_and_gpu_use` (mock resource metadata only).
- `...::test_resume_skips_completed_task`.
- `...::test_corrupt_checkpoint_is_rejected_without_overwrite`.
- `...::test_mismatched_manifest_checkpoint_is_rejected`.

Executor subprocesses use its artificial adapter only and task-owned temporary
fixtures. They do not access an existing scientific input/cache. The old executor's
resource-cap policy is tested as existing behavior, not adopted by the new pipeline.

## Excluded tests and reason

All 7 functions in `test_df_hamiltonian.py` are excluded: the file calls chemistry,
DF construction, operator application, diagonalization/ground solve or their
numerical conversion path even when one decomposition is mocked. No claim of
scientific equivalence can follow from running them under this instruction.

The other 9 functions in `test_rte_integer_counts.py` exercise finite-law numerical
bounds, samplers or other RTE/scientific constructors beyond plain count validation.
They were not run. The H4 full-wrapper trajectory executor test builds/compiles a
real circuit and was excluded. Remaining executor tests are outside the needed
regression slice; no new scientific coverage is claimed. The old H4 donor's broad
generation/operator/compile suites were not transplanted or run; the new adapter's
own injected mock tests cover the boundary instead.

## Reproduction of the allowed test slice

From this same checkout with pytest/NumPy available:

```bash
PYTHONPATH="$PWD/src" python -B -m pytest -q -p no:cacheprovider \
  tests/test_df_research_pipeline.py \
  tests/test_rte_integer_counts.py::test_integer_count_rejects_bool_float_and_string \
  tests/test_rte_integer_counts.py::test_integer_count_accepts_python_and_numpy_integers \
  tests/test_parallel_validation_executor.py::test_task_fingerprint_is_deterministic_and_binds_inputs \
  tests/test_parallel_validation_executor.py::test_atomic_json_write_leaves_no_temporary_file \
  tests/test_parallel_validation_executor.py::test_deterministic_aggregate_ignores_completion_and_runtime_order \
  tests/test_parallel_validation_executor.py::test_resource_guard_limits_workers_memory_and_gpu_use \
  tests/test_parallel_validation_executor.py::test_resume_skips_completed_task \
  tests/test_parallel_validation_executor.py::test_corrupt_checkpoint_is_rejected_without_overwrite \
  tests/test_parallel_validation_executor.py::test_mismatched_manifest_checkpoint_is_rejected
```

The initial test collection lacked the same-checkout src import path; rerunning
with the shown path resolved that technical setup issue. Later source/fixture
refinements were retested; the final recorded 62 cases all passed. No science run
was used for debugging. Raw local JUnit paths/logs remain private; the public
TEST_MANIFEST.json contains sanitized case names and status only.
