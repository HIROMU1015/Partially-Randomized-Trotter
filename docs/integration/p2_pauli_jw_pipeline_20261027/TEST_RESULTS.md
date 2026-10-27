# P2 technical test results

Actual current run: **109 PASS / 0 FAIL / 1 SKIP** (110 collected).
Task-only Python3.11.1 / pytest8.3.5 environment was reused; no new packages or shared changes.
Plan/synthetic probes also ran with `-I -S` and no site packages, NumPy, SciPy, Qiskit, FLINT, PySCF or OpenFermion imported.
The real core availability probe, using its original package name, fails on missing `matplotlib`.

## What passed

- H4 Hybrid, H6 Hybrid and H6 pure-RTE artificial end-to-end wiring (3 cases), and all four existing H6 signatures (4 cases).
- Supplied terms/sign/constant identity and config/law/trajectory/mean/account/compile signature routing.
- SOURCE/bit/label/state/norm/method/prefix/q/r/K/PF/law/normalization/axis/control/prep/phase/compiler rejection.
- Distinct source/input/state/norm/seed/cohort-role/cohort-ID/compiler stage binding; wrong seed stops before mean.
- Missing callback refusal before input; real input/provider refusal; STOP/MISSING/FAILED/UNQUALIFIED halt once without retry.
- No unacquired N/G/SE/price/rare-event probability imputation, including zero/infinity/NaN substitutes.
- Adapter-selected canonical import paths, missing-dependency cause and static actual API/dataclass compatibility.
- Byte-preserved global initializers; all-site-free CLI paths, H2 locator-only entry, disabled science modes before output.
- Optional writer error before directory creation; directory collision/original-output preservation with an explicitly artificial writer.

All numerical callbacks are Mock objects, including Hamiltonian construction, config/law generation,
seed acquisition, trajectory/build/wrapper, mean/bias/normalization, N/G and compiler price.
Arbitrary signed unit terms are invented strings/scalars, not recovered scientific data or physical Hamiltonians.
No actual event distribution, sampling, state/vector/matrix operation, numerical oracle or Qiskit operation ran.

## SKIP and tests not run

One SKIP: `test_real_existing_writer_if_normal_dependencies_are_available`.
The old eager initializer requires missing `matplotlib`; writer import fails before directory creation.
Artificial-writer tests do not prove real atomic publication in a complete environment.

Existing sources were inspected, not run:

|Test source|Reason excluded|
|---|---|
|test_pauli_hybrid_rte_preparation.py (7)|artificial saved-row tests, but the file loads real modules/scripts under alternate names; P2 forbids using alias loading to bypass the import contract|
|test_pauli_rte_event_extension.py (6)|numerical finite law/seed/sampling/trajectory/operator/build/compile plus alternate package setup|
|test_h6_hybrid_fairness_extension.py (8)|mixes mock tests with law/seed/circuit/shot/covariance work and alternate package setup; not collected as a whole|
|test_h6_hybrid_fairness_extension_mean.py (2)|state action/Arb oracle/mean and alternate package setup; not permitted|

The safe routing/identity/cohort/qualification topics are covered by new tests without executing real code under an alias.
These 23 old test functions are **not run**, not extra SKIPs or PASSes. No broad repository suite ran.
P1's previous 78 PASS/1 SKIP, its older62 PASS and adoption19 PASS are not added to P2 counts.
The first P2 run had four test-fixture identity assertions using Mock.return_value although the mock used side_effect;
the assertions were corrected to inspect the actual returned payload. No numerical workaround was used.
Final tests also cover the adapter-selected callback boundary; all final failures are zero.

## Reproduction and evidence

From this same checkout with pytest available:

```bash
PYTHONPATH="$PWD/src" python -B -m pytest -q -p no:cacheprovider tests/test_pauli_jw_research_pipeline.py
```

Private task-owned JUnit/test summaries retain actual case names/statuses. The test source and this grouped report are public.
All results are the primary Codex's technical checks, not an independent scientific review.
Not verified: complete optional stack, real payload loading/identity/state/norm/term ordering, law/operator/phase/compiled equivalence,
numerical inclusion, N/G/SE, statistical coverage, performance or scientific superiority.
