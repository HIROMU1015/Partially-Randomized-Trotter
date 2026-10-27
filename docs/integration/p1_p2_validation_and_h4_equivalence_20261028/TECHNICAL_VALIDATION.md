# Technical validation

The isolated task venv uses Python3.11.1, pytest8.3.5 and the primary pinned
versions in [requirements-validation.txt](requirements-validation.txt).
Primary numpy/scipy/matplotlib/openfermion/openfermionpyscf/pyscf/qiskit/numba
pins reuse the repository requirements. Jupyter, qiskit-aer and qiskit-nature
were unnecessary and not installed. python-flint0.6.0 was added only because
the normal Pauli callback module imports it; no Arb scientific arithmetic or
historical-environment numerical equivalence is claimed. Transitive packages
are recorded with `pip freeze` in the private task record; `pip check` passes.
Neither shared venv nor old research environments/packages were modified.

The focused static closure of the unchanged eager initializer includes 67 local
modules for each fixed version. Normal import uses real installed packages.
Both versions expose 413 original ImportFrom names with identical provider
objects, also after star import and reload. P1's adapter binds actual same-checkout
DF core APIs, and P2 binds declared actual callback objects for H4-Hybrid,
H6-Hybrid and H6-RTE. These bindings were imported/inspected, never executed.

## Final current test executions

| Slice | PASS | FAIL | SKIP |
|---|---:|---:|---:|
| P1 new pipeline at `2c865c66171fc9d1972b67c1257c57e5684fe26c` | 79 | 0 | 0 |
| P1 legacy slice (integer6 + executor7) | 13 | 0 | 0 |
| P1 new writer11 + eager export/import2 | 13 | 0 | 0 |
| P2 pipeline at `f0a035961583d976e8967511a78f9da66758f0bb` | 121 | 0 | 0 |
| P2 legacy selected variants | 13 | 0 | 0 |
| P2 new writer11 + eager export/import2 | 13 | 0 | 0 |

Exact current case names are in [TECHNICAL_TEST_RUNS.json](TECHNICAL_TEST_RUNS.json).
Old test counts and intermediate repeats are not added to these results.
Initial added-test failures were JSON tuple/list expectation errors, corrected
in the test only. No implementation fix, SOURCE/kernel/initializer/runner edit
or representative-branch promotion was needed.

## Actual writer boundary

`trotterlib.parallel_validation_executor.atomic_write_json` is called through
each real pipeline writer. Artificial records/old files live only in fresh
task/pytest directories. Tests observe the old destination and complete sibling
temporary file immediately before actual os.replace; final content and cleanup
are checked. Stream-write, fsync, replace errors and KeyboardInterrupt are
injected into that actual writer path. The original exception object propagates,
old bytes survive, and temporary files are absent. JSON serialization failure
also preserves old output. Existing directory/file collision is refused.
A new-wrapper failure leaves an empty newly created directory; automatic retry
into it is refused. No claim of power-loss directory durability or cleanup
after uncatchable process termination is made by these tests.

## Legacy test selection

The P1 six integer tests only validate plain integer input. Its seven executor
tests cover task fingerprint/input binding, atomic JSON, artificial aggregation,
mock resources, completed resume, corrupt checkpoint and manifest mismatch.
Their subprocesses use the existing artificial adapter in new task fixtures.
Old real H4 trajectory/build/compile tests, finite-law/sampler/operator tests
and the remaining broad executor suite were not run.

From the earlier P2 23 functions, selected 13: preparation7 (artificial saved
values/missing/cohort handling), event-domain/seed metadata2, H6 dispatch/identity/
mean-callback/cohort mocks4. Ten were excluded: event-law/operator/circuit tests4,
H6 law/sampling/compile/accounting-covariance tests4, mean-oracle tests2.
The selected bodies are published with explicit normal package imports; the
original alternate file/package loaders are not used. No physical H6/Pauli
input was opened and no numerical oracle ran. The seed test checks metadata
addresses, not trajectories or sampling.

## Reproduction

Use independent fixed DF and Pauli checkouts, install the listed packages in a
new venv, then run the versioned technical harness with that venv's Python:

```bash
python scripts/integration/run_p1_p2_technical_validation.py --pauli-checkout PAULI_CHECKOUT
python scripts/integration/run_p1_p2_technical_validation.py --pauli-checkout PAULI_CHECKOUT --execute --output NEW_PRIVATE_OUTPUT
```

The first command prints the test plan only. The second accepts only the fixed
technical slices, checks original implementation scopes through Git, refuses
existing output and writes separate logs/JUnit. The H4 scientific gate is not
implemented as an automatic launch mode. No optional installation occurs inside
the harness. Use ordinary package imports and the proper checkout src path;
P2 modules are not imported from another worktree into the DF scientific kernel.
