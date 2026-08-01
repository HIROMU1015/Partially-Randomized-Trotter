# Latest-main audit for the finite RTE milestone

Audit date: 2026-08-01 (Asia/Tokyo)

Audited branch: `main`, synchronized with `origin/main` before changes

Audited commit: `60287636d6bef0e56f1593fcf756bae102aa716c`

## 1. Tests before modification

Command:

```text
.venv311/bin/python -m pytest -q
```

Collection contained 39 tests in five tracked files:

- `test_df_hamiltonian.py`: 7
- `test_df_partial_randomized_pf.py`: 10
- `test_grouped_uwc_comparison.py`: 7
- `test_pauli_partial_cgs_validation.py`: 8
- `test_uwc_preprocessor.py`: 7

Result: **39 passed, 4 warnings in 2.31 s**.  All four warnings were the same
`ComplexWarning` in `chemistry_hamiltonian.py:307` reached by grouped-UWC tests.

## 2. Stale and quarantined artifacts

The tracked file
`artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json`
remains explicitly quarantined by `VALIDATION_STATUS.md`,
`artifacts/validation_manifest.json`, and the artifact README.  Its 35-entry
Cgs source tables were removed in commit `98f960c` after a ground-state
mismatch, so its 635 internally consistent candidates are not valid research
results and must not be used as test expectations.

The corrected aggregate/split DF Cgs tables are still absent from tracked
files.  Several ignored local H3/H4 regeneration, diagnostic, cache, and
screening files exist in the working checkout; they were not modified and are
not canonical repository evidence.  Grouped-UWC machine-readable outputs
named by the prose report also remain untracked/missing.  Historical high-order
PF coefficient artifacts remain historical evidence only.

## 3. DF Hamiltonian hash and cache key

`_df_hamiltonian_hash` in `df_partial_randomized_pf.py` hashes a canonical JSON
payload containing the constant, full one-body array hash, full lambda array
hash, each full G-matrix hash, shapes/dtypes, selected metadata, and the weight
rule.  Rounded scalar summaries and norms are included for inspection, but
tests confirm that matrices with equal norms and different entries have
different hashes.

The ground-state cache payload is schema version 3 and contains that
Hamiltonian hash, a physical-sector hash (including the exact basis-index byte
hash), matrix-free backend/thread/chunk settings, `eigsh`, `ncv`, solver
tolerance, and `expand_state=True`.  The filename is SHA-256 of the canonical
payload.  Loading checks the schema, key, and payload SHA-256.  The Cgs JSON
cache independently uses schema version 7 and
`df_hd_deterministic_surrogate_v2`.

## 4. Constant and one-body terms at an `L_D` split

`split_df_hamiltonian_by_ld` ranks and divides only the squared DF two-body
fragments.  `select_df_h_d` calls `DFHamiltonian.select_blocks`, which retains
the complete constant and one-body correction for every `L_D`, including
`L_D=0`.  Therefore the current tail comprises only the unselected DF squared
fragments; the constant and one-body terms are always deterministic.

`lambda_r` is the sum of the chosen fragment ranking weights
(`abs(lambda_l) * ||G_l||_F^2` by default).  It is a DF screening proxy, not yet
the Pauli/involution LCU 1-norm required by Appendix A RTE.

The DF circuit builder applies the retained constant as an explicit circuit
global phase (`energy_shift`).  That is correct for uncontrolled statevector
evolution; a future controlled event must turn it into relative ancilla phase.

## 5. Meaning of the existing `rte` option

Before this milestone there was no RTE event distribution or RTE event
circuit.  Selecting `randomized_method="rte"` only changed
`randomized_gamma` from 1 to 2 in the simplified

```text
B0 = (280/9) * G_rand * gamma * (0.1*pi)^2
B(kappa) = B0 * kappa * exp(2/kappa)
G_rand_proxy = B(kappa) * lambda_r^2 / eps_qpe^2.
```

It did not change a sampled circuit, integer RTE steps, Taylor order, finite
normalization, attenuation, shot count, or transpilation.  This path is now
labeled `legacy_analytic_proxy` / fidelity Level 0 in newly produced outputs.

## 6. Meaning of the existing DF circuit cost

The screening cost is not a transpile of the complete product-formula circuit.
Each basis-change primitive is transpiled separately; its RZ count/depth is
summed twice per occurrence.  Diagonal RZ/RZZ cost is computed by an analytic
dependency model and added to the primitive sums.  Consequently it misses
cross-block cancellation, rotation fusion, routing/layout overhead, and global
compiler optimization.

`scripts/check_partial_randomized_diagnostics.py --full-circuit` can build and
transpile a complete deterministic step for selected diagnostics, but this is
optional and is not the canonical screening value.  No finite-RPE round or RTE
event cost existed at audit time.

## 7. Test tracking policy

The previous `.gitignore` re-included only
`test_pauli_partial_cgs_validation.py` after ignoring `tests/*`.  Existing
tracked tests remained in Git due to index state, but any new RTE test was
silently ignored.  The directory-wide test ignore has been removed: all test
source files can now be tracked, while `__pycache__` remains ignored by the
repository-wide rule.

## Research-use classification

Usable as implementation foundations: DF Hamiltonian construction, physical
sectors and matrix-free solver; DF fragment ranking/splitting and circuit
blocks; Givens/diagonal DF evolution; full-H target state, exact-tail
partial-S2 and degeneracy handling; passing H2/H3 validation tests.

Legacy baseline only: external `G_rand`, `B(kappa)`, anchor-Cgs screening,
Pauli partial-randomized PF, fragment-summed analytic cost.

Do not use: the quarantined tracked DF screening JSON, deleted Cgs-table
values, or ignored/local regeneration files as published expectations or
finite-RPE totals.

## 8. Follow-up finite-RTE and DF-tail implementation basis

The follow-up work started from clean commit
`64c4b3afc90aeac76a35908dd969ccb464bf0618`, equal to `origin/main` at the
start of the run. The pre-change suite result was **48 passed, 4 warnings in
2.39 s**. The four warnings remained the existing grouped-UWC `ComplexWarning`
at `chemistry_hamiltonian.py:307`.

The current finite-RTE implementation uses arXiv:2503.05647v2 Appendix A,
Eqs. (A18)–(A40). The repository PDF was identified as v1 and retained only as
legacy reference material; hashes and dates are in `rte_source_versions.md`.

This follow-up adds exact DF central-block I/Z/ZZ extraction and dense H2
validation, but no DF RTE circuit builder, compiled circuit cost, finite-RPE
total, regenerated screening artifact, or research conclusion. The quarantined
screening JSON remains excluded from implementation inputs and test
expectations.

The post-change lightweight suite result is **57 passed, 4 warnings in 3.02 s**
(`3.58 s` wall time measured by `/usr/bin/time`). The same four pre-existing
grouped-UWC `ComplexWarning` instances remain; no new warning class or warning
site was introduced.
