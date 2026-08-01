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

## 9. Symbolic DF-tail and composed truncation follow-up

This follow-up started from clean `main` commit
`231d17e393f573fd29c659eaf1fd9bae2a68d8f9`, equal to `origin/main`. Its
pre-change suite result was **57 passed, 4 warnings in 3.10 s** (`3.66 s` wall
time).

Normal DF-tail extraction now stores symbolic I/Z/ZZ components and an
executable local-operation basis registry without constructing many-body dense
unitaries or component matrices. Synthetic 20- and 26-qubit blocks verify the
quadratic component path and dense-call exclusion. These are H10/H13-sized
symbolic checks, not chemistry, statevector, or circuit results. Guarded dense
references remain available only below an explicit qubit limit for small-system
validation.

Finite Taylor accounting now distinguishes short-step, occurrence, and
heterogeneous RPE-round residual bounds. A baseline equal-log-budget allocator
chooses minimal even cutoffs and rechecks the composed round bound. This is not
a product-formula error model, attenuation/shot model, event circuit builder,
transpilation result, or RPE total-cost result.

The post-change suite result is **67 passed, 4 warnings in 2.95 s** (`3.51 s`
wall time). The four warnings are the same pre-existing grouped-UWC
`ComplexWarning` instances at `chemistry_hamiltonian.py:307`. The quarantined
screening JSON was not used as input, expectation, or research evidence.

## 10. Dense-free symbolic tail to event-input connection

This follow-up started from clean `main` commit
`f5f4782d5f70420aefd12c9fb63c1263fbcc2aed`, equal to `origin/main`. The
pre-change suite result was **67 passed, 4 warnings in 3.29 s** (`3.84 s` wall
time). The validation manifest passed before implementation.

The DF normal path now converts an extraction into normalized symbolic
`RTEComponent` probabilities, creates a finite config/distribution, classically
samples a small fixed-seed event set, and validates future-builder requests
against executable registry definitions. No dense operator is required.
Deterministic-only empty randomized tails are represented explicitly.

Count values now require Python or NumPy integers; bool, float, string, and
negative inputs are not silently truncated. Registry operations require a
stable matrix fingerprint from a supported at-most-four-qubit local operation;
opaque or wider operations are rejected.

Synthetic 20/26-qubit tests cover only the symbolic integration path. Small
systems compare symbolic results with guarded dense references. No event
circuit builder, transpilation, H10/H13 chemistry/statevector, GPU run, quantum
shots, screening regeneration, or quarantined artifact use is part of this
follow-up.

The post-change suite result is **110 passed, 4 warnings in 2.63 s** (`3.20 s`
wall time). The four warnings remain the pre-existing grouped-UWC
`ComplexWarning` instances at `chemistry_hamiltonian.py:307`. The validation
manifest and `git diff --check` both passed.

## 11. Qiskit DF-RTE event circuits and compiled expectations

This follow-up started from clean `main` commit
`949b4a666c52f6ef2eec35d5c432b69abd7f8a52`, equal to `origin/main`. The
pre-change suite result was **110 passed, 4 warnings in 2.68 s** (`3.24 s`
wall time). The validation manifest passed, and the quarantined screening JSON
was not used.

`QiskitDFRTEEventCircuitBuilder` now constructs ordered single events and
short occurrences. It distinguishes product Z/ZZ involutions, signed Z/ZZ
rotations, identity phase, product sign, and paired-Taylor phase. Controlled
circuits implement the complete `diag(I, U)` convention: local basis changes
remain uncontrolled, only the central action is controlled, and scalar phase
becomes ancilla-relative. Raw adjacent equal-basis pairs may be reused within
or across event boundaries; identities are conservative reuse barriers.

Small-system tests cover all 68 events of a two-qubit faithful-identity case
at finite Taylor cutoff 2 and compare the complete Qiskit matrix with the
dense event oracle. Controlled representatives include identity, order-2
product phase, and ZZ rotation, without discarding relative phase. Ordered
multi-event circuits and reuse-on/off circuits also match the dense reference.
Synthetic 20/26-qubit tests stop at circuit construction and prohibit
many-body `Operator`, statevector, dense DF helpers, full enumeration, and
transpilation.

Compiled cost records now separate integer per-circuit measurements from
floating expectations. Exact enumeration probability-weights every compiled
event once. Monte Carlo reports the unweighted mean, unbiased variance, and
standard error, with semantic-circuit/compiler caching. A representative
controlled one-component case at dimensionless time 1.2 has exact expected RZ
count `5.331747250524967`; 100 classical samples at seed 8 give `5.41` with
standard error `0.04943110704237104`. These values are test fixtures for the
algorithm, not chemistry resource results.

For a representative three-event occurrence (12 classical sequence samples,
seed 3), the mean complete-sequence RZ count is `3.0`, the mean sum of
individually transpiled event counts is `9.0`, and the recorded nonadditive
difference is `-6.0`. This difference includes adjacent basis reuse and
transpiler cancellation. It is not a partial-S2 or RPE total.

The post-change lightweight suite result is **128 passed, 4 warnings in 3.80
s** (`4.35 s` wall time). The warnings are still the same four pre-existing
grouped-UWC `ComplexWarning` instances at `chemistry_hamiltonian.py:307`; no
new warning was introduced. The validation manifest and `git diff --check`
passed. No commit or push was performed during that implementation run, so at
that run's endpoint `HEAD` and `origin/main` remained at the starting SHA. The
completed changes were subsequently committed and pushed as `2c01920`; this
historical statement must not be read as the repository's current state.

## 12. Compiled DF partial-S2 one-step follow-up

This follow-up started from clean `main` commit
`2c01920ece9c598d44c6657f430eff68316dcb64`, equal to `origin/main`. The
pre-change suite result was **128 passed, 4 warnings in 3.32 s** (`3.89 s`
wall time), and the validation manifest passed.

`split_df_hamiltonian_by_ld` now applies strict integer validation and retains
`lambda_r` only as a compatibility ranking proxy, exposed explicitly as
`ranking_proxy_lambda_r`. `prepare_df_partial_s2` checks that the ranked prefix
and suffix are a disjoint complete fragment partition, keeps constant and
one-body corrections deterministic, preserves ranked-prefix order, and
extracts the suffix into an independent `exact_rte_lambda_r`. A full
deterministic prefix produces a valid empty randomized tail.

The one-step builder explicitly applies deterministic blocks forward at half
time, one existing RTE occurrence at full step time, and deterministic blocks
in reverse at half time. It does not place a complete internal `H_D` S2 on
both sides. One-body and squared-fragment central evolutions now share one
global-phase/RZ/RZZ primitive representation. Controlled steps leave all
basis operations uncontrolled and control only those central primitives.
Constant correction, extracted tail identity, and RTE-internal phases remain
separate and are each applied exactly once under their stated policy.

For the two-qubit reference, uncontrolled `L_D=0`, intermediate, and full
deterministic steps differed from independently composed dense references by
approximately `6.8e-16` to `1.4e-15` in matrix norm. Controlled `diag(I,U)`
comparisons, including ancilla-relative phase, differed by approximately
`5.9e-16` to `1.1e-15`. The deterministic-only limit also matches the existing
second-order DF circuit builder.

Level-5 exact and Monte Carlo estimators compile the complete one-step circuit
and matching forward/RTE/reverse additive parts. In a one-qubit controlled
test with ranking proxy `0.7`, exact RTE lambda `0.35`, two RTE events, and
finite cutoff 2, the four exact event sequences give expected full-step RZ
count `11.443382024588187`. One hundred classical samples at seed 8 give mean
`11.5`, unbiased variance `0.25252525252525254`, and standard error
`0.050251890762960605`. In the matching uncontrolled case, full-step RZ is
`1.0`, additive RZ is `3.0`, and the partial-S2 nonadditive difference is
`-2.0`. These are algorithmic fixtures, not chemistry resource estimates.

The exact test uses four unique full-step circuits and ten unique full/part
cache entries; a repeated evaluation produces 16 cache hits without new
transpilation. The 100-sample run uses four unique full-step circuits, ten
unique full/part entries, and 390 hits. Cache identity separates partition,
time, event order, control/ancilla, reuse, and compiler conditions.

Synthetic 20/26-qubit tests stop after partition, preparation, one-event
sampling, and partial-S2 circuit construction. They prohibit many-body
`Operator`, statevector, dense DF helpers, complete enumeration, transpilation,
and chemistry generation. The quarantined screening JSON remains unused.

The post-change suite result is **152 passed, 4 warnings in 5.05 s** (`5.63 s`
wall time). All warnings are the same pre-existing grouped-UWC `ComplexWarning`
instances at `chemistry_hamiltonian.py:307`. The validation manifest and
`git diff --check` passed. No commit or push was performed during this
follow-up; the changes remain in the working tree as requested.
