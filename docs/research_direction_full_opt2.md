# M06-F all-r coherent optimization-level-2 reoptimization

## Status

This is the preregistered compute protocol and current local status for the
WP11-selected follow-up. The 36-cell initial compute completed on 2026-09-24,
but five `(delta,r)` groups require the preregistered fresh 32-trajectory
extension. The evidence remains local dirty-worktree evidence rather than
immutable CI or a final total-cost evaluation.

## Fixed scientific scope

- H4 linear chain, 1.0 Angstrom, STO-3G, 8 qubits, DF rank 12
- fixed Hamiltonian snapshot SHA-256
  `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`
- candidates `L_D=3,12`, `delta=0.01,0.02`, finite Taylor order `K=2`
- complete controlled partial-S2 / cosine and sine Hadamard wrappers
- fixed `support_run_le_1` policy for randomized `L_D=3`
- Qiskit 1.3.0, basis `rz,sx,x,cx`, optimization level 2, transpiler seed 17
- no backend, coupling map, layout method, or routing method
- primary metric compiled RZ count; optimization-level-1 and level-2 intervals
  are never combined

## Initial compute

The new randomized cells are:

- `delta=0.01`, `r=1,2,4,8,16,32`, `q=1,2,8`
- `delta=0.02`, `r=1,2,4,8,16`, `q=1,2,8`

Each cell uses one canonical eight-trajectory stream and records every
trajectory seed, both axes, both full and selected policies, all six compiled
metrics, aggregate mean/SE/sample count, compiler settings, input/source hashes,
and elapsed time. The `q=1,2` cells are calibration and `q=8` is the fixed unused
holdout. Seeds reproduce the corresponding WP05-a/WP05-b physical trajectories,
so the compiler context is the changed factor.

The existing `delta=0.02,r=32,q=1,2,16,32` opt2 artifact is reused only after
its fingerprint, input hashes, snapshot, policy, and compiler settings pass.
The missing deterministic `L_D=12,delta=0.01,q=1,2,8` provider is compiled in
the same context. This gives 36 checkpointed cell tasks and 1,062 direct
transpiles in the initial batch. No `q>32` circuit is added.

## Execution safety

`scripts/run_research_direction_full_opt2_compute.py` uses the bounded parallel
executor. Every child receives:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
NUMBA_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=
```

Task specifications, per-attempt logs, worker results, atomic checkpoints,
batch status, deterministic aggregate, and run provenance are kept in a new
dated output directory. Resume skips completed checkpoints and never replaces
an existing manifest or analysis artifact.

## Acceptance and analysis

For every selected `(delta,r)` proxy:

- selected-policy RZ holdout error must be at most 5%;
- selected-policy error for every compiled metric must be at most 5%;
- direct RZ relative standard error must be at most 2%.

A failing `(delta,r)` produces a separate fresh 32-trajectory extension
manifest for its `q=1,2,8` cells; the original failed checkpoints remain.
Only a complete passing compute proceeds to coherent optimization of beta,
alpha, integer shots, and round schedules for both candidates and both deltas.
Sampling uncertainty and per-r measured discrepancy are retained separately,
and the earlier mixed opt1/opt2 focused result is stored as a separate reference.

The analysis also recomputes the common state-preparation break-even. It does
not claim scientific superiority, final total cost, H12 transfer, backend/noise
validity, or direct `q>32` validity.

## Initial result

All 36 initial tasks completed without failure or interruption. The batch
performed 1,062 direct transpiles and produced aggregate fingerprint
`ae0e0d9b616da5c31cbdd09d27d2b6e103cc07de05f0b400ab04f51da8f0c63a`.
Seven of the twelve randomized `(delta,r)` groups passed every preregistered
check. The selected-policy holdout errors remained below 5% in every group,
but the maximum direct RZ relative standard error exceeded 2% for:

- `delta=0.01, r=16`: 3.911%
- `delta=0.01, r=32`: 2.416%
- `delta=0.02, r=8`: 2.735%
- `delta=0.02, r=16`: 2.543%
- `delta=0.02, r=32`: 2.141%

The analysis fingerprint is
`b256b47a83fb54657716d4d7f910a8772aa0f583c54ec9a63259f37784d0bcaf`.
It has status `requires_fresh_32_trajectory_extension`; therefore coherent
reoptimization and the `L_D=3/12` comparison have not run. The generated
extension manifest contains only the five failing groups at `q=1,2,8`: 15
tasks and 1,920 direct transpiles, fingerprint
`c150a17c92ade7bd5257a8b99a92bdfdae688fb467000500a9e92754ac23e997`.
