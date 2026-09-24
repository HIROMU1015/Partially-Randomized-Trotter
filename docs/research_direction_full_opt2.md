# M06-F all-r coherent optimization-level-2 reoptimization

## Status

This is the preregistered compute protocol and current local status for the
WP11-selected follow-up. The 36-cell initial compute and the preregistered
15-cell fresh-32 extension completed on 2026-09-25. The integrated 51/51 audit
passes both proxy gates, and the coherent optimization-level-2 reoptimization
is complete. The evidence remains local dirty-worktree evidence rather than
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

## Completion audit

The immutable-file audit runner
`scripts/run_research_direction_full_opt2_completion.py` reloaded all 36 task
specifications, worker results, checkpoints, and the deterministic aggregate.
It found no failed, missing, partial, duplicate, corrupt, or mismatched initial
task. The tmux session and matching parent/worker processes were absent. The
completed runner state implies exit code 0, although the original shell `$?`
was not saved independently. Runtime was 17,475.324 seconds (4:51:15.324).

The audit also verified the `q=1,2` calibration / fixed holdout partition,
the single optimization-level-2 compiler context, seeds, dependency versions,
input/source hashes, compute commit and dirty state, one-thread worker settings,
and CPU-only execution. All 111 compute JSON files loaded successfully and the
rebuilt aggregate retained fingerprint
`ae0e0d9b616da5c31cbdd09d27d2b6e103cc07de05f0b400ab04f51da8f0c63a`.

The audit artifact is
`wp11_all_r_opt2_completion_audit_20260924_230342.json`, fingerprint
`7d184c4ebde0665fbc85452c69b9de14997f101fb51f9e2969fd13bf1d5ebf35`.
It distinguishes the complete initial batch (36/36) from the full
preregistered workflow (36/51): the 15 fresh-32 extension tasks have no output.
Consequently beta/alpha/shot/schedule reoptimization, `L_D=3/12`, delta,
mixed-versus-coherent, and updated preparation break-even comparisons remain
blocked rather than being estimated from an under-precision sample.

The completion-audit tests passed 3/3 and the related M06-F/executor/full-scope/
compiler-transfer/decision-cost selection passed 33/33. The local full suite
reported 562 passed, 2 failed, and 5 warnings. The two failures are outside the
M06-F change surface: a historical compiled-cost reference records Python
3.11.0rc1 while this environment is Python 3.12.3, and a saved four-round
artifact has a DF-preparation hash mismatch. They remain explicit residual
repository test gaps rather than being rewritten by this audit.

## Fresh-32 completion and integrated audit

The preregistered extension ran only the five failing groups at `q=1,2,8`.
It completed 15/15 tasks and 1,920 direct transpiles with zero failed, missing,
partial, duplicate, or seed-overlap cases. The independent wrapper exit code
was zero and runtime was 13,833 seconds (3:50:33). All workers used the same
Qiskit 1.3.0 optimization-level-2 compiler context, one thread per numerical
library, `CUDA_VISIBLE_DEVICES=`, and no GPU assignment. Both initial and
extension aggregates rebuild exactly from their checkpoints.

Across all twelve randomized cells, the maximum direct-RZ relative standard
error is 1.9844%, the maximum selected-policy RZ holdout error is 4.4898%, and
the maximum selected-policy all-metric holdout error is 4.7488%. Thus both the
2% precision gate and 5% holdout gate pass. The full-basis RZ diagnostic reaches
5.1359% at `delta=0.02,r=32`, but it is not the preregistered selected-policy
gate.

The final integrated audit is
`wp11_all_r_opt2_fresh32_audit_20260925_065827.json`, content fingerprint
`b39960a630746e2c05009f8d7e13bd982ff565b3a3c70a7abfc2dc65dc7009ca`.
It records 51/51/0 expected/completed/failed tasks and preserves the original
eight-trajectory evidence separately from the fresh replacement cells.

## Direct compiled-RZ measurements

The table gives selected-policy direct compiled-RZ mean +/- standard error.
Cosine and sine values are identical in these runs; both axes and the full-basis
measurements remain explicit in the machine-readable artifact.

| delta | r | q=1 | q=2 | q=8 |
|---:|---:|---:|---:|---:|
| 0.01 | 1 | 4,792.125 +/- 9.965 | 9,369.375 +/- 10.490 | 36,690.500 +/- 25.198 |
| 0.01 | 2 | 4,909.125 +/- 3.388 | 9,601.500 +/- 9.588 | 37,695.500 +/- 22.585 |
| 0.01 | 4 | 5,076.000 +/- 44.592 | 9,841.750 +/- 63.445 | 38,712.250 +/- 114.873 |
| 0.01 | 8 | 5,429.750 +/- 51.432 | 10,403.875 +/- 120.199 | 41,126.375 +/- 216.189 |
| 0.01 | 16 | 5,799.281 +/- 84.815 | 11,412.250 +/- 105.983 | 45,462.438 +/- 202.799 |
| 0.01 | 32 | 6,799.094 +/- 85.333 | 13,778.594 +/- 163.373 | 54,360.375 +/- 275.137 |
| 0.02 | 1 | 4,799.875 +/- 6.548 | 9,367.125 +/- 10.429 | 36,690.125 +/- 23.165 |
| 0.02 | 2 | 4,915.625 +/- 12.080 | 9,584.625 +/- 10.173 | 37,673.875 +/- 10.453 |
| 0.02 | 4 | 5,044.875 +/- 47.172 | 9,905.500 +/- 42.689 | 38,617.750 +/- 118.579 |
| 0.02 | 8 | 5,267.375 +/- 34.728 | 10,401.281 +/- 53.711 | 41,114.781 +/- 120.619 |
| 0.02 | 16 | 5,904.281 +/- 68.894 | 11,650.031 +/- 112.471 | 45,389.969 +/- 212.652 |
| 0.02 | 32 | 6,742.281 +/- 87.696 | 13,792.125 +/- 144.457 | 53,681.031 +/- 285.042 |

The fixed `q=8` holdout is separate from the `q=1,2` affine calibration.
No direct `q>32` validation is inferred from this table.

## Coherent opt2 reoptimization

All four candidate grids use one compiler context and jointly reoptimize beta,
cost-sensitive alpha, integer shots, and round schedules:

| candidate | compiled-RZ point estimate | shots | beta PF / RTE / stat |
|---|---:|---:|---:|
| `L_D=3, delta=0.01` | 1.568700e12 | 12,868 | 0.00351250 / 2.4721e-8 / 0.396487 |
| `L_D=3, delta=0.02` | 1.263314e12 | 13,588 | 0.01405001 / 9.8996e-8 / 0.385950 |
| `L_D=12, delta=0.01` | 2.492378e12 | 11,548 | 0.00351946 / 0 / 0.396481 |
| `L_D=12, delta=0.02` | 1.327822e12 | 11,162 | 0.01407783 / 0 / 0.385922 |

Both candidates select `delta=0.02`. The selected `L_D=3` rounds use
`r=[1x8,2x2,4x1,8x2,16x2,32x3]`; per-axis shots range from 261 to 510 and
alpha from 2.8898e-8 to 0.0169571. The deterministic endpoint uses `r=0` in
all 18 rounds, with per-axis shots 144--476 and alpha
9.6813e-8--0.0125000. Complete round rows are stored in the artifact.

At zero preparation cost, the point ratio is 0.951418, so `L_D=3` is 4.858%
lower. The local-5% intervals are `[1.1416,1.3850]e12` and
`[1.2614,1.3942]e12`; the per-r selected-discrepancy and 25% transfer
intervals also overlap. This is a point preference, not interval superiority.

The earlier mixed-context focused estimate gave `L_D=3` 1.215990e12 and an
8.422% point advantage. Coherent reoptimization raises the `L_D=3` estimate
by 4.73239e10, or 3.8918%, and reduces the point advantage by 3.564 percentage
points. The missing-all-r compiler-context asymmetry is therefore resolved for
this H4 measurement domain, while compiler/backend transfer is not.

The coherent result uses 13,588 versus 11,162 shots. With a common omitted
preparation cost `P`, the point break-even is 26,590,335 compiled-RZ
equivalents per shot. Every propagated interval already overlaps at `P=0`,
so no nonnegative common `P` establishes robust interval superiority.
Preparation circuits were not measured, and this calculation must not be
reported as a final total-cost evaluation.

The final coherent artifact is
`wp11_all_r_opt2_coherent_analysis_20260925_065827.json`, content fingerprint
`5ce368a94daa39680b4edc0cfb59b30168d8bc159ad2538b29cb67b928e3cdba`.

## Research routing

WP11's selected discriminator is now complete. Because the coherent intervals
still overlap, further local compiler precision is stopped. T1 remains scoped
to H4 conditional limits; T2, T5, and T6 remain limited; T3 remains held; T4
and T7 remain primary, now emphasizing external transfer and auditable negative
results. The external-instance pilot is reopened as the next discriminator
rather than inferred from H4. A long-q holdout remains conditional because
`q>32`, state preparation, coupling/backend effects, H12 transfer, and
external reproduction are still unresolved.

## Final validation

The extension-analysis tests pass 3/3. The M06-F, bounded-executor, full-scope,
compiler-transfer, and decision-cost related selection passes 36/36. The full
local suite reports 565 passed, 2 failed, and 4 warnings. Both failures predate
and are outside this change: the historical level-5R compiled-cost reference
records Python 3.11.0rc1 while this environment is Python 3.12.3, and the saved
four-round artifact has a DF-preparation hash mismatch. Manifest validation and
Python syntax checks pass. Black and isort are not installed in the local
environment, so their standalone checks were unavailable.
