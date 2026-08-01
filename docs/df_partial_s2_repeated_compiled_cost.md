# Repeated DF partial-S2 circuits and compiled expectations

## Scope and terminology

This Level-5 repeated-step validation path explicitly constructs a short
trajectory of complete partial-S2 steps. Three counts are intentionally
different:

- `rte_steps` is the number of finite-RTE events in one tail occurrence;
- `repetition_count=q` is the number of complete partial-S2 steps in the
  constructed trajectory; and
- `2**m` is only the future mapping from RPE round index `m` to
  `repetition_count`, available through `repetition_count_for_rpe_round`.

`repetition_count` accepts Python/NumPy integers only. Boolean, floating,
string, zero, and negative inputs are rejected. `q=1` reuses the existing
`DFPartialS2StepRequest` semantics and produces the same circuit and compiled
metrics. This work does not implement an RPE estimator or RPE total cost.

## Request, trajectory, and application order

`DFPartialS2RepeatedRequest` stores one preparation, step time, RTE config and
finite distribution, `q` ordered occurrences, control/ancilla condition,
step seeds, master seed, and construction policy. Each occurrence is validated
by the existing one-step request validator.

A trajectory is step-major ordered data:

```text
step 0: event 0, event 1, ...
step 1: event 0, event 1, ...
...
step q-1
```

Qiskit circuit instructions append step 0 through step `q-1`. Consequently,
the dense matrix convention is `U_(q-1) ... U_1 U_0`: the rightmost matrix
acts first. The trajectory fingerprint contains the tail hash, master and
per-step seeds, step order, event order, ordered event fingerprints, and each
occurrence fingerprint. The circuit fingerprint additionally contains
preparation/partition hashes, `q`, time, control/ancilla condition, and
construction/boundary policy.

`make_df_partial_s2_repeated_request` uses the supplied seed directly for the
single step when `q=1`. For `q>1`, `random.Random(master_seed)` generates one
independent step seed per occurrence; each step seed samples exactly
`rte_steps` events. Monte Carlo adds one outer level: the estimator's master
seed generates trajectory seeds, and each trajectory seed generates its step
seeds by the same rule. This is classical trajectory selection, not quantum
shot sampling.

## Raw and boundary-optimized construction

`QiskitDFPartialS2RepeatedCircuitBuilder` supports two paths:

- `raw_concatenation` constructs every existing untranspiled one-step circuit
  and appends them in step order;
- `boundary_optimized` applies only locally proven identities before the
  complete trajectory is transpiled.

At each boundary the previous reverse sweep ends with deterministic block
`D_0(delta/2)`, and the next forward sweep begins with the same recorded block
`D_0(delta/2)`. The optimized builder replaces

```text
B_0^-1 D_0(delta/2) B_0 B_0^-1 D_0(delta/2) B_0
```

with `B_0^-1 D_0(delta) B_0`. This cancels the adjacent equal basis pair and
combines the same generator's two half-time diagonal evolutions. Constant and
extracted-identity phases still occur once per partial-S2 step, but because
they commute they are aggregated into one phase of `q` times the per-step
value. In a controlled circuit this becomes exactly one aggregate relative
ancilla phase. The transpiler remains free to combine adjacent rotations.

The builder does not reorder DF fragments or events, exchange operations
between unequal bases, move anything across an RTE occurrence, or alter
Taylor/sign/rotation/faithful-identity phases inside an event. Small-system
tests compare the raw and optimized unitaries, including controlled
`diag(I,U)`, to numerical tolerance.

## Attenuation and finite Taylor truncation

Attenuation is metadata and is never added to a gate count. For one occurrence
the existing `finite_rte_attenuation(config)` gives `a_step`. Independent
repetition records

```text
log(a_total) = q * log(a_step)
a_total      = exp(log(a_total))
```

The log value is primary. If exponentiation underflows or saturates, the
representable total is `None` and the corresponding flag is set. The record
also binds the RTE config and tail hash.

Taylor truncation reuses `rte_occurrence_truncation_from_config`. Metadata
separates the finite order and one-short-step bound, one-occurrence bound, the
identical randomized bound for one partial-S2 step, and the bound composed
over `q` occurrences. The configured per-short-step tolerance and allocator
origin are recorded. This is only finite-RTE Taylor error; partial-S2
product-formula bias is explicitly not included.

## Full compiled cost and matched diagnostics

Both exact and Monte Carlo APIs compile the whole short trajectory once. The
primary cost follows the requested construction policy. They also report:

```text
A. selected full repeated compiled cost
B. matched sum of the q one-step circuits transpiled separately
C. cross-step difference = A - B
D. sum of forward-half + RTE-occurrence + reverse-half primitive costs
```

For direct policy comparison, the same sampled trajectory also yields raw and
boundary-optimized full costs and `boundary_optimized - raw`. RZ count/depth,
CX count/depth, total depth, and circuit size are retained. One circuit's
measurements remain integers; weighted expectations and Monte Carlo
statistics remain floats. `CircuitCost.fidelity_level` stays 5, while
`circuit_granularity="repeated_partial_s2_steps"`, `repetition_count`,
`estimate_kind`, and boundary policy distinguish this Level 5-R validation
from the existing one-step Level 5 result. Level 6 remains reserved for future
selected long-RPE confirmation.

## Exact enumeration, Monte Carlo, and cache identity

If one step has `M` possible ordered event sequences, `q` repetitions have
`M**q` trajectories. The exact API computes this integer before invoking
`itertools.product`, event enumeration, or circuit construction, and rejects
the job when `maximum_trajectories` is exceeded. Each trajectory weight is the
product of all event probabilities across all steps. The total mass is checked
against one, and standard error is `None`.

The Monte Carlo API samples complete trajectories from the finite
distribution. Its compiled-cost estimate is an unweighted arithmetic mean;
sampled event probability is not multiplied again. It returns the master and
trajectory seeds, sample count, unbiased variance, standard error, min/max,
unique circuit counts, and cache hits.

The transpile cache combines the compiler-independent circuit fingerprint
with all compiler settings and Qiskit version. Repetition count, trajectory
step/event order, seeds, tail identity, control/ancilla condition, construction
policy, and compiler conditions therefore cannot collide. Repeating the same
exact evaluation reuses its entries.

## Short-trajectory reference measurements

The one-qubit finite model used by the tests has two possible one-event step
sequences. With all-to-all logical basis gates `rz,sx,x,cx`, Qiskit
optimization level 1, transpiler seed 17, and the boundary-optimized policy,
the exact uncontrolled results are:

| `q` | trajectories | full RZ | matched-step RZ | cross-step RZ | per-boundary RZ |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.0 | 1.0 | 0.0 | n/a |
| 2 | 4 | 1.0 | 2.0 | -1.0 | -1.0 |
| 3 | 8 | 1.0 | 3.0 | -2.0 | -1.0 |
| 4 | 16 | 1.0 | 4.0 | -3.0 | -1.0 |

Thus this fixture demonstrates why multiplying a one-step compiled cost by
`q` is not the full repeated compiled cost: complete-trajectory transpilation
combines the adjacent rotations. Raw and explicit boundary-optimized compiled
RZ values both become 1.0 in this uncontrolled fixture because the transpiler
finds the same simplification.

For the controlled `q=2` fixture, raw RZ expectation is
`17.66349450104994`, boundary-optimized full RZ is
`14.663494501049938`, matched per-step RZ is `18.66349450104994`, and the
respective boundary and cross-step differences are `-3.0` and `-4.0`. A
100-trajectory Monte Carlo run at master seed 8 gives RZ mean `14.66`,
unbiased variance `0.4488888888888889`, and standard error
`0.0669991708074726`; the exact value lies within that sampling uncertainty.

Compiler dependence is material. For the uncontrolled `q=2` fixture,
optimization level 0 gives expected full RZ/depth/size
`16.32698900209987`, whereas levels 1 and 3 give `1.0` for those metrics.
These numbers are algorithmic regression fixtures, not chemistry resource
estimates or hardware timing predictions.

## Long-repetition exclusions

Only short trajectories should be fully expanded and transpiled. This release
does not extrapolate a boundary proxy and does not construct long `2**m`
circuits. It also excludes Hadamard tests, X/Y measurement, quantum shots,
attenuation-corrected shot counts, state preparation, a finite RPE estimator,
all-round total cost, noise simulation, and backend jobs. Synthetic 20/26
qubit smoke tests stop after dense-free circuit construction; they do not use
`Operator`, statevector, full enumeration, or transpilation.
