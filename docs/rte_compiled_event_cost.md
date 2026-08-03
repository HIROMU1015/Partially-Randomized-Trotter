# Compiled DF-RTE event cost

## Scope

The finite-circuit cost path measures actually constructed and transpiled
DF-RTE circuits at three granularities:

1. a single event expectation; and
2. a short occurrence containing exactly `config.rte_steps` events.
3. a complete single partial-S2 step containing one such occurrence; and
4. a short trajectory containing multiple complete partial-S2 steps.

The third API is implemented in `df_partial_s2_cost.py` and documented in
`df_partial_s2_compiled_cost.md`. The fourth is implemented in
`df_partial_s2_repeated_cost.py` and documented in
`df_partial_s2_repeated_compiled_cost.md`. It fully transpiles only short
repetitions; neither path builds state preparation, a Hadamard test, or an RPE
round.
Classical RTE event sampling is the only sampling performed. There are no
quantum shots, statevector/noise simulation, or backend jobs.

## Compiler record and per-circuit cost

`CompilerSettings` fixes basis gates, optional backend name, optional coupling
map, optimization level, layout/routing methods, transpiler seed, and Qiskit
version. With no backend and no coupling map the result is explicitly an
all-to-all logical cost. A compiler-settings hash is part of every cache key;
costs from different settings are not one comparable series.

`transpile_and_measure_cost` passes those settings to Qiskit and returns a
`TranspiledCircuitCost`. This is one actual transpiled circuit, so RZ/CX
counts, filtered RZ/CX depths, total depth, size, and qubit count are Python
integers. It also stores pre/post gate dictionaries, global phase, compiler
settings/hash, and circuit fingerprint. RZ and CX depths use independent gate
filters and are not copies of total depth.

The cache is a bounded LRU and stores metric records rather than transpiled
circuit bodies by default. A miss still returns the transpiled circuit so a
small number of selected diagnostics can retain it explicitly.

`CircuitCost` retains floating fields because a distribution expectation or a
sample mean need not be integral. No RZ-to-T synthesis estimate is applied.

## Exact and Monte Carlo event expectations

`estimate_exact_compiled_event_cost` uses `iter_rte_events`, checks that
event probabilities sum to one, transpiles each canonical event circuit, and
computes each expected metric exactly within the fixed finite model:

```text
sum(event_probability * actual_transpiled_metric)
```

It records `exact_compiled_expectation`, the number of enumerated events, and
no standard error. `max_events` bounds enumeration.

`estimate_monte_carlo_compiled_event_cost` samples events classically from the
finite distribution. It reports the unweighted sample mean, unbiased sample
variance, standard error, sample count, seed, minimum, maximum, unique circuit
count, and cache hits for every metric. Event probability must not be
multiplied again because it was already used to draw each event:

```text
sum(actual_transpiled_sample_metric) / sample_count
```

This distinction is also reflected by `estimate_kind`. An exact compiled
expectation is exact only for the recorded tail, finite Taylor distribution,
controlled/reuse condition, Qiskit version, and compiler settings. It is not
an exact hardware duration or noisy cost.

## Short occurrence cost and nonadditivity

`estimate_compiled_occurrence_cost` samples complete short occurrences. For
the same occurrence sample it records both:

```text
additive = sum(cost of each event transpiled separately)
sequence = cost of the complete ordered sequence transpiled once
difference = sequence - additive
```

The difference captures cross-event basis reuse, compiler cancellations,
phase combination, and possible routing nonadditivity. Means, unbiased sample
variances, and standard errors are retained separately for all three series.
`maximum_rte_steps`, `maximum_samples`, and
`maximum_untranspiled_circuit_size` prevent accidental large expansion. A
side-effect-free instruction-count planner rejects oversized work before
Qiskit circuit allocation; a post-build guard remains. Means and Monte Carlo
variances use online Welford accumulation.

## Streaming and total-work preflight

`iter_rte_events()` and `iter_sample_rte_events()` are the compiled-cost
streaming primitives. The existing tuple-returning `enumerate_rte_events()` and
`sample_rte_events()` remain compatibility APIs for small validation jobs.
Single-event exact and Monte Carlo costing consume each event once; no complete
event sample is retained. A short occurrence retains only its current
`rte_steps` event tuple because that tuple is the circuit request itself.

Let `N` be the event/sample count, `r` the occurrence event count, `u_event` a
structural upper bound for one event circuit, and `u_seq` the bound for one
occurrence circuit. Preflight counts are independent of cache hits:

```text
single-event:     builds = cache requests = N
                  instructions <= N * u_event
short occurrence: builds = cache requests = N * (1 + r)
                  instructions <= N * (u_seq + r * u_event)
```

`maximum_build_requests`, `maximum_transpile_requests`, and
`maximum_planned_instruction_applications` are checked before builder creation.
The result records the plan, actual cache requests/misses, actual built
instruction total, budget, and workload-policy version. Existing per-circuit
post-build size guards remain active.

## Cache identity

`TranspiledCircuitCostCache` keys the recursively serialized, fully numeric
actual Qiskit circuit together with compiler settings and canonical backend
target data. Custom definitions, controls, conditions, global phase,
qubit/clbit placement, and nested control-flow blocks are included. Symbolic
or unsupported circuits and non-canonical backends bypass the cache. Backend
context is validated before lookup. Caller fingerprints remain provenance and
do not prevent safe deduplication of identical actual circuits.

## Fidelity levels

The current research-cost ladder is:

- Level 0: legacy analytic proxy or paper upper bound.
- Level 1: symbolic primitive sum.
- Level 2: constructed but untranspiled event circuit.
- Level 3: transpiled single-event expectation.
- Level 4: transpiled short event-sequence expectation, including the paired
  additive comparison.
- Level 5: compiled expectation for one complete partial-S2 step, plus short
  repeated-step validation distinguished as Level 5-R metadata.
- Level 6: future long-RPE confirmation for selected candidates.

Levels 3 through 5 still exclude state preparation, attenuation-driven shot
counts, quantum measurement shots, long `2**m` repetition, and total RPE cost.
