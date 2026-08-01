# Compiled DF-RTE event cost

## Scope

This module measures the cost of actually constructed and transpiled finite
DF-RTE circuits at two granularities:

1. a single event expectation; and
2. a short occurrence containing exactly `config.rte_steps` events.

It does not build partial-S2, state preparation, a Hadamard test, or an RPE
round. Classical RTE event sampling is the only sampling performed. There are
no quantum shots, statevector/noise simulation, or backend jobs.

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

`CircuitCost` retains floating fields because a distribution expectation or a
sample mean need not be integral. No RZ-to-T synthesis estimate is applied.

## Exact and Monte Carlo event expectations

`estimate_exact_compiled_event_cost` uses `enumerate_rte_events`, checks that
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
`maximum_rte_steps` and `maximum_untranspiled_circuit_size` prevent accidental
large sequence expansion.

## Cache identity

`TranspiledCircuitCostCache` combines a semantic event/sequence fingerprint
with the compiler-settings hash. The semantic fingerprint includes the tail,
ordered applications, Taylor order and signed angle, controlled/ancilla
condition, and basis-reuse policy. A setting change, controlled change, event
order change, or reuse-policy change cannot reuse an incompatible result.

## Fidelity levels

The current research-cost ladder is:

- Level 0: legacy analytic proxy or paper upper bound.
- Level 1: symbolic primitive sum.
- Level 2: constructed but untranspiled event circuit.
- Level 3: transpiled single-event expectation.
- Level 4: transpiled short event-sequence expectation, including the paired
  additive comparison.
- Level 5: future partial-S2 or short RPE repetition.
- Level 6: future long-RPE confirmation for selected candidates.

Levels 3 and 4 still exclude state preparation, attenuation-driven shot
counts, quantum measurement shots, partial-S2 cost, and total RPE cost.
