# Short-round RPE Hadamard compiled cost

`trotterlib.df_rpe_hadamard_compiled_cost` evaluates the compiled cost of a
complete, single short-round Hadamard interrogation for `q_m=1,2,4`. For each
repeated partial-S2 trajectory, it constructs and transpiles two independent
circuits:

```text
cosine: H -- controlled U_m -- H -- ancilla measurement
sine:   H -- controlled U_m -- S-dagger -- H -- ancilla measurement
```

The transpiler receives each complete wrapper. The implementation does not add
an analytic H/S-dagger/measurement cost to an evolution-only transpilation.
State preparation, backend execution, quantum shots, noise, phase recovery,
full multi-round RPE, and long-circuit proxies remain outside the scope
`single_hadamard_interrogation_without_state_preparation`.

## Exact and Monte Carlo evaluation

`estimate_exact_compiled_rpe_hadamard_cost` consumes the existing canonical
exact trajectory stream once. Each trajectory probability weights both axis
costs, and exact statistics are accumulated in a canonical circuit-semantics
order. A trajectory-space or workload limit is checked before circuit
construction. A failed build or transpilation raises; a partial exact average
is never returned as complete.

`estimate_monte_carlo_compiled_rpe_hadamard_cost` uses the existing
master/trajectory/step seed hierarchy. One sampled evolution result is wrapped
on both axes, so cosine and sine always use the same classical trajectory
sample. Each axis retains its sample mean, unbiased sample variance, and
standard error. The classical sample count `S_MC` estimates expected compiled
cost; it is not a quantum Hadamard-shot count.

Every successful result records `complete=True`, measurement inclusion,
state-preparation exclusion, retained paired trajectory records, provenance
digests, wrapper circuit-semantics digests, compiler/backend context, workload
counts, and axis-specific compiled-cost evaluation fingerprints. Provenance
and circuit semantics remain separate: unused seeds may change audit
provenance without changing a wrapper's compiler-independent fingerprint.

## Resource-accounting adapter

`DFRPEHadamardCompiledCostProvider` implements the existing
`RPERoundCostProvider` boundary. It supplies separate cosine and sine expected
costs, after which the unchanged accounting formula applies:

```text
G_m = N_m,c * E[C_m,c] + N_m,s * E[C_m,s].
```

`S_MC` is metadata and is not multiplied into `G_m`. The provider does not
verify fresh-IID RTE trajectories per quantum shot; certification continues to
depend on the independently supplied `RPEHadamardSamplingPolicy`.

`DFLevel5RCompiledCostProvider` remains available with its original
`compiled_time_evolution_subcircuit` scope and continues to exclude Hadamard
gates and measurement.
