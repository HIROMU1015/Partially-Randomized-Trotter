# Short-round RPE Hadamard interrogation wrapper

`trotterlib.rpe_hadamard_interrogation` wraps one already-constructed,
ordinary-controlled `DFPartialS2RepeatedCircuitResult`. It constructs one
cosine/X or sine/Y Hadamard interrogation for the same recorded RTE
trajectory.

This is a `single_hadamard_interrogation_without_state_preparation`, not a
full RPE circuit. The wrapper does not prepare the system state, execute a
backend or quantum shots, reconstruct a phase, optimize resource allocations,
or build a long-round proxy. It supports only `q_m=1,2,4`, corresponding to
`m=0,1,2`.

## Signal and gate convention

The wrapped signal is

```text
Z_m = <psi|U_m|psi>.
```

The measured bit is converted by `x_b=(-1)^b`, so bit 0 maps to `+1` and bit
1 maps to `-1`. The ancilla is immediately after the system register.

The cosine circuit appends

```text
H -- controlled U_m -- H -- Z measurement,
```

and therefore has `E[x_b]=Re(Z_m)`. The sine circuit appends

```text
H -- controlled U_m -- S-dagger -- H -- Z measurement,
```

and has `E[x_b]=Im(Z_m)`. In particular, for
`U_m=exp(-i H t_m)` and an energy eigenstate,

```text
Im(Z_m) = -sin(E*t_m).
```

The wrapper does not add a sign flip. This convention is directly compatible
with `atan2(Im(Z_m), Re(Z_m))`.

`include_measurement=False` leaves the circuit unitary for exact
`Statevector` or `Operator` validation. `include_measurement=True` adds one
classical register and measures only the existing control ancilla.

## Reuse and provenance

The builder composes the existing controlled evolution circuit as-is. It does
not call `.control()` and does not resample the RTE trajectory. The result
retains the wrapped trajectory, provenance, and circuit-semantics
fingerprints. Its wrapper fingerprint additionally binds:

- wrapper schema version and circuit scope;
- cosine/sine axis and measurement inclusion;
- bit-to-value and real/imaginary estimator conventions;
- ancilla position and `(m,q_m,t_m,delta_time)`;
- wrapped trajectory, provenance, and circuit-semantics fingerprints.

The audit-oriented `wrapper_fingerprint` deliberately changes when trajectory
provenance changes, even if the constructed circuit does not. The separate
`wrapper_circuit_semantics_fingerprint` excludes trajectory seeds and
provenance, and instead binds the wrapped circuit-semantics fingerprint plus
the wrapper gates, axis, measurement choice, ancilla, timing, version, and
scope. `compiler_independent_fingerprint` uses this circuit-semantics value so
identical deterministic circuits are not distinguished only by their seeds.

Constructing one wrapper does not verify
`fresh_iid_rte_trajectory_per_hadamard_shot`. A future multi-shot or batch
builder must create and audit a fresh independent trajectory for every
Hadamard shot before using the current Hoeffding certification. Reusing the
same randomized trajectory for multiple quantum shots is outside that
guarantee and requires a separate circuit-randomness concentration analysis.

The existing `DFLevel5RCompiledCostProvider` and resource-accounting objective
remain unchanged. A wrapper-inclusive compiled-cost provider and short-`q`
full-interrogation validation are separate next steps.
