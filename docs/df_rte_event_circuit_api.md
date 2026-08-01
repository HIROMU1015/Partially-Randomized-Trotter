# DF RTE event circuit API

`trotterlib.df_rte_circuit` defines the serializable requests, results, and
builder Protocol. `trotterlib.df_rte_qiskit.QiskitDFRTEEventCircuitBuilder` is
the concrete implementation. It resolves executable local gates from the
`DFBasisRegistry`; request records never need to contain a many-body matrix.

## Preparation and construction

`DFRTEEventPreparation` joins one `SymbolicRTETail`, its ordered component
specifications, and its basis registry. The normal path is:

```text
DF extraction -> symbolic tail -> finite config/distribution
-> classical event sample -> validated request -> Qiskit circuit
```

`request_for_event` creates one `DFRTEEventCircuitRequest`.
`sample_occurrence_request` samples exactly `config.rte_steps` events and
creates a `DFRTEEventSequenceCircuitRequest`. The latter rejects mismatches in
tail ID/hash, RTE lambda, finite Taylor cutoff, dimensionless step time, and
finite-distribution normalization. RTE normalization and attenuation remain
classical correction records; neither is emitted as a gate.

Construction is explicit:

```python
builder = QiskitDFRTEEventCircuitBuilder(
    basis_registry=preparation.basis_registry,
)
event_result = builder.build_event(event_request)
occurrence_result = builder.build_sequence(occurrence_request)
```

Both builders consume `event.application_sequence` verbatim. They do not sort
by component, fragment, basis, probability, or coefficient.

## Component and phase rules

Every non-identity component is emitted in the established DF orientation:

```text
inverse ordered basis operations
-> central diagonal operation
-> forward ordered basis operations
```

For a product occurrence the central operation is the involution itself: Z
emits one Z, while ZZ emits a Z on both support qubits. A negative coefficient
is a scalar phase of pi; coefficient magnitude has already entered event
sampling and is not a rotation angle.

For the final rotation occurrence, the central operation represents
`exp(-i * angle * Z)` or `exp(-i * angle * ZZ)`. Qiskit's half-angle
convention therefore uses `RZ(2*angle)` or `RZZ(2*angle)`. The signed value is
`event.unsigned_rotation_angle`: despite its historical property name, it is
the base angle multiplied by the component coefficient sign.

The Taylor phase (`event.phase`), product-sign phase, signed rotation angle,
and identity phase are accumulated independently. Paired finite-RTE events
currently allow only exact `+1` and `-1` Taylor phases; another complex phase
is rejected. An identity product emits only its sign phase, and an identity
rotation emits only `-event.unsigned_rotation_angle` as phase.

For an uncontrolled circuit these scalars become Qiskit global phase and do
not count as physical gates. For a controlled circuit they are one relative
phase operation on the ancilla and may not be dropped.

## Controlled structure

A controlled result represents `diag(I_system, U_event)` (or the ordered
sequence analogue). Basis operations are deliberately not controlled:

```text
uncontrolled inverse basis
-> controlled central Z/ZZ product or rotation
-> uncontrolled forward basis
```

Thus the basis pair cancels on the ancilla-zero branch. The central product Z
uses CZ; product ZZ uses two CZ gates. Controlled `RZ(2*angle)` and
`RZZ(2*angle)` implement rotations. Identity and all scalar phases act as an
ancilla-relative phase. The ancilla must be outside the system register.

## Adjacent basis reuse and audit metadata

The baseline policy is `raw_adjacent_equal_basis`. A forward/inverse boundary
is removed only when consecutive, non-identity applications have exactly the
same basis ID and canonical basis hash. This works inside one event and across
event boundaries without reordering. An identity closes the active basis, so
the baseline does not optimize through identity applications. A different
basis is always closed and reopened.

Set `cancel_adjacent_equal_bases=False` to use the `disabled` policy. Both
policies are unitary-equivalent. `DFRTEEventCircuitResult` records the actual
component order, controlled flag, event/application counts, naive and emitted
basis-change counts, cancelled pairs, basis switches, global and relative
phase, reuse policy, qubit count, and a canonical event/sequence fingerprint.

Small-system tests compare complete matrices, including global or relative
phase. Synthetic 20/26-qubit tests construct circuits without `Operator`,
statevector, dense DF helpers, full event enumeration, or transpilation.

Compiled-cost APIs and their limits are described in
[`rte_compiled_event_cost.md`](rte_compiled_event_cost.md).
