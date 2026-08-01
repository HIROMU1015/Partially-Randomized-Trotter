# DF RTE event circuit types and Protocol

`trotterlib.df_rte_circuit` currently contains types and a `Protocol`. It does
not contain an actual circuit builder and must not be reported as one.

## Typed boundary

`DFRTEComponentCircuitSpec` represents a non-identity DF-conjugated Z/ZZ
involution. `DFRTEIdentityCircuitSpec` is a separate type whose uncontrolled
action is global phase and controlled action is relative ancilla phase. Both
carry coefficient magnitude and sign. Non-identity specs also carry fragment
ID, basis ID and canonical basis hash, diagonal support, complete basis-change
operation metadata, and system size.

`DFRTEEventCircuitRequest` and `DFRTEEventSequenceCircuitRequest` validate that
every event application has a matching spec with identical coefficient sign,
magnitude, and identity classification. They reject baseline event reordering.
They also validate the basis ID/hash and diagonal support propagated by the
event.

`DFRTEEventPreparation` is the dense-free handoff bundle. It contains a
`SymbolicRTETail`, the ordered component specs, and a `DFBasisRegistry`. Its
`sample_events` method performs fixed-seed classical event sampling;
`request_for_event` and `sample_requests` resolve every basis ID/hash against
the registry before constructing `DFRTEEventCircuitRequest`. Registry conflicts
or event fingerprint mismatches are rejected rather than guessed.

The future builder boundary is:

```text
DFRTEEventCircuitBuilder.build_event(request) -> DFRTEEventCircuitResult
DFRTEEventCircuitBuilder.build_sequence(request) -> DFRTEEventCircuitResult
```

## Required semantics for a future implementation

For a DF component, the established native builder applies the inverse
basis-operation sequence, the central diagonal operation, and then the forward
sequence. The dense reference in `df_rte_tail` follows this exact circuit
orientation; labels do not assume a different matrix convention.

The future builder receives the preparation bundle. The registry supplies
executable Qiskit operations by basis ID without materializing a many-body
unitary. Equal basis hashes may be reused only when occurrences are already
adjacent in the preserved event order.

The builder must:

1. consume `event.application_sequence` in its existing order;
2. emit $(-1)^{n/2}$, each signed product occurrence, and the final signed
   rotation without conflating product phase with angle reversal;
3. control only the central diagonal action when mathematically valid;
4. convert identity/global phase to relative ancilla phase in a controlled
   event;
5. cancel only adjacent equal-basis inverse/forward pairs;
6. preserve baseline event and step order;
7. build, transpile, and count only—controlled resource validation does not
   run statevector or quantum-shot sampling.

The result type records the component order actually used, safely cancelled
basis pairs, relative ancilla phase, and basis-switch count.

## Implemented DF-to-event-input path

`trotterlib.df_rte_tail` now performs the prerequisite conversion:

$$
\lambda_l\left(\sum_k\eta_{lk}n_k\right)^2
=c_I I+\sum_k c_k Z_k+\sum_{k<j}c_{kj}Z_kZ_j.
$$

The implemented path is:

```text
symbolic DF extraction
-> SymbolicRTETail probabilities
-> finite distribution/config
-> fixed-seed classical event sampling
-> validated DF event circuit request
```

It supports faithful identity-in-tail, extracted deterministic phase, and an
explicit deterministic-only empty randomized tail. The normal path is
dense-free. Guarded dense references are used only as an 8-qubit-or-smaller
test oracle. Synthetic 20/26-qubit integration tests are symbolic scalability
checks, not H10/H13 chemistry or statevector validation.

Actual Qiskit event construction, primitive transpilation, compiled event cost,
partial-S2 circuits, and RPE circuits remain the next milestones.
