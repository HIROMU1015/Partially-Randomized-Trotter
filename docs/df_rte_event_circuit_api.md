# DF RTE event circuit types and Protocol

`trotterlib.df_rte_circuit` currently contains types and a `Protocol`. It does
not contain an actual circuit builder and must not be reported as one.

## Typed boundary

`DFRTEComponentCircuitSpec` represents a non-identity DF-conjugated Z/ZZ
involution. `DFRTEIdentityCircuitSpec` is a separate type whose uncontrolled
action is global phase and controlled action is relative ancilla phase. Both
carry coefficient magnitude and sign. Non-identity specs also carry fragment
ID, basis ID, diagonal support, complete basis-change operation metadata, and
system size.

`DFRTEEventCircuitRequest` and `DFRTEEventSequenceCircuitRequest` validate that
every event application has a matching spec with identical coefficient sign,
magnitude, and identity classification. They reject baseline event reordering.

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

## DF-to-RTE conversion already implemented

`trotterlib.df_rte_tail` now performs the prerequisite conversion:

$$
\lambda_l\left(\sum_k\eta_{lk}n_k\right)^2
=c_I I+\sum_k c_k Z_k+\sum_{k<j}c_{kj}Z_kZ_j.
$$

It aggregates identical support only inside one fragment/basis, retains fixed
canonical ordering and hashing, and computes the actual RTE coefficient L1.
It supports both faithful identity-in-tail and extracted deterministic-phase
policies. Small dense references verify the central expansion, basis-conjugated
fragment, multiple-fragment tail, and controlled identity phase.

Circuit construction itself remains the next milestone after these types and
validated inputs.
