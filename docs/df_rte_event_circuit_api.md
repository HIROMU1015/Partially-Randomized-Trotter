# Proposed DF RTE event circuit API

The next milestone should implement the protocol already declared in
`trotterlib.df_rte_circuit`.  This document fixes the intended semantics before
circuit construction starts.

## Boundary

```text
DFRTEEventCircuitBuilder.build_event(
    DFRTEEventCircuitRequest(
        event: RTEEvent,
        component_specs: tuple[DFRTEComponentCircuitSpec, ...],
        controlled: bool,
        ancilla_qubit: int | None,
        preserve_event_order: True,
        cancel_adjacent_equal_bases: True,
        control_diagonal_only: True,
        identity_as_relative_ancilla_phase: True,
    )
) -> DFRTEEventCircuitResult
```

For the (r) integer steps of one tail evolution, the parallel
`DFRTEEventSequenceCircuitRequest` / `build_sequence` entry point accepts an
ordered tuple of events.  This is where a basis cancellation spanning two
adjacent RTE-step boundaries is detected; it still cannot reorder either
event or component order.

Each component spec identifies one **proven Hermitian involution** and its DF
fragment/basis provenance.  Following Sec. VII B of the primary paper, the
baseline component is

\[
\widetilde P_m=(U^{(l)})^\dagger P_mU^{(l)},
\]

where `diagonal_involution_id` identifies the diagonal Pauli (P_m).  Multiple
components from the same DF fragment share `basis_id`.  The API intentionally
does not claim that the complete squared DF fragment is one involution.

## Required behavior

1. Consume `event.selected_component_ids` in its existing circuit order.  The
   request rejects `preserve_event_order=False`.
2. Emit the order-(n) phase ((-1)^{n/2}), the (n) product components, and
   the final rotation with the exact finite-RTE angle.
3. For a component represented by (U_l^\dagger P_mU_l), keep (U_l) and
   (U_l^\dagger) uncontrolled when mathematically valid and control only the
   central diagonal operation.
4. Convert identity/global phase to ancilla-relative phase in controlled
   circuits.  Do not discard Qiskit `global_phase` during control conversion.
5. Cancel only adjacent (U_lU_l^\dagger) pairs with identical `basis_id`.
   Record the count and the resulting basis-switch count.
6. Never group or reorder separated equal-fragment events in the baseline.
   Basis-aware block randomization belongs in a separate experiment and must
   revalidate the expected operator or quantify additional bias.
7. Build/transpile/count only.  Do not run statevector or quantum-shot
   sampling for controlled resource circuits.

## Returned evidence

`DFRTEEventCircuitResult` returns the circuit, the component order actually
used, number of safely cancelled basis-change pairs, relative ancilla phase,
and basis-switch count.  A later cost layer should transpile this result and
create `CircuitCost` with:

- primary `rz_count`;
- `rz_depth`, `cx_count`, `cx_depth`, `total_depth`, `circuit_size`;
- complete `CompilerSettings` and fidelity level;
- `exact_finite_distribution` for enumerated small-system expectations or
  `empirical_compiled_estimate` plus sample variance/error/confidence interval
  for stratified classical event sampling.

## Required DF-to-RTE conversion

Appendix A RTE requires (P_l^2=I), while a complete chemistry DF fragment
(U_l^\dagger D_lU_l) generally has a multi-valued spectrum.  The baseline
conversion is therefore the construction stated in Sec. VII B:

1. expand the diagonal number-operator polynomial (D_l) into identity, Z,
   and ZZ Pauli terms with their exact signed coefficients;
2. conjugate each non-identity diagonal Pauli by the fragment orbital basis
   change to obtain (\widetilde P_m);
3. retain the source fragment and shared basis IDs so adjacent event components
   can safely reuse basis changes;
4. define (\lambda_R), component probabilities, identity/global phase, and
   the tail hash from this exact expansion.

Coefficient aggregation, zero thresholds, and identity handling must be fixed
and dense-validated against the existing DF Hamiltonian before circuit costs
are produced.  A generalized non-involutory RTE or block encoding would be a
separate method, not the baseline.  Substituting a raw DF fragment into the
Pauli formula remains forbidden.
