# Exact DF-tail extraction

`trotterlib.df_rte_tail` converts each selected native squared DF fragment into
the Hermitian involutions required by finite RTE. It does not treat an entire
multi-eigenvalue DF block as one involution.

## Symbolic normal path

`extract_df_diagonal_tail` is a symbolic path. It stores signed I/Z/ZZ
coefficients, fragment provenance, diagonal support, local basis-operation
metadata, executable runtime operations, basis hashes, ordering, identity
policy, threshold audit data, and coefficient L1 norms. It does not construct:

- a many-body basis unitary;
- a dense Pauli matrix or conjugated component;
- a dense tail Hamiltonian.

The number of symbolic components for one generic $N$-qubit squared fragment
is at most $1+N+N(N-1)/2$. Synthetic 20- and 26-qubit tests exercise this path
while failing if a dense reference helper or `Operator(circuit).data` is used.
This is a symbolic scalability check only. It is not an H10/H13 chemistry,
statevector, or circuit-execution result.

For $n_k=(I-Z_k)/2$,

$$
\lambda\left(\sum_k\eta_kn_k\right)^2
=c_I I+\sum_k c_kZ_k+\sum_{k<j}c_{kj}Z_kZ_j,
$$

where

$$
c_I=\frac{\lambda}{2}\left(\sum_k\eta_k^2+
\sum_{k<j}\eta_k\eta_j\right),
$$

$$
c_k=-\frac{\lambda}{2}\eta_k\sum_j\eta_j,
\qquad
c_{kj}=\frac{\lambda}{2}\eta_k\eta_j.
$$

Components are ordered by fragment, support size, and support indices.
Identical support is aggregated only within the same fragment/basis. Equal
Z/ZZ support from distinct orbital bases remains distinct.

## Dense-free RTE normalization

`extraction_to_symbolic_rte_tail` converts the randomized DF components to a
`SymbolicRTETail`. For every retained randomized coefficient $h_l$ it sets

$$
p_l=\frac{|h_l|}{\lambda_R},
\qquad
\lambda_R=\sum_l|h_l|.
$$

The conversion preserves canonical component order, coefficient magnitude and
sign, identity classification, support, fragment and component IDs, basis ID
and hash, basis-operation metadata, threshold audit data, identity policy, and
tail provenance. It allocates no object whose size is exponential in the
system qubit count. `make_rte_config` accepts this symbolic tail through the
same lightweight tail interface used by the dense reference type.

If thresholding or `extract_identity_phase` leaves no randomized component,
`SymbolicRTETail.is_deterministic_only` is true, `lambda_r` is zero, and the
component tuple is empty. This is a valid explicit result. APIs that require
random event sampling raise `DeterministicOnlyRTETailError` instead of dividing
by zero. A faithful identity-only tail remains a one-component randomized tail.

## Basis definitions and registry

`BasisChangeMetadata` is serializable. `DFBasisDefinition` additionally keeps
the runtime Qiskit gate sequence, and `DFBasisRegistry` retrieves it by
`basis_id`. The basis hash is computed from system size and the ordered local
operation metadata. Standard registry operations must act on at most four
qubits and be convertible to a local Qiskit `Operator` matrix. That local
matrix hash is mandatory. Wider or opaque operations are rejected with
`BasisFingerprintError` rather than accepted under ambiguous name/parameter
metadata. A many-body basis matrix hash is never constructed.

With automatically generated IDs, identical operation sequences share the
same basis ID and hash across fragments. Fragment IDs remain part of component
IDs, so physical provenance is never merged. Registering different operation
sequences under the same explicit basis ID raises an error. This permits a
future builder to recognize adjacent equal bases without authorizing event
reordering.

`prepare_df_rte_event_inputs` returns a `DFRTEEventPreparation` containing the
symbolic normalized tail, component circuit specs, and the executable registry.
It verifies that the registry definitions and component fingerprints agree.
The bundle can classically sample finite RTE events and turn each sample into a
validated future-builder request without reconstructing any dense operator.

Two identity policies are available:

- `faithful_identity_in_tail`: identity coefficients participate in RTE L1 and
  event probabilities;
- `extract_identity_phase`: identity coefficients are summed into an exact
  deterministic phase baseline and excluded from RTE L1.

Both reconstruct the same retained Hamiltonian. In a controlled evolution the
extracted phase must act relatively on the ancilla; it cannot be discarded as
a system global phase.

`rte_lambda_r` is the actual retained coefficient L1. The existing DF fragment
ranking sum is stored separately as `ranking_proxy_lambda_r` and is never used
as the RTE normalization.

The threshold default is zero. A nonzero threshold records dropped count,
dropped L1, operator error bound, and policy. Hashing covers canonical
pre-threshold components plus identity and threshold policy.

`DFTailExtractionMetadata` separately records threshold input/retained/dropped
counts, retained and extracted identity counts, final randomized count,
threshold L1/error, extracted identity coefficient, and randomized L1. Moving
identity to deterministic phase is exact and is not counted as threshold
error. `normalization_metadata` describes only the final randomized component
set, so its counts agree with `len(extraction.components)`.

## Guarded dense references

`basis_change_unitary`, `diagonal_pauli_matrix`, `component_dense_operator`,
`dense_extracted_df_tail`, `dense_df_block_hamiltonian`, and
`extraction_to_normalized_rte_tail` are small-system validation helpers. Each
checks `max_dense_qubits` before allocating a dense many-body object; the
default is 8 qubits. They are not part of normal RTE event generation.
