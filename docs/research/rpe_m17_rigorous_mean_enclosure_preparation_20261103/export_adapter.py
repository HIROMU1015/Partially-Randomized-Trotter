"""Export only when separately called with a prepared context; inert on import."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite replay parameter")
    return value


def matrix_pairs(matrix):
    return [[[finite(z.real).hex(), finite(z.imag).hex()] for z in row] for row in matrix]


def export_basis(basis_id, basis_hash, descriptions, operations, *, n_qubits):
    import numpy as np
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    if len(descriptions) != len(operations):
        raise ValueError("Basis descriptor count mismatch")
    exported = []
    for description, (gate, wires) in zip(descriptions, operations, strict=True):
        # Names alone cannot establish ideal unitary semantics.
        if gate.base_class not in (RZGate, XXPlusYYGate):
            raise ValueError("Unsupported ideal basis gate class")
        raw = np.asarray(Operator(gate).data, dtype=np.complex128)
        matrix_hash = hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8).tobytes()).hexdigest()
        desc = asdict(description)
        if matrix_hash != desc['matrix_sha256'] or tuple(desc['qubits']) != tuple(wires):
            raise ValueError("Basis local matrix/descriptor mismatch")
        exported.append({'name': gate.name, 'qubits': list(wires),
                         'params': [finite(value).hex() for value in gate.params],
                         'legacy_descriptor': desc, 'local_binary64_matrix': matrix_pairs(raw)})
    return {'basis_id': basis_id, 'basis_hash': basis_hash, 'n_qubits': n_qubits,
            'operations': exported, 'ideal_semantics': 'analytic standard gate at exact binary64 parameters'}


def export_preparation(preparation, psi, ordered, *, n_qubits, n_electrons, delta):
    """Synthetic or separately authorized SOURCE replay, not a saved-input claim."""
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
    from trotterlib.df_rte_tail import canonical_basis_hash
    import numpy as np
    components = preparation.rte_preparation.symbolic_tail.components
    coefficients = preparation.tail_extraction.components
    if [c.component_id for c in components] != [c.component_id for c in coefficients]:
        raise ValueError("Probability/coefficient component order mismatch")
    bases = {}

    def basis(key, fingerprint, descriptions, operations):
        if canonical_basis_hash(n_qubits, descriptions) != fingerprint:
            raise ValueError("Canonical basis hash mismatch")
        item = export_basis(key, fingerprint, descriptions, operations, n_qubits=n_qubits)
        if key in bases and bases[key] != item:
            raise ValueError("Conflicting basis definitions")
        bases[key] = item

    exported_components = []
    for component, coefficient in zip(components, coefficients, strict=True):
        definition = preparation.tail_extraction.basis_definition(component.basis_id)
        basis(definition.basis_id, definition.basis_hash, definition.metadata.operations, definition.runtime_operations)
        if (component.coefficient_sign != coefficient.coefficient_sign
                or component.basis_hash != definition.basis_hash):
            raise ValueError("Component sign/basis mismatch")
        exported_components.append({'id': component.component_id, 'coefficient': finite(coefficient.coefficient),
            'coefficient_abs': finite(component.coefficient_abs), 'sign': component.coefficient_sign,
            'probability64': finite(component.probability), 'basis_id': component.basis_id,
            'support': list(component.diagonal_pauli_support), 'fragment_id': component.df_fragment_id})
    blocks = []
    for block in preparation.deterministic_blocks:
        basis(block.basis_id, block.basis_hash, block.basis_change_operations, block.runtime_basis_operations)
        if block.block_kind == 'one_body':
            primitive = one_body_diagonal_primitives(np.asarray(block.diagonal_eigenvalues), delta/2)
        elif block.block_kind == 'df_fragment':
            primitive = df_squared_diagonal_primitives(np.asarray(block.diagonal_eta), block.lam, delta/2)
        else:
            raise ValueError("Unknown deterministic block")
        blocks.append({'block_id': block.block_id, 'basis_id': block.basis_id, 'primitives': {
            'global_phase': finite(primitive.global_phase).hex(),
            'rz': [[int(q), finite(angle).hex()] for q, angle in primitive.rz],
            'rzz': [[int(q), int(k), finite(angle).hex()] for q, k, angle in primitive.rzz]}})
    return {'kind': 'RPE_M17_SOURCE_REPLAY_V1', 'provenance': 'SOURCE_REPLAY_GENERATED; not old saved component payload',
            'n_qubits': n_qubits, 'n_electrons': n_electrons, 'ordered_sector': [int(i) for i in ordered],
            'state': [[finite(z.real).hex(), finite(z.imag).hex()] for z in psi],
            'preparation_hash': preparation.preparation_hash, 'tail_hash': preparation.tail_extraction.tail_hash,
            'lambda_r64': finite(preparation.exact_rte_lambda_r), 'components': exported_components,
            'bases': bases, 'deterministic_blocks': blocks,
            'step_phase_angles': [finite(-delta*preparation.constant_coefficient).hex(),
                                  finite(-delta*preparation.extracted_identity_coefficient).hex()],
            'identity_policy': preparation.identity_policy, 'coefficient_atol': preparation.coefficient_atol,
            'deterministic_fragment_indices': list(preparation.deterministic_fragment_indices),
            'randomized_fragment_indices': list(preparation.randomized_block_indices)}


def replay(root, original):
    """Real scientific entry: one fixed legacy preparation, no state solve or RNG."""
    prior = original.prior_runner()
    prior.source_and_input_identity()
    api = original.load_numerical_api()
    h = api['load_connected_cluster_hamiltonian_snapshot'](root/prior.H_PATH)
    if api['df_hamiltonian_hash'](h) != prior.H_SHA or h.n_qubits != 8 or h.n_blocks != 12:
        raise ValueError("Fixed Hamiltonian mismatch")
    sector = api['PhysicalSector'].number_sector(n_qubits=8, n_electrons=4)
    import numpy as np
    with np.load(root/prior.STATE_PATH, allow_pickle=False) as saved:
        psi = saved['state']
        if not np.array_equal(saved['basis_indices'], sector.basis_indices) or str(saved['hamiltonian_hash']) != prior.H_SHA:
            raise ValueError("Fixed state sector/H mismatch")
    if api['_array_fingerprint'](psi) != prior.STATE_FP:
        raise ValueError("Fixed state fingerprint mismatch")
    prep = api['prepare_df_partial_s2'](h, api['split_df_hamiltonian_by_ld'](h, 3),
        identity_policy='extract_identity_phase', coefficient_atol=1e-12)
    if (prep.preparation_hash != prior.PREPARATION_SHA
            or prep.tail_extraction.tail_hash != '65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af'
            or len(prep.rte_preparation.symbolic_tail.components) != 218 or len(prep.deterministic_blocks) != 4):
        raise ValueError("SOURCE replay inventory/hash mismatch")
    api['_validate_components'](prep.rte_preparation.symbolic_tail.components)
    ordered = sector.basis_indices[api['_qiskit_to_openfermion_sector_permutation'](sector)]
    return export_preparation(prep, psi, ordered, n_qubits=8, n_electrons=4, delta=.02)
