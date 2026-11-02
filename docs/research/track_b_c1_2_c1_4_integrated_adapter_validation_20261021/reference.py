"""Selected-event numerical references; no synthesis, eigensolve or sampling."""
from functools import lru_cache
import math
import numpy as np
from qiskit.quantum_info import Operator, Statevector
from adapter import hadamard, matrix_hash

TOLERANCE = 1e-10  # frozen G10 strict phase/operator and first-moment diagnostic

def saved_basis(doc, n):
    matrix = Operator(np.eye(1 << n, dtype=complex))
    for record in doc['operations_in_time_order']:
        raw = record['local_matrix']
        local = np.array([[float.fromhex(v) for v in row] for row in raw['real_hex']]) + 1j * np.array(
            [[float.fromhex(v) for v in row] for row in raw['imag_hex']])
        if matrix_hash(local) != record['descriptor']['matrix_sha256']:
            raise ValueError('REFERENCE_LOCAL_BYTES_MISMATCH')
        matrix = matrix.compose(Operator(local), qargs=record['descriptor']['qubits'])
    return matrix.data

def q_reference(rows, bases, n):
    @lru_cache(None)
    def q(label):
        row = rows[label]
        diagonal = np.array([row['sign'] * (-1) ** sum((k >> j) & 1 for j in row['support'])
                             for k in range(1 << n)])
        basis = bases[row['basis_id']]
        return (basis * diagonal) @ basis.conj().T
    return q

def event_reference(event, q, n, sigma=1, scalar_phase=0.0):
    product = np.eye(1 << n, dtype=complex)
    for label in event['word']:
        product = product @ q(label)
    phi = math.atan(float(event['ratio']))
    rotation = math.cos(phi) * np.eye(1 << n) - 1j * sigma * math.sin(phi) * q(event['child'])
    return np.exp(-1j * scalar_phase) * (-1j * sigma) ** len(event['word']) * rotation @ product

def controlled_reference(active):
    d = len(active)
    matrix = np.eye(2 * d, dtype=complex)
    matrix[d:, d:] = active
    return matrix

def compare(circuit, active, state, n):
    actual = Operator(circuit).data
    expected = controlled_reference(active)
    error = float(np.linalg.norm(actual - expected, 'fro'))
    d = 1 << n
    state_in = Statevector(np.concatenate((state, np.zeros(d, dtype=complex))))
    signal = np.vdot(state, active @ state)
    observations = {}
    for axis, target in (('X', signal.real), ('Y', signal.imag)):
        evolved = state_in.evolve(hadamard(circuit, n, axis)).data
        observed = float(np.sum(np.abs(evolved[:d]) ** 2) - np.sum(np.abs(evolved[d:]) ** 2))
        observations[axis] = {'reference': float(target), 'wrapper': observed,
                              'absolute_error_diagnostic': abs(observed - float(target))}
    return {'controlled_Frobenius_error_upper_for_spectral_diagnostic': error,
            'inactive_Frobenius_error': float(np.linalg.norm(actual[:d, :d] - np.eye(d), 'fro')),
            'active_Frobenius_error': float(np.linalg.norm(actual[d:, d:] - active, 'fro')),
            'actual_adjoint_Frobenius_error': float(np.linalg.norm(Operator(circuit.inverse()).data - actual.conj().T, 'fro')),
            'XY': observations, 'strict_relative_phase_comparison': True,
            'operator_numerical_gate_passed': error <= TOLERANCE,
            'tolerance': TOLERANCE, 'rigorous_native_certificate': False}
