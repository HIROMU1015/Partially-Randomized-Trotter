"""Candidate A only. Dormant until a separately approved scientific execution.

Imports are lazy: importing this module performs no decomposition or build.
Nature's initial occupation prefix is checked and omitted, never executed.
"""
from .circuits import BasisProvider


POLICY = 'nature_0.7.2_rectangular_V_transpose_T_first_m_singleton_only'


def gaussian_suffix(operations, register, rank):
    """Consume exactly the pinned helper's X prefix; reject unexpected recipes.

    Separated from the numerical helper so the protocol can be tested with
    artificial tokens without constructing any quantum circuit.
    """
    operations = iter(operations)
    for index in range(rank):
        try:
            gate, qubits = next(operations)
        except StopIteration as error:
            raise ValueError('Missing Slater initialization prefix') from error
        if gate.name != 'x' or tuple(qubits) != (register[index],):
            raise ValueError('Unexpected Slater initialization prefix')
    for gate, qubits in operations:
        qubits = tuple(qubits)
        positions = tuple(register.index(q) for q in qubits)
        if (gate.name != 'xx_plus_yy' or len(positions) != 2
                or positions[0] != positions[1] + 1):
            raise ValueError('Unexpected Gaussian suffix; pinned Nature recipe required')
        yield gate, qubits


def rectangular_gaussian(frame, support):
    """G only, on all physical modes. No state preparation or square completion.

    V.T is the adopted creation-coefficient convention. Its projector/operator
    equivalence remains a FUTURE scientific gate, not a claim of this adapter.
    """
    from importlib.metadata import version
    if version('qiskit-nature') != '0.7.2':
        raise ValueError('Candidate requires the fixed Nature 0.7.2 helper')
    import numpy as np
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit_nature.second_q.circuit.library.initial_states.utils.givens_rotations import (
        _prepare_slater_determinant_jw,
    )
    matrix = np.asarray(frame, dtype=complex)
    n = len(matrix)
    if (matrix.shape != (n, n) or tuple(sorted(set(support))) != tuple(support)
            or len(support) not in (1, 2) or any(p < 0 or p >= n for p in support)
            or not np.isfinite(matrix).all()):
        raise ValueError('Finite square frame and selected support1/2 required')
    # A private working copy: the helper may update its input during elimination.
    coefficients = matrix[:, list(support)].T.copy()
    register = QuantumRegister(n)
    circuit = QuantumCircuit(register)
    operations = _prepare_slater_determinant_jw(register, coefficients)
    for gate, qubits in gaussian_suffix(operations, list(register), len(support)):
        circuit.append(gate, qubits)
    return circuit


class RectangularBasisProvider(BasisProvider):
    """Opt-in provider. Same-frame runs and physical Pauli use the legacy path."""
    def __init__(self, modes, frames):
        super().__init__(modes, frames)
        self._rectangular_cache = {}

    def reference_support(self, atom, singleton):
        if singleton and atom.kind == 'FRAME_Z':
            return tuple(range(len(atom.support)))
        return super().reference_support(atom, singleton)

    def basis(self, atom, singleton):
        if not singleton or atom.kind != 'FRAME_Z':
            return super().basis(atom, singleton)
        # Original operator identity is retained; equal canonical T is NOT a
        # basis-sharing criterion. The existing router still groups basis_key.
        key = (POLICY, atom.frame_id, atom.basis_key, atom.support)
        if key not in self._rectangular_cache:
            self._rectangular_cache[key] = rectangular_gaussian(
                self.frames[atom.frame_id], atom.support)
        return self._rectangular_cache[key].copy()
