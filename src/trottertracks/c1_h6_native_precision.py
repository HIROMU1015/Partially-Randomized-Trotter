# This code is part of a Qiskit project.
# (C) Copyright IBM 2022, 2023.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0
# Modified 2026-10-25: task-local numeric variant of Qiskit Nature 0.7.2's
# _bogoliubov_transform_num_conserving_jw. Retain its elimination schedule,
# RZ/XXPlusYY gates and U.T convention. Replace approximate zero tests with
# exact zeros, use LAPACK for Givens, and atan2 to retain small angles.
"""H6-only precision repair; no package edits or changed sharing/compiler rules."""
import math
from .c1_stage2_native import NativeBuilder as ExistingBuilder

VERSION = 'nature_jw_exact_zero_lapack_atan2_v1'


def _givens(a, b):
    """Existing LAPACK rotation, preserving Nature's exact-zero phase choices."""
    import numpy as np
    from scipy.linalg.lapack import zlartg
    if a == 0:
        c, s = 0., 1.
    elif b == 0:
        c, s = 1., 0.
    else:
        c, s, _ = zlartg(complex(a), complex(b))
    return np.array([[c, s], [-complex(s).conjugate(), c]], complex)


def gaussian(matrix):
    """Reuse Nature's number-conserving JW schedule without dropping small U.

    Basis-only RZ vacuum phase cancels in B D B.inverse(), as in the old
    NativeBuilder. The small n-by-n transfer check is not a full-Fock operator
    certificate. At most two-qubit matrices are used for this check.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    from qiskit.quantum_info import Operator
    from qiskit_nature.utils import apply_matrix_to_slices
    u = np.asarray(matrix, complex)
    if (u.ndim != 2 or u.shape[0] != u.shape[1] or not len(u)
            or not np.isfinite(u).all()):
        raise ValueError('GAUSSIAN_INPUT_SCHEMA')
    n = len(u)
    if np.max(abs(u.conj().T @ u - np.eye(n))) > 1e-10:
        raise ValueError('GAUSSIAN_NONUNITARY_INPUT')
    current = u.T.copy()
    left, right = [], []
    for layer in range(n - 1):
        for j in range(layer + 1):
            if layer % 2 == 0:
                target, row = layer - j, n - j - 1
                if current[row, target] != 0:
                    g = _givens(current[row, target + 1], current[row, target])
                    right.append((g, (target + 1, target)))
                    current = apply_matrix_to_slices(current, g,
                        [(Ellipsis, target + 1), (Ellipsis, target)])
            else:
                target, col = n - layer + j - 1, j
                if current[target, col] != 0:
                    g = _givens(current[target - 1, col], current[target, col])
                    left.append((g, (target - 1, target)))
                    current = apply_matrix_to_slices(current, g, [target - 1, target])
    for g, (i, j) in reversed(left):
        g = g.conj().T.copy()
        g[:, 0] *= current[i, i]
        g[:, 1] *= current[j, j]
        new_g = _givens(g[1, 1], g[1, 0])
        right.append((new_g.T, (i, j)))
        phase = g @ new_g
        current[i, i], current[j, j] = phase[0, 0], phase[1, 1]
    qc = QuantumCircuit(n)
    transfer = np.eye(n, dtype=complex)
    for i in range(n):
        phi = float(np.angle(current[i, i]))
        qc.append(RZGate(phi), [i])
        transfer[i] *= np.exp(1j * phi)
    for g, (i, j) in reversed(right):
        theta = math.atan2(abs(g[0, 1]), float(g[0, 0].real))
        phi = float(np.angle(g[0, 1]))
        gate = XXPlusYYGate(2 * theta, phi - math.pi / 2)
        qc.append(gate, [j, i])
        small = Operator(gate).data[np.ix_([1, 2], [1, 2])]
        transfer[[j, i]] = small @ transfer[[j, i]]
    residual = float(np.max(abs(transfer - u)))
    if not math.isfinite(residual) or residual > 1e-10:
        raise ValueError('GAUSSIAN_TRANSFER_MISMATCH: ' + str(residual))
    return qc, dict(version=VERSION, modes=n, single_particle_max_entry=residual,
        allowance=1e-10, coverage='single particle relative to vacuum phase',
        full_operator_norm_bound=None)


class PrecisionBuilder(ExistingBuilder):
    """Change Gaussian numerical precision only; reuse all native PF emitters."""
    def __init__(self):
        super().__init__()
        self.basis_reports = {}

    def _basis(self, model, block):
        key = (model.source_identity, model.input_identity, block.rotation)
        if key not in self.cache:
            self.call('gaussian_build', model.label + ':' + block.block_id)
            self.cache[key], report = gaussian(block.rotation)
            self.basis_reports[model.label + ':' + block.block_id] = report
        return self.cache[key]
