"""Full-space binary64 circuit operators using the existing local-gate bridge.

Aer evolves the complete identity matrix, including every input and output.
Actual inverse circuits must be passed separately; no inverse is inferred from
the forward operator. No projection, probe sampling, phase fit or cost-circuit
optimization is performed here.
"""
import threading

VERSION = 'full_identity_local_matrix_aer_cpu_double_v1'
_local = threading.local()


def decode(circuit, *, threads=1):
    import numpy as np
    from qiskit_aer import AerSimulator
    from .c1_h6_state_action import measured_tail, simulation_circuit

    if type(threads) is not int or threads < 1:
        raise ValueError('POSITIVE_UNITARY_THREADS_REQUIRED')
    clean = measured_tail(circuit) if circuit.num_clbits else circuit
    body = simulation_circuit(clean, native_basis=True)
    body.save_unitary()
    if getattr(_local, 'threads', None) != threads:
        _local.backend = AerSimulator(method='unitary', device='CPU',
            precision='double', max_parallel_threads=threads,
            max_parallel_experiments=1, zero_threshold=0.0,
            enable_truncation=False)
        _local.threads = threads
    result = _local.backend.run(body, shots=1).result()
    if not result.success:
        raise ValueError('FULL_UNITARY_BACKEND_FAILURE:' + str(result.status))
    matrix = np.asarray(result.get_unitary(0), dtype=np.complex128)
    dimension = 1 << clean.num_qubits
    if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
        raise ValueError('FULL_UNITARY_MATRIX_SCHEMA')
    return matrix
