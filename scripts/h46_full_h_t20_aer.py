"""Exact-definition Aer action verifier; priced native/compiled circuits unchanged."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def simulate_statevector(qc,initial):
    backend=AerSimulator(method='statevector',device='CPU',precision='double',
        max_parallel_threads=1,zero_threshold=0)
    supported=set(backend.configuration().basis_gates)|{'barrier'}
    # Includes native controlled RZZ (crzz) and Gaussian XXPlusYY. Only their
    # exact definitions are expanded, with no optimization or approximate synthesis.
    for _ in range(8):
        unknown={i.operation.name for i in qc.data}-supported
        if not unknown:break
        before=unknown
        qc=qc.decompose(gates_to_decompose=sorted(unknown))
        after={i.operation.name for i in qc.data}-supported
        if after==before:
            raise ValueError('UNSUPPORTED_ACTION_GATE:'+','.join(sorted(after)))
    else:raise ValueError('ACTION_DEFINITION_DEPTH')
    phase=float(qc.global_phase)
    if phase!=0.:
        qc=qc.copy();qc.global_phase=0.
    sim=QuantumCircuit(qc.num_qubits)
    sim.set_statevector(np.array(initial,dtype=complex,copy=True))
    sim.compose(qc,inplace=True)
    sim.save_statevector()
    # Aer 0.15.1's run() retains a global executor thread and does not pass
    # the executor option into AerJob. Call that job's unchanged CPU execution
    # function synchronously so the legacy fingerprint precondition is kept.
    result=backend._execute_circuits_job([sim],None,{'shots':1},'T20_ACTION_ONLY')
    if not result.success:raise ValueError('AER_STATEVECTOR')
    return np.exp(1j*phase)*np.asarray(result.get_statevector(sim),dtype=complex)
