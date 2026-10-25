"""Small-gate Aer CPU state actions; never construct a full circuit unitary.

Qiskit supplies matrices of at most three-qubit operations. Larger composite
instructions are traversed through their official definitions, keeping all
global phases. No custom numerical simulator or fermionic gate implementation.
"""
import math

VERSION='local_matrix_aer_cpu_state_action_v1'
PROBE_SEED=20261025
COMPONENT_ATOL=1e-10
SIGNAL_ATOL=2e-10


def simulation_circuit(circuit):
    """Exact gate-input conversion for simulation only, never a cost circuit.

Converting local matrices avoids a separate raw-circuit optimizing compiler and
Aer-specific native XXPlusYY semantics. Cost metrics use original P0 outputs.
"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UnitaryGate
    if circuit.num_clbits or circuit.parameters:raise ValueError('BOUND_UNITARY_CIRCUIT_REQUIRED')
    out=QuantumCircuit(circuit.num_qubits);out.global_phase=float(circuit.global_phase)
    def append(operation,qubits):
        if operation.name=='barrier':return
        if operation.name in ('measure','reset','initialize','set_statevector','save_statevector'):
            raise ValueError('NONUNITARY_OR_PREPARATION_OPERATION')
        if operation.num_qubits<=3:
            matrix=Operator(operation).data
            if matrix.shape[0]>8:raise ValueError('LOCAL_MATRIX_DIMENSION')
            out.append(UnitaryGate(matrix),qubits);return
        definition=operation.definition
        if definition is None:raise ValueError('NO_SMALL_GATE_DEFINITION')
        out.global_phase+=float(definition.global_phase)
        for item in definition.data:
            if item.clbits:raise ValueError('CLASSICAL_OPERATION')
            append(item.operation,[qubits[definition.find_bit(q).index] for q in item.qubits])
    for item in circuit.data:
        if item.clbits:raise ValueError('CLASSICAL_OPERATION')
        append(item.operation,[circuit.find_bit(q).index for q in item.qubits])
    return out


class StateActions:
    """Cache only task-local circuit conversions; Aer runs deterministic states."""
    def __init__(self,threads=1):
        from qiskit_aer import AerSimulator
        if type(threads) is not int or threads<1:raise ValueError('POSITIVE_THREADS')
        self.backend=AerSimulator(method='statevector',device='CPU',precision='double',
            max_parallel_threads=threads,max_parallel_experiments=threads)
        self.cache={};self.threads=threads

    def apply(self,circuit,states):
        import numpy as np
        from qiskit import QuantumCircuit
        states=np.asarray(states,dtype=np.complex128)
        if (states.ndim!=2 or states.shape[1]!=(1<<circuit.num_qubits)
                or not np.isfinite(states).all()):raise ValueError('STATE_BATCH_SCHEMA')
        if id(circuit) not in self.cache:
            body=simulation_circuit(circuit);phase=float(body.global_phase);body.global_phase=0.
            self.cache[id(circuit)]=(circuit,body,phase)
        original,body,phase=self.cache[id(circuit)]
        if original is not circuit:raise ValueError('CIRCUIT_CACHE_IDENTITY')
        batch=[]
        for psi in states:
            qc=QuantumCircuit(circuit.num_qubits);qc.set_statevector(psi)
            qc.compose(body,inplace=True);qc.save_statevector();batch.append(qc)
        # No measurements or Born samples. shots=1 is Aer snapshot plumbing,
        # not a shot prescription, RPE experiment or resource evaluation.
        result=self.backend.run(batch,shots=1).result()
        if not result.success:raise ValueError('STATE_ACTION_BACKEND_FAIL: '+str(result.status))
        out=np.asarray([np.asarray(result.get_statevector(i)) for i in range(len(batch))])
        # Aer applies a circuit's global phase before set_statevector, which
        # replaces it. Use the existing simulate_statevector helper's policy:
        # strip the DECLARED circuit phase and restore it after state evolution.
        # This is not a phase fit against a reference or an observed output.
        out*=np.exp(1j*phase)
        if not np.isfinite(out).all():raise ValueError('NONFINITE_STATE_ACTION')
        return out


def sector_embedding(indices,vector,system_dimension):
    import numpy as np
    out=np.zeros(system_dimension,complex);out[indices]=vector;return out


def probes(indices,ground,system_dimension,kernel):
    """One full-Fock control0 probe, three control1 sector probes, ground + control.

This finite list is NEVER a norm certificate or complete operator comparison.
All output amplitudes, including leakage, remain in the full controlled space.
"""
    import numpy as np
    d=len(ground);rng=np.random.default_rng(PROBE_SEED)
    def random(n):
        x=rng.normal(size=n)+1j*rng.normal(size=n);return x/np.linalg.norm(x)
    basis=np.eye(d,dtype=complex)[0];vectors=[basis,random(d),np.asarray(ground)]
    zero=np.zeros(system_dimension,complex);full=random(system_dimension)
    initial=[np.r_[full,zero]];expected=[initial[0]]
    for vector in vectors:
        initial.append(np.r_[zero,sector_embedding(indices,vector,system_dimension)])
        expected.append(np.r_[zero,sector_embedding(indices,kernel@vector,system_dimension)])
    psi=sector_embedding(indices,ground,system_dimension)
    target=sector_embedding(indices,kernel@ground,system_dimension)
    initial.append(np.r_[psi,psi]/math.sqrt(2));expected.append(np.r_[psi,target]/math.sqrt(2))
    return np.asarray(initial),np.asarray(expected)


def state_gate(actual,expected,*,label):
    import numpy as np
    from .c1_stage3_molecular import component_check
    report=component_check(actual,expected,COMPONENT_ATOL,label=label)
    norms=np.linalg.norm(actual,axis=1)
    drift=float(np.max(abs(norms-1)))
    if drift>COMPONENT_ATOL:raise ValueError('PROBE_STATE_NORM_FAIL')
    report.update(per_probe_L2=np.linalg.norm(actual-expected,axis=1).tolist(),
        norm_drift=drift,coverage='finite fixed input states; full output amplitudes',
        operator_norm_bound=None,rigorous_enclosure=None)
    return report


def measured_tail(circuit):
    """Existing terminal-ancilla readout policy; strip only its final measure."""
    positions=[i for i,item in enumerate(circuit.data) if item.operation.name=='measure']
    if positions!=[len(circuit.data)-1]:raise ValueError('ONE_TERMINAL_MEASUREMENT_REQUIRED')
    item=circuit.data[-1]
    if circuit.find_bit(item.qubits[0]).index!=circuit.num_qubits-1:raise ValueError('ANCILLA_LAST_MEASURE')
    return circuit.remove_final_measurements(inplace=False)


def ancilla_z(states):
    import numpy as np
    states=np.asarray(states);half=states.shape[-1]//2
    return np.sum(abs(states[...,:half])**2,axis=-1)-np.sum(abs(states[...,half:])**2,axis=-1)
