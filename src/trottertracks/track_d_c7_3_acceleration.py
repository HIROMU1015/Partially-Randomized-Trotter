"""Worker-local exact conversion cache and <=9-qubit Aer unitary adapter.

No cost-circuit/compiler modification, approximation, phase fit, or sampling.
The existing official Aer backends supply every numerical operation. Never
materialize a 17-qubit unitary. Select only in the new handoff SOURCE workers.
"""
from numbers import Integral
import struct
from uuid import uuid4

from . import track_d_state_action_acceleration as batched

_CACHE={}
_COUNTS=dict(hits=0,misses=0,uncached=0)
_TYPES=None


def _key(operation):
    global _TYPES
    import numpy as np
    from qiskit.circuit.library import (RZGate, RXGate, RYGate, UGate, PhaseGate,
        RZZGate, XXPlusYYGate, CRZGate, CPhaseGate, CXGate, CZGate, CCXGate,
        CSwapGate, HGate, XGate, SXGate, SGate, SdgGate)
    if _TYPES is None:
        classes=(RZGate,RXGate,RYGate,UGate,PhaseGate,RZZGate,XXPlusYYGate,CRZGate,CPhaseGate,
                 CXGate,CZGate,CCXGate,CSwapGate,HGate,XGate,SXGate,SGate,SdgGate)
        singleton_classes=tuple(type(cls()) for cls in (CXGate,CZGate,CCXGate,CSwapGate,HGate,XGate,SXGate,SGate,SdgGate))
        _TYPES=set(classes+singleton_classes)
    if type(operation) not in _TYPES:return None
    params=[]
    for value in operation.params:
        if isinstance(value,Integral):params.append(('int',int(value)))
        elif isinstance(value,(float,np.floating)) and np.asarray(value).dtype.itemsize<=8:
            params.append(('double',struct.pack('!d',float(value))))
        elif isinstance(value,complex):params.append(('complex',struct.pack('!dd',value.real,value.imag)))
        else:return None
    return (type(operation),operation.num_qubits,getattr(operation,'ctrl_state',None),tuple(params))


def cached_simulation_circuit(circuit):
    """Same local-matrix/definition traversal as existing simulation_circuit.

    Only exact standard-gate parameters are cached. Matrices remain official
    Qiskit Operator results, at most8x8; arbitrary gates use the old conversion.
    Shared validated UnitaryGate objects are simulation-only and never mutated.
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
            key=_key(operation)
            if key is not None and key in _CACHE:
                gate=_CACHE[key];_COUNTS['hits']+=1
            else:
                matrix=Operator(operation).data
                if matrix.shape[0]>8:raise ValueError('LOCAL_MATRIX_DIMENSION')
                gate=UnitaryGate(matrix)
                if key is not None:_CACHE[key]=gate;_COUNTS['misses']+=1
                else:_COUNTS['uncached']+=1
            out.append(gate,qubits,copy=False);return
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


class SmallUnitary:
    """Full <=9-qubit matrix from official Aer double unitary propagation.

    Complete matrix coverage replaces only slow Qiskit circuit Operator
    evaluation. It does not replace the independent saved CAR/raw targets.
    Reference17-qubit checks retain the existing complete-column state action.
    """
    def __init__(self,threads):
        from qiskit_aer import AerSimulator
        if type(threads) is not int or threads<1:raise ValueError('POSITIVE_THREADS')
        self.threads=threads
        self.backend=AerSimulator(method='unitary',device='CPU',precision='double',
            max_parallel_threads=threads,max_parallel_experiments=1,statevector_parallel_threshold=1)
        self.last_metadata={}
    def __call__(self,circuit):
        import numpy as np
        from qiskit_aer.jobs import AerJob
        from threadpoolctl import threadpool_limits
        if circuit.num_qubits>9:raise ValueError('FULL_REFERENCE_OPERATOR_FORBIDDEN')
        body=cached_simulation_circuit(circuit)
        phase=float(body.global_phase);body.global_phase=0.;body.save_unitary()
        with batched._NativeExecutor(self.threads) as executor,threadpool_limits(limits=1,user_api='blas'):
            job=AerJob(self.backend,str(uuid4()),self.backend._execute_circuits_job,
                circuits=[body],parameter_binds=None,run_options={'shots':1},executor=executor)
            job.submit();result=job.result()
            if not result.success:raise ValueError('UNITARY_BACKEND_FAIL:'+str(result.status))
            value=np.asarray(result.get_unitary(0),dtype=np.complex128).copy()
            self.last_metadata={**result.metadata,**result.results[0].metadata}
        value*=np.exp(1j*phase)
        if value.shape!=(1<<circuit.num_qubits,)*2 or not np.isfinite(value).all():raise ValueError('FULL_SMALL_UNITARY_SCHEMA')
        return value


def install(threads):
    """Explicit selection in a fresh NEW worker only; no source-file patch."""
    from . import c1_h6_state_action as primitive
    from . import track_d_h4_reference as base
    batched.simulation_circuit=cached_simulation_circuit
    primitive.StateActions=batched.BatchedStateActions
    engine=SmallUnitary(threads);base.small_operator=engine
    return engine


def cache_statistics():
    return dict(_COUNTS,entries=len(_CACHE),key_policy='exact standard class/control state and binary64 parameter bytes',
                matrix_source='official Qiskit Operator of at most3-qubit gate',cost_circuit_changed=False)
