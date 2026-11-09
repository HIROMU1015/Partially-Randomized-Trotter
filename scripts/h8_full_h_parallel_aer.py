"""Exact small-gate action matrices and measured Aer threading; never a price lowerer."""
from pathlib import Path
import json
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

THREADS=1
DIRECTORY=None
ACTION=0


def configure(threads,directory=None):
    global THREADS,DIRECTORY,ACTION
    if int(threads)<1:raise ValueError('POSITIVE_AER_THREADS')
    THREADS=int(threads);DIRECTORY=Path(directory) if directory else None;ACTION=0
    if DIRECTORY:DIRECTORY.mkdir(parents=True,exist_ok=False)


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def action_circuit(qc,progress=None):
    """Replace unsupported <=3q gates only by their exact local Operator.

    Cache keys are the existing content-aware fingerprint keys, including gate
    definitions, control state and parameters. No whole-register matrix, phase
    fitting, compilation, approximate synthesis, or mutation of the priced input.
    """
    from h46_b3_fast_fingerprint import ContentKeys
    # The existing keys require an immutable input during one call, not forever.
    keys=ContentKeys();matrices={}
    backend=AerSimulator(method='statevector',device='CPU')
    supported=set(backend.configuration().basis_gates)|{'barrier'}
    out=QuantumCircuit(*qc.qregs,*qc.cregs)
    if out.qubits!=qc.qubits or out.clbits!=qc.clbits:raise ValueError('REGISTER_LAYOUT')
    replaced=hits=0;started=time.monotonic()
    for index,item in enumerate(qc.data):
        op=item.operation
        if op.name in supported:
            out._append(item)
        else:
            if not 1<=op.num_qubits<=3 or item.clbits or getattr(op,'condition',None) is not None:
                raise ValueError('UNSUPPORTED_LOCAL_MATRIX:'+op.name)
            key=keys.operation(op,qc)
            gate=matrices.get(key)
            if gate is None:
                matrix=np.array(Operator(op).data,dtype=complex,order="C",copy=True)
                error=float(np.linalg.norm(matrix.conj().T@matrix-np.eye(len(matrix))))
                if not np.isfinite(matrix).all() or error>1e-12:raise ValueError('EXACT_LOCAL_UNITARY')
                # Aer 0.15.1 interprets read-only C-order matrices differently;
                # pass the standard writable NumPy representation to UnitaryGate.
                gate=UnitaryGate(matrix,check_input=False);matrices[key]=gate
            else:hits+=1
            out._append(CircuitInstruction(gate,item.qubits,item.clbits));replaced+=1
        if progress and (index+1)%100000==0:
            progress(dict(stage='EXACT_LOCAL_MATRIX_CONVERSION',done=index+1,total=len(qc.data),
                elapsed=time.monotonic()-started,unique_matrices=len(matrices),threads=THREADS))
    return out,dict(input_instructions=len(qc.data),action_instructions=len(out.data),
        replaced_local_gates=replaced,matrix_cache_hits=hits,unique_cached_matrices=len(matrices),
        conversion_seconds=time.monotonic()-started,global_phase=float(qc.global_phase),
        priced_input_unchanged=True,largest_matrix_qubits=3)


def simulate_statevector(qc,initial):
    global ACTION
    index=ACTION;ACTION+=1;started=time.monotonic()
    destination=DIRECTORY
    def progress(value):
        if destination:write(destination/'progress.json',dict(action=index,**value))
    action,receipt=action_circuit(qc,progress)
    sim=QuantumCircuit(*action.qregs,*action.cregs)
    sim.set_statevector(np.array(initial,dtype=complex,copy=True));sim.compose(action,inplace=True)
    sim.save_statevector();del action
    progress(dict(stage='AER_STATEVECTOR',threads=THREADS,elapsed=time.monotonic()-started,**receipt))
    backend=AerSimulator(method='statevector',device='CPU',precision='double',
        max_parallel_threads=THREADS,max_parallel_experiments=1,zero_threshold=0.,
        fusion_enable=True,fusion_threshold=0,fusion_max_qubit=4)
    result=backend._execute_circuits_job([sim],None,{'shots':1},'H8_PARALLEL_ACTION_ONLY')
    if not result.success:raise ValueError('AER_STATEVECTOR')
    value=np.exp(1j*receipt['global_phase'])*np.asarray(result.get_statevector(sim),dtype=complex)
    metadata=result.results[0].metadata
    receipt.update(action=index,requested_threads=THREADS,
        parallel_state_update=metadata.get('parallel_state_update'),
        wall_seconds=time.monotonic()-started,phase_alignment=False,
        status='ACTION_COMPUTED_NOT_YET_CHECKED')
    if destination:
        with (destination/f'{index}_vector.npy').open('xb') as f:np.save(f,value,allow_pickle=False)
        write(destination/f'{index}_action.json',receipt)
        write(destination/'progress.json',receipt)
    return value
