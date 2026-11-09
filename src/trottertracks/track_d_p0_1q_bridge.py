"""Limited native 1q-run resynthesis + existing CX bridge, without transpile.

Use the official Qiskit1.3 ZSXX run synthesizer for target-free P0 basis. Only
strictly shorter native runs are substituted; no RZ commuting through CX.
This is not a reconstruction of the full P0 pipeline/translation schedule.
"""
import math
from .track_d_native_resources import NativeInstruction, NativeCounter, count_native
from .track_d_p0_cx_bridge import CXBoundaryBridge


def oneq_runs(operations,num_qubits):
    import numpy as np
    import qiskit
    from qiskit.circuit.library import RZGate,SXGate,XGate
    from qiskit.circuit import CircuitInstruction,Qubit
    from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import Optimize1qGatesDecomposition
    if qiskit.__version__!='1.3.0':raise ValueError('Qiskit1.3.0 run rules required')
    synthesis=Optimize1qGatesDecomposition(['rz','sx','x','cx'])
    if synthesis._global_decomposers!=['ZSXX'] or synthesis.error_map is not None:
        raise ValueError('target-free fixed native basis required')
    pending=[[] for _ in range(num_qubits)];output=[];phase=[];replaced=0
    def flush(wire):
        nonlocal replaced
        run=pending[wire]
        if not run:return
        matrix=np.eye(2,dtype=complex)
        for op in run:
            gate=RZGate(op.params[0]) if op.name=='rz' else (SXGate() if op.name=='sx' else XGate())
            matrix=gate.to_matrix()@matrix
        # Call only the bounded 2x2 synthesizer, never TransformationPass.run.
        sequence=synthesis._resynthesize_run(matrix,wire)
        if sequence is not None and len(sequence)<len(run):
            bit=Qubit()
            for standard,params in sequence:
                gate=CircuitInstruction.from_standard(standard,(bit,),params).operation
                output.append(NativeInstruction(gate.name,(wire,),params=tuple(gate.params)))
            phase.append(float(sequence.global_phase));replaced+=1
        else:output.extend(run)
        pending[wire]=[]
    for operation in operations:
        if operation.name in ('rz','sx','x'):
            pending[operation.qubits[0]].append(operation)
        else:
            for wire in operation.qubits:flush(wire)
            output.append(operation)
    for wire in range(num_qubits):flush(wire)
    return output,math.fsum(phase),replaced


def optimize_native(operations,num_qubits,num_clbits=0,*,global_phase=0.):
    """Finite strict-size-decreasing fixed point of the declared partial rules.

    Counts/phase refer to this resulting native word. General P0 equivalence is
    deliberately unqualified. Working storage can grow with input length.
    """
    initial=NativeCounter(num_qubits,num_clbits,global_phase=global_phase)
    word=[]
    for op in operations:initial.feed(op);word.append(op)
    phase=float(global_phase);rounds=[]
    while True:
        before=len(word)
        folded,addition,replaced=oneq_runs(iter(word),num_qubits)
        phase=math.fsum((phase,addition))
        cx=CXBoundaryBridge(num_qubits,num_clbits).extend(iter(folded))
        word=list(cx.operations())
        rounds.append(dict(input_size=before,output_size=len(word),oneq_runs_replaced=replaced,
                           cx_pairs_removed=cx.cancelled_pairs))
        if len(word)==before:break
        if len(word)>before:raise ValueError('partial bridge must strictly decrease size')
    result=count_native(iter(word),num_qubits,num_clbits,global_phase=phase)
    return dict(input=initial.result(),output=result,operations=word,global_phase=phase,
                rounds=rounds,p0_qualification='P0_UNQUALIFIED',
                rule='official_ZSXX_strict_shorter_1q_runs_then_wire_adjacent_CX')
