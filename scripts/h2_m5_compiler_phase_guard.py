"""Process-local Qiskit 1.3 translator scalar bookkeeping, without state fitting.

The 1.3.0 Rust replace_node parameterized branch omits a numeric template
global_phase. A zero-valued, live symbolic scalar parameter keeps the same
template phase in its supported ParameterExpression branch. Every pass checks
that the original translator's complete gate/angle/wire stream is unchanged.
No shared library, SessionEquivalenceLibrary or on-disk runtime is changed.
"""
from contextlib import contextmanager
import hashlib
import math
import threading

POLICY='H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1'

def require(ok,why):
    if not ok:raise ValueError(why)

def stream_digest(dag):
    out=hashlib.sha256()
    for node in dag.topological_op_nodes():
        require(not getattr(node.op,'condition',None),'NO_CONDITIONED_PHASE_GUARD_GATE')
        vals=[]
        for v in node.op.params:
            f=float(v);require(math.isfinite(f),'FINITE_BOUND_COMPILER_PARAMS');vals.append(f.hex())
        row=(node.op.name,vals,[dag.find_bit(q).index for q in node.qargs],
             [dag.find_bit(q).index for q in node.cargs])
        out.update((repr(row)+'\n').encode())
    return out.hexdigest()

@contextmanager
def scalar_guard():
    from qiskit.circuit import Gate,Parameter
    from qiskit import QuantumCircuit
    from qiskit.circuit.equivalence import EquivalenceLibrary
    from qiskit.transpiler.passes import BasisTranslator
    require(threading.current_thread() is threading.main_thread(),'PRIVATE_SINGLE_THREADED_COMPILER')
    original=BasisTranslator.run;receipts=[]
    def repaired(self,dag):
        require(self._target is None and self._min_qubits==0,'FIXED_UNROUTED_TRANSLATOR')
        lib=self._equiv_lib;local=EquivalenceLibrary(base=lib)
        new=dag.copy_empty_like();names={}
        for node in dag.topological_op_nodes():
            op=node.op
            require(not getattr(op,'condition',None),'NO_CLASSICAL_CONTROL_IN_PHASE_GUARD')
            if op.params and op.name not in self._target_basis:
                require(not node.cargs and not op.is_parameterized(),'BOUND_NATIVE_GATE_ONLY')
                key=(op.name,op.num_qubits,len(op.params))
                if key not in names:
                    params=[Parameter('p'+str(i)) for i in range(len(op.params))]
                    scalar=Parameter('exact_zero_scalar')
                    source=op.to_mutable();source.params=params
                    eq=QuantumCircuit(op.num_qubits);eq.append(source,range(op.num_qubits))
                    eq.global_phase=scalar
                    name='h2_scalar_guard_'+op.name
                    local.add_equivalence(Gate(name,op.num_qubits,params+[scalar]),eq)
                    names[key]=name
                op=Gate(names[key],op.num_qubits,list(op.params)+[0.])
            new.apply_operation_back(op,node.qargs,node.cargs)
        new.global_phase=dag.global_phase
        baseline=original(self,dag)
        try:
            self._equiv_lib=local;fixed=original(self,new)
        finally:self._equiv_lib=lib
        before,after=stream_digest(baseline),stream_digest(fixed)
        require(before==after,'TRANSLATOR_GATE_ANGLE_WIRE_STREAM_CHANGED')
        receipts.append(dict(policy=POLICY,gate_angle_wire_stream_sha256=before,
          unchanged_gate_stream=True,original_phase=float(baseline.global_phase),
          guarded_phase=float(fixed.global_phase),scalar_parameter_value=0.,
          input_or_output_state_fit=False,postcompile_phase_correction=False))
        return fixed
    BasisTranslator.run=repaired
    try:yield receipts
    finally:BasisTranslator.run=original

def transpile(circuit,**settings):
    from qiskit import transpile as native
    require(settings.get('num_processes')==1,'NO_NESTED_COMPILER_PROCESSES')
    with scalar_guard() as receipts:result=native(circuit,**settings)
    result.metadata=dict(result.metadata or {},H2_phase_guard=dict(policy=POLICY,passes=receipts,
      unchanged_gate_stream=all(r['unchanged_gate_stream'] for r in receipts),
      state_fit=False,raw_validation_threshold_changed=False))
    return result
