# Frozen artificial-only oracle from 8f77bebf circuits.py through serialize.
"""Exact Qiskit serialization and full controlled Gaussian/Pauli wrappers."""
import math
from trottertracks.resource_applicability.h4_geometry.identity import Stop, require, exact, fingerprint

LEAVES = frozenset('id x y z h s sdg t tdg sx sxdg rx ry rz p u cx cy cz ch crx cry crz cp cu swap cswap ccx rzx rxx ryy rzz dcx ecr'.split())


def number(value, _arrays=None):
    from qiskit.circuit import ParameterExpression
    require(not isinstance(value, ParameterExpression), 'symbolic parameter (even bound expressions must be numeric)')
    import numpy as np
    if isinstance(value, np.ndarray):
        require(value.dtype.kind in 'fciub', 'matrix parameter dtype')
        if _arrays is not None and id(value) in _arrays:
            return _arrays[id(value)][1]
        encoded = {'array':number(value.tolist(),_arrays),'shape':list(value.shape),'dtype':value.dtype.str}
        if _arrays is not None:
            # Retain the original array so an id cannot be recycled mid-call.
            _arrays[id(value)] = (value,encoded)
        return encoded
    if isinstance(value,(list,tuple)):
        return [number(x,_arrays) for x in value]
    if isinstance(value,np.generic):
        value = value.item()
    return exact(value)


def decode(value):
    if isinstance(value,list):
        return [decode(x) for x in value]
    if isinstance(value,dict):
        if set(value)=={'real64_hex'}:
            v = float.fromhex(value['real64_hex']);require(math.isfinite(v),'nonfinite decoded parameter');return v
        if set(value)=={'complex128_hex'}:
            a,b=map(float.fromhex,value['complex128_hex']);require(math.isfinite(a) and math.isfinite(b),'nonfinite decoded parameter');return complex(a,b)
        if set(value)=={'array','shape','dtype'}:
            import numpy as np
            return np.asarray(decode(value['array']),dtype=value['dtype']).reshape(value['shape'])
    return value


def serialize(circuit, axis, _stack=(), _arrays=None):
    if _arrays is None:
        _arrays = {}
    require(axis in ('cosine','sine'), 'measurement axis')
    require(id(circuit) not in _stack and len(_stack)<64, 'recursive custom definition')
    stack = (*_stack,id(circuit))
    def condition(op):
        cond = getattr(op,'condition',None)
        if cond is None:
            return None
        bits,val = cond
        from qiskit.circuit import Clbit, ClassicalRegister
        if isinstance(bits,Clbit):
            target = {'clbit':circuit.find_bit(bits).index}
        elif isinstance(bits,ClassicalRegister):
            target = {'register':[circuit.find_bit(b).index for b in bits], 'name':bits.name}
        else:
            raise Stop('unsupported expression condition')
        return {'target':target,'value':int(val)}
    instructions=[]
    for item in circuit.data:
        op=item.operation
        base=op.base_class
        entry=dict(name=op.name, num_qubits=op.num_qubits, num_clbits=op.num_clbits,
                   parameters=[number(p,_arrays) for p in op.params],
                   qubits=[circuit.find_bit(b).index for b in item.qubits],
                   clbits=[circuit.find_bit(b).index for b in item.clbits],
                   condition=condition(op), ctrl_state=getattr(op,'ctrl_state',None),
                   num_ctrl_qubits=getattr(op,'num_ctrl_qubits',None), label=op.label)
        closed_name=op.name.rsplit('_o',1)[0] if getattr(op,'ctrl_state',None) is not None else op.name
        if closed_name in LEAVES and base.__module__.startswith('qiskit.circuit.library.standard_gates'):
            entry['kind']='standard'
            entry['standard_name']=closed_name
        elif op.name in ('measure','reset','barrier') and base.__module__.startswith('qiskit.circuit.'):
            entry['kind']='instruction'
        elif base.__name__=='UnitaryGate' and base.__module__.startswith('qiskit.circuit.library.generalized_gates.unitary'):
            entry['kind']='unitary'
        else:
            from qiskit.circuit import Gate, Instruction, ControlledGate
            require(type(op) in (Gate,Instruction,ControlledGate) and op.definition is not None, 'unsupported operation class')
            entry['kind']='definition'
            entry['gate_class']='controlled' if isinstance(op,ControlledGate) else ('gate' if isinstance(op,Gate) else 'instruction')
            if isinstance(op,ControlledGate):
                from qiskit import QuantumCircuit
                closed=op.copy().to_mutable()
                closed.ctrl_state=2**op.num_ctrl_qubits-1
                entry['closed_name']=closed.name
                entry['definition']=serialize(closed.definition,axis,stack,_arrays)
                entry['effective_definition']=serialize(op.definition,axis,stack,_arrays)
                base_qc=QuantumCircuit(op.base_gate.num_qubits)
                base_qc.append(op.base_gate,range(op.base_gate.num_qubits))
                entry['base_gate']=serialize(base_qc,axis,stack,_arrays)
            else:
                entry['definition']=serialize(op.definition,axis,stack,_arrays)
        instructions.append(entry)
    return dict(format='ordered-numerical-full-circuit-v1',axis=axis,
                qubits=list(range(circuit.num_qubits)),clbits=list(range(circuit.num_clbits)),
                qregs=[{'name':r.name,'bits':[circuit.find_bit(b).index for b in r]} for r in circuit.qregs],
                cregs=[{'name':r.name,'bits':[circuit.find_bit(b).index for b in r]} for r in circuit.cregs],
                global_phase=number(circuit.global_phase),instructions=instructions)
