"""Frozen resume01 oracle; artificial-only, no science or compile execution."""
import math
from trottertracks.resource_applicability.h4_geometry.identity import Stop, require, exact, fingerprint
from trottertracks.resource_applicability.h4_geometry.streaming import LazyList, array_values

LEAVES = frozenset('id x y z h s sdg t tdg sx sxdg rx ry rz p u cx cy cz ch crx cry crz cp cu swap cswap ccx rzx rxx ryy rzz dcx ecr'.split())

def number(value, _arrays=None, _encoding_memo=None):
    from qiskit.circuit import ParameterExpression
    require(not isinstance(value, ParameterExpression), 'symbolic parameter (even bound expressions must be numeric)')
    import numpy as np
    if isinstance(value, np.ndarray):
        require(value.dtype.kind in 'fciub', 'matrix parameter dtype')
        if _encoding_memo is not None:
            encoded = _encoding_memo.encode(value)
            if encoded is not None:
                return encoded
        if _arrays is not None and id(value) in _arrays:
            return _arrays[id(value)][1]
        encoded = {'array':array_values(value),'shape':list(value.shape),'dtype':value.dtype.str}
        if _arrays is not None and len(_arrays) < 64:
            # Retain the original array so an id cannot be recycled mid-call.
            _arrays[id(value)] = (value,encoded)
        return encoded
    if isinstance(value,(list,tuple)):
        return LazyList(lambda: (number(x,_arrays,_encoding_memo) for x in value))
    if isinstance(value,np.generic):
        value = value.item()
    return exact(value)

def serialize(circuit, axis, _stack=(), _arrays=None, _encoding_memo=None):
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
    def entries():
        for item in circuit.data:
            op=item.operation
            base=op.base_class
            entry=dict(name=op.name, num_qubits=op.num_qubits, num_clbits=op.num_clbits,
                       parameters=[number(p,_arrays,_encoding_memo) for p in op.params],
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
                    entry['definition']=serialize(closed.definition,axis,stack,_arrays,_encoding_memo)
                    entry['effective_definition']=serialize(op.definition,axis,stack,_arrays,_encoding_memo)
                    base_qc=QuantumCircuit(op.base_gate.num_qubits)
                    base_qc.append(op.base_gate,range(op.base_gate.num_qubits))
                    entry['base_gate']=serialize(base_qc,axis,stack,_arrays,_encoding_memo)
                else:
                    entry['definition']=serialize(op.definition,axis,stack,_arrays,_encoding_memo)
            yield entry
    instructions=LazyList(entries)
    return dict(format='ordered-numerical-full-circuit-v1',axis=axis,
                qubits=list(range(circuit.num_qubits)),clbits=list(range(circuit.num_clbits)),
                qregs=[{'name':r.name,'bits':[circuit.find_bit(b).index for b in r]} for r in circuit.qregs],
                cregs=[{'name':r.name,'bits':[circuit.find_bit(b).index for b in r]} for r in circuit.cregs],
                global_phase=number(circuit.global_phase),instructions=instructions)

def numerical_fingerprint(circuit, axis, *, encoding_memo=None):
    return fingerprint('h4-numerical-circuit-v1',serialize(circuit,axis,_encoding_memo=encoding_memo))

def build_evolution(preparation, template, events):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZGate,RZZGate
    n=preparation['n'];qc=QuantumCircuit(n+1)
    delta=template['T']/template['q']
    require(delta==template['delta'], 'PF delta')
    active=None
    def use_basis(block):
        nonlocal active
        basis=block['basis']
        if active is basis:
            return
        if active is not None:
            qc.compose(active,range(n),inplace=True)
        qc.compose(basis.inverse(),range(n),inplace=True)
        active=basis
    def apply_component(block,support,angle,product=False,sign=1):
        use_basis(block)
        if product:
            for p in support:qc.cz(n,p)
            if sign<0:qc.p(math.pi,n)
        elif not support:qc.p(-angle*sign,n)
        elif len(support)==1:qc.append(RZGate(2*angle*sign).control(),[n,support[0]])
        elif len(support)==2:qc.append(RZZGate(2*angle*sign).control(),[n,*support])
        else:raise Stop('unsupported diagonal support')
    def block_evolution(block,time_value):
        for support,c in block['coefficients'].items():
            if c!=0:apply_component(block,support,time_value*c)
    # Shared registered full basis across adjacent applications. Ancilla scalar
    # phases commute with system basis changes and do not break this boundary.
    for outer in range(template['q']):
        qc.p(-delta*preparation['constant'],n)
        for b in preparation['deterministic']:
            block_evolution(b,delta/2)
        for event in events[outer]:
            if event['order']//2%2:
                qc.p(math.pi,n)
            for c in event['products']:
                apply_component(c['block'],c['support'],0,product=True,sign=c['sign'])
            c=event['rotation']
            apply_component(c['block'],c['support'],event['angle'],sign=c['sign'])
        for b in reversed(preparation['deterministic']):
            block_evolution(b,delta/2)
    if active is not None:
        qc.compose(active,range(n),inplace=True)
    return qc

def wrapper(evolution,axis):
    from qiskit import QuantumCircuit
    require(axis in ('cosine','sine'), 'axis')
    qc=QuantumCircuit(evolution.num_qubits,1);anc=evolution.num_qubits-1
    qc.h(anc);qc.compose(evolution,inplace=True)
    if axis=='sine':
        qc.sdg(anc)
    qc.h(anc);qc.measure(anc,0)
    return qc
