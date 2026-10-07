"""Exact Qiskit serialization and full controlled Gaussian/Pauli wrappers."""
import math
from .identity import Stop, require, exact, fingerprint
from .streaming import LazyList, array_values

LEAVES = frozenset('id x y z h s sdg t tdg sx sxdg rx ry rz p u cx cy cz ch crx cry crz cp cu swap cswap ccx rzx rxx ryy rzz dcx ecr'.split())


def number(value, _arrays=None):
    from qiskit.circuit import ParameterExpression
    require(not isinstance(value, ParameterExpression), 'symbolic parameter (even bound expressions must be numeric)')
    import numpy as np
    if isinstance(value, np.ndarray):
        require(value.dtype.kind in 'fciub', 'matrix parameter dtype')
        if _arrays is not None and id(value) in _arrays:
            return _arrays[id(value)][1]
        encoded = {'array':array_values(value),'shape':list(value.shape),'dtype':value.dtype.str}
        if _arrays is not None and len(_arrays) < 64:
            # Retain the original array so an id cannot be recycled mid-call.
            _arrays[id(value)] = (value,encoded)
        return encoded
    if isinstance(value,(list,tuple)):
        return LazyList(lambda: (number(x,_arrays) for x in value))
    if isinstance(value,np.generic):
        value = value.item()
    return exact(value)


def decode(value):
    if isinstance(value,(list,LazyList)):
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
    def entries():
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
            yield entry
    instructions=LazyList(entries)
    return dict(format='ordered-numerical-full-circuit-v1',axis=axis,
                qubits=list(range(circuit.num_qubits)),clbits=list(range(circuit.num_clbits)),
                qregs=[{'name':r.name,'bits':[circuit.find_bit(b).index for b in r]} for r in circuit.qregs],
                cregs=[{'name':r.name,'bits':[circuit.find_bit(b).index for b in r]} for r in circuit.cregs],
                global_phase=number(circuit.global_phase),instructions=instructions)


def deserialize(record):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit, Clbit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import UnitaryGate
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
    qc=QuantumCircuit()
    qs=[Qubit() for _ in record['qubits']];cs=[Clbit() for _ in record['clbits']]
    qc.add_bits(qs);qc.add_bits(cs)
    for r in record['qregs']:
        qc.add_register(QuantumRegister(bits=[qs[i] for i in r['bits']],name=r['name']))
    for r in record['cregs']:
        qc.add_register(ClassicalRegister(bits=[cs[i] for i in r['bits']],name=r['name']))
    qc.global_phase=decode(record['global_phase'])
    standard=get_standard_gate_name_mapping()
    for item in record['instructions']:
        params=[decode(p) for p in item['parameters']]
        if item['kind']=='standard':
            template=standard[item['standard_name']]
            op=template.base_class(*params) if params else template.copy()
            if item['ctrl_state'] is not None:
                op=op.to_mutable();op.ctrl_state=item['ctrl_state']
        elif item['kind']=='unitary':
            op=UnitaryGate(params[0])
        elif item['kind']=='definition':
            from qiskit.circuit import Gate, Instruction, ControlledGate
            definition=deserialize(item['definition'])
            if item['gate_class']=='controlled':
                base=deserialize(item['base_gate']).data[0].operation
                op=ControlledGate(item['closed_name'],item['num_qubits'],params,num_ctrl_qubits=item['num_ctrl_qubits'],
                                  definition=definition,ctrl_state=item['ctrl_state'],base_gate=base)
            elif item['gate_class']=='gate':
                op=Gate(item['name'],item['num_qubits'],params);op.definition=definition
            else:
                op=Instruction(item['name'],item['num_qubits'],item['num_clbits'],params);op.definition=definition
        elif item['name']=='measure':
            from qiskit.circuit import Measure
            op=Measure()
        elif item['name']=='reset':
            from qiskit.circuit import Reset
            op=Reset()
        elif item['name']=='barrier':
            from qiskit.circuit import Barrier
            op=Barrier(item['num_qubits'])
        else:
            raise Stop('unsupported serialized operation')
        op=op.to_mutable();op.label=item['label']
        cond=item['condition']
        if cond is not None:
            t=cond['target']
            target=cs[t['clbit']] if 'clbit' in t else next(r for r in qc.cregs if r.name==t['name'])
            op.condition=(target,cond['value'])
        qc.append(op,[qs[i] for i in item['qubits']],[cs[i] for i in item['clbits']])
    return qc


def numerical_fingerprint(circuit, axis):
    return fingerprint('h4-numerical-circuit-v1',serialize(circuit,axis))


def gaussian_basis(U):
    """Pinned server's dense JW Gaussian fallback; no alternate compiler policy."""
    import numpy as np
    from scipy.linalg import logm, expm
    from .inputs import one_body_operator, reverse_bits
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    require(U.shape[0] <= 8, 'Gaussian dense scope')
    anti=logm(U);anti=(anti-anti.conj().T)*0.5
    mat=expm(one_body_operator(anti))
    perm=[reverse_bits(i,U.shape[0]) for i in range(2**U.shape[0])]
    qc=QuantumCircuit(U.shape[0])
    qc.append(UnitaryGate(mat[np.ix_(perm,perm)]),range(U.shape[0]))
    return qc


def diagonal_coefficients(eta, lam=None):
    """Exact I/Z/ZZ expansion, exact-zero pruning only."""
    import numpy as np
    eta=np.asarray(eta,dtype=float)
    if lam is None:
        return {():float(np.sum(eta))/2,**{(k,):float(-x/2) for k,x in enumerate(eta)}}
    result={():float(lam*np.dot(eta,eta)/2)}
    for i,x in enumerate(eta):
        result[(i,)]=float(-lam*x*np.sum(eta)/2)
        for j in range(i+1,len(eta)):
            pair=float(lam*x*eta[j]/2)
            result[()]+=pair;result[(i,j)]=pair
    return result


def diagonalize_block(matrix, lam=None, *, basis_builder=gaussian_basis):
    import numpy as np
    values,U=np.linalg.eigh(matrix)
    order=np.argsort(np.abs(values))[::-1]
    return {'basis':basis_builder(U[:,order]),'coefficients':diagonal_coefficients(values[order],lam)}


def append_component(qc, block, support, angle, *, product=False, sign=1):
    from qiskit.circuit.library import RZGate, RZZGate
    n=qc.num_qubits-1;anc=n
    basis=block['basis']
    qc.compose(basis.inverse(),range(n),inplace=True)
    if product:
        for p in support:
            qc.cz(anc,p)
        if sign<0:
            qc.p(math.pi,anc)
    elif not support:
        qc.p(-angle*sign,anc)
    elif len(support)==1:
        qc.append(RZGate(2*angle*sign).control(),[anc,support[0]])
    elif len(support)==2:
        qc.append(RZZGate(2*angle*sign).control(),[anc,*support])
    else:
        raise Stop('unsupported diagonal support')
    qc.compose(basis,range(n),inplace=True)


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


def append_block(qc,block,time_value):
    from qiskit.circuit.library import RZGate,RZZGate
    n=qc.num_qubits-1;basis=block['basis']
    qc.compose(basis.inverse(),range(n),inplace=True)
    for support,c in block['coefficients'].items():
        if c != 0:
            angle=time_value*c
            if not support:
                qc.p(-angle,n)
            elif len(support)==1:
                qc.append(RZGate(2*angle).control(),[n,support[0]])
            elif len(support)==2:
                qc.append(RZZGate(2*angle).control(),[n,*support])
            else:
                raise Stop('unsupported diagonal support')
    qc.compose(basis,range(n),inplace=True)


def wrapper(evolution,axis):
    from qiskit import QuantumCircuit
    require(axis in ('cosine','sine'), 'axis')
    qc=QuantumCircuit(evolution.num_qubits,1);anc=evolution.num_qubits-1
    qc.h(anc);qc.compose(evolution,inplace=True)
    if axis=='sine':
        qc.sdg(anc)
    qc.h(anc);qc.measure(anc,0)
    return qc


def metrics(compiled):
    counts=compiled.count_ops()
    def depth(name):
        return compiled.depth(filter_function=lambda item:item.operation.name==name)
    return {'rz_count':int(counts.get('rz',0)),'rz_depth':int(depth('rz')),
            'cx_count':int(counts.get('cx',0)),'cx_depth':int(depth('cx')),
            'total_depth':int(compiled.depth()),'circuit_size':int(compiled.size())}
