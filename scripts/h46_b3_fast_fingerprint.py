"""Emit the unchanged canonical v2 SHA using call-local standard-gate bytes.

The original recursive operation writer produces every cached payload. Current
content keys inspect custom and explicit mutable definitions before byte reuse.
No circuit identity or digest survives a call. The input remains immutable
during a call, as required by the original writer. Compilation is untouched.
"""
from contextlib import contextmanager
import importlib
import math
import threading
import numpy as np
from trottertracks.resource_applicability import ax2b_stream_fingerprint_v5 as legacy
from trottertracks.resource_applicability.h6_hash_reuse_v1 import _LAZY_STANDARD as _BASE_LAZY,_ALIASES
from qiskit.circuit.library import XXPlusYYGate,RXXGate,RYYGate,RZZGate
from trotterlib.rte_compiled_cost import _numeric_parameter_payload,_condition_payload

_LAZY_STANDARD=dict(_BASE_LAZY)
_LAZY_STANDARD.update({cls:cls._define for cls in (XXPlusYYGate,RXXGate,RYYGate,RZZGate)})


def parameter_key(value):
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,bool): return ('bool',value)
    if isinstance(value,int): return ('int',value)
    if isinstance(value,float):
        if not math.isfinite(value): raise ValueError('Circuit parameters must be finite.')
        return ('float',value.hex())
    return None


def freeze(value):
    if isinstance(value,dict): return tuple((k,freeze(v)) for k,v in sorted(value.items()))
    if isinstance(value,(list,tuple)): return tuple(freeze(v) for v in value)
    return value


class ContentKeys:
    """Call-local content interning; replaces JSON/SHA guards, not final SHA."""
    def __init__(self):
        self.memo={};self.active=set();self.interned={};self.lazy={}

    def intern(self,record):
        old=self.interned.get(record)
        if old is not None:return old
        if len(self.interned)>=65536:return record
        value=len(self.interned)+1;self.interned[record]=value;return value

    def numeric(self,value):
        key=parameter_key(value)
        if key is not None:return key
        if isinstance(value,np.ndarray) and np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():
            raise ValueError('Circuit parameters must be finite.')
        return ('numeric',freeze(_numeric_parameter_payload(value)))

    def node(self,key,references,build):
        if key in self.active:raise ValueError('Recursive custom-gate definitions are not cacheable.')
        old=self.memo.get(key)
        if old is not None:
            assert all(a is b for a,b in zip(old[0],references))
            return old[1]
        self.active.add(key)
        try:value=build()
        finally:self.active.remove(key)
        if len(self.memo)<262144:self.memo[key]=(references,value)
        return value

    def circuit(self,circuit):
        def build():
            if circuit.parameters:raise ValueError('Symbolic circuits are not cacheable.')
            if getattr(circuit,'layout',None) is not None:raise ValueError('Circuit layout is not cacheable.')
            if getattr(circuit,'calibrations',{}):raise ValueError('Circuit calibrations are not cacheable.')
            items=tuple((tuple(circuit.find_bit(q).index for q in x.qubits),
                tuple(circuit.find_bit(c).index for c in x.clbits),self.operation(x.operation,circuit)) for x in circuit.data)
            return self.intern(('circuit',circuit.num_qubits,circuit.num_clbits,self.numeric(circuit.global_phase),items))
        return self.node(('circuit',id(circuit)),(circuit,),build)

    def operation(self,op,circuit):
        def build():
            cls=type(op);base=getattr(op,'base_gate',None);blocks=getattr(op,'blocks',())
            fields=(cls,op.name,op.num_qubits,op.num_clbits,tuple(self.numeric(v) for v in op.params),
                freeze(_condition_payload(getattr(op,'condition',None),circuit)),
                int(op.ctrl_state) if hasattr(op,'ctrl_state') else None,
                int(op.num_ctrl_qubits) if hasattr(op,'num_ctrl_qubits') else None,
                self.operation(base,circuit) if base is not None else None,
                tuple(self.circuit(c) for c in blocks))
            plain=(cls in _LAZY_STANDARD and cls._define is _LAZY_STANDARD[cls]
                and '_define' not in vars(op) and vars(op).get('_definition') is None
                and base is None and not blocks)
            if plain and fields in self.lazy:return self.lazy[fields]
            definition=getattr(op,'definition',None)
            value=self.intern(('operation',fields,self.circuit(definition) if definition is not None else None))
            if plain and len(self.lazy)<65536:self.lazy[fields]=value
            return value
        return self.node(('operation',id(op),id(circuit)),(op,circuit),build)


class FastWriter(legacy._Writer):
    def __init__(self,sink):
        super().__init__(sink)
        self.operations={};self.operation_bytes=0
        self.hits=0;self.misses=0
        self.keys=ContentKeys()

    def _circuit(self,circuit):
        if circuit.parameters:raise ValueError('Symbolic circuits are not cacheable.')
        if getattr(circuit,'layout',None) is not None:raise ValueError('Circuit layout is not cacheable.')
        if getattr(circuit,'calibrations',{}):raise ValueError('Circuit calibrations are not cacheable.')
        qindex={q:i for i,q in enumerate(circuit.qubits)}
        cindex={c:i for i,c in enumerate(circuit.clbits)}
        positions_q={};positions_c={}
        self.sink.write(b'{"fingerprint_policy":"qiskit_recursive_numeric_circuit_v2","global_phase":')
        self.scalar(_numeric_parameter_payload(circuit.global_phase))
        self.sink.write(b',"instructions":[')
        for index,item in enumerate(circuit.data):
            qb=positions_q.get(item.qubits)
            if qb is None:
                qb=legacy._ENCODER.encode([qindex[q] for q in item.qubits]).encode();positions_q[item.qubits]=qb
            cb=positions_c.get(item.clbits)
            if cb is None:
                cb=legacy._ENCODER.encode([cindex[c] for c in item.clbits]).encode();positions_c[item.clbits]=cb
            self.sink.write((b',' if index else b'')+b'{"clbits":'+cb+b',"operation":')
            self.operation(item.operation,circuit)
            self.sink.write(b',"qubits":'+qb+b'}')
        self.sink.write(b'],"num_clbits":');self.scalar(int(circuit.num_clbits))
        self.sink.write(b',"num_qubits":');self.scalar(int(circuit.num_qubits));self.sink.write(b'}')

    def operation(self,op,circuit):
        key=self.keys.operation(op,circuit)
        cached=self.operations.get(key)
        if cached is not None:
            self.hits+=1;self.sink.write(cached[1]);return
        self.misses+=1
        if self.operation_bytes>=64*2**20:
            return super().operation(op,circuit)
        parent=self.sink;capture=legacy._Capture(parent);self.sink=capture
        try: super().operation(op,circuit)
        finally: self.sink=parent
        if not capture.overflow:
            payload=bytes(capture.buffer)
            self.operations[key]=(None,payload);self.operation_bytes+=len(payload)


class FingerprintSession:
    def __init__(self): self.calls=0;self.hits=0;self.misses=0

    def fingerprint(self,circuit):
        with legacy.fingerprint_stage(circuit):
            sink=legacy._HashSink();writer=FastWriter(sink)
            writer.circuit(circuit);sink.flush()
            self.calls+=1;self.hits+=writer.hits;self.misses+=writer.misses
            return sink.hasher.hexdigest()


@contextmanager
def use_fast_fingerprint():
    if threading.current_thread() is not threading.main_thread() or threading.active_count()!=1:
        raise ValueError('HASH_REUSE_SINGLE_THREAD_REQUIRED')
    modules=[importlib.import_module('.'+name,'trottertracks.resource_applicability') for name in _ALIASES]
    if any(m.canonical_qiskit_circuit_fingerprint is not legacy.canonical_qiskit_circuit_fingerprint for m in modules):
        raise ValueError('HASH_REUSE_ALIAS_ALREADY_CHANGED')
    session=FingerprintSession()
    try:
        for module in modules: module.canonical_qiskit_circuit_fingerprint=session.fingerprint
        yield session
    finally:
        for module in modules: module.canonical_qiskit_circuit_fingerprint=legacy.canonical_qiskit_circuit_fingerprint
