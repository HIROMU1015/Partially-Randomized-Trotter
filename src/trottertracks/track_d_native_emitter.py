"""Opt-in stream sink for existing emitters; no transpile or molecular I/O.

Reuse standard gate definitions and the official one-qubit Euler decomposer.
This produces a phase-tracked native word, not a full-P0 price certificate.
"""
from dataclasses import dataclass
import math
from types import FunctionType

from .track_d_native_resources import NativeInstruction, _integer, _real, _mapping


class NativeLowerer:
    """Recursive official definitions, plus bounded 1q synthesis for U gates.

    No full circuit/DAG or new angle cache is built. Existing emitter/frame and
    standard definition caches remain owned by their original providers.
    """
    def __init__(self):
        import qiskit
        from qiskit.synthesis import OneQubitEulerDecomposer
        if qiskit.__version__ != '1.3.0':
            raise ValueError('qualified technical implementation requires Qiskit1.3.0')
        self._oneq = OneQubitEulerDecomposer('ZSXX')

    def emit(self, gate, qubits, clbits, consumer):
        qubits, clbits = tuple(qubits), tuple(clbits)
        if (getattr(gate, '_condition', None) is not None or getattr(gate, '_directive', False)
                or gate.is_parameterized()):
            raise ValueError('bound unconditional gate required')
        if gate.name in ('rz','sx','x','cx','measure'):
            from qiskit.circuit import Measure
            from qiskit.circuit.library import RZGate, SXGate, XGate, CXGate
            classes=dict(rz=RZGate,sx=SXGate,x=XGate,cx=CXGate,measure=Measure)
            if gate.base_class is not classes[gate.name]:raise ValueError('custom native-name gate')
            consumer(NativeInstruction(gate.name,qubits,clbits,tuple(gate.params)))
            return 0.
        from qiskit.circuit import ControlledGate
        standard=gate.base_class.__module__.startswith('qiskit.circuit.library.standard_gates.')
        controlled_standard=(isinstance(gate,ControlledGate) and
            gate.base_gate.base_class.__module__.startswith('qiskit.circuit.library.standard_gates.'))
        if clbits or not (standard or controlled_standard):
            raise ValueError('only standard unitary gate definitions are supported')
        definition=gate.definition
        if definition is not None:
            if definition.num_clbits:raise ValueError('classical definition unsupported')
            phase=float(definition.global_phase)
            for item in definition.data:
                mapped=tuple(qubits[definition.find_bit(q).index] for q in item.qubits)
                phase=math.fsum((phase,self.emit(item.operation,mapped,(),consumer)))
            return phase
        if gate.num_qubits != 1:
            raise ValueError('no standard finite definition; no dense multi-qubit synthesis')
        # Fundamental U has no recursive definition. Synthesis is 2x2 only.
        circuit=self._oneq(gate.to_matrix(),simplify=False)
        phase=float(circuit.global_phase)
        for item in circuit.data:
            phase=math.fsum((phase,self.emit(item.operation,qubits,(),consumer)))
        return phase


@dataclass(frozen=True)
class StreamBlock:
    """Replayable bounded descriptor, with an explicitly supplied actual inverse.

    Each callback emits local native instructions and returns its global phase.
    A callback may read chunks lazily; no gate word is stored by this interface.
    """
    num_qubits: int
    forward: object
    adjoint: object = None
    num_clbits: int = 0

    def inverse(self):
        if self.adjoint is None:raise ValueError('actual inverse emitter must be supplied')
        return StreamBlock(self.num_qubits,self.adjoint,self.forward,self.num_clbits)


def circuit_chunk(circuit, lowerer=None):
    """Reuse a caller-owned bounded chunk, including actual inverse gate order."""
    lowerer=lowerer or NativeLowerer()
    if circuit.parameters:raise ValueError('bound chunk required')
    def emit(consumer,inverse):
        if inverse and circuit.num_clbits:raise ValueError('unitary inverse chunk required')
        phase=-float(circuit.global_phase) if inverse else float(circuit.global_phase)
        data=reversed(circuit.data) if inverse else circuit.data
        for item in data:
            gate=item.operation.inverse() if inverse else item.operation
            q=tuple(circuit.find_bit(b).index for b in item.qubits)
            c=tuple(circuit.find_bit(b).index for b in item.clbits)
            phase=math.fsum((phase,lowerer.emit(gate,q,c,consumer)))
        return phase
    return StreamBlock(circuit.num_qubits,lambda emit_:emit(emit_,False),
                       lambda emit_:emit(emit_,True),circuit.num_clbits)


class NativeSink:
    """Small QuantumCircuit emission protocol; callback consumes each native op."""
    def __init__(self,num_qubits,num_clbits=0,*,consumer,lowerer=None):
        self.num_qubits=_integer(num_qubits,'num_qubits')
        self.num_clbits=_integer(num_clbits,'num_clbits')
        self.consumer=consumer; self.lowerer=lowerer or NativeLowerer()
        self._phase=0.; self.emitted=0

    @property
    def global_phase(self):return self._phase
    @global_phase.setter
    def global_phase(self,value):self._phase=_real(value,'global phase')

    def _send(self,op):
        if any(q>=self.num_qubits for q in op.qubits) or any(c>=self.num_clbits for c in op.clbits):
            raise ValueError('emitter wire outside declared register')
        self.consumer(op);self.emitted+=1

    def append(self,gate,qargs,cargs=(),*,copy=True):
        phase=self.lowerer.emit(gate,tuple(qargs),tuple(cargs or ()),self._send)
        self.global_phase=math.fsum((self.global_phase,phase))
        return self

    def compose(self,other,qubits=None,clbits=None,*,inplace=False,copy=True):
        if not inplace:raise ValueError('stream sink requires explicit inplace compose')
        block=other if isinstance(other,StreamBlock) else circuit_chunk(other,self.lowerer)
        q=_mapping(range(block.num_qubits) if qubits is None else qubits,
                   block.num_qubits,self.num_qubits,'compose qubits')
        c=_mapping(range(block.num_clbits) if clbits is None else clbits,
                   block.num_clbits,self.num_clbits,'compose clbits')
        def consume(op):
            self._send(NativeInstruction(op.name,tuple(q[i] for i in op.qubits),
                tuple(c[i] for i in op.clbits),op.params))
        self.global_phase=math.fsum((self.global_phase,_real(block.forward(consume),'block phase')))
        return self

    def rz(self,angle,q):
        from qiskit.circuit.library import RZGate
        return self.append(RZGate(angle),[q])
    def p(self,angle,q):
        from qiskit.circuit.library import PhaseGate
        return self.append(PhaseGate(angle),[q])
    def cx(self,a,b):
        from qiskit.circuit.library import CXGate
        return self.append(CXGate(),[a,b])
    def x(self,q):
        from qiskit.circuit.library import XGate
        return self.append(XGate(),[q])
    def sx(self,q):
        from qiskit.circuit.library import SXGate
        return self.append(SXGate(),[q])
    def rzz(self,angle,a,b):
        from qiskit.circuit.library import RZZGate
        return self.append(RZZGate(angle),[a,b])
    def h(self,q):
        from qiskit.circuit.library import HGate
        return self.append(HGate(),[q])
    def sdg(self,q):
        from qiskit.circuit.library import SdgGate
        return self.append(SdgGate(),[q])
    def cswap(self,a,b,c):
        from qiskit.circuit.library import CSwapGate
        return self.append(CSwapGate(),[a,b,c])
    def measure(self,q,c):
        return self._send(NativeInstruction('measure',(q,),(c,)))


def legacy_to_stream(emitter,*args,consumer,lowerer=None,**kwargs):
    """Bind the existing function's global constructor without patching a module.

    Supported legacy functions resolve QuantumCircuit from module globals.
    Functions importing it locally must instead provide a bounded circuit chunk.
    Their code/other global dependencies remain the original objects. No source
    rewriting, global monkey patch, general builder copy, or import hook is used.
    """
    from qiskit import QuantumCircuit
    if (not isinstance(emitter,FunctionType) or emitter.__globals__.get('QuantumCircuit') is not QuantumCircuit
            or 'QuantumCircuit' in emitter.__code__.co_varnames):
        raise ValueError('use bounded circuit_chunk for a locally importing emitter')
    created=[]
    def factory(*registers,**options):
        if options or len(registers) not in (1,2):raise ValueError('integer-register emitter protocol only')
        sink=NativeSink(*registers,consumer=consumer,lowerer=lowerer)
        created.append(sink);return sink
    namespace=dict(emitter.__globals__,QuantumCircuit=factory)
    bound=FunctionType(emitter.__code__,namespace,emitter.__name__,emitter.__defaults__,emitter.__closure__)
    bound.__kwdefaults__=emitter.__kwdefaults__
    result=bound(*args,**kwargs)
    if len(created)!=1 or result is not created[0]:raise ValueError('one returned emission sink required')
    return result
