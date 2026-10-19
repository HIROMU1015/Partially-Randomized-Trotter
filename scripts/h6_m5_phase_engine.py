"""Analytic little-endian gate replay; no transpile, synthesis or phase correction.

Extended kernels never call Qiskit to_matrix and preserve serialized binary64
parameters by exact promotion. Only 1/2-qubit matrices and state-sized indices.
"""
from functools import lru_cache
import math
import numbers
import numpy as np

WRAPPER = frozenset(('h','p','rz','rzz','cx','xx_plus_yy','id','barrier','measure'))
COMPILED = frozenset(('rz','sx','x','cx','id','barrier','measure'))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def precision():
    a, b = np.finfo(np.longdouble), np.finfo(np.float64)
    extra = np.longdouble(1)+np.longdouble(b.eps)/2 > np.longdouble(1)
    require(a.nmant > b.nmant and bool(extra), 'EXTENDED_PRECISION_NOT_WIDER_NO_FALLBACK')
    return dict(binary64_mantissa_bits=b.nmant+1, extended_mantissa_bits=a.nmant+1,
        longdouble_itemsize=np.dtype(np.longdouble).itemsize,
        clongdouble_itemsize=np.dtype(np.clongdouble).itemsize, extra_precision_verified=True)


def parameter(value):
    require(isinstance(value, numbers.Real) and not isinstance(value, (bool,np.bool_)) and
        math.isfinite(float(value)), 'FINITE_REAL_BINARY64_PARAMETER')
    require(np.longdouble(value) == np.longdouble(float(value)), 'SERIALIZED_BINARY64_ONLY')
    return float(value)


def validate_circuit(qc, kind, nqubits):
    """Reject custom lookalike gates and nonterminal/classical operations."""
    from qiskit.circuit.library import HGate, PhaseGate, RZGate, RZZGate, CXGate, XXPlusYYGate, IGate, SXGate, XGate
    from qiskit.circuit import Barrier, Measure
    classes = dict(h=HGate,p=PhaseGate,rz=RZGate,rzz=RZZGate,cx=CXGate,
        xx_plus_yy=XXPlusYYGate,id=IGate,sx=SXGate,x=XGate,barrier=Barrier,measure=Measure)
    require(kind in ('WRAPPER','COMPILED'), 'REGISTERED_CIRCUIT_KIND')
    require(qc.num_qubits == nqubits and qc.num_clbits == 1 and qc.layout is None, 'QUBITS_CLBITS_LAYOUT')
    require(not qc.parameters, 'NO_SYMBOLIC_CIRCUIT_PARAMETERS')
    parameter(qc.global_phase)
    counts = {}; measured = False
    for item in qc.data:
        op = item.operation; name = op.name
        require(name in (WRAPPER if kind == 'WRAPPER' else COMPILED), 'UNKNOWN_GATE:'+name)
        require(op.base_class is classes[name], 'NO_CUSTOM_GATE_LOOKALIKE:'+name)
        require(getattr(op,'condition',None) is None, 'NO_CLASSICAL_CONDITION')
        for value in op.params:
            parameter(value)
        counts[name] = counts.get(name,0)+1
        if name == 'measure':
            require(not measured and len(item.qubits) == len(item.clbits) == 1 and
                qc.find_bit(item.qubits[0]).index == nqubits-1 and
                qc.find_bit(item.clbits[0]).index == 0, 'FINAL_ANCILLA_MEASUREMENT_ONLY')
            measured = True
        else:
            require(not measured or name == 'barrier', 'NO_MID_CIRCUIT_MEASUREMENT')
            require(not item.clbits, 'NO_CLASSICAL_ARGUMENTS')
    require(measured, 'EXACT_ONE_FINAL_MEASUREMENT')
    return dict(nqubits=nqubits,ancilla=nqubits-1,layout='None',gate_counts=counts,
        global_phase=parameter(qc.global_phase),final_measurements_removed_only=True)


def finite_vector(v, length):
    require(v.shape == (length,) and np.isfinite(v).all(), 'FINITE_VECTOR_SHAPE')


def ordinary(qc, low):
    """Exact recovery simulator convention, including its separate global phase."""
    from qiskit.quantum_info import Statevector
    circuit = qc.remove_final_measurements(inplace=False)
    phase = float(circuit.global_phase or 0.)
    if phase != 0:
        circuit = circuit.copy(); circuit.global_phase = 0.
    vec = np.asarray(Statevector(low).evolve(circuit).data)
    if phase != 0:
        vec = np.exp(1j*phase)*vec
    finite_vector(vec,len(low))
    return vec


def cis(angle):
    angle = np.longdouble(angle)
    return np.clongdouble(np.cos(angle)) + np.clongdouble(1j)*np.sin(angle)


@lru_cache(maxsize=4096)
def kernel(name, params):
    """Exact-value cached small analytic matrices (not rounded matrix imports)."""
    t = tuple(np.longdouble(p) for p in params); j = np.clongdouble(1j)
    if name == 'h':
        return np.array([[1,1],[1,-1]],dtype=np.clongdouble)/np.sqrt(np.longdouble(2))
    if name == 'sx':
        return np.array([[1+j,1-j],[1-j,1+j]],dtype=np.clongdouble)/2
    if name == 'x':
        return np.array([[0,1],[1,0]],dtype=np.clongdouble)
    if name == 'cx':
        return np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],dtype=np.clongdouble)
    if name in ('rz','p','rzz'):
        if name == 'p':
            return np.array([1,cis(t[0])],dtype=np.clongdouble)
        a,b=cis(-t[0]/2),cis(t[0]/2)
        return np.array([a,b] if name == 'rz' else [a,b,b,a],dtype=np.clongdouble)
    if name == 'xx_plus_yy':
        co,si=np.cos(t[0]/2),np.sin(t[0]/2)
        m=np.eye(4,dtype=np.clongdouble)
        m[1,1]=m[2,2]=co
        m[1,2]=-j*si*cis(-t[1]);m[2,1]=-j*si*cis(t[1])
        return m
    raise ValueError('UNKNOWN_ANALYTIC_GATE:'+name)


def extended(qc, low, *, progress=None, cancelled=None):
    precision(); n=qc.num_qubits
    finite_vector(np.asarray(low),2**n)
    v=np.asarray(low,dtype=np.clongdouble).copy()
    all_indices=np.arange(len(v),dtype=np.int64); indices={}
    for i,item in enumerate(qc.data):
        if i % 16384 == 0:
            require(cancelled is None or not cancelled(), 'OWN_REPLAY_SIGNAL')
            if progress: progress(i,len(qc.data))
        op=item.operation;name=op.name
        if name in ('measure','barrier','id'): continue
        qs=tuple(qc.find_bit(q).index for q in item.qubits)
        params=tuple(parameter(p) for p in op.params)
        m=kernel(name,params)
        if qs not in indices:
            mask=sum(1<<q for q in qs)
            base=all_indices[(all_indices & mask)==0]
            indices[qs]=tuple(base+sum(((k>>b)&1)<<q for b,q in enumerate(qs)) for k in range(2**len(qs)))
        ix=indices[qs]
        if m.ndim == 1:
            for a,idx in zip(m,ix): v[idx] *= a
        elif name in ('cx','x'):
            a,b=(1,3) if name=='cx' else (0,1)
            tmp=v[ix[a]].copy();v[ix[a]]=v[ix[b]];v[ix[b]]=tmp
        elif name == 'xx_plus_yy':
            a,b=v[ix[1]].copy(),v[ix[2]].copy()
            v[ix[1]]=m[1,1]*a+m[1,2]*b;v[ix[2]]=m[2,1]*a+m[2,2]*b
        else:
            a,b=v[ix[0]].copy(),v[ix[1]].copy()
            v[ix[0]]=m[0,0]*a+m[0,1]*b;v[ix[1]]=m[1,0]*a+m[1,1]*b
    v *= cis(parameter(qc.global_phase))
    finite_vector(v,len(low))
    return v


def norm(v):
    return np.sqrt(np.sum(v.real*v.real+v.imag*v.imag))


def comparison(before, after, *, expected_hadamard=None, extended_metrics=False):
    dtype=np.clongdouble if extended_metrics else np.complex128
    a,b=np.asarray(before,dtype=dtype),np.asarray(after,dtype=dtype)
    finite_vector(a,len(a));finite_vector(b,len(a))
    overlap=np.sum(np.conj(a)*b) if extended_metrics else np.vdot(a,b)
    angle=np.arctan2(overlap.imag,overlap.real) if overlap!=0 else None
    nrm=norm if extended_metrics else np.linalg.norm
    alignment=cis(-angle) if extended_metrics and angle is not None else (np.exp(-1j*angle) if angle is not None else None)
    half=len(a)//2
    def observable(v):
        return (np.sum(v[:half].real**2+v[:half].imag**2)-np.sum(v[half:].real**2+v[half:].imag**2)
            if extended_metrics else np.vdot(v[:half],v[:half]).real-np.vdot(v[half:],v[half:]).real)
    def branch_phase(x,y):
        o=np.sum(np.conj(x)*y)
        return float(np.arctan2(o.imag,o.real)) if o!=0 else None
    raw=float(nrm(b-a))
    result=dict(raw_error=raw,raw_gate_PASS=math.isfinite(raw) and raw<1e-9,
        before_norm=float(nrm(a)),after_norm=float(nrm(b)),overlap_real=float(overlap.real),
        overlap_imag=float(overlap.imag),common_phase=None if angle is None else float(angle),
        phase_aligned_error_DIAGNOSTIC_ONLY=None if alignment is None else float(nrm(alignment*b-a)),
        low_branch_error=float(nrm(b[:half]-a[:half])),high_branch_error=float(nrm(b[half:]-a[half:])),
        low_branch_phase_DIAGNOSTIC_ONLY=branch_phase(a[:half],b[:half]),
        high_branch_phase_DIAGNOSTIC_ONLY=branch_phase(a[half:],b[half:]),
        hadamard_before=float(observable(a)),hadamard_after=float(observable(b)),
        acceptance_override=False,full_H_reference_phase_aligned=False,
        old_STOP_remains_STOP=True,metrics_precision='EXTENDED' if extended_metrics else 'BINARY64')
    if expected_hadamard is not None: result['independent_expected_hadamard']=float(expected_hadamard)
    require(all(v is None or not isinstance(v,float) or math.isfinite(v) for v in result.values()), 'FINITE_METRICS')
    return result
