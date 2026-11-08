"""Uniform operator qualification, never a measurement/Born engine.

Reuse the fixed interval context and local matrix certificates. The small
explicit numpy update is used ONLY to enclose full input isometries, with its
own IEEE binary64 roundoff bound; it does not replace the old exact engine.
Real descriptor/gate evaluation is a deferred scientific call.
"""
from fractions import Fraction as F
import math

import numpy as np

from . import a1_engine_connection as old
from .df_correction_comparison.intervals import configure,iv,upper,outward_upper

U=F(1,2**53)
ETA=F(1,2**1074)


def certificate(gate):
    """Only new exact primitives are added; prior analytical gates are reused."""
    if gate.name in old.ALLOWED:
        return old.certify_gate(gate)
    from qiskit.circuit.library import XGate,SXGate,ZGate,CXGate,SwapGate,CSwapGate
    exact_classes=(XGate,SXGate,ZGate,CXGate,SwapGate,CSwapGate)
    if not isinstance(gate,exact_classes) or gate.params:
        raise ValueError('unsupported qualification gate: '+gate.name)
    _,matrix,sha=old.matrix_record(gate);d=len(matrix)
    expected=[[(F(int(i==j)),F(0)) for j in range(d)] for i in range(d)]
    if isinstance(gate,SXGate):
        expected=[[(F(1,2),F(1,2)),(F(1,2),-F(1,2))],
                  [(F(1,2),-F(1,2)),(F(1,2),F(1,2))]]
    elif isinstance(gate,ZGate):
        expected[1][1]=(-F(1),F(0))
    else:
        permutation=[]
        for j in range(d):
            if isinstance(gate,XGate):i=j^1
            elif isinstance(gate,CXGate):i=j^2 if j&1 else j
            elif isinstance(gate,SwapGate):i=((j&1)<<1)|((j>>1)&1)
            else:i=j^6 if (j&1 and ((j>>1)&1)!=((j>>2)&1)) else j
            permutation.append(i)
        expected=[[(F(int(i==permutation[j])),F(0)) for j in range(d)] for i in range(d)]
    if matrix!=expected:raise ValueError('exact primitive matrix mismatch')
    return dict(key=(gate.name,()),matrix_sha256=sha,matrix=matrix,error_upper=F(0))


def key(gate):
    return gate.name,tuple(old.finite(p).hex() for p in gate.params)


def clean_body(circuit,*,logical_ancilla,permutation=None,reset_wires=()):
    """Audit the channel boundary before dropping ideal initial resets/readout."""
    from qiskit import QuantumCircuit
    width=circuit.num_qubits
    perm=list(range(width)) if permutation is None else list(permutation)
    if sorted(perm)!=list(range(width)):raise ValueError('invalid final permutation')
    if circuit.num_clbits!=1:raise ValueError('exactly one measured signal bit required')
    out=QuantumCircuit(width);out.global_phase=circuit.global_phase
    touched=set();reset=[];measure=[];measured=set()
    for item in circuit.data:
        qs=tuple(circuit.find_bit(q).index for q in item.qubits)
        if item.operation.name=='reset':
            if len(qs)!=1 or qs[0] in touched or qs[0] in reset:
                raise ValueError('reset not a unique initial vacuum reset')
            reset.append(qs[0]);continue
        if item.operation.name=='measure':
            cs=tuple(circuit.find_bit(q).index for q in item.clbits)
            if qs!=(perm[logical_ancilla],) or cs!=(0,):
                raise ValueError('signal measurement/layout mismatch')
            measure.append(qs);measured.update(qs);continue
        if item.clbits or set(qs)&measured:raise ValueError('dynamic or post-readout operation')
        if item.operation.name=='barrier':continue
        touched.update(qs);out.append(item.operation,qs)
    if len(measure)!=1 or set(reset)!=set(reset_wires):
        raise ValueError('reset/readout contract mismatch')
    return out


def rounded_apply(data,matrix,qs):
    """Explicit real/imag multiply-add; no dot/BLAS, no state normalization."""
    qs=tuple(qs);width=(data.shape[0]-1).bit_length();d=1<<len(qs)
    if (data.shape[0]!=(1<<width) or len(set(qs))!=len(qs)
        or any(type(q) is not int or q<0 or q>=width for q in qs)
        or len(matrix)!=d):
        raise ValueError('operator update shape/wires')
    mask=sum(1<<q for q in qs)
    base=np.arange(data.shape[0]);base=base[(base&mask)==0]
    offsets=np.array([sum(((j>>k)&1)<<q for k,q in enumerate(qs)) for j in range(d)])
    indices=base[:,None]|offsets[None,:]
    original=data[indices] # owned copy: source not overwritten by later rows
    out=np.empty_like(original)
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        for i in range(d):
            re=np.zeros_like(original[:,0,:].real);im=np.zeros_like(re)
            for j in range(d):
                ar,ai=(float(x) for x in matrix[i][j]);xr=original[:,j,:].real;xi=original[:,j,:].imag
                re=np.add(re,np.subtract(np.multiply(ar,xr),np.multiply(ai,xi)))
                im=np.add(im,np.add(np.multiply(ar,xi),np.multiply(ai,xr)))
            out[:,i,:]=re+1j*im
    if not np.isfinite(out).all():raise ValueError('nonfinite operator contraction')
    data[indices]=out


def _scalar_certificate(angle):
    """Global phase, independently enclosed, never fitted from output."""
    import mpmath as mp
    configure();theta=F(float(angle));value=complex(math.cos(float(angle)),math.sin(float(angle)))
    s=F(0)
    for actual,ball in ((value.real,mp.iv.cos(iv(theta))),(value.imag,mp.iv.sin(iv(theta)))):
        lo,hi=old.bounds(ball-iv(F(actual)));s+=max(abs(lo),abs(hi))**2
    return dict(matrix=[[(F(value.real),F(value.imag))]],
                error_upper=outward_upper(upper(mp.iv.sqrt(iv(s)))))


def operation_bound(delta,d):
    """Normwise gate + explicit real arithmetic envelope, with underflow."""
    if d<1 or F(delta)<0:raise ValueError('invalid gate bound')
    k=4*d+4
    gamma=k*U/(1-k*U)
    # Absolute real 2d representation has norm <= sqrt(2d)*(1+delta).
    # 2d is a conservative rational substitute for sqrt(2d).
    rho=2*d*gamma*(1+F(delta))
    return F(delta)+rho,k


def simulate_isometry(circuit,system_qubits,*,cache=None):
    """ALL 2^n input columns: operator qualification, not physical-state shots."""
    width=circuit.num_qubits;n=system_qubits
    if not (1<=n<=8 and n<=width<=17) or circuit.num_clbits:
        raise ValueError('finite operator qualification scope')
    if (np.dtype(float).itemsize!=8 or np.finfo(float).eps!=2.**-52
        or np.float64(1.)+np.float64(2.**-53)!=1.
        or np.float64(np.finfo(float).tiny)*np.float64(.5)!=2.**-1023):
        raise ValueError('IEEE rounding/gradual-underflow prerequisites unverified')
    rows,columns=1<<width,1<<n
    data=np.zeros((rows,columns),complex)
    data[np.arange(columns),np.arange(columns)]=1.
    cache={} if cache is None else cache
    largest=F(0);steps=0;largest_k=0
    def charge(record,d):
        nonlocal largest,steps,largest_k
        eff,k=operation_bound(record['error_upper'],d)
        largest=max(largest,eff);largest_k=max(largest_k,k);steps+=1
    phase=float(circuit.global_phase)
    if phase:
        row=_scalar_certificate(phase);charge(row,1)
        # A scalar is the d=1 instance of exactly the same update.
        rounded_apply(data,row['matrix'],())
    for item in circuit.data:
        g=item.operation;token=key(g)
        if token not in cache:cache[token]=certificate(g)
        row=cache[token]
        # No mutable matrix may silently replace the certified matrix.
        actual=old.matrix_record(g)[2]
        if actual!=row['matrix_sha256']:raise ValueError('matrix changed after certification')
        charge(row,1<<g.num_qubits)
        rounded_apply(data,row['matrix'],tuple(circuit.find_bit(q).index for q in item.qubits))
    if steps:
        configure()
        product=outward_upper(upper((1+iv(largest))**steps))
        if product-1>=1:raise ValueError('operator roundoff envelope >=1')
        # ||input isometry||F=sqrt(columns)<=columns. Each real operation has
        # |rounding error|<=u|exact result|+ETA, including gradual underflow.
        under=2*rows*columns*largest_k*ETA/(1-largest_k*U)
        error=outward_upper(columns*(product-1)+steps*under*product)
    else:error=F(0)
    return data,dict(frobenius_simulation_error_upper=str(error),columns=columns,rows=rows,
        steps=steps,effective_local_error_upper=str(largest),input_norm_upper=columns,
        model='IEEE binary64 round-to-nearest, gradual underflow; explicit real multiply/add',
        gate_rounding_included=True,global_phase_included=True,BLAS_dot_used=False)


def qualify_compilation(raw,compiled,n,*,cache=None):
    """Uniform projected operator error, phase-sensitive, all target inputs."""
    ancilla=n if raw.num_qubits==n+1 else 2*n
    resets=() if raw.num_qubits==n+1 else tuple(range(n,2*n))
    permutation=(compiled.layout.final_index_layout(filter_ancillas=True)
                 if compiled.layout is not None else list(range(compiled.num_qubits)))
    if compiled.num_qubits!=raw.num_qubits:raise ValueError('untracked compiler ancillas')
    left=clean_body(raw,logical_ancilla=ancilla,reset_wires=resets)
    right=clean_body(compiled,logical_ancilla=ancilla,permutation=permutation,reset_wires=resets)
    x,bx=simulate_isometry(left,n,cache=cache);y,by=simulate_isometry(right,n,cache=cache)
    indices=np.arange(len(y));physical=np.zeros(len(y),dtype=np.int64)
    for q,p in enumerate(permutation):physical|=((indices>>q)&1)<<p
    y=y[physical]
    with np.errstate(over='raise',invalid='raise'):
        re=np.subtract(x.real,y.real);im=np.subtract(x.imag,y.imag)
    maximum=F(float(max(np.max(np.abs(re)),np.max(np.abs(im)))))
    magnitude=F(float(max(np.max(np.abs(x.real)),np.max(np.abs(x.imag)))))
    magnitude_y=F(float(max(np.max(np.abs(y.real)),np.max(np.abs(y.imag)))))
    # Exact enclosure of subtraction then a conservative sqrt(2*entries).
    component=maximum+U*(magnitude+magnitude_y)+2*ETA
    count=2*x.size;factor=math.isqrt(count)+(math.isqrt(count)**2<count)
    residual=outward_upper(factor*component)
    operator=outward_upper(residual+F(bx['frobenius_simulation_error_upper'])+
                           F(by['frobenius_simulation_error_upper']))
    if operator>=1:raise ValueError('uninformative operator enclosure >=1')
    return dict(status='UNIFORM_BOUND_OBTAINED_NOT_A_SCIENCE_ACCEPTANCE',
        projected_operator_error_upper=str(operator),coordinate_error_upper=str(2*operator),
        raw_simulation=bx,compiled_simulation=by,observed_max_component=str(maximum),
        final_logical_to_physical=permutation,phase_fitting=False,
        qualification_domain='all normalized system states; signal=0, fresh ideal vacuum reference',
        readout_and_reset_structure_verified=True,allowance_adopted=False)
