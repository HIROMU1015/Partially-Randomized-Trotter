"""Post-hoc phase-sensitive full-basis certificates for fixed ideal circuits.

Existing Arb gate arithmetic is reused. This verifier interprets stored gate
parameters directly;it never uses Qiskit definitions,Operator,or a compiler as
its mathematical reference. It certifies finite circuit pairs,not unseen words
or raw-Hamiltonian correctness. No molecular loader or sampling is present.
"""
from fractions import Fraction
import math
import os

from . import c_pr_aware_df_zsxx_enclosure as z

VERSION='fixed_full_basis_Arb_recenter_v1'
ARITY={'id':1,'x':1,'z':1,'sx':1,'sxdg':1,'rz':1,'ry':1,'p':1,
       'h':1,'s':1,'sdg':1,'cx':2,'rzz':2,'crz':2,'xx_plus_yy':2,'crzz':3}
PARAMS={k:(2 if k=='xx_plus_yy' else 1 if k in ('rz','ry','p','rzz','crz','crzz') else 0) for k in ARITY}


def record(circuit,*,reduce_ancilla=False,strip_terminal_measurement=False):
    """Decode bound circuit metadata only when called in an allowed scope.

    Gate class and all arities/parameters are checked. Unknown gates are not
    expanded through unverified Qiskit definitions. Reduce only an explicitly
    proven ancilla-only shell;its embedded operator distance is unchanged.
    """
    from qiskit.circuit.library import (IGate,XGate,ZGate,SXGate,SXdgGate,RZGate,
        RYGate,PhaseGate,HGate,SGate,SdgGate,CXGate,RZZGate,CRZGate,XXPlusYYGate)
    from qiskit.circuit import ControlledGate
    classes=dict(zip(['id','x','z','sx','sxdg','rz','ry','p','h','s','sdg','cx','rzz','crz','xx_plus_yy'],
        [IGate,XGate,ZGate,SXGate,SXdgGate,RZGate,RYGate,PhaseGate,HGate,SGate,SdgGate,CXGate,RZZGate,CRZGate,XXPlusYYGate]))
    if circuit.parameters:raise ValueError('BOUND_UNITARY_CIRCUIT_REQUIRED')
    n=circuit.num_qubits
    if n<1:raise ValueError('QUBITS_REQUIRED')
    items=list(circuit.data);positions=[i for i,x in enumerate(items) if x.operation.name=='measure']
    stripped=False
    if positions:
        last=items[-1]
        if not (strip_terminal_measurement and reduce_ancilla and positions==[len(items)-1]
                and circuit.num_clbits==1 and len(last.qubits)==len(last.clbits)==1
                and circuit.find_bit(last.qubits[0]).index==n-1
                and circuit.find_bit(last.clbits[0]).index==0):
            raise ValueError('ONLY_REGISTERED_TERMINAL_ANCILLA_MEASUREMENT')
        items=items[:-1];stripped=True
    elif circuit.num_clbits:raise ValueError('UNUSED_CLASSICAL_BITS_NOT_REGISTERED')
    rows=[]
    for item in items:
        op=item.operation;name=op.name
        if isinstance(op,ControlledGate) and op.ctrl_state!=(1<<op.num_ctrl_qubits)-1:
            raise ValueError('OPEN_CONTROL_NOT_REGISTERED')
        if name not in ARITY:raise ValueError('UNSUPPORTED_FIXED_GATE:'+name)
        if name=='crzz':
            if not (isinstance(op,ControlledGate) and op.num_ctrl_qubits==1
                    and op.ctrl_state==1 and isinstance(op.base_gate,RZZGate)):
                raise ValueError('UNKNOWN_CRZZ_SEMANTICS')
        elif not isinstance(op,classes[name]):raise ValueError('GATE_CLASS_MISMATCH:'+name)
        if isinstance(op,ControlledGate) and op.ctrl_state!=(1<<op.num_ctrl_qubits)-1:
            raise ValueError('OPEN_CONTROL_NOT_REGISTERED')
        qubits=[circuit.find_bit(q).index for q in item.qubits]
        if len(qubits)!=ARITY[name] or len(op.params)!=PARAMS[name]:raise ValueError('GATE_ARITY')
        values=[float(v) for v in op.params]
        if any(not math.isfinite(v) for v in values):raise ValueError('NONFINITE_GATE')
        if reduce_ancilla:
            if qubits!=[n-1]:raise ValueError('SHELL_NOT_ANCILLA_ONLY')
            qubits=[0]
        rows.append(dict(name=name,qubits=qubits,params_hex=[v.hex() for v in values]))
    phase=float(circuit.global_phase)
    if not math.isfinite(phase):raise ValueError('NONFINITE_PHASE')
    return dict(qubits=1 if reduce_ancilla else n,embedded_qubits=n,
        ancilla_only_reduction=reduce_ancilla,terminal_measurement_removed=stripped,
        phase_hex=phase.hex(),gates=rows,
        global_phase_fit=False,parameter_semantics='exact stored binary64 dyadics')


def _validate(r):
    n=r['qubits']
    if type(n) is not int or n<1 or r.get('global_phase_fit') is not False:
        raise ValueError('FIXED_RECORD_SCOPE')
    if not math.isfinite(float.fromhex(r['phase_hex'])):raise ValueError('NONFINITE_PHASE')
    for g in r['gates']:
        name=g['name'];qs=g['qubits']
        if name not in ARITY or len(qs)!=ARITY[name] or len(set(qs))!=len(qs):raise ValueError('LOCAL_GATE_SCHEMA')
        if any(type(q) is not int or not 0<=q<n for q in qs):raise ValueError('QUBIT_ORDER')
        if len(g['params_hex'])!=PARAMS[name] or any(not math.isfinite(float.fromhex(v)) for v in g['params_hex']):
            raise ValueError('GATE_PARAMETER_SCHEMA')


def _local(name,params):
    from flint import arb,acb,acb_mat
    if name in ('id','x','z','sx','sxdg','rz','ry','p'):
        return z.gate_matrix(name,None if not params else params[0])
    I=acb(0,1)
    if name=='h':
        h=1/arb(2).sqrt();return acb_mat([[h,h],[h,-h]])
    if name in ('s','sdg'):return acb_mat([[1,0],[0,I if name=='s' else -I]])
    if name=='cx':return acb_mat([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
    t=z._real(params[0]);e0=(-I*t/2).exp();e1=(I*t/2).exp()
    if name in ('rzz','crz','crzz'):
        if name=='rzz':diagonal=[e0,e1,e1,e0]
        elif name=='crz':diagonal=[1,e0,1,e1]
        else:diagonal=[1,e0,1,e1,1,e1,1,e0]
        return acb_mat([[value if i==j else 0 for j in range(len(diagonal))] for i,value in enumerate(diagonal)])
    if name=='xx_plus_yy':
        beta=z._real(params[1]);c=(t/2).cos();s=(t/2).sin()
        return acb_mat([[1,0,0,0],[0,c,-I*s*(-I*beta).exp(),0],
                        [0,-I*s*(I*beta).exp(),c,0],[0,0,0,1]])
    raise ValueError('UNREVIEWED_LOCAL_GATE')


def _midpoint_vector(ball):
    """Recenter and retain an outward vector2 error,avoiding interval wrapping."""
    from flint import acb,acb_mat
    midpoint=acb_mat([[acb(ball[i,0].real.mid(),ball[i,0].imag.mid())] for i in range(ball.nrows())])
    delta=Fraction(z._norm(ball-midpoint)['upper_exact'])
    return midpoint,delta


def _column(r,column,cache):
    """Exact ideal state-action within a certified vector2 radius.

    If ||v-c||<=E and exact local unitary B is enclosed by Ball(B),then
    Ball(B)c contains Bc. Recenter to c' with outward delta: E'<=E+delta.
    All gate/phase evaluation and arithmetic error is included in delta.
    """
    from flint import acb_mat,acb
    n=r['qubits'];d=1<<n;v=acb_mat([[int(i==column)] for i in range(d)]);error=Fraction()
    for gate in r['gates']:
        qs=tuple(gate['qubits']);key=(gate['name'],tuple(gate['params_hex']))
        if key not in cache:cache[key]=_local(key[0],[float.fromhex(x) for x in key[1]])
        B=cache[key];size=1<<len(qs);mask=sum(1<<q for q in qs)
        # All disjoint local blocks in one existing FLINT matrix multiply.
        bases=[i for i in range(d) if i&mask==0]
        offsets=[sum(((k>>j)&1)<<q for j,q in enumerate(qs)) for k in range(size)]
        packed=acb_mat([[v[base|offset,0] for base in bases] for offset in offsets])
        result=B*packed;out=acb_mat(d,1)
        for i,offset in enumerate(offsets):
            for j,base in enumerate(bases):out[base|offset,0]=result[i,j]
        v,delta=_midpoint_vector(out);error+=delta
    phase=(acb(0,1)*z._real(float.fromhex(r['phase_hex']))).exp()
    v,delta=_midpoint_vector(z._scale(v,phase));return v,error+delta


_PAIR=None
_CACHE=None
def _init_pair(left,right):
    global _PAIR,_CACHE
    _PAIR=(left,right);_CACHE={}


def _pair_column(column):
    from flint import ctx
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        a,ea=_column(_PAIR[0],column,_CACHE);b,eb=_column(_PAIR[1],column,_CACHE)
        difference=z._norm(a-b);radius=ea+eb
        upper=Fraction(difference['upper_exact'])+radius
        lower=max(Fraction(),Fraction(difference['lower_exact'])-radius)
        return dict(column=column,upper_exact=str(upper),lower_exact=str(lower),
                    evaluation_radius_left_exact=str(ea),evaluation_radius_right_exact=str(eb))


def compare(left,right,*,parallel=True):
    """All full-Fock basis columns,not a finite-probe extrapolation.

    Pure columns are independent. Process-local Arb context prevents shared
    precision mutations. Workers follow actual CPU affinity/quota/memory.
    The serial path is for tiny invented circuits,not an artificial CPU cap.
    """
    from flint import ctx,arb
    z.runtime_model();_validate(left);_validate(right)
    if left['qubits']!=right['qubits']:raise ValueError('FIXED_DIMENSION_MISMATCH')
    d=1<<left['qubits'];jobs=range(d)
    work=d*d*(len(left['gates'])+len(right['gates']))
    plan=resource_plan(d)
    if parallel and work>100000 and plan['workers']>1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init_pair,initargs=(left,right)) as pool:
            rows=list(pool.map(_pair_column,jobs))
    else:
        with z._BALL_LOCK:
            _init_pair(left,right);rows=[_pair_column(i) for i in jobs]
        plan.update(workers_used=1,reason='tiny workload or explicit artificial-test serial path')
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        squared=sum((z._real(Fraction(row['upper_exact']))**2 for row in rows),arb(0))
        upper=z._fraction(squared.sqrt().upper())
    return dict(status='VALIDATED_FIXED_CIRCUIT_PAIR',norm='full_operator_2',phase_fit=False,
        precision_bits=z.PRECISION,dimension=d,columns_evaluated=d,
        upper_exact=str(min(Fraction(2),upper)),lower_exact=str(max(Fraction(r['lower_exact']) for r in rows)),
        upper_method='Frobenius of all certified column2 bounds;exact ideal gates plus recentered evaluation radii',
        columns=rows,resource_plan=plan,whole_law_bound=None,raw_Hamiltonian_reference_bound=None,
        CA_achieved=None,shots=None,total_resources=None)


def resource_plan(dimension):
    """No arbitrary resource cap;respect actual affinity,cgroup quota and RAM."""
    from pathlib import Path
    cpus=len(os.sched_getaffinity(0));quota=None
    p=Path('/sys/fs/cgroup/cpu.max')
    if p.exists():
        q,period=p.read_text().split()
        if q!='max':quota=max(1,math.ceil(int(q)/int(period)));cpus=min(cpus,quota)
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    # Fork-free Python/Qiskit process baseline plus two state vectors and gate cache.
    estimate=512*1024**2+dimension*4096
    workers=max(1,min(dimension,cpus,mem['MemAvailable']//estimate))
    return dict(workers=workers,CPU_affinity=len(os.sched_getaffinity(0)),cgroup_CPU_quota=quota,
        available_RAM_bytes=mem['MemAvailable'],estimated_RAM_per_worker_bytes=estimate,
        method='independent full-basis columns;process-local gate cache;FLINT block matmul;recenter after each gate',
        CPU_cap=None,RAM_cap=None,wall_cap=None)


def identity(n):return dict(qubits=n,phase_hex=0.0.hex(),gates=[],global_phase_fit=False)


def adjoint(r):
    """Exact mathematical adjoint of a record,not an actual inverse builder."""
    _validate(r);out=dict(r,gates=[],phase_hex=(-float.fromhex(r['phase_hex'])).hex())
    inverse={'sx':'sxdg','sxdg':'sx','s':'sdg','sdg':'s'}
    for gate in reversed(r['gates']):
        g=dict(gate,params_hex=list(gate['params_hex']))
        if g['name'] in inverse:g['name']=inverse[g['name']]
        elif g['name'] in ('rz','ry','p','rzz','crz','crzz','xx_plus_yy'):
            g['params_hex'][0]=(-float.fromhex(g['params_hex'][0])).hex()
        out['gates'].append(g)
    return out
