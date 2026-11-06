"""Strict actual-angle Clifford+T acquisition and compressed native pricing."""
from __future__ import annotations
import fcntl
from functools import lru_cache
from fractions import Fraction
import hashlib
import math
import time
import mpmath as mp
from .track_c_h4 import OUT,save_json
from trottertracks.algorithm_codesign.synthesis_placement.economics import sequence_matrix


def exact_word_matrix(sequence):
    """Reuse pygridsynth's exact cyclotomic ring before interval evaluation.

    This avoids interval wrapping across thousands of H/T factors. W remains
    in the exact matrix, so this is a strict operator guard without phase fit.
    """
    from pygridsynth.domega_unitary import DOmegaUnitary
    unitary=DOmegaUnitary.from_gates(sequence.replace('t','TTTTTTT'))
    root=mp.iv.sqrt(2)
    def entry(x):
        a,b,c,d=x.u.a,x.u.b,x.u.c,x.u.d
        return mp.iv.mpc(d+(c-a)/root,b+(c+a)/root)/(root**x.k)
    return [[entry(x) for x in row] for row in unitary.to_matrix]


def strict_guard(sequence,theta):
    n,d=float(theta).as_integer_ratio();a=mp.iv.mpf(n)/d
    u=exact_word_matrix(sequence)
    target=[[mp.iv.exp(-mp.iv.j*a/2),mp.iv.mpc(0)],
            [mp.iv.mpc(0),mp.iv.exp(mp.iv.j*a/2)]]
    bound=mp.iv.sqrt(sum((abs(u[i][j]-target[i][j])**2 for i in range(2) for j in range(2)),mp.iv.mpf(0)))
    raw=bound._mpi_[1];sign,mantissa,exponent,bits=raw
    if bits<0:raise ArithmeticError('nonfinite interval guard endpoint')
    return Fraction((-1 if sign else 1)*mantissa)*Fraction(2)**exponent


@lru_cache(None)
def acquire(theta,exponent):
    """One backend acquisition per positive exact-binary angle/precision key."""
    theta=abs(float(theta));spec=f'{theta.hex()}:10^{exponent}:pygridsynth2.0.0:strict:seed17'
    key=hashlib.sha256(spec.encode()).hexdigest();directory=OUT/'synthesis'
    directory.mkdir(exist_ok=True);path=directory/f'{key}.json'
    with (directory/f'{key}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            import json
            return json.loads(path.read_text())
        dps=max(100,-exponent+60);mp.mp.dps=dps;mp.iv.dps=dps
        epsilon=Fraction(10)**exponent
        k=round(theta/(math.pi/2))
        fast='W'*((-k)%8)+'S'*(k%4)
        start=time.monotonic();error=strict_guard(fast,theta)
        backend_dps=None
        route='strict_clifford_fast_path'
        if error<=epsilon:sequence=fast
        else:
            from pygridsynth.config import GridsynthConfig
            from pygridsynth.gridsynth import gridsynth_gates
            n,d=theta.as_integer_ratio()
            config=GridsynthConfig(seed=17,up_to_phase=False)
            sequence=gridsynth_gates(mp.mpf(n)/d,mp.mpf(10)**exponent/4,cfg=config)
            backend_dps=config.dps
            try:error=strict_guard(sequence,theta)
            except ArithmeticError:error=None
            route='pygridsynth_2.0.0'
        initial_error=error;initial_dps=dps
        # Retain word-length headroom as a fallback for an inconclusive guard.
        risky=sum(sequence.count(g) for g in 'HTtW')
        guard_dps=dps
        if error is None or error>epsilon:
            guard_dps=max(dps,-exponent+math.ceil(.5*math.log10(2)*risky)+50)
            mp.mp.dps=guard_dps;mp.iv.dps=guard_dps
            try:error=strict_guard(sequence,theta)
            except ArithmeticError:error=None
        if error is None or error>epsilon or len(sequence)>20000:
            save_json(directory/f'{key}.failed.json',dict(key=key,spec=spec,sequence=sequence,
                strict_error_upper=None if error is None else str(error),initial_error_upper=None if initial_error is None else str(initial_error),
                initial_dps=initial_dps,guard_dps=guard_dps,status='UNQUALIFIED'))
            raise ArithmeticError(f'strict guard or word bound failed: {key}')
        result=dict(key=key,spec=spec,theta_hex=theta.hex(),precision_exponent=exponent,
            sequence=sequence,sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
            T=sequence.count('T')+sequence.count('t'),Tdagger=sequence.count('t'),
            Clifford=sum(sequence.count(g) for g in 'HSX'),global_W=sequence.count('W'),
            depth=sum(g!='W' for g in sequence),strict_error_upper=str(error),
            dps=guard_dps,initial_dps=initial_dps,initial_error_upper=None if initial_error is None else str(initial_error),
            guard_dps=guard_dps,backend_dps=backend_dps,
            initial_guard_pass=bool(initial_error is not None and initial_error<=epsilon),route=route,
            guard_method='exact cyclotomic word matrix + outward interval strict Frobenius',
            wall_seconds=time.monotonic()-start,status='PASS')
        save_json(path,result)
        return result


def t_metrics(qc,exponent):
    """Count the explicit strict lowering of a globally opt2-compiled wrapper.

    A sequence is a written matrix product; inverse angles use its actual
    adjoint. Scalar W is tracked in each cache record, with no physical gate.
    No projective fit or T-price addition to a separately priced Q circuit.
    """
    T=Td=Clifford=CX=0;depths=[0]*qc.num_qubits;keys=set()
    for instruction in qc.data:
        op=instruction.operation;wires=[qc.find_bit(q).index for q in instruction.qubits]
        if op.name=='rz':
            theta=float(op.params[0]);row=acquire(theta,exponent);keys.add(row['key'])
            T+=row['T'];Td+=row['Tdagger'] if theta>=0 else row['T']-row['Tdagger']
            Clifford+=row['Clifford'];length=row['depth']
            if wires:depths[wires[0]]+=length
        elif op.name=='sx':
            # SX = H S H exactly, including its phase.
            Clifford+=3;depths[wires[0]]+=3
        elif op.name=='x':Clifford+=1;depths[wires[0]]+=1
        elif op.name=='cx':
            CX+=1;d=max(depths[q] for q in wires)+1
            for q in wires:depths[q]=d
        elif op.name=='measure':depths[wires[0]]+=1
        elif op.name=='barrier':
            d=max(depths[q] for q in wires)
            for q in wires:depths[q]=d
        else:raise ValueError('unlowered native instruction '+op.name)
    return dict(T=T,Tdagger=Td,CX=CX,Clifford=Clifford,depth=max(depths),RZ_remaining=0,
                synthesis_keys=sorted(keys),post_synthesis_policy='explicit cached sequences; common Q opt2 fusion precedes synthesis')
