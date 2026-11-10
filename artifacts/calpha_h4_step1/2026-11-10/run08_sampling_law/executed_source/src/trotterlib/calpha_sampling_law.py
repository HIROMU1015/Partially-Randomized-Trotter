"""Fixed Gaussian gate semantics and exact finite-CDF masses for alpha0-P.

No chemistry, circuit generation or scientific evaluation occurs on import.
Uses the existing interval helper and existing builder's RZ/XXPlusYY gates.
"""
from fractions import Fraction
import math


def finite_cdf_masses(probabilities, bits=53):
    """NumPy 1.26.4 choice(replace=True) under fresh uniform grid draws.

    Narrow reuse of h4_ball_evidence.cdf_binary64/grid_masses, without that
    module's unavailable FLINT import or its unrelated study controller.
    """
    cumulative=[]; running=0.
    for p in probabilities:
        if not math.isfinite(p) or p<0:
            raise ValueError('FINITE_NONNEGATIVE_PROBABILITIES_REQUIRED')
        running=float(running+p);cumulative.append(running)
    if not cumulative or running<=0:
        raise ValueError('NONEMPTY_POSITIVE_MASS_REQUIRED')
    cdf=[float(x/running) for x in cumulative]
    size=1<<bits;previous=0;masses=[]
    for p in cdf:
        x=Fraction.from_float(p)*size
        upper=-(-x.numerator//x.denominator)
        masses.append(Fraction(upper-previous,size));previous=upper
    if sum(masses)!=1 or any(x<0 for x in masses):
        raise ValueError('CDF_GRID_INVALID')
    return masses,cdf


def describe_operations(operations):
    result=[]
    for gate,qubits in operations:
        if gate.name not in ('rz','xx_plus_yy'):
            raise ValueError('UNSUPPORTED_FIXED_BASIS_GATE:'+gate.name)
        params=[float(x) for x in gate.params]
        inv=gate.inverse()
        expected=[-params[0],*params[1:]]
        if inv.name!=gate.name or [float(x) for x in inv.params]!=expected:
            raise ValueError('BUILDER_INVERSE_SEMANTICS_MISMATCH')
        result.append(dict(name=gate.name,qubits=list(qubits),parameters=params))
    return result


def basis_gate_interval(bounds,operations,states,n_qubits):
    """Apply local number-preserving gates, retaining every full-column row.

    The exact known vacuum phase is removed only from a comparison reference;
    it cancels in U D U-dagger in the unchanged actual builder.
    """
    b=bounds;c=b.c;out=c.eye(len(states));lookup={s:i for i,s in enumerate(states)}
    vacuum_phase=c.mpf(0)
    for op in operations:
        if op['name']=='rz':
            q,=op['qubits'];angle=b.real(op['parameters'][0])
            vacuum_phase-=angle/2
            for i,state in enumerate(states):
                factor=c.exp(-c.j*angle*(1-2*((state>>q)&1))/2)
                for j in range(len(states)):
                    out[i,j]*=factor
        elif op['name']=='xx_plus_yy':
            q0,q1=op['qubits'];theta,beta=map(b.real,op['parameters'])
            cosine=c.cos(theta/2);sine=c.sin(theta/2)
            upper=-c.j*sine*c.exp(-c.j*beta)
            lower=-c.j*sine*c.exp(c.j*beta)
            for i,state in enumerate(states):
                if (state>>q0)&1 and not (state>>q1)&1:
                    other=lookup[state^(1<<q0)^(1<<q1)]
                    left=[out[i,j] for j in range(len(states))]
                    right=[out[other,j] for j in range(len(states))]
                    for j in range(len(states)):
                        out[i,j]=cosine*left[j]+upper*right[j]
                        out[other,j]=lower*left[j]+cosine*right[j]
        else:
            raise ValueError('UNSUPPORTED_GATE')
    return out*c.exp(-c.j*vacuum_phase)


def centered_head_error(bounds,basis_error,literal_norm,literal_gram_error,
                        native_phases,reconstruction_phases):
    """Duhamel-free product difference, centering the diagonal algebraically.

    A is the exact unitary gate basis, B the saved literal lift. This is the
    triangle inequality for A(D-cI)A* - B(D-cI)B*, plus diagonal and Gram terms.
    No basis, scalar phase or input program is modified.
    """
    b=bounds;c=b.c
    from .calpha_interval_bounds import endpoint
    middle=(min(float(endpoint(x)) for x in native_phases)
            +max(float(endpoint(x,True)) for x in native_phases))/2
    center=c.exp(c.j*b.real(middle))
    deviation=max((abs(c.exp(c.j*x)-center).b for x in native_phases),
                  key=lambda x:endpoint(x,True))
    diagonal=max((abs(x-y).b for x,y in zip(native_phases,reconstruction_phases)),
                 key=lambda x:endpoint(x,True))
    upper=basis_error*(1+literal_norm)*deviation+literal_norm**2*diagonal+literal_gram_error
    return upper.b,deviation.b,diagonal.b
