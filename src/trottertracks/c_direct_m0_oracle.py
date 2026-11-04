"""Exact symmetric-matrix ground isolation and an Arb192 state-distance bound.

This is a mathematical primitive, not execution permission or a choice of the
research Hamiltonian. Readiness tests use invented matrices exclusively.
"""
from fractions import Fraction as F
import math

from . import c_fresh_iid_calibration as fresh
from . import c_iid_grid_expected_bias as expected
from . import c_pr_aware_df_zsxx_enclosure as z


def exact_real(value):
    if isinstance(value,complex):
        if value.imag != 0: raise ValueError('EXACT_REAL_H_REQUIRED_NO_PROJECTION')
        value=value.real
    if isinstance(value,float):
        if not math.isfinite(value): raise ValueError('FINITE_EXACT_INPUT')
        return F(value)
    return expected.rational(value)


def symmetric_matrix(values):
    A=[[exact_real(x) for x in row] for row in values];n=len(A)
    if n<2 or any(len(row)!=n for row in A): raise ValueError('SQUARE_ORACLE_MATRIX')
    if any(A[i][j]!=A[j][i] for i in range(n) for j in range(n)):
        raise ValueError('EXACT_SYMMETRY_REQUIRED_NO_HERMITIANIZATION')
    return A


def matrix_id(A):
    return fresh.digest([[str(x) for x in row] for row in A])


def propose_isolation(values):
    """One numerical eigensolve proposes sigma; no numerical gap is certified."""
    import numpy as np
    A=symmetric_matrix(values)
    eig=np.linalg.eigvalsh(np.array([[float(x) for x in row] for row in A]))
    if not np.isfinite(eig).all() or eig[0]>=eig[1]:
        raise ValueError('NUMERICAL_ISOLATION_PROPOSAL_FAILED')
    shift=(F(float(eig[0]))+F(float(eig[1])))/2
    return dict(H_id=matrix_id(A),shift_exact=str(shift),
        lowest_two_observed=[float(x).hex() for x in eig[:2]],
        qualification='proposal_only_not_a_gap_or_oracle_certificate')


def inertia_below(values, shift):
    """Exact rational unpivoted LDL^T, with STOP at a zero pivot.

    Congruence by nonsingular unit lower L preserves inertia. No floating
    tolerance, pivot perturbation or retry is used. A zero pivot can occur for
    nonsingular matrices too; this path then returns no spectral conclusion.
    """
    A=symmetric_matrix(values);n=len(A);s=expected.rational(shift)
    L=[[F(int(i==j)) for j in range(n)] for i in range(n)];D=[]
    for k in range(n):
        pivot=A[k][k]-s-sum((L[k][j]**2*D[j] for j in range(k)),F())
        if not pivot: raise ValueError('ZERO_LDL_PIVOT_STOP_NO_REGULARIZATION')
        D.append(pivot)
        for i in range(k+1,n):
            L[i][k]=(A[i][k]-sum((L[i][j]*L[k][j]*D[j] for j in range(k)),F()))/pivot
    return dict(H_id=matrix_id(A),shift_exact=str(s),
        negative=sum(x<0 for x in D),positive=sum(x>0 for x in D),zero=0,
        pivots_exact=[str(x) for x in D],method='exact rational LDL transpose congruence',
        floating_point_inertia_used=False)


def qualify_ground(values, amplitudes, proposal):
    """Distance of the ORIGINAL, unnormalized centre to a ground eigenvector.

    Let u=v/||v||, mu=<u,Hu>, r=||(H-mu)u||. If inertia(H-sigma I)
    has exactly one negative eigenvalue and mu<sigma, then lambda_1>sigma
    and ||P_exc u|| <= r/(sigma-mu). A phase-aligned unit ground state g
    satisfies ||u-g|| <= sqrt(2)*r/(sigma-mu). Add | ||v||-1 |.
    """
    from flint import ctx
    A=symmetric_matrix(values);n=len(A)
    if proposal.get('H_id')!=matrix_id(A): raise ValueError('PROPOSAL_ORACLE_TARGET_IDENTITY')
    if len(amplitudes)!=n or any(len(x)!=2 for x in amplitudes): raise ValueError('SECTOR_VECTOR_SHAPE')
    v=[[exact_real(x) for x in pair] for pair in amplitudes]
    norm2=sum((a*a+b*b for a,b in v),F())
    if norm2<=0: raise ValueError('NONZERO_ORACLE_CENTRE')
    Hv=[[sum((A[i][j]*v[j][axis] for j in range(n)),F()) for axis in range(2)] for i in range(n)]
    rayleigh=sum((v[i][0]*Hv[i][0]+v[i][1]*Hv[i][1] for i in range(n)),F())/norm2
    inertia=inertia_below(A,proposal['shift_exact']);sigma=F(inertia['shift_exact'])
    if inertia['negative']!=1 or inertia['positive']!=n-1 or rayleigh>=sigma:
        raise ValueError('UNIQUE_GROUND_ISOLATION_NOT_CERTIFIED')
    residual2=sum(((Hv[i][j]-rayleigh*v[i][j])**2 for i in range(n) for j in range(2)),F())/norm2
    sin2=residual2/(sigma-rayleigh)**2
    if sin2>=1: raise ValueError('ORACLE_OVERLAP_BOUND_UNRESOLVED')
    with z._BALL_LOCK,ctx.workprec(192):
        norm=z._real(norm2).sqrt()
        lo,hi=z._fraction(norm.lower()),z._fraction(norm.upper())
        normalization=max(abs(1-lo),abs(1-hi))
        angle=expected.sqrt_upper(2*sin2)
        eta=z._fraction(z._real(normalization+angle).upper())
    return dict(status='VALIDATED_DISTANCE_TO_EXACT_INPUT_MATRIX_GROUND',H_id=matrix_id(A),
        vector_id=fresh.digest([[str(x) for x in pair] for pair in v]),
        precision_bits=192,rounding_enclosed=True,
        centre_norm_squared_exact=str(norm2),centre_norm_interval=[str(lo),str(hi)],
        rayleigh_exact=str(rayleigh),normalized_residual_squared_exact=str(residual2),
        excited_weight_upper_exact=str(sin2),normalization_distance_upper_exact=str(normalization),
        vector2_distance_upper_exact=str(eta),ground_phase='aligned to the original centre;signals invariant to this phase',
        inertia=inertia,oracle_target='ground of THIS exact symmetric input matrix only',
        existing_own_DF_target_identity_established=False,
        input_normalized_or_projected=False,statistical_confidence=None)
