"""Opt-in high-precision construction of the existing strong PF operator.

Array values, signed factors and Fraction schedules are fixed inputs. No I/O,
circuits, state solve, symmetrization, unitary projection or phase fitting.
The numerical native representation is not an emitted gate implementation.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from types import SimpleNamespace

import mpmath
import numpy as np

from . import b2_p1_strong_pf_validation as legacy
from .b2_p1_strong_pf_h4_diagnosis import hp_helpers
from trottertracks.c1_stage3_reference import _ordered_exponentials


def exact_real(ctx, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('REAL_INPUT_REQUIRED')
    if isinstance(value, Fraction):
        return ctx.mpf(value.numerator)/value.denominator
    if isinstance(value, (int, np.integer)):
        return ctx.mpf(int(value))
    if not isinstance(value, (float, np.float64)) or not np.isfinite(value):
        raise ValueError('FINITE_BINARY64_REAL_REQUIRED')
    return ctx.mpf(float(value))


def maxabs(matrix):
    return max((abs(x) for x in matrix), default=matrix.ctx.mpf(0))


def unitarity(matrix):
    return maxabs(matrix.H*matrix-matrix.ctx.eye(matrix.rows))


def validate_matrix(matrix):
    a=np.asarray(matrix)
    if a.dtype not in (np.dtype('float64'),np.dtype('complex128')) or a.ndim!=2 or a.shape[0]!=a.shape[1] or not len(a) or len(a)%2 or not np.isfinite(a).all():
        raise ValueError('FINITE_BINARY64_EVEN_SQUARE_REQUIRED')
    if not np.array_equal(a,a.conj().T):
        raise ValueError('EXACT_HERMITIAN_INPUT_REQUIRED_NO_SYMMETRIZATION')
    if not np.array_equal(a,np.kron(a[::2,::2],np.eye(2))):
        raise ValueError('EXACT_INTERLEAVED_SPIN_LIFT_REQUIRED')
    return a


def checked_basis(n, indices):
    if indices is None:
        return tuple(range(1<<n))
    raw=tuple(indices)
    if not raw or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in raw):
        raise ValueError('INTEGER_BASIS_INDICES_REQUIRED')
    values=tuple(int(x) for x in raw)
    if len(set(values))!=len(values) or any(x<0 or x>=1<<n for x in values):
        raise ValueError('DISTINCT_BASIS_INDICES_REQUIRED')
    def sector(x):
        return (sum((x>>q)&1 for q in range(0,n,2)),sum((x>>q)&1 for q in range(1,n,2)))
    sectors={sector(x) for x in values}
    if set(values)!={x for x in range(1<<n) if sector(x) in sectors}:
        raise ValueError('COMPLETE_SPIN_SECTORS_REQUIRED_NO_PROJECTION')
    return values  # Preserve supplied Qiskit-sector permutation exactly.


def minor_determinant(ctx, matrix):
    """Small LU determinant with exact-zero pivots, no numerical rank cutoff.

    mpmath 1.3.0 det can fail on a structurally singular spin minor (None
    pivot). Keep this workaround local; never alter the installed library.
    """
    a=matrix.copy();out=ctx.mpf(1)
    for k in range(a.rows):
        pivot=max(range(k,a.rows),key=lambda i:abs(a[i,k]))
        if a[pivot,k]==0:
            return ctx.mpf(0)
        if pivot!=k:
            for j in range(a.cols):a[k,j],a[pivot,j]=a[pivot,j],a[k,j]
            out=-out
        out*=a[k,k]
        for i in range(k+1,a.rows):
            ratio=a[i,k]/a[k,k]
            for j in range(k+1,a.cols):a[i,j]-=ratio*a[k,j]
            a[i,k]=0
    return out


def fock_lift_hp(ctx, orbital, indices):
    """Existing exterior-minor definition with HP arithmetic and explicit order.

    The old helper hardcodes complex128/np.linalg.det. This small arithmetic
    adapter retains its occupations, minors and vacuum=1 definition.
    """
    n=orbital.rows; occupied=[tuple(q for q in range(n) if x&(1<<q)) for x in indices]
    out=ctx.matrix(len(indices))
    for i,left in enumerate(occupied):
        for j,right in enumerate(occupied):
            if len(left)!=len(right):
                continue
            out[i,j]=minor_determinant(ctx,ctx.matrix([[orbital[p,q] for q in right] for p in left])) if left else 1
    return out


@lru_cache(None)
def integer_bilinears(n, indices):
    """Reuse existing Qiskit-order CAR signs; no rounded molecular arithmetic."""
    annihilators=legacy.p.car_reference().annihilators(n)
    ix=np.ix_(indices,indices);terms=[]
    for p in range(n):
        for q in range(n):
            if p%2!=q%2:
                continue
            matrix=(annihilators[p].conj().T@annihilators[q])[ix].tocoo()
            for i,j,value in zip(matrix.row,matrix.col,matrix.data,strict=True):
                if value not in (-1,1):
                    raise ValueError('EXACT_INTEGER_CAR_REQUIRED')
                terms.append((p,q,int(i),int(j),int(value.real)))
    return tuple(terms)


def car_generator(ctx, orbital, indices):
    result=ctx.matrix(len(indices))
    for p,q,i,j,sign in integer_bilinears(orbital.rows,indices):
        result[i,j]+=sign*orbital[p,q]
    return result


class ExponentialArithmetic:
    """Minimal existing _ordered_exponentials arithmetic protocol."""
    def __init__(self,ctx): self.ctx=ctx
    def eye(self,n): return np.asarray(self.ctx.eye(n).tolist(),dtype=object)
    def exponential(self,time,generator):
        a=self.ctx.matrix(np.asarray(generator).tolist())
        return np.asarray(self.ctx.expm(-self.ctx.j*exact_real(self.ctx,Fraction(time))*a).tolist(),dtype=object)


@dataclass(frozen=True)
class PrecisionBlock:
    rotation: tuple
    basis: object
    energies: tuple


def prepare_precision(one_body, lambdas, factors, constant, *, digits=80, basis_indices=None):
    """Construct from fixed original factors, never from old eigensystems."""
    if isinstance(digits,bool) or not isinstance(digits,int) or digits<1:
        raise ValueError('POSITIVE_DECIMAL_PRECISION_REQUIRED')
    arrays=tuple(validate_matrix(x) for x in (one_body,*tuple(factors)))
    n=len(arrays[0]); weights=tuple(lambdas)
    if len(weights)!=len(arrays)-1 or any(a.shape!=(n,n) for a in arrays):
        raise ValueError('COMPATIBLE_SIGNED_FACTORS_REQUIRED')
    ctx=mpmath.mp.clone();ctx.dps=digits
    indices=checked_basis(n,basis_indices);coefficients=tuple(exact_real(ctx,x) for x in weights)
    matrices=tuple(hp_helpers().mp_matrix(ctx,a) for a in arrays)
    blocks=[];generators=[];diagnostics=[]
    for k,(array,matrix) in enumerate(zip(arrays,matrices,strict=True)):
        spatial=ctx.matrix([[matrix[2*i,2*j] for j in range(n//2)] for i in range(n//2)])
        eigenvalues,spatial_u=ctx.eighe(spatial)
        order=sorted(range(n//2),key=lambda i:abs(eigenvalues[i]),reverse=True)
        orbital=ctx.matrix(n);eta=[]
        for j,old_index in enumerate(order):
            eta.extend((eigenvalues[old_index],eigenvalues[old_index]))
            for i in range(n//2):
                orbital[2*i,2*j]=orbital[2*i+1,2*j+1]=spatial_u[i,old_index]
        basis=fock_lift_hp(ctx,orbital,indices)
        linear=tuple(ctx.fsum(eta[q] for q in range(n) if x&(1<<q)) for x in indices)
        energies=linear if k==0 else tuple(coefficients[k-1]*v*v for v in linear)
        blocks.append(PrecisionBlock(tuple(tuple(orbital[i,j] for j in range(n)) for i in range(n)),basis,energies))
        lifted=car_generator(ctx,matrix,indices)
        generator=lifted if k==0 else coefficients[k-1]*(lifted*lifted)
        generators.append(generator)
        diagnostics.append(dict(orbital_unitarity=unitarity(orbital),
            orbital_reconstruction=maxabs(orbital*ctx.diag(eta)*orbital.H-matrix),
            Fock_unitarity=unitarity(basis),
            generator_agreement=maxabs(basis*ctx.diag(energies)*basis.H-generator)))
    return SimpleNamespace(ctx=ctx,n_modes=n,indices=indices,model=SimpleNamespace(blocks=tuple(blocks)),
        constant=exact_real(ctx,constant),generators=tuple(generators),diagnostics=tuple(diagnostics),
        relative_cache={},exponential_cache={})


def precise_step(preparation, *, formula, substeps, delta_time=.02):
    """Same chronological schedule; independent HP native and CAR constructs."""
    p=preparation;ctx=p.ctx;blocks=p.model.blocks
    schedule=legacy.strong_baseline_schedule(len(blocks),substeps=substeps,formula=formula,delta_time=delta_time)
    groups=legacy.strong.native.runs(p.model,schedule,optimized=True)
    result=blocks[groups[0][0][0]].basis.H
    max_transition=ctx.mpf(0)
    for g,group in enumerate(groups):
        i=group[0][0]
        phases=[ctx.exp(-ctx.j*ctx.fsum(exact_real(ctx,t)*blocks[k].energies[row] for k,t in group))
                for row in range(len(p.indices))]
        result=ctx.matrix([[phases[row]*result[row,col] for col in range(result.cols)] for row in range(result.rows)])
        if g+1<len(groups):
            j=groups[g+1][0][0]
            if (i,j) not in p.relative_cache:
                left=ctx.matrix(blocks[j].rotation);right=ctx.matrix(blocks[i].rotation)
                relative=fock_lift_hp(ctx,left.H*right,p.indices)
                residual=maxabs(relative-blocks[j].basis.H*blocks[i].basis)
                p.relative_cache[i,j]=(relative,residual)
            relative,residual=p.relative_cache[i,j]
            max_transition=max(max_transition,residual);result=relative*result
        else:
            result=blocks[i].basis*result
    scalar=ctx.exp(-ctx.j*p.constant*exact_real(ctx,Fraction(delta_time)))
    native=scalar*result
    car_object=_ordered_exponentials(tuple(np.asarray(g.tolist(),dtype=object) for g in p.generators),
        schedule,p.exponential_cache,arithmetic=ExponentialArithmetic(ctx))
    car=scalar*ctx.matrix(car_object.tolist())
    return SimpleNamespace(ctx=ctx,native=native,car=car,schedule=schedule,
        diagnostics=dict(native_unitarity=unitarity(native),CAR_unitarity=unitarity(car),
            native_CAR_difference=maxabs(native-car),transition_homomorphism=max_transition),
        indices=p.indices,emitter_correctness='UNASSESSED',qualification='synthetic_or_explicitly_authorized_use_only')
