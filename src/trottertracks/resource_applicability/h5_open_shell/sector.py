"""Unequal-spin occupation actions; all arrays are supplied by the caller.

Interleaved alpha/beta, OpenFermion big-endian occupation indices. The grouped
spin tensor product includes the fermionic permutation sign explicitly.
"""
from collections import OrderedDict
import math

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.linalg import expm

from ..h8_sparse_reference_v1 import OccupationCSR, small_one_body
from ..ax2b_independent_reference import _ladder
from trotterlib.df_hamiltonian import PhysicalSector, df_linear_operator


def sector_for(spatial, alpha, beta):
    if any(type(v) is not int for v in (spatial, alpha, beta)):
        raise ValueError('INTEGER_SECTOR')
    if not 1 <= spatial <= 7 or not 0 <= alpha <= spatial or not 0 <= beta <= spatial:
        raise ValueError('SECTOR_SCOPE')
    return PhysicalSector.spin_sector(n_qubits=2*spatial, nelec_alpha=alpha, nelec_beta=beta)


def spin_structure(matrix, n):
    a=np.asarray(matrix,dtype=complex)
    if a.shape!=(n,n) or not np.isfinite(a).all() or n%2:
        raise ValueError('SPIN_MATRIX')
    if np.any(a[::2,1::2]!=0) or np.any(a[1::2,::2]!=0):
        raise ValueError('CROSS_SPIN')
    if not np.array_equal(a[::2,::2],a[1::2,1::2]):
        raise ValueError('COMMON_SPATIAL_ORBITALS')
    if np.linalg.norm(a-a.conj().T)>1e-12:
        raise ValueError('HERMITICITY')
    return a


class SpinPrimitive:
    """Two independent occupation eigensystems, including when their sizes match."""
    def __init__(self, matrix, sector, *, weight=None, cache_size=32):
        n=sector.n_qubits;m=n//2
        expected=sector_for(m,sector.nelec_alpha,sector.nelec_beta)
        if not np.array_equal(expected.basis_indices,sector.basis_indices):
            raise ValueError('SECTOR_IDENTITY_OR_ORDER')
        a=spin_structure(matrix,n)
        if weight is not None and (np.iscomplexobj(weight) or not math.isfinite(float(weight))):
            raise ValueError('SIGNED_REAL_WEIGHT')
        if type(cache_size) is not int or not 0<=cache_size<=32:raise ValueError('CACHE_LIMIT')
        self.alpha_basis=tuple(i for i in range(1<<m) if i.bit_count()==sector.nelec_alpha)
        self.beta_basis=tuple(i for i in range(1<<m) if i.bit_count()==sector.nelec_beta)
        self.alpha_values,self.alpha_vectors=np.linalg.eigh(small_one_body(a[::2,::2],self.alpha_basis))
        self.beta_values,self.beta_vectors=np.linalg.eigh(small_one_body(a[1::2,1::2],self.beta_basis))
        ap={v:i for i,v in enumerate(self.alpha_basis)};bp={v:i for i,v in enumerate(self.beta_basis)}
        self.sides=(len(ap),len(bp));flat=[];signs=[]
        for index in sector.basis_indices:
            bits=[(int(index)>>(n-1-i))&1 for i in range(n)]
            alpha=sum(bits[2*i]<<(m-1-i) for i in range(m))
            beta=sum(bits[2*i+1]<<(m-1-i) for i in range(m))
            flat.append(ap[alpha]*len(bp)+bp[beta])
            swaps=sum(bits[2*i+1]*bits[2*j] for i in range(m) for j in range(i+1,m))
            signs.append(-1 if swaps%2 else 1)
        if len(flat)!=sector.dimension or len(set(flat))!=math.prod(self.sides):
            raise ValueError('SPIN_BASIS_BIJECTION')
        self.flat=np.array(flat);self.signs=np.array(signs)
        values=self.alpha_values[:,None]+self.beta_values[None,:]
        self.energies=values if weight is None else float(weight)*values**2
        self.cache_size=cache_size;self.phases=OrderedDict()

    def __call__(self, vector, time):
        x=np.asarray(vector,dtype=complex)
        if x.shape!=self.flat.shape or not np.isfinite(x).all() or not math.isfinite(time):
            raise ValueError('PRIMITIVE_INPUT')
        v=np.empty(math.prod(self.sides),dtype=complex);v[self.flat]=self.signs*x
        v=v.reshape(self.sides);a=self.alpha_vectors;b=self.beta_vectors
        phase=self.phases.pop(time,None)
        if phase is None:phase=np.exp(-1j*time*self.energies)
        if self.cache_size:
            self.phases[time]=phase
            if len(self.phases)>self.cache_size:self.phases.popitem(last=False)
        result=a@((a.conj().T@v@b.conj())*phase)@b.T
        return self.signs*result.reshape(-1)[self.flat]


def spin_squared(sector):
    """S^2 = Ms(Ms+1) + S- S+; S+ connects a different occupation sector."""
    n=sector.n_qubits;m=n//2
    if sector.dimension>1225 or n>14:raise ValueError('SPIN_DIAGNOSTIC_SCOPE')
    destinations={};rows=[];cols=[];values=[]
    for col,index in enumerate(sector.basis_indices):
        for orbital in range(m):
            removed=_ladder(int(index),2*orbital+1,n,False)
            if removed is None:continue
            created=_ladder(removed[0],2*orbital,n,True)
            if created is None:continue
            row=destinations.setdefault(created[0],len(destinations))
            rows.append(row);cols.append(col);values.append(removed[1]*created[1])
    plus=csr_matrix((values,(rows,cols)),shape=(len(destinations),sector.dimension),dtype=complex)
    ms=(sector.nelec_alpha-sector.nelec_beta)/2
    return (plus.conj().T@plus+ms*(ms+1)*eye(sector.dimension,dtype=complex)).tocsr()


def independent_df_matrix(ham,sector):
    if sector.dimension>100:raise ValueError('H5_DENSE_REFERENCE_SCOPE')
    oracle=OccupationCSR(sector.basis_indices,ham.n_qubits)
    result=ham.constant*np.eye(sector.dimension,dtype=complex)+oracle.one(ham.one_body).toarray()
    for weight,matrix in zip(ham.lambdas,ham.g_matrices,strict=True):
        g=oracle.one(matrix);result+=weight*(g@g).toarray()
    if np.linalg.norm(result-result.conj().T)>1e-10:raise ValueError('REFERENCE_HERMITICITY')
    return result


def validate_state(ham,sector,state,*,T=.8,maximum_matvecs=100):
    """Saved DF state checks, not ROHF determinant diagnostics or a certificate."""
    x=np.asarray(state,dtype=complex)
    if x.shape!=(sector.dimension,) or not np.isfinite(x).all() or abs(np.linalg.norm(x)-1)>1e-12:
        raise ValueError('STATE_NORMALIZATION')
    if maximum_matvecs<sector.dimension:raise ValueError('REFERENCE_MATVEC_BUDGET')
    independent=independent_df_matrix(ham,sector)
    operator,_=df_linear_operator(ham,sector,backend='python')
    columns=np.column_stack([operator@e for e in np.eye(sector.dimension,dtype=complex)])
    error=float(np.linalg.norm(columns-independent))
    if error>1e-9+1e-10*max(1.,np.linalg.norm(independent)):raise ValueError('INDEPENDENT_DF_ACTION')
    energy=float(np.real(np.vdot(x,independent@x)))
    residual=float(np.linalg.norm(independent@x-energy*x))
    if residual>1e-9+1e-10*max(1.,abs(energy)):raise ValueError('STATE_RESIDUAL')
    lowest=float(np.linalg.eigvalsh(independent)[0])
    if abs(energy-lowest)>1e-9+1e-10*max(1.,abs(lowest)):raise ValueError('STATE_LOWEST_ENERGY')
    ss=spin_squared(sector);s2=float(np.real(np.vdot(x,ss@x)))
    s2res=float(np.linalg.norm(ss@x-.75*x))
    if s2res>1e-9+1e-10:raise ValueError('DF_STATE_NOT_DOUBLET')
    evolved=expm(-1j*T*independent)@x
    split=expm(-1j*T/2*independent)@(expm(-1j*T/2*independent)@x)
    agreement=float(np.linalg.norm(evolved-split))
    if agreement>1e-10:raise ValueError('REFERENCE_SPLIT_TIME')
    return dict(energy=energy,residual=residual,S_squared=s2,S_squared_target_residual=s2res,
        S_squared_variance=float(np.linalg.norm(ss@x-s2*x)**2),
        action_difference=error,reference_split_difference=agreement,
        reference=complex(np.vdot(x,evolved)),reference_matvecs=sector.dimension,
        doublet_engineering_check=True,formal_certificate=False),independent
