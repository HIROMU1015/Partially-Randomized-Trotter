"""Bounded unequal-spin CSR reference, with no dense sector/Fock Hamiltonian.

The production excitation-table action and independent ladder-enumerated CSR
action remain separate. Numeric work needs caller-supplied admitted arrays.
"""
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from ..h5_open_shell.sector import sector_for, SpinPrimitive, spin_squared
from ..h8_sparse_reference_v1 import OccupationCSR, counted_operator, reference_pair


def bounded_solver(action,dimension,initial,calls,*,solver=eigsh,category='solver_matvec'):
    if type(dimension) is not int or not 3<=dimension<=1225:
        raise ValueError('H7_SOLVER_DIMENSION')
    x=np.asarray(initial,dtype=complex)
    if x.shape!=(dimension,) or not np.isfinite(x).all() or abs(np.linalg.norm(x)-1)>1e-12:
        raise ValueError('H7_SOLVER_INITIAL_STATE')
    def apply(v):
        calls.take(category)
        y=np.asarray(action(np.asarray(v).reshape(dimension)),dtype=complex)
        if y.shape!=(dimension,) or not np.isfinite(y).all():raise ValueError('H7_SOLVER_ACTION')
        return y.copy()  # Production numba kernels may reuse an output workspace.
    op=LinearOperator((dimension,dimension),matvec=apply,rmatvec=apply,dtype=complex)
    values,vectors=solver(op,k=1,which='SA',tol=1e-12,maxiter=1000,ncv=min(40,dimension),v0=x.copy())
    values=np.asarray(values);vectors=np.asarray(vectors)
    if (values.shape!=(1,) or vectors.shape!=(dimension,1) or not np.isfinite(values).all()
        or not np.isfinite(vectors).all() or np.any(values.imag!=0)):
        raise ValueError('H7_SOLVER_RESULT')
    return (values,vectors),op


class SparseReference:
    def __init__(self,ham,sector,calls):
        if ham.n_qubits!=sector.n_qubits or sector.dimension>1225:raise ValueError('H7_CSR_SCOPE')
        # Validates common spatial blocks, cross-spin zeros and Hermiticity.
        from ..h5_open_shell.sector import spin_structure
        spin_structure(ham.one_body,ham.n_qubits)
        for g in ham.g_matrices:spin_structure(g,ham.n_qubits)
        if not np.isfinite(ham.constant) or not np.isfinite(ham.lambdas).all():raise ValueError('H7_CSR_NONFINITE')
        calls.take('reference_preparation')
        self.ham,self.sector,self.calls=ham,sector,calls
        topology=OccupationCSR(sector.basis_indices,ham.n_qubits)
        self.one=topology.one(ham.one_body)
        self.blocks=tuple(topology.one(g) for g in ham.g_matrices)
        for g in (self.one,*self.blocks):
            for a in (g.data,g.indices,g.indptr):a.flags.writeable=False

    def raw_action(self,x,*,prefix=None,tail_from=None,identity=0.,scale=1.):
        n=self.sector.dimension;x=np.asarray(x,dtype=complex)
        if x.shape!=(n,) or not np.isfinite(x).all() or not math.isfinite(scale) or scale<=0:
            raise ValueError('H7_REFERENCE_ACTION_INPUT')
        if tail_from is None:
            indices=range(len(self.blocks) if prefix is None else prefix)
            y=self.ham.constant*x+self.one@x
        else:
            indices=range(tail_from,len(self.blocks));y=-identity*x
        for i in indices:y=y+self.ham.lambdas[i]*(self.blocks[i]@(self.blocks[i]@x))
        out=y/scale
        if not np.isfinite(out).all():raise ValueError('H7_REFERENCE_ACTION_NONFINITE')
        return out

    def operator(self,*,prefix=None,tail_from=None,identity=0.,scale=1.,budget=None):
        n=self.sector.dimension
        if tail_from is None:
            indices=range(len(self.blocks) if prefix is None else prefix)
            trace=self.ham.constant*n+float(self.one.diagonal().real.sum())
        else:
            indices=range(tail_from,len(self.blocks));trace=-identity*n
        trace+=sum(float(self.ham.lambdas[i])*float(np.sum(np.abs(self.blocks[i].data)**2)) for i in indices)
        action=lambda x:self.raw_action(x,prefix=prefix,tail_from=tail_from,identity=identity,scale=scale)
        return counted_operator(n,action,trace/scale,self.calls if budget is None else budget)

    def pair(self,state,T,*,prefix=None):
        self.calls.take('reference_exponential',3)
        op,trace=self.operator(prefix=prefix)
        return reference_pair(op,trace,state,T)

    def validate(self,state,production,*,T=.8,solver=eigsh):
        x=np.asarray(state,dtype=complex);n=self.sector.dimension
        if x.shape!=(n,) or not np.isfinite(x).all() or abs(np.linalg.norm(x)-1)>1e-12:
            raise ValueError('H7_STATE_NORMALIZATION')
        error2=norm2=0.;basis=np.zeros(n,dtype=complex)
        op,_=self.operator()
        for i in range(n):
            basis.fill(0);basis[i]=1
            self.calls.take('action_check_matvec')
            actual=np.asarray(production@basis,dtype=complex).copy();independent=op@basis
            if actual.shape!=(n,) or not np.isfinite(actual).all():raise ValueError('H7_PRODUCTION_ACTION')
            error2+=float(np.linalg.norm(actual-independent))**2
            norm2+=float(np.linalg.norm(independent))**2
        error=math.sqrt(error2)
        if error>1e-9+1e-10*max(1.,math.sqrt(norm2)):raise ValueError('H7_INDEPENDENT_DF_ACTION')
        energy=float(np.real(np.vdot(x,op@x)));residual=float(np.linalg.norm(op@x-energy*x))
        if residual>1e-9+1e-10*max(1.,abs(energy)):raise ValueError('H7_STATE_RESIDUAL')
        # A deterministic generic vector also has components in other total-spin
        # sectors. This is an engineering eigensolver check, not a certificate.
        rng=np.random.default_rng(707123);v=rng.normal(size=n)+1j*rng.normal(size=n);v/=np.linalg.norm(v)
        self.calls.take('reference_eigensolve')
        # The CSR operator already charges every underlying action; do not wrap
        # it in another shared counter and double charge those same matvecs.
        values,vectors=solver(op,k=1,which='SA',tol=1e-12,maxiter=1000,ncv=min(40,n),v0=v)
        values=np.asarray(values);vectors=np.asarray(vectors)
        if values.shape!=(1,) or vectors.shape!=(n,1) or not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise ValueError('H7_REFERENCE_SOLVER_RESULT')
        if abs(np.linalg.norm(vectors[:,0])-1)>1e-12:raise ValueError('H7_REFERENCE_SOLVER_NORM')
        low=float(values[0].real);lowres=float(np.linalg.norm(op@vectors[:,0]-low*vectors[:,0]))
        if np.any(values.imag!=0) or lowres>1e-9+1e-10*max(1.,abs(low)) or abs(energy-low)>1e-9+1e-10*max(1.,abs(low)):
            raise ValueError('H7_STATE_LOWEST_ENERGY')
        ss=spin_squared(self.sector);s2=float(np.real(np.vdot(x,ss@x)));sres=float(np.linalg.norm(ss@x-.75*x))
        if sres>1.1e-9:raise ValueError('H7_DF_STATE_NOT_DOUBLET')
        reference,check=self.pair(x,T)
        return dict(energy=energy,residual=residual,S_squared=s2,S_squared_target_residual=sres,
            S_squared_variance=float(np.linalg.norm(ss@x-s2*x)**2),action_difference=error,
            action_columns_checked=n,reference=reference,reference_split_difference=check['state_difference'],
            reference_checks=check,reference_lowest_energy=low,reference_lowest_residual=lowres,
            sparse_entries=self.one.nnz+sum(g.nnz for g in self.blocks),
            doublet_engineering_check=True,formal_certificate=False,dense_sector_matrix=False)
