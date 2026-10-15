"""Explicit neutral ROHF input adapter and weighted DF projection.

No molecule, decomposition, state or file is obtained at import time. Artificial
tests inject providers; the real path is called only by the admitted runner.
"""
from decimal import Decimal, localcontext
import hashlib
import math
from pathlib import Path

import numpy as np
from .sector import spin_structure
from .contract import target


def array_identity(array):
    a=np.ascontiguousarray(array)
    if a.dtype.hasobject or not np.isfinite(a).all():raise ValueError('ARRAY_IDENTITY')
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())


def rohf_integrals(scratch,calls,*,provider=None):
    """Exactly one ROHF kernel; no implicit save or charged-chain helper."""
    scratch=Path(scratch).resolve()
    if not scratch.is_relative_to('/home/AbeHiromu') or not scratch.is_dir():raise ValueError('PRIVATE_SCRATCH')
    if provider is None:
        from pyscf import gto, scf
        from openfermionpyscf._run_pyscf import compute_integrals
        from openfermion.chem.molecular_data import spinorb_from_spatial
        provider=(gto.Mole,scf.ROHF,compute_integrals,spinorb_from_spatial)
    Mole,ROHF,compute_integrals,spinorb=provider;spec=target()
    calls.take('integral_build')
    mol=Mole();mol.atom=spec['geometry'];mol.basis=spec['basis'];mol.charge=0;mol.spin=1
    mol.unit='Angstrom';mol.symmetry=False;mol.verbose=0;mol.build()
    if (mol.nelectron,mol.nao_nr(),tuple(mol.nelec))!=(5,5,(3,2)) or str(mol.unit).lower()!='angstrom':
        raise ValueError('NEUTRAL_H5_ROHF_TARGET')
    mf=ROHF(mol);mf.verbose=0;mf.conv_tol=1e-9;mf.max_cycle=50
    mf.chkfile=str(scratch/'rohf.chk');calls.take('scf');mf.kernel()
    mo=np.asarray(mf.mo_coeff)
    if not mf.converged or not math.isfinite(float(mf.e_tot)):raise ValueError('ROHF_NOT_CONVERGED')
    if mo.shape!=(5,5) or not np.isfinite(mo).all() or np.iscomplexobj(mo):raise ValueError('COMMON_REAL_SPATIAL_MO')
    overlap=np.asarray(mol.intor_symmetric('int1e_ovlp'))
    if np.linalg.norm(mo.T@overlap@mo-np.eye(5))>1e-9:raise ValueError('MO_ORTHONORMALITY')
    one,two=compute_integrals(mol,mf);h1,h2=spinorb(one,two)
    arrays=dict(constant=np.array(mol.energy_nuc(),dtype='<f8'),one_body=np.asarray(h1,dtype='<c16'),
        two_body=np.asarray(.5*h2,dtype='<c16'),spatial_one_body=np.asarray(one),
        spatial_two_body=np.asarray(two),common_MO=mo.copy())
    if arrays['one_body'].shape!=(10,10) or arrays['two_body'].shape!=(10,)*4:raise ValueError('AO_MO_SPIN_LAYOUT')
    receipt=dict(target=spec,scf_converged=True,scf_energy=float(mf.e_tot),conv_tol=1e-9,max_cycle=50,
        integral_convention='OpenFermion compute_integrals (ps|qr), spinorb_from_spatial, two_body/2',
        orbital_order='returned ROHF MO columns; interleaved alpha,beta; no resort',
        arrays={k:array_identity(v) for k,v in arrays.items()},state_is_ROHF_determinant=False)
    return arrays,receipt


def weighted_df(constant,one_body,two_body,calls,*,decomposer=None):
    """Signed returned order, tol-only1e-8, cutoff0, same N-based projection rule."""
    one=np.asarray(one_body,dtype='<c16');two=np.asarray(two_body,dtype='<c16')
    n=len(one)
    if n%2 or not 2<=n<=14 or one.shape!=(n,n) or two.shape!=(n,)*4:
        raise ValueError('INPUT_INTEGRAL_LAYOUT')
    if not math.isfinite(float(constant)) or not np.isfinite(one).all() or not np.isfinite(two).all():
        raise ValueError('NONFINITE_INTEGRALS')
    if decomposer is None:
        from openfermion import low_rank_two_body_decomposition
        decomposer=low_rank_two_body_decomposition
    calls.take('df_decomposition')
    weights,raw,correction,truncation=decomposer(two.copy(),truncation_threshold=1e-8)
    weights=np.asarray(weights);raw=np.asarray(raw,dtype='<c16');correction=np.asarray(correction,dtype='<c16')
    if (weights.ndim!=1 or len(weights)<1 or np.iscomplexobj(weights) or not np.isfinite(weights).all()
        or raw.shape!=(len(weights),n,n) or correction.shape!=(n,n)
        or not np.isfinite(raw).all() or not np.isfinite(correction).all()
        or not math.isfinite(float(truncation)) or not 0<=float(truncation)<=1e-8):
        raise ValueError('DF_DECOMPOSITION_LAYOUT')
    weights=np.asarray(weights,dtype='<f8');corrected=one+correction
    # Check spin structure before projection; allow the weighted Hermiticity defect.
    def structure(a):
        if np.any(a[::2,1::2]!=0) or np.any(a[1::2,::2]!=0):raise ValueError('DF_CROSS_SPIN')
        if not np.array_equal(a[::2,::2],a[1::2,1::2]):raise ValueError('DF_ALPHA_BETA')
    for a in (one,correction,corrected,*raw):structure(a)
    projected=(corrected+corrected.conj().T)/2;blocks=(raw+raw.conj().transpose(0,2,1))/2
    for a in (projected,*blocks):spin_structure(a,n)
    dh=float(np.linalg.norm(projected-corrected));rows=[]
    with localcontext() as ctx:
        ctx.prec=60;D=Decimal.from_float;total=Decimal(0)
        for i,(w,g,p) in enumerate(zip(weights,raw,blocks,strict=True)):
            s=float(np.linalg.norm(g));d=float(np.linalg.norm(p-g))
            contribution=D(abs(float(w)))*D(d)*(2*D(s)+D(d));total+=contribution
            rows.append(dict(index=i,lambda_signed=float(w),raw=array_identity(g),after=array_identity(p),
                difference_F=d,weighted_contribution=str(contribution)))
        bounds={str(N):str(N*D(dh)+N*N*total) for N in (n,n//2)}
        if Decimal(bounds[str(n)])>Decimal('9.9e-11'):raise ValueError('WEIGHTED_PROJECTION_BUDGET')
    from ..h8_saved_projection_v1 import reconstruct, canonical_antisym
    before_one,before_two=reconstruct(corrected,weights,raw)
    after_one,after_two=reconstruct(projected,weights,blocks)
    from trotterlib.df_hamiltonian import DFHamiltonian
    receipt=dict(policy='H5_WEIGHTED_HERMITIAN_PROJECTION_V1',df_tol=1e-8,cutoff=0.,
        coefficient_order='decomposer_generation_order',actual_rank=len(weights),all_returned_fragments=True,
        primary_N=n,also_record_N=n//2,eta_N_hartree_decimal=bounds,decision_limit_hartree=9.9e-11,
        corrected_one_body_change_F=dh,fragments=rows,lambdas=array_identity(weights),
        provider_truncation=float(truncation),
        raw_integral_one_residual_F=float(np.linalg.norm(before_one-one)),
        raw_integral_two_antisym_residual_F=float(np.linalg.norm(before_two-canonical_antisym(two))),
        projected_coefficient_change_F=[float(np.linalg.norm(after_one-before_one)),float(np.linalg.norm(after_two-before_two))],
        formal_certificate=False,df_rank_matched_to_other_systems=False)
    ham=DFHamiltonian(float(constant),projected,weights,tuple(blocks),receipt)
    raw_arrays=dict(weights=weights,blocks=raw,correction=correction,truncation=np.array(truncation))
    return ham,receipt,raw_arrays
