"""Explicit neutral ROHF input adapter and weighted DF projection.

No molecule, decomposition, state or file is obtained at import time. Artificial
tests inject providers; the real path is called only by the admitted runner.
"""
from decimal import Decimal, localcontext
import hashlib
import math
from pathlib import Path

import numpy as np
from ..h5_open_shell.input import array_identity, weighted_df as _weighted_df
from .contract import target


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
    if (mol.nelectron,mol.nao_nr(),tuple(mol.nelec))!=(7,7,(4,3)) or str(mol.unit).lower()!='angstrom':
        raise ValueError('NEUTRAL_H7_ROHF_TARGET')
    mf=ROHF(mol);mf.verbose=0;mf.conv_tol=1e-9;mf.max_cycle=50
    mf.chkfile=str(scratch/'rohf.chk');calls.take('scf');mf.kernel()
    mo=np.asarray(mf.mo_coeff)
    if not mf.converged or not math.isfinite(float(mf.e_tot)):raise ValueError('ROHF_NOT_CONVERGED')
    if mo.shape!=(7,7) or not np.isfinite(mo).all() or np.iscomplexobj(mo):raise ValueError('COMMON_REAL_SPATIAL_MO')
    overlap=np.asarray(mol.intor_symmetric('int1e_ovlp'))
    if np.linalg.norm(mo.T@overlap@mo-np.eye(7))>1e-9:raise ValueError('MO_ORTHONORMALITY')
    one,two=compute_integrals(mol,mf);h1,h2=spinorb(one,two)
    arrays=dict(constant=np.array(mol.energy_nuc(),dtype='<f8'),one_body=np.asarray(h1,dtype='<c16'),
        two_body=np.asarray(.5*h2,dtype='<c16'),spatial_one_body=np.asarray(one),
        spatial_two_body=np.asarray(two),common_MO=mo.copy())
    if arrays['one_body'].shape!=(14,14) or arrays['two_body'].shape!=(14,)*4:raise ValueError('AO_MO_SPIN_LAYOUT')
    receipt=dict(target=spec,scf_converged=True,scf_energy=float(mf.e_tot),conv_tol=1e-9,max_cycle=50,
        integral_convention='OpenFermion compute_integrals (ps|qr), spinorb_from_spatial, two_body/2',
        orbital_order='returned ROHF MO columns; interleaved alpha,beta; no resort',
        arrays={k:array_identity(v) for k,v in arrays.items()},state_is_ROHF_determinant=False)
    return arrays,receipt


def weighted_df(*args,**kwargs):
    ham,receipt,raw=_weighted_df(*args,**kwargs)
    receipt['policy']='H7_WEIGHTED_HERMITIAN_PROJECTION_V1'
    receipt['same_projection_rule_as']='H5_WEIGHTED_HERMITIAN_PROJECTION_V1'
    return ham,receipt,raw
