"""Independent saved-data checks. No molecular engine, SVD, or eigensolver.

Floating-point checks are technical evidence, not interval certificates.
The CAR implementation here does not use the generator's normal ordering,
vectorized ladder maps, Pauli algebra, or Gaussian kernels.
"""
import hashlib
import itertools
import json
from pathlib import Path
import math

import numpy as np


class EvidenceError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_hamiltonian(h, eri, constant):
    n = 2*len(h); dim = 1 << n
    matrix = np.eye(dim, dtype=complex)*constant
    def add(ops, value):
        if value == 0: return
        for column in range(dim):
            state = column; sign = 1
            for mode, create in reversed(ops):
                occupied = bool(state & (1 << mode))
                if occupied == bool(create): break
                sign *= -1 if (state & ((1 << mode)-1)).bit_count() % 2 else 1
                state ^= 1 << mode
            else:
                matrix[state, column] += value*sign
    for p, q in itertools.product(range(len(h)), repeat=2):
        for spin in (0, 1):
            add(((2*p+spin,1),(2*q+spin,0)), h[p,q])
    for p,q,r,s in itertools.product(range(len(h)), repeat=4):
        for alpha,beta in itertools.product((0,1),repeat=2):
            add(((2*p+alpha,1),(2*r+beta,1),(2*s+beta,0),(2*q+alpha,0)), eri[p,q,r,s]/2)
    return matrix


def check_norm(error, sigma, vh, declared, tolerance=1e-8):
    error,sigma,vh = map(np.asarray,(error,sigma,vh))
    k = error.shape[1] if error.ndim == 2 else -1
    if sigma.shape != (k,) or vh.shape != (k,k) or k < 1:
        raise EvidenceError('NORM_SHAPE')
    if not all(np.all(np.isfinite(x)) for x in (error,sigma,vh)) or not math.isfinite(declared):
        raise EvidenceError('NONFINITE_NORM')
    if np.any(sigma < 0) or np.any(sigma[:-1] < sigma[1:]) or abs(float(sigma[0])-declared)>tolerance:
        raise EvidenceError('SINGULAR_VALUES')
    v = vh.conj().T; gram = error.conj().T @ error
    defect = float(np.linalg.norm(v.conj().T@v-np.eye(k),'fro'))
    residual = float(np.linalg.norm(gram@v-v*sigma[None,:]**2,'fro'))
    gf = float(np.linalg.norm(gram,'fro'))
    if defect > 1e-10 or residual > 1e-10*max(1.,gf):
        raise EvidenceError('NORM_WITNESS_RESIDUAL')
    lower = float(np.linalg.norm(error@v[:,0])/np.linalg.norm(v[:,0]))
    upper = math.sqrt(max(0., (1+defect)*float(sigma[0])**2 + residual*math.sqrt(1+defect)+gf*defect))
    if declared < lower-tolerance or upper > declared+tolerance:
        raise EvidenceError('NORM_WITNESS_ENCLOSURE')
    return {'lower_float':lower,'upper_float':upper,'unitarity_defect_fro':defect,
            'gram_residual_fro':residual,'SVD_calls':0,
            'qualification':'binary64 residual enclosure;not outward-rounded certificate'}


def check_raw(directory, contract):
    directory = Path(directory)
    seal = json.loads((directory/'raw_seal.json').read_text())
    if seal.get('status') != 'SEALED_BEFORE_SELECTION':
        raise EvidenceError('RAW_NOT_SEALED')
    for name,digest in seal['files'].items():
        path = directory/name
        if path.resolve().parent != directory.resolve() or sha256(path)!=digest:
            raise EvidenceError('RAW_HASH:'+name)
    with np.load(directory/'input.npz',allow_pickle=False) as pack:
        h,eri,constant = pack['h'],pack['eri'],float(pack['constant'])
    target = contract['target']; n=2*len(h); dim=1<<n
    if n!=target['spin_orbitals'] or dim!=contract['limits']['full_Fock_dimension']:
        raise EvidenceError('TARGET_SHAPE')
    H = raw_hamiltonian(h,eri,constant)
    with np.load(directory/'derived.npz',allow_pickle=False) as data:
        if float(np.linalg.norm(H-data['H'],'fro')) > 1e-8:
            raise EvidenceError('ORIGINAL_HAMILTONIAN')
        if 'df_factors' in data:
            rebuilt=sum((lam*np.einsum('pq,rs->pqrs',f,f) for lam,f in zip(data['df_lambdas'],data['df_factors'])),np.zeros_like(eri))
            if np.max(np.abs(rebuilt-eri))>1e-10:
                raise EvidenceError('SAVED_DF_COEFFICIENTS')
    columns = [i for i in range(dim) if i.bit_count()==target['particles']]
    if len(columns)!=contract['limits']['nu_input_columns']:
        raise EvidenceError('SECTOR_COLUMNS')
    probes=json.loads((directory/'probes.json').read_text()); expected = {
        (m,t,r) for m in contract['methods'] for t in contract['grid']['times_atomic'] for r in contract['grid']['steps']}
    if len(probes)!=len(expected) or {(p['method'],p['T'],p['r']) for p in probes}!=expected:
        raise EvidenceError('PROBE_COVERAGE')
    witnesses=[]
    for row in probes:
        name=row['file']
        if name!=f"error_{row['method']}_T{row['T']}_r{row['r']}.npz":
            raise EvidenceError('PROBE_IDENTITY')
        if name not in seal['files']:
            raise EvidenceError('UNSEALED_PROBE')
        with np.load(directory/name,allow_pickle=False) as data:
            if data['error'].shape!=(dim,len(columns)) or not np.array_equal(data['columns'],columns):
                raise EvidenceError('FULL_OUTPUT_SECTOR_METRIC')
            witnesses.append(check_norm(data['error'],data['sigma'],data['vh'],row['error']))
    return {'status':'PASS','raw_files':len(seal['files']),'probes':len(probes),
            'worst_witness_excess':max(w['upper_float']-p['error'] for w,p in zip(witnesses,probes)),
            'independent_raw_CAR':True,'molecular_jobs':0,'SVD_calls':0,
            'limitation':'saved error witnesses do not independently regenerate every PF trajectory'}
