#!/usr/bin/env python3
"""New independent H2/H6 input namespace; historical runs are read only."""
import hashlib, importlib, json, os
from pathlib import Path
import sys, types
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('hx_input');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules[pkg.__name__]=pkg
P=importlib.import_module(pkg.__name__+'.pauli_signal_adapter')
OUT=ROOT/'artifacts/pauli_h2_h6_comparison/2026-10-25'
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(n):
    dest=OUT/f'H{n}'/'inputs';dest.mkdir(parents=True,exist_ok=False)
    if n==2:
        old=ROOT/'artifacts/non_df_h2_pilot/2026-10-22/run01/inputs'
        d=json.loads((old/'pauli.json').read_text());receipt=json.loads((old/'receipt.json').read_text())
        a=np.load(old/'snapshot.npz');state=a['state'];energy=float(a['energy'])
        h=P.PauliHamiltonian(4,d['constant'],tuple(map(tuple,d['terms'])),json.dumps(d['metadata']))
        provenance=dict(reused=True,old_source='b5ed2c6e2399b55a0c24eb8437f842892a6703ef',
            old_files={str(p.relative_to(ROOT)):sha(p) for p in (old/'pauli.json',old/'receipt.json',old/'snapshot.npz')},
            old_run_STOP_not_reopened=True,receipt=receipt)
    elif n==6:
        from pyscf import ao2mo, fci, gto, lib, scf
        from scipy.sparse.linalg import eigsh
        lib.param.TMPDIR=str(dest)
        mol=gto.M(atom=[('H',(0,0,float(j))) for j in range(n)],unit='Angstrom',basis='sto-3g',charge=0,spin=0,
                  symmetry=False,output=str(dest/'pyscf.log'),verbose=4)
        mf=scf.RHF(mol);mf.conv_tol=1e-12;mf.max_cycle=100;mf.init_guess='minao';mf.chkfile=str(dest/'scf.chk')
        hf=float(mf.kernel())
        if not mf.converged:raise ValueError('SCF_NOT_CONVERGED')
        mo=np.asarray(mf.mo_coeff);one=mo.T@mf.get_hcore()@mo
        eri=ao2mo.restore(1,ao2mo.kernel(mol,mo),n);enuc=float(mol.energy_nuc())
        md=dict(molecule='H6',geometry=[['H',[0,0,float(j)]] for j in range(n)],distance=1.,basis='STO-3G',charge=0,spin=0,
                SCF_converged=True,hf_energy=hf,qubit_order='qiskit_little_endian',coefficient_cutoff=0.)
        h=P.PauliHamiltonian.from_spatial_integrals(one,eri,nuclear_repulsion=enuc,metadata=md,model_id='H6_PAULI_COMPARISON_20261025')
        full=h.sparse_operator()
        indices=np.array([i for i in range(1<<(2*n)) if sum((i>>j)&1 for j in range(0,2*n,2))==3 and sum((i>>j)&1 for j in range(1,2*n,2))==3])
        sector=full[indices][:,indices]
        vals,vec=eigsh(sector,k=1,which='SA',tol=1e-13,v0=np.ones(len(indices)))
        energy=float(vals[0]);v=vec[:,0];v*=np.exp(-1j*np.angle(v[np.argmax(abs(v))]));state=np.zeros(1<<(2*n),complex);state[indices]=v
        efci,ci=fci.direct_spin1.kernel(one,eri,n,(3,3),ecore=enuc,tol=1e-12,max_cycle=150)
        residual=float(np.linalg.norm(full@state-energy*state))
        if abs(energy-efci)>1e-9 or residual>1e-10:raise ValueError('PAULI_STATE_FCI_MISMATCH')
        np.savez_compressed(dest/'integrals_state.npz',mo_coeff=mo,one_mo=one,eri_chemist=eri,nuclear_repulsion=enuc,
                            state=state,energy=energy,sector_indices=indices,fci_energy=efci)
        provenance=dict(reused=False,reason='available historical snapshot has DF primitives/state, lacks non-DF MO integrals; new common Pauli input authorized',
                        SCF_calls=1,FCI_calls=1,DF_calls=0,JW_calls=1,sector_dimension=len(indices),
                        FCI_energy=float(efci),state_energy=energy,state_residual_empirical=residual,
                        MO_orthogonality=float(np.linalg.norm(mo.T@mf.get_ovlp()@mo-np.eye(n))))
    else:raise ValueError(n)
    terms=h.terms;m=len(terms);z=next((i for i,(s,c) in enumerate(terms) if not set(s)<=set('IZ')),m)
    raw=[[float(x.real),float(x.imag)] for x in state]
    model=dict(molecule=f'H{n}',n_qubits=2*n,qubit_order='qiskit_little_endian',state=raw,energy=energy,
               hamiltonian=dict(constant=h.constant,terms=terms,metadata=h.metadata),Hamiltonian_hash=h.hamiltonian_hash,
               nonidentity_terms=m,all_Z_prefix=z,physical_conditions=dict(distance=1.,basis='STO-3G',neutral=True,singlet=True,T=.8),
               provenance=provenance,state_hash=P.array_hash(state),state_renormalized=False)
    save(dest/'model.json',model)
    save(dest/'provenance.json',dict(model_sha256=sha(dest/'model.json'),state_hash=model['state_hash'],
                                   Hamiltonian_hash=h.hamiltonian_hash,nonidentity_terms=m,all_Z_prefix=z,provenance=provenance))
    print(json.dumps(dict(molecule=n,terms=m,all_Z_prefix=z,state_hash=model['state_hash'])),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
