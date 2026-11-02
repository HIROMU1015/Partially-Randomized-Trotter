"""H2 dimension adapter of fixed H4 F; generation requires an explicit H2 grant."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import h2_pr_resource_contract as c


def array_record(a):
    import numpy as np
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest())


def snapshot(path,arrays):
    import numpy as np
    c.require(not any(np.asarray(a).dtype.hasobject for a in arrays.values()),'NO_OBJECT_ARRAY')
    with path.open('xb') as f:np.savez(f,**arrays)
    with np.load(path,allow_pickle=False) as saved:
        c.require(set(saved.files)==set(arrays) and all(np.array_equal(saved[k],v) for k,v in arrays.items()),'SNAPSHOT_ROUNDTRIP')
    return dict(sha256=c.sha(path),bytes=path.stat().st_size,arrays={k:array_record(v) for k,v in arrays.items()})


def projected(one,weights,blocks,correction):
    """Same weighted triangle inequality as H6/H8; N=4 full-Fock budget.

    Float norms and Decimal accumulation are engineering checks, not interval
    certificates. Signed coefficients and every returned fragment are retained.
    """
    import numpy as np
    c.require(one.shape==(4,4) and blocks.shape==(len(weights),4,4) and correction.shape==(4,4),'DF_LAYOUT')
    c.require(weights.ndim==1 and np.isrealobj(weights) and all(np.isfinite(x).all() for x in (one,weights,blocks,correction)),'FINITE_SIGNED_DF')
    corrected=one+correction;post=(corrected+corrected.conj().T)/2;gs=(blocks+blocks.conj().transpose(0,2,1))/2
    for a in (corrected,*blocks):
        c.require(not np.any(a[::2,1::2]) and not np.any(a[1::2,::2]) and np.array_equal(a[::2,::2],a[1::2,1::2]),'SPIN_STRUCTURE')
    with localcontext() as ctx:
        ctx.prec=60;D=Decimal.from_float;dh=float(np.linalg.norm(post-corrected));terms=[]
        for w,g,p in zip(weights,blocks,gs,strict=True):
            d,s=float(np.linalg.norm(p-g)),float(np.linalg.norm(g))
            terms.append(D(abs(float(w)))*D(d)*(2*D(s)+D(d)))
        eta=4*D(dh)+16*sum(terms,Decimal(0))
        c.require(eta<=Decimal('9.9e-11'),'WEIGHTED_PROJECTION_BUDGET')
    return post,gs,dict(policy='WEIGHTED_HERMITIAN_PROJECTION_FROM_SAVED_RAW_V1',full_fock_N=4,
        eta_hartree_decimal=str(eta),decision_limit='9.9e-11',corrected_one_body_change=dh,
        fragment_terms=[str(t) for t in terms],signed_weights_preserved=True,
        all_returned_fragments=True,coefficient_cutoff=0.,status='PASS_ENGINEERING',representation_error_certified=False)


def generate(config,grant,output):
    c.verify(config,grant,interpreter=True)
    import numpy as np
    from scipy.linalg import eigh
    from openfermion import MolecularData,low_rank_two_body_decomposition
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf._run_pyscf import prepare_pyscf_molecule,compute_scf,compute_integrals
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector,df_linear_operator
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    from trotterlib.pr2_s0_s1_validation import _canonicalize_state_phase,_state_hash
    from trottertracks.resource_applicability.ax2b_independent_reference import occupation_df_matrix
    p=c.plan()
    c.save(output/'integral_build_reserved.json',dict(count=1,policy=p,nonrefundable=True))
    molecule=MolecularData([(a,tuple(v)) for a,v in p['geometry']],p['basis'],p['multiplicity'],p['charge'],
        filename=str(output/'scratch'/'molecule_not_saved'))
    mol=prepare_pyscf_molecule(molecule)
    c.require(str(mol.unit).lower()=='angstrom' and not mol.symmetry and mol.nao_nr()==2 and mol.nelectron==2,'H2_MOLECULE_SCOPE')
    mf=compute_scf(mol);mf.verbose=0;mf.conv_tol=p['scf_conv_tol'];mf.max_cycle=p['scf_max_cycle'];mf.chkfile=str(output/'scratch'/'scf.chk')
    c.reserve(output,'scf_kernel')
    mf.kernel();c.require(bool(mf.converged) and np.isfinite(mf.e_tot),'SCF_CONVERGENCE')
    c.reserve(output,'integral_build')
    spatial_one,spatial_two=compute_integrals(mol,mf);one,two=spinorb_from_spatial(spatial_one,spatial_two)
    one=np.asarray(one,dtype='<c16');two=np.asarray(.5*two,dtype='<c16');constant=float(mol.energy_nuc())
    integrals=snapshot(output/'integrals.npz',dict(constant=np.array(constant),one_body=one,two_body=two,
        spatial_one_body=spatial_one,spatial_two_body=spatial_two,canonical_orbitals=mf.mo_coeff))
    c.save(output/'integrals_receipt.json',dict(**integrals,hf_energy=float(mf.e_tot),converged=True,policy=p))
    c.save(output/'df_decomposition_reserved.json',dict(count=1,kwargs=dict(truncation_threshold=1e-8),final_rank_supplied=False,nonrefundable=True))
    c.reserve(output,'df_decomposition')
    weights,blocks,correction,truncation=low_rank_two_body_decomposition(two.copy(),truncation_threshold=1e-8)
    c.require(np.isfinite(truncation) and 0<=truncation<=1e-8 and not np.any(np.imag(weights)),'DF_TRUNCATION_AND_REAL_WEIGHT')
    weights=np.asarray(weights.real,dtype='<f8');blocks=np.asarray(blocks,dtype='<c16');correction=np.asarray(correction,dtype='<c16')
    raw=snapshot(output/'raw_df.npz',dict(lambdas=weights,g_matrices=blocks,correction=correction))
    corrected,gs,projection=projected(one,weights,blocks,correction)
    md=dict(input_policy='TOL_ONLY_NO_CONFIG_FALLBACK',df_tol_requested=1e-8,df_rank_actual=len(weights),
        decomposition_kwargs=dict(truncation_threshold=1e-8),final_rank_supplied=False,coefficient_order='decomposer_generation_order',
        subsequent_coefficient_cutoff=0.,df_truncation_value=float(truncation),projection=projection)
    ham=DFHamiltonian(constant,corrected,weights,tuple(gs),md)
    sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
    op,_=df_linear_operator(ham,sector,backend='python')
    c.require(sector.dimension==4,'H2_SPIN_SECTOR')
    c.reserve(output,'sector_matrix_columns',4)
    eye=np.eye(4,dtype=complex);matrix=np.column_stack([op@v for v in eye])
    c.reserve(output,'independent_occupation_matrix')
    independent=np.asarray(occupation_df_matrix(ham.constant,ham.one_body,ham.lambdas,ham.g_matrices,sector.basis_indices),dtype=complex)
    error=float(np.linalg.norm(matrix-independent));c.require(error<1e-9,'DF_INDEPENDENT_OCCUPATION_ORACLE')
    herm=float(np.linalg.norm(matrix-matrix.conj().T));c.require(herm<1e-10,'DF_SECTOR_HERMITICITY')
    c.reserve(output,'solver_eigh')
    eigenvalues,vectors=eigh(matrix)
    gap=float(eigenvalues[1]-eigenvalues[0])
    c.require(gap>p['minimum_ground_gap'],'DEGENERATE_STATE_POLICY_REQUIRES_NEW_CONTRACT')
    state=np.asarray(vectors[:,0],dtype='<c16');full=np.zeros(16,dtype='<c16');full[sector.basis_indices]=state
    full,state=_canonicalize_state_phase(full,state)
    c.require(abs(float(np.linalg.norm(state))-1)<p['state_norm_tolerance'],'STATE_NORMALIZATION')
    residual=float(np.linalg.norm(matrix@state-eigenvalues[0]*state));c.require(residual<1e-10,'SAVED_STATE_RESIDUAL')
    arrays=dict(constant=np.array(constant,dtype='<f8'),one_body=corrected,lambdas=weights,g_matrices=gs,
        sector_basis_indices=np.asarray(sector.basis_indices,dtype='<i8'),state_vector=full,sector_state_vector=state)
    receipt=snapshot(output/'tol_only_snapshot.npz',arrays)
    result=dict(SOURCE=config['SOURCE'],scope=c.SCOPE,stage='F',status='COMPLETE_MANDATORY_STOP',**receipt,rank=len(weights),hamiltonian_metadata=md,
        hamiltonian_hash=df_hamiltonian_hash(ham),state_hash=_state_hash(full,state),projection=projection,
        independent_sector_error=error,hermiticity_error=herm,state_energy=float(eigenvalues[0]),state_residual=residual,state_gap=gap,
        sector=dict(n_qubits=4,nelec_alpha=1,nelec_beta=1,dimension=4),generation_policy=p,
        state_renormalized_after_saving=False,ground_state_certified=False,raw_df_receipt=raw,integrals_receipt=integrals,
        actual_calls=c.actual_calls(output),legacy_input_modified=False)
    c.save(output/'tol_only_receipt.json',result)
    return result


def load(config):
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    from trotterlib.pr2_s0_s1_validation import _state_hash
    path=c.owned(config['mapping']['tol_snapshot']);receipt=c.read(c.owned(config['mapping']['tol_receipt']))
    c.require(receipt['scope']==c.SCOPE and receipt['generation_policy']==c.plan(),'H2_SAVED_INPUT_POLICY')
    c.require(c.sha(path)==receipt['sha256'] and receipt['status']=='COMPLETE_MANDATORY_STOP','TOL_INPUT_COMPLETION')
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    c.require({k:array_record(v) for k,v in arrays.items()}==receipt['arrays'],'TOL_ARRAY_IDENTITY')
    ham=DFHamiltonian(float(arrays['constant']),arrays['one_body'],arrays['lambdas'],tuple(arrays['g_matrices']),receipt['hamiltonian_metadata'])
    sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
    full,state=arrays['state_vector'],arrays['sector_state_vector']
    c.require(np.array_equal(arrays['sector_basis_indices'],sector.basis_indices) and np.array_equal(full[sector.basis_indices],state),'TOL_STATE_LIFT')
    c.require(abs(np.linalg.norm(state)-1)<1e-12 and df_hamiltonian_hash(ham)==receipt['hamiltonian_hash'] and
              _state_hash(full,state)==receipt['state_hash'],'TOL_HAMILTONIAN_STATE_IDENTITY')
    return ham,sector,full,state,dict(hamiltonian_hash=receipt['hamiltonian_hash'],state_hash=receipt['state_hash'],rank=receipt['rank'],projection=receipt['projection'])
