"""H4 contract, IO, explicit rejection and artificial integration tests."""
import builtins
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from trotterlib.h4_execution_contract import completed_contract,checked_pack,activate,CASES,seed_for,require_current_contract
from trotterlib.h4_molecular_inputs import dimension_gate
from trotterlib.h4_pilot_science import artificial_inputs,save_inputs,load_inputs,integrate
from trotterlib.isolated_qiskit_execution import sha,save,read,digest
ROOT=Path(__file__).resolve().parents[1]


def test_completed_candidate_fixed_scope_budgets_inactive():
    p=completed_contract()
    assert not any(p[k] for k in ['execution_authorized','grant_valid','dispatch_enabled'])
    assert p['readiness']['scientific_dispatch_implemented']
    assert p['execution']['call_budget']['full_wrapper_compile_axes']==1028
    assert p['execution']['call_budget']['production_RTE_events']==16384
    assert p['evolution']==read(ROOT/'artifacts/non_df_h4_readiness/2026-10-23/H4_execution_contract_DRAFT.json')['evolution']
    assert p['MC']['trajectories_per_PR']==256
    assert p['execution']['watch_policy']['run_timeout'] is None
    assert len(set(seed_for(*case.rsplit('_',1)) for case in CASES))==4


@pytest.mark.parametrize('field',['evolution','MC','numerical','environment','fixed_candidates'])
def test_contract_mutation_cannot_select_new_scientific_conditions(field):
    p=completed_contract();p[field]['changed']=True
    with pytest.raises(ValueError,match='CONTRACT_CHANGED'):require_current_contract(p)


@pytest.mark.parametrize('shape',[(2,2),(4,3),(8,8),(0,0)])
def test_H4_dimension_gate_rejects_other_systems(shape):
    with pytest.raises(ValueError,match='FOUR_SPATIAL'):dimension_gate(np.zeros(shape))


def test_H4_dimension_gate_36_sector_and_nonfinite():
    s=dimension_gate(np.eye(4));assert s.dimension==36 and s.n_qubits==8
    bad=np.eye(4);bad[0,0]=np.nan
    with pytest.raises(ValueError):dimension_gate(bad)


def test_unapproved_execute_rejected_before_science_import(tmp_path,monkeypatch):
    original=builtins.__import__
    def observe(name,*args,**kwargs):
        if name in {'trotterlib.h4_pilot_science','trotterlib.h4_molecular_inputs','pyscf','qiskit'}:
            raise AssertionError('SCIENCE_IMPORT_BEFORE_GRANT')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',observe)
    with pytest.raises(PermissionError,match='NO_VALID_GRANT'):checked_pack(tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def test_cli_rejects_H4_dispatch_with_no_grant(tmp_path):
    p=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h4_non_df_pilot.py'),
        '--execute','--pack',str(tmp_path/'never_created')],capture_output=True,text=True)
    assert p.returncode!=0 and 'H4_NO_VALID_GRANT' in p.stderr
    assert not (tmp_path/'never_created').exists()


def test_grant_symlink_rejected_without_target_mutation(tmp_path):
    target=tmp_path/'target';target.mkdir();pack=tmp_path/'alias';pack.symlink_to(target,target_is_directory=True)
    with pytest.raises(PermissionError):checked_pack(pack)
    assert list(target.iterdir())==[]


def test_activation_requires_exact_explicit_approval_and_clean_SOURCE(tmp_path,monkeypatch):
    import trotterlib.h4_execution_contract as c
    plan=tmp_path/'plan.json';approval=tmp_path/'approval.json';pack=tmp_path/'pack'
    save(plan,completed_contract());save(approval,{'approve_H4_input_and_four_cases_once':False})
    monkeypatch.setattr(c,'git',lambda *a:'test_invalid_SOURCE' if a[0]=='rev-parse' else '')
    with pytest.raises(ValueError,match='APPROVAL'):activate(plan,approval,pack)
    assert not pack.exists()
    monkeypatch.setattr(c,'git',lambda *a:'dirty')
    with pytest.raises(ValueError,match='CLEAN'):activate(plan,approval,pack)
    assert not pack.exists()


@pytest.fixture
def sealed_artificial(tmp_path):
    p,d,s,r=artificial_inputs();folder=tmp_path/'inputs'
    receipt=save_inputs(folder,p,d,s,r,artificial=True)
    expected={'receipt_sha256':sha(folder/'receipt.json'),'snapshot_sha256':sha(folder/'snapshot.npz')}
    return folder,expected,receipt


def test_hashed_snapshot_roundtrip_and_fixed_prefix(sealed_artificial):
    folder,b,receipt=sealed_artificial;p,d,s,r=load_inputs(folder,b)
    assert len(s)==256 and p.num_qubits==d.n_qubits==8
    assert r['prefixes']==receipt['prefixes'] and r['artificial_only']
    assert 'one_body_always_retained' in r['prefixes']['df_native']


@pytest.mark.parametrize('name',['snapshot.npz','pauli.json','receipt.json'])
def test_snapshot_corruption_stops_before_cases(sealed_artificial,name):
    folder,b,receipt=sealed_artificial;p=folder/name;p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='CHANGED'):load_inputs(folder,b)


def test_integration_never_invents_missing_cases(tmp_path):
    r=integrate(tmp_path,completed_contract(),{})
    assert r['status']=='PARTIAL_RESULTS_NOT_FOUR_METHOD_COMPARISON'
    assert r['rows']==[] and r['missing_cases']==list(CASES)


def test_wrong_artificial_plan_cannot_enter_molecular_dispatch(tmp_path):
    from trotterlib.h4_pilot_execution import binding,synthetic_plan
    p=synthetic_plan();p['scope']='H4'
    save(tmp_path/'artificial_config.json',{'scope':'ARTIFICIAL_EIGHT_QUBIT_ONLY','plan':p,'SOURCE':'TEST'})
    with pytest.raises(PermissionError):binding(tmp_path,True)


def test_primary_N_G_null_when_allowance_exceeds_headroom():
    from trotterlib.h2_non_df_pilot import precision_accounting
    result=dict(signal={'signed_bias':{'real':0.,'imag':0.}},
        shots={'axes':{a:{'allowance':1.} for a in ['cosine','sine']},'evidence_ref':'ARTIFICIAL'},
        normalization={'log_B_upper':0.},circuit_cost={'samples':[]})
    plan={'evaluation':{'alpha_axis':.025},'numerical':{'maximum_u_over_headroom':.01}}
    row=precision_accounting(result,.0005,plan)
    assert row['total_RZ'] is None and row['shots']['N_total'] is None


def test_molecular_generation_path_with_mock_provider_only(tmp_path,monkeypatch):
    """Run real H4 shape/operator/snapshot logic, mock ALL molecule/FCI/DF calls."""
    from types import SimpleNamespace
    import pyscf.gto,pyscf.scf,pyscf.ao2mo,pyscf.fci
    import trotterlib.df_hamiltonian as dfmod
    from trotterlib.h4_molecular_inputs import build_h4_inputs
    h=np.diag([-.3,-.1,.2,.4]);U=np.array([.01,.02,.03,.04]);eri=np.zeros((4,4,4,4))
    for i,u in enumerate(U):eri[i,i,i,i]=u
    counts={'SCF':0,'DF':0,'FCI':0,'integral':0}
    class Molecule:
        def build(self):return self
        def energy_nuc(self):return .2
    class RHF:
        mo_coeff=np.eye(4);converged=True
        def __init__(self,mol):pass
        def kernel(self):counts['SCF']+=1;return -.57
        def get_hcore(self):return h
        def get_ovlp(self):return np.eye(4)
    def ao(*a,**kw):counts['integral']+=1;return eri.copy()
    def mock_df(*,constant,one_body,two_body,df_rank,df_tol,metadata):
        counts['DF']+=1
        assert df_rank is None and df_tol==1e-8 and one_body.shape==(8,8)
        gs=[]
        for i in range(4):
            g=np.zeros((8,8));g[2*i,2*i]=g[2*i+1,2*i+1]=1.;gs.append(g)
        return dfmod.DFHamiltonian(constant,one_body-np.diag(np.repeat(U/2,2)),U/2,tuple(gs),metadata)
    def mock_fci(*a,**kw):
        counts['FCI']+=1
        assert a[2:]==(4,(2,2))
        return 2*(h[0,0]+h[1,1])+U[0]+U[1]+.2,np.zeros((6,6))
    monkeypatch.setattr(pyscf.gto,'Mole',Molecule)
    monkeypatch.setattr(pyscf.scf,'RHF',RHF)
    monkeypatch.setattr(pyscf.ao2mo,'kernel',ao)
    monkeypatch.setattr(pyscf.ao2mo,'restore',lambda *a:eri.copy())
    monkeypatch.setattr(pyscf.fci.direct_spin1,'kernel',mock_fci)
    monkeypatch.setattr(dfmod,'df_hamiltonian_from_integrals',mock_df)
    import pyscf.lib
    monkeypatch.setattr(pyscf.lib.param,'TMPDIR',str(tmp_path))
    p,d,state,receipt=build_h4_inputs(tmp_path/'data',completed_contract())
    assert counts=={'SCF':1,'DF':1,'FCI':1,'integral':1}
    assert p.num_qubits==d.n_qubits==8 and receipt['physical_sector_dimension']==36
    assert receipt['all_gates_passed'] and (tmp_path/'data/snapshot.npz').exists()


@pytest.mark.parametrize('mutation',['SOURCE','contract_digest','grant_valid','dispatch_enabled','pack_namespace','authorized_cases','grant_id'])
def test_authority_mutations_refused_without_issuing_grant(tmp_path,monkeypatch,mutation):
    import trotterlib.h4_execution_contract as c
    # In-memory records with a deliberately non-Git SOURCE, never an issued grant.
    plan=c.completed_contract();source={'SOURCE':'TEST_ONLY_NOT_A_GIT_COMMIT'}
    approval=dict(schema='h4_explicit_batch_approval_v1',SOURCE=source['SOURCE'],
        contract_digest=digest(plan),pack_namespace=str(tmp_path),approve_H4_input_and_four_cases_once=True,
        adopted_proposal=True,retry=False,push=False,scientific_review=False)
    monkeypatch.setattr(c,'sha',lambda p:'IN_MEMORY_TEST_HASH')
    grant=dict(schema='h4_one_shot_grant_v1',SOURCE=source['SOURCE'],contract_digest=digest(plan),
        source_manifest_sha256='IN_MEMORY_TEST_HASH',approval_sha256='IN_MEMORY_TEST_HASH',
        pack_namespace=str(tmp_path),claim_directory=str(c.ROOT/'.h4_execution_claims'/digest(approval)),
        authorized_cases=list(c.CASES),execution_authorized=True,grant_valid=True,dispatch_enabled=True,
        one_shot=True,automatic_retry=False,mandatory_STOP=True)
    grant['grant_id']=digest(grant)
    c.verify_authority_records(tmp_path,plan,source,approval,grant)
    grant[mutation]='CORRUPT_OR_RENAMED_RUN'
    with pytest.raises(PermissionError,match='GRANT_BINDING'):
        c.verify_authority_records(tmp_path,plan,source,approval,grant)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('marker',['CONSUMED.json','STOP.json','run'])
def test_consumed_or_STOP_is_nonrefundable(tmp_path,marker):
    from trotterlib.h4_execution_contract import check_consumption
    (tmp_path/marker).touch()
    with pytest.raises(PermissionError,match='CONSUMED_OR_STOPPED'):check_consumption(tmp_path)


@pytest.mark.parametrize('mutation',['SOURCE','files','runtime_python','runtime_file'])
def test_source_and_runtime_binding_refuse_changes(tmp_path,monkeypatch,mutation):
    import trotterlib.h4_execution_contract as c
    f=tmp_path/'runtime';f.write_bytes(b'artificial runtime hash fixture')
    source={'SOURCE':'TEST_ONLY_SOURCE','files':{'a':'one'},'environment':{
        'python':str(Path(sys.executable).resolve()),'versions':{},'files':{str(f):sha(f)}}}
    monkeypatch.setattr(c,'git',lambda *a:'TEST_ONLY_SOURCE')
    monkeypatch.setattr(c,'source_manifest',lambda:{'a':'one'})
    monkeypatch.setattr(c,'environment_manifest',lambda plan:{'versions':{}})
    c.verify_sealed_source(source,{})
    if mutation=='SOURCE':source['SOURCE']='OTHER'
    if mutation=='files':source['files']['a']='changed'
    if mutation=='runtime_python':source['environment']['python']='other'
    if mutation=='runtime_file':f.write_bytes(b'changed')
    with pytest.raises(PermissionError):c.verify_sealed_source(source,{})


def test_production_spin_lift_retains_tiny_coefficients_without_legacy_cutoff():
    from trotterlib.h4_molecular_inputs import spin_lift_unthresholded
    one=np.eye(4);one[0,1]=one[1,0]=1e-17
    eri=np.zeros((4,)*4);eri[1,2,3,0]=2e-17
    a,b=spin_lift_unthresholded(one,eri)
    assert a[0,2]==a[1,3]==1e-17
    assert b[2,7,1,4]==2e-17


def test_snapshot_marker_binding_is_checked(sealed_artificial):
    folder,b,receipt=sealed_artificial;b['snapshot_sha256']='wrong'
    with pytest.raises(ValueError,match='SNAPSHOT_CHANGED'):load_inputs(folder,b)


@pytest.mark.parametrize('damage',['missing_case','missing_event','duplicate_sample','corrupt_price','corrupt_event'])
def test_saved_eight_qubit_integration_detects_partial_or_damaged_evidence(tmp_path,damage):
    import shutil
    from trotterlib.h4_pilot_execution import synthetic_plan
    fixture=ROOT/'artifacts/non_df_h4_execution_foundation/2026-10-23/technical/eight_qubit_final/run'
    shutil.copytree(fixture,tmp_path/'run')
    run=tmp_path/'run';case=run/'pauli_jw_PR';r=read(case/'result.json');bind=r['provenance']['source']
    sample=r['circuit_cost']['samples'][0];tid=sample['trajectory_id']
    if damage=='missing_case':
        (case/'status.json').unlink()
        out=integrate(run,synthetic_plan(),bind)
        assert out['status']=='PARTIAL_RESULTS_NOT_FOUR_METHOD_COMPARISON'
        assert out['missing_cases']==['pauli_jw_PR'] and len(out['rows'])==15
        return
    if damage=='missing_event':(case/'trajectories'/tid/'events.json').unlink()
    if damage=='duplicate_sample':
        r['circuit_cost']['samples'][1]=sample
        (case/'result.json').write_text(__import__('json').dumps(r))
    if damage in {'corrupt_price','corrupt_event'}:
        file=case/'trajectories'/tid/('price_join.json' if damage=='corrupt_price' else 'events.json')
        row=read(file)
        if damage=='corrupt_price':row['compiled_RZ']['sine']+=1
        else:row['events'].pop()
        file.write_text(__import__('json').dumps(row))
    with pytest.raises((ValueError,FileNotFoundError)):
        integrate(run,synthetic_plan(),bind)
