"""Synthetic inputs/fake chemistry+solver/dummy children only; no H6 evidence."""
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trottertracks.resource_applicability import ax2b_h6_input_generation_contract_v1 as contract
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h6_input_generation_port_v1 import InputGenerationPort, atomic_npz
from trottertracks.resource_applicability.ax2b_h6_input_generation_watchdog_v1 import InputProgress, supervise
from trottertracks.resource_applicability.ax2b_h6_input_generation_audit_v1 import audit_saved, npz_bytes
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def forbid_real_science(monkeypatch, tmp_path):
    def forbidden(*a, **k):raise AssertionError('MOLECULE_SOLVER_SAMPLING_CIRCUIT_FORBIDDEN')
    import openfermion
    import openfermionpyscf
    import pyscf.gto
    import scipy.sparse.linalg
    import trotterlib.df_hamiltonian as df
    import trotterlib.rte as rte
    import qiskit
    monkeypatch.setattr(openfermion.MolecularData,'__init__',forbidden)
    monkeypatch.setattr(openfermion,'low_rank_two_body_decomposition',forbidden)
    monkeypatch.setattr(openfermionpyscf,'run_pyscf',forbidden)
    monkeypatch.setattr(pyscf.gto.Mole,'build',forbidden)
    monkeypatch.setattr(scipy.sparse.linalg,'eigsh',forbidden)
    monkeypatch.setattr(df,'build_df_h_d_from_molecule',forbidden)
    monkeypatch.setattr(qiskit.QuantumCircuit,'__init__',forbidden)
    monkeypatch.setattr(qiskit,'transpile',forbidden)
    for name in ('sample_rte_events','iter_sample_rte_events','sample_event_mean_operator'):
        monkeypatch.setattr(rte,name,forbidden)
    original_load=np.load
    def only_tmp_load(path,*a,**k):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        return original_load(path,*a,**k)
    monkeypatch.setattr(np,'load',only_tmp_load)
    original, original_path=builtins.open,Path.open
    def check(path):
        if (isinstance(path,(str,Path)) and ('/artifacts/' in str(path) or '/.runtime/' in str(path))
                and not Path(path).resolve().is_relative_to(tmp_path.resolve())):
            raise AssertionError('REAL_SCIENTIFIC_ARTIFACT_IO_FORBIDDEN')
    def safe_open(path,*a,**k):check(path);return original(path,*a,**k)
    def safe_path(path,*a,**k):check(path);return original_path(path,*a,**k)
    monkeypatch.setattr(builtins,'open',safe_open);monkeypatch.setattr(Path,'open',safe_path)


def sealed(tmp_path, monkeypatch):
    cpu=min(os.sched_getaffinity(0));m=contract.preparation()
    m.update(execution_plan_sealed=True,assigned_resources={'assigned_cpu':cpu,'science_workers':1,'blas_threads':1},
             source_commit='a'*40,source_hashes={'synthetic':'b'*64},environment={'fixture':True})
    relative=contract.OUTPUT_NAMESPACE+'synthetic_one_shot'
    output=tmp_path/relative
    m['intended_exclusive_output']={'repository_path':relative,'absolute_path':str(output.resolve())}
    g={'schema':'track_a_ax2b_h6_input_generation_authorization_v1','approved_by_user':True,
       'kind':'H6_INPUT_GENERATION','manifest_digest':digest(m),'assigned_cpu':cpu,
       'exclusive_output':str(output.resolve()),'retry':False,'resume':False}
    monkeypatch.setattr(contract,'verify_sources',lambda *a:None)
    monkeypatch.setattr(contract,'environment',lambda:{'fixture':True})
    return m,g,output


def synthetic_payload(*args):
    return {'constant':.1,'one_body':np.eye(12)*.2,'two_body':np.zeros((12,)*4),
            'spatial_one_body':np.eye(6),'spatial_two_body':np.zeros((6,)*4),
            'canonical_orbitals':np.eye(6),'hf_energy':1.,'scf_converged':True,
            'scf_conv_tol':1e-9,'scf_max_cycle':50,'scf_cycles':1}


def fake_decomposer(two,**kwargs):
    assert kwargs=={'truncation_threshold':1e-8}
    return np.array([.2,.3]),np.stack([np.eye(12),2*np.eye(12)]),np.zeros((12,12)),1e-9


def fake_solver(operator,**kwargs):
    assert {k:v for k,v in kwargs.items() if k!='v0'}=={'k':1,'which':'SA','tol':1e-12,'maxiter':1000,'ncv':40}
    v=kwargs['v0'];assert np.count_nonzero(v)==1
    assert np.array_equal(operator @ v,2*v)
    assert np.array_equal(operator.rmatvec(v),2*v)
    return np.array([2.]),(3j*v).reshape(-1,1)


def synthetic_run(tmp_path, monkeypatch):
    m,g,_=sealed(tmp_path,monkeypatch);out=tmp_path/'synthetic';out.mkdir()
    writer=AtomicWriter(out,byte_cap=128*2**20,diagnostics_cap=512)
    progress=InputProgress(writer,cap=256)
    writer.write('frozen_preparation.json',m);writer.write('authorization.json',g)
    raw=json.dumps(g,indent=1).encode()+b'\n';(out/'authorization_source.json').write_bytes(raw)
    writer.write('launch_binding.json',{'manifest_digest':digest(m),'authorization_digest':digest(g)})
    writer.write('worker_claim.json',{'manifest_digest':digest(m),'authorization_digest':digest(g),
                                     'assigned_cpu':g['assigned_cpu'],'retry':False,'resume':False})
    port=InputGenerationPort(m,writer,progress,grant_sha256=hashlib.sha256(raw).hexdigest(),
         integrals=synthetic_payload,decomposer=fake_decomposer,solver=fake_solver,
         operator_factory=lambda h,s:np.eye(400)*2)
    port.execute()
    terminal={'status':'H6_INPUT_GENERATION_COMPLETE','snapshot_receipt_saved':True,'reason':None,
              'calls_attempted':port.calls.used,'calls_completed':port.completed,
              'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
              'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
              'mandatory_stop':True,'next_stage_authorized':False}
    writer.write('worker_terminal.json',terminal,terminal=True)
    writer.write('terminal_status.json',{**terminal,'worker_terminal':terminal,'worker_exit_code':0},terminal=True)
    return port,out


def test_metadata_cli_without_site_packages(tmp_path):
    p=tmp_path/'metadata.json'
    r=subprocess.run([sys.executable,'-S',str(ROOT/contract.RUNNER),'--output',str(p)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert json.loads(p.read_text())==contract.preparation()
    assert not contract.preparation()['execution_plan_sealed']


def test_provider_wiring_without_molecule_or_implicit_save(tmp_path,monkeypatch):
    from trottertracks.resource_applicability.ax2b_h6_input_generation_port_v1 import obtain_integrals
    import openfermion
    import openfermion.chem.molecular_data as chemistry
    import openfermionpyscf._run_pyscf as provider
    import pyscf.lib
    seen=[]
    def molecular(geometry,basis,multiplicity,charge,**kwargs):
        seen.append((geometry,basis,multiplicity,charge,kwargs));return SimpleNamespace()
    mol=SimpleNamespace(unit='angstrom',symmetry=False,nao_nr=lambda:6,nelectron=6,energy_nuc=lambda:1.5)
    mf=SimpleNamespace(converged=True,e_tot=-1.,mo_coeff=np.eye(6),cycles=3)
    mf.kernel=lambda:seen.append(('kernel',mf.conv_tol,mf.max_cycle,mf.chkfile))
    monkeypatch.setattr(openfermion,'MolecularData',molecular)
    monkeypatch.setattr(provider,'prepare_pyscf_molecule',lambda m:mol)
    monkeypatch.setattr(provider,'compute_scf',lambda m:mf)
    monkeypatch.setattr(provider,'compute_integrals',lambda m,s:(np.eye(6),np.ones((6,)*4)))
    monkeypatch.setattr(chemistry,'spinorb_from_spatial',lambda a,b:(np.eye(12),np.ones((12,)*4)*3))
    monkeypatch.setattr(pyscf.lib,'num_threads',lambda n:seen.append(('threads',n)))
    result=obtain_integrals(tmp_path,contract.plan()['integrals'])
    assert np.array_equal(result['two_body'],np.ones((12,)*4)*1.5)
    assert result['constant']==1.5 and result['scf_cycles']==3
    assert seen[0]==('threads',1) and seen[-1][0:3]==('kernel',1e-9,50)
    assert seen[1][1:4]==('sto-3g',1,0)
    assert list(tmp_path.iterdir())==[]


def test_cli_no_grant_refuses_before_numeric_imports(tmp_path):
    p=tmp_path/'never'
    r=subprocess.run([sys.executable,'-S',str(ROOT/contract.RUNNER),'--output',str(p),'--execute'],capture_output=True,text=True)
    assert r.returncode!=0 and 'Separate pinned' in r.stderr
    assert 'numpy' not in r.stderr and not p.exists()


def test_valid_seal_metadata_gate(tmp_path,monkeypatch):
    m,g,out=sealed(tmp_path,monkeypatch)
    assert contract.validate_launch(tmp_path,m,g,requested=True,output=out)==g['assigned_cpu']


@pytest.mark.parametrize('field,value',[('execution_plan_sealed',False),('science_authorized',True),
 ('H6_status','AUTHORIZED'),('next_stage_authorized',True),('kind','H6_TECHNICAL'),('status','COMPLETE')])
def test_manifest_rejection(tmp_path,monkeypatch,field,value):
    m,g,out=sealed(tmp_path,monkeypatch);m[field]=value;g['manifest_digest']=digest(m)
    with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,requested=True,output=out)


@pytest.mark.parametrize('field,value',[('approved_by_user',False),('schema','track_a_ax2b_supplement_authorization_v1'),
 ('retry',True),('resume',True),('assigned_cpu',True),('manifest_digest','c'*64)])
def test_grant_rejection(tmp_path,monkeypatch,field,value):
    m,g,out=sealed(tmp_path,monkeypatch);g[field]=value
    with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,requested=True,output=out)


@pytest.mark.parametrize('mutation',['df_tol','budget','geometry','boolean_worker'])
def test_fixed_conditions_and_json_type_identity(tmp_path,monkeypatch,mutation):
    m,g,out=sealed(tmp_path,monkeypatch)
    if mutation=='df_tol':m['plan']['df_policy']['df_tol']=1e-6
    if mutation=='budget':m['plan']['caps_proposed']['total_wall_seconds']+=1
    if mutation=='geometry':m['plan']['target']['geometry_angstrom']=2.
    if mutation=='boolean_worker':m['assigned_resources']['science_workers']=True
    g['manifest_digest']=digest(m)
    with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,requested=True,output=out)


def test_output_and_consumed_worker_rejected(tmp_path,monkeypatch):
    m,g,out=sealed(tmp_path,monkeypatch);out.mkdir(parents=True)
    with pytest.raises(ValueError,match='NEW_EXCLUSIVE'):contract.validate_launch(tmp_path,m,g,requested=True,output=out)
    (out/'launch_binding.json').write_text(json.dumps({'manifest_digest':digest(m),'authorization_digest':digest(g)}))
    (out/'worker_claim.json').write_text('{}')
    with pytest.raises(ValueError,match='ALREADY_CLAIMED'):contract.validate_launch(tmp_path,m,g,requested=True,output=out,worker=True)


def test_source_and_environment_gates_before_work(tmp_path,monkeypatch):
    m,g,out=sealed(tmp_path,monkeypatch)
    def reject(*args):raise ValueError('SYNTHETIC_SOURCE_CHANGE')
    monkeypatch.setattr(contract,'verify_sources',reject)
    with pytest.raises(ValueError,match='SOURCE_CHANGE'):contract.validate_launch(tmp_path,m,g,requested=True,output=out)
    monkeypatch.setattr(contract,'verify_sources',lambda *a:None)
    monkeypatch.setattr(contract,'environment',lambda:{'changed':True})
    with pytest.raises(ValueError,match='ENVIRONMENT_CHANGED'):contract.validate_launch(tmp_path,m,g,requested=True,output=out)


def test_synthetic_snapshot_loader_and_stdlib_saved_audit(tmp_path,monkeypatch):
    port,out=synthetic_run(tmp_path,monkeypatch)
    assert port.calls.used['solver_matvec']==3 and port.completed['solver_matvec']==3
    receipt=json.loads((out/'snapshot_receipt.json').read_text())
    assert receipt['loader_roundtrip_checked']
    assert receipt['metadata']['state_policy']['global_phase_policy']=='largest_sector_amplitude_real_positive_v1'
    with np.load(out/'h6_input_snapshot.npz',allow_pickle=False) as data:
        v=data['sector_state_vector'];pivot=np.argmax(abs(v))
        assert v[pivot].real>0 and abs(v[pivot].imag)<1e-15
    audit=audit_saved(out)
    assert audit['status']=='SAVED_INPUT_CONSISTENCY_PASS' and audit['checks']['actual_rank']==2
    assert audit['missing_records']==[] and not audit['numerical_allowance_certified']
    r=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/resource_applicability/audit_track_a_ax2b_h6_input_generation_v1.py'),
                      '--saved-run',str(out),'--output',str(tmp_path/'audit.json')],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_saved_byte_tamper_and_missing_not_pass(tmp_path,monkeypatch):
    _,out=synthetic_run(tmp_path,monkeypatch)
    p=out/'h6_input_snapshot.npz';original=p.read_bytes();p.write_bytes(original+b'x')
    with pytest.raises(ValueError,match='NPZ_BYTES'):audit_saved(out)
    p.write_bytes(original);p.unlink()
    with pytest.raises(ValueError,match='MISSING_RECORDS'):audit_saved(out)


def test_saved_parent_stop_keeps_completed_bytes_without_promotion(tmp_path,monkeypatch):
    _,out=synthetic_run(tmp_path,monkeypatch)
    p=out/'terminal_status.json';r=json.loads(p.read_text())
    r.update(status='H6_INPUT_GENERATION_STOP',reason='FAKE_PARENT_CAP')
    p.write_text(json.dumps(r))
    result=audit_saved(out)
    assert result['status']=='SAVED_INPUT_STOP_RECORDED'
    assert result['science_run_status']=='H6_INPUT_GENERATION_STOP'
    assert 'h6_input_snapshot.npz' in result['file_hashes']


def test_residual_uses_same_before_call_budget(tmp_path,monkeypatch):
    m,_,_=sealed(tmp_path,monkeypatch);m['plan']['caps_proposed']['solver_matvec']=2
    out=tmp_path/'cap';out.mkdir();writer=AtomicWriter(out,byte_cap=128*2**20)
    p=InputGenerationPort(m,writer,InputProgress(writer,cap=256),grant_sha256='b'*64,
         integrals=synthetic_payload,decomposer=fake_decomposer,solver=fake_solver,
         operator_factory=lambda h,s:np.eye(400)*2)
    with pytest.raises(RuntimeError,match='CALL_BUDGET:solver_matvec'):p.execute()
    assert p.calls.used['solver_matvec']==2 and p.completed['solver_matvec']==2
    assert not (out/'h6_input_snapshot.npz').exists()


@pytest.mark.parametrize('failure',['scf','rank','cross_spin','residual','no_convergence'])
def test_failure_does_not_publish_snapshot(tmp_path,monkeypatch,failure):
    m,_,_=sealed(tmp_path,monkeypatch);out=tmp_path/'fail';out.mkdir();writer=AtomicWriter(out,byte_cap=128*2**20)
    def payload(*args):
        v=synthetic_payload();v['scf_converged']=failure!='scf';return v
    def decomp(*args,**kwargs):
        w,b,c,t=fake_decomposer(*args,**kwargs)
        if failure=='rank':w,b=w[:1],b[:1]
        if failure=='cross_spin':b[0,0,1]=b[0,1,0]=.1
        return w,b,c,t
    def solver(op,**kwargs):
        if failure=='no_convergence':raise RuntimeError('FAKE_ARPACK_NO_CONVERGENCE')
        e,v=fake_solver(op,**kwargs)
        return (np.array([3.]),v) if failure=='residual' else (e,v)
    p=InputGenerationPort(m,writer,InputProgress(writer,cap=256),grant_sha256='b'*64,
                          integrals=payload,decomposer=decomp,solver=solver,
                          operator_factory=lambda h,s:np.eye(400)*2)
    with pytest.raises((ValueError,RuntimeError)):p.execute()
    assert not (out/'h6_input_snapshot.npz').exists()


def test_npz_caps_exclusive_and_progress_cap(tmp_path):
    writer=AtomicWriter(tmp_path,byte_cap=2**20)
    with pytest.raises(RuntimeError,match='EXPANDED_CAP'):atomic_npz(writer,'toy.npz',{'a':np.ones(10)},expanded_cap=2)
    atomic_npz(writer,'toy.npz',{'a':np.ones(10)},expanded_cap=2**20)
    with pytest.raises(FileExistsError):atomic_npz(writer,'toy.npz',{'a':np.ones(10)},expanded_cap=2**20)
    p=InputProgress(writer,cap=1);p.update(point='fixture')
    with pytest.raises(RuntimeError,match='PROGRESS_CAP'):p.update(point='fixture2')


@pytest.mark.parametrize('mode',['sleep','complete','invalid_phase','log_cap','failed_worker'])
def test_dummy_watchdog_preserves_stop_and_completion(tmp_path,mode):
    out=tmp_path/'watchdog';out.mkdir();caps=deepcopy(contract.plan()['caps_proposed'])
    caps.update(total_wall_seconds=2.,phase_wall_seconds=dict.fromkeys(contract.PHASES,1.))
    if mode=='sleep':caps['phase_wall_seconds']['integrals']=.1
    script="import json,pathlib,time,sys\np=pathlib.Path(sys.argv[1])\n"
    if mode=='sleep':script+="(p/'progress_0000.json').write_text(json.dumps({'point':'dummy_saved'}));time.sleep(5)\n"
    elif mode=='invalid_phase':script+="(p/'phase_state_snapshot.json').write_text(json.dumps({'phase':'state_snapshot','elapsed':.01}));time.sleep(5)\n"
    elif mode=='log_cap':script+="print('x'*100000);time.sleep(5)\n"
    elif mode=='failed_worker':script+="raise RuntimeError('dummy failure')\n"
    else:
        for phase in contract.PHASES:script+=f"(p/'phase_{phase}.json').write_text(json.dumps({{'phase':{phase!r},'elapsed':.01}}))\n"
        terminal={'status':'H6_INPUT_GENERATION_COMPLETE','snapshot_receipt_saved':True,'N':None,'G':None,
                  'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
                  'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
                  'mandatory_stop':True,'next_stage_authorized':False}
        script+=f"(p/'worker_terminal.json').write_text(json.dumps({terminal!r}))\n"
    r=supervise([sys.executable,'-S','-c',script,str(out)],out,caps=caps)
    assert r['status']==('H6_INPUT_GENERATION_COMPLETE' if mode=='complete' else 'H6_INPUT_GENERATION_STOP')
    assert r['mandatory_stop'] and not r['next_stage_authorized']
    if mode=='sleep':assert r['latest_progress']['point']=='dummy_saved' and 'PHASE_WALL_CAP' in r['reason']


def test_exact_grant_bytes_retained_not_canonicalized(tmp_path):
    spec=importlib.util.spec_from_file_location('fixture_runner',ROOT/contract.RUNNER)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    source=tmp_path/'original.json';raw=b'{"z": 1, "a": 2}\n';source.write_bytes(raw)
    out=tmp_path/'output';out.mkdir();writer=AtomicWriter(out,byte_cap=2**20)
    runner.publish_exact_authorization(writer,source,hashlib.sha256(raw).hexdigest())
    assert (out/'authorization_source.json').read_bytes()==raw
    with pytest.raises(ValueError,match='AUTHORIZATION_CHANGED'):runner.publish_exact_authorization(writer,source,'b'*64)
