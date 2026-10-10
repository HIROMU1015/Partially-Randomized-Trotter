"""Synthetic preparation tests only: no CLI launch, real molecule, or real DF call."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import pytest
from trottertracks.resource_applicability import ax2b_h6_df_diagnostic_contract_v1 as c
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_limits import CallBudget
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2b_h6_df_diagnostic_port_v1 import (
    DiagnosticPort,describe,normal_order,antisymmetrize,array_record,write_npz)
from trottertracks.resource_applicability.ax2b_h6_df_diagnostic_watchdog_v1 import DiagnosticProgress,supervise
from trottertracks.resource_applicability.ax2b_h6_df_diagnostic_audit_v1 import audit_saved

@pytest.fixture(autouse=True)
def prohibit_real_science(monkeypatch,tmp_path):
    def forbidden(*a,**k):raise AssertionError('REAL_SCIENCE_FORBIDDEN_IN_SYNTHETIC_TEST')
    monkeypatch.setitem(sys.modules,'openfermion',types.SimpleNamespace(low_rank_two_body_decomposition=forbidden,MolecularData=forbidden))
    monkeypatch.setattr(np.linalg,'eigh',forbidden)
    original=np.load
    def temporary_only(path,*a,**k):
        if isinstance(path,(str,Path)) and not Path(path).resolve().is_relative_to(tmp_path):forbidden()
        return original(path,*a,**k)
    monkeypatch.setattr(np,'load',temporary_only)

def values(rank=2):
    one=np.zeros((12,12),complex);two=np.zeros((12,)*4,complex)
    gs=np.zeros((rank,12,12),complex)
    for i in range(rank):gs[i,i%12,i%12]=1.
    return one,two,np.ones(rank),gs,np.zeros((12,12),complex),np.array(0.,dtype='<f8')

def make_port(tmp_path,rank=2,mutate=None):
    one,two,weights,gs,correction,trunc=values(rank)
    if mutate:mutate(weights,gs)
    m=c.preparation();m['source_commit']='a'*40;m['input_identity']={'input_sha256':'b'*64,
        'integral_array_records':{'one_body':array_record(one),'two_body':array_record(two)}}
    out=tmp_path/'run';out.mkdir();writer=AtomicWriter(out,byte_cap=m['plan']['caps']['output_bytes'])
    seen=[]
    def fake(tensor,**kwargs):
        seen.append(kwargs);assert np.array_equal(tensor,two);tensor.flat[0]=9.
        return weights,gs,correction,trunc
    p=DiagnosticPort(m,writer,DiagnosticProgress(writer,cap=128),root=tmp_path,decomposer=fake,payload=(one,two))
    return p,out,m,seen

def test_fixed_plan_has_only_tol_and_forbidden_work_zero():
    p=c.plan();assert p['kwargs']=={'truncation_threshold':1e-8}
    assert p['hermitization_tolerance']==1e-10
    assert all(p['caps'][k]==0 for k in ('molecular_build','state_solver','signal','trajectory','occurrence','compile'))
    assert p['historical_raw_bytes_identity_claim'] is False

def test_raw_first_all_fragments_even_zero_lambda_failure(tmp_path):
    def mutate(w,g):w[15]=0.;g[15,0,1]=1j
    p,out,m,seen=make_port(tmp_path,16,mutate);summary=p.execute()
    assert seen==[{'truncation_threshold':1e-8}]
    row=summary['fragment_15'];assert row['lambda']==0 and row['hermitization_change_frobenius']>1e-10
    assert row['weighted_projection_antisym_two_body_difference_frobenius']==0
    assert 15 in summary['failed_fragment_indices'] and len(summary['fragments'])==16
    assert summary['H6_input_accepted'] is False and summary['historical_raw_bytes_identity_claim'] is False
    raw=np.load(out/'raw_decomposition.npz');assert raw['g_matrices_raw'][15,0,1]==1j
    assert (out/'raw_decomposition_receipt.json').stat().st_mtime_ns <= (out/'diagnostic_summary.json').stat().st_mtime_ns

def test_tiny_lambda_separate_from_matrix_deviation(tmp_path):
    p,out,m,seen=make_port(tmp_path,2,lambda w,g:(w.__setitem__(1,1e-12),g.__setitem__((1,0,1),2.)))
    s=p.execute();row=s['fragments'][1]
    assert row['abs_lambda']==1e-12 and row['hermitization_change_frobenius']>1
    assert 0<row['weighted_projection_one_body_difference_frobenius']<1e-9
    assert len(s['fragments'])==2

def test_all_bad_fragments_and_no_promotion_on_good_fragments(tmp_path):
    p,out,m,seen=make_port(tmp_path,16,lambda w,g:(g.__setitem__((3,0,1),1.),g.__setitem__((15,0,1),2.)))
    s=p.execute();assert s['failed_fragment_indices']==[3,15]
    (tmp_path/'good').mkdir()
    p,out,m,seen=make_port(tmp_path/'good')
    s=p.execute();assert s['all_original_hermitization_checks_satisfied'] is True and s['H6_input_accepted'] is False

def test_known_normal_order_fixture_has_zero_coefficient_residual():
    one,two,w,g,corr,trunc=values(1);q,t=normal_order(g[0]);corr=-q;two=t.copy()
    s,_=describe(one,two,w,g,corr,trunc,budget=CallBudget(fragment_diagnostics=36))
    assert s['representation_diagnostics']['raw_normal_order_one_body_residual_frobenius']==0
    assert s['representation_diagnostics']['raw_normal_order_antisym_two_body_residual_frobenius']==0

@pytest.mark.parametrize('bad', ['nonfinite','negative_truncation','over_rank','fragment_budget'])
def test_bad_summary_keeps_exact_raw(tmp_path,bad):
    rank=37 if bad=='over_rank' else 2
    p,out,m,seen=make_port(tmp_path,rank)
    old=p.decomposer
    if bad=='fragment_budget':p.calls=CallBudget(decomposition=1,fragment_diagnostics=0)
    elif bad=='nonfinite':
        def call(t,**kw):
            a=list(old(t,**kw));a[1][0,0,0]=np.nan;return tuple(a)
        p.decomposer=call
    elif bad=='negative_truncation':
        def call(t,**kw):
            a=list(old(t,**kw));a[-1]=np.array(-1.);return tuple(a)
        p.decomposer=call
    with pytest.raises((ValueError,RuntimeError)):p.execute()
    assert (out/'raw_decomposition.npz').exists() and (out/'raw_decomposition_receipt.json').exists()
    assert not (out/'diagnostic_summary.json').exists()

def test_decomposer_exception_no_fabricated_raw(tmp_path):
    p,out,m,seen=make_port(tmp_path)
    def fail(*a,**k):raise ValueError('FAKE_DECOMPOSER_FAILURE')
    p.decomposer=fail
    with pytest.raises(ValueError):p.execute()
    assert p.calls.used['decomposition']==1 and not (out/'raw_decomposition.npz').exists()

def test_npz_bounds_dtype_exclusive(tmp_path):
    writer=AtomicWriter(tmp_path,byte_cap=2**20)
    with pytest.raises(ValueError):write_npz(writer,'x.npz',{'a':np.array([object()])},2**20)
    with pytest.raises(RuntimeError):write_npz(writer,'x.npz',{'a':np.zeros(100000)},1024)
    write_npz(writer,'x.npz',{'a':np.ones(2)},2**20)
    before=(tmp_path/'x.npz').read_bytes()
    with pytest.raises(FileExistsError):write_npz(writer,'x.npz',{'a':np.zeros(2)},2**20)
    assert (tmp_path/'x.npz').read_bytes()==before

def test_normal_order_sign_and_antisym_independent_fock_identity():
    # Tiny algebra fixture, no molecule, sector library, or diagonalization.
    n=2;dim=4;ann=[]
    for mode in range(n):
        a=np.zeros((dim,dim),complex)
        for state in range(dim):
            if (state>>mode)&1:a[state^(1<<mode),state]=(-1)**((state&((1<<mode)-1)).bit_count())
        ann.append(a)
    g=np.array([[1.,2j],[3.,-.5]],complex)
    A=sum(g[p,q]*ann[p].conj().T@ann[q] for p in range(n) for q in range(n))
    one,two=normal_order(g);anti=antisymmetrize(two)
    def coefficient_operator(t):
        return sum(one[p,q]*ann[p].conj().T@ann[q] for p in range(n) for q in range(n))+sum(t[p,q,r,s]*ann[p].conj().T@ann[q].conj().T@ann[r]@ann[s] for p in range(n) for q in range(n) for r in range(n) for s in range(n))
    assert np.allclose(A@A,coefficient_operator(two),atol=1e-13)
    assert np.allclose(A@A,coefficient_operator(anti),atol=1e-13)

def gate_fixture(tmp_path,monkeypatch):
    m=c.preparation();m.update(execution_plan_sealed=True,source_commit='a'*40,source_hashes={'x':'y'},input_identity={'fake':1},environment={'fake':2})
    cpu=min(os.sched_getaffinity(0));m['assigned_resources']={'assigned_cpu':cpu,'science_workers':1,'blas_threads':1}
    name=c.NAMESPACE+'launch_v1';out=tmp_path/name;m['exclusive_output']={'repository_path':name,'absolute_path':str(out)}
    grant={'schema':'track_a_h6_df_diagnostic_authorization_v1','kind':m['kind'],'approved_by_user':True,
           'manifest_digest':digest(m),'assigned_cpu':cpu,'exclusive_output':str(out),'retry':False,'resume':False}
    monkeypatch.setattr(c,'verify_sources',lambda *a:None);monkeypatch.setattr(c,'verify_input',lambda *a:{'fake':1});monkeypatch.setattr(c,'environment',lambda:{'fake':2})
    return m,grant,out

def test_sealed_preparation_is_not_authorization(tmp_path,monkeypatch):
    m,g,out=gate_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='NEW_DIAGNOSTIC_GRANT_REQUIRED'):c.validate_launch(tmp_path,m,None,out,requested=True)
    assert c.validate_launch(tmp_path,m,g,out,requested=True)==g['assigned_cpu']

@pytest.mark.parametrize('field,value',[('schema','track_a_ax2b_h6_input_generation_authorization_v1'),('approved_by_user',False),('retry',True),('resume',True),('assigned_cpu',True),('manifest_digest','bad'),('kind','H6_INPUT_GENERATION')])
def test_reject_old_grants_and_wrong_binding(tmp_path,monkeypatch,field,value):
    m,g,out=gate_fixture(tmp_path,monkeypatch);g[field]=value
    with pytest.raises(ValueError):c.validate_launch(tmp_path,m,g,out,requested=True)

@pytest.mark.parametrize('field,value',[('execution_plan_sealed',False),('science_authorized',True),('launch_allowed',True),('next_stage_authorized',True),('H6_status','GO'),('assigned_resources',{'assigned_cpu':True,'science_workers':True,'blas_threads':True}),('environment',{'changed':1}),('input_identity',{'changed':1})])
def test_manifest_mutation_rejected(tmp_path,monkeypatch,field,value):
    m,g,out=gate_fixture(tmp_path,monkeypatch);m[field]=value;g['manifest_digest']=digest(m)
    with pytest.raises(ValueError):c.validate_launch(tmp_path,m,g,out,requested=True)

def test_plan_change_and_existing_output_rejected(tmp_path,monkeypatch):
    m,g,out=gate_fixture(tmp_path,monkeypatch);m['plan']['kwargs']['final_rank']=12;g['manifest_digest']=digest(m)
    with pytest.raises(ValueError,match='SEAL_OR_PLAN'):c.validate_launch(tmp_path,m,g,out,requested=True)
    m,g,out=gate_fixture(tmp_path,monkeypatch);out.mkdir(parents=True)
    with pytest.raises(ValueError,match='EXCLUSIVE_OUTPUT'):c.validate_launch(tmp_path,m,g,out,requested=True)

def test_worker_claim_one_shot(tmp_path,monkeypatch):
    m,g,out=gate_fixture(tmp_path,monkeypatch);out.mkdir(parents=True)
    (out/'launch_binding.json').write_text(json.dumps({'manifest_digest':digest(m),'authorization_digest':digest(g)}))
    assert c.validate_launch(tmp_path,m,g,out,requested=True,worker=True)==g['assigned_cpu']
    (out/'worker_claim.json').write_text('{}')
    with pytest.raises(ValueError,match='ONE_SHOT'):c.validate_launch(tmp_path,m,g,out,requested=True,worker=True)

def audit_fixture(tmp_path):
    p,out,m,seen=make_port(tmp_path);summary=p.execute()
    grant={'schema':'track_a_h6_df_diagnostic_authorization_v1','approved_by_user':True,'manifest_digest':digest(m),'assigned_cpu':0,'retry':False,'resume':False}
    text=json.dumps(grant,indent=3)+'\n';sha=hashlib.sha256(text.encode()).hexdigest()
    records={'frozen_diagnostic.json':m,'authorization.json':grant,
        'launch_binding.json':{'manifest_digest':digest(m),'authorization_digest':digest(grant)},
        'worker_claim.json':{'manifest_digest':digest(m),'authorization_digest':digest(grant),'assigned_cpu':0,'retry':False,'resume':False,'authorization_sha256':sha}}
    worker={'status':'H6_DF_DIAGNOSTIC_RECORDED','authorization_sha256':sha,'H6_status':'H6_NOT_AUTHORIZED','mandatory_stop':True,'next_stage_authorized':False}
    records['worker_terminal.json']=worker
    records['terminal_status.json']={'status':worker['status'],'worker_exit_code':0,'worker_terminal':worker,'H6_status':'H6_NOT_AUTHORIZED','mandatory_stop':True,'next_stage_authorized':False}
    for name,row in records.items():(out/name).write_text(json.dumps(row))
    (out/'authorization_source.json').write_text(text)
    return out

def test_saved_audit_checks_raw_rows_and_original_grant_bytes(tmp_path):
    out=audit_fixture(tmp_path);r=audit_saved(out)
    assert r['status']=='SAVED_DIAGNOSTIC_BYTES_PASS' and not r['missing_records']

@pytest.mark.parametrize('tamper',['raw_bytes','row_hash','lambda','promotion','missing','grant_whitespace'])
def test_saved_audit_tampering(tmp_path,tamper):
    out=audit_fixture(tmp_path)
    if tamper=='raw_bytes':
        p=out/'raw_decomposition.npz';p.write_bytes(p.read_bytes()+b'x')
    elif tamper=='missing':(out/'diagnostic_summary.json').unlink()
    elif tamper=='grant_whitespace':
        p=out/'authorization_source.json';p.write_text(p.read_text()+' ')
    else:
        p=out/'diagnostic_summary.json';s=json.loads(p.read_text())
        if tamper=='row_hash':s['fragments'][0]['raw']['sha256']='bad'
        elif tamper=='lambda':s['fragments'][0]['lambda']=8
        else:s['H6_input_accepted']=True
        p.write_text(json.dumps(s))
    with pytest.raises(ValueError):audit_saved(out)

def test_saved_stop_never_promoted(tmp_path):
    out=audit_fixture(tmp_path)
    w=json.loads((out/'worker_terminal.json').read_text());w['status']='H6_DF_DIAGNOSTIC_STOP'
    p=json.loads((out/'terminal_status.json').read_text());p.update(status=w['status'],worker_terminal=w,worker_exit_code=1)
    (out/'worker_terminal.json').write_text(json.dumps(w));(out/'terminal_status.json').write_text(json.dumps(p))
    (out/'diagnostic_summary.json').unlink();r=audit_saved(out)
    assert r['status']=='SAVED_DIAGNOSTIC_STOP_RECORDED' and 'diagnostic_summary.json' in r['missing_records']
    assert 'partial_raw_decomposition' in r['checks']

def test_saved_partial_raw_tamper_detected_after_stop(tmp_path):
    out=audit_fixture(tmp_path);w=json.loads((out/'worker_terminal.json').read_text());w['status']='H6_DF_DIAGNOSTIC_STOP'
    p=json.loads((out/'terminal_status.json').read_text());p.update(status=w['status'],worker_terminal=w,worker_exit_code=1)
    (out/'worker_terminal.json').write_text(json.dumps(w));(out/'terminal_status.json').write_text(json.dumps(p))
    raw=out/'raw_decomposition.npz';raw.write_bytes(raw.read_bytes()+b'x')
    with pytest.raises(ValueError,match='SAVED_NPZ_BYTES'):audit_saved(out)

@pytest.mark.parametrize('mode',['wall','log','exit'])
def test_watchdog_dummy_process_caps(tmp_path,mode):
    caps=c.plan()['caps'].copy();caps['total_wall_seconds']=.15;caps['phase_wall_seconds']=dict.fromkeys(c.PHASES,.15)
    caps['log_bytes']=128
    code='import time;time.sleep(1)' if mode=='wall' else ('print("x"*1000)' if mode=='log' else 'raise SystemExit(1)')
    r=supervise([sys.executable,'-c',code],tmp_path,caps=caps)
    assert r['status']=='H6_DF_DIAGNOSTIC_STOP' and r['next_stage_authorized'] is False
    assert r['worker_log_bytes']<=128

def test_cli_static_gate_only_no_runner_started():
    p=Path(__file__).resolve().parents[3]/'scripts/resource_applicability/run_track_a_h6_df_diagnostic_v1.py'
    tree=ast.parse(p.read_text())
    assert not any(isinstance(n,ast.Import) and any(a.name in ('numpy','openfermion') for a in n.names) for n in tree.body)
    imports=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module and n.module.endswith('ax2b_h6_df_diagnostic_port_v1')]
    calls=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validate_launch']
    assert len(imports)==1 and min(calls)<imports[0]
