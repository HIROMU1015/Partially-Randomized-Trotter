"""Artificial endpoint pipeline / call, identity, FAIL and passive boundaries."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_near_practical as m
from trottertracks.c_pr_integrated_evaluation import arrays


def fixture(root):
    from test_research_c_separated_J_U import invented
    g,J,U,_=invented(n=4,rank=8,ld=2)
    J,U,x,_,_=m.algorithm.near.pr.corrected.codec.roundtrip(J,U)
    arrays(root/'input.npz',g=g)
    arrays(root/'initial.npz',J=J,U=U,x0=x)
    (root/'scales.json').write_text(json.dumps(dict(initial_tensor_objective=m.OBJECTIVE['loss_scale'],initial_smooth_tail_mass=m.OBJECTIVE['mass_scale'])))
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# invented runner\n')
    (root/'SOURCE.json').write_text(json.dumps(dict(files={m.ENTRY:m.io.sha(entry)})))
    c={**m.fixed(), 'source_manifest':'SOURCE.json','source_manifest_sha256':m.io.sha(root/'SOURCE.json'),
       'input_paths':dict(input='input.npz',initial='initial.npz',scales='scales.json'),
       'saved_inputs':{str(p.relative_to(root)):m.io.sha(p) for p in root.iterdir() if p.suffix in ('.npz','.json') and p.name!='SOURCE.json'},
       'fit_packages':{},'evaluation_packages':{},'ffsim_sources':{},'evaluation_python':sys.executable}
    (root/'contract.json').write_text(json.dumps(c));return c


class Service:
    mode='pass'
    def __init__(self,*a):pass
    def request(self,operation,**v):
        if operation=='assess':
            if self.mode=='correctness':raise ValueError('ARTIFICIAL_CORRECTNESS')
            return dict(budget_PASS=self.mode!='DFfail',sector_error=1e-10,numerical_guard=1e-12,exact_tail_lambda_Ha=1.)
        return dict(status='READY')
    def close(self):pass


def arm(name,g,cold,spec,physical,call,record):
    counts={}
    def local(kind,label,fn):
        counts[kind]=counts.get(kind,0)+1
        return call(kind,name+'_'+label,fn)
    local.counts=counts
    local('optimization_driver','mock finite',lambda:None)
    # No synthetic optimizer iterations here; they are covered in near_stable.
    row=local('physical_assessment','initial',lambda:physical(name+'_initial',cold['J'],cold['U']))
    return dict(arm=name,objective_version=name,spec=asdict(spec),status='MAXIMUM_ACCEPTED_UPDATES',
        endpoint_eligible=True,physical=row,solver_metrics={},raw_optimizer_status_counts={},actual_CA_achieved=None),dict(J=cold['J'],U=cold['U']),local


@pytest.mark.parametrize('mode',['pass','DFfail','correctness'])
def test_artificial_end_to_end_fresh_partial_STOP_and_no_downstream(tmp_path,monkeypatch,mode):
    fixture(tmp_path);monkeypatch.setattr(Service,'mode',mode)
    monkeypatch.setattr(m,'specs',lambda *a:{'PR':m.algorithm.near.pr.Objective(**m.OBJECTIVE),
        'ENTRY':m.algorithm.near.Objective(8,4,2,m.OBJECTIVE['loss_scale'],1.,.1)})
    if mode=='correctness':
        with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS'):
            m.run(tmp_path,tmp_path/'contract.json',scientific_permission=True,service_factory=Service,arm_runner=arm)
    else:
        r=m.run(tmp_path,tmp_path/'contract.json',scientific_permission=True,service_factory=Service,arm_runner=arm)
        if mode=='DFfail':assert r['status']=='COMMON_INITIAL_DF_BUDGET_NOT_MET' and not r['arms']
        else:
            assert set(r['arms'])=={'PR','ENTRY'} and r['own_objective_winner'] is None
            assert r['PR_resource_winner'] is None and r['actual_CA_achieved'] is None
            assert len({a['candidate_identity']['arm'] for a in r['arms'].values()})==2
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed']==(mode=='correctness') and stop['scientific_units']<=m.MAX_CALLS
    with pytest.raises(ValueError,match='FRESH_NO_RETRY_RESUME'):m.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('drift',['source','candidate','objective','limit'])
def test_passive_wrong_identity_or_source_stops_before_decode(tmp_path,monkeypatch,drift):
    fixture(tmp_path);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decoded'))
    m.preflight(tmp_path,tmp_path/'contract.json')
    if drift=='source':(tmp_path/m.ENTRY).write_text('wrong source')
    if drift=='candidate':(tmp_path/'initial.npz').write_bytes(b'wrong candidate')
    if drift in ('objective','limit'):
        c=m.io.read(tmp_path/'contract.json')
        if drift=='objective':c['arms']=['RC2','ENTRY']
        else:c['maximum_logical_units']+=1
        (tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')


def test_permission_refusal_before_any_preflight_or_output(tmp_path):
    with pytest.raises(PermissionError,match='SCIENTIFIC_PERMISSION'):m.run(tmp_path,tmp_path/'absent.json')
    assert not list(tmp_path.iterdir())
    root=Path(__file__).resolve().parents[2]
    r=subprocess.run([sys.executable,'-B',str(root/m.ENTRY),'--contract','absent.json','--execute'],capture_output=True,text=True)
    assert r.returncode and 'SCIENTIFIC_PERMISSION' in r.stderr


def test_common_x0_is_exact_and_wrong_array_rejected(tmp_path):
    c=fixture(tmp_path);packet,cold,scales=m.load(tmp_path,c)
    assert m.initial_identity(packet,cold,scales,c)['saved_x0_identity_verified_for_both_arms']
    cold['x0'][0]+=.1
    with pytest.raises(ValueError,match='X0_MISMATCH'):m.initial_identity(packet,cold,scales,c)


def test_call_bound_and_fair_opportunity_not_doubled_old_count():
    assert m.MAX_CALLS==1629 and sum(m.LIMITS.values())==1629
    assert m.MAX_POINTS==129 and m.ARM_LIMITS['inner_J_solve']==387
    assert m.ARM_LIMITS['independent_terminal_assessment']==17
    assert m.ARM_LIMITS['physical_assessment']==17
    assert m.STEP.max_updates==16 and m.MAX_TRIALS==128


def test_passive_import_does_no_numerical_work():
    root=Path(__file__).resolve().parents[2]
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_near_practical;assert not any(n in sys.modules for n in ('numpy','scipy','jax','ffsim','qiskit'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=root,check=True)


def test_arm_limit_stops_before_extra_solve_and_identity_is_explicit():
    from test_research_c_separated_J_U import invented
    g,J,U,s=invented();cold=dict(J=J,U=U)
    executed=[]
    def call(k,l,f):executed.append(k);return f()
    def too_many(g,J,U,s,physical,**kw):
        for i in range(m.ARM_LIMITS['inner_J_solve']+1):kw['call']('inner_J_solve',str(i),lambda:None)
    with pytest.raises(ValueError,match='ARM_CALL_LIMIT'):
        m.run_arm('PR',g,cold,s,None,call,lambda *a:None,trajectory=too_many)
    assert executed.count('inner_J_solve')==387
    with pytest.raises(ValueError,match='OBJECTIVE_ARM_IDENTITY'):
        m.run_arm('ENTRY',g,cold,s,None,call,lambda *a:None)


def test_actual_new_runner_CAR_worker_and_locked_service_on_artificial_input(tmp_path):
    from test_research_c_pr_integrated_pipeline import synthetic_packet
    packet,J,U=synthetic_packet();arrays(tmp_path/'input2.npz',**packet)
    # This worker never reads fitter initialization; byte-only placeholders
    # avoid importing the JAX/ffsim-only fixture into the CAR environment.
    (tmp_path/'initial.npz').write_bytes(b'artificial unused fitter payload')
    (tmp_path/'scales.json').write_text('{}')
    paths=dict(input='input2.npz',initial='initial.npz',scales='scales.json')
    c={**m.fixed(),'input_paths':paths,'source_manifest':'SOURCE.json',
       'saved_inputs':{p:m.io.sha(tmp_path/p) for p in paths.values()},
       'fit_packages':{},'evaluation_packages':{},'ffsim_sources':{},'evaluation_python':sys.executable}
    (tmp_path/m.ENTRY).parent.mkdir(parents=True)
    # Run the REAL new entry in the evaluation interpreter, artificial files only.
    root=Path(__file__).resolve().parents[2]
    (tmp_path/m.ENTRY).write_bytes((root/m.ENTRY).read_bytes())
    (tmp_path/'src').symlink_to(root/'src',target_is_directory=True)
    (tmp_path/'SOURCE.json').write_text(json.dumps(dict(files={m.ENTRY:m.io.sha(tmp_path/m.ENTRY)})))
    c['source_manifest_sha256']=m.io.sha(tmp_path/'SOURCE.json')
    (tmp_path/'contract.json').write_text(json.dumps(c))
    out=tmp_path/m.FRESH;out.mkdir(parents=True)
    arrays(tmp_path/'candidate.npz',J=J,U=U)
    service=m.Service(tmp_path,tmp_path/'contract.json',c,out)
    try:
        assert service.request('initialize')['status']=='READY'
        row=service.request('assess',label='artificial',path=str(tmp_path/'candidate.npz'))
        assert row['budget_PASS'] and row['sector_error']<1e-10
        assert row['representation']['max_entry']<1e-10 and row['rigorous_enclosure'] is None
        with pytest.raises(ValueError,match='OPTIMIZATION_ONLY'):service.request('performance')
    finally:service.close()
