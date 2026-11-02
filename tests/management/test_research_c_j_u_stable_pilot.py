"""Artificial pipeline only; reuse prior snapshot/sector fixtures and E2E."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_j_u_stable_pilot as m
from test_research_c_separated_J_U_consistent_pilot import fixture as prior_fixture, write, FIT_PYTHON


def fixture(root):
    c=prior_fixture(root)
    c.update(scope=m.SCOPE,base_commit=m.BASE,conditions=m.CONDITIONS,
        algorithm_policy=asdict(m.algorithm.DEFAULT),call_limits=m.LIMITS,
        parent_limits=m.PARENT,per_arm_limits=m.WORKER,maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH,entry='scripts/run_research_c_j_u_stable.py')
    (root/'scripts').mkdir()
    (root/c['entry']).write_text('# artificial CLI identity only\n')
    manifest=m.io.read(root/'SOURCE.json')
    manifest['files'][c['entry']]=m.io.sha(root/c['entry'])
    write(root/'SOURCE.json',manifest)
    c['source_manifest_sha256']=m.io.sha(root/'SOURCE.json')
    write(root/'contract.json',c)
    return c


def test_passive_no_decode_no_optimizer_and_fixed_call_budget(tmp_path,monkeypatch):
    c=fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    assert m.MAX_CALLS==1563 and sum(m.WORKER.values())==776
    assert m.WORKER['inner_J_solve']==384
    c['algorithm_policy']['max_points']=129
    write(tmp_path/'contract.json',c)
    with pytest.raises(ValueError,match='FIXED_REDUCED'):m.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('fault',[None,'representation'])
def test_real_artificial_workers_guarded_PR_and_physical_parent_STOP(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path);out=tmp_path/m.FRESH
    code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from trottertracks import c_j_u_stable_pilot as m;r=Path(sys.argv[2]);c=m.preflight(r,r/'contract.json',fit=True);m.fit_worker(r,Path(sys.argv[3]),c,sys.argv[4])"
    def launch(root,path,output,contract,python,workers,calls):
        from concurrent.futures import ThreadPoolExecutor
        def one(role):
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_ENABLE_X64='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (output/(role+'_worker.txt')).open('x') as log:
                subprocess.run([str(FIT_PYTHON),'-B','-c',code,str(ROOT/'src'),str(root),str(output/role),role],
                               env=env,stdout=log,stderr=log,check=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in [pool.submit(one,k) for k in m.ROLES]:f.result()
    monkeypatch.setattr(m.common,'launch_workers',launch)
    original=m.subprocess.check_output
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:'SYNTHETIC\n' if args==['git','rev-parse','HEAD'] else original(args,**kw))
    if fault:
        def bad(*a,**k):raise ValueError('ARTIFICIAL_REPRESENTATION_FAIL')
        monkeypatch.setattr(m.molecule,'representation_gate',bad)
    with threadpool_limits(limits=1):
        if fault:
            with pytest.raises(ValueError,match='ARTIFICIAL_REPRESENTATION_FAIL'):
                m.run(tmp_path,tmp_path/'contract.json',out,FIT_PYTHON)
        else:
            r=m.run(tmp_path,tmp_path/'contract.json',out,FIT_PYTHON)
            assert r['status']=='ONE_OR_BOTH_FIRST_ORDER_UNQUALIFIED'
            assert r['qualified_tail_comparison'] is None
            assert r['reports']['pr']['optimization']['status']=='REDUCED_GRADIENT_UNQUALIFIED'
            assert r['reports']['rc2']['optimization']['first_order_qualified']
            assert r['PF_error'] is r['shots'] is r['CA_achieved'] is None
    stop=m.io.read(out/'STOP.json')
    assert stop['scientific_units']==(33 if fault else 37)
    assert stop['failed']==bool(fault) and stop['mandatory_stop']
    assert stop['call_counts']['inner_J_solve']==6
    assert stop['call_counts']['precision_state']==4 and stop['call_counts']['reduced_admission']==2
    assert stop['retry']==stop['resume']==0
    assert all(m.io.sha(tmp_path/p)==h for p,h in c['saved_inputs'].items())
    assert m.io.read(out/'pr'/'optimization.json')['qualification']['status']=='REDUCED_GRADIENT_UNQUALIFIED'
    with pytest.raises(ValueError,match='FRESH_REDUCED'):m.run(tmp_path,tmp_path/'contract.json',out,FIT_PYTHON)


@pytest.mark.parametrize('qualified,budget',[(True,True),(False,True),(True,False)])
def test_comparison_qualification_DF_and_unknown_resource_separate(qualified,budget):
    rows={k:dict(sector_error=1e-8 if budget else 1.,numerical_guard=1e-9) for k in m.ROLES}
    reports={}
    for k in m.ROLES:
        s=dict(components=dict(objective=1.,exact_tail_mass_Ha=1. if k=='pr' else 2.),first_order_met=qualified)
        reports[k]=dict(old_metric=deepcopy(s),optimization=dict(terminal=s,first_order_qualified=qualified,
            qualification=dict(status='NUMERICAL_PILOT_ADMITTED' if qualified else 'J_UNQUALIFIED')))
    r=m.decide(rows,reports,{})
    assert (r['qualified_tail_comparison'] is not None)==(qualified and budget)
    assert r['cross_objective_raw_F_ranking'] is r['total_cost'] is None


def test_legacy_reduced_comparison_preserves_old_status():
    rows={k:dict(sector_error=1e-8,numerical_guard=1e-9) for k in m.ROLES}
    s=dict(components=dict(objective=1.,exact_tail_mass_Ha=2.),first_order_met=False)
    old=dict(status='REDUCED_GRADIENT_UNQUALIFIED',terminal=s,qualification={'old_allowance':1e-6})
    reports={k:dict(old_metric=deepcopy(s),optimization=dict(terminal=deepcopy(s),
        status='U_UNQUALIFIED',first_order_qualified=False,qualification={'latest_pair':True})) for k in m.ROLES}
    refs={k:dict(legacy_reduced_optimization=deepcopy(old)) for k in m.ROLES}
    r=m.decide(rows,reports,refs)
    assert r['legacy_reduced_comparison']['pr']['saved_old_status']==old['status']
    assert r['legacy_reduced_comparison']['pr']['old_result_not_reclassified']
    assert refs['pr']['legacy_reduced_optimization']==old


def test_input_failure_prevents_science(tmp_path,monkeypatch):
    fixture(tmp_path);(tmp_path/'input.npz').write_bytes(b'invented mismatch')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):
        m.preflight(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()
