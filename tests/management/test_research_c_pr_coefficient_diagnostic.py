"""Synthetic/replay/mock E2E only. Scientific NPZ data is never opened."""
from pathlib import Path
from dataclasses import asdict
import json,sys
import numpy as np
import pytest
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_coefficient_diagnostic as m
from test_research_c_pr_coefficient_point import failing_invented


def test_actual_SciPy_replay_recovers_next_query_without_evaluating_it():
    samples=[];third=[]
    class Captured(Exception):pass
    def original(x):
        if len(samples)==2:third.append(x.copy());raise Captured()
        value=float(np.exp(10*x[0])-x[0]);gradient=np.array([10*np.exp(10*x[0])-1])
        samples.append(dict(eta=x.copy(),value=value,gradient=gradient));return value,gradient
    with pytest.raises(Captured):
        minimize(original,np.zeros(1),jac=True,method='BFGS',options=dict(maxiter=64,gtol=1e-7,norm=2,xrtol=0.,c1=1e-4,c2=.9))
    np.testing.assert_array_equal(m.capture_third_query(samples),third[0])
    broken=[dict(r) for r in samples];broken[1]['eta']=broken[1]['eta']+.01
    with pytest.raises(ValueError,match='SAVED_REPLAY_QUERY_MISMATCH'):m.capture_third_query(broken)


@pytest.mark.parametrize('mode',['repeat','accepted','returned'])
def test_replay_fails_closed_instead_of_searching_another_condition(mode):
    rows=[dict(eta=np.array([0.]),value=1.,gradient=np.array([1.])),dict(eta=np.array([1.]),value=2.,gradient=np.array([1.]))]
    def driver(fun,x,**kw):
        fun(x)
        if mode=='repeat':fun(x)
        if mode=='accepted':kw['callback'](None)
    with pytest.raises(ValueError):m.capture_third_query(rows,driver=driver)


def fixture(root,monkeypatch):
    p,J,raw,info=failing_invented(41)
    U=p.mapping.U.copy();paths={k:k+('.jsonl' if k=='trace' else '.npz') for k in ('input','initial','envelope0','envelope1','state1','failing','trace')}
    for path in paths.values():(root/path).write_bytes(b'artificial file identity only')
    files={'entry.py':'# artificial SOURCE\n'}
    for path,body in files.items():(root/path).write_text(body)
    source={'files':{path:m.prior.io.sha(root/path) for path in files}}
    (root/'SOURCE.json').write_text(json.dumps(source))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,entry='scripts/run_research_c_pr_coefficient_point.py',
        maximum_logical_units=10,call_limits=m.LIMITS,fresh_output=m.FRESH,retry=0,resume=0,
        new_inner_solves=0,new_optimization_trajectories=0,mandatory_stop=True,objective=asdict(p.spec),
        source_manifest='SOURCE.json',source_manifest_sha256=m.prior.io.sha(root/'SOURCE.json'),
        saved_inputs={path:m.prior.io.sha(root/path) for path in paths.values()},input_paths=paths,packages={},ffsim_sources={})
    (root/'contract.json').write_text(json.dumps(c));monkeypatch.setattr(m,'OBJECTIVE',c['objective'])
    zero=np.zeros(48)
    data=dict(input={'g':p.g},initial={'J':J,'U':U},failing=raw,state1={'U':U},
        envelope0={'eta':zero,'gradient':zero},envelope1={'eta':zero,'gradient':zero},
        trace=[dict(kind='precision_state',label=f'{i}_refine2',components={'objective':1.}) for i in range(2)])
    return c,data


def test_passive_hash_checks_no_NPZ_decode_and_no_optimizer(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive array decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'input.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()


@pytest.mark.parametrize('fault',[None,'point','gradient','not_reproduced'])
def test_artificial_E2E_success_partial_FAIL_and_fresh_STOP(tmp_path,monkeypatch,fault):
    c,data=fixture(tmp_path,monkeypatch)
    if fault=='point':data['failing']['x0']=data['failing']['x0']+.01
    if fault=='gradient':
        real=m.model.CoefficientChart.assess
        def bad(self,z,**kw):raise ValueError('NEW_POINT_CONSISTENCY_MISMATCH')
        monkeypatch.setattr(m.model.CoefficientChart,'assess',bad)
    if fault=='not_reproduced':
        def agrees(self,z):
            value,g=self.p.vg(self.x0+self.W@z,independent=True)
            return value,self.W.T@g
        monkeypatch.setattr(m.model.legacy.CoefficientChart,'vg',agrees)
    loader=lambda *a:data;replay=lambda samples:np.zeros(48)
    if fault in ('point','gradient'):
        with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'contract.json',loader=loader,replay=replay)
    else:m.run(tmp_path,tmp_path/'contract.json',loader=loader,replay=replay)
    out=tmp_path/m.FRESH;stop=m.prior.io.read(out/'STOP.json')
    assert stop['scientific_units']==({'point':5,'gradient':8,'not_reproduced':6}.get(fault,10))
    assert stop['failed']==(fault in ('point','gradient')) and stop['retry']==stop['resume']==0
    if not fault:assert m.prior.io.read(out/'result.json')['status']=='NEW_SAVED_POINT_NUMERICAL_CONSISTENCY_PASS'
    if fault=='not_reproduced':assert m.prior.io.read(out/'result.json')['status']=='INCONCLUSIVE_LEGACY_MISMATCH_NOT_REPRODUCED'
    with pytest.raises(ValueError,match='FRESH_POINT_RUN'):m.run(tmp_path,tmp_path/'contract.json',loader=loader,replay=replay)
    assert all(m.prior.io.sha(tmp_path/p)==h for p,h in c['saved_inputs'].items())
