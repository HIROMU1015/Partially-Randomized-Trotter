"""m3 caller tests through the actual accelerated S/C path, synthetic events/fake compiler."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from compiled_price_artificial import fixture,backend
import run_pr_df_ld1_m3 as M

@pytest.fixture
def toy(monkeypatch,tmp_path):
    science,ctx,c,cell,plan,requests=fixture(q=2,R=3)
    from trotterlib import rte
    import numpy as np
    def forbidden(*args,**kwargs):raise AssertionError('No molecular inputs or real sampler in technical tests')
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(rte,'sample_rte_events',forbidden)
    monkeypatch.setattr(M.P,'summarize',forbidden);monkeypatch.setattr(M.R.Q,'fresh_summary',forbidden)
    seeds=[r.seed for r in requests];draws=[];compiles=[]
    ctx.h['sample_requests']=lambda *args:(draws.append(args) or requests,seeds,[])
    def compile_fake(wrapper,settings,**kwargs):
        compiles.append(kwargs['actual_circuit_fingerprint'])
        return SimpleNamespace(compiler_settings_hash='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b',
             rz_count=100+len(compiles),rz_depth=10,cx_count=20,cx_depth=10,total_depth=30,circuit_size=200)
    ctx.compile=compile_fake;cell.update(m=3,role='CALIBRATION',seeds=[seeds])
    plan.update(cells=[cell],compiler={'settings_hash':'1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'})
    plan['menus']['H4']['candidates'][0]['law_binding']={'model':'SYNTHETIC_FINITE_K2_ONLY'}
    plan['runtime']['old_cache']=str(tmp_path/'unused_cache')
    b=backend(science,ctx,plan,True);private=tmp_path/'private';private.mkdir()
    from compiled_price_artificial import dfq
    state=dict(plan=plan,ctx=ctx,backend=b,science=science,dfq=dfq,private=str(private))
    return state,draws,compiles

def test_plan_fixes_m3_only_and_separates_m4_and_saved_calibration():
    plan=M.make_inputs();assert len(plan['cells'])==1 and plan['cells'][0]['m']==3
    c=plan['cells'][0];h=plan['future_holdout']['cell']
    assert (c['candidate_id'],h['candidate_id'])==('045843b2429bece6f29b','3df0cfd3356c358aeda8')
    assert plan['compile_budget']==8 and c['path_indices']==h['path_indices']==[0,1,2,3]
    assert c['cohort']!=h['cohort'] and not plan['future_holdout']['execution_authorized']
    a={s for row in c['seeds'] for s in row};b={s for row in h['seeds'] for s in row}
    assert not a&b
    assert plan['model_plan']['holdout_rule']['tolerance'] is None
    assert not plan['model_plan']['fit_authorized'] and not plan['model_plan']['residual_aggregation_authorized']

def test_actual_entry_uses_one_sample_and_two_axes_same_digest(toy):
    state,draws,compiles=toy;result=M.run_path(0,state)
    assert len(draws)==1 and len(compiles)==2 and result['C']==[101,102]
    assert result['new_compiles']==2 and len(result['S'])==2 and 'D' not in result
    assert [r['axis'] for r in result['compile_log']]==['cosine','sine']
    d=Path(result['private_directory'])
    assert json.loads((d/'STRUCTURAL.json').read_text())['event_digest']==result['event_digest']
    for i,a in enumerate(('cosine','sine')):
        receipt=json.loads((d/(a+'.json')).read_text())
        assert receipt['fingerprint']==result['fingerprints'][i]
        assert receipt['metrics']['rz_count']==result['C'][i]
    assert not result['full_wrapper_statevector_validation']

def test_seed_mismatch_stops_before_any_compile(toy):
    state,draws,compiles=toy;state['plan']['cells'][0]['seeds']=[[999,999]]
    with pytest.raises(ValueError,match='SEEDS'):M.run_path(0,state)
    assert len(draws)==1 and compiles==[]
    assert (Path(state['private'])/'STOP_p0.json').is_file()

def test_second_axis_failure_preserves_first_receipt_and_no_retry(toy):
    state,draws,compiles=toy;original=state['ctx'].compile
    def fail_second(*a,**k):
        if len(compiles)==1:raise RuntimeError('synthetic second axis failure')
        return original(*a,**k)
    state['ctx'].compile=fail_second
    with pytest.raises(RuntimeError,match='second axis'):M.run_path(0,state)
    d=Path(state['private'])/'fresh'/state['plan']['cells'][0]['id']/'0'
    assert (d/'cosine.json').is_file() and not (d/'sine.json').is_file()
    stop=json.loads((Path(state['private'])/'STOP_p0.json').read_text())
    assert [r['axis'] for r in stop['compile_log']]==['cosine','sine']
    assert stop['compile_log'][0]['status']=='COMPLETE' and stop['compile_log'][1]['status']=='STARTED'
    assert len(draws)==1 and len(compiles)==1

def test_axis_receipt_mismatch_is_rejected(toy):
    state,draws,compiles=toy;original=state['backend'].price_requests
    def wrong(*a,**k):
        result=original(*a,**k);d=Path(a[2]);p=d/'sine.json';v=json.loads(p.read_text());v['fingerprint']='wrong';p.write_text(json.dumps(v));return result
    state['backend'].price_requests=wrong
    with pytest.raises(ValueError,match='AXIS_RECEIPT'):M.run_path(0,state)
    assert len(draws)==1 and len(compiles)==2
