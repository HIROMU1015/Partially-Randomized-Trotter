"""Synthetic integration, physical reference, and conditional-accounting tests."""
from dataclasses import asdict
from pathlib import Path
import json,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_integrated_runner as m


def fixture(root,monkeypatch):
    from trottertracks.c_pr_integrated_evaluation import arrays
    from test_research_c_separated_J_U import invented
    g,J,U,s=invented(n=4,rank=8,ld=2)
    inputs=dict(input='input.npz',initial='initial.npz',scales='scales.json')
    arrays(root/'input.npz',h=np.diag(np.arange(4)*.1),g=g,constant=np.asarray(0.))
    arrays(root/'initial.npz',J=J,U=U);(root/'scales.json').write_text('{}')
    arrays(root/'terminal.npz',J=J,U=U);arrays(root/'independent.npz',J=J,U=U,eta=np.zeros(48));arrays(root/'event.npz',eta=np.zeros(48))
    (root/'old_result.json').write_text(json.dumps(dict(accepted_U_updates=1,terminal_label='1')))
    (root/'old_STOP.json').write_text(json.dumps(dict(failed=False,scientific_units=19)))
    accepted=dict(terminal='terminal.npz',independent='independent.npz',event='event.npz',result='old_result.json',STOP='old_STOP.json')
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# synthetic entry\n')
    (root/'SOURCE.json').write_text(json.dumps(dict(files={m.ENTRY:m.io.sha(entry)})))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,entry=m.ENTRY,fresh_output=m.FRESH,policy=asdict(m.algorithm.DEFAULT),fractions=list(m.algorithm.FRACTIONS),max_updates=64,objective=asdict(s),DF_budget_Ha=.00019875,target=m.performance.m4.TARGET,sampling=m.performance.SAMPLING,parent_call_limits=m.LIMITS,performance_call_limits=m.performance.LIMITS,maximum_logical_units=m.MAX_CALLS,retry=0,resume=0,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),input_paths=inputs,accepted=accepted,fit_packages={},evaluation_packages={},ffsim_sources={},optimization_references={})
    c['saved_inputs']={str(p.relative_to(root)):m.io.sha(p) for p in root.iterdir() if p.is_file() and p.name!='SOURCE.json'}
    (root/'contract.json').write_text(json.dumps(c));monkeypatch.setattr(m.prior,'OBJECTIVE',asdict(s));return c


class Service:
    calls=[];mode='pass'
    def __init__(self,*a):self.__class__.calls=[]
    def request(self,operation,**kw):
        self.calls.append(operation)
        if operation=='assess':
            if self.mode=='correctness':raise ValueError('SYNTHETIC_CORRECTNESS')
            return dict(budget_PASS=self.mode!='DFfail',exact_tail_lambda_Ha=1.,sector_error=1e-9,numerical_guard=1e-9)
        if operation=='performance':return dict(status='SYNTHETIC_PERFORMANCE')
        return dict(status='READY')
    def close(self):pass


@pytest.mark.parametrize('mode',['pass','DFfail','correctness'])
def test_phase_A_dependency_STOP_and_fresh_scope(tmp_path,monkeypatch,mode):
    fixture(tmp_path,monkeypatch);monkeypatch.setattr(Service,'mode',mode)
    def trajectory(g,j,u,s,physical,**kw):
        p=physical('B_initial',j,u);state=dict(first_order_met=False)
        row=dict(id='B_initial',state=state,physical=p,update=0)
        return dict(status='LIMITED',can_performance=True,best=dict(row=row,J=j,U=u),terminal=None,history=[row])
    monkeypatch.setattr(m.algorithm,'trajectory',trajectory)
    if mode=='correctness':
        with pytest.raises(ValueError,match='SYNTHETIC_CORRECTNESS'):m.run(tmp_path,tmp_path/'contract.json',service_factory=Service)
    else:
        r=m.run(tmp_path,tmp_path/'contract.json',service_factory=Service)
        assert ('performance' in Service.calls)==(mode=='pass')
        if mode=='DFfail':assert r['trajectory'] is None
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed']==(mode=='correctness') and stop['scientific_units']<=m.MAX_CALLS
    with pytest.raises(ValueError,match='FRESH_INTEGRATED'):m.run(tmp_path,tmp_path/'contract.json',service_factory=Service)


def test_passive_input_mismatch_and_scope(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    m.preflight(tmp_path,tmp_path/'contract.json')
    (tmp_path/'event.npz').write_bytes(b'synthetic mismatch')
    with pytest.raises(ValueError,match='SOURCE_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')


def synthetic_packet():
    from trottertracks.c_separated_j_u import linear
    U=np.broadcast_to(np.eye(4),(8,4,4)).copy();J=np.broadcast_to(np.eye(4)*1e-4,(8,4,4)).copy()
    return dict(h=np.diag(np.arange(4)*.1),g=linear.reconstruct(J,U),constant=np.asarray(0.)),J,U


def test_real_synthetic_sector_generalK_and_original_kappa(tmp_path):
    from trottertracks.c_pr_integrated_evaluation import Evaluation,arrays
    p,J,U=synthetic_packet();arrays(tmp_path/'input.npz',**p);arrays(tmp_path/'candidate.npz',J=J,U=U)
    out=tmp_path/'out';out.mkdir();e=Evaluation(tmp_path,dict(input_paths={'input':'input.npz'},DF_budget_Ha=.00019875),out)
    row=e.assess('synthetic',tmp_path/'candidate.npz')
    assert row['budget_PASS'] and row['sector_error']<1e-10 and row['tensor_frobenius']==0
    assert row['kappa']=='original_g_corrected' and row['rigorous_enclosure'] is None


def test_real_synthetic_PF_RTE_RPE_and_mock_native_cost(tmp_path,monkeypatch):
    from trottertracks import c_pr_integrated_performance as perf
    from trottertracks.c_pr_integrated_evaluation import Evaluation,arrays
    p,J,U=synthetic_packet();arrays(tmp_path/'input.npz',**p);arrays(tmp_path/'candidate.npz',J=J,U=U)
    out=tmp_path/'out';out.mkdir();e=Evaluation(tmp_path,dict(input_paths={'input':'input.npz'},DF_budget_Ha=.00019875),out);e.assess('synthetic',tmp_path/'candidate.npz')
    metrics={k:1 for k in perf.m4.METRICS};(tmp_path/'shells.json').write_text(json.dumps({k:dict(metrics=metrics) for k in ('head','X','Y')}))
    def native(*a):
        rows=[dict(metrics) for _ in range(32)];return rows,perf.b2.calibration(rows)
    monkeypatch.setattr(perf,'native',native)
    c=dict(reference_records={'shells':'shells.json'},input_paths={'input':'input.npz'},saved_inputs={'input.npz':'SYNTHETIC'})
    r=perf.run(tmp_path,c,out/'C',p,e.ctx,e.physical,e.cache['synthetic'])
    assert len(r['rounds'])==13 and r['lanes']['ideal_exact_native']['total_shots']>0
    assert r['lanes']['evidence']['total_shots'] is None
    assert r['cost_support'] is r['actual_CA_achieved'] is r['whole_law_synthesis'] is None
    assert r['lanes']['ideal_exact_native']['conditional_totals']['totals'] is not None


def test_real_synthetic_worker_pipe(tmp_path):
    import subprocess
    from trottertracks.c_pr_integrated_evaluation import arrays
    p,J,U=synthetic_packet();arrays(tmp_path/'input.npz',**p);arrays(tmp_path/'candidate.npz',J=J,U=U)
    out=tmp_path/'out';out.mkdir()
    c=dict(input_paths={'input':'input.npz'},DF_budget_Ha=.00019875)
    code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from trottertracks.c_pr_integrated_evaluation import worker;worker(Path(sys.argv[2]),json.loads(sys.argv[3]),Path(sys.argv[4]))"
    requests=[dict(operation='initialize'),dict(operation='assess',label='synthetic',path=str(tmp_path/'candidate.npz')),dict(operation='close')]
    r=subprocess.run([sys.executable,'-B','-c',code,str(Path(__file__).resolve().parents[2]/'src'),str(tmp_path),json.dumps(c),str(out)],
        input=''.join(json.dumps(x)+'\n' for x in requests),text=True,capture_output=True,check=True)
    replies=[json.loads(x) for x in r.stdout.splitlines()]
    assert len(replies)==2 and all(x['ok'] for x in replies) and replies[1]['value']['budget_PASS']
