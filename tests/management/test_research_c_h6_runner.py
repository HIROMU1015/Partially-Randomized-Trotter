"""Artificial integration only. Saved H6/H4 molecular arrays are never loaded."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pickle
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_h6_runtime as run
from trottertracks import c_h6_downstream as down
from trottertracks import c_h6_validation_preparation as design
from trottertracks.c1_stage3_integrated import Calls


def small_backend(tmp_path):
    from trottertracks.c_near_transfer import System,PhysicalContext
    from trottertracks.c1_h6_initialization import reconstruct
    from trottertracks.c_pr_aware_df_stage_b_runner import save_arrays
    from trottertracks.c1_stage1 import bit_reverse_indices
    j=np.array([[[.0001,.00001],[.00001,.0002]],
                [[-.0001,.00002],[.00002,.00015]],[[.0002,-.00001],[-.00001,.0001]]])
    u=np.tile(np.eye(2),(3,1,1))
    packet=dict(h=np.diag([-.02,.01]),g=reconstruct(j,u),constant=np.array(.03))
    save_arrays(tmp_path/'input.npz',**packet);digest=run.io.sha(tmp_path/'input.npz')
    system=System(2,3,1,1.,digest);physical=PhysicalContext(packet,system,observed_input_sha256=digest)
    save_arrays(tmp_path/'candidate.npz',J=j,U=u);candidate=run.io.sha(tmp_path/'candidate.npz')
    df,generators=physical.assess(j,u,candidate_sha256=candidate,budget_Ha=.00019875)
    save_arrays(tmp_path/'reference.npz',J=j,U=u,generators=np.array(generators),H=sum(generators))
    df['reference_path']='reference.npz'
    from trottertracks.c_integrated_fit_worker import _plain
    checkpoint=_plain(dict(candidate_path='candidate.npz',candidate_sha256=candidate,physical=df))
    common=dict(p=packet,ctx=physical.context,system=asdict(system),input_identity=digest,
        basis=bit_reverse_indices(4)[physical.context.sector.basis_indices])
    out=tmp_path/'backend';out.mkdir();journal=tmp_path/'journal';journal.mkdir()
    call=Calls(journal,design.workload()['stage_units'],218350)
    backend=down.Backend(tmp_path,dict(source_sha256='synthetic'),out,call,common)
    task=dict(id='invented_PR',arm='PR',strength=.1,start='C0',budget=64,geometry='INVENTED')
    return backend,task,checkpoint,call


def test_real_reused_engines_small_artificial_native_accounting_D(tmp_path,monkeypatch):
    from trottertracks import c_pr_aware_df_stage_b as b
    checked=[];original=b.tail_gate
    def tracked(*args,**kwargs):
        checked.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(b,'tail_gate',tracked)
    backend,task,checkpoint,call=small_backend(tmp_path)
    a=backend.prepare(task,checkpoint)
    for r in (32,64,128):
        policy=backend.references(a,r)
        assert policy['eligible'] and policy['r']==r
    assert len(a['exact'])==13 and len(a['pf_errors'])==13
    assert len(checked)==1 and len(a['exponential_cache'])==2
    rows,labels=backend.native(a,a['policies'][32],'selection',32)
    assert len(rows)==32 and labels[0]=='COMPLETE_SECTOR_NUMERICAL_CHECK_PASS'
    assert labels[1:]==[down.UNTESTED]*31
    assert call.counts['native_state_actions']==6*8
    cost=backend.account(a,a['policies'][32],rows,labels)
    b=cost['bindings']['ideal_exact_native'];total=cost['result']['lanes']['ideal_exact_native']['conditional_totals']['totals']['rz_count']
    assert total==pytest.approx(b['W']*cost['calibration']['sample_mean']['rz_count']+b['shell']['rz_count'])
    assert cost['bindings']['evidence'] is None
    D=backend.deterministic(a)
    assert D['candidate']==a['candidate'] and D['input_sha256']==backend.common['input_identity']
    assert D['ideal_exact_native']['cost']['totals'] is not None
    assert call.counts['native_state_actions']==2*6*8
    report=json.loads((a['out']/'r32_selection/sample000_correctness.json').read_text())
    assert 'complex_XY' in report and report['output_dimension']==32
    assert len(list((a['out']/'r32_selection').glob('*inverse.qpy')))==2
    assert len(list((a['out']/'r32_selection').glob('*.qpy')))==66


def test_r32_zero_attenuation_cannot_poison_r64_PF_cache(tmp_path,monkeypatch):
    from trottertracks import c_pr_aware_df_m4 as m4
    original=m4.rte_round
    def zero_round(kernel,m,saved,**kwargs):
        if kernel['config'].rte_steps==32:
            return dict(pf_operator_error=saved,rte_corrected_operator_error=None,attenuation=0.)
        return original(kernel,m,saved,**kwargs)
    monkeypatch.setattr(m4,'rte_round',zero_round)
    backend,task,checkpoint,_=small_backend(tmp_path);a=backend.prepare(task,checkpoint)
    assert not backend.references(a,32)['eligible']
    assert len(a['pf_errors'])==13
    assert backend.references(a,64)['eligible']


@pytest.mark.parametrize('fault',['candidate','input','reference'])
def test_candidate_reference_identity_is_not_shared(tmp_path,fault):
    backend,task,checkpoint,call=small_backend(tmp_path)
    if fault=='candidate':checkpoint['candidate_sha256']='f'*64
    if fault=='input':checkpoint['physical']['input_sha256']='f'*64
    if fault=='reference':
        with np.load(tmp_path/'reference.npz') as source:values={k:source[k] for k in source.files}
        values['J']=values['J']+.001
        from trottertracks.c_pr_aware_df_stage_b_runner import save_arrays
        (tmp_path/'reference.npz').unlink();save_arrays(tmp_path/'reference.npz',**values)
    with pytest.raises(ValueError):backend.prepare(task,checkpoint)


class FakePhysical:
    def __init__(self,owner):self.owner=owner;self.closed=False
    def request(self,operation,**data):
        if operation=='initialize':
            path=self.owner.out/'invented_common.pkl'
            with path.open('wb') as stream:pickle.dump(dict(input_identity='a'*64,ctx=SimpleNamespace(d=400),system=data['system']),stream)
            return dict(common_pickle=str(path.relative_to(self.owner.root)))
        return dict(budget_PASS=self.owner.mode!='initial_DF')
    def close(self):self.closed=True


class FakeBackend:
    workers=2
    def __init__(self,root,c,out,call,common,**kwargs):
        self.root,self.c,self.out,self.call,self.common=Path(root),c,Path(out),call,common
        self.mode=c.get('synthetic_mode','pass');self.checked=[];self.draws=[];self.D=[]
        self.shells={'reports':{x:dict(metrics={k:2 for k in self.metrics()}) for x in ('head','X','Y')}}
    @staticmethod
    def metrics():
        from trottertracks.c1_stage3_accuracy import METRICS
        return METRICS
    def ensure_shells(self):return self.shells
    def prepare(self,task,checkpoint):
        if self.mode=='endpoint_DF' and task['strength']==.3:return None
        return dict(task=task,candidate=checkpoint['candidate_sha256'],policies={})
    def references(self,a,r):
        from trottertracks import c_rte_r_retuning as ret
        refs=[dict(m=m,reference=dict(pf_operator_error=.00001,rte_corrected_operator_error=.00001,attenuation=.99)) for m in range(13)]
        packet=dict(target_id=a['candidate']+str(r),rounds=refs,df_bound_Ha=1e-8,error_evidence=None,shells=self.shells['reports'])
        lanes=ret.assess_lanes(packet,ret.m4.accuracy.AccuracyTarget(**ret.m4.TARGET),lambda k,l,f:f())
        value=dict(r=r,eligible=not(self.mode=='r32_ineligible' and a['task']['arm']=='ENTRY' and r==32),packet=packet,lanes=lanes)
        a['policies'][r]=value;return value
    def native(self,a,p,role,count):
        r=p['r'];seed=down.seed_policy(self.c['source_sha256'],a['task'],a['candidate'],r,role)
        self.draws.append((a['task']['arm'],a['candidate'],r,role,count,seed['cohort']))
        self.checked.append((a['task']['id'],r,role,0))
        self.call('native_state_actions','mock complete800',lambda:None,count=4800)
        if self.mode in ('phase','inverse','leakage'):
            def fail():raise ValueError('ARTIFICIAL_'+self.mode)
            self.call('native_build_compile_and_reports','mock failure',fail)
        g=a['task']['strength']
        unit=100 if (r==32 and g==.03) or (r==128 and g==.1 and self.mode!='overlap') else 200
        if r==128 and g==.1 and self.mode!='overlap':unit=90
        rows=[{k:unit+i%9 for k in self.metrics()} for i in range(count)]
        return rows,['COMPLETE_SECTOR_NUMERICAL_CHECK_PASS']+[down.UNTESTED]*(count-1)
    def account(self,a,p,rows,labels):return down.Backend.account(self,a,p,rows,labels)
    def deterministic(self,a):
        self.D.append(a['candidate']);self.checked.append((a['task']['id'],0,'D',0))
        self.call('native_state_actions','mock D complete800',lambda:None,count=4800)
        totals={k:1e12 for k in self.metrics()}
        return dict(candidate=a['candidate'],input_sha256=self.common['input_identity'],ideal_exact_native=dict(cost=dict(totals=totals)))


class SyntheticRun(run.Run):
    def fit(self,task,physical=None,affinity=None):
        op=task['operation']
        if op=='prepare_codec':
            path=self.out/'invented_C0';path.write_bytes(b'invented neutral')
            return dict(path=str(path.relative_to(self.root)),sha256=run.io.sha(path))
        if op=='tensor_scale':return dict(objective_from_tensor=1.)
        if op=='specs':
            shared=dict(rank=14,norb=6,ld=4,loss_scale=1.,mass_scale=1.,epsilon=1e-8)
            return dict(PR=dict(shared,gamma=task['strength']),ENTRY=dict(shared,strength=task['strength']))
        assert op=='cell' and task['system']['norb']==6 and task['system']['rank']==14
        assert task['objective_spec']['ld']==4
        token='same' if self.mode=='same_candidate' else task['job_id']
        candidate=hashlib.sha256(token.encode()).hexdigest()
        checkpoint=dict(candidate_sha256=candidate,physical=dict(budget_PASS=True))
        return dict(scientific_units=3213,checkpoints={'16':dict(checkpoint),'64':dict(checkpoint)},
            J_stationarity='MET',U_stationarity='NOT_MET',accepted_U_updates=64)


def synthetic(tmp_path,mode='pass'):
    c=deepcopy(json.loads((ROOT/run.CONTRACT).read_text()))
    c.update(fresh_output='new/run01',source_sha256='synthetic',synthetic_mode=mode)
    c['physical_generation']['input_sha256']='a'*64
    source=tmp_path/c['execution_SOURCE'];source.parent.mkdir(parents=True);source.write_text('{}')
    contract=tmp_path/run.CONTRACT;contract.parent.mkdir(parents=True);contract.write_text('{}')
    c['neutral'].update(path='old_C0',sha256='b'*64)
    owner=SyntheticRun(tmp_path,c,fit_python=sys.executable,eval_python=sys.executable,
        backend_factory=FakeBackend,physical_factory=FakePhysical)
    owner.mode=mode
    old=tmp_path/'old_STOP';old.write_text('old stopped evidence')
    return owner,old


@pytest.mark.parametrize('mode',['pass','overlap','same_candidate','endpoint_DF','r32_ineligible','initial_DF'])
def test_synthetic_parent_E2E_counts_selection_final_D_and_qualifications(tmp_path,mode):
    owner,old=synthetic(tmp_path,mode);before=old.read_bytes()
    result=owner.execute();stop=run.io.read(owner.out/'STOP.json')
    assert not stop['failed'] and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert stop['scientific_units']<=218350 and old.read_bytes()==before
    if mode=='initial_DF':
        assert result['status']=='UNRESOLVED_INITIAL_DF' and not result['final'];return
    assert len(result['optimization'])==6 and len(result['selection'])==18
    assert all(set(v['checkpoints'])=={'16','64'} for v in result['optimization'].values())
    assert all(row['cohort_role']=='selection' and row['samples']==32 for row in result['selection'])
    assert len(result['final'])==(2 if mode=='overlap' else 3 if mode=='r32_ineligible' else 4)
    for value in result['final'].values():
        assert len(value['rows'])==128
        assert value['qualification']['unchecked_price_circuit_equivalence_assumed']
        assert value['result']['numerical_labels'][1:]==[down.UNTESTED]*127
    for value in result['comparisons'].values():
        if value.get('status')=='UNRESOLVED_MISSING_FINAL_PAIR':continue
        assert value['confidence']==.975 and value['samples_per_arm']==128
        assert value['ratio_interval'] is None
    if mode=='same_candidate':assert len(result['D'])==1 and len(result['final'])==4
    assert stop['call_counts']['native_state_actions']<=124800
    if mode=='pass':assert stop['call_counts']['native_state_actions']==124800
    with pytest.raises(FileExistsError):owner.execute()


@pytest.mark.parametrize('fault',['phase','inverse','leakage'])
def test_gate_failure_propagates_partial_and_mandatory_STOP(tmp_path,fault):
    owner,old=synthetic(tmp_path,fault)
    with pytest.raises((ValueError,RuntimeError)):owner.execute()
    stop=run.io.read(owner.out/'STOP.json')
    assert stop['failed'] and stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert (owner.out/'partial.json').exists() and (owner.out/'failure.json').exists()
    assert not (owner.out/'result.json').exists() and old.read_text()=='old stopped evidence'


def test_close_failure_cannot_publish_success(tmp_path):
    owner,_=synthetic(tmp_path,'initial_DF')
    class BadClose(FakePhysical):
        def close(self):raise RuntimeError('ARTIFICIAL_CLOSE_FAILURE')
    owner.physical_factory=BadClose
    with pytest.raises(RuntimeError,match='SERVICE_CLOSE_FAILED'):owner.execute()
    assert run.io.read(owner.out/'STOP.json')['failed']
    assert not (owner.out/'result.json').exists()


def test_real_pool_RPC_synthetic_selection_and_final128(tmp_path,monkeypatch):
    import shutil
    from trottertracks.c_h6_parallel import PoolBackend
    backend,task,checkpoint,call=small_backend(tmp_path)
    monkeypatch.setenv('PYTHONPATH',str(ROOT/'src'))
    script=tmp_path/run.ENTRY;script.parent.mkdir();shutil.copyfile(ROOT/run.ENTRY,script)
    contract=dict(source_sha256='invented-RPC',workload_ceiling=design.workload(),
        physical_generation=dict(input_sha256=backend.common['input_identity']),
        fresh_output='pool_run',environment={},runtime_batch_columns=3)
    owner=run.Run(tmp_path,contract,fit_python=sys.executable,eval_python=sys.executable)
    owner.out.mkdir();owner.call=Calls(owner.out,design.workload()['stage_units'],218350)
    pool=PoolBackend(owner,backend.common,[(task,checkpoint)],dict(workers=1,threads_per_worker=1))
    try:
        a=pool.prepare(task,checkpoint);policy=pool.references(a,32)
        for role,n in (('selection',32),('final',128)):
            rows,labels=pool.native(a,policy,role,n)
            account=pool.account(a,policy,rows,labels)
            assert len(rows)==n and account['qualification']['unchecked_price_circuit_equivalence_assumed']
            pool.park(a)
        d=pool.deterministic(a)
        assert d['candidate']==checkpoint['candidate_sha256']
    finally:
        for service in owner.services:service.close()
    counts=owner.call.counts
    assert counts['native_state_actions']==3*6*8
    assert 'first_registered_native_batch' in (owner.out/'progress.jsonl').read_text()
    folder=owner.out/'evaluation_workers/worker0/candidates'/task['id']
    sel=run.io.read(folder/'r32_selection/sample000.json')
    final=run.io.read(folder/'r32_final/sample000.json')
    assert sel['seed']!=final['seed'] and sel['cohort']['role']!=final['cohort']['role']


def test_artificial_six_orbital_physical_RPC_and_fit_codec(tmp_path,monkeypatch):
    import shutil
    from trottertracks.c_integrated_execution_rpc import RpcProcess
    from trottertracks.c_near_transfer import System
    from trottertracks.c1_h6_initialization import reconstruct
    from trottertracks.c_pr_aware_df_stage_b_runner import save_arrays
    j=np.zeros((14,6,6));j[-1]=np.eye(6)*1e-6;u=np.tile(np.eye(6),(14,1,1))
    save_arrays(tmp_path/'invented_input.npz',h=np.diag(np.arange(6)*.001),g=reconstruct(j,u),constant=np.array(.01))
    save_arrays(tmp_path/'invented_C0.npz',J=j,U=u)
    system=asdict(System(6,14,4,1.,run.io.sha(tmp_path/'invented_input.npz')))
    monkeypatch.setenv('PYTHONPATH',str(ROOT/'src'))
    with RpcProcess(tmp_path,sys.executable,'physical',tmp_path/'physical',
            script_path=ROOT/run.ENTRY,config=dict(threads=1)) as physical:
        context=physical.request('initialize',system=system,input_rel='invented_input.npz')
        row=physical.request('assess',label='invented_rank14',path=str(tmp_path/'invented_C0.npz'))
        assert row['actual_rank']==14 and row['budget_PASS']
        with np.load(tmp_path/row['reference_path']) as data:assert data['generators'].shape==(15,400,400)
    fit_python=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-server-20261106/.venv-fit/bin/python')
    script=tmp_path/run.ENTRY;script.parent.mkdir();shutil.copyfile(ROOT/run.ENTRY,script)
    c=dict(fresh_output='codec_run',environment={},source_sha256='invented')
    owner=run.Run(tmp_path,c,fit_python=fit_python,eval_python=sys.executable);owner.out.mkdir()
    task=dict(root=str(tmp_path),system=system,input_path='invented_input.npz',
        cold_path='invented_C0.npz',cold_sha256=run.io.sha(tmp_path/'invented_C0.npz'),
        operation='prepare_codec',job_id='invented_codec',out='codec_run/codec')
    result=owner.fit(task)
    assert result['original_JU_preserved']
    with np.load(tmp_path/result['path']) as data:assert data['x0'].shape==(504,)
