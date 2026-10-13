"""Artificial metadata/mock supervisor tests; no scientific NPZ or transpile."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from trottertracks.resource_applicability import h8_resource_map_contract_v1 as c
from trottertracks.resource_applicability import h8_resource_map_handoff_v1 as h
from trottertracks.resource_applicability import h8_resource_map_execution_v1 as ex


@pytest.fixture(autouse=True)
def no_science(monkeypatch):
    import numpy as np
    import qiskit
    def forbidden(*a,**k):raise AssertionError('scientific execution forbidden')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(qiskit,'transpile',forbidden)


def task():
    cell=next(x for x in c.cells() if x['id']=='B2_p21_q2_2nd_r4_K4')
    return dict(kind='COST',cell=cell,cell_id=cell['id'],ordinary=False,
        validate=False,phase='exploration',replica=2,
        seed=c.task_seed(h.PARENT_SOURCE,cell['id'],'exploration',2))


def evidence(t):
    events=[[dict(taylor_order=0) for _ in range(t['cell']['r'])] for _ in range(t['cell']['q'])]
    tr=dict(task=t,events=events,event_digest=c.digest(events),observed_orders=[0]*t['cell']['R'])
    diag=dict(kind='begin',context=dict(stage='numeric_fingerprint',control='symmetric_directional',instructions=180000))
    return tr,diag,dict(VmHWM_bytes=2*2**30)


def manifests():
    old=dict(source_commit=h.PARENT_SOURCE,source_hashes={p:'a'*64 for p in h.CHANGED_SOURCE_PATHS},
        input_identity={'input':'fixed'},profile={'qiskit':'1.3.0'},compiler_hash=c.COMPILER,
        plan=c.plan(),cumulative_carry=c.carry(),proposed_resources=c.resources(),
        observation_policy=c.observation(),initial_registrations=c.signal_groups(),call_budget=c.call_budget())
    old['source_hashes']['science.py']='c'*64
    new=deepcopy(old);new['source_commit']='d'*40
    for p in h.CHANGED_SOURCE_PATHS:new['source_hashes'][p]='b'*64
    new['source_hashes'][h.NEW_SOURCE_PATH]='e'*64
    new.update(seed_source_commit=h.PARENT_SOURCE,handoff_authorization_basis=h.AUTHORIZATION)
    return old,new


def test_small_actual_p21_promotes_with_guards():
    t=task();tr,d,m=evidence(t)
    proof=h.light_case_evidence(t,tr,d,m)
    assert proof['rss_stop_bytes']==9*2**30 and not proof['hard_memory_bound_proved']
    assert ex.rss_reservation(t)==2**30


@pytest.mark.parametrize('mutation',['rare','forged_orders','wrong_task','bad_digest','oversize','ordinary','axis','wrong_stage','missing_peak','large_peak'])
def test_unknown_or_rare_not_promoted(mutation):
    t=task();tr,d,m=evidence(t)
    if mutation=='rare':
        tr['events'][0][0]['taylor_order']=2;tr['event_digest']=c.digest(tr['events']);tr['observed_orders'][0]=2
    if mutation=='forged_orders':
        tr['events'][0][0]['taylor_order']=2;tr['event_digest']=c.digest(tr['events'])
    if mutation=='wrong_task':tr['task']=dict(t,seed=1)
    if mutation=='bad_digest':tr['event_digest']='f'*64
    if mutation=='oversize':d['context']['instructions']=200001
    if mutation=='ordinary':t['ordinary']=True
    if mutation=='axis':d['context']['axis']='sine'
    if mutation=='wrong_stage':d['context']['stage']='wrapper_build'
    if mutation=='missing_peak':m={}
    if mutation=='large_peak':m['VmHWM_bytes']=6*2**30+1
    assert h.light_case_evidence(t,tr,d,m) is None


def test_source_transition_keeps_science_and_seed():
    old,new=manifests();before=c.cost_tasks
    proof=h.compatibility(new,old)
    assert proof['seed_source']==old['source_commit'] and not proof['scientific_source_changed']
    t=task();assert t['seed']==c.task_seed(proof['seed_source'],t['cell_id'],'exploration',2)
    assert t['seed']!=c.task_seed(new['source_commit'],t['cell_id'],'exploration',2)


@pytest.mark.parametrize('key',['source_commit','seed_source_commit','compiler_hash','input_identity','profile','plan','science_hash','call_budget'])
def test_transition_rejects_scientific_mutation(key):
    old,new=manifests()
    if key=='source_commit':old[key]='f'*40
    elif key=='seed_source_commit':new[key]='f'*40
    elif key=='science_hash':new['source_hashes']['science.py']='f'*64
    elif key=='call_budget':new[key]['trajectory']+=1
    elif key=='plan':new[key]['cells'][0]['q']=16
    else:new[key]='changed'
    with pytest.raises(ValueError):h.compatibility(new,old)


def test_held_reservation_budget_not_refunded():
    old,new=manifests();new['resume_parent']={'additional_held_reservations':{'compile':6,'trajectory':3}}
    new['call_budget']['compile']+=6;new['call_budget']['trajectory']+=3
    assert h.compatibility(new,old)['extra_reservations']=={'compile':6,'trajectory':3}


def test_only_minimum_budget_extension(tmp_path):
    # Four ineligible deterministic cells leave compile headroom, no PR headroom.
    old,_=manifests();pd=tmp_path/'exploration';pd.mkdir()
    tasks=[]
    for cell in c.cells():
        if cell['method']=='B0' and cell['q']==8:continue
        if cell['method']=='B0' and cell['q']==4 and cell['prefix']==21:continue
        for replica in range(8 if cell['R'] else 1):
            tasks.append(dict(cell=cell,ordinary=replica==0 and cell['id'] in c.plan()['sensitivity_cells'],validate=replica==0))
    (pd/'registered_tasks.json').write_text(json.dumps({'registrations':[{'task':t} for t in tasks]}))
    held=dict(compile=6,trajectory=3,occurrence=3,control_probe=0)
    assert c.minimum_resume_extension(old,tmp_path,held)=={'trajectory':3,'occurrence':3}
    assert held['compile']==6  # Retained even though the cap has sufficient headroom.


def test_native_identity_required_and_signal_reused(monkeypatch,tmp_path):
    assert tmp_path.is_relative_to('/home/AbeHiromu')
    old,new=manifests();parent=tmp_path/'parent';parent.mkdir()
    identity=dict(pid=9999999,start_ticks=123,pgrp=9999999,session=9999999)
    ledger=dict(known={'compile':2},reserved={'compile':2,'trajectory':1})
    for n,v in [('manifest.json',old),('coordinator_identity.json',identity),
        ('execution_terminal.json',{'SOURCE':h.PARENT_SOURCE,'ledger':ledger})]:
        (parent/n).write_text(json.dumps(v))
    native=tmp_path/'native.json';native.write_text(json.dumps(dict(identity=identity,exit_code=1,wait_reaped=True)))
    from trottertracks.resource_applicability import h6_server_resources_v1 as rs
    def absent(pid):raise FileNotFoundError()
    monkeypatch.setattr(rs,'process_identity',absent)
    pd=parent/'signal';pd.mkdir();d=pd/'group_0000';d.mkdir()
    st={'kind':'SIGNAL','cell_ids':['toy']}
    files={'registered_tasks.json':{'registrations':[{'index':0,'task':st}]},
        'execution_terminal.json':dict(all_dispatched_reaped=True,cleanup_errors=[],retired=[],completed_indices=[0])}
    for n,v in files.items():(pd/n).write_text(json.dumps(v))
    for n,v in {'native_reap_receipt.json':{'exit_code':0,'reaped':True},
        'worker_terminal.json':{'status':'COMPLETE','source_commit':h.PARENT_SOURCE,'task':st},
        'signal_result.json':{'signal_count':1},'signal_summary.json':{'signals':[{'cell':{'id':'toy'}}]}}.items():
        (d/n).write_text(json.dumps(v))
    bound=c.bind_resume_parent(new,parent,native)
    assert bound['resume_parent']['parent_ledger']==ledger
    assert bound['resume_parent']['accepted']['signal']['0']['result']['signal_count']==1
    assert bound['call_budget']['trajectory']==833
    assert not bound['resume_parent']['budget_extension_approved']
    bad=json.loads(native.read_text());bad['identity']['start_ticks']=124;native.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='NATIVE_IDENTITY'):c.bind_resume_parent(manifests()[1],parent,native)


def test_eight_small_workers_mock_admission(monkeypatch,tmp_path):
    """Reach eight simultaneously without exceeding 72GiB or real child processes."""
    assert tmp_path.is_relative_to('/home/AbeHiromu')
    run=tmp_path/'run';run.mkdir();host={'host_available_bytes':500*2**30,'effective_available_bytes':500*2**30,
        'oom':{'mock':{'oom':0,'oom_kill':0}},'full_psi_avg10':{'host':0,'mock':0}}
    (run/'fresh_resources.json').write_text(json.dumps({'resources':{'memory':host}}))
    spawned=[];reaped=[];t=task();registrations=[dict(index=i,task=dict(t,replica=i)) for i in range(8)]
    def spawn(argv,**kw):
        d=Path(argv[argv.index('--worker')+1]);b=json.loads((d/'task_binding.json').read_text())
        tr,diag,_=evidence(b['task'])
        request=h.memory_request(b['task'],tr,{'symmetric_directional':180000},'d'*40)
        for n,v in [('trajectory.json',tr),('diagnostic_0000.json',diag),('memory_admission_request.json',request)]:
            (d/n).write_text(json.dumps(v))
        p=SimpleNamespace(pid=900001+len(spawned));spawned.append(p);return p
    monkeypatch.setattr(ex.subprocess,'Popen',spawn)
    monkeypatch.setattr(ex,'process_identity',lambda pid:dict(pid=pid,start_ticks=pid,pgrp=pid,session=pid,state='R'))
    monkeypatch.setattr(ex.os,'waitid',lambda *a:SimpleNamespace() if len(list((run/'exploration').glob('group_*/memory_admission.json')))==8 else None)
    monkeypatch.setattr(ex,'stop_owned_group',lambda p,identity:reaped.append(p.pid) or 0)
    monkeypatch.setattr(ex,'host_memory',lambda:host)
    monkeypatch.setattr(ex,'process_memory',lambda pid:dict(VmRSS_bytes=400*2**20,VmHWM_bytes=400*2**20,VmSize_bytes=800*2**20))
    monkeypatch.setattr(ex.time,'sleep',lambda _:None)
    monkeypatch.setattr(ex,'checkpoint',lambda *a,**k:{'complete_only':True})
    monkeypatch.setattr(ex,'admission',lambda d,t,*a,**k:{'calls_attempted':ex.demand(t),'rows':[]})
    m=dict(source_commit='d'*40,proposed_resources=c.resources(),observation_policy=c.observation(),call_budget=c.call_budget())
    ledger={'known':{k:0 for k in ex.demand(t)},'reserved':{k:0 for k in ex.demand(t)}}
    done,term=ex.supervisor(tmp_path,run,'exploration',registrations,m,{},ledger)
    assert len(spawned)==len(reaped)==len(done)==8
    assert term['first_stop_reason'] is None and ledger['reserved']['trajectory']==0
    assert len(list((run/'exploration').glob('normal_promotion_*.json')))==8
    assert max(json.loads(p.read_text())['total_reserved_rss_bytes'] for p in (run/'exploration').glob('memory_admission_*.json'))<=72*2**30
    assert len({json.loads(p.read_text())['binding']['slot'] for p in (run/'exploration').glob('dispatch_*.json')})==8


def test_heavy_request_keeps_24_gib():
    t=task();tr,_,_=evidence(t)
    assert h.memory_request(t,tr,{'symmetric_directional':300000},'d'*40)['requested_rss_bytes']==24*2**30
    tr['events'][0][0]['taylor_order']=2;tr['event_digest']=c.digest(tr['events']);tr['observed_orders'][0]=2
    assert h.memory_request(t,tr,{'symmetric_directional':1000},'d'*40)['requested_rss_bytes']==24*2**30


@pytest.mark.parametrize('wrong_grant',[False,True])
def test_native_build_waits_for_exact_memory_grant(monkeypatch,tmp_path,wrong_grant):
    assert tmp_path.is_relative_to('/home/AbeHiromu')
    from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
    writer=AtomicWriter(tmp_path,byte_cap=2**20);progress=[];t=task();tr,_,_=evidence(t)
    def observer_sleep(seconds):
        assert not progress[-1]['point']=='memory_admission_granted'
        request=json.loads((tmp_path/'memory_admission_request.json').read_text())
        writer.write('memory_admission.json',dict(request_digest=c.digest(request),
            source_commit='d'*40,task_digest=c.digest(t),rss_stop_bytes=(24 if wrong_grant else 9)*2**30))
    monkeypatch.setattr(h.time,'sleep',observer_sleep)
    p=SimpleNamespace(update=lambda **kw:progress.append(kw))
    if wrong_grant:
        with pytest.raises(ValueError,match='WORKER_MEMORY_ADMISSION_BINDING'):
            h.await_memory_admission(writer,p,t,tr,{'symmetric_directional':180000},'d'*40)
    else:
        h.await_memory_admission(writer,p,t,tr,{'symmetric_directional':180000},'d'*40)
        assert progress[-1]['point']=='memory_admission_granted'
