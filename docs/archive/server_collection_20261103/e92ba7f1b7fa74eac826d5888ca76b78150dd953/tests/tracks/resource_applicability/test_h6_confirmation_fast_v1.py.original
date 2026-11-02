"""Eight workers are mocks. No eight real workers/science arrays/transpile."""
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from trottertracks.resource_applicability import h6_confirmation_fast_contract_v1 as c
from trottertracks.resource_applicability import h6_confirmation_fast_execution_v1 as e
from trottertracks.resource_applicability import h6_confirmation_execution_v1 as old
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.h6_matched_accounting_v1 import seed

ROOT=Path(__file__).resolve().parents[3]


def manifest(root):
    regs=[dict(index=i,task=dict(cell_id='B2_SYNTHETIC',cell=dict(id='B2_SYNTHETIC',method='B2',q=1),
        phase='confirmation',replica=i,seed=seed('B2_SYNTHETIC','confirmation',i),validate=False,ordinary=False)) for i in range(32)]
    selection=dict(selected_ids=['B2_SYNTHETIC'],registrations=regs)
    reuse=dict(accepted=[dict(index=0,directory='/home/AbeHiromu/SYNTHETIC')],parent_source='a'*40)
    return dict(schema=c.SCHEMA,kind=c.KIND,repository_root=str(root),source_commit='b'*40,
        base_manifest=dict(profile={'sha256':'c'*64},plan={'caps_proposed':{'output_bytes':2*2**30,'log_bytes':2**20}}),
        exploration_binding={'selection':selection},confirmation_reuse=reuse,
        cumulative_accounting=dict(proposed_caps={'compile':1712,'trajectory':840},amendments={}),
        hash_reuse_policy=c.hash_policy(),registrations=c.remaining(selection,reuse),
        roundoff_preparation_binding={'archive_sha256':'d'*64},proposed_resources=c.proposed_resources(),
        observation_policy=c.observation_policy(),candidate_selection_sealed=True,execution_plan_sealed=True,
        approved=False,runtime_authorization=False,allowed_cpus=[],confirmation_authorized=False,
        H8_authorized=False,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,
        absolute_launch_command=None,one_shot_lock=c.LOCK,
        exclusive_output=dict(repository_path=c.NAMESPACE+'launch_confirmation_fast_v1',
            absolute_path=str(root/c.NAMESPACE/'launch_confirmation_fast_v1')))


def test_eight_distinct_slots_and_ram_headroom_math():
    r=c.proposed_resources();assert c.slots(r)==8
    workers=[x[0] for x in r['cost_cpu_sets']]
    assert len(set(workers))==8 and not set(workers)&set(r['coordinator_cpu_set'])
    assert sorted(workers+r['coordinator_cpu_set'])==r['assigned_cpus']
    assert r['worker_rss_stop_bytes']*8+r['coordinator_rss_reserve_bytes']+r['aggregate_margin_bytes']==76*2**30
    assert r['job_ram_budget_bytes']==96*2**30 and r['cost_address_space_bytes']==32*2**30
    assert c.observation_policy()['fresh_host_available_bytes']==int(248.25*2**30)
    r['aggregate_rss_bytes']=40*2**30;assert c.slots(r)==4


@pytest.mark.parametrize('change',['valid','worker4','wrong_cpu','hash_policy','seed','reuse','unsealed','output','flags'])
def test_exact_eight_slot_scope_and_complete_only_remaining(tmp_path,change):
    m=manifest(tmp_path);out=Path(m['exclusive_output']['absolute_path'])
    if change=='worker4':m['proposed_resources']['cost_workers']=4
    elif change=='wrong_cpu':m['proposed_resources']['cost_cpu_sets'][-1]=[20]
    elif change=='hash_policy':m['hash_reuse_policy']['guard_every_call']=False
    elif change=='seed':m['registrations'][0]['task']['seed']+=1
    elif change=='reuse':m['registrations'].insert(0,m['exploration_binding']['selection']['registrations'][0])
    elif change=='unsealed':m['execution_plan_sealed']=False
    elif change=='output':out=tmp_path/'other'
    elif change=='flags':m['runtime_authorization']=True
    if change=='valid':c.worker_scope(tmp_path,out,m)
    else:
        with pytest.raises(ValueError):c.worker_scope(tmp_path,out,m)


@pytest.mark.parametrize('change',['valid','draft','old_kind','worker4','reuse','hash_policy','cap','integer_approval'])
def test_old_grant_cannot_authorize_new_source_roles_hash_or_reuse(tmp_path,change):
    m=manifest(tmp_path);g=c.grant_draft(m)
    g.update(approved_by_user=True,server_usage_approved_by_user=True,confirmation_authorized=True)
    if change=='draft':g['approved_by_user']=False
    elif change=='old_kind':g['kind']='H6_TOP2_UNION_INDEPENDENT32_CONFIRMATION_ONLY_V1'
    elif change=='worker4':g['resources']['cost_workers']=4
    elif change=='reuse':g['confirmation_reuse_digest']='0'*64
    elif change=='hash_policy':g['hash_reuse_policy_digest']='0'*64
    elif change=='cap':g['proposed_cumulative_caps']['trajectory']=832
    elif change=='integer_approval':g['approved_by_user']=1
    if change=='valid':c.validate_grant(m,g)
    else:
        with pytest.raises(ValueError):c.validate_grant(m,g)


def test_partial_budget_not_refunded_and_remaining_seed_order_unchanged():
    regs=[{'index':i,'task':{'cell':{'q':1},'seed':i}} for i in range(256)]
    binding=dict(selection={'registrations':regs},carry_known_calls=dict(compile=1188,control_probe=360,trajectory=578,occurrence=2176),
        carry_reservations=dict(compile=12,control_probe=4,trajectory=6,occurrence=48))
    base={'plan':{'caps_proposed':dict(compile=1712,control_probe=1024,trajectory=832,occurrence=6656)}}
    reuse=dict(accepted=[dict(index=i) for i in range(16)],completed_known_calls=dict(compile=32,control_probe=0,trajectory=16,occurrence=16),
        partial_task_reservation=dict(compile=8,control_probe=0,trajectory=4,occurrence=4),
        parent_approved_caps=dict(compile=1712,control_probe=1024,trajectory=840,occurrence=6656))
    a=c.continuation_accounting(binding,base,reuse)
    assert a['admission_totals']['compile']==1720 and a['admission_totals']['trajectory']==844
    assert a['amendments']=={'compile':{'original':1712,'proposed':1720},'trajectory':{'original':840,'proposed':844}}
    assert a['unknown_actual_totals'] is None and not a['cap_amendment_approved'] and not a['budget_reset']
    assert [x['task']['seed'] for x in c.remaining(binding['selection'],reuse)]==list(range(16,256))


def test_monitor_is_frozen_except_new_directory_prefix():
    for name in ('owned_exit_flag','retain_exiting_memory','supervise'):
        assert inspect.getsource(getattr(e,name))==inspect.getsource(getattr(old,name)).replace('cost_confirmation_','cost_confirmation_fast_')


@pytest.mark.parametrize('mode',['normal','pressure','worker_rss','aggregate_rss','worker_as','missing','cleanup'])
def test_eight_mock_workers_dispatch_observe_and_cleanup(tmp_path,monkeypatch,mode):
    m=manifest(tmp_path);m['registrations']=m['registrations'][:16]
    m['observation_policy'].update(interval_seconds=.001,journal_emit_seconds=.001)
    created=[];retired=[];probes={};simultaneous=[0];peak=[0]
    class Fake:
        def __init__(self,*a,**k):
            self.pid=900000+len(created);self.returncode=None;created.append(self)
            simultaneous[0]+=1;peak[0]=max(peak[0],simultaneous[0])
    def waitid(kind,pid,*flags):
        probes[pid]=probes.get(pid,0)+1
        if mode=='missing':return None
        return object() if probes[pid]>1 else None
    def retire(child,identity):
        retired.append(child.pid);simultaneous[0]-=1
        if mode=='cleanup' and child.pid==900000:raise PermissionError('synthetic cleanup failure')
        return 0
    host=dict(host_available_bytes=1024*2**30,effective_available_bytes=1024*2**30,
        oom={'own':{'oom':0,'oom_kill':0}},full_psi_avg10={'host':6. if mode in ('pressure','cleanup') else 0.,'own':0.})
    def memory(pid):
        if mode=='missing' and pid!=os.getpid():raise ValueError('PROCESS_MEMORY_MISSING')
        return {'VmSize_bytes':(33 if mode=='worker_as' else 1)*2**30,
            'VmRSS_bytes':(10 if mode=='worker_rss' else 8)*2**30 if pid!=os.getpid() else 1}
    monkeypatch.setattr(e.subprocess,'Popen',Fake);monkeypatch.setattr(e.os,'waitid',waitid)
    monkeypatch.setattr(e,'process_identity',lambda pid:dict(pid=pid,start_ticks=1,pgrp=pid,session=pid))
    monkeypatch.setattr(e,'host_memory',lambda:deepcopy(host));monkeypatch.setattr(e,'process_memory',memory)
    monkeypatch.setattr(e,'process_tree_rss',lambda pid:77*2**30 if mode=='aggregate_rss' else 1)
    monkeypatch.setattr(e,'stop_owned_group',retire)
    if mode=='missing':monkeypatch.setattr(e,'owned_exit_flag',lambda *a:False)
    report=e.supervise(ROOT,tmp_path,m,command=lambda *a:['synthetic'])
    assert peak[0]==8 and set(x['slot'] for x in report['dispatched'])==set(range(8))
    assert retired==[x.pid for x in created] and len(set(retired))==len(retired)
    assert report['all_dispatched_reaped']==(mode!='cleanup')
    if mode=='normal':assert report['completed_indices']==[x['index'] for x in m['registrations']] and report['first_stop_reason'] is None
    elif mode in ('pressure','cleanup'):assert report['first_stop_reason']=='MEMORY_PRESSURE:host'
    elif mode=='worker_rss':assert report['first_stop_reason']=='WORKER_RSS_CAP'
    elif mode=='worker_as':assert report['first_stop_reason']=='WORKER_AS_CAP'
    elif mode=='aggregate_rss':assert report['first_stop_reason']=='AGGREGATE_RSS_CAP'
    else:assert report['first_stop_reason'].startswith('SUPERVISOR_ERROR:ValueError')


def test_fast_runner_is_dry_and_old_output_unused(tmp_path):
    script=str(ROOT/c.RUNNER)
    code=f'import sys,runpy;sys.argv=[{script!r}];\ntry:runpy.run_path({script!r},run_name="__main__")\nexcept SystemExit as x:assert x.code==0\nassert not set(sys.modules)&{{"numpy","qiskit","pyscf","numba","cupy"}}'
    result=subprocess.run([sys.executable,'-I','-P','-B','-c',code],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    out=tmp_path/'unused'
    result=subprocess.run([sys.executable,'-I','-P','-B',script,'--execute','--output',str(out)],capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and not out.exists()


@pytest.mark.parametrize('change',['valid','running','cleanup','worker_present','partial','wrong_seed','wrong_grant','wrong_launch','summary','one_axis'])
def test_closed_parent_complete_only_receipt_and_native_bindings(tmp_path,monkeypatch,change):
    from trottertracks.resource_applicability import h6_confirmation_contract_v1 as pc
    from trottertracks.resource_applicability.h6_matched_reuse_v3 import COMPILER_HASH
    path=tmp_path/'parent';path.mkdir();repo=tmp_path/'repo';repo.mkdir()
    m=manifest(repo);binding=deepcopy(m['exploration_binding'])
    for row in binding['selection']['registrations']:
        row['task']['cell'].update(r=1,R=1,K=2,prefix=5,formula='2nd')
    m.update(schema=pc.SCHEMA,kind=pc.KIND,source_commit=c.PARENT_CONFIRMATION_SOURCE,
        proposed_resources=c.original_resources(),exploration_binding=binding,
        registrations=binding['selection']['registrations'],source_hashes={},
        cumulative_accounting=dict(known_calls=dict(compile=1188,control_probe=360,trajectory=578,occurrence=2176,primitive=1470,reference_matvec=400),
            proposed_caps=dict(compile=1712,control_probe=1024,trajectory=840,occurrence=6656)),
        exclusive_output={'absolute_path':str(path)})
    g=pc.grant_draft(m);g.update(approved_by_user=True,server_usage_approved_by_user=True,
        trajectory_cap_amendment_approved=True,confirmation_authorized=True)
    fixed=repo/'artifacts/resource_applicability/track_a_h6_confirmation_preparation_v1/2026-10-12'
    (fixed/'sealed_v1').mkdir(parents=True);(fixed/'launch_authorization_v1').mkdir()
    (fixed/'sealed_v1/manifest_confirmation_v1.json').write_text(json.dumps(m))
    (fixed/'launch_authorization_v1/authorization_approved_v1.json').write_text(json.dumps(g))
    for name,v in [('frozen_manifest.json',m),('authorization.json',g),('authorization_source.json',g)]:
        (path/name).write_text(json.dumps(v))
    launch=dict(manifest_digest=digest(m),authorization_digest=digest(g),authorization_source_sha256=c.file_hash(path/'authorization_source.json'))
    if change=='wrong_launch':launch['manifest_digest']='0'*64
    (path/'launch_binding.json').write_text(json.dumps(launch))
    report=dict(completed_indices=[0],dispatched=[dict(index=i,pid=99980+i,task_digest=digest(m['registrations'][i]['task'])) for i in (0,1)],
        retired=[dict(index=i,pid=99980+i,exit_code=0 if i==0 else -9,reaped=True) for i in (0,1)],
        all_dispatched_reaped=True,ownership_cleanup_verified=True,cleanup_errors=[])
    if change=='cleanup':report['cleanup_errors']=['synthetic failure']
    for name,v in [('execution_claim.json',{'coordinator_identity':{'pid':99970,'start_ticks':1}}),
        ('execution_terminal.json',dict(source_commit=m['source_commit'],status='H6_CONFIRMATION_STOP',mandatory_stop=True,
            retry=False,resume=False,resource_summary=report)),('confirmation_resource_summary.json',report)]:
        (path/name).write_text(json.dumps(v))
    directory=path/'cost_confirmation_0000';directory.mkdir();task=deepcopy(m['registrations'][0]['task'])
    if change=='wrong_seed':task['seed']+=1
    events=[[{'taylor_order':0}]];event_digest=digest(events)
    rows=[dict(cell_id=task['cell_id'],phase=task['phase'],replica=0,seed=task['seed'],axis=a,control='symmetric_directional',
        event_digest=event_digest,compiler_hash=COMPILER_HASH,fingerprint='a'*64,
        metrics={k:1 for k in ('RZ','CX','RZ_depth','CX_depth','depth','size')}) for a in ('cosine','sine')]
    if change=='one_axis':rows.pop()
    objs={'worker_terminal.json':dict(status='TASK_STOP' if change=='partial' else 'TASK_COMPLETE',reason=None,task=task,source_commit=m['source_commit']),
        'worker_claim.json':dict(source_commit=m['source_commit'],task_digest=digest(task),resource_limits={'rlimit_as':[32*2**30]*2}),
        'task_binding.json':dict(task=task),'trajectory.json':dict(task=task,event_digest=event_digest,events=events,
            finite_distribution={'synthetic':True},observed_orders=[0]),
        'cost_result.json':dict(task=task,rows=rows,calls_attempted=dict(compile=2,control_probe=0,trajectory=1,occurrence=1,primitive=0,reference_matvec=0))}
    for row in rows:objs['wrapper_symmetric_directional_'+row['axis']+'.json']=row
    for name,v in objs.items():(directory/name).write_text(json.dumps(v))
    if change=='wrong_grant':
        changed=deepcopy(g);changed['resources']['cost_workers']=8;(path/'authorization.json').write_text(json.dumps(changed))
    if change=='summary':(path/'confirmation_resource_summary.json').write_text('{}')
    monkeypatch.setattr(c,'committed_hashes',lambda *a:None)
    def identity(pid):
        if change=='running' and pid==99970:return {'start_ticks':1}
        if change=='worker_present' and pid==99980:return {'start_ticks':2}
        raise FileNotFoundError()
    monkeypatch.setattr(c,'process_identity',identity)
    if change=='valid':
        receipt=c.parent_confirmation_receipt(path,binding)
        assert [x['index'] for x in receipt['accepted']]==[0] and receipt['partial_indices']==[1]
        assert receipt['completed_known_calls']['compile']==2 and receipt['partial_task_reservation']['compile']==2
        assert receipt['partial_actual_totals'] is None and not receipt['partial_reused']
        assert str(directory/'trajectory.json') in receipt['metadata_inventory']
    else:
        with pytest.raises(ValueError):c.parent_confirmation_receipt(path,binding)
