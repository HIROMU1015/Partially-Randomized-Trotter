"""Metadata/mocks and at most four tiny children; no science arrays/transpile."""
from copy import deepcopy
import errno
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import pytest
from trottertracks.resource_applicability import h6_parallel_contract_v1 as c
from trottertracks.resource_applicability import h6_parallel_execution_v1 as e
from trottertracks.resource_applicability import h6_matched_reuse_v3 as old
from trottertracks.resource_applicability.ax2a_preparation import digest

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def no_real_arrays(monkeypatch):
    original=Path.open
    def guard(p,*a,**k):
        if str(p).endswith('.npz'):raise AssertionError('NO_SCIENCE_ARRAYS')
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'open',guard)


def candidate():
    return dict(proposed_resources=c.proposed_resources(),observation_policy=c.observation_policy(),
        base_manifest={'plan':{'caps_proposed':{'output_bytes':2*2**30,'log_bytes':2**20}}},
        registrations=[dict(index=i,task={'cell_id':'SYNTHETIC_'+str(i),'seed':i}) for i in range(450,455)],
        source_commit='a'*40,exclusive_output={'absolute_path':'/home/AbeHiromu/SYNTHETIC'},
        base_profile='fixture',approved=False,runtime_authorization=False)


def host():
    return dict(host_available_bytes=1024*2**30,effective_available_bytes=1024*2**30,
        oom={'own':{'oom':0,'oom_kill':0}},full_psi_avg10={'host':0.,'own':0.})


def test_memory_reservations_bound_slot_count():
    r=c.proposed_resources();assert c.slots(r)==4
    r['aggregate_rss_bytes']=22*2**30;assert c.slots(r)==2
    r['aggregate_rss_bytes']=13*2**30;assert c.slots(r)==1
    r['aggregate_rss_bytes']=4*2**30
    with pytest.raises(ValueError):c.slots(r)


def test_exact_142_registration_and_seed_order():
    base={'reuse_registry':{'missing':[dict(index=i,task={'seed':i},partial_event_digest='old') for i in [447,*range(450,592)]]}}
    rows=c.registrations(base);assert len(rows)==142 and rows[0]['index']==450
    assert [x['task']['seed'] for x in rows]==list(range(450,592))
    assert all(set(x)=={'index','task'} for x in rows)
    base['reuse_registry']['missing'].pop()
    with pytest.raises(ValueError):c.registrations(base)


@pytest.mark.parametrize('tamper',['draft','retry','source','cpus','old_two_worker','phase','integer_approval','valid'])
def test_new_grant_required_for_parallel_scope(tamper):
    m=candidate();m['base_manifest']['profile']={'sha256':'b'*64}
    g=c.grant_draft(m);g.update(approved_by_user=True,server_usage_approved_by_user=True)
    if tamper=='draft':g['approved_by_user']=False
    if tamper=='retry':g['retry']=True
    if tamper=='source':g['source_commit']='c'*40
    if tamper=='cpus':g['resources']['cost_cpu_sets']=[[0],[2]]
    if tamper=='old_two_worker':
        g['resources'].update(cost_workers=2,cost_cpu_sets=[[20],[21]],worker_rss_stop_bytes=18*2**30)
    if tamper=='phase':g['confirmation_authorized']=True
    if tamper=='integer_approval':g['approved_by_user']=1
    if tamper=='valid':c.validate_grant(m,g)
    else:
        with pytest.raises(ValueError):c.validate_grant(m,g)


def test_grant_resources_do_not_alias_manifest():
    m=candidate();m['base_manifest']['profile']={'sha256':'b'*64}
    g=c.grant_draft(m);g['resources']['cost_cpu_sets'][0][0]=0
    assert m['proposed_resources']['cost_cpu_sets']==[[20],[21],[24],[25]]


@pytest.mark.parametrize('tamper',['valid','unsealed','cpus','output','root','tasks','lock'])
def test_child_rechecks_scope_before_scientific_dispatch(tmp_path,tamper):
    m=candidate();m.update(schema='h6_parallel_exploration_manifest_v1',kind=c.KIND,
        execution_plan_sealed=True,allowed_cpus=[],retry=False,resume=False,
        confirmation_authorized=False,H8_authorized=False,mandatory_stop=True,
        repository_root=str(tmp_path),one_shot_lock=c.LOCK)
    m['base_manifest']['reuse_registry']={'missing':[dict(index=i,task={'seed':i}) for i in [447,*range(450,592)]]}
    m['registrations']=c.registrations(m['base_manifest'])
    out=tmp_path/c.NAMESPACE/'launch_parallel_v1'
    m['exclusive_output']=dict(repository_path=c.NAMESPACE+'launch_parallel_v1',absolute_path=str(out))
    if tamper=='unsealed':m['execution_plan_sealed']=False
    if tamper=='cpus':m['proposed_resources']['cost_cpu_sets'][0]=[0]
    if tamper=='output':out=tmp_path/'other'
    if tamper=='root':m['repository_root']=str(tmp_path/'other')
    if tamper=='tasks':m['registrations'][0]['task']['seed']+=1
    if tamper=='lock':m['one_shot_lock']='other.lock'
    if tamper=='valid':c.worker_scope(tmp_path,out,m)
    else:
        with pytest.raises(ValueError):c.worker_scope(tmp_path,out,m)


def test_dry_runner_has_no_science_import_or_launch():
    script=str(ROOT/c.RUNNER)
    code=f'import sys,runpy;sys.argv=[{script!r}];\ntry:runpy.run_path({script!r},run_name="__main__")\nexcept SystemExit as exc:assert exc.code==0\nassert not set(sys.modules)&{{"numpy","qiskit","pyscf","numba","cupy"}}'
    p=subprocess.run([sys.executable,'-I','-P','-B','-c',code],capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr


def test_missing_grant_cannot_create_output(tmp_path):
    out=tmp_path/'not_created'
    p=subprocess.run([sys.executable,'-I','-P','-B',str(ROOT/c.RUNNER),'--execute','--output',str(out)],capture_output=True,text=True,timeout=20)
    assert p.returncode!=0 and not out.exists()


def test_admission_keeps_every_frozen_check_except_expected_as():
    expected=inspect.getsource(old.admit_group).replace(
        'def admit_group(directory, task, *, source=OLD_SOURCE, compiler_hash=COMPILER_HASH):',
        'def admit_parallel_group(directory, task, *, source, compiler_hash=COMPILER_HASH):').replace('[8*2**30]*2','[32*2**30]*2')
    assert inspect.getsource(e.admit_parallel_group)==expected


@pytest.mark.parametrize('tamper',['valid','seed','event','partial','compiler','validation','limit'])
def test_32g_result_admission_matches_frozen_metadata_contract(tmp_path,tamper):
    fixture=runpy.run_path(str(ROOT/'tests/tracks/resource_applicability/test_h6_matched_v3.py'))['fixture_group']
    task,objects=fixture(tmp_path,ordinary=True)
    objects['worker_claim.json']['resource_limits']['rlimit_as']=[32*2**30]*2
    if tamper=='seed':task['seed']+=1
    if tamper=='event':objects['trajectory.json']['event_digest']='0'*64
    if tamper=='partial':objects['worker_terminal.json']['status']='TASK_STOP'
    if tamper=='compiler':objects['cost_result.json']['rows'][0]['compiler_hash']='0'*64
    if tamper=='validation':objects['validation_ordinary.json']['max_error']=1.
    if tamper=='limit':objects['worker_claim.json']['resource_limits']['rlimit_as']=[8*2**30]*2
    for name,value in objects.items():(tmp_path/name).write_text(json.dumps(value))
    if tamper=='valid':assert e.admit_parallel_group(tmp_path,task,source=old.OLD_SOURCE)['task']==task
    else:
        with pytest.raises(ValueError):e.admit_parallel_group(tmp_path,task,source=old.OLD_SOURCE)


@pytest.mark.parametrize('mode',['overlap','worker_exit','rss','worker_rss','missing','stale'])
def test_two_tiny_children_overlap_and_cleanup(tmp_path,monkeypatch,mode):
    m=candidate();m['registrations']=m['registrations'][:2]
    m['observation_policy'].update(interval_seconds=.01,journal_emit_seconds=.01)
    monkeypatch.setattr(e,'host_memory',host)
    if mode=='rss':monkeypatch.setattr(e,'process_tree_rss',lambda pid:41*2**30)
    if mode=='worker_rss':monkeypatch.setattr(e,'process_memory',lambda pid:{'VmSize_bytes':11*2**30,'VmRSS_bytes':10*2**30 if pid!=os.getpid() else 1})
    if mode=='missing':monkeypatch.setattr(e,'process_memory',lambda pid:(_ for _ in ()).throw(FileNotFoundError()))
    if mode=='stale':
        times=iter([0.,0.,6.,6.,6.,6.]);monkeypatch.setattr(e.time,'monotonic',lambda:next(times,6.))
    def command(row,directory,slot):
        code='import time,os,sys,resource;from pathlib import Path;resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(128*2**20,128*2**20));os.sched_setaffinity(0,{int(sys.argv[3])});Path(sys.argv[1]).write_text(str(os.getpid()));end=time.monotonic()+3;\nwhile not Path(sys.argv[2]).exists():\n assert time.monotonic()<end;time.sleep(.01)\ntime.sleep(.1)\n'
        if mode=='worker_exit' and slot==0:code+='raise SystemExit(1)\n'
        return [sys.executable,'-I','-P','-B','-c',code,str(tmp_path/f'ready_{slot}'),str(tmp_path/f'ready_{1-slot}'),str(20+slot)]
    report=e.supervise(ROOT,tmp_path,m,command=command)
    assert len(report['dispatched'])==2 and {x['slot'] for x in report['dispatched']}=={0,1}
    assert report['all_dispatched_reaped'] and report['log_and_journal_closed']
    assert all(not Path(f'/proc/{x["pid"]}').exists() for x in report['dispatched'])
    if mode=='overlap':
        assert report['first_stop_reason'] is None and report['completed_indices']==[450,451]
        assert (tmp_path/'ready_0').exists() and (tmp_path/'ready_1').exists()
    if mode=='worker_exit':assert report['first_stop_reason']=='WORKER_EXIT:1'
    if mode=='rss':assert report['first_stop_reason']=='AGGREGATE_RSS_CAP'
    if mode=='worker_rss':assert report['first_stop_reason']=='WORKER_RSS_CAP'
    if mode=='missing':assert 'SUPERVISOR_ERROR:FileNotFoundError' in report['first_stop_reason']
    if mode=='stale':assert report['first_stop_reason']=='OBSERVATION_STALE'


def test_cleanup_error_keeps_first_reason_and_continues_other_slot(tmp_path,monkeypatch):
    m=candidate();m['registrations']=m['registrations'][:2];calls=[];pid=iter([991,992])
    class Fake:
        def __init__(self,*a,**k):self.pid=next(pid)
    monkeypatch.setattr(e.subprocess,'Popen',Fake)
    monkeypatch.setattr(e,'process_identity',lambda p:{'start_ticks':1})
    monkeypatch.setattr(e.os,'waitid',lambda *a:None)
    monkeypatch.setattr(e,'process_memory',lambda p:{'VmSize_bytes':1,'VmRSS_bytes':1})
    monkeypatch.setattr(e,'process_tree_rss',lambda p:1)
    pressure=host();pressure['full_psi_avg10']['host']=6.
    monkeypatch.setattr(e,'host_memory',lambda:pressure)
    def retire(p,identity):
        calls.append(p.pid)
        if p.pid==991:raise OSError(errno.EPERM,'fixture')
        return 0
    monkeypatch.setattr(e,'stop_owned_group',retire)
    report=e.supervise(ROOT,tmp_path,m,command=lambda *a:['synthetic'])
    assert calls==[991,992] and report['first_stop_reason']=='MEMORY_PRESSURE:host'
    assert report['cleanup_errors'] and not report['all_dispatched_reaped']


def test_four_tiny_children_overlap_and_cleanup(tmp_path,monkeypatch):
    m=candidate();m['registrations']=m['registrations'][:4]
    m['observation_policy'].update(interval_seconds=.01,journal_emit_seconds=.01)
    monkeypatch.setattr(e,'host_memory',host)
    def command(row,directory,slot):
        # The barrier can open only while all four children are alive.
        code='import time,os,sys,resource;from pathlib import Path;resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(128*2**20,128*2**20));os.sched_setaffinity(0,{int(sys.argv[3])});root=Path(sys.argv[1]);(root/("ready_"+sys.argv[2])).write_text(str(os.getpid()));end=time.monotonic()+3;\nwhile not all((root/("ready_"+str(i))).exists() for i in range(4)):\n assert time.monotonic()<end;time.sleep(.01)\ntime.sleep(.1)\n'
        return [sys.executable,'-I','-P','-B','-c',code,str(tmp_path),str(slot),str(m['proposed_resources']['cost_cpu_sets'][slot][0])]
    report=e.supervise(ROOT,tmp_path,m,command=command)
    assert report['first_stop_reason'] is None and report['completed_indices']==[450,451,452,453]
    assert {x['slot'] for x in report['dispatched']}=={0,1,2,3}
    assert report['all_dispatched_reaped'] and report['log_and_journal_closed']
    assert all((tmp_path/f'ready_{i}').exists() for i in range(4))
    assert all(not Path(f'/proc/{x["pid"]}').exists() for x in report['dispatched'])


def test_four_worker_cpu_and_rss_partition_is_consistent():
    r=c.proposed_resources()
    worker_cpus=[x[0] for x in r['cost_cpu_sets']]
    assert len(set(worker_cpus))==4 and not set(worker_cpus)&set(r['coordinator_cpu_set'])
    assert set(worker_cpus)|set(r['coordinator_cpu_set'])==set(r['assigned_cpus'])
    assert r['worker_rss_stop_bytes']*4+r['coordinator_rss_reserve_bytes']+r['aggregate_margin_bytes']==r['aggregate_rss_bytes']
    assert r['aggregate_rss_bytes']==40*2**30 and r['job_ram_budget_bytes']==64*2**30
    assert r['cost_address_space_bytes']==32*2**30


def test_frozen_port_is_only_scientific_dispatch_and_no_confirmation():
    source=inspect.getsource(e.worker)
    assert 'from .h6_matched_cost_port_v3 import cost_task' in source
    assert 'cost_task(root,effective,b' in source
    assert not any(x in inspect.getsource(e.orchestrate) for x in ('confirmation_ids(','resource_maps(','tasks('))
    assert 'validate_plan(root,m,launch=True)' in inspect.getsource(e.orchestrate)
