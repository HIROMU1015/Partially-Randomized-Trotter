"""Owned kernel exit transitions; metadata fixtures and <=2 tiny real children."""
from types import SimpleNamespace
from pathlib import Path
import json
import os
import sys
import runpy
import pytest
from trottertracks.resource_applicability import h6_parallel_execution_v1 as e
from trottertracks.resource_applicability import h6_parallel_contract_v1 as c

ROOT=Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('case',['exiting','live','pid','ticks','group','session','reaped'])
def test_kernel_exit_flag_requires_all_owned_identity_fields(monkeypatch,case):
    fields=['0']*22;fields[0]='R';fields[2]='999';fields[3]='999';fields[6]='4';fields[19]='101'
    child=SimpleNamespace(pid=999,returncode=None);identity={'pid':999,'start_ticks':101}
    if case=='live':fields[6]='0'
    if case=='pid':identity['pid']=998
    if case=='ticks':fields[19]='102'
    if case=='group':fields[2]='998'
    if case=='session':fields[3]='998'
    if case=='reaped':child.returncode=0
    monkeypatch.setattr(Path,'read_text',lambda path:'999 (ARTIFICIAL command) '+' '.join(fields))
    if case in ('exiting','live'):assert e.owned_exit_flag(child,identity) is (case=='exiting')
    else:
        with pytest.raises(ValueError,match='OWNERSHIP'):e.owned_exit_flag(child,identity)


@pytest.mark.parametrize('cached',[True,False])
def test_missing_exiting_memory_is_retained_or_reserved_never_zero(cached):
    v={'process':SimpleNamespace(pid=999)};r=c.proposed_resources()
    if cached:v['last_memory']={'VmRSS_bytes':123456,'VmSize_bytes':234567}
    memory,observation=e.retain_exiting_memory(v,r,10.)
    assert memory['VmRSS_bytes']==(123456 if cached else r['worker_rss_stop_bytes'])
    assert memory['VmSize_bytes']==(234567 if cached else r['cost_address_space_bytes'])
    assert observation['live_sample'] is False and observation['kernel_exiting']
    assert v['exit_pending_since']==10.


def test_exit_deadline_is_not_extended_by_repeated_missing_samples():
    v={'process':SimpleNamespace(pid=999)};r=c.proposed_resources()
    e.retain_exiting_memory(v,r,10.)
    for now in [10.02,10.2,10.99]:e.retain_exiting_memory(v,r,now)
    assert v['exit_pending_since']==10.
    with pytest.raises(ValueError,match='TIMEOUT'):e.retain_exiting_memory(v,r,11.001)


@pytest.mark.parametrize('mode',['delayed_exit','live_missing','wrong_owner','pending_timeout','permission','invalid_value'])
def test_supervisor_handles_exit_transition_while_other_worker_is_observed(tmp_path,monkeypatch,mode):
    fixture=runpy.run_path(str(ROOT/'tests/tracks/resource_applicability/test_h6_parallel_v1.py'))
    m=fixture['candidate']();m['registrations']=m['registrations'][:2]
    m['observation_policy'].update(interval_seconds=.005,journal_emit_seconds=.005)
    monkeypatch.setattr(e,'host_memory',fixture['host'])
    real_memory=e.process_memory;real_identity=e.process_identity
    missing_pid=[];sibling_samples=[];reads=[]
    def memory(pid):
        if pid==os.getpid():return real_memory(pid)
        identity=real_identity(pid)
        if identity['state']=='Z':return real_memory(pid)
        if not missing_pid:missing_pid.append(pid)
        if pid==missing_pid[0]:
            reads.append(pid)
            if mode=='permission':raise PermissionError('ARTIFICIAL_PERMISSION')
            if mode=='invalid_value':raise ValueError('ARTIFICIAL_BAD_NUMBER')
            raise ValueError('PROCESS_MEMORY_MISSING')
        sibling_samples.append(pid);return real_memory(pid)
    monkeypatch.setattr(e,'process_memory',memory)
    def flag(child,identity):
        if mode=='wrong_owner':raise ValueError('EXIT_PENDING_OWNERSHIP_MISMATCH')
        return mode not in ('live_missing','permission','invalid_value')
    # Kernel transition is a fixture; real waitid/ownership/reap are retained.
    monkeypatch.setattr(e,'owned_exit_flag',flag)
    if mode=='pending_timeout':monkeypatch.setattr(e,'EXIT_PENDING_SECONDS',.03)
    def command(row,directory,slot):
        code='import time,os,resource,sys;resource.setrlimit(resource.RLIMIT_AS,(128*2**20,128*2**20));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,{int(sys.argv[1])});time.sleep(.25 if sys.argv[2]=="0" else .35)'
        return [sys.executable,'-I','-P','-B','-c',code,str(20+slot),str(slot)]
    report=e.supervise(ROOT,tmp_path,m,command=command)
    assert report['all_dispatched_reaped'] and not report['cleanup_errors']
    assert all(not Path('/proc/'+str(x['pid'])).exists() for x in report['dispatched'])
    if mode=='delayed_exit':
        assert report['first_stop_reason'] is None and report['completed_indices']==[450,451]
        assert len(reads)>2 and sibling_samples
        assert any(x['state']=='exit_confirmed' for x in report['exit_observation_transitions'])
        journal=[json.loads(x) for x in (tmp_path/'resource_observations.jsonl').read_text().splitlines()]
        pending=[v for row in journal for v in row['workers'] if v['memory_observation']['kind'].startswith('exit_pending')]
        assert pending and all(v['memory']['VmRSS_bytes']>0 for v in pending)
        assert all(row['observation_seconds']<m['observation_policy']['maximum_observation_seconds'] for row in journal)
    elif mode=='live_missing':assert report['first_stop_reason']=='SUPERVISOR_ERROR:ValueError:PROCESS_MEMORY_MISSING'
    elif mode=='wrong_owner':assert 'OWNERSHIP' in report['first_stop_reason']
    elif mode=='pending_timeout':assert 'EXIT_PENDING_MEMORY_TIMEOUT' in report['first_stop_reason']
    elif mode=='permission':assert 'PermissionError' in report['first_stop_reason']
    else:assert 'ARTIFICIAL_BAD_NUMBER' in report['first_stop_reason']
    if mode!='delayed_exit':
        failure=json.loads((tmp_path/'supervisor_failure.json').read_bytes())
        assert failure['observation_subject']['pid']==missing_pid[0]
        assert failure['first_stop_reason']==report['first_stop_reason']
