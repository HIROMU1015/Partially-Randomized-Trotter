import json
from pathlib import Path
import signal
import sqlite3
from types import SimpleNamespace
import pytest
from trottertracks import rpe_local0_search as central, rpe_pbs as pbs
from trottertracks.rpe_q3_store import digest


def synthetic_run(tmp_path):
    r = tmp_path / 'search-fixture'
    inv = 'a' * 32
    history = r / 'invocations' / inv
    history.mkdir(parents=True)
    (r/'distributed/pbs').mkdir(parents=True)
    (r/'FAST_DRIVER.lock').touch()
    (r/'distributed/pbs/MANAGER.lock').touch()
    launch = dict(invocation_id=inv, pid=2147483647)
    ended = dict(invocation_id=inv, returncode=-signal.SIGTERM)
    for name, value in {
        'ROUND.json': {'hchain': {'fixture': True}, 'executor': {'distributed': {'pbs': {'fixture': True}}}},
        'APPLIED_SCOPE.json': {'entry':'method'},
        'NORMAL_DRIVER.json': {'invocation_id':inv,'pid':2147483646},
        'distributed/ENDPOINT.json': {'pid':2147483646},
        'distributed/pbs/STATE.json': {'allocations': {'token':{'status':'R','observed':{'job_id':'123.fixture'}}}},
        'invocations/'+inv+'/LAUNCH.json':launch, 'invocations/'+inv+'/EXIT.json':ended,
    }.items():
        central.write(r/name,value)
    job = r/'jobs/PREPARATION:compile/prep-fixture'
    job.mkdir(parents=True)
    request = {'fixture':True}
    central.write(job/'REQUEST.json',request)
    central.write(job/'LAUNCH.json',{'host_id':'pbs-fixture','attempt_id':'attempt'})
    db = sqlite3.connect(r/'distributed/QUEUE.sqlite3')
    db.execute('create table jobs(job_id,method,state,host_id,attempt_id,request_sha256,request)')
    db.execute('insert into jobs values(?,?,?,?,?,?,?)',
               ('prep-fixture','PREPARATION:compile','RUNNING','pbs-fixture','attempt',digest(request),json.dumps(request)))
    db.commit();db.close()
    return r,inv,launch,ended,job


def test_live_pbs_reconnect_preserves_owner(tmp_path):
    r,inv,launch,ended,job=synthetic_run(tmp_path)
    assert central.reconcile_controller_kill(r,launch,ended,inv)['allocations']==['123.fixture']
    central.write(job/'LAUNCH.json',{'host_id':'other','attempt_id':'attempt'})
    with pytest.raises(central.RequestError,match='launch owner differs'):
        central.reconcile_controller_kill(r,launch,ended,inv)


def test_user_stop_preserved(tmp_path):
    r,inv,launch,ended,job=synthetic_run(tmp_path)
    (r/'STOP_REQUESTED.json').touch()
    with pytest.raises(central.RequestError,match='STOP_OR_SEARCH_TERMINATION'):
        central.reconcile_controller_kill(r,launch,ended,inv)


def test_old_reservation_survives_worker_detail_update(monkeypatch):
    monkeypatch.setattr(pbs,'resource_need',lambda request,profile,settings,host:
                        (1,4) if settings.get('hchain_preparation_memory') else (1,315))
    settings={'hchain_preparation_memory':{'preserve_assignments_before':100}}
    row={'request':{'preparation':{'stage':'compile'}},'state':'RUNNING','assigned_epoch':99,'detail':{'pid':1}}
    assert pbs.owned_resource_need(row,{},settings,'pbs')==(1,315)
    row['assigned_epoch']=101
    assert pbs.owned_resource_need(row,{},settings,'pbs')==(1,4)
    row['detail']={'pbs_launch_resources':{'cpu':2,'memory_bytes':17}}
    assert pbs.owned_resource_need(row,{},settings,'pbs')==(2,17)


@pytest.mark.parametrize('local_only',[False,True])
def test_hchain_sigkill_exact_owner_reconciliation(tmp_path, local_only):
    r,inv,launch,ended,job=synthetic_run(tmp_path)
    ended['returncode']=-signal.SIGKILL
    central.write(r/'invocations'/inv/'EXIT.json',ended)
    if local_only:
        spec=central.load(r/'ROUND.json')
        spec['executor']['distributed']={'local_worker':{'host_id':'local-fixture'}}
        central.write(r/'ROUND.json',spec)
        with pytest.raises(central.RequestError,match='owner outside the fixed local host'):
            central.reconcile_controller_kill(r,launch,ended,inv)
        with sqlite3.connect(r/'distributed/QUEUE.sqlite3') as db:
            db.execute("UPDATE jobs SET host_id='local-fixture'")
        central.write(job/'LAUNCH.json',{'host_id':'local-fixture','attempt_id':'attempt'})
    before=(r/'distributed/QUEUE.sqlite3').read_bytes()
    result=central.reconcile_controller_kill(r,launch,ended,inv)
    assert result['returncode']==-signal.SIGKILL
    assert result['allocations']==([] if local_only else ['123.fixture'])
    assert (r/'distributed/QUEUE.sqlite3').read_bytes()==before
    (r/'STOP_REQUESTED.json').touch()
    with pytest.raises(central.RequestError,match='STOP_OR_SEARCH_TERMINATION'):
        central.reconcile_controller_kill(r,launch,ended,inv)


def test_hchain_real_sigkill_receipt_reaches_normal_resume_without_new_attempt(tmp_path,monkeypatch):
    import sys
    r,inv,launch,ended,job=synthetic_run(tmp_path)
    state=tmp_path/'registry'
    state.mkdir()
    central.write(state/'CURRENT_TARGET.json',dict(run_id=r.name,run_directory=str(r)))
    central.write(r/'CONTROLLER_CONFIG.json',dict(state_directory=str(state)))
    central.write(r/'LAUNCH.json',launch)
    (r/'invocations'/inv/'EXIT.json').unlink()
    spec=central.load(r/'ROUND.json')
    spec['science_approved']=True  # artificial fixture only
    central.write(r/'ROUND.requested.json',spec)
    central.write(r/'APPLIED_SCOPE.json',dict(entry='method',signature='fixture'))
    central.write(r/'AUTHORIZATION.json',dict(kind='EXPLICIT_USER_START',scope_signature='fixture',authorized_utc=1))
    command=[sys.executable,'-B','-c','import os,signal; os.kill(os.getpid(),signal.SIGKILL)']
    assert central.supervise_run(r,inv,command)==-signal.SIGKILL
    killed=central.load(r/'EXIT.json')
    assert killed['observed_by']=='DETACHED_CONTROLLER_SUPERVISOR'
    gateway=object.__new__(central.Gateway)
    gateway.state=state
    before=(r/'distributed'/'QUEUE.sqlite3').read_bytes()
    called=[]
    def start(run,request,*,interrupted):
        assert request=={'action':'resume'}
        assert interrupted['from_invocation']==inv
        result=central.reconcile_controller_kill(run,launch,killed,inv)
        assert result['allocations']==['123.fixture']
        central.write(run/'AUTO_RESUME.json',dict(authorized_utc=interrupted['authorized_utc']))
        called.append(run)
        return 'ARTIFICIAL_RESUME'
    monkeypatch.setattr(gateway,'start',start)
    assert gateway.resume_interrupted(r,inv)=='ARTIFICIAL_RESUME'
    with pytest.raises(central.RequestError,match='one unattended resume'):
        gateway.resume_interrupted(r,inv)
    assert called==[r]
    assert (r/'distributed'/'QUEUE.sqlite3').read_bytes()==before
