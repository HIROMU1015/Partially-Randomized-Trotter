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
