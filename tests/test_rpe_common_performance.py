"""Synthetic operational regression, no scientific calculator is invoked."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_rpe_hchain_preparation as H
import build_rpe_q3_cpu_policy as B
from trottertracks import rpe_host_profiles as P
from trottertracks import rpe_q3_resources as R


def test_hchain_signal_results_use_reserved_workers_and_preserve_order():
    def calculate(cell):
        time.sleep(.02)
        return dict(id=cell['id'], value=cell['id']**2, pid=os.getpid())
    provider = SimpleNamespace(signal=calculate)
    cells = [dict(id=i) for i in range(8)]
    serial = list(H.signal_results(provider, cells, 1))
    parallel = list(H.signal_results(provider, cells, 3))
    assert [(a['id'],a['value']) for a in parallel] == [(a['id'],a['value']) for a in serial]
    assert all(a['pid'] != os.getpid() for a in parallel)
    assert 1 < len({a['pid'] for a in parallel}) <= 3


def test_hchain_axis_reservation_reuses_provider_slot_and_restores_it():
    old = object()
    pool = SimpleNamespace(_axis_slot=old, AXIS_RESERVED=2, AXIS_PARALLEL=3)
    def use():
        with pool._axis_slot.get_lock():
            ok = pool._axis_slot.value == pool.AXIS_RESERVED
            pool._axis_slot.value = pool.AXIS_PARALLEL
            return ok
    pool.use_spare_axis = use
    provider = SimpleNamespace(pool=pool)
    with H.price_axis_slot(provider, 1):
        assert pool._axis_slot is old
    with H.price_axis_slot(provider, 2):
        assert use()
        assert not use()
    assert pool._axis_slot is old
    with pytest.raises(ValueError, match='does not support'):
        with H.price_axis_slot(SimpleNamespace(), 2): pass


def test_hchain_cpu_environment_is_bounded(monkeypatch):
    monkeypatch.setenv('RPE_HCHAIN_PREPARATION_CPUS', '8')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: {1,2,3})
    assert H.preparation_cpus() == 3
    monkeypatch.delenv('RPE_HCHAIN_PREPARATION_CPUS')
    assert H.preparation_cpus() == 1


def _profile(host='pbs-0', cpus=8):
    return dict(host_id=host, root='/fixed', interpreter='/fixed/python',
                route_views={'b'*40:'/fixed/risk'}, dp_workers=cpus-1, task_cpu_count=cpus)

def _request(kind='hchain_scoped_q3', system='H8'):
    return dict(method='FULL_PF', route=dict(kind=kind,view_commit='b'*40,binding=dict(system=system)),
                N_X=[8,5],N_Y=[8,5],execution_profile=dict(dp_workers=None))

def _rows(key):
    return [dict(key=key,job_id=f'{cpu}-{i}',completed=True,task_cpu_count=cpu,dp_workers=cpu-1,wall_seconds=sec)
            for cpu,secs in ((8,[10,12]),(4,[12,14]),(2,[60,80])) for i,sec in enumerate(secs)]

@pytest.mark.parametrize('kind,system',[('pf_b_pf6','H4'),('hchain_scoped_q3','H6'),('hchain_scoped_q3','H8')])
def test_cpu_policy_selects_only_same_measured_profile_and_workload(kind,system):
    profile=_profile(); request=_request(kind,system); key=P.q3_cpu_measurement_key(request,profile)
    policy=dict(route_kinds=[kind],measurements={key:_rows(key)})
    d=dict(pbs=dict(workers=[dict(host_id='pbs-0',profile=profile)]))
    P.apply_pbs_q3_cpu_policies(d,{'pbs-0':policy})
    chosen=P.select_host_profile(profile,{'request':request})
    assert (chosen['task_cpu_count'],chosen['dp_workers']) == (4,3)
    assert 'q3_cpu_policy' not in chosen
    assert P.select_host_profile(chosen, {'request':request}) == chosen
    assert P.select_host_profile(profile,{'request':dict(request,N_X=[8,4])})['task_cpu_count']==8
    assert P.select_host_profile(profile,{'request':_request(kind,'other')})['task_cpu_count']==8
    assert P.select_host_profile(dict(profile,interpreter='/other'),{'request':request})['task_cpu_count']==8
    assert d['pbs_pinned_settings']['workers'][0]['profile']['task_cpu_count']==8
    assert 'q3_cpu_policy' not in d['pbs_pinned_settings']['workers'][0]['profile']


def test_no_reduction_with_unmeasured_or_duplicate_footprints():
    profile=_profile(); request=_request();key=P.q3_cpu_measurement_key(request,profile)
    rows=_rows(key); rows=[r for r in rows if r['task_cpu_count']==8] + [rows[2], rows[2]]
    assert P.measured_q3_cpu_choice(request,profile,{key:rows})==8
    assert P.measured_q3_cpu_choice(request,profile,{key:[]}) is None
    assert P.measured_q3_cpu_choice(request,dict(profile,dp_workers=1),{key:rows}) is None


def test_owned_q3_sampler_on_synthetic_process(tmp_path):
    before=R.child_cpu_seconds()
    p=subprocess.Popen([sys.executable,'-c','import time; a=bytearray(8*1024*1024); time.sleep(.35)'])
    with R.sample_q3_execution(p.pid,True,interval=.04) as sample:
        assert p.wait()==0
    req=_request();profile=_profile();a=dict(request=req,job_id='j',attempt_id='a',request_sha256='sha',pbs_launch_resources={'cpu':8})
    out=R.q3_execution_observation(a,profile,dict(returncode=0,signal=None,wall_seconds=.35),sample,before)
    assert out['key']==P.q3_cpu_measurement_key(req,profile)
    assert out['samples']>=2 and out['peak_rss_bytes']>=8*1024*1024
    assert out['task_cpu_count']==8
    assert R.q3_execution_observation(a,profile,dict(returncode=1,signal=None,wall_seconds=.35),sample,before) is None


def test_cpu_policy_builder_accepts_only_matched_successful_completed_receipts(tmp_path):
    run=tmp_path/'run';dist=run/'distributed';dist.mkdir(parents=True)
    db=sqlite3.connect(dist/'QUEUE.sqlite3');db.execute('create table jobs(job_id,attempt_id,host_id,request_sha256,request,eligibility,state,method)')
    req=_request();key=P.q3_cpu_measurement_key(req,_profile())
    for i,row in enumerate(_rows(key)):
        j,a,sha=row['job_id'],f'a{i}',f'sha{i}';p=_profile(cpus=row['task_cpu_count']);leaf=dist/'transfers'/j/a;leaf.mkdir(parents=True)
        state='ACCEPTED' if row['task_cpu_count']!=2 else 'RUNNING'
        db.execute('insert into jobs values(?,?,?,?,?,?,?,?)',(j,a,p['host_id'],sha,json.dumps(req),json.dumps(dict(execution_profiles={p['host_id']:p})),state,req['method']))
        receipt=dict(returncode=0,signal=None,wall_seconds=row['wall_seconds'])
        sample=dict(schema='RPE_Q3_EXECUTION_RESOURCES_V1',key=key,job_id=j,attempt_id=a,host_id=p['host_id'],request_sha256=sha,
                    method=req['method'],route_kind=req['route']['kind'],task_cpu_count=row['task_cpu_count'],dp_workers=row['dp_workers'],wall_seconds=row['wall_seconds'],samples=3,peak_rss_bytes=100)
        for n,v in [('Q3_EXECUTION_RESOURCES.json',sample),('PROCESS_RETURN.json',receipt),('OUTCOME.json',dict(request_id=j,status='INCONCLUSIVE'))]:
            (leaf/n).write_text(json.dumps(v))
    db.commit();db.close()
    policy=B.build_policy([run,run]);assert sorted(policy)==['pbs-0']
    assert len(policy['pbs-0']['measurements'][key])==4
    assert P.measured_q3_cpu_choice(req,_profile(),policy['pbs-0']['measurements'])==4

@pytest.mark.parametrize('system',['H6','H8'])
def test_hchain_normal_preview_applies_new_release_resources(tmp_path, system):
    from copy import deepcopy
    import hashlib
    from test_rpe_hchain_normal_entry import fixed_scope, pool, transport, HOSTS
    from trottertracks import rpe_hchain_scope
    plan,opts=fixed_scope(tmp_path);plan['scope']['system']=system;plan['binding']['system']=system
    path=Path(opts['plan']['path']);path.write_text(json.dumps(plan));opts['plan']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    source='a'*40
    settings=pool(tmp_path, preparation_task_seconds=[dict(id='signal',stage='signal',provider='HCHAIN_SCOPED',source=source,seconds=60,memory_bytes=2*1024**3)])
    for w in settings['workers']:
        profile=w['profile'];profile.update(dp_workers=7,task_cpu_count=8)
        profile['preparation_profiles']=[dict(stage='signal',provider='HCHAIN_SCOPED',source_commit=source,profile=dict(host_id=w['host_id'],root=str(ROOT),interpreter=sys.executable,task_cpu_count=1))]
    distributed=transport(settings);distributed['preparation_profiles']={w['host_id']:deepcopy(w['profile']) for w in settings['workers']}
    cfg=tmp_path/'round.json';cfg.write_text(json.dumps(dict(max_new_q3_per_method={'FULL_PF':0},executor=dict(scheduler='concurrent',distributed=distributed))))
    profile=settings['workers'][0]['profile']; req=_request(system=system);key=P.q3_cpu_measurement_key(req,profile)
    policy={'pbs_preparation_resources':[dict(host=HOSTS[0],source=source,provider='HCHAIN_SCOPED',stage='signal',task_cpu_count=4,memory_bytes=8*1024**3)],
            'pbs_q3_cpu_policies':{HOSTS[0]:dict(route_kinds=['hchain_scoped_q3'],measurements={key:_rows(key)})}}
    _,spec,_=rpe_hchain_scope.resolve(dict(hchain=opts,round=str(cfg),release_worker_policy=policy),dict(action='preview',scope=dict(system=system)))
    d=spec['executor']['distributed'];p=d['pbs']['workers'][0]['profile']
    assert p['preparation_profiles'][0]['profile']['task_cpu_count']==4
    assert d['preparation_profiles'][HOSTS[0]]['preparation_profiles'][0]['profile']['task_cpu_count']==4
    assert P.select_host_profile(p,dict(request=req))['task_cpu_count']==4
    assert distributed['pbs']['workers'][0]['profile']['preparation_profiles'][0]['profile']['task_cpu_count']==1
    assert 'q3_cpu_policy' not in distributed['pbs']['workers'][0]['profile']
    assert not spec['science_approved']


def test_h4_schedule_uses_same_cpu_policy():
    import run_rpe_schedule_search as schedule
    profile=_profile();req=_request('pf_b_pf6','H4');key=P.q3_cpu_measurement_key(req,profile)
    d=dict(pbs=dict(workers=[dict(host_id='pbs-0',profile=profile)]))
    spec=dict(executor=dict(distributed=d))
    schedule.apply_worker_trial(spec,dict(pbs_q3_cpu_policies={'pbs-0':dict(route_kinds=['pf_b_pf6'],measurements={key:_rows(key)})}))
    assert P.select_host_profile(profile,dict(request=req))['task_cpu_count']==4


def test_parallel_hchain_reservation_never_reuses_serial_memory(monkeypatch):
    from trottertracks import rpe_pbs as PBS, rpe_q3_worker
    host='pbs';source='a'*40;plan='b'*64
    req=dict(method='HCHAIN',route=dict(kind='preparation'),preparation=dict(provider='HCHAIN_SCOPED',stage='compile',source=dict(commit=source),inputs={'PLAN.json':plan}))
    settings=dict(resources=dict(mem='350gb',ncpus=96),preparation_task_seconds=[dict(stage='compile',provider='HCHAIN_SCOPED',source=source,seconds=60,memory_bytes=18*PBS.GIB)],hchain_preparation_memory=dict(source=source,stages={'compile':{plan:{'memory_bytes':4*PBS.GIB}}}),allocation_sizing={'q3_startup_memory_bytes_per_task':160*PBS.GIB})
    profile=dict(task_cpu_count=2)
    monkeypatch.setattr(rpe_q3_worker,'select_host_profile',lambda profile,a:profile)
    assert PBS.resource_need(req,profile,settings,host)==(2,18*PBS.GIB)
    assert PBS.resource_need(req,dict(task_cpu_count=1),settings,host)==(1,4*PBS.GIB)


def test_hchain_profile_reservation_is_bounded_by_actual_work():
    from trottertracks import rpe_preparation_worker as W
    profile=dict(host_id='pbs',preparation_stage='signal',preparation_provider='HCHAIN_SCOPED',task_cpu_count=8)
    req=dict(preparation=dict(provider='HCHAIN_SCOPED',stage='signal',source={'commit':'a'},command=['--targets',json.dumps([['a',0],['b',0]])]))
    assert W.select_preparation_profile(profile,{'request':req})['task_cpu_count']==2
    assert profile['task_cpu_count']==8
    price=dict(preparation=dict(req['preparation'],stage='compile'))
    assert W.select_preparation_profile(dict(profile,preparation_stage='compile'),{'request':price})['task_cpu_count']==2


def test_h4_existing_profile_without_coordinator_reservation_can_select_one_dp_cpu():
    profile=_profile(cpus=2);profile.pop('task_cpu_count');profile['dp_workers']=2
    req=_request('pf_b_pf6','H4');key=P.q3_cpu_measurement_key(req,profile)
    rows=[dict(key=key,job_id=f'{cpu}-{i}',completed=True,task_cpu_count=cpu,dp_workers=cpu,wall_seconds=sec)
          for cpu,secs in ((2,[10,12]),(1,[13,15])) for i,sec in enumerate(secs)]
    policy=dict(route_kinds=['pf_b_pf6'],measurements={key:rows})
    selected=P.select_host_profile(dict(profile,q3_cpu_policy=policy),dict(request=req))
    assert (selected['task_cpu_count'],selected['dp_workers'])==(1,1)
    # An unrelated H-chain-style extra-coordinator measurement is not evidence.
    other=[dict(row,dp_workers=row['dp_workers']-1) for row in rows]
    assert P.measured_q3_cpu_choice(req,profile,{key:other}) is None


def test_independent_local_q3_identity_preserves_original_pbs_host():
    request=_request();base=_profile(host='pbs-0')
    local=dict(base,host_id='local',q3_measurement_host_id='pbs-0')
    key=P.q3_cpu_measurement_key(request,base)
    assert P.q3_cpu_measurement_key(request,local)==key
    policy=dict(route_kinds=['hchain_scoped_q3'],measurements={key:_rows(key)})
    assert P.select_host_profile(dict(local,q3_cpu_policy=policy),dict(request=request))['task_cpu_count']==4
    assert P.q3_cpu_measurement_key(request,dict(local,q3_measurement_host_id='other'))!=key


def test_imported_independent_metadata_can_build_policy(tmp_path):
    from trottertracks.rpe_q3_worker import digest
    pbs=tmp_path/'distributed/pbs';pbs.mkdir(parents=True)
    state=dict(allocations={'token':dict(spec=dict(profile=dict(host_id='pbs-0')))})
    (pbs/'STATE.json').write_text(json.dumps(state))
    req=_request();key=P.q3_cpu_measurement_key(req,_profile())
    for i,row in enumerate(_rows(key)):
        if row['task_cpu_count']==2:continue
        job=row['job_id'];reqj=dict(req,request_id=job);leaf=pbs/'offline-history/jobs'/req['method']/job;leaf.mkdir(parents=True)
        sample=dict(schema='RPE_Q3_EXECUTION_RESOURCES_V1',key=key,job_id=job,attempt_id=f'a{i}',request_sha256=digest(reqj),
                    host_id='local',policy_host_id='pbs-0',method=req['method'],route_kind=req['route']['kind'],
                    task_cpu_count=row['task_cpu_count'],dp_workers=row['dp_workers'],wall_seconds=row['wall_seconds'],samples=3,peak_rss_bytes=100)
        for name,value in [('REQUEST.json',reqj),('OUTCOME.json',dict(request_id=job,status='INCONCLUSIVE')),
                           ('PROCESS_RETURN.json',dict(returncode=0,signal=None,wall_seconds=row['wall_seconds'])),('IMPORTED.json',dict(token='token')),
                           ('Q3_EXECUTION_RESOURCES.json',sample)]:
            (leaf/name).write_text(json.dumps(value))
    policy=B.build_policy([tmp_path]);assert len(policy['pbs-0']['measurements'][key])==4
    assert P.measured_q3_cpu_choice(req,_profile(),policy['pbs-0']['measurements'])==4


def test_independent_export_only_accepts_bound_completed_resource_sample(tmp_path,monkeypatch):
    from trottertracks.rpe_pbs_offline import OfflineSearch
    from trottertracks.rpe_q3_worker import digest
    profile=_profile(host='local'); req=dict(_request(),request_id='j')
    monkeypatch.setenv('RPE_Q3_MEASUREMENT_HOST_ID','pbs-0')
    key=P.q3_cpu_measurement_key(req,profile);assert key==P.q3_cpu_measurement_key(req,_profile(host='pbs-0'))
    d=tmp_path/'distributed';leaf=d/'transfers/j/a';leaf.mkdir(parents=True)
    db=sqlite3.connect(d/'QUEUE.sqlite3');db.execute('create table jobs(job_id,attempt_id,host_id,request_sha256,state,eligibility)')
    db.execute('insert into jobs values(?,?,?,?,?,?)',('j','a','local',digest(req),'ACCEPTED',json.dumps(dict(execution_profiles={'local':profile}))))
    db.commit();db.close()
    sample=dict(schema='RPE_Q3_EXECUTION_RESOURCES_V1',key=key,job_id='j',attempt_id='a',request_sha256=digest(req),host_id='local',policy_host_id='pbs-0',method=req['method'],route_kind=req['route']['kind'],task_cpu_count=8,dp_workers=7,wall_seconds=1.,samples=3,peak_rss_bytes=100)
    (leaf/'Q3_EXECUTION_RESOURCES.json').write_text(json.dumps(sample))
    receipt=dict(returncode=0,signal=None,wall_seconds=1.);outcome=dict(request_id='j',status='INCONCLUSIVE')
    owner=SimpleNamespace(profile={'host_id':'pbs-0'})
    assert OfflineSearch._q3_resource_receipt(owner,tmp_path,'j',req,receipt,outcome)==sample
    assert OfflineSearch._q3_resource_receipt(SimpleNamespace(profile={'host_id':'other'}),tmp_path,'j',req,receipt,outcome) is None
