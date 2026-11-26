"""Operational tests only: no PBS or scientific calculation."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
from trottertracks import rpe_optimizer_concurrent as concurrent
from trottertracks import rpe_q3_store as store
from trottertracks.rpe_preparation_artifacts import ArtifactCache


def test_nonblocking_independent_preparations_skip_full_row_reads(tmp_path):
    n=SimpleNamespace(
        run_dir=tmp_path, state=SimpleNamespace(closed=False,failure=None,tasks={}),
        independent_hchain_preparations={'independent'}, preparation_pending={}, preparation_holds={})
    n.store=Mock()
    n.store.preparation_states.return_value={'independent':('PREPARATION:compile','QUEUED'),
                                             'dependent':('PREPARATION:compile','QUEUED')}
    n.store.row.side_effect=lambda name:{'job':name}
    n._preparation_dependency_hold=lambda row:row['job']=='dependent'
    ready=concurrent.DistributedExecutor.ready_preparations(n,{'independent','dependent'},wait=False,stop_requested=lambda:False)
    assert ready==['dependent']
    n.store.row.assert_called_once_with('dependent')


def test_latest_queue_index_created_without_changing_rows(tmp_path):
    p=tmp_path/'queue.sqlite3'
    store.JobStore(p)
    with sqlite3.connect(p) as db:
        names=[row[1] for row in db.execute("pragma index_list('jobs')")]
        assert 'jobs_enqueue_order' in names
        assert db.execute('select count(*) from jobs').fetchone()==(0,)


def test_completion_gateway_rotates_even_one_free_slot(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('fair_gateway',ROOT/'scripts/rpe_pbs_gateway.py')
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    for token in ('a','b'):
        requests=tmp_path/'allocations'/token/'requests';requests.mkdir(parents=True)
        for i in range(4):
            (requests/f'complete-{token}-{i}.json').write_text('{}')
    request=dict(request_mode='completion',limit=1,bounded_results=True)
    first=g.exchange(tmp_path,request)['requests']
    second=g.exchange(tmp_path,request)['requests']
    assert list(first)==['a'] and list(second)==['b']
    batch=g.exchange(tmp_path,dict(request,limit=4))['requests']
    assert {token:len(entries) for token,entries in batch.items()}=={'a':2,'b':2}
    some=g.exchange(tmp_path,dict(request,limit=4,exclude={'a':[f'complete-a-{i}' for i in range(4)]}))['requests']
    assert list(some)==['b'] and len(some['b'])==4


def test_live_harvest_rotates_hosts_without_extra_gateway_threads():
    from trottertracks.rpe_pbs import PbsManager
    node=SimpleNamespace(settings={'request_timeout_seconds':20},live_harvest_checked={},
        allocations=[{'status':'R','token':'a','host_id':'host-a'},{'status':'R','token':'b','host_id':'host-b'}],
        harvest_pool=Mock(),_collect_live_results=Mock())
    node.harvest_pool.submit.return_value=Mock()
    owned=[dict(method='PREPARATION:compile',assigned_epoch=time.time()-100,job_id=f'{host}-{i}',
                attempt_id='attempt',state='COMPLETED_PENDING_TRANSFER',host_id=host)
           for host in ('host-a','host-b') for i in range(30)]
    PbsManager._advance_live_harvest(node,owned)
    assert node.live_harvest_pending[0]=='a'
    assert len(node.harvest_pool.submit.call_args.args[2])==16
    PbsManager._advance_live_harvest(node,owned)
    assert node.live_harvest_pending[0]=='b'
    assert len(node.harvest_pool.submit.call_args.args[2])==16


def test_completed_result_not_delayed_by_unreserved_probe_holds(tmp_path):
    job=tmp_path/'jobs/PREPARATION:compile/finished';job.mkdir(parents=True)
    (job/'OUTCOME.json').write_text('{}')
    n=SimpleNamespace(run_dir=tmp_path,state=SimpleNamespace(closed=False,failure=None,tasks={}),
        independent_hchain_preparations={'finished','pending'},preparation_pending={'pending':{}},preparation_holds={})
    n.store=Mock()
    n.store.preparation_states.return_value={'finished':('PREPARATION:compile','ACCEPTED')}
    n._preparation_probe_hold=Mock(return_value={'status':'HOST_ELIGIBILITY_HOLD'})
    assert concurrent.DistributedExecutor.ready_preparations(n,{'finished','pending'},wait=False,stop_requested=lambda:False)==['finished']
    n._preparation_probe_hold.assert_not_called()
    n.store.preparation_states.return_value={}
    assert concurrent.DistributedExecutor.ready_preparations(n,{'pending'},wait=False,stop_requested=lambda:False)==['pending']
    n._preparation_probe_hold.assert_called_once()


def test_batch_queue_snapshot_preserves_hchain_candidate_priority(tmp_path):
    import threading
    tasks={}
    for name in ('slow','fast'):
        t=SimpleNamespace(job=tmp_path/name,status='QUEUED',rule={'kind':'preparation'},
                          proposal={},demand=True,owner='same',view=SimpleNamespace(method='HCHAIN'),enqueue=1)
        tasks[t.job]=t
    n=SimpleNamespace(state=SimpleNamespace(cv=threading.Condition(),tasks=tasks,owner_clock={}),
        distributed={'controller_queue_first':True},hchain_preparation_priority={'slow':(2,), 'fast':(0,)},
        store=Mock(),pbs_manager=None,_task_ready=Mock(return_value=True))
    n.store.rows_for.return_value={s:{'state':'QUEUED','request':{}} for s in ('slow','fast')}
    n._ready_queue=lambda **kwargs:concurrent.DistributedExecutor._ready_queue(n,**kwargs)
    ready,rows=concurrent.DistributedExecutor._ready_queue_snapshot(n)
    assert [t.job.name for t in ready]==['fast','slow']
    n.store.rows_for.assert_called_once()
    assert all(c.args[1] in rows.values() for c in n._task_ready.call_args_list)


def test_advisory_pbs_demand_does_not_hold_queue_lock(tmp_path):
    from trottertracks.rpe_pbs import PbsManager
    n=SimpleNamespace(executor=SimpleNamespace(state=SimpleNamespace(cv=threading.Condition()),
        preparation_pending={'leaf':{'request':{},'failures':{},'failure_epochs':{}}},hosts={'pbs-0':{}}))
    def estimate(prep,hosts):
        assert not n.executor.state.cv._is_owned()
        assert prep==n.executor.preparation_pending and prep is not n.executor.preparation_pending
        assert hosts==('pbs-0',)
        return ('snapshot',)
    n._estimate_demand=estimate
    assert PbsManager._demand(n)==('snapshot',)
