"""Technical regressions for independent candidate and controller lifecycles."""
from collections import Counter, deque
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import rpe_optimizer_concurrent as C
from trottertracks import pr_candidate_completion as P
from trottertracks import rpe_preparation_bridge as B
from trottertracks.rpe_plan_history_cache import _validation_digest_bytes
from trottertracks.rpe_width_checkpoint import PointJournal, replay
from trottertracks.rpe_scope_progress import scope_progress
from trottertracks.rpe_preparation_queue import normalize_preparation_request
from trottertracks.rpe_q3_coordinator import digest
import run_rpe_schedule_search as S


def test_background_receipts_follow_exact_owner_even_outside_selected_wave(tmp_path):
    root=tmp_path/'run';root.mkdir();rows={}
    for job,owner in [('accepted-a','accepted-a'),('accepted-b','accepted-b'),('old-alias','other-owner')]:
        d=root/job;d.mkdir();(d/'DF_SIGNALS.DISTRIBUTED.json').write_text(json.dumps({'job_id':owner}))
        rows[job]={'request':{'preparation':dict(provider='DF_SIGNALS',controller_root=str(tmp_path),outputs=['workspace/run/'+job])}}
    bridge=B.DistributedPreparationBridge.__new__(B.DistributedPreparationBridge);bridge.root=tmp_path
    bridge.accepted_revision=0;bridge.import_received_stage=Mock(return_value=True)
    bridge.executor=SimpleNamespace(run_dir=root,store=SimpleNamespace(unadopted_preparation_successes=lambda:list(rows),row=rows.get),
        state=SimpleNamespace(cv=threading.Condition()),log=Mock())
    assert bridge.recover_accepted_stages(object())==['accepted-a','accepted-b']
    assert bridge.accepted_revision==2 and bridge.import_received_stage.call_count==2


def test_same_split_shortlist_lanes_do_not_wait_for_other_candidates(tmp_path,monkeypatch):
    comp=P.Completion.__new__(P.Completion);comp.settings={'exploratory_cost':True};comp.directory=tmp_path
    comp.stage_executor=SimpleNamespace(executor=SimpleNamespace(store=SimpleNamespace(row=lambda _:None)))
    comp.state={'records':{}};barrier=threading.Barrier(2);calls=[]
    for j in ('a','b'):
        folder=tmp_path/j;folder.mkdir();comp.state['records'][j]=dict(method='PR_DF',status='READY_FOR_SEARCH',prices=None,
            configuration_id=j,configuration={'fixture':j},design={'L_D':2},request={'design':{'L_D':2}},directory=str(folder))
    comp.price_registry_digest=lambda:'same';comp.df_price_evidence_snapshot=lambda record:record['configuration_id']
    comp.initial_quote=lambda *a:{'complete':False};comp.save=lambda:None
    def price(cref,*a,**kw):
        calls.append(cref['fixture']);barrier.wait(timeout=5);raise P.PreparationWait('queued',job_id=cref['fixture'])
    comp.df_price=price;monkeypatch.setattr(P,'value',lambda x:x)
    comp.prepare_shortlist_prices({'a','b'},stop_requested=lambda:False)
    assert set(calls)=={'a','b'} and all(r['shortlist_price_attempt']['reason']=='queued' for r in comp.state['records'].values())


def test_validator_ignores_only_unreferenced_stdlib_imports():
    filename = 'rpe_optimizer_concurrent.py'
    source = (ROOT/'src/trottertracks'/filename).read_bytes()
    original = _validation_digest_bytes(filename, source)
    assert original == _validation_digest_bytes(filename, source.replace(b'import hashlib\n',b'',1))
    template = b'import hashlib\nclass DistributedExecutor:\n def ingest(self):\n  return hashlib.sha256(b"input").hexdigest()\n'
    assert _validation_digest_bytes(filename, template) != _validation_digest_bytes(filename, template.replace(b'input',b'other'))
    assert _validation_digest_bytes(filename, template) != _validation_digest_bytes(filename, template.replace(b'import hashlib\n',b'import hashlib as unused\n'))


def test_independent_queue_survives_rank_change_and_explicit_prune():
    executor = C.ConcurrentExecutor.__new__(C.ConcurrentExecutor)
    executor.method = 'ARTIFICIAL'
    executor.state = SimpleNamespace(cv=threading.Condition(), tasks={})
    executor.job_dir = lambda p: Path(p['configuration_id'])
    executor._fill_standby = Mock()
    executor._enqueue_demand = Mock(return_value=True)
    executor._cancel = Mock(side_effect=lambda task: setattr(task, 'status', 'CANCELLED'))
    def task(cid, origin='INDEPENDENT_GRID'):
        p = dict(configuration_id=cid, vector=(1,2), origin=dict(kind=origin))
        t = SimpleNamespace(view=executor, job=Path(cid), owner=(executor.method,cid), demand=False,
                            status='QUEUED', rule={'kind':'artificial'}, proposal=p)
        executor.state.tasks[t.job] = t
        return t
    first = task('previous')
    conditional = task('conditional', 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES')
    running = task('running');running.status='RUNNING'
    next_p = dict(configuration_id='next', vector=(2,3))
    executor.prefetch_standby(next_p, [], retain_independent=True)
    assert first.status == 'QUEUED'
    assert conditional.status == 'CANCELLED'
    executor.retire_queued_allocations({('previous',(1,2)), ('running',(1,2))})
    assert first.status == 'CANCELLED'
    assert running.status == 'RUNNING'


def test_shared_descriptor_only_serializes_unsatisfied_prerequisite():
    comp=P.Completion.__new__(P.Completion)
    comp.saved_df_descriptor=lambda request: request.get('descriptor')
    comp.parallel_pauli_preparation=lambda: True
    base=dict(method='PR_DF',design={'L_D':4})
    assert comp.preparation_dependency_key(base)==('descriptor',4)
    assert comp.preparation_dependency_key(dict(base,descriptor='fixed')) is None
    assert comp.preparation_dependency_key(dict(method='PR_PAULI')) is None


def test_persistent_cycle_keeps_owner_and_updates_batch_budget(tmp_path):
    driver=S.WidthDriver.__new__(S.WidthDriver)
    driver.work_conserving_search=True;driver.run=tmp_path;driver.log=Mock()
    executor=SimpleNamespace(state=SimpleNamespace(cv=threading.Condition(),closed=False,failure=None,
        budgets={'A':10},by_method={'A':Counter(launched=7,adopted=6)}),
        _user_stop_requested=lambda:False,finish=Mock())
    optimize=Mock()
    driver.waiting_executor=executor;driver.waiting_optimize=optimize
    cap={'A':12}; routes={'A':['new']}; expected={'a':'fixed'}
    with patch.object(S.K.CLI, 'make_executor') as factory:
        selected, selector=driver.cycle_executor(routes, expected, {}, {'executor':{'distributed':{'port':0}}},cap)
    factory.assert_not_called();executor.finish.assert_not_called()
    assert selected is executor and selector is optimize
    assert executor.routes == routes and executor.expected == expected
    assert executor.state.budgets is cap and executor.state.retain_between_cycles
    assert executor.state.cycle_metric_baseline=={'A':dict(launched=7,adopted=6)}
    assert driver.waiting_executor is None


def test_point_journal_replays_newer_state_and_ignores_incomplete_append(tmp_path):
    path=tmp_path/'POINT_PROGRESS.jsonl'
    original=dict(method='PR_DF',family='F',kappa='1',status='COMPLETION')
    state=dict(points=[deepcopy(original)])
    j=PointJournal(path,state)
    state['points'][0].update(status='REGISTERED',configuration_id='C',
                              completion=dict(key='K',status='READY_FOR_SEARCH'))
    j.record(state['points'][0])
    old=dict(points=[deepcopy(original)])
    assert replay(old,path)==1 and old['points'][0]['configuration_id']=='C'
    with path.open('ab') as f:f.write(b'{"seq":2')
    restart=PointJournal(path,old)
    old['points'][0]['evaluation_configuration_id']='CARRIER'
    restart.record(old['points'][0])
    assert replay(state,path)==2
    assert state['points'][0]['evaluation_configuration_id']=='CARRIER'
    saved={}
    restart.checkpoint(old,lambda:saved.update(deepcopy(old)))
    assert replay(saved,path)==2 and saved['points'][0]['status']=='REGISTERED'


def test_accepted_legacy_preparation_receipt_can_resume_without_retry(tmp_path):
    stage='DF_SIGNALS'
    request=normalize_preparation_request(dict(route=dict(kind='preparation'), original_contract={},
        preparation=dict(stage='signal',provider=stage,source=dict(commit='fixed',files={}),
                         command=['python','signal'],controller_root=str(tmp_path),input_hashes=[],inputs={})))
    job=request['request_id']
    row=dict(job_id=job,method='PREPARATION:signal',request=request,request_sha256=digest(request),
             state='ACCEPTED',attempt_id='ATTEMPT',completion_sha256='accepted',detail={})
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    marker=dict(job_id=job,request=request,source='fixed')
    put(tmp_path/(stage+'.DISTRIBUTED.json'),marker)
    output=tmp_path/'jobs'/row['method']/job
    put(output/'OUTCOME.json',dict(status='PREPARATION_COMPLETE',request_id=job,request_sha256=row['request_sha256']))
    put(output/'PROCESS_RETURN.json',dict(returncode=0,signal=None))
    bridge=B.DistributedPreparationBridge(SimpleNamespace(run_dir=tmp_path,store=SimpleNamespace(row=lambda _:row)),tmp_path)
    reason=f"DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:{stage}:"+str(dict(status='HISTORICAL_INPUT_WAIT',job_id=job))
    assert bridge.recovery_evidence(tmp_path,stage,job)['action']=='IMPORT_SUCCESSFUL_STAGE'
    assert bridge.recoverable_transport(tmp_path,reason)
    (tmp_path/'STOP.json').write_text('{}')
    assert not bridge.recoverable_transport(tmp_path,reason)


def test_live_adopted_allocation_is_not_reported_as_completed_grid():
    document=dict(methods={'A':dict(rows=[dict(coordinates={'M':12},status='EVALUATION_WAITING',
                    evaluation_configuration_id='cfg')])})
    obs={'cfg':dict(q3_horizons={12:{'EVALUATION_COMPLETED':1}})}
    summary=scope_progress(document,observations=obs)
    assert summary['A']['live_counts']=={'EVALUATION_PARTIAL':1}


def make_width_fixture(names, completion):
    driver=S.WidthDriver.__new__(S.WidthDriver)
    driver.work_conserving_search=True;driver.methods=['PR_DF']
    driver.wide=dict(per_decision=4);driver.a=SimpleNamespace(plan_only=False)
    driver.spec=dict(science_approved=True,candidate_completion=dict(parallel=2))
    driver.log=Mock();driver.check_completed_configuration=lambda *args:None
    driver.state=dict(cycle=1, families={name:dict(L_D=4,requested_delta_pf={'numerator':1,'denominator':1})
                       for name in names}, points=[dict(method='PR_DF',family=name,kappa='1',
                           status='COMPLETION',completion_request=dict(family=name)) for name in names])
    driver.completion_api=lambda:completion
    return driver


def test_same_split_independent_candidates_run_concurrently():
    barrier=threading.Barrier(2)
    class Completion:
        def preparation_dependency_key(self, request):return None
        def complete(self, request):
            barrier.wait(timeout=5)
            name=request['family']
            return dict(status='READY_FOR_SEARCH',key=name,configuration_id=name)
    driver=make_width_fixture(['a','b'],Completion())
    seen=[]
    driver.complete_points(announce_ready=lambda _,record:seen.append(record['key']),continuous=True)
    assert set(seen)=={'a','b'}


def test_accepted_return_reenters_same_wave_while_peer_is_still_running():
    from contextlib import nullcontext
    peer=threading.Event()
    job='prep-'+'a'*24
    store=SimpleNamespace(_connection=lambda:nullcontext(SimpleNamespace(execute=lambda _:SimpleNamespace(fetchone=lambda:(1,)))),
        row=lambda _:dict(state='ACCEPTED',attempt_id='attempt',completion_sha256='receipt'))
    class Completion:
        stage_executor=SimpleNamespace(executor=SimpleNamespace(store=store))
        state={'records':{}}
        calls=Counter()
        def preparation_dependency_key(self, request):return None
        def recoverable_transport(self, record):return True
        def complete(self, request):
            name=request['family'];self.calls[name]+=1
            if name=='peer':
                assert peer.wait(5), 'Recovered receipt did not reenter while peer ran'
            elif self.calls[name]==1:
                record=dict(status='WAITING_DISTRIBUTED_PREPARATION',key=name,
                            preparation_recovery=dict(job_id=job),reason='old wait')
                self.state['records'][name]=record
                return record
            else:
                peer.set()
            return dict(status='READY_FOR_SEARCH',key=name,configuration_id=name)
    comp=Completion()
    driver=make_width_fixture(['recovered','peer'],comp)
    seen=[]
    driver.complete_points(announce_ready=lambda _,record:seen.append(record['key']),continuous=True)
    assert set(seen)=={'recovered','peer'}
    assert comp.calls=={'recovered':2,'peer':1}


def test_real_queue_can_dispatch_while_planner_advances_cycle(tmp_path):
    import subprocess
    from fractions import Fraction
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run=tmp_path/'run'
    route=dict(kind='artificial',interpreter=sys.executable,configuration_ids='*',
               artificial=dict(weights=[1,1],scale='1/1000',tau='1'))
    routes={'PR_PAULI':[route]}
    problem=dict(accuracy=dict(e=1),target=dict(energy_ha=0),rpe=dict(M=1))
    ex=C.DistributedExecutor(ROOT,run,routes,{},problem,optimizer_source=source,
        max_concurrent=2,budgets={'PR_PAULI':4},distributed=dict(port=0,affinity_wait_seconds=0,
        work_conserving_search=True))
    def proposal(name):
        return dict(configuration_id=name,vector=(1,1,1,1),cost=Fraction(4),origin=dict(kind='ARTIFICIAL'))
    try:
        first=ex.for_method('PR_PAULI')._enqueue(proposal('first'),demand=True)
        second=ex.for_method('PR_PAULI')._enqueue(proposal('second'),demand=True)
        poll=dict(host_id='worker',slots=1,cached_prefixes=[],source_sha=source)
        assigned,=ex.rpc('/v1/poll',poll)['assignments']
        assert assigned['job_id']==first.job.name
        driver=S.WidthDriver.__new__(S.WidthDriver)
        driver.work_conserving_search=True;driver.run=run;driver.log=Mock()
        driver.waiting_executor=ex;driver.waiting_optimize=Mock()
        choose,_=driver.cycle_executor(routes,{},problem,{'executor':{'distributed':{'port':0}}},{'PR_PAULI':6})
        assert choose is ex
        assert ex.store.row(first.job.name)['attempt_id']==assigned['attempt_id']
        assert ex.store.row(second.job.name)['state']=='QUEUED'
        third=ex.for_method('PR_PAULI')._enqueue(proposal('third'),demand=True)
        next_poll=dict(poll,host_id='other')
        another,=ex.rpc('/v1/poll',next_poll)['assignments']
        assert another['job_id'] in (second.job.name,third.job.name)
        assert another['job_id']!=assigned['job_id']
        # Repeated cycles release obsolete in-memory views, not durable jobs.
        disposable=ex.for_method('PR_PAULI')._enqueue(proposal('disposable'),demand=True)
        with ex.state.cv:
            ex._cancel(disposable)
        ex.advance_cycle()
        assert disposable.job not in ex.state.tasks
        assert ex.store.row(disposable.job.name)['state']=='CANCELLED'
        assert ex.store.row(first.job.name)['attempt_id']==assigned['attempt_id']
    finally:
        # The fixture deliberately owns no actual scientific child.
        with ex.state.cv:
            for task in ex.state.tasks.values():
                if task.status=='RUNNING':
                    task.status='DONE';task.event.set()
            ex.state.active=0
            ex.state.cv.notify_all()
        ex.finish()


def test_verified_pre_body_failure_retries_once_for_any_provider(tmp_path):
    stage='PF_AUDIT'
    request=normalize_preparation_request(dict(route=dict(kind='preparation'), original_contract={},
        preparation=dict(stage='audit',provider=stage,source=dict(commit='fixed',files={}),
                         command=['python','audit'],controller_root=str(tmp_path),input_hashes=[],inputs={})))
    job=request['request_id']
    row=dict(job_id=job,method='PREPARATION:audit',request=request,request_sha256=digest(request),
             state='ACCEPTED',attempt_id='ATTEMPT',completion_sha256='accepted',detail={'outcome_status':'PREPARATION_FAILED'})
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    marker=dict(job_id=job,request=request,source='fixed')
    put(tmp_path/(stage+'.DISTRIBUTED.json'),marker)
    output=tmp_path/'jobs'/row['method']/job
    put(output/'OUTCOME.json',dict(status='PREPARATION_FAILED',request_id=job,request_sha256=row['request_sha256'],
        artifacts={},workspace_error=dict(step='fixed workspace load',scientific_body_started=False)))
    put(output/'PROCESS_RETURN.json',dict(returncode=2,signal=None))
    bridge=B.DistributedPreparationBridge(SimpleNamespace(run_dir=tmp_path,store=SimpleNamespace(row=lambda _:row),
        _user_stop_requested=lambda:False),tmp_path,approved=True)
    replacement=bridge._transport_repair(tmp_path,stage,marker)
    assert replacement['request']['preparation']['supersedes_preparation']['job_id']==job
    assert replacement['request']['preparation']['source']==request['preparation']['source']
    assert replacement['request']['preparation']['command']==request['preparation']['command']
    (tmp_path/'STOP.json').write_text('{}')
    assert bridge._transport_repair(tmp_path,stage,marker) is None


def test_legacy_validator_cache_reuses_only_verified_original_source(tmp_path):
    from trottertracks.rpe_plan_history_cache import _same_validator
    name='rpe_optimizer_concurrent.py'
    old=tmp_path/'old'/'trottertracks'/name
    new=tmp_path/'new'/'trottertracks'/name
    old.parent.mkdir(parents=True);new.parent.mkdir(parents=True)
    source=b"import hashlib\nclass DistributedExecutor:\n def ingest(self):\n  return 1\n"
    old.write_bytes(source)
    updated=source.replace(b"import hashlib\n",b"import threading\n")
    new.write_bytes(updated)
    previous={str(old):_validation_digest_bytes(name,source,prune_imports=False)}
    current={str(new):_validation_digest_bytes(name,updated)}
    assert _same_validator(previous,current)
    new.write_bytes(updated.replace(b"return 1", b"return 2"))
    changed={str(new):_validation_digest_bytes(name,new.read_bytes())}
    assert not _same_validator(previous,changed)
    old.write_bytes(source.replace(b"return 1",b"return 3"))
    assert not _same_validator(previous,current)


def test_only_relocated_host_proof_becomes_stale():
    from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
    ex=object.__new__(DistributedExecutor)
    ex.distributed={'host_transport_paths':{'wsl':{'/old/source':'/flat/source'}}}
    old={'execution_profiles':{'wsl':{'route_views':{'commit':'/old/source'}},'local':{'root':'/old/source'}}}
    new={'execution_profiles':{'wsl':{'route_views':{'commit':'/flat/source'}}}}
    assert ex._relocated_host_profile('wsl',old)
    assert not ex._relocated_host_profile('local',old)
    assert not ex._relocated_host_profile('wsl',new)
    assert not ex._relocated_host_profile('wsl',None)
