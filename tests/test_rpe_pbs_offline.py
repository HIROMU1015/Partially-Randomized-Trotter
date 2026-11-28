"""Technical checks for the in-allocation search and chunked result import."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from trottertracks import rpe_pbs
from trottertracks.rpe_pbs_offline import OfflineSearch, prepare_plan
from trottertracks.rpe_q3_worker import atomic, read

ROOT = Path(__file__).resolve().parents[1]


def test_offline_parts_advance_and_import_only_completed_results(tmp_path, monkeypatch):
    revision = 'a'*40
    spec = {'science_approved': True, 'methods':['PR_PAULI', 'PR_DF'],
            'round_id':'fixed', 'max_new_q3_per_method':{'PR_PAULI':'UNLIMITED', 'PR_DF':'UNLIMITED'},
            'search_overrides':{'requested_widths':{'PR_PAULI':[1,2]}}, 'executor':{}}
    run = tmp_path/'central'
    run.mkdir()
    for name, value in (('ROUND.json', spec), ('PROBLEM.json', {}), ('APPLIED_SCOPE.json', {'run_id':'search-test'})):
        atomic(run/name, value)
    settings = {'code_root':str(ROOT), 'interpreter':sys.executable,
                'source':revision, 'memory_bytes_per_task':2*rpe_pbs.GIB}
    atomic(run/'ROUND.json', dict(spec, max_new_q3_per_method={'PR_PAULI':1, 'PR_DF':'UNLIMITED'}))
    assert prepare_plan(run, settings) is None
    atomic(run/'ROUND.json', spec)
    plan = prepare_plan(run, settings)
    remote = tmp_path/'remote'
    allocation = remote/'allocations'/'token'
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    commands = []

    def launch(command, **_):
        commands.append(command)
        return SimpleNamespace(poll=lambda: 0)

    profile = {'dp_workers':2, 'pbs':{'library_threads':1},
               'slot_policy':{'resource_limits':{'cpu_count':8, 'memory_bytes':12*rpe_pbs.GIB}}}
    resources = {'cpu_count':8, 'maximum_slots':8, 'memory_bytes':12*rpe_pbs.GIB}
    search = OfflineSearch(plan, profile, allocation, resources, popen=launch)
    search.validated_entry = ROOT/'scripts/run_rpe_schedule_search.py'
    assert not search.advance(disconnected=False, runnable=True, batch=8)
    assert search.advance(disconnected=True, runnable=True, batch=8)
    part = search.part
    assert read(part/'ROUND.json')['search_overrides']['requested_widths'] == {'PR_PAULI':[1]}
    assert commands[0][commands[0].index('--batch')+1] == '4'
    atomic(part/'SEARCH_TERMINATION.json', {'reason':'ARTIFICIAL_COMPLETE'})
    job = part/'jobs'/'PR_PAULI'/'saved-job'
    job.mkdir(parents=True)
    atomic(job/'REQUEST.json', {'request_id':'saved-job', 'method':'PR_PAULI', 'optimizer_source':revision})
    atomic(job/'PROCESS_RETURN.json', {'returncode':0, 'signal':None})
    atomic(job/'OUTCOME.json', {'status':'ARTIFICIAL', 'value':10**7167})
    assert search.advance(disconnected=True, runnable=True, batch=8)
    assert read(search.part/'ROUND.json')['search_overrides']['requested_widths'] == {'PR_PAULI':[2]}
    assert search.revision == 2
    assert not search.advance(disconnected=False, runnable=True, batch=8)
    assert len(commands) == 2

    spec = importlib.util.spec_from_file_location('offline_gateway_test', ROOT/'scripts/rpe_pbs_gateway.py')
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    connection = SimpleNamespace(call=lambda value: gateway.handle(remote, value), close=lambda: None)
    manager = object.__new__(rpe_pbs.PbsManager)
    manager.offline_plan, manager.settings, manager.directory = plan, {}, run/'distributed'/'pbs'
    manager.executor = SimpleNamespace(computation_source=revision)
    manager.log = lambda *_args, **_kwargs: None
    with patch.object(rpe_pbs, 'Gateway', return_value=connection):
        manager._collect_offline_history('token')
        manager._collect_offline_history('token')
    recovered = manager.directory/'offline-history'/'jobs'/'PR_PAULI'/'saved-job'
    assert (recovered/'OUTCOME.json').read_bytes() == (job/'OUTCOME.json').read_bytes()
    coverage = read(manager.directory/'offline-history'/'coverage'/'token.json')
    assert coverage['parts'][part.name]['termination'] == {'reason':'ARTIFICIAL_COMPLETE'}


def independent_fixture(root):
    from trottertracks.rpe_pbs_offline import prepare_plan
    root.mkdir(parents=True, exist_ok=True)
    problem = {'methods': {'PR_PAULI': {'L_D_values':[4,5],
                                        'generation':{'formulas':['2nd']},
                                        'absolute_kappa_values':[1,2]},
                           'FULL_PF': {'p_values':[6,7], 'formulas':['2nd']}},
               'search':{'scaled_shots':{'M':[12]}}}
    spec = dict(science_approved=True, methods=list(problem['methods']), round_id='fixed',
                search_to_rule_stop=True, max_new_q3_per_method={'PR_PAULI':7,'FULL_PF':5},
                search_overrides={'requested_widths':{'PR_PAULI':[1,2,3]},
                                  'work_conserving_search':True},
                executor={'distributed':{'preparation':{'budgets':{'signal':9,'compile':5,'audit':0}}}})
    for name,value in [('PROBLEM.json',problem),('ROUND.json',spec),
                       ('APPLIED_SCOPE.json',{'run_id':'search-artificial','scope':{}})]:
        atomic(root/name,value)
    plan=prepare_plan(root,dict(mode='independent',pbs_share='1/2',code_root=str(ROOT),
        interpreter=sys.executable,source='a'*40,memory_bytes_per_task=2*rpe_pbs.GIB))
    atomic(root/'PBS_PARTITION_PLAN.json',plan)
    return plan


def test_independent_blocks_disjoint_and_finite_budget_preserved(tmp_path):
    from trottertracks.rpe_pbs_offline import restrict_round, pbs_parts, merge_partition_maps
    plan=independent_fixture(tmp_path/'run')
    central=[p for p in plan['parts'] if p['owner']=='central']
    remote=pbs_parts(plan)
    assert len(central)+len(remote)==5
    assert not {p['id'] for p in central}&{p['id'] for p in remote}
    local=restrict_round(plan,central)
    children=[restrict_round(plan,[p]) for p in remote]
    for method,cap in plan['round']['max_new_q3_per_method'].items():
        assert sum(s['max_new_q3_per_method'].get(method,0) for s in [local,*children])==cap
    for stage,cap in plan['round']['executor']['distributed']['preparation']['budgets'].items():
        assert sum(s['executor']['distributed']['preparation']['budgets'][stage] for s in [local,*children])==cap
    full=merge_partition_maps(plan,None,{})
    assert sum(row['status']=='PBS_BLOCK_PENDING' for method in full['methods'].values() for row in method['rows'])>0
    assert all(row['owner'] in ('central','pbs') for method in full['methods'].values() for row in method['rows'])


def test_independent_keeps_running_connected_claims_disjoint_and_resumes(tmp_path, monkeypatch):
    from trottertracks.rpe_pbs_offline import pbs_parts, all_parts_finished
    plan=independent_fixture(tmp_path/'run')
    remote=tmp_path/'remote'
    durable=remote/'independent-search'
    resources={'cpu_count':8,'maximum_slots':8,'memory_bytes':12*rpe_pbs.GIB}
    commands=[]
    class Child:
        value=None
        def poll(self): return self.value
    def launch(command,**kwargs):
        child=Child()
        commands.append((command,kwargs,child))
        return child
    def make(token):
        allocation=remote/'allocations'/token; allocation.mkdir(parents=True)
        profile={'host_id':token,'dp_workers':2,'pbs':{'library_threads':1,'independent_root':str(durable)}}
        atomic(allocation/'PROFILE.json',profile)
        obj=OfflineSearch(plan,profile,allocation,resources,popen=launch)
        obj.validated_entry=ROOT/'scripts/run_rpe_schedule_search.py'
        return obj
    first,second=make('pbs1'),make('pbs2')
    assert first.advance(disconnected=False,runnable=True,batch=8)
    assert second.advance(disconnected=False,runnable=True,batch=8)
    assert first.part != second.part
    assert '--max-cycles' not in commands[0][0]
    assert commands[0][1]['pass_fds']
    assert first.advance(disconnected=True,runnable=True,batch=8)
    assert first.advance(disconnected=False,runnable=True,batch=8)
    assert not (first.part/'STOP_REQUESTED.json').exists()
    # The third allocation cannot select a block that remains owned.
    third=make('pbs3')
    assert not third.advance(disconnected=False,runnable=True,batch=8)
    old=first.part
    assert first.advance(disconnected=False,runnable=False,batch=8,stop=True)
    assert read(old/'STOP_REQUESTED.json')['reason']=='PBS_BLOCK_HANDOFF'
    commands[0][2].value=0
    assert not first.advance(disconnected=False,runnable=False,batch=8,stop=True)
    # Reopen the same persistent directory on a new allocation with the
    # same fixed scientific inputs; a temporary allocation name is irrelevant.
    replacement=make('pbs4')
    assert replacement.advance(disconnected=False,runnable=True,batch=8)
    assert replacement.part==old
    assert not (old/'STOP_REQUESTED.json').exists()
    for obj in (second,replacement):
        atomic(obj.part/'SEARCH_TERMINATION.json',{'reason':'REQUESTED_WIDTH_RULE_EXHAUSTED'})
        obj.process.value=0
        obj.advance(disconnected=False,runnable=True,batch=8)
    assert all_parts_finished(plan,read(durable/'COVERAGE.json')['parts'])


@pytest.mark.parametrize("hchain, invalid", [(False, None), (True, None), (False, "stage"), (True, "stage"), (True, "scope")])
def test_independent_preparation_export_and_exact_reuse(tmp_path, hchain, invalid):
    import threading
    import hashlib
    from trottertracks.rpe_preparation_artifacts import ArtifactCache
    from trottertracks.rpe_preparation_queue import normalize_preparation_request
    from trottertracks.rpe_q3_store import JobStore,digest
    from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
    from trottertracks.rpe_pbs_offline import pbs_parts
    run=tmp_path/'central';plan=independent_fixture(run)
    if hchain:
        plan['hchain_preparation'] = dict(owner='fixed-owner', binding={'system':'H8'},
            stages={'signal':['physical']})
        plan['round']['hchain'] = {'provider_source':{'commit':'a'*40,'files':{}}}
        atomic(run/'PBS_PARTITION_PLAN.json',plan)
    remote=tmp_path/'remote';allocation=remote/'allocations'/'token';allocation.mkdir(parents=True)
    durable=remote/'independent-search'
    profile={'host_id':'pbs0','dp_workers':2,'pbs':{'library_threads':1,'independent_root':str(durable)}}
    atomic(allocation/'PROFILE.json',profile)
    search=OfflineSearch(plan,profile,allocation,{'cpu_count':8,'maximum_slots':8,'memory_bytes':12*rpe_pbs.GIB},
                         popen=lambda *a,**k:SimpleNamespace(poll=lambda:None))
    block=pbs_parts(plan)[0]
    part=durable/'parts'/block['id'];part.mkdir(parents=True)
    request=normalize_preparation_request(dict(route={'kind':'preparation'},
        preparation={'stage':'signal','provider':'artificial','artificial':True,'source':{'commit':'a'*40,'files':{}},
                     'outputs':['result.json']}))
    if hchain:
        from run_rpe_hchain_preparation import preparation_request
        request=normalize_preparation_request(preparation_request(
            dict(owner='fixed-owner',binding={'system':'H8'},scope={'price_paths':1},
                 cases=[{'rounds':[{'id':'physical','method':'FULL'}]}]),
            'd'*64, 'signal', [['physical',0]], source={'commit':'a'*40,'files':{}}, assets={}))
    job=request['request_id'];method='PREPARATION:audit' if invalid == 'stage' else 'PREPARATION:signal'
    raw=part/'input';raw.write_bytes(b'artificial result');sha=hashlib.sha256(raw.read_bytes()).hexdigest()
    cache=ArtifactCache(part/'distributed'/'artifacts');cache.put_file(raw,sha)
    origin={'request_sha256':digest(request),'contract':{'evaluation':None,'dependencies':[]},
            'attempt_id':'attempt1','host_id':'local','completion_sha256':'f'*64}
    store=JobStore(part/'distributed'/'QUEUE.sqlite3')
    assert store.import_partition_preparation(job,method,request,origin['contract'],origin)
    jobdir=part/'jobs'/method/job;jobdir.mkdir(parents=True)
    atomic(jobdir/'REQUEST.json',request);atomic(jobdir/'PROCESS_RETURN.json',{'returncode':0,'signal':None})
    atomic(jobdir/'OUTCOME.json',{'status':'PREPARATION_COMPLETE','request_id':job,
                                  'request_sha256':digest(request),'artifacts':{'result.json':sha}})
    assert search.export(part)==1
    assert search.export(part)==0
    search.record_coverage(part)
    spec=importlib.util.spec_from_file_location('independent_gateway_test',ROOT/'scripts/rpe_pbs_gateway.py')
    gateway=importlib.util.module_from_spec(spec);spec.loader.exec_module(gateway)
    connection=SimpleNamespace(call=lambda value:gateway.handle(remote,value),close=lambda:None)
    manager=object.__new__(rpe_pbs.PbsManager)
    manager.offline_plan,manager.settings,manager.directory=plan,{},run/'distributed'/'pbs'
    manager.executor=SimpleNamespace(computation_source='a'*40,run_dir=run,
        artifact_cache=ArtifactCache(run/'distributed'/'artifacts'),state=SimpleNamespace(cv=threading.Condition()))
    manager.log=lambda *a,**k:None
    with patch.object(rpe_pbs,'Gateway',return_value=connection):
        if invalid:
            if invalid == 'scope':
                plan['hchain_preparation']['stages']['signal'] = ['different-physical-id']
            with pytest.raises(RuntimeError,match='Independent receipts pending'):
                manager._collect_offline_history('token')
            assert manager.executor.artifact_cache.missing([sha])
            assert not list((manager.directory/'offline-history/jobs').glob('*/*/IMPORTED.json'))
            return
        manager._collect_offline_history('token')
        manager._collect_offline_history('token')
    assert manager.executor.artifact_cache.path(sha).read_bytes()==b'artificial result'
    history=manager.directory/'offline-history'
    destination=JobStore(run/'distributed'/'QUEUE.sqlite3')
    ex=SimpleNamespace(prior_run_dirs=[history],run_dir=run,store=destination,
                       artifact_cache=manager.executor.artifact_cache,log=lambda **k:None)
    assert not DistributedExecutor._reuse_partition_preparation(ex,request,method,{'wrong':True})
    assert DistributedExecutor._reuse_partition_preparation(ex,request,method,origin['contract'])
    assert destination.row(job)['state']=='ACCEPTED'
    assert destination.counts(method)==0
    assert destination.preparation_reservations().get('signal',0)==0
    assert not destination.import_partition_preparation(job,method,request,origin['contract'],origin)
    assert destination.adopt(job)
    assert not destination.adopt(job)


def test_regular_width_driver_only_seeds_central_partition(tmp_path,monkeypatch):
    import argparse
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_rpe_schedule_search as driver
    from trottertracks.rpe_pbs_offline import restrict_round
    run=tmp_path/'run';plan=independent_fixture(run)
    spec=read(run/'ROUND.json')
    spec.update(live_ready_refill=True,overlap_ready_batch=True)
    spec['executor']['scheduler']='concurrent'
    spec['search_overrides']['rule']='RPE_SCALED_SHOTS_V1'
    problem=read(run/'PROBLEM.json')
    problem['search']['rule']='RPE_SCALED_SHOTS_V1'
    atomic(run/'PROBLEM.json',problem)
    atomic(run/'ROUND.json',spec)
    plan=prepare_plan(run,plan['settings'])
    atomic(run/'PBS_PARTITION_PLAN.json',plan)
    spec['release_worker_policy']={'pbs_offline_search':plan['settings']}
    atomic(run/'ROUND.json',spec)
    monkeypatch.setattr(driver.WidthDriver,'check_fixed_inputs',lambda self:None)
    monkeypatch.setattr(driver.WidthDriver,'recheck_unsupported_families',lambda self:None)
    monkeypatch.setattr(driver.WidthDriver,'write_scope_map',lambda self,*a:None)
    seen=[]
    monkeypatch.setattr(driver.WidthDriver,'seed',lambda self:seen.append((self.methods,self.widths)))
    controller=driver.WidthDriver(argparse.Namespace(run_dir=run,round=run/'ROUND.json',
                                      problem=run/'PROBLEM.json',plan_only=False))
    local=restrict_round(plan,[p for p in plan['parts'] if p['owner']=='central'])
    assert controller.spec['methods']==local['methods']
    assert [w['raw'] for w in controller.widths['PR_PAULI']]==local['search_overrides']['requested_widths']['PR_PAULI']
    assert controller.problem['methods']['FULL_PF']['p_values']==local['method_overrides']['FULL_PF']['p_values']
    assert seen
    assert read(run/'ROUND.json')['methods']==plan['round']['methods']


def test_packages_supply_multiple_widths_and_publish_global_minimum(tmp_path):
    from trottertracks.rpe_pbs_offline import pbs_units,unit_members,part_documents,merge_partition_reports
    run=tmp_path/'run';plan=independent_fixture(run)
    # One 96-CPU reservation can receive several method/width blocks at once.
    settings=dict(plan['settings'],packages_per_node=1)
    plan=prepare_plan(run,settings)
    packages=pbs_units(plan)
    assert len(packages)==1 and len(unit_members(plan,packages[0]))==2
    profile={'host_id':'pbs0','slots':24,'dp_workers':4,'pbs':{'library_threads':1}}
    path=tmp_path/'independent'/'parts'/packages[0]['id'];path.mkdir(parents=True)
    _,child=part_documents(plan,packages[0],profile,path)
    assert child['executor']['distributed']['host_eligibility'] == {'schema':'RPE_Q3_HOST_ELIGIBILITY_V1','jobs':{}}
    assert set(child['methods'])=={'PR_PAULI','FULL_PF'}
    def report(config,cost):
        best=dict(configuration_id=config,M=12,cost=cost)
        return dict(methods={'PR_PAULI':{'by_configuration':[{'configuration_id':config,'M':12,'best':best}],
                    'new_calls':1,'stop_reason':'EXPLORATORY_GRID_COMPLETE',
                    'scaled_shots':{'M12_best':best,'all_M_best':best,'comparison_only_best':None,
                                    'M12_minimum_status':'PROVISIONAL_GRID_MINIMUM',
                                    'all_M_minimum_status':'PROVISIONAL_GRID_MINIMUM'}}})
    central=report('local',20);remote={packages[0]['id']:report('remote',10)}
    merged=merge_partition_reports(plan,central,remote,{})
    assert merged['methods']['PR_PAULI']['scaled_shots']['all_M_best']['configuration_id']=='remote'
    assert len(merged['methods']['PR_PAULI']['by_configuration'])==2
    assert merged['methods']['PR_PAULI']['stop_reason']=='PBS_BLOCKS_PENDING'
    assert not merged['partitioned_search']['pbs_complete']
    coverage={packages[0]['id']:{'termination':{'reason':'REQUESTED_WIDTH_RULE_EXHAUSTED'}}}
    assert merge_partition_reports(plan,central,remote,coverage)['partitioned_search']['pbs_complete']


def test_independent_manager_starts_node_without_central_queue_and_slows_exchange(tmp_path):
    plan=independent_fixture(tmp_path/'run')
    manager=object.__new__(rpe_pbs.PbsManager)
    manager.offline_plan=plan;manager.directory=tmp_path/'manager'
    manager.state={'allocations':{}};manager.allocations=[];manager.controls={}
    manager.settings={'poll_seconds':.25};manager.configured={'pbs0':{}}
    manager.compute_storage_sampling=True;manager.storage_launcher={'path':'artificial'}
    manager.executor=SimpleNamespace(_host_admission_policy=lambda:({},None),
        _host_admission=lambda *a:{'accepting':True})
    calls=[]
    manager._submit=lambda host,worker,demand:calls.append((host,demand))
    manager.save=lambda:None;manager.log=lambda *a,**k:None
    assert manager._independent_demand(False)
    assert calls==[('pbs0',{'independent_blocks':True,'uncompleted_blocks':2})]
    manager.allocations=[{'status':'R','worker':{'independent_active':True}}]
    assert manager._poll_delay()==10
    manager.allocations[0]['worker']['independent_active']=False
    assert manager._poll_delay()==.25


def test_fixed_generation_signal_view_from_controller_profile(tmp_path, monkeypatch):
    """Relocation changes only the path used by the legacy fixed provider."""
    sys.path.insert(0, str(ROOT/'scripts'))
    import run_rpe_schedule_search as driver
    import run_rpe_optimizer_config_generator as generator

    view = tmp_path/'signal-view'
    original = {'kind':'prdf_a_fixed_grid', 'view':str(tmp_path/'generator')}
    instance = object.__new__(driver.WidthDriver)
    instance.problem = {'methods':{'PR_DF':{'generation':original}}}
    instance.spec = {'candidate_completion':{'controller_paths':{'SIGNAL_VIEW':str(view)}}}
    requests = []
    monkeypatch.setattr(driver.O, 'run_generator', lambda root, g, items:
                        (requests.append(g) or {'items':items}))
    assert instance.generate_families('PR_DF', ['example']) == ['example']
    assert requests == [dict(original, signal_view=str(view))]
    assert 'signal_view' not in original

    legacy = SimpleNamespace(VIEW=tmp_path/'old-view')
    marker = object()
    def signal_modules():
        assert legacy.VIEW == view.resolve()
        return (None, None, None, marker)
    legacy.signal_modules = signal_modules
    monkeypatch.setitem(sys.modules, 'run_pr_df_ld7_v0', legacy)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    assert generator.generator('prdf_a_fixed_grid', tmp_path, signal_view=view)[0] is marker


def test_partition_carries_parent_verified_holds(tmp_path):
    from trottertracks.rpe_pbs_offline import pbs_units, part_documents
    run = tmp_path/'run'
    initial = independent_fixture(run)
    spec = read(run/'ROUND.json')
    hold = dict(method='PR_PAULI', job_id='a'*20, reason='UNRESOLVED')
    spec['executor']['prior_q3_holds'] = [hold]
    atomic(run/'ROUND.json', spec)
    atomic(run/'DUPLICATE_PREFLIGHT.json', dict(status='PASS', prior_q3_holds=[hold]))
    plan = prepare_plan(run, dict(initial['settings'], packages_per_node=1))
    profile = dict(host_id='pbs0', slots=8, dp_workers=1, pbs={})
    _, child = part_documents(plan, pbs_units(plan)[0], profile, tmp_path/'part')
    assert 'prior_q3_holds' not in child['executor']
    assert child['executor']['partition_prior_q3_holds'] == ([hold] if 'PR_PAULI' in child['methods'] else [])


@pytest.mark.parametrize("hchain", [False, True])
def test_independent_durable_checkout_keeps_parts_with_fixed_git_source(tmp_path, hchain):
    import subprocess
    from trottertracks.rpe_pbs_offline import OfflineSearch
    source = tmp_path/'fixed'
    source.mkdir()
    subprocess.run(['git','init','-q',str(source)],check=True)
    entry = source/'scripts'/('run_rpe_method_optimizer.py' if hchain else 'run_rpe_schedule_search.py')
    entry.parent.mkdir();entry.write_text('# fixed artificial entry\n')
    old=source/'unrelated-history';old.mkdir();(old/'missing.json').write_text('historical data')
    subprocess.run(['git','-C',str(source),'add','scripts','unrelated-history'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixed'],check=True)
    revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    blob=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD:unrelated-history/missing.json'],text=True).strip()
    (source/'.git/objects'/blob[:2]/blob[2:]).unlink()
    search=object.__new__(OfflineSearch);search.directory=tmp_path/'durable';search.plan={'round':{'hchain':hchain}}
    part=search.directory/'parts'/'one';part.mkdir(parents=True);(part/'ROUND.json').write_text('{}')
    actual=search._durable_entry(source,revision)
    assert actual.read_bytes()==entry.read_bytes()
    assert part.relative_to(actual.parents[1]).as_posix()=='parts/one'
    assert (part/'ROUND.json').read_text()=='{}'
    subprocess.run(['git','-C',str(search.directory),'add','parts/one/ROUND.json'],check=True)
    subprocess.run(['git','-C',str(search.directory),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','--only','-qm','evidence','--','parts/one/ROUND.json'],check=True)
    assert search._durable_entry(source,revision)==actual


def test_fixed_input_recovery_preserves_pre_science_stop_and_refuses_owned_job(tmp_path):
    from trottertracks.rpe_pbs_offline import OfflineSearch
    search=object.__new__(OfflineSearch);search.independent=True
    root=tmp_path/'fixed';part=root/'parts'/'first';part.mkdir(parents=True)
    entry=root/'scripts'/'run_rpe_schedule_search.py'
    stop={'reason':'RUNTIME_EXCEPTION','message':'not in the subpath','error_type':'ValueError'}
    atomic(part/'KAPPA_STATE.json',{'stop':stop}); atomic(part/'ROUND.json',{'fixed':True})
    job=part/'jobs'/'PR_DF'/'owned';job.mkdir(parents=True)
    search._recover_fixed_input_failure(part,entry)
    assert (part/'KAPPA_STATE.json').exists()
    job.rmdir()
    search._recover_fixed_input_failure(part,entry)
    assert not (part/'KAPPA_STATE.json').exists()
    archive=next((part/'technical-stop-history').iterdir())
    assert read(archive/'KAPPA_STATE.json')['stop']==stop
    assert read(archive/'ROUND.json')['fixed'] is True


def test_independent_storage_recovery_refreshes_blocked_offer(tmp_path):
    from types import SimpleNamespace
    from trottertracks.rpe_pbs_offline import advance_independent
    search=SimpleNamespace(directory=tmp_path/'run',process=None,pending=lambda:True,plan={'settings':{}},
                           advance=lambda **kw: kw['runnable'])
    search.directory.mkdir(); worker_dir=tmp_path/'allocation';worker_dir.mkdir()
    worker=SimpleNamespace(offline_search=search,directory=worker_dir,records={},
         resource_report={'storage':{'accepting_new_jobs':False}},accepting=lambda:True,
         remaining=lambda:1000,drain_seconds=1,minimum_task_seconds=10,slots=4,
         _resource_offer=lambda:{'storage':{'accepting_new_jobs':True}})
    assert advance_independent(worker,{}) is True
    assert (worker_dir/'INDEPENDENT_ACTIVE').exists()

def test_controller_overlay_preserves_fixed_entry_and_checks_identity(tmp_path):
    import hashlib
    import json
    import sys
    import pytest
    from trottertracks.rpe_pbs_offline import OfflineSearch
    search=object.__new__(OfflineSearch)
    search.plan={'settings':{'source':'fixed','interpreter':sys.executable}}
    entry=tmp_path/'scripts'/'run_rpe_schedule_search.py'
    entry.parent.mkdir();entry.write_text('pass')
    modules={}
    for name in ('rpe_host_profiles','rpe_q3_resources','rpe_q3_store','rpe_preparation_bridge',
                 'rpe_q3_memory','rpe_pbs','rpe_live_q3_readiness','rpe_optimizer_concurrent'):
        path=tmp_path/(name+'.py');path.write_text('pass')
        modules[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    search.profile={'pbs':{'offline_controller_overlay':{'base_source':'fixed','modules':modules}}}
    assert search._controller_command(entry)[-3]==str(entry)
    assert [x[0] for x in json.loads(search._controller_command(entry)[-1])]==list(modules)
    modules['rpe_q3_store']['sha256']='invalid'
    with pytest.raises(ValueError,match='overlay differs'):
        search._controller_command(entry)


def test_readiness_uses_the_native_workers_fixed_base(tmp_path):
    import hashlib
    import json
    import subprocess
    from shutil import copyfile
    from trottertracks.rpe_live_q3_readiness import _HOST_PROBE

    base = tmp_path/'fixed'
    package = base/'src/trottertracks'
    package.mkdir(parents=True)
    (package/'__init__.py').touch()
    (package/'rpe_host_eligibility.py').write_text(
        'def current_host_fingerprint(*args): return {"fixed": True}\n')
    check = base/'scripts/check_rpe_q3_host_readiness.py'
    check.parent.mkdir()
    check.write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'assignment = json.loads(Path(sys.argv[2]).read_text())\n'
        'output = Path(sys.argv[6]); output.mkdir()\n'
        '(output/"READINESS.json").write_text(json.dumps(dict('
        'status="REAL_HOST_DRY_CHECK_PASS", scientific_evaluation_calls=0, '
        'request_sha256=assignment["request_sha256"])))\n')
    overlay = tmp_path/'overlay'
    script = overlay/'scripts/run_rpe_q3_worker.py'
    script.parent.mkdir(parents=True)
    copyfile(ROOT/'scripts/run_rpe_native_worker_overlay.py', script)
    modules = overlay/'src/trottertracks'
    modules.mkdir(parents=True)
    paths = {}
    for name in ('rpe_q3_timing', 'rpe_host_profiles', 'rpe_q3_resources', 'rpe_worker_storage', 'rpe_q3_worker'):
        path = modules/(name+'.py')
        path.write_text('def profile_core(profile): return profile\n'
                        'def select_host_profile(profile, assignment): return profile\n'
                        if name == 'rpe_host_profiles' else '')
        paths[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic(overlay/'NATIVE_WORKER_OVERLAY.json', dict(base_root=str(base), modules=paths))
    assignment = dict(request={'route':{'kind':'test'}}, request_sha256='fixed-request', input_hashes=[])
    report = subprocess.run([sys.executable, '-IB', '-c', _HOST_PROBE], check=True, text=True,
                            capture_output=True, input=json.dumps(dict(assignment=assignment,
                            worker_root=str(overlay), native_worker_script=str(script), profile={'root':str(base)})))
    result = json.loads(report.stdout)
    assert result['status'] == 'PASS' and result['fingerprint'] == {'fixed': True}

def test_parent_storage_sample_reused_without_directory_scan(tmp_path, monkeypatch):
    from trottertracks import rpe_worker_storage as storage
    import time
    sample=tmp_path/'STORAGE_USAGE_GATEWAY.json'
    policy={'limit_bytes':50*1024**3,'roots':[str(tmp_path)],'sample_path':str(tmp_path/'global.json'),
            'parent_compute_sample':str(sample)}
    report={'limit_bytes':policy['limit_bytes'],'used_bytes':10,'accepting_new_jobs':True,
            'measurement_seconds':15,'filesystem_min_free_bytes':0}
    monkeypatch.setattr(storage,'_used_bytes',lambda *a,**k: (_ for _ in ()).throw(AssertionError('duplicate scan')))
    atomic(sample,{'policy':storage.policy_identity(policy),'sampled_epoch':time.time(),'report':report})
    assert storage.report(policy)['accepting_new_jobs']
    atomic(sample,{'policy':storage.policy_identity(policy),'sampled_epoch':time.time()-200,'report':report})
    assert not storage.report(policy)['accepting_new_jobs']


def test_hchain_resume_selects_new_adapter_without_changing_fixed_package(tmp_path, monkeypatch):
    from copy import deepcopy
    from trottertracks import rpe_pbs_offline as offline
    plan = independent_fixture(tmp_path/'run')
    unit = offline.pbs_units(plan)[0]
    fixed = offline.restrict_round(plan, offline.unit_members(plan, unit))
    fixed['hchain'] = {'case_ids':['fixed-case']}
    monkeypatch.setattr(offline, 'restrict_round', lambda *_: deepcopy(fixed))
    plan['round']['hchain'] = {'case_ids':['fixed-case']}
    path = tmp_path/'part';path.mkdir()
    profile = dict(host_id='pbs0', slots=12, dp_workers=7, root='/fixed', pbs={})
    problem, old = offline.part_documents(plan, unit, profile, path)
    atomic(path/'PROBLEM.json', problem);atomic(path/'ROUND.json', old)
    adapter = dict(adapter_root='/updated-adapter',adapter_manifest=dict(commit='new',files={'scripts/run_rpe_optimizer_q3_job.py':'hash'}))
    profile['pbs']['offline_controller_overlay'] = dict(base_source=plan['settings']['source'], q3_adapter=adapter)
    _, new = offline.part_documents(plan, unit, profile, path)
    for name in ('local_worker','preparation_profiles'):
        target = new['executor']['distributed'][name]
        if name == 'preparation_profiles': target = target['local']
        assert all(target[k]==v for k,v in adapter.items())
        assert target['root']=='/fixed'
    search=object.__new__(offline.OfflineSearch)
    search.plan=plan;search.profile=profile;search.independent=True;search.claim=None
    search._entry=lambda:tmp_path/'entry.py';search._recover_fixed_input_failure=lambda *_:None
    search._controller_command=lambda _:['artificial'];search.environment={}
    search.popen=lambda *a,**k:SimpleNamespace()
    assert search._start_part(path,unit,1)
    assert read(path/'ROUND.json')==new
    assert read(next((path/'technical-stop-history').glob('ROUND.before-transport-*.json')))==old
    # Only the selected adapter may differ in the resumed fixed package.
    changed=deepcopy(new);changed['max_new_q3_per_method']['FULL_PF']=99
    atomic(path/'ROUND.json', changed)
    with pytest.raises(ValueError,match='part inputs changed'):
        search._start_part(path,unit,1)


def test_selected_bundled_offline_module_is_not_shadowed_by_pinned_root(tmp_path, monkeypatch):
    import zipfile
    import trottertracks
    from trottertracks import rpe_pbs_offline as fixed
    import run_rpe_pbs_storage_worker as launcher
    packed=tmp_path/'selected.pyz'
    with zipfile.ZipFile(packed,'w') as archive:
        archive.writestr('trottertracks/rpe_pbs_offline.py','RELEASE = "selected"\n')
    name='trottertracks.rpe_pbs_offline'
    monkeypatch.setitem(sys.modules,name,fixed)
    monkeypatch.setattr(trottertracks,'rpe_pbs_offline',fixed)
    result=launcher.load_bundled_offline(packed)
    assert result.RELEASE=='selected'
    assert result.__file__.startswith(str(packed))
    assert sys.modules[name] is result and trottertracks.rpe_pbs_offline is result

def test_shared_local_git_inputs_omit_bundle_only_with_visible_locators(tmp_path, monkeypatch):
    import subprocess
    from trottertracks.rpe_preparation_bridge import DistributedPreparationBridge
    root=tmp_path/'source';root.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    git('init','-q')
    (root/'fixed.json').write_text('{}')
    (root/'unrelated-history.bin').write_bytes(b'not required by any locator')
    git('add','fixed.json','unrelated-history.bin')
    git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixed')
    source=git('rev-parse','HEAD')
    missing=git('rev-parse','HEAD:unrelated-history.bin')
    (root/'.git/objects'/missing[:2]/missing[2:]).unlink()
    bridge=object.__new__(DistributedPreparationBridge)
    bridge.settings={'local_shared_git_inputs':True}
    monkeypatch.setenv('GIT_ALTERNATE_OBJECT_DIRECTORIES',str(root/'.git/objects'))
    import hashlib
    from trottertracks.rpe_preparation_worker import _verify_fixed_git_inputs
    inputs=[{'commit':source,'path':'fixed.json','sha256':hashlib.sha256(b'{}').hexdigest()}]
    worker=tmp_path/'worker';subprocess.run(['git','init','-q',str(worker)],check=True)
    _verify_fixed_git_inputs(worker,inputs)
    assert bridge._shared_local_git_inputs(source,inputs)
    assert bridge._bundle(root,source,inputs,shared_base=True)==[]
    assert not bridge._shared_local_git_inputs(source,[{'commit':source,'path':'missing'}])
    bridge.settings={}
    assert not bridge._shared_local_git_inputs(source,inputs)

def test_coverage_read_contention_keeps_last_snapshot(tmp_path, monkeypatch):
    import sqlite3
    from trottertracks.rpe_pbs_offline import OfflineSearch
    search=object.__new__(OfflineSearch);search.directory=tmp_path/'run'
    search.independent=True;search.revision=0;search.directory.mkdir()
    part=search.directory/'parts'/'one';(part/'distributed').mkdir(parents=True)
    (part/'distributed/QUEUE.sqlite3').touch()
    snapshot={'changes':0,'parts':{}}
    atomic(search.directory/'COVERAGE.json',snapshot)
    def busy(*args,**kwargs):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(sqlite3,'connect',busy)
    search.record_coverage(part)
    assert read(search.directory/'COVERAGE.json')==snapshot
    assert search.revision==0



def test_common_worker_refresh_replaces_operational_entry_and_existing_members(tmp_path):
    import zipfile
    from refresh_rpe_pbs_worker_launcher import refresh, MODULES, OPTIONAL_MODULES
    base, output = tmp_path/'base.pyz', tmp_path/'updated.pyz'
    members = {**MODULES, **OPTIONAL_MODULES}
    with zipfile.ZipFile(base,'w') as archive:
        for name in members:
            archive.writestr(name, 'OLD = True\n')
        archive.writestr('__main__.py', '# Start the fixed PBS transport with allocation-local gateway storage samples.\n')
        archive.writestr('pilot-specific.py', 'keep = True\n')
    original = base.read_bytes()
    refresh(base,output,ROOT)
    assert base.read_bytes()==original
    with zipfile.ZipFile(output) as archive:
        for name, source in members.items():
            assert archive.read(name)==(ROOT/source).read_bytes()
        assert archive.read('__main__.py')==(ROOT/'scripts/run_rpe_pbs_storage_worker.py').read_bytes()
        assert archive.read('pilot-specific.py')==b'keep = True\n'


def test_resource_receipt_is_optional_even_with_frozen_missing_validator(tmp_path, monkeypatch):
    import trottertracks.rpe_q3_resources as resources
    from trottertracks import rpe_pbs_offline as offline
    # A real completed result must still export when the old frozen transport
    # cannot validate the optional resource measurement.
    search=object.__new__(OfflineSearch);search.directory=tmp_path/'export'
    search.directory.mkdir();search.independent=True;search.revision=0
    part=tmp_path/'part';job=part/'jobs'/'PR_DF'/'completed'
    job.mkdir(parents=True)
    atomic(job/'REQUEST.json', {'request_id':'completed'})
    atomic(job/'PROCESS_RETURN.json', {'returncode':0, 'signal':None})
    atomic(job/'OUTCOME.json', {'status':'ARTIFICIAL'})
    transfer=part/'distributed/transfers/completed/attempt';transfer.mkdir(parents=True)
    atomic(transfer/'Q3_EXECUTION_RESOURCES.json', {'bad':'optional'})
    monkeypatch.setitem(sys.modules,'trottertracks._pipeline_q3_resources',None)
    monkeypatch.delattr(resources,'valid_q3_execution_receipt')
    assert search.export(part)==1
    assert search.export(part)==0
    assert read(search.directory/'jobs/PR_DF/completed/OUTCOME.json')=={'status':'ARTIFICIAL'}
    assert 'q3_execution_resources' not in read(search.directory/'RESULTS.json')['PR_DF/completed']
    assert (transfer/'Q3_EXECUTION_RESOURCES.json').is_file()


def test_refresh_includes_resource_validator_even_if_absent_in_legacy_package(tmp_path):
    import zipfile
    from refresh_rpe_pbs_worker_launcher import refresh, MODULES
    base, output=tmp_path/'base.pyz',tmp_path/'new.pyz'
    with zipfile.ZipFile(base,'w') as archive:
        for name in MODULES:archive.writestr(name,'OLD=True\n')
    refresh(base,output,ROOT)
    with zipfile.ZipFile(output) as archive:
        assert b'def valid_q3_execution_receipt' in archive.read('trottertracks/_pipeline_q3_resources.py')


def test_sampling_survives_worker_child_drain(monkeypatch):
    import threading
    import run_rpe_pbs_storage_worker as launcher
    stop=threading.Event();calls=[]
    def assert_sampling_active(flag):
        assert not flag.is_set()
    def run(command):
        assert '--worker-only' in command
        assert not stop.is_set()
        calls.append('worker exited, children draining')
        # The parent sampler remains live until the subprocess has finished
        # reaping every detached child, even on a nonzero worker return.
        assert not stop.is_set()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(launcher.subprocess,'run',run)
    monkeypatch.setattr(launcher,'become_worker_subreaper',lambda:None)
    monkeypatch.setattr(launcher,'drain_adopted_worker_children',lambda:assert_sampling_active(stop))
    assert launcher.run_sampled_worker(Path('worker.pyz'),Path('run_rpe_pbs_worker.py'),[],stop)==1
    assert calls and stop.is_set()
    waited=[]
    def waitpid(pid,flags):
        assert (pid,flags)==(-1,0)
        if waited:raise ChildProcessError
        waited.append(123);return (123,0)
    monkeypatch.setattr(launcher.os,'waitpid',waitpid)
    launcher.drain_worker_children()
    assert waited==[123]


def test_session_restart_seeds_only_identical_fresh_measurement(tmp_path):
    import time
    import run_rpe_pbs_storage_worker as launcher
    from trottertracks import rpe_worker_storage as storage
    shared,sample=tmp_path/'shared.json',tmp_path/'session.json'
    policy={'limit_bytes':100,'roots':[str(tmp_path)]}
    record=dict(policy=policy,sampled_epoch=time.time()-3,report={'used_bytes':80,'accepting_new_jobs':True})
    atomic(shared,record)
    assert launcher.seed_fresh_storage_sample(shared,sample,policy,30,storage)
    assert read(sample)==record
    sample.unlink()
    assert not launcher.seed_fresh_storage_sample(shared,sample,dict(policy,limit_bytes=90),30,storage)
    record['sampled_epoch']-=60;atomic(shared,record)
    assert not launcher.seed_fresh_storage_sample(shared,sample,policy,30,storage)
    assert not sample.exists()


def test_sampler_parent_waits_for_detached_child_after_abrupt_worker_exit(tmp_path):
    import subprocess
    import os
    if sys.platform!='linux':pytest.skip('PBS subreaper is Linux-specific')
    marker=tmp_path/'child-finished'
    fake=tmp_path/'fake_worker.py'
    fake.write_text('import os,time\nfrom pathlib import Path\n'
                    'if os.fork(): os._exit(7)\n'
                    'time.sleep(.2)\nPath('+repr(str(marker))+').write_text("done")\nos._exit(0)\n')
    # Isolate subreaper state from pytest and all other workers.
    code=('from pathlib import Path\nimport threading\n'
          'import run_rpe_pbs_storage_worker as launcher\n'
          'stop=threading.Event()\n'
          'assert launcher.run_sampled_worker(Path('+repr(str(fake))+'),Path("unused"),[],stop)==7\n'
          'assert stop.is_set() and Path('+repr(str(marker))+').read_text()=="done"\n')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts'))
    subprocess.run([sys.executable,'-B','-c',code],env=env,check=True,timeout=10)


def test_bundled_resource_receipt_export_uses_matching_validator(tmp_path, monkeypatch):
    import sqlite3
    from trottertracks import rpe_host_profiles as profiles, rpe_q3_resources as resources
    from trottertracks.rpe_q3_worker import digest
    from types import ModuleType
    # Package aliases are resolved before the fixed transport's missing symbol.
    packed_profiles=ModuleType('trottertracks._pipeline_host_profiles')
    packed_profiles.q3_cpu_measurement_key=profiles.q3_cpu_measurement_key
    packed_resources=ModuleType('trottertracks._pipeline_q3_resources')
    packed_resources.valid_q3_execution_receipt=resources.valid_q3_execution_receipt
    monkeypatch.setitem(sys.modules,packed_profiles.__name__,packed_profiles)
    monkeypatch.setitem(sys.modules,packed_resources.__name__,packed_resources)
    monkeypatch.delattr(resources,'valid_q3_execution_receipt')
    search=object.__new__(OfflineSearch);search.directory=tmp_path/'export';search.directory.mkdir()
    search.independent=True;search.revision=0;search.profile={'host_id':'pbs-test'}
    part=tmp_path/'part';job=part/'jobs'/'PR_DF'/'j';job.mkdir(parents=True)
    req=dict(method='PR_DF',request_id='j',route=dict(kind='hchain_scoped_q3',view_commit='b'*40),N_X=[4,8],N_Y=[5,8])
    runtime=dict(host_id='local',root='/fixed/runtime',interpreter='/fixed/python',dp_workers=2,task_cpu_count=2)
    measured=dict(runtime,q3_measurement_host_id='pbs-test')
    receipt=dict(returncode=0,signal=None,wall_seconds=2.)
    outcome=dict(request_id='j',status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD')
    sample=dict(schema='RPE_Q3_EXECUTION_RESOURCES_V1',key=profiles.q3_cpu_measurement_key(req,measured),
        memory_class_key=profiles.q3_cpu_measurement_key(req,measured,memory_class=True),job_id='j',attempt_id='a',
        host_id='local',policy_host_id='pbs-test',request_sha256=digest(req),method='PR_DF',route_kind='hchain_scoped_q3',
        dp_workers=2,task_cpu_count=2,samples=3,peak_rss_bytes=1024,wall_seconds=2.)
    for name,value in [('REQUEST.json',req),('PROCESS_RETURN.json',receipt),('OUTCOME.json',outcome)]:atomic(job/name,value)
    transfer=part/'distributed/transfers/j/a';transfer.mkdir(parents=True)
    atomic(transfer/'Q3_EXECUTION_RESOURCES.json',sample)
    with sqlite3.connect(part/'distributed/QUEUE.sqlite3') as db:
        db.execute('CREATE TABLE jobs(job_id,attempt_id,host_id,request_sha256,state,eligibility)')
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?)',('j','a','local',digest(req),'ACCEPTED',
            __import__('json').dumps({'execution_profiles':{'local':runtime}})))
    assert search.export(part)==1
    assert read(search.directory/'RESULTS.json')['PR_DF/j']['q3_execution_resources']==sample


def test_interrupted_export_hardlink_does_not_block_other_results(tmp_path, monkeypatch):
    import os
    from trottertracks import rpe_pbs_offline as offline
    search=object.__new__(OfflineSearch);search.directory=tmp_path/'export'
    search.directory.mkdir();search.independent=True;search.revision=0
    part=tmp_path/'part'
    for name in ('old','new','busy'):
        job=part/'jobs'/'PR_DF'/name;job.mkdir(parents=True)
        atomic(job/'REQUEST.json',{'request_id':name})
        atomic(job/'PROCESS_RETURN.json',{'returncode':0,'signal':None})
        atomic(job/'OUTCOME.json',{'status':'ARTIFICIAL'})
    src=part/'jobs/PR_DF/old/REQUEST.json';dst=search.directory/'jobs/PR_DF/old';dst.mkdir(parents=True)
    os.link(src,dst/'REQUEST.json.tmp')
    # A separate IO error must not prevent either other job from being published.
    publish=offline.publish_result_file
    def sometimes_busy(source,target):
        if source.parent.name=='busy':raise OSError('temporary IO unavailable')
        publish(source,target)
    monkeypatch.setattr(offline,'publish_result_file',sometimes_busy)
    assert search.export(part)==2
    assert src.samefile(dst/'REQUEST.json') and not (dst/'REQUEST.json.tmp').exists()
    assert set(read(search.directory/'RESULTS.json'))=={'PR_DF/old','PR_DF/new'}
    monkeypatch.setattr(offline,'publish_result_file',publish)
    assert search.export(part)==1 and search.export(part)==0


def test_independent_input_wait_is_not_relaunched_on_timer_alone(tmp_path, monkeypatch):
    plan = independent_fixture(tmp_path/'run')
    allocation = tmp_path/'remote'/'allocations'/'token'
    allocation.mkdir(parents=True)
    profile = {'host_id':'pbs0', 'dp_workers':2, 'pbs':{'library_threads':1,
        'independent_root':str(tmp_path/'remote'/'independent-search')}}
    resources = {'cpu_count':8, 'maximum_slots':8, 'memory_bytes':12*rpe_pbs.GIB}
    launches = []
    def launch(*args, **kwargs):
        launches.append(args)
        return SimpleNamespace(poll=lambda:0)
    search = OfflineSearch(plan, profile, allocation, resources, popen=launch)
    search.validated_entry = ROOT/'scripts/run_rpe_schedule_search.py'
    assert search.advance(disconnected=False, runnable=True, batch=8)
    held = search.part
    atomic(held/'KAPPA_STATE.json', {'stop':{'reason':'WAITING_FOR_INPUTS'}})
    search.advance(disconnected=False, runnable=False, batch=8)
    assert read(held/'BLOCK_STATUS.json')['status'] == 'WAITING_FOR_INPUTS'
    assert held.name in search.input_holds
    assert not (held/'SEARCH_TERMINATION.json').exists()
    search.parts = [p for p in search.parts if p['id'] == held.name]
    search.recheck_at = 0
    search.retry_after.clear()
    assert not search.advance(disconnected=False, runnable=True, batch=8)
    assert len(launches) == 1
    # A fresh keeper/worker opens the same durable block, not a timer retry.
    other = OfflineSearch(plan, profile, allocation, resources, popen=launch)
    other.validated_entry = search.validated_entry
    other.parts = search.parts
    assert not other.advance(disconnected=False, runnable=True, batch=8)
    assert len(launches) == 1
    atomic(other.directory/'CENTRAL_SHORTLIST.json', {'new_input':True})
    other.recheck_at = 0
    assert other.advance(disconnected=False, runnable=True, batch=8)
    assert len(launches) == 2


def test_preparation_budget_weighting_preserves_total_and_zero_leaf_blocks():
    from trottertracks.rpe_pbs_offline import divide_budget, hchain_preparation_weights
    parts = [dict(id='short',case_ids=['a']),dict(id='long',case_ids=['b']),dict(id='none',case_ids=[])]
    plan = dict(cases=[dict(id='a', rounds=[dict(id='one')]),
                       dict(id='b', rounds=[dict(id=str(i)) for i in range(4)])], designs=[])
    weights = hchain_preparation_weights(plan, parts)
    assert divide_budget(10, parts, weights=weights['signal']) == {'short':2,'long':8,'none':0}
    assert sum(divide_budget(7, parts, weights=weights['signal']).values()) == 7
    assert divide_budget('UNLIMITED', parts, weights=weights['signal']) == {p['id']:'UNLIMITED' for p in parts}


def test_hchain_idle_package_rotation_requires_same_revision_and_no_owners(tmp_path):
    from trottertracks.rpe_q3_store import JobStore
    search=object.__new__(OfflineSearch)
    search.independent=True; search.parts_dir=tmp_path
    search.part=tmp_path/'first';search.part.mkdir()
    search.parts=[{'id':'first'},{'id':'second'}]
    store=JobStore(search.part/'distributed/QUEUE.sqlite3')
    marker=search.part/'HCHAIN_IDLE_WAIT.json'
    atomic(marker,{'reason':'NO_RESERVED_WORK','queue_revision':store.idle_revision()})
    assert store.reserve('ARTIFICIAL','job',{'artificial':True},1,True)
    search._rotate_waiting_part()
    assert not (search.part/'STOP_REQUESTED.json').exists()
    store.cancel('ARTIFICIAL','job')
    search._rotate_waiting_part()
    assert not (search.part/'STOP_REQUESTED.json').exists()
    atomic(marker,{'reason':'NO_RESERVED_WORK','queue_revision':store.idle_revision()})
    search._rotate_waiting_part()
    assert read(search.part/'STOP_REQUESTED.json')['reason']=='PBS_BLOCK_HANDOFF'


def test_independent_successor_wait_estimate_only_counts_eligible_ready_q3(tmp_path):
    from trottertracks.rpe_pbs_offline import queued_independent_q3_seconds
    import sqlite3,json
    with sqlite3.connect(tmp_path/'fixture.sqlite3') as db:
        db.execute('CREATE TABLE jobs(method TEXT,state TEXT,request TEXT,eligibility TEXT)')
        def add(method,state,kind,host):
            db.execute('INSERT INTO jobs VALUES(?,?,?,?)',(method,state,json.dumps(
                {'route':{'kind':kind,'view_commit':'fixed'}}),
                json.dumps({'eligible_host_ids':[host]})))
        add('FULL_PF','QUEUED','hchain_scoped_q3','local')
        add('PR_DF','QUEUED','hchain_scoped_q3','different')
        add('PR_DF','RUNNING','hchain_scoped_q3','local')
        add('PREPARATION:compile','QUEUED','preparation','local')
        policy={'q3_task_seconds':[dict(method=method,route_kind='hchain_scoped_q3',
                view_commit='fixed',hosts=['pbs-fixture'],seconds=seconds)
                for method,seconds in [('FULL_PF',100),('PR_DF',200)]]}
        assert queued_independent_q3_seconds(db,policy,'pbs-fixture')==100
        assert queued_independent_q3_seconds(db,policy,'other')==0


def test_independent_successor_uses_only_fresh_matching_worker_estimate(tmp_path):
    from unittest.mock import Mock
    import time
    manager=object.__new__(rpe_pbs.PbsManager)
    manager.offline_plan=independent_fixture(tmp_path/'run')
    manager.directory=tmp_path/'manager'
    manager.settings={'managed_afterany_successor':True}
    manager.configured={'pbs0':{}}
    manager.state={'allocations':{'token':{'status':'R','spec':{'profile':{'host_id':'pbs0'}}}}}
    manager.controls={}
    manager.executor=SimpleNamespace(_host_admission_policy=lambda:({},None),
        _host_admission=lambda *a:{'accepting':True})
    manager._offline_contact=lambda *_:None
    manager.session_control=lambda *a,**k:{}
    manager._stage_managed_successor=Mock()
    manager.save=lambda:None;manager.log=lambda *_a,**_k:None
    worker=dict(independent_active=True,job_id='1.fixture',host_id='pbs0',
                independent_waiting=dict(seconds=180,epoch=time.time(),part='fixture'))
    allocation=dict(token='token',host_id='pbs0',job_id='1.fixture',status='R',worker=worker)
    manager.allocations=[allocation]
    assert manager._independent_demand(False)
    manager._stage_managed_successor.assert_called_once_with(
        allocation,manager.state['allocations']['token'],[],waiting_seconds=180)
    manager._stage_managed_successor.reset_mock()
    worker['independent_waiting']['epoch']=time.time()-1000
    assert manager._independent_demand(False)
    manager._stage_managed_successor.assert_not_called()
    worker['independent_waiting']['epoch']=time.time()
    assert manager._independent_demand(True)
    manager._stage_managed_successor.assert_not_called()


def test_independent_successor_wait_estimate_requires_ready_preparation_inputs(tmp_path):
    import sqlite3,json
    from trottertracks.rpe_pbs_offline import queued_independent_q3_seconds
    from trottertracks.rpe_preparation_artifacts import ArtifactCache
    from trottertracks.rpe_q3_worker import atomic
    sha='a'*64
    cache=ArtifactCache(tmp_path/'distributed'/'artifacts')
    request={'method':'PREPARATION:compile','route':{'kind':'preparation'},'preparation':{
        'stage':'compile','provider':'fixture','source':{'commit':'fixed'},
        'dependencies':['parent'],'dependency_inputs':{'parent':{'output':'input'}}}}
    policy={'preparation_task_seconds':[dict(stage='compile',provider='fixture',source='fixed',seconds=200)]}
    eligibility=json.dumps({'eligible_host_ids':['local']})
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE jobs(job_id TEXT,method TEXT,state TEXT,request TEXT,eligibility TEXT)')
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)',('child','PREPARATION:compile','QUEUED',json.dumps(request),eligibility))
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)',('parent','PREPARATION:signal','RUNNING','{}',eligibility))
        estimate=lambda:queued_independent_q3_seconds(db,policy,'pbs-fixture',part=tmp_path)
        assert estimate()==0
        db.execute("UPDATE jobs SET state='ACCEPTED' WHERE job_id='parent'")
        outcome=tmp_path/'jobs'/'PREPARATION:signal'/'parent'/'OUTCOME.json'
        outcome.parent.mkdir(parents=True)
        atomic(outcome,{'status':'PREPARATION_COMPLETE','artifacts':{'output':sha}})
        assert estimate()==0
        cache.path(sha).parent.mkdir(parents=True)
        cache.path(sha).touch()
        assert estimate()==200


def test_opt_in_partition_lease_fails_closed_on_missing_expired_or_future_contact():
    from trottertracks.rpe_pbs_offline import partition_contact_valid
    legacy={'settings':{}}
    plan={'settings':{'partition_start_requires_contact':True,'partition_contact_seconds':20}}
    assert partition_contact_valid(legacy,None,now=100)
    assert partition_contact_valid(plan,90,now=100)
    assert not partition_contact_valid(plan,None,now=100)
    assert not partition_contact_valid(plan,80,now=100)
    assert not partition_contact_valid(plan,101,now=100)


def test_opt_in_partition_lease_blocks_new_process_but_does_not_stop_existing_one(tmp_path):
    import time
    plan=independent_fixture(tmp_path/'run')
    plan['settings'].update(partition_start_requires_contact=True,partition_contact_seconds=30)
    allocation=tmp_path/'remote'/'allocations'/'token';allocation.mkdir(parents=True)
    profile={'host_id':'pbs0','dp_workers':2,'pbs':{'library_threads':1,
        'independent_root':str(tmp_path/'remote'/'independent-search')}}
    resources={'cpu_count':8,'maximum_slots':8,'memory_bytes':12*rpe_pbs.GIB}
    launches=[]
    def launch(*args,**kwargs):
        launches.append(args);return SimpleNamespace(poll=lambda:None)
    search=OfflineSearch(plan,profile,allocation,resources,popen=launch)
    search.validated_entry=ROOT/'scripts/run_rpe_schedule_search.py'
    assert not search.advance(disconnected=False,runnable=True,batch=8,contact=None)
    assert not launches
    assert search.advance(disconnected=False,runnable=True,batch=8,contact=time.time())
    search.next_export_at=float('inf')
    assert search.advance(disconnected=True,runnable=True,batch=8,contact=None)
    assert len(launches)==1 and not (search.part/'STOP_REQUESTED.json').exists()
    search.claim.close()


def test_unstarted_partition_fence_replays_and_worker_cannot_restart_after_contact(tmp_path):
    import time,json,hashlib,fcntl
    import rpe_pbs_gateway as gateway
    from trottertracks.rpe_pbs_offline import pbs_units,local_handoff_scope,unit_members
    plan=independent_fixture(tmp_path/'central')
    plan['settings'].update(partition_start_requires_contact=True,partition_handoff=True,partition_contact_seconds=30)
    plan['round']['executor']['distributed']['local_worker']={'host_id':'local','slots':2}
    remote=tmp_path/'remote';allocation=remote/'allocations'/'token';allocation.mkdir(parents=True)
    raw=json.dumps(plan,sort_keys=True,separators=(',',':')).encode();sha=hashlib.sha256(raw).hexdigest()
    plans=remote/'offline-plans';plans.mkdir();(plans/(sha+'.json')).write_bytes(raw)
    atomic(allocation/'PROFILE.json',{'pbs':{'offline_plan':{'sha256':sha}}})
    atomic(allocation/'STATE.json',{'partition_handoff_protocol':1})
    atomic(allocation/'CONTROL.json',{'contact':100})
    unit=pbs_units(plan)[0];request={'unit':unit['id'],'plan_sha256':sha}
    with pytest.raises(ValueError,match='not safely expired'):
        gateway.fence_unstarted_partition(remote,request,now=120)
    part=remote/'independent-search'/'parts'/unit['id']
    with (part/'OWNER.lock').open('a+b') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):gateway.fence_unstarted_partition(remote,request,now=200)
    receipt=gateway.fence_unstarted_partition(remote,request,now=200)
    assert gateway.fence_unstarted_partition(remote,request,now=201)==receipt
    _,spec=local_handoff_scope(plan,receipt)
    members=unit_members(plan,unit)
    assert sum(spec['max_new_q3_per_method'].values())==sum(p['q3_budget'] for p in members)
    assert not spec['science_approved'] and spec['executor']['distributed']['execution_hosts']==['local']
    profile={'host_id':'pbs0','dp_workers':2,'pbs':{'library_threads':1,'independent_root':str(remote/'independent-search')}}
    resources={'cpu_count':8,'maximum_slots':8,'memory_bytes':12*rpe_pbs.GIB}
    search=OfflineSearch(plan,profile,allocation,resources,popen=lambda *_a,**_kw:pytest.fail('fenced unit launched'))
    search.parts=[unit]
    assert not search.pending()
    assert not search.advance(disconnected=False,runnable=True,batch=8,contact=time.time())
    assert not read(remote/'independent-search'/'COVERAGE.json')['parts'][unit['id']].get('termination')


def test_local_handoff_completion_requires_accepted_local_receipt(tmp_path):
    """Artificial queue and files only; no fixed scientific child is launched."""
    import os
    from trottertracks.rpe_pbs_offline import (pbs_units, partition_handoff_record,
        local_handoff_scope, register_local_handoff_completion, local_handoff_coverage)
    from trottertracks.rpe_q3_store import JobStore
    plan=independent_fixture(tmp_path/'central')
    plan['settings'].update(partition_start_requires_contact=True,partition_handoff=True)
    plan['round']['executor']['distributed']['local_worker']={'host_id':'local','slots':2}
    unit=pbs_units(plan)[0]
    receipt=partition_handoff_record(plan,unit)
    problem,spec=local_handoff_scope(plan,receipt)
    root=tmp_path/'central'/'distributed'/'pbs'/'partition-handoffs';directory=root/unit['id'];directory.mkdir(parents=True)
    for name,value in [('HANDOFF.json',receipt),('PROBLEM.json',problem),('ROUND.json',spec)]:
        atomic(directory/name,value)
    assert not local_handoff_coverage(plan,root)[unit['id']].get('termination')
    child=directory/'child-run';child.mkdir()
    for name,value in [('PROBLEM.json',problem),('ROUND.json',dict(spec,science_approved=True)),
                       ('ROUND_RESULT.json',{'methods':{method:{} for method in spec['methods']}}),
                       ('SEARCH_TERMINATION.json',{'reason':'REQUESTED_WIDTH_RULE_EXHAUSTED'})]:
        atomic(child/name,value)
    (child/'distributed').mkdir()
    store=JobStore(child/'distributed'/'QUEUE.sqlite3')
    method=next(m for m in spec['methods'] if spec['max_new_q3_per_method'][m]>0)
    request={'request_id':'artificial-job','artificial':True}
    assert store.reserve(method,'artificial-job',request,1,True)
    row,=store.claim('local',1)
    with pytest.raises(ValueError,match='unconfirmed ownership'):
        register_local_handoff_completion(plan,directory)
    job=child/'jobs'/method/'artificial-job';job.mkdir(parents=True)
    stage=child/'distributed'/'transfers'/'artificial-job'/row['attempt_id'];stage.mkdir(parents=True)
    atomic(job/'REQUEST.json',request)
    for name,value in [('OUTCOME.json',{'request_id':'artificial-job'}),('PROCESS_RETURN.json',{'returncode':0})]:
        atomic(stage/name,value);os.link(stage/name,job/name)
    store.complete('local','artificial-job',row['attempt_id'],row['request_sha256'],{'artificial':'files'})
    parent=JobStore(tmp_path/'central'/'distributed'/'QUEUE.sqlite3')
    assert parent.reserve(method,'artificial-job',request,1,True)
    with pytest.raises(ValueError,match='another local owner'):
        register_local_handoff_completion(plan,directory)
    assert parent.cancel(method,'artificial-job')
    complete=register_local_handoff_completion(plan,directory)
    assert complete['counts']=={method:1}
    assert register_local_handoff_completion(plan,directory)==complete
    assert local_handoff_coverage(plan,root)[unit['id']]['termination']==complete['termination']
    atomic(child/'ROUND_RESULT.json',{'methods':{}})
    with pytest.raises(ValueError,match='report changed'):
        local_handoff_coverage(plan,root)
