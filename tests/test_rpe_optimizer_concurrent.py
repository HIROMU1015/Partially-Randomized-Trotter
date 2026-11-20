"""Artificial workers only; do not import/execute any H4 scientific Q3 leaf."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import ast
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_optimizer_concurrent as C
from trottertracks import rpe_q3_coordinator as Q
from trottertracks import rpe_scaled_shots as S

METHODS=['FULL_PF','TRUNCATED_PF','PR_PAULI','PR_DF']


def baseline():
    module=types.ModuleType('trottertracks._throughput_baseline');module.__package__='trottertracks'
    text=subprocess.check_output(['git','show','1e88cd85cc8d2739a4dce43d138b94b153309333:src/trottertracks/rpe_method_optimizer.py'],cwd=ROOT)
    exec(compile(text,'fixed:1e88cd85c','exec'),module.__dict__)
    return module


def fixture(methods=METHODS,configurations=2,lookahead=3,steps=100):
    search=dict(decrement_steps=[1],large_step_headroom='9/10',seed_rules=[],max_steps_per_method=steps,
                max_new_q3_per_method=0,incumbent_scope='CONFIGURATION',lookahead=lookahead)
    problem=dict(accuracy=dict(fixture='ARTIFICIAL'),target=dict(energy_ha=0),rpe=dict(fixture=True),search=search,
                 methods={m:{}for m in methods},q3_routes={})
    saved=O.SavedBackend([],{});candidates=[]
    for m in methods:
        for k in range(configurations):
            cid='ARTIFICIAL_'+m+'_'+str(k);prices=[Fraction((13+i)*(k+1))for i in range(13)for _ in O.AXES]
            c=dict(method=m,family='PR',key=dict(L_D=k+1),configuration_id=cid,name=m+':'+cid,prices=prices,
                caps=[2]*13,ready=True,needs=[],labels=[],price_status='ARTIFICIAL',signal_source=None,saved_points=[])
            candidates.append(c);saved.table[cid,tuple([2]*26)]=dict(status=O.PASS,MSE_lower=Fraction(13,2),MSE_upper=Fraction(13,2),source='SAVED',label='ARTIFICIAL_V0')
    context=dict(candidates=candidates,saved=saved,tau=Fraction(7),expected={},generate=lambda *a:None,generation={})
    route=dict(kind='artificial',interpreter=sys.executable,configuration_ids='*',view_commit=None,
               artificial=dict(weights=[1]*13,scale='1',tau='7',sleep_seconds=.025))
    return problem,context,{m:[deepcopy(route)]for m in methods}


def canonical(result):
    return {m:dict(best=None if r['best']is None else(r['best']['configuration_id'],r['best']['N_X'],r['best']['N_Y'],r['best']['cost']),
        stop=r['stop_reason'],trace=[(t['configuration_id'],t['vector'],t['status'])for t in r['trace']],
        by=[(x['configuration_id'],x['stop_reason'])for x in r.get('by_configuration')or[]])for m,r in result.items()}


def run_fixture(directory,fast,*,methods=METHODS,configurations=2,budget=None,steps=100,unknown=False,dp_workers=None,scope='CONFIGURATION'):
    problem,context,routes=fixture(methods,configurations,steps=steps)
    problem['search']['incumbent_scope']=scope
    if unknown:
        for rules in routes.values():rules[0]['artificial']['force_unknown']=True
    budgets={m:budget for m in methods}
    if fast:
        ex=C.ConcurrentExecutor(ROOT,directory,routes,{},problem,max_concurrent=6,budgets=budgets,science_approved=True,dp_workers=dp_workers)
        result=C.optimize(ROOT,problem,context=context,executor=ex,methods=methods,max_new=budgets)
        metrics=ex.metrics()
    else:
        B=baseline();ex=B.Executor(ROOT,directory,routes,{},problem,max_concurrent=6)
        result=B.optimize(ROOT,problem,context=context,executor=ex,methods=methods,max_new=budgets)
        metrics=dict(launched=sum(ex.launched(m)for m in methods))
    return result,metrics


class SchedulerTests(unittest.TestCase):
    def test_native_pf_template_request_accepts_a_partial_registry(self):
        executor = C.ConcurrentExecutor.__new__(C.ConcurrentExecutor)
        executor.method = 'TRUNCATED_PF'
        executor.optimizer_source = 'synthetic-test'
        executor.expected = {}
        executor.dp_workers = 1
        executor.prefix_cache_directory = None
        executor.compact_q3_evidence = False
        locator = dict(commit='a'*40, path='saved-cases.json', inputs_selector=['inputs'])
        registry = {'another-case': dict(configuration=dict(commit='b'*40, path='other.json'))}
        route = dict(kind='pf_b_pf6', configurations=registry, templates=[locator])
        proposal = dict(configuration_id='saved-case', vector=(1,)*26, cost=Fraction(1), origin={'M': 12})

        request = executor.request(proposal, Path('native-q3'), route)

        self.assertNotIn('configurations', request['route'])
        self.assertEqual(route['configurations'], registry)
        self.assertEqual(request['route']['templates'], [locator])
        with patch.object(Q.subprocess, 'check_output', return_value=b'saved template bytes') as show:
            Q.input_hashes(ROOT, request)
        show.assert_called_once_with(['git', 'show', locator['commit']+':'+locator['path']], cwd=ROOT)

        proposal['configuration_id'] = 'another-case'
        registered = executor.request(proposal, Path('registered-q3'), route)
        self.assertEqual(registered['route'], route)

    def test_projected_q3_request_keeps_the_fixed_source_inputs_addressable(self):
        problem, context, routes = fixture(['PR_DF'], 1)
        problem['rpe']['M'] = 12
        candidate = context['candidates'][0]
        source_id = candidate['configuration_id']
        locator = dict(commit='a'*40, path='signals.json')
        binding = dict(configuration=locator, signal_packet=locator, binding_id='fixed-binding',
                       total_configuration_id='fixed-total')
        route = dict(routes['PR_DF'][0], kind='prdf_d2_common_entry',
                     configurations={source_id: binding}, contract_template=locator)
        with tempfile.TemporaryDirectory() as tmp:
            root = C.ConcurrentExecutor(ROOT, tmp, {'PR_DF': [route]}, {}, problem)
            try:
                ex = root.for_method('PR_DF')
                original = ex.request(dict(configuration_id=source_id, vector=(1,)*26,
                                           cost=Fraction(1), origin={}), Path('source'), route)
                projected = S.horizon_candidates([candidate], problem, {'M': [11]}, ex)[0]
                projected_id = projected['configuration_id']
                projected_rule = ex.route(projected_id)
                request = ex.request(dict(configuration_id=projected_id, vector=(1,)*24,
                                          cost=Fraction(1), origin={'M': 11}), Path('prefix'), projected_rule)
                self.assertEqual(request['configuration_id'], projected_id)
                self.assertEqual(request['expected']['M'], 11)
                self.assertEqual(request['expected']['binding_id'], 'fixed-binding')
                self.assertEqual(projected_rule['configurations'], {source_id: binding})
                self.assertEqual(request['route']['configurations'], {source_id: binding, projected_id: binding})
                with patch.object(Q.subprocess, 'check_output', return_value=b'saved signal bytes') as show:
                    self.assertEqual(Q.input_hashes(ROOT, request), Q.input_hashes(ROOT, original))
                    show.assert_called_with(['git', 'show', locator['commit']+':'+locator['path']], cwd=ROOT)
                template_route = dict(projected_rule, kind='pf_b_pf6', configurations={}, templates=[locator])
                template_request = ex.request(dict(configuration_id=projected_id, vector=(1,)*24,
                                                   cost=Fraction(1), origin={'M': 11}),
                                              Path('template-prefix'), template_route)
                self.assertNotIn('configurations', template_request['route'])
                self.assertEqual(template_route['configurations'], {})
                with patch.object(Q.subprocess, 'check_output', return_value=b'saved template bytes'):
                    inputs = Q.input_hashes(ROOT, template_request)
                self.assertEqual([(item['commit'], item['path']) for item in inputs],
                                 [(locator['commit'], locator['path'])])
                missing_source = dict(projected_rule, configurations={})
                request = ex.request(dict(configuration_id=projected_id, vector=(1,)*24,
                                          cost=Fraction(1), origin={'M': 11}), Path('missing'), missing_source)
                self.assertEqual(request['route']['configurations'], {})
            finally:
                root.finish()

    def test_prior_job_prefers_matching_completed_result_over_first_foreign_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ex=O.Executor.__new__(O.Executor)
            ex.method='PR_PAULI'
            ex.prior_run_dirs=[root/'old',root/'same']
            proposal=dict(configuration_id='fixed',vector=(1,2))
            jobs=[p/'jobs/PR_PAULI/job' for p in ex.prior_run_dirs]
            for job in jobs:
                job.mkdir(parents=True)
                (job/'OUTCOME.json').touch()
            with patch.object(ex,'job_dir',return_value=root/'job'), patch.object(
                    ex,'same_route',side_effect=lambda job,cid:job==jobs[1]):
                self.assertEqual(ex.prior_job(proposal),jobs[1])
                self.assertEqual(ex.prior_job(proposal,exclude=[jobs[1]]),jobs[0])

    def test_four_methods_same_adoption_order_best_and_stop(self):
        with tempfile.TemporaryDirectory()as tmp:
            old,_=run_fixture(Path(tmp)/'old',False)
            new,metrics=run_fixture(Path(tmp)/'new',True)
            self.assertEqual(canonical(old),canonical(new))
            self.assertLessEqual(metrics['peak_active'],6)
            self.assertEqual(metrics['new_real_route_launches'],0)
            self.assertGreaterEqual(metrics['speculative_launched'],0)

    def test_finite_budget_reserved_before_launch(self):
        with tempfile.TemporaryDirectory()as tmp:
            old,_=run_fixture(Path(tmp)/'old',False,budget=2)
            new,metrics=run_fixture(Path(tmp)/'new',True,budget=2)
            self.assertEqual(canonical(old),canonical(new))
            self.assertEqual(metrics['launched'],8)
            for m,r in new.items():self.assertEqual(r['new_calls'],2)

    def test_queued_prefetch_budget_includes_every_reservation(self):
        p,c,r=fixture(['PR_DF'],1)
        with tempfile.TemporaryDirectory()as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,r,{},p,max_concurrent=1,budgets={'PR_DF':2})
            ex=root.for_method('PR_DF');cfg=c['candidates'][0]
            props=[dict(configuration_id=cfg['configuration_id'],vector=tuple([n]*26),cost=Fraction(n),origin={})for n in (2,1,3)]
            ex.prefetch(props)
            self.assertLessEqual(root.state.reserved['PR_DF'],2)
            self.assertEqual(ex.answer(props[2])['status'],'BUDGET_EXHAUSTED')
            root.finish();self.assertLessEqual(root.metrics()['launched'],2)

    def test_price_retirement_leaves_running_and_other_method_work(self):
        problem, context, routes=fixture(['PR_PAULI','PR_DF'],configurations=3)
        with tempfile.TemporaryDirectory() as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,routes,{},problem,max_concurrent=1)
            try:
                ex=root.for_method('PR_PAULI')
                # Hold the dispatcher while arranging two outcomes of the queue race.
                with ex.state.cv:
                    proposals=[dict(configuration_id=c['configuration_id'],vector=(1,)*26,
                                    cost=Fraction(1),origin={}) for c in context['candidates']]
                    assigned=ex._enqueue(proposals[0])
                    assigned.status='RUNNING'
                    queued=ex._enqueue(proposals[1])
                    untouched=ex._enqueue(proposals[2])
                    other=root.for_method('PR_DF')._enqueue(proposals[3])
                    ex.standby=C.deque(proposals[:3])
                    ex.retire_priced_out({p['configuration_id'] for p in proposals[:2]+proposals[3:4]})
                    self.assertEqual([t.status for t in (assigned,queued,untouched,other)],
                                     ['RUNNING','CANCELLED','QUEUED','QUEUED'])
                    self.assertEqual(list(ex.standby),[proposals[2]])
                    assigned.status='QUEUED'
                    ex._cancel(assigned)
                    ex._cancel(untouched)
                    other.view._cancel(other)
            finally:
                root.finish()
            self.assertEqual(root.metrics()['launched'],0)

    def test_standby_replenishes_after_worker_exit_even_while_demand_is_blocked(self):
        problem, context, routes = fixture(['PR_DF'], configurations=4)
        proposals = [dict(configuration_id=c['configuration_id'], vector=(1,) * 26,
                          cost=Fraction(1), origin={}) for c in context['candidates']]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root = C.ConcurrentExecutor(ROOT, directory, routes, {}, problem,
                                        max_concurrent=2, budgets={'PR_DF': 4})
            ex = root.for_method('PR_DF')
            demand, last = ex.job_dir(proposals[0]), ex.job_dir(proposals[-1])
            run_adapter = O.Executor.run_adapter

            def gated(view, job, rule, mode):
                if job != demand or mode != 'execute':
                    return run_adapter(view, job, rule, mode)
                # Even the last standby job must launch before demand completes.
                gate = (
                    "from pathlib import Path; import os,sys,time\n"
                    "end=time.monotonic()+8\n"
                    "while not Path(sys.argv[1]).exists():\n"
                    "    if time.monotonic()>end: sys.exit(71)\n"
                    "    time.sleep(.01)\n"
                    "os.execv(sys.executable,[sys.executable,'-B',*sys.argv[2:]])\n"
                )
                with (job / 'EXECUTE.log').open('a') as log:
                    return subprocess.Popen([sys.executable, '-B', '-c', gate, str(last / 'LAUNCH.json'),
                          str(view.adapter), '--request', str(job / 'REQUEST.json'), '--output', str(job), '--mode', mode],
                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

            try:
                with patch.object(O.Executor, 'run_adapter', gated):
                    ex.prefetch_standby(proposals[0], proposals[1:])
                    response = ex.answer(proposals[0])
            finally:
                root.finish()
            self.assertIn(response['status'], (O.PASS, O.FAIL))
            events = [json.loads(l) for l in (directory / 'PROGRESS.jsonl').read_text().splitlines()]
            launched = {e['job']: e['monotonic'] for e in events if e['event'] == 'LAUNCH'}
            exits = {e['job']: e['monotonic'] for e in events if e['event'] == 'EXIT'}
            self.assertEqual(len(launched), 4)
            self.assertLess(launched[last.name], exits[demand.name])
            self.assertGreater(launched[last.name], min(exits.values()))
            self.assertEqual(root.metrics()['peak_active'], 2)
            self.assertGreaterEqual(sum(e['event'] == 'Q3_STANDBY_REFILL' for e in events), 2)
            self.assertEqual(set(launched), set(exits))

            resume = C.ConcurrentExecutor(ROOT, directory, routes, {}, problem,
                                          max_concurrent=2, budgets={'PR_DF': 4})
            try:
                view = resume.for_method('PR_DF')
                view.prefetch_standby(proposals[0], proposals[1:])
                self.assertEqual(view.answer(proposals[0])['MSE_upper'], response['MSE_upper'])
            finally:
                resume.finish()
            self.assertEqual(resume.metrics()['launched'], 0)

    def test_ready_configuration_order_uses_measured_q3_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in (False, True):
                problem, context, routes = fixture(['PR_DF'], configurations=2, steps=1)
                problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2', work_conserving_search=enabled)
                slow, fast = context['candidates']
                slow['prices'] = [Fraction(3)] * 26
                fast['prices'] = [Fraction(4)] * 2 + [Fraction(3)] * 24
                root = C.ConcurrentExecutor(ROOT, Path(tmp) / str(enabled), routes, {}, problem,
                                            max_concurrent=1, budgets={'PR_DF': 1})
                view = root.for_method('PR_DF')
                root.state.observations['PR_DF'].extend(
                    dict(id='receipt-' + c['configuration_id'], method='PR_DF', configuration_id=c['configuration_id'],
                         kind='SHOT_Q3', profile=view.search_profile(c['configuration_id']), status=O.PASS,
                         finished_at=time.monotonic() - 10, seconds=seconds)
                    for c, seconds in ((slow, 100), (fast, 10)))
                r = C.optimize(ROOT, problem, context=context, executor=root, max_new={'PR_DF': 1})
                first = r['PR_DF']['trace'][0]
                self.assertEqual(first['configuration_id'], (fast if enabled else slow)['configuration_id'])
                self.assertEqual('order_reason' in first, enabled)
                self.assertEqual(root.metrics()['launched'], 1)

    def test_distributed_order_uses_real_artificial_worker_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            problem, context, routes = fixture(['PR_DF'], configurations=2, steps=1)
            problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2', work_conserving_search=True)
            slow, fast = context['candidates']
            slow['prices'] = [Fraction(3)] * 26
            fast['prices'] = [Fraction(4)] * 2 + [Fraction(3)] * 24
            template = routes['PR_DF'][0]
            routes['PR_DF'] = [dict(template, configuration_ids=[c['configuration_id']],
                artificial=dict(template['artificial'], sleep_seconds=seconds))
                for c, seconds in ((slow, 1), (fast, .02))]
            profile = dict(host_id='fixture-observed', root=str(ROOT), interpreter=sys.executable,
                           slots=1, route_views={}, work_dir=str(Path(tmp)/'worker'))
            root = C.DistributedExecutor(ROOT, tmp, routes, {}, problem, max_concurrent='auto', budgets={'PR_DF': 3},
                optimizer_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                distributed=dict(port=0, work_conserving_search=True, local_worker=profile))
            view = root.for_method('PR_DF')
            try:
                for candidate in (slow, fast):
                    vector = (2,) * 26
                    seed = dict(configuration_id=candidate['configuration_id'], vector=vector,
                                cost=O.cost(vector, candidate['prices']), origin=dict(kind='ARTIFICIAL_SEED'))
                    self.assertEqual(view.answer(seed)['status'], O.PASS)
                self.assertEqual(len(view.search_observations()), 2)
                result = C.optimize(ROOT, problem, context=context, executor=root, max_new={'PR_DF': 3})['PR_DF']
                first = result['trace'][0]
                self.assertEqual(first['configuration_id'], fast['configuration_id'])
                self.assertEqual(first['order_baseline'], slow['configuration_id'])
                self.assertEqual(first['order_reason'], 'IMMEDIATE_SAVING_X_EMPIRICAL_PASS_PER_ESTIMATED_TIME')
                self.assertEqual(root.store.counts('PR_DF', started_only=True), 3)
            finally:
                root.finish()

    def test_completed_resume_no_duplicate_launch(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp);first,_=run_fixture(path,True,methods=['PR_DF'],budget=2)
            markers={str(p):p.read_bytes()for p in path.glob('jobs/*/*/LAUNCH.json')}
            second,metrics=run_fixture(path,True,methods=['PR_DF'],budget=2)
            self.assertEqual(canonical(first),canonical(second));self.assertEqual(metrics.get('launched',0),0)
            self.assertEqual(markers,{str(p):p.read_bytes()for p in path.glob('jobs/*/*/LAUNCH.json')})

    def test_stop_releases_preparation_waiting_for_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = object.__new__(C.DistributedExecutor)
            executor.run_dir = Path(tmp)
            executor.state = types.SimpleNamespace(cv=threading.Condition(), closed=False,
                                                   failure=None, tasks={})
            executor.store = types.SimpleNamespace(row=lambda job: None)
            executor.preparation_pending = {'prep-artificial': {'decision': {'eligible_host_ids': []}}}
            executor.preparation_holds = {}
            executor._preparation_probe_hold = lambda pending: None
            executor.log = lambda **event: None
            (executor.run_dir/'STOP_REQUESTED.json').touch()
            with patch.object(executor.state.cv, 'wait', side_effect=AssertionError('waited after Stop')):
                result = executor.await_preparation('prep-artificial')
            self.assertEqual(result['status'], 'PREPARATION_STOPPED_BEFORE_RESERVATION')
            self.assertEqual(result['preparation_calls'], 0)

    def test_pending_and_crash_claim_not_retried(self):
        p,c,r=fixture(['PR_DF'],1)
        with tempfile.TemporaryDirectory()as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,r,{},p,max_concurrent=1,budgets={'PR_DF':1});ex=root.for_method('PR_DF')
            cfg=c['candidates'][0];prop=dict(configuration_id=cfg['configuration_id'],vector=tuple([2]*26),cost=Fraction(1),origin={})
            job=ex.job_dir(prop);job.mkdir(parents=True);(job/'STARTING.json').write_text('{}')
            self.assertEqual(ex.answer(prop)['status'],'STALE_JOB');root.finish()
            self.assertFalse((job/'LAUNCH.json').exists())

    def test_duplicate_demands_share_one_job(self):
        p,c,r=fixture(['PR_DF'],1)
        with tempfile.TemporaryDirectory()as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,r,{},p,max_concurrent=2,budgets={'PR_DF':1});ex=root.for_method('PR_DF')
            cfg=c['candidates'][0];prop=dict(configuration_id=cfg['configuration_id'],vector=tuple([2]*26),cost=Fraction(1),origin={})
            answers=[];threads=[threading.Thread(target=lambda:answers.append(ex.answer(prop)))for _ in range(3)]
            for t in threads:t.start()
            for t in threads:t.join()
            root.finish();self.assertEqual(root.metrics()['launched'],1);self.assertEqual(len(answers),3)
            self.assertEqual(root.metrics()['adopted'],1)

    def test_speculation_is_recorded_and_adopted_in_demand_order(self):
        p,c,r=fixture(['PR_DF'],1)
        r['PR_DF'][0]['artificial']['sleep_by_first_shot']={'2':.15}
        with tempfile.TemporaryDirectory()as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,r,{},p,max_concurrent=2,budgets={'PR_DF':2});ex=root.for_method('PR_DF')
            cfg=c['candidates'][0]
            props=[dict(configuration_id=cfg['configuration_id'],vector=tuple([n]*26),cost=Fraction(n),origin={})for n in (2,1)]
            ex.prefetch(props)
            for _ in range(200):
                if root.state.metrics['launched']==2:break
                time.sleep(.005)
            answers=[ex.answer(p)for p in props];root.finish()
            self.assertEqual(len(answers),2)
            self.assertEqual(root.metrics()['speculative_launched'],1)
            self.assertEqual(root.metrics()['speculative_adopted'],1)
            events=[json.loads(l)for l in(Path(tmp)/'PROGRESS.jsonl').read_text().splitlines()]
            self.assertEqual([e['job']for e in events if e['event']=='EXIT'],[ex.job_dir(props[1]).name,ex.job_dir(props[0]).name])
            self.assertEqual([e['job']for e in events if e['event']=='ADOPT'],[ex.job_dir(p).name for p in props])

    def test_unknown_keeps_same_stop_as_legacy(self):
        with tempfile.TemporaryDirectory()as tmp:
            old,_=run_fixture(Path(tmp)/'old',False,methods=['PR_PAULI'],unknown=True)
            new,_=run_fixture(Path(tmp)/'new',True,methods=['PR_PAULI'],unknown=True)
            self.assertEqual(canonical(old),canonical(new))
            self.assertEqual(new['PR_PAULI']['stop_reason'],'UNKNOWN_RESULT')

    def test_execution_profile_changes_only_acceleration(self):
        spec=importlib.util.spec_from_file_location('_profile_adapter',ROOT/'scripts/run_rpe_optimizer_q3_job.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        inputs=dict(statistical_contract={'fixed':1},options={'fixed':2},adopted_evaluation={'fixed':3},
                    acceleration={'workers_per_configuration':8,'outcome_batch':8,'source':'FROZEN'})
        changed=adapter.execution_profile(inputs,dict(execution_profile=dict(dp_workers=2)))
        for key in adapter.CONTRACT:self.assertEqual(inputs[key],changed[key])
        self.assertEqual(inputs['acceleration']['workers_per_configuration'],8)
        self.assertEqual(changed['acceleration']['workers_per_configuration'],2)

    def test_two_drivers_cannot_own_same_directory(self):
        p,c,r=fixture(['PR_DF'],1)
        with tempfile.TemporaryDirectory()as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,r,{},p)
            try:
                with self.assertRaisesRegex(ValueError,'already owns'):C.ConcurrentExecutor(ROOT,tmp,r,{},p)
            finally:root.finish()

    def test_method_scope_trajectory_is_unchanged(self):
        with tempfile.TemporaryDirectory()as tmp:
            old,_=run_fixture(Path(tmp)/'old',False,scope='METHOD')
            new,_=run_fixture(Path(tmp)/'new',True,scope='METHOD')
            self.assertEqual(canonical(old),canonical(new))

    def test_resumed_live_jobs_occupy_capacity_without_relaunch(self):
        p,c,r=fixture(['PR_DF'],1);r['PR_DF'][0]['artificial']['sleep_seconds']=.25
        with tempfile.TemporaryDirectory()as tmp:
            old=baseline().Executor(ROOT,tmp,r,{},p,max_concurrent=1).for_method('PR_DF')
            cid=c['candidates'][0]['configuration_id']
            first=dict(configuration_id=cid,vector=tuple([2]*26),cost=Fraction(2),origin={})
            second=dict(configuration_id=cid,vector=tuple([1]*26),cost=Fraction(1),origin={})
            oldjob=old.launch(first,r['PR_DF'][0])
            root=C.ConcurrentExecutor(ROOT,tmp,r,{},p,max_concurrent=1,budgets={'PR_DF':2},wait_for_running=True)
            ex=root.for_method('PR_DF');ex.answer(second);root.finish();old.finish()
            self.assertEqual(root.metrics()['peak_active'],1)
            self.assertEqual(len(list(Path(tmp).glob('jobs/*/*/LAUNCH.json'))),2)
            oldstart=json.loads((oldjob/'LAUNCH.json').read_text())['started']
            oldwall=json.loads((oldjob/'PROCESS_RETURN.json').read_text())['wall_seconds']
            newstart=json.loads((ex.job_dir(second)/'LAUNCH.json').read_text())['started']
            self.assertGreaterEqual(newstart,oldstart+oldwall)

    def test_actual_cli_round_path_with_artificial_context(self):
        spec=importlib.util.spec_from_file_location('_throughput_cli',ROOT/'scripts/run_rpe_method_optimizer.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        problem,context,routes=fixture(configurations=1);problem['execution_routes']=routes
        with tempfile.TemporaryDirectory()as tmp:
            tmp=Path(tmp);pf=tmp/'problem.json';rf=tmp/'round.json';run=tmp/'run'
            pf.write_text(json.dumps(problem));rf.write_text(json.dumps(dict(round_id='ARTIFICIAL_CLI',methods=METHODS,
                science_approved=True,max_new_q3_per_method={m:1 for m in METHODS},initial='all',
                executor=dict(scheduler='concurrent',max_concurrent_jobs=2,dp_workers=1,
                    prefix_cache_directory=str(tmp/'prefix'),prefix_checkpoint_rounds=[11],prefix_cache_max_bytes=32000,compact_q3_evidence=True,parallel_dp_residual=True))))
            with patch.object(O,'prepare',return_value=context),patch.object(sys,'argv',
                ['run_rpe_method_optimizer.py','--problem',str(pf),'--round',str(rf),'--run-dir',str(run)]):cli.main()
            data=json.loads((run/'ROUND_RESULT.json').read_text())
            self.assertEqual(len(list(run.glob('jobs/*/*/LAUNCH.json'))),4)
            self.assertEqual(set(data['methods']),set(METHODS))
            for request in run.glob('jobs/*/*/REQUEST.json'):
                profile=json.loads(request.read_text())['execution_profile']
                self.assertEqual(profile['prefix_cache_directory'],str(tmp/'prefix'))
                self.assertEqual(profile['prefix_checkpoint_rounds'],[11])
                self.assertEqual(profile['prefix_cache_max_bytes'],32000)
                self.assertIs(profile['compact_q3_evidence'],True)
                self.assertIs(profile['parallel_dp_residual'],True)
            self.assertTrue(all(r['executor_metrics']['new_real_route_launches']==0 for r in data['methods'].values()))

    def test_host_only_prefix_cache_can_receive_the_budget(self):
        problem, context, routes = fixture(['PR_DF'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = C.ConcurrentExecutor(ROOT, tmp, routes, {}, problem, prefix_cache_max_bytes=32000)
            try:
                ex = root.for_method('PR_DF')
                candidate = context['candidates'][0]
                proposal = dict(configuration_id=candidate['configuration_id'], vector=(1,) * 26,
                                cost=Fraction(1), origin={})
                request = ex.request(proposal, ex.job_dir(proposal), routes['PR_DF'][0])
                self.assertNotIn('prefix_cache_directory', request['execution_profile'])
                self.assertEqual(request['execution_profile']['prefix_cache_max_bytes'], 32000)
            finally:
                root.finish()

    def test_launch_persistence_failure_stops_queue_and_waits_started_child(self):
        p,c,r=fixture(['PR_DF'],1);r['PR_DF'][0]['artificial']['sleep_seconds']=.15
        with tempfile.TemporaryDirectory()as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,r,{},p,max_concurrent=1,budgets={'PR_DF':2})
            ex=root.for_method('PR_DF');cid=c['candidates'][0]['configuration_id']
            props=[dict(configuration_id=cid,vector=tuple([n]*26),cost=Fraction(n),origin={})for n in (2,1)]
            original=C._write
            def fail_launch(path,value):
                if Path(path).name=='LAUNCH.json':raise OSError('ARTIFICIAL_DISK_FAILURE_AFTER_POPEN')
                return original(path,value)
            with patch.object(C,'_write',side_effect=fail_launch):
                with root.state.cv:ex.prefetch(props)
                answer=ex.answer(props[0]);root.finish()
            self.assertEqual(answer['status'],'UNKNOWN')
            self.assertEqual(root.metrics()['launched'],1)
            self.assertEqual(root.metrics()['peak_active'],1)
            self.assertEqual(len(list(Path(tmp).glob('jobs/*/*/ARTIFICIAL_RESULT.json'))),1)
            self.assertTrue((ex.job_dir(props[0])/'PROCESS_RETURN.json').exists())
            self.assertFalse((ex.job_dir(props[1])/'REQUEST.json').exists())
            with self.assertRaisesRegex(RuntimeError,'no new launches'):ex.answer(props[1])

    def test_fixed_selection_except_explicit_execution_readiness_guard(self):
        old=ast.parse(subprocess.check_output(['git','show','5b48aabbc:src/trottertracks/rpe_method_optimizer.py'],cwd=ROOT))
        new=ast.parse((ROOT/'src/trottertracks/rpe_method_optimizer.py').read_bytes())
        # The route repair adds only a pre-proposal availability gate. Retain
        # the strict AST comparison for every ranking, price and update rule.
        guard=ast.dump(ast.parse("not c['ready'] or c.get('execution_needs')",mode='eval').body)
        isolated=ast.dump(ast.parse("""if hold:
    candidate=next(c for c in ready if c['configuration_id']==choice['configuration_id'])
    held.append(hold_candidate(candidate,choice,answer,backend))
    continue
""").body[0])
        local_unknown=ast.dump(ast.parse('''if is_local_unknown(answer):
    candidate=next(c for c in ready if c['configuration_id']==choice['configuration_id'])
    new_calls+=answer.get('calls',0)
    held.append(hold_candidate(candidate,choice,answer,backend))
    continue
''').body[0])
        class StripReadiness(ast.NodeTransformer):
            count=0
            isolated_count=0
            local_unknown_count=0
            def visit_If(self,node):
                if ast.dump(node)==local_unknown:
                    self.local_unknown_count+=1;return None
                if ast.dump(node)==isolated:
                    self.isolated_count+=1;return None
                return self.generic_visit(node)
            def visit_BoolOp(self,node):
                if ast.dump(node)==guard:
                    self.count+=1
                    return ast.parse("not c['ready']",mode='eval').body
                return self.generic_visit(node)
        for name in ('proposals','method_state','run_method','run_by_configuration'):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)
            strip=StripReadiness();b=strip.visit(b)
            self.assertEqual(strip.count,1 if name in('proposals','run_method')else 0,name)
            self.assertEqual(strip.isolated_count,1 if name=='run_method'else 0,name)
            self.assertEqual(strip.local_unknown_count,1 if name=='run_method'else 0,name)


if __name__=='__main__':unittest.main(verbosity=2)
