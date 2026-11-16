"""B2 normal-entry integration checks with artificial state and workers only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import run_rpe_schedule_search as W
import run_rpe_kappa_neighbourhood as K
import test_rpe_optimizer_concurrent as F
from trottertracks import rpe_optimizer_concurrent as C


class EntryTests(unittest.TestCase):
    def driver(self, run):
        d = W.WideDriver.__new__(W.WideDriver)
        d.run = Path(run);d.state = dict(cycle=1, points=[])
        d.a = SimpleNamespace(plan_only=False, batch=4, problem=Path('ARTIFICIAL_PROBLEM.json'))
        d.spec = dict(science_approved=True, max_new_q3_per_method={'PR_PAULI': 5},
                      candidate_completion={'parallel': 2}, executor=dict(scheduler='concurrent', dp_workers=2,
                      prefix_cache_directory='prefix', prefix_checkpoint_rounds=[5, 11]))
        d.save = lambda: None
        return d

    def test_wide_batch_preserves_profile_caps_total_budget_and_freezes_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp);original = deepcopy(d.spec)
            for i in range(3):
                job = d.run / 'jobs/PR_PAULI' / str(i);job.mkdir(parents=True)
                (job / 'LAUNCH.json').write_text('{}')
            def fake_run(*args, **kwargs):
                spec = json.loads((d.run / 'ROUND_INVOCATION.json').read_text())
                self.assertEqual(spec['max_new_q3_per_method'], {'PR_PAULI': 5})
                self.assertEqual(spec['executor'], original['executor'])
                self.assertNotIn('candidate_completion', spec)
                (d.run / 'ROUND_RESULT.json').write_text(json.dumps(dict(methods=dict(PR_PAULI=dict(stop_reason='Q3_BUDGET_EXHAUSTED',new_calls=2)))))
                return SimpleNamespace(returncode=0)
            with patch.object(K.subprocess, 'run', side_effect=fake_run):
                self.assertEqual(d.batch(ready_only=True)[0], 0)
            self.assertEqual(d.spec, original)

    def test_completion_wait_runs_ready_work_and_absorbs_in_selection_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp);released = threading.Event();started = threading.Event();calls=[]
            d.state['points'] = [dict(L_D=ld, kappa='2', status='COMPLETION',requested_cycle=1) for ld in (6,7)]
            class ArtificialCompletion:
                def complete(self, request):
                    calls.append(request['design']['L_D']);started.set()
                    if not released.wait(3):raise AssertionError('ready work did not run during preparation')
                    ld=request['design']['L_D']
                    return dict(status='READY_FOR_SEARCH' if ld==7 else 'HELD_WITH_REASON',
                                configuration_id='ARTIFICIAL_7',key=str(ld),reason='ARTIFICIAL_HOLD')
            d.completion_api=lambda _:ArtificialCompletion()
            def ready_work():
                self.assertTrue(started.wait(1));self.assertTrue(all(p['status']=='COMPLETION' for p in d.state['points']))
                released.set()
            d.completion_requests(on_wait=ready_work)
            self.assertEqual([p['status'] for p in d.state['points']],['HELD','REGISTERED'])
            self.assertCountEqual(calls,[6,7])
            d.completion_requests(on_wait=lambda:self.fail('restarted completed/held preparation'))
            self.assertCountEqual(calls,[6,7])

    def test_plan_only_cannot_start_preparation_or_ready_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.driver(tmp);d.a.plan_only=True
            d.state['points']=[dict(L_D=6,kappa='2',status='COMPLETION',requested_cycle=1)]
            d.completion_api=lambda _:self.fail('plan-only started preparation')
            d.completion_requests(on_wait=lambda:self.fail('plan-only ran Q3'))
            self.assertEqual(d.state['points'][0]['status'],'HELD')

    def test_concurrent_executor_stays_open_through_preparation_continuation(self):
        problem,context,routes=F.fixture(methods=['PR_PAULI'],configurations=1)
        with tempfile.TemporaryDirectory() as tmp:
            ex=C.ConcurrentExecutor(ROOT,tmp,routes,{},problem,max_concurrent=2,budgets={'PR_PAULI':2})
            try:
                first=C.optimize(ROOT,problem,context=context,executor=ex,methods=['PR_PAULI'],max_new={'PR_PAULI':1},close_executor=False)
                self.assertFalse(ex.state.closed)
                before=ex.launched('PR_PAULI')
                second=C.optimize(ROOT,problem,context=context,executor=ex,methods=['PR_PAULI'],max_new={'PR_PAULI':2},close_executor=False)
                self.assertLessEqual(ex.launched('PR_PAULI'),2)
                self.assertGreaterEqual(ex.launched('PR_PAULI'),before)
                self.assertFalse(ex.state.closed)
                self.assertIn('speculative_completed',ex.for_method('PR_PAULI').prefetch_state())
            finally:ex.finish()
            metrics=ex.metrics()
            self.assertTrue(ex.state.closed)
            self.assertGreaterEqual(metrics['idle_slot_seconds_waiting_selected_result'],0)
            self.assertEqual(metrics['new_real_route_launches'],0)

    def test_finish_waits_for_unselected_resumed_worker(self):
        """A resumed lookahead may never be demanded, but its return gate stays."""
        problem, context, routes = F.fixture(methods=['PR_DF'], configurations=1)
        routes['PR_DF'][0]['artificial']['sleep_seconds'] = .15
        with tempfile.TemporaryDirectory() as tmp:
            old = F.baseline().Executor(ROOT, tmp, routes, {}, problem, max_concurrent=1).for_method('PR_DF')
            proposal = dict(configuration_id=context['candidates'][0]['configuration_id'],
                            vector=tuple([2]*26), cost=1, origin={})
            job = old.launch(proposal, routes['PR_DF'][0])
            current = None
            try:
                current = C.ConcurrentExecutor(ROOT, tmp, routes, {}, problem, max_concurrent=1,
                                               budgets={'PR_DF': 1}, wait_for_running=True)
                current.finish()
                self.assertTrue((job / 'PROCESS_RETURN.json').exists())
                self.assertFalse(current.state.external)
                self.assertEqual(current.metrics()['launched'], 0)
            finally:
                if current is not None:
                    current.finish()
                old.finish()

    def test_new_run_carries_family_constants_and_preserves_scientific_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);previous=tmp/'previous';previous.mkdir();run=tmp/'next'
            old=dict(round_id='ARTIFICIAL_PREVIOUS',family_constants={'1':[],'5':[]},
                points=[dict(L_D=6,status='HELD',reason='COMPLETION_REQUIRED_NOT_RUN_IN_THIS_INVOCATION'),
                        dict(L_D=7,status='HELD',reason='SCIENTIFIC_STOP_PRESERVED')],
                stop=dict(reason='OLD_STOP_PRESERVED'),saved_packets=[],splits={})
            (previous/'KAPPA_STATE.json').write_text(json.dumps(old))
            spec=dict(round_id='ARTIFICIAL_NEXT',signals=dict(worktree=str(tmp),directory='artificial',saved_packets=[]),
                continues=dict(run_dir=str(previous)),search_overrides=dict(schedule_wide={}),
                routes_override=dict(PR_PAULI=[dict(configurations_file=str(run/'registry.json'))]),
                method_overrides=dict(PR_PAULI=dict(generation=dict(kappa_points_file=str(run/'points.json')))))
            rf=tmp/'round.json';rf.write_text(json.dumps(spec))
            with patch.object(W.WideDriver,'seed'):
                d=W.WideDriver(SimpleNamespace(run_dir=run,round=rf))
            self.assertEqual(set(d.state['family_constants']),{'1','5'})
            self.assertEqual([p['status'] for p in d.state['points']],['COMPLETION','HELD'])
            self.assertEqual(d.state['continuation']['previous_stop'],old['stop'])
            self.assertEqual(json.loads((previous/'KAPPA_STATE.json').read_text()),old)

    def test_persisted_mandatory_stop_prevents_prefetch_and_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.driver(tmp);d.state['stop']=dict(reason='V2_UNKNOWN_RESULT')
            d.plan=lambda:self.fail('mandatory STOP reopened planning')
            d.completion_requests=lambda **kw:self.fail('mandatory STOP reopened preparation')
            d.batch=lambda **kw:self.fail('mandatory STOP reopened Q3')
            d.loop()
            self.assertEqual(d.state['stop']['reason'],'V2_UNKNOWN_RESULT')

    def test_prior_complete_job_is_reused_even_for_direct_executor_demand(self):
        from fractions import Fraction
        problem,context,routes=F.fixture(methods=['PR_PAULI'],configurations=1)
        c=context['candidates'][0]
        proposal=dict(configuration_id=c['configuration_id'],vector=tuple([1]*26),cost=Fraction(1),origin={})
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            old=C.ConcurrentExecutor(ROOT,tmp/'old',routes,{},problem,max_concurrent=1,budgets={'PR_PAULI':1})
            try:want=old.for_method('PR_PAULI').answer(proposal)
            finally:old.finish()
            fresh=C.ConcurrentExecutor(ROOT,tmp/'new',routes,{},problem,max_concurrent=1,
                budgets={'PR_PAULI':0},science_approved=False,prior_run_dirs=[tmp/'old'])
            try:
                ex=fresh.for_method('PR_PAULI');ex.prefetch([proposal]);got=ex.answer(proposal,allow_new=False)
                self.assertEqual(got['source'],'PRIOR_ROUND_JOB')
                self.assertEqual(got['MSE_upper'],want['MSE_upper'])
                self.assertEqual(got['calls'],0);self.assertEqual(ex.launched('PR_PAULI'),0)
            finally:fresh.finish()

    def test_interim_batch_failure_is_persisted_before_any_second_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.driver(tmp);d.a.max_cycles=4;d.spec['overlap_ready_batch']=True
            d.state['points']=[dict(L_D=5,status='EVALUATED')]
            d.plan=lambda:([],dict(stop='ARTIFICIAL'))
            drained=[]
            def complete(on_wait):
                on_wait();drained.append(True)
            d.completion_requests=complete
            d.batch=lambda **kwargs:(17,None)
            d.absorb=lambda _:self.fail('failed interim batch was ingested')
            d.loop()
            self.assertTrue(drained)
            self.assertEqual(d.state['stop']['reason'],'V2_INVOCATION_FAILED')
            self.assertEqual(d.state['stop']['returncode'],17)
            d.batch=lambda **kwargs:self.fail('failed interim invocation was restarted')
            d.loop()

    def test_example_copies_fixed_price_registry_before_normal_entry(self):
        import prepare_rpe_throughput_round as E
        base=ROOT/'configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json'
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);saved=tmp/'saved';saved.mkdir()
            spec=json.loads(base.read_text())
            prices=saved/spec['price_rules_overrides']['registry_file']
            prices.parent.mkdir(parents=True)
            prices.write_text(json.dumps(dict(schema='ARTIFICIAL_PRICE_REGISTRY',entries=[])))
            output=tmp/'round.json';run=tmp/'run'
            result=E.prepare(base,output,run,saved)
            self.assertEqual(Path(result['price_rules_overrides']['registry_file']).read_bytes(),prices.read_bytes())
            self.assertFalse(result['science_approved'])
            self.assertEqual(result['max_new_q3_per_method'],spec['max_new_q3_per_method'])
            self.assertEqual(result['scope'],spec['scope'])
            with self.assertRaises(ValueError):E.prepare(base,output,run,saved)

    def test_wide_overlap_calls_normal_cli_while_completion_is_waiting(self):
        import run_rpe_method_optimizer as CLI
        problem,context,routes=F.fixture(methods=['PR_PAULI'],configurations=1)
        problem['execution_routes']=routes
        problem['search']['rule']='METHOD_MINIMUM_PRIORITY_V2'
        with tempfile.TemporaryDirectory() as tmp:
            d=self.driver(tmp);d.a.max_cycles=3;d.a.batch=2
            pf=Path(tmp)/'problem.json';pf.write_text(json.dumps(problem));d.a.problem=pf
            cid=context['candidates'][0]['configuration_id']
            d.spec.update(round_id='ARTIFICIAL_WIDE_OVERLAP',methods=['PR_PAULI'],initial='all',
                max_new_q3_per_method={'PR_PAULI':1},overlap_ready_batch=True)
            d.state['points']=[dict(L_D=1,kappa='2',status='EVALUATED',configuration_id=cid),
                               dict(L_D=6,kappa='2',status='COMPLETION',requested_cycle=1)]
            d.plan=lambda:([],dict(stop='ARTIFICIAL'))
            started=threading.Event();released=threading.Event();calls=[]
            class Completion:
                def complete(self, request):
                    started.set()
                    if not released.wait(3):raise AssertionError('ordinary ready batch never returned')
                    return dict(status='READY_FOR_SEARCH',configuration_id='ARTIFICIAL_NEW',key='new')
            d.completion_api=lambda _:Completion()
            original_run=K.subprocess.run
            def ordinary_cli(command, **kwargs):
                if not any(str(x).endswith('run_rpe_method_optimizer.py') for x in command):
                    return original_run(command, **kwargs)
                self.assertTrue(started.wait(1));calls.append(command)
                with patch.object(K.O,'prepare',return_value=context),patch.object(CLI,'head',return_value='ARTIFICIAL_FIXED_SOURCE'), \
                     patch.object(sys,'argv',command[2:]):
                    CLI.main()
                released.set()
                return SimpleNamespace(returncode=0)
            from trottertracks import pr_candidate_completion as PC
            with patch.object(K.subprocess,'run',side_effect=ordinary_cli), \
                 patch.object(PC,'prepare_for_search',side_effect=AssertionError('interim batch must not call scientific preparation')):
                d.loop()
            self.assertEqual(len(calls),1)
            self.assertEqual(d.state['stop']['reason'],'V2_Q3_BUDGET_EXHAUSTED')
            self.assertEqual(d.state['points'][1]['status'],'REGISTERED')
            jobs=list((Path(tmp)/'jobs').glob('PR_PAULI/*/REQUEST.json'))
            self.assertEqual(len(jobs),1)
            request=json.loads(jobs[0].read_text())
            self.assertEqual(request['route']['kind'],'artificial')
            self.assertEqual(request['execution_profile']['dp_workers'],2)

    def test_method_preview_releases_stale_other_configuration_reservation(self):
        import time
        problem,context,routes=F.fixture(methods=['PR_PAULI'],configurations=2)
        routes['PR_PAULI'][0]['artificial']['sleep_seconds']=.15
        a,b=context['candidates']
        def proposal(c,n):return dict(configuration_id=c['configuration_id'],vector=tuple([n]*26),cost=n,origin={})
        first,stale,next_pick=proposal(a,2),proposal(b,2),proposal(a,1)
        with tempfile.TemporaryDirectory() as tmp:
            root=C.ConcurrentExecutor(ROOT,tmp,routes,{},problem,max_concurrent=1,budgets={'PR_PAULI':2})
            ex=root.for_method('PR_PAULI')
            try:
                ex.prefetch([first,stale],scope='METHOD')
                deadline=time.monotonic()+3
                while not (ex.job_dir(first)/'LAUNCH.json').exists():
                    if time.monotonic()>deadline:self.fail('artificial worker did not launch')
                    time.sleep(.005)
                ex.prefetch([next_pick],scope='METHOD')
                self.assertEqual(root.state.tasks[ex.job_dir(stale)].status,'CANCELLED')
                result=ex.answer(next_pick)
                self.assertNotEqual(result['status'],'BUDGET_EXHAUSTED')
                self.assertFalse((ex.job_dir(stale)/'LAUNCH.json').exists())
                self.assertEqual(ex.launched('PR_PAULI'),2)
            finally:root.finish()

    def test_restart_reuses_completed_speculation_and_resets_observation_window(self):
        problem,context,routes=F.fixture(methods=['PR_PAULI'],configurations=1)
        cid=context['candidates'][0]['configuration_id']
        proposals=[dict(configuration_id=cid,vector=tuple([n]*26),cost=n,origin={})for n in(2,1)]
        with tempfile.TemporaryDirectory() as tmp:
            old=C.ConcurrentExecutor(ROOT,tmp,routes,{},problem,max_concurrent=2,budgets={'PR_PAULI':2})
            try:
                ex=old.for_method('PR_PAULI');ex.prefetch(proposals)
                for task in list(old.state.tasks.values()):self.assertTrue(task.event.wait(3))
                self.assertEqual(ex.prefetch_state()['speculative_completed'],1)
            finally:old.finish()
            resumed=C.ConcurrentExecutor(ROOT,tmp,routes,{},problem,max_concurrent=2,
                budgets={'PR_PAULI':2},science_approved=False)
            try:
                ex=resumed.for_method('PR_PAULI')
                self.assertEqual(ex.prefetch_state()['speculative_completed'],0)
                for proposal in proposals:self.assertEqual(ex.answer(proposal,allow_new=False)['calls'],0)
                self.assertEqual(ex.launched('PR_PAULI'),2)
                self.assertEqual(resumed.metrics()['launched'],0)
            finally:resumed.finish()


if __name__ == '__main__':unittest.main(verbosity=2)
