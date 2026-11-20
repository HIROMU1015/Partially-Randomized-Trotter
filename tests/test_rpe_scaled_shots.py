"""Artificial gateway/controller/worker acceptance for the finite multiplier rule."""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction as F
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_scaled_shots as S
from trottertracks.rpe_local0_search import Gateway, load, write
from trottertracks.rpe_release_registry import register

import run_rpe_schedule_search as entry
from test_rpe_optimizer_concurrent import fixture
from test_rpe_large_shot_bounds import FIXED as DECODER_SOURCE, FIXTURE as DECODER_VIEW

TAU = F(S.EPSILON)**2
METHODS = ['FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF']


def small_policy(horizons=(11,12,13)):
    return S.settings(dict(M=list(horizons), grid=dict(min_k=-12,max_k=-11,divisor=4), compare_paper=False))


class ScaleTests(unittest.TestCase):
    @unittest.skipUnless((DECODER_VIEW/'src/trottertracks/rpe_total_risk.py').is_file(),
                         'requires the fixed Z1+T1 source view')
    def test_normal_start_and_resume_with_large_shot_evidence(self):
        method = 'FULL_PF'
        with tempfile.TemporaryDirectory(prefix='artificial-large-shots-') as tmp:
            root = Path(tmp)
            problem, context, routes = fixture([method], configurations=1, steps=100)
            problem.update(problem=dict(system='ARTIFICIAL_SCALED_SHOTS',
                                        hamiltonian_data=dict(expected=dict(fragment_count=10))),
                           rpe=dict(M=12), execution_routes=routes)
            problem['accuracy']['epsilon_E_decimal'] = S.EPSILON
            context['saved'].table.clear()
            context['tau'] = TAU
            context['candidates'][0]['prices'] = [1]*26
            routes[method][0]['artificial'].update(tau=str(TAU), finite_support_fixture=dict(
                view=str(DECODER_VIEW), commit=DECODER_SOURCE, target=0.0))
            spec = dict(round_id='TEMPLATE',methods=[method],science_approved=False,
                        max_new_q3_per_method={method:2},
                        executor=dict(scheduler='concurrent',max_concurrent_jobs=1,large_shot_bounds=True))
            write(root/'PROBLEM.json',problem)
            write(root/'ROUND.json',spec)
            config = dict(problem=str(root/'PROBLEM.json'), round=str(root/'ROUND.json'),
                          entry='schedule',state_directory=str(root/'state'),code_root=str(ROOT),
                          code_source='ARTIFICIAL_TEST',computation_source='ARTIFICIAL_FIXED',
                          python=sys.executable)

            def launch(run, problem, selected):
                launch_receipt = dict(invocation_id='artificial-large-shot')
                write(run/'LAUNCH.json', launch_receipt)
                with redirect_stdout(io.StringIO()):
                    entry.main(['--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.json'),
                                '--run-dir',str(run)])
                write(run/'EXIT.json',dict(launch_receipt,returncode=0))

            gateway=Gateway(config,preflight=lambda *a:dict(status='PASS',artificial=True),launch=launch)
            scope = dict(methods=[method],scaled_shots=dict(M=[12],grid=dict(min_k=16,max_k=16,divisor=4),
                                                           compare_paper=False))
            with patch.object(O,'prepare',return_value=context):
                result = gateway.handle(dict(action='start',instruction='Artificial finite support',scope=scope))
                run = Path(result['run_directory'])
                outcomes = {p: p.read_bytes() for p in run.glob('jobs/*/*/OUTCOME.json')}
                self.assertTrue(outcomes)
                large = [load(p) for p in outcomes if load(p)['identity']['allocation'][0]['N_X'] == 944]
                self.assertEqual(len(large), 1)
                self.assertEqual(large[0]['status'], O.PASS)
                self.assertEqual(large[0]['evaluator'], 'RPE_FINITE_SUPPORT_MSE_V1')
                gateway.handle(dict(action='resume', run_id=result['run_id'], instruction='Artificial finite support'))
                self.assertEqual(outcomes, {p:p.read_bytes() for p in run.glob('jobs/*/*/OUTCOME.json')})

    def test_ceil_exact_even_around_integer_and_aliases(self):
        self.assertEqual(S.base_shots(11), list(range(55, 10, -4)))
        self.assertEqual(S.base_shots(12), list(range(59, 10, -4)))
        self.assertEqual(S.base_shots(13), list(range(63, 10, -4)))
        self.assertEqual(S.dyadic_count(64, -12, 4), 8)
        self.assertEqual(S.dyadic_count(64, -11, 4), 10)
        self.assertEqual(S.paper_counts([59,11], F(2)), [161,30])
        self.assertEqual(S.paper_counts([11], F(10**45)), [12])
        p = S.settings(dict(M=[0], grid=dict(min_k=-17,max_k=-13,divisor=4)))
        menu = S.allocation_menu(0,p)
        self.assertLess(len(menu),6)
        self.assertEqual(sum(len(rows) for rows in menu.values()),6)
        p = S.settings(dict(M=[0], grid=dict(min_k=0,max_k=0,divisor=4)))
        panel = S.allocation_menu(0,p,kappa=F(2))
        self.assertEqual([x['kind'] for x in panel[11,11]],['GRID','UNIT_REFERENCE'])
        self.assertEqual([x['kind'] for x in panel[30,30]],['PAPER_REFERENCE'])

    @staticmethod
    def candidate(cid, prices, *, M=0, kappa='4'):
        return dict(configuration_id=cid,name=cid,method='PR_PAULI',key={'L_D':3,'formula':'2nd'},M=M,
                    kappa_value=kappa,ready=True,needs=[],labels=[],prices=prices)

    @staticmethod
    def bounds(pass_):
        n=TAU/2 if pass_ else 2*TAU
        return n,n

    def test_same_kappa_several_allocations_and_comparison_excluded(self):
        # Grid shot counts are 6, 8, 11. The exact paper reference has 19.
        candidate=self.candidate('c', [1,1])
        asks=[]
        def answer(cid, vector):
            asks.append(vector)
            return self.bounds(vector[0] in (6,19))
        policy = S.settings(dict(M=[0],grid=dict(min_k=-2,max_k=0,divisor=2)))
        result=S.run('PR_PAULI',[candidate],{},O.FixtureBackend(answer,TAU),dict(scaled_shots=policy),TAU,None)
        self.assertEqual(asks,[(6,6),(19,19)])
        row,=result['by_configuration']
        self.assertEqual(row['best']['N_X'],[6])
        self.assertEqual(row['comparison_best']['N_X'],[19])
        self.assertEqual(row['minimum_status'],'GRID_MINIMUM_CONFIRMED')
        self.assertEqual({x['status'] for x in row['allocations']},{O.PASS,'COST_PRUNED'})
        pass_which=[(6,6),(8,8),(11,11)]
        flaky = [False,True,False]
        calls=[]
        def nonmonotone(cid, vector):
            calls.append(vector)
            return self.bounds(flaky[pass_which.index(vector)])
        result=S.run('PR_PAULI',[candidate],{},O.FixtureBackend(nonmonotone,TAU),
            dict(scaled_shots=dict(policy,compare_paper=False)),TAU,None)
        self.assertEqual(calls,pass_which[:2])
        self.assertEqual(result['by_configuration'][0]['best']['N_X'],[8])

    def test_user_stop_keeps_the_pending_multiplier_and_reason(self):
        p = S.settings(dict(M=[0], grid=dict(min_k=0, max_k=0, divisor=4)))
        c = self.candidate('c', [2, 2])
        result = S.run('PR_PAULI', [c], {}, None, dict(scaled_shots=p), TAU, 1,
                       flagged_stop=lambda: True)
        self.assertEqual(result['stop_reason'], 'USER_REQUESTED_STOP')
        self.assertEqual(result['pending_request']['status'], 'PENDING')
        self.assertEqual(result['new_calls'], 0)

    def test_fleet_supplies_independent_paper_comparison_before_grid_answer(self):
        policy = S.settings(dict(M=[0], grid=dict(min_k=-2, max_k=0, divisor=2)))
        search = dict(scaled_shots=policy, prefetch_independent_references=True)
        candidate = self.candidate('c', [1, 1])
        for budget, expected in ((None, [(19, 19)]), (2, [])):
            waves = []
            answer = O.FixtureBackend(lambda cid, vector: self.bounds(True), TAU)
            result = S.run('PR_PAULI', [candidate], {}, answer, search, TAU, budget,
                submit_standby=lambda first, rest: waves.append((first['vector'], [r['vector'] for r in rest])))
            self.assertEqual(waves[0], ((6, 6), expected))
            self.assertEqual(result['by_configuration'][0]['best']['N_X'], [6])

    def test_late_ready_wakeup_preserves_owned_allocation_and_adopts_it_once(self):
        first = self.candidate('a', [2, 2])
        late = self.candidate('b', [1, 1])
        policy = S.settings(dict(M=[0], grid=dict(min_k=0, max_k=0, divisor=1), compare_paper=False))
        answers = []
        class Backend:
            def answer(_, row):
                answers.append(row['configuration_id'])
                if len(answers) == 1:
                    return dict(status='RESELECT_READY', calls=0)
                return dict(status=O.PASS, MSE_lower=TAU/2, MSE_upper=TAU/2, calls=1, source='fixture')
        delivered = [False]
        def refresh():
            if answers and not delivered[0]:
                delivered[0] = True
                return [late], {}
            return [], {}
        result = S.run('PR_PAULI', [first], {}, Backend(), dict(scaled_shots=policy), TAU, None,
                       refresh_ready=refresh)
        self.assertEqual(answers, ['a', 'b', 'a'])
        self.assertEqual(result['new_calls'], 2)
        self.assertTrue(all(r['minimum_status'] == 'GRID_MINIMUM_CONFIRMED' for r in result['by_configuration']))

    def test_work_slice_keeps_due_jobs_but_explicit_limit_still_stops(self):
        policy = S.settings(dict(M=[0], grid=dict(min_k=0, max_k=0, divisor=1), compare_paper=False))
        candidates = [self.candidate(k, [1, 1]) for k in 'abcd']
        for mode, hard_limit, initial, expected in [('drain', None, 'bc', 'abc'),
                ('refill', None, 'bc', 'abcd'), ('refill', None, '', 'abdc'), ('refill', 2, 'bc', 'ab')]:
            owned = set()
            choices = []
            def supply(first, rest):
                choices.append(first['configuration_id'])
                # Two more jobs already passed input checks at the first yield.
                if len(choices) == 1:
                    owned.update(initial)
                elif mode == 'refill' and rest:
                    owned.add(rest[-1]['configuration_id'])
            search = dict(scaled_shots=policy, work_slice_steps=1, max_steps_per_method=hard_limit,
                          prefetch_independent_references=True, ready_work_at_slice=mode)
            answer = O.FixtureBackend(lambda cid, vector: self.bounds(True), TAU)
            S.run('PR_PAULI', candidates, {}, answer, search, TAU, None,
                  submit_standby=supply, free=lambda row: row['configuration_id'] in owned)
            self.assertEqual(''.join(choices), expected)

    def test_received_point_prunes_only_costlier_allocations_of_its_own_configuration(self):
        policy = S.settings(dict(M=[0], grid=dict(min_k=-2, max_k=0, divisor=2), compare_paper=False))
        first = self.candidate('a', [1, 1])
        second = self.candidate('b', [4, 4])
        pending = {('a', (8, 8)): dict(status=O.PASS, MSE_lower=TAU/2, MSE_upper=TAU/2,
                                     source='accepted', calls=0)}

        def completed(proposal_for):
            for key in tuple(pending):
                self.assertIsNotNone(proposal_for(key))
                yield key, pending.pop(key)

        asked = []

        def answer(cid, vector):
            asked.append((cid, vector))
            return self.bounds(cid == 'b')

        result = S.run('PR_PAULI', [first, second], {}, O.FixtureBackend(answer, TAU),
                       dict(scaled_shots=policy), TAU, None, read_completed=completed)
        self.assertEqual(asked, [('a', (6, 6)), ('b', (6, 6))])
        self.assertEqual(result['by_configuration'][0]['best']['N_X'], [8])
        self.assertEqual(result['by_configuration'][0]['minimum_status'], 'GRID_MINIMUM_CONFIRMED')
        self.assertEqual(result['by_configuration'][1]['best']['N_X'], [6])

    def test_off_grid_reference_costs_less_without_becoming_grid_best(self):
        p = S.settings(dict(M=[0], grid=dict(min_k=2, max_k=2, divisor=1)))
        c = self.candidate('c', [2, 2], kappa='2')
        answer = O.FixtureBackend(lambda cid, vector: self.bounds(True), TAU)
        result = S.run('PR_PAULI', [c], {}, answer, dict(scaled_shots=p), TAU, None)
        summary = result['scaled_shots']
        self.assertEqual(summary['all_M_best']['N_X'], [44])
        self.assertEqual(summary['comparison_only_best']['N_X'], [11])
        self.assertEqual(summary['comparison_only_delta_RZ'], -132)

    def test_technical_hold_and_equal_cost_smaller_multiplier_are_unresolved(self):
        a=self.candidate('zero', [0,0]);b=self.candidate('independent',[3,3], kappa='8')
        class Backend:
            def answer(self, row):
                if row['configuration_id']=='zero' and row['N_X']==[6]:
                    return dict(status='UNKNOWN',failure_kind='PRE_API_SCOPE_REJECTION',reason='COUNT_ONLY_SCOPE_EXCEEDED',calls=1)
                return dict(status=O.PASS,MSE_lower=0,MSE_upper=TAU,calls=1,source='ARTIFICIAL')
        policy=dict(M=[0],grid=dict(min_k=-2,max_k=-1,divisor=2),compare_paper=False)
        out=S.run('PR_PAULI',[a,b],{},Backend(),dict(scaled_shots=policy),TAU,None)
        rows={r['configuration_id']:r for r in out['by_configuration']}
        self.assertEqual(rows['zero']['minimum_status'],'EVALUATED_MINIMUM')
        self.assertEqual(rows['zero']['best']['N_X'],[8])
        self.assertEqual(rows['zero']['allocations'][0]['status'],'TECHNICAL_HOLD')
        self.assertEqual(rows['independent']['minimum_status'],'GRID_MINIMUM_CONFIRMED')

    def test_no_host_does_not_consume_the_work_slice_or_repeat_a_proposal(self):
        candidates = [self.candidate('a', [1, 1]), self.candidate('b', [3, 3])]
        asked = []

        class NoHost:
            def answer(self, row):
                asked.append((row['configuration_id'], row['vector']))
                return dict(status='HOST_ELIGIBILITY_HOLD', reason='NO_MATCHING_ELIGIBLE_HOST', calls=0)

        policy = dict(M=[0], grid=dict(min_k=-2, max_k=0, divisor=1), compare_paper=False)
        result = S.run('PR_PAULI', candidates, {}, NoHost(), dict(scaled_shots=policy, work_slice_steps=1), TAU, 5)
        self.assertEqual(len(asked), 6)
        self.assertEqual(len(set(asked)), len(asked))
        self.assertEqual(result['stop_reason'], 'UNRESOLVED_SCALED_SCOPE')
        self.assertEqual(result['new_calls'], 0)
        self.assertTrue(all(row['reason'] == 'NO_MATCHING_ELIGIBLE_HOST' for row in result['held']))

    def test_normal_schedule_batches_and_input_hold_resume(self):
        method = 'FULL_PF'
        for case, total_budget in [('batch', 5), ('capped', 2), ('input_hold', 5), ('host_hold', 5)]:
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix='artificial-scaled-schedule-') as tmp:
                staged = Path(tmp)
                problem, context, routes = fixture([method], configurations=1, lookahead=1)
                problem.update(problem=dict(system='ARTIFICIAL_SCALED_SHOTS', hamiltonian_data=dict(
                    expected=dict(fragment_count=10))), rpe=dict(M=1), execution_routes=routes)
                problem['accuracy']['epsilon_E_decimal'] = S.EPSILON
                context['tau'] = TAU
                context['saved'].table.clear()
                base, = context['candidates']
                base['prices'] = [F(1)] * 4
                routes[method][0]['artificial'].update(weights=[1]*3, scale='1', tau=str(TAU))
                executor = dict(scheduler='concurrent', max_concurrent_jobs=1)
                if case == 'host_hold':
                    hosts = staged/'HOST_ELIGIBILITY.json'
                    write(hosts, dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
                    executor['distributed'] = dict(port=0, host_eligibility=str(hosts))
                spec = dict(round_id='ARTIFICIAL', methods=[method], science_approved=True,
                            max_new_q3_per_method={method: total_budget}, executor=executor, search_to_rule_stop=True)
                write(staged/'PROBLEM.json', problem)
                write(staged/'ROUND.json', spec)

                def driver_factory(args):
                    # Supply only the candidate discovery and immutable inputs for this fixture.
                    d = entry.WidthDriver.__new__(entry.WidthDriver)
                    d.run, d.a = args.run_dir, args
                    d.state_file = d.run/'KAPPA_STATE.json'
                    d.state = load(d.state_file) if d.state_file.exists() else dict(cycle=0, families={}, points=[])
                    d.spec, d.problem = load(args.round), load(args.problem)
                    d.methods, d.widths = [method], {}
                    d.rule_stop = d.scaled_shots = True
                    d.live_ready_refill = False
                    d.wide = dict(per_decision=1 if case == 'host_hold' else 8)
                    d.bootstrap_families = d.complete_points = lambda: None
                    d.plan = lambda: ([], {})
                    d.context = lambda _: deepcopy(context)
                    d.completion_api = lambda: SimpleNamespace(state={'records': {}},
                        install=lambda _, scoped, records: scoped['execution_routes'])
                    return d

                invocations = []
                def launch(run, problem, active):
                    invocations.append(str(len(invocations)))
                    identity = dict(invocation_id=invocations[-1])
                    write(run/'LAUNCH.json', identity)
                    with redirect_stdout(io.StringIO()):
                        entry.main(['--problem', str(run/'PROBLEM.json'), '--round', str(run/'ROUND.json'),
                                    '--run-dir', str(run), '--batch', '1', '--max-cycles', '6'],
                                   driver_factory=driver_factory)
                    write(run/'EXIT.json', dict(identity, returncode=0))

                config = dict(problem=str(staged/'PROBLEM.json'), round=str(staged/'ROUND.json'), entry='schedule',
                              state_directory=str(staged/'state'), code_root=str(ROOT), code_source='ARTIFICIAL',
                              computation_source='ARTIFICIAL_FIXED', python=sys.executable)
                gateway = Gateway(config, preflight=lambda *args: dict(status='PASS'), launch=launch)
                policy = dict(M=[1, 2] if case == 'input_hold' else [1],
                              grid=dict(min_k=0 if case == 'input_hold' else -2, max_k=0, divisor=1),
                              compare_paper=False)
                initial = gateway.handle(dict(action='start', instruction='Artificial only',
                    scope=dict(methods=[method], scaled_shots=policy)))
                run = Path(initial['run_directory'])
                final = run/'SEARCH_TERMINATION.json'
                first_stop = load(run/'KAPPA_STATE.json')['stop']
                def launches():
                    return {p: p.read_bytes() for p in run.glob('jobs/*/*/LAUNCH.json')}
                before = launches()
                resume = dict(action='resume', instruction='Artificial resume', run_id=initial['run_id'])

                if case == 'host_hold':
                    self.assertEqual(first_stop['reason'], 'SCOPE_HELD_Q3')
                    self.assertFalse(final.exists())
                    self.assertFalse(before)
                    self.assertEqual(load(run/'KAPPA_STATE.json')['cycle'], 1)
                    rows = load(run/'ROUND_RESULT.json')['methods'][method]['held']
                    self.assertEqual(len(rows), 3)
                    self.assertTrue(all(r['reason'] == 'NO_MATCHING_ELIGIBLE_HOST' for r in rows))
                    self.assertEqual(gateway.handle(resume)['status'], 'STARTED')
                    self.assertFalse(launches())
                    from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
                    from test_rpe_distributed_controller import ControllerFailures

                    enqueue = DistributedExecutor._enqueue
                    worker = ControllerFailures()

                    def complete_on_available_host(view, proposal, demand=False):
                        task = enqueue(view, proposal, demand)
                        if task and task.status == 'QUEUED':
                            assigned, = view.rpc('/v1/poll', dict(host_id='server1', slots=1, capacity=1,
                                                                  source_sha=view.optimizer_source))['assignments']
                            self.assertEqual(assigned['job_id'], task.job.name)
                            view.rpc('/v1/complete', worker.completion(assigned))
                        return task

                    with patch.object(DistributedExecutor, '_eligibility_for', return_value=None), \
                         patch.object(DistributedExecutor, '_enqueue', complete_on_available_host):
                        self.assertEqual(gateway.handle(resume)['status'], 'STARTED')
                    self.assertEqual(len(launches()), 3)

                elif case == 'input_hold':
                    self.assertEqual(first_stop['reason'], 'SCOPE_HELD_INPUTS')
                    self.assertFalse(final.exists())
                    self.assertEqual(len(before), 1)
                    # Resume still works with a suspension marker from the older release.
                    write(final, first_stop)
                    context['candidates'].append(dict(base, name='new-inputs', M=2, prices=[F(1)]*6,
                        configuration_id=base['configuration_id']+'-M2', source_configuration_id=base['configuration_id']))
                    self.assertEqual(gateway.handle(resume)['status'], 'STARTED')
                    archived, = (run/'stop-history').glob('legacy-search-termination-*.json')
                    self.assertEqual(load(archived), first_stop)
                    self.assertEqual(len(launches()), 2)
                    self.assertEqual({p: launches()[p] for p in before}, before)
                else:
                    self.assertEqual(len(before), 2 if case == 'capped' else 3)
                self.assertEqual(load(final)['reason'],
                    'Q3_BUDGET_EXHAUSTED' if case == 'capped' else 'REQUESTED_WIDTH_RULE_EXHAUSTED')
                before = launches()
                self.assertEqual(gateway.handle(resume)['status'], 'REUSED_COMPLETED')
                self.assertEqual(launches(), before)

    def test_normal_schedule_rechecks_wait_without_a_second_start(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            driver = entry.WidthDriver.__new__(entry.WidthDriver)
            driver.run, driver.state_file = run, run/'KAPPA_STATE.json'
            driver.a = SimpleNamespace(max_cycles=None, plan_only=False)
            driver.state = dict(cycle=0, points=[], families={})
            driver.methods, driver.widths = ['FULL_PF', 'PR_DF'], {}
            driver.rule_stop = driver.scaled_shots = driver.work_conserving_search = True
            driver.spec = dict(science_approved=True, executor=dict(distributed=dict(port=0)))
            driver.bootstrap_families = driver.complete_points = lambda: None
            driver.plan = lambda: ([], {})
            completed = dict(stop_reason='SCALED_SHOT_GRID_COMPLETE', new_calls=0)
            held = dict(stop_reason='UNRESOLVED_SCALED_SCOPE', new_calls=0,
                        held=[dict(status='TECHNICAL_HOLD', reason='NO_MATCHING_ELIGIBLE_HOST')])
            driver.batch = lambda: dict(FULL_PF=completed, PR_DF=held if driver.state['cycle'] == 1 else completed)
            real_wait = driver.wait_for_recovery

            def wait(reasons, seconds):
                self.assertEqual(reasons, ['ELIGIBLE_HOST'])
                self.assertEqual(seconds, 15)
                real_wait(reasons, 0)
                self.assertEqual(load(driver.state_file)['recovery_wait']['reasons'], reasons)

            with patch.object(driver, 'wait_for_recovery', side_effect=wait) as wait_for_recovery:
                driver.loop()
            self.assertEqual(wait_for_recovery.call_count, 1)
            self.assertEqual(driver.state['cycle'], 2)
            self.assertNotIn('recovery_wait', driver.state)
            self.assertEqual(driver.state['stop']['reason'], 'REQUESTED_WIDTH_RULE_EXHAUSTED')

    def test_artificial_normal_start_queue_resume_and_legacy_pin(self):
        with tempfile.TemporaryDirectory(prefix='artificial-scaled-rpe-') as tmp:
            root=Path(tmp)
            problem, context, routes=fixture(METHODS, configurations=2, steps=100)
            problem['problem']=dict(system='ARTIFICIAL_SCALED_SHOTS',hamiltonian_data=dict(expected=dict(fragment_count=10)))
            problem['accuracy']['epsilon_E_decimal']=S.EPSILON
            problem['rpe']={'M':12}
            problem['execution_routes']=routes
            context['tau']=TAU
            context['saved'].table.clear()
            for m in METHODS:
                routes[m][0]['artificial'].update(weights=[1]*14,scale='1/1000000000',tau=str(TAU))
                a,b=[c for c in context['candidates'] if c['method']==m]
                a['prices']=[2]*24+[1,1]
                b['prices']=[1]*24+[100,100]
                a['kappa_value']='4'
                b['kappa_value']='8'
                native=dict(b,configuration_id=b['configuration_id']+'-M13',name=b['name']+'-M13',
                            source_configuration_id=b['configuration_id'],M=13,prices=[10]*28)
                context['candidates'].append(native)
            problem['search']['lookahead']=2
            spec=dict(round_id='TEMPLATE',methods=METHODS,science_approved=False,
                      max_new_q3_per_method={m:100 for m in METHODS},
                      executor=dict(scheduler='concurrent',max_concurrent_jobs=4))
            for label,data in [('PROBLEM.json',problem),('ROUND.json',spec)]:
                write(root/label,data)
            config=dict(problem=str(root/'PROBLEM.json'),round=str(root/'ROUND.json'),entry='schedule',
                state_directory=str(root/'state'),code_root=str(ROOT),code_source='OLD',
                computation_source='ARTIFICIAL_FIXED',python=sys.executable)
            old_service=root/'old.json';new_service=root/'new.json';registry=root/'releases.json'
            write(old_service,config)
            write(new_service,dict(config,code_source='NEW',scaled_shots_default=True))
            register(registry,'old',old_service,activate=True)
            launch_log=[]
            def launch(run,problem,spec):
                if spec.get('search_overrides', {}).get('rule') != S.RULE:
                    launch_log.append(('old',spec))
                    return
                launch_log.append(('new',spec))
                argv=['--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.json'),'--run-dir',str(run)]
                with redirect_stdout(io.StringIO()):
                    entry.main(argv)
            gateway=Gateway(dict(config,release_registry=str(registry)),preflight=lambda *a:dict(status='PASS',artificial=True),launch=launch)
            req=dict(action='start',instruction='Artificial only',scope=dict(methods=METHODS))
            first=gateway.handle(req)
            old_run=Path(first['run_directory'])
            old_round=(old_run/'ROUND.json').read_bytes()
            register(registry,'new',new_service,activate=True)
            preview=gateway.handle(dict(req,action='preview'))
            self.assertEqual(preview['applied']['search']['scaled_shots'],S.settings())
            selected=dict(req,scope=dict(req['scope'],scaled_shots=small_policy()))
            with patch.object(O,'prepare',return_value=context):
                result=gateway.handle(selected)
                run=Path(result['run_directory'])
                stored=load(run/'ROUND_RESULT.json')['methods']
                jobs={p:p.read_bytes() for p in run.glob('jobs/*/*/LAUNCH.json')}
                self.assertTrue(jobs)
                again=gateway.handle(dict(action='resume',instruction='Artificial resume',run_id=result['run_id']))
                self.assertEqual(again['run_id'],result['run_id'])
                self.assertEqual(jobs,{p:p.read_bytes() for p in run.glob('jobs/*/*/LAUNCH.json')})
                old=gateway.handle(dict(action='resume',instruction='Old pinned resume',run_id=first['run_id']))
                self.assertEqual(old['release']['id'],'old')
                self.assertEqual(launch_log[-1][0], 'old')
                self.assertNotIn('scaled_shots', launch_log[-1][1].get('search_overrides', {}))
                self.assertEqual((old_run/'ROUND.json').read_bytes(),old_round)
            for method, row in stored.items():
                summary=row['scaled_shots']
                self.assertEqual(summary['all_M_best']['M'],11)
                self.assertEqual(summary['M12_best']['M'],12)
                self.assertIn('_1',summary['all_M_best']['configuration_id'] and
                    next(c['source_configuration_id'] for c in row['by_configuration']
                         if c['configuration_id']==summary['all_M_best']['configuration_id']))
                self.assertTrue(summary['M12_best']['configuration_id'].endswith('_0'))
                for case in row['by_configuration']:
                    for allocation in case['allocations']:
                        self.assertEqual(len(allocation['N_X']),case['M']+1)
                        self.assertEqual(allocation['N_X'],allocation['N_Y'])
                self.assertTrue(any(g['required_rounds']==[13] for g in row['by_configuration']))
                self.assertTrue(any(g['M']==13 and g['best'] for g in row['by_configuration']))
                self.assertEqual(summary['all_M_minimum_status'],'EVALUATED_MINIMUM')
