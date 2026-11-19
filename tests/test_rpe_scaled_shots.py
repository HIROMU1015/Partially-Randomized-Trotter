"""Artificial gateway/controller/worker acceptance for the finite multiplier rule."""
from contextlib import redirect_stdout
from fractions import Fraction as F
import io
from pathlib import Path
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

TAU = F(S.EPSILON)**2
METHODS = ['FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF']


def small_policy(horizons=(11,12,13)):
    return S.settings(dict(M=list(horizons), grid=dict(min_k=-12,max_k=-11,divisor=4), compare_paper=False))


class ScaleTests(unittest.TestCase):
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
