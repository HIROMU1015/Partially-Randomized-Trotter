"""Tiny artificial launch-contract regressions for the ordinary WidthDriver.

No H4 inputs, numerical leaves, Q3, signals or price fits are used. The driver
constructor is bypassed only to install synthetic family metadata; its actual
plan() and the existing wide-schedule planner run unchanged.
"""
from copy import deepcopy
from concurrent.futures import Future
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
spec=importlib.util.spec_from_file_location('_ordinary_width_driver_launch_contract',ROOT/'scripts/run_rpe_schedule_search.py')
ENTRY=importlib.util.module_from_spec(spec);spec.loader.exec_module(ENTRY)


def encoded(value):
    value=Fraction(value);return dict(numerator=value.numerator,denominator=value.denominator)


def driver(methods):
    instance=ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
    instance.wide=dict(ladder_ratio='65536',max_kappa='4',explore_ratio='2',window='3/2',per_decision=8)
    families={};points=[]
    # These are arbitrary fixture units, not molecular costs or prices.
    costs={'PR_PAULI':[50,40,55],'PR_DF':[130,120,135]}
    for method in methods:
        key=method+'|ARTIFICIAL'
        families[key]=dict(status='READY_METADATA',method=method,L_D=1,formula='2nd',K=2,
            requested_delta_pf=encoded(Fraction(1,8)),rows=[dict(m=0,c=Fraction(1),q=1,J=1)])
        for k,cost in zip((1,2,4),costs[method]):
            points.append(dict(family=key,method=method,L_D=1,kappa=str(k),status='EVALUATED',best=encoded(cost)))
    instance.state=dict(cycle=0,families=families,points=points,history=[])
    return instance


def proposals(instance):
    rows,report=instance.plan()
    return sorted((p['family'],str(p['kappa']),p['reason'])for p in rows)


class IndependentMethodLaunchContract(unittest.TestCase):
    def test_unlimited_wave_yields_past_old_cap_and_stops_on_held_scope(self):
        instance=driver(['PR_DF'])
        with tempfile.TemporaryDirectory() as tmp:
            instance.rule_stop=True
            instance.spec=dict(science_approved=True,max_new_q3_per_method={'PR_DF':'UNLIMITED'})
            instance.a=SimpleNamespace(max_cycles=None,plan_only=False)
            instance.run=Path(tmp);instance.log=lambda **row:None;instance.save=lambda:None
            instance.bootstrap_families=lambda:None;instance.complete_points=lambda:None
            instance.plan=lambda:([],{})
            instance.state['points'].append(dict(status='HELD',reason='artificial unsupported input'))
            calls=[]
            def batch():
                calls.append(True)
                instance.state['q3_launched']={'PR_DF':16+len(calls)}
                return {'PR_DF':dict(stop_reason='WORK_SLICE_YIELDED' if len(calls)==1 else 'NO_CANDIDATE_LEFT_IN_RANGE',new_calls=1)}
            instance.batch=batch
            instance.loop()
            self.assertEqual(len(calls),2)
            self.assertEqual(instance.state['stop']['reason'],'SCOPE_HELD_INPUTS')

    def test_distributed_family_preparation_reenters_the_planner(self):
        instance=ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        instance.run=Path(temporary.name)
        family=dict(status='FAMILY_PREPARATION_PENDING')
        instance.state=dict(cycle=0,families={'fixture':family},points=[])
        instance.a=SimpleNamespace(max_cycles=2,plan_only=False)
        instance.spec=dict(science_approved=True,executor={'distributed':{'preparation':{'enabled':True}}})
        instance.log=lambda **kwargs:None;instance.save=lambda:None
        instance.bootstrap_families=lambda:self.fail('Descriptor preparation must wait for the shared executor')
        planned=[]
        def plan():
            planned.append(family['status'])
            return [],{}
        def batch(prepare_during):
            self.assertTrue(prepare_during)
            family['status']='READY_METADATA'
            return {'PR_DF':dict(stop_reason='NO_CANDIDATE_LEFT_IN_RANGE',new_calls=0)}
        instance.plan=plan;instance.batch=batch;instance.loop()
        self.assertEqual(planned,['FAMILY_PREPARATION_PENDING','READY_METADATA'])

    def test_used_q3_budget_stops_additional_preparation_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            instance=ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
            instance.run=Path(directory)
            instance.state=dict(cycle=0,families={},points=[dict(status='WAITING_INPUTS')])
            instance.a=SimpleNamespace(max_cycles=2,plan_only=False)
            instance.spec=dict(science_approved=True,max_new_q3_per_method={'PR_DF':1})
            instance.log=lambda **kwargs:None;instance.save=lambda:None
            instance.bootstrap_families=lambda:None;instance.complete_points=lambda:None
            instance.plan=lambda:([dict(fixture=True)],{})
            def batch():
                instance.state['q3_launched']={'PR_DF':1}
                return {'PR_DF':dict(stop_reason='Q3_BUDGET_EXHAUSTED',new_calls=1)}
            instance.batch=batch;instance.loop()
            self.assertEqual(instance.state['cycle'],1)
            self.assertEqual(instance.state['stop']['reason'],'Q3_BUDGET_EXHAUSTED')

    def test_loop_continues_deferred_completion_but_does_not_spin_on_waiting_inputs(self):
        for waiting in(False,True):
            with self.subTest(waiting=waiting):
                instance=ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
                temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
                instance.run=Path(temporary.name)
                instance.state=dict(cycle=0,families={},points=[dict(status='WAITING_INPUTS'if waiting else'COMPLETION')for _ in range(32)])
                instance.a=SimpleNamespace(max_cycles=2,plan_only=False);instance.spec=dict(science_approved=True)
                instance.log=lambda **kwargs:None;instance.save=lambda:None
                instance.bootstrap_families=lambda:None;instance.plan=lambda:([],{})
                batches=[]
                def complete():
                    for p in [p for p in instance.state['points']if p['status']=='COMPLETION'][:16]:p['status']='REGISTERED'
                def batch():
                    batches.append(True)
                    for p in instance.state['points']:
                        if p['status']=='REGISTERED':p.update(status='EVALUATED',best=None,promising=0)
                    return {'PR_DF':dict(stop_reason='STEP_LIMIT',new_calls=0)}
                instance.complete_points=complete;instance.batch=batch;instance.loop()
                if waiting:
                    self.assertEqual(len(batches),1)
                    self.assertEqual(instance.state['stop']['reason'],'WAITING_FOR_INPUTS')
                else:
                    self.assertEqual(len(batches),2)
                    self.assertTrue(all(p['status']=='EVALUATED'for p in instance.state['points']))
                    self.assertEqual(instance.state['stop']['reason'],'REQUESTED_WIDTH_RULE_EXHAUSTED')

    def test_222_pending_points_dispatch_bounded_fair_waves_and_preserve_the_rest(self):
        methods=('PR_PAULI','PR_DF');instance=driver(methods);calls=[];events=[]
        instance.state['points']=[]
        for method in methods:
            for f in range(6):
                family=method+'|LD%d|WIDTH%d'%(f//2+1,f%2)
                for k in range(19 if f<3 else 18):
                    instance.state['points'].append(dict(method=method,family=family,kappa=str(2**k),
                        status='COMPLETION',completion_request=dict(key='%s:%d'%(family,k))))
            instance.state['points'].append(dict(method=method,family=method+'|LD1|WIDTH0',kappa='1/3',
                status='EVALUATED',best=encoded(100 if method=='PR_PAULI'else 200)))
        held=dict(method=methods[0],family='HELD',kappa='9',status='HELD',completion_request=dict(key='DO_NOT_RETRY'))
        instance.state['points'].append(held)
        original={p['completion_request']['key']:deepcopy(p)for p in instance.state['points']if p['status']=='COMPLETION'}
        self.assertEqual(len(original),222)
        class Completion:
            def plan(self,request):
                calls.append(request['key'])
                return dict(status='READY_FOR_SEARCH',key=request['key'],configuration_id=request['key'])
            def complete(self,*args):raise AssertionError('No scientific provider in technical fixture')
        instance.methods=list(methods);instance.a=SimpleNamespace(plan_only=True)
        instance.spec={};instance.log=lambda **event:events.append(event)
        instance.completion_api=lambda:Completion();instance.check_completed_configuration=lambda *args:None
        instance.complete_points()
        selected=[p for p in instance.state['points']if p['status']=='REGISTERED']
        self.assertEqual(len(calls),16)
        for method in methods:
            mine=[p for p in selected if p['method']==method]
            self.assertEqual(len(mine),8)
            self.assertEqual(len({p['family']for p in mine}),6)
        remaining=[p for p in instance.state['points']if p['status']=='COMPLETION']
        self.assertEqual(len(remaining),206)
        for p in remaining:self.assertEqual(p,original[p['completion_request']['key']])
        self.assertEqual(held['status'],'HELD')
        instance.complete_points()  # Same cycle never exceeds its existing limit.
        self.assertEqual(len(calls),16)
        instance.state['cycle']+=1;instance.complete_points()
        self.assertEqual(len(calls),32)
        self.assertEqual(len(set(calls)),32)
        self.assertEqual(sum(p['status']=='COMPLETION'for p in instance.state['points']),190)
        starts=[e for e in events if e['event']=='PREPARATION_WAVE_START']
        self.assertEqual(starts[0]['selected_by_method'],{m:8 for m in methods})
        self.assertEqual(starts[0]['deferred'],206)

    def test_preparation_supplies_both_widths_within_the_existing_wave_budget(self):
        for enabled in (False, True):
            with self.subTest(work_conserving_search=enabled):
                d = ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
                d.work_conserving_search = enabled
                d.methods = ['PR_DF']
                d.wide = dict(per_decision=2)
                d.a = SimpleNamespace(plan_only=True)
                d.spec = {}
                d.log = lambda **event: None
                d.check_completed_configuration = lambda *args: None
                widths = dict(A=.1, B=.1, C=.05)
                d.state = dict(cycle=1, families={f: dict(requested_delta_pf=w) for f, w in widths.items()},
                               points=[dict(method='PR_DF', family=f, kappa='1', status='COMPLETION',
                                            completion_request=dict(key=f)) for f in widths])
                selected = []

                class Completion:
                    def plan(self, request):
                        key = request['key']
                        selected.append(key)
                        return dict(status='READY_FOR_SEARCH', key=key, configuration_id=key)

                d.completion_api = Completion
                d.complete_points()
                d.complete_points()
                self.assertEqual(selected, ['A', 'C'] if enabled else ['A', 'B'])

    def test_live_ready_runs_independent_df_splits_in_parallel_and_skips_held_prices(self):
        from trottertracks.pr_candidate_completion import key

        d=ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
        d.work_conserving_search=False
        d.methods=['PR_DF','PR_PAULI'];d.wide=dict(per_decision=8);d.a=SimpleNamespace(plan_only=False)
        d.spec=dict(science_approved=True,candidate_completion=dict(parallel=2))
        d.log=lambda **event:None;d.check_completed_configuration=lambda *args:None
        splits=dict(df1=1,df1_other_width=1,df2=2,df_held=3,pauli=4,pauli2=5)
        d.state=dict(cycle=1,families={family:dict(L_D=split) for family,split in splits.items()},
                     points=[dict(method='PR_PAULI' if family.startswith('pauli') else 'PR_DF',family=family,kappa='1',
                                  status='WAITING_INPUTS' if family=='df_held' else 'COMPLETION',
                                  completion_request=dict(family=family)) for family in splits])
        started=threading.Barrier(4)
        first_finished=threading.Event()
        calls=[]

        class Completion:
            def parallel_pauli_preparation(self):
                return True

            def price_deferred_keys(self):
                return {key(dict(family='df_held'))}

            def complete(self,request):
                family=request['family'];calls.append(family)
                if family in ('df1','df2','pauli','pauli2'):
                    started.wait(timeout=5)
                    if family=='df1':first_finished.set()
                elif family=='df1_other_width':
                    assert first_finished.is_set()
                else:
                    raise AssertionError('The unchanged price hold must not redispatch')
                return dict(status='READY_FOR_SEARCH',key=family,configuration_id=family)

        d.completion_api=Completion
        ready=[]
        d.complete_points(announce_ready=lambda method,record:ready.append(record['key']))
        self.assertCountEqual(ready,['df1','df2','df1_other_width','pauli','pauli2'])
        self.assertLess(calls.index('df1'),calls.index('df1_other_width'))
        d.complete_points(announce_ready=lambda *_:self.fail('No duplicate preparation'))
        self.assertEqual(len(calls),5)

    def test_continuous_preparation_refills_a_free_lane_while_a_peer_is_waiting(self):
        d = ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
        d.work_conserving_search = True
        d.methods = ['PR_DF']
        d.wide = dict(per_decision=2)
        d.a = SimpleNamespace(plan_only=False)
        d.spec = dict(science_approved=True, candidate_completion=dict(parallel=2))
        d.log = lambda **event: None
        d.check_completed_configuration = lambda *args: None
        splits = {'slow': 7, 'first': 8, 'next': 9, 'shared': 7}
        d.state = dict(cycle=1, families={family:dict(L_D=split, requested_delta_pf={'numerator':1,'denominator':1})
                                          for family,split in splits.items()},
                       points=[dict(method='PR_DF', family=family, kappa='1', status='COMPLETION',
                                    completion_request=dict(family=family)) for family in splits])
        slow_started = threading.Event()
        released = threading.Event()
        calls = []

        class Completion:
            def complete(self, request):
                family = request['family']
                calls.append(family)
                if family == 'slow':
                    slow_started.set()
                    if not released.wait(5):
                        raise AssertionError('Independent candidate did not refill the free lane')
                elif family == 'next':
                    assert slow_started.is_set() and not released.is_set()
                    released.set()
                elif family == 'shared':
                    assert released.is_set()
                return dict(status='READY_FOR_SEARCH', key=family, configuration_id=family)

        d.completion_api = Completion
        stop = threading.Event()
        executor = SimpleNamespace(state=SimpleNamespace(cv=threading.Condition(), failure=None),
                                   _user_stop_requested=stop.is_set, for_method=lambda _:d,
                                   has_spare_slot=lambda **kw:self.fail('Preparation waited for a Q3 slot'))
        announce, wait_for_preparation, _ = d.live_ready_source(executor, None, None, stop, continuous=True)
        d.complete_points(announce_ready=announce, wait_for_preparation=wait_for_preparation, continuous=True)
        self.assertCountEqual(calls, splits.keys())
        self.assertLess(calls.index('first'), calls.index('next'))
        self.assertLess(calls.index('next'), calls.index('shared'))
        self.assertEqual(d.state['cycle'], 1)

    def test_pf_wave_notifies_selector_without_waiting_for_another_method(self):
        d = driver(['PR_DF'])
        d.methods = ['PR_DF', 'TRUNCATED_PF']
        d.log = lambda **event: None
        executor = SimpleNamespace(state=SimpleNamespace(cv=threading.Condition()))
        _, _, ready = d.live_ready_source(executor, None, {'methods': {}}, threading.Event())
        cid = 'pf-ready'
        method = 'TRUNCATED_PF'
        old = dict(method=method, configuration_id=cid, ready=False)
        pf = dict(old, ready=True)
        view = SimpleNamespace(routes={}, expected={}, for_method=lambda _: trial)
        trial = SimpleNamespace(route=lambda _: {'kind': 'artificial'}, refresh_execution_hold=lambda _: False,
                                expected_for=lambda identity, _: identity)
        original = dict(candidates=[old], expected={}, saved=SimpleNamespace(table={}))
        pending, take = ready.subscribe(method, original, [old], view)
        future = Future()
        future.add_done_callback(ready.announce_pf)
        self.assertFalse(pending())
        future.set_result((dict(candidates=[pf], expected={cid: cid},
                                saved=SimpleNamespace(table={(cid, (1, 1)): 'saved'})), {}, {}))
        self.assertTrue(ready.has_new(method))
        self.assertFalse(ready.has_new('PR_DF'))
        self.assertTrue(pending())
        added, saved = take()
        self.assertEqual([c['configuration_id'] for c in added], [cid])
        self.assertEqual(saved.table, {(cid, (1, 1)): 'saved'})
        self.assertEqual(view.expected, {cid: cid})
        self.assertEqual(original['candidates'], [old])
        self.assertEqual(original['saved'].table, {})
        self.assertFalse(ready.has_new(method))
        self.assertEqual(take(), ([], None))

    def test_joint_method_planning_matches_independent_method_refinements(self):
        methods=('PR_PAULI','PR_DF')
        independent=[]
        for method in methods:
            result=proposals(driver([method]))
            self.assertTrue(any(reason.startswith('REFINE')for _,_,reason in result),method)
            independent.extend(result)
        # Running both methods must not remove a method's refinement because a
        # different method has a lower incumbent. Within-method rules stay intact.
        joint=proposals(driver(methods))
        self.assertEqual(joint,sorted(independent))

    def test_confirmed_minimum_is_used_by_the_actual_next_plan(self):
        instance=driver(['PR_DF'])
        instance.state['points'][1]['confirmed_all_best']=encoded(119)
        _,report=instance.plan()
        self.assertEqual(ENTRY.O.frac(report['per_method']['PR_DF']['global_best']),119)

    def test_batch_finishes_then_confirms_unadopted_jobs_without_gain_cutoff(self):
        O=ENTRY.O;method='PR_DF';cid='ARTIFICIAL_ONLY_CONFIGURATION';tau=Fraction(1,100)
        with tempfile.TemporaryDirectory(prefix='width-confirm-artificial-')as tmp:
            run=Path(tmp);route=dict(kind='artificial',interpreter=sys.executable,view_commit=None)
            problem=dict(methods={method:{}},execution_routes={method:[route]},search={},accuracy={},target=dict(energy_ha=0),rpe={})
            prices=[Fraction(1,10**9)]+[Fraction(1)]*25
            candidate=dict(method=method,configuration_id=cid,name='ARTIFICIAL',key=dict(artificial=True),ready=True,
                           generated=False,prices=prices,needs=[],caps=[4]*13)
            expected={cid:dict(total_configuration_id='ARTIFICIAL_TOTAL',binding_id='ARTIFICIAL_BINDING')}
            context=dict(candidates=[candidate],expected=expected,tau=tau)
            executor=O.Executor(run,run,{method:[route]},expected,problem=problem,
                                optimizer_source='ARTIFICIAL_SOURCE',prior_run_dirs=[])
            adopted=tuple([2]*26);cheap=tuple([1]+[2]*25)
            adopted_cost=O.cost(adopted,prices);cheap_cost=O.cost(cheap,prices)
            finished=[]

            def receipt(vector,*,upper=tau/2,request_cost=None,binding='ARTIFICIAL_BINDING'):
                view=executor.for_method(method)
                proposal=dict(configuration_id=cid,vector=tuple(vector),cost=O.cost(vector,prices),origin=dict(kind='ARTIFICIAL'))
                job=view.job_dir(proposal);job.mkdir(parents=True)
                request=view.request(proposal,job,route)
                if request_cost is not None:request['cost']=O.enc(request_cost)
                nx,ny=O.split(vector)
                outcome=dict(artificial=True,status=O.PASS,MSE_lower=O.enc(0),MSE_upper=O.enc(upper),
                    identity=dict(configuration_id=cid,total_configuration_id='ARTIFICIAL_TOTAL',binding_id=binding,
                        allocation=[dict(m=m,N_X=x,N_Y=y)for m,(x,y)in enumerate(zip(nx,ny))],contract_matches=True))
                (job/'REQUEST.json').write_text(json.dumps(request))
                (job/'OUTCOME.json').write_text(json.dumps(outcome))

            def finish():
                self.assertFalse(list((run/'jobs').glob('*/*/OUTCOME.json')))
                receipt(adopted);receipt(cheap)
                receipt([2,1]+[2]*24,upper=tau*2)  # false PASS: inconsistent exact bound
                receipt([2,2,1]+[2]*23,request_cost=Fraction(0))  # wrong own price
                receipt([2,2,2,1]+[2]*22,binding='OTHER')  # wrong physical binding
                finished.append(True)

            executor.finish=finish
            best=dict(candidate='ARTIFICIAL',configuration_id=cid,configuration=candidate['key'],N_X=O.split(adopted)[0],
                      N_Y=O.split(adopted)[1],status=O.PASS,MSE_lower=Fraction(0),MSE_upper=tau/2,
                      cost=adopted_cost,cost_display=O.show(adopted_cost))
            result={method:dict(best=best,stop_reason='STEP_LIMIT',new_calls=0,pending_request=None,
                by_configuration=[dict(configuration_id=cid,best=dict(vector=list(adopted),cost=adopted_cost,
                    cost_display=O.show(adopted_cost),status=O.PASS,MSE_lower=Fraction(0),MSE_upper=tau/2),promising=0)],
                ready_candidates=[cid],not_ready_candidates=[],evaluated_count=1,
                trace=[dict(configuration_id=cid,vector=list(adopted))])}
            instance=ENTRY.WidthDriver.__new__(ENTRY.WidthDriver)
            instance.run=run;instance.problem=problem;instance.methods=[method]
            instance.live_ready_refill=False
            instance.spec=dict(execution_routes={method:[route]},executor={},science_approved=True)
            instance.widths={method:[dict(raw=encoded(Fraction(1,8)))]};instance.a=SimpleNamespace(batch=2)
            point=dict(method=method,status='REGISTERED',configuration_id=cid,completion=dict(key='RECORD'))
            instance.state=dict(points=[point]);instance.context=lambda _:context
            class Completion:
                state=dict(records={'RECORD':dict(key='RECORD',configuration_id=cid)})
                def install(self,*args):return {method:[route]}
            instance.completion_api=lambda:Completion()
            with patch.object(ENTRY.K.CLI,'make_executor',return_value=(executor,lambda *a,**kw:deepcopy(result))):
                actual=instance.batch()[method]
            self.assertEqual(finished,[True])
            self.assertEqual(adopted_cost-cheap_cost,Fraction(1,10**9))
            self.assertEqual(actual['best']['cost'],adopted_cost)
            self.assertEqual(actual['rule_adopted_best']['cost'],adopted_cost)
            self.assertEqual(actual['confirmed_all_best']['cost'],cheap_cost)
            self.assertFalse(actual['confirmed_all_best']['adopted_by_rule'])
            self.assertEqual(point['best'],cheap_cost)
            self.assertEqual(point['rule_adopted_best'],adopted_cost)
            ledger=actual['completed_ledger']
            self.assertEqual(ledger['jobs_checked'],4)
            self.assertEqual(ledger['jobs_verified'],2)
            self.assertEqual(len(ledger['refused']),1)
            self.assertTrue(any(not r['status_consistent']for r in ledger['rows']))
            self.assertTrue(any(not r['price_match']for r in ledger['rows']))
            self.assertTrue(all(len(r['outcome_sha256'])==64 for r in ledger['rows']))
            saved=json.loads((run/'CONFIRMED_MINIMA.json').read_text())
            self.assertEqual(saved['new_scientific_calls'],0)
            self.assertEqual(O.frac(saved['methods'][method]['verified_all_min']),cheap_cost)


if __name__=='__main__':unittest.main(verbosity=2)
