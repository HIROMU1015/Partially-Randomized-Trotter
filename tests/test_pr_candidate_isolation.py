"""Candidate isolation and incremental READY acceptance; synthetic workers only."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
import io,json,sys,tempfile,time,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import rpe_method_optimizer as O,rpe_optimizer_concurrent as Q,pr_candidate_completion as C
import test_rpe_optimizer_concurrent as F
import check_pr_width_search as W
import run_rpe_schedule_search as E

class Isolation(unittest.TestCase):
    def test_df_path_port_preserves_descriptor_inputs_and_exact_lambda(self):
        from trottertracks import pr_df_descriptor_extension as df
        from fractions import Fraction
        loc_a = dict(commit='a'*40, path='descriptor.json')
        loc_b = dict(commit='b'*40, path='descriptor.json')
        source = dict(path='scripts/derive_pr_df_candidate_descriptor.py', commit='194e00a8a058a7a009e186bcd03b38c7329ec2f7')
        doc = dict(schema='PR_DF_TAIL_DESCRIPTOR_V1', prefix=6, descriptor=dict(lambda_R=0.25, scalar=0),
                   execution_source=source, binding={k: 'fixture' for k in df.BINDING_KEYS},
                   input_source={'input': 'original'}, saved_index=df.INDEX, derivation_source=df.DERIVATION)
        other = deepcopy(doc)
        other['execution_source']['commit'] = 'b0c04ae1915661af8de4f34957304e2cb0e7aed9'
        other['resources'] = {'wall_seconds': 3}
        def config(loc):
            return dict(spec=dict(L_D=6, split=dict(lambda_R=Fraction(1,4), lambda_source=loc),
                                  binding=doc['binding']))
        with patch.object(E, 'read_loc', side_effect=lambda loc: doc if loc == loc_a else other):
            self.assertTrue(E.equivalent_df_descriptor(config(loc_a),config(loc_b)))
            other['input_source']['input'] = 'changed'
            self.assertFalse(E.equivalent_df_descriptor(config(loc_a),config(loc_b)))
            other['input_source'] = doc['input_source']
            other['execution_source']['commit'] = 'c'*40
            self.assertFalse(E.equivalent_df_descriptor(config(loc_a),config(loc_b)))

    def test_step_limit_continues_with_remaining_budget_while_producer_waits(self):
        import threading,test_pr_width_overlap as V
        released=threading.Event();passes=[];Base,_=V.classes(True)
        class Driver(Base):
            def completion_api(self):
                comp=super().completion_api()
                if not getattr(comp,'wait_hook',False):
                    original=comp.complete
                    def complete(req):
                        if W.KS._frac(req['design']['delta_pf'])!=W.Fraction(.1):
                            if not released.wait(8):raise AssertionError('Queue did not continue within remaining budget')
                        return original(req)
                    comp.complete=complete;comp.wait_hook=True
                return comp
        make=E.K.CLI.make_executor
        def executor(*args,**kwargs):
            ex,opt=make(*args,**kwargs)
            def optimize(*a,**k):
                passes.append(True)
                if len(passes)>=2:released.set()
                return opt(*a,**k)
            return ex,optimize
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=W.WidthSearchTests().files(root)
            spec=json.loads((root/'ROUND.json').read_text())
            spec['search_overrides'].update(requested_widths={'PR_PAULI':[.1,.2],'PR_DF':[.1,.05]},max_steps_per_method=1)
            spec['executor']=dict(scheduler='concurrent',max_concurrent_jobs=2,dp_workers=1)
            spec['max_new_q3_per_method']={'PR_PAULI':2,'PR_DF':2};spec['overlap_ready_batch']=True
            W.write(root/'ROUND.json',spec)
            with patch.object(E.K.CLI,'make_executor',side_effect=executor),redirect_stdout(io.StringIO()):d=E.main(args,driver_factory=Driver)
            self.assertTrue(released.is_set());self.assertGreaterEqual(len(passes),2)
            self.assertEqual(len(list(d.run.glob('jobs/*/*/LAUNCH.json'))),4)

    def test_legacy_local_holds_preserve_other_candidates(self):
        for status in('STALE_JOB','IN_PROGRESS','NO_ROUTE'):
            with self.subTest(status=status):
                problem,ctx,_=F.fixture(methods=['PR_PAULI'],configurations=2,lookahead=0,steps=5)
                ctx['candidates'][1]['prices']=list(ctx['candidates'][0]['prices'])
                blocked=ctx['candidates'][0]['configuration_id'];calls=[]
                class Backend:
                    def answer(self,proposal):
                        calls.append(proposal['configuration_id'])
                        if proposal['configuration_id']==blocked:return dict(status=status,calls=0,job='ARTIFICIAL',reason='review')
                        return dict(status=O.PASS,calls=1,job=None,source='ARTIFICIAL',MSE_lower=1,MSE_upper=1)
                known=O.initial_knowledge(ctx['saved'],ctx['candidates'])
                r=O.run_method('PR_PAULI',ctx['candidates'],known,Backend(),problem['search'],ctx['tau'],ctx['generate'],1)
                self.assertEqual(calls.count(blocked),1);self.assertTrue(any(c!=blocked for c in calls))
                self.assertEqual(r['held'][0]['scope'],'CONFIGURATION')
                self.assertNotIn(r['stop_reason'],O.HOLD.values())

    def test_hold_releases_only_after_existing_complete_result_guards_accept(self):
        problem,ctx,routes=F.fixture(methods=['PR_PAULI'],configurations=1)
        with tempfile.TemporaryDirectory()as tmp:
            ex=O.Executor(ROOT,Path(tmp),routes,{},problem);view=ex.for_method('PR_PAULI')
            cid=ctx['candidates'][0]['configuration_id']
            proposal=dict(configuration_id=cid,vector=tuple([1]*26),cost=O.cost([1]*26,ctx['candidates'][0]['prices']),origin={'kind':'ARTIFICIAL'})
            view.answer(proposal);ex.finish();job=view.job_dir(proposal)
            view.hold_configuration(proposal,dict(status='IN_PROGRESS',job=str(job)))
            self.assertIsNone(view.refresh_execution_hold(cid));self.assertEqual(ex.execution_holds['PR_PAULI'],{})

    def test_fatal_flag_stops_next_preparation_without_killing_current_work(self):
        from types import SimpleNamespace
        calls=[];events=[]
        d=E.WidthDriver.__new__(E.WidthDriver);d.state=dict(cycle=1,points=[dict(method='PR_DF',family='f',kappa=str(n),status='COMPLETION',completion_request={'n':n})for n in(1,2)])
        d.wide={};d.methods=['PR_DF'];d.a=SimpleNamespace(plan_only=True);d.spec={};d.log=lambda **e:events.append(e)
        class Completion:
            def plan(self,request):calls.append(request);return dict(status='READY_FOR_SEARCH',key=str(request['n']),configuration_id=str(request['n']))
        d.completion_api=lambda:Completion();d.check_completed_configuration=lambda *args:None
        d.complete_points(stop_requested=lambda:bool(calls))
        self.assertEqual(len(calls),1);self.assertEqual(d.state['points'][1]['status'],'COMPLETION')
        self.assertTrue(any(e['event']=='PREPARATION_YIELD_AFTER_FATAL'for e in events))

    def test_stale_candidate_other_method_saved_best_and_resume_no_retry(self):
        problem,ctx,routes=F.fixture(methods=['PR_PAULI','PR_DF'],configurations=2,lookahead=0,steps=3)
        problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2',kappa_neighbourhood={'outer':'ARTIFICIAL'})
        blocked=ctx['candidates'][0]['configuration_id'];original=Q.ConcurrentExecutor.answer;hits=[]
        def answer(view,proposal,allow_new=True):
            if proposal['configuration_id']==blocked:
                hits.append(True)
                return dict(status='STALE_JOB',source='ARTIFICIAL_PRIOR',reason='UNRESOLVED_ROUTE',job='ARTIFICIAL_PRIOR',calls=0)
            return original(view,proposal,allow_new)
        with tempfile.TemporaryDirectory()as tmp:
            run=Path(tmp);ex=Q.ConcurrentExecutor(ROOT,run,routes,{},problem,max_concurrent=3,budgets={m:2 for m in routes},science_approved=True)
            with patch.object(Q.ConcurrentExecutor,'answer',answer):r=Q.optimize(ROOT,problem,context=ctx,executor=ex,max_new={m:2 for m in routes})
            self.assertEqual(len(hits),1)
            self.assertEqual(r['PR_PAULI']['best']['configuration_id'],blocked)
            self.assertEqual(len(r['PR_PAULI']['execution_holds']),1)
            self.assertTrue(any(x['kind']=='SAVED_JOB_REVIEW_REQUIRED'for x in r['PR_PAULI']['input_requests']))
            self.assertTrue(all(x['stop_reason']not in('STALE_JOB_NEEDS_REVIEW','UNKNOWN_RESULT')for x in r.values()))
            jobs=list(run.glob('jobs/*/*/REQUEST.json'));self.assertTrue(jobs)
            self.assertTrue(all(json.loads(f.read_text())['configuration_id']!=blocked for f in jobs))
            self.assertTrue(any(f.parent.parent.name=='PR_DF'for f in jobs))
            (run/'ROUND_RESULT.json').write_text(json.dumps(O.jsonable(dict(methods=r))))
            launches={str(f):f.read_bytes()for f in run.glob('jobs/*/*/LAUNCH.json')}
            _,ctx2,_=F.fixture(methods=list(routes),configurations=2,lookahead=0,steps=3)
            ex2=Q.ConcurrentExecutor(ROOT,run,routes,{},problem,max_concurrent=3,budgets={m:2 for m in routes},science_approved=True)
            with patch.object(Q.ConcurrentExecutor,'answer',answer):r2=Q.optimize(ROOT,problem,context=ctx2,executor=ex2,max_new={m:2 for m in routes})
            self.assertEqual(len(hits),1)
            self.assertEqual(launches,{str(f):f.read_bytes()for f in run.glob('jobs/*/*/LAUNCH.json')})
            self.assertEqual(r2['PR_PAULI']['execution_holds'],r['PR_PAULI']['execution_holds'])

    def test_numerical_unknown_remains_fatal(self):
        problem,ctx,routes=F.fixture(methods=['PR_PAULI'],configurations=1,lookahead=0,steps=3)
        problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2')
        with tempfile.TemporaryDirectory()as tmp:
            ex=Q.ConcurrentExecutor(ROOT,Path(tmp),routes,{},problem,max_concurrent=1,budgets={'PR_PAULI':1},science_approved=True)
            with patch.object(Q.ConcurrentExecutor,'answer',return_value=dict(status='UNKNOWN',reason='NUMERICAL_GATE_FAILED',source='ARTIFICIAL',calls=0,MSE_lower=None,MSE_upper=None,job=None)):
                r=Q.optimize(ROOT,problem,context=ctx,executor=ex,max_new={'PR_PAULI':1})['PR_PAULI']
            self.assertEqual(r['stop_reason'],'UNKNOWN_RESULT');self.assertEqual(r['execution_holds'],[])

    def test_snapshot_does_not_mutate_or_save_live_producer(self):
        import threading
        c=C.Completion.__new__(C.Completion);c.lock=threading.RLock()
        c.state=dict(records={'one':dict(key='one',status='READY_FOR_SEARCH',value=1)})
        c.save=lambda:(_ for _ in()).throw(AssertionError('Producer save forbidden'))
        def install(s,context,problem,records):
            self.assertIsNot(s,c);records[0]['value']=2;s.save();return {'ARTIFICIAL':[]}
        with patch.object(C.Completion,'install',install):_,rows=c.install_snapshot({}, {},{'one'})
        self.assertEqual(rows[0]['value'],2);self.assertEqual(c.state['records']['one']['value'],1)

    def test_late_ready_q3_runs_before_slow_preparation_returns_and_budget_stays_finite(self):
        observations=[]
        class Completion(W.ArtificialCompletion):
            def complete(self,req):
                new=W.KS._frac(req['design']['delta_pf'])!=W.Fraction(.1)
                if new and req['method']=='PR_DF':
                    deadline=time.monotonic()+8
                    while True:
                        completed=False
                        for f in self.root.glob('jobs/PR_PAULI/*/OUTCOME.json'):
                            request=json.loads((f.parent/'REQUEST.json').read_text())
                            if request['configuration_id']in observations:completed=True;break
                        if completed:break
                        if time.monotonic()>deadline:raise AssertionError('Late READY was not scheduled while preparation was pending')
                        time.sleep(.01)
                self.state['price_received']=True
                r=super().complete(req)
                if new and req['method']=='PR_PAULI':observations.append(r['configuration_id'])
                return r
        class Driver(W.ArtificialDriver):
            def completion_api(self):
                if self.completion is None:self.completion=Completion(self.run)
                return self.completion
            def __init__(self,args):
                super().__init__(args)
                if self.state.get('BASE_SEEDED'):return
                self.plan();comp=self.completion_api()
                for point in self.state['points']:
                    if W.KS._frac(point['requested_delta_pf'])!=W.Fraction(.1):continue
                    r=comp.complete(point['completion_request'])
                    point.update(status='REGISTERED',configuration_id=r['configuration_id'],completion=dict(key=r['key'],status=r['status']))
                self.state['BASE_SEEDED']=True;self.save()
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=W.WidthSearchTests().files(root)+['--batch','1']
            spec=json.loads((root/'ROUND.json').read_text());spec['search_overrides']['requested_widths']={'PR_PAULI':[.1,.2],'PR_DF':[.1,.05]}
            spec['executor']=dict(scheduler='concurrent',max_concurrent_jobs=2,dp_workers=1)
            spec['max_new_q3_per_method']={'PR_PAULI':2,'PR_DF':2};spec['overlap_ready_batch']=True
            W.write(root/'ROUND.json',spec)
            with redirect_stdout(io.StringIO()):d=E.main(args,driver_factory=Driver)
            events=[json.loads(l)for l in(d.run/'DRIVER_PROGRESS.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event']=='LATE_READY_INSTALLED_DURING_PREPARATION'for e in events))
            self.assertEqual(len(list(d.run.glob('jobs/*/*/LAUNCH.json'))),4)
            launches={str(f):f.read_bytes()for f in d.run.glob('jobs/*/*/LAUNCH.json')}
            with redirect_stdout(io.StringIO()):E.main(args,driver_factory=Driver)
            self.assertEqual(launches,{str(f):f.read_bytes()for f in d.run.glob('jobs/*/*/LAUNCH.json')})

if __name__=='__main__':unittest.main(verbosity=2)
