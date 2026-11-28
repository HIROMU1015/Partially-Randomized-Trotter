"""Exception/drain, explicit local UNKNOWN and resume; artificial work only."""
from contextlib import redirect_stdout
from fractions import Fraction
import io,json,os,pathlib,shutil,sys,tempfile,threading,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import rpe_method_optimizer as O,rpe_optimizer_concurrent as Q,pr_candidate_completion as C
import test_rpe_optimizer_concurrent as F
import test_pr_width_overlap as V
import check_pr_width_search as W
import run_rpe_schedule_search as E


class FailureContinuity(unittest.TestCase):
    def scope_fixture(self,root):
        problem,context,routes=F.fixture(methods=['PR_DF'],configurations=1,lookahead=0,steps=2)
        ex=O.Executor(ROOT,root,routes,{},problem,max_concurrent=1)
        ex.state=SimpleNamespace(failure=None)
        self.addCleanup(ex.finish);view=ex.for_method('PR_DF');cid=context['candidates'][0]['configuration_id']
        proposal=dict(configuration_id=cid,vector=tuple([1]*26),cost=Fraction(26),origin={})
        job=view.job_dir(proposal);job.mkdir(parents=True)
        (job/'REQUEST.json').write_text(json.dumps(O.jsonable(view.request(proposal,job,view.route(cid)))))
        return view,job,proposal

    def test_phase_records_original_exception_immediately(self):
        with tempfile.TemporaryDirectory()as tmp:
            run=pathlib.Path(tmp)
            try:
                with E.K.CLI.round_phase(run,'ARTIFICIAL'):
                    raise ValueError('original diagnostic')
            except ValueError:
                rows=[json.loads(s)for s in(run/'RUNTIME_ERRORS.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['error_type'],'ValueError')
                self.assertIn('original diagnostic',rows[0]['traceback'])
                self.assertEqual(rows[0]['phase'],'ARTIFICIAL')
            else:self.fail('Original exception swallowed')

    def test_phase_notifies_before_error_write_and_keeps_primary_on_log_failure(self):
        from contextlib import redirect_stderr
        notified=threading.Event()
        with tempfile.TemporaryDirectory()as tmp:
            run=pathlib.Path(tmp);original=O.record_runtime_error
            def record(*a,**k):
                self.assertTrue(notified.is_set());return original(*a,**k)
            with patch.object(O,'record_runtime_error',side_effect=record):
                with self.assertRaisesRegex(ValueError,'primary error'):
                    with E.K.CLI.round_phase(run,'ARTIFICIAL',on_error=lambda error:notified.set()):
                        raise ValueError('primary error')
            phase=E.K.CLI.round_phase(run,'ARTIFICIAL_LOG_FAILURE');phase.__enter__()
            with redirect_stderr(io.StringIO()),patch.object(pathlib.Path,'open',side_effect=OSError('ARTIFICIAL log disk failure')):
                try:raise ValueError('preserve primary')
                except ValueError as error:
                    self.assertFalse(phase.__exit__(type(error),error,error.__traceback__))

    def test_method_exception_is_saved_before_other_method_drains(self):
        started=threading.Event();release=threading.Event();notified=threading.Event();caught=[]
        problem,context,routes=F.fixture(methods=['PR_PAULI','PR_DF'])
        with tempfile.TemporaryDirectory()as tmp:
            run=pathlib.Path(tmp);ex=Q.ConcurrentExecutor(ROOT,run,routes,{},problem,max_concurrent=2)
            ex.state.failure_callback=lambda error:notified.set()
            def optimize(*a,**k):
                if k['methods']==['PR_PAULI']:
                    started.set();release.wait(5);raise ValueError('secondary failure during drain')
                self.assertTrue(started.wait(2));raise RuntimeError('method failure before drain')
            def work():
                try:
                    with patch.object(O,'optimize',side_effect=optimize):Q.optimize(ROOT,problem,context=context,executor=ex)
                except BaseException as error:caught.append(error)
            thread=threading.Thread(target=work);thread.start()
            try:
                self.assertTrue(notified.wait(3));self.assertTrue(thread.is_alive())
                deadline=time.monotonic()+3
                while not (run/'RUNTIME_ERRORS.jsonl').exists():
                    if time.monotonic()>deadline:self.fail('Exception not saved before drain')
                    time.sleep(.005)
                rows=[json.loads(s)for s in(run/'RUNTIME_ERRORS.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['phase'],'METHOD_OPTIMIZE')
                self.assertIn('before drain',rows[0]['message'])
            finally:release.set();thread.join(5)
            self.assertFalse(thread.is_alive());self.assertIsInstance(caught[0],RuntimeError)
            self.assertIn('method failure before drain',str(caught[0]))

    def test_scope_rejection_requires_zero_calls_and_unchanged_request(self):
        with tempfile.TemporaryDirectory()as tmp:
            view,job,p=self.scope_fixture(pathlib.Path(tmp))
            answer=dict(status='UNKNOWN',reason='COUNT_ONLY_SCOPE_EXCEEDED',calls=0,scientific_evaluation_calls=0)
            self.assertEqual(view.scope_unknown(job,p,answer)['failure_scope'],'CONFIGURATION')
            self.assertEqual(view.scope_unknown(job,p,dict(answer,scientific_evaluation_calls=1))['failure_scope'],'GLOBAL')
            request=json.loads((job/'REQUEST.json').read_text());request['expected']['binding_id']='BAD'
            (job/'REQUEST.json').write_text(json.dumps(request))
            self.assertEqual(view.scope_unknown(job,p,answer)['failure_scope'],'GLOBAL')

    def test_only_explicit_interrupt_is_local_not_sigkill_or_numerical(self):
        with tempfile.TemporaryDirectory()as tmp:
            view,job,p=self.scope_fixture(pathlib.Path(tmp))
            answer=dict(status='UNKNOWN',reason='ADAPTER_EXIT_WITHOUT_OUTCOME',calls=1)
            for signal,scope in [(15,'CONFIGURATION'),(2,'CONFIGURATION'),(9,'GLOBAL'),(11,'GLOBAL')]:
                (job/'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=None,signal=signal)))
                self.assertEqual(view.scope_unknown(job,p,answer)['failure_scope'],scope)
            for reason in ['IDENTITY_MISMATCH:contract','NUMERICAL_GATE_FAILED','MemoryError']:
                self.assertEqual(view.scope_unknown(job,p,dict(answer,reason=reason,failure_scope='CONFIGURATION'))['failure_scope'],'GLOBAL')
            view.state.failure='RESULT_DISK_UNAVAILABLE'
            self.assertEqual(view.scope_unknown(job,p,answer)['failure_scope'],'GLOBAL')

    def test_local_unknown_continues_and_resume_never_restarts_held_candidate(self):
        problem,context,routes=F.fixture(methods=['PR_PAULI','PR_DF'],configurations=2,lookahead=0,steps=3)
        problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2',kappa_neighbourhood={'outer':'ARTIFICIAL'})
        blocked=context['candidates'][0]['configuration_id'];original=Q.ConcurrentExecutor.answer;hits=[]
        def answer(view,p,allow_new=True):
            if p['configuration_id']==blocked:
                hits.append(True)
                return dict(status='UNKNOWN',failure_scope='CONFIGURATION',reason='COUNT_ONLY_SCOPE_EXCEEDED',calls=0,
                    scientific_evaluation_calls=0,job=None,source='ARTIFICIAL',MSE_lower=None,MSE_upper=None)
            return original(view,p,allow_new)
        with tempfile.TemporaryDirectory()as tmp:
            run=pathlib.Path(tmp)
            def execute(ctx):
                ex=Q.ConcurrentExecutor(ROOT,run,routes,{},problem,max_concurrent=3,budgets={m:2 for m in routes})
                with patch.object(Q.ConcurrentExecutor,'answer',answer):return Q.optimize(ROOT,problem,context=ctx,executor=ex,max_new={m:2 for m in routes})
            result=execute(context)
            self.assertEqual(len(hits),1);self.assertEqual(result['PR_PAULI']['execution_holds'][0]['status'],'UNKNOWN')
            self.assertTrue(all(r['stop_reason']!='UNKNOWN_RESULT'for r in result.values()))
            self.assertTrue(list(run.glob('jobs/PR_DF/*/OUTCOME.json')))
            (run/'ROUND_RESULT.json').write_text(json.dumps(O.jsonable(dict(methods=result))))
            before={str(p):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')}
            _,ctx,_=F.fixture(methods=list(routes),configurations=2,lookahead=0,steps=3)
            execute(ctx)
            self.assertEqual(len(hits),1);self.assertEqual(before,{str(p):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')})

    def test_local_unknown_consumes_attempt_budget(self):
        for priority in (False,True):
            problem,context,_=F.fixture(methods=['PR_PAULI'],configurations=2,lookahead=0,steps=3)
            # Both configurations retain a cheaper proposal after the first hold.
            prices=list(context['candidates'][0]['prices'])
            for candidate in context['candidates']:candidate['prices']=list(prices)
            calls=[]
            class Backend:
                def answer(self,proposal):
                    calls.append(proposal)
                    return dict(status='UNKNOWN',failure_scope='CONFIGURATION',reason='COUNT_ONLY_SCOPE_EXCEEDED',
                        scientific_evaluation_calls=0,calls=1,job=None,source='ARTIFICIAL',MSE_lower=None,MSE_upper=None)
            if priority:
                problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2',kappa_neighbourhood={'outer':'ARTIFICIAL'})
                result=O.optimize(ROOT,problem,context=context,backend=Backend(),max_new=1)['PR_PAULI']
            else:
                known=O.initial_knowledge(context['saved'],context['candidates'])
                result=O.run_method('PR_PAULI',context['candidates'],known,Backend(),problem['search'],context['tau'],context['generate'],1)
            self.assertEqual(len(calls),1);self.assertEqual(result['new_calls'],1)
            self.assertEqual(result['stop_reason'],'Q3_BUDGET_EXHAUSTED')

    def test_scope_finishes_independent_slices_with_unresolved_q3_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = E.WidthDriver.__new__(E.WidthDriver)
            driver.run = pathlib.Path(tmp)
            driver.a = SimpleNamespace(max_cycles=3, plan_only=False)
            driver.state = dict(cycle=0, points=[], families={})
            driver.spec = dict(science_approved=True)
            driver.rule_stop = True
            driver.widths = {}
            driver.bootstrap_families = driver.complete_points = driver.save = lambda: None
            driver.log = lambda **kwargs: None
            driver.plan = lambda: ([], {})
            attempt = dict(method='PR_PAULI', job_id='job', attempt_id='attempt', host_id='wsl')
            calls = []

            def batch():
                calls.append(True)
                return {'PR_PAULI': dict(stop_reason='NO_CANDIDATE_LEFT_IN_RANGE', new_calls=0,
                                         executor_metrics=dict(unresolved_q3_attempts=[attempt])),
                        'TRUNCATED_PF': dict(stop_reason='WORK_SLICE_YIELDED' if len(calls) == 1
                                             else 'NO_CANDIDATE_LEFT_IN_RANGE', new_calls=1)}

            driver.batch = batch
            driver.loop()
            self.assertEqual(len(calls), 2)
            self.assertEqual(driver.state['stop']['reason'], 'SCOPE_HELD_Q3')
            self.assertEqual(driver.state['stop']['unresolved_q3_attempts'], [attempt])

    def test_stage_boundary_yield_reuses_success_and_never_retries_failure(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=pathlib.Path(tmp);comp=C.Completion.__new__(C.Completion);comp.source='ARTIFICIAL';comp.settings={};events=[]
            flag=[False];comp.stop_requested=lambda:flag[0];comp.log=lambda **row:events.append(row)
            cmd=[sys.executable,'-c','print("artificial stage")']
            with patch.object(C,'ROOT',root):
                started=root/'ARTIFICIAL_CHILD_STARTED';release=root/'ARTIFICIAL_CHILD_RELEASE'
                wait_code='from pathlib import Path; import time; Path('+repr(str(started))+').touch(); '+\
                    '\nwhile not Path('+repr(str(release))+').exists(): time.sleep(.005)'
                results=[]
                child=threading.Thread(target=lambda:results.append(comp.call([sys.executable,'-c',wait_code],root/'stages','FIRST',science=True)))
                child.start();deadline=time.monotonic()+3
                try:
                    while not started.exists():
                        if time.monotonic()>deadline:self.fail('Artificial child did not start')
                        time.sleep(.005)
                    flag[0]=True;self.assertTrue(child.is_alive())
                finally:release.touch();child.join(4)
                self.assertFalse(child.is_alive());self.assertEqual(results[0]['returncode'],0)
                receipt=(root/'stages/FIRST.RETURN.json').read_bytes();flag[0]=True
                self.assertEqual(comp.call(cmd,root/'stages','FIRST')['returncode'],0)
                with self.assertRaises(C.PreparationYield):comp.call(cmd,root/'stages','NEXT',science=True)
                self.assertFalse((root/'stages/NEXT.STARTED.json').exists())
                flag[0]=False;comp.call(cmd,root/'stages','NEXT',science=True)
                self.assertEqual(receipt,(root/'stages/FIRST.RETURN.json').read_bytes())
                C.put(root/'stages/FAILED.RETURN.json',{'returncode':2})
                with self.assertRaisesRegex(C.Held,'PRIOR_STAGE_FAILED_NO_RETRY'):comp.call(cmd,root/'stages','FAILED',science=True)
            starts=[r for r in events if r['event']=='STAGE_START']
            self.assertEqual([r['stage']for r in starts],['FIRST','NEXT'])

    def test_completion_request_is_immutable_and_historical_key_is_preserved(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=pathlib.Path(tmp);comp=C.Completion.__new__(C.Completion)
            comp.directory=root;comp.lock=threading.RLock();comp.settings={};comp.state={'records':{}}
            comp.state_path=root/'STATE.json'
            comp.save=lambda:None;comp.log=lambda **row:None
            def resolve(request,directory):
                request['tail_descriptor']={'fixture':'ARTIFICIAL_DESCRIPTOR'}
                raise C.PreparationYield('NEXT_STAGE')
            comp.pauli_configuration=resolve
            request={'method':'PR_PAULI','selection':{},'design':{'L_D':1}}
            original=json.loads(json.dumps(request));key=C.key(request)
            with patch.object(C,'ROOT',root):
                one=comp.complete(request);two=comp.complete(request)
                self.assertEqual(request,original);self.assertEqual(one['key'],key);self.assertEqual(two['key'],key)
                self.assertEqual(len(comp.state['records']),1)
                self.assertEqual(comp.state['records'][key]['request'],original)
                historical=json.loads(json.dumps(original))
                historical['tail_descriptor']={'fixture':'ARTIFICIAL_DESCRIPTOR'}
                # Emulate old persisted post-resolution data explicitly; new
                # records remain immutable under the original stable key.
                comp.state['records'][key]['request']=json.loads(json.dumps(historical))
                self.assertNotEqual(C.key(historical),key)
                preserved=comp.complete(historical)
                self.assertEqual(preserved['key'],key);self.assertEqual(len(comp.state['records']),1)
                comp.state['records'][key]['status']='HELD_WITH_REASON'
                comp.state['records'][key]['reason']='PRIOR_STAGE_FAILED_NO_RETRY:ARTIFICIAL'
                comp.ingest_price_results=lambda:None;comp.refresh_price_hold=lambda record,**kwargs:None
                comp.price_registry_digest=lambda:'fixture'
                held=comp.complete(historical)
                self.assertEqual(held['reason'],'PRIOR_STAGE_FAILED_NO_RETRY:ARTIFICIAL')
                # Python equality merges signed zeros; request serialization must not.
                comp.state['records'][key]['request']['boundary_time']=-0.0
                changed=dict(historical,boundary_time=0.0)
                other=comp.complete(changed)
                self.assertNotEqual(other['key'],key)
                self.assertEqual(comp.state['records'][key]['reason'],'PRIOR_STAGE_FAILED_NO_RETRY:ARTIFICIAL')

    def test_normal_cli_exception_notifies_active_preparation_and_preserves_stop(self):
        active=threading.Event();seen=[]
        Base,_=V.classes(True)
        class Driver(Base):
            def completion_api(self):
                comp=super().completion_api()
                if not getattr(comp,'failure_test_hook',False):
                    original=comp.complete
                    def complete(request):
                        if W.KS._frac(request['design']['delta_pf'])!=Fraction(.1):
                            active.set();deadline=time.monotonic()+4
                            while not (self.run/'RUNTIME_ERRORS.jsonl').exists():
                                if time.monotonic()>deadline:raise AssertionError('Exception hidden until preparation exits')
                                time.sleep(.005)
                            seen.append(dict(error_already_saved=True,stop_requested=comp.stop_requested()))
                        return original(request)
                    comp.complete=complete;comp.failure_test_hook=True
                return comp
        from trottertracks import rpe_priority_search as P
        original=P.run_priority
        def fail(*a,**k):
            if a[0]=='PR_PAULI':
                self.assertTrue(active.wait(3));raise RuntimeError('ARTIFICIAL_NORMAL_ENTRY_FAILURE')
            return original(*a,**k)
        with tempfile.TemporaryDirectory()as tmp:
            root=pathlib.Path(tmp);args=W.WidthSearchTests().files(root)
            spec=json.loads((root/'ROUND.json').read_text())
            spec['search_overrides'].update(requested_widths={'PR_PAULI':[.1,.2],'PR_DF':[.1,.05]})
            spec['executor']=dict(scheduler='concurrent',max_concurrent_jobs=2,dp_workers=1)
            spec['max_new_q3_per_method']={'PR_PAULI':2,'PR_DF':2};spec['overlap_ready_batch']=True
            W.write(root/'ROUND.json',spec)
            with redirect_stdout(io.StringIO()),patch.object(P,'run_priority',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'ARTIFICIAL_NORMAL_ENTRY_FAILURE'):E.main(args,driver_factory=Driver)
            self.assertEqual(seen,[dict(error_already_saved=True,stop_requested=True)])
            run=root/'run';state=json.loads((run/'KAPPA_STATE.json').read_text())
            self.assertEqual(state['stop']['reason'],'RUNTIME_EXCEPTION')
            before={str(p):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')}
            with redirect_stdout(io.StringIO()):again=E.main(args,driver_factory=Driver)
            self.assertEqual(again.state['stop']['reason'],'RUNTIME_EXCEPTION')
            self.assertEqual(before,{str(p):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')})
            self.assertFalse(any(t.name=='rpe-job-dispatch'and t.is_alive()for t in threading.enumerate()))
            output=os.environ.get('RPE_FAILURE_ARTIFACT_DIR')
            if output:
                dest=pathlib.Path(output);dest.mkdir(parents=True,exist_ok=True)
                for name in ['RUNTIME_ERRORS.jsonl','ROUND_STAGES.jsonl','DRIVER_PROGRESS.jsonl','KAPPA_STATE.json','PROGRESS.jsonl']:
                    if(run/name).exists():shutil.copyfile(run/name,dest/name)
                for name in ['PROBLEM.json','ROUND.json']:shutil.copyfile(root/name,dest/name)


if __name__=='__main__':unittest.main(verbosity=2)


def test_exact_replay_active_uses_only_explicit_owned_attempt():
    from types import SimpleNamespace
    rows={'old':{'state':'RUNNING'},'new':{'state':'QUEUED'}}
    ex=SimpleNamespace(prior_q3_holds=[{'job_id':'old','replay_exact':False},{'job_id':'new','replay_exact':True}],
        store=SimpleNamespace(row=rows.get))
    assert E.WidthDriver.exact_replay_active(ex)
    rows['new']['state']='ACCEPTED'
    assert not E.WidthDriver.exact_replay_active(ex)
