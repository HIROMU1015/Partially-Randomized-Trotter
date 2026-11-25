import pathlib,sys,tempfile,json,unittest,io,argparse
from copy import deepcopy
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.rpe_local0_search import Gateway,RequestError,resolve,number,write,load,digest
from rpe_local0_acceptance_fixture import build,seed

def fixture_config(directory):
    p=directory/'problem.json';r=directory/'round.json'
    problem=dict(problem=dict(system='ARTIFICIAL_LOCAL0_SCOPE_ACCEPTANCE_ONLY',hamiltonian_data=dict(expected=dict(fragment_count=10))),
        accuracy=dict(epsilon_E_decimal='1'),target=dict(energy_ha=0),rpe=dict(M=12),
        methods=dict(PR_PAULI=dict(family='PR',L_D_range=dict(min=1,max_offset_from_fragment_count=-1),caps=[4]*13,generation={})),
        search=dict(decrement_steps=[1,2],large_step_headroom='9/10',seed_rules=['B_PRICE'],max_steps_per_method=10,max_new_q3_per_method=0),
        execution_routes=dict(PR_PAULI=[dict(kind='artificial',interpreter=sys.executable,configuration_ids='*',artificial=dict(weights=[1]*13,scale='1',tau='1'))]))
    spec=dict(round_id='ARTIFICIAL_BASE',methods=['PR_PAULI'],science_approved=False,max_new_q3_per_method=dict(PR_PAULI=0),initial='all',executor=dict(prior_run_dirs=[]))
    write(p,problem);write(r,spec)
    import subprocess
    return dict(problem=str(p),round=str(r),entry='method',state_directory=str(directory/'state'),code_root=str(ROOT),
        code_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),computation_source='ARTIFICIAL_FIXED_SOURCE',python=sys.executable,plan_template=False)

def request(lo,hi,budget=0):
    return dict(action='start',instruction='Artificial acceptance: specified range and start',scope=dict(methods=['PR_PAULI'],per_method=dict(PR_PAULI=dict(L_D=dict(min=lo,max=hi))),shots='auto',stop=dict(max_new_q3_per_method=budget)))

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.home=pathlib.Path(self.tmp.name);self.config=fixture_config(self.home)

    def test_missing_exit_can_resume_only_before_dispatch(self):
        import os
        import sqlite3
        from trottertracks.rpe_local0_search import restore_pre_dispatch_launch

        run = self.home/'pre-dispatch'
        previous = dict(invocation_id='completed-test-invocation')
        launched = dict(invocation_id='lost-test-invocation')
        ended = dict(previous, returncode=0)
        missed = run/'invocations'/launched['invocation_id']
        before = run/'invocations'/previous['invocation_id']
        for path, value in ((run/'LAUNCH.json', launched), (run/'NORMAL_DRIVER.json', launched),
                            (missed/'LAUNCH.json', launched), (before/'LAUNCH.json', previous),
                            (before/'EXIT.json', ended), (run/'EXIT.json', ended)):
            write(path, value)
            os.utime(path, (20, 20))
        (run/'distributed/relay').mkdir(parents=True)
        for path in ('FAST_DRIVER.lock', 'distributed/relay/RELAY.lock', 'PROGRESS.jsonl'):
            (run/path).touch()
            os.utime(run/path, (10, 10))
        with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
            db.execute('CREATE TABLE jobs(state TEXT, assigned_epoch REAL, started_epoch REAL)')
            db.execute("INSERT INTO jobs VALUES ('UNKNOWN', 30, NULL)")
        self.assertFalse(restore_pre_dispatch_launch(run, launched, ended))
        self.assertEqual(load(run/'LAUNCH.json'), launched)
        with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
            db.execute('UPDATE jobs SET assigned_epoch=5')
        (run/'STOP_REQUESTED.json').touch()
        self.assertFalse(restore_pre_dispatch_launch(run, launched, ended))
        (run/'STOP_REQUESTED.json').unlink()
        self.assertTrue(restore_pre_dispatch_launch(run, launched, ended))
        self.assertEqual(load(run/'LAUNCH.json'), previous)
        self.assertEqual(load(missed/'LAUNCH.json'), launched)
        self.assertFalse((missed/'EXIT.json').exists())
        with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
            self.assertEqual(db.execute('SELECT * FROM jobs').fetchall(), [('UNKNOWN', 5, None)])

    def test_driver_interrupt_reuses_checkpoint_and_never_restarts_after_stop(self):
        import signal
        from unittest.mock import Mock, patch
        from trottertracks.rpe_local0_search import execute_run
        from trottertracks.rpe_q3_store import JobStore

        for stopped, overlay in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(stopped=stopped, overlay=overlay):
                run = self.home/f'driver-{stopped}-{overlay}'
                problem = dict(problem=dict(system='ARTIFICIAL'))
                spec = dict(round_id='artificial', search_to_rule_stop=True,
                            search_overrides=dict(work_conserving_search=True),
                            executor=dict(distributed=dict(port=0)))
                write(run/'PROBLEM.json', problem)
                write(run/'ROUND.json', spec)
                write(run/'APPLIED_SCOPE.json', dict(entry='schedule', signature='test', scope={},
                      problem_sha256=digest(problem), round_requested_sha256=digest(spec)))
                write(run/'AUTHORIZATION.json', dict(kind='EXPLICIT_USER_START', scope_signature='test'))
                write(run/'DUPLICATE_PREFLIGHT.json', dict(status='PASS'))
                write(run/'KAPPA_STATE.json', dict(schema='RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1', round_id='artificial'))
                (run/'FAST_DRIVER.lock').touch()
                queue = JobStore(run/'distributed/QUEUE.sqlite3')
                queue.reserve('FULL_PF', 'artificial-job', dict(route=dict(kind='artificial')), 1, True)
                owner, = queue.claim('artificial-host', 1)
                if stopped:
                    write(run/'STOP_REQUESTED.json', dict(reason='USER_REQUESTED_STOP'))
                first = Mock(pid=10)
                first.wait.return_value = -signal.SIGKILL
                second = Mock(pid=11)
                second.wait.return_value = 0
                selection = (dict(source='updated-operations', driver_sha256='fixture',
                                  restart_driver_after_kill=True) if overlay else None)
                with patch('trottertracks.rpe_local0_search.fixed_implementation', return_value={}), \
                     patch('trottertracks.rpe_local0_search.schedule_driver',
                           return_value=ROOT/'scripts/run_rpe_schedule_search.py'), \
                     patch('trottertracks.rpe_local0_search.subprocess.Popen', side_effect=[first, second]) as spawn:
                    execute_run(self.config, run, 'same-invocation', schedule_resume=selection)
                recovery = load(run/'invocations/same-invocation/DRIVER_INTERRUPTION.json')
                self.assertEqual(recovery['recovery_hold'], 'STOP_OR_SEARCH_TERMINATION' if stopped else None)
                self.assertEqual(spawn.call_count, 1 if stopped else 2)
                self.assertEqual(queue.row('artificial-job')['attempt_id'], owner['attempt_id'])
                self.assertEqual(queue.counts('FULL_PF'), 1)
                if not stopped:
                    self.assertEqual(spawn.call_args_list[0], spawn.call_args_list[1])
                    self.assertEqual(load(run/'EXIT.json')['returncode'], 0)

    def test_selected_schedule_checks_queue_store_before_start(self):
        from hashlib import sha256
        from trottertracks.rpe_local0_search import schedule_driver

        run = self.home/'run'
        write(run/'APPLIED_SCOPE.json', dict(run_id=run.name, entry='schedule'))
        operations = self.home/'operations'
        driver = operations/'scripts/run_rpe_schedule_search.py'
        store = operations/'src/trottertracks/rpe_q3_store.py'
        for part in (driver, store):
            part.parent.mkdir(parents=True)
            part.write_text('# Artificial deployment\n')
        selection = dict(run_id=run.name, base_source=self.config['code_source'], root=str(operations),
                         driver_sha256=sha256(driver.read_bytes()).hexdigest(),
                         q3_store_sha256=sha256(store.read_bytes()).hexdigest())
        self.assertEqual(schedule_driver(self.config, run, selection), driver)
        store.write_text('# Changed queue store\n')
        with self.assertRaisesRegex(RequestError, 'Q3 store differs'):
            schedule_driver(self.config, run, selection)

    def test_selected_schedule_checks_pinned_optimizer_binding_before_start(self):
        from hashlib import sha256
        from trottertracks.rpe_local0_search import schedule_driver

        run = self.home/'run'
        write(run/'APPLIED_SCOPE.json', dict(run_id=run.name, entry='schedule'))
        operations = self.home/'operations'
        driver = operations/'scripts/run_rpe_schedule_search.py'
        concurrent = operations/'src/trottertracks/rpe_optimizer_concurrent.py'
        method = concurrent.with_name('rpe_method_optimizer.py')
        digest = {}
        for path in (driver, concurrent, method):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Artificial deployment\n')
            digest[path] = sha256(path.read_bytes()).hexdigest()
        selection = dict(run_id=run.name, base_source=self.config['code_source'], root=str(operations),
                         driver_sha256=digest[driver],
                         worker_trial=dict(optimizer_sha256=digest[concurrent], method_optimizer_sha256=digest[method]))
        self.assertEqual(schedule_driver(self.config, run, selection), driver)
        method.write_text('# Changed method optimizer\n')
        with self.assertRaisesRegex(RequestError, 'rpe_method_optimizer'):
            schedule_driver(self.config, run, selection)

    def test_explicit_resume_preserves_killed_controller_attempts_and_stop(self):
        import signal
        from trottertracks.rpe_q3_store import JobStore

        invocation = '8' * 32
        launched = []
        gateway = Gateway(self.config, preflight=lambda *_: {'status': 'PASS'},
                          launch=lambda run, *_: launched.append(run), resume_controller_kill=invocation)
        run = pathlib.Path(gateway.handle(dict(request(4, 6), action='prepare'))['run_directory'])
        spec = load(run/'ROUND.requested.json')
        spec['executor']['distributed'] = {}
        write(run/'ROUND.requested.json', spec)
        write(run/'ROUND.json', spec)
        scope = load(run/'APPLIED_SCOPE.json')
        write(run/'APPLIED_SCOPE.json', dict(scope, round_requested_sha256=digest(spec)))
        launch = dict(invocation_id=invocation, pid=2147483647)
        ended = dict(invocation_id=invocation, returncode=-signal.SIGKILL)
        history = run/'invocations'/invocation
        for location in (run, history):
            write(location/'LAUNCH.json', launch)
            write(location/'EXIT.json', ended)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/relay/STATUS.json', dict(stopped=True, pid=launch['pid']))
        write(run/'distributed/LOCAL_DAEMON.json', dict(pid=launch['pid']))
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        write(run/'KAPPA_STATE.json', dict(schema='RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1', round_id=spec['round_id']))
        (run/'FAST_DRIVER.lock').touch()
        queue = JobStore(run/'distributed/QUEUE.sqlite3')
        queue.reserve('PR_PAULI', 'artificial-job', dict(route=dict(kind='artificial')), 2, True)
        assignment, = queue.claim('worker', 1)
        before = queue.row('artificial-job')
        resume = dict(action='resume', run_id=run.name, instruction='Resume artificial checkpoint')
        with self.assertRaisesRegex(RequestError, 'job still has unresolved ownership'):
            gateway.handle(resume)
        queue.update('worker', 'artificial-job', assignment['attempt_id'], 'UNKNOWN',
                     dict(automatic_retry=False, resource_released=True))
        stop = run/'STOP_REQUESTED.json'
        write(stop, dict(reason='USER_REQUESTED_STOP'))
        with self.assertRaisesRegex(RequestError, 'STOP_OR_SEARCH_TERMINATION'):
            gateway.handle(resume)
        stop.unlink()
        self.assertEqual(gateway.handle(resume)['status'], 'STARTED')
        self.assertEqual(launched, [run])
        self.assertEqual(queue.row('artificial-job')['attempt_id'], before['attempt_id'])
        self.assertEqual(queue.row('artificial-job')['state'], 'UNKNOWN')
        self.assertEqual(load(history/'EXIT.json'), ended)
        saved = load(run/'stop-history'/('controller_kill-'+invocation+'.json'))
        self.assertEqual(saved['held_attempts'], [dict(job_id='artificial-job', attempt_id=before['attempt_id'])])

    def test_killed_fixed_endpoint_reattaches_original_attempt(self):
        import signal
        from trottertracks.rpe_local0_search import reconcile_controller_kill
        from trottertracks.rpe_q3_store import JobStore
        invocation='c'*32;run=self.home/'fixed-reattach';pid=2147483647
        launch=dict(invocation_id=invocation,pid=pid);ended=dict(invocation_id=invocation,returncode=-signal.SIGKILL)
        for location in (run,run/'invocations'/invocation):
            write(location/'LAUNCH.json',launch);write(location/'EXIT.json',ended)
        write(run/'NORMAL_DRIVER.json',launch)
        write(run/'ROUND.json',dict(round_id='fixed',executor=dict(distributed=dict(port=8763))))
        write(run/'KAPPA_STATE.json',dict(schema='RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1',round_id='fixed'))
        write(run/'distributed/ENDPOINT.json',dict(pid=pid,url='http://127.0.0.1:8763'))
        write(run/'distributed/relay/STATUS.json',dict(pid=pid,stopped=False))
        write(run/'distributed/LOCAL_DAEMON.json',dict(pid=pid))
        write(run/'distributed/pbs/STATE.json',dict(allocations={'a':dict(status='R',spec=dict(run_id=str(run)))}))
        (run/'FAST_DRIVER.lock').touch()
        store=JobStore(run/'distributed/QUEUE.sqlite3');request=dict(route=dict(kind='artificial'))
        store.reserve('PR_DF','original',request,2,True)
        assignment,=store.claim('worker',1)
        write(run/'jobs/PR_DF/original/REQUEST.json',request)
        receipt=reconcile_controller_kill(run,launch,ended,invocation)
        self.assertEqual(receipt['reattached_attempts'],[dict(job_id='original',attempt_id=assignment['attempt_id'],state='ASSIGNED')])
        self.assertEqual(store.row('original')['attempt_id'],assignment['attempt_id'])
        self.assertEqual(store.row('original')['state'],'ASSIGNED')
        write(run/'jobs/PR_DF/original/REQUEST.json',dict(other=True))
        with self.assertRaisesRegex(RequestError,'saved request differs'):
            reconcile_controller_kill(run,launch,ended,invocation)

    def test_unattended_recovery_preserves_start_and_waits_for_a_receipt(self):
        import signal
        from trottertracks.rpe_q3_store import JobStore

        invocation = '9' * 32
        started = []
        gateway = Gateway(self.config, preflight=lambda *_: {'status': 'PASS'},
                          launch=lambda run, *_: started.append(run))
        run = pathlib.Path(gateway.handle(dict(request(4, 6), action='prepare'))['run_directory'])
        scope = load(run/'APPLIED_SCOPE.json')
        spec = load(run/'ROUND.requested.json')
        spec['executor']['distributed'] = {'port': 39000}
        write(run/'ROUND.requested.json', spec)
        write(run/'ROUND.json', spec)
        write(run/'APPLIED_SCOPE.json', dict(scope, entry='schedule', round_requested_sha256=digest(spec)))
        original = dict(kind='EXPLICIT_USER_START', instruction='Artificial start',
                        scope_signature=scope['signature'], authorized_utc=123)
        write(run/'AUTHORIZATION.json', original)
        launch = dict(invocation_id=invocation, pid=2147483647)
        history = run/'invocations'/invocation
        write(run/'LAUNCH.json', launch)
        write(history/'LAUNCH.json', launch)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/relay/STATUS.json', dict(stopped=False, pid=launch['pid']))
        write(run/'distributed/LOCAL_DAEMON.json', dict(pid=launch['pid']))
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        write(run/'KAPPA_STATE.json', dict(schema='RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1', round_id=spec['round_id']))
        (run/'FAST_DRIVER.lock').touch()
        queue = JobStore(run/'distributed/QUEUE.sqlite3')
        queue.reserve('PR_PAULI', 'still-queued', dict(route=dict(kind='artificial')), 1, True)
        write(run/'EXIT.json', dict(invocation_id='8' * 32, returncode=-signal.SIGKILL))
        with self.assertRaisesRegex(RequestError, 'no matching killed-controller receipt'):
            gateway.resume_interrupted(run, invocation)
        ended = dict(invocation_id=invocation, returncode=-signal.SIGKILL)
        write(run/'EXIT.json', ended)
        write(history/'EXIT.json', ended)
        write(run/'STOP_REQUESTED.json', dict(reason='USER_REQUESTED_STOP'))
        with self.assertRaisesRegex(RequestError, 'explicit STOP'):
            gateway.resume_interrupted(run, invocation)
        (run/'STOP_REQUESTED.json').unlink()
        self.assertEqual(gateway.resume_interrupted(run, invocation), 'STARTED')
        with self.assertRaisesRegex(RequestError, 'one unattended resume'):
            gateway.resume_interrupted(run, invocation)
        self.assertEqual(started, [run])
        self.assertEqual(load(run/'AUTHORIZATION.json'), original)
        self.assertEqual(queue.row('still-queued')['state'], 'QUEUED')
        self.assertEqual(load(history/'EXIT.json'), ended)

    def test_technical_ast_resume_requires_ended_owners_and_preserves_stop(self):
        import sqlite3

        started = []
        invocation = '1' * 32
        gateway = Gateway(self.config, preflight=lambda *args: {'status': 'PASS'},
                          launch=lambda run, *args: started.append(run), resume_ast_failure=invocation)
        prepared = gateway.handle(dict(request(4, 6), action='prepare'))
        run = pathlib.Path(prepared['run_directory'])
        launch = dict(invocation_id=invocation, pid=2147483647)
        ended = dict(invocation_id=invocation, returncode=1)
        history = run/'invocations'/invocation
        for p in (run, history):
            write(p/'LAUNCH.json', launch)
            write(p/'EXIT.json', ended)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/relay/STATUS.json', dict(stopped=True, pid=2147483647))
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        stop = dict(reason='RUNTIME_EXCEPTION', error_type='SystemError',
                    message='AST constructor recursion depth mismatch (synthetic)')
        write(run/'KAPPA_STATE.json', dict(stop=stop))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(dict(
            stop, phase='DRIVER', traceback='pf_optimizer_connection.py'))+'\n')
        (run/'FAST_DRIVER.lock').touch()
        with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
            db.execute('CREATE TABLE jobs (state TEXT, attempt_id TEXT, detail TEXT)')
            db.execute("INSERT INTO jobs VALUES ('RUNNING','attempt','{}')")
        resume = dict(action='resume', instruction='Artificial technical restart')
        with self.assertRaisesRegex(RequestError, 'unresolved execution or recovery'):
            gateway.start(run, resume)
        self.assertEqual(started, [])
        with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
            db.execute("UPDATE jobs SET state='UNKNOWN', detail=?",
                       (json.dumps(dict(automatic_retry=False, resource_released=True)),))
        self.assertEqual(gateway.start(run, resume), 'STARTED')
        self.assertEqual(started, [run])
        self.assertEqual(load(run/'stop-history'/('ast-'+invocation+'.json')), stop)
        self.assertEqual(load(run/'EXIT.json'), ended)
        self.assertEqual(load(history/'EXIT.json'), ended)

    def test_resume_restores_archived_input_after_ownership_guard_and_before_preflight(self):
        import gzip
        from trottertracks.rpe_preparation_retention import file_sha256, restore_resume_archives

        started = []
        state = pathlib.Path(self.config['state_directory'])
        archive = state.parent/'archives/preparation-inputs/retained'
        archive.mkdir(parents=True)
        source = '1'*40
        ref = 'refs/rpe-preparation/input-bundles/inputs-' + '2'*40
        payload = b'artificial encoded pack entry'
        blob = archive/'pack-entries.bin'
        blob.write_bytes(payload)
        sha = file_sha256(blob)

        def preflight(run, problem, spec):
            self.assertEqual((run/'distributed/artifacts'/sha[:2]/sha).read_bytes(), payload)
            return {'status':'PASS'}

        gateway = Gateway(self.config, preflight=preflight, launch=lambda run,*args:started.append(run))
        prepared = gateway.handle(dict(request(4,6),action='prepare'))
        run = pathlib.Path(prepared['run_directory'])
        original = run/'jobs/PREPARATION:signal/artificial/REQUEST.json'
        cache = run/'distributed/artifacts'/sha[:2]/sha
        write(original, {'preparation':{'source':{'commit':source},'input_hashes':[],
                                        'input_git_bundles':[{'ref':ref,'sha256':sha}]}})
        write(archive/'ARCHIVE.json', {'requests':[{'request':str(original), 'request_sha256':file_sha256(original),
                                                   'artifact_sha256':sha}]})
        with gzip.open(archive/(sha+'.recipe.json.gz'),'wt') as file:
            json.dump(dict(sha256=sha,ref=ref,source=source,header='',trailer='',entries=[[0,len(payload)]]),file)
        write(archive.with_name('retained.reclaimed.json'),
              {'archive':str(archive),'bundles':{sha:{'run_id':run.name,'request':str(original),'source_path':str(cache)}}})
        resume = dict(action='resume',run_id=run.name,instruction='Artificial archived input acceptance')
        write(run/'LAUNCH.json', {'invocation_id':'artificial-invocation'})
        with self.assertRaisesRegex(RequestError, 'previous launch has no completion receipt'):
            gateway.handle(resume)
        self.assertFalse(cache.exists())
        self.assertEqual(started, [])
        write(run/'EXIT.json', {'invocation_id':'artificial-invocation','returncode':0})
        result = gateway.handle(resume)
        self.assertEqual(result['status'], 'STARTED')
        self.assertEqual(result['preflight']['preparation_archive_recovery']['restored'], [sha])
        self.assertEqual(started, [run])
        self.assertEqual(restore_resume_archives(state,run), {'restored':[], 'already_present':[sha]})

    def test_projected_dispatch_resume_preserves_unowned_queue_and_failed_invocation(self):
        import sqlite3

        started = []
        invocation = '2'*32
        gateway = Gateway(self.config, preflight=lambda *args: {'status': 'PASS'},
                          launch=lambda run, *args: started.append(run), resume_dispatch_failure=invocation)
        result = gateway.handle(dict(request(4, 6), action='prepare'))
        run = pathlib.Path(result['run_directory'])
        launch = dict(invocation_id=invocation, pid=2147483647)
        ended = dict(invocation_id=invocation, returncode=1)
        history = run/'invocations'/invocation
        for p in (run, history):
            write(p/'LAUNCH.json', launch)
            write(p/'EXIT.json', ended)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/relay/STATUS.json', dict(stopped=True, pid=2147483647))
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        stop = dict(reason='RUNTIME_EXCEPTION', error_type='KeyError', message=repr('a'*64))
        write(run/'KAPPA_STATE.json', dict(stop=stop))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(dict(stop, phase='DRIVER',
            traceback="in prefetch_standby\nsource = configurations[projection['source_configuration_id']]"))+'\n')
        (run/'FAST_DRIVER.lock').touch()
        database = run/'distributed/QUEUE.sqlite3'
        with sqlite3.connect(database) as db:
            db.execute('CREATE TABLE jobs (state TEXT, attempt_id TEXT, detail TEXT, host_id TEXT,'
                       ' assigned_epoch REAL, started_epoch REAL)')
            db.execute("INSERT INTO jobs VALUES ('QUEUED',NULL,'{}','worker',NULL,NULL)")
        resume = dict(action='resume', instruction='Artificial technical restart')
        with self.assertRaisesRegex(RequestError, 'acquired an attempt'):
            gateway.start(run, resume)
        with sqlite3.connect(database) as db:
            db.execute("UPDATE jobs SET host_id=NULL")
        self.assertEqual(gateway.start(run, resume), 'STARTED')
        self.assertEqual(started, [run])
        self.assertEqual(load(history/'EXIT.json'), ended)
        self.assertEqual(load(run/'stop-history'/('dispatch-'+invocation+'.json')), stop)
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute("SELECT state FROM jobs").fetchone(), ('QUEUED',))

    def test_dispatch_resume_after_startup_exit_reattaches_earlier_assignments(self):
        import os
        import sqlite3
        import time
        from trottertracks.rpe_q3_store import digest as request_digest

        started = []
        invocation = '3'*32
        gateway = Gateway(self.config, preflight=lambda *args: {'status': 'PASS'},
                          launch=lambda run, *args: started.append(run), resume_dispatch_failure=invocation)
        result = gateway.handle(dict(request(4, 6), action='prepare'))
        run = pathlib.Path(result['run_directory'])
        history = run/'invocations'/invocation
        launch = dict(invocation_id=invocation, pid=2147483647)
        ended = dict(invocation_id=invocation, returncode=1)
        for directory in (run, history):
            write(directory/'LAUNCH.json', launch)
            write(directory/'EXIT.json', ended)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'KAPPA_STATE.json', dict(stop=dict(reason='USER_REQUESTED_STOP')))
        write(run/'distributed/relay/STATUS.json', dict(stopped=True, pid=2147483646))
        write(run/'distributed/ENDPOINT.json', dict(pid=2147483645))
        (run/'PROGRESS.jsonl').touch()
        original = time.time()-30
        for path in ('distributed/ENDPOINT.json', 'PROGRESS.jsonl'):
            os.utime(run/path, (original, original))
        for lock in ('FAST_DRIVER.lock','distributed/relay/RELAY.lock','distributed/pbs/MANAGER.lock'):
            (run/lock).parent.mkdir(parents=True, exist_ok=True)
            (run/lock).touch()
        job = run/'jobs'/'PR_DF'/'old-job'
        old_request = dict(artificial=True)
        write(job/'REQUEST.json', old_request)
        write(job/'LAUNCH.json', dict(host_id='pbs-0', attempt_id='old-attempt'))
        database = run/'distributed/QUEUE.sqlite3'
        with sqlite3.connect(database) as db:
            db.execute('CREATE TABLE jobs (job_id,method,state,host_id,attempt_id,request_sha256,request)')
            db.execute('CREATE TABLE events (epoch REAL)')
            owned = ('old-job', 'PR_DF', 'ASSIGNED', 'pbs-0', 'old-attempt',
                     request_digest(old_request), json.dumps(old_request))
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?)', owned)
        self.assertEqual(gateway.start(run, dict(action='resume', instruction='Retry artificial bootstrap')), 'STARTED')
        self.assertEqual(started, [run])
        self.assertEqual(load(history/'EXIT.json'), ended)
        self.assertEqual(load(run/'stop-history'/('dispatch-'+invocation+'.json'))['reason'],
                         'CONTROLLER_STARTUP_EXIT_BEFORE_DISPATCH')
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute('SELECT * FROM jobs').fetchone(), owned)

    def test_pf_request_mismatch_resumes_without_retrying_the_failed_invocation_jobs(self):
        import sqlite3
        from trottertracks.rpe_q3_store import digest as queue_digest

        started = []
        invocation = '3' * 32
        gateway = Gateway(self.config, preflight=lambda *args: {'status': 'PASS'},
                          launch=lambda run, *args: started.append(run), resume_pf_request_failure=invocation)
        result = gateway.handle(dict(request(4, 6), action='prepare'))
        run = pathlib.Path(result['run_directory'])
        launch = dict(invocation_id=invocation, pid=2147483647)
        ended = dict(invocation_id=invocation, returncode=1)
        history = run/'invocations'/invocation
        for directory in (run, history):
            write(directory/'LAUNCH.json', launch)
            write(directory/'EXIT.json', ended)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/relay/STATUS.json', dict(stopped=True, pid=2147483647))
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        stop = dict(reason='RUNTIME_EXCEPTION', error_type='ValueError',
                    message='PF completion directory belongs to another selected request')
        write(run/'KAPPA_STATE.json', dict(stop=stop))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(dict(stop, phase='DRIVER',
            traceback='pf_candidate_selection.py\npf_optimizer_connection.py, in complete_selected'))+'\n')
        (run/'FAST_DRIVER.lock').touch()
        database = run/'distributed/QUEUE.sqlite3'
        with sqlite3.connect(database) as db:
            db.execute('CREATE TABLE jobs (state TEXT, attempt_id TEXT, detail TEXT, host_id TEXT,'
                       ' assigned_epoch REAL, started_epoch REAL, job_id TEXT, method TEXT,'
                       ' request_sha256 TEXT, request TEXT)')
            db.execute("INSERT INTO jobs(state,attempt_id,detail) VALUES "
                       "('ACCEPTED','done','{}'),('RUNNING','original-attempt','{}')")
        resume = dict(action='resume', instruction='Artificial PF request recovery')
        with self.assertRaisesRegex(RequestError, 'unresolved execution or recovery'):
            gateway.start(run, resume)
        with sqlite3.connect(database) as db:
            db.execute("UPDATE jobs SET state='UNKNOWN', detail=? WHERE attempt_id='original-attempt'",
                       (json.dumps(dict(automatic_retry=False, resource_released=True)),))
        from trottertracks.rpe_local0_search import reconcile_runtime_failure
        encoding = dict(stop, error_type='TypeError', message="unhashable type: 'list'")
        fault = dict(encoding, phase='DRIVER', traceback=(
            'pf_candidate_selection.py\npf_distributed_preparation.py, in complete_selected\n'
            '    consumed.update(audited)'))
        write(run/'KAPPA_STATE.json', dict(stop=encoding))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(fault)+'\n')
        self.assertEqual(reconcile_runtime_failure(run, launch, ended, invocation, 'pf_request'), encoding)
        with sqlite3.connect(database) as db:
            db.execute("UPDATE jobs SET assigned_epoch=? WHERE state='UNKNOWN'", ((history/'LAUNCH.json').stat().st_mtime,))
        with self.assertRaisesRegex(RequestError, 'acquired an attempt'):
            reconcile_runtime_failure(run, launch, ended, invocation, 'pf_request')
        with sqlite3.connect(database) as db:
            db.execute("UPDATE jobs SET assigned_epoch=NULL")
        redelivery = dict(stop, error_type='KeyError', message="'configuration_id'")
        write(run/'KAPPA_STATE.json', dict(stop=redelivery))
        fault = dict(redelivery, phase='DRIVER', traceback=(
            'pf_candidate_selection.py\npf_optimizer_connection.py, in complete_selected\n'
            "    record['configuration_id'] == candidates[0]"))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(fault)+'\n')
        job = run/'jobs/PREPARATION:compile/original-job'
        saved = dict(preparation={'stage': 'compile'})
        write(job/'REQUEST.json', saved)
        write(job/'LAUNCH.json', dict(host_id='server2', attempt_id='original-attempt'))
        with sqlite3.connect(database) as db:
            db.execute("""UPDATE jobs SET state='RUNNING', job_id='original-job',
                          method='PREPARATION:compile', host_id='server2', request_sha256=?, request=?
                          WHERE attempt_id='original-attempt'""", (queue_digest(saved), json.dumps(saved)))
        write(run/'distributed/pbs/STATE.json', dict(allocations={'pilot': {'status': 'R'}}))
        write(run/'STOP_REQUESTED.json', {'reason': 'USER_REQUESTED_STOP'})
        with self.assertRaisesRegex(RequestError, 'user STOP prevents recovery'):
            reconcile_runtime_failure(run, launch, ended, invocation, 'pf_request')
        (run/'STOP_REQUESTED.json').unlink()
        write(job/'LAUNCH.json', dict(host_id='server2', attempt_id='another-attempt'))
        with self.assertRaisesRegex(RequestError, 'saved launch owner differs'):
            reconcile_runtime_failure(run, launch, ended, invocation, 'pf_request')
        write(job/'LAUNCH.json', dict(host_id='server2', attempt_id='original-attempt'))
        self.assertEqual(reconcile_runtime_failure(run, launch, ended, invocation, 'pf_request'), redelivery)
        launcher = '/fixture/old/pf-runtime/python'
        output = run/'pf_auto/candidate/ORIGINAL_OWNER_CHECK.json'
        command = [launcher, '-B', 'check_pf_preparation_owner.py', '--output', str(output)]
        import subprocess
        scoped = dict(stop, error_type='CalledProcessError',
                      message=str(subprocess.CalledProcessError(1, command)))
        write(run/'ROUND.json', {'pf_completion': {'interpreter': launcher}})
        write(run/'KAPPA_STATE.json', dict(stop=scoped))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(dict(scoped, phase='DRIVER',
            traceback='pf_candidate_selection.py\npf_distributed_preparation.py, in complete_selected'))+'\n')
        (run/'DRIVER.log').write_text('ValueError: unrelated provider failure\n')
        with self.assertRaisesRegex(RequestError, 'outside the PF candidate request binding'):
            reconcile_runtime_failure(run, launch, ended, invocation, 'pf_request')
        (run/'DRIVER.log').write_text(f'ValueError: {str(output)!r} is not in the subpath of /fixture/old\n')
        self.assertEqual(reconcile_runtime_failure(run, launch, ended, invocation, 'pf_request'), scoped)

        # A PID disappearing in the fixed native duplicate probe is another
        # metadata failure. A separately owned preparation can still reattach.
        import hashlib
        adapter = run/'overlay/scripts/run_pf_preparation_stage.py'
        adapter.parent.mkdir(parents=True)
        adapter.write_text('# fixture metadata adapter\n')
        provider = run/'pf_auto/candidate/provider_workspace'
        out = provider/'docs/research/distributed_selected_pf/candidate/preflight.METADATA_RESULT.json'
        inp = out.with_name('preflight.METADATA_INPUT.json')
        write(inp, {'phase': 'preflight'})
        selection = dict(run_id=run.name, root=str(adapter.parents[1]), worker_trial={
            'pf_stage_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
            'pf_controller_runtime': {'interpreter': launcher}})
        selected_launch = dict(launch, command=['--schedule-resume', json.dumps(selection)])
        for directory in (run, history):
            write(directory/'LAUNCH.json', selected_launch)
        command = [launcher, '-B', str(adapter), '--provider-root', str(provider),
                   '--input', str(inp), '--output', str(out)]
        disappeared = dict(stop, error_type='CalledProcessError', message=str(subprocess.CalledProcessError(1, command)))
        write(run/'KAPPA_STATE.json', dict(stop=disappeared))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(dict(disappeared, phase='DRIVER',
            traceback="pf_candidate_selection.py\npf_distributed_preparation.py, in complete_selected\n"
                      "    source=local('preflight')"))+'\n')
        (run/'DRIVER.log').write_text('Traceback (most recent call last):\n'
            f'  File "{provider / "scripts/pf_price_stages.py"}", line 147, in active_price_jobs\n'
            'ProcessLookupError: [Errno 3] No such process\n')
        self.assertEqual(reconcile_runtime_failure(run, selected_launch, ended, invocation, 'pf_request'),
                         disappeared)
        marker = out.with_name('PF_COMPILE.DISTRIBUTED.json')
        write(marker, {'job_id': 'different'})
        with self.assertRaisesRegex(RequestError, 'outside the PF candidate request binding'):
            reconcile_runtime_failure(run, selected_launch, ended, invocation, 'pf_request')
        for directory in (run, history):
            write(directory/'LAUNCH.json', launch)
        (run/'ROUND.json').unlink()
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute("SELECT state,attempt_id FROM jobs WHERE job_id='original-job'").fetchone(),
                             ('RUNNING', 'original-attempt'))
            db.execute("UPDATE jobs SET state='UNKNOWN' WHERE job_id='original-job'")
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        write(run/'KAPPA_STATE.json', dict(stop=stop))
        fault.update(stop, traceback='pf_candidate_selection.py\npf_optimizer_connection.py, in complete_selected')
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(fault)+'\n')
        self.assertEqual(gateway.start(run, resume), 'STARTED')
        self.assertEqual(started, [run])
        self.assertEqual(load(history/'EXIT.json'), ended)
        self.assertEqual(load(run/'stop-history'/('pf_request-'+invocation+'.json')), stop)
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute('SELECT state, attempt_id FROM jobs ORDER BY rowid').fetchall(),
                             [('ACCEPTED', 'done'), ('UNKNOWN', 'original-attempt')])

    def test_detached_launcher_keeps_original_attempts_and_does_not_invent_exit(self):
        from trottertracks.rpe_q3_store import JobStore

        invocation = 'd' * 32
        gateway = Gateway(self.config)
        run = pathlib.Path(gateway.handle(dict(request(4, 6), action='prepare'))['run_directory'])
        spec = load(run/'ROUND.requested.json')
        spec['executor']['distributed'] = {}
        write(run/'ROUND.json', spec)
        # Exercise the recovery separately from the immutable requested spec.
        from trottertracks.rpe_local0_search import reconcile_controller_detachment
        launch = dict(invocation_id=invocation, pid=2147483647)
        history = run/'invocations'/invocation
        write(history/'LAUNCH.json', launch)
        write(run/'LAUNCH.json', launch)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/ENDPOINT.json', {'pid': launch['pid']})
        write(run/'distributed/relay/STATUS.json', dict(stopped=False, pid=2147483647))
        write(run/'KAPPA_STATE.json', dict(schema='RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1',
                                         round_id=spec['round_id']))
        (run/'FAST_DRIVER.lock').touch()
        store = JobStore(run/'distributed/QUEUE.sqlite3')
        req = dict(route=dict(kind='artificial'), configuration_id='ARTIFICIAL')
        store.reserve('PR_PAULI', 'test-job', req, 1, True)
        assignment, = store.claim('server2', 1)
        write(run/'jobs/PR_PAULI/test-job/REQUEST.json', req)
        write(run/'STOP_REQUESTED.json', {'reason': 'USER_REQUESTED_STOP'})
        with self.assertRaisesRegex(RequestError, 'STOP_OR_SEARCH_TERMINATION'):
            reconcile_controller_detachment(run, launch, invocation)
        (run/'STOP_REQUESTED.json').unlink()
        proof = reconcile_controller_detachment(run, launch, invocation)
        self.assertEqual(proof['reason'], 'CONTROLLER_DETACHED_WITHOUT_EXIT_RECEIPT')
        self.assertFalse((history/'EXIT.json').exists())
        self.assertEqual(store.row('test-job')['attempt_id'], assignment['attempt_id'])

    def test_status_observes_local_q3_body_before_the_worker_state_report(self):
        from trottertracks.rpe_local0_search import local0_progress
        from trottertracks.rpe_q3_store import JobStore

        run=self.home/'telemetry'
        spool=self.home/'worker'
        store=JobStore(run/'distributed'/'QUEUE.sqlite3')
        task=dict(route=dict(kind='artificial'),configuration_id='ARTIFICIAL')
        store.reserve('PR_PAULI','job',task,1,True)
        row,=store.claim('local',1)
        self.assertEqual(row['state'],'ASSIGNED')
        write(run/'distributed'/'LOCAL_WORKER.json',dict(host_id='local',work_dir=str(spool)))
        attempt=spool/'job'/row['attempt_id'];attempt.mkdir(parents=True)
        (attempt/'Q3_BODY_TIMING.jsonl').write_text('{"event":"Q3_BODY_START"}\n')
        assignment=attempt/'ASSIGNMENT.json'
        owner={key:row[key] for key in ('job_id','attempt_id','request_sha256')}
        write(assignment,dict(owner,request_sha256='not-the-assigned-request'))
        before=local0_progress({}, {}, run)
        self.assertEqual(before['stages'][0]['execution_started'],0)
        write(assignment,owner)
        observed=local0_progress({}, {}, run)
        self.assertEqual(observed['stages'][0]['execution_started'],1)
        self.assertEqual(observed['stages'][0]['execution_completed'],0)
        self.assertEqual(observed['candidates'][0]['state'],'EVALUATION_RUNNING')
        self.assertEqual(store.row('job')['state'],'ASSIGNED')

    def test_status_keeps_individual_recoveries_for_candidates_in_same_family(self):
        from trottertracks.rpe_local0_search import local0_progress

        run=self.home/'telemetry'
        wait=dict(reasons=['DF_PRICE_ORIGINAL_ATTEMPT'],recheck_epoch=1)
        recovery=dict(job_id='original',attempt_id='attempt',action='RECOVER_ORIGINAL_ATTEMPT')
        points=[
            dict(method='PR_DF',family='LD6',status='PRICE_NOT_EVALUATED',completion=dict(
                key='first',price_actions=[dict(m=0,axis='X',action='RECONCILE_ORIGINAL_ATTEMPT')])),
            dict(method='PR_DF',family='LD6',status='WAITING_INPUTS',completion=dict(
                key='second',preparation_recovery=recovery)),
        ]
        write(run/'KAPPA_STATE.json',dict(points=points,recovery_wait=wait))
        progress=local0_progress({}, {}, run)
        self.assertEqual(progress['recovery_wait'],wait)
        first,second=progress['candidates']
        self.assertEqual(first['candidate_id'],'first')
        self.assertEqual(first['price_actions'][0]['axis'],'X')
        self.assertEqual(second['candidate_id'],'second')
        self.assertEqual(second['preparation_recovery'],recovery)

    def test_start_returns_a_small_duplicate_check_summary_and_keeps_full_evidence(self):
        refused=[dict(job=str(i),reason='DIFFERENT_JOB_KEY_OR_ROUTE') for i in range(100)]
        g=Gateway(self.config,preflight=lambda *args:dict(status='PASS',refused=refused),
                  launch=lambda *args:None)
        result=g.handle(request(4,6))
        self.assertEqual(result['preflight']['refused_count'],100)
        self.assertEqual(len(result['preflight']['refused']),16)
        self.assertEqual(load(result['preflight']['report_path'])['refused'],refused)
        self.assertGreaterEqual(result['preflight']['elapsed_seconds'],0)

    def test_fixed_pbs_only_preview_and_prepare_remove_other_compute_paths(self):
        path = pathlib.Path(self.config['round'])
        spec = load(path)
        distributed = spec.setdefault('executor', {}).setdefault('distributed', {})
        distributed.update(local_worker={'host_id':'local'}, relay={'workers':[]},
            required_hosts=['local'], preparation_profiles={'local':{},'pbs-0':{}},
            pbs={'workers':[{'host_id':'pbs-0'}]})
        write(path, spec)
        gateway = Gateway(self.config)
        value = request(4, 6)
        value['scope']['execution_hosts'] = ['pbs-0']
        preview = gateway.handle(dict(value, action='preview'))
        self.assertEqual(preview['applied']['execution_hosts'], ['pbs-0'])
        gateway.handle(dict(value, action='prepare'))
        run = self.home/'state'/'runs'/preview['run_id']
        pinned = load(run/'ROUND.requested.json')['executor']['distributed']
        self.assertEqual(pinned['preparation_profiles'], {'pbs-0':{}})
        self.assertEqual(pinned['required_hosts'], [])
        self.assertNotIn('local_worker', pinned)
        self.assertNotIn('relay', pinned)

    def test_shared_pbs_walltime_is_displayed_and_pinned_at_prepare(self):
        common = pathlib.Path(self.config['round'])
        round_spec = load(common)
        profile = dict(route_profiles=[dict(route_kind='pauli', view_commit='view',
                                           profile=dict(adapter_manifest=dict(commit='adapter')))])
        round_spec['executor'] = dict(distributed=dict(
            pbs=dict(transport_source='worker', workers=[dict(host_id='pbs-0', profile=profile)])))
        write(common, round_spec)
        path = self.home/'PBS_WALLTIME.json'
        policy = dict(schema='RPE_PBS_WALLTIME_POLICY_V1', id='policy-v1',
                      computation_source=self.config['computation_source'], transport_source='worker',
                      q3=[dict(id='pauli', method='PR_PAULI', route_kind='pauli', view_commit='view',
                               adapter_source='adapter', seconds=14400)], preparation=[])
        write(path, policy)
        self.config['pbs_walltime_policy'] = str(path)
        g = Gateway(self.config)
        selected = request(4, 6)
        preview = g.handle(dict(selected, action='preview'))
        prepared = g.handle(dict(selected, action='prepare'))
        self.assertEqual(preview['run_id'], prepared['run_id'])
        self.assertEqual(preview['progress']['effective']['pbs_walltime']['q3'][0]['seconds'], 14400)
        work = pathlib.Path(prepared['run_directory'])
        pinned = load(work/'ROUND.requested.json')['executor']['distributed']['pbs']
        self.assertEqual(pinned['q3_task_seconds'][0]['seconds'], 14400)
        selected['scope']['pbs_walltime_overrides'] = dict(pauli=15000)
        override = g.handle(dict(selected, action='preview'))
        self.assertEqual(override['applied']['pbs_walltime']['q3'][0]['seconds'], 15000)
        status = g.handle(dict(action='status', run_id=prepared['run_id']))
        self.assertEqual(status['applied']['pbs_walltime']['q3'][0]['seconds'], 14400)

    def test_method_subset_keeps_its_settings_and_all_duplicate_history(self):
        selected = dict(method='PR_PAULI', run_directory='/prior/pauli')
        excluded = dict(method='TRUNCATED_PF', run_directory='/prior/pf')
        round_spec = load(self.config['round'])
        round_spec['q3_budget_parents'] = [selected, excluded]
        history = ['/prior/pauli', '/prior/pf']
        round_spec['executor']['prior_run_dirs'] = history
        selected_hold = dict(method='PR_PAULI', job_id='a'*20)
        excluded_hold = dict(method='TRUNCATED_PF', job_id='b'*20)
        round_spec['executor']['prior_q3_holds'] = [selected_hold, excluded_hold]
        round_spec['search_overrides'] = dict(requested_widths={'PR_DF': ['1/5']})
        round_spec['width_families'] = [dict(method='PR_DF', L_D=8)]
        write(self.config['round'], round_spec)

        _, scoped, _ = resolve(self.config, request(4, 6, 'UNLIMITED'))

        self.assertEqual(scoped['q3_budget_parents'], [selected])
        self.assertEqual(scoped['executor']['prior_run_dirs'], history)
        self.assertEqual(scoped['executor']['prior_q3_holds'], [selected_hold])
        self.assertEqual(scoped['width_families'], [])
        from trottertracks.rpe_schedule_search import requested_widths
        self.assertIsNone(requested_widths(load(self.config['problem']), scoped))
        self.config['entry'] = 'schedule'
        explicit = request(4, 6, 'UNLIMITED')
        explicit['scope']['per_method']['PR_PAULI']['delta_pf'] = [dict(kind='exact', value='1/7')]
        problem, scoped, _ = resolve(self.config, explicit)
        self.assertEqual(requested_widths(problem, scoped)['PR_PAULI'][0]['exact'], '1/7')

    def test_explicit_unlimited_crosses_old_cap_and_uses_rule_termination(self):
        gateway=build(self.config)
        previous=gateway.handle(request(4,4,1))
        source=pathlib.Path(previous['run_directory'])
        old=(source/'ROUND.json').read_bytes()
        current=gateway.handle(request(4,4,'UNLIMITED'))
        parent=pathlib.Path(current['run_directory'])
        applied=load(parent/'ROUND.json')
        self.assertIsNone(applied['max_steps_per_method'])
        self.assertEqual(applied['max_new_q3_per_method'],{'PR_PAULI':'UNLIMITED'})
        self.assertNotEqual(current['run_id'],previous['run_id'])
        self.assertEqual(current['preflight']['reused']['PR_PAULI'],1)
        jobs=load(self.home/'state'/'ARTIFICIAL_CALLS.json')
        self.assertGreater(sum(update['run']==str(parent) for update in jobs),1)
        self.assertEqual(load(parent/'SEARCH_TERMINATION.json')['reason'],'SEARCH_RULE_EXHAUSTED')
        self.assertEqual(gateway.handle(dict(action='resume',run_id=current['run_id'],instruction='Artificial replay'))['status'],
                         'REUSED_COMPLETED')
        self.assertEqual(jobs,load(self.home/'state'/'ARTIFICIAL_CALLS.json'))
        self.assertEqual(old,(source/'ROUND.json').read_bytes())
    def test_partial_overlap_uses_normal_entry_and_retains_unseen_point(self):
        gateway=build(self.config);old=gateway.handle(request(4,6));old_run=pathlib.Path(old['run_directory'])
        prior=seed(old_run,self.config);old_round=(old_run/'ROUND.json').read_bytes()
        current=gateway.handle(request(5,7,1));new_run=pathlib.Path(current['run_directory']);result=load(new_run/'ROUND_RESULT.json')['methods']['PR_PAULI']
        calls=load(pathlib.Path(self.config['state_directory'])/'ARTIFICIAL_CALLS.json')
        self.assertNotEqual(old['run_id'],current['run_id']);self.assertEqual(old_round,(old_run/'ROUND.json').read_bytes())
        self.assertEqual(digest(load(pathlib.Path(prior['job'])/'OUTCOME.json')),prior['outcome_sha256'])
        self.assertEqual(current['preflight']['reused']['PR_PAULI'],1)
        self.assertEqual(result['prior_rounds']['loaded'],1)
        self.assertEqual(sum(c['configuration_id']=='ARTIFICIAL_LD5' for c in calls),0)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0]['configuration_id'],'ARTIFICIAL_LD6')
        self.assertIn('ARTIFICIAL_LD7',result['ready_candidates'])
        self.assertFalse(any(e['configuration_id']=='ARTIFICIAL_LD7' for e in result['evaluated']))
        self.assertIsNotNone(result['pending_request'])
        before=len(calls);again=gateway.handle(request(5,7,1));self.assertEqual(again['run_id'],current['run_id'])
        bare=gateway.handle(dict(action='start',instruction='探索を開始して'));self.assertEqual(bare['run_id'],current['run_id'])
        self.assertEqual(len(load(pathlib.Path(self.config['state_directory'])/'ARTIFICIAL_CALLS.json')),before)
        evidence=dict(status='PASS',old_run=old['run_id'],new_run=current['run_id'],normal_entry='run_rpe_method_optimizer.round_mode',
            reused_at_preflight=current['preflight']['reused'],reused_at_normal_start=result['prior_rounds']['loaded'],same_q3_relaunches=0,
            artificial_new_calls=len(calls),unseen_candidate='ARTIFICIAL_LD7',unseen_remains=True,old_config_and_receipt_unchanged=True,
            same_scope_reuses_run=True,real_scientific_calls=0)
        write(pathlib.Path(self.config['state_directory'])/'OVERLAP_ACCEPTANCE.json',evidence)
    def test_exact_fraction_and_binary64_are_different(self):
        exact=number(dict(kind='exact',value='1/5'),'width');binary=number(dict(kind='binary64',value='0.2'),'width')
        self.assertNotEqual(exact[1]['exact'],binary[1]['exact']);self.assertEqual(exact[1]['exact'],'1/5')
    def test_normal_width_preview_enables_refill_only_for_selected_release(self):
        self.config['entry'] = 'schedule'
        scope = request(4, 6, 2)
        scope['action'] = 'preview'
        scope['scope']['per_method']['PR_PAULI']['delta_pf'] = [dict(kind='exact', value='1/5')]
        without = Gateway(self.config).handle(scope)
        self.config['live_ready_refill'] = True
        with_refill = Gateway(self.config).handle(scope)
        _, spec, _ = resolve(self.config, scope)
        self.assertNotIn('live_ready_refill', without['applied'])
        self.assertTrue(with_refill['applied']['live_ready_refill'])
        self.assertTrue(spec['overlap_ready_batch'])
        self.assertNotEqual(without['run_id'], with_refill['run_id'])
        self.config['work_conserving_search'] = True
        work = Gateway(self.config).handle(scope)
        self.assertTrue(work['applied']['work_conserving_search'])
        self.assertNotEqual(work['run_id'], with_refill['run_id'])

    def test_unsupported_kappa_modes_report_specific_field_and_create_no_run(self):
        for mode in ('absolute','multiplier'):
            q=request(4,7);q['scope']['per_method']['PR_PAULI']['kappa']=dict(mode=mode,min='2',max='64')
            with self.assertRaises(RequestError) as caught:build(self.config).handle(q)
            self.assertIn('kappa.'+mode,caught.exception.field)
        self.assertFalse((pathlib.Path(self.config['state_directory'])/'runs').exists())
    def test_omitted_settings_and_explicit_scope_start_grant(self):
        q=request(4,7);problem,spec,scope=resolve(self.config,q)
        self.assertTrue(spec['science_approved']);self.assertEqual(spec['max_steps_per_method'],10)
        self.assertEqual(problem['methods']['PR_PAULI']['caps'],[4]*13)
    def test_missing_confirmed_target_is_not_guessed(self):
        with self.assertRaisesRegex(RequestError,'no confirmed target'):build(self.config).handle(dict(action='start',instruction='探索を開始して'))
    def test_unknown_fields_do_not_silently_shrink_scope(self):
        q=request(4,7);q['scope']['per_method']['PR_PAULI']['unimplemented_bound']=7
        with self.assertRaises(RequestError):build(self.config).handle(q)
    def test_preview_and_start_have_same_effective_run_id(self):
        g=build(self.config);q=request(4,6);q['action']='preview';preview=g.handle(q)
        self.assertFalse((pathlib.Path(self.config['state_directory'])/'runs').exists())
        q['action']='start';started=g.handle(q);self.assertEqual(preview['run_id'],started['run_id'])
    def test_normal_scope_uses_central_preparation_defaults_and_keeps_explicit_zero(self):
        template = load(self.config['round'])
        template['executor']['distributed'] = dict(preparation=dict(budgets=dict(signal=0, compile=0, audit=0)))
        write(self.config['round'], template)
        self.config['preparation_budget_defaults'] = dict(signal='UNLIMITED', compile='UNLIMITED', audit=3)
        q = request(4, 6)
        q['action'] = 'preview'
        gateway = Gateway(self.config)
        preview = gateway.handle(q)
        self.assertEqual(preview['applied']['preparation_budgets'], self.config['preparation_budget_defaults'])
        self.assertFalse(preview['started'])
        q['scope']['preparation_budgets'] = dict(compile=0)
        limited = gateway.handle(q)
        self.assertEqual(limited['applied']['preparation_budgets'], dict(signal='UNLIMITED', compile=0, audit=3))
        self.assertNotEqual(preview['run_id'], limited['run_id'])
        q['action'] = 'start'
        _, spec, _ = resolve(self.config, q)
        self.assertEqual(spec['executor']['distributed']['preparation']['budgets'], limited['applied']['preparation_budgets'])
        self.assertEqual(spec['max_new_q3_per_method'], {'PR_PAULI': 0})
        self.assertEqual(load(self.config['round']), template)
    def test_run_index_links_to_git_relative_evidence_without_moving_existing_runs(self):
        checkout = self.home/'controller'
        self.config.update(code_root=str(checkout), run_evidence_root=str(checkout/'search-evidence'))
        q = request(4, 6)
        q['action'] = 'prepare'
        gateway = Gateway(self.config)
        result = gateway.handle(q)
        indexed = pathlib.Path(result['run_directory'])
        self.assertEqual(indexed.parent, self.home/'state/runs')
        self.assertTrue(indexed.is_symlink())
        self.assertEqual(indexed.resolve().relative_to(checkout), pathlib.Path('search-evidence', indexed.name))
        self.assertTrue((indexed/'ROUND.requested.json').is_file())
        self.assertEqual(gateway.handle(q)['run_id'], result['run_id'])
    def test_server2_can_confirm_target_without_grant_then_bare_start(self):
        g=build(self.config);q=request(4,6);q['action']='prepare';q.pop('instruction');prepared=g.handle(q)
        run=pathlib.Path(prepared['run_directory']);self.assertFalse((run/'AUTHORIZATION.json').exists());self.assertFalse((run/'LAUNCH.json').exists())
        started=g.handle(dict(action='start',instruction='探索を開始して'))
        self.assertEqual(started['run_id'],prepared['run_id']);self.assertTrue((run/'AUTHORIZATION.json').exists())
    def test_contract_change_does_not_import_prior_answers(self):
        g=build(self.config);old=g.handle(request(4,6));seed(old['run_directory'],self.config)
        p=load(self.config['problem']);p['accuracy']['epsilon_E_decimal']='2';write(self.config['problem'],p)
        changed=g.handle(request(5,7));self.assertEqual(changed['preflight']['reused']['PR_PAULI'],0)
    def test_missing_prior_directory_is_not_treated_as_unseen(self):
        r=load(self.config['round']);r['executor']['prior_run_dirs']=[str(self.home/'missing-source')];write(self.config['round'],r)
        with self.assertRaisesRegex(RequestError,'10.5'):build(self.config).handle(request(5,7))
    def test_unidentified_rows_do_not_create_a_false_ownership_match(self):
        from unittest.mock import patch
        from rpe_local0_acceptance_fixture import context, O
        gateway = Gateway(self.config)
        prior = self.home/'prior'
        attempt = prior/'jobs'/'PREPARATION:signal'/'unrelated'
        write(attempt/'REQUEST.json', dict(configuration_id=None))
        write(attempt/'LAUNCH.json', dict(pid=1))
        problem = load(self.config['problem'])
        spec = load(self.config['round'])
        spec['executor']['prior_run_dirs'] = [str(prior)]
        candidate_context = context(problem)
        candidate_context['candidates'].append(dict(candidate_context['candidates'][0],
                                                   configuration_id=None, ready=False))
        with patch.object(O, 'prepare', return_value=candidate_context):
            report = gateway.normal_preflight(self.home/'next', problem, spec)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['owned_unresolved_prior_jobs'], [])

    def test_existing_launched_unknown_job_is_not_restarted(self):
        g=build(self.config);old=g.handle(request(4,6));prior=seed(old['run_directory'],self.config)
        job=pathlib.Path(prior['job']);(job/'OUTCOME.json').unlink();write(job/'LAUNCH.json',dict(pid=1))
        with self.assertRaisesRegex(RequestError,'10.5'):g.handle(request(5,7))
    def test_corrupt_same_q3_receipt_holds_before_start(self):
        gateway=build(self.config);old=gateway.handle(request(4,6));prior=seed(old['run_directory'],self.config)
        path=pathlib.Path(prior['job'])/'OUTCOME.json';out=load(path);out['identity']['binding_id']='CONFLICT';write(path,out)
        with self.assertRaisesRegex(RequestError,'10.5'):gateway.handle(request(5,7,1))
        calls=load(pathlib.Path(self.config['state_directory'])/'ARTIFICIAL_CALLS.json');self.assertEqual(calls,[])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=pathlib.Path);a=ap.parse_args();OUTPUT=a.output;OUTPUT.mkdir(parents=True,exist_ok=True)
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(OUTPUT/'TEST_RESULT.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,log=log.getvalue()))
    print(log.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
