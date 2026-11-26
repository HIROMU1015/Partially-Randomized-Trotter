"""Continuation from a shutdown fault preserves scientific ownership and budgets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from trottertracks.rpe_local0_search import Gateway, RequestError
from trottertracks.rpe_method_optimizer import FAIL, Executor
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
from trottertracks.rpe_priority_search import run_priority
from trottertracks.rpe_q3_coordinator import digest
from trottertracks.rpe_release_registry import fingerprint
from trottertracks.rpe_search_continuation import (
    reconciled_prior_runs, verify, verify_prior_q3_holds, verify_retired_preparation_attempts,
)

from test_rpe_optimizer_concurrent import fixture


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root/'search-original'
        (self.original/'distributed').mkdir(parents=True)
        self.successor = self.root/'search-successor'
        self.problem = {'problem': {'system': 'ARTIFICIAL'}}
        self.old = {
            'methods': ['PR_PAULI'], 'base_commit': 'science',
            'max_new_q3_per_method': {'PR_PAULI': 3},
            'executor': {'distributed': {'preparation': {'budgets': {'signal': 1, 'compile': 1, 'audit': 1}}}},
        }
        self.save('APPLIED_SCOPE.json', dict(code_source='old', signature='sig',
                round_requested_sha256=fingerprint(self.old), problem_sha256=fingerprint(self.problem)))
        self.save('ROUND.requested.json', self.old)
        self.save('PROBLEM.json', self.problem)
        self.save('LAUNCH.json', dict(invocation_id='owned'))
        self.save('EXIT.json', dict(invocation_id='owned', returncode=1))
        self.save('KAPPA_STATE.json', dict(stop=dict(reason='RUNTIME_EXCEPTION', error_type='TimeoutExpired',
                  message=str(self.original/'distributed/LOCAL_WORKER.json'))))
        self.save('ROUND_RESULT.json', dict(methods=dict(PR_PAULI=dict(stop_reason='NO_CANDIDATE_LEFT_IN_RANGE'))))
        self.rows = [('a', 'PREPARATION:signal', 'ACCEPTED', 'local', 'first', 'digest-a'),
                     ('b', 'PR_PAULI', 'ACCEPTED', 'server2', 'second', 'digest-b')]
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3') as db:
            db.execute('CREATE TABLE jobs (job_id, method, state, host_id, attempt_id, request_sha256)')
            db.executemany('INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)', self.rows)
        self.round = deepcopy(self.old)
        self.round['executor']['prior_run_dirs'] = [str(self.original)]
        self.round['executor']['distributed']['preparation']['budgets']['signal'] = 0
        self.round['max_new_q3_per_method']['PR_PAULI'] = 2
        self.round['continuation'] = dict(
            parent_directory=str(self.original), parent_run_id=self.original.name, parent_code_source='old',
            parent_signature='sig', parent_queue_sha256=fingerprint(self.rows),
            consumed={'PREPARATION:signal': 1, 'PR_PAULI': 1})

    def save(self, name, body):
        (self.original/name).write_text(json.dumps(body))

    def test_drain_handoff_only_accepts_existing_preparation_owners(self):
        (self.original/'distributed/relay').mkdir()
        self.save('NORMAL_DRIVER.json', dict(invocation_id='owned', pid=123))
        self.save('distributed/relay/STATUS.json', {'stopped': False})
        stop = {'reason':'USER_REQUESTED_STOP', 'requested_epoch':1}
        self.save('STOP_REQUESTED.json', stop)
        fixed = dict(job_id='a',method='PREPARATION:signal',host_id='local',attempt_id='first',request_sha256='digest-a')
        self.round['reconciled_prior_runs']=[dict(parent_directory=str(self.original),invocation_id='owned',
            mode='STOP_DRAIN_PREPARATION',stop_sha256=fingerprint(stop),owned_preparation=[fixed])]
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3') as db:
            db.execute("UPDATE jobs SET state='RUNNING' WHERE job_id='a'")
        self.assertEqual(reconciled_prior_runs(self.round)[self.original],[fixed])
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3') as db:
            db.execute("UPDATE jobs SET state='RUNNING' WHERE job_id='b'")
        with self.assertRaisesRegex(ValueError,'unfenced work'):
            reconciled_prior_runs(self.round)
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3') as db:
            db.execute("UPDATE jobs SET state='ACCEPTED' WHERE job_id='b'")
            db.execute("UPDATE jobs SET state='ACCEPTED' WHERE job_id='a'")
        self.assertEqual(reconciled_prior_runs(self.round)[self.original],[fixed])

    def test_reconcile_shutdown_without_rewriting_stop_or_unknown_attempt(self):
        (self.original/'EXIT.json').unlink()
        (self.original/'FAST_DRIVER.lock').touch()
        (self.original/'distributed/relay').mkdir()
        (self.original/'distributed/pbs').mkdir()
        self.save('LAUNCH.json', dict(invocation_id='owned', pid=2**30))
        self.save('NORMAL_DRIVER.json', dict(invocation_id='owned', pid=2**30))
        self.save('STOP_REQUESTED.json', {'reason': 'USER_REQUESTED_STOP'})
        self.save('distributed/relay/STATUS.json', {'pid': 2**30, 'stopped': True})
        self.save('distributed/pbs/STATE.json', {'allocations': {
            'token': {'status': 'R', 'observed': {'job_id': '17.pbs'}}}})
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3') as db:
            db.execute('ALTER TABLE jobs ADD COLUMN detail')
            db.execute('ALTER TABLE jobs ADD COLUMN request')
            db.execute("UPDATE jobs SET method='PREPARATION:signal', state='UNKNOWN', detail=?, request=? WHERE job_id='b'",
                       (json.dumps({'automatic_retry': False}),
                        json.dumps({'preparation': {'provider': 'PAULI_SIGNALS'}})))
            rows = db.execute('SELECT * FROM jobs ORDER BY job_id').fetchall()
        evidence = dict(parent_directory=str(self.original), invocation_id='owned',
                        queue_sha256=fingerprint(rows), terminal_pbs_jobs={})
        self.round['reconciled_prior_runs'] = [evidence]
        self.round['methods'] = ['PR_DF']
        with self.assertRaisesRegex(ValueError, 'unresolved PBS'):
            reconciled_prior_runs(self.round)
        evidence['terminal_pbs_jobs']['17.pbs'] = 'F'
        self.assertEqual(reconciled_prior_runs(self.round)[self.original][0]['attempt_id'], 'second')
        self.assertFalse((self.original/'EXIT.json').exists())
        self.assertTrue((self.original/'STOP_REQUESTED.json').exists())
        self.round['methods'] = ['PR_PAULI']
        with self.assertRaisesRegex(ValueError, 'unresolved attempt in scope'):
            reconciled_prior_runs(self.round)

    def test_retired_preparation_validates_same_request_and_stopped_allocation(self):
        (self.successor/'distributed').mkdir(parents=True)
        (self.original/'distributed/relay').mkdir()
        (self.original/'distributed/pbs').mkdir()
        self.save('distributed/relay/STATUS.json', {'stopped': True})
        self.save('distributed/pbs/STATE.json', {'allocations': {'token': {
            'spec': {'profile': {'host_id': 'pbs-0'}}, 'observed': {'job_id': 'test.pbs'}}}})
        request = {'request_id': 'prep-123', 'preparation': {'stage': 'signal'}}
        row = ('prep-123', 'PREPARATION:signal', 'pbs-0', 'attempt', digest(request),
               json.dumps(request), 'UNKNOWN', json.dumps({'automatic_retry': False}))
        for run in (self.original, self.successor):
            with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
                db.execute('DROP TABLE IF EXISTS jobs')
                db.execute('CREATE TABLE jobs (job_id, method, host_id, attempt_id, request_sha256, request, state, detail)')
                db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)', row)
        hold = dict(attempt_id='attempt', request_sha256=digest(request),
                    allocation_receipt={'pbs_job_id': 'test.pbs', 'job_state': 'F'})
        lineage = self.successor/'RUN_LINEAGE.json'
        record = dict(parent_directory=str(self.original), retired_preparation_attempts={'prep-123': hold})
        lineage.write_text(json.dumps(record))
        self.assertEqual(list(verify_retired_preparation_attempts(self.successor).values()), [('prep-123', 'attempt')])
        next_run = self.root/'search-next'
        (next_run/'distributed').mkdir(parents=True)
        (self.successor/'distributed/relay').mkdir()
        (self.successor/'distributed/pbs').mkdir()
        for name in ('LAUNCH.json', 'EXIT.json', 'distributed/relay/STATUS.json'):
            (self.successor/name).write_bytes((self.original/name).read_bytes())
        (self.successor/'distributed/pbs/STATE.json').write_text(json.dumps({'allocations': {}}))
        db = 'distributed/QUEUE.sqlite3'
        (next_run/db).write_bytes((self.successor/db).read_bytes())
        (next_run/'RUN_LINEAGE.json').write_text(json.dumps(dict(record, parent_directory=str(self.successor))))
        self.assertEqual(set(verify_retired_preparation_attempts(next_run)),
                         {p/'jobs/PREPARATION:signal/prep-123' for p in (self.successor,self.original)})
        with sqlite3.connect(self.successor/db) as connection:
            connection.execute("UPDATE jobs SET attempt_id='different'")
        with self.assertRaisesRegex(ValueError, 'Retired preparation ownership differs'):
            verify_retired_preparation_attempts(next_run)
        hold['allocation_receipt']['job_state'] = 'R'
        lineage.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Retired preparation ownership differs'):
            verify_retired_preparation_attempts(self.successor)

    def test_exact_replay_preserves_request_and_resumes_without_a_second_attempt(self):
        import sys

        method = 'PR_PAULI'
        problem, context, routes = fixture([method], 1)
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        proposal = dict(configuration_id=context['candidates'][0]['configuration_id'],
                        vector=(2,)*26, cost=10, origin=dict(kind='ARTIFICIAL_REPLAY'))
        old = Executor(root, self.original, routes, {}, problem, optimizer_source=source).for_method(method)
        job = old.job_dir(proposal)
        request = old.request(proposal, job, old.route(proposal['configuration_id']))
        request['execution_profile'] = dict(dp_workers=3)
        job.mkdir(parents=True)
        (job/'REQUEST.json').write_text(json.dumps(request))
        original_launch = dict(host_id='old-host', attempt_id='previous-attempt')
        (job/'LAUNCH.json').write_text(json.dumps(original_launch))
        fixed = dict(parent_directory=str(self.original), method=method, job_id=job.name,
                     request_sha256=digest(request), replay_exact=True)
        profile = dict(host_id='test-local', root=str(root), interpreter=sys.executable, slots=1,
                       route_views={}, work_dir=str(self.root/'worker'))
        attempts = []
        for _ in range(2):
            successor = DistributedExecutor(root, self.successor, routes, {}, problem, optimizer_source=source,
                prior_run_dirs=[self.original, self.successor], budgets={method: 1},
                distributed=dict(port=0, local_worker=profile))
            successor.prior_q3_holds = [fixed]
            try:
                self.assertTrue(successor.replay_prior_q3())
                row = successor.store.row(job.name)
                self.assertEqual(row['state'], 'ACCEPTED')
                self.assertEqual(row['request'], request)
                attempts.append((row['attempt_id'], row['adopted_epoch']))
            finally:
                successor.finish()
        self.assertEqual(attempts[0], attempts[1])
        self.assertNotEqual(attempts[0][0], original_launch['attempt_id'])
        self.assertEqual(json.loads((job/'LAUNCH.json').read_text()), original_launch)
        self.assertFalse((job/'OUTCOME.json').exists())

    def test_shutdown_stop_is_not_reported_as_completed(self):
        gateway = Gateway({'state_directory': str(self.root)}, launch=Mock())
        with self.assertRaisesRegex(RequestError, 'requires reconciliation'):
            gateway.start(self.original, {'action': 'resume'})
        gateway.launch.assert_not_called()

    def test_handoff_spends_only_remaining_budget_and_keeps_parent_receipts(self):
        before = {p.name: p.read_bytes() for p in self.original.glob('*.json')}
        status = verify(self.successor, self.problem, self.round)
        self.assertEqual(status['remaining_q3'], {'PR_PAULI': 2})
        self.assertEqual(status['remaining']['signal'], 0)
        self.assertEqual(status['consumed'], {'PREPARATION:signal': 1, 'PR_PAULI': 1})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.original.glob('*.json')})
        self.round['max_new_q3_per_method']['PR_PAULI'] = 3
        with self.assertRaisesRegex(ValueError, 'increases or resets'):
            verify(self.successor, self.problem, self.round)

    def test_source_and_scientific_stop_and_unknown_attempt_are_rejected(self):
        self.round['continuation']['parent_code_source'] = 'different'
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            verify(self.successor, self.problem, self.round)
        self.round['continuation']['parent_code_source'] = 'old'
        self.save('ROUND_RESULT.json', dict(methods=dict(PR_PAULI=dict(stop_reason='UNKNOWN_RESULT'))))
        with self.assertRaisesRegex(ValueError, 'scientific STOP'):
            verify(self.successor, self.problem, self.round)
        self.save('ROUND_RESULT.json', dict(methods=dict(PR_PAULI=dict(stop_reason='NO_CANDIDATE_LEFT_IN_RANGE'))))
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3') as db:
            db.execute("UPDATE jobs SET state='UNKNOWN' WHERE job_id='b'")
        with self.assertRaisesRegex(ValueError, 'unresolved or unknown'):
            verify(self.successor, self.problem, self.round)

    def test_idle_poller_is_reaped_but_owned_attempt_keeps_its_daemon(self):
        worker = Mock(pid=23, returncode=-9)
        worker.wait.side_effect = [subprocess.TimeoutExpired('poll', 10), -9]
        executor = object.__new__(DistributedExecutor)
        executor.run_dir = self.original
        executor.local_worker = worker
        executor._close_worker_rpc = Mock()
        executor._local_attempts_settled = Mock(return_value=True)
        executor._finish_local_worker()
        worker.kill.assert_called_once_with()
        executor._close_worker_rpc.assert_called_once_with()
        receipt = json.loads((self.original/'distributed/LOCAL_WORKER_EXIT.json').read_text())
        self.assertEqual(receipt['reason'], 'IDLE_POLLER_STOPPED_AFTER_RPC_CLOSE')
        worker.reset_mock()
        worker.wait.side_effect = subprocess.TimeoutExpired('poll', 10)
        executor._local_attempts_settled.return_value = False
        with self.assertRaises(subprocess.TimeoutExpired):
            executor._finish_local_worker()
        worker.kill.assert_not_called()

    def test_local_interrupted_write_is_scoped_before_reading_corrupt_outcome(self):
        from trottertracks.rpe_method_optimizer import Executor
        original = self.root/'old_study'/'run02'
        job_id,method='a'*20,'PR_DF'
        job=original/'jobs'/method/job_id
        job.mkdir(parents=True)
        problem=dict(problem={},rpe={},target={'energy_ha':0},accuracy={})
        (original.parent/'PROBLEM.json').write_text(json.dumps(problem))
        (original/'CONTROLLER_PID.json').write_text(json.dumps({'pid': 999999999}))
        request=dict(request_id=job_id,method=method,configuration_id='fixed',N_X=[2]*13,N_Y=[3]*13)
        (job/'REQUEST.json').write_text(json.dumps(request))
        for name in ('STARTING.json','LAUNCH.json'):
            (job/name).write_text(json.dumps({'pid': 999999999}))
        for name in ('PROCESS_RETURN.json','OUTCOME.json'):
            (job/name).touch()
        spec=dict(methods=[method],executor=dict(prior_run_dirs=[str(original)],prior_q3_holds=[
            dict(parent_directory=str(original),method=method,job_id=job_id,
                 kind='LOCAL_INTERRUPTED_WRITE',request_sha256=digest(request))]))
        holds=verify_prior_q3_holds(self.root,self.successor,problem,spec)
        self.assertEqual(holds[str(job)]['N_Y'],[3]*13)
        executor=Executor(self.root,self.successor,{}, {},problem,prior_run_dirs=[original]).for_method(method)
        executor.prior_q3_holds=list(holds.values())
        known,refused=executor.prior_records([dict(configuration_id='fixed',ready=True)])
        self.assertFalse(known)
        self.assertEqual(refused,[dict(job=str(job),reason='PRIOR_Q3_PERSISTENCE_HOLD')])
        self.assertEqual((job/'OUTCOME.json').read_bytes(),b'')
        (job/'OUTCOME.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hold changed'):
            verify_prior_q3_holds(self.root,self.successor,problem,spec)

    def test_prior_hold_keeps_the_exact_shots_out_of_dispatch_and_adoption(self):
        problem, context, _ = fixture(['TRUNCATED_PF'], 1)
        problem.update(problem={'system': 'ARTIFICIAL'}, rpe={}, target={})
        candidate, = context['candidates']
        shots = candidate['caps']
        method, cid, job_id = 'TRUNCATED_PF', candidate['configuration_id'], 'd'*20
        job = self.original/'jobs'/method/job_id
        job.mkdir(parents=True)
        (self.original/'distributed/relay').mkdir()
        self.save('PROBLEM.json', problem)
        self.save('distributed/relay/STATUS.json', dict(stopped=True))
        request = dict(request_id=job_id, method=method, configuration_id=cid, N_X=shots, N_Y=shots)
        (job/'REQUEST.json').write_text(json.dumps(request))
        (job/'LAUNCH.json').write_text(json.dumps(dict(host_id='wsl', attempt_id='attempt')))
        ledger = self.original/'distributed/QUEUE.sqlite3'
        with sqlite3.connect(ledger) as db:
            db.execute('DROP TABLE jobs')
            db.execute('''CREATE TABLE jobs (job_id, state, method, host_id, attempt_id, request_sha256, request,
                detail, completed_epoch, adopted_epoch, completion_sha256)''')
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,NULL,NULL,NULL)', (
                job_id, 'UNKNOWN', method, 'wsl', 'attempt', digest(request), json.dumps(request),
                json.dumps(dict(automatic_retry=False, reason='MISSING_EXIT_RECEIPT'))))
        fixed = dict(parent_directory=str(self.original), method=method, job_id=job_id, host_id='wsl',
                     attempt_id='attempt', request_sha256=digest(request))
        spec = dict(methods=[method], executor=dict(prior_run_dirs=[str(self.original)], prior_q3_holds=[fixed]))
        holds = verify_prior_q3_holds(self.root, self.successor, problem, spec)
        self.assertEqual(set(holds), {str(job)})
        self.assertEqual(holds[str(job)]['configuration_id'], cid)
        backend = Mock()
        backend.answer.return_value = dict(status=FAIL, source='ARTIFICIAL', MSE_upper=2, calls=1)
        standby = Mock()
        search = dict(problem['search'], seed_rules=['alternate'], max_steps_per_method=1, work_conserving_search=True)
        report = run_priority(method, [candidate], {}, backend, search, 1, lambda *_: {'shots': [1]*26}, 1,
                              prior_q3_holds=holds.values(), standby=standby, prefetch_state=lambda: {'capacity': 2})
        self.assertEqual(report['new_calls'], 1)
        self.assertEqual(backend.answer.call_args.args[0]['vector'], (1,)*26)
        self.assertTrue(all(p['vector'] == (1,)*26 for args, _ in standby.call_args_list
                            for p in [args[0], *args[1]]))
        self.assertEqual(report['prior_q3_holds'][0]['job_id'], job_id)
        fixed['attempt_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            verify_prior_q3_holds(self.root, self.successor, problem, spec)
        fixed['attempt_id'] = 'attempt'
        (job/'OUTCOME.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            verify_prior_q3_holds(self.root, self.successor, problem, spec)
        outcome = dict(status='UNKNOWN', request_id=job_id, scientific_evaluation_calls=0,
                       reason='Sibling failure before science')
        (job/'OUTCOME.json').write_text(json.dumps(outcome))
        (job/'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=2, automatic_retry=False)))
        with sqlite3.connect(ledger) as db:
            db.execute("UPDATE jobs SET state='ACCEPTED', detail=?, completed_epoch=1, completion_sha256='saved'",
                       (json.dumps(dict(outcome_status='UNKNOWN')),))
        fixed.update(kind='TERMINAL_WORKER_UNKNOWN', completion_sha256='saved',
                     outcome_sha256=hashlib.sha256((job/'OUTCOME.json').read_bytes()).hexdigest(),
                     process_return_sha256=hashlib.sha256((job/'PROCESS_RETURN.json').read_bytes()).hexdigest())
        terminal = verify_prior_q3_holds(self.root, self.successor, problem, spec)
        self.assertEqual(terminal[str(job)]['reason'], outcome['reason'])
        outcome['scientific_evaluation_calls'] = 1
        (job/'OUTCOME.json').write_text(json.dumps(outcome))
        with self.assertRaisesRegex(ValueError, 'zero-call receipt'):
            verify_prior_q3_holds(self.root, self.successor, problem, spec)
