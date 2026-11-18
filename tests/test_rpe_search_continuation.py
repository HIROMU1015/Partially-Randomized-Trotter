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
from trottertracks.rpe_method_optimizer import FAIL
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
from trottertracks.rpe_priority_search import run_priority
from trottertracks.rpe_q3_coordinator import digest
from trottertracks.rpe_release_registry import fingerprint
from trottertracks.rpe_search_continuation import verify, verify_prior_q3_holds

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
