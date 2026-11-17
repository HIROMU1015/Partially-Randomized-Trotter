"""Continuation from a shutdown fault preserves scientific ownership and budgets."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from trottertracks.rpe_local0_search import Gateway, RequestError
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
from trottertracks.rpe_release_registry import fingerprint
from trottertracks.rpe_search_continuation import verify


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
