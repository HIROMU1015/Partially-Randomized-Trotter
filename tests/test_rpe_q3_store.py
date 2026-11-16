"""Artificial execution-state tests, without scientific inputs or workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.rpe_q3_store import JobStore, digest


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'queue.sqlite3'
        self.store = JobStore(self.path)

    def reserve(self, job='j1', method='PR_PAULI', budget=4, approved=True):
        return self.store.reserve(method, job, {'request_id': job, 'artificial': True}, budget, approved, contract='FIXED')

    def complete(self, row, files=None):
        return self.store.complete(row['host_id'], row['job_id'], row['attempt_id'], row['request_sha256'],
                                   {'OUTCOME.json': 'ARTIFICIAL'} if files is None else files)

    def test_reservation_survives_reopen_and_budget_is_per_method(self):
        self.assertTrue(self.reserve(budget=1))
        self.store = JobStore(self.path)
        self.assertEqual(self.store.counts('PR_PAULI'), 1)
        self.assertFalse(self.reserve('j2', budget=1))
        self.assertTrue(self.reserve('j2', method='PR_DF', budget=1))
        self.assertTrue(self.reserve(budget=1, approved=False))
        self.assertFalse(self.reserve('j3', approved=False))
        self.assertEqual(len(self.store.rows()), 2)

    def test_concurrent_budget_reservations_are_atomic(self):
        def claim(i):
            return JobStore(self.path).reserve('PR_PAULI', 'j' + str(i), {'request_id': str(i)}, 3, True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(claim, range(12)))
        self.assertEqual(sum(results), 3)
        self.assertEqual(self.store.counts('PR_PAULI'), 3)

    def test_two_hosts_cannot_claim_same_scientific_job(self):
        for i in range(10):
            self.reserve('j' + str(i), budget=10)
        def take(host):
            return JobStore(self.path).claim(host, 10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            batches = list(pool.map(take, ['server1', 'server2']))
        jobs = [row for batch in batches for row in batch]
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({row['job_id'] for row in jobs}), 10)
        self.assertEqual(len({row['attempt_id'] for row in jobs}), 10)
        self.assertEqual(self.store.counts('PR_PAULI', started_only=True), 10)

    def test_assignment_never_expires_after_disconnect(self):
        self.reserve()
        first = self.store.claim('server1', 1)[0]
        self.store = JobStore(self.path)
        self.assertEqual(self.store.claim('server2', 100), [])
        self.assertEqual(self.store.claim('server1', 1), [])
        self.assertEqual(self.store.row('j1')['attempt_id'], first['attempt_id'])
        self.assertFalse(self.store.cancel('PR_PAULI', 'j1'))
        with self.assertRaises(ValueError):
            self.store.update('server2', 'j1', first['attempt_id'], 'RUNNING')

    def test_only_unassigned_queue_can_release_reservation(self):
        self.reserve(budget=1)
        self.assertTrue(self.store.cancel('PR_PAULI', 'j1'))
        self.assertEqual(self.store.counts('PR_PAULI'), 0)
        self.assertTrue(self.reserve('j2', budget=1))
        self.assertFalse(self.reserve(budget=1))
        self.store.cancel('PR_PAULI', 'j2')
        self.assertTrue(self.reserve(budget=1))
        self.assertEqual(self.store.counts('PR_PAULI'), 1)

    def test_controller_order_survives_prefix_hints(self):
        for name in ('early', 'middle', 'late'):
            self.reserve(name)
        selected = self.store.claim('server1', 2, cached_prefixes=['late'], allowed_jobs=['late', 'middle'])
        self.assertEqual([row['job_id'] for row in selected], ['late', 'middle'])
        self.assertEqual(self.store.row('early')['state'], 'QUEUED')
        self.assertEqual(self.store.claim('server2', 1, allowed_jobs=[]), [])

    def test_completion_retransmit_and_adoption_are_exactly_once(self):
        self.reserve()
        row = self.store.claim('server1', 1)[0]
        self.store.update('server1', 'j1', row['attempt_id'], 'RUNNING', {'pid': 123})
        self.store.update('server1', 'j1', row['attempt_id'], 'COMPLETED_PENDING_TRANSFER')
        accepted, duplicate = self.complete(row)
        self.assertFalse(duplicate)
        self.assertEqual(accepted['state'], 'ACCEPTED')
        self.assertTrue(self.store.adopt('j1'))
        self.store = JobStore(self.path)
        self.assertTrue(self.complete(row)[1])
        self.assertFalse(self.store.adopt('j1'))
        self.assertEqual(self.store.counts('PR_PAULI'), 1)
        self.assertEqual([e['event'] for e in self.store.events()].count('ACCEPT'), 1)
        self.assertEqual([e['event'] for e in self.store.events()].count('ADOPT'), 1)
        with self.assertRaises(ValueError):
            self.complete(row, {'OUTCOME.json': 'CHANGED'})

    def test_hash_owner_and_state_regression_rejected(self):
        self.reserve()
        row = self.store.claim('server1', 1)[0]
        self.store.update('server1', 'j1', row['attempt_id'], 'STARTING')
        self.store.update('server1', 'j1', row['attempt_id'], 'RUNNING')
        with self.assertRaises(ValueError):
            self.store.update('server1', 'j1', row['attempt_id'], 'ASSIGNED')
        with self.assertRaises(ValueError):
            self.store.complete('server1', 'j1', 'DIFFERENT', row['request_sha256'], {})
        with self.assertRaises(ValueError):
            self.store.complete('server1', 'j1', row['attempt_id'], digest('DIFFERENT'), {})
        self.store.update('server1', 'j1', row['attempt_id'], 'UNKNOWN')
        self.assertEqual(self.store.claim('server2', 2), [])
        self.store.update('server1', 'j1', row['attempt_id'], 'COMPLETED_PENDING_TRANSFER')
        self.complete(row)

    def test_scientific_identity_cannot_change_under_same_job(self):
        self.reserve()
        with self.assertRaises(ValueError):
            self.store.reserve('PR_PAULI', 'j1', {'different': True}, 4, True, contract='FIXED')
        with self.assertRaises(ValueError):
            self.store.reserve('PR_DF', 'j1', {'request_id': 'j1', 'artificial': True}, 4, True, contract='FIXED')
        with self.assertRaises(ValueError):
            self.store.reserve('PR_PAULI', 'j1', {'request_id': 'j1', 'artificial': True}, 4, True, contract='CHANGED')


if __name__ == '__main__':
    unittest.main()
