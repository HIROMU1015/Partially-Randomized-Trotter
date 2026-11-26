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

    def test_delta_snapshot_tracks_changed_jobs_not_whole_queue(self):
        self.reserve('a');self.reserve('b')
        rev,changed,rows=self.store.queue_rows_delta(['a','b'],None,{})
        self.assertEqual(set(rows),{'a','b'})
        next_rev,changed,updates=self.store.queue_rows_delta(['a','b'],rev,rows)
        self.assertEqual((next_rev,changed,updates),(rev,set(),{}))
        self.store.claim('worker',1,allowed_jobs=['a'])
        next_rev,changed,updates=self.store.queue_rows_delta(['a','b'],rev,rows)
        self.assertEqual(changed,{'a'});self.assertEqual(set(updates),{'a'})
        self.assertEqual(updates['a']['state'],'ASSIGNED');self.assertGreater(next_rev,rev)

    def test_concurrent_budget_reservations_are_atomic(self):
        def claim(i):
            return JobStore(self.path).reserve('PR_PAULI', 'j' + str(i), {'request_id': str(i), 'artificial': True}, 3, True)
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

    def test_workload_envelope_survives_same_attempt_replay(self):
        self.reserve()
        envelope = dict(preparation_workload={'peak_units': 42}, pbs_launch_resources={'memory_bytes': 5})
        first = self.store.claim('server1', 1, assignment_detail=envelope)[0]
        self.store.update('server1', 'j1', first['attempt_id'], 'ASSIGNED', {'worker': 'durable'})
        current = JobStore(self.path).row('j1')
        self.assertEqual(current['attempt_id'], first['attempt_id'])
        self.assertEqual(current['detail']['preparation_workload'], envelope['preparation_workload'])
        self.assertEqual(current['detail']['pbs_launch_resources'], envelope['pbs_launch_resources'])
        self.store.update('server1','j1',first['attempt_id'],'RUNNING',{'pid':123})
        self.assertEqual(self.store.row('j1')['detail']['pbs_launch_resources'],envelope['pbs_launch_resources'])
        self.assertEqual(self.store.claim('server1', 1), [])

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

    def test_real_job_without_any_eligible_host_never_reserves(self):
        request = dict(request_id='real', route=dict(kind='synthetic_real_route'))
        self.assertFalse(self.store.reserve('METHOD', 'real', request, 1, True))
        empty = dict(request_sha256=digest(request), eligible_host_ids=[], host_fingerprints={})
        self.assertFalse(self.store.reserve('METHOD', 'real', request, 1, True, eligibility=empty))
        self.assertEqual(self.store.counts('METHOD'), 0)
        self.assertIsNone(self.store.row('real'))

    def test_claim_enforces_membership_and_current_facts_and_keeps_priority(self):
        facts = {'a': dict(runtime='synthetic-a'), 'b': dict(runtime='synthetic-b')}
        for job, hosts in [('first', ['a']), ('second', ['b']), ('third', ['b'])]:
            request = dict(request_id=job, route=dict(kind='synthetic_real_route'))
            policy = dict(request_sha256=digest(request), eligible_host_ids=hosts,
                          host_fingerprints={h:digest(facts[h]) for h in hosts})
            self.assertTrue(self.store.reserve('METHOD', job, request, 3, True, eligibility=policy))
        self.assertEqual(self.store.claim('b', 2), [])
        self.assertEqual(self.store.claim('b', 1, host_fingerprints={'second':dict(runtime='changed')}), [])
        rows = self.store.claim('b', 1, host_fingerprints={j:facts['b'] for j in ['first','second','third']})
        self.assertEqual([r['job_id'] for r in rows], ['second'])
        self.assertEqual(self.store.row('first')['state'], 'QUEUED')
        self.assertEqual(self.store.row('third')['state'], 'QUEUED')
        reopened = JobStore(self.path)
        self.assertEqual(reopened.row('second')['eligibility'], self.store.row('second')['eligibility'])

    def test_queued_hold_releases_budget_but_assigned_owner_cannot_change(self):
        request = dict(request_id='real', route=dict(kind='synthetic_real_route'))
        facts = dict(runtime='synthetic')
        policy = dict(request_sha256=digest(request), eligible_host_ids=['a'], host_fingerprints={'a':digest(facts)})
        empty = dict(request_sha256=digest(request), eligible_host_ids=[], host_fingerprints={})
        self.assertTrue(self.store.reserve('METHOD', 'real', request, 1, True, eligibility=policy))
        self.assertFalse(self.store.set_eligibility('real', empty))
        self.assertEqual(self.store.counts('METHOD'), 0)
        self.assertTrue(self.store.reserve('METHOD', 'real', request, 1, True, eligibility=policy))
        assigned = self.store.claim('a', 1, host_fingerprints={'real':facts})[0]
        self.assertFalse(self.store.set_eligibility('real', empty))
        self.assertEqual(self.store.row('real')['attempt_id'], assigned['attempt_id'])
        self.assertEqual(self.store.row('real')['state'], 'ASSIGNED')
        self.assertEqual(self.store.claim('b', 1, host_fingerprints={'real':facts}), [])


if __name__ == '__main__':
    unittest.main()


def test_resume_rows_omit_accepted_request_and_eligibility(tmp_path):
    from trottertracks.rpe_q3_store import JobStore
    store=JobStore(tmp_path/'queue.sqlite3')
    req={'route':{'kind':'artificial'},'large':'x'*500000}
    assert store.reserve('METHOD','done',req,2,True)
    row,=store.claim('host',1)
    store.complete_hashed('host','done',row['attempt_id'],row['request_sha256'],'complete')
    assert store.reserve('METHOD','pending',req,2,True)
    active,accepted=store.resume_rows()
    assert len(active)==1 and active[0]['request']==req
    assert accepted==[{'job_id':'done','method':'METHOD','attempt_id':row['attempt_id'],'state':'ACCEPTED'}]
    assert store.row('done')['request']==req
