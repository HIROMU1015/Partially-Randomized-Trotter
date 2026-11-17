"""Synthetic controller failures at durable claim/completion boundaries.

No adapter is launched: outcomes stand in for already completed artificial jobs.
"""
import base64
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor


class ControllerFailures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'run'
        self.executors = []
        self.addCleanup(self.close_all)
        self.source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        self.routes = {'PR_PAULI': [dict(kind='artificial', interpreter=sys.executable, configuration_ids='*',
            artificial=dict(weights=[1, 1], scale='1/1000', tau='1'))]}
        self.problem = dict(accuracy=dict(e=1), target=dict(energy_ha=0), rpe=dict(M=1))

    def executor(self, capacity=2):
        result = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent=capacity, budgets={'PR_PAULI': 8},
            distributed=dict(port=0, affinity_wait_seconds=0))
        self.executors.append(result)
        return result

    def close_one(self, executor):
        # The fixture starts no processes, so release only its synthetic tasks
        # to avoid hanging test teardown when testing a missing _finished call.
        with executor.state.cv:
            for task in executor.state.tasks.values():
                if task.status == 'RUNNING':
                    task.status = 'DONE'; task.event.set()
            executor.state.active = 0
            executor.state.cv.notify_all()
        executor.finish()

    def close_all(self):
        for executor in self.executors:
            self.close_one(executor)

    def proposal(self, name):
        return dict(configuration_id=name, vector=(1, 1, 1, 1), cost=Fraction(4), origin=dict(kind='ARTIFICIAL'))

    def enqueue(self, executor, name):
        return executor.for_method('PR_PAULI')._enqueue(self.proposal(name), demand=True)

    def poll(self, executor, slots=1):
        return executor.rpc('/v1/poll', dict(host_id='server1', slots=slots, cached_prefixes=[], source_sha=self.source))

    def completion(self, assignment):
        request = assignment['request']
        allocation = [dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(request['N_X'], request['N_Y']))]
        outcome = dict(status=O.PASS, MSE_lower=dict(numerator=1, denominator=500),
            MSE_upper=dict(numerator=1, denominator=500), artificial=True,
            identity=dict(request['expected'], allocation=allocation, contract_matches=True),
            request_id=assignment['job_id'], route='artificial')
        files = {'OUTCOME.json': outcome, 'PROCESS_RETURN.json': dict(returncode=0, signal=None)}
        return dict(host_id='server1', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
            request_sha256=assignment['request_sha256'],
            files={name: base64.b64encode(json.dumps(value).encode()).decode() for name, value in files.items()})

    def test_lost_poll_response_replays_same_attempt_without_overfilling(self):
        executor = self.executor(capacity=2)
        first_task = self.enqueue(executor, 'first')
        second_task = self.enqueue(executor, 'second')
        lost = self.poll(executor)['assignments']
        replayed = self.poll(executor)['assignments']
        self.assertEqual(replayed, lost)
        self.assertEqual(len(replayed), 1)
        self.assertEqual(executor.store.row(second_task.job.name)['state'], 'QUEUED')
        self.assertEqual(executor.store.counts('PR_PAULI', started_only=True), 1)
        executor.rpc('/v1/complete', self.completion(lost[0]))
        self.assertTrue(first_task.event.is_set())
        next_assignment = self.poll(executor)['assignments']
        self.assertEqual(next_assignment[0]['job_id'], second_task.job.name)

    def test_retransmit_after_accept_before_publish_releases_waiter_once(self):
        executor = self.executor()
        task = self.enqueue(executor, 'first')
        assigned = self.poll(executor)['assignments'][0]
        payload = self.completion(assigned)
        with patch.object(executor, '_publish_transfer', side_effect=OSError('ARTIFICIAL publish interruption')):
            with self.assertRaises(OSError):
                executor.rpc('/v1/complete', payload)
        self.assertEqual(executor.store.row(task.job.name)['state'], 'ACCEPTED')
        self.assertFalse(task.event.is_set())
        response = executor.rpc('/v1/complete', payload)
        self.assertEqual(response, dict(ok=True, duplicate=True))
        self.assertTrue(task.event.is_set(), 'Accepted retransmission must release the original waiter')
        self.assertEqual(task.status, 'DONE')
        self.assertEqual(executor.state.active, 0)
        self.assertEqual(executor.state.metrics['finished'], 1)
        executor.rpc('/v1/complete', payload)
        self.assertEqual(executor.state.metrics['finished'], 1)

    def test_controller_restart_publishes_accepted_staging_without_new_attempt(self):
        executor = self.executor()
        proposal = self.proposal('first')
        task = self.enqueue(executor, 'first')
        assigned = self.poll(executor)['assignments'][0]
        payload = self.completion(assigned)
        with patch.object(executor, '_publish_transfer', side_effect=OSError('ARTIFICIAL publish interruption')):
            with self.assertRaises(OSError):
                executor.rpc('/v1/complete', payload)
        self.assertFalse((task.job/'OUTCOME.json').exists())
        self.close_one(executor)
        resumed = self.executor()
        self.assertTrue((task.job/'OUTCOME.json').exists())
        answer = resumed.for_method('PR_PAULI').answer(proposal)
        self.assertEqual(answer['status'], O.PASS)
        self.assertEqual(answer['calls'], 0)
        self.assertEqual(resumed.store.row(task.job.name)['attempt_id'], assigned['attempt_id'])
        self.assertEqual(resumed.store.counts('PR_PAULI', started_only=True), 1)
        self.assertEqual(self.poll(resumed)['assignments'], [])
        self.assertEqual([e['event'] for e in resumed.store.events()].count('ACCEPT'), 1)
        self.assertEqual([e['event'] for e in resumed.store.events()].count('ADOPT'), 1)

    def test_successful_artifact_requires_both_zero_return_and_no_signal(self):
        executor = self.executor()
        task = self.enqueue(executor, 'first')
        payload = self.completion(self.poll(executor)['assignments'][0])
        payload['files']['PROCESS_RETURN.json'] = base64.b64encode(json.dumps(dict(returncode=0, signal=9)).encode()).decode()
        with self.assertRaisesRegex(ValueError, 'clean adapter exit'):
            executor.rpc('/v1/complete', payload)
        self.assertNotEqual(executor.store.row(task.job.name)['state'], 'ACCEPTED')
        self.assertFalse((task.job/'OUTCOME.json').exists())

    def test_prior_records_counts_host_reuse_once_and_excludes_transfer_validation(self):
        executor = self.executor()
        proposal = self.proposal('first')
        task = self.enqueue(executor, 'first')
        assigned = self.poll(executor)['assignments'][0]
        payload = self.completion(assigned)
        executor.rpc('/v1/complete', payload)
        view = executor.for_method('PR_PAULI')
        self.assertEqual(executor.metrics()['by_host']['server1']['reused'], 0)

        # Validation of the transfer copy is not reuse by the optimizer, even
        # after the same attempt has been durably accepted and published.
        staging = self.run/'distributed'/'transfers'/assigned['job_id']/assigned['attempt_id']
        self.assertEqual(view.ingest(staging, proposal, calls=0)['status'], O.PASS)
        self.assertEqual(executor.metrics()['by_host']['server1']['reused'], 0)
        self.assertEqual(view.ingest(task.job, proposal, calls=1)['status'], O.PASS)
        self.assertEqual(executor.metrics()['by_host']['server1']['reused'], 0)

        # Use the normal saved-result entry, rather than answer(), because
        # optimize loads prior_records before asking for any new proposals.
        view.prior_run_dirs = [self.run]
        candidate = dict(configuration_id='first', ready=True, prices=[Fraction(1)]*4)
        known, refused = view.prior_records([candidate])
        self.assertEqual(refused, [])
        self.assertEqual(known[('first', proposal['vector'])]['source'], 'PRIOR_ROUND_JOB')
        self.assertEqual(known[('first', proposal['vector'])]['calls'], 0)
        self.assertEqual(executor.metrics()['by_host']['server1']['reused'], 1)

        # Reloading the catalog and directly ingesting/answering the same saved
        # job within this invocation must not inflate the host's reuse count.
        view.prior_records([candidate])
        view.ingest(task.job, proposal, calls=0)
        view.answer(proposal)
        self.assertEqual(executor.metrics()['by_host']['server1']['reused'], 1)
        events = [json.loads(line) for line in (self.run/'PROGRESS.jsonl').read_text().splitlines()]
        reused = [event for event in events if event['event'] == 'RESULT_REUSE']
        self.assertEqual(len(reused), 1)
        self.assertEqual((reused[0]['host_id'], reused[0]['job']), ('server1', task.job.name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
