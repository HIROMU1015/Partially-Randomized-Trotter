"""Synthetic controller failures at durable claim/completion boundaries.

No adapter is launched: outcomes stand in for already completed artificial jobs.
"""
import base64
import errno
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_q3_coordinator as Q
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

    def executor(self, capacity=2, **distributed):
        result = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent=capacity, budgets={'PR_PAULI': 8},
            distributed=dict(port=0, affinity_wait_seconds=0, **distributed))
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

    def test_standby_probe_allows_completion_and_rechecks_stop_before_reserving(self):
        executor = self.executor(live_q3_readiness=True)
        self.enqueue(executor, 'owned')
        assignment = self.poll(executor)['assignments'][0]
        executor.distributed['host_eligibility'] = dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={})
        started, release, completed = threading.Event(), threading.Event(), threading.Event()

        def slow_probe(view, request, job, decision):
            started.set()
            if not release.wait(5):
                raise AssertionError('Completion was blocked by readiness')
            return dict(decision, eligible_host_ids=['server1'], host_fingerprints={'server1':'verified'})

        def finish_owned():
            executor.rpc('/v1/complete', self.completion(assignment))
            completed.set()

        view = executor.for_method('PR_PAULI')
        waiting = self.proposal('standby')
        with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=slow_probe):
            standby = threading.Thread(target=view.prefetch_standby, args=(None,[waiting]))
            standby.start()
            receiver = threading.Thread(target=finish_owned)
            try:
                self.assertTrue(started.wait(2))
                receiver.start()
                self.assertTrue(completed.wait(1))
                Q.atomic_json(executor.run_dir/'STOP_REQUESTED.json', {'reason':'USER_REQUESTED_STOP'})
            finally:
                release.set()
                standby.join(5)
                if receiver.ident is not None:
                    receiver.join(5)
        self.assertIsNone(executor.store.row(view.job_dir(waiting).name))
        self.assertEqual(executor.store.row(assignment['job_id'])['state'], 'ACCEPTED')

    def test_joining_host_refreshes_queued_readiness_without_blocking_poll(self):
        executor = self.executor(live_q3_readiness=True,
                                 host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
        started, release = threading.Event(), threading.Event()
        fingerprint = {'fixture':'current'}

        def probe(view, request, job, decision):
            hosts = ['server1']
            if 'joined' in view.hosts:
                started.set()
                if not release.wait(5):
                    raise AssertionError('Joining host poll was blocked')
                hosts.append('joined')
            return dict(decision, eligible_host_ids=hosts, host_fingerprints={h:Q.digest(fingerprint) for h in hosts})

        with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=probe):
            task = self.enqueue(executor, 'new-host')
            poll = dict(host_id='joined', slots=1, source_sha=self.source,
                        host_fingerprints={task.job.name:fingerprint})
            try:
                self.assertEqual(executor.rpc('/v1/poll', poll)['assignments'], [])
                self.assertTrue(started.wait(2))
            finally:
                release.set()
            with executor.state.cv:
                self.assertTrue(executor.state.cv.wait_for(lambda: not executor.live_readiness_inflight, timeout=3))
            claimed = executor.rpc('/v1/poll', poll)['assignments']
            self.assertEqual([row['job_id'] for row in claimed], [task.job.name])
            executor.rpc('/v1/complete', dict(self.completion(claimed[0]), host_id='joined'))

    def test_no_eligible_host_is_rechecked_without_reserving_twice(self):
        executor = self.executor(live_q3_readiness=True,
                                 host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
        view = executor.for_method('PR_PAULI')
        proposal = self.proposal('on-retry')
        fingerprint = {'fixture':'current'}

        def recovered(view, request, job, decision):
            return dict(decision, eligible_host_ids=['server1'],
                        host_fingerprints={'server1':Q.digest(fingerprint)})

        with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=lambda _, rq, job, result: result) as probe:
            self.assertFalse(view._enqueue(proposal))
            self.assertFalse(view._enqueue(proposal))
            self.assertEqual(probe.call_count, 1)
            self.assertIsNone(executor.store.row(view.job_dir(proposal).name))
        cache = executor.eligibility_policy_cache['live_checks']
        key, (checked, hosts, result) = next(iter(cache.items()))
        cache[key] = (checked - 11, hosts, result)
        with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=recovered) as probe:
            task = view._enqueue(proposal)
            self.assertIs(view._enqueue(proposal), task)
            self.assertEqual(probe.call_count, 1)
            assignment, = executor.rpc('/v1/poll', dict(host_id='server1', slots=1, source_sha=self.source,
                host_fingerprints={task.job.name:fingerprint}))['assignments']
            executor.rpc('/v1/complete', self.completion(assignment))
        self.assertEqual(executor.store.counts('PR_PAULI', started_only=True), 1)

    def test_standby_refills_after_another_polling_host_joins_and_finishes(self):
        executor = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent='auto', budgets={'PR_PAULI': 4},
            distributed=dict(port=0, affinity_wait_seconds=0, work_conserving_search=True))
        self.executors.append(executor)
        view = executor.for_method('PR_PAULI')
        proposals = [self.proposal(name) for name in ('demand', 'ready-one', 'ready-two', 'ready-three', 'outside-budget')]
        view.prefetch_standby(proposals[0], proposals[1:])
        first = self.poll(executor)['assignments'][0]
        self.assertEqual(first['request']['configuration_id'], 'demand')
        answers = []
        waiter = threading.Thread(target=lambda: answers.append(view.answer(proposals[0])))
        waiter.start()
        try:
            for proposal in proposals[1:-1]:
                reply = executor.rpc('/v1/poll', dict(host_id='joined', source_sha=self.source, slots=1, capacity=1))
                assignment = reply['assignments'][0]
                self.assertEqual(assignment['request']['configuration_id'], proposal['configuration_id'])
                self.assertEqual(executor.store.row(first['job_id'])['state'], 'ASSIGNED')
                executor.rpc('/v1/complete', dict(self.completion(assignment), host_id='joined'))
                with executor.state.cv:
                    executor.state.cv.wait_for(lambda: executor.store.counts('PR_PAULI') == 4 or
                        any(t.status == 'QUEUED' for t in executor.state.tasks.values()), timeout=3)
            self.assertEqual(executor.store.counts('PR_PAULI', started_only=True), 4)
            self.assertEqual(executor.rpc('/v1/poll', dict(host_id='joined', source_sha=self.source,
                                                         slots=1, capacity=1))['assignments'], [])
        finally:
            executor.rpc('/v1/complete', self.completion(first))
            waiter.join(timeout=3)
        self.assertEqual(answers[0]['status'], O.PASS)

    def test_work_conserving_records_distinct_idle_slot_reasons(self):
        from trottertracks.rpe_preparation_queue import normalize_preparation_request
        from trottertracks.rpe_q3_timing import read_lines
        executor = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent='auto', budgets={'PR_PAULI': 1},
            distributed=dict(port=0, live_ready_refill=True, work_conserving_search=True,
                             host_stage_capabilities={'waiting-host': ['q3'], 'prep-host': ['signal']}))
        self.executors.append(executor)

        def poll(host, observe_only=False):
            return executor.rpc('/v1/poll', dict(host_id=host, source_sha=self.source, slots=1, capacity=1,
                                                observe_only=observe_only))

        poll('waiting-host')
        prep = normalize_preparation_request(dict(route={'kind': 'preparation'}, method='ARTIFICIAL',
            preparation=dict(stage='signal', provider='ARTIFICIAL_SIGNAL', artificial=True,
                source=executor.source_manifest, identity='idle-reason', outputs=['products'],
                command=['@PYTHON@', '@SOURCE@/scripts/artificial_preparation_leaf.py'])))
        executor.submit_preparation(prep, budget=1, approved=True)
        self.assertFalse(poll('waiting-host')['assignments'])
        assignment = poll('prep-host')['assignments'][0]
        executor.rpc('/v1/state', dict(host_id='prep-host', job_id=assignment['job_id'],
                     attempt_id=assignment['attempt_id'], state='RUNNING'))
        poll('waiting-host')
        self.enqueue(executor, 'waits-for-assignment')
        poll('waiting-host', observe_only=True)
        snapshots = read_lines(self.run/'PROGRESS.jsonl')
        reasons = {event.get('hosts', {}).get('waiting-host', {}).get('idle_reason') for event in snapshots}
        self.assertTrue({'CANDIDATE_SHORTAGE', 'PREPARATION_WAIT', 'ELIGIBLE_HOST_SHORTAGE', 'ASSIGNMENT_WAIT'} <= reasons)

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
        evidence = b'ARTIFICIAL raw evidence\n' * 100_000
        payload['files']['raw/RESULT.json'] = base64.b64encode(evidence).decode()
        link = Q.os.link

        def interrupt_before_outcome(source, destination):
            if source.name == 'OUTCOME.json':
                raise OSError('ARTIFICIAL publish interruption')
            return link(source, destination)

        with patch.object(Q.os, 'link', side_effect=interrupt_before_outcome):
            with self.assertRaises(OSError):
                executor.rpc('/v1/complete', payload)
        self.assertFalse((task.job/'OUTCOME.json').exists())
        stage = self.run/'distributed'/'transfers'/assigned['job_id']/assigned['attempt_id']
        self.assertTrue((task.job/'raw/RESULT.json').samefile(stage/'raw/RESULT.json'))
        self.close_one(executor)
        resumed = self.executor()
        for name in payload['files']:
            self.assertTrue((task.job/name).samefile(stage/name), name)
        self.assertEqual((task.job/'raw/RESULT.json').read_bytes(), evidence)
        answer = resumed.for_method('PR_PAULI').answer(proposal)
        self.assertEqual(answer['status'], O.PASS)
        self.assertEqual(answer['calls'], 0)
        self.assertEqual(resumed.store.row(task.job.name)['attempt_id'], assigned['attempt_id'])
        self.assertEqual(resumed.store.counts('PR_PAULI', started_only=True), 1)
        self.assertEqual(self.poll(resumed)['assignments'], [])
        self.assertEqual([e['event'] for e in resumed.store.events()].count('ACCEPT'), 1)
        self.assertEqual([e['event'] for e in resumed.store.events()].count('ADOPT'), 1)

    def test_transfer_can_publish_to_another_filesystem_and_reject_changed_legacy_copy(self):
        stage = self.run/'transfer'
        stage.mkdir(parents=True)
        raw = b'ARTIFICIAL exact bytes'
        (stage/'OUTCOME.json').write_bytes(raw)
        job = self.run/'job'
        with patch.object(Q.os, 'link', side_effect=OSError(errno.EXDEV, 'artificial mount')):
            Q.publish_staged_files(stage, job)
        outcome = job/'OUTCOME.json'
        self.assertEqual(outcome.read_bytes(), raw)
        self.assertFalse(outcome.samefile(stage/'OUTCOME.json'))
        # Previous releases retain separate copies. Replay compares the
        # exact bytes and leaves both file identities intact.
        Q.publish_staged_files(stage, job)
        self.assertFalse(outcome.samefile(stage/'OUTCOME.json'))
        outcome.write_bytes(b'changed'.ljust(len(raw), b' '))
        with self.assertRaisesRegex(ValueError, 'Changed immutable artifact: OUTCOME.json'):
            Q.publish_staged_files(stage, job)
        self.assertEqual((stage/'OUTCOME.json').read_bytes(), raw)

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
        view.optimizer_history_directory = self.run/'cache'/'prior-q3'
        candidate = dict(configuration_id='first', ready=True, prices=[Fraction(1)]*4)
        known, refused = view.prior_records([candidate])
        self.assertEqual(refused, [])
        self.assertEqual(known[('first', proposal['vector'])]['source'], 'PRIOR_ROUND_JOB')
        self.assertEqual(known[('first', proposal['vector'])]['calls'], 0)
        self.assertEqual(executor.metrics()['by_host']['server1']['reused'], 1)

        # Reloading the catalog and directly ingesting/answering the same saved
        # job within this invocation must not inflate the host's reuse count.
        with patch.object(view, 'ingest', side_effect=AssertionError('Cached evidence should not be revalidated')), \
             patch.object(view, 'record_prior_result_reuse', wraps=view.record_prior_result_reuse) as reuse:
            self.assertEqual(view.prior_records([candidate]), (known, refused))
            reuse.assert_called_once_with(task.job)
        view.ingest(task.job, proposal, calls=0)
        view.answer(proposal)
        self.assertEqual(executor.metrics()['by_host']['server1']['reused'], 1)
        events = [json.loads(line) for line in (self.run/'PROGRESS.jsonl').read_text().splitlines()]
        reused = [event for event in events if event['event'] == 'RESULT_REUSE']
        self.assertEqual(len(reused), 1)
        self.assertEqual((reused[0]['host_id'], reused[0]['job']), ('server1', task.job.name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
