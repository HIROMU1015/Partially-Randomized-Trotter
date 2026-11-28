"""Synthetic controller failures at durable claim/completion boundaries.

No adapter is launched: outcomes stand in for already completed artificial jobs.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import errno
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_q3_coordinator as Q
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
from trottertracks import rpe_scaled_shots as S
from test_rpe_optimizer_concurrent import fixture


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

    def test_pbs_handoff_preserves_unclaimed_job_for_resume_without_reselection(self):
        first = self.executor()
        queued = self.enqueue(first, 'queued-before-handoff')
        job = queued.job.name
        (self.run/'STOP_REQUESTED.json').write_text(json.dumps({'reason':'PBS_BLOCK_HANDOFF'}))
        self.assertTrue(first._user_stop_requested())
        self.assertEqual(queued.status,'CANCELLED')  # only invocation-local
        self.assertEqual(first.store.row(job)['state'],'QUEUED')
        self.assertIsNone(first.store.row(job)['attempt_id'])
        self.close_one(first)
        (self.run/'STOP_REQUESTED.json').unlink()
        resumed = self.executor()
        self.assertEqual(resumed.store.row(job)['state'],'QUEUED')
        assignment, = self.poll(resumed,1)['assignments']
        self.assertEqual(assignment['job_id'],job)
        self.assertEqual(resumed.store.counts('PR_PAULI'),1)

    def test_actual_user_stop_still_cancels_unclaimed_job(self):
        executor=self.executor()
        queued=self.enqueue(executor,'queued-before-stop')
        (self.run/'STOP_REQUESTED.json').write_text(json.dumps({'reason':'USER_REQUESTED_STOP'}))
        self.assertTrue(executor._user_stop_requested())
        self.assertEqual(executor.store.row(queued.job.name)['state'],'CANCELLED')

    def test_historical_reuse_count_follows_append_and_log_replacement(self):
        self.run.mkdir()
        log = self.run / 'PROGRESS.jsonl'
        row = b'{"event":"RESULT_REUSE","host_id":"server1"}'
        log.write_bytes(row + b'\n' + row[:8])
        executor = SimpleNamespace(run_dir=self.run, state=SimpleNamespace(), lock=threading.Lock())
        count = lambda: dict(DistributedExecutor._reuse_counts(executor))
        self.assertEqual(count(), {'server1': 1})
        with log.open('ab') as stream:
            stream.write(row[8:] + b'\n')
        self.assertEqual(count(), {'server1': 2})
        self.assertEqual(count(), {'server1': 2})
        replacement = log.with_suffix('.new')
        replacement.write_bytes(row.replace(b'server1', b'server2') + b'\n')
        replacement.replace(log)
        self.assertEqual(count(), {'server2': 1})

    def test_reuse_history_deduplicates_same_job_across_invocations(self):
        from trottertracks.rpe_optimizer_concurrent import reuse_history, queue_wait_signature
        self.run.mkdir()
        log = self.run/'PROGRESS.jsonl'
        log.write_text('\n'.join(json.dumps(dict(event='RESULT_REUSE', host_id='server1', job='original', invocation=i))
                                 for i in (1, 2))+'\n')
        counts, jobs = reuse_history(log)
        self.assertEqual(dict(counts), {'server1': 1})
        self.assertEqual(jobs, {'original'})
        payload = {'hosts': {'server1': {'storage': {'used_bytes': 10, 'filesystem_free_bytes': 20,
                                                   'reason': None, 'accepting_new_jobs': True}}}}
        changed = deepcopy(payload)
        changed['hosts']['server1']['storage']['used_bytes'] = 11
        self.assertEqual(queue_wait_signature(payload), queue_wait_signature(changed))
        changed['hosts']['server1']['storage']['accepting_new_jobs'] = False
        self.assertNotEqual(queue_wait_signature(payload), queue_wait_signature(changed))

    def test_delayed_result_file_write_does_not_block_independent_poll(self):
        executor = self.executor()
        self.enqueue(executor, 'slow-transfer')
        first, = self.poll(executor, slots=1)['assignments']
        second = self.enqueue(executor, 'ready-on-other-host')
        entered, release = threading.Event(), threading.Event()
        install = Q.install_files

        def slower_install(directory, files):
            entered.set()
            self.assertTrue(release.wait(5))
            install(directory, files)

        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(Q, 'install_files', side_effect=slower_install):
            completion = pool.submit(executor.rpc, '/v1/complete', self.completion(first))
            try:
                self.assertTrue(entered.wait(3))
                poll = pool.submit(executor.rpc, '/v1/poll', dict(host_id='other', slots=1, source_sha=self.source))
                assigned, = poll.result(timeout=3)['assignments']
                self.assertEqual(assigned['job_id'], second.job.name)
                self.assertFalse(completion.done())
            finally:
                release.set()
            self.assertEqual(completion.result(timeout=3), dict(ok=True, duplicate=False))

    def executor(self, capacity=2, **distributed):
        result = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent=capacity, budgets={'PR_PAULI': 8},
            distributed=dict(port=0, affinity_wait_seconds=0, **distributed))
        self.executors.append(result)
        return result

    def test_local_second_slot_only_pairs_sized_price_calibrations(self):
        from types import SimpleNamespace
        trial = DistributedExecutor.__new__(DistributedExecutor)
        profile = {'host_id': 'local', 'preparation_profiles': [dict(stage='compile',
                   provider='DF_CALIBRATION', profile=dict(host_id='local', task_cpu_count=4))]}
        trial.distributed = dict(local_worker=profile, local_calibration_pair=dict(memory_reserve_bytes=100))
        calib = dict(route={'kind': 'preparation'}, preparation=dict(stage='compile', provider='DF_CALIBRATION',
                     source={}, parallel_df_price_axes=True))
        trial._owned_attempts = lambda: [dict(host_id='local', job_id='first')]
        trial.store = SimpleNamespace(row=lambda _: {'request': calib})
        with patch('trottertracks.rpe_q3_resources.resource_snapshot', return_value={
                'machine_mem_available_bytes': 200, 'allowed_cpu_count': 8}) as resources:
            self.assertTrue(trial._local_calibration_pair_allows('local', calib))
            self.assertFalse(trial._local_calibration_pair_allows('local', {'route': {'kind': 'Q3'}}))
            resources.return_value['allowed_cpu_count'] = 7
            self.assertFalse(trial._local_calibration_pair_allows('local', calib))
            resources.return_value.update(allowed_cpu_count=8, remaining_group_memory_bytes=99)
            self.assertFalse(trial._local_calibration_pair_allows('local', calib))
            trial.distributed['local_calibration_pair']['pf_preparation'] = True
            profile['preparation_profiles'].append(dict(stage='audit', provider='PF_AUDIT',
                profile=dict(host_id='local', task_cpu_count=1)))
            audit = dict(route={'kind': 'preparation'},
                         preparation=dict(stage='audit', provider='PF_AUDIT', source={}))
            trial.store.row = lambda _: {'request': audit}
            resources.return_value.update(remaining_group_memory_bytes=100)
            self.assertTrue(trial._local_calibration_pair_allows('local', audit))

    def test_preparation_preference_yields_only_while_worker_can_take_the_task(self):
        import time
        ex = self.executor(preparation_host_preference=dict(
            providers=['PF_AUDIT'], host='local0-wsl', seconds=12))
        row = dict(job_id='fixture', enqueued_epoch=time.time(),
                   request=dict(preparation=dict(provider='PF_AUDIT')))
        ex.rpc('/v1/poll', dict(host_id='local0-wsl', slots=1, capacity=1,
                               source_sha=self.source, observe_only=True))
        policy, error = ex._host_admission_policy()
        self.assertTrue(ex._wait_for_preparation_preference('local', row, policy, error))
        self.assertFalse(ex._wait_for_preparation_preference('local0-wsl', row, policy, error))
        ex.preparation_pending['fixture'] = dict(failures={'local0-wsl': 'HOLD'})
        self.assertFalse(ex._wait_for_preparation_preference('local', row, policy, error))
        ex.preparation_pending.clear()
        row['enqueued_epoch'] -= 13
        self.assertFalse(ex._wait_for_preparation_preference('local', row, policy, error))
        prep = dict(method='ARTIFICIAL', route=dict(kind='preparation'),
                    preparation=dict(stage='audit', provider='PF_AUDIT', artificial=True,
                                     command=['ARTIFICIAL_ONLY'], outputs=['receipt.json']))
        job = ex.submit_preparation(prep, approved=True, budget=1)
        poll = dict(slots=1, capacity=1, source_sha=self.source)
        self.assertEqual(ex.rpc('/v1/poll', dict(poll, host_id='local'))['assignments'], [])
        assignments = ex.rpc('/v1/poll', dict(poll, host_id='local0-wsl'))['assignments']
        self.assertEqual([item['job_id'] for item in assignments], [job])

    def test_slot_offer_cannot_claim_a_different_resource_class(self):
        server = SimpleNamespace(server_address=('127.0.0.1', 1), shutdown=lambda: None,
                                 server_close=lambda: None)
        with patch.object(Q, 'serve', return_value=(server, SimpleNamespace(join=lambda: None))):
            ex = self.executor()
        jobs = {}
        for provider in ('COLD', 'MEASURED_A', 'MEASURED_B'):
            request = dict(method='ARTIFICIAL', route=dict(kind='preparation'),
                preparation=dict(stage='audit', provider=provider, artificial=True,
                                 command=['ARTIFICIAL_ONLY'], outputs=['receipt.json']))
            jobs[provider] = ex.submit_preparation(request, approved=True, budget=3)
        value = dict(host_id='local', slots=2, capacity=2, source_sha=self.source,
                     resource_report=dict(admitted_task_classes=['preparation:audit:MEASURED_A',
                                                                 'preparation:audit:MEASURED_B'],
                         admitted_task_class_slots={'preparation:audit:MEASURED_A':0,'preparation:audit:MEASURED_B':1}))
        assignments = ex.rpc('/v1/poll', value)['assignments']
        self.assertEqual({item['job_id'] for item in assignments}, {jobs['MEASURED_B']})
        self.assertEqual(ex.store.row(jobs['MEASURED_A'])['state'], 'QUEUED')
        self.assertEqual(ex.store.row(jobs['COLD'])['state'], 'QUEUED')

    def close_one(self, executor):
        # The fixture starts no processes, so release only its synthetic tasks
        # to avoid hanging test teardown when testing a missing _finished call.
        from trottertracks.rpe_optimizer_concurrent import transition_task
        with executor.state.cv:
            for task in executor.state.tasks.values():
                if task.status == 'RUNNING':
                    transition_task(executor.state,task,'DONE'); task.event.set()
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

    def test_hchain_start_fence_allows_replay_but_not_new_claims_until_ranked(self):
        executor = self.executor(hchain_initialize_priority=True)
        first = self.enqueue(executor, 'before-ranking')
        second = self.enqueue(executor, 'second-before-ranking')
        self.assertEqual(self.poll(executor, slots=2)['assignments'], [])
        self.assertEqual(executor.store.row(first.job.name)['state'], 'QUEUED')
        executor.set_hchain_preparation_priority({})
        owned, = self.poll(executor, slots=1)['assignments']
        executor.hchain_priority_pending = True  # restored owner's replay during next Resume
        replay = self.poll(executor, slots=2)['assignments']
        self.assertEqual(len(replay), 1)
        self.assertEqual(replay[0]['attempt_id'], owned['attempt_id'])
        self.assertEqual(executor.store.row(second.job.name)['state'], 'QUEUED')
        self.assertFalse(executor.rpc('/v1/complete', self.completion(owned))['duplicate'])
        executor.set_hchain_preparation_priority({})
        next_assignment, = self.poll(executor, slots=1)['assignments']
        self.assertEqual(next_assignment['job_id'], second.job.name)
        executor.rpc('/v1/complete', self.completion(next_assignment))

    def test_q3_resend_allows_only_appended_upload_timings(self):
        executor = self.executor()
        task = self.enqueue(executor, 'q3-resend')
        assignment, = self.poll(executor)['assignments']
        payload = self.completion(assignment)
        name = 'TRANSPORT_TIMING.jsonl'
        original = b'{"event":"RESULT_BUNDLE_READY"}\n'
        encode = lambda raw: base64.b64encode(raw).decode('ascii')
        payload['files'][name] = encode(original)
        self.assertFalse(executor.rpc('/v1/complete', payload)['duplicate'])
        accepted = executor.store.row(task.job.name)
        retry = dict(payload, files=dict(payload['files'], **{name: encode(original + b'{"event":"WORKER_TRANSFER_START"}\n')}))
        self.assertTrue(executor.rpc('/v1/complete', retry)['duplicate'])
        for change in ({'OUTCOME.json': encode(b'{}')}, {name: encode(b'{"event":"WORKER_TRANSFER_START"}\n')},
                       {name: encode(original + b'{"event":"ADAPTER_PROCESS_END"}\n')}):
            with self.assertRaisesRegex(ValueError, 'Duplicate completion changed artifacts'):
                executor.rpc('/v1/complete', dict(retry, files={**retry['files'], **change}))
        self.assertEqual(executor.store.row(task.job.name), accepted)
        self.assertEqual([row['event'] for row in executor.store.events()].count('ACCEPT'), 1)
        saved = self.run/'distributed'/'transfers'/task.job.name/assignment['attempt_id']/name
        self.assertEqual(saved.read_bytes(), original)

    def test_lost_worker_contact_yields_only_owned_attempt_and_recovers_after_restart(self):
        executor = self.executor()
        first = self.enqueue(executor, 'first')
        second = self.enqueue(executor, 'independent')
        assignment, = self.poll(executor)['assignments']
        self.assertEqual(assignment['job_id'], first.job.name)
        old_attempt = executor.store.row(first.job.name)['attempt_id']
        with patch('trottertracks.rpe_optimizer_concurrent.time.time', return_value=time.time()+600):
            executor._check_worker_contact()
        held = executor.for_method('PR_PAULI').answer(self.proposal('first'))
        self.assertTrue(O.is_local_unknown(held))
        self.assertEqual(held['reason'], 'WORKER_CONTACT_LOST')
        self.assertEqual(executor.store.row(first.job.name)['state'], 'ASSIGNED')
        other, = executor.rpc('/v1/poll', dict(host_id='server2', slots=1, source_sha=self.source))['assignments']
        self.assertEqual(other['job_id'], second.job.name)
        executor.rpc('/v1/complete', dict(self.completion(other), host_id='server2'))
        self.assertEqual(executor.for_method('PR_PAULI').answer(self.proposal('independent'))['status'], O.PASS)
        executor.finish()
        resumed = self.executor()
        replay, = self.poll(resumed)['assignments']
        self.assertEqual(replay['attempt_id'], old_attempt)
        self.assertEqual(resumed.store.counts('PR_PAULI', started_only=True), 2)
        resumed.rpc('/v1/complete', self.completion(replay))
        self.assertEqual(resumed.for_method('PR_PAULI').answer(self.proposal('first'))['status'], O.PASS)
        self.assertTrue(resumed.rpc('/v1/complete', self.completion(replay))['duplicate'])
        self.assertEqual(sum(e['event'] == 'ADOPT' for e in resumed.store.events()), 2)

    def test_native_durable_owner_does_not_consume_next_delivery_window(self):
        executor=self.executor(capacity=2)
        first=self.enqueue(executor,'first')
        owned,=self.poll(executor,slots=1)['assignments']
        second=self.enqueue(executor,'second')
        result=executor.rpc('/v1/poll',dict(host_id='server1',slots=1,capacity=2,
            cached_prefixes=[],source_sha=self.source,durable_assignments=[[owned['job_id'],owned['attempt_id'],owned['request_sha256']]]))
        self.assertEqual([a['job_id'] for a in result['assignments']],[second.job.name])
        self.assertEqual(executor.store.row(first.job.name)['attempt_id'],owned['attempt_id'])

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

    def test_coordinator_first_leaves_remote_work_and_runs_local_preparation(self):
        from trottertracks.rpe_preparation_queue import normalize_preparation_request
        executor = self.executor(controller_queue_first=True,
                                 host_stage_capabilities={'local0-wsl': ['q3']})

        def poll(host, observe_only=False):
            return executor.rpc('/v1/poll', dict(host_id=host, source_sha=self.source, slots=1,
                                                capacity=1, observe_only=observe_only))['assignments']

        poll('local0-wsl', observe_only=True)
        evaluation = self.enqueue(executor, 'remote-first')
        self.enqueue(executor, 'surplus')
        prep = normalize_preparation_request(dict(route={'kind': 'preparation'}, method='ARTIFICIAL',
            preparation=dict(stage='signal', provider='ARTIFICIAL_SIGNAL', artificial=True,
                source=executor.source_manifest, identity='replenish', outputs=['products'],
                command=['@PYTHON@', '@SOURCE@/scripts/artificial_preparation_leaf.py'])))
        prep_id = executor.submit_preparation(prep, budget=1, approved=True)
        self.assertEqual([a['job_id'] for a in poll('local')], [prep_id])
        self.assertEqual([a['job_id'] for a in poll('local0-wsl')], [evaluation.job.name])

    def test_coordinator_prefetch_starts_checks_before_reserving_jobs(self):
        executor = self.executor(controller_queue_first=True, live_q3_readiness=True)
        executor.live_readiness_workers = 2
        view = executor.for_method('PR_PAULI')
        first, second = self.proposal('demand'), self.proposal('standby')
        saved = self.proposal('already-known')

        def checked(request, job, *, prefetch=False, **kwargs):
            if prefetch:
                self.assertIsNone(executor.store.row(job))
            return None

        with patch.object(view, '_eligibility_for', side_effect=checked) as check, \
                patch.object(view, 'has_job', side_effect=lambda p: p == saved):
            view.prefetch_standby(first, [saved, second])
        calls = [call for call in check.call_args_list if call.kwargs.get('prefetch')]
        self.assertEqual([call.args[1] for call in calls[:2]], [view.job_dir(p).name for p in (first, second)])
        self.assertTrue(all(not call.kwargs['blocking'] for call in calls))
        self.assertTrue(all(executor.store.row(view.job_dir(p).name) for p in (first, second)))

    def test_coordinator_does_not_promote_q3_ahead_of_older_ready_preparation(self):
        from trottertracks.rpe_preparation_queue import normalize_preparation_request
        executor = self.executor(controller_queue_first=True, live_ready_refill=True, work_conserving_search=True)
        prep = normalize_preparation_request(dict(route={'kind': 'preparation'}, method='ARTIFICIAL',
            preparation=dict(stage='signal', provider='ARTIFICIAL_SIGNAL', artificial=True,
                source=executor.source_manifest, identity='independent', outputs=['products'],
                command=['@PYTHON@', '@SOURCE@/scripts/artificial_preparation_leaf.py'])))
        prepare_id = executor.submit_preparation(prep, budget=1, approved=True)
        evaluation = self.enqueue(executor, 'demand')
        view = executor.for_method('PR_PAULI')
        ahead = view._enqueue(dict(self.proposal('conditional'),
                                  origin=dict(kind='CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES')))
        self.assertEqual([t.job.name for t in executor._ready_queue()], [prepare_id, evaluation.job.name, ahead.job.name])

    def test_preparation_binding_mismatch_does_not_block_reconciliation_or_q3(self):
        executor = self.executor()
        prep = dict(route=dict(kind='preparation'), method='ARTIFICIAL',
                    preparation=dict(stage='compile', provider='ARTIFICIAL_COMPILE', artificial=True,
                    source=executor.source_manifest, identity='compile', outputs=['products'],
                    command=['@PYTHON@', '@SOURCE@/scripts/artificial_preparation_leaf.py']))
        job_id = executor.submit_preparation(prep, budget=1, approved=True)
        assignment, = self.poll(executor)['assignments']
        self.assertEqual(assignment['job_id'], job_id)
        payload = dict(host_id='server1', job_id=job_id, attempt_id=assignment['attempt_id'],
                       request_sha256=assignment['request_sha256'])
        outcome = dict(request_id=job_id, request_sha256=assignment['request_sha256'], stage='audit',
                       source_manifest=executor.source_manifest, status='PREPARATION_COMPLETE', artifacts={})
        files = {'OUTCOME.json': outcome, 'PROCESS_RETURN.json': dict(returncode=0, signal=None)}
        payload['files'] = {name:base64.b64encode(json.dumps(value).encode()).decode()
                            for name,value in files.items()}
        with self.assertRaisesRegex(ValueError, 'stage/SOURCE differs'):
            executor.rpc('/v1/complete', payload)
        result = executor.preparation_result(job_id, wait=True, adopt=False)
        self.assertEqual(result['reason'], 'WORKER_RESULT_MISMATCH')
        self.assertIsNone(executor.store.row(job_id)['adopted_epoch'])
        self.assertEqual(executor.store.row(job_id)['attempt_id'], assignment['attempt_id'])
        other = self.enqueue(executor, 'independent')
        assigned, = executor.rpc('/v1/poll', dict(host_id='other', slots=1, source_sha=self.source))['assignments']
        self.assertEqual(assigned['job_id'], other.job.name)

    def test_shared_spool_outside_run_notification_creates_no_owner(self):
        from trottertracks.rpe_pbs_worker import AllocationWorker

        executor = self.executor()
        old = dict(host_id='server1', job_id='prep-older-run', attempt_id='saved-attempt',
                   state='UNKNOWN')
        self.assertEqual(executor.rpc('/v1/state', old),
                         dict(ok=True, disposition='STATE_OUTSIDE_CURRENT_RUN'))
        self.assertIsNone(executor.store.row(old['job_id']))
        with self.assertRaisesRegex(ValueError, 'Different execution host/attempt'):
            executor.rpc('/v1/complete', old)
        queued = self.enqueue(executor, 'current-request')
        assignment, = self.poll(executor)['assignments']
        same_job = dict(old, job_id=queued.job.name)
        for stale in (same_job, dict(same_job, host_id='other', attempt_id=assignment['attempt_id'])):
            self.assertEqual(executor.rpc('/v1/state', stale),
                             dict(ok=True, disposition='STATE_OUTSIDE_CURRENT_RUN'))
            with self.assertRaisesRegex(ValueError, 'Different execution host/attempt'):
                executor.rpc('/v1/complete', stale)
        self.assertEqual(executor.store.row(queued.job.name)['attempt_id'], assignment['attempt_id'])
        self.assertEqual(executor.store.row(queued.job.name)['state'], 'ASSIGNED')

        worker = AllocationWorker.__new__(AllocationWorker)
        worker.host_id, worker.reported = same_job['host_id'], {}
        attempt = self.run/'synthetic-old-attempt'
        with patch.object(worker, '_reconcile', return_value=dict(state='UNKNOWN', detail={})), \
                patch.object(worker, 'transport', side_effect=executor.rpc, create=True) as transport:
            worker._service_attempt(attempt, same_job)
            worker._service_attempt(attempt, same_job)
            transport.assert_called_once()
        self.assertEqual(executor.rpc('/v1/state', dict(same_job, attempt_id=assignment['attempt_id'],
                                                        state='RUNNING')), dict(ok=True))
        independent = self.enqueue(executor, 'independent-after-old-state')
        next_assignment, = executor.rpc('/v1/poll', dict(host_id='other', slots=1,
                                                         source_sha=self.source))['assignments']
        self.assertEqual(next_assignment['job_id'], independent.job.name)

    def test_existing_queue_with_explicit_auto_round_keeps_auto_capacity(self):
        from trottertracks.rpe_q3_store import JobStore

        JobStore(self.run/'distributed'/'QUEUE.sqlite3')
        Q.atomic_json(self.run/'ROUND.json', {'executor': {'max_concurrent_jobs': 'auto'}})
        executor = self.executor(capacity='auto')
        self.assertTrue(executor.state.allocation_auto)
        self.assertEqual(executor.allocation_settings['max_concurrent_jobs'], 'auto')
        self.enqueue(executor, 'first')
        self.enqueue(executor, 'second')
        assignments = self.poll(executor, slots=2)['assignments']
        self.assertEqual(len(assignments), 2)

    def test_missing_worker_exit_is_held_without_blocking_independent_assignment(self):
        ex = self.executor(capacity='auto')
        proposal = self.proposal('unresolved')
        view = ex.for_method('PR_PAULI')
        task = view._enqueue(proposal, demand=True)
        assignment, = self.poll(ex)['assignments']
        owner = {key: assignment[key] for key in ('job_id', 'attempt_id')}
        owner['host_id'] = 'server1'
        detail = dict(reason='LAUNCH_STATE_UNCERTAIN_NO_ADAPTER_EXIT_RECEIPT', automatic_retry=False)
        ex.rpc('/v1/state', dict(owner, state='UNKNOWN', detail=detail))
        answer = view.answer(proposal)
        self.assertTrue(O.is_local_unknown(answer))
        self.assertEqual(answer['calls'], 1)
        view.hold_configuration(proposal, answer)
        second = self.enqueue(ex, 'independent')
        other, = ex.rpc('/v1/poll', dict(host_id='other', slots=1, source_sha=self.source))['assignments']
        self.assertEqual(other['job_id'], second.job.name)
        ex.rpc('/v1/complete', dict(self.completion(other), host_id='other'))
        self.assertEqual(ex.store.row(task.job.name)['state'], 'UNKNOWN')
        self.assertEqual(ex.store.row(task.job.name)['attempt_id'], assignment['attempt_id'])
        self.assertTrue(O.is_local_unknown(view.answer(proposal)))
        self.assertEqual(ex.metrics()['unresolved_q3_attempts'][0]['job_id'], task.job.name)
        self.assertFalse(self.poll(ex)['assignments'])
        ex.rpc('/v1/state', dict(owner, state='UNKNOWN', detail=dict(detail, resource_released=True)))
        third = self.enqueue(ex, 'same-host-after-exit')
        replacement, = self.poll(ex)['assignments']
        self.assertEqual(replacement['job_id'], third.job.name)
        self.assertTrue(O.is_local_unknown(view.answer(proposal)))
        self.assertEqual(ex.store.row(task.job.name)['attempt_id'], owner['attempt_id'])

    def test_standby_probe_allows_completion_and_rechecks_stop_before_reserving(self):
        executor = self.executor(live_q3_readiness=True)
        self.enqueue(executor, 'owned')
        assignment = self.poll(executor)['assignments'][0]
        executor.distributed['host_eligibility'] = dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={})
        started, release, completed = threading.Event(), threading.Event(), threading.Event()

        def slow_probe(view, request, job, decision, *, publish):
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

    def test_slow_prior_lookup_does_not_block_independent_completion(self):
        executor = self.executor()
        self.enqueue(executor, 'owned')
        assignment = self.poll(executor)['assignments'][0]
        started, release, completed = threading.Event(), threading.Event(), threading.Event()
        view = executor.for_method('PR_PAULI')
        waiting = self.proposal('waiting')

        def slow_prior(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise AssertionError('Completion was blocked by prior history')
            return None

        def finish_owned():
            executor.rpc('/v1/complete', self.completion(assignment))
            completed.set()

        with patch.object(view, 'prior_job', side_effect=slow_prior):
            standby = threading.Thread(target=view._enqueue, args=(waiting,))
            receiver = threading.Thread(target=finish_owned)
            standby.start()
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

        def probe(view, request, job, decision, *, publish):
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

        def recovered(view, request, job, decision, *, publish):
            return dict(decision, eligible_host_ids=['server1'],
                        host_fingerprints={'server1':Q.digest(fingerprint)})

        with patch('trottertracks.rpe_live_q3_readiness.probe',
                   side_effect=lambda _, rq, job, result, **kw: result) as probe:
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

    def test_late_host_pass_is_visible_to_scheduler_before_that_host_polls(self):
        executor = self.executor(live_q3_readiness=True,
                                 host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
        release = threading.Event()
        view = executor.for_method('PR_PAULI')

        def check(_, request, job, decision, *, publish):
            def ready(hosts):
                return dict(decision, eligible_host_ids=hosts,
                            host_fingerprints={h: Q.digest({'host': h}) for h in hosts},
                            host_decisions={h: {'status': 'PASS'} for h in hosts}, execution_profiles={})
            publish(ready(['server1']))
            if not release.wait(4):
                raise AssertionError('Expected the first host before the late PASS')
            return ready(['server1', 'pbs-0'])

        with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=check):
            try:
                task = view._enqueue(self.proposal('late-host'))
                original = executor.store.row(task.job.name)
                self.assertEqual(original['eligibility']['eligible_host_ids'], ['server1'])
            finally:
                release.set()
            with executor.state.cv:
                self.assertTrue(executor.state.cv.wait_for(lambda: not executor.live_readiness_inflight, 3))
        row = executor.store.row(task.job.name)
        self.assertEqual(row['eligibility']['eligible_host_ids'], ['server1', 'pbs-0'])
        self.assertEqual(task.supply_hosts, frozenset({'server1', 'pbs-0'}))
        self.assertEqual(row['request_sha256'], original['request_sha256'])
        self.assertEqual(row['state'], 'QUEUED')
        self.assertIsNone(row['attempt_id'])

    def test_first_verified_host_claims_before_other_host_finishes_its_probe(self):
        executor = self.executor(capacity=2, live_q3_readiness=True,
                                 host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
        view = executor.for_method('PR_PAULI')
        proposal = self.proposal('early-pass')
        local = {'fixture':'local'}
        remote = {'fixture':'remote'}
        released = threading.Event()

        def probe(executor, request, job, decision, *, publish):
            def snapshot(facts):
                return dict(decision, eligible_host_ids=list(facts),
                            host_fingerprints={host:Q.digest(value) for host,value in facts.items()},
                            host_decisions={host:{'status':'PASS'} for host in facts}, execution_profiles={})
            publish(snapshot({'server1':local}))
            if not released.wait(5):
                raise AssertionError('Assignment waited for the slow host')
            return snapshot({'server1':local, 'remote':remote})

        with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=probe):
            try:
                task = view._enqueue(proposal, demand=True)
                self.assertTrue(executor.live_readiness_inflight)
                poll = dict(host_id='server1', slots=1, source_sha=self.source,
                            host_fingerprints={task.job.name:local})
                assignment, = executor.rpc('/v1/poll', poll)['assignments']
                executor.rpc('/v1/complete', self.completion(assignment))
            finally:
                released.set()
            with executor.state.cv:
                self.assertTrue(executor.state.cv.wait_for(lambda: not executor.live_readiness_inflight, 3))
            answer = executor.rpc('/v1/poll', dict(poll, host_id='remote',
                                                 host_fingerprints={task.job.name:remote}))
            self.assertEqual(answer['assignments'], [])
        self.assertEqual(executor.store.counts('PR_PAULI', started_only=True), 1)

    def test_pbs_checks_standby_on_compute_worker_before_reserving(self):
        from trottertracks.rpe_pbs import PbsManager, stamp

        executor = self.executor(capacity='auto', live_q3_readiness=True,
                                 host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
        host, token, allocation_id = 'pbs-0', 'owned-allocation', 'artificial.pbs'
        worker = dict(host_id=host, profile={'host_id': host})
        settings = dict(remote_root='/unused', workers=[worker])
        manager = PbsManager(executor, settings, gateway=SimpleNamespace())
        executor.pbs_manager = manager
        try:
            manager.state['allocations'][token] = dict(
                spec=dict(profile=worker['profile'], reason={}), status='R', submitted_clock=stamp())
            observed = dict(accepting=True, job_id=allocation_id, hostname='compute-node',
                            pid=123, source_sha='fixed-worker', start_clock=stamp(), remaining_seconds=300)
            manager.allocations = [dict(token=token, host_id=host, job_id=allocation_id,
                                        status='R', worker=observed)]
            view = executor.for_method('PR_PAULI')
            proposal = self.proposal('prefetched-on-compute-node')
            job = view.job_dir(proposal)
            request = view.request(proposal, job, view.route(proposal['configuration_id']))
            facts = dict(schema='RPE_Q3_CURRENT_HOST_FINGERPRINT_V1', request_sha256=Q.digest(request))

            def check(_, rq, job_id, decision, *, publish):
                ready = dict(decision, eligible_host_ids=[host], host_decisions={host: {'status': 'PASS'}},
                             host_fingerprints={host: Q.digest(facts)}, execution_profiles={})
                publish(ready)
                return ready

            def poll(rpc_id, **payload):
                call = dict(request_id=rpc_id, endpoint='/v1/poll', clock=stamp(),
                            payload=dict(host_id=host, slots=2, capacity=2, source_sha=self.source,
                                resource_report=dict(pbs=dict(job_id=allocation_id, accepting=True,
                                    remaining_seconds=300, minimum_task_seconds=1)), **payload))
                response = manager._dispatch(token, call)
                self.assertTrue(response['ok'], response)
                return response['result']

            initial = poll('initial')
            self.assertEqual(initial['eligibility_challenges'], [])
            with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=check):
                view._eligibility_for(request, job.name, blocking=True)
                with executor.state.cv:
                    pending = executor.pbs_q3_check_queue[host]
                    first_received = pending[job.name][0]
                    second = self.proposal('later-pbs-preflight')
                    second_job = view.job_dir(second)
                    second_request = view.request(second, second_job, view.route(second['configuration_id']))
                    subsequent = dict(pending[job.name][2], request_sha256=Q.digest(second_request))
                    with patch.object(view, '_live_readiness_window', return_value=2):
                        view._plan_pbs_q3_checks(second_job.name, second_request, subsequent)
                        view._plan_pbs_q3_checks(job.name, request, pending[job.name][2])
                    self.assertEqual(list(pending), [job.name, second_job.name])
                    self.assertEqual(pending[job.name][0], first_received)
                    pending.pop(second_job.name)
                challenge, = poll('preflight')['eligibility_challenges']
                self.assertEqual(challenge['request_sha256'], Q.digest(request))
                self.assertNotIn('attempt_id', challenge)
                self.assertIsNone(executor.store.row(job.name))
                self.assertEqual(poll('proof', host_fingerprints={job.name: facts})['assignments'], [])
                self.assertIsNone(executor.store.row(job.name))
                task = view._enqueue(proposal, demand=True)
                self.assertEqual(task.job, job)
                assignment, = poll('next')['assignments']
                self.assertEqual(assignment['job_id'], job.name)
                replay, = poll('replay')['assignments']
                self.assertEqual(replay, assignment)
                receipt = dict(self.completion(assignment), host_id=host)
                executor.rpc('/v1/complete', receipt)
                self.assertEqual(poll('after-completion')['assignments'], [])
                self.assertEqual(executor.store.counts('PR_PAULI', started_only=True), 1)
                events = [json.loads(line) for line in (manager.directory/'EVENTS.jsonl').read_text().splitlines()]
                self.assertTrue(any(e['event']=='PBS_Q3_ASSIGNED_WITH_PREFETCHED_PROOF' and job.name in e['jobs']
                                    for e in events))
        finally:
            executor.pbs_manager = None
            manager.lock.close()

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

    def test_standby_prepares_the_next_wave_while_pbs_is_full(self):
        ex = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent='auto', budgets={'PR_PAULI': 5},
            distributed=dict(port=0, affinity_wait_seconds=0, work_conserving_search=True, ready_queue_waves=1))
        self.executors.append(ex)
        request = dict(host_id='pbs-0', slots=2, capacity=2, source_sha=self.source)
        ex.rpc('/v1/poll', dict(request, observe_only=True))
        view = ex.for_method('PR_PAULI')
        proposals = [self.proposal(f'wave-{i}') for i in range(5)]
        view.prefetch_standby(proposals[0], proposals[1:])
        first = ex.rpc('/v1/poll', request)['assignments']
        self.assertEqual(len(first), 2)
        self.assertEqual(sum(x['state'] == 'QUEUED' for x in ex.store.rows()), 2)
        for task in first:
            ex.rpc('/v1/complete', dict(self.completion(task), host_id='pbs-0'))
        second = ex.rpc('/v1/poll', request)['assignments']
        self.assertEqual(len(second), 2)
        self.assertTrue({a['job_id'] for a in first}.isdisjoint(a['job_id'] for a in second))

    def test_queue_observer_reports_actual_thread_cpu_and_wall_time(self):
        ex=self.executor(capacity=2)
        self.enqueue(ex,'profile-fixture')
        ex.rpc('/v1/poll',dict(host_id='server1',slots=1,capacity=1,source_sha=self.source,observe_only=True))
        ex._timing_queue()
        with ex.state.cv:
            self.assertTrue(ex.state.cv.wait_for(lambda:bool(getattr(ex.state,'queue_observer_profile',{})),timeout=5))
            profile=dict(ex.state.queue_observer_profile)
        self.assertGreater(profile['samples'],0)
        self.assertGreater(profile['thread_cpu_seconds'],0)
        self.assertGreater(profile['wall_seconds'],0)
        self.assertLessEqual(profile['report_cpu_seconds'],profile['thread_cpu_seconds'])

    def test_spare_slot_matching_does_not_hold_queue_lock_and_discards_stale_result(self):
        from concurrent.futures import ThreadPoolExecutor
        ex=self.executor(capacity=1)
        self.enqueue(ex,'fixture-demand')
        ex.rpc('/v1/poll',dict(host_id='server1',slots=1,capacity=1,source_sha=self.source,observe_only=True))
        entered=threading.Event();release=threading.Event()
        def slow_match(*args,**kwargs):
            entered.set();assert release.wait(3);return set()
        with patch.object(DistributedExecutor,'_match_ready_jobs',side_effect=slow_match), ThreadPoolExecutor() as pool:
            future=pool.submit(ex.has_spare_slot)
            try:
                self.assertTrue(entered.wait(3))
                self.assertTrue(ex.state.cv.acquire(timeout=1))
                try:ex.state.cv.notify_all()
                finally:ex.state.cv.release()
            finally:release.set()
            self.assertFalse(future.result(timeout=3))

    def test_full_standby_window_does_not_starve_another_method_demand(self):
        ex = self.executor(capacity=1)
        view = ex.for_method('PR_PAULI')
        other = SimpleNamespace(method='PR_DF')
        view.state.tasks['standby'] = SimpleNamespace(status='QUEUED', demand=False,
            rule={'kind':'artificial'}, view=other)
        view.state.active = 1
        try:
            self.assertFalse(view._standby_has_capacity())
            self.assertTrue(view._demand_has_queue_room())
            view.state.tasks['demand'] = SimpleNamespace(status='QUEUED', demand=True,
                rule={'kind':'artificial'}, view=view)
            self.assertFalse(view._demand_has_queue_room())
        finally:
            view.state.tasks.clear()
            view.state.active = 0

    def test_standby_method_turn_balances_only_ready_competing_methods(self):
        from collections import deque
        ex=self.executor(capacity=4)
        a=ex.for_method('PR_PAULI'); b=ex.for_method('PR_DF')
        a.standby=deque([{}]); b.standby=deque([{}])
        ex.state.standby_views={'PR_PAULI':a,'PR_DF':b}
        ex.state.tasks['standby']=SimpleNamespace(status='QUEUED',demand=False,rule={'kind':'artificial'},view=a)
        try:
            self.assertFalse(a._standby_method_turn())
            self.assertTrue(b._standby_method_turn())
            b._awaiting_probe_revision=getattr(ex.state,'probe_revision',0)
            self.assertTrue(a._standby_method_turn())
        finally:
            ex.state.tasks.clear();ex.state.standby_views.clear()

    def test_fleet_prepares_two_waves_for_each_host_before_pbs_first_offer(self):
        from types import SimpleNamespace
        ex = self.executor(capacity='auto', controller_queue_first=True, work_conserving_search=True)
        for host in ('local', 'server2', 'local0-wsl'):
            ex.rpc('/v1/poll', dict(host_id=host, slots=1, capacity=1, source_sha=self.source, observe_only=True))
        ex.state.budgets['PR_PAULI'] = 64
        view = ex.for_method('PR_PAULI')
        gib = 1024**3
        # The memory allowance fits four two-CPU Q3s, despite 96 configured CPUs.
        view.pbs_manager = SimpleNamespace(configured={'pbs-0': {'profile': {'dp_workers': 2}}},
            settings={'resources': {'ncpus': 96, 'mem': '33gb'}, 'allocation_sizing': {
                'per_task_resources': True, 'q3_startup_memory_bytes_per_task': 8*gib}},
            accepting_allocation=lambda *_: False)
        candidates = [self.proposal(f'candidate-{i}') for i in range(48)]
        view.prefetch_standby(candidates[0], candidates[1:])
        rows = ex.store.rows()
        self.assertEqual(len(rows), 3 + 2*(3+4))
        self.assertTrue(all(row['state'] == 'QUEUED' for row in rows))
        from trottertracks.rpe_optimizer_concurrent import set_task_supply_hosts
        with view.state.cv:
            for task in view.state.tasks.values():
                set_task_supply_hosts(view.state,task,{'server2'})
        self.assertTrue(view._standby_can_fill())
        view._fill_standby()
        self.assertGreater(len(ex.store.rows()), len(rows))
        delivered = set()
        for host in ('local', 'server2', 'local0-wsl', 'pbs-0'):
            poll = dict(host_id=host, slots=1, capacity=1, source_sha=self.source)
            for wave in range(2):
                assignment, = ex.rpc('/v1/poll', poll)['assignments']
                self.assertNotIn(assignment['job_id'], delivered)
                delivered.add(assignment['job_id'])
                ex.rpc('/v1/complete', dict(self.completion(assignment), host_id=host))
        self.assertEqual(len(delivered), 8)

    def test_pbs_reserves_one_buffered_q3_while_cpu_is_full_and_keeps_attempt(self):
        from trottertracks.rpe_pbs import GIB
        executor = self.executor(capacity=2, pbs_ready_buffer=True)
        executor.distributed['pbs'] = dict(workers=[dict(host_id='pbs-0', profile=dict(dp_workers=2))],
            task_seconds=dict(q3=1), allocation_sizing=dict(q3_startup_memory_bytes_per_task=GIB))
        view = executor.for_method('PR_PAULI')
        tasks = [view._enqueue(self.proposal(f'buffer-{i}'), demand=i==0) for i in range(4)]
        poll = dict(host_id='pbs-0', source_sha=self.source, slots=4, capacity=8,
            resource_report=dict(pbs=dict(remaining_seconds=120, accepting=True, launch_fit=True,ready_buffer=True,
                resource_pool=dict(cpu_count=8, memory_bytes=12*GIB))))
        with patch('trottertracks.rpe_pbs.resource_need_with_q3_memory', return_value=(4, GIB, False)):
            first, second = executor.rpc('/v1/poll', poll)['assignments']
        self.assertFalse(first['pbs_launch_resources']['standby'])
        self.assertFalse(second['pbs_launch_resources']['standby'])
        self.assertNotIn('q3_memory_measured', first['pbs_launch_resources'])
        for owner in (first, second):
            executor.rpc('/v1/state', dict(host_id='pbs-0', job_id=owner['job_id'], attempt_id=owner['attempt_id'], state='RUNNING'))
        with patch('trottertracks.rpe_pbs.resource_need_with_q3_memory', return_value=(2, GIB, True)):
            buffered = executor.rpc('/v1/poll', dict(poll,slots=0))['assignments']
        standby, = buffered
        self.assertTrue(standby['pbs_launch_resources']['standby'])
        self.assertIs(standby['pbs_launch_resources']['q3_memory_measured'], True)
        row = executor.store.row(standby['job_id'])
        self.assertEqual(row['attempt_id'], standby['attempt_id'])
        self.assertEqual(row['detail']['pbs_launch_resources'], standby['pbs_launch_resources'])
        self.assertEqual(executor.store.row(tasks[3].job.name)['state'], 'QUEUED')
        self.assertIsNone(row['started_epoch'])
        self.assertFalse(executor.store.requeue_pbs_buffer(row['job_id'], row['attempt_id']))
        with patch('trottertracks.rpe_pbs.resource_need_with_q3_memory', return_value=(4, GIB, False)):
            replay, = executor.rpc('/v1/poll', dict(poll,slots=0))['assignments']
        self.assertEqual(standby, replay)
        identity = {key: row[key] for key in ('job_id', 'attempt_id', 'request_sha256')}
        detail = dict(reason='PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH', scientific_execution_started=False,
                      pbs_buffer_reserved=True, automatic_retry=False)
        executor.store.update('pbs-0', row['job_id'], row['attempt_id'], 'UNKNOWN', detail)
        self.assertFalse(executor.store.requeue_pbs_buffer(row['job_id'], row['attempt_id']))
        detail.update(resource_released=True, pbs_admission_exit=dict(token='ended', **identity))
        executor.store.update('pbs-0', row['job_id'], row['attempt_id'], 'UNKNOWN', detail)
        self.assertTrue(executor.store.requeue_pbs_buffer(row['job_id'], row['attempt_id']))
        self.assertEqual(executor.store.row(row['job_id'])['state'], 'QUEUED')

    def test_pbs_bulk_offer_delivers_multiple_waves_without_starting_them(self):
        from trottertracks.rpe_pbs import GIB
        executor = self.executor(capacity=2, pbs_ready_buffer=True, pbs_standby_waves=2)
        executor.distributed['pbs'] = dict(workers=[dict(host_id='pbs-0', profile=dict(dp_workers=2))],
            task_seconds=dict(q3=1), allocation_sizing=dict(q3_startup_memory_bytes_per_task=GIB))
        view = executor.for_method('PR_PAULI')
        tasks = [view._enqueue(self.proposal(f'bulk-{i}'), demand=i<2) for i in range(8)]
        large = {t.job.name for t in tasks[:2]}
        poll = dict(host_id='pbs-0', source_sha=self.source, slots=2, capacity=8,
            resource_report=dict(pbs=dict(remaining_seconds=120, accepting=True, launch_fit=True, ready_buffer=True,
                resource_pool=dict(cpu_count=8, memory_bytes=12*GIB))))
        with patch('trottertracks.rpe_pbs.resource_need_with_q3_memory',
                   side_effect=lambda request, *_: (4 if request['request_id'] in large else 2, GIB, False)):
            assignments = executor.rpc('/v1/poll', poll)['assignments']
        standby = [item for item in assignments if item['pbs_launch_resources']['standby']]
        self.assertEqual(len(assignments), 8)
        self.assertEqual(len(standby), 6)
        self.assertTrue(all(executor.store.row(item['job_id'])['started_epoch'] is None for item in standby))

    def test_pbs_buffered_work_leaves_room_for_a_late_priority_request(self):
        from trottertracks.rpe_pbs import GIB
        executor = self.executor(capacity=2, pbs_ready_buffer=True)
        executor.distributed['pbs'] = dict(workers=[dict(host_id='pbs-0', profile=dict(dp_workers=2))],
            task_seconds=dict(q3=1), allocation_sizing=dict(q3_startup_memory_bytes_per_task=GIB))
        view = executor.for_method('PR_PAULI')
        for i in range(2):
            self.enqueue(executor, f'active-{i}')
        standby = view._enqueue(self.proposal('waiting-standby'), demand=False)
        poll = dict(host_id='pbs-0', source_sha=self.source, slots=2, capacity=8,
            resource_report=dict(pbs=dict(remaining_seconds=120, accepting=True, launch_fit=True, ready_buffer=True,
                resource_pool=dict(cpu_count=8, memory_bytes=12*GIB))))
        with patch('trottertracks.rpe_pbs.resource_need_with_q3_memory',
                   side_effect=lambda request, *_: (2 if request['request_id'] == standby.job.name else 4, GIB, False)):
            first, second, prepositioned = executor.rpc('/v1/poll', poll)['assignments']
            self.assertTrue(prepositioned['pbs_launch_resources']['standby'])
            for owner in (first, second):
                executor.rpc('/v1/state', dict(host_id='pbs-0', job_id=owner['job_id'],
                    attempt_id=owner['attempt_id'], state='RUNNING'))
            executor.rpc('/v1/complete', dict(self.completion(first), host_id='pbs-0'))
            priority = self.enqueue(executor, 'late-priority')
            manager = SimpleNamespace(configured={'pbs-0': {}}, settings={}, startup_jobs=lambda **_: set(),
                                      accepting_allocation=lambda *_: True, replacement_fenced=lambda *_: False)
            with patch.object(executor, 'pbs_manager', manager), \
                 patch.object(executor, '_pending_preparation_challenges', return_value=[]) as check:
                answer = executor.rpc('/v1/poll', dict(poll, slots=1, durable_assignments=[[
                    prepositioned['job_id'], prepositioned['attempt_id'], prepositioned['request_sha256']]]))
                check.assert_called_once_with('pbs-0', 2)
            fresh, = answer['assignments']
            self.assertEqual(fresh['job_id'], priority.job.name)
            self.assertFalse(fresh['pbs_launch_resources']['standby'])
            self.assertEqual(executor.store.row(standby.job.name)['attempt_id'], prepositioned['attempt_id'])

    def test_pbs_adopts_core_before_deferred_evidence_and_recovers_archive_after_restart(self):
        from trottertracks.rpe_pbs import PbsManager
        from trottertracks._pipeline_pbs_q3_worker import result_bundle

        executor = self.executor(capacity=1, pbs_gateway_completion_workers=2,
            pbs_dispatch_gateway=dict(path='/fixed/gateway', sha256='fixed', split_exchange=True,
                selective_harvest=True, bounded_results=True, defer_q3_evidence=True))
        settings = dict(ssh_alias='artificial', remote_root='/private/test', workers=[dict(host_id='pbs-0')])
        self.enqueue(executor, 'archive')
        poll = dict(host_id='pbs-0', slots=1, source_sha=self.source)
        owner, = executor.rpc('/v1/poll', poll)['assignments']
        attempt = Path(self.tmp.name)/'worker'
        attempt.mkdir()
        full = self.completion(owner)
        for name, data in full['files'].items():
            (attempt/name).write_bytes(base64.b64decode(data))
        extra = attempt/'application'/'RESULT.json'
        extra.parent.mkdir()
        extra.write_bytes(b'artificial evidence' * 90_000)
        deferred = {}
        files = result_bundle(attempt, deferred=deferred)
        self.assertEqual(files, full['files'])
        self.assertEqual(deferred, {'application/RESULT.json': extra.stat().st_size})
        gateway = SimpleNamespace(close=lambda: None)
        manager = PbsManager(executor, settings, gateway=gateway)
        try:
            manager.state['allocations']['token'] = dict(status='F', spec=dict(profile=dict(host_id='pbs-0')))
            payload = dict(full, host_id='pbs-0', pbs_evidence=deferred)
            packet = dict(request_id='complete-core', endpoint='/v1/complete', payload=payload)
            response = manager._dispatch_ready('token', packet)
            marker = manager.evidence_directory/owner['job_id']/(owner['attempt_id']+'.json')
            self.assertEqual(executor.store.row(owner['job_id'])['state'], 'ACCEPTED')
            self.assertFalse(response['result']['duplicate'])
            self.assertTrue(response['result']['evidence_pending'])
            central = self.run/'jobs'/'PR_PAULI'/owner['job_id']
            self.assertTrue((central/'OUTCOME.json').exists())
            self.assertFalse((central/'application/RESULT.json').exists())
            next_task = self.enqueue(executor, 'next-while-archiving')
            self.assertEqual(executor.rpc('/v1/poll', poll)['assignments'][0]['job_id'], next_task.job.name)
            manager.save()
        finally:
            manager.close(retain_allocation=True)
        manager = PbsManager(executor, settings, gateway=gateway)
        try:
            self.assertTrue(marker.exists())
            status = dict(request_id='complete-status', endpoint='/v1/complete',
                          payload={key: payload[key] for key in ('job_id','attempt_id','request_sha256','host_id')})
            status['payload']['pbs_evidence_status'] = True
            self.assertTrue(manager._dispatch_ready('token', status)['result']['evidence_pending'])
            with patch.object(manager, '_read_bulk', return_value=extra.read_bytes()) as read:
                manager._advance_deferred_evidence()
                manager.evidence_pending[1].result(timeout=3)
                self.assertEqual(read.call_args.kwargs['attempt_id'], owner['attempt_id'])
            final = manager._dispatch_ready('token', status)['result']
            self.assertTrue(final['duplicate'])
            self.assertFalse(final['evidence_pending'])
            self.assertEqual((central/'application/RESULT.json').read_bytes(), extra.read_bytes())
            self.assertFalse(marker.exists())
            with executor.store._connection() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM events WHERE job_id=? AND event='ACCEPT'",
                                           (owner['job_id'],)).fetchone()[0], 1)
        finally:
            manager.close(retain_allocation=True)

    def test_proved_unstarted_pbs_ordinary_assignment_returns_to_same_queue(self):
        executor = self.executor()
        task = self.enqueue(executor, 'before-science')
        assignment, = executor.rpc('/v1/poll', dict(host_id='pbs-0', source_sha=self.source,
                                                   slots=1, capacity=1))['assignments']
        owner = {field: assignment[field] for field in ('job_id', 'attempt_id', 'request_sha256')}
        self.assertEqual(executor.store.counts('PR_PAULI', observed_only=True), 0)
        decline = dict(reason='PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH', automatic_retry=False,
                       scientific_execution_started=False, resource_released=True,
                       pbs_admission_exit=dict(owner, token='finished-allocation'))
        answer = executor.rpc('/v1/state', dict(owner, host_id='pbs-0', state='UNKNOWN', detail=decline))
        self.assertEqual(answer['disposition'], 'REQUEUED_PROVED_UNSTARTED')
        self.assertEqual(executor.store.row(task.job.name)['state'], 'QUEUED')
        self.assertEqual(task.status, 'QUEUED')
        from trottertracks.rpe_optimizer_concurrent import queued_count,tasks_with_status
        self.assertIn(task,tasks_with_status(executor.state,'QUEUED'))
        self.assertEqual(queued_count(executor.state,'q3'),1)
        self.assertTrue((task.job/'attempt-history'/assignment['attempt_id']/'LAUNCH.json').is_file())
        self.assertEqual(executor.store.counts('PR_PAULI'), 1)
        self.assertEqual(executor.store.counts('PR_PAULI', observed_only=True), 0)

    def test_pbs_refills_after_q3_exit_before_transfer_ack(self):
        from trottertracks.rpe_pbs import GIB
        executor = self.executor(capacity=3)
        executor.distributed['pbs'] = dict(workers=[dict(host_id='pbs-0', profile=dict(dp_workers=2))],
            task_seconds=dict(q3=1), allocation_sizing=dict(q3_startup_memory_bytes_per_task=2*GIB))
        first = self.enqueue(executor, 'awaiting-transfer')
        second = self.enqueue(executor, 'replacement')
        poll = dict(host_id='pbs-0', source_sha=self.source, slots=3, capacity=3,
            resource_report=dict(pbs=dict(remaining_seconds=120, accepting=True,
                resource_pool=dict(cpu_count=2, memory_bytes=3*GIB))))
        owned, = executor.rpc('/v1/poll', poll)['assignments']
        self.assertEqual(owned['job_id'], first.job.name)
        identity = dict(host_id='pbs-0', job_id=owned['job_id'], attempt_id=owned['attempt_id'])
        executor.rpc('/v1/state', dict(identity, state='RUNNING'))
        self.assertFalse(executor.rpc('/v1/poll', poll)['assignments'])
        executor.rpc('/v1/state', dict(identity, state='UNKNOWN'))
        self.assertFalse(executor.rpc('/v1/poll', poll)['assignments'])
        executor.rpc('/v1/state', dict(identity, state='COMPLETED_PENDING_TRANSFER'))
        replacement, = executor.rpc('/v1/poll', poll)['assignments']
        self.assertEqual(replacement['job_id'], second.job.name)
        self.assertEqual(executor.store.row(first.job.name)['attempt_id'], owned['attempt_id'])
        self.assertEqual(executor.store.row(first.job.name)['state'], 'COMPLETED_PENDING_TRANSFER')
        completion = dict(self.completion(owned), host_id='pbs-0')
        self.assertFalse(executor.rpc('/v1/complete', completion)['duplicate'])
        self.assertTrue(executor.rpc('/v1/complete', completion)['duplicate'])

    def test_readiness_window_uses_fresh_cpu_deficit_and_check_time(self):
        executor = self.executor(capacity='auto', live_q3_readiness=True,
                                 live_readiness_max_workers=8)
        executor.state.ready_cpu_supply = {'pbs-0': {'missing_cpu': 20, 'q3_cpus_per_job': 2}}
        with patch.object(executor, '_offer_fresh', return_value=True):
            executor.state.q3_readiness_seconds = 1
            self.assertEqual(executor._live_readiness_window(), 2)
            executor.state.q3_readiness_seconds = 4
            self.assertEqual(executor._live_readiness_window(), 8)
            executor.state.ready_cpu_supply['pbs-0']['missing_cpu'] = 0
            self.assertEqual(executor._live_readiness_window(), executor.live_readiness_workers)

    def test_parallel_readiness_reuses_fast_host_lane_while_other_host_times_out(self):
        ex = self.executor(capacity=4, controller_parallel=True, live_readiness_max_workers=1,
            live_q3_readiness=True, host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
        view = ex.for_method('PR_PAULI')
        gate, slow_started = threading.Event(), threading.Event()
        observed = []

        def probe(view, request, job, decision, *, publish, dispatch, targets):
            def check(item):
                host, _ = item
                observed.append((job, host))
                if host == 'slow':
                    slow_started.set()
                    gate.wait(5)
                return host, dict(facts=host)

            def receive(host, result):
                decision['eligible_host_ids'].append(host)
                decision.setdefault('host_fingerprints', {})[host] = Q.digest(result)
                decision.setdefault('host_decisions', {})[host] = dict(status='PASS')
                publish(deepcopy(decision))

            def finish(deferred):
                return dict(decision, deferred_hosts=sorted(deferred))

            dispatch([(host, None) for host in ('server1', 'slow') if targets is None or host in targets],
                     check, receive, finish)

        def wait_for_pass(proposal):
            job = view.job_dir(proposal)
            with ex.state.cv:
                self.assertTrue(ex.state.cv.wait_for(lambda: any(
                    key[0] == job.name and 'server1' in decision['eligible_host_ids']
                    for key, decision in ex.live_readiness_partial.items()) or any(
                    key[0] == job.name and 'server1' in cached[2]['eligible_host_ids']
                    for key, cached in ex.eligibility_policy_cache['live_checks'].items()), timeout=3))
            return view._enqueue(proposal, defer_readiness=True)

        with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=probe):
            try:
                first = self.proposal('first-with-slow-host')
                view._enqueue(first, defer_readiness=True)
                self.assertTrue(slow_started.wait(3))
                self.assertIsNotNone(wait_for_pass(first))
                following = self.proposal('follows-on-fast-host')
                view._enqueue(following, defer_readiness=True)
                task = wait_for_pass(following)
                self.assertIsNotNone(ex.store.row(task.job.name))
                self.assertEqual(ex.live_readiness_host_counts['slow'], 1)
                self.assertNotIn((task.job.name, 'slow'), observed)
                assignment, = ex.rpc('/v1/poll', dict(host_id='server1', slots=1, source_sha=self.source,
                    host_fingerprints={task.job.name: {'facts': 'server1'}}))['assignments']
                self.assertEqual(assignment['job_id'], task.job.name)
            finally:
                gate.set()
                with ex.state.cv:
                    self.assertTrue(ex.state.cv.wait_for(lambda: not ex.live_readiness_inflight, timeout=3))

    def test_standby_dry_checks_roll_while_slow_request_is_pending(self):
        from concurrent.futures import ThreadPoolExecutor
        ex = self.executor(capacity=4, controller_queue_first=True, work_conserving_search=True,
            live_q3_readiness=True, host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={}))
        ex.live_readiness_pool.shutdown()
        ex.live_readiness_pool = ThreadPoolExecutor(2)
        ex.live_readiness_workers = 2
        view = ex.for_method('PR_PAULI')
        ex.state.budgets['PR_PAULI'] = 4
        gate = threading.Event()
        observed = []
        def check(executor, request, job, decision, *, publish):
            name = request['configuration_id']
            observed.append(name)
            if name == 'slow':
                gate.wait(5)
            return dict(decision, eligible_host_ids=['server1'], host_fingerprints={'server1': 'fixture'})
        proposals = [self.proposal(i) for i in ('demand', 'slow', 'fast', 'next', 'extra')]
        try:
            with patch('trottertracks.rpe_live_q3_readiness.probe', side_effect=check):
                view.prefetch_standby(proposals[0], proposals[1:])
                import time
                end = time.monotonic() + 3
                while time.monotonic() < end and ex.store.counts('PR_PAULI') < 3:
                    with ex.state.cv:
                        ex.state.cv.wait_for(view._standby_can_fill, timeout=.1)
                    view._fill_standby()
                self.assertIsNone(ex.store.row(view.job_dir(proposals[1]).name))
                self.assertIsNotNone(ex.store.row(view.job_dir(proposals[3]).name))
                gate.set()
                end = time.monotonic() + 3
                while time.monotonic() < end and ex.store.counts('PR_PAULI') < 4:
                    with ex.state.cv:
                        ex.state.cv.wait_for(view._standby_can_fill, timeout=.1)
                    view._fill_standby()
                self.assertEqual(ex.store.counts('PR_PAULI'), 4)
                self.assertEqual(len(observed), len(set(observed)))
        finally:
            gate.set()

    def test_work_conserving_records_distinct_idle_slot_reasons(self):
        from trottertracks.rpe_preparation_queue import normalize_preparation_request
        from trottertracks.rpe_q3_timing import read_lines
        executor = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent='auto', budgets={'PR_PAULI': 1},
            distributed=dict(port=0, live_ready_refill=True, work_conserving_search=True,
                             host_stage_capabilities={'waiting-host': ['q3'], 'prep-host': ['signal']}))
        self.executors.append(executor)

        def poll(host, observe_only=False):
            # Treat each synthetic change as a separate telemetry sample.
            executor.state.next_queue_timing = 0
            reply = executor.rpc('/v1/poll', dict(host_id=host, source_sha=self.source, slots=1, capacity=1,
                                                observe_only=observe_only))
            with executor.state.cv:
                self.assertTrue(executor.state.cv.wait_for(
                    lambda: not executor.state.queue_observation_active, timeout=2))
            return reply

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

    def test_scaled_completion_arriving_before_registration_is_reoffered_once(self):
        executor = self.executor()
        view = executor.for_method('PR_PAULI')
        proposal = dict(self.proposal('early'), origin=dict(kind=S.RULE, M=1))
        task = view._enqueue(proposal, demand=True)
        assignment, = self.poll(executor)['assignments']
        receipt = self.completion(assignment)
        executor.rpc('/v1/complete', receipt)

        self.assertEqual(list(view.read_scaled_completions(lambda key: None)), [])
        self.assertFalse(view.scaled_completion_pending())
        view.refresh_scaled_completions({'other'})
        self.assertFalse(view.scaled_completion_pending())
        view.refresh_scaled_completions({'early'})
        accepted, = view.read_scaled_completions(lambda key: proposal)
        self.assertEqual(accepted[0][0], 'early')
        self.assertIsNotNone(executor.store.row(task.job.name)['adopted_epoch'])
        executor.rpc('/v1/complete', receipt)
        view.refresh_scaled_completions({'early'})
        self.assertEqual(list(view.read_scaled_completions(lambda key: proposal)), [])

    def test_scaled_adoption_with_two_hosts_and_resume(self):
        from run_rpe_method_optimizer import make_executor

        method = 'PR_PAULI'
        tau = Fraction(S.EPSILON)**2
        for scenario in ('late_ready', 'live_standby', 'drained_standby', 'same_series'):
            with self.subTest(scenario=scenario):
                same_series = scenario == 'same_series'
                problem, context, routes = fixture([method])
                problem['rpe']['M'] = 1
                problem['search'].update(rule=S.RULE, work_conserving_search=True, lookahead=2,
                                         work_slice_steps=1 if scenario == 'drained_standby' else None,
                                         scaled_shots=dict(M=[1], grid=dict(min_k=-1 if same_series else 0,
                                                                            max_k=0, divisor=1),
                                                           compare_paper=False))
                context['tau'] = tau
                context['saved'] = O.SavedBackend([], {})
                first, second = context['candidates']
                first.update(prices=[Fraction(1)]*4)
                second.update(prices=[Fraction(2)]*4)
                winner = first if same_series else second
                ready = threading.Event()
                draining = threading.Event()

                def subscribe(*_):
                    def take():
                        if not ready.is_set():
                            return [], None
                        ready.clear()
                        return [second], None
                    return ready.is_set, take

                source = None
                if scenario == 'late_ready':
                    context['candidates'] = [first]
                    source = SimpleNamespace(subscribe=subscribe, has_new=lambda _: ready.is_set())
                if same_series:
                    context['candidates'] = [first]
                    problem['search']['max_steps_per_method'] = None
                run = self.run/scenario
                spec = dict(methods=[method], science_approved=True,
                            max_new_q3_per_method={method: 'UNLIMITED' if same_series else 2},
                            executor=dict(scheduler='concurrent', max_concurrent_jobs=2,
                                          prior_run_dirs=[str(run)],
                                          distributed=dict(port=0, work_conserving_search=True)))
                ex, optimize = make_executor(ROOT, run, routes, {}, problem, spec)
                self.executors.append(ex)
                answers, errors = [], []

                def finish(view):
                    if view.run_dir == run and view.method == method:
                        draining.set()
                    return original_finish(view)

                original_finish = DistributedExecutor.finish

                def search():
                    try:
                        with patch.object(DistributedExecutor, 'finish', finish):
                            answers.append(optimize(ROOT, problem, context=context, executor=ex, ready_source=source,
                                                    max_new=spec['max_new_q3_per_method']))
                    except Exception as exc:
                        errors.append(exc)

                def claim(host):
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        assignments = ex.rpc('/v1/poll', dict(host_id=host, slots=1, capacity=1,
                                                             source_sha=self.source))['assignments']
                        if assignments:
                            assigned, = assignments
                            return assigned
                        time.sleep(.01)
                    self.fail(f'{host} did not claim an eligible artificial job: {errors}')

                def complete(task, host, passes):
                    payload = self.completion(task)
                    out = json.loads(base64.b64decode(payload['files']['OUTCOME.json']))
                    bound = O.enc(tau/2 if passes else tau*2)
                    out.update(status=O.PASS if passes else O.FAIL, MSE_lower=bound, MSE_upper=bound)
                    payload.update(host_id=host)
                    payload['files']['OUTCOME.json'] = base64.b64encode(json.dumps(out).encode()).decode()
                    self.assertFalse(ex.rpc('/v1/complete', payload)['duplicate'])
                    self.assertTrue(ex.rpc('/v1/complete', payload)['duplicate'])

                thread = threading.Thread(target=search, daemon=True)
                thread.start()
                try:
                    with ex.state.cv:
                        self.assertTrue(ex.state.cv.wait_for(lambda: bool(ex.store.rows()) or errors, timeout=5))
                    a = claim('server2')
                    self.assertEqual(a['request']['configuration_id'], first['configuration_id'])
                    if scenario == 'late_ready':
                        ex.rpc('/v1/poll', dict(host_id='pbs-0', slots=1, capacity=1,
                                               source_sha=self.source, observe_only=True))
                        with ex.state.cv:
                            ready.set()
                            ex.state.cv.notify_all()
                    with ex.state.cv:
                        self.assertTrue(ex.state.cv.wait_for(lambda: len(ex.store.rows()) == 2 or errors, timeout=5))
                    b = claim('pbs-0')
                    self.assertEqual(b['request']['configuration_id'], winner['configuration_id'])
                    if same_series:
                        self.assertEqual(a['request']['N_X'], [8, 6])
                        self.assertEqual(b['request']['N_X'], [15, 11])
                    if scenario == 'drained_standby':
                        complete(a, 'server2', False)
                        self.assertTrue(draining.wait(timeout=5))
                        complete(b, 'pbs-0', True)
                    else:
                        complete(b, 'pbs-0', True)
                        deadline = time.monotonic() + 5
                        while ex.store.row(b['job_id'])['adopted_epoch'] is None and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertIsNotNone(ex.store.row(b['job_id'])['adopted_epoch'])
                        self.assertEqual(ex.store.row(a['job_id'])['state'], 'ASSIGNED')
                        complete(a, 'server2', False)
                    thread.join(timeout=8)
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(errors, errors)
                    result = answers[0][method]
                    self.assertEqual(result['best']['configuration_id'], winner['configuration_id'])
                    if scenario == 'drained_standby':
                        self.assertIsNone(result['rule_adopted_best'])
                    self.assertEqual(result['held'], [])
                    self.assertEqual(result['evaluated_count'], 2)
                    if scenario != 'late_ready':
                        adopted = next(row for row in result['trace'] if row['job'] and row['job'].endswith(b['job_id']))
                        self.assertFalse(adopted['selected_by_rule'])
                finally:
                    self.close_one(ex)
                    thread.join(timeout=8)

                resumed, optimize = make_executor(ROOT, run, routes, {}, problem, spec)
                self.executors.append(resumed)
                context['candidates'] = [first] if same_series else [first, second]
                result = optimize(ROOT, problem, context=context, executor=resumed,
                                  max_new=spec['max_new_q3_per_method'])[method]
                self.assertEqual(result['best']['configuration_id'], winner['configuration_id'])
                self.assertEqual(result['evaluated_count'], 2)
                self.assertEqual(resumed.store.counts(method, started_only=True), 2)
                events = resumed.store.events()
                self.assertEqual([e['event'] for e in events].count('ACCEPT'), 2)
                self.assertEqual([e['event'] for e in events].count('ADOPT'), 2)

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
        accepted=resumed.store.row(task.job.name)
        resumed._publish_transfer(accepted,startup=True)
        marker=self.run/'distributed'/'publication_checks'/assigned['job_id']/(assigned['attempt_id']+'.json')
        self.assertTrue(marker.is_file())
        saved=json.loads(marker.read_text());stable=max(stamp[-1] for root in ('stage','job') for stamp in saved[root].values())+2_000_000_000
        with patch('trottertracks.rpe_optimizer_concurrent.time.time_ns',return_value=stable), \
             patch.object(Path,'rglob',side_effect=AssertionError('unchanged publication must not walk files')):
            resumed._publish_transfer(accepted,startup=True)
        (task.job/'raw/RESULT.json').unlink()
        resumed._publish_transfer(accepted,startup=True)
        self.assertTrue((task.job/'raw/RESULT.json').samefile(stage/'raw/RESULT.json'))
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
        held = executor.for_method('PR_PAULI').answer(self.proposal('first'))
        self.assertTrue(O.is_local_unknown(held))
        self.assertEqual(held['reason'], 'WORKER_RESULT_MISMATCH')
        other = self.enqueue(executor, 'independent')
        assignment, = executor.rpc('/v1/poll', dict(host_id='other', slots=1, source_sha=self.source))['assignments']
        self.assertEqual(assignment['job_id'], other.job.name)

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


def test_task_index_tracks_queue_claim_cancel_resume_and_removal():
    from trottertracks.rpe_optimizer_concurrent import TaskIndex,tasks_with_status,transition_task
    state=SimpleNamespace(tasks=TaskIndex())
    for number in range(1000):
        job=Path(str(number));state.tasks[job]=SimpleNamespace(job=job,status='DONE')
    first=SimpleNamespace(job=Path('first'),status='QUEUED')
    state.tasks[first.job]=first
    assert tasks_with_status(state,'QUEUED')==(first,)
    transition_task(state,first,'RUNNING')
    assert tasks_with_status(state,'QUEUED')==()
    assert tasks_with_status(state,'RUNNING')==(first,)
    transition_task(state,first,'DONE')
    fresh=SimpleNamespace(job=first.job,status='QUEUED')
    state.tasks[first.job]=fresh
    transition_task(state,fresh,'CANCELLED')
    assert not tasks_with_status(state,'QUEUED')
    assert state.tasks.pop(first.job) is fresh
    assert not tasks_with_status(state,'CANCELLED')
    assert len(tasks_with_status(state,'DONE'))==1000


def test_task_index_counts_change_only_on_owner_and_demand_transitions():
    from trottertracks.rpe_optimizer_concurrent import TaskIndex,queued_count,set_task_demand,transition_task
    state=SimpleNamespace(tasks=TaskIndex())
    first=SimpleNamespace(job=Path('one'),status='QUEUED',demand=False,
                          rule={'kind':'fixture'},view=SimpleNamespace(method='PR'))
    state.tasks[first.job]=first
    assert queued_count(state,'all')==queued_count(state,'q3')==1
    assert queued_count(state,'speculative','PR')==1
    set_task_demand(state,first,True)
    assert queued_count(state,'speculative','PR')==0
    assert queued_count(state,'demand','PR')==queued_count(state,'demand')==1
    transition_task(state,first,'RUNNING')
    assert queued_count(state,'all')==queued_count(state,'demand','PR')==0
    transition_task(state,first,'DONE')
    assert queued_count(state,'q3')==0


def test_task_index_supply_counts_are_updated_for_eligibility_and_ownership():
    from trottertracks.rpe_optimizer_concurrent import TaskIndex,queued_supply_counts,set_task_supply_hosts,transition_task
    state=SimpleNamespace(tasks=TaskIndex())
    task=SimpleNamespace(job=Path('task'),status='QUEUED',rule={'kind':'fixture'},supply_hosts=None)
    state.tasks[task.job]=task
    assert queued_supply_counts(state,('a','b'))=={'a':1,'b':1}
    set_task_supply_hosts(state,task,frozenset({'b'}))
    assert queued_supply_counts(state,('a','b'))=={'a':0,'b':1}
    transition_task(state,task,'RUNNING')
    assert queued_supply_counts(state,('a','b'))=={'a':0,'b':0}
    transition_task(state,task,'QUEUED')
    assert queued_supply_counts(state,('a','b'))=={'a':0,'b':1}
    del state.tasks[task.job]
    assert queued_supply_counts(state,('a','b'))=={'a':0,'b':0}
