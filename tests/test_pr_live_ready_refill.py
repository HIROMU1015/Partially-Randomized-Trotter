"""Normal width entry can dispatch a prepared candidate while Q3 is waiting."""
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import tempfile
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

import test_pr_width_overlap as overlap
import check_rpe_distributed_q3 as distributed
from trottertracks.rpe_preparation_queue import normalize_preparation_request
from trottertracks.rpe_q3_store import JobStore
from trottertracks.rpe_q3_timing import summarize


class LiveReadyRefillTests(unittest.TestCase):
    def run_fixture(self, directory, enabled, idle_method=None):
        directory = Path(directory)
        args, _, _ = overlap.setup(directory, True)
        spec = overlap.E.O.load_json(directory / 'ROUND.json')
        spec['live_ready_refill'] = enabled
        spec['search_overrides'].update(lookahead=4, lookahead_mode='HYBRID')
        spec['executor']['max_concurrent_jobs'] = 4
        overlap.F.write(directory / 'ROUND.json', spec)
        base, _ = overlap.classes(True)
        both_preparing = threading.Barrier(2)
        prepared = set()
        first_selector_done = threading.Event()

        class Completion(overlap.F.ArtificialCompletion):
            def __init__(self, root):
                super().__init__(root)
                self.lock = threading.RLock()

            def complete(self, request):
                design = request['design']
                if overlap.F.KS._frac(design['delta_pf']) != Fraction(.1):
                    # Observe genuine worker launch before making new routes ready.
                    deadline = time.monotonic() + 10
                    while len(list(self.root.glob('jobs/*/*/LAUNCH.json'))) < 2:
                        if time.monotonic() > deadline:
                            raise AssertionError('Initial Q3 jobs never started')
                        time.sleep(.01)
                    if enabled:
                        both_preparing.wait(10)
                    if request['method'] == idle_method:
                        if not first_selector_done.wait(5):
                            raise AssertionError('The first selector did not become idle')
                    with self.lock:
                        prepared.add(request['method'])
                        self.state['price_received'] = True
                        return super().complete(request)
                with self.lock:
                    return super().complete(request)

            def install(self, context, problem, records):
                routes = super().install(context, problem, records)
                for rec in records:
                    design = rec['request']['design']
                    if overlap.F.KS._frac(design['delta_pf']) == Fraction(.1):
                        for rule in routes[rec['request']['method']]:
                            if rec['configuration_id'] in rule['configuration_ids']:
                                method = rec['request']['method']
                                if enabled and method != idle_method:
                                    rule['artificial']['release_on_second_launch'] = list(dict.fromkeys(
                                        [method, idle_method] if idle_method else [method]))
                return routes

        class Driver(base):
            def completion_api(self):
                if self.completion is None:
                    self.completion = Completion(self.run)
                return self.completion

        original = overlap.E.O.optimize

        def observe_idle(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('ready_source') is not None and kwargs.get('methods') == [idle_method]:
                first_selector_done.set()
            return result

        run_adapter = overlap.E.O.Executor.run_adapter

        def gated_adapter(view, job, rule, mode):
            methods = rule['artificial'].get('release_on_second_launch')
            if not methods or mode != 'execute':
                return run_adapter(view, job, rule, mode)
            # The artificial child waits for another launch before calling the
            # existing adapter. This exercises ordering without a sleep race.
            gate = """from pathlib import Path
import os, sys, time
root = Path(sys.argv[1]); methods = sys.argv[2].split(','); deadline = time.monotonic() + 12
while any(len(list((root / m).glob('*/LAUNCH.json'))) < 2 for m in methods):
    if time.monotonic() > deadline:
        sys.exit('No second launch while the first worker held its slot')
    time.sleep(.01)
os.execv(sys.executable, [sys.executable, '-B', *sys.argv[3:]])
"""
            cmd = [rule['interpreter'], '-B', '-c', gate, str(view.run_dir / 'jobs'), ','.join(methods),
                   str(view.adapter), '--request', str(job / 'REQUEST.json'), '--output', str(job), '--mode', mode]
            with (job / 'EXECUTE.log').open('a') as log:
                return subprocess.Popen(cmd, cwd=view.root, stdout=log, stderr=subprocess.STDOUT)

        with (patch.object(overlap.E.O, 'optimize', side_effect=observe_idle),
              patch.object(overlap.E.O.Executor, 'run_adapter', gated_adapter)):
            result = overlap.call(args, Driver)
        self.assertEqual(prepared, {'PR_DF', 'PR_PAULI'})
        self.assertFalse((result.run / 'RUNTIME_ERRORS.jsonl').exists())
        events = overlap.records(result.run / 'PROGRESS.jsonl')
        exits = [r for r in events if r['event'] == 'EXIT']
        self.assertEqual(len(exits), 4)
        self.assertTrue(all(r['returncode'] == 0 for r in exits))
        self.assertLessEqual(max(r['active'] for r in events if r['event'] == 'LAUNCH'), 4)
        both = []
        for method in ('PR_DF', 'PR_PAULI'):
            launched = [r for r in events if r['event'] == 'LAUNCH' and r['method'] == method]
            self.assertEqual(len(launched), 2)
            first, second = launched
            exit_first = next(r for r in events if r['event'] == 'EXIT' and r['job'] == first['job'])
            both.append(event_duration(second, first, exit_first))
        markers = {p: p.read_bytes() for p in result.run.glob('jobs/*/*/LAUNCH.json')}
        before_resume = overlap.E.O.load_json(result.run / 'CONFIRMED_MINIMA.json')['methods']
        overlap.call(args, Driver)
        self.assertEqual(markers, {p: p.read_bytes() for p in result.run.glob('jobs/*/*/LAUNCH.json')})
        after_resume = overlap.E.O.load_json(result.run / 'CONFIRMED_MINIMA.json')['methods']
        return both, events, [(m, before_resume[m]['verified_all_min'], after_resume[m]['verified_all_min'])
                              for m in sorted(before_resume)]

    def test_late_ready_refills_slots_before_inflight_answers_and_resumes(self):
        with tempfile.TemporaryDirectory(prefix='pr-live-ready-') as tmp:
            baseline, base_events, base_totals = self.run_fixture(Path(tmp) / 'old', False)
            live, live_events, live_totals = self.run_fixture(Path(tmp) / 'live', True)
        self.assertTrue(all(delta >= 0 for delta in baseline), baseline)
        self.assertTrue(all(delta < 0 for delta in live), live)
        self.assertFalse(any(r['event'] == 'Q3_WAIT_YIELD_FOR_READY' for r in base_events))
        self.assertEqual({r['method'] for r in live_events if r['event'] == 'Q3_WAIT_YIELD_FOR_READY'},
                         {'PR_DF', 'PR_PAULI'})
        self.assertEqual(live_totals, base_totals)
        self.assertTrue(all(first == resumed and first is not None for _, first, resumed in live_totals))

    def test_idle_method_refills_slots_while_peer_is_still_running(self):
        with tempfile.TemporaryDirectory(prefix='pr-idle-ready-') as tmp:
            _, events, _ = self.run_fixture(tmp, True, idle_method='PR_DF')
        df_first, df_next = [r for r in events if r['event'] == 'LAUNCH' and r['method'] == 'PR_DF']
        pauli_first = next(r for r in events if r['event'] == 'LAUNCH' and r['method'] == 'PR_PAULI')
        ended = {r['job']: r['monotonic'] for r in events if r['event'] == 'EXIT'}
        self.assertGreater(df_next['monotonic'], ended[df_first['job']])
        self.assertLess(df_next['monotonic'], ended[pauli_first['job']])

    def test_normal_width_entry_refills_auto_queue_on_two_worker_spools(self):
        with tempfile.TemporaryDirectory(prefix='pr-distributed-refill-') as temporary:
            evidence = self.distributed_fixture(Path(temporary))
        if output := os.environ.get('RPE_LIVE_READY_TEST_EVIDENCE'):
            distributed.write(output, evidence)

    def test_inflight_attempt_is_recovered_after_controller_restart(self):
        with tempfile.TemporaryDirectory(prefix='pr-live-refill-restart-') as temporary:
            result = distributed.fault_acceptance(Path(temporary), work_conserving_search=True)
            self.assertTrue(result['same_attempt_recovered'])
            self.assertEqual(result['recovered_job_execute_count'], 1)
            self.assertEqual(result['completed_resume_new_launches'], 0)

    def distributed_fixture(self, directory):
        args, base, _ = overlap.setup(directory, True)
        port = distributed.free_port()
        local = distributed.profile(directory, 'fixture-primary')
        peer = distributed.profile(directory, 'fixture-peer')
        peer_path = directory/'PROFILE_peer.json'
        distributed.write(peer_path, peer)
        spec = overlap.E.O.load_json(directory/'ROUND.json')
        spec.update(live_ready_refill=True)
        spec['search_overrides']['lookahead'] = 2
        spec['search_overrides']['work_conserving_search'] = True
        spec['executor'].update(max_concurrent_jobs='auto', distributed=dict(
            local_worker=local, port=port, affinity_wait_seconds=0,
            preparation={'budgets': {'signal': 1, 'compile': 1}},
            host_admission={'hosts': {'fixture-absent': {'accepting': True, 'optional': True}}},
            required_hosts=['fixture-absent'], host_stage_capabilities={'fixture-primary': ['q3']}))
        distributed.write(directory/'ROUND.json', spec)
        late = {}

        class Completion(overlap.F.ArtificialCompletion):
            def __init__(self, directory):
                super().__init__(directory)
                self.lock = threading.RLock()

            def bind_stage_executor(self, runtime):
                self.executor = runtime.executor

            def complete(self, request):
                method, design = request['method'], request['design']
                if overlap.F.KS._frac(design['delta_pf']) == Fraction(.1):
                    with self.lock:
                        return super().complete(request)
                if method == 'PR_PAULI':
                    return dict(status='WAITING_INPUTS', reason='ARTIFICIAL_MISSING_PRICE')
                cid = overlap.F.configuration(method, design['delta_pf'], design['kappa'], design['L_D'])['configuration_id']
                jobs = []
                for stage in ('signal', 'compile'):
                    task = normalize_preparation_request(dict(route={'kind': 'preparation'}, method='ARTIFICIAL',
                        preparation=dict(stage=stage, provider='ARTIFICIAL_'+stage, artificial=True,
                            source=self.executor.source_manifest, identity=cid, outputs=['products'], command=[
                                '@PYTHON@', '-B', '@SOURCE@/scripts/artificial_preparation_leaf.py',
                                '--stage', stage, '--output', '@WORK@/products'])))
                    jobs.append(self.executor.submit_preparation(task, budget=1, approved=True,
                                                                contract={'ARTIFICIAL': True}))
                for job in jobs:
                    result = self.executor.preparation_result(job, wait=True)
                    if result['status'] != 'PREPARATION_COMPLETE':
                        raise AssertionError(result)
                with self.lock:
                    self.state['price_received'] = True
                    result = super().complete(request)
                    late.update(configuration_id=cid, preparation=jobs)
                    return result

            def install(self, context, problem, records):
                routes = super().install(context, problem, records)
                for method, rules in routes.items():
                    for rule in rules:
                        first = next((record for record in records if record.get('configuration_id') in rule['configuration_ids']), None)
                        if first is not None:
                            original = overlap.F.KS._frac(first['request']['design']['delta_pf']) == Fraction(.1)
                            rule['artificial']['sleep_seconds'] = 6 if method == 'PR_PAULI' and original else .5
                return routes

        class Driver(base):
            def completion_api(self):
                if self.completion is None:
                    self.completion = Completion(self.run)
                return self.completion

        run = directory/'run'
        store_path = run/'distributed'/'QUEUE.sqlite3'
        proxy = distributed.FaultProxy('http://127.0.0.1:'+str(port))
        worker = None
        controller = ThreadPoolExecutor(max_workers=1)
        future = controller.submit(overlap.call, args, Driver)
        try:
            def current_attempt(profile, method, configuration=None):
                for path in Path(profile['work_dir']).glob('*/*/ASSIGNMENT.json'):
                    row = overlap.E.O.load_json(path)
                    if row['method'] != method or (configuration and row['request']['configuration_id'] != configuration):
                        continue
                    stages = path.with_name('JOB_STAGES.jsonl')
                    if stages.exists() and 'ARTIFICIAL_EVALUATION_START' in stages.read_text():
                        return path.parent
                if future.done():
                    future.result()
                return None

            first = distributed.wait_for(lambda: current_attempt(local, 'PR_PAULI'),
                reason='Initial demand did not reach the primary worker')
            with (directory/'peer.log').open('w') as log:
                worker = subprocess.Popen([sys.executable, '-B', str(overlap.ROOT/'scripts/run_rpe_q3_worker.py'),
                    '--profile', str(peer_path), '--controller', proxy.url, '--poll-seconds', '.02'],
                    stdout=log, stderr=subprocess.STDOUT)
            attempt = distributed.wait_for(lambda: late and current_attempt(peer, 'PR_DF', late['configuration_id']),
                reason='Prepared DF candidate did not refill an idle peer slot')
            self.assertFalse((first/'PROCESS_RETURN.json').exists(), 'First worker must still be executing')
            original_assignment = (attempt/'ASSIGNMENT.json').read_bytes()
            proxy.offline = True
            try:
                distributed.wait_for(lambda: (attempt/'WORKER_STATE.json').exists() and
                    overlap.E.O.load_json(attempt/'WORKER_STATE.json')['state'] == 'COMPLETED_PENDING_TRANSFER',
                    reason='Disconnected worker must preserve the original result')
                self.assertFalse((run/'jobs'/'PR_DF'/attempt.parent.name/'OUTCOME.json').exists())
            finally:
                proxy.offline = False
            future.result(timeout=25)
            self.assertEqual(original_assignment, (attempt/'ASSIGNMENT.json').read_bytes())
            self.assertEqual((attempt/'JOB_STAGES.jsonl').read_text().count('ARTIFICIAL_EVALUATION_START'), 1)
            queue = JobStore(store_path)
            before = distributed.launch_markers(directory)
            rows = queue.rows()
            row = queue.row(attempt.parent.name)
            self.assertEqual(row['attempt_id'], attempt.name)
            self.assertEqual(row['request']['configuration_id'], late['configuration_id'])
            self.assertEqual(row['request']['N_X'], [1]*13)
            self.assertIsNotNone(row['adopted_epoch'])
            self.assertEqual({queue.row(job)['host_id'] for job in late['preparation']}, {'fixture-peer'})
            events = [event['event'] for event in queue.events() if event['job_id'] == row['job_id']]
            self.assertEqual((events.count('ASSIGN'), events.count('ACCEPT'), events.count('ADOPT')), (1, 1, 1))
            minima = overlap.E.O.load_json(run/'CONFIRMED_MINIMA.json')['methods']
            self.assertEqual(minima['PR_DF']['confirmed_all_best']['configuration_id'], late['configuration_id'])
            timing = summarize(run)
            self.assertEqual(set(timing['by_host']), {'fixture-primary', 'fixture-peer'})
            metric = next(r for r in timing['rows'] if r['job_id'] == row['job_id'])
            self.assertIsNotNone(metric['A_candidate_ready_to_first_assignment_seconds'])
            self.assertIsNotNone(metric['D_receive_to_search_adopt_seconds'])
            self.assertGreater(timing['idle_slot_observations'].get('CANDIDATE_SHORTAGE', 0), 0)
            self.assertLessEqual(sum(row['assigned_epoch'] is not None and row['method']=='PR_DF' for row in rows),2)
            self.assertLessEqual(sum(row['assigned_epoch'] is not None and row['method']=='PR_PAULI' for row in rows),2)
            for stage in ('signal', 'compile'):
                self.assertEqual(queue.counts('PREPARATION:'+stage, started_only=True), 1)
            overlap.call(args, Driver)
            self.assertEqual(before, distributed.launch_markers(directory))
            self.assertEqual(queue.rows(), rows)
            return dict(artificial_only=True, distinct_worker_spools=2, actual_os_hosts=1,
                scientific_calls=0, resumed_additional_launches=0, same_attempt=row['attempt_id'],
                configuration_id=late['configuration_id'], ready_to_assignment_seconds=metric['A_candidate_ready_to_first_assignment_seconds'],
                receipt_to_search_adoption_seconds=metric['D_receive_to_search_adopt_seconds'],
                idle_slots=timing['idle_slot_observations'], by_host={h:dict(q3=r['count'], prefix_hits=r['dp_prefix_hits'],
                    prefix_event_coverage=r['prefix_event_coverage']) for h,r in timing['by_host'].items()})
        finally:
            proxy.offline = False
            if not future.done():
                distributed.write(run/'STOP_REQUESTED.json', {'reason': 'ARTIFICIAL_TEST_CLEANUP'})
            controller.shutdown(wait=True)
            if worker is not None and worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
            proxy.close()


def event_duration(second, first, exit_first):
    assert second['invocation'] == first['invocation']
    return second['monotonic'] - exit_first['monotonic']
