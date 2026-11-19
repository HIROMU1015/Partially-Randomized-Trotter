"""Synthetic checks of PBS allocation boundaries and durable communication only."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_pbs_worker as P


def profile():
    return dict(host_id='pbs-artificial', root=str(ROOT), work_dir='/tmp/unlaunched',
                interpreter=sys.executable, route_views={}, dp_workers=2,
                slot_policy=dict(mode='auto', initial_slots=1),
                pbs=dict(worker_source='ARTIFICIAL', library_threads=2,
                         minimum_task_seconds=5, drain_seconds=2))


def allocation():
    return dict(job_id='41.prism', ncpus=8, memory_bytes=8*1024**3,
                walltime_seconds=180, select_nodes=1)


class AllocationTests(unittest.TestCase):
    def test_actual_pbs_bound_wins_over_machine_cpu_and_nodefile(self):
        p = profile()
        p['route_profiles'] = [dict(profile=dict(dp_workers=3))]
        result, resources = P.allocation_profile(p, allocation(),
            environ=dict(PBS_JOBID='41.prism', PBS_NP='96'),
            snapshot=dict(allowed_cpu_count=96, allowed_memory_bytes=2**40))
        self.assertEqual(resources['cpu_count'], 8)
        self.assertEqual(resources['task_cpu_count'], 6)
        self.assertEqual(result['slot_policy']['max_slots'], 1)
        self.assertEqual(result['route_profiles'][0]['profile']['dp_workers'], 3)
        self.assertEqual(p['slot_policy'], dict(mode='auto', initial_slots=1))

    def test_cpu_memory_and_compiler_footprint_share_one_bound(self):
        p = profile()
        p['preparation_profiles'] = [dict(stage='compile', profile=dict(task_cpu_count=3))]
        p['pbs']['task_memory_bytes'] = 3*1024**3
        result, resources = P.allocation_profile(p, allocation(),
            environ=dict(PBS_JOBID='41.prism'),
            snapshot=dict(allowed_cpu_count=6, allowed_memory_bytes=4*1024**3))
        self.assertEqual(result['slot_policy']['max_slots'], 1)
        self.assertEqual(resources['memory_bytes'], 4*1024**3)
        self.assertTrue(resources['one_shared_slot_pool'])
        p['pbs']['task_memory_bytes'] = 20*1024**3
        with self.assertRaisesRegex(ValueError, 'memory footprint'):
            P.allocation_profile(p, allocation(), environ=dict(PBS_JOBID='41.prism'), snapshot={})

    def test_job_identity_single_node_and_oversized_dp_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PBS_JOBID'):
            P.allocation_profile(profile(), allocation(), environ={})
        multi = dict(allocation(), select_nodes=2)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            P.allocation_profile(profile(), multi, environ=dict(PBS_JOBID='41.prism'))
        small = dict(allocation(), ncpus=2)
        with self.assertRaisesRegex(ValueError, 'CPU footprint'):
            P.allocation_profile(profile(), small, environ=dict(PBS_JOBID='41.prism'), snapshot={})


class SpoolTests(unittest.TestCase):
    def test_timeout_restart_reuses_the_one_acquisition_before_next_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            first = P.SpoolTransport(directory, deadline=time.monotonic()+1,
                                     timeout_seconds=.005, poll_seconds=.001)
            with self.assertRaises(TimeoutError):
                first('/v1/poll', dict(host_id='ARTIFICIAL', slots=2))
            original = next(first.requests.glob('*.json'))
            saved = json.loads(original.read_text())
            restarted = P.SpoolTransport(directory, deadline=time.monotonic()+1,
                                         timeout_seconds=.005, poll_seconds=.001)
            with self.assertRaises(TimeoutError):
                restarted('/v1/poll', dict(host_id='ARTIFICIAL', slots=1))
            self.assertEqual(list(restarted.requests.glob('*.json')), [original])
            self.assertEqual(json.loads(original.read_text()), saved)
            P.atomic(restarted.responses/original.name, dict(ok=True, result=dict(assignments=['SAME_ORIGINAL_ATTEMPT'])))
            result = restarted('/v1/poll', dict(host_id='ARTIFICIAL', slots=1))
            self.assertEqual(result['assignments'], ['SAME_ORIGINAL_ATTEMPT'])
            self.assertEqual(list(restarted.requests.glob('*.json')), [])

    def test_timeout_retains_request_for_controller_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = P.SpoolTransport(directory, deadline=time.monotonic()+1,
                                         timeout_seconds=.02, poll_seconds=.001)
            with self.assertRaises(TimeoutError):
                transport('/v1/complete', dict(job_id='ARTIFICIAL'))
            requests = list((Path(directory)/'requests').glob('*.json'))
            self.assertEqual(len(requests), 1)
            self.assertEqual(json.loads(requests[0].read_text())['endpoint'], '/v1/complete')

    def test_response_roundtrip_cleans_only_consumed_message(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = P.SpoolTransport(directory, deadline=time.monotonic()+5,
                                         timeout_seconds=2, poll_seconds=.001)
            def respond():
                while not list(transport.requests.glob('*.json')):
                    time.sleep(.001)
                request = next(transport.requests.glob('*.json'))
                data = json.loads(request.read_text())
                P.atomic(transport.responses/request.name,
                         dict(ok=True, request_id=data['request_id'], result=dict(ok=True, duplicate=True)))
            thread = threading.Thread(target=respond)
            thread.start()
            result = transport('/v1/complete', dict(job_id='ARTIFICIAL'))
            thread.join(timeout=2)
            self.assertTrue(result['duplicate'])
            self.assertEqual(list(transport.requests.glob('*.json')), [])
            self.assertEqual(list(transport.responses.glob('*.json')), [])


class AdmissionTests(unittest.TestCase):
    def worker(self, directory):
        worker = P.AllocationWorker.__new__(P.AllocationWorker)
        worker.directory = Path(directory)
        worker.deadline = time.monotonic()+1
        worker.minimum_task_seconds = 5
        worker.idle_release_seconds = 0
        worker.idle_since = None
        worker.drain_seconds = 2
        worker.draining = False
        worker.stop = threading.Event()
        worker.host_id = 'pbs-artificial'
        worker.profile = profile()
        worker.profile['work_dir'] = directory
        return worker

    def test_delayed_assignment_never_launches_past_admission_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(directory)
            attempt = Path(directory)/'ARTIFICIAL'/'attempt'
            attempt.mkdir(parents=True)
            with patch.object(P.Worker, '_launch') as launch:
                worker._launch(attempt)
            launch.assert_not_called()
            state = json.loads((attempt/'WORKER_STATE.json').read_text())
            self.assertEqual(state['state'], 'UNKNOWN')
            self.assertFalse(state['detail']['scientific_execution_started'])
            self.assertFalse(state['detail']['automatic_retry'])

    def test_authoritative_pbs_bounds_allow_auto_slots_without_job_cgroup(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(directory)
            worker.pbs_resources = dict(actual=dict(allocation(), ncpus=2))
            observed = dict(allowed_cpu_count=96, allowed_memory_bytes=2**40,
                            allocation_bounds_complete=False, missing_controls=['cpu', 'memory'])
            with patch.object(P.Worker, 'resource_snapshot', return_value=observed):
                resources = worker.resource_snapshot()
            self.assertEqual(resources['allowed_cpu_count'], 2)
            self.assertEqual(resources['allowed_memory_bytes'], 8*1024**3)
            self.assertTrue(resources['allocation_bounds_complete'])
            self.assertEqual(resources['missing_controls'], ['cpu', 'memory'])

    def test_new_allocation_offers_its_own_slots_with_a_shared_attempt_spool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool = root/'spool'
            spool.mkdir()
            snapshot = dict(allowed_cpu_count=12, allowed_memory_bytes=49*1024**3)
            old = self.worker(root/'small')
            new = self.worker(root/'large')
            for worker, slots in ((old, 1), (new, 6)):
                worker.directory.mkdir()
                worker.profile.update(root=str(ROOT), work_dir=str(spool),
                                      slot_policy=dict(mode='auto', initial_slots=slots))
                worker.work = spool
                worker.records = {}
            with patch.object(P.AllocationWorker, 'resource_snapshot', return_value=snapshot):
                self.assertEqual(old._resource_offer()['slots'], 1)
                self.assertEqual(new._resource_offer()['slots'], 6)
                self.assertEqual(old._resource_offer()['slots'], 1)
            for worker, slots in ((old, 1), (new, 6)):
                state = json.loads((worker.directory/'SLOT_STATE.json').read_text())
                self.assertEqual(state['target_slots'], slots)

    def test_idle_release_writes_exit_without_waiting_for_walltime(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(directory)
            worker.deadline = time.monotonic()+60
            worker.poll_seconds = .001
            worker.last_heartbeat = -1
            worker.exit_reason = None
            worker.start_clock = P.stamp()
            worker.pbs_resources = dict(actual=allocation())
            worker.slots = 1
            P.atomic(Path(directory)/'CONTROL.json', dict(release_when_idle=True))
            with patch.object(worker, 'tick') as tick, patch.object(worker, 'pending', return_value=False), patch.object(worker, 'close'):
                worker.run_allocation()
            tick.assert_called_once_with(acquire=True)
            state = json.loads((Path(directory)/'STATE.json').read_text())
            self.assertEqual(state['state'], 'EXITED')
            self.assertEqual(state['exit_reason'], 'CONTROLLER_IDLE_RELEASE')
            self.assertFalse(state['accepting'])


if __name__ == '__main__':
    unittest.main()
