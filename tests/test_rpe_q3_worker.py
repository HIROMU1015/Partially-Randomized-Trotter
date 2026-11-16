"""Artificial transport/process recovery checks; no H4 numerical leaf is run."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_q3_worker as W


def fixture(directory):
    profile = dict(host_id='worker-test', root=str(ROOT), work_dir=str(Path(directory) / 'spool'),
        interpreter=sys.executable, route_views={}, prefix_cache_directory=str(Path(directory) / 'prefix'), slots=1)
    request = dict(request_id='aabb0011', method='PR_DF', configuration_id='ARTIFICIAL', N_X=[2]*13, N_Y=[2]*13,
        expected=dict(configuration_id='ARTIFICIAL'),
        route=dict(kind='artificial', interpreter=sys.executable, view_commit=None,
            artificial=dict(weights=[1]*13, scale='1', tau='7', sleep_seconds=.08)))
    assignment = dict(job_id=request['request_id'], method=request['method'], attempt_id='attempt-1', request=request,
        request_sha256=W.digest(request), source_manifest=dict(commit=W.head(ROOT), files={
            'scripts/run_rpe_optimizer_q3_job.py':hashlib.sha256((ROOT/'scripts/run_rpe_optimizer_q3_job.py').read_bytes()).hexdigest()}),
        input_hashes=[], prefix_key='artificial-prefix')
    return profile, assignment


class FakeController:
    def __init__(self, assignment, lose_ack=False):
        self.assignment = assignment; self.offered = False; self.completed = None; self.messages = []; self.lose_ack = lose_ack
        self.complete_calls = 0

    def __call__(self, endpoint, payload):
        self.messages.append((endpoint, payload))
        if endpoint == '/v1/poll':
            assignments = [] if self.offered else [self.assignment]
            self.offered = True
            return dict(assignments=assignments, stop=False)
        if endpoint == '/v1/transfer-metric':
            return dict(ok=True)
        if endpoint == '/v1/state':
            return dict(ok=True)
        if endpoint == '/v1/complete':
            self.complete_calls += 1
            duplicate = self.completed is not None
            if self.completed is not None:
                assert self.completed == payload
            self.completed = payload
            if self.lose_ack and self.complete_calls == 1:
                raise OSError('Artificial lost acknowledgement after server accepted result')
            return dict(ok=True, duplicate=duplicate)
        raise AssertionError(endpoint)


def finish(worker, timeout=8):
    deadline = time.monotonic()+timeout
    while worker.pending() and time.monotonic() < deadline:
        worker.tick(acquire=False); time.sleep(.015)
    if worker.pending():
        raise AssertionError('Artificial worker did not finish: '+str({str(p):W.read(p/'WORKER_STATE.json')for p in worker.records}))


class WorkerTests(unittest.TestCase):
    def test_artificial_execute_transfer_duplicate_ack_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary); remote = FakeController(assignment, lose_ack=True)
            worker = W.Worker(profile, 'http://127.0.0.1:1', poll_seconds=.01, transport=remote)
            worker.tick(); finish(worker)
            self.assertEqual(remote.complete_calls, 2)
            timings = [p['timing'] for route,p in remote.messages if route == '/v1/transfer-metric']
            self.assertEqual(len(timings), 1)
            self.assertGreater(timings[0]['bytes'], 0)
            files = remote.completed['files']
            self.assertNotIn('REQUEST.json', files); self.assertNotIn('LAUNCH.json', files)
            self.assertIn('PROCESS_RETURN.json', files)
            self.assertEqual(json.loads(base64.b64decode(files['PROCESS_RETURN.json']))['returncode'], 0)
            result = json.loads(base64.b64decode(files['OUTCOME.json']))
            self.assertEqual(result['MSE_upper'], dict(numerator=13, denominator=2))
            attempt = next(iter(worker.records))
            self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'], 'TRANSFERRED')
            stages = (attempt/'JOB_STAGES.jsonl').read_text()
            self.assertEqual(stages.count('ARTIFICIAL_EVALUATION_START'), 1)
            worker.close()
            resumed = W.Worker(profile, 'http://127.0.0.1:1', transport=remote)
            resumed.tick(acquire=False)
            self.assertEqual((attempt/'JOB_STAGES.jsonl').read_text(), stages)
            resumed.close()
            receipt = W.read(attempt/'PROCESS_RETURN.json'); receipt['returncode'] = 1
            W.atomic(attempt/'PROCESS_RETURN.json', receipt)
            with self.assertRaisesRegex(ValueError, 'Nonzero adapter exit'):
                W.validate_completed(profile, assignment, attempt)

    def test_source_request_and_input_hashes_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary)
            wrong = deepcopy(assignment); wrong['request']['N_X'][0] = 1
            with self.assertRaisesRegex(ValueError, 'Request SHA256'):
                W.validate_assignment(profile, wrong)
            wrong = deepcopy(assignment); wrong['source_manifest']['files']['scripts/run_rpe_optimizer_q3_job.py'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SOURCE hash'):
                W.validate_assignment(profile, wrong)
            wrong = deepcopy(assignment); wrong['input_hashes'] = [dict(commit=W.head(ROOT), path='AGENTS.md', sha256='0'*64)]
            with self.assertRaisesRegex(ValueError, 'input hash'):
                W.validate_assignment(profile, wrong)
            self.assertFalse((Path(profile['work_dir'])/'aabb0011').exists())

    def test_host_paths_resolve_without_changing_original_or_scientific_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary); before = deepcopy(assignment)
            resolved = W.validate_assignment(profile, assignment)
            self.assertEqual(assignment, before)
            self.assertEqual(resolved['request_id'], assignment['job_id'])
            self.assertEqual(resolved['execution_profile']['prefix_cache_directory'], profile['prefix_cache_directory'])

    def test_host_dp_workers_override_preserves_original_request_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary)
            assignment['request']['execution_profile'] = dict(dp_workers=8)
            assignment['request_sha256'] = W.digest(assignment['request'])
            original = deepcopy(assignment)
            resolved = W.validate_assignment(dict(profile, dp_workers=2), assignment)
            self.assertEqual(resolved['execution_profile']['dp_workers'], 2)
            self.assertEqual(assignment, original)
            for key in ('method', 'configuration_id', 'N_X', 'N_Y', 'expected', 'request_id'):
                self.assertEqual(resolved[key], original['request'][key])
            for invalid in (0, -1, True, '2'):
                with self.assertRaisesRegex(ValueError, 'host dp_workers'):
                    W.validate_assignment(dict(profile, dp_workers=invalid), assignment)

    def test_ambiguous_start_is_unknown_and_never_reexecuted(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary); remote = FakeController(assignment)
            worker = W.Worker(profile, 'http://127.0.0.1:1', transport=remote)
            worker._acquire(assignment); attempt = next(iter(worker.records))
            W.save_state(attempt, 'STARTING')
            worker.tick(acquire=False)
            self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'], 'UNKNOWN')
            self.assertFalse((attempt/'SCIENTIFIC_STARTING.json').exists())
            worker.close()
            resumed = W.Worker(profile, 'http://127.0.0.1:1', transport=remote)
            resumed.tick(acquire=False)
            self.assertFalse((attempt/'SCIENTIFIC_STARTING.json').exists())
            resumed.close()

    def test_polling_restart_adopts_live_supervisor_and_saved_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary)
            assignment['request']['route']['artificial']['sleep_seconds'] = .3
            assignment['request_sha256'] = W.digest(assignment['request'])
            remote = FakeController(assignment)
            worker = W.Worker(profile, 'http://127.0.0.1:1', transport=remote)
            worker.tick(); worker.tick(acquire=False)
            attempt = next(iter(worker.records)); process = worker.processes[attempt]
            deadline = time.monotonic()+4
            while not (attempt/'WORKER_LAUNCH.json').exists() and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue((attempt/'WORKER_LAUNCH.json').exists())
            worker.close()
            resumed = W.Worker(profile, 'http://127.0.0.1:1', transport=remote)
            finish(resumed)
            process.wait(timeout=4)
            self.assertEqual((attempt/'JOB_STAGES.jsonl').read_text().count('ARTIFICIAL_EVALUATION_START'), 1)
            self.assertEqual(W.read(attempt/'PROCESS_RETURN.json')['returncode'], 0)
            resumed.close()

    def test_ssh_mailbox_submit_status_duplicate_and_ack(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary)
            first = W.mailbox(profile, 'submit', assignment)
            self.assertEqual(first, dict(ok=True, duplicate=False))
            duplicate = W.mailbox(profile, 'submit', assignment)
            self.assertEqual(duplicate, dict(ok=True, duplicate=True))
            deadline = time.monotonic()+8
            while time.monotonic() < deadline:
                status = W.mailbox(profile, 'status')
                row = status['attempts'][0]
                if row['state'] == 'COMPLETED_PENDING_TRANSFER':break
                self.assertIn(row['state'], ('STARTING', 'RUNNING'))
                time.sleep(.02)
            self.assertEqual(row['state'], 'COMPLETED_PENDING_TRANSFER')
            self.assertEqual(status['slots'], 1)
            self.assertIn('OUTCOME.json', row['files'])
            self.assertEqual(status['cached_prefixes'], ['artificial-prefix'])
            ack = dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
            self.assertEqual(W.mailbox(profile, 'ack', ack), dict(ok=True, duplicate=False))
            self.assertEqual(W.mailbox(profile, 'ack', ack), dict(ok=True, duplicate=True))
            status = W.mailbox(profile, 'status')
            self.assertEqual(status['attempts'][0]['state'], 'TRANSFERRED')
            self.assertNotIn('files', status['attempts'][0])
            attempt = Path(profile['work_dir'])/assignment['job_id']/assignment['attempt_id']
            self.assertEqual((attempt/'JOB_STAGES.jsonl').read_text().count('ARTIFICIAL_EVALUATION_START'), 1)
            profile_path = Path(temporary)/'profile.json';profile_path.write_text(json.dumps(profile))
            output = subprocess.check_output([sys.executable, '-B', str(ROOT/'scripts/run_rpe_q3_worker.py'),
                '--profile', str(profile_path), '--mailbox', 'status'], input=b'{}', cwd=ROOT)
            self.assertEqual(len(output.splitlines()), 1)
            self.assertEqual(json.loads(output)['host_id'], profile['host_id'])

    def test_host_daemon_lock_and_unsafe_bundle_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile, _ = fixture(temporary)
            worker = W.Worker(profile, 'http://127.0.0.1:1', transport=lambda *a:None)
            with self.assertRaisesRegex(ValueError, 'Another polling worker'):
                W.Worker(profile, 'http://127.0.0.1:1')
            for path in ('../SOURCE.json', '/absolute', 'a/../OUTCOME.json'):
                with self.assertRaises(ValueError):W.safe_relative(path)
            worker.close()


if __name__ == '__main__':
    unittest.main()
