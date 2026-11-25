"""Artificial transport/process recovery checks; no H4 numerical leaf is run."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

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
    def test_missing_preparation_bundle_keeps_independent_q3_readiness(self):
        worker = object.__new__(W.Worker)
        failed = {'job_id': 'prep-missing'}
        independent = {'job_id': 'q3-ready'}

        def prepare(challenges):
            if failed in challenges:
                raise OSError('missing input bundle')

        with patch.object(worker, '_prepare_challenge_artifacts', side_effect=prepare), \
             patch.object(worker, '_check_challenges',
                          return_value={'host_fingerprints': {'q3-ready': 'verified'}}) as check:
            result = worker._check_readiness_batch([failed, independent], {})
        check.assert_called_once_with([independent], {})
        self.assertEqual(result['host_fingerprints'], {'q3-ready': 'verified'})
        self.assertEqual(result['readiness_errors'], {'prep-missing': 'missing input bundle'})

    def test_prelaunch_failure_releases_only_after_supervisor_unlocks(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            W.atomic(attempt/'WORKER_STATE.json', dict(state='UNKNOWN', detail=dict(
                automatic_retry=False, reason='ValueError: Existing adapter dry-check failed')))
            W.atomic(attempt/'WORKER_DRY-CHECK_RETURN.json', dict(returncode=1))
            W.atomic(attempt/'SUPERVISOR_STARTED.json', dict(pid=143))
            W.atomic(attempt/'SUPERVISOR_LAUNCH.json', dict(pid=143))
            with (attempt/'WORKER_PROCESS.lock').open('wb') as lock:
                W.fcntl.flock(lock, W.fcntl.LOCK_EX)
                self.assertFalse(W.failed_dry_check_released(attempt))
            self.assertTrue(W.failed_dry_check_released(attempt))
            (attempt/'SCIENTIFIC_STARTING.json').touch()
            self.assertFalse(W.failed_dry_check_released(attempt))
            self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'], 'UNKNOWN')

    def test_native_supervisor_preserves_a_scoped_provider_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root/'provider-calls.jsonl'
            launcher = root/'provider-python'
            body = root/'provider-launch.py'
            body.write_text('import json,os,sys\nfrom pathlib import Path\n'+
                'args=sys.argv[1:]\n'+
                'script=next((Path(a).name for a in args if a.endswith(".py")), "")\n'+
                'if script=="run_rpe_q3_worker.py": raise SystemExit("Provider launcher rejects transport supervisor")\n'+
                'with open('+repr(str(calls))+',"a") as stream: stream.write(json.dumps(args)+"\\n")\n'+
                'os.execv('+repr(sys.executable)+',['+repr(sys.executable)+',*args])\n')
            launcher.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(body))+' \"$@\"\n')
            launcher.chmod(0o755)
            profile, assignment = fixture(directory)
            profile['interpreter'] = str(launcher)
            controller = FakeController(assignment)
            worker = W.Worker(profile, 'http://unused', transport=controller)
            try:
                worker.tick()
                finish(worker)
                attempt = worker.work/assignment['job_id']/assignment['attempt_id']
                self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'], 'TRANSFERRED',
                    {p.name:p.read_text() for p in attempt.glob('WORKER_*') if p.is_file()})
                self.assertEqual(W.read(attempt/'PROCESS_RETURN.json')['returncode'], 0)
                self.assertEqual(W.read(attempt/'WORKER_PROFILE.json')['interpreter'], str(launcher))
                commands = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertTrue(any('run_rpe_optimizer_q3_job.py' in str(args) for args in commands))
                self.assertFalse(any('run_rpe_q3_worker.py' in str(args) for args in commands))
                self.assertEqual(controller.complete_calls, 1)
            finally:
                for process in worker.processes.values():
                    process.wait(timeout=8)
                worker.close()

    def test_unknown_detached_session_releases_only_after_the_session_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                       start_new_session=True)
            try:
                W.atomic(attempt/'SUPERVISOR_STARTED.json', dict(pid=process.pid, session_id=process.pid))
                W.atomic(attempt/'SUPERVISOR_LAUNCH.json', dict(pid=process.pid))
                (attempt/'WORKER_PROCESS.lock').touch()
                (attempt/'TRANSPORT_TIMING.jsonl').write_text(json.dumps(W.stamp()))
                self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'UNKNOWN')
            finally:
                process.terminate()
                process.wait(timeout=5)
            self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'EXITED')
            self.assertFalse((attempt/'PROCESS_RETURN.json').exists())

    def test_unknown_attempt_only_releases_capacity_with_exit_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            current = 'worker-test:4e765e4d-eda5-4ab3-a055-3dd044269f59'
            with patch.object(W, 'stamp', return_value={'clock_id': current}):
                self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'UNKNOWN')
                timing = attempt / 'TRANSPORT_TIMING.jsonl'
                timing.write_text(json.dumps({'clock_id': current}))
                self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'UNKNOWN')
                old_boot = '93477fd0-0fe1-4542-80a1-8bffdea6d1ce'
                timing.write_text(json.dumps({'clock_id': 'elsewhere:' + old_boot}))
                self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'UNKNOWN')
                timing.write_text(json.dumps({'clock_id': 'worker-test:' + old_boot}))
                self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'EXITED')
                self.assertEqual(W.resource_state(attempt, 'COMPLETED_PENDING_TRANSFER'),
                                 'COMPLETED_PENDING_TRANSFER')
                timing.unlink()
                (attempt / 'PROCESS_RETURN.json').touch()
                self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'UNKNOWN')
                W.atomic(attempt / 'PROCESS_RETURN.json', {'returncode': 1})
                self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'EXITED')
                W.atomic(attempt / 'WORKER_LAUNCH.json', {'pid': 99})
                with patch.object(W, 'process_alive', return_value=True):
                    self.assertEqual(W.resource_state(attempt, 'UNKNOWN'), 'UNKNOWN')

    def test_storage_limit_stops_new_offers_and_preserves_completion(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(W.Storage, 'SAMPLE_SECONDS', 0):
            base = Path(temporary)
            files = base/'files'; files.mkdir()
            profile, assignment = fixture(temporary)
            profile['storage_limit'] = dict(limit_bytes=8192, roots=[str(files)], sample_path=str(base/'usage.json'))
            remote = FakeController(assignment)
            worker = W.Worker(profile, 'http://127.0.0.1:1', transport=remote)
            try:
                worker.tick()
                self.assertEqual(len(worker.records), 1)
                large = files/'fixture.data'
                large.write_bytes(b'x'*32768)
                worker.tick()
                held = [value for path, value in remote.messages if path=='/v1/poll'][-1]
                self.assertEqual(held['slots'], 0)
                self.assertFalse(held['accepting_new_jobs'])
                self.assertEqual(held['resource_report']['storage']['reason'], 'STORAGE_LIMIT_REACHED')
                finish(worker)
                self.assertEqual(remote.complete_calls, 1)
                self.assertEqual(large.stat().st_size, 32768)
            finally:
                for process in worker.processes.values():
                    process.wait(timeout=8)
                worker.close()
            status = W.mailbox(profile, 'status', {'metadata_only':True})
            self.assertFalse(status['accepting_new_jobs'])
            self.assertEqual(status['slots'], 0)
            large.unlink()
            reopened = W.mailbox(profile, 'status', {'metadata_only':True})
            self.assertTrue(reopened['accepting_new_jobs'])
            self.assertEqual(reopened['slots'], 1)

    def test_storage_counts_unique_blocks_and_fails_closed_on_unreadable_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base/'worker'; root.mkdir()
            nested = root/'spool'; nested.mkdir()
            second = base/'cache'; second.mkdir()
            original = nested/'file'; original.write_bytes(b'a'*32768)
            (second/'link').hardlink_to(original)
            policy = dict(limit_bytes=2**20, roots=[str(root),str(nested),str(second)],
                          sample_path=str(base/'report.json'))
            report = W.Storage.report(policy)
            self.assertEqual(report['used_bytes'], W.Storage._used_bytes([root, second]))
            self.assertLess(report['used_bytes'], 65536)
            with patch.object(W.Storage, '_used_bytes', side_effect=OSError('fixture unreadable')), \
                 patch.object(W.Storage, 'SAMPLE_SECONDS', 0):
                self.assertEqual(W.Storage.report(policy)['reason'], 'STORAGE_USAGE_UNAVAILABLE')

    def test_storage_can_evict_only_checkpoint_cache_when_spool_has_no_safe_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'prefix'
            cache.mkdir()
            checkpoint = cache / ('1' * 64 + '.json.gz')
            checkpoint.write_bytes(b'x' * 32768)
            admissions = cache / '.prefix-admissions.json'
            admissions.write_text('{}')
            protected = cache / 'PROTECTED_RECEIPT.json'
            protected.write_text('retained')
            policy = dict(roots=[str(cache)], limit_bytes=32768, sample_path=str(root / 'usage.json'))
            report = W.Storage.report(policy, prefix_cache=str(cache), refresh=True)
            self.assertTrue(report['accepting_new_jobs'])
            self.assertEqual(report['evicted_prefix_checkpoints'], 1)
            self.assertFalse(checkpoint.exists())
            self.assertEqual(admissions.read_text(), '{}')
            self.assertEqual(protected.read_text(), 'retained')

    def test_storage_reclaims_only_acknowledged_worker_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runs = base/'runs'
            spool = runs/'run-a'/'workers'/'pbs-0'

            def make(job, state, status='PREPARATION_COMPLETE'):
                attempt = spool/job/'attempt-1'
                workspace = attempt/'workspaces'
                workspace.mkdir(parents=True)
                (workspace/'expanded-input').write_bytes(b'0'*131072)
                (attempt/'WORKER_PROCESS.lock').touch()
                W.atomic(attempt/'WORKER_PROFILE.json', dict(work_dir=str(spool)))
                W.atomic(attempt/'ASSIGNMENT.json', dict(job_id=job, attempt_id=attempt.name,
                    request_sha256='fixed-request', request=dict(route=dict(kind='preparation'))))
                W.atomic(attempt/'OUTCOME.json', dict(status=status,
                    request_id=job, request_sha256='fixed-request', artifacts={}))
                W.atomic(attempt/'PROCESS_RETURN.json', dict(returncode=0))
                W.save_state(attempt, state)
                return attempt

            acknowledged = make('acknowledged', 'TRANSFERRED')
            pending = make('pending', 'COMPLETED_PENDING_TRANSFER')
            unknown = make('unknown', 'UNKNOWN')
            running = make('running', 'RUNNING')
            failed = make('failed', 'TRANSFERRED', 'PREPARATION_FAILED')
            dry_check = acknowledged/'_dry_check'
            dry_check.mkdir()
            (dry_check/'input').write_bytes(b'1'*131072)
            limit = W.Storage._used_bytes([runs])
            policy = dict(limit_bytes=limit, roots=[str(runs)], sample_path=str(base/'usage.json'))
            first = W.Storage.report(policy, pbs_runs=str(runs), refresh=True)
            self.assertTrue(first['accepting_new_jobs'])
            self.assertEqual(first['reclaimed_copies'], 1)
            self.assertFalse(dry_check.exists())
            self.assertTrue((acknowledged/'workspaces').is_dir())

            policy['limit_bytes'] = first['used_bytes']
            second = W.Storage.report(policy, spools=(str(spool),), refresh=True)
            self.assertTrue(second['accepting_new_jobs'])
            self.assertFalse((acknowledged/'workspaces').exists())
            self.assertEqual(W.read(acknowledged/'WORKER_STATE.json')['state'], 'TRANSFERRED')
            self.assertTrue((acknowledged/'OUTCOME.json').is_file())
            for attempt in (pending, unknown, running, failed):
                self.assertTrue((attempt/'workspaces'/'expanded-input').is_file())

    def test_transferred_cleanup_is_immediate_and_preserves_unreceived_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root/'job'/'attempt'
            dry = attempt/'_dry_check'
            dry.mkdir(parents=True)
            (dry/'temporary').write_bytes(b'artificial')
            (attempt/'WORKER_PROCESS.lock').touch()
            W.atomic(attempt/'ASSIGNMENT.json', dict(job_id='job', attempt_id='attempt',
                request=dict(route=dict(kind='artificial'))))
            W.atomic(attempt/'OUTCOME.json', dict(status='ARTIFICIAL'))
            policy = dict(roots=[str(root)])
            W.save_state(attempt, 'COMPLETED_PENDING_TRANSFER')
            W.Storage.reclaim_transferred_attempt(policy, attempt)
            self.assertTrue(dry.exists())
            W.save_state(attempt, 'TRANSFERRED')
            W.Storage.reclaim_transferred_attempt(policy, attempt)
            self.assertFalse(dry.exists())
            self.assertTrue((attempt/'OUTCOME.json').is_file())

    def test_transferred_preparation_failure_requires_proof_that_body_never_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for state, prelaunch, body_log in [('TRANSFERRED', True, False),
                                               ('TRANSFERRED', False, False),
                                               ('TRANSFERRED', True, True),
                                               ('UNKNOWN', True, False)]:
                with self.subTest(state=state, prelaunch=prelaunch, body_log=body_log):
                    attempt = root/f'{state}-{prelaunch}-{body_log}'/'attempt'
                    workspace = attempt/'workspace'
                    workspace.mkdir(parents=True)
                    (workspace/'expanded-input').write_text('working copy')
                    (attempt/'WORKER_PROCESS.lock').touch()
                    W.atomic(attempt/'ASSIGNMENT.json', dict(job_id=attempt.parent.name, attempt_id=attempt.name,
                        request_sha256='fixed', request=dict(route=dict(kind='preparation'))))
                    W.atomic(attempt/'OUTCOME.json', dict(status='PREPARATION_FAILED', request_id=attempt.parent.name,
                        request_sha256='fixed', artifacts={}, git_bundles=[],
                        workspace_error=dict(scientific_body_started=False) if prelaunch else {}))
                    W.atomic(attempt/'PROCESS_RETURN.json', dict(returncode=2))
                    if body_log:
                        (attempt/'PREPARATION_BODY.log').touch()
                    W.save_state(attempt, state)
                    W.Storage.reclaim_transferred_attempt(dict(roots=[str(root)]), attempt)
                    removed = state == 'TRANSFERRED' and prelaunch and not body_log
                    self.assertEqual(workspace.exists(), not removed)
                    self.assertTrue((attempt/'OUTCOME.json').is_file())
                    self.assertTrue((attempt/'PROCESS_RETURN.json').is_file())

    def test_storage_reclaims_configured_route_checkpoint_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default, routed = root/'default', root/'routed'
            default.mkdir(); routed.mkdir()
            checkpoint = routed/('1'*64+'.json.gz')
            checkpoint.write_bytes(b'x'*32768)
            profile = dict(prefix_cache_directory=str(default), route_profiles=[
                dict(profile={'prefix_cache_directory':str(routed)})])
            policy = dict(roots=[str(root)], limit_bytes=32768, sample_path=str(root/'usage.json'))
            report = W.Storage.report(policy, cache_directories=W.Storage.prefix_caches(profile), refresh=True)
            self.assertTrue(report['accepting_new_jobs'])
            self.assertEqual(report['evicted_prefix_checkpoints'], 1)
            self.assertFalse(checkpoint.exists())

    def test_storage_watcher_follows_the_resumed_invocation(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        from reclaim_rpe_worker_storage import invocation_active

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.assertFalse(invocation_active(run))
            W.atomic(run/'LAUNCH.json', dict(invocation_id='new'))
            W.atomic(run/'EXIT.json', dict(invocation_id='previous'))
            self.assertTrue(invocation_active(run))
            W.atomic(run/'EXIT.json', dict(invocation_id='new'))
            self.assertFalse(invocation_active(run))

    def test_local_cache_duplicate_reclaim_refetches_queued_and_preserves_unmatched_files(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        from reclaim_rpe_worker_storage import local_duplicate_cache
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, cache = root/'run', root/'cache'
            central = run/'distributed/artifacts'
            central.mkdir(parents=True)
            profile = dict(host_id='local', artifact_cache_directory=str(cache), work_dir=str(root/'spool'),
                           storage_limit=dict(roots=[str(cache)], limit_bytes=1024, sample_path=str(root/'usage.json')))
            previous, other = root/'previous', root/'other'
            for parent in (previous, other):
                parent.mkdir()
                W.atomic(parent/'LAUNCH.json', dict(invocation_id=parent.name))
                W.atomic(parent/'EXIT.json', dict(invocation_id=parent.name))
            eligible = {}
            for name, copies in (('duplicate', (cache,central)), ('queued', (cache,central)),
                                 ('previous', (cache,previous/'distributed/artifacts')),
                                 ('different', (cache,other/'distributed/artifacts')), ('private', (cache,))):
                raw = name.encode()
                digest = hashlib.sha256(raw).hexdigest()
                eligible[name] = digest
                for directory in copies:
                    target = directory/digest[:2]/digest
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
            W.atomic(run/'CONTROLLER_CONFIG.json', dict(state_directory=str(root)))
            W.atomic(root/'CURRENT_TARGET.json', dict(run_directory=str(run)))
            W.atomic(root/'RUNS.json', {p.name:dict(run_directory=str(p),evaluation_identity=identity)
                                       for p,identity in ((run,'same'),(previous,'same'),(other,'different'))})
            (root/'REGISTRY.lock').touch(); (root/'usage.lock').touch()
            W.atomic(root/'usage.json', dict(policy=dict(limit_bytes=1024,roots=[str(cache)]),
                       report=dict(reason='STORAGE_LIMIT_REACHED',limit_bytes=1024,used_bytes=1024**3)))
            with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
                db.execute('CREATE TABLE jobs(host_id,state,request)')
                db.execute('INSERT INTO jobs VALUES (NULL,?,?)', ('QUEUED',json.dumps({'sha':eligible['queued']})))
            self.assertEqual(set(local_duplicate_cache(run,profile)),{eligible['duplicate'],eligible['previous'],eligible['queued']})
            for name in ('different','private'):
                digest = eligible[name]
                self.assertEqual((cache/digest[:2]/digest).read_text(),name)
            for name in ('duplicate','queued'):
                digest = eligible[name]
                self.assertEqual((central/digest[:2]/digest).read_text(),name)

    def test_remote_cache_reclaim_preserves_unknown_and_unverified_copies(self):
        from trottertracks.rpe_worker_storage import duplicate_inventory, reclaim_verified_duplicates
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root/'cache'
            spool = root/'spool'
            attempt = spool/'job'/'attempt'
            attempt.mkdir(parents=True)
            sample = root/'usage.json'
            policy = dict(limit_bytes=1024, roots=[str(cache)])
            profile = dict(work_dir=str(spool), artifact_cache_directory=str(cache),
                           storage_limit=dict(policy, sample_path=str(sample)))
            W.atomic(sample, dict(policy=policy, report=dict(reason=None, limit_bytes=1024, used_bytes=980)))
            sample.with_suffix('.lock').touch()
            files = {}
            for name in ('duplicated', 'unknown', 'local-only'):
                sha = hashlib.sha256(name.encode()).hexdigest()
                path = cache/sha[:2]/sha
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
                files[name] = path
            W.atomic(attempt/'ASSIGNMENT.json', {'input': files['unknown'].name})
            inventory = duplicate_inventory(profile)
            proof = {files[name].name: inventory[files[name].name] for name in ('duplicated', 'unknown')}
            W.atomic(attempt/'WORKER_STATE.json', {'state': 'RUNNING'})
            self.assertEqual(reclaim_verified_duplicates(profile, proof), {})
            W.atomic(attempt/'WORKER_STATE.json', {'state': 'UNKNOWN'})
            second = root/'older-pbs-run'/'workers'/'pbs-0'
            other = second/'old-job'/'attempt'
            other.mkdir(parents=True)
            W.atomic(other/'WORKER_STATE.json', {'state': 'UNKNOWN'})
            W.atomic(other/'ASSIGNMENT.json', {'input': files['duplicated'].name})
            self.assertEqual(reclaim_verified_duplicates(profile, proof, spools=[second, spool]), {})
            W.atomic(attempt/'WORKER_STATE.json', {'state': 'COMPLETED_PENDING_TRANSFER'})
            self.assertEqual(reclaim_verified_duplicates(profile, proof), {})
            W.atomic(attempt/'OUTCOME.json', {'artifacts': {'saved': files['unknown'].name}})
            W.atomic(attempt/'PROCESS_RETURN.json', {'returncode': 0})
            self.assertEqual(set(reclaim_verified_duplicates(profile, proof)), {files['duplicated'].name})
            self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'], 'COMPLETED_PENDING_TRANSFER')
            self.assertTrue(files['unknown'].is_file())
            self.assertTrue(files['local-only'].is_file())

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

    def test_fresh_process_reads_and_recovers_large_exact_json_without_execution(self):
        # Fresh -I processes avoid a false PASS from an earlier catalog/adapter
        # import having already removed Python's integer-string digit limit.
        with tempfile.TemporaryDirectory() as temporary:
            profile, assignment = fixture(temporary)
            directory = Path(temporary)
            large = '9'*4500
            number = directory/'LARGE_EXACT.json'
            number.write_text('{"denominator":1,"numerator":'+large+'}')
            probe = '''import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
from trottertracks import rpe_q3_worker as W
value = W.read(sys.argv[2])
print(json.dumps(dict(limit=getattr(sys,'get_int_max_str_digits',lambda:0)(),
    canonical_sha256=hashlib.sha256(W.canonical(value)).hexdigest())))
'''
            observed = json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', probe,
                str(ROOT), str(number)], cwd=ROOT))
            self.assertEqual(observed['limit'], 0)
            self.assertEqual(observed['canonical_sha256'], hashlib.sha256(number.read_bytes()).hexdigest())

            # Seed saved artificial completion artifacts only. Mailbox status
            # must collect them without starting any adapter or supervisor.
            attempt = Path(profile['work_dir'])/assignment['job_id']/assignment['attempt_id']
            attempt.mkdir(parents=True)
            W.atomic(attempt/'ASSIGNMENT.json', assignment)
            W.atomic(attempt/'WORKER_PROFILE.json', profile)
            W.save_state(attempt, 'RUNNING', synthetic_saved_completion=True)
            W.atomic(attempt/'PROCESS_RETURN.json', dict(returncode=0, signal=None))
            body = ('"status":"MODEL_BOUND_MEETS_ADOPTED_THRESHOLD",'
                '"MSE_upper":{"numerator":'+large+',"denominator":1}')
            (attempt/'ARTIFICIAL_RESULT.json').write_text('{'+body+'}')
            outcome = '{'+body+',"request_id":'+json.dumps(assignment['job_id'])+',"route":"artificial"}'
            (attempt/'OUTCOME.json').write_text(outcome)
            original_hash = hashlib.sha256((attempt/'OUTCOME.json').read_bytes()).hexdigest()
            profile_path = directory/'PROFILE.json'; W.atomic(profile_path, profile)
            response = json.loads(subprocess.check_output([sys.executable, '-I', '-B',
                str(ROOT/'scripts/run_rpe_q3_worker.py'), '--profile', str(profile_path), '--mailbox', 'status'],
                input=b'{}', cwd=ROOT))
            row = response['attempts'][0]
            self.assertEqual(row['state'], 'COMPLETED_PENDING_TRANSFER', row)
            self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'], 'COMPLETED_PENDING_TRANSFER')
            transferred = base64.b64decode(row['files']['OUTCOME.json'])
            self.assertEqual(hashlib.sha256(transferred).hexdigest(), original_hash)
            self.assertEqual(hashlib.sha256((attempt/'OUTCOME.json').read_bytes()).hexdigest(), original_hash)
            for name in ('SCIENTIFIC_STARTING.json', 'SUPERVISOR_STARTED.json', 'WORKER_LAUNCH.json', 'JOB_STAGES.jsonl'):
                self.assertFalse((attempt/name).exists(), 'Saved completion recovery must not execute an adapter')

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
            self.assertEqual(row['state'], 'COMPLETED_PENDING_TRANSFER', row)
            self.assertEqual(status['slots'], 1)
            self.assertIn('OUTCOME.json', row['files'])
            self.assertEqual(status['cached_prefixes'], ['artificial-prefix'])
            ack = dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
            compact = W.mailbox(profile, 'status', {'metadata_only':True})
            self.assertNotIn('files', compact['attempts'][0])
            request = dict(ack, request_sha256=assignment['request_sha256'])
            self.assertEqual(W.mailbox(profile, 'result', request)['files'], row['files'])
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                W.mailbox(profile, 'result', dict(request, request_sha256='other'))
            self.assertEqual(W.mailbox(profile, 'ack', ack), dict(ok=True, duplicate=False))
            self.assertEqual(W.mailbox(profile, 'ack', ack), dict(ok=True, duplicate=True))
            status = W.mailbox(profile, 'status')
            self.assertEqual(status['attempts'][0]['state'], 'TRANSFERRED')
            self.assertNotIn('files', status['attempts'][0])
            compact = W.mailbox(profile, 'status', {'metadata_only':True})
            self.assertEqual(compact['attempts'], [])
            self.assertEqual(compact['cached_prefixes'], ['artificial-prefix'])
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
