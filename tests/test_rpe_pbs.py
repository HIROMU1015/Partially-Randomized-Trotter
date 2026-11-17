"""Synthetic PBS submission, recovery and controller adoption checks.

No scheduler, SSH command or scientific worker is executed by these tests.
"""
import base64
from copy import deepcopy
import getpass
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_pbs as P
from trottertracks import rpe_preparation_worker as Preparation
from trottertracks.rpe_preparation_artifacts import ArtifactCache

spec = importlib.util.spec_from_file_location('pbs_gateway_test', ROOT/'scripts/rpe_pbs_gateway.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def settings():
    return dict(ssh_alias='ARTIFICIAL', remote_root='/shared/artificial-run',
                transport_source='ARTIFICIAL_FIXED_SOURCE', transport_root='/fixed/source',
                python='/private/python', drain_seconds=5,
                resources=dict(ncpus=2, mem='2gb', walltime='00:03:00', queue='workq'),
                workers=[dict(host_id='pbs-0', profile=dict(root='/fixed/scientific',
                    interpreter='/private/python', route_views={}, dp_workers=1))])


class FakeGateway:
    def __init__(self):
        self.calls = []
        self.harvest = dict(attempts=[], artifact_hashes=[], incomplete=[])

    def call(self, value):
        self.calls.append(deepcopy(value))
        if value['action'] == 'harvest':
            return deepcopy(self.harvest)
        if value['action'] == 'exchange':
            return dict(allocations=[], requests={})
        return dict(ok=True)

    def close(self):
        pass


class FakeExecutor:
    def __init__(self, directory):
        self.run_dir = Path(directory)
        self.state = SimpleNamespace(cv=threading.Condition(), tasks={}, closed=False,
                                     budgets={'PR_DF':7}, reserved={'PR_DF':0})
        self.preparation_pending = {}
        self.hosts = {}
        self.allocation_config = dict(offer_ttl_seconds=2)
        self.artifact_cache = ArtifactCache(self.run_dir/'artifacts')
        self.rows = {}
        self.store = SimpleNamespace(row=lambda key:self.rows.get(key))
        self.calls = []

    def rpc(self, endpoint, payload):
        self.calls.append((endpoint, deepcopy(payload)))
        if endpoint == '/v1/complete':
            self.rows[payload['job_id']] = dict(state='ACCEPTED')
        return dict(ok=True, assignments=[])

    def _user_stop_requested(self):
        return False

    def _owned_attempts(self):
        return []


class AdmissionTests(unittest.TestCase):
    def test_scientific_stage_requires_estimate_and_respects_remaining_time(self):
        report = dict(pbs=dict(accepting=True, remaining_seconds=40, minimum_task_seconds=2))
        compile = dict(route=dict(kind='preparation'), preparation=dict(stage='compile'))
        self.assertFalse(P.fits_request(compile, report, dict(drain_seconds=5)))
        self.assertTrue(P.fits_request(compile, report, dict(drain_seconds=5, task_seconds=dict(compile=30))))
        self.assertFalse(P.fits_request(compile, report, dict(drain_seconds=5, task_seconds=dict(compile=40))))
        self.assertTrue(P.fits_request(dict(route=dict(kind='artificial')), report, dict(drain_seconds=5)))
        report['pbs']['accepting'] = False
        self.assertFalse(P.fits_request(compile, report, dict(task_seconds=dict(compile=1))))

    def test_pbs_metadata_does_not_change_preparation_scientific_binding(self):
        fixed = dict(host_id='pbs-0', root='/fixed', interpreter='/private/python', dp_workers=1)
        operational = dict(fixed, pbs=dict(allocation_dir='/shared/next', worker_source='PINNED'))
        self.assertEqual(Preparation.preparation_profile_core(fixed),
                         Preparation.preparation_profile_core(operational))


class ManagerTests(unittest.TestCase):
    def manager(self, directory, gateway=None):
        executor = FakeExecutor(directory)
        gateway = gateway or FakeGateway()
        manager = P.PbsManager(executor, settings(), gateway=gateway)
        self.addCleanup(manager.close)
        return manager, executor, gateway

    def submit(self, manager):
        config = settings()['workers'][0]
        manager._submit('pbs-0', config, dict(ready_tasks=3, shortage=2))
        return next(iter(manager.state['allocations']))

    def test_allocation_submission_preserves_task_scientific_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            token = self.submit(manager)
            self.assertEqual(executor.state.budgets, {'PR_DF':7})
            self.assertEqual(executor.state.reserved, {'PR_DF':0})
            self.assertEqual(executor.calls, [])
            self.assertEqual(gateway.calls[0]['action'], 'ensure')
            self.assertEqual(gateway.calls[0]['spec']['token'], token)
            self.assertEqual(P.read(manager.path)['allocations'][token]['status'], 'INTENT')

    def test_restart_keeps_submission_token_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = FakeExecutor(directory)
            first = P.PbsManager(executor, settings(), gateway=FakeGateway())
            token = self.submit(first)
            first.close()
            restored = P.PbsManager(executor, settings(), gateway=FakeGateway())
            self.assertEqual(list(restored.state['allocations']), [token])
            restored.close()
            changed = settings(); changed['resources']['mem'] = '4gb'
            with self.assertRaisesRegex(ValueError, 'pinned PBS settings'):
                P.PbsManager(executor, changed, gateway=FakeGateway())

    def test_response_replay_does_not_repeat_rpc_and_changed_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            token = self.submit(manager)
            request = dict(request_id='same-message', endpoint='/v1/state',
                           payload=dict(host_id='pbs-0', job_id='original-task', state='RUNNING'))
            first = manager._dispatch(token, request)
            self.assertEqual(manager._dispatch(token, request), first)
            self.assertEqual(len(executor.calls), 1)
            altered = deepcopy(request); altered['payload']['job_id'] = 'other-task'
            with self.assertRaisesRegex(ValueError, 'changed content'):
                manager._dispatch(token, altered)

    def test_queued_held_and_exited_allocations_cannot_offer_new_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            token = self.submit(manager)
            for index, state in enumerate(('Q','H','F','HISTORY_MISSING_RECONCILE_SPOOL')):
                manager.allocations = [dict(token=token, status=state, worker=dict(accepting=True))]
                manager._dispatch(token, dict(request_id='request-'+str(index), endpoint='/v1/poll',
                    payload=dict(host_id='pbs-0', slots=2, capacity=2)))
                payload = executor.calls[-1][1]
                self.assertEqual(payload['slots'], 0)
                self.assertEqual(payload['capacity'], 0)
                self.assertFalse(payload['accepting_new_jobs'])

    def test_saved_cas_harvest_reuses_original_attempt_once(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            token = self.submit(manager)
            raw = b'ARTIFICIAL_COMPILE_RECEIPT'
            sha = hashlib.sha256(raw).hexdigest()
            assignment = dict(job_id='original-task', attempt_id='original-attempt', request_sha256='fixed-request')
            gateway.harvest = dict(host_id='pbs-0', artifact_hashes=[sha],
                blobs={sha:base64.b64encode(raw).decode()},
                attempts=[dict(assignment=assignment, files={'OUTCOME.json':'ARTIFICIAL'})], incomplete=[])
            manager._harvest(dict(token=token, status='F'))
            manager._harvest(dict(token=token, status='F'))
            self.assertEqual(executor.artifact_cache.path(sha).read_bytes(), raw)
            self.assertEqual(len(executor.calls), 1)
            self.assertEqual(executor.calls[0][0], '/v1/complete')
            self.assertEqual(executor.calls[0][1]['attempt_id'], 'original-attempt')
            self.assertEqual(len(gateway.calls), 3)  # ensure, inventory, missing blobs

    def test_missing_history_holds_running_attempt_until_worker_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            token = self.submit(manager)
            assignment = dict(job_id='original-task', attempt_id='original-attempt', request_sha256='fixed-request')
            executor.rows['original-task'] = dict(state='RUNNING')
            gateway.harvest = dict(host_id='pbs-0', artifact_hashes=[], attempts=[],
                                   incomplete=[dict(assignment=assignment, state=dict(state='RUNNING'))])
            missing = dict(token=token, status='HISTORY_MISSING_RECONCILE_SPOOL', worker=dict(state='READY'))
            manager._harvest(missing)
            self.assertFalse(manager.state['allocations'][token]['harvested'])
            self.assertEqual(executor.calls, [])
            manager._harvest(dict(missing, worker=dict(state='EXITED')))
            self.assertTrue(manager.state['allocations'][token]['harvested'])
            self.assertEqual(executor.calls[0][0], '/v1/state')
            self.assertEqual(executor.calls[0][1]['state'], 'UNKNOWN')
            self.assertFalse(executor.calls[0][1]['detail']['automatic_retry'])


class GatewayTests(unittest.TestCase):
    def test_lost_qsub_reply_recovers_original_scheduler_job_without_resubmit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); token = 'artificial-token'
            allocation = root/'allocations'/token
            G.atomic(allocation/'SPEC.json', dict(token=token, profile=dict(host_id='pbs-0')))
            with patch.object(G.subprocess, 'run', side_effect=OSError('reply lost')) as submit:
                with self.assertRaises(OSError):
                    G.submit_once(allocation)
                G.submit_once(allocation)
                self.assertEqual(submit.call_count, 1)
            job_id = '123.prism'
            job = dict(Job_Name='RPE'+token[:12], Job_Owner=getpass.getuser()+'@prism', job_state='R')
            with patch.object(G, 'qstat', return_value={job_id:job}):
                snapshot = G.snapshot(root)
            self.assertEqual(snapshot[0]['job_id'], job_id)
            self.assertTrue(G.read(allocation/'JOB_ID.json')['recovered'])
            with patch.object(G, 'qstat', return_value={}):
                missing = G.snapshot(root)
            self.assertEqual(missing[0]['status'], 'HISTORY_MISSING_RECONCILE_SPOOL')
            self.assertEqual(missing[0]['job_id'], job_id)

    def test_saved_artifact_harvest_never_calls_a_worker_or_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); allocation = root/'allocation'; work = root/'worker'
            attempt = work/'task'/'original-attempt'
            G.atomic(allocation/'SPEC.json', dict(profile=dict(host_id='pbs-0')))
            G.atomic(allocation/'PROFILE.json', dict(work_dir=str(work)))
            G.atomic(attempt/'ASSIGNMENT.json', dict(job_id='task', attempt_id='original-attempt', request=dict(route=dict(kind='artificial'))))
            G.atomic(attempt/'WORKER_STATE.json', dict(state='COMPLETED_PENDING_TRANSFER'))
            G.atomic(attempt/'PROCESS_RETURN.json', dict(returncode=0))
            G.atomic(attempt/'OUTCOME.json', dict(status='ARTIFICIAL'))
            with patch.object(G.subprocess, 'run') as process:
                result = G.harvest(allocation)
            process.assert_not_called()
            self.assertEqual(len(result['attempts']), 1)
            self.assertEqual(result['attempts'][0]['assignment']['attempt_id'], 'original-attempt')
            self.assertIn('PROCESS_RETURN.json', result['attempts'][0]['files'])


if __name__ == '__main__':
    unittest.main()
