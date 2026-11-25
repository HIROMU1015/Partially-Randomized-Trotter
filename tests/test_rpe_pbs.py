"""Synthetic PBS submission, recovery and controller adoption checks.

No scheduler, SSH command or scientific worker is executed by these tests.
"""
import base64
from copy import deepcopy
import getpass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_pbs as P
from trottertracks import rpe_preparation_worker as Preparation
from trottertracks.rpe_preparation_artifacts import ArtifactCache

spec = importlib.util.spec_from_file_location('pbs_gateway_test', ROOT/'scripts/rpe_pbs_gateway.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
spec = importlib.util.spec_from_file_location('pbs_storage_worker_test', ROOT/'scripts/run_rpe_pbs_storage_worker.py')
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


class StorageSampleTests(unittest.TestCase):
    def test_storage_launcher_verifies_bytes_before_restoring_execute_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker launcher'
            path.write_text('#!/bin/sh\ntest "$1" = --check-imports && test "$4" = --profile && test -f "$5"\n')
            path.chmod(0o600)
            gateway = P.Gateway(dict(gateway_python=sys.executable, transport_root=directory),
                                ssh=SimpleNamespace(command=lambda: ['/bin/sh', '-c']))
            with self.assertRaises(subprocess.CalledProcessError):
                gateway.verify_storage_launcher(dict(path=str(path), sha256='0'*64))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            gateway.verify_storage_launcher(dict(path=str(path), sha256=expected))
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)

    def test_slow_compute_scan_respects_headroom_and_does_not_change_byte_limit(self):
        from trottertracks import rpe_worker_storage as storage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = dict(limit_bytes=50*P.GIB, roots=[directory], sample_path=str(root/'sample.json'))
            with patch.object(storage, '_used_bytes', return_value=42*P.GIB) as measure:
                report = storage.report(policy, du_timeout=W.COMPUTE_DU_TIMEOUT)
                self.assertTrue(report['accepting_new_jobs'])
                measure.assert_called_once_with([directory], timeout=W.COMPUTE_DU_TIMEOUT)
            sample = Path(policy['sample_path'])
            facts = json.loads(sample.read_bytes())
            facts.update(sampled_epoch=100.)
            facts['report']['measurement_seconds'] = 45.
            P.atomic(sample, facts)
            observer = W.GatewayStorage(sample, max_age=10, local_clock=True)
            with patch.object(W.time, 'time', return_value=144.):
                self.assertTrue(observer.report(policy)['accepting_new_jobs'])
                facts['report']['used_bytes'] = 49*P.GIB
                P.atomic(sample, facts)
                self.assertFalse(observer.report(policy)['accepting_new_jobs'])

    def test_compute_worker_uses_first_fresh_sample_and_rejects_stale_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.json'
            policy = dict(limit_bytes=50*P.GIB, roots=['/shared'])
            report = dict(limit_bytes=policy['limit_bytes'], used_bytes=P.GIB,
                          accepting_new_jobs=True, reason=None)
            P.atomic(path, dict(policy=policy, sampled_epoch=12345., report=report))
            observer = W.GatewayStorage(path, max_age=10, local_clock=True)
            with patch.object(W.time, 'time', side_effect=[12346., 12356.]):
                self.assertEqual(observer.report(policy), report)
                self.assertFalse(observer.report(policy)['accepting_new_jobs'])

    def test_refresh_is_measured_on_the_worker_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.json'
            policy = dict(limit_bytes=50*P.GIB, roots=['/shared'])
            report = dict(limit_bytes=policy['limit_bytes'], used_bytes=P.GIB,
                          accepting_new_jobs=True, reason=None)
            observer = W.GatewayStorage(path, max_age=10)
            with patch.object(W.time, 'monotonic', side_effect=[100., 103., 104., 114.]):
                P.atomic(path, dict(policy=policy, sampled_epoch=12345., report=report))
                self.assertFalse(observer.report(policy)['accepting_new_jobs'])
                P.atomic(path, dict(policy=policy, sampled_epoch=12348., report=report))
                self.assertEqual(observer.report(policy), report)
                self.assertEqual(observer.report(policy), report)
                self.assertFalse(observer.report(policy)['accepting_new_jobs'])


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
        self.store = SimpleNamespace(row=lambda key:self.rows.get(key), preparation_reservations=lambda: {})
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
    def test_q3_adapter_selection_preserves_pinned_pbs_and_preparation_profiles(self):
        routes = [dict(route_kind='pf_b_pf6', view_commit='pf-view',
                       profile=dict(adapter_manifest=dict(commit='old'), dp_workers=2)),
                  dict(route_kind='prdf_d2_common_entry', view_commit='df-view',
                       profile=dict(adapter_manifest=dict(commit='old'), dp_workers=2))]
        pbs = dict(workers=[dict(host_id='pbs-0', profile=dict(route_profiles=routes, preparation_profiles=[{'input':'fixed'}]))],
                   q3_task_seconds=[dict(id='pf', route_kind='pf_b_pf6', view_commit='pf-view', hosts=['pbs-0'],
                                         adapter_source='old', seconds=14400, memory_bytes=8192)])
        original = deepcopy(pbs)
        distributed = dict(pbs=pbs, pbs_pinned_settings=deepcopy(pbs))
        new = dict(commit='new', files={'scripts/run_rpe_optimizer_q3_job.py':'hash'})
        P.apply_q3_adapter(distributed, dict(host_id='pbs-0', rule_ids=['pf'], root='/fixed/adapter', manifest=new))
        self.assertEqual(distributed['pbs_pinned_settings'], original)
        self.assertEqual(routes[0]['profile'], dict(adapter_root='/fixed/adapter', adapter_manifest=new, dp_workers=2))
        self.assertEqual(routes[1], original['workers'][0]['profile']['route_profiles'][1])
        self.assertEqual(pbs['workers'][0]['profile']['preparation_profiles'], [{'input':'fixed'}])
        self.assertEqual(pbs['q3_task_seconds'][0], dict(original['q3_task_seconds'][0], adapter_source='new'))

    def test_add_horizon_keeps_short_route_and_fixed_pbs_contract(self):
        from trottertracks.rpe_host_profiles import select_host_profile

        host = 'pbs-0'
        base = dict(host_id=host, interpreter='/fixed/python', route_views={'view': '/fixed/view'},
                    adapter_root='/fixed/base', adapter_manifest={'commit': 'old'}, dp_workers=3)
        route = dict(route_kind='pf_b_pf6', view_commit='view')
        pbs = dict(transport_root='/fixed/base', transport_source='old',
                   workers=[dict(host_id=host, profile=dict(base, route_profiles=[dict(route, profile=base)]))],
                   q3_task_seconds=[dict(route, method='FULL_PF', hosts=[host], seconds=60, memory_bytes=4096)])
        original = deepcopy(pbs)
        alternate = dict(base, adapter_root='/fixed/horizon',
                         adapter_manifest={'commit': 'new', 'files': {'adapter.py': 'hash'}},
                         route_views=dict(base['route_views'], risk='/fixed/risk'))
        change = dict(host_id=host, transport_root='/fixed/horizon', transport_source='new',
                      route_profiles=[dict(route, profile=alternate, horizon_risk_commit='risk')])
        with self.assertRaisesRegex(ValueError, 'runtime or resource settings'):
            bad = deepcopy(change)
            bad['route_profiles'][0]['profile']['interpreter'] = '/different/python'
            P.apply_q3_horizon({'pbs': deepcopy(pbs)}, bad)
        distributed = {'pbs': pbs}
        P.apply_q3_horizon(distributed, change)
        self.assertEqual(distributed['pbs_pinned_settings'], original)
        profile = pbs['workers'][0]['profile']
        request = dict(method='FULL_PF', route=dict(kind='pf_b_pf6', view_commit='view'))
        self.assertEqual(select_host_profile(profile, {'request': request}), base)
        request['route']['horizon_risk_source'] = {'commit': 'risk'}
        self.assertEqual(select_host_profile(profile, {'request': request}), alternate)
        self.assertEqual(P.task_estimate(request, pbs, host), 60)
        def old_selector(selected, _):
            # The original run's selector has no horizon key.
            route, = selected['route_profiles']
            self.assertEqual(route['profile'], alternate)
            return route['profile']
        with patch('trottertracks.rpe_q3_worker.select_host_profile', side_effect=old_selector):
            self.assertEqual(P.resource_need(request, profile, pbs, host), (3, 4096))

    def test_pbs_packs_more_measured_q3_without_changing_cpu_or_larger_request(self):
        host = 'pbs-0'
        adapter, view, risk = ('a'*40, 'b'*40, 'c'*40)
        runtime = {'dp_workers': 2, 'adapter_manifest': {'commit': adapter}}
        request = dict(method='PR_DF', route=dict(kind='prdf_d2_common_entry', view_commit=view,
                       horizon_risk_source={'commit': risk}), N_X=[53]*14, N_Y=[53]*14)
        settings = dict(resources={'ncpus': 96, 'mem': '350gb'}, workers=[dict(host_id=host, profile=runtime)],
                        allocation_sizing={'q3_startup_memory_bytes_per_task': 8*P.GIB})
        original = deepcopy(settings)
        distributed = dict(pbs=settings)
        estimate = dict(host=host, method='PR_DF', route_kind=request['route']['kind'],
                        view_commit=view, horizon_risk_commit=risk, adapter_source=adapter,
                        dp_workers=2, rounds=14, min_peak_shots=8, max_peak_shots=53, memory_bytes=4*P.GIB)
        with patch('trottertracks.rpe_q3_worker.select_host_profile', return_value=runtime) as select:
            self.assertEqual(P.packed_resources([request]*48, runtime, settings, host), (86, 345, 43))
            P.apply_q3_measured_memory(distributed, [estimate])
            self.assertEqual(P.packed_resources([request]*48, runtime, settings, host), (96, 193, 48))
            self.assertEqual(P.resource_need(dict(request, N_X=[54]*14), runtime, settings, host), (2, 8*P.GIB))
            self.assertEqual(P.resource_need(dict(request, N_X=[53]*13, N_Y=[53]*13), runtime, settings, host), (2, 8*P.GIB))
            select.return_value = dict(runtime, adapter_manifest={'commit': 'd'*40})
            self.assertEqual(P.resource_need(request, runtime, settings, host), (2, 8*P.GIB))
        self.assertEqual(distributed['pbs_pinned_settings'], original)

    def test_release_q3_estimate_binds_the_route_and_worker_and_can_be_disabled(self):
        route = dict(route_kind='pauli', view_commit='view', adapter_source='new-adapter')
        policy = dict(schema='RPE_PBS_WALLTIME_POLICY_V1', id='walltime-v1',
                      computation_source='science', transport_source='worker',
                      q3=[dict(route, id='pauli', method='PR_PAULI', seconds=14400)],
                      preparation=[])
        workers = [dict(host_id=host, profile=dict(route_profiles=[
            dict(route_kind='pauli', view_commit='view',
                 profile=dict(adapter_manifest=dict(commit=adapter)))]))
                   for host, adapter in [('reviewed', 'new-adapter'), ('legacy', 'old-adapter')]]
        options = dict(transport_source='worker', workers=workers, task_seconds=dict(q3=1))
        P.apply_walltime_policy(options, policy, {}, computation_source='science')
        request = dict(method='PR_PAULI', route=dict(kind='pauli', view_commit='view'))
        self.assertEqual(P.task_estimate(request, options, 'reviewed'), 14400)
        self.assertIsNone(P.task_estimate(request, options, 'legacy'))
        self.assertIsNone(P.task_estimate(dict(request, method='PR_DF'), options))
        report = dict(pbs=dict(accepting=True, remaining_seconds=14429))
        self.assertFalse(P.fits_request(request, report, options, 'reviewed'))
        report['pbs']['remaining_seconds'] += 1
        self.assertTrue(P.fits_request(request, report, options, 'reviewed'))
        P.apply_walltime_policy(options, policy, {'pauli': 0}, computation_source='science')
        self.assertFalse(P.fits_request(request, report, options, 'reviewed'))
        with self.assertRaises(ValueError):
            P.apply_walltime_policy(options, policy, {'unknown': 1}, computation_source='science')

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

    def test_preparation_estimate_requires_matching_provider_and_source(self):
        report = dict(pbs=dict(accepting=True, remaining_seconds=210))
        settings = dict(drain_seconds=30, task_seconds=dict(compile=1),
                        preparation_task_seconds=[dict(stage='compile', provider='PAULI_COMPILE',
                                                       source='PINNED', seconds=180)])
        preparation = dict(stage='compile', provider='PAULI_COMPILE', source=dict(commit='PINNED'))
        self.assertTrue(P.fits_request(dict(preparation=preparation), report, settings))
        self.assertFalse(P.fits_request(dict(preparation=dict(preparation, provider='DF_COMPILE')), report, settings))
        self.assertFalse(P.fits_request(dict(preparation=dict(preparation, source=dict(commit='OTHER'))), report, settings))
        report['pbs']['remaining_seconds'] = 209
        self.assertFalse(P.fits_request(dict(preparation=preparation), report, settings))

    def test_df_signal_parallelism_matches_the_reserved_cpus(self):
        command = ['/fixed/python', '-B', '/fixed/run_pr_df_candidate_signal_jobs.py', '--source', 'fixed']
        bound = Preparation.bind_df_signal_workers(command, 2)
        self.assertEqual(bound, command + ['--workers','2'])
        self.assertEqual(Preparation.bind_df_signal_workers(bound, 2), bound)
        with self.assertRaisesRegex(ValueError, 'differs from its reserved CPUs'):
            Preparation.bind_df_signal_workers(bound, 4)


class ManagerTests(unittest.TestCase):
    def test_q3_worker_proofs_expire_and_are_invalidated_by_failure_or_worker_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = self.manager(directory)
            worker = dict(accepting=True, job_id='allocated', hostname='node', pid=1, source_sha='fixed',
                          start_clock=dict(clock_id='node-boot', monotonic=1.0))
            facts = dict(schema='RPE_Q3_CURRENT_HOST_FINGERPRINT_V1', request_sha256='fixed-request')
            proof = dict(host_fingerprints={'job': facts, 'prep': dict(schema='PREPARATION')})
            clock = dict(clock_id='node-boot', monotonic=2.0)

            def poll(payload, *, at=10.0, current=worker, token='original', sent=clock):
                with patch.object(P.time, 'monotonic', return_value=at):
                    reused = manager._reuse_q3_worker_proofs(token, current, sent, payload)
                return reused, payload.get('host_fingerprints', {})

            self.assertEqual(poll(deepcopy(proof))[0], set())
            self.assertEqual(poll({}, at=11), ({'job'}, {'job': facts}))
            self.assertEqual(poll({}, token='other-allocation'), (set(), {}))
            self.assertEqual(poll({}, at=10 + P.Q3_WORKER_PROOF_SECONDS), (set(), {}))
            poll(deepcopy(proof))
            self.assertEqual(poll(dict(readiness_errors={'job': 'changed source'})), (set(), {}))
            self.assertEqual(poll({}), (set(), {}))
            poll(deepcopy(proof))
            self.assertEqual(poll({}, current=dict(worker, accepting=False)), (set(), {}))
            self.assertEqual(poll({}), (set(), {}))
            poll(deepcopy(proof), current=dict(worker, monotonic=clock['monotonic']+P.Q3_WORKER_PROOF_SECONDS))
            self.assertEqual(poll({}), (set(), {}))
            poll(deepcopy(proof))
            self.assertEqual(poll(deepcopy(proof), current=dict(worker, pid=2,
                start_clock=dict(clock_id='node-boot', monotonic=3.0)))[0], set())
            self.assertEqual(poll({}, current=dict(worker, pid=2),
                                  sent=dict(clock, monotonic=4.0)), (set(), {}))

            # Background checks can finish long before a durable poll is
            # delivered. The request time must not renew the proof's age.
            late = dict(clock, monotonic=100.)
            background = dict(deepcopy(proof), readiness_checked_monotonic={'job': 2.})
            poll(deepcopy(background), sent=late, current=dict(worker, monotonic=110.))
            self.assertEqual(poll({}, at=21), ({'job'}, {'job': facts}))
            self.assertEqual(poll({}, at=22), (set(), {}))
            self.assertNotIn('job', poll(deepcopy(background), sent=late,
                                        current=dict(worker, monotonic=123.))[1])

    def test_prefetch_checks_worker_visibility_and_keeps_other_poll_work(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            manager.settings.update(stage_artifacts_over_ssh=True, request_timeout_seconds=20)
            raw = b'x' * (1024 * 1024)
            sha = hashlib.sha256(raw).hexdigest()
            blob = base64.b64encode(raw).decode()
            executor.artifact_cache.put({sha: blob})
            target = Path(directory) / 'shared-cache'
            token = 'owned-allocation'
            manager.state['allocations'][token] = dict(status='R', submitted_clock=P.stamp(), spec=dict(profile=dict(
                host_id='pbs-0', artifact_cache_directory=str(target))))
            manager.allocations = [dict(token=token, status='R', worker=dict(accepting=True))]
            done = threading.Event()
            self.addCleanup(done.set)
            transfers = []

            def transfer(source, destination, identity):
                transfers.append((destination, identity))
                if not done.wait(3):
                    raise TimeoutError('test transport was not released')

            gateway.stage_artifact = transfer
            needs_bundle = [dict(job_id=f'prep-{n}', request=dict(preparation=dict(
                workspaces={'input':dict(input_git_bundles=[dict(sha256=sha)])}))) for n in range(2)]
            ready = dict(job_id='independent', request=dict())
            assignment = dict(job_id='already-claimed')

            def rpc(endpoint, payload):
                executor.calls.append((endpoint, payload))
                return (dict(stop=False, assignments=[assignment], eligibility_challenges=[*needs_bundle, ready])
                        if endpoint == '/v1/poll' else dict(blobs={sha:blob}))

            executor.rpc = rpc
            def poll(rpc_id, visible=()):
                return manager._dispatch_ready(token, dict(request_id=rpc_id, endpoint='/v1/poll',
                    payload=dict(host_id='pbs-0', slots=1, cached_artifact_sha256=list(visible))))['result']

            initial = poll('first')
            self.assertEqual(initial['assignments'], [assignment])
            self.assertEqual(initial['eligibility_challenges'], [ready])
            done.set()
            manager.prefetched_inputs[token,sha]['future'].result(3)
            self.assertEqual(poll('gateway-acked')['eligibility_challenges'], [ready])
            self.assertEqual(poll('worker-visible', [sha])['eligibility_challenges'], [*needs_bundle, ready])
            self.assertEqual(poll('first'), initial)
            manager.prefetched_inputs[token,sha]['worker_visibility_deadline'] = 0.
            self.assertEqual(poll('worker-missing')['eligibility_challenges'], [*needs_bundle, ready])
            request = dict(request_id='worker-get', endpoint='/v1/artifacts/get',
                           payload=dict(host_id='pbs-0', hashes=[sha]))
            self.assertEqual(manager._dispatch_ready(token, request)['result'], dict(blobs={sha:blob}))
            self.assertEqual(manager._dispatch_ready(token, request)['result'], dict(blobs={sha:blob}))
            self.assertEqual(transfers, [(ArtifactCache(target).path(sha), sha)])
            self.assertEqual(sum(endpoint == '/v1/artifacts/get' for endpoint, _ in executor.calls), 1)
            self.assertEqual(executor.state.reserved['PR_DF'], 0)

    def test_shared_artifact_transfer_leaves_poll_available_and_replays_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            manager.settings['stage_artifacts_over_ssh'] = True
            payload = b'x' * (1024 * 1024)
            sha = hashlib.sha256(payload).hexdigest()
            executor.artifact_cache.put({sha: base64.b64encode(payload).decode()})
            target = Path(directory) / 'shared-cache'
            token = 'owned-allocation'
            manager.state['allocations'][token] = dict(status='F', spec=dict(profile=dict(
                host_id='pbs-0', artifact_cache_directory=str(target))))
            ready = threading.Event()
            release = threading.Event()
            transfers = []

            def transfer(source, destination, identity):
                ready.set()
                if not release.wait(3):
                    raise TimeoutError('test transport was not released')
                self.assertEqual(identity, sha)
                transfers.append(destination)
                destination.parent.mkdir(parents=True)
                destination.write_bytes(source.read_bytes())

            gateway.stage_artifact = transfer
            task = dict(request_id='first', endpoint='/v1/artifacts/get',
                        payload=dict(host_id='pbs-0', hashes=[sha]))
            self.assertIsNone(manager._dispatch_ready(token, task))
            self.assertTrue(ready.wait(3))
            poll = dict(request_id='second', endpoint='/v1/poll', payload=dict(host_id='pbs-0', slots=0))
            self.assertTrue(manager._dispatch_ready(token, poll)['ok'])
            self.assertIsNone(manager._dispatch_ready(token, task))
            release.set()
            manager.artifact_transfers[token, 'first'][1].result(3)
            expected = dict(ok=True, result=dict(blobs={}))
            self.assertEqual(manager._dispatch_ready(token, task), expected)
            self.assertEqual(manager._dispatch_ready(token, task), expected)
            self.assertEqual(transfers, [ArtifactCache(target).path(sha)])
            self.assertEqual(ArtifactCache(target).get([sha]), {sha: base64.b64encode(payload).decode()})
            self.assertEqual([endpoint for endpoint, _ in executor.calls], ['/v1/poll'])

    def test_preparation_trial_changes_the_live_pool_and_preserves_pinned_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            fixed = settings()
            fixed['resources'].update(ncpus=96, mem='350gb')
            profile = fixed['workers'][0]['profile']
            profile['preparation_profiles'] = [
                dict(stage='compile', provider='DF_CALIBRATION', source_commit=source,
                     profile=dict(task_cpu_count=2))
                for source in ('PINNED', 'NEW_ESTIMATE')
            ]
            fixed['preparation_task_seconds'] = [dict(stage='compile', provider='DF_CALIBRATION',
                source='PINNED', seconds=3600)]
            fixed['allocation_sizing'] = dict(per_task_resources=True, q3_startup_memory_bytes_per_task=8*P.GIB)
            executor = FakeExecutor(directory)
            base = P.PbsManager(executor, fixed, gateway=FakeGateway())
            pinned_file = base.directory/'SETTINGS.json'
            original = pinned_file.read_bytes()
            base.close()
            executor.distributed = dict(pbs=deepcopy(fixed), preparation_profiles={'pbs-0': deepcopy(profile)})
            P.apply_preparation_resources(executor.distributed, [dict(host='pbs-0', provider='DF_CALIBRATION',
                source='PINNED', task_cpu_count=6, memory_bytes=18*P.GIB, interpreter='/fixed/compiler/python')])
            P.apply_preparation_resources(executor.distributed, [dict(host='pbs-0', provider='DF_CALIBRATION',
                source='NEW_ESTIMATE', stage='compile', seconds=1200, task_cpu_count=4, memory_bytes=32*P.GIB)])
            manager = P.PbsManager(executor, executor.distributed['pbs'], gateway=FakeGateway())
            self.addCleanup(manager.close)
            request = dict(route=dict(kind='preparation'), preparation=dict(stage='compile',
                provider='DF_CALIBRATION', source=dict(commit='PINNED')))
            effective = manager.configured['pbs-0']['profile']
            self.assertEqual(effective['preparation_profiles'][0]['profile']['task_cpu_count'], 6)
            self.assertEqual(effective['preparation_profiles'][0]['profile']['interpreter'], '/fixed/compiler/python')
            self.assertEqual(executor.distributed['preparation_profiles']['pbs-0']['preparation_profiles'],
                             effective['preparation_profiles'])
            self.assertEqual(P.packed_resources([request]*20, effective, manager.settings, 'pbs-0'), (96, 289, 16))
            new_request = deepcopy(request)
            new_request['preparation']['source']['commit'] = 'NEW_ESTIMATE'
            self.assertEqual(P.task_estimate(new_request, manager.settings, 'pbs-0'), 1200)
            self.assertIsNone(P.task_estimate(new_request, fixed, 'pbs-0'))
            new_request['preparation']['stage'] = 'holdout'
            self.assertIsNone(P.task_estimate(new_request, manager.settings, 'pbs-0'))
            self.assertEqual(pinned_file.read_bytes(), original)

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

    def test_preparation_wait_is_extended_only_for_live_gateway_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = self.manager(directory)
            before = manager.preparation_readiness_wait_seconds('pbs-0')
            self.assertLess(before, 300)
            manager.allocations = [dict(host_id='pbs-0', status='R', worker=dict(accepting=True))]
            self.assertEqual(manager.preparation_readiness_wait_seconds('pbs-0'), 300)
            manager.last_error = 'transport offline'
            self.assertEqual(manager.preparation_readiness_wait_seconds('pbs-0'), before)

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

    def test_confirmed_shared_pilot_replays_owned_assignment_with_no_queued_work(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            manager.hold_node = True
            token = self.submit(manager)
            owner = dict(job_id='original-task', host_id='pbs-0', attempt_id='original-attempt', state='ASSIGNED')
            original = dict(owner, started_epoch=None, request=dict(route=dict(kind='artificial')))
            executor.rows[owner['job_id']] = original
            closed = dict(token=token, status='F', host_id='pbs-0', pilot_token='held', job_id='41.prism',
                          scheduler=dict(job_state='R'), worker=dict(state='EXITED'), session_end=dict(epoch=1))
            manager.state['allocations'][token].update(status='F', harvested=True)
            call = gateway.call

            def snapshot(value):
                if value['action'] == 'exchange':
                    return dict(allocations=[closed], requests={})
                return call(value)

            with patch.object(executor, '_owned_attempts', return_value=[owner]), \
                    patch.object(manager, '_demand', return_value=([], [], 0, 0)), \
                    patch.object(gateway, 'call', side_effect=snapshot):
                manager.cycle()
            replay = gateway.calls[-1]['spec']
            self.assertEqual(replay['reason']['replay_pilot'], dict(token='held', job_id='41.prism'))
            self.assertNotEqual(replay['token'], token)
            self.assertEqual(executor.rows[owner['job_id']], original)
            self.assertEqual(executor.calls, [])
            self.assertIsNone(manager._assigned_pilot_replay('pbs-0', [dict(owner, state='UNKNOWN')]))

    def test_sized_allocation_briefly_keeps_only_its_checked_queue_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            config = manager.configured['pbs-0']
            manager.settings['resources']['mem'] = '5gb'
            manager.settings['allocation_sizing'] = dict(
                per_task_resources=True, q3_startup_memory_bytes_per_task=2*P.GIB)
            jobs = [dict(route=dict(kind='artificial'), request_id=f'job-{i}') for i in range(3)]
            for index, job in enumerate(jobs):
                executor.rows[job['request_id']] = dict(state='QUEUED',
                    eligibility=dict(eligible_host_ids=['pbs-0'] if index != 1 else []))
            spec = manager._submit('pbs-0', config, dict(shortage=3), jobs)
            record = manager.state['allocations'][spec['token']]
            self.assertEqual(spec['reason']['requested_task_capacity'], 2)
            self.assertEqual(manager.startup_jobs(), {'job-0'})
            self.assertFalse(manager.startup_jobs(except_host='pbs-0'))
            record.update(status='HISTORY_MISSING_RECONCILE_SPOOL',observed=dict(job_id='test.PBS'))
            self.assertEqual(manager.startup_jobs(), {'job-0'})
            record['worker_storage_hold'] = 'STORAGE_USAGE_UNAVAILABLE'
            self.assertEqual(manager.startup_jobs(), {'job-0'})
            record['worker_storage_hold'] = 'STORAGE_LIMIT_REACHED'
            self.assertFalse(manager.startup_jobs())
            record.pop('worker_storage_hold')
            record['submitted_clock']['monotonic'] -= spec['reason']['startup_wait_seconds'] + 1
            self.assertFalse(manager.startup_jobs())
            self.assertTrue(all(r['state']=='QUEUED' for r in executor.rows.values()))

    def test_storage_hold_prevents_new_pbs_allocation_and_reopens(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(P, 'SAMPLE_SECONDS', 0):
            manager, executor, gateway = self.manager(directory)
            worker = manager.configured['pbs-0']
            worker['profile']['storage_limit'] = dict(limit_bytes=50*2**30)
            ready = dict(route=dict(kind='artificial'))
            manager._demand = lambda: ([(['pbs-0'],ready)], [], 1, 0)
            gateway.storage_report = lambda p: dict(limit_bytes=p['limit_bytes'], used_bytes=p['limit_bytes'],
                                                  accepting_new_jobs=False, reason='STORAGE_LIMIT_REACHED')
            manager.cycle()
            self.assertEqual(manager.state['allocations'], {})
            self.assertFalse(any(call['action']=='ensure' for call in gateway.calls))
            gateway.storage_report = lambda p: dict(limit_bytes=p['limit_bytes'], used_bytes=0,
                accepting_new_jobs=False, reason='FILESYSTEM_SPACE_LOW',
                filesystem_free_bytes=2**30, filesystem_min_free_bytes=5*2**30)
            manager.cycle()
            self.assertEqual(manager.state['allocations'], {})
            self.assertEqual(manager.state['storage']['pbs-0']['reason'], 'FILESYSTEM_SPACE_LOW')
            gateway.storage_report = lambda p: dict(limit_bytes=p['limit_bytes'], used_bytes=0,
                                                  accepting_new_jobs=True, reason=None)
            manager.cycle()
            self.assertEqual(len(manager.state['allocations']), 1)
            self.assertEqual(sum(call['action']=='ensure' for call in gateway.calls), 1)

    def test_controller_transport_detachment_never_sends_pilot_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            manager.hold_node = manager.retain_between_batches = True
            token = self.submit(manager)
            row = manager.state['allocations'][token]
            row.update(status='R')
            allocation = dict(token=token, status='R', job_id='SAVED.PBS', host_id='pbs-0',
                              worker=dict(accepting=True), pilot_token='shared')
            gateway.call = Mock(return_value=dict(allocations=[allocation], requests={}))
            manager.detached = True
            manager.stop.set()
            executor.state.closed = True
            manager.cycle()
            self.assertFalse(manager.controls.get(token, {}).get('stop', False))

    def test_full_shared_pilot_drains_only_its_session_after_owned_work(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            manager.hold_node = True
            manager._demand = lambda: ([], [], 0, 0)
            token = self.submit(manager)
            row = manager.state['allocations'][token]
            row.update(status='R', worker_storage_hold='STORAGE_LIMIT_REACHED')
            allocation = dict(token=token, status='R', job_id='SAVED.PBS', host_id='pbs-0',
                              worker=dict(accepting=False), pilot_token='shared')
            gateway.call = Mock(return_value=dict(allocations=[allocation], requests={}))
            executor._owned_attempts = lambda: [dict(host_id='pbs-0')]
            manager.cycle()
            self.assertFalse(manager.controls[token]['stop'])
            executor._owned_attempts = lambda: []
            manager.cycle()
            self.assertEqual(manager.controls[token], dict(stop=True, pause=False, release_when_idle=False))
            self.assertTrue(P.read(manager.path)['allocations'][token]['storage_maintenance_requested'])
            self.assertTrue(all(call.args[0]['action']=='exchange' for call in gateway.call.call_args_list))

    def test_ready_q3_sizing_uses_fixed_cpu_footprint_and_pins_the_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = FakeExecutor(directory)
            gateway = FakeGateway()
            config = settings()
            config['resources'].update(ncpus=96, mem='350gb')
            config['workers'][0]['profile']['dp_workers'] = 2
            config.update(task_seconds={'q3': 60},
                          allocation_sizing={'q3_startup_memory_bytes_per_task': 8 * 1024**3})
            manager = P.PbsManager(executor, config, gateway=gateway)
            try:
                request = dict(route={'kind': 'pauli'})
                single = manager._submit('pbs-0', config['workers'][0], {'shortage': 1}, [request])
                batch = manager._submit('pbs-0', config['workers'][0], {'shortage': 3}, [request] * 3)
                bounded = manager._submit('pbs-0', config['workers'][0], {'shortage': 50}, [request] * 50)
                for spec, cpus, slots in ((single, 2, 1), (batch, 6, 3), (bounded, 96, 43)):
                    self.assertEqual(spec['resources']['ncpus'], cpus)
                    self.assertEqual(spec['resources']['mem'], '350gb')
                    self.assertEqual(spec['profile']['slot_policy']['initial_slots'], slots)
                    self.assertEqual(P.read(manager.path)['allocations'][spec['token']]['spec'], spec)
                self.assertEqual(executor.calls, [])
            finally:
                manager.close()

    def test_mixed_tasks_pack_96_cpu_without_charging_q3_at_compile_weight(self):
        config = settings()
        config['resources'].update(ncpus=96, mem='350gb')
        profile = config['workers'][0]['profile']
        profile['dp_workers'] = 2
        profile['preparation_profiles'] = [
            dict(stage='compile', provider='compile', source_commit='pinned',
                 profile=dict(profile, task_cpu_count=8))]
        config.update(allocation_sizing=dict(per_task_resources=True,
                      q3_startup_memory_bytes_per_task=7*P.GIB),
                      preparation_task_seconds=[
                          dict(stage='compile', provider='compile', source='pinned',
                               seconds=60, memory_bytes=20*P.GIB)])
        q3 = dict(route=dict(kind='artificial'))
        compile = dict(route=dict(kind='preparation'),
                       preparation=dict(stage='compile', provider='compile', source={'commit':'pinned'}))
        self.assertEqual(P.packed_resources([q3]*4+[compile], profile, config, 'pbs-0'), (16, 49, 5))
        self.assertEqual(P.packed_resources([q3]*44+[compile], profile, config, 'pbs-0'), (96, 329, 45))
        config['allocation_sizing']['q3_startup_memory_bytes_per_task'] = 8*P.GIB
        self.assertEqual(P.packed_resources([q3]*48, profile, config, 'pbs-0'), (86, 345, 43))
        config['preparation_task_seconds'][0].pop('memory_bytes')
        self.assertEqual(P.packed_resources([compile, q3], profile, config, 'pbs-0'), (2, 9, 1))
        self.assertEqual(P.packed_resources([compile], profile, config, 'pbs-0'), (8, 350, 1))

    def test_sizing_only_counts_unmatched_ready_compatible_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            executor.hosts = {'local': {}}
            executor._host_admission_policy = lambda: ({}, None)
            executor._host_admission = lambda host, *_: dict(accepting=host == 'pbs-0')
            executor._host_supports_stage = lambda host, request: True
            executor._offer_free = lambda host, *_: 1
            executor._match_ready_jobs = lambda tasks, slots: {tasks[0].job.name}
            executor._ready_queue = lambda: [SimpleNamespace(job=Path(name)) for name in executor.rows]
            executor.store.artificial = lambda request: request['route']['kind'] == 'artificial'
            for name in ('covered', 'ready', 'missing_estimate'):
                executor.rows[name] = dict(state='QUEUED', request=dict(id=name,
                    route=dict(kind='pauli' if name == 'missing_estimate' else 'artificial')),
                    eligibility=dict(eligible_host_ids=['local', 'pbs-0']))
            ready, pending, queued, covered = manager._demand()
            self.assertEqual([(hosts, req['id']) for hosts, req in ready], [(['pbs-0'], 'ready')])
            self.assertEqual((pending, queued, covered), ([], 2, 1))
            manager.close()

    def test_measured_memory_scales_request_and_fences_other_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = FakeExecutor(directory)
            config = settings()
            config['resources'].update(ncpus=96, mem='350gb')
            config['workers'][0]['profile']['dp_workers'] = 2
            gib = 1024**3
            config.update(allocation_sizing={'q3_startup_memory_bytes_per_task': 8*gib},
                          preparation_task_seconds=[
                              dict(stage='signal', provider='signal', source='pinned', seconds=12, memory_bytes=4*gib),
                              dict(stage='compile', provider='compile', source='pinned', seconds=12)])
            manager = P.PbsManager(executor, config, gateway=FakeGateway())
            try:
                signal = dict(preparation=dict(stage='signal', provider='signal', source={'commit':'pinned'}))
                compile = dict(preparation=dict(stage='compile', provider='compile', source={'commit':'pinned'}))
                spec = manager._submit('pbs-0', config['workers'][0], {}, [signal]*2)
                self.assertEqual((spec['resources']['ncpus'],spec['resources']['mem']), (4,'9gb'))
                report = dict(pbs=dict(accepting=True, remaining_seconds=100, sized_task_memory_bytes=4*gib))
                self.assertTrue(P.fits_request(signal, report, config))
                self.assertFalse(P.fits_request(compile, report, config))
                full = manager._submit('pbs-0', config['workers'][0], {}, [compile])
                self.assertEqual(full['resources']['mem'], '350gb')
            finally:
                manager.close()

    def test_only_idle_undersized_allocation_drains_for_later_demand(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            token = self.submit(manager)
            manager.settings['allocation_sizing'] = {'q3_startup_memory_bytes_per_task': 1024**3}
            manager.settings['resources']['ncpus'] = 96
            ready = dict(route={'kind': 'artificial'})
            manager._demand = lambda: ([(['pbs-0'], ready)] * 4, [], 4, 0)
            observed = dict(token=token, status='R', host_id='pbs-0', job_id='artificial',
                            worker={'accepting': True})
            original_call = gateway.call
            gateway.call = lambda message: (dict(allocations=[observed], requests={})
                                            if message['action'] == 'exchange' else original_call(message))
            executor._owned_attempts = lambda: [dict(host_id='pbs-0')]
            manager.cycle()
            self.assertFalse(manager.controls[token]['stop'])
            executor._owned_attempts = lambda: []
            manager.cycle()
            self.assertFalse(manager.controls[token]['stop'])
            self.assertNotIn('resize_requested', manager.state['allocations'][token])
            self.assertTrue(manager.accepting_allocation('pbs-0', 'artificial'))
            self.assertFalse(manager.accepting_allocation('pbs-0', 'previous-allocation'))
            manager.state['allocations'][token]['first_accepting_clock']['monotonic'] -= 60
            manager.cycle()
            self.assertFalse(manager.controls[token]['stop'])
            for request_id, slots in [('blocked-storage', 0), ('eligible-offer', 1)]:
                manager._dispatch(token, dict(endpoint='/v1/poll', request_id=request_id,
                    payload=dict(host_id='pbs-0', slots=slots, capacity=1,
                                 accepting_new_jobs=bool(slots))))
            first_offer = manager.state['allocations'][token]['first_usable_offer_clock']
            manager.state['allocations'][token]['first_usable_offer_clock'] = dict(
                first_offer, monotonic=first_offer['monotonic']-60)
            manager.cycle()
            self.assertTrue(manager.controls[token]['stop'])
            self.assertTrue(manager.state['allocations'][token]['resize_requested'])
            self.assertEqual(len(manager.state['allocations']), 1)
            response = manager._dispatch(token, dict(endpoint='/v1/poll', request_id='late',
                payload=dict(host_id='pbs-0', slots=1, capacity=1)))
            self.assertTrue(response['ok'])
            self.assertEqual(executor.calls[-1][1]['slots'], 0)
            observed = dict(observed, status='F')
            gateway.harvest['host_id'] = 'pbs-0'
            manager.cycle()
            self.assertEqual(len(manager.state['allocations']), 2)
            replacement = next(row for other, row in manager.state['allocations'].items() if other != token)
            self.assertEqual(replacement['spec']['resources']['ncpus'], 4)

    def test_recent_required_completion_keeps_live_allocation_for_successor(self):
        from trottertracks.rpe_q3_store import JobStore
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            token = self.submit(manager)
            executor.store = JobStore(Path(directory)/'QUEUE.sqlite3')
            manager._demand = lambda: ([], [], 0, 0)
            observed = dict(token=token, host_id='pbs-0', job_id='42', status='R',
                            worker=dict(accepting=True))
            gateway.call = lambda message: dict(allocations=[observed], requests={})
            executor._owned_attempts = lambda: [dict(host_id='pbs-0',state='ASSIGNED')]
            manager.cycle()
            self.assertFalse(manager.controls[token]['release_when_idle'])
            executor._owned_attempts = lambda: []
            manager.cycle()
            self.assertTrue(manager.controls[token]['release_when_idle'])
            request = dict(preparation={})
            with executor.store._connection(write=True) as db:
                db.execute(
                    "INSERT INTO jobs(job_id, method, request, request_sha256, contract, state, "
                    "enqueued_epoch, updated_epoch, completed_epoch, detail) "
                    "VALUES ('prep', 'PREPARATION:compile', ?, 'fixed', '{}', 'ACCEPTED', 0, 0, ?, ?)",
                    (json.dumps(request), P.time.time(), '{"outcome_status":"PREPARATION_COMPLETE"}'))
            manager.cycle()
            self.assertFalse(manager.controls[token]['release_when_idle'])
            with executor.store._connection(write=True) as db:
                db.execute("UPDATE jobs SET completed_epoch=0")
            manager.cycle()
            self.assertTrue(manager.controls[token]['release_when_idle'])

    def test_full_node_keeps_the_same_allocation_until_run_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            config = settings()
            config['resources'].update(ncpus=96, mem='350gb')
            config['allocation_sizing'] = dict(per_task_resources=True, q3_startup_memory_bytes_per_task=8*P.GIB)
            executor = FakeExecutor(directory)
            executor.distributed = {'pbs_hold_node': True}
            gateway = FakeGateway()
            first = P.PbsManager(executor, config, gateway=gateway)
            self.addCleanup(first.close)
            first.retain_between_batches = True
            request = dict(route=dict(kind='artificial'))
            spec = first._submit('pbs-0', first.configured['pbs-0'], {}, [request])
            self.assertEqual(spec['resources'], config['resources'])
            self.assertEqual(spec['profile']['slot_policy']['initial_slots'], 96)
            self.assertNotIn('startup_jobs', spec['reason'])
            token = spec['token']
            observed = dict(token=token, status='Q', host_id='pbs-0', job_id='test.PBS', worker={})
            original = gateway.call
            gateway.call = lambda message: (original(message) or dict()) | (
                dict(allocations=[observed], requests={}) if message['action'] == 'exchange' else {})
            first._demand = lambda: ([], [], 0, 0)
            first.cycle()
            self.assertEqual(first.controls[token], dict(stop=False, pause=False, release_when_idle=False))
            self.assertNotIn('queued_resize_since', first.state['allocations'][token])
            observed.update(status='R', worker={'accepting': True})
            first.cycle()
            self.assertFalse(first.controls[token]['release_when_idle'])
            executor.state.closed = True
            first.cycle()
            first.close(retain_allocation=True)
            self.assertTrue(first.detached)
            executor.state.closed = False
            second = P.PbsManager(executor, config, gateway=gateway)
            self.addCleanup(second.close)
            second._demand = lambda: ([(['pbs-0'], request)], [], 1, 0)
            second.cycle()
            self.assertTrue(second.accepting_allocation('pbs-0', 'test.PBS'))
            self.assertEqual(list(second.state['allocations']), [token])
            self.assertFalse(any(call['action'] == 'delete' for call in gateway.calls))
            second.close()
            self.assertEqual(gateway.calls[-1]['controls'][token], dict(stop=True, release_when_idle=True))
            first.detached = False  # the successor already performed final shutdown

    def test_unused_queued_allocation_resizes_for_persistent_checked_demand(self):
        for initial, remaining in ((4, 0), (4, 2), (2, 8)):
            with self.subTest(initial=initial, remaining=remaining), tempfile.TemporaryDirectory() as directory:
                manager, executor, gateway = self.manager(directory)
                manager.settings['resources'].update(ncpus=8, mem='9gb')
                manager.settings['allocation_sizing'] = dict(per_task_resources=True,
                    q3_startup_memory_bytes_per_task=P.GIB)
                request = dict(route=dict(kind='artificial'))
                spec = manager._submit('pbs-0', manager.configured['pbs-0'], {}, [request]*initial)
                token = spec['token']
                observed = dict(token=token, status='Q', host_id='pbs-0', job_id='test.PBS', worker={})
                original = gateway.call
                gateway.call = lambda message: (dict(allocations=[observed], requests={})
                                                if message['action'] == 'exchange' else original(message))
                gateway.harvest['host_id'] = 'pbs-0'
                demand = [(['pbs-0'], request)] * remaining
                manager._demand = lambda: ([], demand, remaining, 0)
                manager._recent_preparation_completion = lambda _: False
                record = manager.state['allocations'][token]
                if initial < remaining:
                    manager.cycle()
                    self.assertNotIn('queued_resize_since', record)
                    executor._owned_attempts = lambda: [{'host_id': 'pbs-0'}]
                    manager._demand = lambda: (demand, [], remaining, 0)
                    manager.cycle()
                    self.assertNotIn('queued_resize_since', record)
                    executor._owned_attempts = lambda: []
                manager._demand = lambda: (demand, [], remaining, 0)
                manager.cycle()
                self.assertFalse(any(call['action'] == 'delete' for call in gateway.calls))
                record['queued_resize_since']['monotonic'] -= manager.readiness_wait_seconds() + 1
                manager.cycle()
                self.assertTrue(manager.controls[token]['stop'])
                self.assertTrue(P.read(manager.path)['allocations'][token]['resize_requested'])
                self.assertEqual([call['token'] for call in gateway.calls if call['action'] == 'delete'], [token])
                observed = dict(observed, status='R', worker={'accepting': True})
                manager.cycle()
                self.assertFalse(manager.accepting_allocation('pbs-0', 'test.PBS'))
                self.assertEqual(len(manager.state['allocations']), 1)
                observed = dict(observed, status='F')
                manager.cycle()
                replacement = [call['spec'] for call in gateway.calls if call['action'] == 'ensure'
                               and call['spec']['token'] != token]
                self.assertEqual([spec['resources']['ncpus'] for spec in replacement], [remaining] if remaining else [])

    def test_preparation_reserved_with_local_readiness_can_wake_pbs(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            manager.settings['preparation_task_seconds'] = [
                dict(stage='signal', provider='DF_SIGNALS', source='fixed', seconds=120)]
            request = dict(route=dict(kind='preparation'),
                           preparation=dict(stage='signal', provider='DF_SIGNALS', source=dict(commit='fixed')))
            executor.rows['signal'] = dict(request=request, state='QUEUED', eligibility=dict(eligible_host_ids=['local']))
            executor.preparation_pending['signal'] = dict(request=request, failures={})
            executor.hosts = dict(local={})
            executor._host_admission_policy = lambda: ({}, None)
            executor._host_admission = lambda *_: dict(accepting=True)
            executor._host_supports_stage = lambda *_: True
            executor._ready_queue = lambda: [SimpleNamespace(job=Path('signal'))]
            executor._offer_free = lambda *_: 0
            executor._match_ready_jobs = lambda tasks, slots: {task.job.name for task in tasks} if slots else set()
            executor.store.artificial = lambda _: False

            self.assertEqual(manager._demand(), ([], [(['pbs-0'],request)], 1, 0))
            executor._offer_free = lambda *_: 1
            self.assertEqual(manager._demand(), ([], [], 1, 1))
            executor.preparation_pending['signal']['failures']['pbs-0'] = 'runtime mismatch'
            executor.preparation_pending['signal']['failure_epochs'] = {'pbs-0': time.time()}
            self.assertEqual(manager._demand(), ([], [], 0, 0))
            executor._offer_free = lambda *_: 0
            # Old failures allow a fresh preflight, not an eligible assignment.
            executor.preparation_pending['signal']['failure_epochs']['pbs-0'] -= 61
            self.assertEqual(manager._demand(), ([], [(['pbs-0'], request)], 1, 0))
            executor._ready_queue = lambda: []  # not yet restored in memory
            self.assertEqual(manager._demand(), ([], [(['pbs-0'], request)], 1, 0))
            executor.rows['signal']['state'] = 'RUNNING'
            self.assertEqual(manager._demand(), ([], [], 0, 0))

    def test_mixed_methods_request_capacity_for_all_independent_preparation_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            manager.settings['resources'].update(ncpus=96, mem='350gb')
            manager.settings['allocation_sizing'] = {'q3_startup_memory_bytes_per_task': 8 * 1024**3}
            manager.configured['pbs-0']['profile']['dp_workers'] = 2
            supported = ('PF_SIGNALS', 'DF_SIGNALS', 'PAULI_SIGNALS')
            manager.settings['preparation_task_seconds'] = [
                dict(stage='signal', provider=provider, source='fixed',
                     seconds=120, memory_bytes=2 * 1024**3)
                for provider in supported]
            executor._host_admission_policy = lambda: ({}, None)
            executor._host_admission = lambda *_: dict(accepting=True)
            executor._host_supports_stage = lambda *_: True
            executor._ready_queue = lambda: []
            executor._match_ready_jobs = lambda *_: set()
            executor.store.artificial = lambda _: False
            for provider in (*supported, 'UNSUPPORTED'):
                request = dict(route=dict(kind='preparation'), preparation=dict(
                    stage='signal', provider=provider, source=dict(commit='fixed')))
                executor.preparation_pending[provider] = dict(request=request, failures={})

            manager.cycle()

            spec, = [call['spec'] for call in gateway.calls if call['action'] == 'ensure']
            self.assertEqual(spec['resources']['ncpus'], 6)
            self.assertEqual(spec['resources']['mem'], '7gb')
            self.assertEqual(spec['profile']['slot_policy']['initial_slots'], 3)
            self.assertEqual(executor.calls, [])

    def test_pbs_sizes_unreserved_readiness_checks_to_remaining_stage_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            manager.settings['preparation_task_seconds'] = [
                dict(stage='signal', provider='DF_SIGNALS', source='fixed', seconds=120)]
            request = dict(route=dict(kind='preparation'), preparation=dict(
                stage='signal', provider='DF_SIGNALS', source=dict(commit='fixed')))
            executor._host_admission_policy = lambda: ({}, None)
            executor._host_admission = lambda *_: dict(accepting=True)
            executor._host_supports_stage = lambda *_: True
            executor._ready_queue = lambda: []
            executor._match_ready_jobs = lambda *_: set()
            executor.store.artificial = lambda _: False
            executor.preparation_pending.update({
                job: dict(request=request, budget=4, approved=True) for job in ('one', 'two', 'three')})
            executor.store.preparation_reservations = lambda: {'signal': 4}
            self.assertEqual(manager._demand(), ([], [], 0, 0))
            executor.store.preparation_reservations = lambda: {'signal': 3}
            self.assertEqual(manager._demand(), ([], [(['pbs-0'], request)], 1, 0))

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

    def test_retained_pbs_worker_ignores_old_replays_and_old_offer(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            old = self.submit(manager)
            new = manager._submit('pbs-0', settings()['workers'][0], dict(ready_tasks=3))['token']
            owner = ['old-job', 'old-attempt', 'old-hash']
            manager.state['allocations'][new]['spec']['reason']['retained_replacement'] = dict(old_owners=[owner])
            manager.allocations = [dict(token=old, status='R', retained_replacement='new-pilot'),
                dict(token=new, status='R', worker=dict(accepting=True))]
            old_request = dict(request_id='old-poll', endpoint='/v1/poll',
                               payload=dict(host_id='pbs-0', slots=0, capacity=0))
            self.assertEqual(manager._dispatch(old, old_request)['result']['assignments'], [])
            self.assertEqual(executor.calls, [])
            new_request = dict(request_id='new-poll', endpoint='/v1/poll',
                payload=dict(host_id='pbs-0', slots=1, capacity=1, durable_assignments=[['new','a','hash']]))
            self.assertTrue(manager._dispatch(new, new_request)['ok'])
            self.assertEqual(executor.calls[-1][1]['durable_assignments'], [['new','a','hash'], owner])

    def test_retained_pbs_replacement_reuses_demand_but_connects_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            old = self.submit(manager)
            manager.allocations = [dict(token=old, status='R', host_id='pbs-0',
                                        pilot_token='old-pilot', retained_replacement='new-pilot')]
            request = dict(request_id='queued')
            with patch.object(manager, '_demand', return_value=([(['pbs-0'], request)], [], 1, 0)), \
                    patch.object(executor.store, '_connection', create=True) as connection:
                connection.return_value.__enter__.return_value.execute.return_value = [('old', 'attempt', 'hash')]
                manager._connect_retained_replacement()
                manager._connect_retained_replacement()
            self.assertEqual(len(gateway.calls), 2)
            self.assertEqual(gateway.calls[-1]['spec']['reason']['retained_replacement'],
                             dict(pilot_token='new-pilot', predecessor='old-pilot',
                                  old_owners=[['old', 'attempt', 'hash']]))

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
            executor.rows['original-task'] = dict(state='RUNNING', attempt_id='original-attempt', host_id='pbs-0')
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

    def test_changed_collector_rechecks_only_unresolved_prelaunch_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, gateway = self.manager(directory)
            token = self.submit(manager)
            manager.settings['gateway_sha256'] = 'new-collector'
            manager.state['allocations'][token]['harvested'] = True
            manager.allocations = [dict(token=token, host_id='pbs-0', status='F')]
            manager.harvest_gateway = gateway
            gateway.harvest = dict(host_id='pbs-0', attempts=[], incomplete=[], artifact_hashes=[])
            with patch.object(executor.store, '_connection', create=True) as connection:
                connection.return_value.__enter__.return_value.execute.return_value = [
                    dict(job_id='failed-dry', host_id='pbs-0', attempt_id='attempt', needs_prelaunch_proof=True),
                    dict(job_id='uncertain', host_id='pbs-0', attempt_id='other', needs_prelaunch_proof=False)]
                try:
                    manager._advance_harvest()
                    manager.harvest_pending[-1].result(timeout=3)
                    manager._advance_harvest()
                    manager._advance_harvest()
                    calls = [c for c in gateway.calls if c['action'] == 'harvest']
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]['owners'], [dict(job_id='failed-dry', attempt_id='attempt')])
                    self.assertEqual(manager.state['allocations'][token]['harvest_gateway_sha256'], 'new-collector')
                finally:
                    manager.close()

    def test_admission_exit_releases_capacity_without_completing_or_retrying(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, executor, _ = self.manager(directory)
            token = self.submit(manager)
            assignment = dict(job_id='old', attempt_id='attempt', request_sha256='fixed')
            executor.rows['old'] = dict(assignment, state='ASSIGNED', host_id='pbs-0', started_epoch=None)
            detail = dict(reason='PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH', scientific_execution_started=False,
                          automatic_retry=False, resource_released=True, pbs_admission_exit=dict(assignment, token=token))
            data = dict(host_id='pbs-0', attempts=[], incomplete=[dict(assignment=assignment,
                        state=dict(state='UNKNOWN', detail=detail))])
            manager._harvest(dict(token=token, host_id='pbs-0', status='F'), fetched=(data, {}),
                             owners={('old', 'attempt')})
            self.assertEqual(executor.calls, [('/v1/state', dict(host_id='pbs-0', job_id='old',
                attempt_id='attempt', state='UNKNOWN', detail=detail))])


class GatewayTests(unittest.TestCase):
    def test_exchange_does_not_replay_bulk_rpc_from_ended_pilot_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for token in ('closed', 'live'):
                session = root/'allocations'/token
                (session/'requests').mkdir(parents=True)
                P.atomic(session/'requests'/'complete-test.json', {'request_id':'complete-test'})
            P.atomic(root/'allocations'/'closed'/'SESSION_END.json', {'reason':'WORKER_EXITED'})
            result = G.exchange(root, {'request_mode':'completion', 'limit':2}, snapshotter=lambda *_: [])
            self.assertEqual(set(result['requests']), {'live'})
            self.assertTrue((root/'allocations'/'closed'/'requests'/'complete-test.json').is_file())

    def test_failed_ssh_login_is_shared_between_gateways(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(P.tempfile, 'gettempdir', return_value=directory), \
             patch.object(P.time, 'time', return_value=100) as clock, \
             patch.object(P.subprocess, 'check_output', return_value='hostname prism\n'), \
             patch.object(P.subprocess, 'run') as run:
            def connect(args, **kwargs):
                if 'ControlMaster=yes' in args:
                    self.assertIn('-NTf', args)
                    self.assertFalse(any(option.startswith('-') and 'M' in option for option in args))
                    raise subprocess.CalledProcessError(255, args)
                return SimpleNamespace(returncode=255)
            run.side_effect = connect
            for _ in range(2):
                with self.assertRaises(subprocess.CalledProcessError):
                    P.SharedSSH('prism').command()
                with self.assertRaisesRegex(OSError, 'reconnect backoff'):
                    P.SharedSSH('prism').command()
                clock.return_value += 6
            with self.assertRaisesRegex(OSError, 'reconnect backoff'):
                P.SharedSSH('prism').command()
        self.assertEqual(sum('ControlMaster=yes' in call.args[0] for call in run.call_args_list), 2)

    def test_successful_master_can_be_reused_and_paused(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(P.tempfile, 'gettempdir', return_value=directory), \
             patch.object(P.subprocess, 'check_output', return_value='hostname prism\n') as config, \
             patch.object(P.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            cmd = P.SharedSSH('prism').command()
            self.assertIn('ProxyCommand=/bin/false', cmd)
            self.assertIn('-O', run.call_args.args[0])
            config.return_value = 'proxycommand /usr/bin/false\n'
            with self.assertRaisesRegex(OSError, 'access is paused'):
                P.SharedSSH('prism').command()
            self.assertEqual(run.call_count, 1)

    def test_scheduler_cache_keeps_fast_polls_and_allows_forced_refresh(self):
        with patch.object(G.time, 'monotonic', return_value=0) as clock, \
             patch.object(G, 'qstat', return_value={'job': {'job_state': 'R'}}) as query:
            scheduler = G.SchedulerCache()
            for tick in range(30):
                clock.return_value = tick
                self.assertIn('job', scheduler.query(['job']))
            self.assertEqual(query.call_count, 2)
            scheduler.query(['job'], force=True)
            scheduler.query([])
            scheduler.query([])
            self.assertEqual(query.call_count, 4)
            self.assertEqual(query.call_args.args, ())

    def test_bulk_spool_returns_only_sizes_until_read_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'allocations'/'owned'
            path = directory/'requests'/'complete-one.json'
            exact = dict(request_id='complete-one', endpoint='/v1/complete',
                        payload={'files': {'OUTCOME.json': 'x'*(2*G.BULK_CHUNK_BYTES+17)}})
            G.atomic(path, exact)
            data = G.exchange(root, {'request_mode': 'completion', 'bounded_results': True, 'limit': 4})
            receipt, = data['requests']['owned']
            self.assertEqual(receipt, {'request_id': 'complete-one', 'spool_bytes': path.stat().st_size})
            gateway = SimpleNamespace(call=lambda value: G.handle(root, value))
            saved = P.PbsManager._read_bulk(gateway, 'owned', 'spool', 'complete-one',
                                            size=receipt['spool_bytes'])
            self.assertEqual(saved, path.read_bytes())
            self.assertEqual(json.loads(saved), exact)

    def test_explicit_gateway_runtime_survives_exchange_to_harvest_reconnect(self):
        class Process:
            def __init__(self):
                self.stdin = io.StringIO()
                self.stdout = io.StringIO('{"ok":true,"result":{}}\n')
                self.alive = True
            def poll(self): return None if self.alive else 0
            def terminate(self): self.alive = False
            def wait(self, timeout=None): return 0
        config = dict(ssh_alias='existing-alias', gateway_python='/private/fixed/python',
                      gateway_script='/fixed/rpe_pbs_gateway.py', remote_root='/owned/spool')
        gateway = P.Gateway(config)
        with patch.object(P.subprocess, 'Popen', side_effect=lambda *a, **k: Process()) as start, \
             patch.object(gateway.ssh, 'command', return_value=['ssh', 'existing-alias']):
            gateway.call(dict(action='exchange'))
            gateway.close()
            gateway.call(dict(action='harvest', token='saved-original'))
            gateway.close()
        self.assertEqual(start.call_count, 2)
        for call in start.call_args_list:
            remote = shlex.split(call.args[0][-1])
            self.assertEqual(remote[0], config['gateway_python'])
            self.assertEqual(remote[2], config['gateway_script'])

    def test_json_query_avoids_pbs20_user_table_and_empty_run_does_not_query(self):
        process = SimpleNamespace(returncode=0, stdout='{"Jobs":{}}', stderr='')
        with patch.object(G.subprocess, 'run', return_value=process) as run:
            self.assertEqual(G.qstat(), {})
            self.assertNotIn('-u', run.call_args[0][0])
        with tempfile.TemporaryDirectory() as directory, patch.object(G, 'qstat') as query:
            self.assertEqual(G.snapshot(Path(directory)), [])
            query.assert_not_called()

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

    def test_batches_known_job_queries_and_keeps_finished_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = {}
            for index, state in enumerate(('F', 'R')):
                token = 'artificial-{}'.format(index)
                allocation = root/'allocations'/token
                job_id = '{}.prism'.format(index)
                G.atomic(allocation/'SPEC.json', dict(token=token, profile=dict(host_id='pbs-0')))
                G.atomic(allocation/'JOB_ID.json', dict(job_id=job_id))
                jobs[job_id] = dict(Job_Name='RPE'+token[:12], job_state=state)
            cache = {}
            with patch.object(G, 'qstat', side_effect=[jobs, {'1.prism': jobs['1.prism']}]) as query:
                first = G.snapshot(root, cache)
                second = G.snapshot(root, cache)
            self.assertEqual(first, second)
            self.assertEqual([call.args for call in query.call_args_list], [('0.prism', '1.prism'), ('1.prism',)])

    def test_missing_history_in_batch_preserves_other_job_results(self):
        details = dict(Job_Owner=getpass.getuser()+'@prism', job_state='R')
        reply = SimpleNamespace(returncode=1, stdout=json.dumps(dict(Jobs={'1.prism': details})),
                                stderr='qstat: Unknown Job Id 0.prism\n')
        with patch.object(G.subprocess, 'run', return_value=reply):
            self.assertEqual(G.qstat('0.prism','1.prism'), {'1.prism': details})

    def test_harvest_reads_only_the_requested_original_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = root/'allocation'
            work = root/'worker'
            G.atomic(allocation/'SPEC.json', dict(profile=dict(host_id='pbs-0')))
            G.atomic(allocation/'PROFILE.json', dict(work_dir=str(work)))
            artifact = b'saved dependency'
            sha = hashlib.sha256(artifact).hexdigest()
            path = work/'artifacts'/sha[:2]/sha
            path.parent.mkdir(parents=True)
            path.write_bytes(artifact)
            for job in ('already-transferred', 'needs-recovery'):
                attempt = work/job/'attempt'
                G.atomic(attempt/'ASSIGNMENT.json', dict(job_id=job, attempt_id='attempt'))
                G.atomic(attempt/'PROCESS_RETURN.json', dict(returncode=0))
                G.atomic(attempt/'OUTCOME.json', dict(status='ARTIFICIAL', numerator=10**7167,
                                                     artifacts={'dependency': sha}))
            # Traversing this unrelated directory would reject its symlink.
            (work/'already-transferred'/'attempt'/'unrelated').symlink_to(root)
            recovered = G.harvest(allocation, owners=[dict(job_id='needs-recovery', attempt_id='attempt')])
            self.assertEqual([item['assignment']['job_id'] for item in recovered['attempts']], ['needs-recovery'])
            saved = work/'needs-recovery'/'attempt'/'OUTCOME.json'
            self.assertEqual(base64.b64decode(recovered['attempts'][0]['files']['OUTCOME.json']), saved.read_bytes())
            manifest = G.harvest(allocation, owners=[dict(job_id='needs-recovery', attempt_id='attempt')],
                                 manifest_only=True)
            self.assertEqual(manifest['attempts'][0]['files']['OUTCOME.json'], saved.stat().st_size)
            gateway = SimpleNamespace(call=lambda value: G.read_bulk_chunk(allocation, value))
            raw = P.PbsManager._read_bulk(gateway, 'allocation', 'attempt', 'OUTCOME.json',
                                         job_id='needs-recovery', attempt_id='attempt')
            self.assertEqual(raw, saved.read_bytes())
            end_to_end = SimpleNamespace(call=lambda value: (
                G.harvest(allocation, owners=value['owners'], manifest_only=value['manifest_only'])
                if value['action'] == 'harvest' else G.read_bulk_chunk(allocation, value)))
            manager = object.__new__(P.PbsManager)
            manager.bounded_results = True
            manager.executor = SimpleNamespace(artifact_cache=ArtifactCache(root/'central-cache'))
            data, blobs = manager._fetch_harvest(end_to_end, 'allocation', {('needs-recovery', 'attempt')})
            manager.executor.artifact_cache.put(blobs)
            self.assertEqual(manager.executor.artifact_cache.path(sha).read_bytes(), artifact)
            received = base64.b64decode(data['attempts'][0]['files']['OUTCOME.json'])
            self.assertEqual(received, saved.read_bytes())
            self.assertEqual(json.loads(received)['numerator'], 10**7167)

            later = work/'new-session'/'attempt'
            G.atomic(later/'ASSIGNMENT.json', dict(job_id='new-session', attempt_id='attempt'))
            G.atomic(later/'WORKER_STATE.json', dict(state='RUNNING'))
            G.atomic(later/'WORKER_PROFILE.json', dict(pbs=dict(allocation_dir=str(root/'active-allocation'))))
            result = G.harvest(allocation, owners=[dict(job_id='new-session', attempt_id='attempt')])
            self.assertEqual(result['incomplete'], [])

    def test_selective_harvest_proves_old_dry_check_without_session_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = root/'allocation'
            work = root/'worker'
            G.atomic(allocation/'SPEC.json', dict(profile=dict(host_id='pbs-0')))
            G.atomic(allocation/'PROFILE.json', dict(work_dir=str(work)))
            owner = dict(job_id='old-preparation', attempt_id='attempt')
            attempt = work/owner['job_id']/owner['attempt_id']
            G.atomic(attempt/'ASSIGNMENT.json', dict(owner, request_sha256='fixed'))
            G.atomic(attempt/'WORKER_PROFILE.json', dict(work_dir=str(work)))
            G.atomic(attempt/'WORKER_STATE.json', dict(state='UNKNOWN', detail={
                'automatic_retry': False, 'reason': 'ValueError: Existing adapter dry-check failed'}))
            G.atomic(attempt/'SUPERVISOR_STARTED.json', dict(pid=30))
            G.atomic(attempt/'SUPERVISOR_LAUNCH.json', dict(pid=30))
            G.atomic(attempt/'WORKER_DRY-CHECK_RETURN.json', dict(returncode=1, signal=None))
            (attempt/'WORKER_PROCESS.lock').touch()
            result = G.harvest(allocation, owners=[owner])
            item, = result['incomplete']
            proof = item['state']['detail']['prelaunch_rejection']
            self.assertEqual(proof['kind'], 'FAILED_DRY_CHECK_BEFORE_EXECUTION')
            self.assertEqual(proof['attempt_id'], owner['attempt_id'])
            self.assertEqual(result['attempts'], [])

            (attempt/'PREPARATION_STARTED.json').touch()
            self.assertEqual(G.harvest(allocation, owners=[owner])['incomplete'], [])
            self.assertNotIn('prelaunch_rejection', G.read(attempt/'WORKER_STATE.json')['detail'])

    def test_ended_session_can_prove_an_unlaunched_attempt_with_a_route_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = root/'old-session'
            work = root/'worker'
            attempt = work/'job'/'attempt'
            assignment = dict(job_id='job', attempt_id='attempt', request_sha256='fixed')
            owners = [dict(job_id='job', attempt_id='attempt')]
            G.atomic(allocation/'SPEC.json', dict(profile=dict(host_id='pbs-0')))
            G.atomic(allocation/'PROFILE.json', dict(work_dir=str(work)))
            worker = dict(state='EXITED', clock_id='clock', monotonic=5, pid=2,
                          start_clock=dict(clock_id='clock', monotonic=1))
            G.atomic(allocation/'STATE.json', worker)
            G.atomic(attempt/'ASSIGNMENT.json', assignment)
            G.atomic(attempt/'WORKER_PROFILE.json', dict(host_id='pbs-0'))
            G.atomic(attempt/'WORKER_STATE.json', dict(state='UNKNOWN', clock_id='clock', monotonic=4,
                detail=dict(reason='PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH',
                            scientific_execution_started=False, automatic_retry=False)))
            timing = dict(job_id='job', attempt_id='attempt', event='WORKER_ASSIGNMENT_DURABLE',
                          clock_id='clock', monotonic=3, pid=2)
            (attempt/'TRANSPORT_TIMING.jsonl').write_text(json.dumps(timing)+'\n')
            item, = G.harvest(allocation, owners=owners)['incomplete']
            self.assertTrue(item['state']['detail']['resource_released'])
            self.assertEqual(item['state']['detail']['pbs_admission_exit']['token'], 'old-session')
            G.atomic(allocation/'STATE.json', dict(worker, pid=3))
            self.assertEqual(G.harvest(allocation, owners=owners)['incomplete'], [])
            G.atomic(allocation/'STATE.json', worker)
            (attempt/'SCIENTIFIC_STARTING.json').touch()
            self.assertEqual(G.harvest(allocation, owners=owners)['incomplete'], [])

    def test_saved_artifact_harvest_never_calls_a_worker_or_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); allocation = root/'allocation'; work = root/'worker'
            attempt = work/'task'/'original-attempt'
            G.atomic(allocation/'SPEC.json', dict(profile=dict(host_id='pbs-0')))
            G.atomic(allocation/'PROFILE.json', dict(work_dir=str(work)))
            G.atomic(attempt/'ASSIGNMENT.json', dict(job_id='task', attempt_id='original-attempt', request=dict(route=dict(kind='artificial'))))
            G.atomic(attempt/'WORKER_STATE.json', dict(state='COMPLETED_PENDING_TRANSFER'))
            G.atomic(attempt/'PROCESS_RETURN.json', dict(returncode=0))
            # Exact DP bounds can exceed Python's default 4300-digit limit.
            exact = 10**7167 + 4321
            G.atomic(attempt/'OUTCOME.json', dict(status='ARTIFICIAL', exact_numerator=exact))
            original = (attempt/'OUTCOME.json').read_bytes()
            with patch.object(G.subprocess, 'run') as process:
                result = G.harvest(allocation)
            process.assert_not_called()
            self.assertEqual(len(result['attempts']), 1)
            self.assertEqual(result['attempts'][0]['assignment']['attempt_id'], 'original-attempt')
            self.assertIn('PROCESS_RETURN.json', result['attempts'][0]['files'])
            recovered = base64.b64decode(result['attempts'][0]['files']['OUTCOME.json'])
            self.assertEqual(recovered, original)
            value = json.loads(recovered)['exact_numerator']
            self.assertIs(type(value), int)
            self.assertEqual(value, exact)

            owners = [dict(job_id='task', attempt_id='original-attempt')]
            live = G.harvest(allocation, owners=owners, completed_only=True)
            self.assertEqual(live['attempts'], result['attempts'])
            G.atomic(attempt/'WORKER_STATE.json', dict(state='RUNNING'))
            self.assertEqual(G.harvest(allocation, owners=owners, completed_only=True)['attempts'], [])

    def test_stop_notice_alone_does_not_prove_pbs_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocation = root/'allocations'/'pending'
            request = dict(action='intent_present', token='pending')
            self.assertEqual(G.handle(root, request), dict(present=False))
            G.atomic(allocation/'CONTROL.json', dict(stop=True))
            self.assertEqual(G.handle(root, request), dict(present=False))
            self.assertEqual(G.read(allocation/'CONTROL.json'), dict(stop=True))
            G.atomic(allocation/'PILOT.json', dict(token='reserved'))
            self.assertEqual(G.handle(root, request), dict(present=True))


if __name__ == '__main__':
    unittest.main()
