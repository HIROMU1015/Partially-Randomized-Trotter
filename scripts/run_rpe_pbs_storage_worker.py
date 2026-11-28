#!/usr/bin/env python3
"""Start the fixed PBS transport with allocation-local gateway storage samples."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import threading
import time
import zipfile


COMPUTE_DU_TIMEOUT = 150


class GatewayStorage:
    def __init__(self, sample, *, max_age, local_clock=False, policy_fingerprint=None):
        self.policy_fingerprint = policy_fingerprint or (lambda policy: {key: policy[key] for key in ('limit_bytes', 'roots')})
        self.sample = sample
        self.max_age = max_age
        self.local_clock = local_clock
        self.in_progress = sample.with_name("STORAGE_SAMPLE_IN_PROGRESS.json")
        self.generation = None
        self.observed = None

    def report(self, policy, **_):
        if policy is None:
            return None
        unavailable = dict(limit_bytes=policy['limit_bytes'], used_bytes=None,
                           accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
        try:
            data = json.loads(self.sample.read_bytes())
            if data['policy'] != self.policy_fingerprint(policy):
                return unavailable
            generation = data['sampled_epoch']
            if type(generation) not in (int, float):
                return unavailable
            now = time.monotonic()
            if self.local_clock:
                report = data['report']
                age = self.max_age
                used = report.get('used_bytes')
                measured = report.get('measurement_seconds')
                elapsed = time.time() - generation
                if 0 <= elapsed <= age:
                    return report
                # Retain a positive measurement only while a bounded new scan
                # of the identical policy is actually in progress. A stalled
                # sampler, low disk space or a quota near its limit fails closed.
                if (type(used) is int and 0 <= used <= policy['limit_bytes'] * 95 // 100
                        and report.get('accepting_new_jobs') is True
                        and type(measured) in (int, float) and 0 <= measured <= COMPUTE_DU_TIMEOUT
                        and 0 <= elapsed <= age + COMPUTE_DU_TIMEOUT):
                    active = json.loads(self.in_progress.read_bytes())
                    started = active['started_epoch']
                    if (active['policy'] == data['policy']
                            and generation <= started <= generation + age
                            and 0 <= time.time()-started <= COMPUTE_DU_TIMEOUT):
                        minimum = report.get('filesystem_min_free_bytes', policy['limit_bytes']//10)
                        roots = policy['roots']
                        if isinstance(roots, dict):
                            roots = roots.values()
                        existing = [root for root in roots if Path(root).exists()]
                        if existing and all(shutil.disk_usage(root).free >= minimum for root in existing):
                            return dict(report, measurement_in_progress=True, measurement_age_seconds=elapsed)
                # During a slow new sweep an actual observed capacity hold
                # remains actionable. Never promote a stale positive offer.
                if (self.max_age < elapsed <= 2 * COMPUTE_DU_TIMEOUT + self.max_age
                        and report.get('accepting_new_jobs') is False
                        and report.get('reason') in ('STORAGE_LIMIT_REACHED', 'FILESYSTEM_SPACE_LOW')
                        and type(used) is int):
                    return dict(report, measurement_stale=True)
                return unavailable
            if self.generation is not None and generation != self.generation:
                self.observed = now
            self.generation = generation
            if self.observed is not None and now-self.observed <= self.max_age:
                return data['report']
        except (OSError, KeyError, ValueError, TypeError):
            pass
        return unavailable


def seed_fresh_storage_sample(shared_sample, sample, expected_policy, max_age, storage):
    """Reuse the previous session's actual fresh scan, never renew its epoch."""
    try:
        record = json.loads(shared_sample.read_bytes())
        if (record['policy'] == expected_policy
                and 0 <= time.time()-record['sampled_epoch'] <= max_age):
            # Caller verifies the shared filename's policy, but compare the
            # complete expected policy as well before accepting its sample.
            storage.atomic_json(sample, record)
            return True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return False


def configure_artifact_transport(launcher):
    """Use streaming CAS I/O without replacing a fixed provider or its receipts."""
    if launcher.suffix != '.pyz':return
    with zipfile.ZipFile(launcher) as package:
        if 'trottertracks/_pipeline_preparation_artifacts.py' not in package.namelist():return
    import trottertracks
    path = str(launcher)+'/trottertracks'
    if path not in trottertracks.__path__:trottertracks.__path__.append(path)
    from trottertracks import rpe_preparation_artifacts as pinned, _pipeline_preparation_artifacts as transport
    # The streaming cache methods form one implementation. Copying only
    # methods into an older class left its initialization and helpers missing.
    # Preserve the class identity for fixed modules that already imported it,
    # including staticmethod descriptors, while installing the coherent API.
    for name, implementation in vars(transport.ArtifactCache).items():
        if name not in ('__module__', '__dict__', '__weakref__', '__doc__'):
            setattr(pinned.ArtifactCache, name, implementation)


def load_bundled_module(launcher, module_name):
    """Select the explicitly packaged operational module, not the pinned copy."""
    import importlib.util
    import zipimport
    import trottertracks
    name = 'trottertracks.'+module_name
    loader = zipimport.zipimporter(str(launcher)+'/trottertracks')
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(trottertracks, module_name, module)
    return module


def load_bundled_offline(launcher):
    return load_bundled_module(launcher, 'rpe_pbs_offline')


def configure_allocation_worker(profile, launcher):
    configure_artifact_transport(launcher)
    members = set()
    if launcher.suffix == '.pyz':
        with zipfile.ZipFile(launcher) as package:
            members = set(package.namelist())
        # Same-name modules in the fixed checkout must not shadow an
        # explicitly packaged operational update.
        for name in ('rpe_pbs_offline','rpe_dry_check_recovery','rpe_pbs_attempt_control'):
            if 'trottertracks/'+name+'.py' in members:
                load_bundled_module(launcher, name)
    from trottertracks import rpe_pbs_worker as pbs
    from trottertracks import rpe_q3_worker as worker

    transport = pbs
    cycle = worker
    bundled = 'trottertracks/_collection_pbs_worker.py' in members
    if bundled:
        # The deployment bundles only the operational implementation.
        # SOURCE, adapter and result validation still use the fixed checkout.
        import trottertracks
        trottertracks.__path__.append(str(launcher)+'/trottertracks')
        from trottertracks import _collection_pbs_worker as transport
        from trottertracks import _collection_q3_worker as cycle
        if 'trottertracks/_pipeline_q3_resources.py' in members:
            from trottertracks import _pipeline_q3_resources as resources
            cycle.Resources = transport.Resources = resources

    from trottertracks import rpe_pbs_attempt_control as attempt_control

    class MeasuredAllocationWorker(transport.AllocationWorker):
        def pending(self):
            if (not self.draining or not (self.directory/'PILOT.json').is_file()
                    or not self.control().get('stop')):
                return super().pending()
            # A disconnected controller may be unable to ACK before Stop. Its
            # normal harvest must still validate these files in the original run.
            for attempt in self.records:
                saved = worker.read(attempt/'WORKER_STATE.json')
                state = saved['state']
                if attempt_control.unreleased_unknown(attempt, saved):
                    return True
                if state in ('TRANSFERRED', 'UNKNOWN'):
                    continue
                if (state != 'COMPLETED_PENDING_TRANSFER' or
                        not all((attempt/name).is_file() for name in ('OUTCOME.json', 'PROCESS_RETURN.json'))):
                    return True
            return False

        def _resource_offer(self):
            report = super()._resource_offer()
            observations = {item['attempt_id']: item for item in report.get('preparation_observations', ())}
            for attempt, assignment in self.records.items():
                spec = assignment['request'].get('preparation', {})
                if not spec or assignment['attempt_id'] in observations:
                    continue
                state = worker.read(attempt/'WORKER_STATE.json')['state']
                if state not in ('RUNNING', 'COMPLETED_PENDING_TRANSFER'):
                    continue
                marker = attempt/'SUPERVISOR_STARTED.json'
                path = attempt/'OWNED_RESOURCES.json'
                prior = worker.read(path) if path.is_file() else {}
                profile = worker.read(attempt/'WORKER_PROFILE.json')
                peak = prior.get('peak_rss_bytes', 0)
                if state == 'RUNNING' and marker.is_file():
                    pid = worker.read(marker).get('pid')
                    if worker.process_alive(pid, attempt):
                        sample = cycle.Resources.owned_tree_sample(pid)
                        if sample['rss_bytes'] > peak:
                            peak = sample['rss_bytes']
                            worker.atomic(path, dict(worker.stamp(), peak_rss_bytes=peak,
                                task_cpu_count=profile.get('task_cpu_count'), provider=spec['provider']))
                observe = getattr(cycle.Resources, 'owned_preparation_observation', None)
                if observe:
                    item = observe(attempt, assignment, profile, peak)
                    if item:observations[item['attempt_id']] = item
            if observations:report['preparation_observations'] = list(observations.values())
            return report

    pbs.AllocationWorker = MeasuredAllocationWorker
    if bundled:
        MeasuredAllocationWorker._service_attempt = cycle.Worker._service_attempt
        MeasuredAllocationWorker._poll_assignments = cycle.Worker._poll_assignments
        from trottertracks import _pipeline_release_deployment as deployment

        def validate_release(self, selected, assignment):
            deployment.validate_assignment_release(
                worker.read(selected['manifest']), worker.read(selected['layout']), assignment,
                self.profile, input_hash_cache=getattr(self, 'batch_input_hashes', None),
                request_validator=cycle.validate_assignment)

        MeasuredAllocationWorker._validate_release_assignment = validate_release

        if 'trottertracks/_pipeline_pbs_preparation_batch.py' in members:
            from trottertracks import _pipeline_pbs_preparation_batch as preparation_batch

            class CombinedAllocationWorker(preparation_batch.BatchPreparationAcquisition, MeasuredAllocationWorker):
                preparation_acquisition_workers = transport.READINESS_WORKERS

            pbs.AllocationWorker = CombinedAllocationWorker


def drain_worker_children():
    """The worker owns the same detached children that the keeper would reap.

    Keep capacity observation alive in the parent until they have exited;
    never signal or relaunch a scientific child during technical shutdown.
    """
    while True:
        try:
            os.waitpid(-1, 0)
        except InterruptedError:
            continue
        except ChildProcessError:
            return


def become_worker_subreaper():
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot track allocation worker children')


def drain_adopted_worker_children():
    # After an abrupt worker death (including SIGKILL), Linux reparents its
    # descendants to this subreaper's main thread. Do not waitpid(-1) here:
    # the independent sampling threads have their own du subprocesses.
    children = Path(f'/proc/self/task/{os.getpid()}/children')
    while True:
        pids = children.read_text().split()
        if not pids:
            return
        for pid in pids:
            try:
                os.waitpid(int(pid), 0)
            except (ChildProcessError, InterruptedError):
                pass


def run_sampled_worker(launcher, script, args, stop_sampling, sampler_threads=()):
    # Sampling/maintenance stay in the launcher process. Exceptions in the
    # allocation worker cannot stop these threads while its owned children drain.
    try:
        become_worker_subreaper()
        returncode = subprocess.run([sys.executable, '-B', str(launcher), '--worker-only',
                                    str(script), *args]).returncode
        drain_adopted_worker_children()
        return returncode
    finally:
        stop_sampling.set()
        for thread in sampler_threads:
            thread.join()


def main():
    args = sys.argv[1:]
    launcher = Path(sys.argv[0]).resolve()
    if args[:1] == ['--supervise-attempt']:
        _, transport_root, attempt = args
        sys.path.insert(0, str(Path(transport_root)/'src'))
        import trottertracks
        trottertracks.__path__.append(str(launcher)+'/trottertracks')
        configure_artifact_transport(launcher)
        from trottertracks import _collection_q3_worker
        with zipfile.ZipFile(launcher) as package:
            if 'trottertracks/_pipeline_q3_resources.py' in package.namelist():
                from trottertracks import _pipeline_q3_resources
                _collection_q3_worker.Resources = _pipeline_q3_resources
        raise SystemExit(_collection_q3_worker.supervise(Path(attempt)))
    worker_only = args[:1] == ['--worker-only']
    if worker_only:
        args.pop(0)
    check_imports = args[:1] == ['--check-imports']
    if check_imports:
        args.pop(0)
    if args and args[0] == '-B':
        sys.dont_write_bytecode = True
        args.pop(0)
    script = Path(args.pop(0)).resolve()
    if script.name != 'run_rpe_pbs_worker.py' or args[:1] != ['--profile']:
        raise ValueError('Expected the fixed PBS transport and profile')
    profile_path = Path(args[1]).resolve()
    profile = json.loads(profile_path.read_bytes())
    directory = Path(profile['pbs']['allocation_dir']).resolve()
    if profile_path != directory/'PROFILE.json':
        raise ValueError('PBS profile is outside its allocation')
    sys.path.insert(0, str(script.parents[1]/'src'))
    from trottertracks import rpe_worker_storage as storage
    sample = directory/'STORAGE_USAGE_GATEWAY.json'
    compute_sampling = profile['pbs'].get('compute_storage_sampling') is True
    sample_report = storage.report
    sampler = None
    if compute_sampling and launcher.suffix == '.pyz':
        import trottertracks
        trottertracks.__path__.append(str(launcher)+'/trottertracks')
        from trottertracks import _pbs_storage_report as sampler
        sample_report = sampler.report
    observer = GatewayStorage(sample, max_age=30 if compute_sampling else 2*storage.SAMPLE_SECONDS, local_clock=compute_sampling,
                              policy_fingerprint=sampler.policy_identity if sampler else None)
    storage.report = observer.report
    os.environ.update(RPE_PBS_OPERATIONAL_LAUNCHER=str(launcher),
                      RPE_PBS_TRANSPORT_ROOT=str(script.parents[1]))
    configure_allocation_worker(profile, launcher)
    if check_imports:
        selected = sys.modules.get('trottertracks.rpe_pbs_offline')
        modules = {name:getattr(sys.modules.get('trottertracks.'+name),'__file__',None)
                   for name in ('rpe_pbs_offline','rpe_dry_check_recovery','rpe_pbs_attempt_control',
                                'rpe_pbs_worker','rpe_q3_worker','_collection_pbs_worker','_collection_q3_worker',
                                '_pipeline_q3_resources')}
        print(json.dumps(dict(status='PBS_WORKER_IMPORTS_PASS', profile=str(profile_path),
                              offline_module=getattr(selected, '__file__', None), modules=modules)))
        return
    stop_sampling = threading.Event()
    if compute_sampling and profile.get('storage_limit') and not worker_only:
        policy = dict(profile['storage_limit'], sample_path=str(sample))
        if sampler:
            policy['exclusion_file'] = str(sampler.storage_exclusion_file(profile['storage_limit']))

        # A shared-filesystem policy can have several physical reservations.
        # Reuse the existing report lock/cache for exactly that byte policy;
        # each allocation still gets its own fresh receipt and observation.
        policy_key = hashlib.sha256(json.dumps(observer.policy_fingerprint(policy),
                                               sort_keys=True).encode()).hexdigest()[:16]
        configured_sample = Path(profile['storage_limit']['sample_path'])
        shared_sample = configured_sample.with_name('STORAGE_COMPUTE_'+policy_key+'.json')
        shared_policy = dict(policy, sample_path=str(shared_sample))
        seed_fresh_storage_sample(shared_sample, sample, observer.policy_fingerprint(policy),
                                  observer.max_age, storage)

        # Admission uses one allocation-local measurement. The previous
        # sampler mixed multiple full-directory rechecks into that measurement,
        # leaving no current capacity offer for minutes.
        maintenance_due = threading.Event()
        maintenance_policy = dict(policy, sample_path=str(directory/'STORAGE_USAGE_MAINTENANCE.json'))

        def bind_storage_stop():
            for module in (storage, sampler):
                bind = getattr(module, 'set_storage_stop_event', None)
                if bind is not None:
                    bind(stop_sampling)

        def maintain_storage():
            bind_storage_stop()
            try:
                os.nice(10)
            except OSError:
                pass
            while not stop_sampling.is_set():
                if not maintenance_due.wait(5):
                    continue
                maintenance_due.clear()
                if stop_sampling.is_set():
                    break
                try:
                    data = json.loads(sample.read_bytes())
                    if data['policy'] == observer.policy_fingerprint(policy):
                        implementation = sampler if sampler is not None else storage
                        result = implementation.reclaim_from_measurement(maintenance_policy, data['report'],
                            spools=[profile['work_dir']], cache_directories=storage.prefix_caches(profile),
                            pbs_runs=str(directory.parents[2]), du_timeout=COMPUTE_DU_TIMEOUT)
                        storage.atomic_json(Path(maintenance_policy['sample_path']), dict(epoch=time.time(), **result))
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                    try:
                        storage.atomic_json(Path(maintenance_policy['sample_path']),
                                            dict(epoch=time.time(), status='CLEANUP_DEFERRED', reason=str(error)))
                    except OSError:
                        pass  # a temporary filesystem error must not end maintenance
                # The common central storage service also reclaims verified
                # copies; avoid repeated expensive reconciliation at poll rate.
                stop_sampling.wait(300)

        def sample_storage():
            bind_storage_stop()
            try:
                os.nice(5)
            except OSError:
                pass
            while not stop_sampling.is_set():
                marker = observer.in_progress
                temp = marker.with_suffix('.tmp')
                try:
                    temp.write_text(json.dumps(dict(policy=observer.policy_fingerprint(policy),
                                                    started_epoch=time.time())))
                    os.replace(temp, marker)
                    # No spool/caches on this measurement-only path.
                    # The same byte policy and exclusions still apply.
                    result = sample_report(shared_policy, refresh=False, proactive=True, measurement_only=True,
                                           du_timeout=COMPUTE_DU_TIMEOUT)
                    # Preserve the actual sample epoch, not the time a second
                    # allocation reads it. An unavailable scan never renews a
                    # previously positive measurement.
                    try:
                        record = json.loads(shared_sample.read_bytes())
                        if record['policy'] != observer.policy_fingerprint(policy) or record['report'] != result:
                            raise ValueError('Shared measurement unavailable')
                        storage.atomic_json(sample, record)
                    except (OSError, ValueError, KeyError, TypeError):
                        storage.atomic_json(sample, dict(policy=observer.policy_fingerprint(policy),
                            sampled_epoch=time.time(), report=dict(limit_bytes=policy['limit_bytes'],
                            used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')))
                    if result and result.get('reclaim_pending'):
                        maintenance_due.set()
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                    try:
                        storage.atomic_json(sample, dict(policy=observer.policy_fingerprint(policy),
                            sampled_epoch=time.time(), report=dict(limit_bytes=policy['limit_bytes'],
                            used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')))
                    except OSError:
                        pass  # last sample ages out; next scan retries
                finally:
                    try:
                        marker.unlink(missing_ok=True)
                    except OSError:
                        pass
                stop_sampling.wait(storage.SAMPLE_SECONDS)

        threads = [threading.Thread(target=maintain_storage, name='pbs-storage-maintenance', daemon=True),
                   threading.Thread(target=sample_storage, name='pbs-storage-sample', daemon=True)]
        for thread in threads:
            thread.start()
        raise SystemExit(run_sampled_worker(launcher, script, args, stop_sampling, threads))
    if worker_only:
        # PBS's Linux keeper already uses this same subreaper mechanism. Put
        # it inside the worker subprocess so that the sampling parent remains
        # live throughout a normal or exceptional worker drain.
        become_worker_subreaper()
    sys.argv = [str(script), *args]
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        if worker_only:
            drain_worker_children()
        stop_sampling.set()


if __name__ == '__main__':
    main()
