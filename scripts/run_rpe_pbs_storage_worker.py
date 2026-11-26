#!/usr/bin/env python3
"""Start the fixed PBS transport with allocation-local gateway storage samples."""
import json
import os
from pathlib import Path
import runpy
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
                # A cold sweep can take longer than the poll interval. Keep a
                # measured sweep usable during the next one only with headroom.
                if (type(used) is int and 0 <= used <= policy['limit_bytes'] * 95 // 100
                        and type(measured) in (int, float) and 0 <= measured <= COMPUTE_DU_TIMEOUT):
                    age += measured
                elapsed = time.time() - generation
                if 0 <= elapsed <= age:
                    return report
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


def configure_artifact_transport(launcher):
    """Use streaming CAS I/O without replacing a fixed provider or its receipts."""
    if launcher.suffix != '.pyz':return
    with zipfile.ZipFile(launcher) as package:
        if 'trottertracks/_pipeline_preparation_artifacts.py' not in package.namelist():return
    import trottertracks
    path = str(launcher)+'/trottertracks'
    if path not in trottertracks.__path__:trottertracks.__path__.append(path)
    from trottertracks import rpe_preparation_artifacts as pinned, _pipeline_preparation_artifacts as transport
    for name in ('missing', 'materialize', 'capture', 'put_file'):
        setattr(pinned.ArtifactCache, name, getattr(transport.ArtifactCache, name))


def configure_allocation_worker(profile, launcher):
    configure_artifact_transport(launcher)
    from trottertracks import rpe_pbs_worker as pbs
    from trottertracks import rpe_q3_worker as worker

    transport = pbs
    cycle = worker
    members = set()
    if launcher.suffix == '.pyz':
        with zipfile.ZipFile(launcher) as package:
            members = set(package.namelist())
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
        raise SystemExit(_collection_q3_worker.supervise(Path(attempt)))
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
    observer = GatewayStorage(sample, max_age=2*storage.SAMPLE_SECONDS, local_clock=compute_sampling,
                              policy_fingerprint=sampler.policy_identity if sampler else None)
    storage.report = observer.report
    os.environ.update(RPE_PBS_OPERATIONAL_LAUNCHER=str(launcher),
                      RPE_PBS_TRANSPORT_ROOT=str(script.parents[1]))
    configure_allocation_worker(profile, launcher)
    if check_imports:
        print(json.dumps(dict(status='PBS_WORKER_IMPORTS_PASS', profile=str(profile_path))))
        return
    stop_sampling = threading.Event()
    if compute_sampling and profile.get('storage_limit'):
        policy = dict(profile['storage_limit'], sample_path=str(sample))
        if sampler:
            policy['exclusion_file'] = str(sampler.storage_exclusion_file(profile['storage_limit']))

        def sample_storage():
            # On the PBS Linux nodes nice applies to this background thread.
            try:
                os.nice(5)
            except OSError:
                pass
            while not stop_sampling.is_set():
                if not (directory/'OFFLINE_ACTIVE').exists():
                    sample_report(policy, spools=[profile['work_dir']],
                                  cache_directories=storage.prefix_caches(profile), refresh=True,
                                  proactive=True, pbs_runs=str(directory.parents[2]),
                                  du_timeout=COMPUTE_DU_TIMEOUT)
                stop_sampling.wait(storage.SAMPLE_SECONDS)

        threading.Thread(target=sample_storage, name='pbs-storage-sample', daemon=True).start()
    sys.argv = [str(script), *args]
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        stop_sampling.set()


if __name__ == '__main__':
    main()
