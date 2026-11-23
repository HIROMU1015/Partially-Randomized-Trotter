#!/usr/bin/env python3
"""Start the fixed PBS transport with allocation-local gateway storage samples."""
import json
from pathlib import Path
import runpy
import sys
import time
import zipfile


class GatewayStorage:
    def __init__(self, sample, *, max_age):
        self.sample = sample
        self.max_age = max_age
        self.generation = None
        self.observed = None

    def report(self, policy, **_):
        if policy is None:
            return None
        unavailable = dict(limit_bytes=policy['limit_bytes'], used_bytes=None,
                           accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
        try:
            data = json.loads(self.sample.read_bytes())
            if data['policy'] != {key: policy[key] for key in ('limit_bytes', 'roots')}:
                return unavailable
            generation = data['sampled_epoch']
            if type(generation) not in (int, float):
                return unavailable
            now = time.monotonic()
            if self.generation is not None and generation != self.generation:
                self.observed = now
            self.generation = generation
            if self.observed is not None and now-self.observed <= self.max_age:
                return data['report']
        except (OSError, KeyError, ValueError, TypeError):
            pass
        return unavailable


def configure_allocation_worker(profile, launcher):
    from trottertracks import rpe_pbs_worker as pbs
    from trottertracks import rpe_q3_worker as worker

    transport = pbs
    cycle = worker
    bundled = False
    if launcher.suffix == '.pyz':
        with zipfile.ZipFile(launcher) as package:
            bundled = 'trottertracks/_collection_pbs_worker.py' in package.namelist()
    if bundled:
        # The deployment bundles only the operational implementation.
        # SOURCE, adapter and result validation still use the fixed checkout.
        import trottertracks
        trottertracks.__path__.append(str(launcher)+'/trottertracks')
        from trottertracks import _collection_pbs_worker as transport
        from trottertracks import _collection_q3_worker as cycle

    class MeasuredAllocationWorker(transport.AllocationWorker):
        def pending(self):
            if (not self.draining or not (self.directory/'PILOT.json').is_file()
                    or not self.control().get('stop')):
                return super().pending()
            # A disconnected controller may be unable to ACK before Stop. Its
            # normal harvest must still validate these files in the original run.
            for attempt in self.records:
                state = worker.read(attempt/'WORKER_STATE.json')['state']
                if state in ('TRANSFERRED', 'UNKNOWN'):
                    continue
                if (state != 'COMPLETED_PENDING_TRANSFER' or
                        not all((attempt/name).is_file() for name in ('OUTCOME.json', 'PROCESS_RETURN.json'))):
                    return True
            return False

        def _resource_offer(self):
            report = super()._resource_offer()
            for attempt, assignment in self.records.items():
                spec = assignment['request'].get('preparation', {})
                if spec.get('provider') not in ('DF_CALIBRATION', 'DF_HOLDOUT'):
                    continue
                if worker.read(attempt/'WORKER_STATE.json')['state'] != 'RUNNING':
                    continue
                marker = attempt/'SUPERVISOR_STARTED.json'
                if not marker.is_file():
                    continue
                pid = worker.read(marker).get('pid')
                if not worker.process_alive(pid, attempt):
                    continue
                sample = worker.Resources.owned_tree_sample(pid)
                path = attempt/'OWNED_RESOURCES.json'
                prior = worker.read(path) if path.is_file() else {}
                if sample['rss_bytes'] > prior.get('peak_rss_bytes', 0):
                    profile = worker.read(attempt/'WORKER_PROFILE.json')
                    worker.atomic(path, dict(worker.stamp(), peak_rss_bytes=sample['rss_bytes'],
                        task_cpu_count=profile.get('task_cpu_count'), provider=spec['provider']))
            return report

    pbs.AllocationWorker = MeasuredAllocationWorker
    if bundled:
        MeasuredAllocationWorker._service_attempt = cycle.Worker._service_attempt
        MeasuredAllocationWorker._poll_assignments = cycle.Worker._poll_assignments


def main():
    args = sys.argv[1:]
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
    observer = GatewayStorage(directory/'STORAGE_USAGE_GATEWAY.json',
                              max_age=2*storage.SAMPLE_SECONDS)
    storage.report = observer.report
    configure_allocation_worker(profile, Path(sys.argv[0]).resolve())
    if check_imports:
        print(json.dumps(dict(status='PBS_WORKER_IMPORTS_PASS', profile=str(profile_path))))
        return
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
