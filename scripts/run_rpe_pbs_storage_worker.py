#!/usr/bin/env python3
"""Start the fixed PBS transport with allocation-local gateway storage samples."""
import json
from pathlib import Path
import runpy
import sys
import time


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


def main():
    args = sys.argv[1:]
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
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
