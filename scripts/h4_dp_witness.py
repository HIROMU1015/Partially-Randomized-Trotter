"""Future own-driver native exit witness; DRAFT requests fail without spawning."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import h4_dp_contract as c


def observe():
    out = dict(monotonic=time.monotonic(), diagnostic_only=True, PSI_triggers_termination=False)
    for name in ['/proc/meminfo', '/proc/pressure/cpu', '/proc/pressure/memory', '/proc/self/cgroup']:
        try:
            out[name] = Path(name).read_text()
        except OSError as error:
            out[name] = dict(observation_unavailable=type(error).__name__)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    args = parser.parse_args()
    spec, grant = c.read(args.spec), c.read(args.grant)
    c.require(spec.get('status') == 'SEALED_FOR_EXECUTION' and grant.get('approved_by_user') is True
              and grant.get('runtime_authorization') is True, 'DRAFT_UNAUTHORIZED')
    # The child performs the full approval/SOURCE/input/runtime gate before science.
    run = Path(spec['output']).parent
    c.require(run.is_dir() and not (run / 'native_driver_exit.json').exists(), 'UNUSED_WITNESS_ROOT')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    start = time.monotonic()
    with (run / 'driver_stdout.log').open('x') as stdout, (run / 'driver_stderr.log').open('x') as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('run_h4_dp_verification.py')),
            '--spec', str(args.spec), '--grant', str(args.grant)], stdout=stdout, stderr=stderr, env=environment)
        with (run / 'resource_observations.jsonl').open('x') as observations:
            while True:
                observations.write(c.encoded(observe()).decode() + '\n'); observations.flush()
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
    c.save(run / 'native_driver_exit.json', dict(SOURCE=spec['SOURCE'], pid=process.pid,
        exit_code=code, same_parent_reaped=True, wall_seconds=time.monotonic()-start,
        next_stage_authorized=False, result_sha256=c.sha(Path(spec['output'])/'result.json')
        if code == 0 and (Path(spec['output'])/'result.json').is_file() else None))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
