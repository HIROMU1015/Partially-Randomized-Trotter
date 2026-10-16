#!/usr/bin/env python3
"""Reserve one external launch marker, then exec the frozen guarded v5 runner.

No numerical imports, input decoding, threshold changes, loops or retries.
The runner independently checks the committed authority and every frozen input.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
CONTRACT = 'artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/execution_contract_v5_adopted.json'
AUTHORITY = 'artifacts/track_b_m1s_gate0_v5_execution_authorization/2026-10-16/v1/execution_authorization.json'
RUNNER = 'scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v5.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization-commit', required=True)
    args = parser.parse_args()
    ref = args.authorization_commit
    if re.fullmatch('[0-9a-f]{40}', ref) is None:
        raise ValueError('Full fixed authorization SHA required')
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(ROOT), *argv])
    if git('rev-parse', 'HEAD').decode().strip() != ref:
        raise ValueError('Fixed authorization HEAD required')
    if git('status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('Clean dedicated checkout required before reserving launch')
    raw = git('show', ref + ':' + AUTHORITY)
    if raw != (ROOT / AUTHORITY).read_bytes():
        raise ValueError('Frozen authorization bytes differ')
    authority = json.loads(raw)
    if (authority.get('schema') != 'm1s_gate0_separate_execution_authorization_v5'
            or authority.get('explicit_user_scientific_execution_approval') is not True):
        raise ValueError('Separate explicit execution approval required')
    config_raw = (ROOT / CONTRACT).read_bytes()
    if hashlib.sha256(config_raw).hexdigest() != authority['contract_sha256']:
        raise ValueError('Frozen contract bytes differ')
    config = json.loads(config_raw)
    env = os.environ.copy()
    env.update(config['runtime']['threads'])
    command = [sys.executable, str(ROOT / RUNNER), 'execute', '--contract', CONTRACT,
               '--authorization-commit', ref, '--authorization-path', AUTHORITY]
    marker = Path('/tmp') / ('m1s_gate0_v5_once_' + ref + '.launch.json')
    launch = {'schema': 'm1s_gate0_v5_exclusive_launch_v1', 'authorization_commit': ref,
              'authorization_path': AUTHORITY, 'authorization_sha256': hashlib.sha256(raw).hexdigest(),
              'contract_path': CONTRACT, 'contract_sha256': authority['contract_sha256'],
              'source_commit': authority['source_commit'], 'output_directory': authority['output_directory'],
              'pid_preserved_by_exec': os.getpid(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command': command, 'thread_environment': config['runtime']['threads'],
              'marker_path': str(marker), 'maximum_runner_invocations': 1,
              'automatic_retry': False, 'scientific_wall_time_cap': None,
              'consumed_on_launch_even_if_runner_guard_or_runtime_fails': True}
    # This marker is outside the checkout, preserving the runner's clean-HEAD guard.
    with marker.open('x') as stream:
        json.dump(launch, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


if __name__ == '__main__':
    main()
