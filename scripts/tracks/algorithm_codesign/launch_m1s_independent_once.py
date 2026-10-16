#!/usr/bin/env python3
"""One child invocation of frozen independent SOURCE after separate approval.

Infrastructure only: metadata preflight, external exclusive launch audit,
resource/thread configuration and process wait. No numerical helper calls.
No retries, no wall/CPU-time limit, no modification of frozen SOURCE/contract.
"""
import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
CONTRACT = 'artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/independent_validation_contract_adopted.json'
AUTHORITY = 'artifacts/track_b_m1s_independent_execution_authorization/2026-10-16/v1/execution_authorization.json'


def exclusive(path, value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class MetadataOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith(('.reference', '.execution')) or fullname.split('.')[0] in {'numpy', 'scipy', 'openfermion', 'qiskit'}:
            raise RuntimeError('Science import forbidden in supervisor metadata preflight: '+fullname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization-commit', required=True)
    args = parser.parse_args()
    commit = args.authorization_commit
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Full authorization SHA required')
    prefix = Path('/tmp')/('m1s_independent_once_'+commit)
    marker = Path(str(prefix)+'.launch.json')
    if marker.exists():
        raise FileExistsError('Launch already consumed; preserve and STOP')
    sys.path.insert(0, str(ROOT/'src'))
    blocker = MetadataOnly()
    sys.meta_path.insert(0, blocker)
    from trottertracks.algorithm_codesign.m1s_gate0_independent_v1 import control
    for key, value in control.POLICY['resources']['thread_environment'].items():
        os.environ[key] = value
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    permit = control.authorize(ROOT, CONTRACT, commit, AUTHORITY)
    # This guarded call reads only raw metadata/bytes, and does not dispatch science.
    exclusive(Path(str(prefix)+'.preflight.json'), {
        'schema': 'm1s_independent_authorized_metadata_preflight_v1',
        'authorization_commit': commit, 'guard_passed': True, 'audit': permit.audit,
        'contract_complete': control.validate_contract(permit.config, require_adopted=True)['contract_complete'],
        'frozen_SOURCE_sha256': permit.config['source_sha256'],
        'thread_environment': {k: os.environ[k] for k in control.POLICY['resources']['thread_environment']},
        'reference_kernel_calls': 0, 'NPZ_array_decodes': 0, 'numerical_tests': 0,
        'scientific_runner_invocations_before_launch': 0,
        'separate_user_execution_approval_bound': True,
        'clean_authorization_HEAD_verified': True, 'mandatory_STOP_after_run': True})
    command = [sys.executable, '-B', 'scripts/tracks/algorithm_codesign/run_m1s_gate0_independent_v1.py', 'run',
               '--contract', CONTRACT, '--authorization-commit', commit, '--authorization-path', AUTHORITY]
    limits = permit.config['resources']
    marker_record = {'schema': 'm1s_independent_exclusive_launch_record_v1', 'authorization_commit': commit,
                     'supervisor_pid': os.getpid(), 'command': command, 'cwd': str(ROOT),
                     'launch_consumed': True, 'planned_scientific_runner_invocations': 1,
                     'automatic_retry': False, 'time_unix': time.time(),
                     'SOURCE_commit': permit.config['source_commit'], 'contract_commit': permit.audit['contract_commit'],
                     'SOURCE_sha256': permit.config['source_sha256'], 'limits': limits,
                     'contract_one_shot_consumed_on_exclusive_output_reservation': True,
                     'this_extra_marker_refuses_any_second_launch': True, 'mandatory_STOP': True}
    exclusive(marker, marker_record)

    def child_limits():
        # Apply the identical adopted caps before interpreter startup as well.
        resource.setrlimit(resource.RLIMIT_AS, (limits['memory_bytes'], resource.getrlimit(resource.RLIMIT_AS)[1]))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits['output_bytes'], resource.getrlimit(resource.RLIMIT_FSIZE)[1]))

    started = time.monotonic()
    child = None
    exit_code = None
    launch_error = None
    try:
        with Path(str(prefix)+'.console.log').open('xb') as console:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                     stdout=console, stderr=subprocess.STDOUT, preexec_fn=child_limits,
                                     start_new_session=True)
            exit_code = child.wait()
    except BaseException as exc:
        launch_error = {'type': type(exc).__name__, 'reason': str(exc)}
        # Never restart a child, or remove its output/marker. Leave live child intact.
        raise
    finally:
        exclusive(Path(str(prefix)+'.completion.json'), {
            'schema': 'm1s_independent_launch_completion_v1', 'authorization_commit': commit,
            'child_pid': child.pid if child is not None else None, 'child_returncode': exit_code,
            'elapsed_seconds': time.monotonic()-started, 'launch_error': launch_error,
            'scientific_runner_invocations': 1 if child is not None else 0,
            'output_directory': permit.audit['output_directory'], 'automatic_retry': False,
            'external_marker_sha256': hashlib.sha256(marker.read_bytes()).hexdigest(), 'mandatory_STOP': True})
    print(json.dumps({'authorization_commit': commit, 'child_returncode': exit_code,
                      'external_launch_marker': str(marker), 'scientific_runner_invocations': 1,
                      'automatic_retry': False, 'mandatory_STOP': True}))
    return exit_code if exit_code is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
