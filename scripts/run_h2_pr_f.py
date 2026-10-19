"""Run exactly one separately approved H2 F attempt and record native wait4."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import h2_pr_resource_contract as c


def worker(spec, grant):
    attempt = c.owned(spec['attempt'])
    marker = c.read(attempt / 'ONE_SHOT.json')
    c.require(marker['parent_pid'] == os.getppid() and marker['spec_digest'] == c.digest(spec)
              and marker['grant_digest'] == c.digest(grant), 'OWNED_ONE_SHOT_CHILD_REQUIRED')
    output = attempt / 'output'
    output.mkdir()
    start = time.monotonic(); code = 2
    try:
        scratch = output / 'scratch'; scratch.mkdir()
        for key, name in [('TMPDIR', 'tmp'), ('NUMBA_CACHE_DIR', 'numba'),
                          ('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'cache')]:
            path = scratch / name; path.mkdir(); os.environ[key] = str(path)
        import tempfile
        tempfile.tempdir = os.environ['TMPDIR']
        sys.path.insert(0, str(Path(spec['provider_root']) / 'src'))
        # Every real molecular import is downstream of the approval/identity gate.
        import h2_tol_only_input as science
        result = science.generate(spec, grant, output)
        c.require(result['actual_calls'] == spec['call_budget'], 'EXACT_F_CALL_COUNTS')
        c.save(output / 'calls_ledger.json', dict(actual=c.actual_calls(output), nonrefundable=True))
        c.verify(spec, grant, interpreter=True)
        c.save(output / 'result.json', result)
        code = 0
    except BaseException as exc:
        c.save(output / 'STOP.json', dict(reason=type(exc).__name__ + ':' + str(exc),
            traceback=traceback.format_exc(), actual_calls=c.actual_calls(output), automatic_retry=False))
    c.save(output / 'terminal.json', dict(SOURCE=spec['SOURCE'], spec_digest=c.digest(spec),
        status='COMPLETE_MANDATORY_STOP' if code == 0 else 'STOP', exit_code=code,
        wall_seconds=time.monotonic() - start, next_stage_authorized=False, automatic_retry=False))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    spec, grant = c.read(args.spec), c.read(args.grant)
    c.verify(spec, grant, interpreter=True)
    if args.worker:
        return worker(spec, grant)
    attempt = c.owned(spec['attempt'])
    c.require(not attempt.exists(), 'ONE_SHOT_ALREADY_CLAIMED_NO_RETRY')
    fresh = c.observation()
    # Atomic directory creation consumes this attempt even if the driver crashes.
    attempt.mkdir(parents=True, exist_ok=False)
    c.save(attempt / 'ONE_SHOT.json', dict(SOURCE=spec['SOURCE'], spec_digest=c.digest(spec),
        grant_digest=c.digest(grant), parent_pid=os.getpid(), nonrefundable=True))
    c.save(attempt / 'resource_observation.json', fresh)
    command = [spec['python'], '-B', '-X', 'pycache_prefix=' + str(c.PRIVATE / 'runtime_pycache'),
               str(Path(__file__).resolve()), '--spec', str(args.spec.resolve()),
               '--grant', str(args.grant.resolve()), '--worker']
    started = time.monotonic()
    try:
        with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1'))
            c.save(attempt / 'native_started.json', dict(pid=child.pid, parent_pid=os.getpid(),
                SOURCE=spec['SOURCE'], spec_digest=c.digest(spec), command=command))
            pid, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
    except BaseException as exc:
        c.save(attempt / 'driver_STOP.json', dict(reason=type(exc).__name__ + ':' + str(exc),
            native_completion_unconfirmed=True, next_stage_authorized=False, automatic_retry=False))
        return 2
    c.save(attempt / 'native_exit.json', dict(SOURCE=spec['SOURCE'], pid=pid,
        exit_code=child.returncode, same_parent_reaped=True, wait4_complete=True,
        wall_seconds=time.monotonic() - started, peak_RSS_bytes=usage.ru_maxrss * 1024,
        next_stage_authorized=False, automatic_retry=False))
    if child.returncode != 0:
        c.save(attempt / 'driver_STOP.json', dict(reason='NONZERO_NATIVE_EXIT',
            native_exit_code=child.returncode, next_stage_authorized=False, automatic_retry=False))
        return child.returncode if child.returncode > 0 else 2
    try:
        c.verify(spec, grant, interpreter=True)
        output = attempt / 'output'
        result, terminal = c.read(output / 'result.json'), c.read(output / 'terminal.json')
        c.require(terminal['status'] == result['status'] == 'COMPLETE_MANDATORY_STOP'
                  and not (output / 'STOP.json').exists(), 'F_NATIVE_RESULT_COMPLETION')
        c.require(terminal['SOURCE'] == result['SOURCE'] == spec['SOURCE']
                  and terminal['spec_digest'] == c.digest(spec) and result['scope'] == c.SCOPE
                  and result['stage'] == 'F', 'F_RESULT_IDENTITY')
        c.require(c.sha(output / 'tol_only_snapshot.npz') == result['sha256']
                  and c.read(output / 'tol_only_receipt.json') == result, 'F_SAVED_INPUT_IDENTITY')
        c.require(c.actual_calls(output) == spec['call_budget'] == result['actual_calls'], 'F_FINAL_LEDGER')
        c.save(attempt / 'completion_audit.json', dict(status='COMPLETE_MANDATORY_STOP',
            SOURCE=spec['SOURCE'], result_sha256=c.sha(output / 'result.json'),
            actual_calls=result['actual_calls'], native_exit_code=0, same_parent_reaped=True,
            independent_scientific_review=False, next_stage_authorized=False))
    except BaseException as exc:
        c.save(attempt / 'driver_STOP.json', dict(reason=type(exc).__name__ + ':' + str(exc),
                                                 next_stage_authorized=False, automatic_retry=False))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
