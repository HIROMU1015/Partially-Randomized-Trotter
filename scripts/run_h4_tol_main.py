"""One separately approved rank10 D/A stage. Phase0 grants always reject."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import h4_tol_main_contract as c


def execute(contract, grant):
    c.authorize(contract, grant)
    output = Path(contract['output'])
    with Path(contract['lock']).open('x') as stream:
        stream.write(c.old.encoded(dict(SOURCE=contract['SOURCE'], contract_digest=c.digest(contract))).decode())
    output.mkdir(exist_ok=False)
    try:
        return run_claimed(contract, output)
    except BaseException as error:
        if not (output/'STOP.json').exists():
            c.save(output/'STOP.json', dict(status='TECHNICAL_STOP', reason=type(error).__name__+':'+str(error),
                reservations_retained=True, next_stage_authorized=False))
        raise


def run_claimed(contract, output):
    # Own-process environment only. Historical worker/memory/wall caps are not inherited.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS', 'QISKIT_NUM_PROCS'):
        os.environ[name] = '1'
    os.environ['QISKIT_PARALLEL'] = 'FALSE'; os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for variable in ('TMPDIR', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
        path = output/variable.lower(); path.mkdir(); os.environ[variable] = str(path)
    tempfile.tempdir = str(output/'tmpdir')
    font = contract['font_seed']; raw = Path(font['path']).read_bytes()
    c.require(__import__('hashlib').sha256(raw).hexdigest() == font['sha256'], 'FONT_SEED')
    (output/'mplconfigdir'/font['name']).write_bytes(raw)
    sys.dont_write_bytecode = True
    start = time.monotonic()
    try:
        import h4_tol_main_science
        result = h4_tol_main_science.execute(contract, output)
        # Final checks cover only this run's actual closure, never unrelated worktrees.
        for row in contract['inputs']:
            c.require(c.sha(row['path']) == row['sha256'], 'FINAL_INPUT_CHANGED')
        for name, expected in contract['source_hashes'].items():
            c.require(c.sha(c.ROOT/name) == expected, 'FINAL_SOURCE_CHANGED')
        for group in ('runtime_hashes', 'provider_hashes'):
            for path, expected in contract[group].items():
                c.require(c.sha(path) == expected, 'FINAL_EXECUTION_CLOSURE_CHANGED')
        c.save(output/'result.json', result)
        c.save(output/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=contract['SOURCE'],
            own_workers_reaped=True, wall_seconds=time.monotonic()-start,
            next_stage_authorized=False, automatic_retry=False))
        return result
    except BaseException as error:
        c.save(output/'STOP.json', dict(status='TECHNICAL_STOP', reason=type(error).__name__+':'+str(error),
            traceback=traceback.format_exc(), reservations_retained=True, next_stage_authorized=False))
        raise


def witness(contract, grant, args):
    # Full preflight before a child is created; the child repeats the same gate.
    c.authorize(contract, grant)
    run = Path(contract['output']).parent
    c.require(run.is_dir() and not (run/'native_exit.json').exists(), 'UNUSED_WITNESS_ROOT')
    start = time.monotonic()
    with (run/'stdout.log').open('x') as stdout, (run/'stderr.log').open('x') as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--stage', args.stage,
            '--contract', str(args.contract.resolve()), '--grant', str(args.grant.resolve())],
            stdout=stdout, stderr=stderr, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        with (run/'resource_observations.jsonl').open('x') as stream:
            while True:
                observations = {}
                for name in ('/proc/meminfo', '/proc/pressure/cpu', '/proc/pressure/memory'):
                    try: observations[name] = Path(name).read_text()
                    except OSError as error: observations[name] = dict(observation_error=str(error))
                stream.write(c.old.encoded(dict(time=time.time(), diagnostic_only=True, observations=observations)).decode()+'\n')
                stream.flush()
                try:
                    code = process.wait(timeout=5); break
                except subprocess.TimeoutExpired: pass
    c.save(run/'native_exit.json', dict(SOURCE=contract['SOURCE'], exit_code=code, pid=process.pid,
        same_parent_reaped=True, wall_seconds=time.monotonic()-start, next_stage_authorized=False,
        result_sha256=c.sha(Path(contract['output'])/'result.json') if code == 0 else None))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('D', 'A', 'P', 'S_STATS'), required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--witness', action='store_true')
    args = parser.parse_args()
    contract, grant = c.read(args.contract), c.read(args.grant)
    c.require(contract['stage'] == args.stage, 'CLI_STAGE_BINDING')
    if args.witness: return witness(contract, grant, args)
    execute(contract, grant)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
