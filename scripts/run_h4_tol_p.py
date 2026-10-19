"""One-shot DF10 P entrypoint. This file is never activated by final sealing."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import h4_tol_p_contract as p
c = p.c


def execute(spec, grant):
    p.authorize(spec, grant)
    output, lock = Path(spec['output']), Path(spec['lock'])
    with lock.open('x') as stream:
        stream.write(c.encoded(dict(SOURCE=spec['SOURCE'], contract_digest=c.digest(spec))).decode())
    output.mkdir(exist_ok=False)
    c.save(output/'calls_ledger.json', dict(maximum=p.plan()['calls'], nonrefundable=True,
        one_shot=True, automatic_retry=False))
    try:
        from h4_tol_p_science import execute as scientific_execute
        result = scientific_execute(spec, output)
        p.verify_bytes(spec)
        result.update(SOURCE=spec['SOURCE'], contract_digest=c.digest(spec), status='COMPLETE_MANDATORY_STOP')
        c.save(output/'result.json', result)
        c.save(output/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
            result_sha256=c.sha(output/'result.json'), own_workers_reaped=True, final_closure_checked=True,
            automatic_retry=False, next_stage_authorized=False))
        return result
    except BaseException as error:
        c.save(output/'STOP.json', dict(reason=type(error).__name__+':'+str(error), traceback=traceback.format_exc(),
            reservations_retained=True, automatic_retry=False, next_stage_authorized=False))
        raise


def witness(spec, grant, args):
    p.authorize(spec, grant)
    run = Path(spec['output']).parent
    c.require(run.is_dir() and not (run/'native_exit.json').exists(), 'UNUSED_P_WITNESS')
    start = time.monotonic()
    with (run/'stdout.log').open('x') as out, (run/'stderr.log').open('x') as err:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--spec', str(args.spec.resolve()),
            '--grant', str(args.grant.resolve())], stdout=out, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                NUMBA_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', RAYON_NUM_THREADS='1'))
        code = proc.wait()
    result = Path(spec['output'])/'result.json'
    c.save(run/'native_exit.json', dict(SOURCE=spec['SOURCE'], exit_code=code, same_parent_reaped=True,
        wall_seconds=time.monotonic()-start, result_sha256=c.sha(result) if code == 0 and result.exists() else None,
        next_stage_authorized=False))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True); parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--witness', action='store_true'); args = parser.parse_args()
    spec, grant = c.read(args.spec), c.read(args.grant)
    if args.witness: raise SystemExit(witness(spec, grant, args))
    execute(spec, grant)
