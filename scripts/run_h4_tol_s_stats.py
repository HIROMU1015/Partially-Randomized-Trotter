"""DF10 S_STATS one-shot runner/witness. INVALID grants fail before numerics."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import h4_tol_s_stats_contract as s
c=s.c


def execute(spec,grant):
    summary=s.authorize(spec,grant)
    output,lock=Path(spec['output']),Path(spec['lock'])
    c.require(output.parent==lock.parent,'OWN_STATS_RUN_ROOT')
    with lock.open('x') as stream:stream.write(c.encoded(dict(SOURCE=spec['SOURCE'],spec_digest=c.digest(spec))).decode())
    output.mkdir(exist_ok=False);started=time.monotonic()
    c.save(output/'calls_ledger.json',dict(reserved=s.plan()['calls'],nonrefundable=True,automatic_retry=False))
    try:
        from h4_tol_s_stats_numeric import analyze
        result=analyze(summary)
        s.verify_bytes(spec)
        result.update(SOURCE=spec['SOURCE'],status='COMPLETE_MANDATORY_STOP',spec_digest=c.digest(spec),
            input_summary_sha256=c.sha(spec['summary']),fixed_A_commit=s.A_COMMIT,fixed_D_commit=s.D_COMMIT)
        c.save(output/'result.json',result)
        c.save(output/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=spec['SOURCE'],
            result_sha256=c.sha(output/'result.json'),numeric_wall_seconds=time.monotonic()-started,
            final_closure_checked=True,automatic_retry=False,next_stage_authorized=False))
        return result
    except BaseException as error:
        c.save(output/'STOP.json',dict(reason=type(error).__name__+':'+str(error),traceback=traceback.format_exc(),
            reserved_pass_retained=True,automatic_retry=False,next_stage_authorized=False))
        raise


def witness(spec,grant,args):
    s.authorize(spec,grant)
    run=Path(spec['output']).parent
    c.require(run.is_dir() and not (run/'native_exit.json').exists(),'UNUSED_STATS_WITNESS')
    started=time.monotonic()
    with (run/'stdout.log').open('x') as out,(run/'stderr.log').open('x') as err:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--spec',str(args.spec.resolve()),
            '--grant',str(args.grant.resolve())],stdout=out,stderr=err,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
        code=proc.wait()
    result=Path(spec['output'])/'result.json'
    c.save(run/'native_exit.json',dict(SOURCE=spec['SOURCE'],exit_code=code,same_parent_reaped=True,
        wall_seconds=time.monotonic()-started,result_sha256=c.sha(result) if code==0 and result.exists() else None,
        next_stage_authorized=False))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--grant',type=Path,required=True)
    parser.add_argument('--witness',action='store_true');args=parser.parse_args()
    spec,grant=c.read(args.spec),c.read(args.grant)
    if args.witness:raise SystemExit(witness(spec,grant,args))
    execute(spec,grant)
