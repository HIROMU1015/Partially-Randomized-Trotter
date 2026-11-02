"""H2-only S_STATS sole numeric pass, supervised native wait4, then STOP."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import h2_s_stats_contract as s
c=s.c


def execute(spec,grant):
    summary=s.authorize(spec,grant)
    output,lock=Path(spec['output']),Path(spec['lock'])
    with lock.open('x') as stream:
        stream.write(c.encoded(dict(SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),grant_digest=c.digest(grant))).decode())
    output.mkdir(exist_ok=False);started=time.monotonic()
    c.save(output/'calls_ledger.json',dict(reserved=s.plan()['calls'],nonrefundable=True,automatic_retry=False))
    try:
        from h2_s_stats_numeric import analyze
        result=analyze(summary);s.verify_bytes(spec)
        result.update(SOURCE=spec['SOURCE'],stage='S_STATS',status='COMPLETE_MANDATORY_STOP',
            spec_digest=c.digest(spec),input_summary_sha256=c.sha(spec['summary']),
            fixed_A_commit=s.A_COMMIT,fixed_D_commit=s.D_COMMIT)
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
    run=Path(spec['attempt']);run.mkdir(exist_ok=False)
    started=time.monotonic()
    command=[spec['python'],'-B','-S',str(Path(__file__).resolve()),'--spec',str(args.spec.resolve()),'--grant',str(args.grant.resolve())]
    with (run/'stdout.log').open('x') as out,(run/'stderr.log').open('x') as err:
        proc=subprocess.Popen(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
        c.save(run/'native_started.json',dict(pid=proc.pid,parent_pid=os.getpid(),command=command))
        pid,status,usage=os.wait4(proc.pid,0);proc.returncode=os.waitstatus_to_exitcode(status)
    result=Path(spec['output'])/'result.json'
    c.save(run/'native_exit.json',dict(SOURCE=spec['SOURCE'],pid=pid,exit_code=proc.returncode,
        same_parent_reaped=True,wait4_complete=True,wall_seconds=time.monotonic()-started,
        peak_RSS_bytes=usage.ru_maxrss*1024,result_sha256=c.sha(result) if proc.returncode==0 and result.exists() else None,
        automatic_retry=False,next_stage_authorized=False))
    return proc.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--grant',type=Path,required=True)
    parser.add_argument('--witness',action='store_true');args=parser.parse_args()
    spec,grant=c.read(args.spec),c.read(args.grant)
    if args.witness:return witness(spec,grant,args)
    execute(spec,grant);return 0


if __name__=='__main__':raise SystemExit(main())
