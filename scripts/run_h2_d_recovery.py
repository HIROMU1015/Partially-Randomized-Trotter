"""One explicitly authorized D_run02, existing supervised launch, then STOP."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import h2_d_recovery_contract as c


def worker(spec,grant):
    attempt=c.owned(spec['attempt']);marker=c.read(attempt/'ONE_SHOT.json')
    c.require(marker['parent_pid']==os.getppid() and marker['spec_digest']==c.digest(spec)
        and marker['grant_digest']==c.digest(grant),'OWNED_ONE_SHOT_WORKER')
    output=attempt/'output';output.mkdir();code=2;start=time.monotonic()
    try:
        scratch=output/'scratch';scratch.mkdir()
        for key,name in [('TMPDIR','tmp'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','cache')]:
            p=scratch/name;p.mkdir();os.environ[key]=str(p)
        import tempfile
        tempfile.tempdir=os.environ['TMPDIR']
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):os.environ[key]='1'
        import h2_d_recovery_science
        result=h2_d_recovery_science.execute(spec,grant,output)
        c.authorize(spec,grant);c.save(output/'result.json',result);code=0
    except BaseException as exc:
        c.save(output/'STOP.json',dict(reason=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc(),
            automatic_retry=False,next_stage_authorized=False,reservations_preserved=True))
    c.save(output/'terminal.json',dict(stage='D',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),
        status='COMPLETE_MANDATORY_STOP' if code==0 else 'STOP',exit_code=code,
        wall_seconds=time.monotonic()-start,automatic_retry=False,next_stage_authorized=False))
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--grant',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=parser.parse_args(argv);spec,grant=c.read(a.spec),c.read(a.grant);c.authorize(spec,grant)
    if a.worker:return worker(spec,grant)
    attempt=c.owned(spec['attempt']);c.require(not attempt.exists(),'ONE_SHOT_CONSUMED_NO_RETRY')
    observed=c.old.f.observation();attempt.mkdir(parents=True,exist_ok=False)
    c.save(attempt/'ONE_SHOT.json',dict(parent_pid=os.getpid(),SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),
        grant_digest=c.digest(grant),nonrefundable=True))
    c.save(attempt/'resource_observation.json',observed)
    command=[spec['python'],'-B','-X','pycache_prefix='+str(c.PRIVATE/'panels_runtime_pycache'),
        str(Path(__file__).resolve()),'--spec',str(a.spec.resolve()),'--grant',str(a.grant.resolve()),'--worker']
    start=time.monotonic()
    try:
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
            c.save(attempt/'native_started.json',dict(pid=child.pid,parent_pid=os.getpid(),command=command))
            pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
        c.save(attempt/'native_exit.json',dict(pid=pid,exit_code=child.returncode,same_parent_reaped=True,
            wait4_complete=True,wall_seconds=time.monotonic()-start,peak_RSS_bytes=usage.ru_maxrss*1024))
        c.require(child.returncode==0,'NONZERO_NATIVE_EXIT_NO_RETRY')
        c.authorize(spec,grant)
        result=c.read(attempt/'output/result.json');terminal=c.read(attempt/'output/terminal.json')
        c.require(result['status']==terminal['status']=='COMPLETE_MANDATORY_STOP' and result['stage']=='D'
            and result['SOURCE']==spec['SOURCE'] and result['binding']==c.old.BINDING
            and terminal['spec_digest']==c.digest(spec) and not (attempt/'output/STOP.json').exists(),'D_RECOVERY_COMPLETION')
        reserved={k:0 for k in c.CALLS}
        for p in sorted((attempt/'output').glob('reservation_*.json')):
            r=c.read(p);reserved[r['category']]+=r['count']
        c.require(reserved==result['calls_reserved']==c.CALLS,'EXACT_RECOVERY_CALLS')
        c.require(len(result['accounting']['rows'])==350 and len(result['measurements'])==28,'D70_ACCOUNTING_COVERAGE')
        c.save(attempt/'completion_audit.json',dict(status='COMPLETE_MANDATORY_STOP',stage='D',SOURCE=spec['SOURCE'],
            result_sha256=c.sha(attempt/'output/result.json'),native_exit_code=0,same_parent_reaped=True,
            wait4_complete=True,calls_reserved=reserved,old_attempt_preserved=True,old_reservation_refunds=0,
            independent_scientific_review=False,automatic_retry=False,next_stage_authorized=False))
        return 0
    except BaseException as exc:
        c.save(attempt/'driver_STOP.json',dict(reason=type(exc).__name__+':'+str(exc),automatic_retry=False,
            next_stage_authorized=False,native_completion_PASS_not_implied=True))
        return 2


if __name__=='__main__':raise SystemExit(main())
