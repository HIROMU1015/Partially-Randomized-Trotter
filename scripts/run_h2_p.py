"""H2 P driver and owned worker. A separate P science grant is mandatory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import h2_p_contract as c


def claim_attempt(spec,grant):
    c.authorize(spec,grant)
    attempt=c.owned(spec['attempt'])
    c.require(not attempt.exists(),'P_ONE_SHOT_CONSUMED_NO_RETRY')
    attempt.mkdir(parents=True,exist_ok=False)
    c.save(attempt/'ONE_SHOT.json',dict(parent_pid=os.getpid(),SOURCE=spec['SOURCE'],
        spec_digest=c.digest(spec),grant_digest=c.digest(grant),nonrefundable=True))
    return attempt


def worker(spec,grant):
    c.authorize(spec,grant);attempt=c.owned(spec['attempt']);marker=c.read(attempt/'ONE_SHOT.json')
    c.require(marker['parent_pid']==os.getppid() and marker['SOURCE']==spec['SOURCE']
        and marker['spec_digest']==c.digest(spec) and marker['grant_digest']==c.digest(grant),'OWNED_P_WORKER')
    output=attempt/'output';output.mkdir(exist_ok=False);code=2;started=time.monotonic()
    try:
        scratch=output/'scratch';scratch.mkdir()
        for key,name in [('TMPDIR','tmp'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','cache')]:
            path=scratch/name;path.mkdir();os.environ[key]=str(path)
        import tempfile
        tempfile.tempdir=os.environ['TMPDIR']
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):os.environ[key]='1'
        import h2_p_science
        result=h2_p_science.execute(spec,grant,output);c.authorize(spec,grant)
        c.save(output/'result.json',result);code=0
    except BaseException as exc:
        c.save(output/'STOP.json',dict(reason=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc(),
            automatic_retry=False,next_stage_authorized=False,reservations_preserved=True))
    c.save(output/'terminal.json',dict(stage='P',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),exit_code=code,
        status='COMPLETE_MANDATORY_STOP' if code==0 else 'STOP',wall_seconds=time.monotonic()-started,
        automatic_retry=False,next_stage_authorized=False))
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--grant',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);spec,grant=c.read(args.spec),c.read(args.grant)
    c.authorize(spec,grant)
    if args.worker:return worker(spec,grant)
    attempt=claim_attempt(spec,grant)
    c.save(attempt/'resource_observation.json',c.base.f.observation())
    command=[spec['python'],'-B',str(Path(__file__).resolve()),'--spec',str(args.spec.resolve()),
        '--grant',str(args.grant.resolve()),'--worker']
    started=time.monotonic()
    try:
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
            c.save(attempt/'native_started.json',dict(pid=child.pid,parent_pid=os.getpid(),command=command))
            pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
        c.save(attempt/'native_exit.json',dict(pid=pid,exit_code=child.returncode,same_parent_reaped=True,
            wait4_complete=True,wall_seconds=time.monotonic()-started,peak_RSS_bytes=usage.ru_maxrss*1024))
        c.require(child.returncode==0,'NONZERO_P_EXIT_NO_RETRY');c.authorize(spec,grant)
        result=c.read(attempt/'output/result.json');terminal=c.read(attempt/'output/terminal.json')
        c.require(result['stage']=='P' and result['SOURCE']==spec['SOURCE'] and result['binding']==c.base.BINDING
            and result['status']==terminal['status']=='COMPLETE_MANDATORY_STOP'
            and terminal['spec_digest']==result['spec_digest']==c.digest(spec)
            and not (attempt/'output/STOP.json').exists(),'P_NATIVE_COMPLETION')
        c.check_measurements(result['measurements'],c.read(spec['seed_registry']),spec['endpoints'])
        reserved={k:0 for k in c.CALLS}
        for p in sorted((attempt/'output').glob('reservation_*.json')):
            r=c.read(p);c.require(r['nonrefundable'],'NONREFUNDABLE_P_CALLS');reserved[r['category']]+=r['count']
        c.require(reserved==result['calls_reserved']==c.CALLS,'P_COMPLETE_CALL_LEDGER')
        c.save(attempt/'completion_audit.json',dict(status='COMPLETE_MANDATORY_STOP',stage='P',SOURCE=spec['SOURCE'],
            result_sha256=c.sha(attempt/'output/result.json'),native_exit_code=0,same_parent_reaped=True,
            wait4_complete=True,calls_reserved=reserved,independent_scientific_review=False,
            automatic_retry=False,next_stage_authorized=False))
        return 0
    except BaseException as exc:
        c.save(attempt/'driver_STOP.json',dict(reason=type(exc).__name__+':'+str(exc),automatic_retry=False,
            next_stage_authorized=False,native_completion_PASS_not_implied=True))
        return 2


if __name__=='__main__':raise SystemExit(main())
