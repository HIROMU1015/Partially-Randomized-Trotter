"""Observe and reap only Popen children created by this diagnostic."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time


def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    c=json.loads(Path(args.config).read_text());area=Path(c['source_root']).parent;out=Path(c['output'])
    out.mkdir(exist_ok=False);(out/'one_shot.lock').write_text(c['SOURCE']+'\n')
    def observe():
        m={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','MemTotal')}
        v=os.statvfs(out)
        return dict(utc=datetime.now(timezone.utc).isoformat(),memory=m,
            host_pressure=Path('/proc/pressure/memory').read_text(),
            available_bytes=v.f_bavail*v.f_frsize,available_inodes=v.f_favail,
            affinity_available=sorted(os.sched_getaffinity(0)))
    fresh=observe();write(out/'admission_PRIVATE.json',fresh)
    # Prospective requirement based on six small diagnostic workers, not a
    # persistent process limit. No resource setrlimit, affinity, or PSI kill.
    if fresh['memory']['MemAvailable']<12*2**30 or fresh['available_bytes']<2*2**30:
        raise RuntimeError('CURRENT_RESOURCES_INSUFFICIENT_FOR_THIS_DIAGNOSTIC')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1',NUMBA_DISABLE_CUDA='1',CUDA_VISIBLE_DEVICES='',
        QISKIT_PARALLEL='FALSE',QISKIT_NUM_PROCS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        MPLCONFIGDIR=str(area/'env/mpl'),XDG_CACHE_HOME=str(area/'env/cache'),
        NUMBA_CACHE_DIR=str(area/'env/numba'),TMPDIR=str(area/'tmp'))
    children=[];complete=[];start=time.monotonic()
    try:
        for group in c['plan']['groups']:
            log=(out/(group['id']+'.log')).open('xb')
            child=subprocess.Popen([c['python'],'-P','-B',str(Path(c['source_root'])/'scripts/run_h46_new2_native.py'),
                '--config',args.config,'--group',group['id']],cwd=c['source_root'],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((group['id'],child,log))
        write(out/'process_start_PRIVATE.json',dict(SOURCE=c['SOURCE'],children=[dict(group=g,pid=p.pid) for g,p,l in children]))
        with (out/'observation_PRIVATE.jsonl').open('x') as journal:
            last=0
            while any(p.poll() is None for g,p,l in children):
                states=[]
                for group,child,log in children:
                    row=dict(group=group,pid=child.pid,exit_code=child.poll())
                    try:
                        lines=Path('/proc',str(child.pid),'status').read_text().splitlines()
                        row.update({k:v.strip() for k,v in (l.split(':',1) for l in lines) if k in ('VmRSS','VmPeak','Threads','State')})
                    except FileNotFoundError:row['proc_status']='exited_or_reaped'
                    states.append(row)
                row=dict(elapsed=time.monotonic()-start,children=states,host=observe())
                journal.write(json.dumps(row)+'\n');journal.flush()
                if time.monotonic()-last>30:print(json.dumps(row),flush=True);last=time.monotonic()
                time.sleep(2)
    finally:
        for group,child,log in children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            code=child.wait();log.close()
            complete.append(dict(group=group,pid=child.pid,native_exit_code=code,native_wait_reaped=True))
        write(out/'completion_witness_PRIVATE.json',dict(SOURCE=c['SOURCE'],children=complete,
            all_owned_children_reaped=True,wall_seconds=time.monotonic()-start,final_observation=observe()))
    print(json.dumps(dict(completion=complete,wall_seconds=time.monotonic()-start)),flush=True)
    return 0 if all(r['native_exit_code']==0 for r in complete) else 1


if __name__=='__main__':raise SystemExit(main())
