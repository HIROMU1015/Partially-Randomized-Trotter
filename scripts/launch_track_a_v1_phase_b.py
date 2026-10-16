"""Sealed Phase B only; reuse the tested owned-process external observer."""
import argparse
import os
from pathlib import Path
import subprocess

import track_a_v1_runner as runner
from launch_track_a_v1_phase_a import supervise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','grant','fresh-gate','control'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();c=runner.read_json(a.config);g=runner.read_json(a.grant)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runner.ROOT,text=True).strip()
    runner.verify_grant(c,g,config_sha256=runner.sha(a.config),source=source,fresh=runner.read_json(a.fresh_gate))
    if c.get('phase_b') is not True:raise ValueError('PHASE_B_ONLY')
    for name,h in c['source_hashes'].items():
        if runner.sha(runner.ROOT/name)!=h:raise ValueError('LAUNCHER_SOURCE_CHANGED')
    for path in (a.config,a.grant,a.fresh_gate,a.control):
        if not path.is_absolute() or path.resolve()!=path or not path.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('PRIVATE_ABSOLUTE_PATH_REQUIRED')
    scope='compile_validation';output=c['outputs'][scope]
    if str(a.control)!=c['control'] or a.control.exists() or Path(output).exists() or Path(c['locks'][scope]).exists():
        raise ValueError('ONE_SHOT_ALREADY_SPENT_OR_CONTROL_CHANGED')
    fs=os.statvfs(a.control.parent)
    if fs.f_bavail*fs.f_frsize<c['resources']['stage_storage_bytes']+2*2**30 or fs.f_favail<4096:
        raise ValueError('FRESH_STORAGE_GATE')
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
              'NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):env[k]='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    command=[c['python'],'-B',str(runner.ROOT/'scripts/track_a_v1_runner.py'),'--execute-approved',
        '--config',str(a.config),'--grant',str(a.grant),'--fresh-gate',str(a.fresh_gate),'--output',output]
    old=os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0,{g['monitor_cpu']})
        supervise(command,env=env,control=a.control,output=output,limits=g['resources'],allowed_cpus=g['allowed_cpus'])
    finally:os.sched_setaffinity(0,old)


if __name__=='__main__':main()
