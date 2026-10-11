#!/usr/bin/env python3
"""One authorized batch, no overwrite/resume/retry. STOP on any terminal result."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['QISKIT_PARALLEL']='FALSE'
os.environ['MPLCONFIGDIR']='/tmp/prt-n1-recovery-mpl'
Path(os.environ['MPLCONFIGDIR']).mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
PLAN='docs/research/n1_decode_recovery_v1_execution_plan_20261011.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(path,value):
    data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(data)>32*1024**2:raise ValueError('32 MiB JSON shard cap')
    path.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit('One batch only: existing output refused')
    if out!=ROOT/'artifacts/n1_decode_recovery_v1/2026-10-11/run1':raise SystemExit('frozen run1 path required')
    plan=json.loads((ROOT/PLAN).read_bytes())
    if not plan['this_batch_authorized'] or plan['next_scientific_stage_authorized']:raise SystemExit('authorization contract')
    origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
    if 'HIROMU1015/Partially-Randomized-Trotter' not in origin:raise SystemExit('repository safety')
    for p,s in plan['source_input_sha256'].items():
        if sha(ROOT/p)!=s:raise SystemExit('frozen identity mismatch: '+p)
    subprocess.run(['git','diff','--quiet'],cwd=ROOT,check=True);subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='n1-decode-recovery-execution-20261011':raise SystemExit('frozen branch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise SystemExit('clean frozen execution worktree required')
    packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','qiskit','qiskit-nature','mpmath','pytest')}
    if any(packages[k]!=v for k,v in plan['required_packages'].items()):raise SystemExit('frozen package identity')
    mem=Path('/proc/meminfo').read_text();available=int(next(x.split()[1] for x in mem.splitlines() if x.startswith('MemAvailable:')))*1024
    if available<10*1024**3:raise SystemExit('aggregate worker memory unavailable')
    out.mkdir(parents=True);resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,32*1024**2));start=time.monotonic();status='TECHNICAL_STOP';code=1
    receipt={'schema_version':1,'source_commit':plan['source_commit'],'execution_commit':head,'branch':branch,'plan_sha256':sha(ROOT/PLAN),'input_sha256':plan['source_input_sha256'],
       'authorization_sha256':plan['authorization_sha256'],'python':sys.version,'platform':platform.platform(),'packages':packages,
       'resources':plan['resources'],'mem_available_before_bytes':available,'cpu_online':os.cpu_count(),
       'threads':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_PARALLEL')},
       'scientific_attempt':1,'scientific_retries':0,'new_molecular_loads':0,'gpu_calls':0,'quantum_shots':0}
    save(out/'launch_receipt.json',receipt)
    try:
        from trottertracks.representation_exploration.n1_decode_recovery_v1 import load_contexts,run_n1,preflight
        save(out/'all_context_preflight.json',preflight(ROOT/'artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json'));print('all contexts preflight complete; no PF/compile/truth',flush=True)
        contexts=load_contexts(ROOT/'artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json');completed=[]
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(run_n1,(c,str(out/c['context']))):c['context'] for c in contexts}
            for f in as_completed(futures):
                row=f.result();completed.append(row);save(out/'n1_checkpoints.json',completed);print('N1 completed',row['context'],row['status'],row['performance'],flush=True)
        requests=sum(x['performance']['wrapper_requests'] for x in completed);calls=sum(x['performance']['transpile_calls'] for x in completed)
        if requests>3072 or calls>3116:raise ValueError('aggregate scientific compile cap')
        status='UNRESOLVED_PF_CERTIFICATE' if any(x['status']=='UNRESOLVED_PF_CERTIFICATE' for x in completed) else 'N1_RECOVERY_COMPLETE_AWAITING_SCIENTIFIC_REVIEW'
        save(out/'result.json',{'status':status,'source_commit':plan['source_commit'],'execution_commit':head,'N1':sorted(completed,key=lambda x:x['context']),'N3_executions':0,'N2_executions':0,
            'wrapper_requests':requests,'transpile_calls':calls,'mandatory_stop':True,'next_scientific_stage_authorized':False,'central_theme':None,
            'certificate_scope':'ideal_gate_certificate_with_numerical_compile_validation',
            'scientific_retries':0,'new_conditions':0});code=0
    except Exception as e:
        if isinstance(e,AssertionError):status='CERTIFICATE_OR_SEMANTICS_STOP'
        save(out/'failure.json',{'status':status,'traceback':traceback.format_exc(),'source_commit':plan['source_commit'],'scientific_attempt':1,'scientific_retries':0,'mandatory_stop':True});traceback.print_exc()
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        outputs={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file()}
        total=sum(x['bytes'] for x in outputs.values())
        if total>512*1024**2:status='OUTPUT_CAP_STOP';code=1
        save(out/'run_audit.json',dict(receipt,status=status,wall_seconds=time.monotonic()-start,parent_cpu_seconds=u.ru_utime+u.ru_stime,
          children_cpu_seconds=children.ru_utime+children.ru_stime,parent_peak_rss_bytes=u.ru_maxrss*1024,
          child_max_peak_rss_bytes=children.ru_maxrss*1024,outputs=outputs,total_output_bytes=total,
          mandatory_stop=True,next_scientific_stage_authorized=False,central_theme=None))
        save(out/'MANDATORY_STOP.json',{'status':status,'mandatory_stop':True,'next_scientific_stage_authorized':False,'central_theme':None,'scientific_retries':0})
        print(status,'mandatory STOP',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
