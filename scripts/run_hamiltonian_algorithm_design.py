#!/usr/bin/env python3
"""One frozen finite-construction batch; spawn two workers, BLAS/Qiskit threads one."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
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
os.environ.setdefault('MPLCONFIGDIR','/tmp/prt-representation-matplotlib')
sys.path.insert(0,str(ROOT/'src'))
CAP=64*1024**2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if len(data)>CAP:raise ValueError('64 MiB per-file cap')
    path.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit('No overwrite/resume; existing output refused')
    if not out.is_relative_to(ROOT/'artifacts/hamiltonian_algorithm_design'):
        raise SystemExit('dedicated output namespace required')
    subprocess.run(['git','diff','--quiet'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=subprocess.check_output(['git','ls-files','-z','src','scripts/run_hamiltonian_algorithm_design.py',
          'scripts/verify_hamiltonian_algorithm_design.py','tests/test_hamiltonian_algorithm_design.py',
          'docs/research/hamiltonian_algorithm_design_scope.md','docs/research/hamiltonian_construction_inputs'],cwd=ROOT).decode().split('\0')
    hashes={p:sha(ROOT/p) for p in paths if p and (ROOT/p).is_file()}
    out.mkdir(parents=True);resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    start=time.monotonic();status='TECHNICAL_STOP';code=1
    try:
        from trottertracks.representation_exploration.algorithm_design import TASKS,run_job
        results=[]
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            for task,result in zip(TASKS,pool.map(run_job,TASKS),strict=True):
                save(out/('checkpoint_'+task[0]+'_'+task[1]+'.json'),result)
                results.append(result)
                print('completed',*task,'IR',len(result['native_ir']),flush=True)
        ir=[]
        for r in results:ir.extend(r.pop('native_ir'))
        if len(ir)>768:raise ValueError('aggregate compilation cap')
        result={'schema_version':1,'status':'FINITE_CONSTRUCTION_FEASIBILITY_COMPLETE_AWAITING_GPT_REVIEW',
                'source_commit':head,'contexts':results,'compiled_circuits':len(ir),
                'next_stage_authorized':False,'central_hypothesis_adopted':None,'mandatory_stop':True,
                'scope':'fixed synthetic small-system development; no molecule/CI/novelty/FT or final RPE claim'}
        save(out/'result.json',result);save(out/'native_ir.json',ir)
        status=result['status'];code=0;print(status,'compiles',len(ir),flush=True)
    except Exception:
        save(out/'failure.json',{'status':status,'source_commit':head,'traceback':traceback.format_exc()})
        traceback.print_exc()
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        save(out/'run_audit.json',{'schema_version':1,'status':status,'source_commit':head,'source_sha256':hashes,
             'base_commit':'f98050e9402e8c65bb0f3dd27c7bd30d12fe7069','command':[sys.executable,*sys.argv],
             'python':sys.version,'platform':platform.platform(),'packages':{x.metadata['Name']:x.version for x in importlib.metadata.distributions()},
             'wall_seconds':time.monotonic()-start,'parent_cpu_seconds':usage.ru_utime+usage.ru_stime,
             'children_cpu_seconds':children.ru_utime+children.ru_stime,'parent_peak_rss_bytes':usage.ru_maxrss*1024,
             'resources':{'workers':2,'worker_address_space_bytes':4*1024**3,'aggregate_worker_address_space_budget':8*1024**3,
                          'wall_cap':None,'cpu_time_cap':None,'file_cap_bytes':CAP,'aggregate_compile_cap':768,
                          'per_context_compile_cap':384,'qubit_cap':13,'native_gates_per_circuit':20000},
             'threads':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_PARALLEL')},
             'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()},
             'new_molecular_loads':0,'quantum_shots':0,'gpu_calls':0,'next_stage_authorized':False,'central_hypothesis_adopted':None})
    return code


if __name__=='__main__':raise SystemExit(main())
