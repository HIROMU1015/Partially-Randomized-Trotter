#!/usr/bin/env python3
"""One separately authorized N3 batch; no resume, retry or overwrite."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0,str(ROOT/'src'))
PLAN='docs/research/n3_direction_generator_v1_execution_plan_20261011.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>32*1024**2:raise ValueError('JSON shard cap')
    path.write_bytes(raw)
def worker(case):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    from trottertracks.representation_exploration.n3_direction_generator_v1 import generate_case
    return generate_case(case)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    if out!=ROOT/'artifacts/n3_direction_generator_v1/2026-10-11/run1' or out.exists():raise SystemExit('fixed new run1 only; no retry/resume/overwrite')
    plan=json.loads((ROOT/PLAN).read_bytes())
    if not plan['this_batch_authorized'] or plan['next_scientific_stage_authorized']:raise SystemExit('authorization')
    if subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()!='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git':raise SystemExit('repository safety')
    for path,s in plan['source_input_sha256'].items():
        if sha(ROOT/path)!=s:raise SystemExit('frozen identity '+path)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise SystemExit('clean frozen execution worktree required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='n3-generator-transfer-execution-20261011':raise SystemExit('fixed branch')
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if mem<10*1024**3:raise SystemExit('memory availability')
    out.mkdir(parents=True);resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,32*1024**2));start=time.monotonic();status='TECHNICAL_STOP';code=1
    receipt={'source_commit':plan['source_commit'],'execution_commit':head,'branch':branch,'plan_sha256':sha(ROOT/PLAN),'source_input_sha256':plan['source_input_sha256'],'authorization_sha256':plan['authorization_sha256'],'scientific_attempt':1,'scientific_retries':0,'resources':plan['resources'],'cpu_online':os.cpu_count(),'mem_available_before_bytes':mem,'thread_environment':{k:os.environ[k] for k in plan['resources']['thread_environment']},'gpu_calls':0,'quantum_shots':0}
    save(out/'launch_receipt.json',receipt)
    try:
        from trottertracks.representation_exploration.n3_direction_generator_v1 import INPUT_SHA,diagnose_case
        inputpath=ROOT/'artifacts/n3_generator_n1_recovery/2026-10-11/preparation/n3_transfer_inputs.json'
        if sha(inputpath)!=INPUT_SHA:raise ValueError('input pack identity')
        cases=json.loads(inputpath.read_bytes())['cases'];records={}
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(worker,c):c['id'] for c in cases}
            for f in as_completed(futures):
                row=f.result();records[row['input_id']]=row;save(out/'generation_checkpoint.json',records);print('certificate generation completed',row['input_id'],flush=True)
        ledger=[records[c['id']] for c in cases];save(out/'certificate_ledger.json',ledger);seal=sha(out/'certificate_ledger.json');save(out/'ledger_seal.json',{'sha256':seal,'before_many_body_diagnostics':True,'cases':12,'method_records':72})
        print('all certificate records frozen before truth',seal,flush=True)
        diagnostics=[]
        for case,row in zip(cases,ledger):
            if sha(out/'certificate_ledger.json')!=seal:raise AssertionError('SOUNDNESS_STOP: changed ledger')
            diagnostics.append(diagnose_case(case,row))
        save(out/'diagnostics.json',{'certificate_ledger_sha256':seal,'cases':diagnostics,'ground_eigenvalue_calls':24})
        accepts=[{m['method']:m['accepted'] for m in row['methods']} for row in ledger]
        groups=sorted({c['base'] for c,x in zip(cases,accepts) if x['direct_one_variable'] and not x['common_norm_only']});gains=[c['id'] for c,x in zip(cases,accepts) if x['direct_one_variable'] and not x['five_point_common']]
        status='GENERATED_TRANSFER_GAIN' if len(groups)>=2 and gains else 'TRANSFER_CERTIFIED_NO_NEW_GRID_GAIN' if len(groups)>=2 else 'NO_TRANSFER_ADVANTAGE_IN_FROZEN_SCOPE'
        save(out/'result.json',{'status':status,'source_commit':plan['source_commit'],'execution_commit':head,'certificate_ledger_sha256':seal,'accepted_counts':{name:sum(x[name] for x in accepts) for name in accepts[0]},'direct_vs_norm_base_groups':groups,'direct_vs_common_five_point_cases':gains,'mandatory_stop':True,'next_scientific_stage_authorized':False,'central_theme':None,'scientific_attempts':1,'scientific_retries':0,'method_records':72,'ground_eigenvalue_calls':24});code=0
    except Exception as e:
        if isinstance(e,AssertionError) and 'SOUNDNESS' in str(e):status='SOUNDNESS_STOP'
        save(out/'failure.json',{'status':status,'traceback':traceback.format_exc(),'source_commit':plan['source_commit'],'scientific_attempt':1,'scientific_retries':0,'mandatory_stop':True});traceback.print_exc()
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF);v=resource.getrusage(resource.RUSAGE_CHILDREN);outputs={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file()};total=sum(x['bytes'] for x in outputs.values())
        if total>512*1024**2:status='TECHNICAL_STOP';code=1
        save(out/'run_audit.json',dict(receipt,status=status,wall_seconds=time.monotonic()-start,parent_cpu_seconds=u.ru_utime+u.ru_stime,children_cpu_seconds=v.ru_utime+v.ru_stime,parent_peak_rss_bytes=u.ru_maxrss*1024,child_max_peak_rss_bytes=v.ru_maxrss*1024,outputs=outputs,total_output_bytes=total,mandatory_stop=True,next_scientific_stage_authorized=False,central_theme=None))
        save(out/'MANDATORY_STOP.json',{'status':status,'mandatory_stop':True,'next_scientific_stage_authorized':False,'central_theme':None,'scientific_retries':0});print(status,'mandatory STOP',flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
