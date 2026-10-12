#!/usr/bin/env python3
"""Default identity-only preflight. Science launch requires separate frozen authorization."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.representation_exploration.n3_width_execution_v2 import (
    EXECUTION_BRANCH,OUTPUT_PATH,RESOURCES,THREADS,MemoryWatch,Output,file_sha,preflight,validate_authorization,
)
PLAN='docs/research/n3_width_comparison_v2_execution_plan_20261012.json'
for key in THREADS:os.environ[key]='1'


def worker(case,source_fingerprint='synthetic'):
    resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['worker_AS_bytes'],RESOURCES['worker_AS_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(RESOURCES['log_bytes'],RESOURCES['log_bytes']))
    # Dependency imports are environment setup, excluded consistently from method timings.
    startup=time.perf_counter()
    import cvxpy,numpy,scs
    from trottertracks.representation_exploration.n3_width_pipeline_v2 import acquire_case
    setup=time.perf_counter()-startup;record=acquire_case(case);record['worker_dependency_setup_wall_seconds']=setup
    record['source_fingerprint']=source_fingerprint
    return record


def execute(plan,auth_path):
    preflight(ROOT,plan);auth=json.loads(auth_path.read_bytes());validate_authorization(auth,plan,file_sha(ROOT/PLAN))
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('CLEAN_EXECUTION_WORKTREE')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=EXECUTION_BRANCH:raise ValueError('EXECUTION_BRANCH')
    if subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip() not in (
        'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git','https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'):
        raise ValueError('REPOSITORY_OWNER')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_commit'],'HEAD'],cwd=ROOT,check=True)
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if available<RESOURCES['minimum_available_bytes']:raise ValueError('MEMORY_AVAILABILITY')
    out=ROOT/OUTPUT_PATH;out.mkdir(parents=True,exist_ok=False);writer=Output(out);watch=MemoryWatch();started=time.perf_counter();records={};failure=None
    writer.write('launch_receipt.json',{'source_commit':plan['source_commit'],'source_fingerprint':plan['source_fingerprint'],
        'plan_sha256':file_sha(ROOT/PLAN),'input_sha256':plan['input_sha256'],'authorization_sha256':file_sha(auth_path),
        'attempt':1,'retries':0,'resources':RESOURCES,'threads':{k:os.environ[k] for k in THREADS}})
    # Spawned workers inherit standard descriptors. Hard per-file FSIZE bounds
    # include native-library stdout/stderr, not just Python logging.
    log=(out/'run.log').open('xb');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(RESOURCES['log_bytes'],RESOURCES['log_bytes']))
    cases=json.loads((ROOT/plan['input_path']).read_bytes())['cases'];watch.start()
    try:
        with ProcessPoolExecutor(max_workers=RESOURCES['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs={pool.submit(worker,c,plan['source_fingerprint']):c['id'] for c in cases}
            for job in as_completed(jobs):
                name=jobs[job];record=job.result();writer.write(name+'.json',record);records[name]=record['status']
        if watch.violation:raise MemoryError('AGGREGATE_RSS_BUDGET')
    except Exception as ex:failure={'type':type(ex).__name__,'message':str(ex)[:4096],'traceback':traceback.format_exc()[-24000:]}
    finally:watch.stop()
    good=failure is None and len(records)==len(cases) and all(s=='FINITE_COMPARISON_COMPLETE' for s in records.values())
    status='FINITE_COMPARISON_COMPLETE_AWAITING_SCIENTIFIC_REVIEW' if good else 'TECHNICAL_OR_SOUNDNESS_STOP'
    writer.write('result.json',{'status':status,'case_statuses':records,'input_order':plan['input_ids'],'failure':failure,
        'wall_seconds':time.perf_counter()-started,'memory':watch.report(),'output_bytes_before_summary':writer.total,
        'next_stage_authorized':False,'mandatory_stop':True,'central_theme':None},stop_record=True)
    writer.write('MANDATORY_STOP.json',{'status':status,'mandatory_stop':True,'next_stage_authorized':False,'central_theme':None,'retry':0},stop_record=True)
    print(status)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--authorization',type=Path);args=p.parse_args()
    plan=json.loads((ROOT/PLAN).read_bytes())
    if args.execute:
        if args.authorization is None:p.error('Separate explicit scientific execution authorization required')
        execute(plan,args.authorization)
    else:print(json.dumps(preflight(ROOT,plan),indent=2))


if __name__=='__main__':main()
