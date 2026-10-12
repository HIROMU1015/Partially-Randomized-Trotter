#!/usr/bin/env python3
"""Default identity-only preflight; separate explicit saved-data posthoc authorization required."""
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
from trottertracks.representation_exploration.n3_saved_attribution_execution_v1 import (
    BRANCH,OUTPUT,PLAN,PREPARATION,RESOURCES,THREADS,Output,authorization,preflight,sha)


def worker(case,relative):
    resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['worker_AS_bytes'],RESOURCES['worker_AS_bytes']))
    from trottertracks.representation_exploration.n3_saved_attribution_v1 import analyze
    record=json.loads((ROOT/relative).read_bytes())
    result=analyze(case,record,relative)
    if len(result['candidates'])>RESOURCES['maximum_candidates']:raise ArithmeticError('CANDIDATE_COUNT_CAP')
    if any(k in sys.modules for k in ('numpy','scipy','cvxpy','scs')):raise ValueError('NUMERICAL_LIBRARY_IMPORTED')
    return result


def execute(plan,authpath):
    preflight(ROOT,plan);auth=json.loads(authpath.read_bytes());authorization(auth,plan,sha(ROOT/PLAN))
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('CLEAN_EXECUTION_WORKTREE')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=BRANCH:raise ValueError('EXECUTION_BRANCH')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
    if remote not in ('git@github.com:HIROMU1015/Partially-Randomized-Trotter.git','https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'):
        raise ValueError('REPOSITORY_OWNER')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_commit'],'HEAD'],cwd=ROOT,check=True)
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if available<RESOURCES['minimum_available_bytes']:raise ValueError('MEMORY_AVAILABILITY')
    for k in THREADS:os.environ[k]='1'
    out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=False);writer=Output(out);start=time.perf_counter();failure=None;done=[]
    writer.write('launch_receipt.json',{'source_commit':plan['source_commit'],'source_fingerprint':plan['source_fingerprint'],
        'registry_sha256':plan['registry_sha256'],'plan_sha256':sha(ROOT/PLAN),'authorization_sha256':sha(authpath),
        'attempt':1,'retries':0,'resources':RESOURCES})
    log=(out/'run.log').open('xb');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(RESOURCES['log_bytes'],RESOURCES['log_bytes']))
    from trottertracks.representation_exploration.n3_width_execution_v2 import MemoryWatch
    from trottertracks.representation_exploration.n3_saved_attribution_v1 import recertify_connection
    watch=MemoryWatch();watch.start()
    registry=json.loads((ROOT/PREPARATION/'saved_input_registry.json').read_bytes())
    cases=json.loads((ROOT/registry['input_path']).read_bytes())['cases']
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs={pool.submit(worker,c,registry['raw_prefix']+'/'+c['id']+'.json'):c['id'] for c in cases}
            for job in as_completed(jobs):
                name=jobs[job];writer.write(name+'.json',job.result());done.append(name)
        proof=recertify_connection(json.loads((ROOT/PREPARATION/'SOS_OCC_existing_proof_connection.json').read_bytes()))
        writer.write('selected_existing_proof_recheck.json',proof)
        if watch.violation:raise MemoryError('AGGREGATE_RSS')
    except Exception as ex:
        failure={'type':type(ex).__name__,'message':str(ex)[:4096],'traceback':traceback.format_exc()[-24000:]}
    finally:watch.stop()
    good=failure is None and len(done)==6
    status='SAVED_POSTHOC_COMPLETE_AWAITING_SCIENTIFIC_JUDGEMENT' if good else 'POSTHOC_TECHNICAL_OR_SOUNDNESS_STOP'
    writer.write('result.json',{'status':status,'completed_cases':done,'failure':failure,'wall_seconds':time.perf_counter()-start,
        'memory':watch.report(),'new_producer_calls':0,'new_solver_calls':0,'new_certificate_searches':0,
        'retries':0,'mandatory_stop':True,'next_stage_authorized':False,'central_theme':None},stop=True)
    writer.write('MANDATORY_STOP.json',{'status':status,'mandatory_stop':True,'next_stage_authorized':False,'retries':0},stop=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--authorization',type=Path)
    args=parser.parse_args();plan=json.loads((ROOT/PLAN).read_bytes())
    if args.execute:
        if args.authorization is None:parser.error('Separate explicit numeric posthoc authorization required')
        execute(plan,args.authorization)
    else:print(json.dumps(preflight(ROOT,plan),indent=2))


if __name__=='__main__':main()
