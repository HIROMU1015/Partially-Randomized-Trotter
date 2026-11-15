"""Adopted LD7 prices: calibration8 paths, later independent holdout2 paths.

Reuse the exact prefix-capable e827 adapter/fast and existing path worker.
No historical source/input/raw/STOP is edited; no sampling retry.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import traceback
import run_pr_df_ld_split as R

ROOT=R.ROOT
PUBLIC='docs/research/d_pr_df_ld7_price_results_20261116'
CALLER='scripts/run_pr_df_ld7_prices.py'
PREVIOUS='0defa286c53a1b9acf513be453907bcad83102b3'
VDIR='docs/research/d_pr_df_ld7_v0_results_20261116'
REUSED='ea9e66092465d39dfcd4aeb6fea8e9f09808cabb'
PRICE_PYTHON='/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python'
read,write,blob,head=R.read,R.write,R.blob,R.head

def checked_source(source):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Unfixed LD7 caller')
    if blob(REUSED,'scripts/run_pr_df_ld_split.py')!=(ROOT/'scripts/run_pr_df_ld_split.py').read_bytes():raise ValueError('Reused path worker SOURCE differs')

def request(config,m,source):
    scalar=read(PREVIOUS,VDIR+'/generation/SCALAR.json')
    inputs=dict(configuration=dict(commit=PREVIOUS,path=VDIR+'/generation/CONFIGURATION.json'),
        tail=deepcopy(config['spec']['split']['lambda_source']),scalar=dict(commit=PREVIOUS,path=VDIR+'/generation/SCALAR.json'))
    return R.D.from_b(config,m,config['rounds'][m]['physical_candidate'],input_source=inputs,
        execution_source=dict(commit=source,path=CALLER),expected_scalar=scalar['total'])

def prepare(source):
    checked_source(source)
    config=read(PREVIOUS,VDIR+'/generation/CONFIGURATION.json');gate=read(PREVIOUS,VDIR+'/generation/SCHEDULE_GATE.json')
    if config['spec']['L_D']!=7 or gate['status']!='PASS'or any(r['R_outer']!=1 or r['structure']['allocation']!=[1]for r in config['rounds']):raise ValueError('LD7 domain')
    saved=read('e0b3cad2a024070389f757160f3a20d7da02c5ff','docs/research/d_pr_df_ld_split_results_20261115/ACQUISITION_INPUTS.json')
    seed=R.P.trial_seed_function(ROOT);cells=[]
    for m,n,role,cohort in ((0,4,'CALIBRATION','D_PR_DF_LD7_PRICE_CAL_20261116_V1'),(6,4,'CALIBRATION','D_PR_DF_LD7_PRICE_CAL_20261116_V1'),(3,2,'HOLDOUT','D_PR_DF_LD7_M3_PRICE_HOLDOUT_20261116_V1')):
        req=request(config,m,source);c=req['physical_context']['physical_candidate']
        seeds=[[seed(c,cohort,i,j,0)for j in range(c['q'])]for i in range(n)]
        cells.append(dict(m=m,request=req,role=role,cohort=cohort,indices=list(range(n)),seeds=seeds))
    seeds=[s for c in cells for ss in c['seeds']for s in ss]
    if len(set(seeds))!=len(seeds):raise ValueError('Calibration/holdout seed overlap')
    write(ROOT/PUBLIC/'ACQUISITION_INPUTS.json',dict(source=source,caller=CALLER,cells=cells,runtime=saved['runtime'],price_python=PRICE_PYTHON,
        saved_input_files=saved['saved_input_files'],local_support=saved['local_support'],reused_worker=REUSED,
        maximum_paths=10,maximum_compiles=20,calibration_compiles=16,holdout_compiles=4,
        full_wrapper_statevector_validation=False,numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',
        sequence='calibration raw publication -> frozen two-axis model and26 positive references publication -> holdout'))

def acquire(stage):
    inputs=json.loads((ROOT/PUBLIC/'ACQUISITION_INPUTS.json').read_text());source=inputs['source'];checked_source(source)
    delta=json.loads((ROOT/PUBLIC/'DELTA_CHECK.json').read_text())
    if delta['equivalent_results']or delta['matching_jobs']or delta['unresolved']:raise ValueError('Reconcile duplicate/unknown work')
    for p,h in inputs['local_support'].items():
        if R.sha(ROOT/p)!=h:raise ValueError('Support source mismatch:'+p)
    config=read(PREVIOUS,VDIR+'/generation/CONFIGURATION.json')
    cells=[c for c in inputs['cells']if c['role']==('CALIBRATION'if stage=='calibration'else'HOLDOUT')]
    for c in cells:
        if c['request']!=request(config,c['m'],source):raise ValueError('Fixed request differs')
    freeze=None
    if stage=='holdout':
        freeze=json.loads((ROOT/PUBLIC/'HOLDOUT_INPUTS.json').read_text())
        model=read(freeze['model_commit'],PUBLIC+'/FROZEN_MODELS.json');prices=read(freeze['model_commit'],PUBLIC+'/AXIS_PRICES.json')
        if model['status']!='FROZEN_2_AXIS_MODELS'or len(prices['rows'])!=26 or any(R.dec(x['C_ref'])<=0 for x in prices['rows']):raise ValueError('Frozen price prerequisite')
    # The reused worker intentionally receives only owned process-local output/state.
    R.PUBLIC=PUBLIC+'/'+stage
    runtime_inputs=dict(inputs,cache_root=str(ROOT/PUBLIC/stage/'isolated_cache'))
    out=ROOT/R.PUBLIC/'acquisition';out.mkdir(parents=True)
    write(out/'EXECUTION.json',dict(source=source,inputs_commit=head(),stage=stage,freeze=freeze,started=time.time()))
    start=time.perf_counter();rows=[];processes=[]
    try:
        ham,md=R.restore(cells[0]['request'],inputs['saved_input_files'],inputs['runtime'])
        runtimes={};contexts={}
        for cell in cells:
            m=cell['m'];pc=cell['request']['physical_context'];r=R.D.NativeRuntime(ROOT,ham,binding=pc['binding'],input_source=pc['input_source'],execution_source=pc['execution_source'])
            runtimes[m]=r;contexts[m]=r.context(cell['request']);c=contexts[m]
            if list(c.prep.deterministic_fragment_indices)!=list(range(7))or list(c.prep.randomized_block_indices)!=[7,8,9]:raise ValueError('Actual preparation prefix7 mismatch')
        write(out/'PREPARATION.json',dict(input_identity=md,contexts={str(m):dict(preparation_hash=c.prep.preparation_hash,scalar=c.scalar,scalar_hex=float(c.scalar).hex(),
            lambda_R=c.prep.rte_preparation.symbolic_tail.lambda_r,head=list(c.prep.deterministic_fragment_indices),tail=list(c.prep.randomized_block_indices))for m,c in contexts.items()}))
        R.STATE.update(inputs=runtime_inputs,runtimes=runtimes,contexts=contexts)
        context=mp.get_context('fork');flag=context.Event();jobs=[(c,i)for c in cells for i in c['indices']]
        with ProcessPoolExecutor(max_workers=len(jobs),mp_context=context,initializer=R.init_price,initargs=(flag,))as pool:
            pending={pool.submit(R.acquire_path,job):(job[0]['m'],job[1])for job in jobs};processes=list(pool._processes.values())
            while pending:
                done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    job=pending.pop(f)
                    try:r=f.result()
                    except BaseException:flag.set();r=dict(m=job[0],index=job[1],status='WORKER_ERROR',error=traceback.format_exc(),compiles=[])
                    rows.append(r);write(out/f'SUPERVISOR_m{job[0]}_p{job[1]}.json',r);print('PATH',job,r['status'],flush=True)
                if pending:print('ACTIVE',list(pending.values()),flush=True)
        good=len(rows)==len(jobs)and all(r['status']=='VALID'for r in rows)and all(p.exitcode==0 for p in processes)
        write(out/'RESULT.json',dict(status='ACQUIRED'if good else'STOP_PARTIAL',rows=sorted(rows,key=lambda r:(r['m'],r['index'])),
            new_paths=sum(r['status']=='VALID'for r in rows),compile_attempts=sum(len(r['compiles'])for r in rows),
            completed_compiles=sum(c['status']=='COMPLETE'for r in rows for c in r['compiles']),wall_seconds=time.perf_counter()-start,
            process_exitcodes=[dict(pid=p.pid,exitcode=p.exitcode)for p in processes],source=source,input_commit=head(),mandatory_stop=not good))
        return 0 if good else 2
    except BaseException:
        write(out/'STOP.json',dict(error=traceback.format_exc(),rows=rows,wall_seconds=time.perf_counter()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','calibration','holdout'));p.add_argument('--source');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    else:raise SystemExit(acquire(a.mode))
