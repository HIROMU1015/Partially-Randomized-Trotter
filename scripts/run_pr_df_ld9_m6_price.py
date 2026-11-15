"""One LD9 m6 cell, four independent normal paths and at most8 X/Y compiles.

Reuses LD7's from_b request/restore/NativeRuntime/Adapter/path worker. No fit,
other price point, signal, Q3 or total-cost entry point.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import run_pr_df_ld7_prices as L
R=L.R
ROOT=R.ROOT
PUBLIC='docs/research/d_pr_df_ld9_m6_price_20261116'
CALLER='scripts/run_pr_df_ld9_m6_price.py'
BASE='460d77d8f56ef0bde41dbddaa27852dd8b05513e'
CONFIGDIR='docs/research/d_pr_df_ld9_v0_results_20261116'
LD7='5c5d87ab32d03bb107a9a6bfed3cba253aef2aa0'
LD7DIR='docs/research/d_pr_df_ld7_price_results_20261116'
COHORT='D_PR_DF_LD9_M6_DIRECT_PRICE_20261116_V1'
read,write,blob,head=R.read,R.write,R.blob,R.head

def checked(source):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Unfixed acquisition SOURCE')
    if blob('8d4503e00a55d073c85624939762cbaa62c1ed56','scripts/run_pr_df_ld7_prices.py')!=(ROOT/'scripts/run_pr_df_ld7_prices.py').read_bytes():raise ValueError('Reused LD7 caller changed')
    if blob(L.REUSED,'scripts/run_pr_df_ld_split.py')!=(ROOT/'scripts/run_pr_df_ld_split.py').read_bytes():raise ValueError('Path worker SOURCE changed')

def request(config,source):
    # Generic helper, with only owned process-local input/SOURCE locators changed.
    L.PREVIOUS=BASE;L.VDIR=CONFIGDIR;L.CALLER=CALLER
    return L.request(config,6,source)

def prepare(source):
    checked(source);config=read(BASE,CONFIGDIR+'/generation/CONFIGURATION.json');req=request(config,source)
    c=req['physical_context']['physical_candidate']
    if config['spec']['L_D']!=9 or(c['id'],c['T'],c['q'],c['R_outer'])!=('09fa97187d6eee510d0c',6.4,64,1):raise ValueError('Fixed LD9 m6 physical mismatch')
    saved=read(LD7,LD7DIR+'/ACQUISITION_INPUTS.json');seed=R.P.trial_seed_function(ROOT)
    seeds=[[seed(c,COHORT,i,j,0)for j in range(64)]for i in range(4)]
    old=[v for v in saved['cells']if v['m']==6][0]
    if old['cohort']==COHORT or len({s for row in seeds for s in row})!=256 or {s for row in seeds for s in row}&{s for row in old['seeds']for s in row}:raise ValueError('Cohort/seed identity or overlap')
    write(ROOT/PUBLIC/'INPUTS.json',dict(source=source,cell=dict(m=6,request=req,cohort=COHORT,indices=list(range(4)),seeds=seeds),
        runtime=saved['runtime'],saved_input_files=saved['saved_input_files'],local_support=saved['local_support'],price_python=L.PRICE_PYTHON,
        cache_root=str(ROOT/PUBLIC/'isolated_cache'),maximum_paths=4,maximum_compiles=8,fit=0,new_signal=0,new_Q3=0,LD9_total_model_cost=None,
        LD7_reference=dict(commit=LD7,path=LD7DIR+'/calibration/acquisition/RESULT.json',selector=['rows'],m=6),
        full_wrapper_statevector_validation=False,numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH'))

def acquire():
    inputs=json.loads((ROOT/PUBLIC/'INPUTS.json').read_text());source=inputs['source'];checked(source)
    d=json.loads((ROOT/PUBLIC/'DELTA_CHECK.json').read_text())
    if d['equivalent_results']or d['matching_jobs']or d['unresolved']:raise ValueError('Unresolved duplicate/job/output')
    for path,h in inputs['local_support'].items():
        if R.sha(ROOT/path)!=h:raise ValueError('Support SOURCE mismatch:'+path)
    cell=inputs['cell'];config=read(BASE,CONFIGDIR+'/generation/CONFIGURATION.json')
    if cell['request']!=request(config,source):raise ValueError('Fixed request differs')
    cache=Path(inputs['cache_root'])
    if cache.exists()and any(cache.iterdir()):raise ValueError('Unresolved cache collision')
    R.PUBLIC=PUBLIC;out=ROOT/PUBLIC/'acquisition';out.mkdir()
    write(out/'EXECUTION.json',dict(source=source,input_commit=head(),maximum_paths=4,maximum_compiles=8,started=time.time()))
    start=time.perf_counter();rows=[];processes=[]
    try:
        ham,md=R.restore(cell['request'],inputs['saved_input_files'],inputs['runtime']);pc=cell['request']['physical_context']
        runtime=R.D.NativeRuntime(ROOT,ham,binding=pc['binding'],input_source=pc['input_source'],execution_source=pc['execution_source']);ctx=runtime.context(cell['request'])
        if list(ctx.prep.deterministic_fragment_indices)!=list(range(9))or list(ctx.prep.randomized_block_indices)!=[9]:raise ValueError('Actual prefix9 head/tail mismatch')
        write(out/'PREPARATION.json',dict(input_identity=md,preparation_hash=ctx.prep.preparation_hash,head=list(ctx.prep.deterministic_fragment_indices),tail=list(ctx.prep.randomized_block_indices),
            scalar=ctx.scalar,scalar_hex=float(ctx.scalar).hex(),lambda_R=ctx.prep.rte_preparation.symbolic_tail.lambda_r,tail_scalar=ctx.prep.extracted_identity_coefficient))
        R.STATE.update(inputs=inputs,runtimes={6:runtime},contexts={6:ctx})
        context=mp.get_context('fork');flag=context.Event()
        with ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=R.init_price,initargs=(flag,))as pool:
            pending={pool.submit(R.acquire_path,(cell,i)):i for i in cell['indices']};processes=list(pool._processes.values())
            while pending:
                done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    i=pending.pop(f)
                    try:r=f.result()
                    except BaseException:flag.set();r=dict(m=6,index=i,status='WORKER_ERROR',error=traceback.format_exc(),compiles=[])
                    rows.append(r);write(out/f'SUPERVISOR_p{i}.json',r);print('PATH',i,r['status'],flush=True)
                if pending:print('ACTIVE',list(pending.values()),flush=True)
        ok=len(rows)==4 and all(r['status']=='VALID'for r in rows)and all(p.exitcode==0 for p in processes)
        write(out/'RESULT.json',dict(status='ACQUIRED'if ok else'STOP_PARTIAL',rows=sorted(rows,key=lambda r:r['index']),new_paths=sum(r['status']=='VALID'for r in rows),
            compile_attempts=sum(len(r['compiles'])for r in rows),completed_compiles=sum(v['status']=='COMPLETE'for r in rows for v in r['compiles']),
            process_exitcodes=[dict(pid=p.pid,exitcode=p.exitcode)for p in processes],wall_seconds=time.perf_counter()-start,source=source,mandatory_stop=True))
        return 0 if ok else 2
    except BaseException:
        write(out/'STOP.json',dict(error=traceback.format_exc(),rows=rows,wall_seconds=time.perf_counter()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','acquire'));p.add_argument('--source');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    else:raise SystemExit(acquire())
