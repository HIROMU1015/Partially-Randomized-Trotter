"""Adopted LD1/LD2 finite comparison. No retry or additional scientific menu."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
from trottertracks import p1_efficient_zz as P
BASE='e827367629ca6c2920d9d908fc1631314399db86'
PLAN='f42f6e9b97372dacb132d278b31044c391e7b921'
PUBLIC='docs/research/d_pr_df_ld_split_results_20261115'
COHORT='D_PR_DF_LD2_SPLIT_CAL_20261115_V1'
CALLER='scripts/run_pr_df_ld_split.py'
STATE={}
FAILURE=None

def blob(commit,path):return P._blob(ROOT,commit,path)
def read(commit,path):return json.loads(blob(commit,path))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def enc(x):
    x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)
def dec(x):return Fraction(x['numerator'],x['denominator']) if isinstance(x,dict) else Fraction(x)
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def pinned_module(name,path,commit=BASE):
    m=types.ModuleType(name);m.__file__=str(ROOT/path);m.__package__='trottertracks';sys.modules[name]=m
    exec(compile(blob(commit,path),str(ROOT/path)+'@'+commit,'exec'),m.__dict__);return m

def price_modules():
    fast=pinned_module('trottertracks._split_fast','src/trottertracks/compiled_price_fast.py')
    adapter=pinned_module('trottertracks._split_adapter','src/trottertracks/candidate_price_adapter.py');adapter.fast=fast
    sys.modules['trottertracks.candidate_price_adapter']=adapter
    requests=pinned_module('trottertracks._split_requests','src/trottertracks/ld12_price_requests.py');requests.D=adapter
    return adapter,requests,fast

D,L,F=price_modules()

def prepare(source):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Caller not committed')
    plan=read(PLAN,'docs/research/d_pr_df_ld_split_cost_plan_20261115/NEXT_FINITE_PLAN.json')
    old=read('2f423befbba5075bad4ede3057f89edd788d2661','docs/research/d_pr_df_ld1_m3_results_20261115/INPUTS.json')
    seed=P.trial_seed_function(ROOT);cells=[]
    for cell in plan['stages'][0]['cells']:
        req=L.request_for(ROOT,'PR-LD2',cell['m'],execution_source=dict(commit=source,path=CALLER))
        c=req['physical_context']['physical_candidate']
        if (c['id'],c['q'],c['R_outer'],c['prefix'])!=(cell['physical_id'],cell['q'],cell['R_outer'],2):raise ValueError('Plan physical mismatch')
        seeds=[[seed(c,COHORT,i,j,0) for j in range(c['q'])] for i in range(4)]
        cells.append(dict(m=cell['m'],request=req,cohort=COHORT,seeds=seeds,indices=list(range(4))))
    allseeds=[s for c in cells for ss in c['seeds'] for s in ss]
    if len(set(allseeds))!=len(allseeds):raise ValueError('Seed overlap')
    write(ROOT/PUBLIC/'ACQUISITION_INPUTS.json',dict(schema='PR_DF_LD_SPLIT_ACQUISITION_V1',source=source,caller=CALLER,
        plan=dict(commit=PLAN,path='docs/research/d_pr_df_ld_split_cost_plan_20261115/NEXT_FINITE_PLAN.json'),
        cells=cells,runtime=old['runtime'],saved_input_files=old['saved_input_files'],
        local_support={p:sha(ROOT/p) for p in ('src/trottertracks/p1_efficient_zz.py','src/trottertracks/efficient_controlled_zz.py','src/trottertracks/h4_df10_price_v1.py','src/trottertracks/track_d_native_resources.py','scripts/h46_b3_fast_fingerprint.py')},
        cache_root=str(ROOT/PUBLIC/'isolated_cache'),maximum_new_paths=12,maximum_compiles=24,
        numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',full_wrapper_statevector_validation=False))

def restore(request,inputs,runtime):
    provider=Path(runtime['provider_root']);sys.path.insert(0,str(provider/'src'))
    import trottertracks
    trottertracks.__path__.insert(0,str(provider/'src/trottertracks'))
    import numpy,qiskit
    if (numpy.__version__,qiskit.__version__)!=('1.26.4','1.3.0'):raise ValueError('Runtime version')
    for v in inputs.values():
        if sha(v['path'])!=v['sha256'] or Path(v['path']).stat().st_size!=v['bytes']:raise ValueError('Saved input mismatch')
    science=P.load_c1_backend(ROOT)['r_price_phase1_science']
    def require(ok,reason):
        if not ok:raise ValueError(reason)
    scope=dict(Path=Path,c=types.SimpleNamespace(read=science.read,sha=sha,require=require))
    science.functions(ROOT,P.CARRIER,'scripts/h46_m5_tol_saved_input.py',['array_record','load'],scope)
    ham,sector,full,state,md=scope['load'](dict(mapping=dict(tol_snapshot=inputs['H4_tol_snapshot.npz']['path'],tol_receipt=inputs['H4_tol_receipt.json']['path'])))
    binding=request['physical_context']['binding']
    if md['hamiltonian_hash']!=binding['hamiltonian'] or md['state_hash']!=binding['state'] or md['rank']!=10 or len(state)!=36:raise ValueError('H/state/rank identity')
    return ham,md

def init_price(flag):
    global FAILURE
    FAILURE=flag

def acquire_path(job):
    cell,index=job;m=cell['m'];req=cell['request'];ctx=STATE['contexts'][m];runtime=STATE['runtimes'][m]
    directory=ROOT/PUBLIC/'acquisition'/f'm{m}'/f'p{index}';log=ROOT/PUBLIC/'acquisition'/f'm{m}_p{index}_STAGES.jsonl'
    compiles=[];start=time.perf_counter();cpu=time.process_time();orig=ctx.compile;active={'axis':None}
    def stage(name,status,seconds=0.,value=None):
        with log.open('a')as f:f.write(json.dumps(dict(stage=name,status=status,seconds=seconds,elapsed=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
        if status=='STARTED' and name.endswith('_compile'):active['axis']=name.split('_')[0]
    def compile_one(wrapper,settings,**kwargs):
        if len(compiles)>=2:raise ValueError('Two compile limit')
        if wrapper.num_qubits!=9 or wrapper.num_clbits!=1 or wrapper.count_ops().get('measure')!=1:raise ValueError('Full measured wrapper')
        e=dict(axis=active['axis'],status='STARTED');compiles.append(e);then=time.perf_counter()
        try:
            v=orig(wrapper,settings,**kwargs);e['status']='COMPLETE';return v
        finally:e['wall_seconds']=time.perf_counter()-then
    try:
        if FAILURE.is_set():return dict(m=m,index=index,status='HELD_SIBLING_FAILURE',compiles=[])
        stage('sample','STARTED');requests,seeds,laws=ctx.h['sample_requests'](req['physical_context']['physical_candidate'],ctx.prep,cell['cohort'],index);stage('sample','COMPLETE')
        if seeds!=cell['seeds'][index]:raise ValueError('Frozen seeds mismatch')
        eventrows=[[e.to_dict() for e in r.rte_occurrence.events]for r in requests]
        digest=runtime.science.m.digest(eventrows);events=P.event_metrics([e for row in eventrows for e in row]);del eventrows
        ctx.compile=compile_one
        record=D.Adapter(ROOT,runtime,cache_root=STATE['inputs']['cache_root'],observer=stage).acquire(req,requests,cohort=cell['cohort'],index=index,directory=directory)
        if record['seeds']!=seeds or record['event_digest']!=digest or record['new_compiles']!=len(compiles) or record['cache_hits']:raise ValueError('Receipt/seed/digest/cache mismatch')
        for r,axis in zip(record['records'],('X','Y')):
            raw=json.loads((directory/(D.AXES[axis]+'.json')).read_text())
            if r['axis']!=axis or r['metrics']['rz_count']!=r['C'] or r['fingerprint']!=raw['fingerprint'] or raw['compiler']!=D.p.PROFILE['compiler']:raise ValueError('Axis metrics/fingerprint mismatch')
        result=dict(status='VALID',m=m,index=index,record=record,events=events,compiles=compiles,
            wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gate_status='INPUT_PREPARATION_REQUEST_LOWERING_COMPILER_FINGERPRINT_CHECKED',numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',full_wrapper_statevector_validation=False)
        write(directory/'PATH_RECEIPT.json',result);return result
    except BaseException:
        FAILURE.set();result=dict(status='STOP',m=m,index=index,compiles=compiles,error=traceback.format_exc(),wall_seconds=time.perf_counter()-start)
        write(ROOT/PUBLIC/'acquisition'/f'STOP_m{m}_p{index}.json',result);return result
    finally:ctx.compile=orig

def acquire():
    inputs=json.loads((ROOT/PUBLIC/'ACQUISITION_INPUTS.json').read_text());source=inputs['source']
    if (ROOT/CALLER).read_bytes()!=blob(source,CALLER):raise ValueError('Execution code mismatch')
    for p,h in inputs['local_support'].items():
        if sha(ROOT/p)!=h:raise ValueError('Support SOURCE mismatch:'+p)
    cache=Path(inputs['cache_root'])
    if cache.exists() and any(cache.iterdir()):raise ValueError('Unresolved output/cache collision')
    for cell in inputs['cells']:
        if cell['request']!=L.request_for(ROOT,'PR-LD2',cell['m'],execution_source=dict(commit=source,path=CALLER)):raise ValueError('Fixed request mismatch')
    out=ROOT/PUBLIC/'acquisition';out.mkdir()
    write(out/'EXECUTION.json',dict(execution_source=source,input_commit=head(),inputs_path=PUBLIC+'/ACQUISITION_INPUTS.json',started=time.time(),pid=os.getpid()))
    start=time.perf_counter();rows=[]
    try:
        ham,md=restore(inputs['cells'][0]['request'],inputs['saved_input_files'],inputs['runtime'])
        runtimes={};contexts={}
        for cell in inputs['cells']:
            pc=cell['request']['physical_context'];r=D.NativeRuntime(ROOT,ham,binding=pc['binding'],input_source=pc['input_source'],execution_source=pc['execution_source'])
            runtimes[cell['m']]=r;contexts[cell['m']]=r.context(cell['request'])
        write(out/'PREPARATION.json',dict(input_identity=md,contexts={str(m):dict(preparation_hash=c.prep.preparation_hash,scalar=c.scalar,scalar_hex=float(c.scalar).hex(),head=list(c.prep.deterministic_fragment_indices),tail=list(c.prep.randomized_block_indices))for m,c in contexts.items()}))
        STATE.update(inputs=inputs,runtimes=runtimes,contexts=contexts)
        context=mp.get_context('fork');flag=context.Event()
        # Independent path jobs; no historical CPU/worker cap is imposed. One worker per fixed path.
        with ProcessPoolExecutor(max_workers=12,mp_context=context,initializer=init_price,initargs=(flag,))as pool:
            pending={pool.submit(acquire_path,(cell,i)):(cell['m'],i)for cell in inputs['cells']for i in cell['indices']}
            while pending:
                done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    job=pending.pop(f)
                    try:r=f.result()
                    except BaseException:flag.set();r=dict(m=job[0],index=job[1],status='WORKER_ERROR',error=traceback.format_exc(),compiles=[])
                    rows.append(r);write(out/f'SUPERVISOR_m{job[0]}_p{job[1]}.json',r)
                    print('PATH',job,r['status'],flush=True)
                if pending:print('ACTIVE',list(pending.values()),flush=True)
        good=len(rows)==12 and all(r['status']=='VALID'for r in rows)
        write(out/'RESULT.json',dict(status='ACQUIRED'if good else'STOP_PARTIAL',rows=sorted(rows,key=lambda r:(r['m'],r['index'])),new_paths=sum(r['status']=='VALID'for r in rows),compile_attempts=sum(len(r['compiles'])for r in rows),completed_compiles=sum(c['status']=='COMPLETE'for r in rows for c in r['compiles']),wall_seconds=time.perf_counter()-start,execution_source=source,input_commit=head(),mandatory_stop=not good))
        return 0 if good else 2
    except BaseException:
        write(out/'STOP.json',dict(error=traceback.format_exc(),rows=rows,wall_seconds=time.perf_counter()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','acquire'));p.add_argument('--source');a=p.parse_args()
    if a.stage=='prepare':prepare(a.source)
    else:raise SystemExit(acquire())
