"""Resume only four previously unstarted Q3 calls, using the original AST runtime."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
CALLER='scripts/run_pr_df_ld_split_q3_resume.py'
PUBLIC='docs/research/d_pr_df_ld_split_q3_resume_20261116'
PRIOR='docs/research/d_pr_df_ld_split_results_20261115'
OLD='e0b3cad2a024070389f757160f3a20d7da02c5ff'
FIXED='4d3aa8a666e153188addfb3a4f2b8ef10733a3dd'
ALLOC='4ef70274261cf18bd9356a15d780eb5b6b61c60c'
PYTHON='/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python'
LABELS=('LD1_A_SHAPE','LD1_B_PRICE','LD2_A_SHAPE','LD2_B_PRICE')
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)

def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def read(commit,path):return json.loads(blob(commit,path))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# Preserve the previous caller and SOURCE view. Only orchestration/output is new.
Q=types.ModuleType('_split_resume_saved_caller');Q.__file__=str(ROOT/'scripts/run_pr_df_ld_split_q3.py')
sys.modules[Q.__name__]=Q
exec(compile(blob(OLD,'scripts/run_pr_df_ld_split_q3.py'),Q.__file__+'@'+OLD,'exec'),Q.__dict__)
U=Q.U

def runtime_check():
    if sys.version_info[:3]!=(3,11,0)or sys.version_info.releaselevel!='candidate' or sys.version_info.serial!=1:raise ValueError('Use successful fixed-AST Python3.11.0rc1 environment')
    U.check_source()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=Q.VIEW,text=True).strip()!=Q.Q3_SOURCE:raise ValueError('Fixed SOURCE view changed')
    # No sampling/signal/numerical evaluation. Constructor exercises exact old guard.
    U.R.Observed(ROOT/PUBLIC,'preflight_constructor',workers=1,outcome_batch=8)
    return dict(executable=sys.executable,version=sys.version,platform=platform.platform(),
        precedent=dict(commit='738df226232e539cc084dfad2aed2feae82f3073',path='docs/research/pauli_ld2_reallocation_q3_results_20261115/TESTS.txt'),
        expected_AST_hash=U.R.P.REFERENCE_DP_AST_SHA256,constructor='PASS_UNCHANGED_GUARD',kernel_source=U.R.P.REFERENCE_SOURCE,
        accelerator_source=U.R.ACCELERATOR,guard_modified=False)

def fixed_inputs():
    inputs=read(FIXED,PRIOR+'/Q3_CALL_INPUTS.json');allocation=read(ALLOC,PRIOR+'/ALLOCATION_INPUTS.json')
    if tuple(c['label']for c in inputs['cases'])!=LABELS or inputs['held']:raise ValueError('Four fixed cases required')
    if inputs['cases']!=allocation['cases']:raise ValueError('Allocation/case correspondence differs')
    if len({(c['original_configuration_id'],tuple((r['N_X'],r['N_Y'])for r in c['allocation']['shots']))for c in inputs['cases']})!=4:raise ValueError('Duplicate candidate/vector')
    for c in inputs['cases']:
        U.metadata(c) # own saved LD signal / canonical binding; no risk evaluation
    return inputs

def prepare(source):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Commit caller before input fixation')
    env=runtime_check();inputs=fixed_inputs()
    # Preserve original input values; only execution provenance/output is added.
    write(ROOT/PUBLIC/'INPUTS.json',dict(source=source,fixed_inputs=dict(commit=FIXED,path=PRIOR+'/Q3_CALL_INPUTS.json'),
        fixed_allocations=dict(commit=ALLOC,path=PRIOR+'/ALLOCATION_INPUTS.json'),prior_STOP=dict(commit=OLD,path=PRIOR+'/q3_calls/COMPLETION.json'),
        inputs=inputs,runtime=env,maximum_scientific_calls=4,retries=0,output=PUBLIC+'/run01',
        no_price_or_allocation_regeneration=True,scientific_scope='FOUR_FIXED_PR_DF_Q3_ONLY'))

def supervise(cases,inputs,source,output,*,monitor=U.P.monitor_pending,poll_seconds=2):
    """Same U.worker path; capture outer-process exit status after owned pool joins."""
    output=Path(output);start=time.perf_counter();context=mp.get_context('fork');failure=context.Event()
    receipts=[];errors=[];monitor_errors=[];peaks={c['label']:{}for c in cases};enabled=True;processes=[]
    if cases:
        with ProcessPoolExecutor(max_workers=len(cases),mp_context=context,initializer=U.init_worker,initargs=(failure,))as pool:
            pending={pool.submit(U.worker,c,inputs,source,str(output/c['label'])):c['label']for c in cases}
            processes=list(pool._processes.values())
            while pending:
                done,_=wait(pending,timeout=poll_seconds,return_when=FIRST_COMPLETED)
                for f in done:
                    label=pending.pop(f)
                    try:
                        r=f.result();receipts.append(r);U.R.write_new(output/label/'SUPERVISOR_RECEIPT.json',r)
                        if r['status']=='UNKNOWN':failure.set()
                        print(label+': '+r['status'],flush=True)
                    except BaseException as exc:failure.set();errors.append(dict(label=label,type=type(exc).__name__,reason=str(exc)))
                if pending and enabled:
                    try:monitor(output,pending,start,peaks)
                    except Exception as exc:
                        failure.set();enabled=False;monitor_errors.append(dict(type=type(exc).__name__,reason=str(exc)))
                        U.R.write_new(output/'MONITOR_ERROR.json',dict(errors=monitor_errors,scientific_retry=False))
    return dict(receipts=receipts,errors=errors,monitor_errors=monitor_errors,
        process_exitcodes=[dict(pid=p.pid,exitcode=p.exitcode,signal=-p.exitcode if p.exitcode and p.exitcode<0 else None)for p in processes],
        sampled_process_tree_peak_RSS_KiB={k:v.get('job')for k,v in peaks.items()},wall_seconds=time.perf_counter()-start)

def execute():
    package=json.loads((ROOT/PUBLIC/'INPUTS.json').read_text());source=package['source']
    if (ROOT/CALLER).read_bytes()!=blob(source,CALLER):raise ValueError('Execution SOURCE changed')
    if package['inputs']!=fixed_inputs()or package['runtime']!=runtime_check():raise ValueError('Fixed inputs/runtime changed')
    delta=json.loads((ROOT/PUBLIC/'DELTA_CHECK.json').read_text())
    if delta['equivalent_receipts']or delta['matching_jobs'] or delta['unresolved']:raise ValueError('Reconcile equivalent/unknown work before submissions')
    inputs=package['inputs'];out=ROOT/PUBLIC/'run01';out.mkdir()
    for c in inputs['cases']:(out/c['label']).mkdir()
    src=dict(commit=source,path=CALLER,inputs=dict(commit=head(),path=PUBLIC+'/INPUTS.json'),
        original_fixed_inputs=package['fixed_inputs'],reused_route=inputs['reused_route'],runtime=package['runtime'])
    U.R.write_new(out/'EXECUTION.json',dict(source=src,started_UTC=U.R.utc(),maximum_scientific_calls=4))
    r=supervise(inputs['cases'],inputs,src,out)
    ok=(not r['errors']and not r['monitor_errors']and len(r['receipts'])==4 and all(x['status']!='UNKNOWN'and x['scientific_evaluation_calls']==1 for x in r['receipts'])and all(p['exitcode']==0 for p in r['process_exitcodes']))
    U.R.write_new(out/'COMPLETION.json',dict(status='FOUR_FIXED_Q3_SAVED'if ok else'UNKNOWN_STOP_NO_RETRY',**r,finished_UTC=U.R.utc(),mandatory_stop=True))
    return 0 if ok else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','execute'));p.add_argument('--source');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    else:raise SystemExit(execute())
