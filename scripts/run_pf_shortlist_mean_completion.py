"""One approved batch, <=48 new means +30 saved copies; no retry/N/MSE/C/G."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pf_shortlist_mean_completion as F
import run_truncated_fullh_signals as TR
CALLER='scripts/run_pf_shortlist_mean_completion.py'
COMPLETE='SIX_SHORTLIST_SIGNAL_PACKETS_COMPLETE_NOT_RMSE_QUALIFIED'
BRANCHES={
 'A':('codex/a-full-h-p2-pauli-b1-price-design-20261114',F.STAGING),
 'B':('codex/rpe-total-dp-acceleration-20261114',F.B_SOURCE),
 'C':('codex/m5-m4-receipt-admission-20261114','0b4b2d941cd669aaa45231e1236bc2d3158a9d12'),
 'T2':('codex/truncated-pf-optimization-plan-20261114',F.TRUNC),
 'A_signal':('codex/pr-r-dependent-n-model-20261111',F.FULL),
 'D':('codex/d-price-records-c-bridge-20261114','acea1118a4df4e727e13398ce0f8ae11511c7396')}


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')


def duplicate_check():
    """Refresh relevant publication/process delta only; reuse the 710-row audit."""
    refs=subprocess.check_output(['git','ls-remote','--heads','origin',*[x[0] for x in BRANCHES.values()]],cwd=ROOT,text=True)
    remotes={r.split()[1].removeprefix('refs/heads/'):r.split()[0] for r in refs.splitlines()}
    rows=[]
    for role,(branch,fixed) in BRANCHES.items():
        if remotes.get(branch)!=fixed:raise ValueError('Relevant public SOURCE advanced; inspect delta first: '+role)
        rows.append(dict(role=role,branch=branch,remote_head=remotes[branch],prior_inspected_head=fixed,delta=False))
    processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,etime=,args='],text=True)
    # Limit to this scientific family. The observing shell/caller itself is
    # excluded; do not inspect or alter other users' processes.
    matches=[line for line in processes.splitlines() if any(s in line for s in
        ('run_h4_full_pf_heff_scan.py','run_h4_full_pf_followup.py','run_h4_truncated_t2.py','run_pf_shortlist_mean_completion.py --execute'))
        and 'bash -' not in line and str(os.getpid())!=line.strip().split()[0]]
    if matches:raise ValueError('Potential duplicate mean job; inspect before any science: '+str(matches))
    changed=subprocess.check_output(['git','diff','--name-only',F.S.CELL_PLAN['commit'],BRANCHES['C'][1]],cwd=ROOT,text=True).splitlines()
    if any(p.startswith(('artifacts/','src/')) or '/run' in p for p in changed):
        raise ValueError('New C data/code requires duplicate acceptance audit')
    return dict(checked_UTC=datetime.now(timezone.utc).isoformat(),roles=rows,
        source_audit=F.S.COVERAGE,scope='Existing 710-record inventory plus relevant public HEAD deltas and own-user active family processes',
        C_since_cell_plan=changed,new_reusable_means=0,matching_active_jobs=matches,
        new_candidates=48,saved_reuse=30,legacy_m3_replaced_by_new_direct=4,
        m0_shared_physical_rounds='2 saved means accepted explicitly; no extra calls',
        conclusion='NO_NEW_ACCEPTABLE_EQUIVALENT_MEANS_FOUND_IN_INSPECTED_SCOPE',
        uninspected='Unpublished external-server runs are not claimed checked')


def preflight(directory):
    source=dict(fixture='METADATA_PREFLIGHT_ONLY_NOT_SCIENTIFIC_SOURCE')
    check=duplicate_check();prepared=F.prepare(ROOT,source)
    out=Path(directory);out.mkdir(parents=True,exist_ok=False)
    write(out/'DUPLICATE_CHECK.json',check)
    write(out/'PLAN.json',dict(new=prepared['new'],saved_sources=prepared['saved_sources'],steps=prepared['steps'],
        canonical_ids={s['label']:s['request']['configuration']['id'] for s in prepared['schedules']},
        new_count=48,saved_count=30,unique_saved_steps=10,new_reference=0,new_spectrum=0,new_step_build=0))
    print('PREFLIGHT PASS: 48 new / 30 saved / 10 saved operators; no molecular arrays or science called.',flush=True)


def fixed_source(source):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=source:
        raise ValueError('Fixed current SOURCE commit required')
    paths=[F.MODULE,CALLER,'src/trottertracks/baseline_signal_pipeline.py',
        'src/trottertracks/baseline_signal_numerics.py','src/trottertracks/rpe_optimization_pipeline.py',
        'src/trottertracks/saved_pf_mean_staging.py']
    records=[]
    for path in paths:
        raw=F.C.blob(ROOT,source,path)
        if (ROOT/path).read_bytes()!=raw:raise ValueError('Used SOURCE changed: '+path)
        records.append(dict(commit=source,path=path,sha256=hashlib.sha256(raw).hexdigest()))
    return dict(commit=source,caller=CALLER,module=F.MODULE,
        numerical_kernel='baseline_signal_numerics.evaluate PF branch; unchanged',
        recipe='SAVED_DIRECT_EXPM_PF_OPERATOR_MATRIX_POWER_V1',P=0),records


def execute(out,source,plan_dir):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output required; no retry/resume')
    out.mkdir(parents=True);start=time.monotonic();stop=threading.Event();lock=threading.Lock();started=[];finished=[]
    result=dict(status='STARTED',scientific_source=source,started_UTC=datetime.now(timezone.utc).isoformat(),
        candidate_results=[],failures=[],new_mean_evaluations=0,saved_copies=0,
        new_step_builds=0,new_spectra=0,new_references=0,N=None,MSE=None,C=None,G=None,winner=None,
        automatic_retry=False,mandatory_stop=True)
    write(out/'EXECUTION.json',result)
    try:
        provenance,result['code_sources']=fixed_source(source)
        published=F.read(ROOT,source,str(Path(plan_dir)/'PLAN.json'))
        duplicate=F.read(ROOT,source,str(Path(plan_dir)/'DUPLICATE_CHECK.json'))
        result['duplicate_check_source']=dict(commit=source,path=str(Path(plan_dir)/'DUPLICATE_CHECK.json'))
        if duplicate['new_reusable_means']!=0:raise ValueError('Update new call scope to exclude reused data')
        prepared=F.prepare(ROOT,provenance)
        if prepared['new']!=published['new'] or prepared['saved_sources']!=published['saved_sources'] or prepared['steps']!=published['steps']:
            raise ValueError('Frozen preflight scientific menu changed')
        write(out/'SAVED_MEAN_SOURCES.json',prepared['saved_sources'])
        inputs=F.load_inputs(ROOT,prepared['shared']['binding'])
        steps=F.load_steps(ROOT,prepared['steps'])
        F.N.kernels(str(ROOT))  # Prime frozen function imports before threads.
        write(out/'INPUTS.json',dict(source=inputs['source'],observed=inputs['observed'],constant=inputs['constant'],
            B0I_p7_tail_scalar=inputs['tail_scalar'],unique_saved_steps=prepared['steps']))
        api=F.I.providers(str(ROOT));reader=TR.reference_reader(ROOT,prepared['shared'],api)
        original_evaluate=F.N.evaluate
        approved={(r['configuration_id'],r['physical_id']) for r in prepared['new']}
        def guarded(ctx,request,reference,*,source):
            key=(request['identity']['configuration']['configuration_id'],request['physical_candidate']['id'])
            with lock:
                if stop.is_set():raise ValueError('NOT_STARTED_AFTER_GLOBAL_STOP')
                if key not in approved or key in started:
                    stop.set();raise ValueError('Unapproved/duplicate scientific call')
                started.append(key)
            try:
                record=original_evaluate(ctx,request,reference,source=source)
                entry=ctx.entries[request['physical_candidate']['id']]
                record['input_source']=dict(saved_step=entry['step_source'],saved_input=deepcopy(ctx.metadata['source']),
                    original_coverage_status=entry['original_status'],scalar_rule=F.S.SCALAR_RULE[request['physical_candidate']['method']],
                    scalar_values=dict(H_constant=ctx.constant,tail_identity=ctx.tail if request['physical_candidate']['method']=='B0I' else None))
                with lock:
                    finished.append(dict(key=key,status=record['status']))
                    if record['status']!='NUMERICAL_MEAN_AVAILABLE':
                        stop.set();result['failures'].append(dict(key=key,type='NUMERICAL_GATE_OR_NONFINITE',reason=record.get('reason')))
                return record
            except Exception as e:
                with lock:stop.set();result['failures'].append(dict(key=key,type=type(e).__name__,reason=str(e)))
                raise
        def checked_reader(desc):
            if stop.is_set():raise ValueError('NOT_STARTED_AFTER_GLOBAL_STOP')
            return reader(desc)
        def worker(schedule):
            label=schedule['label'];request=schedule['request']
            ctx=F.SavedStepContext(ROOT,request,inputs,[v for v in prepared['new'] if v['label']==label],steps)
            packet=F.I.produce_signals(ROOT,request,out/label,context_factory=lambda:ctx,reference_provider=checked_reader,saved=schedule['saved'])
            write(out/label/'B_SIGNAL_INPUT.json',F.I.signal_input(packet))
            p=packet['producer_result']
            row=dict(label=label,configuration_id=request['configuration']['id'],status=p['status'],
                new_mean_evaluations=p['new_mean_evaluations'],saved_copies=p['saved_copies'],
                available_rounds=[m['m'] for m in packet['means'] if m['raw'] is not None],errors=p['errors'])
            if row['status']!='BASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIED':stop.set()
            print(label+' '+row['status']+' new='+str(row['new_mean_evaluations'])+' saved='+str(row['saved_copies']),flush=True)
            return row
        # Six natural candidate tasks, bounded by actual CPU affinity rather
        # than an arbitrary low worker cap. Per-worker output/caches are disjoint.
        workers=min(len(prepared['schedules']),len(os.sched_getaffinity(0)))
        result['resources']=dict(workers=workers,affinity_cpus=len(os.sched_getaffinity(0)),
            memory_available_before=next((s.strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')),None))
        # Only this process's evaluator binding is wrapped for scope/STOP.
        # The underlying fixed C arithmetic and all external code stay intact.
        with patch.object(F.N,'evaluate',guarded),ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(worker,s):s['label'] for s in prepared['schedules']}
            for future in as_completed(futures):
                try:result['candidate_results'].append(future.result())
                except Exception as e:
                    stop.set();result['failures'].append(dict(label=futures[future],type=type(e).__name__,reason=str(e)))
        result['new_mean_evaluations']=len(started);result['finished_scientific_calls']=finished
        result['saved_copies']=sum(s['saved_copies'] for s in result['candidate_results'])
        complete=(not stop.is_set() and len(result['candidate_results'])==6 and len(started)<=48
                  and result['saved_copies']+len(started)==78)
        result['status']=COMPLETE if complete else 'STOP_PARTIAL_RESULTS_NO_RETRY'
        for spec in [*inputs['source']['inputs'].values(),inputs['source']['context']]:
            if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()!=spec['sha256']:
                raise ValueError('Used input bytes changed')
        import numpy,scipy
        result['runtime']=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__)
    except Exception as e:
        stop.set();result['status']='STOP_INPUT_SOURCE_OR_RUNTIME_EXCEPTION'
        result['failures'].append(dict(type=type(e).__name__,reason=str(e)))
    result['new_mean_evaluations']=len(started)
    result['elapsed_seconds']=time.monotonic()-start;result['completed_UTC']=datetime.now(timezone.utc).isoformat()
    write(out/'RESULT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--out',required=True);parser.add_argument('--source');parser.add_argument('--plan-dir',default=F.OUT+'preflight')
    args=parser.parse_args()
    if args.preflight:preflight(args.out)
    else:
        if not args.source:parser.error('--source required')
        value=execute(args.out,args.source,args.plan_dir)
        print(json.dumps(dict(status=value['status'],new=value['new_mean_evaluations'],saved=value['saved_copies'],failures=value['failures']),indent=2),flush=True)
        sys.exit(0 if value['status']==COMPLETE else 2)
