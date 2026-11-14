"""Two explicitly permitted accelerated replacements; reference caller reused.

No new target, loss, allocation, count support, decoder or science algorithm.
The original stopped run remains in its original branch/namespace.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
from trottertracks import rpe_total_dp_parallel as P
from trottertracks import rpe_total_binding as T

REFERENCE='8652cd873632c6adc03e5eb2f9cd5dd10e9b9d6c'
REFERENCE_CALLER='scripts/run_pauli_total_q3_fixed.py'
REFERENCE_INPUT='docs/research/pauli_total_q3_fixed_results_20261114/INPUTS.json'
ACCELERATOR='9e294778266c9c867a8b1b98e040c2e94b1cb635'
STOPPED='0a6f8e4a5'
DIRECTORY='docs/research/pauli_total_q3_accelerated_results_20261114'
INPUT=DIRECTORY+'/INPUTS.json'


def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def reference():
    module=ModuleType('_fixed_pauli_total_reference');module.__file__=str(ROOT/REFERENCE_CALLER)
    exec(compile(blob(REFERENCE,REFERENCE_CALLER),REFERENCE+':'+REFERENCE_CALLER,'exec'),module.__dict__)
    # Disable ONLY its old state-major progress observer. It is inappropriate
    # for process-parallel outcome batches; the scientific gates are unchanged.
    def observing_patch(owner,name,*args,**options):
        if owner is T.T and name=='step':return nullcontext()
        return patch.object(owner,name,*args,**options)
    module.patch=SimpleNamespace(object=observing_patch)
    return module


L=reference();write_new=L.write_new;utc=L.utc


def prepare():
    L.check_source()
    if (ROOT/'src/trottertracks/rpe_total_dp_parallel.py').read_bytes()!=blob(ACCELERATOR,'src/trottertracks/rpe_total_dp_parallel.py'):
        raise ValueError('Accelerator SOURCE differs')
    inputs=json.loads(blob(REFERENCE,REFERENCE_INPUT));original=deepcopy(inputs)
    inputs.update(scope='TWO_USER_APPROVED_ACCELERATED_REPLACEMENTS_SAME_SCIENTIFIC_INPUTS',
        previous_input=dict(commit=REFERENCE,path=REFERENCE_INPUT),
        stopped_original_run=dict(commit=subprocess.check_output(['git','rev-parse',STOPPED],cwd=ROOT,text=True).strip(),
            path='docs/research/pauli_total_q3_fixed_results_20261114/run01/SUPERSEDED_STOP.json'))
    inputs['acceleration']=dict(source=dict(commit=ACCELERATOR,path='src/trottertracks/rpe_total_dp_parallel.py'),
        workers_per_configuration=8,outcome_batch=8,
        selection='Artificial benchmark: 8 and 16 workers have similar wall time; use 8 per independent configuration.')
    inputs['options']['total_dp_acceleration_source']=deepcopy(inputs['acceleration'])
    if (inputs['cases']!=original['cases'] or inputs['statistical_contract']!=original['statistical_contract']
            or {k:v for k,v in inputs['options'].items() if k!='total_dp_acceleration_source'}!=original['options']):
        raise ValueError('Approved scientific conditions changed')
    (ROOT/DIRECTORY).mkdir(exist_ok=True);write_new(ROOT/INPUT,inputs)
    print('Prepared exactly the same two cases and scientific contracts, with explicit accelerated backend.',flush=True)


class Observed(P.Accelerator):
    def __init__(self,output,label,**options):
        self.output=Path(output);self.label=label;self.started=time.perf_counter()
        super().__init__(**options)
    def propagate(self,*args,**kwargs):
        m=args[3]
        with (self.output/('ACCEL_PROGRESS_'+self.label+'.jsonl')).open('a') as stream:
            stream.write(json.dumps(dict(stage='ROUND_START',m=m,UTC=utc(),wall_seconds=time.perf_counter()-self.started))+'\n')
        value=super().propagate(*args,**kwargs)
        with (self.output/('ACCEL_PROGRESS_'+self.label+'.jsonl')).open('a') as stream:
            stream.write(json.dumps(dict(stage='ROUND_PROPAGATION_DONE',m=m,UTC=utc(),
                wall_seconds=time.perf_counter()-self.started,observation=self.observations[-1] if self.observations and self.observations[-1]['m']==m else None))+'\n')
        return value


def worker(case,inputs,source,output):
    cfg=inputs['acceleration'];start_children=resource.getrusage(resource.RUSAGE_CHILDREN)
    engine=Observed(output,case['label'],workers=cfg['workers_per_configuration'],outcome_batch=cfg['outcome_batch'])
    with patch.object(T.Q,'total_state_dp',engine):
        result=L.worker(case,inputs,source,output)
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu=children.ru_utime+children.ru_stime-start_children.ru_utime-start_children.ru_stime
    result['performance'].update(child_cpu_seconds=child_cpu,
        process_tree_cpu_seconds=result['performance']['cpu_seconds']+child_cpu,
        max_single_child_RSS_KiB=children.ru_maxrss,
        RSS_scope='parent peak in peak_worker_RSS_KiB; child peak is maximum individual child, not sum')
    result['acceleration']=dict(backend=deepcopy(cfg),rounds=deepcopy(engine.observations))
    return result


def resources(output,labels,started,peaks):
    rows=[]
    def descendants(pid):
        seen=set();todo=[pid]
        while todo:
            value=todo.pop()
            if value in seen:continue
            seen.add(value)
            try:todo.extend(map(int,Path('/proc',str(value),'task',str(value),'children').read_text().split()))
            except FileNotFoundError:pass
        return seen
    for label in labels:
        path=output/('START_'+label+'.json')
        if not path.exists():continue
        pid=json.loads(path.read_text())['pid'];rss=0;cpu=0.;present=[]
        for child in descendants(pid):
            try:
                proc=Path('/proc',str(child))
                if proc.stat().st_uid!=os.getuid():continue
                raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
                rss+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')//1024
                cpu+=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK');present.append(child)
            except (FileNotFoundError,ProcessLookupError):pass
        peaks[label]=max(peaks.get(label,0),rss)
        rows.append(dict(label=label,parent_pid=pid,own_process_tree_pids=present,
            present_tree_CPU_seconds=cpu,tree_RSS_KiB=rss,sampled_peak_tree_RSS_KiB=peaks[label]))
    with (output/'RESOURCE_MONITOR.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(UTC=utc(),elapsed_seconds=time.perf_counter()-started,jobs=rows))+'\n')


def execute():
    L.check_source();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,'scripts/'+Path(__file__).name):
        if (ROOT/path).read_bytes()!=blob(commit,path):raise ValueError('Runner/input not fixed at SOURCE')
    if (ROOT/'src/trottertracks/rpe_total_dp_parallel.py').read_bytes()!=blob(ACCELERATOR,'src/trottertracks/rpe_total_dp_parallel.py'):
        raise ValueError('Accelerator SOURCE changed')
    inputs=json.loads((ROOT/INPUT).read_bytes());output=ROOT/DIRECTORY/'run01';output.mkdir()
    source=dict(commit=commit,path=INPUT,caller='scripts/'+Path(__file__).name)
    write_new(output/'EXECUTION.json',dict(source=source,B_source=L.BASE,kernel_source=L.ENGINE,
        accelerator=inputs['acceleration'],stopped_original_run=inputs['stopped_original_run'],
        started_UTC=utc(),evaluations_requested=2,MC_calls=0,old_partial_DP_calls=0,prices_or_G=False))
    if [c['label'] for c in inputs['cases']]!=['LD1','LD2']:raise ValueError('Exactly two fixed cases required')
    started=time.perf_counter();peaks={};completed=[];unknown=[];errors=[]
    with ProcessPoolExecutor(max_workers=len(inputs['cases'])) as pool:
        pending={pool.submit(worker,case,inputs,source,str(output)):case['label'] for case in inputs['cases']}
        while pending:
            done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
            for future in done:
                label=pending.pop(future)
                try:
                    result=future.result();result['performance']['sampled_peak_process_tree_RSS_KiB']=peaks.get(label)
                    write_new(output/('RESULT_'+label+'.json'),result);completed.append(label)
                    if result['status'].startswith('UNKNOWN'):unknown.append(label)
                    print(label+': '+result['status'],flush=True)
                except Exception as exc:errors.append(dict(label=label,type=type(exc).__name__,message=str(exc)))
            resources(output,list(pending.values()),started,peaks)
    write_new(output/'COMPLETION.json',dict(status='TWO_RESULTS_SAVED' if not unknown and not errors else 'UNKNOWN_STOP_NO_RETRY',
        completed=sorted(completed),unknown=unknown,errors=errors,wall_seconds=time.perf_counter()-started,
        finished_UTC=utc(),sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
