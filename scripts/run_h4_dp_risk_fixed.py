"""Two fixed saved-H4 DP evaluations; no retry, new allocation, MC or cost.

Reuses the previous evaluate_one caller and immutable numerical kernel.
The parent samples ONLY its own worker PIDs; it never changes resource limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PLAN_COMMIT='4139bed3c3364f4cf2afa3451d4a956961319060'
PLAN_PATH='docs/research/rpe_dp_finite_scope_preparation_20261114/FINITE_SCOPE.json'
ENGINE='06db9aefec13bb5c5be6d7c0f81f29b3a10b929c'
LEGACY='scripts/run_pr_ld1_ld2_deterministic_risk.py'
DIRECTORY='docs/research/h4_dp_risk_fixed_results_20261114'
INPUT=DIRECTORY+'/INPUTS.json'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def old_caller():
    spec=importlib.util.spec_from_file_location('fixed_legacy_h4_caller',ROOT/LEGACY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare():
    plan=json.loads(git('show',PLAN_COMMIT+':'+PLAN_PATH))
    inherited=plan['inherited_scientific_inputs']
    original=json.loads(git('show',inherited['commit']+':'+inherited['path']))
    inputs=deepcopy(original)
    inputs.update(engine_source=deepcopy(plan['engine_source']),
        inherited_scientific_inputs=deepcopy(inherited),
        finite_scope_source=dict(commit=PLAN_COMMIT,path=PLAN_PATH),
        scope='TWO_FIXED_SAVED_H4_PARTIAL_DP_DIAGNOSTICS_NO_SHOT_SEARCH')
    inputs['options']['state_refinement']=dict(**plan['proposed_state_refinement'],
        source=dict(commit=PLAN_COMMIT,path=PLAN_PATH,selector=['proposed_state_refinement']))
    assert inputs['cases']==original['cases']
    assert inputs['statistical_contract']==original['statistical_contract']
    assert {k:v for k,v in inputs['options'].items() if k!='state_refinement'}==original['options']
    assert inputs['options']['state_refinement']['maximum_frontier_states']==8192
    assert inputs['options']['state_refinement']['maximum_transition_pairs']==141559036
    (ROOT/DIRECTORY).mkdir(exist_ok=True)
    write_new(ROOT/INPUT,inputs)
    print('PREPARED fixed inherited conditions and one DP scope; no evaluation called',flush=True)


def worker(case,inputs,source,output):
    output=Path(output);pid=os.getpid();start_cpu=time.process_time();start=time.perf_counter()
    write_new(output/('START_'+case['label']+'.json'),dict(label=case['label'],pid=pid,
        parent_pid=os.getppid(),started_UTC=utc(),source=source))
    try:
        result=old_caller().evaluate_one(case,inputs,source)
        result['performance'].update(cpu_seconds=time.process_time()-start_cpu,
            worker_total_wall_seconds=time.perf_counter()-start,worker_pid=pid,finished_UTC=utc())
        return result
    except Exception as exc:
        write_new(output/('ERROR_'+case['label']+'.json'),dict(type=type(exc).__name__,
            message=str(exc),pid=pid,source=source,automatic_retry=False,mandatory_stop=True))
        raise


def own_resources(output,labels,started):
    observations=[]
    for label in labels:
        path=output/('START_'+label+'.json')
        if not path.exists():continue
        info=json.loads(path.read_text());pid=info['pid']
        if info['parent_pid']!=os.getpid():continue
        try:
            stat=Path('/proc/'+str(pid)+'/stat').read_text()
            fields=stat[stat.rfind(')')+2:].split()
            observations.append(dict(label=label,pid=pid,state=fields[0],
                cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                RSS_KiB=int(fields[21])*os.sysconf('SC_PAGE_SIZE')//1024))
        except FileNotFoundError:pass
    row=dict(elapsed_seconds=time.perf_counter()-started,observed_UTC=utc(),workers=observations)
    with (output/'RESOURCE_MONITOR.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def execute():
    source_commit=git('rev-parse','HEAD')
    for path in (INPUT,'scripts/'+Path(__file__).name):
        assert git('show',source_commit+':'+path)+'\n'==(ROOT/path).read_text()
    protected=['src/trottertracks/rpe_deterministic_risk.py','src/trottertracks/rpe_decoder_fast.py',
        'src/trottertracks/rpe_decoder_statistics.py','src/trottertracks/rpe_signal_statistics_caller.py',
        'src/trottertracks/rpe_shot_allocation.py','src/trottertracks/pr_rpe_rmse_controller.py',
        'tests/check_rpe_signal_statistics_caller.py',LEGACY]
    assert not git('diff',ENGINE,'--name-only','--',*protected)
    inputs=json.loads((ROOT/INPUT).read_text())
    assert [v['label'] for v in inputs['cases']]==['LD1','LD2']
    output=ROOT/DIRECTORY/'run01';output.mkdir()
    source=dict(commit=source_commit,path=INPUT,caller='scripts/'+Path(__file__).name)
    write_new(output/'EXECUTION.json',dict(source=source,engine_source=inputs['engine_source'],
        inherited_scientific_inputs=inputs['inherited_scientific_inputs'],
        finite_scope_source=inputs['finite_scope_source'],evaluations_requested=2,
        started_UTC=utc(),MC_calls=0,additional_allocations=False,prices_or_G=False))
    started=time.perf_counter();completed=[];errors=[]
    with ProcessPoolExecutor(max_workers=len(inputs['cases'])) as pool:
        pending={pool.submit(worker,case,inputs,source,str(output)):case['label'] for case in inputs['cases']}
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                label=pending.pop(future)
                try:
                    result=future.result();write_new(output/('RESULT_'+label+'.json'),result)
                    completed.append(label)
                    print(label+': '+result['record']['diagnostic']['deterministic_risk']['status'],flush=True)
                except Exception as exc:
                    errors.append(dict(label=label,type=type(exc).__name__,message=str(exc)))
            own_resources(output,list(pending.values()),started)
    write_new(output/'COMPLETION.json',dict(status='TWO_RESULTS_SAVED' if not errors else 'ERROR_NO_RETRY',
        completed=sorted(completed),errors=errors,wall_seconds=time.perf_counter()-started,finished_UTC=utc(),
        sufficient_N=None,global_RMSE_bound=None,G=None,winner=None,mandatory_stop=True))
    if errors:raise RuntimeError('Fixed evaluation failed; see saved error, no retry')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    args=parser.parse_args();(prepare if args.mode=='prepare' else execute)()
