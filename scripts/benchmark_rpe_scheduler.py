"""Baseline/fast scheduling benchmark: invented workers, no real Q3 or kappa search."""
from copy import deepcopy
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_optimizer_concurrent as C
spec=importlib.util.spec_from_file_location('_scheduler_fixtures',ROOT/'tests/test_rpe_optimizer_concurrent.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


def benchmark(directory,fast,slots,workers,workload,lookahead=3,saturated=False):
    methods=['PR_PAULI']if saturated else F.METHODS
    problem,context,routes=F.fixture(methods=methods,configurations=9 if saturated else 1,lookahead=lookahead)
    for rules in routes.values():
        a=rules[0]['artificial'];a['sleep_seconds']=.06
        if workload=='fork_cpu':a['load']=dict(serial_seconds=.015,parallel_rounds=2,chunks=8,
            units_per_chunk=10000,memory_MiB=2,baseline_dp_workers=8)
    directory.mkdir(parents=True);start=time.perf_counter()
    if fast:
        ex=C.ConcurrentExecutor(ROOT,directory,routes,{},problem,max_concurrent=slots,
            budgets={m:None for m in methods},science_approved=True,dp_workers=workers)
    else:
        B=F.baseline();ex=B.Executor(ROOT,directory,routes,{},problem,max_concurrent=slots)
    # One identical RSS sampler for both versions, owned artificial jobs only.
    stop=threading.Event();samples=[]
    def observe():
        while not stop.is_set():
            pids=[]
            for p in directory.glob('jobs/*/*/LAUNCH.json'):
                if not(p.parent/'PROCESS_RETURN.json').exists():
                    try:pids.append(json.loads(p.read_text())['pid'])
                    except (ValueError,OSError):pass
            samples.append(dict(t=time.perf_counter()-start,n=len(pids),rss=sum(C._tree_rss(p)for p in pids)))
            stop.wait(.05)
    thread=threading.Thread(target=observe);thread.start()
    try:
        if fast:result=C.optimize(ROOT,problem,context=context,methods=methods,max_new={m:None for m in methods},executor=ex)
        else:result=B.optimize(ROOT,problem,context=context,methods=methods,max_new={m:None for m in methods},executor=ex)
    finally:stop.set();thread.join()
    wall=time.perf_counter()-start
    launched=list(directory.glob('jobs/*/*/LAUNCH.json'));adopted=sum(len(r['trace'])for r in result.values())
    if fast:metrics=ex.metrics()
    else:
        log=[json.loads(l)for l in(directory/'PROGRESS.jsonl').read_text().splitlines()]
        active=set();peak=0
        for e in log:
            if e['event']=='LAUNCH':active.add(e['job'])
            elif e['event']=='EXIT':active.discard(e['job'])
            peak=max(peak,len(active))
        metrics=dict(peak_active=peak)
    row=dict(scheduler='concurrent'if fast else'legacy_1e88cd85c',slots=slots,dp_workers=workers or 8,
        workload=workload,scenario='SINGLE_METHOD_NINE_CONFIGURATIONS'if saturated else'FOUR_METHODS_ONE_CONFIGURATION',
        lookahead=lookahead,wall_seconds=wall,launched_artificial_Q3=len(launched),
        adopted_artificial_Q3=adopted,maximum_concurrent_jobs=metrics['peak_active'],
        sampled_tree_peak_RSS_KiB=max((s['rss']for s in samples),default=0),
        search=O.jsonable(F.canonical(result)),executor_metrics=metrics,real_Q3_calls=0)
    (directory/'BENCHMARK.json').write_text(json.dumps(row,indent=2)+'\n')
    print(row['scheduler'],slots,workers,workload,round(wall,3),'s, launched',len(launched),'adopted',adopted,flush=True)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--repetitions',type=int,default=3);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for repeat in range(a.repetitions):
        for workload in ('latency','fork_cpu'):
            prefix=str(repeat)+'_'+workload
            base=benchmark(a.output/(prefix+'_legacy'),False,18,None,workload);base['repeat']=repeat;rows.append(base)
            profiles=[(18,8),(18,4),(18,1),(8,2)]if workload=='fork_cpu'else[(18,8),(8,8)]
            for slots,workers in profiles:
                row=benchmark(a.output/('%s_fast_%s_%s'%(prefix,slots,workers)),True,slots,workers,workload)
                row['repeat']=repeat;row['same_best_adoption_order_stop']=row['search']==base['search']
                row['speedup_vs_legacy']=base['wall_seconds']/row['wall_seconds']
                if not row['same_best_adoption_order_stop']:raise ValueError('Trajectory differs')
                rows.append(row)
        base=benchmark(a.output/(str(repeat)+'_saturated_legacy'),False,18,None,'latency',saturated=True)
        base['repeat']=repeat;rows.append(base)
        row=benchmark(a.output/(str(repeat)+'_saturated_fast'),True,18,8,'latency',saturated=True)
        row['repeat']=repeat;row['same_best_adoption_order_stop']=row['search']==base['search']
        row['speedup_vs_legacy']=base['wall_seconds']/row['wall_seconds']
        if not row['same_best_adoption_order_stop']:raise ValueError('Saturated trajectory differs')
        rows.append(row)
    result=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        baseline='1e88cd85cc8d2739a4dce43d138b94b153309333',scope='ARTIFICIAL_EXECUTION_WORKLOAD_ONLY',
        memory_scope='same 50ms owned-job RSS sampler for both; shared pages may be double-counted',
        real_Q3_calls=0,rows=rows,
        caveat='Synthetic fork/CPU scaling does not identify the fastest real-Q3 worker profile; no real speedup measured.')
    (a.output/'BENCHMARK_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
