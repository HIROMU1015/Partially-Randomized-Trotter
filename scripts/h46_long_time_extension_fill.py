"""Fill unstarted q32 tasks when the main pool releases CPU resources.

The original worker continues normally. Existing cost_trial's exclusive mkdir
is the atomic claim BEFORE RNG/build/compile: a contention loser executes no
scientific call. Task IDs, seeds, cohort, n and numerical kernels stay fixed.
"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing as mp
from pathlib import Path
import time
import h46_long_time_native as native


def run():
    native.study.install()
    from pf_common_run_contract import resources
    plan=native.read(native.PUBLIC/'extension_native_selection.json');cohort='EXTENSION_EXPLORATION'
    jobs=[]
    for row in plan['candidates']:
        count=9 if row['method']=='PR' else 1
        jobs.extend((row,cohort,i) for i in range(count))
    for system in native.study.PREFIX:
        prefixes=sorted({r['prefix'] for r,_,_ in jobs if r['system']==system})
        native.CONTEXT[system]=native.native_context(system,prefixes)
    obs=resources();workers=max(1,min(len(jobs),int(obs['idle_cpu']),obs['available_ram']//(2*2**30)))
    native.save(native.PRIVATE/'extension_fill_resources.json',dict(observation=obs,workers=workers,
        reason='Main pool finished; original extension retains its one running worker; only unstarted tasks claimed'))
    submitted=set();futures={};failed={};collisions=0
    def key(job):
        return job[0]['id'],job[2]
    def directory(job):
        return native.PRIVATE/'cost'/cohort/job[0]['id']/str(job[2])
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        while True:
            terminal=0
            for job in jobs:
                row,_,index=job;folder=directory(job)
                if (folder/'STOP.json').exists():
                    failed[row['id']]=native.read(folder/'STOP.json')
                if (folder/'result.json').exists() or row['id'] in failed:
                    terminal+=1;continue
                if key(job) in submitted or folder.exists():
                    continue
                first=folder.parent/'0/result.json'
                if index>0 and not first.exists():
                    continue
                submitted.add(key(job));futures[pool.submit(native.cost_trial,job)]=job
            if terminal==len(jobs) and not futures:
                break
            if futures:
                done,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
                for f in done:
                    job=futures.pop(f)
                    try:
                        f.result()
                    except FileExistsError:
                        collisions+=1  # Another authorized dispatcher owns it; no science duplicated.
                    except BaseException as exc:
                        failed[job[0]['id']]=dict(reason=str(exc))
            else:
                time.sleep(1)
            native.save(native.PRIVATE/'extension_fill_progress.json',dict(terminal=terminal,total=len(jobs),
                failed_candidates=list(failed),claim_contentions=collisions,helper_tasks_submitted=len(submitted)))
    results={}
    for row in plan['candidates']:
        if row['id'] in failed:
            continue
        n=9 if row['method']=='PR' else 1
        trials=[native.read(native.PRIVATE/'cost'/cohort/row['id']/str(i)/'result.json') for i in range(n)]
        if any(t['status']!='PASS' or t['candidate']!=row or t['cohort']!=cohort or t['index']!=i for i,t in enumerate(trials)):
            raise ValueError('Q32_FIXED_TASK_RESULT_IDENTITY')
        results[row['id']]=native.summarize_trials(row,cohort,trials)
    native.save(native.PUBLIC/'extension_parallel_costs.json.gz',results)
    summary=dict(total_tasks=len(jobs),accepted_tasks=sum(v['n'] for v in results.values()),
        accepted_groups=len(results),failed_candidates=failed,helper_workers=workers,
        helper_claim_contentions=collisions,duplicate_scientific_execution=0,
        prior_dispatch_reports_preserved=True,cohort_pooling=False,all_helper_workers_joined=True)
    native.save(native.PUBLIC/'extension_parallel_summary.json',summary);return summary


if __name__=='__main__':
    print(json.dumps(run()))
