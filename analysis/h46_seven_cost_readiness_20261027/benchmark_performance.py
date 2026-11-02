"""Pure scheduling arithmetic/microbenchmarks, never molecular execution."""
import heapq
import json
from pathlib import Path
import statistics
import time

import execute as x
import performance as p

HERE=Path(__file__).resolve().parent


def makespan(jobs, slots):
    ends=[0.]*slots;heapq.heapify(ends)
    for row,_,_ in jobs:
        work=x.native.cost_proxy(row)*(8 if row['system']=='H4' else 12)**2
        heapq.heappush(ends,heapq.heappop(ends)+work)
    return max(ends)


def main():
    plan=x.read_plan();scenarios=[]
    for cohort in plan['cohorts']:
        jobs=[(v['candidate'],cohort,i) for v in plan['candidates']
              for i in range(v['n_exploration'] if cohort==plan['cohorts'][0] else v['n_confirmation'])]
        ordered=p.order_jobs(jobs,x.native.cost_proxy)
        for slots in (8,32,128):
            a,b=makespan(jobs,slots),makespan(ordered,slots)
            scenarios.append(dict(cohort=cohort,simulated_workers=slots,
                FIFO_work_units=a,heavy_first_work_units=b,model_reduction_fraction=1-b/a,
                measured_molecular_time=False))
    original=x.prepare.core.schedule
    arguments=[(v['candidate']['formula'],v['candidate']['q'],'ABS_TIME_PER_OUTER',v['candidate']['R_outer'],v['candidate']['T']) for v in plan['candidates']]
    old_times=[];new_times=[];rounds=1000
    for repeat in range(3):
        memo=p.ScheduleMemo(original)
        for f,q,policy,r,t in arguments:
            assert memo(f,q,policy,r,T=t)==original(f,q,policy,r,T=t)
        start=time.perf_counter()
        for _ in range(rounds):
            for f,q,policy,r,t in arguments:
                original(f,q,policy,r,T=t)
        old_times.append(time.perf_counter()-start)
        start=time.perf_counter()
        for _ in range(rounds):
            for f,q,policy,r,t in arguments:
                memo(f,q,policy,r,T=t)
        new_times.append(time.perf_counter()-start)
    calls=rounds*len(arguments)
    # Exact schedule arguments, rather than physical candidate identity, define
    # the arithmetic memo; all seven fixed argument sets are distinct here.
    unique_schedules=len(set(arguments))
    value=dict(scope='LOCAL_PURE_SCHEDULE_AND_SCHEDULING_MODEL_ONLY',
        schedule_calls_per_repeat=calls, uncached_schedule_evaluations=calls,
        cached_schedule_evaluations_including_warmup=unique_schedules,
        uncached_seconds_median=statistics.median(old_times),cached_seconds_median=statistics.median(new_times),
        schedule_microbenchmark_speed_ratio=statistics.median(old_times)/statistics.median(new_times),
        scheduling_scenarios=scenarios, actual_molecular_speedup=None, actual_build_cache_hit_rate=None,
        new_scientific_calls=0, fixed_trajectory_count=plan['scientific_calls']['trajectories_total'])
    (HERE/'performance_checks.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    main()
