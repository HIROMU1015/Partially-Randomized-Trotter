"""Prepare only five already-fixed exploration seeds using measured cache RSS.

The earlier controller conservatively reserves a full action-check peak for
idle workers. This one-shot tail uses remaining jobs and distinguishes index0
action checks from cache-only indices. It never accepts a price sample.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import math
import os
import h46_full_h_t20_opt as s
import prefetch_h46_full_h_t20 as p

WORK=s.PRIVATE/'prefetch_tail'


def job(row,index):
    p.install();p.previous.WORK=WORK
    return p.previous.precompile(row,index)


def main():
    p.install();WORK.mkdir(exist_ok=False)
    plan=s.read(s.PUBLIC/'price_exploration_v2.json')
    rows=plan['candidates'];target=next(r for r in rows if r['system']=='H6' and r['s']==16384 and r['K']==2)
    jobs=[(target,i) for i in range(4,9)]
    for r,i in jobs:
        s.full.require(not p.previous.trial_path(r,i).exists(),'MAIN_ALREADY_STARTED')
        s.full.require(not p.previous.job_path(r,i).exists(),'ALREADY_PREFETCHED')
    aux=[s.read(f) for f in p.WORK.glob('*.json')]
    full=[s.read(p.previous.trial_path(r,0)/'result.json') for r in rows
          if (p.previous.trial_path(r,0)/'result.json').exists()]
    def estimate(r,index):
        values=full if index==0 else aux
        # No extrapolation beyond factor two from the largest same-system run.
        values=[v for v in values if v['candidate']['system']==r['system']]
        largest=max(v['candidate']['s'] for v in values)
        s.full.require(r['s']<=2*largest,'RSS_EXTRAPOLATION')
        values=[v for v in values if v['candidate']['s']>=largest/2]
        return 2**30+1.25*max(v['peak_RSS_bytes']/v['candidate']['s'] for v in values)*r['s']
    progress=s.read(s.PRIVATE/'exploration_v2_progress.json')
    s.full.require(not progress['errors'] and progress['complete']>=73,'HEALTHY_FIXED_EXPLORATION')
    pid=s.read(s.PUBLIC/'prefetch_status.json')['main_pid']
    s.full.require(p.live(pid),'MAIN_BINDING')
    children=p.previous.children(pid)
    remaining=[(r,i) for r in rows for i in range(9)
               if not (p.previous.trial_path(r,i)/'result.json').exists()]
    # Reserve peaks for the largest possible concurrent remaining main jobs.
    forecast=sum(sorted((estimate(r,i) for r,i in remaining),reverse=True)[:len(children)])
    live=sum(p.previous.rss(pid) for pid in children)
    from pf_common_run_contract import resources
    obs=resources();free=p.previous.headroom(obs['available_ram'],forecast,live,0)
    required=sum(estimate(r,i) for r,i in jobs)
    s.full.require(required<free and len(jobs)<=math.floor(obs['idle_cpu']),'TAIL_RESOURCE_ADMISSION')
    s.save(WORK/'resources.json',dict(observation=obs,remaining_main_jobs=len(remaining),
        main_peak_forecast=forecast,main_current_RSS=live,headroom=free,auxiliary_required=required,
        measurements='completed full index0 and exact same-system cache-only jobs; 25% growth plus 1GiB',
        fixed_jobs=[dict(id=r['id'],index=i) for r,i in jobs],statistical_samples=0))
    with ProcessPoolExecutor(max_workers=len(jobs),mp_context=mp.get_context('fork')) as pool:
        fs=[pool.submit(job,r,i) for r,i in jobs]
        for f in fs:f.result()
    s.save(s.PUBLIC/'prefetch_tail_status.json',dict(status='PASS',cache_ready=len(jobs),
        indices=list(range(4,9)),candidate=target,statistical_samples=0,
        source=__file__,all_workers_joined=True,main_and_original_prefetch_SOURCE_unchanged=True,
        resource_record=str((WORK/'resources.json').relative_to(s.ROOT))))


if __name__=='__main__':main()
