"""Reuse B3's exact cache prefetch; never adds or accepts statistical samples."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
from pathlib import Path
import time
from types import SimpleNamespace
import h46_full_h_t20_opt as study
from h46_full_h_t20_aer import simulate_statevector
import h46_b3_prefetch as previous

study.aer_statevector=simulate_statevector
COHORT='FULL_H_T20_EXPLORATION_V2'
WORK=study.PRIVATE/'prefetch'


def install():
    native=study.setup()
    previous.b=SimpleNamespace(setup=study.setup,PRIVATE=study.PRIVATE,ROOT=study.ROOT,
        native_context=lambda system,prefixes:study.context(system))
    previous.COHORT=COHORT;previous.WORK=WORK
    return native


def cache_job(row,index):
    install()
    return previous.precompile(row,index)


def live(pid):
    try:
        proc=Path('/proc',str(pid))
        return (proc/'cwd').resolve()==study.ROOT and b'scripts/run_h46_full_h_t20_opt.py' in (proc/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return False


def main(pid):
    install();WORK.mkdir(exist_ok=True)
    plan=study.read(study.PUBLIC/'price_exploration_v2.json');rows=plan['candidates']
    for system in ('H4','H6'):study.context(system)
    historical=study.read(study.ROOT/'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/price_trials.json')
    slopes={system:max(t['peak_RSS_bytes']/t['candidate']['s'] for t in historical if t['candidate']['system']==system) for system in ('H4','H6')}
    jobs=[(r,i) for r in rows for i in range(1,9)]
    active={};done=[];errors=[];fresh=[];seen=set();updated=0
    from pf_common_run_contract import resources
    with ProcessPoolExecutor(max_workers=len(jobs),mp_context=mp.get_context('fork')) as pool:
        while jobs or active:
            for future,(r,i) in list(active.items()):
                if not future.done():continue
                del active[future]
                try:
                    future.result();done.append(study.read(previous.job_path(r,i)))
                except BaseException as exc:errors.append(dict(id=r['id'],index=i,reason=str(exc)))
            progress=study.read(study.PRIVATE/'exploration_v2_progress.json')
            if not live(pid) or progress['errors'] or errors or progress['complete']==progress['total']:jobs=[]
            for r in rows:
                for i in range(9):
                    path=previous.trial_path(r,i)/'result.json'
                    if path in seen or not path.exists():continue
                    seen.add(path);v=study.read(path)
                    if v['status']=='PASS' and i==0:fresh.append(v)
            obs=resources()
            science_pids=previous.children(pid)
            live_rss=sum(previous.rss(p) for p in science_pids)
            live_rows=[r for r in rows for i in range(9) if previous.trial_path(r,i).exists() and not (previous.trial_path(r,i)/'result.json').exists()]
            estimate_main=lambda r:previous.memory_estimate(r,slopes,fresh)
            forecast=sum(estimate_main(r) for r in live_rows)
            forecast+=max(0,len(science_pids)-len(live_rows))*max(map(estimate_main,rows))
            # Use full-trial measurements as a conservative cache-job estimate:
            # cache-only jobs omit occupation and statevector checks, not gates.
            estimate=lambda r:previous.memory_estimate(r,slopes,fresh)
            aux_pids=previous.children(__import__('os').getpid())
            aux_rss=sum(previous.rss(p) for p in aux_pids)
            reserve=max(0,sum(estimate(r) for r,i in active.values())-aux_rss)
            free=previous.headroom(obs['available_ram'],forecast,live_rss,reserve)
            slots=max(0,min(int(obs['idle_cpu']),obs['affinity']-len(science_pids)-len(active)))
            for r,i in sorted(jobs,key=lambda j:estimate(j[0])):
                if previous.trial_path(r,i).exists() or previous.job_path(r,i).exists():jobs.remove((r,i));continue
                if slots<=0 or estimate(r)>free:break
                active[pool.submit(cache_job,r,i)]=(r,i);jobs.remove((r,i))
                free-=estimate(r);slots-=1
            if time.monotonic()-updated>15 or not (jobs or active):
                axes=[a for d in done for a in d['axes'].values()]
                study.save(study.PUBLIC/'prefetch_status.json',dict(stage='RUNNING' if jobs or active else ('FAILED_DRAINED' if errors else 'COMPLETE'),
                    main_pid=pid,active=len(active),pending=len(jobs),cache_ready=len(done),errors=errors,
                    new_compiled_axes=sum(not a['cache_hit'] for a in axes),already_cached_axes=sum(a['cache_hit'] for a in axes),
                    statistical_samples=0,source='scripts/h46_b3_prefetch.py:precompile',controller_source=__file__,
                    observation=obs,main_future_peak_forecast=forecast,remaining_memory_headroom=free,
                    original_controller_and_SOURCE_unchanged=True,main_native_and_compiled_checks_retained=True))
                updated=time.monotonic()
            if jobs or active:time.sleep(2)
    if errors:raise RuntimeError('PREFETCH_ERROR_DRAINED_MAIN_UNCHANGED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main-pid',type=int,required=True)
    main(p.parse_args().main_pid)
