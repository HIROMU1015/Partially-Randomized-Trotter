"""Resume only unfinished B3 trials after the unexplained pool worker exit.

Scientific kernels, seeds, candidates, n and compiler are unchanged. Individual
forked tasks isolate abrupt exits and expose their OS exit status; shared eager
occupation caches avoid rebuilding the same matrices in every worker.
"""
from __future__ import annotations
import fcntl
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import h46_b3 as b
from h46_b3_continue import cached_context

NATIVE=None
ORIGINAL_COMPILE=None


def compiled_with_representative_action(wrapper,fingerprint,representative):
    """A recovered cache hit still supplies QPY for the actual vector check."""
    result,qc,reused=ORIGINAL_COMPILE(wrapper,fingerprint,representative)
    if representative and qc is None:
        from qiskit import qpy
        path=b.PRIVATE/'compiled_cache'/(fingerprint+'_compiled.qpy.gz')
        if path.exists():
            qc=qpy.load(io.BytesIO(gzip.decompress(path.read_bytes())))[0]
        else:
            # A nonrepresentative earlier hit need not have saved QPY. Compile
            # this representative once with the same original compiler kernel.
            from trotterlib.rte import CompilerSettings
            from trotterlib.rte_compiled_cost import transpile_and_measure_cost
            compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
            cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
            if {k:getattr(cost,k) for k in NATIVE.METRICS}!=result['metrics']:
                raise ValueError('RECOVERED_COMPILER_PRICE_MISMATCH')
            qc=cost.transpiled_circuit
            stream=io.BytesIO();qpy.dump(qc,stream)
            temp=path.with_name(path.name+'.partial.'+str(os.getpid()))
            temp.write_bytes(gzip.compress(stream.getvalue(),compresslevel=1,mtime=0));temp.replace(path)
    return result,qc,reused


def trial(job):
    try:
        NATIVE.cost_trial(job)
    except BaseException:
        traceback.print_exc()
        raise


def event_digest(path):
    if not path.exists(): path=path.with_suffix('.json.gz')
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def rss(pid):
    try:
        return int(next(x.split()[1] for x in Path('/proc',str(pid),'status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
    except (FileNotFoundError,StopIteration): return 0


def available_ram():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024


def run_cost(cohort,planfile):
    global NATIVE,ORIGINAL_COMPILE
    b.native_context=cached_context;NATIVE=b.setup()
    if ORIGINAL_COMPILE is None: ORIGINAL_COMPILE=NATIVE.compile_cached
    NATIVE.compile_cached=compiled_with_representative_action
    from h46_m5_contract import core
    from pf_common_run_contract import resources
    plan=b.read(b.PUBLIC/planfile)
    lock=(b.PRIVATE/(cohort+'_runner.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    groups={};jobs=[];old_events={};known=set();seeds=[];lookup={r['id']:r for r in plan['candidates']}
    for folder in (b.BASE_PRIVATE,b.PRIVATE):
        for p in folder.glob('*_seeds.json.gz'):
            if folder!=b.PRIVATE or p.name!=cohort+'_seeds.json.gz': known.update(b.read(p))
    for s in b.signal.PREFIX:
        cfg=b.read(b.signal.ORIGIN/(s+'_PR_EXPLORATION/config_APPROVED_PRIVATE.json'))
        known.update(b.read(cfg['excluded_seeds_path'])['values'])
    for r in plan['candidates']:
        groups[r['id']]=[]
        for i in range(plan.get('n_by_candidate',{}).get(r['id'],9)):
            for step in core.schedule(r['formula'],r['q'],'ABS_TIME_PER_OUTER',r['R_outer'],T=r['T']):
                seed=b.trial_seed(r,cohort,i,step['outer'],step['piece'])
                if seed in known: raise ValueError('SEED_COLLISION')
                known.add(seed);seeds.append(seed)
            directory=b.PRIVATE/'cost'/cohort/r['id']/str(i)
            if (directory/'result.json').exists():
                saved=b.read(directory/'result.json')
                if saved['status']!='PASS' or saved['candidate']!=r or saved['cohort']!=cohort or saved['index']!=i:
                    raise ValueError('COMPLETED_TRIAL_BINDING')
                groups[r['id']].append(saved);continue
            if directory.exists():
                if (directory/'STOP.json').exists(): raise ValueError('PRESERVED_SCIENTIFIC_STOP')
                preserved=directory.with_name(directory.name+'_unfinished_run01')
                if preserved.exists(): raise ValueError('UNFINISHED_ARCHIVE_ALREADY_EXISTS')
                directory.rename(preserved)
                if (preserved/'events_PRIVATE.json').exists(): old_events[r['id'],i]=preserved/'events_PRIVATE.json'
            jobs.append((r,cohort,i))
    path=b.PRIVATE/(cohort+'_seeds.json.gz')
    if path.exists() and b.read(path)!=seeds: raise ValueError('FIXED_SEED_REGISTRY_CHANGED')
    b.save(path,seeds)
    for s in b.signal.PREFIX: NATIVE.CONTEXT[s]=cached_context(s,[0])
    obs=resources();workers=max(1,min(len(jobs) or 1,int(obs['idle_cpu'])))
    # Forecast RAM from completed largest-event trials, rather than applying a
    # fixed worker/RAM cap from another study. Physical availability is also
    # checked before each launch. RSS is a planning estimate, not a guarantee.
    slopes={}
    completed=[t for values in groups.values() for t in values]
    for s in b.signal.PREFIX:
        calibration=[t for t in completed if t['candidate']['system']==s]
        if not calibration:
            prior=b.PUBLIC/'b3_exploration_costs.json.gz'
            if prior.exists():
                calibration=[b.read(name) for g in b.read(prior).values() if g['candidate']['system']==s for name in g['trial_files']]
        maximum=max((t['candidate']['s'] for t in calibration),default=1)
        peak=max((t['peak_RSS_bytes'] for t in calibration if t['candidate']['s']==maximum),default=2**30)
        slopes[s]=max(0,(peak-512*2**20)/maximum)
    estimate=lambda j:1024*2**20+slopes[j[0]['system']]*j[0]['s']
    b.save(b.PRIVATE/(cohort+'_recovery_resources.json'),dict(observation=obs,workers=workers,
        RSS_per_short_step_calibration=slopes,per_task_exit_isolation=True,shared_number_operator_cache=True,
        old_completed_trials=len(completed),pending_trials=len(jobs),scientific_conditions_changed=False))
    progress=dict(total=len(completed)+len(jobs),complete=len(completed),errors=[],active=0,
        reused_completed=len(completed),replayed_unfinished_event_matches=0,worker_exitcodes={})
    ready0={r['id'] for r in plan['candidates'] if any(t['index']==0 for t in groups[r['id']])}
    failed=set();active={};pending=list(jobs);context=mp.get_context('fork');updated=0
    while pending or active:
        for pid,(process,job) in list(active.items()):
            if process.is_alive(): continue
            process.join();del active[pid];r,_,i=job
            progress['worker_exitcodes'][str(process.exitcode)]=progress['worker_exitcodes'].get(str(process.exitcode),0)+1
            directory=b.PRIVATE/'cost'/cohort/r['id']/str(i)
            if process.exitcode==0 and (directory/'result.json').exists():
                t=b.read(directory/'result.json')
                if t['status']!='PASS': raise ValueError('UNACCEPTED_TRIAL')
                if (r['id'],i) in old_events:
                    if event_digest(old_events[r['id'],i])!=event_digest(directory/'events_PRIVATE.json'):
                        raise ValueError('UNFINISHED_TRAJECTORY_EVENT_REPLAY_MISMATCH')
                    progress['replayed_unfinished_event_matches']+=1
                groups[r['id']].append(t);progress['complete']+=1
                if i==0: ready0.add(r['id'])
            else:
                failed.add(r['id']);error=dict(candidate=r,index=i,exitcode=process.exitcode,
                    native_STOP=(directory/'STOP.json').exists(),reason='ISOLATED_TRIAL_EXIT')
                progress['errors'].append(error);b.save(directory/'technical_exit.json',error)
        reserved=sum(max(0,estimate(job)-rss(pid)) for pid,(process,job) in active.items())
        free=max(0,min(available_ram(),obs['available_ram'])-reserved)
        candidates=sorted((j for j in pending if j[0]['id'] not in failed and (j[2]==0 or j[0]['id'] in ready0)),
            key=lambda j:j[0]['s'],reverse=True)
        for job in candidates:
            if len(active)>=workers: break
            if estimate(job)>free: continue
            process=context.Process(target=trial,args=(job,));process.start()
            active[process.pid]=(process,job);pending.remove(job);free-=estimate(job)
        # Do not launch further work for a task whose native/scientific check
        # failed. Independent candidates already running are allowed to finish.
        pending=[j for j in pending if j[0]['id'] not in failed]
        progress['active']=len(active);progress['pending']=len(pending)
        if time.monotonic()-updated>5:
            b.save(b.PRIVATE/(cohort+'_progress.json'),progress);updated=time.monotonic()
        if pending and not active and not candidates: raise ValueError('NO_READY_TRIAL')
        if pending and not active and free<min(map(estimate,pending)):
            b.save(b.PRIVATE/(cohort+'_resource_wait.json'),dict(available=free,next_required=min(map(estimate,pending))))
        time.sleep(.2)
    results={r['id']:NATIVE.summarize_trials(r,cohort,groups[r['id']]) for r in plan['candidates'] if r['id'] not in failed}
    b.save(b.PUBLIC/(cohort.lower()+'_costs.json.gz'),results)
    b.save(b.PRIVATE/(cohort+'_progress.json'),progress)
    b.save(b.PUBLIC/(cohort.lower()+'_cost_summary.json'),dict(progress,groups=len(results),all_workers_joined=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if failed: raise ValueError('ISOLATED_TASK_STOPS_PRESERVED')
    return dict(progress,groups=len(results))


if __name__=='__main__':
    import h46_b3_finish as finish
    b.run_cost=run_cost
    print(json.dumps(run_cost('B3_EXPLORATION','native_selection.json')),flush=True)
    finish.pipeline()
