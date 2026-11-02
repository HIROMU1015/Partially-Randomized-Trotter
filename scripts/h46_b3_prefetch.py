"""Backfill idle CPUs with the exact pending exploration compiler cache entries.

The live controller and its pinned SOURCE are untouched. No trial result or
statistical sample is produced here. The original compiler's fingerprint lock
prevents duplicate compilation; the original controller still performs every
required native/compiled action check and accepts the trial itself.
"""
from __future__ import annotations
import fcntl
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
import h46_b3 as b
from h46_b3_durable import save

COHORT = 'B3_EXPLORATION'
WORK = b.PRIVATE/'parallel_prefetch'
STATUS = b.PUBLIC/'parallel_prefetch.json'


def job_path(row, index):
    return WORK/(row['id']+'_'+str(index)+'.json')


def trial_path(row, index):
    return b.PRIVATE/'cost'/COHORT/row['id']/str(index)


def precompile(row, index):
    """Use the unchanged sampler, signed native builder and compiler cache."""
    import faulthandler
    faulthandler.enable(all_threads=True)
    started = time.monotonic()
    native = b.setup()
    native.save = save
    from trottertracks.resource_applicability import ax2b_native_df_v5 as gates
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    ctx = b.native_context(row['system'], [0])
    prep = ctx.preparations[0]
    requests, seeds, _ = native.sample_requests(row, prep, COHORT, index)
    scalar = ctx.ham.constant+prep.extracted_identity_coefficient
    axes = {}
    with use_fingerprint_reuse() as fp:
        evolution = native.compose(row, prep, requests, scalar, fp.fingerprint)
        del requests
        for axis in ('cosine', 'sine'):
            wrapper = gates.build_native_hadamard_wrapper(
                evolution, axis=axis, include_measurement=True,
                max_instructions=len(evolution.circuit.data)+4)
            fingerprint = fp.fingerprint(wrapper)
            tick = time.monotonic()
            compiled, qc, reused = native.compile_cached(wrapper, fingerprint, index==0)
            if fp.fingerprint(wrapper) != fingerprint:
                raise ValueError('COMPILER_MUTATED_INPUT')
            axes[axis] = dict(fingerprint=fingerprint, cache_hit=reused,
                              seconds=time.monotonic()-tick,
                              compiler=compiled['compiler'])
            del wrapper, qc
    save(job_path(row, index), dict(status='CACHE_READY', candidate=row,
        cohort=COHORT, index=index, seeds=seeds, axes=axes,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        wall_seconds=time.monotonic()-started, statistical_samples=0,
        native_action_checks_deferred_to_original_controller=True))


def rss(pid):
    try:
        return 1024*int(next(line.split()[1] for line in
            Path('/proc', str(pid), 'status').read_text().splitlines()
            if line.startswith('VmRSS:')))
    except (FileNotFoundError, StopIteration):
        return 0


def children(pid):
    try:
        return [int(p) for p in Path('/proc', str(pid), 'task', str(pid), 'children').read_text().split()]
    except FileNotFoundError:
        return []


def live_main(pid):
    try:
        proc = Path('/proc', str(pid))
        return (proc/'cwd').resolve()==b.ROOT and b'scripts/h46_b3_spawn_continue.py' in (proc/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False


def headroom(available, forecast, live_rss, auxiliary_reserve):
    # Reserve forecast future growth of the main and auxiliary workers.
    return max(0, available-max(0, forecast-live_rss)-auxiliary_reserve)


def memory_estimate(row, slopes, measurements, *, auxiliary=False):
    """Refresh only from completed same-system processes, with growth margin.

    Main forecasts use fresh full trials including all action checks. At most
    a factor of two in event count is extrapolated. No current low RSS is used
    as evidence of a low eventual peak. Auxiliary measurements skip no checks
    in the main controller; they describe only the smaller cache-building job.
    """
    values=[t for t in measurements if t['candidate']['system']==row['system']]
    slope=slopes[row['system']]
    if not values:
        return 2**30+slope*row['s']
    largest=max(t['candidate']['s'] for t in values)
    if not auxiliary:
        if largest*2<row['s']:
            return 2**30+slope*row['s']
        calibration=[t for t in values if t['candidate']['s']>=largest/2]
        measured=max(max(0,t['peak_RSS_bytes']-512*2**20)/t['candidate']['s'] for t in calibration)
        revised=2**30+1.25*measured*row['s']
        original=2**30+slope*row['s']
        exceeded=any(t['peak_RSS_bytes']>2**30+slope*t['candidate']['s'] for t in calibration)
        return max(original,revised) if exceeded else min(original,revised)
    # Retain the original per-event slope for auxiliary jobs; only their fixed
    # startup allocation is calibrated from their own completed peak RSS.
    base=max(512*2**20,max(t['peak_RSS_bytes']-slope*t['candidate']['s'] for t in values))
    return 1.25*base+slope*row['s']


def process_started_epoch(pid):
    boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    start=int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    return boot+start/os.sysconf('SC_CLK_TCK')


def main(main_pid):
    WORK.mkdir(exist_ok=True)
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not live_main(main_pid):
            raise ValueError('LIVE_CONTROLLER_BINDING')
        provenance = b.read(b.PUBLIC/'scientific_provenance.json')
        for name, expected in provenance['new_source_files'].items():
            if b.signal.sha(b.ROOT/name)!=expected:
                raise ValueError('PINNED_SCIENTIFIC_SOURCE_CHANGED:'+name)
        rows = b.read(b.PUBLIC/'native_selection.json')['candidates']
        calibration = b.read(b.PRIVATE/(COHORT+'_recovery_resources.json'))
        slopes = calibration['RSS_per_short_step_calibration']
        main_estimate = lambda row: 2**30+slopes[row['system']]*row['s']
        jobs = sorted([(r, i) for r in rows for i in range(9)],
            key=lambda j: (main_estimate(j[0]), j[0]['id'], j[1]))
        jobs = [(r, i) for r, i in jobs if not trial_path(r,i).exists() and not job_path(r,i).exists()]
        from pf_common_run_contract import resources
        context = mp.get_context('spawn')
        active = {}; completed = []; errors = []; skipped = 0
        for path in WORK.glob('*.json'):
            result = b.read(path)
            if result.get('status')=='CACHE_READY': completed.append(result)
        started_epoch=process_started_epoch(main_pid)
        fresh=[];measured_files=set();refreshed=0
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        updated = 0
        while jobs or active:
            for pid, (process, row, index) in list(active.items()):
                if process.is_alive(): continue
                process.join(); del active[pid]
                path = job_path(row,index)
                if process.exitcode==0 and path.exists():
                    completed.append(b.read(path))
                else:
                    errors.append(dict(id=row['id'], index=index, exitcode=process.exitcode))
            progress = b.read(b.PRIVATE/(COHORT+'_progress.json'))
            if not live_main(main_pid) or progress['complete']==progress['total'] or errors or progress['errors']:
                jobs = []  # Drain healthy auxiliary workers; never stop science.
            obs = resources()
            if time.monotonic()-refreshed>10:
                for row in rows:
                    for index in range(9):
                        path=trial_path(row,index)/'result.json'
                        if path in measured_files or not path.exists(): continue
                        measured_files.add(path)
                        if path.stat().st_mtime<started_epoch: continue
                        result=b.read(path)
                        if result['status']=='PASS' and result['index']==0: fresh.append(result)
                refreshed=time.monotonic()
            estimate=lambda row:memory_estimate(row,slopes,completed,auxiliary=True)
            # Active numeric trial directories identify live main jobs. Account
            # conservatively for a just-spawned child before it creates one.
            live_rows = [r for r in rows for i in range(9)
                if trial_path(r,i).exists() and not (trial_path(r,i)/'result.json').exists()]
            science_pids = [pid for pid in children(main_pid) if rss(pid)>128*2**20]
            forecast = sum(memory_estimate(r,slopes,fresh) for r in live_rows)
            forecast += max(0,max(len(science_pids),progress['active'])-len(live_rows))*max(map(main_estimate,rows))
            auxiliary_reserve = sum(max(0,estimate(r)-rss(pid)) for pid,(_,r,_) in active.items())
            free = headroom(obs['available_ram'],forecast,sum(map(rss,science_pids)),auxiliary_reserve)
            slots = max(0,min(math.floor(obs['idle_cpu']),
                obs['affinity']-max(progress['active'],len(science_pids))-len(active)))
            for row,index in sorted(jobs,key=lambda j:estimate(j[0])):
                if trial_path(row,index).exists() or job_path(row,index).exists():
                    jobs.remove((row,index));skipped+=1;continue
                if slots<=0: break
                required = estimate(row)
                if required>free: break  # shortest first, so larger jobs cannot fit
                process = context.Process(target=precompile,args=(row,index))
                process.start();active[process.pid]=(process,row,index)
                jobs.remove((row,index));slots-=1;free-=required
            if time.monotonic()-updated>10 or not (jobs or active):
                axes = [v for t in completed for v in t['axes'].values()]
                save(STATUS,dict(stage='RUNNING' if jobs or active else ('FAILED_DRAINED' if errors else 'COMPLETE'),
                    updated_epoch=time.time(), source='scripts/h46_b3_prefetch.py', source_sha256=source_hash,
                    main_pid=main_pid, active=len(active), pending=len(jobs), cache_ready_trials=len(completed),
                    new_compiled_axes=sum(not a['cache_hit'] for a in axes),
                    already_cached_axes=sum(a['cache_hit'] for a in axes), skipped_main_started=skipped,
                    errors=errors, observation=obs, main_peak_forecast_bytes=forecast,
                    remaining_memory_headroom_bytes=free, statistical_samples=0,
                    original_controller_unchanged=True, main_future_memory_growth_reserved=True,
                    full_trial_memory_calibration=dict(fresh_completed=len(fresh),growth_margin=1.25,
                        maximum_event_extrapolation_factor=2,
                        source='PASS full trials since current science controller start',
                        fallback='original per-system calibration until comparable full trial completes'),
                    initial_idle_controller_preserved='parallel_prefetch/controller_v1_SOURCE.py and controller_v1_status.json; no workers or compiles',
                    original_native_and_compiled_action_checks_retained=True,
                    sampler_and_compiler='UNCHANGED_FIXED_COHORT_SEEDS_AND_FINGERPRINT_LOCKED_CACHE'))
                updated=time.monotonic()
            if jobs or active: time.sleep(2)
        if errors: raise RuntimeError('PREFETCH_FAILED; main science and completed cache entries retained')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--main-pid',type=int,required=True)
    main(parser.parse_args().main_pid)
