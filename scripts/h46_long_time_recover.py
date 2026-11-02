"""Continue this study after loss of its execution sessions.

Completed trials are read without resampling or compilation. Interrupted trial
directories are retained in a separate attempt directory; their replacement
uses the original candidate, cohort, index, seeds and event law. Historical
contracts and scientific kernels are unchanged.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

import h46_long_time_native as native

ARCHIVE = native.PRIVATE / 'interrupted_run01'
ORIGINAL_SAMPLE_REQUESTS = native.sample_requests


def inventory(plan, cohort, saved, root):
    completed, interrupted, jobs = {}, [], []
    for row in plan['candidates']:
        if row['id'] in saved:
            continue
        n = plan.get('screening_trajectories_per_new_PR', 9) if row['method'] == 'PR' else 1
        completed[row['id']] = []
        for index in range(n):
            job = (row, cohort, index)
            folder = root / cohort / row['id'] / str(index)
            if (folder / 'STOP.json').exists():
                raise ValueError('EXISTING_SCIENTIFIC_STOP:' + str(folder))
            if (folder / 'result.json').exists():
                trial = native.read(folder / 'result.json')
                if (trial['status'] != 'PASS' or trial['candidate'] != row
                        or trial['cohort'] != cohort or trial['index'] != index):
                    raise ValueError('SAVED_TRIAL_IDENTITY:' + str(folder))
                completed[row['id']].append(trial)
            else:
                jobs.append(job)
                if folder.exists():
                    interrupted.append(folder)
    return completed, interrupted, jobs


def preserve_interrupted(folders, root, archive):
    for folder in folders:
        target = archive / 'cost' / folder.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('PRIOR_ATTEMPT_ALREADY_PRESERVED:' + str(target))
        folder.rename(target)


def same_event_requests(row, prep, cohort, index):
    requests, seeds, laws = ORIGINAL_SAMPLE_REQUESTS(row, prep, cohort, index)
    old = ARCHIVE / 'cost' / cohort / row['id'] / str(index) / 'events_PRIVATE.json'
    if old.exists():
        observed = dict(events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests],
                        seeds=seeds, conditional=False, cohort=cohort)
        if json.loads(json.dumps(observed)) != native.read(old):
            raise ValueError('INTERRUPTED_TRIAL_EVENT_OR_SEED_MISMATCH')
    return requests, seeds, laws


def continue_base(plan, saved, completed, jobs):
    from pf_common_run_contract import resources
    cohort = 'EXPLORATION'
    for system in native.study.PREFIX:
        prefixes = sorted({r['prefix'] for r in plan['candidates']
                           if r['system'] == system and r['id'] not in saved})
        if prefixes:
            native.CONTEXT[system] = native.native_context(system, prefixes)
    obs = resources()
    workers = max(1, min(len(jobs) or 1, int(obs['idle_cpu']), obs['available_ram'] // (2 * 2**30)))
    native.save(native.PRIVATE / 'recovery_cost_resources.json', dict(observation=obs, workers=workers))
    initial = sum(map(len, completed.values()))
    progress = dict(total=initial + len(jobs), complete=initial, errors=[],
                    reused_completed_trials=initial, remaining_at_recovery=len(jobs),
                    reused_groups=sum(r['id'] in saved for r in plan['candidates']))
    native.save(native.PRIVATE / 'recovery_cost_progress.json', progress)
    native.sample_requests = same_event_requests
    missing = {}
    for job in jobs:
        missing.setdefault(job[0]['id'], []).append(job)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        pending = {}
        for identity, group in missing.items():
            # A representative must pass before starting its remaining samples.
            ready = [j for j in group if j[2] == 0] or group
            for job in ready:
                pending[pool.submit(native.cost_trial, job)] = job
        while pending:
            future = next(as_completed(pending))
            job = pending.pop(future)
            row = job[0]
            try:
                completed[row['id']].append(future.result())
                progress['complete'] += 1
                if job[2] == 0:
                    for other in missing[row['id']]:
                        if other[2] > 0:
                            pending[pool.submit(native.cost_trial, other)] = other
            except BaseException as exc:
                progress['errors'].append(dict(candidate=row, index=job[2], reason=str(exc)))
            native.save(native.PRIVATE / 'recovery_cost_progress.json', progress)
    if progress['errors']:
        raise RuntimeError('RECOVERY_TRIAL_STOP:see recovery_cost_progress.json')
    results = {}
    for row in plan['candidates']:
        if row['id'] in saved:
            results[row['id']] = {**saved[row['id']], 'candidate': row,
                'reuse_provenance': saved[row['id']].get('origin'), 'origin': 'SAVED_COST_NO_BUILD_OR_COMPILE'}
        else:
            trials = sorted(completed[row['id']], key=lambda t: t['index'])
            expected = plan.get('screening_trajectories_per_new_PR', 9) if row['method'] == 'PR' else 1
            if [t['index'] for t in trials] != list(range(expected)):
                raise ValueError('RECOVERY_TRIAL_COUNT')
            results[row['id']] = native.summarize_trials(row, cohort, trials)
    native.save(native.PUBLIC / 'exploration_costs.json.gz', results)
    native.save(native.PUBLIC / 'exploration_cost_summary.json', dict(progress,
        evaluated_groups=len(results), failed_groups=0, all_workers_joined=True,
        population_tail_guarantee=False, interrupted_attempt_preserved=str(ARCHIVE)))


def run():
    native.study.install()
    import h46_long_time_pipeline as pipeline
    if ARCHIVE.exists():
        raise FileExistsError('RECOVERY_ATTEMPT_ALREADY_EXISTS')
    # Verify the scientific kernels against the recorded pre-interruption
    # provenance. Downstream report/controller changes are separate.
    provenance = native.read(native.PUBLIC / 'scientific_provenance.json')
    for name in ['scripts/h46_long_time_native.py', 'scripts/h46_long_time_signal.py',
                 'scripts/h46_new2_native.py', 'scripts/h46_new2_core.py',
                 'scripts/h46_m5_tol_native.py', 'scripts/h46_m5_tol_saved_input.py',
                 'src/trotterlib/h46_m5_plan.py']:
        if native.sha(native.ROOT / name) != provenance['source_files'][name]:
            raise ValueError('SCIENTIFIC_SOURCE_CHANGED:' + name)
    plan = native.read(native.PUBLIC / 'native_selection.json')
    extension_plan = native.read(native.PUBLIC / 'extension_native_selection.json')
    saved = native.read(native.PRIVATE / 'saved_prices.json')
    root = native.PRIVATE / 'cost'
    completed, interrupted, jobs = inventory(plan, 'EXPLORATION', saved, root)
    ext_completed, ext_interrupted, ext_jobs = inventory(extension_plan, 'EXTENSION_EXPLORATION', {}, root)
    for pid in [1182411, 1185112, 1200942]:
        if Path('/proc', str(pid)).exists():
            raise RuntimeError('PRIOR_EXECUTION_STILL_PRESENT:' + str(pid))
    ARCHIVE.mkdir()
    snapshots = {f.name: native.read(f) for f in [
        native.PRIVATE / 'EXPLORATION_cost_progress.json',
        native.PRIVATE / 'EXTENSION_EXPLORATION_cost_progress.json',
        native.PRIVATE / 'pipeline_status.json']}
    incident = dict(detected_at=datetime.now().astimezone().isoformat(),
        observation='Prior workers and execution sessions absent; no scientific STOP found; cause unconfirmed',
        base_completed=sum(map(len, completed.values())), base_remaining=len(jobs),
        base_interrupted=len(interrupted), extension_completed=sum(map(len, ext_completed.values())),
        extension_remaining=len(ext_jobs), extension_interrupted=len(ext_interrupted),
        interrupted_paths=[str(f.relative_to(root)) for f in interrupted + ext_interrupted],
        prior_progress=snapshots, original_logs_preserved=True, scientific_source_unchanged=True,
        completed_trials_reexecuted=0, seed_and_cohort_unchanged=True, old_grants_reused=False)
    native.save(ARCHIVE / 'interruption.json', incident)
    preserve_interrupted(interrupted + ext_interrupted, root, ARCHIVE)
    native.save(native.PUBLIC / 'execution_interruption.json', incident)
    # Retain the full event check for the resumed extension representative too.
    native.sample_requests = same_event_requests
    pipeline.status('RECOVERING_MAIN_NATIVE_COST', completed=incident['base_completed'],
                    remaining=incident['base_remaining'], interrupted_history=str(ARCHIVE))
    try:
        continue_base(plan, saved, completed, jobs)
        pipeline.run(os.getpid())
    except BaseException as exc:
        pipeline.status('STAGE_STOP', reason=str(exc), traceback=traceback.format_exc(),
                        automatic_scientific_retry=False)
        raise


if __name__ == '__main__':
    run()
