"""Read-only operational and time-causal DP-prefix analysis of saved RPE runs.

No Q3 kernel, signal loader, count-table generator or scientific evaluator is
imported. Equality of saved inputs establishes reuse opportunities conditional
on the same numerical runtime; it does not recreate absent DP checkpoints.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import heapq
import json
from pathlib import Path
import statistics
import time

CONTRACT = ('statistical_contract', 'options', 'adopted_evaluation')
PHYSICAL = ('configuration', 'signal_packet', 'evaluation_target', 'binding_id', 'total_configuration_id')
PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'


def read(path, default=None):
    return json.loads(path.read_bytes()) if path.exists() else default


def lines(path):
    """As in D2 analyze_rpe_scheduler_logs: ignore an incomplete JSONL tail."""
    out = []
    if path.exists():
        for line in path.read_bytes().splitlines():
            try:
                out.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                pass
    return out


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def summary(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return dict(n=len(values), sum=sum(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=values[min(len(values)-1, int(.9*len(values)))], minimum=values[0], maximum=values[-1])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fraction(value):
    if isinstance(value, dict):
        return Fraction(value['numerator'], value['denominator'])
    return Fraction(value)


def input_identity(request, inputs, source):
    """Mirror D2's cache namespace, then bind the saved numerical SOURCE.

    Labels/allocation IDs and optimizer provenance do not change numerical
    input. The route, physical locators, every contract/scope field, kernel,
    accelerator and recorded interpreter/runtime do. Missing runtime details
    are retained as an explicit limitation, never advertised as a cache HIT.
    """
    case = inputs['cases'][0]
    namespace = dict(method=request['method'], configuration_id=request['configuration_id'], expected=request['expected'],
                     route_commit=request['route']['view_commit'], physical={k: case.get(k) for k in PHYSICAL},
                     contract={k: inputs[k] for k in CONTRACT}, adapter_prefix_version='RPE_TOTAL_DP_PREFIX_V1')
    runtime = source.get('runtime')
    identity = dict(namespace=namespace, kernel_source=inputs.get('kernel_source'), C_source=inputs.get('C_source'),
                    acceleration=inputs.get('acceleration'), interpreter=request['route'].get('interpreter'), runtime=runtime,
                    total_rounds=len(case['allocation']['shots']))
    shots = tuple((r['N_X'], r['N_Y']) for r in case['allocation']['shots'])
    if shots != tuple(zip(request['N_X'], request['N_Y'])):
        raise ValueError('Request / CASE_INPUTS allocation mismatch')
    return digest(identity), shots, bool(runtime)


def interval_stats(jobs, slots):
    """Exact launch/return intervals, not second-rounded progress events."""
    terminal = [j for j in jobs if j.get('end') is not None]
    if not terminal:
        return None
    start, end = min(j['start'] for j in terminal), max(j['end'] for j in terminal)
    events = sorted([(j['start'], 1) for j in terminal] + [(j['end'], -1) for j in terminal])
    active, previous, hist = 0, start, Counter()
    for now, delta in events:
        hist[active] += now-previous
        active += delta
        previous = now
    span = end-start
    occupied = sum(n*t for n, t in hist.items())
    return dict(start=start, end=end, span_seconds=span, sum_terminal_job_wall_seconds=occupied,
                active_union_seconds=span-hist.get(0, 0), terminal_job_peak=max(hist),
                time_weighted_terminal_jobs=None if not span else occupied/span,
                configured_slots=slots,
                idle_slot_seconds=None if slots is None else slots*span-occupied,
                slot_utilization=None if not slots or not span else occupied/(slots*span),
                seconds_by_terminal_jobs=dict(sorted(hist.items())),
                limitation='Only terminal intervals; unreturned launches are excluded, so occupancy is a lower bound if any remain.')


def prefix_opportunities(jobs, boundaries):
    """Use only ledger-verified jobs completed by the target's launch time.

    Sweeping launch and completion events prevents later-completed jobs from
    becoming donors. Prefix shot equality plus the same raw signal and exact
    probability rule is a metadata opportunity; no table is evaluated here.
    """
    jobs = sorted(jobs, key=lambda j: (j['start'], j['job']))
    donors = sorted((j for j in jobs if j.get('donor')), key=lambda j: (j['end'], j['job']))
    available, running, heap = {}, {}, []
    all_keys = set()
    complete_index, rows = 0, []

    def keys(job):
        return [(job['input_identity'], job['shots'][:m+1]) for m in boundaries if m+1 < len(job['shots'])]

    for j in jobs:
        while complete_index < len(donors) and donors[complete_index]['end'] <= j['start']:
            donor = donors[complete_index]
            for key in keys(donor):
                available.setdefault(key, donor['job'])
            complete_index += 1
        while heap and heap[0][0] <= j['start']:
            _, _, old_keys = heapq.heappop(heap)
            for key in old_keys:
                running[key] -= 1
        best, donor_job, active_best = 0, None, 0
        current_keys = keys(j)
        for key in current_keys:
            rounds = len(key[1])
            if key in available and rounds > best:
                best, donor_job = rounds, available[key]
            if running.get(key, 0):
                active_best = max(active_best, rounds)
            all_keys.add(key)
        rows.append(dict(job=j['job'], run=j['run'], launch=j['start'], completed_donor=donor_job,
                         reusable_rounds=best, shared_with_running_rounds=active_best,
                         runtime_fully_recorded=j['runtime_fully_recorded']))
        if j.get('end') is not None:
            for key in current_keys:
                running[key] = running.get(key, 0)+1
            heapq.heappush(heap, (j['end'], j['job'], current_keys))
    return dict(boundaries=boundaries, evaluated_launches=len(rows),
                launches_with_earlier_completed_prefix=sum(r['reusable_rounds'] > 0 for r in rows),
                sum_reusable_round_occurrences=sum(r['reusable_rounds'] for r in rows),
                maximum_reusable_rounds_histogram=dict(sorted(Counter(r['reusable_rounds'] for r in rows).items())),
                hits_by_run=dict(Counter(r['run'] for r in rows if r['reusable_rounds'] > 0)),
                launches_sharing_prefix_with_still_running_job=sum(r['shared_with_running_rounds'] > 0 for r in rows),
                running_only_prefix_opportunities=sum(r['reusable_rounds'] == 0 and r['shared_with_running_rounds'] > 0 for r in rows),
                distinct_checkpoint_keys_if_every_launch_saved=len(all_keys),
                proven_historical_cache_hits=0, actual_checkpoint_bytes=None,
                limitation='Input-compatible opportunity only. Historical frontier/mass checkpoints and complete runtime fingerprints are absent.',
                jobs=rows)


def paired_driver_stages(events):
    starts, durations, unmatched = {}, {}, Counter()
    for event in events:
        label = event.get('event', '')
        if label.endswith('_START'):
            starts[(event.get('cycle'), label[:-6])] = timestamp(event['UTC'])
        elif label.endswith('_END'):
            key = event.get('cycle'), label[:-4]
            if key in starts:
                durations.setdefault(key[1], []).append(timestamp(event['UTC'])-starts.pop(key))
            else:
                unmatched[label] += 1
    return dict(paired_seconds={k: summary(v) for k, v in durations.items()},
                unmatched_starts=[dict(cycle=k[0], stage=k[1], start=v) for k, v in starts.items()],
                unmatched_ends=dict(unmatched),
                limitation='Saved driver UTC has one-second resolution. Durations are phase spans, not CPU or disjoint scientific job work.')


def load_job(folder, run_name, ledger_row):
    launch = read(folder/'LAUNCH.json')
    if launch is None:
        return None
    ret = read(folder/'PROCESS_RETURN.json', {})
    request = read(folder/'REQUEST.json', {})
    inputs = read(folder/'CASE_INPUTS.json')
    source = read(folder/'SOURCE.json', {})
    stages = {v['stage']: timestamp(v['UTC']) for v in lines(folder/'JOB_STAGES.jsonl')}
    worker = {v['stage']: timestamp(v['UTC']) for v in lines(folder/folder.name/'STAGES.jsonl')}
    resource = read(folder/folder.name/'RESOURCE_FINAL.json', {})
    supervision = read(folder/'SUPERVISION.json', {})
    identity, shots, runtime = input_identity(request, inputs, source) if inputs is not None else (None, (), False)
    wall = ret.get('wall_seconds')
    end = None if wall is None else launch['started']+wall
    job = dict(job=folder.name, run=run_name, method=request.get('method'), start=launch['started'], end=end,
               wall_seconds=wall, returncode=ret.get('returncode'), signal=ret.get('signal'),
               input_identity=identity, shots=shots, runtime_fully_recorded=runtime,
               verified_in_ledger=ledger_row is not None,
               adopted_by_rule=None if ledger_row is None else ledger_row['adopted_by_rule'] == 'True',
               status=None if ledger_row is None else ledger_row['status'], cost=request.get('cost'),
               origin=request.get('origin', {}).get('kind'), source=source.get('optimizer_source'),
               tree_CPU_seconds=resource.get('process_tree_CPU_seconds'),
               sampled_tree_peak_RSS_KiB=max(supervision.get('sampled_process_tree_peak_RSS_KiB', {}).values(), default=None),
               dp_workers=(inputs or {}).get('acceleration', {}).get('workers_per_configuration'),
               donor=bool(ledger_row and ret.get('returncode') == 0 and end is not None and 'TOTAL_DP_RETURNED' in worker))
    def elapsed(first, last, stream=stages):
        return None if first not in stream or last not in stream else stream[last]-stream[first]
    job['timings'] = dict(launch_to_worker_start=None if 'JOB_START' not in stages else stages['JOB_START']-job['start'],
        case_build_seconds=elapsed('JOB_START', 'CASE_BUILT'), metadata_seconds=elapsed('CASE_BUILT', 'METADATA_PASS'),
        launch_to_Q3_seconds=None if 'SCIENTIFIC_CALL_START' not in worker else worker['SCIENTIFIC_CALL_START']-job['start'],
        call_to_DP_seconds=elapsed('SCIENTIFIC_CALL_START', 'TOTAL_DP_START', worker),
        DP_seconds=elapsed('TOTAL_DP_START', 'TOTAL_DP_RETURNED', worker),
        post_DP_evaluation_seconds=elapsed('TOTAL_DP_RETURNED', 'EVALUATION_COMPLETE', worker),
        route_return_to_adapter_end_seconds=elapsed('SCIENTIFIC_ROUTE_RETURNED', 'JOB_END'),
        adapter_end_to_process_return_seconds=None if end is None or 'JOB_END' not in stages else end-stages['JOB_END'],
        result_import_seconds=None)
    return job


def run_summary(run, name, jobs):
    spec = read(run/'ROUND.json', {})
    driver = lines(run/'DRIVER_PROGRESS.jsonl')
    progress = lines(run/'PROGRESS.jsonl')
    values = lambda key: [j.get(key) for j in jobs]
    return dict(run=name, directory=str(run), actual_launches=len(jobs), verified_completed_jobs=sum(j['verified_in_ledger'] for j in jobs),
        returned_jobs=sum(j['end'] is not None for j in jobs), failed_returns=Counter(str(j['returncode']) for j in jobs if j['returncode'] != 0),
        log_span_seconds=None if not driver else timestamp(driver[-1]['UTC'])-timestamp(driver[0]['UTC']),
        operational_intervals=interval_stats(jobs, spec.get('executor', {}).get('max_concurrent_jobs')),
        summed_tree_CPU_seconds=sum(j['tree_CPU_seconds'] or 0 for j in jobs), CPU_samples=sum(j['tree_CPU_seconds'] is not None for j in jobs),
        CPU_scope='Sum of saved scientific-worker process-tree CPU through API/result persistence; excludes controller and outer adapter CPU.',
        historical_available_host_CPU_count=None,
        job_wall_seconds=summary(values('wall_seconds')), sampled_tree_peak_RSS_KiB=summary(values('sampled_tree_peak_RSS_KiB')),
        DP_worker_counts=dict(Counter(j['dp_workers'] for j in jobs)),
        timings={key: summary(j['timings'][key] for j in jobs) for key in jobs[0]['timings']} if jobs else {},
        driver_stages=paired_driver_stages(driver), driver_events=driver,
        progress_events=dict(Counter(e.get('event') for e in progress)),
        ready_but_idle_slot_seconds=None, dependency_wait_slot_seconds=None,
        missing_causal_wait_reason='Legacy logs have no ready-queue or result-dependency intervals; occupancy alone cannot attribute idle slots.',
        adopted_verified_jobs=sum(j['adopted_by_rule'] is True for j in jobs),
        result_import_seconds=None,
        qualification='Operational measurements only; no scientific evaluation or speedup experiment.')


def reach_levels(jobs, levels):
    rows = []
    if not jobs:
        return rows
    begin = min(j['start'] for j in jobs)
    for level in levels:
        matching = [j for j in jobs if j.get('end') is not None and j.get('status') == PASS and fraction(j['cost']) <= fraction(level)]
        first = min(matching, key=lambda j: (j['end'], j['job'])) if matching else None
        row = dict(cost_level_RZ=str(level), reached=first is not None)
        if first:
            end = first['end']
            launched = [j for j in jobs if j['start'] <= end]
            complete = [j for j in launched if j.get('end') is not None and j['end'] <= end]
            row.update(job=first['job'], run=first['run'], cost_RZ=str(fraction(first['cost'])),
                wall_from_first_launch_seconds=end-begin, launched_jobs=len(launched), completed_jobs=len(complete),
                completed_tree_CPU_lower_bound_seconds=sum(j['tree_CPU_seconds'] or 0 for j in complete),
                eventual_launched_tree_CPU_upper_bound_seconds=None if any(j['tree_CPU_seconds'] is None for j in launched) else sum(j['tree_CPU_seconds'] for j in launched),
                exact_CPU_at_threshold_seconds=None,
                CPU_limitation='Final per-job CPU lacks its time profile; CPU of still-running jobs at the threshold is unknown.')
        rows.append(row)
    return rows


def analyze(base, round_file, ledger_path):
    started = time.perf_counter()
    spec = read(round_file)
    ledger_rows = list(csv.DictReader(ledger_path.open()))
    ledger = {r['job']: r for r in ledger_rows if r['verified'] == 'True'}
    runs = [(base/p).resolve() for p in spec['executor']['prior_run_dirs'][:4]]
    all_jobs, by_run, seen = [], [], set()
    for run in runs:
        name = str(run.relative_to(run.parent.parent))
        progress = lines(run/'PROGRESS.jsonl')
        launched_here = {e['job'] for e in progress if e.get('event') == 'LAUNCH'}
        jobs, copied = [], 0
        for launch_file in sorted(run.glob('jobs/*/*/LAUNCH.json')):
            folder = launch_file.parent
            if folder.name not in launched_here:
                copied += 1
                continue
            j = load_job(folder, name, ledger.get(folder.name))
            launch_key = j['job'], j['start']
            if launch_key in seen:
                continue
            seen.add(launch_key)
            all_jobs.append(j)
            jobs.append(j)
        entry = run_summary(run, name, jobs)
        entry['launch_artifacts_without_local_progress_excluded'] = copied
        request_folders = list(run.glob('jobs/*/*/REQUEST.json'))
        entry['saved_request_directories'] = len(request_folders)
        entry['request_directories_without_launch_artifact'] = sum(not (p.parent/'LAUNCH.json').exists() for p in request_folders)
        by_run.append(entry)
        print(json.dumps(dict(stage='SAVED_RUN_READ', run=name, actual_launches=len(jobs), copied_excluded=copied)), flush=True)
    eligible = [j for j in all_jobs if j['input_identity'] is not None]
    configs = ([11], [5, 11], list(range(12)))
    prefix = [prefix_opportunities(eligible, boundaries) for boundaries in configs]
    compact_profiles = []
    for boundaries in [[m] for m in range(12)] + [[m, 11] for m in range(11)] + [[5, 8, 11], [8, 10, 11]]:
        profile = prefix_opportunities(eligible, boundaries)
        compact_profiles.append({k: v for k, v in profile.items() if k != 'jobs'})
    return dict(schema='H4_RPE_SEARCH_OPERATIONAL_ANALYSIS_V1', created_UTC=datetime.now(timezone.utc).isoformat(),
        base_source='d8e806265c3ded8e1184181c37c2d7ec71f08ade',
        reused_analysis_source='1eda4ba3f468dfba89823941222843ef5cfe0402:scripts/analyze_rpe_scheduler_logs.py',
        round_source=str(round_file), ledger_source=str(ledger_path), ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        ledger_rows=len(ledger_rows), ledger_run_counts=dict(Counter(r['run'] for r in ledger_rows)),
        actual_launched_jobs=len(all_jobs), verified_launched_jobs=sum(j['verified_in_ledger'] for j in all_jobs),
        by_run=by_run, prefix_opportunities=prefix, checkpoint_boundary_profiles=compact_profiles,
        historical_reach=reach_levels(all_jobs, ['501233190.43', '437894306.43', '414301229.37', '410199237']),
        read_seconds=time.perf_counter()-started, new_real_Q3_calls=0,
        limitations=['Prefix opportunities require numerical-runtime equality not fully recorded in the old Pauli SOURCE.',
            'No historical count table or DP frontier is regenerated. Old OUTCOME summaries cannot create cache checkpoints.',
            'Prefix opportunity is not a wall-time or CPU speedup estimate. No per-round DP timestamps exist in these logs.',
            'Sampled per-job RSS peaks cannot be summed to infer simultaneous memory peak.',
            'CPU availability and host saturation are not inferred from job utilization or CPU service ratios.',
            'Saved CPU is scientific-worker process-tree CPU; complete controller/adapter CPU and import CPU were not recorded.',
            'Driver log spans may include gaps between invocations or failed starts and are not active-controller CPU/runtime.',
            'The four input run locators are selected explicitly from round05 prior_run_dirs; copied prior jobs are excluded using local LAUNCH events.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Read-only original B2 worktree for relative run locators')
    parser.add_argument('--round', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.base.resolve()):
        raise ValueError('Output must be outside the observed B2 worktree')
    result = analyze(args.base, args.round, args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(actual_launched_jobs=result['actual_launched_jobs'], verified_launched_jobs=result['verified_launched_jobs'],
        prefix=[{k: v for k, v in x.items() if k not in ('jobs',)} for x in result['prefix_opportunities']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
