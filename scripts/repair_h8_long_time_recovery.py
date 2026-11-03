"""Finish original Stage B samples after a JSON representation recovery error.

Keep the first recovery's healthy PF workers running. Repeat only the six PR
trials rejected before circuit construction; compare their original events as
canonical JSON. No additional candidate, index, seed or sample is introduced.
"""
import argparse
import gzip
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

import h8_long_time_adapter as a
import h8_long_time_runtime as runtime
import resume_h8_long_time_execution as recovery

ORIGINAL = {}
OUTPUT = {}


def same_event_metadata(saved, actual):
    # JSON arrays restore as lists even when to_dict originally returned tuples.
    # All values, ordering, seeds and event parameters must still match.
    a.require(a.digest(saved) == a.digest(actual), 'EXACT_ORIGINAL_EVENTS_ONLY')


def price_original(job):
    import h46_long_time_native as native
    row, endpoint, cohort, index, batch = job
    original = ORIGINAL[batch]
    output = OUTPUT[batch]
    native.PRIVATE = output
    compile_new = native.compile_cached
    sample = native.sample_requests

    def original_events(row, prep, cohort, index):
        requests, seeds, laws = sample(row, prep, cohort, index)
        actual = dict(seeds=seeds, events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests],
                      conditional=False, cohort=cohort)
        same_event_metadata(a.read(original / 'prices' / row['id'] / str(index) / 'events_PRIVATE.json'), actual)
        return requests, seeds, laws

    def saved_compile(wrapper, fingerprint, representative):
        from qiskit import qpy
        cache = original / 'compiled_cache'
        path = cache / (fingerprint + '.json')
        qpath = cache / (fingerprint + '_compiled.qpy.gz')
        if not path.exists() or (representative and not qpath.exists()):
            return compile_new(wrapper, fingerprint, representative)
        value = a.read(path)
        a.require(value['fingerprint'] == fingerprint and value['compiler'] == a.COMPILER, 'ORIGINAL_CACHE_IDENTITY')
        qc = None
        if representative:
            qc, = qpy.load(io.BytesIO(gzip.decompress(qpath.read_bytes())))
            a.require(qc.count_ops().get('rz', 0) == value['metrics']['rz_count'], 'ORIGINAL_QPY_PRICE')
        return value, qc, True

    native.sample_requests, native.compile_cached = original_events, saved_compile
    directory = output / 'prices' / row['id'] / str(index)
    try:
        return runtime.price_job((row, endpoint, cohort, index, str(directory)))
    except BaseException as exc:
        a.save(output / ('failure_' + row['id'] + '_' + str(index) + '.json'),
               dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise
    finally:
        native.sample_requests, native.compile_cached = sample, compile_new


def run(private):
    import h46_long_time_native as native
    signal = a.read(private / 'stage_A_combined.json')
    endpoints = {r['endpoint']['candidate']['id']: r['endpoint'] for r in signal['results']}
    scopes, configs, completed, missing, seeds = {}, {}, {}, {}, {}
    for name, filename in [('stage_B01', 'config_B.json'), ('stage_B02', 'config_B02.json')]:
        original = private / 'runs' / name
        config = a.read(private / filename)
        scope = a.read(original / 'scope.json')
        a.preflight(config)
        runtime.validate_scope('B', scope, config)
        a.require(a.sha(scope['stage_A']) == scope['stage_A_sha256']
                  and signal['binding'] == config['binding']
                  and signal['source_files'] == config['signal_source_files'], 'FIXED_SIGNAL_BYTES_INPUT_SOURCE')
        a.check_reference(signal['reference'], config['binding'])
        completed[name], missing[name] = recovery.trial_plan(scope, endpoints, original)
        restored = recovery.restored_seeds(config, scope, original)
        a.require(not (set(restored) & set(seeds)), 'DISJOINT_ORIGINAL_TRIALS')
        seeds.update(restored)
        scopes[name], configs[name], ORIGINAL[name] = scope, config, original
    root = private / 'recovery02'
    root.mkdir(exist_ok=False)
    for name in scopes:
        OUTPUT[name] = root / name
        OUTPUT[name].mkdir()
    jobs = [(*job, name) for name in scopes for job in missing[name] if job[0]['method'] == 'PR']
    a.require(len(jobs) == 6 and sum(map(len, completed.values())) == 36, 'FIXED_INTERRUPTED_POOL')
    provenance = dict(runtime_SOURCE=configs['stage_B01']['SOURCE'], recovery_script_sha256=a.sha(__file__),
        new_samples=0, workers=min(len(jobs), len(os.sched_getaffinity(0))),
        event_comparison='EXACT_CANONICAL_JSON', healthy_PF_workers_untouched=True)
    a.save(root / 'recovery.json', provenance)
    runtime.CONTEXT = runtime.Context(configs['stage_B01'], root, [7, 8], signal=False)
    native.trial_seed = lambda row, cohort, index, outer, piece: seeds[(row['id'], index, outer, piece)]
    print(json.dumps(dict(status='PR_RECOVERY_RUNNING', trials=len(jobs), **provenance)), flush=True)
    new_trials = runtime.map_jobs(price_original, jobs, provenance['workers'])
    print(json.dumps(dict(status='PR_RECOVERY_COMPLETE', trials=len(new_trials))), flush=True)
    # The first recovery may finish with its representation-only HOLD. Preserve
    # that evidence; accept no other failure and never stop its healthy workers.
    for name in scopes:
        pf_jobs = [job for job in missing[name] if job[0]['method'] != 'PR']
        paths = [private / 'recovery01' / name / 'prices' / job[0]['id'] / str(job[3]) / 'result.json'
                 for job in pf_jobs]
        while not all(p.exists() for p in paths):
            hold = private / 'recovery01' / name / 'HOLD.json'
            if hold.exists():
                a.require(a.read(hold)['reason'] == 'ValueError:EXACT_ORIGINAL_EVENTS_ONLY',
                          'UNRELATED_RECOVERY_HOLD')
            time.sleep(15)
        partials = [a.read(path) for path in paths]
        trials = completed[name] + partials + [r for r in new_trials if r['candidate_id'] in
            {row['id'] for row in scopes[name]['candidates']}]
        summaries = {}
        for row in scopes[name]['candidates']:
            selected = sorted((r for r in trials if r['candidate_id'] == row['id']), key=lambda r: r['index'])
            n = scopes[name]['n'] if row['method'] == 'PR' else 1
            a.require([r['index'] for r in selected] == list(range(n)), 'EXACT_ORIGINAL_COHORT_N')
            stats = a.paired_G(selected, endpoints[row['id']], scopes[name]['cohort'])
            if row['method'] == 'PR':
                observed = {str(k): sum(r['orders'].get(str(k), 0) for r in selected) for k in (0, 2, 4)}
                stats['rarity'] = native.rarity(selected[0]['laws'], runtime.CONTEXT.preps[row['prefix']], row['K'], n, observed)
            summaries[row['id']] = stats
        meta = dict(provenance, reused_trials=len(completed[name]), missing_trials=len(missing[name]),
                    original_scope_sha256=a.sha(ORIGINAL[name] / 'scope.json'))
        result = dict(status='PASS', stage='B', binding=configs[name]['binding'], source_files=configs[name]['source_files'],
            SOURCE=configs[name]['SOURCE'], signal_SOURCE=signal['SOURCE'], scope_digest=a.digest(scopes[name]),
            environment={n: importlib.metadata.version(n) for n in ('qiskit', 'numpy', 'scipy', 'numba')},
            trials=trials, summaries=summaries, pooled=False, recovery=meta)
        a.save(OUTPUT[name] / 'result.json', result)
        a.save(ORIGINAL[name] / 'result.json', result)
    from report_h8_long_time_execution import collect
    result = collect(private, a.ROOT / 'artifacts/resource_applicability/track_a_h8_long_time/2026-11-03')
    a.save(private / 'collection_status.json', dict(status='COLLECTED_EXPLORATION_ONLY',
        candidates=len(result['rows']), trajectories=len(result['trials']), confirmation='NOT_RUN',
        native_SOURCE=result['native_SOURCE']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    try:
        run(args.private.resolve())
    except BaseException as exc:
        target = args.private / 'recovery02/HOLD.json'
        if target.parent.exists():
            a.save(target, dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise
