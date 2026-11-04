"""Recover the interrupted fixed Stage B cohort without drawing new samples.

The scientific runtime and original partial files remain unchanged. Completed
trials are reused; missing trials use their original seeds/events in a new
directory. This is a recovery entry point, not an additional scientific scope.
"""
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import h8_long_time_adapter as a


def restored_seeds(config, scope, original):
    values = []
    mapping = {}
    for row in scope['candidates']:
        if row['method'] != 'PR':
            continue
        for index in range(scope['n']):
            for piece in a.schedule(row):
                key = (row['id'], index, piece['outer'], piece['piece'])
                seed = int(a.digest(['H8_LONG_TIME_V1', config['SOURCE'], config['source_files'],
                    config['binding'], scope['cohort'], *key])[:16], 16)
                mapping[key] = seed
                values.append(seed)
    saved = a.read(original / 'seeds_PRIVATE.json')
    a.require(saved == dict(cohort=scope['cohort'], values=values)
              and len(values) == len(set(values)), 'EXACT_ORIGINAL_SEEDS_ONLY')
    return mapping


def trial_plan(scope, endpoints, original):
    a.require(not (original / 'HOLD.json').exists()
              and not (original / 'result.json').exists(), 'INTERRUPTED_WITHOUT_HOLD_ONLY')
    completed, missing = [], []
    for row in scope['candidates']:
        ep = endpoints[row['id']]
        for index in range(scope['n'] if row['method'] == 'PR' else 1):
            directory = original / 'prices' / row['id'] / str(index)
            path = directory / 'result.json'
            if not path.exists():
                a.require((directory / 'events_PRIVATE.json').is_file(), 'ORIGINAL_SAMPLED_EVENTS_REQUIRED')
                missing.append((row, ep, scope['cohort'], index))
                continue
            value = a.read(path)
            a.require(value['status'] == 'PASS' and value['candidate_id'] == row['id']
                      and value['index'] == index and value['endpoint_digest'] == a.digest(ep)
                      and value['cohort'] == scope['cohort'] and value['conditional'] is False,
                      'COMPLETE_TRIAL_IDENTITY')
            events = a.read(directory / 'events_PRIVATE.json')
            a.require(value['event_digest'] == a.digest(events['events']), 'SAVED_EVENTS_DIGEST')
            if index == 0:
                a.require(len(value['checks']) == 7 and all(
                    c['phase_alignment'] is False and 0 <= c['error'] < c['tolerance']
                    for c in value['checks']), 'COMPLETE_REPRESENTATIVE_ONLY')
            completed.append(value)
    return completed, missing


def resume_batch(private, batch):
    import h8_long_time_runtime as runtime
    import h46_long_time_native as native
    original = private / 'runs' / batch
    config = a.read(private / ('config_B.json' if batch == 'stage_B01' else 'config_B02.json'))
    scope = a.read(original / 'scope.json')
    a.preflight(config)
    runtime.validate_scope('B', scope, config)
    a.require(a.sha(scope['stage_A']) == scope['stage_A_sha256'], 'FIXED_STAGE_A_BYTES')
    signal = a.read(scope['stage_A'])
    a.require(signal['status'] == 'PASS' and signal['binding'] == config['binding']
              and signal['source_files'] == config['signal_source_files'], 'FIXED_SIGNAL_INPUT_SOURCE')
    a.check_reference(signal['reference'], config['binding'])
    endpoints = {r['endpoint']['candidate']['id']: r['endpoint'] for r in signal['results']}
    a.require(all(endpoints[r['id']]['status'] == 'ELIGIBLE' for r in scope['candidates']), 'ELIGIBLE_ONLY')
    completed, missing = trial_plan(scope, endpoints, original)
    seed_map = restored_seeds(config, scope, original)
    output = private / 'recovery01' / batch
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(runtime_SOURCE=config['SOURCE'], recovery_script_sha256=a.sha(__file__),
        original_scope_sha256=a.sha(original / 'scope.json'),
        original_seed_file_sha256=a.sha(original / 'seeds_PRIVATE.json'),
        completed_trial_hashes={f"{r['candidate_id']}/{r['index']}": a.sha(
            original / 'prices' / r['candidate_id'] / str(r['index']) / 'result.json') for r in completed},
        reused_trials=len(completed), missing_trials=len(missing), new_samples=0,
        workers=max(1, min(len(missing), len(os.sched_getaffinity(0)))))
    a.save(output / 'recovery.json', provenance)
    print(json.dumps(dict(batch=batch, status='RECOVERING', **provenance)), flush=True)
    try:
        runtime.CONTEXT = runtime.Context(config, output, [r['prefix'] for r in scope['candidates']], signal=False)
        a.require(a.read(output / 'preparation_binding.json') == a.read(original / 'preparation_binding.json'),
                  'ORIGINAL_PREPARATION_BINDING')
        native.PRIVATE = output
        native.trial_seed = lambda row, cohort, index, outer, piece: seed_map[(row['id'], index, outer, piece)]
        sample = native.sample_requests

        def same_events(row, prep, cohort, index):
            requests, seeds, laws = sample(row, prep, cohort, index)
            saved = a.read(original / 'prices' / row['id'] / str(index) / 'events_PRIVATE.json')
            events = [[e.to_dict() for e in req.rte_occurrence.events] for req in requests]
            a.require(saved == dict(seeds=seeds, events=events, conditional=False, cohort=cohort),
                      'EXACT_ORIGINAL_EVENTS_ONLY')
            return requests, seeds, laws

        native.sample_requests = same_events
        compile_new = native.compile_cached

        def compile_saved(wrapper, fingerprint, representative):
            old_cache = original / 'compiled_cache'
            path = old_cache / (fingerprint + '.json')
            if not path.exists():
                return compile_new(wrapper, fingerprint, representative)
            value = a.read(path)
            a.require(value['fingerprint'] == fingerprint and value['compiler'] == a.COMPILER,
                      'ORIGINAL_COMPILER_CACHE_IDENTITY')
            qc = None
            if representative:
                from qiskit import qpy
                import io
                compiled_path = old_cache / (fingerprint + '_compiled.qpy.gz')
                if not compiled_path.exists():
                    return compile_new(wrapper, fingerprint, representative)
                qc, = qpy.load(io.BytesIO(gzip.decompress(compiled_path.read_bytes())))
                a.require(qc.count_ops().get('rz', 0) == value['metrics']['rz_count'], 'ORIGINAL_QPY_PRICE')
            record = dict(original_record_sha256=a.sha(path),
                compiled_qpy_sha256=a.sha(compiled_path) if representative else None)
            record_path = output / 'cache_reuse' / (fingerprint + '_' + str(os.getpid()) + '.json')
            if not record_path.exists():
                a.save(record_path, record)
            else:
                a.require(a.read(record_path) == record, 'CONSISTENT_CACHE_REUSE')
            return value, qc, True

        native.compile_cached = compile_saved
        jobs = [(*job, str(output / 'prices' / job[0]['id'] / str(job[3]))) for job in missing]
        trials = completed + runtime.map_jobs(runtime.price_job, jobs, provenance['workers'])
        summaries = {}
        for row in scope['candidates']:
            selected = sorted((r for r in trials if r['candidate_id'] == row['id']), key=lambda r: r['index'])
            n = scope['n'] if row['method'] == 'PR' else 1
            a.require([r['index'] for r in selected] == list(range(n)), 'EXACT_ORIGINAL_COHORT_N')
            stats = a.paired_G(selected, endpoints[row['id']], scope['cohort'])
            if row['method'] == 'PR':
                observed = {str(k): sum(r['orders'].get(str(k), 0) for r in selected) for k in (0, 2, 4)}
                stats['rarity'] = native.rarity(selected[0]['laws'], runtime.CONTEXT.preps[row['prefix']], row['K'], n, observed)
            summaries[row['id']] = stats
        result = dict(status='PASS', stage='B', binding=config['binding'], source_files=config['source_files'],
            SOURCE=config['SOURCE'], signal_SOURCE=signal['SOURCE'], scope_digest=a.digest(scope),
            environment={n: importlib.metadata.version(n) for n in ('qiskit', 'numpy', 'scipy', 'numba')},
            trials=trials, summaries=summaries, pooled=False, recovery=provenance)
        a.save(output / 'result.json', result)
        a.save(original / 'result.json', result)
        print(json.dumps(dict(batch=batch, status='PASS', trials=len(trials))), flush=True)
    except BaseException as exc:
        a.save(output / 'HOLD.json', dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--batch', choices=('stage_B01', 'stage_B02'))
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    if args.batch:
        resume_batch(args.private.resolve(), args.batch)
        return
    a.require(not (args.private / 'recovery01').exists(), 'RECOVERY_ALREADY_STARTED')
    children = [subprocess.Popen([sys.executable, '-B', '-u', __file__, '--private', str(args.private.resolve()),
        '--batch', name, '--execute']) for name in ('stage_B01', 'stage_B02')]
    codes = [child.wait() for child in children]
    a.require(codes == [0, 0], 'RECOVERY_CHILD_FAILURE_NO_AUTOMATIC_RETRY')
    from report_h8_long_time_execution import collect
    result = collect(args.private.resolve(), a.ROOT / 'artifacts/resource_applicability/track_a_h8_long_time/2026-11-03')
    a.save(args.private / 'collection_status.json', dict(status='COLLECTED_EXPLORATION_ONLY',
        candidates=len(result['rows']), trajectories=len(result['trials']), confirmation='NOT_RUN',
        native_SOURCE=result['native_SOURCE']))


if __name__ == '__main__':
    main()
