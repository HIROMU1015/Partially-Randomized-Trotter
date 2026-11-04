"""Revalidate the two approved original HOLDs with operator-derived phases.

Keep the base scientific runtime, seeds, events, tolerances and every old HOLD.
Use a separate supplement SOURCE and output directory. No Stage C or new draw.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import time

import h8_long_time_adapter as a
import h8_long_time_runtime as runtime
import h8_compiler_phase_bookkeeping as phase
import resume_h8_long_time_execution as resume
from repair_h8_long_time_recovery import same_event_metadata

FILES = ('scripts/h8_compiler_phase_bookkeeping.py', 'scripts/repair_h8_compiler_phase.py',
         'scripts/resume_h8_long_time_execution.py', 'scripts/repair_h8_long_time_recovery.py',
         'scripts/report_h8_long_time_execution.py')
TARGETS = {'7eb0a3d691b94089969b': 'stage_B01', '3bce8dc9530cddf679db': 'stage_B02'}
PRIVATE = OUTPUT = SUPPLEMENT = None
CONFIGS, SCOPES = {}, {}


def completed_trials(private):
    paths = []
    for name in ('stage_B01', 'stage_B02'):
        for base in ('runs', 'recovery01', 'recovery02'):
            paths.extend((private / base / name).glob('prices/*/*/result.json'))
    paths.extend((private / 'recovery03').glob('*/result.json'))
    completed = {}
    for path in paths:
        value = a.read(path)
        key = value['candidate_id'], value['index']
        a.require(key not in completed and value['status'] == 'PASS', 'UNIQUE_COMPLETED_ORIGINALS')
        completed[key] = value
    return completed


def finish_trial(job):
    import numpy as np
    from qiskit import qpy
    import h46_long_time_native as native
    import trotterlib.df_trotter.circuit as circuits
    row, endpoint, cohort, index, batch = job
    folder = OUTPUT / row['id']
    folder.mkdir()
    original = PRIVATE / 'runs' / batch
    config = CONFIGS[batch]
    native.PRIVATE = folder
    old_compile, old_sample = native.compile_cached, native.sample_requests
    old_simulate, old_independent = circuits.simulate_statevector, runtime.independent_trajectory
    certificates, phases, raw_reusable = {}, {}, {}
    busy, count = False, 0
    prior = PRIVATE / 'recovery03' / row['id'] / 'checkpoints'
    identity = None
    if prior.exists():
        scope = a.read(PRIVATE / 'recovery03/scope.json')
        keys = ('runtime_SOURCE', 'source_files', 'binding', 'driver_sha256',
                'original_scope_sha256', 'original_seeds_sha256')
        identity = {k: scope[k] for k in keys}
        a.require(identity['runtime_SOURCE'] == config['SOURCE']
                  and identity['source_files'] == config['source_files']
                  and identity['binding'] == config['binding']
                  and identity['driver_sha256'] == a.sha(PRIVATE / 'resume_two_representatives.py')
                  and identity['original_scope_sha256'] == a.sha(original / 'scope.json')
                  and identity['original_seeds_sha256'] == a.sha(original / 'seeds_PRIVATE.json'),
                  'UNCHANGED_PREVIOUS_CHECKPOINT_SCOPE')

    def original_vector(label, initial=None):
        metadata, path = prior / (label + '.json'), prior / (label + '.npy')
        if not metadata.exists():
            return None
        parts = [identity, row['id'], index, label]
        if initial is not None:
            parts.append(hashlib.sha256(initial.tobytes()).hexdigest())
        saved = a.read(metadata)
        a.require(identity is not None and saved['key'] == a.digest(parts)
                  and a.sha(path) == saved['sha256'], 'UNCHANGED_PREVIOUS_VECTOR')
        return np.load(path, allow_pickle=False)

    def vector(label, compute, initial=None, quarters=None):
        print(json.dumps(dict(candidate=row['id'], stage=label, state='START')), flush=True)
        start = time.monotonic()
        value = original_vector(label, initial)
        reused = value is not None
        if reused and quarters is not None:
            # Exact known global-phase action on a saved raw compiled vector.
            # The factor comes from gate operators, never a state overlap.
            value = (1j ** quarters) * value
        if value is None:
            value = compute()
        directory = folder / 'checkpoints'
        directory.mkdir(exist_ok=True)
        path = directory / (label + '.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        a.save(directory / (label + '.json'), dict(sha256=a.sha(path), supplement=SUPPLEMENT,
            reused_original_vector=reused, analytic_global_phase_quarters=quarters,
            elapsed_seconds=time.monotonic() - start))
        print(json.dumps(dict(candidate=row['id'], stage=label, state='SAVED', reused=reused,
                              elapsed_seconds=time.monotonic() - start)), flush=True)
        return value

    def expected(ctx, candidate, requests):
        nonlocal busy
        busy = True
        try:
            return vector('algebraic_expected', lambda: old_independent(ctx, candidate, requests))
        finally:
            busy = False

    def simulate(circuit, initial):
        nonlocal count
        if busy:
            return old_simulate(circuit, initial)
        labels = ('low_control', 'high_control', 'hadamard_cosine', 'compiled_cosine',
                  'hadamard_sine', 'compiled_sine')
        a.require(count < len(labels), 'ORIGINAL_CHECK_SEQUENCE')
        label = labels[count]
        count += 1
        quarters = None
        if label.startswith('compiled_'):
            axis = label.removeprefix('compiled_')
            quarters = phases[axis]['delta_phase_quarters']
            if (prior / (label + '.json')).exists():
                a.require(raw_reusable[axis], 'SAME_ORIGINAL_RAW_COMPILED_CIRCUIT')
        return vector(label, lambda: old_simulate(circuit, initial), initial, quarters)

    def sampled(candidate, prep, cohort, index):
        requests, seeds, laws = old_sample(candidate, prep, cohort, index)
        actual = dict(seeds=seeds, events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests],
                      conditional=False, cohort=cohort)
        same_event_metadata(a.read(original / 'prices' / row['id'] / str(index) / 'events_PRIVATE.json'), actual)
        return requests, seeds, laws

    def compiled(wrapper, fingerprint, representative):
        a.require(representative and len(phases) < 2, 'TWO_ORIGINAL_REPRESENTATIVE_AXES')
        axis = ('cosine', 'sine')[len(phases)]
        old_cache = original / 'compiled_cache'
        metadata, qpath = old_cache / (fingerprint + '.json'), old_cache / (fingerprint + '_compiled.qpy.gz')
        raw_reusable[axis] = metadata.exists() and qpath.exists()
        if raw_reusable[axis]:
            value = a.read(metadata)
            qc, = qpy.load(io.BytesIO(gzip.decompress(qpath.read_bytes())))
            hit = True
        else:
            value, qc, hit = old_compile(wrapper, fingerprint, True)
            if metadata.exists():
                a.require(value == a.read(metadata), 'SAME_ORIGINAL_COMPILER_PRICE')
        a.require(qc is not None and value['compiler'] == a.COMPILER
                  and value['fingerprint'] == fingerprint
                  and qc.count_ops().get('rz', 0) == value['metrics']['rz_count'], 'FIXED_RAW_COMPILED_PRICE')
        print(json.dumps(dict(candidate=row['id'], stage='local_operator_' + axis, state='START')), flush=True)
        proof = phase.certify(wrapper, certificates, workers=max(1, len(os.sched_getaffinity(0)) // 2))
        corrected, record = phase.correct_phase(wrapper, qc, proof)
        a.save(folder / 'phase' / (axis + '_certificate_PRIVATE.json'), proof)
        stream = io.BytesIO()
        qpy.dump(corrected, stream)
        path = folder / 'phase' / (axis + '_corrected.qpy.gz')
        with path.open('xb') as output:
            output.write(gzip.compress(stream.getvalue(), compresslevel=1, mtime=0))
        phases[axis] = dict(**record, wrapper_fingerprint=fingerprint, corrected_qpy_sha256=a.sha(path),
            original_raw_qpy_sha256=a.sha(qpath) if raw_reusable[axis] else None,
            rz_count=value['metrics']['rz_count'])
        a.save(folder / 'phase' / (axis + '.json'), phases[axis])
        print(json.dumps(dict(candidate=row['id'], stage='local_operator_' + axis, state='PASS',
                              **record)), flush=True)
        return value, corrected, hit

    native.sample_requests, native.compile_cached = sampled, compiled
    circuits.simulate_statevector, runtime.independent_trajectory = simulate, expected
    try:
        value = runtime.price_job((row, endpoint, cohort, index, str(folder / 'trial')))
        a.require(count == 6 and len(phases) == 2 and len(value['checks']) == 7, 'ALL_SEVEN_ORIGINAL_CHECKS')
        value = dict(value, compiler_phase_supplement=dict(**SUPPLEMENT, axes=phases,
                     previous_HOLD_preserved=True, new_samples=0))
        a.save(folder / 'result.json', value)
        return value
    except BaseException as exc:
        a.save(folder / 'HOLD.json', dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise
    finally:
        native.sample_requests, native.compile_cached = old_sample, old_compile
        circuits.simulate_statevector, runtime.independent_trajectory = old_simulate, old_independent


def assemble(completed, repaired, signal, workers):
    import h46_long_time_native as native
    all_trials = list(completed.values()) + repaired
    endpoints = {r['endpoint']['candidate']['id']: r['endpoint'] for r in signal['results']}
    a.require(len(all_trials) == len({(v['candidate_id'], v['index']) for v in all_trials}) == 51,
              'EXACT_COMPLETE_ORIGINAL_POOL')
    for name, scope in SCOPES.items():
        ids = {r['id'] for r in scope['candidates']}
        trials = [v for v in all_trials if v['candidate_id'] in ids]
        summaries = {}
        for row in scope['candidates']:
            selected = sorted((v for v in trials if v['candidate_id'] == row['id']), key=lambda v: v['index'])
            n = scope['n'] if row['method'] == 'PR' else 1
            a.require([v['index'] for v in selected] == list(range(n)), 'ORIGINAL_EXACT_COHORT')
            stats = a.paired_G(selected, endpoints[row['id']], scope['cohort'])
            if row['method'] == 'PR':
                observed = {str(k): sum(v['orders'].get(str(k), 0) for v in selected) for k in (0, 2, 4)}
                stats['rarity'] = native.rarity(selected[0]['laws'], runtime.CONTEXT.preps[row['prefix']], row['K'], n, observed)
            summaries[row['id']] = stats
        config = CONFIGS[name]
        a.save(OUTPUT / 'batches' / name / 'result.json', dict(status='PASS', stage='B',
            binding=config['binding'], source_files=config['source_files'], SOURCE=config['SOURCE'],
            signal_SOURCE=signal['SOURCE'], scope_digest=a.digest(scope), pooled=False,
            environment={k: importlib.metadata.version(k) for k in ('qiskit', 'numpy', 'scipy', 'numba')},
            trials=trials, summaries=summaries, compiler_phase_supplement=SUPPLEMENT,
            recovery=dict(runtime_SOURCE=config['SOURCE'], recovery_script_sha256=a.sha(__file__),
                reused_trials=sum(key[0] in ids for key in completed), missing_trials=1,
                new_samples=0, workers=workers)))
    from report_h8_long_time_execution import collect
    result = collect(PRIVATE, a.ROOT / 'artifacts/resource_applicability/track_a_h8_long_time/2026-11-04',
                     batch_root=OUTPUT / 'batches')
    a.save(OUTPUT / 'collection_status.json', dict(status=result['status'], candidates=len(result['rows']),
        trajectories=len(result['trials']), confirmation='NOT_RUN', compiler_phase_supplement=SUPPLEMENT))


def run(private, source):
    global PRIVATE, OUTPUT, SUPPLEMENT
    PRIVATE, OUTPUT = private, private / 'phase_repair01'
    a.require(len(source) == 40 and all(c in '0123456789abcdef' for c in source), 'SUPPLEMENT_SOURCE_COMMIT')
    hashes = {}
    for path in FILES:
        data = subprocess.check_output(['git', 'show', source + ':' + path], cwd=a.ROOT)
        hashes[path] = hashlib.sha256(data).hexdigest()
        a.require(a.sha(a.ROOT / path) == hashes[path], 'FIXED_SUPPLEMENT_SOURCE:' + path)
    SUPPLEMENT = dict(SOURCE=source, source_files=hashes, phase_policy=phase.POLICY,
                      statevector_phase_alignment=False, original_HOLDs_preserved=True)
    seeds, jobs = {}, []
    signal = a.read(private / 'stage_A_combined.json')
    endpoints = {r['endpoint']['candidate']['id']: r['endpoint'] for r in signal['results']}
    for name, filename in (('stage_B01', 'config_B.json'), ('stage_B02', 'config_B02.json')):
        config = a.read(private / filename)
        scope = a.read(private / 'runs' / name / 'scope.json')
        a.preflight(config)
        runtime.validate_scope('B', scope, config)
        a.require(a.sha(scope['stage_A']) == scope['stage_A_sha256']
                  and signal['binding'] == config['binding']
                  and signal['source_files'] == config['signal_source_files'], 'FIXED_SIGNAL_INPUT_SOURCE')
        a.check_reference(signal['reference'], config['binding'])
        restored = resume.restored_seeds(config, scope, private / 'runs' / name)
        a.require(not (seeds.keys() & restored.keys()), 'DISJOINT_ORIGINAL_SEEDS')
        seeds.update(restored)
        CONFIGS[name], SCOPES[name] = config, scope
        for row in scope['candidates']:
            if TARGETS.get(row['id']) == name:
                a.require(row['method'] == 'PR' and row['prefix'] == 8 and row['R_outer'] == 32
                          and row['K'] == 4 and endpoints[row['id']]['status'] == 'ELIGIBLE', 'APPROVED_ORIGINAL_REPRESENTATIVES')
                jobs.append((row, endpoints[row['id']], 'EXPLORATION', 0, name))
    completed = completed_trials(private)
    a.require(len(completed) == 49 and len(jobs) == 2
              and not ({(r, 0) for r in TARGETS} & completed.keys()), 'ONLY_TWO_ORIGINAL_HOLDS')
    for path in (private / 'recovery02/stage_B01/failure_7eb0a3d691b94089969b_0.json',
                 private / 'recovery03/3bce8dc9530cddf679db/HOLD.json'):
        a.require(a.read(path)['reason'] == 'ValueError:COMPILED_VECTOR_cosine', 'PRESERVED_ORIGINAL_HOLD')
    OUTPUT.mkdir()
    workers = min(2, len(os.sched_getaffinity(0)))
    a.save(OUTPUT / 'scope.json', dict(supplement=SUPPLEMENT, jobs=[dict(candidate=j[0], index=j[3], batch=j[4]) for j in jobs],
        new_samples=0, completed_reused=49, workers=workers, stage_C='NOT_RUN'))
    (OUTPUT / 'context').mkdir()
    runtime.CONTEXT = runtime.Context(CONFIGS['stage_B01'], OUTPUT / 'context', [7, 8], signal=False)
    import h46_long_time_native as native
    native.trial_seed = lambda row, cohort, index, outer, piece: seeds[(row['id'], index, outer, piece)]
    results = runtime.map_jobs(finish_trial, jobs, workers)
    a.save(OUTPUT / 'result.json', dict(status='PASS', trials=results, supplement=SUPPLEMENT, new_samples=0))
    assemble(completed, results, signal, workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--supplement-source', required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    try:
        run(args.private.resolve(), args.supplement_source)
    except BaseException as exc:
        target = args.private / 'phase_repair01/HOLD.json'
        if target.parent.exists():
            a.save(target, dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise
