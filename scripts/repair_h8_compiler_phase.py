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
PREVIOUS_SOURCE = 'f04ef60c960fd2a68bcf0506f4c505f0bc872b78'
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


def find_raw_cache(roots, fingerprint):
    for root in roots:
        metadata = root / (fingerprint + '.json')
        qpath = root / (fingerprint + '_compiled.qpy.gz')
        if metadata.is_file() and qpath.is_file():
            return metadata, qpath
    return None


def restore_evolution(wrapper, row, native, fingerprint):
    """Invert the existing H/U/H/measure wrapper and verify exact rebuilding."""
    from qiskit import QuantumCircuit
    n = wrapper.num_qubits - 1
    items = wrapper.data
    a.require(wrapper.num_clbits == 1 and len(items) >= 3
              and [items[i].operation.name for i in (0, -2, -1)] == ['h', 'h', 'measure']
              and all(tuple(wrapper.find_bit(q).index for q in items[i].qubits) == (n,)
                      for i in (0, -2, -1))
              and all(not item.clbits for item in items[1:-2]), 'SAVED_COSINE_WRAPPER_LAYOUT')
    circuit = QuantumCircuit(*wrapper.qregs)
    circuit.data = list(items[1:-2])
    circuit.global_phase = float(wrapper.global_phase)
    evolution = native.NativeEvolution(circuit, n, row['T'], row['q'], row['formula'],
        'symmetric_directional', len(circuit.data), fingerprint(circuit))
    rebuilt = native.build_native_hadamard_wrapper(evolution, axis='cosine',
        include_measurement=True, max_instructions=len(circuit.data) + 4)
    a.require(fingerprint(rebuilt) == fingerprint(wrapper), 'EXACT_SAVED_WRAPPER_REBUILD')
    return evolution


def same_compiled_circuit(left, right):
    if (left.num_qubits, left.num_clbits, len(left.data), float(left.global_phase)) != (
            right.num_qubits, right.num_clbits, len(right.data), float(right.global_phase)):
        return False
    def instructions(qc):
        qi, ci = {q: i for i, q in enumerate(qc.qubits)}, {c: i for i, c in enumerate(qc.clbits)}
        for item in qc.data:
            yield (item.operation.name, tuple(float(p).hex() for p in item.operation.params),
                   tuple(qi[q] for q in item.qubits), tuple(ci[c] for c in item.clbits))
    return all(a == b for a, b in zip(instructions(left), instructions(right)))


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
    old_compile, old_sample, old_compose = native.compile_cached, native.sample_requests, native.compose
    old_simulate, old_independent = circuits.simulate_statevector, runtime.independent_trajectory
    certificates, phases, raw_reusable = {}, {}, {}
    busy, count = False, 0
    prior = PRIVATE / 'recovery03' / row['id'] / 'checkpoints'
    previous = PRIVATE / 'phase_repair01' / row['id']
    previous_scope = None
    if (PRIVATE / 'phase_repair01/scope.json').exists():
        previous_scope = a.read(PRIVATE / 'phase_repair01/scope.json')
        a.require(previous_scope['supplement']['SOURCE'] == PREVIOUS_SOURCE
                  and dict(candidate=row, index=index, batch=batch) in previous_scope['jobs'],
                  'PREVIOUS_SAME_ORIGINAL_TRIAL')
        for name, expected_hash in previous_scope['supplement']['source_files'].items():
            data = subprocess.check_output(['git', 'show', PREVIOUS_SOURCE + ':' + name], cwd=a.ROOT)
            a.require(hashlib.sha256(data).hexdigest() == expected_hash, 'PREVIOUS_SUPPLEMENT_SOURCE')
        if previous_scope['supplement']['source_files']:
            a.require(a.read(PRIVATE / 'phase_repair01/context/preparation_binding.json')
                      == a.read(OUTPUT / 'context/preparation_binding.json'), 'PREVIOUS_SAME_PREPARATION_BINDING')
        same_event_metadata(a.read(previous / 'trial/events_PRIVATE.json'),
                            a.read(original / 'prices' / row['id'] / str(index) / 'events_PRIVATE.json'))
        for axis in ('cosine', 'sine'):
            proof_path = previous / 'phase' / (axis + '_certificate_PRIVATE.json')
            record_path = previous / 'phase' / (axis + '.json')
            if proof_path.exists() and record_path.exists():
                proof, record = a.read(proof_path), a.read(record_path)
                a.require(a.digest(proof) == record['certificate_sha256']
                          and proof['compiler'] == a.COMPILER and proof['policy'] == phase.POLICY,
                          'PREVIOUS_LOCAL_OPERATOR_TESTS')
                certificates.update(proof['records'])
    cache_roots = [original / 'compiled_cache']
    cache_roots.extend(sorted((PRIVATE / 'recovery03' / row['id']).glob('attempt_*/compiled_cache')))
    cache_roots.append(previous / 'compiled_cache')
    corrected_reusable = {}
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

    def previous_vector(label):
        metadata, path = previous / 'checkpoints' / (label + '.json'), previous / 'checkpoints' / (label + '.npy')
        if previous_scope is None or not metadata.exists():
            return None
        saved = a.read(metadata)
        a.require(saved['supplement'] == previous_scope['supplement']
                  and a.sha(path) == saved['sha256'], 'PREVIOUS_REPAIR_VECTOR_IDENTITY')
        if label.startswith('compiled_'):
            axis = label.removeprefix('compiled_')
            a.require(corrected_reusable.get(axis, False), 'SAME_PREVIOUS_CORRECTED_CIRCUIT')
        return np.load(path, allow_pickle=False)

    def vector(label, compute, initial=None, quarters=None):
        print(json.dumps(dict(candidate=row['id'], stage=label, state='START')), flush=True)
        start = time.monotonic()
        value = previous_vector(label)
        already_corrected = value is not None
        if value is None:
            value = original_vector(label, initial)
        reused = value is not None
        if reused and not already_corrected and quarters is not None:
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
            reused_original_vector=reused, reused_previous_repair_vector=already_corrected,
            analytic_global_phase_quarters=quarters,
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

    def saved_evolution(candidate, prep, requests, scalar, fingerprint):
        if previous_scope is None:
            return old_compose(candidate, prep, requests, scalar, fingerprint)
        record = a.read(previous / 'phase/cosine.json')
        fp = record['wrapper_fingerprint']
        for root in cache_roots:
            path = root / (fp + '_wrapper.qpy.gz')
            if path.exists():
                wrapper, = qpy.load(io.BytesIO(gzip.decompress(path.read_bytes())))
                a.require(wrapper.num_qubits == prep.num_system_qubits + 1
                          and fingerprint(wrapper) == fp, 'SAME_SAVED_NATIVE_WRAPPER')
                evolution = restore_evolution(wrapper, candidate, runtime.CONTEXT.native, fingerprint)
                a.save(folder / 'native_wrapper_reuse.json', dict(wrapper_fingerprint=fp,
                    saved_wrapper_sha256=a.sha(path), previous_supplement_SOURCE=PREVIOUS_SOURCE,
                    exact_cosine_rebuild=True, original_events_matched=True))
                return evolution
        raise ValueError('SAVED_NATIVE_WRAPPER_REQUIRED')

    def compiled(wrapper, fingerprint, representative):
        a.require(representative and len(phases) < 2, 'TWO_ORIGINAL_REPRESENTATIVE_AXES')
        axis = ('cosine', 'sine')[len(phases)]
        old_cache = original / 'compiled_cache'
        original_metadata = old_cache / (fingerprint + '.json')
        found = find_raw_cache(cache_roots, fingerprint)
        raw_reusable[axis] = found is not None
        if found:
            metadata, qpath = found
            value = a.read(metadata)
            qc, = qpy.load(io.BytesIO(gzip.decompress(qpath.read_bytes())))
            hit = True
        else:
            value, qc, hit = old_compile(wrapper, fingerprint, True)
        if original_metadata.exists():
            a.require(value == a.read(original_metadata), 'SAME_ORIGINAL_COMPILER_PRICE')
        a.require(qc is not None and value['compiler'] == a.COMPILER
                  and value['fingerprint'] == fingerprint
                  and qc.count_ops().get('rz', 0) == value['metrics']['rz_count'], 'FIXED_RAW_COMPILED_PRICE')
        print(json.dumps(dict(candidate=row['id'], stage='local_operator_' + axis, state='START')), flush=True)
        proof = phase.certify(wrapper, certificates, workers=max(1, len(os.sched_getaffinity(0)) // 2))
        corrected, record = phase.correct_phase(wrapper, qc, proof)
        old_phase = previous / 'phase' / (axis + '.json')
        corrected_reusable[axis] = False
        if old_phase.exists():
            old_record = a.read(old_phase)
            old_qpy = previous / 'phase' / (axis + '_corrected.qpy.gz')
            a.require(old_record['wrapper_fingerprint'] == fingerprint
                      and old_record['delta_phase_quarters'] == record['delta_phase_quarters']
                      and a.sha(old_qpy) == old_record['corrected_qpy_sha256'], 'PREVIOUS_CORRECTED_QPY')
            old_corrected, = qpy.load(io.BytesIO(gzip.decompress(old_qpy.read_bytes())))
            a.require(same_compiled_circuit(corrected, old_corrected), 'PREVIOUS_CORRECTED_INSTRUCTIONS_PHASE')
            corrected_reusable[axis] = True
        a.save(folder / 'phase' / (axis + '_certificate_PRIVATE.json'), proof)
        stream = io.BytesIO()
        qpy.dump(corrected, stream)
        path = folder / 'phase' / (axis + '_corrected.qpy.gz')
        with path.open('xb') as output:
            output.write(gzip.compress(stream.getvalue(), compresslevel=1, mtime=0))
        phases[axis] = dict(**record, wrapper_fingerprint=fingerprint, corrected_qpy_sha256=a.sha(path),
            original_raw_qpy_sha256=a.sha(qpath) if found else None,
            rz_count=value['metrics']['rz_count'])
        a.save(folder / 'phase' / (axis + '.json'), phases[axis])
        print(json.dumps(dict(candidate=row['id'], stage='local_operator_' + axis, state='PASS',
                              **record)), flush=True)
        return value, corrected, hit

    native.sample_requests, native.compile_cached, native.compose = sampled, compiled, saved_evolution
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
        native.sample_requests, native.compile_cached, native.compose = old_sample, old_compile, old_compose
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
    PRIVATE, OUTPUT = private, private / 'phase_repair02'
    a.require(len(source) == 40 and all(c in '0123456789abcdef' for c in source), 'SUPPLEMENT_SOURCE_COMMIT')
    hashes = {}
    for path in FILES:
        data = subprocess.check_output(['git', 'show', source + ':' + path], cwd=a.ROOT)
        hashes[path] = hashlib.sha256(data).hexdigest()
        a.require(a.sha(a.ROOT / path) == hashes[path], 'FIXED_SUPPLEMENT_SOURCE:' + path)
    SUPPLEMENT = dict(SOURCE=source, source_files=hashes, previous_supplement_SOURCE=PREVIOUS_SOURCE, phase_policy=phase.POLICY,
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
    for cid, reason in [('7eb0a3d691b94089969b', 'ValueError:UNCERTIFIED_GATE_FAMILY:sdg'),
                        ('3bce8dc9530cddf679db', 'ValueError:SAME_ORIGINAL_RAW_COMPILED_CIRCUIT')]:
        a.require(a.read(private / 'phase_repair01' / cid / 'HOLD.json')['reason'] == reason,
                  'ONLY_IDENTIFIED_TECHNICAL_INTERRUPTION')
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
        target = args.private / 'phase_repair02/HOLD.json'
        if target.parent.exists():
            a.save(target, dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise
