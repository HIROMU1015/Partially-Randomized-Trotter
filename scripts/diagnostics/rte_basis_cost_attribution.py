#!/usr/bin/env python3
"""Small saved-H4 diagnostic; additive attribution is not full-cost lineage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import qiskit
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector

from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.research_direction_sequence_policy import _derived_seed, _request_for_seed
from trotterlib.rte import CompilerSettings
from trotterlib.rte_compiled_cost import transpile_and_measure_cost
from trotterlib.rte_connected_cluster_cost_validation import load_connected_cluster_hamiltonian_snapshot

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('rz_count', 'cx_count', 'circuit_size')
ROLES = ('basis', 'central', 'other')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if len(data) > 512 * 1024:
        raise RuntimeError('REVIEW_FILE_CAP')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def gate_ir(circuit):
    return {
        'qubits': circuit.num_qubits, 'global_phase': float(circuit.global_phase),
        'operations': [
            {'name': item.operation.name,
             'qubits': [circuit.find_bit(q).index for q in item.qubits],
             'parameters': [float(p) for p in item.operation.params]}
            for item in circuit.data
        ],
    }


class TaggedBuilder(QiskitDFRTEEventCircuitBuilder):
    """Observe production append sites without changing the emitted circuit."""

    def _build(self, **kwargs):
        self.spans = []
        result = super()._build(**kwargs)
        cursor = 0
        for span in self.spans:
            if span['start'] != cursor:
                raise RuntimeError('UNEXPECTED_UNTAGGED_INTERIOR')
            cursor = span['end']
        if cursor < len(result.circuit.data):
            self.spans.append({'role': 'other', 'start': cursor, 'end': len(result.circuit.data)})
        return result

    def _append_basis(self, circuit, definition, *, inverse):
        start = len(circuit.data)
        super()._append_basis(circuit, definition, inverse=inverse)
        if len(circuit.data) != start:
            self.spans.append({'role': 'basis', 'start': start, 'end': len(circuit.data),
                               'basis_hash': definition.basis_hash, 'inverse': inverse})

    def _append_central_application(self, circuit, event, application, **kwargs):
        start = len(circuit.data)
        phase = super()._append_central_application(circuit, event, application, **kwargs)
        if len(circuit.data) != start:
            self.spans.append({'role': 'central', 'start': start, 'end': len(circuit.data),
                               'application_role': application.role})
        return phase


def partition_circuit(circuit, spans):
    """Group contiguous same-role instructions; preserve chronological boundaries."""
    groups = []
    cursor = 0
    for s in spans:
        if s['role'] not in ROLES or s['start'] != cursor or s['end'] <= cursor:
            raise ValueError('NONEXHAUSTIVE_OR_OVERLAPPING_TAGS')
        cursor = s['end']
        if groups and groups[-1]['role'] == s['role']:
            groups[-1]['end'] = cursor
            groups[-1]['sources'].append(s)
        else:
            groups.append({'role': s['role'], 'start': s['start'], 'end': cursor, 'sources': [s]})
    if cursor != len(circuit.data):
        raise ValueError('NONEXHAUSTIVE_OR_OVERLAPPING_TAGS')
    reconstructed = QuantumCircuit(circuit.num_qubits)
    reconstructed.global_phase = circuit.global_phase
    parts = []
    for group in groups:
        part = QuantumCircuit(circuit.num_qubits, name='attribution_component')
        for item in circuit.data[group['start']:group['end']]:
            wires = [circuit.find_bit(q).index for q in item.qubits]
            part.append(item.operation, wires)
            reconstructed.append(item.operation, wires)
        parts.append((group, part))
    if reconstructed != circuit:
        raise RuntimeError('TAGGING_CHANGED_CIRCUIT')
    return parts


def qpy_bytes(circuit):
    buf = io.BytesIO()
    qpy.dump(circuit, buf)
    return buf.getvalue()


def guarded_process():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    def check(_signum, _frame):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 768 * 1024**2:
            raise MemoryError('PROCESS_RSS_CAP')
    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, .1, .1)


def worker(job):
    key, payload, full, compiler_dict = job
    start, cpu = time.monotonic(), time.process_time()
    circuit = qpy.load(io.BytesIO(payload))[0]
    compiler = CompilerSettings(**compiler_dict)
    cost = transpile_and_measure_cost(circuit, compiler)
    native = cost.transpiled_circuit
    record = {
        'key': key, 'input_qpy_sha256': digest(payload),
        'raw': gate_ir(circuit), 'native': gate_ir(native),
        'metrics': {m: int(getattr(cost, m)) for m in METRICS},
        'depth': int(cost.total_depth),
        'native_gate_counts': dict(cost.posttranspile_gate_counts),
        'compiler_settings_hash': cost.compiler_settings_hash,
        'actual_circuit_fingerprint': cost.actual_circuit_fingerprint,
        'wall_seconds': time.monotonic() - start, 'cpu_seconds': time.process_time() - cpu,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    return key, record, native if full else None


def summarize(rows):
    categories = {r: {m: sum(row['additive'][r][m] for row in rows) for m in METRICS} for r in ROLES}
    additive = {m: sum(categories[r][m] for r in ROLES) for m in METRICS}
    full = {m: sum(row['full'][m] for row in rows) for m in METRICS}
    return {
        'rows': len(rows), 'additive_categories': categories, 'additive_total': additive,
        'whole_compiled_total': full,
        'basis_fraction_exact': {m: {'numerator': categories['basis'][m], 'denominator': additive[m]}
                                 for m in METRICS},
        'basis_percent_additive': {m: 100 * categories['basis'][m] / additive[m] if additive[m] else None
                                  for m in METRICS},
        'additive_minus_whole': {m: additive[m] - full[m] for m in METRICS},
        'gap_percent_of_whole': {m: 100 * (additive[m] - full[m]) / full[m] if full[m] else None
                                for m in METRICS},
        'exact_optimized_cost_lineage_claimed': False,
    }


def run(out):
    start, cpu = time.monotonic(), time.process_time()
    guarded_process()
    plan_path = ROOT / 'artifacts/rte_basis_cost_attribution/2026-10-13/run_plan.json'
    plan = json.loads(plan_path.read_text())
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    dump(out / 'run_start.json', {'source_commit': source, 'plan_sha256': digest(plan_path.read_bytes()),
                                'user_execution_authorized': True, 'old_markers_reused': False})
    old_path = ROOT / plan['saved_wp06b']['path']
    old_bytes = old_path.read_bytes()
    if digest(old_bytes) != plan['saved_wp06b']['sha256']:
        raise RuntimeError('SAVED_RECORD_IDENTITY_MISMATCH')
    old = json.loads(old_bytes)
    snap_path = ROOT / old['source_evidence']['snapshot']['path']
    if digest(snap_path.read_bytes()) != old['source_evidence']['snapshot']['sha256']:
        raise RuntimeError('SAVED_SNAPSHOT_IDENTITY_MISMATCH')
    if qiskit.__version__ != old['configuration']['compiler']['qiskit_version']:
        raise RuntimeError('COMPILER_VERSION_MISMATCH')
    h = load_connected_cluster_hamiltonian_snapshot(snap_path)
    prep = prepare_df_partial_s2(h, split_df_hamiltonian_by_ld(h, 3), identity_policy='extract_identity_phase')
    actual = {k: getattr(prep, k) for k in ('hamiltonian_hash', 'partition_hash', 'preparation_hash')}
    actual['tail_hash'] = prep.tail_extraction.tail_hash
    if any(v != old['physical_instance'][k] for k, v in actual.items()):
        raise RuntimeError('PREPARATION_IDENTITY_MISMATCH')
    selected, replay = [], []
    for length in (3, 6):
        for index in range(12):
            seed = _derived_seed(old['configuration']['seeds']['holdout'], 'holdout', length, index)
            request, event_hash = _request_for_seed(prep, delta_time=.02, sequence_length=length,
                                                  finite_taylor_order=2, seed=seed)
            replay.append({'length': length, 'index': index, 'seed': seed, 'event_digest': event_hash})
            if index < 4:
                selected.append((length, index, seed, request, event_hash))
    stream_hash = digest(''.join(r['event_digest'] for r in replay).encode())
    if stream_hash != old['holdout_results']['event_stream_digest']:
        raise RuntimeError('REPLAY_STREAM_IDENTITY_MISMATCH')
    dump(out / 'input_identity.json', {'snapshot': old['source_evidence']['snapshot'],
        'physical_instance': actual, 'original_stream_digest': stream_hash, 'stream_matches': True,
        'old_seed_replay_count': len(replay), 'new_random_seeds': 0, 'old_stream': replay})
    cp = dict(basis_gates=('rz', 'sx', 'x', 'cx'), backend_name=None, coupling_map=None,
              optimization_level=1, layout_method=None, routing_method=None,
              transpiler_seed=17, qiskit_version=qiskit.__version__)
    jobs, contexts = {}, []
    def register(circuit, *, full=False):
        payload = qpy_bytes(circuit)
        key = digest(payload)
        if key not in jobs:
            jobs[key] = (key, payload, full, cp)
        elif full and not jobs[key][2]:
            jobs[key] = (key, payload, True, cp)
        return key
    for length, index, seed, request, event_hash in selected:
        ordinary = QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
        reference = ordinary.build_sequence(request)
        tagged = TaggedBuilder(basis_registry=prep.rte_preparation.basis_registry)
        built = tagged.build_sequence(request)
        if built.circuit != reference.circuit:
            raise RuntimeError('OBSERVER_CHANGED_BASELINE')
        circuit = built.circuit
        if len(circuit.data) > plan['limits']['instructions_per_circuit']:
            raise RuntimeError('INSTRUCTION_CAP')
        parts = partition_circuit(circuit, tagged.spans)
        full_key = register(circuit, full=True)
        descriptors = []
        for group, part in parts:
            descriptors.append(dict(group, key=register(part)))
        row_id = f'length{length}_index{index}'
        input_record = {
            'id': row_id, 'length': length, 'index': index, 'seed': seed,
            'event_digest': event_hash, 'events': [e.to_dict() for e in request.events],
            'raw_circuit': gate_ir(circuit), 'segments': descriptors, 'full_key': full_key,
            'cancelled_basis_change_pairs': built.cancelled_basis_change_pairs,
            'emitted_basis_change_count': built.emitted_basis_change_count,
            'basis_switch_count': built.basis_switch_count,
            'relative_ancilla_phase': built.relative_ancilla_phase,
        }
        dump(out / 'inputs' / f'{row_id}.json', input_record)
        contexts.append((input_record, circuit))
    if len(jobs) > plan['limits']['maximum_native_compile_calls']:
        raise RuntimeError('PLANNED_COMPILE_CAP')
    dump(out / 'compile_inventory.json', {'unique_jobs': len(jobs), 'compiler': cp,
         'keys': list(jobs), 'workers': 2, 'threads_per_worker': 1,
         'cache_contract': 'Exact same-run QPY bytes only; no reuse of old measured costs'})
    print(f'INPUT_IDENTITY_PASS; selected=8; planned_unique_compiles={len(jobs)}', flush=True)
    records, natives = {}, {}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=guarded_process) as pool:
        futures = {pool.submit(worker, job): key for key, job in jobs.items()}
        for future in as_completed(futures):
            key, record, native = future.result()
            dump(out / 'compiled' / f'{key}.json', record)
            records[key] = record
            if native is not None:
                natives[key] = native
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > plan['limits']['output_bytes']:
                raise RuntimeError('OUTPUT_CAP')
    rows = []
    for info, raw in contexts:
        checks = []
        for label in ('+00000000', '+10101010', '100000000'):
            state = Statevector.from_label(label)
            checks.append(float(np.max(np.abs(state.evolve(raw).data - state.evolve(natives[info['full_key']]).data))))
        if max(checks) > 1e-10:
            raise RuntimeError('STRICT_COMPILED_PHASE_VALIDATION_FAILED')
        additive = {r: {m: 0 for m in METRICS} for r in ROLES}
        for segment in info['segments']:
            for m in METRICS:
                additive[segment['role']][m] += records[segment['key']]['metrics'][m]
        row = {k: info[k] for k in ('id', 'length', 'index', 'seed', 'event_digest',
               'cancelled_basis_change_pairs', 'emitted_basis_change_count', 'basis_switch_count')}
        row.update(additive=additive, full=records[info['full_key']]['metrics'],
                   strict_statevector_checks=checks, tagged_raw_equals_production=True)
        row['summary'] = summarize([row])
        rows.append(row)
    dump(out / 'rows.json', rows)
    summary = {
        'status': 'LOCAL_ADDITIVE_BASIS_COST_ATTRIBUTION_COMPLETE', 'source_commit': source,
        'configuration': old['configuration'], 'basis_policy': 'registered_full_basis_with_existing_reuse',
        'selection': 'first four old WP06b holdout seeds at each length 3 and 6; fixed before measurement',
        'pooled_unweighted_selected_rows': summarize(rows),
        'by_length': {str(n): summarize([r for r in rows if r['length'] == n]) for n in (3, 6)},
        'native_compile_calls': len(records), 'unique_planned_compile_calls': len(jobs),
        'wall_seconds': time.monotonic() - start,
        'parent_cpu_seconds': time.process_time() - cpu,
        'worker_compile_cpu_seconds': sum(r['cpu_seconds'] for r in records.values()),
        'parent_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'maximum_worker_peak_rss_bytes': max(r['peak_rss_bytes'] for r in records.values()),
        'new_random_seeds': 0, 'saved_stream_events_replayed': 24, 'T_synthesis': 0,
        'molecule_generation': 0, 'GPU': 0, 'solver': 0, 'shots': 0,
        'limitations': ['Additive separately compiled segments, not exact optimized gate lineage.',
                       'Selected eight short saved columns, not the unconditional RTE population mean.',
                       'No T-count or shot-weighted full-PR total resource claim.',
                       'Default full-basis policy; no support-policy comparison or new algorithm search.'],
        'mandatory_STOP': True,
    }
    dump(out / 'summary.json', summary)
    with (out / 'resource_rows.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'length', 'metric', 'basis', 'central', 'other', 'additive_total',
                         'whole_compiled', 'basis_percent_additive', 'gap_percent_of_whole'])
        for r in rows:
            for m in METRICS:
                s = r['summary']
                writer.writerow([r['id'], r['length'], m, *[r['additive'][role][m] for role in ROLES],
                                 s['additive_total'][m], r['full'][m], s['basis_percent_additive'][m],
                                 s['gap_percent_of_whole'][m]])
    print(json.dumps({'status': summary['status'], 'compiles': len(records),
                     'pooled': summary['pooled_unweighted_selected_rows'],
                     'wall_seconds': summary['wall_seconds']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.output)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        dump(args.output / 'failure.json', {'status': 'TECHNICAL_INCONCLUSIVE',
             'error': repr(exc), 'traceback': traceback.format_exc(), 'mandatory_STOP': True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
