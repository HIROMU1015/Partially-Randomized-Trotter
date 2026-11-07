"""Fingerprint-only synthetic benchmark: no molecular data, compiler or solver.

Fork a fresh child for each observation after imports but before circuit building.
Run observations sequentially to avoid competition from our own workers. Alternate
legacy/stream order across repetitions; RSS is each child's cumulative high-water,
not a sampled stage peak. Python peaks are per-stage incremental traced allocations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, XXPlusYYGate

from trotterlib.rte_compiled_cost import _circuit_payload, canonical_qiskit_circuit_fingerprint
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def synthetic_circuit(kind, repeats):
    q = QuantumCircuit(3, 1)
    q.global_phase = .123
    for i in range(repeats):
        angle = (i % 17 - 8) / 37.
        if kind == 'flat':
            q.rz(angle, 0); q.sx(1); q.cx(0, 1); q.x(2)
        else:
            q.crz(angle, 2, 0)
            q.append(RZZGate(-angle).control(), [2, 0, 1])
            q.append(XXPlusYYGate(angle, .13).control(), [2, 1, 0])
            q.p(-angle, 2)
    q.h(2); q.measure(2, 0)
    if kind == 'one_large_definition':
        # A counterexample to a whole-circuit constant-memory claim: only one
        # top-level instruction, but its expanded definition is still materialized.
        d = q.remove_final_measurements(inplace=False)
        # Qiskit's default circuit name includes PID in fork children. A gate's
        # name is part of the canonical input, so bind it explicitly.
        d.name = 'synthetic_fingerprint_body'
        outer = QuantumCircuit(3, 1)
        outer.append(d.to_gate(), range(3)); outer.measure(2, 0)
        return outer
    return q


def rss():
    current = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('VmRSS:'))
    return {'current_KiB': int(current), 'process_peak_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def profile(call):
    gc.collect()
    before = rss()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        value = call()
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, {'seconds_with_tracemalloc': elapsed, 'Python_traced_peak_bytes': peak,
                   'Python_traced_retained_bytes': current, 'RSS_before': before, 'RSS_after': rss(),
                   'stage_peak_RSS_KiB': None}


def worker(kind, repeats, mode, connection):
    try:
        circuit, build = profile(lambda: synthetic_circuit(kind, repeats))
        result = dict(kind=kind, repeats=repeats, mode=mode, instructions=len(circuit.data),
                      qubits=circuit.num_qubits, build=build, load_average=os.getloadavg())
        if mode == 'phases':
            payload, a = profile(lambda: _circuit_payload(circuit))
            encoded, b = profile(lambda: json.dumps(payload, sort_keys=True, separators=(',', ':')).encode())
            digest, c = profile(lambda: hashlib.sha256(encoded).hexdigest())
            result.update(phases={'recursive_payload': a, 'JSON_UTF8': b, 'SHA256': c},
                          fingerprint=digest, encoded_bytes=len(encoded))
        else:
            function = (canonical_qiskit_circuit_fingerprint if mode == 'legacy'
                        else streamed_canonical_qiskit_circuit_fingerprint)
            digest, measured = profile(lambda: function(circuit))
            # A separate, warmed observation without tracing. It is not the cold
            # run above and must not be combined with its allocation measurements.
            start = time.perf_counter(); warm_digest = function(circuit)
            warm_seconds = time.perf_counter() - start
            if warm_digest != digest:
                raise AssertionError('Hash changed within one unchanged circuit')
            result.update(fingerprint=digest, full=measured, warm_seconds_without_tracemalloc=warm_seconds)
        connection.send({'status': 'PASS', **result})
    except BaseException as exc:
        connection.send({'status': 'FAIL', 'error': repr(exc), 'kind': kind, 'repeats': repeats, 'mode': mode})
    finally:
        connection.close()


def run(output, *, limitation_only=False):
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    paths = ['src/trotterlib/rte_compiled_cost.py', 'src/trotterlib/streamed_circuit_fingerprint.py',
             'scripts/benchmark_streamed_circuit_fingerprint.py', 'tests/test_streamed_circuit_fingerprint.py']
    hashes = {}
    for path in paths:
        raw = (ROOT/path).read_bytes()
        if raw != subprocess.check_output(['git', 'show', source+':'+path], cwd=ROOT):
            raise ValueError('Benchmark SOURCE must be committed: '+path)
        hashes[path] = hashlib.sha256(raw).hexdigest()
    rows = []
    result = dict(schema_version='streamed_fingerprint_artificial_v1', status='RUNNING',
        scope='artificial circuits only; no H4 build/compile or scientific reanalysis',
        source_commit=source, source_sha256=hashes, runtime={'host': platform.node(), 'python': sys.version,
        'qiskit': qiskit.__version__, 'available_cpus': len(os.sched_getaffinity(0)),
        'MemAvailable': next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))},
        protocol='fresh fork child after imports; independent build; sequential observations; alternating method order; three repetitions',
        notes=['No reuse of H4 inputs or circuits', 'Per-phase traces exclude preexisting live objects',
               'Full cold fingerprint includes Qiskit lazy definition allocation',
               'RSS is cumulative child high-water; stage RSS peaks unmeasured',
               'Warm untraced durations are separate observations, not cold timings'],
        rows=rows, summary=[], real_H4_calls=0, real_H4_compiles=0, m17_feasibility=None,
        limitation_only=limitation_only)
    ctx = mp.get_context('fork')
    cases = [(k, n) for k in ('flat', 'controlled') for n in (64, 256, 1024)]
    if limitation_only:
        cases = []
    cases.append(('one_large_definition', 256))
    for kind, n in cases:
        case_rows = []
        for repetition in range(3):
            modes = ('legacy', 'stream') if repetition % 2 == 0 else ('stream', 'legacy')
            if repetition == 0:
                modes += ('phases',)
            for mode in modes:
                receive, send = ctx.Pipe(duplex=False)
                process = ctx.Process(target=worker, args=(kind, n, mode, send))
                process.start(); send.close()
                try:
                    row = receive.recv()
                finally:
                    receive.close(); process.join()
                if process.exitcode != 0 or row['status'] != 'PASS':
                    raise RuntimeError(str(row))
                row['repetition'] = repetition
                rows.append(row); case_rows.append(row)
                (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                print(kind, n, mode, repetition, 'done', flush=True)
        if len({row['fingerprint'] for row in case_rows}) != 1:
            raise AssertionError('Legacy/stream hash disagreement')
        summary = dict(kind=kind, repeats=n, instructions=case_rows[0]['instructions'],
                       fingerprint=case_rows[0]['fingerprint'], all_hashes_equal=True)
        for mode in ('legacy', 'stream'):
            selected = [row for row in case_rows if row['mode'] == mode]
            summary[mode] = {key: {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
                for key, values in {
                    'Python_traced_peak_bytes': [r['full']['Python_traced_peak_bytes'] for r in selected],
                    'cold_seconds_with_tracemalloc': [r['full']['seconds_with_tracemalloc'] for r in selected],
                    'warm_seconds_without_tracemalloc': [r['warm_seconds_without_tracemalloc'] for r in selected],
                    'process_peak_RSS_KiB': [r['full']['RSS_after']['process_peak_KiB'] for r in selected],
                }.items()}
        result['summary'].append(summary)
    result['status'] = 'PASS_IDENTICAL_SHA_ARTIFICIAL_ONLY'
    for path, expected in hashes.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Benchmark SOURCE changed: '+path)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limitation-only', action='store_true')
    args = parser.parse_args(); result = run(args.output, limitation_only=args.limitation_only)
    print(result['status'], flush=True)
