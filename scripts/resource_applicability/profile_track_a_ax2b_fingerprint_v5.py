#!/usr/bin/env python3
"""Fixed synthetic serialization benchmark; no molecule, sampling or compile."""
import argparse
import cProfile
import gc
import json
import os
from pathlib import Path
import pstats
import resource
import statistics
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.cpu not in os.sched_getaffinity(0):
        raise ValueError('New output and available toy CPU required')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError('One-thread environment required:' + key)
    os.sched_setaffinity(0, {args.cpu})
    resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))
    import numpy as np
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    import trotterlib.rte_compiled_cost as cost
    import trotterlib.rte as rte
    from trottertracks.resource_applicability import ax2b_stream_fingerprint_v4 as old
    from trottertracks.resource_applicability import ax2b_stream_fingerprint_v5 as new
    def forbidden(*a, **k):
        raise AssertionError('Scientific loading, sampling and compilation forbidden')
    np.load = qiskit.transpile = cost.transpile = rte.sample_rte_events = forbidden

    def repeated(controlled):
        inner = QuantumCircuit(1, name='toy_inner')
        inner.rz(.13, 0); inner.p(-.17, 0); inner.global_phase = .23
        outer = QuantumCircuit(2, name='toy_outer')
        for _ in range(4): outer.append(inner.to_gate(), [0])
        outer.cx(0, 1)
        gate = outer.to_gate()
        if controlled: gate = gate.control(ctrl_state=0)
        n = 3 if controlled else 2
        c = QuantumCircuit(n, name='toy_repeated_controlled' if controlled else 'toy_repeated_nested')
        for _ in range(100 if controlled else 200): c.append(gate, list(range(n)))
        return c

    flat = QuantumCircuit(2, name='toy_flat')
    for i in range(128): flat.rz(.013 * i, i % 2)
    distinct = QuantumCircuit(1, name='toy_distinct')
    for i in range(128):
        d = QuantumCircuit(1); d.rz(.013 * i, 0)
        g = Gate('distinct', 1, []); g.definition = d
        distinct.append(g, [0], copy=False)
    d = QuantumCircuit(1)
    for i in range(256): d.rz(.031 * i, 0)
    g = Gate('large', 1, []); g.definition = d
    large = QuantumCircuit(1, name='toy_large_definition')
    for _ in range(2): large.append(g, [0], copy=False)
    fixtures = {'flat': flat, 'repeated_nested': repeated(False),
                'repeated_controlled': repeated(True), 'distinct_definitions': distinct,
                'oversize_definition': large}
    functions = {'v4': old.canonical_qiskit_circuit_fingerprint,
                 'v5': new.canonical_qiskit_circuit_fingerprint}
    rows = []
    for name, circuit in fixtures.items():
        expected = cost.canonical_qiskit_circuit_fingerprint(circuit)
        for fn in functions.values():
            assert fn(circuit) == expected
        samples = {version: [] for version in functions}
        for repetition in range(3):
            order = ('v4','v5') if repetition % 2 == 0 else ('v5','v4')
            for version in order:
                gc.collect(); start = time.perf_counter()
                assert functions[version](circuit) == expected
                samples[version].append(time.perf_counter() - start)
        medians = {version: statistics.median(values) for version, values in samples.items()}
        rows.append({'fixture': name, 'top_level_instructions': len(circuit.data),
                     'hash_equal_legacy_v4_v5': True, 'sha256': expected,
                     'warm_wall_seconds': samples, 'median_wall_seconds': medians,
                     'v4_over_v5_median_speedup': medians['v4'] / medians['v5']})
    profiles = {}
    for version, fn in functions.items():
        profile = cProfile.Profile()
        assert profile.runcall(fn, fixtures['repeated_controlled']) == next(r['sha256'] for r in rows if r['fixture'] == 'repeated_controlled')
        stats = pstats.Stats(profile)
        entries = [{'file': key[0], 'line': key[1], 'function': key[2], 'primitive_calls': v[0],
                    'total_calls': v[1], 'self_seconds': v[2], 'cumulative_seconds': v[3]} for key,v in stats.stats.items()]
        profiles[version] = {'total_calls': stats.total_calls, 'profiled_seconds': stats.total_tt,
                             'top_self_seconds': sorted(entries, key=lambda e:e['self_seconds'], reverse=True)[:20]}
    memory = {}
    circuit = fixtures['repeated_nested']
    expected = cost.canonical_qiskit_circuit_fingerprint(circuit)
    for version, fn in {'legacy': cost.canonical_qiskit_circuit_fingerprint, **functions}.items():
        gc.collect(); tracemalloc.start()
        try:
            assert fn(circuit) == expected
            retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        memory[version] = {'python_traced_peak_bytes': peak, 'python_traced_retained_bytes': retained}
    ratio = {r['fixture']:r['v4_over_v5_median_speedup'] for r in rows}
    gates = {'all_hashes_equal': all(r['hash_equal_legacy_v4_v5'] for r in rows),
             'nested_and_controlled_median_speedup_at_least_2': all(ratio[n] >= 2 for n in ('repeated_nested','repeated_controlled')),
             'other_fixtures_no_median_slowdown_above_1p5': all(ratio[n] >= 1/1.5 for n in ('flat','distinct_definitions','oversize_definition')),
             'toy_python_peak_below_legacy_quarter': memory['v5']['python_traced_peak_bytes'] < memory['legacy']['python_traced_peak_bytes']/4,
             'toy_transient_excess_below_512KiB': memory['v5']['python_traced_peak_bytes'] - memory['v5']['python_traced_retained_bytes'] < 524288}
    result = {'schema':'track_a_ax2b_fingerprint_synthetic_profile_v5', 'scope':'fixed synthetic circuits only',
              'cpu_affinity':sorted(os.sched_getaffinity(0)), 'CPU_exclusivity_guaranteed':False,
              'repetitions':3, 'alternating_order':True, 'warm_definitions':True,
              'fixtures':rows, 'profiles':profiles, 'python_tracemalloc_nested_fixture':memory,
              'gates':gates, 'all_synthetic_gates_passed':all(gates.values()),
              'cache_limits':{'payload_bytes':new.CACHE_BYTES,'entry_bytes':new.CACHE_ENTRY_BYTES,
                              'entries':new.CACHE_ENTRIES,'hash_buffer_bytes':new.HASH_BUFFER_BYTES,'scope':'single hash call'},
              'scientific_load_sampling_compile_calls':0, 'H4_run_performed':False,
              'H4_runtime_or_peak_prediction':'UNDETERMINED', 'next_stage_authorized':False,
              'limits':'cProfile adds overhead; tracemalloc is not RSS; no H4/H6 extrapolation'}
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'speedups':ratio,'memory':memory,'gates':gates},ensure_ascii=False))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
