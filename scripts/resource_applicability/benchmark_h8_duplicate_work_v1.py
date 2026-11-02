#!/usr/bin/env python3
"""Small artificial microbenchmark. No scientific input/sampling/transpile."""
import argparse
import json
from pathlib import Path
import resource
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.is_absolute() or not a.output.is_relative_to('/home/AbeHiromu') or a.output.exists():
        raise ValueError('EXCLUSIVE_HOME_BENCHMARK_RESULT')
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); signal.alarm(120)
    import numpy as np
    import qiskit
    from qiskit import QuantumCircuit
    from trotterlib import rte_compiled_cost as compilation
    from trotterlib.df_trotter.circuit import simulate_statevector
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import FingerprintReuse
    from trottertracks.resource_applicability.h8_hash_stream_reuse_v1 import StreamingFingerprintReuse
    from trottertracks.resource_applicability.h8_control_reuse_v1 import hadamard_states_from_branches
    from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution, build_native_hadamard_wrapper
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
    def forbidden(*args, **kwargs):
        raise AssertionError('NO_SCIENTIFIC_INPUT_OR_TRANSPILE')
    np.load = forbidden; qiskit.transpile = forbidden; compilation.transpile = forbidden
    def circuit():
        c = QuantumCircuit(17); c.global_phase = -.17
        for k in range(600):
            i = k % 15
            c.rz(.17, i); c.cx(i, i+1); c.p(-.23, i); c.rzz(.11, i, i+1)
        return c
    timing = {}; fingerprints = []; stats = {}
    for name, cls in [('old', FingerprintReuse), ('new', StreamingFingerprintReuse)]:
        cold = []; warm = []
        for _ in range(3):
            c = circuit(); cache = cls()
            started = time.perf_counter(); first = cache.fingerprint(c); cold.append(time.perf_counter()-started)
            started = time.perf_counter(); second = cache.fingerprint(c); warm.append(time.perf_counter()-started)
            assert first == second
            fingerprints.append(first); stats[name] = cache.snapshot(); cache.clear()
        timing[name] = dict(cold_seconds=cold, warm_seconds=warm,
                           cold_median_seconds=statistics.median(cold), warm_median_seconds=statistics.median(warm))
    assert len(set(fingerprints)) == 1
    n = 8; c = QuantumCircuit(n+1)
    for k in range(60):
        i = k % (n-1); c.crz(.31, n, i); c.cx(n, i+1); c.cp(-.17, n, i)
    c.p(.19, n)
    e = NativeEvolution(c, n, .8, 1, 'artificial', 'ordinary', 1000, canonical_qiskit_circuit_fingerprint(c))
    rng = np.random.default_rng(17); psi = rng.normal(size=1<<n)+1j*rng.normal(size=1<<n); psi /= np.linalg.norm(psi)
    low = np.concatenate((psi, np.zeros_like(psi))); high = np.concatenate((np.zeros_like(psi), psi))
    # Build old fixture wrappers before timing: this comparison isolates state
    # action savings and excludes the additionally avoided wrapper/hash work.
    wrappers = {axis: build_native_hadamard_wrapper(e, axis=axis, include_measurement=False, max_instructions=1000)
                for axis in ('cosine', 'sine')}
    old_time = []; new_time = []; error = 0.
    for _ in range(3):
        start = time.perf_counter()
        x, y = simulate_statevector(c, low), simulate_statevector(c, high)
        old = {axis: simulate_statevector(w, low) for axis, w in wrappers.items()}
        old_time.append(time.perf_counter()-start)
        start = time.perf_counter()
        a0, b0 = simulate_statevector(c, low), simulate_statevector(c, high)
        new = hadamard_states_from_branches(a0, b0, n)
        new_time.append(time.perf_counter()-start)
        error = max(error, *(float(np.linalg.norm(old[axis]-new[axis])) for axis in old))
    assert error < 1e-12
    result = dict(schema='h8_duplicate_work_artificial_benchmark_v1', hash_fixture_instructions=2400,
        hash_timing=timing, hash_stats=stats, canonical_digest=fingerprints[0],
        hash_cold_speed_ratio=timing['old']['cold_median_seconds']/timing['new']['cold_median_seconds'],
        hash_warm_speed_ratio=timing['old']['warm_median_seconds']/timing['new']['warm_median_seconds'],
        control_system_qubits=n, control_evolution_instructions=len(c.data),
        control_old_seconds=old_time, control_new_seconds=new_time,
        control_simulation_speed_ratio=statistics.median(old_time)/statistics.median(new_time),
        control_state_max_difference=error, full_evolution_passes_old=4, full_evolution_passes_new=2,
        actual_H8_speed_measured=False, real_trajectory=0, actual_transpile=0, real_input_reads=0,
        shared_environment_changed=False, wall_cap_seconds=120,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    with a.output.open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('hash_cold_speed_ratio','hash_warm_speed_ratio',
        'control_simulation_speed_ratio','control_state_max_difference','actual_H8_speed_measured')}))


if __name__ == '__main__':
    main()
