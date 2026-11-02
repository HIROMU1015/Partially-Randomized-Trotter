"""Standalone synthetic controlled-Givens CPU fixture, without research imports."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import resource
import sys
import time

VERSION = "synthetic_controlled_givens_v0"
THREAD_ENV = {
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "QISKIT_PARALLEL": "false",
    "QISKIT_NUM_PROCS": "1", "RAYON_NUM_THREADS": "1",
}
COMPILER = {
    "basis_gates": ["rz", "sx", "x", "cx"], "optimization_level": 1,
    "seed_transpiler": 17, "backend": None, "coupling_map": None,
    "initial_layout": None, "layout_method": None, "routing_method": None,
    "target": None, "approximation_degree": 1.0,
    "unitary_synthesis_method": "default", "unitary_synthesis_plugin_config": None,
    "hls_config": None, "qubits_initially_zero": True, "num_processes": 1,
}
WORKERS = [1, 6, 12, 16]
SCALES = {"small": 16, "medium": 64, "large": 128}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_environment():
    for key, value in THREAD_ENV.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"Process-only environment requires {key}={value}")


def freeze(directory):
    tasks = []
    for scale, repeats in SCALES.items():
        for index in range(10):
            seed_key = {"policy": VERSION, "synthetic_master_seed": 20261005, "scale": scale, "index": index}
            tasks.append({"task_id": f"{scale}-{index:02d}", "scale": scale,
                          "layers": repeats, "seed": int(digest(seed_key)[:16], 16),
                          "axis": "cosine" if index % 2 == 0 else "sine"})
    definition = {
        "schema_version": VERSION, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tasks": tasks, "tasks_fingerprint": digest(tasks), "system_qubits": 8,
        "total_qubits": 9, "classical_bits": 1, "ancilla_qubit": 8,
        "fixed_scales_before_first_transpile": SCALES, "worker_conditions": WORKERS,
        "compiler": COMPILER, "thread_environment": THREAD_ENV,
        "benchmark_transpile_cap": 120, "semantic_test_transpile_cap": 4,
        "total_transpile_cap": 124, "wall_cap_seconds": 1800,
        "per_process_address_space_cap_bytes": 4 * 1024**3,
        "per_process_cpu_time_cap_seconds": 600, "own_process_nice": 19,
        "guard": {"load1_plus_requested_workers_max": 32, "minimum_mem_available_bytes": 64 * 1024**3,
                  "cpu_pressure_avg10_max_percent": 5.0},
        "research_coefficients_used": False, "science_inputs_used": False,
        "scale_is_not_required_to_match_old_compiled_gate_counts": True,
    }
    save(directory / "task_definitions_v0.json", definition, exclusive=True)
    print(json.dumps({"status": "SYNTHETIC_TASKS_FROZEN", "task_count": len(tasks), "fingerprint": definition["tasks_fingerprint"]}), flush=True)


def build(task):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    rng = random.Random(task["seed"])
    circuit = QuantumCircuit(9, 1)
    circuit.h(8)
    circuit.p(0.173, 8)  # Relative phase, not an ignorable full-wrapper scalar.
    for _ in range(task["layers"]):
        gates = [XXPlusYYGate(rng.uniform(0.1, 1.1), rng.uniform(-0.7, 0.7)).control(1) for _ in range(7)]
        for index, gate in enumerate(gates):
            circuit.append(gate, [8, index, index + 1])
        for index in range(8):
            circuit.crz(rng.uniform(-1.3, 1.3), 8, index)
        for index in reversed(range(7)):
            circuit.append(gates[index].inverse(), [8, index, index + 1])
    if task["axis"] == "sine": circuit.sdg(8)
    circuit.h(8)
    circuit.measure(8, 0)
    return circuit


def compile_one(circuit):
    from qiskit import transpile
    return transpile(circuit, **COMPILER)


def tests(directory):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.quantum_info import Operator, Statevector
    rng = np.random.default_rng(981205)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    initial = Statevector(np.concatenate([state, np.zeros(4)]))
    checks = []
    actual_transpiles = 0
    for phase in [0.0, 0.173]:
        unitary = QuantumCircuit(2)
        unitary.append(XXPlusYYGate(0.71, -0.29), [0, 1])
        unitary.rz(-0.37, 0)
        unitary.rz(0.23, 1)
        unitary.global_phase = phase
        u = Operator(unitary).data
        controlled = QuantumCircuit(3)
        controlled.append(unitary.to_gate().control(1), [2, 0, 1])
        expected = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), u]])
        control_residual = float(np.max(np.abs(Operator(controlled).data - expected)))
        assert control_residual < 1e-12
        signal = np.vdot(state, u @ state)
        for axis in ["cosine", "sine"]:
            wrapper = QuantumCircuit(3, 1)
            wrapper.h(2)
            wrapper.compose(controlled, inplace=True)
            if axis == "sine": wrapper.sdg(2)
            wrapper.h(2)
            wrapper.measure(2, 0)
            actual_transpiles += 1
            compiled = compile_one(wrapper)
            expected_value = float(signal.real if axis == "cosine" else signal.imag)
            residuals = []
            for candidate in [wrapper, compiled]:
                probabilities = initial.evolve(candidate.remove_final_measurements(inplace=False)).probabilities()
                estimate = float(probabilities[:4].sum() - probabilities[4:].sum())
                residuals.append(abs(estimate - expected_value))
            assert max(residuals) < 1e-12
            assert compiled.num_clbits == 1 and compiled.count_ops().get("measure") == 1
            assert set(compiled.count_ops()) <= set(COMPILER["basis_gates"]) | {"measure"}
            checks.append({"phase": phase, "axis": axis, "controlled_operator_residual": control_residual,
                           "axis_residual_before_after_compile": residuals, "compiled_size": compiled.size()})
    result = {"status": "PURE_SYNTHETIC_WRAPPER_TESTS_PASS", "checks": checks,
              "actual_transpiles": actual_transpiles, "synthetic_wrapper_records": len(checks),
              "science_wrapper_records": 0, "quantum_shots": 0, "repository_module_imports": 0}
    save(directory / "synthetic_tests_v0.json", result, exclusive=True)
    print(json.dumps(result), flush=True)


def own_limits():
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 19: os.nice(19 - current)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))


def worker(task):
    validate_environment()
    start = time.perf_counter()
    circuit = build(task)
    built = time.perf_counter()
    compiled = compile_one(circuit)
    finished = time.perf_counter()
    return {"task_id": task["task_id"], "scale": task["scale"], "seed": task["seed"], "axis": task["axis"],
            "build_wall_s": built - start, "transpile_wall_s": finished - built,
            "uncompiled_size": circuit.size(), "compiled_size": compiled.size(), "compiled_depth": compiled.depth(),
            "gate_counts": {str(k): int(v) for k, v in compiled.count_ops().items()},
            "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "pid": os.getpid(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "os_thread_count": int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Threads:'))),
            "status": "COMPLETE", "actual_transpile_calls": 1}


def host_state():
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, val = line.split(':', 1)
        if key in ['MemAvailable', 'SwapFree']: mem[key] = int(val.strip().split()[0]) * 1024
    pressure = Path('/proc/pressure/cpu').read_text().splitlines()[0]
    avg10 = float(next(x.split('=')[1] for x in pressure.split() if x.startswith('avg10=')))
    return {"loadavg": list(os.getloadavg()), "memory": mem, "cpu_pressure_avg10": avg10}


def allowed(state, workers):
    reasons = []
    if state['loadavg'][0] + workers > 32: reasons.append('shared load plus worker request exceeds conservative 32-core threshold')
    if state['memory']['MemAvailable'] < 64 * 1024**3: reasons.append('available RAM below 64 GiB')
    if state['cpu_pressure_avg10'] > 5: reasons.append('shared CPU pressure exceeds 5 percent')
    if len(os.sched_getaffinity(0)) < workers * 4: reasons.append('requested workers exceed one quarter of available CPU affinity')
    return reasons


def owned_rss(pids):
    total = 0
    for pid in pids:
        try:
            lines = Path(f'/proc/{pid}/status').read_text().splitlines()
            total += int(next(line.split()[1] for line in lines if line.startswith('VmRSS:')))
        except (OSError, StopIteration, ValueError):
            pass
    return total


def terminate_owned_pool(pool):
    for process in list(pool._processes.values()):
        if process.is_alive(): process.terminate()
    pool.shutdown(wait=True, cancel_futures=True)


def run(directory):
    definition = json.loads((directory / 'task_definitions_v0.json').read_text())
    assert definition['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert definition['tasks_fingerprint'] == digest(definition['tasks'])
    assert (directory / 'synthetic_tests_v0.json').exists()
    if (directory / 'benchmark_result_v0.json').exists(): raise RuntimeError('No implicit benchmark retry/resume')
    overall_start = time.perf_counter()
    overall = {"status": "RUNNING", "task_fingerprint": definition['tasks_fingerprint'], "conditions": [],
               "science_wrapper_records": 0, "quantum_shots": 0, "gpu_access": 0}
    actual_calls = 0
    for count in WORKERS:
        before = host_state()
        reasons = allowed(before, count)
        if reasons:
            overall['conditions'].append({"workers": count, "status": "SKIPPED_RESOURCE_GUARD", "reasons": reasons, "before": before})
            continue
        condition = {"workers": count, "status": "RUNNING", "before": before, "records": [], "failures": [], "peak_owned_pool_rss_kib": 0}
        overall['conditions'].append(condition)
        start = time.perf_counter()
        pool = ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn'), initializer=own_limits)
        futures = {pool.submit(worker, task): task for task in definition['tasks']}
        pending = set(futures)
        stopped = False
        while pending:
            ready, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            pids = [os.getpid()] + [process.pid for process in pool._processes.values()]
            condition['peak_owned_pool_rss_kib'] = max(condition['peak_owned_pool_rss_kib'], owned_rss(pids))
            for future in ready:
                task = futures[future]
                try:
                    record = future.result()
                    condition['records'].append(record)
                    actual_calls += record['actual_transpile_calls']
                except Exception as error:
                    condition['failures'].append({"task_id": task['task_id'], "error_type": type(error).__name__, "error": str(error), "retry": False, "actual_transpile_calls": None})
            if time.perf_counter() - overall_start >= 1780:
                condition['stop_reason'] = 'global wall budget before 1800 seconds'
                stopped = True
            current = host_state()
            if current['loadavg'][0] > 32 or current['memory']['MemAvailable'] < 64 * 1024**3 or current['cpu_pressure_avg10'] > 5:
                condition['stop_reason'] = 'shared resource guard changed during condition'
                stopped = True
            if condition['failures']: stopped = True
            if stopped:
                condition['unresolved_task_ids'] = [futures[f]['task_id'] for f in pending]
                terminate_owned_pool(pool)
                break
            if len(condition['records']) % 5 == 0 and ready:
                print(json.dumps({"workers": count, "completed": len(condition['records']), "peak_owned_rss_kib": condition['peak_owned_pool_rss_kib']}), flush=True)
        if not stopped: pool.shutdown(wait=True)
        condition['wall_s'] = time.perf_counter() - start
        condition['after'] = host_state()
        condition['status'] = 'STOPPED_WITHOUT_RETRY' if stopped else 'COMPLETE'
        condition['records'].sort(key=lambda row: row['task_id'])
        condition['tasks_per_min'] = len(condition['records']) * 60 / condition['wall_s']
        save(directory / 'benchmark_result_v0.json', overall)
        print(json.dumps({key: condition[key] for key in ['workers', 'status', 'wall_s', 'tasks_per_min', 'peak_owned_pool_rss_kib']}), flush=True)
        if stopped: break
    completed = [x for x in overall['conditions'] if x['status'] == 'COMPLETE']
    baseline = next((x for x in completed if x['workers'] == 1), None)
    for condition in completed:
        condition['speedup_vs_one_worker'] = baseline['wall_s'] / condition['wall_s'] if baseline else None
        condition['parallel_efficiency'] = condition['speedup_vs_one_worker'] / condition['workers'] if baseline else None
        if baseline:
            first = {x['task_id']: (x['compiled_size'], x['compiled_depth'], x['gate_counts']) for x in baseline['records']}
            condition['same_gate_metrics_as_one_worker'] = all(first[x['task_id']] == (x['compiled_size'], x['compiled_depth'], x['gate_counts']) for x in condition['records'])
    overall.update(status='SYNTHETIC_BENCHMARK_COMPLETE' if len(completed) == 4 else 'SYNTHETIC_BENCHMARK_PARTIAL_WITH_STOP',
                   wall_s=time.perf_counter() - overall_start, actual_transpile_calls=actual_calls,
                   allocated_task_slots=sum(len(x.get('records', [])) + len(x.get('failures', [])) + len(x.get('unresolved_task_ids', [])) for x in overall['conditions']),
                   synthetic_wrapper_records=sum(len(x.get('records', [])) for x in overall['conditions']),
                   oom_observed=any('MemoryError' in x.get('error_type', '') for c in overall['conditions'] for x in c.get('failures', [])),
                   no_implicit_retry=True, resource_limits_scope='own driver and children only')
    assert actual_calls <= 120
    save(directory / 'benchmark_result_v0.json', overall)
    print(json.dumps({key: overall[key] for key in ['status', 'wall_s', 'actual_transpile_calls', 'synthetic_wrapper_records']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'tests', 'run'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    validate_environment()
    own_limits()
    {'freeze': freeze, 'tests': tests, 'run': run}[args.mode](args.directory)


if __name__ == '__main__':
    main()
