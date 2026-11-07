"""Artificial B2 pipeline pilot; split emission, hashes, wrappers and DAGs.

Six independent fork children: Q1/Q2/Q8 times legacy/streamed fingerprint.
One observation per condition, without production speed/large-round claims.
Reuse the earlier profiler; segment at the real injected hash callback instead
of returning a placeholder hash or reimplementing gate/wrapper construction.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc

import qiskit
from qiskit.converters import circuit_to_dag

from benchmark_streamed_circuit_fingerprint import profile, rss
from b2_cost_method_cases import artificial_round, event_pattern
from trotterlib.b2_streamed_native_cost import stream_native
from trotterlib.pauli_signal_adapter import digest
from trotterlib.o1_m1_resource_bridge import _compile_pair
from trotterlib.rpe_resource_accounting import RPE_COST_METRICS
from trotterlib.rte_compiled_cost import (
    canonical_qiskit_circuit_fingerprint, compiler_settings_hash,
    transpile_and_measure_cost,
)
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint
from trottertracks.resource_applicability.ax2a_native_df import build_native_hadamard_wrapper

ROOT = Path(__file__).resolve().parents[1]


def begin_segment():
    gc.collect()
    before = rss()
    tracemalloc.start()
    return before, time.perf_counter()


def end_segment(state):
    elapsed = time.perf_counter() - state[1]
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'seconds_with_tracemalloc': elapsed, 'Python_traced_peak_bytes': peak,
            'Python_traced_retained_bytes': current, 'RSS_before': state[0],
            'RSS_after': rss(), 'stage_peak_RSS_KiB': None}


def measured_native(round_, events, hasher, stages):
    state = begin_segment()
    calls = 0
    def final_hash(circuit):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise AssertionError('One final native fingerprint expected')
        stages['flat_native_emission'] = end_segment(state)
        value, stages['native_fingerprint'] = profile(lambda: hasher(circuit))
        return value
    try:
        result = stream_native(round_, iter(events), fingerprint_function=final_hash)
        if calls != 1:
            raise AssertionError('Native hasher was not called')
        return result
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()


def measured_wrapper(evolution, axis, hasher, stages):
    prefix = 'wrapper_' + axis + '_'
    state = begin_segment()
    copying = None
    calls = 0
    def validate_hash(circuit):
        nonlocal copying, calls
        calls += 1
        if calls != 1:
            raise AssertionError('One validation fingerprint expected')
        stages[prefix+'precheck'] = end_segment(state)
        value, stages[prefix+'validation_fingerprint'] = profile(lambda: hasher(circuit))
        copying = begin_segment()
        return value
    try:
        result = build_native_hadamard_wrapper(evolution, axis=axis,
            max_instructions=len(evolution.circuit.data)+4, fingerprint_function=validate_hash)
        if calls != 1 or copying is None:
            raise AssertionError('Wrapper hasher was not called')
        stages[prefix+'compose_and_measure'] = end_segment(copying)
        return result
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()


def worker(m, mode, connection):
    result = dict(m=m, Q=1 << m, mode=mode, stages={}, axes={}, compile_calls_started=0)
    start = time.perf_counter()
    try:
        hasher = (canonical_qiskit_circuit_fingerprint if mode == 'legacy'
                  else streamed_canonical_qiskit_circuit_fingerprint)
        (_, round_), result['stages']['artificial_preparation'] = profile(lambda: artificial_round(m, 2))
        events = tuple(event_pattern(round_, 'mixed'))
        result['input'] = dict(model='ARTIFICIAL_TWO_MODE_B2_RANK4', L_D=2,
            delta=round_.delta, r=round_.r, K=round_.K, event_count=len(events),
            event_identity=digest([e.to_dict() for e in events]),
            preparation_hash=round_.preparation.preparation_hash, pattern='mixed', new_sampling=False)
        compiler = replace(round_.context.compiler, optimization_level=1, transpiler_seed=7)
        if (tuple(compiler.basis_gates) != ('rz','sx','x','cx') or
                compiler.backend_name is not None or compiler.coupling_map is not None):
            raise ValueError('Expected logical all-to-all artificial compiler')
        result['compiler_hash'] = compiler_settings_hash(compiler)
        result['backend_fingerprint'] = 'no_backend'
        result['load_average'] = os.getloadavg()
        evolution = measured_native(round_, events, hasher, result['stages'])
        result['native'] = dict(instructions=len(evolution.circuit.data),
            fingerprint=evolution.circuit_fingerprint, qubits=evolution.circuit.num_qubits,
            global_phase=float(evolution.circuit.global_phase))
        copied, result['stages']['native_copy_diagnostic'] = profile(evolution.circuit.copy)
        result['copy_instructions'] = len(copied.data)
        del copied; gc.collect()
        result['RSS_after_copy_release'] = rss()
        wrappers, costs = {}, {}
        input_fingerprints = {}
        def wrapper_builder(value, axis):
            wrapper = measured_wrapper(value, axis, hasher, result['stages'])
            wrappers[axis] = wrapper
            return wrapper
        def input_hash(wrapper):
            axis = next(k for k,v in wrappers.items() if v is wrapper)
            fingerprint, result['stages'][axis+'_input_fingerprint'] = profile(lambda: hasher(wrapper))
            input_fingerprints[axis] = fingerprint
            return fingerprint
        def compile_one(wrapper, settings, *, backend, actual_circuit_fingerprint):
            axis = next(k for k,v in wrappers.items() if v is wrapper)
            fingerprint = input_fingerprints[axis]
            if settings != compiler or backend is not None or actual_circuit_fingerprint != fingerprint:
                raise ValueError('Fixed compiler and actual wrapper hash required')
            dag, result['stages'][axis+'_input_DAG'] = profile(lambda: circuit_to_dag(wrapper, copy_operations=True))
            before_operations = dag.size()
            del dag; gc.collect()
            released = rss()
            result['compile_calls_started'] += 1
            cost, result['stages'][axis+'_whole_compile_and_metrics'] = profile(
                lambda: transpile_and_measure_cost(wrapper, compiler,
                    actual_circuit_fingerprint=fingerprint, backend=None))
            compiled = cost.transpiled_circuit
            after_fingerprint, result['stages'][axis+'_compiled_fingerprint'] = profile(lambda: hasher(compiled))
            dag, result['stages'][axis+'_compiled_DAG'] = profile(lambda: circuit_to_dag(compiled, copy_operations=True))
            after_operations = dag.size()
            del dag; gc.collect()
            if wrapper.num_clbits != 1 or compiled.count_ops().get('measure') != 1:
                raise ValueError('Full measured wrapper required')
            result['axes'][axis] = dict(input_fingerprint=fingerprint, compiled_fingerprint=after_fingerprint,
                instructions=len(wrapper.data), measurements=1, qubits=compiled.num_qubits,
                input_DAG_operations=before_operations, compiled_DAG_operations=after_operations,
                input_DAG_RSS_after_release=released, compiled_DAG_RSS_after_release=rss(),
                compiled_global_phase=cost.global_phase,
                metrics={name:getattr(cost,name) for name in RPE_COST_METRICS},
                scope='single_hadamard_interrogation_without_state_preparation',
                state_preparation_included=False)
            return cost
        costs = _compile_pair(evolution, compiler, compile_function=compile_one,
            backend=None, wrapper_builder=wrapper_builder, fingerprint_function=input_hash)
        gc.collect()
        result['retained_at_end'] = dict(native=True, wrappers=list(wrappers), compiled_results=list(costs),
            diagnostic_DAGs=False, extra_native_copy=False, explicit_events=len(events), RSS=rss())
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        result['wall_seconds_with_instrumentation'] = time.perf_counter()-start
        connection.send(result); connection.close()


def compare(reference, candidate):
    checks = {name:reference[name] == candidate[name]
              for name in ('m','Q','input','compiler_hash','backend_fingerprint','native')}
    differences = {}
    for axis in ('cosine','sine'):
        a,b = reference['axes'][axis], candidate['axes'][axis]
        for name in ('input_fingerprint','compiled_fingerprint','instructions','measurements','qubits',
                     'input_DAG_operations','compiled_DAG_operations','compiled_global_phase','scope','state_preparation_included'):
            checks[axis+'_'+name] = a[name] == b[name]
        differences[axis] = {name:b['metrics'][name]-a['metrics'][name] for name in RPE_COST_METRICS}
        checks[axis+'_six_metrics'] = not any(differences[axis].values())
    return dict(Q=reference['Q'], status='EXACT_MATCH' if all(checks.values()) else 'MISMATCH',
                checks=checks, signed_metric_differences=differences)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    paths = ['scripts/benchmark_b2_fingerprint_pipeline.py','scripts/benchmark_streamed_circuit_fingerprint.py',
        'src/trotterlib/streamed_circuit_fingerprint.py','src/trotterlib/b2_streamed_native_cost.py',
        'src/trottertracks/resource_applicability/ax2a_native_df.py','src/trotterlib/rte_compiled_cost.py',
        'src/trotterlib/o1_m1_resource_bridge.py',
        'tests/b2_cost_method_cases.py','tests/test_b2_fingerprint_pipeline.py']
    hashes = {}
    for path in paths:
        raw=(ROOT/path).read_bytes()
        if raw != subprocess.check_output(['git','show',source+':'+path],cwd=ROOT):
            raise ValueError('Committed SOURCE required: '+path)
        hashes[path] = hashlib.sha256(raw).hexdigest()
    result = dict(status='RUNNING', source_commit=source, source_sha256=hashes, rows=[], comparisons=[],
        runtime={'python':sys.version,'qiskit':qiskit.__version__,'host':platform.node(),
                 'available_cpus':len(os.sched_getaffinity(0)), 'MemAvailable':next(
                     l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))},
        protocol='one observation per mode/Q; fresh fork per observation; sequential; alternate method order; stage-incremental tracing',
        RSS_scope='current before/after plus child cumulative high-water; individual stage RSS peak unmeasured',
        real_H4_calls=0, real_H4_compiles=0, expected_prices=None, price_MC=None, m17_feasibility=None,
        total_RPE_cost=None)
    def save():
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    ctx=mp.get_context('fork')
    try:
        for index,m in enumerate((0,1,3)):
            pair={}
            for mode in (('legacy','stream') if index%2==0 else ('stream','legacy')):
                receive,send=ctx.Pipe(duplex=False)
                process=ctx.Process(target=worker,args=(m,mode,send))
                process.start();send.close()
                try:
                    row=receive.recv()
                finally:
                    receive.close();process.join()
                result['rows'].append(row);save()
                if process.exitcode != 0 or row['status'] != 'PASS':
                    raise RuntimeError(str(row.get('error','worker exit')))
                pair[mode]=row
                print('Q',1<<m,mode,'done',flush=True)
            comparison=compare(pair['legacy'],pair['stream'])
            result['comparisons'].append(comparison);save()
            if comparison['status'] != 'EXACT_MATCH':
                raise ValueError('Pipeline identity or price mismatch')
        for path,sha in hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != sha:
                raise ValueError('SOURCE changed: '+path)
        result['status']='PASS_ARTIFICIAL_PIPELINE_IDENTICAL'
    except Exception as exc:
        result.update(status='INCONCLUSIVE',stop_reason=repr(exc))
    save()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    print(result['status'],flush=True)
    sys.exit(0 if result['status']=='PASS_ARTIFICIAL_PIPELINE_IDENTICAL' else 1)
