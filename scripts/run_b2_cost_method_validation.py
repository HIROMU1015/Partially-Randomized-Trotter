"""Artificial-only B2 method comparison. No molecular input options or MC."""
from dataclasses import asdict
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np
from qiskit.quantum_info import Operator, Pauli, Statevector
from qiskit.converters import circuit_to_dag
from b2_cost_method_cases import (artificial_round,event_pattern,expected_controlled,
    measurement_error,native_record)
from trotterlib.b2_p1_compiled_cost import request_instruction_bound
from trotterlib.b2_streamed_native_cost import (stream_native,cached_compiler,
    measured_metrics,split_compile,iter_b2_step_requests)
from trotterlib.df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider, _compile_pair
from trotterlib.rpe_resource_accounting import RPE_COST_METRICS
from trotterlib.rte_compiled_cost import (TranspiledCircuitCostCache,
    canonical_qiskit_circuit_fingerprint,compiler_settings_hash,transpile_and_measure_cost)
from trottertracks.resource_applicability.ax2a_native_df import build_native_hadamard_wrapper
from trottertracks.o1_m1_parallel_price import ParallelMatchedNoPrepCostProvider

COMPILER_HASH = 'a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8'
CASES = [(i,ld,'mixed') for i in range(4) for ld in (2,3)] + [(2,2,'repeat')]


def forbid_molecular_inputs():
    def forbidden(*args,**kwargs):
        raise RuntimeError('Artificial validation only: saved numerical input/solver forbidden')
    np.load = forbidden
    import trotterlib.df_hamiltonian as module
    module.solve_df_ground_state = forbidden
    module.build_df_h_d_from_molecule = forbidden


def metrics(cost):
    return {k:getattr(cost,k) for k in RPE_COST_METRICS}


def difference(reference,candidate):
    signed = {k:candidate[k]-reference[k] for k in RPE_COST_METRICS}
    return dict(status='EXACT_MATCH' if not any(signed.values()) else 'APPROXIMATE',
        reference=reference,candidate=candidate,signed_difference=signed,
        absolute_difference={k:abs(v) for k,v in signed.items()},
        relative_difference={k:(signed[k]/reference[k] if reference[k] else
                              (0. if candidate[k] == 0 else None)) for k in signed})


def timed_build(function):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    value = function()
    elapsed = time.perf_counter()-start
    _,peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value,dict(elapsed_seconds=elapsed,Python_traced_peak_bytes=peak,
        process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        instructions=len(value.circuit.data))


def reference_native(round_,pattern):
    request = round_.explicit_request(tuple(event_pattern(round_,pattern)))
    return round_.native(request,max_instructions=request_instruction_bound(request))


def request_memory_diagnostic():
    """Synthetic Q64 requests only; neither Q64 circuit is built or compiled."""
    _,round_=artificial_round(6,3)
    rows={}
    for name in ('materialized','stream'):
        gc.collect();tracemalloc.start();tick=time.perf_counter()
        if name=='materialized':
            value=round_.explicit_request(tuple(event_pattern(round_,'mixed')))
            count=len(value.iter_step_requests())
        else:
            value=None
            count=sum(1 for _ in iter_b2_step_requests(round_,event_pattern(round_,'mixed')))
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        rows[name]=dict(request_count=count,Python_traced_peak_bytes=peak,
                        elapsed_seconds=time.perf_counter()-tick)
        del value
    return dict(Q=64,artificial_only=True,circuit_builds=0,compile_calls=0,methods=rows)


def synthetic_worker(task):
    """B2-only task injected into the original provider's unchanged scheduler."""
    forbid_molecular_inputs()
    index,ld,pattern = task
    _,round_ = artificial_round(index,ld)
    if compiler_settings_hash(round_.context.compiler) != COMPILER_HASH:
        raise ValueError('Fixed compiler mismatch')
    tick = time.perf_counter()
    evolution = stream_native(round_,event_pattern(round_,pattern))
    pair = _compile_pair(evolution,round_.context.compiler,
        compile_function=transpile_and_measure_cost,backend=None,
        wrapper_builder=lambda e,axis:build_native_hadamard_wrapper(e,axis=axis,
            max_instructions=len(e.circuit.data)+4))
    return os.getpid(),dict(case=list(task),axes={axis:metrics(c) for axis,c in pair.items()},
        compiler_hash=COMPILER_HASH,evolution_fingerprint=evolution.circuit_fingerprint,
        elapsed_seconds=time.perf_counter()-tick),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def validate_case(index,ld,pattern):
    h,round_ = artificial_round(index,ld)
    reference,ref_perf = timed_build(lambda:reference_native(round_,pattern))
    streamed,stream_perf = timed_build(lambda:stream_native(round_,event_pattern(round_,pattern)))
    events = tuple(event_pattern(round_,pattern))
    request = round_.explicit_request(events)
    fused_circuit = QiskitDFPartialS2RepeatedCircuitBuilder().build(request,
        construction_policy='boundary_optimized').circuit
    fused = native_record(fused_circuit,round_)
    target = expected_controlled(h,round_,events)
    semantics = {}
    for name,evolution in (('reference',reference),('stream',streamed),('boundary_fusion',fused)):
        error = float(np.max(abs(Operator(evolution.circuit).data-target)))
        semantics[name] = dict(controlled_operator_max_abs=error,phase_fitting=False,
                              passed=error<1e-10)
        if error >= 1e-10:
            raise ValueError('STOP controlled relative phase/order: '+name)
    if reference.circuit_fingerprint != streamed.circuit_fingerprint:
        raise ValueError('STOP stream does not preserve original native circuit identity')
    result = dict(case=[index,ld,pattern],Q=round_.q,source_input='ARTIFICIAL_TWO_MODE_B2_RANK4',
        hamiltonian_hash=round_.context.df_hamiltonian,compiler_hash=compiler_settings_hash(round_.context.compiler),
        original_native_fingerprint=reference.circuit_fingerprint,
        stream_native_fingerprint=streamed.circuit_fingerprint,mathematical_correctness=semantics,
        build_performance=dict(reference=ref_perf,stream=stream_perf),axes={})
    cache = TranspiledCircuitCostCache(retain_transpiled_circuits=True)
    cached = cached_compiler(cache)
    psi = np.array([.2,.3j,-.4,.5j]);psi/=np.linalg.norm(psi)
    signal = np.vdot(psi,target[4:,4:]@psi)
    for axis in ('cosine','sine'):
        wrapper = build_native_hadamard_wrapper(reference,axis=axis,
            max_instructions=len(reference.circuit.data)+4)
        tick = time.perf_counter()
        whole = transpile_and_measure_cost(wrapper,round_.context.compiler,backend=None)
        elapsed = time.perf_counter()-tick
        whole_metrics = metrics(whole)
        reference_meaning=measurement_error(wrapper,whole.transpiled_circuit)
        if reference_meaning['measured_Z_effect_max_abs'] >= 1e-10:
            raise ValueError('STOP whole-compile reference meaning is ineligible')
        row = dict(reference=whole_metrics,reference_compile_seconds=elapsed,
            wrapper_fingerprint=canonical_qiskit_circuit_fingerprint(wrapper),
            uncompiled_DAG_operation_nodes=len(circuit_to_dag(wrapper).op_nodes()),
            compiled_DAG_operation_nodes=len(circuit_to_dag(whole.transpiled_circuit).op_nodes()),
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,candidates={})
        expected = signal.real if axis=='cosine' else signal.imag
        actual = Statevector(np.r_[psi,np.zeros(4)]).evolve(
            wrapper.remove_final_measurements(inplace=False)).expectation_value(Pauli('ZII')).real
        row['wrapper_Re_Im_error'] = float(abs(actual-expected))
        if abs(actual-expected) >= 1e-10:
            raise ValueError('STOP X/Y convention')
        for name,evolution in (('stream',streamed),('boundary_fusion',fused)):
            w = build_native_hadamard_wrapper(evolution,axis=axis,
                max_instructions=len(evolution.circuit.data)+4)
            tick = time.perf_counter()
            cost = cached(w,round_.context.compiler,backend=None)
            cold = time.perf_counter()-tick
            tick = time.perf_counter()
            again = cached(w,round_.context.compiler,backend=None)
            warm = time.perf_counter()-tick
            if metrics(cost)!=metrics(again):
                raise ValueError('STOP cached metrics differ')
            meaning = measurement_error(wrapper,cost.transpiled_circuit)
            if meaning['measured_Z_effect_max_abs'] >= 1e-10:
                raise ValueError('STOP compiled measurement meaning: '+name)
            row['candidates'][name] = dict(**difference(whole_metrics,metrics(cost)),
                mathematical_correctness=meaning,compile_cold_seconds=cold,
                cache_warm_lookup_seconds=warm,wrapper_instructions=len(w.data))
        tick = time.perf_counter()
        split,sums = split_compile(wrapper,round_.context.compiler,pieces=max(2,round_.q))
        meaning = measurement_error(wrapper,split)
        if meaning['measured_Z_effect_max_abs'] >= 1e-10:
            raise ValueError('STOP split compilation measurement meaning')
        row['candidates']['split_join'] = dict(**difference(whole_metrics,measured_metrics(split)),
            mathematical_correctness=meaning,compile_seconds=time.perf_counter()-tick,
            pieces=max(2,round_.q),naive_block_metric_sum=sums,
            depth_policy='measure composed compiled DAG; never sum block depths')
        row['reference_compiled_meaning'] = reference_meaning
        result['axes'][axis] = row
    result['cache'] = dict(hits=cache.hit_count,misses=cache.miss_count,
        draws_removed=0,policy='reuse compilation only; retain each observation')
    return result


def run(output):
    forbid_molecular_inputs()
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    _,test_round = artificial_round(0,3)
    compiler = test_round.context.compiler
    if compiler_settings_hash(compiler)!=COMPILER_HASH or importlib.metadata.version('qiskit')!='1.3.0':
        raise ValueError('STOP fixed compiler/runtime mismatch')
    if platform.node()!='GPUA100-gleap02':
        raise ValueError('STOP expected SERVER 2')
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    source_paths=('src/trotterlib/b2_streamed_native_cost.py',
        'src/trottertracks/o1_m1_parallel_price.py','tests/b2_cost_method_cases.py',
        'scripts/run_b2_cost_method_validation.py','tests/test_b2_streamed_native_cost.py')
    hashes={}
    for path in source_paths:
        raw=(ROOT/path).read_bytes()
        if raw!=subprocess.check_output(['git','show',source+':'+path],cwd=ROOT):
            raise ValueError('STOP uncommitted technical SOURCE: '+path)
        hashes[path]=hashlib.sha256(raw).hexdigest()
    result = dict(schema_version='b2_artificial_cost_method_validation_v1',status='RUNNING',
        source_commit=source,source_sha256=hashes,artificial_only=True,real_H4_build_compile=0,price_MC=None,
        expected_B2_prices=None,confidence_intervals=None,total_RPE_cost=None,mandatory_stop=True,
        compiler=asdict(compiler),compiler_hash=COMPILER_HASH,backend_fingerprint='no_backend',
        scope='single_hadamard_interrogation_without_state_preparation',cases=[],
        runtime=dict(python=sys.version,host=platform.node(),available_cpus=len(os.sched_getaffinity(0)),
            load_average=os.getloadavg(),packages={p:importlib.metadata.version(p)
                for p in ('qiskit','numpy','scipy','openfermion','qiskit-nature')}))
    started = time.perf_counter()
    try:
        for case in CASES:
            row = validate_case(*case)
            result['cases'].append(row)
            print('Artificial case '+str(case)+' validated',flush=True)
            (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        result['request_memory_diagnostic']=request_memory_diagnostic()
        tasks = CASES + CASES[:2]  # repeated observations stay in submission order
        tick = time.perf_counter()
        serial = [synthetic_worker(t)[1] for t in tasks]
        serial_elapsed = time.perf_counter()-tick
        provider = ParallelMatchedNoPrepCostProvider(compiler)
        tick = time.perf_counter()
        parallel = provider._map(synthetic_worker,tasks,memory_probe=True)
        parallel_elapsed = time.perf_counter()-tick
        for task,left,right in zip(tasks,serial,parallel,strict=True):
            if left['case'] != list(task) or right['case'] != list(task) or any(
                    left[k]!=right[k] for k in ('axes','compiler_hash','evolution_fingerprint')):
                raise ValueError('STOP parallel ordering/identity/price mismatch')
        result['parallel'] = dict(status='EXACT_MATCH',serial_seconds=serial_elapsed,
            parallel_seconds=parallel_elapsed,speedup=serial_elapsed/parallel_elapsed,
            execution=provider.last_execution,ordered_task_count=len(tasks),
            repeated_observations_retained=True,rows=parallel)
        result['status']='ARTIFICIAL_METHOD_VALIDATION_COMPLETE'
    except Exception as exc:
        result.update(status='STOP_TECHNICAL_FAILURE',error=type(exc).__name__+': '+str(exc))
    result['elapsed_seconds']=time.perf_counter()-started
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    local=ROOT/'.venv/cache';local.mkdir(parents=True,exist_ok=True)
    os.environ['XDG_CACHE_HOME']=str(local)
    os.environ['MPLCONFIGDIR']=str(local/'matplotlib')
    result=run(args.output)
    print(result['status'],result.get('error',''),flush=True)
    sys.exit(0 if result['status']=='ARTIFICIAL_METHOD_VALIDATION_COMPLETE' else 1)
