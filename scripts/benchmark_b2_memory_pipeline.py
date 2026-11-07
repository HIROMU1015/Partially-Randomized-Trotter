"""Artificial-only production pair pricing and request-storage measurements.

Reuse the corrected external RSS observer, existing flat unit and B2 fixtures.
Large legacy hash/retention observations are reused from c6f3f079; no H4 or MC.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import benchmark_large_artificial_memory as old
from b2_cost_method_cases import artificial_round, event_pattern
from trotterlib.b2_streamed_native_cost import iter_b2_step_requests, measured_metrics
from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider
from trotterlib.rpe_resource_accounting import RPE_COST_METRICS
from trotterlib.rte_compiled_cost import measure_compiled_metrics, transpile_and_measure_cost
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint as streamed_hash
from trottertracks.resource_applicability.ax2a_native_df import NativeEvolution, build_native_hadamard_wrapper

ROOT = Path(__file__).resolve().parents[1]


def pipeline_worker(unit, blocks, efficient, traced, connection, *, collect_before_stage=True):
    meter = old.StageMeter(connection, trace_python=traced, collect_before_stage=collect_before_stage)
    result = dict(kind='flat_pipeline', mode='efficient' if efficient else 'reference',
        blocks=blocks, traced=traced, observer_collects_before_stage=collect_before_stage,
        whole_compiles=0, axes={})
    original_transpile = old.compiler_module.transpile
    try:
        body = meter.measure('flat_generation', lambda: old.flat_body(unit, blocks, not efficient))
        hasher = streamed_hash if efficient else old.compiler_module.canonical_qiskit_circuit_fingerprint
        fp = meter.measure('native_fingerprint', lambda: hasher(body))
        result.update(instructions=len(body.data), fingerprint=fp,
            native_gate_digest=meter.measure('native_gate_digest', lambda: old.instruction_digest(body)))
        evolution = NativeEvolution(body, 8, .02, 1, 'partial_2nd', 'ordinary', len(body.data), fp)
        current_axis = None
        provider = MatchedNoPrepCostProvider(old.compiler_settings(), memory_efficient=efficient)
        def input_hash(c):
            value = meter.measure(current_axis+'_input_fingerprint', lambda: hasher(c))
            result['axes'][current_axis]['input_fingerprint'] = value
            return value
        def metrics(c):
            values = meter.measure(current_axis+'_compiled_metrics',
                lambda: (measure_compiled_metrics(c) if efficient else measured_metrics(c)))
            digest = meter.measure(current_axis+'_compiled_gate_digest', lambda: old.instruction_digest(c))
            result['axes'][current_axis].update(metrics=values, compiled_gate_digest=digest,
                compiled_global_phase=float(c.global_phase))
            return values
        def compile_engine(*a, **k):
            result['whole_compiles'] += 1
            return meter.measure(current_axis+'_whole_compile', lambda: original_transpile(*a, **k))
        old.compiler_module.transpile = compile_engine
        if efficient:
            provider.cost_cache.fingerprint_function = input_hash
            provider.cost_cache.metric_function = metrics
        else:
            provider.fingerprint_function = input_hash
            provider.compile_function = lambda c, settings, **kw: transpile_and_measure_cost(
                c, settings, metric_function=metrics, **kw)
        def wrapper(value, axis):
            nonlocal current_axis
            current_axis = axis
            result['axes'][axis] = {}
            return meter.measure(axis+'_wrapper_validation_and_copy', lambda: build_native_hadamard_wrapper(
                value, axis=axis, max_instructions=len(body.data)+4, fingerprint_function=hasher))
        start = time.perf_counter()
        pair = provider.compile_pair(evolution, wrapper_builder=wrapper)
        result.update(pair_observed_wall_seconds=time.perf_counter()-start,
            pair_end_memory=old.process_memory(os.getpid()),
            retained_compiled_circuits=sum(cost.transpiled_circuit is not None for cost in pair.values()),
            cache_misses=None if provider.cost_cache is None else provider.cost_cache.miss_count)
        assert result['whole_compiles'] == 2
        assert result['retained_compiled_circuits'] == (0 if efficient else 2)
        for axis, cost in pair.items():
            assert result['axes'][axis]['metrics'] == {k:getattr(cost, k) for k in RPE_COST_METRICS}
            assert cost.actual_circuit_fingerprint == result['axes'][axis]['input_fingerprint']
        result['status'] = 'PASS_ARTIFICIAL_ONLY'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
    finally:
        old.compiler_module.transpile = original_transpile
        connection.send(dict(event='result', result=result)); connection.close()


def request_worker(index, materialized, connection):
    """Q is an artificial request count; no circuit build, compile or RNG."""
    meter = old.StageMeter(connection)
    result = dict(kind='requests_only', Q=1<<index,
        mode='materialized' if materialized else 'stream', circuit_builds=0, whole_compiles=0)
    try:
        _, round_ = artificial_round(index, 2)
        def consume():
            if materialized:
                request = round_.explicit_request(tuple(event_pattern(round_, 'mixed')))
                steps = request.iter_step_requests()
                return len(steps)
            return sum(1 for _ in iter_b2_step_requests(round_, event_pattern(round_, 'mixed')))
        result['step_requests'] = meter.measure('step_request_replay', consume)
        assert result['step_requests'] == round_.q
        result.update(status='PASS_ARTIFICIAL_ONLY', r=round_.r, event_count=round_.q*round_.r)
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
    finally:
        connection.send(dict(event='result', result=result)); connection.close()


def production_worker(unit, blocks, efficient, traced, connection):
    return pipeline_worker(unit, blocks, efficient, traced, connection, collect_before_stage=False)


def run(output, *, production_companion=False):
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip()
    paths = ['scripts/benchmark_b2_memory_pipeline.py','scripts/benchmark_large_artificial_memory.py',
        'src/trotterlib/rte_compiled_cost.py','src/trotterlib/streamed_circuit_fingerprint.py',
        'src/trotterlib/o1_m1_resource_bridge.py','src/trotterlib/b2_p1_compiled_cost.py',
        'src/trotterlib/b2_streamed_native_cost.py','src/trotterlib/b2_rpe_connection.py',
        'src/trottertracks/o1_m1_parallel_price.py']
    hashes = {}
    for path in paths:
        raw = (ROOT/path).read_bytes()
        assert raw == subprocess.check_output(['git','show',source+':'+path], cwd=ROOT), path
        hashes[path] = hashlib.sha256(raw).hexdigest()
    unit, template = old.prepare_unit(old.compiler_settings())
    result = dict(status='RUNNING', source_commit=source, source_sha256=hashes, template=template,
        compiler_hash=template['compiler_hash'], setup_template_compiles=1, rows=[], cases=[], comparisons=[],
        real_H4_calls=0, price_MC=None, strong_PF_price=None, total_RPE_cost=None,
        m17_real_H4_feasibility=None, mandatory_stop=True, production_companion=production_companion,
        runtime=dict(host=platform.node(), python=sys.version, cpus=len(os.sched_getaffinity(0)),
            initial_available_memory_bytes=old.available_memory()[0], load_average=os.getloadavg()),
        protocol='Corrected external observer; stage Python tracing on/off separately; sequential workers for timing isolation; one observation per condition.')
    def save():
        (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        if result['rows']:
            keys = list(dict.fromkeys(k for r in result['rows'] for k in r))
            with (output/'stages.csv').open('w', newline='') as f:
                w=csv.DictWriter(f, fieldnames=keys, lineterminator='\n');w.writeheader();w.writerows(result['rows'])
    ctx=mp.get_context('spawn')
    def execute(function, args, label):
        gc.collect()
        if result['rows']:
            peak=max(r['observed_stage_peak_RSS_KiB'] for r in result['rows'])*1024
            if old.available_memory()[0] < 2*peak:
                raise MemoryError('Insufficient actual memory headroom for next observation')
        parent, child=ctx.Pipe(); process=ctx.Process(target=function,args=(*args,child))
        process.start();child.close()
        rows, case, minimum=old.observe(process,parent,lambda msg:print(label+' '+msg,flush=True));parent.close()
        case.update(label=label,minimum_available_memory_bytes=minimum)
        result['cases'].append(case)
        for row in rows:result['rows'].append(dict(label=label,**row))
        save()
        if case['status']!='PASS_ARTIFICIAL_ONLY':raise RuntimeError(str(case))
        return case
    try:
        if production_companion:
            case=execute(production_worker,(unit,math.ceil(262144/len(unit.data)),True,False),
                'flat_262144_production_no_stage_gc_trace0')
            prior=json.loads((ROOT/'artifacts/b2_memory_pipeline/2026-11-07/run01/result.json').read_text())
            reference=next(c for c in prior['cases'] if c.get('instructions')==case['instructions'] and c.get('mode')=='efficient' and not c['traced'])
            checks=dict(native=case['fingerprint']==reference['fingerprint'],axes=case['axes']==reference['axes'],
                no_observer_gc=not case['observer_collects_before_stage'],no_retained_compiled=case['retained_compiled_circuits']==0)
            result['comparisons'].append(dict(reference_source=prior['source_commit'],checks=checks))
            assert all(checks.values()),checks
            for path,expected in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
            result['status']='PASS_ARTIFICIAL_PIPELINE_ONLY';save();return result
        for traced in (False, True):
            for target in (4096, 65536, 262144):
                blocks=math.ceil(target/len(unit.data))
                candidate=execute(pipeline_worker,(unit,blocks,True,traced),f'flat_{target}_efficient_trace{int(traced)}')
                old_run='run01' if traced else 'run02'
                previous=json.loads((ROOT/'artifacts/large_artificial_memory/2026-11-07'/old_run/'result.json').read_text())
                prior=next((c for c in previous['cases'] if c['mode']=='stream' and c['instructions']==candidate['instructions']),None)
                if prior is not None:
                    checks=dict(native_fingerprint=prior['canonical_fingerprint']==candidate['fingerprint'],
                        cosine_metrics=prior['cosine_metrics']==candidate['axes']['cosine']['metrics'],
                        sine_metrics=prior['sine_metrics']==candidate['axes']['sine']['metrics'])
                    result['comparisons'].append(dict(label=candidate['label'],reference_source=previous['source_commit'],checks=checks))
                    assert all(checks.values()), checks
                if target==4096:
                    reference=execute(pipeline_worker,(unit,blocks,False,traced),f'flat_{target}_reference_trace{int(traced)}')
                    checks=dict(native=reference['fingerprint']==candidate['fingerprint'],axes=reference['axes']==candidate['axes'])
                    result['comparisons'].append(dict(label=candidate['label'],reference='contemporaneous_small_pair',checks=checks))
                    assert all(checks.values()),checks
        for index in (10,17):
            for materialized in (True,False):
                execute(request_worker,(index,materialized),f'Q{1<<index}_requests_'+('materialized' if materialized else 'stream'))
        for path, expected in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
        result['status']='PASS_ARTIFICIAL_PIPELINE_ONLY'
    except Exception as exc:
        result.update(status='STOP',stop_reason=repr(exc))
    save();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--production-companion',action='store_true')
    args=p.parse_args();result=run(args.output,production_companion=args.production_companion);print(result['status'],flush=True)
    sys.exit(0 if result['status']=='PASS_ARTIFICIAL_PIPELINE_ONLY' else 1)
