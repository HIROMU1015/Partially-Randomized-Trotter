"""Stage RSS pilot on artificial nine-qubit flat circuits, never molecular data.

The parent observes its own spawned worker from /proc every 5 ms. Stage begin/end
handshakes keep returned objects alive for the final RSS sample. Python tracing
and kernel high-water readings supplement, rather than replace, sampled RSS.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import sys
import time
import tracemalloc
import subprocess

import qiskit
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from benchmark_streamed_circuit_fingerprint import synthetic_circuit
from trotterlib.b2_streamed_native_cost import measured_metrics
from trotterlib.rpe_hadamard_interrogation import wrap_controlled_no_prep
from trotterlib.rte import CompilerSettings
import trotterlib.rte_compiled_cost as compiler_module
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint as streamed_hash

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = .005


def compiler_settings():
    if qiskit.__version__ != '1.3.0':
        raise ValueError('Qiskit 1.3.0 required')
    return CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,qiskit.__version__)


def compile_only(circuit, settings):
    """Same compiler engine/settings as transpile_and_measure_cost, no hash work.

    This is a conversion/workspace diagnostic, not a TranspiledCircuitCost with
    a fabricated identity. Fingerprints are measured separately, never spoofed.
    """
    compiler_module._validate_transpile_context(settings,None)
    return compiler_module.transpile(circuit,basis_gates=list(settings.basis_gates),
        optimization_level=settings.optimization_level,seed_transpiler=settings.transpiler_seed)


def prepare_unit(settings):
    original = synthetic_circuit('controlled',1)
    body = QuantumCircuit(3,name='artificial_controlled_unit')
    # Reuse the four controlled primitives; exclude the helper's final H/measure
    # and its unrelated global phase. These gates each preserve control-0 I.
    for item in original.data[:4]:
        body.append(item.operation,[original.find_bit(q).index for q in item.qubits])
    cost = compiler_module.transpile_and_measure_cost(body,settings,
        actual_circuit_fingerprint=streamed_hash(body),backend=None)
    unit = cost.transpiled_circuit
    if unit.num_clbits or set(unit.count_ops())-set(settings.basis_gates):
        raise ValueError('Flat gate-only compiled unit required')
    return unit,dict(native_instructions=len(body.data),flat_instructions=len(unit.data),
        compiler_hash=cost.compiler_settings_hash,compiled_fingerprint=streamed_hash(unit),
        global_phase=float(unit.global_phase),gate_counts=dict(unit.count_ops()))


def tail_iterator(unit, blocks):
    for i in range(blocks):
        tail = QuantumCircuit(9,name='artificial_tail')
        tail.compose(unit,qubits=(i%8,(i+1)%8,8),inplace=True)
        yield tail


def flat_body(unit, blocks, materialize):
    tails = tail_iterator(unit,blocks)
    if materialize:
        tails = list(tails)
    body = QuantumCircuit(9,name='artificial_long_flat')
    for tail in tails:
        body.compose(tail,inplace=True)
    return body


def instruction_digest(circuit):
    """Bounded-memory input check for standard rz/sx/x/cx rows, not cache policy.

    This cheap diagnostic does not stand in for the recursive canonical hash.
    Compiler probes call Qiskit's engine directly and use no placeholder hashes.
    """
    h=hashlib.sha256()
    h.update(json.dumps([circuit.num_qubits,circuit.num_clbits,float(circuit.global_phase).hex()]).encode())
    for item in circuit.data:
        op=item.operation
        row=[f'{type(op).__module__}.{type(op).__qualname__}',op.name,
             [compiler_module._numeric_parameter_payload(p) for p in op.params],
             [circuit.find_bit(q).index for q in item.qubits],
             [circuit.find_bit(c).index for c in item.clbits]]
        h.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode());h.update(b'\n')
    return h.hexdigest()


def process_memory(pid):
    values={}
    for line in (Path('/proc')/str(pid)/'status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:','VmSwap:')):
            key,value,*_=line.split();values[key[:-1]]=int(value)
    return values


def available_memory():
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                       if l.startswith('MemAvailable:')))*1024
    limits=[]
    cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    while cg!=Path('/sys/fs'):
        maximum,current=cg/'memory.max',cg/'memory.current'
        if maximum.exists() and current.exists():
            text=maximum.read_text().strip()
            if text!='max':
                headroom=max(0,int(text)-int(current.read_text()))
                available=min(available,headroom);limits.append(dict(path=str(cg),headroom_bytes=headroom))
        if cg==Path('/sys/fs/cgroup'):break
        cg=cg.parent
    return available,limits


class StageMeter:
    def __init__(self,connection,trace_python=True):
        self.connection=connection
        self.trace_python=trace_python

    def measure(self,name,function):
        gc.collect()
        self.connection.send({'event':'begin','stage':name,'memory':process_memory(os.getpid())})
        if self.connection.recv()!='start':raise RuntimeError('Observer handshake')
        if self.trace_python:tracemalloc.start()
        start=time.perf_counter();error=None
        try:
            value=function()
        except BaseException as exc:
            error=exc
        elapsed=time.perf_counter()-start
        current,peak=tracemalloc.get_traced_memory() if self.trace_python else (None,None)
        if self.trace_python:tracemalloc.stop()
        self.connection.send({'event':'end','stage':name,'elapsed_seconds':elapsed,
            'seconds_with_tracemalloc':elapsed if self.trace_python else None,'tracemalloc_enabled':self.trace_python,
            'Python_traced_peak_bytes':peak,'Python_traced_retained_bytes':current,
            'memory':process_memory(os.getpid()),'ru_maxrss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'error':None if error is None else repr(error)})
        if self.connection.recv()!='recorded':raise RuntimeError('Observer completion handshake')
        if error is not None:raise error
        return value


def worker(unit,blocks,mode,settings,connection,trace_python=True):
    meter=StageMeter(connection,trace_python)
    result=dict(blocks=blocks,mode=mode,template_compiles=0,whole_compiles=0)
    try:
        body=meter.measure('flat_generation',lambda:flat_body(unit,blocks,mode=='materialized'))
        result.update(instructions=len(body.data),qubits=body.num_qubits,
            global_phase=float(body.global_phase),gate_counts=dict(body.count_ops()))
        result['instruction_digest']=meter.measure('instruction_digest',lambda:instruction_digest(body))
        if mode=='stream':
            copied=meter.measure('native_copy',body.copy)
            del copied;gc.collect()
            wrappers={};compiled={}
            for axis in ('cosine','sine'):
                wrappers[axis]=meter.measure(axis+'_wrapper',lambda:wrap_controlled_no_prep(
                    body,ancilla_qubit=8,axis=axis,include_measurement=True))
                dag=meter.measure(axis+'_input_DAG',lambda:circuit_to_dag(wrappers[axis],copy_operations=True))
                result[axis+'_input_DAG_operations']=dag.size();del dag;gc.collect()
                result['whole_compiles']+=1
                compiled[axis]=meter.measure(axis+'_whole_compile',lambda:compile_only(wrappers[axis],settings))
                result[axis+'_metrics']=meter.measure(axis+'_compiled_metrics',lambda:measured_metrics(compiled[axis]))
                dag=meter.measure(axis+'_compiled_DAG',lambda:circuit_to_dag(compiled[axis],copy_operations=True))
                result[axis+'_compiled_DAG_operations']=dag.size();del dag;gc.collect()
            result['pipeline_end_memory']=process_memory(os.getpid())
            result['pipeline_end_retention']='native + two wrappers + two compiled circuits; diagnostic copy/DAGs released'
            del wrappers,compiled;gc.collect()
            # Keep expensive hash diagnostics after the compiler/Rust probes so
            # the legacy payload's allocator history does not pollute them.
            new=meter.measure('streamed_fingerprint',lambda:streamed_hash(body))
            old=meter.measure('legacy_fingerprint',lambda:compiler_module.canonical_qiskit_circuit_fingerprint(body))
            if old!=new:raise ValueError('Canonical fingerprint disagreement')
            result['canonical_fingerprint']=new
        result['status']='PASS_ARTIFICIAL_ONLY'
    except BaseException as exc:
        result.update(status='INCONCLUSIVE',stop_reason=repr(exc))
    finally:
        if tracemalloc.is_tracing():tracemalloc.stop()
        connection.send({'event':'result','result':result});connection.close()


def observe(process,connection,progress=None):
    """Observe only the worker spawned by this parent; no shared settings writes."""
    rows=[];active=None;result=None;heartbeat=time.perf_counter();last_progress=heartbeat
    available,limits=available_memory();minimum_available=available
    while process.is_alive() or connection.poll():
        try:memory=process_memory(process.pid)
        except (FileNotFoundError,ProcessLookupError):memory=None
        if memory is not None and 'VmRSS' not in memory:memory=None
        now=time.perf_counter()
        if active is not None and memory is not None:
            active['samples']+=1
            active['sampled_peak_RSS_KiB']=max(active['sampled_peak_RSS_KiB'],memory['VmRSS'])
            active['maximum_sample_gap_seconds']=max(active['maximum_sample_gap_seconds'],now-active['_last_sample_at'])
            active['_last_sample_at']=now
        if now-heartbeat>=.1:
            available,limits=available_memory();minimum_available=min(minimum_available,available)
            # No fixed RAM cap: stop only if actual remaining headroom cannot
            # cover another increment as large as this stage has already added.
            growth=0 if active is None or memory is None else max(0,memory['VmRSS']-active['RSS_before_KiB'])*1024
            if growth and available<growth:
                process.terminate();process.join(5)
                if process.is_alive():process.kill();process.join()
                return rows,dict(status='RESOURCE_STOP',stop_reason='available memory below observed stage growth',
                    available_bytes=available,observed_growth_bytes=growth),minimum_available
            heartbeat=now
        if active is not None and progress and now-last_progress>=30:
            progress('running '+active['stage']+' RSS KiB '+str(None if memory is None else memory['VmRSS'])+
                     ' available GiB '+str(round(available/(1024**3),1)))
            last_progress=now
        if connection.poll(INTERVAL):
            try:event=connection.recv()
            except EOFError:break
            if event['event']=='begin':
                if active is not None:raise RuntimeError('Overlapping stages')
                m=process_memory(process.pid)
                sampled_at=time.perf_counter()
                active=dict(stage=event['stage'],RSS_before_KiB=m['VmRSS'],HWM_before_KiB=m['VmHWM'],
                    sampled_peak_RSS_KiB=m['VmRSS'],samples=1,maximum_sample_gap_seconds=0.,
                    requested_interval_seconds=INTERVAL,_first_sample_at=sampled_at,_last_sample_at=sampled_at)
                connection.send('start')
                if progress:progress('begin '+event['stage'])
            elif event['event']=='end':
                if active is None or event['stage']!=active['stage']:raise RuntimeError('Stage mismatch')
                worker_memory=event.pop('memory');event.pop('event');event.pop('stage')
                m=process_memory(process.pid);sampled_at=time.perf_counter()
                active['samples']+=1
                active['maximum_sample_gap_seconds']=max(active['maximum_sample_gap_seconds'],sampled_at-active['_last_sample_at'])
                active['observed_stage_wall_seconds']=sampled_at-active.pop('_first_sample_at')
                active.pop('_last_sample_at')
                m['VmHWM']=max(m['VmHWM'],worker_memory['VmHWM'])
                active.update(event,RSS_after_KiB=m['VmRSS'],HWM_after_KiB=m['VmHWM'])
                active['sampled_peak_RSS_KiB']=max(active['sampled_peak_RSS_KiB'],m['VmRSS'])
                active['kernel_HWM_increased']=m['VmHWM']>active['HWM_before_KiB']
                active['observed_stage_peak_RSS_KiB']=max(active['sampled_peak_RSS_KiB'],
                    m['VmHWM'] if active['kernel_HWM_increased'] else 0)
                rows.append(active)
                if progress:progress('end '+active['stage']+' RSS KiB '+str(active['observed_stage_peak_RSS_KiB']))
                active=None;connection.send('recorded')
            elif event['event']=='result':result=event['result']
    process.join()
    if result is None:result=dict(status='INCONCLUSIVE',stop_reason='worker exited without result',exitcode=process.exitcode)
    if active is not None:result.update(status='INCONCLUSIVE',unfinished_stage=active['stage'])
    return rows,result,minimum_available


def run(output,targets,*,rss_only=False):
    if not targets or targets!=sorted(set(targets)) or min(targets)<1:
        raise ValueError('Positive increasing workload sizes required')
    output.mkdir(parents=True,exist_ok=False)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    paths=['scripts/benchmark_large_artificial_memory.py','scripts/benchmark_streamed_circuit_fingerprint.py',
        'src/trotterlib/streamed_circuit_fingerprint.py','src/trotterlib/rte_compiled_cost.py',
        'src/trotterlib/rpe_hadamard_interrogation.py','src/trotterlib/b2_streamed_native_cost.py']
    hashes={}
    for path in paths:
        raw=(ROOT/path).read_bytes()
        if raw!=subprocess.check_output(['git','show',source+':'+path],cwd=ROOT):raise ValueError('Committed SOURCE required')
        hashes[path]=hashlib.sha256(raw).hexdigest()
    settings=compiler_settings();unit,unit_record=prepare_unit(settings)
    result=dict(status='RUNNING',source_commit=source,source_sha256=hashes,compiler=asdict(settings),
        compiler_hash=compiler_module.compiler_settings_hash(settings),template=unit_record,setup_template_compiles=1,
        target_instructions=targets,rows=[],cases=[],comparisons=[],real_H4_calls=0,price_MC=None,total_RPE_cost=None,
        python_tracing=not rss_only,rss_only=rss_only,
        m17=dict(Q=131072,real_H4_measured=False,feasibility=None),mandatory_stop=True,
        runtime=dict(host=platform.node(),python=sys.version,qiskit=qiskit.__version__,available_cpus=len(os.sched_getaffinity(0)),
            initial_available_memory_bytes=available_memory()[0],finite_cgroup_headrooms=available_memory()[1],load_average=os.getloadavg()),
        protocol='spawned independent workers; parent /proc RSS observer; nominal 5ms sampling; stage handshakes; tracing flag recorded; one observation per condition')
    def save():
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if result['rows']:
            with (output/'stages.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
    ctx=mp.get_context('spawn')
    try:
        for target in targets:
            blocks=math.ceil(target/len(unit.data));pair={}
            for mode in (('stream',) if rss_only else ('materialized','stream')):
                available,_=available_memory()
                if result['cases']:
                    previous=result['cases'][-1]
                    previous_rows=[r for r in result['rows'] if r['instructions']==previous['instructions']]
                    forecast=max(r['observed_stage_peak_RSS_KiB'] for r in previous_rows)*1024*(blocks*len(unit.data)/previous['instructions'])
                    if forecast>available:
                        raise MemoryError('Next workload forecast exceeds actual available memory')
                parent,child=ctx.Pipe();process=ctx.Process(target=worker,args=(unit,blocks,mode,settings,child,not rss_only))
                process.start();child.close()
                rows,case,min_available=observe(process,parent,
                    lambda msg:print(str(target)+' '+mode+' '+msg,flush=True))
                parent.close()
                case.update(target_instructions=target,minimum_available_memory_bytes=min_available)
                result['cases'].append(case)
                for row in rows:result['rows'].append(dict(target_instructions=target,blocks=blocks,
                    instructions=blocks*len(unit.data),mode=mode,**row))
                save()
                if case['status']!='PASS_ARTIFICIAL_ONLY':raise RuntimeError(str(case))
                pair[mode]=case
            if 'materialized' in pair:
                checks={name:pair['materialized'][name]==pair['stream'][name]
                        for name in ('instructions','qubits','global_phase','gate_counts','instruction_digest')}
                result['comparisons'].append(dict(target_instructions=target,checks=checks))
                if not all(checks.values()):raise ValueError('Generated flat circuits differ')
            save()
        for path,sha in hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('SOURCE changed')
        result['status']='PASS_ARTIFICIAL_MEMORY_STAGES'
    except Exception as exc:
        result.update(status='STOP',stop_reason=repr(exc))
    save();return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-instructions',type=int,nargs='+',default=[4096,65536,262144])
    parser.add_argument('--rss-only',action='store_true',help='stream-only companion without Python tracing')
    args=parser.parse_args();result=run(args.output,args.target_instructions,rss_only=args.rss_only)
    print(result['status'],flush=True)
    sys.exit(0 if result['status']=='PASS_ARTIFICIAL_MEMORY_STAGES' else 1)
