#!/usr/bin/env python3
"""Fixed saved q4_sample00 only. Prediction and oracle comparison are separate.

No circuit build, transpile, state simulation, sampler or model fit is called.
Original SOURCE files are reused unchanged; instrumentation observes the rules.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT='artifacts/track_d_h4_native_bridge/2026-11-10/run01'
DOC='docs/research/track_d_h4_native_bridge_20261110'
Q4='artifacts/track_d_c6a_paired_price/2026-11-06/run01'
BASE='8b6ee8677d447ddc22d1587acdbe85b343ec7d58'
SCIENCE='6b6667026ec8b670f610e41141d195883a9eb513'
ENDPOINT='a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
LAW='c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5'
PACKET='1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
METRICS=('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size','qubit_count')
WRAPPERS=tuple((method,axis) for method in ('symmetric_directional','reference') for axis in ('X','Y'))
PINNED=('src/trottertracks/track_d_native_emitter.py','src/trottertracks/track_d_native_resources.py',
        'src/trottertracks/track_d_p0_cx_bridge.py','src/trottertracks/track_d_p0_1q_bridge.py',
        'src/trottertracks/track_d_cut_h4_saved.py')
EXEC=('scripts/run_track_d_h4_native_bridge.py','tests/check_track_d_h4_native_bridge.py',DOC+'/PLAN.md')+PINNED
PROFILE=dict(qiskit='1.3.0',optimization_level=1,seed_transpiler=17,
             basis_gates=['rz','sx','x','cx'],approximation_degree=1.,qubits_initially_zero=False)


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def metrics(result):return {k:result['metrics'][k] for k in METRICS}
def peak_mib():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024


def archive_manifest(archive):
    relative=Q4+'/paths/q4_sample00/payload_archive.json'
    data=(ROOT/relative).read_bytes()
    if data!=git('show',BASE+':'+relative):raise ValueError('fixed archive index changed')
    manifest=json.loads(data)
    from trottertracks.track_d_cut_h4_saved import sha as original_sha
    payload=Path(archive).read_bytes()
    if len(payload)!=manifest['bytes'] or original_sha(payload)!=manifest['sha256']:
        raise ValueError('archive identity mismatch')
    if manifest['execution_source']!=SCIENCE:raise ValueError('C6a SOURCE mismatch')
    return manifest


def member(archive,manifest,name):
    binding=next(r for r in manifest['members'] if r['path']==name)
    with tarfile.open(archive,'r:gz') as t:
        names=t.getmembers()
        if len(names)!=20 or {m.name for m in names}!={r['path'] for r in manifest['members']}:
            raise ValueError('fixed archive member set')
        item=t.getmember(name)
        if not item.isfile():raise ValueError('regular saved member required')
        data=t.extractfile(item).read()
    if len(data)!=binding['bytes'] or sha(data)!=binding['sha256']:raise ValueError('member identity:'+name)
    return data


def observed_optimize(word,nq,nc,phase):
    """Same rule implementation; process-local observers only, no source edit."""
    from trottertracks import track_d_p0_1q_bridge as implementation
    original=implementation.CXBoundaryBridge
    original_oneq=implementation.oneq_runs
    timings=[];frontiers=[]
    class ObservedCX(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.max_refs=0;self.max_wire=0
        def feed(self,op):
            result=super().feed(op)
            self.max_refs=max(self.max_refs,sum(map(len,self._frontier)))
            self.max_wire=max(self.max_wire,max(map(len,self._frontier),default=0))
            return result
        def extend(self,ops):
            start=time.perf_counter();result=super().extend(ops)
            frontiers.append(dict(seconds=time.perf_counter()-start,
                peak_live_nodes=self.peak_live_nodes,retained_node_slots=len(self._nodes),
                peak_frontier_references=self.max_refs,peak_single_wire_stack=self.max_wire))
            return result
    def observed_oneq(*a,**kw):
        start=time.perf_counter();value=original_oneq(*a,**kw)
        timings.append(time.perf_counter()-start);return value
    with (patch.object(implementation,'CXBoundaryBridge',ObservedCX),
          patch.object(implementation,'oneq_runs',observed_oneq)):
        result=implementation.optimize_native(iter(word),nq,nc,global_phase=phase)
    return result,dict(oneq_seconds_by_round=timings,cx_rounds=frontiers,
        peak_live_nodes=max(r['peak_live_nodes'] for r in frontiers),
        peak_retained_node_slots=max(r['retained_node_slots'] for r in frontiers),
        peak_frontier_references=max(r['peak_frontier_references'] for r in frontiers),
        peak_single_wire_stack=max(r['peak_single_wire_stack'] for r in frontiers))


def save_word(path,word):
    # Private derived angles/ordered payload. Never add this directory to Git.
    with gzip.open(path,'wt') as f:
        for op in word:f.write(json.dumps([op.name,op.qubits,op.clbits,op.params],separators=(',',':'))+'\n')
    return sha(Path(path).read_bytes())


def predict_worker(task):
    archive,manifest,method,axis=task
    started=time.perf_counter();peaks={'worker_start':peak_mib()}
    import qiskit
    from qiskit import qpy
    from trottertracks.track_d_native_emitter import NativeSink
    from trottertracks.track_d_native_resources import NativeCounter
    if qiskit.__version__!='1.3.0':raise ValueError('fixed Qiskit1.3 required')
    peaks['imports']=peak_mib();name=method+'_'+axis
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('new P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('new P0 forbidden'))):
        start=time.perf_counter()
        data=member(archive,manifest,'circuits/'+name+'_raw.qpy')
        values=qpy.load(io.BytesIO(data))
        if len(values)!=1:raise ValueError('one saved raw wrapper required')
        raw=values[0]
        if raw.num_qubits!=(9 if method=='symmetric_directional' else 17) or raw.num_clbits!=1:
            raise ValueError('fixed saved wrapper width')
        load_time=time.perf_counter()-start;peaks['raw_loaded']=peak_mib()
        start=time.perf_counter();word=[]
        sink=NativeSink(raw.num_qubits,raw.num_clbits,consumer=word.append)
        sink.compose(raw,inplace=True)
        lower_time=time.perf_counter()-start;peaks['lowered']=peak_mib()
        start=time.perf_counter()
        result,observer=observed_optimize(word,raw.num_qubits,raw.num_clbits,sink.global_phase)
        bridge_time=time.perf_counter()-start;peaks['bridge']=peak_mib()
        start=time.perf_counter()
        counter=NativeCounter(raw.num_qubits,raw.num_clbits,capture_transfer=True,
                              global_phase=result['global_phase']).extend(iter(result['operations']))
        counted=metrics(counter.result());template=counter.template()
        if counted!=metrics(template.result()) or counted!=metrics(result['output']):
            raise ValueError('native counter/transfer disagreement')
        count_time=time.perf_counter()-start;peaks['counted']=peak_mib()
        start=time.perf_counter();local=ROOT/OUT/'local'
        payload_hashes={stage:save_word(local/(name+'_'+stage+'.jsonl.gz'),ops)
                        for stage,ops in [('native',word),('bridge',result['operations'])]}
        payload_time=time.perf_counter()-start
        row=dict(method=method,axis=axis,status='PREDICTED_P0_UNQUALIFIED',
            raw_qpy_sha256=sha(data),raw_operations=len(raw.data),native_instructions=len(word),
            native_metrics=metrics(result['input']),predicted_metrics=counted,
            raw_operation_counts=dict(Counter(i.operation.name for i in raw.data)),
            native_counts=dict(Counter(i.name for i in word)),
            optimized_counts=dict(Counter(i.name for i in result['operations'])),
            phases=dict(saved_raw=float(raw.global_phase),native=sink.global_phase,optimized=result['global_phase']),
            rounds=result['rounds'],observer=observer,
            performance=dict(saved_read_and_QPY_load_seconds=load_time,lowering_seconds=lower_time,
                bridge_seconds=bridge_time,count_and_transfer_seconds=count_time,private_payload_write_seconds=payload_time,
                worker_total_seconds=time.perf_counter()-started,peak_RSS_MiB=peak_mib(),peak_RSS_MiB_by_stage=peaks,
                maximum_native_word_container=len(word),
                bridge_includes_its_input_and_output_counting=True,instrumentation_overhead_included=True),
            private_payload_sha256=payload_hashes,unsupported_domains_encountered=[],
            price_or_compiled_used_for_prediction=False,general_P0_qualified=False,
            new_transpile=0,new_circuit_build=0,new_simulation=0)
    write(ROOT/OUT/(name+'_prediction.json'),row)
    return row


def predict(archive):
    start=time.perf_counter();out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    (out/'local').mkdir();(out/'local/.gitignore').write_text('*\n')
    manifest=archive_manifest(archive);source=git('rev-parse','HEAD').decode().strip()
    pins={}
    for p in EXEC:
        data=(ROOT/p).read_bytes()
        if data!=git('show',source+':'+p):raise ValueError('unfixed execution source:'+p)
        if p in PINNED and data!=git('show',BASE+':'+p):raise ValueError('original algorithm modified')
        pins[p]=sha(data)
    from trottertracks.track_d_cut_h4_saved import checked_record
    inputs={}
    for name in ('event_bindings.json','SOURCE.json','paths/q4_sample00/payload_archive.json',
                 'paths/q4_sample00/precompile_checks.json','paths/q4_sample00/paired_observables.json'):
        relative=Q4+'/'+name;data=(ROOT/relative).read_bytes()
        if data!=git('show',BASE+':'+relative):raise ValueError('fixed scientific input changed')
        inputs[relative]=sha(data)
    bindings=read(ROOT/Q4/'event_bindings.json')
    binding=next(r for r in bindings if r['sample_id']=='q4_sample00')
    checked_record(member(archive,manifest,'events.json'),binding,SCIENCE,ENDPOINT,PACKET,LAW)
    # Compile profile files contain no resource oracle. No prices.json is opened here.
    for method,axis in WRAPPERS:
        relative=Q4+'/paths/q4_sample00/'+method+'_'+axis+'_compile.json'
        data=(ROOT/relative).read_bytes()
        if data!=git('show',BASE+':'+relative):raise ValueError('fixed compile record changed')
        inputs[relative]=sha(data)
        record=json.loads(data)
        if record['status']!='PASS' or record['compiler']!=PROFILE:raise ValueError('saved P0 profile mismatch')
    write(out/'SOURCE.json',dict(execution_source=source,base=BASE,files=pins,input_hashes=inputs,C6a_execution_source=SCIENCE,
        archive_sha256=manifest['sha256'],archive_bytes=manifest['bytes'],event_sha256=binding['record_sha256'],
        endpoint=ENDPOINT,law_sha256=LAW,input_packet_sha256=PACKET,q=4,cut=2,C1_head_tail=False,
        compiler_profile=PROFILE,topology_free=True,declared_global_phase_preserved=True,
        raw_payload_public=False,scope='four saved q4_sample00 raw wrappers; prediction before oracle'))
    workers=min(len(WRAPPERS),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(predict_worker,(archive,manifest,m,a)) for m,a in WRAPPERS]
        rows=[]
        for future in as_completed(futures):
            row=future.result();rows.append(row);print('PREDICTION_SAVED',row['method'],row['axis'],flush=True)
    for p,h in pins.items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('SOURCE changed during prediction')
    for p,h in inputs.items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('fixed input changed during prediction')
    if archive_manifest(archive)['sha256']!=manifest['sha256']:raise ValueError('original archive changed')
    write(out/'PREDICTIONS.json',dict(status='PREDICTIONS_FROZEN_BEFORE_ORACLE',execution_source=source,
        workers=workers,wall_seconds=time.perf_counter()-start,
        records=sorted(rows,key=lambda r:(r['method'],r['axis'])),
        new_P0=0,new_H4_build=0,new_simulation=0,oracle_used=False))


def compare(archive,prediction_commit):
    out=ROOT/OUT;frozen=(out/'PREDICTIONS.json').read_bytes()
    if frozen!=git('show',prediction_commit+':'+OUT+'/PREDICTIONS.json'):
        raise ValueError('predictions must be committed unchanged before comparison')
    predictions=json.loads(frozen);manifest=archive_manifest(archive)
    source=read(out/'SOURCE.json')
    for p,h in source['files'].items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('prediction algorithm changed before comparison')
    data=(ROOT/Q4/'prices.json').read_bytes()
    if data!=git('show',BASE+':'+Q4+'/prices.json'):raise ValueError('saved price index changed')
    prices=[r for r in json.loads(data) if r['sample_id']=='q4_sample00']
    if len(prices)!=4:raise ValueError('four saved oracle rows required')
    import qiskit
    from qiskit import qpy
    from trottertracks.track_d_native_resources import count_native,iter_qiskit_native
    rows=[]
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('new P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('new P0 forbidden'))):
        for prediction in predictions['records']:
            m,a=prediction['method'],prediction['axis'];name=m+'_'+a
            row=next(r for r in prices if (r['method'],r['axis'])==(m,a))
            if (row['q']!=4 or row['request']['cut_after_complete_step']!=2 or row['compiler']!=PROFILE
                or row['optimization_policy']!='strong_whole_path_half_sweep_relative_basis_scalar_aggregation_v1'):
                raise ValueError('saved scope/profile mismatch')
            expected={k:row['metrics'][k] for k in METRICS if k!='qubit_count'};expected['qubit_count']=row['peak_qubits']
            raw=member(archive,manifest,'circuits/'+name+'_compiled.qpy')
            circuits=qpy.load(io.BytesIO(raw))
            if len(circuits)!=1:raise ValueError('one compiled oracle required')
            c=circuits[0];recount=metrics(count_native(iter_qiskit_native(c),c.num_qubits,c.num_clbits))
            if recount!=expected:raise ValueError('saved compiled/price index disagreement')
            delta={k:prediction['predicted_metrics'][k]-expected[k] for k in METRICS}
            rows.append(dict(method=m,axis=a,predicted=prediction['predicted_metrics'],saved_P0=expected,
                residual=delta,matched_metrics=sum(v==0 for v in delta.values()),
                classification='FINITE_EXACT_MATCH_P0_UNQUALIFIED' if all(v==0 for v in delta.values()) else 'PARTIAL_MATCH_P0_UNQUALIFIED',
                compiled_qpy_sha256=sha(raw),compiled_counts=dict(c.count_ops()),
                saved_compiled_global_phase=float(c.global_phase),
                phase_difference_mod_2pi=math.remainder(prediction['phases']['optimized']-float(c.global_phase),2*math.pi),
                full_input_correctness_newly_evaluated=False))
    write(out/'COMPARISON.json',dict(prediction_commit=prediction_commit,prediction_sha256=sha(frozen),
        saved_prices_sha256=sha(data),records=rows,all_28_resources_match=all(r['matched_metrics']==7 for r in rows),
        P0_generalized=False,new_P0=0,new_simulation=0))
    write(out/'STOP.json',dict(mandatory_stop=True,status='FIXED_Q4_ANALYSIS_COMPLETE',next_owner='user/GPT',
        prediction_commit=prediction_commit,new_P0=0,new_sampling=0,new_build=0,new_simulation=0,
        new_fit=0,N_or_13_round_updated=False,raw_public=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['predict','compare'])
    p.add_argument('--archive',required=True);p.add_argument('--prediction-commit')
    a=p.parse_args()
    try:
        if a.phase=='predict':predict(a.archive)
        else:
            if not a.prediction_commit:raise ValueError('prediction commit required')
            compare(a.archive,a.prediction_commit)
    except Exception as error:
        out=ROOT/OUT
        if out.is_dir():
            message=str(error).replace(str(ROOT),'<task>').replace(a.archive,'<private-archive>')
            write(out/('FAILED_'+a.phase+'.json'),dict(status='STOP_PARTIAL',error_type=type(error).__name__,error=message))
        raise


if __name__=='__main__':main()
