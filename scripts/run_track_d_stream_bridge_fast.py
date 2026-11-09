#!/usr/bin/env python3
"""Paired old/stream bridge on saved q4/q8 only; no P0 or state simulation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from run_track_d_h4_native_bridge import ROOT,WRAPPERS,METRICS,read,write,sha,git,metrics,peak_mib,archive_manifest,member
from run_track_d_phase_q8_transfer import selected_member,Q8
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.track_d_native_resources import NativeInstruction,NativeCounter
from trottertracks.track_d_stream_bridge_fast import optimize_native_stream,ObservedCX,iter_raw_native
from trottertracks.track_d_native_phase160 import PhaseLedger
from trottertracks import track_d_p0_1q_bridge as old

BASE='817a1cd54da11ac20dc5a84477bf0f742a567214'
OUT='artifacts/track_d_stream_bridge_fast/2026-11-10/run01'
DOC='docs/research/track_d_stream_bridge_fast_20261110'
Q4='artifacts/track_d_h4_native_bridge/2026-11-10/run01'
PHASE='artifacts/track_d_phase_q8_transfer/2026-11-10/run01'
PINNED=('src/trottertracks/track_d_native_emitter.py','src/trottertracks/track_d_native_resources.py',
 'src/trottertracks/track_d_p0_1q_bridge.py','src/trottertracks/track_d_p0_cx_bridge.py',
 'src/trottertracks/track_d_native_phase160.py','scripts/run_track_d_h4_native_bridge.py')
EXEC=PINNED+('src/trottertracks/track_d_stream_bridge_fast.py','scripts/run_track_d_stream_bridge_fast.py',
 'tests/check_track_d_stream_bridge_fast.py',DOC+'/PLAN.md')


def canonical(op):return json.dumps([op.name,op.qubits,op.clbits,op.params],separators=(',',':'))+'\n'
def word_hash(word):
    h=hashlib.sha256()
    for op in word:h.update(canonical(op).encode())
    return h.hexdigest()
def compressed_word_hash(path):
    with gzip.open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def baseline(operations,nq,nc,phase):
    previous=old.oneq_runs;round_stats=[];additions=[];cx_stats=[]
    def oneq(ops,width):
        # list_iterator reduction exposes the original list without copying it.
        source=ops.__reduce__()[1][0]
        start=time.perf_counter();result=previous(ops,width)
        round_stats.append(dict(seconds=time.perf_counter()-start,input_size=len(source),
            input_list_bytes=source.__sizeof__(),folded_size=len(result[0]),folded_list_bytes=result[0].__sizeof__()))
        additions.append(result[1]);return result
    class CX(ObservedCX):
        def extend(self,ops):
            start=time.perf_counter();result=super().extend(ops)
            cx_stats.append(dict(seconds=time.perf_counter()-start,**self.statistics()));return result
    with patch.object(old,'oneq_runs',oneq),patch.object(old,'CXBoundaryBridge',CX):
        result=old.optimize_native(operations,nq,nc,global_phase=phase)
    result['phase_additions']=additions
    result['performance']=dict(oneq_rounds=round_stats,cx_rounds=cx_stats,
        peak_word_length=max((r['input_size'] for r in round_stats),default=0),
        peak_word_list_bytes=max((r['input_list_bytes'] for r in round_stats),default=0),
        full_folded_word_materialized=True,original_input_word_materialized=True)
    return result


def prepare_worker(task):
    q,method,axis,q4_archive,q8_root=task;start=time.perf_counter();name=method+'_'+axis
    import qiskit
    from qiskit import qpy
    if qiskit.__version__!='1.3.0':raise ValueError('fixed Qiskit1.3.0')
    selection=read(ROOT/PHASE/'SELECTION.json')
    expected=(read(ROOT/Q4/(name+'_prediction.json')) if q==4 else read(ROOT/PHASE/('q8_prediction_'+name+'.json')))
    golden=ROOT/(Q4 if q==4 else PHASE)/'local'/(name+'_bridge.jsonl.gz')
    golden_hash=expected['private_payload_sha256']['bridge'] if q==4 else expected['private_word_sha256']
    if sha(golden.read_bytes())!=golden_hash:
        raise ValueError('old output identity')
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
        data=(member(q4_archive,archive_manifest(q4_archive),'circuits/'+name+'_raw.qpy') if q==4
              else selected_member(q8_root,selection,'circuits/'+name+'_raw.qpy'))
        if sha(data)!=expected['raw_qpy_sha256']:raise ValueError('fixed raw bytes')
        c=qpy.load(io.BytesIO(data))[0];ledger=PhaseLedger([float(c.global_phase)])
        statistics={};counter=NativeCounter(c.num_qubits,c.num_clbits)
        key='q'+str(q)+'_'+name;path=ROOT/OUT/'local'/(key+'_native.jsonl.gz');h=hashlib.sha256()
        with gzip.open(path,'wt') as f:
            for op in iter_raw_native(c,ledger,statistics):
                counter.feed(op);line=canonical(op);h.update(line.encode());f.write(line)
        if q==4:
            oldnative=ROOT/Q4/'local'/(name+'_native.jsonl.gz')
            if sha(oldnative.read_bytes())!=expected['private_payload_sha256']['native'] or compressed_word_hash(oldnative)!=h.hexdigest():
                raise ValueError('bounded emitter changed q4 native word')
        prior_phase=read(ROOT/PHASE/('q4_phase_'+name+'.json' if q==4 else 'q8_prediction_'+name+'.json'))
        if ledger.result()!=prior_phase['native_phase']:raise ValueError('native phase changed')
        phase_path=ROOT/OUT/'local'/(key+'_phase.json.gz')
        with gzip.open(phase_path,'wt') as f:json.dump(ledger.values,f)
        return dict(case=key,q=q,method=method,axis=axis,nq=c.num_qubits,nc=c.num_clbits,
            raw_qpy_sha256=sha(data),native_gzip_sha256=sha(path.read_bytes()),native_word_sha256=h.hexdigest(),
            native_metrics=metrics(counter.result()),native_phase=ledger.result(),
            phase_input_sha256=sha(phase_path.read_bytes()),expected_metrics=expected['predicted_metrics'],
            expected_rounds=expected['rounds'],expected_output_word_sha256=compressed_word_hash(golden),
            expected_legacy_phase=(expected['phases']['optimized'] if q==4 else expected['original_bridge_phase']),
            expected_precise_phase=prior_phase['final_phase'],statistics=statistics,
            preparation_seconds=time.perf_counter()-start,preparation_peak_RSS_MiB=peak_mib())


def prepare(q4_archive,q8_root):
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    (out/'local').mkdir(mode=0o700);(out/'local/.gitignore').write_text('*\n')
    source=git('rev-parse','HEAD').decode().strip();pins={}
    for p in EXEC:
        data=(ROOT/p).read_bytes()
        if data!=git('show',source+':'+p):raise ValueError('unfixed execution SOURCE')
        if p in PINNED and data!=git('show',BASE+':'+p):raise ValueError('protected old source changed')
        pins[p]=sha(data)
    write(out/'SOURCE.json',dict(execution_source=source,base=BASE,files=pins,
        scope='fixed saved q4/q8 eight wrappers, exact old/new bridge performance',raw_public=False))
    tasks=[(q,m,a,q4_archive,q8_root) for q in (4,8) for m,a in WRAPPERS]
    rows=[];start=time.perf_counter();workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(prepare_worker,t) for t in tasks]):
            row=f.result();rows.append(row);print('PREPARED',row['case'],flush=True)
    write(out/'INPUTS.json',dict(execution_source=source,workers=workers,wall_seconds=time.perf_counter()-start,
        records=sorted(rows,key=lambda r:r['case']),new_sampling=0,new_P0=0,new_whole_path=0))


def benchmark_worker(task):
    row,variant=task;import qiskit
    import numpy
    from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import Optimize1qGatesDecomposition
    path=ROOT/OUT/'local'/(row['case']+'_native.jsonl.gz')
    phase_path=ROOT/OUT/'local'/(row['case']+'_phase.json.gz')
    if sha(path.read_bytes())!=row['native_gzip_sha256'] or sha(phase_path.read_bytes())!=row['phase_input_sha256']:
        raise ValueError('fixed benchmark input')
    with gzip.open(phase_path,'rt') as f:ledger=PhaseLedger(json.load(f))
    read_time=0.;decoded=0
    def source():
        nonlocal read_time,decoded
        with gzip.open(path,'rt') as f:
            while True:
                start=time.perf_counter();line=f.readline()
                if not line:read_time+=time.perf_counter()-start;break
                name,q,c,p=json.loads(line);op=NativeInstruction(name,tuple(q),tuple(c),tuple(p))
                read_time+=time.perf_counter()-start;decoded+=1;yield op
    start=time.perf_counter()
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
        function=baseline if variant=='old' else optimize_native_stream
        result=function(source(),row['nq'],row['nc'],global_phase=ledger.running)
    duration=time.perf_counter()-start;peak=peak_mib()
    final=PhaseLedger(ledger.values)
    for addition in result['phase_additions']:final.add(addition)
    checks=dict(output_word=word_hash(result['operations'])==row['expected_output_word_sha256'],
        all_7_metrics=metrics(result['output'])==row['expected_metrics'],rounds=result['rounds']==row['expected_rounds'],
        legacy_phase=result['global_phase'].hex()==row['expected_legacy_phase'].hex(),
        precise_phase=final.result()==row['expected_precise_phase'],input_count=decoded==row['native_metrics']['circuit_size'])
    if not all(checks.values()):raise ValueError('exact differential mismatch:'+row['case']+':'+variant+':'+str(checks))
    record=dict(case=row['case'],variant=variant,status='PASS',checks=checks,metrics=metrics(result['output']),
        output_word_sha256=word_hash(result['operations']),global_phase=result['global_phase'],
        precise_phase=final.result(),phase_additions=result['phase_additions'],rounds=result['rounds'],
        performance=dict(total_seconds=duration,input_decode_seconds=read_time,
            compute_minus_decode_seconds=duration-read_time,peak_RSS_MiB=peak,decoded=decoded,
            internals=result['performance']),general_P0_qualified=False)
    write(ROOT/OUT/(row['case']+'_'+variant+'.json'),record);return record


def benchmark():
    out=ROOT/OUT;inputs=(out/'INPUTS.json').read_bytes();commit=git('rev-parse','HEAD').decode().strip()
    if inputs!=git('show',commit+':'+OUT+'/INPUTS.json'):raise ValueError('inputs must be committed first')
    pins=read(out/'SOURCE.json')['files']
    for p,h in pins.items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('execution SOURCE changed')
    rows=json.loads(inputs)['records'];tasks=[(r,v) for r in rows for v in ('old','fast')]
    start=time.perf_counter();records=[];workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(benchmark_worker,t) for t in tasks]):
            row=f.result();records.append(row);print('BENCHMARK',row['case'],row['variant'],'PASS',flush=True)
    write(out/'RESULT.json',dict(status='ALL_EXACT_PASS',input_commit=commit,workers=workers,
        wall_seconds=time.perf_counter()-start,records=sorted(records,key=lambda r:(r['case'],r['variant'])),
        new_P0=0,new_sampling=0,new_simulation=0,q16_or_above=False))
    write(out/'STOP.json',dict(mandatory_stop=True,status='FIXED_Q4_Q8_BENCHMARK_COMPLETE',next_owner='user/GPT'))


def artificial(toy_root):
    from qiskit import qpy
    from trottertracks.track_d_native_emitter import NativeSink
    directory=ROOT/'artifacts/track_d_c7_2_toy/2026-11-07/run01'
    index=read(directory/'payload_index.json');binding={r['path']:r['sha256'] for r in index['members']}
    oldresults=read(ROOT/'artifacts/track_d_native_emitter_development/2026-11-09/run01/artificial_results.json')
    import qiskit
    records=[]
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
        for row in oldresults['results']:
            name=row['case'];data=(Path(toy_root)/'local'/(name+'_raw.qpy')).read_bytes()
            if sha(data)!=binding[name+'_raw.qpy']:raise ValueError('fixed artificial raw')
            c=qpy.load(io.BytesIO(data))[0];word=[];sink=NativeSink(c.num_qubits,c.num_clbits,consumer=word.append)
            sink.compose(c,inplace=True)
            before=baseline(iter(word),c.num_qubits,c.num_clbits,sink.global_phase)
            after=optimize_native_stream(iter(word),c.num_qubits,c.num_clbits,global_phase=sink.global_phase)
            passed=(word_hash(before['operations'])==word_hash(after['operations']) and before['rounds']==after['rounds']
                and before['global_phase'].hex()==after['global_phase'].hex()
                and before['phase_additions']==after['phase_additions']
                and metrics(after['output'])==row['saved_P0_metrics'])
            if not passed:raise ValueError('artificial exact mismatch:'+name)
            records.append(dict(case=name,status='PASS',all_7_match_P0=True,word_and_phase_exact=True))
    write(ROOT/'artifacts/track_d_stream_bridge_fast/2026-11-10/technical/artificial54.json',dict(records=records,new_P0=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','benchmark','artificial'])
    p.add_argument('--q4-archive');p.add_argument('--q8-root');p.add_argument('--toy-root');a=p.parse_args()
    try:
        if a.phase=='prepare':
            if not a.q4_archive or not a.q8_root:raise ValueError('fixed saved inputs required')
            prepare(a.q4_archive,a.q8_root)
        elif a.phase=='benchmark':benchmark()
        else:artificial(a.toy_root)
    except Exception as error:
        out=ROOT/OUT
        if out.exists():
            message=str(error).replace(str(ROOT),'<task>')
            for location in (a.q4_archive,a.q8_root,a.toy_root):
                if location:message=message.replace(location,'<private-input>')
            write(out/('FAILED_'+a.phase+'.json'),dict(status='STOP_PARTIAL',error_type=type(error).__name__,error=message))
        raise


if __name__=='__main__':main()
