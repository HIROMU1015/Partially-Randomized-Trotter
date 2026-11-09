#!/usr/bin/env python3
"""Saved C7-2 artificial gates only; no H4 input, P0 transpile, fit or RPE."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.track_d_native_resources import NativeCounter
from trottertracks.track_d_native_emitter import NativeSink,NativeLowerer
from trottertracks.track_d_p0_1q_bridge import optimize_native

TOY='artifacts/track_d_c7_2_toy/2026-11-07/run01'
OUT='artifacts/track_d_native_emitter_development/2026-11-09/run01'
DOC='docs/research/track_d_native_emitter_development_20261109'
METRICS=('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size','qubit_count')
SOURCE_FILES=('scripts/run_track_d_native_emitter_development.py',
    'src/trottertracks/track_d_native_emitter.py','src/trottertracks/track_d_p0_1q_bridge.py',
    'src/trottertracks/track_d_native_resources.py','src/trottertracks/track_d_p0_cx_bridge.py',
    'tests/check_track_d_native_emitter_bridge.py',DOC+'/PLAN.md')


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def run(toy_root,out):
    import qiskit
    from qiskit import qpy,QuantumCircuit
    from qiskit.circuit import Measure
    from qiskit.circuit.library import RZGate,SXGate,XGate,CXGate
    from qiskit.quantum_info import Operator
    import numpy as np
    classes=dict(rz=RZGate,sx=SXGate,x=XGate,cx=CXGate,measure=Measure)
    if qiskit.__version__!='1.3.0':raise ValueError('fixed Qiskit1.3.0 required')
    def native_qc(word,nq,nc,phase):
        # Small artificial correctness oracle only, not the streaming path.
        c=QuantumCircuit(nq,nc);c.global_phase=phase
        for op in word:c.append(classes[op.name](*op.params),list(op.qubits),list(op.clbits))
        return c
    def unitary(c):return Operator(c.remove_final_measurements(inplace=False) if c.num_clbits else c).data
    def check(actual,target):
        delta=unitary(actual)-target
        record=dict(max_entry=float(np.max(abs(delta))),frobenius=float(np.linalg.norm(delta)),
                    component_atol=1e-10,rtol=0,phase_fit=False)
        if not np.isfinite(record['max_entry']) or record['max_entry']>1e-10:raise ValueError('artificial phase/unitary mismatch')
        return record
    with (ROOT/TOY/'case_metrics.csv').open() as f:
        prices={(r['case'],r['stage']):{k:int(r[k]) for k in METRICS} for r in csv.DictReader(f)}
    index=json.loads((ROOT/TOY/'payload_index.json').read_text())
    bindings={r['path']:r['sha256'] for r in index['members']}
    cases=sorted({name for name,stage in prices if stage=='raw'})
    if len(cases)!=54:raise ValueError('fixed54 artificial cases required')
    results=[];lowerer=NativeLowerer()
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('new P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('new P0 forbidden'))):
        for name in cases:
            path=Path(toy_root)/'local'/(name+'_raw.qpy')
            if not path.is_file():
                results.append(dict(case=name,status='HOLD',reason='saved artificial raw QPY unavailable'));continue
            data=path.read_bytes()
            if sha(data)!=bindings[path.name]:raise ValueError('saved artificial payload identity')
            values=qpy.load(io.BytesIO(data))
            if len(values)!=1:raise ValueError('one saved artificial circuit required')
            raw=values[0]
            if raw.num_qubits>5 or raw.num_qubits!=prices[name,'raw']['qubit_count']:
                raise ValueError('fixed small artificial domain only')
            word=[];counter=NativeCounter(raw.num_qubits,raw.num_clbits)
            def consume(op):counter.feed(op);word.append(op)
            sink=NativeSink(raw.num_qubits,raw.num_clbits,consumer=consume,lowerer=lowerer)
            try:sink.compose(raw,inplace=True)
            except ValueError as error:
                results.append(dict(case=name,status='UNSUPPORTED',reason=str(error)));continue
            target=unitary(raw)
            lower_check=check(native_qc(word,raw.num_qubits,raw.num_clbits,sink.global_phase),target)
            result=optimize_native(iter(word),raw.num_qubits,raw.num_clbits,global_phase=sink.global_phase)
            bridge_check=check(native_qc(result['operations'],raw.num_qubits,raw.num_clbits,result['global_phase']),target)
            actual={k:result['output']['metrics'][k] for k in METRICS};expected=prices[name,'compiled']
            row=dict(case=name,status='PASS',saved_raw_sha256=bindings[path.name],
                native_metrics={k:counter.result()['metrics'][k] for k in METRICS},
                bridge_metrics=actual,saved_P0_metrics=expected,
                delta_from_saved_P0={k:actual[k]-expected[k] for k in METRICS},
                all_metrics_match=actual==expected,lowering_correctness=lower_check,
                bridge_correctness=bridge_check,rounds=result['rounds'],
                declared_native_global_phase=sink.global_phase,optimized_global_phase=result['global_phase'],
                p0_qualification='P0_UNQUALIFIED')
            results.append(row);print(name,'P0-all-metrics-match',row['all_metrics_match'],flush=True)
    success=[r for r in results if r['status']=='PASS']
    summary=dict(status='TECHNICAL_COMPLETE_MANDATORY_STOP',cases=54,
        processed=len(success),hold_or_unsupported=54-len(success),
        all_metrics_match=sum(r['all_metrics_match'] for r in success),
        per_metric_matches={k:sum(r['delta_from_saved_P0'][k]==0 for r in success) for k in METRICS},
        max_lowering_component=max((r['lowering_correctness']['max_entry'] for r in success),default=None),
        max_bridge_component=max((r['bridge_correctness']['max_entry'] for r in success),default=None),
        new_P0_transpile=0,new_H4_build_or_reanalysis=0,new_sampling=0,new_fit=0,N_or_RPE_updated=False,
        p0_generalized=False,raw_payload_uploaded=False,results=results)
    write(out/'artificial_results.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--toy-root',required=True);a=p.parse_args()
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    source=git('rev-parse','HEAD').decode().strip();pins={}
    for relative in SOURCE_FILES:
        data=(ROOT/relative).read_bytes()
        if data!=git('show',source+':'+relative):raise ValueError('unfixed execution source')
        pins[relative]=sha(data)
    inputs={}
    for name in ('SOURCE.json','payload_index.json','case_metrics.csv','operation_traces.json','summary.json'):
        path=TOY+'/'+name;data=(ROOT/path).read_bytes()
        if data!=git('show','709d6ed9c234bf6d5a8848d3a30be69d88fd6476:'+path):raise ValueError('fixed artificial index changed')
        inputs[path]=sha(data)
    write(out/'SOURCE.json',dict(execution_source=source,files=pins,input_hashes=inputs,
        base='b49ee3fc153bed786635b352a6ffb89c944b5adb',qiskit='1.3.0',
        scope='saved C7-2 artificial cases only; no P0 or H4 execution',raw_payload_uploaded=False))
    try:
        result=run(a.toy_root,out)
        if not all(sha((ROOT/r).read_bytes())==h for r,h in pins.items()):raise ValueError('execution source changed')
        if not all(sha((ROOT/r).read_bytes())==h for r,h in inputs.items()):raise ValueError('fixed input changed')
        write(out/'STOP.json',dict(status=result['status'],mandatory_stop=True,
            source_input_preserved=True,new_P0_transpile=0,H4_reanalysis=0,next_owner='user/GPT'))
    except Exception as error:
        message=str(error).replace(str(ROOT),'<task>').replace(a.toy_root,'<saved-artificial>')
        write(out/'STOP.json',dict(status='STOP_PARTIAL',mandatory_stop=True,error=message))
        raise


if __name__=='__main__':main()
