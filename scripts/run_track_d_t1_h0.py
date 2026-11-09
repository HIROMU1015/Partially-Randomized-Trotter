#!/usr/bin/env python3
"""One bounded T1/H0 campaign: saved artificial traces, 3 toy P0, 2 H4 paths.

Never sample, build/recompile H4, load NPZ, fit models, or simulate H4 states.
Private input locations are CLI arguments and are not projected into results.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.track_d_native_resources import NativeInstruction, count_native, iter_qiskit_native
from trottertracks.track_d_p0_cx_bridge import CXBoundaryBridge

OUT='artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01'
DOC='docs/research/track_d_t1_h0_p0_bridge_20261109'
TOY='artifacts/track_d_c7_2_toy/2026-11-07/run01'
Q4='artifacts/track_d_c6a_paired_price/2026-11-06/run01'
Q64='artifacts/track_d_c8_1_q64/2026-11-08/run04_limited_completion'
METRICS=('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size','qubit_count')
PROFILE=dict(qiskit='1.3.0',optimization_level=1,seed_transpiler=17,
             basis_gates=['rz','sx','x','cx'],approximation_degree=1.,qubits_initially_zero=False)
EXEC=('scripts/run_track_d_t1_h0.py','src/trottertracks/track_d_p0_cx_bridge.py',
      'src/trottertracks/track_d_native_resources.py','tests/check_track_d_p0_cx_bridge.py',DOC+'/PLAN.md',
      'src/trottertracks/c1_h6_compiler_precision.py','src/trottertracks/pr_dictionary_validation/gates.py',
      'src/trottertracks/pr_dictionary_validation/fixture.py')


def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def metrics(result):return {k:result['metrics'][k] for k in METRICS}
def difference(actual,expected):return {k:actual[k]-expected[k] for k in METRICS}


def phase_a(out):
    directory=ROOT/TOY
    traces=read(directory/'operation_traces.json');summary=read(directory/'summary.json')
    details={r['name']:r for r in summary['case_results']}
    with (directory/'case_metrics.csv').open() as stream:
        expected={(r['case'],r['stage']):{k:int(r[k]) for k in METRICS} for r in csv.DictReader(stream)}
    assert len(traces)==54 and len(expected)==108
    first=['cx_adjacent','cx_commuting_control_rz','cx_noncommuting_target_rz']
    rows=[]
    for name in first+sorted(set(traces)-set(first)):
        result=dict(case=name,stages={},saved_delta=difference(expected[name,'compiled'],expected[name,'raw']))
        for stage in ('raw','compiled'):
            record=traces[name][stage];nq=expected[name,stage]['qubit_count']
            nc=len(details[name][stage]['weighted_wire_finish']['total_depth']['classical_finish'])
            unsupported=sorted({op[0] for op in record['ops'] if op[0] not in ('rz','sx','x','cx','measure')})
            if nc not in (0,1):raise ValueError('TOY_CLASSICAL_MAPPING_NOT_COVERED')
            if unsupported:
                result['stages'][stage]=dict(status='UNSUPPORTED_DOMAIN',opcodes=unsupported,
                                             saved_metrics=expected[name,stage])
                continue
            # Fixed hadamard_wrapper measures ancilla to c0; original SOURCE
            # and saved wire-finish records pin this reconstruction, not guessing.
            def instructions():
                for opcode,qubits,parameters in record['ops']:
                    if opcode=='measure' and (nc!=1 or qubits!=[nq-1]):
                        raise ValueError('TOY_MEASUREMENT_SOURCE_MISMATCH')
                    yield NativeInstruction(opcode,tuple(qubits),(0,) if opcode=='measure' else (),tuple(parameters))
            bridge=CXBoundaryBridge(nq,nc,global_phase=record['global_phase']).extend(instructions())
            value=bridge.result();actual=metrics(value['input'])
            if actual!=expected[name,stage]:raise ValueError('TOY_NATIVE_COUNTER_MISMATCH:'+name+':'+stage)
            reduced=metrics(value['output'])
            result['stages'][stage]=dict(status='PASS',native_metrics=actual,
                bridge_metrics=reduced,removed_cx=value['removed_cx'],
                bridge_vs_saved_P0=difference(reduced,expected[name,'compiled']),
                bridge_matches_saved_P0=reduced==expected[name,'compiled'],
                p0_qualification='P0_UNQUALIFIED')
        rows.append(result)
    report=dict(status='PASS',cases=54,records=rows,
        native_stage_count=sum(r['stages'][s]['status']=='PASS' for r in rows for s in ('raw','compiled')),
        native_raw_count=sum(r['stages']['raw']['status']=='PASS' for r in rows),
        native_compiled_count=sum(r['stages']['compiled']['status']=='PASS' for r in rows),
        new_compile_calls=0)
    write(out/'artificial_trace_comparison.json',report)
    print('Phase A complete:',report['native_raw_count'],'raw,',report['native_compiled_count'],'compiled native records',flush=True)


def phase_d(out):
    import qiskit
    import numpy as np
    from qiskit import QuantumCircuit, qpy
    from qiskit.quantum_info import Operator
    from trottertracks.c1_h6_compiler_precision import compiled_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation.gates import matrix_gate
    if qiskit.__version__!='1.3.0' or json.loads(json.dumps(asdict(CompilerProfile())))!=PROFILE:
        raise ValueError('FIXED_P0_PROFILE')
    words={'full':[(0,1),(1,2),(1,2),(0,1)],'first_chunk':[(0,1),(1,2)],'last_chunk':[(1,2),(0,1)]}
    rows=[]
    for number,(name,word) in enumerate(words.items(),1):
        raw=QuantumCircuit(3)
        for a,b in word:raw.cx(a,b)
        with (out/'local'/(name+'_raw.qpy')).open('xb') as stream:qpy.dump(raw,stream)
        write(out/(name+'_compile_started.json'),dict(call=number,status='STARTED',compiler=PROFILE))
        compiled=compiled_wrapper(raw,CompilerProfile())  # The only new P0 call site.
        with (out/'local'/(name+'_compiled.qpy')).open('xb') as stream:qpy.dump(compiled,stream)
        check=matrix_gate(Operator(compiled).data,Operator(raw).data,1e-10,label=name+':phase_preserving')
        # Existing matrix_gate checks component abs1e-10 with rtol0, no phase fit.
        if not np.isfinite(check['max_entry']):raise ValueError('NONFINITE_TOY_CHECK')
        bridge=CXBoundaryBridge(3).extend(iter_qiskit_native(raw)).result()
        row=dict(case=name,compile_call=number,correctness=check,phase_fit=False,
            native_input=metrics(bridge['input']),bridge_output=metrics(bridge['output']),
            compiled=metrics(count_native(iter_qiskit_native(compiled),3,global_phase=compiled.global_phase)),
            raw_sha256=digest((out/'local'/(name+'_raw.qpy')).read_bytes()),
            compiled_sha256=digest((out/'local'/(name+'_compiled.qpy')).read_bytes()))
        row['bridge_matches_P0']=row['bridge_output']==row['compiled'];rows.append(row)
        write(out/(name+'_compile_result.json'),row)
    report=dict(status='PASS',compile_calls=3,compiler=PROFILE,records=rows,
                independent_chunks_sum={k:rows[1]['compiled'][k]+rows[2]['compiled'][k]
                    for k in METRICS if k!='qubit_count'},
                independent_chunks_width=3,full_word=rows[0]['compiled'])
    write(out/'artificial_new_P0.json',report);print('Phase D complete: 3 fixed P0 calls',flush=True)


def native_parts(raw):
    """Conservative opaque barriers: never cancel across an unsupported gate.

    Local scalar reductions are not a prediction of the full compiled price.
    No raw word or angle is projected into public outputs.
    """
    unknown=Counter();counts=Counter();segments=0;removed=0;bridge=None
    def finish():
        nonlocal segments,removed,bridge
        if bridge is not None:
            removed+=2*bridge.cancelled_pairs;segments+=1;bridge=None
    for item in raw.data:
        view=SimpleNamespace(data=(item,),find_bit=raw.find_bit)
        try:op=next(iter_qiskit_native(view))
        except ValueError:
            unknown[item.operation.name]+=1;finish();continue
        counts[op.name]+=1
        if bridge is None:bridge=CXBoundaryBridge(raw.num_qubits,raw.num_clbits)
        bridge.feed(op)
    finish()
    return dict(native_instruction_counts=dict(counts),unsupported_instruction_counts=dict(unknown),
        native_segments=segments,local_removed_cx=removed,
        full_pre_P0_price_prediction=None,classification='PRE_P0_LOCAL_RULE_ONLY')


def h4_worker(task):
    from qiskit import qpy
    label,arm,axis,location,bindings,expected=task
    name=arm+'_'+axis
    if label=='q4_sample00':
        if not Path(location).is_file():return dict(sample_id=label,arm=arm,axis=axis,status='HOLD',reason='ARCHIVE_UNAVAILABLE')
        with tarfile.open(location,'r:gz') as archive:
            data={s:archive.extractfile('circuits/'+name+'_'+s+'.qpy').read() for s in ('raw','compiled')}
    else:
        paths={s:Path(location)/'local/q64_sample00/circuits'/(name+'_'+s+'.qpy') for s in ('raw','compiled')}
        if any(not p.is_file() for p in paths.values()):return dict(sample_id=label,arm=arm,axis=axis,status='HOLD',reason='SAVED_QPY_UNAVAILABLE')
        data={s:p.read_bytes() for s,p in paths.items()}
    for stage in ('raw','compiled'):
        if digest(data[stage])!=bindings[stage]:raise ValueError('H4_QPY_HASH_MISMATCH:'+label+':'+name+':'+stage)
    compiled_list=qpy.load(io.BytesIO(data['compiled']))
    if len(compiled_list)!=1:raise ValueError('ONE_COMPILED_CIRCUIT_REQUIRED')
    compiled=compiled_list[0]
    bridge=CXBoundaryBridge(compiled.num_qubits,compiled.num_clbits,
                             global_phase=compiled.global_phase).extend(iter_qiskit_native(compiled))
    result=bridge.result();actual=metrics(result['input'])
    if actual!=expected:raise ValueError('H4_SAVED_PRICE_MISMATCH:'+label+':'+name)
    reduced=metrics(result['output'])
    del bridge,compiled,compiled_list
    raw_list=qpy.load(io.BytesIO(data['raw']))
    if len(raw_list)!=1:raise ValueError('ONE_RAW_CIRCUIT_REQUIRED')
    parts=native_parts(raw_list[0])
    return dict(status='PASS',sample_id=label,arm=arm,axis=axis,compiled_metrics=actual,
        saved_metrics=expected,compiled_recount_matches=True,compiled_bridge_metrics=reduced,
        compiled_bridge_removed_cx=result['removed_cx'],compiled_bridge_unchanged=reduced==actual,
        raw_native_parts=parts,input_hashes=bindings,
        p0_price_prediction=False,source_scope='C1_OFF_SAVED_P0',
        classification='COMPILED_RECOUNT_NOT_PRE_P0_PREDICTION')


def phase_e(out,q4_archive,q64_root):
    q4_manifest=read(ROOT/Q4/'paths/q4_sample00/payload_archive.json')
    q4_prices=[r for r in read(ROOT/Q4/'prices.json') if r['sample_id']=='q4_sample00']
    q64_root=Path(q64_root)
    fixed_q64=lambda name:json.loads(git('show','fe361b206a6572142775ac01972545d312e5fb14:'+Q64+'/'+name))
    q64_prices=[r for r in fixed_q64('prices.json') if r['sample_id']=='q64_sample00']
    if len(q4_prices)!=4 or len(q64_prices)!=4:raise ValueError('FIXED_PRICE_ROWS')
    if Path(q4_archive).is_file() and digest(Path(q4_archive).read_bytes())!=q4_manifest['sha256']:
        raise ValueError('Q4_ARCHIVE_IDENTITY')
    q4_bindings={r['path']:r['sha256'] for r in q4_manifest['members']}
    q64_bindings={r['path']:r['sha256'] for r in fixed_q64('payload_index.json')['members']}
    tasks=[]
    for label,prices,location in [('q4_sample00',q4_prices,str(q4_archive)),('q64_sample00',q64_prices,str(q64_root))]:
        for row in prices:
            if row['compiler']!=PROFILE or row['optimization_policy']!='strong_whole_path_half_sweep_relative_basis_scalar_aggregation_v1':
                raise ValueError('H4_P0_OR_C1_POLICY_MISMATCH')
            if row['q']!=(4 if label=='q4_sample00' else 64) or row['request']['cut_after_complete_step']!=row['q']//2:
                raise ValueError('H4_FIXED_PATH_CUT')
            arm,axis=row['method'],row['axis'];name=arm+'_'+axis
            if arm not in ('symmetric_directional','reference') or axis not in ('X','Y'):raise ValueError('H4_METHOD_AXIS')
            prefix='circuits/' if label=='q4_sample00' else 'q64_sample00/circuits/'
            hashes={stage:(q4_bindings if label=='q4_sample00' else q64_bindings)[prefix+name+'_'+stage+'.qpy']
                    for stage in ('raw','compiled')}
            expected={k:row['metrics'][k] for k in METRICS if k!='qubit_count'}
            expected['qubit_count']=row['peak_qubits']
            if 'qubit_count' in row['metrics'] and row['metrics']['qubit_count']!=row['peak_qubits']:
                raise ValueError('H4_SAVED_WIDTH_FIELDS_DISAGREE')
            tasks.append((label,arm,axis,location,hashes,expected))
    rows=[]
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for future in as_completed([pool.submit(h4_worker,t) for t in tasks]):
            row=future.result();rows.append(row)
            write(out/(row['sample_id']+'_'+row['arm']+'_'+row['axis']+'.json'),row)
            print('Phase E',row['sample_id'],row['arm'],row['axis'],row['status'],flush=True)
    report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'PARTIAL_HOLD',
        records=sorted(rows,key=lambda r:(r['sample_id'],r['arm'],r['axis'])),paths=2,wrappers=8,
        new_H4_compile=0,new_H4_simulation=0)
    write(out/'saved_H4_comparison.json',report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q4-archive',required=True);parser.add_argument('--q64-root',required=True)
    args=parser.parse_args();out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    (out/'local').mkdir();(out/'local/.gitignore').write_text('*\n')
    source=git('rev-parse','HEAD').decode().strip();pins={}
    for p in EXEC:
        data=(ROOT/p).read_bytes()
        if data!=git('show',source+':'+p):raise ValueError('UNFIXED_EXECUTION_SOURCE:'+p)
        pins[p]=digest(data)
    inputs={}
    for name in ('SOURCE.json','summary.json','case_metrics.csv','operation_traces.json'):
        p=TOY+'/'+name;data=(ROOT/p).read_bytes()
        if data!=git('show','709d6ed9c234bf6d5a8848d3a30be69d88fd6476:'+p):raise ValueError('FIXED_TOY_SOURCE_INPUT')
        inputs[p]=digest(data)
    for name in ('payload_index.json','prices.json','SOURCE.json'):
        data=git('show','fe361b206a6572142775ac01972545d312e5fb14:'+Q64+'/'+name)
        original=Path(args.q64_root)/name
        if original.exists() and original.read_bytes()!=data:raise ValueError('FIXED_Q64_SOURCE_INPUT')
        inputs[Q64+'/'+name]=digest(data)
    for name in ('prices.json','paths/q4_sample00/payload_archive.json'):
        p=Q4+'/'+name;data=(ROOT/p).read_bytes()
        if data!=git('show','c583abba520a711895d21b7ec9590688349a7bd1:'+p):raise ValueError('FIXED_Q4_INDEX')
        inputs[p]=digest(data)
    write(out/'SOURCE.json',dict(execution_source=source,files=pins,input_hashes=inputs,
        T0_source='c583abba520a711895d21b7ec9590688349a7bd1',compiler=PROFILE,
        scope='saved54 artificial + fixed3 toy P0 + savedH4 q4_sample00/q64_sample00',raw_payload_public=False))
    try:
        phase_a(out);phase_d(out);phase_e(out,args.q4_archive,args.q64_root)
        preserved=all(digest((ROOT/p).read_bytes())==h for p,h in pins.items())
        if not preserved:raise ValueError('EXECUTION_SOURCE_CHANGED')
        write(out/'result.json',dict(status='TECHNICAL_COMPLETE_MANDATORY_STOP',source_preserved=True,
            scientific_conditions_changed=False,new_P0_calls=3,new_H4_compile=0,new_H4_simulation=0,
            new_sampling=0,new_fit=0,new_N_or_RPE_accounting=0))
        write(out/'STOP.json',dict(status='TECHNICAL_COMPLETE_MANDATORY_STOP',mandatory_stop=True,
            next_owner='user/GPT',T1_H0_scope_complete=True))
    except Exception as error:
        # Retain partial evidence and consumed compile-call markers; no retry here.
        message=str(error)
        for p in (str(ROOT),args.q4_archive,args.q64_root):message=message.replace(p,'<private-input>')
        write(out/'STOP.json',dict(status='STOP_PARTIAL',mandatory_stop=True,
            error_type=type(error).__name__,error=message,
            compile_calls_started=len(list(out.glob('*_compile_started.json')))))
        raise


if __name__=='__main__':main()
