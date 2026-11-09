#!/usr/bin/env python3
"""Saved q4 phase accounting and one ID-ordered saved q8 price prediction."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from unittest.mock import patch

from run_track_d_h4_native_bridge import (ROOT,WRAPPERS,METRICS,PROFILE,ENDPOINT,LAW,PACKET,
    read,write,sha,git,metrics,peak_mib,archive_manifest,member,save_word,observed_optimize)
import sys
sys.path.insert(0,str(ROOT/'src'))

BASE='1d03ba9cbd04ff68ded88ec33aac3ae7553d4c46'
Q8='artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion'
OUT='artifacts/track_d_phase_q8_transfer/2026-11-10/run01'
DOC='docs/research/track_d_phase_q8_transfer_20261110'
ORIGIN='23725fc6f6d6d0c2bec93e31e32074711804e250'
OLD='artifacts/track_d_h4_native_bridge/2026-11-10/run01'
PINNED=('src/trottertracks/track_d_native_emitter.py','src/trottertracks/track_d_native_resources.py',
 'src/trottertracks/track_d_p0_1q_bridge.py','src/trottertracks/track_d_p0_cx_bridge.py',
 'scripts/run_track_d_h4_native_bridge.py','src/trottertracks/c1_h6_compiler_precision.py')
EXEC=PINNED+('scripts/run_track_d_phase_q8_transfer.py','src/trottertracks/track_d_native_phase160.py',
 'tests/check_track_d_native_phase160.py',DOC+'/PLAN.md')


def fixed(relative):
    data=(ROOT/relative).read_bytes()
    if data!=git('show',BASE+':'+relative):raise ValueError('fixed public input changed:'+relative)
    return data


def selected_member(root,selection,name):
    binding=next(r for r in selection['members'] if r['path']==name)
    relative='events/'+selection['sample_id']+'.json' if name=='events.json' else selection['sample_id']+'/'+name
    for run in ('run01_accelerated_completion','run01'):
        p=Path(root)/run/'local'/relative
        if p.is_file():
            data=p.read_bytes()
            if len(data)!=binding['bytes'] or sha(data)!=binding['sha256']:raise ValueError('saved member identity:'+name)
            return data
    raise FileNotFoundError('selected member unavailable:'+name)


def select(root):
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    (out/'local').mkdir(mode=0o700);(out/'local/.gitignore').write_text('*\n')
    binding_data=fixed(Q8+'/event_bindings.json');bindings=json.loads(binding_data)
    selected=min(bindings,key=lambda r:r['sample_id']);sid=selected['sample_id']
    if sid!='q8_sample00' or selected['q']!=8 or selected['cut_after_complete_step']!=4:raise ValueError('fixed first q8 ID')
    index_path=Q8+'/payload_indices/'+sid+'.json';index=json.loads(fixed(index_path))
    names={'events.json'}|{'circuits/'+m+'_'+a+'_'+s+'.qpy' for m,a in WRAPPERS for s in ('raw','compiled')}
    selection=dict(sample_id=sid,q=8,cut=4,selection_rule='smallest saved sample_id before reading prices',
        members=[r for r in index['members'] if r['path'] in names],binding=selected,
        index_sha256=sha(fixed(index_path)),event_bindings_sha256=sha(binding_data),input_hashes={})
    if len(selection['members'])!=9:raise ValueError('four paired wrappers and event required')
    # Identity checks of compiled bytes are not price/algorithm inputs.
    for name in names:selected_member(root,selection,name)
    event=json.loads(selected_member(root,selection,'events.json'))
    for k in ('q','sample','sample_id','seed_domain','seeds','cut_after_complete_step'):
        if event[k]!=selected[k]:raise ValueError('event binding:'+k)
    if ((event['execution_source'],event['endpoint'],event['law_sha256'],event['input_packet_sha256'])
            !=(ORIGIN,ENDPOINT,LAW,PACKET) or len(event['ordered_events'])!=8
            or any(len(step)!=32 for step in event['ordered_events'])):raise ValueError('fixed event SOURCE/law')
    for name in ['SOURCE.json','paths/'+sid+'/precompile_checks.json','paths/'+sid+'/paired_observables.json']:
        selection['input_hashes'][Q8+'/'+name]=sha(fixed(Q8+'/'+name))
    for method,axis in WRAPPERS:
        rel=Q8+'/paths/'+sid+'/'+method+'_'+axis+'_compile.json';data=fixed(rel)
        record=json.loads(data)
        if record['status']!='PASS' or record['compiler']!=PROFILE:raise ValueError('P0 profile')
        selection['input_hashes'][rel]=sha(data)
    conditions=read(Path(root)/'run01_accelerated_completion/local'/sid/'conditions.json')
    if (conditions['q']!=8 or conditions['h']!=.1 or conditions['cut_after_complete_step']!=4
            or conditions['reference_gadgets']!=1 or conditions['short_gadget_repetition']
            or conditions['optimization_policy']!='strong_whole_path_half_sweep_relative_basis_scalar_aggregation_v1'):
        raise ValueError('whole-path/C1 OFF conditions')
    selection['conditions_sha256']=sha((Path(root)/'run01_accelerated_completion/local'/sid/'conditions.json').read_bytes())
    write(out/'SELECTION.json',selection)


def worker(task):
    kind,location,selection,method,axis=task;started=time.perf_counter();name=method+'_'+axis
    import qiskit
    from qiskit import qpy
    from trottertracks.track_d_native_phase160 import lower_with_ledger,optimize_with_ledger
    from trottertracks.track_d_native_resources import NativeCounter
    if qiskit.__version__!='1.3.0':raise ValueError('fixed Qiskit1.3')
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
        t=time.perf_counter()
        data=(member(location,archive_manifest(location),'circuits/'+name+'_raw.qpy') if kind=='q4_phase'
              else selected_member(location,selection,'circuits/'+name+'_raw.qpy'))
        circuits=qpy.load(io.BytesIO(data));assert len(circuits)==1;raw=circuits[0]
        if raw.num_qubits!=(9 if method=='symmetric_directional' else 17) or raw.num_clbits!=1:raise ValueError('wrapper width')
        load_seconds=time.perf_counter()-t;t=time.perf_counter();word=[]
        ledger=lower_with_ledger(raw,word.append);lower_seconds=time.perf_counter()-t
        t=time.perf_counter();result,observer,final,additions=optimize_with_ledger(word,raw.num_qubits,raw.num_clbits,ledger,observed_optimize)
        bridge_seconds=time.perf_counter()-t;t=time.perf_counter()
        phase=final.result();counter=NativeCounter(raw.num_qubits,raw.num_clbits,capture_transfer=True,
            global_phase=phase['normalized']).extend(iter(result['operations']))
        price=metrics(counter.result())
        if price!=metrics(counter.template().result()) or price!=metrics(result['output']):raise ValueError('transfer/count')
        count_seconds=time.perf_counter()-t
        row=dict(kind=kind,method=method,axis=axis,raw_qpy_sha256=sha(data),native_instructions=len(word),
            predicted_metrics=price,native_phase=ledger.result(),final_phase=phase,
            original_bridge_phase=result['global_phase'],rounds=result['rounds'],observer=observer,
            performance=dict(load_seconds=load_seconds,lowering_seconds=lower_seconds,bridge_seconds=bridge_seconds,
                count_transfer_seconds=count_seconds,worker_total_seconds=time.perf_counter()-started,
                peak_RSS_MiB=peak_mib(),maximum_native_word_container=len(word)),
            optimizer_rules_changed=False,P0_generalized=False,new_P0=0,new_build=0,new_simulation=0)
        if kind=='q4_phase':
            old=read(ROOT/OLD/(name+'_prediction.json'))
            if old['phases']['native']!=ledger.running or old['phases']['optimized']!=result['global_phase']:
                raise ValueError('legacy phase reproduction')
            for stage,ops in [('native',word),('bridge',result['operations'])]:
                path=ROOT/OLD/'local'/(name+'_'+stage+'.jsonl.gz')
                if sha(path.read_bytes())!=old['private_payload_sha256'][stage]:raise ValueError('old private payload')
                with gzip.open(path,'rt') as f:previous=[json.loads(line) for line in f]
                current=[[o.name,list(o.qubits),list(o.clbits),list(o.params)] for o in ops]
                if current!=previous:raise ValueError('phase-only change altered native word')
            before=next(r for r in read(ROOT/OLD/'COMPARISON.json')['records'] if (r['method'],r['axis'])==(method,axis))
            row.update(old_7_prices_preserved=price==old['predicted_metrics'],native_words_unchanged=True,
                old_phase_difference=before['phase_difference_mod_2pi'],
                new_phase_difference=math.remainder(phase['normalized']-before['saved_compiled_global_phase'],2*math.pi))
        else:
            row['private_word_sha256']=save_word(ROOT/OUT/'local'/(name+'_bridge.jsonl.gz'),result['operations'])
            row['oracle_used']=False
        # Phase terms are private decomposition payload, never public.
        t=time.perf_counter()
        with gzip.open(ROOT/OUT/'local'/(kind+'_'+name+'_phase_terms.json.gz'),'wt') as f:
            json.dump(dict(raw_phase_terms=ledger.values,oneq_phase_terms=additions),f)
        row['performance']['phase_terms_write_seconds']=time.perf_counter()-t
        row['performance']['worker_total_seconds']=time.perf_counter()-started
        row['performance']['peak_RSS_MiB']=peak_mib()
        write(ROOT/OUT/(kind+'_'+name+'.json'),row)
    return row


def execute(q8_root,q4_archive):
    out=ROOT/OUT;selection=read(out/'SELECTION.json');source=git('rev-parse','HEAD').decode().strip()
    pins={}
    for p in EXEC:
        data=(ROOT/p).read_bytes()
        if data!=git('show',source+':'+p):raise ValueError('unfixed SOURCE')
        if p in PINNED and data!=git('show',BASE+':'+p):raise ValueError('fixed old SOURCE changed')
        pins[p]=sha(data)
    if (out/'SELECTION.json').read_bytes()!=git('show',source+':'+OUT+'/SELECTION.json'):raise ValueError('unfixed selection')
    write(out/'SOURCE.json',dict(execution_source=source,base=BASE,files=pins,
        selection_sha256=sha((out/'SELECTION.json').read_bytes()),scope='q4 phase and selected q8 four raw predictions',
        compiler=PROFILE,raw_public=False))
    tasks=[(kind,location,selection,m,a) for kind,location in [('q4_phase',q4_archive),('q8_prediction',q8_root)] for m,a in WRAPPERS]
    start=time.perf_counter();rows=[];workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):
            row=f.result();rows.append(row);print(row['kind'],row['method'],row['axis'],'SAVED',flush=True)
    for p,h in pins.items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('SOURCE changed')
    write(out/'Q4_PHASE.json',dict(records=sorted([r for r in rows if r['kind']=='q4_phase'],key=lambda r:(r['method'],r['axis']))))
    write(out/'PREDICTIONS.json',dict(execution_source=source,status='FROZEN_BEFORE_Q8_ORACLE',workers=workers,
        combined_phase_q8_wall_seconds=time.perf_counter()-start,
        records=sorted([r for r in rows if r['kind']=='q8_prediction'],key=lambda r:(r['method'],r['axis'])),oracle_used=False))


def compare(root,commit):
    out=ROOT/OUT;commit=git('rev-parse',commit).decode().strip();data=(out/'PREDICTIONS.json').read_bytes()
    if data!=git('show',commit+':'+OUT+'/PREDICTIONS.json'):raise ValueError('unfixed predictions')
    selection=read(out/'SELECTION.json');pred=json.loads(data);prices=json.loads(fixed(Q8+'/prices.json'))
    import qiskit
    from qiskit import qpy
    from trottertracks.track_d_native_resources import count_native,iter_qiskit_native
    rows=[]
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden'))):
        for r in pred['records']:
            m,a=r['method'],r['axis'];name=m+'_'+a
            row=next(p for p in prices if p['sample_id']==selection['sample_id'] and (p['method'],p['axis'])==(m,a))
            if row['compiler']!=PROFILE or row['q']!=8 or row['request']['cut_after_complete_step']!=4:raise ValueError('oracle profile/scope')
            expected={k:row['metrics'][k] for k in METRICS if k!='qubit_count'};expected['qubit_count']=row['peak_qubits']
            payload=selected_member(root,selection,'circuits/'+name+'_compiled.qpy');c=qpy.load(io.BytesIO(payload))[0]
            if metrics(count_native(iter_qiskit_native(c),c.num_qubits,c.num_clbits))!=expected:raise ValueError('saved oracle/index')
            residual={k:r['predicted_metrics'][k]-expected[k] for k in METRICS}
            rows.append(dict(method=m,axis=a,predicted=r['predicted_metrics'],saved_P0=expected,residual=residual,
                matched_metrics=sum(v==0 for v in residual.values()),
                legacy_phase_difference=math.remainder(r['original_bridge_phase']-float(c.global_phase),2*math.pi),
                precise_phase_difference=math.remainder(r['final_phase']['normalized']-float(c.global_phase),2*math.pi),
                new_full_input_correctness=False,P0_generalized=False))
    write(out/'COMPARISON.json',dict(prediction_commit=commit,prediction_sha256=sha(data),records=rows,
        all_28_match=all(r['matched_metrics']==7 for r in rows),new_P0=0,new_simulation=0))
    write(out/'STOP.json',dict(mandatory_stop=True,status='FIXED_PHASE_Q8_COMPLETE',next_owner='user/GPT',
        new_sampling=0,new_build=0,new_P0=0,new_simulation=0,new_fit=0,N_or_RPE_updated=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','execute','compare'])
    p.add_argument('--q8-root',required=True);p.add_argument('--q4-archive');p.add_argument('--prediction-commit')
    a=p.parse_args()
    try:
        if a.phase=='select':select(a.q8_root)
        elif a.phase=='execute':
            if not a.q4_archive:raise ValueError('saved q4 archive required')
            execute(a.q8_root,a.q4_archive)
        else:
            if not a.prediction_commit:raise ValueError('fixed prediction commit required')
            compare(a.q8_root,a.prediction_commit)
    except Exception as error:
        out=ROOT/OUT
        if out.exists():
            message=str(error).replace(str(ROOT),'<task>').replace(a.q8_root,'<private-q8>')
            if a.q4_archive:message=message.replace(a.q4_archive,'<private-q4>')
            write(out/('FAILED_'+a.phase+'.json'),dict(status='STOP_PARTIAL',error_type=type(error).__name__,error=message))
        raise


if __name__=='__main__':main()
