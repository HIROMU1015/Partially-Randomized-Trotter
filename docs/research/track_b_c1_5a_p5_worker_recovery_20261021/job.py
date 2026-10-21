"""P5R child roles: one sampler owner, fresh exec logical providers, no inherited pool."""
from collections import Counter
from fractions import Fraction as F
from functools import reduce
from operator import mul
import faulthandler
import gzip
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from common import HERE,OLD,read,save,canonical,require,verify,journal,write_gzip,digest,proc_info

def path_record(sampler,trial,timers,contract_id):
    from bridge import connect
    d=connect()
    from trottertracks.algorithm_codesign.g6_f3_contract import Evidence
    steps=[];events=[]
    for microstep in range(4):
        with timers.stage('sampling_checks_event_Evidence_storage'):step=sampler.sample(trial,microstep)
        require(step['event'] is not None,'CANONICAL_P5_UNEXPECTED_ZERO')
        events.append(step['event']);steps.append(step)
    rotations=[sampler.angle(e,timers) for e in events]
    for step,(_,aid) in zip(steps,rotations):step['angle_id']=aid
    with timers.stage('weight_path_Evidence'):
        weight=reduce(mul,(e['weight'] for e in events),F(1));proposal=reduce(mul,(e['proposal'] for e in events),F(1))
        exact=Evidence.create(dict(schema='C1_5A_PREMEASUREMENT_PATH_V1',contract_id=contract_id,
            method='closed_P5_full',trial=trial,model_id=sampler.model['adapter'].model_id,labels_id=sampler.model['labels_id'],
            microsteps=[{k:v for k,v in s.items() if k!='event'} for s in steps],kind='ACCEPTED',weight=weight,proposal=proposal,
            measurement_performed=False,coherent_signal_estimator=None,all_four_steps_generated=True,zero_quantum_calls=False,
            scalar_phase_exact=F(4,5)*sampler.model['scalar']))
        require(exact.read()['weight']==weight,'PATH_EVIDENCE');w=float(weight);require(math.isfinite(w*w),'FINITE_OBSERVED_WEIGHT')
    return dict(exact=exact.to_dict(),exact_digest=exact.digest,logical=None,observed_weight_binary64=w,
        observed_weight_square_binary64=w*w,zero_reasons=[],reduced_lengths=[len(e['word']) for e in events],
        raw_degrees=[len(e['raw_word']) for e in events])

def decode_path(out,row,model,events,angles):
    from bridge import connect
    d=connect()
    from trottertracks.algorithm_codesign.g6_f3_contract import Evidence,_unpack
    exact=Evidence.create(_unpack(row['exact']));require(exact.digest==row['exact_digest'],'PATH_DIGEST')
    record=exact.read();trial=record['trial']
    require(len(record['microsteps'])==4 and record['kind']=='ACCEPTED' and record['measurement_performed'] is False and
        record['all_four_steps_generated'] is True and record['zero_quantum_calls'] is False and
        record['scalar_phase_exact']==F(4,5)*model['scalar'],'P5_PATH')
    path_events=[];rotations=[]
    for k,step in enumerate(record['microsteps']):
        eid=step['event_id'];aid=step['angle_id']
        if eid not in events:
            raw=gzip.decompress((out/'events'/(eid+'.json.gz')).read_bytes());packet=Evidence(raw.decode(),eid).read()
            require(packet['model_id']==record['model_id'] and packet['labels_id']==record['labels_id'] and
                packet['contract_id']==record['contract_id'] and packet['method']=='closed_P5_full','EVENT_BINDING')
            events[eid]=packet['event']
        if aid not in angles:
            raw=gzip.decompress((out/'angles'/(aid+'.json.gz')).read_bytes());a=Evidence(raw.decode(),aid).read()
            inner=Evidence.create(a['angle']).read();representative=float.fromhex(a['representative_binary64'])
            require(float(inner['signed_representative'])==representative and inner['sigma']==1 and
                inner['precision_check']['passed'] and inner['context']['contract_id']==record['contract_id'],'ANGLE_EVIDENCE')
            angles[aid]=(representative,inner['ratio'])
        e=events[eid];require(e['proposal']>0 and e['proposal']*e['weight']==e['coefficient'],'ACTUAL_WEIGHT')
        require(e['raw_word']==e['word'] and e['phase_i_power']==len(e['word'])%4 and
            e['circuit_time_word']==tuple(reversed(e['word'])),'ORDER_PHASE')
        require(step['header']['trial_id']=='closed_P5_full:'+str(trial)+':'+str(k),'MICROSTEP_ID')
        bits=d.Bits('closed_P5_full',trial,k)
        require(step['bits_key_id']==__import__('hashlib').sha256(bits.key).hexdigest(),'BITS_KEY_BINDING')
        for draw in step['bits_draws']:
            require(bits.bits(draw['width'])==draw['value'],'RECORDED_BITSTREAM')
        require(angles[aid][1]==e['ratio'],'ANGLE_RATIO_BINDING')
        path_events.append(e);rotations.append(angles[aid][0])
    require(reduce(mul,(e['weight'] for e in path_events),F(1))==record['weight'],'WEIGHT_PRODUCT')
    require(reduce(mul,(e['proposal'] for e in path_events),F(1))==record['proposal'],'PROPOSAL_PRODUCT')
    require(record['model_id']==model['adapter'].model_id and record['labels_id']==model['labels_id'],'MODEL_LABEL')
    return record,path_events,rotations

def producer():
    from bridge import connect
    d=connect()
    from selector import IndexedP5
    from trottertracks.algorithm_codesign.g6_f3_contract import Evidence,fingerprint
    verify();auth=read(HERE/'authorization_receipt.json');out=HERE/'result';out.mkdir(exist_ok=True)
    timers=d.Timers();started=time.perf_counter()
    with timers.stage('R1_input_registry_setup'): model=d.load_model()
    with timers.stage('P5_normalizer_completion_setup'):
        sampler=d.Sampler('closed_P5_full',model,auth['contract_SHA256'],out)
    with timers.stage('group_prefix_selector_setup'):
        sampler.g=IndexedP5.attach(sampler.g)
    save(out/'setup.json',dict(groups=len(sampler.g.groups),model_id=model['adapter'].model_id,
        labels_id=model['labels_id'],p_id=fingerprint(model['p']),x=Evidence.create(model['x']).to_dict(),
        precision=Evidence.create(sampler.precision).to_dict(),
        normalizer=Evidence.create(dict(B=(sampler.g.B.lo,sampler.g.B.hi),U=(sampler.g.U.lo,sampler.g.U.hi) if hasattr(sampler.g,'U') else None)).to_dict(),
        stages=timers.rows,process=proc_info(os.getpid()),
        max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        canonical_certificate='UNCERTIFIED_EXPLORATORY',sampler_instances=1,group_table_never_forked=True))
    online=d.Timers();start=time.perf_counter()
    for batch in range(32):
        partial=out/'sampling'/(f'batch_{batch:03}.jsonl.partial');partial.parent.mkdir(exist_ok=True)
        with partial.open('xb') as stream:
            for trial in range(batch*32,(batch+1)*32):
                journal(out/'producer.events.jsonl',event='TRIAL_START',batch=batch,trial=trial)
                row=path_record(sampler,trial,online,auth['contract_SHA256'])
                with online.stage('path_durable_storage'):
                    stream.write(canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
                journal(out/'producer.events.jsonl',event='TRIAL_COMMITTED',batch=batch,trial=trial)
        with online.stage('sampling_batch_compression'):
            raw=partial.read_bytes();target=partial.with_name(f'batch_{batch:03}.jsonl.gz')
            write_gzip(target,raw);require(gzip.decompress(target.read_bytes())==raw,'BATCH_ROUNDTRIP');partial.unlink()
        save(out/'producer_progress.json',dict(completed_trials=(batch+1)*32,completed_batch=batch,online_wall=time.perf_counter()-start))
    verify()
    save(out/'producer_summary.json',dict(complete=True,trials=1024,microsteps=4096,setup_stages=timers.rows,
        online_stages=online.rows,online_wall_seconds=time.perf_counter()-start,total_wall_seconds=time.perf_counter()-started,
        angle_cache_hits=sampler.angle_hits,angle_cache_misses=sampler.angle_misses,
        event_cache_hits=sampler.event_cache_hits,event_cache_misses=sampler.event_cache_misses,
        SOURCE_derived_online_query_counts=dict(sampler.calls),max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        final_process=proc_info(os.getpid()),coherent_signal_estimator=None,retry=0))

def provider(batch):
    from bridge import connect
    d=connect()
    from trottertracks.algorithm_codesign.g6_f3_contract import Evidence,_unpack
    verify();out=HERE/'result';timers=d.Timers();events={};angles={};start=time.perf_counter()
    with timers.stage('R1_input_registry_setup'):model=d.load_model()
    source=out/'sampling'/f'batch_{batch:03}.jsonl.gz'
    lines=gzip.decompress(source.read_bytes()).splitlines();require(len(lines)==32,'BATCH_PLAN')
    partial=out/'batches'/f'batch_{batch:03}.jsonl.partial';partial.parent.mkdir(exist_ok=True)
    with partial.open('xb') as stream:
        for line in lines:
            row=json.loads(line)
            with timers.stage('Evidence_decode_validate'):
                record,path_events,rotations=decode_path(out,row,model,events,angles)
                trial=record['trial'];require(trial//32==batch,'BATCH_TRIAL_ID')
                require(record['contract_id']==read(HERE/'authorization_receipt.json')['contract_SHA256'],'CONTRACT_BINDING')
            journal(out/'workers'/f'batch_{batch:03}.events.jsonl',event='TRIAL_START',batch=batch,trial=trial)
            with timers.stage('logical_build'):
                circuit=d.build_tail(model['adapter'],model['scalar'],path_events,rotations)
            with timers.stage('logical_accounting'):
                row['logical']=d.logical_resources(circuit,path_events)
            with timers.stage('worker_path_storage'):
                stream.write(canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
            journal(out/'workers'/f'batch_{batch:03}.events.jsonl',event='TRIAL_COMMITTED',batch=batch,trial=trial)
    target=out/'batches'/f'batch_{batch:03}.jsonl.gz'
    with timers.stage('worker_batch_compression'):
        raw=partial.read_bytes();write_gzip(target,raw);require(gzip.decompress(target.read_bytes())==raw,'BATCH_ROUNDTRIP');partial.unlink()
    verify()
    save(out/'receipts'/f'batch_{batch:03}.json',dict(batch=batch,completed=32,pid=os.getpid(),
        first_trial=batch*32,last_trial=batch*32+31,stages=timers.rows,wall_seconds=time.perf_counter()-start,
        compressed_bytes=target.stat().st_size,sha256=digest(target),
        max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,event_lookup_entries=len(events),angle_lookup_entries=len(angles)))

if __name__=='__main__':
    faulthandler.enable(all_threads=True)
    role=sys.argv[1]
    try:
        require((HERE/'attempt_marker.json').exists(),'NEW_ONE_SHOT_MARKER_REQUIRED')
        if role=='producer':producer()
        elif role=='provider':provider(int(sys.argv[2]))
        else:raise ValueError('UNKNOWN_ROLE')
    except BaseException as exc:
        save(HERE/'result/failures'/(role+'_'+str(os.getpid())+'.json'),dict(role=role,
            exception=repr(exc),traceback=traceback.format_exc(),process=proc_info(os.getpid()),retry=0,mandatory_STOP=True))
        raise
