"""Offline same-author verification. Does not import producer/authority/science.

Reconstruct synthetic documents independently from SHA counter input and join
all trial evidence to worker PID, global trace, state, receipts and caller.
Codec/result validators are shared and therefore not an independent review.
"""
from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
from . import codec as C, result as R, storage_consumer as B
from .. import g6_w2c_b3_offline_result as V


def reference(arm,axis,counter,tid,layer):
    domain='ARTIFICIAL_REAL_ADAPTER_V1/'+arm+'/'+axis
    before_counter=counter
    draws=[]
    def take():
        nonlocal counter
        value=int.from_bytes(sha256(domain.encode()+counter.to_bytes(8,'big')).digest(),'big')>>96
        draws.append(dict(width=160,counter_before=counter,counter_after=counter+1,
                          value_hex=hex(value),discarded_bits=96));counter+=1
        return value
    token=take()
    reason=('RAW_NONREDUCED','COIN_REJECT','ACCEPTED','ACCEPTED')[token%4] if arm=='full_return' else 'ACCEPTED'
    raw=(0,0) if reason=='RAW_NONREDUCED' or arm!='full_return' else (0,1)
    stages=['order']
    if arm=='full_return':
        for _ in range(2):token^=take();stages.append('label')
        if reason!='RAW_NONREDUCED':token^=take();stages.append('acceptance')
        if reason=='ACCEPTED':token^=take();stages.append('child')
    else:token^=take();stages.append('child')
    event=None
    if reason=='ACCEPTED':
        ratio=F(2**24+(token%8192)+1,2**26+57);weight=ratio*F(7,11)
        word=raw if arm=='full_return' else ()
        event=dict(word=word,raw_word=raw,child=2,ratio=ratio,weight=weight,
                   coefficient=weight/2,proposal=F(1,2),phase_i_power=2,
                   provider_calls={0:1,1:1,2:2} if word else {2:2},helper=1,
                   event_unitary='i^phase exp(-i*sigma*atan(ratio)*Q_child) Q(word)',
                   circuit_time_word=word[::-1])
    ids=dict(input_id='ARTIFICIAL_INPUT_V1',policy_id='ARTIFICIAL_PRECISION_V1',
             source_id='ARTIFICIAL_SOURCE_G7_F3_SHAPED_V1',preflight_id='ARTIFICIAL_PREFLIGHT_V1',
             trial_id=tid,sigma=1)
    consumption=dict(schema='b3_ARTIFICIAL_adapter_consumption_v1',domain=domain,draws=draws,
                     counter_after=counter,bits_consumed=160*counter,bits_consumed_before=160*before_counter,artificial_only=True,
                     real_consumption_equivalence=False)
    trace=dict(draws=[dict(stage=s,width=160,value=int(d['value_hex'],16)) for s,d in zip(stages,draws)],
               raw_word=raw,degree=2,bits_consumed=160*len(draws),draw_count=len(draws),
               completed_bit_map_calls_by_stage={s:stages.count(s) for s in ('order','label','acceptance','child')},
               extra_random_draws_for_audit=0,fresh_IID_verified=False)
    if arm=='full_return':
        data=dict(schema='g6_f3_trial_v1',**ids,run_completion_or_confidence_certified=False)
        if event is None:data.update(kind='ZERO',reason=reason,trace=trace,is_valid_zero_trial=True)
        else:data.update(kind='EVENT',event=event,trace=trace)
        record=C.Document.create(data)
        if layer=='B':adapted=R.adapt_f3(record,expected=ids)
        else:
            obs=C.Document.create(dict(schema='g7_observation_candidate_v1',**ids,arm=arm,
                branch=reason,trace=trace,packet_calls=0 if reason=='RAW_NONREDUCED' else 1,
                group_index=None,event=event))
            adapted=R.adapt_g7(event,observation=obs,expected=ids,arm=arm)
    else:
        record=C.Document.create(dict(schema='g7_observation_candidate_v1',**ids,arm=arm,
            branch=reason,trace=consumption,packet_calls=0,group_index=0,event=event))
        adapted=R.adapt_g7(event,observation=record,expected=ids,arm=arm)
    angle=None
    if layer=='B' and event is not None:
        source=C.restore(adapted['codec'],source_identity=ids['source_id']).read()
        low=F(1,8);high=low+F(1,10**8);mid=(low+high)/2;radius=(high-low)/2
        a=C.Document.create(dict(schema='g6_f3_angle_enclosure_v1',ratio=event['ratio'],sigma=1,
            root_bits=32,terms=32,tau=F(1,10**6),context={**{k:ids[k] for k in (*R.IDS,'sigma')},
                'event_id':adapted['event_identity'],'parent':event['word'],'child':2},
            sqrt_interval=None,half_argument_interval=None,endpoint_series_evidence=None,
            phi_interval=(low,high),representative=mid,representative_error_upper=radius,
            signed_phi_interval=(low,high),signed_representative=mid,
            precision_check=dict(name='representative_radius',left=radius,operator='<=',right=F(1,10**6),passed=True),
            native_rotation_acquired=False,physical_provider_certified=False,
            basis={'scope':'ARTIFICIAL_SHAPE_ONLY_NOT_AN_ATAN_ENCLOSURE'}))
        query=dict(word=event['word'],child=2)
        if arm!='full_return':query.update(group_index=0,raw_word=raw)
        if arm!='full_return':
            data=a.read();data['context']=dict(event_id=C.digest_bytes(C.driver_canonical(event).encode()),
                input={'id':'ARTIFICIAL_INPUT_V1'},policy={'id':'ARTIFICIAL_PRECISION_V1'},preflight=ids['preflight_id'])
            angle=R.canonical_angle_binding(adapted,requery_input=query,requery_event=event,
                angle_record=C.Document.create(data),input_payload={'id':'ARTIFICIAL_INPUT_V1'},
                policy_payload={'id':'ARTIFICIAL_PRECISION_V1'},synthetic=True)
        else:
            angle=R.angle_binding(adapted,requery_input=query,requery_event=event,angle_record=a,synthetic=True)
    return adapted,consumption,angle,counter


def verify(root,trusted,source_root=None):
    result=B.verify_storage(root,trusted,source_root)
    result['schema']='b3_closed_document_adapter_offline_v1'
    result['adapter_dispatch']=dict(complete=False,verified_trials=0)
    if result['status']=='INTEGRITY_MISMATCH':return result
    root=Path(root);out=root/'output'
    try:
        if not(out/'final.json').exists():return result
        c=B._read(out/'candidate.json');m=B._read(out/'manifest.json');r=B._read(out/'final.json')
        binding=c['endpoint_binding']
        V.require(c['integration']=='ONE_GLOBAL_COORDINATOR_CLOSED_DOCUMENT_ADAPTER_V1' and
                  binding['schema']=='b3_closed_real_adapter_binding_v1' and binding['real_dispatch'] is False and
                  binding['backend']=='SYNTHETIC_DOCUMENTS_ONLY' and
                  m['endpoint_binding']==B._identity(binding)==r['port_binding'],'ADAPTER_BINDING')
        for name in ('summary.json','receipt.json'):
            if (out/name).exists():V.require(B._read(out/name)['port_binding']==m['endpoint_binding'],'RECEIPT_ADAPTER_BINDING')
        trace,_=B.T.read_prefix(out/'operations.jsonl',r['evidence_v2']['trace_prefix'],m['run_id'],r['evidence_v2']['trace_complete'])
        complete={x['global_trial_id']:x for x in trace if x['op']=='COMPLETE'}
        enter={x['global_trial_id']:x for x in trace if x['op']=='ENTER'}
        rows={p['task']['key']:p for p in r['pairs']}
        counts=Counter();full=hashed=joined=0;chain=sha256();gaps=[]
        for task in m['tasks']:
            path=out/(task['key']+'.jsonl');row=rows.get(task['key'])
            if row is None or not path.exists():continue
            counter={'X':0,'Y':0};indices={'X':0,'Y':0}
            for envelope in V.Reader(path):
                p=envelope.payload
                if p.get('kind') not in V.KINDS or 'trial_id' not in p:continue
                tid=p['trial_id'];phase,pa,arm,layer,axis,idx=tid.split('/');idx=int(idx)
                V.require((phase,pa,arm,layer)==tuple(task[k] for k in ('phase','pass_name','arm','layer')) and
                          axis in counter and idx==indices[axis],'ADAPTER_TRIAL_ORDER')
                indices[axis]+=1
                a,consumption,angle,counter[axis]=reference(arm,axis,counter[axis],tid,layer)
                kind,event=a['driver_kind'],a['event'];gid=m['run_id']+'/'+task['condition']+'/'+tid
                offset=task['offset']+(0 if axis=='X' else task['n'])+idx
                evhash=V.identity(dict(kind=kind,event=event))
                dispatch=dict(schema='b3_ARTIFICIAL_adapter_dispatch_v1',artificial_only=True,
                    endpoint_binding=m['endpoint_binding'],candidate_binding=B._identity(c),manifest_binding=B._identity(m),
                    runtime_binding=B._identity(c['worker_runtime']),factory=binding['adapter'],run_id=m['run_id'],
                    global_trial_id=gid,global_offset=offset,task_key=task['key'],worker_pid=row['pid'],
                    event_digest=evhash,adapted_digest=C.fingerprint(a),consumption_digest=C.fingerprint(consumption))
                evidence=dict(artificial=True,origin='SYNTHETIC_DOCUMENTS_NOT_REAL_G7_F3',consumption=consumption,
                              adapted=a,dispatch=dispatch)
                if layer=='B':
                    identity=C.fingerprint(dict(arm=arm,source='ARTIFICIAL_SOURCE_G7_F3_SHAPED_V1'))
                    state=C.fingerprint(dict(artificial=True,arm=arm,layer=layer))
                    original=(a['g7_observation']['f3']['document'] if arm=='full_return' and layer=='A' else a['codec']['f3']['document'])
                    calls=idx+(task['n'] if axis=='Y' else 0)
                    seal=C.DriverDocument.create(dict(arm=arm,identity=identity,state_id=state,payload=original,
                        completed_calls_before=calls,confidence_certified=False))
                    evidence['audit']=R.canonical_audit(seal,arm=arm,identity=identity,state_id=state,completed_calls_before=calls)
                V.require(p['kind']==kind and V.event_hash(p)==evhash,'ADAPTER_EVENT_DIGEST')
                if 'evidence' in p:
                    V.require(V.identity(p['evidence'])==V.identity(evidence),'ADAPTER_FULL_EVIDENCE_CODEC_DIGEST')
                    V.require(V.identity(p['angle'])==V.identity(angle),'ADAPTER_ANGLE_REQUERY');full+=1
                else:
                    V.require(p['evidence_sha256']==V.identity(evidence) and p['angle_sha256']==V.identity(angle),
                              'ADAPTER_HASHED_EVIDENCE_CODEC_DIGEST');hashed+=1
                if gid in complete:
                    V.require(gid in enter and complete[gid]['global_offset']==enter[gid]['global_offset']==offset,
                              'ADAPTER_GLOBAL_TRACE_JOIN');joined+=1
                else:gaps.append(gid)
                counts[kind]+=1;chain.update(C.canonical(dict(dispatch=dispatch,adapted=a,angle=angle)).encode())
            if row.get('status')=='OK':
                for axis in row['outcome']['axes']:
                    V.require(axis['H_bits_supplied']==160*counter[axis['axis']],'ADAPTER_BIT_TOTAL')
        success=result['status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'
        if success:V.require(full+hashed==joined==m['planned'] and not gaps,'ADAPTER_COMPLETE_JOIN')
        result['adapter_dispatch']=dict(complete=success,verified_trials=full+hashed,full_documents=full,
            hashed_documents=hashed,scheduler_joined=joined,counts=dict(counts),unjoined=gaps,
            evidence_chain_sha256=chain.hexdigest(),real_G7_F3_compatibility='NOT_ESTABLISHED')
    except (V.Invalid,C.Rejected,KeyError,TypeError,ValueError,OSError) as exc:
        result.update(status='INCOMPLETE_EVIDENCE' if isinstance(exc,FileNotFoundError) else 'INTEGRITY_MISMATCH',
                      issues=result['issues']+[str(exc)])
    return result
