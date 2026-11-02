"""Independently join artificial replay, raw/F01, ordered permits and state."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from .. import g6_w2c_b3_offline_result as V
from ..g6_w2c_b3_bundle_independent_identity_v1 import equal,hashed
from ..g6_w2c_b3_bundle_independent_trace_v1 import scan
from ..g6_w2c_b3_api_independent_v1.bits import replay

def expected(c,m,t,pid,axis,index,counter):
    V.require(type(pid)is int and pid>0 and type(index)is int and index>=0,'DISPATCH_INT_TYPES')
    kind,event,consumption,end=replay(t['arm'],axis,counter)
    if event is not None:event['artificial_port_variant']=m['port_variant']
    evhash=V.identity(dict(kind=kind,event=event));local=V.trial(dict(t,N=t['n']),axis,index)
    d=dict(schema='b3_ARTIFICIAL_global_port_dispatch_v1',artificial_only=True,endpoint_binding=m['endpoint_binding'],candidate_binding=hashed(c),manifest_binding=hashed(m),runtime_binding=hashed(c['worker_runtime']),factory=c['endpoint_binding']['adapter'],variant=m['port_variant'],run_id=m['run_id'],global_trial_id=m['run_id']+'/'+t['condition']+'/'+local,global_offset=t['offset']+(t['n'] if axis=='Y' else 0)+index,task_key=t['key'],worker_pid=pid,port_event_digest=evhash,consumption_digest=V.identity(consumption))
    evidence=dict(artificial=True,consumption=consumption,dispatch=d)
    angle=dict(artificial=True,not_a_mathematical_enclosure=True,event_requery_surrogate=V.identity(event),actual_angle=None) if t['layer']=='B' and kind=='EVENT' else None
    return kind,event,evidence,angle,end

def payload_check(payload,c,m,t,pid,axis,index,counter):
    equal(payload['trial_id'],V.trial(dict(t,N=t['n']),axis,index),'PORT_LOCAL_TRIAL_SEQUENCE');equal(payload['axis'],axis,'PORT_AXIS')
    kind,event,evidence,angle,end=expected(c,m,t,pid,axis,index,counter)
    equal(payload['kind'],kind,'PORT_KIND_FROM_BITS');equal(V.event_hash(payload),V.identity(dict(kind=kind,event=event)),'PORT_EVENT_FROM_BITS')
    if t['phase']=='profile' and index>=16:
        equal(payload.get('retention'),'HASHED_FULL_DOCUMENTS_NOT_RETAINED','PORT_RETENTION')
        equal(payload['evidence_sha256'],V.identity(evidence),'PORT_HASHED_DISPATCH_CONSUMPTION');equal(payload['angle_sha256'],V.identity(angle),'PORT_HASHED_ANGLE')
    else:
        equal(payload['event'],event,'PORT_EXACT_EVENT');equal(payload['evidence'],evidence,'PORT_EXACT_DISPATCH_CONSUMPTION');equal(payload['angle'],angle,'PORT_EXACT_ANGLE_REQUERY')
    return end,evidence

def verify(out,c,m,r,tasks):
    enters={};terminals={};reservations={};trace_gaps=[]
    def visit(x,size):
        if x['op'] in ('RESERVE','ENTER','COMPLETE','FAIL','UNKNOWN','CANCEL'):
            group=reservations if x['op']=='RESERVE' else enters if x['op']=='ENTER' else terminals
            gid=x['global_trial_id'];V.require(gid not in group,'DISPATCH_DUPLICATE_TRACE_ID');group[gid]=x
    ev=r['evidence_v2'];scan(out/'operations.jsonl',ev['trace_prefix'],m['run_id'],complete=ev['trace_complete'],visit=visit)
    bykey={row['task']['key']:row for row in r['pairs']};state=(out/'trial_states.bin').read_bytes();digest=sha256();counts=Counter();bitcounts=Counter();domains=set();seen=set();full=hashed_count=joined=visible=0;prefixes={};unjoined=[];typed=[]
    for t in tasks:
        row=bykey.get(t['key']);path=out/(t['key']+'.jsonl')
        if row is None or not path.exists():continue
        counters={'X':0,'Y':0};indices={'X':0,'Y':0};last_axis='X'
        reader=V.Reader(path)
        for envelope in reader:
            p=envelope.payload
            if p.get('kind') not in V.KINDS or 'trial_id' not in p:continue
            axis=p['axis'];V.require(axis in counters and not(last_axis=='Y' and axis=='X'),'PORT_AXIS_ORDER');last_axis=axis;i=indices[axis]
            end,evidence=payload_check(p,c,m,t,row['pid'],axis,i,counters[axis]);counters[axis]=end;indices[axis]+=1
            d=evidence['dispatch'];gid=d['global_trial_id'];offset=d['global_offset'];V.require(gid not in seen,'PORT_DUPLICATE_GLOBAL_TRIAL');seen.add(gid)
            if gid in terminals and terminals[gid]['op']=='COMPLETE':
                V.require(gid in enters and gid in reservations,'PORT_WITHOUT_GLOBAL_PERMIT')
                for tr in (reservations[gid],enters[gid],terminals[gid]):equal(tr['global_offset'],offset,'PORT_TRACE_GLOBAL_OFFSET');equal(tr['task_key'],t['key'],'PORT_TRACE_TASK')
                if ev['statefile_complete']:equal(state[offset],4,'PORT_COMPLETED_PHYSICAL_STATE')
                joined+=1
            else:unjoined.append(gid)
            key=(t['arm'],axis,i);value=(V.event_hash(p),V.identity(evidence['consumption']))
            if key in prefixes:equal(prefixes[key],value,'SP1_AB_PHASE_PASS_PORT_PREFIX')
            prefixes[key]=value;domains.add(evidence['consumption']['domain']);counts[t['condition']+'/'+t['arm']+'/'+t['layer']+'/'+axis]+=1
            bitcounts[t['condition']+'/'+t['arm']+'/'+t['layer']+'/'+axis]+=160*len(evidence['consumption']['draws']);visible+=1
            full+=int('evidence' in p);hashed_count+=int('evidence' not in p);digest.update(V.canonical(dict(dispatch=d,consumption=evidence['consumption'],event_hash=V.event_hash(p))).encode())
        failure=(row.get('failure') or {}).get('trial_failure_provenance')
        if failure:
            cur=failure['current_attempt'];equal(failure['mandatory_STOP'],True,'F01_STOP');equal(failure['is_random_zero'],False,'F01_NOT_RANDOM_ZERO');equal(failure['retry'],0,'F01_RETRY');equal(failure['resume'],False,'F01_RESUME')
            equal(cur['trial_completed'],False,'FAILED_TRIAL_NOT_COMPLETED')
            if cur.get('result_acquired') and cur.get('event_present'):
                axis=cur['axis'];index=cur['trial_index'];_,event,evidence,angle,_=expected(c,m,t,row['pid'],axis,index,counters[axis])
                # Legacy after_event injects a deliberately invalid phase. Its F01
                # must retain that failing event; do not reinterpret as a good event.
                f01event=failure.get('inflight_event')
                if f01event is not None and f01event.get('phase_i_power')==99:
                    V.require(any(f[2]=='after_event' for f in m['condition_faults'].get(t['condition'],[])),'UNDECLARED_BAD_PHASE');event['phase_i_power']=99
                equal(f01event,event,'F01_INFLIGHT_PORT_EVENT')
            typed.append(dict(task=t['key'],type=failure['type'],reason=failure['reason'],stage=cur['stage'],completed=False))
    return dict(verified_trials=visible,scheduler_complete_joined=joined,full_document_trials=full,hashed_document_trials=hashed_count,domains=len(domains),cell_mode_counts=dict(counts),consumed_bits=dict(bitcounts),evidence_chain_sha256=digest.hexdigest(),unjoined_or_partial=unjoined,typed_failures=typed,science_equivalence=False,complete=False)
