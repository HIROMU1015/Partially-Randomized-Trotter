"""Independent judgment from raw, not Producer API calls. Same-author self-check.

No import of bridge, adapter, contract, API ports, scheduler or authority.
"""
from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json,sys
from . import storage_consumer as B
from .. import g6_w2c_b3_offline_result as V

plan,_read,_sha=B.plan,B._read,B._sha

def surrogate(arm,axis,counter,variant):
    """Only artificial namespace bytes, independently reproduced from fixed spec."""
    domain='ARTIFICIAL_API_BINDING_V1/'+arm+'/'+axis
    draws=[]
    def draw():
        nonlocal counter
        value=int.from_bytes(sha256(domain.encode()+counter.to_bytes(8,'big')).digest(),'big')>>96
        draws.append(dict(width=160,counter_before=counter,counter_after=counter+1,value_hex=hex(value),discarded_bits=96))
        counter+=1
        return value
    token=draw()
    kind=('ZERO_R','ZERO_C','EVENT','EVENT')[token%4]if arm=='full_return'else'EVENT'
    if kind!='ZERO_R':token^=draw()
    event=None
    if kind=='EVENT':
        if arm=='full_return':token^=draw()
        ratio=F((1<<24)+(token&8191)+1,(1<<26)+57);weight=ratio*F(7,11)
        raw=(0,1)if arm=='full_return'else(0,0);word=raw if arm=='full_return'else()
        event=dict(raw_word=raw,word=word,child=2,ratio=ratio,weight=weight,coefficient=weight/2,proposal=F(1,2),
            phase_i_power=2,circuit_time_word=tuple(reversed(word)),provider_calls={0:1,1:1,2:2}if word else{2:2},artificial_port_variant=variant)
    trace=dict(schema='b3_ARTIFICIAL_consumption_v1',domain=domain,draws=draws,counter_after=counter,bits_consumed=160*counter,
        artificial_only=True,real_consumption_equivalence=False)
    return kind,event,trace,counter

def verify(root,trusted,source_root=None):
    result=B.verify_storage(root,trusted,source_root)
    result['schema']='b3_closed_port_global_offline_acceptance_v1'
    result['port_dispatch']=dict(complete=False,verified_trials=0)
    if result['status']=='INTEGRITY_MISMATCH':return result
    root=Path(root);out=root/'output'
    try:
        if not(out/'final.json').exists():return result
        c=B._read(out/'candidate.json');m=B._read(out/'manifest.json');r=B._read(out/'final.json')
        expected=c['endpoint_binding']
        V.require(c['integration']=='ONE_GLOBAL_COORDINATOR_REAL_PORT_DISPATCH_V1'and
            m['endpoint_binding']==B._identity(expected)==r['port_binding'],'PORT_BINDING')
        V.require(expected['original_port_commit']=='82896d53fa97235fa848d6c0833eb193af8b99d6'
            and expected['real_dispatch']is False and expected['artificial_only']is True,'FIXED_CLOSED_PORT')
        V.require(expected['port_source']=={p:h for p,h in c['source_pins'].items()if '/g6_w2c_b3_api_binding_v1/'in p},'PORT_SOURCE_HASHES')
        V.require(m['port_variant']in ('PORT_V1','PORT_V2')and m['seed_namespace']=='ARTIFICIAL_API_BINDING_V1','ARTIFICIAL_VARIANT_SEED')
        for f in ('summary.json','receipt.json'):
            if(out/f).exists()and(out/f).read_bytes().strip():
                V.require(B._read(out/f)['port_binding']==m['endpoint_binding'],'POSTCLOSE_PORT_BINDING')
            elif result['status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE':
                V.require(False,'MISSING_POSTCLOSE_PORT_BINDING')
        tasks,_=B.plan(m['profile_n']);rows={x['task']['key']:x for x in r['pairs']}
        trace,_=B.T.read_prefix(out/'operations.jsonl',r['evidence_v2']['trace_prefix'],m['run_id'],r['evidence_v2']['trace_complete'])
        terminals={x['global_trial_id']:x for x in trace if x['op']=='COMPLETE'}
        enters={x['global_trial_id']:x for x in trace if x['op']=='ENTER'}
        counts=Counter();bit_counts=Counter();digest=sha256();joined=0;visible=0;full=hashed=0;domains=set();gaps=[]
        for task in tasks:
            path=out/(task['key']+'.jsonl');row=rows.get(task['key'])
            if row is None or not path.exists():continue
            counters={'X':0,'Y':0}
            try:
                for envelope in V.Reader(path):
                    payload=envelope.payload
                    if payload.get('kind')not in V.KINDS or'trial_id'not in payload:continue
                    visible+=1
                    tid=payload['trial_id'];fields=tid.split('/')
                    V.require(len(fields)==6,'PORT_LOCAL_ID')
                    phase,pa,arm,layer,axis,index=fields;index=int(index)
                    V.require((phase,pa,arm,layer)==(task['phase'],task['pass_name'],task['arm'],task['layer'])and axis in counters,'PORT_TASK_CELL')
                    kind,event,consumption,after=surrogate(arm,axis,counters[axis],m['port_variant']);counters[axis]=after
                    V.require(payload['kind']==kind,'PORT_RESULT_KIND')
                    evhash=V.identity(dict(kind=kind,event=event))
                    V.require(V.event_hash(payload)==evhash,'PORT_EVENT_PAYLOAD_WEIGHT')
                    gid=m['run_id']+'/'+task['condition']+'/'+tid
                    offset=task['offset']+(0 if axis=='X'else task['n'])+index
                    dispatch=dict(schema='b3_ARTIFICIAL_global_port_dispatch_v1',artificial_only=True,
                        endpoint_binding=m['endpoint_binding'],candidate_binding=B._identity(c),manifest_binding=B._identity(m),
                        runtime_binding=B._identity(c['worker_runtime']),factory=expected['adapter'],variant=m['port_variant'],run_id=m['run_id'],
                        global_trial_id=gid,global_offset=offset,task_key=task['key'],worker_pid=row['pid'],port_event_digest=evhash,
                        consumption_digest=V.identity(consumption))
                    evidence=dict(artificial=True,consumption=consumption,dispatch=dispatch)
                    angle=None
                    if layer=='B'and kind=='EVENT':angle=dict(artificial=True,not_a_mathematical_enclosure=True,event_requery_surrogate=V.identity(event),actual_angle=None)
                    if'evidence'in payload:
                        V.require(V.identity(payload['evidence'])==V.identity(evidence),'PORT_FULL_EVIDENCE_CONSUMPTION')
                        V.require(V.identity(payload['angle'])==V.identity(angle),'PORT_ANGLE_REQUERY')
                        full+=1
                    else:
                        V.require(payload['evidence_sha256']==V.identity(evidence)and payload['angle_sha256']==V.identity(angle),'PORT_HASHED_EVIDENCE_CONSUMPTION')
                        hashed+=1
                    if gid in terminals:
                        V.require(gid in enters and terminals[gid]['global_offset']==enters[gid]['global_offset']==offset,'PORT_SCHEDULER_TRACE_JOIN')
                        joined+=1
                    else:gaps.append(gid)
                    key='/'.join((task['condition'],arm,layer,axis));counts[key]+=1;bit_counts[key]+=160*len(consumption['draws']);domains.add(consumption['domain'])
                    digest.update(V.canonical(dict(dispatch=dispatch,consumption=consumption,event_hash=evhash)).encode())
            except V.Invalid:
                raise
        complete=result['status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'
        if complete:V.require(joined==visible==r['states']['COMPLETED']==m['planned']and not gaps and len(domains)==8,'NO_UNDISPATCHED_SUCCESS')
        result['port_dispatch']=dict(complete=complete,verified_trials=visible,scheduler_complete_joined=joined,
            full_document_trials=full,hashed_document_trials=hashed,domains=len(domains),cell_mode_counts=dict(counts),
            consumed_bits=dict(bit_counts),evidence_chain_sha256=digest.hexdigest(),unjoined_or_partial=gaps,
            raw_binding='Port evidence in existing pair output; no external port grid or extra trial sidecar',
            actual_G7_law_or_bits_equivalence=False)
    except (V.Invalid,KeyError,TypeError,ValueError,OSError)as exc:
        result.update(status='INCOMPLETE_EVIDENCE'if isinstance(exc,FileNotFoundError)else'INTEGRITY_MISMATCH',issues=result['issues']+[str(exc)])
    result['consumer_forbidden_imports']=[n for n in sys.modules if n.rsplit('.',1)[-1]in ('bridge','adapter','contract','ports','exercise','authority','g6_w2c_b3_parallel_scheduler','g7_generator','g6_f3_generator')]
    return result
