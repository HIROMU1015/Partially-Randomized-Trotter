"""Replay only the explicitly ARTIFICIAL domain, including exact surrogate events."""
from fractions import Fraction as F
from hashlib import sha256
from .. import g6_w2c_b3_offline_result as V
from .contract import SEED,equal
ARMS=V.ARMS

def replay(arm,axis,counter):
    V.require(arm in ARMS and axis in ('X','Y') and type(counter)is int and counter>=0,'ARTIFICIAL_CELL_COUNTER')
    domain=SEED+'/'+arm+'/'+axis;draws=[]
    def draw():
        i=counter+len(draws);v=int.from_bytes(sha256(domain.encode()+i.to_bytes(8,'big')).digest(),'big')>>96
        draws.append(dict(width=160,counter_before=i,counter_after=i+1,value_hex=hex(v),discarded_bits=96));return v
    token=draw();kind=('ZERO_R','ZERO_C','EVENT','EVENT')[token%4] if arm=='full_return' else 'EVENT'
    if kind!='ZERO_R':token^=draw()
    event=None
    if kind=='EVENT':
        if arm=='full_return':token^=draw()
        ratio=F((1<<24)+(token&8191)+1,(1<<26)+57);weight=F(7,11)*ratio
        raw=[0,1] if arm=='full_return' else [0,0];word=raw if arm=='full_return' else []
        event=dict(raw_word=raw,word=word,child=2,ratio=ratio,weight=weight,coefficient=weight/2,proposal=F(1,2),phase_i_power=2,circuit_time_word=word[::-1],provider_calls={'0':1,'1':1,'2':2} if word else {'2':2})
    end=counter+len(draws)
    trace=dict(schema='b3_ARTIFICIAL_consumption_v1',domain=domain,draws=draws,counter_after=end,bits_consumed=160*end,artificial_only=True,real_consumption_equivalence=False)
    return kind,event,trace,end

def trial(payload,record,task,run,counters,indices):
    local=record['local_trial_id'];axis=payload['axis'];V.require(axis in ('X','Y'),'TRIAL_AXIS')
    expected='/'.join((task['phase'],task['pass_name'],task['arm'],task['layer'],axis,str(indices[axis])))
    equal(local,expected,'LOCAL_TRIAL_SEQUENCE');equal(payload['trial_id'],local,'RESULT_TRIAL_ID')
    equal(record['global_trial_id'],run+'/'+task['condition']+'/'+local,'GLOBAL_TRIAL_MODE_ID')
    kind,event,tr,end=replay(task['arm'],axis,counters[axis])
    equal(record['domain'],tr['domain'],'SEED_DOMAIN');equal(record['consumption'],tr,'EXACT_CONSUMPTION_TRACE')
    equal(payload['kind'],kind,'SURROGATE_KIND')
    expected_evidence=dict(artificial=True,consumption=tr)
    ev_digest=V.event_hash(dict(kind=kind,event=event))
    equal(record['event_digest'],ev_digest,'EVENT_DIGEST')
    expected_angle=None
    if kind=='EVENT' and task['layer']=='B':
        expected_angle=dict(artificial=True,not_a_mathematical_enclosure=True,event_requery_surrogate=sha256(V.canonical(event).encode()).hexdigest(),actual_angle=None)
    if task['phase']=='profile' and indices[axis]>=16:
        equal(payload.get('retention'),'HASHED_FULL_DOCUMENTS_NOT_RETAINED','PROFILE_RETENTION')
        equal(payload.get('event_sha256'),ev_digest,'PROFILE_EVENT_DIGEST')
        equal(payload.get('evidence_sha256'),sha256(V.canonical(expected_evidence).encode()).hexdigest(),'PROFILE_EVIDENCE_DIGEST')
        equal(payload.get('angle_sha256'),sha256(V.canonical(expected_angle).encode()).hexdigest(),'PROFILE_ANGLE_DIGEST')
        V.require(not {'event','evidence','angle'} & set(payload),'PROFILE_DOCUMENT_SCOPE')
    else:
        equal(payload.get('event'),event,'EXACT_SURROGATE_EVENT')
        equal(payload.get('evidence'),expected_evidence,'RESULT_CONSUMPTION_EVIDENCE')
        equal(payload.get('angle'),expected_angle,'ANGLE_REQUERY_SURROGATE')
    counters[axis]=end;indices[axis]+=1
    return kind
