"""Synthetic documents only. Deliberately does not implement a G7 law."""
from fractions import Fraction as F
from . import codec as C, result as R

SOURCE = 'ARTIFICIAL_SOURCE_G7_F3_SHAPED_V1'


def context(trial_id):
    return dict(input_id='ARTIFICIAL_INPUT_V1',policy_id='ARTIFICIAL_PRECISION_V1',
                source_id=SOURCE,preflight_id='ARTIFICIAL_PREFLIGHT_V1',trial_id=trial_id,sigma=1)


def build(bits, *, trial_id, arm):
    start = len(bits.trace)
    bits_before = bits.consumed_bits
    token = bits.bits(160)
    branch = ('RAW_NONREDUCED','COIN_REJECT','ACCEPTED','ACCEPTED')[token%4] if arm == 'full_return' else 'ACCEPTED'
    raw = (0,0) if branch == 'RAW_NONREDUCED' or arm != 'full_return' else (0,1)
    stages = ['order']
    if arm == 'full_return':
        for _ in raw:
            token ^= bits.bits(160);stages.append('label')
        if branch != 'RAW_NONREDUCED':
            token ^= bits.bits(160);stages.append('acceptance')
        if branch == 'ACCEPTED':
            token ^= bits.bits(160);stages.append('child')
    else:
        token ^= bits.bits(160);stages.append('child')
    event = None
    if branch == 'ACCEPTED':
        ratio=F((1<<24)+(token&8191)+1,(1<<26)+57);weight=ratio*F(7,11)
        w=raw if arm=='full_return' else ()
        event=dict(word=w,raw_word=raw,child=2,ratio=ratio,weight=weight,coefficient=weight/2,
                   proposal=F(1,2),phase_i_power=2,provider_calls={0:1,1:1,2:2} if w else {2:2},
                   helper=1,event_unitary='i^phase exp(-i*sigma*atan(ratio)*Q_child) Q(word)',
                   circuit_time_word=tuple(reversed(w)))
    draws=bits.trace[start:]
    consumption=dict(schema='b3_ARTIFICIAL_adapter_consumption_v1',domain=bits.domain,
                     draws=draws,counter_after=bits.counter,bits_consumed=bits.consumed_bits,bits_consumed_before=bits_before,
                     artificial_only=True,real_consumption_equivalence=False)
    trace=dict(draws=[dict(stage=s,width=d['width'],value=int(d['value_hex'],16)) for s,d in zip(stages,draws)],
               raw_word=raw,degree=len(raw),bits_consumed=160*len(draws),draw_count=len(draws),
               completed_bit_map_calls_by_stage={s:stages.count(s) for s in ('order','label','acceptance','child')},
               extra_random_draws_for_audit=0,fresh_IID_verified=False)
    ids=context(trial_id)
    if arm=='full_return':
        data=dict(schema='g6_f3_trial_v1',**ids,run_completion_or_confidence_certified=False)
        if event is None:data.update(kind='ZERO',reason=branch,trace=trace,is_valid_zero_trial=True)
        else:data.update(kind='EVENT',event=event,trace=trace)
    else:
        data=dict(schema='g7_observation_candidate_v1',**ids,arm=arm,branch=branch,
                  trace=consumption,packet_calls=0,group_index=0,event=event)
    return C.Document.create(data),consumption


def adapt(record, *, trial_id, arm, layer):
    data=record.read();expected=context(trial_id)
    if arm=='full_return' and layer=='B':
        return R.adapt_f3(record,expected=expected)
    if arm=='full_return':
        branch='ACCEPTED' if data['kind']=='EVENT' else data['reason']
        observation=C.Document.create(dict(schema='g7_observation_candidate_v1',**expected,
            arm=arm,branch=branch,trace=data['trace'],packet_calls=0 if branch=='RAW_NONREDUCED' else 1,
            group_index=None,event=data.get('event')))
        return R.adapt_g7(data.get('event'),observation=observation,expected=expected,arm=arm)
    return R.adapt_g7(data['event'],observation=record,expected=expected,arm=arm)


def synthetic_angle(adapted):
    """F3-shaped enclosure, no atan or other mathematical calculation."""
    e=adapted['event'];source=C.restore(adapted['codec'],source_identity=SOURCE).read()
    lo,hi=F(1,8),F(1,8)+F(1,10**8);radius=(hi-lo)/2;mid=(hi+lo)/2
    record=C.Document.create(dict(schema='g6_f3_angle_enclosure_v1',ratio=e['ratio'],sigma=1,
        root_bits=32,terms=32,tau=F(1,10**6),
        context={**{k:source[k] for k in (*R.IDS,'sigma')},'event_id':adapted['event_identity'],
                 'parent':e['word'],'child':e['child']},sqrt_interval=None,half_argument_interval=None,
        endpoint_series_evidence=None,phi_interval=(lo,hi),representative=mid,
        representative_error_upper=radius,signed_phi_interval=(lo,hi),signed_representative=mid,
        precision_check=dict(name='representative_radius',left=radius,operator='<=',right=F(1,10**6),passed=True),
        native_rotation_acquired=False,physical_provider_certified=False,
        basis={'scope':'ARTIFICIAL_SHAPE_ONLY_NOT_AN_ATAN_ENCLOSURE'}))
    query=dict(word=e['word'],child=e['child'])
    if source['schema']=='g7_observation_candidate_v1' and source['arm']!='full_return':
        query.update(group_index=source['group_index'],raw_word=e['raw_word'])
    if source['schema']=='g7_observation_candidate_v1' and source['arm']!='full_return':
        data=record.read();data['context']=dict(event_id=C.digest_bytes(C.driver_canonical(e).encode()),
            input={'id':'ARTIFICIAL_INPUT_V1'},policy={'id':'ARTIFICIAL_PRECISION_V1'},preflight=source['preflight_id'])
        record=C.Document.create(data)
    return query,record
