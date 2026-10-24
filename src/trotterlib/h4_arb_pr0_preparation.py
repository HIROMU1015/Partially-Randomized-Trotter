"""Selected PR0 C-target bridge. Real inputs: static export ONLY.

Numerical entry points, including the private kernel, accept tiny artificial
fixtures only. No real H4 dispatch, grant issuer or certificate acceptance.
"""
from fractions import Fraction as Q
from pathlib import Path

from . import h4_arb_bridge as B
from . import h4_certificate_control as C
from .h4_conditional_cng import extract

ROOT=Path(__file__).resolve().parents[2]
C_SOURCE='e80d7510880f8db993b777c627bb8e95b05a070d'
R0_SOURCE='84a326598b006384d13c7a354a3eec6881ee841c'
TID='93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97'
KIND='SELECTED_C_TARGET_EXACT_OPERANDS_V1'


def _word(label,n):
    C.require(type(label) is str and len(label)==n and not set(label)-set('IXYZ'),'PAULI_WORD')
    return label


def build_plan(model,join,*,domain='SAVED_PR0_NO_EXECUTION',fixture_id=None):
    """Exact metadata connection, no state action or numerical backend import."""
    C.require(domain in ('ARTIFICIAL','SAVED_PR0_NO_EXECUTION'),'PLAN_DOMAIN')
    case=model['cases']['PR'];cfg=case['config'];n=model['n_qubits'];q,r=cfg['q'],cfg['r']
    C.require(model['representation']=='pauli_jw' and model['qubit_order']=='qiskit_little_endian','PAULI_LAYOUT')
    C.require(type(n) is int and n>0 and type(q) is int and q>0 and type(r) is int and r>0 and cfg['method']=='PR','MODEL_SHAPE')
    C.require(join['method']=='PR' and (join['q'],join['r'])==(q,r) and len(join['events'])==q*r,'SELECTED_EVENTS_COUNT')
    if domain=='SAVED_PR0_NO_EXECUTION':
        C.require(n==8 and cfg==dict(T=.8,q=8,r=4,K=4,method='PR'),'FIXED_H4_PR0')
        C.require(join['trajectory_id']==TID and join['trajectory_index']==0,'PR0_IDENTITY')
        verifier=extract('src/trotterlib/rte_event_audit.py','verify_join',dict(digest=C.digest))
        verifier(join,expected_case='pauli_jw_PR')
        C.require(all(e['taylor_order']==0 for e in join['events']),'SAVED_PR0_ALL_ORDER0')
    else:C.require(type(fixture_id) is str and fixture_id.startswith('TOY_'),'ARTIFICIAL_FIXTURE')
    delta=B.exact_binary64(model['step_time'])
    blocks=[]
    for b in case['blocks']:
        coefficient=B.exact_binary64(b['coefficient'])
        blocks.append(dict(label=_word(b['label'],n),coefficient_binary64_exact=str(coefficient),
            angle_kind='EXACT_PRODUCT_OF_BINARY64_OPERANDS',angle_exact=str(coefficient*delta/2)))
    components={str(i):dict(label=_word(t['label'],n),sign=t['sign'],coefficient_abs_binary64_exact=str(B.exact_binary64(t['coefficient_abs'])))
                for i,t in enumerate(case['components'])}
    C.require(all(type(t['sign']) is int and t['sign'] in (-1,1) for t in components.values()),'COMPONENT_SIGNS')
    law_angles=[str(B.exact_binary64(a)) for a in case['law']['angles']]
    C.require(len(law_angles)==3,'C_SAVED_ROTATION_ANGLES')
    events=[]
    for i,e in enumerate(join['events']):
        order=e['taylor_order'];ids=[*e['product_component_ids'],e['rotation_component_id']]
        C.require(type(order) is int and order in (0,2,4) and len(e['product_component_ids'])==order,'EVENT_PRODUCT_COUNT')
        C.require((e['event_position'],e['outer_step_index'],e['inner_RTE_position'])==(i,i//r,i%r),'CHRONOLOGICAL_EVENTS')
        C.require(e['selected_component_ids']==ids and len(e['application_sequence'])==order+1,'EVENT_SELECTED_ORDER')
        for j,(cid,app) in enumerate(zip(ids,e['application_sequence'])):
            C.require(cid in components,'EVENT_COMPONENT_ID');term=components[cid]
            C.require((app['application_index'],app['component_id'],app['role'])==(j,cid,'product' if j<order else 'rotation'),'EVENT_APPLICATION_ORDER')
            C.require(app['coefficient_sign']==term['sign'] and B.exact_binary64(app['coefficient_abs'])==Q(term['coefficient_abs_binary64_exact']),'EVENT_SIGN_OPERAND')
            C.require(app['is_identity']==all(x=='I' for x in term['label']) and app.get('basis_id') is None
                and app.get('basis_hash') is None and app.get('df_fragment_id') is None and not app.get('basis_change_operations',[]),'NO_DF_BASIS_SUBSTITUTION')
        angle=B.exact_binary64(e['rotation_angle']);phase=B.exact_binary64(e['phase']['real'])
        C.require(str(angle)==law_angles[order//2] and phase==(-1)**(order//2) and B.exact_binary64(e['phase']['imag'])==0,'EVENT_ANGLE_ORDER_PHASE')
        events.append(dict(event_position=i,outer_step_index=i//r,inner_RTE_position=i%r,taylor_order=order,
            product_component_ids=list(e['product_component_ids']),rotation_component_id=e['rotation_component_id'],
            rotation_angle_kind='SAVED_BINARY64_VALUE',rotation_angle_exact=str(angle),order_phase_exact=str(phase)))
    operands=[str(B.exact_binary64(x)) for x in case['native_scalar_phases']]
    plan=dict(kind=KIND,domain=domain,fixture_id=fixture_id,n_system=n,axis='cosine',q=q,r=r,
        step_time_binary64_exact=str(delta),retained_blocks=blocks,components=components,events=events,
        law_rotation_angles_binary64_exact=law_angles,native_scalar_binary64_operands=operands,
        native_scalar_exact_sum=str(sum(map(Q,operands),Q(0))),B_saved_binary64_exact=str(B.exact_binary64(case['saved_normalization']['B'])),
        trajectory_id=join['trajectory_id'],C_SOURCE=C_SOURCE,R0_SOURCE=R0_SOURCE,
        event_record_hash=join.get('event_record_hash'),trajectory_record_hash=join.get('trajectory_record_hash'),
        event_application='P_l1 then P_l2 ... then signed rotation; order phase once; no extra product-sign phase',
        schedule='forward retained, selected events for step, reverse retained, native scalar; repeat q',
        PR_law_mean_used=False,probability_weights_applied=False,normalization_B_applied_to_target=False)
    validate_plan(plan)
    return plan


def validate_plan(p):
    C.require(p['kind']==KIND and p['domain'] in ('ARTIFICIAL','SAVED_PR0_NO_EXECUTION') and p['axis']=='cosine','SELECTED_PLAN_KIND_AXIS')
    n,q,r=p['n_system'],p['q'],p['r']
    C.require(type(n) is int and n>0 and type(q) is int and q>0 and type(r) is int and r>0,'PLAN_SHAPE')
    delta=B.encoded_parameter(p['step_time_binary64_exact'])
    C.require(not p['PR_law_mean_used'] and not p['probability_weights_applied'] and not p['normalization_B_applied_to_target'],'NO_MEAN_OR_TARGET_SCALE')
    C.require(B.encoded_parameter(p['B_saved_binary64_exact'])>0,'POSITIVE_SAVED_B')
    for b in p['retained_blocks']:
        _word(b['label'],n);coefficient=B.encoded_parameter(b['coefficient_binary64_exact'])
        C.require(b['angle_kind']=='EXACT_PRODUCT_OF_BINARY64_OPERANDS' and Q(b['angle_exact'])==coefficient*delta/2,'RETAINED_EXACT_PRODUCT')
    for term in p['components'].values():
        _word(term['label'],n);C.require(type(term['sign']) is int and term['sign'] in (-1,1),'COMPONENT_SIGNS')
        C.require(B.encoded_parameter(term['coefficient_abs_binary64_exact'])>=0,'COMPONENT_ABS_OPERAND')
    C.require(len(p['events'])==q*r and len(p['law_rotation_angles_binary64_exact'])==3,'FINITE_SELECTED_SCHEDULE')
    for i,e in enumerate(p['events']):
        k=e['taylor_order'];C.require(type(k) is int and k in (0,2,4),'EVENT_ORDER')
        C.require((e['event_position'],e['outer_step_index'],e['inner_RTE_position'])==(i,i//r,i%r),'PLAN_EVENT_ORDER')
        C.require(len(e['product_component_ids'])==k and all(c in p['components'] for c in [*e['product_component_ids'],e['rotation_component_id']]),'PLAN_COMPONENTS')
        C.require(e['rotation_angle_kind']=='SAVED_BINARY64_VALUE' and B.encoded_parameter(e['rotation_angle_exact'])==B.encoded_parameter(p['law_rotation_angles_binary64_exact'][k//2]),'ROTATION_BINARY64')
        C.require(Q(e['order_phase_exact'])==(-1)**(k//2),'ORDER_PHASE_ONCE')
    operands=[B.encoded_parameter(x) for x in p['native_scalar_binary64_operands']]
    C.require(sum(operands,Q(0))==Q(p['native_scalar_exact_sum']),'EXACT_NATIVE_SCALAR_SUM')
    if p['domain']=='SAVED_PR0_NO_EXECUTION':
        C.require((n,q,r,p['trajectory_id'])==(8,8,4,TID) and all(e['taylor_order']==0 for e in p['events']),'REAL_PR0_STATIC_SCOPE')
    return p


def _guard(p,raw,bits,fixture_id):
    C.require(p.get('domain')=='ARTIFICIAL' and p.get('n_system') in (2,3),'NO_REAL_H4_ARB_TARGET_ACTION')
    C.require(type(fixture_id) is str and fixture_id.startswith('TOY_') and p.get('fixture_id')==fixture_id,'ARTIFICIAL_FIXTURE_BINDING')
    C.require(type(bits) is int and bits in B.PRECISIONS,'FINITE_PRECISION_SCHEDULE')
    validate_plan(p);C.require(len(raw)==1<<p['n_system'],'RAW_STATE_DIMENSION')
    for pair in raw:
        C.require(type(pair) in (list,tuple) and len(pair)==2,'RAW_COMPLEX_PAIR')
        for x in pair:B.exact_binary64(x)


def _pauli_map(label,sign):
    mapping=[]
    for col in range(1<<len(label)):
        row=col;turn=0 if sign==1 else 2
        for q,s in enumerate(reversed(label)):
            bit=(col>>q)&1
            if s in 'XY':row^=1<<q
            if s=='Y':turn+=1 if not bit else -1
            elif s=='Z' and bit:turn+=2
        mapping.append((row,turn%4))
    return mapping


def _pauli_action(vector,mapping,backend):
    out=[None]*len(vector)
    for z,(row,turn) in zip(vector,mapping):
        real,imag=z.real,z.imag
        out[row]=z if turn==0 else backend.acb(-imag,real) if turn==1 else -z if turn==2 else backend.acb(imag,-real)
    return out


def _target_artificial(p,raw,bits,fixture_id):
    _guard(p,raw,bits,fixture_id)  # even private entry rejects H4 before Arb import
    backend=B._backend()
    with backend.ctx.workprec(bits):
        nu=sum((B.exact_binary64(x)**2+B.exact_binary64(y)**2 for x,y in raw),Q(0))
        C.require(nu>0,'POSITIVE_RAW_NORM')
        bra=[backend.complex_ball(x) for x in raw];vector=list(bra);error=Q(0);maps={};trig={}
        C.require(all(z.real.is_exact() and z.imag.is_exact() for z in vector),'EXACT_INITIAL_BINARY64_STATE')
        def pauli(v,label,sign):
            key=(label,sign)
            if key not in maps:maps[key]=_pauli_map(*key)
            return _pauli_action(v,maps[key],backend)
        def rotate(v,e,label,sign,angle):
            if angle not in trig:
                a=backend.real(angle);trig[angle]=(a.cos(),backend.acb(0,-a.sin()))
            cosine,sine=trig[angle];pv=pauli(v,label,sign)
            return B._recenter([cosine*z+sine*w for z,w in zip(v,pv)],e,backend)
        scalar=backend.phase(backend.real(Q(p['native_scalar_exact_sum'])))
        for step in range(p['q']):
            for b in p['retained_blocks']:vector,error=rotate(vector,error,b['label'],1,Q(b['angle_exact']))
            for e in p['events'][step*p['r']:(step+1)*p['r']]:
                for cid in e['product_component_ids']:
                    term=p['components'][cid];vector=pauli(vector,term['label'],term['sign'])
                term=p['components'][e['rotation_component_id']]
                vector,error=rotate(vector,error,term['label'],term['sign'],Q(e['rotation_angle_exact']))
                if Q(e['order_phase_exact'])==-1:vector=[-z for z in vector]
            for b in reversed(p['retained_blocks']):vector,error=rotate(vector,error,b['label'],1,Q(b['angle_exact']))
            vector,error=B._recenter([scalar*z for z in vector],error,backend)
        dot=sum((a.conjugate()*z for a,z in zip(bra,vector)),backend.acb(0))
        # Exact bra: |<raw|x-m>| <= sqrt(nu)*e, including both real/imag components.
        transfer=(backend.real(nu).sqrt().upper()*backend.real(error)).upper()
        denominator=backend.real(nu);C.require(denominator.lower()>0,'NORM_DENOMINATOR_EXCLUDES_ZERO')
        signal=(dot.real+backend.arb(0,transfer))/denominator;C.require(signal.is_finite(),'FINITE_SELECTED_SIGNAL')
        endpoints=backend.endpoints(signal)
        record=dict(status='ARTIFICIAL_SELECTED_C_TARGET_ENCLOSURE_ONLY',fixture_id=fixture_id,precision_bits=bits,
            interval=endpoints,interval_width_exact=str(Q(endpoints['upper'])-Q(endpoints['lower'])),raw_norm_exact=str(nu),
            state_l2_error_upper_exact=str(error),inner_product_error_upper_exact=str(backend.dyadic(transfer)),
            selected_events=len(p['events']),retained_rotations=2*p['q']*len(p['retained_blocks']),
            product_applications=sum(e['taylor_order'] for e in p['events']),native_scalar_applications=p['q'],
            PR_law_mean_used=False,normalization_B_applied_to_target=False,state_renormalized=False,
            formal_certificate_accepted=False,real_H4_state_actions=0)
        return signal,record


def evaluate_target_artificial(plan,raw,*,bits=64,fixture_id):
    return _target_artificial(plan,raw,bits,fixture_id)


def compare_artificial(plan,program,raw,*,budget_exact,bits=64,fixture_id):
    _guard(plan,raw,bits,fixture_id);B._artificial_guard(program,raw,bits,fixture_id)
    C.require(program['n_system']==plan['n_system'] and program['axis']=='cosine','COMMON_RAY_COSINE_LAYOUT')
    _,target=evaluate_target_artificial(plan,raw,bits=bits,fixture_id=fixture_id)
    compiled,record=B.evaluate_artificial(program,raw,bits=bits,fixture_id=fixture_id)
    C.require(record['raw_norm_exact']==target['raw_norm_exact'],'COMMON_EXACT_NU')
    _,rho=B._residual_interval(compiled,target['interval'],B_saved_exact=plan['B_saved_binary64_exact'],budget_exact=budget_exact,bits=bits)
    return dict(status='ARTIFICIAL_TWO_ACTION_LOCAL_ENCLOSURE_NO_H4',target=target,compiled=record,rho=rho,
        artificial_state_actions=2,real_H4_state_actions=0,B_applied_exactly_once=True,formal_certificate_accepted=False,LOCAL_BUDGET_MET=False)


def execution_request(*_args,**_kwargs):
    raise PermissionError('PREPARATION_ONLY_PR0_REAL_H4_AND_GRANTS_DISABLED')
