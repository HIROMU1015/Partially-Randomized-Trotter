"""Descriptor and saved-value audit only. No matrix, trig or Arb evaluation."""
from collections import Counter,defaultdict
from fractions import Fraction as F
from .c_per_word_population_range import _digest


def _phase(gates):
    phases=dict(id=F(),x=F(1,2),z=F(1,2),h=F(1,2),sx=F(1,4),sxdg=F(-1,4),
                s=F(1,4),sdg=F(-1,4),rz=F(),ry=F(),p=F(),cx=F(1,4))
    return (sum((phases[g['name']] for g in gates),F()),
            sum((F(float.fromhex(g['params_hex'][0]))/2 for g in gates if g['name']=='p'),F()))


def audit(saved,plan,cert,C,R):
    """Reconcile every stored contribution without generating a new bound."""
    from . import c_cx_structural_blocks as old
    old.verify_saved_plan(saved,C,R)
    if plan!=old.partition(saved,max_support=plan['max_support'],guard=plan['guard'],exact_clifford=plan['exact_clifford']):
        raise ValueError('OLD_COMPLETE_PARTITION_REPLAY')
    if (cert['plan_id']!=plan['plan_id'] or cert['left_id']!=_digest(C) or cert['right_id']!=_digest(R)):
        raise ValueError('SAVED_WITNESS_IDENTITY')
    groups=defaultdict(lambda:dict(count=0,upper=F(),extracted_pi=F(),gauge_pi=F(),rad=F()))
    counts=Counter();positions=defaultdict(list);bodies={}
    for i,f in enumerate(plan['factors']):
        if f['kind']=='shared_CX':continue
        body=dict(kind=f['kind'],support=list(range(len(f['support']))),left=f['left'],right=f['right'])
        key=_digest(body);counts[key]+=1;positions[key].append(i);bodies[key]=body
        w=cert['block_witnesses'][key];pa,ra=_phase(f['left']);pb,rb=_phase(f['right'])
        if f['kind']=='oneq':
            if w['sign'] not in (1,-1):raise ValueError('SU2_CENTER_SIGN')
            gauge=F(w['sign']==-1)
        else:
            if not 0<=w['root_index']<w['root_order']:raise ValueError('CENTRAL_ROOT')
            gauge=-F(2*w['root_index'],w['root_order'])
        if F(w['delta_pi_exact'])!=pa-pb+gauge or F(w['delta_rad_exact'])!=ra-rb:
            raise ValueError('PHASE_ACCOUNTING')
        d=groups[f['kind']];d['count']+=1;d['upper']+=F(w['upper_exact'])
        d['extracted_pi']+=pa-pb;d['gauge_pi']+=gauge;d['rad']+=ra-rb
    if dict(counts)!=cert['block_multiplicities']:raise ValueError('BLOCK_MULTIPLICITY')
    if sum((d['upper'] for d in groups.values()),F())!=F(cert['blocks_upper_exact']):raise ValueError('BLOCK_TOTAL')
    dc=Counter()
    for e in saved['expansions']:
        for key,d in e['definitions'].items():dc[key]+=d['multiplicity']
    if dict(dc)!=cert['definition_multiplicities']:raise ValueError('DEFINITION_MULTIPLICITY')
    translation=sum((F(cert['definition_witnesses'][k]['upper_exact'])*v for k,v in dc.items()),F())
    if translation!=F(cert['translation_upper_exact']):raise ValueError('TRANSLATION_TOTAL')
    expansion_rad=F(saved['expansions'][0]['phase_exact_rad'])-F(saved['expansions'][1]['phase_exact_rad'])
    pi=sum((d['extracted_pi']+d['gauge_pi'] for d in groups.values()),F())
    rad=expansion_rad+sum((d['rad'] for d in groups.values()),F())
    if pi!=F(cert['global_delta_pi_exact']) or rad!=F(cert['global_delta_rad_exact']):raise ValueError('GLOBAL_PHASE_TOTAL')
    high=[]
    for key in sorted(counts,key=lambda k:F(cert['block_witnesses'][k]['upper_exact'])*counts[k],reverse=True):
        w=cert['block_witnesses'][key]
        if F(w['upper_exact'])<F('1e-8'):continue
        examples=[]
        for index in positions[key]:
            f=plan['factors'][index]
            examples.append(dict(index=index,support=f['support'],window=f.get('window_pieces'),
                left=f['left'],right=f['right'],neighbors=[dict(index=j,kind=plan['factors'][j]['kind'],
                    support=plan['factors'][j].get('support',plan['factors'][j].get('gate',{}).get('qubits')))
                    for j in range(max(0,index-2),min(len(plan['factors']),index+3)) if j!=index]))
        high.append(dict(body_id=key,kind=bodies[key]['kind'],multiplicity=counts[key],upper_exact=w['upper_exact'],
            stored_contribution_exact=str(F(w['upper_exact'])*counts[key]),witness=w,occurrences=examples))
    phase=[]
    for original,e in zip((C,R),saved['expansions']):
        definition_rad=sum((F(d['phase_exact_rad'])*d['multiplicity'] for d in e['definitions'].values()),F())
        if F(float.fromhex(original['phase_hex']))+definition_rad!=F(e['phase_exact_rad']):raise ValueError('EXPANSION_PHASE')
        phase.append(dict(saved_phase_hex=original['phase_hex'],saved_phase_exact_rad=str(F(float.fromhex(original['phase_hex']))),
            primitive_definition_phase_exact_rad=str(definition_rad),expanded_phase_exact_rad=e['phase_exact_rad']))
    return dict(kind='SAVED_VALUES_AND_SYMBOLIC_ACCOUNTING_ONLY',version='block_witness_audit_v1',
        plan_id=plan['plan_id'],group_totals={k:{a:(v if a=='count' else str(v)) for a,v in d.items()} for k,d in groups.items()},
        high_contribution_grouping_threshold='1/100000000 (report grouping only,not a scientific allowance)',
        high_contributions=high,exact_multiplicities_match=True,exact_partition_replay_match=True,
        exact_translation_total_matches=True,phase_origins=phase,expanded_delta_rad_exact=str(expansion_rad),
        global_pi_exact=str(pi),global_rad_exact=str(rad),stored_global_phase_upper_exact=cert['global_phase_upper_exact'],
        stored_blocks_upper_exact=cert['blocks_upper_exact'],stored_translation_upper_exact=cert['translation_upper_exact'],
        new_H4_operator_bound=None,new_H4_Arb_evaluations=0,new_partition_H4_evaluations=0,
        interpretation='individual local upper endpoints are not local lower bounds;scalar gauge and local terms form one compensated representation,not independent physical error measurements')
