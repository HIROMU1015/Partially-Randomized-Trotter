"""Static controlled primitive accounting, not circuit compile/T synthesis."""
import math
import numpy as np
from .qc_onebody_repartition_pilot_v1 import PilotError


def frame_counts(spatial):
    """Adjacent spatial QR rotations, duplicated by spin; parity is explicit.

    Tiny entries are NOT dropped: this counts a conservative construction.
    Exact identity (after symbolic same-frame cancellation) needs no gates.
    Complex Givens are priced as one declared controlled primitive, with
    phase parameters separately visible; no hardware/T cost is asserted.
    """
    a = np.array(spatial, dtype=complex, copy=True)
    n = len(a); rotations = 0
    for col in range(n):
        for row in range(n-1, col, -1):
            x, y = a[row-1, col], a[row, col]
            if y == 0:
                continue
            radius = math.sqrt(abs(x)**2+abs(y)**2)
            G = np.array([[x.conjugate()/radius,y.conjugate()/radius],[-y/radius,x/radius]])
            a[[row-1,row], :] = G @ a[[row-1,row], :]
            rotations += 1
    phases = sum(np.angle(z) != 0 for z in np.diag(a))
    return {'controlled_Givens': 2*rotations, 'controlled_phase': 2*phases,
            'controlled_pair_phase': 0, 'fermionic_parity_links': 4*rotations}


def add(left, right, multiplier=1):
    return {k: int(left.get(k,0)+multiplier*right.get(k,0)) for k in
            ['controlled_Givens','controlled_phase','controlled_pair_phase','fermionic_parity_links']}


def sector_phase_coefficients(actor):
    if 'kind' not in actor:
        return actor['linear'],actor['pairs'],0.
    values=np.asarray(actor['eta']);mean=float(values.mean());centered=values-mean;particles=actor['particles']
    if actor['kind']=='onebody':
        return centered,np.zeros_like(actor['pairs']),mean*particles
    weight=actor['weight'];linear=weight*centered**2+2*weight*mean*particles*centered-actor['shift']
    scalar=weight*mean*mean*particles*particles
    linear_mean=float(linear.mean());linear=linear-linear_mean;scalar+=linear_mean*particles
    return linear,2*weight*np.outer(centered,centered),scalar


def phase_counts(actor):
    linear,pairs,_=sector_phase_coefficients(actor)
    return {'controlled_Givens': 0,
            'controlled_phase': int(np.count_nonzero(linear)),
            'controlled_pair_phase': int(np.count_nonzero(np.triu(pairs,1))),
            'fermionic_parity_links': 0}


def circuit_counts(actors, nodes, constant, queries):
    """Exact same-fragment/time fusion; adjacent basis sharing for ALL methods.

    Internal relative frames depend only on two cached orbital frames, not h.
    The query boundary is the same A frame, so repeated-query cost is computed
    in closed form without expanding millions of gates or queries.
    """
    # The known fixed-N scalar is removed equally for every method. A scalar
    # fragment needs no Gaussian circuit, even if its stored frame is dense.
    active=[]
    for j,t in nodes:
        counts=phase_counts(actors[j])
        if counts['controlled_phase']==0 and counts['controlled_pair_phase']==0:
            continue
        if active and active[-1][0]==j:active[-1]=(j,active[-1][1]+t)
        else:active.append((j,t))
    nodes=[(j,t) for j,t in active if t!=0]
    scalar=constant!=0 or any(sector_phase_coefficients(a)[2]!=0 for a in actors)
    if not nodes:
        return {'controlled_Givens':0,'controlled_phase':int(scalar),'controlled_pair_phase':0,'fermionic_parity_links':0}
    total = add({},{}); previous = None; cache = {}
    for j, t in nodes:
        u = actors[j]['orbital_frame'][::2,::2]
        # Coordinates of physical state in previous frame -> next frame.
        if previous is None:
            relative = u.conj().T
        elif previous[0] == j:
            relative = np.eye(len(u),dtype=complex)
        else:
            relative = u.conj().T @ previous[1]
        key = (previous[0] if previous else -1, j)
        if key not in cache:
            cache[key] = frame_counts(relative)
        total = add(total, cache[key]); total = add(total, phase_counts(actors[j]))
        previous = (j,u)
    exit_counts = frame_counts(previous[1]); total = add(total,exit_counts)
    first = nodes[0][0]; last = nodes[-1][0]
    if first != last:
        raise PilotError('QUERY_BOUNDARY_NOT_SAME_FRAGMENT')
    entry = frame_counts(actors[first]['orbital_frame'][::2,::2].conj().T)
    boundary_removed = add(add(entry,exit_counts),phase_counts(actors[first]))
    combined = add({k:v*queries for k,v in total.items()},boundary_removed,-(queries-1))
    if scalar:
        combined['controlled_phase'] += 1  # known controlled scalar over whole trial
    if any(x<0 for x in combined.values()):
        raise PilotError('NEGATIVE_CIRCUIT_ACCOUNTING')
    return combined


def repeats(overlap, failure):
    if not 0 < overlap <= 1:
        raise PilotError('INVALID_OVERLAP')
    if overlap >= 1:
        return 1
    return max(1,math.ceil(math.log(failure)/math.log1p(-overlap)))


def eligible(record, target, contract):
    threshold = target * contract['evaluation']['budget_fractions']['deterministic']
    overlap_floor = contract['evaluation']['minimum_overlap']
    if record['ambiguous']:
        return False, 'BRANCH_AMBIGUOUS'
    if abs(record['bias'])+record['numerical_allowance'] >= threshold:
        return False, 'BIAS_OR_NUMERICAL_BOUNDARY'
    if record['overlap']-1e-10 < overlap_floor:
        return False, 'OVERLAP_OR_NUMERICAL_BOUNDARY'
    return True, 'ELIGIBLE_DECLARED_DOUBLE_PRECISION_MARGIN'


def account(record, configurations, contract, target, price):
    ok, reason = eligible(record,target,contract)
    result = {'method':record['method'],'h_index':record['h_index'],'target_Ha':target,
              'price_id':price['id'],'eligible':ok,'reason':reason}
    if not ok:
        return result
    config = configurations[record['config']]
    eps_stat = target*contract['evaluation']['budget_fractions']['statistical']
    raw = record['experiments']; ampl = sum(abs(x['weight']) for x in raw)
    total = add({},{}); trials=0; queries=[]
    for exp in raw:
        q = max(1,math.ceil(math.pi/(exp['h']*(eps_stat/ampl))))
        r = repeats(min(1.,exp['overlap']),.01/len(raw))
        primitive = circuit_counts(config['actors'],exp['nodes'],config['constant'],q)
        total = add(total,primitive,r); trials += r
        queries.append({'h':exp['h'],'weight':exp['weight'],'coherent_queries':q,'repeat_trials':r})
    score = sum(total[k]*price[k] for k in ('controlled_Givens','controlled_phase','controlled_pair_phase'))
    rotations = total['controlled_phase']+total['controlled_pair_phase']+2*total['controlled_Givens']
    response = record['config']=='response_selected'
    corrected = record['method'].startswith('CS2:')
    result.update({'primitive_vector':total,'conditional_price_total':int(score),'state_preparations':trials,
                   'experiments':queries,'extrapolation_weight_L1':ampl,
                   'uniform_synthesis_angle_allowance':target*contract['evaluation']['budget_fractions']['synthesis']/max(1,rotations),
                   'information':{'gamma_required':response,'leading_coefficient_required':response or corrected,
                                  'coefficient':record['config'] if response or corrected else None,
                                  'acquisition_quantum_cost':'UNDETERMINED_NOT_ZERO_BY_DEFAULT'},
                   'cost_scope':'STATIC_CONTROLLED_PRIMITIVE_HEISENBERG_PROXY_IDEAL_INFORMATION_ORACLE_H;NOT_COMPILED_T_OR_FULL_QPE'})
    return result


def comparison(costs, contract):
    cells=[]
    for target in contract['evaluation']['energy_targets_Ha']:
        for price in contract['resources']['pricing']:
            eligible_rows = [r for r in costs if r['target_Ha']==target and r['price_id']==price['id'] and r['eligible']]
            candidate = [r for r in eligible_rows if r['method'] in ('S2:response_selected','CS2:response_selected')]
            baseline = [r for r in eligible_rows if r not in candidate]
            key=lambda r:(r['conditional_price_total'],r['method'],r['h_index'])
            R=min(candidate,key=key) if candidate else None; B=min(baseline,key=key) if baseline else None
            cell={'target_Ha':target,'price_id':price['id'],'selected_response':R,'strong_control':B,
                  'precision_selection':'POSTHOC_GRID_ORACLE_NOT_OPERATIONAL_SELECTOR'}
            if R is None or B is None:
                cell.update({'status':'UNDETERMINED_ELIGIBLE_GRID_INSUFFICIENT','ratio':None})
            else:
                ratio=R['conditional_price_total']/B['conditional_price_total']
                threshold=1-contract['evaluation']['registered_effect_fraction']
                cell.update({'ratio':ratio,'additional_information_budget':B['conditional_price_total']-R['conditional_price_total'],
                             'status':'CONDITIONAL_COMPONENT_GAIN' if ratio<threshold-1e-10 else 'NO_REGISTERED_COMPONENT_GAIN'})
            cells.append(cell)
    states=[r['status'] for r in cells]
    classification=('CONDITIONAL_COMPONENT_GAIN_IDEAL_INFORMATION' if all(x=='CONDITIONAL_COMPONENT_GAIN' for x in states)
                    else 'UNDETERMINED' if any(x.startswith('UNDETERMINED') for x in states)
                    else 'NO_UNIFORM_REGISTERED_COMPONENT_GAIN')
    return {'classification':classification,'cells':cells,'formal_research_GO':False,
            'exclusive_response_generator_advantage':False,'operational_precision_selection_certified':False,
            'full_ground_energy_total_advantage_certified':False,'new_GPT_review_started':False,'mandatory_stop':True}
