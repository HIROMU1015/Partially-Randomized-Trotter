"""Canonical finite-mean resource vector and common Bernstein shot accounting.

No signal/oracle, IS, simulation, synthesis or molecule access.
"""
from fractions import Fraction as F
import hashlib,json
import mpmath as mp
from .numeric import alpha_enclosure,ivf,enclosure,record,synthesis_key,validate_saved


def native_cost(circuit,cache,epsilon):
    t,cx,one,error=0,0,0,F(0)
    rows=[]
    for g in circuit:
        rows.append({'name':g.name,'wires':g.wires,'angle_key':g.angle.key if g.angle else None,'phase':g.phase})
        if g.name=='RZ':
            saved=validate_saved(cache[synthesis_key(g.angle,epsilon)],g.angle,epsilon)
            t+=saved['T_count'];one+=saved['one_qubit_count'];error+=F(saved['strict_operator_error_upper'])
        elif g.name=='CX':cx+=1
        elif g.name=='GLOBAL':pass  # exact mathematical phase retained, not a gate
        elif g.name in ('H','X','Y','Z','S','Sdg'):one+=1
        else:raise ValueError('unreviewed native gate')
    return {'T':t,'CX':cx,'1Q':one,'strict_event_error_upper':error,
            'IR_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
            'IR_gate_count':len(circuit)}


def canonical_profile(events,circuits,cache,epsilon,controlled):
    if len(events)!=len(circuits) or not events:raise ValueError('event/support mismatch')
    bounds=[alpha_enclosure(e) for e in events]
    alphas=[(lo+hi)/2 for lo,hi in bounds];B=sum(alphas)
    if B<=0 or any(a<=0 for a in alphas):raise ArithmeticError('invalid canonical support')
    probs=[a/B for a in alphas]
    assert sum(probs)==1
    coefficient_error=sum(max(abs(a-lo),abs(a-hi)) for a,(lo,hi) in zip(alphas,bounds))
    expected={'T':F(0),'CX':F(0),'1Q':F(0)};bias=coefficient_error;operator_error=coefficient_error;rows=[]
    for e,circuit,p,a,bound in zip(events,circuits,probs,alphas,bounds,strict=True):
        cost=native_cost(circuit,cache,epsilon)
        for k in expected:expected[k]+=p*cost[k]
        operator_error+=a*cost['strict_event_error_upper']
        # Approximated CRZ factors need not retain exact ancilla-0 identity.
        # Use the full joint-unitary observable/channel bound <= 2*delta,
        # rather than silently treating the circuit as an exact controlled U.
        bias+=2*a*cost['strict_event_error_upper']
        rows.append({'label':e.label,'a':str(e.a),'b':str(e.b),'label_probability':str(e.label_probability),
            'word':e.word,'rotation':e.rotation,'rotation_sign':e.rotation_sign,'phase_i_power':e.phase,
            'complement':e.complement,'ideal_coefficient':record(bound),'implemented_coefficient':str(a),
            'canonical_probability_exact':str(p),
            'native_cost':{k:str(v) if isinstance(v,F) else v for k,v in cost.items()}})
    Blo,Bhi=sum(lo for lo,hi in bounds),sum(hi for lo,hi in bounds)
    return {'ideal_B':record((Blo,Bhi)),'ideal_B_squared':record((Blo*Blo,Bhi*Bhi)),
        'implemented_B':str(B),'implemented_weight_second_moment':str(B*B),
        'corrected_weight_range':str(B),'E_native_cost':{k:str(v) for k,v in expected.items()},
        'workspace_qubits_beyond_2_system':int(controlled),'coefficient_L1_bias_upper':str(coefficient_error),
        'coefficient_and_strict_synthesis_bias_upper':str(bias),'events':rows,
        'coefficient_and_joint_operator_error_upper':str(operator_error),
        'coherent_measurement_synthesis_factor':2,
        'sampling':'exact rational normalization of midpoint coefficients; canonical only; no IS/PAI',
        'physical_circuit_cost_metric':'native IR after shared exact cancellation, additive primitive Clifford+T counts; no post-synthesis whole-string optimizer'}


def confidence_budget(profile,task):
    B=F(profile['implemented_B']);bias=F(profile['coefficient_and_strict_synthesis_bias_upper'])
    if F(profile['coefficient_L1_bias_upper'])>F(task['coefficient_bias_cap']):
        raise ArithmeticError('coefficient interval width exceeds frozen numerical cap')
    remaining=F(task['epsilon_axis'])-bias
    if remaining<=0:return {'status':'INFEASIBLE_BIAS','bias_upper':str(bias),'sufficient_shots_per_axis':None}
    alpha=F(task['alpha_axis'])
    if not 0<alpha<1:raise ValueError('invalid failure allocation')
    # |X-E X| <= 2B and Var(X) <= E X^2 <= B^2, without exact signal.
    bound=(2*ivf(B*B)+ivf(F(4,3))*ivf(B)*ivf(remaining))*mp.iv.ln(ivf(2/alpha))/ivf(remaining*remaining)
    upper=enclosure(bound)[1];shots=-(-upper.numerator//upper.denominator)
    if shots>task['shot_cap_per_axis']:
        return {'status':'INFEASIBLE_SHOT_CAP','required_shots_per_axis':shots,'sufficient_shots_per_axis':None}
    c={k:F(v) for k,v in profile['E_native_cost'].items()}
    return {'status':'ELIGIBLE_COMMON_FINITE_CONFIDENCE_TASK','sufficient_shots_per_axis':shots,
        'sufficient_shots_total':2*shots,'epsilon_stat_axis_lower':str(remaining),'bias_upper':str(bias),
        'variance_upper':str(B*B),'centered_range_upper':str(2*B),
        'G_T':str(2*shots*c['T']),'G_CX':str(2*shots*c['CX']),
        'G_1Q_with_Hadamard_preparation_readout':str(shots*(2*c['1Q']+5)),
        'state_preparation_T_CX':0,'readout_1Q_Re':2,'readout_1Q_Im':3,
        'exact_signal_used':False,'statistical_estimation_shots_executed':0}


def interval_ratio(numerator,denominator):
    lo,hi=F(numerator['lo']),F(numerator['hi']);a,b=F(denominator['lo']),F(denominator['hi'])
    if a<=0:return {'status':'ZERO_OR_UNRESOLVED_BASELINE','ratio':None}
    out=(lo/b,hi/a)
    return {'status':'STRICTLY_LOWER' if out[1]<1 else 'STRICTLY_HIGHER' if out[0]>1 else 'OVERLAP_OR_EQUAL',
            'ratio':record(out),'materiality_or_research_GO':False}
