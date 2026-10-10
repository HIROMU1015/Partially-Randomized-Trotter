"""Result-prior matched finite-confidence resource policy, no backend calls."""
from fractions import Fraction as F
from .g7_generator import ARMS,make_generator
from .g7_reference import reference_events
from .g8_bounds import accepted_cap,acceptance_upper,log_upper,ceil
from .g9_p5 import P5Closed
from .g9_native import cts_events,native_ir,cost

P=(F(1,5),F(3,10),F(1,2));X=F(5,7)


def generators():
    return [make_generator(P,X,5,a) for a in ARMS]+[P5Closed(P,X)]


def plan(g=None,cts=None):
    eta=rho=F(1,10**12);alpha=F(49,22000);epsilon=F(1,10**6)
    common_bias=2*(8*rho+8*(1+rho)*2*epsilon)
    remaining=F(1,200)-common_bias
    B=g.B.hi if g else sum(e['coefficient'] for e in cts)
    if B>=8:raise ArithmeticError('common analytic representation norm envelope violated')
    kappa=(1+rho)**2/(1-eta)**7
    m2=kappa*B*(g.U.hi if g and g.arm=='full_return' else B)
    W=(1+rho)*B/(1-eta)**7
    N=ceil(log_upper(2/alpha)*(2*m2/remaining**2+4*W/(3*remaining)))
    z=acceptance_upper(g) if g and g.arm=='full_return' else F(1)
    return {'N_per_axis':N,'m2_upper':m2,'range_upper':W,'common_bias_upper':common_bias,'remaining':remaining,
        'alpha_axis':alpha,'estimator_failure_22_axes':22*alpha,'resource_failure_per_row':F(1,11000),
        'resource_failure_11_rows':F(1,1000),'total_failure':'1/20','exact_provider_delta':'0',
        'accepted_call_cap_two_axes':accepted_cap(2*N,z,t=10),'acceptance_upper':z,'hard_attempt_cap_two_axes':2*N,
        'uses_signal_or_full_normalizer_for_local_budget':False}


def row(arm,events,plan,cache,helper=False):
    from .g9_matrix import circuit_error,event_operator,target
    import numpy as np
    q=sum(e['proposal'] for e in events);m2=sum(e['proposal']*e['weight']**2 for e in events)
    W=max(e['weight'] for e in events)
    if not 0<q<=1 or m2>plan['m2_upper'] or W>plan['range_upper']:
        raise ArithmeticError('reference/budget bound failure')
    mean=sum(float(e['coefficient'])*event_operator(e) for e in events)
    residual=float(np.linalg.norm(mean-target(P,X),2))
    if residual>1e-10:raise ArithmeticError('ideal operator mean/phase failure')
    result=[];per={k:F(0) for k in ('T','CX','1Q')};maxideal=maxreal=0.;maxT=0
    for e in events:
        gates=native_ir(e,helper);c=cost(gates,cache)
        ideal=circuit_error(e,gates,None,helper);real=circuit_error(e,gates,cache,helper)
        if ideal>1e-10 or real>float(c['strict_error_upper'])+1e-10:
            raise ArithmeticError('controlled native operator/relative phase failure')
        maxideal=max(maxideal,ideal);maxreal=max(maxreal,real);maxT=max(maxT,c['T'])
        for k in per:per[k]+=e['proposal']*c[k]
        result.append({'event':e,'native_ir':gates,'cost':c,'ideal_phase_reference_error':ideal,'realized_reference_error':real})
    N=plan['N_per_axis'];C=plan['accepted_call_cap_two_axes']
    return {'arm':arm,'implementation':'generic_helper_diagnostic' if helper else 'direct_primary',
        'primary':not helper,'workspace_beyond_system':2 if helper else 1,'events':result,'budget':plan,
        'reference_acceptance':q,'reference_m2':m2,'reference_range':W,'ideal_mean_matrix_residual_diagnostic':residual,
        'max_ideal_native_phase_error_diagnostic':maxideal,'max_realized_operator_error_diagnostic':maxreal,
        'event_strict_error_analytic_bound':'2/1000000','matrix_tolerance_is_diagnostic_not_bias_budget':True,
        'per_trial_native_cost':per,'two_axis_expected_native_cost':{k:2*N*v for k,v in per.items()},
        'expected_accepted_calls':2*N*q,'T_prep_readout_affine_coefficient':2*N*q,
        'common_1Q_outer_prep_readout_two_axes':5*N*q,
        'registered_worst_event_T':maxT,'accepted_tail_T_upper':C*maxT,'hard_attempt_T_upper':2*N*maxT,
        'all_event_reference_not_injected_into_local_generator':True,
        'whole_circuit_global_optimization':False,'physical_quantum_shots_executed':0}


def event_lists(gs):
    return [list(g.reference_events()) if isinstance(g,P5Closed) else list(reference_events(g)) for g in gs]
