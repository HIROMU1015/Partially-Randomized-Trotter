"""Opt-in local directional probes, not another optimizer/backtracking run.

Re-solve J at three signed scales, retain all existing numerical admissions,
and report finite differences with explicit empirical value-resolution scales.
No true reduced-gradient error bound or accepted U update is manufactured.
"""
from dataclasses import replace
import math
from . import c_pr_guarded_step as previous

stable=previous.stable
DEFAULT=replace(previous.DEFAULT,max_points=7)
FRACTIONS=(1/8,1/32,1/128)
VERSION='PR_three_symmetric_local_direction_probes_v1'


def collect(evaluate,zero,*,record=lambda *a:None):
    import numpy as np
    initial=None;probes=[]
    try:initial=evaluate(zero)
    except stable.Unqualified as e:
        return dict(status=e.status,initial=None,probes=probes)
    previous._valid(initial)
    direction=-initial['gradient'];slope=float(initial['gradient']@direction)
    if not math.isfinite(slope):raise ValueError('NONFINITE_LOCAL_DIRECTIONAL_SLOPE')
    if slope>=0:
        return dict(status='NO_NONZERO_INITIAL_DIRECTION',initial=initial,probes=probes)
    if initial['value']<=0:
        return dict(status='NONPOSITIVE_INITIAL_F_NO_SCALE',initial=initial,probes=probes)
    scale=float(initial['value']/(-slope))
    if not math.isfinite(scale) or scale<=0:raise ValueError('NONFINITE_LOCAL_SCALE')
    record('local_direction','initial',dict(F0=initial['value'],reported_slope=slope,alpha_scale=scale,
        fractions=list(FRACTIONS),true_directional_derivative=None),dict(direction=direction))
    for level,fraction in enumerate(FRACTIONS):
        h=scale*fraction;pair=dict(level=level,fraction=fraction,h=h,sides={})
        probes.append(pair)
        for sign in (1,-1):
            eta=initial['eta']+sign*h*direction
            if np.array_equal(eta,initial['eta']):
                return dict(status='LOCAL_STEP_ROUNDING_STALLED',initial=initial,probes=probes,slope=slope,scale=scale)
            label=str(level)+('_plus' if sign==1 else '_minus')
            record('local_probe_attempt',label,dict(h=h,sign=sign,fraction=fraction),dict(eta=eta))
            try:
                p=evaluate(eta);previous._valid(p)
                if not np.array_equal(p['eta'],eta):raise ValueError('LOCAL_PROBE_ETA_IDENTITY')
                pair['sides'][str(sign)]=dict(status='NUMERICAL_PILOT_ADMITTED',point=p)
                record('local_probe',label,dict(status='NUMERICAL_PILOT_ADMITTED',point_label=p['label']),None)
            except stable.Unqualified as e:
                pair['sides'][str(sign)]=dict(status=e.status,point=None)
                record('local_probe',label,dict(status=e.status,objective_used=False,gradient_used=False),None)
                if e.status not in ('J_UNQUALIFIED','REDUCED_GRADIENT_UNQUALIFIED'):
                    return dict(status=e.status,initial=initial,probes=probes,slope=slope,scale=scale)
    return dict(status='REGISTERED_LOCAL_PROBES_COMPLETE',initial=initial,probes=probes,slope=slope,scale=scale)


def value_profile(p):
    """Observed scales only. Agreement/refinement cannot bound inner bias."""
    state=p['state'];ref=p['qualification']['objective_refinement']
    return dict(value=p['value'],
        refinement_value_change=abs(ref['after']-ref['before']),
        extended_binary64_value_difference=state['value_same_array']['maxabs'],
        independent_AD_value_difference=state['binary64_AD_value_same_array']['maxabs'],
        unknown_inner_solution_value_error=None,certified_value_error_bound=None)


def comparison_scale(a,b,policy=DEFAULT):
    """Sensitivity scale, not an interval enclosing the exact min_J F."""
    registered=policy.value_atol+policy.value_rtol*max(abs(a['value']),abs(b['value']))
    observed=[a[k] for k in ('refinement_value_change','extended_binary64_value_difference','independent_AD_value_difference')]
    observed += [b[k] for k in ('refinement_value_change','extended_binary64_value_difference','independent_AD_value_difference')]
    return max(registered,*observed)


def analyse(experiment,policy=DEFAULT):
    initial=experiment['initial'];out=[];prev=None
    if initial is None:
        return dict(status='INITIAL_UNQUALIFIED_NO_LOCAL_DIAGNOSTICS',rows=out,
            true_reduced_gradient_qualification=None,accepted_U_updates=0)
    base=value_profile(initial)
    for pair in experiment['probes']:
        row={k:pair[k] for k in ('level','fraction','h')};sides=pair['sides']
        row['side_statuses']={s:x['status'] for s,x in sides.items()}
        if not all(s in sides and sides[s]['point'] is not None for s in ('1','-1')):
            row.update(status='UNRESOLVED_MISSING_QUALIFIED_PAIR',central_FD=None,FD_sensitivity_scale=None)
            prev=None;out.append(row);continue
        plus,minus=(value_profile(sides[s]['point']) for s in ('1','-1'));h=pair['h']
        delta=plus['value']-base['value'];change_scale=comparison_scale(base,plus,policy)
        fd=(plus['value']-minus['value'])/(2*h)
        fd_scale=comparison_scale(plus,minus,policy)/(2*h)
        signal=bool(abs(plus['value']-minus['value'])>comparison_scale(plus,minus,policy))
        row.update(status='NUMERICAL_PAIR_DIAGNOSTICS_AVAILABLE',plus=plus,minus=minus,
            forward_difference=(plus['value']-base['value'])/h,
            backward_difference=(base['value']-minus['value'])/h,central_FD=fd,
            reported_slope=experiment['slope'],FD_minus_reported_slope=fd-experiment['slope'],
            FD_sensitivity_scale=fd_scale,FD_signal_above_comparison_scale=signal,
            interlevel_FD_change=None if prev is None else abs(fd-prev),
            forward_delta_F=delta,forward_comparison_scale=change_scale,
            forward_observation='EMPIRICAL_DECREASE' if delta < -change_scale else 'EMPIRICAL_INCREASE' if delta>change_scale else 'UNRESOLVED_VALUE_SCALE',
            truncation_error_bound=None,unknown_inner_solution_error=None,true_gradient_error_bound=None)
        prev=fd;out.append(row)
    complete=len(out)==len(FRACTIONS) and all(r['central_FD'] is not None for r in out)
    return dict(status='LOCAL_DIRECTIONAL_DIAGNOSTICS_AVAILABLE' if complete else 'LOCAL_DIAGNOSTICS_PARTIAL',
        probe_status=experiment['status'],initial=base,rows=out,
        reported_slope=experiment.get('slope'),alpha_scale=experiment.get('scale'),
        interpretation='Comparison scales are empirical engineering sensitivity scales,NOT error enclosures;FD truncation and inner bias remain unbounded',
        true_reduced_gradient_qualification=None,certified_descent=None,accepted_U_updates=0,
        outer_BFGS_invocations=0,optimizer_iterations=0)
