"""One opt-in small Armijo trial with numerical admission, not convergence.

The same three-solve PR oracle is unchanged. A trial is accepted only after an
independent state re-evaluation; unknown inner bias is never called an enclosure.
"""
from dataclasses import dataclass, asdict, replace
import math
from . import c_pr_local_direction as local
from .c_pr_coefficient_point import compare

previous = local.previous
stable = local.stable
Oracle = previous.Oracle
DEFAULT = replace(previous.DEFAULT, outer_maxiter=1, max_points=2)
VERSION = 'PR_one_small_scale_qualified_Armijo_step_v1'


@dataclass(frozen=True)
class StepPolicy:
    fraction: float = 1/8
    c1: float = 1e-4

    def __post_init__(self):
        if not 0 < self.fraction < 1 or not 0 < self.c1 < 1:
            raise ValueError('SMALL_STEP_POLICY')


STEP = StepPolicy()


def propose(evaluate, initial, step=STEP, policy=DEFAULT, *, record=lambda *a:None):
    """Exactly one trial, no backtracking and no acceptance before assessment."""
    import numpy as np
    previous._valid(initial)
    direction = -initial['gradient']
    slope = float(initial['gradient'] @ direction)
    if not math.isfinite(slope):
        raise ValueError('NONFINITE_SMALL_STEP_SLOPE')
    if slope >= 0:
        return initial, dict(status='NO_NONZERO_INITIAL_DIRECTION', trials=0, accepted_U_updates=0)
    if initial['value'] <= 0:
        return initial, dict(status='NONPOSITIVE_INITIAL_F_NO_SCALE', trials=0, accepted_U_updates=0)
    alpha_scale = float(initial['value']/(-slope))
    alpha = alpha_scale * step.fraction
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('NONFINITE_SMALL_STEP_SCALE')
    eta = initial['eta'] + alpha * direction
    if np.array_equal(eta, initial['eta']):
        return initial, dict(status='STEP_ROUNDING_STALLED', trials=0, accepted_U_updates=0)
    decision = dict(trials=1, alpha=alpha, alpha_scale=alpha_scale, fraction=step.fraction,
        initial_slope=slope, objective=None, gradient_used=False, accepted_U_updates=0,
        value_error_bound=None, true_reduced_gradient_qualification=None, certified_descent=None)
    record('small_trial_attempt', '0', dict(decision), dict(eta=eta, direction=direction))
    try:
        p = evaluate(eta)
    except stable.Unqualified as e:
        decision.update(status=e.status, decision='REJECT_NUMERICALLY_UNQUALIFIED')
        record('small_trial', '0', dict(decision), None)
        return initial, decision
    previous._valid(p)
    if not np.array_equal(p['eta'], eta):
        raise ValueError('SMALL_TRIAL_ETA_IDENTITY')
    bound = float(initial['value'] + step.c1 * alpha * slope)
    scale = local.comparison_scale(local.value_profile(initial), local.value_profile(p), policy)
    # This is a conservative engineering comparison, not a bound on min_J F.
    passed = p['value'] + scale <= bound and p['value'] + scale < initial['value']
    decision.update(status='PROVISIONAL_SMALL_ARMIJO_PASS' if passed else 'SMALL_STEP_REJECTED_ARMIJO_OR_VALUE_SCALE',
        decision='ASSESS_BEFORE_ACCEPTANCE' if passed else 'REJECT_ARMIJO_OR_VALUE_SCALE',
        objective=p['value'], gradient_used=True, Armijo_bound=bound, comparison_scale=scale,
        objective_plus_comparison_scale=float(p['value']+scale),
        actual_decrease=float(initial['value']-p['value']),
        qualified_gradient_l2=float(np.linalg.norm(p['gradient'])), strong_Wolfe_claim=False)
    record('small_trial', '0', dict(decision), None)
    return (p if passed else initial), decision


def assess(g, p, base_U, spec, policy=DEFAULT, *, record=lambda *a:None,state_evaluator=None):
    """Independent same-array terminal state; zero further J solves/updates."""
    import numpy as np
    row, gj, gu, chart, binary = (state_evaluator or stable.state)(g, p['J'], p['U'], spec, policy,
        base_U=base_U, eta=p['eta'])
    checks = dict(
        value=compare(row['components']['objective'], p['value'], atol=policy.value_atol, rtol=policy.value_rtol),
        chart_gradient=compare(chart, p['gradient'], atol=policy.gradient_atol, rtol=policy.gradient_rtol))
    same_arrays = dict(J=p['J'], U=p['U'], eta=p['eta'], gradient_J=gj,
        physical_gradient_U=gu, chart_gradient=chart, binary64_gradient_U=binary)
    data = dict(state=row, consistency=checks, inherited_qualification=p['qualification'],
        stage='provisional_trial_before_committing_acceptance', new_J_solves=0,
        true_reduced_gradient_qualification=None, certified_descent=None)
    record('independent_terminal_state', p['label'], data, same_arrays)
    if not all(c['PASS'] for c in checks.values()) or row['orthogonality_maxabs'] > policy.orthogonality:
        raise ValueError('INDEPENDENT_TERMINAL_STATE_MISMATCH')
    qualified = row['J_stationarity']['status'] == 'MET'
    data.update(status='INDEPENDENT_TERMINAL_J_MET' if qualified else 'POST_UPDATE_J_UNQUALIFIED',
        J_numerically_qualified=qualified,
        U_first_order_met=row['U_stationarity']['status'] == 'MET',
        first_order_met=row['first_order_met'], rigorous_stationarity=False)
    record('independent_terminal_assessment', p['label'], dict(data), None)
    return data


def optimize(g, J, U, spec, policy=DEFAULT, *, call=lambda k,l,f:f(), record=lambda *a:None):
    """Initial + one trial, at most one accepted U update, no outer optimizer."""
    import numpy as np
    oracle = Oracle(g, J, U, spec, policy, call=call, record=record)
    zero = np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    initial = None
    post = None
    try:
        initial = oracle.evaluate(zero)
    except stable.Unqualified as e:
        terminal = oracle.points[0]
        decision = dict(status=e.status, trials=0, accepted_U_updates=0)
    else:
        terminal = initial
        if initial['state']['first_order_met']:
            decision = dict(status='BOTH_FIRST_ORDER_MET', trials=0, accepted_U_updates=0)
        else:
            candidate, decision = propose(oracle.evaluate, initial, policy=policy, record=record)
            if decision['status'] == 'PROVISIONAL_SMALL_ARMIJO_PASS':
                post = call('independent_terminal_assessment', candidate['label'],
                    lambda:assess(g, candidate, U, spec, policy, record=record))
                if post['J_numerically_qualified']:
                    terminal = candidate
                    decision.update(status='QUALIFIED_SMALL_ARMIJO_STEP_ACCEPTED',
                        accepted_U_updates=1, decision='ACCEPT_AFTER_INDEPENDENT_ASSESSMENT')
                    record('accepted_U_update', candidate['label'], dict(decision), dict(eta=candidate['eta']))
                else:
                    decision.update(status=post['status'], decision='REJECT_POST_ASSESSMENT')
    updates = decision['accepted_U_updates']
    result = dict(**decision, outer_method=VERSION, policy=asdict(policy), step_policy=asdict(STEP),
        terminal_label=terminal['label'], terminal_kind=('accepted_U_update' if updates else
            'qualified_initial_no_update' if initial else 'unqualified_initial_diagnostic'),
        terminal=post['state'] if updates else terminal['state'], terminal_qualification=terminal['qualification'],
        independent_terminal_assessment=post,
        accepted_labels=([initial['label']] if initial else [])+([terminal['label']] if updates else []),
        distinct_U_points=len(oracle.points), inner_solves=3*len(oracle.points),
        refinement_solves=2*len(oracle.points), last_trial_label=oracle.points[-1]['label'],
        point_qualifications=[dict(label=p['label'], **p['qualification']) for p in oracle.points],
        outer_BFGS_invocations=0, outer_solver_invocations=0, new_ffsim_fits=0,
        DF_budget=None, PR_performance=None, RC2_continuation=0, retry=0, resume=0,
        rigorous_reduced_gradient=False, rigorous_stationarity=False)
    record('small_step_terminal', 'final', result, None)
    return terminal['J'], terminal['U'], result
