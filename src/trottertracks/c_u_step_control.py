"""Bounded Armijo steps for the unchanged PR reduced oracle.

No molecular I/O on import. Numerical admissions and value-resolution scales
remain engineering checks, not error bounds for the exact reduced objective.
"""
from dataclasses import dataclass, replace
import math
from . import c_pr_integrated as previous

small = previous.small
stable = previous.stable
VERSION = 'PR_recentered_bounded_offset_independent_Armijo_v1'


@dataclass(frozen=True)
class StepPolicy:
    max_trials: int = 32
    max_updates: int = 64
    contraction: float = .5
    c1: float = 1e-4
    alpha_max: float = 1.
    displacement_cap: float = 1.01
    accepted_alpha_growth: float = 2.

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.max_trials, self.max_updates)):
            raise ValueError('FINITE_TRIAL_AND_UPDATE_LIMITS')
        values = (self.contraction, self.c1, self.alpha_max,
                  self.displacement_cap, self.accepted_alpha_growth)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('FINITE_POSITIVE_STEP_POLICY')
        if not self.contraction < 1 or not self.c1 < 1 or self.accepted_alpha_growth < 1:
            raise ValueError('ARMIJO_STEP_POLICY')


STEP = StepPolicy()
DEFAULT = replace(previous.DEFAULT, max_points=STEP.max_trials + 1,
                  outer_maxiter=STEP.max_updates)
GATES = ('final_inner_J', 'final_independent_J', 'rank_nullspace',
         'full_numeric_rank', 'latest_pair_stability', 'block_nonincrease',
         'refinement_nonincrease')


def valid_point(p):
    """Check the existing oracle contract before using values or gradients."""
    small.previous._valid(p)
    if not all(p['qualification']['admission_gates'][k] for k in GATES):
        raise ValueError('QUALIFIED_POINT_HAS_FAILED_ADMISSION_GATE')
    profile = small.local.value_profile(p)
    if not all(math.isfinite(profile[k]) and profile[k] >= 0 for k in (
            'refinement_value_change', 'extended_binary64_value_difference',
            'independent_AD_value_difference')):
        raise ValueError('NONFINITE_VALUE_COMPARISON_PROFILE')
    return profile


def search(evaluate, initial, step=STEP, policy=DEFAULT, *, previous_alpha=None,
           record=lambda *a: None):
    """One direction, finite halving, provisional acceptance only.

    Rejected/unqualified trial gradients never define a new direction. Only
    J/stability qualification failures are rejectable; correctness errors escape.
    """
    import numpy as np
    base = valid_point(initial)
    direction = -initial['gradient']
    with np.errstate(over='ignore', invalid='ignore'):
        slope = float(initial['gradient'] @ direction)
        norm = float(np.linalg.norm(initial['gradient']))
    if not math.isfinite(slope) or not math.isfinite(norm):
        raise ValueError('NONFINITE_DESCENT_DIRECTION')
    result = dict(trials=0, accepted_U_updates=0, certified_descent=None,
                  true_reduced_gradient_qualification=None, value_error_bound=None)
    if slope >= 0:
        return initial, dict(result, status='NO_QUALIFIED_DESCENT_DIRECTION')
    alpha0 = min(step.alpha_max, step.displacement_cap / norm)
    if previous_alpha is not None:
        if not math.isfinite(previous_alpha) or previous_alpha <= 0:
            raise ValueError('INVALID_PREVIOUS_ACCEPTED_ALPHA')
        alpha0 = min(alpha0, step.accepted_alpha_growth * previous_alpha)
    result.update(alpha0=alpha0, initial_slope=slope)
    seen = set()
    qualified = 0
    unresolved_value = False
    for i in range(step.max_trials):
        alpha = alpha0 * step.contraction ** i
        eta = initial['eta'] + alpha * direction
        if not np.isfinite(eta).all():
            raise ValueError('NONFINITE_TRIAL_ETA')
        if alpha == 0 or np.array_equal(eta, initial['eta']) or eta.tobytes() in seen:
            return initial, dict(result, trials=i, status='STEP_ROUNDING_STALLED')
        seen.add(eta.tobytes())
        event = dict(trial=i, alpha=alpha, alpha0=alpha0, initial_slope=slope,
                     objective=None, gradient_used=False, accepted_U_updates=0)
        record('backtracking_trial_attempt', str(i), dict(event), dict(eta=eta, direction=direction))
        try:
            candidate = evaluate(eta)
        except stable.Unqualified as exc:
            if exc.status not in ('J_UNQUALIFIED', 'REDUCED_GRADIENT_UNQUALIFIED'):
                raise
            event.update(status=exc.status, decision='REJECT_NUMERICALLY_UNQUALIFIED')
            record('backtracking_trial', str(i), event, None)
            continue
        trial_profile = valid_point(candidate)
        if not np.array_equal(candidate['eta'], eta):
            raise ValueError('BACKTRACKING_TRIAL_ETA_IDENTITY')
        scale = small.local.comparison_scale(base, trial_profile, policy)
        bound = float(initial['value'] + step.c1 * alpha * slope)
        upper = float(candidate['value'] + scale)
        if not all(math.isfinite(x) for x in (scale, bound, upper)):
            raise ValueError('NONFINITE_ARMIJO_COMPARISON')
        qualified += 1
        passed = upper <= bound and upper < initial['value']
        decrease = float(initial['value'] - candidate['value'])
        unresolved_value |= 0 <= decrease <= scale
        event.update(status='PROVISIONAL_ARMIJO_PASS' if passed else 'ARMIJO_OR_VALUE_SCALE_REJECTED',
                     decision='ASSESS_BEFORE_ACCEPTANCE' if passed else 'REJECT_ARMIJO_OR_VALUE_SCALE',
                     objective=candidate['value'], Armijo_bound=bound,
                     comparison_scale=scale, objective_plus_comparison_scale=upper,
                     actual_decrease=decrease, gradient_used=False)
        record('backtracking_trial', str(i), dict(event), None)
        if passed:
            return candidate, dict(result, **{k: event[k] for k in (
                'status', 'alpha', 'alpha0', 'initial_slope', 'Armijo_bound',
                'comparison_scale', 'objective_plus_comparison_scale', 'actual_decrease')},
                trials=i + 1, strong_Wolfe_claim=False)
    status = ('BACKTRACKING_NUMERICAL_QUALIFICATION_EXHAUSTED' if not qualified else
              'VALUE_RESOLUTION_UNQUALIFIED' if unresolved_value else
              'BACKTRACKING_TRIAL_LIMIT_NO_ACCEPT')
    return initial, dict(result, trials=step.max_trials, status=status)


def trajectory(g, Jcold, Ustart, spec, physical, *, step=STEP, policy=DEFAULT,
               call=lambda k, l, f: f(), record=lambda *a: None):
    """Recenter only after independent terminal AND physical-budget checks.

    A new oracle always receives Jcold. Its two within-point refinements are
    unchanged. The final export is the last physically qualified accepted point.
    """
    import numpy as np
    if policy.max_points < step.max_trials + 1:
        raise ValueError('INSUFFICIENT_REGISTERED_ORACLE_POINT_BUDGET')
    zero = np.zeros(spec.rank * spec.norb * (spec.norb - 1) // 2)
    history, trials = [], []
    points = updates = optimization_updates = stalls = 0
    current = None
    previous_alpha = None

    def emit(prefix):
        return lambda kind, label, data, arrays: record(kind, prefix + label, data, arrays)

    def finish(status):
        return dict(status=status, history=history, trials=trials, terminal=current,
                    accepted_U_updates=updates, optimization_accepted_updates=optimization_updates,
                    physically_qualified_updates=updates, point_evaluations=points,
                    inner_solves=3 * points, refinement_solves=2 * points,
                    true_reduced_gradient_qualification=None, certified_descent=None,
                    actual_CA_achieved=None, downstream_performance=None, retry=0, resume=0)

    oracle = small.Oracle(g, Jcold, Ustart, spec, policy, call=call, record=emit('initial_'))
    try:
        current = oracle.evaluate(zero)
    except stable.Unqualified as exc:
        points = len(oracle.points)
        return finish(exc.status)
    points = len(oracle.points)
    valid_point(current)
    row = call('physical_assessment', 'initial', lambda: physical('initial', current['J'], current['U']))
    history.append(dict(id='initial', update=0, state=current['state'],
                        qualification=current['qualification'], physical=row))
    if not row['budget_PASS']:
        current = None
        return finish('INITIAL_DF_BUDGET_NOT_MET')
    for k in range(step.max_updates):
        if current['state']['first_order_met']:
            return finish('BOTH_FIRST_ORDER_MET')
        base_U = current['U']
        prefix = f'k{k:03d}_'
        oracle = small.Oracle(g, Jcold, base_U, spec, policy, call=call, record=emit(prefix))
        center = dict(current, eta=zero.copy(), label='0')
        oracle.points.append(center)
        oracle.cache[zero.tobytes()] = center
        candidate, decision = search(oracle.evaluate, center, step, policy,
                                     previous_alpha=previous_alpha, record=emit(prefix))
        points += len(oracle.points) - 1
        trials.append(dict(update_attempt=k, **decision))
        record('step_decision', str(k), trials[-1], None)
        if decision['status'] != 'PROVISIONAL_ARMIJO_PASS':
            return finish(decision['status'])
        saved = {}

        def independent_record(kind, label, data, arrays):
            if kind == 'independent_terminal_state':
                saved.update(arrays)
            emit(prefix)(kind, label, data, arrays)

        post = call('independent_terminal_assessment', prefix + candidate['label'],
                    lambda: small.assess(g, candidate, base_U, spec, policy, record=independent_record))
        if not post['J_numerically_qualified']:
            return finish(post['status'])
        optimization_updates += 1
        label = f'update{optimization_updates:03d}'
        row = call('physical_assessment', label, lambda: physical(label, candidate['J'], candidate['U']))
        history.append(dict(id=label, update=optimization_updates, state=post['state'],
                            qualification=candidate['qualification'], physical=row, step=decision))
        record('physical_candidate', label, history[-1], None)
        if not row['budget_PASS']:
            return finish('PROVISIONAL_CANDIDATE_DF_BUDGET_NOT_MET')
        current = dict(candidate, state=post['state'], eta=zero.copy(),
                       gradient=saved['physical_gradient_U'].copy())
        updates += 1
        previous_alpha = decision['alpha']
        record('accepted', label, dict(objective=current['value'], physically_qualified=True,
                                      gradient_recentered=True), dict(J=current['J'], U=current['U']))
        stalls = stalls + 1 if decision['actual_decrease'] <= (
            policy.stall_atol + policy.stall_rtol * abs(center['value'])) else 0
        if current['state']['first_order_met']:
            return finish('BOTH_FIRST_ORDER_MET')
        if stalls >= policy.stall_cycles:
            return finish('REGISTERED_OBJECTIVE_STALL')
    return finish('MAXIMUM_ACCEPTED_UPDATES')
