"""Q3 direct model MSE for the opt-in total estimator Z1+T1 (no MC, exact rationals).

Reuses B's evaluate/_bound (count tables, closure certificate, coarse and
partial-state bounds) unchanged, and adds for the TOTAL estimator only:
  * W0 N4 first-unsafe update-width bound (lightweight, O(M) after tables);
  * an exact state DP in which zero/tie are transitions, not absorbed losses,
    with a state-dependent bound for dropped states and exact E[E], E[E^2];
  * min over completed VALID upper bounds; disagreement -> UNKNOWN.
Proven real-arithmetic inequalities are kept apart from the declared binary64
transition allowance. The MSE is conditional on the supplied raw means, the
supplied target and the IID exact-affine count law; it is not a physical
total-error guarantee, sufficient N, Minkowski budget or scientific contract.
"""
from copy import deepcopy
from fractions import Fraction
import math

from . import rpe_deterministic_risk as D
from . import rpe_total_decoder as T

K, S, F, a = D.K, D.S, D.F, D.a
PI_LO = Fraction(math.pi)                              # binary64 pi < real pi
PI_UP = Fraction(math.nextafter(math.pi, math.inf))    # > real pi
TRANSITION_ALLOWANCE = Fraction(1, 10**12)             # rad per transition, declared (not certified)
DIVISION_RELATIVE = Fraction(1, 2**53)                 # round-to-nearest of fl(-theta/h)
SUBNORMAL = Fraction(1, 2**1074)
VERIFIED = 'VERIFIED_FINITE_BINARY64_SUPPORT'
PREREQUISITES = ('MISSING_STATISTICAL_INPUT', 'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN',
                 'PHYSICAL_MEANS_UNQUALIFIED')
ALLOWANCE = dict(per_transition_rad=a.exact_value(TRANSITION_ALLOWANCE),
    status='DECLARED_BINARY64_ENGINEERING_ALLOWANCE_NOT_CERTIFIED',
    proven_part='REAL_ARITHMETIC_UPDATE_WIDTH_PI_OVER_2M_AND_PRINCIPAL_GUARD',
    division='|fl(x)-x| <= 2^-53|x| + 2^-1074 for fl(-theta/h)',
    pi_enclosure='float(pi) < pi < nextafter(float(pi), inf)')


def circular_upper(x, y):
    """Upper bound of the real circular distance of two reals in [-pi_up, pi_up]."""
    d = abs(Fraction(x) - Fraction(y))
    return min(d, 2*PI_UP - d)


def division_slack(h):
    return PI_UP/h*DIVISION_RELATIVE + SUBNORMAL


def energy_loss_cap(radius, *, h, theta, loss_upper):
    """Squared-energy-error cap when the final phase is within `radius` of theta.

    Real-arithmetic statement, valid ONLY under the principal guard
    |theta| + radius < pi (no +-pi crossing, linear = circular distance).
    """
    if abs(theta) + radius < PI_LO:
        return min(loss_upper, (radius/h + division_slack(h))**2), True
    return loss_upper, False


def loss_range(h, target):
    """B's resolved-loss upper L_D and |E_hat| <= fl(pi_f/h) for the total output."""
    edge = math.pi/h
    if not math.isfinite(edge) or not math.isfinite(math.nextafter(edge, math.inf)):
        return None, None
    upper = a.exact_scalar(math.nextafter(edge, math.inf)); ee = a.exact_scalar(target)
    return max((upper-ee)**2, (-upper-ee)**2), a.exact_scalar(edge)


def tables_for(entry, options):
    """The same count tables as rpe_deterministic_risk._bound (same arguments)."""
    prepared = entry['prepared']; h = float(a.exact_scalar(prepared['h']))
    target = entry['evaluation_target']['energy_ha']; radii = options['phase_radii']
    return [D.round_distribution(dict(m=r['m'], raw=r['mean']['raw'], **{k: r['shots'][k] for k in ('N_X', 'N_Y')}),
        h=h, target_energy=target, radius=radii[r['m']], probability_rule=options['probability_rule'],
        maximum_outcomes=options['maximum_round_outcomes']) for r in prepared['rows']]


def lemma_bound(tables, partial, *, h, target):
    """W0 N4: U = J_M + sum_f P(F=f) l_f. Valid for the TOTAL estimator only.

    Needs B's good-closure certificate (all-good prefix = oracle). f=0,1 keep
    L_D. For f>=2 the start radius is the exact max distance of round f-1 good
    oracle estimates, then pi(2^(1-f)-2^-M) plus the declared allowance.
    """
    M = len(tables)-1
    if (partial.get('decoder_certificate') or {}).get('status') != VERIFIED:
        return dict(status='UNKNOWN', MSE_upper=None, reason='GOOD_CLOSURE_CERTIFICATE_REQUIRED')
    if partial.get('terminal_good_MSE_contribution') is None or partial.get('first_unsafe') is None:
        return dict(status='UNKNOWN', MSE_upper=None, reason='PARTIAL_COARSE_TERMS_UNAVAILABLE')
    hh = Fraction(h); theta = -hh*Fraction(target); L_D, _ = loss_range(h, target)
    total = a.exact_scalar(partial['terminal_good_MSE_contribution']); caps = []
    for f, row in enumerate(partial['first_unsafe']):
        pf = a.exact_scalar(row['exact']); item = dict(f=f, probability=a.exact_value(pf))
        good = [o['oracle_estimate'] for o in tables[f-1]['outcomes'] if o['good']] if f else []
        if f <= 1 or not good:
            cap = L_D; item.update(reason='F_LE_1_OR_NO_GOOD_PREVIOUS_STATE_KEEPS_L_D', principal_guard=None)
        else:
            start = max(circular_upper(v, theta) for v in good)
            radius = start + PI_UP*(Fraction(2, 2**f)-Fraction(1, 2**M)) + (M-f+1)*TRANSITION_ALLOWANCE
            cap, guard = energy_loss_cap(radius, h=hh, theta=theta, loss_upper=L_D)
            item.update(start_radius=a.exact_value(start), radius=a.exact_value(radius), principal_guard=guard,
                        reason=None if guard else 'PRINCIPAL_GUARD_FAILED_KEEPS_L_D')
        total += pf*cap; item['loss_cap'] = a.exact_value(cap); caps.append(item)
    return dict(status='TOTAL_ESTIMATOR_FIRST_UNSAFE_LEMMA_BOUND', MSE_upper=a.exact_value(total),
        terminal_good_MSE_contribution=deepcopy(partial['terminal_good_MSE_contribution']), caps=caps,
        loss_upper=a.exact_value(L_D), applies_to=T.ESTIMATOR, partial_decoder_loss_family=False)


def _square_range(lo, hi):
    if lo <= 0 <= hi:
        return Fraction(0), max(lo*lo, hi*hi)
    return min(lo*lo, hi*hi), max(lo*lo, hi*hi)


def total_state_dp(tables, *, h, target, scope):
    """Exact Z1+T1 state DP; zero holds and tie choices remain states.

    Mass is integer over a common denominator. A state dropped by a declared
    scope keeps its binary64 value s; its future loss is bounded by
    d(s,theta)+pi(2^-m-2^-M)+allowance under the principal guard, else L_D.
    The flag records whether the ORIGINAL partial decoder would be unresolved.
    """
    import numpy as np
    if not isinstance(scope, dict) or not scope.get('source'):
        raise ValueError('Explicit total-state scope/SOURCE required')
    maximum_states = S._integer(scope.get('maximum_frontier_states'), 'maximum_frontier_states')
    maximum_pairs = S._integer(scope.get('maximum_transition_pairs'), 'maximum_transition_pairs')
    M = len(tables)-1; hh = Fraction(h); E = a.exact_scalar(target); theta = -hh*E
    L_D, edge = loss_range(h, target); slack = division_slack(hh)
    out = dict(status='UNKNOWN', estimator=T.ESTIMATOR, MSE_lower=None, MSE_upper=None, MSE_exact=None,
        rounds=[], scope=deepcopy(scope), numerical='COMMON_DENOMINATOR_INTEGER_MASSES',
        state_equivalence='(PARTIAL_UNRESOLVED_FLAG, IDENTICAL_BINARY64_HEX)', transition_allowance=deepcopy(ALLOWANCE))
    if L_D is None:
        out['reason'] = 'DECODER_ENERGY_NUMERICAL_RANGE'; return out
    res = dict(mass=Fraction(0), loss=Fraction(0), lo=Fraction(0), hi=Fraction(0), sq_lo=Fraction(0),
               sq_hi=Fraction(0), guard_failed=Fraction(0), flagged=Fraction(0), events=0)

    def drop(flag, state, mass, completed):
        if state is None:
            cap, guard, lo, hi = L_D, False, -edge, edge
        else:
            s = Fraction(float.fromhex(state))
            R = PI_UP*(Fraction(1, 2**completed)-Fraction(1, 2**M)) + (M-completed)*TRANSITION_ALLOWANCE
            cap, guard = energy_loss_cap(circular_upper(s, theta)+R, h=hh, theta=theta, loss_upper=L_D)
            lo, hi = -edge, edge
            if abs(s) + R < PI_LO:   # the final phase stays in [s-R, s+R]
                lo, hi = max(-edge, -(s+R)/hh-slack), min(edge, -(s-R)/hh+slack)
        sq = _square_range(lo, hi)
        res['mass'] += mass; res['loss'] += mass*cap; res['lo'] += mass*lo; res['hi'] += mass*hi
        res['sq_lo'] += mass*sq[0]; res['sq_hi'] += mass*sq[1]; res['events'] += 1
        res['guard_failed'] += 0 if guard else mass; res['flagged'] += mass if flag else 0

    states = {(False, None): 1}; denominator = 1; pairs = 0; worst = None
    for m, table in enumerate(tables):
        if table.get('status') != 'ENUMERATED' or table['m'] != m or table['q'] != 1 << m:
            out['reason'] = 'INCOMPLETE_CONSECUTIVE_COUNT_TABLES'; return out
        den = table['denominator']; grouped = {}
        for item in table['outcomes']:
            phase = item['principal']; weight = item['weight']
            if (not math.isfinite(phase) or not -math.pi <= phase < math.pi
                    or type(weight) is not int or weight < 0):
                out['reason'] = 'INVALID_COUNT_SUPPORT'; return out
            key = (bool(item['zero']), float(phase).hex())
            grouped[key] = grouped.get(key, 0) + weight
        if type(den) is not int or den <= 0 or sum(grouped.values()) != den:
            out['reason'] = 'COUNT_MASS_NOT_CONSERVED'; return out
        groups = [(zero, float.fromhex(phase), weight) for (zero, phase), weight in grouped.items() if weight]
        phases = np.array([v[1] for v in groups]); zeros = np.array([v[0] for v in groups], dtype=bool)
        previous = denominator; denominator *= den; new = {}; zero_mass = tie_mass = 0
        for (flag, state), mass in sorted(states.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            if pairs + len(groups) > maximum_pairs:
                drop(flag, state, Fraction(mass, previous), m-1); continue
            if m:
                estimate = float.fromhex(state)
                values, ties = T.step(phases, np.full(len(phases), estimate), table['q'], zeros)
                if not np.all(np.isfinite(values)):
                    out['reason'] = 'NONFINITE_TOTAL_DECODER_TRANSITION'; return out
                if np.any(values < -math.pi) or np.any(values > math.pi):
                    out['reason'] = 'TOTAL_DECODER_OUTSIDE_PRINCIPAL_RANGE'; return out
                move = np.abs(values-estimate); move = np.minimum(move, 2*math.pi-move)
                excess = float(np.max(move-math.pi/table['q']))
                worst = excess if worst is None else max(worst, excess)
            else:
                values = T.initial(phases, zeros); ties = np.zeros(len(groups), dtype=bool)
            pairs += len(groups)
            for i, (zero, _, weight) in enumerate(groups):
                contribution = mass*weight; tie = bool(ties[i])
                key = (flag or zero or tie, float(values[i]).hex())
                new[key] = new.get(key, 0) + contribution
                if zero:
                    zero_mass += contribution
                elif tie:
                    tie_mass += contribution
        if m < M and len(new) > maximum_states:
            ordered = sorted(new.items(), key=lambda kv: (-kv[1], str(kv[0])))
            for (flag, state), mass in ordered[maximum_states:]:
                drop(flag, state, Fraction(mass, denominator), m)
            new = dict(ordered[:maximum_states])
        states = new
        if Fraction(sum(states.values()), denominator) + res['mass'] != 1:
            out['reason'] = 'STATE_MASS_NOT_CONSERVED'; return out
        out['rounds'].append(dict(m=m, grouped_count_outcomes=len(groups), retained_states=len(states),
            transition_pairs_cumulative=pairs,
            zero_hold_probability_tracked=a.exact_value(Fraction(zero_mass, denominator)),
            tie_argmin_probability_tracked=a.exact_value(Fraction(tie_mass, denominator)),
            residual_probability=a.exact_value(res['mass'])))
    if worst is not None and worst > float(TRANSITION_ALLOWANCE):
        out['reason'] = 'DECLARED_TRANSITION_ALLOWANCE_EXCEEDED'; out['max_observed_excess_rad'] = worst
        return out
    mse = m1 = m2 = eta = flagged = Fraction(0)
    for (flag, state), mass in states.items():
        w = Fraction(mass, denominator); e = a.exact_scalar(-float.fromhex(state)/h); loss = (e-E)**2
        mse += w*loss; m1 += w*e; m2 += w*e*e
        if flag:
            flagged += w
        else:
            eta += w*loss
    r = res['mass']; complete = r == 0
    lower, upper = mse, mse + res['loss']
    mean = (m1 + res['lo'], m1 + res['hi']); second = (m2 + res['sq_lo'], m2 + res['sq_hi'])
    bias = (mean[0]-E, mean[1]-E); bias_sq = _square_range(*bias)
    variance = (max(Fraction(0), lower-bias_sq[1]), upper-bias_sq[0])
    if complete and (mse != bias_sq[0] + (m2-m1*m1) or bias_sq[0] != bias_sq[1]):
        out['reason'] = 'MSE_BIAS_VARIANCE_IDENTITY_FAILED'; return out
    out.update(status='COMPLETE_TOTAL_STATE_DP' if complete else 'PARTIAL_TOTAL_STATE_DP_WITH_STATE_RESIDUAL_BOUND',
        MSE_lower=a.exact_value(lower), MSE_upper=a.exact_value(upper), MSE_exact=a.exact_value(mse) if complete else None,
        retained_MSE_contribution=a.exact_value(mse), residual_probability=a.exact_value(r),
        residual_loss_contribution_upper=a.exact_value(res['loss']), residual_dropped_states=res['events'],
        residual_principal_guard_failed_probability=a.exact_value(res['guard_failed']),
        moments=dict(status='EXACT' if complete else 'INTERVAL_FROM_STATE_RESIDUAL',
            mean_energy=[a.exact_value(v) for v in mean], second_moment=[a.exact_value(v) for v in second],
            bias=[a.exact_value(v) for v in bias], bias_squared=[a.exact_value(v) for v in bias_sq],
            variance=[a.exact_value(m2-m1*m1)]*2 if complete else [a.exact_value(v) for v in variance],
            meaning='DESIGN_DIAGNOSTIC_BIAS_AND_VARIANCE_OF_TOTAL_ESTIMATOR_NOT_QUALIFICATION'),
        partial_decoder=dict(resolved_MSE_contribution=a.exact_value(eta) if complete else None,
            resolved_MSE_contribution_lower=a.exact_value(eta),
            unresolved_probability=a.exact_value(flagged) if complete else None,
            unresolved_probability_bounds=[a.exact_value(flagged + res['flagged']), a.exact_value(flagged + r)]),
        any_zero=a.exact_value(1-math.prod(1-Fraction(sum(o['weight'] for o in t['outcomes'] if o['zero']), t['denominator'])
                                           for t in tables)),
        transition_pairs=pairs, max_observed_transition_excess_rad=worst,
        loss_upper=a.exact_value(L_D), applies_to=T.ESTIMATOR)
    return out


def _value(record, key):
    value = record.get(key)
    return None if value is None else a.exact_scalar(value)


def _total(entry, partial, options):
    out = dict(status='UNKNOWN', estimator=T.ESTIMATOR, rules=deepcopy(T.RULES), MSE_upper=None, MSE_lower=None,
        RMSE_upper_display=None, bounds={}, moments=None, q3=None, MC_calls=0, sufficient_N=None,
        global_RMSE_bound=None, G=None, scientific_energy_qualification='UNKNOWN',
        transition_allowance=deepcopy(ALLOWANCE), physical_total_error_guarantee=False)
    if partial.get('reason') in PREREQUISITES:
        out['reason'] = partial['reason']; return out
    model = options.get('model_loss_contract') or {}
    if model.get('adopted') is not True or not model.get('source'):
        out['reason'] = 'ALL_EVENT_LOSS_NOT_ADOPTED'; return out
    prepared = entry['prepared']; h = float(a.exact_scalar(prepared['h']))
    target = entry['evaluation_target']['energy_ha']
    tables = tables_for(entry, options)
    if any(t['status'] != 'ENUMERATED' for t in tables):
        out['reason'] = 'OUTSIDE_DECLARED_COUNT_ENUMERATION_SCOPE'; return out
    L_D, _ = loss_range(h, target)
    if L_D is None:
        out['reason'] = 'DECODER_ENERGY_NUMERICAL_RANGE'; return out
    if partial.get('resolved_loss_upper') is not None and a.exact_scalar(partial['resolved_loss_upper']) != L_D:
        out['reason'] = 'LOSS_RANGE_DIFFERS_FROM_PARTIAL_EVALUATOR'; return out
    uppers = {}; lowers = {}
    coarse = partial.get('coarse_MSE_upper', partial.get('MSE_upper') if 'state_refinement' not in partial else None)
    if coarse is not None:   # bad_loss >= L_D: valid for the total estimator (W0 proposition 2)
        uppers['partial_coarse'] = a.exact_scalar(coarse)
    refined = partial.get('state_refinement') or {}
    hi = _value(partial.get('unresolved_loss') or {}, 'upper')
    if refined.get('MSE_upper') is not None:
        if hi is not None and hi >= L_D:
            uppers['partial_state_dp'] = a.exact_scalar(refined['MSE_upper'])
        lowers['partial_resolved_contribution'] = a.exact_scalar(refined['resolved_MSE_contribution'])
    out['bounds']['partial_state_dp_valid_for_total'] = None if not refined else 'partial_state_dp' in uppers
    lemma = lemma_bound(tables, partial, h=h, target=target)
    out['bounds']['lemma'] = lemma
    if lemma['MSE_upper'] is not None:
        uppers['lemma'] = a.exact_scalar(lemma['MSE_upper'])
    dp = None
    if options.get('total_state_refinement') is not None:
        dp = total_state_dp(tables, h=h, target=target, scope=options['total_state_refinement'])
        out['bounds']['total_state_dp'] = dp
        if dp['MSE_upper'] is not None:
            uppers['total_state_dp'] = a.exact_scalar(dp['MSE_upper'])
            lowers['total_state_dp'] = a.exact_scalar(dp['MSE_lower'])
            out['moments'] = deepcopy(dp['moments'])
    out['bounds']['upper_candidates'] = {k: a.exact_value(v) for k, v in uppers.items()}
    out['bounds']['lower_candidates'] = {k: a.exact_value(v) for k, v in lowers.items()}
    if not uppers:
        out['reason'] = 'NO_VALID_TOTAL_ESTIMATOR_UPPER_BOUND'; return out
    method = min(uppers, key=lambda k: (uppers[k], k)); upper = uppers[method]
    lower = max(lowers.values()) if lowers else Fraction(0)
    if lower > upper:
        out['reason'] = 'INCONSISTENT_TOTAL_ESTIMATOR_BOUNDS'; return out
    if dp is not None and dp['status'] == 'COMPLETE_TOTAL_STATE_DP' and refined.get('status') == 'COMPLETE_STATE_LOSS_FAMILY_BOUND':
        link = dp['partial_decoder']
        if (a.exact_scalar(link['resolved_MSE_contribution']) != a.exact_scalar(refined['resolved_MSE_contribution'])
                or a.exact_scalar(link['unresolved_probability']) != a.exact_scalar(refined['unresolved_probability'])):
            out['reason'] = 'TOTAL_DP_INCONSISTENT_WITH_PARTIAL_DP'; return out
    out.update(status='CONDITIONAL_BINARY64_TOTAL_ESTIMATOR_MODEL_BOUND', MSE_upper=a.exact_value(upper),
        MSE_lower=a.exact_value(lower), MSE_upper_method='MIN_OF_COMPLETED_VALID_UPPER_BOUNDS:' + method,
        RMSE_upper_display=D._sqrt_display_upper(upper),
        combination_rule='MIN_ONLY_NO_SUM_ACROSS_FIRST_UNSAFE_AND_STATE_PARTITIONS')
    out['q3'] = q3_summary(out, options.get('q3_limit'))
    return out


def q3_summary(total, limit=None):
    """Q3 roles: pass/fail by the valid upper bound; bias^2/Var only for design."""
    within = None
    if limit is not None:
        if not limit.get('source') or limit.get('scope') != 'MODEL_MSE_FILTER_ONLY':
            raise ValueError('Explicit model MSE limit SOURCE/scope required; no contract inferred')
        within = total['MSE_upper'] is not None and a.exact_scalar(total['MSE_upper']) <= a.exact_scalar(limit['MSE_upper'])
    return dict(acceptance='MODEL_MSE_UPPER_LE_EPSILON_E_SQUARED', model_MSE_upper=deepcopy(total['MSE_upper']),
        limit=deepcopy(limit), within_limit=within,
        meaning='MODEL_CONDITIONAL_FILTER_NOT_SCIENTIFIC_QUALIFICATION_OR_SUFFICIENT_N',
        design_diagnostics=deepcopy(total.get('moments')),
        minkowski_reference=dict(status='SEPARATE_REFERENCE_NOT_COMBINED',
            direct_risk_includes_PF_and_finite_K_phase_shift=True, delta_E_added=False),
        claim_scope='CONDITIONAL_ON_SUPPLIED_RAW_MEANS_TARGET_AND_IID_EXACT_AFFINE_LAW',
        transfer_terms=dict(target='UNKNOWN', raw_numerical_allowance='UNKNOWN', actual_sampler_law='UNKNOWN'),
        physical_total_error_guarantee=False)


def evaluate(config, signal_packet, allocations, *, evaluation_target, statistical_contract,
             integration, source, options):
    """B's evaluate (unchanged, kept as partial_decoder_risk) + total estimator risk.

    The canonical configuration/error contract still names the original decoder;
    the opt-in estimator is bound through deterministic_input.options.
    """
    if options.get('estimator') != T.ESTIMATOR:
        raise ValueError('Explicit opt-in total estimator required: ' + T.ESTIMATOR)
    records = D.evaluate(config, signal_packet, allocations, evaluation_target=evaluation_target,
        statistical_contract=statistical_contract, integration=integration, source=source, options=options)
    entries = K.prepare(config, signal_packet, allocations, evaluation_target=evaluation_target,
        statistical_contract=statistical_contract, integration=integration)
    for record, entry in zip(records, entries):
        if record['allocation']['allocation_id'] != entry['allocation']['allocation_id']:
            raise ValueError('Evaluation/allocation order mismatch')
        partial = record['diagnostic']['deterministic_risk']
        record['diagnostic'] = dict(deterministic_risk=_total(entry, partial, options), partial_decoder_risk=partial)
    return records
