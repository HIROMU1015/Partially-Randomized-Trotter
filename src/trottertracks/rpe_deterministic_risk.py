"""Finite-count, no-MC risk bounds for the unchanged scalar-compatible decoder.

Probabilities and energy-loss sums are exact rational numbers. Phase/branch
operations reuse the existing binary64 decoder. A finite adjacent-support
certificate replaces an assumed pi/3 rule. Physical/scientific qualification
and sufficient N are NOT produced here. Real-input evaluation needs permission.
"""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import math

from . import rpe_signal_statistics_caller as K
from . import rpe_decoder_fast as F

B, S, W, p, a = K.B, K.S, K.W, K.p, K.B.a
EXACT_AFFINE = 'EXACT_AFFINE_OF_BINARY64_RAW'
LEGACY_AFFINE = 'LEGACY_BINARY64_AFFINE_RAW'
TIE_TOLERANCE = 1e-12


def _axis_probability(raw, probability_rule):
    S._finite(raw, 'raw coordinate')
    if not -1 <= raw <= 1:
        raise ValueError('Raw +/-1 mean outside [-1,1]; no clipping')
    if probability_rule == EXACT_AFFINE:
        return (1 + a.exact_scalar(raw))/2
    elif probability_rule == LEGACY_AFFINE:
        return a.exact_scalar((1.+raw)/2.)
    else:
        raise ValueError('Explicit exact-affine or legacy binary64 probability rule required')


def binomial_axis(raw, shots, *, probability_rule):
    """Integer binomial numerators with one common denominator; no tail loss."""
    S._integer(shots, 'shots', 1)
    probability = _axis_probability(raw, probability_rule)
    aa, dd = probability.numerator, probability.denominator
    if aa in (0, dd):
        return dict(shots=shots, probability=a.exact_value(probability),
            counts=[0 if aa == 0 else shots], numerators=[1], denominator=1)
    bb = dd-aa; denominator = dd**shots; weights = [bb**shots]
    for k in range(shots):
        value, remainder = divmod(weights[-1]*(shots-k)*aa, (k+1)*bb)
        if remainder:
            raise ArithmeticError('Nonintegral binomial recurrence')
        weights.append(value)
    if sum(weights) != denominator:
        raise ArithmeticError('Binomial mass not conserved')
    return dict(shots=shots, probability=a.exact_value(probability),
        counts=list(range(shots+1)), numerators=weights, denominator=denominator)


def probability_report(value):
    """Exact value survives below binary64 underflow; log is diagnostic only."""
    value = value if isinstance(value, Fraction) else a.exact_scalar(value)
    if not 0 <= value <= 1:
        raise ValueError('Probability outside [0,1]')
    log = None if not value else (math.log1p(-float(1-value)) if value > Fraction(1,2)
        else math.log(value.numerator)-math.log(value.denominator))
    return dict(exact=a.exact_value(value), zero_exact=value == 0, log_probability=log)


def _sqrt_display_upper(value):
    # Avoid converting a tiny MSE to float BEFORE taking its square root.
    with localcontext() as context:
        context.prec = 40
        root = float((Decimal(value.numerator)/Decimal(value.denominator)).sqrt())
    if not math.isfinite(root):
        return None
    while a.exact_scalar(root)**2 < value:
        root = math.nextafter(root, math.inf)
        if not math.isfinite(root):
            return None
    return root


def round_distribution(row, *, h, target_energy, radius, probability_rule, maximum_outcomes):
    """All nonzero count outcomes, with original count->float->atan2 order.

    'unsafe' means outside a declared good-phase set, not an independently
    identified wrong branch. Branch ties depend on the preceding estimate.
    """
    import numpy as np
    S._integer(row['m'], 'm'); S._integer(maximum_outcomes, 'maximum_outcomes', 1)
    S._finite(h, 'h'); S._finite(target_energy, 'target energy'); S._finite(radius, 'phase radius')
    if h <= 0 or not 0 < radius < math.pi:
        raise ValueError('Positive h and explicit phase radius in (0,pi) required')
    # Check the finite count domain BEFORE constructing potentially large powers.
    sizes = []
    for axis, component in (('X','real'), ('Y','imag')):
        n = S._integer(row['N_'+axis], 'N_'+axis, 1)
        probability = _axis_probability(row['raw'][component], probability_rule)
        sizes.append(1 if probability in (0,1) else n+1)
    cells = math.prod(sizes)
    if cells > maximum_outcomes:
        return dict(status='UNKNOWN', reason='OUTSIDE_DECLARED_COUNT_ENUMERATION_SCOPE', m=row['m'], cells=cells)
    axes = {axis: binomial_axis(row['raw'][component], row['N_'+axis], probability_rule=probability_rule)
            for axis, component in (('X','real'), ('Y','imag'))}
    coordinates = {axis: [S.count_mean(k, row['N_'+axis])['mean'] for k in axes[axis]['counts']]
                   for axis in ('X','Y')}
    q = 1 << row['m']; theta = -h*target_energy
    if not math.isfinite(q*theta):
        return dict(status='UNKNOWN', reason='REFERENCE_PHASE_NUMERICAL_RANGE', m=row['m'])
    reference = float(F._wrap(q*theta)); outcomes = []
    for ix, kx in enumerate(axes['X']['counts']):
        x = coordinates['X'][ix]
        for iy, ky in enumerate(axes['Y']['counts']):
            y = coordinates['Y'][iy]
            phase = math.atan2(y, x); principal = float(F._wrap(phase))
            delta = float(F._wrap(phase-reference)); zero = x == y == 0
            outcomes.append(dict(X_plus=kx, Y_plus=ky, phase=phase, principal=principal,
                delta=delta, zero=zero,
                weight=axes['X']['numerators'][ix]*axes['Y']['numerators'][iy]))
    phases = np.array([v['principal'] for v in outcomes])
    if q == 1:
        values, branches, gaps = phases, np.zeros(len(phases), dtype=int), None
    else:
        values, branches, best, runner, _ = F._nearest(phases,
            np.full(len(phases), float(F._wrap(theta))), q)
        gaps = runner-best
    for i, item in enumerate(outcomes):
        tie = gaps is not None and float(gaps[i]) <= TIE_TOLERANCE
        item.update(oracle_estimate=float(values[i]), oracle_branch=int(branches[i]), oracle_tie=tie,
            good=not item['zero'] and abs(item['delta']) < radius and not tie)
    denominator = axes['X']['denominator']*axes['Y']['denominator']
    return dict(status='ENUMERATED', m=row['m'], q=q, radius=radius, outcomes=outcomes,
        denominator=denominator, axes=axes, cells=cells)


def transition_distribution(table, previous_estimate):
    """Conditional next-round statuses at ONE supplied resolved prior state."""
    import numpy as np
    if table['status'] != 'ENUMERATED' or table['m'] == 0:
        raise ValueError('An enumerated noninitial round is required')
    S._finite(previous_estimate, 'previous estimate')
    out = []
    for item in table['outcomes']:
        value, branch, best, runner, _ = F._nearest(np.array([item['principal']]),
            np.array([previous_estimate]), table['q'])
        tie = abs(float(runner[0]-best[0])) <= TIE_TOLERANCE
        status = ('UNRESOLVED_ZERO_COORDINATES' if item['zero'] else
                  'UNRESOLVED_BRANCH_TIE' if tie else S.DECODED)
        out.append(dict(X_plus=item['X_plus'], Y_plus=item['Y_plus'], status=status,
            estimate=None if status != S.DECODED else float(value[0]), selected_branch=int(branch[0]),
            oracle_mismatch=status == S.DECODED and float(value[0]) != item['oracle_estimate'],
            probability=a.exact_value(Fraction(item['weight'], table['denominator']))))
    return out


def verify_good_transitions(tables, *, maximum_pairs):
    """Inductive finite-support proof using actual R3/scalar tie semantics.

    Every possible good previous estimate is tested against every good current
    phase, a superset of reachable good histories. No joint history enumeration.
    """
    import numpy as np
    S._integer(maximum_pairs, 'maximum_pairs', 1); checked = 0
    if any(t['status'] != 'ENUMERATED' for t in tables):
        return dict(status='UNKNOWN', reason='ROUND_COUNT_SCOPE_INCOMPLETE', checked_pairs=0)
    for previous, current in zip(tables, tables[1:]):
        old = np.unique([v['oracle_estimate'] for v in previous['outcomes'] if v['good']])
        phases = {v['principal']: v['oracle_estimate'] for v in current['outcomes'] if v['good']}
        if checked + len(old)*len(phases) > maximum_pairs:
            return dict(status='UNKNOWN', reason='OUTSIDE_DECLARED_TRANSITION_VERIFICATION_SCOPE', checked_pairs=checked)
        for phase, oracle in phases.items():
            if not len(old):
                continue
            value, _, best, runner, _ = F._nearest(np.full(len(old), phase), old, current['q'])
            checked += len(old)
            if np.any(value != oracle) or np.any(np.abs(runner-best) <= TIE_TOLERANCE):
                return dict(status='UNKNOWN', reason='GOOD_SET_NOT_CLOSED_UNDER_ORIGINAL_DECODER',
                    m=current['m'], checked_pairs=checked)
    return dict(status='VERIFIED_FINITE_BINARY64_SUPPORT', checked_pairs=checked,
        method='ALL_ADJACENT_GOOD_SUPPORT_PAIRS_ORIGINAL_SCALAR_TIE_TOLERANCE')


def _bound(entry, options):
    prepared = entry['prepared']; rows = prepared['rows']
    out = dict(status='UNKNOWN', MSE_upper=None, RMSE_upper_display=None, rounds=[],
        decoder_certificate=None, scientific_energy_qualification='UNKNOWN', sufficient_N=None,
        global_RMSE_bound=None, G=None, numerical=dict(probabilities='EXACT_RATIONAL_NO_TRUNCATION',
            decoder='UNCHANGED_BINARY64_COUNT_ATAN2_R3', real_arithmetic_phase_enclosure=None),
        unresolved_energy_imputation=None, unique_completed_estimator_defined=False,
        probability_rule=options['probability_rule'], MC_calls=0)
    if not prepared['input_complete']:
        out['reason'] = 'MISSING_STATISTICAL_INPUT'; return out
    if prepared['M'] > 12:
        out['reason'] = 'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN'; return out
    contract = entry['statistical_contract']; S.iid_policy(contract['simulation_policy'])
    if prepared['decoder'] != S.DECODER or any(r['law_binding']['definition']['model'] !=
            contract['simulation_policy']['target_law'] for r in rows):
        raise ValueError('Original decoder and matching IID target law required')
    if not prepared['physical_qualification_complete'] and contract.get('allow_conditional_unqualified') is not True:
        out['reason'] = 'PHYSICAL_MEANS_UNQUALIFIED'; return out
    radii = options['phase_radii']
    if len(radii) != prepared['M']+1:
        raise ValueError('One explicit phase radius for every round required')
    out['ideal_geometry_diagnostic'] = [dict(m=m, two_previous_plus_current=2*radii[m-1]+radii[m],
        strict_real_geometry_condition=2*radii[m-1]+radii[m] < math.pi,
        sufficient_for_binary64_decoder=False) for m in range(1, len(radii))]
    h = float(a.exact_scalar(prepared['h'])); target = entry['evaluation_target']['energy_ha']
    tables = [round_distribution(dict(m=r['m'], raw=r['mean']['raw'], **{k:r['shots'][k] for k in ('N_X','N_Y')}),
        h=h, target_energy=target, radius=radii[r['m']], probability_rule=options['probability_rule'],
        maximum_outcomes=options['maximum_round_outcomes']) for r in rows]
    certificate = verify_good_transitions(tables, maximum_pairs=options['maximum_verification_pairs'])
    out['decoder_certificate'] = certificate
    for table in tables:
        if table['status'] != 'ENUMERATED':
            out['rounds'].append({k:v for k,v in table.items() if k != 'outcomes'}); continue
        den = table['denominator']; items = table['outcomes']
        good = Fraction(sum(v['weight'] for v in items if v['good']), den)
        zero = Fraction(sum(v['weight'] for v in items if v['zero']), den)
        out['rounds'].append(dict(m=table['m'], cells=table['cells'],
            good=probability_report(good), unsafe=probability_report(1-good), zero=probability_report(zero),
            reference_oracle_tie=probability_report(Fraction(sum(v['weight'] for v in items if v['oracle_tie'] and not v['zero']), den)),
            actual_tie_probability='DEPENDS_ON_PRECEDING_RESOLVED_STATE'))
    if certificate['status'] != 'VERIFIED_FINITE_BINARY64_SUPPORT':
        out['reason'] = certificate['reason']; return out
    prefix = Fraction(1); first_bad = []
    for table in tables:
        good = Fraction(sum(v['weight'] for v in table['outcomes'] if v['good']), table['denominator'])
        first_bad.append(prefix*(1-good)); prefix *= good
    any_zero = 1-math.prod(1-a.exact_scalar(r['zero']['exact']) for r in out['rounds'])
    out.update(first_unsafe=[probability_report(v) for v in first_bad], all_good=probability_report(prefix),
        any_zero=probability_report(any_zero), unresolved_probability_bounds=dict(
            lower=a.exact_value(any_zero), upper=a.exact_value(1-prefix),
            meaning='ANY_ZERO_SUBSET_UNRESOLVED_SUBSET_ANY_UNSAFE'))
    model = options.get('model_loss_contract') or {}
    loss = contract['summary_options']['unresolved_loss']
    if (model.get('adopted') is not True or not model.get('source')
            or loss.get('kind') != 'BOUNDED_INTERVAL' or not loss.get('source')):
        out['reason'] = 'ALL_EVENT_LOSS_NOT_ADOPTED_OR_UNDEFINED'; return out
    lo = a.exact_scalar(loss['lower']); hi = a.exact_scalar(loss['upper'])
    if not 0 <= lo <= hi:
        raise ValueError('Nonnegative ordered unresolved-loss bounds required')
    edge = math.pi/h
    if not math.isfinite(edge) or not math.isfinite(math.nextafter(edge, math.inf)):
        out['reason'] = 'DECODER_ENERGY_NUMERICAL_RANGE'; return out
    edge = math.nextafter(edge, math.inf)
    ee = a.exact_scalar(target)
    resolved_loss = max((a.exact_scalar(edge)-ee)**2, (-a.exact_scalar(edge)-ee)**2)
    bad_loss = max(resolved_loss, hi)
    terminal = tables[-1]
    terminal_contribution = sum((Fraction(v['weight'], terminal['denominator'])*
        (a.exact_scalar(-v['oracle_estimate']/h)-ee)**2
        for v in terminal['outcomes'] if v['good']), Fraction(0))
    earlier_good = math.prod(a.exact_scalar(r['good']['exact']) for r in out['rounds'][:-1])
    bound = earlier_good*terminal_contribution + sum(first_bad)*bad_loss
    # Presentation only; eligibility uses the exact MSE and exact squared limit.
    display = _sqrt_display_upper(bound)
    out.update(status='CONDITIONAL_BINARY64_DECODER_MODEL_BOUND', MSE_upper=a.exact_value(bound),
        RMSE_upper_display=display,
        terminal_good_MSE_contribution=a.exact_value(earlier_good*terminal_contribution),
        resolved_loss_upper=a.exact_value(resolved_loss), unsafe_loss_upper=a.exact_value(bad_loss),
        all_event_loss_contract=deepcopy(model), unresolved_loss=deepcopy(loss),
        drift_refinement='UNQUALIFIED_NOT_USED', confidence=None)
    return out


def evaluate(config, signal_packet, allocations, *, evaluation_target, statistical_contract,
             integration, source, options):
    """Whole-vector primary entry. No MC, confidence interval or N production."""
    if not source or not options.get('source'):
        raise ValueError('Evaluation and deterministic options SOURCE required')
    if options.get('probability_rule') not in (EXACT_AFFINE, LEGACY_AFFINE):
        raise ValueError('Explicit supported probability rule required')
    entries = K.prepare(config, signal_packet, allocations, evaluation_target=evaluation_target,
        statistical_contract=statistical_contract, integration=integration)
    results = []
    for entry in entries:
        bound = _bound(entry, options)
        binding = {k:deepcopy(entry[k]) for k in ('configuration_id','allocation','statistical_contract',
                                                  'evaluation_target','signal_input')}
        binding.update(options=deepcopy(options), source=deepcopy(source))
        results.append(dict(configuration_id=config['id'], allocation=deepcopy(entry['allocation']),
            statistical_contract=deepcopy(statistical_contract), evaluation_target=deepcopy(evaluation_target),
            signal_input=deepcopy(entry['signal_input']), source=deepcopy(source),
            deterministic_input=binding, diagnostic=dict(deterministic_risk=bound),
            n_packet=None, sufficient_N=None, global_RMSE_bound=None, G=None))
    return results


def compare(config, evaluations, prices, *, model_limit, qualifications, **cost_options):
    """Filter whole allocations by a supplied model MSE limit, then reuse W.

    Formal N evidence remains externally supplied; no QUALIFIED flags or
    scientific packets are manufactured. Synthetic qualifications are fixtures.
    """
    if not model_limit.get('source') or model_limit.get('scope') != 'MODEL_MSE_FILTER_ONLY':
        raise ValueError('Explicit model-filter limit/SOURCE/scope required; no adopted energy contract inferred')
    limit = a.exact_scalar(model_limit['MSE_upper'])
    if limit < 0:
        raise ValueError('Nonnegative model MSE filter limit required')
    if not set(qualifications) <= {v['allocation']['allocation_id'] for v in evaluations}:
        raise ValueError('Foreign allocation qualification')
    accepted = []; filters = []
    for record in evaluations:
        if record['configuration_id'] != config['id'] or record.get('n_packet') is not None:
            raise ValueError('Deterministic diagnostic configuration/type mismatch')
        risk = record['diagnostic']['deterministic_risk']; value = risk.get('MSE_upper')
        usable = value is not None and a.exact_scalar(value) <= limit
        packet = qualifications.get(record['allocation']['allocation_id'])
        if packet is not None and not p.same((packet.get('rmse_evidence') or {}).get('deterministic_input'),
                                            record['deterministic_input']):
            raise ValueError('External N evidence uses different deterministic signal/law/target/options')
        evaluation = deepcopy(record); evaluation['n_packet'] = deepcopy(packet) if usable else None
        accepted.append(evaluation)
        filters.append(dict(allocation_id=record['allocation']['allocation_id'],
            status='MODEL_UPPER_BOUND_WITHIN_LIMIT' if usable else 'UNKNOWN_OR_UPPER_BOUND_NOT_SUFFICIENT',
            external_N_packet_present=packet is not None))
    result = W.compare(config, [v['allocation'] for v in evaluations], accepted, prices, **cost_options)
    return dict(model_filter=filters, model_limit=deepcopy(model_limit), comparison=result,
        scientific_energy_qualification='UNCHANGED_EXTERNAL_EVIDENCE_ONLY', optimizer_scope='SUPPLIED_FINITE_WHOLE_ALLOCATION_MENU')
