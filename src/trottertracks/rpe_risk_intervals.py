"""Fixed finite-family risk intervals, conditional on supplied model/support.

Reuses the original trial summary, Clopper-Pearson and bounded-loss envelope.
No decoder, MC runner, energy-contract qualifier, N producer or optimizer.
Applying summarize() to real trials is a scientific numerical analysis.
"""
from copy import deepcopy
import math

from . import rpe_decoder_statistics as S

p, a = S.p, S.B.a
_BINDING = ('configuration_id', 'allocation', 'statistical_contract', 'evaluation_target', 'signal_input')
_FAILURES = {'resolved_probability', 'resolved_contribution', 'model_risk'}


def _probability(value):
    q = a.exact_scalar(value)
    if not 0 < q < 1:
        raise ValueError('Explicit failure probability in (0,1) required')
    return q


def _confidence(alpha):
    q = _probability(alpha)
    c = float(1-q)
    if a.exact_scalar(c) < 1-q:
        c = math.nextafter(c, 1.)
    if not 0 < c < 1:
        raise ValueError('Requested confidence is outside the binary64 CI domain')
    return c


def _binding(entry):
    return {key: deepcopy(entry[key]) for key in _BINDING}


def _family(family):
    if not family.get('source') or not family.get('members'):
        raise ValueError('Explicit fixed confidence family/SOURCE required')
    ids = [v['id'] for v in family['members']]
    if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique nonempty confidence member IDs required')
    total = 0
    for member in family['members']:
        failures = member['failures']
        if not {'resolved_probability', 'resolved_contribution'} <= set(failures) <= _FAILURES:
            raise ValueError('Explicit probability/contribution and optional model risk budgets required')
        for value in failures.values():
            total += _probability(value); _confidence(value)
    if total > _probability(family['failure_probability']):
        raise ValueError('Simultaneous failure budget exceeded')


def confidence_family(members, *, failure_probability, source):
    """Metadata only; all member budgets/N/MC counts/seeds are external inputs.

    Each member supplies id, entry (existing caller prepare output), repetitions,
    seed, result source, failures and any uniform support/model loss contract.
    A union bound requires no independence
    between members; within each member the repeated whole-RPE trials are IID.
    """
    rows = []
    for member in members:
        S._integer(member['repetitions'], 'repetitions', 1); S._integer(member['seed'], 'seed')
        if not member.get('source'):
            raise ValueError('Member result SOURCE required')
        rows.append(dict(id=member['id'], binding=_binding(member['entry']),
            repetitions=member['repetitions'], seed=member['seed'], source=deepcopy(member['source']),
            failures=deepcopy(member['failures']),
            conditional_loss_support=deepcopy(member.get('conditional_loss_support')),
            model_loss_contract=deepcopy(member.get('model_loss_contract'))))
    result = dict(source=deepcopy(source), failure_probability=deepcopy(failure_probability), members=rows)
    _family(result)
    return result


def _support(value, scope):
    if (not value or not value.get('source') or value.get('scope') != scope
            or value.get('upper') is None):
        return None
    upper = S._finite(value['upper'], 'uniform squared-loss support')
    if upper <= 0:
        raise ValueError('Positive uniform squared-loss support required')
    return upper


def validate_request(entry, *, repetitions, seed, source, family, member_id):
    """Check a member's identity/law/SOURCE and CI options before MC."""
    _family(family)
    matches = [v for v in family['members'] if v['id'] == member_id]
    if len(matches) != 1:
        raise ValueError('Unplanned confidence family member')
    member = matches[0]
    if (not p.same(member['binding'], _binding(entry)) or not p.same(source, member['source'])
            or repetitions != member['repetitions'] or seed != member['seed']):
        raise ValueError('Confidence member configuration/allocation/signal/law/target/SOURCE/MC scope mismatch')
    contract = entry['statistical_contract']
    S.iid_policy(contract['simulation_policy'])
    if contract['summary_options'].get('iid_repetitions') is not True:
        raise ValueError('Intervals require an explicit IID whole-RPE repetition assumption')
    _support(member.get('conditional_loss_support'), 'ALL_DECODED_OUTCOMES')
    model_loss_contract = member.get('model_loss_contract')
    if model_loss_contract is not None:
        if type(model_loss_contract.get('adopted')) is not bool:
            raise ValueError('Explicit model loss adoption boolean required')
        if model_loss_contract['adopted']:
            if not model_loss_contract.get('source') or 'model_risk' not in member['failures']:
                raise ValueError('Adopted model loss needs SOURCE and its own failure budget')
            _support(model_loss_contract.get('loss_support'), 'ALL_DECODER_OUTCOMES')
    return member


def conditional_ratio_interval(*, probability_interval, contribution_interval, loss_upper):
    """On simultaneous p,v intervals, enclose v/p ONLY when p_L > 0."""
    pl, pu = probability_interval; vl, vu = contribution_interval
    values = (pl, pu, vl, vu, loss_upper)
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 0 <= pl <= pu <= 1 or not 0 <= vl <= vu or loss_upper <= 0):
        raise ValueError('Ordered finite probability/contribution/support intervals required')
    if pl == 0:
        return dict(status='UNKNOWN', reason='RESOLVED_PROBABILITY_LOWER_BOUND_ZERO', MSE=None, RMSE=None)
    lo, hi = vl/pu, min(loss_upper, vu/pl)
    if lo > hi:
        return dict(status='UNKNOWN', reason='INCONSISTENT_P_V_INTERVALS', MSE=None, RMSE=None)
    return dict(status='CONDITIONAL_MODEL_INTERVAL', MSE=[lo, hi], RMSE=[math.sqrt(lo), math.sqrt(hi)])


def summarize(evaluation, *, family, member_id):
    """Extend an existing caller diagnostic; never issue qualified N evidence.

    The original summary remains untouched. Y=1_D*(Ehat-Etarget)^2 has zero
    contribution on U by definition, which does NOT set the all-event loss on U.
    """
    sim = evaluation['diagnostic']['simulation']; old = evaluation['diagnostic']['summary']
    planned = next((v for v in family['members'] if v['id'] == member_id), None)
    if planned is None:
        raise ValueError('Unplanned confidence family member')
    member = validate_request(evaluation, repetitions=sim.get('repetitions', planned['repetitions']),
        seed=sim.get('seed', planned['seed']), source=evaluation['source'], family=family, member_id=member_id)
    conditional_loss_support = member.get('conditional_loss_support')
    model_loss_contract = member.get('model_loss_contract')
    out = dict(conditional_empirical_RMSE=None if old is None else old['conditional_on_decoded_RMSE'],
        conditional_interval=dict(status='UNKNOWN', reason='NO_EVALUATED_TRIALS', MSE=None, RMSE=None),
        unresolved=None, model_risk=dict(status='UNKNOWN', reason='ALL_EVENT_LOSS_NOT_ADOPTED', MSE=None, RMSE=None),
        confidence_family=dict(source=deepcopy(family['source']), member_id=member_id,
            failure_probability=deepcopy(family['failure_probability']), members=[dict(id=v['id'],
                failures=deepcopy(v['failures'])) for v in family['members']],
            covered_outputs=['unresolved.probability_interval', 'conditional_interval', 'model_risk'],
            scope='FIXED_PREDECLARED_FINITE_CALLS_NO_OPTIONAL_STOPPING'),
        scientific_energy_contract=dict(status='UNKNOWN', rmse_bound_ha=None,
            configuration_id=evaluation['configuration_id'],
            reason='MINKOWSKI_PHYSICAL_NUMERICAL_TARGET_STATE_AND_SAMPLER_TRANSFER_NOT_ESTABLISHED'),
        interval_inputs=dict(conditional_loss_support=deepcopy(conditional_loss_support),
            model_loss_contract=deepcopy(model_loss_contract)),
        risk_upper_input=dict(quantity='MODEL_ONLY_MSE_HA_SQUARED', conditional_MSE_upper=None,
            all_event_model_MSE_upper=None, scientific_energy_RMSE_upper=None, sufficient_N=None),
        numeric_rigor='BINARY64_NOT_INTERVAL_ARITHMETIC', sufficient_N=None, global_RMSE_bound=None, G=None)
    if old is None or sim.get('trials') is None:
        return out
    trials = sim['trials']; n = len(trials)
    if n != member['repetitions'] or any('probability' in row for row in trials):
        raise ValueError('Fixed unweighted IID trial ensemble required')
    # Validate statuses/loss with the existing summary rather than a new parser.
    verified = S.summarize(trials, target_energy_ha=evaluation['evaluation_target']['energy_ha'],
        **evaluation['statistical_contract']['summary_options'])
    if not p.same(verified, old):
        raise ValueError('Summary differs from the supplied trials/target/loss')
    confidence = _confidence(member['failures']['resolved_probability'])
    cp = p.accounting.existing._legacy('rpe_hadamard_failure_validation')._clopper_pearson_interval
    u_interval = list(cp(old['unresolved_count'], n, confidence=confidence))
    p_interval = [1-u_interval[1], 1-u_interval[0]]
    out['unresolved'] = dict(count=old['unresolved_count'], empirical_probability=old['unresolved_probability'],
        probability_interval=u_interval, marginal_confidence=confidence,
        resolved_probability_interval=p_interval, method='EXISTING_CLOPPER_PEARSON')
    upper = _support(conditional_loss_support, 'ALL_DECODED_OUTCOMES')
    if upper is None:
        out['conditional_interval']['reason'] = 'MISSING_UNIFORM_DECODED_LOSS_SUPPORT'
    else:
        target = evaluation['evaluation_target']['energy_ha']
        if any((r['energy_estimate_ha']-target)**2 > upper for r in trials if r['status'] == S.DECODED):
            raise ValueError('Observed decoded loss exceeds supplied support; no clipping')
        exact_alpha = _probability(member['failures']['resolved_contribution'])
        alpha = float(exact_alpha)
        if a.exact_scalar(alpha) > exact_alpha:
            alpha = math.nextafter(alpha, 0.)
        radius = upper*math.sqrt((math.log(2)-math.log(alpha))/(2*n))
        value = old['resolved_MSE_contribution']
        v_interval = [max(0., value-radius), min(upper, value+radius)]
        interval = conditional_ratio_interval(probability_interval=p_interval,
            contribution_interval=v_interval, loss_upper=upper)
        interval.update(probability_interval=p_interval, contribution_interval=v_interval,
            loss_support=deepcopy(conditional_loss_support),
            failure_probability_sum=a.exact_value(_probability(member['failures']['resolved_probability'])
                + _probability(member['failures']['resolved_contribution'])),
            method='CP_RESOLUTION_PLUS_HOEFFDING_CONTRIBUTION_UNION_BOUND')
        out['conditional_interval'] = interval
    model = model_loss_contract or {}
    if model.get('adopted'):
        loss = evaluation['statistical_contract']['summary_options']['unresolved_loss']
        support = model.get('loss_support')
        if loss['kind'] != 'BOUNDED_INTERVAL':
            out['model_risk']['reason'] = 'UNDEFINED_UNRESOLVED_LOSS'
        elif _support(support, 'ALL_DECODER_OUTCOMES') is None:
            out['model_risk']['reason'] = 'MISSING_UNIFORM_ALL_EVENT_LOSS_SUPPORT'
        else:
            result = S.summarize(trials, target_energy_ha=evaluation['evaluation_target']['energy_ha'],
                unresolved_loss=loss, iid_repetitions=True,
                confidence=_confidence(member['failures']['model_risk']), loss_support=support)
            ci = result['population_RMSE_CI']
            out['model_risk'] = dict(status='DECLARED_MODEL_LOSS_FAMILY_INTERVAL',
                MSE=ci['MSE_interval'], RMSE=ci['interval'], existing_envelope=ci,
                loss_contract=deepcopy(model), unresolved_loss=deepcopy(loss),
                unique_completed_estimator_defined=False, includes_physical_transfer=False)
    out['risk_upper_input'] = dict(quantity='MODEL_ONLY_MSE_HA_SQUARED',
        conditional_MSE_upper=None if out['conditional_interval']['MSE'] is None else out['conditional_interval']['MSE'][1],
        all_event_model_MSE_upper=None if out['model_risk']['MSE'] is None else out['model_risk']['MSE'][1],
        scientific_energy_RMSE_upper=None, sufficient_N=None)
    return out
