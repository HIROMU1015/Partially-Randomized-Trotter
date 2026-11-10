"""Thin, passive connection of finite-time DF10 records to existing RPE arithmetic.

No data files, sampler, circuit provider, eigenproblem or scientific runner are
loaded here. Real-data numerical calls require their separately permitted scope.
Saved-point inspection and time/identity matching do not solve N or sum G.
"""
from copy import deepcopy
import math
from fractions import Fraction
from types import SimpleNamespace

from . import c1_stage3_accuracy as existing

AXES = ('X', 'Y')
SCOPE = 'single_hadamard_interrogation_without_state_preparation'
PRICE_KINDS = ('DIRECT_COMPILED_DETERMINISTIC', 'DIRECT_COMPILED_FINITE_SAMPLE',
               'MODEL_PREDICTION')


def exact_scalar(value):
    """Exact supplied integer/binary64 price; never round a quantum shot count."""
    if type(value) is int:return Fraction(value)
    if type(value) is float and math.isfinite(value):return Fraction.from_float(value)
    if isinstance(value,dict) and set(value)=={'numerator','denominator'}:
        if type(value['numerator']) is int and type(value['denominator']) is int and value['denominator']>0:
            return Fraction(value['numerator'],value['denominator'])
    raise ValueError('Finite integer/binary64 or exact rational required')


def exact_value(value):
    value=exact_scalar(value) if not isinstance(value,Fraction) else value
    return dict(numerator=value.numerator,denominator=value.denominator)


def _finite(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(name + ' must be a finite nonnegative/positive scalar')
    return value


def inspect_saved_point(point, *, expected_system='H4', expected_rank=10,
                        expected_binding=None):
    """Copy saved fields only. Never reconstruct raw signals, B, A, N or G."""
    c = point['candidate']
    if c['system'] != expected_system or point['binding']['rank'] != expected_rank:
        raise ValueError('Saved point differs from the expected system/rank')
    if expected_binding is not None and any(
            point['binding'].get(k) != v for k, v in expected_binding.items()):
        raise ValueError('Saved scientific binding differs from the expected input')
    if c['method'] not in ('FULL', 'B0', 'B0I', 'PR'):
        raise ValueError('B3 and pure-RTE families are outside this connection')
    for key in ('hamiltonian', 'snapshot', 'state', 'rank_policy'):
        if not point['binding'].get(key):
            raise ValueError('Explicit saved scientific binding required')
    for p in point['prices']:
        if p['candidate'] != c or p['binding'] != point['binding']:
            raise ValueError('Signal/price candidate or scientific binding mismatch')
        if p['scope'] != SCOPE or p['compiler'] != point['compiler']:
            raise ValueError('Full-wrapper compiler/scope mismatch')
        if p['classification'] not in PRICE_KINDS:
            raise ValueError('Explicit direct/model price classification required')
    result = deepcopy(point)
    result.update(status='SAVED_POINT_PENDING_RPE_QUALIFICATION', N_X=None, N_Y=None,
                  rpe_round=None, full_rpe_resources=None)
    return result


def bind_saved_round(point, *, target_id, m, base_time, expected_system='H4',
                     expected_rank=10, expected_binding=None):
    """Match an explicit dyadic time; q_PF remains the saved PF step count."""
    p = inspect_saved_point(point, expected_system=expected_system,
                            expected_rank=expected_rank,
                            expected_binding=expected_binding)
    if not target_id or type(m) is not int or m < 0:
        raise ValueError('Explicit target ID and nonnegative round required')
    _finite(base_time, 'base time', positive=True)
    if math.ldexp(base_time, m) != p['candidate']['T']:
        raise ValueError('No interpolation, signal powering or time substitution')
    p.update(target_id=target_id, m=m, base_time=base_time, q_RPE=1 << m,
             q_PF=p['candidate']['q'], candidate_id=p['candidate']['id'])
    return p


def assess_supplied_round(row, *, policy, corrected_signal_disk,
                          normalization_error_raw, implementation_error_raw,
                          law_error_raw, reference_provenance, iid_provenance,
                          error_provenance, df_accounting,
                          evidence_kind='numerically_qualified_pilot'):
    """Use a SUPPLIED state-specific disk to exp(-i E_target T), not an op norm.

    The disk must include the reference/state/target mismatch. It is NOT inferred
    from saved coordinate bias or a small eigenpair residual. The caller supplies
    normalization/implementation/law error in raw units, including the effect of
    saved log_B_margin; unknown is None, never an implicit zero. B is the WHOLE
    interrogation normalization. Existing geometry and Hoeffding do the solving.
    """
    beta_total, beta_stat = policy['beta_total'], policy['beta_stat']
    if not 0 < beta_stat < beta_total < math.pi/3:
        raise ValueError('Strict dyadic phase budgets required')
    for a in AXES:
        alpha = _finite(policy['alpha_'+a], 'axis alpha', positive=True)
        if alpha >= 1:
            raise ValueError('Axis alpha must be below one')
    if df_accounting not in ('own_DF_target_no_DF_energy_shift',
                              'original_H_error_in_signal_once'):
        raise ValueError('Explicit target/DF counting convention required')
    if evidence_kind != 'numerically_qualified_pilot':
        raise ValueError('This saved-signal pilot does not establish rigorous bounds')
    if policy['law_model'] not in ('ideal_finite_RTE_law', 'actual_implemented_law', 'deterministic'):
        raise ValueError('Explicit measurement-law model required')
    missing = any(x is None for x in (corrected_signal_disk,
        normalization_error_raw, implementation_error_raw, law_error_raw,
        row['saved_signal'].get('log_B')))
    if missing or not all((reference_provenance, iid_provenance, error_provenance)):
        return dict(status='UNRESOLVED_REFERENCE_OR_ERROR', N_X=None, N_Y=None,
                    target_id=row['target_id'], m=row['m'])
    log_B = _finite(row['saved_signal']['log_B'], 'whole-interrogation log B')
    disk = _finite(corrected_signal_disk, 'corrected state-specific signal disk')
    raw_extras = [_finite(x, 'raw error') for x in
                  (normalization_error_raw, implementation_error_raw, law_error_raw)]
    attenuation = math.exp(-log_B)
    raw_error = math.fsum((attenuation*disk, *raw_extras))
    evidence = SimpleNamespace(m=row['m'], target_id=row['target_id'],
                               evidence_kind=evidence_kind,
                               law_model=policy['law_model'])
    axes = {}
    for a in AXES:
        target = SimpleNamespace(beta_total=beta_total, beta_stat=beta_stat,
                                 alpha_axis=policy['alpha_'+a])
        axes[a] = existing._assess_raw_signal(target, evidence,
                         attenuation=attenuation, raw_error=raw_error)
    passed = all(x['status'] == 'CONDITIONAL_ROUND_PASS' for x in axes.values())
    return dict(status='CONDITIONAL_ROUND_PASS' if passed else 'UNQUALIFIED_ROUND',
        m=row['m'], target_id=row['target_id'], candidate_id=row['candidate_id'],
        binding=deepcopy(row['binding']), candidate=deepcopy(row['candidate']),
        q_PF=row['q_PF'], q_RPE=row['q_RPE'], base_time=row['base_time'], axes=axes,
        N_X=axes['X'].get('shots'), N_Y=axes['Y'].get('shots'),
        alpha_X=policy['alpha_X'], alpha_Y=policy['alpha_Y'],
        normalization_applied_once=True, df_accounting=df_accounting,
        qualification='CONDITIONAL_ON_SUPPLIED_STATE_SPECIFIC_SIGNAL_DISK',
        evidence_kind=evidence_kind, reference_provenance=reference_provenance,
        error_provenance=error_provenance, iid_provenance=iid_provenance,
        full_rpe_energy_success=False, unconditional_success_claim=False)


def failure_budget(plans, *, alpha_total):
    """Union bound allocation; no claim of cross-round independence."""
    _finite(alpha_total, 'total alpha', positive=True)
    if alpha_total >= 1:
        raise ValueError('Total alpha must be below one')
    if len({p['m'] for p in plans}) != len(plans):
        raise ValueError('Duplicate round')
    alpha = math.fsum(_finite(p['alpha_'+a], 'axis alpha', positive=True)
                      for p in plans for a in AXES)
    return dict(allocated_failure=alpha, alpha_total=alpha_total,
                within_budget=alpha <= alpha_total)


def aggregate_supplied_axes(plans, prices, *, expected_rounds, alpha_total,
                            compiler, metrics=existing.METRICS, partial_rounds=False, exact=False):
    """Small axis-N extension: old aggregate_totals requires equal X/Y shots.

    This function is arithmetic only, with no energy/horizon qualification or
    prediction. The caller must separately establish the common task. Means are
    supplied frozen prices. Finite-cohort and model uncertainty remain distinct.
    """
    plans, prices, expected = tuple(plans), tuple(prices), tuple(expected_rounds)
    unavailable = lambda status: dict(status=status, totals=None)
    start = expected[0] if expected and partial_rounds else 0
    if (not expected or type(start) is not int or start < 0
            or expected != tuple(range(start, start+len(expected)))):
        raise ValueError('Explicit consecutive dyadic horizon required')
    if len({p['m'] for p in plans}) != len(plans):
        raise ValueError('Duplicate round')
    if {p['m'] for p in plans} != set(expected):
        return unavailable('INCOMPLETE_ROUNDS')
    if (len({p['target_id'] for p in plans}) != 1
            or len({p['base_time'] for p in plans}) != 1
            or any(p['q_RPE'] != 1 << p['m'] or p['candidate']['T'] !=
                   math.ldexp(p['base_time'], p['m']) for p in plans)):
        raise ValueError('One common energy task/base time and dyadic schedule required')
    if any(p['status'] != 'CONDITIONAL_ROUND_PASS' or
           any(p.get('N_'+a) is None for a in AXES) for p in plans):
        return unavailable('UNQUALIFIED_ROUNDS')
    if not failure_budget(plans, alpha_total=alpha_total)['within_budget']:
        return unavailable('FAILURE_BUDGET_EXCEEDED')
    keys = [(p['m'], p['axis']) for p in prices]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate price axis')
    if set(keys) != {(m, a) for m in expected for a in AXES}:
        return unavailable('INCOMPLETE_PRICE_COVERAGE')
    if not metrics or any(g not in existing.METRICS for g in metrics):
        raise ValueError('Use explicit count/depth/size metrics; width is separate')
    lookup = {p['m']: p for p in plans}
    terms = {g: [] for g in metrics}
    widths, origins = [], []
    for price in prices:
        plan = lookup[price['m']]
        if (price['candidate'] != plan['candidate'] or price['binding'] != plan['binding']
                or price['compiler'] != compiler or price['scope'] != SCOPE):
            raise ValueError('Price candidate/binding/compiler/full-wrapper mismatch')
        if not price.get('source'):
            return unavailable('MISSING_PRICE_PROVENANCE')
        kind = price['classification']
        if kind not in PRICE_KINDS:
            raise ValueError('Explicit direct/model origin required')
        if kind == 'MODEL_PREDICTION':
            model = price.get('model')
            if not model or not model.get('source') or not model.get('coverage'):
                return unavailable('UNQUALIFIED_MODEL_CONTEXT')
            var = model.get('q_variable')
            if var not in ('q_PF', 'q_RPE') or model.get('q') != plan[var]:
                raise ValueError('Price model q must name the matching q variable')
            if model.get('binding') != plan['binding'] or model.get('compiler') != compiler:
                raise ValueError('Foreign price-model scientific/compiler context')
        nshots = plan['N_'+price['axis']]
        if type(nshots) is not int or nshots <= 0:
            raise ValueError('Positive integer quantum shots required')
        if any(price.get('mean', {}).get(g) is None for g in metrics):
            return unavailable('MISSING_PRICE_METRIC')
        for g in metrics:
            value=price['mean'][g]
            if exact:
                value=exact_scalar(value)
                if value<0:raise ValueError('Nonnegative one-shot price required')
            else:value=_finite(value, 'one-shot price')
            terms[g].append(nshots*value)
        widths.append(price.get('peak_qubits'))
        origins.append(dict(m=price['m'], axis=price['axis'], classification=kind,
                            source=price.get('source'), model=price.get('model')))
    return dict(status='SUPPLIED_AXIS_COST_SUM',
        totals={g: exact_value(sum(values)) if exact else math.fsum(values) for g, values in terms.items()},
        numeric_encoding='exact_rational' if exact else 'legacy_scalar',
        peak_qubits=max(widths) if all(v is not None for v in widths) else None,
        price_origins=origins, total_covariance=None, total_confidence_interval=None,
        classical_cost_samples_are_shot_multipliers=False,
        normalization_is_price_multiplier=False, full_rpe_energy_success=False,
        unconditional_total_resource_claim=False,
        partial_only=partial_rounds, included_rounds=expected)
