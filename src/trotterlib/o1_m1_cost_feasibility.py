"""Unpriced C0 cost screen for explicitly supplied, validated signal records.

No input I/O, circuit, sampling, optimizer or price-provider call occurs here.
Applying these functions to saved molecular values requires execution permission.
The returned price ceiling is necessary for beating the whole baseline budget;
it is never a sufficient condition or a positive lower bound on circuit price.
"""
from fractions import Fraction
import math

from .o1_m1_resource_bridge import account_radius_round
from .rpe_resource_accounting import RPEErrorAllocation
from .rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE
from .rte import require_integer_count


def _positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + ' must be positive and finite')
    return number


def account_saved_c0_signal(*, saved_row, dictionary, config, specification,
                            context_hash, arm_hash):
    """Reuse C0 allocation and actual saved bounds, without calculating a price.

    The caller binds the saved row/config/context to its immutable SOURCE/input.
    No signal at another q/r/K/delta is extrapolated. The function cannot create
    missing q16/q32 r4/r8 observations or validate a molecular source by itself.
    """
    if dictionary not in ('P', 'DF') or config is None:
        raise ValueError('Explicit P/DF finite-RTE law required; no unit-radius default')
    if (specification.q_m not in (16, 32, 64)
            or specification.delta_time != .05 or config.evolution_time != .05
            or config.rte_steps not in (2, 4, 8) or config.finite_taylor_order != 2
            or (saved_row['q'], saved_row['delta'], saved_row['r'], saved_row['K'])
            != (specification.q_m, .05, config.rte_steps, 2)):
        raise ValueError('Saved signal and fixed q/delta/r/K differ; no transfer')
    saved = saved_row['arms'][dictionary]
    if saved['feasible'] is not True or saved['reasons']:
        raise ValueError('A qualified saved signal arm is required')
    bounds = dict(saved['bounds'])
    for field, limit in (('model_phase_bound', .03), ('implementation_phase_bound', .01)):
        value = float(bounds[field])
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('Fixed C0 ' + field + ' exceeded')
    # Preserve all fourteen C0 measurement-axis allocations, even when only
    # one of the seven registered rounds is screened.
    allocation = RPEErrorAllocation(.06, .03, .15, .05/14, .05/14)
    account = account_radius_round(context_hash=context_hash, arm_hash=arm_hash,
        specification=specification, allocation=allocation, beta_rpe=.30,
        cost_metric='cx_count', config=config, cost=None, **bounds)
    for key in ('attenuation', 'normalization_product', 'rho_observed_lower_bound'):
        observed = _positive(saved[key], 'Saved ' + key)
        value = account[key]
        if value is None or not math.isclose(observed, value, rel_tol=1e-12, abs_tol=0.):
            raise ValueError('Saved ' + key + ' differs from this law/bounds')
    account.update(dictionary=dictionary, r=config.rte_steps, K=config.finite_taylor_order,
        signal_source_requirement='caller-bound immutable same-input/state law and same-round saved signal',
        total_registered_rounds=7, alpha_axis=.05/14,
        full_rpe_improvement=None)
    return account


def optimistic_unit_price_ceiling(account, baseline, *, comparison_compiler_hash):
    """Necessary weighted mean unit-price ceiling against all seven baseline rounds.

    For nonnegative costs in all other rounds, a candidate total below B needs
    N_cos*C_cos + N_sin*C_sin < B in this round alone. Thus its shot-weighted
    average full-no-prep unit price must be below B/(N_cos+N_sin). Missing other
    round prices are not assigned zero in an aggregate: zero is only the
    explicitly optimistic relaxation used to obtain this necessary condition.
    """
    if (not account['feasible'] or account['shots'] is None
            or account['round_total_cost'] is not None or account['compiler_hash'] is not None):
        raise ValueError('A feasible unpriced conditional round account is required')
    if (account.get('total_registered_rounds') != 7 or account.get('alpha_axis') != .05/14
            or account['q'] not in (16, 32, 64)):
        raise ValueError('C0 seven-round allocation must remain fixed')
    if (baseline['scope'] != 'registered_rounds_only' or baseline['round_count'] != 7
            or baseline['maximum_round'] != 6):
        raise ValueError('A complete seven-round baseline budget is required')
    alpha = float(baseline['alpha_bound'])
    if not math.isfinite(alpha) or not 0 < alpha <= .05:
        raise ValueError('Baseline failure allocation differs from C0')
    context = baseline['context']
    if context.get('arm_hash') != 'strong_DF_PF_4_substeps_spin_fused':
        raise ValueError('Fixed strong DF-PF baseline required; PR aggregate is not this budget')
    for key in ('context_hash', 'backend_hash', 'cost_metric', 'cost_scope', 'delta_time', 'beta_rpe'):
        if account[key] != context[key]:
            raise ValueError('Baseline physical/comparison context differs: ' + key)
    if (account['cost_metric'] != 'cx_count'
            or account['cost_scope'] != RPE_HADAMARD_INTERROGATION_SCOPE
            or account['delta_time'] != .05 or account['beta_rpe'] != .30
            or not comparison_compiler_hash or context['compiler_hash'] != comparison_compiler_hash):
        raise ValueError('Explicit matching C0 metric/scope/compiler is required')
    total = _positive(baseline['total_cost'], 'Baseline total budget')
    n_cos = require_integer_count(account['shots']['cosine'], name='cosine shots', minimum=1)
    n_sin = require_integer_count(account['shots']['sine'], name='sine shots', minimum=1)
    count = n_cos + n_sin
    exact = Fraction.from_float(total) / count
    return dict(status='NECESSARY_UNIT_PRICE_CONDITION_ONLY', round_index=account['round_index'],
        q=account['q'], r=account['r'], K=account['K'], dictionary=account['dictionary'],
        quantum_interrogations_this_round=count,
        baseline_total_budget=total, comparison_compiler_hash=comparison_compiler_hash,
        strict_unit_price_ceiling=dict(numerator=exact.numerator, denominator=exact.denominator),
        unit_price_definition='(N_cos*E[C_cos]+N_sin*E[C_sin])/(N_cos+N_sin)',
        condition='shot-weighted expected full no-prep unit price < exact ceiling',
        omitted_round_costs='nonnegative only; their prices remain unassessed',
        round_price_lower_bound=None, full_candidate_total_cost=None,
        full_rpe_improvement=None, guarantee_status='not_certified',
        price_uncertainty_qualified=False,
        qualification='conditional on supplied baseline budget and validated saved signal; not sufficient')
