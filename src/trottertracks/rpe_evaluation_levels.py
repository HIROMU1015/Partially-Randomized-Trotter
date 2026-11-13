"""Two passive evidence views, never a nominal-to-sufficient N conversion.

This is readiness, not an accounting controller. No N formula, G, risk, winner
or qualification is produced. Actual qualified accounting stays in the guarded
B/W entry; a future paper-model accounting caller must retain its nominal scope.
"""
from copy import deepcopy

from . import rpe_price_implementation_guard as R

B, p = R.B, R.B.p
PAPER = 'PAPER_EMPIRICAL_NOMINAL_ONLY'
MSE = 'MSE_QUALIFIED'


def prepare(config, *, paper_packet=None, mse_packet=None, prices=(), cohorts=None,
            root=None, price_bindings=(), comparison_policy=None, price_policy='qualified_expectation'):
    """Keep contracts, SOURCES, coverage and missing data separate; no arithmetic.

    Supplied paper rows use nominal_N_X/Y; MSE packets keep the existing B
    schema. Only B.assess handles MSE eligibility. The implementation guard
    precedes all B price inspection when a policy is supplied.
    """
    B._validate(config); cohorts = {} if cohorts is None else cohorts
    checked = None
    if comparison_policy is not None:
        if root is None:
            raise ValueError('Root required for price implementation guard')
        checked = R.check(root, config, prices, cohorts=cohorts, bindings=price_bindings,
                          policy=comparison_policy)
    if paper_packet is not None:
        if (paper_packet.get('status') != PAPER or not paper_packet.get('source')
                or paper_packet.get('configuration_id') != config['id']
                or not p.same(paper_packet.get('policy'), config['spec']['rmse_policy'])
                or paper_packet.get('rigorous_sufficient_N') is not False
                or paper_packet.get('sufficient_N') is not None
                or paper_packet.get('global_RMSE_bound') is not None):
            raise ValueError('Source-bound paper nominal model packet required')
        rows = paper_packet.get('rounds', [])
        if [r['m'] for r in rows] != [r['m'] for r in config['rounds']]:
            raise ValueError('Whole nominal schedule required')
        for row, canonical in zip(rows, config['rounds']):
            if (not row.get('source') or not p.same(row.get('identity'), canonical['identity'])
                    or not p.same(row.get('law_binding'), canonical['law_binding'])
                    or any(type(row.get('nominal_N_' + axis)) is not int or row['nominal_N_' + axis] < 1
                           for axis in p.AXES)):
                raise ValueError('Nominal identity/law/SOURCE or exact integer N mismatch')
    if mse_packet is not None and mse_packet.get('status') == PAPER:
        raise ValueError('Nominal evidence is not an MSE N packet')
    assessment = B.assess(config, mse_packet, prices, cohorts=cohorts, price_policy=price_policy)
    price_coverage = [dict(m=c['m'], axis=c['axis'], C=deepcopy(c['C'])) for c in assessment['coverage']]
    price_ready = all(c['C']['qualification'] == 'AVAILABLE' for c in price_coverage) and checked is not None
    def view(level, packet, ready, reasons):
        return dict(level=level, configuration_id=config['id'], source=deepcopy((packet or {}).get('source')),
            contract=deepcopy(config['spec']['rmse_policy']), N_input=deepcopy(packet),
            price_input=dict(records=deepcopy(prices), cohorts=deepcopy(cohorts),
                bindings=deepcopy(price_bindings), policy=deepcopy(comparison_policy),
                implementation_check=deepcopy(checked), price_policy=price_policy),
            price_coverage=deepcopy(price_coverage), readiness='INPUTS_READY' if ready else 'UNKNOWN',
            reasons=reasons, G=None, winner=None, sufficient_N=None, global_RMSE_bound=None,
            scientific_qualification_issued=False)
    price_reasons = ([] if price_ready else ['MISSING_OR_UNQUALIFIED_C_OR_IMPLEMENTATION_POLICY'])
    return dict(paper=view(PAPER, paper_packet, paper_packet is not None and price_ready,
                    ([] if paper_packet is not None else ['MISSING_NOMINAL_PACKET']) + price_reasons),
        mse=view(MSE, mse_packet, assessment['eligible'] and price_ready,
                    list(assessment['reasons']) + price_reasons),
        mse_assessment=assessment, statistics_executed=False, cost_evaluated=False,
        normalization_is_price_multiplier=False, classical_samples_are_shot_multipliers=False)
