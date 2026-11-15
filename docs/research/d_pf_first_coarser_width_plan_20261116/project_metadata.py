"""Four-candidate projection of completed coverage; no scientific evaluation.

Reuse the existing cached Git/JSON locator reader, not its inventory builder.
Never loads arrays/QPY or invokes signal/price evaluators or runtime modules.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'d_pf_width_price_coverage_20261115'))
from build_metadata import loc, resolve, require

B = loc('7f27e18f06dde12127c85be898777b685d29bacc',
        'docs/research/pf_joint_width_shot_plan_20261115/WIDTH_SIGNAL_COVERAGE.json')
D = loc('486fa84ccf2b63c73f71d3adcedada72dc6ebe99',
        'docs/research/d_pf_width_price_coverage_20261115/PRICE_COVERAGE.json')
Q = loc('b4296b110b129d226dbef2743c14f725a27c3bf6',
        'docs/research/pf6_cost_budget_q3_results_20261115/SUMMARY.json')
SELECTED = 'dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0'
PREFIXES = ('7754671b', '5576418d', '9e930a47', 'dd8bfc24')
ZERO = dict(new_signal=0, new_price=0, build=0, compile=0, QPY_audit=0,
            fit=0, allocation_generation=0, Q3=0, weighted_cost=0)


def saved_width(row):
    if 'actual_delta_over_h' in row:
        return row['actual_delta_over_h']
    value = row['actual_delta_pf']
    return str(Fraction(value['numerator'], value['denominator']) / Fraction(.1))


def main():
    require(saved_width(dict(actual_delta_over_h='16/3')) ==
            saved_width(dict(actual_delta_pf=dict(numerator=3602879701896397,
                                                 denominator=6755399441055744))) == '16/3',
            'Full/B0I saved width schema fixture')
    signals, prices, q3 = resolve(B), resolve(D), resolve(Q)
    candidates = []
    for prefix in PREFIXES:
        bi, b = next((i, c) for i, c in enumerate(signals['candidates'])
                     if c['configuration_id'].startswith(prefix))
        di, d = next((i, c) for i, c in enumerate(prices['candidates'])
                     if c['configuration_id'] == b['configuration_id'])
        current = next(c for c in signals['candidates'] if c['current_width'] and
                       c['method'] == b['method'] and c['formula'] == b['formula'])
        qi, baseline = next((i, r) for i, r in enumerate(q3['rows']) if
                            r['configuration_id'] == current['configuration_id'] and
                            r['label'].endswith('__B_PRICE'))
        old_schedule = resolve(current['schedule'])
        saved = b['current_saved_mean_rounds']
        require(saved == d['signal_rounds']['SAVED_CURRENT_MEAN_NEEDS_CANDIDATE_PACKET_BINDING'],
                'B/D saved-signal coverage differs')
        require(b['missing_current_mean_rounds'] == d['signal_rounds']['NEW_MEAN_NEEDED'],
                'B/D missing-signal coverage differs')
        schedule = []
        for m in range(13):
            row = next(r for r in prices['rows'] if r['configuration_id'] == b['configuration_id']
                       and r['m'] == m and r['axis'] == 'X')
            old = old_schedule[m]
            schedule.append(dict(m=m, T=row['physical_candidate']['T'],
                q=row['physical_candidate']['q'], actual_delta_over_h=row['actual_delta_over_h'],
                physical_id=row['physical_id'], current_q=old['q_PF'],
                current_actual_delta_over_h=saved_width(old),
                changed=(row['physical_candidate']['q'] != old['q_PF'] or
                         row['actual_delta_over_h'] != saved_width(old))))
        candidates.append(dict(method=b['method'], formula=b['formula'],
            requested_delta_over_h=b['requested_delta_over_h'], configuration_id=b['configuration_id'],
            signal_coverage=loc(B['commit'], B['path'], 'candidates', bi),
            price_coverage=loc(D['commit'], D['path'], 'candidates', di),
            schedule_source=deepcopy(b['schedule']), physical_schedule=schedule,
            saved_mean_rounds=saved, missing_mean_rounds=b['missing_current_mean_rounds'],
            saved_mean_sources=deepcopy(b['round_source_locators']),
            complete_signal_packet=b['source_bound_13round_packet'],
            direct_price_evidence_rounds=d['saved_direct_rounds'],
            direct_price_evidence_axes=d['price_counts']['DIRECT_SAME_PHYSICAL_EVIDENCE'],
            missing_price_axes=d['missing_price_axes'],
            adopted_coarse_price_rule=None, consumer_binding='REQUIRED_BEFORE_SCIENTIFIC_INTAKE',
            current_configuration_id=current['configuration_id'],
            current_configuration_source=current['configuration_locator'],
            current_signal_packet=current['source_bound_13round_packet'],
            current_schedule_source=current['schedule'],
            current_comparison_point=loc(Q['commit'], Q['path'], 'rows', qi),
            saved_current_cost_RZ=baseline['conditional_cost'], saved_current_Q3_status=baseline['status']))
    chosen = next(c for c in candidates if c['configuration_id'] == SELECTED)
    baseline = resolve(chosen['current_comparison_point'])
    rows = [(i, r) for i, r in enumerate(prices['rows']) if r['configuration_id'] == SELECTED]
    reused_prices = [dict(m=r['m'], axis=r['axis'], physical_id=r['physical_id'],
                         source=loc(D['commit'], D['path'], 'rows', i),
                         evidence=deepcopy(r['price_evidence']), consumer_binding=r['consumer_binding'])
                     for i, r in rows if r['price_status'] == 'DIRECT_SAME_PHYSICAL_EVIDENCE']
    require(len(reused_prices) == 10, 'Expected ten same-physical direct evidence axes')
    plan = dict(status='PROPOSAL_ONLY_NOT_SCIENTIFIC_EXECUTION_AUTHORIZATION',
        configuration_id=SELECTED, method='B0I_P7', formula='4th', requested_delta_over_h='32/5',
        h_RPE=0.1, schedule_source=chosen['schedule_source'],
        saved_mean_sources=chosen['saved_mean_sources'],
        new_mean_requests=[deepcopy(r) for r in chosen['physical_schedule'] if r['m'] in chosen['missing_mean_rounds']],
        saved_step_operator=loc('5fb2c32e39e63267ea1748861269c362a93a1b86',
             'docs/research/truncated_pf_t2_results_20261114/run01/cell_p7_4th_32o5/OPERATORS.npz'),
        saved_step_field='PF_step_without_global_scalars',
        saved_step_scope=loc('5fb2c32e39e63267ea1748861269c362a93a1b86',
             'docs/research/truncated_pf_t2_results_20261114/run01/cell_p7_4th_32o5/CELL_SCOPE.json'),
        fixed_allocation=deepcopy(baseline['allocation']), fixed_allocation_source=chosen['current_comparison_point'],
        allocation_reuse='EXACT_INTEGER_VECTOR_ONLY_OLD_ID_PRESERVED_AS_SOURCE',
        future_coarse_allocation_id=None, Q3_contract=loc(Q['commit'], Q['path'], 'adopted_evaluation'),
        current_signal_packet=chosen['current_signal_packet'],
        future_limits=dict(reused_mean_rounds=7, new_complex_mean_rounds=6,
            new_Q3_whole_allocation_calls=1, new_reference_evaluation=0, new_operator_build=0,
            new_circuit_build=0, new_compile=0, fit=0, new_allocation=0, retries=0),
        Q3_condition='Only after 13 source-bound rounds and original numerical/intake gates pass',
        saved_direct_price_evidence=reused_prices, missing_price_axes=chosen['missing_price_axes'],
        new_price_evidence_in_first_stage=0,
        cost_challenges=dict(same_formula_current_RZ=435641568, p7_best_saved_RZ=435481090),
        subsequent_price_pilot='Not part of first stage; separately approve m5/m6/m9 X/Y, at most 6 wrappers, only after deciding to retain this candidate',
        actual_scientific_calls=deepcopy(ZERO), mandatory_stop=True)
    comparison = dict(status='FOUR_CANDIDATE_METADATA_DELTA_ONLY', candidates=candidates,
        recommended_configuration_id=SELECTED, actual_scientific_calls=deepcopy(ZERO), mandatory_stop=True)
    for name, value in [('CANDIDATES.json', comparison), ('FIRST_FINITE_PLAN.json', plan)]:
        path=HERE/name
        require(not path.exists(), 'Preserve existing projection: '+name)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print('PASS: four existing coverage projections; seven reused and six missing signal rounds; no scientific calls.')


if __name__ == '__main__':
    main()
