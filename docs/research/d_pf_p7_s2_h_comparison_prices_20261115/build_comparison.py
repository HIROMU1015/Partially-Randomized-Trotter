"""Fixed saved-price/Q3 join and adopted-rule arithmetic; stdlib only.

No circuit, QPY, simulator, Q3 evaluator, optimizer or fit is imported.
The comparison table is separate from the immutable direct-price receipts.
"""
import csv
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIG = 'e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc'
RULE = 'B0I_P7_S2_H_5513Q_PLUS307_COMPARISON_V1'
PRICE = dict(commit='e04a14797a56a586dcb72065a55f4d97bfa1479b',
             path='docs/research/d_pf_p7_s2_h_direct_prices_20261115/run01/AXIS_PRICES.json')
ACCURACY_DIR = 'docs/research/p7_s2_d01_same_shots_q3_results_20261115/'
JOIN = dict(commit='bec032fc26093177f529754060df06d5375944b4', path=ACCURACY_DIR+'C_RESULT_JOIN.json')
SUMMARY = dict(commit=JOIN['commit'], path=ACCURACY_DIR+'SUMMARY.json')
SHOTS = [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(ref):
    raw = subprocess.check_output(['git', 'show', ref['commit']+':'+ref['path']], cwd=ROOT)
    return json.loads(raw), dict(ref, sha256=hashlib.sha256(raw).hexdigest())


def rational(value):
    return Fraction(value['numerator'], value['denominator'])


def fraction_record(value):
    value = Fraction(value)
    with localcontext() as ctx:
        ctx.prec = 40
        display = str(Decimal(value.numerator) / Decimal(value.denominator))
    return dict(numerator=value.numerator, denominator=value.denominator, decimal=display)


def build(prices, joined, summary):
    require(prices['configuration_id'] == joined['configuration_id'] == CONFIG, 'Configuration mismatch')
    require(summary['case']['configuration_id'] == CONFIG, 'Q3 summary configuration mismatch')
    allocation = joined['allocation']
    require(allocation == summary['case']['allocation'], 'Q3 allocation mismatch')
    expected_shots = [dict(m=m, N_X=n, N_Y=n) for m, n in enumerate(SHOTS)]
    require(allocation['shots'] == expected_shots, 'Not the fixed user-supplied allocation')
    require(joined['model_status'] == summary['status'] == summary['coarse_model_status'] ==
            'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', 'Saved Q3 acceptance missing')
    physical = {r['m']: r for r in joined['physical_rows']}
    require(len(physical) == len(joined['physical_rows']) == 13 and set(physical) == set(range(13)),
            'Incomplete or duplicate schedule')
    keys = {(r['m'], r['axis']) for r in prices['rows']}
    require(len(prices['rows']) == len(keys) == 26 and keys ==
            {(m, a) for m in range(13) for a in ('X', 'Y')}, 'Incomplete or duplicate price axes')
    rows = []
    checks = []
    direct_cost = extrapolated_cost = 0
    costs = []
    for index, saved in sorted(enumerate(prices['rows']), key=lambda pair: (pair[1]['m'], pair[1]['axis'])):
        m, axis = saved['m'], saved['axis']
        candidate = physical[m]
        require(saved['configuration_id'] == CONFIG and saved['physical_id'] ==
                candidate['physical_candidate']['id'], 'Physical identity mismatch')
        require(saved['physical_candidate'] == candidate['physical_candidate'], 'Physical schedule mismatch')
        require(candidate['physical_candidate']['method'] == 'B0I' and
                candidate['physical_candidate']['prefix'] == 7 and
                candidate['physical_candidate']['formula'] == '2nd', 'Wrong route/prefix/formula')
        q = saved['q']
        require(q == candidate['q_PF'] == 2**m, 'q mismatch')
        require(Fraction.from_float(saved['T']) == rational(candidate['T']) == q*Fraction.from_float(.1),
                'Total-time binary64 mismatch')
        require(Fraction.from_float(saved['actual_delta']) == rational(candidate['actual_delta_pf']) ==
                Fraction.from_float(.1), 'Actual step width mismatch')
        require(saved['T_binary64_hex'] == saved['T'].hex() and
                saved['actual_delta_binary64_hex'] == (.1).hex(), 'Binary64 metadata mismatch')
        reference = 5513*q + 307  # Adopted coefficients, never fitted here.
        direct = m <= 4
        if direct:
            require(saved['status'] == 'ACCEPTED_DIRECT' and saved['receipt'] and saved['sidecar'],
                    'Accepted direct receipt missing')
            require(saved['gate_result']['C_guard'] == saved['gate_result']['original_action_gates'] == 'PASS',
                    'Direct implementation gate not passed')
            price = saved['compiled_RZ']  # Preserve the observation, even though it equals reference.
            require(type(price) is int and price == reference, 'Saved direct price differs from adopted rule')
            checks.append(dict(m=m, q=q, axis=axis, observed_RZ=price, rule_RZ=reference,
                               difference_RZ=price-reference, role='RULE_SELECTION_OBSERVATION_NOT_HOLDOUT'))
        else:
            require(saved['status'] == 'NOT_ACQUIRED' and saved['compiled_RZ'] is None and
                    saved['receipt'] is None, 'Unobserved axis has conflicting direct evidence')
            price = reference
        row = {k: deepcopy(saved[k]) for k in ('configuration_id', 'physical_id', 'original_canonical_id',
                                             'm', 'axis', 'T', 'q', 'actual_delta', 'T_binary64_hex',
                                             'actual_delta_binary64_hex', 'physical_candidate')}
        row.update(unit_price_RZ=price,
                   origin='DIRECT_COMPILED' if direct else 'ADOPTED_RULE_EXTRAPOLATION',
                   direct_compiled_RZ=saved['compiled_RZ'], direct_evidence_status=saved['status'],
                   direct_receipt=deepcopy(saved['receipt']),
                   direct_sidecar=deepcopy(saved.get('sidecar')),
                   direct_gate_result=deepcopy(saved['gate_result']),
                   input_row=dict(PRICE, selector=['rows', index]),
                   rule_id=RULE, rule_error_bound_RZ=None,
                   independent_rule_validation=False)
        rows.append(row)
        n = expected_shots[m]['N_'+axis]
        contribution = n*price
        costs.append(dict(m=m, axis=axis, physical_id=saved['physical_id'], q=q, shots=n,
                          unit_price_RZ=price, origin=row['origin'], contribution_RZ=contribution))
        if direct:
            direct_cost += contribution
        else:
            extrapolated_cost += contribution
    total = direct_cost + extrapolated_cost
    require(total == 2*sum(n*(5513*(2**m)+307) for m, n in enumerate(SHOTS)), 'Independent arithmetic check')
    coarse = summary['comparison_source']['coarse_conditional_cost_RZ']
    require(coarse == 435481090 and summary['comparison_source']['coarse_cost_applies_to'] ==
            'S2_16h_over_7_ONLY', 'Coarse cost reference mismatch')
    table = dict(schema='P7_S2_H_COMPARISON_PRICE_TABLE_V1', configuration_id=CONFIG,
                 quantity='RZ_COUNT_PER_MEASURED_FULL_CONTROLLED_HADAMARD_SHOT_P0_EXCLUDED',
                 status='USER_ADOPTED_COMPARISON_RULE_NOT_FULL_DIRECT_EVIDENCE',
                 rule=dict(id=RULE, expression='5513*q+307', slope=5513, intercept=307,
                           observed_q=[1, 2, 4, 8, 16], extrapolated_q=[2**m for m in range(5, 13)],
                           rule_selection_physical_points=5, rule_selection_axis_observations=10,
                           independent_holdout_axes=0, fit_performed_here=False,
                           error_bound_RZ=None, rigorous_bound=False,
                           scope='ONLY_FIXED_CONFIGURATION_S2_ACTUAL_BINARY64_WIDTH_0.1_XY_M0_TO_M12'),
                 price_definition_source=dict(PRICE, path=PRICE['path'].replace('run01/AXIS_PRICES.json', 'HANDOFF_B.md')),
                 source_price_table=deepcopy(PRICE), source_accuracy_join=deepcopy(JOIN),
                 counts=dict(total=26, direct=10, extrapolated=16, missing_comparison_prices=0,
                             independently_validated_long_q=0),
                 direct_rule_checks=checks, rows=rows)
    cost = dict(schema='P7_S2_H_FIXED_ALLOCATION_MODEL_COST_V1',
                configuration_id=CONFIG, total_view_id=joined['total_view_id'], binding_id=joined['binding_id'],
                status='CONDITIONAL_MODEL_TOTAL_WITH_EXTRAPOLATED_PRICES', rule_id=RULE,
                allocation=deepcopy(allocation), per_axis_contributions=costs,
                total_RZ=total, direct_part_RZ=direct_cost, extrapolated_part_RZ=extrapolated_cost,
                direct_fraction=fraction_record(Fraction(direct_cost, total)),
                extrapolated_fraction=fraction_record(Fraction(extrapolated_cost, total)),
                extrapolated_percentage=fraction_record(Fraction(100*extrapolated_cost, total)),
                coarse_width_16h_over_7_achieved_RZ=coarse,
                difference_h_minus_16h_over_7_RZ=total-coarse,
                percentage_change_from_coarse=fraction_record(Fraction(100*(total-coarse), coarse)),
                Q3_saved_quote=dict(source=deepcopy(SUMMARY),
                                    fine_status=summary['status'], coarse_status=summary['coarse_model_status'],
                                    fine_MSE_Ha2=summary['fine_MSE'], coarse_MSE_Ha2=summary['coarse_MSE'],
                                    comparison=deepcopy(summary['accuracy_comparison']),
                                    new_Q3_calls=0),
                price_model_uncertainty_RZ=None, certified_total=False, G_qualified=None,
                sufficient_N=None, optimized_allocation=None, winner=None,
                new_compile_calls=0, QPY_audit_calls=0, fit_calls=0, new_Q3_calls=0,
                new_shot_allocation_calls=0, mandatory_stop=True)
    return table, cost


def main():
    started = time.perf_counter()
    module = str(Path(__file__).relative_to(ROOT))
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(subprocess.check_output(['git', 'show', source+':'+module], cwd=ROOT) == Path(__file__).read_bytes(),
            'Commit this script before fixed-input arithmetic')
    prices, p_ref = read(PRICE)
    joined, j_ref = read(JOIN)
    summary, s_ref = read(SUMMARY)
    table, cost = build(prices, joined, summary)
    provenance = dict(execution_SOURCE=source, caller=module, inputs=[p_ref, j_ref, s_ref])
    table['provenance'] = cost['provenance'] = provenance
    out = HERE/'run01'
    out.mkdir(exist_ok=False)  # Preserve any prior output.
    for filename, data in [('COMPARISON_PRICES.json', table), ('FIXED_ALLOCATION_COST.json', cost)]:
        (out/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    fields = ['configuration_id', 'physical_id', 'm', 'axis', 'q', 'T', 'actual_delta',
              'unit_price_RZ', 'origin', 'direct_compiled_RZ', 'direct_receipt', 'rule_id']
    with (out/'COMPARISON_PRICES.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in table['rows']:
            record = {k: row[k] for k in fields}
            record['direct_receipt'] = json.dumps(record['direct_receipt'], separators=(',', ':'))
            writer.writerow(record)
    print(json.dumps(dict(total_RZ=cost['total_RZ'], direct_part_RZ=cost['direct_part_RZ'],
                          extrapolated_part_RZ=cost['extrapolated_part_RZ'],
                          extrapolated_percentage=cost['extrapolated_percentage'],
                          difference_RZ=cost['difference_h_minus_16h_over_7_RZ'],
                          wall_seconds=time.perf_counter()-started, source=source)))


if __name__ == '__main__':
    main()
