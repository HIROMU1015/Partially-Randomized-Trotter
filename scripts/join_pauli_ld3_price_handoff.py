#!/usr/bin/env python3
"""Metadata-only join of the published LD3 26-axis reference price with B's LD3 total-view index.

Reads PRICE_26AXIS.json at its publication commit and B's C_RESULT_JOIN.json (2de33a4f4); checks
that every (m, axis) row has the same original physical ID and VP shots; writes HANDOFF_BC.json.
No price, coefficient, total or Q3 is recomputed here (values are copied with their locators).

    python -B scripts/join_pauli_ld3_price_handoff.py <price commit>
"""
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = 'docs/research/a_pauli_ld3_cal_price_20261116'
B_JOIN = dict(commit='2de33a4f4', path='docs/research/pauli_ld3_vp_q3_results_20261116/C_RESULT_JOIN.json')


def git_json(commit, path):
    return json.loads(subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT))


def main(price_commit):
    price = git_json(price_commit, BASE + '/PRICE_26AXIS.json')
    if price['status'] != 'PRICE_RULE_SUPPORTED_MODEL_REFERENCE':
        raise ValueError('No supported LD3 price to hand off')
    join = git_json(B_JOIN['commit'], B_JOIN['path'])
    b = {(r['m'], r['axis']): r for r in join['rows']}
    rows, total = [], Fraction(0)
    for i, r in enumerate(price['rows']):
        jb = b[(r['m'], r['axis'])]
        if (jb['original_candidate_id'] != r['physical_id'] or jb['N'] != r['N']
                or jb['original_configuration_id'] != price['configuration_id']
                or jb['total_configuration_id'] != price['total_view']['id']):
            raise ValueError('B total-view index differs at m=%d %s' % (r['m'], r['axis']))
        total += Fraction(r['N_times_C_ref_exact'])
        rows.append(dict(m=r['m'], axis=r['axis'], q=r['q'], R=r['R'], events=r['events'], N=r['N'],
                         original_configuration_id=price['configuration_id'], original_candidate_id=r['physical_id'],
                         total_configuration_id=jb['total_configuration_id'], total_candidate_id=jb['total_candidate_id'],
                         binding_id=jb['binding_id'], allocation_id=jb['allocation_id'],
                         C_ref_exact=r['C_ref_exact'], N_times_C_ref_exact=r['N_times_C_ref_exact'],
                         value_source=r['value_source'],
                         price_source=dict(commit=price_commit, path=BASE + '/PRICE_26AXIS.json', selector=['rows', i]),
                         b_index_source=dict(B_JOIN, selector=['rows', join['rows'].index(jb)])))
    if total != Fraction(price['T_LD3']['numerator'], price['T_LD3']['denominator']) or len(rows) != 26:
        raise ValueError('Copied rows do not reproduce the published total')
    value = dict(schema='A_PAULI_LD3_PRICE_HANDOFF_BC_V1', kind='MODEL_REFERENCE_PRICE_NOT_E_C',
                 rule=price['rule'], coefficients=price['coefficients'],
                 coefficient_source=dict(commit=price_commit, path=BASE + '/PRICE_26AXIS.json', selector=['coefficients']),
                 calibration_cell=price['calibration_cell'], configuration_source=price['configuration_source'],
                 total_view=price['total_view'], VP=price['VP'], rows=rows,
                 T_LD3=price['T_LD3'], T_LD3_display=price['T_LD3_display'], LD2=price['LD2'],
                 difference_LD3_minus_LD2=price['difference_LD3_minus_LD2'], ratio_LD3_over_LD2=price['ratio_LD3_over_LD2'],
                 qualified_for_candidate=False, E_C=None, CI=None, G=None, winner=None, mandatory_stop=True)
    with (ROOT / BASE / 'HANDOFF_BC.json').open('x') as f:
        json.dump(value, f, indent=1); f.write('\n')
    print(json.dumps(dict(rows=len(rows), T_LD3=value['T_LD3_display'])))


if __name__ == '__main__':
    main(sys.argv[1])
