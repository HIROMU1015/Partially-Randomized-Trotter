"""Documentation arithmetic on frozen G10 v3 saved fields only (stdlib).

No synthesis, matrix/circuit calculation, sampler, budget/lower recomputation,
proposal search, LP or scientific runner import. Existing lower values are read.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

# Trusted fixed saved rationals can exceed the interpreter's string-digit limit.
# This affects documentation serialization only, not a science precision policy.
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[3]
ACCESS = Path('artifacts/track_b_g10_v3_review_access/2026-10-10')
OUT = Path('artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10')
BASE = 'e65e3c0680fff4cfc4243ed5d6c87429cd5ce753'


def summarize(root=ROOT):
    root = Path(root)
    inputs = {}

    def load(relative):
        if str(relative).lower().endswith('.npz'):
            raise PermissionError('NPZ rejected before access')
        raw = (root/relative).read_bytes()
        inputs[str(relative)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return json.loads(raw, parse_float=Decimal)

    def display(value):
        value = F(value)
        with localcontext() as context:
            context.prec = 20
            return str(Decimal(value.numerator)/Decimal(value.denominator))

    def number(value):
        return {'exact_rational': str(F(value)), 'decimal_display': display(value)}

    table = load(ACCESS/'resource_table_exact.json')
    assert table['original_result_commit'] == 'fcd3ea6217bc00b667180cec149a70102d75f07e'
    assert table['original_result_identity'] == {
        'bytes': 66842494, 'sha256': '64a867dd6cc8f5f535880607616dea72f13af1360c7468f91eef44f79c543a2f'}
    rows = {(row['saved_fields']['degree'], row['saved_fields']['arm']): row['saved_fields']
            for row in table['rows']}
    assert len(rows) == 17
    comparisons = []
    for m, control in [(3, 'closed_P3_tail'), (5, 'closed_P5_full'), (7, 'closed_P5_tail')]:
        full, closed = rows[m, 'full_return'], rows[m, control]
        delta_T = F(full['two_axis_expected_native_cost']['T'])-F(closed['two_axis_expected_native_cost']['T'])
        delta_K = F(full['T_prep_readout_affine_coefficient'])-F(closed['T_prep_readout_affine_coefficient'])
        assert delta_T > 0 and delta_K > 0
        comparisons.append({'degree': m, 'registered_control': control,
                            'delta_T_full_minus_control': number(delta_T),
                            'delta_K_full_minus_control': number(delta_K),
                            'T_relative_difference': number(delta_T/F(closed['two_axis_expected_native_cost']['T'])),
                            'K_relative_difference': number(delta_K/F(closed['T_prep_readout_affine_coefficient'])),
                            'canonical_full_T_more_costly_for_all_common_h_nonnegative': True})
    m3p, m3c = rows[3, 'partial_return_tail'], rows[3, 'closed_P3_tail']
    h3 = (F(m3c['two_axis_expected_native_cost']['T'])-F(m3p['two_axis_expected_native_cost']['T'])) / (
        F(m3p['T_prep_readout_affine_coefficient'])-F(m3c['T_prep_readout_affine_coefficient']))
    full, closed, cts = rows[7, 'full_return'], rows[7, 'closed_P5_tail'], rows[7, 'matched_CTS']
    lower = full['fixed_dictionary_policy_lower']
    lower_T_gap = F(lower['intercept_lower'])-F(closed['two_axis_expected_native_cost']['T'])
    lower_K_gap = F(lower['prep_slope_lower'])-F(closed['T_prep_readout_affine_coefficient'])
    assert lower_T_gap > 0 and lower_K_gap < 0
    lower_edge = -lower_T_gap/lower_K_gap
    assert lower_T_gap+970*lower_K_gap > 0
    cx_edge = (F(closed['two_axis_expected_native_cost']['CX'])-F(cts['two_axis_expected_native_cost']['CX'])) / (
        F(cts['T_prep_readout_affine_coefficient'])-F(closed['T_prep_readout_affine_coefficient']))
    cts_checks = []
    for m, arm in [(5, 'closed_P5_full'), (7, 'closed_P5_tail')]:
        control, c = rows[m, arm], rows[m, 'matched_CTS']['fixed_dictionary_policy_lower']
        dt = F(c['intercept_lower'])-F(control['two_axis_expected_native_cost']['T'])
        dk = F(c['prep_slope_lower'])-F(control['T_prep_readout_affine_coefficient'])
        assert dt > 0 and dk > 0
        cts_checks.append({'degree': m, 'control': arm, 'delta_saved_CTS_lower_intercept': number(dt),
                           'delta_saved_CTS_lower_slope': number(dk), 'positive_for_all_h_nonnegative': True})
    root_rows = []
    for row_id, arm in [(14, 'full_return'), (15, 'closed_P5_tail')]:
        part = load(ACCESS/f'events/row_{row_id:02d}_m7_{arm}_part_000.json')
        roots = part['bindings'][:3]
        assert part['row_index'] == row_id and part['event_start'] == 0
        assert [binding['event']['child'] for binding in roots] == [0, 1, 2]
        assert all(binding['event']['word'] == [] for binding in roots)
        share = sum(F(binding['event']['proposal']) for binding in roots)/F(rows[7, arm]['reference_acceptance'])
        root_rows.append({'arm': arm, 'original_event_indices': [0, 1, 2],
                          'root_tangent': roots[0]['event']['ratio'],
                          'native_T_by_child': [binding['cost']['T'] for binding in roots],
                          'share_of_accepted_circuits': number(share)})
    assert [a-b for a,b in zip(root_rows[0]['native_T_by_child'],root_rows[1]['native_T_by_child'])] == [4,4,4]
    affine = [{'row_index': item['row_index'], 'degree': item['saved_fields']['degree'],
               'arm': item['saved_fields']['arm'], 'T_intercept': item['saved_fields']['two_axis_expected_native_cost']['T'],
               'prep_T_slope': item['saved_fields']['T_prep_readout_affine_coefficient'],
               'native_vector': item['saved_fields']['two_axis_expected_native_cost'],
               'acceptance': item['saved_fields']['reference_acceptance'],
               'N_per_axis': item['saved_fields']['budget']['N_per_axis'],
               'fixed_dictionary_policy_lower': item['saved_fields']['fixed_dictionary_policy_lower']}
              for item in table['rows']]
    return {'kind': 'G10_V3_DOCUMENTATION_ONLY_SAVED_FIELD_ARITHMETIC', 'status': 'PASS',
            'input_commit': BASE, 'input_identities': inputs, 'original_result_identity': table['original_result_identity'],
            'canonical_full_comparisons': comparisons, 'm3_partial_P3_prep_T_boundary': number(h3),
            'm7_saved_full_lower_minus_registered_P5': {'intercept': number(lower_T_gap),
                'slope': number(lower_K_gap), 'lower_separation_endpoint': number(lower_edge),
                'gap_at_h_970': number(lower_T_gap+970*lower_K_gap),
                'meaning': 'Fixed-dictionary sufficient-budget lower vs achieved registered budget; not actual crossover or quantum query lower.'},
            'm7_P5_CTS_prep_CX_boundary': number(cx_edge), 'CTS_saved_lower_checks': cts_checks,
            'm7_root_saved_examples': root_rows, 'all_row_saved_affine_fields': affine,
            'lower_recomputed_or_sampling_optimized': False, 'outcome_reclassified': False,
            'synthesis_matrix_circuit_sampling_LP_DF_GPU_new_science': 0,
            'mandatory_STOP': True, 'next_science_authorized': False}


if __name__ == '__main__':
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))
