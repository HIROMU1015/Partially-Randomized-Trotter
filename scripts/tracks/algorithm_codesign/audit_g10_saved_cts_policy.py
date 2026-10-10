"""One saved-only audit; reads G9 raw coefficients, never a synthesis backend."""
import hashlib
import json
from pathlib import Path
from fractions import Fraction as F
from trottertracks.algorithm_codesign.g10_saved import (
    affine_policy_lower, cts_coarse_check, serial,
)

ROOT = Path(__file__).resolve().parents[3]
INPUT = 'artifacts/track_b_g9_p5_native_result/2026-10-10/v2/result_v1.json'
EXPECTED = '3eb8430014a7881b5189ca8779a58ccd7ab2f50c37a58795bd0458b174abda20'
OUTPUT = 'artifacts/track_b_g10_degree_preparation/2026-10-10/saved_policy_audit_v1.json'


def execute():
    raw = (ROOT/INPUT).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise PermissionError('G9 saved input identity changed')
    result = json.loads(raw)
    if result['status'] != 'G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE':
        raise PermissionError('incomplete saved rows cannot support this audit')
    rows = {r['arm']: r for r in result['rows'] if r['primary']}
    closed = rows['closed_P5_full']
    output = {'kind': 'G10_A_SAVED_ONLY_FIXED_POLICY_AUDIT',
              'input_commit': 'c95736fd2990f5ef6dd1cb5866421fd78bb28687',
              'input': INPUT, 'input_sha256': EXPECTED,
              'CTS_coarse_review_bound': cts_coarse_check(rows['matched_CTS'], closed),
              'all_fixed_dictionary_policy_lowers': {},
              'old_status_and_markers_unchanged': True,
              'new_synthesis_sampler_matrix_circuit_solver_science': 0,
              'fixed_dictionary_fixed_precision_fixed_confidence_only': True}
    for arm, row in rows.items():
        lower = affine_policy_lower(row['events'], row['budget']['remaining'],
                                    row['budget']['alpha_axis'])
        lower['closed_P5_separated_for_all_h_ge_0'] = (
            F(closed['two_axis_expected_native_cost']['T']) < lower['intercept_lower']
            and F(closed['T_prep_readout_affine_coefficient']) <= lower['prep_slope_lower'])
        lower['nonseparation_is_not_negative_evidence'] = True
        output['all_fixed_dictionary_policy_lowers'][arm] = lower
    path = ROOT/OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(serial(output), stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'CTS_all_h_bound': True, 'rows': len(rows),
                      'new_science_runs': 0, 'output': OUTPUT}))


if __name__ == '__main__':
    execute()
