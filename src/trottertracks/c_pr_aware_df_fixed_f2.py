"""Fixed F2 polar execution binding; reuse the unchanged CAR evaluator/runner.

Chat permission is sufficient. These checks bind scientific target and saved
evidence, not proof of user permission. No molecular decode during preflight.
"""
from pathlib import Path

from . import c1_stage1_execution as io
from . import c_pr_aware_df_fixed_car as car
from . import c_pr_aware_df_fixed_car_execution as execution

VERSION = 'fixed_F2_polar_execution_binding_v1'
SCOPE = 'C_PR_AWARE_DF_FIXED_F2_POLAR_V1'
BASE = '52185764ccda60ddb4be04465cc644ed56deeb98'
DRAFT = 'docs/research/research_c_pr_aware_df_fixed_car_contract_draft_20261027.json'
DRAFT_SHA256 = '670d40c62e9d336e8c23b16d2f79f036807280b4888e31c02587209b0f489d7f'
FRESH = 'artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-run01'


def scope_gate(c, draft):
    """The new adopted target never mutates/reuses the old executable=false."""
    if (c['scope'] != SCOPE or c['execution_binding'] != VERSION
            or c['execution_base_commit'] != BASE or c['experiment'] != 'F2'
            or c['arm'] != 'baseline' or c['trajectory_index'] != 0
            or c['target'] != car.TARGET or c['target_policy_status'] != 'ADOPTED_FIXED_F2_ONLY'
            or c['executable'] is not True or c['execution_blockers']
            or c['fresh_output'] != FRESH or c['mandatory_stop'] is not True
            or c['call_limits'] != execution.LIMITS['F2'] or c['maximum_logical_units'] != 7
            or c['precision_bits'] != 192 or c['allowance_exact'] != '1/100000000'
            or c['operator_norm'] != 'phase_sensitive_full512_operator2'
            or c['global_phase_fit'] is not False or c['retry'] or c['resume']):
        raise ValueError('FIXED_F2_TARGET_SCOPE')
    if draft['executable'] is not False or draft['target_policy_status'] != 'PROPOSED_NOT_ADOPTED':
        raise ValueError('OLD_DRAFT_NOT_PRESERVED')
    # Preserve numerical semantics/input policy. Only the new target adoption,
    # SOURCE/contract/output binding differs from the old readiness proposal.
    for key in ('conditions', 'additional_conditions', 'compiler', 'packages', 'circuits',
                'input', 'export', 'trajectory', 'sampling_SOURCE_sha256',
                'original_input_sha256', 'reuse', 'literal_target', 'ideal_target',
                'reference_method', 'reference_uncertainty', 'reference_unitarity'):
        if c[key] != draft[key]: raise ValueError('FIXED_F2_POLICY_CHANGED:' + key)
    for path, digest in draft['saved_inputs'].items():
        if c['saved_inputs'].get(path) != digest:
            raise ValueError('FIXED_F2_SAVED_INPUT_CHANGED:' + path)


def prior_gate(c, b2_source, b2_stop, b1_contract, b1_pair):
    """Same raw QPY, phase, dimension and existing certificate; no reanalysis."""
    from fractions import Fraction
    raw = c['circuits']['raw']
    if (b2_source['execution_commit'] != c['B2_execution_SOURCE_commit']
            or b2_source['sampling_SOURCE_sha256'] != c['sampling_SOURCE_sha256']
            or b2_stop['failed'] or b2_stop['scientific_units'] != 652
            or b2_stop['retry_within_run'] or b2_stop['resume'] or not b2_stop['mandatory_stop']):
        raise ValueError('FIXED_F2_B2_PROVENANCE')
    if (b1_contract['circuits']['kernel_raw'] != raw
            or b1_contract['saved_inputs'][raw] != c['saved_inputs'][raw]
            or c['reuse']['raw_sha256'] != c['saved_inputs'][raw]
            or b1_contract['expected_qubits'] != 9
            or b1_contract['global_phase_fit'] is not False
            or b1_pair['dimension'] != 512 or b1_pair['columns_evaluated'] != 512
            or b1_pair['norm'] != 'full_operator_2' or b1_pair['phase_fit'] is not False
            or b1_pair['status'] != 'VALIDATED_FIXED_CIRCUIT_PAIR'
            or b1_pair['gate_status'] != 'VALIDATED_WITHIN_FIXED_ALLOWANCE'
            or b1_pair['precision_bits'] != 192
            or Fraction(b1_pair['upper_exact']) > Fraction(c['allowance_exact'])):
        raise ValueError('FIXED_F2_TRIANGLE_RAW_PHASE_DIMENSION')


def preflight(root, path):
    root = Path(root).resolve()
    if io.sha(root/DRAFT) != DRAFT_SHA256: raise ValueError('OLD_DRAFT_HASH')
    c = io.read(path)
    scope_gate(c, io.read(root/DRAFT))
    result = execution.preflight(root, path)
    for name in (DRAFT, c['B2_source_record'], c['B2_stop_record'], c['B1_contract_record'],
                 c['reuse']['compiled_raw']):
        execution._pinned(root, c, name)
    prior_gate(c, io.read(root/c['B2_source_record']), io.read(root/c['B2_stop_record']),
               io.read(root/c['B1_contract_record']), io.read(root/c['reuse']['compiled_raw']))
    result.update(target=c['target'], target_policy_status=c['target_policy_status'],
                  old_draft_executable=False, reused_compiled_raw=True,
                  F3_started=False, scientific_calls=0)
    return result


def run(root, path):
    preflight(root, path)
    # This unchanged one-shot runner owns Calls, partial evidence and STOP.
    return execution.run(root, path)
