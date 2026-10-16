#!/usr/bin/env python3
"""Audit the saved one-shot only: schema, provenance, hashes and stored bounds.

No NPZ opening, reference import, Pauli generation, new W/v/rho or allowance
evaluation. Integer cross-products check inequalities already in the ledger;
this is not another scientific reference run or numerical correctness test.
"""
import argparse
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
A = '2a55ba0103c68a074b7be7bc43b90b383bb63233'
C = '3ec5721bda8ac449f61b77eb28b66dfcc5f0e8c8'
CONTRACT = 'artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/independent_validation_contract_adopted.json'
AUTH = 'artifacts/track_b_m1s_independent_execution_authorization/2026-10-16/v1/execution_authorization.json'
OUTPUT = 'artifacts/track_b_m1s_independent_reference_result/2026-10-16/v1_one_shot'
AUDIT = 'artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1'
RESULT_SHA = '32c73f57a1b3fada6bc185311bd56007628849222ad77e4a33e8383a2d0085ac'


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'openfermion', 'qiskit', 'zipfile'} or 'm1s_gate0' in fullname:
            raise RuntimeError('No science imports in saved-only audit: '+fullname)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.PIPE)


def strict_json(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError('Duplicate JSON key '+k)
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report')
    args = parser.parse_args()
    sys.meta_path.insert(0, NoScience())
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)) and str(arguments[0]).lower().endswith(('.npz', ".npz'")):
            raise RuntimeError('No NPZ opens in saved-only audit')
    sys.addaudithook(audit)
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'status': 'PASS' if ok else 'FAIL'})
        if not ok:
            raise AssertionError(name)
    def load(p):
        return strict_json((ROOT/p).read_bytes())
    def ratio(q):
        assert set(q) == {'numerator_hex', 'denominator_hex', 'decimal_display_only'}
        assert re.fullmatch(r'-?0x[0-9a-f]+', q['numerator_hex']) and re.fullmatch(r'0x[0-9a-f]+', q['denominator_hex'])
        n, d = int(q['numerator_hex'], 16), int(q['denominator_hex'], 16)
        assert d > 0 and max(abs(n).bit_length(), d.bit_length()) <= 65536
        return n, d
    def within_saved(a, b):
        an, ad = ratio(a); bn, bd = ratio(b)
        return an >= 0 and bn >= 0 and an*bd <= bn*ad
    def within_scalar(a, n, d):
        an, ad = ratio(a)
        return an >= 0 and an*d <= n*ad

    raw = (ROOT/OUTPUT/'result.json').read_bytes()
    r = strict_json(raw); config = load(CONTRACT); auth = load(AUTH)
    context = load(OUTPUT+'/execution_context.json')
    marker = load(AUDIT+'/exclusive_launch_marker.json'); launch = load(AUDIT+'/launch_completion.json')
    preflight = load(AUDIT+'/authorized_metadata_preflight.json')
    check('fixed_raw_result_hash', sha(raw) == RESULT_SHA)
    check('raw_result_checksum_file', (ROOT/OUTPUT/'result.sha256').read_text() == RESULT_SHA+'\n')
    check('schema', r['schema'] == 'm1s_gate0_independent_result_v1')
    check('adopted_contract_config_unchanged', r['config'] == config == context['config'])
    check('contract_C_bytes', (ROOT/CONTRACT).read_bytes() == git('show', C+':'+CONTRACT))
    check('dedicated_authority_A_bytes', (ROOT/AUTH).read_bytes() == git('show', A+':'+AUTH))
    check('authority_binds_contract', auth['contract_commit'] == C and auth['contract_sha256'] == sha((ROOT/CONTRACT).read_bytes()) and auth['explicit_user_scientific_execution_approval'] is True)
    check('exact_source_chain', auth['source_commit'] == config['source_commit'] and subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', config['source_commit'], C]).returncode == 0 and subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', C, A]).returncode == 0)
    check('audit_identity', r['audit'] == context['audit'] == preflight['audit'] and r['audit']['authorization_commit'] == A and r['audit']['contract_commit'] == C)
    check('audit_authorization_hash', r['audit']['authorization_sha256'] == sha((ROOT/AUTH).read_bytes()))
    check('one_child_invocation_no_retry', launch['scientific_runner_invocations'] == 1 and launch['child_returncode'] == 0 and launch['launch_error'] is None and launch['automatic_retry'] is False)
    check('launch_marker_consumed', marker['launch_consumed'] is True and marker['automatic_retry'] is False and marker['planned_scientific_runner_invocations'] == 1)
    check('contract_one_shot_consumed', context['attempt_consumed'] is True and context['automatic_retry'] is False and r['automatic_retry'] is False)
    check('preflight_metadata_only', preflight['guard_passed'] is True and preflight['reference_kernel_calls'] == preflight['NPZ_array_decodes'] == preflight['numerical_tests'] == preflight['scientific_runner_invocations_before_launch'] == 0)
    prefix = Path('/tmp')/('m1s_independent_once_'+A)
    for suffix, name in [('.launch.json', 'exclusive_launch_marker.json'), ('.preflight.json', 'authorized_metadata_preflight.json'), ('.completion.json', 'launch_completion.json'), ('.console.log', 'supervisor_console.log')]:
        check('external_raw_audit_copy:'+suffix, Path(str(prefix)+suffix).read_bytes() == (ROOT/AUDIT/name).read_bytes())
    check('available_scope_PASS', r['available_comparator_status'] == 'PASS')
    check('full_scope_UNDETERMINED_not_false_PASS', r['status'] == 'UNDETERMINED' and 'Missing frozen full/HD comparator' in r['full_scope_limitation'])
    check('no_failure_field', 'failure' not in r)
    check('all_three_split_rows', [x['split'] for x in r['rows']] == ['G3', 'G6', 'G9'])
    progress = r['progress']
    check('all_independent_references_completed', progress['completed_reference_splits'] == progress['completed_exact_splits'] == ['G3', 'G6', 'G9'] and progress['completed_exact_blocks'] == 12)
    check('frozen_reference_phase', progress['phase'] == 'INDEPENDENT_REFERENCE_FROZEN')
    snapshot = json.dumps(r['reference'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    check('canonical_reference_snapshot_hash', sha(snapshot) == r['independent_reference_sha256'] == progress['independent_reference_sha256'])
    check('partial_reference_matches_frozen_reference', progress['partial_reference'] == r['reference'])
    check('comparison_ledger_matches_rows', progress['comparisons'] == r['rows'])
    supports = {'full': r['reference']['full_support']}
    for row in r['rows']:
        s = row['split']; law = row['law_comparison']; metrics = row['metric_comparison']; ref = r['reference']['splits'][s]
        check(s+':available_PASS_full_UNDETERMINED', row['available_comparator_status'] == 'PASS' and row['full_requested_scope_status'] == 'UNDETERMINED')
        check(s+':law_and_metric_PASS', law['status'] == metrics['status'] == ref['metrics']['status'] == 'PASS')
        check(s+':missing_full_HD_preserved', law['saved_full_HD_nonidentity_match'] == 'UNDETERMINED_NOT_SAVED_BY_V5')
        check(s+':support_sign_no_discrepancy', law['support_missing_count'] == law['support_spurious_count'] == law['sign_flip_count'] == 0 and law['support_or_sign_witnesses'] == {'flipped': [], 'missing': [], 'spurious': []})
        check(s+':coefficient_saved_error_within_saved_allowance', within_saved(law['coefficient_L1_error_Ha'], law['coefficient_L1_allowance_Ha']))
        check(s+':identity_saved_error_within_saved_allowance', within_saved(law['identity_sum_error_Ha'], law['identity_sum_allowance_Ha']))
        check(s+':bounded_witnesses_no_coefficient_cutoff', len(law['coefficient_difference_witnesses']) <= 64)
        for key, power in [('W', 24), ('v', 24), ('rho', 12)]:
            stored = metrics['checks'][key]
            check(s+':'+key+':fixed_tolerance_PASS', stored['status'] == 'PASS' and stored['tolerance'] == '2^-'+str(power) and stored['units'] == 'dimensionless')
            check(s+':'+key+':stored_exact_error_within_fixed_ceiling', within_scalar(stored['absolute_difference'], 1, 1 << power))
        check(s+':absolute_oracle_quality_reported', ref['metrics']['exact_oracle_absolute_quality'] is True)
        check(s+':saved_residual_squared_bound', within_scalar(ref['metrics']['full_H_residual_squared_Ha2'], 1, 10**16))
        check(s+':reference_rounding_allowance_zero', ratio(ref['metrics']['reference_error_allowance']) == (0, 1))
        supports[s+':HR'] = ref['HR_support']; supports[s+':HD'] = ref['HD_support']
    for name, support in supports.items():
        bitmap = bytes.fromhex(support['zero_bitmap_hex'])
        check('complete_support_bitmap:'+name, len(bitmap) == 8192 and support['cells_examined'] == 65536 and sha(bitmap) == support['zero_bitmap_sha256'] and 65536-sum(x.bit_count() for x in bitmap) == support['exact_nonzero'])
    limits = config['resources']; use = r['resources']
    check('resource_caps_retained', use['workers'] == 1 and use['address_space_limit_bytes'] == limits['memory_bytes'] == 2147483648 and use['result_and_log_per_file_limit_bytes'] == limits['output_bytes'] == limits['log_bytes'] == 134217728 and use['wall_time_cap'] is None)
    check('recorded_resident_memory_within_cap', use['maxrss_linux_KiB']*1024 <= limits['memory_bytes'])
    check('primitive_events_within_cap', type(r['primitive_events']) is int and 0 < r['primitive_events'] <= limits['max_primitive_events'])
    check('valid_recorded_elapsed_time', isinstance(use['elapsed_seconds'], float) and math.isfinite(use['elapsed_seconds']) and use['elapsed_seconds'] >= 0)
    for name, cap in [('result.json', limits['output_bytes']), ('runner.log', limits['log_bytes']), ('execution_context.json', limits['context_bytes'])]:
        check('output_cap:'+name, (ROOT/OUTPUT/name).stat().st_size <= cap)
    for p, digest in config['source_sha256'].items():
        b = (ROOT/p).read_bytes()
        check('frozen_SOURCE_preserved:'+p, b == git('show', config['source_commit']+':'+p) and sha(b) == digest)
    b = (ROOT/config['comparator']['path']).read_bytes()
    check('original_v5_result_preserved', sha(b) == config['comparator']['sha256'])
    check('no_scientific_GO_or_platform_certificate', r['scientific_GO_or_no_GO_determined'] is False and r['v5_platform_coverage_certified'] is False and r['posthoc_independent_reproducibility_check'] is True)
    check('mandatory_STOP', r['mandatory_STOP'] is True and marker['mandatory_STOP'] is True and launch['mandatory_STOP'] is True)
    report = {'schema': 'm1s_independent_saved_only_audit_v1', 'PASS': len(checks), 'FAIL': 0, 'checks': checks,
              'raw_result_sha256': RESULT_SHA, 'available_comparator_status': r['available_comparator_status'], 'full_scope_status': r['status'],
              'NPZ_opens': 0, 'reference_kernel_calls': 0, 'new_Pauli_generation': 0, 'new_W_v_rho_calculations': 0,
              'new_allowance_formula_evaluations': 0, 'saved_integer_ratio_bound_checks_only': True,
              'additional_scientific_runs': 0, 'automatic_retry': False, 'mandatory_STOP': True}
    if args.write_report:
        (ROOT/args.write_report).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
