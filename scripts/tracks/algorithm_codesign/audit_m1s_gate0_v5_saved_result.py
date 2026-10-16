#!/usr/bin/env python3
"""Saved-byte/status audit only: no numerical imports, regeneration or formulas."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
A = '8da63d672f779c947dc295eb2c5d524d927ac66b'
CC = '50b3669b9253eb889fa4daa87e793646df2d5138'
SC = '69de022dc8142f703423b1ce0bd42f05e123954e'
RESULT = 'artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot'
AUDIT = 'artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1'
AUTH = 'artifacts/track_b_m1s_gate0_v5_execution_authorization/2026-10-16/v1/execution_authorization.json'
CONTRACT = 'artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/execution_contract_v5_adopted.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'status': 'PASS' if ok else 'FAIL'})
    def raw(path):
        return (ROOT / path).read_bytes()
    def obj(path):
        return json.loads(raw(path))
    def sha(data):
        return hashlib.sha256(data).hexdigest()
    def git(*a):
        return subprocess.check_output(['git', '-C', str(ROOT), *a])
    rraw = raw(RESULT + '/result.json')
    r = json.loads(rraw)
    c = obj(CONTRACT)
    auth = obj(AUTH)
    context = obj(RESULT + '/execution_context.json')
    launch = obj(AUDIT + '/exclusive_launch_marker.json')
    completion = obj(AUDIT + '/launch_completion.json')
    guard = obj(AUDIT + '/prelaunch_guard.json')
    old = obj('artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json')
    check('raw_result_checksum', sha(rraw) == raw(RESULT + '/result.sha256').decode().strip())
    check('v5_result_schema', r['schema'] == 'm1s_gate0_result_v5')
    check('saved_context_identical', context == {'audit': r['audit'], 'config': r['config']})
    check('frozen_contract_bytes', raw(CONTRACT) == git('show', CC + ':' + CONTRACT) and c == r['config'])
    check('frozen_authority_bytes', raw(AUTH) == git('show', A + ':' + AUTH))
    check('contract_and_authority_hash_binding', r['audit']['authorization_sha256'] == sha(raw(AUTH)) and r['audit']['contract_sha256'] == auth['contract_sha256'] == sha(raw(CONTRACT)))
    check('commits_bound', r['audit']['authorization_commit'] == A and r['audit']['contract_commit'] == auth['contract_commit'] == CC and r['audit']['source_commit'] == auth['source_commit'] == SC)
    check('adoption_and_execution_approval_separate', obj('artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json')['scientific_execution_authorized'] is False and auth['explicit_user_scientific_execution_approval'] is True)
    for path, expected in c['source_sha256'].items():
        check('frozen_SOURCE:' + path, raw(path) == git('show', SC + ':' + path) and sha(raw(path)) == expected)
    for path, expected in c['tail_bound_provenance']['files_sha256'].items():
        check('frozen_proof:' + path, raw(path) == git('show', c['tail_bound_provenance']['proof_commit'] + ':' + path) and sha(raw(path)) == expected)
    for a, b in ((c['tail_bound_provenance']['proof_commit'], SC), (SC, CC), (CC, A)):
        check('commit_order:' + a + '<' + b, a != b and subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', a, b]).returncode == 0)
    check('frozen_input_raw_bytes_only', len(raw(c['input']['path'])) == 8652 and sha(raw(c['input']['path'])) == c['input']['sha256'])
    check('new_v5_namespace_binding', auth['output_directory'] == r['audit']['output_directory'] == RESULT)
    check('exclusive_launch_identity', launch['authorization_commit'] == A and launch['authorization_sha256'] == sha(raw(AUTH)) and launch['contract_sha256'] == sha(raw(CONTRACT)))
    check('marker_original_unchanged', Path(launch['marker_path']).read_bytes() == raw(AUDIT + '/exclusive_launch_marker.json'))
    check('one_launch_no_retry', launch['maximum_runner_invocations'] == completion['runner_invocations'] == 1 and launch['automatic_retry'] is False and completion['automatic_retry'] is False and launch['consumed_on_launch_even_if_runner_guard_or_runtime_fails'] is True)
    check('outer_and_saved_status_consistent', completion['exit_code'] == 0 and json.loads(completion['output'])['status'] == r['status'])
    check('complete_diagnostics_status', r['status'] == 'HEADROOM_DIAGNOSTICS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT' and not r.get('failure') and r.get('technical_categories') == [])
    check('mandatory_STOP_no_scientific_GO', r['mandatory_STOP'] is True and r['automatic_scientific_GO_STOP'] is False)
    check('conditional_classification', r['evidence_classification'] == c['evidence_classification'] == 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
    check('coverage_adopted_not_certified', c['formula_policy']['coverage_model_adopted'] is True and r['state']['theorem_applicability'] == 'NOT_CERTIFIED' and r['state']['residual_quality_gate']['eigenstate_distance_certified'] is False)
    check('all_reached_gates_PASS', bool(r['gates']) and all(g['status'] == 'PASS' for g in r['gates']))
    names = {g['name'] for g in r['gates']}
    check('initial_correctness_gates_reached', {'input_raw_identity', 'state_sector_embedding_exact', 'full_H_state_residual_2', 'full_H_hermiticity_F'} <= names and all('basis_ladder_exact_' + str(i) + '_' + str(b) in names for i in range(8) for b in (False, True)))
    check('17_precision_conditions', len(set(c['precision']) - {'dtype'}) == 17)
    check('all_three_splits_in_order', [x['split'] for x in r['rows']] == c['split_ids'] == ['G3', 'G6', 'G9'])
    check('global_ledger_unchanged_from_v4', r['global_formula_resolution_before_metrics'] == old['formula_resolution_before_metrics'])
    for row, older, gamma_index in zip(r['rows'], old['rows'], (722, 719, 716)):
        label = row['split'] + ':'
        ledger = row['error_ledger']['formula_resolution']
        local = row['error_ledger']['tail_local_coefficient_error']
        recorded = ledger['resolved_law_limits']
        split_names = {g['name'] for g in r['gates'] if g.get('split') == row['split']}
        check(label + 'metric_correctness_gates_reached', {'probability_normalization', 'Pauli_expectation_imaginary', 'Pauli_mean_consistency', 'W_internal_range', 'v_internal_range', 'W_v_interval_consistency', 'rho_interval_domain_consistency'} <= split_names)
        check(label + 'diagnostic_complete_not_scientific_decision', row['metric_status'] == 'HEADROOM_DIAGNOSTIC_ONLY' and row['technical_stop'] is False and row['scientific_decision'] == 'REQUIRES_USER_GPT_JUDGMENT')
        check(label + 'saved_metrics_present', all(type(row[k]) is float for k in ('W', 'v', 'rho')))
        check(label + 'recorded_budgets_within_adopted_ceilings', recorded['W_error_budget'] <= c['precision']['eps_W'] and recorded['v_error_budget'] <= c['precision']['eps_v'] and row['rho_error_budget_uncertainty'] <= c['precision']['eps_rho'])
        check(label + 'local_global_ledger_separation', ledger['global_coefficient_error_used_for_law'] is False and ledger['computed_before_W_v_rho'] is True and local == ledger['tail_local_coefficient_error'] == r['tail_coefficient_resolution_before_metrics'][row['split']])
        check(label + 'preregistered_tail_model', local['gamma_index'] == gamma_index and local['computed_before_coefficient_generation'] is True and local['resource_cap_or_observed_counter_used'] is False and local['coverage'] == r['evidence_classification'])
        check(label + 'not_rigorous_platform_certificate', row['error_ledger']['error_budgets_are_rigorous_bounds'] is False)
        check(label + 'canonical_table_digest', sha(json.dumps(row['canonical_table'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) == row['canonical_table_sha256'])
        check(label + 'same_canonical_coefficients_as_v4', row['canonical_table_sha256'] == older['canonical_table_sha256'])
        check(label + 'pre_metrics_ledger_digest', sha(json.dumps(ledger, sort_keys=True, allow_nan=False).encode()) == row['error_ledger']['formula_resolution_sha256_before_metrics'])
        words = [x['word'] for x in row['canonical_table']]
        check(label + 'canonical_word_structure', words == sorted(set(words)) and all(len(w) == 8 and set(w) <= set('IXYZ') and w != 'I' * 8 for w in words))
        check(label + 'no_finite_step_or_native_resource_result', row['finite_step_dressing'] == 'NOT_EVALUATED' and row['native_shots_resources'] == 'NOT_ESTABLISHED')
    check('runtime_record_matches_contract', r['resources']['python'] == c['runtime']['python'] and r['resources']['numpy'] == c['runtime']['numpy'] and r['resources']['workers'] == 1 and r['resources']['backend'] == 'cpu')
    check('thread_environment_recorded', launch['thread_environment'] == c['runtime']['threads'] and set(launch['thread_environment'].values()) == {'1'})
    check('recorded_process_caps', r['applied_process_limits']['address_space_bytes'] == c['limits']['memory_bytes'] and r['applied_process_limits']['per_file_bytes_including_native_log'] == c['limits']['output_bytes'])
    check('no_scientific_wall_cap', c['limits']['wall_time_seconds'] is None and r['resources']['wall_time_cap'] is None and launch['scientific_wall_time_cap'] is None)
    check('recorded_contribution_cap', r['resources']['product_contributions'] <= c['limits']['max_product_contributions'])
    for name, cap in [('result.json', c['limits']['output_bytes']), ('runner.log', c['limits']['log_bytes']), ('execution_context.json', 4 * 1024 ** 2)]:
        check('file_cap:' + name, len(raw(RESULT + '/' + name)) <= cap)
    check('prelaunch_non_numerical_guard_recorded', guard['full_frozen_authority_SOURCE_contract_proof_input_runtime_guard'] == 'PASS' and guard['scientific_runner_invocations_before_launch'] == 0 and guard['numpy_imported'] is False)
    check('v4_STOP_null_retained', old['status'] == 'TECHNICAL_STOP' and all(all(x[k] is None for k in ('W', 'v', 'rho')) for x in old['rows']) and raw('artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json') == git('show', A + ':artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json'))
    check('v4_consumed_marker_retained', Path('/tmp/m1s_gate0_v4_once_b70abf111e6459c45c24331a2c7d99c05e7dc228.launch.json').read_bytes() == raw('artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/exclusive_launch_marker.json'))
    check('scope_extension_unperformed', {'sampling', 'simulation', 'spectrum', 'gap', 'circuits', 'compile', 'resources', 'T1c', 'Q4', 'Q5'} <= set(r['unperformed']))
    report = {'schema': 'm1s_gate0_v5_saved_identity_status_audit_v1', 'scope': 'saved bytes, hashes, statuses and recorded preregistered ceiling comparisons only', 'scientific_reexecution': False, 'NPZ_array_decodes': 0, 'budget_formula_evaluations': 0, 'Pauli_regenerations': 0, 'metric_recomputations': 0, 'result_sha256': sha(rraw), 'PASS': sum(x['status'] == 'PASS' for x in checks), 'FAIL': sum(x['status'] == 'FAIL' for x in checks), 'checks': checks, 'mandatory_STOP': True}
    if args.write_report:
        (ROOT / AUDIT / 'saved_identity_status_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))
    return 1 if report['FAIL'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
