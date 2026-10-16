#!/usr/bin/env python3
"""Audit saved Gate0 v4 identities/status only; never import or execute science.

No NPZ decoding, Pauli regeneration, budget evaluation or metric recomputation.
The optional report is restricted to the new execution audit namespace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
A = 'b70abf111e6459c45c24331a2c7d99c05e7dc228'
CC = '5e57955baa79e42f37aa233ac011c60539e431fd'
SC = '606036db52da2f080cf846e48824e5e418430762'
RESULT = 'artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot'
AUDIT = 'artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot'
AUTH = 'artifacts/track_b_m1s_gate0_execution_authorization/2026-10-16/v4/execution_authorization.json'
CONTRACT = 'artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/execution_contract_v4_adopted.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    checks = []
    def check(name, condition):
        checks.append({'name': name, 'status': 'PASS' if condition else 'FAIL'})
    def raw(path):
        return (ROOT / path).read_bytes()
    def obj(path):
        return json.loads(raw(path))
    def sha(data):
        return hashlib.sha256(data).hexdigest()
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(ROOT), *argv])
    result_raw = raw(RESULT + '/result.json')
    r = json.loads(result_raw)
    config = obj(CONTRACT)
    auth = obj(AUTH)
    context = obj(RESULT + '/execution_context.json')
    launch = obj(AUDIT + '/exclusive_launch_marker.json')
    completion = obj(AUDIT + '/launch_completion.json')
    check('result_raw_checksum', sha(result_raw) == raw(RESULT + '/result.sha256').decode().strip())
    check('result_schema', r['schema'] == 'm1s_gate0_result_v4')
    check('saved_context_equals_result_config_and_audit', context == {'config': r['config'], 'audit': r['audit']})
    check('saved_contract_equals_formal_contract', r['config'] == config)
    check('formal_contract_frozen_bytes', raw(CONTRACT) == git('show', CC + ':' + CONTRACT))
    check('authorization_frozen_bytes', raw(AUTH) == git('show', A + ':' + AUTH))
    check('authorization_hash_binding', r['audit']['authorization_sha256'] == sha(raw(AUTH)))
    check('contract_hash_binding', auth['contract_sha256'] == r['audit']['contract_sha256'] == sha(raw(CONTRACT)))
    check('commit_binding', auth['source_commit'] == r['audit']['source_commit'] == SC and auth['contract_commit'] == r['audit']['contract_commit'] == CC and r['audit']['authorization_commit'] == A)
    check('adoption_and_separate_execution_distinct', obj('artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json')['scientific_execution_authorized'] is False and auth['explicit_user_scientific_execution_approval'] is True and A != CC)
    for path, expected in config['source_sha256'].items():
        check('frozen_source:' + path, raw(path) == git('show', SC + ':' + path) and sha(raw(path)) == expected)
    check('frozen_input_raw_hash_only', sha(raw(config['input']['path'])) == config['input']['sha256'])
    check('output_namespace_binding', auth['output_directory'] == r['audit']['output_directory'] == RESULT)
    check('exclusive_launch_identity', launch['authorization_commit'] == A and launch['authorization_sha256'] == sha(raw(AUTH)) and launch['contract_sha256'] == sha(raw(CONTRACT)))
    check('one_launch_no_retry', launch['maximum_runner_invocations'] == completion['runner_invocations'] == 1 and launch['automatic_retry'] is False and completion['automatic_retry'] is False)
    check('one_shot_consumed_on_failure', launch['consumed_on_launch_even_if_runner_guard_or_runtime_fails'] is True)
    check('observed_outer_exit_zero_not_science_pass', completion['exit_code'] == 0 and json.loads(completion['output'])['status'] == r['status'] == 'TECHNICAL_STOP')
    check('mandatory_stop', r['mandatory_STOP'] is True and r['automatic_scientific_GO_STOP'] is False)
    check('conditional_classification', r['evidence_classification'] == config['evidence_classification'] == 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
    check('coverage_adopted_not_certified', config['formula_policy']['coverage_model_adopted'] is True and r['state']['theorem_applicability'] == 'NOT_CERTIFIED')
    check('precision_items_17', len(set(config['precision']) - {'dtype'}) == 17)
    check('absolute_residual_policy_fixed', config['formula_policy']['oracle_absolute_residual_quality_hartree'] == 1e-8 and config['precision']['eps_state_residual']['formula_id'] == 'absolute_original_solver_acceptance_v1')
    check('all_reached_gate_records_pass', bool(r['gates']) and all(g['status'] == 'PASS' for g in r['gates']))
    names = {g['name'] for g in r['gates']}
    required = {'input_raw_identity', 'state_sector_embedding_exact', 'one_body_formula_input_hermiticity_exact', 'full_H_state_residual_2', 'full_H_hermiticity_F', 'identity_phase_ledger_F'}
    required.update('g_' + str(i) + '_formula_input_hermiticity_exact' for i in range(12))
    required.update('basis_ladder_exact_' + str(i) + '_' + str(b) for i in range(8) for b in (False, True))
    check('reached_input_reference_residual_gate_inventory', required <= names)
    check('post_precision_metric_gates_not_reached', not names & {'probability_normalization', 'Pauli_expectation_imaginary', 'Pauli_mean_consistency', 'W_internal_range', 'v_internal_range', 'rho_interval_domain_consistency'})
    check('all_three_splits_preserved_in_order', [x['split'] for x in r['rows']] == config['split_ids'] == ['G3', 'G6', 'G9'])
    for row in r['rows']:
        label = row['split'] + ':'
        ledger = row['error_ledger']['formula_resolution']
        check(label + 'unresolved_before_metrics', row['metric_status'] == 'PRECISION_UNRESOLVED_BEFORE_METRICS' and row['technical_stop'] is True and row['technical_stop_category'] == 'PRECISION_UNRESOLVED')
        check(label + 'metrics_null_preserved', all(row[k] is None for k in ('W', 'v', 'rho', 'rho_interval')))
        check(label + 'recorded_budgets_exceed_adopted_tolerances', ledger['resolved_law_limits']['W_error_budget'] > config['precision']['eps_W'] and ledger['resolved_law_limits']['v_error_budget'] > config['precision']['eps_v'])
        check(label + 'formula_ledger_not_recomputed', ledger['computed_before_W_v_rho'] is True and ledger['coverage'] == r['evidence_classification'] and row['error_ledger']['error_budgets_are_rigorous_bounds'] is False)
        check(label + 'saved_canonical_table_digest', sha(json.dumps(row['canonical_table'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) == row['canonical_table_sha256'])
        check(label + 'saved_formula_ledger_digest', sha(json.dumps(ledger, sort_keys=True, allow_nan=False).encode()) == row['error_ledger']['formula_resolution_sha256_before_metrics'])
        words = [x['word'] for x in row['canonical_table']]
        check(label + 'saved_table_word_structure', words == sorted(set(words)) and all(len(w) == 8 and set(w) <= set('IXYZ') and w != 'I' * 8 for w in words))
        check(label + 'no_new_scientific_decision', row['scientific_decision'] == 'REQUIRES_USER_GPT_JUDGMENT' and row['native_shots_resources'] == 'NOT_ESTABLISHED' and row['finite_step_dressing'] == 'NOT_EVALUATED')
    check('resource_profile_recorded', r['resources']['workers'] == 1 and r['resources']['backend'] == 'cpu' and r['resources']['python'] == config['runtime']['python'] and r['resources']['numpy'] == config['runtime']['numpy'])
    check('thread_environment_bound', launch['thread_environment'] == config['runtime']['threads'] and set(launch['thread_environment'].values()) == {'1'})
    check('recorded_process_caps', r['applied_process_limits']['address_space_bytes'] == config['limits']['memory_bytes'] == 4294967296 and r['applied_process_limits']['per_file_bytes_including_native_log'] == config['limits']['output_bytes'] == 268435456)
    check('no_wall_cap', config['limits']['wall_time_seconds'] is None and r['resources']['wall_time_cap'] is None and launch['scientific_wall_time_cap'] is None)
    check('recorded_contribution_cap', r['resources']['product_contributions'] <= config['limits']['max_product_contributions'])
    for name, cap in [('result.json', config['limits']['output_bytes']), ('runner.log', config['limits']['log_bytes']), ('execution_context.json', 4 * 1024 ** 2)]:
        check('saved_file_cap:' + name, len(raw(RESULT + '/' + name)) <= cap)
    check('scope_extension_unperformed', {'sampling', 'simulation', 'spectrum', 'gap', 'circuits', 'compile', 'resources', 'T1c', 'Q4', 'Q5'} <= set(r['unperformed']))
    report = {'schema': 'm1s_gate0_v4_saved_identity_status_audit_v1', 'scope': 'saved bytes, JSON structure, recorded statuses and preregistered threshold comparisons only', 'scientific_reexecution': False, 'NPZ_array_decodes': 0, 'budget_formula_evaluations': 0, 'Pauli_regenerations': 0, 'metric_recomputations': 0, 'result_sha256': sha(result_raw), 'PASS': sum(c['status'] == 'PASS' for c in checks), 'FAIL': sum(c['status'] == 'FAIL' for c in checks), 'checks': checks, 'mandatory_STOP': True}
    if args.write_report:
        target = ROOT / AUDIT / 'saved_identity_status_audit.json'
        target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
    return 1 if report['FAIL'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
