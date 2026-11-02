"""Saved evidence verification only; reuse controller hash verifier, no actions."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
NS = BASE / 'run01'
env = runpy.run_path(str(ROOT / 'scripts/run_h4_r1_order2.py'))['controller']()
A, C = env['A'], env['C']


def need(value, label):
    C.require(value, label)


def main():
    seal = C.read(BASE / 'seal.json')
    env['verify'](seal)
    need(seal['SOURCE'] == 'ca9a1c0e2426565ba63cc8bfd0247f73339d95b8', 'SCIENTIFIC_SOURCE')
    # The metadata join helper is read by AST from this tracked SOURCE too.
    join_helper = 'src/trotterlib/rte_event_audit.py'
    data = subprocess.check_output(['git', 'show', seal['SOURCE'] + ':' + join_helper], cwd=ROOT)
    need(C.hashlib.sha256(data).hexdigest() == C.sha(ROOT / join_helper), 'TRACKED_JOIN_HELPER_SOURCE')
    prior = C.read(BASE / 'prior_manifest_rows.json')
    manifest = C.read(ROOT / 'artifacts/validation_manifest.json')
    for k, h in prior['row_sha256'].items():
        need(C.digest(manifest[k]) == h, 'OLD_MANIFEST:' + k)
    contract, grant = C.read(BASE / 'active_contract.json'), C.read(NS / 'grant.json')
    need(grant['grant_id'] == C.digest({k: v for k, v in grant.items() if k != 'grant_id'}), 'GRANT_ID')
    need(grant['namespace'] == str(NS) and grant['one_shot'], 'ONE_SHOT_NAMESPACE')
    need(grant['contract_sha256'] == C.sha(BASE / 'active_contract.json'), 'CONTRACT')
    need(grant['seal_sha256'] == contract['seal_sha256'] == C.sha(BASE / 'seal.json'), 'SEAL')
    need(grant['approval_sha256'] == contract['approval_sha256'] == C.sha(BASE / 'approval.txt'), 'APPROVAL')
    need(C.read(NS / 'CONSUMED.json')['grant_id'] == grant['grant_id'], 'CONSUMED')
    need(contract['keys'] == ['PR1_cosine'] and contract['logical_diagnostics'] == 1 and contract['max_initial_state_actions'] == 3, 'FINITE_TARGET')
    need(contract['state_action_limits'] == A.ACTION_LIMITS and contract['limits'] == A.LIMITS, 'FINITE_LIMITS')
    ledger = C.read(NS / 'call_ledger.json')
    need(ledger['used'] == ledger['limits'] == A.LIMITS, 'ALL_CALL_COUNTS')
    need(dict(Counter(e['call'] for e in ledger['events'])) == A.LIMITS, 'CALL_EVENTS')
    need(all(e['reserved_before_call'] for e in ledger['events']), 'RESERVED')
    summary = C.read(NS / 'result_summary.json')
    need(summary['logical_diagnostics'] == 1 and summary['successful_state_actions'] == summary['total_state_action_attempts'] == 3, 'SUCCESS_COUNTS')
    need(summary['forbidden_calls'] == contract['forbidden_calls'] and all(v == 0 for v in summary['forbidden_calls'].values()), 'FORBIDDEN_ZERO')
    need(summary['historical_hash'] == 'FAIL_CAUSE_UNKNOWN_UNCHANGED' and summary['original_native'] == 'UNKNOWN_UNCHANGED', 'HISTORY_UNKNOWN')
    need(summary['original_N_G_null'] and summary['C_L'] == 'COMBINATION_BLOCKED' and summary['old_compiler'] == 'MISSING_EVIDENCE', 'OLD_GATES')
    need(summary['population_cost_tail'] == 'UNKNOWN' and summary['binary64_only'] and summary['mandatory_STOP'], 'EMPIRICAL_STOP')
    binding = C.read(NS / 'preflight.json')
    need(binding == C.read(BASE / 'structural_preflight.json'), 'SAME_PREFLIGHT')
    need(binding['status'] == 'PASS' and binding['QPY_decodes'] == 4 and binding['state_actions'] == 0, 'PREFLIGHT_SCOPE')
    need(binding['trajectory_id'] == A.TID and binding['trajectory_record_hash'] == A.THASH and binding['event_record_hash'] == A.EHASH, 'FIXED_RECORD')
    need(binding['aggregate_phase'] == A.SAVED_PHASE, 'FIXED_PHASE')
    state = C.read(NS / 'state_identity.json')
    old_state = C.read(A.R0BASE / 'run01/state_identity.json')
    need({k: state[k] for k in old_state} == old_state and not state['raw_state_changed'], 'REUSED_RAW_NORM')
    receipts = {}
    for name in A.ACTION_LIMITS:
        receipt, claim = C.read(NS / (name + '.json')), C.read(NS / (name + '_claim.json'))
        event = next(e for e in ledger['events'] if e['call'] == name)
        need(event['UTC'] <= claim['UTC'] <= receipt['UTC'], 'TIMELINE')
        need(claim['reserved_before_call'] and claim['call'] == receipt['call'] == name, 'CLAIM')
        need(receipt['binary64'] and receipt['numerical_error_upper'] is None and receipt['status'] == 'EMPIRICAL_ONLY_NO_RHO_UPPER', 'NO_NUMERICAL_BOUND')
        need(len(receipt['output_state_pairs']) == (256 if name == 'PR1_C_target' else 512), 'VECTOR_SHAPE')
        if name != 'PR1_C_target':
            need(not receipt['state_renormalized'] and not receipt['output_norm_division'] and not receipt['probability_API_used'], 'RAW_UNCLIPPED')
            need(receipt['raw_norm'] == state['nu_binary64'], 'SAME_NORM')
            need(receipt['signal'] == (receipt['p0_raw'] - receipt['p1_raw']) / receipt['raw_norm'], 'SAVED_OBSERVABLE_IDENTITY')
        receipts[name] = receipt
    ideal = receipts['PR1_C_target']
    need(not ideal['PR_law_mean_used'] and not ideal['normalization_B_applied_to_target'] and ideal['selected_trajectory_id'] == A.TID, 'SELECTED_C_TARGET')
    result = C.read(NS / 'PR1_result.json')
    need(C.read(NS / 'results.json') == [result], 'RESULT_ARRAY')
    mi, mo, mc = [receipts[k]['signal'] for k in ('PR1_input', 'PR1_output', 'PR1_C_target')]
    need([result[k] for k in ('m_input', 'm_output', 'm_C')] == [mi, mo, mc], 'SAVED_SIGNALS')
    need(F(result['m_C_representative_exact']) == F(mc), 'EXACT_C_POINT')
    target = C.read(BASE / 'inputs.json')['target']
    guard = next(g for g in C.read(A.P.BASE / 'guard_inputs_quoted.json')['axes'] if (g['case'], g['axis'], g['epsilon_sig']) == ('pauli_jw_PR', 'cosine', '0.0005'))
    b, g = F(result['B_saved_exact']), F(result['rho_budget_exact'])
    need(b == F(target['B_saved']) == F(guard['scale']['B_saved_exact']), 'B_SAVED_ONCE')
    need(g == F(guard['rho_guard_exact']), 'QUOTED_BUDGET')
    delta, eta, direct = [F(result[k + '_exact']) for k in ('delta', 'eta', 'd')]
    need(delta == b * abs(F(mo) - F(mi)) and eta == b * abs(F(mi) - F(mc)) and direct == b * abs(F(mo) - F(mc)), 'SCALAR_RESIDUAL_IDENTITIES')
    need(direct <= delta + eta and result['triangle_verified'], 'SCALAR_TRIANGLE')
    need(F(result['d_over_budget_exact']) == direct / g and F(result['budget_minus_d_exact']) == g - direct, 'BUDGET_IDENTITIES')
    for k in ['delta', 'eta', 'd', 'B_saved', 'rho_budget', 'd_over_budget', 'budget_minus_d']:
        need(result[k] == float(F(result[k + '_exact'])), 'DISPLAY:' + k)
    need(direct < g and result['empirical_relation'] == 'BELOW_SAVED_BUDGET', 'EMPIRICAL_RELATION')
    for record in [result, summary]:
        need(all(record[k] is None for k in ['rho_upper', 'numerical_error_upper', 'PR_law_rho']), 'NULL_UPPERS')
        need(not record['LOCAL_BUDGET_MET'] and not record['formal_certificate_accepted'], 'NO_CERTIFICATE')
    comparison = C.read(NS / 'R0_R1_comparison.json')
    need(comparison['rows'] == [*C.read(A.R0BASE / 'run01/results.json'), result], 'R0_QUOTE_ONLY')
    need(not comparison['causal_order2_effect_established'] and not comparison['law_wide_inference'], 'LIMITED_COMPARISON')
    stop, native = C.read(NS / 'STOP.json'), C.read(NS / 'native/worker_native_exit.json')
    need(stop['status'] == 'COMPLETE_MANDATORY_STOP' and stop['validation_error'] is None, 'STOP')
    need(stop['native_exit'] == native and native['exit_code'] == 0 and native['reaped'], 'NATIVE_EXIT')
    need(native['scope'] == 'NEW_R1_ORDER2_BINARY64_EMPIRICAL_ONLY' and stop['all_owned_reaped'], 'NEW_SCOPE_REAP')
    need(not stop['original_H4_native_verified'] and not stop['automatic_next_stage'], 'OLD_NATIVE_NOT_REPLACED')
    need(C.read(NS / 'native/STOP.json')['reaped_pids'] == [native['pid']], 'OWN_WORKER')
    need(C.read(NS / 'WORKER_CLAIM.json')['pid'] == C.read(NS / 'native/worker_identity.json')['pid'] == native['pid'], 'IDENTITY')
    need((NS / 'native/stderr.log').stat().st_size == 0, 'STDERR_EMPTY')
    inventory = {str(p.relative_to(ROOT)): C.sha(p) for p in sorted(NS.rglob('*')) if p.is_file()}
    C.save(BASE / 'output_inventory.json', inventory)
    verification = dict(status='PASS_SAVED_EVIDENCE_CONSISTENCY_ONLY', UTC=C.utc(), SOURCE=seal['SOURCE'],
        code_files=len(seal['code_files']), runtime=len(C.read(BASE / 'runtime.json')['files']),
        inputs=len(C.read(BASE / 'inputs.json')['files']), preserved=len(C.read(BASE / 'preservation_before.json')),
        old_manifest_rows_unchanged=prior['row_count'], output_files=len(inventory), output_inventory_sha256=C.sha(BASE / 'output_inventory.json'),
        tracked_join_helper_current_SOURCE_match=True, logical_diagnostics=1, successful_state_actions=3,
        total_state_action_attempts=3, native_exit=native, grant_id=grant['grant_id'],
        new_state_actions_in_this_verifier=0, QPY_decodes_in_this_verifier=0,
        scalar_checks='identities among already saved values only; no vector reductions or new signal analysis',
        author_checks_only=True, rho_upper=None, numerical_error_upper=None, PR_law_rho=None,
        formal_certificate_accepted=False, mandatory_STOP=True)
    C.save(BASE / 'verification.json', verification)
    print(env['json'].dumps(verification))


if __name__ == '__main__':
    main()
