"""Read-only state/SOURCE/receipt/scalar verification; no new signal evaluation."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import runpy

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
NS = BASE / 'run01'
env = runpy.run_path(str(ROOT / 'scripts/run_h4_r1_sine.py'))['controller']()
A, C = env['A'], env['C']
need = C.require


def main():
    seal = C.read(BASE / 'seal.json')
    env['verify'](seal)
    need(seal['SOURCE'] == '198d4916f7f62b775d9f60020c42ffb07d4c1c89', 'SCIENTIFIC_SOURCE')
    prior = C.read(BASE / 'prior_manifest_rows.json')
    manifest = C.read(ROOT / 'artifacts/validation_manifest.json')
    for k, h in prior['row_sha256'].items(): need(C.digest(manifest[k]) == h, 'OLD_MANIFEST:' + k)
    contract, grant = C.read(BASE / 'active_contract.json'), C.read(NS / 'grant.json')
    need(grant['grant_id'] == C.digest({k: v for k, v in grant.items() if k != 'grant_id'}), 'GRANT_ID')
    need(grant['namespace'] == str(NS) and grant['one_shot'], 'ONE_SHOT_NAMESPACE')
    need(grant['contract_sha256'] == C.sha(BASE / 'active_contract.json'), 'CONTRACT')
    need(grant['seal_sha256'] == contract['seal_sha256'] == C.sha(BASE / 'seal.json'), 'SEAL')
    need(grant['approval_sha256'] == contract['approval_sha256'] == C.sha(BASE / 'approval.txt'), 'APPROVAL')
    need(C.read(NS / 'CONSUMED.json')['grant_id'] == grant['grant_id'], 'CONSUMED')
    need(contract['axis'] == 'sine' and contract['keys'] == list(A.KEYS) and contract['logical_diagnostics'] == 3 and contract['max_initial_state_actions'] == 6, 'FINITE_SINE_SCOPE')
    ledger = C.read(NS / 'call_ledger.json')
    need(ledger['used'] == ledger['limits'] == contract['limits'] == A.LIMITS, 'FINITE_COUNTS')
    need(contract['state_action_limits'] == A.ACTION_LIMITS, 'SIX_ACTIONS_ONLY')
    need(dict(Counter(e['call'] for e in ledger['events'])) == A.LIMITS and all(e['reserved_before_call'] for e in ledger['events']), 'RESERVATIONS')
    summary = C.read(NS / 'result_summary.json')
    need(summary['logical_diagnostics'] == 3 and summary['successful_state_actions'] == summary['total_state_action_attempts'] == 6, 'SUCCESS_COUNTS')
    need(summary['forbidden_calls'] == contract['forbidden_calls'] and all(v == 0 for v in summary['forbidden_calls'].values()), 'FORBIDDEN_ZERO')
    need(summary['cosine_reevaluation'] == summary['new_C_target_state_actions'] == 0, 'CACHE_ONLY')
    need(summary['historical_hash'] == 'FAIL_CAUSE_UNKNOWN_UNCHANGED' and summary['original_native'] == 'UNKNOWN_UNCHANGED', 'OLD_HASH_NATIVE')
    need(summary['original_N_G_null'] and summary['C_L'] == 'COMBINATION_BLOCKED' and summary['old_compiler'] == 'MISSING_EVIDENCE', 'OLD_GATES')
    need(summary['population_cost_tail'] == 'UNKNOWN' and summary['binary64_only'] and summary['mandatory_STOP'], 'EMPIRICAL_STOP')
    preflight = C.read(NS / 'preflight.json')
    need(preflight == C.read(BASE / 'structural_preflight.json') and preflight['status'] == 'PASS' and preflight['QPY_decodes'] == 15 and preflight['state_actions'] == 0, 'PREFLIGHT')
    state, old_state = C.read(NS / 'state_identity.json'), C.read(A.OLD0 / 'run01/state_identity.json')
    need({k: state[k] for k in old_state} == old_state and not state['raw_state_changed'], 'UNCHANGED_RAW_NORM')
    nu = F(state['nu_exact'])
    receipts = {}
    for name in A.ACTION_LIMITS:
        receipt, claim = C.read(NS / (name + '.json')), C.read(NS / (name + '_claim.json'))
        event = next(e for e in ledger['events'] if e['call'] == name)
        need(event['UTC'] <= claim['UTC'] <= receipt['UTC'] and claim['reserved_before_call'], 'CLAIM_ORDER')
        need(claim['call'] == receipt['call'] == name and len(receipt['output_state_pairs']) == 512, 'RECEIPT_IDENTITY')
        need(receipt['binary64'] and receipt['numerical_error_upper'] is None and receipt['status'] == 'EMPIRICAL_ONLY_NO_RHO_UPPER', 'NUMERIC_POLICY')
        need(not receipt['state_renormalized'] and not receipt['output_norm_division'] and not receipt['probability_API_used'], 'UNCLIPPED_RAW_RAY')
        need(receipt['raw_norm'] == state['nu_binary64'], 'COMMON_NU')
        need(receipt['signal'] == (receipt['p0_raw'] - receipt['p1_raw']) / receipt['raw_norm'], 'SAVED_SCALAR_OBSERVABLE')
        receipts[name] = receipt
    targets = C.read(BASE / 'inputs.json')['targets']
    guards = C.read(A.P.BASE / 'guard_inputs_quoted.json')['axes']
    cresult = C.read(ROOT / A.P.CRESULT)
    results = C.read(NS / 'results.json')
    need([r['key'] for r in results] == list(A.CASES), 'THREE_RESULTS')
    for row, target in zip(results, targets):
        key = row['key']
        need(row == C.read(NS / (key + '_result.json')) and row['axis'] == 'sine' and row['trajectory_id'] == target['trajectory_id'], 'RESULT_IDENTITY')
        cache = C.read(NS / (key + '_C_target_reused.json'))
        need(cache['new_C_target_state_actions'] == 0 and not cache['compiled_signal_error_enclosed'], 'NO_NEW_TARGET_BOUND')
        if key == 'PF':
            original = cresult['implemented_law']['PF']['corrected_encoded_mean']['imag']
            need(cache['saved_raw_interval'] == original, 'PF_SAVED_IMAG_INTERVAL')
            lo, hi = map(F, cache['normalized_interval_exact'])
            need(lo == F(original['lower']) / nu and hi == F(original['upper']) / nu, 'PF_SAVED_INTERVAL_CONVERSION')
            need(F(cresult['state_norm']['lower']) <= nu <= F(cresult['state_norm']['upper']), 'NORM_IN_SAVED_C_INTERVAL')
            mc = (lo + hi) / 2
            need(lo <= F(cache['representative_exact']) == mc <= hi, 'PF_MIDPOINT')
        else:
            old = A.OLD0 if key == 'PR0' else A.OLD1
            saved = C.read(old / ('run01/' + key + '_C_target.json'))
            need(not saved['PR_law_mean_used'] and saved['selected_trajectory_id'] == target['trajectory_id'], 'INDIVIDUAL_COMPLEX_CACHE')
            mc = F(saved['complex_signal']['imag'])
            need(F(cache['representative_exact']) == mc, 'EXACT_CACHED_IMAG')
        mi, mo = receipts[key + '_input']['signal'], receipts[key + '_output']['signal']
        need(row['m_C'] == float(mc) and F(row['m_C_representative_exact']) == mc and row['m_input'] == mi and row['m_output'] == mo, 'SAVED_SIGNALS')
        gr = next(g for g in guards if (g['case'], g['epsilon_sig'], g['axis']) == ('pauli_jw_' + row['method'], '0.0005', 'sine'))
        b, g = F(row['B_saved_exact']), F(row['rho_budget_exact'])
        need(b == F(target['B_saved']) == F(gr['scale']['B_saved_exact']) and g == F(gr['rho_guard_exact']), 'QUOTED_SINE_BUDGET_B_ONCE')
        delta, eta, direct = [F(row[k + '_exact']) for k in ('delta', 'eta', 'd')]
        need(delta == b * abs(F(mo) - F(mi)) and eta == b * abs(F(mi) - mc) and direct == b * abs(F(mo) - mc), 'SAVED_RESIDUAL_IDENTITIES')
        need(direct <= delta + eta and row['triangle_verified'], 'SCALAR_TRIANGLE')
        need(F(row['d_over_budget_exact']) == direct / g and F(row['budget_minus_d_exact']) == g - direct, 'SAVED_BUDGET_IDENTITIES')
        for k in ['delta', 'eta', 'd', 'B_saved', 'rho_budget', 'd_over_budget', 'budget_minus_d']:
            need(row[k] == float(F(row[k + '_exact'])), 'DISPLAY:' + k)
        relation = 'BELOW_SAVED_BUDGET' if direct < g else 'EQUAL_SAVED_BUDGET' if direct == g else 'ABOVE_SAVED_BUDGET'
        need(row['empirical_relation'] == relation, 'EMPIRICAL_RELATION')
    for record in [summary, *results]:
        need(all(record[k] is None for k in ['rho_upper', 'numerical_error_upper', 'PR_law_rho']) and not record['LOCAL_BUDGET_MET'] and not record['formal_certificate_accepted'], 'NO_GUARANTEE_PROMOTION')
    comparison = C.read(NS / 'axes_comparison.json')
    need(comparison['sine'] == results and comparison['cosine'] == [*C.read(A.OLD0 / 'run01/results.json'), *C.read(A.OLD1 / 'run01/results.json')] and not comparison['law_wide_inference'], 'OLD_COSINE_QUOTES_ONLY')
    stop, native = C.read(NS / 'STOP.json'), C.read(NS / 'native/worker_native_exit.json')
    need(stop['status'] == 'COMPLETE_MANDATORY_STOP' and stop['validation_error'] is None and stop['native_exit'] == native, 'STOP')
    need(native['exit_code'] == 0 and native['reaped'] and native['scope'] == 'NEW_R1_SINE_BINARY64_EMPIRICAL_ONLY' and stop['all_owned_reaped'], 'NEW_NATIVE_EXIT_REAP')
    need(not stop['original_H4_native_verified'] and not stop['automatic_next_stage'], 'OLD_NATIVE_UNCHANGED')
    need(C.read(NS / 'native/STOP.json')['reaped_pids'] == [native['pid']] and C.read(NS / 'WORKER_CLAIM.json')['pid'] == native['pid'], 'OWN_WORKER')
    need((NS / 'native/stderr.log').stat().st_size == 0, 'STDERR_EMPTY')
    inventory = {str(p.relative_to(ROOT)): C.sha(p) for p in sorted(NS.rglob('*')) if p.is_file()}
    C.save(BASE / 'output_inventory.json', inventory)
    result = dict(status='PASS_SAVED_EVIDENCE_CONSISTENCY_ONLY', UTC=C.utc(), SOURCE=seal['SOURCE'],
        code_files=len(seal['code_files']), runtime=len(C.read(BASE / 'runtime.json')['files']), inputs=len(C.read(BASE / 'inputs.json')['files']),
        preserved=len(C.read(BASE / 'preservation_before.json')), old_manifest_rows_unchanged=prior['row_count'],
        output_files=len(inventory), output_inventory_sha256=C.sha(BASE / 'output_inventory.json'), logical_diagnostics=3,
        successful_state_actions=6, total_state_action_attempts=6, new_C_target_state_actions=0, cosine_reevaluation=0,
        grant_id=grant['grant_id'], native_exit=native, new_state_actions_in_this_verifier=0, QPY_decodes_in_this_verifier=0,
        scalar_checks='identities among saved values only; no vector reductions or new signal analysis',
        author_checks_only=True, rho_upper=None, numerical_error_upper=None, PR_law_rho=None, formal_certificate_accepted=False, mandatory_STOP=True)
    C.save(BASE / 'verification.json', result)
    print(env['json'].dumps(result))


if __name__ == '__main__': main()
