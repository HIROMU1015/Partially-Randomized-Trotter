"""Read saved hashes, receipts and scalar identities; never execute a state action.

Uses the existing metadata helpers. No Qiskit/numpy/Arb imports, circuit decode,
sampling, vector reduction, new signal evaluation, or N/G accounting.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
NS = BASE / 'run01'
spec = importlib.util.spec_from_file_location('r0_saved_metadata', BASE / 'prepare_metadata.py')
meta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta)
read, sha, digest, save = meta.read, meta.sha, meta.digest, meta.save


def require(value, label):
    if not value:
        raise ValueError(label)


def main():
    seal = read(BASE / 'seal.json')
    source = seal['SOURCE']
    require(source == '84a326598b006384d13c7a354a3eec6881ee841c', 'FIXED_SOURCE')
    for p, h in seal['code_files'].items():
        require(sha(ROOT / p) == h, 'CODE:' + p)
        committed = subprocess.check_output(['git', 'show', source + ':' + p], cwd=ROOT)
        require(meta.hashlib.sha256(committed).hexdigest() == h, 'COMMITTED_CODE:' + p)
    for p, h in seal['bound_manifests'].items():
        require(sha(BASE / p) == h, 'SEALED_MANIFEST:' + p)
    counts = {}
    for label, files in (
        ('inputs', read(BASE / 'inputs.json')['files']),
        ('preserved', read(BASE / 'preservation_before.json')),
        ('runtime', read(BASE / 'runtime.json')['files']),
    ):
        for p, h in files.items():
            require(sha(ROOT / p) == h, label + ':' + p)
        counts[label] = len(files)
    require(counts == dict(inputs=89, preserved=192, runtime=6999), 'EXPECTED_FILE_COUNTS')
    runtime = read(BASE / 'runtime.json')
    require(sha(Path(runtime['python'])) == seal['python_sha256'], 'PYTHON')
    prior = read(BASE / 'prior_manifest_rows.json')
    manifest = read(ROOT / 'artifacts/validation_manifest.json')
    require(prior['row_count'] == 182, 'OLD_MANIFEST_COUNT')
    for key, h in prior['row_sha256'].items():
        require(digest(manifest[key]) == h, 'OLD_MANIFEST_ROW:' + key)

    contract, grant = read(BASE / 'active_contract.json'), read(NS / 'grant.json')
    require(grant['grant_id'] == digest({k: v for k, v in grant.items() if k != 'grant_id'}), 'GRANT_ID')
    require(grant['contract_sha256'] == sha(BASE / 'active_contract.json'), 'CONTRACT_BINDING')
    require(grant['seal_sha256'] == contract['seal_sha256'] == sha(BASE / 'seal.json'), 'SEAL_BINDING')
    require(grant['approval_sha256'] == contract['approval_sha256'] == sha(BASE / 'approval.txt'), 'APPROVAL_BINDING')
    require(grant['namespace'] == str(NS) and grant['one_shot'] is True, 'NAMESPACE_ONE_SHOT')
    consumed = read(NS / 'CONSUMED.json')
    require(consumed['grant_id'] == grant['grant_id'] and consumed['nonrefundable'], 'CONSUMED')
    require(contract['SOURCE'] == source and contract['axis'] == 'cosine', 'FIXED_SCOPE')
    require(contract['schema'] == read(BASE / 'output_schema.json'), 'SCHEMA')
    require(contract['mandatory_STOP'] and not contract['automatic_retry'] and not contract['automatic_next_stage'], 'MANDATORY_STOP')
    require(all(v == 0 for v in contract['forbidden_calls'].values()), 'FORBIDDEN_CONTRACT')
    actions = ['PF_input', 'PF_output', 'PR0_input', 'PR0_output', 'PR0_C_target']
    limits = {k: 1 for k in actions + ['preflight', 'PF_diagnostic', 'PR0_diagnostic']}
    ledger = read(NS / 'call_ledger.json')
    require(ledger['used'] == ledger['limits'] == contract['limits'] == limits, 'FINITE_CALL_COUNTS')
    require(contract['state_action_limits'] == {k: 1 for k in actions}, 'ACTION_LIMITS')
    require(dict(Counter(e['call'] for e in ledger['events'])) == limits, 'CALL_EVENTS')
    require(all(e['reserved_before_call'] for e in ledger['events']), 'RESERVED_BEFORE_CALL')

    summary = read(NS / 'result_summary.json')
    require(summary['logical_diagnostics'] == 2 and summary['successful_state_actions'] == summary['total_state_action_attempts'] == 5, 'SUCCESS_COUNTS')
    require(summary['binary64_only'] and summary['mandatory_STOP'], 'BINARY64_STOP')
    require(summary['forbidden_calls'] == contract['forbidden_calls'], 'FORBIDDEN_SUMMARY')
    require(summary['historical_hash'] == 'FAIL_CAUSE_UNKNOWN_UNCHANGED' and summary['original_native'] == 'UNKNOWN_UNCHANGED', 'OLD_UNKNOWN')
    require(summary['original_N_G_null'] and summary['C_L'] == 'COMBINATION_BLOCKED' and summary['old_compiler'] == 'MISSING_EVIDENCE', 'OLD_GATES')
    require(summary['rho_upper'] is None and summary['PR_law_rho'] is None and not summary['formal_certificate_accepted'], 'NO_CERTIFICATE')
    require(summary['population_cost_tail'] == 'UNKNOWN', 'PR_TAIL_UNKNOWN')
    preflight = read(NS / 'preflight.json')
    require(preflight == read(BASE / 'structural_preflight.json'), 'PREFLIGHT_SAME')
    require(preflight['status'] == 'PASS' and preflight['QPY_decodes'] == 8 and preflight['state_actions'] == 0, 'PREFLIGHT_SCOPE')
    targets = read(ROOT / 'artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json')['targets']
    guards = read(ROOT / 'artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json')['axes']
    for rec in preflight['records']:
        target = next(t for t in targets if t['key'] == rec['key'])
        require(rec['trajectory_id'] == target['trajectory_id'], 'TRAJECTORY')
        for role, prefix in [('input_qpy', 'input'), ('compiled_qpy', 'output')]:
            require(rec[role]['sha256'] == target[prefix + '_sha256'], 'QPY_SHA')
            require(rec[role]['fingerprint'] == target[prefix + '_fingerprint'], 'QPY_FINGERPRINT')
            require(rec[role]['terminal_measurement'] == 'wire8 -> clbit0', 'ANCILLA')

    state = read(NS / 'state_identity.json')
    require(not state['raw_state_changed'] and state['state_hash'] == preflight['state_hash'], 'RAW_STATE')
    nu = F(state['nu_exact'])
    pf = read(NS / 'PF_C_target_saved.json')
    require(F(pf['saved_norm_interval']['lower']) <= nu <= F(pf['saved_norm_interval']['upper']), 'RAW_NORM_IN_SAVED_C_INTERVAL')
    lo, hi = map(F, pf['normalized_interval_exact'])
    require(lo == F(pf['saved_raw_interval']['lower']) / nu and hi == F(pf['saved_raw_interval']['upper']) / nu, 'PF_SAVED_INTERVAL_SCALAR_IDENTITY')
    require(lo <= F(pf['representative_exact']) <= hi and F(pf['representative_exact']) == (lo + hi) / 2, 'PF_MIDPOINT')
    require(not pf['compiled_signal_error_enclosed'], 'NO_NEW_INTERVAL_DIAGNOSIS')
    for name in actions:
        claim, receipt = read(NS / (name + '_claim.json')), read(NS / (name + '.json'))
        event = next(e for e in ledger['events'] if e['call'] == name)
        require(event['UTC'] <= claim['UTC'] <= receipt['UTC'], 'RECEIPT_ORDER:' + name)
        require(claim['reserved_before_call'] and claim['call'] == receipt['call'] == name, 'RECEIPT:' + name)
        require(receipt['binary64'] and receipt['status'] == 'EMPIRICAL_ONLY_NO_RHO_UPPER' and receipt['numerical_error_upper'] is None, 'EMPIRICAL_RECEIPT')
        require(len(receipt['output_state_pairs']) == (256 if name == 'PR0_C_target' else 512), 'OUTPUT_SHAPE')
        if name != 'PR0_C_target':
            require(not receipt['state_renormalized'] and not receipt['output_norm_division'] and not receipt['probability_API_used'], 'RAW_RAY_OBSERVABLE')
            require(receipt['raw_norm'] == state['nu_binary64'], 'SAME_NORM')
            require(receipt['signal'] == (receipt['p0_raw'] - receipt['p1_raw']) / receipt['raw_norm'], 'SAVED_SCALAR_OBSERVABLE_IDENTITY')
    selected = read(NS / 'PR0_C_target.json')
    require(not selected['PR_law_mean_used'] and not selected['normalization_B_applied_to_target'], 'SELECTED_TARGET_NOT_LAW_MEAN')

    results = read(NS / 'results.json')
    require([r['key'] for r in results] == ['PF', 'PR0'], 'RESULT_SCOPE')
    for row in results:
        key = row['key']
        target = next(t for t in targets if t['key'] == key + ('0_cosine' if key == 'PF' else '_cosine'))
        guard = next(g for g in guards if (g['case'], g['epsilon_sig'], g['axis']) == ('pauli_jw_' + row['method'], '0.0005', 'cosine'))
        require(row == read(NS / (key + '_result.json')), 'PER_KEY_RESULT')
        require(row['trajectory_id'] == target['trajectory_id'], 'RESULT_TRAJECTORY')
        mi, mo = read(NS / (key + '_input.json'))['signal'], read(NS / (key + '_output.json'))['signal']
        mc = F(pf['representative_exact']) if key == 'PF' else F(selected['signal'])
        require(row['m_input'] == mi and row['m_output'] == mo and row['m_C'] == float(mc), 'SAVED_SIGNALS')
        require(F(row['m_C_representative_exact']) == mc, 'EXACT_REPRESENTATIVE')
        b, g = F(row['B_saved_exact']), F(row['rho_budget_exact'])
        require(b == F(target['B_saved']) == F(guard['scale']['B_saved_exact']), 'B_SAVED_ONCE')
        require(g == F(guard['rho_guard_exact']), 'QUOTED_GUARD')
        delta, eta, direct = F(row['delta_exact']), F(row['eta_exact']), F(row['d_exact'])
        require(delta == b * abs(F(mo) - F(mi)) and eta == b * abs(F(mi) - mc) and direct == b * abs(F(mo) - mc), 'SAVED_RESIDUAL_IDENTITIES')
        require(direct <= delta + eta and row['triangle_verified'], 'SAVED_TRIANGLE')
        require(F(row['d_over_budget_exact']) == direct / g and F(row['budget_minus_d_exact']) == g - direct, 'SAVED_BUDGET_IDENTITIES')
        for name in ['delta', 'eta', 'd', 'B_saved', 'rho_budget', 'd_over_budget', 'budget_minus_d']:
            require(row[name] == float(F(row[name + '_exact'])), 'DISPLAY:' + name)
        require(direct < g and row['empirical_relation'] == 'BELOW_SAVED_BUDGET', 'EMPIRICAL_RELATION')
        require(row['status'] == 'EMPIRICAL_ONLY_NO_RHO_UPPER' and not row['LOCAL_BUDGET_MET'] and not row['formal_certificate_accepted'], 'NO_GUARANTEE_PROMOTION')
        require(all(row[k] is None for k in ['rho_upper', 'numerical_error_upper', 'PR_law_rho']), 'NULL_UPPERS')

    stop, native = read(NS / 'STOP.json'), read(NS / 'native/worker_native_exit.json')
    require(stop['status'] == 'COMPLETE_MANDATORY_STOP' and stop['validation_error'] is None, 'STOP')
    require(stop['native_exit'] == native and native['exit_code'] == 0 and native['reaped'], 'NATIVE_EXIT_REAP')
    require(stop['all_owned_reaped'] and not stop['original_H4_native_verified'] and not stop['automatic_next_stage'], 'STOP_SCOPE')
    require(native['scope'] == 'NEW_R0_BINARY64_EMPIRICAL_ONLY', 'NEW_NATIVE_ONLY')
    require(read(NS / 'native/STOP.json')['reaped_pids'] == [native['pid']], 'OWNED_WORKER')
    require(read(NS / 'WORKER_CLAIM.json')['pid'] == native['pid'] == read(NS / 'native/worker_identity.json')['pid'], 'WORKER_IDENTITY')
    require((NS / 'native/stderr.log').stat().st_size == 0, 'STDERR_EMPTY')
    inventory = {str(p.relative_to(ROOT)): sha(p) for p in sorted(NS.rglob('*')) if p.is_file()}
    save(BASE / 'output_inventory.json', inventory)
    result = dict(status='PASS_SAVED_EVIDENCE_CONSISTENCY_ONLY', UTC=datetime.now(timezone.utc).isoformat(),
                  SOURCE=source, code_files=len(seal['code_files']), bound_manifests=len(seal['bound_manifests']),
                  **counts, old_manifest_rows_unchanged=prior['row_count'], output_files=len(inventory),
                  output_inventory_sha256=sha(BASE / 'output_inventory.json'), grant_id=grant['grant_id'],
                  logical_diagnostics=2, successful_state_actions=5, total_state_action_attempts=5,
                  new_state_actions_in_this_verifier=0, QPY_decodes_in_this_verifier=0,
                  scalar_checks='identities among already saved values only; no vector reductions or new signal analysis',
                  native_exit=native, mandatory_STOP=True, formal_certificate_accepted=False,
                  rho_upper=None, numerical_error_upper=None, PR_law_rho=None,
                  author_checks_only=True, historical_states_unchanged=True)
    save(BASE / 'verification.json', result)
    print(meta.json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
