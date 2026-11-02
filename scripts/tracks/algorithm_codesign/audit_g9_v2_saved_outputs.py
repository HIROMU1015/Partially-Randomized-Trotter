"""G9 v2 saved bytes/rational accounting only; no science-module imports."""
import hashlib
import json
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / 'artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceil(q):
    return (q.numerator + q.denominator - 1) // q.denominator


def saved_log_upper(q):
    """Fixed 96-term rational accounting check, not a budget search."""
    n = 0
    while q >= 2:
        n += 1
        q /= 2
    def bound(z):
        return 2 * sum((z ** (2*k+1) / (2*k+1) for k in range(96)), F(0)) + 2*z**193 / (193*(1-z*z))
    return n * bound(F(1, 3)) + bound((q-1)/(q+1))


def saved_cap(M, z):
    if z == 1:
        return M
    v = M*z
    square = 20*v
    scale = 1 << 256
    k = isqrt(square.numerator*scale*scale // square.denominator)
    lo = F(k, scale)
    hi = lo if lo*lo == square else F(k+1, scale)
    return min(M, ceil(v+hi+F(20, 3)))


def main():
    c = json.loads((PREP / 'contract_v2.json').read_text())
    directory = ROOT / c['result_directory']
    raw = json.loads((directory / 'result_v1.json').read_text())
    marker = json.loads((directory / 'one_shot_consumed.json').read_text())
    stop = json.loads((directory / 'STOP.json').read_text())
    auth = json.loads((ROOT / c['authorization_path']).read_text())
    checks = {}
    checks['complete_status_no_technical_reason'] = raw['status'] == stop['status'] == 'G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE' and raw['technical_reason'] is None and raw['prefix_rows_usable_for_final_research_decision'] is True
    checks['source_authorization_binding'] = raw['source_commit'] == auth['source_commit'] == marker['source_commit'] and raw['execution_HEAD'] == raw['authorization_commit'] == marker['execution_HEAD']
    checks['direct_authorization_child'] = subprocess.check_output(['git', '-C', str(ROOT), 'show', '-s', '--format=%P', raw['execution_HEAD']], text=True).split() == [raw['source_commit']]
    changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', '-z', raw['source_commit'], raw['execution_HEAD']]).decode().split('\0')[:-1]
    checks['authorization_only_diff'] = set(changed) == {c['authorization_path'], c['optional_receipt_path']}
    checks['authorization_identity'] = digest(ROOT/c['authorization_path']) == raw['authorization_sha256'] == marker['authorization_sha256'] and marker['authorization'] == auth
    checks['contract_identity'] = digest(PREP/'contract_v2.json') == raw['contract_sha256'] == marker['contract_sha256'] == auth['contract_sha256']
    checks['marker_identity'] = digest(directory/'one_shot_consumed.json') == raw['marker_sha256']
    checks['one_shot_no_retry'] = raw['runs'] == marker['runs'] == auth['runs'] == 1 and raw['retries'] == marker['retries'] == auth['retries'] == 0
    checks['STOP_no_next_stage_or_adoption'] = raw['mandatory_STOP'] is True and stop['mandatory_STOP'] is True and raw['next_science_authorized'] is False and stop['next_science_authorized'] is False and raw['method_or_novelty_adopted'] is False
    checks['no_unregistered_science'] = raw['new_inputs_grid_LP_DF_molecule_NPZ_GPU_quantum_measurement_trajectory'] == raw['actual_quantum_shots'] == 0
    manifest = json.loads((PREP/'source_manifest_v2.json').read_text())
    critical_errors = [p for p, h in manifest['sha256'].items() if digest(ROOT/p) != h]
    checks['critical_source_unchanged'] = not critical_errors
    ledger = json.loads((ROOT/c['protected_ledger']).read_text())
    old_errors = []
    for p, record in ledger.items():
        if p.lower().endswith('.npz'):
            raise PermissionError('NPZ access forbidden')
        data = (ROOT/p).read_bytes()
        if p in c['append_only_paths']:
            data = data[:record['bytes']]
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            old_errors.append(p)
    checks['protected_history_unchanged'] = not old_errors
    checks['recorded_protected_checks'] = all(raw[k]['protected_paths'] == len(ledger) and raw[k]['violations'] == [] for k in ('protected_before', 'protected_after'))
    old_raw_directory = ROOT/'artifacts/track_b_g9_p5_native_result/2026-10-10/v1'
    old_audit = json.loads((old_raw_directory/'saved_output_audit.json').read_text())
    checks['v1_raw_marker_STOP_unchanged'] = all(digest(old_raw_directory/p) == h for p, h in old_audit['immutable_result_hashes'].items())
    keys = json.loads((ROOT/c['synthesis_inventory']).read_text())['keys']
    cache = raw['synthesis_cache']
    checks['exact_frozen_inventory'] = set(cache) == {k['ratio'] for k in keys} and len(cache) == 19
    sequence_checks = []
    old_cache = json.loads((ROOT/c['reuse_G7_result']).read_text())['synthesis_cache']
    for key in keys:
        e = cache[key['ratio']]
        s = e['sequence']
        ok = (e['angle_key'] == key['angle_key'] and e['epsilon'] == key['epsilon']
              and e['key'] == key['angle_key']+':epsilon:'+key['epsilon']
              and set(s) <= set('HTtSXW')
              and hashlib.sha256(s.encode()).hexdigest() == e['sequence_sha256']
              and e['T_count'] == s.count('T')+s.count('t')
              and e['Tdagger_count'] == s.count('t')
              and e['one_qubit_count'] == len(s)-s.count('W')
              and e['global_W_count'] == s.count('W')
              and e['error_pass'] is True and 0 <= F(e['strict_operator_error_upper']) <= F(key['epsilon']))
        if key['acquisition'] == 'reuse_G7':
            ok = ok and e == old_cache[key['ratio']] and e['sequence_sha256'] == key['sequence_sha256']
        else:
            ok = ok and e['resource']['wall_seconds'] <= c['caps']['per_key_wall_seconds'] and e['resource']['cpu_seconds'] <= c['caps']['per_key_cpu_seconds']
        sequence_checks.append({'ratio': key['ratio'], 'acquisition': key['acquisition'], 'passed': ok,
                                'sequence_sha256': e['sequence_sha256'], 'T_count': e['T_count'],
                                'Tdagger_count': e['Tdagger_count'], 'saved_strict_error_upper': e['strict_operator_error_upper']})
    checks['all_saved_sequences_and_guards'] = all(e['passed'] for e in sequence_checks)
    checks['acquisition_counts_and_cache_cap'] = raw['new_synthesis_calls'] == 1 and raw['reused_G7_sequence_keys'] == 18 and raw['synthesis_cache_bytes'] == len(json.dumps(cache).encode()) <= 1024*1024 and len(cache) <= 32
    expected = [(a, 'direct_primary') for a in c['direct_primary_arms']] + [(a, 'generic_helper_diagnostic') for a in c['helper_diagnostic_arms']]
    checks['complete_unique_row_set'] = [(r['arm'], r['implementation']) for r in raw['rows']] == expected and len(raw['rows']) == 11
    eps, rho = F(c['primitive_error']), F(c['rho'])
    bias = 2*(8*rho+8*(1+rho)*2*eps)
    log = saved_log_upper(2/F(c['alpha_axis']))
    row_checks = []
    event_count = zero_T_CTS = 0
    for row in raw['rows']:
        budget = row['budget']
        N, cap = budget['N_per_axis'], budget['accepted_call_cap_two_axes']
        sums = {k: F(0) for k in ('T', 'CX', '1Q')}
        q = m2 = F(0)
        W = F(0)
        counts_match = weights_match = diagnostics_pass = True
        max_T = 0
        for binding in row['events']:
            event_count += 1
            event, saved_cost = binding['event'], binding['cost']
            proposal, coefficient, weight = (F(event[k]) for k in ('proposal', 'coefficient', 'weight'))
            q += proposal
            m2 += proposal*weight*weight
            W = max(W, weight)
            weights_match = weights_match and proposal > 0 and coefficient > 0 and proposal*weight == coefficient
            literal = {k: 0 for k in sums}
            rotation_count = 0
            for g in binding['native_ir']:
                if g[0] == 'R':
                    e = cache[g[2]]
                    literal['T'] += e['T_count']
                    literal['1Q'] += e['one_qubit_count']
                    rotation_count += 1
                elif g[0] == 'CX':
                    literal['CX'] += 1
                else:
                    literal['T'] += g[0] in ('T', 't')
                    literal['1Q'] += g[0] not in ('W', 'w')
            counts_match = counts_match and literal == {k: saved_cost[k] for k in sums} and rotation_count in (0, 2) and F(saved_cost['strict_error_upper']) == rotation_count*eps
            diagnostics_pass = diagnostics_pass and binding['ideal_phase_reference_error'] <= 1e-10 and binding['realized_reference_error'] <= float(F(saved_cost['strict_error_upper']))+1e-10
            max_T = max(max_T, saved_cost['T'])
            for k in sums:
                sums[k] += proposal*saved_cost[k]
            if row['arm'] == 'matched_CTS' and saved_cost['T'] == 0:
                zero_T_CTS += 1
        ok = (weights_match and counts_match and diagnostics_pass
              and q == F(row['reference_acceptance']) and 0 < q <= F(budget['acceptance_upper']) <= 1
              and m2 == F(row['reference_m2']) <= F(budget['m2_upper'])
              and W == F(row['reference_range']) <= F(budget['range_upper'])
              and F(budget['common_bias_upper']) == bias
              and F(budget['remaining']) == F(c['epsilon_axis'])-bias > 0
              and F(budget['alpha_axis']) == F(c['alpha_axis'])
              and N == ceil(log*(2*F(budget['m2_upper'])/F(budget['remaining'])**2+4*F(budget['range_upper'])/(3*F(budget['remaining']))))
              and 0 < N <= c['caps']['shot_cap_per_axis']
              and cap == saved_cap(2*N, F(budget['acceptance_upper']))
              and budget['hard_attempt_cap_two_axes'] == 2*N
              and row['registered_worst_event_T'] == max_T
              and row['accepted_tail_T_upper'] == cap*max_T
              and row['hard_attempt_T_upper'] == 2*N*max_T
              and F(row['T_prep_readout_affine_coefficient']) == F(row['expected_accepted_calls']) == 2*N*q
              and F(row['common_1Q_outer_prep_readout_two_axes']) == 5*N*q
              and row['workspace_beyond_system'] == (1 if row['primary'] else 2)
              and row['physical_quantum_shots_executed'] == 0
              and row['ideal_mean_matrix_residual_diagnostic'] <= 1e-10
              and row['all_event_reference_not_injected_into_local_generator'] is True)
        ok = ok and all(F(row['per_trial_native_cost'][k]) == sums[k] and F(row['two_axis_expected_native_cost'][k]) == 2*N*sums[k] for k in sums)
        row_checks.append({'arm': row['arm'], 'implementation': row['implementation'], 'events': len(row['events']), 'passed': ok})
    checks['all_saved_row_accounting_caps_diagnostics'] = all(r['passed'] for r in row_checks)
    checks['row_event_counts'] = event_count == 1866 and [r['events'] for r in row_checks] == [273, 264, 258, 63, 63, 24, 273, 264, 258, 63, 63]
    checks['CTS_zero_T_real_events_preserved'] = zero_T_CTS == 10
    checks['familywise_allocation'] = 22*F(c['alpha_axis'])+11*F(c['resource_failure_per_row']) == F(c['familywise_failure']) == F(1, 20)
    checks['P5_recorded_formal_audit'] = raw['formal_P5_audit']['formal_coefficients_exact'] is True and raw['formal_P5_audit']['parents_checked'] == 31 and raw['formal_P5_audit']['groups'] == 10 and raw['formal_P5_audit']['events'] == 63
    prep_math = json.loads((ROOT/'artifacts/track_b_g9_p5_native_preparation/2026-10-10/mathematical_preparation_v1.json').read_text())
    checks['CTS_certificate_preparation_binding'] = raw['CTS_operator_certificate'] == prep_math['CTS_first_moment_preparation'] and F(raw['CTS_operator_certificate']['coefficient_mean_error_upper']) <= 8*rho
    u = raw['resource']
    checks['wall_CPU_RSS_output_caps'] = u['wall_seconds'] <= c['caps']['wall_seconds'] and u['cpu_seconds'] <= c['caps']['cpu_seconds'] and u['peak_RSS_KiB'] <= c['caps']['RSS_MiB']*1024 and (directory/'result_v1.json').stat().st_size <= c['caps']['output_bytes']
    result = {'scope': 'saved exact rational/sequence/count/hash/budget verification only; no science module imports or matrix/strict-guard/generator/backend calls',
              'passed': all(checks.values()), 'checks': checks, 'row_checks': row_checks,
              'sequence_checks': sequence_checks, 'event_bindings': event_count,
              'zero_T_CTS_real_events': zero_T_CTS, 'critical_source_paths': len(manifest['sha256']),
              'critical_source_errors': critical_errors, 'protected_paths': len(ledger), 'protected_errors': old_errors,
              'source_S': raw['source_commit'], 'authorization_A': raw['authorization_commit'],
              'immutable_result_hashes': {n: digest(directory/n) for n in ('result_v1.json', 'one_shot_consumed.json', 'STOP.json')},
              'old_v1_hashes': old_audit['immutable_result_hashes'],
              'classification_not_recomputed': True, 'no_new_scientific_claim_or_threshold': True,
              'real_backend_or_science_calls_in_audit': 0, 'retry': 0, 'mandatory_STOP': True}
    with (directory/'saved_output_audit_v2.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'checks': len(checks), 'event_bindings': event_count,
                      'zero_T_CTS_real_events': zero_T_CTS, 'failed_checks': [k for k,v in checks.items() if not v], 'mandatory_STOP': True}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
