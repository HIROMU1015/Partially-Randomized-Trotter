"""Read saved G9 bytes only; never import a generator, solver or synthesizer."""
import hashlib
import json
import subprocess
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / 'artifacts/track_b_g9_p5_native_preparation/2026-10-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    c = json.loads((PREP / 'contract_v1.json').read_text())
    directory = ROOT / c['result_directory']
    raw = json.loads((directory / 'result_v1.json').read_text())
    marker = json.loads((directory / 'one_shot_consumed.json').read_text())
    stop = json.loads((directory / 'STOP.json').read_text())
    checks = {}
    checks['marker_digest'] = raw['marker_sha256'] == digest(directory / 'one_shot_consumed.json')
    checks['source_binding'] = raw['source_commit'] == marker['source_commit']
    checks['contract_binding'] = raw['contract_sha256'] == marker['contract_sha256'] == digest(PREP / 'contract_v1.json')
    checks['review_binding'] = digest(ROOT / c['authorization']['snapshot']) == c['authorization']['sha256']
    checks['delegation_binding'] = marker['review_authorization'] == c['authorization']
    checks['one_run_no_retry'] = raw['runs'] == marker['runs'] == 1 and raw['retries'] == marker['retries'] == 0
    checks['mandatory_STOP'] = raw['mandatory_STOP'] is True and stop['mandatory_STOP'] is True and stop['next_science_authorized'] is False
    checks['classification_preserved'] = raw['status'] == stop['status'] == 'G9_TECHNICAL_INCONCLUSIVE'
    checks['no_outcome_prefix'] = raw['rows'] == [] and raw['prefix_rows_usable_for_final_research_decision'] is False
    checks['technical_reason'] = raw['technical_reason'] == 'TypeError: cannot create mpf from Fraction(1, 1000000)'
    checks['new_measurement_or_input_zero'] = raw['actual_quantum_shots'] == raw['new_inputs_grid_LP_DF_molecule_NPZ_GPU_quantum_measurement_trajectory'] == 0
    manifest = json.loads((PREP / 'source_manifest_v1.json').read_text())
    critical_errors = [p for p, h in manifest['sha256'].items() if digest(ROOT / p) != h]
    checks['critical_source_bytes'] = not critical_errors
    checks['test_preparation_only'] = manifest['focused_tests_passed'] is True and manifest['tests'] == 23
    ledger = json.loads((ROOT / c['protected_ledger']).read_text())
    old_errors = []
    for p, v in ledger.items():
        if p.lower().endswith('.npz'):
            raise PermissionError('NPZ access prohibited')
        data = (ROOT / p).read_bytes()
        if p in c['append_only_paths']:
            data = data[:v['bytes']]
        if hashlib.sha256(data).hexdigest() != v['sha256']:
            old_errors.append(p)
    checks['protected_bytes_and_prefixes'] = not old_errors
    checks['recorded_launch_protection'] = all(raw[k]['violations'] == [] and raw[k]['protected_paths'] == len(ledger) for k in ('protected_before', 'protected_after'))
    inventory = json.loads((PREP / 'synthesis_inventory_v1.json').read_text())
    old = json.loads((ROOT / c['reuse_G7_result']).read_text())['synthesis_cache']
    keys = {k['ratio']: k for k in inventory['keys']}
    identities = []
    for ratio, row in raw['synthesis_cache'].items():
        k = keys[ratio]
        s = row['sequence']
        ok = (k['acquisition'] == 'reuse_G7' and row == old[ratio]
              and row['angle_key'] == k['angle_key'] and row['epsilon'] == k['epsilon']
              and row['key'] == k['angle_key'] + ':epsilon:' + k['epsilon']
              and digest_string(s) == row['sequence_sha256'] == k['sequence_sha256']
              and set(s) <= set('HTtSXW')
              and row['T_count'] == s.count('T') + s.count('t')
              and row['Tdagger_count'] == s.count('t')
              and row['one_qubit_count'] == len(s) - s.count('W')
              and row['global_W_count'] == s.count('W')
              and row['error_pass'] is True
              and 0 <= F(row['strict_operator_error_upper']) <= F(k['epsilon']))
        identities.append({'ratio': ratio, 'identity_and_saved_guard_pass': ok})
    checks['reused_saved_identities'] = len(identities) == raw['reused_G7_sequence_keys'] == 18 and all(v['identity_and_saved_guard_pass'] for v in identities)
    checks['new_key_unacquired'] = raw['new_synthesis_calls'] == 1 and set(raw['synthesis_cache']) == {k['ratio'] for k in inventory['keys'] if k['acquisition'] == 'reuse_G7'}
    checks['shared_cache_cap'] = len(raw['synthesis_cache']) <= 32 and len(json.dumps(raw['synthesis_cache']).encode()) <= 1024 * 1024
    usage = raw['resource']
    checks['recorded_wall_cpu_RSS_within_caps'] = usage['wall_seconds'] <= c['caps']['wall_seconds'] and usage['cpu_seconds'] <= c['caps']['cpu_seconds'] and usage['peak_RSS_KiB'] <= c['caps']['RSS_MiB'] * 1024
    checks['output_cap'] = (directory / 'result_v1.json').stat().st_size <= c['caps']['output_bytes']
    checks['confidence_allocation'] = 22 * F(c['alpha_axis']) + 11 * F(c['resource_failure_per_row']) == F(c['familywise_failure'])
    checks['source_review_snapshot_unchanged'] = digest(ROOT / c['authorization']['snapshot']) == marker['review_authorization']['sha256']
    source_paths = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', '-z', raw['source_commit'] + '^', raw['source_commit']]).decode().split('\0')[:-1]
    publication_errors = []
    for p in source_paths:
        data = subprocess.check_output(['git', '-C', str(ROOT), 'show', raw['source_commit'] + ':' + p])
        current = (ROOT / p).read_bytes()
        equal = current.startswith(data) if p in c['append_only_paths'] else current == data
        if not equal:
            publication_errors.append(p)
    checks['all_source_commit_paths_preserved'] = not publication_errors
    result = {
        'audit_scope': 'saved bytes/rational arithmetic/source-text inspection only; no backend, matrix, generator or guard re-execution',
        'passed': all(checks.values()), 'checks': checks,
        'immutable_result_hashes': {n: digest(directory / n) for n in ('result_v1.json', 'one_shot_consumed.json', 'STOP.json')},
        'critical_paths': len(manifest['sha256']), 'critical_errors': critical_errors,
        'protected_paths': len(ledger), 'protected_errors': old_errors,
        'source_publication_paths': len(source_paths), 'source_publication_errors': publication_errors,
        'reused_sequence_saved_checks': identities,
        'new_synthesis_helper_attempts_recorded': raw['new_synthesis_calls'],
        'new_sequence_acquisitions': 0,
        'gridsynth_gates_invocations_inferred_from_source_and_error': 0,
        'backend_invocation_counter_in_raw_result': False,
        'backend_inference': 'numeric.synthesize evaluates mp.mpf(epsilon)/4 before invoking gridsynth_gates; Fraction conversion failed at argument evaluation. No backend return or new sequence exists.',
        'launch_tests_limitation': '23 tests covered formula/provider/phase/import; the new synthesis epsilon argument adapter was not tested.',
        'registered_native_rows_completed': 0, 'registered_matrix_checks_completed': 0,
        'registered_shot_budgets_completed': 0, 'full_G9_math_audit_stage_in_run_completed': False,
        'registered_preparation_mathematics_preserved_separately': True,
        'retries': 0, 'next_science_authorized': False, 'mandatory_STOP': True,
        'classification_not_recomputed': True, 'scientific_positive_or_negative_inference': False,
    }
    with (directory / 'saved_output_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'checks': len(checks), 'protected_paths': len(ledger), 'rows': 0, 'mandatory_STOP': True}))
    if not result['passed']:
        raise SystemExit(1)


def digest_string(value):
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__':
    main()
