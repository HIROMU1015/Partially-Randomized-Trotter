#!/usr/bin/env python3
"""Read-only SHA/schema check of D2 preparation. There is no launch command."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/non_df_h4_d2_preparation/2026-10-24'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    contract = read(BASE/'contract_DRAFT.json')
    prep = read(BASE/'preparation.json')
    checks = {}
    checks['draft_contract_SHA'] = sha(BASE/'contract_DRAFT.json') == prep['contract_sha256']
    checks['disabled_no_authority'] = (all(contract[k] is False for k in (
        'execution_authorized', 'grant_valid', 'dispatch_enabled'))
        and all(contract[k] is None for k in ('grant', 'scientific_SOURCE', 'runtime_seal', 'launch_command')))
    checks['two_fixed_targets'] = ([v['case'] for v in contract['targets']] == ['pauli_jw_PF', 'pauli_jw_PR']
        and all(v['trajectory_index'] == 0 for v in contract['targets'])
        and contract['future_scientific_call_limits'] == dict(
            saved_body_validation=2, central_control_build=2, cost_transpile=2,
            generic_full_build=0, generic_baseline_compile=0, sampling=0, state_signal=0, N_G=0))
    checks['one_shot_no_escalation'] = (contract['automatic_retry'] is False
        and contract['automatic_resume'] is False and contract['automatic_D3'] is False
        and contract['mandatory_STOP'] is True)
    for name, manifest in (('preservation', read(BASE/'preservation_before.json')),
                           ('inputs', prep['input_files']), ('source', prep['code_files'])):
        bad = [p for p, h in manifest.items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
        checks[name+'_SHA'] = dict(files=len(manifest), mismatches=bad, passed=not bad)
    tests = read(BASE/'artificial_tests.json')
    checks['artificial_tests'] = (tests['tests'] == 12 and tests['failures'] == 0 and tests['errors'] == 0
        and tests['real_H4_build'] == 0 and tests['real_H4_compile'] == 0 and tests['scientific_grants_issued'] == 0)
    baseline = read(ROOT/'artifacts/non_df_h4_d1_source_execution/2026-10-24/result_summary.json')
    checks['fixed_profile'] = contract['fixed'] == baseline['fixed']
    target_ok = []
    for target in contract['targets']:
        prior = next(x for x in baseline['selected'] if x['case'] == target['case'])
        metrics = read(ROOT/target['D1_metrics'])
        source_input = read(ROOT/target['D1_input_metadata'])
        target_ok.append(all(target[k] == prior[k] for k in (
            'case', 'trajectory_id', 'trajectory_index', 'event_record_hash', 'trajectory_record_hash'))
            and target['baseline_RZ_quote'] == metrics['RZ'] == prior['saved_cosine_RZ']
            and target['baseline_current_input_fingerprint'] == source_input['input_fingerprint']
            and target['historical_input_fingerprint'] == source_input['historical_fingerprint']
            and target['body_qpy_sha256'] == sha(ROOT/target['body_qpy']))
    checks['baseline_join'] = all(target_ok) and len(target_ok) == 2
    checks['no_quantum_import'] = not any(k == 'qiskit' or k.startswith('qiskit.') for k in sys.modules)
    checks['no_science_dispatcher'] = contract['production_dispatcher_status'] == 'NOT_IMPLEMENTED_NO_EXECUTION_ENTRY'
    passed = all(v if isinstance(v, bool) else v['passed'] for v in checks.values())
    return dict(status='D2_PREPARATION_CHECK_PASS' if passed else 'D2_PREPARATION_CHECK_FAIL',
                checks=checks, mandatory_STOP=True, science_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', required=True)
    parser.parse_args()
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(result['status'] != 'D2_PREPARATION_CHECK_PASS')
