#!/usr/bin/env python3
"""Read saved technical evidence only; no molecular/Qiskit imports or dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/non_df_h4_execution_foundation/2026-10-23'
CASES = ('pauli_jw_PF', 'pauli_jw_PR', 'df_native_PF', 'df_native_PR')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def tests(path):
    cases = ET.parse(path).getroot().findall('.//testcase')
    check(not any(t.find('failure') is not None or t.find('error') is not None for t in cases), str(path))
    return cases


def verify():
    plan = read(BASE/'H4_execution_contract_DRAFT.json')
    check(all(plan[k] is False for k in ['execution_authorized', 'grant_valid', 'dispatch_enabled']), 'ACTIVE_CONTRACT')
    check(not (ROOT/'.h4_execution_claims').exists(), 'LIVE_H4_CLAIM_EXISTS')
    protected = read(BASE/'preservation_before.json')['files']
    changes = []
    for name, expected in protected.items():
        path = ROOT/name
        actual = {'symlink': str(path.readlink())} if path.is_symlink() else {'sha256': sha(path)}
        if actual != expected:
            changes.append(name)
    allowed = ['src/trotterlib/README.md']
    check(set(changes).issubset(allowed), 'PROTECTED_SOURCE_OR_RESULT_CHANGED:'+str(changes))

    broad = tests(BASE/'technical/final_tests.xml')
    latest = tests(BASE/'technical/final_new_tests.xml')
    old = [t for t in broad if not t.attrib['classname'].startswith(('tests.test_h4_event_audit', 'tests.test_h4_execution_foundation'))]
    check(len(old) == 371 and len(latest) == 65, 'TEST_SCOPE_OR_COUNT')
    check(not any(t.find('skipped') is not None for t in old+latest), 'UNEXPECTED_SKIP')

    run = BASE/'technical/eight_qubit_final'
    summary = read(run/'summary.json')
    check(summary['status'] == 'FOUR_CASES_COMPLETE' and len(summary['rows']) == 20, 'ARTIFICIAL_COVERAGE')
    check(all(r['total_RZ'] is None and r['shots']['N_total'] is None for r in summary['rows']), 'NULL_ACCOUNTING')
    records = events = compiles = 0
    probability_errors, trajectory_errors, forced_errors = [], [], []
    calls = {}
    for case in CASES:
        folder = run/'run'/case
        result = read(folder/'result.json')
        check(read(folder/'status.json')['artificial_only'], 'MOLECULAR_RESULT_IN_TECHNICAL_RUN')
        calls[case] = read(folder/'actual_call_accounting.json')
        check(calls[case]['event_audit_additional_draws'] == calls[case]['validation_replay_draws'] == 0, 'REPLAY')
        compiles += calls[case]['wrapper_compile_requests']
        probability_errors.append(result['circuit_cost']['compiled_signal_max_error'])
        validation = read(folder/'validation.json')
        for row in validation['trajectory_operator_checks']:
            check(row['gate']['PASS'] and not row['replayed_sampling'], 'TRAJECTORY_CHECK')
            trajectory_errors.append(row['gate']['error'])
        for row in validation['forced_order_checks']:
            check(row['gate']['PASS'] and not row['compile'] and not row['cost_sample'], 'FORCED_CHECK')
            forced_errors.append(row['gate']['error'])
        seen = set()
        for index, sample in enumerate(result['circuit_cost']['samples']):
            tid = sample['trajectory_id']
            check(tid not in seen, 'DUPLICATE_TRAJECTORY'); seen.add(tid)
            joined = read(folder/'trajectories'/tid/'price_join.json')
            trajectory_hash = joined.pop('trajectory_record_hash')
            check(digest(joined) == trajectory_hash == sample['trajectory_record_hash'], 'PRICE_JOIN_HASH')
            check(joined['compiled_RZ'] == sample['RZ'] and joined['seed'] == sample['seed'], 'PRICE_SEED_JOIN')
            check(joined['binding'] == result['provenance']['source'] and joined['case_id'] == case, 'INPUT_CASE_BINDING')
            event_record = read(folder/'trajectories'/tid/'events.json')
            event_hash = event_record.pop('event_record_hash')
            check(digest(event_record) == event_hash == joined['event_record_hash'], 'EVENT_HASH')
            check(all(joined[k] == v for k, v in event_record.items()), 'PRECOMPILE_EVENT_MATCH')
            check(tid == digest(dict(case_id=case, binding=joined['binding'], index=index, seed=sample['seed'])), 'TRAJECTORY_ID')
            expected = 32 if case.endswith('_PR') else 0
            check(len(joined['events']) == joined['event_count'] == expected, 'EVENT_COUNT')
            for i, event in enumerate(joined['events']):
                check((event['event_position'], event['outer_step_index'], event['inner_RTE_position']) == (i, i//4, i%4), 'EVENT_POSITION')
                check(event['distribution_id'] == joined['distribution_id'], 'EVENT_DISTRIBUTION')
            records += 1; events += expected
    check((records, events, compiles) == (6, 128, 12), 'ARTIFICIAL_CALL_BUDGET')
    for name in ['STOP.json', 'input_run/STOP.json', 'run/STOP.json']:
        stop = read(run/name)
        check(stop['all_children_reaped'] and not stop['automatic_retry'] and not stop['automatic_next_stage'], 'CHILD_CLEANUP_OR_STOP')
        check(not stop.get('pending_owned_processes', []), 'PENDING_CHILD')

    source_file = BASE/'SOURCE.json'
    if source_file.exists():
        source = read(source_file)
        check(all(sha(ROOT/name) == expected for name, expected in source['files'].items()), 'TECHNICAL_SOURCE_BYTES')
    inventory = BASE/'evidence_inventory.json'
    if inventory.exists():
        for name, expected in read(inventory)['files'].items():
            p = BASE/name
            actual = {'symlink': str(p.readlink())} if p.is_symlink() else {'sha256': sha(p)}
            check(actual == expected, 'EVIDENCE_BYTES:'+name)
    return dict(status='PASS_SAVED_TECHNICAL_EVIDENCE_ONLY', real_H4_scientific_calls=0,
        execution_authorized=False, grant_valid=False, dispatch_enabled=False,
        preservation=dict(inspected=len(protected), unchanged=len(protected)-len(changes),
            allowed_documentation_index_changes=changes, scientific_source_and_old_artifact_changes=0),
        tests=dict(old_regression_pass=len(old), new_final_pass=len(latest), unique_pass=len(old)+len(latest),
            fail=0, error=0, skip=0, deselected_molecular_tests=3,
            scope='21 selected existing modules plus two new modules; not a full repository suite'),
        artificial=dict(system_qubits=8, wrapper_qubits=9, cases=4, precision_rows=20, null_N_G_rows=20,
            joined_trajectories=records, captured_production_events=events, wrapper_compile_requests=compiles,
            compiled_probability_max_error=max(probability_errors), trajectory_max_error=max(trajectory_errors),
            forced_order_max_error=max(forced_errors), all_children_reaped=True, actual_call_accounting=calls),
        mandatory_STOP=True, scientific_grant_issued=False, push=False, detailed_scientific_review=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered)
    else:
        print(rendered, end='')
