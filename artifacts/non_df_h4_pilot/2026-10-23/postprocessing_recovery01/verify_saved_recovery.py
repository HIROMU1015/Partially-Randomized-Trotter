#!/usr/bin/env python3
"""Audit saved H4 data without accepting missing original supervisor completion.

Derived from ../verify_saved_run.py. Scientific checks unchanged. Read recovered
case table here; retain input STOP checks; explicitly leave native exit/reap/STOP
unverified. This is not the original complete-pipeline PASS. Stdlib only.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent.parent
PACK = BASE/'run01'
FOUNDATION = ROOT/'artifacts/non_df_h4_execution_foundation/2026-10-23'
CASES = ('pauli_jw_PF', 'pauli_jw_PR', 'df_native_PF', 'df_native_PR')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify(*, runtime_files=False):
    source, plan, grant, approval = [read(PACK/f) for f in ['SOURCE.json', 'contract.json', 'grant.json', 'approval.json']]
    check(grant['SOURCE'] == source['SOURCE'] == approval['SOURCE'] == '0c329b30a81accb35c212be8defa66fee0dd0f12', 'SOURCE')
    check(grant['contract_digest'] == approval['contract_digest'] == digest(plan), 'CONTRACT')
    check(grant['source_manifest_sha256'] == sha(PACK/'SOURCE.json'), 'SOURCE_SEAL')
    check(grant['approval_sha256'] == sha(PACK/'approval.json'), 'APPROVAL_SEAL')
    check(grant['pack_namespace'] == approval['pack_namespace'] == str(PACK), 'NAMESPACE')
    check(grant['authorized_cases'] == list(CASES), 'AUTHORIZED_CASES')
    check(grant['one_shot'] and not grant['automatic_retry'], 'ONE_SHOT')
    check(read(PACK/'user_authorization_record.json')['user_message'] == '次の作業に進んで大丈夫', 'USER_AUTHORIZATION')
    check(read(PACK/'CONSUMED.json')['grant_id'] == grant['grant_id'], 'CONSUMPTION')
    for name, expected in source['files'].items():
        check(sha(ROOT/name) == expected, 'SOURCE_FILE:'+name)
        committed = subprocess.check_output(['git', 'show', source['SOURCE']+':'+name], cwd=ROOT)
        check(hashlib.sha256(committed).hexdigest() == expected, 'COMMITTED_SOURCE:'+name)
    if runtime_files:
        for name, expected in source['environment']['files'].items():
            check(sha(name) == expected, 'RUNTIME_FILE:'+name)

    preserved = read(FOUNDATION/'preservation_before.json')['files']
    documentation_changes = []
    for name, expected in preserved.items():
        path = ROOT/name
        actual = {'symlink': str(path.readlink())} if path.is_symlink() else {'sha256': sha(path)}
        if actual != expected:
            check(name == 'src/trotterlib/README.md', 'OLD_PROTECTED_FILE:'+name)
            documentation_changes.append(name)
    for name, expected in read(FOUNDATION/'evidence_inventory.json')['files'].items():
        path = FOUNDATION/name
        actual = {'symlink': str(path.readlink())} if path.is_symlink() else {'sha256': sha(path)}
        check(actual == expected, 'OLD_FOUNDATION_EVIDENCE:'+name)

    data = PACK/'input_run/input_generation/data'
    receipt = read(data/'receipt.json')
    check(receipt['status'] == 'PASS' and receipt['all_gates_passed'] and not receipt['artificial_only'], 'MOLECULAR_INPUT')
    check((receipt['spatial_orbitals'], receipt['spin_orbitals'], receipt['physical_sector_dimension']) == (4, 8, 36), 'DIMENSION')
    check(all(g['PASS'] and g['error'] <= g['tolerance'] for g in receipt['gates'].values()), 'INPUT_GATES')
    check(receipt['SCF_calls'] == receipt['DF_calls'] == receipt['FCI_crosscheck_calls'] == 1, 'INPUT_CALLS')
    check(sha(data/'snapshot.npz') == receipt['snapshot_sha256'], 'INPUT_SNAPSHOT')
    check(sha(data/'pauli.json') == receipt['pauli_json_sha256'], 'INPUT_PAULI')
    binding = dict(SOURCE=source['SOURCE'], grant_id=grant['grant_id'], contract_digest=digest(plan),
        receipt_sha256=sha(data/'receipt.json'), snapshot_sha256=receipt['snapshot_sha256'])
    summary = read(BASE/'postprocessing_recovery01/recovered_case_summary.json')
    check(summary['status'] == 'FOUR_CASES_COMPLETE' and not summary['missing_cases'], 'COMPLETE_SUMMARY')
    check(len(summary['rows']) == 20, 'PRECISION_COVERAGE')
    trajectory_total = event_total = compile_total = 0
    all_ids = set(); costs = {}; orders = {}; errors = []; trajectory_errors = []; forced_errors = []
    precision_rows = []; all_calls = {}
    for case in CASES:
        folder = PACK/'run'/case
        status, result, validation, calls = [read(folder/f) for f in ['status.json', 'result.json', 'validation.json', 'actual_call_accounting.json']]
        check(status['status'] == 'COMPLETE_FIXED_CASE' and not status['artificial_only'], 'CASE:'+case)
        check(result['provenance']['source'] == read(folder/'marker.json')['binding'] == binding, 'CASE_BINDING:'+case)
        check(result['validation']['molecular_validation'] and validation['status'] == 'PASS', 'CASE_VALIDATION:'+case)
        check(validation['thresholds'] == plan['numerical'], 'THRESHOLDS:'+case)
        check(not validation['production_sampling_replayed'], 'VALIDATION_REPLAY')
        for row in validation['trajectory_operator_checks']:
            check(row['gate']['PASS'] and not row['replayed_sampling'] and not row['phase_alignment'], 'TRAJECTORY_GATE')
            trajectory_errors.append(row['gate']['error'])
        for row in validation['forced_order_checks']:
            check(row['gate']['PASS'] and not row['compile'] and not row['cost_sample'], 'FORCED_GATE')
            forced_errors.append(row['gate']['error'])
        expected = 256 if case.endswith('_PR') else 1
        samples = result['circuit_cost']['samples']
        check(len(samples) == result['circuit_cost']['sample_count'] == status['event_join_count'] == expected, 'FIXED_TRAJECTORIES')
        check(calls['paired_trajectories'] == expected and calls['wrapper_compile_requests'] == 2*expected, 'CALL_BUDGET')
        check(calls['new_compiles']+calls['cache_hits'] == 2*expected, 'CACHE_ACCOUNTING')
        check(calls['event_audit_additional_draws'] == calls['validation_replay_draws'] == 0 and not calls['automatic_retry'], 'EXTRA_DRAWS_OR_RETRY')
        check(len(list((folder/'trajectories').glob('*/events.json'))) == len(list((folder/'trajectories').glob('*/price_join.json'))) == expected, 'EXTRA_OR_MISSING_RECORD')
        histogram = Counter(); case_events = 0
        for index, sample in enumerate(samples):
            tid = sample['trajectory_id']
            check(tid not in all_ids, 'DUPLICATE_ID'); all_ids.add(tid)
            joined = read(folder/'trajectories'/tid/'price_join.json')
            trajectory_hash = joined.pop('trajectory_record_hash')
            check(trajectory_hash == sample['trajectory_record_hash'] == digest(joined), 'JOIN_HASH')
            event_record = read(folder/'trajectories'/tid/'events.json')
            event_hash = event_record.pop('event_record_hash')
            check(digest(event_record) == event_hash == joined['event_record_hash'] == sample['event_record_hash'], 'EVENT_HASH')
            check(all(joined[k] == v for k,v in event_record.items()), 'PRECOMPILE_EVENT_MATCH')
            check(joined['binding'] == binding and joined['case_id'] == case, 'EVENT_BINDING')
            check(joined['trajectory_index'] == index and joined['seed'] == sample['seed'], 'TRAJECTORY_SEED_INDEX')
            check(tid == digest(dict(case_id=case, binding=binding, index=index, seed=sample['seed'])), 'TRAJECTORY_ID')
            check(joined['compiled_RZ'] == sample['RZ'] and not joined['event_costs_additive'], 'PRICE_JOIN')
            check(not joined['resampling_for_audit'] and joined['production_returned_events'], 'AUDIT_REPLAY')
            expected_events = 32 if case.endswith('_PR') else 0
            check(joined['event_count'] == len(joined['events']) == expected_events, 'EVENT_COUNT')
            for i,event in enumerate(joined['events']):
                check((event['event_position'],event['outer_step_index'],event['inner_RTE_position']) == (i,i//4,i%4), 'EVENT_ORDER')
                check(event['distribution_id'] == joined['distribution_id'], 'EVENT_DISTRIBUTION')
                check(len(event['application_sequence']) == event['taylor_order']+1, 'APPLICATION_ORDER')
                histogram[event['taylor_order']] += 1
            case_events += expected_events
        check(calls['production_RTE_events'] == case_events, 'ACTUAL_EVENT_COUNT')
        precision = read(folder/'precision_rows.json')
        check([row['epsilon_sig'] for row in precision] == plan['precision']['epsilon_sig'], 'PRECISION_LIST')
        check(result['shots'] == precision[0]['shots'] and result['total_RZ'] == precision[0]['total_RZ'], 'PRIMARY_ACCOUNTING')
        for row in precision:
            check((row['total_RZ'] is None) == (row['shots']['N_total'] is None), 'NULL_N_G')
            match = next(r for r in summary['rows'] if r['case'] == case and r['epsilon_sig'] == row['epsilon_sig'])
            check(all(match[k] == v for k,v in row.items()), 'SUMMARY_ROW')
        precision_rows.extend(dict(case=case, **row) for row in precision)
        costs[case] = result['circuit_cost']['axes']
        orders[case] = dict(sorted(histogram.items()))
        all_calls[case] = calls
        errors.append(result['circuit_cost']['compiled_signal_max_error'])
        check(errors[-1] <= plan['numerical']['compiled_probability_agreement'], 'COMPILED_PROBABILITY')
        trajectory_total += expected; event_total += case_events; compile_total += 2*expected
    check((trajectory_total,event_total,compile_total) == (514,16384,1028), 'TOTAL_BUDGET')
    # The original input stage finished normally; its evidence still applies.
    stop = read(PACK/'input_run/STOP.json')
    check(stop['all_children_reaped'] and not stop['automatic_retry'] and not stop['automatic_next_stage'], 'INPUT_STOP')
    check(not stop.get('pending_owned_processes', []) and not stop.get('persistence_errors', []), 'INPUT_CLEANUP')
    # Native case supervisor completion cannot be reconstructed from saved results.
    terminal = read(BASE/'postprocessing_recovery01/terminal_observation.json')
    check(not terminal['observation']['worker']['active'], 'WORKER_STILL_RUNNING')
    check(terminal['all_children_reaped'] is None and terminal['worker_exit_code'] is None, 'NO_FABRICATED_EXIT_OR_REAP')
    return dict(schema='h4_pilot_saved_verification_v1',status='SAVED_DATA_VALIDATED_SUPERVISION_INCOMPLETE',original_native_completion_verified=False,SOURCE=source['SOURCE'],
        cases=4,precision_rows=20,null_N_G_rows=sum(r['total_RZ'] is None for r in precision_rows),
        trajectories=trajectory_total,production_events=event_total,wrapper_compile_requests=compile_total,
        maximum_compiled_probability_error=max(errors),maximum_trajectory_operator_error=max(trajectory_errors),
        maximum_forced_order_error=max(forced_errors),observed_event_orders=orders,
        event_orders_are_observed_counts_only=True,population_cost_guaranteed=False,
        compiled_cost_axes=costs,actual_calls=all_calls,input_gates=receipt['gates'],prefixes=receipt['prefixes'],
        old_preservation=dict(inspected=len(preserved),documentation_index_changes=documentation_changes,old_scientific_bytes_changed=0),
        source_files_checked=len(source['files']),runtime_files_checked=len(source['environment']['files']) if runtime_files else 0,
        runtime_seal_unmodified=True if runtime_files else None,all_children_reaped=None,
        original_worker_exit_code=None,original_native_STOP_present=(PACK/'STOP.json').exists(),
        original_stage_STOP_present=(PACK/'run/STOP.json').exists(),
        grant_consumed=True,grant_effectively_reusable=False,scientific_retry=False,additional_scientific_calls_for_verification=0,
        mandatory_STOP=True,detailed_scientific_review=False,push=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-files', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(runtime_files=args.runtime_files), ensure_ascii=False, indent=2, sort_keys=True)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(result)
    else:
        print(result, end='')
