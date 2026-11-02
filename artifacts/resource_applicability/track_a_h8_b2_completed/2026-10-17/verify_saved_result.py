"""Verify published saved evidence only; never import the scientific runner."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    for row in read('publication_files_v1.json')['files']:
        path = Path(row['path'])
        assert not path.is_absolute() and '..' not in path.parts
        raw = (ROOT / path).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], str(path)
        if (ROOT / path).is_relative_to(HERE) and path.suffix in ('.json', '.csv', '.md'):
            assert b'/home/' not in raw, str(path)
    binding = read('b2_full_pass_binding_v1.json')
    receipts = read('execution_receipts_v1.json')
    audit = read('preservation_receipt_v1.json')
    source = read('source_input_binding_v1.json')
    assert binding['status'] == 'PASS_REGISTERED_FOUR_AXES'
    assert binding['registered_axes'] == 4 and binding['saved_axes_accepted'] == 3 and binding['new_axes_completed'] == 1
    assert not binding['Phase_C_authorized'] and not binding['noncancellation_stage_authorized']
    assert not binding['full_grid_guaranteed'] and not binding['rigorous_bias_guarantee']
    assert not binding['PR_population_mean_guarantee'] and not binding['final_total_cost_evaluation_performed']
    assert not binding['original_runs_retroactively_changed'] and not binding['single_original_attempt_completed']
    assert binding['old_STOPs_and_unsettled_reservations_preserved'] and binding['mandatory_STOP']
    expected = {('B0_p15_q1_4th', 'cosine'): 168353, ('B0_p15_q1_4th', 'sine'): 168353,
                ('B0I_p21_q8_4th', 'cosine'): 1852640, ('B0I_p21_q8_4th', 'sine'): 1852640}
    axes = binding['axes']
    assert len(axes) == 4 and {(x['case_id'], x['axis']): x['measured_RZ'] for x in axes} == expected
    for row in axes:
        check = row['model_check']
        # Check the previously saved axis comparison. No model fit or cost accounting.
        error = abs(Fraction(check['exact_prediction']) - row['measured_RZ']) / row['measured_RZ']
        assert error == Fraction(check['exact_relative_error']) < Fraction(1, 50)
        assert check['PASS'] and row['native_exit_code'] == 0 and row['reaped']
        assert row['fingerprint_before'] == row['fingerprint_after']
        assert row['model_sha256'] == binding['model_sha256']
        assert row['compiler_hash'] == binding['compiler_sha256']
        assert row['profile_sha256'] == binding['profile_sha256']
    terminal = receipts['raw_driver_terminal']
    assert axes == terminal['saved_B0_axes'] + [terminal['saved_cosine_axis']] + terminal['axes']
    assert terminal['SOURCE'] == source['SOURCE'] == binding['final_execution_SOURCE'] == audit['SOURCE']
    assert terminal['status'] == receipts['raw_launcher_terminal']['status'] == receipts['admission_terminal']['status'] == 'COMPLETE_MANDATORY_STOP'
    assert receipts['final_gate']['status'] == 'PASS_MANDATORY_STOP' and receipts['final_gate'] == binding['final_gate']
    assert receipts['raw_launcher_terminal']['native_exit_code'] == receipts['native_launcher_wait']['native_exit_code'] == 0
    assert receipts['native_launcher_wait']['wait_completed']
    assert receipts['raw_launcher_terminal']['all_reaped'] and not receipts['raw_launcher_terminal']['cleanup_errors']
    assert receipts['raw_launcher_terminal']['first_stop'] is None and receipts['final_resource']['reason'] is None
    assert receipts['sine_parallel_receipt']['all_reaped'] and receipts['sine_parallel_receipt']['status'] == 'PASS'
    assert terminal['actual_calls']['compile'] == terminal['actual_calls']['model_error_check'] == terminal['actual_calls']['qpy_load'] == 1
    assert all(value == 0 for key, value in terminal['actual_calls'].items()
               if key not in ('compile', 'model_error_check', 'qpy_load'))
    assert terminal['irrevocable_reservations'] == terminal['settlements']
    assert not terminal['next_stage_authorized'] and not terminal['fit_performed'] and not terminal['retry']
    assert len(source['source_hashes']) == 33 and source['provider_files'] == 265 and source['runtime_files'] == 34273
    assert len(source['input_receipts']) == 31
    for name, expected_sha in source['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_sha, name
    assert source['config_sha256'] == binding['private_config_sha256'] == audit['config_sha256']
    assert source['private_valid_grant_sha256'] == binding['private_grant_sha256'] == audit['grant_sha256']
    raw_files = audit['prepared_files']
    assert raw_files['output01/terminal.json']['sha256'] == binding['raw_terminal_sha256']
    assert raw_files['control01/B2_final_gate.json']['sha256'] == binding['raw_final_gate_sha256']
    assert audit['preservation']['immutable_files_unchanged'] == 702
    assert audit['preservation']['frozen_worktrees_unchanged'] == 135
    assert len(audit['preservation']['concurrent_worktree_observations']) == 1
    assert audit['prior_immutable_files_unchanged'] == 286
    assert audit['status'] == 'PASS_COMPLETED_RUN_METADATA_AND_BYTE_AUDIT'
    assert not audit['scientific_execution_started_by_audit'] and not audit['Phase_C_authorized']
    resource = read('resource_summary_v1.json')
    assert all(value == 0 for row in resource['OOM_event_deltas'].values() for value in row.values())
    assert resource['own_processes_reaped'] and not resource['shared_OS_cgroup_scheduler_changes']
    assert resource['max_monitor_gap_seconds'] <= 5
    assert read('technical_validation_v1.json')['test_methods'] == 26
    old = read('prior_STOP_preservation_v1.json')
    assert old['original_B2_run01_status'] == old['original_B0_run02_status'] == 'STOP'
    assert old['most_recent_technical_STOP']['driver_terminal']['status'] == 'STOP'
    assert old['refunded_or_erased_reservations'] == 0 and not old['incomplete_axes_used_as_measured']
    with (HERE / 'registered_axis_measurements_v1.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and {(r['case_id'], r['axis']): int(r['measured_RZ']) for r in rows} == expected
    manifest = json.loads((ROOT / 'artifacts/validation_manifest.json').read_text())['track_a_h8_b2_completed_results_20261017']
    assert manifest['native_launcher_exit_code'] == 0 and not manifest['Phase_C_authorized']
    print(json.dumps(dict(status='PASS_SAVED_B2_FOUR_AXES_AND_COMPLETION_RECEIPTS',
                         registered_axes=4, new_scientific_execution=0, cost_reaccount=0,
                         Phase_C_authorized=False, mandatory_STOP=True)))


if __name__ == '__main__':
    main()
