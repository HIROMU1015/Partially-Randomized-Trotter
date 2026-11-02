#!/usr/bin/env python3
"""Verify saved H4 pilot JSON only; no arrays, signals, sampling or compile."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.resource_applicability.ax2b_h4_contract_v4 import digest, file_hash, source_hashes, h4_plan
from trottertracks.resource_applicability.ax2b_limits import exclusive_json

METRICS = ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')


def read_json(path):
    if Path(path).stat().st_size > 4 * 1024 * 1024:
        raise ValueError('OVERSIZE_JSON')
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def check(condition, name):
    if not condition:
        raise ValueError('SAVED_EVIDENCE_MISMATCH:' + name)


def number(value, name):
    check(type(value) in (int, float) and math.isfinite(value), name)
    return value


def complex_value(value):
    return complex(number(value['real'], 'complex_real'), number(value['imag'], 'complex_imag'))


def verify(root, run):
    root, run = Path(root), Path(run)
    terminal = read_json(run / 'terminal_status.json')
    manifest = read_json(run / 'frozen_preparation.json')
    auth = read_json(run / 'authorization.json')
    plan = manifest['plan']
    check(plan == h4_plan(), 'registered_plan')
    check(source_hashes(root) == manifest['source_hashes'], 'frozen_source')
    check(file_hash(root / manifest['snapshot']['path']) == manifest['snapshot']['sha256'], 'snapshot_bytes')
    check(auth.get('approved_by_user') is True and auth['manifest_digest'] == digest(manifest), 'authorization')
    check(auth['exclusive_output'] == str(run.resolve()), 'output_binding')
    marker = read_json(run / 'launch_binding.json')
    check(marker == {'manifest_digest': digest(manifest), 'authorization_digest': digest(auth)}, 'parent_binding')
    claim = read_json(run / 'worker_claim.json')
    check(claim['assigned_cpu'] == auth['assigned_cpu'] and claim['manifest_digest'] == digest(manifest)
          and claim['authorization_digest'] == digest(auth) and claim['resumption_allowed'] is False, 'child_claim')
    check(terminal['mandatory_stop'] is True and terminal['next_stage_authorized'] is False, 'terminal_stop')
    complete = terminal['status'] == 'H4_TECHNICAL_PILOT_COMPLETE'
    if not all((run / name).exists() for name in ('worker_terminal.json', 'input_reference.json', 'primitive_lowering.json')):
        check(not complete, 'incomplete_reference_claimed_complete')
        return {'status': 'SAVED_PARTIAL_STOP_PRESERVED', 'terminal': terminal,
                'manifest_digest': digest(manifest), 'source_and_snapshot_unchanged': True,
                'complete_pilot_verified': False, 'mandatory_stop': True}
    if complete:
        check(terminal['worker_exit_code'] == 0 and terminal['reason'] is None, 'worker_success')
    worker = read_json(run / 'worker_terminal.json')
    check(worker == terminal['worker_terminal'], 'worker_terminal')
    count = worker['diagnostic_records_written']
    check(type(count) is int and 0 <= count <= plan['implementation']['maximum_diagnostic_records'], 'diagnostic_count')
    diagnostics = {p.name for p in run.glob('diagnostic_*.json')}
    check(diagnostics == {f'diagnostic_{i:04d}.json' for i in range(count)}, 'diagnostic_file_set')
    for name in sorted(diagnostics):
        record = read_json(run / name)
        check(record['kind'] in ('begin', 'end', 'group_released') and isinstance(record['context'], dict), 'diagnostic_kind')
        check(number(record['elapsed_seconds'], 'diagnostic_elapsed') >= 0, 'diagnostic_elapsed')
        check(all(type(value) is int and value >= 0 for key, value in record['memory'].items() if key.endswith('_bytes')),
              'diagnostic_memory')
    failure = worker['failure_diagnostics']
    if failure is not None and 'artifact' in failure:
        check(failure['artifact'] == 'failure_diagnostics.json', 'failure_artifact_name')
        detail = read_json(run / 'failure_diagnostics.json')
        check(detail['context'] == failure['context'] and detail['locals_recorded'] is False, 'failure_context')
        check(detail['traceback_frame_limit'] == 32 and len(detail['traceback_frames']) <= 32, 'failure_trace_bound')
        check(failure['last_frame'] == (detail['traceback_frames'][-1] if detail['traceback_frames'] else None), 'failure_last_frame')
    if complete:
        check(failure is None and not (run / 'failure_diagnostics.json').exists(), 'completed_with_failure')
    if complete:
        check(worker['status'] == 'H4_TECHNICAL_PILOT_COMPLETE'
              and worker['completed_correctness_cells'] == 8 and worker['compiled_wrappers'] == 28, 'completed_counts')
    check(worker['accuracy_eligibility'] == 'UNDETERMINED' and worker['numerical_allowance_certified'] is False,
          'no_accuracy_certificate')
    caps, gates = plan['caps'], plan['gates']
    check(terminal['wall_seconds'] <= caps['total_wall_seconds'] + terminal['watchdog_poll_seconds'] + 1,
          'wall_cap')
    check(sum(p.stat().st_size for p in run.iterdir() if p.is_file()) <= caps['output_bytes'], 'disk_cap')
    reference = read_json(run / 'input_reference.json')
    check(reference['metadata'] == manifest['snapshot']['metadata'], 'reference_metadata')
    check(reference['total_numerical_allowance_certified'] is False and reference['eligibility_status'] == 'UNDETERMINED',
          'reference_scope')
    for key, tolerance in (('sector_all_columns_error', gates['agreement_tolerance']),
                           ('expm_eigh_state_discrepancy', gates['reference_discrepancy_tolerance']),
                           ('exact_phase_surrogate_allowance', gates['agreement_tolerance'])):
        check(0 <= number(reference[key], key) <= tolerance, key)
    primitive = read_json(run / 'primitive_lowering.json')
    check(0 <= number(primitive['max_state_error'], 'primitive_error') <= gates['agreement_tolerance'], 'primitive_gate')
    reference_signal = complex_value(reference['reference_signal'])
    expected_cells = {cell['id']: cell for cell in plan['correctness_cells']}
    present_cells = {p.name.removesuffix('_correctness.json') for p in run.glob('H4_*_correctness.json')}
    check(present_cells <= expected_cells.keys() and len(present_cells) == worker['completed_correctness_cells'], 'cell_file_set')
    if complete: check(present_cells == expected_cells.keys(), 'cell_file_set')
    cell_records = []
    for cell_id, cell in expected_cells.items():
        if cell_id not in present_cells: continue
        record = read_json(run / (cell_id + '_correctness.json'))
        check(record['cell'] == cell and complex_value(record['reference']) == reference_signal, 'cell_identity')
        signal = complex_value(record['signal'])
        total_error = complex_value(record['signed_total_error'])
        check(abs(signal - reference_signal - total_error) <= 1e-12, 'signed_total_error')
        check(record['accuracy_eligibility'] == 'UNDETERMINED' and record['numerical_allowance_certified'] is False, 'cell_scope')
        counts = record['action_counts']
        check(0 <= counts['tail'] <= caps['tail_matvecs_per_signal'] and
              0 <= counts['deterministic'] <= caps['deterministic_actions_per_signal'], 'signal_action_caps')
        if cell['method'] == 'B0':
            exact = complex_value(record['exact_truncated_signal'])
            check(abs(exact - reference_signal - complex_value(record['signed_discard_error'])) <= 1e-12, 'discard_component')
            check(abs(signal - exact - complex_value(record['signed_PF_error'])) <= 1e-12, 'PF_component')
        elif cell['method'] in ('B2', 'B3'):
            raw = complex_value(record['raw_signal'])
            B = number(record['B'], 'B')
            check(B >= 1 and abs(math.log(B) - record['log_B']) <= 1e-12, 'log_normalization')
            check(abs(B * raw - signal) <= 1e-9, 'corrected_raw')
            pf = complex_value(record['pf_exact_tail_signal'])
            check(abs(signal - pf - complex_value(record['signed_finite_error'])) <= 1e-12, 'finite_component')
            check(abs(pf - reference_signal - complex_value(record['signed_outer_pf_error'])) <= 1e-12, 'outer_PF_component')
            check(counts['tail'] == 2 * cell['R'] * (cell['K'] + 1), 'finite_matvec_count')
        cell_records.append(record)
    expected_tasks = {task['id']: task for task in plan['wrapper_tasks']}
    present_tasks = {p.name.removesuffix('_cost.json') for p in run.glob('H4_*_cost.json')}
    check(present_tasks <= expected_tasks.keys() and len(present_tasks) == worker['compiled_wrappers'], 'wrapper_file_set')
    if complete: check(present_tasks == expected_tasks.keys(), 'wrapper_file_set')
    if present_tasks: check(len(present_cells) == 8, 'all_correctness_before_cost')
    groups, trajectories, compiler_hashes = {}, {}, set()
    for task_id, task in expected_tasks.items():
        if task_id not in present_tasks: continue
        record = read_json(run / (task_id + '_cost.json'))
        check(record['task'] == task, 'wrapper_task_identity')
        check(record['winner_claim'] is False and record['shot_estimate_performed'] is False, 'wrapper_claim_scope')
        for name in METRICS:
            check(type(record['metrics'][name]) is int and record['metrics'][name] >= 0, 'compiled_metric')
        check(record['pretranspile_instructions'] <= caps['untranspiled_instructions'] and
              record['transpiled_instructions'] <= caps['transpiled_instructions'], 'instruction_caps')
        check(number(record['classical_compile_wall_seconds'], 'compile_wall') >= 0, 'compile_time')
        compiler_hashes.add(record['compiler_settings_hash'])
        key = (task['cell']['id'], task['replica'])
        if key not in trajectories:
            trajectory = read_json(run / f'{key[0]}_rep{key[1]}_trajectory.json')
            check(trajectory['cell'] == task['cell'] and trajectory['replica'] == task['replica']
                  and trajectory['seed'] == task['trajectory_seed'], 'trajectory_identity')
            check(trajectory['event_digest'] == digest(trajectory['events_by_outer_step']), 'explicit_event_digest')
            check(0 <= number(trajectory['comparison_max_state_error'], 'control_error') <= gates['agreement_tolerance'], 'control_gate')
            if task['trajectory_seed'] is not None:
                check(len(trajectory['events_by_outer_step']) == task['cell']['q'], 'outer_occurrences')
                check(all(len(events) == task['cell']['R'] // task['cell']['q'] for events in trajectory['events_by_outer_step']), 'RTE_event_count')
            trajectories[key] = trajectory
        check(record['event_digest'] == trajectories[key]['event_digest'], 'paired_event_reuse')
        group = (task['cell']['id'], task['control_policy'], task['axis'])
        groups.setdefault(group, []).append(record)
    check(len(compiler_hashes) == (1 if present_tasks else 0), 'compiler_count')
    if complete: check(len(trajectories) == 7, 'trajectory_counts')
    used_trajectory_files = {f'{key[0]}_rep{key[1]}_trajectory.json' for key in trajectories}
    present_trajectory_files = {p.name for p in run.glob('H4_*_trajectory.json')}
    if complete: check(present_trajectory_files == used_trajectory_files, 'trajectory_file_set')
    else: check(used_trajectory_files <= present_trajectory_files, 'trajectory_file_set')
    summary_path = run / 'wrapper_cost_summary.json'
    if complete: check(summary_path.exists(), 'summary_missing')
    summary = read_json(summary_path) if summary_path.exists() else {'groups': []}
    if summary_path.exists():
        check(summary['compiled_wrappers'] == len(present_tasks) and len(summary['groups']) == len(groups), 'summary_count')
        check(len({(g['cell_id'], g['control_policy'], g['axis']) for g in summary['groups']}) == len(groups), 'summary_unique')
    for group in summary['groups']:
        rows = groups[(group['cell_id'], group['control_policy'], group['axis'])]
        check(group['n'] == len(rows), 'summary_n')
        for name in METRICS:
            values = [row['metrics'][name] for row in rows]
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else None
            measured = group['metrics'][name]
            check(measured['mean'] == mean and measured['min'] == min(values) and measured['max'] == max(values), 'summary_arithmetic')
            check((measured['sample_sd'] is None) if sd is None else abs(measured['sample_sd'] - sd) <= 1e-9, 'summary_SD')
    calls = worker['calls']
    check(worker['compiled_wrappers'] <= calls['compile'] <= 28 and calls['trajectory'] <= 4 and calls['occurrence'] <= 16,
          'sample_compile_counts')
    if complete: check(calls['compile'] == 28 and calls['trajectory'] == 4 and calls['occurrence'] == 16, 'sample_compile_counts')
    for key, cap in (('primitive', 'primitive_validation_actions'), ('control_probe', 'control_probe_actions'),
                     ('reference_matvecs', 'reference_matvecs_total')):
        check(0 <= calls[key] <= caps[cap], 'call_cap:' + key)
    checked = sorted(p for p in run.iterdir() if p.is_file())
    return {'status': 'H4_SAVED_PILOT_EVIDENCE_VERIFIED' if complete else 'H4_SAVED_PARTIAL_PILOT_EVIDENCE_VERIFIED',
            'complete_pilot_verified': complete, 'terminal': terminal,
            'manifest_digest': digest(manifest), 'source_hash_count_verified': len(manifest['source_hashes']),
            'correctness_cells_verified': len(present_cells), 'compiled_wrappers_verified': len(present_tasks),
            'uncompleted_wrapper_task_ids': sorted(expected_tasks.keys() - present_tasks),
            'trajectory_records_verified': len(trajectories), 'random_trajectory_samples': calls['trajectory'],
            'compiler_settings_hash': next(iter(compiler_hashes)) if compiler_hashes else None,
            'source_and_snapshot_unchanged': True,
            'files': [{'path': str(p.relative_to(run)), 'sha256': file_hash(p)} for p in checked],
            'scope': 'saved-record consistency, no independent numerical recomputation',
            'total_numerical_allowance_certified': False, 'accuracy_eligibility': 'UNDETERMINED',
            'immutable_CI': False, 'mandatory_stop': True, 'next_stage_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(ROOT, args.run)
    exclusive_json(args.output, report)
    print(report['status'])


if __name__ == '__main__':
    main()
