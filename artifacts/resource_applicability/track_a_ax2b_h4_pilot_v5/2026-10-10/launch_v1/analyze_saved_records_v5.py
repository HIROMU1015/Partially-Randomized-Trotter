"""Audit saved JSON and hashes only; stdlib, no scientific libraries or reruns."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def require(value, message):
    if not value:
        raise ValueError(message)


def main(root, old_run, prior_v4_run, output):
    run = output.parent / 'run_v1'
    terminal = load(run / 'terminal_status.json')
    manifest = load(run / 'frozen_preparation.json')
    plan = manifest['plan']
    for name, expected in manifest['source_hashes'].items():
        require(sha(root / name) == expected, 'changed_source:' + name)
    require(sha(root / manifest['snapshot']['path']) == manifest['snapshot']['sha256'], 'changed_input')
    expected_cells = {c['id']: c for c in plan['correctness_cells']}
    cells = {p.stem.removesuffix('_correctness'): load(p) for p in run.glob('H4_*_correctness.json')}
    require(set(cells) == set(expected_cells), 'correctness_file_set')
    same_cells = []
    for name, row in cells.items():
        require(row['cell'] == expected_cells[name], 'cell_identity:' + name)
        require(row['accuracy_eligibility'] == 'UNDETERMINED' and row['numerical_allowance_certified'] is False, 'claim_scope')
        filename = name + '_correctness.json'
        require(sha(run / filename) == sha(old_run / filename), 'v3_correctness_bytes:' + name)
        same_cells.append(filename)
    expected_tasks = {t['id']: t for t in plan['wrapper_tasks']}
    costs = {p.stem.removesuffix('_cost'): load(p) for p in run.glob('H4_*_cost.json')}
    require(set(costs) <= set(expected_tasks), 'unexpected_cost')
    metric_names = ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')
    same_costs, groups, compiler_hashes = [], {}, set()
    for name, row in costs.items():
        require(row['task'] == expected_tasks[name], 'task_identity:' + name)
        require(row['winner_claim'] is False and row['shot_estimate_performed'] is False, 'cost_scope')
        require(all(type(row['metrics'][m]) is int and row['metrics'][m] >= 0 for m in metric_names), 'metric_type')
        require(row['pretranspile_instructions'] <= plan['caps']['untranspiled_instructions'] and row['transpiled_instructions'] <= plan['caps']['transpiled_instructions'], 'instruction_cap')
        require(math.isfinite(row['classical_compile_wall_seconds']) and row['classical_compile_wall_seconds'] >= 0, 'compile_time')
        compiler_hashes.add(row['compiler_settings_hash'])
        task = row['task']
        trajectory = load(run / f"{task['cell']['id']}_rep{task['replica']}_trajectory.json")
        require(row['event_digest'] == trajectory['event_digest'], 'event_pair')
        prior_path = old_run / (name + '_cost.json')
        if prior_path.exists():
            prior = load(prior_path)
            keys = ('task', 'event_digest', 'metrics', 'wrapper_fingerprint', 'compiler_settings_hash', 'pretranspile_instructions', 'transpiled_instructions', 'quantum_scope')
            require(all(row[k] == prior[k] for k in keys), 'v3_cost_identity:' + name)
            same_costs.append(name)
        key = (task['cell']['id'], task['control_policy'], task['axis'])
        groups.setdefault(key, []).append(row)
    require(len(compiler_hashes) == 1, 'compiler_identity')
    trajectory_rows, same_trajectories = [], []
    for p in sorted(run.glob('H4_*_trajectory.json')):
        a = load(p)
        require(a['comparison_max_state_error'] <= plan['gates']['agreement_tolerance'], 'control_gate')
        old = old_run / p.name
        if old.exists():
            require(sha(p) == sha(old), 'v3_trajectory_bytes:' + p.name)
            same_trajectories.append(p.name)
        trajectory_rows.append(a)
    diagnostic_paths = sorted(run.glob('diagnostic_*.json'))
    require([p.name for p in diagnostic_paths] == [f'diagnostic_{i:04d}.json' for i in range(len(diagnostic_paths))], 'diagnostic_sequence')
    require(len(diagnostic_paths) <= plan['implementation']['maximum_diagnostic_records'], 'diagnostic_cap')
    ds = [load(p) for p in diagnostic_paths]
    stack, durations, last_elapsed = [], {}, 0
    for d in ds:
        require(math.isfinite(d['elapsed_seconds']) and d['elapsed_seconds'] >= last_elapsed, 'diagnostic_time')
        last_elapsed = d['elapsed_seconds']
        require(all(type(v) is int and v >= 0 for k, v in d['memory'].items() if k.endswith('_bytes')), 'diagnostic_memory')
        if d['kind'] == 'begin':
            stack.append(d)
        elif d['kind'] == 'end':
            require(stack and stack[-1]['context'] == d['context'], 'diagnostic_nesting')
            start = stack.pop()
            stage = d['context']['stage']
            durations.setdefault(stage, []).append(d['elapsed_seconds'] - start['elapsed_seconds'])
        else:
            require(d['kind'] == 'group_released' and not stack, 'release_event')
    stats = []
    for (cell, policy, axis), rows in sorted(groups.items()):
        stats.append({'cell': cell, 'control_policy': policy, 'axis': axis, 'n': len(rows), 'metrics': {m: {'mean': statistics.mean(r['metrics'][m] for r in rows), 'min': min(r['metrics'][m] for r in rows), 'max': max(r['metrics'][m] for r in rows), 'sample_sd': statistics.stdev(r['metrics'][m] for r in rows) if len(rows) > 1 else None} for m in metric_names}})
    stages = {s: {'completed_intervals': len(v), 'inclusive_total_seconds': sum(v), 'max_seconds': max(v)} for s, v in durations.items()}
    raw = sorted(p for p in run.iterdir() if p.is_file())
    require(sum(p.stat().st_size for p in raw) <= plan['caps']['output_bytes'], 'output_cap')
    reference = load(run / 'input_reference.json')
    primitive = load(run / 'primitive_lowering.json')
    report = {'schema': 'track_a_ax2b_h4_v5_saved_record_analysis_v1', 'raw_runner_status': terminal['status'], 'reason': terminal['reason'], 'parent_wall_seconds': terminal['wall_seconds'], 'worker_terminal_present': (run / 'worker_terminal.json').exists(), 'failure_diagnostics_present': (run / 'failure_diagnostics.json').exists(), 'scope': 'saved JSON consistency and v3 byte/identity comparison; no independent science recomputation', 'source_hashes_verified': len(manifest['source_hashes']), 'input_unchanged': True, 'correctness_cells_saved': len(cells), 'correctness_v3_byte_identical': sorted(same_cells), 'wrappers_saved': len(costs), 'planned_wrappers': len(expected_tasks), 'uncompleted_tasks': sorted(set(expected_tasks) - set(costs)), 'common_wrapper_identity_and_metrics_equal_v3': sorted(same_costs), 'trajectory_records_saved': len(trajectory_rows), 'trajectory_v3_byte_identical': sorted(same_trajectories), 'random_trajectory_records_saved': sum(r['seed'] is not None for r in trajectory_rows), 'random_calls_if_no_worker_terminal': 'UNAVAILABLE; partial event records are not terminal call counters', 'diagnostic_count': len(ds), 'groups_released': sum(d['kind'] == 'group_released' for d in ds), 'last_diagnostic': ds[-1], 'open_intervals_at_termination': [{'context': d['context'], 'begin_elapsed_seconds': d['elapsed_seconds']} for d in stack], 'completed_stage_intervals': stages, 'timing_scope': 'inclusive nested intervals overlap; incomplete intervals excluded; not pure transpilation timing', 'recorded_max_memory_bytes': {k: max(d['memory'].get(k, 0) for d in ds) for k in ('VmRSS_bytes','VmSize_bytes','VmHWM_bytes','cumulative_peak_rss_bytes')}, 'memory_scope': 'recorded snapshots/cumulative maxima over executed partial scope; not full 28-wrapper peak or failure-allocation peak', 'technical_cost_groups': stats, 'compiler_settings_hash': next(iter(compiler_hashes)), 'quantum_resource_scope': 'measured full Hadamard wrapper without state preparation; no shots/total-cost or population ranking', 'compile_and_fingerprint_record_wall_seconds_sum': sum(r['classical_compile_wall_seconds'] for r in costs.values()), 'reference_diagnostics': {k: reference[k] for k in ('sector_all_columns_error','expm_eigh_state_discrepancy','exact_phase_surrogate_allowance')}, 'primitive_max_state_error': primitive['max_state_error'], 'raw_file_count': len(raw), 'raw_output_bytes': sum(p.stat().st_size for p in raw), 'raw_files': [{'path': str(p.relative_to(root)), 'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in raw], 'accuracy_eligibility': 'UNDETERMINED', 'total_numerical_allowance_certified': False, 'immutable_CI': False, 'science_attempts': 1, 'retry': False, 'resume': False, 'H6_H8_GPU_executed': False, 'mandatory_stop': True, 'next_stage_authorized': False}
    v4_cells, v4_costs, v4_trajectories = [], [], []
    for filename in same_cells:
        require(sha(run / filename) == sha(prior_v4_run / filename), 'v4_correctness_bytes:' + filename)
        v4_cells.append(filename)
    for name, row in costs.items():
        path = prior_v4_run / (name + '_cost.json')
        if path.exists():
            prior = load(path)
            keys = ('task', 'event_digest', 'metrics', 'wrapper_fingerprint', 'compiler_settings_hash', 'pretranspile_instructions', 'transpiled_instructions', 'quantum_scope')
            require(all(row[k] == prior[k] for k in keys), 'v4_cost_identity:' + name)
            v4_costs.append(name)
    for path in sorted(run.glob('H4_*_trajectory.json')):
        old_path = prior_v4_run / path.name
        if old_path.exists():
            require(sha(path) == sha(old_path), 'v4_trajectory_bytes:' + path.name)
            v4_trajectories.append(path.name)
    worker = load(run / 'worker_terminal.json') if (run / 'worker_terminal.json').exists() else None
    report.update(correctness_v4_byte_identical=sorted(v4_cells),
                  common_wrapper_identity_and_metrics_equal_v4=sorted(v4_costs),
                  trajectory_v4_byte_identical=sorted(v4_trajectories),
                  worker_calls=worker['calls'] if worker else None,
                  worker_complete=worker is not None and worker['status'] == 'H4_TECHNICAL_PILOT_COMPLETE')
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: report[k] for k in ('raw_runner_status','reason','correctness_cells_saved','wrappers_saved','diagnostic_count','groups_released','recorded_max_memory_bytes')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--old-run', required=True, type=Path)
    parser.add_argument('--prior-v4-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), args.old_run.resolve(), args.prior_v4_run.resolve(), args.output.resolve())
