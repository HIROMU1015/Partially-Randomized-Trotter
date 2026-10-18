"""Audit completed D metadata and copy its results; never rerun cost accounting."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess

import h4_dp_contract as c

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    run, public = args.run.resolve(), args.artifacts.resolve()
    c.require(ROOT in public.parents, 'PROJECT_LOCAL_ARTIFACTS')
    spec = c.read(run / 'spec_SEALED_PRIVATE.json')
    grant = c.read(run / 'grant_PRIVATE.json')
    output = Path(spec['output'])
    terminal = c.read(output / 'terminal.json')
    witness = c.read(run / 'native_driver_exit.json')
    result = c.read(output / 'result.json')
    c.require(terminal['status'] == 'COMPLETE_MANDATORY_STOP'
              and terminal['native_workers_reaped'] is True, 'COMPLETE_NATIVE_WORKERS')
    c.require(witness['exit_code'] == 0 and witness['same_parent_reaped'] is True
              and witness['result_sha256'] == c.sha(output / 'result.json'), 'NATIVE_DRIVER_ACCEPTANCE')
    c.require(not (output / 'STOP.json').exists(), 'NO_CURRENT_STOP')
    c.require(spec['stage'] == 'D' and spec['SOURCE'] == grant['SOURCE'] == terminal['SOURCE'] == witness['SOURCE'],
              'STAGE_SOURCE_IDENTITY')
    c.require(c.digest(spec) == grant['spec_digest'], 'SEALED_SPEC_DIGEST')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT')
    c.validate_tasks(spec)
    for name, h in spec['source_hashes'].items():
        c.require(c.sha(ROOT / name) == h, 'SOURCE_FILE_CHANGED')
        raw = subprocess.check_output(['git', 'show', spec['SOURCE'] + ':' + name], cwd=ROOT)
        c.require(__import__('hashlib').sha256(raw).hexdigest() == h, 'GIT_SOURCE_BYTES')
    for row in spec['inputs']:
        c.require(c.sha(row['path']) == row['sha256'], 'INPUT_CHANGED')
    # The successful runner has already checked the complete runtime/provider
    # closure after all native work. This audit verifies that terminal evidence.
    preservation = c.read(ROOT / '.h4_dp_phase0_private/stop_preservation_manifest.json')
    for row in preservation:
        c.require(c.sha(row['path']) == row['sha256'] and c.sha(row['copy']) == row['sha256'],
                  'HISTORICAL_RECORD_CHANGED')
    directories = {p.name for p in (output / 'tasks').iterdir() if p.is_dir()}
    c.require(directories == {t['id'] for t in spec['tasks']}, 'EXACT_TASK_DIRECTORIES')
    compiler = c.read(spec['native_config'])['binding']['compiler']
    measured, controls, calls = [], [], Counter()
    for task in spec['tasks']:
        directory = output / 'tasks' / task['id']
        item = c.read(directory / 'result.json')
        ledger = c.read(directory / 'calls_ledger.json')
        control = c.read(directory / 'control.json')
        expected = dict(native_evolution=1, control_probe=4, wrapper_compile=2)
        c.require(item['cell_id'] == task['cell']['id'] and item['calls'] == ledger['reserved'] == expected,
                  'TASK_CALL_LEDGER')
        c.require(ledger['allowed'] == expected and ledger['nonrefundable'] is True, 'TASK_BUDGET')
        c.require(control['status'] == 'PASS' and control['control_probes'] == 4
                  and control['maximum_error'] < 1e-9, 'CONTROL_ACCEPTANCE')
        c.require(set(item['axes']) == set(c.AXES), 'PAIRED_AXES')
        for axis in c.AXES:
            row = c.read(directory / ('measured_' + axis + '.json'))
            c.require(row == item['axes'][axis] and row['axis'] == axis
                      and row['compiler_hash'] == compiler and len(row['fingerprint']) == 64,
                      'AXIS_METRIC_IDENTITY')
            c.require(all(type(row[k]) is int and row[k] >= 0 for k in
                          ['RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size']) and row['RZ'] > 0,
                      'INTEGER_NATIVE_METRICS')
        measured.append(dict(task_id=task['id'], cell=task['cell'], SOURCE=spec['SOURCE'],
                             axes=item['axes'], calls=item['calls']))
        controls.append(dict(cell_id=item['cell_id'], **control))
        calls.update(item['calls'])
    c.require(result['actual_calls'] == dict(calls, deterministic_accounting_pass=1), 'GLOBAL_CALL_LEDGER')
    c.require(result['domain'] == 104 and len(result['rows']) == 1040 and len(result['winners']) == 10,
              'REGISTERED_ENDPOINT_COVERAGE')
    c.require(len({(r['cell_id'], r['bias_mode'], r['epsilon']) for r in result['rows']}) == 1040,
              'UNIQUE_ENDPOINT_ROWS')
    c.require(len(result['excluded_ids']) == 33 and result['zero_fill'] is False
              and result['global_optimum_certified'] is False, 'EXCLUSION_AND_CLAIM_BOUNDARY')
    saved = c.read(spec['saved_measurements'])
    c.require(len(saved) == 16 and len(measured) == 55, 'MEASUREMENT_COVERAGE')
    known_pids = [p.name for p in (output / 'worker_contexts').iterdir() if p.is_dir()]
    workers_absent = all(not Path('/proc', pid).exists() for pid in known_pids)
    public.mkdir(parents=True, exist_ok=True)
    c.require(not (public / 'completion_audit.json').exists(), 'UNUSED_PUBLIC_AUDIT')
    # Copy the authoritative one-pass result; do not recompute G, shots, or winners.
    c.save(public / 'deterministic_result.json', result)
    c.save(public / 'new_native_measurements.json', measured)
    c.save(public / 'native_controls.json', controls)
    c.save(public / 'native_exit_projection.json', {k: v for k, v in witness.items() if k != 'pid'})
    resources = c.read(run / 'fresh_resources_PRIVATE.json')
    c.save(public / 'resource_projection.json', {k: v for k, v in resources.items()
           if k not in ['cgroup_observations']})
    registration_path = ROOT / 'artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/P_registration.json'
    registration = c.read(registration_path)
    pf_by_epsilon = {str(w['epsilon']): w for w in result['winners'] if w['bias_mode'] == 'USUAL_SIGNED'}
    assignments = []
    for item in registration['assignments']:
        winner = pf_by_epsilon[str(item['epsilon'])]
        c.require(winner['best'] is not None, 'P_COMPARATOR_STILL_PENDING')
        assignments.append(dict(item, PF_cell_id=winner['best']['cell_id'],
            PF_G_exact=winner['best']['G_exact'], PF_saved_shots=winner['best']['shots'],
            expected_direction=None, n=None))
    c.save(public / 'P_after_D_contract_DRAFT.json', dict(status='DRAFT_UNAUTHORIZED',
        D_result_sha256=c.sha(output / 'result.json'), D_SOURCE=spec['SOURCE'],
        original_registration_sha256=c.sha(registration_path), PF_status='FIXED_BY_PREREGISTERED_D_RULE',
        PF_by_epsilon=pf_by_epsilon, assignments=assignments,
        n_fixed={cid: None for cid in sorted({x['cell_id'] for x in assignments})},
        pending=['S_STATS_APPROVAL', 'NUMERICAL_N_AND_BASELINE_DIRECTION', 'P_FIXED_SOURCE_SEEDS_AND_COLLISION_CHECK',
                 'P_EXECUTION_APPROVAL'],
        n_calculated=False, baseline_direction_compared=False, new_seeds_generated=0,
        S_STATS_authorized=False, P_authorized=False))
    audit = dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
        result_sha256=c.sha(output / 'result.json'), sealed_spec_digest=c.digest(spec),
        user_approval_receipt_sha256=grant['approval_receipt_sha256'],
        native_driver_exit=0, same_parent_reaped=True, native_workers=len(known_pids),
        all_native_worker_exits_zero_checked_by_runner=True, known_worker_pids_absent_at_audit=workers_absent,
        registered_candidates=104, new_measured_candidates=55, reused_measured_candidates=16,
        excluded_by_saved_ineligibility=33, endpoint_rows=1040, endpoint_winners=10,
        actual_calls=result['actual_calls'], all_controls_passed=True,
        maximum_control_error=max(x['maximum_error'] for x in controls),
        final_runtime_provider_closure_checked_by_successful_runner=True,
        code_source_git_and_input_bytes_checked_at_audit=True,
        prior_records_and_copies_preserved=len(preservation),
        workflow_technical_recovery_authorized=spec['workflow_technical_recovery_authorized'],
        additional_technical_retries=0, scientific_accounting_recomputed_by_audit=False,
        P_started=False, S_STATS_started=False, remote_push=False,
        full_registered_domain_directly_measured=False, global_optimum_certified=False,
        next_stage_authorized=False)
    c.save(public / 'completion_audit.json', audit)
    c.save(run / 'completion_audit_PRIVATE.json', audit)
    progress = [__import__('json').loads(line) for line in (run / 'progress_observations.jsonl').read_text().splitlines()]
    c.save(public / 'parallel_execution_observations.json', progress)
    paths = [ROOT / name for name in spec['source_hashes']]
    paths += [run / name for name in ['user_approval_PRIVATE.json', 'fresh_resources_PRIVATE.json',
        'spec_SEALED_PRIVATE.json', 'grant_PRIVATE.json', 'fresh_gate_PRIVATE.json', 'ONE_SHOT.json',
        'native_driver_exit.json', 'driver_stdout.log', 'driver_stderr.log', 'resource_observations.jsonl',
        'progress_observations.jsonl', 'completion_audit_PRIVATE.json']]
    paths += [output / name for name in ['start.json', 'terminal.json', 'result.json']]
    paths += [output / 'tasks' / task['id'] / name for task in spec['tasks']
              for name in ['calls_ledger.json', 'control.json', 'measured_cosine.json',
                           'measured_sine.json', 'result.json']]
    store = run / 'preservation_store'
    store.mkdir(exist_ok=False)
    snapshots = []
    for path in paths:
        h = c.sha(path)
        copy = store / h
        if not copy.exists():
            copy.write_bytes(path.read_bytes())
        c.require(c.sha(copy) == h, 'SUCCESSFUL_RUN_SNAPSHOT')
        snapshots.append(dict(path=str(path), sha256=h, copy=str(copy)))
    c.save(run / 'successful_run_preservation_PRIVATE.json', snapshots)
    c.save(public / 'successful_run_preservation_projection.json', dict(records=len(snapshots),
        copies_verified=True, manifest_sha256=c.sha(run / 'successful_run_preservation_PRIVATE.json'),
        private_paths_published=False))
    print('D_ACCEPTED; 55 NEW CANDIDATES / 110 AXES / 1040 ENDPOINTS; MANDATORY_STOP')


if __name__ == '__main__':
    main()
