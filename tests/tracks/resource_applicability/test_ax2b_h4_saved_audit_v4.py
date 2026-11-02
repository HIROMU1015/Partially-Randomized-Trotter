"""Small fabricated saved records only; no scientific files/libraries."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
from trottertracks.resource_applicability.ax2b_h4_contract_v4 import h4_plan, digest, file_hash

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('ax2b_saved_audit_v4', ROOT / 'scripts/resource_applicability/verify_track_a_ax2b_h4_pilot_v4.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(tmp_path, monkeypatch):
    run = tmp_path / 'fabricated_run'
    run.mkdir()
    (tmp_path / 'toy_saved.bin').write_bytes(b'not scientific data')
    plan = h4_plan()
    manifest = {'plan': plan, 'source_hashes': {'toy.py': 'synthetic'},
                'snapshot': {'path': 'toy_saved.bin', 'sha256': file_hash(tmp_path / 'toy_saved.bin'), 'metadata': {'toy': True}}}
    auth = {'approved_by_user': True, 'manifest_digest': digest(manifest), 'assigned_cpu': 3,
            'exclusive_output': str(run.resolve())}
    monkeypatch.setattr(audit, 'source_hashes', lambda root: manifest['source_hashes'])
    def write(name, value): (run / name).write_text(json.dumps(value, allow_nan=False))
    write('frozen_preparation.json', manifest)
    write('authorization.json', auth)
    write('launch_binding.json', {'manifest_digest': digest(manifest), 'authorization_digest': digest(auth)})
    write('worker_claim.json', {'assigned_cpu': 3, 'manifest_digest': digest(manifest),
                              'authorization_digest': digest(auth), 'resumption_allowed': False})
    worker = {'status': 'H4_TECHNICAL_PILOT_COMPLETE', 'completed_correctness_cells': 8, 'compiled_wrappers': 28,
              'accuracy_eligibility': 'UNDETERMINED', 'numerical_allowance_certified': False,
              'diagnostic_records_written': 0, 'failure_diagnostics': None,
              'calls': {'compile': 28, 'trajectory': 4, 'occurrence': 16, 'primitive': 962,
                        'control_probe': 108, 'reference_matvecs': 3621}}
    write('worker_terminal.json', worker)
    write('terminal_status.json', {'status': worker['status'], 'worker_terminal': worker, 'worker_exit_code': 0,
          'reason': None, 'mandatory_stop': True, 'next_stage_authorized': False, 'wall_seconds': 1,
          'watchdog_poll_seconds': .05})
    cr = lambda value: {'real': value, 'imag': 0.0}
    write('input_reference.json', {'metadata': {'toy': True}, 'total_numerical_allowance_certified': False,
          'eligibility_status': 'UNDETERMINED', 'sector_all_columns_error': 0,
          'expm_eigh_state_discrepancy': 0, 'exact_phase_surrogate_allowance': 0, 'reference_signal': cr(1)})
    write('primitive_lowering.json', {'max_state_error': 0})
    for cell in plan['correctness_cells']:
        record = {'cell': cell, 'reference': cr(1), 'signal': cr(1), 'signed_total_error': cr(0),
                  'accuracy_eligibility': 'UNDETERMINED', 'numerical_allowance_certified': False,
                  'action_counts': {'tail': 0, 'deterministic': 0}}
        if cell['method'] == 'B0':
            record.update(exact_truncated_signal=cr(.9), signed_discard_error=cr(-.1), signed_PF_error=cr(.1))
        if cell['method'] in ('B2', 'B3'):
            record.update(raw_signal=cr(.5), B=2, log_B=math.log(2), pf_exact_tail_signal=cr(.9),
                          signed_finite_error=cr(.1), signed_outer_pf_error=cr(-.1))
            record['action_counts']['tail'] = 2 * cell['R'] * (cell['K'] + 1)
        write(cell['id'] + '_correctness.json', record)
    groups = {}
    for task in plan['wrapper_tasks']:
        cell = task['cell']
        key = f"{cell['id']}_rep{task['replica']}"
        events = [[{'synthetic': True}] * (cell['R'] // cell['q']) for _ in range(cell['q'])] if task['trajectory_seed'] is not None else None
        write(key + '_trajectory.json', {'cell': cell, 'replica': task['replica'], 'seed': task['trajectory_seed'],
              'events_by_outer_step': events, 'event_digest': digest(events), 'comparison_max_state_error': 0})
        write(task['id'] + '_cost.json', {'task': task, 'winner_claim': False, 'shot_estimate_performed': False,
              'metrics': dict.fromkeys(audit.METRICS, 2), 'compiler_settings_hash': 'synthetic compiler',
              'pretranspile_instructions': 2, 'transpiled_instructions': 2,
              'classical_compile_wall_seconds': 0, 'event_digest': digest(events)})
        groups.setdefault((cell['id'], task['control_policy'], task['axis']), []).append(task)
    write('wrapper_cost_summary.json', {'compiled_wrappers': 28, 'groups': [
          {'cell_id': key[0], 'control_policy': key[1], 'axis': key[2], 'n': len(tasks),
           'metrics': {name: {'mean': 2.0, 'min': 2, 'max': 2, 'sample_sd': 0.0 if len(tasks) > 1 else None}
                       for name in audit.METRICS}} for key, tasks in groups.items()]})
    return run


def mutate(run, name, change):
    value = json.loads((run / name).read_text())
    change(value)
    (run / name).write_text(json.dumps(value, allow_nan=False))


def test_fabricated_complete_saved_bundle(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    result = audit.verify(tmp_path, run)
    assert result['complete_pilot_verified'] is True
    assert result['compiled_wrappers_verified'] == 28 and result['trajectory_records_verified'] == 7
    assert result['total_numerical_allowance_certified'] is False
    assert result['accuracy_eligibility'] == 'UNDETERMINED' and result['next_stage_authorized'] is False


@pytest.mark.parametrize('name,change', [
    ('H4_B0_q4_correctness.json', lambda row: row['signed_discard_error'].update(real=0)),
    ('H4_B2_K2_correctness.json', lambda row: row.update(B=3)),
    ('H4_B3_K6_correctness.json', lambda row: row['action_counts'].update(tail=111)),
    ('H4_B2_K2_rep0_ordinary_cosine_cost.json', lambda row: row.update(event_digest='wrong')),
    ('H4_B1_S4_q1_rep0_ordinary_cosine_cost.json', lambda row: row.update(transpiled_instructions=5000001)),
    ('H4_B1_S4_q4_rep0_ordinary_cosine_cost.json', lambda row: row.update(winner_claim=True)),
    ('H4_B2_K2_rep0_trajectory.json', lambda row: row.update(seed=0)),
    ('wrapper_cost_summary.json', lambda row: row['groups'][0]['metrics']['rz_count'].update(mean=3)),
    ('input_reference.json', lambda row: row.update(expm_eigh_state_discrepancy=1e-3)),
    ('worker_claim.json', lambda row: row.update(assigned_cpu=4)),
])
def test_mutations_are_rejected(tmp_path, monkeypatch, name, change):
    run = fixture(tmp_path, monkeypatch)
    mutate(run, name, change)
    with pytest.raises(ValueError, match='SAVED_EVIDENCE_MISMATCH'):
        audit.verify(tmp_path, run)


def test_missing_wrapper_is_not_complete(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    (run / 'H4_B0_q4_rep0_ordinary_cosine_cost.json').unlink()
    with pytest.raises(ValueError, match='wrapper_file_set'):
        audit.verify(tmp_path, run)


def test_unregistered_wrapper_is_not_silently_ignored(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    (run / 'H4_EXTRA_cost.json').write_text('{}')
    with pytest.raises(ValueError, match='wrapper_file_set'):
        audit.verify(tmp_path, run)


def test_stopped_partial_records_preserved_without_promoting_complete(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    mutate(run, 'terminal_status.json', lambda row: row.update(status='H4_TECHNICAL_PILOT_STOP'))
    result = audit.verify(tmp_path, run)
    assert result['status'] == 'H4_SAVED_PARTIAL_PILOT_EVIDENCE_VERIFIED'
    assert result['complete_pilot_verified'] is False


def test_actual_failure_shape_verifies_only_24_records(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    for path in run.glob('H4_B1_S4_q4*_cost.json'): path.unlink()
    (run / 'H4_B1_S4_q4_rep0_trajectory.json').unlink()
    (run / 'wrapper_cost_summary.json').unlink()
    worker = json.loads((run / 'worker_terminal.json').read_text())
    worker.update(status='H4_TECHNICAL_PILOT_STOP', compiled_wrappers=24, reason='MemoryError:')
    worker['calls'].update(compile=24, control_probe=96)
    (run / 'worker_terminal.json').write_text(json.dumps(worker))
    mutate(run, 'terminal_status.json', lambda row: row.update(status='H4_TECHNICAL_PILOT_STOP',
            worker_terminal=worker, worker_exit_code=1, reason='WORKER_FAILED_OR_INCOMPLETE'))
    result = audit.verify(tmp_path, run)
    assert result['compiled_wrappers_verified'] == 24 and result['correctness_cells_verified'] == 8
    assert len(result['uncompleted_wrapper_task_ids']) == 4
    assert result['complete_pilot_verified'] is False and result['mandatory_stop'] is True


def test_diagnostic_count_and_file_set_are_checked(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    worker = json.loads((run / 'worker_terminal.json').read_text())
    worker['diagnostic_records_written'] = 1
    (run / 'worker_terminal.json').write_text(json.dumps(worker))
    mutate(run, 'terminal_status.json', lambda row: row.update(worker_terminal=worker))
    with pytest.raises(ValueError, match='diagnostic_file_set'):
        audit.verify(tmp_path, run)


def test_complete_bundle_cannot_also_claim_failure(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    worker = json.loads((run / 'worker_terminal.json').read_text())
    worker['failure_diagnostics'] = {'diagnostic_error': 'fabricated'}
    (run / 'worker_terminal.json').write_text(json.dumps(worker))
    mutate(run, 'terminal_status.json', lambda row: row.update(worker_terminal=worker))
    with pytest.raises(ValueError, match='completed_with_failure'):
        audit.verify(tmp_path, run)
