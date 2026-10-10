"""Synthetic metadata/grant gates only; no saved molecule or numerical backend."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/resource_applicability/seal_track_a_ax2b_h4_limited_v1.py'
spec = importlib.util.spec_from_file_location('h4_seal_synthetic', SCRIPT)
seal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seal)


@pytest.fixture
def receipt():
    plan = seal.preparation('H4_LIMITED')['plan']
    return {'status': 'H4P_SAVED_RECEIPT_REAUDIT_PASS',
        'original_parent_status': 'H4_NATIVE_RECEIPT_STOP',
        'original_parent_reason': 'TERMINAL_INVALID:JSON_INPUT_SIZE',
        'original_parent_STOP_unchanged': True, 'science_manifest_sealed': False,
        'launch_authorized': False, 'H6_status': 'H6_NOT_AUTHORIZED',
        'contract_status': 'DRAFT_NOT_AUTHORIZATION', 'mandatory_stop': True,
        'next_stage_authorized': False, 'new_molecular_work': {'signal': 0, 'compile': 0},
        'environment': {'python': 'synthetic', 'packages': {}},
        'input_binding': {'path': 'fake.npz', 'sha256': '0' * 64, 'metadata': {'synthetic': True}},
        'registered_primitive_time_pairs': 179, 'registered_future_probe_actions': 537,
        'original_coverage_binding': {'sealed': True, 'actual_rank': 12,
            'schedule_digest': seal.digest(plan['cells']), 'expected_bounds': {
                'primitive_actions': 537, 'primitive_probe_count': 3,
                'structural_bounds_are_not_compiled_costs': True,
                'cells': [{'cell_id': c['id'], 'wrapper_instruction_upper_bounds':
                           {'ordinary': 1, 'symmetric_directional': 2}} for c in plan['cells']]}}}


def compose(receipt, **kwargs):
    options = dict(source_commit='1' * 40, hashes={'fake.py': '2' * 64},
        observed_environment=receipt['environment'], cpu=3, affinity=[3, 4],
        output_absolute='/tmp/synthetic-h4-output')
    options.update(kwargs)
    return seal.compose(receipt, **options)


def test_seal_keeps_scope_and_all_non_authorization_flags(receipt):
    before = copy.deepcopy(receipt)
    manifest = compose(receipt)
    assert manifest['plan'] == seal.preparation('H4_LIMITED')['plan']
    assert manifest['coverage_binding'] == receipt['original_coverage_binding']
    assert manifest['execution_plan_sealed'] is True
    assert manifest['science_authorized'] is manifest['launch_allowed'] is False
    assert manifest['status'] == 'H4_LIMITED_NOT_AUTHORIZED'
    assert manifest['H6_status'] == 'H6_NOT_AUTHORIZED'
    assert manifest['contract_status'] == 'DRAFT_NOT_AUTHORIZATION'
    assert manifest['plan']['N'] is manifest['plan']['G'] is None
    assert manifest['plan']['numerical_allowance_certified'] is False
    assert manifest['plan']['accuracy_eligibility'] == 'UNDETERMINED'
    assert manifest['assigned_resources'] == {'assigned_cpu': 3, 'science_workers': 1, 'blas_threads': 1}
    assert manifest['intended_exclusive_output']['repository_path'] == seal.FUTURE_OUTPUT
    assert receipt == before
    manifest['coverage_binding']['expected_bounds']['primitive_actions'] = -1
    assert receipt == before


@pytest.mark.parametrize(('key', 'value', 'reason'), [
    ('status', 'FAIL', 'REAUDIT_PASS_REQUIRED'),
    ('original_parent_status', 'H4_NATIVE_RECEIPT_COMPLETE', 'PRESERVE_ORIGINAL_STOP'),
    ('original_parent_reason', None, 'PRESERVE_ORIGINAL_STOP'),
    ('original_parent_STOP_unchanged', False, 'PRESERVE_ORIGINAL_STOP'),
    ('science_manifest_sealed', True, 'REAUDIT_STOP_FLAGS'),
    ('launch_authorized', True, 'REAUDIT_STOP_FLAGS'),
    ('H6_status', 'AUTHORIZED', 'REAUDIT_STOP_FLAGS'),
    ('contract_status', 'APPROVED', 'REAUDIT_STOP_FLAGS'),
    ('mandatory_stop', False, 'REAUDIT_STOP_FLAGS'),
    ('next_stage_authorized', True, 'REAUDIT_STOP_FLAGS'),
    ('new_molecular_work', {'compile': 1}, 'REAUDIT_NEW_SCIENCE'),
    ('new_molecular_work', {'compile': False}, 'REAUDIT_NEW_SCIENCE'),
    ('registered_primitive_time_pairs', 178, 'PROBE_COVERAGE'),
    ('registered_future_probe_actions', 536, 'PROBE_COVERAGE'),
])
def test_invalid_saved_flags_rejected(receipt, key, value, reason):
    receipt[key] = value
    with pytest.raises(ValueError, match=reason):
        compose(receipt)


@pytest.mark.parametrize(('key', 'value'), [('sealed', False), ('actual_rank', 13),
    ('actual_rank', True), ('schedule_digest', '0' * 64)])
def test_coverage_binding_rejected(receipt, key, value):
    receipt['original_coverage_binding'][key] = value
    with pytest.raises(ValueError, match='COVERAGE_BINDING'):
        compose(receipt)


def test_cell_reordering_rejected(receipt):
    receipt['original_coverage_binding']['expected_bounds']['cells'].reverse()
    with pytest.raises(ValueError, match='REGISTERED_CELL_SET'):
        compose(receipt)


@pytest.mark.parametrize('value', [-1, 1000001, True, 1.5])
def test_invalid_instruction_caps_rejected(receipt, value):
    receipt['original_coverage_binding']['expected_bounds']['cells'][0]['wrapper_instruction_upper_bounds']['ordinary'] = value
    with pytest.raises(ValueError, match='INSTRUCTION_CAP'):
        compose(receipt)


@pytest.mark.parametrize('cpu', [True, -1, 5, '3'])
def test_cpu_binding_rejected(receipt, cpu):
    with pytest.raises(ValueError, match='ASSIGNED_CPU_UNAVAILABLE'):
        compose(receipt, cpu=cpu)


def test_environment_mismatch_rejected(receipt):
    with pytest.raises(ValueError, match='ENVIRONMENT_CHANGED'):
        compose(receipt, observed_environment={'python': 'changed'})


def test_unapproved_sealed_manifest_stops_before_io_or_worker(receipt, tmp_path, monkeypatch):
    from trottertracks.resource_applicability import ax2b_bound_launch_v2 as gate
    def forbidden(*args, **kwargs):
        pytest.fail('source/input/environment/process I/O reached without a grant')
    for name in ('verify_sources', 'verify_input', 'environment'):
        monkeypatch.setattr(gate, name, forbidden)
    output = tmp_path / 'unused'
    assert seal.unauthorized_gate(compose(receipt), output)['launch_rejected'] is True
    for requested, grant in [(False, None), (True, {}), (True, {'approved_by_user': False})]:
        with pytest.raises(ValueError, match='SEPARATE_EXPLICIT_USER_GRANT_REQUIRED'):
            gate.validate_launch(tmp_path, compose(receipt), grant, requested=requested, output=output)
    assert not output.exists()


def test_changed_published_bytes_rejected(tmp_path, monkeypatch):
    (tmp_path / 'data.json').write_bytes(b'{"x":1}')
    monkeypatch.setattr(seal, 'committed_blob', lambda *args: b'{"x":2}')
    with pytest.raises(ValueError, match='PUBLISHED_BYTES_CHANGED'):
        seal.published(tmp_path, '1' * 40, 'data.json')


@pytest.mark.parametrize('flag', ['--execute', '--worker', '--authorization'])
def test_seal_cli_cannot_launch(flag):
    result = subprocess.run([sys.executable, str(SCRIPT), '--source-commit', '1' * 40,
        '--assigned-cpu', '3', flag], capture_output=True, text=True)
    assert result.returncode == 2 and 'unrecognized arguments' in result.stderr


def test_science_cli_requires_separate_pinned_grant(tmp_path):
    output = tmp_path / 'unused'
    result = subprocess.run([sys.executable,
        str(ROOT / 'scripts/resource_applicability/run_track_a_ax2b_bound_v2.py'),
        '--kind', 'H4_LIMITED', '--execute', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 2 and 'Separate pinned authorization' in result.stderr
    assert not output.exists()
