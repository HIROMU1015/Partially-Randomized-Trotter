"""Metadata-only preparation: reject dangerous input without molecular work."""
import ast
import builtins
import json
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/resource_applicability/prepare_track_a_ax2b_h4_prelaunch_v3.py'


@pytest.fixture
def metadata(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion'}:
            raise AssertionError('Numerical import forbidden in metadata tests: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    return runpy.run_path(str(SCRIPT), run_name='metadata_fixture')


def write_npz(path, layout, *, bad=None, duplicate=False, extra=False):
    with zipfile.ZipFile(path, 'w') as archive:
        for key, (dtype, shape) in layout.items():
            header = {'descr': dtype, 'shape': shape, 'fortran_order': False}
            if key == 'one_body' and bad == 'object':
                header['descr'] = '|O'
            if key == 'one_body' and bad == 'shape':
                header['shape'] = (8, 7)
            text = (repr(header) + '\n').encode()
            count = 1
            for dim in shape:
                count *= dim
            version = b'\x09\x00' if bad == 'version' and key == 'one_body' else b'\x01\x00'
            payload = bytes(count * int(dtype[2:]))
            if bad == 'length' and key == 'one_body':
                payload += b'!'
            archive.writestr(key + '.npy', b'\x93NUMPY' + version + len(text).to_bytes(2, 'little') + text + payload)
        archive.writestr('metadata_json.npy', b'not-read-by-header-inspector')
        if duplicate:
            with pytest.warns(UserWarning, match='Duplicate name'):
                archive.writestr('constant.npy', b'!')
        if extra:
            archive.writestr('unexpected.npy', b'!')


def test_header_only_fixture_passes_without_numerical_imports(metadata, tmp_path):
    path = tmp_path / 'fixture.npz'
    write_npz(path, metadata['LAYOUT'])
    result = metadata['layout_headers'](path)
    assert len(result) == 7
    assert all(r['payload_decoded'] is False for r in result.values())


@pytest.mark.parametrize('bad', ['object', 'shape', 'version', 'length'])
def test_bad_headers_rejected_before_array_load(metadata, tmp_path, bad):
    path = tmp_path / 'fixture.npz'
    write_npz(path, metadata['LAYOUT'], bad=bad)
    with pytest.raises(ValueError, match='NPY_'):
        metadata['layout_headers'](path)


@pytest.mark.parametrize('argument', ['duplicate', 'extra'])
def test_archive_keys_fail_closed(metadata, tmp_path, argument):
    path = tmp_path / 'fixture.npz'
    write_npz(path, metadata['LAYOUT'], **{argument: True})
    with pytest.raises(ValueError, match='NPZ_KEYS_OR_DUPLICATES'):
        metadata['layout_headers'](path)


def test_static_times_equal_frozen_runtime_function_including_E(metadata):
    # Evaluate only the pure schedule function, without importing its numerical module.
    source = ROOT / 'src/trottertracks/resource_applicability/ax2b_molecular_ports_v2.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validation_times')
    namespace = {'primitive_time_schedule': metadata['primitive_time_schedule']}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    cells = metadata['scope_plan']('H4_LIMITED')['cells']
    coverage = metadata['static_coverage'](cells, .8)
    for raw, row in zip(cells, coverage['cells'], strict=True):
        cell = dict(raw, formula=raw['order'], r=raw['R'] // raw['q'] if raw['R'] else None)
        assert row['registered_validation_times_v2'] == namespace['validation_times'](cell, T=.8)
    assert coverage['unique_primitive_time_count'] == 179
    assert coverage['primitive_actions'] == 537
    b2 = next(r for r in coverage['cells'] if r['cell_id'] == 'H4_B2_K2')
    assert (0, -.05) in b2['registered_validation_times_v2']
    assert b2['native_tail_matvecs'] == 48
    assert coverage['expected_bounds_for_launcher'] is None
    assert not coverage['actual_instruction_bounds_sealed']


def test_missing_user_grant_precedes_any_io(metadata, monkeypatch):
    from trottertracks.resource_applicability import ax2b_bound_launch_v2 as launch
    def deny(*args, **kwargs):
        raise AssertionError('No input/source access before an explicit grant.')
    monkeypatch.setattr(launch, 'verify_sources', deny)
    monkeypatch.setattr(launch, 'verify_input', deny)
    with pytest.raises(ValueError, match='SEPARATE_EXPLICIT_USER_GRANT_REQUIRED'):
        launch.validate_launch(ROOT, metadata['preparation']('H4_LIMITED'), None,
                               requested=True, output='/unused')


def test_unsealed_manifest_is_rejected_even_with_mock_grant(metadata):
    from trottertracks.resource_applicability import ax2b_bound_launch_v2 as launch
    manifest = metadata['preparation']('H4_LIMITED')
    manifest['input_binding'] = {'path': 'fixture-only', 'sha256': '0' * 64, 'metadata': {}}
    grant = {'approved_by_user': True, 'schema': 'track_a_ax2b_bound_authorization_v2',
             'kind': 'H4_LIMITED', 'manifest_digest': metadata['digest'](manifest)}
    with pytest.raises(ValueError, match='SEALED_PREPARATION_REQUIRED_NOT_AUTHORIZATION'):
        launch.validate_launch(ROOT, manifest, grant, requested=True, output='/unused')


def test_CLI_execute_absent_and_existing_output_protected(tmp_path):
    output = tmp_path / 'protected.json'
    output.write_text(json.dumps({'saved': 'unchanged'}))
    before = output.read_bytes()
    for arguments in (['--execute'], []):
        completed = subprocess.run([sys.executable, '-S', str(SCRIPT), '--output', str(output), *arguments],
                                   capture_output=True, text=True)
        assert completed.returncode == 2
        assert output.read_bytes() == before
    absent = tmp_path / 'not-created.json'
    result = subprocess.run([sys.executable, '-S', str(SCRIPT), '--output', str(absent), '--execute'],
                            capture_output=True, text=True)
    assert result.returncode == 2 and not absent.exists()


def test_preparation_contains_no_assignment_authorization_or_science_seal(metadata):
    manifest = metadata['preparation']('H4_LIMITED')
    assert manifest['science_authorized'] is False and manifest['launch_allowed'] is False
    assert manifest['assigned_resources'] is None and manifest['execution_plan_sealed'] is False
    assert manifest['plan']['caps_proposed']['compile'] == 0
    assert manifest['plan']['caps_proposed']['trajectory'] == 0
    assert manifest['mandatory_stop'] is True
