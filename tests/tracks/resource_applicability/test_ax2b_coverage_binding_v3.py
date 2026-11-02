"""Metadata/mock-only regression tests; never load arrays or build circuits."""
import ast
import builtins
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from trottertracks.resource_applicability import ax2b_coverage_binding_v3 as coverage
from trottertracks.resource_applicability import ax2b_bound_launch_v3 as launch
from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
from trottertracks.resource_applicability.ax2a_preparation import digest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/sealed_preparation_manifest_v1.json'
NUMERICAL = {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion', 'pyscf'}


@pytest.fixture(autouse=True)
def forbid_science(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in NUMERICAL:
            raise AssertionError('NUMERICAL_IMPORT_FORBIDDEN')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    original_open = Path.open
    def open_metadata(path, *args, **kwargs):
        if str(path).endswith('.npz'):
            raise AssertionError('SNAPSHOT_PAYLOAD_FORBIDDEN')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_metadata)


def isolated(path, name, namespace, *, method=False):
    tree = ast.parse(path.read_text())
    nodes = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MolecularPort') if method else tree.body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def registered_metadata():
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest['coverage_binding']['expected_bounds']
    actual = copy.deepcopy(expected)
    canonical = isolated(ROOT / 'src/trottertracks/resource_applicability/ax2b_stage_validation_v2.py', 'canonical_cell', {})
    times = isolated(ROOT / 'src/trottertracks/resource_applicability/ax2b_molecular_ports_v2.py', 'validation_times',
                     {'primitive_time_schedule': primitive_time_schedule})
    for cell, row in zip(manifest['plan']['cells'], actual['cells'], strict=True):
        canonicalized = canonical(cell)
        row['schedule'] = primitive_time_schedule(canonicalized, T=manifest['plan']['T'])
        row['schedule']['registered_validation_times_v2'] = times(canonicalized, T=manifest['plan']['T'])
    return manifest, expected, actual


class Writer:
    def __init__(self):
        self.records = {}
    def write(self, name, value, **kwargs):
        assert name not in self.records and kwargs['diagnostic'] is True
        self.records[name] = json.loads(json.dumps(value, allow_nan=False))


@pytest.mark.parametrize('index', range(8))
def test_registered_cell_real_schedule_json_round_trip(index):
    _, expected, actual = registered_metadata()
    assert expected['cells'][index] != actual['cells'][index]
    assert coverage.canonical_coverage(expected['cells'][index]) == coverage.canonical_coverage(actual['cells'][index])
    assert json.loads(coverage.canonical_coverage(actual['cells'][index])) == expected['cells'][index]
    # Wrapper bounds are retained saved metadata; this is not native validation.
    assert actual['cells'][index]['wrapper_instruction_upper_bounds'] == expected['cells'][index]['wrapper_instruction_upper_bounds']


def test_registered_full_coverage_round_trip():
    _, expected, actual = registered_metadata()
    assert actual != expected and digest(actual) == digest(expected)
    writer = Writer()
    receipt = coverage.assert_coverage_binding(expected, actual, writer)
    assert receipt['equal'] is True and receipt['differences'] == []
    assert writer.records['actual_coverage.json'] == expected
    assert expected['primitive_actions'] == 537 and expected['primitive_probe_count'] == 3


@pytest.mark.parametrize('mutation', ['time', 'index', 'order', 'missing_time', 'duplicate_time',
                                      'missing_cell', 'extra_cell', 'bound', 'count', 'bool_count', 'float_count'])
def test_value_and_coverage_changes_still_reject_and_save(mutation):
    _, expected, actual = registered_metadata()
    seq = actual['cells'][0]['schedule']['ordinary_one_outer_step']
    if mutation == 'time':
        seq[0] = (seq[0][0], math.nextafter(seq[0][1], math.inf))
    elif mutation == 'index':
        seq[0] = (seq[0][0] + 1, seq[0][1])
    elif mutation == 'order':
        seq[0], seq[1] = seq[1], seq[0]
    elif mutation == 'missing_time':
        seq.pop()
    elif mutation == 'duplicate_time':
        seq.append(seq[0])
    elif mutation == 'missing_cell':
        actual['cells'].pop()
    elif mutation == 'extra_cell':
        actual['cells'].append(copy.deepcopy(actual['cells'][0]))
    elif mutation == 'bound':
        actual['cells'][0]['wrapper_instruction_upper_bounds']['ordinary'] += 1
    elif mutation == 'count':
        actual['primitive_actions'] += 1
    elif mutation == 'bool_count':
        actual['primitive_probe_count'] = True
    elif mutation == 'float_count':
        actual['primitive_probe_count'] = 3.0
    writer = Writer()
    with pytest.raises(ValueError, match='ACTUAL_COVERAGE_CHANGED'):
        coverage.assert_coverage_binding(expected, actual, writer)
    receipt = writer.records['coverage_comparison_v3.json']
    assert receipt['equal'] is False and receipt['differences']
    assert receipt['actual_sha256'] != receipt['expected_sha256']
    assert 'actual_coverage.json' in writer.records


@pytest.mark.parametrize('left,right', [(0.0, -0.0), (1, 1.0), (1, True), (False, 0), (None, 'null')])
def test_scalar_identity_not_python_loose_equality(left, right):
    assert coverage.canonical_coverage({'x': left}) != coverage.canonical_coverage({'x': right})


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, 1j, {1: 2}, {1, 2}])
def test_invalid_coverage_rejects_without_writer(value):
    writer = Writer()
    with pytest.raises(ValueError, match='COVERAGE_'):
        coverage.assert_coverage_binding({'x': 0}, {'x': value}, writer)
    assert not writer.records


def test_bounded_caps_and_cycle(monkeypatch):
    recursive = []; recursive.append(recursive)
    with pytest.raises(ValueError, match='COVERAGE_CYCLE'):
        coverage.canonical_coverage(recursive)
    monkeypatch.setattr(coverage, 'MAX_NODES', 3)
    with pytest.raises(ValueError, match='COVERAGE_STRUCTURE_CAP'):
        coverage.canonical_coverage([1, 2, 3])
    monkeypatch.setattr(coverage, 'MAX_NODES', 200_000)
    monkeypatch.setattr(coverage, 'MAX_DEPTH', 2)
    with pytest.raises(ValueError, match='COVERAGE_STRUCTURE_CAP'):
        coverage.canonical_coverage([[[[0]]]])
    monkeypatch.setattr(coverage, 'MAX_BYTES', 8)
    with pytest.raises(ValueError, match='COVERAGE_BYTE_CAP'):
        coverage.canonical_coverage('123456789')
    with pytest.raises(ValueError, match='COVERAGE_BYTE_CAP'):
        coverage.canonical_coverage(['abc', 'def'])


def test_difference_receipt_is_bounded():
    writer = Writer()
    with pytest.raises(ValueError, match='ACTUAL_COVERAGE_CHANGED'):
        coverage.assert_coverage_binding(list(range(100)), list(range(1, 101)), writer)
    receipt = writer.records['coverage_comparison_v3.json']
    assert len(receipt['differences']) == coverage.MAX_DIFFERENCES
    assert receipt['differences_truncated'] is True
    assert len(json.dumps(receipt).encode()) < 8192


@pytest.mark.parametrize('mismatch', [False, True])
def test_setup_actual_serialization_boundary_before_reference(mismatch):
    manifest, expected, actual = registered_metadata()
    manifest['coverage_binding']['expected_bounds'] = expected
    if mismatch:
        actual['primitive_actions'] += 1
    writer, events = Writer(), []
    class Vector:
        def copy(self): return self
        def __truediv__(self, divisor): return self
        def __itruediv__(self, divisor): return self
    vector = Vector()
    def reference(*args, **kwargs):
        events.append('reference')
        raise RuntimeError('SYNTHETIC_REFERENCE_BOUNDARY_REACHED')
    namespace = {'verify_input': lambda *args: 12, 'safe_path': lambda root, name: Path(name),
        '_load_snapshot_once': lambda path: (NS(n_qubits=8, n_blocks=12), NS(dimension=36), vector, vector, {}, {}),
        'primitive_sector_certificate': lambda *args: {}, 'checked_basis_bridge': lambda *args: ([], vector),
        '_prepare': lambda *args: NS(), '_prepare_discard': lambda *args: NS(),
        'np': NS(linalg=NS(norm=lambda value: 1.0)), 'actual_bounds': lambda *args, **kwargs: actual,
        'check_bounds': lambda *args: None, 'assert_coverage_binding': coverage.assert_coverage_binding,
        'df_linear_operator': reference}
    setup = isolated(ROOT / 'src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py', 'setup', namespace, method=True)
    port = NS(phase=lambda name: events.append(name), root=ROOT, manifest=manifest,
              kind='H4_LIMITED', cells=manifest['plan']['cells'], plan=manifest['plan'], caps={}, writer=writer)
    with pytest.raises((ValueError, RuntimeError), match='ACTUAL_COVERAGE_CHANGED' if mismatch else 'SYNTHETIC_REFERENCE_BOUNDARY_REACHED'):
        setup(port)
    assert ('reference' in events) is (not mismatch)
    assert writer.records['coverage_comparison_v3.json']['equal'] is (not mismatch)
    assert 'actual_coverage.json' in writer.records


def fixture_launch(tmp_path, monkeypatch):
    manifest = launch.preparation()
    manifest.update(source_commit='1' * 40, source_hashes={}, environment={}, input_binding={},
                    execution_plan_sealed=True, assigned_resources={'assigned_cpu': 3, 'science_workers': 1, 'blas_threads': 1},
                    intended_exclusive_output={'repository_path': 'future', 'absolute_path': str(tmp_path / 'future')},
                    coverage_binding={'sealed': True, 'actual_rank': 12, 'expected_bounds': {},
                                      'schedule_digest': digest(manifest['plan']['cells'])})
    auth = {'schema': 'track_a_ax2b_bound_authorization_v3', 'approved_by_user': True,
            'manifest_digest': digest(manifest), 'kind': 'H4_LIMITED', 'assigned_cpu': 3,
            'exclusive_output': str(tmp_path / 'future'), 'retry': False, 'resume': False}
    monkeypatch.setattr(launch.os, 'sched_getaffinity', lambda pid: {3})
    monkeypatch.setattr(launch, 'verify_sources', lambda *args: None)
    monkeypatch.setattr(launch, 'verify_input', lambda *args: 12)
    monkeypatch.setattr(launch, 'environment', lambda: {})
    return manifest, auth


@pytest.mark.parametrize('mutation', [None, 'old_schema', 'H6', 'intended_output', 'retry', 'resume'])
def test_new_grant_scope_and_output_binding(tmp_path, monkeypatch, mutation):
    manifest, auth = fixture_launch(tmp_path, monkeypatch)
    if mutation is None:
        assert launch.validate_launch(tmp_path, manifest, auth, requested=True, output=tmp_path / 'future') == 3
    else:
        if mutation == 'old_schema': auth['schema'] = 'track_a_ax2b_bound_authorization_v2'
        elif mutation == 'H6': manifest['kind'] = auth['kind'] = 'H6_TECHNICAL'
        elif mutation == 'intended_output': manifest['intended_exclusive_output']['absolute_path'] = str(tmp_path / 'other')
        else: auth[mutation] = True
        auth['manifest_digest'] = digest(manifest)
        with pytest.raises(ValueError):
            launch.validate_launch(tmp_path, manifest, auth, requested=True, output=tmp_path / 'future')
    assert not (tmp_path / 'future').exists()


def test_absent_grant_before_any_metadata_access(tmp_path, monkeypatch):
    def forbidden(*args): raise AssertionError('EARLY_GATE_BROKEN')
    for name in ('verify_sources', 'verify_input', 'environment'):
        monkeypatch.setattr(launch, name, forbidden)
    with pytest.raises(ValueError, match='SEPARATE_EXPLICIT_USER_GRANT_REQUIRED'):
        launch.validate_launch(tmp_path, launch.preparation(), None, requested=True, output=tmp_path / 'future')


@pytest.mark.parametrize('args', [[], ['--execute'], ['--kind', 'H6_TECHNICAL']])
def test_new_cli_has_no_implicit_science_or_H6(tmp_path, args):
    path = tmp_path / 'out'
    result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/resource_applicability/run_track_a_ax2b_bound_v3.py'),
                             '--output', str(path), *args], capture_output=True, text=True)
    if not args:
        assert result.returncode == 0
        record = json.loads(path.read_text())
        assert record['science_authorized'] is record['launch_allowed'] is False
    else:
        assert result.returncode != 0 and not path.exists()


@pytest.mark.parametrize('flag', ['--execute', '--worker', '--authorization'])
def test_reseal_cli_cannot_execute_or_grant(flag):
    result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/resource_applicability/seal_track_a_ax2b_h4_limited_v2.py'),
                             '--source-commit', '1' * 40, '--assigned-cpu', '3', flag], capture_output=True, text=True)
    assert result.returncode == 2 and 'unrecognized arguments' in result.stderr


def test_reseal_retains_full_saved_coverage_and_science_plan():
    script = ROOT / 'scripts/resource_applicability/seal_track_a_ax2b_h4_limited_v2.py'
    spec = importlib.util.spec_from_file_location('h4_coverage_reseal_fixture', script)
    seal = importlib.util.module_from_spec(spec); spec.loader.exec_module(seal)
    manifest, expected, _ = registered_metadata()
    before = copy.deepcopy(manifest)
    receipt = seal.schedule_round_trip(ROOT, manifest)
    assert manifest == before and manifest['coverage_binding']['expected_bounds'] == expected
    assert receipt['strict_canonical_equality'] is True and receipt['actual_native_bounds_recomputed'] is False
    assert coverage.canonical_coverage(manifest['plan']) == coverage.canonical_coverage(launch.preparation()['plan'])
