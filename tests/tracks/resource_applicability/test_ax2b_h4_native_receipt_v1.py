"""Engineering tests only: injected metadata/native ports and dummy children.

No molecular NPZ payload, native orbital construction, signal or circuit calls.
"""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from trottertracks.resource_applicability import ax2b_h4_native_receipt_v1 as native
from trottertracks.resource_applicability import ax2b_h4_native_watchdog_v1 as watchdog
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h6_controller import BoundedWriter
from trottertracks.resource_applicability.ax2b_limits import exclusive_json

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def static():
    # Saved JSON arithmetic only; never open the molecular input.
    return json.loads((ROOT/native.STATIC).read_text())


def sealed(monkeypatch, root):
    manifest = native.draft()
    manifest.update(execution_plan_sealed=True, source_commit='0'*40,
        source_hashes={}, environment={'synthetic': True}, input_binding={'synthetic': True},
        static_binding={'synthetic': True}, assigned_resources={'assigned_cpu': 3, 'science_workers': 1, 'blas_threads': 1})
    monkeypatch.setattr(native.os, 'sched_getaffinity', lambda pid: {3})
    monkeypatch.setattr(native, 'verify_sources', lambda *a: None)
    monkeypatch.setattr(native, 'verify_input', lambda *a: 12)
    monkeypatch.setattr(native, 'environment', lambda: {'synthetic': True})
    monkeypatch.setattr(native, 'static_receipt', lambda *a: {})
    output = root/native.OUTPUT
    auth = {'schema': 'track_a_ax2b_h4_native_authorization_v1', 'kind': native.KIND,
        'approved_by_user': True, 'manifest_digest': digest(manifest), 'assigned_cpu': 3,
        'exclusive_output': str(output), 'retry': False, 'resume': False}
    return manifest, auth, output


class FakePort:
    synthetic_only = True

    def __init__(self, static, fail=None, mutate=None):
        self.static, self.fail, self.mutate = static, fail, mutate
        self.loads, self.prepares = 0, []

    def load(self):
        self.loads += 1
        if self.fail == 'load':
            raise MemoryError('synthetic load failure')
        return object()

    def prepare(self, ham, cell):
        self.prepares.append(cell['id'])
        if self.fail == cell['id']:
            raise MemoryError('synthetic preparation failure')
        old = next(row for row in self.static['static_coverage']['cells'] if row['cell_id'] == cell['id'])
        schedule = deepcopy(old['schedule'])
        schedule['registered_validation_times_v2'] = deepcopy(old['registered_validation_times_v2'])
        record = {'cell_id': cell['id'], 'deterministic_blocks': [
            {'primitive_id': 'one' if i == 0 else str(i-1), 'n_qubits': 8, 'order_index': i}
            for i in range(cell['prefix']+1)],
            'bounds_row': {'cell_id': cell['id'], 'schedule': schedule,
                'wrapper_instruction_upper_bounds': {'ordinary': 100, 'symmetric_directional': 200}}}
        if self.mutate:
            self.mutate(record)
        return record


def manifest_stub():
    return {'source_commit': '0'*40, 'input_binding': {'synthetic': True}, 'environment': {'synthetic': True}}


def run_fake(tmp_path, static, **options):
    port = FakePort(static, **options)
    writer = BoundedWriter(tmp_path, byte_cap=2**20)
    terminal = native.run_receipt(port, writer, manifest_stub(), static)
    return port, terminal


def test_draft_never_authorizes():
    value = native.draft()
    assert not value['execution_plan_sealed'] and not value['launch_allowed']
    assert value['status'] == 'H4_NATIVE_RECEIPT_NOT_AUTHORIZED'
    assert value['plan']['caps']['total_wall_seconds'] == 900
    assert value['plan']['caps']['address_space_bytes'] == 8*2**30
    assert value['plan']['caps']['output_bytes'] == 16*2**20
    assert value['plan']['science_manifest_sealed'] is False
    assert all(count == 0 for count in value['plan']['forbidden_calls'].values())


@pytest.mark.parametrize('auth,requested', [(None, True), ({}, True), ({'approved_by_user': False}, True),
                                           ({'approved_by_user': True}, False)])
def test_no_grant_before_any_input_or_source_io(monkeypatch, tmp_path, auth, requested):
    def forbidden(*a):
        pytest.fail('I/O must follow explicit user grant')
    for name in ('verify_sources', 'verify_input', 'environment', 'static_receipt'):
        monkeypatch.setattr(native, name, forbidden)
    with pytest.raises(ValueError, match='SEPARATE_EXPLICIT_H4P'):
        native.validate_launch(tmp_path, native.draft(), auth, requested=requested, output=tmp_path/'out')


@pytest.mark.parametrize('kind,schema', [('H4_LIMITED', 'track_a_ax2b_bound_authorization_v2'),
    ('H6_TECHNICAL', 'track_a_ax2b_bound_authorization_v2'), ('H4_NATIVE_RECEIPT', 'wrong')])
def test_science_and_h6_grants_cannot_authorize_h4p(monkeypatch, tmp_path, kind, schema):
    manifest, auth, output = sealed(monkeypatch, tmp_path)
    auth.update(kind=kind, schema=schema)
    with pytest.raises(ValueError, match='H4P_AUTHORIZATION_SCHEMA'):
        native.validate_launch(tmp_path, manifest, auth, requested=True, output=output)


@pytest.mark.parametrize('mutation', ['unsealed', 'science', 'cell', 'cap', 'cpu', 'resume', 'output', 'digest'])
def test_fixed_manifest_gates(monkeypatch, tmp_path, mutation):
    manifest, auth, output = sealed(monkeypatch, tmp_path)
    if mutation == 'unsealed': manifest['execution_plan_sealed'] = False
    if mutation == 'science': manifest['science_authorized'] = True
    if mutation == 'cell': manifest['plan']['cells'].pop()
    if mutation == 'cap': manifest['plan']['caps']['native_preparation_calls'] = 9
    if mutation == 'cpu': auth['assigned_cpu'] = 4
    if mutation == 'resume': auth['resume'] = True
    if mutation == 'output': output = tmp_path/'wrong'
    if mutation != 'digest': auth['manifest_digest'] = digest(manifest)
    else: auth['manifest_digest'] = 'f'*64
    with pytest.raises(ValueError):
        native.validate_launch(tmp_path, manifest, auth, requested=True, output=output)


def test_existing_output_and_one_shot_claim(monkeypatch, tmp_path):
    manifest, auth, output = sealed(monkeypatch, tmp_path)
    assert native.validate_launch(tmp_path, manifest, auth, requested=True, output=output) == 3
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match='EXCLUSIVE_OUTPUT'):
        native.validate_launch(tmp_path, manifest, auth, requested=True, output=output)
    exclusive_json(output/'launch_binding.json', {'manifest_digest': digest(manifest), 'authorization_digest': digest(auth)})
    assert native.validate_launch(tmp_path, manifest, auth, requested=True, output=output, worker=True) == 3
    exclusive_json(output/'worker_claim.json', {'synthetic': True})
    with pytest.raises(ValueError, match='ONE_SHOT'):
        native.validate_launch(tmp_path, manifest, auth, requested=True, output=output, worker=True)


@pytest.mark.parametrize('gate', ['verify_sources', 'verify_input', 'environment', 'static_receipt'])
def test_dependency_change_stops_launch(monkeypatch, tmp_path, gate):
    manifest, auth, output = sealed(monkeypatch, tmp_path)
    def changed(*a):
        raise ValueError('SYNTHETIC_DEPENDENCY_CHANGED')
    monkeypatch.setattr(native, gate, changed)
    with pytest.raises(ValueError, match='SYNTHETIC_DEPENDENCY_CHANGED'):
        native.validate_launch(tmp_path, manifest, auth, requested=True, output=output)
    assert not output.exists()


def test_controller_one_load_eight_preparations_exact_coverage(tmp_path, static):
    port, terminal = run_fake(tmp_path, static)
    assert terminal['status'] == native.KIND+'_COMPLETE'
    assert terminal['synthetic_only'] is True
    assert port.loads == 1 and len(port.prepares) == 8
    receipt = json.loads((tmp_path/'native_receipt.json').read_text())
    assert receipt['coverage_binding']['expected_bounds']['primitive_actions'] == 537
    assert receipt['science_manifest_sealed'] is False
    assert terminal['calls'] == {'snapshot_loads': 1, 'native_preparation_calls': 8, **dict.fromkeys(native.FORBIDDEN, 0)}
    assert receipt['N'] is None and receipt['G'] is None


@pytest.mark.parametrize('fail,attempts,completed', [('load', 0, 0), ('H4_B1_S2_q1', 1, 0), ('H4_B2_K2', 4, 3)])
def test_failure_retains_partials_without_retry(tmp_path, static, fail, attempts, completed):
    port, terminal = run_fake(tmp_path, static, fail=fail)
    assert terminal['status'] == native.KIND+'_STOP'
    assert terminal['calls']['native_preparation_calls'] == attempts
    assert terminal['completed_preparations'] == completed
    assert port.loads == 1 and len(port.prepares) == attempts
    assert len(list(tmp_path.glob('*_native.json'))) == completed
    assert not (tmp_path/'native_receipt.json').exists()


@pytest.mark.parametrize('change', ['cap', 'times', 'schedule', 'identity', 'block_count', 'nonfinite'])
def test_native_receipt_mismatch_stops(tmp_path, static, change):
    def mutate(record):
        if change == 'cap': record['bounds_row']['wrapper_instruction_upper_bounds']['ordinary'] = 1000001
        if change == 'times': record['bounds_row']['schedule']['registered_validation_times_v2'][0][1] += .001
        if change == 'schedule': record['bounds_row']['schedule']['synthetic_extra'] = True
        if change == 'identity' and record['cell_id'] == 'H4_B0_q4': record['deterministic_blocks'][0]['order_index'] = 99
        if change == 'block_count': record['deterministic_blocks'].pop()
        if change == 'nonfinite': record['synthetic_nan'] = float('nan')
    _, terminal = run_fake(tmp_path, static, mutate=mutate)
    assert terminal['status'] == native.KIND+'_STOP'
    assert not (tmp_path/'native_receipt.json').exists()


def test_byte_cap_before_write_keeps_terminal_reserve(tmp_path):
    writer = BoundedWriter(tmp_path, byte_cap=70000)
    with pytest.raises(RuntimeError, match='OUTPUT_WRITE_CAP'):
        writer.write('too_large.json', {'x': 'x'*5000})
    assert not (tmp_path/'too_large.json').exists()
    writer.write('terminal.json', {'status': 'STOP'}, terminal=True)


@pytest.mark.parametrize('child,expected', [
    ('import time; time.sleep(5)', 'TOTAL_WALL_CAP'),
    ('print("x"*100000)', 'WORKER_LOG_OR_OUTPUT_CAP'),
    ('pass', 'TERMINAL_INVALID'),
    ('raise SystemExit(3)', 'WORKER_NONZERO_EXIT'),
    ('from pathlib import Path; Path("worker_terminal.json").write_text("{}")', 'TERMINAL_INVALID')])
def test_watchdog_stops_dummy_children(tmp_path, static, child, expected):
    caps = native.plan()['caps'] | {'total_wall_seconds': .2, 'log_bytes': 1024}
    # Dummy children can create only synthetic files in this temporary directory.
    command = [sys.executable, '-c', 'import os; os.chdir('+repr(str(tmp_path))+'); '+child]
    report = watchdog.supervise(command, tmp_path, caps=caps, manifest=manifest_stub(), static=static, poll_seconds=.01)
    assert report['status'] == native.KIND+'_STOP' and expected in report['reason']
    assert report['mandatory_stop'] and not report['next_stage_authorized']
    assert report['worker_log_bytes'] <= 1024


def test_parent_checks_receipt_bytes_and_all_cells(tmp_path, static):
    _, terminal = run_fake(tmp_path, static)
    terminal['synthetic_only'] = False  # Synthetic fixture simulates production flags only.
    path = tmp_path/'native_receipt.json'
    receipt = json.loads(path.read_text()); receipt['synthetic_only'] = False
    path.write_text(json.dumps(receipt))
    terminal['receipt_sha256'] = native.file_hash(path)
    exclusive_json(tmp_path/'worker_terminal.json', terminal)
    assert watchdog.verify_terminal(tmp_path, manifest_stub(), static)['completed_preparations'] == 8
    cell = tmp_path/'H4_B2_K2_native.json'
    record = json.loads(cell.read_text()); record['synthetic_added'] = True
    cell.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='RECEIPT_BINDING'):
        watchdog.verify_terminal(tmp_path, manifest_stub(), static)


def test_synthetic_terminal_never_accepted_as_production(tmp_path, static):
    _, terminal = run_fake(tmp_path, static)
    exclusive_json(tmp_path/'worker_terminal.json', terminal)
    with pytest.raises(ValueError, match='WORKER_FAILED'):
        watchdog.verify_terminal(tmp_path, manifest_stub(), static)


def test_production_port_reuses_only_frozen_load_prepare_bounds(monkeypatch):
    # Replace every numerical dependency before the ProductionPort import factory.
    events = []
    ham = SimpleNamespace(n_qubits=8, n_blocks=12)
    module = SimpleNamespace(_load_snapshot_once=lambda p: (events.append('load') or ham,
        SimpleNamespace(dimension=36), object(), object(), {}, {}))
    monkeypatch.setitem(sys.modules, 'trotterlib.pr2_new_series_validation', module)
    @dataclass
    class Operation:
        name: str = 'synthetic'
    @dataclass
    class Spec:
        component_id: str = 'synthetic'
    block = SimpleNamespace(original_fragment_index=None, block_id='synthetic', basis_id='basis',
        basis_hash='0'*64, num_system_qubits=8, order_index=0, basis_change_operations=(Operation(),),
        runtime_basis_operations=(object(),), diagonal_eigenvalues=(0.,))
    prep = SimpleNamespace(deterministic_blocks=(block,), rte_preparation=SimpleNamespace(component_specs=(Spec(),)),
        preparation_hash='p', hamiltonian_hash='h', partition_hash='s', ld=6, constant_coefficient=0.,
        extracted_identity_coefficient=0., exact_rte_lambda_r=1., identity_policy='extract_identity_phase', coefficient_atol=0.)
    execution = SimpleNamespace(_prepare=lambda h, p: events.append('prepare') or prep,
        _prepare_discard=lambda h, p: events.append('discard') or prep)
    monkeypatch.setitem(sys.modules, 'trotterlib.pr2_matched_accuracy_m1_execution', execution)
    monkeypatch.setitem(sys.modules, 'trottertracks.resource_applicability.ax2b_molecular_ports_v2',
        SimpleNamespace(actual_bounds=lambda *a, **k: {'cells': [{'synthetic': True}]},
                        check_bounds=lambda *a: events.append('bounds')))
    monkeypatch.setitem(sys.modules, 'trottertracks.resource_applicability.ax2b_native_df_v5',
        SimpleNamespace(block_instruction_bound=lambda *a: 10))
    port = native.ProductionPort(ROOT, {'input_binding': {'path': 'synthetic.npz'}})
    assert not events
    assert port.load() is ham
    record = port.prepare(ham, {'id': 'synthetic', 'method': 'B0', 'prefix': 6})
    assert events == ['load', 'discard', 'bounds']
    assert record['component_specs_digest'] == digest([{'component_id': 'synthetic'}])
    assert record['deterministic_blocks'][0]['runtime_basis_operation_count'] == 1


@pytest.mark.parametrize('arguments', [[], ['--execute'], ['--worker'], ['--execute', '--worker']])
def test_cli_default_and_missing_grant_never_import_numerics(tmp_path, arguments):
    runner = ROOT/native.RUNNER
    destination = tmp_path/'draft.json'
    code = '''import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy','scipy','mpmath','qiskit','openfermion'}:
        raise AssertionError('NUMERICAL_IMPORT_BEFORE_GRANT:'+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = %r
runpy.run_path(%r, run_name='__main__')
''' % ([str(runner), '--output', str(destination), *arguments], str(runner))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert 'NUMERICAL_IMPORT_BEFORE_GRANT' not in result.stderr
    if arguments:
        assert result.returncode != 0 and not destination.exists()
    else:
        assert result.returncode == 0
        assert json.loads(destination.read_text())['execution_plan_sealed'] is False


def test_worker_limits_claim_before_port_and_setup_failure_stop(monkeypatch, tmp_path, static):
    spec = importlib.util.spec_from_file_location('synthetic_h4p_cli', ROOT/native.RUNNER)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    output = tmp_path/'out'; output.mkdir()
    manifest = native.draft(); manifest['execution_plan_sealed'] = True
    mp, ap = tmp_path/'manifest.json', tmp_path/'auth.json'
    exclusive_json(mp, manifest)
    exclusive_json(ap, {'manifest_sha256': native.file_hash(mp)})
    events = []
    monkeypatch.setattr(cli, 'validate_launch', lambda *a, **k: 3)
    monkeypatch.setattr(cli, 'static_receipt', lambda *a: static)
    def limits(*a):
        assert not (output/'worker_claim.json').exists()
        events.append('limits')
    monkeypatch.setattr(cli, 'install_worker_limits', limits)
    def factory(*a):
        assert (output/'worker_claim.json').exists()
        events.append('port')
        raise ImportError('synthetic setup failure')
    monkeypatch.setattr(cli, 'ProductionPort', factory)
    monkeypatch.setattr(sys, 'argv', ['synthetic', '--execute', '--worker', '--output', str(output),
        '--manifest', str(mp), '--manifest-sha256', native.file_hash(mp), '--authorization', str(ap),
        '--authorization-sha256', native.file_hash(ap)])
    assert cli.main() == 1 and events == ['limits', 'port']
    terminal = json.loads((output/'worker_terminal.json').read_text())
    assert terminal['status'] == native.KIND+'_STOP' and 'ImportError' in terminal['reason']


@pytest.mark.parametrize('target', ['manifest', 'authorization'])
def test_cli_exact_file_pins_reject_before_gate(monkeypatch, tmp_path, target):
    spec = importlib.util.spec_from_file_location('synthetic_h4p_pins', ROOT/native.RUNNER)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    mp, ap = tmp_path/'manifest.json', tmp_path/'auth.json'
    exclusive_json(mp, native.draft())
    exclusive_json(ap, {'manifest_sha256': native.file_hash(mp)})
    pins = {'manifest': native.file_hash(mp), 'authorization': native.file_hash(ap)}
    pins[target] = '0'*64
    monkeypatch.setattr(cli, 'validate_launch', lambda *a, **k: pytest.fail('Hash gate must come first'))
    output = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['synthetic', '--execute', '--output', str(output),
        '--manifest', str(mp), '--manifest-sha256', pins['manifest'], '--authorization', str(ap),
        '--authorization-sha256', pins['authorization']])
    with pytest.raises(ValueError, match='PINNED_LAUNCH_FILE_HASH'):
        cli.main()
    assert not output.exists()


def test_parent_success_still_requires_stop_and_complete_counters(tmp_path, static):
    _, terminal = run_fake(tmp_path, static)
    terminal['synthetic_only'] = False
    rp = tmp_path/'native_receipt.json'
    receipt = json.loads(rp.read_text()); receipt['synthetic_only'] = False
    rp.write_text(json.dumps(receipt)); terminal['receipt_sha256'] = native.file_hash(rp)
    exclusive_json(tmp_path/'worker_terminal.json', terminal)
    report = watchdog.supervise([sys.executable, '-c', 'pass'], tmp_path,
        caps=native.plan()['caps'], manifest=manifest_stub(), static=static)
    assert report['status'] == native.KIND+'_COMPLETE'
    assert not report['science_manifest_sealed'] and not report['next_stage_authorized']
    terminal['calls']['native_preparation_calls'] = 7
    (tmp_path/'worker_terminal.json').write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match='WORKER_FAILED_OR_INCOMPLETE'):
        watchdog.verify_terminal(tmp_path, manifest_stub(), static)
