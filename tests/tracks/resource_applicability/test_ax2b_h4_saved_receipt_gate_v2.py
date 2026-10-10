"""Synthetic JSON/capacity/provenance tests; no numerical backend or NPZ load."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from trottertracks.resource_applicability import ax2b_h4_saved_receipt_gate_v2 as gate
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_native_receipt_v1 import draft, plan, FORBIDDEN, KIND

ROOT = Path(__file__).resolve().parents[3]


def encoded(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode()


@pytest.fixture
def static():
    # Saved schedule arithmetic JSON only. Molecular input is never opened.
    path = ROOT/'artifacts/resource_applicability/track_a_ax2b_h4_prelaunch_preparation/2026-10-10/metadata_preflight_v3.json'
    return json.loads(path.read_text())


def fixture_run(tmp_path, static, *, padding=0):
    manifest = draft()
    manifest.update(execution_plan_sealed=True, source_commit='0'*40,
        input_binding={'synthetic': True}, environment={'synthetic': True})
    grant = {'schema': 'track_a_ax2b_h4_native_authorization_v1', 'kind': KIND,
        'approved_by_user': True, 'manifest_digest': digest(manifest),
        'manifest_sha256': gate.sha(encoded(manifest)), 'assigned_cpu': 3, 'retry': False, 'resume': False}
    records = []
    for cell, old in zip(plan()['cells'], static['static_coverage']['cells']):
        blocks = []
        for i in range(cell['prefix']+1):
            pairs = 0 if i == 0 else 28
            blocks.append({'primitive_id': 'one' if i == 0 else str(i-1), 'n_qubits': 8,
                'basis_operations': [{'synthetic': 'op'}], 'runtime_basis_operation_count': 1,
                'instruction_bounds': {'UNCONTROLLED': 10+pairs, 'ORDINARY': 11+pairs,
                    'DIRECTIONAL': 27+5*pairs}})
        specs = [{'diagonal_pauli_support': [0], 'basis_change_operations': [{'synthetic': 'op'}]}] if cell['R'] else []
        schedule = deepcopy(old['schedule'])
        schedule['registered_validation_times_v2'] = old['registered_validation_times_v2']
        tail = cell['q']*(2*(cell['K']*3+3)+1) if cell['R'] else 0
        pieces = 3 if cell['order'] == '4th' else 1
        bound = {policy: cell['q']*pieces*sum(b['instruction_bounds'][mode]
            for b in blocks for mode in modes)+tail+5 for policy, modes in (
                ('ordinary', ('ORDINARY', 'ORDINARY')),
                ('symmetric_directional', ('UNCONTROLLED', 'DIRECTIONAL')))}
        record = {'cell_id': cell['id'], 'ld': cell['prefix'], 'identity_policy': 'extract_identity_phase',
            'coefficient_atol': 0., 'deterministic_blocks': blocks, 'component_specs': specs,
            'component_spec_count': len(specs), 'component_specs_digest': digest(specs),
            'bounds_row': {'cell_id': cell['id'], 'schedule': schedule, 'wrapper_instruction_upper_bounds': bound}}
        if cell['id'] == 'H4_B3_K6': record['synthetic_padding'] = 'x'*padding
        records.append(record)
    flags = {'N': None, 'G': None, 'science_manifest_sealed': False,
        'numerical_allowance_certified': False, 'accuracy_eligibility': 'UNDETERMINED',
        'mandatory_stop': True, 'next_stage_authorized': False, 'H6_status': 'H6_NOT_AUTHORIZED',
        'contract_status': 'DRAFT_NOT_AUTHORIZATION'}
    receipt = dict(flags, schema='track_a_ax2b_h4_native_receipt_v1', kind=KIND,
        source_commit=manifest['source_commit'], input_binding=manifest['input_binding'],
        environment=manifest['environment'], manifest_digest=digest(manifest), synthetic_only=False,
        cell_receipt_digests={r['cell_id']: digest(r) for r in records},
        coverage_binding={'sealed': True, 'actual_rank': 12, 'schedule_digest': digest(plan()['cells']),
            'expected_bounds': gate.assemble_bounds(records, static)}, structural_bounds_are_not_compiled_costs=True)
    rows = {r['cell_id']+'_native.json': r for r in records}
    rows.update({'frozen_preparation.json': manifest, 'authorization.json': grant, 'native_receipt.json': receipt,
        'launch_binding.json': {'manifest_digest': digest(manifest), 'authorization_digest': digest(grant)},
        'worker_claim.json': {'manifest_digest': digest(manifest), 'authorization_digest': digest(grant), 'assigned_cpu': 3, 'resume': False},
        'worker_terminal.json': dict(flags, status=KIND+'_COMPLETE', reason=None, completed_preparations=8,
            manifest_digest=digest(manifest), calls={'snapshot_loads': 1, 'native_preparation_calls': 8, **dict.fromkeys(FORBIDDEN, 0)},
            receipt_sha256=gate.sha(encoded(receipt)), synthetic_only=False)})
    preterminal = sum(len(encoded(row)) for row in rows.values())+3
    rows['terminal_status.json'] = dict(flags, status=KIND+'_STOP', reason='TERMINAL_INVALID:JSON_INPUT_SIZE',
        manifest_digest=digest(manifest), worker_exit_code=0, worker_terminal=None,
        output_bytes_before_terminal=preterminal, worker_log_bytes=3, wall_seconds=2.)
    for name, row in rows.items(): (tmp_path/name).write_bytes(encoded(row))
    (tmp_path/'worker.log').write_bytes(b'log')
    return manifest


def test_large_json_and_aggregate_budget_regression(tmp_path, static):
    manifest = fixture_run(tmp_path, static, padding=4443419)
    big = tmp_path/'H4_B3_K6_native.json'
    assert big.stat().st_size > 4*2**20
    blobs, rows = gate.read_output(tmp_path, 16*2**20)
    assert gate.validate_payloads(blobs, rows, manifest, static)['primitive_actions'] == 537
    assert rows['terminal_status.json']['status'] == KIND+'_STOP'


def test_aggregate_gate_before_payload_reads(tmp_path, static, monkeypatch):
    fixture_run(tmp_path, static)
    monkeypatch.setattr(gate, 'read_bytes', lambda *a: pytest.fail('Aggregate gate must precede every read'))
    with pytest.raises(ValueError, match='AGGREGATE_OUTPUT_CAP'):
        gate.read_output(tmp_path, 10)


def test_multiple_large_files_cannot_exceed_total_budget(tmp_path, static):
    fixture_run(tmp_path, static)
    (tmp_path/'H4_B2_K2_native.json').write_bytes(b'x'*(9*2**20))
    (tmp_path/'H4_B3_K6_native.json').write_bytes(b'x'*(9*2**20))
    with pytest.raises(ValueError, match='AGGREGATE_OUTPUT_CAP'):
        gate.read_output(tmp_path, 16*2**20)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'symlink', 'directory', 'small_record'])
def test_missing_extra_and_nonregular_files(tmp_path, static, mutation):
    fixture_run(tmp_path, static)
    target = tmp_path/'H4_B0_q4_native.json'
    if mutation == 'missing': target.unlink()
    if mutation == 'extra': (tmp_path/'unexpected.json').write_text('{}')
    if mutation == 'symlink': target.unlink(); target.symlink_to(tmp_path/'native_receipt.json')
    if mutation == 'directory': target.unlink(); target.mkdir()
    if mutation == 'small_record': (tmp_path/'worker_terminal.json').write_bytes(b' '*8193)
    with pytest.raises(ValueError): gate.read_output(tmp_path, 16*2**20)


@pytest.mark.parametrize('data', [b'{', b'[]', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":1,"x":2}'])
def test_invalid_nonfinite_duplicate_json(data):
    with pytest.raises(ValueError): gate.strict_json(data)


def test_actual_read_cap_and_zero_byte_boundary(tmp_path):
    p = tmp_path/'data'; p.write_bytes(b'12345')
    with pytest.raises(ValueError, match='READ_CAP'): gate.read_bytes(p, 4)
    assert gate.read_bytes(p, 5) == b'12345'
    p.write_bytes(b''); assert gate.read_bytes(p, 0) == b''
    p.unlink(); p.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError, match='FILE_TYPE'): gate.read_bytes(p, 10)


@pytest.mark.parametrize('mutation', ['receipt_hash', 'cell_bytes', 'component_digest', 'schedule', 'bound',
    'calls', 'bool_calls', 'parent_status', 'certified', 'next_stage', 'input', 'synthetic', 'grant', 'manifest_seal'])
def test_integrity_and_status_gates(tmp_path, static, mutation):
    manifest = fixture_run(tmp_path, static)
    blobs, rows = gate.read_output(tmp_path, 16*2**20)
    if mutation == 'receipt_hash': blobs['native_receipt.json'] += b' '
    if mutation == 'cell_bytes': rows['H4_B0_q4_native.json']['synthetic_extra'] = True
    if mutation == 'component_digest': rows['H4_B2_K2_native.json']['component_specs_digest'] = '0'*64
    if mutation == 'schedule': rows['H4_B1_S2_q1_native.json']['bounds_row']['schedule']['synthetic_change'] = True
    if mutation == 'bound': rows['H4_B1_S2_q1_native.json']['bounds_row']['wrapper_instruction_upper_bounds']['ordinary'] += 1
    if mutation == 'calls': rows['worker_terminal.json']['calls']['snapshot_loads'] = 2
    if mutation == 'bool_calls': rows['worker_terminal.json']['calls']['snapshot_loads'] = True
    if mutation == 'parent_status': rows['terminal_status.json']['status'] = KIND+'_COMPLETE'
    if mutation == 'certified': rows['native_receipt.json']['numerical_allowance_certified'] = True
    if mutation == 'next_stage': rows['worker_terminal.json']['next_stage_authorized'] = True
    if mutation == 'input': rows['native_receipt.json']['input_binding'] = {'wrong': True}
    if mutation == 'synthetic': rows['worker_terminal.json']['synthetic_only'] = True
    if mutation == 'grant': rows['authorization.json']['approved_by_user'] = False
    if mutation == 'manifest_seal': manifest['science_authorized'] = True
    with pytest.raises(ValueError): gate.validate_payloads(blobs, rows, manifest, static)


def test_native_read_and_validation_leave_original_bytes_unchanged(tmp_path, static):
    manifest = fixture_run(tmp_path, static)
    before = {p.name: gate.sha(p.read_bytes()) for p in tmp_path.iterdir()}
    blobs, rows = gate.read_output(tmp_path, 16*2**20)
    gate.validate_payloads(blobs, rows, manifest, static)
    assert before == {p.name: gate.sha(p.read_bytes()) for p in tmp_path.iterdir()}


def test_historical_source_closure_uses_executed_commit(tmp_path):
    def git(*a): return subprocess.check_output(['git', *a], cwd=tmp_path)
    git('init', '-q')
    paths = ['src/trottertracks/old.py', gate.BOUND_RUNNER, gate.NATIVE_RUNNER]
    for name in paths:
        p = tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic\n')
    git('add', '--', *paths)
    git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid', 'commit', '-qm', 'synthetic source')
    commit = git('rev-parse', 'HEAD').decode().strip()
    hashes = {name: gate.sha((tmp_path/name).read_bytes()) for name in paths}
    (tmp_path/'src/trottertracks/later.py').write_text('# synthetic future extension\n')
    gate.verify_historical_sources(tmp_path, commit, hashes)
    (tmp_path/paths[0]).write_text('# changed\n')
    with pytest.raises(ValueError, match='HISTORICAL_LOCAL_SOURCE_CHANGED'):
        gate.verify_historical_sources(tmp_path, commit, hashes)


@pytest.mark.parametrize('missing', ['path', 'commit'])
def test_historical_missing_or_false_provenance(tmp_path, missing):
    if missing == 'path':
        with pytest.raises(ValueError): gate.committed_blob(tmp_path, '0'*40, '../outside')
    else:
        with pytest.raises(ValueError, match='COMMIT_REQUIRED'): gate.committed_closure(tmp_path, 'not-commit', ())


@pytest.mark.parametrize('arguments', [['--execute'], ['--worker'], []])
def test_saved_cli_has_no_launch_path_or_numerical_import(tmp_path, arguments):
    result = subprocess.run([sys.executable, str(ROOT/gate.AUDIT_RUNNER), '--output', str(tmp_path/'out.json'),
        '--audit-source-commit', '0'*40, *arguments], capture_output=True, text=True)
    assert result.returncode != 0 and not (tmp_path/'out.json').exists()
    assert 'NUMERICAL_IMPORT_FORBIDDEN' not in result.stderr
    if arguments: assert 'unrecognized arguments' in result.stderr
    else: assert 'dedicated v2 re-audit directory' in result.stderr
