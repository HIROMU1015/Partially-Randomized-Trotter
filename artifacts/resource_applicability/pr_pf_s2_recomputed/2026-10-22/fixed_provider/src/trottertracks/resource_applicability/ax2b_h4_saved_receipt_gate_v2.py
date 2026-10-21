"""Budget-aligned saved H4-P verification. No launch or numerical backend.

Frozen v1 remains unchanged. A new audit PASS does not rewrite its parent STOP.
Both aggregate bytes and each read are bounded by the ORIGINAL output budget.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess

from .ax2a_preparation import digest
from .ax2b_bound_launch_v2 import RUNNER as BOUND_RUNNER, environment, safe_path, verify_input
from .ax2b_h4_native_receipt_v1 import (
    FORBIDDEN, KIND, OUTPUT, RUNNER as NATIVE_RUNNER, assemble_bounds, plan,
)

AUDIT_RUNNER = 'scripts/resource_applicability/audit_track_a_ax2b_h4_native_receipt_v2.py'
RESULT_COMMIT = '1173dd3342e88239458bcce17ae6a4047f8f1fef'
REGISTRY = 'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/execution_inventory_v1.json'
PREPARATION = 'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'DUPLICATE_JSON_KEY:'+key)
            result[key] = value
        return result
    def number(text):
        value = float(text)
        require(math.isfinite(value), 'NONFINITE_JSON_NUMBER')
        return value
    def constant(text):
        raise ValueError('NONFINITE_JSON_CONSTANT:'+text)
    value = json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)
    require(isinstance(value, dict), 'JSON_OBJECT_REQUIRED')
    return value


def read_bytes(path, cap):
    """Bound actual bytes as well as stat size; reject symlink/nonregular files."""
    require(type(cap) is int and cap >= 0, 'READ_BUDGET')
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= cap, 'FILE_TYPE_OR_READ_CAP')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and (before.st_dev, before.st_ino) == (info.st_dev, info.st_ino),
                'FILE_REPLACED_BEFORE_READ')
        data = stream.read(cap+1)
        after = os.fstat(stream.fileno())
    require(len(data) <= cap, 'ACTUAL_READ_CAP')
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and len(data) == before.st_size, 'FILE_CHANGED_DURING_READ')
    return data


def output_names():
    return {cell['id']+'_native.json' for cell in plan()['cells']} | {
        'authorization.json', 'frozen_preparation.json', 'launch_binding.json', 'native_receipt.json',
        'terminal_status.json', 'worker.log', 'worker_claim.json', 'worker_terminal.json'}


def read_output(output, byte_cap):
    """One aggregate precheck, then a decrementing budget across every file."""
    output = Path(output)
    require(output.is_dir() and not output.is_symlink(), 'OUTPUT_DIRECTORY')
    paths = {p.name: p for p in output.iterdir()}
    require(set(paths) == output_names(), 'EXACT_OUTPUT_FILE_SET')
    infos = [p.lstat() for p in paths.values()]
    require(all(stat.S_ISREG(i.st_mode) for i in infos), 'OUTPUT_REGULAR_FILES_REQUIRED')
    require(type(byte_cap) is int and byte_cap > 0 and sum(i.st_size for i in infos) <= byte_cap,
            'AGGREGATE_OUTPUT_CAP')
    blobs, used = {}, 0
    for name in sorted(paths):
        data = read_bytes(paths[name], byte_cap-used)
        used += len(data)
        blobs[name] = data
    parsed = {}
    for name, data in blobs.items():
        if name.endswith('.json'):
            if name in ('worker_terminal.json', 'terminal_status.json', 'launch_binding.json',
                        'worker_claim.json', 'authorization.json'):
                require(len(data) <= 8192, 'SMALL_RECORD_CAP')
            parsed[name] = strict_json(data)
    return blobs, parsed


def committed_blob(root, commit, name):
    safe_path(root, name)
    return subprocess.check_output(['git', 'show', commit+':'+name], cwd=root)


def committed_closure(root, commit, runners):
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit), 'COMMIT_REQUIRED')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=root,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    names = {s.decode() for s in subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', '-z', commit], cwd=root).split(b'\0') if s}
    return {name for name in names if name.endswith('.py')
            and name.startswith(('src/trotterlib/', 'src/trottertracks/'))} | set(runners)


def verify_historical_sources(root, commit, hashes):
    """Use the executed commit tree, not an expanded future worktree closure."""
    require(isinstance(hashes, dict) and set(hashes) == committed_closure(root, commit, (BOUND_RUNNER, NATIVE_RUNNER)),
            'HISTORICAL_SOURCE_CLOSURE')
    for name, expected in hashes.items():
        require(sha(read_bytes(safe_path(root, name), 4*2**20)) == expected, 'HISTORICAL_LOCAL_SOURCE_CHANGED:'+name)
        require(sha(committed_blob(root, commit, name)) == expected, 'HISTORICAL_COMMIT_SOURCE_BYTES:'+name)


def audit_source_hashes(root, commit):
    paths = committed_closure(root, commit, (BOUND_RUNNER, NATIVE_RUNNER, AUDIT_RUNNER))
    current = {str(p.relative_to(root)) for folder in ('src/trotterlib', 'src/trottertracks')
               for p in (Path(root)/folder).rglob('*.py')} | {BOUND_RUNNER, NATIVE_RUNNER, AUDIT_RUNNER}
    require(paths == current, 'AUDITOR_SOURCE_CLOSURE')
    hashes = {}
    for name in sorted(paths):
        data = read_bytes(safe_path(root, name), 4*2**20)
        require(data == committed_blob(root, commit, name), 'AUDITOR_LOCAL_COMMIT_BYTES:'+name)
        hashes[name] = sha(data)
    return hashes


def structural_record(record, cell):
    """Reproduce frozen instruction-count arithmetic from saved metadata only."""
    require(record['cell_id'] == cell['id'] and record['ld'] == cell['prefix'], 'NATIVE_CELL_BINDING')
    require(record['identity_policy'] == 'extract_identity_phase' and record['coefficient_atol'] == 0., 'NATIVE_POLICY')
    blocks, specs = record['deterministic_blocks'], record['component_specs']
    require(len(specs) == record['component_spec_count'] and digest(specs) == record['component_specs_digest'], 'COMPONENT_SPEC_DIGEST')
    require(len(blocks) == cell['prefix']+1, 'NATIVE_BLOCK_COUNT')
    for i, block in enumerate(blocks):
        require(block['primitive_id'] == ('one' if i == 0 else str(i-1)) and block['n_qubits'] == 8,
                'PRIMITIVE_ID_OR_REGISTER')
        count = block['runtime_basis_operation_count']
        require(type(count) is int and count == len(block['basis_operations']), 'BASIS_OPERATION_COUNT')
        n, basis = 8, 2*count
        pairs = 0 if i == 0 else n*(n-1)//2
        require(block['instruction_bounds'] == {'UNCONTROLLED': basis+n+pairs,
            'ORDINARY': basis+n+pairs+1, 'DIRECTIONAL': basis+3*n+5*pairs+1}, 'BLOCK_BOUND_ARITHMETIC')
    tail = 0
    if cell['method'] in ('B2', 'B3'):
        product, rotation = 0, 0
        for spec in specs:
            support = spec['diagonal_pauli_support']
            if support:
                basis = 2*len(spec['basis_change_operations'])
                product, rotation = max(product, basis+len(support)), max(rotation, basis+1)
        r = cell['R']//cell['q']
        tail = cell['q']*(r*(cell['K']*product+rotation)+1)
    pieces = 3 if cell['order'] == '4th' else 1
    expected = {policy: cell['q']*pieces*sum(block['instruction_bounds'][mode]
        for block in blocks for mode in modes)+tail+5 for policy, modes in (
            ('ordinary', ('ORDINARY', 'ORDINARY')),
            ('symmetric_directional', ('UNCONTROLLED', 'DIRECTIONAL')))}
    require(expected == record['bounds_row']['wrapper_instruction_upper_bounds'], 'WRAPPER_BOUND_ARITHMETIC')
    return expected


def validate_payloads(blobs, rows, manifest, static):
    require(manifest['plan'] == plan() and manifest['kind'] == KIND
            and manifest['execution_plan_sealed'] is True and manifest['science_authorized'] is False
            and manifest['launch_allowed'] is False and manifest['mandatory_stop'] is True
            and manifest['contract_status'] == 'DRAFT_NOT_AUTHORIZATION', 'FIXED_H4P_MANIFEST')
    worker, receipt, parent = (rows[k] for k in ('worker_terminal.json', 'native_receipt.json', 'terminal_status.json'))
    grant = rows['authorization.json']
    require(grant['schema'] == 'track_a_ax2b_h4_native_authorization_v1' and grant['kind'] == KIND
            and grant['approved_by_user'] is True and grant['manifest_digest'] == digest(manifest)
            and grant['manifest_sha256'] == sha(blobs['frozen_preparation.json'])
            and grant['assigned_cpu'] == 3 and grant['retry'] is False and grant['resume'] is False, 'ORIGINAL_GRANT_BINDING')
    require(rows['launch_binding.json'] == {'manifest_digest': digest(manifest), 'authorization_digest': digest(grant)},
            'ORIGINAL_LAUNCH_BINDING')
    require(rows['worker_claim.json'] == {'manifest_digest': digest(manifest), 'authorization_digest': digest(grant),
            'assigned_cpu': 3, 'resume': False}, 'ORIGINAL_WORKER_CLAIM')
    require(parent['status'] == KIND+'_STOP' and parent['reason'] == 'TERMINAL_INVALID:JSON_INPUT_SIZE'
            and parent['worker_exit_code'] == 0 and parent['worker_terminal'] is None, 'ORIGINAL_STOP_REQUIRED')
    expected_calls = {'snapshot_loads': 1, 'native_preparation_calls': 8, **dict.fromkeys(FORBIDDEN, 0)}
    require(worker['status'] == KIND+'_COMPLETE' and worker['reason'] is None
            and worker['completed_preparations'] == 8 and worker['calls'] == expected_calls
            and all(type(v) is int for v in worker['calls'].values()), 'WORKER_COMPLETION_COUNTERS')
    require(worker['receipt_sha256'] == sha(blobs['native_receipt.json']), 'NATIVE_RECEIPT_HASH')
    records = [rows[cell['id']+'_native.json'] for cell in plan()['cells']]
    for record, cell in zip(records, plan()['cells']):
        structural_record(record, cell)
    bounds = assemble_bounds(records, static)
    require(receipt['schema'] == 'track_a_ax2b_h4_native_receipt_v1' and receipt['kind'] == KIND
            and receipt['source_commit'] == manifest['source_commit']
            and receipt['environment'] == manifest['environment'] and receipt['input_binding'] == manifest['input_binding']
            and receipt['manifest_digest'] == worker['manifest_digest'] == parent['manifest_digest'] == digest(manifest)
            and receipt['cell_receipt_digests'] == {r['cell_id']: digest(r) for r in records}
            and receipt['coverage_binding'] == {'sealed': True, 'actual_rank': 12,
                'schedule_digest': digest(plan()['cells']), 'expected_bounds': bounds}
            and receipt['structural_bounds_are_not_compiled_costs'] is True, 'RECEIPT_SOURCE_OR_COVERAGE')
    for row in (worker, receipt):
        require(row['synthetic_only'] is False, 'SYNTHETIC_NOT_EXECUTION_EVIDENCE')
    for row in (parent, worker, receipt):
        require(row['N'] is None and row['G'] is None and row['science_manifest_sealed'] is False
                and row['numerical_allowance_certified'] is False and row['accuracy_eligibility'] == 'UNDETERMINED'
                and row['mandatory_stop'] is True and row['next_stage_authorized'] is False
                and row['H6_status'] == 'H6_NOT_AUTHORIZED' and row['contract_status'] == 'DRAFT_NOT_AUTHORIZATION',
                'SCIENCE_OR_STOP_FLAGS')
    total = sum(len(b) for b in blobs.values())
    require(parent['output_bytes_before_terminal'] == total-len(blobs['terminal_status.json'])
            and parent['worker_log_bytes'] == len(blobs['worker.log']) <= plan()['caps']['log_bytes'], 'OUTPUT_LOG_ACCOUNTING')
    require(type(parent['wall_seconds']) in (int, float) and math.isfinite(parent['wall_seconds'])
            and 0 <= parent['wall_seconds'] <= plan()['caps']['total_wall_seconds'], 'ORIGINAL_WALL_CAP')
    return bounds


def audit_saved_run(root, audit_commit):
    root = Path(root).resolve()
    auditor_hashes = audit_source_hashes(root, audit_commit)
    archived_registry = committed_blob(root, RESULT_COMMIT, REGISTRY)
    require(read_bytes(safe_path(root, REGISTRY), 4*2**20) == archived_registry, 'ORIGINAL_REGISTRY_BYTES')
    registry = strict_json(archived_registry)
    blobs, rows = read_output(safe_path(root, OUTPUT), plan()['caps']['output_bytes'])
    output_hashes = {}
    for name, data in blobs.items():
        path = OUTPUT+'/'+name
        require(sha(data) == registry['file_hashes'][path]
                and data == committed_blob(root, RESULT_COMMIT, path), 'ORIGINAL_RESULT_BYTES:'+name)
        output_hashes[path] = {'sha256': sha(data), 'bytes': len(data)}
    manifest = rows['frozen_preparation.json']
    require(blobs['frozen_preparation.json'] == committed_blob(root, RESULT_COMMIT, PREPARATION+'native_preparation_manifest_v1.json'),
            'ORIGINAL_PREPARATION_BYTES')
    verify_historical_sources(root, manifest['source_commit'], manifest['source_hashes'])
    # Hash/header/Unicode metadata only: no saved ndarray payload decoding.
    verify_input(root, manifest['input_binding'], 'H4_LIMITED')
    require(environment() == manifest['environment'], 'ENVIRONMENT_CHANGED')
    binding = manifest['static_binding']
    static_data = read_bytes(safe_path(root, binding['path']), 4*2**20)
    require(sha(static_data) == binding['sha256'] and static_data == committed_blob(root, binding['commit'], binding['path']),
            'STATIC_RECEIPT_BYTES')
    bounds = validate_payloads(blobs, rows, manifest, strict_json(static_data))
    # Ensure original bytes still match at the end; never write into the old run.
    after, _ = read_output(safe_path(root, OUTPUT), plan()['caps']['output_bytes'])
    require({k: sha(v) for k, v in after.items()} == {k: sha(v) for k, v in blobs.items()}, 'RESULT_CHANGED_DURING_AUDIT')
    return {'schema': 'track_a_ax2b_h4p_saved_receipt_reaudit_v2', 'status': 'H4P_SAVED_RECEIPT_REAUDIT_PASS',
        'audit_source_commit': audit_commit, 'audit_source_hashes': auditor_hashes,
        'original_result_commit': RESULT_COMMIT, 'executed_source_commit': manifest['source_commit'],
        'executed_source_hashes': manifest['source_hashes'], 'input_binding': manifest['input_binding'],
        'environment': manifest['environment'], 'original_parent_status': rows['terminal_status.json']['status'],
        'original_parent_reason': rows['terminal_status.json']['reason'], 'original_parent_STOP_unchanged': True,
        'original_worker_status': rows['worker_terminal.json']['status'], 'original_files': output_hashes,
        'native_receipt_sha256': sha(blobs['native_receipt.json']),
        'read_policy': {'aggregate_bytes': plan()['caps']['output_bytes'],
                        'each_json_bounded_by_remaining_aggregate_budget': True, 'small_record_bytes': 8192},
        'saved_bytes_per_verification_pass': sum(len(b) for b in blobs.values()),
        'original_output_verification_passes': 2,
        'original_coverage_binding': rows['native_receipt.json']['coverage_binding'],
        'registered_primitive_time_pairs': 179, 'registered_future_probe_actions': bounds['primitive_actions'],
        'new_molecular_work': dict.fromkeys(('snapshot_numeric_load', 'native_preparation', 'signal', 'reference',
            'matvec_probe', 'sampling', 'wrapper_build', 'compile', 'solver', 'input_generation', 'H6', 'H8', 'gpu'), 0),
        'eligible_for_future_manifest_review': True, 'science_manifest_sealed': False, 'launch_authorized': False,
        'H4_science_status': 'H4_LIMITED_NOT_AUTHORIZED', 'H6_status': 'H6_NOT_AUTHORIZED',
        'N': None, 'G': None, 'numerical_allowance_certified': False, 'accuracy_eligibility': 'UNDETERMINED',
        'retry': False, 'resume': False, 'contract_status': 'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop': True, 'next_stage_authorized': False}
