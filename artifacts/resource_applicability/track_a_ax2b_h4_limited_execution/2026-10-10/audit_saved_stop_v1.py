"""Read-only H4 limited STOP audit and metadata type diagnosis. No backend."""
import ast
import builtins
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
original_import = builtins.__import__
def metadata_import(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion'}:
        raise RuntimeError('NUMERICAL_IMPORT_FORBIDDEN_IN_STOP_AUDIT:' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = metadata_import
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_bound_launch_v2 import environment, verify_sources, verify_input
from trottertracks.resource_applicability.ax2b_h4_saved_receipt_gate_v2 import read_bytes, sha, strict_json
from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule

SOURCE = '61091c2cb00eb871d7a692b125219d34d99cc923'
AUTH_COMMIT = '2f4f536ffff62d9da4218210de56d7042785f2ff'
META = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10'
OUTPUT = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v1'
MANIFEST = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/sealed_preparation_manifest_v1.json'
PORT = 'src/trottertracks/resource_applicability/ax2b_molecular_ports_v2.py'
STAGE = 'src/trottertracks/resource_applicability/ax2b_stage_validation_v2.py'


def isolated_function(path, name, namespace):
    """Use only pure schedule helpers; never import/execute their backend module."""
    data = read_bytes(ROOT / path, 4 * 2**20)
    assert data == subprocess.check_output(['git', 'show', SOURCE + ':' + path], cwd=ROOT)
    nodes = [n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, path, 'exec'), namespace)
    return namespace[name]


def main():
    output = ROOT / OUTPUT
    names = {'authorization.json', 'frozen_preparation.json', 'launch_binding.json',
             'phase_input_reference.json', 'terminal_status.json', 'worker.log',
             'worker_claim.json', 'worker_terminal.json'}
    assert {p.name for p in output.iterdir()} == names and not output.is_symlink()
    assert all(stat.S_ISREG((output / n).lstat().st_mode) for n in names)
    blobs = {n: read_bytes(output / n, 4 * 2**20 if n == 'frozen_preparation.json'
                          else 65536 if n == 'worker.log' else 8192) for n in names}
    rows = {n: strict_json(b) for n, b in blobs.items() if n.endswith('.json')}
    manifest = rows['frozen_preparation.json']
    assert blobs['frozen_preparation.json'] == read_bytes(ROOT / MANIFEST, 4 * 2**20)
    assert sha(blobs['frozen_preparation.json']) == '679511b859630992870be50e2ec4d15fa24c4794ddec2bc8b6d127f3e65a58c2'
    auth = rows['authorization.json']
    auth_bytes = read_bytes(ROOT / META / 'authorization_v1.json', 8192)
    assert blobs['authorization.json'] == auth_bytes
    assert auth_bytes == subprocess.check_output(['git', 'show', AUTH_COMMIT + ':' + META + '/authorization_v1.json'], cwd=ROOT)
    assert sha(auth_bytes) == '6c6f66ead40e835b1fe30dc662cefdc8d052d262d8b03ae2669f4f8ec8c63324'
    assert auth['approved_by_user'] is True and auth['manifest_digest'] == digest(manifest)
    assert auth['exclusive_output'] == manifest['intended_exclusive_output']['absolute_path'] == str(output)
    assert auth['assigned_cpu'] == 3 and auth['retry'] is auth['resume'] is False
    binding = {'manifest_digest': digest(manifest), 'authorization_digest': digest(auth)}
    assert rows['launch_binding.json'] == binding
    assert rows['worker_claim.json'] == dict(binding, assigned_cpu=3, resume=False)
    parent, worker = rows['terminal_status.json'], rows['worker_terminal.json']
    assert parent['status'] == worker['status'] == 'H4_LIMITED_STOP'
    assert parent['reason'] == 'WORKER_FAILED_OR_INCOMPLETE'
    assert worker['reason'] == 'ValueError:ACTUAL_COVERAGE_CHANGED'
    assert parent['worker_terminal'] == worker and parent['worker_exit_code'] == 1
    assert worker['completed_correctness_cells'] == worker['compiled_wrappers'] == 0
    assert worker['calls'] == dict.fromkeys(('compile', 'control_probe', 'occurrence',
                                           'primitive', 'reference_matvec', 'trajectory'), 0)
    assert all(type(v) is int for v in worker['calls'].values())
    for row in (worker, parent):
        assert row['mandatory_stop'] is True and row['next_stage_authorized'] is False
        assert row['N'] is row['G'] is None
    assert worker['accuracy_eligibility'] == 'UNDETERMINED' and worker['numerical_allowance_certified'] is False
    assert parent['worker_log_bytes'] == len(blobs['worker.log']) <= manifest['plan']['caps_proposed']['log_bytes']
    assert 0 < parent['wall_seconds'] < manifest['plan']['caps_proposed']['total_wall_seconds']
    assert sum(len(b) for b in blobs.values()) <= manifest['plan']['caps_proposed']['output_bytes']
    assert rows['phase_input_reference.json']['phase'] == 'input_reference'
    assert manifest['source_commit'] == SOURCE
    verify_sources(ROOT, SOURCE, manifest['source_hashes'])
    assert verify_input(ROOT, manifest['input_binding'], 'H4_LIMITED') == 12
    assert manifest['environment'] == environment()
    old = json.loads((ROOT / 'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10/reaudit_receipt_v2.json').read_text())
    assert manifest['coverage_binding'] == old['original_coverage_binding']

    # Stored bounds plus actual stdlib schedule helpers, not a runtime receipt.
    canonical = isolated_function(STAGE, 'canonical_cell', {})
    times = isolated_function(PORT, 'validation_times', {'primitive_time_schedule': primitive_time_schedule})
    expected = manifest['coverage_binding']['expected_bounds']
    metadata_runtime_form = copy.deepcopy(expected)
    for raw, row in zip(manifest['plan']['cells'], metadata_runtime_form['cells'], strict=True):
        cell = canonical(raw)
        row['schedule'] = primitive_time_schedule(cell, T=manifest['plan']['T'])
        row['schedule']['registered_validation_times_v2'] = times(cell, T=manifest['plan']['T'])
    assert metadata_runtime_form != expected
    assert digest(metadata_runtime_form) == digest(expected)
    type_differences = []
    def compare(a, b, path='$'):
        if type(a) is not type(b):
            type_differences.append({'path': path, 'runtime_schedule_type': type(a).__name__,
                                     'stored_type': type(b).__name__})
        if isinstance(a, dict):
            assert set(a) == set(b)
            for key in a: compare(a[key], b[key], path + '.' + key)
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + '[' + str(i) + ']')
        else:
            assert a == b
    compare(metadata_runtime_form, expected)
    assert type_differences and all(d['runtime_schedule_type'] == 'tuple' and d['stored_type'] == 'list'
                                    for d in type_differences)
    missing = ['actual_coverage.json', 'input_reference.json', 'primitive_validation.json']
    missing += [c['id'] + '_correctness.json' for c in manifest['plan']['cells']]
    missing += [c['id'] + '_mp' + str(dps) + '.json' for c in manifest['plan']['cells'] for dps in manifest['plan']['dps']]
    missing += [id_ + '_explicit_order' + str(order) + '.json'
                for id_ in ('H4_B2_K2', 'H4_B3_K6') for order in (0, 2)]
    assert all(not (output / n).exists() for n in missing)
    assert {n: sha(read_bytes(output / n, len(b))) for n, b in blobs.items()} == {n: sha(b) for n, b in blobs.items()}
    report = {'schema': 'track_a_ax2b_h4_limited_saved_stop_audit_v1', 'status': 'SAVED_STOP_AUDIT_PASS',
        'science_status': 'H4_LIMITED_STOP', 'worker_reason': worker['reason'], 'parent_reason': parent['reason'],
        'execution_base_commit': AUTH_COMMIT, 'science_source_commit': SOURCE,
        'source_hashes_checked': len(manifest['source_hashes']), 'manifest_sha256': sha(blobs['frozen_preparation.json']),
        'authorization_sha256': sha(auth_bytes), 'file_hashes': {OUTPUT + '/' + n: sha(b) for n, b in blobs.items()},
        'output_file_count': len(blobs), 'output_bytes': sum(len(b) for b in blobs.values()),
        'wall_seconds': parent['wall_seconds'], 'worker_log_bytes': len(blobs['worker.log']),
        'observed_calls': worker['calls'], 'correctness_cells_completed': 0,
        'missing_planned_records': missing, 'actual_runtime_bounds_saved': False,
        'snapshot_load_and_eight_preparations': 'inferred from frozen source control flow reaching coverage equality; not independently recorded counters',
        'type_diagnosis': {'kind': 'STATIC_SAVED_METADATA_INTERFACE_DIAGNOSIS',
            'backend_imported': False, 'native_reprepared': False, 'new_signal_or_probe': False,
            'stored_bounds_with_runtime_schedule_representation_equal_by_Python': False,
            'stored_bounds_with_runtime_schedule_representation_equal_by_canonical_JSON_digest': True,
            'tuple_list_differences': len(type_differences), 'representative_differences': type_differences[:8],
            'limitations': 'Actual failed runtime bounds were not saved; remaining values cannot be retrospectively confirmed. Stored wrapper bounds are retained, not recomputed.'},
        'one_launch_consumed': True, 'retry': False, 'resume': False, 'source_or_manifest_repaired': False,
        'N': None, 'G': None, 'accuracy_eligibility': 'UNDETERMINED', 'numerical_allowance_certified': False,
        'H6_status': 'H6_NOT_AUTHORIZED', 'contract_status': 'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop': True, 'next_stage_authorized': False}
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
