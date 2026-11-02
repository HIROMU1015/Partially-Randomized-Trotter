"""Read-only stdlib audit of saved H4-P JSON. Never changes frozen STOP/source.

The frozen parent stopped at its 4 MiB per-JSON read gate. This supplementary
artifact audit accepts existing files only within the original 16 MiB output
budget; it neither re-runs H4-P nor certifies a successful original launch.
"""
import builtins
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
original_import = builtins.__import__
def metadata_only_import(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion'}:
        raise RuntimeError('NUMERICAL_IMPORT_FORBIDDEN_IN_SAVED_AUDIT:'+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = metadata_only_import
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_native_receipt_v1 import (
    FORBIDDEN, KIND, OUTPUT, assemble_bounds, environment, plan,
    static_receipt, verify_input, verify_sources,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def saved_json(path, cap):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > cap:
        raise ValueError('SAVED_FILE_EXCEEDS_REGISTERED_OUTPUT_BUDGET')
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def main():
    output = ROOT/OUTPUT
    manifest_path = ROOT/'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/native_preparation_manifest_v1.json'
    manifest = saved_json(manifest_path, 4*2**20)
    caps = manifest['plan']['caps']
    assert manifest['plan'] == plan()
    files = {p.name: p for p in output.iterdir() if p.is_file()}
    expected = {cell['id']+'_native.json' for cell in plan()['cells']} | {
        'authorization.json', 'frozen_preparation.json', 'launch_binding.json', 'native_receipt.json',
        'terminal_status.json', 'worker.log', 'worker_claim.json', 'worker_terminal.json'}
    assert set(files) == expected
    total = sum(p.stat().st_size for p in files.values())
    assert total <= caps['output_bytes']
    original_hashes = {name: sha(p.read_bytes()) for name, p in files.items()}
    verify_sources(ROOT, manifest['source_commit'], manifest['source_hashes'])
    verify_input(ROOT, manifest['input_binding'], 'H4_LIMITED')
    assert environment() == manifest['environment']
    static = static_receipt(ROOT, manifest['static_binding'])
    assert files['frozen_preparation.json'].read_bytes() == manifest_path.read_bytes()
    grant_path = Path(__file__).with_name('authorization_v1.json')
    grant = saved_json(grant_path, 8192)
    assert files['authorization.json'].read_bytes() == grant_path.read_bytes()
    assert grant['manifest_sha256'] == sha(manifest_path.read_bytes())
    assert grant['manifest_digest'] == digest(manifest)
    assert grant['approved_by_user'] is True and grant['kind'] == KIND
    assert saved_json(files['launch_binding.json'], 8192) == {
        'manifest_digest': digest(manifest), 'authorization_digest': digest(grant)}
    assert saved_json(files['worker_claim.json'], 8192) == {
        'manifest_digest': digest(manifest), 'authorization_digest': digest(grant),
        'assigned_cpu': 3, 'resume': False}
    worker = saved_json(files['worker_terminal.json'], 8192)
    parent = saved_json(files['terminal_status.json'], 8192)
    receipt = saved_json(files['native_receipt.json'], caps['output_bytes'])
    assert parent['status'] == KIND+'_STOP' and parent['reason'] == 'TERMINAL_INVALID:JSON_INPUT_SIZE'
    assert parent['worker_exit_code'] == 0 and parent['worker_terminal'] is None
    assert parent['output_bytes_before_terminal'] == total-files['terminal_status.json'].stat().st_size
    assert parent['wall_seconds'] < caps['total_wall_seconds']
    assert parent['worker_log_bytes'] == files['worker.log'].stat().st_size <= caps['log_bytes']
    assert worker['status'] == KIND+'_COMPLETE' and worker['reason'] is None
    assert worker['manifest_digest'] == digest(manifest) and worker['completed_preparations'] == 8
    assert worker['calls'] == {'snapshot_loads': 1, 'native_preparation_calls': 8, **dict.fromkeys(FORBIDDEN, 0)}
    assert worker['receipt_sha256'] == original_hashes['native_receipt.json']
    records, summaries = [], []
    for cell in plan()['cells']:
        path = files[cell['id']+'_native.json']
        record = saved_json(path, caps['output_bytes'])
        assert record['cell_id'] == cell['id'] and record['ld'] == cell['prefix']
        assert record['identity_policy'] == 'extract_identity_phase' and record['coefficient_atol'] == 0.
        specs, blocks = record['component_specs'], record['deterministic_blocks']
        assert len(specs) == record['component_spec_count']
        assert digest(specs) == record['component_specs_digest']
        for block in blocks:
            n = block['n_qubits']; assert n == 8
            assert block['runtime_basis_operation_count'] == len(block['basis_operations'])
            basis = 2*block['runtime_basis_operation_count']
            pairs = 0 if block['primitive_id'] == 'one' else n*(n-1)//2
            assert block['instruction_bounds'] == {'UNCONTROLLED': basis+n+pairs,
                'ORDINARY': basis+n+pairs+1, 'DIRECTIONAL': basis+3*n+5*pairs+1}
        tail = 0
        if cell['method'] in ('B2', 'B3'):
            product, rotation = 0, 0
            for spec in specs:
                support = spec['diagonal_pauli_support']
                if not support:
                    continue
                basis = 2*len(spec['basis_change_operations'])
                product = max(product, basis+len(support))
                rotation = max(rotation, basis+1)
            r = cell['R']//cell['q']
            tail = cell['q']*(r*(cell['K']*product+rotation)+1)
        pieces = 3 if cell['order'] == '4th' else 1
        structural = {}
        for policy, modes in [('ordinary', ('ORDINARY', 'ORDINARY')),
                              ('symmetric_directional', ('UNCONTROLLED', 'DIRECTIONAL'))]:
            structural[policy] = cell['q']*pieces*sum(block['instruction_bounds'][mode]
                for block in blocks for mode in modes)+tail+5
        assert structural == record['bounds_row']['wrapper_instruction_upper_bounds']
        records.append(record)
        summaries.append({'cell_id': cell['id'], 'native_record_bytes': path.stat().st_size,
            'block_count': len(blocks), 'component_spec_count': len(specs),
            'structural_instruction_upper_bounds': structural, 'tail_structural_bound': tail})
    bounds = assemble_bounds(records, static)
    assert receipt['cell_receipt_digests'] == {r['cell_id']: digest(r) for r in records}
    assert receipt['coverage_binding'] == {'sealed': True, 'actual_rank': 12,
        'schedule_digest': digest(plan()['cells']), 'expected_bounds': bounds}
    assert receipt['source_commit'] == manifest['source_commit']
    assert receipt['manifest_digest'] == digest(manifest)
    assert receipt['input_binding'] == manifest['input_binding'] and receipt['environment'] == manifest['environment']
    assert receipt['structural_bounds_are_not_compiled_costs'] is True
    for row in (worker, receipt):
        assert row['synthetic_only'] is False
    for row in (parent, worker, receipt):
        assert row['N'] is None and row['G'] is None
        assert row['science_manifest_sealed'] is False and row['numerical_allowance_certified'] is False
        assert row['accuracy_eligibility'] == 'UNDETERMINED'
        assert row['mandatory_stop'] is True and row['next_stage_authorized'] is False
        assert row['H6_status'] == 'H6_NOT_AUTHORIZED' and row['contract_status'] == 'DRAFT_NOT_AUTHORIZATION'
    assert {name: sha(p.read_bytes()) for name, p in files.items()} == original_hashes
    freeze_path = ROOT/'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/preparation_source_freeze_v1.json'
    freeze = saved_json(freeze_path, 4*2**20)
    for name, expected_hash in freeze['source_hashes'].items():
        assert sha((ROOT/name).read_bytes()) == expected_hash
        assert sha(subprocess.check_output(['git', 'show', freeze['source_commit']+':'+name], cwd=ROOT)) == expected_hash
    too_large = {name: path.stat().st_size for name, path in files.items()
                 if name.endswith('_native.json') and path.stat().st_size > 4*2**20}
    assert too_large == {'H4_B3_K6_native.json': 4443419}
    result = {'schema': 'track_a_ax2b_h4p_saved_evidence_audit_v1',
        'status': 'SAVED_RECEIPT_BYTES_AND_STRUCTURAL_ARITHMETIC_MATCH_ORIGINAL_RUN_STOP',
        'original_parent_status': parent['status'], 'original_parent_reason': parent['reason'],
        'original_worker_status': worker['status'], 'parent_stop_unchanged': True,
        'preparation_source_commit': manifest['source_commit'], 'source_closure_checked': 170,
        'preparation_freeze_local_and_commit_checked': len(freeze['source_hashes']),
        'input_sha256': manifest['input_binding']['sha256'], 'manifest_sha256': sha(manifest_path.read_bytes()),
        'authorization_sha256': sha(grant_path.read_bytes()),
        'native_receipt_sha256': original_hashes['native_receipt.json'],
        'output_files': {str(path.relative_to(ROOT)): {'bytes': path.stat().st_size, 'sha256': original_hashes[name]}
                         for name, path in files.items()},
        'frozen_parent_json_read_cap_bytes': 4*2**20, 'saved_files_exceeding_frozen_read_cap': too_large,
        'original_output_cap_bytes': caps['output_bytes'], 'original_output_total_bytes': total,
        'supplementary_audit_read_cap_bytes': caps['output_bytes'],
        'supplementary_audit_is_not_a_repaired_launch': True,
        'worker_calls': worker['calls'], 'registered_time_pairs': 179, 'registered_probe_actions': 537,
        'probe_actions_executed': 0, 'cells': summaries, 'wall_seconds': parent['wall_seconds'],
        'RSS_peak_bytes': None, 'N': None, 'G': None, 'accuracy_eligibility': 'UNDETERMINED',
        'numerical_allowance_certified': False, 'H4_science_manifest_sealed': False,
        'H4_science_status': 'H4_LIMITED_NOT_AUTHORIZED', 'H6_status': 'H6_NOT_AUTHORIZED',
        'contract_status': 'DRAFT_NOT_AUTHORIZATION', 'mandatory_stop': True, 'next_stage_authorized': False,
        'new_molecular_work_in_supplementary_audit': 0, 'retry': False, 'resume': False}
    destination = Path(__file__).with_name('saved_evidence_audit_v1.json')
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'original_parent_reason', 'source_closure_checked',
        'preparation_freeze_local_and_commit_checked', 'original_output_total_bytes', 'cells')}))


if __name__ == '__main__':
    main()
