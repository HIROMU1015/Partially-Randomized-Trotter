#!/usr/bin/env python3
"""Seal H4 metadata only. This entry point cannot launch or create a grant."""
from __future__ import annotations

import argparse
import builtins
import copy
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_bound_launch_v2 import (
    environment, preparation, safe_path, source_paths, verify_input, verify_sources,
    validate_launch,
)
from trottertracks.resource_applicability.ax2b_h4_saved_receipt_gate_v2 import (
    audit_saved_run, committed_blob, read_bytes, require, sha, strict_json,
)
from trottertracks.resource_applicability.ax2b_limits import exclusive_json

SELF = 'scripts/resource_applicability/seal_track_a_ax2b_h4_limited_v1.py'
TEST = 'tests/tracks/resource_applicability/test_ax2b_h4_limited_seal_v1.py'
REAUDIT_COMMIT = 'ab98bed720bc9d3ba5d45c619f1a1575fccef4a1'
AUDIT_SOURCE = '941eda0b22bfedda10ece5ed4cdefad77cf3d2b2'
REAUDIT_DIR = 'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10'
STATIC = 'artifacts/resource_applicability/track_a_ax2b_h4_prelaunch_preparation/2026-10-10/metadata_preflight_v3.json'
HEADER_READER = 'scripts/resource_applicability/prepare_track_a_ax2b_h4_prelaunch_v3.py'
FUTURE_OUTPUT = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v1'
SEAL_DIR = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10'
NUMERICAL = {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion'}


def install_import_guard():
    original = builtins.__import__
    def metadata_import(name, *args, **kwargs):
        if name.split('.')[0] in NUMERICAL:
            raise RuntimeError('NUMERICAL_IMPORT_FORBIDDEN_IN_H4_SEAL:' + name)
        return original(name, *args, **kwargs)
    builtins.__import__ = metadata_import


def published(root, commit, name):
    data = read_bytes(safe_path(root, name), 4 * 2**20)
    require(data == committed_blob(root, commit, name), 'PUBLISHED_BYTES_CHANGED:' + name)
    return data


def compose(receipt, *, source_commit, hashes, observed_environment, cpu, affinity,
            output_absolute):
    """Pure JSON binding; never derive bounds from new native preparations."""
    require(receipt['status'] == 'H4P_SAVED_RECEIPT_REAUDIT_PASS', 'REAUDIT_PASS_REQUIRED')
    require(receipt['original_parent_status'] == 'H4_NATIVE_RECEIPT_STOP'
            and receipt['original_parent_reason'] == 'TERMINAL_INVALID:JSON_INPUT_SIZE'
            and receipt['original_parent_STOP_unchanged'] is True, 'PRESERVE_ORIGINAL_STOP')
    require(receipt['science_manifest_sealed'] is False and receipt['launch_authorized'] is False
            and receipt['H6_status'] == 'H6_NOT_AUTHORIZED'
            and receipt['contract_status'] == 'DRAFT_NOT_AUTHORIZATION'
            and receipt['mandatory_stop'] is True and receipt['next_stage_authorized'] is False,
            'REAUDIT_STOP_FLAGS')
    require(all(type(v) is int and v == 0 for v in receipt['new_molecular_work'].values()),
            'REAUDIT_NEW_SCIENCE')
    require(receipt['environment'] == observed_environment, 'ENVIRONMENT_CHANGED')
    require(type(cpu) is int and cpu in affinity, 'ASSIGNED_CPU_UNAVAILABLE')
    manifest = preparation('H4_LIMITED')
    coverage = copy.deepcopy(receipt['original_coverage_binding'])
    require(coverage['sealed'] is True and type(coverage['actual_rank']) is int
            and coverage['actual_rank'] == 12
            and coverage['schedule_digest'] == digest(manifest['plan']['cells']), 'COVERAGE_BINDING')
    bounds = coverage['expected_bounds']
    require([r['cell_id'] for r in bounds['cells']] == [c['id'] for c in manifest['plan']['cells']],
            'REGISTERED_CELL_SET')
    require(bounds['primitive_actions'] == 537 and bounds['primitive_probe_count'] == 3
            and receipt['registered_primitive_time_pairs'] == 179
            and receipt['registered_future_probe_actions'] == 537
            and bounds['structural_bounds_are_not_compiled_costs'] is True, 'PROBE_COVERAGE')
    caps = manifest['plan']['caps_proposed']
    require(bounds['primitive_actions'] <= caps['primitive'], 'PRIMITIVE_CAP')
    for row in bounds['cells']:
        require(set(row['wrapper_instruction_upper_bounds']) == {'ordinary', 'symmetric_directional'}
                and all(type(v) is int and 0 <= v <= caps['untranspiled_instructions']
                        for v in row['wrapper_instruction_upper_bounds'].values()), 'INSTRUCTION_CAP')
    manifest.update(source_commit=source_commit, source_hashes=dict(hashes),
        environment=copy.deepcopy(observed_environment), input_binding=copy.deepcopy(receipt['input_binding']),
        coverage_binding=coverage, assigned_resources={'assigned_cpu': cpu, 'science_workers': 1, 'blas_threads': 1},
        execution_plan_sealed=True,
        intended_exclusive_output={'repository_path': FUTURE_OUTPUT, 'absolute_path': str(output_absolute)},
        H6_status='H6_NOT_AUTHORIZED',
        seal_meaning='fixed preparation metadata only; separate explicit pinned launch grant absent')
    return manifest


def unauthorized_gate(manifest, output):
    """Real frozen launch gate, without constructing an approved grant."""
    try:
        validate_launch(ROOT, manifest, None, requested=True, output=output)
    except ValueError as error:
        require(str(error) == 'SEPARATE_EXPLICIT_USER_GRANT_REQUIRED', 'UNEXPECTED_LAUNCH_GATE')
        return {'authorization': None, 'approved_grant_created': False,
                'gate_reason': str(error), 'launch_rejected': True, 'worker_started': False}
    raise RuntimeError('UNAUTHORIZED_LAUNCH_ACCEPTED')


def build(root, *, source_commit, cpu):
    root = Path(root).resolve()
    install_import_guard()
    tooling = {}
    for name in (SELF, TEST):
        tooling[name] = sha(published(root, source_commit, name))
    saved_bytes = published(root, REAUDIT_COMMIT, REAUDIT_DIR + '/reaudit_receipt_v2.json')
    saved = strict_json(saved_bytes)
    # Saved receipt arithmetic/Git/hash/header/Unicode only. No numerical load.
    fresh = audit_saved_run(root, AUDIT_SOURCE)
    require(digest(fresh) == digest(saved), 'REAUDIT_RECEIPT_CHANGED')
    freeze = strict_json(published(root, REAUDIT_COMMIT, REAUDIT_DIR + '/audit_source_freeze_v2.json'))
    for name, expected in freeze['source_hashes'].items():
        require(sha(published(root, AUDIT_SOURCE, name)) == expected, 'INHERITED_SOURCE_BYTES:' + name)
    hashes = {name: sha(published(root, source_commit, name)) for name in source_paths(root)}
    verify_sources(root, source_commit, hashes)
    require(environment() == saved['environment'], 'ENVIRONMENT_CHANGED')
    require(verify_input(root, saved['input_binding'], 'H4_LIMITED') == 12, 'INPUT_RANK')
    static_bytes = published(root, 'f0a5021b8f97dd9116578c7c3e455b7bc2559792', STATIC)
    static = strict_json(static_bytes)
    require(static['manifest']['plan'] == preparation('H4_LIMITED')['plan'], 'FIXED_PLAN_CHANGED')
    reader = root / HEADER_READER
    spec = importlib.util.spec_from_file_location('h4_frozen_header_reader', reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    headers = module.layout_headers(safe_path(root, saved['input_binding']['path']))
    require(headers == static['NPZ_headers'], 'NPZ_HEADERS_CHANGED')
    affinity = sorted(os.sched_getaffinity(0))
    future_output = safe_path(root, FUTURE_OUTPUT)
    require(not future_output.exists(), 'FUTURE_OUTPUT_EXISTS')
    manifest = compose(saved, source_commit=source_commit, hashes=hashes,
        observed_environment=environment(), cpu=cpu, affinity=affinity, output_absolute=future_output)
    manifest['preparation_provenance'] = {
        'reaudit': {'commit': REAUDIT_COMMIT, 'path': REAUDIT_DIR + '/reaudit_receipt_v2.json',
                    'sha256': sha(saved_bytes), 'audit_source_commit': AUDIT_SOURCE},
        'native_execution_source_commit': saved['executed_source_commit'],
        'native_result_commit': saved['original_result_commit'],
        'native_receipt_sha256': saved['native_receipt_sha256'],
        'original_parent_status': saved['original_parent_status'],
        'original_parent_reason': saved['original_parent_reason'],
        'original_parent_STOP_unchanged': True, 'seal_tool_source_commit': source_commit,
        'seal_tool_hashes': tooling, 'static_preflight_sha256': sha(static_bytes)}
    gate = unauthorized_gate(manifest, future_output)
    require(not any(n.split('.')[0] in NUMERICAL for n in sys.modules), 'NUMERICAL_IMPORT_DURING_SEAL')
    require(not future_output.exists(), 'FUTURE_OUTPUT_CREATED')
    audit = {'schema': 'track_a_ax2b_h4_limited_seal_audit_v1',
        'status': 'H4_LIMITED_PLAN_SEALED_NOT_AUTHORIZED',
        'source_commit': source_commit, 'science_source_closure_checked': len(hashes),
        'inherited_source_freeze_checked': len(freeze['source_hashes']),
        'historical_native_source_closure_checked': len(saved['executed_source_hashes']),
        'seal_tool_hashes': tooling, 'manifest_digest': digest(manifest),
        'input_headers': headers, 'environment': manifest['environment'],
        'registered_cells': [c['id'] for c in manifest['plan']['cells']],
        'primitive_time_pairs_planned': 179, 'primitive_actions_planned': 537,
        'control_probe_actions_planned': static['static_coverage']['control_probe_actions_planned'],
        'assigned_resources': manifest['assigned_resources'], 'allowed_cpu_ids_observed': affinity,
        'cpu_assignment_role': 'future execution plan only; no OS reservation or science worker',
        'exclusive_output': manifest['intended_exclusive_output'], 'science_output_created': False,
        'output_runtime_binding': 'frozen v2 launcher checks the separate grant output; future grant must equal intended output',
        'separate_authorization_gate': gate, 'authorization': None,
        'new_molecular_work': dict(saved['new_molecular_work']), 'H4P_reexecuted': False,
        'original_parent_STOP_unchanged': True, 'execution_plan_sealed': True,
        'science_authorized': False, 'launch_allowed': False,
        'N': None, 'G': None, 'accuracy_eligibility': 'UNDETERMINED', 'numerical_allowance_certified': False,
        'H6_status': 'H6_NOT_AUTHORIZED', 'contract_status': 'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop': True, 'next_stage_authorized': False}
    return manifest, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--assigned-cpu', required=True, type=int)
    args = parser.parse_args()
    manifest_path = ROOT / SEAL_DIR / 'sealed_preparation_manifest_v1.json'
    audit_path = ROOT / SEAL_DIR / 'seal_audit_v1.json'
    require(manifest_path.parent.is_dir() and not manifest_path.exists()
            and not audit_path.exists(), 'EXCLUSIVE_SEAL_METADATA_OUTPUT')
    manifest, audit = build(ROOT, source_commit=args.source_commit, cpu=args.assigned_cpu)
    exclusive_json(manifest_path, manifest)
    audit['manifest_sha256'] = sha(read_bytes(manifest_path, 4 * 2**20))
    exclusive_json(audit_path, audit)
    print(audit['status'] + ' / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
