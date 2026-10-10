#!/usr/bin/env python3
"""Metadata-only v3 coverage reseal. No numerical import, grant or launch."""
from __future__ import annotations

import argparse
import ast
import builtins
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_bound_launch_v3 import (
    environment, preparation, safe_path, source_paths, verify_input, verify_sources, validate_launch,
)
from trottertracks.resource_applicability.ax2b_coverage_binding_v3 import canonical_coverage
from trottertracks.resource_applicability.ax2b_h4_saved_receipt_gate_v2 import (
    committed_blob, read_bytes, require, sha, strict_json,
)
from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
from trottertracks.resource_applicability.ax2b_limits import exclusive_json

OLD_SOURCE = '61091c2cb00eb871d7a692b125219d34d99cc923'
OLD_SEAL = 'c56ebc433b7ae14df3f50fec3d0b95e04c318208'
OLD_STOP = '79858dbe6724dd3ca020d471c1786e9c889652cc'
OLD_DIR = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10'
STOP_DIR = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10'
SEAL_DIR = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_seal_v2/2026-10-10'
FUTURE_OUTPUT = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2'
ADDED_SOURCE = (
    'src/trottertracks/resource_applicability/ax2b_coverage_binding_v3.py',
    'src/trottertracks/resource_applicability/ax2b_bound_launch_v3.py',
    'src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py',
    'scripts/resource_applicability/run_track_a_ax2b_bound_v3.py',
    'scripts/resource_applicability/seal_track_a_ax2b_h4_limited_v2.py',
    'tests/tracks/resource_applicability/test_ax2b_coverage_binding_v3.py',
)
NUMERICAL = {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion', 'pyscf'}


def install_import_guard():
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in NUMERICAL:
            raise RuntimeError('NUMERICAL_IMPORT_FORBIDDEN_IN_H4_RESEAL:' + name)
        return original(name, *args, **kwargs)
    builtins.__import__ = guarded


def published(root, commit, name):
    data = read_bytes(safe_path(root, name), 4 * 2**20)
    require(data == committed_blob(root, commit, name), 'PUBLISHED_BYTES_CHANGED:' + name)
    return data


def isolated(root, path, name, namespace):
    tree = ast.parse(published(root, OLD_SOURCE, path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, 'PURE_SCHEDULE_HELPER')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


def schedule_round_trip(root, manifest):
    """Saved wrapper bounds retained; no native preparations are recomputed."""
    expected = manifest['coverage_binding']['expected_bounds']
    runtime_form = copy.deepcopy(expected)
    canonical = isolated(root, 'src/trottertracks/resource_applicability/ax2b_stage_validation_v2.py', 'canonical_cell', {})
    times = isolated(root, 'src/trottertracks/resource_applicability/ax2b_molecular_ports_v2.py', 'validation_times',
                     {'primitive_time_schedule': primitive_time_schedule})
    for cell, row in zip(manifest['plan']['cells'], runtime_form['cells'], strict=True):
        value = canonical(cell)
        row['schedule'] = primitive_time_schedule(value, T=manifest['plan']['T'])
        row['schedule']['registered_validation_times_v2'] = times(value, T=manifest['plan']['T'])
    expected_bytes, runtime_bytes = canonical_coverage(expected), canonical_coverage(runtime_form)
    require(runtime_form != expected and expected_bytes == runtime_bytes, 'SCHEDULE_ROUND_TRIP')
    return {'scope': 'saved metadata plus frozen pure schedule helpers; not failed runtime receipt',
            'python_direct_equality': False, 'strict_canonical_equality': True,
            'canonical_sha256': sha(expected_bytes), 'canonical_bytes': len(expected_bytes),
            'cells': len(manifest['plan']['cells']), 'actual_native_bounds_recomputed': False}


def build(root, *, source_commit, cpu):
    root = Path(root).resolve()
    install_import_guard()
    old_manifest_bytes = published(root, OLD_SEAL, OLD_DIR + '/sealed_preparation_manifest_v1.json')
    old_manifest = strict_json(old_manifest_bytes)
    require(old_manifest['source_commit'] == OLD_SOURCE, 'OLD_SOURCE_BINDING')
    inherited = strict_json(published(root, OLD_SEAL, OLD_DIR + '/preparation_source_freeze_v1.json'))
    require(len(inherited['source_hashes']) == 189, 'INHERITED_FREEZE_COUNT')
    for path, expected in inherited['source_hashes'].items():
        require(sha(published(root, OLD_SOURCE, path)) == expected, 'FROZEN_SOURCE_CHANGED:' + path)
    saved_stop_bytes = published(root, OLD_STOP, STOP_DIR + '/saved_stop_audit_v1.json')
    saved_stop = strict_json(saved_stop_bytes)
    require(saved_stop['science_status'] == 'H4_LIMITED_STOP' and saved_stop['one_launch_consumed'] is True,
            'KEEP_PRIOR_STOP_AND_CONSUMED_GRANT')
    for path, expected in saved_stop['file_hashes'].items():
        require(sha(published(root, OLD_STOP, path)) == expected, 'OLD_STOP_BYTES:' + path)
    require(environment() == old_manifest['environment'], 'ENVIRONMENT_CHANGED')
    require(verify_input(root, old_manifest['input_binding'], 'H4_LIMITED') == 12, 'INPUT_CHANGED')
    require(canonical_coverage(preparation()['plan']) == canonical_coverage(old_manifest['plan']), 'SCIENCE_PLAN_CHANGED')
    affinity = sorted(os.sched_getaffinity(0))
    require(type(cpu) is int and cpu in affinity, 'ASSIGNED_CPU_UNAVAILABLE')
    future = safe_path(root, FUTURE_OUTPUT)
    require(not future.exists(), 'EXCLUSIVE_OUTPUT_REQUIRED')
    hashes = {path: sha(published(root, source_commit, path)) for path in source_paths(root)}
    verify_sources(root, source_commit, hashes)
    added = {path: sha(published(root, source_commit, path)) for path in ADDED_SOURCE}
    round_trip = schedule_round_trip(root, old_manifest)
    manifest = copy.deepcopy(old_manifest)
    manifest.update(schema='track_a_ax2b_bound_preparation_v3', source_commit=source_commit, source_hashes=hashes,
        assigned_resources={'assigned_cpu': cpu, 'science_workers': 1, 'blas_threads': 1},
        intended_exclusive_output={'repository_path': FUTURE_OUTPUT, 'absolute_path': str(future)})
    manifest['preparation_provenance']['coverage_v3_reseal'] = {
        'prior_seal_commit': OLD_SEAL, 'prior_manifest_sha256': sha(old_manifest_bytes),
        'prior_STOP_commit': OLD_STOP, 'prior_STOP_audit_sha256': sha(saved_stop_bytes),
        'source_commit': source_commit, 'added_source_hashes': added,
        'science_plan_and_input_unchanged': True, 'round_trip': round_trip,
        'prior_failed_actual_bounds_not_saved': True, 'new_grant_absent': True}
    try:
        validate_launch(root, manifest, None, requested=True, output=future)
    except ValueError as error:
        require(str(error) == 'SEPARATE_EXPLICIT_USER_GRANT_REQUIRED', 'UNEXPECTED_GATE')
    else:
        raise RuntimeError('UNAUTHORIZED_LAUNCH_ACCEPTED')
    require(not any(name.split('.')[0] in NUMERICAL for name in sys.modules), 'NUMERICAL_IMPORT_DURING_RESEAL')
    require(not future.exists(), 'SCIENCE_OUTPUT_CREATED')
    freeze_hashes = dict(inherited['source_hashes']); freeze_hashes.update(added)
    freeze = {'schema': 'track_a_ax2b_h4_coverage_v3_preparation_freeze', 'source_commit': source_commit,
              'source_hashes': freeze_hashes, 'total_count': len(freeze_hashes),
              'inherited_freeze_commit': OLD_SEAL, 'inherited_source_commit': OLD_SOURCE,
              'inherited_freeze_count': 189, 'added_source_count': len(added),
              'science_source_closure': len(hashes), 'science_authorized': False, 'launch_allowed': False,
              'H6_status': 'H6_NOT_AUTHORIZED', 'contract_status': 'DRAFT_NOT_AUTHORIZATION',
              'mandatory_stop': True, 'next_stage_authorized': False}
    audit = {'schema': 'track_a_ax2b_h4_limited_reseal_audit_v2', 'status': 'H4_V3_PLAN_SEALED_NOT_AUTHORIZED',
             'source_commit': source_commit, 'manifest_digest': digest(manifest), 'source_closure_checked': len(hashes),
             'preparation_freeze_checked': len(freeze_hashes), 'old_source_freeze_checked': 189,
             'old_STOP_files_checked': len(saved_stop['file_hashes']), 'old_STOP_unchanged': True,
             'round_trip': round_trip, 'input_binding': manifest['input_binding'], 'environment': manifest['environment'],
             'assigned_resources': manifest['assigned_resources'], 'affinity_observed': affinity,
             'resource_role': 'future assignment, not an OS reservation or running worker',
             'exclusive_output': manifest['intended_exclusive_output'], 'science_output_created': False,
             'new_molecular_work': dict.fromkeys(('load', 'prepare', 'signal', 'primitive', 'control_probe',
                                                  'trajectory', 'circuit_build', 'compile'), 0),
             'authorization': None, 'unauthorized_launch_rejected': True, 'science_authorized': False, 'launch_allowed': False,
             'N': None, 'G': None, 'accuracy_eligibility': 'UNDETERMINED', 'numerical_allowance_certified': False,
             'H6_status': 'H6_NOT_AUTHORIZED', 'contract_status': 'DRAFT_NOT_AUTHORIZATION',
             'mandatory_stop': True, 'next_stage_authorized': False}
    return manifest, freeze, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--assigned-cpu', required=True, type=int)
    args = parser.parse_args()
    directory = ROOT / SEAL_DIR
    require(directory.is_dir() and not any(directory.iterdir()), 'EXCLUSIVE_RESEAL_OUTPUT')
    manifest, freeze, audit = build(ROOT, source_commit=args.source_commit, cpu=args.assigned_cpu)
    path = directory / 'sealed_preparation_manifest_v2.json'
    exclusive_json(path, manifest)
    audit['manifest_sha256'] = sha(read_bytes(path, 4 * 2**20))
    exclusive_json(directory / 'preparation_source_freeze_v2.json', freeze)
    exclusive_json(directory / 'seal_audit_v2.json', audit)
    print(audit['status'] + ' / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION')


if __name__ == '__main__':
    main()
