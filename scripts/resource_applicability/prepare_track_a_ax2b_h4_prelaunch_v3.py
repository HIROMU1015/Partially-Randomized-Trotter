#!/usr/bin/env python3
"""Bind H4 metadata and static coverage without numerical loads or execution.

This preparation cannot seal the scientific launcher: native basis operations
are absent from archived receipts. No execute option or approval is provided.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_bound_launch_v2 import (
    preparation, scope_plan, environment, safe_path, verify_sources,
)
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import (
    SNAPSHOT, snapshot_binding,
)
from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule

SOURCE_COMMIT = '8281c2a59e7d3ea0c77ae03a4fe0227361165c1d'
RESULT_COMMIT = 'aa9b4768819680600d99ceb962b22aec16b99fb0'
PRIOR = 'artifacts/resource_applicability/track_a_ax2b_bound_ports_preparation/2026-10-10'
RESULT = 'artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1'
OUTPUT_PROPOSED = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v1'
LAYOUT = {
    'constant': ('<f8', ()), 'one_body': ('<c16', (8, 8)),
    'lambdas': ('<f8', (12,)), 'g_matrices': ('<c16', (12, 8, 8)),
    'sector_basis_indices': ('<i8', (36,)),
    'state_vector': ('<c16', (256,)), 'sector_state_vector': ('<c16', (36,)),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(root, name):
    path = safe_path(root, name)
    if path.stat().st_size > 4 * 2**20:
        raise ValueError('JSON_SIZE')
    return json.loads(path.read_text())


def layout_headers(path):
    """Inspect only NPY headers; scientific payload bytes are never decoded."""
    receipt = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        expected = {key + '.npy' for key in LAYOUT} | {'metadata_json.npy'}
        if len(names) != 8 or set(names) != expected:
            raise ValueError('NPZ_KEYS_OR_DUPLICATES')
        if sum(i.file_size for i in infos) > 16 * 2**20:
            raise ValueError('NPZ_EXPANDED_SIZE')
        for key, (dtype, shape) in LAYOUT.items():
            info = archive.getinfo(key + '.npy')
            with archive.open(info) as stream:
                if stream.read(6) != b'\x93NUMPY':
                    raise ValueError('NPY_MAGIC')
                version = tuple(stream.read(2))
                if version not in ((1, 0), (2, 0), (3, 0)):
                    raise ValueError('NPY_VERSION')
                width = 2 if version == (1, 0) else 4
                length = int.from_bytes(stream.read(width), 'little')
                if not 1 <= length <= 65536:
                    raise ValueError('NPY_HEADER_SIZE')
                header = ast.literal_eval(stream.read(length).decode('utf-8'))
            if header != {'descr': dtype, 'shape': shape, 'fortran_order': False}:
                raise ValueError('NPY_LAYOUT:' + key)
            count = 1
            for dim in shape:
                count *= dim
            if info.file_size != 8 + width + length + count * int(dtype[2:]):
                raise ValueError('NPY_PAYLOAD_LENGTH:' + key)
            receipt[key] = {'dtype': dtype, 'shape': list(shape), 'payload_decoded': False}
    return receipt


def static_coverage(cells, T):
    """Enumerate schedule metadata, not vector actions or native preparation."""
    unique, rows = set(), []
    for raw in cells:
        cell = dict(raw, formula=raw['order'], r=raw['R'] // raw['q'] if raw['R'] else None)
        schedule = primitive_time_schedule(cell, T=T)
        times = set(map(tuple, schedule['unique_primitive_times']))
        extra = []
        if cell['id'] in ('H4_B2_K2', 'H4_B3_K6'):
            half = T / cell['q'] / cell['r'] / 2
            extra = [(i, t) for i in range(cell['prefix'] + 1) for t in (half, -half)]
            times.update(extra)
        for index, time in times:
            unique.add(('one' if index == 0 else str(index - 1), time))
        random = cell['method'] in ('B2', 'B3')
        det = len(schedule['ordinary_one_outer_step']) * cell['q'] * (2 if random else 1)
        tail = 2 * cell['R'] * (cell['K'] + 1) if random else 0
        # Horner matvecs + polynomial endpoint + raw divide_b + scalar.
        native_records = det + tail + (3 * cell['R'] + 2 * cell['q'] if random else cell['q'])
        mp_records = 3 * (det // 2 + cell['R'] + cell['q']) if random else det + cell['q']
        rows.append({'cell_id': cell['id'], 'schedule': schedule,
            'registered_validation_times_v2': sorted(times),
            'explicit_event_half_undo_times': sorted(extra),
            'native_deterministic_actions': det, 'native_tail_matvecs': tail,
            'native_stage_records': native_records, 'MP_stage_records_per_precision': mp_records,
            'wrapper_instruction_upper_bounds': None})
    probes = sorted(unique)
    return {'policy': 'all unique primitive times; saved state + first/last sector columns',
        'unique_primitive_time_pairs': probes, 'unique_primitive_time_count': len(probes),
        'primitive_probe_count': 3, 'primitive_actions': 3 * len(probes), 'cells': rows,
        'explicit_event_groups': 4, 'control_probe_actions_planned': 100,
        'schedule_arithmetic': 'executed binary64, shared PF iterator; not an independent schedule oracle',
        'native_operation_lists_reconstructed': False,
        'actual_instruction_bounds_sealed': False, 'expected_bounds_for_launcher': None}


def build(root, *, cpu_candidate=None, output_relative=OUTPUT_PROPOSED):
    root = Path(root)
    freeze = read_json(root, PRIOR + '/preparation_source_freeze_v2.json')
    drafts = read_json(root, PRIOR + '/source_bound_preparations_v2.json')
    draft = next(p for p in drafts['preparations'] if p['kind'] == 'H4_LIMITED')
    if draft['source_commit'] != SOURCE_COMMIT or freeze['source_commit'] != SOURCE_COMMIT:
        raise ValueError('SOURCE_PROVENANCE')
    verify_sources(root, SOURCE_COMMIT, draft['source_hashes'])
    for name, expected in freeze['source_hashes'].items():
        if sha(safe_path(root, name).read_bytes()) != expected:
            raise ValueError('PREPARATION_SOURCE_CHANGED:' + name)
    binding = snapshot_binding(root)
    committed_input = subprocess.check_output(['git', 'show', SOURCE_COMMIT + ':' + SNAPSHOT], cwd=root)
    if sha(committed_input) != binding['sha256']:
        raise ValueError('COMMITTED_INPUT_BYTES')
    headers = layout_headers(safe_path(root, SNAPSHOT))
    plan = scope_plan('H4_LIMITED')
    if plan != draft['plan']:
        raise ValueError('FIXED_PLAN_CHANGED')
    receipts = []
    for name in ('run_v1/input_reference.json', 'saved_evidence_audit_v5.json',
                 'run_v1/terminal_status.json'):
        value = read_json(root, RESULT + '/' + name)
        if name.endswith('input_reference.json') and value['metadata'] != binding['metadata']:
            raise ValueError('ARCHIVED_INPUT_METADATA')
        receipts.append({'path': RESULT + '/' + name, 'sha256': sha(safe_path(root, RESULT + '/' + name).read_bytes())})
    for cell in plan['cells']:
        name = RESULT + '/run_v1/' + cell['id'] + '_correctness.json'
        if read_json(root, name)['cell'] != cell:
            raise ValueError('ARCHIVED_CELL_CHANGED:' + cell['id'])
        receipts.append({'path': name, 'sha256': sha(safe_path(root, name).read_bytes())})
    for row in receipts:
        data = subprocess.check_output(['git', 'show', RESULT_COMMIT + ':' + row['path']], cwd=root)
        if sha(data) != row['sha256']:
            raise ValueError('RESULT_COMMIT_BYTES:' + row['path'])
    coverage = static_coverage(plan['cells'], plan['T'])
    if coverage['primitive_actions'] > plan['caps_proposed']['primitive']:
        raise ValueError('STATIC_COVERAGE_CAP')
    for row in coverage['cells']:
        if (row['native_tail_matvecs'] > plan['caps_proposed']['tail_matvecs_per_cell']
                or row['native_deterministic_actions'] > plan['caps_proposed']['deterministic_actions_per_cell']
                or max(row['native_stage_records'], row['MP_stage_records_per_precision']) > 4096):
            raise ValueError('STATIC_STAGE_CAP')
    affinity = sorted(os.sched_getaffinity(0))
    if cpu_candidate is not None and (type(cpu_candidate) is not int or cpu_candidate not in affinity):
        raise ValueError('CPU_CANDIDATE_UNAVAILABLE')
    proposed_output = safe_path(root, output_relative)
    if proposed_output.exists():
        raise ValueError('OUTPUT_PROPOSAL_EXISTS')
    observed = environment()
    manifest = preparation('H4_LIMITED')
    manifest.update(source_commit=SOURCE_COMMIT, source_hashes=draft['source_hashes'],
        input_binding={k: binding[k] for k in ('path', 'sha256', 'metadata')},
        environment=observed,
        coverage_binding={'sealed': False, 'actual_rank': 12,
            'schedule_digest': digest(plan['cells']), 'expected_bounds': None})
    return {'schema': 'track_a_ax2b_h4_metadata_preflight_v3',
        'status': 'H4_METADATA_FIXED_NATIVE_COVERAGE_UNSEALED',
        'manifest': manifest, 'static_coverage': coverage, 'NPZ_headers': headers,
        'input_size_bytes': binding['size_bytes'], 'input_repository_commit': SOURCE_COMMIT,
        'historical_result_commit': RESULT_COMMIT,
        'historical_execution_base': 'b2e1bf65e21893b6c617223b42313623d3186f12',
        'archived_receipts': receipts, 'frozen_preparation_source_hashes_checked': len(freeze['source_hashes']),
        'runner_source_closure_hashes_checked': len(draft['source_hashes']),
        'host_observation': {'allowed_cpu_ids': affinity, 'cpu_candidate': cpu_candidate,
            'CPU_reserved_or_assigned': False, 'environment': observed,
            'environment_matches_prior_synthetic': observed == draft['environment'],
            'environment_role': 'proposed future H4 environment; does not replace old v5 execution'},
        'output_proposed_repository_path': output_relative, 'output_directory_created': False,
        'blockers': ['exact native instruction bounds from input-bound operation lists',
            'actual CPU assignment and exclusive output binding', 'separate explicit pinned launch grant'],
        'scientific_counts': dict.fromkeys(('numerical_array_loads', 'new_Hamiltonians', 'new_states',
            'signals', 'trajectories', 'circuit_builds', 'transpiles', 'compiles'), 0),
        'N': None, 'G': None, 'accuracy_eligibility': 'UNDETERMINED',
        'numerical_allowance_certified': False, 'H6_status': 'H6_NOT_AUTHORIZED',
        'contract_status': 'DRAFT_NOT_AUTHORIZATION', 'mandatory_stop': True,
        'next_stage_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New metadata JSON file only')
    parser.add_argument('--cpu-candidate', type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing metadata output is protected.')
    result = build(ROOT, cpu_candidate=args.cpu_candidate)
    # Strong guard: the entry point must remain stdlib-only.
    if any(n.split('.')[0] in {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion'} for n in sys.modules):
        raise RuntimeError('NUMERICAL_IMPORT_DURING_METADATA_PREFLIGHT')
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(result['status'] + ' / DRAFT_NOT_AUTHORIZATION / H6_NOT_AUTHORIZED')


if __name__ == '__main__':
    raise SystemExit(main())
