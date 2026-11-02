"""Stdlib-only H4 input binding and fail-closed launch contract.

Preparation reads the NPZ metadata string and file bytes, never scientific
arrays. A prepared manifest is not execution authorization.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

from .ax2a_preparation import digest, BASE_COMMIT
from .ax2b_preflight import expanded_pilot_proposal

SNAPSHOT = 'artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz'
SNAPSHOT_SHA = '3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a'
HAM_SHA = 'de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424'
STATE_SHA = '31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd'
VECTOR_SHA = 'c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63'
RUNNER = 'scripts/resource_applicability/run_track_a_ax2b_h4.py'
PACKAGES = ('numpy', 'scipy', 'qiskit', 'openfermion')


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def metadata_only(path):
    """Bounded parsing of scalar Unicode metadata_json.npy, stdlib only."""
    with zipfile.ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if i.filename == 'metadata_json.npy']
        if len(entries) != 1 or entries[0].file_size > 1024 * 1024:
            raise ValueError('SNAPSHOT_METADATA_LAYOUT')
        with archive.open(entries[0]) as stream:
            if stream.read(6) != b'\x93NUMPY':
                raise ValueError('NPY_MAGIC')
            version = tuple(stream.read(2))
            if version not in ((1, 0), (2, 0), (3, 0)):
                raise ValueError('NPY_VERSION')
            size = 2 if version == (1, 0) else 4
            header_length = int.from_bytes(stream.read(size), 'little')
            if not 1 <= header_length <= 65536:
                raise ValueError('NPY_HEADER_SIZE')
            header = ast.literal_eval(stream.read(header_length).decode('utf-8'))
            dtype = header.get('descr', '')
            if header.get('shape') != () or header.get('fortran_order') is not False:
                raise ValueError('NPY_METADATA_SHAPE')
            if not isinstance(dtype, str) or dtype[:2] not in ('<U', '>U'):
                raise ValueError('NPY_METADATA_DTYPE')
            length = int(dtype[2:])
            if not 1 <= length <= 200000:
                raise ValueError('NPY_METADATA_SIZE')
            payload = stream.read(4 * length + 1)
            if len(payload) != 4 * length:
                raise ValueError('NPY_METADATA_PAYLOAD')
    return json.loads(payload.decode('utf-32-le' if dtype[0] == '<' else 'utf-32-be').rstrip('\x00'))


def snapshot_binding(root):
    path = Path(root) / SNAPSHOT
    if file_hash(path) != SNAPSHOT_SHA:
        raise ValueError('H4_SNAPSHOT_HASH')
    metadata = metadata_only(path)
    for key, value in (('hamiltonian_hash', HAM_SHA), ('state_hash', STATE_SHA),
                       ('state_vector_hash', VECTOR_SHA)):
        if metadata.get(key) != value:
            raise ValueError('H4_METADATA_IDENTITY:' + key)
    return {'path': SNAPSHOT, 'sha256': SNAPSHOT_SHA, 'size_bytes': path.stat().st_size,
            'metadata': metadata, 'numerical_arrays_loaded': False,
            'primitive_sector_verified': False, 'basis_order_verified': False,
            'reference_allowance_certified': False}


def h4_plan():
    proposal = expanded_pilot_proposal()
    return {'schema': 'track_a_ax2b_h4_plan_v3', 'T': 0.8,
            'target': {'model': 'linear_H4', 'geometry_angstrom': 1.0,
                       'basis': 'sto-3g', 'df_rank': 12,
                       'state_policy': 'legacy saved normalized state, no solve'},
            'correctness_cells': proposal['H4_correctness_cells'],
            'wrapper_tasks': [t for t in proposal['wrapper_tasks'] if t['cell']['system'] == 'H4'],
            'compiler': proposal['compiler_proposal'],
            'gates': {'normalization_tolerance': 1e-12, 'leakage_tolerance': 1e-12,
                      'agreement_tolerance': 1e-9, 'reference_discrepancy_tolerance': 1e-10},
            'caps': {'cpu_cores': 1, 'blas_threads': 1, 'science_workers': 1,
                     'address_space_bytes': 8589934592, 'phase_wall_seconds': 900,
                     'total_wall_seconds': 2700, 'output_bytes': 536870912,
                     'compile_calls': 28, 'trajectory_samples': 4, 'occurrence_samples': 16,
                     'tail_matvecs_per_signal': 896, 'deterministic_actions_per_signal': 100000,
                     'reference_matvecs_per_action': 20000,
                     'reference_matvecs_total': 10000,
                     'primitive_validation_actions': 2000, 'control_probe_actions': 200,
                     'untranspiled_instructions': 1000000, 'transpiled_instructions': 5000000},
            'phase_order': ['input_reference', 'correctness', 'wrapper_cost'],
            'H6_tasks': [], 'H8_tasks': [], 'gpu': False,
            'scope': 'H4 technical validation/profile only; no winner, shot, energy-accuracy or model-fit claims',
            'retry': False, 'resume': False}


def source_hashes(root):
    root = Path(root)
    paths = sorted(set(root.joinpath('src/trotterlib').rglob('*.py')) |
                   set(root.joinpath('src/trottertracks').rglob('*.py')) | {root / RUNNER})
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def environment_identity():
    return {'python': sys.version.split()[0],
            'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}


def preparation(root):
    return {'schema': 'track_a_ax2b_h4_preparation_v3',
            'status': 'AX2B_H4_RUNNER_PREPARED_SCIENCE_NOT_AUTHORIZED',
            'science_authorized': False, 'launch_allowed': False, 'mandatory_stop': True,
            'base_commit': BASE_COMMIT, 'source_freeze_kind': 'local_uncommitted_byte_hashes',
            'plan': h4_plan(), 'snapshot': snapshot_binding(root),
            'source_hashes': source_hashes(root), 'environment': environment_identity(),
            'assigned_cpu': None, 'new_scientific_calculation_count': 0}


def validate_launch(root, manifest, authorization, *, requested, output, worker=False):
    """Run before scientific imports, array loading or process creation."""
    if requested is not True or authorization.get('approved_by_user') is not True:
        raise ValueError('EXPLICIT_SCIENCE_AUTHORIZATION_REQUIRED')
    if authorization.get('schema') != 'track_a_ax2b_h4_authorization_v3':
        raise ValueError('AUTHORIZATION_SCHEMA')
    if authorization.get('manifest_digest') != digest(manifest):
        raise ValueError('AUTHORIZATION_MANIFEST_BINDING')
    if manifest.get('schema') != 'track_a_ax2b_h4_preparation_v3' or manifest.get('plan') != h4_plan():
        raise ValueError('PLAN_CHANGED')
    if (manifest.get('science_authorized') is not False or manifest.get('launch_allowed') is not False
            or manifest.get('mandatory_stop') is not True or manifest.get('assigned_cpu') is not None
            or manifest.get('base_commit') != BASE_COMMIT
            or manifest.get('source_freeze_kind') != 'local_uncommitted_byte_hashes'
            or manifest.get('status') != 'AX2B_H4_RUNNER_PREPARED_SCIENCE_NOT_AUTHORIZED'
            or type(manifest.get('new_scientific_calculation_count')) is not int
            or manifest.get('new_scientific_calculation_count') != 0):
        raise ValueError('PREPARATION_IS_NOT_AUTHORIZATION')
    if manifest.get('source_hashes') != source_hashes(root):
        raise ValueError('SCIENTIFIC_SOURCE_CHANGED')
    if manifest.get('environment') != environment_identity():
        raise ValueError('ENVIRONMENT_CHANGED')
    if manifest.get('snapshot') != snapshot_binding(root):
        raise ValueError('SNAPSHOT_BINDING_CHANGED')
    cpu = authorization.get('assigned_cpu')
    if type(cpu) is not int or cpu < 0:
        raise ValueError('ASSIGNED_CPU_REQUIRED')
    output = Path(output).resolve()
    if authorization.get('exclusive_output') != str(output) or (output.exists() and not worker):
        raise ValueError('EXCLUSIVE_OUTPUT_BINDING')
    if worker:
        marker = json.loads((output / 'launch_binding.json').read_text())
        if marker != {'manifest_digest': digest(manifest), 'authorization_digest': digest(authorization)}:
            raise ValueError('WORKER_LAUNCH_BINDING')
    # Authorizations cover only this frozen H4 scope and budget. They cannot
    # turn a preparation record into H6/H8 or main-campaign authorization.
    return cpu
