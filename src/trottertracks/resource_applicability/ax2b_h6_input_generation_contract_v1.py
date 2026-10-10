"""Stdlib H6 input-generation proposal and gate; preparation never grants work."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .ax2a_preparation import digest
from .ax2b_bound_launch_v3 import safe_path
from .ax2b_h4_contract_v5 import file_hash

RUNNER = 'scripts/resource_applicability/run_track_a_ax2b_h6_input_generation_v1.py'
PHASES = ('integrals', 'df_decomposition', 'state_snapshot')
OUTPUT_NAMESPACE = 'artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/'
PACKAGES = ('numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion', 'openfermionpyscf', 'pyscf', 'numba', 'h5py')


def plan():
    return {'schema': 'track_a_ax2b_h6_input_generation_plan_v1',
            'target': {'model': 'linear_H6', 'geometry_angstrom': 1., 'basis': 'sto-3g',
                       'geometry': [['H', [0., 0., float(i)]] for i in range(6)],
                       'charge': 0, 'multiplicity': 1, 'n_qubits': 12,
                       'nelec_alpha': 3, 'nelec_beta': 3, 'sector_dimension': 400},
            'integrals': {'provider': 'openfermionpyscf prepare/compute helpers; no MolecularData.save',
                          'scf': 'RHF', 'conv_tol': 1e-9, 'max_cycle': 50, 'threads': 1,
                          'unit': 'Angstrom', 'symmetry': False, 'run_fci': False,
                          'spin_order': 'even alpha / odd beta',
                          'two_body': 'spinorb_from_spatial then factor 1/2, InteractionOperator convention'},
            'df_policy': {'input_policy': 'TOL_ONLY_NO_CONFIG_FALLBACK', 'df_tol': 1e-8,
                          'final_rank_supplied': False, 'coefficient_order': 'decomposer_generation_order',
                          'subsequent_coefficient_cutoff': 0., 'hermitization_tolerance': 1e-10,
                          'accepted_actual_rank': [2, 144]},
            'state_policy': {'solver': 'eigsh', 'k': 1, 'which': 'SA', 'tol': 1e-12,
                             'maxiter': 1000, 'ncv': 40, 'initial': 'fixed Hartree-Fock occupation',
                             'backend': 'numba', 'num_threads': 1, 'block_chunk_size': 1,
                             'residual_absolute_gate': 1e-9, 'residual_relative_gate': 1e-10,
                             'saved_norm_gate': 1e-12,
                             'normalization': 'one binary64 normalization of returned sector vector',
                             'global_phase_policy': 'largest_sector_amplitude_real_positive_v1',
                             'ground_state_certified': False},
            'caps_proposed': {'phase_wall_seconds': dict(zip(PHASES, (900, 300, 900))),
                              'total_wall_seconds': 2100, 'address_space_bytes': 8*2**30,
                              'output_bytes': 128*2**20, 'snapshot_expanded_bytes': 16*2**20,
                              'log_bytes': 65536, 'diagnostics': 512, 'progress_records': 256,
                              'solver_matvec': 10000, 'solver_matvec_per_action': 10000,
                              'integral_build': 1, 'df_decomposition': 1,
                              'trajectory': 0, 'occurrence': 0, 'compile': 0},
            'output_namespace': OUTPUT_NAMESPACE, 'retry': False, 'resume': False, 'gpu': False,
            'H6_pilot_authorized': False, 'H8_tasks': [], 'N': None, 'G': None,
            'numerical_allowance_certified': False, 'accuracy_eligibility': 'UNDETERMINED',
            'mandatory_stop': True, 'next_stage_authorized': False,
            'contract_status': 'DRAFT_NOT_AUTHORIZATION'}


def preparation():
    return {'schema': 'track_a_ax2b_h6_input_generation_preparation_v1', 'kind': 'H6_INPUT_GENERATION',
            'plan': plan(), 'status': 'H6_INPUT_GENERATION_NOT_AUTHORIZED',
            'H6_status': 'H6_NOT_AUTHORIZED', 'source_commit': None, 'source_hashes': None,
            'environment': None, 'assigned_resources': None, 'intended_exclusive_output': None,
            'execution_plan_sealed': False, 'science_authorized': False, 'launch_allowed': False,
            'mandatory_stop': True, 'next_stage_authorized': False,
            'contract_status': 'DRAFT_NOT_AUTHORIZATION'}


def source_paths(root):
    root = Path(root)
    return sorted({str(p.relative_to(root)) for folder in ('src/trotterlib', 'src/trottertracks')
                   for p in (root/folder).rglob('*.py')} | {RUNNER})


def environment():
    # Version lookup and source bytes only; no scientific imports or execution.
    distribution = importlib.metadata.distribution('openfermionpyscf')
    helper = Path(distribution.locate_file('openfermionpyscf/_run_pyscf.py'))
    return {'python': sys.version.split()[0], 'packages': {n: importlib.metadata.version(n) for n in PACKAGES},
            'integral_provider_source_sha256': file_hash(helper)}


def verify_sources(root, commit, hashes):
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('SOURCE_COMMIT_REQUIRED')
    if not isinstance(hashes, dict) or set(hashes) != set(source_paths(root)):
        raise ValueError('INPUT_GENERATION_SOURCE_CLOSURE')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=root, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for name, expected in hashes.items():
        if file_hash(safe_path(root, name)) != expected:
            raise ValueError('LOCAL_SOURCE_CHANGED:' + name)
        blob = subprocess.check_output(['git', 'show', commit+':'+name], cwd=root)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('SOURCE_COMMIT_BYTES:' + name)


def validate_launch(root, manifest, grant, *, requested, output, worker=False):
    if requested is not True or not isinstance(grant, dict) or grant.get('approved_by_user') is not True:
        raise ValueError('SEPARATE_INPUT_GENERATION_GRANT_REQUIRED')
    if grant.get('schema') != 'track_a_ax2b_h6_input_generation_authorization_v1':
        raise ValueError('INPUT_GENERATION_GRANT_SCHEMA')
    if (manifest.get('schema') != 'track_a_ax2b_h6_input_generation_preparation_v1'
            or manifest.get('kind') != 'H6_INPUT_GENERATION'
            or grant.get('kind') != manifest['kind'] or grant.get('manifest_digest') != digest(manifest)):
        raise ValueError('INPUT_GENERATION_GRANT_BINDING')
    required = {'status': 'H6_INPUT_GENERATION_NOT_AUTHORIZED', 'H6_status': 'H6_NOT_AUTHORIZED',
                'execution_plan_sealed': True, 'science_authorized': False, 'launch_allowed': False,
                'mandatory_stop': True, 'next_stage_authorized': False, 'contract_status': 'DRAFT_NOT_AUTHORIZATION'}
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in required.items()):
        raise ValueError('SEALED_INPUT_PREPARATION_REQUIRED')
    if digest(manifest.get('plan')) != digest(plan()):
        raise ValueError('FIXED_INPUT_GENERATION_PLAN')
    if grant.get('retry') is not False or grant.get('resume') is not False:
        raise ValueError('NO_RETRY_OR_RESUME')
    cpu = grant.get('assigned_cpu')
    if type(cpu) is not int or cpu not in os.sched_getaffinity(0):
        raise ValueError('EXPLICIT_AVAILABLE_CPU_REQUIRED')
    if digest(manifest.get('assigned_resources')) != digest({'assigned_cpu': cpu, 'science_workers': 1, 'blas_threads': 1}):
        raise ValueError('ASSIGNED_INPUT_RESOURCES_REQUIRED')
    output = Path(output).resolve()
    intended = manifest.get('intended_exclusive_output') or {}
    name = intended.get('repository_path', '')
    if (not name.startswith(OUTPUT_NAMESPACE) or safe_path(root, name) != output
            or intended.get('absolute_path') != str(output) or grant.get('exclusive_output') != str(output)
            or (not worker and output.exists())):
        raise ValueError('NEW_EXCLUSIVE_INPUT_OUTPUT_REQUIRED')
    if worker:
        marker = output/'launch_binding.json'
        if marker.stat().st_size > 8192 or json.loads(marker.read_text()) != {
                'manifest_digest': digest(manifest), 'authorization_digest': digest(grant)}:
            raise ValueError('WORKER_LAUNCH_BINDING')
        if (output/'worker_claim.json').exists():
            raise ValueError('ONE_SHOT_WORKER_ALREADY_CLAIMED')
    verify_sources(root, manifest.get('source_commit'), manifest.get('source_hashes'))
    if manifest.get('environment') != environment():
        raise ValueError('INPUT_ENVIRONMENT_CHANGED')
    return cpu
