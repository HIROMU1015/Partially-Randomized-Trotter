"""Stdlib preflight and explicit grants; drafts never authorize execution.

This module neither creates an approval nor loads numerical arrays. Source
commit/blobs, local bytes, environment, fixed scope, input and exclusive output
are bound independently. H6 input generation is a separate, absent grant.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .ax2a_preparation import digest
from .ax2b_h4_contract_v5 import h4_plan, SNAPSHOT, SNAPSHOT_SHA, metadata_only, file_hash
from .ax2b_h6_contract import preparation_plan

RUNNER = 'scripts/resource_applicability/run_track_a_ax2b_bound_v3.py'
PHASES = ('input_reference', 'correctness', 'wrapper_cost')


def environment():
    return {'python': sys.version.split()[0], 'packages': {name: importlib.metadata.version(name)
            for name in ('numpy','scipy','mpmath','qiskit','openfermion')}}


def source_paths(root):
    root = Path(root)
    return sorted({str(p.relative_to(root)) for folder in ('src/trotterlib','src/trottertracks')
                   for p in (root/folder).rglob('*.py')} | {RUNNER})


def scope_plan(kind, actual_rank=None):
    if kind == 'H6_TECHNICAL':
        plan = preparation_plan(actual_rank)
        plan['T'] = plan['target']['T']
        plan['unresolved'] = ['input-generation scope/budget/grant', 'input binding', 'assigned resources', 'pilot grant']
        plan['implementation'] = 'dedicated_sector_native_backend_v2'
        plan['probe_policy_v2'] = 'all actual unique primitive times; saved state + first/last sector columns'
        return plan
    if kind != 'H4_LIMITED':
        raise ValueError('SCOPE')
    return {'kind': kind, 'T': .8, 'cells': h4_plan()['correctness_cells'], 'dps': [80,120],
            'caps_proposed': {'phase_wall_seconds': dict(zip(PHASES,(900,1800,300))),
              'total_wall_seconds': 3000, 'address_space_bytes': 8*2**30,
              'output_bytes': 512*2**20, 'log_bytes':65536, 'diagnostics':1024,
              'reference_matvec':10000, 'reference_matvec_per_action':20000,
              'deterministic_actions_per_cell':100000, 'tail_matvecs_per_cell':896,
              'primitive':2000, 'control_probe':200, 'untranspiled_instructions':1000000,
              'compile':0, 'trajectory':0, 'occurrence':0},
            'H6_tasks': [], 'H8_tasks': [], 'retry':False, 'resume':False, 'gpu':False,
            'N':None, 'G':None, 'numerical_allowance_certified':False,
            'accuracy_eligibility':'UNDETERMINED', 'mandatory_stop':True,
            'next_stage_authorized':False, 'contract_status':'DRAFT_NOT_AUTHORIZATION',
            'estimator_scope':'registered explicit representative events; no sampling/compile'}


def preparation(kind='H4_LIMITED'):
    if kind != 'H4_LIMITED':
        raise ValueError('H6_NOT_AUTHORIZED')
    return {'schema':'track_a_ax2b_bound_preparation_v3', 'kind':kind, 'plan':scope_plan(kind),
            'status':'H6_NOT_AUTHORIZED' if kind == 'H6_TECHNICAL' else 'H4_LIMITED_NOT_AUTHORIZED',
            'source_commit':None, 'source_hashes':None, 'environment':None,
            'input_binding':None, 'coverage_binding':None, 'assigned_resources':None,
            'science_authorized':False, 'launch_allowed':False, 'mandatory_stop':True,
            'contract_status':'DRAFT_NOT_AUTHORIZATION', 'execution_plan_sealed':False}


def safe_path(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('REPOSITORY_RELATIVE_PATH')
    result = (Path(root)/path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('PATH_ESCAPES_REPOSITORY')
    return result


def verify_input(root, binding, kind):
    if not isinstance(binding, dict) or set(binding) != {'path','sha256','metadata'}:
        raise ValueError('INPUT_BINDING_REQUIRED')
    path = safe_path(root,binding['path'])
    if path.stat().st_size > 16*2**20 or file_hash(path) != binding['sha256']:
        raise ValueError('INPUT_FILE_HASH_OR_SIZE')
    metadata = metadata_only(path)
    if metadata != binding['metadata']:
        raise ValueError('INPUT_METADATA_CHANGED')
    if kind == 'H4_LIMITED':
        if binding['path'] != SNAPSHOT or binding['sha256'] != SNAPSHOT_SHA:
            raise ValueError('H4_FROZEN_INPUT_REQUIRED')
        return 12
    hm = metadata.get('hamiltonian_metadata',{})
    if any(metadata.get(k) != v for k,v in {'model':'linear_H6','geometry_angstrom':1.,'basis':'sto-3g'}.items()):
        raise ValueError('H6_MODEL_METADATA')
    provenance = metadata.get('input_generation',{})
    for key,length in (('source_commit',40),('authorization_sha256',64)):
        value = provenance.get(key)
        if not isinstance(value,str) or len(value) != length or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('SEPARATE_INPUT_GENERATION_PROVENANCE:'+key)
    rank = hm.get('df_rank_actual')
    required = {'input_policy':'TOL_ONLY_NO_CONFIG_FALLBACK', 'df_tol_requested':1e-8,
                'decomposition_kwargs':{'truncation_threshold':1e-8}, 'final_rank_supplied':False,
                'coefficient_order':'decomposer_generation_order', 'subsequent_coefficient_cutoff':0.}
    if any(hm.get(k) != v for k,v in required.items()) or type(rank) is not int or not 2 <= rank <= 144:
        raise ValueError('H6_TOL_ONLY_PROVENANCE')
    sector = metadata.get('sector',{})
    if any(sector.get(k) != v for k,v in {'n_qubits':12,'nelec_alpha':3,'nelec_beta':3}.items()):
        raise ValueError('H6_SECTOR_METADATA')
    for k in ('input_tensors','one_body_correction','hermitization','df_truncation_value'):
        if k not in hm:
            raise ValueError('H6_INPUT_PROVENANCE_MISSING:'+k)
    return rank


def verify_sources(root, commit, hashes):
    if not isinstance(commit,str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('SOURCE_COMMIT_REQUIRED')
    if not isinstance(hashes,dict) or set(hashes) != set(source_paths(root)):
        raise ValueError('SOURCE_CLOSURE')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=root,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for name, expected in hashes.items():
        path = safe_path(root,name)
        if file_hash(path) != expected:
            raise ValueError('LOCAL_SOURCE_CHANGED:'+name)
        blob = subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('SOURCE_COMMIT_BYTES:'+name)


def validate_launch(root, manifest, authorization, *, requested, output, worker=False):
    # Deliberately before source/input/environment work and numerical imports.
    if requested is not True or not isinstance(authorization,dict) or authorization.get('approved_by_user') is not True:
        raise ValueError('SEPARATE_EXPLICIT_USER_GRANT_REQUIRED')
    if authorization.get('schema') != 'track_a_ax2b_bound_authorization_v3':
        raise ValueError('AUTHORIZATION_SCHEMA')
    if authorization.get('manifest_digest') != digest(manifest) or authorization.get('kind') != manifest.get('kind'):
        raise ValueError('AUTHORIZATION_MANIFEST_BINDING')
    if (manifest.get('schema') != 'track_a_ax2b_bound_preparation_v3'
            or manifest.get('execution_plan_sealed') is not True
            or manifest.get('science_authorized') is not False or manifest.get('launch_allowed') is not False
            or manifest.get('mandatory_stop') is not True or manifest.get('contract_status') != 'DRAFT_NOT_AUTHORIZATION'):
        raise ValueError('SEALED_PREPARATION_REQUIRED_NOT_AUTHORIZATION')
    kind = manifest['kind']
    if kind != 'H4_LIMITED':
        raise ValueError('H6_NOT_AUTHORIZED')
    expected_status = 'H6_NOT_AUTHORIZED' if kind == 'H6_TECHNICAL' else 'H4_LIMITED_NOT_AUTHORIZED'
    if manifest.get('status') != expected_status:
        raise ValueError('PREPARATION_STATUS')
    cpu = authorization.get('assigned_cpu')
    resources = manifest.get('assigned_resources')
    if type(cpu) is not int or cpu < 0 or resources != {'assigned_cpu':cpu,'science_workers':1,'blas_threads':1}:
        raise ValueError('ASSIGNED_RESOURCES_REQUIRED')
    if cpu not in os.sched_getaffinity(0):
        raise ValueError('ASSIGNED_CPU_UNAVAILABLE')
    output = Path(output).resolve()
    intended = manifest.get('intended_exclusive_output', {})
    if intended.get('absolute_path') != str(output) or safe_path(root, intended.get('repository_path', '')) != output:
        raise ValueError('INTENDED_OUTPUT_BINDING')
    if authorization.get('exclusive_output') != str(output) or (not worker and output.exists()):
        raise ValueError('EXCLUSIVE_OUTPUT_REQUIRED')
    if worker:
        marker = json.loads((output/'launch_binding.json').read_text())
        if marker != {'manifest_digest':digest(manifest),'authorization_digest':digest(authorization)}:
            raise ValueError('WORKER_LAUNCH_BINDING')
        if (output/'worker_claim.json').exists():
            raise ValueError('ONE_SHOT_WORKER_ALREADY_CLAIMED')
    if authorization.get('retry') is not False or authorization.get('resume') is not False:
        raise ValueError('NO_RETRY_OR_RESUME')
    verify_sources(root,manifest.get('source_commit'),manifest.get('source_hashes'))
    if manifest.get('environment') != environment():
        raise ValueError('ENVIRONMENT_CHANGED')
    rank = verify_input(root,manifest.get('input_binding'),kind)
    if manifest.get('plan') != scope_plan(kind,rank if kind == 'H6_TECHNICAL' else None):
        raise ValueError('FIXED_SCOPE_CHANGED')
    coverage = manifest.get('coverage_binding')
    if not isinstance(coverage,dict) or coverage.get('sealed') is not True or coverage.get('actual_rank') != rank:
        raise ValueError('ACTUAL_COVERAGE_UNSEALED')
    if coverage.get('schedule_digest') != digest(manifest['plan'].get('primitive_schedules',
                                            manifest['plan'].get('cells'))):
        raise ValueError('COVERAGE_SCHEDULE_BINDING')
    # Exact actual bounds are recomputed by the backend before any action/build.
    if not isinstance(coverage.get('expected_bounds'),dict):
        raise ValueError('INSTRUCTION_AND_PROBE_BOUNDS_REQUIRED')
    return cpu
