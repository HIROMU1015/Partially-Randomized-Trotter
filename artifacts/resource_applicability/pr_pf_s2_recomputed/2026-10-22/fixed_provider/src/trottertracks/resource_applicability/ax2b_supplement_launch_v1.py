"""H4-only supplement preflight. Metadata is never an execution grant."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .ax2a_preparation import digest
from .ax2b_bound_launch_v3 import environment, safe_path, verify_input, scope_plan
from .ax2b_h4_contract_v5 import file_hash
from .ax2b_coverage_binding_v3 import canonical_coverage

RUNNER = 'scripts/resource_applicability/run_track_a_ax2b_h4_supplement_v1.py'
PHASES = ('input_reference', 'primitive_validation', 'validation')
UNITS = ('S4_MP', 'EVENT_CONTROL')
OLD_RESULTS = 'a87cf25548a3b93262027ce780317872d7c4e883'
EXPECTED_COVERAGE_SHA = 'eacf9dec340e9b356090527597cfc1a277775350f050d5b50e2bb26e3c5e9609'


def source_paths(root):
    from .ax2b_bound_launch_v3 import source_paths as old_paths
    return sorted(set(old_paths(root)) | {RUNNER})


def plan(unit):
    if unit not in UNITS:
        raise ValueError('H4_SUPPLEMENT_UNIT')
    value = scope_plan('H4_LIMITED')
    value.update(kind='H4_SUPPLEMENT', unit=unit, old_results_commit=OLD_RESULTS,
                 selected_correctness_ids=['H4_B1_S4_q1','H4_B1_S4_q4'] if unit == 'S4_MP' else [],
                 explicit_groups=[{'cell_id':i,'order':k} for i in ('H4_B2_K2','H4_B3_K6')
                                  for k in (0,2)] if unit == 'EVENT_CONTROL' else [],
                 cache_policy={'scope':'cell/precision local; copies on get/store', 'entries':128,
                               'accounted_heap_bytes':64*2**20, 'exp_generations':64,
                               'polynomial_generations':4, 'lookups':4096, 'stage_records':4096},
                 prerequisite_scope='all original 8-cell schedules, 179 primitive/time pairs x 3 probes',
                 progress_policy='atomic immutable snapshots before/after generators; completed records only',
                 output_namespace='artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/',
                 estimator_scope='explicit 4 representative groups; no full event mean or measured shots',
                 H4_supplement_authorized=False, H6_status='H6_NOT_AUTHORIZED')
    caps = value['caps_proposed']
    # Proposed classical budgets, not an assignment or a speed prediction.
    caps.update(phase_wall_seconds=dict(zip(PHASES,(300,300,1800 if unit == 'S4_MP' else 300))),
                total_wall_seconds=2400 if unit == 'S4_MP' else 900,
                output_bytes=128*2**20, diagnostics=1536, progress_records=1024,
                primitive=537, reference_matvec=36, control_probe=0 if unit == 'S4_MP' else 100)
    return value


def preparation(unit):
    return {'schema':'track_a_ax2b_supplement_preparation_v1','kind':'H4_SUPPLEMENT','unit':unit,
            'plan':plan(unit),'status':'H4_SUPPLEMENT_NOT_AUTHORIZED',
            'source_commit':None,'source_hashes':None,'environment':None,'input_binding':None,
            'coverage_binding':None,'assigned_resources':None,'intended_exclusive_output':None,
            'execution_plan_sealed':False,'science_authorized':False,'launch_allowed':False,
            'mandatory_stop':True,'next_stage_authorized':False,'H6_status':'H6_NOT_AUTHORIZED',
            'contract_status':'DRAFT_NOT_AUTHORIZATION'}


def verify_sources(root, commit, hashes):
    if not isinstance(commit,str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('SOURCE_COMMIT_REQUIRED')
    if not isinstance(hashes,dict) or set(hashes) != set(source_paths(root)):
        raise ValueError('SUPPLEMENT_SOURCE_CLOSURE')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=root,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for name, expected in hashes.items():
        if file_hash(safe_path(root,name)) != expected:
            raise ValueError('LOCAL_SOURCE_CHANGED:'+name)
        data = subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('SOURCE_COMMIT_BYTES:'+name)


def validate_launch(root, manifest, grant, *, requested, output, worker=False):
    # New distinct grant schema: no consumed old grant can enter this runner.
    if requested is not True or not isinstance(grant,dict) or grant.get('approved_by_user') is not True:
        raise ValueError('SEPARATE_EXPLICIT_USER_GRANT_REQUIRED')
    if grant.get('schema') != 'track_a_ax2b_supplement_authorization_v1':
        raise ValueError('SUPPLEMENT_GRANT_SCHEMA')
    if (manifest.get('schema') != 'track_a_ax2b_supplement_preparation_v1'
            or manifest.get('kind') != 'H4_SUPPLEMENT' or manifest.get('unit') not in UNITS):
        raise ValueError('SUPPLEMENT_SCOPE')
    if grant.get('manifest_digest') != digest(manifest) or grant.get('unit') != manifest['unit']:
        raise ValueError('SUPPLEMENT_GRANT_BINDING')
    if (manifest.get('execution_plan_sealed') is not True or manifest.get('science_authorized') is not False
            or manifest.get('launch_allowed') is not False or manifest.get('mandatory_stop') is not True
            or manifest.get('next_stage_authorized') is not False
            or manifest.get('status') != 'H4_SUPPLEMENT_NOT_AUTHORIZED'
            or manifest.get('contract_status') != 'DRAFT_NOT_AUTHORIZATION'
            or manifest.get('H6_status') != 'H6_NOT_AUTHORIZED'):
        raise ValueError('SEALED_PREPARATION_REQUIRED_NOT_AUTHORIZATION')
    if grant.get('retry') is not False or grant.get('resume') is not False:
        raise ValueError('NO_RETRY_OR_RESUME')
    if canonical_coverage(manifest.get('plan')) != canonical_coverage(plan(manifest['unit'])):
        raise ValueError('FIXED_SUPPLEMENT_PLAN')
    cpu = grant.get('assigned_cpu')
    if type(cpu) is not int or cpu not in os.sched_getaffinity(0):
        raise ValueError('EXPLICIT_AVAILABLE_CPU_REQUIRED')
    if canonical_coverage(manifest.get('assigned_resources')) != canonical_coverage(
            {'assigned_cpu':cpu,'science_workers':1,'blas_threads':1}):
        raise ValueError('ASSIGNED_RESOURCES_REQUIRED')
    output = Path(output).resolve()
    intended = manifest.get('intended_exclusive_output') or {}
    repository_path = intended.get('repository_path','')
    if (not repository_path.startswith(manifest['plan']['output_namespace'])
            or intended.get('absolute_path') != str(output) or safe_path(root,repository_path) != output
            or grant.get('exclusive_output') != str(output) or (not worker and output.exists())):
        raise ValueError('NEW_EXCLUSIVE_OUTPUT_REQUIRED')
    if worker:
        marker = output/'launch_binding.json'
        if marker.stat().st_size > 8192 or json.loads(marker.read_text()) != {
                'manifest_digest':digest(manifest),'authorization_digest':digest(grant)}:
            raise ValueError('WORKER_LAUNCH_BINDING')
        if (output/'worker_claim.json').exists():
            raise ValueError('ONE_SHOT_WORKER_ALREADY_CLAIMED')
    verify_sources(root,manifest.get('source_commit'),manifest.get('source_hashes'))
    if manifest.get('environment') != environment():
        raise ValueError('ENVIRONMENT_CHANGED')
    verify_input(root,manifest.get('input_binding'),'H4_LIMITED')
    coverage = manifest.get('coverage_binding')
    if (not isinstance(coverage,dict) or coverage.get('sealed') is not True
            or type(coverage.get('actual_rank')) is not int or coverage['actual_rank'] != 12
            or coverage.get('schedule_digest') != digest(manifest['plan']['cells'])
            or hashlib.sha256(canonical_coverage(coverage.get('expected_bounds'))).hexdigest() != EXPECTED_COVERAGE_SHA):
        raise ValueError('FROZEN_FULL_COVERAGE_REQUIRED')
    return cpu
