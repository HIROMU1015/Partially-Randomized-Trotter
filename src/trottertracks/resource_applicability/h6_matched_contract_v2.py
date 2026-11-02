"""Metadata-only contract. A sealed plan is never permission to compute."""
from __future__ import annotations
import hashlib
import os
import subprocess
from pathlib import Path
from .ax2a_preparation import digest
from .ax2b_h6_pilot_contract_v2 import verify_parent, environment, safe_path, file_hash
from .ax2b_h6_contract import primitive_time_schedule

KIND = 'H6_MATCHED_SIGNAL_RESOURCE_V2'
RUNNER = 'scripts/resource_applicability/run_track_a_h6_matched_v2.py'
PREPARER = 'scripts/resource_applicability/prepare_track_a_h6_matched_v2.py'
AUDITOR = 'scripts/resource_applicability/audit_track_a_h6_matched_v2.py'
NAMESPACE = 'artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/'
PREPARATION_PATH = 'artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/'
# Reuse the exact v1 scientific plan, accounting, port and sampling namespace.
from .h6_matched_contract_v1 import EPSILONS, cells, plan, resources
from .h6_matched_contract_v1 import RUNNER as V1_RUNNER, PREPARER as V1_PREPARER, AUDITOR as V1_AUDITOR


def infrastructure():
    return dict(schema='h6_matched_limit_inheritance_v2',
        coordinator_as_soft_bytes=2*2**30, coordinator_as_hard_bytes=8*2**30,
        signal_as_soft_bytes=8*2**30, signal_as_hard_bytes=8*2**30,
        cost_as_soft_bytes=8*2**30, cost_as_hard_bytes=8*2**30,
        allocation_enforced_by='RLIMIT_AS soft limit; hard limit is inherited ceiling',
        scientific_plan_reused='h6_matched_contract_v1.plan',
        wall_caps_added=False, science_authorized=False)


def check_inherited_limits():
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and hard < 8*2**30:
        raise ValueError('MATCHED_AS_INHERITED_HARD_TOO_SMALL')
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    if hard != resource.RLIM_INFINITY and hard < 2*2**30:
        raise ValueError('MATCHED_FSIZE_INHERITED_HARD_TOO_SMALL')


def source_paths(root):
    root = Path(root)
    return sorted({str(p.relative_to(root)) for d in ('src/trotterlib', 'src/trottertracks')
                   for p in (root/d).rglob('*.py')} | {RUNNER, PREPARER, AUDITOR, V1_RUNNER, V1_PREPARER, V1_AUDITOR})


def verify_sources(root, commit, hashes):
    if not isinstance(commit, str) or len(commit) != 40 or any(x not in '0123456789abcdef' for x in commit):
        raise ValueError('MATCHED_SOURCE_COMMIT')
    if not isinstance(hashes, dict) or set(hashes) != set(source_paths(root)):
        raise ValueError('MATCHED_SOURCE_CLOSURE')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'], cwd=root, check=True,
                   stdout=subprocess.DEVNULL)
    for p, sha in hashes.items():
        blob = subprocess.check_output(['git','show',commit+':'+p], cwd=root)
        if file_hash(safe_path(root,p)) != sha or hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError('MATCHED_SOURCE_CHANGED:'+p)


def preparation():
    return dict(schema='h6_matched_preparation_v2', kind=KIND, plan=plan(),
        execution_infrastructure=infrastructure(),
        science_plan_digest_v1=digest(plan()),
        predecessor_preparation_commit='12183db780bef14c25c93ba6523c44c52edd7828',
        predecessor_stop_commit='274b984971f9afc3425f3b3d1ce36e1c478150ea',
        source_commit=None, source_hashes=None, input_identity=None, environment=None,
        assigned_resources=None, exclusive_output=None, execution_plan_sealed=False,
        status='PREPARATION_ONLY_NOT_AUTHORIZED', science_authorized=False, launch_allowed=False,
        H6_status='H6_NOT_AUTHORIZED', contract_status='DRAFT_NOT_AUTHORIZATION',
        mandatory_stop=True, next_stage_authorized=False)


def validate_launch(root, manifest, grant, output, *, requested=False):
    if requested is not True or not isinstance(grant, dict) or grant.get('approved_by_user') is not True:
        raise ValueError('NEW_MATCHED_RUN_GRANT_REQUIRED')
    if grant.get('schema') != 'h6_matched_authorization_v2' or grant.get('kind') != KIND:
        raise ValueError('MATCHED_GRANT_SCHEMA')
    expected = preparation()
    for key in ('execution_infrastructure','science_plan_digest_v1',
                'predecessor_preparation_commit','predecessor_stop_commit'):
        if manifest.get(key) != expected[key]:
            raise ValueError('MATCHED_V2_INFRASTRUCTURE_BINDING:'+key)
    check_inherited_limits()
    for k in ('schema','kind','status','science_authorized','launch_allowed','H6_status',
              'contract_status','mandatory_stop','next_stage_authorized'):
        if type(manifest.get(k)) is not type(expected[k]) or manifest[k] != expected[k]:
            raise ValueError('MATCHED_MANIFEST_FLAG:'+k)
    if manifest.get('execution_plan_sealed') is not True or digest(manifest.get('plan')) != digest(plan()):
        raise ValueError('MATCHED_FIXED_PLAN')
    if (grant.get('manifest_digest') != digest(manifest) or grant.get('source_commit') != manifest.get('source_commit')
            or grant.get('one_shot') is not True or grant.get('retry') is not False or grant.get('resume') is not False):
        raise ValueError('MATCHED_GRANT_BINDING')
    if manifest.get('assigned_resources') != resources(grant.get('assigned_cpus')):
        raise ValueError('MATCHED_RESOURCES')
    intended = manifest.get('exclusive_output') or {}
    name = intended.get('repository_path', '')
    path = Path(output).resolve()
    if (not name.startswith(NAMESPACE) or safe_path(root,name) != path or intended.get('absolute_path') != str(path)
            or grant.get('exclusive_output') != str(path) or path.exists()):
        raise ValueError('MATCHED_EXCLUSIVE_OUTPUT')
    verify_sources(root, manifest.get('source_commit'), manifest.get('source_hashes'))
    if manifest.get('input_identity') != verify_parent(root):
        raise ValueError('MATCHED_INPUT_BINDING')
    if manifest.get('environment') != environment():
        raise ValueError('MATCHED_ENVIRONMENT')
    return manifest['assigned_resources']
