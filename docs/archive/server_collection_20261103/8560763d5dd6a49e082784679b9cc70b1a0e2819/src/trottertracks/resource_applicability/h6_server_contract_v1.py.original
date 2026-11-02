"""Resource-only Phase A contract; the frozen H6 scientific plan is unchanged."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from .ax2a_preparation import digest
from .h6_matched_contract_v1 import plan,environment,verify_parent,file_hash,safe_path
from .h6_matched_reuse_v3 import evidence,verify_evidence,committed_hashes
from .h6_server_resources_v1 import resources,observation_policy,fresh_resources

SCIENCE_SOURCE='5a903aea2492c2d211c1e15e7737df0d50694c12'
PUBLIC_COMMIT='054bb7b2b2ca788b47ebdc5c14c115f24c8b6379'
KIND='H6_SERVER_PHASE_A_ONLY_V1'
RUNNER='scripts/resource_applicability/run_track_a_h6_server_phase_a_v1.py'
PREPARER='scripts/resource_applicability/prepare_track_a_h6_server_v1.py'
AUDITOR='scripts/resource_applicability/audit_track_a_h6_server_v1.py'
NAMESPACE='artifacts/resource_applicability/track_a_h6_server_phase_a_v1/2026-10-11/'
PREPARATION='artifacts/resource_applicability/track_a_h6_server_preparation_v1/2026-10-11/'
OLD_MANIFEST='artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/sealed_manifest_v3.json'


def source_paths(root):
    old=json.loads((Path(root)/OLD_MANIFEST).read_text())['source_hashes']
    return sorted(set(old)|{RUNNER,PREPARER,AUDITOR,
        'src/trottertracks/resource_applicability/h6_server_contract_v1.py',
        'src/trottertracks/resource_applicability/h6_server_execution_v1.py',
        'src/trottertracks/resource_applicability/h6_server_resources_v1.py',
        'src/trottertracks/resource_applicability/h6_server_quota_v1.py'})


def verify_science(root):
    committed_hashes(root,PUBLIC_COMMIT,{OLD_MANIFEST:file_hash(Path(root)/OLD_MANIFEST)})
    hashes=json.loads((Path(root)/OLD_MANIFEST).read_text())['source_hashes']
    if len(hashes)!=224:raise ValueError('SCIENCE_CLOSURE_COUNT')
    committed_hashes(root,SCIENCE_SOURCE,hashes)
    return digest(hashes)


def phase_scope(registry):
    missing=registry['missing'];first=[x for x in missing if x['index']==447]
    if (len(missing)!=143 or len(first)!=1 or first[0]['task']['cell_id']!='B1_p19_q8_4th'
        or first[0]['task']['phase']!='exploration' or first[0]['task']['replica']!=0
        or first[0]['task']['ordinary'] is not False):raise ValueError('PHASE_A_FIXED_SELECTION')
    return dict(index=447,task=first[0]['task'],task_digest=digest(first[0]['task']),
        primary_wrappers=2,axes=['cosine','sine'],remaining_indices=[x['index'] for x in missing if x['index']!=447],
        remaining_groups=142,phase_b_authorized=False,confirmation_authorized=False,
        automatic_next_phase=False,task_caps=dict(compile=2,control_probe=4,trajectory=0,occurrence=0))


def flags():
    return dict(H6_status='H6_NOT_AUTHORIZED',contract_status='DRAFT_NOT_AUTHORIZATION',
        science_authorized=False,launch_allowed=False,mandatory_stop=True,next_stage_authorized=False)


def validate_manifest(m):
    if m.get('schema')!='h6_server_phase_a_manifest_v1' or m.get('kind')!=KIND:raise ValueError('SERVER_MANIFEST_SCHEMA')
    for k,v in flags().items():
        if type(m.get(k)) is not type(v) or m[k]!=v:raise ValueError('SERVER_MANIFEST_FLAG:'+k)
    if m.get('execution_plan_sealed') is not True or digest(m.get('plan'))!=digest(plan()):raise ValueError('SERVER_SCIENCE_PLAN')
    if m.get('science_plan_digest')!=digest(plan()):raise ValueError('SERVER_SCIENCE_PLAN_DIGEST')
    if (m['science_source']!=SCIENCE_SOURCE or m['phase_scope']!=phase_scope(m['reuse_registry'])
        or m['assigned_resources']!=resources(m['assigned_resources']['assigned_cpus'])
        or m['observation_policy']!=observation_policy()):raise ValueError('SERVER_FIXED_SCOPE')
    if m['environment']!=m['reuse_registry']['environment']:raise ValueError('SERVER_LEGACY_ENVIRONMENT')


def grant_draft(m):
    return dict(schema='h6_server_phase_a_authorization_v1',kind=KIND,
        approved_by_user=False,server_usage_approved_by_user=False,authorization_only=True,
        manifest_digest=digest(m),source_commit=m['source_commit'],
        profile_sha256=m['profile']['sha256'],task_digest=m['phase_scope']['task_digest'],
        reuse_registry_digest=digest(m['reuse_registry']),assigned_cpus=m['assigned_resources']['assigned_cpus'],
        exclusive_output=m['exclusive_output']['absolute_path'],one_shot=True,retry=False,resume=False,
        phase_b_authorized=False,confirmation_authorized=False,H8_authorized=False,
        mandatory_stop=True,next_stage_authorized=False)


def validate_grant(m,g,output):
    expected=grant_draft(m);expected.update(approved_by_user=True,server_usage_approved_by_user=True)
    if g!=expected or str(Path(output).resolve())!=g['exclusive_output']:
        raise ValueError('NEW_APPROVED_PHASE_A_ONLY_GRANT_REQUIRED')


def verify_profile(root,m):
    ref=m['profile'];path=safe_path(root,ref['repository_path'])
    if file_hash(path)!=ref['sha256']:raise ValueError('SERVER_PROFILE_CHANGED')
    p=json.loads(path.read_text())
    if (p['environment']!=m['environment'] or Path(sys.executable).resolve()!=Path(p['python']).resolve()
        or Path(sys.prefix).resolve()!=Path(p['venv']).resolve()):
        raise ValueError('SERVER_PROFILE_PYTHON')
    for item in p['critical_file_hashes']:
        path=Path(item['path'])
        if not path.is_relative_to('/home/AbeHiromu') or file_hash(path)!=item['sha256']:
            raise ValueError('SERVER_NATIVE_BINARY_OR_SOURCE:'+str(path))
    return p


def verify_bindings(root,m):
    validate_manifest(m);verify_science(root)
    if set(m['source_hashes'])!=set(source_paths(root)):raise ValueError('SERVER_SOURCE_CLOSURE')
    committed_hashes(root,m['source_commit'],m['source_hashes'])
    verify_evidence(root,m['reuse_registry'])
    if verify_parent(root)!=m['input_identity'] or environment()!=m['environment']:raise ValueError('SERVER_INPUT_ENVIRONMENT')
    verify_profile(root,m)


def validate_launch(root,m,g,output,*,requested=False):
    if requested is not True:raise ValueError('EXPLICIT_PHASE_A_EXECUTION_REQUIRED')
    validate_manifest(m);validate_grant(m,g,output);verify_bindings(root,m)
    if m.get('repository_root')!=str(Path(root).resolve()):raise ValueError('SERVER_WORKTREE_BINDING')
    intended=m['exclusive_output'];out=Path(output).resolve()
    if not intended['repository_path'].startswith(NAMESPACE) or safe_path(root,intended['repository_path'])!=out:
        raise ValueError('SERVER_EXCLUSIVE_NAMESPACE')
    fresh=fresh_resources(m['assigned_resources'],m['observation_policy'],out)
    control=safe_path(root,m['one_shot_lock'])
    if control.exists() or control.is_symlink():raise ValueError('SERVER_ONE_SHOT_CONSUMED')
    return fresh
