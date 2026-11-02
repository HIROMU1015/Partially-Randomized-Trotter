"""New one-shot supplementation contract; sealed evidence is not permission."""
from pathlib import Path
from .ax2a_preparation import digest
from .h6_matched_contract_v1 import plan, resources, file_hash, safe_path, environment, verify_parent
from .h6_matched_contract_v2 import source_paths as previous_sources, check_inherited_limits
from .h6_matched_reuse_v3 import OLD_SOURCE, OLD_RESULT, committed_hashes, verify_evidence

KIND='H6_MATCHED_RESOURCE_SUPPLEMENT_V3'
RUNNER='scripts/resource_applicability/run_track_a_h6_matched_v3.py'
PREPARER='scripts/resource_applicability/prepare_track_a_h6_matched_v3.py'
AUDITOR='scripts/resource_applicability/audit_track_a_h6_matched_v3.py'
NAMESPACE='artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/'
PREPARATION_PATH='artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/'


def source_paths(root):return sorted(set(previous_sources(root))|{RUNNER,PREPARER,AUDITOR})


def verify_sources(root,commit,hashes):
    if (not isinstance(commit,str) or len(commit)!=40 or any(x not in '0123456789abcdef' for x in commit)
        or set(hashes)!=set(source_paths(root))):raise ValueError('SUPPLEMENT_SOURCE_CLOSURE')
    committed_hashes(root,commit,hashes)


def supplement_policy():
    return dict(original_science_source=OLD_SOURCE,original_result_commit=OLD_RESULT,
        registered_exploration_groups=592,reused_signals=92,reused_b3=84,
        reused_exploration_groups=449,reused_wrappers=902,missing_exploration_groups=143,
        new_exploration_wrappers=286,maximum_confirmation_groups=256,
        maximum_new_wrappers=798,maximum_integrated_wrappers=1700,
        original_wrapper_cap=1708,original_compile_cap=1712,
        memory_policy='coordinator AS soft2/hard8 GiB; workers soft/hard8 GiB unchanged',
        first_task_index=447,first_task_cell='B1_p19_q8_4th',first_task_exclusive=True,
        remainder_order='original registration indices; two disposable workers',
        incomplete_event_policy='regenerate from ORIGINAL integer seed; compare saved partial digest if present',
        sampling_rule_reused='h6_matched_accounting_v1.seed/tasks; sampling port unchanged',
        second_axis='rebuild native evolution from same prepared blocks and SAME steps, no resampling',
        wrapper_memory='consume single evolution into same canonical wrapper; release input after transpile',
        selection_requires='exact 592-task set, replicas, both axes, no duplicates before any map or selection',
        confirmation='original top2 B2 per epsilon; union<=8 frozen before independent32 trajectories',
        aborted_old_calls=None,aborted_old_calls_scope='unknown, excluded from exact completed-call totals, separately disclosed',
        no_old_resume=True,no_old_retry=True,no_new_retry=True,no_extra_candidates=True,
        H8_authorized=False,scientific_execution_authorized=False)


def preparation():
    return dict(schema='h6_matched_preparation_v3',kind=KIND,plan=plan(),
        science_plan_digest_v1=digest(plan()),supplement_policy=supplement_policy(),
        source_commit=None,source_hashes=None,validation_source_hashes=None,
        input_identity=None,environment=None,assigned_resources=None,reuse_registry=None,
        exclusive_output=None,execution_plan_sealed=False,
        status='PREPARATION_ONLY_NOT_AUTHORIZED',science_authorized=False,launch_allowed=False,
        H6_status='H6_NOT_AUTHORIZED',contract_status='DRAFT_NOT_AUTHORIZATION',
        mandatory_stop=True,next_stage_authorized=False)


def validate_manifest(m):
    expected=preparation()
    for k in ('schema','kind','status','science_authorized','launch_allowed','H6_status',
        'contract_status','mandatory_stop','next_stage_authorized','science_plan_digest_v1','supplement_policy'):
        if type(m.get(k)) is not type(expected[k]) or m[k]!=expected[k]:raise ValueError('SUPPLEMENT_MANIFEST:'+k)
    if m.get('execution_plan_sealed') is not True or digest(m.get('plan'))!=digest(plan()):
        raise ValueError('SUPPLEMENT_SCIENCE_PLAN')


def validate_grant(m,g,output):
    if (not isinstance(g,dict) or g.get('schema')!='h6_matched_authorization_v3' or g.get('kind')!=KIND or
        g.get('approved_by_user') is not True or g.get('authorization_only') is not True or
        g.get('manifest_digest')!=digest(m) or g.get('source_commit')!=m['source_commit'] or
        g.get('reuse_registry_digest')!=digest(m['reuse_registry']) or
        g.get('assigned_cpus')!=m['assigned_resources']['assigned_cpus'] or
        g.get('one_shot') is not True or g.get('retry') is not False or g.get('resume') is not False or
        g.get('exclusive_output')!=str(output) or g.get('H8_authorized') is not False):
        raise ValueError('FRESH_SUPPLEMENT_ONE_SHOT_GRANT_REQUIRED')


def validate_launch(root,m,g,output,*,requested=False):
    if requested is not True:raise ValueError('EXPLICIT_EXECUTION_REQUIRED')
    validate_manifest(m);validate_grant(m,g,Path(output).resolve());check_inherited_limits()
    if m['assigned_resources']!=resources(g['assigned_cpus']):raise ValueError('SUPPLEMENT_RESOURCES')
    out=m['exclusive_output'];path=Path(output).resolve()
    if (not out['repository_path'].startswith(NAMESPACE) or safe_path(root,out['repository_path'])!=path or
        out['absolute_path']!=str(path) or path.exists()):raise ValueError('SUPPLEMENT_EXCLUSIVE_OUTPUT')
    verify_sources(root,m['source_commit'],m['source_hashes'])
    verify_evidence(root,m['reuse_registry'])
    if (verify_parent(root)!=m['input_identity'] or environment()!=m['environment'] or
        m['reuse_registry']['input_identity']!=m['input_identity'] or
        m['reuse_registry']['environment']!=m['environment']):raise ValueError('SUPPLEMENT_INPUT_ENVIRONMENT')
    return m['assigned_resources']
