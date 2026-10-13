"""User-approved 10M instruction /21 compile /44 probe retry, no reset."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

from . import h8_fast_handoff_contract_v1 as handoff
from .ax2a_preparation import digest
from .ax2b_h4_contract_v5 import file_hash

RUNNER='scripts/resource_applicability/run_track_a_h8_fast_retry_v1.py'
PREPARER='scripts/resource_applicability/prepare_track_a_h8_fast_retry_v1.py'
PIPELINE='scripts/resource_applicability/continue_track_a_h8_fast_retry_v1.py'
AUTHORIZATION='1000万instructions・累積compile21/control probe44を承認し、高速化版S4/q8を一度再実行、成功後に残7へ接続'
SCOPES=('HEAVY_RETRY_FAST','REMAINING_AFTER_FAST_RETRY')
FAILED_NAMES=handoff.PENDING_NAMES+('heavy01/execution_terminal.json',
    'heavy01/cost_exploration_0003/worker_terminal.json',
    'heavy01/cost_exploration_0003/failure_diagnostics.json',
    'heavy01/cost_exploration_0003/compile_attempt_01.json',
    'heavy01/cost_exploration_0003/validation_symmetric_directional.json',
    'heavy01/cost_exploration_0003/trajectory.json')


def plan():
    p=deepcopy(handoff.plan())
    p['caps_proposed'].update(transpiled_instructions=10000000,compile=21,control_probe=44)
    return p


def source_paths(root):
    return sorted(set(handoff.source_paths(root))|{RUNNER,PREPARER,PIPELINE})


def failed_receipt(parent):
    parent=handoff.private_path(parent);handoff.require_retired_continuation(parent)
    d=parent/'heavy01';w=d/'cost_exploration_0003';m=handoff.read(d/'manifest.json')
    term=handoff.read(d/'execution_terminal.json');wt=handoff.read(w/'worker_terminal.json')
    attempt=handoff.read(w/'compile_attempt_01.json');claim=handoff.read(w/'worker_claim.json')
    coordinator=handoff.read(parent/'launch-heavy01.json')['identity']
    calls=dict(compile=1,control_probe=4,trajectory=0,occurrence=0,primitive=0,reference_matvec=0)
    if (m['source_commit']!=handoff.OLD_SOURCE or m['scope']!='HEAVY'
        or [r['index'] for r in m['registrations']]!=[3]
        or term['first_stop_reason']!='WORKER_EXIT:1' or term['completed_indices']
        or not term['all_dispatched_reaped'] or term['cleanup_errors']
        or len(term['retired'])!=1 or term['retired'][0]['exit_code']!=1
        or wt['status']!='STOP' or wt['reason']!='RuntimeError:TRANSPILED_INSTRUCTION_CAP'
        or wt['source_commit']!=handoff.OLD_SOURCE or wt['scope']!='HEAVY'
        or attempt['task']!=m['registrations'][0]['task'] or attempt['axis']!='cosine'
        or attempt['calls_attempted_before_fingerprint']!=calls
        or list(w.glob('wrapper_*.json')) or (w/'cost_result.json').exists()
        or len(list(w.glob('compile_attempt_*.json')))!=1):
        raise ValueError('FAILED_PARENT_ACCOUNTING_OR_SCOPE')
    if not handoff.inactive(coordinator) or not handoff.inactive(claim['identity']):
        raise ValueError('FAILED_PARENT_STILL_ACTIVE')
    return dict(status='PRESERVED_FAILED_INSTRUCTION_CAP',directory=str(parent),
        source_commit=handoff.OLD_SOURCE,file_hashes={n:file_hash(parent/n) for n in FAILED_NAMES},
        known_calls=calls,completed_wrappers=0,actual_transpiled_instruction_count=None,
        exact_exit_timestamp=None,coordinator_exit_code=None,refund=0,reset=False)


def registrations(parent,scope):
    signals=handoff.read(Path(parent)/'signal01/cost_exploration_0000/signal_summary.json')
    if scope=='HEAVY_RETRY_FAST':return handoff.old_registrations(handoff.OLD_SOURCE,'HEAVY',signals)
    if scope=='REMAINING_AFTER_FAST_RETRY':return handoff.remaining_registrations(signals)
    raise ValueError('FAST_RETRY_SCOPE')


def resources(scope):
    return handoff.resources('HEAVY',1) if scope=='HEAVY_RETRY_FAST' else handoff.resources('REMAINING',3)


def observation(scope):
    return handoff.observation('HEAVY',1) if scope=='HEAVY_RETRY_FAST' else handoff.observation('REMAINING',3)


def verify_completed_heavy(root,directory,expected_source):
    from .h8_fast_handoff_contract_v1 import inactive
    d=handoff.private_path(directory);m=handoff.read(d/'manifest.json');g=handoff.read(d/'grant.json')
    if m['source_commit']!=expected_source or m['scope']!='HEAVY_RETRY_FAST':
        raise ValueError('SUCCESSFUL_RETRY_SOURCE_OR_SCOPE')
    verify_binding(root,m,g,worker=True)
    t=handoff.read(d/'execution_terminal.json');w=d/'cost_exploration_0003'
    wt=handoff.read(w/'worker_terminal.json');result=handoff.read(w/'cost_result.json')
    if (t['first_stop_reason'] is not None or t['completed_indices']!=[3]
        or not t['all_dispatched_reaped'] or t['cleanup_errors']
        or len(t['retired'])!=1 or t['retired'][0]['exit_code']!=0
        or wt['status']!='COMPLETE' or wt['reason'] is not None
        or wt['scope']!='HEAVY_RETRY_FAST' or wt['source_commit']!=expected_source):
        raise ValueError('FAST_HEAVY_RETRY_NOT_COMPLETE')
    # Importing this metadata validator never runs its main/worker.
    import runpy
    validator=runpy.run_path(str(Path(root)/'scripts/resource_applicability/run_track_a_h8_fast_handoff_v1.py'))['admit']
    validator(w,m['registrations'][0]['task'],m)
    parent_id=handoff.read(d/'parent_identity.json');worker_id=handoff.read(w/'worker_claim.json')['identity']
    if not inactive(parent_id) or not inactive(worker_id):raise ValueError('FAST_HEAVY_RETRY_STILL_ACTIVE')
    names=('manifest.json','grant.json','parent_identity.json','execution_terminal.json',
        'cost_exploration_0003/worker_terminal.json','cost_exploration_0003/cost_result.json',
        'cost_exploration_0003/trajectory.json','cost_exploration_0003/validation_symmetric_directional.json',
        'cost_exploration_0003/wrapper_symmetric_directional_cosine.json',
        'cost_exploration_0003/wrapper_symmetric_directional_sine.json')
    return dict(status='ACCEPTED_FAST_HEAVY_RETRY',directory=str(d),source_commit=expected_source,
        file_hashes={n:file_hash(d/n) for n in names},result=result,all_reaped=True,cleanup_errors=[])


def make_manifest(root,parent,source,output,scope,heavy_directory=None):
    if scope not in SCOPES:raise ValueError('FAST_RETRY_SCOPE')
    # Reuse prior signal/input binding and old seed selection, preserving the
    # prior contract's runtime files. Override ONLY the approved caps/scope.
    m=handoff.pending_template(root,parent,source,output)
    m.update(schema='h8_fast_retry21_scope_v1',scope=scope,plan=plan(),
        source_hashes={p:file_hash(Path(root)/p) for p in source_paths(root)},
        registrations=registrations(parent,scope),proposed_resources=resources(scope),
        observation_policy=observation(scope),preserved_failure=failed_receipt(parent),
        approved_caps=dict(transpiled_instructions=10000000,compile=21,control_probe=44,
            trajectory=4,occurrence=15),
        accounting=dict(old_failed_compile=1,new_heavy_compile=2,remaining_compile=18,total_compile=21,
            old_completed_control_probe=4,new_heavy_control_probe=4,remaining_control_probe=36,
            total_control_probe=44,trajectory=4,occurrence=15,refund=0,reset=False),
        validation_method=handoff.METHOD,validation_method_changed=True,
        automatic_retry=False,explicit_single_retry_authorized=True,technical_plan_sealed=True,
        conditional_start='one explicit heavy retry; remaining only after complete fast heavy retry')
    # The old 20/40 bookkeeping remains a historical plan, never active limits.
    m.pop('handoff_budget',None)
    m.pop('retry',None)
    m['retry_policy']='ONE_USER_APPROVED_HEAVY_RETRY_NO_AUTOMATIC_RETRY'
    if scope=='REMAINING_AFTER_FAST_RETRY':
        if heavy_directory is None:raise ValueError('FAST_HEAVY_RECEIPT_REQUIRED')
        m['accepted_retry_heavy']=verify_completed_heavy(root,heavy_directory,source)
    g=dict(schema='h8_fast_retry21_authorization_v1',approved_by_user=True,
        authorization_basis=AUTHORIZATION,source_commit=source,manifest_sha256=digest(m),
        output=m['output'],scope=scope,approved_caps=m['approved_caps'],one_shot=True,
        automatic_retry=False,H8_full_campaign_authorized=False,mandatory_stop=True)
    return m,g


def verify_binding(root,m,g,*,worker=False):
    root=Path(root);scope=m['scope'];parent=handoff.private_path(m['old_parent']['directory'])
    if (scope not in SCOPES or m['schema']!='h8_fast_retry21_scope_v1'
        or m['H8_full_campaign_authorized'] is not False or m['automatic_retry'] is not False
        or m['seed_source_commit']!=handoff.OLD_SOURCE or m['technical_plan_sealed'] is not True
        or m['explicit_single_retry_authorized'] is not True
        or m['validation_method']!=handoff.METHOD or m['validation_method_changed'] is not True):
        raise ValueError('FAST_RETRY_FIXED_FLAGS')
    if (g.get('schema')!='h8_fast_retry21_authorization_v1' or g.get('approved_by_user') is not True
        or g.get('authorization_basis')!=AUTHORIZATION or g.get('source_commit')!=m['source_commit']
        or g.get('manifest_sha256')!=digest(m) or g.get('output')!=m['output'] or g.get('scope')!=scope
        or g.get('one_shot') is not True or g.get('automatic_retry') is not False
        or g.get('H8_full_campaign_authorized') is not False or g.get('approved_caps')!=m['approved_caps']):
        raise ValueError('FAST_RETRY_AUTH_BINDING')
    if digest(m['plan'])!=digest(plan()) or m['profile']!=handoff.profile():raise ValueError('FAST_RETRY_PLAN_PROFILE')
    caps=dict(transpiled_instructions=10000000,compile=21,control_probe=44,trajectory=4,occurrence=15)
    accounting=dict(old_failed_compile=1,new_heavy_compile=2,remaining_compile=18,total_compile=21,
        old_completed_control_probe=4,new_heavy_control_probe=4,remaining_control_probe=36,
        total_control_probe=44,trajectory=4,occurrence=15,refund=0,reset=False)
    if m['approved_caps']!=caps or m['accounting']!=accounting:raise ValueError('FAST_RETRY_CARRY_CAPS')
    heavy_old=handoff.read(parent/'heavy01/manifest.json')
    if (m['input_identity']!=handoff.input_identity(m['input_identity']['directory'])
        or m['input_identity']!=heavy_old['input_identity'] or m['cumulative_carry']!=heavy_old['cumulative_carry']
        or m['base_manifest']!=heavy_old['base_manifest']):raise ValueError('FAST_RETRY_INPUT_OR_CARRY')
    if digest(m['preserved_failure'])!=digest(failed_receipt(parent)):raise ValueError('FAILED_RECEIPT_CHANGED')
    for name,h in m['old_parent']['pending_hashes'].items():
        if file_hash(parent/name)!=h:raise ValueError('OLD_PARENT_CHANGED:'+name)
    if set(m['old_parent']['pending_hashes'])!=set(handoff.PENDING_NAMES):raise ValueError('OLD_PARENT_CLOSURE')
    if file_hash(m['signal_binding']['path'])!=m['signal_binding']['sha256']:raise ValueError('SIGNAL_RECEIPT_CHANGED')
    if digest(m['registrations'])!=digest(registrations(parent,scope)):raise ValueError('OLD_SEED_TASK_CHANGED')
    if m['proposed_resources']!=resources(scope) or m['observation_policy']!=observation(scope):raise ValueError('FAST_RETRY_RESOURCES')
    if m['compiler_hash']!=handoff.COMPILER:raise ValueError('FAST_RETRY_COMPILER')
    handoff.private_path(m['output'])
    if set(m['source_hashes'])!=set(source_paths(root)):raise ValueError('FAST_RETRY_SOURCE_CLOSURE')
    if not worker:
        subprocess.run(['git','merge-base','--is-ancestor',handoff.FAST_SOURCE,m['source_commit']],cwd=root,check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','merge-base','--is-ancestor',m['source_commit'],'HEAD'],cwd=root,check=True,stdout=subprocess.DEVNULL)
    for p,h in m['source_hashes'].items():
        if file_hash(root/p)!=h:raise ValueError('FAST_RETRY_SOURCE_CHANGED:'+p)
        if not worker and hashlib.sha256(subprocess.check_output(['git','show',m['source_commit']+':'+p],cwd=root)).hexdigest()!=h:
            raise ValueError('FAST_RETRY_SOURCE_BLOB:'+p)
    if any(m['source_hashes'].get(p)!=h for p,h in heavy_old['source_hashes'].items()):raise ValueError('FAST_RETRY_OLD_SCIENCE_CHANGED')
    if scope=='REMAINING_AFTER_FAST_RETRY':
        heavy=m['accepted_retry_heavy']
        if digest(heavy)!=digest(verify_completed_heavy(root,heavy['directory'],m['source_commit'])):
            raise ValueError('COMPLETE_HEAVY_CHANGED')
    return True
