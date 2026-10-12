"""Eight-worker/hash-reuse continuation. No grant or running-parent takeover.

Admission needs a closed parent, complete-only receipts and fresh accounting.
An implementation request never authorizes a scientific runner or budget reset.
"""
from copy import deepcopy
from pathlib import Path

from .ax2a_preparation import digest
from .h6_confirmation_contract_v1 import (
    BASE, REPORT, RESULT_COMMIT, PARENT_SOURCE, private_path, exploration_binding,
    accounting, rare_order_plan, demand, sources as original_sources)
from .h6_confirmation_contract_v1 import validate_grant as parent_grant
from .h6_matched_reuse_v3 import read, file_hash, committed_hashes
from .h6_server_contract_v1 import safe_path, verify_bindings
from .h6_server_resources_v1 import observation_policy as original_policy, fresh_resources, process_identity
from .h6_parallel_contract_v1 import proposed_resources as original_resources, slots, roundoff_binding
from .h6_matched_accounting_v1 import seed

PREPARATION = 'artifacts/resource_applicability/track_a_h6_confirmation_fast_preparation_v1/2026-10-12/'
NAMESPACE = 'artifacts/resource_applicability/track_a_h6_confirmation_fast_v1/2026-10-12/'
RUNNER = 'scripts/resource_applicability/run_track_a_h6_confirmation_fast_v1.py'
PREPARER = 'scripts/resource_applicability/prepare_track_a_h6_confirmation_fast_v1.py'
AUDITOR = 'scripts/resource_applicability/audit_track_a_h6_confirmation_fast_v1.py'
LOCK = '.server-preparation/h6-confirmation-fast-v1/one-shot.lock'
KIND = 'H6_INDEPENDENT32_COMPLETE_ONLY_HASH_REUSE_EIGHT_WORKER_V1'
SCHEMA = 'h6_confirmation_fast_manifest_v1'
PARENT_CONFIRMATION_SOURCE = '73018cbcc39803890974c7ee51401063e6c8daf6'
PARENT_CONFIRMATION_REVIEW = '3b6d733a107273a027c910ee1a61638b5a3c241e'


def sources(root):
    return sorted(set(original_sources(root)) | {RUNNER, PREPARER, AUDITOR,
        'src/trottertracks/resource_applicability/h6_hash_reuse_v1.py',
        'src/trottertracks/resource_applicability/h6_confirmation_fast_contract_v1.py',
        'src/trottertracks/resource_applicability/h6_confirmation_fast_execution_v1.py'})


def hash_policy():
    return dict(policy='content_checked_canonical_v2_reuse_v1', payload_cache_bytes=32*2**20,
        entry_bytes=2**20, cache_entries=8192, per_call_guard_nodes=8192, digest_entries=1024,
        cross_call_object_identity_keys=False, guard_every_call=True, worker_group_scope=True,
        numeric_array_nonfinite='STOP', old_canonical_bytes_and_digest=True,
        call_local_pinned_lazy_standard_definition_memo=True)


def proposed_resources():
    r = original_resources()
    r.update(assigned_cpus=[20,21,22,23,24,25,26,27,28,29],
        cost_cpu_sets=[[20],[21],[24],[25],[26],[27],[28],[29]], cost_workers=8,
        aggregate_rss_bytes=76*2**30, job_ram_budget_bytes=96*2**30)
    return r


def observation_policy():
    p = original_policy()
    p.update(fresh_host_available_bytes=int(248.25*2**30), fresh_effective_available_bytes=96*2**30)
    return p


def parent_confirmation_receipt(path, binding):
    """Read a stopped original4 run; never read arrays or change that run."""
    from .h6_confirmation_execution_v1 import admit_confirmation_group
    path = private_path(path)
    pm = read(path/'frozen_manifest.json'); g = read(path/'authorization.json')
    selection = binding['selection']; parent_root = private_path(pm['repository_root'])
    original = parent_root/'artifacts/resource_applicability/track_a_h6_confirmation_preparation_v1/2026-10-12'
    fixed = [original/'sealed_v1/manifest_confirmation_v1.json',
        original/'launch_authorization_v1/authorization_approved_v1.json']
    committed_hashes(parent_root,PARENT_CONFIRMATION_REVIEW,
        {str(f.relative_to(parent_root)):file_hash(f) for f in fixed})
    if pm != read(fixed[0]) or g != read(fixed[1]):raise ValueError('FAST_PARENT_FIXED_GRANT_OR_MANIFEST')
    parent_grant(pm,g)
    if (read(path/'authorization_source.json') != g or read(path/'launch_binding.json') != dict(
            manifest_digest=digest(pm),authorization_digest=digest(g),
            authorization_source_sha256=file_hash(path/'authorization_source.json'))):
        raise ValueError('FAST_PARENT_NATIVE_LAUNCH_BINDING')
    if (pm['source_commit'] != PARENT_CONFIRMATION_SOURCE
            or pm['exploration_binding'] != binding
            or pm['proposed_resources'] != original_resources()
            or pm['exclusive_output']['absolute_path'] != str(path)):
        raise ValueError('FAST_PARENT_CONFIRMATION_BINDING')
    committed_hashes(Path(pm['repository_root']), pm['source_commit'], pm['source_hashes'])
    identity = read(path/'execution_claim.json')['coordinator_identity']
    try:current = process_identity(identity['pid'])
    except FileNotFoundError:current = None
    if current is not None and current['start_ticks'] == identity['start_ticks']:
        raise ValueError('FAST_PARENT_STILL_RUNNING')
    terminal = read(path/'execution_terminal.json'); report = terminal['resource_summary']
    if (terminal['source_commit'] != PARENT_CONFIRMATION_SOURCE
            or terminal['status'] not in ('H6_CONFIRMATION_STOP','H6_CONFIRMATION_COMPLETE_MANDATORY_STOP')
            or not terminal['mandatory_stop'] or terminal['retry'] or terminal['resume']
            or report is None or not report['all_dispatched_reaped']
            or not report['ownership_cleanup_verified'] or report['cleanup_errors']):
        raise ValueError('FAST_PARENT_TERMINAL_OR_CLEANUP')
    regs = pm['registrations']; completed = report['completed_indices']
    dispatched = [x['index'] for x in report['dispatched']]
    if (regs != selection['registrations'] or completed != sorted(set(completed))
            or len(dispatched) != len(set(dispatched)) or not set(completed) <= set(dispatched)
            or not set(dispatched) <= set(range(len(regs)))):
        raise ValueError('FAST_PARENT_DISPATCH')
    retired = {x['index']:x for x in report['retired']}
    if (set(retired) != set(dispatched) or len(retired) != len(report['retired'])
            or any(not x['reaped'] for x in retired.values())
            or read(path/'confirmation_resource_summary.json') != report):
        raise ValueError('FAST_PARENT_REAP')
    for item in report['dispatched']:
        if item['task_digest'] != digest(regs[item['index']]['task']):
            raise ValueError('FAST_PARENT_DISPATCH_TASK')
        try:process_identity(item['pid'])
        except FileNotFoundError:continue
        # No worker start ticks were persisted by the old dispatcher. Reject
        # a present PID conservatively instead of guessing whether it was reused.
        raise ValueError('FAST_PARENT_WORKER_PID_PRESENT')
    accepted = []; inventory = {}; calls = dict.fromkeys(pm['cumulative_accounting']['known_calls'],0)
    for i in completed:
        if retired[i]['exit_code'] != 0:raise ValueError('FAST_PARENT_COMPLETE_EXIT')
        directory = path/f'cost_confirmation_{i:04d}'
        result = admit_confirmation_group(directory, regs[i]['task'],source=pm['source_commit'])
        accepted.append(dict(index=i,directory=str(directory),task_digest=digest(regs[i]['task']),
            cost_result_sha256=file_hash(directory/'cost_result.json')))
        for k,v in result['calls_attempted'].items():calls[k] += v
        for f in directory.glob('*.json'):
            if f.name.startswith(('diagnostic_','progress_','compile_attempt_')):continue
            inventory[str(f)] = dict(bytes=f.stat().st_size,sha256=file_hash(f))
    partial = sorted(set(dispatched)-set(completed))
    reserved = demand([regs[i] for i in partial])
    for name in ('frozen_manifest.json','authorization.json','authorization_source.json',
            'launch_binding.json','execution_claim.json','execution_terminal.json','confirmation_resource_summary.json'):
        f = path/name
        inventory[str(f)] = dict(bytes=f.stat().st_size,sha256=file_hash(f))
    return dict(parent_run=str(path),parent_source=pm['source_commit'],accepted=accepted,
        completed_known_calls=calls,partial_indices=partial,partial_task_reservation=reserved,
        partial_actual_totals=None,metadata_inventory=inventory,
        parent_approved_caps=deepcopy(g['proposed_cumulative_caps']),
        source_profile_input_selection_required=True,partial_reused=False,
        unknown_actual_totals=None,budget_reset=False,failed_calls_refunded=False)


def remaining(selection, reuse):
    done = {x['index'] for x in reuse['accepted']}
    return [deepcopy(x) for x in selection['registrations'] if x['index'] not in done]


def continuation_accounting(binding, base, reuse):
    a = accounting(binding, base)
    known = deepcopy(a['known_calls']); reserved = deepcopy(a['reservations'])
    for k,v in reuse['completed_known_calls'].items():known[k] += v
    for k,v in reuse['partial_task_reservation'].items():reserved[k] = reserved.get(k,0)+v
    needed = demand(remaining(binding['selection'],reuse))
    original = deepcopy(reuse['parent_approved_caps']); proposed = deepcopy(original)
    totals = {k:known[k]+reserved.get(k,0)+v for k,v in needed.items()}
    for k,v in totals.items():proposed[k] = max(proposed[k],v)
    return dict(known_calls=known,reservations=reserved,new_demand=needed,admission_totals=totals,
        original_caps=original,proposed_caps=proposed,
        amendments={k:dict(original=original[k],proposed=proposed[k]) for k in original if original[k]!=proposed[k]},
        unknown_actual_totals=None,reservations_are_not_actual_totals=True,
        cap_amendment_approved=False,failed_calls_refunded=False,budget_reset=False)


def output_binding(root,m):
    relative = NAMESPACE+'launch_confirmation_fast_v1'
    if (m['exclusive_output'] != dict(repository_path=relative,absolute_path=str(safe_path(root,relative)))
            or m['one_shot_lock'] != LOCK):raise ValueError('FAST_OUTPUT_BINDING')


def validate_scope(root,m):
    flags = dict(approved=False,runtime_authorization=False,allowed_cpus=[],confirmation_authorized=False,
        H8_authorized=False,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False)
    if (m.get('schema') != SCHEMA or m.get('kind') != KIND or m.get('repository_root') != str(Path(root).resolve())
            or any(type(m.get(k)) is not type(v) or m[k] != v for k,v in flags.items())):
        raise ValueError('FAST_SCOPE_FLAGS')
    if (m['proposed_resources'] != proposed_resources() or slots(m['proposed_resources']) != 8
            or m['observation_policy'] != observation_policy() or m['hash_reuse_policy'] != hash_policy()
            or m['candidate_selection_sealed'] is not True or m['execution_plan_sealed'] is not True
            or m['absolute_launch_command'] is not None):raise ValueError('FAST_RESOURCES_OR_SEAL')
    selection = m['exploration_binding']['selection']; regs = selection['registrations']
    selected = selection['selected_ids']; counts = {}
    if selected != sorted(set(selected)) or not 0 < len(selected) <= 8:raise ValueError('FAST_SELECTION')
    for i,row in enumerate(regs):
        t = row['task']
        if (type(row['index']) is not int or row['index'] != i or t['phase'] != 'confirmation'
                or t['cell']['method'] != 'B2' or t['cell']['id'] != t['cell_id'] or t['cell_id'] not in selected
                or t['validate'] is not False or t['ordinary'] is not False
                or type(t['replica']) is not int or not 0 <= t['replica'] < 32
                or type(t['seed']) is not int or t['seed'] != seed(t['cell_id'],'confirmation',t['replica'])):
            raise ValueError('FAST_TASK_SEED')
        counts.setdefault(t['cell_id'],[]).append(t['replica'])
    if (set(counts) != set(selected) or any(sorted(v) != list(range(32)) for v in counts.values())
            or len(regs) != 32*len(selected) or m['registrations'] != remaining(selection,m['confirmation_reuse'])):
        raise ValueError('FAST_COMPLETE32_REMAINDER')
    output_binding(root,m)


def validate_plan(root,m,*,launch=False):
    root = Path(root);validate_scope(root,m);base = read(root/BASE)
    if m['base_manifest'] != base or m['base_sha256'] != file_hash(root/BASE):raise ValueError('FAST_BASE')
    if set(m['source_hashes']) != set(sources(root)):raise ValueError('FAST_SOURCE_CLOSURE')
    committed_hashes(root,m['source_commit'],m['source_hashes']);verify_bindings(root,base)
    binding = exploration_binding(root)
    reuse = parent_confirmation_receipt(m['confirmation_reuse']['parent_run'],binding)
    if (m['exploration_binding'] != binding or m['confirmation_reuse'] != reuse
            or m['registrations'] != remaining(binding['selection'],reuse)
            or m['roundoff_preparation_binding'] != binding['roundoff_preparation_binding']
            or m['cumulative_accounting'] != continuation_accounting(binding,base,reuse)
            or m['rare_order_plan'] != rare_order_plan(binding) or not roundoff_binding(root,m)):
        raise ValueError('FAST_BINDING_OR_ACCOUNTING')
    if launch and (safe_path(root,LOCK).exists() or safe_path(root,LOCK).is_symlink()):
        raise ValueError('FAST_ONE_SHOT_USED')


def grant_draft(m):
    return dict(schema='h6_confirmation_fast_authorization_v1',kind=KIND,approved_by_user=False,
        server_usage_approved_by_user=False,cap_amendment_approved=False,confirmation_authorized=False,
        manifest_digest=digest(m),source_commit=m['source_commit'],profile_sha256=m['base_manifest']['profile']['sha256'],
        exploration_binding_digest=digest(m['exploration_binding']),task_list_digest=digest(m['registrations']),
        candidate_selection_digest=digest(m['exploration_binding']['selection']),
        confirmation_reuse_digest=digest(m['confirmation_reuse']),hash_reuse_policy_digest=digest(m['hash_reuse_policy']),
        cumulative_accounting_digest=digest(m['cumulative_accounting']),
        proposed_cumulative_caps=deepcopy(m['cumulative_accounting']['proposed_caps']),resources=deepcopy(m['proposed_resources']),
        exclusive_output=m['exclusive_output']['absolute_path'],
        roundoff_preparation_binding_digest=digest(m['roundoff_preparation_binding']),
        one_shot=True,retry=False,resume=False,H8_authorized=False,next_stage_authorized=False,mandatory_stop=True)


def validate_grant(m,g):
    expected = grant_draft(m)
    expected.update(approved_by_user=True,server_usage_approved_by_user=True,
        cap_amendment_approved=bool(m['cumulative_accounting']['amendments']),confirmation_authorized=True)
    keys = ('approved_by_user','server_usage_approved_by_user','cap_amendment_approved','confirmation_authorized',
        'one_shot','retry','resume','H8_authorized','next_stage_authorized','mandatory_stop')
    if g != expected or any(type(g.get(k)) is not bool for k in keys):raise ValueError('NEW_FAST_EXACT_GRANT_REQUIRED')


def worker_scope(root,out,m):
    validate_scope(root,m)
    if str(Path(out).resolve()) != m['exclusive_output']['absolute_path']:raise ValueError('FAST_WORKER_OUTPUT')


def launch_gate(root,m,g,out):
    validate_grant(m,g);validate_plan(root,m,launch=True)
    if str(Path(out).resolve()) != m['exclusive_output']['absolute_path']:raise ValueError('FAST_LAUNCH_OUTPUT')
    return fresh_resources(m['proposed_resources'],m['observation_policy'],out)
