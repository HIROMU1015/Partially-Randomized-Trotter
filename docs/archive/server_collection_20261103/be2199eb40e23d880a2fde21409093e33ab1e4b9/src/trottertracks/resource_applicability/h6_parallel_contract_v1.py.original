"""Four independent frozen cost groups; preparation is never authorization."""
from pathlib import Path
from copy import deepcopy
from .ax2a_preparation import digest
from .h6_matched_reuse_v3 import read, committed_hashes, admit_group, COMPILER_HASH
from .h6_server_contract_v1 import (source_paths as base_sources, verify_bindings,
    file_hash, safe_path)
from .h6_server_resources_v1 import observation_policy, fresh_resources, process_identity
from .h6_server_execution_v1 import admit_new_group

BASE='artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/sealed_manifest_server_v1.json'
PREPARATION='artifacts/resource_applicability/track_a_h6_roundoff_preparation_v1/2026-10-12/'
NAMESPACE='artifacts/resource_applicability/track_a_h6_roundoff_exploration_v1/2026-10-12/'
RUNNER='scripts/resource_applicability/run_track_a_h6_parallel_v1.py'
PREPARER='scripts/resource_applicability/prepare_track_a_h6_parallel_v1.py'
KIND='H6_PARALLEL_ROUNDOFF_EXPLORATION_V1'
SCHEMA='h6_parallel_roundoff_manifest_v1'
LOCK='.server-preparation/h6-parallel-roundoff-v1/one-shot.lock'
PRIOR_STOP='artifacts/resource_applicability/track_a_h6_parallel_preparation_v2/2026-10-12/launch_authorization_v1/launch_stop_receipt_v1.json'


def sources(root):
    return sorted(set(base_sources(root))|{RUNNER,PREPARER,
        'src/trottertracks/resource_applicability/h6_parallel_contract_v1.py',
        'src/trottertracks/resource_applicability/h6_parallel_execution_v1.py',
        'src/trottertracks/resource_applicability/h6_frozen_preparation_v1.py',
        'src/trottertracks/resource_applicability/h6_roundoff_preparation_v1.py',
        'src/trottertracks/resource_applicability/h6_roundoff_cost_port_v1.py'})


def roundoff_binding(root,m):
    """Bind a bounded numerical bridge receipt; this is not a call-total claim."""
    from .h6_roundoff_preparation_v1 import policy,OPERATOR_CAP,NATIVE_INPUT_CAP
    ref=m.get('roundoff_preparation_binding')
    if ref is None:return False
    if type(ref) is not dict or set(ref)!={'archive_path','archive_sha256','archive_bytes','receipt_path','receipt_sha256'}:
        raise ValueError('ROUNDING_RECEIPT_REFERENCE')
    for k in ('archive_path','receipt_path'):
        path=Path(ref[k])
        if (not path.is_absolute() or not path.is_relative_to('/home/AbeHiromu')
            or any(p.is_symlink() for p in (path,*path.parents))):raise ValueError('ROUNDING_PRIVATE_PATH')
    archive=Path(ref['archive_path']);receipt=Path(ref['receipt_path'])
    if (type(ref['archive_bytes']) is not int or not 0<ref['archive_bytes']<=16*2**20
        or archive.stat().st_size!=ref['archive_bytes'] or file_hash(archive)!=ref['archive_sha256']
        or receipt.stat().st_size>2**20 or file_hash(receipt)!=ref['receipt_sha256']):
        raise ValueError('ROUNDING_RECEIPT_BYTES')
    proof=read(receipt)
    if (proof.get('status')!='PASS_USER_APPROVED_ROUNDOFF' or proof.get('policy')!=policy()
        or proof.get('deterministic_preparation_unchanged') is not True
        or proof.get('original_hamiltonian_hash')!=m['base_manifest']['input_identity']['hamiltonian_hash']
        or proof.get('diagnostic_calls')!={'trajectory':1,'occurrence':8}
        or any(proof.get(k)!=0 for k in ('compile','signal','input_generation'))
        or proof.get('partial_450',{}).get('exact_event_choices') is not True
        or proof.get('partial_450',{}).get('only_numeric_differences') is not True):
        raise ValueError('ROUNDING_DIAGNOSTIC_RECEIPT')
    keys={'src/trottertracks/resource_applicability/h6_roundoff_preparation_v1.py',
        'src/trottertracks/resource_applicability/h6_roundoff_cost_port_v1.py',
        'src/trottertracks/resource_applicability/h6_frozen_preparation_v1.py'}
    if (set(proof.get('diagnostic_source_hashes',{}))!=keys
        or any(proof['diagnostic_source_hashes'][p]!=m['source_hashes'][p] for p in keys)):
        raise ValueError('ROUNDING_DIAGNOSTIC_SOURCE')
    checks=proof.get('fragment_checks',[])
    if [x.get('index') for x in checks]!=list(range(19)):raise ValueError('ROUNDING_FRAGMENT_SET')
    for x in checks:
        for name,cap in [('native_input_weighted_bound',NATIVE_INPUT_CAP),('aligned_weighted_difference_bound',OPERATOR_CAP),('unitarity',1e-12)]:
            v=x.get(name)
            if not isinstance(v,(int,float)) or not 0<=v<=cap:raise ValueError('ROUNDING_OPERATOR_CHECK')
    if sum(x['aligned_weighted_difference_bound'] for x in checks)>OPERATOR_CAP:
        raise ValueError('ROUNDING_TOTAL_OPERATOR_CHECK')
    if sum(x['native_input_weighted_bound'] for x in checks)>NATIVE_INPUT_CAP:
        raise ValueError('ROUNDING_TOTAL_NATIVE_INPUT_CHECK')
    expected=next(x['partial_event_digest'] for x in m['base_manifest']['reuse_registry']['missing'] if x['index']==450)
    if proof['partial_450']['original_event_digest']!=expected:raise ValueError('ROUNDING_OLD_PARTIAL_BINDING')
    return True


def proposed_resources():
    # RSS limits are sampled STOP thresholds, not kernel reservations. All old
    # host/cgroup, aggregate RSS40GiB and job RAM64GiB guards remain in force.
    return dict(assigned_cpus=[20,21,22,23,24,25],coordinator_cpu_set=[22,23],
        cost_cpu_sets=[[20],[21],[24],[25]],cost_workers=4,cost_threads=1,
        cost_address_space_bytes=32*2**30,worker_rss_stop_bytes=9*2**30,
        coordinator_address_space_bytes=2*2**30,coordinator_hard_address_space_bytes=32*2**30,
        aggregate_rss_bytes=40*2**30,job_ram_budget_bytes=64*2**30,
        coordinator_rss_reserve_bytes=2*2**30,aggregate_margin_bytes=2*2**30,
        memory_is_reservation=False,gpu=False)


def slots(r):
    room=r['aggregate_rss_bytes']-r['coordinator_rss_reserve_bytes']-r['aggregate_margin_bytes']
    if room<=0 or r['worker_rss_stop_bytes']<=0:raise ValueError('PARALLEL_RSS_RESERVATION')
    return min(r['cost_workers'],len(r['cost_cpu_sets']),room//r['worker_rss_stop_bytes'])


def registrations(base):
    rows=[dict(index=x['index'],task=deepcopy(x['task'])) for x in base['reuse_registry']['missing'] if x['index']!=447]
    if [x['index'] for x in rows]!=list(range(450,592)):raise ValueError('PARALLEL_FIXED_142_TASKS')
    return rows


def phase_a_receipt(base, *, require=False):
    run=Path(base['exclusive_output']['absolute_path'])
    if not run.is_relative_to('/home/AbeHiromu') or any(p.is_symlink() for p in (run,*run.parents)):
        raise ValueError('PARALLEL_PHASE_A_PATH')
    terminal=run/'execution_terminal.json'
    if not terminal.exists():
        if require:raise ValueError('PARALLEL_PHASE_A_NOT_COMPLETE')
        return None
    t=read(terminal)
    if (t['status']!='H6_SERVER_PHASE_A_COMPLETE_MANDATORY_STOP' or t['reason'] is not None
        or t['source_commit']!=base['source_commit']):raise ValueError('PARALLEL_PHASE_A_FAILED')
    if read(run/'frozen_manifest.json')!=base:raise ValueError('PARALLEL_PHASE_A_MANIFEST')
    summary=read(run/'phase_a_resource_final_summary.json')
    if not summary['process_reaped'] or summary['cleanup_error'] or summary['first_stop_reason']:
        raise ValueError('PARALLEL_PHASE_A_CLEANUP')
    result=admit_new_group(run/'cost_exploration_0447',base['phase_scope']['task'],base['source_commit'])
    identity=read(run/'execution_claim.json')['coordinator_identity']
    try:current=process_identity(identity['pid'])
    except FileNotFoundError:current=None
    if current and current['start_ticks']==identity['start_ticks']:
        if require:raise ValueError('PARALLEL_PHASE_A_COORDINATOR_STILL_PRESENT')
        return None
    return dict(run=str(run),source_commit=base['source_commit'],task_digest=digest(result['task']),
        terminal_sha256=file_hash(terminal),cost_result_sha256=file_hash(run/'cost_exploration_0447/cost_result.json'),
        resource_summary_sha256=file_hash(run/'phase_a_resource_final_summary.json'),
        completed_group=447,wrappers=2,mandatory_stop=True)


def carry_calls(root,base,phase_a,*,roundoff=False):
    counts=dict(base['reuse_registry']['original_signal_calls'])
    paths=[Path(root)/x['directory'] for x in base['reuse_registry']['accepted']]
    if phase_a:paths.append(Path(phase_a['run'])/'cost_exploration_0447')
    for p in paths:
        for k,v in read(p/'cost_result.json')['calls_attempted'].items():counts[k]+=v
    if roundoff:
        # Two observed diagnostic calls: a JSON comparison failure then a
        # successful bridge. This excludes the separately RESERVED unknown
        # production STOP totals; no unknown total is converted into exact.
        counts['trajectory']+=2;counts['occurrence']+=16
    return counts


def prior_stop_accounting(root):
    path=Path(root)/PRIOR_STOP;old=read(path)
    if (old.get('actual_call_totals') is not None or old.get('budget_not_reset') is not True
        or old.get('attempted_trajectory_lower_bound')!=4
        or old.get('attempted_occurrence_lower_bound')!=32):raise ValueError('ROUNDING_PRIOR_STOP_ACCOUNTING')
    return dict(receipt_path=PRIOR_STOP,receipt_sha256=file_hash(path),actual_call_totals=None,
        exact_total_unknown=True,verified_lower_bounds={'trajectory':4,'occurrence':32},
        conservative_reservation={'trajectory':4,'occurrence':32},
        reservation_is_not_an_exact_total=True,budget_reset=False,failed_calls_refunded=False,
        unknown_earlier_attempts_preserved=True)


def static_demand(rows):
    return dict(compile=sum(4 if x['task']['ordinary'] else 2 for x in rows),
        control_probe=sum((8 if x['task']['ordinary'] else 4) if x['task']['validate'] else 0 for x in rows),
        trajectory=sum(bool(x['task']['cell']['R']) for x in rows),
        occurrence=sum(x['task']['cell']['q'] if x['task']['cell']['R'] else 0 for x in rows))


def validate_plan(root,m, *, launch=False):
    root=Path(root);base=read(safe_path(root,BASE))
    if m.get('schema')!=SCHEMA or m.get('kind')!=KIND:
        raise ValueError('PARALLEL_SCHEMA')
    if m['base_sha256']!=file_hash(root/BASE) or m['base_manifest']!=base:
        raise ValueError('PARALLEL_BASE_BINDING')
    if (m['proposed_resources']!=proposed_resources() or m['observation_policy']!=observation_policy()
        or m['registrations']!=registrations(base) or slots(m['proposed_resources'])!=4):
        raise ValueError('PARALLEL_SCOPE_OR_RESOURCES')
    for k,v in dict(approved=False,runtime_authorization=False,allowed_cpus=[],
        confirmation_authorized=False,H8_authorized=False,retry=False,resume=False,mandatory_stop=True).items():
        if type(m.get(k)) is not type(v) or m[k]!=v:raise ValueError('PARALLEL_PREPARATION_FLAG:'+k)
    if m['repository_root']!=str(root.resolve()):raise ValueError('PARALLEL_WORKTREE')
    output_binding(root,m)
    if set(m['source_hashes'])!=set(sources(root)):raise ValueError('PARALLEL_SOURCE_CLOSURE')
    committed_hashes(root,m['source_commit'],m['source_hashes']);verify_bindings(root,base)
    ready=roundoff_binding(root,m)
    if m['carry_known_calls']!=carry_calls(root,base,m['phase_a_receipt'],roundoff=ready):raise ValueError('PARALLEL_CARRY_CHANGED')
    if m.get('prior_stop_accounting')!=prior_stop_accounting(root):raise ValueError('ROUNDING_PRIOR_STOP_BINDING')
    demand=static_demand(m['registrations'])
    if m['static_demand']!=demand:raise ValueError('PARALLEL_STATIC_DEMAND')
    reserved=m['prior_stop_accounting']['conservative_reservation']
    if any(m['carry_known_calls'][k]+reserved.get(k,0)+v>base['plan']['caps_proposed'][k] for k,v in demand.items()):
        raise ValueError('PARALLEL_CUMULATIVE_CALL_CAP')
    if launch:
        if not m['execution_plan_sealed'] or not ready:raise ValueError('PARALLEL_NOT_TECHNICALLY_SEALED')
        if phase_a_receipt(base,require=True)!=m['phase_a_receipt']:raise ValueError('PARALLEL_PHASE_A_CHANGED')
    elif m['execution_plan_sealed']!=(m['phase_a_receipt'] is not None and ready):
        raise ValueError('PARALLEL_SEAL_STATE')


def grant_draft(m):
    return dict(schema='h6_parallel_authorization_v1',kind=KIND,approved_by_user=False,
        server_usage_approved_by_user=False,manifest_digest=digest(m),source_commit=m['source_commit'],
        profile_sha256=m['base_manifest']['profile']['sha256'],task_list_digest=digest(m['registrations']),
        roundoff_preparation_binding_digest=digest(m.get('roundoff_preparation_binding')),
        prior_stop_accounting_digest=digest(m.get('prior_stop_accounting')),
        resources=deepcopy(m['proposed_resources']),exclusive_output=m['exclusive_output']['absolute_path'],
        one_shot=True,retry=False,resume=False,confirmation_authorized=False,H8_authorized=False,mandatory_stop=True)


def validate_grant(m,g):
    expected=grant_draft(m);expected.update(approved_by_user=True,server_usage_approved_by_user=True)
    if m.get('roundoff_preparation_binding') is None or g!=expected or any(type(g.get(k)) is not bool for k in (
        'approved_by_user','server_usage_approved_by_user','one_shot','retry','resume',
        'confirmation_authorized','H8_authorized','mandatory_stop')):
        raise ValueError('NEW_PARALLEL_PHASE_B_GRANT_REQUIRED')


def output_binding(root,m):
    ref=m['exclusive_output'];expected=NAMESPACE+'launch_parallel_v1'
    if (ref['repository_path']!=expected or ref['absolute_path']!=str(safe_path(root,expected))
        or m['one_shot_lock']!=LOCK):raise ValueError('PARALLEL_OUTPUT_BINDING')


def worker_scope(root,out,m):
    """Cheap child-side scope check before any scientific import or affinity."""
    if (m['schema']!=SCHEMA or m['kind']!=KIND
        or m['execution_plan_sealed'] is not True or m['approved'] is not False
        or m['runtime_authorization'] is not False or m['allowed_cpus']!=[]
        or m['retry'] is not False or m['resume'] is not False
        or m['confirmation_authorized'] is not False or m['H8_authorized'] is not False
        or m['mandatory_stop'] is not True or m['repository_root']!=str(Path(root).resolve())
        or m['proposed_resources']!=proposed_resources()
        or m['observation_policy']!=observation_policy()
        or m['registrations']!=registrations(m['base_manifest'])
        or type(m.get('roundoff_preparation_binding')) is not dict):
        raise ValueError('PARALLEL_WORKER_SCOPE')
    output_binding(root,m)
    if str(Path(out).resolve())!=m['exclusive_output']['absolute_path']:
        raise ValueError('PARALLEL_WORKER_OUTPUT')


def launch_gate(root,m,g,out):
    validate_grant(m,g);validate_plan(root,m,launch=True)
    ref=m['exclusive_output'];out=Path(out).resolve()
    if (not ref['repository_path'].startswith(NAMESPACE) or safe_path(root,ref['repository_path'])!=out
        or str(out)!=ref['absolute_path']):raise ValueError('PARALLEL_OUTPUT_BINDING')
    lock=safe_path(root,m['one_shot_lock'])
    if lock.exists() or lock.is_symlink():raise ValueError('PARALLEL_ONE_SHOT_CONSUMED')
    return fresh_resources(m['proposed_resources'],m['observation_policy'],out)
