"""Metadata-only one-shot s2 pilot contract. Preparation issues no grant."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .ax2a_preparation import digest
from .h6_matched_reuse_v3 import file_hash,committed_hashes
from .h6_server_contract_v1 import safe_path,verify_profile,verify_science,environment
from .h6_confirmation_fast_contract_v1 import sources as inherited_sources,hash_policy,private_path
from .h6_parallel_contract_v1 import proposed_resources as inherited_resources,roundoff_binding,slots
from .h6_server_resources_v1 import observation_policy,fresh_resources
from .h6_conditional_s2_sampling_v1 import PHASE,pilot_seed,conditional_binomial,order_seed,component_step_seeds

BASE_RESULT='eee1343f6e99aa7661892ddfbe1067e974e2b74a'
BASE='artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/sealed_manifest_server_v1.json'
SUMMARY='artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/'
ANALYSIS='artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/'
PREPARATION='artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_preparation_v1/2026-10-13/'
NAMESPACE='artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_v1/2026-10-13/'
RUNNER='scripts/resource_applicability/run_track_a_h6_conditional_s2_pilot_v1.py'
PREPARER='scripts/resource_applicability/prepare_track_a_h6_conditional_s2_pilot_v1.py'
AUDITOR='scripts/resource_applicability/audit_track_a_h6_conditional_s2_pilot_v1.py'
LOCK='.server-preparation/h6-conditional-s2-pilot-v1/one-shot.lock'
SCHEMA='h6_conditional_s2_pilot_manifest_v1'
KIND='H6_CONDITIONAL_S2_COST_ONLY_8_TRAJECTORY_16_WRAPPER_V1'
CELL=dict(id='B2_p5_q8_2nd_r2_K2',method='B2',prefix=5,q=8,formula='2nd',r=2,R=16,K=2)
COMPLETED_RUN=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-hash-reuse-eight-worker-20261012')/'artifacts/resource_applicability/track_a_h6_confirmation_fast_v1/2026-10-12/launch_confirmation_fast_v1'
NATIVE_EXIT=Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261012/confirmation-fast-handoff-v1/coordinator_exit.json')


def read(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size>16*2**20:raise ValueError('PILOT_JSON_PATH_OR_SIZE')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('PILOT_DUPLICATE_JSON_KEY')
            d[k]=v
        return d
    return json.loads(p.read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('PILOT_NONFINITE_JSON')))


def sources(root):
    return sorted(set(inherited_sources(root))|{RUNNER,PREPARER,AUDITOR,
        'src/trottertracks/resource_applicability/h6_conditional_s2_sampling_v1.py',
        'src/trottertracks/resource_applicability/h6_conditional_s2_contract_v1.py',
        'src/trottertracks/resource_applicability/h6_conditional_s2_execution_v1.py'})


def registrations():
    return [dict(index=i,task=dict(cell_id=CELL['id'],cell=deepcopy(CELL),phase=PHASE,
        replica=i,seed=pilot_seed(CELL['id'],i),validate=False,ordinary=False)) for i in range(8)]


def proposed_resources():
    r=inherited_resources();r.update(assigned_cpus=[20,22,23],cost_cpu_sets=[[20]],cost_workers=1)
    return r


def prior_binding(root):
    root=Path(root);names=[SUMMARY+'result_summary_v1.json',SUMMARY+'summary_audit_v1.json',
        SUMMARY+'integrated_confirmation_v1.json',ANALYSIS+'additional_information_proposal_v1.json',
        ANALYSIS+'selected_trajectory_cost_evidence_v1.json',ANALYSIS+'trajectory_metrics_v1.csv']
    hashes={p:file_hash(root/p) for p in names};committed_hashes(root,BASE_RESULT,hashes)
    result=read(root/SUMMARY/'result_summary_v1.json');audit=read(root/SUMMARY/'summary_audit_v1.json')
    carry=result['calls'];receipt=result['completion'];resource=result['resources']
    if (result['source_commit']!='87b6b9f46b7e8ba9abe132ee21478c114b6f46f7'
            or receipt['group_count']!=256 or receipt['wrapper_count']!=512
            or result['coverage']['exploration_groups']!=592
            or carry['known_calls']!={'compile':1700,'control_probe':360,'trajectory':834,'occurrence':3136,'primitive':1470,'reference_matvec':400}
            or carry['reservations']!={'compile':20,'control_probe':4,'trajectory':10,'occurrence':64}
            or carry['historical_unknown_actual_totals'] is not None
            or carry['budget_reset'] or carry['failed_calls_refunded']
            or not resource['all_dispatched_reaped'] or resource['cleanup_errors']):
        raise ValueError('PILOT_COMPLETION_CARRY_BINDING')
    for name,h in audit['native_file_hashes'].items():
        if file_hash(private_path(COMPLETED_RUN/name))!=h:raise ValueError('PILOT_NATIVE_COMPLETION_CHANGED')
    if file_hash(private_path(NATIVE_EXIT))!=audit['native_exit_sha256'] or read(NATIVE_EXIT)['exit_code']!=0:
        raise ValueError('PILOT_NATIVE_EXIT')
    terminal=read(COMPLETED_RUN/'execution_terminal.json')
    if terminal['status']!='H6_CONFIRMATION_FAST_COMPLETE_MANDATORY_STOP' or terminal['reason'] is not None:
        raise ValueError('PILOT_PARENT_NOT_COMPLETE')
    parent=read(COMPLETED_RUN/'frozen_manifest.json')
    for path,h in parent['source_hashes'].items():
        if file_hash(root/path)!=h:raise ValueError('PILOT_INHERITED_SOURCE_CHANGED')
    data=read(root/ANALYSIS/'selected_trajectory_cost_evidence_v1.json')['records']
    target=[x for x in data if x['cell_id']==CELL['id'] and x['phase']=='confirmation']
    if len(target)!=32 or any(x['finite_distribution']!=target[0]['finite_distribution'] for x in target):
        raise ValueError('PILOT_SAVED_FINITE_LAW')
    law=target[0]['finite_distribution']
    if law['orders']!=[0,2] or law['finite_taylor_order']!=2:raise ValueError('PILOT_SAVED_K2')
    current=conditional_binomial(16,law['order_probabilities'][1])
    all_seeds={x['seed'] for x in data}
    # All old exploration seeds, including the other64 cells, are frozen too.
    with (root/ANALYSIS/'trajectory_metrics_v1.csv').open() as f:
        import csv
        for x in csv.DictReader(f):
            seed=int(x['seed']);q=int(next(t[1:] for t in x['cell_id'].split('_') if t.startswith('q')))
            all_seeds.add(seed);all_seeds.update(component_step_seeds(seed,q))
    streams=[dict(master=r['task']['seed'],order=order_seed(r['task']['seed']),
        component_steps=list(component_step_seeds(r['task']['seed'],8))) for r in registrations()]
    new_seeds=[x for row in streams for x in [row['master'],row['order'],*row['component_steps']]]
    if all_seeds&set(new_seeds) or len(new_seeds)!=len(set(new_seeds)):raise ValueError('PILOT_SEED_COLLISION')
    return dict(result_commit=BASE_RESULT,report_hashes=hashes,carry=carry,
        completion_rows_digest=receipt['confirmation_rows_digest'],old_trajectory_seeds_digest=digest(sorted(all_seeds)),
        pilot_stream_seeds=streams,known_old_master_and_step_seeds=len(all_seeds),
        native_completion_hashes=audit['native_file_hashes'],native_exit_sha256=audit['native_exit_sha256'],
        finite_distribution=law,s2_probability=current['s2_probability'],
        roundoff_preparation_binding=parent['roundoff_preparation_binding'])


def accounting(prior,base):
    carry=deepcopy(prior['carry']);caps=deepcopy(base['plan']['caps_proposed'])
    caps.update(compile=1736,trajectory=852)
    new=dict(compile=16,control_probe=0,trajectory=8,occurrence=64,primitive=0,reference_matvec=0)
    total={k:carry['known_calls'][k]+carry['reservations'].get(k,0)+v for k,v in new.items()}
    if any(total[k]>caps[k] for k in total):raise ValueError('PILOT_CUMULATIVE_CAP')
    return dict(**carry,new_demand=new,individual_RTE_event_draws=128,proposed_caps=caps,
        known_plus_reserved_plus_new=total,cap_amendments={'compile':{'old':1720,'proposed':1736},
        'trajectory':{'old':844,'proposed':852}},cap_amendments_approved=False)


def validate_scope(root,m):
    flags=dict(approved=False,runtime_authorization=False,allowed_cpus=[],retry=False,resume=False,
        mandatory_stop=True,next_stage_authorized=False,H8_authorized=False,absolute_launch_command=None)
    if (m.get('schema')!=SCHEMA or m.get('kind')!=KIND or m.get('repository_root')!=str(Path(root).resolve())
            or any(type(m.get(k)) is not type(v) or m[k]!=v for k,v in flags.items())):
        raise ValueError('PILOT_MANIFEST_SCHEMA_FLAGS')
    relative=NAMESPACE+'launch_pilot_v1'
    if (digest(m['registrations'])!=digest(registrations()) or digest(m['target_cell'])!=digest(CELL) or m['phase']!=PHASE
            or digest(m['proposed_resources'])!=digest(proposed_resources()) or slots(m['proposed_resources'])!=1
            or m['observation_policy']!=observation_policy() or m['hash_reuse_policy']!=hash_policy()
            or m['exclusive_output']!=dict(repository_path=relative,absolute_path=str(safe_path(root,relative)))
            or m['one_shot_lock']!=LOCK or m['execution_plan_sealed'] is not True
            or m['conditioning']!=dict(stratum='s2',scope='whole trajectory',R=16,K=2,
                count_law='Binomial(16,p2) conditional M>=1; uniform M-subset',
                component_law='original independent component probabilities',quantum_weights_changed=False)):
        raise ValueError('PILOT_FIXED_SCOPE')


def validate_plan(root,m,*,launch=False):
    root=Path(root);validate_scope(root,m)
    if m['base_manifest']!=read(root/BASE) or m['base_sha256']!=file_hash(root/BASE):raise ValueError('PILOT_BASE')
    if set(m['source_hashes'])!=set(sources(root)):raise ValueError('PILOT_SOURCE_CLOSURE')
    committed_hashes(root,m['source_commit'],m['source_hashes']);verify_science(root)
    base=m['base_manifest'];verify_profile(root,base)
    if environment()!=base['environment']:raise ValueError('PILOT_ENVIRONMENT')
    for p,h in base['input_identity']['parent_raw_hashes'].items():
        if file_hash(safe_path(root,p))!=h:raise ValueError('PILOT_INPUT_BYTES')
    prior=prior_binding(root)
    if (m['prior_binding']!=prior or m['cumulative_accounting']!=accounting(prior,base)
            or m['roundoff_preparation_binding']!=prior['roundoff_preparation_binding']
            or not roundoff_binding(root,m)):
        raise ValueError('PILOT_PRIOR_LAW_CARRY_ROUNDOFF')
    if launch and (safe_path(root,LOCK).exists() or safe_path(root,LOCK).is_symlink()):raise ValueError('PILOT_ONE_SHOT_USED')
    if launch and (safe_path(root,m['exclusive_output']['repository_path']).exists()
            or safe_path(root,m['exclusive_output']['repository_path']).is_symlink()):raise ValueError('PILOT_OUTPUT_USED')


def grant_draft(m):
    return dict(schema='h6_conditional_s2_pilot_authorization_v1',kind=KIND,
        approved_by_user=False,server_usage_approved_by_user=False,conditional_pilot_authorized=False,
        cumulative_cap_amendments_approved=False,manifest_digest=digest(m),source_commit=m['source_commit'],
        profile_sha256=m['base_manifest']['profile']['sha256'],input_identity_digest=digest(m['base_manifest']['input_identity']),
        prior_binding_digest=digest(m['prior_binding']),task_list_digest=digest(m['registrations']),
        conditioning_digest=digest(m['conditioning']),accounting_digest=digest(m['cumulative_accounting']),
        proposed_cumulative_caps=deepcopy(m['cumulative_accounting']['proposed_caps']),
        resources=deepcopy(m['proposed_resources']),observation_policy_digest=digest(m['observation_policy']),
        hash_reuse_policy_digest=digest(m['hash_reuse_policy']),roundoff_binding_digest=digest(m['roundoff_preparation_binding']),
        exclusive_output=m['exclusive_output']['absolute_path'],one_shot_lock=m['one_shot_lock'],
        one_shot=True,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,H8_authorized=False)


def validate_grant(m,g):
    expected=grant_draft(m)
    expected.update(approved_by_user=True,server_usage_approved_by_user=True,
        conditional_pilot_authorized=True,cumulative_cap_amendments_approved=True)
    if digest(g)!=digest(expected) or any(type(g.get(k)) is not bool for k,v in expected.items() if type(v) is bool):
        raise ValueError('NEW_APPROVED_CONDITIONAL_S2_PILOT_GRANT_REQUIRED')


def launch_gate(root,m,g,out):
    validate_grant(m,g);validate_plan(root,m,launch=True)
    if str(Path(out).resolve())!=m['exclusive_output']['absolute_path']:raise ValueError('PILOT_LAUNCH_OUTPUT')
    return fresh_resources(m['proposed_resources'],m['observation_policy'],out)
