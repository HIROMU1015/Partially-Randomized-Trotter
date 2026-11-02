"""Two isolated EXPLORATION contracts. No authorization is issued by this module."""
from pathlib import Path
from types import FunctionType
import copy
import json
from functools import lru_cache
import h46_m5_cost_contract as original
import h46_pr_pilot_contract as pilot
import h46_pr_exploration_design as design
import h46_pr_pilot_storage_runtime as storage

ROOT=design.ROOT
STAGES=design.STAGES
MANIFEST,SIGNAL_RECORDS=original.MANIFEST,original.SIGNAL_RECORDS
sha,read,seal,digest,require=(getattr(original,n) for n in ('sha','read','seal','digest','require'))
save=storage.save
COMPILE_TOLERANCE,OTHER_TOLERANCE=original.COMPILE_TOLERANCE,original.OTHER_TOLERANCE
manifest,task_limits,consumed=original.manifest,original.task_limits,original.consumed
reserve=FunctionType(original.reserve.__code__,dict(original.__dict__,save=save),'reserve',original.reserve.__defaults__)
NEW_PATHS=('scripts/h46_pr_exploration_design.py','scripts/h46_pr_exploration_contract.py',
 'scripts/h46_pr_exploration_runtime.py','scripts/run_h46_pr_exploration.py',
 'scripts/prepare_h46_pr_exploration.py','tests/test_h46_pr_exploration.py',design.PROPOSAL,
 'scripts/prepare_h46_pr_exploration_execution.py',
 'scripts/launch_h46_pr_exploration_acquisition.py',
 'scripts/finalize_h46_pr_exploration_acquisition.py',
 'tests/test_h46_pr_exploration_execution.py')
SOURCE_PATHS=tuple(dict.fromkeys(pilot.SOURCE_PATHS+NEW_PATHS+tuple(x['path'] for x in design.proposal()['inputs'])))
_base_verify=FunctionType(original.verify.__code__,dict(original.__dict__,ROOT=ROOT,SOURCE_PATHS=SOURCE_PATHS,
 STAGES=STAGES),'exploration_base_verify',original.verify.__defaults__)

def pipeline(cfg):
    return digest(dict(SOURCE=cfg['SOURCE'],stage=cfg['stage'],manifest_sha256=sha(ROOT/MANIFEST),
                       decisions=cfg['decisions'],runtime=cfg['runtime']))

@lru_cache(maxsize=16)
def _proposal_tasks(raw_proposal,raw_domain,option_id,stage):
    # Keys include full input bytes. verify() still checks every SOURCE/input
    # SHA and committed byte identity on every invocation, including cache hits.
    option=design.selected_option(json.loads(raw_proposal),option_id)
    counts={r['id']:option['n_E_by_candidate_proposed'][r['id']] for r in design.groups() if r['system']==stage[:2]}
    return counts,design.tasks_for(stage,counts)

def proposal_tasks(cfg):
    counts,tasks=_proposal_tasks((ROOT/design.PROPOSAL).read_bytes(),
        (ROOT/design.pilot.DOMAIN).read_bytes(),cfg['decisions']['option_id'],cfg['stage'])
    require(cfg['decisions']['n_E_by_candidate']==counts,'ADOPTED_COUNTS_MATCH_NAMED_PROPOSAL')
    return copy.deepcopy(tasks)

def verify(cfg):
    _base_verify(cfg)
    require(cfg['cohort']==cfg['plan']['cohort']==design.COHORT,'EXPLORATION_COHORT_BOUND')
    require(cfg['proposal_sha256']==cfg['plan']['proposal_sha256']==sha(ROOT/design.PROPOSAL),'PROPOSAL_SHA')
    require(cfg['tasks']==cfg['plan']['tasks']==proposal_tasks(cfg),'EXACT_EXPLORATION_TASKS')
    require(cfg['call_scope']==cfg['plan']['call_scope']==design.call_scope(cfg['tasks'],'PR'),'EXACT_CALL_SCOPE')
    require(cfg['storage']==cfg['plan']['storage']==storage.storage_policy(),'REUSED_LOSSLESS_STORAGE_POLICY')
    resources=cfg['resources']
    require(all(resources.get(k) is True for k in ('prewarm_readonly_contexts','worker_input_cache'))
       and resources.get('process_start_method')=='fork'
       and resources.get('worker_allocation_policy')=='FRESH_JOINT_CPU_RAM_V1','EXPLORATION_OPTIMIZATIONS')
    indexed={r['path'] for r in cfg['sealed_files']}
    require(cfg['registry_path'] in indexed and cfg['excluded_seeds_path'] in indexed,'REGISTRY_EXCLUSIONS_SEALED')
    expected=design.registry(cfg['tasks'],cfg['SOURCE'],cfg['campaign_id'],design.COHORT,read(cfg['excluded_seeds_path'])['values'])
    require(read(cfg['registry_path'])==expected,'FRESH_EXPLORATION_SEEDS_NO_PILOT_MIXING')
    require(not any(Path(r['path']).is_relative_to(Path(cfg['run_root'])) for r in cfg['sealed_files']),'INPUT_OUTPUT_DISJOINT')

def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_EXPLORATION_EXECUTION_APPROVAL_REQUIRED')
    require(cfg.get('stage') in STAGES and cfg.get('cohort')==design.COHORT,'ONLY_EXPLORATION_TWO_CONTRACTS')
    dec=cfg['decisions']
    require(all(dec.get(k) is True for k in ('sample_design_adopted','finite_sample_limit_accepted',
        'unknown_rare_tail_accepted','resources_storage_policy_accepted','unit_cost_acquisition_only_accepted')),'UNADOPTED_EXPLORATION_DECISIONS')
    require(dec.get('formal_shot_policy_adopted') is False and dec.get('AD03')=='DEFERRED'
        and dec.get('pooling') is False and dec.get('adaptive_top_up') is False
        and dec.get('statistics_execution_authorized') is False,'NO_POOLING_TOPUP_SHOT_RANKING_OR_STATISTICS')
    tasks=proposal_tasks(cfg);scope=design.call_scope(tasks,'PR')
    require(cfg['tasks']==tasks and cfg['call_scope']==scope,'ADOPTED_TASKS_SCOPE')
    require(all(g.get(k)==cfg[k] for k in ('SOURCE','stage','cohort','plan_sha256','pipeline_identity','call_scope'))
       and bool(g.get('grant_id')),'EXACT_EXPLORATION_GRANT')
    require(g.get('retry') is False and g.get('next_phase_authorized') is False
       and g.get('storage_policy_sha256')==digest(cfg['storage']),'ONE_SHOT_STORAGE_BOUND')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['run_root']==cfg['plan']['run_root'] and not root.is_relative_to(ROOT)
        and not ROOT.is_relative_to(root),'ISOLATED_NEW_EXPLORATION_ROOT')
    for key,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:
        require(cfg[key]==g.get(key)==str(root/name),'EXPLORATION_PATH_'+key)
    supervisor={k:str(root.parent/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==g.get('supervisor')==supervisor,'EXPLORATION_SUPERVISOR')
    approval=read(cfg['approval_receipt_path'])
    require(approval.get('execution_authorized') is True and all(approval.get(k)==cfg[k] for k in
      ('SOURCE','stage','cohort','decisions','call_scope','storage')),'EXPLICIT_SEPARATE_SCOPE_APPROVAL')
    require(g.get('approval_reference')==cfg['approval_receipt_path']+'#sha256='+sha(cfg['approval_receipt_path']),'APPROVAL_SHA')
    require(cfg['resources'].get('proxy_policy_adopted') is True
       and cfg['resources'].get('full_output_capacity_adopted') is True,'FRESH_CAPACITY_POLICY_UNADOPTED')
    resources=cfg['resources']
    require(all(type(resources.get(k)) is int and resources[k]>0 for k in
      ('worker_RSS_proxy_bytes','disk_proxy_bytes','inode_proxy')),'POSITIVE_ADOPTED_RESOURCE_PROXIES')
    lease_path=resources.get('joint_admission_path')
    require(lease_path in {r['path'] for r in cfg['sealed_files']}
      and sha(lease_path)==resources.get('joint_admission_sha256'),'SEALED_JOINT_ADMISSION')
    lease=read(lease_path)
    require(lease.get('SOURCE')==cfg['SOURCE'] and lease.get('runtime_caps') is False
      and lease.get('adopted') is True and lease.get('peak_growth_proxy_bytes')==resources['disk_proxy_bytes']
      and type(lease.get('workers',{}).get(cfg['stage'][:2])) is int
      and lease['workers'][cfg['stage'][:2]]>=1,'ADOPTED_JOINT_RESOURCE_LEASE')
    verify(cfg)
