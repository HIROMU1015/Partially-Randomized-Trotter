"""Isolated DESIGN_PILOT authorization; disabled drafts cannot read inputs."""
from pathlib import Path
from types import FunctionType
import h46_m5_cost_contract as original
import h46_pr_design_pilot as design

ROOT=Path(__file__).resolve().parents[1]
STAGES=design.STAGES
MANIFEST=original.MANIFEST
SIGNAL_RECORDS=original.SIGNAL_RECORDS
sha,read,save,seal,digest,require=(getattr(original,n) for n in ('sha','read','save','seal','digest','require'))
COMPILE_TOLERANCE=original.COMPILE_TOLERANCE
OTHER_TOLERANCE=original.OTHER_TOLERANCE
manifest=original.manifest
task_limits=original.task_limits
consumed=original.consumed
reserve=original.reserve
NEW_PATHS=('scripts/h46_pr_design_pilot.py','scripts/h46_pr_pilot_contract.py',
 'scripts/h46_pr_pilot_native.py','scripts/run_h46_pr_design_pilot.py',
 'scripts/prepare_h46_pr_design_pilot.py','tests/test_h46_pr_design_pilot.py',
 'references/h46_pr_design_pilot_proposal_v1.json')
SOURCE_PATHS=tuple(dict.fromkeys(original.SOURCE_PATHS+NEW_PATHS))
SOURCE_PATHS+=('artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/run_timing.json',)

# Reuse the unchanged byte/closure/input/runtime/compiler verifier. Only new
# SOURCE paths and accepted stage names differ. Scientific gates are unchanged.
_base_verify=FunctionType(original.verify.__code__,dict(original.__dict__,ROOT=ROOT,
    SOURCE_PATHS=SOURCE_PATHS,STAGES=STAGES), 'pilot_base_verify',original.verify.__defaults__)

def pipeline(cfg):
    return digest(dict(SOURCE=cfg['SOURCE'],stage=cfg['stage'],manifest_sha256=sha(ROOT/MANIFEST),
                       decisions=cfg['decisions'],runtime=cfg['runtime']))

def verify(cfg):
    _base_verify(cfg)
    require(cfg['cohort']==cfg['plan']['cohort']==design.COHORT,'DESIGN_PILOT_COHORT_BOUND')
    require(cfg['proposal_sha256']==cfg['plan']['proposal_sha256']==sha(ROOT/NEW_PATHS[-1]),'PROPOSAL_VERSION_BOUND')
    require(cfg['decisions']['formal_shot_policy_adopted'] is False and cfg['decisions']['AD03']=='DEFERRED','NO_SHOT_OR_MAIN_ADOPTION')

def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_PILOT_EXECUTION_APPROVAL_REQUIRED')
    require(cfg['stage'] in STAGES and cfg.get('cohort')==design.COHORT,'PILOT_ONLY_NO_EXPLORATION')
    dec=cfg['decisions']
    require(all(dec.get(k) is True for k in ('pilot_design_adopted','nominal_variance_precision_limit_accepted',
      'rare_event_limit_accepted','resources_policy_accepted','unit_cost_only_accepted')),'PILOT_DECISIONS_NOT_ADOPTED')
    require(dec.get('formal_shot_policy_adopted') is False and dec.get('AD03')=='DEFERRED','NO_SHOT_OR_MAIN_ADOPTION')
    require(dec.get('adaptive_top_up') is False and dec.get('pooling') is False,'NO_TOP_UP_OR_POOLING')
    require(cfg['resources'].get('proxy_policy_adopted') is True,'RESOURCE_PROXY_POLICY_NOT_ADOPTED')
    tasks=design.tasks_for(cfg['stage'],dec['n_P_by_candidate']);scope=design.call_scope(tasks,'PR')
    resources=cfg['resources'];per_trajectory=resources.get('storage_bytes_per_trajectory_proxy')
    require(resources.get('full_output_capacity_adopted') is True and type(per_trajectory) is int
      and per_trajectory>0 and resources['disk_proxy_bytes']>=per_trajectory*len(tasks),'FULL_PILOT_SCRATCH_CAPACITY_REQUIRED')
    require(dec.get('statistics_execution_authorized') is False,'PILOT_ACQUISITION_ONLY_NOT_NUMERICAL_EVALUATION')
    require(cfg['tasks']==cfg['plan']['tasks']==tasks and cfg['call_scope']==cfg['plan']['call_scope']==scope,'EXACT_PILOT_TASKS_AND_CALLS')
    require(g.get('SOURCE')==cfg['SOURCE'] and g.get('stage')==cfg['stage'] and g.get('cohort')==design.COHORT
      and g.get('plan_sha256')==cfg['plan_sha256'] and g.get('pipeline_identity')==cfg['pipeline_identity']
      and g.get('call_scope')==scope and g.get('grant_id'),'SEPARATE_PILOT_GRANT')
    require(g.get('retry') is False and g.get('next_phase_authorized') is False,'NO_RETRY_OR_EXPLORATION_TRANSITION')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['plan']['run_root']==cfg['run_root'] and not root.is_relative_to(ROOT)
            and not ROOT.is_relative_to(root),'ISOLATED_PILOT_RUN_ROOT')
    for k,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:
        require(cfg[k]==g.get(k)==str(root/name),'PILOT_OUTPUT_PATH_'+k)
    supervisor={k:str(root.parent/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==g.get('supervisor')==supervisor,'PILOT_SUPERVISOR')
    a=read(cfg['approval_receipt_path'])
    require(a.get('execution_authorized') is True and a.get('SOURCE')==cfg['SOURCE'] and a.get('stage')==cfg['stage']
      and a.get('cohort')==design.COHORT and a.get('decisions')==dec and a.get('call_scope')==scope,'EXPLICIT_SCOPE_APPROVAL_RECEIPT')
    require(g.get('approval_reference')==cfg['approval_receipt_path']+'#sha256='+sha(cfg['approval_receipt_path']),'PILOT_APPROVAL_HASH')
    verify(cfg)
    registry=design.registry(tasks,cfg['SOURCE'],cfg['campaign_id'],design.COHORT,read(cfg['excluded_seeds_path'])['values'])
    indexed={r['path'] for r in cfg['sealed_files']}
    require(cfg['registry_path'] in indexed and cfg['excluded_seeds_path'] in indexed and read(cfg['registry_path'])==registry,'FRESH_PILOT_SEEDS_AND_EXCLUSIONS')
    require(not any(Path(r['path']).is_relative_to(root) for r in cfg['sealed_files']),'PILOT_INPUT_OUTPUT_DISJOINT')
