"""Three independent confirmation packs. Drafts cannot authorize science."""
import copy
from functools import lru_cache
import json
from pathlib import Path
import subprocess
from types import FunctionType,SimpleNamespace
import h246_confirmation_design as d
import h46_m5_cost_contract as old
import h46_pr_pilot_contract as pilot
import h46_pr_design_pilot as law
import h46_pr_pilot_storage_runtime as storage

ROOT=d.ROOT
STAGES=('H2_CONFIRMATION','H4_CONFIRMATION','H6_CONFIRMATION')
read,sha,digest,require=old.read,old.sha,old.digest,old.require
save,seal=storage.save,old.seal
manifest,task_limits,consumed=old.manifest,old.task_limits,old.consumed
COMPILE_TOLERANCE,OTHER_TOLERANCE=1e-7,1e-9
SIGNAL_RECORDS=old.SIGNAL_RECORDS
NEW_PATHS=('scripts/h246_confirmation_design.py','scripts/h246_confirmation_contract.py',
 'scripts/h246_confirmation_runtime.py','scripts/prepare_h246_confirmation.py',
 'scripts/run_h246_confirmation.py','scripts/launch_h246_confirmation.py',
 'scripts/h246_confirmation_evaluation.py','scripts/h2_outer32_contract.py',
 'scripts/h2_outer32_runtime.py','scripts/h2_m5_compiler_phase_guard.py',
 'tests/test_h246_confirmation_design.py','tests/test_h246_confirmation_readiness.py',d.PROPOSAL)
SOURCE_PATHS=tuple(dict.fromkeys(pilot.SOURCE_PATHS+NEW_PATHS+(d.RESULT,)))
PLAN_KEYS=('SOURCE','stage','cohort','tasks','call_scope','campaign_id','registry_path','excluded_seeds_path',
 'runtime','provider_root','systems','signal_path','binding','inputs','reference_path','preparation_hash',
 'compiler_phase_policy','compiler_settings_hash','compile_tolerance','other_tolerance','storage','resources',
 'run_root','output','ledger','marker','supervisor','origin_config_path','proposal_sha256','comparison',
 'analysis_scope','approval_receipt_path','approval_receipt_sha256','decisions')

def proposal():return read(ROOT/d.PROPOSAL)
def comparison(stage):
    require(stage in STAGES,'PRIMARY_THREE_ONLY_NO_AUXILIARY')
    return copy.deepcopy(next(p for p in proposal()['comparisons'] if p['id']==stage[:2]+'_PRIMARY'))
def tasks_for(stage):
    p=comparison(stage);tasks=[]
    for group,n in zip(p['groups'],p['recommended']['n_C']):
        require(type(n) is int and n>=2,'FIXED_PROPOSED_N_C')
        tasks.extend(dict(id='HC1__'+group['candidate']['id']+'__'+str(i),candidate=group['candidate'],
          sample=i,axes=['cosine','sine'],stage=stage) for i in range(n))
    return tasks

def analysis_scope(stage):
    return dict(fixed_comparisons=1,groups=2,epsilon=.0005,mode='USUAL_SIGNED',allowance_multiplier=1,
      frozen_shot_reuse=True,new_shot_calculation=0,new_ranking=0,
      paired_within_trajectory=True,cross_candidate_pairing=False,pooling=False,adaptive_top_up=False,
      family_size=3,family_alpha=.05,interval='NORMAL_BONFERRONI_APPROXIMATE_NOT_CERTIFIED',
      rare_cost_imputation=False,AD03='DEFERRED',mandatory_STOP=True)

def decisions():
    return dict(candidate_reselection=False,pooling=False,adaptive_top_up=False,rare_cost_imputation=False,
      new_shot_policy=False,AD03='DEFERRED',scientific_execution_adopted=False)

@lru_cache(maxsize=8)
def source_blobs(source):
    raw=subprocess.check_output(['git','cat-file','--batch'],cwd=ROOT,
      input=('\n'.join(source+':'+p for p in SOURCE_PATHS)+'\n').encode())
    pos=0;out={}
    for path in SOURCE_PATHS:
        end=raw.index(b'\n',pos);header=raw[pos:end].split()
        require(len(header)==3 and header[1]==b'blob','COMMITTED_CONFIRMATION_SOURCE:'+path)
        size=int(header[2]);out[path]=raw[end+1:end+1+size];pos=end+size+2
    require(pos==len(raw),'EXACT_SOURCE_STREAM');return out

def verify(cfg):
    require(cfg['stage'] in STAGES and cfg['cohort']==d.COHORT,'PRIMARY_CONFIRMATION_COHORT')
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'PLAN_SHA')
    require(all(cfg.get(k)==cfg['plan'].get(k) for k in PLAN_KEYS),'ALL_PLAN_BINDINGS')
    require(cfg['proposal_sha256']==sha(ROOT/d.PROPOSAL),'PROPOSAL_SHA')
    require(sha(ROOT/d.RESULT)==d.RESULT_SHA,'EXACT_FROZEN_SHOT_AND_DESIGN_INPUT')
    require(cfg['tasks']==tasks_for(cfg['stage']) and cfg['comparison']==comparison(cfg['stage']),'FIXED_CANDIDATES_N_SHOTS')
    require(cfg['analysis_scope']==analysis_scope(cfg['stage']),'FIXED_ANALYSIS_NO_RESELECTION')
    require(cfg['call_scope']==law.call_scope(cfg['tasks'],'PR'),'EXACT_ACQUISITION_CALLS')
    require(cfg['storage']==storage.storage_policy(),'BYTE_LOSSLESS_STORAGE_UNCHANGED')
    require(cfg['compiler_settings_hash']==law.COMPILER_HASH and cfg['compile_tolerance']==COMPILE_TOLERANCE
      and cfg['other_tolerance']==OTHER_TOLERANCE,'ORIGINAL_COMPILER_THRESHOLDS')
    rows=cfg['sealed_files'];index={r['path']:r for r in rows}
    require(len(index)==len(rows) and digest(rows)==cfg['plan']['closure_digest'],'UNIQUE_SEALED_CLOSURE')
    for r in rows:
        path=Path(r['path'])
        require(str(path.resolve())==r['path'] and path.stat().st_size==r['bytes']
          and sha(path)==r['sha256'],'SEALED_SHA:'+r['role'])
    for rel,blob in source_blobs(cfg['SOURCE']).items():
        path=ROOT/rel
        require(str(path.resolve()) in index and path.read_bytes()==blob,'CURRENT_SOURCE_IDENTITY:'+rel)
    for key in ('registry_path','excluded_seeds_path','origin_config_path','signal_path','approval_receipt_path'):
        require(cfg[key] in index,'BOUND_INPUT:'+key)
    require(sha(cfg['approval_receipt_path'])==cfg['approval_receipt_sha256'],'APPROVAL_SHA')
    origin=read(cfg['origin_config_path'])
    for key in ('systems','runtime','provider_root'):
        require(cfg[key]==origin[key],'REUSE_EXPLORATION_INPUT:'+key)
    if cfg['stage']=='H2_CONFIRMATION':
        for key in ('inputs','binding','preparation_hash','reference_path','compiler_phase_policy','signal_path'):
            require(cfg[key]==origin[key],'H2_UNCHANGED_INPUT_GUARD:'+key)
    else:require(cfg['signal_path']==str(ROOT/old.SIGNAL_RECORDS),'FIXED_H46_SIGNAL')
    expected=law.registry(cfg['tasks'],cfg['SOURCE'],cfg['campaign_id'],d.COHORT,
      read(cfg['excluded_seeds_path'])['values'])
    require(read(cfg['registry_path'])==expected,'FRESH_SEEDS_DISJOINT_KNOWN_COHORTS')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'NEW_EXTERNAL_RUN_ROOT')
    require(not any(Path(r['path']).is_relative_to(root) for r in rows),'READ_INPUT_WRITE_OUTPUT_DISJOINT')
    for key,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:
        require(cfg[key]==str(root/name),'OWN_OUTPUT_PATH:'+key)
    supervisor={k:str(root.parent/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),
      ('exit','external_exit_witness_PRIVATE.json'),('log','native_PRIVATE.log')]}
    require(cfg['supervisor']==supervisor,'OWN_SUPERVISOR_PATHS')
    r=cfg['resources']
    require(r['runtime_caps'] is False and r['host_PSI_kill'] is False and r['process_start_method']=='fork'
      and r['prewarm_readonly_contexts'] and r['worker_input_cache'],'LOCAL_EFFICIENT_RESOURCE_POLICY')
    require(all(type(r[k]) is int and r[k]>0 for k in ('worker_RSS_proxy_bytes','raw_pair_proxy_bytes',
      'final_growth_proxy_bytes','inode_proxy')),'POSITIVE_PLANNING_PROXIES_NOT_CAPS')

def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_NEW_CONFIRMATION_SCIENCE_APPROVAL_REQUIRED')
    require(cfg.get('stage') in STAGES,'THREE_PRIMARY_CONTRACTS_ONLY')
    require(cfg['decisions']==dict(decisions(),scientific_execution_adopted=True),'EXACT_CONFIRMATION_ADOPTION')
    require(all(g.get(k)==cfg.get(k) for k in ('SOURCE','stage','cohort','plan_sha256','call_scope',
      'analysis_scope','output','ledger','marker','supervisor')) and bool(g.get('grant_id')),'EXACT_ONE_SHOT_GRANT')
    require(g.get('retry') is False and g.get('next_phase_authorized') is False,'NO_RETRY_OR_NEXT_PHASE')
    a=read(cfg['approval_receipt_path'])
    require(a.get('execution_authorized') is True and all(a.get(k)==cfg[k] for k in
      ('SOURCE','stage','cohort','comparison','call_scope','analysis_scope','storage','decisions')),'USER_EXACT_SCOPE_ADOPTION_REQUIRED')
    require(g.get('approval_reference')==cfg['approval_receipt_path']+'#sha256='+sha(cfg['approval_receipt_path']),'GRANT_APPROVAL_SHA')
    verify(cfg)

reserve=FunctionType(old.reserve.__code__,dict(old.__dict__,save=save),'reserve',old.reserve.__defaults__)

def native_contract(cfg):
    view=SimpleNamespace(**{k:globals()[k] for k in ('ROOT','sha','read','save','require','digest',
      'authorize','verify','consumed','reserve','task_limits','storage','manifest','COMPILE_TOLERANCE','OTHER_TOLERANCE')})
    view.SIGNAL_RECORDS=cfg['signal_path']
    view.design=SimpleNamespace(COHORT=d.COHORT,event_stratum=law.event_stratum,
       COMPILER_HASH=law.COMPILER_HASH,NATIVE_POLICY=law.NATIVE_POLICY)
    return view

def activate(draft_path,approval_path,destination):
    """Future operator action ONLY after explicit user approval; no launch.

    Reads an externally recorded affirmative receipt. This readiness task never
    calls this function. Creates new files; never updates the disabled pack.
    """
    cfg=copy.deepcopy(read(draft_path));verify(cfg)
    require(cfg['authorized'] is False and cfg['grant']['valid'] is False,'UNUSED_DISABLED_PACK')
    receipt=read(approval_path);adopted=dict(decisions(),scientific_execution_adopted=True)
    require(receipt.get('execution_authorized') is True and all(receipt.get(k)==cfg[k] for k in
      ('SOURCE','stage','cohort','comparison','call_scope','analysis_scope','storage'))
      and receipt.get('decisions')==adopted,'EXPLICIT_USER_RECEIPT_EXACT_CANDIDATES_COUNTS_AND_BUDGET')
    # No stale SOURCE/input, seed or runtime bypass when authorization is issued.
    dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=False);root=dest/'run01'
    approval=dest/'approval_APPROVED_PRIVATE.json';save(approval,receipt)
    prior=cfg['approval_receipt_path'];cfg.update(authorized=True,decisions=adopted,
      approval_receipt_path=str(approval),approval_receipt_sha256=sha(approval),run_root=str(root),
      output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),
      supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),
        ('exit','external_exit_witness_PRIVATE.json'),('log','native_PRIVATE.log')]})
    cfg['sealed_files']=[r for r in cfg['sealed_files'] if r['path']!=prior]+[seal(approval,'explicit_new_user_approval')]
    plan=dict(status='APPROVED_ONE_SHOT_CONFIRMATION_NOT_YET_STARTED',**{k:cfg[k] for k in PLAN_KEYS},
      closure_digest=digest(cfg['sealed_files']))
    pp=dest/'plan_APPROVED_PRIVATE.json';save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=sha(pp))
    cfg['grant']=dict(valid=True,grant_id=cfg['stage']+'_RUN01_'+digest(plan),retry=False,next_phase_authorized=False,
      approval_reference=str(approval)+'#sha256='+sha(approval),
      **{k:cfg[k] for k in ('SOURCE','stage','cohort','plan_sha256','call_scope','analysis_scope','output','ledger','marker','supervisor')})
    authorize(cfg);path=dest/'config_APPROVED_PRIVATE.json';save(path,cfg);save(dest/'grant_APPROVED_PRIVATE.json',cfg['grant'])
    return path
