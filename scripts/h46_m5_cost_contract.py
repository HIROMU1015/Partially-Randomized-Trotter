"""Independent DET/PR unit-cost contracts. Invalid drafts fail before science."""
from pathlib import Path
import subprocess
import sys
import h46_m5_cost_design as design
from h6_m5_phase_repair_contract import sha,read,save,seal,digest,require
import h46_m5_signal_contract as signal_contract

ROOT=Path(__file__).resolve().parents[1]
STAGES=('H4_DET','H6_DET','H4_PR','H6_PR')
MANIFEST='references/h46_m5_cost_readiness_v1.json'
SIGNAL_RECORDS='artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json'
NEW_PATHS=('scripts/h46_m5_cost_design.py','scripts/h46_m5_cost_contract.py',
 'scripts/h46_m5_cost_native.py','scripts/run_h46_m5_cost.py',
 'scripts/prepare_h46_m5_cost_readiness.py','tests/test_h46_m5_cost_readiness.py',MANIFEST,SIGNAL_RECORDS)
SOURCE_PATHS=tuple(dict.fromkeys(signal_contract.SOURCE_PATHS+NEW_PATHS))
COMPILE_TOLERANCE=1e-7
OTHER_TOLERANCE=1e-9

def original_signal(config_path):
    cfg=read(config_path)
    candidates=[Path(r['path']) for r in cfg['sealed_files'] if r['role']=='signal_SOURCE:scripts/h46_m5_signal_contract.py']
    require(len(candidates)==1,'ONE_ORIGINAL_SIGNAL_CONTRACT')
    p=candidates[0];root=p.parents[1]
    require(p.read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':scripts/h46_m5_signal_contract.py'],cwd=root),'ORIGINAL_SIGNAL_SOURCE_BYTES')
    code='import sys;sys.path.insert(0,sys.argv[1]);import h46_m5_signal_contract as c;v=c.read(sys.argv[2]);c.authorize(v);c.consumed(v)'
    r=subprocess.run([sys.executable,'-B','-c',code,str(root/'scripts'),str(config_path)],capture_output=True,text=True)
    require(r.returncode==0,'ORIGINAL_SIGNAL_SEALS:'+r.stderr[-1500:])
    require(read(Path(cfg['run_root'])/'completion_witness.json')['status']=='SIGNAL_COMPLETE_MANDATORY_STOP','SIGNAL_COMPLETED')
    return cfg

def manifest():return read(ROOT/MANIFEST)

def expected_tasks(stage,n_by_candidate):
    m=manifest();reuse={(r['candidate_id'],r['axis']) for r in m['reuse_axes'] if r['identity_join']=='MATCHED_WITHIN_M5_NATIVE_SERIES'}
    return design.tasks_for(stage,m['candidate_table'],reuse,n_by_candidate)

def task_limits(task):
    scope=design.call_scope([task],'PR' if task['candidate']['method']=='PR' else 'DET')
    return {'TASK':1,**{k:scope[k] for k in ('trajectories','request_sampling_calls','native_piece_compositions','native_evolution','native_low_actions','native_high_actions','independent_oracle_actions','uncompiled_wrapper_actions','compiled_wrapper_actions','compile_axes')}}

def verify(cfg):
    require(cfg['stage'] in STAGES,'COST_STAGE_ONLY_NOT_PF788')
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'COST_PLAN_BYTES')
    require(cfg['plan']['SOURCE']==cfg['SOURCE'] and cfg['plan']['stage']==cfg['stage'],'COST_SOURCE_STAGE_BINDING')
    rows=cfg['sealed_files'];indexed={r['path']:r for r in rows}
    require(len(rows)==len(indexed) and digest(rows)==cfg['plan']['closure_digest'],'COST_CLOSURE')
    for r in rows:
        p=Path(r['path']);require(str(p.resolve())==r['path'] and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'SEALED_BYTES:'+r['role'])
    for rel in SOURCE_PATHS:
        p=ROOT/rel;require(str(p.resolve()) in indexed and p.read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'COMMITTED_COST_SOURCE:'+rel)
    require(cfg['origin_signal_config'] in indexed,'ORIGINAL_SIGNAL_CONFIG_SEALED')
    origin=original_signal(cfg['origin_signal_config'])
    require(origin['SOURCE']==manifest()['signal_SOURCE'] and origin['plan_sha256']==manifest()['signal_plan_sha256'],'FIXED_RUN02_ORIGIN')
    require(cfg['systems']==origin['systems'] and cfg['runtime']==origin['runtime'] and cfg['provider_root']==origin['provider_root'],'SAME_SIGNAL_INPUT_RUNTIME')
    require(manifest()['signal_run_commit']==design.BASE and manifest()['native_policy']==design.NATIVE_POLICY,'M5_SERIES_NOT_OLD_MAIN_PROFILE')
    require(cfg['compiler']==design.COMPILER and cfg['compiler_settings_hash']==design.COMPILER_HASH,'FIXED_ORIGINAL_COMPILER')
    require(cfg['compile_tolerance']==COMPILE_TOLERANCE and cfg['other_tolerance']==OTHER_TOLERANCE,'CT01_NOT_SHOT_ALLOWANCE')
    require(cfg['decisions']==cfg['plan']['decisions'] and cfg['resources']==cfg['plan']['resources'],'DECISIONS_RESOURCES_BOUND')
    require(all(cfg[k]==cfg['plan'][k] for k in ('campaign_id','registry_path','excluded_seeds_path')),'SEED_PATHS_AND_NAMESPACE_BOUND')
    require(cfg['resources']['runtime_caps'] is False and cfg['resources']['host_PSI_kill'] is False,'NO_SHARED_RESOURCE_KILL_OR_ARBITRARY_CAP')
    require(cfg['approval_receipt_path'] in indexed and sha(cfg['approval_receipt_path'])==cfg['approval_receipt_sha256']==cfg['plan']['approval_receipt_sha256'],'APPROVAL_RECEIPT_SEALED')
    require(cfg['pipeline_identity']==cfg['plan']['pipeline_identity']==digest(dict(SOURCE=cfg['SOURCE'],stage=cfg['stage'],manifest_sha256=sha(ROOT/MANIFEST),decisions=cfg['decisions'],runtime=cfg['runtime'])),'COST_PIPELINE_BINDING')

def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_UNIT_COST_EXECUTION_APPROVAL_REQUIRED')
    require(cfg['stage'] in STAGES,'FOUR_ISOLATED_COST_STAGES')
    decisions=cfg['decisions']
    require(decisions.get('cost_domain_accepted') is True and decisions.get('empirical_signal_selection_limit_accepted') is True and decisions.get('resources_policy_accepted') is True,'UNADOPTED_COST_DECISIONS')
    require(decisions.get('formal_shot_policy_adopted') is False,'UNIT_COST_ONLY_NO_SHOT_ADOPTION')
    if cfg['stage'].endswith('PR'):
        require(decisions.get('exploration_design_accepted') is True and decisions.get('rare_event_limit_accepted') is True,'EXPLORATORY_DESIGN_AND_RARE_LIMIT_REQUIRED')
    tasks=expected_tasks(cfg['stage'],decisions.get('n_by_candidate',{}));scope=design.call_scope(tasks,cfg['stage'].split('_')[1])
    require(cfg['tasks']==tasks and cfg['plan']['tasks']==tasks and cfg['call_scope']==cfg['plan']['call_scope']==scope,'EXACT_TASKS_AND_CALL_BUDGET')
    require(g.get('SOURCE')==cfg['SOURCE'] and g.get('stage')==cfg['stage'] and g.get('plan_sha256')==cfg['plan_sha256'] and g.get('pipeline_identity')==cfg['pipeline_identity'] and g.get('call_scope')==scope and g.get('grant_id'),'EXACT_SEPARATE_GRANT')
    require(g.get('retry') is False and g.get('next_phase_authorized') is False,'NO_RETRY_NO_CROSS_STAGE_APPROVAL')
    root=Path(cfg['run_root']).resolve();require(str(root)==cfg['plan']['run_root']==cfg['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'NEW_UNIT_COST_ROOT')
    for k,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:require(cfg[k]==g.get(k)==str(root/name),'COST_PATH:'+k)
    supervisor={k:str(root.parent/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==g.get('supervisor')==supervisor,'COST_SUPERVISOR_PATHS')
    a=read(cfg['approval_receipt_path'])
    require(a.get('execution_authorized') is True and a.get('SOURCE')==cfg['SOURCE'] and a.get('stage')==cfg['stage'] and a.get('decisions')==decisions and a.get('call_scope')==scope,'NEW_EXPLICIT_SCOPED_APPROVAL')
    require(g.get('approval_reference')==cfg['approval_receipt_path']+'#sha256='+sha(cfg['approval_receipt_path']),'APPROVAL_REFERENCE')
    verify(cfg)
    if cfg['stage'].endswith('PR'):
        from h46_m5_contract import core
        excluded=read(cfg['excluded_seeds_path'])['values'];registry=design.registry(tasks,cfg['SOURCE'],cfg['campaign_id'],'EXPLORATION',excluded,core.schedule)
        require(read(cfg['registry_path'])==registry and cfg['registry_path'] in {r['path'] for r in cfg['sealed_files']} and cfg['excluded_seeds_path'] in {r['path'] for r in cfg['sealed_files']},'FRESH_EXCLUDED_EXPLORATION_REGISTRY')
    require(not any(Path(r['path']).is_relative_to(root) for r in cfg['sealed_files']),'COST_INPUT_OUTPUT_DISJOINT')

def consumed(cfg):require(read(cfg['marker'])==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']),'COST_CONSUMED_MARKER')

def reserve(cfg,task,kind,index=0):
    consumed(cfg);require(task in cfg['tasks'],'REGISTERED_COST_TASK')
    require(kind in task_limits(task) and type(index) is int and 0<=index<task_limits(task)[kind],'EXACT_CALL_KIND_INDEX')
    save(Path(cfg['ledger'])/(task['id']+'__'+kind+'__'+str(index)+'.json'),dict(task=task['id'],kind=kind,index=index,SOURCE=cfg['SOURCE'],stage=cfg['stage'],state='RESERVED_NONREFUNDABLE',actual_call_completion_separate=True))
