"""H2-only staged one-shot extension, explicitly authorized through acquisition.

No scientific imports. Historical H2 and active H4/H6 contracts are never edited.
"""
import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys
import h46_pr_pilot_storage_runtime as storage
import h46_pr_design_pilot as previous
import h46_pr_pilot_contract as old_contract
import h46_m5_cost_contract as original

ROOT=Path(__file__).resolve().parents[1]
sha,read,seal,digest,require=original.sha,original.read,original.seal,original.digest,original.require
save=storage.save
COMPILE_TOLERANCE=1e-7
OTHER_TOLERANCE=1e-9
STAGES=('H2_SIGNAL_OUTER32','H2_DESIGN_PILOT','H2_PILOT_DESIGN','H2_PR_EXPLORATION')
FORMULAS=('2nd','4th','4th(m5_best)')
NEW_PATHS=('scripts/h2_outer32_contract.py','scripts/h2_outer32_runtime.py','scripts/h2_outer32_pipeline.py',
 'scripts/run_h2_outer32_stage.py','scripts/prepare_h2_outer32.py','scripts/h2_m5_compiler_phase_guard.py',
 'tests/test_h2_outer32.py')
SOURCE_PATHS=tuple(dict.fromkeys(old_contract.SOURCE_PATHS+NEW_PATHS+
 ('scripts/h46_pr_pilot_evaluation.py','scripts/h46_pr_exploration_runtime.py','scripts/h46_pr_exploration_contract.py',
  'scripts/h46_pr_exploration_design.py','scripts/prepare_h46_pr_exploration_execution.py',
  'scripts/launch_h46_pr_exploration_acquisition.py','scripts/finalize_h46_pr_exploration_acquisition.py',
  'tests/test_h46_pr_exploration_execution.py','references/h46_pr_exploration_proposal_v1.json')))

@lru_cache(maxsize=4)
def _groups():
    rows=[]
    for formula in FORMULAS:
        for q in (1,2,4,8):
            r=dict(system='H2',method='PR',formula=formula,prefix=1,q=q,K=4,
                r_policy='ABS_TIME_PER_OUTER',r_value=32,kind='PR_SIGNAL',role='PRIMARY',input_series='H2_TOL_ONLY_DF3')
            r['id']='H2O32_'+digest(r)[:16];rows.append(r)
    return rows

def groups():return copy.deepcopy(_groups())
def cohort(stage):return 'DESIGN_PILOT' if stage=='H2_DESIGN_PILOT' else 'EXPLORATION' if stage=='H2_PR_EXPLORATION' else None

def tasks_for(stage,counts):
    require(stage in STAGES[1::2] and set(counts)=={r['id'] for r in groups()},'H2_EXACT12_GROUP_COUNTS')
    result=[]
    for row in groups():
        n=counts[row['id']];require(type(n) is int and n>=33,'H2_MIN33_ADOPTED')
        result.extend(dict(id=('H2DP1__' if cohort(stage)=='DESIGN_PILOT' else 'H2E1__')+row['id']+'__'+str(i),
            candidate=row,sample=i,axes=['cosine','sine'],stage=stage) for i in range(n))
    return result

registry=previous.registry
event_stratum=previous.event_stratum
task_limits=original.task_limits

def counts_from_evaluation(value):
    require(value['status']=='H2_PILOT_DESIGN_PASS' and len(value['groups'])==12,'COMPLETE_H2_PILOT_DESIGN')
    rows={g['candidate']['id']:g for g in value['groups']};counts={};unresolved=[]
    require(set(rows)=={r['id'] for r in groups()},'NO_GROUP_SELECTION')
    for key,g in rows.items():
        require(g['statistics']['n']==33,'PILOT33_ONLY')
        proposal=next(r['n_E_nominal_only'] for r in g['statistics']['n_E_sensitivity']
            if r['relative_MCSE_target']==.02 and r['variance_inflation']==4)
        # Predeclared finite-sample minimum, never a zero-variance guarantee.
        if proposal is None:counts[key]=33;unresolved.append(key)
        else:require(type(proposal) is int and proposal>=0,'FINITE_N_PROPOSAL');counts[key]=max(33,proposal)
    return counts,unresolved

def scope(stage,tasks):
    if stage=='H2_SIGNAL_OUTER32':return dict(candidate_signals=12,full_H_reference_jobs=1,snapshot_restores=1,new_sampling=0,new_compile=0,new_shot=0,ranking=False)
    if stage=='H2_PILOT_DESIGN':return dict(groups=12,pilot_trajectories=396,scenarios=108,new_sampling=0,new_compile=0,new_shot=0,ranking=False)
    return previous.call_scope(tasks,'PR')

@lru_cache(maxsize=4)
def source_blobs(source):
    # Immutable git objects cached; CURRENT files and all input hashes remain
    # checked on every verify. One process replaces many git subprocesses.
    names=[source+':'+rel for rel in SOURCE_PATHS]
    raw=subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join(names)+'\n').encode(),cwd=ROOT)
    pos=0;values={}
    for rel in SOURCE_PATHS:
        end=raw.index(b'\n',pos);header=raw[pos:end].split();require(len(header)==3 and header[1]==b'blob','COMMITTED_H2_BLOB')
        n=int(header[2]);values[rel]=raw[end+1:end+1+n];pos=end+1+n+1
    require(pos==len(raw),'EXACT_SOURCE_OBJECT_STREAM');return values

def verify(cfg):
    require(cfg['stage'] in STAGES and read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'H2_PLAN_SHA')
    p=cfg['plan'];require(p==plan_fields(cfg),'H2_PLAN_BINDINGS')
    rows=cfg['sealed_files'];indexed={r['path']:r for r in rows}
    require(len(rows)==len(indexed) and p['closure_digest']==digest(rows),'H2_UNIQUE_CLOSURE')
    for r in rows:
        path=Path(r['path']);require(str(path.resolve())==r['path'] and path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],'SEALED_BYTES:'+r['role'])
    for rel,blob in source_blobs(cfg['SOURCE']).items():
        path=ROOT/rel;require(str(path) in indexed and path.read_bytes()==blob,'COMMITTED_H2_SOURCE:'+rel)
    require(cfg['binding']['rank']==3 and cfg['binding']['n_qubits']==4 and cfg['binding']['rank_policy']=='tol_only_1e-8'
        and cfg['binding']['T']==.8 and cfg['binding']['P']==0,'FIXED_H2_INPUT_BINDING')
    require(cfg['runtime']['qiskit']=='1.3.0' and cfg['runtime']['provider_root']==cfg['provider_root'],'H2_COMMON_RUNTIME')
    require(sys.executable==cfg['runtime']['python'],'FIXED_H2_COMMON_PYTHON')
    require(cfg['storage']==storage.storage_policy(),'FIXED_BYTE_STORAGE')
    require(cfg['resources']['runtime_caps'] is False and cfg['resources']['host_PSI_kill'] is False,'NO_SHARED_RESOURCE_KILL')
    require(cfg['compiler_phase_policy']=='H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1','EXISTING_H2_PHASE_POLICY')
    expected=[]
    if cfg['stage'] in (STAGES[1],STAGES[3]):
        counts={r['id']:33 for r in groups()} if cfg['stage']==STAGES[1] else counts_from_evaluation(read(cfg['evaluation_path']))[0]
        expected=tasks_for(cfg['stage'],counts)
        require(cfg['tasks']==expected,'EXACT_H2_STAGE_TASKS')
        expected_registry=registry(expected,cfg['SOURCE'],cfg['campaign_id'],cohort(cfg['stage']),read(cfg['excluded_seeds_path'])['values'])
        require(read(cfg['registry_path'])==expected_registry,'H2_FRESH_COHORT_SEEDS')
        require(cfg['signal_path'] in indexed and cfg['registry_path'] in indexed and cfg['excluded_seeds_path'] in indexed,'H2_SEALED_SCIENCE_INPUTS')
    require(cfg['call_scope']==scope(cfg['stage'],expected),'H2_EXACT_STAGE_SCOPE')
    root=Path(cfg['run_root']);require(root==root.resolve() and not root.is_relative_to(ROOT),'NEW_PRIVATE_H2_ROOT')
    require(not any(Path(r['path']).is_relative_to(root) for r in rows),'H2_INPUT_OUTPUT_DISJOINT')

def plan_fields(cfg):
    keys=('SOURCE','stage','runtime','provider_root','inputs','binding','resources','storage','compiler_phase_policy',
        'tasks','call_scope','campaign_id','registry_path','excluded_seeds_path','signal_path','preparation_hash',
        'evaluation_path','reference_path','systems','run_root','output','ledger','marker','supervisor','approval_path')
    return dict(**{k:cfg[k] for k in keys},closure_digest=digest(cfg['sealed_files']))

def authorize(cfg):
    g=cfg.get('grant',{});require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_H2_PIPELINE_APPROVAL_REQUIRED')
    require(g.get('SOURCE')==cfg['SOURCE'] and g.get('stage')==cfg['stage'] and g.get('plan_sha256')==cfg['plan_sha256']
        and g.get('call_scope')==cfg['call_scope'] and bool(g.get('grant_id')),'H2_SEPARATE_EXACT_GRANT')
    require(g.get('retry') is False and g.get('shot_accounting') is False and g.get('ranking') is False
        and g.get('independent_confirmation') is False,'NO_H2_RETRY_SHOT_RANKING_CONFIRMATION')
    approval=read(cfg['approval_path']);require(approval['SOURCE']==cfg['SOURCE'] and approval['authorized_stages']==list(STAGES)
        and approval['no_intermediate_reapproval'] is True,'USER_H2_THROUGH_ACQUISITION_APPROVAL')
    root=Path(cfg['run_root'])
    for key,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:require(cfg[key]==str(root/name),'H2_OWN_PATH:'+key)
    require(cfg['supervisor']=={k:str(root.parent/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]},'H2_OWN_SUPERVISOR')
    verify(cfg)

def consumed(cfg):require(read(cfg['marker'])==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']),'H2_CONSUMED_MARKER')

def reserve(cfg,task,kind,index=0):
    consumed(cfg);require(task in cfg['tasks'] and kind in task_limits(task) and type(index) is int and 0<=index<task_limits(task)[kind],'REGISTERED_H2_CALL')
    save(Path(cfg['ledger'])/(task['id']+'__'+kind+'__'+str(index)+'.json'),
        dict(SOURCE=cfg['SOURCE'],stage=cfg['stage'],task=task['id'],kind=kind,index=index,state='RESERVED_NONREFUNDABLE'))
