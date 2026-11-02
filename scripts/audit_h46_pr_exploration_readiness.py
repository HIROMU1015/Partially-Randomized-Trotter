"""Read-only readiness audit, with public metadata projection (no seed export)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import h46_pr_exploration_contract as c
import h46_pr_exploration_design as d
from prepare_h46_pr_exploration import capacity_preview

def main():
    p=argparse.ArgumentParser();p.add_argument('--draft-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--prior-config',action='append',required=True);p.add_argument('--private-audit',required=True);a=p.parse_args()
    dest=Path(a.output);private=Path(a.private_audit);private.mkdir(parents=True,exist_ok=False)
    protected={}
    for config in a.prior_config:
        old=c.read(config)
        for row in old.get('sealed_files',old.get('plan',{}).get('sealed_files',[])):
            protected[row['path']]=row
        for path in (Path(config),Path(old['plan_path'])):
            protected[str(path.resolve())]=c.seal(path,'old_contract')
        root=Path(old['run_root'])
        for name in ('CONSUMED.json','completion_witness.json','end_audit.json','STOP.json'):
            path=root/name
            if path.exists():protected[str(path.resolve())]=c.seal(path,'old_closed_run')
    def checked(row):
        path=Path(row['path']);c.require(path.stat().st_size==row['bytes'] and c.sha(path)==row['sha256'],'PRESERVATION:'+row['role'])
        return row['bytes']
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        byte_total=sum(pool.map(checked,protected.values()))
    c.save(private/'preserved_files_PRIVATE.json',list(protected.values()))
    configs=[];summaries=[];rejections=[];mutations=[]
    for stage in d.STAGES:
        config=Path(a.draft_root)/stage/'config_DRAFT_PRIVATE.json';cfg=c.read(config);c.verify(cfg);configs.append(cfg)
        c.require(not cfg['authorized'] and not cfg['grant']['valid'] and not Path(cfg['run_root']).exists()
          and not Path(cfg['marker']).exists(),'NO_EXECUTION_OR_CONSUMPTION')
        for flag in ('--execute','--supervise','--audit'):
            run=subprocess.run([sys.executable,'-B',str(d.ROOT/'scripts/run_h46_pr_exploration.py'),flag,'--config',str(config)],capture_output=True,text=True)
            c.require(run.returncode==2 and 'EXPLICIT_EXPLORATION_EXECUTION_APPROVAL_REQUIRED' in run.stdout,'INVALID_DRAFT_CLI_REFUSAL')
            rejections.append(dict(stage=stage,mode=flag,exit=run.returncode,reason=json.loads(run.stdout)['reason'],PASS=True))
        for key,value in [('cohort','DESIGN_PILOT'),('compiler_settings_hash','wrong'),('tasks',[]),('call_scope',{}),('proposal_sha256','wrong')]:
            bad=copy.deepcopy(cfg);bad[key]=value
            try:c.verify(bad)
            except (ValueError,KeyError):mutations.append(dict(stage=stage,field=key,PASS=True))
            else:raise ValueError('MUTATED_DRAFT_ACCEPTED:'+key)
        summaries.append(dict(SOURCE=cfg['SOURCE'],stage=stage,cohort=cfg['cohort'],
          plan_sha256=cfg['plan_sha256'],closure_digest=cfg['plan']['closure_digest'],sealed_files=len(cfg['sealed_files']),
          call_scope=cfg['call_scope'],config_sha256=c.sha(config),grant_valid=False,
          n_E_adopted=None,scientific_execution_authorized=False,marker_exists=False,run_root_exists=False))
    c.save(dest/'contract_drafts.json',summaries)
    c.save(dest/'execution_refusals.json',dict(cli=rejections,metadata_mutations=mutations,scientific_calls=0,PASS=True))
    source=configs[0]['SOURCE']
    c.require(all(v['SOURCE']==source for v in configs),'SAME_SCIENCE_SOURCE_SEPARATE_CONTRACTS')
    c.save(dest/'source_closure.json',dict(SOURCE=source,files=[dict(path=rel,sha256=c.sha(d.ROOT/rel),bytes=(d.ROOT/rel).stat().st_size) for rel in c.SOURCE_PATHS]))
    origin=c.read(configs[0]['origin_signal_config'])
    identity=[]
    for system in ('H4','H6'):
        info=origin['systems'][system]
        identity.append(dict(system=system,binding_digest=c.digest(info['binding']),
          inputs_digest=c.digest(info['inputs']),files=[{k:row[k] for k in ('role','sha256','bytes') if k in row}
            | {'input_label':label} for label,row in info['inputs'].items()]))
    c.save(dest/'input_identity.json',dict(SOURCE=source,signal_SOURCE=origin['SOURCE'],
      signal_plan_sha256=origin['plan_sha256'],systems=identity,compiler=c.design.COMPILER,
      compiler_settings_hash=c.design.COMPILER_HASH,compile_tolerance=c.COMPILE_TOLERANCE,
      other_tolerance=c.OTHER_TOLERANCE,private_paths_not_published=True))
    exclusion=c.read(configs[0]['excluded_seeds_path'])
    registers=[c.read(cfg['registry_path']) for cfg in configs]
    values=[[v for row in reg.values() for v in row.values()] for reg in registers]
    c.require(not set(values[0])&set(values[1]),'CROSS_SYSTEM_SEED_COLLISION')
    c.save(dest/'seed_audit.json',dict(SOURCE=source,cohort=d.COHORT,proposed_seed_counts=dict(zip(('H4','H6'),map(len,values))),
      excluded_registry_sources=[{k:row[k] for k in ('sha256','bytes')} for row in exclusion['sources']],
      known_seed_count=len(set(exclusion['values'])-set(values[1])),
      source_SHA_registry_verified=True,collisions=0,finite_known_registry_coverage=True,
      all_other_research_seeds_covered=False,seed_values_exported=False,refresh_before_execution=True,
      adopted=False,confirmation_cohort_reuses_exploration=False))
    c.save(dest/'capacity_proposal.json',capacity_preview(Path(a.draft_root)))
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:list(pool.map(checked,protected.values()))
    c.save(dest/'preservation.json',dict(status='PASS',files_checked=len(protected),bytes=byte_total,
      modified_files=0,original_grant_ledger_marker_STOP_rewritten=False,raw_or_archive_rewritten=False,
      check_scope='FROZEN_EVALUATION_INPUTS_SOURCE_RUNTIME_AND_ORIGINAL_CLOSED_RUN_METADATA',
      raw_QPY_NPZ_redecoded=False,other_worktrees_frozen=False,shared_environment_changes=False))
    print(json.dumps(dict(status='PASS',stages=2,CLI_rejections=len(rejections),metadata_mutations=len(mutations),
      protected_files=len(protected),SOURCE=source,scientific_execution_started=0)))

if __name__=='__main__':main()
