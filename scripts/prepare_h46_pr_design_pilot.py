"""Generate proposals and seal disabled pilot drafts; never issue valid grants."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import h46_pr_design_pilot as d
import h46_pr_pilot_contract as c

PROPOSAL='references/h46_pr_design_pilot_proposal_v1.json'
TIMING='artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/run_timing.json'

def write_proposal():
    c.save(d.ROOT/PROPOSAL,d.proposal())

def seal_drafts(source,signal_config,excluded_seeds,private):
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=True)
    origin=c.original.original_signal(signal_config)
    c.require(origin['SOURCE']==c.manifest()['signal_SOURCE'],'EXACT_SIGNAL_ORIGIN')
    files={r['path']:r for r in origin['sealed_files']}
    item=c.seal(Path(signal_config).resolve(),'pilot_original_signal_config')
    files[item['path']]=item
    c.require(c.read(d.ROOT/PROPOSAL)==d.proposal(),'PROPOSAL_MATCHES_METADATA_GENERATOR')
    import run_h46_pr_design_pilot,concurrent.futures.process,concurrent.futures.thread
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).resolve().is_relative_to('/usr/lib/python3.12'):
            item=c.seal(Path(p).resolve(),'pilot_stdlib_runtime');files[item['path']]=item
    for rel in c.SOURCE_PATHS:
        path=d.ROOT/rel
        c.require(path.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=d.ROOT),'COMMITTED_PILOT_SOURCE_'+rel)
        item=c.seal(path,'pilot_SOURCE:'+rel);files[item['path']]=item
    ex=c.read(excluded_seeds)
    c.require(len(ex['values'])==len(set(ex['values'])) and all(type(v) is int for v in ex['values']),'KNOWN_SEED_EXCLUSIONS')
    # New private copy: never modify the old exclusion registry.
    ep=private/'known_excluded_seeds_PRIVATE.json';c.save(ep,dict(values=ex['values'],
      scope='KNOWN_PRIOR_REGISTRY_CLOSURE_NOT_EVERY_OTHER_RESEARCH_SEED',source_sha256=c.sha(excluded_seeds),
      refresh_required_before_execution=True))
    item=c.seal(ep,'pilot_known_seed_exclusions');files[item['path']]=item
    observed=c.read(d.ROOT/TIMING);timing={r['stage'][:2]:r for r in observed}
    outputs=[]
    for stage in d.STAGES:
        system=stage[:2];dest=private/stage;dest.mkdir(exist_ok=False);root=dest/'future_run01'
        decisions=dict(pilot_design_adopted=False,nominal_variance_precision_limit_accepted=False,
          rare_event_limit_accepted=False,resources_policy_accepted=False,unit_cost_only_accepted=False,
          n_P_by_candidate={r['id']:None for r in d.groups() if r['system']==system},
          formal_shot_policy_adopted=False,AD03='DEFERRED',adaptive_top_up=False,pooling=False,
          statistics_execution_authorized=False)
        approval=dest/'approval_REQUIRED_PRIVATE.json'
        c.save(approval,dict(execution_authorized=False,SOURCE=source,stage=stage,cohort=d.COHORT,
                            decisions=decisions,call_scope=None))
        closure=dict(files);item=c.seal(approval,'pilot_approval_required');closure[item['path']]=item
        # Resource planning arithmetic only. A 2x margin over the saved same-
        # input DET stage observation is a proposed proxy, never an upper bound.
        resources=dict(worker_RSS_proxy_bytes=timing[system]['final_peak_RSS_KiB']*1024*2,
          proxy_source='2X_SAVED_SAME_INPUT_DET_Q_RANGE_STAGE_OBSERVATION_NOT_A_BOUND',
          proxy_multiplier_proposed=2,proxy_policy_adopted=False,
          disk_proxy_bytes=sum(x['bytes'] for x in origin['systems'][system]['inputs'].values()),
          disk_proxy_not_full_output_capacity=True,runtime_caps=False,host_PSI_kill=False,
          live_own_resource_observation=True)
        supervisor={k:str(dest/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
        cfg=dict(SOURCE=source,stage=stage,cohort=d.COHORT,origin_signal_config=str(Path(signal_config).resolve()),
          systems=origin['systems'],runtime=origin['runtime'],provider_root=origin['provider_root'],
          compiler=d.COMPILER,compiler_settings_hash=d.COMPILER_HASH,compile_tolerance=1e-7,other_tolerance=1e-9,
          sealed_files=list(closure.values()),decisions=decisions,tasks=None,call_scope=None,
          campaign_id=None,registry_path=None,excluded_seeds_path=str(ep),resources=resources,authorized=False,
          approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval),run_root=str(root),
          output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),supervisor=supervisor,
          proposal_sha256=c.sha(d.ROOT/PROPOSAL))
        cfg['pipeline_identity']=c.pipeline(cfg)
        plan=dict(status='DRAFT_PILOT_N_AND_EXECUTION_UNAPPROVED',SOURCE=source,stage=stage,cohort=d.COHORT,
          closure_digest=c.digest(cfg['sealed_files']),decisions=decisions,resources=resources,tasks=None,call_scope=None,
          run_root=str(root),supervisor=supervisor,approval_receipt_sha256=c.sha(approval),
          pipeline_identity=cfg['pipeline_identity'],campaign_id=None,registry_path=None,
          excluded_seeds_path=str(ep),proposal_sha256=cfg['proposal_sha256'])
        pp=dest/'plan_DRAFT_PRIVATE.json';c.save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
        cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,SOURCE=source,stage=stage,cohort=d.COHORT,
          plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],call_scope=None,
          retry=False,next_phase_authorized=False,**{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
        c.verify(cfg);c.save(dest/'config_DRAFT_PRIVATE.json',cfg);c.save(dest/'grant_INVALID_PRIVATE.json',cfg['grant'])
        outputs.append(dict(stage=stage,cohort=d.COHORT,plan_sha256=cfg['plan_sha256'],
          closure_digest=plan['closure_digest'],sealed_files=len(cfg['sealed_files']),resources=resources,
          n_P_adopted=None,grant_valid=False,marker_exists=Path(cfg['marker']).exists(),run_root_exists=root.exists()))
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--write-proposal',action='store_true');p.add_argument('--seal-source')
    p.add_argument('--signal-config');p.add_argument('--excluded-seeds');p.add_argument('--private');a=p.parse_args()
    if a.write_proposal:write_proposal();return
    c.require(all((a.seal_source,a.signal_config,a.excluded_seeds,a.private)),'METADATA_SEAL_ARGUMENTS')
    print(json.dumps(seal_drafts(a.seal_source,a.signal_config,a.excluded_seeds,a.private),ensure_ascii=False))

if __name__=='__main__':main()
