"""Seal the user's explicit 33/group acquisition approval, never start science.

Private authorization only. Historical disabled drafts and old raw stay intact.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import h46_pr_pilot_contract as c
import h46_pr_design_pilot as d

BASE='3d0b5477ce04604ce873c079d9c66d02f2782239'
TIMING='artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/run_timing.json'
RAW={'H4':438570748,'H6':2026673924}
COMPRESSED={'H4':13343802,'H6':61925972}

def prepare(source,signal_config,excluded_seeds,private):
    import run_h46_pr_design_pilot as launcher
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=False)
    origin=c.original.original_signal(signal_config)
    c.require(origin['SOURCE']==c.manifest()['signal_SOURCE'],'FIXED_COMPLETED_SIGNAL')
    c.require(sys.executable==origin['runtime']['python'],'FIXED_PREPARATION_PYTHON')
    files={r['path']:r for r in origin['sealed_files']}
    for path,role in [(signal_config,'pilot_original_signal_config')]:
        item=c.seal(Path(path).resolve(),role);files[item['path']]=item
    import concurrent.futures.process,concurrent.futures.thread
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).resolve().is_relative_to('/usr/lib/python3.12'):
            item=c.seal(Path(p).resolve(),'pilot_stdlib_runtime');files[item['path']]=item
    for rel in c.SOURCE_PATHS:
        path=d.ROOT/rel
        c.require(path.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=d.ROOT),'COMMITTED_PILOT_SOURCE:'+rel)
        item=c.seal(path,'pilot_SOURCE:'+rel);files[item['path']]=item
    old=c.read(excluded_seeds)['values']
    c.require(len(old)==len(set(old)) and all(type(v) is int for v in old),'KNOWN_EXCLUDED_SEEDS')
    timing={r['stage'][:2]:r for r in c.read(d.ROOT/TIMING)}
    rss={s:r['final_peak_RSS_KiB']*1024*2 for s,r in timing.items()}
    tasks={s:d.tasks_for(s+'_DESIGN_PILOT',{r['id']:33 for r in d.groups() if r['system']==s}) for s in timing}
    from run_h46_m5_signal import fresh_resources
    probe=dict(run_root=str(private/'unused_probe'),systems={'H4':{},'H6':{}},
      resources=dict(system_RSS_proxy_bytes={'H4':1,'H6':1},disk_proxy_bytes=0))
    observation=fresh_resources(probe)
    demand={s:len(tasks[s])*r['launcher_wall_seconds']*len(r['workers_reaped'])/
      (19 if s=='H4' else 27) for s,r in timing.items()}
    # Demand proxy uses prior DET elapsed worker-seconds/trajectory, not new
    # pilot outcomes. Allocation changes no sampling law or sample count.
    workers=c.storage.allocation(observation['idle_CPU_equivalents'],observation['effective_available_RAM_bytes'],
      {s:len(t) for s,t in tasks.items()},rss,demand)
    final={s:len(tasks[s])*(4*COMPRESSED[s]+16*1024*1024) for s in tasks}
    scratch={s:workers[s]*RAW[s] for s in tasks}
    restoration=max(RAW.values());peak=sum(final.values())+sum(scratch.values())+restoration
    c.require(observation['disk_free_bytes']>=peak,'JOINT_PEAK_DISK_ADMISSION')
    capacity=dict(SOURCE=source,base_commit=BASE,observation=observation,workers=workers,
      worker_RSS_proxy_bytes=rss,prior_timing_demand_proxy=demand,final_growth_proxy_bytes=final,
      worker_raw_partial_quarantine_proxy_bytes=scratch,restoration_reserve_bytes=restoration,
      joint_peak_growth_proxy_bytes=peak,proxy_is_upper_bound=False,
      existing_results_deleted=0,output_truncation=False,shared_environment_changes=False)
    cp=private/'joint_capacity_PRIVATE.json';c.save(cp,capacity)
    item=c.seal(cp,'pilot_joint_resource_admission');files[item['path']]=item
    campaign='H46_OUTER32_DESIGN_PILOT_RUN01_'+source
    regs={s:d.registry(tasks[s],source,campaign,d.COHORT,old) for s in tasks}
    seeds={s:[v for row in r.values() for v in row.values()] for s,r in regs.items()}
    c.require(not set(seeds['H4'])&set(seeds['H6']),'CROSS_SYSTEM_SEED_COLLISION')
    output=[]
    for system in ('H4','H6'):
        stage=system+'_DESIGN_PILOT';dest=private/stage;dest.mkdir();root=dest/'run01'
        rp=dest/'registry_PRIVATE.json';ep=dest/'excluded_seeds_PRIVATE.json'
        c.save(ep,dict(values=old+seeds['H6' if system=='H4' else 'H4'],
          source_sha256=c.sha(excluded_seeds),scope='KNOWN_PRIOR_4603_PLUS_OTHER_SYSTEM_NEW_PILOT',
          finite_known_registry_coverage=True,not_all_seeds_of_all_research=True))
        c.save(rp,regs[system])
        closure=dict(files)
        for path,role in [(rp,'pilot_fresh_registry'),(ep,'pilot_excluded_seeds')]:
            item=c.seal(path,role);closure[item['path']]=item
        decisions=dict(pilot_design_adopted=True,nominal_variance_precision_limit_accepted=True,
          rare_event_limit_accepted=True,resources_policy_accepted=True,unit_cost_only_accepted=True,
          n_P_by_candidate={r['id']:33 for r in d.groups() if r['system']==system},
          formal_shot_policy_adopted=False,AD03='DEFERRED',adaptive_top_up=False,pooling=False,
          statistics_execution_authorized=False)
        scope=d.call_scope(tasks[system],'PR');storage=c.storage.storage_policy()
        resources=dict(worker_RSS_proxy_bytes=rss[system],proxy_source='2X_SAVED_SAME_INPUT_DET_OBSERVATION_NOT_BOUND',
          proxy_policy_adopted=True,proxy_multiplier_proposed=2,admission_workers=workers[system],
          joint_capacity_sha256=c.sha(cp),disk_proxy_bytes=peak,
          disk_proxy_not_full_output_capacity=False,storage_bytes_per_trajectory_proxy=final[system]//len(tasks[system]),
          full_output_capacity_adopted=True,runtime_caps=False,host_PSI_kill=False,live_own_resource_observation=True,
          inode_proxy=sum(sum(c.task_limits(t).values())+100 for t in tasks[system])*2)
        approval=dest/'approval_APPROVED_PRIVATE.json'
        c.save(approval,dict(execution_authorized=True,SOURCE=source,stage=stage,cohort=d.COHORT,
          decisions=decisions,call_scope=scope,storage=storage,n_P_each=33,
          user_authority='USER_EXPLICIT_H4_H6_PILOT_33_792_TRAJECTORIES_1584_AXES_EXECUTION_APPROVAL',
          no_reapproval_after_preparation=True,retry=False,statistics_execution_authorized=False))
        item=c.seal(approval,'pilot_explicit_execution_approval');closure[item['path']]=item
        supervisor={k:str(dest/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),
          ('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
        cfg=dict(SOURCE=source,stage=stage,cohort=d.COHORT,origin_signal_config=str(Path(signal_config).resolve()),
          systems=origin['systems'],runtime=origin['runtime'],provider_root=origin['provider_root'],
          compiler=d.COMPILER,compiler_settings_hash=d.COMPILER_HASH,compile_tolerance=1e-7,other_tolerance=1e-9,
          sealed_files=list(closure.values()),decisions=decisions,tasks=tasks[system],call_scope=scope,
          campaign_id=campaign,registry_path=str(rp),excluded_seeds_path=str(ep),resources=resources,storage=storage,
          authorized=True,approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval),run_root=str(root),
          output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),supervisor=supervisor,
          proposal_sha256=c.sha(d.ROOT/'references/h46_pr_design_pilot_proposal_v1.json'))
        cfg['preservation_old_DET_root']=str(Path(excluded_seeds).resolve().parent/'authorization_DET_run01')
        cfg['pipeline_identity']=c.pipeline(cfg)
        plan=dict(status='APPROVED_ONE_SHOT_ACQUISITION_ONLY',SOURCE=source,stage=stage,cohort=d.COHORT,
          closure_digest=c.digest(cfg['sealed_files']),decisions=decisions,resources=resources,storage=storage,
          tasks=cfg['tasks'],call_scope=scope,run_root=str(root),supervisor=supervisor,
          approval_receipt_sha256=c.sha(approval),pipeline_identity=cfg['pipeline_identity'],campaign_id=campaign,
          registry_path=str(rp),excluded_seeds_path=str(ep),proposal_sha256=cfg['proposal_sha256'])
        plan['preservation_old_DET_root']=cfg['preservation_old_DET_root']
        pp=dest/'plan_APPROVED_PRIVATE.json';c.save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
        cfg['grant']=dict(valid=True,grant_id=stage+'_RUN01_'+c.digest(plan),
          approval_reference=str(approval)+'#sha256='+c.sha(approval),SOURCE=source,stage=stage,cohort=d.COHORT,
          plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],call_scope=scope,
          storage_policy_sha256=c.digest(storage),retry=False,next_phase_authorized=False,
          **{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
        c.authorize(cfg)
        config=dest/'config_APPROVED_PRIVATE.json';c.save(config,cfg)
        c.save(dest/'grant_APPROVED_PRIVATE.json',cfg['grant'])
        c.save(dest/'preflight_PRIVATE.json',dict(status='PASS',SOURCE=source,plan_sha256=cfg['plan_sha256'],
          scientific_calls_started=0,seed_count=len(seeds[system]),seed_collisions=0,sealed_files=len(closure),
          workers=workers[system],call_scope=scope,storage=storage))
        output.append(dict(stage=stage,config=str(config),plan_sha256=cfg['plan_sha256'],
          workers=workers[system],seed_count=len(seeds[system]),call_scope=scope))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--signal-config',required=True)
    p.add_argument('--excluded-seeds',required=True);p.add_argument('--private',required=True);a=p.parse_args()
    print(__import__('json').dumps(prepare(a.source,a.signal_config,a.excluded_seeds,a.private)))

if __name__=='__main__':main()
