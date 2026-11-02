"""Seal explicitly approved exploration packs, retaining the fixed disk plan.

No sampling, compile, signal, statistical evaluation, shot or ranking. This
command never launches; the separate controller consumes each one-shot grant.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import sys
import h46_pr_exploration_contract as c
import h46_pr_exploration_design as d
import prepare_h46_pr_exploration as draft

CAPACITY='artifacts/resource_applicability/track_a_h46_pr_exploration_readiness/2026-10-20/capacity_proposal.json'
CAPACITY_COMMIT='2fcdec15341c0b3371b37ab5d5eb5bec1009de54'

def allocate_existing_plan(observation,capacity,tasks,demand):
    """Fresh CPU/RAM allocation within the unchanged adopted scratch reserve."""
    peak=capacity['concurrent_peak_growth_proxy_bytes']
    c.require(observation['disk_free_bytes']>=peak,'FIXED_CAPACITY_PLAN_DISK_GATE')
    reserve=peak-sum(capacity['final_growth_proxy_bytes'].values())-capacity['restoration_reserve_bytes']
    cpu=math.floor(observation['idle_CPU_equivalents']);ram=observation['effective_available_RAM_bytes']
    rss=capacity['worker_RSS_proxy_bytes'];raw=capacity['raw_pair_proxy_bytes'];best=None
    for a in range(1,min(tasks['H4'],cpu-1,ram//rss['H4'],reserve//(2*raw['H4']))+1):
        b=min(tasks['H6'],cpu-a,(ram-a*rss['H4'])//rss['H6'],(reserve-2*a*raw['H4'])//(2*raw['H6']))
        if b<1:continue
        score=(max(demand['H4']/a,demand['H6']/b),-(a+b))
        if best is None or score<best[0]:best=(score,dict(H4=a,H6=b))
    c.require(best is not None,'FIXED_PLAN_CANNOT_ADMIT_BOTH_STAGES')
    return best[1]

def approve(cfg,dest,lease_path,capacity,det_root):
    """New immutable approved files. Never mutate an existing draft/grant."""
    cfg=copy.deepcopy(cfg);dest=Path(dest).resolve();dest.mkdir(exist_ok=False)
    system=cfg['stage'][:2];lease=c.read(lease_path)
    for key in ('sample_design_adopted','finite_sample_limit_accepted','unknown_rare_tail_accepted',
                'resources_storage_policy_accepted','unit_cost_acquisition_only_accepted'):
        cfg['decisions'][key]=True
    root=dest/'run01';cfg.update(authorized=True,run_root=str(root),output=str(root/'output'),
        ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'))
    cfg['supervisor']={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),
        ('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    cfg['resources'].update(proxy_policy_adopted=True,full_output_capacity_adopted=True,
        worker_RSS_proxy_bytes=capacity['worker_RSS_proxy_bytes'][system],
        joint_admission_path=str(lease_path),joint_admission_sha256=c.sha(lease_path),
        disk_proxy_bytes=capacity['concurrent_peak_growth_proxy_bytes'],
        inode_proxy=sum(sum(c.task_limits(t).values())+100 for t in cfg['tasks'])*2)
    cfg['preservation_old_DET_root']=str(Path(det_root).resolve())
    approval=dest/'approval_APPROVED_PRIVATE.json'
    c.save(approval,dict(execution_authorized=True,**{k:cfg[k] for k in
        ('SOURCE','stage','cohort','decisions','call_scope','storage')},
        user_instruction='その方針で進めてよいが、できるだけ並列化高速化簡単なものでいいがしてから行って',
        adopted_option=d.RECOMMENDED,interpretation='ADOPT_RECOMMENDED_DESIGN_AND_H4_H6_ONE_SHOT_UNIT_COST_ACQUISITION',
        statistics_execution_authorized=False,retry=False))
    old_approval=cfg['approval_receipt_path']
    cfg.update(approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval))
    closure={r['path']:r for r in cfg['sealed_files'] if r['path']!=old_approval}
    for path,role in [(approval,'exploration_explicit_user_approval'),(lease_path,'exploration_joint_admission'),
                      (c.ROOT/CAPACITY,'unchanged_adopted_capacity_plan')]:
        item=c.seal(path,role);closure[item['path']]=item
    cfg['sealed_files']=list(closure.values());cfg['pipeline_identity']=c.pipeline(cfg)
    plan=dict(status='APPROVED_ONE_SHOT_EXPLORATION_UNIT_COST_ACQUISITION_ONLY',**{k:cfg[k] for k in
        ('SOURCE','stage','cohort','decisions','tasks','call_scope','resources','storage','campaign_id',
         'registry_path','excluded_seeds_path','approval_receipt_sha256','run_root','supervisor',
         'proposal_sha256','pipeline_identity','preservation_old_DET_root')},closure_digest=c.digest(cfg['sealed_files']))
    pp=dest/'plan_APPROVED_PRIVATE.json';c.save(pp,plan)
    cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
    cfg['grant']=dict(valid=True,grant_id=cfg['stage']+'_RUN01_'+c.digest(plan),retry=False,next_phase_authorized=False,
        approval_reference=str(approval)+'#sha256='+c.sha(approval),storage_policy_sha256=c.digest(cfg['storage']),
        **{k:cfg[k] for k in ('SOURCE','stage','cohort','plan_sha256','pipeline_identity','call_scope',
                            'output','ledger','marker','supervisor')})
    c.authorize(cfg)
    path=dest/'config_APPROVED_PRIVATE.json';c.save(path,cfg);c.save(dest/'grant_APPROVED_PRIVATE.json',cfg['grant'])
    c.save(dest/'preflight_PRIVATE.json',dict(status='PASS',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        workers=lease['workers'][system],call_scope=cfg['call_scope'],sealed_files=len(closure),
        config_sha256=c.sha(path),seed_collisions=0,scientific_calls_started=0,retry=False))
    return dict(stage=cfg['stage'],config=str(path),plan_sha256=cfg['plan_sha256'],
                workers=lease['workers'][system],call_scope=cfg['call_scope'])

def prepare(source,signal_config,exclusions,private,det_root):
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=False)
    # Fresh registries also exclude the prior unused proposal registries when
    # explicitly supplied. No directory-wide registry discovery or rehash retry.
    draft.prepare(source,signal_config,exclusions,private/'new_disabled_packs')
    configs=[c.read(private/'new_disabled_packs'/stage/'config_DRAFT_PRIVATE.json') for stage in d.STAGES]
    import subprocess
    capacity=c.read(c.ROOT/CAPACITY)
    c.require((c.ROOT/CAPACITY).read_bytes()==subprocess.check_output(
        ['git','show',CAPACITY_COMMIT+':'+CAPACITY],cwd=c.ROOT),'UNCHANGED_CAPACITY_PLAN')
    from run_h46_m5_signal import fresh_resources
    observation=fresh_resources(dict(run_root=str(private/'unused_probe'),systems={'H4':{},'H6':{}},
        resources=dict(system_RSS_proxy_bytes={'H4':1,'H6':1},disk_proxy_bytes=capacity['concurrent_peak_growth_proxy_bytes'])))
    from prepare_h46_pr_pilot_execution import TIMING
    saved={r['stage'][:2]:r for r in c.read(c.ROOT/TIMING)}
    counts={cfg['stage'][:2]:len(cfg['tasks']) for cfg in configs}
    demand={s:counts[s]*r['launcher_wall_seconds']*len(r['workers_reaped'])/(19 if s=='H4' else 27)
            for s,r in saved.items()}
    workers=allocate_existing_plan(observation,capacity,counts,demand)
    inodes=sum(sum(sum(c.task_limits(t).values())+100 for t in cfg['tasks'])*2 for cfg in configs)
    c.require(os.statvfs(private).f_favail>=inodes,'JOINT_INODE_ADMISSION')
    lease=private/'joint_admission_APPROVED_PRIVATE.json'
    c.save(lease,dict(SOURCE=source,adopted=True,runtime_caps=False,workers=workers,observation=observation,
        peak_growth_proxy_bytes=capacity['concurrent_peak_growth_proxy_bytes'],
        fixed_capacity_plan_sha256=c.sha(c.ROOT/CAPACITY),capacity_reestimated=False,
        host_PSI_kill=False,shared_environment_changes=False,old_files_modified=0,prior_timing_demand_proxy=demand))
    rows=[approve(cfg,private/cfg['stage'],lease,capacity,det_root) for cfg in configs]
    c.save(private/'preparation_summary_PRIVATE.json',dict(stages=rows,SOURCE=source,statistics=False,scientific_calls=0))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--signal-config',required=True)
    p.add_argument('--exclude',action='append',required=True);p.add_argument('--private',required=True)
    p.add_argument('--det-root',required=True);a=p.parse_args()
    print(__import__('json').dumps(prepare(a.source,a.signal_config,a.exclude,a.private,a.det_root)))

if __name__=='__main__':main()
