"""Seal the user's H6 restart approval; no molecular sampling/compile here."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import h6_pr_pilot_continuation_contract as c
from h46_pr_pilot_repair import POLICY

def prepare(source,old_path,private):
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=False)
    from audit_h6_pr_pilot_cutoff_repair import audit
    report,part=audit(old_path)
    old=c.read(old_path)
    c.require(report['result_set_sha256']==c.ACCEPTED_RESULT_SET,'SAME74_FRESH_BYTE_AUDIT')
    c.save(private/'saved74_fresh_audit_PRIVATE.json',report)
    accepted=[]
    for tid in part['completed_ids']:
        p=Path(old['output'])/tid/'result.json'
        accepted.append(dict(id=tid,path=str(p),sha256=c.sha(p)))
    oldstates=[Path(old['marker']),Path(old['supervisor']['exit']),
      Path(old['run_root'])/'completion_witness.json']
    failed=next(t for t in old['tasks'] if t['candidate']['formula']=='4th(m5_best)'
      and t['candidate']['q']==2 and t['sample']==0)
    oldstates.append(Path(old['output'])/failed['id']/'STOP.json')
    receipt=dict(status='SEPARATELY_ACCEPTED74_ORIGINAL_STOP_RETAINED',
      old_SOURCE=old['SOURCE'],old_plan_sha256=old['plan_sha256'],old_config_path=str(Path(old_path).resolve()),
      old_config_sha256=c.sha(old_path),result_set_sha256=report['result_set_sha256'],
      fresh_byte_audit_sha256=c.sha(private/'saved74_fresh_audit_PRIVATE.json'),
      accepted_results=accepted,old_state_files=[dict(path=str(p),sha256=c.sha(p)) for p in oldstates],
      original_STOP_preserved=True,old_compile_reservations_retained=172,
      original_registry_sha256=c.sha(old['registry_path']),reuse_authorized=True,
      user_authority='USER_RESTART_APPROVAL_FOLLOWING_74_REUSE_322_CONTINUATION_PROPOSAL',**part)
    ap=private/'acceptance74_APPROVED_PRIVATE.json';c.save(ap,receipt)
    tasks=c.partition(old['tasks'],part['completed_ids'],part['remaining_ids'])
    scope=c.design.call_scope(tasks,'PR')
    c.require(scope['trajectories']==322 and scope['compile_axes']==644
      and scope['request_sampling_calls']==3021,'FIXED_CONTINUATION_CALL_SCOPE')
    root=private/'run01';cfg=copy.deepcopy(old)
    cfg.update(SOURCE=source,continuation_mode=c.MODE,acceptance_path=str(ap),acceptance_sha256=c.sha(ap),
      tasks=tasks,call_scope=scope,run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),
      marker=str(root/'CONSUMED.json'))
    cfg['supervisor']={k:str(private/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),
      ('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    resources=cfg['resources']
    resources.update(optimization_policy=POLICY,prewarm_readonly_contexts=True,process_start_method='fork',
      worker_input_cache=True,worker_allocation_policy='FRESH_SINGLE_STAGE_CPU_RAM_V1')
    # Keep the same 4x compressed +16 MiB/trajectory planning policy. Reserve
    # raw/partial space for all workers that current CPU/RAM can admit.
    from run_h46_m5_signal import fresh_resources
    from h46_pr_pilot_repair import worker_count
    probe=dict(cfg,systems={'H4':{},'H6':{}},resources=dict(
      system_RSS_proxy_bytes={'H4':1,'H6':1},disk_proxy_bytes=0))
    observation=fresh_resources(probe)
    workers=worker_count(len(tasks),observation['idle_CPU_equivalents'],
      observation['effective_available_RAM_bytes'],resources['worker_RSS_proxy_bytes'])
    resources['admission_workers']=workers
    resources['disk_proxy_bytes']=(resources['storage_bytes_per_trajectory_proxy']*len(tasks)
      +(workers+1)*2026673924)
    resources['inode_proxy']=sum(sum(c.task_limits(t).values())+100 for t in tasks)*2
    c.require(observation['disk_free_bytes']>=resources['disk_proxy_bytes'],'CONTINUATION_PEAK_DISK_ADMISSION')
    c.save(private/'capacity_PRIVATE.json',dict(observation=observation,workers=workers,
      peak_proxy_bytes=resources['disk_proxy_bytes'],proxy_not_output_upper_bound=True))
    resources['joint_capacity_sha256']=c.sha(private/'capacity_PRIVATE.json')
    approval=private/'approval_APPROVED_PRIVATE.json'
    c.save(approval,dict(execution_authorized=True,SOURCE=source,stage=cfg['stage'],cohort=cfg['cohort'],
      decisions=cfg['decisions'],call_scope=scope,storage=cfg['storage'],n_P_each=33,
      acceptance_sha256=c.sha(ap),reuse74_authorized=True,continue322_authorized=True,
      fixed_original_registry=True,user_authority='USER_2026_10_20_RESTART_APPROVAL',
      retry=False,next_phase_authorized=False,statistics_execution_authorized=False))
    cfg.update(approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval))
    files={r['path']:r for r in old['sealed_files']}
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel
        c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),
          'COMMITTED_CONTINUATION_SOURCE:'+rel)
        item=c.seal(p,'continuation_SOURCE:'+rel);files[item['path']]=item
    for path,role in [(Path(old_path).resolve(),'original_stopped_config'),(ap,'independent74_acceptance'),
      (approval,'new_explicit_continuation_approval'),(private/'capacity_PRIVATE.json','fresh_capacity'),
      (private/'saved74_fresh_audit_PRIVATE.json','fresh74_byte_audit')]:
        item=c.seal(path,role);files[item['path']]=item
    cfg['sealed_files']=list(files.values());cfg['pipeline_identity']=c.pipeline(cfg)
    plan={key:cfg[key] for key in ('SOURCE','stage','cohort','decisions','resources','storage','tasks','call_scope',
      'run_root','supervisor','approval_receipt_sha256','pipeline_identity','campaign_id','registry_path',
      'excluded_seeds_path','proposal_sha256','preservation_old_DET_root','continuation_mode','acceptance_sha256')}
    plan.update(status='APPROVED_SEPARATE_ACCEPT74_CONTINUE322_ONE_SHOT',closure_digest=c.digest(cfg['sealed_files']))
    pp=private/'plan_APPROVED_PRIVATE.json';c.save(pp,plan)
    cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
    cfg['grant']=dict(valid=True,grant_id='H6_CONTINUE322_'+c.digest(plan),SOURCE=source,
      stage=cfg['stage'],cohort=cfg['cohort'],plan_sha256=cfg['plan_sha256'],
      pipeline_identity=cfg['pipeline_identity'],call_scope=scope,
      storage_policy_sha256=c.digest(cfg['storage']),acceptance_sha256=c.sha(ap),
      approval_reference=str(approval)+'#sha256='+c.sha(approval),retry=False,next_phase_authorized=False,
      **{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
    c.authorize(cfg)
    cp=private/'config_APPROVED_PRIVATE.json';c.save(cp,cfg)
    c.save(private/'grant_APPROVED_PRIVATE.json',cfg['grant'])
    c.save(private/'preflight_PRIVATE.json',dict(status='PASS',SOURCE=source,
      plan_sha256=cfg['plan_sha256'],accepted_pairs=74,pending_pairs=322,compile_axes=644,
      original_registry_preserved=True,old_STOP_preserved=True,workers_admission=workers,
      resource_workers_reprobe_after_context=True,new_scientific_calls=0))
    return dict(status='PREPARED',SOURCE=source,config=str(cp),accepted_pairs=74,pending_pairs=322,
      compile_axes=644,workers_admission=workers,peak_disk_proxy_bytes=resources['disk_proxy_bytes'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--old-config',required=True)
    p.add_argument('--private',required=True);a=p.parse_args()
    print(__import__('json').dumps(prepare(a.source,a.old_config,a.private)))
if __name__=='__main__':main()
