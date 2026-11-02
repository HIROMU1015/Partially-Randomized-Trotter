"""Seal disabled confirmation packs. Never issues valid grants or runs science."""
import argparse
import copy
import csv
import json
from pathlib import Path
import statistics
import subprocess
import h246_confirmation_contract as c

RUNTIME_ROLES=('runtime','package_identity','fixed_python','symengine_distribution')

def exclusions(paths):
    values=set();sources=[]
    for path in dict.fromkeys(map(str,paths)):
        data=c.read(path)
        seeds=data['values'] if 'values' in data else [v for r in data.values() for v in r.values()]
        c.require(all(type(v) is int for v in seeds),'INTEGER_KNOWN_SEEDS')
        values.update(seeds);sources.append(c.seal(path,'explicit_known_seed_registry'))
    return sorted(values),sources

def workload(origin,comparison):
    """Technical duration/size proxy; no scientific cost aggregation."""
    root=Path(origin['run_root']);audit=c.read(root/'end_audit.json')
    c.require(audit['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP' and not audit['errors'],'ORIGIN_COMPLETED')
    rows={r['id']:r for r in audit['results']};out=[];files=[]
    for group,n in zip(comparison['groups'],comparison['recommended']['n_C']):
        tasks=[t for t in origin['tasks'] if t['candidate']==group['candidate']]
        c.require(len(tasks)>=2,'MATCHED_EXPLORATION_GROUP_TASKS')
        durations=[];stored=[]
        for task in tasks:
            p=Path(origin['output'])/task['id']/'result.json';r=c.read(p)
            c.require(c.sha(p)==rows[task['id']]['result_sha256'] and r['task']==task and r['SOURCE']==origin['SOURCE'],
              'ORIGIN_TECHNICAL_TIMING_SHA')
            durations.append(r['wall_seconds']);stored.append(rows[task['id']]['stored_bytes'])
            files.append(c.seal(p,'origin_selected_task_metadata'))
        out.append(dict(candidate_id=group['candidate']['id'],n_C=n,saved_n=len(tasks),
          saved_mean_task_seconds=statistics.mean(durations),saved_max_task_seconds=max(durations),
          saved_mean_archived_bytes=statistics.mean(stored),saved_max_archived_bytes=max(stored),
          worker_seconds_proxy=n*statistics.mean(durations),upper_bound=False))
    return out,files

def prepare(source,origins,private):
    source=subprocess.check_output(['git','rev-parse',source+'^{commit}'],cwd=c.ROOT,text=True).strip()
    c.require(len(source)==40,'FULL_COMMIT_ID_REQUIRED')
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=False)
    proposal=c.proposal();c.require(proposal['saved_result_sha256']==c.d.RESULT_SHA,'FROZEN_DESIGN_INPUT')
    origin_cfgs={s:c.read(p) for s,p in origins.items()}
    known,seed_sources=exclusions([x for cfg in origin_cfgs.values() for x in (cfg['excluded_seeds_path'],cfg['registry_path'])])
    campaign='H246_HIGHER_PF_CONFIRMATION_PROPOSED_RUN01_'+source
    registries={s:c.law.registry(c.tasks_for(s+'_CONFIRMATION'),source,campaign,c.d.COHORT,known) for s in origins}
    seeds={s:[v for r in reg.values() for v in r.values()] for s,reg in registries.items()}
    c.require(len(set(v for row in seeds.values() for v in row))==sum(map(len,seeds.values())),'CROSS_STAGE_NEW_SEED_COLLISION')
    cap=c.read(c.ROOT/'artifacts/resource_applicability/track_a_h46_pr_exploration_readiness/2026-10-20/capacity_proposal.json')
    summaries=[]
    for system,origin in origin_cfgs.items():
        stage=system+'_CONFIRMATION';dest=private/stage;dest.mkdir();root=dest/'future_run01'
        comparison=c.comparison(stage);tasks=c.tasks_for(stage)
        closure={r['path']:r for r in origin['sealed_files'] if any(k in r['role'].lower() for k in RUNTIME_ROLES)}
        # Keep only actual source/input/runtime dependencies. Unrelated old QPY,
        # grants and worktree HEAD/status are observations, never freezing gates.
        def add(path,role):
            row=c.seal(Path(path).resolve(),role);closure[row['path']]=row
        for rel,blob in c.source_blobs(source).items():
            c.require((c.ROOT/rel).read_bytes()==blob,'CURRENT_COMMITTED_SOURCE:'+rel);add(c.ROOT/rel,'confirmation_SOURCE:'+rel)
        origin_path=Path(origins[system]).resolve();add(origin_path,'origin_completed_acquisition_config')
        for path in (Path(origin['run_root'])/'end_audit.json',Path(origin['run_root'])/'completion_witness.json',Path(origin['supervisor']['exit'])):
            add(path,'origin_technical_end_evidence')
        if system=='H2':
            for row in origin['inputs'].values():add(row['path'],'saved_H2_input')
            for key in ('signal_path','reference_path'):add(origin[key],'saved_H2_'+key)
        else:
            for row in origin['systems'][system]['inputs'].values():add(row['path'],'saved_'+system+'_input')
            for name in ('reference.npz','receipt.json'):add(Path(origin['systems'][system]['reference_root'])/name,'saved_'+system+'_reference')
            add(c.ROOT/c.SIGNAL_RECORDS,'saved_H46_signal')
        for row in seed_sources:closure[row['path']]=row
        timing,metadata=workload(origin,comparison)
        for row in metadata:closure[row['path']]=row
        rp=dest/'registry_PROPOSED_PRIVATE.json';ep=dest/'excluded_seeds_PRIVATE.json'
        c.save(rp,registries[system]);c.save(ep,dict(values=sorted(set(known+[v for s,row in seeds.items() if s!=system for v in row])),
          sources=seed_sources,known_registry_coverage_only=True,refresh_before_execution=True,pooling=False))
        add(rp,'new_confirmation_seed_registry');add(ep,'known_cohort_exclusions')
        raw=origin['resources'].get('raw_pair_proxy_bytes',cap['raw_pair_proxy_bytes'].get(system))
        final_per=origin['resources'].get('per_trajectory_growth_proxy_bytes') or (4*cap['compressed_pair_proxy_bytes'][system]+16*1024*1024)
        resources=dict(runtime_caps=False,host_PSI_kill=False,process_start_method='fork',prewarm_readonly_contexts=True,
          worker_input_cache=True,worker_RSS_proxy_bytes=origin['resources']['worker_RSS_proxy_bytes'],
          raw_pair_proxy_bytes=raw,final_growth_proxy_bytes=final_per*len(tasks),
          inode_proxy=sum(sum(c.task_limits(t).values())+100 for t in tasks)*2,
          workload_proxy_seconds=sum(r['worker_seconds_proxy'] for r in timing),fresh_joint_admission_required=True,
          proxy_is_upper_bound=False,output_truncation=False)
        storage=c.storage.storage_policy();dec=c.decisions();approval=dest/'approval_REQUIRED_PRIVATE.json'
        c.save(approval,dict(execution_authorized=False,SOURCE=source,stage=stage,cohort=c.d.COHORT,
          comparison=comparison,call_scope=c.law.call_scope(tasks,'PR'),analysis_scope=c.analysis_scope(stage),storage=storage,
          decisions=dec,required_user_action='ADOPT_EXACT_N_C_AND_ONE_SHOT_SCIENTIFIC_SCOPE'))
        add(approval,'future_explicit_user_approval_required')
        cfg={k:copy.deepcopy(origin.get(k)) for k in ('systems','runtime','provider_root','binding','inputs','reference_path',
          'preparation_hash','compiler_phase_policy')}
        cfg.update(SOURCE=source,stage=stage,cohort=c.d.COHORT,tasks=tasks,comparison=comparison,
          call_scope=c.law.call_scope(tasks,'PR'),analysis_scope=c.analysis_scope(stage),campaign_id=campaign,
          registry_path=str(rp),excluded_seeds_path=str(ep),signal_path=origin['signal_path'] if system=='H2' else str(c.ROOT/c.SIGNAL_RECORDS),
          compiler_settings_hash=c.law.COMPILER_HASH,compile_tolerance=1e-7,other_tolerance=1e-9,
          resources=resources,storage=storage,origin_config_path=str(origin_path),decisions=dec,
          proposal_sha256=c.sha(c.ROOT/c.d.PROPOSAL),authorized=False,approval_activation_path=None,approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval),
          run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),
          supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('exit','external_exit_witness_PRIVATE.json'),('log','native_PRIVATE.log')]})
        cfg['sealed_files']=list(closure.values())
        plan=dict(status='DRAFT_EXPLICIT_NEW_SCIENCE_APPROVAL_REQUIRED',**{k:cfg[k] for k in c.PLAN_KEYS},closure_digest=c.digest(cfg['sealed_files']))
        pp=dest/'plan_DRAFT_PRIVATE.json';c.save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
        cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,retry=False,next_phase_authorized=False,
          **{k:cfg[k] for k in ('SOURCE','stage','cohort','plan_sha256','call_scope','analysis_scope','output','ledger','marker','supervisor')})
        c.verify(cfg);cp=dest/'config_DRAFT_PRIVATE.json';c.save(cp,cfg);c.save(dest/'grant_INVALID_PRIVATE.json',cfg['grant'])
        summaries.append(dict(stage=stage,SOURCE=source,n_C=comparison['recommended']['n_C'],
          call_scope=cfg['call_scope'],analysis_scope=cfg['analysis_scope'],plan_sha256=cfg['plan_sha256'],config_sha256=c.sha(cp),
          sealed_files=len(closure),known_excluded_seeds=len(c.read(ep)['values']),proposed_piece_seeds=len(seeds[system]),
          seed_collisions=0,resources=resources,timing_proxy=timing,grant_valid=False,execution_authorized=False,
          scientific_calls_started=0,run_root_exists=root.exists(),marker_exists=Path(cfg['marker']).exists()))
    c.save(private/'preparation_summary_PRIVATE.json',dict(stages=summaries,old_files_modified=0,science_started=False,mandatory_STOP=True))
    return summaries

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--private',required=True)
    for s in ('h2','h4','h6'):p.add_argument('--'+s+'-origin',required=True)
    a=p.parse_args();rows=prepare(a.source,{s:getattr(a,s.lower()+'_origin') for s in ('H2','H4','H6')},a.private)
    print(json.dumps(rows,sort_keys=True))

if __name__=='__main__':main()
