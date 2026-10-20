"""Build sealed *disabled* exploration packs. Cannot issue an executable grant."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import h46_pr_exploration_design as d
import h46_pr_exploration_contract as c

def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def design_outputs(destination):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    proposal=d.proposal();c.save(d.ROOT/d.PROPOSAL,proposal)
    c.save(dest/'sample_design_proposals.json',proposal)
    summary=[];counts=[]
    for option in proposal['options']:
        summary.append({k:option[k] for k in ('id','relative_MCSE_target','variance_multiplier','allocation',
           'minimum_n_proposed','trajectories','compile_axes','adopted','population_guarantee')})
        for key in ('request_sampling_calls','rte_events','native_evolution','compiled_wrapper_actions'):
            summary[-1][key]=sum(b[key] for b in option['system_call_scope_proposed'].values())
        for group in proposal['groups']:
            counts.append(dict(option_id=option['id'],system=group['system'],formula=group['formula'],q=group['q'],
              candidate_id=group['id'],n_E_proposed=option['n_E_by_candidate_proposed'][group['id']],n_E_adopted=None))
    write_csv(dest/'option_budgets_36.csv',summary);write_csv(dest/'group_allocation_864.csv',counts)
    chosen=d.selected_option(proposal,d.RECOMMENDED)
    rec=[]
    stats={g['candidate']['id']:g['statistics'] for s in ('H4','H6') for g in
           json.loads(d.frozen_bytes(d.EVIDENCE+s+'_pilot_statistics.json'))['groups']}
    for group in proposal['groups']:
        s=stats[group['id']];scenario=next(r for r in s['n_E_sensitivity'] if r['relative_MCSE_target']==.02 and r['variance_inflation']==4)
        rec.append(dict(system=group['system'],formula=group['formula'],q=group['q'],candidate_id=group['id'],
          pilot_n=s['n'],saved_covariance_cs=s['covariance'][0][1],saved_pair_MCSE=s['pair_unit_RZ_MCSE'],
          saved_n_E_rho2pct_f4=scenario['n_E_nominal_only'],n_E_proposed=chosen['n_E_by_candidate_proposed'][group['id']],
          n_E_adopted=None,S0_observed=s['strata']['S0']['observed_count'],
          S2_single_observed=s['strata']['S2_single']['observed_count'],S2_multi_observed=s['strata']['S2_multi']['observed_count'],
          S4_observed=s['strata']['S4']['observed_count'],S4_population_cost_bound=None,shot_weighted=False))
    write_csv(dest/'recommended_groups_24.csv',rec)
    c.save(dest/'recommended_call_scope.json',chosen['system_call_scope_proposed'])
    return proposal

def capacity_preview(private):
    """Technical planning only, using prior fixed DET storage/RAM proxies.

    No constant worker cap. This observation/lease preview is not an execution
    gate or adopted lease; refresh and seal only after user adoption.
    """
    from prepare_h46_pr_pilot_execution import RAW,COMPRESSED,TIMING
    from run_h46_m5_signal import fresh_resources
    private=Path(private).resolve()
    chosen=d.selected_option(c.read(d.ROOT/d.PROPOSAL),d.RECOMMENDED)
    timing={row['stage'][:2]:row for row in c.read(d.ROOT/TIMING)}
    rss={s:row['final_peak_RSS_KiB']*1024*2 for s,row in timing.items()}
    counts={s:b['trajectories'] for s,b in chosen['system_call_scope_proposed'].items()}
    probe=dict(run_root=str(private/'unused_probe'),systems={'H4':{},'H6':{}},
      resources=dict(system_RSS_proxy_bytes={'H4':1,'H6':1},disk_proxy_bytes=0))
    observation=fresh_resources(probe)
    demand={s:counts[s]*row['launcher_wall_seconds']*len(row['workers_reaped'])/(19 if s=='H4' else 27)
            for s,row in timing.items()}
    workers=c.storage.allocation(observation['idle_CPU_equivalents'],observation['effective_available_RAM_bytes'],counts,rss,demand)
    final={s:counts[s]*(4*COMPRESSED[s]+16*1024*1024) for s in counts}
    temporary={s:workers[s]*RAW[s] for s in counts}
    # Two full raw-pair buffers per worker cover raw + compressed partial even
    # if compression expands an incompressible stream; restoration separately.
    concurrent=sum(final.values())+2*sum(temporary.values())+max(RAW.values())
    sequential=max(sum(final.values())+2*workers[s]*RAW[s]+RAW[s] for s in counts)
    return dict(status='UNADOPTED_PLANNING_OBSERVATION_NOT_FRESH_EXECUTION_GATE',
      option_id=d.RECOMMENDED,observation=observation,workers_proposed=workers,
      worker_RSS_proxy_bytes=rss,worker_proxy_source='2X_SAVED_SAME_INPUT_DET_PEAK_NOT_BOUND',
      final_growth_proxy_bytes=final,raw_pair_proxy_bytes=RAW,compressed_pair_proxy_bytes=COMPRESSED,
      per_worker_raw_plus_partial_buffers=2,restoration_reserve_bytes=max(RAW.values()),
      concurrent_peak_growth_proxy_bytes=concurrent,sequential_peak_using_same_worker_shares_proxy_bytes=sequential,
      current_disk_sufficient_for_proxy=observation['disk_free_bytes']>=concurrent,
      proxy_is_upper_bound=False,resources_adopted=False,runtime_caps=False,
      sequential_order_does_not_reduce_final_storage=True,
      existing_results_deleted=0,output_truncation=False,shared_environment_changes=False)

def collect_exclusions(paths):
    """Read only explicitly supplied prior registries. No discovery traversal."""
    values=set();sources=[]
    for path in paths:
        path=Path(path).resolve();data=c.read(path)
        if 'values' in data:seeds=data['values']
        else:
            seeds=[]
            for row in data.values():seeds.extend(row.values() if isinstance(row,dict) else row)
        c.require(all(type(x) is int for x in seeds),'EXPLICIT_INTEGER_SEED_REGISTRY')
        values.update(seeds);sources.append(dict(path=str(path),bytes=path.stat().st_size,sha256=c.sha(path)))
    return sorted(values),sources

def prepare(source,signal_config,exclusion_paths,private):
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=False)
    proposal=c.read(d.ROOT/d.PROPOSAL);c.require(proposal==d.proposal(),'FROZEN_PROPOSAL')
    origin=c.original.original_signal(signal_config)
    c.require(sys.executable==origin['runtime']['python'],'FIXED_PYTHON_FOR_PACK')
    from h46_pr_exploration_runtime import get_runtime
    get_runtime()  # load/seal reused execution modules, not molecular inputs
    files={r['path']:r for r in origin['sealed_files']}
    for path,role in [(Path(signal_config).resolve(),'exploration_original_signal_config')]:
        item=c.seal(path,role);files[item['path']]=item
    import concurrent.futures.process,concurrent.futures.thread
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).resolve().is_relative_to('/usr/lib/python3.12'):
            item=c.seal(Path(path).resolve(),'exploration_stdlib_runtime');files[item['path']]=item
    for rel in c.SOURCE_PATHS:
        path=d.ROOT/rel
        c.require(path.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=d.ROOT),'COMMITTED_EXPLORATION_SOURCE:'+rel)
        item=c.seal(path,'exploration_SOURCE:'+rel);files[item['path']]=item
    known,sources=collect_exclusions(exclusion_paths)
    chosen=d.selected_option(proposal,d.RECOMMENDED)
    tasks={s:d.tasks_for(s+'_PR_EXPLORATION',{r['id']:chosen['n_E_by_candidate_proposed'][r['id']]
              for r in d.groups() if r['system']==s}) for s in ('H4','H6')}
    campaign='H46_MAIN_EXPLORATION_PROPOSED_RUN01_'+source
    regs={s:d.registry(tasks[s],source,campaign,d.COHORT,known) for s in tasks}
    seeds={s:[v for row in registry.values() for v in row.values()] for s,registry in regs.items()}
    c.require(not set(seeds['H4'])&set(seeds['H6']),'CROSS_SYSTEM_SEED_COLLISION')
    output=[]
    for system in ('H4','H6'):
        stage=system+'_PR_EXPLORATION';dest=private/stage;dest.mkdir();root=dest/'future_run01'
        rp=dest/'registry_PROPOSED_PRIVATE.json';ep=dest/'excluded_seeds_PRIVATE.json'
        c.save(rp,regs[system]);c.save(ep,dict(values=known+seeds['H6' if system=='H4' else 'H4'],
          finite_known_registry_coverage=True,not_all_research_seeds=True,refresh_before_execution=True,sources=sources))
        closure=dict(files)
        for path,role in [(rp,'exploration_fresh_proposed_seeds'),(ep,'exploration_known_exclusions')]:
            item=c.seal(path,role);closure[item['path']]=item
        decisions=dict(option_id=d.RECOMMENDED,n_E_by_candidate={r['id']:chosen['n_E_by_candidate_proposed'][r['id']]
          for r in d.groups() if r['system']==system},sample_design_adopted=False,finite_sample_limit_accepted=False,
          unknown_rare_tail_accepted=False,resources_storage_policy_accepted=False,unit_cost_acquisition_only_accepted=False,
          formal_shot_policy_adopted=False,AD03='DEFERRED',pooling=False,adaptive_top_up=False,statistics_execution_authorized=False)
        resources=dict(process_start_method='fork',prewarm_readonly_contexts=True,worker_input_cache=True,
          worker_allocation_policy='FRESH_JOINT_CPU_RAM_V1',runtime_caps=False,host_PSI_kill=False,
          proxy_policy_adopted=False,full_output_capacity_adopted=False,
          worker_RSS_proxy_bytes=None,joint_admission_path=None,joint_admission_sha256=None,
          disk_proxy_bytes=None,inode_proxy=None,fresh_observation_required=True)
        scope=d.call_scope(tasks[system],'PR');storage=c.storage.storage_policy()
        approval=dest/'approval_REQUIRED_PRIVATE.json'
        c.save(approval,dict(execution_authorized=False,SOURCE=source,stage=stage,cohort=d.COHORT,
          decisions=decisions,call_scope=scope,storage=storage))
        item=c.seal(approval,'exploration_approval_required');closure[item['path']]=item
        supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
        cfg=dict(SOURCE=source,stage=stage,cohort=d.COHORT,origin_signal_config=str(Path(signal_config).resolve()),
          systems=origin['systems'],runtime=origin['runtime'],provider_root=origin['provider_root'],
          compiler=d.COMPILER,compiler_settings_hash=d.COMPILER_HASH,compile_tolerance=1e-7,other_tolerance=1e-9,
          sealed_files=list(closure.values()),decisions=decisions,tasks=tasks[system],call_scope=scope,
          campaign_id=campaign,registry_path=str(rp),excluded_seeds_path=str(ep),resources=resources,storage=storage,
          authorized=False,approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval),
          run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),
          supervisor=supervisor,proposal_sha256=c.sha(d.ROOT/d.PROPOSAL))
        cfg['pipeline_identity']=c.pipeline(cfg)
        plan=dict(status='DRAFT_NO_DESIGN_OR_EXECUTION_ADOPTION',**{k:cfg[k] for k in (
          'SOURCE','stage','cohort','decisions','tasks','call_scope','resources','storage','campaign_id',
          'registry_path','excluded_seeds_path','approval_receipt_sha256','run_root','supervisor','proposal_sha256','pipeline_identity')},
          closure_digest=c.digest(cfg['sealed_files']))
        pp=dest/'plan_DRAFT_PRIVATE.json';c.save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
        cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,storage_policy_sha256=c.digest(storage),
          retry=False,next_phase_authorized=False,**{k:cfg[k] for k in (
          'SOURCE','stage','cohort','plan_sha256','pipeline_identity','call_scope','output','ledger','marker','supervisor')})
        c.verify(cfg);config=dest/'config_DRAFT_PRIVATE.json';c.save(config,cfg)
        c.save(dest/'grant_INVALID_PRIVATE.json',cfg['grant'])
        output.append(dict(stage=stage,cohort=d.COHORT,plan_sha256=cfg['plan_sha256'],
          closure_digest=plan['closure_digest'],sealed_files=len(cfg['sealed_files']),
          config_sha256=c.sha(config),known_excluded_seeds=len(known),proposed_seeds=len(seeds[system]),
          seed_collisions=0,call_scope=scope,grant_valid=False,design_adopted=False,
          run_root_exists=root.exists(),marker_exists=Path(cfg['marker']).exists()))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--design-output');p.add_argument('--source');p.add_argument('--signal-config')
    p.add_argument('--exclude',action='append',default=[]);p.add_argument('--private');a=p.parse_args()
    if a.design_output:design_outputs(a.design_output);return
    c.require(all((a.source,a.signal_config,a.exclude,a.private)),'DRAFT_SEAL_ARGUMENTS')
    print(json.dumps(prepare(a.source,a.signal_config,a.exclude,a.private),sort_keys=True))

if __name__=='__main__':main()
