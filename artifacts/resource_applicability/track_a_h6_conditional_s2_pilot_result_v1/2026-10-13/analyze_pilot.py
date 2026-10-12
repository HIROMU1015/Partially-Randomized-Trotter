"""Saved metadata-only analysis of completed pilot; no scientific imports/calls."""
import csv,hashlib,json,math,statistics,subprocess
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];DEST=Path(__file__).resolve().parent
SOURCE='843265b624eac8fbb9acea47ef21cab1207e1880';PHASE='conditional_s2_pilot_v1';CELL='B2_p5_q8_2nd_r2_K2'
OUT=ROOT/'artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_v1/2026-10-13/launch_pilot_v1'
ANALYSIS=ROOT/'artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13'
EXIT=Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261013/conditional-s2-pilot-launch-v2/coordinator_exit.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(name,obj):
 with (DEST/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def stats(v):return dict(n=len(v),mean=statistics.mean(v),min=min(v),max=max(v),sample_sd=statistics.stdev(v),standard_error=statistics.stdev(v)/math.sqrt(len(v)))
def main():
 m=read(OUT/'frozen_manifest.json');term=read(OUT/'execution_terminal.json');receipt=read(OUT/'pilot_complete_receipt.json');exit=read(EXIT);resource=read(OUT/'pilot_resource_summary.json');calls=read(OUT/'aggregate_calls.json')
 assert m['source_commit']==term['source_commit']==receipt['source_commit']==exit['source_commit']==SOURCE
 assert term['status']=='H6_CONDITIONAL_S2_PILOT_COMPLETE_MANDATORY_STOP' and term['reason'] is None
 assert exit['exit_code']==0 and exit['wait_reaped'] and not exit['automatic_retry']
 assert receipt['group_count']==8 and receipt['wrapper_count']==16 and receipt['mandatory_stop']
 assert resource['first_stop_reason'] is None and resource['cleanup_errors']==[]
 assert resource['all_dispatched_reaped'] and resource['ownership_cleanup_verified'] and resource['log_and_journal_closed']
 assert resource['completed_indices']==list(range(8)) and len(resource['dispatched'])==len(resource['retired'])==8
 for p,h in m['source_hashes'].items():assert sha(ROOT/p)==h,p
 for p,h in m['base_manifest']['input_identity']['parent_raw_hashes'].items():assert sha(ROOT/p)==h,p
 rows=[];groups=[];hashes=[];pair=[];hist=Counter();cpu_claims=[]
 for reg in m['registrations']:
  i=reg['index'];task=reg['task'];d=OUT/f'cost_confirmation_fast_{i:04d}'
  trajectory=read(d/'trajectory.json');condition=read(d/'conditional_sampling_receipt.json');result=read(d/'cost_result.json');workerterm=read(d/'worker_terminal.json');claim=read(d/'worker_claim.json')
  assert task['phase']==PHASE and task['cell_id']==CELL and task['seed']==m['registrations'][i]['task']['seed']
  assert workerterm['status']=='TASK_COMPLETE' and workerterm['reason'] is None and claim['source_commit']==SOURCE
  assert claim['actual_cpus']==[20] and claim['threads']==1 and claim['resource_limits']['rlimit_as']==[32*2**30]*2
  assert all(digest(obj['task'])==digest(task) for obj in (trajectory,condition,result,workerterm))
  assert digest(trajectory['events'])==trajectory['event_digest']==condition['event_digest']
  orders=[e['taylor_order'] for step in trajectory['events'] for e in step];assert len(orders)==16 and set(orders)<=set([0,2]) and 2 in orders
  assert condition['observed_orders']==orders and condition['order2_count']==orders.count(2)
  assert trajectory['finite_distribution']==condition['original_finite_distribution']==m['prior_binding']['finite_distribution']
  assert all(condition[k] is False for k in ('normalization_changed','quantum_event_weights_changed','component_selection_probability_changed'))
  assert result['calls_attempted']==dict(compile=2,trajectory=1,occurrence=8,primitive=0,reference_matvec=0,control_probe=0)
  p2=trajectory['finite_distribution']['order_probabilities'][1];pi2=-math.expm1(16*math.log1p(-p2));eventp=math.prod(e['event_probability'] for step in trajectory['events'] for e in step)
  assert math.isclose(condition['s2_probability'],pi2,rel_tol=1e-14)
  assert math.isclose(condition['conditional_event_sequence_probability'],eventp/pi2,rel_tol=1e-13)
  metrics={};hist[orders.count(2)]+=1
  for row in result['rows']:
   assert row['phase']==PHASE and row['cell_id']==CELL and row['seed']==task['seed'] and row['replica']==task['replica']
   assert row['control']=='symmetric_directional' and row['compiler_hash']=='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
   assert row==read(d/f"wrapper_symmetric_directional_{row['axis']}.json")
   assert row['event_digest']==trajectory['event_digest'] and all(type(v) is int and v>=0 for v in row['metrics'].values())
   metrics[row['axis']]=row['metrics'];rows.append(row)
  assert set(metrics)==set(['cosine','sine']) and len(result['rows'])==2
  pair.append(metrics);cpu_claims.append(claim)
  names=['task_binding.json','worker_claim.json','worker_terminal.json','cost_result.json','trajectory.json','conditional_sampling_receipt.json','wrapper_symmetric_directional_cosine.json','wrapper_symmetric_directional_sine.json','hash_reuse_statistics.json']
  groups.append(dict(index=i,task=task,orders=orders,n2=orders.count(2),event_digest=trajectory['event_digest'],metrics=metrics,conditional_receipt=condition,file_sha256={n:sha(d/n) for n in names}))
  hashes.extend(dict(path=str((d/n).relative_to(ROOT)),bytes=(d/n).stat().st_size,sha256=sha(d/n)) for n in names)
 assert digest(rows)==receipt['rows_digest']
 expected=dict(m['cumulative_accounting']['known_calls'])
 for k,v in m['cumulative_accounting']['new_demand'].items():expected[k]+=v
 assert calls['known_calls']==expected and calls['reservations']==m['cumulative_accounting']['reservations']
 assert calls['historical_unknown_actual_totals'] is None and not calls['budget_reset'] and not calls['failed_calls_refunded'] and calls['unaccepted_new_tasks_fully_reserved']==0
 assert all(expected[k]+calls['reservations'].get(k,0)<=m['cumulative_accounting']['proposed_caps'][k] for k in expected)
 saved=read(ANALYSIS/'selected_trajectory_cost_evidence_v1.json')['records'];zero=[x for x in saved if x['cell_id']==CELL and x['phase']=='confirmation']
 assert len(zero)==32 and all(x['n2']==x['n4']==0 and len(x['orders'])==16 for x in zero)
 assert all(x['finite_distribution']==m['prior_binding']['finite_distribution'] for x in zero)
 old=read(ANALYSIS/'cost_reliability_analysis_v1.json');epsrows=[x for x in old['epsilon_cell_phase_rows'] if x['cell_id']==CELL and x['phase']=='confirmation']
 assert len(epsrows)==4;pi0=1-pi2
 axisstats={axis:{metric:stats([p[axis][metric] for p in pair]) for metric in pair[0][axis]} for axis in ('cosine','sine')}
 comparisons=[]
 for e in epsrows:
  n=e['shot_counts'];v0=[sum(n[a]*x['metrics'][a]['RZ'] for a in ('cosine','sine')) for x in zero];v2=[sum(n[a]*x[a]['RZ'] for a in ('cosine','sine')) for x in pair]
  st0,st2=stats(v0),stats(v2);assert st0['mean']==e['empirical_zero_G'] and st0['mean']==e['G']['mean']
  mixed=pi0*st0['mean']+pi2*st2['mean'];se=math.sqrt((pi0*st0['standard_error'])**2+(pi2*st2['standard_error'])**2)
  comparisons.append(dict(epsilon=e['epsilon'],saved_shot_counts=n,PF_G=e['PF_G'],PF_cell_id=e['PF_cell_id'],s0_confirmation=st0,s2_pilot=st2,pi0=pi0,pi2=pi2,stratified_G_point_estimate=mixed,stratified_G_standard_error=se,
   empirical_confirmation_G_before=st0['mean'],relative_G_change=(mixed/st0['mean']-1),PR_PF_ratio_before=e['PR_PF_ratio'],PR_PF_ratio_stratified=mixed/e['PF_G'],
   RZ_reduction_percent_stratified=100*(1-mixed/e['PF_G']),s2_break_even_G_assuming_s0_mean_fixed=e['rare_union_break_even_G_conditional'],
   target_is_prior_postconfirmation_point_winner=e['epsilon']==.001,prior_exploration_winner='B2_p5_q8_2nd_r2_K4' if e['epsilon']==.001 else None,
   target_selected_after_confirmation=True,formal_confidence_interval=None,population_mean_certified=False,exact_tail_bound=None))
 history=read(ROOT/'artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_preparation_v1/2026-10-13/launch_authorization_v1/prelaunch_rejection_receipt_v1.json')
 write('pilot_metadata_evidence_v1.json',dict(schema='h6_conditional_s2_pilot_metadata_evidence_v1',source_commit=SOURCE,groups=groups,wrapper_rows=rows,native_file_hashes=hashes,old_exploration_confirmation_not_pooled=True,new_scientific_sampling_for_analysis=0,new_transpiles_for_analysis=0))
 write('cost_contribution_analysis_v1.json',dict(schema='h6_conditional_s2_cost_contribution_analysis_v1',source_commit=SOURCE,cell_id=CELL,s2_probability=pi2,s2_pilot_groups=8,s0_confirmation_groups=32,s2_order2_count_histogram=dict(hist),s2_axis_metric_statistics=axisstats,comparisons=sorted(comparisons,key=lambda x:-x['epsilon']),prior_other_three_point_winners_unmodified=True,preselected_K4_winner_unmodified=True,
  sampling_scope='Original s2 law, not forced single order2; finite sample may miss multiple order2 and rare components',method='Known-probability stratified point estimate pi0*mean(s0)+pi2*mean(s2); paired shot weighting; sqrt(pi0^2*SE0^2+pi2^2*SE2^2) under independent samples; descriptive SE, no finite-support bound',
  conditional_cost_support_covered=False,compiled_RZ_population_bound=None,population_winner='UNDETERMINED_POPULATION_MEAN',old_compiler_bit_equivalence_verified=False,additional_scientific_scope_authorized=False))
 copied=['execution_terminal.json','pilot_resource_summary.json','aggregate_calls.json','pilot_complete_receipt.json','conditional_resource_statistics.json']
 for name in copied:
  with (DEST/name).open('xb') as f:f.write((OUT/name).read_bytes())
 with (DEST/'native_coordinator_exit_v1.json').open('xb') as f:f.write(EXIT.read_bytes())
 root_hashes=[dict(path=str((OUT/n).relative_to(ROOT)),bytes=(OUT/n).stat().st_size,sha256=sha(OUT/n)) for n in copied+['execution_claim.json','authorization_source.json','frozen_manifest.json','launch_binding.json','resource_observations.jsonl']]
 write('result_audit_v1.json',dict(status='LOCAL_COMPLETED_CONDITIONAL_PILOT_METADATA_AND_ARITHMETIC_VERIFIED',source_commit=SOURCE,groups=8,wrappers=16,source_files_verified=257,input_byte_files_verified=33,old_runtime_source_changed=False,old_scientific_source_changed=False,old_results_changed=False,
  archive_reused=m['roundoff_preparation_binding'],native_root_hashes=root_hashes,native_exit_sha256=sha(EXIT),grant_sha256=sha(OUT/'authorization_source.json'),prelaunch_rejection=history,coordinator_cpu_set=read(OUT/'execution_claim.json')['resource_limits']['actual_cpus'],worker_cpu_sets=[x['actual_cpus'] for x in cpu_claims],carry=calls,
  no_execution_retry_after_claim=True,science_scope_expanded=False,resource_limits_unchanged=True,CI=False,independent_scientific_review=False,mandatory_stop=True,next_stage_authorized=False,H8_authorized=False,new_science_for_analysis=0,new_transpiles_for_analysis=0))
 with (DEST/'pilot_wrapper_metrics_v1.csv').open('x',newline='') as f:
  fields=['replica','seed','axis','RZ','CX','RZ_depth','CX_depth','depth','size','event_digest','compiler_hash','fingerprint'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
  for row in rows:w.writerow({k:row[k] if k in row else row['metrics'][k] for k in fields})
 print(json.dumps(dict(groups=8,wrappers=16,pi2=pi2,s2_cos_RZ_mean=axisstats['cosine']['RZ']['mean'],s2_cos_RZ_SE=axisstats['cosine']['RZ']['standard_error'],n2_histogram=dict(hist),comparison_epsilon_001=next(x for x in comparisons if x['epsilon']==.001),known_calls=calls['known_calls'],reservations=calls['reservations'],wall_seconds=resource['wall_seconds'])))
if __name__=='__main__':main()
