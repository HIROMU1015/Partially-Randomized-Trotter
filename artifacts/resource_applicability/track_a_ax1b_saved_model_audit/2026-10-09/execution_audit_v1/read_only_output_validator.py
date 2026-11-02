"""Read-only checks and extracts of the one completed run; no analysis/fit calls."""
import pathlib,json,csv,hashlib,subprocess,sys,collections,datetime
root=pathlib.Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928')
out=pathlib.Path(pathlib.Path('/tmp/ax1b_latest_launch_directory.txt').read_text().strip())
sys.path.insert(0,str(root/'src'))
from trottertracks.resource_applicability.ax1b_contract import sha256,digest,require,check_schema,FLAGS
from trottertracks.resource_applicability.ax1b_execution import validate_preparation
from trottertracks.resource_applicability.ax1b_evaluation import validate_selection_record
launch='fc297cd9ab840018c4f35b2764d0b8be07c57285';g=lambda *a:subprocess.check_output(['git',*a],cwd=root)
bundle=json.loads((root/'artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/identity_compatibility_manifest_v1.json').read_text())
allow,plan=validate_preparation(root,bundle)
p=root/plan['output']['directory'];auth=json.loads((out/'execution_authorization_v1.json').read_text());authhash=digest(auth)
require(g('rev-parse','HEAD').decode().strip()==launch,'IMPLEMENTATION','launch source changed')
require({f.name for f in p.iterdir()}==set(plan['output']['planned_files']),'SCHEMA','registered 12 outputs differ')
manifest=json.loads((p/'output_manifest.json').read_text());require(manifest['source_commit']==launch and manifest['mandatory_stop'] is True and manifest['next_stage_authorized'] is False,'SCHEMA','manifest source/STOP flags')
require(manifest['input_allowlist_sha256']==plan['input_allowlist']['sha256'] and manifest['model_configuration_sha256']==plan['model_configuration_sha256'],'CONTRACT_CONFLICT','manifest input/model binding')
require({e['path'] for e in manifest['files']}==set(plan['output']['planned_files'])-{'output_manifest.json'},'SCHEMA','manifest file coverage')
outputhashes=[]
for e in manifest['files']:
 require(sha256((p/e['path']).read_bytes())==e['sha256'],'INPUT_IDENTITY','output hash differs')
for f in sorted(p.iterdir()):outputhashes.append(dict(path=f.name,sha256=sha256(f.read_bytes()),bytes=f.stat().st_size))
require(sha256((p/'output_manifest.json').read_bytes())=='823d07c35fd4a03ddf09ed9dbf595c208edf3bdfa433d906c6c76929f668a8e9','INPUT_IDENTITY','runner output manifest differs')
terminal=json.loads((p/'terminal_status.json').read_text())
require(terminal['status']=='AX1B_COMPLETE_WITH_DECLARED_NA' and terminal['source_commit']==launch and terminal['authorization_sha256']==authhash and terminal['environment']==auth['environment'] and terminal['mandatory_stop'] is True and terminal['next_stage_authorized'] is False,'SCHEMA','terminal authorization binding')
inputs=json.loads((p/'input_identity_audit.json').read_text())
require(len(inputs)==45 and {e['path']:e['sha256'] for e in inputs}=={e['path']:e['sha256'] for e in allow['entries']} and all(e['schema_checked'] and not e['embedded_paths_followed'] for e in inputs),'INPUT_IDENTITY','input audit differs')
for e in allow['entries']:require(sha256((root/e['path']).read_bytes())==e['sha256'],'INPUT_IDENTITY','original input bytes changed')
training={r['candidate_fingerprint']:r for r in allow['membership']['TRAIN_M1_210']['rows']};diag={r['candidate_fingerprint'] for name in ['DIAG_PM1_8','DIAG_M2_5'] for r in allow['membership'][name]['rows']}
fits=json.loads((p/'model_fits.json').read_text());fit_membership=[]
for f in fits['fits']:
 for axis,record in f['axes'].items():
  ids=set(record['training_ids']);require(ids<=set(training) and not ids&diag,'INPUT_IDENTITY','diagnostics leaked into training')
  fold=f['fold_id']
  if fold=='full210':require(ids==set(training),'INPUT_IDENTITY','full210 fit membership')
  else:
   family,group=fold.split(':',1);field={'leave_one_q_out':'q','leave_one_prefix_out':'rank','leave_one_method_out':'method','leave_one_random_K_out':'K'}[family]
   expected={fp for fp,c in training.items() if not (str(c[field])==group and (family!='leave_one_random_K_out' or c['method'] in {'B2','B3'}))}
   require(ids==expected,'INPUT_IDENTITY','fold membership differs')
  require(record['status']=='FIT_OK' and record['audit']['kkt_residual']<=1e-8,'NUMERICAL_FIT','saved KKT/status mismatch')
  fit_membership.append(dict(model_id=f['model_id'],fold_id=fold,axis=axis,training_count=len(ids),KKT_status='PASS'))
schemas=json.loads((root/bundle['schemas']['path']).read_text())['schemas'];predictions=[]
for line in (p/'predictions.jsonl').read_text().splitlines():
 r=json.loads(line);check_schema(r,schemas['prediction'])
 require(r['model_source_commit']==launch and r['model_configuration_sha256']==plan['model_configuration_sha256'],'IMPLEMENTATION','prediction source/config mismatch')
 require(all(r[k] is None for k in ['axis_bias_pred','N_pred_by_axis','G_operational_pred','regret_operational','eligible_operational_pred']) and r['availability_status']=='N_A_NO_OPERATIONAL_BIAS_PREDICTOR','SCHEMA','operational NA changed')
 predictions.append(r)
shot=json.loads((p/'shot_availability.json').read_text());require(shot['reference_reproduced_rows']==67346 and shot['structural_model']['C_pred'] is None,'SCHEMA','PM2 reproduction/STRUCT availability')
rows=list(csv.DictReader((p/'conditional_oracle_selection.csv').open()))
def decoded(row):
 result={}
 for k,v in row.items():
  if v=='':result[k]=None
  elif v in ['True','False']:result[k]=v=='True'
  elif v[0] in '[{':result[k]=json.loads(v)
  elif k in ['regret','full_set_min_G_ref','common_support_regret','known_eligible_subset_regret','known_eligible_prediction_subset_regret','coverage_denominator_reference_eligible','registered_direct_count']:
   result[k]=json.loads(v)
  else:result[k]=v
 return result
status_counts=collections.Counter();unknown=0;common_exclusions=0
for r in rows:
 d=decoded(r)
 if d['row_kind']=='candidate_conditional_oracle_work':continue
 validate_selection_record(d)
 if d['row_kind']=='selection_diagnostic':status_counts[d['status']]+=1;unknown+=len(d['undetermined_reference_candidates'])
 else:
  for node in [d['full_set_diagnostic'],d['common_set_diagnostic']]:check_schema(node,schemas['selection_diagnostic'])
  common_exclusions+=len(d['excluded_direct_candidates'])
 require(d.get('single_frozen_model') is not True or d['fold_id']=='full210','SCHEMA','OOF mislabeled frozen model')
# Compare already-produced finite probabilities/weights against saved finite distributions.
normal=list(csv.DictReader((p/'normalization_audit.csv').open()));signals={}
for e in allow['entries']:
 if e['schema']['schema_version'] in ['pr2_matched_accuracy_m1_a_result_v2','pr2_matched_accuracy_m2_transfer_result_v2']:
  v=json.loads((root/e['path']).read_text());saved=v['signal_records'] if 'signal_records' in v else [r['signal'] for r in v['candidate_results']]
  signals.update({s['candidate_fingerprint']:s for s in saved})
for r in normal:
 dist=signals[r['candidate_fingerprint']]['finite_distribution'];require(json.loads(r['orders'])==dist['orders'],'SCHEMA','finite order mismatch')
 for output_field,input_field in [('probabilities','order_probabilities'),('weights','unnormalized_order_weights')]:
  a=json.loads(r[output_field]);b=dist[input_field];require(len(a)==len(b),'SCHEMA','finite distribution alignment')
  require(all(abs(x-y)<=1e-9+1e-10*abs(y) for x,y in zip(a,b)),'SCHEMA','saved finite distribution differs')
 require(float(r['log_bound_slack'])>=-1e-10,'SCHEMA','stored bound slack violation')
paired=list(csv.DictReader((p/'paired_cost_statistics.csv').open()));sampling=collections.Counter()
for r in paired:
 if r['statistics']:
  s=json.loads(r['statistics']);require(s['n'] in [1,32] and s['formal_ci'] is False,'SCHEMA','paired uncertainty classification')
  sampling[s['n']]+=1
 elif r['reference_RZ_total']:
  s=json.loads(r['reference_RZ_total']);require(s['formal_ci'] is False and s['interval_kind']=='ENGINEERING_INTERVAL_ONLY','SCHEMA','formal interval claim')
before=json.loads((out/'preservation_before.json').read_text())
require(sha256(g('diff','--binary'))==before['diff_sha256'] and not g('diff','--cached','--name-only'),'IMPLEMENTATION','dirty diff/index changed')
for path,expected in before['protected_hashes'].items():require((sha256((root/path).read_bytes()) if (root/path).is_file() else None)==expected,'IMPLEMENTATION','protected file changed')
current_untracked={x for x in g('ls-files','--others','--exclude-standard','-z').decode().split('\0') if x}
expected_new={str(f.relative_to(root)) for f in p.iterdir()}
require(current_untracked==set(before['old_untracked'])|expected_new,'IMPLEMENTATION','unexpected untracked changes')
process=json.loads((out/'process_audit.json').read_text());require(process['launch_count']==1 and process['exit_code']==0 and len(process['successful_allowlisted_reads'])==45 and process['forbidden_science_generation_import_attempts']==0,'IMPLEMENTATION','one-run process audit')
validation=dict(schema_version='track_a_ax1b_saved_output_validation_v1',status='READ_ONLY_VALIDATION_PASS',source_commit=launch,
 output_directory=plan['output']['directory'],output_count=12,total_output_bytes=sum(e['bytes'] for e in outputhashes),output_files=outputhashes,
 output_manifest_sha256=sha256((p/'output_manifest.json').read_bytes()),authorization_canonical_sha256=authhash,environment_fingerprint_sha256=auth['environment']['environment_fingerprint_sha256'],
 input_hashes_and_schemas_verified=45,model_axis_fit_count=len(fit_membership),fit_training_and_KKT_checks=fit_membership,prediction_count=len(predictions),
 PM2_reference_reproduced_rows=shot['reference_reproduced_rows'],finite_distribution_comparisons=len(normal),selection_status_counts=dict(status_counts),
 undetermined_reference_candidates_in_selection=unknown,common_support_excluded_candidates=common_exclusions,paired_metric_sampling_counts=dict(sampling),
 old_dirty_diff_sha256=before['diff_sha256'],old_dirty_protected=True,protected_hash_count=len(before['protected_hashes']),
 no_refit_or_new_performance_evaluation=True,analysis_launch_count=1,mandatory_stop=True,next_stage_authorized=False)
(out/'output_validation_audit.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n')
metrics=list(csv.DictReader((p/'cost_metrics.csv').open()));main=[r for r in metrics if r['row_kind']=='group_summary' and (r['fold_id']=='POOLED_OUT_OF_FOLD_COST' or (r['fold_id']=='full210' and r['group']=='diagnostic_kind'))]
primary_selection=[r for r in rows if r['row_kind']=='selection_diagnostic' and r['fold_id'] in ['full210','POOLED_Q_CROSS_FITTED']]
secondary=[r for r in metrics if r['row_kind']=='group_summary' and r['group']=='axis' and r['group_value']=='"cosine"' and r['fold_id'].startswith('leave_one')]
report_data=dict(schema_version='track_a_ax1b_existing_result_extract_v1',extraction_only=True,source_files=['model_fits.json','cost_metrics.csv','conditional_oracle_selection.csv'],
 complexity_gate=fits['complexity_gate'],main_stored_cost_summaries=main,primary_stored_selection_rows=primary_selection,secondary_stored_cost_summaries=secondary,
 action_rank_diagnostics=fits['index_diagnostics'],missing_operational_predictions=True,full_structural_model_NA=True,mandatory_stop=True,next_stage_authorized=False)
(out/'existing_result_extract.json').write_text(json.dumps(report_data,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:validation[k] for k in ['status','output_count','total_output_bytes','model_axis_fit_count','prediction_count','PM2_reference_reproduced_rows','finite_distribution_comparisons','selection_status_counts','undetermined_reference_candidates_in_selection','common_support_excluded_candidates','paired_metric_sampling_counts']}))
