"""Format completed fixed-run JSON/CSV and byte-check provenance; no new science."""
from pathlib import Path
import json,hashlib,csv,collections
base=Path('artifacts/research_c_c1_h6_synthesis_bound/2026-10-25');p=base/'run01'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,x):Path(f).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
r=read(p/'result.json');stop=read(p/'STOP.json');identity=read(base/'execution_identity.json');c=read(identity['contract']);manifest=read(c['source_manifest']);groups=read(p/'route_identity.json')
assert r['status']=='H6_NUMERICAL_RESTRICTED_BOUND_COMPLETE' and not stop['failed']
assert identity['contract_sha256']==sha(identity['contract']) and identity['source_manifest_sha256']==sha(c['source_manifest'])
for group in [manifest['files'],c['saved_inputs'],identity['protected_hashes']]:
 for f,h in group.items():assert sha(f)==h,f
assert sha(p/'contract.json')==sha(identity['contract'])
e=[json.loads(s) for s in (p/'science_calls.jsonl').read_text().splitlines()];a={};term={};counts=collections.Counter();complete=collections.Counter()
for x in e:
 key=(x['kind'],x['label'])
 if x['event']=='attempt':assert key not in a;a[key]=x;counts[x['kind']]+=x.get('count',1)
 else:assert key in a and key not in term;term[key]=x;assert x['event']=='COMPLETE';complete[x['kind']]+=a[key].get('count',1)
assert a.keys()==term.keys() and dict(counts)==stop['call_counts'] and counts==complete
assert sum(counts.values())==stop['scientific_units']==135+800*len(groups['groups'])==3335
assert all(counts[k]<=c['call_limits'][k] for k in counts)
assert stop['retry']==stop['resume']==0 and len(r['routes'])==8
kernel_rows=[];round_rows=[];column_rows=[];batch_maxima={}
for labels in groups['groups']:
 label=labels[0];column=read(p/'groups'/(label+'_columns.json'));batches=[json.loads(s) for s in (p/'groups'/(label+'_batches.jsonl')).read_text().splitlines()]
 assert column['status']=='COMPLETE' and column['covered_columns']==800 and column['column_order']==list(range(800))
 assert [v['column'] for v in column['columns']]==list(range(800))
 seen=[i for b in batches for i in b['columns']];assert sorted(seen)==list(range(800)) and len(set(seen))==800
 bounds={i:b for b in batches for i in b['columns']}
 for v in column['columns']:
  b=bounds[v['column']];column_rows.append(dict(unique_group=label,column=v['column'],control=v['column']//400,sector_column=v['column']%400,full_output_residual_L2=v['L2'],max_component_upper_from_batch=b['max_entry'],norm_drift_upper_from_batch=b['norm_drift'],leakage_L2_upper_from_total_residual=v['L2']))
 batch_maxima[label]=dict(max_norm_drift=max(b['norm_drift'] for b in batches),batch_max_component=max(b['max_entry'] for b in batches))
 for alias in labels:assert groups['identities'][alias]['signature']==groups['identities'][label]['signature']
for label,v in r['routes'].items():
 k=v['kernel'];rows=v['rounds'];assert len(rows)==13 and [t['m'] for t in rows]==list(range(13))
 assert v['status']=='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL'
 assert all(x['status']=='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL' and x['phase_bias_bound_pilot']<=x['registered_phase_room'] for x in rows)
 assert k['evidence_level']=='B_NUMERICAL_OPERATOR_UPPER_BOUND' and k['rigorous_operator_bound'] is None and k['global_full_Fock_operator_bound'] is None
 assert all(v[f] is None for f in ['CA_achieved','shots','total_cost'])
 for t in rows:
  row={x:y for x,y in t.items() if not isinstance(y,(dict,list))};row['route']=label
  row['axis_X_state_bound_pilot']=t['synthesis_axis_state_bound_pilot']['X'];row['axis_Y_state_bound_pilot']=t['synthesis_axis_state_bound_pilot']['Y'];round_rows.append(row)
 end=rows[-1];kernel_rows.append(dict(route=label,evaluated_route=k['evaluated_route'],coverage=k['covered_columns'],kernel_bound=k['restricted_operator_bound_pilot'],max_component=k['max_entry'],**batch_maxima[k['evaluated_route']],reference_defect_max=k['unitarity_defect_max_entry'],reference_defect_F=k['unitarity_defect_frobenius'],reference_alpha=k['reference_norm_upper_pilot'],leakage_operator_upper_from_total_error=k['restricted_operator_bound_pilot'],m12_repeated_kernel_bound=end['operator_bound_pilot'],m12_synthesis_complex_bound=end['complex_signal_bound_pilot'],m12_saved_PF_error=end['saved_PF_operator_error'],m12_combined_complex_bound=end['combined_complex_error_pilot'],m12_phase_bound=end['phase_bias_bound_pilot'],max_phase_bound=max(x['phase_bias_bound_pilot'] for x in rows),classification=v['status']))
for filename,rows in [('kernel_bounds.csv',kernel_rows),('round_bounds.csv',round_rows),('column_residuals.csv',column_rows)]:
 with (base/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
summary=dict(status=r['status'],scope=c['scope'],SOURCE_commit=identity['SOURCE_commit'],SOURCE_tree=identity['SOURCE_tree'],branch=identity['branch'],contract=identity['contract'],contract_sha256=identity['contract_sha256'],source_manifest_sha256=identity['source_manifest_sha256'],conditions=c['conditions'],compiler=c['compiler'],evidence_level=r['evidence_level'],unique_groups=len(groups['groups']),route_count=8,complete_columns_per_unique_group=800,unique_state_action_columns=3200,input_dimension=800,output_dimension=8192,scientific_units=stop['scientific_units'],maximum_logical_units=c['maximum_logical_units'],call_counts=dict(counts),scientific_COMPLETE=sum(complete.values()),scientific_FAIL=0,inflight_calls=0,round_rows=104,round_PASS=104,route_classifications={k:v['status'] for k,v in r['routes'].items()},kernel_rows=kernel_rows,resources=read(p/'resources.json'),rigorous_enclosure=None,global_full_Fock_operator_bound=None,CA_achieved=None,shots=None,total_cost=None,new_fit=0,optimizer=0,rank_search=0,DF_recomputation=0,PF_recomputation=0,build_compile=0,retry=0,resume=0,mandatory_stop=True,leakage_record='Full8192 outputs retained. Leakage is included in total residual; separate measured leakage not persisted. Reported leakage bounds reuse total residual L2/Frobenius (level B), not a claim of zero leakage.',column_record='L2 persisted per column; component and norm-drift maxima persisted per8-column batch. CSV gives these as upper bounds per member, not separate per-column point measurements.')
save(base/'execution_summary.json',summary)
raw={str(f):sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
post=dict(status='PASS',SOURCE_files=len(manifest['files']),saved_input_hashes=len(c['saved_inputs']),protected_files_unchanged=len(identity['protected_hashes']),contract_unchanged=True,source_manifest_unchanged=True,coverage='4x800 unique basis columns, each full8192 output;8 routes via exact IR identities',complete_logical_units=sum(complete.values()),FAIL=0,inflight=0,raw_files=len(raw),raw_sha256=raw,derived_CSV_rows=dict(kernel_bounds=8,round_bounds=104,column_residuals=3200))
save(base/'postflight.json',post)
print(json.dumps(dict(postflight={k:v for k,v in post.items() if k!='raw_sha256'},kernel_rows=kernel_rows,resources=summary['resources']),indent=2))
