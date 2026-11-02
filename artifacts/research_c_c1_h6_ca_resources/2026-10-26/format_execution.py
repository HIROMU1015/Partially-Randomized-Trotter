"""Format completed saved-H6 scalar results; no solver/operator recalculation."""
from pathlib import Path
import json,hashlib,csv,collections
base=Path('artifacts/research_c_c1_h6_ca_resources/2026-10-26');out=base/'run01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
r=read(out/'result.json');stop=read(out/'STOP.json');ident=read(base/'execution_identity.json');c=read(ident['contract']);src=read(c['source_manifest'])
assert r['status']=='H6_SAVED_CA_RPE_SERIAL_RESOURCES_COMPLETE' and not stop['failed']
assert sha(ident['contract'])==ident['contract_sha256'] and sha(c['source_manifest'])==ident['source_manifest_sha256']
for group in [src['files'],c['saved_inputs'],ident['protected_hashes']]:
 for p,h in group.items():assert sha(p)==h,p
assert sha(out/'contract.json')==sha(ident['contract']) and sha(out/'SOURCE.json')==sha(c['source_manifest'])
events=[json.loads(s) for s in (out/'science_calls.jsonl').read_text().splitlines()];attempt={};terminal={};counts=collections.Counter()
for e in events:
 key=(e['kind'],e['label'])
 if e['event']=='attempt':assert key not in attempt;attempt[key]=e;counts[e['kind']]+=e.get('count',1)
 else:assert key in attempt and key not in terminal and e['event']=='COMPLETE';terminal[key]=e
assert attempt.keys()==terminal.keys() and dict(counts)==stop['call_counts']==r['call_counts']
assert sum(counts.values())==stop['scientific_units']==r['scientific_units']==c['maximum_logical_units']==119
assert all(counts[k]<=v for k,v in c['call_limits'].items()) and stop['retry']==stop['resume']==0
candidates=[];axes=[]
for key in c['candidate_order']:
 v=r['candidates'][key];p=read(out/'candidates'/(key.replace(':','_')+'.packet.json'));plans=v['rounds'];cost={(x['m'],x['axis']):x for x in v['per_shot_costs']}
 assert v['status']=='NUMERICAL_PILOT_CA_PASS' and len(plans)==13 and len(cost)==26 and [x['m'] for x in plans]==list(range(13))
 assert v['CA_achieved'] is v['rigorous_enclosure'] is v['guaranteed_actual_shots'] is None
 assert v['guaranteed_total_resource_superiority'] is False and v['numerical_energy_assessment']['statistical_failure_upper']<=.01
 for q in plans:
  assert q['status']=='CONDITIONAL_ROUND_PASS' and q['statistical_margin']>=0 and q['raw_signal_error']==q['source_combined_complex_error']
  assert q['quantum_shots_drawn']==0 and type(q['shots']) is int and q['shots']>0
  for axis in ('X','Y'):
   x=cost[q['m'],axis];assert x['kernel_occurrences']==v['s']*(1<<q['m'])
   axes.append(dict(candidate=key,m=q['m'],axis=axis,occurrences=x['kernel_occurrences'],shots_decimal=q['axes_shots_decimal'][axis],saved_PF_error=q['source_PF_error'],saved_synthesis_complex_bound=q['source_synthesis_complex_error'],combined_complex_error=q['raw_signal_error'],radius_lower=q['radius_lower'],beta_deterministic=q['beta_deterministic'],beta_total=q['beta_total'],alpha_axis=q['alpha_axis'],statistical_margin=q['statistical_margin'],**{'per_shot_'+k:value for k,value in x['mean'].items()}))
 total=v['cost'];assert total['shots']==total['executions']==total['measurement_count']==sum(2*q['shots'] for q in plans)
 for k,value in total['totals'].items():assert value.is_integer() and 0<=value<1<<53 and total['intervals'][k]==[value,value]
 candidates.append(dict(candidate=key,classification=v['status'],df_norm_plus_guard_Ha=p['df_error_with_guard'],numerical_energy_error_Ha=v['numerical_energy_assessment']['energy_error_bound_ha'],candidate_failure_upper=v['numerical_energy_assessment']['statistical_failure_upper'],shots=total['shots'],executions=total['executions'],measurements=total['measurement_count'],**{k:int(value) for k,value in total['totals'].items()}))
comparisons=[]
for s in (1,4):
 a,b=(r['candidates'][f'{arm}:s{s}'] for arm in ('CDF','XDF'))
 assert [p['shots'] for p in a['rounds']]==[p['shots'] for p in b['rounds']]
 x,y=(v['cost']['totals']['rz_count'] for v in (a,b))
 comparisons.append(dict(r['comparisons'][str(s)],CDF_minus_XDF_RZ=int(x-y),CDF_RZ_premium_percent=(x/y-1)*100,shots_identical_all_rounds=True))
for name,rows in [('candidate_totals.csv',candidates),('round_axis_results.csv',axes)]:
 with (base/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
summary=dict(status=r['status'],scope=c['scope'],SOURCE_commit=ident['SOURCE_commit'],SOURCE_tree=ident['SOURCE_tree'],branch=ident['branch'],contract=ident['contract'],contract_sha256=ident['contract_sha256'],source_manifest_sha256=ident['source_manifest_sha256'],conditions=c['conditions'],accuracy_target=c['accuracy_target'],compiler=c['compiler'],cost_scope=c['cost_scope'],scientific_units=119,maximum_logical_units=119,call_counts=dict(counts),scientific_COMPLETE=119,scientific_FAIL=0,inflight_calls=0,candidate_PASS=4,round_PASS=52,round_axis_cost_rows=104,candidates=candidates,comparisons=comparisons,resources=read(out/'resources.json'),evidence_level='B_NUMERICAL_OPERATOR_UPPER_BOUND',rigorous_enclosure=None,CA_achieved=None,guaranteed_actual_shots=None,guaranteed_total_resource_superiority=False,state_preparation_cost=None,statistical_scope='one predeclared candidate at failure upper0.01; no four-candidate or post-selection guarantee',new_fit=0,optimizer=0,rank_search=0,DF_recomputation=0,PF_recomputation=0,synthesis_recomputation=0,state_action=0,build_compile=0,quantum_shots_drawn=0,retry=0,resume=0,mandatory_stop=True)
save(base/'execution_summary.json',summary)
raw={str(p):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
post=dict(status='PASS',SOURCE_files=len(src['files']),saved_input_hashes=len(c['saved_inputs']),protected_files_unchanged=len(ident['protected_hashes']),contract_unchanged=True,source_manifest_unchanged=True,complete_logical_units=119,FAIL=0,inflight=0,raw_files=len(raw),raw_sha256=raw,derived_CSV_rows=dict(candidate_totals=4,round_axis_results=104))
save(base/'postflight.json',post)
print(json.dumps(dict(candidates=candidates,comparisons=comparisons,postflight={k:v for k,v in post.items() if k!='raw_sha256'}),indent=2))
