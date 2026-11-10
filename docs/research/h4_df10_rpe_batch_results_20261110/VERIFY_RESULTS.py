"""Read-only validation/export of the authorized batch. No signal/N solver."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src')]
from trottertracks import pf_pr_rpe_pipeline as pipeline

HERE=Path(__file__).resolve().parent
raw_bytes=(HERE/'run01/result.json').read_bytes();raw=json.loads(raw_bytes)
handoff=json.loads((HERE/'N_HANDOFF.json').read_bytes())
proposal=json.loads((ROOT/'docs/research/h4_df10_rpe_batch_preparation_20261110/PLAN.json').read_bytes());task=proposal['task']
blob=lambda commit,path:subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert raw['scientific_source']=='90dcc3ed20f30dfab6b22a68cc4adb9461485a98'
assert raw['status']=='H4_N_BATCH_EVALUATED' and not raw['errors'] and raw['source_preserved'] and raw['input_preserved']
assert sha(HERE/'run01/result.json')==handoff['raw_result_sha256']
for path,digest in raw['source_files'].items():
 assert sha(ROOT/path)==digest and (ROOT/path).read_bytes()==blob(raw['scientific_source'],path)
for ref,digest in raw['saved_sources'].items():
 commit,path=ref.split(':',1);assert hashlib.sha256(blob(commit,path)).hexdigest()==digest
index=json.loads(blob('7a8bc7f132e166390c943e6b32c92cc6abe1baf4','artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json'))
for spec in list(index['saved_inputs']['H4']['inputs'].values())+[index['T20_reference_files']['H4']]:assert sha(spec['path'])==spec['sha256']
d1=json.loads(blob('cefc01622a3cd75dd16d30de7b9fb68274f3eddd','docs/research/h4_df10_rpe_d1_20261110/result.json'))
byid={n['identity']['candidate']['id']:n for n in raw['records']};verified={n['identity']['candidate']['id']:n for n in handoff['shots']}
for old in d1['rows']:
 n=byid[old['candidate_id']];v=verified[old['candidate_id']]
 assert n['plan']==v['plan']==old['ideal_numerical_pilot'] and n['actual']==v['actual']==old['actual_implementation']
assert len(byid)==len(verified)==26
for row in task['candidates']:
 n=byid[row['candidate']['id']];v=verified[row['candidate']['id']];d=n['diagnostics']
 assert n['identity']==pipeline.identity(task,row)==v['identity']
 assert json.loads((HERE/'run01'/(row['candidate']['id']+'.json')).read_bytes())==n
 assert n['plan']['status']==v['plan']['status']=='CONDITIONAL_ROUND_PASS'
 assert all(n['actual']['N_'+a] is None and v['actual']['N_'+a] is None for a in ('X','Y'))
 assert d['ground_state_certified'] is False and d['full_rpe_energy_success'] is False
 if row['m']==3:continue
 assert d['signed_weights']==[1.] and d['allocation']==([4] if row['candidate']['method']=='PR' else [0])
 for a in ('X','Y'):
  axis=n['plan']['axes'][a];check=v['diagnostics']['integer_checks'][a]
  assert axis['radius_lower']>0 and axis['beta_total']<=task['configuration']['beta_total']
  assert axis['quantum_shots_drawn']==0
  assert v['plan']['N_'+a]>=int(check['N_high_precision_ceil_decimal']) and v['plan']['N_'+a]>=n['plan']['N_'+a]
 if row['m']!=4:
  assert d['numerical_reference']['signal_difference']<=1e-10
  ev=d['signal']['allowance_evidence'];assert ev['exact_tail_norm_error']<=1e-9 and ev['raw_corrected_vector_closure']<=1e-9
  assert d['event_count']==(4*row['q_PF'] if row['candidate']['method']=='PR' else 0)
refs=raw['scope_observed']['references'];new=[n for n in raw['records'] if n['identity']['m']!=3]
inventory=pipeline.plan(task,handoff['shots'],[])
assert not inventory['missing_N'] and not inventory['unqualified_N'] and inventory['totals'] is None
h6path='docs/research/h6_df19_rpe_m8_12_results_20261110/run01/result.json'
assert (ROOT/h6path).read_bytes()==blob(raw['scientific_source'],h6path)
checks=dict(status='PASS_AUTHORIZED_BATCH_VALIDATION',raw_scientific_source=raw['scientific_source'],
 integer_check_source=handoff['integer_check_source'],fixed_source_files_preserved=len(raw['source_files']),
 saved_artifact_hashes_preserved=len(raw['saved_sources']),saved_input_context_files_preserved=3,
 old_m3_N_lanes_equal=True,H6_result_preserved=True,original_raw_result_preserved=True,
 reference_pair_difference_max=max(r['signal_difference'] for r in refs.values()),
 exact_tail_norm_error_max=max(n['diagnostics']['signal']['allowance_evidence']['exact_tail_norm_error'] for n in new if n['identity']['m']!=4),
 raw_corrected_vector_closure_max=max(n['diagnostics']['signal']['allowance_evidence']['raw_corrected_vector_closure'] for n in new if n['identity']['m']!=4),
 beta_deterministic_max=max(n['plan']['axes']['X']['beta_deterministic'] for n in new),
 integer_axes_checked=48,integer_axes_corrected=len(handoff['changed_axes']),
 common_interface_missing_N=[],common_interface_unqualified_N=[],
 actual_N_qualified=False,full_RPE_qualified=False,C=None,G=None,mandatory_stop=True)
(HERE/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
fields=['m','method','candidate_id','T','q_RPE','q_PF','formula','prefix','R_outer','K','source_route','reference_real','reference_imag','corrected_real','corrected_imag','raw_real','raw_imag','bias_real','bias_imag','log_B','B','attenuation','corrected_disk','radius_lower','beta_det','beta_det_plus_stat','N_X_raw','N_Y_raw','N_X_handoff','N_Y_handoff','status','raw_scientific_source']
with (HERE/'ROUND_SUMMARY.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
 for row in task['candidates']:
  c=row['candidate'];n=byid[c['id']];v=verified[c['id']];d=n['diagnostics'];sig=d.get('signal',{});axis=n['plan']['axes']['X']
  # Original m3 diagnostics predate signal storage; copy the source fixture's
  # recorded complex fields only when present, never rerun/rederive m3.
  rec=dict(m=row['m'],method=c['method'],candidate_id=c['id'],T=c['T'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],formula=c['formula'],prefix=c['prefix'],R_outer=c['R_outer'],K=c['K'],
   source_route=n['origin'],log_B=d['normalization']['log_B'],B=d['normalization']['B'],attenuation=d['normalization']['attenuation'],
   corrected_disk=d['error_disk']['corrected_signal_disk'],radius_lower=axis['radius_lower'],beta_det=axis['beta_deterministic'],beta_det_plus_stat=axis['beta_total'],
   N_X_raw=n['plan']['N_X'],N_Y_raw=n['plan']['N_Y'],N_X_handoff=v['plan']['N_X'],N_Y_handoff=v['plan']['N_Y'],status=n['plan']['status'],raw_scientific_source=n['source'].get('scientific_source',raw['scientific_source']))
  for field in ('reference','corrected','raw','bias'):
   value=sig.get('total_signed_bias' if field=='bias' else field,{})
   for part in ('real','imag'):rec[field+'_'+part]=value.get(part,'')
  writer.writerow(rec)
print(json.dumps(checks,ensure_ascii=False))
