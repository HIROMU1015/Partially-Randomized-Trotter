"""Read-only saved Stage S audit; no production helper or sin/log reevaluation.

Checks exact interval integer floors and rational algebra using integer pairs.
This is not independent H4/mean/sin/log generation or a platform certificate.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[3]
BASE='0dd660ece1f2e342d2666d37d4b4536adb5e8ca7'
E='artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01'
O='artifacts/track_b_m1s_stage_s_result/2026-10-17/attempt_01'
C='b4b4c12848d0cc65025e1ce9eb95902aec7d97fe'
A='437e6bbc4cc2c550268d8644b8fc2bc0ec56e791'
CP='artifacts/track_b_m1s_stage_s_formal_adoption/2026-10-17/v1/execution_contract_adopted.json'
CHECKS=[]
def check(name,ok):CHECKS.append({'name':name,'status':'PASS' if ok else 'FAIL'})
def git(*args,root=ROOT):return subprocess.check_output(['git','-C',str(root),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(p):return json.loads((ROOT/p).read_text())
def dump(p,d):(ROOT/p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def rational(s):
 n,d=map(int,s.split('/'))
 if d<=0:raise ValueError('nonpositive denominator')
 return n,d
def le(a,b):return a[0]*b[1]<=b[0]*a[1]
def lt(a,b):return a[0]*b[1]<b[0]*a[1]
def mul(a,b):return a[0]*b[0],a[1]*b[1]
def div(a,b):
 if b[0]<=0:raise ValueError('positive rational divisor needed')
 return a[0]*b[1],a[1]*b[0]
def interval(p):return [rational(x) for x in p]
def keys():return list(itertools.product(['G3','G6','G9'],[2,4],['1/8','1/16'],['1','4'],['ordinary','independent_pm']))
def old_status(k):return 'G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED' if k[0]=='G9' else 'NOT_SATISFIED' if k[0]=='G6' and k[3]=='1' else 'SATISFIED'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True,type=Path);a=p.parse_args()
 d=load(O+'/result.json');c=load(CP)
 schema=load('artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/result_schema.json')
 Draft202012Validator(schema).validate(d);check('saved_result_schema',True)
 check('complete_no_failure',d['status']=='STAGE_S_COMPLETE_AWAITING_USER_JUDGMENT' and d['failure'] is None)
 check('one_shot_no_retry',d['one_shot_consumed'] is True and d['retry_count']==0)
 check('S_C_A_provenance',d['provenance']['S']==c['source_identity']['commit'] and d['provenance']['C']==C and d['provenance']['A']==A)
 check('contract_hash',d['provenance']['contract_sha256']==sha((ROOT/CP).read_bytes()))
 marker=load(O+'/one_shot_consumed.json');check('dedicated_marker_consumed',marker['consumed'] is True and marker['retry'] is False and marker['provenance']==d['provenance'])
 launch=load(E+'/launch_receipt.json');check('one_production_launch_returncode_zero',launch['production_launches']==1 and launch['retry_count']==0 and launch['returncode']==0)
 check('no_GO_RPE_native_total_claim',all(d[x] is None for x in ['scientific_GO','RPE_success','native_costs','total_resources']) and d['global_integer_impossibility'] is False)
 check('conditional_coverage_only',d['evidence']=='CONDITIONAL_ENGINEERING_NOT_CERTIFIED' and d['coverage_scope']=='SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY')
 check('cap_open_no_grid_no_new_pi',d['cap']['endpoint_open'] is True and d['cap']['gamma_grid'] is None and d['cap']['pi_recomputed'] is False)
 cap=interval(d['cap']['interval_rad']);pi=interval(d['adapter']['saved_pi_interval_rad'])
 check('cap_is_saved_pi_over_three',all(x[0]*p[1]*3==p[0]*x[1] for x,p in zip(cap,pi)))
 check('full108_inventory_zero_not_reanalyzed',d['adapter']['108_row_inventory_checked'] is True and d['adapter']['zero_controls_reanalyzed'] is False)
 rows=d['comparisons'];ids=[tuple(r[k] for k in ['split','R','delta','T','comparator']) for r in rows]
 check('all48_unique_fixed_grid',len(ids)==len(set(ids))==48 and set(ids)==set(keys()))
 K=interval(d['statistics']['K_interval']);check('common_positive_K',lt((0,1),K[0]) and le(K[0],K[1]))
 check('same_adopted_confidence',d['statistics']['alpha_axis']=='1/40' and d['statistics']['confidence_scope']=='PER_METHOD_ROW' and c['selected_science']['alpha_row']=='1/20')
 check('fixed_precision_no_refinement',d['arithmetic']['bits']==128 and d['arithmetic']['refinements']==0)
 check('bounded_steps',d['arithmetic']['bound_steps']<=c['resources']['max_bound_steps'])
 counts=collections.Counter();by_split={};eligible=0
 for idx,(k,r) in enumerate(zip(ids,rows)):
  label=str(idx)+':'+repr(k);s=r['new_stage_s_status'];counts[s]+=1;by_split.setdefault(k[0],collections.Counter())[s]+=1
  check('old_status_preserved:'+label,r['old_pi3_gate_status']==old_status(k))
  if old_status(k)!='SATISFIED':
   expected='G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED' if k[0]=='G9' else 'CARRIED_PI3_NOT_SATISFIED_NOT_EVALUATED'
   check('carry_status:'+label,s==expected)
   check('no_new_numbers_on_carry:'+label,not any(f in r for f in ['R_cont_cap_interval','S_shared_at_cap_interval','S_comparator_at_cap_interval','n_shared_cap_left','n_comparator_cap_left']))
   continue
  eligible+=1
  check('ideal_raw_no_model_double_count:'+label,r['model_drift_added_again'] is False)
  av=r['input_bounds'];aa=rational(av['A_shared_lower']);cc=rational(av['A_comparator_lower']);u=rational(av['bias_shared_upper_rad']);v=rational(av['bias_comparator_upper_rad'])
  check('direct_tradeoff_domain:'+label,lt((0,1),cc) and lt(cc,aa) and le(aa,(1,1)) and le((0,1),v) and lt(v,u) and lt(u,cap[0]))
  R=interval(r['R_cont_cap_interval']);Ss=interval(r['S_shared_at_cap_interval']);Sc=interval(r['S_comparator_at_cap_interval'])
  check('positive_ordered_intervals:'+label,all(lt((0,1),p[0]) and le(p[0],p[1]) for p in [R,Ss,Sc]))
  check('continuous_cap_shared_lower_design:'+label,lt((1,1),R[0]))
  # R = Sc/Ss. Independently ensure the two reported interval expressions overlap.
  algebra=[div(Sc[0],Ss[1]),div(Sc[1],Ss[0])]
  check('R_and_S_interval_algebra_compatible:'+label,le(R[0],algebra[1]) and le(algebra[0],R[1]))
  vals=[]
  for name,pair in [('shared',Ss),('comparator',Sc)]:
   ns=r['n_'+name+'_cap_left'];floors=[x[0]//x[1]+1 for x in pair]
   check('independent_floor_plus_one:'+name+':'+label,ns['interval']==floors)
   check('integer_resolution:'+name+':'+label,ns['resolved']==(floors[0]==floors[1]) and ns['value']==(floors[0] if floors[0]==floors[1] else None))
   check('two_axes_total:'+name+':'+label,r['N_'+name+'_cap_left_interval']==[2*x for x in floors])
   vals.append(ns)
  if not all(n['resolved'] for n in vals):expected='ARITHMETIC_UNRESOLVED'
  else:expected='CAP_EDGE_STRICT_INTEGER_WITNESS' if vals[0]['value']<vals[1]['value'] else 'CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED'
  check('integer_classification:'+label,s==expected)
  check('existence_not_global_impossibility:'+label,r['nonempty_open_region_existence_witness']==(s=='CAP_EDGE_STRICT_INTEGER_WITNESS') and r['all_gamma_impossibility_established'] is False)
 check('exactly24_numeric_contrasts',eligible==24)
 check('all36_margin_and72_sine_calls_returned',d['arithmetic']['scalar_margins_completed_in_returned_cells']==36 and d['arithmetic']['sin_endpoint_calls_completed_in_returned_cells']==72 and d['arithmetic']['log_calls']==1)
 check('hardware_based_workers',d['resources']['selected_workers']==min(d['resources']['logical_cpus'],12) and d['resources']['CPU_RAM_GPU_worker_caps'] is None)
 for path,e in c['source_identity']['files'].items():
  raw=(ROOT/path).read_bytes();check('source_unchanged:'+path,len(raw)==e['bytes'] and sha(raw)==e['sha256'] and raw==git('show',c['source_identity']['commit']+':'+path))
 for name,e in c['inputs'].items():
  raw=(ROOT/e['path']).read_bytes();check('saved_input_preserved:'+name,len(raw)==e['bytes'] and sha(raw)==e['sha256'] and raw==git('show',e['commit']+':'+e['path']))
 for folder in [ROOT/E,ROOT/O]:
  for path in folder.iterdir():
   if path.is_file():check('output_log_byte_cap:'+str(path.relative_to(ROOT)),path.stat().st_size<=c['resources']['log_bytes' if path.suffix in ['.log','.txt'] else 'result_bytes'])
 # Display only saved output, no gamma evaluation/selection or new shot forecast.
 columns=['split','R','delta','T','comparator','old_pi3_gate_status','old_absolute_contrast_status','new_stage_s_status','R_cont_cap_interval','S_shared_at_cap_interval','S_comparator_at_cap_interval','n_shared_cap_left','n_comparator_cap_left','N_shared_cap_left_interval','N_comparator_cap_left_interval']
 with (ROOT/E/'stage_s_comparisons.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=columns,lineterminator="\n");writer.writeheader()
  for r in rows:writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k,'') for k in columns})
 summary={'status':d['status'],'scope':c['scope'],'selected_science':c['selected_science'],'comparison_counts':dict(counts),'by_split':{g:dict(v) for g,v in by_split.items()},'raw_result_path':O+'/result.json','raw_result_bytes':(ROOT/O/'result.json').stat().st_size,'raw_result_sha256':sha((ROOT/O/'result.json').read_bytes()),'production_launches':1,'retry_count':0,'source_changed':False,'scientific_GO':None,'RPE_success':None,'native_costs':None,'mandatory_STOP':True}
 dump(E+'/saved_result_summary.json',summary)
 # Preserve every pre-existing non-index blob; allowed current updates are prepend indexes/manifest.
 indexes=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md'];manifest='artifacts/validation_manifest.json'
 pchecks=[]
 def pcheck(n,v):pchecks.append({'name':n,'status':'PASS' if v else 'FAIL'})
 mismatches=[];nold=0
 for record in git('ls-tree','-rz',BASE).split(b'\0'):
  if not record:continue
  header,p=record.split(b'\t',1);mode,kind,oid=header.decode().split();p=p.decode()
  if kind!='blob' or p in indexes+[manifest]:continue
  q=ROOT/p;raw=os.readlink(q).encode() if mode=='120000' and q.is_symlink() else q.read_bytes() if q.is_file() else None;nold+=1
  if raw is None or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=oid:mismatches.append(p)
 pcheck('all_old_nonindex_blobs_preserved',not mismatches)
 for path in indexes:pcheck('index_prepend_only:'+path,(ROOT/path).read_bytes().endswith(git('show',BASE+':'+path)))
 oldm=json.loads(git('show',BASE+':'+manifest));newm=load(manifest);pcheck('old_manifest_entries_preserved',all(newm.get(k)==v for k,v in oldm.items()))
 snap=json.loads(a.snapshot.read_text());parent=Path(snap['root']);changed=[]
 for path,e in snap['tracked'].items():
  q=parent/path;got=sha(q.read_bytes()) if q.is_file() else None
  if got!=e:changed.append(path)
 pcheck('parent_tracked_preserved',not changed);pcheck('parent_HEAD_preserved',git('rev-parse','HEAD',root=parent).decode().strip()==snap['root_HEAD'])
 ix=Path(git('rev-parse','--git-path','index',root=parent).decode().strip());ix=ix if ix.is_absolute() else parent/ix
 pcheck('parent_index_preserved',sha(ix.read_bytes())==snap['root_index_sha256']);pcheck('parent_tracked_status_preserved',git('status','--short','--untracked-files=no',root=parent).decode()==snap['root_status'])
 pcheck('review_original_preserved',sha((parent/'m1s_stage_s_scientific_conditions_review_20261017.md').read_bytes())==snap['stage_s_review_sha256'])
 changed_heads=[p for p,h in snap['worktree_HEADs'].items() if git('rev-parse','HEAD',root=Path(p)).decode().strip()!=h]
 pcheck('prior_worktree_HEADs_preserved',not changed_heads)
 preserved={'status':'PASS' if all(x['status']=='PASS' for x in pchecks) else 'FAIL','checks':pchecks,'old_nonindex_blobs_inspected':nold,'old_blob_mismatches':mismatches,'parent_tracked_files':len(snap['tracked']),'parent_changed_files':changed,'prior_worktree_HEADs':len(snap['worktree_HEADs']),'changed_worktree_HEADs':changed_heads,'untracked_other_worktree_inventory_claimed':False}
 dump(E+'/preservation_audit.json',preserved)
 report={'status':'PASS' if all(x['status']=='PASS' for x in CHECKS) else 'FAIL','checks_PASS':sum(x['status']=='PASS' for x in CHECKS),'checks_FAIL':sum(x['status']=='FAIL' for x in CHECKS),'checks':CHECKS,'scope':'SAVED_RESULT_SCHEMA_INTEGER_CROSS_PRODUCT_FLOOR_INTERVAL_PROVENANCE_ONLY','production_helpers_imported':False,'Fraction_used':False,'sin_log_pi_or_H4_reference_recomputed':False,'scientific_GO':None,'mandatory_STOP':True}
 dump(E+'/saved_result_audit.json',report)
 print(json.dumps({'saved_audit':report['status'],'checks_PASS':report['checks_PASS'],'checks_FAIL':report['checks_FAIL'],'preservation':preserved['status'],'comparison_counts':dict(counts)}))
 for x in CHECKS+pchecks:
  if x['status']=='FAIL':print(x['name'])
 return 0 if report['status']==preserved['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
