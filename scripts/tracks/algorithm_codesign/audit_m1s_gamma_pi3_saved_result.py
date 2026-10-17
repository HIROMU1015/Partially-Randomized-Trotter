"""Saved-result/provenance audit, independent integer cross-products for this gate.

Does not import or rerun the numerical SOURCE. Does not regenerate gamma/pi,
Hamiltonians/signals, integer shots, native circuits, or RPE rounds.
"""
from collections import Counter
import hashlib
import json
from math import gcd
import os
from pathlib import Path
import re
import subprocess
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[3]
S='2f800f179c7b29589b2f9a7e532408b16b8fe6d2'
C='bf97fcb2607904f8b96a625c5b64440a78bbbfdc'
A='a1dd87a1f81467bea0ce27cacf9d21e49cff7908'
ART='artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01'
OUT='artifacts/track_b_m1s_gamma_pi3_result/2026-10-17/attempt_01'
AUTH='artifacts/track_b_m1s_gamma_pi3_execution_authorization/2026-10-17/attempt_01/authorization.json'
CONTRACT='artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/execution_contract_adopted.json'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md']
MANIFEST='artifacts/validation_manifest.json'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def digest(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads((ROOT/path).read_bytes())


def integers(text):
    if type(text) is not str or len(text)>25000 or re.fullmatch(r'-?[0-9]+/[1-9][0-9]*',text) is None:
        raise ValueError('bounded saved rational required')
    num,den=map(int,text.split('/'))
    if max(num.bit_length(),den.bit_length())>65536:
        raise ValueError('saved rational bit cap')
    return num,den


def main():
 checks=[]
 def check(name,ok):checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
 c,a,r,supervisor,marker=[load(p) for p in [CONTRACT,AUTH,OUT+'/result.json',OUT+'/supervisor_result.json',OUT+'/exclusive_launch_marker.json']]
 check('authority_only_direct_child_of_fixed_C',git('rev-parse',A+'^').decode().strip()==C and git('diff-tree','--no-commit-id','--name-only','-r',A).decode().splitlines()==[AUTH])
 check('contract_original_C_bytes',git('show',C+':'+CONTRACT)==(ROOT/CONTRACT).read_bytes())
 check('authority_original_A_bytes',git('show',A+':'+AUTH)==(ROOT/AUTH).read_bytes())
 check('separate_user_execution_approval','計算を開始してよい' in a['user_instruction'] and a['scientific_execution_authorized'] is True)
 check('one_shot_no_authority_reuse',a['one_shot'] is True and a['retry'] is False and marker['consumed'] is True and marker['retry'] is False)
 expected={'S':S,'C':C,'A':A,'contract_sha256':digest((ROOT/CONTRACT).read_bytes()),'authority_sha256':digest((ROOT/AUTH).read_bytes()),'input_sha256':c['input']['sha256'],'output_path':OUT,'user_instruction':a['user_instruction'],'python_executable_sha256':a['python_executable_sha256']}
 for label,obj in [('result',r),('supervisor',supervisor),('marker',marker)]:
  for key,value in expected.items():check('provenance:'+label+':'+key,obj['provenance'].get(key)==value)
 check('complete_no_failure',r['status']=='PI3_COMPARISON_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT' and r['failure'] is None)
 check('worker_exit0_no_interruption',supervisor['status']=='COMPLETE_WORKER_EXIT' and supervisor['worker_exitcode']==0 and supervisor['interruption'] is None)
 check('all48_records',len(r['comparisons'])==48)
 keys={(j,n,d,t,m) for j in ['G3','G6','G9'] for n in [2,4] for d in ['1/8','1/16'] for t in ['1','4'] for m in ['ordinary','independent_pm']}
 key=lambda item:tuple(item[k] for k in ['split','R','delta','T','comparator'])
 check('unique_complete_grid',{key(x) for x in r['comparisons']}==keys)
 check('no_placeholder',all(x['status']!='NOT_EVALUATED' for x in r['comparisons']))
 i=c['input'];raw_bytes=(ROOT/i['path']).read_bytes()
 check('saved_input_hash_bytes_commit',len(raw_bytes)==i['bytes'] and digest(raw_bytes)==i['sha256'] and git('show',i['commit']+':'+i['path'])==raw_bytes)
 raw=json.loads(raw_bytes);original={key(x):x for x in raw['comparisons']}
 pilo,pihi=map(integers,raw['arithmetic']['pi_interval'])
 check('saved_pi_copied',r['threshold']['saved_pi_interval_rad']==raw['arithmetic']['pi_interval'])
 for label,reference,actual in zip(['lower','upper'],[pilo,pihi],r['threshold']['threshold_interval_rad']):
  n,d=integers(actual);rn,rd=reference
  check('exact_positive_third:'+label,n*(3*rd)==rn*d and d>0)
 independently_checked=0
 for item in r['comparisons']:
  label=':'.join(map(str,key(item)));old=original[key(item)]
  check('no_auto_scientific_GO:'+label,item['scientific_GO'] is None)
  if item['split']=='G9':
   check('G9_no_numeric_gate_or_rescue:'+label,item['status']=='G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED' and item['threshold_test_performed'] is False and item['boundary_interval_rad'] is None and item['scientific_gain_claim']==old['scientific_gain_claim']=='ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE')
   continue
  check('saved_boundary_unmodified:'+label,item['boundary_interval_rad']==old['boundary_interval_rad'])
  lo,hi=map(integers,old['boundary_interval_rad']);ln,ld=lo;un,ud=hi;pn,pd=pilo;qn,qd=pihi
  # Independent of production Fraction/classify: integer-only cross-products.
  satisfied=(3*un*pd < pn*ud)
  not_satisfied=(3*ln*qd >= qn*ld)
  expect='SATISFIED' if satisfied else 'NOT_SATISFIED' if not_satisfied else 'UNRESOLVED'
  check('integer_cross_product_classification:'+label,item['status']==expect and not (satisfied and not_satisfied))
  check('meaningful_direct_shared_favourable_side:'+label,old['continuous_relation_before']=='SHARED_LOWER' and old['continuous_relation_after']=='SHARED_HIGHER' and item['shared_favourable_side']=='ABOVE_SAVED_ROOT')
  check('no_strict_integer_shot_claim:'+label,item['strict_integer_shot_saving']=='NOT_ESTABLISHED' and item['interpretation']=='CONDITIONAL_CONTINUOUS_DESIGN_INTERSECTION_ONLY')
  independently_checked+=1
 check('independently_checked_all32',independently_checked==32)
 actual_counts=dict(Counter(x['status'] for x in r['comparisons']))
 check('summary_count_consistency',all(r['comparison_counts'].get(k)==actual_counts.get(k,0) for k in r['comparison_counts']) and sum(r['comparison_counts'].values())==48)
 check('unchanged_science_and_resources',r['science']==c['science'] and r['resources']==c['resources'])
 check('adopted_cap_policy_unchanged',r['resources']['cpu_time_seconds'] is None and r['resources']['wall_time_seconds'] is None and r['resources']['workers']==1 and r['resources']['gpu'] is False)
 check('aggregate_memory_budget',supervisor['parent_virtual_bytes']+supervisor['child_address_space_cap']==c['resources']['memory_bytes'])
 check('worker_RSS_within_child_AS',r['usage']['maxrss_KiB_linux']*1024<supervisor['child_address_space_cap'])
 check('no_wall_interruption',supervisor['wall_cap_reached'] is False)
 check('conditional_engineering_only',r['evidence']=='CONDITIONAL_ENGINEERING_NOT_CERTIFIED' and r['rpe_success_guaranteed'] is False and r['resource_improvement_guaranteed'] is False)
 for k in ['absolute_shot_counts','native_costs','total_resources','scientific_GO']:check('outside_scope_null:'+k,r[k] is None)
 check('STOP_no_retry',r['mandatory_STOP'] is True and r['retry'] is False and supervisor['mandatory_STOP'] is True and supervisor['retry'] is False)
 for p,sha in c['source_identity']['files'].items():
  b=(ROOT/p).read_bytes();check('SOURCE_unchanged:'+p,digest(b)==sha and git('show',S+':'+p)==b)
 schema=load('artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1/result_schema.json');Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(r);check('saved_result_schema',True)
 for p in (ROOT/OUT).iterdir():check('result_size:'+p.name,p.stat().st_size<=c['resources']['result_bytes'])
 check('no_partial_tmp_or_failure_sidecar',{p.name for p in (ROOT/OUT).iterdir()}=={'result.json','supervisor_result.json','exclusive_launch_marker.json'})
 for p,rec in load(ART+'/frozen_output_identity.json').items():check('raw_unchanged_since_completion:'+p,digest((ROOT/p).read_bytes())==rec['sha256'] and (ROOT/p).stat().st_size==rec['bytes'])
 for p in ['stdout.log','stderr.log','outer_usage.log']:check('log_size:'+p,(ROOT/ART/p).stat().st_size<=c['resources']['log_bytes'])
 check('stdout_complete',(ROOT/ART/'stdout.log').read_text().strip()==r['status'])
 check('outer_exit0','Exit status: 0' in (ROOT/ART/'outer_usage.log').read_text())
 for p in INDEXES:check('old_index_body:'+p,(ROOT/p).read_bytes().endswith(git('show',C+':'+p)))
 old=json.loads(git('show',C+':'+MANIFEST));new=load(MANIFEST)
 check('old_manifest_entries',all(new.get(k)==v for k,v in old.items()))
 check('only_new_result_manifest_entry',set(new)-set(old)=={'track_b_m1s_gamma_pi3_completed_20261017'})
 record={'schema':'m1s_gamma_pi3_saved_result_independent_integer_audit','PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'checks':checks,'source_commit':S,'contract_commit':C,'authority_commit':A,'numerical_SOURCE_imported_or_rerun':False,'comparison_predicates_independently_checked_with_integer_cross_products':independently_checked,'new_pi_or_gamma_roots':0,'independent_H4_or_saved_boundary_generation_validated':False,'coverage':'INHERITED_CONDITIONAL_ENGINEERING_NOT_CERTIFIED','counts':r['comparison_counts'],'by_split':{j:dict(Counter(x['status'] for x in r['comparisons'] if x['split']==j)) for j in ['G3','G6','G9']},'production_numerical_launches':1,'technical_retries':0,'raw_result_sha256':digest((ROOT/OUT/'result.json').read_bytes()),'scientific_GO':None,'mandatory_STOP':True}
 (ROOT/ART/'saved_result_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['PASS','FAIL','comparison_predicates_independently_checked_with_integer_cross_products','counts','by_split']}))
 return int(record['FAIL']>0)


if __name__=='__main__':raise SystemExit(main())
