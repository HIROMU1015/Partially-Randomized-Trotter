"""Audit saved completion and provenance only; never invoke the numerical engine."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[3]
ART='artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02'
OUT='artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02'
AUTH='artifacts/track_b_m1s_saved_signal_gamma_v2_execution_authorization/2026-10-17/attempt_02/authorization.json'
CONTRACT=ART+'/execution_contract_adopted.json'
S='677dd6957e0e1913b70ca5f999458835ae8a2e23'
C='2cb5160d9051a6568b8acc0cfb29fc169bdfca09'
A='161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c'
BASE='bc62ce6fb45ce3c67404ac8ae693953c588effac'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md']
MANIFEST='artifacts/validation_manifest.json'
MKEY='track_b_m1s_v1_gamma_completed_20261017'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def digest(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads((ROOT/p).read_bytes())


def main():
 checks=[]
 def check(name,ok):checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
 c,a,r,sup,marker=[load(p) for p in [CONTRACT,AUTH,OUT+'/result.json',OUT+'/supervisor_result.json',OUT+'/exclusive_launch_marker.json']]
 expected={'S':S,'C':C,'A':A,'contract_sha256':digest((ROOT/CONTRACT).read_bytes()),'authority_sha256':digest((ROOT/AUTH).read_bytes()),'input_sha256':c['input']['sha256'],'output_path':OUT,'user_instruction':a['user_instruction'],'python_executable_sha256':a['python_executable_sha256']}
 check('authority_only_direct_child_C2',git('rev-parse',A+'^').decode().strip()==C and git('diff-tree','--no-commit-id','--name-only','-r',A).decode().splitlines()==[AUTH])
 check('contract_frozen_in_C2',git('show',C+':'+CONTRACT)==(ROOT/CONTRACT).read_bytes())
 check('authority_frozen_in_A2',git('show',A+':'+AUTH)==(ROOT/AUTH).read_bytes())
 check('new_authority_not_old_consumed_authority',A!='b0b488364205fdf0f8082bbff2a0a585d3ef18f7')
 check('latest_completion_instruction_recorded','しっかり完了するまで行ってよい' in a['user_instruction'])
 for obj,label in [(r,'result'),(sup,'supervisor'),(marker,'marker')]:
  for k,v in expected.items():check('provenance:'+label+':'+k,obj['provenance'].get(k)==v)
 check('completed_without_failure',r['status']=='GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT' and r['failure'] is None)
 check('worker_exit_success',sup['status']=='COMPLETE_WORKER_EXIT' and sup['worker_exitcode']==0 and sup['interruption'] is None)
 check('marker_consumed_once_no_authority_reuse',marker['consumed'] is True and marker['retry'] is False)
 check('no_retry_and_mandatory_STOP',all(x['retry'] is False for x in [r,sup,marker]) and r['mandatory_STOP'] is True and sup['mandatory_STOP'] is True)
 check('all108_rows',len(r['rows'])==108)
 keys={(j,n,d,t,m) for j in ['G3','G6','G9'] for n in [2,4] for d in ['1/8','1/16'] for t in ['0','1','4'] for m in ['shared_pm','ordinary','independent_pm']}
 check('complete_unique_method_grid',{(x['split'],x['R'],x['delta'],x['T'],x['method']) for x in r['rows']}==keys)
 ck={(j,n,d,t,m) for j in ['G3','G6','G9'] for n in [2,4] for d in ['1/8','1/16'] for t in ['1','4'] for m in ['ordinary','independent_pm']}
 check('complete_unique48_contrasts',len(r['comparisons'])==48 and {(x['split'],x['R'],x['delta'],x['T'],x['comparator']) for x in r['comparisons']}==ck)
 check('no_remaining_placeholder',all(x['status']!='NOT_EVALUATED' for x in r['rows']+r['comparisons']))
 for x in r['rows']:
  label=':'.join(str(x[k]) for k in ['split','R','delta','T','method'])
  if x['T']=='0':check('zero_control_not_reanalyzed:'+label,x['status']=='SAVED_ZERO_CONTROL_NOT_REANALYZED');continue
  check('raw_H_bias_not_double_counted:'+label,x.get('bounds',{}).get('model_drift_added_again') is False and x.get('bounds',{}).get('bias_definition')=='min(pi,unclipped_bias_rad)')
  check('no_absolute_shot_count:'+label,x.get('N_axis') is None and x.get('N_total') is None)
  if x['status']=='READY_FOR_GAMMA_DOMAIN':
   check('positive_lower_visibility_and_bias:'+label,0<F(x['bounds']['A_lower'])<=1 and F(x['bounds']['unclipped_bias_rad'])>=0)
 for x in r['comparisons']:
  label=':'.join(str(x[k]) for k in ['split','R','delta','T','comparator'])
  check('descriptive_only:'+label,x['scientific_GO'] is None and x['native_costs'] is None)
  check('ceiling_caveat:'+label,x.get('strict_integer_saving')=='NOT_INFERRED_FROM_CONTINUOUS_ORDER')
  if x['split']=='G9':check('G9_no_relative_rescue:'+label,x['scientific_gain_claim']=='ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE')
  if x['status']=='UNIQUE_TRADEOFF_BOUNDARY':
   lo,hi=map(F,x['boundary_interval_rad']);domain=F(x['common_domain']['lower_open_rad']);pi_lo=F(r['arithmetic']['pi_interval'][0])
   check('root_interval_strictly_inside_domain:'+label,domain<lo<=hi<pi_lo/2)
   check('root_band_not_ranked:'+label,x['relation_rule']=='before root lower endpoint / after root upper endpoint; enclosure band unresolved')
 for key in ['gamma_grid','absolute_shot_counts','scientific_GO','native_costs','total_resources']:check('outside_scope_null:'+key,r[key] is None)
 check('conditional_engineering_only',r['evidence']=='CONDITIONAL_ENGINEERING_NOT_CERTIFIED' and r['physical_shot_guarantee'] is False)
 check('same_science_and_resources',r['science']==c['science'] and r['resources']==c['resources'])
 old=load('artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/execution_contract_adopted.json')
 check('only_contract_source_identity_changed',all(c[k]==old[k] for k in old if k!='source_identity'))
 check('p128_and_per_row_alpha005',r['arithmetic']['bits']==128 and r['science']['alpha_total']=='0.05' and r['science']['confidence_scope']=='PER_METHOD_ROW')
 check('CPU_cap',r['usage']['cpu_user_seconds']+r['usage']['cpu_system_seconds']<c['resources']['cpu_time_seconds'])
 check('wall_cap',sup['wall_cap_reached'] is False and sup['elapsed_wall_seconds']<c['resources']['wall_time_seconds'])
 check('aggregate_memory_budget',sup['parent_virtual_bytes']+sup['child_address_space_cap']==c['resources']['memory_bytes'])
 check('RSS_within_cap',r['usage']['maxrss_KiB_linux']*1024<sup['child_address_space_cap'])
 check('bound_steps_within_cap',r['arithmetic']['bound_steps']<=c['resources']['max_bound_steps'])
 check('no_partial_or_failure_sidecar',{x.name for x in (ROOT/OUT).iterdir()}=={'result.json','supervisor_result.json','exclusive_launch_marker.json'})
 for p in (ROOT/OUT).iterdir():check('output_size:'+p.name,p.stat().st_size<c['resources']['result_bytes'])
 for p in ['stdout.log','stderr.log','outer_usage.log']:check('log_size:'+p,(ROOT/ART/p).stat().st_size<c['resources']['log_bytes'])
 check('outer_exit0','Exit status: 0' in (ROOT/ART/'outer_usage.log').read_text())
 check('stdout_complete',(ROOT/ART/'stdout.log').read_text().strip()==r['status'])
 for p,h in c['source_identity']['files'].items():check('SOURCE_frozen:'+p,digest((ROOT/p).read_bytes())==h and git('show',S+':'+p)==(ROOT/p).read_bytes())
 i=c['input'];b=(ROOT/i['path']).read_bytes();check('original_saved_input_preserved',len(b)==i['bytes'] and digest(b)==i['sha256'] and git('show',i['commit']+':'+i['path'])==b)
 schema=load(ART+'/result_schema.json');Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(r);check('result_schema',True)
 hashes=load(ART+'/frozen_output_identity.json')
 for p,rec in hashes.items():check('raw_unchanged_since_completion:'+p,digest((ROOT/p).read_bytes())==rec['sha256'] and (ROOT/p).stat().st_size==rec['bytes'])
 for p in INDEXES:check('old_index_body_preserved:'+p,(ROOT/p).read_bytes().endswith(git('show',BASE+':'+p)))
 old_manifest=json.loads(git('show',BASE+':'+MANIFEST));new_manifest=load(MANIFEST)
 check('old_manifest_entries_preserved',all(new_manifest.get(k)==v for k,v in old_manifest.items()))
 check('only_completed_manifest_entry_added',set(new_manifest)-set(old_manifest)=={MKEY})
 check('numerical_engine_never_imported',not any('m1s_saved_signal_gamma_v2.analysis'==x.rsplit('algorithm_codesign.',1)[-1] for x in sys.modules))
 record={'schema':'m1s_gamma_saved_completion_audit','PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'checks':checks,'S2':S,'C2':C,'A2':A,'scope':'SAVED_OUTPUT_SCHEMA_PROVENANCE_RESOURCE_AND_CONTRACT_INVARIANTS_ONLY_NO_ENGINE_REEXECUTION','new_engine_evaluations_in_audit':0,'independent_H4_boundary_numerical_validation':False,'row_status_counts':dict(Counter(x['status'] for x in r['rows'])),'contrast_status_counts_by_split':{j:dict(Counter(x['status'] for x in r['comparisons'] if x['split']==j)) for j in ['G3','G6','G9']},'raw_result_sha256':digest((ROOT/OUT/'result.json').read_bytes()),'scientific_GO':None,'mandatory_STOP':True}
 (ROOT/ART/'saved_completion_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:record[k] for k in ['PASS','FAIL','row_status_counts','contrast_status_counts_by_split']}))
 return int(record['FAIL']>0)


if __name__=='__main__':raise SystemExit(main())
