"""Saved failure/metadata audit only. Never run or repair the scientific worker."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[3]
S='f2a5611ca9fc7461c64f2deb74a1f624ac8aa7c5'
C='9aafa8b754aa08e38f92cd06e503a17904908f5e'
A='b0b488364205fdf0f8082bbff2a0a585d3ef18f7'
ART='artifacts/track_b_m1s_v1_gamma_execution_audit/2026-10-17/v1'
OUT='artifacts/track_b_m1s_saved_signal_gamma_v1_result/2026-10-17/v1_one_shot'
AUTH='artifacts/track_b_m1s_saved_signal_gamma_v1_execution_authorization/2026-10-17/v1/authorization.json'
CONTRACT='artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/execution_contract_adopted.json'
DOC='docs/tracks/algorithm_codesign/m1s_v1_gamma_one_shot_technical_stop_20261017.md'
NOTE='docs/research/研究ノート/2026-10-17_track_b_M1S_gamma_one_shot_stop.md'
CHECKER='scripts/tracks/algorithm_codesign/audit_m1s_gamma_v1_saved_stop.py'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','docs/research/研究概要・現状.md']
MANIFEST='artifacts/validation_manifest.json'
MKEY='track_b_m1s_v1_gamma_one_shot_stop_20261017'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def digest(b):return hashlib.sha256(b).hexdigest()


def main():
 checks=[]
 def check(name,ok):checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
 def load(p):return json.loads((ROOT/p).read_bytes())
 c=load(CONTRACT);a=load(AUTH);r=load(OUT+'/result.json');sup=load(OUT+'/supervisor_result.json');mark=load(OUT+'/exclusive_launch_marker.json')
 expected={'S':S,'C':C,'A':A,'contract_sha256':digest((ROOT/CONTRACT).read_bytes()),'authority_sha256':digest((ROOT/AUTH).read_bytes()),'input_sha256':c['input']['sha256'],'output_path':OUT,'user_instruction':a['user_instruction'],'python_executable_sha256':a['python_executable_sha256']}
 check('authority_only_direct_child_of_C',git('rev-parse',A+'^').decode().strip()==C and git('diff-tree','--no-commit-id','--name-only','-r',A).decode().splitlines()==[AUTH])
 check('frozen_authority_bytes',git('show',A+':'+AUTH)==(ROOT/AUTH).read_bytes())
 check('frozen_contract_bytes',git('show',C+':'+CONTRACT)==(ROOT/CONTRACT).read_bytes())
 check('authorized_one_shot_no_retry',a['scientific_execution_authorized'] is True and a['one_shot'] is True and a['retry'] is False)
 for k,v in expected.items():
  for obj,label in [(r,'result'),(sup,'supervisor'),(mark,'marker')]:check('provenance:'+label+':'+k,obj['provenance'].get(k)==v)
 check('exclusive_marker_consumed',mark['consumed'] is True and mark['retry'] is False)
 check('failure_classification',r['status']=='TECHNICAL_STOP' and sup['status']=='TECHNICAL_STOP' and sup['worker_exitcode']==2)
 check('runtime_API_failure_saved',r['failure']['type']=='AttributeError' and "set_int_max_str_digits" in r['failure']['message'])
 check('traceback_preserved','control.py' in r['failure']['traceback'] and 'line 278' in r['failure']['traceback'])
 check('no_arithmetic_started','arithmetic' not in r)
 check('all108_rows_uncomputed',len(r['rows'])==108 and all(x['status']=='NOT_EVALUATED' for x in r['rows']))
 check('all48_comparisons_uncomputed',len(r['comparisons'])==48 and all(x['status']=='NOT_EVALUATED' for x in r['comparisons']))
 expected_keys={(j,n,d,t,m) for j in ['G3','G6','G9'] for n in [2,4] for d in ['1/8','1/16'] for t in ['0','1','4'] for m in ['shared_pm','ordinary','independent_pm']}
 check('full_saved_grid_preserved',{(x['split'],x['R'],x['delta'],x['T'],x['method']) for x in r['rows']}==expected_keys)
 for k in ['absolute_shot_counts','gamma_grid','scientific_GO','native_costs','total_resources']:check('unevaluated_null:'+k,r[k] is None)
 check('adopted_science_unchanged',r['science']==c['science'])
 check('adopted_resources_unchanged',r['resources']==c['resources'])
 check('mandatory_STOP_and_no_retry',r['mandatory_STOP'] is True and sup['mandatory_STOP'] is True and r['retry'] is False and sup['retry'] is False)
 schema=load('artifacts/track_b_m1s_v1_gamma_preparation/2026-10-17/v1/result_schema.json')
 Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(r)
 check('saved_result_schema',True)
 check('CPU_cap_not_reached',r['usage']['cpu_user_seconds']+r['usage']['cpu_system_seconds']<c['resources']['cpu_time_seconds'])
 check('wall_cap_not_reached',sup['wall_cap_reached'] is False and sup['elapsed_wall_seconds']<c['resources']['wall_time_seconds'])
 check('aggregate_AS_budget',sup['parent_virtual_bytes']+sup['child_address_space_cap']==c['resources']['memory_bytes'])
 check('worker_RSS_below_child_AS_cap',r['usage']['maxrss_KiB_linux']*1024<sup['child_address_space_cap'])
 for p in (ROOT/OUT).iterdir():check('result_file_bytes_cap:'+p.name,p.stat().st_size<=c['resources']['result_bytes'])
 for n in ['stdout.log','stderr.log','outer_usage.log']:check('log_bytes_cap:'+n,(ROOT/ART/n).stat().st_size<=c['resources']['log_bytes'])
 check('stdout_records_STOP',(ROOT/ART/'stdout.log').read_text().strip()=='TECHNICAL_STOP')
 check('outer_process_exit2','Exit status: 2' in (ROOT/ART/'outer_usage.log').read_text())
 api=load(ART+'/runtime_api_failure_analysis.json')
 check('same_pinned_runtime_lacks_API',api['has_set_int_max_str_digits'] is False and api['python_executable_sha256']==a['python_executable_sha256'])
 check('failure_precedes_science_import',api['API_call_precedes_engine_import'] is True and api['unconditional_API_call_lines']==[278] and api['scientific_engine_import_lines']==[279])
 for rec in load(ART+'/frozen_output_identity.json')['files']:
  b=(ROOT/rec['path']).read_bytes();check('raw_failure_bytes_preserved:'+rec['path'],len(b)==rec['bytes'] and digest(b)==rec['sha256'])
 for p,h in c['source_identity']['files'].items():
  b=(ROOT/p).read_bytes();check('SOURCE_hash:'+p,digest(b)==h)
  check('SOURCE_original_S:'+p,git('show',S+':'+p)==b)
 i=c['input'];b=(ROOT/i['path']).read_bytes()
 check('saved_input_hash',len(b)==i['bytes'] and digest(b)==i['sha256'])
 check('saved_input_original_commit',git('show',i['commit']+':'+i['path'])==b)
 for p in INDEXES:check('old_index_body:'+p,(ROOT/p).read_bytes().endswith(git('show',C+':'+p)))
 old=json.loads(git('show',C+':'+MANIFEST));new=load(MANIFEST)
 check('old_manifest_entries',all(new.get(k)==v for k,v in old.items()))
 check('only_new_manifest_entry',set(new)-set(old)=={MKEY})
 for p in [DOC,NOTE]:
  text=(ROOT/p).read_text();check('UTF8_LF:'+p,text.endswith('\n') and '\r' not in text)
  check('no_trailing_space:'+p,all(x==x.rstrip() for x in text.splitlines()))
 unchanged=0;changed=[]
 for item in git('ls-tree','-r','-z',C).split(b'\0'):
  if not item:continue
  fields,path=item.split(b'\t',1);p=path.decode()
  if p in INDEXES+[MANIFEST]:continue
  f=ROOT/p
  if not f.is_file():changed.append(p);continue
  b=f.read_bytes()
  if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=fields.split()[2].decode():changed.append(p)
  else:unchanged+=1
 check('all_old_SOURCE_results_contracts_grants_ledgers_markers',not changed)
 paths=set(filter(None,git('diff','--name-only','-z',C).decode().split('\0')))
 paths.update(filter(None,git('ls-files','--others','--exclude-standard','-z').decode().split('\0')))
 check('only_result_and_audit_changes',all(p in INDEXES+[MANIFEST,DOC,NOTE,CHECKER,AUTH] or p.startswith((OUT+'/',ART+'/')) for p in paths))
 result={'schema':'m1s_gamma_one_shot_saved_failure_audit','checks':checks,'PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'inherited_nonindex_files_preserved':unchanged,'unexpected_old_changes':changed,'S':S,'C':C,'A':A,'raw_result_sha256':digest((ROOT/OUT/'result.json').read_bytes()),'one_shot_launches':1,'one_shot_consumed':True,'retry_count':0,'new_scientific_calculations':0,'saved_science_input_decoded':False,'H4_gamma_boundaries':0,'shot_forecasts':0,'rows_evaluated':0,'comparisons_evaluated':0,'row_status_counts':dict(Counter(x['status'] for x in r['rows'])),'comparison_status_counts':dict(Counter(x['status'] for x in r['comparisons'])),'numerical_correctness_established':False,'scientific_GO':None,'mandatory_STOP':True}
 (ROOT/ART/'saved_failure_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['PASS','FAIL','inherited_nonindex_files_preserved','one_shot_launches','one_shot_consumed','retry_count','rows_evaluated','comparisons_evaluated']}))
 return int(result['FAIL']>0)


if __name__=='__main__':raise SystemExit(main())
