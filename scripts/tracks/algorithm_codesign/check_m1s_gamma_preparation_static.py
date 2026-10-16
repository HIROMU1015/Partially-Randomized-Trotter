"""AST/schema/hash audit only; never decode saved scientific result numbers."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1 import control,spec
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md']
MANIFEST='artifacts/validation_manifest.json'
MKEY='track_b_m1s_v1_gamma_preparation_20261017'
DOCS=['docs/tracks/algorithm_codesign/m1s_v1_gamma_dependent_contract_candidate_20261017.md','docs/tracks/algorithm_codesign/m1s_v1_gamma_precision_resource_and_handoff_20261017.md','docs/research/研究ノート/2026-10-17_track_b_M1S_gamma_preparation.md']
REVIEW='docs/research/m1s_v1_scientific_conditions_independent_review_20261017.md'
NEW=set(DOCS+[REVIEW,'scripts/tracks/algorithm_codesign/run_m1s_saved_signal_gamma_v1.py','scripts/tracks/algorithm_codesign/check_m1s_gamma_preparation_static.py','scripts/tracks/algorithm_codesign/benchmark_m1s_gamma_offdomain.py','tests/tracks/algorithm_codesign/test_m1s_gamma_preparation.py'])


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))


def main():
 checks=[]
 def check(name,ok):checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
 def load(p):return control.read_json((ROOT/p).read_bytes())
 c=load(spec.ART+'/execution_contract_candidate.json');control.validate_contract(c)
 for flag in ['adopted','coverage_model_adopted','contract_grants_execution']:
  check('closed:'+flag,c[flag] is False)
 check('direction_change_only_adopted',c['research_direction_adopted'] is True)
 check('all_science_null',all(v is None for v in c['science'].values()))
 check('no_single_gamma_or_grid', 'gamma_rad' not in c['science'] and c['gamma_grid'] is None)
 check('row_confidence_recommendation_not_default',c['policy_recommendations_not_adoption']['confidence_scope']=='PER_METHOD_ROW' and c['science']['confidence_scope'] is None)
 check('no_GO_absolute_shots_native',all(c[k] is None for k in ['scientific_GO','absolute_shot_counts','native_costs','total_resources']))
 check('time_caps_not_adopted',c['resources']['cpu_time_seconds'] is None and c['resources']['wall_time_seconds'] is None)
 try:control.authorize(ROOT,spec.ART+'/execution_contract_candidate.json',None,None)
 except control.ExecutionClosed:check('candidate_closed_before_scientific_entry',True)
 else:check('candidate_closed_before_scientific_entry',False)
 check('analysis_not_imported','trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1.analysis' not in sys.modules)
 check('source_frozen',c['source_identity'] is not None)
 if c['source_identity']:
  S=c['source_identity']['commit']
  for p,sha in c['source_identity']['files'].items():
   raw=(ROOT/p).read_bytes();check('SOURCE_sha:'+p,hashlib.sha256(raw).hexdigest()==sha)
   check('SOURCE_commit_bytes:'+p,git('show',S+':'+p)==raw)
  check('source_descends_from_base',subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',spec.BASE,S],capture_output=True).returncode==0)
 for p in spec.SOURCE_FILES:
  if not p.endswith('.py'):continue
  tree=ast.parse((ROOT/p).read_text());check('AST:'+p,isinstance(tree,ast.Module))
  imports=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
   if isinstance(n,ast.ImportFrom):imports.append(n.module or '')
  check('no_science_library_or_old_helper:'+p,not any(any(x in y for x in ['numpy','scipy','mpmath','qiskit','m1s_saved_signal_v1','m1s_finite','trotterlib']) for y in imports))
  check('no_eval_or_exec:'+p,not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['eval','exec','compile'] for n in ast.walk(tree)))
 engine=(ROOT/spec.PACKAGE/'analysis.py').read_text();bound=(ROOT/spec.PACKAGE/'boundary.py').read_text();ctrl=(ROOT/spec.PACKAGE/'control.py').read_text()
 for token in ['REVIEW_TRADEOFF_DIRECT','REVIEW_TRADEOFF_SWAPPED','NOT_APPLICABLE_EQUAL_BIAS','NOT_APPLICABLE_SHARED_DOMINANCE','NO_COMMON_PHASE_DOMAIN','ARITHMETIC_UNRESOLVED','COEFFICIENT_SIGN_UNRESOLVED','ROOT_EXISTS_LOCATION_UNRESOLVED','NOT_INFERRED_FROM_CONTINUOUS_ORDER']:
  check('boundary_classification:'+token,token in bound)
 for token in ['TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H','ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE','SAVED_ZERO_CONTROL_NOT_REANALYZED',"'model_drift_added_again': False"]:
  check('evidence_semantics:'+token,token in engine)
 for token in ['time caps not adopted','signal.ITIMER_REAL','resource.RLIMIT_CPU',"output.mkdir()",'supervisor_result.json','child_address_space_cap']:
  check('guard_resource:'+token,token in ctrl)
 check('engine_import_after_guard_and_limits',ctrl.index('from .analysis import evaluate')>ctrl.index('resource.setrlimit(resource.RLIMIT_CPU'))
 sch=load(spec.ART+'/result_schema.json');Draft202012Validator.check_schema(sch)
 check('result_schema_metaschema',True)
 check('result_all108',sch['properties']['rows']['minItems']==sch['properties']['rows']['maxItems']==108)
 check('result_all48',sch['properties']['comparisons']['minItems']==sch['properties']['comparisons']['maxItems']==48)
 for r in load(spec.ART+'/input_identity.json')['files']:
  raw=(ROOT/r['path']).read_bytes();check('input_hash:'+r['path'],len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'])
  if r['commit']:check('input_commit:'+r['path'],git('show',r['commit']+':'+r['path'])==raw)
 check('review_original_sha',hashlib.sha256((ROOT/REVIEW).read_bytes()).hexdigest()=='a0d4a884d557e7f2337de8a88f31182b1176b7ab0d7c0bb3f2451a4712b8a575')
 probe=load(spec.ART+'/offdomain_resource_probe.json')
 check('probe_offdomain_only',probe['scope']=='ARTIFICIAL_SCALAR_KERNELS_ONLY' and probe['science_inputs_opened']==0)
 tests=load(spec.ART+'/offdomain_tests.json')
 check('technical_tests_PASS',tests['tests_run']==34 and tests['FAIL']==0 and tests['ERROR']==0)
 for p in INDEXES:check('old_index_body:'+p,(ROOT/p).read_bytes().endswith(git('show',spec.BASE+':'+p)))
 old=control.read_json(git('show',spec.BASE+':'+MANIFEST));new=load(MANIFEST)
 check('old_manifest_entries',all(new.get(k)==v for k,v in old.items()))
 check('only_new_manifest_entry',set(new)-set(old)=={MKEY})
 for p in DOCS:
  text=(ROOT/p).read_text();check('UTF8_LF:'+p,text.endswith('\n') and '\r' not in text)
  check('no_trailing_space:'+p,all(x==x.rstrip() for x in text.splitlines()))
 changed=[];preserved=0
 for item in git('ls-tree','-r','-z',spec.BASE).split(b'\0'):
  if not item:continue
  fields,path=item.split(b'\t',1);p=path.decode();blob=fields.split()[2].decode()
  if p in INDEXES+[MANIFEST]:continue
  f=ROOT/p
  if not f.is_file():changed.append(p);continue
  raw=f.read_bytes();sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
  if sha!=blob:changed.append(p)
  else:preserved+=1
 check('all_old_SOURCE_results_auth_grants_ledgers_markers',not changed)
 paths=set(filter(None,git('diff','--name-only','-z',spec.BASE).decode().split('\0')))
 paths.update(filter(None,git('ls-files','--others','--exclude-standard','-z').decode().split('\0')))
 check('only_authorized_paths',all(p in INDEXES+[MANIFEST] or p in NEW or p.startswith((spec.PACKAGE+'/',spec.ART+'/')) for p in paths))
 check('new_execution_authority_marker_result_absent',not any(p.startswith((spec.AUTH_PREFIX,spec.OUTPUT_PREFIX)) for p in paths))
 result={'schema':'m1s_gamma_static_validation','checks':checks,'PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'inherited_nonindex_files_preserved':preserved,'unexpected_old_changes':changed,'scope':'AST_METADATA_SCHEMA_HASH_ONLY_NO_SAVED_SCIENTIFIC_DECODING','saved_H4_reanalysis':0,'H4_gamma_boundaries':0,'shot_forecasts':0,'new_scientific_calculations':0,'mandatory_STOP':True}
 (ROOT/spec.ART/'static_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['PASS','FAIL','inherited_nonindex_files_preserved','scope']}))
 return int(result['FAIL']>0)


if __name__=='__main__':raise SystemExit(main())
