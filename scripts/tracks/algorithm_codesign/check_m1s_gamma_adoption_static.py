"""Formal contract metadata/hash/closed-guard audit. No scientific evaluation."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1 import control,spec
BASE='fcd1c464fae48743e829fbe765a38141304f879d'
ART='artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1'
CANDIDATE=spec.ART+'/execution_contract_candidate.json'
ADOPTED=ART+'/execution_contract_adopted.json'
DOC='docs/tracks/algorithm_codesign/m1s_v1_gamma_formally_adopted_contract_20261017.md'
NOTE='docs/research/研究ノート/2026-10-17_track_b_M1S_gamma_formal_adoption.md'
CHECKER='scripts/tracks/algorithm_codesign/check_m1s_gamma_adoption_static.py'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','docs/research/研究概要・現状.md']
MANIFEST='artifacts/validation_manifest.json'
MKEY='track_b_m1s_v1_gamma_formal_adoption_20261017'
SCIENCE={'accuracy_semantics':'HIGH_PROBABILITY_CIRCULAR_PHASE_ERROR','confidence_scope':'PER_METHOD_ROW','statistical_policy':'FRESH_IID_HOEFFDING_EQUAL_AXES','coverage_acceptance':'SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY','comparison_policy':'GAMMA_CONTINUOUS_SUFFICIENT_FACTOR_WITH_CEILING_CAVEAT','materiality_policy':'DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO','alpha_total':'0.05','arithmetic_bits':128}


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))


def main():
 checks=[]
 def check(name,ok):checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
 def load(p):return control.read_json((ROOT/p).read_bytes())
 c=load(ADOPTED);old=load(CANDIDATE);record=load(ART+'/user_adoption_record.json');ready=load(ART+'/execution_readiness.json')
 control.validate_contract(c,require_adopted=True)
 check('frozen_production_validator_accepts_complete_adopted_contract',True)
 expected=copy.deepcopy(old)
 expected.update(status='ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL',adopted=True,coverage_model_adopted=True)
 expected['science']=SCIENCE
 expected['resources'].update(cpu_time_seconds=160,wall_time_seconds=320)
 check('only_user_adopted_changes',control.same(c,expected))
 check('candidate_original_bytes',git('show',BASE+':'+CANDIDATE)==(ROOT/CANDIDATE).read_bytes())
 for k,v in SCIENCE.items():check('adopted_science:'+k,control.same(c['science'][k],v))
 for k,v in [('cpu_time_seconds',160),('wall_time_seconds',320)]:check('adopted_resource:'+k,type(c['resources'][k]) is int and c['resources'][k]==v)
 for k in set(old)-{'status','adopted','coverage_model_adopted','science','resources'}:
  check('immutable_contract_field:'+k,control.same(old[k],c[k]))
 check('structural_caps_unchanged',all(control.same(v,c['resources'][k]) for k,v in old['resources'].items() if k not in ['cpu_time_seconds','wall_time_seconds']))
 check('contract_grants_no_execution',c['contract_grants_execution'] is False)
 check('one_shot_no_retry_STOP',c['one_shot'] is True and c['retry'] is False and c['mandatory_STOP'] is True)
 check('direction_and_full_grid_preserved',control.same(c['grid'],old['grid']) and control.same(c['direction'],old['direction']))
 check('no_fixed_gamma_or_numeric_shot_outputs',c['gamma_grid'] is None and c['absolute_shot_counts'] is None)
 check('adoption_record_hash_matches_contract',record['adopted_contract_sha256']==hashlib.sha256((ROOT/ADOPTED).read_bytes()).hexdigest())
 check('adoption_record_references_correct_candidate',record['candidate_commit']==BASE and record['candidate_sha256']==hashlib.sha256((ROOT/CANDIDATE).read_bytes()).hexdigest())
 check('user_explicitly_adopted_and_denied_calculation','正式採択します' in record['user_instruction_verbatim'] and '科学計算は承認しません' in record['user_instruction_verbatim'])
 check('context_traceable','72行同時保証ではない' in record['context_selected_conditions'])
 check('coverage_acceptance_not_platform_certificate',record['coverage_adoption_meaning']=='USER_ACCEPTS_CONDITIONAL_ENGINEERING_MODEL_NOT_PLATFORM_CERTIFICATION' and ready['platform_certificate'] is False)
 for k in ['scientific_execution_authorized','run_ready','contract_grants_execution']:
  check('readiness_closed:'+k,ready[k] is False)
 check('no_authority_commit',ready['execution_authority_commit'] is None)
 check('readiness_hash',ready['contract_sha256']==record['adopted_contract_sha256'])
 check('adopted_science_record',control.same(record['selected_science'],c['science']))
 check('adopted_resource_record',control.same(record['selected_resources'],c['resources']))
 # Metadata-only refusal: contract is complete, but missing A must stop before
 # git access, scientific input paths, or importing any arithmetic module.
 old_inside=control.inside
 def guarded_inside(root,path,prefix=None):
  if path!=ADOPTED:raise AssertionError('scientific/authority path reached without A')
  return old_inside(root,path,prefix)
 with patch.object(control,'inside',side_effect=guarded_inside),patch.object(control,'git',side_effect=AssertionError('git reached without A')):
  try:control.authorize(ROOT,ADOPTED,None,None)
  except control.ExecutionClosed:check('complete_contract_without_A_refused_before_input',True)
  else:check('complete_contract_without_A_refused_before_input',False)
 try:control.require_permit(None)
 except control.ExecutionClosed:check('no_permit_without_authority',True)
 else:check('no_permit_without_authority',False)
 for n in ['analysis','bounds','boundary','supervisor']:
  check('science_module_not_imported:'+n,'trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1.'+n not in sys.modules)
 S=c['source_identity']['commit']
 check('SOURCE_commit_unchanged',S=='f2a5611ca9fc7461c64f2deb74a1f624ac8aa7c5')
 check('SOURCE_9file_inventory',set(c['source_identity']['files'])==set(spec.SOURCE_FILES) and len(spec.SOURCE_FILES)==9)
 for p,sha in c['source_identity']['files'].items():
  raw=(ROOT/p).read_bytes();check('SOURCE_sha:'+p,hashlib.sha256(raw).hexdigest()==sha)
  check('SOURCE_S_bytes:'+p,git('show',S+':'+p)==raw)
  if p.endswith('.py'):check('SOURCE_AST:'+p,isinstance(ast.parse(raw),ast.Module))
 check('SOURCE_precedes_candidate',subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',S,BASE],capture_output=True).returncode==0 and S!=BASE)
 i=c['input'];raw=(ROOT/i['path']).read_bytes()
 check('input_hash_and_bytes',len(raw)==i['bytes'] and hashlib.sha256(raw).hexdigest()==i['sha256'])
 check('input_original_commit',git('show',i['commit']+':'+i['path'])==raw)
 common='docs/research/research_project_instructions_unified_2026-10-16.md'
 check('common_instruction_fixed',hashlib.sha256((ROOT/common).read_bytes()).hexdigest()=='8323be3f65856adf0f4d095d8e68546db3326822023c20e101ba5f596605ebfa')
 for p in INDEXES:check('old_index_body:'+p,(ROOT/p).read_bytes().endswith(git('show',BASE+':'+p)))
 before=control.read_json(git('show',BASE+':'+MANIFEST));after=load(MANIFEST)
 check('old_manifest_entries_preserved',all(after.get(k)==v for k,v in before.items()))
 check('only_new_manifest_entry',set(after)-set(before)=={MKEY})
 for p in [DOC,NOTE]:
  text=(ROOT/p).read_text();check('UTF8_LF:'+p,text.endswith('\n') and '\r' not in text)
  check('no_trailing_space:'+p,all(x==x.rstrip() for x in text.splitlines()))
 unchanged=0;changed=[]
 for item in git('ls-tree','-r','-z',BASE).split(b'\0'):
  if not item:continue
  fields,name=item.split(b'\t',1);p=name.decode()
  if p in INDEXES+[MANIFEST]:continue
  f=ROOT/p
  if not f.is_file():changed.append(p);continue
  b=f.read_bytes()
  if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=fields.split()[2].decode():changed.append(p)
  else:unchanged+=1
 check('all_old_SOURCE_results_contracts_grants_ledgers_markers',not changed)
 paths=set(filter(None,git('diff','--name-only','-z',BASE).decode().split('\0')))
 paths.update(filter(None,git('ls-files','--others','--exclude-standard','-z').decode().split('\0')))
 check('only_adoption_documentation_paths',all(p in INDEXES+[MANIFEST,DOC,NOTE,CHECKER] or p.startswith(ART+'/') for p in paths))
 check('new_execution_authority_or_marker_absent',not any(p.startswith((spec.AUTH_PREFIX,spec.OUTPUT_PREFIX)) for p in paths))
 result={'schema':'m1s_gamma_adoption_static_validation','checks':checks,'PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'inherited_nonindex_files_preserved':unchanged,'unexpected_old_changes':changed,'scope':'CONTRACT_SCHEMA_METADATA_GUARD_AST_HASH_ONLY','new_scientific_calculations':0,'saved_H4_reanalysis':0,'H4_gamma_boundaries':0,'shot_forecasts':0,'numerical_correctness_tests':0,'execution_authorizations_issued':0,'grants_issued':0,'markers_issued':0,'mandatory_STOP':True}
 (ROOT/ART/'static_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['PASS','FAIL','inherited_nonindex_files_preserved','scope']}))
 return int(result['FAIL']>0)


if __name__=='__main__':raise SystemExit(main())
