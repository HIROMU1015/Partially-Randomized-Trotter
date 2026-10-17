"""Static/metadata adoption audit only. Never decode or compare saved boundaries."""
import ast
import copy
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
from trottertracks.algorithm_codesign.m1s_gamma_pi3_v1 import control,spec
BASE='e3828965dec292e9c66f6e7d8480e46059701183'
ART='artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1'
CONTRACT=ART+'/execution_contract_adopted.json'
CANDIDATE=spec.ART+'/execution_contract_candidate.json'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md']
MANIFEST='artifacts/validation_manifest.json'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
 checks=[]
 def check(name,ok):checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
 def load(p):return control.read_json((ROOT/p).read_bytes())
 c=load(CONTRACT);prior=load(CANDIDATE);record=load(ART+'/user_adoption_record.json');ready=load(ART+'/execution_readiness.json')
 control.validate_contract(c,True);check('adopted_contract_supported_by_frozen_source',True)
 changed={k for k in c if not control.same(c[k],prior[k])}
 check('only_status_and_three_adoption_flags_changed',changed=={'status','decision_gate_adopted','adopted','coverage_model_adopted'})
 for k in c:
  if k not in changed:check('candidate_field_unchanged:'+k,control.same(c[k],prior[k]))
 for k in ['decision_gate_adopted','adopted','coverage_model_adopted']:
  check('user_adopted:'+k,c[k] is True and record[k] is True and ready[k] is True)
  d=copy.deepcopy(c);d[k]=False
  try:control.validate_contract(d,True)
  except control.ExecutionClosed:check('missing_adoption_closes:'+k,True)
  else:check('missing_adoption_closes:'+k,False)
 check('still_no_execution_grant',c['contract_grants_execution'] is False and record['contract_grants_execution'] is False and ready['contract_grants_execution'] is False)
 check('no_execution_authorized',record['scientific_execution_authorized'] is False and ready['scientific_execution_authorized'] is False)
 check('verbatim_user_progress_instruction',record['user_instruction_verbatim']=='次の作業に進んで大丈夫')
 check('explicit_context_and_candidate_commit',record['preceding_publication_commit']==BASE and '次は限定gate・契約候補の正式採択です。' in record['preceding_assistant_context_verbatim'])
 check('adoption_not_platform_certificate','not platform/numerical certification' in record['adoption_flag_semantics'] and c['evidence']=='CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
 check('candidate_bytes_preserved',git('show',BASE+':'+CANDIDATE)==(ROOT/CANDIDATE).read_bytes())
 check('candidate_expected_sha',digest((ROOT/CANDIDATE).read_bytes())=='7989546fe64e644c01cb0080bd4623cc1b23504a5f421f446078742646db4c78')
 check('adopted_contract_hashes_agree',digest((ROOT/CONTRACT).read_bytes())==record['adopted_contract_sha256']==ready['contract_sha256'])
 oldgit=control.git
 try:
  def forbidden_git(*args):raise AssertionError('authority-dependent git reached without execution approval')
  control.git=forbidden_git
  try:control.authorize(ROOT,CONTRACT,None,None)
  except control.ExecutionClosed as exc:check('adoption_alone_cannot_run_before_git_or_scientific_input',str(exc)=='invalid hash')
  else:check('adoption_alone_cannot_run_before_git_or_scientific_input',False)
 finally:control.git=oldgit
 check('numerical_engine_never_imported','trottertracks.algorithm_codesign.m1s_gamma_pi3_v1.compare' not in sys.modules)
 try:control.run(ROOT,None)
 except control.ExecutionClosed:check('private_execution_permit_required_before_marker',True)
 else:check('private_execution_permit_required_before_marker',False)
 check('future_execution_authority_and_marker_unset',ready['execution_authority_commit'] is None and ready['launch_marker'] is None)
 check('no_platform_runtime_correctness_assertion',ready['numerical_correctness_validated'] is False and ready['actual_runtime_or_resource_usage_validated'] is False)
 check('no_decoder_adoption_or_GO',ready['decoder_adopted_or_implemented'] is False and c['science']['decoder_adopted_or_implemented'] is False and c['scientific_GO'] is None)
 source=c['source_identity'];check('same_frozen_source_identity',source==prior['source_identity'] and source['commit']=='2f800f179c7b29589b2f9a7e532408b16b8fe6d2')
 for p,sha in source['files'].items():
  b=(ROOT/p).read_bytes();check('SOURCE_sha:'+p,digest(b)==sha);check('SOURCE_original_commit:'+p,git('show',source['commit']+':'+p)==b)
  if p.endswith('.py'):ast.parse(b.decode());check('SOURCE_AST:'+p,True)
 check('source_precedes_adopted_contract_base',subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',source['commit'],BASE],capture_output=True).returncode==0 and source['commit']!=BASE)
 for item in load(ART+'/input_identity.json')['files']:
  b=(ROOT/item['path']).read_bytes();check('input_review_common_hash:'+item['path'],digest(b)==item['sha256'] and ('bytes' not in item or len(b)==item['bytes']))
 check('saved_input_original_commit_bytes',git('show',c['input']['commit']+':'+c['input']['path'])==(ROOT/c['input']['path']).read_bytes())
 sch=load(spec.ART+'/result_schema.json');Draft202012Validator.check_schema(sch);check('inherited_result_schema_metaschema',True)
 check('same_resources_and_none_time_caps',c['resources']==prior['resources'] and c['resources']['cpu_time_seconds'] is None and c['resources']['wall_time_seconds'] is None)
 check('same_full32_plus_G9_16_inventory',c['grid']==prior['grid'] and c['grid']['tested_G3_G6']==32 and c['grid']['G9_carried_not_tested']==16)
 check('old_permissions_not_reusable',ready['old_authorities_and_consumed_markers_reusable'] is False)
 for p in INDEXES:check('old_index_body:'+p,(ROOT/p).read_bytes().endswith(git('show',BASE+':'+p)))
 old=json.loads(git('show',BASE+':'+MANIFEST));new=load(MANIFEST)
 check('all_old_manifest_entries_preserved',all(new.get(k)==v for k,v in old.items()))
 check('only_new_adoption_entry',set(new)-set(old)=={'track_b_m1s_gamma_pi3_formal_adoption_20261017'})
 paths=set(git('diff','--name-only','-z',BASE).decode().split('\0'));paths.update(git('ls-files','--others','--exclude-standard','-z').decode().split('\0'))
 check('no_new_authorization_grant_result_marker',not any(p.startswith((spec.AUTH_PREFIX,spec.OUTPUT_PREFIX)) or p and Path(p).name in ['authorization.json','grant.json','marker.json','exclusive_launch_marker.json'] for p in paths))
 check('new_execution_paths_absent',not (ROOT/spec.AUTH_PREFIX).exists() and not (ROOT/spec.OUTPUT_PREFIX).exists())
 result={'schema':'m1s_gamma_pi3_adoption_static_validation','scope':'CONTRACT_ADOPTION_SOURCE_IDENTITY_GUARD_AST_SCHEMA_METADATA_ONLY','PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'checks':checks,'contract_sha256':digest((ROOT/CONTRACT).read_bytes()),'scientific_JSON_decoded':0,'new_numerical_evaluations':0,'gamma_threshold_comparisons':0,'new_authorizations_grants_markers':0,'scientific_execution_authorized':False,'mandatory_STOP':True}
 (ROOT/ART/'static_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['PASS','FAIL','contract_sha256','new_numerical_evaluations','scientific_execution_authorized']}))
 return int(result['FAIL']>0)


if __name__=='__main__':raise SystemExit(main())
