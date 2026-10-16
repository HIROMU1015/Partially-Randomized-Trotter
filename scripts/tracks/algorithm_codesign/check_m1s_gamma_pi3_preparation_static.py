"""Preparation AST/schema/guard/hash audit; never decode saved scientific JSON.

No numeric kernel tests, saved threshold comparisons, pi/root recomputation,
or independent scientific validation are performed by this checker.
"""
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
REVIEW='docs/research/m1s_gamma_v1_completed_results_scientific_review_20261017.md'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md']
MANIFEST='artifacts/validation_manifest.json'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
 checks=[]
 def check(name,ok):checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
 def load(p):return control.read_json((ROOT/p).read_bytes())
 c=load(spec.ART+'/execution_contract_candidate.json');control.validate_contract(c)
 for k in ['decision_gate_adopted','adopted','coverage_model_adopted','contract_grants_execution']:
  check('closed:'+k,c[k] is False)
 auth=(ROOT/spec.PACKAGE/'control.py').read_text().split('def authorize(',1)[1].split('def seed_result(',1)[0]
 check('science_guard_before_input_or_git',auth.index('validate_contract')<auth.index('git(root,')<auth.index("raw = inside(root, i['path'])"))
 oldgit=control.git
 try:
  def prohibit_git(*args):raise AssertionError('git reached before candidate adoption guard')
  control.git=prohibit_git
  try:control.authorize(ROOT,spec.ART+'/execution_contract_candidate.json',None,None)
  except control.ExecutionClosed:check('candidate_rejected_before_git_input_engine',True)
  else:check('candidate_rejected_before_git_input_engine',False)
 finally:control.git=oldgit
 for name in ['decision_gate_adopted','adopted','coverage_model_adopted']:
  d=copy.deepcopy(c);d[name]=True
  try:control.validate_contract(d,True)
  except control.ExecutionClosed:check('one_flag_cannot_open:'+name,True)
  else:check('one_flag_cannot_open:'+name,False)
 try:control.run(ROOT,None)
 except control.ExecutionClosed:check('execution_permit_required_before_marker',True)
 else:check('execution_permit_required_before_marker',False)
 check('comparison_engine_never_imported','trottertracks.algorithm_codesign.m1s_gamma_pi3_v1.compare' not in sys.modules)
 for p in spec.SOURCE_FILES:
  if not p.endswith('.py'):continue
  tree=ast.parse((ROOT/p).read_text());check('AST:'+p,isinstance(tree,ast.Module))
  imports=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
   if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
  check('no_old_science_or_floating_engine:'+p,not any(any(t in x for t in ['numpy','scipy','mpmath','math','qiskit','m1s_saved_signal','m1s_finite','trotterlib']) for x in imports))
  check('no_eval_exec_compile:'+p,not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['eval','exec','compile'] for n in ast.walk(tree)))
 compare=(ROOT/spec.PACKAGE/'compare.py').read_text();ctrl=(ROOT/spec.PACKAGE/'control.py').read_text()
 tree=ast.parse(compare)
 functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
 for name in ['classify','evaluate']:
  first=ast.unparse(functions[name].body[0]);check('kernel_permit_first:'+name,first=='require_permit(permit)')
 check('no_numeric_module_entry',all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)) or isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) for n in tree.body))
 cond=[ast.unparse(n.test) for n in ast.walk(functions['classify']) if isinstance(n,ast.If)]
 check('strict_satisfied_rule_AST','rational.bounded(3 * upper) < pi_lower' in cond)
 check('nonstrict_negation_rule_AST','rational.bounded(3 * lower) >= pi_upper' in cond)
 for token in ['UNRESOLVED','G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED','ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE','SHARED_LOWER','SHARED_HIGHER','NOT_INFERRED_FROM_CONTINUOUS_ORDER','SAVED_V1_CONDITIONAL_ENGINEERING_ENCLOSURES_ONLY']:
  check('scope_and_classification:'+token,token in compare or token in str(c))
 check('no_new_pi_or_root_function',not any(isinstance(n,ast.FunctionDef) and any(t in n.name for t in ['atan','sin','cos','log','root','pi']) for n in ast.walk(tree)))
 for token in ['authority-only child required','source identity: ','saved input identity','execution worktree must be clean','one-shot output already exists; no retry','output.mkdir()','aggregate_memory_cap','supervisor_result.json']:
  check('guard:'+token,token in ctrl)
 check('engine_deferred_after_limits',ctrl.index('from .compare import evaluate')>ctrl.index('resource.setrlimit(resource.RLIMIT_AS'))
 check('time_caps_none_as_current_policy',c['resources']['cpu_time_seconds'] is None and c['resources']['wall_time_seconds'] is None)
 check('caps_structural_only',c['resources']==spec.RESOURCES)
 check('48_records_32_tested_16_G9',c['grid']['contrast_records']==48 and c['grid']['tested_G3_G6']==32 and c['grid']['G9_carried_not_tested']==16)
 schema=load(spec.ART+'/result_schema.json');Draft202012Validator.check_schema(schema);check('schema_metaschema',True)
 check('schema_all48',schema['properties']['comparisons']['minItems']==schema['properties']['comparisons']['maxItems']==48)
 for k in ['scientific_GO','absolute_shot_counts','native_costs','total_resources']:check('no_scope_expansion:'+k,c[k] is None and schema['properties'][k]['type']=='null')
 for p,sha,n in [(spec.INPUT['path'],spec.INPUT['sha256'],spec.INPUT['bytes']),
                  (REVIEW,'18492941acc4fc7fe7fd46b6f002c21f324e52a661ec13bc0a024c5e86ad7303',None),
                  ('docs/research/research_project_instructions_unified_2026-10-16.md','8323be3f65856adf0f4d095d8e68546db3326822023c20e101ba5f596605ebfa',None)]:
  b=(ROOT/p).read_bytes();check('input_review_instructions_hash:'+p,digest(b)==sha and (n is None or len(b)==n))
 check('input_original_commit_bytes',(ROOT/spec.INPUT['path']).read_bytes()==git('show',spec.INPUT['commit']+':'+spec.INPUT['path']))
 source=c['source_identity']
 check('source_frozen',source is not None)
 if source:
  for p,sha in source['files'].items():
   b=(ROOT/p).read_bytes();check('source_sha:'+p,digest(b)==sha);check('source_commit:'+p,git('show',source['commit']+':'+p)==b)
  check('source_after_result_base',subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',spec.BASE,source['commit']],capture_output=True).returncode==0)
 for p in INDEXES:check('old_index_body:'+p,(ROOT/p).read_bytes().endswith(git('show',spec.BASE+':'+p)))
 old=json.loads(git('show',spec.BASE+':'+MANIFEST));new=load(MANIFEST)
 check('old_manifest_entries',all(new.get(k)==v for k,v in old.items()))
 check('only_new_preparation_entry',set(new)-set(old)=={'track_b_m1s_gamma_pi3_preparation_20261017'})
 changed=set(git('diff','--name-only','-z',spec.BASE).decode().split('\0'))
 changed.update(git('ls-files','--others','--exclude-standard','-z').decode().split('\0'))
 check('no_new_science_authority_or_result_or_marker',not any(p.startswith((spec.AUTH_PREFIX,spec.OUTPUT_PREFIX)) or p and Path(p).name in ['authorization.json','grant.json','marker.json','exclusive_launch_marker.json'] for p in changed))
 check('new_authority_and_result_paths_absent',not (ROOT/spec.AUTH_PREFIX).exists() and not (ROOT/spec.OUTPUT_PREFIX).exists())
 record={'schema':'m1s_gamma_pi3_preparation_static_validation','scope':'AST_METADATA_SCHEMA_GUARD_AND_BYTES_HASH_ONLY','PASS':sum(x['status']=='PASS' for x in checks),'FAIL':sum(x['status']=='FAIL' for x in checks),'checks':checks,'saved_scientific_JSON_decoded':0,'new_numerical_evaluations':0,'gamma_threshold_comparisons':0,'toy_numeric_correctness_tests':0,'new_scientific_execution_authorizations_grants_markers':0,'independent_numerical_correctness_validated':False,'platform_certificate':False,'mandatory_STOP':True}
 (ROOT/spec.ART/'static_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:record[k] for k in ['PASS','FAIL','scope','new_numerical_evaluations','gamma_threshold_comparisons']}))
 return int(record['FAIL']>0)


if __name__=='__main__':raise SystemExit(main())
