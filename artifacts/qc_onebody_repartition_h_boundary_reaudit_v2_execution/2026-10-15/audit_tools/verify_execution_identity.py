"""Independent stdlib evidence/caps/provenance proof; never run numerical audit."""
from pathlib import Path
import hashlib,json,posixpath,re,subprocess
BASE='d96c9c6cb654058201cc15688aa1422beedefac5'
P='artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15'
E='artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15'
R='artifacts/qc_onebody_repartition_h_boundary_reaudit_v2/2026-10-15/attempt_1'
O='artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify(w,root):
 w=Path(w);root=Path(root);read=lambda p:json.loads((w/p).read_text());c=read(P+'/contract.json');a=read(E+'/authorization.json');f=read(P+'/source_manifest.json');audit=read(R+'/independent_audit.json')
 assert a['approved'] is True and a['issuer']=='USER_EXPLICIT_SESSION_INSTRUCTION' and a['scope_id']==c['scope_id'] and a['limits']==c['limits'] and a['attempts']==1 and a['retry']==0
 for field,name in [('contract_SHA256','contract.json'),('source_manifest_SHA256','source_manifest.json'),('input_identity_SHA256','input_identity.json')]:assert a[field]==sha(w/P/name)
 assert read(P+'/authorization_target.json')['approved'] is False
 identity=read(E+'/result_identity.json');marker=read(R+'/attempt_consumed.json')
 assert marker['scope_id']==a['scope_id'] and marker['attempt']==1 and marker['retry']==0
 assert marker['authorization_SHA256']==identity['authorization_SHA256']==sha(w/E/'authorization.json')
 assert marker['source_manifest_SHA256']==a['source_manifest_SHA256'] and marker['input_identity_SHA256']==a['input_identity_SHA256']
 assert identity['saved_raw_SHA256']==sha(w/O/'raw.json') and identity['new_audit_SHA256']==sha(w/R/'independent_audit.json')
 assert identity['saved_original_failed_audit_SHA256']==sha(w/O/'independent_audit.json') and read(O+'/independent_audit.json')['reason']=='NUMERICAL_ALLOWANCE'
 assert audit['status']=='INDEPENDENT_BOUNDARY_JW_PADE_PHASE_COST_POLICY_AUDIT_PASS' and audit['checker_version']==2 and audit['scope_id']==c['scope_id']
 counts=audit['diagnostics']['partial_counts'];assert audit['diagnostics']['stage']=='complete'
 assert counts['PF_recompositions_started']==counts['PF_recompositions_completed']==counts['cells_completed']==audit['PF_recompositions']==4572
 assert counts['expm_calls']==counts['expm_completed']==audit['expm_calls']==91401<=c['limits']['independent_expm_calls']
 assert counts['cell_arrays_loaded']==9294<=c['limits']['saved_cell_array_loads'] and counts['method_records_checked']==4572 and counts['cost_rows_checked']==18288
 assert audit['diagnostics']['peak_RSS_bytes']==265728000<c['limits']['aggregate_RSS_bytes'] and audit['new_eigensolvers_RDM_leading_optimizer_compile_calls']==0
 assert audit['maxima']['PF_matrix_difference']<1e-9 and not audit['formal_research_GO'] and audit['original_results_modified'] is False and audit['old_authorization_reused'] is False
 resource=(w/E/'checker_resource.txt').read_text();assert re.search(r'Exit status:\s*0\s*$',resource) and re.search(r'Maximum resident set size \(kbytes\):\s*259500',resource)
 output=sum(p.stat().st_size for p in (w/R).rglob('*') if p.is_file());logs=(w/E/'checker_console.log').stat().st_size+(w/E/'checker_resource.txt').stat().st_size
 assert output<=c['limits']['output_bytes'] and logs<=c['limits']['log_bytes']
 pres=read(E+'/preservation_index.json')['files']
 for p,x in pres.items():assert (w/p).stat().st_size==x['bytes'] and sha(w/p)==x['SHA256'],p
 for p,x in f['files'].items():assert (w/p).stat().st_size==x['bytes'] and sha(w/p)==x['SHA256'],p
 assert len(pres)==5140 and len(f['files'])==59
 baseline=read(E+'/root_worktree_baseline.json');status=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=root)
 full_root_status_unchanged=hashlib.sha256(status).hexdigest()==baseline['status_SHA256']
 items=status.split(b'\0');current={};i=0
 while i<len(items) and items[i]:
  item=items[i].decode();i+=1;path=item[3:];current[path]={'status':item[:2]}
  if 'R' in item[:2] or 'C' in item[:2]:current[path]['old_path']=items[i].decode();i+=1
 old_paths={row['path'] for row in baseline['records']};added_root_paths=sorted(set(current)-old_paths)
 for row in baseline['records']:
  assert row['path'] in current and current[row['path']]['status']==row['status'],row['path']
 assert not (old_paths-set(current)),'ROOT_BASELINE_PATH_REMOVED'
 for row in baseline['records']:
  if row['identity'] is not None:
   p=root/row['path'];assert p.stat().st_size==row['identity']['bytes'] and sha(p)==row['identity']['SHA256'],row['path']
 old=json.loads(subprocess.check_output(['git','show',BASE+':artifacts/validation_manifest.json'],cwd=w));new=read('artifacts/validation_manifest.json')
 assert len(old['result_sets'])==160 and new['result_sets'][:160]==old['result_sets']
 assert {k:v for k,v in new.items() if k!='result_sets'}=={k:v for k,v in old.items() if k!='result_sets'} and 'description' not in new['relevant_commits'][7]
 s=read(E+'/saved_results_summary.json');assert s['audit']==audit and s['scientific_validation_state']=='INDEPENDENT_V2_NUMERICAL_AUDIT_PASS_FIXED_SAVED_SCOPE' and s['saved_result']['classification']=='NO_UNIFORM_REGISTERED_COMPONENT_GAIN'
 assert len(s['saved_result']['cells'])==4 and all(x['status']=='NO_REGISTERED_COMPONENT_GAIN' for x in s['saved_result']['cells'])
 assert all(v['resolved'] for cell in s['saved_result']['cells'] for v in cell['boundary_status'].values())
 for name in ('N1','N2','N3'):
  assert read(E+'/execution_stop.json')[name]=={'N1':'STOP','N2':'HOLD','N3':'CLOSED'}[name]
 paths=[p.decode() for p in subprocess.check_output(['git','diff','--name-only','-z',BASE],cwd=w).split(b'\0') if p]
 paths+=['docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md','docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_gpt_handoff_20261015.md','docs/research/研究ノート/2026-10-15_qc_onebody_repartition_h_boundary_reaudit_v2_execution.md'];links=[]
 for p in paths:
  if not p.endswith('.md'):continue
  text=(w/p).read_text()
  try:prior=subprocess.check_output(['git','show',BASE+':'+p],cwd=w,stderr=subprocess.DEVNULL).decode()
  except subprocess.CalledProcessError:prior=''
  if prior:assert text.endswith(prior.split('\n',1)[1]),p
  oldlinks=set(re.findall(r'\]\(([^)]+)\)',prior))
  for url in re.findall(r'\]\(([^)]+)\)',text):
   if url in oldlinks or url.startswith(('https:','http:','#','mailto:')):continue
   target=posixpath.normpath(posixpath.join(posixpath.dirname(p),url.split('#')[0]));assert (w/target).is_file(),(p,target);links.append({'document':p,'target':target})
 return {'status':'REAUDIT_V2_IDENTITY_CAPS_PRESERVATION_PASS','protected_old_files':len(pres),'frozen_source_files':59,'root_paths_matched':len(baseline['records']),'root_full_status_unchanged':full_root_status_unchanged,'root_current_paths':len(current),'root_added_paths_outside_this_scope':added_root_paths,
  'old_manifest_records_preserved':160,'non_result_fields_preserved':True,'known_global_description_FAIL_preserved':True,'whole_global_validator':'NOT_RUN_NOT_GLOBAL_PASS',
  'numeric_audit_status':audit['status'],'PF_recompositions':4572,'expm_calls':91401,'cell_projector_array_loads':9294,'method_records_checked':4572,'cost_rows_checked':18288,
  'peak_RSS_bytes':265728000,'output_bytes':output,'log_resource_bytes':logs,'attempts':1,'retry':0,'numeric_reexecution_for_metadata_proof':False,'no_source_or_input_changes':True,
  'legacy_failed_audit_preserved':True,'new_GPT_review':False,'research_adoption_or_closure':False,'relative_links':links,'mandatory_stop':True}
if __name__=='__main__':
 import sys
 print(json.dumps(verify(Path(sys.argv[1]),Path(sys.argv[2])),ensure_ascii=False,indent=2))
