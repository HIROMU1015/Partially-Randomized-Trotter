"""Stdlib metadata/byte proof only; never decode science arrays or rerun audit."""
from pathlib import Path
import hashlib,json,posixpath,re,subprocess
P='artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15'
BASE='3dd8acecbe3fe4e18d11b7e462906706921c8152'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify(w,root):
 w=Path(w);root=Path(root);read=lambda p:json.loads((w/p).read_text())
 protected=read(P+'/preservation_index.json')['files']
 for p,x in protected.items():assert (w/p).stat().st_size==x['bytes'] and sha(w/p)==x['SHA256'],p
 baseline=read(P+'/root_worktree_baseline.json');status=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=root)
 assert hashlib.sha256(status).hexdigest()==baseline['status_SHA256'],'ROOT_STATUS_CHANGED'
 for row in baseline['records']:
  if row['identity'] is not None:
   p=root/row['path'];assert p.stat().st_size==row['identity']['bytes'] and sha(p)==row['identity']['SHA256'],row['path']
 old=json.loads(subprocess.check_output(['git','show',BASE+':artifacts/validation_manifest.json'],cwd=w));new=read('artifacts/validation_manifest.json')
 assert len(old['result_sets'])==158 and new['result_sets'][:158]==old['result_sets']
 assert {k:v for k,v in new.items() if k!='result_sets'}=={k:v for k,v in old.items() if k!='result_sets'}
 assert 'description' not in new['relevant_commits'][7]
 source=read(P+'/source_manifest.json')
 for p,x in source['files'].items():assert (w/p).stat().st_size==x['bytes'] and sha(w/p)==x['SHA256'],p
 assert source['contract_SHA256']==sha(w/P/'contract.json') and source['input_identity_SHA256']==sha(w/P/'input_identity.json')
 assert read(P+'/authorization_target.json')['approved'] is False
 assert not (w/'artifacts/qc_onebody_repartition_h_boundary_reaudit_v2/2026-10-15/attempt_1').exists()
 assert read(P+'/synthetic_test_record.json')['tests']==29
 changed=subprocess.check_output(['git','diff','--name-only','-z',BASE],cwd=w).split(b'\0');links=[]
 paths=[p.decode() for p in changed if p]+['docs/research/qc_onebody_repartition_h_boundary_checker_v2_repair_20261015.md','docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_execution_handoff_20261015.md','docs/research/研究ノート/2026-10-15_qc_onebody_repartition_h_boundary_checker_v2_preparation.md']
 for p in paths:
  if not p.endswith('.md'):continue
  content=(w/p).read_text()
  try:prior=subprocess.check_output(['git','show',BASE+':'+p],cwd=w,stderr=subprocess.DEVNULL).decode()
  except subprocess.CalledProcessError:prior=''
  old_links=set(re.findall(r'\]\(([^)]+)\)',prior))
  if prior:assert content.endswith(prior.split('\n',1)[1]),p
  for url in re.findall(r'\]\(([^)]+)\)',content):
   if url in old_links or url.startswith(('https:','http:','#','mailto:')):continue
   target=posixpath.normpath(posixpath.join(posixpath.dirname(p),url.split('#')[0]));assert (w/target).is_file(),(p,target);links.append({'document':p,'target':target})
 return {'status':'METADATA_BYTE_ONLY_PRESERVATION_PASS','protected_old_files':len(protected),'root_paths_matched':len(baseline['records']),
  'old_manifest_records_preserved':158,'known_global_description_FAIL_preserved':True,'non_result_fields_preserved':True,
  'new_source_files_matched':len(source['files']),'relative_links':links,'H4_registered_array_decode_or_numerical_reaudit':0,
  'real_scientific_validation_state':'UNVALIDATED_PRODUCER_ONLY','new_approval_issued':False,'new_attempts_consumed':0,'mandatory_stop':True}
if __name__=='__main__':
 import sys
 print(json.dumps(verify(Path(sys.argv[1]),Path(sys.argv[2])),ensure_ascii=False,indent=2))
