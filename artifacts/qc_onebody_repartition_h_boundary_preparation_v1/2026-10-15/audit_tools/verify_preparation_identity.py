"""Stdlib preservation/source/header audit; no registered numerical evaluation."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,ast
A='artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15'
BASE='31ea58449eb8e535b8f5d5f62956fb410d35f0ae'
def verify(w,root=None):
 w=Path(w).resolve();sys.path.insert(0,str(w/'src'))
 from trottertracks.ground_pf_repartition.qc_onebody_repartition_h_boundary_execution_v1 import preflight,OUTPUT
 result=preflight(w)
 def read(p):return json.loads((w/p).read_text())
 def sha(p):return hashlib.sha256((w/p).read_bytes()).hexdigest()
 def git(*args,cwd=w):return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.PIPE)
 freeze=read(A+'/source_manifest.json');protected=read(A+'/preservation_index.json')
 assert len(freeze['files'])==42 and protected['count']==len(protected['files'])==480
 for p,x in freeze['files'].items():
  assert sha(p)==x['SHA256'] and (w/p).read_bytes()==git('show',freeze['source_commit']+':'+p),p
  if p.endswith('.py'):ast.parse((w/p).read_text())
 for p,x in protected['files'].items():
  assert sha(p)==x['SHA256'] and (w/p).read_bytes()==git('show',BASE+':'+p),p
 old=json.loads(git('show',BASE+':artifacts/validation_manifest.json'));new=read('artifacts/validation_manifest.json')
 assert len(old['result_sets'])==154 and new['result_sets'][:154]==old['result_sets']
 assert {k:v for k,v in old.items() if k!='result_sets'}=={k:v for k,v in new.items() if k!='result_sets'}
 assert 'description' not in new['relevant_commits'][7]
 target=read(A+'/authorization_target.json');tests=read(A+'/synthetic_tests.json');gates=read(A+'/execution_gate_tests.json')
 assert target['status']=='UNISSUED_APPROVAL_TARGET' and target['approved'] is False
 assert target['contract_SHA256']==sha(A+'/contract.json') and target['source_manifest_SHA256']==sha(A+'/source_manifest.json')
 assert tests['status']=='PASS' and tests['test_count']==35 and tests['registered_array_payloads_deserialized']==0
 assert gates['status']=='PASS' and not gates['registered_attempt_consumed'] and not gates['authorization_issued']
 assert not (w/OUTPUT).exists()
 review=read(A+'/received_review_identity.json');assert sha(review['published_path'])==review['SHA256'] and (w/review['published_path']).stat().st_size==43818
 count=0
 if root:
  root=Path(root);baseline=read(A+'/root_worktree_baseline.json');parts=git('status','--porcelain=v1','-z','--untracked-files=all',cwd=root).decode().split('\0');status={};i=0
  while i<len(parts):
   line=parts[i];i+=1
   if not line:continue
   flag,p=line[:2],line[3:]
   if 'R' in flag or 'C' in flag:i+=1
   if not p.startswith('.worktrees/'):status[p]=flag
  assert set(status)==set(baseline)
  for p,x in baseline.items():
   b=(root/p).read_bytes() if (root/p).is_file() else None
   assert (hashlib.sha256(b).hexdigest() if b is not None else None)==x['SHA256'] and status[p]==x['status'],p
  assert Path(review['original_path']).read_bytes()==(w/review['published_path']).read_bytes();count=len(baseline)
 result.update(status='BOUNDARY_PREPARATION_SOURCE_BYTES_PRESERVATION_PASS',source_commit=freeze['source_commit'],frozen_files=42,protected_old_files=480,
   old_manifest_records_preserved=154,non_result_fields_preserved=True,known_global_description_FAIL_preserved=True,whole_global_validator='NOT_RUN_NOT_GLOBAL_PASS',
   received_review_verbatim_bytes=43818,checked_root_paths=count,synthetic_tests_passed=35,authorization_issued=False,attempt_consumed=False,new_GPT_review_started=False,current_CI=False)
 return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worktree',type=Path,required=True);p.add_argument('--root-worktree',type=Path);p.add_argument('--output',type=Path)
 a=p.parse_args();r=verify(a.worktree,a.root_worktree)
 if a.output:a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
