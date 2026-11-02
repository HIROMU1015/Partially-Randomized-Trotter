"""Stdlib byte/preservation checks; no numerical checker retry or matrix load."""
from pathlib import Path
import json,hashlib,subprocess,argparse
E='artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15'
A='artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15'
O='artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1'
BASE='361cf233fb2e4bb00b340311d27d936d7b44fbb6'
def verify(w,root=None):
 w=Path(w).resolve()
 def read(p):return json.loads((w/p).read_text())
 def sha(p):return hashlib.sha256((w/p).read_bytes()).hexdigest()
 def git(*args,cwd=w):return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.PIPE)
 c=read(A+'/contract.json');f=read(A+'/source_manifest.json');a=read(E+'/authorization.json');pres=read(E+'/preservation_index.json')
 assert len(f['files'])==42 and len(pres['files'])==pres['count']==515
 assert c['registered_execution_authorized'] is False and read(A+'/authorization_target.json')['approved'] is False
 assert a['approved'] is True and a['status']=='ISSUED_USER_EXPLICIT_EXECUTION' and a['issuer']=='USER_EXPLICIT_SESSION_INSTRUCTION'
 assert a['user_instruction_verbatim']=='次の作業に進んで大丈夫'
 assert a['contract_SHA256']==sha(A+'/contract.json')==f['contract_SHA256'] and a['source_manifest_SHA256']==sha(A+'/source_manifest.json')
 assert a['fixed_files']==c['fixed_files'] and a['limits']==c['limits'] and a['attempts']==1 and a['retry']==0
 for p,x in f['files'].items():assert sha(p)==x['SHA256'] and (w/p).read_bytes()==git('show',f['source_commit']+':'+p),p
 for p,x in pres['files'].items():assert sha(p)==x['SHA256'] and (w/p).read_bytes()==git('show',BASE+':'+p),p
 for p,x in c['fixed_files'].items():assert sha(p)==x['SHA256'] and (w/p).stat().st_size==x['bytes'],p
 old=json.loads(git('show',BASE+':artifacts/validation_manifest.json'));new=read('artifacts/validation_manifest.json')
 assert len(old['result_sets'])==156 and new['result_sets'][:156]==old['result_sets']
 assert {k:v for k,v in old.items() if k!='result_sets'}=={k:v for k,v in new.items() if k!='result_sets'} and 'description' not in new['relevant_commits'][7]
 for p,x in read(E+'/evidence_identity.json')['files'].items():assert sha(p)==x['SHA256'] and (w/p).stat().st_size==x['bytes'],p
 marker=read(O+'/attempt_consumed.json');auditmarker=read(O+'/audit_consumed.json');identity=read(O+'/result_identity.json')
 assert marker['authorization_SHA256']==identity['authorization_SHA256']==sha(E+'/authorization.json')
 assert marker['attempt']==auditmarker['attempt']==1 and marker['retry']==auditmarker['retry']==0
 assert identity['contract_SHA256']==sha(A+'/contract.json') and identity['source_manifest_SHA256']==sha(A+'/source_manifest.json') and identity['fixed_files']==c['fixed_files']
 assert sha(O+'/inherited_selector_seal.json')==sha(c['saved_selector_seal'])
 raw=read(O+'/raw.json');stop=read(O+'/stop.json');audit=read(O+'/independent_audit.json');summary=read(E+'/saved_results_summary.json')
 assert stop['status']=='COLLECTION_COMPLETE_AWAIT_INDEPENDENT_AUDIT' and raw['counts']==stop['counts']==summary['producer_counts']
 assert len(raw['primitive_records'])==4572 and len(raw['records'])==4572 and len(raw['costs'])==18288
 assert len(raw['h_grid'])==762 and len(raw['assessment_h_grid'])==381 and raw['cached_anchor_cells']==12
 assert raw['counts']['PF_seed_evaluations']==raw['counts']['PF_Schur_calls']==4560
 assert raw['result']==summary['unvalidated_producer_result']
 assert audit==summary['independent_audit'] and audit['status']=='INDEPENDENT_AUDIT_STOP_NO_RETRY' and audit['reason']=='NUMERICAL_ALLOWANCE'
 assert summary['scientific_validation_state']=='UNVALIDATED_PRODUCER_ONLY' and not summary['source_modified_or_retry_to_make_audit_pass']
 assert not summary['additional_numerical_failure_diagnosis'] and not summary['new_GPT_review_started']
 assert len(list((w/O).parent.glob('attempt_*')))==1
 for r in raw['primitive_records']:assert sha(O+'/cells/'+r['key']+'.npz')==r['array_SHA256']
 for k,n in raw['counts'].items():assert n<=c['limits'][k]
 assert summary['producer_resource']['exit_code']==0 and summary['checker_resource']['exit_code']==2
 assert max(summary[n]['peak_RSS_bytes'] for n in ('producer_resource','checker_resource'))<c['limits']['aggregate_RSS_bytes']
 assert sum(p.stat().st_size for p in (w/O).rglob('*') if p.is_file())==summary['output_bytes']<c['limits']['output_bytes']
 assert summary['console_log_bytes']<c['limits']['log_bytes']
 root_count=0
 if root is not None:
  root=Path(root);base=read(E+'/root_worktree_baseline.json');parts=git('status','--porcelain=v1','-z','--untracked-files=all',cwd=root).decode().split('\0');status={};i=0
  while i<len(parts):
   line=parts[i];i+=1
   if not line:continue
   flag,p=line[:2],line[3:]
   if 'R' in flag or 'C' in flag:i+=1
   if not p.startswith('.worktrees/'):status[p]=flag
  assert set(status)==set(base)
  for p,x in base.items():
   b=(root/p).read_bytes() if (root/p).is_file() else None
   assert (hashlib.sha256(b).hexdigest() if b is not None else None)==x['SHA256'] and status[p]==x['status'],p
  root_count=len(base)
 return {'status':'EXECUTION_BYTE_IDENTITY_CAPS_PRESERVATION_PASS_WITH_NUMERICAL_AUDIT_FAILED','source_commit':f['source_commit'],'frozen_files':42,'protected_old_files':515,'old_manifest_records_preserved':156,'known_global_FAIL_preserved':True,'checked_root_paths':root_count,'producer_attempts':1,'audit_attempts':1,'retries':0,'independent_numerical_audit_passed':False,'independent_failure':'NUMERICAL_ALLOWANCE','scientific_validation_state':'UNVALIDATED_PRODUCER_ONLY','new_numeric_calls_in_this_byte_check':0,'new_GPT_reviews':0,'mandatory_stop':True}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worktree',type=Path,required=True);p.add_argument('--root-worktree',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=verify(a.worktree,a.root_worktree)
 if a.output:a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
