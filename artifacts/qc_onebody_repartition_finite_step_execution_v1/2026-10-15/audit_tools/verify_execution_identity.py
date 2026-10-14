"""Stdlib byte/schema/cap/preservation checks; never reruns numerical science."""
from pathlib import Path
import argparse,hashlib,json,subprocess

E='artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15'
A='artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15'
O='artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1'
BASE='0f92af19bde69fae69e97868b063a34e5afd5ca0'

def verify(w,root=None):
 w=Path(w)
 def read(p):return json.loads((w/p).read_text())
 def sha(p):return hashlib.sha256((w/p).read_bytes()).hexdigest()
 def git(*args,cwd=w):return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.PIPE)
 contract=read(A+'/contract.json');freeze=read(A+'/source_manifest.json');approval=read(E+'/authorization.json')
 assert contract['registered_execution_authorized'] is False
 assert read(A+'/authorization_target.json')['approved'] is False
 assert freeze['source_commit']=='19d5bd74db020564ea9a4b1882e102911e983cdb'
 assert approval['status']=='ISSUED_USER_EXPLICIT_EXECUTION' and approval['approved'] is True
 assert approval['issuer']=='USER_EXPLICIT_SESSION_INSTRUCTION'
 assert approval['user_instruction_verbatim']=='次の作業に進んで大丈夫です'
 assert approval['contract_SHA256']==sha(A+'/contract.json')==freeze['contract_SHA256']
 assert approval['source_manifest_SHA256']==sha(A+'/source_manifest.json')
 assert approval['input_SHA256']==sha(contract['input']['path'])==contract['input']['SHA256']
 assert approval['limits']==contract['limits'] and approval['attempts']==1 and approval['retry']==0
 for p,x in freeze['files'].items():
  assert sha(p)==x['SHA256'] and (w/p).stat().st_size==x['bytes']
  assert (w/p).read_bytes()==git('show',freeze['source_commit']+':'+p),p
 preserved=read(E+'/preservation_index.json')
 assert len(preserved['files'])==preserved['count']==450
 for p,x in preserved['files'].items():
  assert sha(p)==x['SHA256'] and (w/p).stat().st_size==x['bytes']
  assert (w/p).read_bytes()==git('show',BASE+':'+p),p
 old=json.loads(git('show',BASE+':artifacts/validation_manifest.json'));new=read('artifacts/validation_manifest.json')
 assert len(old['result_sets'])==152 and new['result_sets'][:152]==old['result_sets']
 assert {k:v for k,v in old.items() if k!='result_sets'}=={k:v for k,v in new.items() if k!='result_sets'}
 assert 'description' not in new['relevant_commits'][7]
 for p,x in read(E+'/evidence_identity.json')['files'].items():
  assert sha(p)==x['SHA256'] and (w/p).stat().st_size==x['bytes'],p
 marker=read(O+'/attempt_consumed.json');auditmarker=read(O+'/audit_consumed.json')
 assert marker['authorization_SHA256']==sha(E+'/authorization.json')
 assert marker['attempt']==auditmarker['attempt']==1 and marker['retry']==auditmarker['retry']==0
 identity=read(O+'/result_identity.json')
 assert identity['contract_SHA256']==sha(A+'/contract.json') and identity['source_manifest_SHA256']==sha(A+'/source_manifest.json')
 assert identity['input_SHA256']==approval['input_SHA256'] and identity['authorization_SHA256']==sha(E+'/authorization.json')
 raw=read(O+'/raw.json');seal=read(O+'/selector_seal.json');stop=read(O+'/stop.json');audit=read(O+'/independent_audit.json');summary=read(E+'/saved_results_summary.json')
 assert raw['raw_npz_SHA256']==sha(O+'/raw.npz') and raw['selector_seal_SHA256']==sha(O+'/selector_seal.json')
 assert not seal['E0_accessed_by_generator'] and not seal['finite_step_truth_accessed_by_generator']
 assert len(raw['records'])==69 and len(raw['costs'])==276 and len(raw['h_grid'])==6
 assert raw['counts']==stop['counts']==summary['producer_counts']
 assert stop['status']=='COLLECTION_COMPLETE_AWAIT_INDEPENDENT_AUDIT'
 assert raw['result']==summary['registered_result'] and raw['result']['classification']=='NO_UNIFORM_REGISTERED_COMPONENT_GAIN'
 assert all(c['status']=='NO_REGISTERED_COMPONENT_GAIN' for c in raw['result']['cells']) and len(raw['result']['cells'])==4
 assert audit==summary['independent_checker'] and audit['status']=='INDEPENDENT_JW_PADE_DIRECT_RESPONSE_COST_AUDIT_PASS'
 assert audit['PF_recompositions']==36 and audit['expm_calls']==705 and audit['new_eigensolvers']==audit['compile_calls']==0
 assert len(list((w/O).parent.glob('attempt_*')))==1
 for k,n in raw['counts'].items():assert n<=contract['limits'][k]
 assert summary['sequential_aggregate_peak_RSS_bytes']<contract['limits']['aggregate_RSS_bytes']
 assert sum(p.stat().st_size for p in (w/O).rglob('*') if p.is_file())==summary['attempt_output_bytes']<contract['limits']['output_bytes']
 assert sum((w/E/(n+'_console.log')).stat().st_size for n in ['producer','checker'])==summary['console_log_bytes']<contract['limits']['log_bytes']
 assert not summary['new_GPT_scientific_review_started'] and not summary['formal_research_GO']
 assert summary['attempts']==summary['audit_attempts']==1 and summary['retry']==summary['audit_retry']==0
 checked_root=0
 if root is not None:
  root=Path(root);baseline=read(E+'/root_worktree_baseline.json')
  parts=git('status','--porcelain=v1','-z','--untracked-files=all',cwd=root).decode().split('\0');status={};i=0
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
  checked_root=len(baseline)
 return {'status':'EXECUTION_BYTES_SCHEMA_CAPS_PRESERVATION_PASS','frozen_source_files':21,'protected_existing_files':450,'old_manifest_records_preserved':152,'non_result_fields_preserved':True,'known_global_FAIL_preserved':True,'whole_global_validator':'NOT_RUN_NOT_GLOBAL_PASS','checked_root_paths':checked_root,'producer_attempts':1,'independent_audit_attempts':1,'retries':0,'numerical_reexecution_calls':0,'new_GPT_reviews':0,'mandatory_stop':True}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worktree',type=Path,required=True);p.add_argument('--root-worktree',type=Path);p.add_argument('--output',type=Path)
 a=p.parse_args();result=verify(a.worktree,a.root_worktree)
 if a.output:a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
