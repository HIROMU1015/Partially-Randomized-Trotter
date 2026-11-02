"""Verify published bytes and saved audit metadata only; no scientific imports."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(name):return json.loads((HERE/name).read_text())
def main():
 inventory=read('publication_files_v1.json')
 for row in inventory['files']:
  path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
  raw=(ROOT/path).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(path)
 a=read('saved_anchor_audit_v1.json')
 assert a['status']=='SAVED_EIGHT_AXES_STRUCTURALLY_REUSABLE_PENDING_B1_APPROVAL' and a['measured_axes_verified']==8
 assert a['original_acquisition_status']=='STOP' and not a['original_final_gate_replaced']
 assert a['historical_thread_violator']=='UNRESOLVED_NOT_RECORDED' and not a['historical_thread_compliance_certified']
 assert all(not a[k] for k in ['new_compile','new_signal','new_fit','new_residual','new_cost_reaccount','new_valid_grant','B1_authorized','next_stage_authorized','automatic_retry'])
 assert [a[k] for k in ['source_files_verified','provider_files_verified','runtime_files_verified','input_files_verified']]==[15,265,34273,8]
 assert a['original_run02_files_verified']==53
 p=read('preservation_v1.json');assert p['old_protected_files_unchanged']==681 and p['old_worktrees_HEAD_and_status_unchanged']==118
 t=read('technical_validation_v1.json');assert t['new_tests']==14 and t['existing_regression_tests']==83 and t['pinned_runtime']['tests']==97 and t['pinned_runtime']['status']=='PASS'
 s=read('repair_SOURCE_binding_v1.json');assert s['source_files']==16 and not s['new_valid_grant']
 assert s['original_scientific_execution_SOURCE']!=s['repair_SOURCE']
 print(json.dumps(dict(status='PASS_PUBLIC_SAVED_AUDIT_ONLY',saved_axes=8,original_contract='STOP',new_compile=0,new_fit=0,next_stage_authorized=False,mandatory_STOP=True)))
if __name__=='__main__':main()
