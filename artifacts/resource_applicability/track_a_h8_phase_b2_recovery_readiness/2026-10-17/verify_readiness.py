"""Public saved metadata/bytes only; never launch a resource gate or science."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(n):return json.loads((HERE/n).read_text())
def main():
 for row in read('publication_files_v1.json')['files']:
  path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
  raw=(ROOT/path).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(path)
 p=read('execution_contract_UNAPPROVED_v1.json');g=read('grant_INVALID_DRAFT_v1.json');s=read('source_profile_input_binding_v1.json');t=read('technical_validation_v1.json');v=read('preservation_receipt_v1.json')
 assert len(p['axes'])==2 and all(x['case_id']=='B0I_p21_q8_4th' and x['exact_prediction']=='1852640' for x in p['axes'])
 assert p['preparation_concurrency']==2 and p['compile_concurrency']==1 and p['check_axis_before_next_admission']
 assert p['resources']['physical_cores']==5 and (p['resources']['aggregate_RSS_bytes']+p['resources']['admission_reserve_bytes'])/2**30==112.3125
 assert not p['authorization']['approved'] and not p['authorization']['runtime_authorization']
 assert not any(g[k] for k in ('approved_by_user','runtime_authorization','accept_saved_B0_2_axes','valid','retry','refit','C_authorized','next_stage_authorized'))
 assert g['allowed_cpus']==[] and g['command'] is None and g['worker_cpus']==[] and g['witness_cpu'] is None
 for name,digest in s['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 assert hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==g['plan_sha256']==s['plan_sha256']
 assert s['SOURCE']=='77ce12d8c935702dc78f294b5863b4904f9c6a2c' and len(s['source_hashes'])==31 and s['provider_files_verified']==265 and s['runtime_files_verified']==34273 and len(s['input_receipts'])==16
 assert t['unittests']['methods']==34 and t['unittests']['status']=='PASS' and t['synthetic_provider']['compiled_metrics_equal'] and t['synthetic_provider']['QPY_roundtrip_fingerprints_equal']
 assert t['synthetic_provider']['parallel_workers_exit0']==4 and t['synthetic_provider']['real_H8_compile']==0
 assert v['old_protected_files_unchanged']==687 and v['old_worktrees_unchanged']==124 and v['prior_B2_raw_files_unchanged']==45 and v['original_B2_run01_status']=='STOP'
 assert v['new_valid_grants']==v['new_H8_compile']==v['new_fit']==0 and not v['C_authorized']
 print(json.dumps(dict(status='PASS_SAVED_RECOVERY_READINESS_ONLY',parallel_preparation_workers=2,compile_workers=1,new_H8_compile=0,new_valid_grants=0,mandatory_STOP=True)))
if __name__=='__main__':main()
