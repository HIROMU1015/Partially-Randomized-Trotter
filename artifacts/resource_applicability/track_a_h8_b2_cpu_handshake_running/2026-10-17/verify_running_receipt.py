"""Saved running snapshot only: no live gate, compile or scientific analysis."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(n):return json.loads((HERE/n).read_text())
def main():
 for row in read('publication_files_v1.json')['files']:
  p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts
  raw=(ROOT/p).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(p)
 b=read('source_input_binding_v1.json');assert len(b['source_hashes'])==31 and b['provider_files']==265 and b['runtime_files']==34273 and len(b['input_receipts'])==16
 for n,h in b['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 a=read('attempt01_stop_v1.json');assert a['new_scientific_calls']==a['new_compile']==0 and a['owned_cleanup']['all_reaped'] and a['native_exit']['wait_completed']
 s=read('stability_samples_v1.json');assert s['observation_span_seconds']>=180 and s['driver_CPU_increase_seconds']>0 and s['max_monitor_gap_seconds']<=5 and s['all_sampled_identities_alive_unchanged']
 g=read('attempt02_startup_gates_v1.json');assert g['CPU_initial']['status']==g['CPU_refresh']['status']=='PASS' and g['fresh_gate']['driver_bindings_verified']
 p=read('new_execution_plan_v1.json');assert len(p['new_axes'])==2 and p['compile_workers']==1 and not p['C_authorized'] and p['scientific_model_error_threshold_exact']=='1/50'
 assert read('startup_fix_validation_v1.json')['total_test_methods']==40
 print(json.dumps(dict(status='PASS_SAVED_RUNNING_SNAPSHOT_ONLY',new_compile=0,new_fit=0,scientific_completion_claimed=False)))
if __name__=='__main__':main()
