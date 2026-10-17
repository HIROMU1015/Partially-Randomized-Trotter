"""Saved immutable projections only; no live gate, science or compile."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(n):return json.loads((HERE/n).read_text())
def main():
 for row in read('publication_files_v1.json')['files']:
  p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts
  raw=(ROOT/p).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(p)
 b=read('source_input_binding_v1.json');assert len(b['source_hashes'])==33 and b['provider_files']==265 and b['runtime_files']==34273 and len(b['input_receipts'])==31
 for n,h in b['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 c=read('saved_cosine_measurement_v1.json');assert c['measured_RZ']==1852640 and c['native_exit_code']==0 and c['reaped'] and c['model_check']['PASS'] and c['model_check']['exact_relative_error']=='0'
 o=read('origin_technical_STOP_v1.json');assert o['driver_terminal']['status']=='STOP' and o['resource_failure']['reason']=='MEMORY_PRESSURE' and o['native_launcher_wait']['wait_completed']
 g=read('startup_gates_v1.json');assert g['CPU_initial']['status']==g['CPU_refresh']['status']=='PASS' and g['fresh_gate']['driver_bindings_verified']
 s=read('running_samples_v1.json');assert s['observed_span_seconds']>=120 and s['max_observed_monitor_gap_seconds']<=5
 p=read('execution_scope_v1.json');assert p['compile_cap']==p['qpy_load_cap']==1 and p['control_probe_cap']==0 and not p['Phase_C_authorized'] and p['STOP_threshold']=='1/50'
 assert read('technical_validation_v1.json')['test_methods']==26
 assert read('execution_scope_v1.json')['PSI_policy']=='DIAGNOSTIC_OBSERVATION_ONLY' and read('execution_scope_v1.json')['OOM_increment_STOPS']
 print(json.dumps(dict(status='PASS_SAVED_SINE_RUNNING_RECEIPT_ONLY',new_scientific_calculation=0,final_completion_claimed=False)))
if __name__=='__main__':main()
