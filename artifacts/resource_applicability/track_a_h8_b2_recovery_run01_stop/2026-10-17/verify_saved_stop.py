"""Saved STOP metadata/bytes only; no live gate, fit or compile."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(n):return json.loads((HERE/n).read_text())
def main():
 for row in read('publication_files_v1.json')['files']:
  p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts
  raw=(ROOT/p).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(p)
 s=read('stop_summary_v1.json');a=read('launch_admission_terminal_v1.json');c=read('CPU_initial_v1.json');n=read('native_launcher_exit_v1.json');d=read('dispatcher_native_exit_v1.json');v=read('preservation_receipt_v1.json');b=read('source_input_binding_v1.json')
 assert s['status']=='STOP_BEFORE_SCIENTIFIC_DRIVER' and a['status']=='STOP' and a['reason']=='PHYSICAL_CORE_BUSY' and a['stage']=='INITIAL_FRESH_CPU_RESOURCE_GATE'
 assert c['status']=='STOP' and c['failing_CPUs']==[6] and c['own_activity_subtracted'] is False
 failed=[x for x in c['CPUs'] if x['cpu']==6][0];assert failed['busy_ticks']==82 and failed['total_ticks']==100 and failed['busy_fraction']==.82
 assert not a['driver_started'] and not a['science_reservation_journal_exists'] and all(not a['path_state'][k] for k in ('output','control','lock'))
 assert n['native_exit_code']==1 and n['wait_completed'] and d['dispatcher_native_exit_code']==0 and d['wait_completed']
 assert s['actual_scientific_driver']==s['actual_scientific_workers']==s['actual_control_probes']==s['actual_QPY_dump_load']==s['actual_new_H8_compile']==s['new_fit']==s['new_reaccount']==0
 assert not any(s[k] for k in ('CPU_reassigned','automatic_retry','fresh_gate_issued','output_control_lock_created','ram_disk_PSI_OOM_fresh_checks_reached','saved_B0_driver_acceptance_receipt_created','C_authorized','next_stage_authorized'))
 assert v['old_protected_files_unchanged']==688 and v['old_worktrees_unchanged']==125 and v['old_B2_raw_files_unchanged']==45
 assert len(b['source_hashes'])==31 and b['provider_files_verified']==265 and b['runtime_files_verified']==34273 and len(b['input_receipts'])==16
 for name,digest in b['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 print(json.dumps(dict(status='PASS_SAVED_CPU_STOP_ONLY',new_H8_compile=0,new_fit=0,automatic_retry=False,mandatory_STOP=True)))
if __name__=='__main__':main()
