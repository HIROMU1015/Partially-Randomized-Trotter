"""Read-only bytes/metadata check of an acquisition that remains STOP."""
import csv
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def read(name):return json.loads((HERE/name).read_text())
def main():
 m=read('publication_files_v1.json')
 for row in m['files']:
  p=Path(row['path'])
  assert not p.is_absolute() and '..' not in p.parts
  raw=(ROOT/p).read_bytes()
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(p)
 stop=read('STOP_assessment_v1.json');driver=read('driver_STOP_v1.json');launcher=read('resource_and_cleanup_STOP_v1.json')
 assert driver['status']==launcher['status']==read('launcher_admission_STOP_v1.json')['status']=='STOP'
 assert launcher['first_stop']=='RuntimeError:PROCESS_THREADS' and launcher['native_exit_code']==-9
 assert launcher['all_reaped'] and launcher['cleanup_errors']==[]
 assert read('launcher_native_exit_v1.json')['native_exit_code']==1
 assert stop['accepted_axes_saved']==8 and not stop['final_gate_exists']
 assert not any(stop[k] for k in ['automatic_retry','STOP_changed_to_PASS','measurements_automatically_admitted_to_B1','new_fit','new_residual','new_cost_reaccount','B2_or_C_started','next_stage_authorized'])
 assert stop['mandatory_STOP'] and not stop['failing_PID_recorded'] and not stop['failing_thread_count_recorded']
 saved=[r for name in ['train_measurements.json','holdout_measurements.json'] for r in read(name)['axes']]
 expected={(f'B0_p{p}_q{q}_2nd',axis) for p,q in [(14,1),(21,1),(14,8),(21,8)] for axis in ['cosine','sine']}
 assert len(saved)==8 and {(r['case_id'],r['axis']) for r in saved}==expected
 assert all(r['native_exit_code']==0 and r['reaped'] and r['fingerprint_before']==r['fingerprint_after'] and 0<=r['max_control_error']<1e-8 for r in saved)
 assert all(not read(name)['fit_performed'] for name in ['train_measurements.json','holdout_measurements.json'])
 with (HERE/'new_B0_saved_measured_axes_v1.csv').open() as stream:rows=list(csv.DictReader(stream))
 assert len(rows)==8
 for row,original in zip(rows,saved):
  assert all(row[k]==str(original[k]) for k in row),row['case_id']
 workers=[w for wave in read('worker_exit_reap_receipts_v1.json')['waves'] for w in wave['workers']]
 assert len(workers)==8 and all(w['reaped'] and w['exit_code']==0 for w in workers)
 integrity=read('post_STOP_file_integrity_v1.json')
 assert [integrity[k] for k in ['source_files','provider_files','runtime_files','input_files']]==[15,265,34273,8]
 preservation=read('preservation_receipt_v1.json')
 assert preservation['old_protected_files_checked']==678 and preservation['old_worktrees_checked']==117
 assert preservation['acquisition_status']=='STOP' and not preservation['replaces_failed_final_gate']
 print(json.dumps(dict(status='PASS_SAVED_STOP_METADATA_ONLY',saved_axes=8,acquisition_status='STOP',new_compile=0,new_fit=0,next_stage_authorized=False,mandatory_STOP=True)))
if __name__=='__main__':main()
