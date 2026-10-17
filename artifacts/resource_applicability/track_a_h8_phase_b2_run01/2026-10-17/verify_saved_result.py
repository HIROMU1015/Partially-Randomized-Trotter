"""Validate saved B2 bytes/metadata only; never compile, refit or reaccount."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(n):return json.loads((HERE/n).read_text())
def main():
 for row in read('publication_files_v1.json')['files']:
  p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts
  raw=(ROOT/p).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(p)
 s=read('result_summary_v1.json');d=read('execution_terminal_v1.json');a=read('launch_admission_terminal_v1.json');o=read('resource_and_cleanup_v1.json');n=read('launcher_native_exit_v1.json');f=read('final_gate_v1.json');c=read('execution_contract_reference_v1.json');rows=read('axis_measurements_v1.json')['axes']
 assert s['SOURCE']=='244c45e752fa703e286d40ece9ad641430ea9c6e' and s['readiness_commit']=='3a450af0fb431269ac94152807992883c1490b26'
 expected=[(cid,axis) for cid in ['B0_p15_q1_4th','B0I_p21_q8_4th'] for axis in ['cosine','sine']]
 assert [(v['case_id'],v['axis']) for v in rows]==expected[:len(rows)]
 assert s['registered_axes']==4 and s['accepted_axis_receipts']==len(rows) and s['actual_compile_entry_markers']<=4
 assert c['compile_concurrency']==s['compile_concurrency']==1 and c['relative_error_denominator']=='MEASURED_RZ'
 assert c['authorization_projection_not_an_executable_grant'] and c['authorization_for_this_run']
 assert o['all_reaped'] and not o['cleanup_errors']
 if s['status']=='COMPLETE_MANDATORY_STOP':
  assert d['status']==a['status']==o['status']=='COMPLETE_MANDATORY_STOP' and len(rows)==s['axes_PASS']==4
  assert s['actual_compile_entry_markers']==4 and d['actual_calls']==c['call_caps']
  assert f['status']=='PASS_MANDATORY_STOP' and n['native_exit_code']==o['native_exit_code']==0 and n['wait_completed'] and o['first_stop'] is None
  assert d['axes']==rows and all(v['model_check']['PASS'] for v in rows)
 else:
  assert s['status']==a['status']==o['status']=='STOP' and o['first_stop']=='RuntimeError:MEMORY_PRESSURE' and o['native_exit_code']==-9
  assert n['status']=='NATIVE_LAUNCHER_WAIT_RECEIPT_MISSING' and n['native_exit_code'] is None and n['wait_completed'] is False and n['launcher_reported_exit_code']==1
  assert len(rows)==s['axes_PASS']==2 and s['actual_compile_entry_markers']==3 and s['unknown_reserved_compile_calls']==1
  assert d['completed_calls_lower_bound']['compile']==d['completed_calls_lower_bound']['model_error_check']==2 and len(d['irrevocable_reservations'])==3 and len(d['settlements'])==2
  assert f['status']=='NOT_ISSUED'
 for row in rows:
  assert row['fingerprint_before']==row['fingerprint_after'] and row['native_exit_code']==0 and row['reaped']
  assert 0<=row['max_control_error']<1e-8 and row['model_check']['measured_RZ']==row['measured_RZ']
 assert s['new_fit']==s['new_cost_reaccount']==s['new_signal']==s['SCF']==s['DF']==s['state_solver']==s['PR_sampling']==0
 assert not any(s[k] for k in ('automatic_retry','refit','extra_compile','C_authorized','next_stage_authorized','full_grid_guaranteed','rigorous_bias_or_PR_population_guaranteed'))
 p=read('preservation_receipt_v1.json');assert p['status']=='PASS_BYTES_SOURCE_RUNTIME_INPUT_AND_OLD_WORKTREES_UNCHANGED' and p['old_protected_files_unchanged']==684 and p['old_worktrees_unchanged']==123
 assert p['old_B0_run02_status']=='STOP' and p['B1_not_reexecuted'] and p['V2_NO_GO_preserved'] and p['M1_S_not_modified'] and p['old_V2_untracked_unchanged']
 assert [v['files_verified'] for v in p['prior_raw_checks']]==[53,34]
 assert read('call_ledger_projection_v1.json')['actual_compile_entry_markers']==s['actual_compile_entry_markers']
 binding=read('source_profile_input_binding_v1.json');assert len(binding['source_hashes'])==21 and binding['provider_files_verified']==265 and binding['runtime_files_verified']==34273 and len(binding['input_receipts'])==15
 assert binding['model_sha256']=='8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0' and not binding['approved_grant_content_published']
 workers=read('serial_worker_receipts_v1.json')['axes'];assert len(workers)==3
 controls=read('native_control_preparations_v1.json')['cases'];assert len(controls)==2 and all(0<=v['max_control_error']<1e-8 for v in controls)
 control_by_id={v['case']['id']:v for v in controls}
 for ordinal,worker in enumerate(workers):
  receipt=worker['parallel_receipt'];assert receipt['all_reaped'] and not receipt['cleanup_errors'] and len(receipt['workers'])==1 and receipt['workers'][0]['reaped']
  projected={Path(v['logical_path']).name:v for v in worker['files']}
  entered=projected['compile_entered.json']['projection'];assert entered['API_call_entered_not_completion'] and entered['reserved_in_parent'] and entered['SOURCE']==s['SOURCE']
  assert entered['fingerprint']==control_by_id[entered['task']['case_id']]['wrapper_fingerprints'][entered['task']['axis']]
  if ordinal<2:
   row=rows[ordinal];saved=projected['task_result.json'];assert saved['raw_sha256']==row['result_sha256']
   assert all(saved['projection'][k]==row[k] for k in ('case_id','axis','measured_RZ','CX','depth','fingerprint_before','fingerprint_after','compiler_hash','SOURCE'))
   assert receipt['status']=='PASS' and receipt['workers'][0]['exit_code']==0
  else:
   assert receipt['status']=='STOP' and receipt['workers'][0]['exit_code']==2 and 'task_result.json' not in projected and 'task_STOP.json' in projected
 assert d['compile_entry_markers']==0 and read('call_ledger_projection_v1.json')['actual_compile_entry_markers']==3
 print(json.dumps(dict(status='PASS_SAVED_B2_RECEIPTS_ONLY',phase_status=s['status'],saved_axis_receipts=len(rows),new_compile=0,new_fit=0,new_reaccount=0,C_authorized=False,mandatory_STOP=True)))
if __name__=='__main__':main()
