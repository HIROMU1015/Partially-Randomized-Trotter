"""Saved B1 bytes/metadata only; no refit, prediction evaluation or compile."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(name):return json.loads((HERE/name).read_text())
def main():
 for row in read('publication_files_v1.json')['files']:
  p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts
  raw=(ROOT/p).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(p)
 d=read('execution_terminal_v1.json');phase=read('phase_terminal_v1.json');obs=read('resource_and_cleanup_v1.json');native=read('launcher_native_exit_v1.json')
 assert d['status']==phase['status']==obs['status']=='COMPLETE_MANDATORY_STOP'
 assert obs['native_exit_code']==native['native_exit_code']==0 and native['wait_completed']
 assert obs['all_reaped'] and not obs['cleanup_errors'] and obs['first_stop'] is None
 assert d['actual_calls']==read('execution_contract_v1.json')['call_caps']
 models=read('frozen_models_v1.json');canonical=json.dumps(models,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 assert hashlib.sha256(canonical).hexdigest()==d['model_sha256']==phase['model_sha256']
 assert [(m['method'],m['axis']) for m in models['models']]==[(m,a) for m in ['B0','B0I'] for a in ['cosine','sine']]
 for m in models['models']:
  assert all(c['PASS'] and c['relative_error']=='0' for c in m['checks'])
  assert len(m['TRAIN_ids'])==(3 if m['method']=='B0' else 8)
  assert len([c for c in m['checks'] if c['role']=='HOLDOUT'])==(1 if m['method']=='B0' else 0)
 assert not models['S4_measured_validation'] and not models['compiled_grid_guaranteed']
 assert phase['original_acquisition_status']=='STOP' and not phase['B2_authorized'] and not phase['C_authorized']
 assert len(read('input_anchor_registry_v1.json')['rows'])==24
 predictions=read('B2_predictions_UNAPPROVED_v1.json');assert predictions['model_sha256']==d['model_sha256'] and len(predictions['axes'])==4
 assert all(r['status']=='MODEL_ESTIMATE_NOT_MEASURED' and not r['compile_authorized'] for r in predictions['axes'])
 t=read('technical_tests_v1.json');assert t['status']=='PASS' and t['tests']==35
 p=read('preservation_v1.json');assert p['old_protected_files_unchanged']==681 and p['old_worktrees_unchanged']==119
 assert not phase['next_stage_authorized']
 print(json.dumps(dict(status='PASS_SAVED_B1_RESULT_ONLY',four_strata_PASS=True,new_fit=0,new_compile=0,B2_started=False,next_stage_authorized=False,mandatory_STOP=True)))
if __name__=='__main__':main()
