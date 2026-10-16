"""Public bytes and metadata only; never invoke resources or science."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(name):return json.loads((HERE/name).read_text())
def main():
 for row in read('publication_files_v1.json')['files']:
  p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts
  raw=(ROOT/p).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],str(p)
 p=read('execution_contract_UNAPPROVED_v1.json');g=read('grant_INVALID_DRAFT_v1.json');r=read('source_profile_input_binding_v1.json')
 assert not p['authorization']['approved'] and not p['authorization']['runtime_authorization']
 assert g['schema']=='H8_PHASE_B2_NEW_SCOPE_GRANT_V1' and not g['approved_by_user'] and not g['runtime_authorization']
 assert g['allowed_cpus']==[] and g['command'] is None and not g['valid']
 assert p['axes']==r['axes'] and [(x['case_id'],x['axis']) for x in p['axes']]==[(cid,a) for cid in ['B0_p15_q1_4th','B0I_p21_q8_4th'] for a in ['cosine','sine']]
 assert [x['exact_prediction'] for x in p['axes']]==['1175933/7']*2+['1852640']*2
 assert p['compile_concurrency']==1 and p['relative_error_denominator']=='MEASURED_RZ'
 assert p['call_caps']['compile']==4 and p['call_caps']['cost_fit']==p['call_caps']['cost_reaccount']==0
 assert not p['retry'] and not p['refit'] and not p['C_authorized'] and not p['next_stage_authorized']
 assert len(r['source_hashes'])==21 and r['provider_files_verified']==265 and r['runtime_files_verified']==34273 and len(r['input_receipts'])==15
 assert r['status']=='PREPARED_NOT_AUTHORIZED' and not r['fresh_resource_gate_issued'] and not r['new_valid_grants']
 t=read('technical_validation_v1.json');assert t['unittests']['tests']==119 and t['unittests']['status']=='PASS'
 assert t['synthetic_provider']['synthetic_compile_axes']==4 and t['synthetic_provider']['real_H8_compile']==0
 old=read('preservation_receipt_v1.json');assert old['old_protected_files_unchanged']==683 and old['initial_old_worktrees_unchanged']==121 and old['user_authorized_parallel_worktree_updates']==1 and old['second_baseline_old_worktrees_unchanged']==122
 assert old['original_B0_run02_status']=='STOP' and not old['M1_S_implementation_or_contract_modified']
 print(json.dumps(dict(status='PASS_SAVED_B2_READINESS_ONLY',axes=4,compile_concurrency=1,new_H8_compile=0,new_fit=0,new_valid_grants=0,fresh_gate_issued=False,mandatory_STOP=True)))
if __name__=='__main__':main()
