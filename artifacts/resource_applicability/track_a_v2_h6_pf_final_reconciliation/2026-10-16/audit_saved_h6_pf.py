"""Final bounded H6 exploration PF metadata audit; no scientific imports or fit."""
from pathlib import Path
from collections import Counter
import argparse,copy,json,hashlib,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--projects-root',type=Path,required=True);p.add_argument('--output-private',type=Path,required=True);a=p.parse_args()
W=a.projects_root/'partially-randomized-trotter-worktrees';E=a.projects_root/'h6-handoff-evidence';R=W/'v2-residual-analysis-readiness-20261016';T=W/'native-df-theory-evaluation-20261016';O=a.output_private
if not O.is_absolute() or O.exists():raise ValueError('NEW_EXCLUSIVE_METADATA_OUTPUT_REQUIRED')
O.mkdir(mode=0o700);receipts={};cache={}
def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(path):
 path=Path(path)
 if path.suffix in ('.npz','.gz') or path.name.startswith(('theoretical_wrapper_cost','finite_law','primitive_receipts')):raise ValueError('OUT_OF_SCOPE_DATA')
 if str(path) not in cache:
  raw=path.read_bytes();receipts[str(path)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());cache[str(path)]=json.loads(raw)
 return cache[str(path)]
def git(c,f):
 key=c+':'+f
 if key not in cache:
  raw=subprocess.check_output(['git','show',key],cwd=R);receipts[key]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());cache[key]=raw
 return cache[key]
def write(name,v):(O/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
prior=E/'20261016/track-a-v2-provenance-reconciliation-20261016'
reg=read(prior/'final_reconciliation_v3/provenance_registry_v2_PRIVATE.json');assert digest(reg)==digest(copy.deepcopy(reg));oldreg=copy.deepcopy(reg)
catalog=read(prior/'final_collection/raw_metadata_PRIVATE.json')
index_path=T/'artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/saved_candidate_connection_index_v1.json';index=read(index_path)
entries=[r for r in index['H6'] if r['scope']=='H6_EXPLORATION92' and not r['cell_id'].startswith('B2_')];assert len(entries)==20
raw={r['task']['cell_id']:r for r in catalog['records'] if r['system']=='H6' and r['task'].get('phase')=='exploration' and 'cell_id' in r['task'] and not r['task']['cell_id'].startswith('B2_')};assert len(raw)==16
legacy='54986a9f54515c8c2289e38fa16fd81e2b6008c2';server='6a924e52cf1b6541e2780e9d1b756b8049683166';prprovider='87b6b9f46b7e8ba9abe132ee21478c114b6f46f7';profile='4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75'
archive_path=E/'20261012/roundoff-execution-v1/h6-roundoff-preparations-v1.json';archive=read(archive_path);assert receipts[str(archive_path)]['sha256']=='ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010';assert archive['binding']['profile_sha256']==profile
assert archive['export_receipt']['deterministic_preparation_unchanged'] is True;assert archive['export_receipt']['policy']['deterministic_basis']=='unchanged server native'
core=['src/trotterlib/df_rte_qiskit.py','src/trotterlib/df_rte_circuit.py','src/trotterlib/df_rte_tail.py','src/trotterlib/rte.py','src/trotterlib/df_partial_s2.py','src/trotterlib/df_trotter/ops.py','src/trottertracks/resource_applicability/ax2b_native_df_v5.py','src/trotterlib/pr2_matched_accuracy_m1_execution.py']
for f in core:
 for source in (legacy,server):assert git(source,f)==git(prprovider,f),('SHARED_NATIVE_SOURCE_DIFFERENCE',source,f)
for source in (legacy,server):
 code=git(source,'src/trottertracks/resource_applicability/h6_matched_port_v1.py').decode();assert "(_prepare_discard if c['method']=='B0' else _prepare)" in code
 for f in ['src/trottertracks/resource_applicability/h6_matched_execution_v2.py','src/trottertracks/resource_applicability/h6_matched_contract_v2.py']:git(source,f)
for f in ['src/trottertracks/resource_applicability/h6_server_contract_v1.py','src/trottertracks/resource_applicability/h6_server_execution_v1.py','src/trottertracks/resource_applicability/h6_matched_cost_port_v3.py']:git(server,f)
profile_path='artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/environment_compiler_profile_v2.json';assert hashlib.sha256(git(prprovider,profile_path)).hexdigest()==profile
profile_file=W/'h6-native-dependency-fix-20261012'/profile_path;read(profile_file);assert receipts[str(profile_file)]['sha256']==profile
for f in ['src/trottertracks/resource_applicability/h6_roundoff_preparation_v1.py','src/trottertracks/resource_applicability/h6_frozen_preparation_v1.py']:git(prprovider,f)
prep_identity=read(T/'artifacts/resource_applicability/track_a_native_df_theory_evaluation/2026-10-16/preparation_identity_receipt_v1.json')['H6']
rows=[]
for entry in entries:
 cid=entry['cell_id'];rr=next(r for r in reg if r['key']==['H6','H6_EXPLORATION92',cid]);h=int.from_bytes(hashlib.sha256(canonical(['H6',entry['cell']['prefix'],entry['cell']['q'],entry['cell']['formula']])).digest(),'big')%5==0
 item=dict(cell_id=cid,split='holdout' if h else 'train',measured=entry['paired_groups']>0,strictly_comparable_to_resolved_PR=False)
 if not item['measured']:
  assert cid not in raw;item.update(status='UNMEASURED_NO_NEW_ACQUISITION',origin_sources_preserved=entry['execution_sources'])
 else:
  r=raw[cid];d=Path(r['directory']);tb=read(d/'task_binding.json');wc=read(d/'worker_claim.json');wt=read(d/'worker_terminal.json');tr=read(d/'trajectory.json');mf=Path(r['manifest_matches'][0]);m=read(mf);task=tb['task']
  assert task['cell']==entry['cell'] and wc['task_digest']==digest(task)
  assert wc['source_commit']==wt['source_commit']==m['source_commit']==r['source'];assert entry['execution_sources']==[r['source']]
  assert tb['manifest_digest']==hashlib.sha256(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  assert tr['events'] is None and tr['finite_distribution'] is None
  axes={ax:read(d/('wrapper_symmetric_directional_'+ax+'.json')) for ax in ('cosine','sine')}
  for ax,x in axes.items():
   assert x['axis']==ax and x['cell_id']==cid and x['control']=='symmetric_directional'
   for k in ('phase','seed','replica'):assert x[k]==task[k]
   assert x['event_digest']==tr['event_digest'] and x['compiler_hash']==entry['compiler_hash']
  pair={ax:{**{k:str(x[k]) if k in ('replica','seed') else x[k] for k in ['cell_id','phase','replica','seed','control','axis','compiler_hash','event_digest','fingerprint']},'execution_source':r['source']} for ax,x in axes.items()}
  assert digest([pair])==entry['wrapper_identity_digest']
  profile_ref=m.get('profile');prof_ok=bool(profile_ref and profile_ref.get('sha256')==profile)
  assert prof_ok==(cid=='B1_p19_q8_4th')
  direct_profile_files=[f.name for f in d.iterdir() if f.is_file() and 'profile' in f.name.lower()];assert not direct_profile_files
  signal=Path(next(z['directory'] for z in raw.values() if z['source']==legacy)).parent/'signal/prepared_representation.json';saved=read(signal);old=next(x for x in saved['preparations'] if x['cell_id']==cid)
  root=('discard' if entry['cell']['method']=='B0' else 'faithful')+':p'+str(entry['cell']['prefix']);arch=prep_identity[root]
  item.update(origin_SOURCE=r['source'],compiler_options=m['plan']['compiler_proposed'],declared_environment=m['environment'],compiler_profile_receipt_bound=prof_ok,profile_sha256=profile if prof_ok else None,native_composition='PF_NO_RTE_NO_BASIS_REUSE_EXPLICIT_SCALAR',native_core_byte_match=True,cost_worker_preparation='REBUILT_PER_TASK_NOT_FROZEN_ARCHIVE',cost_worker_preparation_hash_saved=False,signal_reference_preparation_hash=old['preparation_hash'],archive_reference_preparation_hash=arch['preparation_hash'],reference_preparation_hash_equal=old['preparation_hash']==arch['preparation_hash'],reference_hash_difference_is_not_physical_difference_certificate=True,archive_declares_unchanged_server_deterministic_preparation=True,complete_per_wrapper_basis_binding=False,status='PROFILE_BOUND_BASIS_LINEAGE_STILL_UNRESOLVED' if prof_ok else 'PROFILE_AND_BASIS_LINEAGE_UNRESOLVED',receipt_digest=digest({f:receipts[str(d/f)] for f in ['task_binding.json','worker_claim.json','worker_terminal.json','trajectory.json','wrapper_symmetric_directional_cosine.json','wrapper_symmetric_directional_sine.json']}))
  if prof_ok:rr['values']['compiler_profile']=profile
 rr['pf_final_evidence']=item;rr['metadata_resolved']=False;rows.append(item)
# Scope guard: no PR, confirmation, H8 or B0I registry row is changed.
changed=[n['key'] for n,o in zip(reg,oldreg) if n!=o];assert len(changed)==20 and all(k[0]=='H6' and k[1]=='H6_EXPLORATION92' and not k[2].startswith('B2_') for k in changed)
assert sum(r['metadata_resolved'] for r in reg)==25
profile_ceiling=Counter(r['split'] for r in rows if r.get('compiler_profile_receipt_bound'))
assert profile_ceiling=={'train':1};assert not any(r['strictly_comparable_to_resolved_PR'] for r in rows)
summary=dict(schema='track_a_v2_h6_pf_final_metadata_audit_v1',status='NO_GO_CANDIDATE_CURRENT_RESIDUAL_DESIGN',decision_owner='USER',investigation_scope='H6_EXPLORATION92_PF_ONLY',PF_entries_checked=20,measured_PF=16,unmeasured_PF=4,compiler_profile_unresolved_measured_PF=15,compiler_profile_bound_measured_PF=1,strictly_comparable_PF=0,profile_only_necessary_condition_ceiling=dict(train=1,holdout=0,not_basis_promotion=True),fixed_minimum_each_train_and_holdout_family=2,prior_resolved_exploration_PR_counts=dict(train=11,holdout=6),fit_eligible_strata=0,prior_global_metadata_resolved=25,global_metadata_resolved_after=25,source_scope_expanded=False,broad_recovery_stopped=True,conditions_or_split_relaxed=False,full_numeric_envelope_certified=False,new_fit=0,new_residual=0,new_compile=0,new_signal=0,new_sampling=0,new_law=0,new_primitive=0,new_runtime_config=0,new_grant=0,runtime_authorization=False,next_stage_authorized=False,mandatory_stop=True)
write('PF_entry_audit_PRIVATE.json',rows);write('provenance_registry_final_PRIVATE.json',reg);write('input_receipts_PRIVATE.json',receipts);write('summary_PRIVATE.json',summary)
print(json.dumps(dict(status=summary['status'],PF_entries=20,measured=16,profile_bound=1,strictly_comparable_PF=0,fit_eligible=0,input_receipts=len(receipts))))
