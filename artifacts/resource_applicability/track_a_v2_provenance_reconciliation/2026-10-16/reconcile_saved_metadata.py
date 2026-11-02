"""Read-only saved metadata reconciliation; never import scientific providers."""
from pathlib import Path
from collections import defaultdict,Counter
import json,csv,hashlib,subprocess,copy
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--projects-root',type=Path,required=True)
p.add_argument('--inventory-private',type=Path,required=True)
p.add_argument('--output-private',type=Path,required=True)
args=p.parse_args()
W=args.projects_root/'partially-randomized-trotter-worktrees';E=args.projects_root/'h6-handoff-evidence'
R=W/'v2-residual-analysis-readiness-20261016';T=W/'native-df-theory-evaluation-20261016';P=args.output_private
if not P.is_absolute() or P.exists():raise ValueError('NEW_EXCLUSIVE_METADATA_OUTPUT_REQUIRED')
P.mkdir(mode=0o700)
A=T/'artifacts/resource_applicability';M=A/'track_a_native_df_metadata_completion/2026-10-16';Q=A/'track_a_native_df_theory_evaluation/2026-10-16'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):
 if str(p) in receipts:return receipts[str(p)]['sha256']
 return hashlib.sha256(p.read_bytes()).hexdigest()
receipts={}; readcache={}; gitcache={}; manifest_digests={}
def read(p):
 p=Path(p)
 if str(p) in readcache:return readcache[str(p)]
 if p.suffix=='.npz' or p.name.startswith(('theoretical_wrapper_cost','same_candidate_saved_cost_connection','primitive_receipts')):raise ValueError('NOT_METADATA_INPUT')
 b=p.read_bytes();receipts[str(p)]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());v=json.loads(b);readcache[str(p)]=v;return v
def git(source,path):
 if (source,path) in gitcache:return gitcache[(source,path)]
 b=subprocess.check_output(['git','show',source+':'+path],cwd=R);gitcache[(source,path)]=b;receipts[source+':'+path]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return b
v=read(args.inventory_private);raw=defaultdict(list)
for r in v['records']:
 t=r['task']
 if 'cell_id' not in t:continue
 raw[(r['system'],t['cell_id'],t['phase'],str(t['replica']),str(t['seed']),r['source'])].append(r)
index=read(M/'saved_candidate_connection_index_v1.json');binding=read(Q/'execution_binding_v1.json');preps=read(Q/'preparation_identity_receipt_v1.json');laws=read(Q/'finite_law_receipts_v1.json');lawby={(x['system'],x['candidate_id']):x for x in laws}
old=read(E/'20261016/track-a-v2-residual-implementation-readiness/provenance_registry.json');registry=copy.deepcopy(old);reg={tuple(x['key']):x for x in registry}
profile_old='4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75';archive='ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010'
sourcefiles=['src/trotterlib/df_rte_qiskit.py','src/trotterlib/df_rte_circuit.py','src/trotterlib/df_rte_tail.py','src/trotterlib/rte.py','src/trotterlib/df_partial_s2.py','src/trotterlib/df_trotter/ops.py','src/trottertracks/resource_applicability/ax2b_native_df_v5.py']
provider={'H6':'87b6b9f46b7e8ba9abe132ee21478c114b6f46f7','H8':'c31ef424175c02379eafebe8fc6c7548c1d9aabe'}
sourceproof={};reference={f:hashlib.sha256(git(provider['H6'],f)).hexdigest() for f in sourcefiles}
for sys,rows in index.items():
 for s in sorted({s for x in rows if not x['scope'].endswith('B0I_FAMILY12') for s in x['execution_sources']}):
  hashes={f:hashlib.sha256(git(s,f)).hexdigest() for f in sourcefiles};assert hashes==reference,('COMMON_SOURCE_CHANGED',s)
  sourceproof[s]=hashes
assert sha(E/'20261012/roundoff-execution-v1/h6-roundoff-preparations-v1.json')==archive
# Existing profile bytes are read as a receipt; no environment is loaded or rebuilt.
profile_path='artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/environment_compiler_profile_v2.json';assert hashlib.sha256(git(provider['H6'],profile_path)).hexdigest()==profile_old
allpairs={};pair_evidence={};summaries=[];individual=[]
for sys,name in [('H6','h6-native-b0i-full-family-run04'),('H8','h8-native-b0i-full-family-run01')]:
 base=E/'20261015'/name;mapping=read(base/'input_mapping_private.json');contract=read(base/'final_contract_v1.json');coverage=read(M/'saved_identity_coverage_v1.json')
 csvs=[Path(mapping['legacy_wrapper_metrics.csv'])]
 if sys=='H6':csvs.append(W/'h6-hash-reuse-eight-worker-20261012/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/wrapper_metrics_v1.csv')
 groups=defaultdict(dict)
 for f in csvs:
  assert any(x['sha256']==sha(f) for x in coverage['input_receipts']),('CSV_RECEIPT',sys)
  receipts[str(f)]=dict(bytes=f.stat().st_size,sha256=sha(f))
  for row in csv.DictReader(f.open()):
   if row['control']!='symmetric_directional':continue
   source=row.get('source_commit',row.get('source'))
   if source is None:
    found=[r['source'] for k,rs in raw.items() if k[:5]==(sys,row['cell_id'],row['phase'],row['replica'],row['seed']) for r in rs]
    assert len(set(found))==1;source=found[0]
   key=(row['phase'],row['cell_id'],row['replica'],row['seed'],row['control']);assert row['axis'] not in groups[key]
   groups[key][row['axis']]={k:row[k] for k in ['cell_id','phase','replica','seed','control','axis','compiler_hash','event_digest','fingerprint']};groups[key][row['axis']]['execution_source']=source
 by=defaultdict(list)
 for key,axes in groups.items():
  assert set(axes)=={'cosine','sine'}
  for k in ['execution_source','compiler_hash','seed','event_digest','cell_id','phase','replica','control']:assert axes['cosine'][k]==axes['sine'][k]
  a=axes['cosine'];rs=raw[(sys,a['cell_id'],a['phase'],a['replica'],a['seed'],a['execution_source'])];matching=[]
  for r in rs:
   d=Path(r['directory']);valid=True
   for axis in ('cosine','sine'):
    f=d/('wrapper_symmetric_directional_'+axis+'.json')
    if not f.exists():valid=False;break
    w=read(f)
    if any(str(w[k])!=str(axes[axis][k]) for k in ['cell_id','phase','replica','seed','control','axis','compiler_hash','event_digest','fingerprint']):valid=False;break
   if valid:matching.append(r)
  assert matching,('NO_RAW_PAIR_MATCH',sys,key)
  assert len({r['task_sha'] for r in matching})==1,('AMBIGUOUS_RAW_TASK',sys,key)
  r=matching[0];d=Path(r['directory']);tb=read(d/'task_binding.json');claim=read(d/'worker_claim.json');assert claim['source_commit']==a['execution_source'];assert len(r['manifest_matches'])==1
  mf=Path(r['manifest_matches'][0]);m=read(mf)
  if str(mf) not in manifest_digests:manifest_digests[str(mf)]=hashlib.sha256(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  md=manifest_digests[str(mf)];assert tb['manifest_digest']==md
  ident=m.get('base_manifest',m)['input_identity'];fixed=binding[sys+'_binding']
  assert ident['hamiltonian_hash']==fixed['hamiltonian'] and ident['state_hash']==fixed['state']
  if sys=='H6':assert ident['snapshot_sha256']==fixed['input'] and ident['df_receipt_sha256']==fixed['df']
  else:assert ident['file_hashes']['h8_input_snapshot.npz']==fixed['input'] and ident['file_hashes']['df_receipt.json']==fixed['df']
  t=tb['task'];assert t['cell_id']==a['cell_id'] and str(t['seed'])==a['seed'] and str(t['replica'])==a['replica'] and t['phase']==a['phase']
  expected_cell=next(e['cell'] for e in index[sys] if e['cell_id']==a['cell_id']);assert t['cell']==expected_cell
  assert claim['task_digest']==digest(t) and m['source_commit']==claim['source_commit']
  assert m.get('base_manifest',m)['plan']['T']==0.8
  seed_source=None
  if a['cell_id'].startswith('B2_'):
   if sys=='H6':
    f='src/trottertracks/resource_applicability/h6_matched_accounting_v1.py';code=git(a['execution_source'],f).decode();assert 'H6_MATCHED_V1_FRESH_COST/{cell_id}/{phase}/{replica}' in code
    namespace='H6_MATCHED_V1_FRESH_COST';seedtext=f"{namespace}/{a['cell_id']}/{a['phase']}/{a['replica']}";seed_source=dict(kind='SOURCE_INDEPENDENT_FIXED_NAMESPACE',namespace=namespace,implementation_commit=provider['H6'],implementation_path=f)
   else:
    seedcommit=m.get('seed_source_commit',m['source_commit']);assert seedcommit=='cb8a56c6b2a772827275f9200b02638030f33f72'
    f='src/trottertracks/resource_applicability/h8_resource_map_contract_v1.py';code=git(a['execution_source'],f).decode();assert "{NAMESPACE}|{source}|{INPUT_SHA}|{cell}|{phase}|{replica}" in code
    namespace='H8_CORE_RESOURCE_MAP_V1_20261013';seedtext=f"{namespace}|{seedcommit}|{binding['H8_binding']['input']}|{a['cell_id']}|{a['phase']}|{a['replica']}";seed_source=dict(kind='EXPLICIT_MANIFEST_SEED_SOURCE',seed_source_commit=seedcommit,namespace=namespace,implementation_commit=seedcommit,implementation_path=f)
   assert int.from_bytes(hashlib.sha256(seedtext.encode()).digest()[:8],'big')==int(a['seed']),('SEED_BITS',sys,key)
  profile=m.get('profile',m.get('base_manifest',{}).get('profile'));ar=m.get('roundoff_preparation_binding',{});frozen=ar.get('archive_sha256')==archive and profile and profile.get('sha256')==profile_old
  lawmatch=False
  if sys=='H6' and frozen and a['cell_id'].startswith('B2_'):
   tr=read(d/'trajectory.json');lawmatch=canonical(tr['finite_distribution'])==canonical(lawby[(sys,a['cell_id'])]['native_distribution'])
  by[(a['phase'],a['cell_id'])].append(dict(axes=axes,seed_source=seed_source,manifest_sha256=sha(mf),raw_receipts_digest=digest({name:receipts[str(d/name)] for name in ['task_binding.json','worker_claim.json','wrapper_symmetric_directional_cosine.json','wrapper_symmetric_directional_sine.json']}),frozen_archive_profile=bool(frozen),lawmatch=lawmatch))
 for entry in index[sys]:
  key=(sys,entry['scope'],entry['cell_id']);rr=reg[key];val=rr['values'];ev={};measured=entry['paired_groups']>0
  if not entry['cell_id'].startswith('B2_'):val['law_binding']='NOT_APPLICABLE_DETERMINISTIC_NO_RTE_LAW'
  if entry['scope'].endswith('B0I_FAMILY12'):
   ev.update(acquisition=entry['acquisition'],compiler_profile_declared=contract.get('profile_reference'),control=contract.get('control'),provider_source=provider[sys],preparation_scalar_receipt='SEE_EXISTING_PER_CASE_RECEIPTS')
   # Preserved B0I measurements: original per-axis receipts rather than synthetic fingerprints.
   results=read(base/'run01/full_family_results_v1.json');case=next(c for c in results['cases'] if c['prefix']==entry['cell']['prefix'] and c['q']==entry['cell']['q']);assert (case['axes'] is not None)==measured
   if measured:
    assert {ax:x['fingerprint'] for ax,x in case['axes'].items()}==entry['axis_fingerprints'];val['control']='symmetric_directional';val['native_policy']='PF_NO_REUSE_RTE_WITHIN_CLOSED_OCCURRENCE_REUSE';ev['basis_preparation_status']='CURRENT_BLUEPRINT_SCALAR_REFERENCE_PRESENT_NOT_FULL_LINEAGE_CERTIFICATE'
    pre,step=entry['cell']['prefix'],entry['cell']['q'];sf=base/('run01/case_%s_%s_screen.json'%(pre,step))
    if not sf.exists():sf=Path(mapping['pilot_scalar_%s_%s.json'%(pre,step)])
    expected=next(x for x in binding['B0I_saved_scalar_receipts'] if x['system']==sys and x['prefix']==pre and x['q']==step)
    assert sha(sf)==expected['receipt_sha256'];screen=read(sf);pb=screen['preparation_binding']
    actual=pb['current_blueprint']['preparation_hash'] if sys=='H6' else pb['observed_preparation_hash']
    assert actual==expected['observed']==preps[sys]['discard:p'+str(pre)]['preparation_hash']
    val['preparation_binding']=dict(scalar_preparation_receipt_sha256=sha(sf),preparation_hash=actual,theory_identity_receipt_sha256=sha(Q/'preparation_identity_receipt_v1.json'),full_block_basis_lineage_certified=False)
    code=git(case['origin_SOURCE'],'scripts/h6_native_b0i_bridge_v1.py').decode()
    assert 'NATIVE_FINAL_ANCILLA_P_ONCE' in code
    val['scalar_policy']='NATIVE_V5_EXPLICIT_CONTROLLED_GLOBAL_SCALAR_PHASE_NO_PHASE_DROP'
    ev['saved_scalar_and_preparation_receipt_matched']=True
    if case['acquisition']!='REUSE_WITH_DECLARED_LIMITS':
     assert case['origin_SOURCE']==contract['SOURCE'];val['compiler_profile']=contract['expected_profile_digest']
    # The preparation hash and declared profile do not independently establish the complete
    # discarded-block Gaussian lineage. Keep basis_variant and lineage_compatibility unresolved.

  else:
   ps=by[(entry['wrapper_phase'],entry['cell_id'])];assert len(ps)==entry['paired_groups']
   observed_pairs=[p['axes'] for p in ps];observed_pairs=sorted(observed_pairs,key=lambda x:(int(x['cosine']['replica']),x['cosine']['seed']));assert digest(observed_pairs)==entry['wrapper_identity_digest']
   ev['paired_raw_receipts']=[{k:p[k] for k in ('manifest_sha256','raw_receipts_digest','frozen_archive_profile','lawmatch')} for p in ps]
   ev['wrapper_index_digest_independently_reproduced']=True
   if measured:
    val['control']='symmetric_directional';val['native_policy']='PF_NO_REUSE_RTE_WITHIN_CLOSED_OCCURRENCE_REUSE';val['scalar_policy']='NATIVE_V5_EXPLICIT_CONTROLLED_GLOBAL_SCALAR_PHASE_NO_PHASE_DROP';ev['source_semantics_common_files_verified']=True
    if entry['cell_id'].startswith('B2_'):
     seeds={digest(p['seed_source']):p['seed_source'] for p in ps};assert len(seeds)==1;val['dataset_seed_source']=next(iter(seeds.values()));ev['all_saved_seed_bits_verified']=True
    if ps and all(p['frozen_archive_profile'] for p in ps):
     val['compiler_profile']=profile_old;val['basis_variant']='H6_FROZEN_ARCHIVE_'+archive
     root=('discard' if entry['cell']['method']=='B0' else 'faithful')+':p'+str(entry['cell']['prefix']);prep=preps[sys][root]
     val['preparation_binding']=dict(archive_sha256=archive,root=root,identity_receipt_sha256=sha(Q/'preparation_identity_receipt_v1.json'),record_digest=digest(prep))
     if entry['cell_id'].startswith('B2_') and all(p['lawmatch'] for p in ps):
      val['law_binding']=dict(receipt_sha256=sha(Q/'finite_law_receipts_v1.json'),candidate_id=entry['cell_id'],native_distribution_bits_matched_all_pairs=True,fraction_normalization_conditional=True)
      # Six shared library files plus native v5 match; frozen archive/profile same for all paired executions.
      val['lineage_compatibility']='H6_FROZEN_ARCHIVE_OLD_PROFILE_COMMON_NATIVE_V5_METADATA_ONLY'
     elif not entry['cell_id'].startswith('B2_'):val['law_binding']='NOT_APPLICABLE_DETERMINISTIC'
  rr['recovered_evidence']=ev;rr['metadata_resolved']=all(val.get(k) not in (None,[], '') for k in (*['input','hamiltonian','state','df','fragment_order','compiler_profile','basis_variant','control','scalar_policy','native_policy','lineage_compatibility','sample_scope'],*['origin_sources','wrapper_identity','dataset_seed_source','preparation_binding','law_binding']))
  rr['evidence_refs'].append(dict(input='recovered_saved_metadata',digest=digest(ev)))
  rr['interpretation']='CONDITIONAL_SAVED_METADATA_ONLY_NO_NUMERIC_ENVELOPE_CERTIFICATE'
  individual.append(dict(key=list(key),measured=measured,origin_status='PRESERVED_UNMEASURED_NO_ORIGIN_TO_RECOVER' if not entry['execution_sources'] else 'EXISTING_ORIGIN_CONFIRMED',wrapper_status='NOT_ACQUIRED_INELIGIBLE' if not measured else 'MATCHED_EXISTING_AXIS_RECEIPTS',seed_status='NOT_APPLICABLE_DETERMINISTIC' if not entry['cell_id'].startswith('B2_') else 'RECOVERED_FROM_TASK_MANIFEST_SOURCE',metadata_resolved=rr['metadata_resolved'],field_status={k:'BOUND_SAVED_EVIDENCE' if val.get(k) not in (None,[], '') else 'UNRESOLVED' for k in ['compiler_profile','basis_variant','control','scalar_policy','native_policy','preparation_binding','law_binding','lineage_compatibility']}))
(P/'provenance_registry_v2_PRIVATE.json').write_text(json.dumps(registry,indent=2)+'\n');(P/'input_receipts_PRIVATE.json').write_text(json.dumps(receipts,indent=2)+'\n');(P/'entry_reconciliation_PRIVATE.json').write_text(json.dumps(individual,indent=2)+'\n');(P/'source_comparison_PRIVATE.json').write_text(json.dumps(sourceproof,indent=2)+'\n')
print('STATUS',Counter(r['metadata_resolved'] for r in registry),'BY_SCOPE',Counter(tuple(r['key'][:2]) for r in registry if r['metadata_resolved']))
print('FIELDS', {k:sum(r['values'].get(k) not in (None,[],'') for r in registry) for k in ['compiler_profile','basis_variant','control','scalar_policy','native_policy','preparation_binding','law_binding','lineage_compatibility','dataset_seed_source']})
print('PAIRS', {sys:sum(r['paired_groups'] for r in index[sys] if not r['scope'].endswith('B0I_FAMILY12')) for sys in index},'RECEIPTS',len(receipts))
