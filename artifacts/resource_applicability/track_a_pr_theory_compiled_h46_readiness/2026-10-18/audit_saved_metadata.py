"""Metadata/bytes only. Do not evaluate gate counts, costs, shots or residuals."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[4]
PRIVATE=Path('/home/AbeHiromu/projects/pr-theory-compiled-h46-evidence/20261018')
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_pr_theory_compiled_h46_readiness/2026-10-18'
H4='0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41'
H4SOURCE='217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
THEORY='7e41670565d9284da9aded94063be591fdcaa3dd'
V2='68a010b35d35a1500dcc901c480fcce433fac4a8'
H4PATH='artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/'
THEORYPATH='artifacts/resource_applicability/track_a_native_df_theory_evaluation/2026-10-16/'
INDEXPATH='artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/saved_candidate_connection_index_v1.json'
RECONPATH='artifacts/resource_applicability/track_a_v2_provenance_reconciliation/2026-10-16/entry_reconciliation.json'
receipts=[];protected=[];cache={}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def git_bytes(ref,path):
    key=(ref,path)
    if key not in cache:
        value=subprocess.check_output(['git','-C',str(ROOT),'show',ref+':'+path]);cache[key]=value
        blob=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',ref+':'+path],text=True).strip()
        receipts.append(dict(commit=ref,path=path,blob=blob,bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
    return cache[key]

def read_git(ref,path):return json.loads(git_bytes(ref,path))

def private_json(path):
    path=Path(path);value=path.read_bytes();protected.append(dict(path=str(path),sha256=hashlib.sha256(value).hexdigest()))
    return json.loads(value)

def save(name,value):
    (BUNDLE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

BUNDLE.mkdir(parents=True,exist_ok=True)
remote=[]
for item in json.loads((PRIVATE/'remote_available.json').read_text()):
    git_bytes(item['ref'],item['path']);r=next(r for r in receipts if r['commit']==item['ref'] and r['path']==item['path'])
    assert r['blob']==item['blob'] and not item.get('error')
    remote.append(dict(**r,url=item['url'],github_fetch='PASS_BLOB_MATCH'))
save('remote_reference_verification.json',remote)

index=read_git(THEORY,INDEXPATH)['H6']
recon=read_git(V2,RECONPATH)['entries'];by_key={tuple(e['key']):e for e in recon}
pf_final={x['cell_id']:x for x in read_git(V2,'artifacts/resource_applicability/track_a_v2_h6_pf_final_reconciliation/2026-10-16/PF_entry_audit.json')['rows']}
binding=read_git(THEORY,THEORYPATH+'execution_binding_v1.json')['H6_binding']
laws=read_git(THEORY,THEORYPATH+'finite_law_receipts_v1.json')
law_index={x['candidate_id']:x for x in laws if x['system']=='H6'}
# Preserve whole old results as byte references, do not parse numerical tables.
git_bytes(THEORY,THEORYPATH+'theoretical_wrapper_costs_v1.json')
git_bytes(THEORY,THEORYPATH+'same_candidate_saved_cost_connection_v1.json.gz')
ledger=[]
for item in index:
    key=('H6',item['scope'],item['cell_id']);old=by_key[key];c=dict(item['cell'])
    c.setdefault('id',item['cell_id']);c.setdefault('method',item['cell_id'].split('_')[0])
    old=json.loads(json.dumps(old))
    pf=pf_final.get(c['id']) if item['scope']=='H6_EXPLORATION92' else None
    if pf and pf.get('compiler_profile_receipt_bound'):
        old['field_status']['compiler_profile']='BOUND_SAVED_EVIDENCE'
    missing=[k for k,v in old['field_status'].items() if v!='BOUND_SAVED_EVIDENCE']
    if not old['measured']:status='NOT_ACQUIRED'
    elif old['metadata_resolved']:status='CONDITIONALLY_ALIGNED'
    else:status='UNRESOLVED_PROVENANCE'
    for axis in ('cosine','sine'):
        ledger.append(dict(system='H6',scope=item['scope'],cell_id=item['cell_id'],axis=axis,cell=c,T=item['T'],
            Hamiltonian=binding['hamiltonian'],input=binding['input'],state=binding['state'],DF=binding['df'],
            fragment_order=binding['fragment_order'],DF_policy='tol-only1e-8/cutoff0/signed-generation/weighted-projection/rank19',
            theory_models=['A','B'],native_mixed_evaluated=False,theory_provenance=dict(commit=THEORY,path=THEORYPATH+'theoretical_wrapper_costs_v1.json'),
            compiled_provenance=dict(commit=THEORY,path=INDEXPATH,execution_sources=item['execution_sources'],audit_commit=V2),
            compiler_hash=item['compiler_hash'],profile_receipt=('4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75' if old['metadata_resolved'] else pf.get('profile_sha256') if pf else None),
            profile_status=old['field_status']['compiler_profile'],basis_variant_status=old['field_status']['basis_variant'],
            preparation_status=old['field_status']['preparation_binding'],control='symmetric_directional',wrapper='full_hadamard',
            scalar_policy='native-final-ancilla-P; reference-unrotated',P=0,
            sampling_phase=item['wrapper_phase'],sample_size=item['paired_groups'],
            finite_law=(dict(commit=THEORY,path=THEORYPATH+'finite_law_receipts_v1.json',candidate_receipt_digest=digest(law_index[c['id']]),
                             orders=law_index[c['id']]['native_distribution']['orders']) if c['method']=='B2' else 'NOT_APPLICABLE_DETERMINISTIC'),
            wrapper_identity_digest=item.get('wrapper_identity_digest'),metadata_status=status,critical_field_status=old['field_status'],
            missing_evidence=missing+(['FULL_NUMERIC_WRAPPER_ENVELOPE_NOT_VERIFIED'] if status=='CONDITIONALLY_ALIGNED' else []),
            exact_event_alignment=False,pretranspile_payload_available=False,strict_comparable=False,
            new_numeric_evaluation=False,cost_values_used=False))

H4ROOT=Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h48-preflight-v1/runs/h4-cost-01')
manifest=private_json(H4ROOT/'manifest.json')
private_json(H4ROOT/'grant.json');private_json(H4ROOT/'execution_terminal.json')
signals=read_git(H4,H4PATH+'signal_metadata.json')['signals']
csv_rows=list(csv.DictReader(io.StringIO(git_bytes(H4,H4PATH+'wrapper_metrics.csv').decode())))
metadata_fields=('cell_id','phase','replica','seed','axis','control','event_digest','compiler_hash','fingerprint')
identity_rows=[{k:r[k] for k in metadata_fields} for r in csv_rows if r['control']=='symmetric_directional']
assert len(identity_rows)==528
assert len({(r['cell_id'],r['replica'],r['axis']) for r in identity_rows})==528
by_csv={(r['cell_id'],r['replica'],r['axis']):r for r in identity_rows}
event_specimens=[]
for reg in manifest['registrations']:
    task=reg['task'];c=task['cell'];directory=H4ROOT/f"cost_exploration_{reg['index']:04d}"
    claim=private_json(directory/'worker_claim.json');tb=private_json(directory/'task_binding.json')
    assert claim['source_commit']==H4SOURCE and tb['task']==task
    traj=private_json(directory/'trajectory.json')
    for axis in ('cosine','sine'):
        r=private_json(directory/f'wrapper_symmetric_directional_{axis}.json')
        meta={k:str(r[k]) for k in metadata_fields}
        # Original phase is a string; CSV string serialization is exact here.
        assert meta==by_csv[(c['id'],str(task['replica']),axis)]
        assert r['event_digest']==traj['event_digest']
    if c['method']=='B2' and task['replica']==0:
        event_specimens.append(dict(cell_id=c['id'],event_digest=traj['event_digest'],
            event_payload_present=bool(traj['events']),finite_law_receipt_digest=digest(traj['finite_distribution']),
            full_prefix_preparation_payload_bound=False))

snapshot=manifest['snapshot'];metadata=snapshot['metadata']
for signal in signals:
    c=signal['cell'];matching=[r for r in identity_rows if r['cell_id']==c['id']]
    n=len({r['replica'] for r in matching})
    for axis in ('cosine','sine'):
        paired=[r for r in matching if r['axis']==axis];assert len(paired)==n
        ledger.append(dict(system='H4',scope='H4_NATIVE_ANCHOR',cell_id=c['id'],axis=axis,cell=c,T=.8,
            Hamiltonian=metadata.get('hamiltonian_hash'),input=manifest['input_identity']['snapshot_sha256'],
            state=metadata['state_hash'],DF=digest(metadata['fragment_hashes']),fragment_order=metadata['fragment_order'],
            DF_policy='legacy-fixed-rank12; not H6 tol-only policy',theory_models=[],native_mixed_evaluated=False,
            theory_provenance=None,compiled_provenance=dict(commit=H4,path=H4PATH+'wrapper_metrics.csv',execution_source=H4SOURCE),
            compiler_hash=paired[0]['compiler_hash'],profile_receipt=None,profile_status='VERSION_DECLARATION_ONLY',
            basis_variant_status='COST_PREPARATION_FULL_BINDING_NOT_VERIFIED',preparation_status='SNAPSHOT_IDENTITY_ONLY',
            control='symmetric_directional',wrapper='full_hadamard',scalar_policy='native-final-ancilla-P',P=0,
            sampling_phase='fixed_anchor_panel',sample_size=n,finite_law='SAVED_TRAJECTORY_DISTRIBUTION; NO_GLOBAL_THEORY_LAW_EXPORT' if c['method']=='B2' else 'NOT_APPLICABLE_DETERMINISTIC',
            wrapper_identity_digest=digest(paired),metadata_status='MISSING_THEORY_AND_PREPARATION_LINEAGE',
            missing_evidence=['NEW_H4_THEORY_TABLE_REQUIRES_APPROVAL','PREPARED_PREFIX_AND_BASIS_FULL_BINDING','FULL_BINARY_PROFILE_AND_NUMERIC_ENVELOPE'],
            exact_event_alignment=False,pretranspile_payload_available=False,strict_comparable=False,new_numeric_evaluation=False,cost_values_used=False))
assert len(ledger)==296
save('comparability_ledger.json',ledger)
with (BUNDLE/'comparability_ledger.csv').open('w') as stream:
    fields=['system','scope','cell_id','axis','sample_size','metadata_status','strict_comparable','compiler_hash','profile_status','basis_variant_status','missing_evidence']
    w=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');w.writeheader()
    for r in ledger:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],list) else r[k] for k in fields})
counts={s:dict(collections.Counter(r['metadata_status'] for r in ledger if r['system']==s)) for s in ('H4','H6')}
save('metadata_audit.json',dict(axis_entries=296,systems=counts,H4_native_rows=528,H4_ordinary_rows_excluded=4,
    H4_candidate_count=16,H6_scope_entries=132,H6_unique_candidates=104,H6_conditional_scope_entries=25,
    strictly_comparable_candidates=0,new_theory_evaluations=0,new_ratios_residuals_rankings_fits=0,new_sampling_compile_signal=0,
    H4_original_task_wrapper_joins=264,H4_event_specimens=event_specimens,old_V2_NO_GO_preserved=True,
    selection_uses_cost_values=False,full_numeric_envelope_certified=False))
minimum=dict(rule='fixed semantic panel; selected without costs, ratios or residuals',
    H4=['B1_p12_q1_2nd','B1_p12_q1_4th','B2_p3_q1_2nd_r4_K2','B2_p3_q8_2nd_r2_K4'],
    H6_PR=['B2_p10_q8_2nd_r1_K2','B2_p15_q8_2nd_r2_K4'],
    H6_PF=['B1_p19_q8_4th'],H6_scope='H6_EXPLORATION92',
    immediate_conditional_proposal=dict(candidates=2,axes=2,models=['A','B'],metrics=['RZ','CX'],comparison_rows=16,
        new_theory_evaluation=0,new_compile=0,PF_PR_winner_or_shot_reaccounting=False,
        scientific_meaning='metadata-conditional law-expectation versus finite-sample point diagnostic; no population or compiler-only claim'),
    deferred_H4_and_PF_reason='H4 lacks theory/preparation closure; H6 PF lacks independent basis binding. No extra compile justified yet.')
for name in minimum['H6_PR']:
    assert any(r['cell_id']==name and r['scope']=='H6_EXPLORATION92' and r['metadata_status']=='CONDITIONALLY_ALIGNED' for r in ledger)
save('minimum_comparison_set.json',minimum)
save('input_manifest.json',dict(metadata_inputs=receipts,numerical_input_sha256=None,
    approved_numerical_dataset=None,scientific_arrays_read=False,old_result_tables_recomputed=False))
for item in protected:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
save('preservation_receipt.json',dict(checked_own_input_files=len(protected),protected_private_inputs_digest=digest(protected),
    all_read_inputs_unchanged=True,original_theory_and_V2_Git_blobs_unchanged=True,old_grant_ledger_marker_written=False,
    other_worktree_HEAD_or_status_used_as_gate=False,other_jobs_or_shared_environment_modified=False,
    actual_scope='fixed Git objects and declared H4 task/manifest/wrapper/trajectory files only; not global worktree freeze'))
(PRIVATE/'protected_input_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
print(json.dumps(dict(ledger_rows=len(ledger),classification=counts,strict=0,conditional_H6=25,protected=len(protected))))
