"""Prepare new approved separate-system packs from fixed saved metadata."""
import argparse
import gzip
from pathlib import Path
import subprocess
import sys
import pf_common_run_contract as c

B='artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22'
MIG=Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/h246-reconstructed-confirmation-20261022')
NEW=('scripts/pf_common_run_contract.py','scripts/prepare_pf_common_run.py','scripts/pf_common_run_native.py',
     'scripts/run_pf_common_run.py','tests/test_pf_common_run.py')

def old_signal(raw,provenance):
    if 'signal' in raw:return dict(signal=raw['signal'],old_u=raw.get('empirical_eligibility',{}).get('u',0.),
        old_margin=raw.get('empirical_eligibility',{}).get('log_B_margin',0.),provenance=provenance)
    if 'allowance_evidence' in raw and 'total_signed_bias' in raw:
        return dict(signal=raw,old_u=0.,old_margin=0.,provenance=provenance)
    a=raw['allowance']
    require_fields=('intermediate_norm_max','B_raw_corrected_difference')
    c.require(all(k in raw for k in require_fields),'MISSING_ORIGINAL_ALLOWANCE_OBSERVATIONS')
    sig=dict(reference=raw['reference'],corrected=raw['signals']['corrected'],raw=raw['signals']['corrected'],
        log_B=raw['log_B'],total_signed_bias=raw['error_decomposition']['total_signed'],
        allowance_evidence=dict(maximum_intermediate_norm=raw['intermediate_norm_max'],
            raw_corrected_vector_closure=raw['B_raw_corrected_difference']))
    return dict(signal=sig,old_u=max(a['real'],a['imag']),old_margin=a['log_B_margin'],provenance=provenance)

def prepare(system,private):
    private=Path(private).resolve();c.require(not private.exists(),'NEW_SYSTEM_PACK')
    private.mkdir(parents=True)
    inventory=c.read(c.ROOT/B/'inventory.json');catalog={e['label']:e for e in inventory['source_catalog']};cache={}
    def saved(label):
        if label not in cache:
            e=catalog[label];raw=c.gitbytes(e['commit'],e['path'])
            import hashlib
            c.require(hashlib.sha256(raw).hexdigest()==e['sha256'],'FIXED_ORIGINAL_METADATA')
            import json
            cache[label]=json.loads(gzip.decompress(raw) if e['path'].endswith('.gz') else raw)
        return cache[label]
    provenance=c.read(c.ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/input_provenance.json')
    binding=next(x['binding'] for x in provenance['input_metadata'] if x['system']==system)
    rows=[];records={};prices={};reuse=[]
    if system=='H2':
        import json
        d=json.loads(subprocess.check_output(['git','cat-file','blob','2b534c7425af6d67bc92c7423a422ac368d9ad07'],cwd=c.ROOT))
        for field in ('hamiltonian','state','snapshot','compiler'):c.require(d['binding'][field]==binding[field],'H2_D70_PR_BINDING:'+field)
        for f in c.FORMULAS:
            for q in (1,2,4,8):
                for mode,prefix in [('FULL',3)]+[(m,p) for p in (0,1,2) for m in ('B0','B0I')]:
                    rows.append(dict(system=system,method=mode,prefix=prefix,q=q,formula=f))
        for raw in d['signals']:
            row=c.norm(system,raw['cell'])
            if row['q']>8 or (row['prefix']==0 and row['formula']=='4th'):continue
            records[c.cid(row)]=old_signal(raw,'H2_D70_blob_2b534c7425af6d67bc92c7423a422ac368d9ad07')
        pool={c.cid(r) for r in rows}
    else:
        entries=[r for r in inventory['rows'] if r['system']==system]
        rows=[{k:r[k] for k in ('system','method','prefix','q','formula')} for r in entries]
        for entry in entries:
            identity=c.cid(entry)
            ev=[e for e in entry['signal_evidence'] if e['raw']]
            if ev:
                e=ev[0];doc=saved(e['source']);raw=doc
                for step in e['pointer'].strip('/').split('/'):
                    if step:raw=raw[int(step)] if isinstance(raw,list) else raw[step]
                records[identity]=old_signal(raw,e)
            accepted=[x for x in entry['cost_evidence'] if x['common_native_identity']]
            if accepted:
                axes={};proof=[]
                for ev in accepted:
                    proof.append(ev)
                    if 'axes' in ev:
                        for ax in ev['axes']:axes[ax['axis']]=ax['metrics']['rz_count']
                    else:axes[ev['axis']]=ev['receipt']['rz_count']
                c.require(set(axes)=={'cosine','sine'},'SAVED_NATIVE_PAIRED_COST')
                prices[identity]=dict(cosine=axes['cosine'],sine=axes['sine'],kind='SAVED_COMMON_NATIVE',proof=proof)
            if not accepted and entry['any_direct_cost_pair']:
                reuse.append(dict(id=identity,status='NOT_REUSED',reason='Legacy Gaussian/native stream differs or historical runtime/preparation closure not identical; options hash alone insufficient',evidence=entry['cost_evidence']))
        pool={t['id'] for t in c.read(c.ROOT/B/'execution_contract_UNAPPROVED.json')['compile_candidate_pool'] if t['candidate']['system']==system}
    c.require(len(rows)==(84 if system=='H2' else 228 if system=='H4' else 276),'FIXED_CANDIDATE_COUNT')
    missing=[r for r in rows if c.cid(r) not in records]
    base=[r for r in missing if r['method']!='B0I'];overlays=[r for r in missing if r['method']=='B0I']
    if system=='H6':
        # Immutable old summary recovers the B0/full part of the former 28.
        import json
        raw=subprocess.check_output(['git','cat-file','blob','9bc570495d256d55868ddd658fff57e8a7c55669'],cwd=c.ROOT)
        doc=json.loads(raw)
        for r in doc['signals']:
            row=c.norm(system,r['cell']) if r['cell']['method']!='B2' else None
            if row and c.cid(row) not in records:records[c.cid(row)]=old_signal(r,'H6_old_summary_blob_9bc570495d256d55868ddd658fff57e8a7c55669')
        old_missing=[r for r in entries if r['signal_status']=='SAVED_ENDPOINT_ONLY_RAW_UNRECOVERED' and c.cid(r) not in records]
        c.save(private/'metadata_recovery.json',dict(recovered_old_candidates=28-len(old_missing),unrecovered=old_missing,
            original_summary_blob='9bc570495d256d55868ddd658fff57e8a7c55669',extra_signal_regeneration=0))
        if old_missing:
            c.save(private/'H6_INPUT_STOP.json',dict(status='MANDATORY_STOP',reason='B0I original norm/closure metadata absent; saved overlay source returns no such fields. No reconstruction by inference authorized.',unrecovered_ids=[r['id'] for r in old_missing],scientific_calls=0,old_metadata_preserved=True))
            c.save(private/'recovered_signals.json',records)
            return dict(system=system,status='INPUT_METADATA_STOP',remaining=len(old_missing))
    cap_base=48 if system=='H2' else 24 if system=='H4' else 51
    c.require(len(base)<=cap_base and len(overlays)<=(36 if system=='H2' else cap_base),'FIXED_SIGNAL_BUDGET')
    c.save(private/'saved_signals.json',records);c.save(private/'saved_prices.json',prices)
    c.save(private/'cost_identity_audit.json',dict(accepted_same_native_pairs=len(prices),legacy_candidates= reuse,
        new_acquisition_allowed_only_in_pool=sorted(pool),legacy_equal_options_not_accepted=True))
    inputs=MIG/'generation'/system/'run01';forward=MIG/'forward'/system/'run01'
    paths=dict(snapshot=inputs/'tol_only_snapshot.npz',receipt=inputs/'tol_only_receipt.json',comparison=inputs/'comparison.json',
               reference=forward/'reference.npz',reference_receipt=forward/'receipt.json')
    comparison=c.read(paths['comparison']);c.require(all(comparison['array_hash_comparisons'].values()) and comparison['same_hamiltonian_fingerprint'] and comparison['same_state_fingerprint'],'SCIENTIFIC_ARRAY_EQUIVALENCE')
    prior_closure=c.read(MIG/'confirmation_run02/H2_CONFIRMATION/config_APPROVED_PRIVATE.json')['sealed_files']
    full=[e for e in prior_closure
          if e['role'] in ('pinned_provider','loaded_runtime_identity','fixed_python_binary','compiler_array_runtime_binary')]
    closure=[]
    for e in full:
        c.require(c.sha(e['path'])==e['sha256'],'SUCCESSFUL_PRIOR_RUNTIME_SOURCE_CLOSURE')
        closure.append(e)
    for e in prior_closure:
        if e['role'].startswith('scientific_SOURCE:'):
            rel=e['role'].split(':',1)[1]
            # The previously executed numerical closure remains immutable.
            if rel not in ('scripts/h246_reconstructed_contract.py','scripts/run_h246_reconstructed.py'):
                path=c.ROOT/rel;c.require(c.sha(path)==e['sha256'],'UNCHANGED_NUMERICAL_SOURCE:'+rel)
                closure.append(c.seal(path,'SOURCE:'+rel))
    for p in c.ROOT.joinpath('scripts').glob('*.py'):
        # Keep numerical dependencies from the original 68-file closure, plus new runner.
        if p.name in {x.split('/')[-1] for x in NEW} or p.name.startswith(('h46_m5','h46_new2','h2_m5','d0_frozen','h246_saved_cost')):
            closure.append(c.seal(p,'SOURCE:'+str(p.relative_to(c.ROOT))))
    closure.extend(c.seal(c.ROOT/p,'SOURCE:'+p) for p in NEW if not p.startswith('scripts/'))
    closure.append(c.seal(c.ROOT/'src/trotterlib/h46_m5_plan.py','SOURCE:src/trotterlib/h46_m5_plan.py'))
    closure.append(c.seal(c.ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/candidate_precision_costs_1000.csv','saved_PR_costs'))
    closure.extend(c.seal(p,'scientific_input:'+k) for k,p in paths.items())
    closure.extend(c.seal(private/p,'fixed_new_metadata') for p in ('saved_signals.json','saved_prices.json','cost_identity_audit.json'))
    prefixes=sorted({r['prefix'] for r in base if r['method']!='FULL'})
    piece=lambda rows:sum(c.PIECES[r['formula']]*r['q'] for r in rows)
    actions=sum(2*(r['prefix']+1)*c.PIECES[r['formula']]*r['q'] for r in base)
    signal_caps=dict(snapshot_restore=1,saved_reference_read=1,full_preparation=1,prefix_preparation=len(prefixes),occupation_oracle=1,
        base_signal=len(base),analytic_overlay=len(overlays),independent_forward=len(base),spectral_piece=piece(base),
        spectral_block_action=actions,forward_block_action=actions,new_full_H_reference=0,sampling=0,compile=0)
    if system!='H2':
        old=c.read(c.ROOT/B/'execution_contract_UNAPPROVED.json')['systems'][system]
        c.require(piece(base)==old['new_signal_signed_S2_pieces'] and actions==old['signal_kernel_caps']['spectral_block_actions'],'ORIGINAL_INTERNAL_SIGNAL_CAPS')
    plan=dict(SOURCE=c.source(),system=system,private_root=str(private),base_commit=c.BASE,rows=rows,signal_base=base,signal_overlays=overlays,
        compile_pool=sorted(pool),signal_caps=signal_caps,qualification_caps=dict(endpoint=5*len(rows)),
        max_compile_candidates=84 if system=='H2' else 149 if system=='H4' else 224,
        cost_cap_template=None if system=='H2' else c.read(c.ROOT/B/'execution_contract_UNAPPROVED.json')['systems'][system]['compile_caps'],
        binding=binding,input_paths={k:str(v) for k,v in paths.items()},provider=str(MIG/'provider'),closure=closure,
        approval=dict(execution_authorized=True,no_intermediate_reapproval=True,scope='User approved common deterministic PF acquisition through PR/PF total-cost comparison including new H2 independent track. Same-source/grant retry or resume forbidden.',
            common_policy_adopted=True,additional_signal_regeneration_outside_budget=False),retry=False,resume=False,
        T=.8,P=0,compiler=c.COMPILER,native=c.NATIVE,existing_PR_samples_changed=False,
        final_mandatory_STOP=True,H2_phase_policy='H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1' if system=='H2' else None)
    c.save(private/'plan_APPROVED.json',plan)
    c.save(private/'approval_receipt.json',dict(plan['approval'],SOURCE=plan['SOURCE'],plan_sha256=c.sha(private/'plan_APPROVED.json')))
    c.verify(plan)
    return dict(system=system,status='SEALED_APPROVED',existing_signals=len(records),new_base=len(base),new_overlay=len(overlays),same_native_prices=len(prices),signal_caps=signal_caps)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--system',required=True,choices=['H2','H4','H6']);p.add_argument('--private',required=True)
    a=p.parse_args();print(c.encoded(prepare(a.system,a.private)).decode())
