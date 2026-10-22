"""Separate approved 63-cell observation acquisition and fixed H6 continuation.

Preparation is metadata-only. Numerical imports occur after a one-shot stage.
Old 0b1886f53 result bundles, grants and STOPs are read-only inputs.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import pf_common_run_contract as c

BASE='0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c'
PREP='artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22'
OLD='artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/H6'
MIG=Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/h246-reconstructed-confirmation-20261022')


def catalog_read(inventory,label):
    e=next(e for e in inventory['source_catalog'] if e['label']==label)
    raw=c.gitbytes(e['commit'],e['path']);c.require(hashlib.sha256(raw).hexdigest()==e['sha256'],'ORIGINAL_GIT_METADATA')
    return json.loads(gzip.decompress(raw) if e['path'].endswith('.gz') else raw)


def old_B0I_inventory(inventory):
    original=catalog_read(inventory,'h6_signals');records={}
    for r in original['signals']:
        if r['cell']['method']=='B0I':records[c.cid(c.norm('H6',r['cell']))]=r
    return records


def common_template(private):
    p=Path(private).resolve();c.require(not p.exists(),'FRESH_PACK_REQUIRED')
    prior=MIG.parent/'pf-common-execution-20261022/H4/plan_APPROVED.json';old=c.read(prior)
    inv=c.read(c.ROOT/PREP/'inventory.json');bind=next(r['binding'] for r in inv['rows'] if r['system']=='H6')
    inp=dict(snapshot=MIG/'recovered/H6/h6_input_snapshot.npz',receipt=MIG/'recovered/H6/snapshot_receipt.json',
             df_receipt=MIG/'recovered/H6/df_receipt.json',reference=MIG/'forward/H6/run01/reference.npz',reference_receipt=MIG/'forward/H6/run01/receipt.json')
    from pf_common_snapshot_receipt import check_original_snapshot
    check_original_snapshot(inp['snapshot'],c.read(inp['receipt']),expected_sha256=bind['snapshot'])
    c.require(c.read(MIG/'forward/H6/run01/completion.json')['status']=='REFERENCE_AND_FROZEN_SHOT_APPLICABILITY_PASS','PREVIOUS_FORWARD_PASS')
    closure=[]
    for e in old['closure']:
        if e['role'].startswith('SOURCE:'):
            rel=e['role'][7:];closure.append(c.seal(c.ROOT/rel,e['role']))
        elif e['role'] in ('pinned_provider','loaded_runtime_identity','fixed_python_binary','compiler_array_runtime_binary','saved_PR_costs'):
            c.require(c.sha(e['path'])==e['sha256'],'PRIOR_PROVIDER_RUNTIME_SHA');closure.append(e)
    for rel in ('scripts/h6_pf_metadata_completion.py','tests/test_h6_pf_metadata_completion.py','scripts/audit_pf_common_run.py','scripts/pf_common_snapshot_receipt.py',
                'tests/test_pf_common_snapshot_receipt.py'):
        closure.append(c.seal(c.ROOT/rel,'SOURCE:'+rel))
    # Explicit original compact/forward/native numerical definitions are in the pinned provider closure.
    closure.extend(c.seal(q,'scientific_input:'+k) for k,q in inp.items())
    closure.extend(c.seal(c.ROOT/OLD/name,'old_H6_STOP_metadata') for name in ('H6_INPUT_STOP.json','metadata_recovery.json','recovered_original_signal_records.json'))
    plan=dict(SOURCE=c.source(),base_commit=BASE,system='H6',private_root=str(p),provider=str(MIG/'provider'),binding=bind,
        input_paths={k:str(q) for k,q in inp.items()},closure=closure,retry=False,resume=False,T=.8,P=0,
        native=c.NATIVE,compiler=c.COMPILER,approval=dict(execution_authorized=True,no_intermediate_reapproval=True,
        scope='User explicitly approved separate 63-cell missing observation calculation and subsequent original 276-cell H6 acquisition through fixed PR/PF comparison; no new PR sampling or pool expansion.'),
        original_H6_STOP_preserved=True,existing_PR_samples_changed=False,final_mandatory_STOP=True)
    p.mkdir(parents=True)
    return p,plan,inv


def finish_plan(private,plan,files):
    plan['closure'].extend(c.seal(private/name,'fixed_new_metadata') for name in files)
    plan=json.loads(c.encoded(plan))
    c.save(private/'plan_APPROVED.json',plan)
    c.save(private/'approval_receipt.json',dict(plan['approval'],SOURCE=plan['SOURCE'],plan_sha256=c.sha(private/'plan_APPROVED.json')))
    c.verify(plan);return dict(status='SEALED_APPROVED',SOURCE=plan['SOURCE'],system='H6',private=str(private))


def prepare_metadata(private):
    p,plan,inv=common_template(private)
    missing=c.read(c.ROOT/OLD/'metadata_recovery.json')['unrecovered_old_metadata'];rows=[r['candidate'] for r in missing]
    c.require(len(rows)==63 and {c.cid(r) for r in rows}==set(c.read(c.ROOT/OLD/'H6_INPUT_STOP.json')['unresolved_ids']),'EXACT_ORIGINAL_63')
    old=old_B0I_inventory(inv);selected={c.cid(r):old[c.cid(r)] for r in rows if c.cid(r) in old}
    c.require(len(selected)==52,'EXACT_52_OLD_SIGNALS')
    # Pure stdlib original merged S2/Yoshida schedule; no molecular import.
    import sys
    sys.path.insert(0,str(MIG/'provider/src'))
    from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule as schedule
    tasks=[]
    for r in rows:
        cell=dict(r,method='B0',id=c.cid(r),r=None,K=None,R=None)
        seq=schedule(cell,T=.8)['ordinary_one_outer_step'];n=len(seq)*r['q']
        tasks.append(dict(candidate=r,cell=cell,schedule=seq,primitive_actions_per_path=n))
    actions=sum(t['primitive_actions_per_path'] for t in tasks);outers=sum(r['q'] for r in rows)
    caps=dict(snapshot_restore=1,saved_reference_read=1,occupation_oracle=1,full_preparation=1,prefix_preparation=0,
        scalar_coefficients=len({r['prefix'] for r in rows}),forward_sector_matrix=16,forward_sector_eigh=16,metadata_candidate=63,
        native_corrected_signal=63,native_raw_signal=63,independent_forward=63,
        native_primitive_build=2*actions,native_primitive_action=2*actions,forward_block_action=actions,
        native_scalar_action=2*outers,forward_scalar_action=outers,final_B0I_vector_rotation=189,
        vector_closure=63,old_signal_comparison=52,compile=0,sampling=0,new_full_H_reference=0)
    plan.update(metadata_tasks=tasks,metadata_caps=caps,rows=rows,old_missing_raw_ids=[c.cid(r) for r in rows if c.cid(r) not in selected],
        observation_definition=dict(maximum_intermediate_norm='Maximum over initial 1, every original merged primitive and outer scalar observation of native corrected/raw and independent forward, plus the final B0I vector scalar rotation; no rescaling.',
          raw_corrected_vector_closure='Explicit norm of separately executed rotated raw minus exp(-log_B)*rotated corrected vector. Deterministic log_B=0. No inferred or inserted zero.',
          old_B0I_signal='Original B0 native corrected with scalar per outer, followed by scalar tail overlay exp(-i*.8*C). Full-H reference unchanged.',
          old_52_signal_tolerance=1e-9,independent_forward_vector_tolerance=1e-9,primitive_sector_leakage_tolerance=1e-12),
        metadata_budget_separate_from_original_102=True)
    account=catalog_read(inv,'h6_account');floors={}
    for row in account['rows']:
        if row.get('factor')==1.0 and row.get('bias_rule')=='signed_signal' and row['cell']['method']=='B0I':
            identity=c.cid(c.norm('H6',row['cell']))
            if identity in {c.cid(r) for r in rows}:floors[identity]=max(floors.get(identity,0.),*(x['allowance'] for x in row['accounting']['axes'].values()))
    c.require(len(floors)==63,'ALL_SAVED_U_FLOORS')
    c.save(p/'old_u_floors.json',floors)
    c.save(p/'old52_signals.json',selected)
    return finish_plan(p,plan,['old52_signals.json','old_u_floors.json'])|dict(caps=caps,new_raw_candidates=len(plan['old_missing_raw_ids']))


def observe_paths(state,actions,forward,cell,scalar,tail,*,on_action=lambda path:None):
    """Original compact/forward recurrences, with explicit second raw path."""
    import numpy as np
    from trottertracks.resource_applicability.h6_matched_port_v1 import compact_signal
    from trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 import recurrence_paths
    from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
    def counted(fn,path):
        def apply(v,t):on_action(path);return fn(v,t)
        return apply
    actual=[]
    for name in ('corrected','raw'):
        budget=ActionBudget(0,100000)
        actual.append(compact_signal(state,[counted(a,name) for a in actions],None,cell=cell,T=.8,scalar=scalar,lambda_r=0.,budget=budget))
    independent=recurrence_paths(state,[counted(a,'forward') for a in forward],None,cell=cell,T=.8,scalar=scalar,lambda_r=0.,maximum_tail=0)
    phase=np.exp(-.8j*tail)
    corrected=phase*actual[0]['vectors']['corrected'];raw=phase*actual[1]['vectors']['corrected'];other=phase*independent['vectors']['corrected']
    maximum=max(*(x['intermediate_norm_max'] for x in [*actual,independent]),*(float(np.linalg.norm(v)) for v in (corrected,raw,other)))
    closure=float(np.linalg.norm(raw-np.exp(-0.)*corrected));error=float(np.linalg.norm(corrected-other))
    return dict(maximum_intermediate_norm=maximum,raw_corrected_vector_closure=closure,forward_vector_error=error,
        corrected=corrected,raw=raw,forward=other,phase=complex(phase),stages=actual[0]['stage_count'])


_CTX=None
_STAGE=None
_OLD52=None

def metadata_one(task):
    import numpy as np,io,os,gzip
    from trottertracks.resource_applicability.ax2a_state_action import project_primitive_checked
    from trottertracks.resource_applicability.ax2b_native_df_v5 import make_native_block_action,block_instruction_bound
    from h46_new2_core import complex_record
    ctx,s=_CTX,_STAGE;r=task['candidate'];identity=c.cid(r);p=r['prefix'];out=s.out/identity;out.mkdir()
    try:
        for kind in ('metadata_candidate','native_corrected_signal','native_raw_signal','independent_forward','vector_closure'):
            s.take(kind,identity)
        n=task['primitive_actions_per_path']
        for kind,count in [('native_primitive_build',2*n),('native_primitive_action',2*n),('forward_block_action',n),
                           ('native_scalar_action',2*r['q']),('forward_scalar_action',r['q']),('final_B0I_vector_rotation',3)]:s.take(kind,identity,count)
        actions=[]
        for block in ctx.fullprep.deterministic_blocks[:p+1]:
            a=make_native_block_action(block,max_instructions=block_instruction_bound(block,'UNCONTROLLED'))
            actions.append(lambda v,t,a=a:project_primitive_checked(lambda x:a(x,t),v,ctx.metadata_qindices,1<<ctx.ham.n_qubits,leakage_tolerance=1e-12))
        v=observe_paths(ctx.metadata_state,actions,ctx.metadata_forward[:p+1],task['cell'],ctx.scalar,ctx.scalars[p])
        c.require(np.isfinite(v['maximum_intermediate_norm']) and v['forward_vector_error']<1e-9 and v['raw_corrected_vector_closure']<1e-9,'METADATA_NUMERIC_GATE')
        z=complex(np.vdot(ctx.metadata_state,v['corrected']));ref=complex(np.vdot(ctx.oracle.initial,ctx.reference))
        old=_OLD52.get(identity);comparison=None
        if old:
            s.take('old_signal_comparison',identity)
            oz=complex(old['signals']['corrected']['real'],old['signals']['corrected']['imag']);oref=complex(old['reference']['real'],old['reference']['imag'])
            bias=old['error_decomposition']['total_signed'];oldbias=complex(bias['real'],bias['imag'])
            comparison=dict(corrected_error=abs(z-oz),reference_error=abs(ref-oref),signed_bias_error=abs(z-ref-oldbias),
                scalar_error=abs(ctx.scalars[p]-old['scalar_receipt']['C']),strictly_below=1e-9)
            c.require(max(comparison[k] for k in ('corrected_error','reference_error','signed_bias_error','scalar_error'))<1e-9,'OLD52_SIGNAL_INCONSISTENCY')
            # Preserve old corrected/reference/bias and u; attach new observations only.
            signal=dict(corrected=copy.deepcopy(old['signals']['corrected']),raw=complex_record(np.vdot(ctx.metadata_state,v['raw'])),
                reference=copy.deepcopy(old['reference']),total_signed_bias=copy.deepcopy(bias),log_B=old['log_B'])
            old_u=max(old['allowance']['real'],old['allowance']['imag']);old_margin=old['allowance']['log_B_margin']
        else:
            signal=dict(corrected=complex_record(z),raw=complex_record(np.vdot(ctx.metadata_state,v['raw'])),reference=complex_record(ref),total_signed_bias=complex_record(z-ref),log_B=0.)
            # The old original combined B0 allowance is available for this newly measured overlay.
            original=ctx.saved_B0[identity.replace('_B0I_','_B0_')]
            old_u=max(original['old_u'],ctx.old_u_floors[identity]);old_margin=original['old_margin']
        signal['allowance_evidence']=dict(maximum_intermediate_norm=v['maximum_intermediate_norm'],raw_corrected_vector_closure=v['raw_corrected_vector_closure'],kind='NEW_ORIGINAL_RECURRENCE_OBSERVATION_NOT_CERTIFICATE')
        record=dict(candidate=r,signal=signal,old_u=old_u,old_margin=old_margin,provenance='NEW_METADATA_OLD52_SIGNAL_PRESERVED' if old else 'NEW11_SIGNAL_AND_METADATA',old_signal_comparison=comparison,
            observed=dict(maximum_intermediate_norm=v['maximum_intermediate_norm'],raw_corrected_vector_closure=v['raw_corrected_vector_closure'],forward_vector_error=v['forward_vector_error'],stages=v['stages']),
            SOURCE=s.plan['SOURCE'],original_restored=False,tail_identity=ctx.scalars[p],reference_rotated=False)
        f=io.BytesIO();np.savez(f,corrected=v['corrected'],raw=v['raw'],forward=v['forward']);raw=f.getvalue();blob=gzip.compress(raw,compresslevel=1,mtime=0)
        c.require(gzip.decompress(blob)==raw,'METADATA_VECTOR_BYTES');(out/'vectors.npz.gz').write_bytes(blob)
        record['vector_archive']=dict(path='vectors.npz.gz',original_sha256=hashlib.sha256(raw).hexdigest(),archive_sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
        c.save(out/'record.json',record);return identity,record
    except BaseException as e:
        import traceback
        c.save(out/'STOP.json',dict(reason=str(e),traceback=traceback.format_exc(),retry=False));raise


def run_metadata(private):
    global _CTX,_STAGE,_OLD52
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    plan=c.read(Path(private)/'plan_APPROVED.json');s=c.Stage(private,'METADATA',plan['metadata_caps']);_STAGE=s
    try:
        import pf_common_run_native as native
        import numpy as np
        from scipy.linalg import eigh
        from trottertracks.resource_applicability.ax2b_independent_reference import occupation_df_matrix
        # install before provider imports (normally entry ensures it as well).
        ctx=native.restore(s,[])
        from trotterlib.df_hamiltonian import PhysicalSector
        sector=PhysicalSector.spin_sector(n_qubits=12,nelec_alpha=3,nelec_beta=3)
        with np.load(plan['input_paths']['snapshot'],allow_pickle=False) as z:ctx.metadata_state=z['sector_state_vector'].copy()
        ctx.metadata_qindices=np.array([int(format(int(i),'012b')[::-1],2) for i in sector.basis_indices])
        ctx.metadata_forward=[]
        for i in range(16):
            s.take('forward_sector_matrix',str(i))
            matrix=occupation_df_matrix(0.,ctx.ham.one_body if i==0 else np.zeros_like(ctx.ham.one_body),() if i==0 else (ctx.ham.lambdas[i-1],),() if i==0 else (ctx.ham.g_matrices[i-1],),sector.basis_indices)
            s.take('forward_sector_eigh',str(i));w,v=eigh(np.asarray(matrix,complex));ctx.metadata_forward.append(lambda x,t,w=w,v=v:v@(np.exp(-1j*t*w)*(v.conj().T@x)))
        from h46_new2_core import matrix_only_tail_coefficients
        for p in sorted({r['candidate']['prefix'] for r in plan['metadata_tasks']}):
            s.take('scalar_coefficients',str(p));ctx.scalars[p]=matrix_only_tail_coefficients(ctx.ham,p)[0]
        ctx.saved_B0=c.read(c.ROOT/OLD/'recovered_original_signal_records.json');ctx.old_u_floors=c.read(Path(private)/'old_u_floors.json');_CTX=ctx;_OLD52=c.read(Path(private)/'old52_signals.json')
        obs=c.resources();n=c.workers(obs,63,2*2**30,64*2**20);c.require(n>0,'AVAILABLE_METADATA_RESOURCES')
        c.save(s.root/'resource_receipt.json',dict(observation=obs,workers=n,nested_BLAS_threads=1,shared_settings_changed=False))
        records={}
        with ProcessPoolExecutor(max_workers=n,mp_context=multiprocessing.get_context('fork')) as pool:
            futures=[pool.submit(metadata_one,t) for t in plan['metadata_tasks']];processes=list(pool._processes.values())
            for f in as_completed(futures):
                identity,value=f.result();records[identity]=value
        c.require(all(p.exitcode==0 and not p.is_alive() for p in processes),'METADATA_WORKER_EXIT_REAP')
        c.save(s.root/'worker_exit_receipt.json',dict(workers=[dict(pid=p.pid,exit_code=p.exitcode,reaped=True) for p in processes],remaining_owned_workers=0,same_parent_join=True))
        c.require(set(records)=={c.cid(r) for r in plan['rows']},'EXACT_63_COMPLETENESS')
        c.save(s.out/'all_metadata_records.json',records)
        return s.finish(dict(records=63,old_signal_checks=52,new_raw_signal_candidates=11,metadata_separate_budget=True))
    except BaseException as e:s.stop(e);raise


def prepare_common(private,metadata_private):
    p,plan,inv=common_template(private);meta=Path(metadata_private);proof=c.read(meta/'METADATA/completion.json')
    c.require(proof['status']=='PASS' and proof['records']==63 and proof['old_signal_checks']==52,'METADATA_STAGE_PASS')
    records=c.read(c.ROOT/OLD/'recovered_original_signal_records.json');records.update(c.read(meta/'METADATA/output/all_metadata_records.json'))
    c.require(len(records)==174,'174_SAVED_AND_REOBSERVED')
    entries=[r for r in inv['rows'] if r['system']=='H6'];rows=[{k:r[k] for k in ('system','method','prefix','q','formula')} for r in entries]
    missing=[r for r in rows if c.cid(r) not in records];base=[r for r in missing if r['method']=='B0'];overlays=[r for r in missing if r['method']=='B0I']
    c.require(len(missing)==102 and len(base)==len(overlays)==51,'ORIGINAL_102_ONLY')
    old=c.read(c.ROOT/PREP/'execution_contract_UNAPPROVED.json');pool={r['id'] for r in old['compile_candidate_pool'] if r['candidate']['system']=='H6'}
    prices={};identity=[]
    for r in entries:
        accepted=[e for e in r['cost_evidence'] if e['common_native_identity']]
        if accepted:
            axes={}
            for e in accepted:
                if 'axes' in e:
                    for a in e['axes']:axes[a['axis']]=a['metrics']['rz_count']
                else:axes[e['axis']]=e['receipt']['rz_count']
            c.require(set(axes)=={'cosine','sine'},'SAME_NATIVE_PAIRED')
            prices[r['id']]=dict(cosine=axes['cosine'],sine=axes['sine'],kind='SAVED_COMMON_NATIVE',proof=accepted)
        elif r['any_direct_cost_pair']:identity.append(dict(id=r['id'],status='NOT_REUSED',reason='Legacy native/preparation/provider/runtime stream identity unproved; same options hash insufficient',evidence=r['cost_evidence']))
    c.require(len(prices)==36 and len(pool)==224,'ORIGINAL_NATIVE_POOL')
    # Same expm matrices are reused verbatim; operation order and vector matvecs stay identical.
    import runpy
    # Coefficients and schedule module only requires numpy/provider at import. This is metadata-only schedule arithmetic.
    from h46_new2_core import weights
    keys={(i,float(.8*w/r['q']).hex()) for r in base for w in weights(r['formula']) for i in range(r['prefix']+1)}
    actions=sum(2*(r['prefix']+1)*c.PIECES[r['formula']]*r['q'] for r in base)
    caps=dict(snapshot_restore=1,saved_reference_read=1,occupation_oracle=1,full_preparation=1,prefix_preparation=9,
      base_signal=51,analytic_overlay=51,independent_forward=51,spectral_piece=1419,spectral_block_action=30588,forward_block_action=30588,
      forward_expm=len(keys),new_full_H_reference=0,sampling=0,compile=0)
    c.require(actions==30588,'ORIGINAL_INTERNAL_SIGNAL_CAPS')
    plan.update(rows=rows,signal_base=base,signal_overlays=overlays,signal_caps=caps,qualification_caps=dict(endpoint=1380),
        compile_pool=sorted(pool),max_compile_candidates=224,cost_cap_template=old['systems']['H6']['compile_caps'],
        metadata_completion=str(meta/'METADATA/completion.json'),forward_propagator_cache=True,metadata_budget_separate=True,H2_phase_policy=None)
    plan['closure'].extend(c.seal(meta/name,'new_metadata_stage_PASS') for name in ('plan_APPROVED.json','METADATA/completion.json','METADATA/output/all_metadata_records.json'))
    c.save(p/'saved_signals.json',records);c.save(p/'saved_prices.json',prices)
    c.save(p/'cost_identity_audit.json',dict(accepted_same_native_pairs=36,legacy_candidates=identity,legacy_equal_options_not_accepted=True,new_acquisition_allowed_only_in_pool=sorted(pool)))
    return finish_plan(p,plan,['saved_signals.json','saved_prices.json','cost_identity_audit.json'])|dict(caps=caps,saved_same_native_prices=36)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare-metadata','metadata','prepare-common']);p.add_argument('--private',required=True);p.add_argument('--metadata-private')
    a=p.parse_args()
    if a.mode=='prepare-metadata':result=prepare_metadata(a.private)
    elif a.mode=='prepare-common':
        # Metadata schedule arithmetic only. Provider path is explicit, no molecular files loaded.
        import sys;sys.path.insert(0,str(MIG/'provider/src'));result=prepare_common(a.private,a.metadata_private)
    else:
        import sys;sys.path.insert(0,str(MIG/'provider/src'));result=run_metadata(a.private)
    print(c.encoded(result).decode())
