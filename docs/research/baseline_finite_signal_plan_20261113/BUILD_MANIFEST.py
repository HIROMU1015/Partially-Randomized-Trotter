"""Metadata-only proposal. No H4 input file, scientific numerical call or test.

Read pinned public JSON/CSV, generate B IDs and C requests, copy SOURCE facts.
Never read a snapshot/context file from its recorded local filesystem path.
"""
from copy import deepcopy
import csv
import gzip
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_optimization_pipeline as I

BASE='5cc7c07e112ce97205948465c94441cba0042416'
A_MENU='a9abf8de75bf3432fcd714e868e5c14c8884b8f8'
A_RESULT='91349e703e0d09ef702ccd108745dfaf1acc29f7'
SCREEN='256caad3414a80de02fb68eca89c13c957ed6354'
OLD='b72ea6b30e48e6a2f5d805903b7bd750b965cf16'
LONG='155529e20a8fac39e00786c40b57830b71203c4f'
COMMON='0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c'
PHYS=dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855',
          path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
           path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
cache={}


def read(commit,path):
    key=commit,path
    if key not in cache:
        data=I.C.blob(ROOT,commit,path)
        cache[key]=json.loads(gzip.decompress(data) if path.endswith('.gz') else data)
    return cache[key]


def build():
    api=I.providers(str(ROOT));P=I.C
    base=read(A_MENU,'docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json')
    bundle=read(A_RESULT,'docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json')
    s=base['spec'];binding=deepcopy(s['binding']);target=deepcopy(bundle['target'])
    assert P.p.same(binding,target['binding']) and target['energy_shift_ha']==0
    physics=read(**PHYS);index=read(**INDEX)
    assert all(binding[k]==v for k,v in physics['binding'].items())
    screen=read(SCREEN,'docs/research/h4_four_method_signal_screen_preparation_20261113/PLAN.json')
    old_result_path='docs/research/h4_four_method_signal_screen_results_20261113/run02/result.json'
    old=read(OLD,old_result_path)
    longpath='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz'
    signals=read(LONG,longpath)
    reused_prices_path='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/reused_compiled_prices.json.gz'
    reused_prices=read(LONG,reused_prices_path)
    receipt_path='docs/research/d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.csv'
    receipts=list(csv.DictReader(io.StringIO(P.blob(ROOT,P.D_SOURCE,receipt_path).decode())))

    full=api.D.full_pf_spec(s,formula='2nd',delta_pf=deepcopy(s['delta_pf']),M=12,rank=10)
    trunc=deepcopy(full);trunc['L_D']=7
    trunc['split'].update(L_D=7,deterministic_scope='TRUNCATED_H',original_PF_method='B0I')
    trunc['split']['retained_hamiltonian'].update(construction='B0I',prefix=7)
    prep=api.D.saved_full_h_preparation('H4');original_prep_binding=deepcopy(prep['binding'])
    assert all(binding[k]==v for k,v in original_prep_binding.items())
    # Add only the two canonical provenance fields to a consumer descriptor.
    # All physical fields and the immutable saved preparation remain unchanged.
    assert set(binding)-set(original_prep_binding)=={'source','decomposition'}
    prep['binding']=deepcopy(binding)
    pure=api.D.full_h_rte_spec(s,M=12,finite_K=2,preparation=prep)
    specs=[('FULL_PF_S2_D01',full,None),('B0I_P7_S2_D01',trunc,None),
           ('FULL_H_K2_QDYADIC_R448',pure,{m:dict(q=1<<m,R_outer=448) for m in range(13)})]
    entries=[];all_counts=dict(new_mean_evaluations=0,saved_acceptance_cells=0,
        unconditional_saved_mean_copies=0,new_reference_actions=0,new_scalar_extractions=0,
        new_Hamiltonian_state_DF_generation=0,sampling_compile_N_RMSE_G=0)
    for label,spec,layouts in specs:
        config=P.configure(ROOT,spec,layouts=layouts,api=api)
        refs=[]
        for row in config['rounds']:
            m=row['m'];saved=physics['rounds'][m];pair=saved['diagnostics']['reference_pair']
            assert {'reference','independent_reference','signal_difference'}<=set(pair)
            assert api.B.a.exact_scalar(saved['physical_candidate']['T'])==api.B.a.exact_scalar(row['T'])
            refs.append(dict(m=m,T=deepcopy(row['T']),binding=deepcopy(binding),
                source=dict(**PHYS,selector=['rounds',m,'diagnostics','reference_pair'])))
        plan=P.prepare(ROOT,config,target=target,references=refs,api=api)
        rows=[]
        for req in plan['requests']:
            m=req['m'];c=req['physical_candidate'];cid=c['id'];saved_sources=[]
            if cid in signals:
                r=signals[cid]
                assert P.p.same(c,r['candidate'])
                saved_sources.append(dict(commit=LONG,path=longpath,key=cid,
                    backend='LEGACY_SPECTRAL',numerical_fields_present=sorted((r.get('numerical') or {}).keys()),
                    new_direct_backend_qualification=False))
            accepted=None
            for i,r in enumerate(old['records']):
                if r['candidate']['id']==cid:
                    assert P.p.same(r['candidate'],c)
                    accepted=dict(commit=OLD,path=old_result_path,selector=['records',i],
                        backend=r['backend'],numerical_fields_present=sorted(r['numerical']),
                        reported_gates_passed=r['numerical']['gates_passed'],current_acceptance='NOT_EVALUATED')
                    saved_sources.append(accepted)
            action='NEW_MEAN_EVALUATION';availability='NEW_SIGNAL_REQUIRED'
            if saved_sources:availability='SAVED_LEGACY_REQUALIFICATION_REQUIRED'
            if label=='FULL_PF_S2_D01' and m==6:
                assert accepted is not None
                action='ONE_SAVED_ACCEPTANCE_BEFORE_ANY_NEW_MEANS'
                all_counts['saved_acceptance_cells']+=1
            else:all_counts['new_mean_evaluations']+=1
            legacy_stop=None
            if label=='FULL_PF_S2_D01' and m==12:
                availability='LEGACY_NUMERICAL_GATE_FAILURE_NEW_BACKEND_UNEVALUATED'
                legacy_stop=dict(commit=OLD,path='docs/research/h4_four_method_signal_screen_results_20261113/run02/SCREEN_STATUS.json',
                    label='F2-FINE',scientific_infeasibility=False,rerun_old_screen=False)
            price=dict(status='NOT_FOUND_IN_CHECKED_SAVED_DOMAIN',new_price_computation=False)
            found=[r for r in receipts if r['physical_candidate_id']==cid]
            if found:
                assert len(found)==1
                r=found[0]
                price=dict(status='D_RECEIPT_METADATA_AVAILABLE_PENDING_CONSUMER_BINDING',
                    source=dict(commit=P.D_SOURCE,path=receipt_path,physical_candidate_id=cid),
                    X_status=r['X_status'],Y_status=r['Y_status'],price_kind=r['price_kind'],
                    price_carrier=r['price_carrier'],compiler=r['compiler'],new_price_computation=False)
            elif cid in reused_prices:
                price=dict(status='SAVED_TRUNCATED_PRICE_SOURCE_NEEDS_D_DOMAIN_RECEIPT',
                    source=dict(commit=LONG,path=reused_prices_path,key=cid),new_price_computation=False)
            rows.append(dict(m=m,physical_candidate=deepcopy(c),q_RPE=req['q_RPE'],B_q_PF=req['q_PF'],
                q_evolution_grouping=c['q'],signed_structure=deepcopy(req['structure']),
                canonical_row_pointer='/configuration/rounds/'+str(m),reference_descriptor=refs[m],
                existing_signal=availability,saved_sources=saved_sources,legacy_stop=legacy_stop,
                planned_action=action,mean_status='NOT_EVALUATED',signal_values=None,
                price_coverage=price,uncertainty=dict(numerical_bound=None,state_error=None,law_distance=None)))
        entries.append(dict(label=label,configuration=config,rounds=rows,numerical_source=dict(commit=BASE,
            pipeline=P.MODULE,numerics=P.NUMERICAL,original_provider='86f37d25ffd5eee3daf7f25ce001b090b1f2cf33'),
            callback=dict(context_factory='baseline_signal_numerics.Context(root,*load_shared_h4(root,shared_binding))',
                reference_reader='copy PHYSICAL_INPUTS.rounds[m].diagnostics.reference_pair with checked descriptor; no solver'),
            scalar_policy='TOTAL_T_ONCE; B0I adds saved prefix7 identity; FullH adds saved full-H identity',
            fresh_output='docs/research/baseline_first_signal_results_20261113/run01/'+label,
            new_evaluation_backend='PURE_FULL_H_GROUPED_SPECTRAL_POWER' if label.startswith('FULL_H_') else 'DIRECT_EXPM_SIGNED_PF_MATRIX_POWER',
            accepted_saved_backend='SIGNED_S2_MACRO_STEP_MATRIX_POWER' if label=='FULL_PF_S2_D01' else None,
            risk_qualification='PROPOSED_COMMON_TASK_NOT_ADOPTED_FOR_BASELINE',
            N=None,C=None,G=None,winner=None))
    assert all_counts['new_mean_evaluations']==38 and all_counts['saved_acceptance_cells']==1
    result=dict(schema='BASELINE_FINITE_SIGNAL_PROPOSAL_V1',status='PLAN_ONLY_AWAITING_USER_SCIENCE_PERMISSION',
        base_commit=BASE,proposal_kind='INITIAL_SEEDS_NOT_METHOD_OPTIMA',
        comparison_PR_sources=dict(menu=A_MENU,results=A_RESULT,B_preliminary='23d74855135e718554d824a269427de9943213d1',
            D_requests='748be6875b5007707d864e16aa7a4fc4250f3714'),
        shared=dict(binding=binding,target=target,h=deepcopy(s['h']),M=12,rounds=13,P=0,
            rmse_policy=deepcopy(s['rmse_policy']),input_index=INDEX,
            saved_input_descriptors=deepcopy(index['saved_inputs']['H4']['inputs']),
            context_descriptor=deepcopy(index['T20_reference_files']['H4']),
            input_files_opened_today=False,hash_and_sector_checks='DEFERRED_TO_APPROVED_LOAD_SHARED_H4',
            prefix7_scalar=dict(source=INDEX,context_sha256=index['T20_reference_files']['H4']['sha256'],
                selector=['prefixes','7','scalar'],action='COPY_SAVED_VALUE_ONLY; MISSING=>STOP; NO_EXTRACTOR'),
            full_h_preparation=prep,original_preparation_binding=original_prep_binding,
            preparation_binding_bridge='Only source/decomposition provenance added; original physical fields identical',
            independent_reference_pairs=dict(source=PHYS,available_metadata_rounds=list(range(13)),
                numeric_pair_gate_recheck='ONLY_AFTER_APPROVAL',new_actions=0)),
        candidates=entries,finite_scope=dict(total_round_means=39,maximum_new_science=all_counts,
            execution_order=['ACCEPT_FULL_PF_m6_FROM_OLD_SOURCE_THEN_STOP_IF_NOT_ACCEPTED',
                'FULL_PF_NEW_REMAINING_12_WITH_SAVED_m6','B0I_NEW_13','FULL_H_NEW_13'],
            old_vs_new_signal_comparison_cells=0,full_direct_fallback_if_acceptance_fails=False,
            accepted_saved_means_if_PASS=1,acceptance_failure='SAVE_REASON_AND_MANDATORY_STOP_NO_NEW_MEANS',
            one_logical_batch=True,automatic_retry=False,new_science_performed=0),
        numerical_contract=dict(dtype='complex128',state_norm_tolerance=1e-12,hermitian_and_power_tolerance=1e-9,
            reference_pair_tolerance=1e-10,raw_corrected_closure_tolerance=1e-9,
            outcome_range='STRICT [-1,1], NO CLAMP',normalizer='Full-H b>1 and finite; deterministic PF B=1; logB/B/attenuation range gates',
            new_error_fit=False,rigorous_bound=False),
        reuse_admission=dict(m6_original_source='see FULL_PF m6 saved_sources',reanalysis_cells=1,
            allowed='Check original saved values/gates/reference and SOURCE; wrap as legacy SPECTRAL, preserve producer source/backend',
            forbidden='No matrix/expm rerun, no original-vs-new mean differences, no direct-expm relabel, no new u or N'),
        handoff=dict(B='raw/corrected/reference/logB/bias/uncertainty + canonical identity/law/producer source, UNQUALIFIED',
            D='same candidate/control/full measured wrapper/compiler/lowering domain; no price collection in this scope'),
        mandatory_stop=True)
    assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','pyscf'))
    return result


if __name__=='__main__':
    manifest=build()
    with (OUT/'BASELINE_FINITE_MANIFEST.json').open('x') as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print('Metadata only: 3 canonical schedules / 39 rows; proposed 38 new means + 1 saved acceptance; science=0')
