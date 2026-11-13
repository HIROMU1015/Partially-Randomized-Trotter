"""Passive metadata projection only; no saved S/C arithmetic or price model call."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_optimization_pipeline as I

SOURCES={
 'A_MENU':dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',path='docs/research/pr_rpe_first_signal_menu_20261113/'),
 'A_CURRENT':dict(commit='2f0de95304000af251a179fefdddf2a10b8fdc14',path='docs/research/pf_effective_hamiltonian_20261113/SCIENCE_PLAN.md'),
 'B_CURRENT':dict(commit='d71721f30c7b7fc13ba0789eefffd6238aca3ebc',path='docs/research/rpe_deterministic_risk_20261113/NEXT_SCIENCE_PLAN.md'),
 'B_PROXY':dict(commit='23d74855135e718554d824a269427de9943213d1',path='docs/research/rpe_cost_preliminary_results_20261113/NEXT_OPTIMIZATION_PLAN.md'),
 'D_CORRECTED':dict(commit='e827367629ca6c2920d9d908fc1631314399db86',path='docs/research/d_t2_scalar_binding_fix_20261113/HANDOFF.md'),
 'D_REQUESTS':dict(commit='748be6875b5007707d864e16aa7a4fc4250f3714',path='docs/research/d_ld12_price_requests_20261113/PRICE_REQUEST_COVERAGE.json'),
 'PF_RECEIPTS':dict(commit='e7dca5f7301c8633b17a59b6a7161331303aef2d',path='docs/research/d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.csv'),
 'P2':dict(commit='39ddd59edf54a3a901e5302207b2765ddbd03fbb',path='docs/research/d_p2_direct_pilot_20261113/run01/RESULT.json'),
 'ES':dict(commit='9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f',path='docs/research/h4_structural_schedule_results_20261113/run01/COVERAGE.json'),
 'C1':dict(commit='2e6a0cdf5a647058cc7afeeb5741bfafa24c974b',path='docs/research/r_price_phase1_results_20261111/REPORT.md'),
 'C_MODEL':dict(commit='1914d0764115d82ec860e08810dafeb50f8d9a42',path='src/trottertracks/rpe_mean_price_model.py')}


def blob(key,suffix=''):
    s=SOURCES[key];return I.C.blob(ROOT,s['commit'],s['path']+suffix)


def load(key,suffix=''):return json.loads(blob(key,suffix))


def build():
    api=I.providers(str(ROOT));p=I.C.p;cells=[];catalog=[]
    d=load('D_REQUESTS');es=load('ES');p2=load('P2')
    receipts=list(csv.DictReader(io.StringIO(blob('PF_RECEIPTS').decode())))
    by_physical={r['physical_candidate_id']:r for r in receipts}
    configs={ld:load('A_MENU','CONFIGURATION_LD'+str(ld)+'.json') for ld in (1,2)}
    binding=configs[1]['spec']['binding']
    pr_profile=deepcopy(d['cells'][0]['price_profile'])
    for ld,cfg in configs.items():
        api.B._validate(cfg)
        for row in cfg['rounds']:
            c=row['physical_candidate'];m=row['m']
            for axis,old_axis in (('X','cosine'),('Y','sine')):
                req=next(r for r in d['cells'] if r['L_D']==ld and r['m']==m and r['axis']==axis)
                assert req['physical_id']==c['id'] and p.same(req['q_PF'],row['q_PF']) and req['R_outer']==row['R_outer']
                assert p.same(req['price_profile'],pr_profile)
                assert all(p.same(v,pr_profile[k]) for k,v in cfg['spec']['profile'].items())
                sf=[(i,r) for i,r in enumerate(es['values']) if r['candidate_id']==c['id'] and r['axis']==old_axis]
                structural=[]
                for i,r in sf:
                    assert (r['q_PF'],r['R_outer'])==(c['q'],c['R_outer'])
                    assert all(r['identity']['binding'][k]==binding[k] for k in ('hamiltonian','state','snapshot','rank'))
                    structural.append(dict(source='ES',selector=['values',i],quantity=r['quantity'],
                        status=r['status'],expectation=r['expectation'],source_configuration_id=es['configuration_id'],
                        canonical_identity_equal=p.same(r['identity'],row['identity']),
                        law_binding_equal=p.same(r['law_binding'],row['law_binding']),consumer_bridge='PENDING_NO_ADMISSION'))
                finite=[]
                for i,r in enumerate(p2['summary']):
                    if r['candidate_id']==c['id'] and r['cohort']=='D_P2_DIRECT_PILOT_V1':
                        finite.append(dict(source='P2',selector=['summary',i],axis=axis,cohort=r['cohort'],
                            n_paths=r['n_paths'],expectation=r['expectation'],role=r['role'],
                            lowering_id='CONTROLLED_ZZ_CX_CRZ_CX_V1',status='SAVED_PAIRED_FINITE_COHORT_ONLY',
                            consumer_bridge='PENDING_NO_ADMISSION'))
                cells.append(dict(label='PR-LD'+str(ld),catalog_role='B_FIXED_PR_SCHEDULE',m=m,axis=axis,
                    configuration_id=cfg['id'],canonical_candidate_id=row['candidate']['id'],physical_id=c['id'],
                    T=c['T'],formula=c['formula'],actual_delta=c['delta'],prefix=ld,q_PF=c['q'],R_outer=c['R_outer'],K=c['K'],
                    candidate_source=dict(source='A_MENU',file='CONFIGURATION_LD'+str(ld)+'.json',selector=['rounds',m]),
                    direct_deterministic=[],finite_cohort=finite,structural_mean=structural,
                    C1=dict(source='C1',status='OUTSIDE_FROZEN_Q_R_AND_LOWERING_DOMAIN' if ld==1 else
                        'FIXED_PREFIX1_MODEL_OUTSIDE_PREFIX2_DOMAIN',qualified=False),
                    additive=dict(source='C_MODEL',status='IMPLEMENTED_NOT_EVALUATED',prediction=None,qualified=False),
                    old_D_request=dict(request_id=req['request_id'],context_key=req['context_key'],source='D_REQUESTS',
                        status='LEGACY_TAIL_ONLY_EXPECTED_SCALAR_ID_DO_NOT_REUSE_FOR_CORRECTED_REQUEST'),
                    scalar=dict(status='USE_D_CORRECTED_TOTAL_SELECTOR_NOT_MATERIALIZED_HERE',source='D_CORRECTED'),
                    profile_ref='PR_NEW_ZZ',
                    target_lowering='CONTROLLED_ZZ_CX_CRZ_CX_V1',compiled_mean_qualified=False,mean_C=None,total_CI=None))
    # Source-derived recipe catalogue, not a newly adopted A execution menu.
    for formula in ('2nd','4th','4th(m5_best)'):
        for width in (.1,.4):
            spec=api.D.full_pf_spec(configs[1]['spec'],formula=formula,delta_pf=width,M=12,rank=10)
            cfg=I.C.configure(ROOT,spec,api=api);label='FULL_PF_'+formula+'_D'+str(width)
            support=[dict(source='PF_RECEIPTS',physical_id=r['physical_candidate_id'],T=r['T'],q_PF=r['q_PF'],
                X_status=r['X_status'],Y_status=r['Y_status']) for r in receipts
                if r['system']=='H4' and r['formula']==formula and float(r['actual_width'])==width
                and r['X_status']==r['Y_status']=='EXACT_METADATA_MATCH']
            catalog.append(dict(label=label,configuration_id=cfg['id'],formula=formula,requested_delta=width,
                role='LEGACY_RECIPE_LOOKUP_CATALOG_A_SELECTION_PENDING',source='B_PROXY',
                saved_exact_calibration_points=support))
            for row in cfg['rounds']:
                m=row['m'];c=dict(system='H4',method='FULL',formula=formula,prefix=10,
                    T=float(api.B.a.exact_scalar(row['T'])),q=row['q_PF'],
                    delta=float(api.B.a.exact_scalar(row['actual_delta_pf'])),R_outer=None,K=None)
                c['id']=p.candidate_id(c);r=by_physical.get(c['id'])
                for axis in ('X','Y'):
                    direct=[]
                    if r is not None and r[axis+'_status']=='EXACT_METADATA_MATCH':
                        assert (r['system'],r['formula'],int(r['q_PF']))==('H4',formula,c['q'])
                        assert r['compiler']==cfg['spec']['profile']['compiler']
                        direct.append(dict(source='PF_RECEIPTS',physical_candidate_id=c['id'],
                            axis_status=r[axis+'_status'],price_kind=r['price_kind'],price_carrier=r['price_carrier'],
                            compiler=r['compiler'],receipt_evidence=r['receipt_evidence'],receipt_native_SOURCE=r['receipt_native_SOURCE'],
                            lowering_scope='SAVED_DETERMINISTIC_NATIVE_POLICY_NOT_AUTOMATICALLY_NEW_ZZ',
                            consumer_bridge='PENDING_V2_TRACE_PROFILE_AND_SCALAR_ADMISSION'))
                    cells.append(dict(label=label,catalog_role='LEGACY_RECIPE_LOOKUP_CATALOG_A_SELECTION_PENDING',
                        m=m,axis=axis,configuration_id=cfg['id'],canonical_candidate_id=row['candidate']['id'],physical_id=c['id'],
                        T=c['T'],formula=formula,requested_delta=width,actual_delta=c['delta'],prefix=10,q_PF=c['q'],R_outer=None,K=None,
                        candidate_source=dict(generator='C.configure+D.full_pf_spec',base_spec_source='A_MENU',recipe_catalog_source='B_PROXY'),
                        direct_deterministic=direct,finite_cohort=[],structural_mean=[],C1=dict(status='PR_C1_NOT_APPLICABLE'),
                        receipt_catalog_status=r[axis+'_status'] if r is not None else 'OUTSIDE_CHECKED_CATALOG',
                        additive=dict(source='C_MODEL',status='IMPLEMENTED_NOT_EVALUATED',prediction=None,qualified=False),
                        scalar=dict(status='ORIGINAL_DETERMINISTIC_NATIVE_SCALAR_ONCE_REQUIRES_RECEIPT_TRACE'),
                        profile_ref='PF_SAVED_NATIVE',
                        target_lowering='SAVED_SCENARIO_OR_EXPLICIT_NEW_PROFILE_BRIDGE_REQUIRED',
                        compiled_mean_qualified=False,mean_C=None,total_CI=None))
    assert len(cells)==208
    assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','pyscf'))
    return dict(status='STATIC_PRICE_EVIDENCE_NO_SCIENTIFIC_REANALYSIS',base_commit=SOURCES['C_MODEL']['commit'],
        sources=SOURCES,physical_binding=deepcopy(binding),unit='FULL_CONTROLLED_MEASURED_XY_WRAPPER_COMPILED_RZ_COUNT_P0',
        profiles=dict(PR_NEW_ZZ=pr_profile,PF_SAVED_NATIVE=dict(profile=deepcopy(configs[1]['spec']['profile']),
            native_policy='M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1',
            new_ZZ_equivalence='NOT_ASSUMED',source='PF_RECEIPTS')),
        Full_PF_catalog=catalog,Full_PF_selection='NOT_ADOPTED_A_HAS_NOT_FIXED_FORMULA_OR_WIDTH',
        B_qualified_N='NOT_OBTAINED_NO_ACTUAL_PRICE_PRECISION_BUDGET',
        D_current_source='D_CORRECTED',request_materialization='NOT_CALLED_NO_SNAPSHOT_CONSTANT_READ',
        coverage=cells,S_C_values_copied_or_recomputed=False,model_fit_or_prediction=False,G=None,mandatory_stop=True)


if __name__=='__main__':
    with (Path(__file__).parent/'COVERAGE.json').open('x') as f:
        json.dump(build(),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print('Static projection: PR52 + FullPF156 cells; no S/C values or model calculations')
