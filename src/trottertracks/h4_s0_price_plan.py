"""Metadata-only binding of fixed A H4 S0 to the published B V2 API.

No signal, normalization, S, price, sampler, compiler, fit or G calls.
B owns schedule generation and identity. This module never reuses a price value
by assigning an old physical ID to a new B ID.
"""
from copy import deepcopy
import json
import subprocess

from . import pf_pr_rpe_pipeline as p
from .bv2_mean_price import V2MeanPriceLookup

A_RELEASE='7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75'
A_PATH='docs/research/pr_rpe_rmse_s0_results_20261113/run01/CONFIGURATION.json'
B_RELEASE='af284d853890155bacd19e61850c196bd1d5c660'
B_V2='6a9d7fd36b687e58aecbed1278c6d7927d6827d6'
PRICE_INVENTORY='docs/research/bv2_mean_price_validation_design_20261113/PRICE_COVERAGE.json'
C_RELEASE='1db4d6e7ebce872c069e337b76632046f60d8b56'
FOCUS={0:('53e00d8b07d1547d19a7',1,2,2),6:('2a8c6718220eec59d419',64,128,8192)}


def read(root,commit,path):return json.loads(subprocess.check_output(['git','show',commit+':'+path],cwd=root))


def prepare(root,*,integration,configuration=None):
    fixed=read(root,A_RELEASE,A_PATH)
    if configuration is not None and not p.same(configuration,fixed):raise ValueError('FIXED_A_S0_METADATA_CHANGED')
    inputs=deepcopy(fixed['inputs']);conditions=fixed['conditions']
    if inputs['kappa_paper']!=conditions['kappa_paper_exact']:raise ValueError('USE_A_EXACT_KAPPA_NOT_APPROXIMATION')
    # Exact current B code, not an in-place migration of the saved V1 packet.
    config=integration.B.generate(integration.a_spec(inputs,'PR'))
    if config['schema']!='PR_RPE_RMSE_V2':raise ValueError('B_V2_REQUIRED')
    saved={r['identity']['m']:r for r in fixed['A_schedule']['rows']
           if p.arm(r['identity']['candidate'])=='PR'}
    if set(saved)!=set(range(13)) or len(config['rounds'])!=13:raise ValueError('FIXED_THIRTEEN_ROUNDS')
    inventory=read(root,C_RELEASE,PRICE_INVENTORY)
    quotes=V2MeanPriceLookup(config,integration=integration)()
    if any(q['mean']['rz_count'] is not None for q in quotes):raise ValueError('NO_S0_PRICE_SUPPLIED')
    requests=[];coverage=[]
    for row in config['rounds']:
        r=saved[row['m']];physical=r['identity']['candidate']
        if (not p.same(row['physical_candidate'],physical) or not p.same(row['structure'],r['identity']['structure'])
             or not p.same(row['round_law_definition']['A_declaration'],r['law_binding'])):raise ValueError('A_B_PHYSICAL_REQUEST_OR_LAW_DIFFERS')
        historical=[v for v in inventory['rows'] if v['candidate_id']==physical['id']]
        direct=any(v['direct_cohorts'] for v in historical);model=any(v['model_cohorts'] for v in historical)
        coverage.append(dict(m=row['m'],physical_candidate_id=physical['id'],B_candidate_id=row['candidate']['id'],
            T=physical['T'],q_PF=physical['q'],R_outer=physical['R_outer'],event_count=row['structure']['event_count'],
            historical_inventory_match=bool(historical),saved_direct_metadata_present=direct,saved_model_metadata_present=model,
            price=None,qualified_mean_compiled_price=False,source=dict(commit=C_RELEASE,path=PRICE_INVENTORY)))
        if row['m'] in FOCUS:
            if (physical['id'],physical['q'],physical['R_outer'],row['structure']['event_count'])!=FOCUS[row['m']]:raise ValueError('FIXED_S0_TARGET_CHANGED')
            requests.append(dict(m=row['m'],physical_candidate=deepcopy(physical),
                A_request_id=r['request_id'],A_source=dict(commit=A_RELEASE,path=A_PATH),
                identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
                source_law_declaration=deepcopy(r['law_binding']),price=None,cohort=None,
                model_qualified_for_candidate=False,uncertainty=dict(systematic=None,joint_covariance=None,total_CI=None),
                quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE'))
    return dict(status='FIXED_A_S0_PRICE_METADATA_ONLY',A_source=dict(commit=A_RELEASE,path=A_PATH),
        B_code_source=B_RELEASE,B_V2_code_source=B_V2,
        binding=deepcopy(inputs['design']['binding']),profile=deepcopy(inputs['design']['profile']),
        structural_contract=deepcopy(inputs['design']['contract']),
        M=conditions['M'],h_exact=conditions['h_exact'],kappa_paper_exact=conditions['kappa_paper_exact'],
        normalization_rule=conditions['normalization_rule'],
        A_schedule_id=fixed['A_schedule']['configuration_id'],
        saved_B_schema=fixed['B_configuration']['schema'],saved_B_id=fixed['B_configuration']['id'],
        B_V2_configuration_id=config['id'],B_identity_status='C_RECONSTRUCTED_FROM_FIXED_B_NOT_YET_B_PUBLISHED_REQUEST',
        focus_requests=requests,round_price_coverage=coverage,
        price_values_copied=False,new_scientific_calls=0,mean_compiled_price=None,full_rpe_resources=None,
        mandatory_stop=True)


def verify_B_publication(plan,published,*,integration):
    """Later exact check against a B-published V2 configuration; no relabel."""
    integration.B._validate(published)
    if published['schema']!='PR_RPE_RMSE_V2' or published['id']!=plan['B_V2_configuration_id']:
        raise ValueError('PUBLISHED_B_CONFIGURATION_DIFFERS')
    rows={r['m']:r for r in published['rounds']}
    for request in plan['focus_requests']:
        row=rows[request['m']]
        if (not p.same(row['physical_candidate'],request['physical_candidate']) or not p.same(row['identity'],request['identity'])
                or not p.same(row['law_binding'],request['law_binding'])):raise ValueError('PUBLISHED_B_PRICE_IDENTITY_LAW_DIFFERS')
    return dict(status='EXACT_METADATA_MATCH_ONLY',scientific_price_qualified=False)
