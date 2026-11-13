#!/usr/bin/env python3
"""Fixed A/C/D packet intake and S3 readiness only. No S3 execution option."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_s3_intake as T
from trottertracks import pr_rpe_rmse_evidence as E


def build():
    values,store=T.load_published(ROOT)
    config=values['config'];packet=values['packet']
    A=T.accept_a(config,packet,values['previous'],values['result'],values['physical'],store=store)
    A.update(packet_source=dict(commit=T.A_RELEASE,path=T.A_DIR+'run01/N_HANDOFF.json'),
        science_source=deepcopy(T.NUMERICAL),physical_input_source=dict(commit=T.PHYSICAL_RELEASE,path=T.PHYSICAL_PATH),
        rows=[dict(m=r['m'],physical_candidate_id=r['physical_candidate']['id'],
            canonical_candidate_id=r['canonical_candidate_id'],source_request_id=r['source_request_id'],
            diagnostics_source=r['diagnostics_source'],scientific_source=r['scientific_source'],
            mean_provenance=r['mean_provenance'],reference_context=r['reference_context'],
            numerical_allowance_source=r['diagnostics']['allowance'].get('source'),
            norm_evidence=r['diagnostics'].get('norm_evidence')) for r in values['physical']['rounds']])
    C=T.accept_k1(config,values['k0'],values['k1'])
    prepared=T.prepare_statistics(config,packet,values['physical'],values['models'])
    C['K0_source']=dict(commit=T.C_RELEASE,path=T.C_DIR+'K0.json')
    prices=[];components=[];cohorts=[]
    for structural in C['diagnostic']['rows']:
        out=T.residual_price(config,structural,None,cohort='S3_C_MINUS_S_NOT_ACQUIRED',
            source=dict(module='trottertracks.rpe_s3_intake',residual_interface=T.D_RELEASE))
        prices.append(out['price_record']);components.append(out['component'])
        cohorts.append(dict(candidate_id=config['rounds'][structural['m']]['candidate']['id'],axis=structural['axis'],
            cohort=out['price_record']['cohort']))
    plan=E.plan([dict(configuration=config,N_packet=packet,prices=prices,cohorts=cohorts,
        diagnostic_references=[dict(source=C['diagnostic']['source'],quantity=C['diagnostic']['original_quantity'])])],
        price_policy='qualified_expectation',unexplored_scope='FIXED_H4_PR_ONLY_NO_SCIENTIFIC_MENU_ADOPTED')
    coverage=store.read(dict(commit='170047ef6e7a72aef5f4c6ff9c73f942097a694e',
        path='docs/research/rpe_decoder_statistics_20261113/metadata01/D_RECEIPTS_ACCEPTANCE.json'))
    readiness={k:deepcopy(prepared[k]) for k in ('configuration_id','M','h','decoder','prior','input_complete',
        'physical_qualification_complete','S3_conditions','physical_contract_source','scientific_execution_ready',
        'probabilities_evaluated','statistics_executed','S3_executed','sufficient_N','global_RMSE_bound','empirical_H4_RMSE','G')}
    readiness.update(whole_schedule_nominal=deepcopy(T.NOMINAL),raw_mean_rounds=13,nominal_axis_cells=26,
        target=deepcopy(prepared['target']),technical_data_ready=True,scientific_conditions_adopted=False,
        price_compiled_axes_qualified=0,residual_mean_available=False,
        minimum_next_scope='One fixed configuration, all 13 rounds/26 nominal axes; loss/law/caller/target/precision/trials still require decisions',
        mandatory_stop=True)
    return dict(A_ACCEPTANCE=A,C_STRUCTURAL_ACCEPTANCE=C,STATISTICS_INPUTS=prepared,S3_READINESS=readiness,
        D_RESIDUAL_INTERFACE=dict(source=dict(commit=T.D_RELEASE,path='src/trottertracks/compiled_residual.py'),
            residual_estimates=None,components=components,price_records=prices,mean_compiled_C=None,
            qualified_for_compiled_price=False,G=None),
        FULL_PF_SAVED_COVERAGE=dict(original_acceptance_source=dict(commit='170047ef6e7a72aef5f4c6ff9c73f942097a694e',
            path='docs/research/rpe_decoder_statistics_20261113/metadata01/D_RECEIPTS_ACCEPTANCE.json'),**coverage),
        EVIDENCE_PLAN=plan,SOURCE_FILES=store.sources)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output directory required')
    products=build();args.out.mkdir(parents=True)
    for name,value in products.items():
        (args.out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(A_rounds=13,nominal_axes=26,C_structural_cells=4,S3_executed=False,mean_compiled_C=None,G=None)))


if __name__=='__main__':main()
