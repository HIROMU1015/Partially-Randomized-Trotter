#!/usr/bin/env python3
"""Pinned A/C/D metadata intake. No scientific/provider/cost execution options."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_diagnostic_intake as M
from trottertracks import rpe_d_evidence_intake as D
from trottertracks import pr_rpe_rmse_evidence as E


def build(*,a_result_ref=None,a_packet_ref=None,c_structural_ref=None):
    store=M.p.SavedArtifacts(ROOT)
    read=lambda commit,path:store.read(dict(commit=commit,path=path))
    config=read(M.S0.B_RELEASE,M.S0.BDIR+'CONFIGURATION_V2.json')
    previous=read(M.S0.A_RELEASE,M.S0.ADIR+'V2_N_HANDOFF.json')
    M.H.validate_A_v2_packet(config,previous,store.read(M.H.A_PACKET))
    preflight=read(M.A_PREPARATION,M.A_PREFLIGHT)
    M.H.same(preflight['configuration_id'],config['id'],'A_PREPARATION_CONFIGURATION_MISMATCH')
    M.H.same(preflight['physical_candidate'],config['rounds'][1]['physical_candidate'],'A_PREPARATION_M1_PHYSICAL_MISMATCH')
    packet=previous
    a_status=dict(status='M1_RESULT_NOT_YET_RECEIVED',preparation_source=dict(commit=M.A_PREPARATION,path=M.A_PREFLIGHT),
        previous_packet=dict(commit=M.S0.A_RELEASE,path=M.S0.ADIR+'V2_N_HANDOFF.json'),
        new_science_result=None,new_nominal_rounds=[],preserved_S0_rounds=[0,6])
    if a_result_ref:
        result=store.read(a_result_ref)
        a_status.update(new_science_result=deepcopy(a_result_ref),result_status=result['status'])
        if a_packet_ref:
            incoming=store.read(a_packet_ref)
            M.H.same(incoming['source']['commit'],result['science_source'],'A_RESULT_NUMERICAL_SOURCE_DIFFERS')
            M.H.same(result['configuration_id'],config['id'],'A_RESULT_CANONICAL_CONFIGURATION_DIFFERS')
            for key in ('nominal_N_X','nominal_N_Y'):
                M.H.same(incoming['rounds'][1][key],result['model_record'][key],'A_RESULT_PACKET_NOMINAL_DIFFERS')
            M.H.same(incoming['rounds'][1]['model_status'],result['model_record']['status'],'A_RESULT_PACKET_MODEL_STATUS_DIFFERS')
            M.H.same(result['model_record']['identity']['candidate'],config['rounds'][1]['physical_candidate'],'A_RESULT_PHYSICAL_CANDIDATE_DIFFERS')
            M.validate_m1_packet(config,incoming,previous,numerical_source=incoming['source'],adapter_source=preflight['source'])
            packet=incoming
            a_status.update(status='PUBLISHED_M1_DIAGNOSTIC_PACKET_ACCEPTED_UNQUALIFIED',packet_source=deepcopy(a_packet_ref),
                new_diagnostic_rounds=[1],new_nominal_rounds=[1] if any(incoming['rounds'][1]['nominal_N_'+a] is not None for a in M.p.AXES) else [])
        else:
            a_status['status']='PUBLISHED_A_RESULT_WITHOUT_N_PACKET_NOT_PROMOTED'
    elif a_packet_ref:
        raise ValueError('Explicit A result SOURCE required with a new packet')

    result=read(M.C_DIAGNOSTIC,M.C_DIR+'run02/RESULT.json')
    checks=read(M.C_DIAGNOSTIC,M.C_DIR+'CHECKS.json')
    reduction=read(M.C_DIAGNOSTIC,M.C_DIR+'REDUCTION.json')
    c_status=M.accept_c_published(config,result,checks,reduction,store.read(M.S0.CPLAN))
    if c_structural_ref:
        structural=store.read(c_structural_ref)
        c_status['new_kernel_evidence']=M.accept_structural_packet(config,structural)
        c_status.update(K0_received=True,K1_received=structural['stage']=='K1',kernel_packet_source=deepcopy(c_structural_ref))
    cov=read(D.COVERAGE,D.COVERAGE_JSON)
    d_coverage=D.accept_saved_coverage(cov)
    examples=read(D.PHYSICAL,D.EXAMPLE_JSON)
    bridges={}
    for name,entry in examples['packets'].items():
        bound=D.accept_example(entry)
        joins=None;join_status='FULL_H_RTE_IS_OUTSIDE_FULL_PF_COVERAGE'
        if bound['physical']['method']=='FULL_PF':
            try:
                joins=D.join_saved_coverage(bound,cov);join_status='EXACT_PHYSICAL_CONTEXT_METADATA_JOIN_ONLY'
            except ValueError as error:
                # Preserve the published artificial MOCK compiler; never repair
                # its profile to make a saved scientific price appear reusable.
                if str(error)!='D_SAVED_COMPILER_PROFILE_MISMATCH':raise
                join_status=str(error)
        bridges[name]=dict(source=dict(commit=D.PHYSICAL,path=D.EXAMPLE_JSON),artificial_example_only=True,
            old_v2_status=entry['v2_status'],configuration_id=bound['v2_configuration']['id'],
            identity_map=bound['identity_map'],original_parameterization=bound['original_parameterization'],
            canonical_parameterization=bound['canonical_parameterization'],saved_coverage=joins,saved_coverage_status=join_status,
            G=None,scientific_conditions_adopted=False)
    plan=E.plan(
        [dict(configuration=config,N_packet=packet,diagnostic_references=[dict(component='C',source=c_status['source'],kind='STRUCTURAL_DIAGNOSTIC_NOT_COMPILED_PRICE')])],
        price_policy='qualified_expectation',unexplored_scope='NO_FULL_PF_TRUNCATED_PF_OR_FULL_H_RTE_SCIENTIFIC_MENU_ADOPTED')
    summary=dict(configuration_id=config['id'],A=a_status,C=c_status,D_saved_totals=d_coverage['totals'],
        four_method_coverage=deepcopy(plan['method_coverage']),
        nominal_diagnostics=[{k:deepcopy(r[k]) for k in ('m','source_candidate_id','nominal_N_X','nominal_N_Y','N_X','N_Y','model_status')} for r in packet['rounds']],
        G=None,winner=None,global_RMSE_bound=None,qualified_mean_compiled_C=None,
        full_pf_saved_inventory_is_a_menu=False,source_files=store.sources,
        science_calls=0,mandatory_stop=True)
    gaps=dict(
        FULL_PF=dict(saved_candidates=d_coverage['totals']['candidates'],saved_evidence_source=d_coverage['source'],
            saved_price_classification='DIRECT_C_METADATA_NOT_CURRENT_RMSE_PRICE_QUALIFICATION',
            missing=['FORMAL_FORMULA_WIDTH_M_RMSE_MENU','SUFFICIENT_N_GLOBAL_RMSE','RECEIPTS_FOR_CONDITIONAL_PRICES','MISSING_OR_OFF_GRID_C'],
            owners=['USER','Codex D','Codex A','Codex C']),
        TRUNCATED_PF=dict(historical_source=dict(commit='6a9d7fd36b687e58aecbed1278c6d7927d6827d6',
            path='docs/research/pr_rpe_method_coverage_20261113/COVERAGE.json'),
            H4_B0_prefix7_is_not_Full_PF=True,missing=['FORMAL_COMPARISON_MENU','TRUNCATION_ERROR_AGAINST_ORIGINAL_H','RMSE_N_AND_PRICE_QUALIFICATION'],
            owners=['USER','Codex A','Codex C']),
        PR=dict(configuration_id=config['id'],new_m1_status=a_status['status'],new_C_status=c_status['kind'],
            missing=['ALL_13_ROUND_SUFFICIENT_N','GLOBAL_RMSE_BOUND_FINITE_K_KAPPA_CONTRACT','ALL_26_AXIS_QUALIFIED_MEAN_COMPILED_C',
                'C_MINUS_S_LAW_DISTANCE_RARE_LONG_Q_SYSTEMATIC_COVARIANCE_CI'],owners=['Codex A','Codex C']),
        FULL_H_RTE=dict(technical_mapping_kind=D.B.EXPLICIT_LAYOUT,DF_B3_is_not_Full_H_RTE=True,
            missing=['FORMAL_FULL_H_PREPARATION_LAYOUT_K_M_RMSE_MENU','FINITE_K_NORMALIZATION_SIGNAL_N_GLOBAL_RMSE','GROUPING_LAW_PROFILE_MATCHED_MEAN_C'],
            owners=['USER','Codex D','Codex A','Codex C']))
    for value in gaps.values():
        value.update(sufficient_N=None,global_RMSE_bound=None,qualified_mean_compiled_C=None,G=None,winner=None,
            scientific_menu_automatically_adopted=False,automatic_execution=False)
    return dict(PUBLISHED_EVIDENCE=summary,EVIDENCE_PLAN=plan,METHOD_GAPS=gaps,
        D_SAVED_COVERAGE=d_coverage,D_ARTIFICIAL_BRIDGES=bridges)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--a-release');parser.add_argument('--a-result-path');parser.add_argument('--a-packet-path')
    parser.add_argument('--c-release');parser.add_argument('--c-packet-path')
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output directory required')
    if bool(args.a_release)!=bool(args.a_result_path) or (args.a_packet_path and not args.a_result_path):
        parser.error('A result release/path required; packet path is optional')
    if bool(args.c_release)!=bool(args.c_packet_path):parser.error('C release/path required together')
    products=build(a_result_ref=dict(commit=args.a_release,path=args.a_result_path) if args.a_release else None,
        a_packet_ref=dict(commit=args.a_release,path=args.a_packet_path) if args.a_packet_path else None,
        c_structural_ref=dict(commit=args.c_release,path=args.c_packet_path) if args.c_release else None)
    args.out.mkdir(parents=True)
    for name,value in products.items():
        (args.out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(A=products['PUBLISHED_EVIDENCE']['A']['status'],D_candidates=products['D_SAVED_COVERAGE']['totals']['candidates'],G=None)))


if __name__=='__main__':main()
