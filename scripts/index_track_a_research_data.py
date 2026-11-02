#!/usr/bin/env python3
"""Metadata-only Track A access views. Never run scientific sources or formulas.

Reuses immutable CSV/JSON/token/Git-reader helpers from the adopted integrations.
--build writes only the new bundle; the default --check verifies it read-only.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

from adopt_h46_standard_y4_confirmation import git_bytes, csv_saved, json_saved
from integrate_h468_paper_evidence import csv_text, json_text, pointer

ROOT = Path(__file__).resolve().parents[1]
OUT = 'docs/data_collection/2026-10-24_track_a_research_data/'
BASE = '8a2fbe9461d9cc262326fa30f9778b295adeb3f8'
ADOPT = '09e71910ee49e0d9d3a2f2f410249f6623022c0d'
THREE = '6dafed86cc82149d55b4e6395382953d5f3631d7'
PF = 'f75a7c25d584b69d90de454731a7f343a421e059'
BATCH = 'eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23'
CNG = 'd2536c4f959bfb4edd7d2deeceb155585fb53eb0'
Y4 = '4d117041469fb8eebf2ad55405c0f7d1b70dea7d'
A = 'artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/'
T = 'docs/data_collection/2026-10-23_three_environment_integration/'
P = 'artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/'
R = 'artifacts/resource_applicability/track_a_h246_reversal_batch_run01/2026-10-24/'
D = 'artifacts/resource_applicability/track_a_h246_order_cost_analysis_run01/2026-10-24/'
Y = 'artifacts/resource_applicability/track_a_h46_standard_y4_confirmation_recovery/2026-10-24/'
URL = 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'
UNKNOWN = 'unknown'
NV = 'not_verified'
NA = 'not_applicable'
LIMIT = 'FINITE_SAVED_SCOPE;EMPIRICAL_ALLOWANCE;NO_POPULATION_OR_RARE_TAIL_GUARANTEE;NO_POOLING;NO_FT_ENERGY_RPE_EXTRAPOLATION'
FIELDS = ['study_id','record_role','category','purpose','system','geometry','basis','DF_rank_policy','Hamiltonian_state_reference','T','epsilon','P','delta_or_signed_time','method_formula','prefix','q','r','K','partition','sampling_law','cohort','n','compiler_native','metric_unit','evidence_type','result_status','adoption_status','SOURCE','result_commit','result_reference','condition_reference','upstream_profile','limitations','original_index_selector','new_independent_observations_from_indexing']

def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(c,p):
    return json_saved(git_bytes(c,p)) if p.endswith('.json') else csv_saved(git_bytes(c,p))


def make_sources():
    sources = {}
    def add(key,c,p,known=None):
        entry = dict(known or {})
        entry.update(commit=c,path=p,url=URL+c+'/'+p)
        if not entry.get('sha256'):
            entry['sha256'] = hashlib.sha256(git_bytes(c,p)).hexdigest()
            entry['verification_scope'] = 'SELECTED_FIXED_GIT_METADATA_OR_PUBLIC_TABLE_READ'
        else:
            entry['verification_scope'] = 'REUSED_UPSTREAM_PIN_AND_AUDIT_NO_NEW_RAW_REAUDIT'
        sources[key] = entry
    for n in ['paper_manifest.json','claim_evidence_matrix_v3.csv','evidence_ledger_v5.csv','paper_data_manifest.csv','evidence_roles.csv','source_catalog.json','adoption_record.json','REPORT.md','STOP.json']:
        add('adopt/'+n,ADOPT,A+n)
    for n in ['README.md','research_index.csv','references.csv','scientific_profiles.json','source_result_audit.csv','gaps_and_requests.json','M1_M3_review_mapping.md','claim_evidence_overlay.csv','review_source_receipt.json','integrity_audit.json','duplicate_version_checks.json']:
        add('three/'+n,THREE,T+n)
    for key,v in load(ADOPT,A+'source_catalog.json')['sources'].items():
        add('adopt_source/'+key,v['commit'],v['path'],v)
    manifest=load(ADOPT,A+'paper_manifest.json')
    v=manifest['main_table']; add('main_table',v['commit'],v['path'],v)
    for n in ['candidate_index.csv','comparisons_all_formulas_15.json','comparisons_within_formula_45.json','old_vs_common_15.json','denominator_decomposition_10.json','paper_manifest.json']:
        add('pf/'+n,PF,P+n)
    for n in ['REPORT.md','analysis.json','cost_decomposition_22.csv','main_C_N_G_6.csv','shot_relationship_axes_88.csv','representative_RTE_6.csv','SOURCE.json','STOP.json']:
        add('cng/'+n,CNG,D+n)
    for n in ['REPORT.md','primary_endpoints_60.csv','primary_directions_15.csv','primary_comparisons_30.csv','H2_new_endpoints_16.csv','same_q_comparisons_8.csv','same_q_directions_4.csv','STOP.json']:
        add('batch/'+n,BATCH,R+n)
    for key,p in [('H2_four','track_a_h2_archived_exploratory_run01/2026-10-24/four_method_saved_output_table.csv'),('H4_four','track_a_h4_four_method_exploratory_run01/2026-10-24/four_method_saved_output_table.csv'),('H6_four','track_a_h6_saved_four_method/2026-10-24/four_method_saved_values.csv')]:
        add(key,CNG,'artifacts/resource_applicability/'+p)
        add(key+'_report',CNG,'artifacts/resource_applicability/'+p.rsplit('/',1)[0]+'/REPORT.md')
    for key,c,p in [
        ('h2_diagonal','43be11d648ca35302bd2c7267e23327fe37e34cb','docs/research/track_a_h2_pr_diagonal_reuse_a_20261023.md'),
        ('h2_static','041511aa567c288bcead32d0bf7579fd7f434e44','docs/research/track_a_h2_saved_cost_static_join_b2_20261023.md'),
        ('h2_strict','4008ce45d26bb858e092d8654187bfcda2dbe180','artifacts/resource_applicability/track_a_h2_accepted_compare/2026-10-23/REPORT.md'),
        ('prior_art',BASE,'docs/research/prior_art_higher_pf_delta_20261024/REPORT.md')]:
        add(key,c,p)
    return sources


def old_rows():
    rows=load(THREE,T+'research_index.csv')
    audits={x['series_key']:x for x in load(THREE,T+'source_result_audit.csv')}
    profiles={x['profile_id']:x for x in load(THREE,T+'scientific_profiles.json')['profiles']}
    result=[]
    for old in rows:
        obj=load(old['manifest_commit'],old['manifest_path'])
        original=pointer(obj,old['manifest_json_pointer'])
        prof=profiles[old['profile_id']]
        s={k:UNKNOWN for k in FIELDS}
        s.update(study_id=old['series_key'],record_role='EXISTING_CATALOGUE_ENTRY_NOT_AN_EXPERIMENT_COUNT',
          category='診断系列',purpose=original.get('title',original.get('scope',prof['purpose'])),
          system=original.get('model',UNKNOWN),geometry=original.get('geometry',UNKNOWN),basis=original.get('basis',UNKNOWN),
          DF_rank_policy=original.get('df_rank_or_policy',UNKNOWN),Hamiltonian_state_reference='ORIGINAL_FIXED_INPUT_MANIFEST;not_verified_by_this_index',
          delta_or_signed_time=original.get('delta_window','not_verified;see original candidate and signed piece schedule'),method_formula=prof['candidates_and_schedule'],cohort=original.get('cohort',prof['cohort']),
          compiler_native=prof['measurement_and_contract'],metric_unit='SEE_ORIGINAL_FIELD;do_not_mix_RZ_depth_CX_G',
          evidence_type='HISTORICAL_OR_TECHNICAL',result_status=original.get('recorded_scope_or_status','SEE_FIXED_ORIGINAL_RESULT_AND_END_AUDIT'),
          adoption_status='not_verified;consult_latest_CEM_row_for_exact_claim',
          SOURCE=audits[old['series_key']]['recorded_SOURCE_json'],result_commit=old['recorded_result_commit'] or UNKNOWN,
          result_reference='three/references.csv#ref_id='+old['primary_ref_id'] if old['primary_ref_id'] else 'three/research_index.csv#series_key='+old['series_key'],
          condition_reference=old['manifest_commit']+':'+old['manifest_path']+'#'+old['manifest_json_pointer'],
          upstream_profile='three/scientific_profiles.json#profile_id='+old['profile_id'],
          limitations=prof['limits']+'; '+str(original.get('limitations','')),
          original_index_selector=old['series_key'],new_independent_observations_from_indexing='0')
        if 'T=0.8' in original.get('delta_window',''):
            s['T']='0.8'
        if original.get('split_L_D'):
            s['prefix']=original['split_L_D']
        pid=old['profile_id']
        if pid in ['COMMON588','OUTER32_EXPLORATION','PR_VS_PR_ORIGINAL_RUN01','PR_VS_PR_RECONSTRUCTED','S2_NEW_AND_RECOVERED','H468_INDEPENDENT_ENDPOINTS','H2_ORIGINAL_P','H2_DIAGNOSTICS_COMPARE']:
            s.update(geometry=original.get('geometry','linear;adjacent1.00Angstrom'),basis='STO-3G',T='0.8',P='0',
              DF_rank_policy=original.get('df_rank_or_policy','SEE_PROFILE_AND_ORIGINAL_INPUT;tol-only1e-8'),
              partition='candidate-specific fixed deterministic/random partition;see source',
              metric_unit='C=compiled full-wrapper RZ count;N=analytic quantum shots;G=sum_axis(N*C);n=classical trajectories')
        if pid=='COMMON588':
            s.update(category='PF構成系列;Baseline系列',system={'PF_COMMON_H2':'H2','PF_COMMON_H4':'H4','PF_COMMON_H6':'H6'}.get(old['original_series_id'],'H2/H4/H6'),epsilon='0.05;0.01;0.005;0.001;0.0005',evidence_type='FIXED_GRID_EXPLORATORY_SAVED_POINT',adoption_status='ADOPTED_FINITE_FIXED_GRID_POINT_RESULT_NOT_FINAL_FREEZE',result_status='SEE_CANDIDATE_ELIGIBILITY;not_all_compiled',q='1;2;4;8',prefix=prof['candidates_and_schedule'],r='PF:not_applicable;PR:see original allocation',K='PF:not_applicable;PR:4',sampling_law='deterministic PF;comparison PR law in fixed profile',n='PF deterministic: not_applicable;PR comparison:group-specific in saved table')
            if old['original_series_id'] in ['PF_COMMON_H2','PF_COMMON_H4','PF_COMMON_H6']:
                s['DF_rank_policy']={'PF_COMMON_H2':'tol-only1e-8;rank3','PF_COMMON_H4':'tol-only1e-8;rank10','PF_COMMON_H6':'tol-only1e-8;rank19'}[old['original_series_id']]
        elif pid=='OUTER32_EXPLORATION':
            s.update(category='PR探索系列',evidence_type='EXPLORATION_NOT_CONFIRMATION',K='4',prefix='1(H2);3(H4);5(H6)',q='1;2;4;8',sampling_law='ABS_TIME_PER_OUTER32;see original signed event law',n='candidate-specific;see fixed group table',adoption_status='EXPLORATION_CONTEXT;not_blanket_formal_claim_adoption')
        elif pid in ['PR_VS_PR_ORIGINAL_RUN01','PR_VS_PR_RECONSTRUCTED']:
            s.update(category='独立確認系列',system='H2/H4/H6',epsilon='0.0005',K='4',sampling_law='ABS_TIME_PER_OUTER32',evidence_type='FIXED_PR_VS_PR_CONFIRMATION;H6_RUN01_NOT_EVALUATED_IF_ORIGINAL',n='group-specific;see profile',adoption_status='CONTRACT_SCOPED_RESULT;not_latest_standard_Y4_confirmation')
        elif pid=='H468_INDEPENDENT_ENDPOINTS':
            s.update(category='独立確認系列;論文用固定系列',system=original.get('model','H4/H6/H8'),evidence_type='INDEPENDENT_FINITE_POINT;COHORTS_SEPARATE',epsilon='0.05;0.0005;see row',n='H4 1108/78;H6 48/104;H8 48/48;select original group',adoption_status='ADOPTED_SCOPED_EVIDENCE;H8_PUBLIC_AUDIT_QUALIFICATION')
        elif pid=='H2_ORIGINAL_P':
            s.update(category='独立確認系列',system='H2',DF_rank_policy='tol-only1e-8;rank3',evidence_type='ORIGINAL_H2_P;A_EXPLORATION_SEPARATE',n='P 80/12;A n8 kept separate',adoption_status='SEE_ORIGINAL_ADOPTION;not_in_H468_main_table')
        elif pid=='H2_DIAGNOSTICS_COMPARE':
            s.update(category='診断系列',system='H2',DF_rank_policy='tol-only1e-8;rank3',epsilon='0.0005',evidence_type='DIAGNOSTICS_OR_STATIC_METADATA_NOT_COST_WIN',adoption_status='STRICT_COMPARE_BLOCKED',result_status='DIAGNOSTICS_COMPLETED;STRICT_COMPARE_NOT_EXECUTED;see identity updates h2_diagonal/h2_static')
        elif pid in ['S2_NEW_AND_RECOVERED','CNG_FIXED_N','OLD_S2_ORIGINAL_27']:
            s.update(category='Costモデル系列',system='H2/H4/H6/H8',evidence_type='CONDITIONAL_SAVED_PACKET;H6_H8_AUXILIARY_FAIL_IF_NEW_S2',adoption_status='CONDITIONAL_REFERENCE;not_blanket_PASS',n='original_H2_and_new_other_systems_not_pooled')
        elif pid=='H6_RUN02_UNIT_COSTS':
            s.update(category='独立確認系列;診断系列',system='H6',metric_unit='compiled unit-cost axis fields only',n='32 task witnesses;64 axes;not_completed_aggregate_confirmation',evidence_type='UNIT_COST_WITNESS_ONLY',adoption_status='NOT_ACCEPTED_AS_COMPLETE_AGGREGATE',result_status='END_AUDIT_NOT_FOUND;SOURCE_NOT_ACQUIRED')
        elif pid=='NON_DF_SEPARATE_SCOPE':
            s.update(category='診断系列',evidence_type='SEPARATE_NON_DF_SCOPE_OR_TECHNICAL',adoption_status='REFERENCE_ONLY_OUTSIDE_MAIN_DF_COMPARISON')
        if old['series_key']=='ABE:H4_TOL_ONLY':
            s.update(category='PR探索系列;独立確認系列;診断系列',evidence_type='MIXED_STAGES:A exploration;S_STATS design;P independent;not pooled',n='P 1108/78;A separate per-stage manifest',adoption_status='P_SCOPED_ADOPTION;not_all_stage_guarantee')
        if old['original_series_id']=='H4_GEOMETRY_LEGACY':
            s.update(category='Geometry系列;Baseline系列',evidence_type='LEGACY_DF12_GEOMETRY_POINT_MAP',adoption_status='SCOPED_PRIOR_EVIDENCE;not_DF10',metric_unit='saved compiled RZ total-cost and baseline maps;per-field units')
        if old['original_series_id'] in ['H6_NATIVE_B0I','H8_PRODUCTION','H468_NEW_A']:
            s['category']='Baseline系列;PR探索系列'
        if old['original_series_id'] in ['MECHANISM_SAVED','static_claim_audit']:
            s['category']='診断系列;Costモデル系列'
        if old['original_series_id']=='PAPER_INDEXES':
            s.update(category='論文用固定系列',evidence_type='REFERENCE_METADATA_NOT_NEW_OBSERVATIONS')
        result.append(s)
    return result


def recent_rows():
    specs=[
      ('H2_FOUR_METHOD','H2','3','1','3','4/2','33/34','H2_four','PR探索系列','EXPLORATORY_DIAGNOSTICS_ACCOUNTING;strict identity separate'),
      ('H4_FOUR_METHOD','H4','10','3','10','8/4','48/33','H4_four','PR探索系列','EXPLORATORY_SAVED_ACCOUNTING'),
      ('H6_FOUR_METHOD','H6','19','5','19','8/4','33/33','H6_four','PR探索系列','SAVED_RETRIEVAL_NO_NEW_H6_CALCULATION'),
      ('H246_CNG_DECOMPOSITION','H2/H4/H6','3/10/19','1/3/5','3/10/19','H2 4/2;H4/H6 8/4','see same underlying exploration groups','cng/cost_decomposition_22.csv','Costモデル系列','SAVED_APPROVED_DECOMPOSITION;not_independent_acquisition'),
      ('H46_SAME_Q','H4/H6','10/19','3/5','10/19','4;8','same exploration groups;not new samples','batch/same_q_comparisons_8.csv','PF構成系列;診断系列','NEGATIVE_CONTROL_NO_FULL_PR_REVERSAL'),
      ('H2_PRECISION_COMPLETION','H2','3','1','3','4/2','33/34','batch/H2_new_endpoints_16.csv','PF構成系列','SAVED_PRECISION_REANALYSIS;not_new_acquisition'),
      ('H4_STANDARD_Y4_CONFIRMATION','H4','10','3',NA,'8/4','S2 25;Y4 45','adopt_source/result_results.json','独立確認系列','REPRODUCED_APPROXIMATE'),
      ('H6_STANDARD_Y4_CONFIRMATION','H6','19','5',NA,'8/4','S2 23;Y4 32','adopt_source/result_results.json','独立確認系列','REPRODUCED_APPROXIMATE'),
      ('H46_STANDARD_Y4_FAILED_ATTEMPT','H4/H6','10/19','3/5',NA,'8/4','new accepted trajectories 0','adopt_source/failed_end_audit.json','診断系列','TECHNICAL_STOP;separate_success_namespace')]
    result=[]
    for sid,system,rank,prpref,fullpref,q,n,ref,cat,status in specs:
        confirm=sid.endswith('_CONFIRMATION'); fail=sid.endswith('_FAILED_ATTEMPT')
        s={k:UNKNOWN for k in FIELDS}
        s.update(study_id=sid,record_role='ADDITIONAL_ACCESS_VIEW_NOT_AN_INDEPENDENT_EXPERIMENT_COUNT',category=cat,purpose=sid,
          system=system,geometry='linear;adjacent1.00Angstrom',basis='STO-3G',DF_rank_policy='tol-only1e-8;rank'+rank,
          Hamiltonian_state_reference='adopt/adoption_record.json#/conditions' if confirm or fail else 'cng/REPORT.md;batch/REPORT.md;original fixed inputs',
          T='0.8',epsilon='0.001' if confirm or fail or sid=='H46_SAME_Q' else '0.0005' if sid.endswith('FOUR_METHOD') else 'see fixed precision rows',P='0',
          delta_or_signed_time='See fixed q/signed-piece schedule;no numerical reconstruction by this index',method_formula='PR S2 / standard Yoshida4' if confirm or fail else 'Full/PR S2 / standard Yoshida4',prefix='PR '+prpref+';Full '+fullpref,
          q=q,r='SOURCE_ABS_TIME_ALLOCATION;not_inferred',K='4',partition='fixed deterministic prefix and random remainder;see input',
          sampling_law='ABS_TIME_PER_OUTER32;signed pieces unchanged',cohort='FIXED_INDEPENDENT_S2_AND_NEW_STANDARD_Y4;NO_POOLING' if confirm else 'FAILED_ATTEMPT_NOT_SUCCESS' if fail else 'SAVED_EXPLORATION;NO_NEW_COHORT',
          n=n,compiler_native='Qiskit1.3.0;rz/sx/x/cx;opt1;seed17;symmetric-directional;full Hadamard without preparation;runtime identity qualification retained',
          metric_unit='C=compiled RZ count;N=analytic quantum shots;G=sum_axis(N*C);n=classical cost trajectories',
          evidence_type='CONDITIONAL_APPROXIMATE_INDEPENDENT_COST_DIRECTION' if confirm else 'TECHNICAL_FAILURE_NOT_SCIENTIFIC_DIRECTION' if fail else 'SAVED_EXPLORATORY_POINT_OR_ACCOUNTING',
          result_status=status,adoption_status='FORMALLY_ADOPTED_CONDITIONAL_INDEPENDENT_COST_DIRECTION' if confirm else 'NOT_ADOPTED_AS_SCIENTIFIC_RESULT' if fail else 'SAVED_RESULT_CONTEXT;NO_NEW_FORMAL_ADOPTION',
          SOURCE='a2a53698a72da7280d217adc7931938121ed732d;numeric c028f1c2885afdba65ef8c62c075738aadd7323d' if confirm else 'SEE_FIXED_REPORT_SOURCE',
          result_commit=Y4 if confirm else '4ce444eb5b7cd1dad7cf61f149aa182949ad60a1' if fail else BATCH if ref.startswith('batch/') else CNG,
          result_reference=ref,condition_reference='adopt/adoption_record.json#/conditions' if confirm or fail else 'cng/REPORT.md;batch/REPORT.md',
          upstream_profile=NA,limitations=LIMIT+';DIFFERENT_Q_IS_NOT_PURE_ORDER_CAUSATION',original_index_selector=NA,new_independent_observations_from_indexing='0')
        if sid=='H2_FOUR_METHOD': s['SOURCE']='aca689060d302b4e1f37d32d7075c9266b637f56'
        if sid=='H4_FOUR_METHOD': s['SOURCE']='4cbbc1b6b49bd64e765ccc596ec5016a1a4d5503'
        if sid=='H6_FOUR_METHOD': s['SOURCE']='recovery adapter 9a22f0604281f6c602e819418c936d9844830262;PR evaluation 9a77d2b0a0b87e246e85f085b37f5923c18e27bb;acquisition 5a448d94ea389ada6e2b2755de6a215ed2034679'
        if fail:
            s['SOURCE']='a94735946e8cf15ffea5c93ae99000f7ab83d23a'
        if sid=='H246_CNG_DECOMPOSITION': s['SOURCE']='2e59a55beec8159c7832ff4ff02513b8976c0eb9'
        if sid in ['H46_SAME_Q','H2_PRECISION_COMPLETION']: s['SOURCE']='1c5bb8a65a7aa7c90707b783eb5d6cf97f4d0bfe'
        result.append(s)
    return result


def comparisons():
    specs=[
      ('Q01','H4_STANDARD_Y4_CONFIRMATION','H6_STANDARD_Y4_CONFIRMATION','REFERENCE_ONLY','Different physical system and independent cohort; cite separately.','adopt/adoption_record.json'),
      ('Q02','H46_SAME_Q','H4_STANDARD_Y4_CONFIRMATION;H6_STANDARD_Y4_CONFIRMATION','LIMITED','Same epsilon; q4/q8 controls and q8-to-q4 primary are different contrasts; controls are saved exploration.','batch/same_q_directions_4.csv'),
      ('Q03','ABE:H4_GEOMETRY_LEGACY','H4_FOUR_METHOD','REFERENCE_ONLY','legacy DF12 vs tol-only DF10; no cross-series direct G ratio or geometry attribution.','three/scientific_profiles.json'),
      ('Q04','GPU01:PF_ADOPTION','GPU02:H246_saved_cost_results','LIMITED','Adopted common grid versus saved PR finite-domain exploration; asymmetric candidate domains; use saved all/within-formula tables.','pf/comparisons_all_formulas_15.json'),
      ('Q05','GPU01:PF_ADOPTION','H2_FOUR_METHOD','REFERENCE_ONLY','COMMON588 and new H2 DIAGNOSTICS empirical allowance/native binding contracts differ; no silent endpoint replacement.','h2_strict'),
      ('Q06','GPU01:H246_RECONSTRUCTED','H4_STANDARD_Y4_CONFIRMATION;H6_STANDARD_Y4_CONFIRMATION','LIMITED','Existing S2 prices reused, not extra S2 samples; original m5/q1 confirmation is not standard Y4/q4.','adopt/evidence_roles.csv'),
      ('Q07','GPU02:H246_independent_confirmation_run01','GPU01:H246_RECONSTRUCTED','REFERENCE_ONLY','Original and reconstructed input/cohort contracts differ; do not pool or repair original H6 failure.','three/scientific_profiles.json'),
      ('Q08','GPU02:H6_independent_recovery_run02','GPU01:H246_RECONSTRUCTED','NOT_COMPARABLE_AS_COMPLETED_CONFIRMATION','Unit-cost witness lacks original end audit/SOURCE; no aggregate PASS inference.','three/gaps_and_requests.json'),
      ('Q09','GPU01:S2_RECOMPUTED','GPU01:CNG_ACCOUNTING','LIMITED','Same selected packet only after matching exact source/input/cohort; G00/G01 fix N; N_circuit/G10/G11 remain null.','three/scientific_profiles.json'),
      ('Q10','ABE:H8_PR_CONFIRMATION','ABE:MECHANISM_SAVED','LIMITED_SAME_OBSERVATIONS','H8 mechanism alias is the same n48 evidence; no independent replication increment.','three/duplicate_version_checks.json'),
      ('Q11','ABE:H4_TOL_ONLY','GPU02:H468_original_confirmation','LIMITED_SAME_OBSERVATIONS','Same selected independent P rows are cross-environment aliases, not independent facilities or additive n.','main_table'),
      ('Q12','H246_CNG_DECOMPOSITION','H2_PRECISION_COMPLETION','LIMITED','H2 other four precisions were completed later under accepted policy; quote combined 60 endpoints; older report pending status is historical.','batch/primary_endpoints_60.csv'),
      ('Q13','H46_SAME_Q','H46_SAME_Q','FULL_WITHIN_DECLARED_CONTRAST','Only each fixed system/method/q/epsilon row S2-vs-standardY4 pair with same input/partition/policy. Across q is a different contrast.','batch/same_q_comparisons_8.csv'),
      ('Q14','H4_STANDARD_Y4_CONFIRMATION','H4_STANDARD_Y4_CONFIRMATION','FULL_WITHIN_DECLARED_CONTRAST','Only its saved independent S2/q8 versus standardY4/q4 fixed-shot comparison; paired cos/sin within trajectory, independent formula groups.','adopt_source/result_independent_comparisons_2.csv'),
      ('Q15','H6_STANDARD_Y4_CONFIRMATION','H6_STANDARD_Y4_CONFIRMATION','FULL_WITHIN_DECLARED_CONTRAST','Only its saved independent S2/q8 versus standardY4/q4 fixed-shot comparison; no general order-effect or population certificate.','adopt_source/result_independent_comparisons_2.csv'),
      ('Q16','GPU01:NON_DF_H4_READINESS','GPU01:PF_ADOPTION','REFERENCE_ONLY','Separate non-DF contract and technical readiness; no transfer of Pauli/native cost metrics. C0 is not Track A evidence.','three/research_index.csv')]
    return [dict(zip(['comparison_id','left_study','right_study','comparability','conditions_and_exclusions','evidence_reference'],x)) for x in specs]


def claim_rows():
    cem=load(ADOPT,A+'claim_evidence_matrix_v3.csv')
    rows=[]
    for c in cem:
        cid=c['id'];primary='adopt/claim_evidence_matrix_v3.csv'
        if cid.startswith('COMMON588'):
            refs='pf/comparisons_all_formulas_15.json;pf/comparisons_within_formula_45.json;pf/denominator_decomposition_10.json'
            placement='MAIN_CANDIDATE_OR_SUPPLEMENT_FIXED_GRID;NOT_ADOPTED_PLACEMENT'
            contrary='H4/H6 other three precisions PF lower;within-formula and all-formula domains differ'
        elif 'STANDARD_Y4' in cid:
            refs='adopt_source/result_results.json;adopt_source/result_independent_groups_4.csv;adopt_source/result_independent_comparisons_2.csv;batch/same_q_directions_4.csv'
            placement='MAIN_CANDIDATE_CONDITIONAL_INDEPENDENT_DIRECTION;NOT_ADOPTED_PLACEMENT'
            contrary='Same-q q4 both Y4 lower; q8 both S2 lower; no reversal in these controls'
        elif cid.startswith(('CE_H4_DF10','CE_H6P','CE_H8_DF28')):
            refs='main_table';placement='EXISTING_MAIN_TABLE_CANDIDATE_NO_REPLACEMENT'
            contrary='CE_H4_DF10_P_1 NOT_REPRODUCED/PF lower; CE_H6P_Q2 reproduced PF lower'
        else:
            refs='adopt/evidence_ledger_v5.csv;three/claim_evidence_overlay.csv'
            placement='SUPPLEMENT_OR_CONTEXT_CANDIDATE;NOT_NEW_ADOPTION'
            contrary='Use CEM unresolved and original negative-history/selection qualification; no new generalization'
        if cid=='CE_M_LEGACY_H4_P':contrary='Legacy DF12 independent confirmation is distinct from tol-only DF10 P'
        if cid=='CE5':contrary='H6 full-domain noncancellation missing; subset values cannot certify all136 candidates'
        if cid=='CE6':placement='REQUIRED_SCOPE_LIMITS;NOT_NEW_ADOPTION'
        rows.append(dict(claim_id=cid,claim_literal=c['claim_candidate'],canonical_cem_reference=primary+'#id='+cid,
          supporting_references=refs,original_result_commits_literal=c['fixed_commits'],original_result_paths_literal=c['result_files'],SOURCE_literal=c['SOURCE'],
          scientific_conditions_literal=c['conditions'],evidence_strength_literal=c['evidence_strength'],
          statistical_precision_limits_literal=c['unresolved'],counterexample_or_negative_control=contrary,
          maximum_usage_strength='SOURCE_SCOPED_ONLY: '+c['evidence_strength'],placement_candidate=placement,
          related_saved_results='cng/cost_decomposition_22.csv;cng/shot_relationship_axes_88.csv' if cid=='CE4' else 'pf/old_vs_common_15.json' if cid=='CE1' else NA,
          latest_adoption_authority='adopt/paper_manifest.json;CEM row authority retained',new_claim_adoption='false'))
    return rows


def paper_assets():
    specs=[
      ('MAIN_TABLE','main_table','Existing six independent endpoint main rows','EXISTING_MAIN;NO_CHANGE','n classical cost samples;N analytic quantum shots;G compiled-RZ weighted cost','Keep per-row audit status;H4 coarse negative;H6 coarse PF lower;H8 audit qualification'),
      ('FIGURE2','adopt/paper_manifest.json#/Figure2_original_files','Existing34 Figure2 files','EXISTING_FIXED_FIGURE;NO_REGENERATION','Original plotted metric per render binding','LegacyDF12/exploration and independent H8 are separate panels;A82 later completion does not replace old files'),
      ('GEOMETRY','three/references.csv#ref_id=R0135','H4 geometry and native B0+I maps','SUPPLEMENT_OR_CONTEXT_CANDIDATE','Original saved G/RZ map units','legacy DF12;read original geometry/DF policy and measured subset'),
      ('BASELINE','adopt/claim_evidence_matrix_v3.csv#id=CE1','Deterministic baseline strengthened comparisons','MAIN_OR_SUPPLEMENT_CANDIDATE','Saved total G and reductions;measured/model separate','Finite registered domain;baseline changes denominator;no new winner'),
      ('PR_DOMAIN','adopt/claim_evidence_matrix_v3.csv#id=CE2','New A candidate-domain extension','SUPPLEMENT_CANDIDATE','Saved finite point G','Exploration n8 versus old union n32;selection effect;H8 independent follow-up is distinct'),
      ('COMMON588','pf/candidate_index.csv','588 fixed candidate metadata;gzip dataset through pf/paper_manifest.json','MAIN_OR_SUPPLEMENT_CANDIDATE','Saved C/N/G with original status','2940 endpoints does not mean588 new compiles;do not pool with PR confirmation'),
      ('COMMON_ALL','pf/comparisons_all_formulas_15.json','15 all-formula comparisons','MAIN_OR_SUPPLEMENT_CANDIDATE','Saved point G','Different question from45 within-formula comparisons'),
      ('COMMON_WITHIN','pf/comparisons_within_formula_45.json','45 within-formula comparisons','SUPPLEMENT_CANDIDATE','Saved point G','m5_best and standardY4 separate'),
      ('PRECISION','batch/primary_directions_15.csv','H2/H4/H6 five registered precision directions','MAIN_OR_SUPPLEMENT_CANDIDATE','Saved lower-cost labels;no new ratio','Fixed q pairs;not best candidate at each epsilon;no interpolation'),
      ('PRECISION_VALUES','batch/primary_endpoints_60.csv','Saved60 endpoint C/N/G/u/bias/headroom fields','SUPPLEMENT_CANDIDATE','Original metric columns and n/SE','H2 later accepted allowance policy;no fresh accounting'),
      ('SAME_Q','batch/same_q_comparisons_8.csv','H4/H6 q4/q8 and epsilon=.001 eight contrasts','MAIN_OR_SUPPLEMENT_NEGATIVE_CONTROL','Saved G/C/N decomposition','No Full/PR reversal at same q;not independent confirmation'),
      ('Y4_GROUPS','adopt_source/result_independent_groups_4.csv','Four independent S2/Y4 cost groups','MAIN_CANDIDATE_CONDITIONAL','Saved meanG/pairedSE;group n','S2 reused n25/23;newY4 n45/32;no pooling'),
      ('Y4_COMPARISONS','adopt_source/result_independent_comparisons_2.csv','H4/H6 approximate two-comparison family','MAIN_CANDIDATE_CONDITIONAL','Saved difference/ratio/normal approximate interval','Family size2 not77;not exact coverage/population or unseen-tail bound'),
      ('CNG','cng/cost_decomposition_22.csv','22 saved exact accounting decompositions','SUPPLEMENT_CANDIDATE','Saved R_Ceff/R_N/G and additive cross term','Explanation of fixed accounting,not operation-level or pure-order causal attribution'),
      ('SHOT','cng/shot_relationship_axes_88.csv','Saved bias/u/B/headroom and shot relationships','SUPPLEMENT_CANDIDATE','Original88 axis fields','Analytic N is not measured quantum shots;empirical u not rigorous certificate'),
      ('RTE','cng/representative_RTE_6.csv','Six registered first trajectories','CONDITIONAL_DIAGNOSTIC','Saved event/basis-adjacency counts','Not all214 trajectories or whole-support cost model;unseen cost not0'),
      ('THEORY_COMPILED','three/references.csv#ref_id=R0083','Separate S2 theory/compiled20-condition packet','CONDITIONAL_REFERENCE','Saved C and theoryG','H6/H8 auxiliary FAIL;H8 new state/hash not old input identity'),
      ('FIXED_N','three/references.csv#ref_id=R0089','Fixed-N CNG G00/G01 packet','CONDITIONAL_REFERENCE','G00/G01 at fixedN;N_circuit/G10/G11 null','Not independent N equality;H6/H8 qualification retained'),
      ('RARE','three/references.csv#ref_id=R0215','H8 rare-tail closure policy and observed-stratum context','LIMITS_AND_SUPPLEMENT','Observed samples only;unseen cost unknown','Existing limited claim closed;no automatic new sampling'),
      ('NEGATIVE_HISTORY','three/gaps_and_requests.json','FAIL/NOT_EVALUATED/missing receipt source list','REQUIRED_QUALIFICATION','No missing values set0','H6run01 SIGBUS;run02 missingend;newS2 H6/H8 FAIL;old H6 parent unknown')]
    rows=[dict(zip(['asset_id','source_reference','content_scope','paper_use_candidate','metric_semantics','limits'],x),numerical_recalculation_required_now='false',new_figure_generated='false') for x in specs]
    for r in load('1923fce8f7461513d1bc47b5061e87cd70917e87','artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/main_table.csv'):
        rows.append(dict(asset_id='MAIN_ROW_'+r['comparison_id'],source_reference='main_table#comparison_id='+r['comparison_id'],content_scope=r['system']+';epsilon='+r['epsilon']+';PR='+r['PR_id']+';PF='+r['PF_id'],paper_use_candidate='EXISTING_MAIN_ROW_NO_REPLACEMENT',metric_semantics='saved_direction_status='+r['saved_direction_status']+';lower_saved_point_cost='+r['lower_saved_point_cost'],limits=r['audit_status']+';'+r['interpretation_limits'],numerical_recalculation_required_now='false',new_figure_generated='false'))
    return rows


def gaps():
    impact1={'O00002','ORIGINAL_RUN01_SOURCE','LOCAL_ONLY_SUPPLEMENTS','NEW_BUNDLE_PUBLICATION','EXECUTION_HOST','LOCAL_VARIANTS'}
    impact2={'H6_COMMON_PARENT','H8_PUBLIC_AUDIT','RARE_UNKNOWN','PHYSICAL_FAIL','H6_RUN01_FAILURE'}
    impact4={'H6_NONCANCEL_FULL_DOMAIN','GAP_001','GAP_008'}
    impact5={'NON_DF_RUN01','H4_READINESS'}
    rows=[]
    for x in load(THREE,T+'gaps_and_requests.json')['items']:
        key=x['id'];level=1 if key in impact1 else 2 if key in impact2 else 4 if key in impact4 else 5 if key in impact5 else 3
        update='Original bounded snapshot status retained;not a fresh all-environment search.'
        ref='three/gaps_and_requests.json#id='+key
        if key in ['O00002','H2_COMPARE_IDENTITY']:
            update='Original recovered and operation-stream evidence documented later;strict full identity remains unresolved. A diagonal primitive plan unexecuted and B2 static scope distinct. Exploratory H2 accounting did not release strict COMPARE.'
            ref+=';h2_diagonal;h2_static;h2_strict'
        if key=='NEW_BUNDLE_PUBLICATION':
            update='Historical old-bundle approval gap only. Current user request plus shared AGENTS authorizes public-only commit/push;not retroactive proof of original receipts.'
        if key=='LAYOUT_AND_ADOPTION':
            update='H4/H6 conditional Y4 evidence adopted at09e71910;paper central claim/layout and finalfreeze remain separate pending decisions.'
            ref+=';adopt/adoption_record.json'
        if key in impact4:
            update+=' No promotion of invalid/full-domain/excluded values. This does not revoke unrelated scoped accepted results.'
        rows.append(dict(issue_id=key,impact_class=str(level),original_status_literal=x['current_status'],current_scope_update=update,
          affected_claims=';'.join(x['affected_claims']) or NA,paper_impact=x['needed_only_if'],needed_only_to_strengthen=x['claim_dependency'],
          minimal_action=x['proposed_action'],responsible=x['recovery_owner'],source_reference=ref,new_scientific_execution_authorized='false',automatic_new_sampling_compile_reanalysis='false'))
    extras=[
      ('EMPIRICAL_REFERENCE',2,'Experience-based allowance and saved full-H reference accuracy remain conditional.','All fixed signal cost claims','Report assumptions;stronger certified bias claim needs separate evidence;no automatic recomputation.'),
      ('CANDIDATE_ASYMMETRY',2,'PF common domain and saved PR candidate domain differ.','COMMON588;CE1;CE2','Name exact registered domain and measured/model cost classes;no global optimizer claim.'),
      ('ORDER_Q_CONFOUNDING',2,'Different-q fixed configurations and same-q controls answer different questions.','CE_H4_STANDARD_Y4_FIXED_SHOT_DIRECTION;CE_H6_STANDARD_Y4_FIXED_SHOT_DIRECTION','State q8-to-q4;include saved same-q negative controls;no pure-order causal effect.'),
      ('POPULATION_AND_TAIL',3,'Saved approximate family intervals do not give rare-tail/population rank certificates.','H46 conditional confirmation','Keep family2 normal approximation and finite-sample caveats;new evidence only if stronger claim selected.'),
      ('STANDARD_Y4_NOT_M5',2,'Existing m5_best confirmations do not establish standard Yoshida4 result;adopted standardY4 now separate.','High-order PR claims','Use four-group adopted standardY4 record and explicit reused S2 identity;no pooling.'),
      ('FT_RPE_EXTRAPOLATION',5,'Compiled RZ finite-time signal cost is not end-to-end energy/QPE/RPE or FT T cost.','All cost claims','Keep as separate future research;prior-art error/cost definitions differ.'),
      ('FORMAL_POLICY_VERSION',1,'Formal attachment SHA7ce342... not acquired;available050e5d... has no10.3.','Metadata organization only','Apply explicit user principles;record version gap without stopping independent organization.'),
      ('LEVEL1_REVIEW_20261024',1,'Exact requested Level1 review file not acquired at known worktree/shared-root locations or fixed catalog.','No new adoption authority inferred','Use existing M1-M3 review mapping/receipt;do not claim read;needed only for relying on new review wording.'),
      ('C0_SECOND_STUDY',5,'Prediction/configuration-selection C0 belongs to a separate worktree/research contract.','Track A boundary','No hypotheses/models/results/authorization transfer;read-only shared source is same evidence.')]
    for key,level,status,affected,action in extras:
        rows.append(dict(issue_id=key,impact_class=str(level),original_status_literal=NA,current_scope_update=status,affected_claims=affected,paper_impact=action,needed_only_to_strengthen='CLAIM_DEPENDENT_NO_BLANKET_RERUN',minimal_action=action,responsible='ユーザー (stronger claim decision);Codex STOP',source_reference='adopt/REPORT.md;three/M1_M3_review_mapping.md' if key not in ['FORMAL_POLICY_VERSION','LEVEL1_REVIEW_20261024','C0_SECOND_STUDY'] else 'policy_and_scope.json',new_scientific_execution_authorized='false',automatic_new_sampling_compile_reanalysis='false'))
    return rows


def schema():
    return dict(schema='TRACK_A_REFERENCE_VIEWS_V1',layers={'1':'Immutable original results at fixed commits','2':'study_index.csv: metadata and original row selectors, not scientific result table','3':'claim_usage.csv/paper_data_index.csv/comparability.csv: access and usage views,not new adoption'},
      missing_values={UNKNOWN:'Not recorded/acquired in the bounded source;never impute0',NA:'Field does not apply to this metadata/technical record',NV:'Declaration or pointer exists but this index did not establish it'},
      field_scope='Family profile describes a family,not exact same condition for every member. Original row/input/candidate controls specific scope. Descriptive strings are not numeric observations.',
      counts='49 existing catalogue entries plus9 current access views are NOT58 independent experiments;cross-host aliases and repeated source rows never add n.',
      metrics={'n':'Classical PR trajectory cost samples;paired cos/sin are one sample','N':'Sufficient analytic quantum measurement shots;not physical shot execution here','C':'Per-axis compiled RZ instructions unless explicitly theory/model/depth/CX','G':'Original saved sum_a N_a C_a;no new arithmetic here','SE':'Original paired cost-sample SE,not shot SE or rigorous uncertainty'},
      reference_resolution='source_reference uses source_catalog.json.sources key;split at # for selector. three/references.csv#ref_id resolves in fixed upstream catalog. commit:path#/pointer directly resolves a fixed original manifest row.',
      roles='Catalogue custody is not original execution host;REPRODUCED can mean preregistered PF-cheaper direction. TEST_PASS is not scientific superiority.',
      adoption='Original CEM v3/ledger v5/paper manifest are authoritative;these views cannot edit or supersede them.',study_fields=FIELDS)


def render():
    rows=old_rows()+recent_rows()
    policy=dict(date_jst='2026-10-24',base_commit=BASE,branch='codex/track-a-data-systematization-20261024',
      user_authorization='Track A metadata organization;public-only commit and push;mandatory STOP',
      shared_instructions_read='AGENTS.md;worktree AGENTS.md;available common research instructions',
      expected_formal_sha256='7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f',
      available_formal_sha256='050e5dc7055a1c114c35eb17913f3ee050092b57444dcffb70949e896548ac28',
      expected_original_acquired=False,available_sections='8.1;8.4;10.1;10.2;12;10.3 absent in available version',
      explicit_user_principle='Prioritize scientific understanding/reuse and useful access;no excess byte audits or blanket reruns',
      Level1_review_20261024='NOT_ACQUIRED;exact filename absent at known supplied shared root/current worktree;no broader search;old M1-M3 mapping reused',
      second_research_C0='OUT_OF_SCOPE;no wait/no access/no changes/no transfer of authorization or independent observations',
      prior_art='Existing prior-art delta referenced only;no new literature investigation or detailed scientific review',
      reused_code=['integrate_h468_paper_evidence.csv_text/json_text/pointer','adopt_h46_standard_y4_confirmation.git_bytes/csv_saved/json_saved'],
      inspected_not_run=['f75a7c25:scripts/adopt_h246_pf_common_results.py (fixed588 adoption only,not appropriate to execute)','existing science runners and grant code are never imported'],
      scientific_calls=dict.fromkeys(['sampling','primitive_evaluation','circuit_generation','compile','simulation','new_cost_ratio_rank_statistics','new_figures','scientific_grants','old_science_SOURCE_execution'],0),
      claim_adoption_changed=False,manuscript_written=False,old_strict_COMPARE='BLOCKED',mandatory_STOP=True)
    sources=make_sources()
    return {'study_index.csv':csv_text(rows),'comparability.csv':csv_text(comparisons()),'claim_usage.csv':csv_text(claim_rows()),'paper_data_index.csv':csv_text(paper_assets()),'unresolved_issues.csv':csv_text(gaps()),'schema.json':json_text(schema()),'source_catalog.json':json_text(dict(schema='SCOPED_PINNED_SOURCE_REFERENCES_V1',sources=sources,upstream_reference_catalogue='three/references.csv',not_a_copy_of_3538_refs=True)),'policy_and_scope.json':json_text(policy)}


def reference_key(ref):
    return ref.split('#',1)[0]


def validate():
    b=ROOT/OUT
    got={n:(b/n).read_text() for n in render()}
    expected=render()
    require(got==expected,'EXACT_METADATA_VIEW_JOIN')
    rows=csv_saved((b/'study_index.csv').read_bytes());sources=json.loads((b/'source_catalog.json').read_text())['sources']
    require(len(rows)==58 and len({r['study_id'] for r in rows})==58,'49_PLUS_9_ACCESS_VIEWS_UNIQUE')
    require({r['original_index_selector'] for r in rows[:49]}=={r['series_key'] for r in load(THREE,T+'research_index.csv')},'ALL49_OLD_KEYS_PRESERVED')
    require(all(r['new_independent_observations_from_indexing']=='0' for r in rows),'NO_NEW_OBSERVATIONS_OR_POOLING')
    require(all(set(r)==set(FIELDS) and all(v!='' for v in r.values()) for r in rows),'NO_BLANK_UNKNOWN_OR_ZERO_IMPUTATION')
    cem=load(ADOPT,A+'claim_evidence_matrix_v3.csv');view=csv_saved((b/'claim_usage.csv').read_bytes())
    require([r['claim_id'] for r in view]==[r['id'] for r in cem],'CEM23_KEYS_EXACT')
    require(all(v['claim_literal']==c['claim_candidate'] and v['scientific_conditions_literal']==c['conditions'] and v['evidence_strength_literal']==c['evidence_strength'] and v['statistical_precision_limits_literal']==c['unresolved'] for c,v in zip(cem,view)),'CEM_CONDITIONS_STRENGTH_LIMITS_LITERAL')
    require(all(v['new_claim_adoption']=='false' for v in view),'NO_NEW_CLAIM_ADOPTION')
    for n,fields in [('claim_usage.csv',['supporting_references','canonical_cem_reference']),('paper_data_index.csv',['source_reference']),('comparability.csv',['evidence_reference'])]:
        for r in csv_saved((b/n).read_bytes()):
            for field in fields:
                for ref in r[field].split(';'):require(reference_key(ref) in sources,'DANGLING_SOURCE:'+ref)
    for k,v in sources.items():
        require(re.fullmatch('[0-9a-f]{40}',v['commit']) is not None and not Path(v['path']).is_absolute() and '..' not in Path(v['path']).parts,'SAFE_FIXED_SOURCE:'+k)
        require(re.fullmatch('[0-9a-f]{64}',v['sha256']) is not None,'SHA_PIN:'+k)
        subprocess.check_output(['git','cat-file','-e',v['commit']+':'+v['path']],cwd=ROOT)
    require('n=classical' in next(r for r in rows if r['study_id']=='H4_STANDARD_Y4_CONFIRMATION')['metric_unit'],'N_VS_n_EXPLICIT')
    require(next(r for r in rows if r['study_id']=='ABE:H4_GEOMETRY_LEGACY')['DF_rank_policy']=='legacy DF12','LEGACY_DF12_SEPARATE')
    require(next(r for r in rows if r['study_id']=='H4_FOUR_METHOD')['DF_rank_policy']=='tol-only1e-8;rank10','TOL_DF10_SEPARATE')
    require(all('standard Yoshida4' in r['method_formula'] for r in rows if r['study_id'].endswith('_STANDARD_Y4_CONFIRMATION')),'STANDARD_Y4_NOT_M5')
    require(next(r for r in rows if r['study_id']=='GPU02:H6_independent_recovery_run02')['adoption_status']=='NOT_ACCEPTED_AS_COMPLETE_AGGREGATE','MISSING_END_NOT_PASS')
    require(any(r['comparability']=='NOT_COMPARABLE_AS_COMPLETED_CONFIRMATION' for r in comparisons()),'FAIL_MISSING_DISTINGUISHED')
    require(all(r['automatic_new_sampling_compile_reanalysis']=='false' for r in gaps()),'GAPS_NOT_AUTOMATIC_RERUNS')
    require(set(x['impact_class'] for x in gaps())==set('12345'),'FIVE_IMPACT_CLASSES')
    require(len([r for r in paper_assets() if r['asset_id'].startswith('MAIN_ROW_')])==6,'SIX_MAIN_ROWS_REFERENCED')
    require(any(r['asset_id']=='MAIN_ROW_H4_DF10_P_1' and 'NOT_REPRODUCED' in r['metric_semantics'] for r in paper_assets()),'NEGATIVE_DIRECTION_RETAINED')
    stop=json.loads((b/'STOP.json').read_text())
    require(stop['status']=='MANDATORY_STOP' and not stop['scientific_execution_performed'] and stop['old_strict_COMPARE']=='BLOCKED','STOP_RETAINED')
    policy=json.loads((b/'policy_and_scope.json').read_text())
    require(all(v==0 for v in policy['scientific_calls'].values()) and policy['old_strict_COMPARE']=='BLOCKED','ZERO_SCIENCE_STRICT_BLOCKED')
    require(policy['second_research_C0'].startswith('OUT_OF_SCOPE'),'SECOND_STUDY_NOT_IMPORTED')
    # Ensure both reused import modules are document-only, and this adapter has no scientific imports.
    module=ast.parse((ROOT/'scripts/index_track_a_research_data.py').read_text())
    allowed={'argparse','ast','csv','hashlib','io','json','pathlib','re','subprocess','adopt_h46_standard_y4_confirmation','integrate_h468_paper_evidence'}
    for node in ast.walk(module):
        if isinstance(node,ast.Import):require(all(a.name in allowed for a in node.names),'SCIENTIFIC_IMPORT_FORBIDDEN')
        if isinstance(node,ast.ImportFrom):require(node.module in allowed,'SCIENTIFIC_IMPORT_FORBIDDEN')
    changed=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    changed+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    allowed_paths={'PROJECT_MAP.md','docs/README.md','docs/research/研究概要・現状.md','scripts/README.md','scripts/index_track_a_research_data.py','docs/data_collection/README.md'}
    require(all(p.startswith(OUT) or p in allowed_paths for p in changed),'ORIGINAL_SOURCE_RESULT_MANIFEST_LEDGER_STOP_PROTECTED')
    for p in b.iterdir():
        if p.is_file():
            t=p.read_text()
            require(not re.search(r'/home/|/tmp/|ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9]+|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY',t),'PRIVATE_LOCATOR_OR_SECRET:'+p.name)
    checks=['EXACT_METADATA_VIEW_JOIN','49_PLUS_9_UNIQUE_VIEWS','ALL49_OLD_KEYS_PRESERVED','NO_NEW_OBSERVATIONS_OR_POOLING','NO_BLANK_OR_ZERO_IMPUTATION','CEM23_KEYS_EXACT','CEM_CONDITIONS_STRENGTH_LIMITS_LITERAL','NO_NEW_CLAIM_ADOPTION','REFERENCES_RESOLVE','FIXED_COMMIT_PATH_SHA_PINS','N_VS_n_EXPLICIT','LEGACY_DF12_SEPARATE','TOL_DF10_SEPARATE','STANDARD_Y4_NOT_M5','MISSING_END_NOT_PASS','FAIL_MISSING_DISTINGUISHED','GAPS_NOT_AUTOMATIC_RERUNS','FIVE_IMPACT_CLASSES','SIX_MAIN_ROWS_REFERENCED','NEGATIVE_DIRECTION_RETAINED','ZERO_SCIENCE_STRICT_BLOCKED','SECOND_STUDY_NOT_IMPORTED','NO_SCIENTIFIC_IMPORT','ONLY_NEW_VIEW_AND_DISCOVERY_PATHS_CHANGED','NO_PRIVATE_LOCATOR_OR_SECRET']
    return dict(status='PASS',checks=checks,scope='Author technical metadata checks only;not independent scientific review or original execution re-audit',science_executed=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',action='store_true');ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    if args.build:
        (ROOT/OUT).mkdir(parents=True,exist_ok=True)
        for name,content in render().items():(ROOT/OUT/name).write_text(content)
    summary=validate()
    if args.build:
        (ROOT/OUT/'validation_summary.json').write_text(json_text(summary))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
