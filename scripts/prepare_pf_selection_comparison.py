"""Fixed-Git literal projection, joins and counts only; no cost arithmetic."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import prepare_pf_direct_compile_inventory as inv
from prepare_pf_q_model_comparison import csv_text

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import pf_selection_comparison_plan as policy

BASE='7430a386bcd4dc545107e413cd022ab7cb261968'
B='artifacts/resource_applicability/'
H6=B+'track_a_v1_reaccount_result/2026-10-16'
H8=B+'track_a_h8_six_axis_reaccount_run01/2026-10-17'
PREP=B+'track_a_pf_q_model_comparison_preparation/2026-10-20'
PLAN=B+'track_a_pf_direct_compile_plan/2026-10-19'
OUT=ROOT/(B+'track_a_pf_selection_comparison_preparation/2026-10-20')


def source(label,path,commit=BASE,fmt='json'):return inv.source(label,path,commit,fmt)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def readout(name):return json.loads((OUT/name).read_text())
def parse_candidate(cid):
    m=re.fullmatch(r'(B0I|B0|B1|B2)_p(\d+)_q(\d+)_(2nd|4th)(?:_r(\d+)_K(\d+))?',cid)
    if not m:raise ValueError('CANDIDATE_ID:'+cid)
    return dict(method=m[1],prefix=m[2],q=m[3],PF=m[4],r=m[5],K=m[6])


def verify_remote():
    receipt=readout('github_input_receipt.json');checked=[]
    for item in receipt['listings']:
        blob=inv.git('rev-parse',BASE+':'+item['path']).decode().strip()
        raw=inv.git('show',BASE+':'+item['path'])
        assert blob==item['git_blob_sha'] and len(raw)==item['bytes'],item['path']
        checked.append(dict(path=item['path'],git_blob_sha=blob,sha256=hashlib.sha256(raw).hexdigest(),mode='REMOTE_BLOB_LOCAL_FULL_BYTES'))
    for item in receipt['fetched']:
        raw=inv.git('show',item['commit']+':'+item['path'])
        assert raw.decode()==item['remote_content'],item['path']
        checked.append(dict(path=item['path'],commit=item['commit'],sha256=hashlib.sha256(raw).hexdigest(),mode='REMOTE_FULL_TEXT_EQUALS_FIXED_GIT'))
    return dict(status='PASS_GITHUB_EVIDENCE_MATCH',base_commit=BASE,checks=checked,
        compressed_bytes_decoded_from_same_local_git_blob=True,new_remote_writes=0)


def collect():
    inv.SOURCES.clear();remote=verify_remote()
    correspondence=source('DIRECT788',PREP+'/axis_correspondence.json')['rows']
    old_sources=source('PRIOR_SOURCE_INDEX',PLAN+'/source_inventory.json')
    old={s['id']:s for s in old_sources}
    bindings=source('DIRECT_BINDINGS',PREP+'/direct_system_bindings.json')
    models=source('MODEL_DEFINITIONS',PREP+'/model_registry.json')
    model_by_key={(m['system'],m['method'],m['axis']):m for m in models}
    historical_binding=source('PRIOR_BINDINGS',PLAN+'/system_bindings.json')
    excluded48=source('EXCLUDED48',PREP+'/excluded_48_axes.csv',fmt='csv')
    pending48={':'.join((r['system'],r['candidate_id'],r['axis'])) for r in excluded48}
    registrations=source('REGISTERED_SETS',B+'track_a_protocol_v2_draft/2026-10-19/candidate_registrations.json')['registries']
    h6rows=source('H6_ACCOUNTING',H6+'/candidate_accounting.json.gz')['rows']
    h8records=source('H8_ACCOUNTING',H8+'/accounting_rows.jsonl.gz',fmt='jsonl')
    h6comparisons=source('H6_COMPARISONS',H6+'/comparisons.json')['rows']
    h8comparisons=source('H8_COMPARISONS',H8+'/comparisons.json')['comparisons']
    h6terminal=source('H6_TERMINAL',H6+'/execution_terminal.json')
    h8authority=source('H8_AUTHORITY',H8+'/authorization_binding.json')
    h8registration=source('H8_REGISTRY','references/h8_phase_c_registration_v1.json')
    source('H6_SCOPE_SOURCE','scripts/track_a_v1_reaccount.py',fmt='bytes')
    source('H6_POLICY_SOURCE','src/trotterlib/track_a_v1_contract.py',fmt='bytes')
    source('H8_POLICY_SOURCE','src/trotterlib/h8_six_axis_reaccount.py',fmt='bytes')
    h8inputs=source('H8_INPUT_LINEAGE','references/h8_six_axis_reaccount_inputs_v1.json')['files']
    h8input_by_role={v['role']:v for v in h8inputs};h8precision={}
    def h8input(role):
        s=h8input_by_role[role];v=source('H8_POLICY_'+role,s['repository_path'],s['commit'])
        assert inv.SOURCES['H8_POLICY_'+role]['sha256']==s['sha256'];return v
    d0=h8input('original:D0');b0i=h8input('original:B0I_accounting');phase_a=h8input('original:Phase_A')
    def precision(row):
        k=(row['cell_id'],str(row['epsilon']),str(row['factor']))
        assert k not in h8precision;h8precision[k]=row['accounting']
    h8ids={r['id'] for r in h8registration['old']+h8registration['additional']}
    for row in d0['H8_EXPLORATION92']:
        if row['cell_id'] in h8ids:precision(row)
    for row in b0i:precision(row)
    for signal in phase_a['signals']:
        for epsilon,by_factor in signal['precision_rows'].items():
            for factor,accounting in by_factor.items():precision(dict(cell_id=signal['cell']['id'],epsilon=epsilon,factor=factor,accounting=accounting))
    for row in d0['H8_CONFIRMATION_UNION6']:
        if row['cell_id'] in h8registration['confirmation_ids']:precision(row)
    assert len(h8precision)==1900
    source('LATEST_D_PUBLIC_SUMMARY',B+'track_a_pr_cost_batch_results/2026-10-18/d_saved_comparison_summary.json')
    source('LATEST_D_PUBLIC_LIMIT','artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/README.md',fmt='bytes')
    h6cells={r['cell_id']:r['cell'] for r in h6rows if r['method']!='B2'}
    h6old=sorted(cid for cid,c in h6cells.items() if c['method']=='B1' or (c['formula']=='2nd' and c['prefix'] in (5,10,15)))
    h6fine=sorted(cid for cid,c in h6cells.items() if c['formula']=='2nd' or c['method']=='B1')
    h6all=sorted(h6cells)
    h8old=sorted(r['id'] for r in h8registration['old']);h8all=sorted(r['id'] for r in h8registration['old']+h8registration['additional'])
    assert (len(h6old),len(h6fine),len(h6all),len(h8old),len(h8all))==(32,96,136,32,184)
    for sys,ids,reg in [('H6_DF19',h6all,'H6_strengthened_S2_S4'),('H8_DF28',h8all,'H8_strengthened184')]:
        assert set(ids)=={r['id'] for r in registrations[reg]['rows']}
        assert bindings[sys]['binding']==historical_binding[sys]['binding']
    domains={'H6_DF19':{'old_grid_S2':h6old,'fine_grid_S2':h6fine,'fine_grid_S2_plus_truncated_S4':h6all},
             'H8_DF28':{'old32':h8old,'strengthened184':h8all}}
    direct={r['task_id']:r for r in correspondence if r['comparison_class']=='CONDITIONAL_COMPARISON'}
    assert Counter(r['system'] for r in direct.values())=={'H6_DF19':192,'H8_DF28':258}
    p7terminal=source('H6_P7_TERMINAL',B+'track_a_v1_p7_compile_run01/2026-10-16/execution_terminal.json')
    assert p7terminal['status']=='COMPLETE_MANDATORY_STOP' and p7terminal['exit_code']==0
    extra={}
    for label in ('H6_P7_COSINE','H6_P7_SINE','H6_D_LATEST','H8_D_B0_p15_q8_2nd','H8_D_B0I_p16_q4_4th'):
        s=old[label];data=source(label,s['path'],s['commit']);assert inv.SOURCES[label]['sha256']==s['sha256']
        system='H6_DF19' if label.startswith('H6') else 'H8_DF28'
        body=data.get('result',data);rows=body.get('rows',[body]);cid=body.get('cell_id',body.get('case',{}).get('id'))
        if label=='H6_D_LATEST':assert data['native_exit']['exit_code']==0 and data['native_exit']['same_parent_reaped']
        for r in rows:
            candidate=r.get('cell_id',cid);tid=':'.join((system,candidate,r['axis']))
            assert r['compiler_hash']==inv.COMPILER
            measured=r.get('measured_RZ',r.get('RZ',r.get('measured')))
            assert type(measured) is int and measured>0
            origin=r.get('SOURCE',data.get('SOURCE')) or (p7terminal['SOURCE'] if label.startswith('H6_P7_') else None)
            assert isinstance(origin,str) and len(origin)==40
            extra[tid]=dict(task_id=tid,direct_RZ_literal=str(measured),
                direct_SOURCE=origin,result_file_sha256=s['sha256'],evidence_source_id=label,
                model_prediction_literal=str(r.get('exact_prediction',r.get('prediction',r.get('model_check',{}).get('exact_prediction','')))),
                profile_condition='INHERITED_OLDER_RECEIPT_RUNTIME_EQUIVALENCE_NOT_CERTIFIED',
                public_evidence_reuse_only=True)
    assert len(extra)==8 and not set(extra)&set(direct)
    # Lossless metadata projection. Never call weighted arithmetic or minimum().
    normalized=[];units={};raw_by_row={}
    def add(system,row,source_id,locator,bias_mode,shots,cohort,baseline_SOURCE,account_override=None):
        if row['cell_id'].startswith('B2') and cohort=='EXPLORATION_N8':return
        cid=row['cell_id'];meta=parse_candidate(cid);axes=row.get('axis_costs',{})
        if isinstance(axes,list):axes={a['axis']:a for a in axes}
        kind='PR' if meta['method']=='B2' else 'PF'
        rid=':'.join((system,bias_mode,str(row['epsilon']),str(row['factor']),cid,cohort))
        assert rid not in raw_by_row;raw_by_row[rid]=row
        account=account_override if account_override is not None else row.get('accounting');eligible=row['eligible']
        if eligible is True:
            assert account is not None and account['cost_acquisition_eligible'] is True
            policy_shots={a:account['axes'][c]['shots'] for a,c in [('cosine','real'),('sine','imag')]}
            if kind=='PR' and not shots:shots=policy_shots
            assert shots==policy_shots,(rid,shots,policy_shots)
        normalized.append(dict(row_id=rid,system=system,candidate_id=cid,**meta,kind=kind,bias_mode=bias_mode,
            epsilon=str(row['epsilon']),allowance_factor=str(row['factor']),eligible=eligible,saved_shots=shots,
            saved_total_RZ_literal=row.get('G_exact'),saved_cost_status=row.get('cost_status'),
            saved_PR_SE_literal=row.get('saved_paired_SE',row.get('G_standard_error')),cohort=cohort,
            paired_samples=row.get('paired_samples'),source_id=source_id,source_locator=locator,
            source_file_sha256=inv.SOURCES[source_id]['sha256'],baseline_SOURCE=baseline_SOURCE,
            saved_bias_and_shot_policy=account,PR_cost_recalculation_allowed=False,
            unit_ids={a:':'.join((system,cid,a)) for a in axes} if kind=='PF' else {},
            saved_row_projection_digest=digest(row),projection_digest_is_not_raw_result_SHA=True))
        if kind=='PF':
            for axis,p in axes.items():
                tid=':'.join((system,cid,axis));literal=str(p['RZ'])
                if tid in units:assert units[tid]['baseline_RZ_literal']==literal and units[tid]['baseline_measured']==p['measured']
                else:units[tid]=dict(task_id=tid,system=system,candidate_id=cid,axis=axis,**meta,
                    baseline_RZ_literal=literal,baseline_measured=p['measured'],baseline_unit_evidence=p,
                    baseline_source_id=source_id,baseline_row_locator=locator,
                    old_profile_pending_48=tid in pending48,
                    direct_input_binding=bindings[system]['binding'],control='symmetric_directional',
                    baseline_input_scope='SAVED_NAMED_SERIES_BINDING; FULL_RUNTIME_TRANSFER_NOT_CERTIFIED',
                    compiler_hash=inv.COMPILER,direct_profile=bindings[system]['profile'],
                    baseline_file_sha256=inv.SOURCES[source_id]['sha256'])
    for i,row in enumerate(h6rows):
        if row['method']=='B2' and row.get('sample_scope')!='H6_CONFIRMATION_UNION8':continue
        accounting=row.get('accounting');shots={a:accounting['axes'][c]['shots'] for a,c in [('cosine','real'),('sine','imag')]} if accounting else {}
        add('H6_DF19',row,'H6_ACCOUNTING',f'/rows/{i}',row['bias_rule'],shots,
            'CONFIRMATION_UNION8_N32' if row['method']=='B2' else 'DETERMINISTIC',h6terminal['SOURCE'])
    for i,record in enumerate(h8records):
        shots={a:record['saved_shots'][c] for a,c in [('cosine','real'),('sine','imag')]} if record.get('saved_shots') else {}
        add('H8_DF28',record['updated'],'H8_ACCOUNTING',f'JSONL_RECORD:{i+1}/updated',record['bias_kind'],shots,
            'CONFIRMATION_UNION6_N32' if record['updated']['cell_id'].startswith('B2') else 'DETERMINISTIC',h8authority['SOURCE'],
            account_override=h8precision[(record['updated']['cell_id'],str(record['updated']['epsilon']),str(record['updated']['factor']))]
                if record['bias_kind']=='USUAL_SIGNED_SAVED' else None)
    assert len(normalized)==6680
    for system,scope in domains.items():
        largest=h6all if system=='H6_DF19' else h8all
        for cid in largest:
            for axis in policy.AXES:
                tid=':'.join((system,cid,axis))
                if tid not in units:
                    units[tid]=dict(task_id=tid,system=system,candidate_id=cid,axis=axis,**parse_candidate(cid),
                        baseline_RZ_literal=None,baseline_measured=None,baseline_unit_evidence=None,
                        old_profile_pending_48=tid in pending48,
                        reason='NO_UNIT_PRICE_IN_SAVED_ELIGIBLE_ROWS; NEVER_INFER_OR_ZERO_FILL',
                        direct_input_binding=bindings[system]['binding'],control='symmetric_directional',
                        compiler_hash=inv.COMPILER,direct_profile=bindings[system]['profile'])
    for tid,u in units.items():
        if u['baseline_measured'] is None:
            u.update(NEW450_ONLY='MISSING_BASELINE_PRICE',ALL_SAVED458='MISSING_BASELINE_PRICE',new_direct=None)
        elif u['baseline_measured']:
            u.update(NEW450_ONLY='RETAIN_OLD_MEASURED',ALL_SAVED458='RETAIN_OLD_MEASURED',new_direct=None)
        elif tid in direct:
            r=direct[tid];assert u['baseline_RZ_literal'] in r['saved_prediction_literals']
            u.update(NEW450_ONLY='REPLACE_NEW450',ALL_SAVED458='REPLACE_NEW450',new_direct=r)
        elif tid in extra:
            e=extra[tid]
            if e['model_prediction_literal']:assert e['model_prediction_literal']==u['baseline_RZ_literal']
            u.update(NEW450_ONLY='MISSING_DIRECT_FOR_NEW450_ONLY',ALL_SAVED458='REPLACE_EXISTING_EXTRA8',new_direct=e)
        else:u.update(NEW450_ONLY='MISSING_DIRECT',ALL_SAVED458='MISSING_DIRECT',new_direct=None)
        declared_model=model_by_key.get((u['system'],u['method'],u['axis']),{}) if u['baseline_measured'] is False else {}
        u['baseline_model_profile_sha256']=declared_model.get('profile_hash')
        u['baseline_model_file_sha256']=declared_model.get('source_sha256')
        u['replacement_RZ_literal']=u['baseline_RZ_literal'] if u['baseline_measured'] is True else (u['new_direct'] or {}).get('direct_RZ_literal')
        u['replacement_SOURCE']=(u['new_direct'] or {}).get('direct_SOURCE')
        u['replacement_result_sha256']=(u['new_direct'] or {}).get('direct_result_sha256',(u['new_direct'] or {}).get('result_file_sha256'))
        u['replacement_evidence_sha_kind']='WORKER_RESULT_SHA256' if 'direct_result_sha256' in (u['new_direct'] or {}) else 'SAVED_RESULT_FILE_SHA256' if u['new_direct'] else 'RETAINED_BASELINE_EVIDENCE'
        u['comparison_condition']='CONDITIONAL_PROFILE_RUNTIME; S4_COMPOSITION_HYPOTHESIS' if u['PF']=='4th' else 'CONDITIONAL_PROFILE_RUNTIME'
    expected_count={'H6_DF19':272,'H8_DF28':368}
    assert Counter(u['system'] for u in units.values())==expected_count
    new_pair_ids={sys:sorted({r['candidate_id'] for r in direct.values() if r['system']==sys}) for sys in domains}
    source_snapshots=dict(H6_DF19=dict(pf_candidates=136,PR_candidates=8,PR_cohort='CONFIRMATION_UNION8_N32',snapshot_SOURCE=h6terminal['SOURCE']),
                          H8_DF28=dict(pf_candidates=184,PR_candidates=6,PR_cohort='CONFIRMATION_UNION6_N32',snapshot_SOURCE=h8authority['SOURCE']))
    baseline_comparisons=[]
    for i,r in enumerate(h6comparisons):
        baseline_comparisons.append(dict(system='H6_DF19',bias_mode=r['bias_rule'],epsilon=str(r['epsilon']),allowance_factor=str(r['factor']),scope=r['scope'],
            saved_result=r,source_id='H6_COMPARISONS',source_locator=f'/rows/{i}'))
    for i,r in enumerate(h8comparisons):
        baseline_comparisons.append(dict(system='H8_DF28',bias_mode=r['bias_kind'],epsilon=str(r['epsilon']),allowance_factor=str(r['factor']),scope=r['scope'],
            saved_result=r,source_id='H8_COMPARISONS',source_locator=f'/comparisons/{i}'))
    endpoint_index=defaultdict(list)
    for r in normalized:endpoint_index[(r['system'],r['bias_mode'],r['epsilon'],r['allowance_factor'])].append(r)
    coverage=[]
    for comparison in baseline_comparisons:
        system=comparison['system'];endpoint=(system,comparison['bias_mode'],comparison['epsilon'],comparison['allowance_factor'])
        endpoint_rows=endpoint_index[endpoint];ids=domains[system][comparison['scope']]
        for strategy in policy.STRATEGIES:
            selected_ids=[cid for cid in ids if cid in new_pair_ids[system]] if strategy=='MATCHED_NEW450_SUBSET' else ids
            projected=[]
            for row in endpoint_rows:
                if row['kind']!='PF':continue
                needed=[units.get(':'.join((system,row['candidate_id'],a))) for a in policy.AXES]
                field='NEW450_ONLY' if strategy=='MATCHED_NEW450_SUBSET' else strategy
                complete=all(u is not None and not u[field].startswith('MISSING') for u in needed)
                projected.append(dict(row,price_coverage='COMPLETE' if complete else 'MISSING'))
            checks=policy.coverage(projected,selected_ids)
            pr=[r for r in endpoint_rows if r['kind']=='PR']
            unresolved_pr=[r['candidate_id'] for r in pr if r['eligible'] is None or (r['eligible'] is True and r['saved_total_RZ_literal'] is None)]
            coverage.append(dict(system=system,bias_mode=comparison['bias_mode'],epsilon=comparison['epsilon'],allowance_factor=comparison['allowance_factor'],
                original_scope=comparison['scope'],strategy=strategy,domain_kind='LIMITED_MATCHED_SUBSET' if strategy=='MATCHED_NEW450_SUBSET' else 'ORIGINAL_REGISTERED_SCOPE',
                candidate_ids=selected_ids,**checks,PR_candidate_count=len(pr),unresolved_PR_candidate_ids=unresolved_pr,
                is_new_numeric_result=False,old_profile_pending_axes_in_scope=sum(u['old_profile_pending_48'] for u in units.values() if u['system']==system and u['candidate_id'] in selected_ids),
                full_runtime_equivalence_verified=False))
    return dict(remote=remote,normalized=normalized,units=list(units.values()),domains=domains,new_pair_ids=new_pair_ids,
        baseline_comparisons=baseline_comparisons,coverage=coverage,source_snapshots=source_snapshots,
        extra8=list(extra.values()),bindings={k:bindings[k] for k in domains},sources=list(inv.SOURCES.values()))


def outputs(data):
    counts={strategy:dict(Counter(r['status'] for r in data['coverage'] if r['strategy']==strategy)) for strategy in policy.STRATEGIES}
    unit_counts={s:dict(Counter(u['ALL_SAVED458'] for u in data['units'] if u['system']==s)) for s in data['domains']}
    contract=dict(status='DRAFT_NOT_AUTHORIZED_NOT_SEALED',base_commit=BASE,
        scope='SAVED_PF_SHOT_COST_REPLACEMENT_AND_SELECTION; EXECUTION_REQUIRES_NEW_USER_APPROVAL',
        scientific_execution_authorized=False,active_grant=None,launch_command=None,
        baseline='Exact historical mixed model/measured policy, not a pure-model counterfactual',
        recommended_strategy='ALL_SAVED458',strategies=policy.STRATEGIES,
        preserve=dict(candidate_domains=data['domains'],PR_cost_and_SE='COPY_SAVED_CONFIRMATION_UNION8_N32_OR_UNION6_N32',
            shots='COPY real->cosine imag->sine; no recomputation',eligibility='COPY INCLUDING null',
            bias='COPY SAME SAVED BIAS MODE, EPSILON, ALLOWANCE FACTOR',P=0,T='0.8',control='symmetric_directional'),
        arithmetic_after_approval=dict(per_candidate='G_variant = N_cosine * RZ_cosine_variant + N_sine * RZ_sine_variant',
            baseline_replay='Exact rational equality to saved G per eligible PF; mismatch STOP',
            selection='exact rational G then lexical candidate_id; keep all exact ties',
            PR='Do not recompute/reselect PR; verify saved selection and copy cost/SE',
            ratios='saved PR total / best PF total in each variant; tie at exactly 1',
            rank_changes='candidate/q/PF/prefix changes, complete sorted order and pairwise inversions; ties distinct',
            PR_PF_reversal='three-state PR_LOWER/PF_LOWER/TIE, no tolerance',
            limited_subset='recompute BOTH baseline and replacement PF minima in identical subset; do not label global winner'),
        missing_policy='No zero fill; unresolved eligibility blocks full-domain minimum; missing eligible cost/axis/shots STOP affected condition',
        inherited_limits=['H6 profile mismatch','H8/model runtime binary equivalence unproven','S4 composition hypothesis',
            'initial572 global final gate missing (428 replacement axes)','48 pending-profile old axes retained only; no reacquisition',
            'Not pure model error isolation; finite PR mean and selection effect, no population winner guarantee'],
        source_and_input_freeze=dict(preparation_source='THIS_PACKET_COMMIT_TO_BE_PINNED_AT_EXECUTION_SEAL',input_sources=data['sources'],
            new_one_shot_grant_ledger_marker_STOP_required=True,old_grants_cannot_authorize_new_scope=True),
        authorization_needed=dict(historical_conditions=100,unique_system_bias_epsilon_factor_conditions=40,
            scopes='H6 3 domains / H8 2 domains; overlapping conditions not independent',
            ALL_SAVED458_counts=counts['ALL_SAVED458'],strategies_must_be_explicitly_selected=True),
        call_scope_after_explicit_approval=dict(shot_recalculation=0,new_model_prediction=0,fit=0,compile=0,sampling=0,
            PR_mean_SE_recalculation=0,PF_saved_cost_passes=1,PF_selection_and_ratio_comparison='only sealed ready conditions',
            excluded338_model_evaluation=0,H2_signal_repair=0,old48_reacquisition=0),
        resources='No arbitrary CPU/RAM/worker limits; index once, reuse per-endpoint totals across scopes; stdlib scalar pass; parallelize only independent useful work; shared environment unchanged',
        no_production_runner_available=True,detailed_scientific_review=False,final_evidence_freeze=False,mandatory_STOP=True)
    schema=dict(status='OUTPUT_DESIGN_ONLY_NO_REAL_RESULTS',
        common_keys=['system','bias_mode','epsilon','allowance_factor','original_scope','strategy','domain_kind',
            'candidate_set_sha256','saved_shot_policy_sha256','SOURCE','input_manifest_sha256','conditional_flags'],
        candidate_totals_columns=['candidate_id','PF','method','prefix','q','eligible','N_cosine','N_sine',
            'baseline_RZ_cosine_exact','baseline_RZ_sine_exact','replacement_RZ_cosine_exact','replacement_RZ_sine_exact',
            'saved_baseline_G_exact','replayed_baseline_G_exact','replacement_G_exact','baseline_replay_matches',
            'unit_provenance','unresolved_reason'],
        selection_comparison_columns=['best_PF_baseline','q_baseline','prefix_baseline','PF_baseline','G_baseline_exact',
            'best_PF_replacement','q_replacement','prefix_replacement','PF_replacement','G_replacement_exact',
            'exact_ties_baseline','exact_ties_replacement','best_candidate_changed','q_changed','PF_changed','prefix_changed',
            'saved_PR_candidate','saved_PR_cohort','saved_PR_total_exact','saved_PR_SE_literal',
            'PR_over_PF_baseline_exact','PR_over_PF_replacement_exact','PR_PF_relation_baseline','PR_PF_relation_replacement',
            'PR_PF_relation_changed','strict_pairwise_inversions','tie_relation_changes','coverage_status','not_global_if_subset'],
        rank_change_columns=['left_candidate','right_candidate','before_relation','after_relation','strict_inversion','tie_change'],
        incomplete_condition='status and unresolved IDs, numeric values null; never zero-fill',
        exact_values='Fraction strings; CSV and JSON same literals',
        reports='candidate and winner tables plus RZ/ratio plots only after separate explicit execution approval')
    templates={name+'_TEMPLATE.csv':','.join(schema['common_keys']+schema[field])+'\n' for name,field in
        [('candidate_totals','candidate_totals_columns'),('selection_comparison','selection_comparison_columns'),('rank_changes','rank_change_columns')]}
    return {'github_verification.json':data['remote'],'saved_condition_rows.json':dict(rows=data['normalized']),
        'saved_condition_rows.csv':csv_text([{k:v for k,v in r.items() if k!='saved_bias_and_shot_policy'} for r in data['normalized']]),
        'axis_price_lineage.json':dict(rows=data['units']),
        'axis_price_lineage.csv':csv_text([{k:v for k,v in u.items() if k not in ('baseline_unit_evidence','new_direct','direct_input_binding','direct_profile')} for u in data['units']]),
        'baseline_saved_comparisons.json':dict(rows=data['baseline_comparisons']),
        'candidate_domains.json':data['domains'],'extra8_existing_receipts.json':dict(rows=data['extra8']),
        'condition_coverage.json':dict(rows=data['coverage']),
        'condition_coverage.csv':csv_text(data['coverage']),
        'source_inventory.json':data['sources'],'input_bindings.json':data['bindings'],
        'execution_contract_DRAFT.json':contract,
        'output_schema.json':schema,**templates,
        'preparation_summary.json':dict(status='PREPARATION_ONLY_NO_NEW_COST_OR_SELECTION',snapshots=data['source_snapshots'],
            saved_rows=len(data['normalized']),axis_prices=len(data['units']),unit_replacement_counts=unit_counts,
            historical_scope_conditions=100,unique_system_bias_epsilon_factor_conditions=40,
            strategy_condition_counts=counts,extra_existing_axes=8,nominal_matching_new_axes=450,
            new_cost_accounting=0,new_shots=0,new_ranking=0,new_model_evaluation=0),
        'grant_INVALID.json':dict(status='INVALID_PREPARATION_ONLY',approved=False,scientific_execution_authorized=False,allowed_calls=[],next_stage_authorized=False),
        'STOP.json':dict(status='PREPARATION_COMPLETE_MANDATORY_STOP',scientific_execution_authorized=False,
            real_weighted_cost_evaluations=0,real_PF_selection=0,real_PR_PF_ratio_comparison=0,
            new_compile_sampling_model_prediction_fit=0,old_results_changed=False,new_remote_writes=0,
            detailed_scientific_review=False,final_evidence_freeze=False)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:policy.reject_production()
    data=collect();files=outputs(data);OUT.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():
        content=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
        if a.check:assert (OUT/name).read_text()==content,name
        else:(OUT/name).write_text(content)
    print(json.dumps(dict(status='PREPARATION_ONLY',files=len(files),summary=files['preparation_summary.json'])))


if __name__=='__main__':main()
