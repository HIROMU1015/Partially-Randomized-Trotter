"""DF10 S_STATS metadata and approval gate; no numerical design at import."""
import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import h4_dp_contract as c
import h4_tol_main_contract as main

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'H4_TOL10_S_STATS_20261019'
A_COMMIT = '85cb5b4d1808434165c421115ed09dab3b63dd00'
D_COMMIT = '406f70570323366da185355550a3900388717fbb'
SCIENCE_SOURCE = '84db3909639b34ec521004c3d8f7ef9b5e6ad851'
TARGETS = ((.05, 'B2_p3_q1_2nd_r4_K2', 'B0I_p5_q1_2nd'),
           (.0005, 'B2_p2_q8_2nd_r2_K4', 'B0_p7_q8_2nd'))
SOURCE_FILES = ('scripts/h4_tol_s_stats_contract.py', 'scripts/h4_tol_s_stats_numeric.py',
    'scripts/run_h4_tol_s_stats.py', 'scripts/prepare_h4_tol_s_stats.py', 'tests/test_h4_tol_s_stats.py')
FORBIDDEN = ('raw_trajectory_aggregation','bootstrap','sampling','compile','signal','shot_recalculation',
    'candidate_selection_or_reranking','old_FP_or_legacy_pooling','P_scientific_execution','new_seed_generation',
    'SCF_DF_state_reference_generation','law_reconstruction','rare_visibility_n_calculation')


def plan():
    draft = main.budgets()['S_STATS']
    return dict(scope=SCOPE, stage='S_STATS', binding=deepcopy(main.BINDING),
        targets=[dict(epsilon=e,cell_id=cid,PF_cell_id=pf) for e,cid,pf in TARGETS],
        fixed_A_commit=A_COMMIT,fixed_D_commit=D_COMMIT,upstream_scientific_SOURCE=SCIENCE_SOURCE,
        family_alpha=draft['family_alpha'],power=draft['power'],family_size=2,
        multiplicity='one-sided Bonferroni alpha/2 for the two fixed primary endpoints',
        SD_inflation_sensitivity=draft['SD_inflation_sensitivity'],
        relative_gap_sensitivity=draft['relative_gap_sensitivity'],
        reference_scenario=draft['proposed_reference_scenario'],minimum_n=2,n_cap=None,
        formula=draft['formula'],SD_input='saved_SE * sqrt(saved_n)',difference='eta * saved_D_PF_G',
        reference_direction='sign(saved_D_PF_G - saved_A_PR_mean_exact)',
        candidate_n='maximum over its registered endpoints; no endpoint double counting',
        required_cohort='EXPLORATION_N8',saved_FP_n32_used=False,legacy_used=False,
        calls=dict(numeric_design_pass=1,normal_scenarios=24,fixed_reference_sign_comparisons=2,
            sensitivity_settings=12,**{k:0 for k in FORBIDDEN}),
        rare_policy='Copy saved stratum counts only; probabilities/visibility/unseen costs remain unknown; no zero filling',
        actual_power_certified=False,population_mean_certified=False,selection_adjusted_inference=False,
        CI_or_p_values=False,candidate_reselection=False,optional_stopping=False,
        P_n=None,P_SOURCE=None,P_seeds=None,P_budget=None,
        P_finalization='PENDING_S_STATS_RESULT_AND_SEPARATE_USER_DECISION_WITH_RARE_TAIL_LIMITS',
        resources=dict(agent_imposed_CPU_RAM_worker_wall_caps=None,processes='one scalar pass; 24 scenarios',
            affinity_modified=False,shared_settings_modified=False,PSI_diagnostic_only=True),
        one_shot=True,automatic_retry=False,next_stage_authorized=False,mandatory_STOP=True)


def source_closure(root=ROOT):
    pending=list(SOURCE_FILES);out={}
    while pending:
        name=pending.pop()
        if name in out:continue
        path=Path(root)/name;c.require(path.is_file(),'STATS_SOURCE_FILE:'+name);out[name]=c.sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                candidate='scripts/'+module.split('.')[0]+'.py'
                if (Path(root)/candidate).is_file():pending.append(candidate)
    return out


def selected_summary(a_result,d_result):
    """Verbatim saved rows/counts only. Never derive SD, power, gap or n here."""
    c.require(a_result['binding']==d_result['binding']==main.BINDING,'DF10_ONLY')
    c.require(a_result['SOURCE']==d_result['SOURCE']==SCIENCE_SOURCE
        and a_result['registered_set_complete'] and d_result['registered_set_complete'],'ACCEPTED_COMPLETE_D_A')
    cells={x['id']:x for x in main.registrations()['A']};out=[]
    for e,cid,pf_id in TARGETS:
        matches=[r for r in a_result['rows'] if r['cell_id']==cid and r['epsilon']==e]
        c.require(len(matches)==1,'FIXED_PR_ENDPOINT');pr=matches[0]
        pf=next(w['best'] for w in d_result['winners'] if w['epsilon']==e)
        aw=next(w['best'] for w in a_result['winners'] if w['epsilon']==e)
        comp=next(w for w in a_result['comparisons'] if w['epsilon']==e)
        c.require(pr==aw==comp['PR'] and pf==comp['PF'] and pf['cell_id']==pf_id,'USER_FIXED_SAVED_WINNERS')
        c.require(pr['cohort']=='EXPLORATION_N8' and pr['n']==8 and pr['eligible'],'NO_FP_OR_LEGACY_COHORT')
        c.require(type(pr['G_RZ_SE']) in (int,float) and math.isfinite(pr['G_RZ_SE']) and pr['G_RZ_SE']>0,
            'COMPLETE_SAVED_PAIRED_SE')
        rare=a_result['new_cost_strata'][cid]
        c.require(set(rare)=={'S0_ONLY','S2_SINGLE','S2_MULTI','S4_PRESENT'}
            and all(type(v) is int and v>=0 for v in rare.values()) and sum(rare.values())==8,'SAVED_RARE_COUNTS_SCHEMA')
        out.append(dict(epsilon=e,cell=deepcopy(cells[cid]),PR=deepcopy(pr),PF=deepcopy(pf),
            saved_point_direction=comp['point_direction'],saved_rare_counts=deepcopy(rare),
            law_probabilities=None,rare_visibility_n=None,unseen_stratum_cost=None,
            rare_counts_are_not_law_probabilities=True,population_mean_certified=False))
    return dict(binding=deepcopy(main.BINDING),A_commit=A_COMMIT,D_commit=D_COMMIT,
        endpoints=out,selected_by_user=True,old_FP_or_legacy_pooled=False)


def invalid_grant(spec):
    return dict(scope=SCOPE,stage='S_STATS',SOURCE=spec.get('SOURCE'),spec_digest=c.digest(spec),
        approved_by_user=False,runtime_authorization=False,one_shot=True,automatic_retry=False,
        next_stage_authorized=False,approval_receipt=None,approval_receipt_sha256=None,launch_command=None)


def verify_bytes(spec):
    c.require(spec['plan']==plan(),'FIXED_STATS_PLAN')
    c.require(spec['source_hashes']==source_closure(),'COMPLETE_STATS_SOURCE_CLOSURE')
    for name,h in spec['source_hashes'].items():
        blob=subprocess.check_output(['git','show',spec['SOURCE']+':'+name],cwd=ROOT)
        c.require(c.sha(ROOT/name)==h==hashlib.sha256(blob).hexdigest(),'STATS_SOURCE_CHANGED')
    for row in spec['inputs']:
        c.require(c.sha(row['path'])==row['sha256'],'STATS_INPUT_CHANGED:'+row['role'])
        if row.get('commit'):
            raw=subprocess.check_output(['git','show',row['commit']+':'+row['repo_path']],cwd=ROOT)
            c.require(hashlib.sha256(raw).hexdigest()==row['sha256'],'FIXED_UPSTREAM_INPUT')
    for path,h in spec['runtime_hashes'].items():c.require(c.sha(path)==h,'STATS_RUNTIME_CHANGED')
    c.require(c.sha(sys.executable)==spec['python_sha256'],'STATS_PYTHON_CHANGED')
    roles={row['role']:row for row in spec['inputs']}
    bound={row['path'] for row in spec['inputs']}
    c.require(spec['summary'] in bound and spec['preparation_receipt'] in bound,'ACTUAL_SUMMARY_BOUND')
    a_result,d_result=c.read(roles['A_result']['path']),c.read(roles['D_result']['path'])
    expected=selected_summary(a_result,d_result)
    c.require(c.read(spec['summary'])==expected,'VERBATIM_SELECTED_SUMMARY')
    for stage,result in [('A',a_result),('D',d_result)]:
        audit=c.read(roles[stage+'_audit']['path']);native=c.read(roles[stage+'_native_exit']['path'])
        h=roles[stage+'_result']['sha256']
        c.require(audit['status']=='COMPLETE_MANDATORY_STOP' and audit['result_sha256']==h
            and audit['native_exit_code']==0 and audit['all_workers_reaped']
            and native['exit_code']==0 and native['same_parent_reaped'] and native['result_sha256']==h,
            'ACCEPTED_NATIVE_D_A_RESULT')
    return expected


def authorize(spec,grant):
    # No scientific input, lock, output, or numeric module before explicit approval.
    c.require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'SEPARATE_EXPLICIT_S_STATS_APPROVAL_REQUIRED')
    c.require(spec.get('stage')==grant.get('stage')=='S_STATS' and grant.get('scope')==SCOPE,'DF10_S_STATS_ONLY')
    c.require(spec.get('status')=='SEALED_FOR_EXECUTION' and spec.get('pending')==[],'PENDING_EXECUTION_APPROVAL')
    c.require(grant.get('one_shot') is True and grant.get('automatic_retry') is False
        and grant.get('next_stage_authorized') is False,'ONE_SHOT_MANDATORY_STOP')
    c.require(spec['SOURCE']==grant['SOURCE'] and c.digest(spec)==grant['spec_digest'],'STATS_CONTRACT_BINDING')
    c.require(c.sha(grant['approval_receipt'])==grant['approval_receipt_sha256'],'STATS_APPROVAL_RECEIPT')
    c.require(not Path(spec['lock']).exists() and not Path(spec['output']).exists(),'USED_STATS_NAMESPACE')
    c.require(spec['fresh_resources']['admitted'] is True and c.sha(spec['fresh_resources']['receipt'])
        ==spec['fresh_resources']['sha256'],'FRESH_STATS_RESOURCE_RECEIPT')
    return verify_bytes(spec)
