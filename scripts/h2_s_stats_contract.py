"""H2-only saved-summary S_STATS contracts. Importing never designs a real n."""
import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h2_panels_contract as main
import h4_tol_s_stats_contract as shared

c = SimpleNamespace(require=main.require, read=main.read, save=main.save, sha=main.sha,
    digest=main.digest, encoded=main.stats.encoded, normal_n_formula=main.stats.normal_n_formula)
ROOT, PRIVATE = main.ROOT, main.PRIVATE
STORE = PRIVATE/'S_STATS_preparation01'
ACTIVATION = PRIVATE/'S_STATS_activation01'
PUBLIC = main.PUBLIC/'S_STATS_preparation01'
COMPLETED = main.PUBLIC/'S_STATS_completed_run01'
SCOPE = 'H2_S_STATS_V1_20261019'
A_COMMIT = '1d2f67ec285956ef4090ea26520d64b11e925f9c'
D_COMMIT = '7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb'
A_SOURCE = 'de72e5d4e48177734c0ddd79dffac14222410658'
D_SOURCE = 'cf99e4452df96291d7e9d37d76d2e91be5cbc413'
SHARED_KERNEL_SOURCE = '8752489bde3903d80d306c4500535c79ac65ef31'
TARGETS = ((.05, 'B2_p1_q1_2nd_r2_K4', 'B1_p3_q1_2nd'),
           (.0005, 'B2_p1_q4_2nd_r2_K4', 'B1_p3_q4_2nd'))
MESSAGE = '数値実行まで進んでよい'
FILES = ('scripts/h2_s_stats_contract.py', 'scripts/h2_s_stats_numeric.py',
    'scripts/prepare_h2_s_stats.py', 'scripts/run_h2_s_stats.py',
    'scripts/audit_h2_s_stats_completed.py', 'tests/test_h2_s_stats.py')


def plan():
    inherited = shared.plan()
    return dict(scope=SCOPE, stage='S_STATS', binding=deepcopy(main.BINDING),
        targets=[dict(epsilon=e,cell_id=cid,PF_cell_id=pf) for e,cid,pf in TARGETS],
        fixed_A_commit=A_COMMIT, fixed_D_commit=D_COMMIT,
        upstream_A_SOURCE=A_SOURCE, upstream_D_SOURCE=D_SOURCE,
        family_alpha=inherited['family_alpha'], power=inherited['power'], family_size=2,
        multiplicity=inherited['multiplicity'],
        SD_inflation_sensitivity=inherited['SD_inflation_sensitivity'],
        relative_gap_sensitivity=inherited['relative_gap_sensitivity'],
        reference_scenario=inherited['reference_scenario'], minimum_n=2, n_cap=None,
        formula=inherited['formula'], SD_input='saved_SE * sqrt(saved_n)',
        difference='eta * saved_D_PF_G', reference_direction='sign(saved_D_PF_G - saved_A_PR_mean_exact)',
        candidate_n=inherited['candidate_n'], required_cohort='EXPLORATION_N8',
        FP_diagnostic_costs_used=False, legacy_used=False, candidate_reselection=False,
        shared_numeric_kernel='Unmodified h4_tol_s_stats_numeric.analyze code with an H2 contract namespace',
        shared_numeric_SOURCE=SHARED_KERNEL_SOURCE,
        saved_ratio_adapter='Inside approved numeric pass only, saved PR/PF ratio supplies legacy direction consistency field',
        calls=deepcopy(inherited['calls']), rare_policy=inherited['rare_policy'],
        actual_power_certified=False, population_mean_certified=False, selection_adjusted_inference=False,
        CI_or_p_values=False, optional_stopping=False,
        P_n=None, P_SOURCE=None, P_seeds=None, P_budget=None,
        P_finalization='PENDING_S_STATS_RESULT_AND_USER_ADOPTION_WITH_RARE_TAIL_LIMITS',
        resources=deepcopy(inherited['resources']), one_shot=True,
        automatic_retry=False, next_stage_authorized=False, mandatory_STOP=True)


def source_closure():
    pending=list(FILES); rows={}
    while pending:
        rel=pending.pop()
        if rel in rows: continue
        path=ROOT/rel; rows[rel]=c.sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                candidate='scripts/'+module.split('.')[0]+'.py'
                if (ROOT/candidate).is_file(): pending.append(candidate)
    return rows


def selected_summary(a, d, audit):
    """Copy saved endpoint rows and saved audit counts; no SD/gap/n calculation."""
    c.require(a['binding']==d['binding']==main.BINDING and a['stage']=='A' and d['stage']=='D'
        and a['status']==d['status']=='COMPLETE_MANDATORY_STOP'
        and a['SOURCE']==A_SOURCE and d['SOURCE']==D_SOURCE, 'ACCEPTED_H2_D_A_ONLY')
    cells={x['id']:x for x in main.registrations()['A']}; endpoints=[]
    for epsilon,cid,pf_id in TARGETS:
        pr=a['accounting']['winners'][str(epsilon)]; pf=d['accounting']['winners'][str(epsilon)]
        comp=next(r for r in a['accounting']['comparisons'] if r['epsilon']==epsilon)
        c.require(pr==comp['PR'] and pf==comp['PF'] and pr in a['accounting']['rows']
            and pf in d['accounting']['rows'] and pr['cell_id']==cid and pf['cell_id']==pf_id,
            'FIXED_SAVED_H2_WINNERS')
        c.require(pr['cohort']=='EXPLORATION_N8' and pr['n']==8 and pr['eligible']
            and type(pr['G_RZ_SE']) in (int,float) and math.isfinite(pr['G_RZ_SE']) and pr['G_RZ_SE']>0,
            'SAVED_H2_N8_POSITIVE_SE')
        c.require(comp['status']=='FINITE_SAMPLE_POINT_COMPARISON' and
            math.isfinite(comp['PR_to_PF_ratio']), 'SAVED_POINT_RATIO')
        rare=audit['winner_stratum_counts'][cid]
        c.require(set(rare)<={'S0_ONLY','S2_SINGLE','S2_MULTI','S4_PRESENT'}
            and all(type(v) is int and v>=0 for v in rare.values()) and sum(rare.values())==8,
            'SAVED_AUDIT_RARE_COUNTS')
        endpoints.append(dict(epsilon=epsilon,cell=deepcopy(cells[cid]),PR=deepcopy(pr),PF=deepcopy(pf),
            saved_PR_to_PF_ratio=comp['PR_to_PF_ratio'],saved_rare_counts=deepcopy(rare),
            law_probabilities=None,rare_visibility_n=None,unseen_stratum_cost=None,
            rare_counts_are_not_law_probabilities=True,population_mean_certified=False))
    return dict(binding=deepcopy(main.BINDING), A_commit=A_COMMIT, D_commit=D_COMMIT,
        endpoints=endpoints, selection_rule='Saved A winners at the original primary eps=.05/.0005; no new ranking',
        accepted_by_user_for_S_STATS=True, old_FP_or_legacy_pooled=False)


def invalid_grant(spec):
    return dict(scope=SCOPE,stage='S_STATS',SOURCE=spec.get('SOURCE'),spec_digest=c.digest(spec),
        approved_by_user=False,runtime_authorization=False,one_shot=True,automatic_retry=False,
        next_stage_authorized=False)


def verify_bytes(spec):
    c.require(spec['plan']==plan(), 'FIXED_H2_S_STATS_PLAN')
    c.require(spec['source_hashes']==source_closure(), 'COMPLETE_H2_STATS_SOURCE_CLOSURE')
    for rel,h in spec['source_hashes'].items():
        blob=subprocess.check_output(['git','--no-optional-locks','show',spec['SOURCE']+':'+rel],cwd=ROOT)
        c.require(c.sha(ROOT/rel)==hashlib.sha256(blob).hexdigest()==h, 'FIXED_STATS_SOURCE:'+rel)
    roles={r['role']:r for r in spec['inputs']}
    c.require(len(roles)==len(spec['inputs']), 'UNIQUE_STATS_INPUT_ROLES')
    for row in spec['inputs']:
        path=main.owned(row['path'])
        c.require(c.sha(path)==row['sha256'], 'STATS_INPUT_CHANGED:'+row['role'])
        if row.get('commit'):
            blob=subprocess.check_output(['git','--no-optional-locks','show',row['commit']+':'+row['repo_path']],cwd=ROOT)
            c.require(hashlib.sha256(blob).hexdigest()==row['sha256'], 'UPSTREAM_GIT_INPUT_IDENTITY')
    for path,h in spec['runtime_hashes'].items(): c.require(c.sha(path)==h, 'STATS_RUNTIME_CHANGED')
    c.require(c.sha(spec['python'])==spec['python_sha256'], 'PINNED_STATS_PYTHON_BYTES')
    c.require(spec['summary']==roles['selected_summaries']['path']
        and spec['preparation_receipt']==roles['preparation_receipt']['path']
        and spec['prior_manifest']==roles['prior_preservation']['path'], 'ACTUALLY_USED_H2_INPUTS')
    stages={}
    for stage,source in [('A',A_SOURCE),('D',D_SOURCE)]:
        result=c.read(roles[stage+'_result']['path']); audit=c.read(roles[stage+'_audit']['path'])
        native=c.read(roles[stage+'_native_exit']['path']); binding=c.read(roles[stage+'_execution_binding']['path'])
        h=roles[stage+'_result']['sha256']
        c.require(result['SOURCE']==audit['SOURCE']==binding['SOURCE']==source
            and audit['status']=='COMPLETE_MANDATORY_STOP' and audit['result_sha256']==h
            and audit['native_exit_code']==native['exit_code']==0
            and audit['same_parent_reaped'] and audit['wait4_complete']
            and native['same_parent_reaped'] and native['wait4_complete'], 'SAVED_NATIVE_D_A_COMPLETION')
        for rel,sha in binding['source_hashes'].items():
            blob=subprocess.check_output(['git','--no-optional-locks','show',source+':'+rel],cwd=ROOT)
            c.require(hashlib.sha256(blob).hexdigest()==sha==c.sha(ROOT/rel), 'UPSTREAM_SCIENCE_SOURCE')
        stages[stage]=result
    expected=selected_summary(stages['A'],stages['D'],c.read(roles['A_audit']['path']))
    c.require(c.read(spec['summary'])==expected, 'VERBATIM_H2_SELECTED_SUMMARY')
    for row in c.read(spec['prior_manifest']):
        c.require(c.sha(row['path'])==row['sha256'], 'PRIOR_SOURCE_RESULT_GRANT_LEDGER_STOP_CHANGED')
    attempt=main.owned(spec['attempt'])
    c.require(attempt==PRIVATE/'attempts/S_STATS_run01' and
        main.owned(spec['output'])==attempt/'output' and main.owned(spec['lock'])==attempt/'ONE_SHOT.json',
        'H2_OWNED_STATS_ATTEMPT')
    return expected


def authorize(spec, grant):
    c.require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'EXPLICIT_H2_S_STATS_APPROVAL_REQUIRED')
    c.require(grant.get('scope')==SCOPE and spec.get('stage')==grant.get('stage')=='S_STATS', 'H2_S_STATS_ONLY')
    c.require(spec.get('status')=='SEALED_FOR_EXECUTION' and spec.get('pending')==[], 'SEALED_STATS_EXECUTION_REQUIRED')
    c.require(grant.get('one_shot') is True and grant.get('automatic_retry') is False
        and grant.get('next_stage_authorized') is False, 'ONE_SHOT_MANDATORY_STOP')
    c.require(spec['SOURCE']==grant['SOURCE'] and c.digest(spec)==grant['spec_digest'], 'STATS_GRANT_DIGEST')
    receipt=main.owned(grant['approval_receipt'])
    c.require(c.sha(receipt)==grant['approval_receipt_sha256'], 'USER_APPROVAL_RECEIPT_BYTES')
    accepted=c.read(receipt)
    c.require(accepted['role']=='user' and accepted['stage']=='S_STATS' and accepted['SOURCE']==spec['SOURCE']
        and accepted['plan_digest']==c.digest(plan()) and accepted['verbatim_user_message'], 'SCOPED_USER_APPROVAL')
    pending=main.owned(spec['pending_spec'])
    c.require(c.sha(pending)==spec['pending_spec_sha256'], 'PENDING_STATS_SEAL_PRESERVED')
    original=c.read(pending); allowed={'status','pending','pending_spec','pending_spec_sha256','fresh_resources'}
    c.require({k:v for k,v in spec.items() if k not in allowed}==
        {k:v for k,v in original.items() if k not in allowed}, 'ACTIVATION_FIXED_STATS_FIELDS')
    resources=spec['fresh_resources']
    c.require(resources['admitted'] and c.sha(main.owned(resources['receipt']))==resources['sha256'], 'FRESH_STATS_RESOURCES')
    c.require(not Path(spec['lock']).exists() and not Path(spec['output']).exists(), 'ONE_SHOT_CONSUMED_NO_RETRY')
    c.require(Path(sys.executable).absolute()==Path(spec['python']).absolute(), 'PINNED_STATS_INTERPRETER')
    return verify_bytes(spec)
