"""Metadata structure checks only. No scientific imports or execution entry point."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_h46_m5_minimal_comparison_plan/2026-10-19'


def validate(plan,domain):
    def require(ok,message):
        if not ok:raise ValueError(message)
    require(plan['status']=='DRAFT_SCOPE_PROPOSAL_NOT_LAUNCH_READY' and
        not plan['scientific_execution_authorized'] and plan['scientific_SOURCE'] is None and plan['valid_grants']==0,'PLAN_ONLY')
    require(plan['adoption_AD03']=='DEFERRED' and not plan['acceptance']['m5_main_adoption_authorized'],'AD03_DEFERRED')
    rows=domain['rows'];primary=[r for r in rows if r['role']=='PRIMARY'];optional=[r for r in rows if r['role']!='PRIMARY']
    require(len(rows)==108 and len(primary)==88 and len(optional)==20,'SPARSE_DOMAIN_COUNTS')
    require(plan['counts']['primary_signal_conditions']==88 and plan['counts']['deterministic_signal_conditions']==64 and
        plan['counts']['PR_signal_conditions']==24 and plan['counts']['optional_signal_conditions']==20,'PLAN_DOMAIN_COUNTS_AGREE')
    require(len({r['id'] for r in rows})==len(rows),'UNIQUE_CANDIDATE_IDENTITIES')
    keys=('system','input_series','kind','method','formula','prefix','q','K','r_policy','r_value','role')
    require(len({tuple(r[k] for k in keys) for r in rows})==len(rows),'UNIQUE_PHYSICAL_CONDITIONS')
    for system,rank,anchor,cuts in [('H4',10,3,[3,8,9]),('H6',19,5,[5,7,11,15])]:
        selected=[r for r in primary if r['system']==system]
        require(all(r['input_series']==system+'_TOL_ONLY_DF'+str(rank) and 1<=r['prefix']<=rank for r in rows if r['system']==system),'INPUT_SERIES_PREFIX')
        det=[r for r in selected if r['kind']=='DET_SIGNAL'];pr=[r for r in selected if r['kind']=='PR_SIGNAL']
        expected={(method,p,q) for method in ('B0','B0I') for p in cuts for q in (1,2,4,8)}|{('FULL',rank,q) for q in (1,2,4,8)}
        require({(r['method'],r['prefix'],r['q']) for r in det}==expected and all(r['formula']=='4th(m5_best)' and r['K'] is None and r['r_policy'] is None for r in det),'DETERMINISTIC_DOMAIN')
        require({(r['formula'],r['q']) for r in pr}=={(f,q) for f in ('2nd','4th','4th(m5_best)') for q in (1,2,4,8)},'COMMON_Q_COVERAGE')
        require(all(r['prefix']==anchor and r['K']==4 and r['r_policy']=='ABS_TIME_PER_OUTER' and r['r_value']==32 for r in pr),'MATCHED_PRIMARY_RTE_BUDGET')
        require(len([r for r in optional if r['system']==system])==10 and all(r['formula']=='4th(m5_best)' for r in optional),'OPTIONAL_SCOPE')
    require(plan['sample_design']['exploration_n_E'] is None and plan['sample_design']['confirmation_n_C'] is None and plan['sample_design']['seed_registry'] is None,'SAMPLE_DESIGN_NOT_ADOPTED')
    require(not plan['sample_design']['pool_pilot_with_exploration'] and not plan['sample_design']['pool_exploration_with_confirmation'],'COHORTS_NOT_POOLED')
    require(plan['comparison']['allowance_value'] is None and plan['comparison']['allowance_multiplier']==1,'ALLOWANCE_NOT_REPLACED_BY_CT01')
    require(plan['acceptance']['compile_vector_strictly_below']==1e-7 and plan['acceptance']['other_native_checks_unchanged'],'CT01_SCOPE')
    require(not plan['acceptance']['unknown_cost_zero_fill'] and plan['acceptance']['global_domain_best_if_missing'] is None,'MISSING_COST_NOT_ZERO')
    require(plan['counts']['maximum_missing_direct_DET_axes_if_all_eligible']==2*64-12,'COMPILE_COUNT_NOT_SIGNAL_COUNT')
    require(plan['resources']['worker_cap'] is None and plan['resources']['RAM_cap'] is None and not plan['resources']['host_PSI_kill'] and not plan['resources']['unrelated_worktrees_frozen'],'RESOURCE_POLICY')
    phases=plan['phases'];ids={p['id'] for p in phases};require(len(ids)==len(phases)==14,'UNIQUE_PHASES')
    for p in phases:
        require(not p['execution_authorized'] and not p['auto_transition'] and p['STOP'],'PHASE_REQUIRES_APPROVAL_AND_STOP')
        require(set(p['depends'])<=ids and all(d[:2]==p['id'][:2] for d in p['depends']),'NO_CROSS_SYSTEM_BLOCKING')
    completed=set()
    while len(completed)<len(phases):
        ready={p['id'] for p in phases if p['id'] not in completed and set(p['depends'])<=completed}
        require(ready,'ACYCLIC_PHASE_DEPENDENCIES');completed.update(ready)
    return dict(primary=len(primary),optional=len(optional),phases=len(phases))


def main():
    read=lambda name:json.loads((BUNDLE/name).read_text())
    plan=read('plan_DRAFT.json');domain=read('candidate_domain_DRAFT.json');counts=validate(plan,domain)
    catalog=read('evidence_catalog.json')['files']
    for r in catalog:
        blob=subprocess.check_output(['git','show',r['commit']+':'+r['path']],cwd=ROOT)
        assert len(blob)==r['bytes'] and hashlib.sha256(blob).hexdigest()==r['sha256'],r['role']
    with (BUNDLE/'deterministic_reuse_candidates.csv').open() as f:reuse=list(csv.DictReader(f))
    assert len(reuse)==12 and len({(r['candidate_id'],r['axis']) for r in reuse})==12
    assert all(r['reuse_for_future_shots']=='NOT_YET_ADOPTED_ALLOWANCE_REQUIRED' for r in reuse)
    with (ROOT/'artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/axis_status.csv').open() as f:old_axes=list(csv.DictReader(f))
    with (ROOT/'artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/axis_cost_receipts.csv').open() as f:new_axes=list(csv.DictReader(f))
    for row in reuse:
        matches=[r for r in old_axes if r['task']==row['original_task'] and r['axis']==row['axis']]
        assert len(matches)==1
        source=next(r for r in new_axes if r['axis']==row['axis']) if row['reuse_status']=='CT01_SEPARATE_RECEIPT' else matches[0]
        assert row['compiled_RZ']==source.get('compiled_RZ',source.get('rz_count'))
        if row['reuse_status']!='CT01_SEPARATE_RECEIPT':assert matches[0]['axis_status']=='PASS'
    with (BUNDLE/'candidate_domain_DRAFT.csv').open() as f:csvrows=list(csv.DictReader(f))
    assert [r['id'] for r in csvrows]==[r['id'] for r in domain['rows']]
    mutations=[('grant',lambda p,d:p.update(scientific_execution_authorized=True)),
        ('pooling',lambda p,d:p['sample_design'].update(pool_pilot_with_exploration=True)),
        ('CT01',lambda p,d:p['acceptance'].update(compile_vector_strictly_below=1e-9)),
        ('budget',lambda p,d:next(r for r in d['rows'] if r['kind']=='PR_SIGNAL' and r['role']=='PRIMARY').update(r_value=2)),
        ('q_coverage',lambda p,d:d['rows'][0].update(q=16)),
        ('AD03',lambda p,d:p.update(adoption_AD03='ADOPTED')),
        ('auto_transition',lambda p,d:p['phases'][0].update(auto_transition=True)),
        ('global_freeze',lambda p,d:p['resources'].update(unrelated_worktrees_frozen=True))]
    rejected=[]
    for name,mutate in mutations:
        p,d=copy.deepcopy(plan),copy.deepcopy(domain);mutate(p,d)
        try:validate(p,d)
        except ValueError as exc:rejected.append(dict(case=name,rejected=True,reason=str(exc)))
        else:raise AssertionError('INVALID_PLAN_ACCEPTED:'+name)
    print(json.dumps(dict(status='STRUCTURE_PASS_NO_SCIENCE',counts=counts,evidence_files_verified=len(catalog),reuse_axes_verified=len(reuse),artificial_negative_cases=rejected,
        author_checks_not_independent_review=True,new_scientific_calculation=0,new_numerical_reanalysis=0),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
