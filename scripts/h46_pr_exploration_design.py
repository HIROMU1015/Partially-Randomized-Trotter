"""Main-exploration proposals from frozen pilot summaries, not new statistics.

No RNG, molecular data, new MCSE estimates, shot calculation or ranking here.
All counts are proposals until the user adopts a named option and execution.
"""
import csv
import json
from pathlib import Path
import subprocess
import h46_pr_design_pilot as pilot

ROOT=Path(__file__).resolve().parents[1]
BASE='bc09510b488bcb1e69fb7579b26dcae1459a8810'
EVIDENCE='artifacts/resource_applicability/track_a_h246_higher_pf_pilot_evaluation/2026-10-20/'
PROPOSAL='references/h46_pr_exploration_proposal_v1.json'
COHORT='EXPLORATION'
STAGES=('H4_PR_EXPLORATION','H6_PR_EXPLORATION')
require=pilot.require
digest=pilot.digest
groups=pilot.groups
registry=pilot.registry
event_stratum=pilot.event_stratum
call_scope=pilot.call_scope
COMPILER,COMPILER_HASH,NATIVE_POLICY=pilot.COMPILER,pilot.COMPILER_HASH,pilot.NATIVE_POLICY

def tasks_for(stage,counts):
    require(stage in STAGES,'EXPLORATION_STAGE_ONLY')
    rows=[r for r in groups() if r['system']==stage[:2]]
    require(set(counts)=={r['id'] for r in rows},'EXACT_12_EXPLORATION_COUNTS')
    tasks=[]
    for r in rows:
        n=counts[r['id']]
        require(type(n) is int and n>=2,'EXPLORATION_N_UNADOPTED_OR_INVALID')
        tasks.extend(dict(id='E1__'+r['id']+'__'+str(i),candidate=r,sample=i,
                          axes=['cosine','sine'],stage=stage) for i in range(n))
    return tasks

def frozen_bytes(rel):
    raw=(ROOT/rel).read_bytes()
    require(raw==subprocess.check_output(['git','show',BASE+':'+rel],cwd=ROOT),
            'FROZEN_PILOT_BYTES:'+rel)
    return raw

def allocate(rows,nominal,allocation,floor):
    """Only integer design arithmetic. Unknown counts remain unknown."""
    require(allocation in ('GROUP','COMMON_4Q') and floor in (2,33),'PROPOSAL_CHOICES')
    values={r['id']:None if nominal[r['id']] is None else max(floor,nominal[r['id']]) for r in rows}
    if allocation=='COMMON_4Q':
        for system in ('H4','H6'):
            for formula in pilot.previous.FORMULAS:
                keys=[r['id'] for r in rows if r['system']==system and r['formula']==formula]
                common=None if any(values[k] is None for k in keys) else max(values[k] for k in keys)
                values.update({k:common for k in keys})
    return values

def option_id(rho,factor,allocation,floor):
    return f'MCSE_{int(rho*100)}PCT_F{factor}_{allocation}_MIN{floor}'

RECOMMENDED=option_id(.02,4,'GROUP',33)

def proposal():
    import hashlib
    rows=groups();inputs=[];statistics={};sensitivity={}
    names=['H4_pilot_statistics.json','H6_pilot_statistics.json','group_statistics_24.csv',
           'n_E_sensitivity_216.csv','family_n_E_sensitivity_54.csv','rare_strata_96.csv','provenance.json']
    raw={}
    for name in names:
        rel=EVIDENCE+name;raw[name]=frozen_bytes(rel)
        inputs.append(dict(path=rel,sha256=hashlib.sha256(raw[name]).hexdigest(),bytes=len(raw[name]),commit=BASE))
    for system in ('H4','H6'):
        saved=json.loads(raw[system+'_pilot_statistics.json'])
        require(saved['n_E_adopted'] is False and saved['ranking'] is False,'PILOT_NOT_MAIN_RANKING')
        for g in saved['groups']:
            candidate=g['candidate'];s=g['statistics'];key=candidate['id']
            require(candidate==next(r for r in rows if r['id']==key),'CANDIDATE_JOIN')
            require(s['n']==33 and s['strata']['S4']['observed_count']==0,'SAVED_33_AND_UNOBSERVED_S4')
            require(not s['population_rank_guaranteed'] and not s['automatic_n_E_adoption'],'PILOT_LIMITS')
            statistics[key]=s
            sensitivity[key]={(t['relative_MCSE_target'],t['variance_inflation']):t['n_E_nominal_only'] for t in s['n_E_sensitivity']}
    # Exact field transfer checks; covariance is retained, never assumed zero.
    for row in csv.DictReader(raw['group_statistics_24.csv'].decode().splitlines()):
        s=statistics[row['candidate_id']]
        require(float(row['cov_cs'])==s['covariance'][0][1] and int(row['n'])==s['n'],'SAVED_COVARIANCE_TRANSFER')
    transferred=list(csv.DictReader(raw['n_E_sensitivity_216.csv'].decode().splitlines()))
    require(len(transferred)==216,'SAVED_216_SCENARIOS')
    for row in transferred:
        candidate=next(r for r in rows if r['system']==row['system'] and r['formula']==row['formula'] and r['q']==int(row['q']))
        expected=sensitivity[candidate['id']][float(row['relative_MCSE_target']),int(row['variance_inflation'])]
        require((int(row['n_E_nominal_only']) if row['n_E_nominal_only'] else None)==expected,'SAVED_SENSITIVITY_TRANSFER')
    options=[]
    for rho in (.05,.02,.01):
        for factor in (1,2,4):
            nominal={r['id']:sensitivity[r['id']][rho,factor] for r in rows}
            for allocation in ('GROUP','COMMON_4Q'):
                for floor in (2,33):
                    counts=allocate(rows,nominal,allocation,floor);budgets={}
                    for system in ('H4','H6'):
                        selected={r['id']:counts[r['id']] for r in rows if r['system']==system}
                        budgets[system]=None if any(n is None for n in selected.values()) else call_scope(tasks_for(system+'_PR_EXPLORATION',selected),'PR')
                    options.append(dict(id=option_id(rho,factor,allocation,floor),relative_MCSE_target=rho,
                        variance_multiplier=factor,allocation=allocation,minimum_n_proposed=floor,
                        n_E_by_candidate_proposed=counts,system_call_scope_proposed=budgets,
                        trajectories=sum(b['trajectories'] for b in budgets.values()) if all(budgets.values()) else None,
                        compile_axes=sum(b['compile_axes'] for b in budgets.values()) if all(budgets.values()) else None,
                        adopted=False,population_guarantee=False))
    return dict(schema='H46_EXPLORATION_READINESS_V1',status='DRAFT_USER_ADOPTION_AND_EXECUTION_REQUIRED',
        pilot_evaluation_commit=BASE,inputs=inputs,cohort=COHORT,groups=rows,options=options,
        recommended_option=RECOMMENDED,n_E_adopted=None,minimum_33_rationale='OPTIONAL_SAME_NORMAL_IID_VARIANCE_REFERENCE_RELATIVE_SE_25PCT_AS_PILOT_NOT_TAIL_GUARANTEE',
        design_basis='SAVED_MAX_AXIS_VARIANCE_TO_MEAN_SQUARED_COUNTS_NO_NEW_STATISTICAL_ANALYSIS',
        covariance_treatment='RETAIN_SAVED_PAIRED_COVARIANCE_NO_AXIS_INDEPENDENCE_ASSUMPTION',
        sampling_law_changed=False,unknown_strata_costs=None,population_mean_guarantee=False,
        formal_shot_policy_adopted=False,AD03='DEFERRED',valid_grants=0,
        scientific_execution_authorized=False,pooling=False,adaptive_top_up=False,mandatory_STOP=True)

def selected_option(plan,option):
    choices=[o for o in plan['options'] if o['id']==option]
    require(len(choices)==1,'ONE_REGISTERED_DESIGN_OPTION');return choices[0]
