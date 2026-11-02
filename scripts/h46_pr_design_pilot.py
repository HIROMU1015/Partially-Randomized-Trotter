"""Outer32 DESIGN_PILOT metadata and future statistics; stdlib only.

No molecular data is evaluated by importing or generating proposals. Numerical
functions below are exercised on artificial observations only during readiness.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import h46_m5_cost_design as previous

BASE='489422d38bad4d5892cf98c580d93cc4b602bf0c'
ROOT=Path(__file__).resolve().parents[1]
DOMAIN='references/h46_m5_signal_domain_v1.json'
COHORT='DESIGN_PILOT'
STAGES=('H4_DESIGN_PILOT','H6_DESIGN_PILOT')
STRATA=('S0','S2_single','S2_multi','S4')
COMPILER=previous.COMPILER
COMPILER_HASH=previous.COMPILER_HASH
NATIVE_POLICY=previous.NATIVE_POLICY
require=previous.require
digest=previous.digest
call_scope=previous.call_scope

def groups():
    b=(ROOT/DOMAIN).read_bytes()
    require(b==subprocess.check_output(['git','show',BASE+':'+DOMAIN],cwd=ROOT),'FIXED_88_DOMAIN')
    rows=[r for r in json.loads(b)['rows'] if r['method']=='PR']
    require(len(rows)==24 and len({r['id'] for r in rows})==24,'EXACT_24_GROUPS')
    for system,prefix in [('H4',3),('H6',5)]:
        subset=[r for r in rows if r['system']==system]
        require({(r['formula'],r['q']) for r in subset}=={(f,q) for f in previous.FORMULAS for q in (1,2,4,8)},'FULL_FORMULA_Q_DOMAIN')
        require(all(r['prefix']==prefix and r['K']==4 and r['r_policy']=='ABS_TIME_PER_OUTER' and r['r_value']==32 for r in subset),'FIXED_OUTER32_K4_PREFIX')
    return rows

def tasks_for(stage,n_by_candidate):
    require(stage in STAGES,'PILOT_STAGE_ONLY')
    rows=[r for r in groups() if r['system']==stage[:2]]
    require(set(n_by_candidate)=={r['id'] for r in rows},'EXACT_12_PILOT_COUNTS')
    result=[]
    for r in rows:
        n=n_by_candidate[r['id']]
        require(type(n) is int and n>=2,'PILOT_N_NOT_ADOPTED')
        result.extend(dict(id='DP1__'+r['id']+'__'+str(i),candidate=r,sample=i,
                           axes=['cosine','sine'],stage=stage) for i in range(n))
    return result

def nominal_variance_sample_size(relative_se):
    require(type(relative_se) in (float,int) and 0<relative_se<1,'VARIANCE_PRECISION_TARGET')
    # Normal IID reference only: Var(S^2)/sigma^4 = 2/(n-1). Not a tail bound.
    return math.ceil(2/(relative_se*relative_se))+1

def proposal():
    rows=groups();options=[]
    for name,target in [('COARSE_SCREEN',.5),('VARIANCE_PLANNING',.25)]:
        n=nominal_variance_sample_size(target)
        system_scopes={}
        for system in ('H4','H6'):
            counts={r['id']:n for r in rows if r['system']==system}
            system_scopes[system]=call_scope(tasks_for(system+'_DESIGN_PILOT',counts),'PR')
        options.append(dict(id=name,n_P_proposed=n,normal_reference_relative_SE_of_variance=target,
                            guarantee=False,adopted=False,system_call_scope_proposed=system_scopes))
    return dict(schema='H46_OUTER32_DESIGN_PILOT_PROPOSAL_V1',base_commit=BASE,
        status='DRAFT_TARGET_AND_SAMPLE_COUNTS_REQUIRE_USER_ADOPTION',cohort=COHORT,
        recommended_option='VARIANCE_PLANNING',options=options,
        rationale='COMMON_NOMINAL_VARIANCE_PRECISION_TARGET;NOT_EQUAL_VARIANCE_OR_NORMAL_TAIL_ASSUMPTION',
        groups=[dict(candidate=r,n_P_proposed=33,n_P_adopted=None,n_E=None,n_C=None,
                     same_law_variance='NOT_OBSERVED',population_tail_bound=None) for r in rows],
        compiler=COMPILER,compiler_settings_hash=COMPILER_HASH,native_policy=NATIVE_POLICY,
        formal_shot_policy_adopted=False,AD03='DEFERRED',valid_grants=0,
        scientific_execution_authorized=False,adaptive_top_up=False,mandatory_STOP=True)

def seed(source,campaign,cohort,task_id,outer,piece):
    require(cohort in ('DESIGN_PILOT','EXPLORATION','CONFIRMATION'),'EXPLICIT_SEPARATE_COHORT')
    require(isinstance(campaign,str) and bool(campaign.strip()),'FRESH_CAMPAIGN')
    return int(digest(['H46_OUTER32_DESIGN_V1',source,campaign,cohort,task_id,outer,piece])[:16],16)

def registry(tasks,source,campaign,cohort,excluded):
    require(len(excluded)==len(set(excluded)) and all(type(x) is int for x in excluded),'UNIQUE_EXCLUDED_INTEGER_SEEDS')
    occupied=set(excluded);result={}
    for t in tasks:
        require(t['id'] not in result,'DUPLICATE_PILOT_TASK')
        r=t['candidate'];pieces={'2nd':1,'4th':3,'4th(m5_best)':11}[r['formula']]
        values={}
        for outer in range(r['q']):
            for piece in range(pieces):
                v=seed(source,campaign,cohort,t['id'],outer,piece)
                require(v not in occupied,'SEED_COLLISION_NO_REHASH_OR_RETRY')
                occupied.add(v);values[f'{outer}:{piece}']=v
        result[t['id']]=values
    return result

def event_stratum(events):
    old=previous.event_stratum(events)
    labels={'s0':'S0','s2_single':'S2_single','s2_multi':'S2_multi','s4':'S4'}
    return {**old,'stratum':labels[old['stratum']]}

def describe_pilot(rows,expected_n,*,relative_MCSE_targets,variance_inflations):
    """Future approved pilot-only statistics. Never rank, shot-weight, or pool.

    Called with artificial rows only during preparation. n_E outputs are nominal
    planning sensitivities, never population guarantees or automatic adoption.
    """
    require(type(expected_n) is int and expected_n>=2 and len(rows)==expected_n,'COMPLETE_FIXED_PILOT_ONLY')
    require(len({r['trajectory_id'] for r in rows})==expected_n,'UNIQUE_PAIRED_TRAJECTORIES')
    require(all(r['cohort']==COHORT and r['conditional'] is False and r['paired'] is True for r in rows),'NO_POOLING_CONDITIONING_OR_UNPAIRED_AXES')
    require(len({r['candidate_id'] for r in rows})==1 and len({r['campaign_id'] for r in rows})==1,'ONE_CANDIDATE_ONE_COHORT')
    require(all(r['stratum'] in STRATA for r in rows),'KNOWN_STRATUM')
    for r in rows:
        require(all(type(r[a]) is int and r[a]>0 for a in ('cosine_RZ','sine_RZ')),'POSITIVE_INTEGER_PAIRED_COSTS')
    x=[r['cosine_RZ'] for r in rows];y=[r['sine_RZ'] for r in rows]
    mx,my=statistics.mean(x),statistics.mean(y)
    vxx,vyy=statistics.variance(x),statistics.variance(y)
    vxy=sum((a-mx)*(b-my) for a,b in zip(x,y))/(expected_n-1)
    # Evaluate pair sums directly to avoid cancellation in almost anticorrelated
    # artificial data. The covariance identity is reported, not assumed zero.
    pairs=[a+b for a,b in zip(x,y)]
    vp=statistics.variance(pairs)
    strata={}
    for label in STRATA:
        selected=[r['cosine_RZ']+r['sine_RZ'] for r in rows if r['stratum']==label]
        strata[label]=dict(observed_count=len(selected),status='OBSERVED' if selected else 'UNOBSERVED',
            observed_conditional_pair_mean=statistics.mean(selected) if selected else None,
            conditional_variance=statistics.variance(selected) if len(selected)>=2 else None,
            population_probability=None,population_tail_cost_bound=None)
    planning=[]
    require(all(type(r) in (float,int) and 0<r<1 for r in relative_MCSE_targets),'MCSE_TARGETS')
    require(all(type(f) in (float,int) and f>=1 for f in variance_inflations),'VARIANCE_INFLATIONS')
    for rho in relative_MCSE_targets:
        for factor in variance_inflations:
            n=None if vxx==0 or vyy==0 else max(2,math.ceil(factor*vxx/(rho*mx)**2),math.ceil(factor*vyy/(rho*my)**2))
            planning.append(dict(relative_MCSE_target=rho,variance_inflation=factor,n_E_nominal_only=n,
                reason='ZERO_OBSERVED_AXIS_VARIANCE_NOT_POPULATION_ZERO' if n is None else 'PLANNING_ONLY_USER_ADOPTION_REQUIRED',
                adopted=False,tail_guarantee=False))
    return dict(n=expected_n,axis_means=dict(cosine=mx,sine=my),
        covariance=[[vxx,vxy],[vxy,vyy]],axis_MCSE=dict(cosine=math.sqrt(vxx/expected_n),sine=math.sqrt(vyy/expected_n)),
        pair_unit_RZ_mean=statistics.mean(pairs),pair_unit_RZ_variance=vp,pair_unit_RZ_MCSE=math.sqrt(vp/expected_n),
        pair_is_unweighted_unit_cost_not_shot_accounting=True,strata=strata,n_E_sensitivity=planning,
        population_variance_guaranteed=False,population_rank_guaranteed=False,
        rare_tail_unresolved=True,formal_shot_policy_adopted=False,
        ranking=False,automatic_top_up=False,automatic_n_E_adoption=False,mandatory_STOP=True)
