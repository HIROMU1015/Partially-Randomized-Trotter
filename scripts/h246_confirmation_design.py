"""Authorized planning arithmetic on fixed saved statistics, never new samples."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

ROOT=Path(__file__).resolve().parents[1]
BASE='8613b3ecda2030a677b1690fe7aa3da23dec4873'
RESULT='artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/evaluation.json'
RESULT_SHA='b6d1309fc615c6df2c3a65c81f0a62dfff327435e98fde3bc2d74f3b80504f5e'
PROPOSAL='references/h246_confirmation_design_v1.json'
COHORT='CONFIRMATION'
FIXED=(('H2',.0005,('2nd',4),('4th',2)),
       ('H4',.0005,('2nd',8),('4th(m5_best)',1)),
       ('H6',.0005,('2nd',8),('4th(m5_best)',1)),
       ('H4',.005,('2nd',4),('4th',2)))

def require(ok,reason):
    if not ok:raise ValueError(reason)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def marginal_n(mean,variance,rho,factor,floor):
    require(mean>0 and variance>=0 and 0<rho<1 and factor>=1 and floor>=2,'DESIGN_INPUTS')
    if variance==0:return floor,True
    return max(floor,math.ceil(factor*variance/(rho*mean)**2)),False

def pair_plan(a,b,*,rho=.02,factor=4,floor=9,family=3,power=.9,halfwidth_fraction=.2):
    """Bonferroni normal approximation; no small-sample/tail guarantee."""
    require(0<power<1 and family>=1 and 0<halfwidth_fraction<1,'PAIR_DESIGN_PARAMETERS')
    gap=a['total_RZ']-b['total_RZ'];require(gap>0,'FIXED_EXPECTED_POSITIVE_DIRECTION')
    va,vb=a['total_RZ_variance'],b['total_RZ_variance']
    require(va is not None and vb is not None,'SAVED_VARIANCES_REQUIRED')
    z=NormalDist().inv_cdf(1-.05/(2*family));zp=NormalDist().inv_cdf(power)
    # Two predeclared goals: nominal power at the observed gap and interval
    # halfwidth <=20% of that gap. Do not adapt to confirmation observations.
    variance_budget=min((gap/(z+zp))**2,(halfwidth_fraction*gap/z)**2)
    sd=[math.sqrt(factor*v) for v in (va,vb)];total_sd=sum(sd)
    joint=[max(floor,math.ceil(s*total_sd/variance_budget)) for s in sd]
    marginal=[marginal_n(r['total_RZ'],v,rho,factor,floor) for r,v in zip((a,b),(va,vb))]
    n=[max(x,m[0]) for x,m in zip(joint,marginal)]
    se=math.sqrt(factor*va/n[0]+factor*vb/n[1])
    return dict(relative_MCSE_target=rho,variance_inflation=factor,minimum_n=floor,
      minimum_rationale='NORMAL_IID_REFERENCE_RELATIVE_SE_OF_VARIANCE_50PCT:ceil(2/.5**2)+1;NOT_GUARANTEE',
      family_size=family,family_alpha=.05,nominal_power_target=power,
      interval='TWO_SIDED_NORMAL_BONFERRONI_APPROXIMATION_ONLY',z=z,z_power=zp,
      gap_halfwidth_fraction_target=halfwidth_fraction,observed_gap=gap,
      marginal_n=[m[0] for m in marginal],difference_precision_n=joint,n_C=n,
      zero_variance_unresolved=[m[1] for m in marginal],
      planned_difference_SE=se,planned_normal_halfwidth=z*se,
      planned_normal_halfwidth_over_observed_gap=z*se/gap,
      guarantee=False,adopted_for_execution=False)

def make_design(value):
    from h46_m5_cost_design import call_scope
    pairs=[]
    for system,eps,left,right in FIXED:
        chosen=[]
        for formula,q in (left,right):
            gs=[g for g in value['groups'] if (g['candidate']['system'],g['candidate']['method'],g['candidate']['formula'],g['candidate']['q'])==(system,'PR',formula,q)]
            require(len(gs)==1,'UNIQUE_PREDECLARED_CANDIDATE')
            g=gs[0];r=next(r for r in g['rows'] if r['mode']=='USUAL_SIGNED' and r['epsilon']==eps)
            require(r['status']=='ELIGIBLE' and r['shot']['N_total'] is not None,'FIXED_SAVED_SHOTS')
            chosen.append(dict(candidate=g['candidate'],cost_row=r,unit_statistics=g['unit_statistics'],strata=g['strata']))
        key=system+('_PRIMARY' if eps==.0005 else '_AUXILIARY')
        family=3 if eps==.0005 else 1
        a,b=[g['cost_row'] for g in chosen]
        scenarios=[pair_plan(a,b,rho=rho,factor=factor,floor=floor,family=family)
          for rho in (.05,.02,.01) for factor in (1,2,4) for floor in (9,33)]
        recommended=next(x for x in scenarios if x['relative_MCSE_target']==.02 and x['variance_inflation']==4 and x['minimum_n']==9)
        tasks=[]
        for group,n in zip(chosen,recommended['n_C']):
            tasks.extend(dict(candidate=group['candidate'],id='DESIGN_ONLY_'+group['candidate']['id']+'_'+str(i)) for i in range(n))
        pairs.append(dict(id=key,system=system,epsilon=eps,primary=eps==.0005,
          expected_direction='LEFT_S2_MINUS_RIGHT_HIGHER_PF_POSITIVE',groups=chosen,
          recommended=recommended,sensitivity_scenarios=scenarios,
          proposed_scientific_calls=call_scope(tasks,'PR'),
          auxiliary_default='DEFER_NOT_INCLUDED_IN_PRIMARY_GRANTS' if eps!=.0005 else None))
    return dict(status='DESIGN_ONLY_PENDING_EXPLICIT_SCIENTIFIC_APPROVAL',base_commit=BASE,
        saved_result_sha256=RESULT_SHA,cohort=COHORT,comparisons=pairs,
        primary_ids=[p['id'] for p in pairs if p['primary']],
        candidate_reselection=False,pilot_or_exploration_pooling=False,adaptive_top_up=False,
        rare_event_policy='UNCONDITIONAL_K4_NO_FORCING_REJECTION_OR_TAIL_COST_IMPUTATION',
        science_started=False,AD03='DEFERRED',mandatory_STOP=True)

def classify_confirmation(left,right,*,z):
    """Freeze direction; do not reselect even when the sign reverses."""
    difference=left['mean']-right['mean'];se=math.hypot(left['SE'],right['SE'])
    low,high=difference-z*se,difference+z*se
    return dict(difference=difference,difference_SE=se,approximate_interval=[low,high],
      point_status='REPRODUCED' if difference>0 else 'REVERSED' if difference<0 else 'TIED',
      population_rank='NOT_CERTIFIED',approximate_direction='POSITIVE' if low>0 else 'NEGATIVE' if high<0 else 'UNRESOLVED',
      interval_is_rigorous=False,rare_tail_bound=None,reselection=False)

def main():
    import argparse,subprocess
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    path=ROOT/RESULT
    require(sha(path)==RESULT_SHA and path.read_bytes()==subprocess.check_output(['git','show',BASE+':'+RESULT],cwd=ROOT),'FIXED_SAVED_RESULT')
    out=make_design(read(path));save(a.output,out)
    print(json.dumps({p['id']:dict(n_C=p['recommended']['n_C'],calls=p['proposed_scientific_calls']) for p in out['comparisons']},ensure_ascii=False))

if __name__=='__main__':main()
