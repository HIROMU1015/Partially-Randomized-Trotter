"""Scalar design kernel. Real summaries may enter only after the runner gate."""
from fractions import Fraction
import math
from statistics import NormalDist

import h4_tol_s_stats_contract as s
c=s.c


def proposed_P_calls(cells,n_fixed):
    total=sum(n_fixed.values())
    return dict(trajectory=total,native_evolution=total,wrapper_compile=2*total,
        occurrence=sum(x['q']*n_fixed[x['id']] for x in cells),
        RTE_event=sum(x['R']*n_fixed[x['id']] for x in cells),
        control_probe_maximum=4*len(cells),independent_event_oracle_maximum=len(cells),
        seed_slots=sum((1 if x['q']==1 else 1+x['q'])*n_fixed[x['id']] for x in cells),
        endpoint_accounting_pass=1,endpoint_comparisons=2,PF_compile=0,signal=0,
        SCF_DF_state_reference=0,bootstrap=0,shot_recalculation=0,P_executed=False)


def analyze(summary):
    """Artificial fixtures or a separately approved sole real design pass."""
    p=s.plan();endpoints=summary['endpoints']
    c.require([(x['epsilon'],x['cell']['id'],x['PF']['cell_id']) for x in endpoints]==list(s.TARGETS),
        'TWO_FIXED_PRIMARY_ENDPOINTS')
    c.require(summary['binding']==s.main.BINDING and not summary['old_FP_or_legacy_pooled'],'DF10_SEPARATE_COHORT')
    z_family=NormalDist().inv_cdf(1-p['family_alpha']/p['family_size'])
    z_power=NormalDist().inv_cdf(p['power']);scenarios=[];directions=[]
    for endpoint in endpoints:
        pr,pf=endpoint['PR'],endpoint['PF'];n0,se=pr['n'],pr['G_RZ_SE'];cid=endpoint['cell']['id']
        c.require(pr['cell_id']==cid and pr['cohort']=='EXPLORATION_N8' and n0==8
            and math.isfinite(se) and se>0,'SAVED_EXPLORATION_N8_ONLY')
        cost=Fraction(pf['G_RZ_exact']);c.require(cost>0,'POSITIVE_SAVED_PF_COST')
        sd=math.sqrt(n0)*se
        gap=cost-Fraction(pr['G_RZ_exact']);direction=(gap>0)-(gap<0)
        c.require(direction==endpoint['saved_point_direction'],'FIXED_EXPLORATION_DIRECTION')
        directions.append(dict(epsilon=endpoint['epsilon'],cell_id=cid,PR_mean_exact=pr['G_RZ_exact'],
            saved_n=n0,saved_paired_SE=se,PF_cell_id=pf['cell_id'],PF_G_exact=pf['G_RZ_exact'],
            PF_minus_PR_exact=str(gap),expected_direction=direction))
        for inflation in p['SD_inflation_sensitivity']:
            for eta in p['relative_gap_sensitivity']:
                n=c.normal_n_formula(inflation*sd,float(eta*cost),z_family,z_power)
                scenarios.append(dict(epsilon=endpoint['epsilon'],cell_id=cid,saved_n=n0,saved_paired_SE=se,
                    s_plan_from_saved_SE=sd,PF_cell_id=pf['cell_id'],PF_G_exact=pf['G_RZ_exact'],
                    SD_inflation=inflation,relative_gap=eta,n=n))
    cells=[e['cell'] for e in endpoints];sensitivities=[]
    for inflation in p['SD_inflation_sensitivity']:
        for eta in p['relative_gap_sensitivity']:
            n_fixed={x['id']:max(r['n'] for r in scenarios if r['cell_id']==x['id']
                and r['SD_inflation']==inflation and r['relative_gap']==eta) for x in cells}
            sensitivities.append(dict(SD_inflation=inflation,relative_gap=eta,n_fixed=n_fixed,
                proposed_P_calls=proposed_P_calls(cells,n_fixed)))
    reference=next(x for x in sensitivities if x['SD_inflation']==4 and x['relative_gap']==.02)
    c.require(len(scenarios)==24 and len(directions)==2 and len(sensitivities)==12,'COMPLETE_FIXED_STATS_SCOPE')
    return dict(plan=p,normal_scenarios=scenarios,reference_directions=directions,sensitivity=sensitivities,
        recommended=reference,quantiles=dict(z_one_sided_Bonferroni=z_family,z_power=z_power),
        rare_information=[dict(cell_id=e['cell']['id'],K=e['cell']['K'],saved_counts=e['saved_rare_counts'],
            law_probabilities=None,n_visibility=None,unseen_cost=None,unobserved_strata_zero_filled=False) for e in endpoints],
        actual_calls=p['calls'],population_mean_certified=False,actual_power_certified=False,
        raw_samples_read=False,old_FP_or_legacy_pooled=False,CI_or_p_values_computed=False,
        new_seeds_generated=0,P_executed=False,next_stage_authorized=False)
