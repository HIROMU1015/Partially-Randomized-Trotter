#!/usr/bin/env python3
"""D1 saved scalar/Fraction analysis only. No generator, matrix or science imports."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
from itertools import combinations
import hashlib
import json
import csv
import math
import os
import resource
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
BASE='fc96571881f301435a655f057709df4d6c1678e0'
RESULT='artifacts/track_b_rte_development_d1_result/2026-10-13/v1'
PREP='artifacts/track_b_rte_development_source/2026-10-13/v1'
OUT='artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1'
METRICS=('T','CX','1Q')


def exact_log_upper(q,terms=96):
    """The frozen Bernstein log constant, not a new shot budget or precision."""
    q=F(q);power=0
    if q<1:raise ValueError('log argument must be >=1')
    while q>=2:q/=2;power+=1
    def f(z):
        return 2*sum(z**(2*k+1)/F(2*k+1) for k in range(terms))+2*z**(2*terms+1)/((2*terms+1)*(1-z*z))
    return power*f(F(1,3))+f((q-1)/(q+1))


def product_allocation(before,after,scale=F(1)):
    """Multilinear Shapley: interactions split symmetrically, exact conservation."""
    keys=list(before);before={k:F(v) for k,v in before.items()};after={k:F(after[k]) for k in keys}
    delta={k:after[k]-before[k] for k in keys};values={};coefficients={}
    for k in keys:
        others=[j for j in keys if j!=k];coefficient=F(0)
        for count in range(len(others)+1):
            for subset in combinations(others,count):
                coefficient+=math.prod(delta[j] if j in subset else before[j] for j in others)/F(count+1)
        coefficients[k]=F(scale)*coefficient;values[k]=delta[k]*coefficients[k]
    whole=F(scale)*(math.prod(after.values())-math.prod(before.values()))
    assert sum(values.values())==whole
    return dict(contributions=values,secant_coefficients=coefficients,total_difference=whole,
                convention='symmetric multilinear interaction allocation; descriptive, not unique causality')


def budget_decomposition(before,after,log_constant):
    """Endpoint formula differences; never create new candidates/budgets or change N."""
    b0,b1=before['budget'],after['budget'];s0,s1=F(b0['remaining']),F(b1['remaining'])
    m0,m1=F(b0['M2_upper']),F(b1['M2_upper']);w0,w1=F(b0['range_upper']),F(b1['range_upper'])
    u0,u1=1/s0**2,1/s1**2;v0,v1=1/s0,1/s1
    moment=(u0+u1)*(m1-m0);ran=F(2,3)*(v0+v1)*(w1-w0)
    margin=(m0+m1)*(u1-u0)+F(2,3)*(w0+w1)*(v1-v0)
    secant=(margin/(s1-s0) if s1!=s0 else -2*(m0+m1)/s0**3-F(2,3)*(w0+w1)/s0**2)
    dt=-2*(F(b1['Taylor_product_upper'])-F(b0['Taylor_product_upper']))
    di=-2*(F(b1['implementation_product_upper'])-F(b0['implementation_product_upper']))
    assert dt+di==s1-s0
    brackets=[2*m0*u0+F(4,3)*w0*v0,2*m1*u1+F(4,3)*w1*v1]
    assert moment+ran+margin==brackets[1]-brackets[0]
    n0,n1=b0['N_per_axis'],b1['N_per_axis']
    rounding=[F(n0)-log_constant*brackets[0],F(n1)-log_constant*brackets[1]]
    assert all(0<=v<1 for v in rounding)
    parts=dict(moment=log_constant*moment,range=log_constant*ran,
               Taylor_margin=log_constant*secant*dt,
               implementation_margin=log_constant*secant*di,
               integer_ceiling=rounding[1]-rounding[0])
    assert sum(parts.values())==n1-n0
    return dict(N_difference=n1-n0,contributions=parts,ceiling_remainders=rounding,
                margin_changes=dict(Taylor=dt,implementation=di),
                endpoint_variance_bracket=[2*m0*u0,2*m1*u1],
                endpoint_range_bracket=[F(4,3)*w0*v0,F(4,3)*w1*v1],
                no_new_N_or_counterfactual_candidate_score=True)


def interval_leq(intercept,slope):
    """Closed h>=0 region where intercept+slope*h<=0."""
    a,b=F(intercept),F(slope)
    if b==0:return dict(empty=a>0,lo=F(0),hi=None)
    cross=-a/b
    if b>0:return dict(empty=cross<0,lo=F(0),hi=cross if cross>=0 else None)
    return dict(empty=False,lo=max(F(0),cross),hi=None)


def pair_relation(before,after,metric):
    da=F(after['expected_native_intercept'][metric])-F(before['expected_native_intercept'][metric])
    db=F(after['prep_readout_slope'])-F(before['prep_readout_slope'])
    crossing=-da/db if db and -da/db>=0 else None
    return dict(baseline=label(before),alternative=label(after),metric=metric,
        difference_intercept=da,difference_slope=db,crossover=crossing,
        alternative_nonworse_region=interval_leq(da,db),baseline_nonworse_region=interval_leq(-da,-db),
        equality_all_h=da==db==0)


def label(row):return 'r'+str(row['steps'])+':'+row['arm']


def ranking(rows,metric,h):
    groups={}
    for r in rows:
        value=F(r['expected_native_intercept'][metric])+F(r['prep_readout_slope'])*h
        groups.setdefault(value,[]).append(label(r))
    return [sorted(groups[v]) for v in sorted(groups)]


def pareto_zero(rows):
    out=[]
    for target in rows:
        dominated=[]
        for candidate in rows:
            if candidate is target:continue
            a=[F(candidate['expected_native_intercept'][k]) for k in METRICS]+[F(candidate['workspace_beyond_system'])]
            b=[F(target['expected_native_intercept'][k]) for k in METRICS]+[F(target['workspace_beyond_system'])]
            if all(v<=w for v,w in zip(a,b)) and any(v<w for v,w in zip(a,b)):dominated.append(label(candidate))
        if not dominated:out.append(label(target))
    return out


def sensitivity(rows,metric):
    pairs=[pair_relation(a,b,metric) for a,b in combinations(rows,2)]
    points=sorted({F(0)}|{v['crossover'] for v in pairs if v['crossover'] is not None})
    at_points=[dict(h=h,ranking_tie_groups=ranking(rows,metric,h)) for h in points]
    intervals=[]
    for i,lo in enumerate(points):
        hi=points[i+1] if i+1<len(points) else None
        witness=(lo+hi)/2 if hi is not None else lo+1
        order=ranking(rows,metric,witness)
        intervals.append(dict(open_lo=lo,open_hi=hi,ranking_tie_groups=order,minimum_envelope_arms=order[0]))
    return dict(metric=metric,pairs=pairs,all_exact_nonnegative_crossings=points,
                endpoint_rankings=at_points,open_interval_rankings=intervals,
                h_is_this_coordinate_prep_readout_cost=True,not_a_new_experimental_grid=True)


def pareto_regions(rows):
    """All directed domination boxes for independent h_T,h_CX,h_1Q>=0."""
    boxes=[];equality=[]
    for a in rows:
        for b in rows:
            if a is b:continue
            if a['workspace_beyond_system']>b['workspace_beyond_system']:continue
            # G(a)-G(b) <=0 in every known quantum coordinate; at least one strict.
            delta_slope=F(a['prep_readout_slope'])-F(b['prep_readout_slope'])
            delta_intercepts={k:F(a['expected_native_intercept'][k])-F(b['expected_native_intercept'][k]) for k in METRICS}
            box={k:interval_leq(delta_intercepts[k],delta_slope) for k in METRICS}
            if any(v['empty'] for v in box.values()):continue
            if delta_slope==0 and all(v==0 for v in delta_intercepts.values()) and a['workspace_beyond_system']==b['workspace_beyond_system']:
                equality.append([label(a),label(b)]);continue
            locus=({k:-delta_intercepts[k]/delta_slope for k in METRICS}
                   if delta_slope and a['workspace_beyond_system']==b['workspace_beyond_system'] else None)
            if locus is not None and any(v<0 for v in locus.values()):locus=None
            if locus is not None and all(box[k]['lo']==box[k]['hi']==locus[k] for k in METRICS):continue
            boxes.append(dict(dominator=label(a),dominated=label(b),coordinate_box=box,
                exclude_all_coordinate_equality_point=locus,workspace=a['workspace_beyond_system'],
                strict_at_least_one_coordinate=True))
    return dict(h0_quantum_front=pareto_zero(rows),directed_dominance_regions=boxes,
        equal_all_h_pairs=equality,coordinates=['G_T(h_T)','G_CX(h_CX)','G_1Q(h_1Q)','workspace'],
        h_coordinates_independent=True,classical_unknown_excluded=True,
        is_full_resource_Pareto_claim=False,not_a_scalar_T_plus_1Q_score=True)


def cost_decomposition(before,after,metric,log_constant):
    def factors(row):
        z=F(row['reference_law_diagnostic_only']['acceptance']);r=row['steps']
        return dict(N=F(row['budget']['N_per_axis']),r=F(r),
                    acceptance_path=z**r,conditional_step_price=F(row['per_step_trial_native'][metric])/z)
    b,a=factors(before),factors(after);parts=product_allocation(b,a,F(2));shots=budget_decomposition(before,after,log_constant)
    nested={k:parts['secant_coefficients']['N']*v for k,v in shots['contributions'].items()}
    assert sum(nested.values())==parts['contributions']['N']
    observed=F(after['expected_native_intercept'][metric])-F(before['expected_native_intercept'][metric])
    assert parts['total_difference']==observed
    slope=product_allocation({k:b[k] for k in ('N','acceptance_path')},{k:a[k] for k in ('N','acceptance_path')},F(2))
    assert slope['total_difference']==F(after['prep_readout_slope'])-F(before['prep_readout_slope'])
    shares={k:(100*v/observed if observed else None) for k,v in parts['contributions'].items()}
    return dict(baseline=label(before),alternative=label(after),metric=metric,before_factors=b,after_factors=a,
        intercept_ratio=F(after['expected_native_intercept'][metric])/F(before['expected_native_intercept'][metric]),
        N_ratio=a['N']/b['N'],r_ratio=a['r']/b['r'],acceptance_path_ratio=a['acceptance_path']/b['acceptance_path'],
        conditional_price_ratio=a['conditional_step_price']/b['conditional_step_price'],
        intercept_difference=observed,cost_product_allocation=parts,signed_percent_of_net_difference=shares,
        N_formula_allocation=shots,nested_cost_from_N=nested,prep_readout_slope_allocation=slope,
        arbitrary_h_difference='intercept_difference+h*prep_readout_slope_allocation.total_difference',
        negative_difference_means_alternative_lower_under_saved_model=True,
        descriptive_allocation_not_unique_causal_effect=True)


def decimal_display(q):
    if q is None:return None
    q=F(q)
    with localcontext() as c:c.prec=12;return str(Decimal(q.numerator)/Decimal(q.denominator))


class Output:
    def __init__(self,path):self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False);self.inventory=[]
    def write(self,name,value):
        def convert(v):
            if isinstance(v,F):return str(v)
            raise TypeError('unsupported analysis value: '+type(v).__name__)
        data=(json.dumps(value,indent=2,ensure_ascii=False,default=convert)+'\n').encode()
        if len(data)>524288:raise RuntimeError('analysis file must be <=512KiB')
        p=self.path/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(data)
        self.inventory.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))


def main():
    start=time.monotonic();cpu=time.process_time()
    paths=[RESULT+'/resource_map.json',RESULT+'/saved_output_audit/saved_output_audit_v1.json',
           RESULT+'/outer_process_receipt.json',PREP+'/contract_v1.json']
    input_identity=[];inputs={}
    for path in paths:
        blob=(ROOT/path).read_bytes();immutable=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+path])
        assert blob==immutable
        input_identity.append(dict(path=path,commit=BASE,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
        inputs[path]=json.loads(blob)
    source=inputs[RESULT+'/resource_map.json'];c=inputs[PREP+'/contract_v1.json']
    assert source['classification']=='D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
    assert inputs[RESULT+'/saved_output_audit/saved_output_audit_v1.json']['status']=='SAVED_OUTPUT_AND_PROVENANCE_AUDIT_PASS'
    rows=source['rows'];assert len(rows)==12 and c['steps']==[1,2]
    log=exact_log_upper(2/F(c['alpha_axis']))
    out=Output(ROOT/OUT);out.write('input_identity_and_scope_v1.json',dict(inputs=input_identity,
        source_S='874ca12d347f8f35e5d8d64f65f24abc4a2df523',authorization_A='f803291638d453873e4ab6f1ecf7bcc607f48e4b',
        result_commit=BASE,scope='post-hoc saved scalar accounting only; no candidate/shot policy/result classification changes',
        common_frozen_log_upper=log,log_terms=96,scientific_run=False,new_science_synthesis_matrix_sampling=0,
        revalidation_of_1890_events_and_36_keys=False,mandatory_STOP=True))
    scalar=[];groups=[];all_decomp=[];primary=[]
    for row in rows:
        r=row['steps'];b=row['budget'];z=F(row['reference_law_diagnostic_only']['acceptance']);m=F(b['M2_upper']);w=F(b['range_upper']);s=F(b['remaining'])
        item=dict(label=label(row),steps=r,x=row['x'],arm=row['arm'],N_per_axis=b['N_per_axis'],
            step_acceptance=z,path_acceptance=z**r,step_weight_M2=F(row['reference_law_diagnostic_only']['M2']),
            path_weight_M2=F(row['reference_law_diagnostic_only']['M2'])**r,
            M2_budget_upper=m,range_budget_upper=w,normalizer_envelope_upper=row['production_step_bounds']['norm_or_envelope_upper'],
            Taylor_product_upper=b['Taylor_product_upper'],implementation_product_upper=b['implementation_product_upper'],
            axis_bias_upper=b['axis_bias_upper'],remaining=s,
            continuous_variance_term_share=(2*m/s**2)/(2*m/s**2+F(4,3)*w/s),
            continuous_range_term_share=(F(4,3)*w/s)/(2*m/s**2+F(4,3)*w/s),
            expected_quantum_calls=row['expected_quantum_calls'],prep_readout_slope=row['prep_readout_slope'],workspace=row['workspace_beyond_system'],
            **{k+'_'+field:row[key][k] for k in METRICS for field,key in [('intercept','expected_native_intercept'),('per_step_trial','per_step_trial_native'),('per_whole_trial','per_whole_trial_native')]})
        scalar.append(item)
    out.write('all_12_saved_scalar_rows_v1.json',scalar)
    for r in c['steps']:
        group=[v for v in rows if v['steps']==r];groups.append((f'r{r}',group))
        rankings={k:ranking(group,k,F(0)) for k in METRICS}
        auxiliary={}
        for name,path in [('N_per_axis',lambda v:F(v['budget']['N_per_axis'])),('M2_budget_upper',lambda v:F(v['budget']['M2_upper'])),('path_acceptance_ascending_not_better',lambda v:F(v['reference_law_diagnostic_only']['acceptance'])**r)]:
            numeric={}
            for row in group:numeric.setdefault(path(row),[]).append(label(row))
            auxiliary[name]=[dict(value=v,arms=arms) for v,arms in sorted(numeric.items())]
        out.write(f'rankings_r{r}_v1.json',dict(h0_total_quantum=rankings,descriptive_numeric_rankings=auxiliary,
                 quantum_only_pareto_h0=pareto_zero(group),classical_unknown=True))
        for i,(before,after) in enumerate(combinations(group,2)):
            for k in METRICS:
                d=cost_decomposition(before,after,k,log);name=f'within_r{r}/pair_{i:02d}_{k}.json';out.write(name,d);all_decomp.append(name)
                if before['arm']=='partial_return_tail' and after['arm']=='closed_P5_full':primary.append(dict(path=name,
                    **{key:d[key] for key in ('baseline','alternative','metric','intercept_ratio','N_ratio','conditional_price_ratio','cost_product_allocation','signed_percent_of_net_difference','nested_cost_from_N')}))
    groups.append(('joint_12',rows))
    for name,group in groups:
        out.write(f'pareto_regions_{name}_v1.json',pareto_regions(group))
        for k in METRICS:out.write(f'sensitivity_{name}_{k}_v1.json',sensitivity(group,k))
    repeat=[]
    for i,arm in enumerate(c['arms']):
        before=next(v for v in rows if v['steps']==1 and v['arm']==arm);after=next(v for v in rows if v['steps']==2 and v['arm']==arm)
        for k in METRICS:
            d=cost_decomposition(before,after,k,log);name=f'repeat_count/arm_{i:02d}_{k}.json';out.write(name,d);repeat.append(name)
    out.write('primary_P5_vs_partial_factor_summary_v1.json',primary)
    out.write('classical_and_information_gaps_v1.json',dict(saved_setup_timing=source['classical_setup_timing'],
        setup_CPU_seconds_total=sum(v['CPU_seconds'] for v in source['classical_setup_timing']),
        synthesis_CPU_seconds_saved=sum(v['CPU_seconds'] for v in source['worker_receipts']),
        outer_process_resource=inputs[RESULT+'/outer_process_receipt.json'],
        setup_is_production_plus_reference_enumeration_not_online_sampling=True,
        online_model='C_setup_arm_r + 2*N_arm_r*c_classical_attempt_arm_r; all trials including zeros',
        unmeasured=['online random-bit draw, dyadic generation, rejection/zero checks','per-trial word/primitive construction or cache access','prep/readout native prices','whole-circuit optimization','DF-scale Pauli acquisition / normalization access cost'],
        no_zero_fill_classical_cost_assumed_free=True,no_common_classical_unit_price_assumed=True,
        all_arms_have_same_explicit_I1_provider_access=True,DF_or_I0_acquisition_advantage_claim=False,
        full_resource_Pareto_undecided=True))
    manifest=dict(kind='D1_SAVED_TECHNICAL_ANALYSIS_COMPLETE',input_result_commit=BASE,
        input_result_classification_unchanged=source['classification'],files=list(out.inventory),
        within_r_pair_metric_decompositions=len(all_decomp),repeat_count_metric_decompositions=len(repeat),
        global_pair_metric_sensitivity=66*3,within_each_r_pair_metric_sensitivity=15*3,
        arithmetic_conservation_assertions_passed=True,repeated_event_sequence_audit=False,
        performance=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         scalar_rows_shared=12,parallel_workers=0,numeric_threads_not_applicable=True,GPU_query=0),
        new_science_synthesis_matrix_sampling_LP=0,original_source_contract_auth_result_marker_STOP_modified=False,
        method_adoption=False,scientific_GO_STOP_reclassification=False,mandatory_STOP=True,next_science_authorized=False)
    out.write('analysis_manifest_v1.json',manifest)
    print(json.dumps({'status':manifest['kind'],'within_r_decompositions':len(all_decomp),'r_contrasts':len(repeat),'files':len(out.inventory),
        'h0_fronts':{name:pareto_zero(group) for name,group in groups},'primary_display':[
            {'baseline':v['baseline'],'alternative':v['alternative'],'metric':v['metric'],'cost_ratio':decimal_display(v['intercept_ratio']),
             'N_ratio':decimal_display(v['N_ratio']),'price_ratio':decimal_display(v['conditional_price_ratio']),
             'signed_percent_shot':decimal_display(v['signed_percent_of_net_difference']['N']),
             'signed_percent_price':decimal_display(v['signed_percent_of_net_difference']['conditional_step_price'])} for v in primary]},ensure_ascii=False))


if __name__=='__main__':main()
