"""Authorized V3-R1: saved fixed-plan prices only; no physical/circuit calls.

python3 analyze_fixed_plan.py --repo REPOSITORY --output OUTPUT_DIRECTORY
Only unchanged selected AST definitions of fixed source files are executed.
"""
from __future__ import annotations
import argparse
import ast
import concurrent.futures
import csv
import dataclasses
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

V3='cc60beb51930d4311d2ded84440682e583ccafc7'
PREFIX='docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/'
AXES=('cosine','sine')
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],env=ENV)


def identity(raw):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def read_inputs(repo):
    index=json.loads(git(repo,'show',V3+':'+PREFIX+'EVIDENCE_INDEX.json'))
    for f in ['REPORT.md','MINIMAL_V3_VALIDATION_PLAN.md','PRICE_EVIDENCE_MAP.csv',
              'UNCERTAINTY_SCOPE.md','EVIDENCE_INDEX.json','validate_metadata.py']:
        raw=git(repo,'show',V3+':'+PREFIX+f)
        index['V3_'+f.split('.')[0]]=dict(commit=V3,path=PREFIX+f,
            url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+V3+'/'+PREFIX+f,**identity(raw))
    def read(item):
        key,e=item;raw=git(repo,'show',e['commit']+':'+e['path'])
        assert identity(raw)=={k:e[k] for k in ('bytes','sha256','blob')},key
        return key,json.loads(raw) if e['path'].endswith('.json') else raw.decode()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        inputs=dict(pool.map(read,index.items()))
    return inputs,index


def reuse_source(inputs):
    choices={
        'PRESRC':['affine_prediction_with_standard_error'],
        'ABLATIONSRC':['AffineAxisCostModel','PairCompiledCostModel'],
        'DECISIONSRC':['_semantic_fingerprint','_axis_model','_pair_model'],
        'OPT2SRC':['_measured_model_half_width','_interval','_comparison'],
    }
    module=types.ModuleType('r1_saved_source');sys.modules[module.__name__]=module
    ns=module.__dict__;ns.update(math=math,hashlib=hashlib,json=json,dataclass=dataclasses.dataclass,
        POLICY_LABEL='support_run_le_1')
    records=[]
    for key,names in choices.items():
        source=inputs[key];tree=ast.parse(source)
        nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        selected=[nodes[n] for n in names]
        unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
        ast.fix_missing_locations(unit);exec(compile(unit,'fixed:'+key,'exec'),ns)
        for n in selected:
            records.append(dict(source_id=key,name=n.name,line=n.lineno,end_line=n.end_lineno,
                unchanged_AST_sha256=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()))
    return ns,records


def covariance_envelope(contributions):
    """Sharp plug-in SD range over all PSD correlations with given diagonals.

    Contributions are signed w_i * marginal_SE_i. Gram-vector triangle bounds
    give max(0,2*max|b_i|-sum|b_i|) <= SD <= sum|b_i|.
    This is not a confidence interval or an estimate of unknown correlations.
    """
    values=[abs(v) for v in contributions];total=math.fsum(values)
    return max(0.,2*max(values,default=0.)-total),total


def difference_band(d,half_width):
    lo,hi=d-half_width,d+half_width
    return dict(difference_interval=[lo,hi],half_width=half_width,
                verdict='CONDITIONAL_SEPARATION' if hi<0 or lo>0 else 'UNDETERMINED',
                separated_candidate='Partial' if hi<0 else 'Full' if lo>0 else None,
                interpretation='1.96_SE_diagnostic_plus_assumed_systematic_band;not_rigorous_95pct_CI')


def correlated_block_variance(values,rho_q,rho_axes):
    """PSD R_q tensor R_axes reference, order q1c,q1s,q2c,q2s."""
    if len(values)!=4 or abs(rho_q)>1 or abs(rho_axes)>1:
        raise ValueError('Expected four weights and correlations in [-1,1]')
    terms=[x*y*(1. if i//2==j//2 else rho_q)*(1. if i%2==j%2 else rho_axes)
           for i,x in enumerate(values) for j,y in enumerate(values)]
    v=math.fsum(terms)
    assert v>=-1.e-12*max(1.,math.fsum(x*x for x in values))
    return max(0.,v)


def analyze(inputs,ns):
    a=inputs['OPT2JSON'];bests=a['coherent_reoptimization']['best_by_ld'];rows=[];groups={};models={}
    candidate_costs={};decimal_costs={};reconstructed={};source_se_terms={};agreements=[]
    for ld,label,policy in [(3,'Partial','support_run_le_1'),(12,'Full','deterministic_tail_free')]:
        best=bests[str(ld)]['best'];terms=[];dterms=[];se_terms=[];new_rounds=[]
        for saved in best['rounds']:
            q,r,k,m=(saved[x] for x in ('q_m','r_m','K_m','round_index'))
            cell=f'ld{ld}:delta0.02:r{r}';direct=a['direct_compiled_rz_measurements'][cell]
            if cell not in models:
                points={str(qc):{'axes':{axis:(
                    {'rz_count':p['compiled_rz'][policy][axis]} if ld==12 else
                    {'policies':{policy:{'rz_count':p['compiled_rz'][policy][axis]}}}) for axis in AXES}}
                    for qc,p in direct['q'].items() if qc in ('1','2')}
                models[cell]=ns['_pair_model'](points,rte_steps=r,finite_taylor_order=k,
                    policy=policy if ld==3 else None,source_fingerprint=a['compute_fingerprint'],
                    source_label='R1_saved_only_'+cell)
            round_terms=[];round_se=[]
            for axis in AXES:
                ar=saved['axes'][axis];n=ar['shots'];price,se=models[cell].predict(axis,q)
                stats=[direct['q'][str(t)]['compiled_rz'][policy][axis] for t in (1,2)]
                weights=[n*(2-q),n*(q-1)];cost=n*price;term_se=n*se
                gid=f'{cell}:K{k}:{policy}:{axis}'
                if gid not in groups:
                    mid='EXTMAN' if direct['source_partition']=='fresh32_extension' else 'INITIALMAN'
                    binding=[dict(q=t['q'],seed=t['seed'],r=t['r'],K=t['k'],
                        sample_count=t['parameters']['sample_count'],seed_source=t['parameters']['seed_source'])
                        for t in inputs[mid]['tasks']
                        if (t['ld'],t['delta'],t['r'])==(ld,.02,r) and t['q'] in (1,2)] if ld==3 else []
                    groups[gid]=dict(group_id=gid,candidate=label,ld=ld,delta_time=.02,r=r,K=k,axis=axis,policy=policy,
                        source_partition=direct['source_partition'],A1=0,A2=0,round_indices=[],q_values=[],
                        anchor_means=[v['mean'] for v in stats],anchor_SE=[v['standard_error'] for v in stats],
                        anchor_sample_counts=[direct['q'][str(t)]['sample_count'] for t in (1,2)],
                        calibration_task_binding=binding,
                        cosine_sine_covariance='UNKNOWN_PAIRED_VECTORS_NOT_FOUND',q1_q2_covariance='UNKNOWN;SOURCE_ASSUMES_INDEPENDENT',
                        source_selector='/direct_compiled_rz_measurements/'+cell)
                g=groups[gid];g['A1']+=weights[0];g['A2']+=weights[1];g['round_indices'].append(m);g['q_values'].append(q)
                terms.append(cost);round_terms.append(cost);se_terms.append(term_se);round_se.append(term_se)
                dterms.append(Decimal(n)*(Decimal(2-q)*Decimal(str(stats[0]['mean']))
                    +Decimal(q-1)*Decimal(str(stats[1]['mean']))))
                agreements.append(dict(unit_price_abs_error=abs(price-ar['predicted_rz_count_per_interrogation']),
                    unit_SE_abs_error=abs(se-ar['propagated_calibration_standard_error'])))
                rows.append(dict(candidate=label,ld=ld,round_index=m,q=q,r=r,K=k,axis=axis,policy=policy,
                    N=n,saved_alpha=ar['alpha'],group_id=gid,coefficient_q1=weights[0],coefficient_q2=weights[1],
                    calibration_q1_mean=stats[0]['mean'],calibration_q2_mean=stats[1]['mean'],
                    calibration_q1_SE=stats[0]['standard_error'],calibration_q2_SE=stats[1]['standard_error'],
                    calibration_q1_n=direct['q']['1']['sample_count'],calibration_q2_n=direct['q']['2']['sample_count'],
                    saved_unit_price=ar['predicted_rz_count_per_interrogation'],reproduced_unit_price=price,
                    reproduced_axis_cost=cost,saved_round_cost_BOTH_AXES=saved['compiled_rz_point_estimate'],
                    source_unit_sampling_SE=se,source_axis_N_times_SE=term_se,
                    holdout_q='8;A0_r32_only_16,32' if ld==3 and r==32 else '8' if ld==3 else '16,32',
                    systematic_scenarios='per_r_q8;local5;transfer25;A0_r32_sensitivity;unknown_long_q' if ld==3 else 'deterministic_q16_q32;local5;transfer25;unknown_long_q',
                    source_id='OPT2JSON',source_selector=f'/coherent_reoptimization/best_by_ld/{ld}/best/rounds/{m}/axes/{axis}'))
            new_rounds.append(dict(saved,compiled_rz_point_estimate=math.fsum(round_terms),
                calibration_standard_error_conservative_sum=math.fsum(round_se)))
        total=math.fsum(terms);candidate_costs[label]=total;decimal_costs[label]=str(sum(dterms,Decimal(0)))
        source_se_terms[label]=math.fsum(se_terms)
        reconstructed[str(ld)]=dict(best,rounds=new_rounds,total_compiled_rz_point_estimate=total,
            conservative_calibration_95_half_width=1.96*source_se_terms[label])
    for g in groups.values():
        g['group_point_cost']=math.fsum(g[key]*mean for key,mean in zip(('A1','A2'),g['anchor_means']))
        g['signed_D_anchor_weights']=[g['A1'],g['A2']] if g['candidate']=='Partial' else [-g['A1'],-g['A2']]
        g['anchor_covariance_with_other_groups']='UNKNOWN;independent_groups_is_reference_only'
        g['SE_if_q1_q2_independent']=math.hypot(g['A1']*g['anchor_SE'][0],g['A2']*g['anchor_SE'][1])
    for row in rows:
        g=groups[row['group_id']]
        row.update(group_A1=g['A1'],group_A2=g['A2'],group_SE_if_q1_q2_independent=g['SE_if_q1_q2_independent'],
            group_point_cost_DO_NOT_SUM_REPEATED_ROWS=g['group_point_cost'])
    gp,gf=candidate_costs['Partial'],candidate_costs['Full'];d=gp-gf;margin=-d
    reconstructed_comparison=ns['_comparison'](reconstructed['3'],reconstructed['12'],a['proxy_diagnostics'],
        a['deterministic_diagnostics'],delta3=.02,delta12=.02)
    b_contributions=[w*se for g in groups.values() for w,se in zip(g['signed_D_anchor_weights'],g['anchor_SE'])]
    independent_se=math.sqrt(math.fsum(x*x for x in b_contributions))
    lo_se,hi_se=covariance_envelope(b_contributions)
    source_half=1.96*math.fsum(source_se_terms.values())
    sampling={
        'saved_source_conservative_round_axis_sum':dict(SE_scale=source_half/1.96,
            assumptions='source_q1_q2_independence;conservative_N_SE_sum_over_rounds_axes;zero_systematic_DIAGNOSTIC_ONLY',**difference_band(d,source_half)),
        'shared_rounds_all_calibration_anchors_independent_REFERENCE':dict(SE=independent_se,
            assumptions='all_anchor_mean_errors_independent_including_paired_axes;not_established',**difference_band(d,1.96*independent_se)),
        'unknown_covariance_PSD_diagonal_envelope':dict(SE_minimum=lo_se,SE_maximum=hi_se,
            assumptions='only_saved_marginal_SE_squared_as_plugin_variances;all_PSD_correlations_allowed;not_population_bounds',
            **difference_band(d,1.96*hi_se)),
    }
    correlation_scenarios=[]
    for rho_axes in (-1.,0.,1.):
        for rho_q in (-1.,0.,1.):
            family_variances=[]
            for family in models:
                gg=[g for g in groups.values() if g['group_id'].startswith(family+':')]
                cs={g['axis']:g for g in gg}
                values=[cs[axis]['signed_D_anchor_weights'][i]*cs[axis]['anchor_SE'][i] for i in (0,1) for axis in AXES]
                family_variances.append(correlated_block_variance(values,rho_q,rho_axes))
            se=math.sqrt(math.fsum(family_variances))
            correlation_scenarios.append(dict(cosine_sine_correlation=rho_axes,q1_q2_correlation=rho_q,
                assumptions='PSD_Kronecker_correlation_within_family;families_independent;hypothetical_NOT_estimated',
                SE=se,**difference_band(d,1.96*se)))
    sampling['hypothetical_pairing_and_anchor_correlation_scenarios']=correlation_scenarios
    family_costs={label:{r:math.fsum(row['reproduced_axis_cost'] for row in rows if row['candidate']==label and row['r']==r)
        for r in sorted({row['r'] for row in rows if row['candidate']==label})} for label in ('Partial','Full')}
    late={label:math.fsum(row['reproduced_axis_cost'] for row in rows if row['candidate']==label and row['round_index']>=15)
          for label in ('Partial','Full')}
    r3_cost=math.fsum(row['reproduced_axis_cost'] for row in rows if row['candidate']=='Partial' and row['round_index'] in (4,5))
    selected_errors={r:a['proxy_diagnostics'][f'delta0.02:r{r}']['maximum_selected_rz_holdout_error'] for r in family_costs['Partial']}
    selected_width=ns['_measured_model_half_width'](reconstructed['3'],a['proxy_diagnostics'],policy='support_run_le_1',delta=.02)
    a0_error=inputs['A0']['summary']['maximum_selected_policy_absolute_relative_error']['absolute_relative_error']
    a0_errors=dict(selected_errors);a0_errors[32]=max(selected_errors[32],a0_error)
    a0_width=math.fsum(family_costs['Partial'][r]*err for r,err in a0_errors.items())
    bands=[('per_r_selected_q8_reference',selected_width,0.,'saved_short_q_discrepancy_convention_applied_to_model_prices;NOT_long_q_bound'),
           ('per_r_full_basis_auxiliary',reconstructed_comparison['measured_model_discrepancy_half_width']['ld3_full_basis_per_r'],0.,'auxiliary_policy_discrepancy_NOT_selected_policy_measurement'),
           ('per_r_A0_r32_replacement_REFERENCE',a0_width,0.,'r32_only_uses_max_q8_A0_selected_RZ_magnitude;direct_denominator_error_used_as_legacy_scenario_fraction;not_refit'),
           ('local_5_percent',.05*gp,.05*gf,'assumed_5pct_per_candidate_price_band;not_measured_bound'),
           ('transfer_25_percent',.25*gp,.25*gf,'assumed_25pct_transfer_price_band;not_measured_bound')]
    systematics=[]
    for name,hp,hf,assumption in bands:
        variants={key:difference_band(d,stat+hp+hf) for key,stat in (
            ('model_band_only_DIAGNOSTIC',0.),('plus_saved_source_sampling',source_half),
            ('plus_independent_anchors_REFERENCE',1.96*independent_se),('plus_unknown_covariance_PSD_upper',1.96*hi_se))}
        systematics.append(dict(name=name,assumption=assumption,partial_model_half_width=hp,full_model_half_width=hf,
            variants=variants,empirical_uniform_long_q_bound=False))
    exposure_specs=[('all_rounds_opposing_fraction',gp,gf),('partial_all_rounds_up',gp,0.),('full_all_rounds_down',0.,gf),
        ('late_three_opposing_fraction',late['Partial'],late['Full']),('partial_late_three_up',late['Partial'],0.),
        ('full_late_three_down',0.,late['Full']),('R3_partial_q16_q32_up',r3_cost,0.),
        ('R3_partial_entire_r1_family_up',family_costs['Partial'][1],0.)]
    thresholds=[]
    for name,pc,fc in exposure_specs:
        exposure=pc+fc
        def allowance(h):return (margin-h)/exposure if h<margin else None
        thresholds.append(dict(name=name,partial_exposure=pc,full_exposure=fc,adverse_D_slope=exposure,
            point_reversal_fraction=margin/exposure,at_threshold='tie;greater_fraction_reverses_to_Full',
            max_symmetric_fraction_for_source_sampling_diagnostic_separation=allowance(source_half),
            max_symmetric_fraction_for_independent_sampling_REFERENCE=allowance(1.96*independent_se),
            max_symmetric_fraction_for_unknown_PSD_sampling=allowance(1.96*hi_se),
            assumption='adverse_fraction_P_up_F_down_only_on_named_subset;all_other_prices_fixed;no_schedule_reoptimization'))
    shifts=[]
    for name,pc,fc in exposure_specs:
        for eps in (0.,.01,.025,.05,.25):
            shifted=d+eps*(pc+fc)
            shifts.append(dict(name=name,fraction=eps,D=shifted,point_preference='Partial' if shifted<0 else 'Full' if shifted>0 else 'TIE'))
    source_saved=a['coherent_reoptimization']['comparison']
    reproduction=dict(saved_G={'Partial':bests['3']['best']['total_compiled_rz_point_estimate'],
            'Full':bests['12']['best']['total_compiled_rz_point_estimate']},reproduced_G=candidate_costs,
        decimal_exact_from_saved_decimal_means=decimal_costs,D=d,positive_partial_point_margin=margin,
        ratio=gp/gf,max_unit_price_abs_error=max(x['unit_price_abs_error'] for x in agreements),
        max_unit_SE_abs_error=max(x['unit_SE_abs_error'] for x in agreements),
        source_comparison_rebuilt=reconstructed_comparison,original_comparison=source_saved,
        aggregate_group_costs={label:math.fsum(g['group_point_cost'] for g in groups.values() if g['candidate']==label) for label in ('Partial','Full')})
    a0_precision=[dict(q=v['q_m'],axis=v['axis'],policy=v['policy'],
        saved_prediction_calibration_relative_SE=v['prediction_standard_error']/v['prediction'],
        saved_direct_relative_SE=v['direct_standard_error']/v['direct_mean'],
        saved_absolute_relative_error=v['absolute_relative_error'])
        for v in inputs['A0']['fixed_holdout_rows'] if v['policy']=='support_run_le_1']
    result=dict(schema_version='PR-RPE-V3-R1-1',scope='fixed_18round_saved_compiled_RZ_no_state_preparation',
        reproduction=reproduction,shared_calibration_groups=list(groups.values()),sampling=sampling,
        anchor_vector=[dict(anchor_id=g['group_id']+f':q{i+1}',candidate=g['candidate'],q=i+1,
            D_weight=g['signed_D_anchor_weights'][i],mean=g['anchor_means'][i],SE=g['anchor_SE'][i],
            diagonal_plugin_variance=g['anchor_SE'][i]**2,
            off_diagonal_covariance='UNKNOWN_except_PSD_zero_variance_cross_terms')
            for g in groups.values() for i in (0,1)],
        dependence=dict(cosine_sine='paired_design_CONFIRMED;numeric_covariance_UNKNOWN',
            q1_q2='different_task_seeds_and_source_independence_assumption;numeric_covariance_UNKNOWN',
            different_r_families='distinct_saved_cohorts;numeric_covariance_UNKNOWN',
            candidate_cross_covariance='sampling_plugin_Full_zero_diagonal_forces_zero_PSD_cross_terms;physical_systematic_covariance_UNKNOWN',
            partial_full_basis_policy_pairing='same_trajectory_CONFIRMED;not_the_Full_LD12_candidate;not_extra_samples',
            raw_vectors='NOT_FOUND_IN_BOUNDED_LOCAL_LOOKUP',
            initial_fresh='fresh32_replaces_selected_failed_groups;not_pooled;no_duplicate_experiment_count'),
        cost_exposures=dict(by_candidate_r=family_costs,late_three=late,R3_q16_q32_partial=r3_cost,
            R3_share_of_partial=r3_cost/gp,R3_5pct_adverse_D_shift=.05*r3_cost,
            remaining_margin_after_R3_5pct=margin-.05*r3_cost),
        information_value=dict(A0_saved_precision=a0_precision,
            R2_intermediate_q_to_final_q_error_law='UNKNOWN;short_holdout_cannot_establish_uniform_late_band',
            R2_current_rank_discrimination='INSUFFICIENT_EVIDENCE;do_not_launch_as_rank_decider',
            R2_possible_separate_purpose='falsify_affine_transfer_at_named_q_if_separately_authorized',
            R3_recommendation='DO_NOT_EXECUTE_FOR_CURRENT_COST_RANK',
            covariance_vector_recovery='no_new_sampling;only_if_existing_trustworthy_vectors_become_available'),
        systematics=systematics,break_even_sensitivity=thresholds,illustrative_adverse_shifts=shifts,
        common_multiplicative_systematic=dict(formula='D_true=(1+e)*D_model',same_sign_for='e>-1',
            assumption='identical_fraction_both_candidates_all_rounds;not_established_by_shared_compiler'),
        unknown_long_q_systematic=dict(width='UNKNOWN',finite_empirical_upper_bound=None,
            verdict='INSUFFICIENT_EVIDENCE',overall_cost_superiority='UNDETERMINED',
            note='conditional finite-band separation is not unconditional superiority'),
        point_preference=dict(verdict='POINT_PREFERENCE',candidate='Partial',changed_from_saved=False),
        numerical_scope_limitations=['saved_SE_are_plugin_estimates','no_selection_bias_correction','no_interval_certification',
            'no_uniform_long_q_bound','no_preparation_backend_noise_generalization','V1_V2_conditions_unchanged'])
    return result,rows


def write_outputs(out,result,rows,index,source_records,elapsed):
    out.mkdir(parents=True,exist_ok=True)
    (out/'DEPENDENCE_AND_SENSITIVITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (out/'R1_FIXED_PLAN_RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    validation=dict(schema_version='PR-RPE-V3-R1-VALIDATION-1',fixed_inputs=index,reused_unchanged_definitions=source_records,
        executed_scope='authorized_saved_price_fixed_plan_reproduction_covariance_and_systematic_sensitivity',
        prohibited_calls=dict(new_trajectory_sampling=0,circuit_build_compile_transpile=0,H_state_operator_signal_generation=0,
            schedule_shot_alpha_beta_reoptimization=0,V1_V2_reruns=0,new_affine_fit=0),
        numerical_precision='Python_float64_math_fsum_hypot;integer_anchor_coefficients;Decimal_independent_point_sum',
        resources=dict(affinity_cpus=len(os.sched_getaffinity(0)),elapsed_seconds=elapsed,
            strategy='parallel_fixed_object_reads;O(72)_scalar_algebra;no_heavy_workload_or_artificial_resource_cap'),
        source_module_imports=0,tests='PENDING',preservation='PENDING_FINAL_OBSERVATION',
        old_M06F_stop_changed=False,mandatory_stop='after_publication_and_full_body_verification')
    (out/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();start=time.monotonic()
    inputs,index=read_inputs(args.repo);ns,records=reuse_source(inputs);result,rows=analyze(inputs,ns)
    write_outputs(args.output,result,rows,index,records,time.monotonic()-start)
    print(json.dumps(dict(rows=len(rows),groups=len(result['shared_calibration_groups']),
        D=result['reproduction']['D'],G=result['reproduction']['reproduced_G'],
        point_reproduction=result['reproduction']['max_unit_price_abs_error'],
        unknown_PSD=result['sampling']['unknown_covariance_PSD_diagonal_envelope'])))


if __name__=='__main__':main()
