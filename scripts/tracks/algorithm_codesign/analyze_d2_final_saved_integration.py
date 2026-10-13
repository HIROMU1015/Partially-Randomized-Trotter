#!/usr/bin/env python3
"""Final D1/D2 affine/depth analysis from committed scalars only.

No event JSON, boundary kernel, compiler, matrix, synthesis, sampler, or solver
is imported/called. Exact half-lines are algebraic price sensitivity, not runs.
"""
from fractions import Fraction as F
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import resource
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
BASE='d1c59b28f4370cc7be65cff60d1ec877fd1491f5'
D1='fc96571881f301435a655f057709df4d6c1678e0'
D1DIR=Path('artifacts/track_b_rte_development_d1_result/2026-10-13/v1')
D2DIR=Path('artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1')
OUT=Path('artifacts/track_b_d2_final_saved_integration/2026-10-13/v1')
SPEC=importlib.util.spec_from_file_location('fixed_d1_affine_math',ROOT/'scripts/tracks/algorithm_codesign/analyze_d1_saved_resources.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def digest(path):
    data=Path(path).read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def fraction_json(v):
    if isinstance(v,F):return str(v)
    raise TypeError('unsupported saved-analysis evidence')


class Output:
    def __init__(self,d):self.root=Path(d);self.root.mkdir(parents=True,exist_ok=False);self.files={}
    def write(self,relative,v):
        path=self.root/relative;data=(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,default=fraction_json)+'\n').encode()
        if len(data)>524288:raise RuntimeError('review file >512 KiB')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as s:s.write(data)
        self.files[relative]=digest(path)


def affine_row(steps,arm,prices,slope,workspace=1):
    return {'steps':steps,'arm':arm,'expected_native_intercept':{k:F(prices[k]) for k in M.METRICS},
            'prep_readout_slope':F(slope),'workspace_beyond_system':workspace}


def envelopes(rows,metric):
    """Reuse fixed tested scalar arithmetic, clarify origin vs actual roots."""
    v=M.sensitivity(rows,metric)
    v['domain_breakpoints_including_origin']=v.pop('all_exact_nonnegative_crossings')
    v['actual_pair_nonnegative_crossings']=sorted({a['crossover'] for a in v['pairs'] if a['crossover'] is not None})
    segments=[]
    for row in v['open_interval_rankings']:
        arms=row['minimum_envelope_arms'];lo=row['open_lo'];hi=row['open_hi']
        if segments and segments[-1]['minimum_arms']==arms:
            segments[-1]['open_hi']=hi
        else:segments.append(dict(open_lo=lo,open_hi=hi,minimum_arms=arms))
    # A lower-envelope switch has a tied exact endpoint. Unrelated crossings
    # inside a merged segment do not switch its minimum.
    v['merged_minimum_envelope_open_regions']=segments
    v['minimum_switch_endpoint_ties']=[dict(h=s['open_lo'],minimum_arms=M.ranking(rows,metric,s['open_lo'])[0]) for s in segments[1:]]
    v['h0_minimum_arms']=M.ranking(rows,metric,F(0))[0]
    return v


def dominates_at(a,b,h):
    av=[a['expected_native_intercept'][k]+a['prep_readout_slope']*F(h[k]) for k in M.METRICS]+[F(a['workspace_beyond_system'])]
    bv=[b['expected_native_intercept'][k]+b['prep_readout_slope']*F(h[k]) for k in M.METRICS]+[F(b['workspace_beyond_system'])]
    return all(x<=y for x,y in zip(av,bv)) and any(x<y for x,y in zip(av,bv))


def structure_summary(row,all_rotation_prices_positive):
    pm={int(k):F(v) for k,v in row['depth_probability_unconditional'].items()}
    Z2=F(row['acceptance_squared'])
    if sum(pm.values(),F(0))!=Z2 or not 0<Z2<=1 or any(k<0 or p<0 for k,p in pm.items()):
        raise ArithmeticError('saved depth probability/acceptance inconsistency')
    saved={k:F(row['cancellation_saving_per_whole_trial'][k]) for k in M.METRICS}
    # Derived only from depth: k inverse pairs remove exactly 2k IR tokens.
    tokens=2*sum((k*p for k,p in pm.items()),F(0))
    v={'arm':row['arm'],'depth_atoms':[dict(depth=k,unconditional_probability=p,conditional_on_nonzero=p/Z2) for k,p in sorted(pm.items())],
       'acceptance_squared':Z2,'zero_trial_mass':1-Z2,
       'cancellation_probability_unconditional':sum(p for k,p in pm.items() if k>0),
       'cancellation_probability_given_nonzero':sum(p for k,p in pm.items() if k>0)/Z2,
       'expected_removed_IR_tokens_unconditional':tokens,
       'expected_removed_IR_tokens_given_nonzero':tokens/Z2,
       'max_observed_positive_depth':max(k for k,p in pm.items() if p),
       'saved_savings_per_whole_trial':saved,
       'joint_depth_removed_gate_cost_table_recorded':False,
       'pair_identity_or_removed_gate_trace_recorded':False,
       'pair_re_evaluation':0,'unique_causal_or_gate_attribution':False}
    if row['arm']!='matched_CTS':
        # Static g9_native.py: V1 has 7 tokens (T1/CX2/1Q5), V2 10
        # (T2/CX4/1Q6). This is a consistency model, not a new gate trace.
        predicted={'T':2*pm.get(7,0)+4*pm.get(10,0),
                   'CX':4*pm.get(7,0)+8*pm.get(10,0),
                   '1Q':2*pm.get(1,0)+10*pm.get(7,0)+12*pm.get(10,0)}
        v.update(provider_template_consistency=dict(assumed_inverse_pair_costs={1:{'T':0,'CX':0,'1Q':2},7:{'T':2,'CX':4,'1Q':10},10:{'T':4,'CX':8,'1Q':12}},
                static_source='g9_native.py: V/exact_rotation/basis/parity/native_ir',
                predicted_aggregate_savings=predicted,equals_saved_all_coordinates=predicted==saved,
                establishes_per_depth_measured_prices=False,
                alternative_correlated_cost_assignments_not_excluded_by_aggregate_data=True),
                basis_cancellation_explanation='consistent with shared V1/V2 inverse at adjacent step boundary; depth1 removes one self-inverse Clifford pair',
                inverse_count_diagnostics_match_total_priced_IR_tokens=(saved['CX']+saved['1Q']==tokens))
    else:
        v.update(T_saving_is_exact_zero=saved['T']==0,
                 no_T_priced_R_removed_follows_nonnegative_cost=(saved['T']==0 and all_rotation_prices_positive),
                 no_T_in_literal_CTS_source=True,
                 basis_cancellation_explanation='saved depth1/2/4/6 with zero T savings; Clifford lowering simplification is compatible; detailed gate costs by depth not identified')
    return v


def main():
    wall,cpu=time.monotonic(),time.process_time();out=Output(ROOT/OUT);inputs={}
    def read(relative,commit=BASE):
        relative=Path(relative)
        if '/events/' in str(relative) or relative.suffix.lower()=='.npz':raise PermissionError('event/NPZ IO outside saved scalar scope')
        data=(ROOT/relative).read_bytes();frozen=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+str(relative)])
        if data!=frozen:raise PermissionError('saved input changed: '+str(relative))
        inputs[str(relative)]={'commit':commit,**digest(ROOT/relative)}
        return json.loads(data)
    terminal=read(D2DIR/'terminal.json');audit=read(D2DIR/'saved_output_audit/saved_output_audit_v1.json')
    if terminal['status']!='D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW' or audit['status']!='PASS':raise ArithmeticError('complete fixed evidence required')
    read(D2DIR/'one_shot_consumed.json');read(D2DIR/'STOP.json');read(D2DIR/'outer_process_receipt.json')
    dc=read('artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json',D1)
    cache=read(D1DIR/'synthesis_cache.json',D1)
    r1=[];old_r2=[];new_r2=[];metadata=[];structures=[]
    for i in range(12):
        d=read(D1DIR/f'rows/row_{i:02d}.json',D1)
        row=affine_row(d['steps'],d['arm'],d['expected_native_intercept'],d['prep_readout_slope'],d['workspace_beyond_system'])
        if d['steps']==1:r1.append(row)
        else:
            old_r2.append(row);new=read(D2DIR/f'rows/row_{i:02d}.json')
            if new['budget_unchanged']!=d['budget'] or new['prep_readout_slope_unchanged']!=d['prep_readout_slope']:
                raise ArithmeticError('budget/slope changed')
            new_r2.append(affine_row(2,new['arm'],new['new_expected_native_intercept'],new['prep_readout_slope_unchanged'],new['workspace_unchanged']))
            structures.append(structure_summary(new,all(v['T_count']>0 for v in cache.values())))
        metadata.append(dict(label=M.label(row),degree=d['degree'],x=d['x'],N_per_axis=d['budget']['N_per_axis'],
                step_acceptance=d['reference_law_diagnostic_only']['acceptance'],budget=d['budget'],
                ideal_target='P5(-i tau R/r)^r; r-specific finite approximation',
                step_native_prices='old D1 per-step law; r2 boundary cost must use D2 whole-trial value'))
    out.write('saved_affine_rows_and_budget_metadata_v1.json',dict(D1_r1=r1,D1_r2_additive=old_r2,D2_r2_boundary=new_r2,metadata=metadata,
              comparison_rule='one exact adjacent inverse pass on full trial; r1 has no step boundary; r2 uses saved D2',
              physical_task=dc['input'],fixed_accuracy_confidence={k:dc[k] for k in ('epsilon_axis','complex_accuracy_upper','alpha_axis','resource_failure_per_row','familywise_alpha')},
              no_new_N_or_law=True))
    groups={'D2_r2':new_r2,'D1r1_D2r2_joint12':r1+new_r2}
    display={'groups':{},'same_arm_repeat_selection':[],'structure':[]}
    envelopes_by={};pareto_by={}
    for group,rows in groups.items():
        envelopes_by[group]={}
        region=M.pareto_regions(rows);pareto_by[group]=region
        out.write('pareto_'+group+'_v1.json',region)
        display['groups'][group]={'quantum_Pareto_h0':region['h0_quantum_front'],'minimum_envelopes':{}}
        for k in M.METRICS:
            v=envelopes(rows,k);envelopes_by[group][k]=v
            out.write('sensitivity_'+group+'_'+k+'_v1.json',v)
            display['groups'][group]['minimum_envelopes'][k]=[dict(open_lo=float(s['open_lo']),open_hi=float(s['open_hi']) if s['open_hi'] is not None else None,minimum_arms=s['minimum_arms']) for s in v['merged_minimum_envelope_open_regions']]
    # Only envelope changes against the already acquired additive baseline.
    updates=[]
    for group,old,new in [('six_r2',old_r2,new_r2),('joint12',r1+old_r2,r1+new_r2)]:
        for k in M.METRICS:
            a,b=envelopes(old,k),envelopes(new,k)
            updates.append(dict(group=group,metric=k,before_minimum_regions=a['merged_minimum_envelope_open_regions'],
                  after_minimum_regions=b['merged_minimum_envelope_open_regions'],
                  before_switch_ties=a['minimum_switch_endpoint_ties'],after_switch_ties=b['minimum_switch_endpoint_ties'],
                  changed=a['merged_minimum_envelope_open_regions']!=b['merged_minimum_envelope_open_regions']))
    out.write('minimum_envelope_updates_v1.json',updates)
    contrasts=[]
    for before,after,old2 in zip(r1,new_r2,old_r2):
        if before['arm']!=after['arm']:raise ArithmeticError('arm ordering')
        for k in M.METRICS:
            v=M.pair_relation(before,after,k);oldv=M.pair_relation(before,old2,k)
            contrasts.append(dict(arm=before['arm'],metric=k,r1_vs_D2r2=v,r1_vs_old_D1r2=oldv,
                     r1_intercept=before['expected_native_intercept'][k],D2r2_intercept=after['expected_native_intercept'][k],
                     r2_over_r1_h0=after['expected_native_intercept'][k]/before['expected_native_intercept'][k],
                     factor_causality_not_identified=True))
            display['same_arm_repeat_selection'].append(dict(arm=before['arm'],metric=k,old_crossing=float(oldv['crossover']) if oldv['crossover'] is not None else None,
                   new_crossing=float(v['crossover']) if v['crossover'] is not None else None,r2_over_r1_h0=float(contrasts[-1]['r2_over_r1_h0']),r2_nonworse_region={name:float(a) if isinstance(a,F) else a for name,a in v['alternative_nonworse_region'].items()}))
    out.write('same_arm_r1_vs_r2_all18_relations_v1.json',contrasts)
    for v in structures:
        out.write('structure/'+v['arm']+'_v1.json',v)
        display['structure'].append(dict(arm=v['arm'],nonzero_probability=float(v['acceptance_squared']),
          cancellation_probability_conditional=float(v['cancellation_probability_given_nonzero']),
          depth_probabilities_conditional={str(a['depth']):float(a['conditional_on_nonzero']) for a in v['depth_atoms']},
          mean_removed_tokens_conditional=float(v['expected_removed_IR_tokens_given_nonzero']),
          saved_savings_per_whole_trial={k:float(x) for k,x in v['saved_savings_per_whole_trial'].items()},
          provider_template_consistency=v.get('provider_template_consistency',{}).get('equals_saved_all_coordinates')))
    out.write('attribution_limits_and_remaining_unknowns_v1.json',dict(
         saved_depth_probability_and_aggregate_savings_complete=True,
         missing=['pair indices/removed gate traces','measured joint depth and T/CX/1Q reduction table','per-key savings by boundary depth','online classical sampler timing','whole-circuit compiler result','independent provider/precision validation'],
         static_provider_template='seven-/ten-token V1/V2 inverse pairs; matches saved aggregate savings; consistency, not uniquely identified depth costs',
         CTS_zero_T='with strictly positive R sequence T prices, nonnegative zero T reduction implies no T-priced R cancellation; gate-class allocation across depths not known',
         scientific_or_causal_attribution=False,pair_rerun=0,
         future_science_GPT_scope_decision_required=True,mandatory_STOP=True))
    out.write('display_summary_v1.json',display)
    out.write('input_identity_v1.json',dict(files=inputs,fixed_affine_math={'path':str(Path(M.__file__).relative_to(ROOT)),**digest(M.__file__)},
            event_IR_files_loaded=0,boundary_kernel_calls=0,scientific_classification_unchanged=True))
    out.write('analysis_manifest_v1.json',dict(status='D2_FINAL_SAVED_INTEGRATION_COMPLETE',input_D2_result_commit=BASE,input_D1_result_commit=D1,
          files=dict(out.files),new_science=0,pair_re_evaluation=0,event_IR_loaded=0,
          compile_synthesis_matrix_sampling_trajectory_LP_DF_NPZ_GPU=0,
          expected_intercepts_N_and_old_classification_unchanged=True,
          groups=2,within_six_pair_metrics=45,joint12_pair_metrics=198,same_arm_r_relations=18,
          scalar_rows_shared=18,numeric_workers=0,
          performance=dict(wall_seconds=time.monotonic()-wall,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
          scientific_method_adoption=False,mandatory_STOP=True,next_science_authorized=False))
    print(json.dumps({'status':'D2_FINAL_SAVED_INTEGRATION_COMPLETE','h0_fronts':{k:v['h0_quantum_front'] for k,v in pareto_by.items()},'minimum_envelopes':{g:display['groups'][g]['minimum_envelopes'] for g in groups}}))


if __name__=='__main__':main()
