"""Existing JSON/CSV metadata -> manuscript tables and UNAPPROVED H8 budgets.

No scientific arrays, random draws, scientific imports, circuit build or runner.
Run from any directory: python3 -B /absolute/path/build_evidence.py
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RA = ROOT / 'artifacts/resource_applicability'
BASE = 'e0e6473d7047158a7f0f788d026a7a89a01898b8'
INPUTS = [
 'track_a_h6_exploration_summary_v1/2026-10-12/result_summary_v1.json',
 'track_a_h6_exploration_summary_v1/2026-10-12/wrapper_metrics_v1.csv',
 'track_a_h6_confirmation_summary_v1/2026-10-12/result_summary_v1.json',
 'track_a_h6_rare_order_cost_analysis_v1/2026-10-13/cost_reliability_analysis_v1.json',
 'track_a_h6_rare_order_cost_analysis_v1/2026-10-13/selected_trajectory_cost_evidence_v1.json',
 'track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/cost_contribution_analysis_v1.json',
 'track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/result_audit_v1.json',
 'track_a_h6_server_preparation_v2/2026-10-12/environment_compiler_profile_v2.json',
]

def read(name):
    return json.loads((RA/name).read_text())

def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def table(name, rows):
    with (HERE/name).open('w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        out.writeheader(); out.writerows(rows)

def prefixes(rank):
    if type(rank) is not int or rank < 2:
        raise ValueError('rank must be an integer >= 2')
    return sorted({math.ceil(rank*k/4) for k in (1,2,3)} - {rank})

def budget(qs=(1,2,4,8), prefix_count=3):
    """Conservative all-eligible upper reservations; no cache savings assumed."""
    b0 = prefix_count*len(qs)
    b1 = 2*len(qs)
    b2 = prefix_count*len(qs)*3*2
    deterministic = b0+b1
    explore_pr = 8*b2
    confirm = 8*32
    tail = 8*4+4*2  # optional preregistered cost diagnostics; not accuracy samples
    return {'q': list(qs), 'prefix_count_scenario': prefix_count,
            'signal_candidates': b0+b1+b2, 'B0':b0,'B1':b1,'B2':b2,
            'exploration_groups_upper': deterministic+explore_pr,
            'exploration_random_trajectories': explore_pr,
            'confirmation_trajectories_upper':confirm,
            'optional_tail_trajectories_upper':tail,
            'new_trajectories_upper':explore_pr+confirm+tail,
            'primary_compile_upper':2*(deterministic+explore_pr+confirm+tail),
            'ordinary_control_anchor_compile_upper':4,
            'new_compile_upper':2*(deterministic+explore_pr+confirm+tail)+4,
            'random_occurrences_upper':8*prefix_count*3*2*sum(qs)+(confirm+tail)*max(qs),
            'primitive_calls':None,'reference_calls':None,'normalization_diagnostics':None,
            'preflight_extra_calls':None,'cache_savings_reserved':0,
            'includes_input_generation':False,'includes_H4_supplement':False}

def permitted_mix(heavy, normal, aggregate=76, normal_rss=9, heavy_rss=24):
    # Scalar admission fixture, not a runtime limiter or grant.
    return (type(heavy) is int and type(normal) is int and min(heavy, normal)>=0
            and heavy+normal<=8 and 4+normal*normal_rss+heavy*heavy_rss<=aggregate)

def build():
    exp=read(INPUTS[0]); conf=read(INPUTS[2]); rare=read(INPUTS[3]); pilot=read(INPUTS[5])
    before=[{'path':str((RA/p).relative_to(ROOT)), 'bytes':(RA/p).stat().st_size,
             'sha256':hashlib.sha256((RA/p).read_bytes()).hexdigest()} for p in INPUTS]
    review=ROOT/'docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md'
    before.append({'path':str(review.relative_to(ROOT)), 'bytes':review.stat().st_size,
                   'sha256':hashlib.sha256(review.read_bytes()).hexdigest(),
                   'role':'user-supplied completed GPT review, adopted, not repeated'})
    rows=[]; panel=[]
    for a in conf['comparisons']:
        if a['allowance_factor'] != '1.0': continue
        eps=a['epsilon']; e=next(x for x in exp['comparisons'] if x['epsilon']==eps and x['allowance_factor']=='1.0')
        fixed_id=a['exploration_best_B2']['cell_id']
        fixed=next(x for x in a['B2_top2'] if x['cell_id']==fixed_id)
        post=a['family_best_point']['B2']; pf=a['family_best_point']['B1']; b0=a['family_best_point']['B0']
        r=next(x for x in rare['comparison'] if x['epsilon']==eps)
        assert math.isclose(fixed['G_point']/pf['G_point'], r['fixed_winner_PR_PF_ratio'],rel_tol=1e-14)
        rows.append({'epsilon':eps,'PF_cell':pf['cell_id'],'PF_G':pf['G_point'], 'PF_shots':pf['N_total'],
            'exploration_K4_cell':fixed_id,'exploration_PR_PF_ratio':e['PR_to_baseline_point_ratio']['B1'],
            'fixed_K4_confirmation_G':fixed['G_point'],'fixed_K4_confirmation_SE':fixed['G_standard_error'],
            'fixed_K4_confirmation_shots':fixed['N_total'],'fixed_K4_confirmation_ratio':fixed['G_point']/pf['G_point'],
            'fixed_K4_reduction_percent':100*(1-fixed['G_point']/pf['G_point']),
            'postconfirmation_cell':post['cell_id'],'postconfirmation_G':post['G_point'],
            'postconfirmation_SE':post['G_standard_error'],'postconfirmation_shots':post['N_total'],
            'postconfirmation_ratio':post['G_point']/pf['G_point'],
            'postconfirmation_reduction_percent':100*(1-post['G_point']/pf['G_point']),
            'postconfirmation_B0_ratio':post['G_point']/b0['G_point'],
            'population_mean_certified':False})
        for p in a['B2_top2']:
            panel.append({'epsilon':eps,'cell_id':p['cell_id'],'G':p['G_point'],
                'SE':p['G_standard_error'],'N_total':p['N_total'],'PF_G':pf['G_point'],
                'ratio':p['G_point']/pf['G_point'],'ratio_SE':p['G_standard_error']/pf['G_point'],
                'sample_count':32,'preselected_exploration_winner':p['cell_id']==fixed_id,
                'postconfirmation_point_minimum':p['cell_id']==post['cell_id']})
    table('h6_precision_results.csv',rows);table('h6_fixed_panel.csv',panel)
    tail=[]
    for p in pilot['comparisons']:
        tail.append({k:p[k] for k in ['epsilon','PF_cell_id','PF_G','PR_PF_ratio_before',
            'PR_PF_ratio_stratified','relative_G_change','stratified_G_point_estimate',
            'stratified_G_standard_error','target_is_prior_postconfirmation_point_winner']})
    table('h6_pilot_effect.csv',tail)
    mu0=pilot['comparisons'][-1]['s0_confirmation']['mean']/sum(pilot['comparisons'][-1]['saved_shot_counts'].values())
    # Paired shot-weighted cost gives the same RZ/shot on both saved axes.
    mu2=pilot['comparisons'][-1]['s2_pilot']['mean']/sum(pilot['comparisons'][-1]['saved_shot_counts'].values())
    table('h6_pilot_strata.csv',[
        {'stratum':'s0 confirmation','probability':1-pilot['s2_probability'],'n':32,'RZ_mean':mu0,
         'RZ_SE':pilot['comparisons'][-1]['s0_confirmation']['standard_error']/sum(pilot['comparisons'][-1]['saved_shot_counts'].values())},
        {'stratum':'s2 conditional pilot','probability':pilot['s2_probability'],'n':8,'RZ_mean':mu2,
         'RZ_SE':pilot['comparisons'][-1]['s2_pilot']['standard_error']/sum(pilot['comparisons'][-1]['saved_shot_counts'].values())}])
    wrappers=list(csv.DictReader((RA/INPUTS[1]).open()))
    records=read(INPUTS[4])['records']
    cw=[x['classical_wall_seconds'] for x in records if x['phase']=='confirmation']
    exp_sum=sum(float(x['classical_wall_seconds']) for x in wrappers)
    work=exp_sum+sum(cw)
    times={'kind':'H6 saved metadata anchors, NOT measured H8 speed or elapsed campaign sum',
      'exploration_sum_wrapper_classical_fields_seconds':exp_sum,
      'confirmation_group_wall_sum_seconds':sum(cw),'confirmation_group_count':len(cw),
      'confirmation_group_median_seconds':statistics.median(cw),'confirmation_group_max_seconds':max(cw),
      'H6_S4_q8_peak_worker_RSS_bytes':max(int(x['worker_peak_rss_bytes']) for x in wrappers if x['cell_id']=='B1_p19_q8_4th'),
      'H6_S4_q8_sum_axis_classical_fields_seconds':sum(float(x['classical_wall_seconds']) for x in wrappers if x['cell_id']=='B1_p19_q8_4th'),
      'assumed_H8_cost_multipliers':[{ 'factor_assumption':f,'worker_hours_anchor_scaled':work*f/3600,
         'eight_worker_ideal_hours_anchor_scaled':work*f/(8*3600)} for f in (4,8,16)],
      'signal_and_reference_time_unmeasured':True,'includes_tail_pilot':False,
      'feasibility_needed_before_elapsed_estimate':True,'wall_limit_proposal_seconds':None}
    write('classical_planning_anchors.json',times)
    scenarios={str(L):{'actual_rank':L,'prefixes':prefixes(L)} for L in (19,24,32,48)}
    memory={'sector_dimension':math.comb(8,4)**2,'full_state_complex128_bytes':2**16*16,
      'ancilla_state_complex128_bytes':2**17*16,'one_dense_sector_matrix_bytes':4900**2*16,
      'dense_full_system_unitary_bytes':(2**16)**2*16,
      'dense_sector_storage_ratio_H8_H6':4900**2/400**2,
      'dense_cubic_operation_ratio_H8_H6':4900**3/400**3,
      'H8_peak_memory_measured':False}
    plan={'schema':'h8_experiment_design_draft_v1','base_commit':BASE,
      'approved':False,'runtime_authorization':False,'grant':None,'allowed_cpus':[],
      'technical_sealed':False,'absolute_launch_command':None,'new_science_executed':False,
      'target_proposal':{'model':'linear H8','geometry_angstrom':1.0,'basis':'sto-3g',
        'system_modes':16,'nelec_alpha':4,'nelec_beta':4,'T':0.8,
        'DF_policy':'tol-only 1e-8, cutoff 0, signed generation order, Hermitian projection',
        'actual_DF_rank':None,'Hamiltonian_SHA':None,'state_SHA':None,
        'epsilon':[0.05,0.01,0.005,0.001]},
      'candidate_policy_proposal':{'prefix_rule':'unique ceil(L*k/4), k=1,2,3; 0<p<L',
        'rank_scenarios':scenarios,'q_core':[1,2,4,8],'q_boundary_option':[1,2,4,8,16],
        'r':[1,2,4],'K':[2,4],'B0':'discard S2','B1':'full S2 and global Yoshida S4',
        'B2':'partial S2 with original finite RTE distribution',
        'transfer_panel':'separate absolute p5 H6-preselected eight settings; unavailable if L<=5',
        'explore_samples':8,'selection':'B2 top2 per epsilon union max8; deterministic tie policy before results',
        'confirm_samples':32,'tail_option':'s2 four each max8; s4 two each max4; diagnostics, not guaranteed tail bound'},
      'core_budget_upper':budget(),'q16_option_budget_upper':budget((1,2,4,8,16)),
      'prior_carry':{'known_compile':1716,'reserved_compile':20,'known_trajectory':842,
         'reserved_trajectory':10,'current_compile_cap':1736,'current_trajectory_cap':852,
         'known_control':360,'reserved_control':4,'known_occurrences':3200,'reserved_occurrences':64,
         'known_primitive':1470,'known_reference':400,'historical_unknown_actual':None,
         'refund':0,'reset':False},
      'memory_arithmetic':memory,
      'resource_proposal':{'workers_max':8,'worker_AS_GiB':32,'normal_worker_RSS_GiB':9,
        'heavy_worker_RSS_GiB':24,'coordinator_RSS_GiB':2,'coord_margin_GiB':2,
        'aggregate_RSS_GiB':76,'job_memory_admission_GiB':96,'internal_threads':1,
        'coordinator_AS_GiB_proposal':8,'inherited_AS_hard_minimum_GiB':32,
        'CPU_candidates_worker':[20,21,24,25,26,27,28,29],'CPU_candidates_coordinator':[22,23],
        'mix_examples':[{'heavy':h,'normal':n,'admissible_scalar_fixture':permitted_mix(h,n)}
                         for h,n in [(0,8),(1,5),(1,6),(3,0),(4,0),(8,0)]],
        'first_heavy_case_workers':1,'actual_H8_limits_approved':False,
        'fresh_available_host_GiB_proposal':248.25,'running_available_host_GiB_proposal':168.25,
        'fresh_effective_available_GiB_proposal':96,'running_effective_available_GiB_proposal':16,
        'pressure_OOM_stop_proposal':'host full PSI >=5%, cgroup full PSI >0 or new OOM; other host PSI policy requires explicit approval',
        'monitor_seconds_max':5,'output_cap_GiB_proposal':32,'storage_admission_GiB_proposal':64,
        'inodes_admission_proposal':100000,'wall_cap_seconds_proposal':None,
        'block_inode_quota_required':True,'bounded_journal_rotation_required':True},
      'optimizations_before_launch_required':['canonical hash reuse with source/input/compiler binding',
        'immutable preparation/primitive caches with byte-identity keys and bounded RSS',
        'independent groups scheduled across up to eight workers, internal thread1',
        'measured weighted admission; single heavy feasibility case first',
        'complete-only checkpoint/resume receipts preserving seeds, carry and separate grants'],
      'technical_dependencies':['new H8 input/state receipt', 'sector400 guards cannot run H8',
        'matrix-free sector/reference strategy plus registered numerical validation',
        'H8 native builder/serialization/source closure', 'profile/compiler binding',
        'measured heavy memory/worker admission','bounded logs and storage/quota gate'],
      'scientific_decisions_pending':['resource-map vs transfer primary estimand', 'prefix/q boundary selection',
         'matrix-free numerical allowance/reference validation','tail diagnostic vs estimator scope',
         'H4 common panel scope and DF policy'],
      'user_start_permission':{'received':True,'condition':'preparation complete',
        'execution_target':['H8 small preflight','H4 comparison supplement'],'H8_full_campaign_authorized':False,'scientific_contract_adopted':False},
      'mandatory_stop':True,'next_stage_authorized':False}
    plan['projected_compile_total_upper_core']=1716+20+plan['core_budget_upper']['new_compile_upper']
    plan['projected_trajectory_total_upper_core']=842+10+plan['core_budget_upper']['new_trajectories_upper']
    write('h8_plan_draft.json',plan)
    write('input_metadata_receipt.json',{'base_commit':BASE,'inputs':before,
       'scientific_arrays_read':0,'new_trajectories':0,'new_compiles':0,'new_grants':0,
       'old_source_modified':False,'H4_interference':False})
    print(json.dumps({'precision_rows':len(rows),'panel_rows':len(panel),
          'core_compile':plan['core_budget_upper']['new_compile_upper'],
          'core_trajectory':plan['core_budget_upper']['new_trajectories_upper'],
          'new_science':0}))

if __name__=='__main__': build()
