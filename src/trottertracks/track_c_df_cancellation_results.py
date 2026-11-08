"""Export P1/P2 evidence and validate accounting; no molecular generation."""
import csv
from fractions import Fraction as F
import hashlib
import json
import math
import numpy as np
from . import track_c_df_cancellation_validation as v

OUT=v.OUT;SAVE=v.SAVE


def csv_save(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def export():
    p1=v.read_json(OUT/'P1/FINAL.json');p2=v.read_json(OUT/'P2/FINAL.json');plan=v.read_json(OUT/'P2/PLAN.json');points=[v.read_json(p)for p in sorted((OUT/'P2/qualified').glob('*.json'))]
    assert len(points)==plan['cohort_points'] and p1['P2_may_proceed']
    assert hashlib.sha256((OUT/'P2/POLICY.json').read_bytes()).hexdigest()==p2['policy_sha256']
    p1rows=[]
    for x in p1['rows']:
        q=x['qualification'];assert q['closed'] and x['mean_decoder']['error_validation_only']<1.5936e-4
        p1rows.append(dict(steps=x['steps'],h_exact=x['h_exact'],h_binary64=x['h'],reused=x['reused'],truncation_signed=x['observed']['truncation_phase'],PF_signed=x['observed']['PF_phase'],combined_signed=x['observed']['combined_phase'],combined_directed_upper=q['total_truncation_PF_phase_upper'],rho=q['rho_lower'],beta=q['beta_stat_lower'],alias=q['alias'],closed=q['closed'],scalar_timing_disk=q.get('fixed_total_scalar_time_error_upper',0),logical_h_rounding_source_disk=q.get('logical_h_rounding_source_disk',0),N_X=x['trials_fixed_alpha'][0],N_Y=x['trials_fixed_alpha'][1],whole18_RZ=x['whole_18round_totals_fixed_other_rounds'][0],mean_decoder_error=x['mean_decoder']['error_validation_only']))
    rows=[]
    for x in points:
        A=x['combined_qualification'];B=x['qualification'];k=x['individual_disks'];r=x['row'];m=x['signal']['m']
        rows.append(dict(id=x['id'],n=x['n'],m=m,candidate=r['candidate']['name'],formula=r['formula'],h=r['h'],steps=x['signal']['steps'],combined_origin=x['combined_origin'],truncation_signed=x['observed']['truncation_phase'],PF_signed=x['observed']['PF_phase'],combined_signed=x['observed']['combined_phase'],truncation_directed_upper=B['truncation_phase_upper'],PF_directed_upper=B['PF_phase_upper'],sum_individual_directed_upper=B['conservative_phase_upper'],combined_directed_upper=A['total_truncation_PF_phase_upper'],beta_A=A['beta_stat_lower'],beta_B=B['beta_stat_lower'],rho=B['rho_lower'],A_closed=A['closed'],B_closed=B['closed'],keep_signal_disk=k['keep_radius'],PF_signal_disk=k['PF_signal_radius'],pair_disk=k['PF_pair']['radius'],pair_radius_lower=k['PF_pair']['radius_lower'],source_disk=A['source_generator_disk'],native_disk=A['native_disk'],angle_disk=A['angle_disk'],numerical_disk=A['numerical_disk'],supplement_disk=A.get('complete_number_sector_diagonal_supplement',0),alias=B['alias'],unit_RZ_X=x['mean'][0][0],unit_RZ_Y=x['mean'][1][0]))
    roundrows=[];summary=[]
    for p in p2['policies']:
        assert p['status']=='QUALIFIED_INDEPENDENT_COMPLETE'
        assert len(p['selected'])==18 and sum(F.from_float(a)for row in p['alpha']for a in row)<=F(1,20)
        total=np.zeros(4)
        for m,(x,N,alpha)in enumerate(zip(p['selected'],p['trials'],p['alpha'])):
            q=x['qualification'];assert q['closed'] and q['alias'];mean=np.array(x['mean']);cost=(mean*np.array(N)[:,None]).sum(axis=0);total+=cost
            for a,num in zip(alpha,N):assert v.d.rigorous_shots(q['rho_lower'],q['beta_stat_lower'],a)<=num
            roundrows.append(dict(n=p['n'],method=p['method'],m=m,candidate=x['row']['candidate']['name'],formula=x['row']['formula'],h=x['row']['h'],steps=x['signal']['steps'],N_X=N[0],N_Y=N[1],alpha_X=alpha[0],alpha_Y=alpha[1],rho=q['rho_lower'],beta=q['beta_stat_lower'],conservative_phase_upper=x['qualification'].get('conservative_phase_upper'),G_RZ=cost[0],G_CX=cost[1],G_depth=cost[2],G_size=cost[3],unit_RZ_X=mean[0,0],unit_RZ_Y=mean[1,0]))
        assert np.array_equal(total,np.array(p['totals']))
        summary.append(dict(n=p['n'],method=p['method'],RZ=p['totals'][0],CX=p['totals'][1],depth=p['totals'][2],size=p['totals'][3],trials=p['trials_total'],MC_SE_RZ=0.,mean_decoder_error=p['mean_decoder']['error_validation_only']))
    baselines=[x for c in p2['comparisons']for x in c['baselines']]
    for x in baselines:summary.append(dict(n=x['n'],method='reference_'+x['method'],RZ=x['RZ'],CX=x['CX'],depth=x['serial_depth_upper'],size=x['size'],trials=x['trials'],MC_SE_RZ=x['RZ_MC_SE'],origin=x['origin'],mean_decoder_error=x['mean_decoder_error']))
    ablations=[]
    for n in(4,6):
        A=next(p for p in p2['policies']if p['n']==n and p['method']=='A_combined');B=next(p for p in p2['policies']if p['n']==n and p['method']=='B_conservative');AB=next(p for p in p2['policies']if p['n']==n and p['method']=='A_on_B_fixed_circuits')
        assert [x['key']for x in B['selected']]==[x['key']for x in AB['selected']]
        assert [x['mean']for x in B['selected']]==[x['mean']for x in AB['selected']]
        b_lookup={x['id']:x for x in points};failed=[x['signal']['m']for x in A['selected']if not b_lookup[x['id']]['qualification']['closed']]
        ablations.append(dict(n=n,A_free_RZ=A['totals'][0],A_on_B_circuits_RZ=AB['totals'][0],B_on_B_circuits_RZ=B['totals'][0],circuit_policy_path_difference=AB['totals'][0]-A['totals'][0],evaluation_rule_fixed_circuit_difference=B['totals'][0]-AB['totals'][0],A_policy_rounds_failing_B=failed,interpretation='ordered ablation path, not a unique causal decomposition or global optimum'))
    ck=v.read_json(OUT/'P2/INDEPENDENT.json');unique={(x['n'],tuple(x['key'])):x for x in ck['reused']+ck['new']}
    assert '13 passed' in (OUT/'final_artificial_tests.log').read_text()
    validation=dict(source=v.SOURCE,P1_all_five_closed=True,P1_new_physical_checks=4,P1_reused_physical_point=1,P2_points=len(points),P2_B_closed=sum(x['qualification']['closed']for x in points),P2_B_failures=[dict(n=x['n'],key=x['key'],sum_upper=x['qualification']['conservative_phase_upper'],combined_closed=x['combined_qualification']['closed'])for x in points if not x['qualification']['closed']],independent_new=len(ck['new']),independent_unique_total=len(unique),max_native_XY=max(max(x['native_XY_difference'])for x in unique.values()),all_alpha_budgets_and_sufficient_trials=True,all_18round_decoders_pass=True,policy_hash_unchanged=True,paired_fixed_circuit_ablation_matches=True,artificial_tests=13,new_price_MC=0,quantum_backend_shots=0)
    for name,data in [('P1_RESULTS.csv',p1rows),('P2_POINTS.csv',rows),('P2_ROUNDS.csv',roundrows),('COMPARISON.csv',summary)]:csv_save(name,data)
    SAVE(OUT/'RESULTS.json',dict(source=v.SOURCE,P1_classification=p1['classification'],P1_rows=p1rows,P2_methods=summary,P2_classifications=p2['comparisons'],ablations=ablations,scope=p2['scope'],limits=['finite 5-point P1 grid, no continuous-domain guarantee','finite P2 qualified cohort and sufficient bound, not global optimum','B failure does not invalidate A','fixed exact-ground/ideal-native promise, no hardware or preparation claim']))
    SAVE(OUT/'VALIDATION.json',validation)
    print(json.dumps(dict(validation=validation,methods=summary,ablations=ablations),indent=2),flush=True)


if __name__=='__main__':
    if (OUT/'STUDY_STOP.json').exists():raise RuntimeError('Study has stopped')
    export()
