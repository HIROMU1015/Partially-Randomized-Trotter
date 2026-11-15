"""Project the completed one-call receipt; no signal/risk/price evaluations."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import run_p7_y4_coarse_q3_recovery as R

HERE=Path(__file__).resolve().parent
RAW='537df2376'  # Immutable completed API/partials/resource publication.
CURRENT='b4296b110b129d226dbef2743c14f725a27c3bf6'
CURRENT_DIR='docs/research/pf6_cost_budget_q3_results_20261115/'


def main():
    F=R.O.q3_runtime();raw=R.O.blob
    publication=subprocess.check_output(['git','rev-parse',RAW],cwd=ROOT,text=True).strip()
    read=lambda path:json.loads((HERE/path).read_text())
    inputs=read('INPUTS.json');case=inputs['inputs']['cases'][0]
    stem='run01/'+case['label']+'/'
    result=read(stem+'RESULT.json');api=read(stem+'API_RETURN.json');completion=read('run01/COMPLETION.json')
    risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp']
    partial=read(stem+'PARTIAL_COARSE.json');usage=read(stem+'RESOURCE_FINAL.json')
    old= json.loads(raw(CURRENT,CURRENT_DIR+'run01/B0I_P7_Y4_16h_over_3__B_PRICE/RESULT.json'))
    old_risk=old['record']['diagnostic']['deterministic_risk']
    assert api==[result['record']] and result['scientific_evaluation_calls']==1
    assert completion['status']=='ONE_FIXED_Q3_SAVED' and not completion['errors'] and not completion['monitor_errors']
    assert completion['process_exit_gate']['all_exit_codes_zero']
    exact=F.D.exact;lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper'])
    tau=exact(inputs['inputs']['adopted_evaluation']['MSE_upper'])
    classification=('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' if hi<=tau else
                    'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD' if lo>tau else 'INCONCLUSIVE')
    assert 0<=lo<=hi and classification==result['status']
    assert sum(exact(row['exact']) for row in partial['first_unsafe'])+exact(partial['all_good']['exact'])==1
    assert all(exact(row['good']['exact'])+exact(row['unsafe']['exact'])==1 for row in partial['rounds'])
    assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
    assert len(dp['rounds'])==13 and all(row['retained_states']<=8192 for row in dp['rounds'][:-1])
    assert old['allocation']['shots']==case['allocation']['shots']
    source=lambda path,*selector:R.O.ref(publication,R.STUDY+'/'+path,*selector)
    current_source=R.O.ref(CURRENT,CURRENT_DIR+'run01/B0I_P7_Y4_16h_over_3__B_PRICE/RESULT.json')
    comparison=[]
    for width,cid,status,value,origin in [
        ('16/3',old['record']['deterministic_input']['canonical_binding']['original_configuration']['id'],old['status'],old_risk,current_source),
        ('32/5',case['configuration_id'],classification,risk,source(stem+'RESULT.json'))]:
        comparison.append(dict(requested_width_over_h=width,configuration_id=cid,status=status,
            MSE_lower=deepcopy(value['MSE_lower']),MSE_upper=deepcopy(value['MSE_upper']),
            MSE_lower_display_Ha2=F.D.display(value['MSE_lower']),MSE_upper_display_Ha2=F.D.display(value['MSE_upper']),
            exact_result_source=origin))
    summary=dict(status='ONE_FIXED_Q3_SAVED',model_status=classification,comparison=comparison,
        configuration_id=case['configuration_id'],total_view_id=case['total_view_id'],binding_id=case['binding_id'],
        allocation=deepcopy(case['allocation']),scientific_API_calls=1,new_signal=0,new_reference=0,
        build=0,compile=0,price=0,fit=0,additional_allocation=0,retry=0,
        exact_tau=deepcopy(inputs['inputs']['adopted_evaluation']['MSE_upper']),
        closure=partial['decoder_certificate'],DP_status=dp['status'],transition_pairs=dp['transition_pairs'],
        residual_probability=deepcopy(dp['residual_probability']),residual_loss_upper=deepcopy(dp['residual_loss_contribution_upper']),
        residual_probability_display=F.D.display(dp['residual_probability']),residual_loss_upper_display=F.D.display(dp['residual_loss_contribution_upper']),
        per_round=F.D.display(dp['rounds']),numerical_allowance=dp['transition_allowance'],
        source=result['source'],runtime=inputs['runtime'],raw_publication=publication,
        original_fixed_inputs=R.FIXED,signal_packet=R.PACKET,
        resource=usage,supervision_wall_seconds=completion['wall_seconds'],
        sampled_process_tree_peak_RSS_KiB=completion['sampled_process_tree_peak_RSS_KiB'],
        process_exit_gate=completion['process_exit_gate'],coarse_cost_RZ=None,sufficient_N=None,
        G_qualified=None,winner=None,mandatory_stop=True)
    binding=result['record']['deterministic_input']['canonical_binding']
    join=dict(kind='FIXED_ALLOCATION_Q3_RESULT_JOIN_NOT_COST_OR_N_QUALIFICATION',
        configuration_id=case['configuration_id'],configuration_source=case['configuration'],
        total_view_id=case['total_view_id'],binding_id=case['binding_id'],signal_packet=R.PACKET,
        allocation=case['allocation'],fixed_allocation_source=R.O.ref(R.FIXED['commit'],R.FIXED['path'],'cases',0,'allocation'),
        assessment=result['adopted_q3_assessment'],model_status=classification,
        MSE_lower=deepcopy(risk['MSE_lower']),MSE_upper=deepcopy(risk['MSE_upper']),
        exact_tau=summary['exact_tau'],result_source=source(stem+'RESULT.json'),
        API_source=source(stem+'API_RETURN.json',0),scientific_API_calls=1,
        rows=[dict(m=m,axis=axis,N=case['allocation']['shots'][m]['N_'+axis],
            physical_id=result['record']['signal_input']['means'][m]['provider_binding']['physical_context']['physical_candidate']['id'],
            original_candidate_id=binding['original_configuration']['rounds'][m]['candidate']['id'],
            total_candidate_id=binding['total_configuration']['rounds'][m]['candidate']['id'])
            for m in range(13) for axis in ('X','Y')],coarse_cost_RZ=None,sufficient_N=None,G_qualified=None,winner=None)
    R.O.write(HERE/'SUMMARY.json',summary);R.O.write(HERE/'C_RESULT_JOIN.json',join)
    R.O.write(HERE/'CHECKS.json',dict(API_equals_record=True,one_scientific_call=True,original_fixed_inputs_reused=True,
        original_guard_passed=True,all_child_exit_zero=True,no_monitor_errors=True,exact_threshold_comparison=True,
        exact_mass_partition=True,residual_loss_accounted=True,old_STOP_preserved=True,independent_review=False,mandatory_stop=True))
    print('SUMMARY',classification,[(c['requested_width_over_h'],c['MSE_lower_display_Ha2'],c['MSE_upper_display_Ha2']) for c in comparison])


if __name__=='__main__':main()
