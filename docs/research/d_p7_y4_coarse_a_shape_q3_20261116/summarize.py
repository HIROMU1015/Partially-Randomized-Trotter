"""Quote three immutable Q3s and project the one approved A_SHAPE receipt."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import run_p7_y4_coarse_a_shape_q3 as A
O,R=A.O,A.R
HERE=Path(__file__).resolve().parent


def main(raw_commit):
    F=O.q3_runtime();exact=F.D.exact
    raw_commit=subprocess.check_output(['git','rev-parse',raw_commit],cwd=ROOT,text=True).strip()
    read=lambda name:O.read(O.ref(raw_commit,A.STUDY+'/'+name))
    inputs=read('INPUTS.json');case=inputs['inputs']['cases'][0];A.validate_case(case)
    stem='run01/'+A.LABEL+'/'
    result=read(stem+'RESULT.json');api=read(stem+'API_RETURN.json');completion=read('run01/COMPLETION.json')
    assert api==[result['record']] and result['scientific_evaluation_calls']==1
    assert completion['status']=='ONE_FIXED_Q3_SAVED' and not completion['errors'] and not completion['monitor_errors']
    assert completion['process_exit_gate']['all_exit_codes_zero']
    tau=exact(inputs['inputs']['adopted_evaluation']['MSE_upper'])
    source=lambda name,*selector:O.ref(raw_commit,A.STUDY+'/'+name,*selector)
    origins=[('16/3','B_PRICE',O.ref(O.B,O.B_DIR+'run01/B0I_P7_Y4_16h_over_3__B_PRICE/RESULT.json')),
             ('16/3','A_SHAPE',O.ref(O.B,O.B_DIR+'run01/B0I_P7_Y4_16h_over_3__A_SHAPE/RESULT.json')),
             ('32/5','B_PRICE',O.ref('537df23766e962fad027bf98ba7edf3c6854d927',R.STUDY+'/run01/B0I_P7_Y4_32h_over_5/RESULT.json')),
             ('32/5','A_SHAPE',source(stem+'RESULT.json'))]
    rows=[]
    for width,shape,origin in origins:
        value=O.read(origin);risk=value['record']['diagnostic']['deterministic_risk']
        assert exact(value['adopted_q3_assessment']['q3']['limit']['MSE_upper'])==tau
        vector=A.VECTOR if shape=='A_SHAPE' else O.VECTOR
        assert [r['N_X'] for r in value['allocation']['shots']]==vector==[r['N_Y'] for r in value['allocation']['shots']]
        lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper'])
        classification=('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' if hi<=tau else
                        'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD' if lo>tau else 'INCONCLUSIVE')
        assert 0<=lo<=hi and classification==value['status']
        rows.append(dict(requested_width_over_h=width,allocation_shape=shape,
            configuration_id=value['record']['deterministic_input']['canonical_binding']['original_configuration']['id'],
            allocation=deepcopy(value['allocation']),MSE_lower=deepcopy(risk['MSE_lower']),MSE_upper=deepcopy(risk['MSE_upper']),
            MSE_lower_display_Ha2=F.D.display(risk['MSE_lower']),MSE_upper_display_Ha2=F.D.display(risk['MSE_upper']),
            status=value['status'],result_source=origin,acquisition='NEW_APPROVED_Q3' if (width,shape)==('32/5','A_SHAPE') else 'QUOTED_SAVED_RESULT'))
    new=rows[-1];risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp']
    partial=read(stem+'PARTIAL_COARSE.json')
    assert sum(exact(row['exact']) for row in partial['first_unsafe'])+exact(partial['all_good']['exact'])==1
    assert all(exact(row['good']['exact'])+exact(row['unsafe']['exact'])==1 for row in partial['rounds'])
    assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
    assert len(dp['rounds'])==13
    pending=new['status']=='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
    both_failed=all(r['status']=='MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD' for r in rows if r['requested_width_over_h']=='32/5')
    handoff='Q3_ACCEPTED_PRICE_PENDING' if pending else 'TWO_TRIED_COARSE_ALLOCATIONS_FAILED' if both_failed else 'Q3_INCONCLUSIVE'
    summary=dict(status='ONE_FIXED_Q3_SAVED',model_status=new['status'],handoff_status=handoff,
        configuration_id=A.O.CID,allocation_id=A.ALLOCATION_ID,comparison=rows,exact_tau=inputs['inputs']['adopted_evaluation']['MSE_upper'],
        tau_display_Ha2=F.D.display(inputs['inputs']['adopted_evaluation']['MSE_upper']),scientific_API_calls=1,saved_Q3s_quoted=3,
        reused_signal_rounds=13,new_signal=0,new_reference=0,build=0,compile=0,price=0,fit=0,additional_allocation=0,retry=0,
        closure=partial['decoder_certificate'],DP_status=dp['status'],transition_pairs=dp['transition_pairs'],
        residual_probability=deepcopy(dp['residual_probability']),residual_loss_upper=deepcopy(dp['residual_loss_contribution_upper']),
        residual_probability_display=F.D.display(dp['residual_probability']),residual_loss_upper_display=F.D.display(dp['residual_loss_contribution_upper']),
        runtime=inputs['runtime'],execution_source=result['source'],raw_publication=raw_commit,
        resource=read(stem+'RESOURCE_FINAL.json'),supervision_wall_seconds=completion['wall_seconds'],
        sampled_process_tree_peak_RSS_KiB=completion['sampled_process_tree_peak_RSS_KiB'],process_exit_gate=completion['process_exit_gate'],
        coarse_cost_RZ=None,sufficient_N=None,G_qualified=None,winner=None,mandatory_stop=True)
    binding=result['record']['deterministic_input']['canonical_binding']
    join=dict(kind='FIXED_ALLOCATION_Q3_RESULT_JOIN_NOT_COST_OR_N_QUALIFICATION',
        configuration_id=case['configuration_id'],configuration_source=case['configuration'],total_view_id=case['total_view_id'],binding_id=case['binding_id'],
        allocation=deepcopy(case['allocation']),fixed_allocation_source=O.ref(result['source']['commit'],A.STUDY+'/INPUTS.json','inputs','cases',0,'allocation'),
        integer_allocation_source=A.INTEGER_SOURCE,signal_packet=R.PACKET,assessment=result['adopted_q3_assessment'],
        model_status=new['status'],handoff_status=handoff,MSE_lower=new['MSE_lower'],MSE_upper=new['MSE_upper'],exact_tau=summary['exact_tau'],
        result_source=source(stem+'RESULT.json'),API_source=source(stem+'API_RETURN.json',0),scientific_API_calls=1,
        rows=[dict(m=m,axis=axis,N=case['allocation']['shots'][m]['N_'+axis],
            physical_id=result['record']['signal_input']['means'][m]['provider_binding']['physical_context']['physical_candidate']['id'],
            original_candidate_id=binding['original_configuration']['rounds'][m]['candidate']['id'],
            total_candidate_id=binding['total_configuration']['rounds'][m]['candidate']['id']) for m in range(13) for axis in ('X','Y')],
        coarse_cost_RZ=None,sufficient_N=None,G_qualified=None,winner=None)
    O.write(HERE/'SUMMARY.json',summary);O.write(HERE/'C_RESULT_JOIN.json',join)
    O.write(HERE/'CHECKS.json',dict(one_scientific_API=True,three_saved_Q3s_quoted=True,old_integers_reused=True,
        original_coarse_signal_reused=True,exact_classification=True,original_AST_guard_passed=True,
        API_equals_record=True,exact_mass_partition=True,residual_loss_accounted=True,all_child_exit_zero=True,
        old_results_and_STOP_preserved=True,coarse_cost_null=True,independent_review=False,mandatory_stop=True))
    print('NEW_Q3',new['status'],handoff)
    for r in rows:print(r['requested_width_over_h'],r['allocation_shape'],r['MSE_lower_display_Ha2'],r['MSE_upper_display_Ha2'],r['status'])
    print('WALL',summary['supervision_wall_seconds'],'DP_TRANSITIONS',summary['transition_pairs'])


if __name__=='__main__':main(sys.argv[1])
