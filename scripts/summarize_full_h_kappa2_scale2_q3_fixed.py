"""Project the one approved return, retain exact gates, copy the saved control."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_q3_projection_helpers',ROOT/'scripts/summarize_pf_shortlist_total_q3_fixed.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
read,exact,display=S.read,S.exact,S.display
DIRECTORY=ROOT/'docs/research/full_h_kappa2_scale2_q3_results_20261115'
PREVIOUS=ROOT/'docs/research/pr_df_fullh_q3_fixed_results_20261115/SUMMARY.json'


def summarize():
    inputs=read(DIRECTORY/'INPUTS.json');run=DIRECTORY/'run01'
    execution=read(run/'EXECUTION.json');completion=read(run/'COMPLETION.json')
    assert len(inputs['cases'])==1 and inputs['maximum_scientific_calls']==1
    case=inputs['cases'][0];label=case['label'];folder=run/label
    result=read(folder/'RESULT.json');receipt=read(folder/'PERSISTED.json');usage=read(folder/'RESOURCE_FINAL.json')
    assert result['source']==execution['source']==usage['source']
    assert result['scientific_evaluation_calls']<=1
    row=dict(label=label,status=result['status'],input_disposition='NEW_EVALUATION',source=result['source'],
        result_path='run01/'+label+'/RESULT.json',scientific_evaluation_calls=result['scientific_evaluation_calls'],
        original_API_return_saved=receipt['api_return_saved'],operational_status=receipt['operational_status'],
        resources=usage,original_configuration_id=case['original_configuration_id'],
        total_configuration_id=case['total_configuration_id'],binding_id=case['binding_id'],
        allocation_id=case['allocation']['allocation_id'],signal_source=case['signal_packet'],
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None)
    joins=[]
    if row['status']=='UNKNOWN':
        row['reason']=result['reason']
    else:
        api=read(folder/'API_RETURN.json');assert api==[result['record']]
        risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp']
        coarse=read(folder/'PARTIAL_COARSE.json');moments=risk['moments']
        lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper']);tau=exact(inputs['adopted_evaluation']['MSE_upper'])
        assert 0<=lo<=hi
        expected=('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'if hi<=tau else
                  'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'if lo>tau else'INCONCLUSIVE')
        assert expected==row['status'] and result['scientific_evaluation_calls']==1
        assert coarse['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
        assert coarse['decoder_certificate']['checked_pairs']<=inputs['options']['maximum_verification_pairs']
        assert all(x['cells']<=inputs['options']['maximum_round_outcomes'] for x in coarse['rounds'])
        assert sum(exact(x['exact'])for x in coarse['first_unsafe'])+exact(coarse['all_good']['exact'])==1
        assert all(exact(x['good']['exact'])+exact(x['unsafe']['exact'])==1 for x in coarse['rounds'])
        assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
        scope=inputs['options']['total_state_refinement']
        assert dp['transition_pairs']<=scope['maximum_transition_pairs'] and len(dp['rounds'])==13
        assert all(x['retained_states']<=scope['maximum_frontier_states'] for x in dp['rounds'][:-1])
        assert 0<=exact(dp['residual_probability'])<=1
        assert hi==min(exact(v)for v in risk['bounds']['upper_candidates'].values())
        assert all(exact(moments[k][0])<=exact(moments[k][1])
                   for k in ('mean_energy','second_moment','bias','bias_squared','variance'))
        assert exact(moments['bias_squared'][0])+exact(moments['variance'][0])<=hi
        assert exact(moments['bias_squared'][1])+exact(moments['variance'][1])>=lo
        assert all(result[x]is None for x in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
        assert all(result[x]==0 for x in ('MC_calls','old_partial_DP_calls','price_calls'))
        row.update(MSE_lower=display(risk['MSE_lower']),MSE_upper=display(risk['MSE_upper']),
            MSE_upper_method=risk['MSE_upper_method'],moments=display(moments),
            upper_candidates=display(risk['bounds']['upper_candidates']),first_unsafe=display(coarse['first_unsafe']),
            all_good=display(coarse['all_good']),any_zero=display(dp['any_zero']),
            total_decoder_unresolved='NONE_BY_TOTAL_DEFINITION',
            old_partial_unresolved_probability_bounds=display(dp['partial_decoder']['unresolved_probability_bounds']),
            per_round=display(dp['rounds']),residual_probability=display(dp['residual_probability']),
            residual_loss_upper=display(dp['residual_loss_contribution_upper']),
            retained_MSE_contribution=display(dp['retained_MSE_contribution']),
            residual_dropped_states=dp['residual_dropped_states'],
            guard_failed_residual_probability=display(dp['residual_principal_guard_failed_probability']),
            decoder_closure=coarse['decoder_certificate'],lemma_status=risk['bounds']['lemma']['status'],
            transition_pairs=dp['transition_pairs'],numerical_allowance=dp['transition_allowance'],
            max_observed_transition_excess_rad=dp['max_observed_transition_excess_rad'],
            gates=dict(API_return_preserved=True,first_unsafe_partition_exact=True,round_probabilities_exact=True,
                DP_mass_checks_in_kernel=True,residual_loss_accounted=True,moment_intervals_consistent=True,
                ordered_bounds_and_exact_threshold=True,finite_scope=True))
        binding=result['record']['deterministic_input']['canonical_binding']
        origin=binding['original_configuration'];total=binding['total_configuration'];means=result['record']['signal_input']['means']
        joins=[dict(label=label,source=result['source'],model_status=row['status'],result_path=row['result_path'],
            API_return_path='run01/'+label+'/API_RETURN.json',original_configuration_id=case['original_configuration_id'],
            total_configuration_id=case['total_configuration_id'],binding_id=case['binding_id'],
            allocation=case['allocation'],signal_packet_source=case['signal_packet'],
            rows=[dict(m=m,axis=axis,physical_id=means[m]['provider_binding']['physical_context']['physical_candidate']['id'],
                original_candidate_id=origin['rounds'][m]['candidate']['id'],total_candidate_id=total['rounds'][m]['candidate']['id'],
                N=case['allocation']['shots'][m]['N_'+axis],
                original_law_selector=['record','deterministic_input','canonical_binding','mapping',m,'original_law_binding'],
                total_law_selector=['record','deterministic_input','canonical_binding','mapping',m,'total_law_binding'])
                for m in range(13)for axis in ('X','Y')],sufficient_N=None,G=None,G_qualified=None)]
    summary=dict(status=completion['status'],source=execution['source'],completion=completion,
        candidates_requested=1,reused_candidates=[],held=[],scientific_evaluation_calls=result['scientific_evaluation_calls'],
        counts={row['status']:1},rows=[row],adopted_evaluation=display(inputs['adopted_evaluation']),
        vector_X=[r['N_X']for r in case['allocation']['shots']],vector_Y=[r['N_Y']for r in case['allocation']['shots']],
        display_precision=25,display_qualification='EXACT_RATIONAL_ENDPOINTS_IN_API_RETURN_DECIMALS_FOR_DISPLAY',
        signal_recomputed=False,old_control_recomputed=False,MC_calls=0,old_partial_DP_calls=0,price_calls=0,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    prior=next(r for r in read(PREVIOUS)['rows'] if r['label']==label)
    assert all(prior[k]==row[k] for k in ('original_configuration_id','total_configuration_id','binding_id','signal_source'))
    comparison=dict(kind='SAVED_CONTROL_AND_APPROVED_SCALE2_NO_RATIOS_OR_FURTHER_RISK_CALCULATION',
        previous_result_source=dict(commit='a82b0f0ec7f0f11d9981ddb2343fcd5378a87503',path=str(PREVIOUS.relative_to(ROOT))),
        control=dict(N_X=[64,60,56,52,47,43,38,34,30,25,21,17,12],
                     N_Y=[64,60,56,52,47,43,38,34,30,25,21,17,12],saved_summary_row=prior),
        scale2=dict(N_X=summary['vector_X'],N_Y=summary['vector_Y'],summary_row=row),
        parity_changes_at_rounds=[4,5,9,10,11],causal_zero_tie_attribution=False,
        sufficient_N=None,G=None,G_qualified=None,winner=None)
    index=dict(kind='FIXED_MODEL_RESULT_INDEX_NOT_SUFFICIENT_N_PACKET',source=execution['source'],
        candidates=joins,sufficient_N=None,G=None,G_qualified=None)
    for name,value in [('SUMMARY.json',summary),('COMPARISON.json',comparison),('C_RESULT_JOIN.json',index)]:
        with(DIRECTORY/name).open('x')as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status=row['status'],operational_status=row['operational_status'],
        calls=result['scientific_evaluation_calls'],MSE_lower=row.get('MSE_lower'),MSE_upper=row.get('MSE_upper')),indent=2))


if __name__=='__main__':summarize()
