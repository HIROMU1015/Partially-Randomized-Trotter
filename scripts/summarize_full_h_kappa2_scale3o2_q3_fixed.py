"""Project one saved scale3o2 API return and two immutable comparison rows."""
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_reuse_q3_display',ROOT/'scripts/summarize_pf_shortlist_total_q3_fixed.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
read,exact,display=S.read,S.exact,S.display
DIRECTORY=ROOT/'docs/research/full_h_kappa2_scale3o2_q3_results_20261115'
PREVIOUS=[('control','a82b0f0ec7f0f11d9981ddb2343fcd5378a87503',
    'docs/research/pr_df_fullh_q3_fixed_results_20261115/SUMMARY.json'),
    ('scale2','422db93fc7bb599a7ea46c5b58e5dec0fda9ef08',
    'docs/research/full_h_kappa2_scale2_q3_results_20261115/SUMMARY.json')]


def summarize():
    inputs=read(DIRECTORY/'INPUTS.json');run=DIRECTORY/'run01';completion=read(run/'COMPLETION.json')
    case=inputs['cases'][0];label=case['label'];folder=run/label
    result=read(folder/'RESULT.json');receipt=read(folder/'PERSISTED.json');usage=read(folder/'RESOURCE_FINAL.json')
    execution=read(run/'EXECUTION.json');process=read(DIRECTORY/'PROCESS_RETURN.json')
    assert len(inputs['cases'])==inputs['maximum_scientific_calls']==1
    assert result['source']==execution['source']==usage['source']
    assert result['scientific_evaluation_calls']<=1
    row=dict(label=label,status=result['status'],source=result['source'],scientific_evaluation_calls=result['scientific_evaluation_calls'],
        original_configuration_id=case['original_configuration_id'],total_configuration_id=case['total_configuration_id'],
        binding_id=case['binding_id'],allocation_id=case['allocation']['allocation_id'],signal_source=case['signal_packet'],
        original_API_return_saved=receipt['api_return_saved'],operational_status=receipt['operational_status'],resources=usage)
    joins=[]
    if result['status']=='UNKNOWN':row['reason']=result['reason']
    else:
        assert read(folder/'API_RETURN.json')==[result['record']]
        risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp'];coarse=read(folder/'PARTIAL_COARSE.json')
        lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper']);tau=exact(inputs['adopted_evaluation']['MSE_upper']);assert 0<=lo<=hi
        status=('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'if hi<=tau else 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'if lo>tau else 'INCONCLUSIVE')
        assert status==row['status'] and result['scientific_evaluation_calls']==1
        assert hi==min(exact(v)for v in risk['bounds']['upper_candidates'].values())
        assert coarse['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
        assert coarse['decoder_certificate']['checked_pairs']<=inputs['options']['maximum_verification_pairs']
        assert all(v['cells']<=inputs['options']['maximum_round_outcomes']for v in coarse['rounds'])
        assert sum(exact(v['exact'])for v in coarse['first_unsafe'])+exact(coarse['all_good']['exact'])==1
        assert all(exact(v['good']['exact'])+exact(v['unsafe']['exact'])==1 for v in coarse['rounds'])
        assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
        scope=inputs['options']['total_state_refinement'];assert dp['transition_pairs']<=scope['maximum_transition_pairs']
        assert len(dp['rounds'])==13 and all(v['retained_states']<=scope['maximum_frontier_states']for v in dp['rounds'][:-1])
        assert 0<=exact(dp['residual_probability'])<=1
        moments=risk['moments']
        assert all(exact(moments[k][0])<=exact(moments[k][1])for k in ('mean_energy','second_moment','bias','bias_squared','variance'))
        assert exact(moments['bias_squared'][0])+exact(moments['variance'][0])<=hi
        assert exact(moments['bias_squared'][1])+exact(moments['variance'][1])>=lo
        assert all(result[k]is None for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
        assert all(result[k]==0 for k in ('MC_calls','old_partial_DP_calls','price_calls'))
        row.update(MSE_lower=display(risk['MSE_lower']),MSE_upper=display(risk['MSE_upper']),MSE_upper_method=risk['MSE_upper_method'],
            moments=display(moments),upper_candidates=display(risk['bounds']['upper_candidates']),first_unsafe=display(coarse['first_unsafe']),
            all_good=display(coarse['all_good']),any_zero=display(dp['any_zero']),total_decoder_unresolved='NONE_BY_TOTAL_DEFINITION',
            old_partial_unresolved_probability_bounds=display(dp['partial_decoder']['unresolved_probability_bounds']),
            per_round=display(dp['rounds']),retained_MSE_contribution=display(dp['retained_MSE_contribution']),
            residual_probability=display(dp['residual_probability']),residual_loss_upper=display(dp['residual_loss_contribution_upper']),
            residual_dropped_states=dp['residual_dropped_states'],guard_failed_residual_probability=display(dp['residual_principal_guard_failed_probability']),
            decoder_closure=coarse['decoder_certificate'],transition_pairs=dp['transition_pairs'],lemma_status=risk['bounds']['lemma']['status'],
            numerical_allowance=dp['transition_allowance'],max_observed_transition_excess_rad=dp['max_observed_transition_excess_rad'],
            gates=dict(API_return_preserved=True,exact_first_unsafe_partition=True,exact_round_mass=True,
                exact_DP_mass_checks_in_kernel=True,residual_loss_accounted=True,ordered_moments_and_bounds=True,
                exact_threshold=True,finite_scope=True))
        binding=result['record']['deterministic_input']['canonical_binding'];origin=binding['original_configuration'];total=binding['total_configuration']
        means=result['record']['signal_input']['means']
        joins=[dict(label=label,source=result['source'],model_status=row['status'],original_configuration_id=case['original_configuration_id'],
            total_configuration_id=case['total_configuration_id'],binding_id=case['binding_id'],allocation=case['allocation'],
            signal_packet_source=case['signal_packet'],API_return_path='run01/'+label+'/API_RETURN.json',
            result_path='run01/'+label+'/RESULT.json',
            rows=[dict(m=m,axis=axis,physical_id=means[m]['provider_binding']['physical_context']['physical_candidate']['id'],
                original_candidate_id=origin['rounds'][m]['candidate']['id'],total_candidate_id=total['rounds'][m]['candidate']['id'],
                N=case['allocation']['shots'][m]['N_'+axis],
                original_law_selector=['record','deterministic_input','canonical_binding','mapping',m,'original_law_binding'],
                total_law_selector=['record','deterministic_input','canonical_binding','mapping',m,'total_law_binding'])for m in range(13)for axis in ('X','Y')],
            sufficient_N=None,G=None,G_qualified=None)]
    row.update(sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None)
    comparison=[]
    for name,sha,path in PREVIOUS:
        previous=json.loads(subprocess.check_output(['git','show',sha+':'+path],cwd=ROOT));v=next(v for v in previous['rows']if v['label']==label)
        assert all(v[k]==row[k]for k in ('original_configuration_id','total_configuration_id','binding_id','signal_source'))
        nx=previous.get('vector',previous.get('vector_X'));ny=previous.get('vector',previous.get('vector_Y'))
        comparison.append(dict(name=name,result_source=dict(commit=sha,path=path),N_X=nx,N_Y=ny,saved_summary_row=v))
    nx=[v['N_X']for v in case['allocation']['shots']];ny=[v['N_Y']for v in case['allocation']['shots']]
    comparison.insert(1,dict(name='scale3o2',result_source=result['source'],N_X=nx,N_Y=ny,summary_row=row))
    exit_gate=completion['process_exit_gate']
    if not exit_gate['all_exit_codes_zero']or process['exitcode']!=0:completion['status']='UNKNOWN_PROCESS_EXIT'
    summary=dict(status=completion['status'],source=execution['source'],completion=completion,process_return=process,
        candidates_requested=1,scientific_evaluation_calls=result['scientific_evaluation_calls'],rows=[row],
        adopted_evaluation=display(inputs['adopted_evaluation']),vector_X=nx,vector_Y=ny,display_precision=25,
        display_qualification='EXACT_ENDPOINTS_IN_API_RETURN_DECIMALS_FOR_DISPLAY',signal_recomputed=False,
        control_recomputed=False,scale2_recomputed=False,MC_calls=0,price_calls=0,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    output=[('SUMMARY.json',summary),('COMPARISON.json',dict(kind='THREE_FIXED_ALLOCATIONS_NO_MONOTONICITY_OR_MINIMUM_N_CLAIM',
        rows=comparison,sufficient_N=None,G=None,G_qualified=None,winner=None)),
        ('C_RESULT_JOIN.json',dict(kind='FIXED_MODEL_RESULT_INDEX_NOT_SUFFICIENT_N_PACKET',source=execution['source'],candidates=joins,
            sufficient_N=None,G=None,G_qualified=None))]
    for name,value in output:
        with(DIRECTORY/name).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=row['status'],operational_status=row['operational_status'],calls=result['scientific_evaluation_calls'],
        MSE_lower=row.get('MSE_lower'),MSE_upper=row.get('MSE_upper')),indent=2))


if __name__=='__main__':summarize()
