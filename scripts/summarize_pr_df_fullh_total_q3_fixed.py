"""Read the permitted results, check exact accounting, write C's result index."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_existing_result_projection',ROOT/'scripts/summarize_pf_shortlist_total_q3_fixed.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
read,exact,display=S.read,S.exact,S.display
DIRECTORY=ROOT/'docs/research/pr_df_fullh_q3_fixed_results_20261115'


def summarize():
    inputs=read(DIRECTORY/'INPUTS.json');run=DIRECTORY/'run01'
    execution=read(run/'EXECUTION.json');completion=read(run/'COMPLETION.json')
    rows=[];joins=[];counts={}
    for case in inputs['cases']:
        label=case['label'];folder=run/label
        if not(folder/'RESULT.json').exists():
            rows.append(dict(label=label,status='UNKNOWN',reason='WORKER_RESULT_NOT_SAVED'))
            counts['UNKNOWN']=counts.get('UNKNOWN',0)+1;continue
        result=read(folder/'RESULT.json');receipt=read(folder/'PERSISTED.json');usage=read(folder/'RESOURCE_FINAL.json')
        status=result['status'];counts[status]=counts.get(status,0)+1
        row=dict(label=label,status=status,input_disposition='NEW_EVALUATION'if result['scientific_evaluation_calls']else'NOT_STARTED',
            source=result['source'],result_path='run01/'+label+'/RESULT.json',
            scientific_evaluation_calls=result['scientific_evaluation_calls'],
            original_API_return_saved=receipt['api_return_saved'],operational_status=receipt['operational_status'],
            resources=usage,original_configuration_id=case['original_configuration_id'],
            total_configuration_id=case['total_configuration_id'],binding_id=case['binding_id'],
            allocation_id=case['allocation']['allocation_id'],signal_source=case['signal_packet'],
            sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None)
        assert result['source']==execution['source']==usage['source']
        if status=='UNKNOWN':row['reason']=result['reason'];rows.append(row);continue
        api=read(folder/'API_RETURN.json');assert api==[result['record']]
        risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp']
        coarse=read(folder/'PARTIAL_COARSE.json');lemma=risk['bounds']['lemma']
        lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper']);tau=exact(inputs['adopted_evaluation']['MSE_upper'])
        assert 0<=lo<=hi
        expected=('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'if hi<=tau else
                  'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'if lo>tau else'INCONCLUSIVE')
        assert expected==status and result['scientific_evaluation_calls']==1
        assert sum(exact(x['exact'])for x in coarse['first_unsafe'])+exact(coarse['all_good']['exact'])==1
        assert all(exact(x['good']['exact'])+exact(x['unsafe']['exact'])==1 for x in coarse['rounds'])
        assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
        assert dp['transition_pairs']<=141559036 and len(dp['rounds'])==13
        assert all(x['retained_states']<=8192 for x in dp['rounds'][:-1])
        moments=risk['moments']
        assert exact(moments['bias_squared'][0])+exact(moments['variance'][0])<=hi
        assert exact(moments['bias_squared'][1])+exact(moments['variance'][1])>=lo
        assert all(result[x]is None for x in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
        assert all(result[x]==0 for x in ('MC_calls','old_partial_DP_calls','price_calls'))
        row.update(MSE_lower=display(risk['MSE_lower']),MSE_upper=display(risk['MSE_upper']),
            MSE_upper_method=risk['MSE_upper_method'],moments=display(moments),
            upper_candidates=display(risk['bounds']['upper_candidates']),
            first_unsafe=display(coarse['first_unsafe']),all_good=display(coarse['all_good']),
            any_zero=display(dp['any_zero']),total_decoder_unresolved='NONE_BY_TOTAL_DEFINITION',
            old_partial_unresolved_probability_bounds=display(dp['partial_decoder']['unresolved_probability_bounds']),
            per_round=display(dp['rounds']),residual_probability=display(dp['residual_probability']),
            residual_loss_upper=display(dp['residual_loss_contribution_upper']),
            guard_failed_residual_probability=display(dp['residual_principal_guard_failed_probability']),
            decoder_closure=coarse['decoder_certificate'],lemma_status=lemma['status'],
            transition_pairs=dp['transition_pairs'],numerical_allowance=dp['transition_allowance'],
            gates=dict(API_return_preserved=True,first_unsafe_partition_exact=True,round_probabilities_exact=True,
                DP_mass_checks_in_kernel=True,residual_loss_accounted=True,moment_intervals_consistent=True,
                exact_threshold_and_ordered_bounds=True))
        binding=result['record']['deterministic_input']['canonical_binding']
        original=binding['original_configuration'];total=binding['total_configuration']
        means=result['record']['signal_input']['means']
        joins.append(dict(label=label,source=result['source'],result_path=row['result_path'],
            API_return_path='run01/'+label+'/API_RETURN.json',
            original_configuration_id=case['original_configuration_id'],total_configuration_id=case['total_configuration_id'],
            canonical_binding_id=case['binding_id'],allocation=case['allocation'],
            configuration_source=case['configuration'],signal_packet_source=case['signal_packet'],
            model_status=status,sufficient_N=None,G_qualified=None,
            rows=[dict(m=m,axis=axis,physical_id=means[m]['provider_binding']['physical_context']['physical_candidate']['id'],
                original_candidate_id=original['rounds'][m]['candidate']['id'],total_candidate_id=total['rounds'][m]['candidate']['id'],
                N=case['allocation']['shots'][m]['N_'+axis],
                original_law_selector=['record','deterministic_input','canonical_binding','mapping',m,'original_law_binding'],
                total_law_selector=['record','deterministic_input','canonical_binding','mapping',m,'total_law_binding'])
                for m in range(13)for axis in ('X','Y')]))
        rows.append(row)
    summary=dict(status=completion['status'],source=execution['source'],completion=completion,
        candidates_requested=3,reused_candidates=[],held=inputs['held'],counts=counts,
        scientific_evaluation_calls=sum(r.get('scientific_evaluation_calls',0)for r in rows),rows=rows,
        adopted_evaluation=display(inputs['adopted_evaluation']),vector=[r['N_X']for r in inputs['cases'][0]['allocation']['shots']],
        display_precision=25,display_qualification='EXACT_RATIONAL_ENDPOINTS_IN_API_RETURN_AND_RESULT_DECIMALS_FOR_DISPLAY',
        signal_recomputed=False,MC_calls=0,old_partial_DP_calls=0,price_calls=0,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    index=dict(kind='FIXED_MODEL_RESULT_INDEX_NOT_SUFFICIENT_N_PACKET',source=execution['source'],candidates=joins,
        sufficient_N=None,G=None,G_qualified=None)
    for name,value in [('SUMMARY.json',summary),('C_RESULT_JOIN.json',index)]:
        with(DIRECTORY/name).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=summary['status'],counts=counts,calls=summary['scientific_evaluation_calls']),indent=2))


if __name__=='__main__':summarize()
