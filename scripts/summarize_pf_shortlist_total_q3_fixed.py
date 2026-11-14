"""Project the six permitted results and exact accounting checks; no risk calls."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'docs/research/pf_shortlist_q3_fixed_results_20261115'


def read(path): return json.loads(path.read_text())
def exact(value): return Fraction(value['numerator'],value['denominator'])


def display(value):
    if isinstance(value,dict) and 'numerator' in value and 'denominator' in value:
        with localcontext() as c:
            c.prec=25
            return str(Decimal(value['numerator'])/Decimal(value['denominator']))
    if isinstance(value,dict): return {k:display(v) for k,v in value.items()}
    if isinstance(value,list): return [display(v) for v in value]
    return value


def summarize():
    inputs=read(DIRECTORY/'INPUTS.json'); run=DIRECTORY/'run01'; execution=read(run/'EXECUTION.json')
    completion=read(run/'COMPLETION.json'); rows=[]; joins=[]; counts={}
    for case in inputs['cases']:
        label=case['label']; folder=run/label
        recovered=not(folder/'RESULT.json').exists() and (folder/'RECOVERED_RESULT.json').exists()
        result_name='RECOVERED_RESULT.json'if recovered else'RESULT.json'
        if not (folder/result_name).exists():
            counts['MISSING_RESULT']=counts.get('MISSING_RESULT',0)+1
            rows.append(dict(label=label,status='UNKNOWN',reason='MISSING_RESULT')); continue
        result=read(folder/result_name); status=result['status']; counts[status]=counts.get(status,0)+1
        assert result['source']==execution['source']
        row=dict(label=label,status=status,source=result['source'],
            result_path='run01/'+label+'/'+result_name,configuration_id=case['configuration_id'],
            total_view_id=case['total_view_id'],binding_id=case['binding_id'],
            signal_packet_source=case['signal_packet'],configuration_source=case['configuration'],
            allocation_id=case['allocation']['allocation_id'],scientific_evaluation_calls=result['scientific_evaluation_calls'],
            performance=result['performance'],evidence_status=result.get('evidence_status','ORIGINAL_API_RETURN'),
            sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None)
        if status=='UNKNOWN': row['reason']=result['reason']; rows.append(row); continue
        risk=(result['model_bound']if recovered else result['record']['diagnostic']['deterministic_risk'])
        dp=risk['bounds']['total_state_dp']
        coarse=read(folder/'PARTIAL_COARSE.json'); moments=risk['moments']
        lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper'])
        tau=exact(inputs['adopted_evaluation']['MSE_upper'])
        assert 0<=lo<=hi
        classification=('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' if hi<=tau else
            'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD' if lo>tau else 'INCONCLUSIVE')
        assert classification==status
        assert sum(exact(r['exact'])for r in coarse['first_unsafe'])+exact(coarse['all_good']['exact'])==1
        assert all(exact(r['good']['exact'])+exact(r['unsafe']['exact'])==1 for r in coarse['rounds'])
        assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
        assert len(dp['rounds'])==13 and all(r['retained_states']<=8192 for r in dp['rounds'][:-1])
        assert dp['transition_pairs']<=141559036
        moment_lo=exact(moments['bias_squared'][0])+exact(moments['variance'][0])
        moment_hi=exact(moments['bias_squared'][1])+exact(moments['variance'][1])
        assert moment_lo<=hi and moment_hi>=lo
        assert all(result[k]is None for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
        assert all(result[k]==0 for k in ('MC_calls','old_partial_DP_calls','price_calls'))
        row.update(MSE_lower=display(risk['MSE_lower']),MSE_upper=display(risk['MSE_upper']),
            best_upper_method=risk['MSE_upper_method'],moments=display(moments),
            valid_upper_candidates=display(risk['bounds']['upper_candidates']),
            first_unsafe=display(coarse['first_unsafe']),all_good=display(coarse['all_good']),
            any_zero=display(dp['any_zero']),total_decoder_unresolved='NONE_BY_TOTAL_DEFINITION',
            old_partial_unresolved_probability_bounds=display(dp['partial_decoder']['unresolved_probability_bounds']),
            per_round=display(dp['rounds']),residual_probability=display(dp['residual_probability']),
            residual_loss_upper=display(dp['residual_loss_contribution_upper']),
            guard_failed_residual_probability=display(dp['residual_principal_guard_failed_probability']),
            retained_MSE_contribution=display(dp['retained_MSE_contribution']),
            decoder_closure=coarse['decoder_certificate'],lemma_status=risk['bounds']['lemma']['status'],
            transition_pairs=dp['transition_pairs'],numerical_allowance=dp['transition_allowance'],
            gates=dict(exact_first_unsafe_partition=True,exact_round_good_unsafe=True,
                exact_DP_mass_checks_in_kernel=True,residual_loss_accounted=True,
                moment_MSE_intervals_consistent=True,ordered_bounds_and_exact_filter=True))
        binding=(result['canonical_binding_metadata']if recovered else result['record']['deterministic_input']['canonical_binding'])
        origin=binding['original_configuration']; total=binding['total_configuration']
        physical_ids=(result['physical_ids']if recovered else
            [v['provider_binding']['physical_context']['physical_candidate']['id']for v in result['record']['signal_input']['means']])
        binding_selector=['canonical_binding_metadata']if recovered else['record','deterministic_input','canonical_binding']
        joins.append(dict(label=label,original_configuration_id=case['configuration_id'],
            total_configuration_id=case['total_view_id'],canonical_binding=case['binding_id'],
            result_path='run01/'+label+'/'+result_name,evidence_status=row['evidence_status'],
            original_packet_source=case['signal_packet'],allocation=case['allocation'],
            rows=[dict(m=m,axis=axis,physical_id=physical_ids[m],
                original_candidate_id=origin['rounds'][m]['candidate']['id'],
                total_candidate_id=total['rounds'][m]['candidate']['id'],N=case['allocation']['shots'][m]['N_'+axis],
                original_law_selector=binding_selector+['mapping',m,'original_law_binding'],
                total_law_selector=binding_selector+['mapping',m,'total_law_binding'])
                for m in range(13)for axis in ('X','Y')],sufficient_N=None,G_qualified=None))
        rows.append(row)
    summary=dict(status=completion['status'],execution=execution,completion=completion,
        evaluations_planned=6,evaluations_executed=sum(r.get('scientific_evaluation_calls',0)for r in rows),
        counts=counts,adopted_evaluation=display(inputs['adopted_evaluation']),
        vector=[r['N_X']for r in inputs['cases'][0]['allocation']['shots']],rows=rows,
        display_precision=25,display_qualification='EXACT_RATIONAL_ENDPOINTS_IN_RESULT_FILES_DECIMALS_FOR_DISPLAY',
        Pauli_recomputed=False,signal_recomputed=False,MC_calls=0,price_calls=0,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    price_join=dict(kind='RESULT_INDEX_ONLY_NO_PRICE_OR_N_QUALIFICATION',source=execution['source'],
        C_alignment=inputs['C_alignment'],candidates=joins,sufficient_N=None,G=None,G_qualified=None)
    for name,value in [('SUMMARY.json',summary),('C_RESULT_JOIN.json',price_join)]:
        with (DIRECTORY/name).open('x')as stream: json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(counts=counts,evaluations_executed=summary['evaluations_executed']),indent=2))


if __name__=='__main__': summarize()
