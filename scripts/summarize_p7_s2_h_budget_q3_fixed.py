"""Verify saved Q3 returns and project two fixed width-h model-cost allocations."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_p7_h_budget_summary', ROOT/'scripts/run_p7_s2_h_budget_q3_fixed.py')
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)
read, display, exact = C.F.D.read, C.F.D.display, C.T.B.a.exact_scalar


def summarize():
    folder = ROOT/C.DIRECTORY; inputs = read(folder/'INPUTS.json'); run = folder/'run01'
    execution = read(run/'EXECUTION.json'); completion = read(run/'COMPLETION.json')
    rows, joins = [], []
    for proposal in inputs['proposals']:
        if proposal['disposition']=='BUDGET_MENU_EMPTY':
            rows.append(dict(rule=proposal['rule'],status='NO_FEASIBLE_PRICE_BUDGET_VECTOR'));continue
        if proposal['disposition']=='SAME_VECTOR_AS_OTHER_RULE':
            old = next(r for r in rows if r['label']==proposal['reused_result'])
            rows.append(dict(old,label=proposal['label'],rule=proposal['rule'],
                disposition=proposal['disposition'],scientific_evaluation_calls=0));continue
        result = None
        if proposal['disposition']=='REUSE_SAVED_FINE_Q3':
            record = C.T._selected(ROOT,proposal['reused_result'])[0]
            assessment = C.F.H.assess(record['diagnostic']['deterministic_risk'],inputs['adopted_evaluation'],record['source'])
            status, calls, source = assessment['status'], 0, record['source']
            usage, receipt = None, dict(api_return_saved=True,operational_status='REUSED_PRESERVED_API')
        else:
            job = run/proposal['label']; result = read(job/'RESULT.json')
            usage = read(job/'RESOURCE_FINAL.json'); receipt = read(job/'PERSISTED.json')
            assert result['source']==usage['source']==execution['source']
            assert result['allocation']==proposal['allocation']
            assert all(result[k] is None for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
            assert all(result[k]==0 for k in ('MC_calls','old_partial_DP_calls','price_calls'))
            status,calls,source = result['status'],result['scientific_evaluation_calls'],result['source']
            record = result.get('record')
            if record is not None: assert read(job/'API_RETURN.json')==[record]
        row = dict(label=proposal['label'],rule=proposal['rule'],disposition=proposal['disposition'],
            allocation=proposal['allocation'],conditional_model_cost_RZ=proposal['conditional_cost'],
            difference_from_coarse_RZ=proposal['conditional_cost']-C.BUDGET,
            budget_RZ=C.BUDGET,unused_budget=proposal['unused_budget'],
            price_partition=proposal['price_partition'],
            extrapolated_percent_exact=C.F.exact(100*exact(proposal['price_partition']['extrapolated_fraction'])),
            status=status,scientific_evaluation_calls=calls,result_source=source,resources=usage,
            original_API_return_saved=receipt['api_return_saved'],operational_status=receipt['operational_status'],
            MSE_lower=None,MSE_upper=None,sufficient_N=None,G_qualified=None)
        if status=='UNKNOWN': row['reason']=result.get('reason') if result else 'UNKNOWN_REUSED_RESULT'
        else:
            risk = record['diagnostic']['deterministic_risk'];dp = risk['bounds']['total_state_dp']
            assert C.F.H.assess(risk,inputs['adopted_evaluation'],source)['status']==status
            lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper'])
            assert 0<=lo<=hi and hi==min(exact(v) for v in risk['bounds']['upper_candidates'].values())
            assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
            assert 0<=exact(dp['residual_probability'])<=1
            assert len(dp['rounds'])==13 and all(r['retained_states']<=8192 for r in dp['rounds'][:-1])
            assert dp['transition_pairs']<=inputs['options']['total_state_refinement']['maximum_transition_pairs']
            moments=risk['moments']
            assert exact(moments['bias_squared'][0])+exact(moments['variance'][0])<=hi
            assert exact(moments['bias_squared'][1])+exact(moments['variance'][1])>=lo
            if result is not None:
                coarse = read(run/proposal['label']/'PARTIAL_COARSE.json')
                assert coarse['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
                assert coarse['decoder_certificate']['checked_pairs']<=inputs['options']['maximum_verification_pairs']
                assert sum(exact(r['exact']) for r in coarse['first_unsafe'])+exact(coarse['all_good']['exact'])==1
                assert all(exact(r['good']['exact'])+exact(r['unsafe']['exact'])==1 for r in coarse['rounds'])
                row['closure']=coarse['decoder_certificate']
            row.update(MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],moments=moments,
                residual_mass=dp['residual_probability'],residual_loss_upper=dp['residual_loss_contribution_upper'],
                any_zero=dp['any_zero'],per_round=dp['rounds'],transition_pairs=dp['transition_pairs'],
                numerical_allowance=dp['transition_allowance'],gates='VALIDATED_EXACT_MASS_RESIDUAL_MOMENTS_SCOPE_THRESHOLD')
        rows.append(row)
        joins.append(dict(configuration_id=proposal['configuration_id'],total_view_id=proposal['total_view_id'],
            binding_id=proposal['binding_id'],signal_packet=proposal['signal_packet'],
            allocation=proposal['allocation'],model_status=status,result_source=source,
            result_path=str((run/proposal['label']/'API_RETURN.json').relative_to(ROOT)) if result else proposal['reused_result'],
            price_terms=proposal['price_terms'],conditional_model_cost_RZ=proposal['conditional_cost'],
            sufficient_N=None,G_qualified=None))
    assert sum(r.get('scientific_evaluation_calls',0) for r in rows)<=inputs['actual_new_call_limit']<=2
    feasible=[r for r in rows if r['status']==C.F.PASS and r['conditional_model_cost_RZ']<=C.BUDGET]
    answer=dict(found=bool(feasible),meaning='CONDITIONAL_MODEL_PRICES_AND_Q3_FIXED_MENU_ONLY',
        allocations=[r['label'] for r in feasible],strict_cost_improvement=any(r['conditional_model_cost_RZ']<C.BUDGET for r in feasible),
        minimum_confirmed_model_cost_RZ=min((r['conditional_model_cost_RZ'] for r in feasible),default=None))
    summary=dict(status=completion['status'],source=execution['source'],rows=rows,answer=answer,
        coarse_comparison=inputs['coarse_comparison'],adopted_evaluation=inputs['adopted_evaluation'],
        completion=completion,process_return=read(folder/'PROCESS_RETURN.json'),
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    C.R.write_new(folder/'EXACT_RESULTS.json',summary)
    C.R.write_new(folder/'SUMMARY.json',display(summary))
    C.R.write_new(folder/'C_RESULT_JOIN.json',dict(kind='CONDITIONAL_MODEL_COST_AND_Q3_INDEX_NOT_N_QUALIFICATION',
        source=execution['source'],rows=joins,sufficient_N=None,G_qualified=None))
    print(json.dumps(dict(answer=answer,rows=[dict(rule=r['rule'],cost=r['conditional_model_cost_RZ'],status=r['status'],
        MSE_lower=display(r['MSE_lower']),MSE_upper=display(r['MSE_upper'])) for r in rows]),indent=2))


if __name__=='__main__':summarize()
