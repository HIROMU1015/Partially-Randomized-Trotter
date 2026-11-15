"""Project fixed PF6 Q3 records and approved conditional costs; no new risk."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_saved_pf6_budget_caller',ROOT/'scripts/run_pf6_cost_budget_q3_fixed.py')
F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
read, exact, display=F.D.read,F.D.exact,F.D.display
DIRECTORY=ROOT/F.DIRECTORY


def confirmed_minima(rows):
    result={}
    for method in ('FULL_PF','TRUNCATED_PF'):
        selected=[r for r in rows if r['method']==method and r['status']==F.PASS]
        if not selected:
            result[method]=dict(conditional_cost=None,allocations=[])
            continue
        best=min(r['conditional_cost'] for r in selected)
        result[method]=dict(conditional_cost=best,
            allocations=[dict(label=r['label'],base_label=r['base_label'],rule=r['rule'],
                allocation_id=r['allocation']['allocation_id'],shots=r['allocation']['shots'],
                configuration_id=r['configuration_id'],source=r['result_source'])
                for r in selected if r['conditional_cost']==best],
            unresolved_allocations=[r['label'] for r in rows if r['method']==method and r['status'] in ('UNKNOWN','INCONCLUSIVE')],
            scope='MINIMUM_CONFIRMED_MODEL_FEASIBLE_COST_IN_FIXED_CURRENT_WIDTH_MENU',
            sufficient_N=None,G_qualified=None)
    return result


def summarize():
    inputs=read(DIRECTORY/'INPUTS.json');run=DIRECTORY/'run01'
    completion=read(run/'COMPLETION.json');execution=read(run/'EXECUTION.json')
    process_return=read(DIRECTORY/'PROCESS_RETURN.json')
    rows=[];joins=[]
    for b in inputs['baseline']:
        rows.append(dict(label=b['label']+'__BASELINE_N0',base_label=b['label'],method=b['method'],rule='BASELINE_N0',
            configuration_id=b['configuration_id'],total_view_id=b['total_view_id'],
            allocation=b['allocation'],conditional_cost=b['conditional_cost'],unused_budget=None,
            status=b['status'],MSE_lower=b['MSE_lower'],MSE_upper=b['MSE_upper'],result_source=b['Q3_source'],
            cost_source=b['cost_source'],input_disposition='REUSE_BASELINE_NO_NEW_SCIENCE',
            scientific_evaluation_calls=0,evidence_status=b['evidence_status'],
            original_API_return_saved=False,operational_status='HISTORICAL_PARTIAL_SAVE_UNCHANGED'))
    case_map={c['label']:c for c in inputs['cases']}
    prices=F.document(F.B2,F.B2_DIR+'AXIS_TERMS.json')['rows']
    for proposal in inputs['proposals']:
        if proposal['disposition']!='NEW_EVALUATION':
            if proposal['disposition']=='BUDGET_MENU_EMPTY':
                continue
            source_row=next(r for r in rows if r['base_label']==proposal['base_label'] and
                (r['rule']=='BASELINE_N0' if proposal['disposition']=='REUSE_BASELINE' else r['label']==proposal['reused_result']))
            row=dict(source_row,label=proposal['label'],rule=proposal['rule'],allocation=proposal['allocation'],
                input_disposition=proposal['disposition'],scientific_evaluation_calls=0)
            rows.append(row);continue
        case=case_map[proposal['label']];folder=run/case['label']
        result=read(folder/'RESULT.json');receipt=read(folder/'PERSISTED.json');usage=read(folder/'RESOURCE_FINAL.json')
        assert result['source']==execution['source']==usage['source']
        assert result['scientific_evaluation_calls']<=1
        assert result['allocation']==case['allocation']
        row=dict(label=case['label'],base_label=case['base_label'],method=case['method'],rule=case['rule'],
            configuration_id=case['configuration_id'],total_view_id=case['total_view_id'],
            allocation=case['allocation'],conditional_cost=case['conditional_cost'],unused_budget=case['unused_budget'],
            budget=case['generation']['budget'],status=result['status'],MSE_lower=None,MSE_upper=None,
            scientific_evaluation_calls=result['scientific_evaluation_calls'],input_disposition='NEW_EVALUATION',
            result_source=dict(**execution['source'],result_path=str((folder/'RESULT.json').relative_to(ROOT))),
            cost_source=F.ref(execution['source']['commit'],F.INPUT,['cases',inputs['cases'].index(case),'price_terms']),
            original_API_return_saved=receipt['api_return_saved'],operational_status=receipt['operational_status'],
            resources=usage,sufficient_N=None,G_qualified=None)
        assert all(result[k] is None for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
        assert all(result[k]==0 for k in ('MC_calls','old_partial_DP_calls','price_calls'))
        if result['status']=='UNKNOWN':
            row['reason']=result.get('reason')
        else:
            assert read(folder/'API_RETURN.json')==[result['record']]
            risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp']
            coarse=read(folder/'PARTIAL_COARSE.json')
            lo,hi=exact(risk['MSE_lower']),exact(risk['MSE_upper'])
            assert 0<=lo<=hi
            assert F.H.assess(risk,inputs['adopted_evaluation'],execution['source'])['status']==result['status']
            assert hi==min(exact(v) for v in risk['bounds']['upper_candidates'].values())
            assert sum(exact(v['exact']) for v in coarse['first_unsafe'])+exact(coarse['all_good']['exact'])==1
            assert all(exact(v['good']['exact'])+exact(v['unsafe']['exact'])==1 for v in coarse['rounds'])
            assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
            assert 0<=exact(dp['residual_probability'])<=1
            scope=inputs['options']['total_state_refinement']
            assert dp['transition_pairs']<=scope['maximum_transition_pairs']
            assert len(dp['rounds'])==13
            assert all(v['retained_states']<=scope['maximum_frontier_states'] for v in dp['rounds'][:-1])
            assert coarse['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
            assert coarse['decoder_certificate']['checked_pairs']<=inputs['options']['maximum_verification_pairs']
            moments=risk['moments']
            assert all(exact(moments[k][0])<=exact(moments[k][1]) for k in ('mean_energy','second_moment','bias','bias_squared','variance'))
            assert exact(moments['bias_squared'][0])+exact(moments['variance'][0])<=hi
            assert exact(moments['bias_squared'][1])+exact(moments['variance'][1])>=lo
            row.update(MSE_lower=display(risk['MSE_lower']),MSE_upper=display(risk['MSE_upper']),
                moments=display(moments),decoder_closure=coarse['decoder_certificate'],
                residual_mass=display(dp['residual_probability']),residual_loss_upper=display(dp['residual_loss_contribution_upper']),
                any_zero=display(dp['any_zero']),total_decoder_unresolved='NONE_BY_TOTAL_DEFINITION',
                guard_failed_residual_mass=display(dp['residual_principal_guard_failed_probability']),
                transition_pairs=dp['transition_pairs'],per_round=display(dp['rounds']),
                numerical_allowance=dp['transition_allowance'],
                gates=dict(untouched_API=True,first_unsafe_partition_exact=True,round_mass_exact=True,
                    DP_mass_checked_in_kernel=True,residual_loss_accounted=True,
                    ordered_bounds_and_moments=True,adopted_exact_threshold=True,finite_scope=True))
        joins.append(dict(label=case['label'],source=execution['source'],model_status=row['status'],
            configuration_id=case['configuration_id'],total_view_id=case['total_view_id'],binding_id=case['binding_id'],
            signal_packet=case['signal_packet'],allocation=case['allocation'],conditional_cost=case['conditional_cost'],
            API_return_path=str((folder/'API_RETURN.json').relative_to(ROOT)),
            rows=[dict(m=t['m'],axis=t['axis'],N=t['N'],physical_id=prices[t['price_source']['selector'][-1]]['physical_id'],
                original_candidate_id=prices[t['price_source']['selector'][-1]]['original_candidate_id'],
                total_candidate_id=prices[t['price_source']['selector'][-1]]['total_candidate_id'],
                price_source=t['price_source'],price_class=t['price_class']) for t in case['price_terms']],
            sufficient_N=None,G_qualified=None))
        rows.append(row)
    calls=sum(r['scientific_evaluation_calls'] for r in rows)
    assert calls<=inputs['actual_new_call_limit']<=12
    summary=dict(status=completion['status'],execution_source=execution['source'],
        fixed_plan=inputs['fixed_plan'],fixed_prices=inputs['fixed_prices'],adopted_evaluation=display(inputs['adopted_evaluation']),
        baseline_count=6,new_scientific_calls=calls,rows=rows,conditional_minima=confirmed_minima(rows),
        completion=completion,process_return=process_return,
        interpretation='CONDITIONAL_FIXED_COMPILER_CLOSURE_AND_Q3_MODEL_MENU_NOT_SUFFICIENT_N_OR_GLOBAL_OPTIMUM',
        display_precision=25,exact_bounds='API_RETURN and original historical components; displayed decimals only',
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    F.R.write_new(DIRECTORY/'SUMMARY.json',summary)
    F.R.write_new(DIRECTORY/'C_RESULT_JOIN.json',dict(kind='CONDITIONAL_COST_AND_MODEL_RESULT_INDEX_NOT_N_QUALIFICATION',
        source=execution['source'],candidates=joins,sufficient_N=None,G_qualified=None))
    with (DIRECTORY/'RESULTS.csv').open('x') as stream:
        fields=['label','rule','conditional_cost','MSE_lower','MSE_upper','status','original_API_return_saved','N_X','N_Y']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row.get(k) for k in fields[:-2]},
                'N_X':str([s['N_X'] for s in row['allocation']['shots']]),
                'N_Y':str([s['N_Y'] for s in row['allocation']['shots']])})
    print(json.dumps(dict(status=summary['status'],calls=calls,
        minima={k:v['conditional_cost'] for k,v in summary['conditional_minima'].items()}),indent=2))


if __name__=='__main__':summarize()
