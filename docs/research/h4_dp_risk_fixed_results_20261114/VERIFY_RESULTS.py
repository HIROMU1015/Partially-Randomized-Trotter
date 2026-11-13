"""Saved two-result checks and exact loss decomposition; never calls risk APIs."""
from fractions import Fraction
from copy import deepcopy
import json
from pathlib import Path
import subprocess

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[2]


def read(path):
    return json.loads(path.read_text())


def saved(source):
    return json.loads(subprocess.check_output(['git','show',source['commit']+':'+source['path']],cwd=ROOT))


def exact(value):
    if isinstance(value,dict) and 'exact' in value:value=value['exact']
    return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)


def encoded(value):
    return dict(numerator=value.numerator,denominator=value.denominator)


def main():
    inputs=read(DIRECTORY/'INPUTS.json');execution=read(DIRECTORY/'run01/EXECUTION.json')
    completion=read(DIRECTORY/'run01/COMPLETION.json')
    original=saved(inputs['inherited_scientific_inputs'])
    assert inputs['cases']==original['cases']
    assert inputs['statistical_contract']==original['statistical_contract']
    assert {k:v for k,v in inputs['options'].items() if k!='state_refinement'}==original['options']
    assert execution['evaluations_requested']==2 and execution['MC_calls']==0
    assert completion['completed']==['LD1','LD2'] and not completion['errors']
    assert len(list((DIRECTORY/'run01').glob('RESULT_*.json')))==2
    expected=inputs['options']['state_refinement']
    assert (expected['maximum_frontier_states'],expected['maximum_transition_pairs'])==(8192,141559036)
    result=dict(source=execution['source'],scope='TWO_FIXED_RAW_MODEL_DIAGNOSTICS_ONLY',cases=[])
    for case in inputs['cases']:
        label=case['label'];outer=read(DIRECTORY/'run01'/('RESULT_'+label+'.json'))
        record=outer['record'];risk=record['diagnostic']['deterministic_risk']
        bundle=saved(case['bundle_source'])
        assert record['source']==execution['source']
        assert record['configuration_id']==bundle['configuration_id']==case['configuration_id']
        assert record['allocation']['shots']==case['allocation']['shots']
        assert record['evaluation_target']==case['evaluation_target']
        contract=deepcopy(record['statistical_contract']);contract.pop('source')
        contract['simulation_policy'].pop('source')
        contract['summary_options']['unresolved_loss'].pop('source')
        assert contract==inputs['statistical_contract']
        options=deepcopy(record['deterministic_input']['options']);options.pop('source')
        options['model_loss_contract'].pop('source')
        assert options==inputs['options']
        assert record['signal_input']==record['deterministic_input']['signal_input']
        assert record['signal_input']['target']==bundle['target']
        assert bundle['target']['truth_proxy_adopted_for_new_simulation'] is False
        for mean,old in zip(record['signal_input']['means'],bundle['means']):
            assert {k:v for k,v in mean.items() if k!='provider_binding'}==old
            assert mean['qualification']=='UNQUALIFIED'
        assert len(record['signal_input']['means'])==len(risk['rounds'])==13
        assert record['deterministic_input']['options']['state_refinement']==expected
        old=saved(dict(commit='6106e4d969bb08e95166cbd2b2cc55b9840c7535',
            path='docs/research/pr_ld1_ld2_deterministic_risk_results_20261113/run01/RESULT_'+label+'.json'))
        legacy=old['record']['diagnostic']['deterministic_risk']
        assert risk['coarse_MSE_upper']==legacy['MSE_upper']
        assert risk['rounds']==legacy['rounds']
        assert risk['unresolved_probability_bounds']==legacy['unresolved_probability_bounds']
        assert risk['scientific_energy_qualification']=='UNKNOWN' and risk['MC_calls']==0
        assert risk['confidence'] is None and risk['unresolved_energy_imputation'] is None
        assert all(record[k] is None for k in ('n_packet','sufficient_N','global_RMSE_bound','G'))
        assert all(outer[k] is None for k in ('sufficient_N','global_RMSE_bound','G','winner'))
        assert not outer['deltaE_addition_performed'] and not outer['formal_energy_precision_test_performed']
        dp=risk['state_refinement']
        if risk['MSE_upper'] is None:
            result['cases'].append(dict(label=label,status='UNKNOWN',reason=risk.get('reason'),
                state_reason=dp.get('reason')))
            print(label+': UNKNOWN preserved; source/vector/null checks PASS');continue
        assert dp['status'] in ('COMPLETE_STATE_LOSS_FAMILY_BOUND','PARTIAL_STATE_BOUND_WITH_RESIDUAL')
        previous_states=1;previous_pairs=0
        for row in dp['rounds']:
            assert exact(row['retained_probability'])+exact(row['absorbed_unresolved_probability'])+exact(row['residual_probability'])==1
            assert row['retained_states']<=8192
            assert row['transition_pairs_cumulative']-previous_pairs==previous_states*row['grouped_count_outcomes']
            previous_pairs=row['transition_pairs_cumulative'];previous_states=row['retained_states']
        assert len(dp['rounds'])==13 and previous_pairs==dp['transition_pairs']<=141559036
        eta=exact(dp['resolved_MSE_contribution']);u=exact(dp['absorbed_unresolved_probability']);r=exact(dp['residual_probability'])
        hi=exact(risk['unresolved_loss']['upper']);lo=exact(risk['unresolved_loss']['lower'])
        bad=exact(dp['residual_loss_upper']);z=exact(dp['any_zero'])
        assert bad==exact(risk['unsafe_loss_upper']) and z==exact(risk['any_zero'])
        assert exact(dp['MSE_upper'])==eta+u*hi+r*bad
        assert exact(dp['MSE_lower'])==eta+u*lo
        assert exact(dp['unresolved_probability_bounds']['lower'])==max(u,z)
        assert exact(dp['unresolved_probability_bounds']['upper'])==u+r
        assert exact(dp['actual_global_tie_probability_bounds']['lower'])==max(Fraction(0),u-z)
        assert exact(dp['actual_global_tie_probability_bounds']['upper'])==u+r-z
        if r:
            assert dp['actual_global_tie_probability'] is None and dp['unresolved_probability'] is None
        else:
            assert exact(dp['actual_global_tie_probability'])==u-z and exact(dp['unresolved_probability'])==u
        coarse=exact(risk['coarse_MSE_upper']);chosen=exact(risk['MSE_upper'])
        assert chosen==min(coarse,exact(dp['MSE_upper']))
        result['cases'].append(dict(label=label,status=risk['status'],
            coarse_MSE_upper=encoded(coarse),DP_MSE_upper=dp['MSE_upper'],selected_MSE_upper=encoded(chosen),
            relative_upper_bound_reduction=encoded((coarse-chosen)/coarse),
            resolved_loss_contribution=encoded(eta),known_unresolved_loss_upper_contribution=encoded(u*hi),
            residual_loss_upper_contribution=encoded(r*bad),absorbed_unresolved_probability=encoded(u),
            residual_probability=encoded(r),zero_probability=encoded(z),
            unresolved_probability_bounds=dp['unresolved_probability_bounds'],
            actual_global_tie_probability_bounds=dp['actual_global_tie_probability_bounds'],
            sufficient_N=None,global_RMSE_bound=None,G=None,winner=None))
        print(label+': PASS unchanged source/input/coarse bound; exact DP mass/loss identities; no pair exhaustion; scientific nulls')
    output=DIRECTORY/'DECOMPOSITION.json'
    if output.exists():
        assert read(output)==result
    else:
        with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print('PASS: saved decomposition exact match; no probability/decoder/risk/MC/cost API imported or called')


if __name__=='__main__':main()
