"""Check saved exact output identities/algebra; never re-evaluate raw/counts."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[2]


def exact(value):
    if isinstance(value,dict) and 'exact' in value:
        value=value['exact']
    return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)


def saved(source):
    return json.loads(subprocess.check_output(['git','show',source['commit']+':'+source['path']],cwd=ROOT))


inputs=json.loads((DIRECTORY/'INPUTS.json').read_text())
completion=json.loads((DIRECTORY/'run01/COMPLETION.json').read_text())
execution=json.loads((DIRECTORY/'run01/EXECUTION.json').read_text())
assert completion['completed']==['LD1','LD2']
assert execution['evaluations_requested']==2 and execution['MC_calls']==0
assert len(list((DIRECTORY/'run01').glob('RESULT_*.json')))==2
for case in inputs['cases']:
    result=json.loads((DIRECTORY/'run01'/('RESULT_'+case['label']+'.json')).read_text())
    record=result['record']; risk=record['diagnostic']['deterministic_risk']
    bundle=saved(case['bundle_source'])
    assert record['configuration_id']==bundle['configuration_id']==case['configuration_id']
    assert record['allocation']['shots']==case['allocation']['shots']
    assert record['allocation']['allocation_id']==case['allocation']['allocation_id']
    assert record['evaluation_target']==case['evaluation_target']
    assert record['source']==execution['source']
    assert record['signal_input']==record['deterministic_input']['signal_input']
    assert record['signal_input']['target']==bundle['target']
    for mean,original in zip(record['signal_input']['means'],bundle['means']):
        assert {k:v for k,v in mean.items() if k!='provider_binding'}==original
        assert mean['qualification']=='UNQUALIFIED'
    assert len(record['signal_input']['means'])==len(risk['rounds'])==13
    assert record['deterministic_input']['options']['phase_radii']==[1.]*13
    assert record['deterministic_input']['options']['probability_rule']=='EXACT_AFFINE_OF_BINARY64_RAW'
    assert record['statistical_contract']['claim_level']=='RAW_MODEL_DIAGNOSTIC_ONLY'
    assert risk['status']=='CONDITIONAL_BINARY64_DECODER_MODEL_BOUND'
    assert risk['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
    assert risk['decoder_certificate']['checked_pairs']<=inputs['options']['maximum_verification_pairs']
    assert all(r['cells']<=(n['N_X']+1)*(n['N_Y']+1)<=inputs['options']['maximum_round_outcomes']
               for r,n in zip(risk['rounds'],case['allocation']['shots']))
    good=Fraction(1); no_zero=Fraction(1)
    for row in risk['rounds']:
        assert exact(row['good'])+exact(row['unsafe'])==1
        assert 0<=exact(row['zero'])<=exact(row['unsafe'])<=1
        good*=exact(row['good']); no_zero*=1-exact(row['zero'])
    unsafe=1-good; zero=1-no_zero
    assert good==exact(risk['all_good'])
    assert unsafe==sum(map(exact,risk['first_unsafe']))==exact(result['derived']['any_unsafe'])
    assert zero==exact(risk['any_zero'])==exact(risk['unresolved_probability_bounds']['lower'])
    assert unsafe==exact(risk['unresolved_probability_bounds']['upper'])
    assert unsafe-zero==exact(result['derived']['zero_free_global_tie_bounds']['upper'])
    assert exact(risk['MSE_upper'])==exact(risk['terminal_good_MSE_contribution'])+unsafe*exact(risk['unsafe_loss_upper'])
    assert exact(risk['unresolved_loss']['upper'])==exact(inputs['statistical_contract']['summary_options']['unresolved_loss']['upper'])
    assert risk['MC_calls']==0 and risk['confidence'] is None
    assert risk['scientific_energy_qualification']=='UNKNOWN'
    assert risk['unresolved_energy_imputation'] is None
    assert all(record[k] is None for k in ('n_packet','sufficient_N','global_RMSE_bound','G'))
    assert all(result[k] is None for k in ('sufficient_N','global_RMSE_bound','G','winner'))
    assert not result['deltaE_addition_performed'] and not result['formal_energy_precision_test_performed']
    print(case['label']+': PASS saved bindings, exact mass/bound identities, two-call scope and unqualified/null preservation')
print('PASS: no raw/binomial/decoder/risk evaluation, MC or cost API imported or called')
