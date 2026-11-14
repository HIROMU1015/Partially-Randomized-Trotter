"""Exact consistency of the two saved outputs; no risk/price provider call."""
from fractions import Fraction
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'SUMMARY.json').read_bytes())
inputs=json.loads((HERE/'INPUTS.json').read_bytes())
def exact(x):return Fraction(x['numerator'],x['denominator'])

assert [r['label'] for r in summary['cases']]==['LD1','LD2']
assert summary['completion']['status']=='TWO_RESULTS_SAVED'
assert not summary['completion']['unknown'] and not summary['completion']['errors']
for row,case in zip(summary['cases'],inputs['cases']):
    record=json.loads((HERE/'run01'/('RESULT_'+row['label']+'.json')).read_bytes())
    assert record['scientific_evaluation_calls']==1 and record['old_partial_DP_calls']==0
    assert record['record']['configuration_id']==case['total_configuration_id']
    assert record['record']['allocation']['shots']==case['allocation']['shots']
    assert 0<=exact(row['MSE_lower'])<=exact(row['MSE_upper'])
    dp=row['total_state_dp'];lo,hi=map(exact,(dp['MSE_lower'],dp['MSE_upper']))
    assert hi==exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])
    assert exact(row['tracked_mass'])+exact(dp['residual_probability'])==1
    b0,b1=map(exact,row['moments']['bias_squared']);v0,v1=map(exact,row['moments']['variance'])
    assert max(lo,b0+v0)<=min(hi,b1+v1)
    zero=exact(dp['any_zero']);u0,u1=map(exact,dp['partial_decoder']['unresolved_probability_bounds'])
    assert exact(row['zero_free_partial_tie_probability_bounds']['lower'])==max(0,u0-zero)
    assert exact(row['zero_free_partial_tie_probability_bounds']['upper'])==u1-zero
    assert dp['transition_pairs']<=141559036
    assert all(r['retained_states']<=8192 for r in dp['rounds'][:-1])
    assert record['record']['diagnostic']['deterministic_risk']['q3']['limit'] is None
    assert all(record.get(key) is None for key in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
print('PASS: two fixed canonical/vector outputs, exact mass/loss/moment/tie consistency; no new evaluation.')
