"""Read original D1 scalar bounds only; no event audit or scientific score update."""
import json
from pathlib import Path
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parents[4]
rows=json.loads((ROOT/'artifacts/track_b_rte_development_d1_result/2026-10-13/v1/resource_map.json').read_text())['rows']
out={}
for r in (1,2):
    group=[v for v in rows if v['steps']==r]
    selectors={
        'weight_M2_path':lambda v:F(v['reference_law_diagnostic_only']['M2'])**r,
        'budget_range_upper':lambda v:F(v['budget']['range_upper']),
        'normalizer_envelope_squared_path':lambda v:F(v['production_step_bounds']['norm_or_envelope_upper'])**(2*r),
        'workspace':lambda v:F(v['workspace_beyond_system']),
        'expected_quantum_calls':lambda v:F(v['expected_quantum_calls'])}
    for k in ('T','CX','1Q'):
        for key in ('per_step_trial_native','per_whole_trial_native','accepted_tail_native','hard_native'):
            selectors[k+'_'+key]=lambda v,k=k,key=key:F(v[key][k])
    values={}
    for name,select in selectors.items():
        ranking={}
        for v in group:ranking.setdefault(select(v),[]).append(v['arm'])
        values[name]=[{'value':str(value),'arms':sorted(arms)} for value,arms in sorted(ranking.items())]
    out['r'+str(r)]=values
path=Path(__file__).with_name('secondary_saved_rankings_v1.json')
with path.open('x') as stream:
    json.dump({'rankings':out,'ascending_numeric_order':True,'tail_hard_are_upper_bounds_not_expected_costs':True,
        'normalizer_envelope_not_actual_variance':True,'input_result_commit':'fc96571881f301435a655f057709df4d6c1678e0',
        'new_science_or_original_reclassification':False,'mandatory_STOP':True},stream,indent=2)
    stream.write('\n')
