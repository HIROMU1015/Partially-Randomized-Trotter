"""Independent arithmetic using published CSV/JSON only; no native run needed."""
import csv,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def load(p):return json.loads(p.read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=2e-14,abs_tol=1e-10),(a,b)
def main():
 with (HERE/'pilot_wrapper_metrics_v1.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==16 and {int(x['replica']) for x in rows}==set(range(8))
 grouped={i:{x['axis']:int(x['RZ']) for x in rows if int(x['replica'])==i} for i in range(8)}
 assert all(set(x)=={'cosine','sine'} for x in grouped.values())
 analysis=load(HERE/'cost_contribution_analysis_v1.json');public=load(HERE/'pilot_metadata_evidence_v1.json')
 assert len(public['groups'])==8 and len(public['wrapper_rows'])==16
 prior=load(ROOT/'artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/selected_trajectory_cost_evidence_v1.json')['records']
 zero=[x for x in prior if x['cell_id']==analysis['cell_id'] and x['phase']=='confirmation']
 assert len(zero)==32 and all(x['n2']==x['n4']==0 for x in zero)
 p2=zero[0]['finite_distribution']['order_probabilities'][1];pi=-math.expm1(16*math.log1p(-p2));close(pi,analysis['s2_probability'])
 checks=0
 for axis in ('cosine','sine'):
  vals=[grouped[i][axis] for i in range(8)];d=analysis['s2_axis_metric_statistics'][axis]['RZ'];close(sum(vals)/8,d['mean']);close(statistics.stdev(vals)/math.sqrt(8),d['standard_error']);checks+=2
 for e in analysis['comparisons']:
  n=e['saved_shot_counts'];v0=[sum(n[a]*x['metrics'][a]['RZ'] for a in ('cosine','sine')) for x in zero];v2=[sum(n[a]*grouped[i][a] for a in ('cosine','sine')) for i in range(8)]
  mean=(1-pi)*(sum(v0)/32)+pi*(sum(v2)/8);se=math.sqrt((1-pi)**2*statistics.variance(v0)/32+pi**2*statistics.variance(v2)/8)
  close(mean,e['stratified_G_point_estimate']);close(se,e['stratified_G_standard_error']);close(mean/e['PF_G'],e['PR_PF_ratio_stratified']);checks+=3
  assert e['population_mean_certified'] is False and e['exact_tail_bound'] is None
 native=load(HERE/'conditional_resource_statistics.json')
 for axis in ('cosine','sine'):
  for k,d in analysis['s2_axis_metric_statistics'][axis].items():close(d['mean'],native['axes'][axis][k]['mean']);close(d['standard_error'],native['axes'][axis][k]['standard_error']);checks+=2
 assert analysis['compiled_RZ_population_bound'] is None
 record=dict(status='LOCAL_PUBLIC_METADATA_ARITHMETIC_PASS',checks=checks,groups=8,wrappers=16,old_s0=32,new_trajectory_calls=0,new_compile_calls=0,CI=False,population_bound_proven=False)
 print(json.dumps(record,sort_keys=True))
if __name__=='__main__':main()
