"""Git-only CSV -> G/paired SE verification; no private input or science imports."""
from pathlib import Path
import csv,json,statistics,math
HERE=Path(__file__).resolve().parent
with (HERE/'wrapper_metrics.csv').open() as f:rows=list(csv.DictReader(f))
model=json.loads((HERE/'resource_map.json').read_text());signals=json.loads((HERE/'signal_metadata.json').read_text())['signals'];by={r['cell']['id']:r for r in signals}
assert len(rows)==532
checked=0
for r in model['rows']:
 a=by[r['cell_id']]['precision_rows'][str(r['epsilon'])]['1.0'];w={'cosine':a['axes']['real']['shots'],'sine':a['axes']['imag']['shots']};pairs={}
 for x in rows:
  if x['cell_id']==r['cell_id'] and x['control']=='symmetric_directional':pairs.setdefault(x['replica'],{})[x['axis']]=int(x['RZ'])
 assert all(set(p)==set(w) for p in pairs.values())
 values=[sum(w[k]*p[k] for k in w) for p in pairs.values()]
 assert math.isclose(statistics.mean(values),r['G_RZ_point'],rel_tol=1e-12,abs_tol=1e-6)
 if len(values)>1:assert math.isclose(statistics.stdev(values)/math.sqrt(len(values)),r['G_RZ_SE'],rel_tol=1e-12,abs_tol=1e-6)
 checked+=1
print(json.dumps({'saved_CSV_rows':len(rows),'G_and_SE_rows_match':checked,'new_scientific_calls':0,'CI_run':False}))
