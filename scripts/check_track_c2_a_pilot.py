"""Read-only checks of the stopped C2-A artifacts; no scientific backend."""
import ast,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/track_c2_a_pilot/2026-11-06'
summary=json.loads((out/'SUMMARY.json').read_text())
assert summary['all18_cost_computed'] is False
assert summary['status']=='STOP_PRICE_UNCERTAINTY_NO_QUALIFIED_CROSSOVER'
prices=[json.loads(p.read_text()) for p in (out/'prices').glob('*.json')]
requests=json.loads((out/'PRICE_REQUESTS.json').read_text())['requests']
assert len(prices)==len(requests)==60
signals=[json.loads(p.read_text()) for p in (out/'signals').glob('*.json')]
assert len(signals)==8 and sum(len(x['rows']) for x in signals)==84
cells={}
for row in prices:
 cells.setdefault((row['task']['name'],row['r'],row['q']),[]).append(row)
 assert len(row['prices'])==2 and all(len(x)==4 for x in row['prices'])
for rows in cells.values():assert abs(sum(x['conditional_probability'] for x in rows)-1)<1e-12
stop=json.loads((out/'STOP.json').read_text())
assert stop['status']=='MANDATORY_STOP' and stop['automatic_resume'] is False
for p in [root/'src/trottertracks/track_c2_a_pilot.py',Path(__file__),root/'scripts/run_track_c2_a_pilot.py']:ast.parse(p.read_text())
print(json.dumps(dict(status='PASS',signal_cells=84,short_price_cells=60,scientific_calls=0,STOP_preserved=True)))
