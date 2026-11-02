"""Existing H4 JSON only; no science arrays, sampler or compiler."""
from pathlib import Path
import json,csv,hashlib,statistics,math
HERE=Path(__file__).resolve().parent
BASE=Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h48-preflight-v1')
SIGNAL=BASE/'runs/h4-signal-03/cost_exploration_0000'
COST=BASE/'runs/h4-cost-01'
SOURCE='217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
def read(p):return json.loads(p.read_text())
def write(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
terminal=read(COST/'execution_terminal.json')
assert terminal['first_stop_reason'] is None and terminal['all_dispatched_reaped'] and not terminal['cleanup_errors']
signals=read(SIGNAL/'signal_summary.json')['signals'];groups=[read(p) for p in sorted(COST.glob('cost_*/cost_result.json'))]
assert len(signals)==16 and len(groups)==264
calls={k:sum(x['calls_attempted'][k] for x in groups) for k in groups[0]['calls_attempted']}
assert calls['compile']==532 and calls['trajectory']==256 and calls['occurrence']==960 and calls['control_probe']==72
rows=[]
for g in groups:
 for r in g['rows']:rows.append({**{k:r[k] for k in ['cell_id','phase','replica','seed','axis','control','event_digest','compiler_hash','fingerprint','classical_wall_seconds','worker_peak_rss_bytes']},**r['metrics']})
with (HERE/'wrapper_metrics.csv').open('w',newline='') as f:
 out=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');out.writeheader();out.writerows(rows)
write('signal_metadata.json',{'source':SOURCE,'signals':signals,'scope':'16 fixed anchors, not old H4 full grid'})
comparisons=[];all_rows=[]
for eps in (0.05,0.01,0.005,0.001):
 eligible=[]
 for s in signals:
  a=s['precision_rows'][str(eps)]['1.0']
  if not a['cost_acquisition_eligible']:continue
  cell=s['cell'];rs=[r for r in rows if r['cell_id']==cell['id'] and r['control']=='symmetric_directional']
  pairs={}
  for r in rs:pairs.setdefault(r['replica'],{})[r['axis']]=r
  assert all(set(p)=={'cosine','sine'} for p in pairs.values())
  shot={'cosine':a['axes']['real']['shots'],'sine':a['axes']['imag']['shots']}
  values=[sum(shot[axis]*p[axis]['RZ'] for axis in shot) for p in pairs.values()]
  entry={'epsilon':eps,'cell_id':cell['id'],'method':cell['method'],'N_total':a['N_total'],'G_RZ_point':statistics.mean(values),'G_RZ_SE':statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None,'n':len(values),'population_mean_certified':False}
  eligible.append(entry);all_rows.append(entry)
 pf=min((r for r in eligible if r['method']=='B1'),key=lambda x:(x['G_RZ_point'],x['cell_id']))
 pr=min((r for r in eligible if r['method']=='B2'),key=lambda x:(x['G_RZ_point'],x['cell_id']),default=None)
 comparisons.append({'epsilon':eps,'PF_point':pf,'PR_point_in_fixed_panel':pr,'PR_PF_RZ_ratio':pr['G_RZ_point']/pf['G_RZ_point'] if pr else None,'point_minimum_not_population_winner':True})
write('resource_map.json',{'rows':all_rows,'comparisons':comparisons,'state_preparation_cost_P':0,'old_H4_dataset_or_H6_results_pooled':False,'common_DF_policy_equivalence_verified':False})
write('result_summary.json',{'source_commit':SOURCE,'signal_candidates':16,'cost_groups':264,'wrappers':532,'actual_calls':calls,'terminal':terminal,'comparisons':comparisons,'target':{'model':'linear H4','geometry_angstrom':1.,'basis':'sto-3g','fixed_DF_rank':12,'system_modes':8,'sector_dimension':36,'T':.8,'delta_window':[.1,.8],'PR_prefix':3},'compiler':'Qiskit1.3.0/basis rz,sx,x,cx/opt1/seed17/no topology','control':'symmetric_directional, measured full wrapper','scope':'fixed anchor panel, not full H4 optimum or size-scaling conclusion','population_mean_bound':None})
write('input_receipt.json',{'inputs':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [SIGNAL/'signal_summary.json',COST/'execution_terminal.json',*sorted(COST.glob('cost_*/cost_result.json'))]],'scientific_arrays_read':0,'new_scientific_calls':0})
print(json.dumps({'groups':len(groups),'wrappers':len(rows),'calls':calls,'comparisons':[{k:r[k] for k in ('epsilon','PR_PF_RZ_ratio')} for r in comparisons]}))
