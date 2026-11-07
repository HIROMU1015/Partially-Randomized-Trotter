"""Matched comparison with the historically registered h=.02/r_m PR.

Reuse saved signal grid and measured D prices. Only missing occurrence price
cells are sampled. No extra coefficient fit or Hamiltonian build is performed.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import os
import subprocess
import numpy as np
from .track_c_d_internal import OUT,ROOT,common,save_json
from .track_c_h4 import RVALUES
from .track_c_d_resources import rte_task,det_task,r_key,SAMPLES
from . import track_c_d_qualify as ql
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase

def run():
    target=OUT/'REGISTERED_COMPARISON.json'
    if target.exists():raise RuntimeError('Do not overwrite registered comparison')
    source=json.loads((OUT/'SIGNALS.json').read_text())['rows'];methods={'original':.02,'order_0321':.02,'vertex_j0':.04,'tuned_0123':.16,'tuned_0321':.16}
    rows=[next(r for r in source if r['name']==name and r['m']==m and r['requested_h']==h and r['rate']==RVALUES[m]) for name,h in methods.items() for m in range(18)]
    cells={r_key(r['time'],r['r']):(r['time'],r['r']) for x in rows for r in x['recipe']};tasks=[]
    for key,(t,n) in cells.items():
        for rare in (False,True):
            for j in range(SAMPLES):
                if not (OUT/'rte_price'/f'{key}_rare{int(rare)}_j{j}.json').exists():tasks.append((t,n,rare,j))
    workers=min(len(os.sched_getaffinity(0)),max(len(tasks),len(rows)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(rte_task,t) for t in tasks]):f.result()
        dets={}
        for x in rows:
            path=OUT/'det_price'/f'{x["name"]}_{x["h"]}.json'
            if not path.exists():det_task((x['case'],x['h']))
            dets[(x['name'],x['h'])]=json.loads(path.read_text())
        records={}
        identity=lambda x:(x['name'],x['m'],x['h'],x['rate'])
        for file in ('QUALIFICATION.json','QUALIFICATION_EXTENSION.json'):
            for r in json.loads((OUT/file).read_text())['rows']:records[identity(r)]=r
        bounds=json.loads((OUT/'LOCAL_INTERVALS.json').read_text())['primitive_bounds']
        for r in json.loads((OUT/'QUALIFICATION_EXTENSION.json').read_text())['checked']:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
        missing=[r for r in rows if identity(r) not in records]
        for f in as_completed([pool.submit(ql.qualify,(r,bounds)) for r in missing]):
            r=f.result();records[identity(r)]=r
    save_json(OUT/'REGISTERED_QUALIFICATION.json',dict(rows=[records[identity(x)] for x in rows],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),new_RTE_samples=len(tasks),new_qualification_points=len(missing)))
    samples={}
    for key,(t,n) in cells.items():
        strata={}
        for rare in (False,True):strata[str(rare)]=[json.loads((OUT/'rte_price'/f'{key}_rare{int(rare)}_j{j}.json').read_text()) for j in range(SAMPLES)]
        samples[key]=strata
    priced=[]
    for x in rows:
        q=records[identity(x)];assert q['closed'];d=dets[(x['name'],x['h'])];g={k:np.array(v,float) for k,v in d['groups'].items()};n=x['steps'];det=g['first']+g['last']+n*g['inside']+(n-1)*g['junction']
        expectation=np.zeros(4)
        for recipe in x['recipe']:
            for rs in samples[r_key(recipe['time'],recipe['r'])].values():expectation+=n*rs[0]['probability']*np.mean([r['metrics'] for r in rs],axis=0)
        vectors=np.array([det+expectation+np.array(d['overhead'][a]) for a in ('X','Y')]);N=q['shots_per_axis']
        priced.append(dict(signal=x,qualification=q,price_mean=vectors.tolist(),G=(N*vectors.sum(axis=0)).tolist()))
    cap={r['signal']['m']:r['qualification']['shots_per_axis'] for r in priced if r['signal']['name']=='original'}
    groups=[(name,[r for r in priced if r['signal']['name']==name]) for name in methods]
    groups.append(('shot_capped',[min((r for r in priced if r['signal']['m']==m and r['qualification']['shots_per_axis']<=cap[m]),key=lambda r:r['G'][0]) for m in range(18)]))
    results=[]
    for name,selected in groups:
        reps={str(rare):np.zeros((SAMPLES,4)) for rare in (False,True)}
        for record in selected:
            x=record['signal'];N=record['qualification']['shots_per_axis']
            for recipe in x['recipe']:
                for rare,rs in samples[r_key(recipe['time'],recipe['r'])].items():reps[rare]+=2*N*x['steps']*rs[0]['probability']*np.array([r['metrics'] for r in rs])
        covariance=sum((np.cov(v,rowvar=False,ddof=1)/SAMPLES for v in reps.values()),np.zeros((4,4)))
        phases=[np.angle(complex(r['signal']['raw']['real'],r['signal']['raw']['imag'])) for r in selected];decode=reconstruct_rpe_phase(phases,[1<<m for m in range(18)])
        energy=-decode['final_phase_estimate']/.02;error=abs(energy-common()['energy']);branch=all(abs(r['base_phase_estimate']+.02*common()['energy'])<=.4/(1<<m)+1e-10 for m,r in enumerate(decode['rounds']));assert branch and error<1.5936e-4
        results.append(dict(name=name,G=np.sum([r['G'] for r in selected],axis=0).tolist(),MC_SE=np.sqrt(np.maximum(0,np.diag(covariance))).tolist(),reps={k:v.tolist() for k,v in reps.items()},selected=selected,
            total_shots=sum(2*r['qualification']['shots_per_axis'] for r in selected),all18_closed=True,all_rounds_correct_branch=branch,energy_error=error))
    base=results[0]
    for r in results:
        r['ratio_to_registered']=(np.array(r['G'])/np.array(base['G'])).tolist();r['paired_delta_MC_SE']=np.sqrt(sum(np.var(np.array(r['reps'][s])-np.array(base['reps'][s]),axis=0,ddof=1)/SAMPLES for s in reps)).tolist()
    save_json(target,dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),methods=methods,registered_r_m=RVALUES,results=results,
        interpretation='Registered h=.02 comparator retained. New fixed meshes .16/.04 are checked at the same registered r_m. Capped policy never increases any round shot count. Actual modular A resources and fixed-matrix native success; no global optimality claim.'))
    print('REGISTERED_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':run()
