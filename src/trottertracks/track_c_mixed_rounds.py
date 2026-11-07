"""Predeclared 18-round continuation of the mixed-error mechanism screen.

No coefficient tuning. Keep two candidates with lower operator error; retain
all strong controls. Original screen raw files are never rewritten.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import csv
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time
from .track_c_mixed_formula import OUT,CACHE_SHA,H_VALUES,RATES,signal_task,save_json,ROOT

NAMES=('mixed_2MN','mixed_zero_D','refinement_control','registered_S2','partial_Y4','full_S2','full_Y4')

def run():
    directory=OUT/'full_rounds';directory.mkdir(exist_ok=False)
    cache=Path(os.environ['TRACK_C_H4_COMMON_CACHE'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    paths=['src/trottertracks/track_c_mixed_formula.py',__file__.split(str(ROOT)+'/')[1]]
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    for p in paths:assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show',source+':'+p],cwd=ROOT)
    # The .005 baseline is required to match refinement_control(h=.01) exactly.
    tasks=[(name,h) for name in NAMES for h in ((.005,*H_VALUES) if name=='registered_S2' else H_VALUES)]
    start=time.monotonic();rows=[];workers=min(len(tasks),len(os.sched_getaffinity(0)))
    worker=partial(signal_task,rounds=tuple(range(18)),output_subdir='full_rounds',fair_counts=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):rows.extend(f.result())
    rows.sort(key=lambda x:(x['name'],x['requested_h'],x['m'],x['rate']))
    save_json(OUT/'FULL_ROUNDS.json',dict(rows=rows))
    choices=[]
    for name in NAMES:
        for m in range(18):
            eligible=[x for x in rows if x['name']==name and x['m']==m and x['feasible']]
            choices.append(dict(name=name,m=m,best=None if not eligible else min(eligible,key=lambda x:x['primitive_shot_surrogate'])))
    pairs=[]
    for name in ('mixed_2MN','mixed_zero_D'):
        for x in [r for r in rows if r['name']==name]:
            ref=next(r for r in rows if r['name']=='refinement_control' and r['requested_h']==x['requested_h'] and r['m']==x['m'] and r['rate']==x['rate'])
            pairs.append(dict(name=name,m=x['m'],h=x['h'],requested_h=x['requested_h'],rate=x['rate'],
                candidate_feasible=x['feasible'],reference_feasible=ref['feasible'],
                candidate_N=x['shots_per_axis'],reference_N=ref['shots_per_axis'],
                candidate_PF=x['PF_phase'],reference_PF=ref['PF_phase'],
                candidate_rho=x['rho_lower'],reference_rho=ref['rho_lower'],
                same_D_R_count=x['deterministic_exponentials']==ref['deterministic_exponentials'] and x['RTE_events']==ref['RTE_events']))
    save_json(OUT/'FULL_PAIRS.json',dict(rows=pairs))
    columns=['name','requested_h','h','m','rate','steps','PF_phase','finite_phase','ideal_radius','log_normalizer','rho_lower','beta_stat','shots_per_axis','feasible','finite_error_upper','numerical_disk','input_disk','deterministic_exponentials','RTE_events','primitive_shot_surrogate']
    with (OUT/'FULL_ROUNDS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    for p,h in hashes.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    save_json(OUT/'FULL_SUMMARY.json',dict(status='CONDITIONAL_SOURCE_SIGNAL_COMPLETE_NATIVE_ERROR_PENDING',source_commit=source,source_sha256=hashes,
        input_cache_sha256=CACHE_SHA,workers=workers,wall_seconds=time.monotonic()-start,
        count=len(rows),unique_actual_conditions=len({(r['name'],r['h'],r['m'],r['rate']) for r in rows}),choices=choices,
        fair_count='Adjacent identical D blocks fused across every step boundary for all families; original screen remains unfused.',
        guarantee='Float64 diagnostic, not a new 18-round native success theorem; new-angle compiler disk has not been qualified.',
        new_SCF=0,new_DF_fit=0,new_ground_solve=0))
    print('FULL_ROUNDS_COMPLETE',len(rows),flush=True)

if __name__=='__main__':run()
