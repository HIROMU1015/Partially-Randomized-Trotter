"""Independent short trajectory matrix versus measured controlled circuit.

Use retained DF component dense references and original D block spectra,
without obtaining reference matrices from the constructed source circuit.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from qiskit.quantum_info import Statevector
from .track_c_mixed_formula import OUT,ROOT,sequence,common,evolution,save_json
from .track_c_mixed_price import NAMES,event_parts,circuit_parts,wrapper
from .track_c_h4 import finite_rte_distribution
from trotterlib.rte import _make_event,event_unitary
from trotterlib.df_rte_tail import component_dense_operator

@lru_cache(None)
def component(index):
    c=common();prep=c['preps'][3];symbol=prep.rte_preparation.symbolic_tail.components[index]
    retained=next(x for x in prep.tail_extraction.components if x.component_id==symbol.component_id)
    full=component_dense_operator(prep.tail_extraction,retained)*symbol.coefficient_sign
    reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
    sector=full[np.ix_(reverse,reverse)]
    assert np.linalg.norm(sector.conj().T@sector-np.eye(70),2)<1e-10
    return symbol.component_id,sector

def check(task):
    rare,rate=task;path=OUT/'paired_price'/f'n1_r{rate}_rare{int(rare)}_j0.json';raw=json.loads(path.read_text())
    h=raw['h'];c=common();prep=c['preps'][3];tails,p,choices=event_parts(h,1,rate,rare,raw['seed'])
    assert choices==raw['event_choices'];n=max(1,math.ceil(h/2*rate/.02));dist=finite_rte_distribution(prep.exact_rte_lambda_r*h/(2*n),2)
    symbols=prep.rte_preparation.symbolic_tail.components;operators=dict(component(i) for x in choices for i in x['indices'])
    events=[event_unitary(_make_event(x['indices'],symbols,dist,1 if x['order']==2 else 0),operators) for x in choices]
    reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices]);initial=np.zeros(512,complex);initial[reverse]=c['state']
    rows=[]
    for name in NAMES:
        state=c['state'].copy();cursor=0
        for kind,index,t in sequence(name,h):
            if kind=='D':state=evolution(c['spectra'][index],t)@state
            else:
                for event in events[cursor:cursor+n]:state=event@state
                cursor+=n
        state*=np.exp(-1j*h*(prep.constant_coefficient+prep.extracted_identity_coefficient))
        signal=complex(np.vdot(c['state'],state));parts=circuit_parts(name,h,1,tails)
        for axis,expected in [('X',signal.real),('Y',signal.imag)]:
            q=wrapper(parts,axis).remove_final_measurements(inplace=False)
            v=Statevector(initial.copy()).evolve(q).data.copy()
            measured=float(np.vdot(v[:256],v[:256]).real-np.vdot(v[256:],v[256:]).real)
            rows.append(dict(name=name,rare=rare,rate=rate,axis=axis,matrix_prediction=float(expected),source_measurement=measured,error=abs(measured-expected)))
    return rows

def run():
    path=OUT/'SOURCE_SEMANTICS.json'
    if path.exists():raise RuntimeError('Do not overwrite source semantics')
    tasks=[(rare,rate) for rare in (False,True) for rate in (1,32)];rows=[]
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed([pool.submit(check,t) for t in tasks]):rows.extend(f.result())
    assert max(x['error'] for x in rows)<1e-10
    save_json(path,dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),checks=rows,maximum_error=max(x['error'] for x in rows),
        interpretation='Float64 independent D spectra plus signed DF component/event_unitary reference versus full measured controlled joint source. Tests X/Y sign and relative scalar/event phases; not an interval guarantee.'))
    print('SOURCE_SEMANTICS_COMPLETE',flush=True)

if __name__=='__main__':run()
