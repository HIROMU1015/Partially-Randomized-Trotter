"""Independent signed-DF matrix reference for the new controlled PR source."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import json
import math
import os
import subprocess
import numpy as np
from qiskit.quantum_info import Statevector
from .track_c_d_internal import OUT,ROOT,common,schedule,evolution,save_json
from .track_c_d_resources import r_part,assemble,wrapper
from .track_c_h4 import finite_rte_distribution
from trotterlib.df_rte_tail import basis_change_unitary
from trotterlib.rte import _make_event,event_unitary

@lru_cache(None)
def basis_rows(name):
    c=common();reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
    return basis_change_unitary(c['preps'][3].tail_extraction.basis_definition(name))[reverse,:]

@lru_cache(None)
def component(index):
    c=common();symbol=c['preps'][3].rte_preparation.symbolic_tail.components[index]
    v=basis_rows(symbol.basis_id);states=np.arange(256);signs=np.ones(256)
    for q in symbol.diagonal_pauli_support:signs*=1-2*((states>>q)&1)
    operator=symbol.coefficient_sign*(v*signs[None,:])@v.conj().T
    assert np.linalg.norm(operator@operator-np.eye(70),2)<1e-10
    return symbol.component_id,operator

def check(task):
    case,h,rare=task;c=common();prep=c['preps'][3];n=math.ceil(h*32/.02);seed=809131+int(rare)
    q,_,choices=r_part(h,n,rare,seed);dist=finite_rte_distribution(prep.exact_rte_lambda_r*h/n,2)
    operators=dict(component(i) for x in choices for i in x['indices']);symbols=prep.rte_preparation.symbolic_tail.components
    state=c['state'].copy()
    for kind,index,t in schedule(case,h):
        if kind=='D':state=evolution(c['spectra'][index],t)@state
        else:
            for choice in choices:
                event=_make_event(choice['indices'],symbols,dist,1 if choice['order']==2 else 0)
                state=event_unitary(event,operators)@state
    state*=np.exp(-1j*h*(prep.constant_coefficient+prep.extracted_identity_coefficient));signal=np.vdot(c['state'],state)
    source=assemble(case,h,1,[q]);reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices]);initial=np.zeros(512,complex);initial[reverse]=c['state'];rows=[]
    for axis,expected in [('X',signal.real),('Y',signal.imag)]:
        v=Statevector(initial.copy()).evolve(wrapper([source],axis).remove_final_measurements(inplace=False)).data.copy()
        measured=float(np.vdot(v[:256],v[:256]).real-np.vdot(v[256:],v[256:]).real)
        rows.append(dict(case=case,h=h,rare=rare,axis=axis,matrix_prediction=float(expected),source_readout=measured,error=abs(measured-expected)))
    return rows

def run():
    path=OUT/'SEMANTICS.json'
    if path.exists():raise RuntimeError('Do not overwrite semantics')
    cases=json.loads((OUT/'DESIGN.json').read_text())['cases'];hs={'original':.02,'order_0321':.02,'vertex_j0':.04,'tuned_0123':.16,'tuned_0321':.16}
    tasks=[(case,hs[case['name']],rare) for case in cases if case['name'] in hs for rare in (False,True)];rows=[]
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed([pool.submit(check,t) for t in tasks]):rows.extend(f.result())
    assert max(x['error'] for x in rows)<1e-10
    save_json(path,dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),rows=rows,maximum_error=max(x['error'] for x in rows),
        interpretation='Independent D spectra plus signed DF involutions and event_unitary versus full joint controlled source; both order strata and X/Y. Float64 checks supplement interval bounds.'))
    print('SEMANTICS_COMPLETE',flush=True)

if __name__=='__main__':run()
