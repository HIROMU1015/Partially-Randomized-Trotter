#!/usr/bin/env python3
"""Additional saved-only spectral-selection/export audit, no project/Qiskit imports."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def mat(x):return np.asarray(x['real'])+1j*np.asarray(x['imag'])
def require(x,m):
    if not x:raise ValueError(m)
def mnorm(x,nu):return max(abs(sum(x[list(s)])) for s in itertools.combinations(range(len(x)),nu))
def gamma(u):
    occ=[[i for i in range(len(u)) if s>>i&1] for s in range(1<<len(u))]
    return np.array([[np.linalg.det(u[np.ix_(a,b)]) if a and len(a)==len(b) else 1. if not a and not b else 0. for b in occ] for a in occ],complex)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    require(not a.output.exists(),'no overwrite')
    result=json.loads((a.run/'result.json').read_text());irs=json.loads((a.run/'native_ir.json').read_text());by={x['label']:x for x in irs}
    recovery=json.loads((a.run/'export_recovery_audit.json').read_text());parent=ROOT/recovery['parent_run'];original=[]
    for name,h in recovery['parent_files'].items():require(hashlib.sha256((parent/name).read_bytes()).hexdigest()==h,'parent hash')
    for c in result['contexts']:
        kind=c.get('context','coefficient_feasibility');check=json.loads((parent/f"checkpoint_{c['track']}_{kind}.json").read_text());original.extend(check.pop('native_ir'));require(check==c,'checkpoint context')
    require(original==irs,'IR export object/order preserved')
    candidates=selections=0;maximum=0.
    families={'native_exact':{'exact'},'N1_min_basis_rz':{'exact','cluster'},'N1_min_basis_cx':{'exact','cluster'},'shifted_cutoff_min_basis_rz':{'exact','shifted_cutoff'},'shifted_cutoff_min_basis_cx':{'exact','shifted_cutoff'},'zero_cutoff_min_basis_rz':{'exact','zero_cutoff'}}
    for context in result['contexts']:
        if context['track']!='N1':continue
        lists=context['generator_candidates'];nu=context['particles']
        for i,cs in enumerate(lists):
            eta,u=np.linalg.eigh(mat(context['input_factors'][i]));w=context['weights'][i]
            for c in cs:
                g=mat(c['g']);d=u.conj().T@g@u;et=np.diag(d).real
                require(np.linalg.norm(d-np.diag(et),2)<1e-10,'spectral approximation commutation')
                b=abs(w)*mnorm(eta-et,nu)*(mnorm(eta,nu)+mnorm(et,nu));e=abs(b-c['bound']);maximum=max(maximum,e);require(e<1e-10,'analytic coefficient bound')
                cost=c['basis_cost'];ir=by[cost['label']];require(np.linalg.norm(gamma(mat(c['frame']))-mat(ir['logical_reference']),2)<1e-10,'absolute exterior frame')
                counts=Counter(x['name'] for x in ir['operations'])
                for metric in ('rz','cx','sx','x'):require(counts[metric]==cost[metric],'basis metric')
                require(len(ir['operations'])==cost['size'],'basis size');candidates+=1
        for selected in context['selected']:
            name=selected['name']
            if name not in families:continue
            metric='cx' if name.endswith('_cx') else 'rz';secondary='rz' if metric=='cx' else 'cx'
            combos=[cs for cs in itertools.product(*lists) if all(c['family'] in families[name] for c in cs) and sum(c['bound'] for c in cs)<=context['model_budget']]
            best=min(combos,key=lambda cs:(sum(c['basis_cost'][metric] for c in cs),sum(c['basis_cost'][secondary] for c in cs),sum(c['bound'] for c in cs)))
            for g,c in zip(selected['factors'],best,strict=True):require(np.linalg.norm(mat(g)-mat(c['g']),2)<1e-10,'finite selector choice')
            require(abs(selected['model_bound']-sum(c['bound'] for c in best))<1e-12,'selected total bound');selections+=1
    out={'status':'SAVED_CONSTRUCTION_SELECTION_AND_EXPORT_PASS','spectral_candidates_checked':candidates,'finite_selected_constructions_checked':selections,'maximum_analytic_bound_difference':maximum,'export_ir_objects_checked':len(irs),'parent_files_checked':len(recovery['parent_files']),'new_scientific_runs':0}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
