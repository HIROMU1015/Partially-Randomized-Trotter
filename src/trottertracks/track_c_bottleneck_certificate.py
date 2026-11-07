"""Uniform source/native bounds for singleton Gaussian replacements.

Single-particle interval columns bound the preserved Fock involutions; no
projection of native intermediate gates. Native primitives reuse the existing
all-context interval machinery. Old SOURCE/error evidence is never rewritten.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import json
import math
import os
import subprocess
import numpy as np
import mpmath as mp
from .track_c_bottleneck import OUT,ROOT,PRIOR,saved
from .track_c_bottleneck_basis import configuration
from .track_c_h4 import common,save_json
from . import track_c_compiler_a_18round as ca
from . import track_c_source_guarantee as sg

@lru_cache(None)
def one_particle(operations):
    mp.iv.prec=192;u=[[sg.C(int(i==j)) for j in range(8)] for i in range(8)]
    for name,wires,params in operations:
        if name=='rz':
            row=wires[0];phase=sg.expi(sg.I(params[0]));u[row]=[phase*x for x in u[row]]
        elif name=='xx_plus_yy':
            matrix=ca.source_matrix(ca.spec(name,params));i,j=wires;assert abs(i-j)==1
            a,b=u[i][:],u[j][:]
            u[i]=[matrix[1][1]*x+matrix[1][2]*y for x,y in zip(a,b)]
            u[j]=[matrix[2][1]*x+matrix[2][2]*y for x,y in zip(a,b)]
        else:raise ValueError(name)
    return u

def ops_key(operations):return tuple((g.name,tuple(q),tuple(float(v) for v in g.params)) for g,q in operations)

def component_interval(index):
    mp.iv.prec=192;c=common();registry,specs,_,_=configuration();old=c['preps'][3].rte_preparation.component_specs[index];new=specs[index]
    original=c['preps'][3].rte_preparation.basis_registry.definition(old.basis_id);replacement=registry.definition(new.basis_id)
    a=one_particle(ops_key(original.runtime_operations));b=one_particle(ops_key(replacement.runtime_operations));errors=[]
    for target,source in enumerate(old.diagonal_pauli_support):
        errors.append(sg.up(mp.iv.sqrt(sum((abs(a[i][source]-b[i][target])**2 for i in range(8)),sg.I(0)))))
    # On the four-electron sector, ||dGamma(Pu-Pv)|| <=4 ||Pu-Pv||;
    # ||Pu-Pv|| <=2||u-v||, and Z=I-2n. Products telescope.
    operator=sg.up(16*sum((sg.I(v) for v in errors),sg.I(0)))
    return dict(component_id=old.component_id,column_error_uppers=errors,involution_error_upper=operator,old_basis=old.basis_id,new_basis=new.basis_id,
        proof='Adjacent Gaussian gates conserve number; N_e=4, 2*N_e*2*column_error per Z; ZZ via telescoping. Source intervals, not native sector projection.')

def basis_error(bounds):
    values=[];registry,_,_,_=configuration()
    for d in registry._definitions.values():
        counts=ca.basis_counter(d.runtime_operations,{})
        values.append(sg.add(*[sg.mul(n,bounds[k]) for k,n in counts.items()]))
    return max(values)

def supplement(q,x,certificate):
    mp.iv.prec=192;applications=3*x['RTE_events']
    # Uniform event-by-event bound also covers run-dependent implementation
    # choices and the full-circuit fallback. Retaining old native/source disks
    # is conservative; these are error bounds, not double-counted gate prices.
    extra=sg.up(2*mp.iv.sqrt(sg.I(2))*sg.I(applications)*(sg.I(certificate['max_involution_error'])+sg.I(certificate['max_native_basis_pair_error'])))
    rho=sg.lower(sg.I(q['rho_lower'])-sg.I(extra));ratio=sg.I(extra)/sg.I(q['rho_lower'])
    phase=sg.up(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))) if extra<q['rho_lower'] else None
    beta=None if phase is None else sg.lower(sg.I(q['beta_stat_lower'])-sg.I(phase))
    return {**q,'rho_original':q['rho_lower'],'beta_original':q['beta_stat_lower'],'rho_lower':rho,'beta_stat_lower':beta,
        'replacement_raw_disk_upper':extra,'closed':bool(q['closed'] and rho>0 and beta is not None and beta>0)}

def run():
    path=OUT/'REPLACEMENT_CERTIFICATE.json'
    if path.exists():raise RuntimeError('Do not overwrite replacement certificate')
    configuration();catalogue=ca.inventory()['catalogue']
    for d in configuration()[0]._definitions.values():ca.basis_counter(d.runtime_operations,catalogue)
    workers=len(os.sched_getaffinity(0));contexts=[];unique={};column=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed(pool.submit(component_interval,i) for i in range(len(configuration()[1]))):column.append(f.result())
        for f in as_completed(pool.submit(ca.context_compile,(mask,catalogue)) for mask in range(128)):
            result=f.result();contexts.append({k:result[k] for k in ('mask','context','post_unroll','symbolic_P_passthrough')})
            for record in result['records']:unique[record['lowering_sha256']]=record
        checked=list(pool.map(ca.interval_check,unique.values()))
    bounds=json.loads((PRIOR/'LOCAL_INTERVALS.json').read_text())['primitive_bounds'].copy()
    for r in checked:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
    save_json(OUT/'REPLACEMENT_LOCAL_INTERVALS.json',dict(contexts=contexts,records=list(unique.values()),checked=checked,primitive_bounds=bounds))
    certificate=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),components=column,max_involution_error=max(x['involution_error_upper'] for x in column),
        max_native_basis_pair_error=basis_error(bounds),contexts=len(contexts),unique_native_lowerings=len(unique),
        sampling_law='Unchanged; error bound uniform over every logical component and every trajectory. Ideal native, saved H4 matrix, fresh bits.')
    save_json(path,certificate);print('REPLACEMENT_CERTIFICATE_COMPLETE',flush=True)

if __name__=='__main__':run()
