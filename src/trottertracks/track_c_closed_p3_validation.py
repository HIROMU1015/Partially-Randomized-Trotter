"""Stage 1--3 correctness and finite-bit qualification for Closed P3 integration."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
import numpy as np
from .track_c_closed_p3 import *
from .algorithm_codesign.return_aggregation import ReturnKernel
from .track_c_d_semantics import component
from trotterlib.rte import finite_taylor_operator


def synthetic():
    paulis=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.])]
    p=(F(1,5),F(3,10),F(1,2));x=F(1,3);rows=[]
    for operators in (paulis,[paulis[0],-paulis[2],np.eye(2)]):
        A=sum(float(v)*q for v,q in zip(p,operators));g=CachedCanonical(p,x,3,'closed_P3_tail',root_bits=BITS,probability_bits=BITS,eta=ETA,rho=RHO)
        kernel=ReturnKernel(p,x,3);cp=certify(g)
        assert kernel.parent(()).a==1-kernel.chi*x*x/2
        for sigma in (-1,1):
            mean=np.zeros_like(A);proposal=F(0);moment=F(0);maximum=F(0);count=0
            for gi,group in enumerate(g.groups):
                words=[()] if group.degree==0 else [(group.first,k) for k in range(3) if k!=group.first]
                for word in words:
                    children=range(3) if not word else [i for i in range(3) if i!=word[0]]
                    for child in children:
                        e=g.event(gi,tuple(word),child);assert e['proposal']*e['weight']==e['coefficient']
                        theta,_=angle(e['ratio']);product=np.eye(2,dtype=complex)
                        for label in reversed(word):product=operators[label]@product
                        v=(1j**e['phase_i_power'])*(np.cos(theta)*np.eye(2)-1j*sigma*np.sin(theta)*operators[child])@product
                        mean+=float(e['coefficient'])*v;proposal+=e['proposal'];moment+=e['proposal']*e['weight']**2;maximum=max(maximum,e['weight']);count+=1
                        if word:
                            par=kernel.parent(word);assert par.angle_tangent==e['ratio']
            target=finite_taylor_operator(A,sigma*float(x),2)
            error=float(np.linalg.norm(mean-target,2));assert error<1e-12 and proposal==1 and maximum<=F.from_float(cp['Wmax']) and moment<=F.from_float(cp['M2_upper'])
            rows.append(dict(sigma=sigma,identity_label=bool(np.allclose(operators[-1],np.eye(2))),events=count,operator_error=error,proposal_sum=str(proposal),observed_exact_M2=str(moment),certified_M2_upper=cp['M2_upper']))
        # Cached map is bit-for-bit identical to the upstream G7 map.
        ds=DigitalSampler(g)
        for seed in range(16):assert ds.sample(BitStream(seed))==g.sample(BitStream(seed))
    return rows


@lru_cache(None)
def h4_operators():return np.stack([component(i)[1] for i in range(218)])


def mean_from_groups(g):
    operators=h4_operators();A=np.einsum('i,ijk->jk',np.array(list(map(float,g.p))),operators);eye=np.eye(70,dtype=complex)
    theta,_=angle(g.groups[0].ratio)
    mean=float(g.roots[0].midpoint)*(math.cos(theta)*eye-1j*math.sin(theta)*np.einsum('i,ijk->jk',np.array(list(map(float,g.root_child))),operators))
    for group,root in zip(g.groups[1:],g.roots[1:]):
        j=group.first;conditional=(A-float(g.p[j])*operators[j])/(1-float(g.p[j]));theta,_=angle(group.ratio)
        mean+=float(root.midpoint)*(-math.cos(theta)*(operators[j]@conditional)+1j*math.sin(theta)*(conditional@operators[j]@conditional))
    return mean,A


def cell_check(cell):
    t,n=cell;start=time.process_time();g=generator(t,n);certificate=certify(g);mean,A=mean_from_groups(g)
    target=finite_taylor_operator(A,float(g.x),2);error=float(np.linalg.norm(mean-target,2))
    assert error<1e-11
    # Digital mapping and exact correction, with nonempty parents forced too.
    sampler=DigitalSampler(g);samples=[]
    for seed in range(8):
        e=sampler.sample(BitStream(seed));assert e==g.sample(BitStream(seed));samples.append(dict(word=list(e['word']),child=e['child'],pi_times_weight_equal_coefficient=e['proposal']*e['weight']==e['coefficient']))
    record=dict(t=t,n=n,certificate=certificate,operator_mean_error=error,source_cpu_seconds=time.process_time()-start,sampled_map_checks=samples)
    save_json(OUT/'cell_checks'/(cellkey(t,n)+'.json'),record);return record


def stage_one():
    target=OUT/'STAGE1.json'
    if target.exists():raise RuntimeError('Preserve completed stage')
    if provenance()['common_sha256']!='68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2':
        from .track_c_closed_p3_inputs import model_identity
        saved=json.loads((ROOT/'artifacts/track_c_closed_p3/2026-11-08/INPUT_REPLAY.json').read_text())
        assert model_identity(common())==saved['scientific_model_sha256']
    c=common();assert len(probabilities())==218
    artificial=synthetic();ops=h4_operators();A=np.einsum('i,ijk->jk',np.array(list(map(float,probabilities()))),ops)
    binding=float(np.linalg.norm(A-c['tails'][3]/c['preps'][3].exact_rte_lambda_r,2));assert binding<1e-10
    old_probability_sum=sum(F.from_float(z.probability) for z in c['preps'][3].rte_preparation.symbolic_tail.components)
    tasks=cells();records=[];start=time.monotonic();workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(pool.submit(cell_check,cell) for cell in tasks):records.append(future.result())
    save_json(target,dict(**provenance(),stage='MATHEMATICAL_AND_DIGITAL_CONNECTION',artificial=artificial,labels=218,basis_groups=len(c['preps'][3].rte_preparation.basis_registry._definitions),
        old_float_probabilities_sum_exact=str(old_probability_sum),exact_p_sum=str(sum(probabilities())),x_domain_all=all(r['certificate']['domain'] for r in records),
        h4_generator_matrix_binding_error=binding,cells=sorted(records,key=lambda r:(r['t'],r['n'])),wall_seconds=time.monotonic()-start,workers=workers,
        scientific_target='same P3(-ix A); exact normalized binary probability p; fixed C generator binding retained',
        inference='Finite grouped expectation checked without enumerating 218^3 words. G6 formal identity and explicit G7 factorization; operator order retained.',passed=True))
    print('STAGE1_PASS',len(records),binding,flush=True)
