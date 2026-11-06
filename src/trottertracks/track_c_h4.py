"""Finite Track C H4 pilot adapters; reuse the repository DF/RTE primitives.

No SCF, DF fit, ground solve, or old launcher is called. See the adopted
PILOT_EXECUTION_SCOPE.md for the finite experiment and conditional claims.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
BASE = '7db4792d4cd0ed9a7abc255565f207dd31a437bd'
DFREF = '4591b8d75a2a6a55c74a51225f0c61648023c8ee'
DFPATH = ('artifacts/rte_connected_cluster_cost_validation/'
 'h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz')
OUT = ROOT/'artifacts/track_c_h4_pilots/2026-11-06/run01'
RVALUES = [1]*8+[2]*2+[4]+[8]*2+[16]*2+[32]*3
METRICS = ('rz','cx','depth','size')

# Avoid the broad convenience __init__ and its unrelated study imports.
if 'trotterlib' not in sys.modules:
    package = types.ModuleType('trotterlib')
    package.__path__ = [str(ROOT/'src/trotterlib')]
    sys.modules['trotterlib'] = package
from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
from trotterlib.df_partial_randomized_pf import rank_df_fragments, split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2, QiskitDFPartialS2CircuitBuilder
from trotterlib.df_rte_tail import basis_change_unitary
from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
from trotterlib.finite_rte_signal_validation import (
    _normalized_symbolic_tail_in_sector, _qiskit_to_openfermion_sector_permutation)
from trotterlib.rte import finite_rte_distribution, step_taylor_truncation_residual_bound
from .track_c_y8 import morales_2025_y8m10b_list


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def complex_json(z):
    return dict(real=float(z.real), imag=float(z.imag))


def git_npz(ref, path):
    raw = subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    return data,hashlib.sha256(raw).hexdigest()


@lru_cache(None)
def published():
    raw = subprocess.check_output(['git','show',
       'b190fe28b98d2f94e1509e19cd3d36a63b2bb99e:src/trottertracks/b2d_adapter/_published.py'],cwd=ROOT)
    ns = {}; exec(compile(raw,'fixed_b2d_published.py','exec'),ns)
    return ns


def weights(formula):
    if formula == 'S2': return [1.]
    if formula == 'Y4':
        a=1/(2-2**(1/3));return [a,1-2*a,a]
    w=morales_2025_y8m10b_list()
    return list(reversed(w[1:]))+[w[0]]+w[1:]


def s2_d_sequence(count, time):
    return [(i,time/2) for i in range(count)]+[(i,time/2) for i in reversed(range(count))]


def cp_d_sequence(count, time, processor=False):
    p=published()
    seq=(p['morales_2025_yp8m8_processor_sequence']() if processor else
         p['symmetric_s2_sequence'](p['morales_2025_yp8m8_kernel_list']()))
    return [item for w in reversed(seq) for item in s2_d_sequence(count,time*float(w))]


def one_matrix(h, sector):
    op,_=df_linear_operator(h,sector,backend='python')
    return np.column_stack([op@v for v in np.eye(sector.dimension)])


def prepare_common():
    OUT.mkdir(parents=True,exist_ok=True)
    df,dfsha=git_npz(DFREF,DFPATH)
    assert dfsha=='13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b'
    ref,refsha=git_npz(BASE,'artifacts/track_c_h4_input_check/2026-11-06/run02/ground_reference.npz')
    assert refsha=='6a350e7dd107f242bb00eae6a8d552ab4f4d1fc8df850c39ba8f838174994ae3'
    ham=DFHamiltonian(constant=df['constant'].item(),one_body=df['one_body'],
        lambdas=df['lambdas'],g_matrices=tuple(df['g_matrices']),metadata=json.loads(df['metadata_json'].item()))
    sector=PhysicalSector(n_qubits=8,basis_indices=ref['basis_indices'],n_electrons=4)
    zero=np.zeros((8,8),complex)
    block_matrices=[one_matrix(DFHamiltonian(0.,ham.one_body,np.array([]),(),{}),sector)]
    for l,g in zip(ham.lambdas,ham.g_matrices):
        block_matrices.append(one_matrix(DFHamiltonian(0.,zero,np.array([l]),(g,),{}),sector))
    spectra=[np.linalg.eigh((a+a.conj().T)/2) for a in block_matrices]
    ranked=sorted(rank_df_fragments(ham),key=lambda f:f.original_index)
    ranked=[replace(f,rank=i) for i,f in enumerate(ranked)]
    preps={};tails={};tail_spectra={}
    for ld in (3,12):
        partition=split_df_hamiltonian_by_ld(ham,ld,ranked_fragments=ranked)
        prep=prepare_df_partial_s2(ham,partition,coefficient_atol=1e-12,
                                  partition_policy='explicit_ordered_partition')
        assert prep.deterministic_fragment_indices==tuple(range(ld))
        preps[ld]=prep
        if ld==3:
            perm=_qiskit_to_openfermion_sector_permutation(sector)
            t=_normalized_symbolic_tail_in_sector(prep,sector,perm)
            assert t.leakage_frobenius_norm<1e-10
            tails[ld]=t.matrix*prep.exact_rte_lambda_r
            tail_spectra[ld]=np.linalg.eigh((tails[ld]+tails[ld].conj().T)/2)
    c=dict(ham=ham,sector=sector,state=ref['state'],energy=ref['energy'].item(),
           physical_H=ref['original_H_in_legacy_basis'],preps=preps,
           spectra=spectra,block_matrices=block_matrices,tails=tails,tail_spectra=tail_spectra)
    with (OUT/'common.pkl').open('wb') as f:pickle.dump(c,f)
    np.savez(OUT/'common_arrays.npz',state=c['state'],basis_indices=sector.basis_indices,
             original_H=c['physical_H'],blocks=np.stack(block_matrices),tail=tails[3])
    save_json(OUT/'INPUTS.json',dict(base_commit=BASE,df_ref=DFREF,df_path=DFPATH,
        df_sha256=dfsha,ground_reference_sha256=refsha,LD3=list(preps[3].deterministic_fragment_indices),
        tail_lambda=preps[3].exact_rte_lambda_r,threshold_operator_error_bound=preps[3].threshold_operator_error_bound,
        new_SCF_calls=0,new_DF_fit_calls=0,new_ground_solves=0,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__),ROOT/'src/trotterlib/rte.py',ROOT/'src/trotterlib/df_partial_s2.py',
                                ROOT/'src/trotterlib/df_rte_qiskit.py']}))
    return c


@lru_cache(None)
def common():
    with (OUT/'common.pkl').open('rb') as f:return pickle.load(f)


def evolution(spectrum,time):
    e,v=spectrum
    return (v*np.exp(-1j*time*e)[None,:])@v.conj().T


def d_operator(c,seq):
    out=np.eye(70,dtype=complex)
    for i,t in seq:out=evolution(c['spectra'][i],t)@out
    return out


def tail_operator(c,time,r):
    spectrum=c['tail_spectra'][3];e,v=spectrum
    x=-1j*time*e/r
    poly=1+x+x*x/2+x*x*x/6
    u=(v*np.power(poly,r)[None,:])@v.conj().T
    dimtime=c['preps'][3].exact_rte_lambda_r*time/r
    dist=finite_rte_distribution(dimtime,2)
    return u,r*math.log(dist.exact_finite_distribution),r*math.log1p(step_taylor_truncation_residual_bound(dimtime,2))


def kernel(c,formula,h,ld,r,processing=False):
    prep=c['preps'][ld];count=ld+1
    exact=np.eye(70,dtype=complex);finite=exact.copy();logb=0.;logerr=0.;recipes=[]
    for w in weights(formula):
        t=float(w*h)
        rj=max(1,math.ceil(abs(t)*r/.02)) if ld==3 else 0
        seq=cp_d_sequence(count,t/2) if processing else [(i,t/2) for i in range(count)]
        left=d_operator(c,seq)
        reverse=cp_d_sequence(count,t/2) if processing else [(i,t/2) for i in reversed(range(count))]
        right=d_operator(c,reverse)
        if ld==3:
            ideal_tail=evolution(c['tail_spectra'][3],t)
            ft,lb,le=tail_operator(c,t,rj);logb+=lb;logerr+=le
        else:ideal_tail=np.eye(70);ft=ideal_tail
        scalar=np.exp(-1j*t*(prep.constant_coefficient+prep.extracted_identity_coefficient))
        exact=scalar*right@ideal_tail@left@exact
        finite=scalar*right@ft@left@finite
        recipes.append(dict(time=t,r=rj))
    return exact,finite,logb,logerr,recipes


def phase_distance(a,b):
    return abs(float(np.angle(a*np.conj(b))))


def shot_rows(rows, prices=None, range_key='range'):
    if not all(row['feasible'] for row in rows):return None
    logs=[]
    for row in rows:
        eps=row['radius_lower']*math.sin(row['beta_stat'])/math.sqrt(2)
        logrange=row['log_range'] if 'log_range' in row else math.log(row[range_key])
        logs.extend([math.log(2)+2*logrange-2*math.log(eps)]*2)
    alpha=np.full(36,.05/36)
    policies=[('uniform',alpha)]
    if prices is not None:
        p=np.asarray(prices).reshape(-1)
        z=p==0
        logweights=np.asarray(logs)[~z]+np.log(p[~z])
        w=np.exp(logweights-logweights.max()) if len(logweights) else np.array([])
        a=np.zeros(36);a[z]=.05/36
        if np.any(~z):a[~z]=np.maximum((.05-a.sum())*w/w.sum(),math.ulp(0.))
        if np.all(a>0):policies.append(('cost_sensitive',a))
    out=[]
    for name,a in policies:
        if math.fsum(a)>.05:
            a=a*(.05/math.fsum(a))*math.nextafter(1.,0.)
        from fractions import Fraction
        while sum(map(Fraction.from_float,map(float,a)))>Fraction.from_float(.05):
            largest=int(np.argmax(a));a[largest]=math.nextafter(float(a[largest]),0.)
        n=[]
        for logc,ai in zip(logs,a):
            logn=logc+math.log(math.log(2)-math.log(float(ai)))
            if logn<700:n.append(math.ceil(math.exp(logn)))
            else:
                import mpmath as mp
                with mp.workdps(max(80,int(logn/math.log(10))+40)):
                    n.append(int(mp.ceil(mp.exp(mp.mpf(logc))*mp.log(2/mp.mpf(float(ai))))))
        cost=None;logcost=None
        if prices is not None:
            terms=[math.log(ni)+math.log(float(ci)) for ni,ci in zip(n,np.asarray(prices).reshape(-1)) if ci>0]
            if terms:
                top=max(terms);logcost=(top+math.log(math.fsum(math.exp(x-top) for x in terms)))/math.log(10)
                if logcost<300:cost=10**logcost
        out.append(dict(policy=name,alpha=a.tolist(),shots=n,shots_log10=[math.log10(x) for x in n],
                        cost=cost,cost_log10=logcost))
    return out


def signal_task(task):
    c=common();pilot=task['pilot'];formula=task.get('formula','S2');h=task.get('h',.02);ld=task.get('ld',3)
    processed=pilot=='C-P';p=None
    if processed:
        ppub=d_operator(c,cp_d_sequence(4,h/2,processor=True));p=ppub.conj().T
    rows=[]
    for m,r in enumerate(RVALUES):
        q=2**m;steps=round(q*.02/h)
        ideal,finite,lb,le,recipe=kernel(c,formula,h,ld,r,processing=processed)
        vi=c['state'] if not task.get('P',False) else p@c['state']
        ui=np.linalg.matrix_power(ideal,steps);uf=np.linalg.matrix_power(finite,steps)
        zi=np.vdot(vi,ui@vi);zf=np.vdot(vi,uf@vi)
        zref=np.exp(-1j*c['energy']*q*.02)
        logb=steps*lb;atten=math.exp(-logb) if logb<745 else 0.
        err=math.expm1(steps*le) if steps*le<700 else math.inf
        # Conditional rounding diagnostic, not an interval certificate.
        defect=float(np.linalg.norm(ideal.conj().T@ideal-np.eye(70),2))
        num=1e-10+steps*(defect+np.finfo(float).eps*70*len(recipe)*4)
        hstate=q*.02*(9.480991703075102e-15+c['preps'][ld].threshold_operator_error_bound)+2*1.9069976415519017e-15/.23263021512018023
        pf=phase_distance(zi,zref)
        rte=math.asin(err/abs(zi)) if err<abs(zi) and abs(zi)>0 else None
        other=math.asin((num+hstate)/abs(zi)) if num+hstate<abs(zi) else None
        budget=None if rte is None or other is None else .4-pf-rte-other
        rho=max(0.,abs(zi)-err-num-hstate)*atten
        feasible=budget is not None and budget>0 and rho>0
        row=dict(m=m,q=q,t=q*.02,r=0 if ld==12 else r,registered_r_m=r,steps=steps,recipe=recipe,
            ideal=complex_json(zi),corrected=complex_json(zf),raw=complex_json(zf*atten),
            radius_ideal=float(abs(zi)),radius_corrected=float(abs(zf)),radius_lower=rho,
            log_normalizer=logb,attenuation=atten,range=1.,beta_PF=pf,beta_original_RTE=rte,
            actual_finite_phase=phase_distance(zf,zi),beta_numerical_input=other,
            numerical_disk=num,input_disk=hstate,finite_error_bound=None if not math.isfinite(err) else err,
            beta_stat=budget,feasible=bool(feasible),evidence='conditional_float64_model')
        rows.append(row)
    payload=dict(task=task,rows=rows,all_rounds_feasible=all(x['feasible'] for x in rows),shots=shot_rows(rows),
                 P_unitarity=None if p is None else float(np.linalg.norm(p.conj().T@p-np.eye(70),2)),
                 alias='external base-window and .4 rad endpoint margin assumed',
                 total_cost_status='WAITING_PRICE')
    save_json(OUT/'signals'/f"{task['name']}.json",payload)
    return task['name'],payload['all_rounds_feasible']


def signal_tasks():
    return [dict(pilot='C-N',formula=f,h=h,ld=ld,name=f'CN_{f}_h{h}_ld{ld}')
            for f in ('S2','Y4','Y8') for h in (.02,.01) for ld in (3,12)]+[
         dict(pilot='C-P',P=p,formula='S2',h=.02,ld=3,name=f'CP_P{int(p)}') for p in (False,True)]


def diagonal_primitives(block,time):
    if block.block_kind=='one_body':
        return one_body_diagonal_primitives(np.array(block.diagonal_eigenvalues),time)
    return df_squared_diagonal_primitives(np.array(block.diagonal_eta),block.lam,time)


def residual_parameters(theta,r):
    k=round(theta/(math.pi/4));coarse=k*math.pi/4;delta=theta-coarse
    x=delta/(2*r)
    dist=finite_rte_distribution(x,2)
    z=(1-1j*x-x*x/2+1j*x*x*x/6)**r
    z*=np.exp(-1j*coarse/2)
    return z,r*math.log(dist.exact_finite_distribution),r*math.log1p(step_taylor_truncation_residual_bound(x,2))


@lru_cache(None)
def d_residual_data():
    c=common();prep=c['preps'][3];basis=c['sector'].basis_indices
    rev=np.array([int(f'{int(x):08b}'[::-1],2) for x in basis])
    states=np.arange(256,dtype=np.int64);result=[]
    for block in prep.deterministic_blocks:
        # Deterministic and tail registries are separate; retain this block's V.
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        qc=QuantumCircuit(8)
        for gate,wires in block.runtime_basis_operations:qc.append(gate,list(wires))
        v=Operator(qc).data[np.ix_(rev,rev)]
        prim=diagonal_primitives(block,.01)
        angles=[];signs=[];supports=[]
        for q,theta in prim.rz:
            angles.append(float(theta));signs.append(1-2*((rev>>q)&1));supports.append((q,))
        for a,b,theta in prim.rzz:
            if abs(theta)<1e-12:continue
            angles.append(float(theta));signs.append(1-2*(((rev>>a)^(rev>>b))&1));supports.append((a,b))
        result.append(dict(v=v,angles=angles,signs=signs,phase=float(prim.global_phase),supports=supports))
    return result


def cs_kernel(c,r,rres,arm):
    dseq=[];dlb=0.;dle=0.
    for item in d_residual_data():
        diag=np.ones(70,dtype=complex)*np.exp(1j*item['phase'])
        for theta,sign in zip(item['angles'],item['signs']):
            z,lb,le=residual_parameters(theta,rres)
            diag*=np.where(sign==1,z,np.conj(z));dlb+=lb;dle+=le
        dseq.append((item['v']*diag[None,:])@item['v'].conj().T)
    left=np.eye(70,dtype=complex);right=left.copy()
    for d in dseq:left=d@left
    for d in reversed(dseq):right=d@right
    prep=c['preps'][3];x=prep.exact_rte_lambda_r*.02/r
    f=c['tails'][3]/prep.exact_rte_lambda_r
    dist=finite_rte_distribution(x,2)
    ft=np.zeros((70,70),complex);rlbmax=0.;rlemax=0.
    for order,weight in zip(dist.orders,dist.unnormalized_order_weights):
        theta=2*math.atan(x/(order+1))
        if arm=='S1':z,lb,le=residual_parameters(theta,rres)
        else:z=np.exp(-1j*theta/2);lb=le=0.
        ft+=weight*(-1)**(order//2)*(z.real*np.eye(70)-1j*(-z.imag)*f)@np.linalg.matrix_power(f,order)
        rlbmax=max(rlbmax,lb);rlemax=max(rlemax,le)
    ft=np.linalg.matrix_power(ft,r)
    scalar=np.exp(-1j*.02*(prep.constant_coefficient+prep.extracted_identity_coefficient))
    return scalar*right@ft@left,2*dlb+r*rlbmax,2*dle+r*rlemax


def cs_signal_task(task):
    c=common();arm=task['arm'];cap=task['cap'];rows=[]
    base=json.loads((OUT/'signals/CN_S2_h0.02_ld3.json').read_text())['rows']
    for m,b in enumerate(base):
        q=2**m;r=RVALUES[m];_,basefinite,_,_,_=kernel(c,'S2',.02,3,r)
        zbase=complex(b['corrected']['real'],b['corrected']['imag'])
        selected=None
        if arm=='S0':selected=(0,basefinite,0.,0.)
        else:
            for rr in (1,2,4,8,16,32):
                k,lb,le=cs_kernel(c,r,rr,arm)
                error=math.exp(b['log_normalizer'])*math.expm1(q*le) if q*le<700 else math.inf
                if error<abs(zbase)*math.sin(.01):
                    selected=(rr,k,lb,error);break
        if selected is None:
            rows.append(dict(m=m,q=q,r=r,feasible=False,reason='residual_bound_not_within_cap_at_r32'));continue
        rr,k,lb,reserr=selected
        z=np.vdot(c['state'],np.linalg.matrix_power(k,q)@c['state'])
        resphase=math.asin(reserr/abs(zbase)) if reserr else 0.
        logrange=b['log_normalizer']+q*lb
        L=math.exp(logrange) if logrange<700 else None
        synth_disk=max(0.,abs(z)-b['numerical_disk']-b['input_disk'])*math.sin(cap)
        rho=max(0.,abs(z)-b['numerical_disk']-b['input_disk']-synth_disk)
        beta_stat=.4-b['beta_PF']-b['beta_original_RTE']-b['beta_numerical_input']-resphase-cap
        row=dict(m=m,q=q,t=q*.02,r=r,r_res=rr,corrected=complex_json(z),
            base_corrected=complex_json(zbase),radius_corrected=float(abs(z)),radius_lower=rho,
            log_range=logrange,range=L,log_original_normalizer=b['log_normalizer'],
            beta_PF=b['beta_PF'],beta_original_RTE=b['beta_original_RTE'],beta_residual_RTE=resphase,
            beta_numerical_input=b['beta_numerical_input'],beta_synthesis=cap,beta_stat=beta_stat,
            residual_error_bound=reserr,actual_residual_phase=phase_distance(z,zbase),
            synthesis_disk_cap=synth_disk,native_signal_status='finite_mean_center_with_joint_error_disk',
            feasible=bool(math.isfinite(logrange) and beta_stat>0 and rho>0),evidence='conditional_float64_model')
        assert phase_distance(z,zbase)<=resphase+2*b['numerical_disk']
        rows.append(row)
    payload=dict(task=task,rows=rows,all_rounds_feasible=all(x['feasible'] for x in rows),
                 shots=shot_rows(rows),total_cost_status='WAITING_PRICE',
                 estimator='corrected weighted; conditional residual normalizers, bounded global range')
    save_json(OUT/'signals'/f"{task['name']}.json",payload)
    return task['name'],payload['all_rounds_feasible'],max(x.get('log_range',0) for x in rows)


def cs_tasks():
    return [dict(pilot='C-S',arm=a,cap=b,name=f'CS_{a}_cap{b}') for a in ('S0','S1','S2') for b in (.01,.02)]
