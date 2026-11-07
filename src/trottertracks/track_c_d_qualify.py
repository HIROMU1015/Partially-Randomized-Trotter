"""Reuse directed Track C matrix/source and local native error certificates.

Qualified scope: fixed saved Hermitian-H matrix, exact ground input, ideal
native rotations and fresh independent bits. No continuum-H inclusion claim.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import mpmath as mp
import numpy as np
from .track_c_d_internal import ROOT,OUT,common,schedule,save_json,kernel
from .track_c_d_resources import split,d_piece
from .track_c_h4 import diagonal_primitives,finite_rte_distribution
from .track_c_compiler_a import compile_reference
from . import track_c_source_guarantee as sg
from . import track_c_compiler_a_18round as ca
from .track_c_h4_finite_bits import proposal_budget,integer_cdf

POLICIES=('original','order_0321','vertex_j0','tuned_0123','tuned_0321','PR_adaptive_RZ','Full_adaptive_RZ','Full_adaptive_CX')
OLD='720416c5b88b227cf76ff7aa729d91a0a98ed58b'

@lru_cache(None)
def old():
    def read(name):return json.loads(subprocess.check_output(['git','show',OLD+':artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/'+name],cwd=ROOT))
    raw=subprocess.check_output(['git','show','3258a89f225d9530fba38e9d2343bc0165dd6d7d:artifacts/track_c_compiler_a_18round/2026-11-06/run01/local_rule_evidence.tar.xz'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:xz') as tar:bounds=json.load(tar.extractfile('PRIMITIVE_UPPERS.json'))
    return read('SOURCE_GENERATORS.json'),read('SPECTRAL_CERTIFICATES.json'),read('GROUND_ALIAS.json'),bounds

def selected():
    data=json.loads((OUT/'G_RESULTS.json').read_text())
    return [x for x in data['policies'] if x['name'] in POLICIES]

@lru_cache(None)
def d_counts(index,t):
    catalogue={};block=common()['preps'][12].deterministic_blocks[index];counts=ca.basis_counter(block.runtime_basis_operations,catalogue)
    p=diagonal_primitives(block,t)
    for _,a in p.rz:ca.add(counts,catalogue,'crz',[a])
    for _,_,a in p.rzz:
        if a!=0:ca.add(counts,catalogue,'crzz',[a])
    return counts,catalogue

def compiler_catalogue(rows):
    catalogue=ca.inventory()['catalogue'];new={}
    for row in rows:
        x=row['signal'];seq=schedule(x['case'],x['h']);groups=split(x['case'],x['h']);ld=12 if groups['full'] else 3
        for role,g in groups.items():
            if role=='full' or not g:continue
            trace={}
            def cb(**kw):
                if type(kw['pass_']).__name__=='UnrollCustomDefinitions':trace['families']=set(kw['dag'].count_ops())-{'rz','cx','barrier'}
            compile_reference(d_piece(tuple(g),ld),callback=cb)
            assert trace['families'] in ({'xx_plus_yy','crz','ccx','p'},{'xx_plus_yy','crz','p'}),trace
        for kind,index,t in seq:
            if kind=='D':new.update(d_counts(index,t)[1])
        if x['steps']>1:new.update(d_counts(seq[0][1],2*seq[0][2])[1])
    catalogue.update(new)
    return catalogue

@lru_cache(None)
def diagonal_error(index,t):
    mp.iv.prec=192;b=common()['preps'][12].deterministic_blocks[index];p=diagonal_primitives(b,t);errors=[]
    for occ in sg.occupation():
        if b.block_kind=='one_body':energy=sum((sg.I(v)*n for v,n in zip(b.diagonal_eigenvalues,occ)),sg.I(0))
        else:energy=sg.I(b.lam)*sum((sg.I(v)*n for v,n in zip(b.diagonal_eta,occ)),sg.I(0))**2
        phi=sg.I(p.global_phase)
        for q,a in p.rz:phi-=sg.I(a)*(1-2*occ[q])/2
        for q,r,a in p.rzz:phi-=sg.I(a)*(1-2*occ[q])*(1-2*occ[r])/2
        errors.append(sg.up(abs(phi+sg.I(t)*energy)))
    return max(errors)

@lru_cache(None)
def unitary(index,t):
    cert=old()[1];c=common()
    if index<0:h=c['tails'][3];e,v=c['tail_spectra'][3];sc=cert['tail']
    else:h=c['block_matrices'][index];e,v=c['spectra'][index];sc=cert['blocks'][index]
    return sg.spectral_unitary(sg.hermitian(h),e,v,t,sc)

def qualify(request):
    x,bounds=request;mp.iv.prec=192;c=common();dcert,_,g,_=old();dgens=dcert['deterministic'];tail=dcert['tail'];seq=schedule(x['case'],x['h']);ld=12 if x['case']['kind'].startswith('full_') else 3
    prep=c['preps'][ld];scalar=float(prep.constant_coefficient+prep.extracted_identity_coefficient)
    u=np.eye(70,dtype=complex);eu=0.;source=0.;tailerr=0.;logB=sg.I(0);compiler_counts=Counter();phasegen=0.
    for kind,index,t in seq:
        a,ea=unitary(index if kind=='D' else -1,t);u,eu=sg.compose(a,ea,u,eu)
        if kind=='D':
            source=sg.add(source,sg.mul(abs(t),dgens[index]['generator_error_upper']),diagonal_error(index,t));compiler_counts.update(d_counts(index,t)[0])
        else:
            source=sg.add(source,sg.mul(abs(t),c['preps'][3].exact_rte_lambda_r,tail['generator_normalized_error_upper']))
            n=max(1,math.ceil(abs(t)*x['rate']/.02));tau=sg.I(prep.exact_rte_lambda_r)*sg.I(t)/n
            dist=finite_rte_distribution(prep.exact_rte_lambda_r*t/n,2);masses=[mp.iv.sqrt(1+tau**2),tau**2/2*mp.iv.sqrt(1+(tau/3)**2)];B=sum(masses,sg.I(0));logB+=n*mp.iv.log(B)
            norm=sum((sg.I(p) for p in dist.order_probabilities),sg.I(0));dp=sum((abs(sg.I(p)/norm-v/B) for p,v in zip(dist.order_probabilities,masses)),sg.I(0))
            angle=max(sg.up(abs(sg.I(math.atan(dist.dimensionless_step_time/(order+1)))-mp.iv.atan2(tau/(order+1),sg.I(1)))) for order in (0,2))
            tailerr=sg.add(tailerr,sg.up(n*(mp.iv.exp(abs(tau))*abs(tau)**4/(24*B)+dp+sg.I(angle))))
            phasegen=sg.add(phasegen,ca.phase_accumulation_bound(n))
    phase=sg.expi(-sg.I(x['h'])*sg.I(scalar));fp=sg.centre(phase);u,eu=sg.compose(fp*np.eye(70),sg.up(abs(phase-sg.C(fp.real,fp.imag))),u,eu)
    # Source primitive arithmetic for actually fused D intervals at all boundaries.
    nsteps=x['steps'];counts=Counter({k:v*nsteps for k,v in compiler_counts.items()})
    source*=nsteps
    if nsteps>1:
        index=seq[0][1];t=seq[0][2];assert seq[-1]==seq[0]
        for k,v in d_counts(index,t)[0].items():counts[k]-=2*(nsteps-1)*v
        counts.update({k:(nsteps-1)*v for k,v in d_counts(index,2*t)[0].items()})
        # Generator error is time-linear, but diagonal arithmetic is recomputed.
        source=sg.add(source,sg.mul(nsteps-1,diagonal_error(index,2*t)))
    source=sg.add(source,sg.up(abs(sg.I(float(-x['t']*scalar))+sg.I(x['t'])*sg.I(scalar))))
    # All partial families here have positive R; retained event angles are covered
    # by the prior all-context local catalogue, including K2's complete support.
    prior_inventory=ca.inventory();maximum_basis=max(sg.add(*[sg.mul(v,bounds[k]) for k,v in bc.items()]) for bc in prior_inventory['tail_basis'])
    assert min(counts.values())>=0
    compiler=sg.add(*[sg.mul(v,bounds[k]) for k,v in counts.items()])
    for recipe in x['recipe']:
        t=recipe['time'];n=recipe['r'];tau=prep.exact_rte_lambda_r*t/n;rotation=[]
        for order in (0,2):
            a=math.atan(tau/(order+1))
            for sign in (-1,1):
                for family in ('crz','crzz'):rotation.append(bounds[ca.key(family,[sign*2*a])])
        compiler=sg.add(compiler,sg.mul(nsteps,n,sg.add(sg.mul(3,maximum_basis),sg.mul(4,bounds[ca.key('cz')]),max(rotation))))
    compiler=sg.add(compiler,sg.mul(2,bounds[ca.key('h')]),bounds[ca.key('sdg')])
    compiler_disk=sg.up(2*mp.iv.sqrt(sg.I(2))*sg.I(compiler));phase_disk=sg.up(2*mp.iv.sqrt(sg.I(2))*sg.I(nsteps)*sg.I(phasegen))
    # Certified binary powering of the same ideal kernel.
    q=nsteps;power=np.eye(70,dtype=complex);ep=0.
    while q:
        if q&1:power,ep=sg.compose(u,eu,power,ep)
        q//=2
        if q:u,eu=sg.compose(u,eu,u,eu)
    ivpsi=[sg.C(z.real,z.imag) for z in c['state']];norm=mp.iv.sqrt(sum((abs(z)**2 for z in ivpsi),sg.I(0)));psi=np.array([sg.centre(z/norm) for z in ivpsi]);ev=g['normalization_error_upper']
    upsi,edot=sg.product(power,psi[:,None]);z=complex(np.vdot(psi,upsi[:,0]));evalerr=sg.add(sg.mul(ep,sg.add(1,ev),sg.add(1,ev)),sg.mul(sg.add(1,ev),edot),sg.up(sg.gamma(8*70+4)*sg.I(sg.fro(psi))*sg.I(sg.fro(upsi))),sg.mul(ev,sg.add(2,ev)),sg.mul(2,g['state_distance_upper']))
    rotated=sg.C(z.real,z.imag)*sg.expi(sg.I(x['t'])*sg.I(g['energy_interval']));zc=sg.centre(rotated)
    eref=sg.add(sg.up(abs(rotated-sg.C(zc.real,zc.imag))),sg.up(abs(sg.I(x['t'])-sg.I(1<<x['m'])*sg.I('.02'))*sg.I(g['Hamiltonian_absolute_spectral_upper'])))
    pf=sg.phase_upper(zc,sg.add(evalerr,eref));rho0=sg.lower((abs(sg.C(z.real,z.imag))-sg.I(evalerr))*mp.iv.exp(-nsteps*logB))
    proposal=proposal_budget(0.,4*x['RTE_events'],218,min(q.probability for q in c['preps'][3].rte_preparation.symbolic_tail.components));proposal_disk=sg.up(2*mp.iv.sqrt(sg.I(2))*sg.I(4*x['RTE_events'])*218*sg.I(2)**(-proposal['bits_per_uniform_draw'])) if ld==3 else 0.
    if ld==3:integer_cdf([z.probability for z in c['preps'][3].rte_preparation.symbolic_tail.components],proposal['bits_per_uniform_draw'])
    extra=sg.add(sg.up(2*mp.iv.sqrt(sg.I(2))*sg.I(sg.add(source,sg.mul(nsteps,tailerr)))),compiler_disk,phase_disk,proposal_disk)
    ratio=sg.I(extra)/sg.I(rho0);phase_extra=sg.up(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))) if extra<rho0 else None
    beta=None if pf is None or phase_extra is None else sg.lower(sg.I('.4')-sg.I(pf)-sg.I(phase_extra));rho=sg.lower(sg.I(rho0)-sg.I(extra));passed=bool(g['all_spectrum_endpoint_margin_pass'] and beta is not None and beta>0 and rho>0)
    shots=None
    if passed:
        nn=4*mp.iv.log(sg.I(1440))/(sg.I(rho)**2*mp.iv.sin(sg.I(beta))**2)
        with mp.workprec(192):shots=int(mp.ceil(mp.mpf(nn._mpi_[1])))
    # Spectral tracking distinguishes small wrapped endpoint phase from bias.
    ideal=kernel(x['case'],x['h'],x['rate'])[0];eig,vec=np.linalg.eig(ideal);overlap=abs(vec.conj().T@c['state'])**2;ig=int(np.argmax(overlap));delta=-float(np.angle(eig[ig]*np.exp(1j*c['energy']*x['h'])))/x['h']
    return dict(name=x['name'],case=x['case'],m=x['m'],h=x['h'],rate=x['rate'],PF_phase_upper=pf,rho_lower=rho,beta_stat_lower=beta,shots_per_axis=shots,closed=passed,
        compiler_disk=compiler_disk,source_phase_disk=phase_disk,source_and_finite_RTE_upper=sg.add(source,sg.mul(nsteps,tailerr)),eval_disk=evalerr,raw_extra_disk=extra,
        dominant_overlap=float(overlap[ig]),dominant_energy_bias=delta,dominant_accumulated_bias_abs=abs(delta*x['t']),
        scope='fixed saved-H matrix; exact ground; ideal native gates; fresh independent finite-bit draws')

def run():
    target=OUT/'QUALIFICATION.json'
    if target.exists():raise RuntimeError('Do not overwrite qualification')
    policies=selected();rows={json.dumps(r['signal'],sort_keys=True):r for p in policies for r in p['selected']};cat=compiler_catalogue(rows.values())
    records=[];workers=min(len(os.sched_getaffinity(0)),len(rows))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        contexts=list(pool.map(ca.context_compile,[(mask,cat) for mask in (11,15)]))
        unique={r['lowering_sha256']:r for context in contexts for r in context['records']}
        checked=list(pool.map(ca.interval_check,unique.values()));bounds=old()[3].copy()
        for r in checked:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
        for f in as_completed([pool.submit(qualify,(r['signal'],bounds)) for r in rows.values()]):records.append(f.result())
    save_json(OUT/'LOCAL_INTERVALS.json',dict(contexts=contexts,checked=checked,primitive_bounds=bounds))
    save_json(target,dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),rows=records,all_closed=all(r['closed'] for r in records),workers=workers,
        reused_certificates=OLD,new_D_context_masks=[11,15],local_lowerings=len(unique),
        qualification='Directed matrix/source bounds reused with new time-dependent diagonal arithmetic and new D-angle local native intervals. Actual modular D contexts checked; RTE complete support uses unchanged old all-context angles/bases. Fixed-matrix model only.'))
    print('QUALIFICATION_COMPLETE',len(records),flush=True)

if __name__=='__main__':run()
