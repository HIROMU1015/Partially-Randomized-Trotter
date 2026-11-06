"""Fixed H4 SOURCE/mean and matrix-model RPE certificate, independent of prices."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import math
import os
import subprocess
import time
import mpmath as mp
import numpy as np
from scipy.linalg import ldl
from .track_c_h4 import ROOT, common, save_json, weights, diagonal_primitives, finite_rte_distribution
from .track_c_compiler_a_18round import I, C, expi, upper, lower, source_matrix, spec
from .track_c_h4_finite_bits import integer_cdf, proposal_budget

BASE='3258a89f225d9530fba38e9d2343bc0165dd6d7d'
OUT=ROOT/'artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified'
BITS=192


def up(x):return upper(x if hasattr(x,'_mpi_') else I(x))
def gamma(n):return I(n)*I(2)**-53/(1-I(n)*I(2)**-53)
def add(*x):return up(sum((I(v) for v in x),I(0)))
def mul(*x):
    y=I(1)
    for v in x:y*=I(v)
    return up(y)


def fro(a):
    mp.iv.prec=BITS;s=I(0)
    for z in np.asarray(a).reshape(-1):s+=I(float(z.real))**2+I(float(z.imag))**2
    return up(mp.iv.sqrt(s))


def product(a,b):
    c=a@b
    return c,up(gamma(8*a.shape[-1]+4)*I(fro(a))*I(fro(b)))


def sub_residual(a,b):
    return add(fro(a-b),up(gamma(2)*(I(fro(a))+I(fro(b)))))


def centre(z):return complex(float(z.real.mid),float(z.imag.mid))
def vector_or_matrix(q):return np.array([[centre(z) for z in row] for row in q],complex)


def hermitian(a):return (a+a.conj().T)/2


@lru_cache(None)
def basis(operations_key):
    """Physical number-preserving SOURCE only; never project native gates."""
    mp.iv.prec=BITS;c=common();indices=[int(f'{int(v):08b}'[::-1],2) for v in c['sector'].basis_indices]
    lookup={v:i for i,v in enumerate(indices)};n=len(indices);a=np.eye(n,dtype=complex);err=0.
    for name,qs,p in operations_key:
        if name=='rz':
            vals=[expi(-I(p[0])/2),expi(I(p[0])/2)];v=[centre(x) for x in vals]
            coeff=max(up(abs(x-C(z.real,z.imag))) for x,z in zip(vals,v))
            a=np.array([row*v[(b>>qs[0])&1] for b,row in zip(indices,a)])
        elif name=='xx_plus_yy':
            m=source_matrix(spec(name,p));v=vector_or_matrix(m)
            coeff=up(mp.iv.sqrt(sum((abs(m[i][j]-C(v[i,j].real,v[i,j].imag))**2 for i in range(4) for j in range(4)),I(0))))
            old=a.copy();q0,q1=qs
            for i,b in enumerate(indices):
                b0=(b>>q0)&1;b1=(b>>q1)&1
                if b0+b1!=1:continue
                j=lookup[b^(1<<q0)^(1<<q1)];row=1 if b0 else 2;col=3-row
                a[i]=v[row,row]*old[i]+v[row,col]*old[j]
        else:raise ValueError('Unknown SOURCE basis family '+name)
        # At most two complex products and an addition per active row.
        rounding=up(gamma(32)*2*mp.iv.sqrt(I(n))*(1+I(err))*(1+I(coeff)))
        err=add(err,mul(coeff,add(1,err)),rounding)
    return a,err


def opskey(operations):
    return tuple((g.name,tuple(q),tuple(float(x) for x in g.params)) for g,q in operations)


def generator_from_basis(operations,diagonal_intervals):
    b,eb=basis(opskey(operations));es=np.array([float(x.mid) for x in diagonal_intervals])
    ed=max(up(abs(x-I(v))) for x,v in zip(diagonal_intervals,es))
    bd=b*es[None,:];db=up(gamma(2)*I(fro(b))*max(abs(es)))
    g,rounding=product(bd,b.conj().T)
    error=add(rounding,mul(db,add(1,eb)),mul(max(abs(es)),add(mul(2,eb),mul(eb,eb))),mul(ed,add(1,eb),add(1,eb)))
    return g,error,eb


def occupation():
    indices=[int(f'{int(v):08b}'[::-1],2) for v in common()['sector'].basis_indices]
    return [[(v>>q)&1 for q in range(8)] for v in indices]


def d_task(index):
    mp.iv.prec=BITS;c=common();block=c['preps'][12].deterministic_blocks[index];energies=[]
    for occ in occupation():
        if block.block_kind=='one_body':e=sum((I(v)*n for v,n in zip(block.diagonal_eigenvalues,occ)),I(0))
        else:e=I(block.lam)*sum((I(v)*n for v,n in zip(block.diagonal_eta,occ)),I(0))**2
        energies.append(e)
    g,eg,eb=generator_from_basis(block.runtime_basis_operations,energies)
    target=hermitian(c['block_matrices'][index]);res=add(sub_residual(g,target),eg)
    diagonals={}
    for formula,ld in [('S2',3),('Y4',12)]:
        if index>ld:continue
        for w in weights(formula):
            t=float(w*.02/2);p=diagonal_primitives(block,t);errors=[]
            for occ,e in zip(occupation(),energies):
                phi=I(p.global_phase)
                for q,a in p.rz:phi-=I(a)*(1-2*occ[q])/2
                for q,r,a in p.rzz:phi-=I(a)*(1-2*occ[q])*(1-2*occ[r])/2
                errors.append(up(abs(phi+I(t)*e)))
            diagonals[t.hex()]=max(errors)
    return dict(index=index,generator_error_upper=res,basis_operator_error_upper=eb,diagonal_phase_error_uppers=diagonals)


def tail_task():
    mp.iv.prec=BITS;c=common();p=c['preps'][3].rte_preparation;cs=p.symbolic_tail.components
    total=sum((I(x.probability) for x in cs),I(0));groups={}
    for x in cs:
        assert not x.is_identity and len(x.diagonal_pauli_support) in (1,2)
        if x.basis_id not in groups:groups[x.basis_id]=[I(0) for _ in range(70)]
        for j,occ in enumerate(occupation()):
            sign=x.coefficient_sign*math.prod(1-2*occ[q] for q in x.diagonal_pauli_support)
            groups[x.basis_id][j]+=I(x.probability)*sign/total
    a=np.zeros((70,70),complex);err=0.
    for name,diag in groups.items():
        definition=p.basis_registry._definitions[name]
        g,e,_=generator_from_basis(definition.runtime_operations,diag)
        rounding=up(gamma(2)*(I(fro(a))+I(fro(g))));a=a+g;err=add(err,e,rounding)
    target=hermitian(c['tails'][3])/c['preps'][3].exact_rte_lambda_r
    target_error=up(gamma(2)*I(fro(target)))
    residual=add(sub_residual(a,target),err,target_error)
    return dict(generator_normalized_error_upper=residual,basis_groups=len(groups),components=len(cs),
        mean_identity='p0(cos(a0)I-i sin(a0)A) - p2 A^2(cos(a2)I-i sin(a2)A)',
        proof='independent component draws; signed Hermitian involutions B P B^dagger; exact positive rational normalization')


def spectral_certificate(h,e,v):
    mp.iv.prec=BITS;n=len(e);vv,em=product(v.conj().T,v);delta=add(sub_residual(vv,np.eye(n)),em)
    assert delta<1
    nv=up(mp.iv.sqrt(1+I(delta)));sigma=lower(mp.iv.sqrt(1-I(delta)))
    polar=up(I(delta)/(1+mp.iv.sqrt(1-I(delta))))
    hv,eh=product(h,v);vd=v*e[None,:];ed=up(gamma(2)*I(fro(v))*max(abs(e)))
    residual=add(sub_residual(hv,vd),eh,ed)
    generator=add(mul(residual,nv),mul(fro(h),delta),mul(max(abs(e)),polar,add(nv,1)))
    return dict(unitarity_defect_upper=delta,polar_distance_upper=polar,V_norm_upper=nv,generator_error_upper=generator)


def spectral_unitary(h,e,v,t,cert):
    mp.iv.prec=BITS;ph=[expi(-I(t)*I(x)) for x in e];values=np.array([centre(x) for x in ph])
    er=max(up(abs(x-C(y.real,y.imag))) for x,y in zip(ph,values))
    vd=v*values[None,:];ed=up(gamma(8)*I(fro(v)))
    u,em=product(vd,v.conj().T)
    error=add(mul(abs(t),cert['generator_error_upper']),
        mul(cert['polar_distance_upper'],add(cert['V_norm_upper'],1)),mul(er,cert['V_norm_upper'],cert['V_norm_upper']),mul(ed,cert['V_norm_upper']),em)
    return u,error


def compose(a,ea,b,eb):
    result=a@b;n=len(a)
    rounding=up(gamma(8*n+4)*n*(1+I(ea))*(1+I(eb)))
    return result,add(ea,eb,mul(ea,eb),rounding)


def inertia(h,cut):
    shift=h-cut*np.eye(len(h));l,d,_=ldl(shift,lower=True,hermitian=True)
    ld,e1=product(l,d);back,e2=product(ld,l.conj().T)
    err=add(sub_residual(back,shift),e2,mul(e1,fro(l)),up(gamma(2)*(I(fro(h))+abs(I(cut))*mp.iv.sqrt(I(len(h))))))
    inv=np.linalg.inv(l);check,ei=product(inv,l)
    defect=add(sub_residual(check,np.eye(len(h))),ei);assert defect<1
    inverse_norm=up(I(fro(inv))/(1-I(defect)));margin=mul(err,inverse_norm,inverse_norm)
    signs=[];distance=math.inf;j=0
    while j<len(h):
        if j+1<len(h) and d[j,j+1]!=0:
            a=I(float(d[j,j].real));b=I(float(d[j+1,j+1].real));z=C(d[j,j+1].real,d[j,j+1].imag)
            root=mp.iv.sqrt((a-b)**2+4*abs(z)**2);vals=[(a+b-root)/2,(a+b+root)/2];j+=2
        else:vals=[I(float(d[j,j].real))];j+=1
        for val in vals:
            if val.b<0:signs.append(-1);distance=min(distance,lower(-val))
            elif val.a>0:signs.append(1);distance=min(distance,lower(val))
            else:raise ArithmeticError('Inertia pivot interval crosses zero')
    assert distance>margin,(distance,margin)
    return dict(cut=cut,negative_count=signs.count(-1),minimum_D_absolute_eigenvalue_lower=distance,
        congruence_perturbation_upper=margin,certified=True,method='verified LDL inertia, no eigenpair/ground solve')


def ground_certificate(h):
    mp.iv.prec=BITS;c=common();psi=c['state'];norm=mp.iv.sqrt(sum((abs(C(z.real,z.imag))**2 for z in psi),I(0)))
    ivpsi=[C(z.real,z.imag)/norm for z in psi];v=np.array([centre(z) for z in ivpsi])
    ev=up(mp.iv.sqrt(sum((abs(x-C(y.real,y.imag))**2 for x,y in zip(ivpsi,v)),I(0))))
    hv,eh=product(h,v[:,None]);hv=hv[:,0];mu=float(np.vdot(v,hv).real)
    emu=add(mul(add(1,ev),eh),up(gamma(8*len(v)+4)*I(fro(v))*I(fro(hv))),mul(ev,add(2,ev),fro(h)))
    residual=add(fro(hv-mu*v),eh,mul(emu,add(1,ev)),up(gamma(8)*(I(fro(hv))+abs(I(mu))*I(fro(v)))),mul(ev,add(fro(h),abs(mu),emu)))
    cut=float(c['energy']+.23263021512018023/2)
    ic=inertia(h,cut);assert ic['negative_count']==1
    gap=lower(I(cut)-I(mu)-I(emu));assert gap>0
    lo=lower(I(mu)-I(emu)-I(residual)**2/I(gap));hi=up(I(mu)+I(emu))
    state_distance=up(mp.iv.sqrt(I(2))*I(residual)/I(gap))
    radius=I(0)
    for i,row in enumerate(h):
        r=abs(I(float(row[i].real)))+sum((abs(C(z.real,z.imag)) for j,z in enumerate(row) if j!=i),I(0))
        if r.b>radius.b:radius=r
    hbound=up(radius);alias=bool((I('.02')*I(hbound)+I('.4')).b<mp.iv.pi.a)
    return dict(energy_interval=[lo,hi],state_distance_upper=state_distance,normalization_error_upper=ev,
        Rayleigh=mu,Rayleigh_error_upper=emu,residual_upper=residual,inertia=ic,
        Hamiltonian_absolute_spectral_upper=hbound,alias_window='phase [-pi,pi), energy (-pi/.02,pi/.02]',
        all_spectrum_endpoint_margin_pass=alias,alias_proof='Gershgorin: delta0 ||H|| + .4 < pi; no E0 supplied to decoder'),v


def phase_upper(z,error):
    zz=C(z.real,z.imag);centre_angle=abs(mp.iv.atan2(zz.imag,zz.real))
    if error>=abs(z):return None
    ratio=I(error)/abs(zz)
    return up(centre_angle+mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2)))


def evaluate(dcert,tailcert):
    mp.iv.prec=BITS;c=common();h=hermitian(c['physical_H']);g,psi=ground_certificate(h)
    save_json(OUT/'GROUND_ALIAS.json',g)
    spectra=[]
    for a,(e,v) in zip(c['block_matrices'],c['spectra']):spectra.append(spectral_certificate(hermitian(a),e,v))
    te,tv=c['tail_spectra'][3];tcert=spectral_certificate(hermitian(c['tails'][3]),te,tv)
    save_json(OUT/'SPECTRAL_CERTIFICATES.json',dict(blocks=spectra,tail=tcert))
    prior=json.loads(subprocess.check_output(['git','show',BASE+':artifacts/track_c_compiler_a_18round/2026-11-06/run01/ROUNDS.json'],cwd=ROOT))
    result=[]
    for oldcase,formula,ld in zip(prior,['S2','Y4'],[3,12]):
        case_rows=[];reference={};unitary_powers=None
        for old in oldcase['rows']:
            r=old['r'] if ld==3 else 0
            if r not in reference:
                kernel=np.eye(70,dtype=complex);ek=0.;source=0.;tailerr=0.;logB=I(0)
                prep=c['preps'][ld];scalar=float(prep.constant_coefficient+prep.extracted_identity_coefficient)
                for w in weights(formula):
                    t=float(w*.02)
                    for seq in (range(ld+1),reversed(range(ld+1))):
                        if not isinstance(seq,range) and ld==3:
                            ut,et=spectral_unitary(hermitian(c['tails'][3]),te,tv,t,tcert)
                            kernel,ek=compose(ut,et,kernel,ek)
                        for j in seq:
                            e,v=c['spectra'][j];u,eu=spectral_unitary(hermitian(c['block_matrices'][j]),e,v,t/2,spectra[j])
                            kernel,ek=compose(u,eu,kernel,ek)
                            source=add(source,mul(abs(t/2),dcert[j]['generator_error_upper']),dcert[j]['diagonal_phase_error_uppers'][(t/2).hex()])
                    phase=expi(-I(t)*I(scalar));fp=centre(phase)
                    erp=up(abs(phase-C(fp.real,fp.imag)))
                    kernel,ek=compose(fp*np.eye(70),erp,kernel,ek)
                    source=add(source,up(abs(I(float(-t*scalar))+I(t)*I(scalar))))
                    if ld==3:
                        source=add(source,mul(abs(t),c['preps'][3].exact_rte_lambda_r,tailcert['generator_normalized_error_upper']))
                        n=max(1,math.ceil(abs(t)*r/.02));tau=I(c['preps'][3].exact_rte_lambda_r)*I(t)/n
                        dist=finite_rte_distribution(c['preps'][3].exact_rte_lambda_r*t/n,2)
                        masses=[mp.iv.sqrt(1+tau**2),tau**2/2*mp.iv.sqrt(1+(tau/3)**2)];B=sum(masses,I(0));logB+=n*mp.iv.log(B)
                        s=sum((I(x) for x in dist.order_probabilities),I(0));dp=sum((abs(I(x)/s-m/B) for x,m in zip(dist.order_probabilities,masses)),I(0))
                        angle=0.
                        for order in (0,2):
                            z=tau/(order+1);want=mp.iv.atan2(z,I(1));actual=math.atan(dist.dimensionless_step_time/(order+1))
                            angle=max(angle,up(abs(I(actual)-want)))
                        # Raw factor telescoping: both means are contractions.
                        rte=n*(mp.iv.exp(abs(tau))*abs(tau)**4/(24*B)+dp+I(angle))
                        tailerr=add(tailerr,up(rte))
                reference[r]=(kernel,ek,source,tailerr,logB)
                if unitary_powers is None:
                    unitary_powers=[(kernel.copy(),ek)]
                    for _ in range(17):
                        u,eu=unitary_powers[-1];unitary_powers.append(compose(u,eu,u,eu))
            kernel,ek,source,tailerr,logB=reference[r]
            # Binary exponent is itself certified against the same unitary.
            u,eu=unitary_powers[old['m']]
            q=old['q'];t=float(q*.02);upsi,edot=product(u,psi[:,None]);zhat=complex(np.vdot(psi,upsi[:,0]))
            ev=g['normalization_error_upper']
            evalerr=add(mul(eu,add(1,ev),add(1,ev)),mul(add(1,ev),edot),up(gamma(8*70+4)*I(fro(psi))*I(fro(upsi))),mul(ev,add(2,ev)),mul(2,g['state_distance_upper']))
            E=I(g['energy_interval']);reference_phase=expi(I(t)*E)
            zrot=C(zhat.real,zhat.imag)*reference_phase
            centre_z=centre(zrot);eref=up(abs(zrot-C(centre_z.real,centre_z.imag)))
            eref=add(eref,up(abs(I(t)-I(q)*I('.02'))*I(g['Hamiltonian_absolute_spectral_upper'])))
            pf=phase_upper(centre_z,add(evalerr,eref))
            attenuation=mp.iv.exp(-q*logB);rho0=lower((abs(C(zhat.real,zhat.imag))-I(evalerr))*attenuation)
            build=add(mul(q,source),mul(q,tailerr))
            compiler=old['raw_compiler_disk_upper'];phasegen=old['raw_source_phase_disk_upper']
            proposal=proposal_budget(0.,4*q*r,218,min(x.probability for x in c['preps'][3].rte_preparation.symbolic_tail.components))
            integer_cdf([x.probability for x in c['preps'][3].rte_preparation.symbolic_tail.components],proposal['bits_per_uniform_draw'])
            proposal_disk=up(2*mp.iv.sqrt(I(2))*I(4*q*r)*218*I(2)**(-proposal['bits_per_uniform_draw'])) if ld==3 else 0.
            proposal['directed_signal_error_upper']=proposal_disk
            extra=add(up(2*mp.iv.sqrt(I(2))*I(build)),compiler,phasegen,proposal_disk)
            beta_extra=None if extra>=rho0 else up(mp.iv.atan2(I(extra)/I(rho0),mp.iv.sqrt(1-(I(extra)/I(rho0))**2)))
            beta=None if pf is None or beta_extra is None else lower(I('.4')-I(pf)-I(beta_extra))
            rho=lower(I(rho0)-I(extra));passed=bool(g['all_spectrum_endpoint_margin_pass'] and beta is not None and beta>0 and rho>0)
            shots=None
            if passed:
                eps=I(rho)*mp.iv.sin(I(beta))/mp.iv.sqrt(I(2));nn=2*mp.iv.log(I(1440))/eps**2
                with mp.workprec(BITS):shots=int(mp.ceil(mp.mpf(nn._mpi_[1])))
            case_rows.append(dict(m=old['m'],q=q,r=r,PF_phase_upper=pf,PF_numeric_evaluation_disk_upper=evalerr,
                source_build_and_finite_RTE_operator_upper=build,raw_extra_disk_upper=extra,raw_radius_lower=rho,
                statistical_phase_lower=beta,shots_per_axis=shots,success_conditions_closed=passed,
                compiler_disk_inherited=compiler,event_phase_disk_inherited=phasegen,finite_bit_proposal=proposal,
                target='fixed saved Hermitian physical-H matrix; exact ground input; ideal native gates; fresh independent bits'))
        result.append(dict(name=oldcase['name'],rows=case_rows,all18_closed=all(x['success_conditions_closed'] for x in case_rows),
            total_shots=2*sum(x['shots_per_axis'] for x in case_rows) if all(x['shots_per_axis'] is not None for x in case_rows) else None))
    return result


def run():
    mp.iv.prec=BITS
    if (OUT/'STOP.json').exists():raise RuntimeError('Stopped A; preserve output')
    OUT.mkdir(parents=True,exist_ok=True);common();start=time.monotonic()
    with ProcessPoolExecutor(max_workers=min(14,len(os.sched_getaffinity(0)))) as pool:
        fs=[pool.submit(d_task,j) for j in range(13)];tail=pool.submit(tail_task)
        dc=sorted([f.result() for f in fs],key=lambda x:x['index']);tc=tail.result()
    save_json(OUT/'SOURCE_GENERATORS.json',dict(deterministic=dc,tail=tc))
    print('A SOURCE generator bounds',max(x['generator_error_upper'] for x in dc),tc,flush=True)
    rows=evaluate(dc,tc);save_json(OUT/'ROUNDS.json',rows)
    status='CLOSED_FIXED_MATRIX_IID_IDEAL_GATE_MODEL' if all(x['all18_closed'] for x in rows) else 'UNQUALIFIED'
    save_json(OUT/'SUMMARY.json',dict(base_commit=BASE,status=status,interval_bits=BITS,
        model_scope='fixed saved Hermitian Hamiltonian representation, supplied exact ground, ideal reference-A gates, independent finite-bit quantum proposal',
        original_integral_representation_guarantee='not newly certified; frozen H4 input correspondence inherited',
        price_stream_independent=True,old_ground_solve_repeated=False,old_input_matching_repeated=False,
        new_SCF=0,new_DF_fit=0,new_ground_solve=0,wall_seconds=time.monotonic()-start))
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',science_workers_remaining=0,automatic_resume=False))
    print('A complete',status,flush=True)
