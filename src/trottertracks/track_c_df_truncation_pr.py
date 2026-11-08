"""Molecule-specific PR pilots; no H4 coefficient or sampling-price transfer.

Closed P3 is the existing G7 generator. Its exact probabilities, coefficients,
digital sampler and weighted range certificate are reused without changing law.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from fractions import Fraction as F
import hashlib
import json
import math
import os
import mpmath as mp
from bisect import bisect_right
import numpy as np
from scipy.linalg import eigh,eig
from qiskit import QuantumCircuit
from . import track_c_df_truncation as d
from . import track_c_closed_p3 as cp
from .track_c_h4_price import conditional_flags
from .track_c_bottleneck_basis import partial_ops
from .track_c_joint_isometry import encode,decode

OUT=d.OUT


@lru_cache(None)
def partition(n,ld):
    c=d.model(n);blocks=d.kept(c,dict(drop=[]));D=blocks[:ld+1];R=blocks[ld+1:];symbols=[];scalar=F.from_float(c['constant'])
    tail=np.zeros_like(c['H'])
    for b in R:
        scalar+=b['coeff'].get((),F(0));co={s:v for s,v in b['coeff'].items() if s and v}
        diag=sum((float(v)*c['signs'][s] for s,v in co.items()),np.zeros(len(c['H'])))
        tail+=(b['B']*diag)@b['B'].conj().T
        symbols.extend(dict(block=b['index'],support=s,sign=1 if v>0 else -1,abs=abs(v)) for s,v in co.items())
    lam=sum((s['abs'] for s in symbols),F(0));p=tuple(s['abs']/lam for s in symbols);e,v=eigh(d.sg.hermitian(tail))
    return dict(D=D,R=R,tail=tail,spectrum=(e,v),symbols=symbols,lam=lam,p=p,scalar=scalar)


def overhead(n,ld,t,axis):
    c={**d.model(n),'constant':float(partition(n,ld)['scalar'])}
    return d.wrapper(c,t,axis)


@lru_cache(None)
def generator(n,ld,h,r):
    p=partition(n,ld)
    return cp.CachedCanonical(p['p'],p['lam']*F.from_float(h)/r,3,'closed_P3_tail',root_bits=cp.BITS,probability_bits=cp.BITS,eta=cp.ETA,rho=cp.RHO)


def probe_one(task):
    n,ld,kind,h=task;c=d.model(n);p=partition(n,ld);case=dict(kind=kind,order=list(range(ld+1)))
    if kind=='Dtuned':case['v']=json.loads((OUT/f'H{n}/BCH_LD{ld}.json').read_text())['selected_v']
    seq=d.schedule(case,h);u=np.eye(len(c['H']),dtype=complex)
    e,v=p['spectrum'];look={b['index']:b for b in p['D']}
    for role,i,t in seq:
        if role=='R':a=(v*np.exp(-1j*t*e))@v.conj().T
        else:
            b=look[i];a=(b['B']*np.exp(-1j*t*b['diagonal']))@b['B'].conj().T
        u=a@u
    u*=np.exp(-1j*h*float(p['scalar']))
    es,vs=eig(u);overlap=abs(vs.conj().T@c['state'])**2;k=int(np.argmax(overlap));pk=float(overlap[k]);bias=-float(np.angle(es[k]*np.exp(1j*c['energy']*h)))/h
    leak=math.asin(min(1.,(1-pk)/pk)) if pk>.5 else math.pi
    viable=[m for m in range(18) if h<=.02*2**m+1e-15 and abs(bias)*.02*2**m+leak<.30]
    row=dict(n=n,ld=ld,kind=kind,h=h,case=case,source=d.source(),tail_lambda_exact=str(p['lam']),components=len(p['p']),dominant_overlap=pk,dominant_energy_bias=bias,nonresonant_leak_bound=leak,viable_rounds=viable)
    if viable:
        groups=[[]]
        for x in seq:
            if x[0]=='R':groups.append([])
            else:groups[-1].append(x)
        from .track_c_mixed_formula import compact
        sequences=dict(first=groups[0],last=groups[-1],junction=compact(groups[-1]+groups[0]))
        row.update(groups={},angle_errors={},variants={})
        for name,s in sequences.items():row['groups'][name],row['angle_errors'][name],row['variants'][name],_=d.price_piece(c,p['D'],s)
    d.save_json(OUT/f'H{n}/pr_probe/ld{ld}_{kind}_{h}.json',row);return row


def bch_one(task):
    from .track_c_mixed_formula import cubic_generator
    n,ld=task;c=d.model(n);p=partition(n,ld);matrices=[d.sg.hermitian(b['matrix']) for b in p['D']]+[p['tail']];values={}
    for kind in ('D2','D4'):
        case=dict(kind=kind,order=list(range(ld+1)));C=cubic_generator(d.schedule(case),matrices);values[kind]=float(np.vdot(c['state'],C@c['state']).real)
    difference=values['D2']-values['D4'];selected=None;roots=[]
    if abs(difference)>1e-15:
        ratio=-values['D4']/difference
        roots=[float(x.real) for x in np.roots([-6.,12.,-6.,1-ratio]) if abs(x.imag)<1e-8 and -2<=x.real<=2]
        if roots:selected=min(roots,key=lambda v:(2*abs(v)+abs(1-2*v),abs(v)))
    row=dict(n=n,ld=ld,source=d.source(),BCH_ground_coefficients=values,real_roots_in_registered_interval=roots,selected_v=selected,domain=[-2,2],method='existing cubic_generator on actual molecule-specific D2/D4 sequence; f(v)=2v^3+(1-2v)^3, cancel scalar leading coefficient; validate actual finite-h signal afterwards')
    if selected is not None:
        case=dict(kind='Dtuned',order=list(range(ld+1)),v=selected);C=cubic_generator(d.schedule(case),matrices);row['direct_tuned_ground_coefficient']=float(np.vdot(c['state'],C@c['state']).real)
        assert abs(row['direct_tuned_ground_coefficient'])<1e-8*max(1.,abs(values['D2']),abs(values['D4']))
    d.save_json(OUT/f'H{n}/BCH_LD{ld}.json',row);return row


def bch():
    tasks=[(n,ld) for n in (2,6) for ld in ([0,1,2] if n==2 else [3,9,14,17])]
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
        for row in pool.map(bch_one,tasks):print('BCH',row['n'],row['ld'],row['selected_v'],flush=True)


def probes():
    tasks=[];expected=0;reused=0
    for n in (2,6):
        for ld in ([0,1,2] if n==2 else [3,9,14,17]):
            kinds=['ordered','D4'];path=OUT/f'H{n}/BCH_LD{ld}.json'
            if path.exists() and json.loads(path.read_text())['selected_v'] is not None:kinds.append('Dtuned')
            for kind in kinds:
                for h in (.0025,.005,.01,.02,.04,.08,.16,.32,.64,1.28,2.56):
                    expected+=1
                    if (OUT/f'H{n}/pr_probe/ld{ld}_{kind}_{h}.json').exists():reused+=1
                    else:tasks.append((n,ld,kind,h))
    if not tasks:return
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
        for j,f in enumerate(as_completed(pool.submit(probe_one,t) for t in tasks)):
            r=f.result()
            if j%20==0:print('PR_PROBE',j+1,len(tasks),r['n'],r['ld'],flush=True)
    name='PR_PROBE_EXTENSION_STATUS.json' if (OUT/'PR_PROBE_STATUS.json').exists() else 'PR_PROBE_STATUS.json'
    d.save_json(OUT/name,dict(source=d.source(),tasks=len(tasks),expected=expected,reused=reused,D_families=['ordered','D4','molecule-specific Dtuned if a finite root exists'],scope='same dyadic h menu as deterministic methods; finite per-molecule menu; no H4 tuned coefficients or catalogue transferred'))


def isometry_key(n,index,support):return f'{n}:{index}:'+','.join(map(str,support))


def isometry_one(task):
    n,index,support=task;c=d.model(n);b=c['blocks'][index];ops=partial_ops(b['sp'],support);modes=c['modes'];s=d.sg;mp.iv.prec=192
    a=[[s.C(int(i==j)) for j in range(modes)] for i in range(modes)]
    for gate,qs in ops:
        if gate.name=='rz':a[qs[0]]=[z*s.expi(s.I(float(gate.params[0]))) for z in a[qs[0]]]
        else:
            q,r=qs;local=d.ca.source_matrix(d.ca.spec(gate.name,gate.params));x,y=a[q],a[r]
            a[q]=[local[1][1]*v+local[1][2]*w for v,w in zip(x,y)];a[r]=[local[2][1]*v+local[2][2]*w for v,w in zip(x,y)]
    col=s.up(mp.iv.sqrt(sum((abs(a[i][j]-s.C(b['sp'][i,k].real,b['sp'][i,k].imag))**2 for j,k in enumerate(support) for i in range(modes)),s.I(0))))
    old=json.loads((OUT/f'H{n}/certificates/basis_{index}.json').read_text())['one_particle_error'];epsilon=s.add(col,old)
    # Q=I-2 b_u^dagger b_u. Fermionic creation/annihilation norms are the
    # one-particle vector norm; a product of at most k commuting Q's telescopes.
    qerror=s.up(8*len(support)*s.I(epsilon)*(1+s.I(epsilon)))
    row=dict(key=isometry_key(n,index,support),n=n,index=index,support=list(support),targets=list(range(len(support))),operations=encode(ops),column_error_upper=epsilon,source_Q_error_upper=qerror,adopt=epsilon<1e-10,source=d.source())
    return row


def build_isometries():
    tasks=set()
    for n in (2,6):
        # All singleton/pair supports of every registered residual. Larger run
        # unions use the original full basis. No H4 catalogue entry is reused.
        for ld in ([0,1,2] if n==2 else [3,9,14,17]):
            for sym in partition(n,ld)['symbols']:tasks.add((n,sym['block'],tuple(sym['support'])))
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:entries=list(pool.map(isometry_one,sorted(tasks)))
    cat={}
    for row in entries:
        if row['adopt']:
            for gate,_ in decode(row['operations']):
                if gate.name=='xx_plus_yy':
                    for g in (gate,gate.inverse()):spec=d.ca.spec(g.name,g.params);cat[spec['key']]=spec
    contexts=[d.ca.context_compile((mask,cat)) for mask in (1,9)];unique={r['lowering_sha256']:r for ct in contexts for r in ct['records']}
    with ProcessPoolExecutor(max_workers=d.workers(len(unique))) as pool:checked=list(pool.map(d.ca.interval_check,unique.values()))
    bounds={}
    for x in checked:bounds[x['source_key']]=max(bounds.get(x['source_key'],0.),x['operator_error_upper'])
    maximum=0.
    for row in entries:
        if not row['adopt']:
            row['basis_native_error_upper']=None
            continue
        total=0.
        for g,_ in decode(row['operations']):
            if g.name=='xx_plus_yy':total=d.sg.add(total,bounds[d.ca.key(g.name,g.params)],bounds[d.ca.key(g.inverse().name,g.inverse().params)])
        row['basis_native_error_upper']=total;maximum=max(maximum,total)
    d.save_json(OUT/'PR_ISOMETRIES.json',dict(source=d.source(),entries={r['key']:r for r in entries},maximum_native_basis_error=maximum,maximum_Q_source_error=max(r['source_Q_error_upper'] for r in entries if r['adopt']),scope='per-molecule new singleton/pair QR; larger/unregistered unions fall back to original basis; column/source/native bounds cover all adoption branches'))
    d.save_json(OUT/'PR_ISOMETRY_COMPILER.json',dict(source=d.source(),contexts=contexts,checked=checked,bounds=bounds))
    print('ISOMETRIES',len(entries),len(checked),flush=True)


@lru_cache(None)
def isometries():
    path=OUT/'PR_ISOMETRIES.json'
    return json.loads(path.read_text()) if path.exists() else dict(entries={},maximum_native_basis_error=0.,maximum_Q_source_error=0.)


@lru_cache(None)
def label_price(n,ld):
    c=d.model(n);p=partition(n,ld);price=0.
    for i,b in enumerate(p['R']):
        q=QuantumCircuit(c['modes']+1)
        for gate,qs in reversed(b['ops']):q.append(gate.inverse(),qs)
        d.diagonal(q,{(0,):F(1,8)},1.,c['modes'],'network')
        for gate,qs in b['ops']:q.append(gate,qs)
        full=d.metric(d.compile_reference(q))[0]
        # A proxy only; complete occurrences are directly compiled later.
        for sym,prob in zip(p['symbols'],p['p']):
            if sym['block']!=b['index']:continue
            entry=isometries()['entries'].get(isometry_key(n,b['index'],sym['support']));cost=full
            if entry and entry['adopt']:
                ops=decode(entry['operations']);temp=QuantumCircuit(c['modes']+1)
                for gate,qs in reversed(ops):temp.append(gate.inverse(),qs)
                d.diagonal(temp,{tuple(entry['targets']):F(1,8)},1.,c['modes'],'network')
                for gate,qs in ops:temp.append(gate,qs)
                cost=min(cost,d.metric(d.compile_reference(temp))[0])
            price+=float(prob)*cost
    return price


def screen(n):
    c=d.model(n);points=[]
    for path in sorted((OUT/f'H{n}/pr_probe').glob('*.json')):
        row=json.loads(path.read_text());h=row['h'];lam=float(F(row['tail_lambda_exact']));g=row.get('groups',{})
        for m in row['viable_rounds']:
            steps=round(.02*2**m/h);phase=abs(row['dominant_energy_bias'])*.02*2**m+row['nonresonant_leak_bound']
            choices=[]
            for r in (1,2,4,8,16,32,64,128,256,512,1024,2048,4096):
                x=lam*h/r;logB=steps*r*math.log(math.sqrt(1+x*x)+x*x/2*math.sqrt(1+(x/3)**2));eta=steps*r*math.exp(x)*x**4/24
                if x<=1 and logB<1.2 and eta<.02:
                    rho=(2*row['dominant_overlap']-1)*math.exp(-logB);beta=.4-phase-.03
                    if beta<=0:continue
                    D=np.array(g['first'])+np.array(g['last'])+(steps-1)*np.array(g['junction']);over=np.array([overhead(n,row['ld'],.02*2**m,a) for a in ('X','Y')])
                    estimate=steps*r*label_price(n,row['ld']);mean=D+over+np.array([estimate,0,0,0])
                    choices.append(dict(key=[row['ld'],row['kind'],h,r,m],signal=dict(m=m,name=f'LD{row["ld"]}_{row["kind"]}',h=h,steps=steps,r=r),qualification=dict(closed=True,rho_lower=rho,beta_stat_lower=beta),mean=mean,row=row))
            if choices:points.append(min(choices,key=lambda p:sum(d.rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],1/720)*p['mean'][b,0] for b in (0,1))))
    return points


def fast_product(a,b):
    return a@b,d.sg.up(d.sg.gamma(8*a.shape[-1]+4)*d.sg.I(d.fast_fro(a))*d.sg.I(d.fast_fro(b)))


def residual(a,b):
    return d.sg.add(d.fast_fro(a-b),d.sg.up(d.sg.gamma(2)*(d.sg.I(d.fast_fro(a))+d.sg.I(d.fast_fro(b)))))


def spectral_certificate(H,e,v):
    s=d.sg;mp.iv.prec=192;vv,em=fast_product(v.conj().T,v);delta=s.add(residual(vv,np.eye(len(e))),em);assert delta<1
    nv=s.up(mp.iv.sqrt(1+s.I(delta)));polar=s.up(s.I(delta)/(1+mp.iv.sqrt(1-s.I(delta))))
    hv,eh=fast_product(H,v);vd=v*e[None,:];ed=s.up(s.gamma(2)*s.I(d.fast_fro(v))*max(abs(e)))
    rr=s.add(residual(hv,vd),eh,ed);gen=s.add(s.mul(rr,nv),s.mul(d.fast_fro(H),delta),s.mul(max(abs(e)),polar,s.add(nv,1)))
    return dict(generator_error_upper=gen,polar_distance_upper=polar,V_norm_upper=nv,unitarity_defect_upper=delta)


def unitary(H,e,v,t,sc):
    s=d.sg;ph=[s.expi(-s.I(t)*s.I(float(x))) for x in e];values=np.array([s.centre(x) for x in ph]);er=max(s.up(abs(x-s.C(z.real,z.imag))) for x,z in zip(ph,values))
    out,em=fast_product(v*values[None,:],v.conj().T)
    error=s.add(s.mul(abs(t),sc['generator_error_upper']),s.mul(sc['polar_distance_upper'],s.add(sc['V_norm_upper'],1)),s.mul(er,sc['V_norm_upper'],sc['V_norm_upper']),s.mul(s.up(s.gamma(8)*s.I(d.fast_fro(v))),sc['V_norm_upper']),em)
    return out,error


@lru_cache(None)
def block_data(n,index,tail=False):
    c=d.model(n);b=c['blocks'][index];co={s:v for s,v in b['coeff'].items() if s or not tail}
    diag=sum((float(v)*c['signs'][s] for s,v in co.items()),np.zeros(len(c['H'])));H=d.sg.hermitian((b['B']*diag)@b['B'].conj().T);e,v=eigh(H);sc=spectral_certificate(H,e,v)
    eb=json.loads((OUT/f'H{n}/certificates/basis_{index}.json').read_text())['exterior_error'];de=0.
    for j in range(len(H)):
        exact=sum((cp.iv(v)*int(c['signs'][s][j]) for s,v in co.items()),d.sg.I(0));de=max(de,d.sg.up(abs(exact-d.sg.I(diag[j]))))
    norm=d.sg.up(1+d.sg.I(eb));maxd=max(abs(diag));fullnorm=d.sg.up(sum((cp.iv(abs(v)) for v in co.values()),d.sg.I(0)))
    gen=d.sg.add(d.sg.mul(fullnorm,eb,d.sg.add(2,eb)),d.sg.mul(de,norm,norm),d.sg.up(d.sg.gamma(8*len(H)+6)*len(H)*d.sg.I(maxd)*d.sg.I(norm)**2))
    bounds=json.loads((OUT/'COMPILER_INTERVALS.json').read_text())['bounds'];native=0.
    for g,_ in b['ops']:
        if g.name=='xx_plus_yy':native=d.sg.add(native,bounds[d.ca.key(g.name,g.params)],bounds[d.ca.key(g.inverse().name,g.inverse().params)])
    return H,e,v,sc,gen,native


@lru_cache(None)
def ground(n):
    c=d.model(n);s=d.sg;H=c['H'];e,v=eigh(H);sc=spectral_certificate(H,e,v);err=s.add(sc['generator_error_upper'],abs(float(e[0])-c['energy']));gap=s.lower(s.I(e[1])-s.I(e[0])-2*s.I(err));assert gap>0
    psi=c['state'];hv=H@psi;res=s.add(d.fast_fro(hv-c['energy']*psi),s.up(s.gamma(8*len(H)+6)*(s.I(d.fast_fro(H))+abs(s.I(c['energy'])))*s.I(d.fast_fro(psi))))
    state=s.up(mp.iv.sqrt(s.I(2))*s.I(res)/s.I(gap));normdef,normupper=d.state_norm_bounds(psi);return err,state,normdef,normupper


def qualify_point(point,n):
    c=d.model(n);row=point['row'];m=point['signal']['m'];h=row['h'];r=point['signal']['r'];steps=point['signal']['steps'];p=partition(n,row['ld']);s=d.sg;mp.iv.prec=192;t=.02*2**m;size=len(c['H'])
    gg=generator(n,row['ld'],h,r);gc=cp.certify(gg);d.save_json(OUT/f'H{n}/pr_cells/{row["ld"]}_{row["kind"]}_{h}_{r}_{m}.json',gc)
    u=np.eye(size,dtype=complex);eu=0.;src=0.;nativeD=0.
    for kind,i,time_ in d.schedule(row['case'],h):
        if kind=='D':H,e,v,sc,gen,native=block_data(n,i);src=s.add(src,s.mul(abs(time_),gen));nativeD=s.add(nativeD,native)
        else:
            H=p['tail'];e,v=p['spectrum'];sc=spectral_certificate(H,e,v)
            gen=sum(block_data(n,b['index'],True)[4] for b in p['R']);gen=s.add(gen,s.up(s.gamma(2*len(p['R'])+2)*sum(d.fast_fro(block_data(n,b['index'],True)[0]) for b in p['R'])))
            src=s.add(src,s.mul(abs(time_),gen))
        a,ea=unitary(H,e,v,time_,sc);u,eu=s.compose(a,ea,u,eu)
    ph=s.expi(-s.I(h)*cp.iv(p['scalar']));centre=s.centre(ph);u,eu=s.compose(centre*np.eye(size),s.up(abs(ph-s.C(centre.real,centre.imag))),u,eu)
    q=steps;power=np.eye(size,dtype=complex);ep=0.
    while q:
        if q&1:power,ep=s.compose(u,eu,power,ep)
        q//=2
        if q:u,eu=s.compose(u,eu,u,eu)
    z=complex(np.vdot(c['state'],power@c['state']));ge,se,normdef,normupper=ground(n);evalerr=s.add(s.mul(ep,s.add(1,normdef)),s.up(s.gamma(16*size+8)*size*(1+s.I(ep))),s.mul(2,se,max(1.,normupper)),normdef)
    rotated=s.C(z.real,z.imag)*s.expi(s.I(t)*s.I(c['energy']));pf=s.phase_upper(s.centre(rotated),s.add(evalerr,s.mul(t,ge)))
    x=cp.iv(p['lam'])*s.I(h)/r;M=steps*r
    tailerr=s.up(mp.iv.exp(M*mp.iv.exp(abs(x))*abs(x)**4/24)-1)
    rooterr=s.mul(M,gc['corrected_mean_bias_step_upper'])
    # Convert/multiply weights, then normalize once. Lower and upper event-weight
    # enclosures certify every prefix stays normal and finite. Gamma covers
    # conversions, multiplications and final normalization, and inflates range.
    weight=s.up(s.gamma(2*M+4));logL=s.up(M*s.I(gc['log_Wmax'][1])+mp.iv.log(1+s.I(weight)));L=s.up(mp.iv.exp(s.I(logL)))
    B=sum((mp.iv.sqrt(cp.iv(group.square)) for group in gg.groups),s.I(0));wmin=B*cp.iv(1-cp.RHO)/cp.iv(1+cp.ETA)**4
    lowlog=M*mp.iv.log(wmin)+M*mp.iv.log(1-s.I(2)**-53)+mp.iv.log(1-s.I(weight))
    prefix_lower=min(0.,s.lower(lowlog));prefix_upper=max(0.,logL);prefix_valid=prefix_lower> -600 and prefix_upper<600
    maxbasis=max(max(block_data(n,b['index'])[5] for b in p['R']),isometries()['maximum_native_basis_error']);native=s.add(s.mul(steps,nativeD),s.mul(M,3,maxbasis))
    native=s.add(native,s.mul(2,json.loads((OUT/'COMPILER_INTERVALS.json').read_text())['bounds'][d.ca.key('h')]),json.loads((OUT/'COMPILER_INTERVALS.json').read_text())['bounds'][d.ca.key('sdg')])
    # Native parity products use binary64 pi and finite phase accumulation.
    phase=s.up(s.I(M)*32*(abs(s.I(math.pi)-mp.iv.pi)+s.gamma(8)*mp.iv.pi))
    ang=row['angle_errors'];angle=s.add(ang['first'],ang['last'],s.mul(steps-1,ang['junction']))
    timing=s.up(abs(s.I(t)-s.I('.02')*(1<<m))*s.I(d.fast_fro(c['H'])))
    joint=s.mul(M,3,isometries()['maximum_Q_source_error'])
    extra=s.add(s.up(s.I(tailerr)/s.I(L)),rooterr,s.mul(steps,src),native,phase,angle,weight,s.up(s.gamma(4)*abs(cp.iv(p['scalar'])*s.I(t))),timing,joint)
    disk=s.up(2*mp.iv.sqrt(2)*s.I(extra));radius=s.lower((abs(s.C(z.real,z.imag))-s.I(evalerr))/s.I(L));rho=s.lower(s.I(radius)-s.I(disk))
    ratio=s.I(disk)/s.I(radius) if radius>disk else None;pe=s.up(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))) if ratio is not None else None
    beta=s.lower(s.I('.4')-s.I(pf)-s.I(pe)) if pf is not None and pe is not None else None
    alias=bool((s.I('.02')*s.I(d.fast_fro(c['H']))+s.I('.4')).b<mp.iv.pi.a)
    q=dict(closed=bool(alias and rho>0 and beta is not None and beta>0 and prefix_valid and gc['log_Wmax'][0]>=0),rho_lower=rho,beta_stat_lower=beta,PF_phase_upper=pf,finite_RTE_error_upper=tailerr,root_angle_error_upper=rooterr,source_disk=s.mul(steps,src),native_disk=native,phase_disk=phase,angle_disk=angle,weight_disk=weight,joint_source_disk=joint,numerical_disk=evalerr,coordinate_disk=disk,log_range=logL,range_upper=L,weight_prefix_log_bounds=[prefix_lower,prefix_upper],weight_prefix_valid=prefix_valid,signal=[z.real,z.imag],alias=alias,
        estimator='weighted Closed P3, normalized once by certified path range; 256-bit proposals; no rejection/zero-fill; binary64 conversion/product/normalization charged',qualification='fixed saved original H; exact original ground; ideal native; fresh independent bits')
    record={**point,'mean':point['mean'].tolist(),'qualification':q,'cell':[n,row['ld'],h,r]}
    d.save_json(OUT/f'H{n}/pr_qualified/{row["ld"]}_{row["kind"]}_{h}_{r}_{m}.json',record);return record


def shortlist():
    tasks={}
    for n in (2,6):
        points=screen(n);opt=d.choose(points)
        if opt is None:continue
        for m in range(18):
            local=sorted((p for p in points if p['signal']['m']==m),key=lambda p:sum(d.rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],1/720)*p['mean'][b,0] for b in (0,1)))[:2]
            for p in local+[opt[1][m]]:tasks[(n,tuple(p['key']))]=(p,n)
    d.save_json(OUT/'PR_SHORTLIST.json',dict(source=d.source(),keys=[dict(n=n,key=list(k)) for n,k in tasks],screen='canonical range sufficient bound and per-label standalone price proxy; final Closed P3 qualifies independently and prices complete occurrences'))
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
        for j,q in enumerate(pool.map(qualification_task,tasks.values())):print('PR_QUALIFIED',j+1,len(tasks),q['qualification']['closed'],flush=True)


def qualification_task(task):
    point,n=task
    try:return qualify_point(point,n)
    except Exception as error:
        # The user explicitly requests continuing feasible molecules/points.
        # Preserve a failed attempt without inserting a fabricated numerical row.
        record=dict(n=n,key=point['key'],source=d.source(),exception=type(error).__name__,message=str(error),qualification=dict(closed=False),status='TECHNICAL_OR_SCIENTIFIC_UNRESOLVED')
        name='_'.join(map(str,point['key']))
        d.save_json(OUT/f'H{n}/pr_failures/{name}_{d.source()[:12]}.json',record)
        return record


def root_phase(q,qubit,power):
    """All signs are relative phases on the original control ancilla."""
    if power:q.p(float(power)*math.pi/2,qubit)


def normalized_measurement(records,range_upper,outcome):
    """Classical estimator interface for one fresh trajectory and a ±1 readout.

    Event generation uses DigitalSampler; stratified price MC is a separate
    procedure. A long path may be streamed, without constructing a giant circuit.
    """
    if outcome not in (-1,1):raise ValueError('Hadamard outcome must be ±1')
    weight=1.
    for record in records:
        value=float(record['weight'])
        if value<=0 or not math.isfinite(value):raise ArithmeticError('outside registered positive-weight domain')
        weight*=value
    result=outcome*weight/range_upper
    if not math.isfinite(result) or abs(result)>1:raise ArithmeticError('outside certified normalized range')
    return result


def controlled_z(q,a,b):
    # Fully native parity polynomial; common global phase has no new control.
    q.rz(math.pi/2,a);q.rz(math.pi/2,b);q.cx(a,b);q.rz(-math.pi/2,b);q.cx(a,b);q.global_phase+=math.pi/4


@lru_cache(None)
def first_nonroot_cdf(root_count,r):
    """Exact conditioning of the existing dyadic law, for price MC only."""
    D=1<<cp.BITS;weight=(D-root_count)<<(cp.BITS*(r-1));total=0;table=[]
    for _ in range(r):
        total+=weight;table.append(total);weight=weight*root_count//D
    assert total==(1<<(cp.BITS*r))-root_count**r
    return tuple(table)


def conditioned_records(g,r,rare,bits):
    """Exact integer conditional strata of G7; the quantum sampling law is unchanged."""
    sampler=cp.DigitalSampler(g);root=next(i for i,x in enumerate(g.groups) if x.mode=='root_collected');D=1<<cp.BITS
    counts=[int(p*D) for p in g.group_law];assert sum(counts)==D
    root_count=counts[root];rare_count=D-root_count;flags=[False]*r
    if rare:
        table=first_nonroot_cdf(root_count,r);first=bisect_right(table,bits.rng.randrange(table[-1]));flags[first]=True
        for j in range(first+1,r):flags[j]=bits.bits(cp.BITS)<rare_count
    others=[i for i in range(len(g.groups)) if i!=root];running=0;conditional=[]
    for i in others:running+=counts[i];conditional.append(running)
    assert running==rare_count
    records=[]
    for flag in flags:
        if not flag:gi=root;word=();child=sampler.draw(sampler.root,bits)
        else:
            gi=others[bisect_right(conditional,bits.rng.randrange(rare_count))];group=g.groups[gi];word=(group.first,sampler.conditional_draw(group.first,bits));child=sampler.conditional_draw(group.first,bits)
        records.append(g.event(gi,word,child))
    root_probability=F(root_count,D)**r
    return records,1-root_probability if rare else root_probability,root_count


def event_circuit(n,ld,records,mode,basis_mode='joint'):
    c=d.model(n);p=partition(n,ld);symbols=p['symbols'];flat=[];phase=0
    for rec in records:
        assert rec['proposal']*rec['weight']==rec['coefficient']
        word=tuple(rec['word']);child=int(rec['child']);phase+=int(rec['phase_i_power'])
        for label in reversed(word):
            s=symbols[label];phase+=2*(s['sign']==-1);flat.append((s['block'],'product',s['support'],0.))
        s=symbols[child];theta,_=cp.angle(rec['ratio']);flat.append((s['block'],'rotation',s['support'],s['sign']*theta))
    q=QuantumCircuit(c['modes']+1);cursor=0;error=0.
    while cursor<len(flat):
        block=flat[cursor][0];stop=cursor+1
        while stop<len(flat) and flat[stop][0]==block:stop+=1
        support=tuple(sorted({i for _,_,s,_ in flat[cursor:stop] for i in s}));entry=isometries()['entries'].get(isometry_key(n,block,support));mapping=None
        if basis_mode=='joint' and entry and entry['adopt']:
            ops=decode(entry['operations']);mapping=dict(zip(entry['support'],entry['targets']))
        else:ops=c['blocks'][block]['ops']
        co={};parity=set()
        for _,role,s,angle in flat[cursor:stop]:
            if mapping is not None:s=tuple(sorted(mapping[i] for i in s))
            if role=='rotation':co[s]=co.get(s,F(0))+F.from_float(angle)
            else:
                for i in s:
                    if i in parity:parity.remove(i)
                    else:parity.add(i)
        for g,qs in reversed(ops):q.append(g.inverse(),qs)
        error+=d.diagonal(q,co,1.,c['modes'],mode)
        for i in sorted(parity):controlled_z(q,c['modes'],i)
        for g,qs in ops:q.append(g,qs)
        cursor=stop
    root_phase(q,c['modes'],phase)
    return q,error


def price_one(task):
    n,ld,h,r,rare,j,phase=task;g=generator(n,ld,h,r)
    ident=f'{n}:{ld}:{float(h).hex()}:{r}:{rare}:{j}:{phase}';seed=int.from_bytes(hashlib.sha256(('DF_truncation_PR:'+ident).encode()).digest()[:8],'big');bits=cp.BitStream(seed)
    records,probability,root_count=conditioned_records(g,r,rare,bits)
    variants={};errs={}
    for mode in ('term_parity','network'):
        for basis_mode in ('full','joint'):
            q,err=event_circuit(n,ld,records,mode,basis_mode);key=basis_mode+'_'+mode;variants[key]=d.metric(d.compile_reference(q)).tolist();errs[key]=err
    chosen=min(variants,key=lambda k:tuple(variants[k]))
    row=dict(n=n,ld=ld,h=h,r=r,rare=rare,j=j,phase=phase,seed=seed,probability=float(probability),stratum_law=dict(root_count=str(root_count),denominator_bits=cp.BITS,events=r,rare=rare),metrics=variants[chosen],selected=chosen,variants=variants,angle_error=errs[chosen],
        records=[dict(group=x['group'],word=list(x['word']),child=x['child']) if 'group' in x else dict(word=list(x['word']),child=x['child']) for x in records],source=d.source())
    d.save_json(OUT/f'H{n}/pr_{phase}/{ld}_{h}_{r}_{int(rare)}_{j}.json',row);return row


def price_cells(phase):
    if phase=='training':
        points=[json.loads(p.read_text()) for n in (2,6) for p in (OUT/f'H{n}/pr_qualified').glob('*.json')]
        cells=sorted({tuple(p['cell']) for p in points if p['qualification']['closed']});samples=32
    else:
        policy=json.loads((OUT/'PR_POLICY.json').read_text());cells=sorted({tuple(p['cell']) for pol in policy['policies'] if pol['status']=='FROZEN' for p in pol['selected']});samples=64
    tasks=[(*cell,rare,j,phase) for cell in cells for rare in (False,True) for j in range(samples)]
    with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:
        for j,row in enumerate(pool.map(price_one,tasks)):
            if j%64==0:print('PR_PRICE',phase,j+1,len(tasks),flush=True)
    d.save_json(OUT/f'PR_{phase.upper()}_STATUS.json',dict(source=d.source(),cells=cells,samples_per_stratum=samples,trajectories=len(tasks),MC='rare/no-rare stratified, complete occurrences; molecule-specific generator and seed namespace'))


def samples(phase):
    data={}
    for n in (2,6):
        for path in (OUT/f'H{n}/pr_{phase}').glob('*.json'):
            r=json.loads(path.read_text());key=(n,r['ld'],r['h'],r['r']);data.setdefault(key,{}).setdefault(r['rare'],[]).append(r)
    for cell in data.values():
        for rs in cell.values():rs.sort(key=lambda r:r['j'])
    return data


def priced(point,rs):
    c=d.model(point['cell'][0]);row=point['row'];g=row['groups'];steps=point['signal']['steps'];cell=tuple(point['cell']);R=sum((sub[0]['probability']*np.array([r['metrics'] for r in sub]).mean(axis=0) for sub in rs[cell].values()),np.zeros(4))*steps
    D=np.array(g['first'])+np.array(g['last'])+(steps-1)*np.array(g['junction']);over=np.array([overhead(point['cell'][0],row['ld'],.02*2**point['signal']['m'],a) for a in ('X','Y')],float)
    return {**point,'D':D.tolist(),'R':R.tolist(),'overhead':over.tolist(),'mean':D+R+over}


def freeze():
    rs=samples('training');policies=[]
    for n in (2,6):
        ps=[priced(json.loads(p.read_text()),rs) for p in sorted((OUT/f'H{n}/pr_qualified').glob('*.json')) if json.loads(p.read_text())['qualification']['closed']]
        best=d.choose(ps)
        if best is None:policies.append(dict(n=n,family='PR',status='UNRESOLVED',reason='no qualified point at every round'));continue
        cost,selected,alpha,ns=best;selected=[{**p,'mean':p['mean'].tolist()} for p in selected]
        policies.append(dict(n=n,family='PR',status='FROZEN',selected=selected,alpha=alpha.tolist(),trials=ns.tolist(),training_RZ=cost,trials_total=int(ns.sum()),selection='qualified finite per-molecule shortlist, direct training price, two-start integer alpha search; no holdout used'))
    d.save_json(OUT/'PR_POLICY.json',dict(source=d.source(),policies=policies,estimator='weighted Closed P3; fresh independent bits; no rejection/zero-fill'))
    print('PR_FROZEN',[(p['n'],p.get('training_RZ')) for p in policies],flush=True)


def independent_one(task):
    n,point=task;c=d.model(n);row=point['row'];ld=row['ld'];h=row['h'];r=point['signal']['r'];p=partition(n,ld);g=generator(n,ld,h,r);sampler=cp.DigitalSampler(g);bits=cp.BitStream(872519+point['signal']['m']);records=[sampler.sample(bits) for _ in range(r)]
    # Replace one event by a registered nonreturn word when available, so the
    # check cannot pass by observing only the high-probability collected root.
    gi=next((i for i,x in enumerate(g.groups) if x.mode=='fixed_first'),None)
    if gi is not None:
        j=g.groups[gi].first;k=sampler.conditional_draw(j,bits);child=sampler.conditional_draw(j,bits);records[0]=g.event(gi,(j,k),child)
    size=len(c['H']);qs={}
    def Q(label):
        if label not in qs:
            sym=p['symbols'][label];b=c['blocks'][sym['block']];qs[label]=sym['sign']*(b['B']*c['signs'][sym['support']])@b['B'].conj().T
        return qs[label]
    vec=c['state'].copy();weight=F(1)
    for rec in records:
        weight*=rec['weight']
        for label in reversed(tuple(rec['word'])):vec=Q(label)@vec
        theta,_=cp.angle(rec['ratio']);vec=math.cos(theta)*vec-1j*math.sin(theta)*(Q(rec['child'])@vec);vec*=1j**rec['phase_i_power']
    expected=complex(np.vdot(c['state'],vec));ids=[sum(1<<i for i in occ) for occ in c['occupations']];initial=np.zeros(1<<(c['modes']+1),complex);initial[ids]=c['state'];errors=[]
    from qiskit.quantum_info import Statevector
    for mode in ('term_parity','network'):
        core,_=event_circuit(n,ld,records,mode)
        for native in (False,True):
            circuit=d.compile_reference(core) if native else core
            for axis in ('X','Y'):
                q=QuantumCircuit(c['modes']+1);a=c['modes'];q.h(a);q.compose(circuit,inplace=True)
                if axis=='Y':q.sdg(a)
                q.h(a);v=Statevector(initial).evolve(q).data;read=float(np.vdot(v[:1<<a],v[:1<<a]).real-np.vdot(v[1<<a:],v[1<<a:]).real);errors.append(abs(read-(expected.real if axis=='X' else expected.imag)))
    assert max(errors)<1e-8
    # D/R ordering, D/D fused boundary, scalar relative phase and both readouts
    # are checked together for one and two complete steps, using a forced rare
    # word. No intermediate spin-sector projection occurs in native simulation.
    from .track_c_mixed_formula import compact
    groups=[[]]
    for item in d.schedule(row['case'],h):
        if item[0]=='R':groups.append([])
        else:groups[-1].append(item)
    first,last=groups;complete_errors=[];Dcache={}
    def apply_d(vector,sequence):
        for _,i,time_ in sequence:
            key=(i,time_)
            if key not in Dcache:Dcache[key]=d.expm(-1j*time_*block_data(n,i)[0])
            vector=Dcache[key]@vector
        return vector
    def apply_r(vector):
        for record in records:
            for label in reversed(tuple(record['word'])):vector=Q(label)@vector
            theta,_=cp.angle(record['ratio']);vector=math.cos(theta)*vector-1j*math.sin(theta)*(Q(record['child'])@vector);vector*=1j**record['phase_i_power']
        return vector
    source_R,_=event_circuit(n,ld,records,'network');native_R=d.compile_reference(source_R)
    for steps_small in (1,2):
        target=c['state'].copy()
        for _ in range(steps_small):target=apply_d(apply_r(apply_d(target,first)),last)
        target*=np.exp(-1j*steps_small*h*float(p['scalar']));expected_complete=complex(np.vdot(c['state'],target))
        qcore=QuantumCircuit(c['modes']+1)
        for j,sequence in enumerate([first]+[compact(last+first)]*(steps_small-1)+[last]):
            role='first' if j==0 else ('last' if j==steps_small else 'junction');mode=row['variants'][role]['selected'];src,_=d.circuit_piece(c,p['D'],sequence,mode);qcore.compose(d.compile_reference(src),inplace=True)
            if j<steps_small:qcore.compose(native_R,inplace=True)
        qcore.p(-steps_small*h*float(p['scalar']),c['modes'])
        for axis in ('X','Y'):
            q=QuantumCircuit(c['modes']+1);a=c['modes'];q.h(a);q.compose(qcore,inplace=True)
            if axis=='Y':q.sdg(a)
            q.h(a);vv=Statevector(initial).evolve(q).data;read=float(np.vdot(vv[:1<<a],vv[:1<<a]).real-np.vdot(vv[1<<a:],vv[1<<a:]).real)
            complete_errors.append(abs(read-(expected_complete.real if axis=='X' else expected_complete.imag)))
    assert max(complete_errors)<1e-8
    # Separately compute the actual corrected finite-P3 ensemble signal, using
    # a direct polynomial rather than an ideal-RTE or single-eigenmode proxy.
    kernel=np.eye(size,dtype=complex);lookup={b['index']:b for b in p['D']}
    for kind,i,t in d.schedule(row['case'],h):
        if kind=='R':
            A=-1j*p['tail']*(t/r);poly=np.eye(size)+A+(A@A)/2+(A@A@A)/6;a=np.linalg.matrix_power(poly,r)
        else:a=d.expm(-1j*t*d.sg.hermitian(lookup[i]['matrix']))
        kernel=a@kernel
    kernel*=np.exp(-1j*h*float(p['scalar']));power=np.linalg.matrix_power(kernel,point['signal']['steps']);z=complex(np.vdot(c['state'],power@c['state']))
    ideal=complex(*point['qualification']['signal']);delta=abs(z-ideal)
    assert delta<point['qualification']['finite_RTE_error_upper']+point['qualification']['numerical_disk']+1e-8
    result=dict(n=n,key=point['key'],source=d.source(),maximum_event_XY_difference=max(errors),complete_D_R_scalar_XY_difference=max(complete_errors),path_weight_exact=dict(numerator_hex=hex(weight.numerator),denominator_hex=hex(weight.denominator)),weight_encoding='exact rational, hexadecimal integers',finite_mean_signal=[z.real,z.imag],finite_mean_ideal_difference=delta,forced_nonreturn=gi is not None,validation='full statevector, no intermediate sector projection; signed ordered products; actual weighted P3 mean')
    d.save_json(OUT/f'H{n}/pr_independent/{ld}_{row["kind"]}_{h}_{r}_{point["signal"]["m"]}.json',result);return result


def finish():
    path=OUT/'PR_POLICY.json';digest=hashlib.sha256(path.read_bytes()).hexdigest();policy=json.loads(path.read_text());rs=samples('holdout');tasks={}
    checks=[];reused=0
    for pol in policy['policies']:
        if pol['status']=='FROZEN':
            for p in pol['selected']:
                row=p['row'];target=OUT/f'H{pol["n"]}/pr_independent/{row["ld"]}_{row["kind"]}_{row["h"]}_{p["signal"]["r"]}_{p["signal"]["m"]}.json'
                if target.exists():
                    check=json.loads(target.read_text());assert check['key']==p['key'];checks.append(check);reused+=1
                else:tasks[(pol['n'],tuple(p['key']))]=(pol['n'],p)
    if tasks:
        with ProcessPoolExecutor(max_workers=d.workers(len(tasks))) as pool:checks.extend(pool.map(independent_one,tasks.values()))
    d.save_json(OUT/'PR_INDEPENDENT_DATA_REUSE.json',dict(source=d.source(),policy_sha256=digest,reused=reused,new_checks=len(tasks),reason='reuse unchanged successful checks; exact weight serialization uses hex to avoid Python decimal digit limit; scientific conditions, policy and holdout unchanged'))
    lookup={(x['n'],tuple(x['key'])):x for x in checks}
    for pol in policy['policies']:
        if pol['status']!='FROZEN':continue
        total=np.zeros(4);det=np.zeros(4);coeff={};phases=[]
        for p,ns in zip(pol['selected'],np.array(pol['trials'])):
            point=priced(p,rs);total+=(point['mean']*ns[:,None]).sum(axis=0);det+=(np.array(point['D'])*ns.sum()+(np.array(point['overhead'])*ns[:,None]).sum(axis=0))
            cell=tuple(p['cell']);coeff[cell]=coeff.get(cell,0)+int(ns.sum())*p['signal']['steps'];z=complex(*lookup[(pol['n'],tuple(p['key']))]['finite_mean_signal']);phases.append(float(np.angle(z)))
        covariance=np.zeros((4,4))
        for cell,k in coeff.items():
            for sub in rs[cell].values():
                values=np.array([r['metrics'] for r in sub]);covariance+=k*k*sub[0]['probability']**2*np.cov(values,rowvar=False,ddof=1)/len(values)
        decoded=d.reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-decoded['final_phase_estimate']/.02;err=abs(energy-d.model(pol['n'])['energy']);assert err<1.5936e-4
        pol.update(status='QUALIFIED_INDEPENDENT_CHECK_COMPLETE',totals=total.tolist(),D_and_wrapper_totals=det.tolist(),R_totals=(total-det).tolist(),MC_covariance=covariance.tolist(),MC_SE=np.sqrt(np.maximum(0,np.diag(covariance))).tolist(),mean_decoder=dict(decoded=decoded,energy=energy,absolute_error_validation_only=err))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    d.save_json(OUT/'PR_FINAL.json',dict(source=d.source(),policy_sha256=digest,policies=policy['policies'],checks=checks,metrics=d.METRICS,energy_tolerance=1.5936e-4,failure_upper=.05,phase_budget=.4,energy_bound=.4/(.02*2**17),scope='same original-H task and ideal-native/fresh-bit assumptions as deterministic comparison; nominal stratified price MC SE is separate'))
    print('PR_FINAL',[(p['n'],p.get('totals')) for p in policy['policies']],flush=True)


if __name__=='__main__':
    import sys
    if (OUT/'STUDY_STOP.json').exists():raise RuntimeError('Study stopped; use an explicitly authorized fresh output')
    if sys.argv[1]=='probes':probes()
    elif sys.argv[1]=='bch':bch()
    elif sys.argv[1]=='isometries':build_isometries()
    elif sys.argv[1]=='qualify':shortlist()
    elif sys.argv[1]=='training':price_cells('training')
    elif sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='holdout':price_cells('holdout')
    elif sys.argv[1]=='finish':finish()
    else:raise ValueError(sys.argv[1])
