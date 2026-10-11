"""Independent post-hoc review calculations on the published 12-input recipe.
Not the repository runner. No new molecular input, native compile, or sampling.
NumPy spectra are diagnostics/proposals only; exact rational checks are separate.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json, hashlib, math
import numpy as np
from scipy.optimize import minimize_scalar

OUT=Path(__file__).resolve().parent
GOAL=F(1,250)

def enc(x):
    if isinstance(x,F):return [x.numerator,x.denominator]
    if isinstance(x,dict):return {k:enc(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [enc(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x

def sha(obj):return hashlib.sha256(json.dumps(enc(obj),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def recipe(m,nu,scale,d,hd_ratio,amp,high):
    active=[i for i in range(m+1) if i!=d];n=m+1
    h=[[F(0) for _ in range(n)] for _ in range(n)]
    for k,i in enumerate(active):h[i][i]=scale*(F(k,m-1)-F(2,5))
    for k in range(m-1):
        i,j=active[k:k+2];h[i][j]=h[j][i]=scale*F((-1)**k,25)
    h[active[0]][active[-1]]=h[active[-1]][active[0]]=scale/50
    h[d][d]=scale*hd_ratio
    dirs=([F(0)]*(m-2)+[F(3,5),F(4,5)]) if high else ([F(4,5),F(3,5)]+[F(0)]*(m-2))
    for i,z in zip(active,dirs):h[i][d]=h[d][i]=amp*z
    pairs=[(active[0],active[1],3*scale/10),(active[-2],active[-1],scale/10),(active[0],d,scale/20)]
    den=[{'i':i,'j':j,'coefficient':v} for i,j,v in pairs]
    base={4:'T1',5:'T2',7:'T3'}[m]
    direction='high_coordinates' if high else 'low_coordinates'
    return dict(id=f'{base}_amp_{amp.numerator}_{amp.denominator}_{direction}',base=base,modes=n,particles=nu,active=active,external=d,scale=scale,amplitude=amp,direction=direction,h=h,density=den)

def fermion_hop(state,i,j):
    if not (state>>j&1):return None
    p=(-1)**((state&((1<<j)-1)).bit_count());s=state^(1<<j)
    if s>>i&1:return None
    p*=(-1)**((s&((1<<i)-1)).bit_count())
    return s|(1<<i),p

def manybody(case):
    n=case['modes'];nu=case['particles'];h=case['h']
    states=[sum(1<<i for i in occ) for occ in combinations(range(n),nu)]
    idx={s:i for i,s in enumerate(states)};H=np.zeros((len(states),len(states)))
    for j,s in enumerate(states):
        for a in range(n):
            for b in range(n):
                z=fermion_hop(s,a,b)
                if z is not None:H[idx[z[0]],j]+=float(h[a][b])*z[1]
        for den in case['density']:H[j,j]+=float(den['coefficient'])*(s>>den['i']&1)*(s>>den['j']&1)
    p=[i for i,s in enumerate(states) if not (s>>case['external']&1)]
    q=[i for i,s in enumerate(states) if s>>case['external']&1]
    return H,H[np.ix_(p,p)],H[np.ix_(q,q)],H[np.ix_(p,q)]

def trial_U(case):
    active=case['active'];nu=case['particles'];h=case['h'];den=case['density']
    occs=list(combinations(active,nu));ss=[sum(1<<i for i in occ) for occ in occs]
    diag=[sum((h[i][i] for i in occ),F(0))+sum((r['coefficient'] for r in den if r['i'] in occ and r['j'] in occ),F(0)) for occ in occs]
    trials=list(diag)
    for k,l in combinations(range(len(ss)),2):
        diff1=[i for i in active if ss[k]>>i&1 and not(ss[l]>>i&1)]
        diff2=[i for i in active if ss[l]>>i&1 and not(ss[k]>>i&1)]
        if len(diff1)==len(diff2)==1:
            v=h[diff1[0]][diff2[0]]
            if v:trials.append((diag[k]+diag[l])/2-abs(v))
    return min(trials)

def rat_psd(M):
    a=[[F(x) for x in row] for row in M];n=len(a);pivs=[];bits=0
    assert all(a[i][j]==a[j][i] for i in range(n) for j in range(n))
    for k in range(n):
        p=a[k][k];pivs.append(p)
        if p<0:return {'ok':False,'pivots':pivs,'max_bits':bits}
        if not p:
            if any(a[k][j] for j in range(k+1,n)):return {'ok':False,'pivots':pivs,'max_bits':bits}
            continue
        for i in range(k+1,n):
            for j in range(i,n):
                a[i][j]-=a[i][k]*a[k][j]/p;a[j][i]=a[i][j]
                bits=max(bits,a[i][j].numerator.bit_length(),a[i][j].denominator.bit_length())
    return {'ok':True,'pivots':pivs,'max_bits':bits}

def shift(M,z):return [[M[i][j]-(z if i==j else 0) for j in range(len(M))] for i in range(len(M))]

def common_lower(A):
    n=len(A);r=min(A[i][i]-sum((abs(A[i][j]) for j in range(n) if i!=j),F(0)) for i in range(n))
    span=max(A[i][i] for i in range(n))-min(A[i][i] for i in range(n))
    vals=[r]
    for rho in [F(0),F(1,20),F(1,10),F(3,20),F(1,5)]:
        lam=r+rho*span;pr=rat_psd(shift(A,lam))
        if pr['ok'] and all(x>0 for x in pr['pivots']):vals.append(lam)
    proposal=F(float(np.linalg.eigvalsh(np.array(A,float))[0]));step=max([F(1)]+[abs(x) for row in A for x in row])/2**24
    floor=(proposal//step)*step
    for offset in (1,2,4,8,16,32):
        lam=floor-offset*step
        if rat_psd(shift(A,lam))['ok']:vals.append(lam)
    ell=max(vals);return ell

def propose(A,v,nu,U,c,particle_sum=False):
    D=c-U+GOAL
    if c<=U:return {'accepted_numeric':False,'precondition':False}
    arr=np.array(A,float);vv=np.outer(np.array(v,float),np.array(v,float));d=float(D)
    def margin(x):
        ev=np.linalg.eigvalsh(d*x*arr-vv)
        low=ev[:nu].sum() if particle_sum else nu*ev[0]
        return low-d*x*float(U)+float(GOAL*D)
    sol=minimize_scalar(lambda x:-margin(x),bounds=(0,1),method='bounded',options={'xatol':1e-13})
    x=max([0.,1.,float(sol.x)],key=margin)
    return {'accepted_numeric':margin(x)>=0,'precondition':True,'x':x,'margin':margin(x),'kappa':d*x}

REFERENCE_WIDTHS=[.0005674814315,3.971960119e-7,.009338985077,6.59557878e-6,.0005065015176,1.080399215e-5,.008366289461,.0001827274452,.001053313997,7.043636624e-7,.01685105753,1.147478621e-5]

def initial():
    cases=[recipe(m,nu,s,d,r,a,high) for m,nu,s,d,r in [(4,1,F(1,2),0,F(3,4)),(5,2,F(1),2,F(1,2)),(7,3,F(2),7,F(1,4))] for a in [F(1,40),F(1,10)] for high in [False,True]]
    assert sha(cases[0])=='58d1c06a8b389819b6277128cce6e9dad156ebd596ea92505b3105f222fe7da6'
    results=[]
    for case,reference in zip(cases,REFERENCE_WIDTHS):
        act=case['active'];d=case['external'];nu=case['particles'];h=case['h'];A=[[h[i][j] for j in act] for i in act];v=[h[i][d] for i in act]
        U=trial_U(case);ell=common_lower(A);c=h[d][d]+(nu-1)*ell
        H,AP,C,B=manybody(case);width=float(np.linalg.eigvalsh(AP)[0]-np.linalg.eigvalsh(H)[0])
        assert abs(width-reference)<5e-12,(case['id'],width,reference)
        extdiag=[sum((r['coefficient'] for r in case['density'] if {r['i'],r['j']}=={i,d}),F(0)) for i in act]
        Ad=[[A[i][j]+(extdiag[i] if i==j else 0) for j in range(len(act))] for i in range(len(act))]
        better=float(h[d][d])+np.linalg.eigvalsh(np.array(Ad,float))[:nu-1].sum()
        pn=propose(A,v,nu,U,F(float(better)),True)
        rec={'id':case['id'],'input_case_sha256':sha(case),'nu':nu,'U':U,'ell':ell,'c_frozen_rule':c,'width_diagnostic':width,
             'norm_frozen':c>U and GOAL*(c-U+GOAL)>=sum((x*x for x in v),F(0)),
             'old_direct_diagnostic':propose(A,v,nu,U,c),
             'particle_sum_same_c_diagnostic':propose(A,v,nu,U,c,True),
             'c_occupancy_aware_numeric':better,'old_direct_better_c_diagnostic':propose(A,v,nu,U,F(float(better))),
             'particle_sum_better_c_diagnostic':pn,
             'actual_C_min_diagnostic':float(np.linalg.eigvalsh(C)[0]),
             'beta2':sum((x*x for x in v),F(0))}
        results.append(rec)
    (OUT/'reconstructed_cases.json').write_text(json.dumps(enc(cases),indent=2))
    (OUT/'initial_diagnostics.json').write_text(json.dumps(enc(results),indent=2))
    for r in results:print(r['id'], 'U,c=',float(r['U']),round(float(r['c_frozen_rule']),7),'cnew=',round(r['c_occupancy_aware_numeric'],7),'old/newSame/newC=',r['old_direct_diagnostic']['accepted_numeric'],r['particle_sum_same_c_diagnostic']['accepted_numeric'],r['particle_sum_better_c_diagnostic']['accepted_numeric'])
    return cases,results

if __name__=='__main__':initial()
