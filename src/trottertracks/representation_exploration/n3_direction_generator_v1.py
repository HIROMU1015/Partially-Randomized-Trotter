"""Coefficient-only N3 proposals and exact rational sufficient certificates.

No many-body diagnostic is imported or called by generate_case. The runner
writes and hashes all certificate records before calling diagnose_case.
"""
from __future__ import annotations
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools as it
import json
import math
import resource
import time
import numpy as np

INPUT_SHA = '9e99bda0d98ede7a5eea97002f36ab00378655884f35b0b0aec9af7692078d12'
DELTA = F(1,250)
METHODS = ('legacy_scalar','common_norm_only','strengthened_scalar','five_point_common','direct_one_variable','five_point_legacy_c')


def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x): return hashlib.sha256(canon(x)).hexdigest()
def enc(x): x=F(x); return [x.numerator,x.denominator]
def dec(x):
    if not isinstance(x,list) or len(x)!=2 or any(type(y)!=int for y in x) or x[1]<=0: raise ValueError('rational schema')
    return F(*x)


class Arithmetic:
    def __init__(self): self.max_bits=0
    def check(self,x):
        x=F(x);bits=max(x.numerator.bit_length(),x.denominator.bit_length());self.max_bits=max(self.max_bits,bits)
        if bits>4096: raise ArithmeticError('UNRESOLVED_ARITHMETIC_CAP')
        return x
    def summation(self,xs):
        s=F(0)
        for x in xs:s=self.check(s+self.check(x))
        return s


def psd(matrix,ar,strict=False):
    n=len(matrix)
    if any(len(row)!=n for row in matrix) or any(matrix[i][j]!=matrix[j][i] for i in range(n) for j in range(n)):raise ValueError('symmetric PSD matrix required')
    a=[[ar.check(x) for x in row] for row in matrix];L=[[F(i==j) for j in range(n)] for i in range(n)];d=[];status='PSD'
    for k in range(n):
        p=ar.check(a[k][k]);d.append(p)
        if p<0:status='NEGATIVE_PIVOT';break
        if p==0:
            if strict:status='ZERO_PIVOT_NOT_SPD';break
            if any(a[k][j]!=0 for j in range(k+1,n)):status='ZERO_PIVOT_NONZERO_ROW';break
            continue
        for i in range(k+1,n):L[i][k]=ar.check(a[i][k]/p)
        for i in range(k+1,n):
            for j in range(i,n):
                a[i][j]=ar.check(a[i][j]-ar.check(L[i][k]*ar.check(p*L[j][k])));a[j][i]=a[i][j]
    return {'accepted':status=='PSD','status':status,'L':[[enc(x) for x in row] for row in L],'pivots':[enc(x) for x in d],
            'strict':strict,'min_pivot':enc(min(d)) if d else None}


def inverse_quadratic(proof,v,ar):
    L=[[dec(x) for x in row] for row in proof['L']];d=list(map(dec,proof['pivots']));y=[]
    for i in range(len(v)):y.append(ar.check(v[i]-ar.summation(ar.check(L[i][j]*y[j]) for j in range(i))))
    return ar.summation(ar.check(ar.check(y[i]*y[i])/d[i]) for i in range(len(v)))


def decode_case(case):
    allowed={'id','base','modes','particles','active','external','scale','amplitude','direction','h','density'}
    if set(case)!=allowed:raise ValueError('unknown/oracle case field')
    n=case['modes'];nu=case['particles'];active=case['active'];ext=case['external']
    if type(n)!=int or n>8 or type(nu)!=int or not 1<=nu<=len(active)<=7:raise ValueError('mode/particle scope')
    if any(type(i)!=int for i in active+[ext]) or len(set(active+[ext]))!=n or set(active+[ext])!=set(range(n)):raise ValueError('one external mode')
    h=[[dec(x) for x in row] for row in case['h']]
    if len(h)!=n or any(len(row)!=n for row in h) or any(h[i][j]!=h[j][i] for i in range(n) for j in range(n)):raise ValueError('real symmetric one-body required')
    density={}
    for row in case['density']:
        if set(row)!={'i','j','coefficient'}:raise ValueError('density schema')
        i,j=row['i'],row['j'];v=dec(row['coefficient'])
        if any(type(k)!=int or not 0<=k<n for k in (i,j)) or i==j or v<0:raise ValueError('nonnegative unordered density required')
        key=tuple(sorted((i,j)))
        if key in density:raise ValueError('duplicate unordered density')
        density[key]=v
    return h,density,active,ext,nu


def trial_bound(h,density,active,nu,ar):
    dets=list(it.combinations(sorted(active),nu));trials=[];diags=[]
    if len(dets)>35:raise ValueError('determinant cap')
    for occ in dets:
        value=ar.summation([h[i][i] for i in occ]+[v for (i,j),v in density.items() if i in occ and j in occ]);diags.append(value)
        trials.append({'kind':'determinant','occupations':[list(occ)],'relative_sign':1,'normalization':'1','value':enc(value)})
    for a,b in it.combinations(range(len(dets)),2):
        pset=set(dets[a])-set(dets[b]);qset=set(dets[b])-set(dets[a])
        if len(pset)==len(qset)==1:
            p,q=next(iter(pset)),next(iter(qset));hop=h[p][q]*(-1)**(dets[b].index(q)+dets[a].index(p))
            if hop:
                value=ar.check((diags[a]+diags[b])/2-abs(hop))
                trials.append({'kind':'connected_equal_pair','occupations':[list(dets[a]),list(dets[b])],
                    'relative_sign':-1 if hop>0 else 1,'normalization':'1/sqrt(2)','fermionic_matrix_element':enc(hop),'value':enc(value)})
    if len(trials)>128:raise ValueError('trial cap')
    idx=min(range(len(trials)),key=lambda i:(dec(trials[i]['value']),i))
    return dec(trials[idx]['value']),trials,idx


def shift(h,lam):return [[h[i][j]-(lam if i==j else 0) for j in range(len(h))] for i in range(len(h))]

def coefficient_bounds(h,density,active,ext,nu,ar):
    start=time.monotonic();U,trials,selected=trial_bound(h,density,active,nu,ar);A=[[h[i][j] for j in active] for i in active];v=[h[i][ext] for i in active]
    m=len(A);r=min(A[i][i]-ar.summation(abs(A[i][j]) for j in range(m) if j!=i) for i in range(m));s=max(A[i][i] for i in range(m))-min(A[i][i] for i in range(m))
    old=[]
    for rho in (F(0),F(1,20),F(1,10),F(3,20),F(1,5)):
        lam=ar.check(r+rho*s);proof=psd(shift(A,lam),ar,strict=True);row={'lambda':enc(lam),'rho':enc(rho),'proof':proof}
        if proof['accepted']:row['kappa']=enc(inverse_quadratic(proof,v,ar))
        old.append(row)
    oldell=max([r]+[dec(x['lambda']) for x in old if x['proof']['accepted']]);refined=[];proposal=None;proposal_error=None
    try:
        proposal=float(np.linalg.eigvalsh(np.asarray(A,float))[0])
        if not math.isfinite(proposal):raise ValueError('nonfinite one-body proposal')
        E=max([F(1)]+[abs(x) for row in A for x in row]);step=E/(1<<24);floor=(F(proposal)/step).__floor__()*step
        for offset in (1,2,4,8,16,32):
            lam=ar.check(floor-offset*step);proof=psd(shift(A,lam),ar);refined.append({'lambda':enc(lam),'offset':offset,'proof':proof})
    except (ValueError,np.linalg.LinAlgError) as e:proposal_error=str(e)
    ell=max([oldell]+[dec(x['lambda']) for x in refined if x['proof']['accepted']]);c=ar.check(h[ext][ext]+(nu-1)*ell)
    return {'U':enc(U),'c':enc(c),'ell':enc(ell),'old_ell':enc(oldell),'legacy_c_for_five_point':enc(h[ext][ext]+(nu-1)*oldell),
            'r':enc(r),'s':enc(s),'trial_states':trials,'selected_trial':selected,'old_lambda':old,'refined_lower':refined,
            'one_body_proposal':proposal,'proposal_failure':proposal_error,'one_body_proposal_calls':1,
            'shared_generation_wall_seconds':time.monotonic()-start,'cost_accounting':'one shared computation per case, same U/c for all common methods'},A,v,U,c


def original_witness(A,v,nu,U,c,k,mu,ar):
    D=c-U+DELTA
    M=[[ar.check(k*A[i][j]-(mu if i==j else 0)-v[i]*v[j]) for j in range(len(v))] for i in range(len(v))]
    proof=psd(M,ar);aff=ar.check(DELTA*D-(k*U-nu*mu));upper=ar.check(D-k)
    valid=c>U and 0<=k<=D and aff>=0 and proof['accepted']
    return {'kappa':enc(k),'mu':enc(mu),'D':enc(D),'affine_margin':enc(aff),'lower_box_margin':enc(k),'upper_box_margin':enc(upper),'PSD':proof,'accepted':valid}


def five_point(bounds,A,v,nu,U,c,ar):
    attempts=[];chosen=None
    for x in bounds['old_lambda']:
        row={'lambda':x['lambda'],'spd':x['proof']['accepted']}
        if row['spd']:
            k=dec(x['kappa']);row['witness']=original_witness(A,v,nu,U,c,k,k*dec(x['lambda']),ar)
            if row['witness']['accepted'] and chosen is None:chosen=row['witness']
        attempts.append(row)
    return {'accepted':chosen is not None,'witness':chosen,'attempts':attempts,'status':'CERTIFIED' if chosen else ('BOUND_PRECONDITION_FAILED' if c<=U else 'UNRESOLVED_NOT_CERTIFIED')}


def numeric_search(A,v,nu,U,c):
    D=c-U+DELTA;An=np.asarray(A,float)-float(U/nu)*np.eye(len(A));vv=np.outer(np.asarray(v,float),np.asarray(v,float));constant=float(DELTA*D/nu)*np.eye(len(A))-vv;calls=0;best=None
    def objective(x):
        nonlocal calls,best
        if calls>=128:raise ValueError('search eigenvalue cap')
        calls+=1;score=float(np.linalg.eigvalsh(float(D)*x*An+constant)[0])
        if not math.isfinite(score):raise ValueError('nonfinite search objective')
        if best is None or (score,-x)>(best[1],-best[0]):best=(x,score)
        return score
    objective(0.);objective(1.);a,b=0.,1.;phi=(math.sqrt(5)-1)/2;x1=b-phi*(b-a);x2=a+phi*(b-a);f1,f2=objective(x1),objective(x2)
    while b-a>2**-40 and calls<128:
        if f1<f2:a=x1;x1=x2;f1=f2;x2=a+phi*(b-a);f2=objective(x2)
        else:b=x2;x2=x1;f2=f1;x1=b-phi*(b-a);f1=objective(x1)
    return {'x':best[0],'score':best[1],'bracket':[a,b],'eigenvalue_calls':calls,'role':'numerical proposal only, no completeness or certification'}


def direct(bounds,A,v,nu,U,c,ar):
    if c<=U:return {'accepted':False,'status':'BOUND_PRECONDITION_FAILED','attempts':[],'search':None,'witness':None}
    try:search=numeric_search(A,v,nu,U,c)
    except (ValueError,np.linalg.LinAlgError) as e:return {'accepted':False,'status':'UNRESOLVED_NOT_CERTIFIED','reason':str(e),'attempts':[],'search':None,'witness':None}
    x=F(search['x']);candidates={F(0),F(1)}
    for bits in (16,24,32,40):
        candidates.add(F((x*(1<<bits)).__floor__(),1<<bits));candidates.add(F((x*(1<<bits)).__ceil__(),1<<bits))
    D=c-U+DELTA;attempts=[];chosen=None
    for xr in sorted(candidates):
        k=ar.check(xr*D);mu=ar.check((k*U-DELTA*D)/nu);row=original_witness(A,v,nu,U,c,k,mu,ar);row['x']=enc(xr);attempts.append(row)
        if row['accepted'] and chosen is None:chosen=row
    if len(attempts)>10:raise ValueError('rational candidate cap')
    return {'accepted':chosen is not None,'status':'CERTIFIED' if chosen else 'UNRESOLVED_NOT_CERTIFIED','search':search,'attempts':attempts,'witness':chosen,'old_grid_fallback':False}


def generate_case(case):
    start=time.monotonic();cpu=time.process_time();ar=Arithmetic();h,density,active,ext,nu=decode_case(case);bounds,A,v,U,c=coefficient_bounds(h,density,active,ext,nu,ar);D=c-U+DELTA
    beta2=ar.summation(x*x for x in v);beta1=ar.summation(abs(x) for x in v);methods=[]
    legacy_occ=sorted(active)[:nu];lu=ar.summation([h[i][i] for i in legacy_occ]+[val for (i,j),val in density.items() if i in legacy_occ and j in legacy_occ]);vnorm=ar.summation([abs(h[i][j]) for i in range(len(h)) for j in range(i+1,len(h))]+list(density.values()));lc=h[ext][ext]+sum(sorted(h[i][i] for i in active)[:nu-1],F(0))-vnorm
    for name in METHODS:
        begin=time.monotonic()
        if name=='legacy_scalar':
            margin=DELTA*(lc-lu+DELTA)-beta1**2;result={'accepted':lc>lu and margin>=0,'legacy_U':enc(lu),'legacy_c':enc(lc),'legacy_vnorm':enc(vnorm),'legacy_trial_occupation':legacy_occ,'margin':enc(margin),'status':'CERTIFIED' if lc>lu and margin>=0 else 'UNRESOLVED_NOT_CERTIFIED'}
        elif name in ('common_norm_only','strengthened_scalar'):
            beta=beta2 if name=='common_norm_only' else beta1**2;margin=ar.check(DELTA*D-beta);ok=c>U and margin>=0;result={'accepted':ok,'margin':enc(margin),'status':'CERTIFIED' if ok else ('BOUND_PRECONDITION_FAILED' if c<=U else 'UNRESOLVED_NOT_CERTIFIED')}
        elif name in ('five_point_common','five_point_legacy_c'):result=five_point(bounds,A,v,nu,U,c if name=='five_point_common' else dec(bounds['legacy_c_for_five_point']),ar)
        else:result=direct(bounds,A,v,nu,U,c,ar)
        result.update({'method':name,'common_U':enc(U),'common_c':enc(c),'beta2':enc(beta2),'beta1':enc(beta1),'selected_modes':len(active) if result['accepted'] else len(h),'used_full_space_fallback':not result['accepted'],'method_wall_seconds_excluding_shared_bounds':time.monotonic()-begin});methods.append(result)
    return {'input_id':case['id'],'base':case['base'],'coefficient_input_sha256':digest(case),'bounds':bounds,'methods':methods,
            'resources':{'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'rational_max_bits_observed':ar.max_bits,'matrix_dimension':len(A),'trial_states':len(bounds['trial_states'])}}


@lru_cache(None)
def sector_operators(n,nu,active):
    occupations=list(it.combinations(range(n),nu));states=[sum(1<<i for i in occ) for occ in occupations];index={s:k for k,s in enumerate(states)};ops=np.zeros((n,n,len(states),len(states)))
    for col,s in enumerate(states):
        for j in range(n):
            if not (s>>j)&1:continue
            removed=s^(1<<j);sg=(-1)**((s&((1<<j)-1)).bit_count())
            for i in range(n):
                if (removed>>i)&1:continue
                target=removed|(1<<i);ops[i,j,index[target],col]=sg*(-1)**((removed&((1<<i)-1)).bit_count())
    aids=[k for k,s in enumerate(states) if all(i in active for i in range(n) if s>>i&1)]
    return states,ops,aids


def diagnose_case(case,record):
    h,density,active,ext,nu=decode_case(case);states,ops,aids=sector_operators(len(h),nu,tuple(active));H=np.tensordot(np.asarray(h,float),ops,axes=2)
    for (i,j),value in density.items():H[np.diag_indices(len(states))]+=float(value)*np.array([((s>>i)&1)*((s>>j)&1) for s in states])
    if len(states)>56:raise ValueError('diagnostic sector dimension cap')
    E=float(np.linalg.eigvalsh(H)[0]);a=float(np.linalg.eigvalsh(H[np.ix_(aids,aids)])[0]);width=a-E
    if a>float(dec(record['bounds']['U']))+1e-10:raise AssertionError('SOUNDNESS_STOP: U')
    if any(x['accepted'] for x in record['methods']) and width>float(DELTA)+1e-10:raise AssertionError('SOUNDNESS_STOP: certified width')
    return {'input_id':case['id'],'full_ground':E,'active_ground':a,'width':width,'target_met_numerically':width<=float(DELTA)+1e-10,'eigenvalue_calls':2,'sector_dimension':len(states),'active_sector_dimension':len(aids),'numeric_tolerance':1e-10,'role':'post-ledger diagnostics only, not proof'}
