"""Independent witnesses for the frozen N3 sufficient conditions.
Reconstructs coefficients from the public recipe; uses neither repository science
module nor stored ground states. Different numerical proposal routine, exact acceptance.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import review_checks as rc
from particle_number_certificate import eye,add,scale,outer,dyadic
OUT=Path(__file__).resolve().parent

def solve(A,b):
    m=len(A);a=[list(row)+[v] for row,v in zip(A,b)]
    for k in range(m):
        if not a[k][k]:
            p=next(i for i in range(k+1,m) if a[i][k]);a[p],a[k]=a[k],a[p]
        q=a[k][k];a[k]=[x/q for x in a[k]]
        for i in range(m):
            if i!=k:
                q=a[i][k];a[i]=[x-q*y for x,y in zip(a[i],a[k])]
    return [a[i][-1] for i in range(m)]

def witness(A,v,nu,U,c,k,mu):
    D=c-U+rc.GOAL;M=add(add(scale(A,k),scale(eye(len(A)),-mu)),scale(outer(v),-1))
    proof=rc.rat_psd(M);aff=rc.GOAL*D-k*U+nu*mu
    return {'accepted':c>U and 0<=k<=D and aff>=0 and proof['ok'],'kappa':k,'mu':mu,'affine_margin':aff,'PSD':proof}

def main():
    rows=[]
    expected_norm=[1,1,0,0,1,1,0,0,0,0,0,0]
    expected_five=[1,1,0,1,0,1,0,0,0,0,0,0]
    expected_direct=[1,1,0,1,1,1,0,0,0,0,0,0]
    for m,nu,s,d,hd in [(4,1,F(1,2),0,F(3,4)),(5,2,F(1),2,F(1,2)),(7,3,F(2),7,F(1,4))]:
        for amp in (F(1,40),F(1,10)):
            for high in (False,True):
                case=rc.recipe(m,nu,s,d,hd,amp,high);act=case['active'];h=case['h']
                A=[[h[i][j] for j in act] for i in act];v=[h[i][d] for i in act];U=rc.trial_U(case);c=h[d][d]+(nu-1)*rc.common_lower(A);D=c-U+rc.GOAL
                norm=c>U and rc.GOAL*D>=sum((x*x for x in v),F(0))
                r=min(A[i][i]-sum((abs(A[i][j]) for j in range(m) if i!=j),F(0)) for i in range(m));span=max(A[i][i] for i in range(m))-min(A[i][i] for i in range(m))
                old=[]
                for rho in (F(0),F(1,20),F(1,10),F(3,20),F(1,5)):
                    lam=r+rho*span;M=rc.shift(A,lam);pr=rc.rat_psd(M)
                    if pr['ok'] and all(x>0 for x in pr['pivots']):
                        x=solve(M,v);k=sum((a*b for a,b in zip(v,x)),F(0));old.append(witness(A,v,nu,U,c,k,k*lam))
                five=any(w['accepted'] for w in old)
                prop=rc.propose(A,v,nu,U,c);direct=False;new=None
                if prop['precondition']:
                    k=dyadic(prop['x'])*D;mu=(k*U-rc.GOAL*D)/nu;new=witness(A,v,nu,U,c,k,mu);direct=new['accepted']
                rows.append({'id':case['id'],'case_sha256':rc.sha(case),'U':U,'c':c,'norm':norm,'five':five,'direct':direct,'direct_witness':new,'five_witnesses':old})
    assert [int(x['norm']) for x in rows]==expected_norm
    assert [int(x['five']) for x in rows]==expected_five
    assert [int(x['direct']) for x in rows]==expected_direct
    direct={x['id'] for x in rows if x['direct']};union={x['id'] for x in rows if x['norm'] or x['five']}
    assert direct==union
    n1=[('planted',16448,2494,1664,41021312,27369472),('dense',16933,2614,1696,44262862,28718368),('separated',13631,5222,3392,71181082,46236352)]
    for _,n,rz,cx,grz,gcx in n1:assert n*rz==grz and n*cx==gcx
    assert 2*(9*9*2*4)+(4*4*2*4)==1424
    output={'rows':rows,'counts':{k:sum(x[k] for x in rows) for k in ('norm','five','direct')},'union_equals_direct':True,'N1_transcribed_selected_products_pass':True,'N1_finite_row_count':1424,'qualification':'No all-row N1 argmin audit, native recompilation, or 72-ledger witness replay is claimed.'}
    (OUT/'original_reproduction.json').write_text(json.dumps(rc.enc(output),indent=2))
    print(output['counts'],'union equals direct',direct==union,'N1 selected arithmetic PASS')

if __name__=='__main__':main()
