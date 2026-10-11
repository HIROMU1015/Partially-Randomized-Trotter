"""Post-hoc coefficient certificates, NOT a registered transfer experiment.
Uses the same published T3 input and retains U and delta. The external bound
and sufficient family are explicitly changed and must not replace frozen results.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import review_checks as rc

OUT=Path(__file__).resolve().parent

def eye(n):return [[F(i==j) for j in range(n)] for i in range(n)]
def add(A,B):return [[x+y for x,y in zip(a,b)] for a,b in zip(A,B)]
def scale(A,z):return [[z*x for x in a] for a in A]
def trace(A):return sum((A[i][i] for i in range(len(A))),F(0))
def outer(v):return [[x*y for y in v] for x in v]
def quad(A,v):return sum((v[i]*A[i][j]*v[j] for i in range(len(v)) for j in range(len(v))),F(0))
def dyadic(x,bits=40):return F(round(float(x)*2**bits),2**bits)

def spectral_lower(G,k):
    """Return exact witness sigma_k(G)>=k*tau-trace(Q).
    Acceptance uses Q>=0 and G-tau I+Q>=0, verified by rational LDL.
    """
    m=len(G)
    if k==0:return {'k':0,'lower':F(0),'trivial_empty_sum':True}
    if k==m:
        tau=max(G[i][i]+sum(abs(G[i][j]) for j in range(m) if j!=i) for i in range(m))+1
        Q=add(scale(eye(m),tau),scale(G,-1))
    else:
        w,V=np.linalg.eigh(np.array(G,float));tau=dyadic((w[k-1]+w[k])/2)
        Qn=(V*np.maximum(float(tau)-w,0))@V.T
        Q=[[dyadic((Qn[i,j]+Qn[j,i])/2) for j in range(m)] for i in range(m)]
        Q=add(Q,scale(eye(m),F(4*m,2**40)))
    R=add(add(G,scale(eye(m),-tau)),Q)
    p1,p2=rc.rat_psd(Q),rc.rat_psd(R)
    if not(p1['ok'] and p2['ok']):raise ArithmeticError('Rational spectral certificate failed')
    return {'k':k,'tau':tau,'Q':Q,'lower':k*tau-trace(Q),'Q_psd':p1,'residual_psd':p2}

def check_spectral(G,cert):
    k=cert['k']
    if k==0:assert cert['lower']==0;return
    m=len(G);Q=cert['Q'];tau=cert['tau']
    assert rc.rat_psd(Q)['ok']
    assert rc.rat_psd(add(add(G,scale(eye(m),-tau)),Q))['ok']
    assert cert['lower']==k*tau-trace(Q)

def old_family_separation(A,v,nu,U,c):
    D=c-U+rc.GOAL;m=len(A);R=add(A,scale(eye(m),-U/nu));M0=add(scale(eye(m),rc.GOAL*D/nu),scale(outer(v),-1))
    opt=rc.propose(A,v,nu,U,c);x=opt['x'];vs=[]
    for xx in [max(0,x-1e-5),min(1,x+1e-5)]:
        M=np.array(add(M0,scale(R,D*F(float(xx)))),float)
        _,V=np.linalg.eigh(M);z=[dyadic(t) for t in V[:,0]];vs.append(z)
    s1,s2=[quad(R,z) for z in vs]
    if s1==s2:raise ArithmeticError('No slope bracket')
    t=-s2/(s1-s2)
    if not 0<=t<=1:raise ArithmeticError('No positive convex separator')
    intercept=t*quad(M0,vs[0])+(1-t)*quad(M0,vs[1]);norm=t*sum(z*z for z in vs[0])+(1-t)*sum(z*z for z in vs[1])
    assert t*s1+(1-t)*s2==0 and intercept<0 and norm>0
    return {'u':vs[0],'v':vs[1],'weight':t,'slope_exact':F(0),'trace_W':norm,'intercept':intercept,'normalized_intercept':intercept/norm,'meaning':'nonzero PSD W has negative trace(W M(kappa)) for every kappa; original one-body LMI is infeasible at this common U,c,delta'}

def main():
    case=rc.recipe(7,3,F(2),7,F(1,4),F(1,40),True)
    act=case['active'];d=case['external'];nu=case['particles'];h=case['h'];A=[[h[i][j] for j in act] for i in act];v=[h[i][d] for i in act]
    ext=[sum((r['coefficient'] for r in case['density'] if {r['i'],r['j']}=={i,d}),F(0)) for i in act]
    Gext=add(A,[[ext[i] if i==j else F(0) for j in range(len(act))] for i in range(len(act))])
    extcert=spectral_lower(Gext,nu-1);check_spectral(Gext,extcert)
    U=rc.trial_U(case);c=h[d][d]+extcert['lower'];D=c-U+rc.GOAL
    assert c>U
    opt=rc.propose(A,v,nu,U,c,True);x=dyadic(opt['x']);kappa=D*x
    K=add(scale(A,kappa),scale(outer(v),-1));kcert=spectral_lower(K,nu);check_spectral(K,kcert)
    margin=kcert['lower']-kappa*U+rc.GOAL*D
    assert 0<=kappa<=D and margin>0
    normmargin=rc.GOAL*D-sum(z*z for z in v);assert normmargin<0
    sep=old_family_separation(A,v,nu,U,c)
    H,AP,C,B=rc.manybody(case)
    record={'classification':'post_hoc_review_certificate_not_frozen_transfer_result','case':case['id'],'input_case_sha256':rc.sha(case),'delta':rc.GOAL,
            'U':U,'c_frozen_rule':h[d][d]+(nu-1)*rc.common_lower(A),'c_occupancy_certificate':c,'D':D,'external_bound_witness':extcert,
            'kappa':kappa,'x':x,'active_spectral_witness':kcert,'affine_margin':margin,'norm_only_same_c_margin':normmargin,
            'original_one_body_family_infeasibility_witness':sep,'width_diagnostic':float(np.linalg.eigvalsh(AP)[0]-np.linalg.eigvalsh(H)[0]),
            'actual_C_min_diagnostic':float(np.linalg.eigvalsh(C)[0]),'certificate_uses_many_body_eigenvalues':False,
            'reconstructed_input':case}
    (OUT/'posthoc_particle_certificate.json').write_text(json.dumps(rc.enc(record),indent=2))
    print(json.dumps({k:float(record[k]) for k in ['U','c_frozen_rule','c_occupancy_certificate','D','kappa','affine_margin','norm_only_same_c_margin','width_diagnostic']},indent=2))
    print('old-family rational separator:',float(sep['normalized_intercept']))
    print('exact PSD/affine checks PASS; certificate file bytes', (OUT/'posthoc_particle_certificate.json').stat().st_size)

if __name__=='__main__':main()
