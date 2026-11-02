"""Independent certificate verifier: neutral JSON and stdlib Fraction only.

No backend imports. Status alone never certifies a claim. Row duals for minimum
with upper-only rows map by nu=-dual_A and u=-dual_H. Farkas maps by the SAME fixed minus sign, derived before execution from
SoPlex 7.0.0 _computeInfeasBox. No orientation search or reconstruction.
"""
from fractions import Fraction as F
import json
import sys
import time

def q(s):
    assert isinstance(s,str) and str(F(s))==s, 'noncanonical or nonrational encoding'
    return F(s)

def dot(a,b): return sum((x*y for x,y in zip(a,b)),F(0))
def qs(a): return [q(x) for x in a]

def verify(problem,result):
    started=time.perf_counter()
    n=len(problem['c']);ma=len(problem['A']);mh=len(problem['H'])
    echo=result['echo']
    echo_ok=all(echo[k]==problem[k] for k in ['c0','c','U','A','b','H','f']) and echo['lower']==['0']*n and echo['equal_lhs']==problem['f']
    assert echo_ok, 'rational I/O mismatch'
    c,U,b,f=map(qs,[problem['c'],problem['U'],problem['b'],problem['f']])
    A,H=[list(map(qs,problem[k])) for k in ['A','H']];c0=q(problem['c0'])
    out={'id':problem['id'],'rational_io':'PASS','status':result['status'],
         'primal':None,'dual':None,'farkas':None,'failures':[]}
    if result['status']=='ECHO_ONLY':
        out.update(PASS=True,verification_seconds=time.perf_counter()-started)
        return out
    if result['status']!=problem['expected_status']:
        out['failures'].append('unexpected status')
    if result['status']=='OPTIMAL':
        if not result.get('primal_available') or not result.get('dual_available'):
            out['failures'].append('exact primal or dual unavailable')
        else:
            x=qs(result['primal']);raw=qs(result['raw_row_dual'])
            assert len(x)==n and len(raw)==ma+mh
            slacks=[bb-dot(row,x) for row,bb in zip(A,b)]
            residuals=[dot(row,x)-ff for row,ff in zip(H,f)]
            violations=[i for i in range(n) if not 0<=x[i]<=U[i]]
            objective=c0+dot(c,x)
            primal_ok=not violations and all(s>=0 for s in slacks) and all(v==0 for v in residuals)
            primal_ok=primal_ok and objective==q(result['objective_with_exact_external_offset']) and objective-c0==q(result['backend_objective_without_offset'])
            out['primal']={'PASS':primal_ok,'inequality_slacks':list(map(str,slacks)),
                           'equality_residuals':list(map(str,residuals)),
                           'bound_violations':violations,'objective':str(objective),'x':result['primal']}
            nu=[-d for d in raw[:ma]];u=[-d for d in raw[ma:]]
            r=[c[j]+sum(nu[i]*A[i][j] for i in range(ma))+sum(u[i]*H[i][j] for i in range(mh)) for j in range(n)]
            correction=sum(min(F(0),r[j])*U[j] for j in range(n))
            L=c0-dot(nu,b)-dot(u,f)+correction
            rc_ok=result.get('reduced_cost_available') and r==qs(result['raw_reduced_cost'])
            dual_ok=all(v>=0 for v in nu) and L<=objective and rc_ok
            out['dual']={'PASS':dual_ok,'nu':list(map(str,nu)),'u':list(map(str,u)),
                         'stationarity_residual':list(map(str,r)),'box_correction':str(correction),
                         'offset':str(c0),'lower':str(L),'upper':str(objective),
                         'exact_gap':str(objective-L),'weak_duality':L<=objective,
                         'reduced_cost_sign_check':bool(rc_ok),'optimality_gap_zero':L==objective}
            if not primal_ok:out['failures'].append('primal certificate failed')
            if not dual_ok:out['failures'].append('dual certificate failed')
    elif result['status']=='INFEASIBLE':
        if not result.get('farkas_available'):
            out['failures'].append('exact Farkas unavailable')
        else:
            raw=qs(result['raw_row_farkas']);assert len(raw)==ma+mh
            # Fixed before execution from SoPlex _computeInfeasBox convention.
            nu=[-v for v in raw[:ma]];u=[-v for v in raw[ma:]]
            r=[sum(nu[i]*A[i][j] for i in range(ma))+sum(u[i]*H[i][j] for i in range(mh)) for j in range(n)]
            lhs=dot(nu,b)+dot(u,f);rhs=sum(min(F(0),r[j])*U[j] for j in range(n))
            ok=all(v>=0 for v in nu) and lhs<rhs
            out['farkas']={'PASS':ok,'orientation':-1,'nu':list(map(str,nu)),
                           'u':list(map(str,u)),'stationarity_residual':list(map(str,r)),
                           'weighted_rhs':str(lhs),'box_minimum':str(rhs),
                           'strict_separation_margin':str(rhs-lhs),'nonnegative_nu':all(v>=0 for v in nu)}
            if not ok:out['failures'].append('Farkas certificate failed')
    else:
        out['failures'].append('backend did not acquire a certificate')
    out['PASS']=not out['failures']
    out['verification_seconds']=time.perf_counter()-started
    return out

if __name__=='__main__':
    p=json.load(open(sys.argv[1]));r=json.load(open(sys.argv[2]))
    output=verify(p,r)
    print(json.dumps(output,indent=2))
    sys.exit(0 if output['PASS'] else 1)
