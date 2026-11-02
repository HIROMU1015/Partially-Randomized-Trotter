"""Small deterministic checks of new-design formulas. No repository imports or quantum benchmarks."""
from __future__ import annotations
from fractions import Fraction as F
import itertools, json, math
from pathlib import Path
import numpy as np

def sector_norm(x, nu):
    x = sorted(float(v) for v in x)
    if nu == 0: return 0.0
    return max(abs(sum(x[:nu])), abs(sum(x[-nu:])))

def givens(n,i,j,theta):
    q=np.eye(n); c,s=math.cos(theta),math.sin(theta)
    q[i,i]=q[j,j]=c; q[i,j]=-s; q[j,i]=s
    return q

def fock(u):
    n=len(u); occ=[[i for i in range(n) if state>>i&1] for state in range(1<<n)]
    ans=np.zeros((1<<n,1<<n),dtype=complex)
    for i,a in enumerate(occ):
        for j,b in enumerate(occ):
            if len(a)==len(b): ans[i,j]=np.linalg.det(u[np.ix_(a,b)]) if a else 1
    return ans

eta=np.array([1,1.002,-.4,-.397]); et=np.array([1.001,1.001,-.3985,-.3985]); n=4; nu=2
W=givens(n,0,1,.6)@givens(n,2,3,-.4)
V=givens(n,0,2,.35)@givens(n,1,3,-.22)
U=V@W
occupation=np.array([[int(s>>i&1) for i in range(n)] for s in range(1<<n)])
D=np.diag((occupation@eta)**2); Dt=np.diag((occupation@et)**2)
HU=fock(U)@D@fock(U).conj().T
HUt=fock(U)@Dt@fock(U).conj().T
HVt=fock(V)@Dt@fock(V).conj().T
idx=np.flatnonzero(occupation.sum(axis=1)==nu)
actual=float(np.linalg.norm((HU-HUt)[np.ix_(idx,idx)],2))
d=sector_norm(eta-et,nu); bound=d*(sector_norm(eta,nu)+sector_norm(et,nu))
assert actual <= bound+1e-12
assert np.linalg.norm(HUt-HVt,2)<1e-12
G=givens(2,0,1,math.pi/4); one=np.diag([1,1.002])
gate_error=float(np.linalg.norm(one-G@one@G.T,2))
gate_formula=abs(1-1.002)*abs(math.sin(math.pi/4))
assert abs(gate_error-gate_formula)<1e-12
out={"scope":"deterministic formula checks only; not molecular validation, new science batch, compiler benchmark or novelty proof", "N1":{
"modes":n,"particles":nu,"eta":eta.tolist(),"clustered_eta":et.tolist(),
"sector_square_error":actual,"analytic_sector_error_bound":bound,
"same_cluster_frame_removal_residual":float(np.linalg.norm(HUt-HVt,2)),
"givens_error":gate_error,"givens_formula":gate_formula}}

S=[[1,0],[1,1],[0,1],[-1,0]]; K=[[F(3,10),F(-1,10)],[F(-1,10),F(1,5)]]
J=[[sum(F(S[i][a])*K[a][b]*F(S[j][b]) for a in range(2) for b in range(2)) for j in range(4)] for i in range(4)]
errs=[]; perrs=[]
for bits in itertools.product((0,1),repeat=4):
    q=[sum(S[i][a]*bits[i] for i in range(4)) for a in range(2)]
    orig=sum(bits[i]*J[i][j]*bits[j] for i in range(4) for j in range(4))
    charge=sum(q[a]*K[a][b]*q[b] for a in range(2) for b in range(2))
    errs.append(abs(orig-charge)); perrs.append(F(2,1000)*bits[0]*bits[3])
assert max(errs)==0 and max(perrs)==F(1,500)
out['N2']={'occupations_checked':16,'S':S,'K':[[str(v) for v in row] for row in K],
'exact_max_identity_error':str(max(errs)), 'symmetric_E03_E30':'1/1000',
'perturbation_exact_norm':str(max(perrs)), 'coefficient_error_bound':'1/500'}

examples=[]
for c,beta,Ubound in [(10.,.1,0.),(10.,.1,1.),(.1,.1,0.)]:
    H=np.array([[0,beta],[beta,c]])
    actual=-float(np.linalg.eigvalsh(H)[0]); g=c-Ubound
    delta=2*beta**2/(math.sqrt(g*g+4*beta**2)+g)
    assert actual<=delta+1e-13
    examples.append({'A_ground':0.,'C_lower':c,'coupling_upper':beta,'A_upper':Ubound,
    'gap_lower':g,'actual_ground_shift':actual,'certificate_shift':delta})
out['N3']={'two_by_two_examples':examples,
'no_positive_gap_policy':'unresolved_or_promote_active_space; never assert compression error certified'}
print(json.dumps(out,indent=2,ensure_ascii=False))
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
