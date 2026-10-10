"""Source-input algebra checks for an authorized GPT review.
No repository import, runner, event generation, optimization, compiler or molecule calculation.
Inputs are rational matrices transcribed from fixed mechanisms.py at 25d7135.
"""
import sympy as s
import json
from pathlib import Path
I=s.eye(2);X=s.Matrix([[0,1],[1,0]]);Y=s.Matrix([[0,-s.I],[s.I,0]]);Z=s.diag(1,-1)
kron=s.kronecker_product
R=s.Rational

def jw_a(n,j):
    a=s.zeros(2**n)
    for b in range(2**n):
        if b>>j & 1:a[b^(1<<j),b]=(-1)**((b & ((1<<j)-1)).bit_count())
    return a

def commute(a,b):return a*b-b*a

def iszero(a):return all(s.simplify(x)==0 for x in a)

# A: exactly the saved 3-mode proxy-mismatch example, not a new fixture sweep.
a=[jw_a(3,j) for j in range(3)]; n=[v.T*v for v in a]
T=a[0].T*a[2]+a[2].T*a[0]
D=R(1,10)*n[0]-R(4,5)*n[1]-R(7,10)*n[2]
F0=n[0]+n[1]-n[2];F1=D+R(1,25)*T
H=F0**2+F1**2;core=F0**2+D**2
residual=H-core
expected=(-R(3,125)*s.eye(8)-R(8,125)*n[1])*T+R(1,625)*(n[0]+n[2]-2*n[0]*n[2])
A_checks={
 'commutes_n1':iszero(commute(H,n[1])),
 'commutes_n0_plus_n2':iszero(commute(H,n[0]+n[2])),
 'residual_exact_identity':iszero(residual-expected),
 'residual_eigenvalues':{str(k):v for k,v in residual.eigenvals().items()},
 'nontrivial_sector_dimensions':[sum(1 for b in range(8) if (b>>1&1)==spect and (b&1)+(b>>2&1)==1) for spect in (0,1)],
}
assert A_checks['commutes_n1'] and A_checks['commutes_n0_plus_n2'] and A_checks['residual_exact_identity']

# A: the existing isotropic control has a diagonal two-qubit target.
a2=[jw_a(2,j) for j in range(2)]
def lift2(g):
    return sum((g[p,q]*a2[p].T*a2[q] for p in range(2) for q in range(2)),s.zeros(4))
iso=lift2(X/s.sqrt(2))**2+lift2(Z/s.sqrt(2))**2
A_checks['isotropic_H_equals_parity_projector']=iszero(iso-(s.eye(4)-kron(Z,Z))/2)
assert A_checks['isotropic_H_equals_parity_projector']

# B: projected physical Hamiltonian and its known direct SU(2) form.
P=s.diag(1,1,0,0);S=2*P-s.eye(4)
Ht=R(7,10)*kron(I,Z)+R(1,5)*kron(Z,X)+R(2,5)*kron(X,X)
Hb=(Ht+S*Ht*S)/2
Hphys=Hb[:2,:2]
B_checks={'projected_Hphys':str(Hphys),
 'direct_expected':iszero(Hphys-(R(7,10)*Z+R(1,5)*X)),
 'square_scalar':iszero(Hphys**2-R(53,100)*I),
 'off_sector_block_zero':iszero((s.eye(4)-P)*Hb*P)}
assert all(B_checks[k] for k in ('direct_expected','square_scalar','off_sector_block_zero'))

# C: exact t-series to order 3 of the saved THRIFT product.
alpha=s.symbols('alpha', real=True)
A=kron(I,Z)+R(7,10)*kron(Z,I)+R(3,10)*kron(Z,Z)
B0=kron(I,X);B1=kron(X,I)

def exp_coeff(M,order=3): return [M**j/s.factorial(j) for j in range(order+1)]
def polyprod(p,q):return [sum((p[j]*q[k-j] for j in range(k+1)),s.zeros(4)) for k in range(4)]
prod=polyprod(polyprod(exp_coeff(-s.I*(A+alpha*B0)),exp_coeff(s.I*A)),exp_coeff(-s.I*(A+alpha*B1)))
ref=exp_coeff(-s.I*(A+alpha*(B0+B1)))
err=[(p-q).applyfunc(s.simplify) for p,q in zip(prod,ref)]
YY=kron(Y,Y)
coeff=s.simplify(s.trace(YY*err[3])/4)
C_checks={'B0_B1_commute':iszero(commute(B0,B1)),
 'time_series_zero_orders':[iszero(err[j]) for j in range(3)],
 'order3_YY_coefficient':str(coeff),
 'order3_only_YY':iszero(err[3]-coeff*YY),
 'expected_leading_norm_at_alpha0p1_t0p2':float(abs(complex(coeff.subs(alpha,R(1,10))))*R(1,5)**3)}
assert C_checks['B0_B1_commute'] and all(C_checks['time_series_zero_orders']) and C_checks['order3_only_YY']

out={'scope':__doc__,'A':A_checks,'B':B_checks,'C':C_checks,'claims':'Exact symbolic checks on transcribed fixed inputs; not reproduction of saved floating rows or performance validation.'}
path=Path(__file__).with_name('independent_review_checks.json')
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
