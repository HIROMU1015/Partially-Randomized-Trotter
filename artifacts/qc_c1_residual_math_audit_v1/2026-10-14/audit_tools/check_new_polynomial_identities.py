"""New formal identities only. No numerical Hamiltonians or eigenvalue calls.

The old 21 BCH/Pauli checks are imported as documented lemmas, not re-executed.
This script does not import that old checker, the library, or any pilot data.
All variables remain unassigned; fractions are polynomial constants only.
"""
from fractions import Fraction
from pathlib import Path
import json,resource,time,sys,hashlib

class Poly:
    def __init__(self,v=0):
        self.terms=dict(v) if isinstance(v,dict) else {():Fraction(v)} if v else {}
    def __add__(self,other):
        other=other if isinstance(other,Poly) else Poly(other)
        d=dict(self.terms)
        for m,c in other.terms.items():d[m]=d.get(m,Fraction(0))+c
        return Poly({m:c for m,c in d.items() if c})
    __radd__=__add__
    def __neg__(self):return Poly({m:-c for m,c in self.terms.items()})
    def __sub__(self,other):return self+-other if isinstance(other,Poly) else self+Poly(-other)
    def __rsub__(self,other):return Poly(other)+-self
    def __mul__(self,other):
        other=other if isinstance(other,Poly) else Poly(other)
        d={}
        for m,c in self.terms.items():
            for n,b in other.terms.items():
                key=tuple(sorted(m+n));d[key]=d.get(key,Fraction(0))+c*b
        return Poly({m:c for m,c in d.items() if c})
    __rmul__=__mul__
    def __pow__(self,n):
        assert isinstance(n,int) and n>=0
        p=Poly(1)
        for _ in range(n):p=p*self
        return p
    def __truediv__(self,n):return self*Fraction(1,n)
    def __eq__(self,other):return self.terms==(other.terms if isinstance(other,Poly) else Poly(other).terms)

def sym(name):return Poly({(name,):Fraction(1)})

def check():
    names=[]
    def eq(name,lhs,rhs):
        assert lhs==rhs,name
        names.append(name)
    GB,GC,T,tB,tC=map(sym,('GB','GC','T','tB','tC'))
    GA=GB+GC+2*T
    d0=GA/12+tB/24+tC/12
    d1=GA/12-tC/24-tB/12
    d2=(GB+GC-T)/12+tB/24-tC/24
    eq('ground_three_kernel_sum_quadratic_form',d0+d1+d2,(GA+GB+GC)/8)
    eq('commuting_BC_nontrivial_kernel_gram',12*d2-(tB-tC)/2,GB+GC-T)
    r,t,z,j=map(sym,('r','t','s','j'))
    p=r+j;q=t+j;Delta=r*t-z*z
    # New strong-C weak-B coefficients, all formal. W=(u0-v0)^2 factors out.
    F=t*(j*r+z*z)**2-2*z*z*p*(j*r+z*z)+r*z*z*p*p
    ACB=F+2*q*Delta**2-q*q*r*Delta
    ACBnum=t**3*r*r+r*(r*r-3*t*t)*z*z+(3*t-2*r)*z**4
    eq('strong_C_ACB_numerator',ACB,ACBnum)
    eq('strong_C_ACB_positive_region_decomposition',ACB,t*Delta**2+(r-t)*z*z*(r*(r+t)-2*z*z))
    BAC=t*Delta**2+z*z*(r-t)*(r*r-2*r*t+z*z)
    BACnum=t**3*r*r+r*r*(r-3*t)*z*z+r*z**4
    eq('strong_C_BAC_numerator',BAC,BACnum)
    rho,k=map(sym,('rho','k'))
    FA=1+(rho*rho-3)*k+(3-2*rho)*k*k
    FB=1+rho*(rho-3)*k+rho*k*k
    eq('FA_normalized_positive_region',FA,(1-k)**2+(rho-1)*k*(rho+1-2*k))
    eq('FA_completed_square',4*(3-2*rho)*FA,(2*(3-2*rho)*k+rho*rho-3)**2+(1-rho)**3*(rho+3))
    eq('FB_completed_square',4*rho*FB,(2*rho*k+rho*(rho-3))**2+rho*(4-rho)*(rho-1)**2)
    return names

if __name__=='__main__':
    tic=time.monotonic();names=check()
    this=Path(__file__)
    out={'status':'NEW_FORMAL_POLYNOMIAL_IDENTITIES_PASS','identity_names':names,
         'identity_count':len(names),'coefficient_variables_remain_unassigned':True,
         'method':'Exact Fraction multivariate polynomial equality;manual analytic positivity/PSD/Riesz arguments remain separately documented',
         'old_21_identity_checks_reexecuted':False,'pilot_results_read':0,
         'numeric_Hamiltonians_generated':0,'eigenvalue_or_ground_solvers':0,
         'parameter_point_substitutions':0,'numeric_posthoc_aggregations':0,
         'new_cost_recomputations':0,'library_imports':0,'optional_dependencies':0,
         'python':sys.version.split()[0],'workers':1,'backend':'stdlib fractions, no BLAS/GPU',
         'wall_seconds_static_check':time.monotonic()-tic,
         'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'checker_SHA256':hashlib.sha256(this.read_bytes()).hexdigest(),
         'scientific_performance_benchmark':False,'external_scientific_reproduction':False,
         'research_adoption_or_closure_decided':False,'mandatory_stop':True}
    target=this.parents[1]/'new_symbolic_checks.json'
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='identity_names'},ensure_ascii=False))
