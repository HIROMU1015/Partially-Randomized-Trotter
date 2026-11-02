"""Literal finite CTS operator ensemble, independent exact Pauli algebra.

Theorem 1 / Supplement Note5 Eq13-15 of Peetz, Smart, Narang 2026.
Keeps I and negative even identity correction separate; no optimized variant.
"""
from fractions import Fraction as F
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('g4_exact',Path(__file__).with_name('g4_independent_certificate.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class Algebraic:
    """Q[c], c=cos(pi/8), c^4=c^2-1/8, positive selected real root."""
    def __init__(self,coefficients):
        p=list(map(F,coefficients if isinstance(coefficients,(tuple,list)) else [coefficients]))
        p+=[F(0)]*max(0,4-len(p))
        for k in range(len(p)-1,3,-1):p[k-2]+=p[k];p[k-4]-=p[k]/8
        self.p=tuple(p[:4])
    def __add__(self,z):
        z=alg(z);return Algebraic([a+b for a,b in zip(self.p,z.p)])
    __radd__=__add__
    def __neg__(self):return Algebraic([-x for x in self.p])
    def __sub__(self,z):return self+-alg(z)
    def __rsub__(self,z):return alg(z)+-self
    def __mul__(self,z):
        z=alg(z);p=[F(0)]*7
        for i,a in enumerate(self.p):
            for j,b in enumerate(z.p):p[i+j]+=a*b
        return Algebraic(p)
    __rmul__=__mul__
    def __truediv__(self,z):return self*F(1,F(z))
    def __eq__(self,z):return self.p==alg(z).p
    def json(self):return list(map(str,self.p))
    def interval(self):
        r=v.sqrt_enclosure(2);c=v.Interval(v.sqrt_enclosure(2+r.lo).lo/2,v.sqrt_enclosure(2+r.hi).hi/2)
        total=v.Interval(0)
        for a in reversed(self.p):total=total*c+a
        return total

def alg(z):return z if isinstance(z,Algebraic) else Algebraic(z)
C=Algebraic([0,1]);S=4*C*C*C-3*C
TIMES={('X','Y'):('Z',1),('Y','X'):('Z',3),('Y','Z'):('X',1),('Z','Y'):('X',3),('Z','X'):('Y',1),('X','Z'):('Y',3)}

def pauli_multiply(a,b):
    axis='';phase=0
    for x,y in zip(a,b):
        if x=='I':z=x if y=='I' else y;q=0
        elif y=='I':z=x;q=0
        elif x==y:z='I';q=0
        else:z,q=TIMES[x,y]
        axis+=z;phase+=q
    return axis,phase%4

def poly_product(a,b):
    out={}
    for (x,p),c in a.items():
        for (y,q),d in b.items():
            axis,extra=pauli_multiply(x,y);phase=(p+q+extra)%4
            sign=-1 if phase>=2 else 1;key=(axis,phase%2)
            out[key]=out.get(key,alg(0))+sign*c*d
    return {k:z for k,z in out.items() if z!=0}

def collected_target(xs):
    x=F(xs);R={('ZI',0):alg(F(3,4)),('IZ',0):C/4,('XY',0):S/4}
    R2=poly_product(R,R);R3=poly_product(R2,R)
    if R2!={('II',0):alg(F(5,8)),('ZZ',0):3*C/8}:raise ArithmeticError('R2 symbolic identity')
    out={('II',0):alg(1)}
    for power,mult,imag in ((R,-x,1),(R2,-x*x/2,0),(R3,x**3/6,1)):
        for (axis,p),z in power.items():
            phase=(p+imag)%2;sign=-1 if p+imag==2 else 1;key=(axis,phase)
            out[key]=out.get(key,alg(0))+sign*mult*z
    expected={('II',0):alg(1-F(5,16)*x*x),('ZZ',0):-3*C*x*x/16,
        ('ZI',1):x*(x*x*(C*C+5)-48)/64,
        ('IZ',1):C*x*(7*x*x-24)/96,
        ('XY',1):S*x*(5*x*x-48)/192,('YX',1):C*S*x**3/64}
    if out!=expected:raise ArithmeticError('finite P3 Pauli expansion')
    return out

def definition(xs):
    x=F(xs);target=collected_target(xs)
    imag={axis:z for (axis,p),z in target.items() if p==1}
    signs={axis:(1 if z.interval().lo>0 else -1 if z.interval().hi<0 else 0) for axis,z in imag.items()}
    if not all(signs.values()):raise ArithmeticError('registered CTS coefficient sign unresolved')
    Ls=sum((signs[k]*z for k,z in imag.items()),alg(0));L=Ls.interval()
    ratio=(L.lo+L.hi)/2;angle_error=2*max(abs(ratio-L.lo),abs(L.hi-ratio))
    norm=v.Interval(v.sqrt_enclosure(1+L.lo*L.lo).lo,v.sqrt_enclosure(1+L.hi*L.hi).hi)
    real={'II':alg(F(5,16)*x*x),'ZZ':3*C*x*x/16}
    events=[]
    for axis,z in real.items():
        events.append({'label':'real_minus_'+axis,'axis':axis,'real_event':True,'phase_i_power':2,'rotation_sign':0,
            'ideal_coefficient':z.interval().json(),'algebraic_coefficient':z.json(),'angle_error_upper':'0'})
    for axis,z in imag.items():
        absz=signs[axis]*z;coef=norm*absz.interval()/L
        events.append({'label':'rotation_'+axis,'axis':axis,'real_event':False,'phase_i_power':0,'rotation_sign':-signs[axis],
            'ideal_coefficient':coef.json(),'imaginary_algebraic_target':z.json(),'angle_error_upper':str(angle_error)})
    return {'x':xs,'events':events,'Ls_algebraic':Ls.json(),'Ls_interval':L.json(),'fixed_rational_rotation_ratio':str(ratio),
        'target_algebraic':{axis+':'+str(p):z.json() for (axis,p),z in target.items()},
        'real_identity_correction_not_fused_into_rotation':True,'Pauli_expansion_product_terms':{'R2':9,'R3_after_R2_collection':6},
        'source_specialization':'finite M3 of published Theorem1 and SupplementNote5 Eq13-15; operator ensemble only, not SCU channel metric'}
