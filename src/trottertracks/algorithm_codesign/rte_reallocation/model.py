"""Fixed degree-three ensembles, with rational description and Pauli controls.

No synthesizer, matrix, stochastic sampling or shared trotterlib imports.
Enumeration is the small pilot evaluator, not I0 generator preprocessing.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from math import factorial

P = (F(3, 4), F(1, 4))
CONTEXTS = ('pauli_commuting', 'pauli_noncommuting', 'distinct_basis')
ARMS = ('ordinary', 'PTSC_K0', 'A')
AXES = {'pauli_commuting': ('ZI', 'IZ'),
        'pauli_noncommuting': ('ZI', 'XX')}
TABLE = {('X','Y'):('Z',1), ('Y','X'):('Z',3),
         ('Y','Z'):('X',1), ('Z','Y'):('X',3),
         ('Z','X'):('Y',1), ('X','Z'):('Y',3)}


def pauli_mul(left, right):
    """Return axis and i**phase for left @ right, without dropping phase."""
    axis, phase = '', 0
    for a,b in zip(left,right,strict=True):
        if a == 'I': c,q = b,0
        elif b == 'I': c,q = a,0
        elif a == b: c,q = 'I',0
        else: c,q = TABLE[a,b]
        axis += c; phase += q
    return axis, phase % 4


def reduce_indices(word):
    out=[]
    for index in word:
        if out and out[-1] == index: out.pop()
        else: out.append(index)
    return tuple(out)


@dataclass(frozen=True)
class Event:
    arm: str
    label: str
    a: F
    b: F
    label_probability: F
    word: tuple = ()              # circuit-time order, integer Q or Pauli axes
    rotation: object = None       # integer Q or Pauli axis
    rotation_sign: int = 1        # exp(-i sign atan(b/a) Q)
    phase: int = 0                # exact scalar i**phase
    complement: bool = False     # A odd-event rewrite only

    @property
    def norm_square(self): return self.a*self.a+self.b*self.b


def description(x, arm):
    x=F(x)
    if x < 0: raise ValueError('x is absolute time')
    if arm not in ARMS: raise ValueError('unknown I0 arm')
    t=[x**n/factorial(n) for n in range(4)]
    if x == 0: return [(0,F(1),F(0))]
    if arm == 'ordinary': return [(0,t[0],t[1]),(2,t[2],t[3])]
    if arm == 'PTSC_K0': return [(0,t[0],t[1]),(2,t[2],F(0)),(3,t[3],F(0))]
    rho=(x+x**3/6)/(1+x*x/2)
    return [(0,F(1),rho),
            (1,2*x**3/(3*(x*x+2)),2*x*x/(x*x+6)),
            (2,x*x*(x*x+2)/(2*(x*x+6)),x**3/6)]


def collected_coefficients(context,x,sigma):
    if context not in AXES: raise ValueError('CTS requires registered Pauli collection context')
    coefficients={}
    for n in range(1,4):
        for word in product(range(2),repeat=n):
            axis,q,prob='II',(-sigma*n)%4,F(1)
            for j in word:
                axis,extra=pauli_mul(AXES[context][j],axis);q=(q+extra)%4;prob*=P[j]
            value=prob*F(x)**n/factorial(n)
            if q in (2,3): value=-value
            key=(axis,q%2); coefficients[key]=coefficients.get(key,F(0))+value
    return {k:v for k,v in coefficients.items() if v}


def events(context,x,sigma,arm):
    if context not in CONTEXTS or type(sigma) is not int or sigma not in (-1,1):
        raise ValueError('unregistered context or time sign')
    x=F(x)
    if arm == 'CTS_collected':
        c=collected_coefficients(context,x,sigma)
        ls=sum(abs(v) for (axis,imag),v in c.items() if imag)
        result=[]
        for (axis,imag),v in sorted(c.items()):
            if not imag:
                result.append(Event(arm,f'C:{axis}',abs(v),F(0),F(1),word=(axis,),phase=2 if v<0 else 0))
            elif ls:
                result.append(Event(arm,f'S:{axis}',F(1),ls,abs(v)/ls,
                                    rotation=axis,rotation_sign=-1 if v>0 else 1))
        if not ls: result.append(Event(arm,'I',F(1),F(0),F(1)))
        return result
    result=[]
    for k,a,b in description(x,arm):
        if a == b == 0: continue
        if b == 0:
            for word in product(range(2),repeat=k):
                prob=F(1)
                for j in word:prob*=P[j]
                result.append(Event(arm,f'{k}:{word}',a,b,prob,word=word,phase=(-sigma*k)%4))
        else:
            for indices in product(range(2),repeat=k+1):
                prob=F(1)
                for j in indices:prob*=P[j]
                result.append(Event(arm,f'{k}:{indices}',a,b,prob,word=indices[1:],rotation=indices[0],
                                    rotation_sign=sigma,phase=(-sigma*k)%4,complement=(arm=='A' and k%2==1)))
    return result


def cts_ls(context,x):
    """Static same-domain angle inventory, no target matrix/cost acquisition."""
    x=F(x)
    if context == 'pauli_commuting': return x-x**3/6
    if context == 'pauli_noncommuting': return x-F(5,8)*x**3/6
    raise ValueError('CTS unavailable for primary I0 context')
