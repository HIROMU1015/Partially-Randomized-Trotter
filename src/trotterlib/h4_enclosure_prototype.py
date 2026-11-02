"""C artificial proof prototypes, not an H4 certificate generator.

Exact rational interval operations demonstrate outward enclosure, finite RTE
normalization and exponential remainder propagation on tiny supplied fixtures.
No file/NPZ reader, science-library import, molecule runner or grant activation.
Full H4 assembly and certificate dispatch are deliberately not implemented.
"""
from fractions import Fraction as Q
from math import factorial, isqrt


class Interval:
    def __init__(self, lo, hi=None):
        self.lo=Q(lo);self.hi=Q(lo if hi is None else hi)
        if self.lo>self.hi:
            raise ValueError('REVERSED_INTERVAL')

    def __add__(self, other):
        other=other if isinstance(other,Interval) else Interval(other)
        return Interval(self.lo+other.lo,self.hi+other.hi)

    __radd__=__add__

    def __neg__(self):
        return Interval(-self.hi,-self.lo)

    def __sub__(self, other):
        return self+-as_interval(other)

    def __mul__(self, other):
        other=as_interval(other)
        products=[a*b for a in [self.lo,self.hi] for b in [other.lo,other.hi]]
        return Interval(min(products),max(products))

    __rmul__=__mul__

    def __truediv__(self, other):
        other=as_interval(other)
        if other.lo<=0<=other.hi:
            raise ValueError('DIVISOR_CONTAINS_ZERO')
        return self*Interval(1/other.hi,1/other.lo)

    def __pow__(self, power):
        if type(power) is not int or power<0:
            raise ValueError('NONNEGATIVE_INTEGER_POWER_ONLY')
        out=Interval(1)
        for _ in range(power):
            out=out*self
        return out

    def contains(self, other):
        other=as_interval(other)
        return self.lo<=other.lo<=other.hi<=self.hi


def as_interval(value):
    return value if isinstance(value,Interval) else Interval(value)


def sqrt_point(value, *, bits=96):
    value=Q(value)
    if value<0 or bits<1:
        raise ValueError('SQRT_DOMAIN')
    a,b=isqrt(value.numerator),isqrt(value.denominator)
    if a*a==value.numerator and b*b==value.denominator:
        return Interval(Q(a,b))
    scale=1<<bits
    floor=isqrt(value.numerator*scale*scale//value.denominator)
    return Interval(Q(floor,scale),Q(floor+1,scale))


def _log_1_to_2(x, terms):
    z=(x-1)/(x+1)
    total=2*sum((z**(2*k+1)/Q(2*k+1) for k in range(terms)),Q(0))
    tail=2*z**(2*terms+1)/(Q(2*terms+1)*(1-z*z))
    return Interval(total,total+tail)


def log_point(value, *, terms=80):
    x=Q(value);k=0
    if x<=0 or terms<1:
        raise ValueError('LOG_DOMAIN')
    while x>2:
        x/=2;k+=1
    while x<1:
        x*=2;k-=1
    return _log_1_to_2(x,terms)+k*_log_1_to_2(Q(2),terms)


def finite_law_prototype(tau, *, K=4, repetitions=32, bits=96):
    tau=abs(Q(tau))
    if K<0 or K%2 or repetitions<1:
        raise ValueError('FIXED_FINITE_LAW_DOMAIN')
    weights=[tau**k/Q(factorial(k))*sqrt_point(1+(tau/Q(k+1))**2,bits=bits)
             for k in range(0,K+1,2)]
    normalizer=sum(weights,Interval(0))
    log_B=repetitions*Interval(log_point(normalizer.lo).lo,log_point(normalizer.hi).hi)
    return dict(weights=weights,normalizer=normalizer,B=normalizer**repetitions,log_B=log_B,
                scope='ARTIFICIAL_PROTOTYPE_ONLY_NOT_H4_CERTIFICATE')


def multiply(A,B):
    if not A or len(A[0])!=len(B):
        raise ValueError('MATRIX_SHAPE')
    return [[sum((A[i][k]*B[k][j] for k in range(len(B))),0)
             for j in range(len(B[0]))] for i in range(len(A))]


def identity(n):
    return [[Q(i==j) for j in range(n)] for i in range(n)]


def exponential_prototype(A, *, degree=24):
    n=len(A)
    if not n or any(len(row)!=n for row in A) or degree<1:
        raise ValueError('MATRIX_EXP_DOMAIN')
    A=[[Q(x) for x in row] for row in A]
    alpha=max(sum(abs(x) for x in row) for row in A)
    scales=0
    while alpha>Q(1,2):
        A=[[x/2 for x in row] for row in A];alpha/=2;scales+=1
    term=identity(n);total=identity(n)
    for k in range(1,degree+1):
        term=[[x/k for x in row] for row in multiply(A,term)]
        total=[[x+y for x,y in zip(a,b)] for a,b in zip(total,term)]
    # Every subsequent absolute Taylor-term ratio is <= alpha/(degree+2).
    error=alpha**(degree+1)/Q(factorial(degree+1))/(1-alpha/Q(degree+2))
    enclosed=[[Interval(x-error,x+error) for x in row] for row in total]
    for _ in range(scales):
        enclosed=multiply(enclosed,enclosed)
    return enclosed


def finite_polynomial(A, *, K=4):
    n=len(A);term=identity(n);total=identity(n)
    # Frozen paired-even K law retains ordinary Taylor degrees through K+1.
    for k in range(1,K+2):
        term=[[x/k for x in row] for row in multiply(A,term)]
        total=[[x+y for x,y in zip(a,b)] for a,b in zip(total,term)]
    return total


def product_error_bound(norms, errors):
    if len(norms)!=len(errors) or any(x<0 for x in [*norms,*errors]):
        raise ValueError('PRODUCT_BOUND_DOMAIN')
    norm=Q(1);error=Q(0)
    for M,d in zip(norms,errors):
        error=(Q(M)+Q(d))*error+Q(d)*norm
        norm*=Q(M)
    return error
