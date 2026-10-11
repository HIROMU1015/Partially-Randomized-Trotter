"""Exact endpoint serialization, fixed-precision scalar certification.

Input intervals are directed mpmath.iv enclosures, never mp point witnesses.
No registered parameters are evaluated at import time.
"""
from fractions import Fraction as F
import mpmath as mp


def configure():
    mp.mp.dps = 100
    mp.iv.dps = 100


def iv(q):
    q = F(q)
    return mp.iv.mpf(q.numerator) / q.denominator


def endpoint(raw):
    sign, mantissa, exponent, bits = raw
    if bits < 0:
        raise ArithmeticError('Nonfinite endpoint; no precision retry')
    return F(-mantissa if sign else mantissa) * F(2) ** exponent


def bounds(x):
    return tuple(endpoint(p) for p in x._mpi_)


def upper(x):
    return bounds(x)[1]


def lower(x):
    return bounds(x)[0]


def record(x):
    a, b = bounds(x)
    return {'lo': str(a), 'hi': str(b)}


def from_record(x):
    a, b = F(x['lo']), F(x['hi'])
    if a > b:
        raise ArithmeticError('Reversed interval')
    return mp.iv.mpf([iv(a).a, iv(b).b])


def midpoint(x):
    a, b = bounds(x)
    return (a + b) / 2


def real_exact(z):
    """Treat binary64 components as exact rationals, with no decimal conversion."""
    return iv(F(float(z)))


def complex_exact(z):
    return mp.iv.mpc(real_exact(z.real), real_exact(z.imag))


def ceil_upper(x):
    q = upper(x)
    return -((-q.numerator) // q.denominator)


def outward_upper(q, bits=256):
    """Fixed outward dyadic serialization; never changes the 100dps context."""
    q=F(q)*(1<<bits)
    return F(-((-q.numerator)//q.denominator),1<<bits)


def outward_lower(q, bits=256):
    q=F(q)*(1<<bits)
    return F(q.numerator//q.denominator,1<<bits)


def scalar_error_bounds(delta, mass, eta, n):
    configure()
    d, b, e = iv(delta), iv(4 * F(mass)), iv(eta)
    x, y = iv(F(21, 20)) * d, b * d**3
    tail5 = mp.iv.exp(x) - sum((x**k / __import__('math').factorial(k) for k in range(5)), iv(0))
    sym = 2*tail5 + y*(mp.iv.exp(x)-1-x) + mp.iv.exp(x)*(mp.iv.exp(y)-1-y)
    rte = mp.iv.exp(d/4) * (d/4)**6 / 720
    ea, eb = d**6*(b*b/e+b*b)/8, d**6*b*b/8
    return {'A': (1+sym+2*ea+rte)**n-1,
            'B': (1+sym+2*eb+eb**2+rte*(1+eb)**2)**n-1}
