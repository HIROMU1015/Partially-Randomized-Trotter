"""Finite-bit canonical proposals; numerical rounding, not a new sampling law.

Price samples use NumPy as recorded. The hypothetical quantum-shot sampler
uses independent b-bit integers with this explicit CDF and retains a bound on
the deviation from the nominal canonical law. No quantum shots are executed.
"""
from bisect import bisect_right
from fractions import Fraction
import math


def integer_cdf(probabilities,bits):
    values=[Fraction.from_float(float(p)) for p in probabilities]
    if any(p<0 for p in values) or not sum(values):raise ValueError('invalid probabilities')
    total=sum(values);scale=1<<bits;running=Fraction(0);cdf=[]
    for p in values:
        running+=p;cdf.append((running.numerator*total.denominator*scale)//(running.denominator*total.numerator))
    cdf[-1]=scale
    masses=[b-a for a,b in zip([0]+cdf[:-1],cdf)]
    if any(p>0 and n==0 for p,n in zip(values,masses)):
        raise ArithmeticError('finite-bit CDF erased a positive support point')
    return tuple(cdf)


def draw_cdf(cdf,bits,rng):
    return bisect_right(cdf,rng.getrandbits(bits))


def weight_interval(factors,dps):
    """Enclose a compressed product of exact-binary normalizers."""
    import mpmath as mp
    old=mp.iv.dps
    try:
        mp.iv.dps=dps;value=mp.iv.mpf(1)
        for base,count in factors:
            numerator,denominator=float(base).as_integer_ratio()
            value*=(mp.iv.mpf(numerator)/denominator)**int(count)
        bounds=[]
        for sign,mantissa,exponent,bits in value._mpi_:
            if bits<0:raise ArithmeticError('nonfinite weight enclosure')
            bounds.append(Fraction((-1 if sign else 1)*mantissa)*Fraction(2)**exponent)
        log_upper=math.nextafter(float(mp.iv.log(value).b),math.inf)
        return tuple(bounds),log_upper
    finally:mp.iv.dps=old


def proposal_budget(log_range,draws,largest_support,minimum_positive_probability=1.):
    # L1 distance of one floor-CDF law is <=2*M*2^-b. The product-law
    # telescoping bound and |weighted outcome|<=L give the stated signal disk.
    target=math.ulp(1.)
    log_coefficient=math.log(2*math.sqrt(2))+math.log(max(1,draws))+math.log(max(1,largest_support))+log_range
    bits=max(64,math.ceil((log_coefficient-math.log(target))/math.log(2))+2)
    bits=max(bits,math.ceil(-math.log2(minimum_positive_probability))+2)
    bound=math.exp(log_coefficient-bits*math.log(2))
    return dict(bits_per_uniform_draw=bits,max_categorical_draws=draws,
                largest_support=largest_support,signal_error_upper=bound,
                minimum_positive_probability_nominal=minimum_positive_probability,
                classical_uniform_bits_upper=bits*draws,
                weight_arithmetic_dps=max(100,math.ceil((log_range+math.log(max(1,draws))-math.log(target))/math.log(10))+30),
                model='canonical float probabilities normalized as exact rationals; integer floor CDF; fresh independent bits',
                price_sampler='separately recorded NumPy float64 conditional sampler; not a quantum-shot implementation test')
