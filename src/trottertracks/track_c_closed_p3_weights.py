"""Compressed finite-arithmetic path weights, reusing C's interval product.

Exact G7 w=alpha/pi is rounded once to binary64 per distinct event. The
whole-path product is enclosed using the existing directed interval power
routine. The resource qualification separately charges this rounding bias and
inflates the range; exact rational weight identities remain the reference.
"""
from collections import Counter
from fractions import Fraction as F
import math
from .track_c_h4_finite_bits import weight_interval


def compressed_weight(weights):
    return compressed_weight_counts(Counter(float(w) for w in weights))


def compressed_weight_counts(counts):
    counts=dict(counts)
    if any(type(n) is not int or n<=0 for n in counts.values()):raise ValueError('positive integer counts required')
    if any(not math.isfinite(w) or w<=0 for w in counts):raise ValueError('nonpositive/nonfinite path weight')
    bounds,log_upper=weight_interval(tuple(counts.items()),100)
    lo,hi=bounds
    if hi-lo>F(1,10**24)*lo:raise ArithmeticError('path weight interval too broad; failure is not zero')
    value=float((lo+hi)/2)
    error=max(abs(F.from_float(value)-lo),abs(F.from_float(value)-hi))
    return value,error,dict(events=sum(counts.values()),distinct_binary_weights=len(counts),
        log_upper=log_upper,interval_lower=str(lo),interval_upper=str(hi),rounding_error_upper=str(error))
