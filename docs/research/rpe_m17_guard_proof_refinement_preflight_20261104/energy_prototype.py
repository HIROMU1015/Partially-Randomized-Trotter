"""Small proof primitives for artificial tests; no loader, RNG or scientific CLI.

Inputs are mathematical bounds that the caller must establish. These functions
do not establish a reachable set, a signed drift, or a real-kernel certificate.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_openblas_kernel_bound_preflight_20261104'))
import kernel_bounds as kernel


def rational(value):
    if not (isinstance(value, Q) or type(value) in (int, str)):
        raise ValueError('Established exact rational bound required')
    return Q(value)


def real_row_error(abs_product_sum, columns):
    """Existing real-output forest bound, retaining its underflow term.

    abs_product_sum bounds the real row's sum of absolute real products;
    it is not a Frobenius norm or a typical-state estimate.
    """
    products = rational(abs_product_sum)
    if products < 0:
        raise ValueError('Nonnegative product-sum bound required')
    k = kernel.path_bounds(1, columns)
    return (k.mv_rel*products+k.dot_abs)/2


def energy_step(norm_squared, signed_drift, signed_radial_error, error_squared):
    """N' <= N + drift + 2*radial + ||e||_H^2, including signed terms.

    drift bounds v*(A*HA-H)v and radial bounds Re((Av)*He).
    No orthogonality, mean-zero roundoff or contraction is inferred here.
    """
    n, d, r, e2 = map(rational, (norm_squared, signed_drift, signed_radial_error, error_squared))
    result = n+d+2*r+e2
    if min(n, e2, result) < 0:
        raise ValueError('Inconsistent energy bounds')
    return result


@dataclass(frozen=True)
class NormAffine:
    """Norm-only map and an upper envelope of every prefix, including input.

    Deliberately has no Growth.gain/additive action-error interface.
    Exact rational prototype; not intended for huge nontrivial rational powers.
    """
    final_scale: Q
    final_offset: Q
    prefix_scale: Q
    prefix_offset: Q

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            object.__setattr__(self, field, rational(getattr(self, field)))
        if (min(self.final_scale, self.final_offset, self.prefix_offset) < 0
                or self.prefix_scale < max(1, self.final_scale)
                or self.prefix_offset < self.final_offset):
            raise ValueError('Invalid norm/prefix envelope')


def primitive(scale, offset=0):
    g, a = rational(scale), rational(offset)
    return NormAffine(g, a, max(1, g), a)


def compose(after, before):
    return NormAffine(after.final_scale*before.final_scale,
                      after.final_scale*before.final_offset+after.final_offset,
                      max(before.prefix_scale, after.prefix_scale*before.final_scale),
                      max(before.prefix_offset, after.prefix_scale*before.final_offset+after.prefix_offset))


def power(value, count):
    if type(count) is not int or count < 0:
        raise ValueError('Nonnegative integer count required')
    result = primitive(1)
    while count:
        if count & 1:
            result = compose(value, result)
        count >>= 1
        if count:
            value = compose(value, value)
    return result
