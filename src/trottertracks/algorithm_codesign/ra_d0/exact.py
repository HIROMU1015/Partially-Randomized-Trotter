"""Rational interval bookkeeping; no science, angle or circuit evaluation."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import isqrt

DPS = 100
DENOMINATOR = 2**60
E = F(1, 200)
DELTA_NUM = F(1, 10**12)


def sqrt_interval(value):
    """Integer proof of an absolute 100-decimal-place enclosure."""
    value = F(value)
    if value < 0:
        raise ValueError("negative radicand")
    scale = 10**DPS
    k = isqrt(value.numerator * scale**2 // value.denominator)
    lo = F(k, scale)
    return lo, lo if lo * lo == value else F(k + 1, scale)


def log_interval(value):
    # Decimal.ln is correctly rounded using ROUND_HALF_EVEN. One representable
    # neighbour on either side encloses its exact real value, including ties.
    value = F(value)
    if value.denominator != 1 or value <= 0:
        raise ValueError("this protocol uses an exact positive integer log input")
    with localcontext() as ctx:
        ctx.prec = DPS
        v = Decimal(value.numerator).ln()
        return F(ctx.next_minus(v)), F(ctx.next_plus(v))


def kappa_upper(n, ell_upper):
    if type(n) is not int or n < 1 or ell_upper <= 0:
        raise ValueError("invalid confidence input")
    a = F(4, 3) * ell_upper
    return (a + sqrt_interval(a*a + 8*n*ell_upper)[1]) / (2*n)


def dot(a, b):
    return sum((F(x)*F(y) for x, y in zip(a, b, strict=True)), F(0))


def ceil(value):
    value = F(value)
    return -(-value.numerator // value.denominator)


def normalize_direction(a, b, degree):
    a, b = F(a), F(b)
    lo, hi = sqrt_interval(a*a + b*b)
    if lo <= 0 or not 0 <= degree <= 3 or (degree == 3 and b):
        raise ValueError("invalid degree column")
    column = [(F(0), F(0)) for _ in range(4)]
    column[degree] = a/hi, a/lo
    if b:
        column[degree+1] = b/hi, b/lo
    return column, (lo, hi)


def quantize(q, y, denominator=DENOMINATOR):
    """Fixed largest remainder; reject rather than clip negative solver values."""
    q = [F(v) for v in q]
    if not q or any(v < 0 for v in q) or sum(q) <= 0 or F(y) <= 0:
        raise ValueError("invalid nominal probability law")
    # Nominal sum error is not silently a certificate. The resulting *new*
    # exact law is what the caller must certify below.
    total = sum(q)
    scaled = [v/total*denominator for v in q]
    counts = [v.numerator // v.denominator for v in scaled]
    left = denominator-sum(counts)
    order = sorted(range(len(q)), key=lambda j: (-(scaled[j]-counts[j]), j))
    for j in order[:left]:
        counts[j] += 1
    yy = F(y)*denominator
    # Nearest rational, deterministic half-up tie handling.
    iy = (2*yy.numerator+yy.denominator)//(2*yy.denominator)
    if iy <= 0:
        raise ValueError("nonpositive quantized inverse normalization")
    return [F(v, denominator) for v in counts], F(iy, denominator)


def mean_residual_upper(columns, target, q, y):
    if any(v < 0 for v in q) or y <= 0:
        raise ValueError("nonnegative law required")
    residual = []
    for k, t in enumerate(target):
        lo = sum(q[j]*columns[j][k][0] for j in range(len(q)))-y*t
        hi = sum(q[j]*columns[j][k][1] for j in range(len(q)))-y*t
        residual.append(max(abs(lo), abs(hi)))
    return sum(residual), residual


def certify_law(columns, target, d, costs, q, y, n, ell_upper, caps=None):
    """B3 implementation certificate, never a B1/B2 membership certificate."""
    q, y = [F(v) for v in q], F(y)
    if (len(q) != len(columns) or any(v < 0 for v in q) or sum(q) != 1 or y <= 0
            or any(DENOMINATOR % v.denominator for v in q+[y])):
        return {"certified": False, "reason": "INVALID_SAMPLER_LAW"}
    xi, residual = mean_residual_upper(columns, target, q, y)
    h = E*y-dot(d, q)-xi
    resources = {k: 2*n*(dot(v, q)+(F(5, 2) if k == "1Q" else 0))
                 for k, v in costs.items()}
    valid = xi <= y*DELTA_NUM and h >= kappa_upper(n, ell_upper)
    valid &= all(resources[k] <= F(v) for k, v in (caps or {}).items())
    return {"certified": valid, "reason": "PASS" if valid else "UNCERTIFIED_NUMERICAL_POINT",
            "xi": str(xi), "residual_by_degree": list(map(str, residual)),
            "h_lower": str(h), "B": str(1/y), "B_squared": str(1/y**2),
            "implementation_bias_upper": str(dot(d, q)/y), "mean_bias_upper": str(xi/y),
            "resources": {k: str(v) for k, v in resources.items()},
            "q_exact": list(map(str, q)), "y_exact": str(y)}
