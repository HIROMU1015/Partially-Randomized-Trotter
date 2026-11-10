"""Pure Q4 interval/CI/joint-geometry helpers. No scientific inputs on import.

Only explicit callers evaluate numbers. Confidence bounds enclose the exact
CI endpoints; witnesses must use INNER endpoints, never the outer enclosure.
The polygon is a rational OUTER polygon for registered irrational native axes.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, product
from math import comb
from pathlib import Path
import subprocess
from types import SimpleNamespace
from .pr_reference_feedback_q1 import AxisCounts

ROOT = Path(__file__).resolve().parents[2]
BASE = 'b159d721cc9e7cee39c246a2b55425e63796b249'
EVIDENCE = '08f9cbe77456ec29d50a992d7165c4d726df7bf7'
BITS = 192
RESOLUTION = F(1, 1 << 128)
ROOT_WIDTH = F(1, 1 << 160)
NATIVE_A = F.from_float(float.fromhex('0x1.921fb54442d18p+0'))


@lru_cache(None)
def pinned_text(path, ref=BASE):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT, text=True)


def pure_functions(path, names, namespace):
    """Pinned AST definitions only; no imports, top-level calls or runners."""
    tree = ast.parse(pinned_text(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError('Missing pinned pure helper')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace


class Arithmetic:
    def __init__(self):
        from mpmath.ctx_iv import MPIntervalContext
        self.iv = MPIntervalContext()
        self.iv.prec = BITS
        ns = pure_functions('scripts/diagnose_pr_reference_q0.py',
                            {'iv', 'exact', 'epsilon'},
                            {'F': F, 'I': self.iv.mpf, 'mp': SimpleNamespace(iv=self.iv)})
        self.r = ns['iv']
        self.exact = ns['exact']
        self.epsilon = ns['epsilon']

    def bounds(self, value):
        return tuple(self.exact(t) for t in value._mpi_)

    def interval(self, lo, hi=None):
        lo, hi = F(lo), F(lo if hi is None else hi)
        if lo > hi:
            raise ValueError('Reversed interval')
        return self.iv.mpf([self.r(lo).a, self.r(hi).b])

    def midpoint(self, value):
        lo, hi = self.bounds(value)
        return (lo+hi)/2

    def record(self, value):
        return list(map(str, self.bounds(value)))


@dataclass(frozen=True)
class Confidence:
    # lower endpoint in [lower_lo,lower_hi], upper in [upper_lo,upper_hi].
    lower_lo: F
    lower_hi: F
    upper_lo: F
    upper_hi: F
    resolved: bool = True

    @property
    def outer(self):
        return self.lower_lo, self.upper_hi

    @property
    def inner(self):
        return self.lower_hi, self.upper_lo


def check_counts(n, k, alpha):
    if type(n) is not int or type(k) is not int or not 0 <= k <= n:
        raise ValueError('Integer count outside 0..n')
    if not 0 <= alpha < 1 or (n and alpha == 0):
        raise ValueError('Measured axes need 0 < alpha < 1')
    if n:
        AxisCounts(n,k)  # Reuse the Q1 bound integer-count contract.


def hoeffding(a, n, k, alpha):
    alpha = F(alpha)
    check_counts(n, k, alpha)
    if not n:
        return Confidence(F(-1), F(-1), F(1), F(1))
    el, eh = a.bounds(a.epsilon(n, alpha))
    c = F(2*k, n)-1
    return Confidence(max(F(-1), c-eh), max(F(-1), c-el),
                      min(F(1), c+el), min(F(1), c+eh))


def binomial_survival(a, n, k, p):
    """P[Bin(n,p)>=k], positive terms; endpoints handled without 0/0."""
    if type(n) is not int or n < 0 or not 0 <= k <= n+1 or not 0 <= p <= 1:
        raise ValueError('Invalid binomial tail')
    if k == 0:
        return a.r(1)
    if k == n+1 or p == 0:
        return a.r(0)
    if p == 1:
        return a.r(1)
    p = a.r(p)
    q = 1-p
    term = comb(n, k)*p**k*q**(n-k)
    total = term
    for j in range(k, n):
        term = term*a.r(F(n-j, j+1))*p/q
        total += term
    return total


def tail_root(a, n, k, target):
    """Bracket the increasing tail root. Overlapping evaluations stay unresolved."""
    lo, hi = F(0), F(1)
    while hi-lo > ROOT_WIDTH:
        mid = (lo+hi)/2
        vl, vh = a.bounds(binomial_survival(a, n, k, mid))
        if vh < target:
            lo = mid
        elif vl > target:
            hi = mid
        elif vl == vh == target:
            return mid, mid, True
        else:
            return lo, hi, False
    return lo, hi, True


def clopper_pearson(a, n, k, alpha):
    alpha = F(alpha)
    check_counts(n, k, alpha)
    if not n:
        return Confidence(F(-1), F(-1), F(1), F(1))
    lower = (F(0), F(0), True) if k == 0 else tail_root(a, n, k, alpha/2)
    mirror = (F(0), F(0), True) if k == n else tail_root(a, n, n-k, alpha/2)
    return Confidence(2*lower[0]-1, 2*lower[1]-1,
                      1-2*mirror[1], 1-2*mirror[0], lower[2] and mirror[2])


def merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if lo > hi:
            continue
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return result


def axis_coefficients(a, geometry):
    if geometry == 'ideal':
        return [(F(1), F(0), F(0)), (F(0), F(1), F(0))]
    if geometry != 'native':
        raise ValueError('Unknown geometry')
    cs = [a.iv.cos(a.r(NATIVE_A)), a.iv.sin(a.r(NATIVE_A))]
    mids = [a.midpoint(x) for x in cs]
    error = sum(max(abs(x-m) for x in a.bounds(v)) for v, m in zip(cs, mids))
    # |x|,|y|<=1: rounding the normal expands each strip by <= error.
    return [(F(1), F(0), F(0)), (mids[0], mids[1], error)]


def polygon(a, cis, geometry):
    planes = [(F(1), F(0), F(1)), (F(-1), F(0), F(1)),
              (F(0), F(1), F(1)), (F(0), F(-1), F(1))]
    for ci, (x, y, e) in zip(cis, axis_coefficients(a, geometry)):
        lo, hi = ci.outer
        planes.extend([(x, y, hi+e), (-x, -y, -lo+e)])
    vertices = set()
    for (x, y, h), (u, v, k) in combinations(planes, 2):
        det = x*v-y*u
        if det:
            p = ((h*v-y*k)/det, (x*k-h*u)/det)
            if all(cx*p[0]+cy*p[1] <= bound for cx, cy, bound in planes):
                vertices.add(p)
    if not vertices:
        return planes, []
    # Exact monotone-chain hull. Degenerate point/segment is retained.
    ps = sorted(vertices)
    def cross(o, x, y):
        return (x[0]-o[0])*(y[1]-o[1])-(x[1]-o[1])*(y[0]-o[0])
    halves = []
    for seq in (ps, list(reversed(ps))):
        half = []
        for p in seq:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half)
    hull = halves[0][:-1]+halves[1][:-1]
    return planes, hull or ps


def point_turns(a, x, y):
    """Split quadrants before atan2; never bridge the negative-axis branch cut."""
    xb, yb = a.bounds(x), a.bounds(y)
    if xb[0] <= 0 <= xb[1] and yb[0] <= 0 <= yb[1]:
        return [(F(0), F(1))]  # Conservative near-origin enclosure.
    def split(pair):
        lo, hi = pair
        return [(lo, F(0)), (F(0), hi)] if lo < 0 < hi else [pair]
    result = []
    for xp, yp in product(split(xb), split(yb)):
        negative_x,negative_y=xp[1]<=0,yp[1]<=0
        absx=(-xp[1],-xp[0]) if negative_x else xp
        absy=(-yp[1],-yp[0]) if negative_y else yp
        quarter=a.iv.atan2(a.interval(*absy),a.interval(*absx))/(2*a.iv.pi)
        if negative_x and negative_y:
            angle=a.r(F(1,2))+quarter
        elif negative_x:
            angle=a.r(F(1,2))-quarter
        elif negative_y:
            angle=1-quarter
        else:
            angle=quarter
        lo,hi=a.bounds(angle)
        result.append((max(F(0),lo),min(F(1),hi)))
    return merge(result)


def radial_bounds(a, theta, planes, rho):
    """OUT proves no common radius; IN proves a common radius exists.

    The common radius interval is intersected across ALL strips. A zero normal
    projection is treated as unresolved unless it is definitely incompatible.
    """
    lower = a.r(rho)
    upper = a.r(1)
    c, s = a.iv.cos(theta), a.iv.sin(theta)
    unresolved = False
    for x, y, h in planes:
        g = a.r(x)*c+a.r(y)*s
        gl, gh = a.bounds(g)
        if gl > 0:
            bound = a.r(h)/g
            ul, uh = a.bounds(upper); bl, bh = a.bounds(bound)
            upper = a.interval(min(ul, bl), min(uh, bh))
        elif gh < 0:
            bound = a.r(h)/g
            ll, lh = a.bounds(lower); bl, bh = a.bounds(bound)
            lower = a.interval(max(ll, bl), max(lh, bh))
        elif gl == gh == 0:
            if h < 0:
                return 'OUT', None
        else:
            # rho*g<=h: use the entire allowed radial range to reject safely.
            if a.bounds(a.interval(rho, 1)*g)[0] > h:
                return 'OUT', None
            unresolved = True
    ll, lh = a.bounds(lower); ul, uh = a.bounds(upper)
    if ll > uh:
        return 'OUT', None
    if not unresolved and lh <= ul:
        return 'IN', (lh, ul)
    return 'UNRESOLVED', (ll, uh)


def angular_region(a, cis, rho, geometry='ideal'):
    """Outer union of polygon/annulus angle components, in turns [0,1].

    Feasibility can change only at a polygon vertex ray or an edge/circle
    intersection. Critical boxes are retained, gaps are rejected only with a
    certified radial OUT at their midpoint. See EXECUTION_SCOPE proof notes.
    """
    rho = F(rho)
    if not 0 < rho <= 1:
        raise ValueError('Physical annulus requires 0 < rho <= 1')
    planes, vs = polygon(a, cis, geometry)
    if not vs:
        return []
    norm2 = lambda p: p[0]*p[0]+p[1]*p[1]
    if max(map(norm2, vs)) < rho*rho:
        return []
    nearest = list(vs)
    for p, q in zip(vs, vs[1:]+vs[:1]):
        d = (q[0]-p[0], q[1]-p[1]); length = norm2(d)
        if length:
            t = max(F(0), min(F(1), -(p[0]*d[0]+p[1]*d[1])/length))
            nearest.append((p[0]+t*d[0], p[1]+t*d[1]))
    min_norm2 = F(0) if all(h >= 0 for _, _, h in planes) else min(map(norm2, nearest))
    if min_norm2 > 1:
        return []
    critical = [(F(0), F(0)), (F(1), F(1))]
    for x, y in vs:
        if x or y:  # Origin is outside rho_L>0; it has no angular boundary.
            critical += point_turns(a, a.r(x), a.r(y))
    edges = list(zip(vs, vs[1:]+vs[:1]))
    for (x, y), (u, v) in edges:
        dx, dy = u-x, v-y
        length = dx*dx+dy*dy
        if not length:
            continue
        dot = x*dx+y*dy
        for radius in (rho, F(1)):
            disc = radius*radius*length-(x*dy-y*dx)**2
            if disc < 0:
                continue
            root = a.iv.sqrt(a.r(disc))
            for sign in (-1, 1):
                t = (-a.r(dot)+sign*root)/a.r(length)
                tl, th = a.bounds(t)
                if th < 0 or tl > 1:
                    continue
                critical += point_turns(a, a.r(x)+t*a.r(dx), a.r(y)+t*a.r(dy))
    critical = merge(critical)
    possible = []
    # Include critical boxes conservatively, but reject definitely radial-empty.
    for lo, hi in critical:
        theta = a.interval(lo, hi)*2*a.iv.pi
        if radial_bounds(a, theta, planes, rho)[0] != 'OUT':
            possible.append((lo, hi))
    for (_, left), (right, _) in zip(critical, critical[1:]):
        if left < right:
            theta = a.r((left+right)/2)*2*a.iv.pi
            if radial_bounds(a, theta, planes, rho)[0] != 'OUT':
                possible.append((left, right))
    return merge(possible)


def residual_region(a, cis, rho, bias, prior, psi, geometry='ideal'):
    regions = angular_region(a, cis, rho, geometry)
    pl, ph = psi
    if not 0 <= pl <= ph <= 1 or not 0 <= bias < 1 or not 0 < prior < 1:
        raise ValueError('Registered turn-center / residual domain exceeded')
    pieces = []
    for lo, hi in regions:
        for j in (-1, 0, 1):
            r = (a.interval(lo+j-ph, hi+j-pl)*2*a.iv.pi
                 +a.interval(-bias, bias))
            rl, rh = a.bounds(r)
            left, right = max(-prior, rl), min(prior, rh)
            if left <= right:
                pieces.append((left, right))
    return merge(pieces)


def width_of(regions):
    if not regions:
        return None
    return regions[-1][1]-regions[0][0]


def q_region(a, ci, rho, bias, prior):
    """Single quadrature CI; general bounded mean input, not Q0 n optimization."""
    if a.bounds(a.r(prior+bias)-a.iv.pi/2)[1] >= 0:
        raise ValueError('Multiple sine branches: unqualified Q inverse')
    lo, hi = ci.outer
    if hi < -1 or lo > 1:
        return []
    # Q0 inverse_endpoint/asin formula, adapted to exact rational endpoints and
    # a private interval context (no mp.mp point conversion/global precision).
    def inverse(v, lower):
        denominator = rho if (v < 0 if lower else v >= 0) else F(1)
        vl, vh = a.bounds(a.r(v)/a.r(denominator))
        vl, vh = max(F(-1), vl), min(F(1), vh)
        def at(x):
            if x == 1:
                return a.iv.pi/2
            if x == -1:
                return -a.iv.pi/2
            z = a.r(x)
            return a.iv.atan2(z, a.iv.sqrt(1-z*z))
        return a.bounds(at(vl))[0] if lower else a.bounds(at(vh))[1]
    left = inverse(lo, True)-bias
    right = inverse(hi, False)+bias
    return [(max(-prior, left), min(prior, right))] if max(-prior, left) <= min(prior, right) else []
