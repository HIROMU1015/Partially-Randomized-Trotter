"""Formula-only frozen five-stage domain; no Hamiltonian or science I/O."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import mpmath as mp
from scipy.integrate import quad
from scipy.optimize import brentq

DPS = 80
ARC_TOL = 1e-11
ORDER_TOL = 1e-12


@dataclass(frozen=True)
class ExactTime:
    """A rational affine expression in a point's one algebraic basis value.

    Named points use their cubic root; generic points use sqrt(g(s)) with
    rational s. Equality/zero testing never uses a floating-point cutoff.
    """
    constant: Fraction = Fraction(0)
    root: Fraction = Fraction(0)

    def __add__(self, other: ExactTime) -> ExactTime:
        return ExactTime(self.constant + other.constant, self.root + other.root)

    def __neg__(self) -> ExactTime:
        return ExactTime(-self.constant, -self.root)

    def __mul__(self, scale: Fraction | int) -> ExactTime:
        return ExactTime(self.constant * scale, self.root * scale)

    def value(self, basis: str) -> mp.mpf:
        return (mp.mpf(self.constant.numerator) / self.constant.denominator
                + mp.mpf(self.root.numerator) / self.root.denominator * mp.mpf(basis))

    def record(self) -> list[str]:
        return [str(self.constant), str(self.root)]

    @property
    def zero(self) -> bool:
        return self.constant == 0 and self.root == 0


@dataclass(frozen=True)
class Point:
    label: str
    component: int | None
    arc: float | None
    definition: str
    basis: str
    weights: tuple[ExactTime, ...]

    @property
    def identity(self) -> str:
        payload = [self.definition, self.basis, [w.record() for w in self.weights]]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def values(self) -> tuple[float, ...]:
        with mp.workdps(DPS):
            return tuple(float(w.value(self.basis)) for w in self.weights)

    def record(self) -> dict:
        values = self.values()
        with mp.workdps(DPS):
            exact_values = [mp.nstr(w.value(self.basis), 60) for w in self.weights]
        return dict(label=self.label, component=self.component, arc=self.arc,
                    definition=self.definition, basis=self.basis,
                    weights=[w.record() for w in self.weights], values=exact_values,
                    binary64_hex=[v.hex() for v in values], identity=self.identity,
                    eta1=abs(math.fsum(values) - 1), eta3=abs(math.fsum(v**3 for v in values)))

    @classmethod
    def from_record(cls, row: dict) -> Point:
        point = cls(row["label"], row["component"], row["arc"], row["definition"],
                    row["basis"], tuple(ExactTime(Fraction(a), Fraction(b))
                                         for a, b in row["weights"]))
        if point.identity != row["identity"]:
            raise ValueError("Frozen coefficient identity mismatch")
        return point


class Domain:
    """Four signed charts form three connected components, all included.

    The positive component is oriented from d<0 at s=3/2 through Suzuki
    to d>0 at s=3/2. Negative components are ordered by d at s=-1/2.
    Positive-chart integration uses r=sqrt(s-s_plus) to remove the cusp.
    """
    def __init__(self):
        with mp.workdps(DPS):
            self.sp_text = mp.nstr(2 / (4 - mp.root(4, 3)), 70)
            self.sn_text = mp.nstr(mp.findroot(lambda s: 6*s**3-24*s*s-18*s-1,
                                             (-mp.mpf('.07'), -mp.mpf('.05'))), 70)
        self.sp, self.sn = float(self.sp_text), float(self.sn_text)
        self.rmax = math.sqrt(1.5 - self.sp)
        self.integral_errors: list[float] = []
        self.lneg = self._integral(self._neg_speed, -.5, self.sn)
        self.lpos_half = self._integral(self._pos_speed, 0, self.rmax)
        self.lengths = (self.lneg, self.lneg, 2*self.lpos_half)

    @staticmethod
    def g(s):
        return 5*s*s - 8*s + 4 - 2/(3*s)

    @staticmethod
    def gp(s):
        return 10*s - 8 + 2/(3*s*s)

    def _neg_speed(self, s):
        return math.sqrt(5 + self.gp(s)**2 / (4*self.g(s)))

    def _pos_speed(self, r):
        s = self.sp + r*r
        # g(s)-g(sp) as a divided difference, avoiding subtraction at the cusp.
        divided = 5*(s+self.sp)-8+2/(3*s*self.sp)
        dd = self.gp(s) / math.sqrt(divided)
        return math.sqrt(20*r*r + dd*dd)

    def _integral(self, fn, lo, hi):
        if lo == hi:
            return 0.0
        value, error = quad(fn, lo, hi, epsabs=ARC_TOL, epsrel=ARC_TOL, limit=200)
        if error > 1e-9:
            raise ValueError("Arc integration uncertainty exceeds frozen tolerance")
        self.integral_errors.append(error)
        return value

    def point(self, component: int, arc: float, label: str = "refinement") -> Point:
        if component not in (0, 1, 2) or not 0 <= arc <= self.lengths[component]:
            raise ValueError("Point outside frozen domain")
        if component < 2:
            if arc == self.lengths[component]:
                cap = ExactTime(Fraction(-2))
                other = ExactTime(Fraction(2), Fraction(1))
                a, b = (cap, other) if component == 0 else (other, cap)
                c = ExactTime(Fraction(1), Fraction(-2))
                return Point(label, component, arc, 'negative cap endpoint; s solves 6s^3-24s^2-18s-1=0',
                             self.sn_text, (a, b, c, b, a))
            s = brentq(lambda x: self._integral(self._neg_speed, -.5, x)-arc,
                       -.5, self.sn, xtol=5e-15)
            sign = -1 if component == 0 else 1
        else:
            if abs(arc-self.lpos_half) < 1e-14:
                return self.suzuki()
            sign = -1 if arc < self.lpos_half else 1
            distance = abs(arc-self.lpos_half)
            r = brentq(lambda x: self._integral(self._pos_speed, 0, x)-distance,
                       0, self.rmax, xtol=5e-15)
            s = self.sp+r*r
        # Interior points have rational s; d is an algebraic radical, not
        # a rounded pair subsequently projected onto an order condition.
        ss = Fraction(repr(s))
        with mp.workdps(DPS):
            ms = mp.mpf(ss.numerator)/ss.denominator
            basis = mp.nstr(sign*mp.sqrt(self.g(ms)), 70)
        a = ExactTime(ss/2, Fraction(1, 2))
        b = ExactTime(ss/2, Fraction(-1, 2))
        c = ExactTime(1-2*ss)
        p = Point(label, component, arc, f"s={ss}; d=sign({sign})sqrt(g(s))",
                  basis, (a, b, c, b, a))
        self.screen(p)
        return p

    def suzuki(self):
        a = ExactTime(Fraction(0), Fraction(1, 2))
        c = ExactTime(Fraction(1), Fraction(-2))
        return Point("Suzuki5", 2, self.lpos_half, "s=2/(4-cuberoot(4)); d=0",
                     self.sp_text, (a, a, c, a, a))

    def yoshida(self):
        with mp.workdps(DPS):
            root = mp.nstr(1/(2-mp.root(2, 3)), 70)
        s = float(root)
        r = math.sqrt(s-self.sp)
        arc = self.lpos_half + self._integral(self._pos_speed, 0, r)
        a, b, c = ExactTime(root=Fraction(1)), ExactTime(), ExactTime(Fraction(1), Fraction(-2))
        return Point("Yoshida3_zero_embedded", 2, arc, "a=1/(2-cuberoot(2)); b=0",
                     root, (a, b, c, b, a))

    @staticmethod
    def screen(point):
        w = point.values()
        if max(abs(x) for x in w) > 2+4e-15 or abs(math.fsum(w)-1) > ORDER_TOL or abs(math.fsum(x**3 for x in w)) > ORDER_TOL:
            raise ValueError("Coefficient fails binary64 cap/order screen")

    def initial_points(self):
        counts = [1, 1, 1]
        amounts = [11*l/sum(self.lengths) for l in self.lengths]
        floors = [math.floor(x) for x in amounts]
        counts = [x+y for x, y in zip(counts, floors)]
        remainder = 14-sum(counts)
        for i in sorted(range(3), key=lambda i: (-(amounts[i]-floors[i]), i))[:remainder]:
            counts[i] += 1
        points = [self.suzuki(), self.yoshida()]
        for component, count in enumerate(counts):
            for i in range(count):
                position = (i+.5)/count
                # A midpoint coincident with Suzuki is replaced by the quarter
                # point in that SAME stratum, fixed before any objective exists.
                if component == 2 and position == .5:
                    position = (i+.25)/count
                points.append(self.point(component, self.lengths[component]*position,
                                         f"stratum_{component}_{i}"))
        if len(points) != 16 or len({p.identity for p in points}) != 16:
            raise ValueError("Initial-point budget/identity failure")
        return counts, points

    def manifest(self):
        counts, points = self.initial_points()
        with mp.workdps(DPS):
            endpoints = []
            for sign in (-1, 1):
                for chart, s_text, d in (
                    ('negative', '-0.5', sign*mp.sqrt(mp.mpf(127)/12)),
                    ('negative', self.sn_text, sign*(mp.mpf(self.sn_text)+4)),
                    ('positive', self.sp_text, mp.mpf(0)),
                    ('positive', '1.5', sign*mp.sqrt(101)/6)):
                    s = mp.mpf(s_text)
                    endpoints.append(dict(chart=chart, sign=sign, s=s_text, d=mp.nstr(d, 60),
                        weights=[mp.nstr(x, 60) for x in ((s+d)/2, (s-d)/2, 1-2*s, (s-d)/2, (s+d)/2)]))
        return dict(schema="bf1_formula_domain_v1", science_input_opened=False,
                    decimal_precision=DPS, order_screen=ORDER_TOL, coefficient_cap=2,
                    chart_count=4, connected_component_count=3,
                    endpoints=dict(negative_s=["-1/2", self.sn_text], positive_s=[self.sp_text, "3/2"],
                                   negative_cap_polynomial="6s^3-24s^2-18s-1=0",
                                   negative_signs=[-1, 1], positive_signs=[-1, 1]),
                    arclength=dict(metric="Euclidean norm of dw in R^5", lengths=self.lengths,
                                   epsabs=ARC_TOL, epsrel=ARC_TOL,
                                   maximum_reported_quadrature_error=max(self.integral_errors)),
                    stratum_counts=counts, initial_points=[p.record() for p in points],
                    branch_endpoints=endpoints,
                    initial_point_count=16, refinement_count_per_arm=16,
                    endpoint_sampling="Unscored endpoints bound refinement intervals with objective +infinity; evaluating a midpoint consumes the same arm budget.",
                    exact_time_representation="rational affine; exact zero only; 80-digit lowering")


def fixed_references(domain):
    """Published fixed points, not additional searched families."""
    native = Point("native_S2", None, None, "one S2 stage", "0", (ExactTime(Fraction(1)),))
    # Morales arXiv:2210.15817v3, Table I LEFT column, Eq.16:
    # chronological list w10,...,w1,w0,w1,...,w10.
    published = [
        ".59358060400850625863514059265224", "-.46916012347004197296293264921328",
        ".2743566425898467907228242878146", ".17193879484656773059919074965377",
        ".23439874482541384415430578747541", "-.48616424480326193899617759997914",
        ".49617367388114660354871757044906", "-.32660218948439130114501815323814",
        ".23271679349369857679445410270557", ".098249557414708533273471906180643"]
    side = [Fraction(x) for x in published]
    weights = [ExactTime(x) for x in reversed(side)] + [ExactTime(1-2*sum(side))] + [ExactTime(x) for x in side]
    morales = Point("Morales_v3_TableI_left_21stage", None, None,
                    "arXiv:2210.15817v3 Table I left; published decimal rationals; center=1-2sum", "0", tuple(weights))
    return [native, domain.yoshida(), domain.suzuki(), morales]
