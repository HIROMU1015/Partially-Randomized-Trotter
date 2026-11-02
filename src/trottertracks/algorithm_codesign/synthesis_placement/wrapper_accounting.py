"""Exact rational accounting fixtures for the proposed SP-1 contract.

No saved SP-0.5 data, synthesis, gate construction, or science runner is loaded.
This kernel assumes canonical independent QPD choices conditional on an outer
path. It does not certify channel equality, finite-RTE semantics, or an adapter
from numerical coefficient intervals. Those remain source-review obligations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("use exact int or Fraction inputs")
    return Fraction(value)


@dataclass(frozen=True)
class Gate:
    coefficients: tuple
    t_counts: tuple
    diamond_errors: tuple

    def __post_init__(self):
        for name in ("coefficients", "t_counts", "diamond_errors"):
            object.__setattr__(self, name, tuple(rational(x) for x in getattr(self, name)))
        n = len(self.coefficients)
        if not n or len(self.t_counts) != n or len(self.diamond_errors) != n:
            raise ValueError("branch arrays must have the same nonzero length")
        if sum(self.coefficients) != 1:
            raise ValueError("trace-preserving QPD requires signed coefficients summing to 1")
        if any(x < 0 or x.denominator != 1 for x in self.t_counts):
            raise ValueError("native T counts are nonnegative integers")
        if any(x < 0 or x > 2 for x in self.diamond_errors):
            raise ValueError("invalid primitive channel diamond-distance bound")


@dataclass(frozen=True)
class Path:
    probability: Fraction
    outer_weight: Fraction
    gates: tuple
    fixed_t: int = 0
    fixed_diamond_error: Fraction = Fraction(0)

    def __post_init__(self):
        object.__setattr__(self, "probability", rational(self.probability))
        object.__setattr__(self, "outer_weight", rational(self.outer_weight))
        object.__setattr__(self, "fixed_t", rational(self.fixed_t))
        object.__setattr__(self, "fixed_diamond_error", rational(self.fixed_diamond_error))
        object.__setattr__(self, "gates", tuple(self.gates))
        if self.probability <= 0 or self.probability > 1:
            raise ValueError("only active outer paths with positive probability are allowed")
        if self.fixed_t < 0 or self.fixed_t.denominator != 1:
            raise ValueError("fixed T cost must be a nonnegative integer")
        if self.fixed_diamond_error < 0:
            raise ValueError("negative fixed error")
        if any(not isinstance(g, Gate) for g in self.gates):
            raise TypeError("expected Gate records")


def population_profile(paths):
    """Return E[W²], max|W|, E[C], E[W²C], and a synthesis-bias upper bound.

    W=b(omega)*prod(gamma)*sign. b includes the outer normalization exactly
    once. The path-conditional |W| is constant for canonical QPD sampling.
    Error telescopes through unitary channels before applying signed weights.
    E[C] and E[W²C] retain the same outer-path correlation; they are distinct.
    """
    paths = tuple(paths)
    if not paths or any(not isinstance(p, Path) for p in paths):
        raise ValueError("nonempty Path population required")
    if sum(p.probability for p in paths) != 1:
        raise ValueError("outer probabilities must sum exactly to 1")
    v2 = cost = joint = bias = Fraction(0)
    radius = Fraction(0)
    for path in paths:
        gamma_product = Fraction(1)
        conditional_cost = path.fixed_t
        conditional_error = path.fixed_diamond_error
        for gate in path.gates:
            gamma = sum(abs(g) for g in gate.coefficients)
            gamma_product *= gamma
            conditional_cost += sum(abs(g) * c for g, c in
                                    zip(gate.coefficients, gate.t_counts)) / gamma
            conditional_error += sum(abs(g) * e for g, e in
                                     zip(gate.coefficients, gate.diamond_errors)) / gamma
        magnitude = abs(path.outer_weight) * gamma_product
        v2 += path.probability * magnitude**2
        cost += path.probability * conditional_cost
        joint += path.probability * magnitude**2 * conditional_cost
        bias += path.probability * magnitude * conditional_error
        radius = max(radius, magnitude)
    return {"second_moment": v2, "range": radius, "expected_T_count": cost,
            "joint_weighted_T": joint, "synthesis_bias_upper": bias}


def log_upper(x, terms=256):
    """Rational upper bound for ln(x), x>=1, from the atanh power series.

    Reduce x=2^k*y, 1<=y<2. For ln(y) and ln(2), the atanh-series
    tail is <= 2*z^(2*N+1)/((2*N+1)*(1-z²)), z=(y-1)/(y+1).
    No floating log or precision-dependent round-to-nearest enters a ceiling.
    """
    x = rational(x)
    if x < 1 or type(terms) is not int or not 1 <= terms <= 4096:
        raise ValueError("invalid logarithm domain or series budget")
    exponent = 0
    reduced = x
    while reduced >= 2:
        reduced /= 2
        exponent += 1

    def series_upper(y):
        z = (y - 1) / (y + 1)
        power = z
        result = Fraction(0)
        for k in range(terms):
            result += 2 * power / (2*k + 1)
            power *= z*z
        return result + 2*power / ((2*terms + 1)*(1-z*z))

    return exponent*series_upper(Fraction(2)) + series_upper(reduced)


def axis_budget(profile, epsilon_axis, alpha_axis, shot_cap,
                model_bias=0, numerical_bias=0, log_terms=256):
    """Conservative two-sided Bernstein sufficient shot count, never clipped.

    This uses Var(Z)<=E[W²] and |Z-EZ|<=2*range, with norm-one +/-1
    outcomes. Counts describe a bound-based resource model, not measured shots
    or the minimum necessary number. No ideal/oracle variance is substituted.
    """
    epsilon_axis, alpha_axis = rational(epsilon_axis), rational(alpha_axis)
    model_bias, numerical_bias = rational(model_bias), rational(numerical_bias)
    if epsilon_axis <= 0 or not 0 < alpha_axis < 1:
        raise ValueError("invalid accuracy/confidence")
    if type(shot_cap) is not int or shot_cap < 1 or min(model_bias, numerical_bias) < 0:
        raise ValueError("invalid cap/bias")
    v2, radius, cost, bias = (rational(profile[k]) for k in
                            ("second_moment", "range", "expected_T_count", "synthesis_bias_upper"))
    if min(v2, radius, cost, bias) < 0 or v2 > radius*radius:
        raise ValueError("inconsistent population profile")
    margin = epsilon_axis - model_bias - numerical_bias - bias
    if margin <= 0:
        return {"status": "BIAS_BUDGET_EXHAUSTED", "margin": margin,
                "shots": None, "expected_total_T": None}
    bound = (2*v2 + Fraction(4, 3)*radius*margin) * log_upper(2/alpha_axis, log_terms) / margin**2
    shots = max(1, (bound.numerator + bound.denominator - 1) // bound.denominator)
    return {"status": "SHOT_CAP_EXCEEDED" if shots > shot_cap else "ELIGIBLE",
            "margin": margin, "shots": shots,
            "expected_total_T": shots*cost}


def total_t(axis_records, batch_init_t=0):
    """Return batch initialization plus eligible per-axis shot cost.

    Per-shot preparation belongs to each Path.fixed_t, not batch_init_t.
    A cap-hit or exhausted bias budget must not be silently used in a winner.
    """
    batch_init_t = rational(batch_init_t)
    if batch_init_t < 0 or batch_init_t.denominator != 1:
        raise ValueError("batch initialization cost must be a nonnegative integer")
    records = tuple(axis_records)
    if len(records) != 2:
        raise ValueError("the complex task requires Re and Im records")
    if any(r["status"] != "ELIGIBLE" for r in records):
        raise ValueError("all axis records must be eligible")
    return batch_init_t + sum(rational(r["expected_total_T"]) for r in records)
