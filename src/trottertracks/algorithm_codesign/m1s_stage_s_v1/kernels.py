"""Pure artificial-testable scalar kernels. No saved-data I/O or execution grant.

cap is a closed rational enclosure of the *open* mathematical endpoint.
The one-sided integer design uses floor(S(cap))+1, never ceil(S(cap)).
"""
from fractions import Fraction as F
from .arithmetic import interval_text


def open_cap_integer(pair):
    """Closed enclosure of lim from below ceil(S), with S strictly decreasing."""
    lo, hi = pair
    if not 0 <= lo <= hi:
        raise ValueError("positive ordered S enclosure required")
    lower = lo.numerator // lo.denominator + 1
    upper = hi.numerator // hi.denominator + 1
    return {
        "interval": [lower, upper],
        "resolved": lower == upper,
        "value": lower if lower == upper else None,
        "rule": "floor(S_at_cap)+1_for_left_limit_of_strictly_decreasing_S",
    }


def classify_integer(shared, comparator):
    if not shared["resolved"] or not comparator["resolved"]:
        return "ARITHMETIC_UNRESOLVED"
    if shared["value"] < comparator["value"]:
        return "CAP_EDGE_STRICT_INTEGER_WITNESS"
    return "CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED"


def margin_at_cap(A, bias, cap, bounds):
    """sin is increasing on this conservatively checked domain, cap<3/2<pi/2."""
    if not (0 < A <= 1 and 0 <= bias < cap[0] <= cap[1] < F(3, 2)):
        raise ValueError("positive cap margin/domain not established")
    lower = bounds.sin_positive(cap[0] - bias)
    upper = bounds.sin_positive(cap[1] - bias)
    pair = bounds.pack((A * lower[0], A * upper[1]))
    if pair[0] <= 0:
        return None
    return pair


def sufficient_at_cap(margin, K, bounds):
    if not (0 < margin[0] <= margin[1] and 0 < K[0] <= K[1]):
        raise ValueError("positive margin and K enclosures required")
    return bounds.pack((K[0] / margin[1]**2, K[1] / margin[0]**2))


def compare_cap(a, u, c, v, shared_margin, comparator_margin, K, bounds):
    """Theorem only for a>c>0 and u>v>=0, never silently swap labels."""
    if not (1 >= a > c > 0 and u > v >= 0):
        raise ValueError("direct tradeoff theorem assumptions not met")
    ms, mc = shared_margin, comparator_margin
    if ms is None or mc is None:
        return {"new_stage_s_status": "ARITHMETIC_UNRESOLVED",
                "reason": "positive margin enclosure not resolved"}
    ratio = bounds.pack(((ms[0] / mc[1])**2, (ms[1] / mc[0])**2))
    ss = sufficient_at_cap(ms, K, bounds)
    sc = sufficient_at_cap(mc, K, bounds)
    ns, nc = open_cap_integer(ss), open_cap_integer(sc)
    status = classify_integer(ns, nc)
    # Eligible saved roots imply R(cap)>1. Ambiguous interval remains unresolved.
    if ratio[1] <= 1:
        raise ValueError("eligible saved root inconsistent with cap margin ratio")
    if ratio[0] <= 1:
        status = "ARITHMETIC_UNRESOLVED"
    return {
        "new_stage_s_status": status,
        "R_cont_cap_interval": interval_text(ratio),
        "R_cont_cap_role": "UNATTAINED_SUPREMUM_NOT_NATIVE_COST_OR_RPE_SUCCESS",
        "S_shared_at_cap_interval": interval_text(ss),
        "S_comparator_at_cap_interval": interval_text(sc),
        "n_shared_cap_left": ns, "n_comparator_cap_left": nc,
        "N_shared_cap_left_interval": [2 * x for x in ns["interval"]],
        "N_comparator_cap_left_interval": [2 * x for x in nc["interval"]],
        "integer_relation": ("SHARED_STRICTLY_LOWER" if status == "CAP_EDGE_STRICT_INTEGER_WITNESS"
                             else "TIE_OR_NOT_LOWER_AT_CAP_ONLY" if status == "CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED"
                             else "UNRESOLVED"),
        "nonempty_open_region_existence_witness": status == "CAP_EDGE_STRICT_INTEGER_WITNESS",
        "all_gamma_impossibility_established": False,
    }
