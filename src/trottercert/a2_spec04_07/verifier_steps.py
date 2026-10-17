"""Independent checker predicates via denominator-cleared inequalities.

Never imports producer_steps. No float roots, leading-only approximations or
automatic precision increase. Returned records are recomputed, not trusted flags.
"""

from fractions import Fraction as F

from ..a2_spec01_03.common import integer, rat, read_rat, require


def predicate(route, h, lam, width, ksos, cbar, d, alpha):
    require(route in ("RF0", "RF1", "EW2", "CENTERED_CBAR2"),
            "INVALID_TASK_OR_CERTIFICATE", "route", "Unknown guarantee route")
    if h == 0:
        return "ZERO_ANCHOR", False
    require(h > 0, "INVALID_TASK_OR_CERTIFICATE", "h", "Negative query")
    if route == "CENTERED_CBAR2":
        denominator = 1 - h * lam / 2
        if denominator <= 0:
            return "DOMAIN_EDGE_OPEN", False
        lhs, rhs = h ** 4 * cbar ** 2, alpha * d ** 2 * denominator ** 2
    else:
        x = h * lam
        if 4 * x >= 1:
            return "DOMAIN_EDGE_OPEN", False
        denominator = 1 - 16 * x ** 2
        n0 = 32 * x ** 2
        n1 = h ** 2 * ksos * denominator + 512 * x ** 4
        numerator = n0 if route == "RF0" else n1 if route == "RF1" else min(n0, n1)
        if numerator >= denominator:
            return "ETA_DOMAIN_EXCLUDED", False
        lhs = numerator ** 2 * width * (width + d)
        rhs = alpha * (denominator - numerator) * denominator * d ** 2
    return ("ACCEPT" if lhs <= rhs else "REJECT_PROBABILITY"), lhs == rhs


def _reconstruct_route(route, lam, width, ksos, cbar, d, alpha, epsilon, budget):
    boundary = F(2) / lam if route == "CENTERED_CBAR2" else F(1, 4) / lam
    left, right, transcript = F(0), boundary, []
    found_root = False
    while right - left > epsilon and len(transcript) < budget:
        midpoint = (left + right) * F(1, 2)
        outcome, equal = predicate(route, midpoint, lam, width, ksos, cbar, d, alpha)
        transcript.append({"h": rat(midpoint), "status": outcome, "boundary_equality": equal})
        if outcome != "ACCEPT":
            right = midpoint
        elif equal:
            left = right = midpoint
            found_root = True
            break
        else:
            left = midpoint
    ls = predicate(route, left, lam, width, ksos, cbar, d, alpha)[0]
    rs = predicate(route, right, lam, width, ksos, cbar, d, alpha)[0]
    return {"route_id": route, "unit": "INV_HA", "domain_edge": rat(boundary),
            "domain": "h>0;h*Lambda/2<1" if route == "CENTERED_CBAR2" else "h>0;h*Lambda<1/4;eta<1",
            "endpoint_bracket": {"lo": rat(left), "hi": rat(right)},
            "endpoint_attained": True if found_root else None,
            "endpoint_kind": "EXACT_RATIONAL_ROOT" if found_root else "RATIONAL_ENCLOSURE",
            "accepted_probe": rat(left) if ls == "ACCEPT" else None,
            "lower_anchor_status": ls, "upper_probe_status": rs,
            "probes": transcript, "bisections": str(len(transcript)),
            "verdict": "ENCLOSED" if right - left <= epsilon else "UNRESOLVED_ARITHMETIC"}


def compare(lam, width, ksos, cbar, policy):
    require(min(lam, width, ksos, cbar) >= 0,
            "INVALID_TASK_OR_CERTIFICATE", "bounds", "Negative certificate quantity")
    if policy is None:
        return {"status": "NOT_EVALUATED_PENDING_POLICY", "policy": None, "routes": [],
                "step_comparison": "NOT_EVALUATED"}
    d, alpha, epsilon = [read_rat(policy[k]) for k in ("d", "alpha_spec", "root_width")]
    budget = integer(policy["max_bisections"], natural=True)
    require(d > 0 and 0 < alpha < 1 and epsilon > 0 and budget > 0,
            "INVALID_TASK_OR_CERTIFICATE", "policy", "Invalid exact synthetic task")
    if lam == 0 or width == 0:
        return {"status": "TRIVIAL_KNOWN_ENERGY", "policy": policy, "routes": [],
                "step_comparison": "NOT_A_RESOURCE_IMPROVEMENT_EXAMPLE"}
    records = [_reconstruct_route(route, lam, width, ksos, cbar, d, alpha, epsilon, budget)
               for route in ("RF0", "RF1", "EW2")]
    if not cbar:
        records.append({"route_id": "CENTERED_CBAR2", "unit": "INV_HA", "domain_edge": rat(F(2) / lam),
                        "domain_supremum": rat(F(2) / lam), "endpoint_attained": False, "verdict": "DOMAIN_EDGE_OPEN"})
        return {"status": "COMMUTATOR_ZERO_EXACT_S2", "policy": policy, "routes": records,
                "exact_counter_route": {"route_id": "COMMUTATOR_ZERO_EXACT_S2", "step_domain": "ALL_REAL_H",
                                        "QPE_alias_readout": "NOT_ESTABLISHED"},
                "step_comparison": "NO_RELATIVE_ADVANTAGE_OVER_EXACT_COUNTER"}
    records.append(_reconstruct_route("CENTERED_CBAR2", lam, width, ksos, cbar, d, alpha, epsilon, budget))
    rlo, rhi = [read_rat(records[2]["endpoint_bracket"][side]) for side in ("lo", "hi")]
    clo, chi = [read_rat(records[3]["endpoint_bracket"][side]) for side in ("lo", "hi")]
    if clo > rhi:
        comparison = "COUNTER_LARGER_SUFFICIENT_STEP"
    elif rlo > chi:
        comparison = "RELATIVE_LARGER_SUFFICIENT_STEP_AGAINST_DECLARED_COUNTER"
    elif rlo == rhi and clo == chi and rlo == clo:
        comparison = "EQUAL_SUFFICIENT_STEP"
    else:
        comparison = "ORDER_UNRESOLVED"
    return {"status": "ENCLOSED" if all(r["verdict"] == "ENCLOSED" for r in records) else "UNRESOLVED_ARITHMETIC",
            "policy": policy, "routes": records, "step_comparison": comparison}
