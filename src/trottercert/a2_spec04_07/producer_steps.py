"""Generator-only exact rational predicates and endpoint proposals."""

from fractions import Fraction as F

from ..a2_spec01_03.common import integer, rat, read_rat, require

ROUTES = ("RF0", "RF1", "EW2", "CENTERED_CBAR2")


def predicate(route, h, lam, width, ksos, cbar, d, alpha):
    require(route in ROUTES, "INVALID_TASK_OR_CERTIFICATE", "route", "Unknown guarantee route")
    if h == 0:
        return "ZERO_ANCHOR", False
    require(h > 0, "INVALID_TASK_OR_CERTIFICATE", "h", "Positive step required")
    if route == "CENTERED_CBAR2":
        if h * lam / 2 >= 1:
            return "DOMAIN_EDGE_OPEN", False
        b = h * h * cbar / (1 - h * lam / 2)
        probability = min(F(1), b * b / (d * d))
    else:
        x = h * lam
        if x >= F(1, 4):
            return "DOMAIN_EDGE_OPEN", False
        eta0 = 32 * x * x / (1 - 16 * x * x)
        eta1 = h * h * ksos + 512 * x ** 4 / (1 - 16 * x * x)
        eta = eta0 if route == "RF0" else eta1 if route == "RF1" else min(eta0, eta1)
        if eta >= 1:
            return "ETA_DOMAIN_EXCLUDED", False
        probability = min(F(1), eta * eta * width * (width + d) / ((1 - eta) * d * d))
    return ("ACCEPT" if probability <= alpha else "REJECT_PROBABILITY"), probability == alpha


def _bracket(route, lam, width, ksos, cbar, d, alpha, epsilon, budget):
    edge = 2 / lam if route == "CENTERED_CBAR2" else 1 / (4 * lam)
    low, high, probes = F(0), edge, []
    exact = False
    for _ in range(budget):
        if high - low <= epsilon:
            break
        h = (low + high) / 2
        status, equality = predicate(route, h, lam, width, ksos, cbar, d, alpha)
        probes.append({"h": rat(h), "status": status, "boundary_equality": equality})
        if status == "ACCEPT":
            low = h
            if equality:
                high, exact = h, True
                break
        else:
            high = h
    status_low, _ = predicate(route, low, lam, width, ksos, cbar, d, alpha)
    status_high, _ = predicate(route, high, lam, width, ksos, cbar, d, alpha)
    resolved = high - low <= epsilon
    return {"route_id": route, "unit": "INV_HA", "domain_edge": rat(edge),
            "domain": "h>0;h*Lambda/2<1" if route == "CENTERED_CBAR2" else "h>0;h*Lambda<1/4;eta<1",
            "endpoint_bracket": {"lo": rat(low), "hi": rat(high)},
            "endpoint_attained": True if exact else None,
            "endpoint_kind": "EXACT_RATIONAL_ROOT" if exact else "RATIONAL_ENCLOSURE",
            "accepted_probe": rat(low) if status_low == "ACCEPT" else None,
            "lower_anchor_status": status_low,
            "upper_probe_status": status_high,
            "probes": probes, "bisections": str(len(probes)),
            "verdict": "ENCLOSED" if resolved else "UNRESOLVED_ARITHMETIC"}


def compare(lam, width, ksos, cbar, policy):
    require(all(v >= 0 for v in (lam, width, ksos, cbar)),
            "INVALID_TASK_OR_CERTIFICATE", "bounds", "Nonnegative certified inputs required")
    if policy is None:
        return {"status": "NOT_EVALUATED_PENDING_POLICY", "policy": None, "routes": [],
                "step_comparison": "NOT_EVALUATED"}
    d, alpha, epsilon = (read_rat(policy[k]) for k in ("d", "alpha_spec", "root_width"))
    budget = integer(policy["max_bisections"], natural=True)
    require(d > 0 and 0 < alpha < 1 and epsilon > 0 and budget > 0,
            "INVALID_TASK_OR_CERTIFICATE", "policy", "Invalid exact synthetic task")
    if width == 0 or lam == 0:
        return {"status": "TRIVIAL_KNOWN_ENERGY", "policy": policy, "routes": [],
                "step_comparison": "NOT_A_RESOURCE_IMPROVEMENT_EXAMPLE"}
    routes = [_bracket(r, lam, width, ksos, cbar, d, alpha, epsilon, budget) for r in ROUTES[:3]]
    if cbar == 0:
        routes.append({"route_id": "CENTERED_CBAR2", "unit": "INV_HA",
                       "domain_edge": rat(2 / lam), "domain_supremum": rat(2 / lam),
                       "endpoint_attained": False, "verdict": "DOMAIN_EDGE_OPEN"})
        return {"status": "COMMUTATOR_ZERO_EXACT_S2", "policy": policy, "routes": routes,
                "exact_counter_route": {"route_id": "COMMUTATOR_ZERO_EXACT_S2", "step_domain": "ALL_REAL_H",
                                        "QPE_alias_readout": "NOT_ESTABLISHED"},
                "step_comparison": "NO_RELATIVE_ADVANTAGE_OVER_EXACT_COUNTER"}
    routes.append(_bracket(ROUTES[3], lam, width, ksos, cbar, d, alpha, epsilon, budget))
    rel, counter = routes[2], routes[3]
    rl, rh = (read_rat(rel["endpoint_bracket"][k]) for k in ("lo", "hi"))
    cl, ch = (read_rat(counter["endpoint_bracket"][k]) for k in ("lo", "hi"))
    if rl > ch:
        order = "RELATIVE_LARGER_SUFFICIENT_STEP_AGAINST_DECLARED_COUNTER"
    elif cl > rh:
        order = "COUNTER_LARGER_SUFFICIENT_STEP"
    elif rl == rh == cl == ch:
        order = "EQUAL_SUFFICIENT_STEP"
    else:
        order = "ORDER_UNRESOLVED"
    return {"status": "ENCLOSED" if all(r["verdict"] == "ENCLOSED" for r in routes) else "UNRESOLVED_ARITHMETIC",
            "policy": policy, "routes": routes, "step_comparison": order}
