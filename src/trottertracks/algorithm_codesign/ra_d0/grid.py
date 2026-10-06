"""Static shots and query recipe. Does not optimize B2/B3 or classify RA-D0."""
from fractions import Fraction as F
from .exact import E, DELTA_NUM, ceil, sqrt_interval, log_interval


def pareto_upper(cost_minima, baseline_vectors):
    """Full vector domination requires every resource coordinate to be covered."""
    if any(v <= 0 for v in cost_minima.values()):
        raise ValueError("UNCERTIFIED_UPPER_BOUND: zero minimum cannot simply be omitted")
    bounds = [max(F(b[k])/(2*v) for k, v in cost_minima.items()) for b in baseline_vectors]
    return ceil(min(bounds))


def coverage(nmin, nmax, anchors, ratio=F(201, 200)):
    if nmin < 1 or nmax < nmin or ratio <= 1:
        raise ValueError("invalid shot domain")
    points = [nmin]
    while points[-1] < nmax:
        points.append(min(nmax, ceil(ratio*points[-1])))
    points = sorted(set(points) | set(anchors))
    inside = [n for n in points if nmin <= n <= nmax]
    # For integer shots between adjacent points, the worst upward rounding
    # ratio is right/(left+1), not right/left and not the nominal recurrence r.
    factor = max([F(1)] + [F(b, a+1) for a, b in zip(inside, inside[1:]) if b > a+1])
    return points, factor


def build_grid(table):
    ell_lo, ell_hi = log_interval(10560)
    out = {}
    for xs, data in table["tables"].items():
        target = [F(v) for v in data["target"]]
        # Enlarge h_max upward and use ell downward: a safe necessary lower
        # bound. Ceil of the lower endpoint cannot exclude an admissible n.
        # The implemented mean tolerance permits |Dq-yt|_1 <= y*delta_num.
        # Use the resulting slightly larger y bound even though it gives the
        # same integer lower boundaries as the ideal formula in this table.
        hmax_hi = E*sqrt_interval(2)[1]/(sum(target)-DELTA_NUM)
        lower = ell_lo*(2+F(4, 3)*hmax_hi)/hmax_hi**2
        nmin = ceil(lower)
        minima = {k: min(F(c["costs"][k]) for c in data["columns"])+(F(5, 2) if k == "1Q" else 0)
                  for k in ("T", "CX", "1Q")}
        baselines = [{"T": p["original_finite_confidence"]["G_T"],
                      "CX": p["original_finite_confidence"]["G_CX"],
                      "1Q": p["original_finite_confidence"]["G_1Q_with_Hadamard_preparation_readout"]}
                     for p in data["B0_saved_profiles"]]
        anchors = sorted({p["original_finite_confidence"]["sufficient_shots_per_axis"]
                          for p in data["B0_saved_profiles"]})
        nmax = pareto_upper(minima, baselines)
        points, factor = coverage(nmin, nmax, anchors)
        out[xs] = {"n_min": nmin, "n_max": nmax, "anchor_shots": anchors,
                   "cost_minima": {k: str(v) for k, v in minima.items()},
                   "points": [{"n": n, "tag": "PRIMARY_ANCHOR" if n in anchors else "COVERAGE_GRID",
                               "in_pareto_bound": nmin <= n <= nmax} for n in points],
                   "integer_coverage_factor_exact": str(factor),
                   "nominal_recurrence_ratio": "201/200", "factor_1p005_certified": factor <= F(201, 200),
                   "zero_minimum_omission_used": False,
                   "full_front_or_cap_preservation_claim": False}
    return {"schema": "ra_d0_result_prior_grid_v1", "ell_interval": list(map(str, (ell_lo, ell_hi))),
            "optimization_calls": 0, "grids": out,
            "query_recipe": {"objectives": ["T", "CX", "1Q"],
                             "budget_sources": ["same_n_feasible_B0", "B2_single_resource_minima"],
                             "budget_values": "DEFERRED_UNTIL_SEPARATE_AUTHORIZATION",
                             "B3_may_set_budgets": False,
                             "B2_infeasibility_requires_certificate": True,
                             "strict_rule": "B3 certified primal upper < B2 certified dual lower",
                             "per_point_max_budget_values_per_coordinate": 10,
                             "per_point_max_paired_queries": 300}}
