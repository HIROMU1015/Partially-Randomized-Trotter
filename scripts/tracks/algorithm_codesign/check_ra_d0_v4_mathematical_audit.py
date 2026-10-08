"""Audit-only rational fixtures; never imports production code or solves saved LPs.

Run with /usr/bin/python3 -B. Only new audit artifacts are written. Rounding
construction and direct interval certificates are separate computations.
"""
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from math import isqrt
from pathlib import Path
import json
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09"
N = 2**60
E, DELTA = F(1, 200), F(1, 10**12)
RESOURCES = ("T", "CX", "1Q")
OVERHEAD = {"T": F(0), "CX": F(0), "1Q": F(5, 2)}
BASE = "d3a7cbb239487ddedf44699378f6c182c1fe5993"


def rational(value):
    if isinstance(value, float):
        raise ValueError("float input is not an exact constructed point")
    return F(value)


def midpoint(group):
    return sum(group["c"], F(0)) / 2


def validate(groups, u, y, target, z=None, denominator=N):
    if denominator < 1 or y < F(1, denominator):
        raise ValueError("invalid inverse normalization")
    if len(groups) != len(u) or not groups:
        raise ValueError("invalid groups")
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("nonunique group index")
    for g, row in zip(groups, u, strict=True):
        if g["c"][0] <= 0 or g["c"][0] > g["c"][1]:
            raise ValueError("positive norm enclosure required")
        if (len(row) != len(g["d"]) or not row or any(rational(v) < 0 for v in row)
                or any(v < 0 for v in g["d"])):
            raise ValueError("invalid precision weights/bias")
        if any(lo > hi for lo, hi in g["D"]):
            raise ValueError("invalid D enclosure")
        if any(c < 0 for r in RESOURCES for c in g["costs"][r]):
            raise ValueError("nonnegative resource coefficients required")
    if sum(midpoint(g)*sum(row) for g, row in zip(groups, u)) != 1:
        raise ValueError("exact normalization required before rounding")
    degree = [sum(g["v"][k]*sum(row) for g, row in zip(groups, u)) for k in range(len(target))]
    if degree != [y*t for t in target]:
        raise ValueError("exact structural degree matching required")
    if z is not None:
        if any(v < 0 for v in z.values()) or sum(z.values()) != y:
            raise ValueError("invalid representation shares")
        if any(sum(row) != z[g["r"]] for g, row in zip(groups, u)):
            raise ValueError("B2 structural membership required")


def lrm(probabilities, total):
    """Exact normalized LRM, with caller's canonical ID order as tie order."""
    probabilities = [rational(v) for v in probabilities]
    if total < 0 or not probabilities or any(v < 0 for v in probabilities) or sum(probabilities) != 1:
        raise ValueError("not a normalized law")
    scaled = [total*p for p in probabilities]
    result = [p.numerator//p.denominator for p in scaled]
    left = total-sum(result)
    order = sorted(range(len(result)), key=lambda j: (-(scaled[j]-result[j]), j))
    for j in order[:left]:
        result[j] += 1
    return result


def decode(groups, u, y, target, z=None, denominator=N):
    validate(groups, u, y, target, z, denominator)
    masses = [midpoint(g)*sum(row) for g, row in zip(groups, u)]
    totals = lrm(masses, denominator)
    counts = []
    for row, mass, total in zip(u, masses, totals):
        counts.append(lrm([v/sum(row) for v in row], total) if mass and total else [0]*len(row))
    yy = y*denominator
    iy = (2*yy.numerator+yy.denominator)//(2*yy.denominator)
    yn = F(iy, denominator)
    zn = None if z is None else {r: v*yn/y for r, v in z.items()}
    return [[F(v, denominator) for v in row] for row in counts], yn, zn, totals, counts


def reserves(groups, target, denominator=N):
    rho = [sum(max(abs(midpoint(g)*lo-v), abs(midpoint(g)*hi-v))
               for (lo, hi), v in zip(g["D"], g["v"])) for g in groups]
    gx = (sum(sum(max(abs(lo), abs(hi)) for lo, hi in g["D"]) for g in groups)
          + sum(abs(v) for v in target)/2)/denominator
    gd = sum(max(g["d"])+sum(g["d"]) for g in groups)/denominator
    gc = {r: sum(max(g["costs"][r])+sum(g["costs"][r]) for g in groups)/denominator for r in RESOURCES}
    return rho, gx, gd, gc


def certificate(groups, q, y, target, n, kappa, caps=None, z=None, denominator=N):
    """Direct endpoint substitution; no Xi/Gamma/continuous acceptance used."""
    sampler = (y > 0 and denominator % y.denominator == 0
               and all(v >= 0 and denominator % v.denominator == 0 for row in q for v in row)
               and sum(sum(row) for row in q) == 1)
    residuals = []
    for k, t in enumerate(target):
        lower = sum(sum(row)*g["D"][k][0] for g, row in zip(groups, q))-y*t
        upper = sum(sum(row)*g["D"][k][1] for g, row in zip(groups, q))-y*t
        residuals.append(max(abs(lower), abs(upper)))
    xi = sum(residuals)
    bias = sum(sum(d*v for d, v in zip(g["d"], row)) for g, row in zip(groups, q))
    costs = {r: 2*n*(sum(sum(c*v for c, v in zip(g["costs"][r], row))
                         for g, row in zip(groups, q))+OVERHEAD[r]) for r in RESOURCES}
    peak = max([0]+[w for g, row in zip(groups, q) for w, v in zip(g["workspace"], row) if v])
    membership_rows = []
    if z is not None:
        for g, row in zip(groups, q):
            residual = max(abs(sum(row)-z[g["r"]]*endpoint) for endpoint in g["c"])
            tau = (3+g["c"][1]/2)/denominator
            membership_rows.append({"group": g["id"], "residual": residual, "tau": tau,
                                    "PASS": residual <= tau})
    member = z is None or (all(v >= 0 for v in z.values()) and sum(z.values()) == y
                           and all(row["PASS"] for row in membership_rows))
    checks = {"sampler": sampler, "membership": member, "mean": xi <= y*DELTA,
              "confidence": E*y-bias-xi >= kappa, "workspace": peak <= 1,
              "caps": all(costs[r] <= b for r, b in (caps or {}).items())}
    return {"checks": checks, "certified": all(checks.values()), "xi": xi, "bias": bias,
            "h": E*y-bias-xi, "resources": costs, "workspace": peak, "membership": membership_rows}


def inner(groups, u, y, target, n, kappa, caps=None, denominator=N):
    rho, gx, gd, gc = reserves(groups, target, denominator)
    xi = sum(r*sum(row) for r, row in zip(rho, u))
    q = [[midpoint(g)*v for v in row] for g, row in zip(groups, u)]
    bias = sum(sum(d*v for d, v in zip(g["d"], row)) for g, row in zip(groups, q))
    gh = E/(2*denominator)+gd+gx
    margins = {"mean": y*DELTA-xi-gx-DELTA/(2*denominator),
               "confidence": E*y-bias-xi-kappa-gh}
    nominal = {r: 2*n*(sum(sum(c*v for c, v in zip(g["costs"][r], row))
                           for g, row in zip(groups, q))+OVERHEAD[r]) for r in RESOURCES}
    margins.update({"cap_"+r: b-nominal[r]-2*n*gc[r] for r, b in (caps or {}).items()})
    return {"PASS": all(v >= 0 for v in margins.values()), "margins": margins,
            "Xi": xi, "Gamma_xi": gx, "Gamma_d": gd, "Gamma_Q": gc, "Gamma_h": gh,
            "nominal_resources": nominal}


def sqrt_up(value):
    scale = 10**100
    k = isqrt(value.numerator*scale**2//value.denominator)
    lo, hi = F(k, scale), F(k+1, scale)
    assert lo*lo <= value <= hi*hi
    return lo if lo*lo == value else hi


SHOTS, ELL = 100_000_000, F(12)  # Artificial, table-independent confidence fixture.
KAPPA = (F(4, 3)*ELL+sqrt_up((F(4, 3)*ELL)**2+8*SHOTS*ELL))/(2*SHOTS)


def group(ident, c=F(1), width=F(0), precisions=3, r="r0", vector=None, dwidth=F(0)):
    c = F(c)
    v = [c] if vector is None else list(vector)
    direction = [a/c for a in v]
    return {"id": ident, "r": r, "c": (c, c+2*width), "v": v,
            "D": [(a-dwidth, a+dwidth) for a in direction],
            "d": [F(3-p, 100_000) if p < 3 else F(1, 100_000) for p in range(precisions)],
            "costs": {"T": [F(2+p) for p in range(precisions)],
                      "CX": [F(2) for p in range(precisions)],
                      "1Q": [F(5+3*p, 2) for p in range(precisions)]},
            "workspace": [1]*precisions}


def b2(groups, theta):
    y = 1/sum(theta[g["r"]]*midpoint(g) for g in groups)
    z = {r: share*y for r, share in theta.items()}
    u = [[z[g["r"]]*F(j+1, sum(range(1, len(g["d"])+1)))
          for j in range(len(g["d"]))] for g in groups]
    target = [sum(g["v"][k] for g in groups if g["r"] == next(iter(theta)))
              for k in range(len(groups[0]["v"]))]
    return u, y, z, target


def scalar_b2(width=F(0), dwidth=F(0), inactive=False, precisions=3):
    groups = [group("r0:a", F(1, 2), width, precisions, "r0", dwidth=dwidth),
              group("r0:b", F(1, 2), width, precisions, "r0", dwidth=dwidth),
              group("r1:a", F(1, 4), width, precisions, "r1", dwidth=dwidth),
              group("r1:b", F(3, 4), width, precisions, "r1", dwidth=dwidth)]
    u, y, z, t = b2(groups, {"r0": F(1) if inactive else F(1, 3), "r1": F(0) if inactive else F(2, 3)})
    return groups, u, y, z, t


def verify_fixture(fixture, denominator=N, require_inner=True):
    groups, u, y, z, target = fixture
    qn, yn, zn, totals, counts = decode(groups, u, y, target, z, denominator)
    q = [[midpoint(g)*v for v in row] for g, row in zip(groups, u)]
    pre = certificate(groups, q, y, target, SHOTS, KAPPA, z=z, denominator=denominator)
    post = certificate(groups, qn, yn, target, SHOTS, KAPPA, z=zn, denominator=denominator)
    inside = inner(groups, u, y, target, SHOTS, KAPPA, denominator=denominator)
    if require_inner:
        assert inside["PASS"], inside["margins"]
        assert post["certified"], post["checks"]
    assert sum(totals) == denominator == sum(sum(row) for row in counts)
    assert all(type(k) is int and k >= 0 for row in counts for k in row)
    assert abs(yn-y) <= F(1, 2*denominator)
    for g, before, after, total, row in zip(groups, q, qn, totals, counts):
        mass = sum(before)
        assert abs(sum(after)-mass) <= F(1, denominator)
        if not mass:
            assert total == 0 and all(k == 0 for k in row)
        else:
            for j in range(len(row)):
                pi = before[j]/mass
                assert abs(row[j]-total*pi) <= 1
    assert pre["xi"] <= inside["Xi"]
    assert post["xi"] <= inside["Xi"]+inside["Gamma_xi"]
    assert abs(post["bias"]-pre["bias"]) <= inside["Gamma_d"]
    assert all(abs(post["resources"][r]-pre["resources"][r]) <= 2*SHOTS*inside["Gamma_Q"][r] for r in RESOURCES)
    assert post["h"] >= E*y-pre["bias"]-inside["Xi"]-inside["Gamma_h"]
    return {"continuous_inner_PASS": inside["PASS"], "decoded_certificate": post,
            "counts": counts, "y": yn, "z": zn, "reserves": inside}


def jsonable(value):
    if isinstance(value, F): return str(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    return value


def static_table_audit():
    table = json.loads((ROOT/"artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json").read_text())
    result = {}
    for xs, data in table["tables"].items():
        target = list(map(F, data["target"]))
        ymax = 2/(sum(target)-DELTA)
        by_id = {c["id"]: c for c in data["columns"]}
        profiles = [p for p in data["B0_saved_profiles"] if p["epsilon"] == "1e-3"]
        norms = {}
        for profile in profiles:
            vs = [F(0)]*len(target)
            for member in profile["memberships"]:
                c = by_id[member["column_id"]]
                lo, hi = map(F, member["ideal_weight_interval"])
                a, b = map(F, c["saved_ideal_ab_exact"])
                assert 0 < lo <= hi and lo*lo <= a*a+b*b <= hi*hi
                vg = [F(0)]*len(target)
                vg[c["degree"]] = a
                if b: vg[c["degree"]+1] = b
                vs = [u+v for u, v in zip(vs, vg)]
                assert all(F(dlo) <= v/hi <= F(dhi) and F(dlo) <= v/lo <= F(dhi)
                           for (dlo, dhi), v in zip(c["D_intervals"], vg))
                assert ymax*(hi-lo)/2 <= F(2, N)
                prototype = c["prototype"]
                if prototype in norms: assert norms[prototype] == (lo, hi)
                norms[prototype] = (lo, hi)
            assert vs == target, (xs, profile["arm"], vs, target)
        groups = []
        for proto, norm in sorted(norms.items()):
            cs = [by_id[proto+":"+ep] for ep in ("1e-3", "1e-4", "1e-6")]
            ideal_keys = ("prototype", "degree", "saved_ideal_ab_exact", "direction_ratio_exact", "D_intervals")
            assert all(all(c[k] == cs[0][k] for k in ideal_keys) for c in cs)
            event_keys = ("source_label", "label_probability", "word", "rotation", "rotation_sign", "phase_i_power", "complement")
            events = [[{k: e[k] for k in event_keys} for e in c["events"]] for c in cs]
            assert events[0] == events[1] == events[2]
            assert all(sum(F(e["label_probability"]) for e in c["events"]) == 1 for c in cs)
            for c in cs:
                assert F(c["d_upper"]) >= 0 and c["workspace_peak"] <= 1
                for r in RESOURCES:
                    assert F(c["costs"][r]) >= 0
                    assert F(c["costs"][r]) == sum(F(e["label_probability"])*F(e["native_cost"][r]) for e in c["events"])
                assert F(c["d_upper"]) == 2*sum(F(e["label_probability"])*F(e["native_cost"]["strict_event_error_upper"]) for e in c["events"])
            column = cs[0]
            vg = [F(0)]*len(target)
            a, b = map(F, column["saved_ideal_ab_exact"])
            vg[column["degree"]] = a
            if b: vg[column["degree"]+1] = b
            D = [tuple(map(F, v)) for v in column["D_intervals"]]
            assert sum(max(abs(lo), abs(hi)) for lo, hi in D) < 2
            groups.append({"id": proto, "r": None, "c": norm, "v": vg, "D": D,
                           "d": [F(c["d_upper"]) for c in cs],
                           "costs": {r: [F(c["costs"][r]) for c in cs] for r in RESOURCES},
                           "workspace": [c["workspace_peak"] for c in cs]})
        expanded = [next(g for g in groups if g["id"] == by_id[m["column_id"]]["prototype"])
                    for p in profiles for m in p["memberships"]]
        _, gx2, gd2, gc2 = reserves(expanded, target)
        _, gx3, gd3, gc3 = reserves(groups, target)
        result[xs] = {"PASS": True, "columns": len(by_id), "B2_alias_groups": len(expanded),
                      "B3_groups": len(groups), "Ymax": ymax,
                      "degree_matching_all_three_representations": True,
                      "precision_D_and_event_identity": True, "positive_norm_enclosures": True,
                      "membership_preflight": True, "column_L1_envelope_less_than_2": True,
                      "cost_bias_nonnegative": True, "workspace_exclusions": data["workspace_exclusions"],
                      "reserves_B2": {"xi": gx2, "d": gd2, "Q": gc2},
                      "reserves_B3": {"xi": gx3, "d": gd3, "Q": gc3},
                      "shared_max_reserves": {"xi": max(gx2, gx3), "d": max(gd2, gd3),
                                              "Q": {r: max(gc2[r], gc3[r]) for r in RESOURCES}}}
    # Independently verify the shared O0 native implementation on saved R1 rows.
    raw = json.loads((ROOT/"artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json").read_text())
    aliases = 0
    for xs, ep, sign in product(("1/8", "1/4"), ("1e-3", "1e-4", "1e-6"), (1, -1)):
        selected = [r for r in raw["resource_rows"] if r["context"] == "distinct_basis" and r["controlled"]
                    and (r["x"], r["epsilon"], r["sigma"]) == (xs, ep, sign)
                    and r["arm"] in ("ordinary", "PTSC_K0")]
        assert len(selected) == 2
        events = [[e for e in r["profile"]["events"] if e["label"].split(":")[0] == "0"] for r in selected]
        # B0 implemented_coefficient varies with normalization; conditional event does not.
        keys = ("label", "label_probability", "word", "rotation", "rotation_sign", "phase_i_power", "complement", "a", "b", "native_cost")
        assert [{k: e[k] for k in keys} for e in events[0]] == [{k: e[k] for k in keys} for e in events[1]]
        assert selected[0]["profile"]["workspace_qubits_beyond_2_system"] == selected[1]["profile"]["workspace_qubits_beyond_2_system"]
        aliases += 1
    return {"tables": result, "cross_arm_O0_native_identity_checks": aliases,
            "registered_optimization_calls": 0, "budget_freeze": False}


def synthetic_audit():
    cases, counterexamples = [], []
    def case(name, function):
        try:
            detail = function()
            cases.append({"name": name, "PASS": True, "detail": detail})
        except Exception as exc:
            cases.append({"name": name, "PASS": False, "error": type(exc).__name__+": "+str(exc)})
    def rejects(function):
        try: function()
        except ValueError: return {"expected_rejection": True}
        raise AssertionError("invalid input accepted")

    case("all_representations_active", lambda: verify_fixture(scalar_b2()))
    case("inactive_representation_and_zero_groups", lambda: verify_fixture(scalar_b2(inactive=True)))
    case("B1_fixed_representation", lambda: verify_fixture(tuple([scalar_b2(inactive=True)[0][:2], scalar_b2(inactive=True)[1][:2], F(1), {"r0": F(1)}, [F(1)]])))
    case("single_precision_support", lambda: verify_fixture(scalar_b2(precisions=1)))
    case("finite_norm_interval_midpoint_shift", lambda: verify_fixture(scalar_b2(width=F(1, 100*N))))
    case("finite_D_interval_width", lambda: verify_fixture(scalar_b2(dwidth=F(1, 100*N))))

    def zero_group():
        gs = [group("a"), group("b")]
        return verify_fixture((gs, [[F(1), F(0), F(0)], [F(0)]*3], F(1), None, [F(1)]))
    case("B3_available_group_zero_mass", zero_group)
    def tiny():
        gs = [group("a", F(1)), group("b", F(1, N*N))]
        u, y, z, target = b2(gs, {"r0": F(1)})
        result = verify_fixture((gs, u, y, z, target))
        assert sum(result["counts"][1]) == 0
        assert sum(u[1]) > 0
        return result
    case("positive_continuous_group_rounds_to_zero_counts", tiny)
    def negative_direction():
        gs = [group("a", vector=[F(3, 5), F(4, 5)]), group("b", vector=[F(4, 5), -F(3, 5)])]
        u, y, z, t = b2(gs, {"r0": F(1)})
        return verify_fixture((gs, u, y, z, t))
    case("ordered_negative_degree_endpoints", negative_direction)
    def nontrivial_mean():
        gs = [group("a", precisions=2, vector=[F(1), F(0), F(0)]),
              group("b", precisions=2, vector=[F(0), F(1), F(0)]),
              group("c", precisions=2, vector=[F(0), F(0), F(1)])]
        masses = [F(1, 7), F(2, 7), F(4, 7)]
        u = [[mass*F(2, 5), mass*F(3, 5)] for mass in masses]
        result = verify_fixture((gs, u, F(1), None, masses))
        assert result["decoded_certificate"]["xi"] > 0
        return result
    case("nonzero_multidegree_rounding_residual", nontrivial_mean)
    def y_half_tie():
        y = 1-F(1, 2*N)
        gs = [group("a", precisions=1)]
        result = verify_fixture((gs, [[F(1)]], y, None, [1/y]))
        assert result["y"]-y == F(1, 2*N)
        return result
    case("y_half_up_tie_attains_half_unit_bound", y_half_tie)

    def boundary(conf=False, cost=False, mean=False):
        gs, u, y, z, t = scalar_b2()
        if mean:
            ng = len(gs)
            width = (DELTA*y-F(2*ng+1, 2*N)-DELTA/(2*N))/(1+F(ng, N))
            for g in gs: g["D"] = [(1-width, 1+width)]
        if conf:
            rho, gx, _, _ = reserves(gs, t)
            xi = sum(r*sum(row) for r, row in zip(rho, u))
            a = F(sum(len(g["d"])+1 for g in gs), N)
            dd = (E*y-xi-KAPPA-E/(2*N)-gx)/(1+a)
            assert dd >= 0
            for g in gs: g["d"] = [dd]*len(g["d"])
        inside = inner(gs, u, y, t, SHOTS, KAPPA)
        caps = {r: inside["nominal_resources"][r]+2*SHOTS*inside["Gamma_Q"][r] for r in RESOURCES} if cost else {}
        check = inner(gs, u, y, t, SHOTS, KAPPA, caps)
        assert check["PASS"]
        if conf: assert check["margins"]["confidence"] == 0
        if mean: assert check["margins"]["mean"] == 0
        if cost: assert all(check["margins"]["cap_"+r] == 0 for r in RESOURCES)
        q, yn, zn, _, _ = decode(gs, u, y, t, z)
        cert = certificate(gs, q, yn, t, SHOTS, KAPPA, caps, zn)
        assert cert["certified"]
        return {"inner_margins": check["margins"], "decoded": cert}
    case("confidence_inner_boundary", lambda: boundary(conf=True))
    case("resource_inner_boundary", lambda: boundary(cost=True))
    case("mean_inner_boundary", lambda: boundary(mean=True))
    case("confidence_and_resource_simultaneous_boundary", lambda: boundary(conf=True, cost=True))

    def nominal_negative():
        gs, u, y, z, t = scalar_b2()
        u[0][0] = -F(1, 10**40)
        return rejects(lambda: decode(gs, u, y, t, z))
    case("near_zero_negative_nominal_rejected_not_clipped", nominal_negative)
    case("non_normalized_law_rejected", lambda: rejects(lambda: lrm([F(1, 4), F(1, 4)], N)))
    case("float_not_promoted_to_exact", lambda: rejects(lambda: lrm([0.5, 0.5], N)))
    case("fixed_index_group_tie", lambda: {"counts": lrm([F(1, 4), F(1, 4), F(1, 2)], 2)} if lrm([F(1, 4), F(1, 4), F(1, 2)], 2) == [1, 0, 1] else (_ for _ in ()).throw(AssertionError("tie")))
    case("fixed_index_precision_tie", lambda: {"counts": lrm([F(1, 2), F(1, 2)], 1)} if lrm([F(1, 2), F(1, 2)], 1) == [1, 0] else (_ for _ in ()).throw(AssertionError("tie")))
    case("zero_count_internal_distribution", lambda: {"counts": lrm([F(1, 3), F(2, 3)], 0)} if lrm([F(1, 3), F(2, 3)], 0) == [0, 0] else (_ for _ in ()).throw(AssertionError("zero")))

    def impossible_structure():
        gs = [group("a")]
        return rejects(lambda: decode(gs, [[F(1), F(0), F(0)]], F(1), [F(2)]))
    case("degree_mismatch_cannot_be_given_zero_mean", impossible_structure)
    def false_zero():
        gs = [group("a")]
        cert = certificate(gs, [[F(1), F(0), F(0)]], F(1), [F(2)], SHOTS, KAPPA)
        assert cert["xi"] == 1 and not cert["checks"]["mean"]
        return cert
    case("direct_certificate_detects_false_zero_residual", false_zero)

    def original_boundary():
        gs = [group("a", precisions=1)]
        gs[0]["d"] = [E-KAPPA]
        cert = certificate(gs, [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        candidate = inner(gs, [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        assert cert["certified"] and cert["h"] == KAPPA
        assert not candidate["PASS"] and candidate["margins"]["confidence"] < 0
        example = {"id": "original_feasible_inner_infeasible", "groups": gs, "u": [[F(1)]],
                   "y": F(1), "target": [F(1)], "original": cert, "inner": candidate,
                   "proof": "single scalar group c=v=t=1 forces u=y=1; H=kappa; positive Gamma_h makes every inner point infeasible"}
        counterexamples.append(example)
        return example
    case("original_confidence_boundary_and_inner_empty", original_boundary)
    def bad_confidence():
        gs = [group("a", precisions=1)]
        gs[0]["d"] = [E+F(1, 1000)]
        cert = certificate(gs, [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        assert cert["h"] < 0 and not cert["checks"]["confidence"]
        return cert
    case("negative_confidence_rejected", bad_confidence)
    def precision_move():
        gs = [group("a", precisions=2)]
        before = [[F(1, 2), F(1, 2)]]
        after = [[F(1, 2)-F(1, N), F(1, 2)+F(1, N)]]
        a = certificate(gs, before, F(1), [F(1)], SHOTS, KAPPA)
        b = certificate(gs, after, F(1), [F(1)], SHOTS, KAPPA, a["resources"])
        assert b["h"] > a["h"] and not b["checks"]["caps"]
        assert b["xi"] == a["xi"] and b["workspace"] == a["workspace"]
        example = {"id": "confidence_improvement_breaks_resource_cap", "before": a, "after": b,
                   "counts_shift": 1, "denominator": N}
        counterexamples.append(example)
        return example
    case("precision_move_confidence_gain_but_resource_failure", precision_move)
    def resource_equality():
        gs = [group("a", precisions=1)]
        a = certificate(gs, [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        b = certificate(gs, [[F(1)]], F(1), [F(1)], SHOTS, KAPPA, a["resources"])
        assert b["certified"]
        return b
    case("original_resource_cap_equality_accepted", resource_equality)
    def reversed_error():
        gs, u, y, z, t = scalar_b2()
        for g in gs: g["d"] = [F(1, 100_000), F(2, 100_000), F(3, 100_000)]
        return verify_fixture((gs, u, y, z, t))
    case("nominal_precision_vs_saved_error_reversed", reversed_error)
    def workspace(active):
        gs = [group("a", precisions=2)]
        gs[0]["workspace"] = [1, 2]
        q = [[F(0), F(1)]] if active else [[F(1), F(0)]]
        cert = certificate(gs, q, F(1), [F(1)], SHOTS, KAPPA)
        assert cert["checks"]["workspace"] == (not active)
        return cert
    case("inactive_high_workspace_variant_does_not_set_peak", lambda: workspace(False))
    case("active_high_workspace_variant_rejected", lambda: workspace(True))
    def zero_cost():
        gs, u, y, z, t = scalar_b2()
        for g in gs:
            g["costs"] = {r: [F(0)]*len(g["d"]) for r in RESOURCES}
        result = verify_fixture((gs, u, y, z, t))
        assert result["decoded_certificate"]["resources"]["1Q"] == 5*SHOTS
        return result
    case("zero_resource_coordinates_keep_fixed_1Q_overhead", zero_cost)

    def no_identity():
        # Both intervals enclose the same ideal D=1, but only the second is wide.
        cert_loose = certificate([group("a", precisions=1)], [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        gs = [group("a", precisions=1)]; gs[0]["D"] = [(F(1), F(5))]
        cert_wide = certificate(gs, [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        assert cert_loose["xi"] == 0 and cert_wide["xi"] == 4
        example = {"id": "unequal_precision_intervals_cannot_use_first_variant_bound", "q": [0, 1],
                   "D_precision1": [["1", "1"]], "D_precision2": [["1", "5"]],
                   "cbar": "1", "v": ["1"], "y": "1", "t": ["1"],
                   "claimed_Xi_using_first_interval": "0", "actual_xi": "4"}
        counterexamples.append(example)
        return example
    case("D_identity_prerequisite_counterexample", no_identity)
    def negative_cost():
        proposed = (max([F(-1)])+sum([F(-1)]))/N
        assert proposed < 0  # Even a zero cost change cannot obey |delta| <= negative Gamma.
        return {"negative_Gamma": proposed, "observed_absolute_change": F(0), "missing_precondition": "C>=0"}
    case("negative_cost_is_outside_Gamma_theorem", negative_cost)
    def preflight_failure():
        gs = [group("a", c=F(25, 16), precisions=1)]
        gs[0]["c"] = (F(1, 8), F(3))
        u, y, z, t = b2(gs, {"r0": F(1)})
        q, yn, zn, _, _ = decode(gs, u, y, t, z, 8)
        cert = certificate(gs, q, yn, t, SHOTS, KAPPA, z=zn, denominator=8)
        assert not cert["checks"]["membership"]
        example = {"id": "failed_membership_preflight", "N_artificial": 8,
                   "c": gs[0]["c"], "y": y, "rounded_y": yn, "Ymax": F(2),
                   "Ymax_rad": F(2)*(gs[0]["c"][1]-gs[0]["c"][0])/2,
                   "required_max": F(2, 8), "certificate": cert}
        counterexamples.append(example)
        return example
    case("membership_preflight_failure_is_not_accepted", preflight_failure)
    def aliases():
        gs = [group("alias0", precisions=1), group("alias1", precisions=1)]
        before = certificate(gs, [[F(1, 4)], [F(3, 4)]], F(1), [F(1)], SHOTS, KAPPA)
        after = certificate([gs[0]], [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        assert before["certified"] and after["certified"]
        for k in ("xi", "bias", "h", "resources", "workspace"):
            assert before[k] == after[k]
        return {"before": before, "after": after, "no_requantization": True}
    case("B2_alias_sum_preserves_B3_certificate", aliases)
    def non_alias():
        gs = [group("a", precisions=1), group("b", precisions=1)]
        gs[1]["costs"]["T"] = [F(10)]
        before = certificate(gs, [[F(1, 2)], [F(1, 2)]], F(1), [F(1)], SHOTS, KAPPA)
        false = certificate([gs[0]], [[F(1)]], F(1), [F(1)], SHOTS, KAPPA)
        assert before["resources"]["T"] != false["resources"]["T"]
        return {"alias_identity_required": True, "before_T": before["resources"]["T"], "false_T": false["resources"]["T"]}
    case("unequal_native_identity_cannot_be_coalesced", non_alias)
    def ymax_warning():
        ymax = F(1)+F(3, 4*N)
        yn = F(N+1, N)
        assert yn > ymax and abs(yn-ymax) <= F(1, 2*N)
        return {"scope": "y<=Ymax alone; not a counterexample to fixed-table simultaneous mean certificate",
                "y": ymax, "Ymax": ymax, "rounded_y": yn,
                "closure": "fixed table column L1<2 and certified mean imply original Ymax bound"}
    case("rounded_y_upper_bound_needs_its_own_argument", ymax_warning)
    case("nonpositive_norm_enclosure_rejected", lambda: rejects(lambda: validate(
        [dict(group("a"), c=(F(0), F(1)))], [[F(2), F(0), F(0)]], F(1), [F(1)])))
    def count_boundary():
        counts = lrm([F(1)-F(1, N), F(1, N)], N)
        assert counts == [N-1, 1]
        return {"counts": counts, "exact_integer_sum": sum(counts)}
    case("2_to_60_integer_counts_no_overflow", count_boundary)

    # Complete enumeration of six-coordinate simplex lattice denominator 3.
    # Each group has two precisions; algorithm N is independently 2,3,4.
    lattice_count = 0
    for weights in product(range(4), repeat=6):
        if sum(weights) != 3: continue
        for small_n in (2, 3, 4):
            def probe(weights=weights, small_n=small_n):
                gs = [group(str(j), precisions=2) for j in range(3)]
                u = [[F(weights[2*j], 3), F(weights[2*j+1], 3)] for j in range(3)]
                result = verify_fixture((gs, u, F(1), None, [F(1)]), small_n, require_inner=False)
                return {"weights_numerators": weights, "artificial_N": small_n,
                        "counts": result["counts"], "rounding_bounds_PASS": True}
            case("exhaustive_lattice_"+str(lattice_count), probe)
            lattice_count += 1
    assert lattice_count == 168
    return {"cases": cases, "PASS": all(c["PASS"] for c in cases),
            "case_count": len(cases), "pass_count": sum(c["PASS"] for c in cases),
            "fail_count": sum(not c["PASS"] for c in cases),
            "exhaustive_cases": lattice_count, "deterministic": True,
            "production_denominator_unchanged": N, "artificial_confidence_fixture": {"n": SHOTS, "ell_upper": ELL, "kappa_upper": KAPPA},
            "certificate_computation_independent_of_reserves": True,
            "registered_table_used_in_synthetic_cases": False, "counterexamples": counterexamples}


def backend_inventory():
    return {"status": "UNVERIFIED_BACKEND", "candidate": "SoPlex",
            "PATH_executables": {x: shutil.which(x) for x in ("soplex", "esolver", "qsopt_ex", "sage", "glpsol")},
            "system_python": sys.executable, "new_dependency_installations": 0,
            "synthetic_solver_calls": 0, "registered_solver_calls": 0,
            "rational_parser_primal_dual_Farkas_runtime_verified": False,
            "runtime_measurements": None, "old_v3_2_seconds_cap_inherited": False,
            "official_sources": ["https://soplex.zib.de/", "https://github.com/scipopt/soplex/blob/master/src/soplex.h"],
            "documented_capability": "rational LP support and rational primal/dual/Farkas APIs; not proof of an available executable or a successful build",
            "license": "official site states Apache-2.0 from 6.0.3; third-party/build dependencies require their own identity inventory",
            "required_before_production": ["approved isolated backend/version/build", "rational input/output round trip", "independent all-row primal/dual/Farkas verification", "off-domain timing/caps"]}


def write(name, value):
    dest = OUT/name
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)+"\n")


def main():
    started, cpu = time.monotonic(), time.process_time()
    # Audit-only safety limits, not an inherited production execution contract.
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (256*1024**2, 256*1024**2))
    prior = json.loads((ROOT/"artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/input_identity_v1.json").read_text())
    manifest = json.loads((ROOT/"artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/evidence_manifest_v1.json").read_text())
    protected = dict(prior["protected_sha256"])
    protected.update({p: meta["sha256"] for p, meta in manifest["files"].items()})
    mp = "artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/evidence_manifest_v1.json"
    protected[mp] = sha256((ROOT/mp).read_bytes()).hexdigest()
    assert len(protected) == 135
    assert not any(p.endswith(".npz") for p in protected)
    for name, expected in protected.items(): assert sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    table = static_table_audit()
    synthetic = synthetic_audit()
    write("input_identity_v1.json", {"base_commit": BASE, "old_source_commit": "45cffb2aa10f9219b6cad929c3ade49fe7d36ca8",
        "old_authorization_commit": "2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9", "old_result_commit": "35f8b949079f15d0348bc082b916324870da7246",
        "design_commit": "06575a3bc9d2354b829e0e9a6c21ad1512f77909", "design_sha256": "5af8430d2d23adb0db0f9be87ecd1b851037984b7e9c1af9cb53f3218a1aa1a7",
        "instruction_attachment_path": "/home/abe/.codex/attachments/f7f69b91-59ae-432b-9629-068eeb8c9503/貼り付けたテキスト.txt",
        "instruction_sha256": sha256((OUT/"inputs/user_audit_instruction_20261009.md").read_bytes()).hexdigest(),
        "audit_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "protected_sha256": protected,
        "protected_count": len(protected), "old_marker_sha256": prior["marker_sha256"], "runtime": {"python": sys.version, "executable": sys.executable}})
    write("rounding_bound_audit_v1.json", {"static_table_preflight": table, "bound_status": "PASS_UNDER_EXPLICIT_PRECONDITIONS",
        "proof_document": "docs/tracks/algorithm_codesign/ra_d0_v4_mathematical_audit_20261009.md",
        "incorrect_assumption_counterexamples": synthetic["counterexamples"], "production_formulas_changed": False})
    write("synthetic_verification_v1.json", synthetic)
    write("exact_solver_feasibility_v1.json", backend_inventory())
    for name, expected in protected.items(): assert sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    write("audit_execution_v1.json", {"synthetic_audit_runs": 1, "wall_seconds": time.monotonic()-started,
        "CPU_seconds": time.process_time()-cpu, "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "audit_CPU_cap_seconds": 30, "audit_address_space_cap_bytes": 256*1024**2,
        "protected_hashes_unchanged_before_after": True, "registered_solver_calls": 0, "synthetic_solver_calls": 0,
        "production_implementation": 0, "new_synthesis": 0, "science": 0, "circuit_matrix_trajectory_DF_molecule_NPZ_GPU": 0,
        "authorization_marker_changes": 0, "mandatory_STOP": True})
    print(json.dumps({"synthetic_cases": synthetic["case_count"], "PASS": synthetic["pass_count"], "FAIL": synthetic["fail_count"],
                      "fixed_table_preflight": "PASS", "backend": "UNVERIFIED_BACKEND", "protected_files": len(protected)}))
    if not synthetic["PASS"]:
        print(json.dumps(jsonable([c for c in synthetic["cases"] if not c["PASS"]]), ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__": main()
