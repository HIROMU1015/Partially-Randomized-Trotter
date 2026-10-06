"""Fixed numerical membership and common implementation certificates v2."""
from fractions import Fraction as F
from .exact import DENOMINATOR, certify_law, quantize
from .lp import build_lp

RESOURCES = ("T", "CX", "1Q")


def membership_allowance(coefficient_upper):
    return (F(3)+F(coefficient_upper)/2)/DENOMINATOR


def build_numerical_lp(data, n, objective, ell, baseline="B3", caps=None,
                       robust=True, representation=None):
    return build_lp(data, n, objective, ell, baseline, caps, robust,
                    representation, numerical=True)


def membership_certificate(data, labels, q, y, baseline, z=None):
    if baseline == "B3":
        return {"certified": True, "groups": [], "latent_z": {}}
    arms = list(dict.fromkeys(arm for arm, _ in labels))
    if baseline == "B1":
        if len(arms) != 1:
            return {"certified": False, "reason": "B1_FIXED_REPRESENTATION_REQUIRED"}
        z = {arms[0]: y}
    if (z is None or set(z) != set(arms) or any(F(v) < 0 for v in z.values())
            or sum(map(F, z.values())) != y):
        return {"certified": False, "reason": "INVALID_LATENT_MEMBERSHIP_WITNESS"}
    profiles = {p["arm"]: p for p in data["B0_saved_profiles"] if p["epsilon"] == "1e-3"}
    groups = []
    for arm in arms:
        for member in profiles[arm]["memberships"]:
            prototype = member["column_id"].split(":")[0]
            mass = sum(q[j] for j, (r, c) in enumerate(labels)
                       if r == arm and c.split(":")[0] == prototype)
            lo, hi = map(F, member["ideal_weight_interval"])
            residual = max(abs(mass-z[arm]*lo), abs(mass-z[arm]*hi))
            tau = membership_allowance(hi)
            groups.append({"arm": arm, "prototype": prototype, "mass": str(mass),
                           "residual_upper": str(residual), "tau": str(tau),
                           "certified": residual <= tau})
    return {"certified": all(g["certified"] for g in groups), "groups": groups,
            "latent_z": {r: str(v) for r, v in z.items()}}


def common_certificate(data, labels, q, y, n, ell, caps=None):
    lookup = {c["id"]: c for c in data["columns"]}
    columns = [[tuple(map(F, v)) for v in lookup[c]["D_intervals"]] for _, c in labels]
    d = [F(lookup[c]["d_upper"]) for _, c in labels]
    costs = {r: [F(lookup[c]["costs"][r]) for _, c in labels] for r in RESOURCES}
    cert = certify_law(columns, list(map(F, data["target"])), d, costs, q, y, n, ell, caps)
    peak = max([0]+[lookup[c].get("workspace_peak", 1) for j, (_, c) in enumerate(labels) if q[j]])
    cert["workspace_peak"] = peak
    cert["certified"] = cert["certified"] and peak <= 1
    return cert


def certify_nominal(data, lp, nominal, n, ell, baseline, objective, caps=None):
    nq = len(lp.labels)
    try:
        q, y = quantize(nominal[:nq], nominal[nq])
        z = None
        if baseline == "B2":
            arms = list(dict.fromkeys(r for r, _ in lp.labels))
            raw = list(map(F, nominal[nq+5:nq+5+len(arms)]))
            if len(raw) != len(arms) or any(v < 0 for v in raw) or sum(raw) <= 0:
                raise ValueError("invalid nominal z")
            # z is a witness, not a sampler law. Preserve its nominal shares
            # and enforce sum z=y exactly after the frozen rounding of y.
            z = {r: v*y/sum(raw) for r, v in zip(arms, raw)}
        member = membership_certificate(data, lp.labels, q, y, baseline, z)
        cert = common_certificate(data, lp.labels, q, y, n, ell, caps)
        cert["membership"] = member
        cert["certified"] &= member["certified"]
        cert["objective_upper"] = cert.get("resources", {}).get(objective)
        cert["reason"] = "PASS" if cert["certified"] else "UNCERTIFIED_NUMERICAL_POINT"
        return cert
    except (ValueError, ArithmeticError):
        return {"certified": False, "reason": "UNCERTIFIED_NUMERICAL_POINT"}


def coalesce_to_B3(data, labels, q):
    ids = [c["id"] for c in data["columns"]]
    result = [F(0)]*len(ids)
    for (_, ident), value in zip(labels, q, strict=True):
        result[ids.index(ident)] += F(value)
    return [(None, ident) for ident in ids], result
