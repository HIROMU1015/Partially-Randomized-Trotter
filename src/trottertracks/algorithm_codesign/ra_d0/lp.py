"""LP construction and rational certificates, independent of solver trust.

The B2 outer relaxation gives safe objective lower bounds, not certified
B2 samplers. Registered solve calls are blocked in this preparation source.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from .exact import E, DELTA_NUM, dot, kappa_upper


@dataclass
class LP:
    c: list
    A: list
    b: list
    H: list
    f: list
    upper: list
    objective_offset: F = F(0)
    domain: str = "SYNTHETIC"
    labels: tuple = ()

    def validate(self):
        size = len(self.c)
        if not size or len(self.upper) != size or any(v < 0 for v in self.upper):
            raise ValueError("invalid LP bounds")
        if len(self.A) != len(self.b) or len(self.H) != len(self.f):
            raise ValueError("invalid LP dimensions")
        if any(len(r) != size for r in self.A+self.H):
            raise ValueError("invalid LP row")


def dual_lower(lp, inequality_multipliers, equality_multipliers):
    """Exact weak-duality bound, with finite-domain residual correction.

For min c*x, A*x<=b, H*x=f, x>=0, use nu>=0 and free u.
Any negative stationarity residual is bounded using certified finite upper
bounds. Float tolerances never decide certificate validity.
"""
    lp.validate()
    nu, u = list(map(F, inequality_multipliers)), list(map(F, equality_multipliers))
    if len(nu) != len(lp.A) or len(u) != len(lp.H) or any(v < 0 for v in nu):
        raise ValueError("invalid dual multipliers")
    residual = [lp.c[j]+sum(v*r[j] for v, r in zip(nu, lp.A))
                +sum(v*r[j] for v, r in zip(u, lp.H)) for j in range(len(lp.c))]
    correction = sum(min(F(0), v)*cap for v, cap in zip(residual, lp.upper))
    lower = lp.objective_offset-dot(nu, lp.b)-dot(u, lp.f)+correction
    return {"lower": lower, "stationarity_correction": correction,
            "multipliers_exact": (nu, u), "residual_exact": residual}


def farkas_certificate(lp, inequality_multipliers, equality_multipliers):
    lp.validate()
    nu, u = list(map(F, inequality_multipliers)), list(map(F, equality_multipliers))
    if len(nu) != len(lp.A) or len(u) != len(lp.H) or any(v < 0 for v in nu):
        raise ValueError("invalid infeasibility multipliers")
    r = [sum(v*row[j] for v, row in zip(nu, lp.A))
         +sum(v*row[j] for v, row in zip(u, lp.H)) for j in range(len(lp.c))]
    minimum = sum(min(F(0), v)*cap for v, cap in zip(r, lp.upper))
    rhs = dot(nu, lp.b)+dot(u, lp.f)
    return {"certified_infeasible": rhs < minimum, "rhs": rhs, "domain_lower": minimum}


def make_farkas_problem(lp):
    """Normalized separation LP, including original finite variable bounds.

    An infeasibility flag triggers at most this separately budgeted auxiliary
    problem, not a retry of the original scientific/design query. Any returned
    ray still must pass exact verification; failed verification is inconclusive.
    """
    lp.validate()
    nv, ma, mh = len(lp.c), len(lp.A), len(lp.H)
    matrix = lp.A+[[F(int(i == j)) for j in range(nv)] for i in range(nv)]
    rhs = lp.b+lp.upper
    size = ma+nv+2*mh
    c = rhs+lp.f+[-v for v in lp.f]
    A = []
    for j in range(nv):
        A.append([-r[j] for r in matrix]+[-r[j] for r in lp.H]+[r[j] for r in lp.H])
    return LP(c, A, [F(0)]*nv, [[F(1)]*size], [F(1)], [F(1)]*size,
              domain=lp.domain)


def check_farkas_output(lp, nominal_ray):
    ma, nv, mh = len(lp.A), len(lp.c), len(lp.H)
    ray = [max(F(0), F(v)) for v in nominal_ray]
    if len(ray) != ma+nv+2*mh:
        raise ValueError("invalid auxiliary certificate dimensions")
    nu = ray[:ma]
    u = [ray[ma+nv+k]-ray[ma+nv+mh+k] for k in range(mh)]
    # The bounded-domain residual correction already accounts for upper-bound
    # multipliers. The verifier does not trust auxiliary feasibility flags.
    return farkas_certificate(lp, nu, u)


def build_lp(data, n, objective, ell_upper, baseline="B3", caps=None, robust=True,
             representation=None, numerical=False):
    """Build interval inner/outer problems; v2 adds numerical membership.

    The legacy entry point keeps its v1 behavior. The numerical v2 entry point
    supports B1/B2 inner membership, followed by independent rounding and
    implementation certification. An outer optimizer is never a sampler law.
    """
    if baseline not in ("B1", "B2", "B3") or objective not in ("T", "CX", "1Q"):
        raise ValueError("unsupported class or objective")
    lookup = {c["id"]: c for c in data["columns"]}
    representations = [p for p in data["B0_saved_profiles"] if p["epsilon"] == "1e-3"]
    if baseline == "B1":
        representations = [p for p in representations if p["arm"] == representation]
        if len(representations) != 1:
            raise ValueError("B1 requires exactly one fixed representation")
    membership = baseline in ("B1", "B2")
    if membership:
        if robust and not numerical:
            raise ValueError("B1/B2 certified primal membership is pending review")
        entries = [(r["arm"], m["column_id"].split(":")[0]+":"+ep)
                   for r in representations for m in r["memberships"] for ep in ("1e-3", "1e-4", "1e-6")]
    else:
        entries = [(None, c["id"]) for c in data["columns"]]
    size_q = len(entries)
    y, start_r = size_q, size_q+1
    start_z = start_r+4
    z_count = len(representations) if membership else 0
    size = start_z+z_count
    A, b, H, f = [], [], [], []
    def add(coefficients, rhs=F(0), equality=False):
        row = [F(0)]*size
        for j, v in coefficients.items():
            row[j] = F(v)
        (H if equality else A).append(row)
        (f if equality else b).append(F(rhs))
    add({j: 1 for j in range(size_q)}, 1, equality=True)
    target = list(map(F, data["target"]))
    ymax = F(2)/(sum(target)-DELTA_NUM)
    # Bounds enclose even numerical-mean points: sum D*q <= sqrt(2) < 2,
    # and |sum D*q-y sum t| <= y delta_num.
    add({y: 1}, ymax)
    for k in range(4):
        lower = {j: F(lookup[c]["D_intervals"][k][0]) for j, (_, c) in enumerate(entries)}
        upper = {j: F(lookup[c]["D_intervals"][k][1]) for j, (_, c) in enumerate(entries)}
        positive = upper if robust else lower
        negative = lower if robust else upper
        add(positive | {y: -target[k], start_r+k: -1})
        add({j: -v for j, v in negative.items()} | {y: target[k], start_r+k: -1})
    add({start_r+k: 1 for k in range(4)} | {y: -DELTA_NUM})
    add({j: F(lookup[c]["d_upper"]) for j, (_, c) in enumerate(entries)}
        | {y: -E} | {start_r+k: 1 for k in range(4)}, -kappa_upper(n, ell_upper))
    if membership:
        add({start_z+r: 1 for r in range(z_count)} | {y: -1}, equality=True)
        for ir, representation in enumerate(representations):
            for m in representation["memberships"]:
                prototype = m["column_id"].split(":")[0]
                group = {j: 1 for j, (arm, c) in enumerate(entries)
                         if arm == representation["arm"] and c.split(":")[0] == prototype}
                lo, hi = map(F, m["ideal_weight_interval"])
                tau = (F(3)+hi/2)/2**60 if numerical else F(0)
                # Inner rows verify both interval endpoints. Outer rows contain
                # every numerical implementation satisfying those endpoints.
                above = lo if robust else hi
                below = hi if robust else lo
                add(group | {start_z+ir: -above}, tau)
                add({j: -v for j, v in group.items()} | {start_z+ir: below}, tau)
    c = [F(0)]*size
    for j, (_, ident) in enumerate(entries):
        c[j] = 2*n*F(lookup[ident]["costs"][objective])
    offset = 5*n if objective == "1Q" else F(0)
    for resource, limit in (caps or {}).items():
        add({j: 2*n*F(lookup[ident]["costs"][resource]) for j, (_, ident) in enumerate(entries)},
            F(limit)-(5*n if resource == "1Q" else 0))
    lp = LP(c, A, b, H, f, [F(1)]*size_q+[ymax]+[ymax*DELTA_NUM]*4
            +[ymax]*z_count, F(offset), "REGISTERED_SAVED_TABLE", tuple(entries))
    lp.validate()
    return lp


def solve_synthetic(lp):
    """No registered-domain override. Source revision/review required to unlock."""
    lp.validate()
    if lp.domain != "SYNTHETIC":
        raise PermissionError("RA-D0 registered optimization is not authorized; mandatory STOP")
    from scipy.optimize import linprog, OptimizeWarning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unrecognized options detected", OptimizeWarning)
        result = linprog([float(v) for v in lp.c],
                     A_ub=[[float(v) for v in r] for r in lp.A] or None,
                     b_ub=[float(v) for v in lp.b] or None,
                     A_eq=[[float(v) for v in r] for r in lp.H] or None,
                     b_eq=[float(v) for v in lp.f] or None,
                     bounds=[(0, float(v)) for v in lp.upper], method="highs-ds",
                     options={"presolve": True, "time_limit": 2, "threads": 1, "parallel": False,
                              "dual_feasibility_tolerance": 1e-9,
                              "primal_feasibility_tolerance": 1e-9})
    output = {"status": int(result.status), "message": result.message}
    if result.status == 0:
        output["nominal_primal"] = [F.from_float(float(v)) for v in result.x]
        # Bound multipliers need not be included: the exact finite-domain
        # correction covers negative stationarity residuals from upper bounds.
        nu = [max(F(0), -F.from_float(float(v))) for v in result.ineqlin.marginals]
        u = [-F.from_float(float(v)) for v in result.eqlin.marginals]
        output["dual_certificate"] = dual_lower(lp, nu, u)
    return output


def strict_witness(primal_certificate, dual_certificate):
    # Infeasibility-only points are descriptive and never this primary flag.
    return bool(primal_certificate.get("certified") and dual_certificate is not None
                and F(primal_certificate["objective_upper"]) < F(dual_certificate["lower"]))
