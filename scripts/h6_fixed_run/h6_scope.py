"""Exact finite H6 scope, physical schedules and short-price calibration design."""

from collections import defaultdict
from fractions import Fraction
from itertools import product
import hashlib
import json
import math


NATIVE = {"FULL_PF": "FULL", "TRUNCATED_PF": "B0I", "PR_DF": "PR"}
LABELS = {"FULL_PF": "Full PF", "TRUNCATED_PF": "B0I", "PR_DF": "PR-DF"}
FORMULAS = {"2nd": 1, "4th": 3, "4th(m5_best)": 11}
AXES = ("X", "Y")


def read(path):
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def exact(value):
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    return Fraction(value)


def enc(value):
    value = exact(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def physical_id(cell):
    identity = [cell[key] for key in ("system", "method", "formula", "prefix", "T", "q", "R_outer", "K")]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]


def case_rows(scope, rank):
    """Enumerate every requested point; preserve requested widths and absolute kappa."""
    h = float(exact(scope["h"]))
    M = scope["M"]
    if not math.isfinite(h) or h <= 0 or type(M) is not int or M < 0:
        raise ValueError("H6 requires positive finite h and a nonnegative integer RPE horizon")
    try:
        math.ldexp(h, M)
    except OverflowError as error:
        raise ValueError("The last H6 round time exceeds binary64") from error
    cap = scope["calibration_max_m"]
    if type(cap) is not int or not 0 <= cap <= M:
        raise ValueError("Invalid maximum direct-compilation round")
    if type(scope["price_paths"]) is not int or scope["price_paths"] < 1:
        raise ValueError("PR-DF requires a positive finite-cohort path count")
    methods = scope["methods"]
    if not methods or not set(methods) <= NATIVE.keys():
        raise ValueError("Unknown or empty H6 method scope")
    for method, scope_method in methods.items():
        pr = method == "PR_DF"
        prefixes = [rank] if method == "FULL_PF" else scope_method["prefixes"]
        if prefixes == "all":
            prefixes = range(rank)
        kappas = [exact(x) for x in scope_method["kappas"]] if pr else [None]
        widths = [exact(x) for x in scope_method["width_over_h"]]
        formulas = scope_method["formulas"]
        order = scope_method.get("K", 2) if pr else None
        orders = order if isinstance(order, list) else [order]
        if (not formulas or not set(formulas) <= FORMULAS.keys() or not prefixes or not widths or not kappas
                or any(type(p) is not int or not 0 <= p <= rank or (p == rank) != (method == "FULL_PF")
                       for p in prefixes)
                or any(width <= 0 for width in widths) or pr and any(k <= 0 for k in kappas)
                or pr and (not orders or any(type(k) is not int or k < 0 or k % 2 for k in orders))):
            raise ValueError(f"Invalid H6 range for {method}")
        for formula, prefix, width, kappa, k_order in dict.fromkeys(product(formulas, prefixes, widths, kappas, orders)):
            yield dict(method=method, formula=formula, prefix=prefix, width_over_h=enc(width),
                       kappa=None if kappa is None else enc(kappa), K=k_order)


def rounds(scope, case, lambda_R=None):
    h = Fraction(float(exact(scope["h"])))
    width = h * exact(case["width_over_h"])
    for m in range(scope["M"] + 1):
        T = h * (1 << m)
        q = math.ceil(T / width)
        r = None
        if case["method"] == "PR_DF":
            if lambda_R is None or not 0 < lambda_R < math.inf:
                raise ValueError(f"No positive PR-DF tail normalization for prefix {case['prefix']}")
            # Existing S2 absolute-kappa mapper; a signed formula needs at least one event per piece.
            r = max(FORMULAS[case["formula"]], math.ceil(exact(case["kappa"]) * exact(lambda_R)**2 * T**2 / q))
        cell = dict(system="H6", method=NATIVE[case["method"]], formula=case["formula"],
                    prefix=case["prefix"], T=float(T), delta=float(T / q), q=q, R_outer=r, K=case["K"])
        yield dict(m=m, actual_step=enc(T / q), id=physical_id(cell), **cell)


def group_key(cell):
    return (cell["method"], cell["formula"], cell["prefix"], exact(cell["actual_step"]), cell["K"])


def grouped_rows(cases):
    grouped = defaultdict(dict)
    for case in cases:
        for cell in case["rounds"]:
            grouped[group_key(cell)][cell["id"]] = cell
    return grouped


def calibration_design(cases, scope):
    """Find independent short price probes for every requested physical step."""
    cap = scope["calibration_max_m"]
    max_time = Fraction(float(exact(scope["h"]))) * (1 << cap)
    candidate_cells = {cell["id"]: cell for case in cases for cell in case["rounds"]}

    def probe(template, step, q, r):
        row = dict(template, m=None, T=float(step * q), delta=float(step), q=q, R_outer=r,
                   actual_step=enc(step), calibration_only=True)
        row["id"] = physical_id(row)
        return candidate_cells.get(row["id"], row)

    designs = {}
    for key, cells in grouped_rows(cases).items():
        observed = sorted((c for c in cells.values() if c["m"] <= cap),
                          key=lambda c: (c["q"], c["R_outer"] or 0, c["id"]))
        future = [c for c in cells.values() if c["m"] > cap]
        design = dict(target_step=enc(key[3]), calibration_step=enc(key[3]))
        if not future:
            designs[key] = dict(design, mode="DIRECT", anchors=observed, holdout=None)
            continue
        lo = observed[0] if observed else future[0]
        hi = next((c for c in observed if c["q"] != lo["q"]), None)
        step = key[3]
        pr = key[0] == "PR"
        if hi is None:
            if 2 * step > max_time:
                designs[key] = dict(design, mode="DIRECT", anchors=observed, holdout=None,
                                    reference=lo, reason="INSUFFICIENT_SAME_STEP_CALIBRATION")
                continue
            # q=1 and q=2 identify repetition cost independently of the RPE grid.
            # Keep both probes inside the maximum direct duration.
            r = FORMULAS[lo["formula"]] if pr else None
            lo, hi = (probe(lo, step, q, r) for q in (1, 2))
        anchors = [lo, hi]
        mode = "AFFINE_Q"
        if pr and len({c["R_outer"] for c in (*cells.values(), *anchors)}) > 1:
            mode = "AFFINE_QR"
            def determinant(c):
                # Linear independence of (1,q,qR).
                return ((hi["q"] - lo["q"]) * (c["q"] * c["R_outer"] - lo["q"] * lo["R_outer"])
                        - (c["q"] - lo["q"]) * (hi["q"] * hi["R_outer"] - lo["q"] * lo["R_outer"]))
            third = next((c for c in observed if exact(c["actual_step"]) == step and determinant(c)), None)
            if third is None:
                third = probe(lo, step, lo["q"], lo["R_outer"] + 1)
            anchors.append(third)
        anchor_ids = {cell["id"] for cell in anchors}
        holdout = next((c for c in observed if c["id"] not in anchor_ids), None)
        designs[key] = dict(design, mode=mode, anchors=anchors, holdout=holdout)
    return designs
