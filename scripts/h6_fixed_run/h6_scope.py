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
    if not math.isfinite(h) or h <= 0 or type(M) is not int or not 1 <= M <= 12:
        raise ValueError("H6 requires positive h and an RPE horizon within 1..12")
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
        kappas = [exact(x) for x in scope_method["kappas"]] if pr else [None]
        widths = [exact(x) for x in scope_method["width_over_h"]]
        formulas = scope_method["formulas"]
        k_order = scope_method.get("K", 2) if pr else None
        if (not formulas or not set(formulas) <= FORMULAS.keys() or not prefixes or not widths or not kappas
                or any(type(p) is not int or not 0 <= p <= rank or (p == rank) != (method == "FULL_PF")
                       for p in prefixes)
                or any(width <= 0 for width in widths) or pr and any(k <= 0 for k in kappas)
                or pr and (type(k_order) is not int or k_order < 0 or k_order % 2)):
            raise ValueError(f"Invalid H6 range for {method}")
        for formula, prefix, width, kappa in dict.fromkeys(product(formulas, prefixes, widths, kappas)):
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


def calibration_design(cases, cap):
    """Short physical anchors and, if needed, an independent event-count calibration cell."""
    designs = {}
    for key, cells in grouped_rows(cases).items():
        observed = sorted((c for c in cells.values() if c["m"] <= cap),
                          key=lambda c: (c["q"], c["R_outer"] or 0, c["id"]))
        future = [c for c in cells.values() if c["m"] > cap]
        if not future:
            designs[key] = dict(mode="DIRECT", anchors=observed, holdout=None)
            continue
        if not observed or observed[0]["q"] == observed[-1]["q"]:
            raise ValueError(f"Need two short q values at the same actual step for {key}")
        lo = observed[0]
        # Compile the least expensive independent repetitions first. Later
        # short rounds remain available as validation points for the model.
        hi = next(c for c in observed if c["q"] != lo["q"])
        anchors = [lo, hi]
        mode = "AFFINE_Q"
        if key[0] == "PR" and len({c["R_outer"] for c in cells.values()}) > 1:
            mode = "AFFINE_QR"
            def determinant(c):
                # Linear independence of (1,q,qR).
                return ((hi["q"] - lo["q"]) * (c["q"] * c["R_outer"] - lo["q"] * lo["R_outer"])
                        - (c["q"] - lo["q"]) * (hi["q"] * hi["R_outer"] - lo["q"] * lo["R_outer"]))
            third = next((c for c in observed if determinant(c)), None)
            if third is None:
                third = dict(lo, R_outer=lo["R_outer"] + 1, calibration_only=True)
                third["id"] = physical_id(third)
            anchors.append(third)
        holdout = next((c for c in observed if c["id"] not in {a["id"] for a in anchors}), None)
        designs[key] = dict(mode=mode, anchors=anchors, holdout=holdout)
    return designs
