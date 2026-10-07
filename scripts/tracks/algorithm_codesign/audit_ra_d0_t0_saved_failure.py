#!/usr/bin/env python3
"""T0: exact substitution of ONE saved failure; no solver or RA-D0 imports.

Only P1_ANCHORS:1/8:767135:minimum:T is admissible. Fixed source formulas
are independently transcribed for forensic arithmetic, never optimization.
Zero-mass groups block projection under the user's section 9.1; no inactive
representation convention is supplied. Existing inputs are never written.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import gzip
import hashlib
import json
from math import isqrt
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
S = "45cffb2aa10f9219b6cad929c3ade49fe7d36ca8"
A = "2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9"
R = "35f8b949079f15d0348bc082b916324870da7246"
SOURCE = "artifacts/track_b_ra_d0_source_review_v3/2026-10-07"
RESULT = "artifacts/track_b_ra_d0_development_result/2026-10-07/v3"
TABLE = "artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json"
POLICY = "artifacts/track_b_ra_d0_source_review_v2/2026-10-07/numerical_baseline_amendment_v2.json"
OUTPUT = "artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07"
TASK = "P1_ANCHORS:1/8:767135:minimum:T"
SCRIPT = "scripts/tracks/algorithm_codesign/audit_ra_d0_t0_saved_failure.py"
RESOURCES = ("T", "CX", "1Q")
PRECISIONS = ("1e-3", "1e-4", "1e-6")
DPS = 100  # Unchanged source exact.py / policy ell and kappa rules.


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            + "\n").encode()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def quantity(value):
    value = F(value)
    with localcontext() as ctx:
        ctx.prec = 32
        decimal = format(Decimal(value.numerator) / Decimal(value.denominator), ".25E")
    return {"exact": str(value), "decimal": decimal}


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("exact substitution dimension mismatch")
    return sum((F(x) * F(y) for x, y in zip(left, right)), F(0))


def confidence_constants(n):
    # No coefficient sqrt acquisition. Only the fixed confidence kappa rule.
    with localcontext() as ctx:
        ctx.prec = DPS
        logarithm = Decimal(10560).ln()
        ell = F(ctx.next_plus(logarithm))
    av = F(4, 3) * ell
    radicand = av * av + 8 * n * ell
    scale = 10 ** DPS
    k = isqrt(radicand.numerator * scale ** 2 // radicand.denominator)
    lower = F(k, scale)
    upper = lower if lower * lower == radicand else F(k + 1, scale)
    return ell, (av + upper) / (2 * n)


def fixed_rounding(q, y, denominator):
    if not q or any(v < 0 for v in q) or sum(q) <= 0 or y <= 0:
        raise ValueError("invalid saved nominal probability law")
    total = sum(q)
    scaled = [v / total * denominator for v in q]
    counts = [v.numerator // v.denominator for v in scaled]
    left = denominator - sum(counts)
    order = sorted(range(len(q)), key=lambda j: (-(scaled[j] - counts[j]), j))
    if not 0 <= left < len(q):
        raise ValueError("invalid largest-remainder count")
    for j in order[:left]:
        counts[j] += 1
    yy = y * denominator
    iy = (2 * yy.numerator + yy.denominator) // (2 * yy.denominator)
    if iy <= 0:
        raise ValueError("nonpositive rounded inverse normalization")
    return [F(v, denominator) for v in counts], F(iy, denominator)


def membership(groups, q, z, denominator):
    rows = []
    for group in groups:
        mass = sum((q[j] for j in group["indices"]), F(0))
        lo, hi = group["coefficient_interval"]
        latent = z[group["arm"]]
        residual = max(abs(mass - latent * lo), abs(mass - latent * hi))
        tau = (3 + hi / 2) / denominator
        midpoint = (lo + hi) / 2
        radius_contribution = abs(latent) * (hi - lo) / 2
        center_residual = abs(mass - latent * midpoint)
        if residual != center_residual + radius_contribution:
            raise ValueError("interval endpoint identity mismatch")
        rows.append({"arm": group["arm"], "prototype": group["prototype"],
                     "indices": group["indices"], "mass": str(mass),
                     "latent_z": str(latent), "coefficient_interval": list(map(str, (lo, hi))),
                     "coefficient_midpoint": str(midpoint), "coefficient_radius": str((hi - lo) / 2),
                     "residual": quantity(residual), "tau": quantity(tau),
                     "residual_over_tau": quantity(residual / tau),
                     "residual_minus_tau": quantity(residual - tau),
                     "center_residual": quantity(center_residual),
                     "interval_radius_contribution": quantity(radius_contribution),
                     "within_fixed_tau": residual <= tau})
    return rows


def law_diagnostic(data, labels, q, y, z, n, kappa, denominator, e, delta, groups):
    """One saved law certificate substitution; no physical operator evaluation."""
    lookup = {column["id"]: column for column in data["columns"]}
    residuals = []
    for degree, target in enumerate(map(F, data["target"])):
        lower = sum(q[j] * F(lookup[c]["D_intervals"][degree][0])
                    for j, (_, c) in enumerate(labels)) - y * target
        upper = sum(q[j] * F(lookup[c]["D_intervals"][degree][1])
                    for j, (_, c) in enumerate(labels)) - y * target
        residuals.append(max(abs(lower), abs(upper)))
    xi = sum(residuals)
    dq = sum(q[j] * F(lookup[c]["d_upper"]) for j, (_, c) in enumerate(labels))
    h = e * y - dq - xi
    costs = {resource: 2 * n * (sum(q[j] * F(lookup[c]["costs"][resource])
                  for j, (_, c) in enumerate(labels)) + (F(5, 2) if resource == "1Q" else 0))
             for resource in RESOURCES}
    peak = max([0] + [lookup[c].get("workspace_peak", 1)
                         for j, (_, c) in enumerate(labels) if q[j]])
    member = membership(groups, q, z, denominator)
    sampler = (sum(q) == 1 and all(v >= 0 and denominator % v.denominator == 0 for v in q)
               and y > 0 and denominator % y.denominator == 0)
    flags = {"sampler": sampler, "membership": all(v["within_fixed_tau"] for v in member),
             "mean": xi <= y * delta, "confidence": h >= kappa, "workspace": peak <= 1}
    return {"flags": flags, "all_pass": all(flags.values()), "xi": str(xi),
            "residual_by_degree": list(map(str, residuals)), "h_lower": str(h),
            "mean_margin": quantity(y * delta - xi), "confidence_margin": quantity(h - kappa),
            "implementation_bias_upper": str(dq / y), "mean_bias_upper": str(xi / y),
            "B": str(1 / y), "B_squared": str(1 / y ** 2), "resources": {r: str(v) for r, v in costs.items()},
            "workspace_peak": peak, "membership_groups": member}


def inner_lp_rows(data, labels, groups, arms, n, kappa, e, delta, denominator):
    """Independent fixed B2 numerical inner-row transcription, not a solve."""
    nq = len(labels)
    y_index, residual_start, latent_start = nq, nq + 1, nq + 5
    size = nq + 5 + len(arms)
    lookup = {c["id"]: c for c in data["columns"]}
    target = list(map(F, data["target"]))
    ymax = 2 / (sum(target) - delta)
    inequalities, equalities = [], []
    def add(category, name, terms, rhs=F(0), equality=False):
        coefficients = [F(0)] * size
        for j, value in terms.items():
            coefficients[j] = F(value)
        (equalities if equality else inequalities).append(
            {"category": category, "name": name, "coefficients": coefficients, "rhs": F(rhs)})
    add("simplex", "sum_q_equals_1", {j: 1 for j in range(nq)}, 1, True)
    add("y_upper", "y_le_ymax", {y_index: 1}, ymax)
    for k in range(4):
        upper = {j: F(lookup[c]["D_intervals"][k][1]) for j, (_, c) in enumerate(labels)}
        lower = {j: -F(lookup[c]["D_intervals"][k][0]) for j, (_, c) in enumerate(labels)}
        add("mean", "degree_" + str(k) + "_upper", upper | {y_index: -target[k], residual_start + k: -1})
        add("mean", "degree_" + str(k) + "_lower", lower | {y_index: target[k], residual_start + k: -1})
    add("mean_cap", "sum_residual_le_y_delta", {residual_start + k: 1 for k in range(4)} | {y_index: -delta})
    add("confidence", "confidence_inner", {j: F(lookup[c]["d_upper"]) for j, (_, c) in enumerate(labels)}
        | {y_index: -e} | {residual_start + k: 1 for k in range(4)}, -kappa)
    add("latent", "sum_z_equals_y", {latent_start + k: 1 for k in range(len(arms))} | {y_index: -1}, equality=True)
    for group in groups:
        lo, hi = group["coefficient_interval"]
        z_index = latent_start + arms.index(group["arm"])
        tau = (3 + hi / 2) / denominator
        name = group["arm"] + "/" + group["prototype"]
        add("numerical_membership", name + ":mass_minus_z_lo", {j: 1 for j in group["indices"]} | {z_index: -lo}, tau)
        add("numerical_membership", name + ":z_hi_minus_mass", {j: -1 for j in group["indices"]} | {z_index: hi}, tau)
    upper_bounds = [F(1)] * nq + [ymax] + [ymax * delta] * 4 + [ymax] * len(arms)
    objective = [2 * n * F(lookup[c]["costs"]["T"]) for _, c in labels] + [F(0)] * (size - nq)
    return inequalities, equalities, upper_bounds, objective


def substitute_rows(rows, primal, equality=False):
    output = []
    for index, row in enumerate(rows):
        lhs = dot(row["coefficients"], primal)
        signed = lhs - row["rhs"]
        violation = abs(signed) if equality else max(F(0), signed)
        output.append({"index": index, "category": row["category"], "name": row["name"],
                       "lhs": str(lhs), "rhs": str(row["rhs"]), "signed_residual": quantity(signed),
                       "slack": quantity(-signed), "violation": quantity(violation)})
    return output


def main():
    # Identity check is entirely Git/bytes. No runtime packages are imported.
    manifest_path = SOURCE + "/source_manifest_v3.json"
    manifest = json.loads((ROOT / manifest_path).read_bytes())
    protected = dict(manifest["critical_sha256"])
    for path in (manifest_path, SOURCE + "/authorization.json",
                 "docs/tracks/algorithm_codesign/ra_d0_execution_authorization_receipt.md"):
        protected[path] = digest(git("show", R + ":" + path))
    result_paths = git("ls-tree", "-r", "--name-only", R, RESULT).decode().splitlines()
    protected.update({p: digest(git("show", R + ":" + p)) for p in result_paths})
    for path, expected in protected.items():
        if path.lower().endswith(".npz"):
            raise PermissionError("NPZ path forbidden")
        raw = (ROOT / path).read_bytes()
        if digest(raw) != expected or raw != git("show", R + ":" + path):
            raise ValueError("fixed input bytes mismatch: " + path)
    if git("show", "-s", "--format=%P", A).decode().strip().split() != [S]:
        raise ValueError("A must have only S as parent")
    if git("show", "-s", "--format=%P", R).decode().strip().split() != [A]:
        raise ValueError("R must have only A as parent")
    remote = git("remote", "get-url", "origin").decode().strip()
    if not remote.startswith(("git@github.com:HIROMU1015/", "https://github.com/HIROMU1015/")):
        raise PermissionError("repository ownership mismatch")
    result = json.loads((ROOT / RESULT / "result.json").read_bytes())
    failure = json.loads((ROOT / RESULT / "technical_failure.json").read_bytes())
    if result["classification"] != "D0_TECHNICAL_INCONCLUSIVE" or failure["last_task"]["call_id"] != TASK:
        raise ValueError("unexpected original failure")
    records = [json.loads(line) for line in gzip.decompress((ROOT / RESULT / "certificates.jsonl.gz").read_bytes()).splitlines() if line]
    if len(records) != 1:
        raise ValueError("only one first saved certificate permitted")
    record = records[0]
    if digest(canonical(record["value"])) != record["certificate_sha256"]:
        raise ValueError("saved canonical certificate hash mismatch")
    value = record["value"]
    if (value["kind"], value["stage"], value["x"], value["n"], value["objective"]) != (
            "B2_SINGLE_RESOURCE_MINIMUM", "P1_ANCHORS", "1/8", 767135, "T"):
        raise ValueError("wrong saved task; no alternative point analysis")
    policy = json.loads((ROOT / POLICY).read_bytes())
    denominator = int(policy["denominator"])
    if denominator != 2 ** 60:
        raise ValueError("fixed denominator policy mismatch")
    e, delta = F(policy["e"]), F(policy["delta_num"])
    # Select only the registered failure context; no analysis of another x.
    data = json.loads((ROOT / TABLE).read_bytes())["tables"][value["x"]]
    profiles = [p for p in data["B0_saved_profiles"] if p["epsilon"] == "1e-3"]
    arms = [p["arm"] for p in profiles]
    labels, groups = [], []
    for profile in profiles:
        for member in profile["memberships"]:
            prototype = member["column_id"].split(":")[0]
            indices = list(range(len(labels), len(labels) + len(PRECISIONS)))
            labels.extend((profile["arm"], prototype + ":" + ep) for ep in PRECISIONS)
            groups.append({"arm": profile["arm"], "prototype": prototype, "indices": indices,
                           "coefficient_interval": tuple(map(F, member["ideal_weight_interval"]))})
    primal = list(map(F, value["solver"]["nominal_primal"]))
    nq = len(labels)
    if len(primal) != nq + 5 + len(arms):
        raise ValueError("saved primal dimension mismatch")
    q0, y0 = primal[:nq], primal[nq]
    z0 = dict(zip(arms, primal[nq + 5:]))
    if any(v < 0 for v in q0 + list(z0.values())) or y0 <= 0 or sum(z0.values()) <= 0:
        raise ValueError("invalid saved nominal law")
    ell, kappa = confidence_constants(value["n"])
    inequalities, equalities, upper_bounds, objective = inner_lp_rows(
        data, labels, groups, arms, value["n"], kappa, e, delta, denominator)
    inequality_audit = substitute_rows(inequalities, primal)
    equality_audit = substitute_rows(equalities, primal, True)
    variable_names = [arm + ":" + c for arm, c in labels] + ["y"] + ["mean_aux_" + str(k) for k in range(4)] + ["z:" + arm for arm in arms]
    bounds = [{"index": j, "name": name, "value": str(primal[j]), "lower": "0",
               "upper": str(upper_bounds[j]), "lower_violation": quantity(max(F(0), -primal[j])),
               "upper_violation": quantity(max(F(0), primal[j] - upper_bounds[j]))}
              for j, name in enumerate(variable_names)]
    dual = value["solver"]["dual_certificate"]
    nu, u = [list(map(F, v)) for v in dual["multipliers_exact"]]
    if len(nu) != len(inequalities) or len(u) != len(equalities) or any(v < 0 for v in nu):
        raise ValueError("saved dual dimension/sign mismatch")
    dual_residual = [objective[j] + sum(v * row["coefficients"][j] for v, row in zip(nu, inequalities))
                     + sum(v * row["coefficients"][j] for v, row in zip(u, equalities)) for j in range(len(primal))]
    correction = sum(min(F(0), v) * cap for v, cap in zip(dual_residual, upper_bounds))
    lower = -dot(nu, [row["rhs"] for row in inequalities]) - dot(u, [row["rhs"] for row in equalities]) + correction
    if dual_residual != list(map(F, dual["residual_exact"])) or correction != F(dual["stationarity_correction"]) or lower != F(dual["lower"]):
        raise ValueError("independent fixed inner LP reconstruction differs from saved dual")
    q3, y2 = fixed_rounding(q0, y0, denominator)
    q1 = [v / sum(q0) for v in q0]
    z2 = {arm: v * y2 / sum(z0.values()) for arm, v in z0.items()}
    states = [("N0_RAW_NOMINAL", q0, y0, z0), ("N1_Q_NORMALIZATION_ONLY", q1, y0, z0),
              ("N2_FIXED_Y_AND_Z_SCALE", q1, y2, z2), ("N3_ACTUAL_FIXED_DYADIC_Q", q3, y2, z2)]
    stages = []
    for name, q, y, z in states:
        stages.append({"stage": name, "q": list(map(str, q)), "y": str(y), "z": {r: str(v) for r, v in z.items()},
                       "simplex_residual": quantity(abs(sum(q) - 1)), "latent_sum_residual": quantity(abs(sum(z.values()) - y)),
                       "membership_groups": membership(groups, q, z, denominator)})
    transitions = []
    for before, after in zip(stages, stages[1:]):
        changes = []
        for previous, current in zip(before["membership_groups"], after["membership_groups"]):
            difference = F(current["residual"]["exact"]) - F(previous["residual"]["exact"])
            changes.append({"arm": current["arm"], "prototype": current["prototype"],
                            "residual_difference": quantity(difference),
                            "direction": "INCREASE" if difference > 0 else "DECREASE" if difference < 0 else "UNCHANGED"})
        transitions.append({"from": before["stage"], "to": after["stage"], "groups": changes})
    common3 = law_diagnostic(data, labels, q3, y2, z2, value["n"], kappa, denominator, e, delta, groups)
    saved = value["implementation_certificate"]
    for field in ("q_exact", "y_exact"):
        expected = list(map(str, q3)) if field == "q_exact" else str(y2)
        if saved[field] != expected:
            raise ValueError("fixed rounding reconstruction differs from saved law")
    for field in ("xi", "residual_by_degree", "h_lower", "implementation_bias_upper", "mean_bias_upper", "B", "B_squared", "resources", "workspace_peak"):
        if saved[field] != common3[field]:
            raise ValueError("fixed saved-law reconstruction differs: " + field)
    if saved["certified"] != common3["all_pass"]:
        raise ValueError("saved combined certificate flag mismatch")
    for reconstructed, old in zip(stages[-1]["membership_groups"], saved["membership"]["groups"]):
        if (reconstructed["arm"], reconstructed["prototype"], reconstructed["residual"]["exact"], reconstructed["tau"]["exact"], reconstructed["within_fixed_tau"]) != (
                old["arm"], old["prototype"], old["residual_upper"], old["tau"], old["certified"]):
            raise ValueError("saved membership reconstruction mismatch")
    if {arm: str(v) for arm, v in z2.items()} != saved["membership"]["latent_z"]:
        raise ValueError("saved latent witness mismatch")
    raw_member = stages[0]["membership_groups"]
    zero_groups = [g for g in raw_member if F(g["mass"]) == 0]
    shares = [{"arm": g["arm"], "prototype": g["prototype"], "shares":
               [str(q0[j] / F(g["mass"])) for j in g["indices"]]}
              for g in raw_member if F(g["mass"]) > 0]
    projection = {"schema": "ra_d0_t0_projection_diagnostic_v1", "task": TASK,
                  "diagnostic_only": True, "is_B2_minimum_or_winner": False,
                  "resource_improvement_claim": False, "zero_mass_policy": "User section 9.1: technical diagnostic failure; no new shares convention",
                  "zero_mass_groups": [{"arm": g["arm"], "prototype": g["prototype"], "mass": g["mass"], "raw_z": g["latent_z"]} for g in zero_groups],
                  "defined_nonzero_group_precision_shares": shares,
                  "latent_relative_shares": {arm: str(v / sum(z0.values())) for arm, v in z0.items()},
                  "fixed_interval_midpoints": [{"arm": g["arm"], "prototype": g["prototype"], "midpoint": g["coefficient_midpoint"]} for g in raw_member]}
    if zero_groups:
        classification = "T0_TECHNICAL_INCONCLUSIVE"
        projection.update({"status": "NOT_APPLIED_ZERO_MASS_GROUP", "technical_reason": "ZERO_MASS_GROUP_PRECISION_SHARES_UNDEFINED_UNDER_USER_9_1",
                           "projection_applications": 0, "post_projection_fixed_quantization_applications": 0,
                           "membership": "NOT_EVALUATED", "mean": "NOT_EVALUATED", "confidence": "NOT_EVALUATED", "workspace": "NOT_EVALUATED",
                           "resources": None, "blocking_issue": "Explicit zero-mass rule blocks the requested projection; no active-support-only convention was invented."})
    else:
        # This branch is unreachable for the pinned saved artifact. It is a
        # single deterministic projection, with no iterations or candidates.
        lambdas = {arm: v / sum(z0.values()) for arm, v in z0.items()}
        unnormalized = [F(0)] * nq
        for group in groups:
            mass = sum(q0[j] for j in group["indices"])
            midpoint = sum(group["coefficient_interval"]) / 2
            for j in group["indices"]:
                unnormalized[j] = lambdas[group["arm"]] * midpoint * q0[j] / mass
        total = sum(unnormalized)
        if total <= 0:
            raise ValueError("diagnostic projection normalization undefined")
        qp, yp = [v / total for v in unnormalized], 1 / total
        zp = {arm: v / total for arm, v in lambdas.items()}
        qpd, ypd = fixed_rounding(qp, yp, denominator)
        zpd = {arm: v * ypd / sum(zp.values()) for arm, v in zp.items()}
        certp = law_diagnostic(data, labels, qpd, ypd, zpd, value["n"], kappa, denominator, e, delta, groups)
        if all(v["within_fixed_tau"] for v in raw_member) and not common3["flags"]["membership"]:
            classification = "T0_QUANTIZATION_DOMINANT"
        elif all(certp["flags"].values()) and not all(v["within_fixed_tau"] for v in raw_member):
            classification = "T0_SUPPORTS_STRUCTURE_PRESERVING_NUMERICAL_REPAIR"
        elif any(not certp["flags"][flag] for flag in ("membership", "mean", "confidence")):
            classification = "T0_DEEPER_NUMERICAL_INCONSISTENCY"
        else:
            classification = "T0_TECHNICAL_INCONCLUSIVE"
        projection.update({"status": "APPLIED_ONCE_DIAGNOSTIC_ONLY", "projection_applications": 1,
                           "post_projection_fixed_quantization_applications": 1, "certificate": certp,
                           "continuous_q": list(map(str, qp)), "continuous_y": str(yp),
                           "continuous_z": {arm: str(v) for arm, v in zp.items()},
                           "fixed_q": list(map(str, qpd)), "fixed_y": str(ypd), "fixed_z": {arm: str(v) for arm, v in zpd.items()}})
    # Recheck every protected byte after arithmetic and before publishing.
    for path, expected in protected.items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("protected bytes changed during T0: " + path)
    imported_forbidden = sorted(name for name in sys.modules if name.split(".")[0] in ("numpy", "scipy", "highspy", "trottertracks", "trotterlib"))
    if imported_forbidden:
        raise PermissionError("forbidden module imported: " + str(imported_forbidden))
    counts = {name: 0 for name in ("solver_calls", "registered_LP", "synthetic_LP", "B2_minima_recomputation", "B3_calls", "Farkas_calls", "RA_D0_runner", "authorization_creation", "marker_creation_deletion_modification", "synthesis", "science", "circuit", "matrix", "trajectory", "DF_molecule_NPZ", "GPU", "retries", "repair_source_implementation")}
    inputs = {"schema": "ra_d0_t0_input_identity_v1", "source_commit": S, "authorization_commit": A,
              "result_commit": R, "task": TASK, "protected_bytes_verified_against_result_commit": True,
              "protected_sha256": protected, "marker_sha256": protected[RESULT + "/one_shot_consumed.json"],
              "certificate_gzip_sha256": protected[RESULT + "/certificates.jsonl.gz"],
              "candidate_table_sha256": protected[TABLE], "contract_sha256": protected[SOURCE + "/execution_contract_v3.json"],
              "source_manifest_sha256": protected[manifest_path], "audit_script_sha256": digest((ROOT / SCRIPT).read_bytes()),
              "formula_sources": ["src/trottertracks/algorithm_codesign/ra_d0/exact.py", "src/trottertracks/algorithm_codesign/ra_d0/lp.py", "src/trottertracks/algorithm_codesign/ra_d0/numerical.py", POLICY],
              "audit_runtime": "Python stdlib only; no NumPy/SciPy/source package imports", "original_classification": result["classification"]}
    worst_eq = max(equality_audit, key=lambda row: F(row["violation"]["exact"]))
    worst_ineq = max(inequality_audit, key=lambda row: F(row["violation"]["exact"]))
    bound_max = max([F(v[key]["exact"]) for v in bounds for key in ("lower_violation", "upper_violation")])
    nominal = {"schema": "ra_d0_t0_nominal_residual_audit_v1", "task": TASK,
               "saved_nominal_primal": value["solver"]["nominal_primal"], "q_variable_labels": labels,
               "q_raw": list(map(str, q0)), "y_raw": str(y0), "z_raw": {r: str(v) for r, v in z0.items()},
               "saved_mean_auxiliaries": list(map(str, primal[nq + 1:nq + 5])), "saved_solver_status": value["solver"]["status"],
               "saved_solver_message": value["solver"]["message"], "saved_solver_optimal_flag_reevaluated": False,
               "saved_dual_certificate": dual, "independent_LP_saved_dual_reconstruction_match": True,
               "saved_implementation_certificate": saved, "saved_resources": saved["resources"],
               "simplex_residual": quantity(abs(sum(q0) - 1)), "latent_mixture_residual": quantity(abs(sum(z0.values()) - y0)),
               "membership_groups": raw_member, "confidence_constants": {"ell_upper": str(ell), "kappa_upper": str(kappa)},
               "LP_constraint_substitution": {"equality_rows": equality_audit, "inequality_rows": inequality_audit,
                  "variable_bounds": bounds, "equality_residual_max": worst_eq["violation"], "worst_equality_row": worst_eq,
                  "inequality_violation_max": worst_ineq["violation"], "worst_inequality_row": worst_ineq,
                  "variable_bound_violation_max": quantity(bound_max), "exact_feasibility_of_saved_primal":
                   not (F(worst_eq["violation"]["exact"]) or F(worst_ineq["violation"]["exact"]) or bound_max),
                  "interpretation": "Exact rational substitution of the saved double-derived point; solver tolerances and optimal flag are unchanged."}}
    decomposition = {"schema": "ra_d0_t0_membership_stage_decomposition_v1", "task": TASK,
                     "denominator": str(denominator), "one_dyadic_unit": quantity(F(1, denominator)),
                     "stages": stages, "transitions": transitions, "N3_matches_saved_implementation_certificate": True,
                     "N3_reconstructed_saved_law_certificate": common3,
                     "coefficient_radius_identity": "residual = abs(mass-z*interval_midpoint)+abs(z)*interval_radius; algebraic interval accounting only",
                     "causal_percentage_attribution": False, "other_objective_x_n_analysis": False}
    verification = {"schema": "ra_d0_t0_verification_v1", "classification": classification,
                    "original_classification_unchanged": result["classification"], "audit_integrity_status": "PASS_EXACT_SAVED_POINT_RECONSTRUCTION",
                    "full_saved_record_canonical_hash_match": True, "independent_LP_all_saved_dual_components_match": True,
                    "all_saved_N3_common_and_membership_fields_match": True, "protected_bytes_unchanged": True,
                    "analyzed_records": 1, "task": TASK, "forbidden_modules_imported": imported_forbidden,
                    "execution_counts_this_T0": counts, "projection_applications": projection["projection_applications"],
                    "projection_blocking_issue": projection.get("blocking_issue"), "new_RA_D0_result": False,
                    "mandatory_STOP": True, "next_stage_authorized": False}
    output = ROOT / OUTPUT
    if output.exists():
        raise FileExistsError("T0 output already exists; refuse overwrite")
    output.mkdir(parents=True)
    for name, body in (("input_identity_v1.json", inputs), ("nominal_residual_audit_v1.json", nominal),
                       ("membership_stage_decomposition_v1.json", decomposition),
                       ("projection_diagnostic_v1.json", projection), ("verification_v1.json", verification)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(body, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.write("\n")
    ordinary = [stage["membership_groups"][0] for stage in stages]
    print(json.dumps({"classification": classification, "ordinary_O0_stage_residuals": [v["residual"] for v in ordinary],
                      "tau": ordinary[0]["tau"], "raw_ratio": ordinary[0]["residual_over_tau"],
                      "projection_status": projection["status"], "saved_N3_flags": common3["flags"],
                      "solver_calls": 0, "mandatory_STOP": True}, indent=2))


if __name__ == "__main__":
    main()
