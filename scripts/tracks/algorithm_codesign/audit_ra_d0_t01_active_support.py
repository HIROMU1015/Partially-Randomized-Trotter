#!/usr/bin/env python3
"""One fixed T0.1 active-support diagnostic; Python stdlib and Fraction only.

No imports of T0/v3 kernels, numerical libraries, or solvers. No source,
authorization, original result, or consumed marker writes. All quantum
operators and circuits remain unevaluated. Only the pinned first saved
nominal point is admissible; no alternate point or projection is tried.
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
T0 = "72192b3475d59f5c56370cb4068d5659789f0ef4"
SOURCE = "artifacts/track_b_ra_d0_source_review_v3/2026-10-07"
OLD_RESULT = "artifacts/track_b_ra_d0_development_result/2026-10-07/v3"
OLD_AUDIT = "artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07"
OUTPUT = "artifacts/track_b_ra_d0_t01_active_support/2026-10-08"
TABLE = "artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json"
POLICY = "artifacts/track_b_ra_d0_source_review_v2/2026-10-07/numerical_baseline_amendment_v2.json"
SCRIPT = "scripts/tracks/algorithm_codesign/audit_ra_d0_t01_active_support.py"
PRECISIONS = ("1e-3", "1e-4", "1e-6")
RESOURCES = ("T", "CX", "1Q")


class ActiveZeroMass(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def quantity(value):
    value = F(value)
    with localcontext() as context:
        context.prec = 32
        display = format(Decimal(value.numerator) / Decimal(value.denominator), ".25E") if value else "0"
    return {"exact": str(value), "decimal": display}


def kappa_rule(n):
    # The fixed confidence rule only; no new coefficient sqrt acquisition.
    with localcontext() as context:
        context.prec = 100
        ell = F(context.next_plus(Decimal(10560).ln()))
    a = F(4, 3) * ell
    v = a * a + 8 * n * ell
    scale = 10 ** 100
    k = isqrt(v.numerator * scale ** 2 // v.denominator)
    low = F(k, scale)
    high = low if low * low == v else F(k + 1, scale)
    return ell, (a + high) / (2 * n)


def fixed_round(q, y, denominator):
    require(bool(q) and all(v >= 0 for v in q) and sum(q) > 0 and y > 0,
            "invalid law; negative q is rejected without clipping")
    scaled = [v / sum(q) * denominator for v in q]
    counts = [v.numerator // v.denominator for v in scaled]
    left = denominator - sum(counts)
    order = sorted(range(len(q)), key=lambda j: (-(scaled[j] - counts[j]), j))
    require(0 <= left < len(q), "invalid largest-remainder amount")
    for j in order[:left]:
        counts[j] += 1
    yy = y * denominator
    rounded_y_count = (2 * yy.numerator + yy.denominator) // (2 * yy.denominator)
    require(rounded_y_count > 0, "nonpositive fixed y")
    return [F(v, denominator) for v in counts], F(rounded_y_count, denominator), {
        "common_denominator": str(denominator), "q_integer_counts": counts,
        "q_counts_sum": sum(counts), "y_integer_count": rounded_y_count,
        "largest_remainder_units_assigned": left, "assigned_indices_in_order": order[:left]}


def membership(groups, q, z, denominator):
    result = []
    for group in groups:
        mass = sum(q[j] for j in group["indices"])
        lo, hi = group["interval"]
        zr = z[group["arm"]]
        residual = max(abs(mass - zr * lo), abs(mass - zr * hi))
        tau = (3 + hi / 2) / denominator
        result.append({"arm": group["arm"], "prototype": group["prototype"],
                       "mass": str(mass), "latent_z": str(zr), "coefficient_interval": list(map(str, (lo, hi))),
                       "residual_upper": quantity(residual), "tau": quantity(tau),
                       "residual_over_tau": quantity(residual / tau), "margin": quantity(tau - residual),
                       "pass": residual <= tau})
    return result


def certificate(data, labels, groups, q, y, z, n, denominator, e, delta, kappa, fixed):
    require(len(q) == len(labels) and all(v >= 0 for v in q) and y > 0,
            "certificate law invalid")
    require(all(v >= 0 for v in z.values()) and sum(z.values()) == y,
            "certificate latent mixture invalid")
    lookup = {c["id"]: c for c in data["columns"]}
    residual = []
    endpoints = []
    for k, t in enumerate(map(F, data["target"])):
        low = sum(q[j] * F(lookup[c]["D_intervals"][k][0]) for j, (_, c) in enumerate(labels)) - y * t
        high = sum(q[j] * F(lookup[c]["D_intervals"][k][1]) for j, (_, c) in enumerate(labels)) - y * t
        residual.append(max(abs(low), abs(high)))
        endpoints.append({"degree": k, "lower": str(low), "upper": str(high), "absolute_upper": str(residual[-1])})
    xi = sum(residual)
    dq = sum(q[j] * F(lookup[c]["d_upper"]) for j, (_, c) in enumerate(labels))
    h = e * y - dq - xi
    resource = {r: 2 * n * (sum(q[j] * F(lookup[c]["costs"][r]) for j, (_, c) in enumerate(labels))
                             + (F(5, 2) if r == "1Q" else 0)) for r in RESOURCES}
    peak = max([0] + [lookup[c].get("workspace_peak", 1) for j, (_, c) in enumerate(labels) if q[j] > 0])
    members = membership(groups, q, z, denominator)
    law_valid = sum(q) == 1 and (not fixed or all(denominator % v.denominator == 0 for v in q + [y]))
    flags = {"membership": all(g["pass"] for g in members), "mean": xi <= y * delta,
             "confidence": h >= kappa, "workspace": peak <= 1}
    return {"flags": flags, "all_four_pass": all(flags.values()), "probability_law_valid": law_valid,
            "dyadic_grid_required": fixed, "membership_groups": members,
            "simplex_residual": str(abs(sum(q) - 1)), "latent_sum_residual": str(abs(sum(z.values()) - y)),
            "xi_upper": quantity(xi), "residual_by_degree": list(map(str, residual)), "mean_endpoints": endpoints,
            "mean_bound": str(y * delta), "mean_margin": quantity(y * delta - xi),
            "d_dot_q": str(dq), "h_lower": str(h), "kappa_upper": str(kappa),
            "confidence_margin": quantity(h - kappa), "workspace_peak": peak,
            "resources": {r: str(v) for r, v in resource.items()}, "diagnostic_cost_only": True,
            "implementation_bias_upper": str(dq / y), "mean_bias_upper": str(xi / y),
            "B": str(1 / y), "B_squared": str(1 / y ** 2)}


def project_once(groups, q0, z0):
    require(all(v >= 0 for v in q0) and sum(q0) > 0 and all(v >= 0 for v in z0.values()) and sum(z0.values()) > 0,
            "invalid raw q/z or normalization; never clip")
    relative = {r: v / sum(z0.values()) for r, v in z0.items()}
    weights = [F(0)] * len(q0)
    shares = []
    for group in groups:
        arm = group["arm"]
        mass = sum(q0[j] for j in group["indices"])
        midpoint = sum(group["interval"]) / 2
        if relative[arm] == 0:
            # Do not evaluate q/mass or define arbitrary inactive shares.
            shares.append({"arm": arm, "prototype": group["prototype"], "case": "A_INACTIVE",
                           "raw_mass": str(mass), "midpoint": str(midpoint), "precision_shares": None})
            continue
        if mass == 0:
            raise ActiveZeroMass(arm + "/" + group["prototype"])
        require(mass > 0 and midpoint >= 0, "invalid active group")
        within = [q0[j] / mass for j in group["indices"]]
        for j, proportion in zip(group["indices"], within):
            weights[j] = relative[arm] * midpoint * proportion
        shares.append({"arm": arm, "prototype": group["prototype"], "case": "B_ACTIVE_POSITIVE_MASS",
                       "raw_mass": str(mass), "midpoint": str(midpoint), "precision_shares": list(map(str, within))})
    W = sum(weights)
    require(W > 0, "nonpositive projection normalization")
    q, y, z = [v / W for v in weights], 1 / W, {r: v / W for r, v in relative.items()}
    require(sum(q) == 1 and sum(z.values()) == y, "common normalization identity failed")
    checks = []
    for group in groups:
        mass = sum(q[j] for j in group["indices"])
        midpoint = sum(group["interval"]) / 2
        equal = mass == z[group["arm"]] * midpoint
        require(equal, "continuous membership midpoint identity failed")
        if relative[group["arm"]] == 0:
            require(z[group["arm"]] == 0 and all(q[j] == 0 for j in group["indices"]), "inactive point gained support")
            maintained = None  # No inactive shares were defined.
        else:
            raw_mass = sum(q0[j] for j in group["indices"])
            maintained = all(q[j] / mass == q0[j] / raw_mass for j in group["indices"])
            require(maintained, "active continuous precision shares changed")
        checks.append({"arm": group["arm"], "prototype": group["prototype"],
                       "midpoint_group_equality": equal, "active_precision_shares_preserved": maintained})
    return q, y, z, relative, {"W": str(W), "unnormalized_q": list(map(str, weights)),
                             "group_shares": shares, "continuous_group_identity_checks": checks}


def run_audit(contract, counters):
    # Read-only lineage / bytes check, including every original T0 output.
    t0_inputs = read_json(OLD_AUDIT + "/input_identity_v1.json")
    t0_manifest = read_json(OLD_AUDIT + "/evidence_manifest_v1.json")
    protected = dict(t0_inputs["protected_sha256"])
    protected.update({p: v["sha256"] for p, v in t0_manifest["files"].items()})
    for path in (OLD_AUDIT + "/evidence_manifest_v1.json", "AGENTS.md"):
        protected[path] = sha(git("show", T0 + ":" + path))
    for path, expected in protected.items():
        require(not path.lower().endswith(".npz"), "molecular path forbidden")
        raw = (ROOT / path).read_bytes()
        require(sha(raw) == expected and raw == git("show", T0 + ":" + path), "protected bytes mismatch: " + path)
    require(t0_inputs["source_commit"] == contract["source_commit"] and t0_inputs["authorization_commit"] == contract["authorization_commit"]
            and t0_inputs["result_commit"] == contract["result_commit"], "input lineage mismatch")
    require(git("remote", "get-url", "origin").decode().strip().startswith(("git@github.com:HIROMU1015/", "https://github.com/HIROMU1015/")), "repository ownership mismatch")
    input_identity = {"schema": "ra_d0_t01_input_identity_v1", "source_commit": contract["source_commit"],
                      "authorization_commit": contract["authorization_commit"], "result_commit": contract["result_commit"], "T0_commit": T0,
                      "only_task": contract["only_task"], "protected_sha256": protected,
                      "marker_sha256": protected[OLD_RESULT + "/one_shot_consumed.json"],
                      "certificate_gzip_sha256": protected[OLD_RESULT + "/certificates.jsonl.gz"],
                      "candidate_table_sha256": protected[TABLE], "source_contract_sha256": protected[SOURCE + "/execution_contract_v3.json"],
                      "projection_contract_sha256": sha((ROOT / OUTPUT / "projection_contract_v1.json").read_bytes()),
                      "audit_script_sha256": sha((ROOT / SCRIPT).read_bytes()), "runtime": "Python stdlib only"}
    values = [json.loads(line) for line in gzip.decompress((ROOT / OLD_RESULT / "certificates.jsonl.gz").read_bytes()).splitlines() if line]
    require(len(values) == 1 and sha(canonical(values[0]["value"])) == values[0]["certificate_sha256"], "saved record count/hash mismatch")
    value = values[0]["value"]
    require((value["kind"], value["stage"], value["x"], value["n"], value["objective"]) ==
            ("B2_SINGLE_RESOURCE_MINIMUM", "P1_ANCHORS", "1/8", 767135, "T"), "other saved task prohibited")
    data = read_json(TABLE)["tables"][value["x"]]
    policy = read_json(POLICY)
    denominator, e, delta = int(policy["denominator"]), F(policy["e"]), F(policy["delta_num"])
    require(denominator == 2 ** 60 == int(contract["denominator"]), "fixed denominator mismatch")
    labels, groups, arms = [], [], []
    for profile in data["B0_saved_profiles"]:
        if profile["epsilon"] != "1e-3":
            continue
        arms.append(profile["arm"])
        for member in profile["memberships"]:
            prototype = member["column_id"].split(":")[0]
            indices = list(range(len(labels), len(labels) + len(PRECISIONS)))
            labels.extend((profile["arm"], prototype + ":" + ep) for ep in PRECISIONS)
            groups.append({"arm": profile["arm"], "prototype": prototype, "indices": indices,
                           "interval": tuple(map(F, member["ideal_weight_interval"]))})
    primal = list(map(F, value["solver"]["nominal_primal"]))
    nq = len(labels)
    require(len(primal) == nq + 5 + len(arms), "saved primal dimension mismatch")
    q0, y0, z0 = primal[:nq], primal[nq], dict(zip(arms, primal[nq + 5:]))
    require(y0 > 0 and all(v >= 0 for v in q0 + list(z0.values())) and sum(q0) > 0 and sum(z0.values()) > 0, "invalid raw nominal point")
    t0_nominal = read_json(OLD_AUDIT + "/nominal_residual_audit_v1.json")
    t0_decomposition = read_json(OLD_AUDIT + "/membership_stage_decomposition_v1.json")
    n0 = t0_decomposition["stages"][0]
    require(list(map(str, q0)) == n0["q"] and str(y0) == n0["y"] and {r: str(v) for r, v in z0.items()} == n0["z"], "independent N0 variable reconstruction mismatch")
    raw_member = membership(groups, q0, z0, denominator)
    for new, old in zip(raw_member, n0["membership_groups"]):
        require((new["arm"], new["prototype"], new["residual_upper"]["exact"], new["tau"]["exact"], new["pass"]) ==
                (old["arm"], old["prototype"], old["residual"]["exact"], old["tau"]["exact"], old["within_fixed_tau"]), "independent N0 membership mismatch")
    ell, kappa = kappa_rule(value["n"])
    require(str(ell) == t0_nominal["confidence_constants"]["ell_upper"] and str(kappa) == t0_nominal["confidence_constants"]["kappa_upper"], "fixed confidence constants differ from T0")
    # Reconstruct OLD N3 rounding for verification, not a new diagnostic law.
    qs, ys, _ = fixed_round(q0, y0, denominator)
    counters["saved_N3_rounding_reconstructions"] += 1
    zs = {r: v * ys / sum(z0.values()) for r, v in z0.items()}
    old_cert = value["implementation_certificate"]
    require(list(map(str, qs)) == old_cert["q_exact"] and str(ys) == old_cert["y_exact"] and
            {r: str(v) for r, v in zs.items()} == old_cert["membership"]["latent_z"], "independent N3 law reconstruction mismatch")
    baseline = certificate(data, labels, groups, qs, ys, zs, value["n"], denominator, e, delta, kappa, True)
    for key in ("residual_by_degree", "h_lower", "resources", "workspace_peak", "implementation_bias_upper", "mean_bias_upper", "B", "B_squared"):
        require(baseline[key] == old_cert[key], "saved N3 field mismatch: " + key)
    require(baseline["xi_upper"]["exact"] == old_cert["xi"], "saved N3 xi mismatch")
    for new, old in zip(baseline["membership_groups"], old_cert["membership"]["groups"]):
        require((new["mass"], new["residual_upper"]["exact"], new["tau"]["exact"], new["pass"]) ==
                (old["mass"], old["residual_upper"], old["tau"], old["certified"]), "saved N3 membership mismatch")
    require(all(baseline["flags"].values()) == old_cert["certified"], "saved N3 combined flag mismatch")
    # All input/applicability checks precede the single diagnostic projection.
    relative0 = {r: v / sum(z0.values()) for r, v in z0.items()}
    for group in groups:
        if relative0[group["arm"]] > 0 and sum(q0[j] for j in group["indices"]) == 0:
            raise ActiveZeroMass(group["arm"] + "/" + group["prototype"])
    counters["projection_applications"] += 1
    qp, yp, zp, relative, projection_details = project_once(groups, q0, z0)
    continuous = certificate(data, labels, groups, qp, yp, zp, value["n"], denominator, e, delta, kappa, False)
    counters["projected_law_quantization_applications"] += 1
    qf, yf, rounding = fixed_round(qp, yp, denominator)
    zf = {r: proportion * yf for r, proportion in relative.items()}
    fixed = certificate(data, labels, groups, qf, yf, zf, value["n"], denominator, e, delta, kappa, True)
    active, inactive = [r for r, v in relative.items() if v > 0], [r for r, v in relative.items() if v == 0]
    require(active == ["ordinary"] and set(inactive) == {"PTSC_K0", "A"}, "unexpected active/inactive representations")
    require(all(qp[j] == qf[j] == 0 for j, (r, _) in enumerate(labels) if r in inactive), "inactive q changed")
    require(sum(qf) == 1 and sum(zf.values()) == yf and all((v * denominator).denominator == 1 for v in qf + [yf]), "fixed exact law identities failed")
    require(fixed["probability_law_valid"], "fixed sampler grid invalid")
    if not fixed["flags"]["membership"]:
        classification = "T01_MEMBERSHIP_STILL_FAIL"
    elif not fixed["flags"]["mean"] or not fixed["flags"]["confidence"]:
        classification = "T01_MEMBERSHIP_REPAIRED_OTHER_CONSTRAINT_FAILED"
    elif not fixed["flags"]["workspace"]:
        classification = "T01_TECHNICAL_INCONCLUSIVE"
    else:
        classification = "T01_PROJECTION_FULL_CERT_PASS"
    for path, expected in protected.items():
        require(sha((ROOT / path).read_bytes()) == expected, "protected input changed during audit: " + path)
    failed = [key for key, passed in fixed["flags"].items() if not passed]
    result = {"schema": "ra_d0_t01_projection_result_v1", "classification": classification, "task": contract["only_task"],
              "active_representations": active, "inactive_representations": inactive,
              "raw_q": list(map(str, q0)), "raw_y": str(y0), "raw_z": {r: str(v) for r, v in z0.items()},
              "relative_representation_shares": {r: str(v) for r, v in relative.items()},
              "variable_labels": labels, "projection": projection_details,
              "continuous_point": {"q": list(map(str, qp)), "y": str(yp), "z": {r: str(v) for r, v in zp.items()}},
              "fixed_point": {"q": list(map(str, qf)), "y": str(yf), "z": {r: str(v) for r, v in zf.items()}, "rounding": rounding},
              "failed_fixed_constraints": failed, "projection_applications": counters["projection_applications"],
              "quantization_applications": counters["projected_law_quantization_applications"],
              "is_B2_optimum_or_new_minimum": False, "performance_improvement_claim": False,
              "original_v3_classification_unchanged": "D0_TECHNICAL_INCONCLUSIVE", "mandatory_STOP": True, "next_stage_authorized": False}
    comparison = {"schema": "ra_d0_t01_certificate_comparison_v1", "task": contract["only_task"],
                  "ell_upper": str(ell), "kappa_upper": str(kappa), "denominator": str(denominator),
                  "points": {"SAVED_N3": baseline, "PROJECTED_CONTINUOUS": continuous, "PROJECTED_FIXED_DYADIC": fixed},
                  "resource_cost_interpretation": "point diagnostics only; never new B2 minima or performance comparison"}
    verification = {"schema": "ra_d0_t01_verification_v1", "classification": classification,
                    "status": "PASS_EXACT_DIAGNOSTIC_RECONSTRUCTION", "independent_N0_reconstruction": True,
                    "independent_N3_reconstruction": True, "ordinary_only_active": True,
                    "inactive_q_continuous_and_fixed_zero": True, "inactive_zero_division_performed": False,
                    "active_continuous_precision_shares_preserved": True, "continuous_q_sum_one": sum(qp) == 1,
                    "continuous_z_sum_y": sum(zp.values()) == yp, "continuous_midpoint_membership_equalities": True,
                    "fixed_common_denominator": str(denominator), "fixed_q_sum_one": sum(qf) == 1,
                    "fixed_q_and_y_on_dyadic_grid": True, "fixed_z_preserves_original_lambda": True,
                    "three_points_all_four_certificates_evaluated": True, "protected_bytes_unchanged": True,
                    "classification_determined_by_fixed_law_only": True, "failed_fixed_constraints": failed,
                    "diagnostic_counts": dict(counters), "technical_blocking_issue": None if classification != "T01_TECHNICAL_INCONCLUSIVE" else "workspace certificate failed",
                    "original_result_reclassification": False, "repair_source_or_new_authorization": False,
                    "mandatory_STOP": True, "next_stage_authorized": False}
    return input_identity, result, comparison, verification


def main():
    counters = {"projection_applications": 0, "projected_law_quantization_applications": 0, "saved_N3_rounding_reconstructions": 0}
    out = ROOT / OUTPUT
    names = ("input_identity_v1.json", "projection_result_v1.json", "certificate_comparison_v1.json", "verification_v1.json")
    require(not any((out / name).exists() for name in names), "T0.1 outputs exist; refuse repeat or overwrite")
    contract = read_json(OUTPUT + "/projection_contract_v1.json")
    require(contract["T0_commit"] == T0 and contract["only_task"] == "P1_ANCHORS:1/8:767135:minimum:T"
            and contract["projection_applications_max"] == contract["post_projection_quantization_applications_max"] == 1,
            "fixed projection contract mismatch")
    try:
        bodies = run_audit(contract, counters)
    except Exception as error:
        classification = "T01_NOT_APPLICABLE_ACTIVE_ZERO_MASS" if isinstance(error, ActiveZeroMass) else "T01_TECHNICAL_INCONCLUSIVE"
        details = {"classification": classification, "technical_reason": type(error).__name__ + ": " + str(error),
                   "diagnostic_counts": dict(counters), "mandatory_STOP": True, "next_stage_authorized": False,
                   "original_v3_classification_unchanged": "D0_TECHNICAL_INCONCLUSIVE", "retry": 0}
        bodies = ({"schema": "ra_d0_t01_input_identity_v1", "T0_commit": T0, "input_verification_incomplete": True,
                   "projection_contract_sha256": sha((out / "projection_contract_v1.json").read_bytes()), "audit_script_sha256": sha((ROOT / SCRIPT).read_bytes())},
                  {"schema": "ra_d0_t01_projection_result_v1", **details},
                  {"schema": "ra_d0_t01_certificate_comparison_v1", "points": {}, "incomplete_prefix_not_usable": True, **details},
                  {"schema": "ra_d0_t01_verification_v1", "status": "TECHNICAL_DIAGNOSTIC_FAILURE", **details})
    forbidden = [name for name in sys.modules if name.split(".")[0] in ("numpy", "scipy", "highspy", "trottertracks", "trotterlib")]
    require(not forbidden, "forbidden modules imported")
    counts = {key: 0 for key in ("solver_calls", "registered_LP", "synthetic_LP", "SciPy_HiGHS_linprog", "B2_minima_reacquisition",
              "B3_solve", "Farkas", "RA_D0_runner", "authorization_marker_changes", "source_v3_T0_script_changes",
              "denominator_tolerance_tau_changes", "second_best_search", "new_angle_precision_synthesis", "science", "circuit",
              "matrix", "trajectory", "DF_molecule_NPZ", "GPU", "retries", "repair_source_v4_authorization")}
    bodies[-1]["forbidden_modules_imported"] = forbidden
    bodies[-1]["execution_counts_this_T01"] = counts
    for name, body in zip(names, bodies):
        with (out / name).open("x", encoding="utf-8") as stream:
            json.dump(body, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
    points = bodies[2].get("points", {})
    print(json.dumps({"classification": bodies[1]["classification"], "point_certificates": {
        name: {"flags": value["flags"], "mean_margin": value["mean_margin"], "confidence_margin": value["confidence_margin"]}
        for name, value in points.items()}, "diagnostic_counts": counters, "solver_calls": 0,
        "RA_D0_runner": 0, "mandatory_STOP": True}, indent=2))


if __name__ == "__main__":
    main()
