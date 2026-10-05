"""Saved-value PM-2 analysis, not a molecular/simulation/compile path.

All calculations below are pure stdlib bookkeeping. Production input access
is behind an explicit launch flag and commit-blob source/contract gates.
"""
from __future__ import annotations

from collections import defaultdict
import builtins
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from . import pm2_precision_contract as c

PREPARATION = "artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05"
PREPARATION_SHA = "da30a1d3b91f44dbe4847a158e86abb1302f0ea14160ca1f77c780e0cabc488e"
OUTPUT = "artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05"
SOURCE_FILES = (
    "src/trottertracks/resource_applicability/pm2_precision_contract.py",
    "src/trottertracks/resource_applicability/pm2_precision_analysis.py",
    "scripts/resource_applicability/run_pr2_pm2_precision_analysis.py",
    "scripts/resource_applicability/run_pr2_pm2_implementation_tests.py",
    "tests/tracks/resource_applicability/test_pm2_precision_analysis.py",
    "docs/research/pr2_pm2_precision_resource_contract_v1.md",
)
ZERO_SCIENCE = {k: 0 for k in c.ZERO_ACTIONS[4:]}


def install_boundary(*, forbid_saved_evidence=False):
    """Diagnostic Python file-access guard, not an OS security sandbox."""
    counter = {"protected_access_attempts": 0, "saved_evidence_access_attempts": 0}
    evidence_names = {Path(path).name for path, _ in c.INPUTS.values()}

    def wrap(function):
        def guarded(path, *args, **kwargs):
            if not isinstance(path, int):
                name = os.fsdecode(os.fspath(path))
                parts = name.replace("\\", "/").split("/")
                if name.lower().endswith((".npz", ".npy", ".pkl", ".pickle")) or ".runtime" in parts or any(p.endswith("_registry") for p in parts):
                    counter["protected_access_attempts"] += 1
                    raise AssertionError("PM-2 attempted protected data access")
                if forbid_saved_evidence and parts[-1] in evidence_names:
                    counter["saved_evidence_access_attempts"] += 1
                    raise AssertionError("implementation tests must use synthetic inputs only")
            return function(path, *args, **kwargs)
        return guarded

    builtins.open, io.open = wrap(builtins.open), wrap(io.open)
    os.open, os.stat, os.lstat, os.scandir = wrap(os.open), wrap(os.stat), wrap(os.lstat), wrap(os.scandir)
    return counter


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6)


def precision_points():
    points = [0.005 * (0.1 / 0.005) ** (i / 300) for i in range(301)]
    points[0], points[-1] = 0.005, 0.1
    return sorted(set(points + [0.05]))


def shot_accounting(bias, normalization, epsilon):
    c.require(c.finite_nonnegative(epsilon) and epsilon > 0, "invalid epsilon")
    c.require(c.finite_nonnegative(normalization) and normalization >= 1, "invalid normalization")
    c.require(set(bias) == set(c.AXES) and all(c.finite_nonnegative(x) for x in bias.values()), "invalid axis bias")
    boundary = math.sqrt(2) * max(bias.values())
    allowances = {a: epsilon / math.sqrt(2) - bias[a] for a in c.AXES}
    shots = {}
    for axis, margin in allowances.items():
        if margin <= 0 or epsilon <= math.sqrt(2) * bias[axis]:
            shots[axis] = None
        else:
            bound = 2 * normalization**2 / margin**2 * math.log(2 / 0.025)
            c.require(math.isfinite(bound), "nonfinite shot bound; no rescue")
            shots[axis] = math.ceil(bound)
    eligible = epsilon > boundary and all(v is not None for v in shots.values())
    return {"epsilon_min": boundary, "accuracy_eligible": eligible, "allowances": allowances,
            "axis_shots": shots, "N_total": sum(shots.values()) if eligible else None}


def sample_statistics(pairs, random):
    """Saved paired costs only. No resampling; unbiased covariance uses n-1."""
    n = 32 if random else 1
    c.require(len(pairs) == n, "missing cost samples; stochastic SE is not zero")
    statistics = {}
    for metric in c.METRICS:
        xs = [p["cosine"][metric] for p in pairs]
        ys = [p["sine"][metric] for p in pairs]
        c.require(all(c.finite_nonnegative(x) for x in xs + ys), "invalid saved cost sample")
        mx, my = math.fsum(xs) / n, math.fsum(ys) / n
        cc = math.fsum((x - mx)**2 for x in xs) / (n - 1) if random else 0.0
        ss = math.fsum((y - my)**2 for y in ys) / (n - 1) if random else 0.0
        cs = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1) if random else 0.0
        statistics[metric] = {"mean_cosine": mx, "mean_sine": my, "cc": cc, "ss": ss, "cs": cs}
    return statistics


def project_candidate(entry, signal, means, pairs, saved_work):
    """Project permitted fields; never follow artifact-embedded paths."""
    c.require(entry["candidate_fingerprint"] == signal["candidate_fingerprint"], "signal identity mismatch")
    c.require(entry["signal_record_fingerprint"] == c.fingerprint(signal), "signal record changed")
    random = entry["parameters"]["method"] in {"B2", "B3"}
    stats = sample_statistics(pairs, random)
    for metric in c.METRICS:
        for axis in c.AXES.values():
            c.require(c.finite_nonnegative(means[axis][metric]), "missing compiled axis mean")
            c.require(near(means[axis][metric], stats[metric]["mean_" + axis]), "saved sample/mean mismatch")
    return {**entry, "bias": dict(signal["axis_bias"]), "B": signal["normalization_multiplier"],
            "means": means, "pairs": pairs, "statistics": stats, "random": random,
            "reference_axis_shots": dict(signal["axis_shots"]), "reference_work": saved_work}


def project_inputs(values, inventory):
    rows = []
    for dataset, entries in inventory.items():
        for entry in entries:
            si, ci = entry["signal_row_index"], entry["compile_row_index"]
            if entry["signal_source"] == "m1_signal":
                signal = values["m1_signal"]["signal_records"][si]
                compiled = values["m1_compile"]["compile_map"][ci]
                axes, work = compiled["compiled_axes"], compiled["matched_accuracy_compiled_work_no_state_preparation"]
            elif entry["signal_source"] == "pm1":
                item = values["pm1"]["candidate_records"][si]
                signal, axes, work = item["signal"], item["axes"], item["work"]
            else:
                c.require(entry["signal_source"] == "m2", "unknown input projection")
                item = values["m2"]["candidate_results"][si]
                signal, means, work = item["signal"], item["axis_one_shot_compiled_means"], item["work_by_metric"]
                pairs = [{a: p["axes"][a]["metrics"] for a in c.AXES.values()}
                         for p in item["compiled"]["paired_trajectory_rows"]]
                rows.append(project_candidate(entry, signal, means, pairs, work))
                continue
            left, right = (axes[a]["retained_trajectory_records"] for a in c.AXES.values())
            c.require([(p["trajectory_index"], p["trajectory_seed"]) for p in left] ==
                      [(p["trajectory_index"], p["trajectory_seed"]) for p in right], "unpaired saved costs")
            pairs = [{"cosine": x["cost"], "sine": y["cost"]} for x, y in zip(left, right)]
            means = {a: {m: axes[a]["metric_statistics"][m]["mean"] for m in c.METRICS} for a in c.AXES.values()}
            rows.append(project_candidate(entry, signal, means, pairs, work))
    return rows


def evaluate_candidate(candidate, epsilon):
    accounting = shot_accounting(candidate["bias"], candidate["B"], epsilon)
    row = {"dataset": candidate["dataset"], "candidate_id": candidate["candidate_id"],
           "candidate_fingerprint": candidate["candidate_fingerprint"], "epsilon": epsilon,
           "epsilon_min": accounting["epsilon_min"], "accuracy_eligible": accounting["accuracy_eligible"],
           "axis_bias_real": candidate["bias"]["real"], "axis_bias_imag": candidate["bias"]["imag"],
           "normalization": candidate["B"], "allowance_real": accounting["allowances"]["real"],
           "allowance_imag": accounting["allowances"]["imag"], "N_real": accounting["axis_shots"]["real"],
           "N_imag": accounting["axis_shots"]["imag"], "N_total": accounting["N_total"],
           "primary_RZ_P0": None, "primary_SE": None, "point_frontier": False,
           "missing_reason": None if accounting["accuracy_eligible"] else "NONPOSITIVE_AXIS_HEADROOM_OR_STRICT_BOUNDARY"}
    row.update({m: None for m in c.METRICS})
    if accounting["accuracy_eligible"]:
        nc, ns = row["N_real"], row["N_imag"]
        for m in c.METRICS:
            row[m] = nc * candidate["means"]["cosine"][m] + ns * candidate["means"]["sine"][m]
            c.require(math.isfinite(row[m]), "nonfinite work; no rescue")
        row["primary_RZ_P0"] = row["rz_count"]
        # This squared centered-pair form equals the contracted covariance
        # formula but avoids cancellation to a negative variance.
        if candidate["random"]:
            stat = candidate["statistics"]["rz_count"]
            variance = math.fsum((nc * (p["cosine"]["rz_count"] - stat["mean_cosine"])
                                 + ns * (p["sine"]["rz_count"] - stat["mean_sine"]))**2
                                for p in candidate["pairs"]) / 31
            row["primary_SE"] = math.sqrt(variance / 32)
        else:
            row["primary_SE"] = 0.0
    return row


def reference_gate(candidates):
    """Must pass before any precision sweep or rankings. No altered thresholds."""
    for candidate in candidates:
        row = evaluate_candidate(candidate, 0.05)
        c.require(row["accuracy_eligible"] == candidate["reference_accuracy_eligible"], "reference eligibility mismatch")
        c.require({"real": row["N_real"], "imag": row["N_imag"]} == candidate["reference_axis_shots"], "reference integer shots mismatch")
        if row["accuracy_eligible"]:
            c.require(all(near(row[m], candidate["reference_work"][m]) for m in c.METRICS), "reference matched-work mismatch")
        else:
            c.require(not candidate["reference_work"] or all(v is None for v in candidate["reference_work"].values()), "ineligible reference work must be missing")
    return {"passed": True, "epsilon": 0.05, "candidates_checked": len(candidates),
            "integer_shots_and_eligibility_exact": True, "work_relative_tolerance": 1e-12, "work_absolute_tolerance": 1e-6}


def mark_frontier(rows):
    eligible = [r for r in rows if r["accuracy_eligible"]]
    for row in eligible:
        row["point_frontier"] = not any(
            all(other[m] <= row[m] for m in c.METRICS) and any(other[m] < row[m] for m in c.METRICS)
            for other in eligible)


def lower_envelope(rows):
    """Affine inequalities on all nonnegative P, including isolated ties."""
    eligible = [r for r in rows if r["accuracy_eligible"]]
    result = []
    for row in eligible:
        lo, hi = 0.0, math.inf
        for other in eligible:
            slope = row["N_total"] - other["N_total"]
            rhs = other["primary_RZ_P0"] - row["primary_RZ_P0"]
            if slope == 0:
                if rhs < 0:
                    hi = -1.0
                    break
            elif slope > 0:
                hi = min(hi, rhs / slope)
            else:
                lo = max(lo, rhs / slope)
        if hi >= lo:
            c.require(math.isfinite(lo) and lo >= 0 and (math.isfinite(hi) or hi == math.inf), "invalid envelope endpoint")
            result.append({"dataset": row["dataset"], "epsilon": row["epsilon"], "candidate_id": row["candidate_id"],
                           "G_RZ_P0": row["primary_RZ_P0"], "N_total": row["N_total"],
                           "P_min": lo, "P_max": None if math.isinf(hi) else hi, "boundary_tie": hi == lo})
    # Interval ends are included: shared end/start points remain ties even
    # when neither line is competitive on a zero-width interval only.
    for row in result:
        endpoints = [row["P_min"]] + ([] if row["P_max"] is None else [row["P_max"]])
        row["boundary_tie"] = row["boundary_tie"] or any(
            any(other is not row and other["P_min"] <= p and (other["P_max"] is None or p <= other["P_max"])
                for other in result) for p in endpoints)
    return sorted(result, key=lambda r: (r["P_min"], r["candidate_id"]))


def representatives(rows, by_id):
    groups = defaultdict(list)
    for row in rows:
        if row["accuracy_eligible"]:
            groups[by_id[row["candidate_id"]]["parameters"]["method"]].append(row)
    result = []
    for method, group in sorted(groups.items()):
        minimum = min(r["primary_RZ_P0"] for r in group)
        for row in group:
            if row["primary_RZ_P0"] != minimum:
                continue
            candidate = by_id[row["candidate_id"]]
            result.append({"dataset": row["dataset"], "epsilon": row["epsilon"], "method": method,
                "candidate_id": row["candidate_id"], "N_total": row["N_total"], "normalization_squared": candidate["B"]**2,
                "bias_real": candidate["bias"]["real"], "bias_imag": candidate["bias"]["imag"],
                "remaining_headroom_real": row["allowance_real"], "remaining_headroom_imag": row["allowance_imag"],
                "mean_RZ_cosine": candidate["means"]["cosine"]["rz_count"], "mean_RZ_sine": candidate["means"]["sine"]["rz_count"],
                "G_RZ_P0": row["primary_RZ_P0"]})
    return result


def analyze(candidates):
    reference = reference_gate(candidates)
    points = precision_points()
    c.require(len(points) <= 302 and len(candidates) * len(points) <= 67346, "analysis cap exceeded")
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["dataset"]].append(candidate)
    ledger, envelope, decomposition, boundaries = [], [], [], []
    for dataset, group in groups.items():
        by_id = {r["candidate_id"]: r for r in group}
        c.require(len(by_id) == len(group), "duplicate candidate within dataset")
        for candidate in group:
            boundary = math.sqrt(2) * max(candidate["bias"].values())
            boundaries.append({"dataset": dataset, "candidate_id": candidate["candidate_id"], "epsilon_min": boundary,
                               "strict_boundary": True, "inside_display_range": 0.005 <= boundary <= 0.1})
        for epsilon in points:
            rows = [evaluate_candidate(candidate, epsilon) for candidate in group]
            mark_frontier(rows)
            ledger.extend(rows)
            envelope.extend(lower_envelope(rows))
            decomposition.extend(representatives(rows, by_id))
    return {"reference_reproduction": reference, "precision_ledger.csv": ledger,
            "P_envelope.csv": envelope, "representative_decomposition.csv": decomposition,
            "eligibility_boundaries.csv": boundaries}


def render_csv(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        c.require(set(row) == set(columns), "CSV fields differ from contract")
        writer.writerow({k: "MISSING" if v is None else v for k, v in row.items()})
    return stream.getvalue()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def validate_launch(root, source_commit, execute_saved_analysis):
    c.require(execute_saved_analysis is True, "explicit saved-analysis launch required; implementation is not execution approval")
    c.require(isinstance(source_commit, str) and len(source_commit) == 40 and
              all(ch in "0123456789abcdef" for ch in source_commit), "full source commit required")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"):
        c.require(os.environ.get(key) == "1", "fixed process environment required: " + key)
    git(root, "merge-base", "--is-ancestor", c.EVIDENCE_COMMIT, source_commit)
    git(root, "merge-base", "--is-ancestor", source_commit, "HEAD")
    audit = {}
    for name in SOURCE_FILES:
        data = (root / name).read_bytes()
        c.require(data == git(root, "show", source_commit + ":" + name), "source differs from frozen commit: " + name)
        audit[name] = hashlib.sha256(data).hexdigest()
    directory = root / PREPARATION
    data = (directory / "manifest.json").read_bytes()
    c.require(hashlib.sha256(data).hexdigest() == PREPARATION_SHA, "preparation manifest changed")
    manifest = json.loads(data)
    for record in manifest["files"]:
        c.require(Path(record["path"]).name == record["path"] and record["path"].endswith(".json"), "invalid preparation file")
        blob = (directory / record["path"]).read_bytes()
        c.require(len(blob) == record["bytes"] and hashlib.sha256(blob).hexdigest() == record["sha256"], "preparation file changed")
        c.require(blob == git(root, "show", source_commit + ":" + PREPARATION + "/" + record["path"]), "preparation not in source commit")
    c.require(data == git(root, "show", source_commit + ":" + PREPARATION + "/manifest.json"), "manifest not frozen")
    settings = json.loads((directory / "contract_settings_v1.json").read_bytes())
    c.require(settings == c.contract_settings(), "frozen contract settings mismatch")
    return audit


def validate_analysis_tables(tables, candidates):
    """Coverage and null/stop semantics, in addition to the reserved schema."""
    points = precision_points()
    expected = {(r["dataset"], r["candidate_id"], e) for r in candidates for e in points}
    rows = tables["precision_ledger.csv"]
    actual = {(r["dataset"], r["candidate_id"], r["epsilon"]) for r in rows}
    c.require(len(rows) == len(actual) and actual == expected, "precision ledger coverage mismatch")
    c.require(tables["reference_reproduction"]["passed"] is True, "reference gate must pass")
    for row in rows:
        if not row["accuracy_eligible"]:
            c.require(row["N_total"] is None and row["primary_SE"] is None and not row["point_frontier"]
                      and all(row[m] is None for m in c.METRICS), "ineligible work must be null")
        else:
            c.require(row["N_total"] == row["N_real"] + row["N_imag"] and
                      all(c.finite_nonnegative(row[m]) for m in c.METRICS) and c.finite_nonnegative(row["primary_SE"]), "invalid eligible work")
    boundary_ids = {(r["dataset"], r["candidate_id"]) for r in tables["eligibility_boundaries.csv"]}
    c.require(len(boundary_ids) == len(candidates) == len(tables["eligibility_boundaries.csv"]), "boundary coverage mismatch")
    c.require(boundary_ids == {(r["dataset"], r["candidate_id"]) for r in candidates}, "boundary identity mismatch")
    for filename in c.contract_settings()["output_columns"]:
        render_csv(tables[filename], c.contract_settings()["output_columns"][filename])


def validate_summary(summary):
    """Enforce the reserved terminal schema without a scientific dependency."""
    schema = c.reserved_result_schema()
    c.require(set(summary) == set(schema["required"]), "summary field set mismatch")
    c.require(summary["schema_version"] == "track_a_pm2_precision_result_v1" and
              summary["analysis_label"] == "POSTHOC_SAVED_VALUES_ONLY", "summary identity mismatch")
    c.require(summary["status"] in (c.COMPLETE, c.FAILURE), "runner must not make a research decision")
    c.require(summary["mandatory_stop"] is True and summary["next_stage_authorized"] is False and
              summary["research_decision"] is None, "mandatory STOP cannot be relaxed")
    c.require(summary["new_science_counts"] == ZERO_SCIENCE, "new science forbidden")
    c.require(summary["preparation_manifest_sha256"] == PREPARATION_SHA, "preparation identity changed")
    c.require(summary["domain_counts"] == {"development": 218, "transfer_fixed_five": 5}, "result domain changed")
    c.require(isinstance(summary["input_identity"], dict) and isinstance(summary["output_files"], list), "invalid summary types")
    if summary["status"] == c.COMPLETE:
        c.require(summary["failure_reason"] is None and isinstance(summary["reference_reproduction"], dict)
                  and summary["reference_reproduction"].get("passed") is True, "reference gate required for complete")
        c.require(set(summary["input_identity"]) == set(c.INPUTS), "complete requires four verified inputs")
        expected = set(c.contract_settings()["future_outputs"]) - {"summary.json", "manifest.json"}
        c.require(len(summary["output_files"]) == len(expected) and set(summary["output_files"]) == expected, "complete output set mismatch")
    else:
        c.require(isinstance(summary["failure_reason"], str) and bool(summary["failure_reason"]), "failure reason required")


def report(tables):
    """Evidence summary, not an automatic scientific/research decision."""
    lines = ["# PM-2 saved-value precision resource map", "", "POSTHOC_SAVED_VALUES_ONLY. Mandatory STOP; next stage not authorized.", "",
             "Development: H4 linear 1.00 Å, STO-3G, DF rank12, 8 qubits, second-order DF-prefix PF, T=0.8,",
             "L_D=0/3/4/5/6/9/12, q=1/2/4/8, delta=0.8/0.4/0.2/0.1, registered r/K only.",
             "Transfer: used H4 1.30 Å, original M2 five configurations only. No held-out reoptimization.", "",
             "Counts below describe point estimates over fixed epsilon points, not statistical winner certification.", "",
             "| Dataset | Epsilon | Eligible | Point Pareto | Primary point-minimum ties | Intervals overlapping primary minimum |", "|---|---:|---:|---|---|---|"]
    grouped = defaultdict(list)
    for row in tables["precision_ledger.csv"]:
        grouped[row["dataset"], row["epsilon"]].append(row)
    for (dataset, epsilon), group in sorted(grouped.items()):
        eligible = [r for r in group if r["accuracy_eligible"]]
        minimum = min((r["primary_RZ_P0"] for r in eligible), default=None)
        ties = [r["candidate_id"] for r in eligible if r["primary_RZ_P0"] == minimum]
        frontier = [r["candidate_id"] for r in eligible if r["point_frontier"]]
        best = [r for r in eligible if r["primary_RZ_P0"] == minimum]
        overlap = [r["candidate_id"] for r in eligible if any(
            r["primary_RZ_P0"] - 2*r["primary_SE"] <= b["primary_RZ_P0"] + 2*b["primary_SE"] and
            b["primary_RZ_P0"] - 2*b["primary_SE"] <= r["primary_RZ_P0"] + 2*r["primary_SE"] for b in best)]
        lines.append(f"| {dataset} | {epsilon:.12g} | {len(eligible)} | {', '.join(frontier) or 'MISSING'} | {', '.join(ties) or 'MISSING'} | {', '.join(overlap) or 'MISSING'} |")
    lines += ["", "Engineering point +/- 2SE is not a formal CI or familywise winner guarantee.",
              "Inspect ledger SE before attributing a precise winner. Shared P is hypothetical RZ-equivalent preparation cost.",
              "No new signal, samples, circuits, compilation, NPZ/runtime/registry access, quantum shots or GPU operations.",
              "PM-0 attribution corrections and missing pure discard/PF decomposition remain unchanged.",
              "No general optimum, energy/RPE total cost, or automatic research conclusion is claimed.", ""]
    return "\n".join(lines)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(root, source_commit, *, execute_saved_analysis=False):
    """Future execution only. No invocation of this function on real data in tests."""
    source_audit = validate_launch(root, source_commit, execute_saved_analysis)
    output = root / OUTPUT
    c.require(not output.exists(), "fixed output already exists; no overwrite/resume")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs, candidates, tables = {}, [], None
    summary = {"schema_version": "track_a_pm2_precision_result_v1", "status": c.FAILURE,
        "analysis_label": "POSTHOC_SAVED_VALUES_ONLY", "preparation_manifest_sha256": PREPARATION_SHA,
        "input_identity": {}, "domain_counts": {"development": 218, "transfer_fixed_five": 5},
        "reference_reproduction": None, "output_files": [], "failure_reason": "analysis not completed",
        "new_science_counts": dict(ZERO_SCIENCE), "mandatory_stop": True,
        "next_stage_authorized": False, "research_decision": None}
    try:
        values, inputs = c.load_inputs(root)
        inventory = c.candidate_inventory(values)
        prepared = json.loads((root / PREPARATION / "candidate_inventory_v1.json").read_bytes())
        c.require(c.fingerprint(inventory) == prepared["inventory_fingerprint"], "candidate inventory changed")
        candidates = project_inputs(values, inventory)
        c.require(len(candidates) == 223, "frozen candidate count changed")
        tables = analyze(candidates)
        validate_analysis_tables(tables, candidates)
        for filename, columns in c.contract_settings()["output_columns"].items():
            (output / filename).write_text(render_csv(tables[filename], columns))
        claim_audit = {key: True for key in c.contract_settings()["claim_audit_required"]}
        claim_audit.update(research_decision=None, next_stage_authorized=False, mandatory_stop=True,
                           formal_CI=False, source_commit=source_commit, source_hashes=source_audit,
                           evidence_kind="LOCAL_SAVED_VALUE_POSTHOC_ANALYSIS_NOT_IMMUTABLE_CI",
                           wall_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output / "claim_audit.json", claim_audit)
        (output / "report.md").write_text(report(tables))
        summary.update(status=c.COMPLETE, reference_reproduction=tables["reference_reproduction"], failure_reason=None,
                       output_files=[f for f in c.contract_settings()["future_outputs"] if f not in {"summary.json", "manifest.json"}])
    except Exception as error:
        summary.update(failure_reason=type(error).__name__ + ": " + str(error))
    summary["input_identity"] = inputs
    validate_summary(summary)
    # Reserved schema is supplemented by validate_analysis_tables above.
    write_json(output / "summary.json", summary)
    names = summary["output_files"] + ["summary.json"]
    files = []
    for name in names:
        data = (output / name).read_bytes()
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"source_commit": source_commit, "status": summary["status"], "files": files,
                "new_science_counts": dict(ZERO_SCIENCE), "mandatory_stop": True,
                "next_stage_authorized": False, "research_decision": None,
                "partial_output_must_not_be_used": summary["status"] == c.FAILURE}
    write_json(output / "manifest.json", manifest)
    return summary
