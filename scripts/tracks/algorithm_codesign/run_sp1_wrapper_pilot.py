#!/usr/bin/env python3
"""SP-1 static plan or separately authorized one-shot stored-sequence pilot."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))

from trottertracks.algorithm_codesign.synthesis_placement.wrapper_sequence import (
    fusion_audit, registered_domain, selected,
)
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import (
    sha, git, verify_source, verify_launch, verify_runtime, consume_marker,
    BudgetGuard, serialize_result,
)
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_result import validate_result

PREPARATION = ROOT/"artifacts/track_b_sp1_wrapper_source/2026-10-06"
CONTRACT = PREPARATION/"contract_v1.json"


def static_plan(root, contract, source_commit=None):
    """Read static angle/key ledger only. No coefficients, G, or signals scored."""
    if source_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise PermissionError("plan source must be a full commit SHA")
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_adapter import StoredLibrary
    audit = fusion_audit(contract)
    ref = contract["saved_input"]
    library = StoredLibrary((root/ref["path"]).read_bytes(), ref["sha256"])
    keys = set()
    for wrapper in registered_domain(contract):
        for path in wrapper["paths"]:
            for native in path["post"]:
                if native.generator == "II":
                    continue
                keys.add(library.baseline_key(native.angle))
                for mask in contract["domain"]["masks"]:
                    if selected(native, mask) and native.angle in library.interpolations:
                        keys.update(f"pi:{k}/4" for k in library.interpolations[native.angle]["notch_indices"])
    if source_commit is not None:
        manifest = verify_source(root, contract)
        manifest_blob = subprocess.check_output(["git", "-C", str(root), "show",
                                                source_commit+":"+contract["source_manifest_path"]])
        if hashlib.sha256(manifest_blob).hexdigest() != sha(root/contract["source_manifest_path"]):
            raise PermissionError("source manifest differs from the fixed commit")
        for relative, digest in manifest["critical_sha256"].items():
            blob = hashlib.sha256(subprocess.check_output(
                ["git", "-C", str(root), "show", source_commit+":"+relative])).hexdigest()
            if blob != digest:
                raise PermissionError("source commit does not contain reviewed critical bytes")
    return {"status": "SP1_STATIC_PLAN_AWAITING_SOURCE_REVIEW",
            "source_commit": source_commit, "contract_sha256": sha(root/CONTRACT.relative_to(ROOT)),
            "wrappers": 12, "mask_rows": 48, "axis_rows": 96, "outer_paths": 16,
            "stored_synthesis_keys_needed": sorted(keys),
            "stored_input_sha256": ref["sha256"],
            "fusion_audit": {k: v for k, v in audit.items() if k != "paths"},
            "new_synthesis_calls": 0, "resource_signal_evaluations": 0,
            "science_execution_authorized": False, "mandatory_STOP": True,
            "next_stage_authorized": False}


def _point(raw):
    import mpmath as mp
    return {"Re": mp.nstr(mp.re(raw), 70), "Im": mp.nstr(mp.im(raw), 70)}


def perform_pilot(root, contract, result, guard):
    """The sole science entry; caller must have bound approval and consumed marker."""
    import mpmath as mp
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_adapter import (
        StoredLibrary, COST_METRIC, resource_row, diagnostic_signal, bounds,
        spec_record, classify_ratio, conditional_record,
    )
    mp.mp.dps = contract["diagnostics"]["mpmath_dps"]
    library = StoredLibrary((root/contract["saved_input"]["path"]).read_bytes(),
                            contract["saved_input"]["sha256"])
    result.update(cost_metric=COST_METRIC, native_spec_library={}, stored_sequence_library={}, rows=[])
    matrix_cache = {}
    wrappers = registered_domain(contract)
    for wrapper in wrappers:
        baseline_g = None
        for mask in contract["domain"]["masks"]:
            guard.check()
            path_specs, path_records = [], []
            for path in wrapper["paths"]:
                specs, identifiers = [], []
                for native in path["post"]:
                    spec = library.spec(native, mask)
                    specs.append(spec)
                    serialized = spec_record(spec)
                    identifier = hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest()
                    result["native_spec_library"].setdefault(identifier, serialized)
                    identifiers.append(identifier)
                    for key in spec["keys"]:
                        if key is not None:
                            # Bind sequence bytes/identity/count/error; do not copy old J scores.
                            result["stored_sequence_library"].setdefault(key, library.sequences[key])
                path_specs.append(specs)
                path_records.append({"path_id": path["path_id"], "probability": str(path["probability"]),
                    "outer_weight": str(path["outer_weight"]), "native_spec_ids_in_order": identifiers,
                    "native_rotations": len(path["post"]),
                    "selected_native_rotations": sum(selected(g, mask) for g in path["post"]),
                    **conditional_record(path, specs, contract)})
            profile, u, axis = resource_row(wrapper["paths"], path_specs, contract)
            diagnostics = diagnostic_signal(wrapper["paths"], path_specs, library, matrix_cache)
            if not all(mp.isfinite(v) for v in (diagnostics["ideal"], diagnostics["finite"],
                    diagnostics["trace_residual"], *diagnostics["axis_residual"].values())):
                raise ArithmeticError("nonfinite point diagnostic; no retry")
            total_bias = profile["synthesis_bias_upper"]+u
            slack = mp.mpf(contract["diagnostics"]["residual_slack"])
            error_bound = mp.mpf(total_bias.numerator)/total_bias.denominator
            if diagnostics["trace_residual"] > slack or any(
                    e > error_bound+slack for e in diagnostics["axis_residual"].values()):
                raise ArithmeticError("point semantic diagnostic inconsistent with saved guard; no retry")
            eligible = axis["status"] == "ELIGIBLE"
            # Same norm-one observable and bias bounds give identical Re/Im budgets.
            g = F(contract["metric"]["batch_init_T"])+2*axis["expected_total_T"] if eligible else None
            axes = []
            for name in contract["domain"]["axes"]:
                axes.append({"axis": name, "status": axis["status"], "shots_sufficient": axis["shots"],
                    "margin_lower": bounds(axis["margin"])["lo"],
                    "alpha_exact": contract["metric"]["alpha_axis"],
                    "epsilon_axis_lower": contract["metric"]["epsilon_axis_lower"],
                    "second_moment": bounds(profile["second_moment"]), "range_upper": bounds(profile["range"])["hi"],
                    "expected_C_T_add": bounds(profile["expected_T_count"]),
                    "joint_E_W2_C_T_add": bounds(profile["joint_weighted_T"]),
                    "synthesis_bias_upper": bounds(profile["synthesis_bias_upper"])["hi"],
                    "coefficient_numeric_bias_upper": bounds(u)["hi"]})
            if mask == "NONE":
                baseline_g = g
            ratio = bounds(g/baseline_g) if g is not None and baseline_g is not None and baseline_g > 0 else None
            classification = (axis["status"] if not eligible else
                "BASELINE" if mask == "NONE" else
                "REFERENCE_INELIGIBLE" if baseline_g is None else
                "ZERO_COST_REFERENCE_NO_POSITIVE_WITNESS" if baseline_g == 0 else classify_ratio(ratio))
            duplicate = ("NONE" if mask == "R" else "D" if mask == "DR" else None) if wrapper["template"] in ("A", "B") else None
            result["rows"].append({"wrapper_id": wrapper["wrapper_id"], "template": wrapper["template"],
                "n": wrapper["n"], "mask": mask, "paths": path_records, "axes": axes,
                "eligible": eligible, "G_T_add": bounds(g) if g is not None else None,
                "G_T_add_over_NONE": ratio, "classification": classification,
                "duplicate_of_mask": duplicate,
                "independent_comparison": bool(mask != "NONE" and duplicate is None),
                "count_as_independent_positive": bool(classification == "MATERIAL_GAIN" and
                                                       mask != "NONE" and duplicate is None),
                "placement_evidence_scope": "C only" if wrapper["template"] == "C" else "accumulation control",
                "diagnostic_point_only": {"ideal_signal": _point(diagnostics["ideal"]),
                    "finite_signal": _point(diagnostics["finite"]),
                    "axis_residual": {a: mp.nstr(e, 70) for a, e in diagnostics["axis_residual"].items()},
                    "trace_residual": mp.nstr(diagnostics["trace_residual"], 70),
                    "used_for_primary_shots_or_selection": False}})
            guard.check()
    if len(result["rows"]) != 48 or sum(len(r["axes"]) for r in result["rows"]) != 96:
        raise RuntimeError("registered result inventory incomplete")
    result["summary"] = {"mask_rows": 48, "axis_rows": 96,
        "eligible_rows": sum(r["eligible"] for r in result["rows"]),
        "independent_material_gain_rows": sum(r["classification"] == "MATERIAL_GAIN" and
            r["count_as_independent_positive"] for r in result["rows"]),
        "placement_C_material_gain_rows": sum(r["template"] == "C" and
            r["classification"] == "MATERIAL_GAIN" for r in result["rows"]),
        "research_GO": None, "automatic_next_stage": None}
    result["status"] = "SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW"


def run():
    # This check precedes matrix construction, coefficients/G, data scoring and marker.
    contract, auth, head = verify_launch(ROOT, CONTRACT)
    guard = BudgetGuard(contract["caps"])
    runtime = verify_runtime(ROOT, contract)
    plan = static_plan(ROOT, contract, auth["source_commit"])
    audit = json.loads((ROOT/contract["static_fusion_audit_path"]).read_text())
    directory = ROOT/contract["result_directory"]
    receipt = {"source_commit": auth["source_commit"], "authorization_commit": head,
        "authorization_sha256": sha(ROOT/contract["authorization_path"]),
        "contract_sha256": sha(CONTRACT), "stored_input_sha256": contract["saved_input"]["sha256"],
        "tool_identity_sha256": contract["tool_identity"]["sha256"],
        "runs": 1, "retries": 0, "mandatory_STOP": True, "next_stage_authorized": False,
        "explicit_execution_instruction": auth["explicit_execution_instruction"]}
    result = {**receipt, "status": "INCONCLUSIVE_MANDATORY_STOP_NO_RETRY", "rows": [],
        "runtime_identity": runtime, "static_plan": plan,
        "fusion_audit": audit,
        "science_scope": "synthetic development/mechanism; no actual finite RTE or DF",
        "actual_compiled_wrapper_T_claim": False, "synthesis_calls": 0, "sampling_calls": 0,
        "GPU_query_use": 0, "next_stage_authorized": False}
    guard.check()  # Preparation is included in the observed run budget.
    marker = consume_marker(directory, receipt)
    # The cap covers all evidence bytes, including the exclusive one-shot marker.
    remaining_output = contract["caps"]["output_bytes"]-marker.stat().st_size
    try:
        with guard:
            perform_pilot(ROOT, contract, result, guard)
            validate_result(result, contract)
            guard.check()
            result["resources"] = guard.usage()
            payload = serialize_result(result, remaining_output)
            guard.check()
            result["resources"] = guard.usage()
            result["resource_snapshot_scope"] = "after model and first serialization; before final serialization/persistence"
            payload = serialize_result(result, remaining_output)
            guard.check()
    except Exception as error:
        result.update(status="INCONCLUSIVE_MANDATORY_STOP_NO_RETRY",
                      failure=f"{type(error).__name__}: {error}"[:1000], resources=guard.usage())
        try:
            validate_result(result, contract)
            payload = serialize_result(result, remaining_output)
        except Exception:
            payload = serialize_result({**receipt, "status": "INCONCLUSIVE_MANDATORY_STOP_NO_RETRY",
                "failure": result["failure"], "partial_rows_not_saved": True,
                "completed_rows_before_failure": len(result.get("rows", [])),
                "resources": guard.usage()}, remaining_output)
    with (directory/"result.json").open("x") as stream:
        stream.write(payload)
    print(json.dumps({"status": result["status"], "mandatory_STOP": True,
                      "result": str(directory/"result.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--source-commit", help="read-only source identity for plan")
    args = parser.parse_args()
    try:
        if args.mode == "plan":
            contract = json.loads(CONTRACT.read_text())
            print(json.dumps(static_plan(ROOT, contract, args.source_commit), indent=2))
        else:
            if args.source_commit:
                raise PermissionError("run source is bound only by separate authorization")
            run()
    except Exception as error:
        parser.exit(2, f"SP-1 launch rejected: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
