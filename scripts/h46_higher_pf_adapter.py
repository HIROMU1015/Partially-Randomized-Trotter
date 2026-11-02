"""Metadata inventory and saved-data adapter; no chemistry or native imports.

inventory() does not evaluate biases, shots, costs, means or rankings.
load_rows() is a numerical-analysis input operation, called only after approval
by the runner (or with synthetic fixtures in tests).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "495496c4e8e027b6ebd52e6835e5cafe3518502c"
DIAGNOSTIC = "7198bbde27317fb0245bdaeeb6b0966a31096e6c"
PROVIDER = "c31ef424175c02379eafebe8fc6c7548c1d9aabe"
H4_SOURCE = "687679a4ae20e4661bde1187dd8ab92ae176bd44"
H6_SOURCE = "f0dd2a9aa4e011b204c4ba6c51a050dff09bc471"
NATIVE = Path("artifacts/resource_applicability/track_a_h46_new2_native/2026-10-18")
SIGNAL = Path("artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18")
PREFIX = {"H4": 3, "H6": 5}
RANK = {"H4": 12, "H6": 19}
GROUP_Q = {"S2": 4, "Y4": 2, "N2": 1}
GROUP_LABEL = {"S2": "2nd", "Y4": "4th", "N2": "4th(new_2)"}
INPUT_HASH = {"H4": "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a",
              "H6": "99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005"}


def kernel():
    # Avoid package __init__, which eagerly imports chemistry dependencies.
    spec = importlib.util.spec_from_file_location("higher_pf_pure", ROOT/"src/trotterlib/higher_pf_readiness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_blob(root, commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)


def checked_json(root, path, commit=BASE):
    data = (root/path).read_bytes()
    if data != git_blob(root, commit, path):
        raise ValueError("SAVED_INPUT_BYTES_MISMATCH:"+str(path))
    return json.loads(data)


def inventory(root=ROOT):
    """Strictly metadata: identifies available groups, not eligible/winning q."""
    k = kernel()
    bindings, coverage = [], []
    def read(path, commit=BASE):
        value = checked_json(root, path, commit)
        data = (root/path).read_bytes()
        bindings.append({"path": str(path), "commit": commit, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
        return value
    completion = read(NATIVE/"final_completion_audit.json")
    if completion["all_authoritative_native_exits_zero_and_reaped"] is not True:
        raise ValueError("NATIVE_COMPLETION_NOT_ACCEPTED")
    profile = read(NATIVE/"environment_compiler_profile.json")
    if profile["new_native_series"]["control"] != "ordinary":
        raise ValueError("CONTROL_SERIES")
    closures = {}
    for name, commit in (("source_closure.json", H4_SOURCE), ("parallel_H6_source_closure.json", H6_SOURCE)):
        closure = read(NATIVE/name)
        for path, expected in closure["paths"].items():
            actual = hashlib.sha256(git_blob(root, commit, path)).hexdigest()
            if actual != expected:
                raise ValueError("SCIENCE_SOURCE_CLOSURE:"+path)
        closures[name] = {"SOURCE": commit, "verified_paths": len(closure["paths"])}
    provider_paths = {}
    for system in PREFIX:
        signal = read(SIGNAL/f"{system}_run02_results.json", DIAGNOSTIC)
        conditions = signal["conditions"]
        if (conditions["rank"], conditions["T"], conditions["geometry_angstrom"], conditions["basis"]) != (RANK[system], .8, 1., "STO-3G"):
            raise ValueError("SIGNAL_CONDITIONS")
        for short, label in GROUP_LABEL.items():
            group_id = f"{system}_{short}"
            cost = read(NATIVE/f"{group_id}_final_result.json")
            if cost["SOURCE"] != (H4_SOURCE if system == "H4" else H6_SOURCE) or cost["provider_SOURCE"] != PROVIDER:
                raise ValueError("NATIVE_SOURCE")
            if cost["group"] != {"id": group_id, "system": system, "label": label,
                                 "prefix": PREFIX[system], "q": GROUP_Q[short]}:
                raise ValueError("NATIVE_GROUP")
            c = cost["conditions"]
            if (c["rank"], c["T"], c["K"], c["geometry_angstrom"], c["basis"], c["total_RTE_steps"]) != (RANK[system], .8, 4, 1., "STO-3G", GROUP_Q[short]*32):
                raise ValueError("NATIVE_CONDITIONS")
            if cost["forced_events_enter_cost_sample"] is not False or len(cost["records"]) != 16:
                raise ValueError("NATIVE_SAMPLE_SCHEMA")
            for record in cost["records"]:
                if record["compiler_settings_hash"] != k.PROFILE:
                    raise ValueError("COMPILER_PROFILE")
            imported = read(NATIVE/f"{group_id}_imported_provider_closure.json")
            for module, item in imported.items():
                path, sha = item["path"], item["sha256"]
                if path in provider_paths and provider_paths[path] != sha:
                    raise ValueError("PROVIDER_CLOSURE_DISAGREEMENT")
                provider_paths[path] = sha
            for q in k.Q_GRID:
                matching = [r for r in signal["records"] if (r["prefix"], r["label"], r["q"]) == (PREFIX[system], label, q)]
                if len(matching) != 1 or len([r for r in matching[0]["finite_rows"] if r["cutoff"] == 4]) != 1:
                    raise ValueError("FINITE_SIGNAL_COVERAGE")
                coverage.append({"candidate_id": f"{group_id}_q{q}", "system": system,
                                 "formula_id": k.FORMULAS[label], "q": q,
                                 "signal_K4_saved": True,
                                 "native_cost": "MEASURED_N8_EXPLORATION_ONLY" if q == GROUP_Q[short] else "MISSING",
                                 "eligibility": "NOT_EVALUATED", "cost_ranking": "NOT_EVALUATED"})
    for path, expected in provider_paths.items():
        if hashlib.sha256(git_blob(root, PROVIDER, path)).hexdigest() != expected:
            raise ValueError("PROVIDER_CLOSURE:"+path)
    for name in ("input_binding.json", "coefficient_order_audit.json", "coefficient_binary64_audit.json", "plan.json", "parallel_H6_plan.json"):
        read(NATIVE/name)
    read(SIGNAL/"input_binding.json", DIAGNOSTIC)
    inputs = checked_json(root, NATIVE/"input_binding.json")
    original_inputs = []
    for name, item in inputs.items():
        data = git_blob(root, item["original_commit"], item["original_path"])
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("ORIGINAL_SAVED_INPUT_BINDING:"+name)
        original_inputs.append({"role": name, **item, "git_blob_verified": True})
    for system in PREFIX:
        if inputs[f"{system}_snapshot.npz"]["sha256"] != INPUT_HASH[system]:
            raise ValueError("SYSTEM_INPUT_BINDING")
    return {"schema": "higher_pf_metadata_inventory_v1", "base": BASE, "provider_SOURCE": PROVIDER,
            "source_closures": closures, "provider_blob_paths_verified": len(provider_paths),
            "bindings": bindings, "original_inputs": original_inputs, "coverage": coverage,
            "counts": {"signal_groups": len(coverage), "saved_cost_groups": 6, "missing_cost_groups": 18},
            "new_science_calls": 0, "real_reanalysis_calls": 0,
            "eligibility_or_ranking_computed": False}


def load_rows(root=ROOT):
    """Future approved B: project verified original bytes into pure-kernel schema."""
    metadata = inventory(root)
    k = kernel()
    rows = []
    for system in PREFIX:
        signal = checked_json(root, SIGNAL/f"{system}_run02_results.json", DIAGNOSTIC)
        for short, label in GROUP_LABEL.items():
            group_id = f"{system}_{short}"
            native = checked_json(root, NATIVE/f"{group_id}_final_result.json")
            for q in k.Q_GRID:
                record = next(r for r in signal["records"] if (r["prefix"], r["label"], r["q"]) == (PREFIX[system], label, q))
                finite = next(f for f in record["finite_rows"] if f["cutoff"] == 4)
                identity = {"SOURCE": native["SOURCE"], "input_sha256": INPUT_HASH[system],
                            "preparation_hash": native["preparation_hash"], "compiler_settings_hash": k.PROFILE,
                            "series": k.SERIES, "control": "ordinary", "formula_id": k.FORMULAS[label],
                            "system": system, "prefix": PREFIX[system], "q": q, "stage": "EXPLORATION"}
                records = []
                if q == GROUP_Q[short]:
                    for saved in native["records"]:
                        records.append({**identity, "replica_id": str(saved["sample"]), "axis": saved["axis"],
                                        "event_digest": saved["event_digest"], "seed_namespace": f"legacy:{H4_SOURCE}:{group_id}",
                                        "forced_event": False, "cost_kind": "MEASURED", **saved["metrics"]})
                rows.append({"candidate_id": f"{group_id}_q{q}", "identity": identity,
                             "total_signed_bias": finite["total_signed_bias"], "log_B": finite["log_B"],
                             "signal_evidence": f"{DIAGNOSTIC}:{SIGNAL}/{system}_run02_results.json#{label}/q{q}/K4",
                             "cost_kind": "MEASURED" if records else "MISSING", "records": records})
    return rows, metadata
