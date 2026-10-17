"""Artificial metadata only: no sampling, signal, native build, or compile."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "trotterlib"))
import targeted_validation_plan as c


def by_id(plan, name):
    return next(n for n in plan["nodes"] if n["id"] == name)


def test_only_new_boundary_cells_and_not_cartesian():
    p = c.build_plan()
    cells = by_id(p, "A_SIGNAL_H8")["scope"]["candidates"]
    assert len(cells) == 16
    old = {(a, q, r, K) for a in (7, 14, 21) for q in (1, 2, 4, 8)
           for r in (1, 2, 4) for K in (2, 4)}
    assert not old & {(a["prefix"], a["q"], a["r"], a["K"]) for a in cells}
    assert sum(a["r"] == 8 for a in cells) == 3
    assert sum(a["q"] == 16 for a in cells) == 1


def test_workload_pairs_and_control_budget_not_hidden():
    w = c.planned_workload(c.build_plan())
    assert w["PR_boundary"]["native_axis_compile_ceiling"] == 256
    assert w["rare_native"]["native_axis_compiles"] == 64
    assert w["epsilon_saved"]["candidate_endpoint_rows"] == 2280
    assert w["time"]["native_axis_compile_ceiling"] == 48
    assert w["geometry"]["native_axis_compile_ceiling"] == 48
    assert w["native_axis_compile_ceiling_total"] == 422
    assert not w["budget_complete"]


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(approved=True), "PLANNING_ONLY"),
    (lambda p: p.update(scientific_dispatch_available=True), "PLANNING_ONLY"),
    (lambda p: p.update(axes=["cosine"]), "PAIRED"),
    (lambda p: p["nodes"].append(copy.deepcopy(p["nodes"][0])), "DUPLICATE_NODE"),
    (lambda p: p["nodes"][0].update(dependencies=["missing"]), "DEPENDENCY"),
    (lambda p: p["nodes"][0].update(dependencies=["A_COST_H8"]), "CYCLE"),
    (lambda p: p["nodes"][0].update(mandatory_stop=False), "STOP"),
    (lambda p: p["nodes"][0].update(production_adapter_ready=True), "UNSEALED"),
    (lambda p: p["nodes"][0]["scope"]["candidates"][0].update(R=1), "OCCURRENCE"),
    (lambda p: p["nodes"][0]["scope"]["candidates"].pop(), "SCOPE_MISMATCH"),
    (lambda p: by_id(p, "F_SIGNAL_H6")["scope"].update(
        new_approval_after_input_freeze=False), "FREEZE"),
])
def test_malformed_or_accidentally_authorized_plan_rejected(mutation, match):
    p = copy.deepcopy(c.build_plan())
    mutation(p)
    with pytest.raises(ValueError, match=match):
        c.validate_plan(p)


def test_no_preparation_or_no_separate_approval_never_ready():
    p = c.build_plan()
    ids = [n["id"] for n in p["nodes"]]
    assert not c.simulate_frontier(p, prepared=ids)["ready"]
    assert not c.simulate_frontier(p, approved=ids)["ready"]
    ready = c.simulate_frontier(p, prepared=ids, approved=ids)["ready"]
    assert {"A_SIGNAL_H8", "B_RARE_H8", "C_EPS_H8", "D_H6_S4",
            "E_TIME_SIGNAL_H8", "F_INPUTS_H6", "F_INPUTS_H8"} <= set(ready)
    assert "A_COST_H8" not in ready


def test_input_freeze_does_not_approve_geometry_science():
    p = c.build_plan()
    ids = [n["id"] for n in p["nodes"]]
    r = c.simulate_frontier(p, states={"F_INPUTS_H6": "PASS"}, prepared=ids,
                            approved=["F_INPUTS_H6"])
    assert "F_SIGNAL_H6" not in r["ready"]
    r = c.simulate_frontier(p, states={"F_INPUTS_H6": "PASS"}, prepared=ids,
                            approved=["F_SIGNAL_H6"])
    assert r["ready"] == ["F_SIGNAL_H6"]


def test_stop_blocks_descendants_without_cancelling_independent_scope():
    p = c.build_plan()
    ids = [n["id"] for n in p["nodes"]]
    r = c.simulate_frontier(p, states={"F_INPUTS_H6": "STOP", "B_RARE_H8": "RUNNING"},
                            prepared=ids, approved=ids)
    assert r["blocked"] == ["F_COST_H6", "F_INPUTS_H6", "F_SIGNAL_H6"]
    assert "F_INPUTS_H8" in r["ready"] and "C_EPS_H8" in r["ready"]
    assert "B_RARE_H8" not in r["ready"]
    assert r["scientific_executions"] == 0


@pytest.mark.parametrize("state", ["PASS", "STOP", "RUNNING"])
def test_no_one_shot_reuse(state):
    p = c.build_plan()
    r = c.simulate_frontier(p, states={"C_EPS_H8": state},
                            prepared=["C_EPS_H8"], approved=["C_EPS_H8"])
    assert not r["ready"]


def test_cli_exports_invalid_grants_and_cannot_execute(tmp_path):
    script = ROOT/"scripts/prepare_track_a_targeted_plan.py"
    out = tmp_path/"plan"
    r = subprocess.run([sys.executable, str(script), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    grant = json.loads((out/"grants_INVALID_v1.json").read_text())
    assert not grant["approved"] and not grant["valid_for_science"]
    assert all(g["command"] is None and not g["authorized"] for g in grant["grants"])
    repeat = subprocess.run([sys.executable, str(script), "--output", str(out)],
                            capture_output=True, text=True)
    assert repeat.returncode != 0
    refused = subprocess.run([sys.executable, str(script), "--execute"],
                             capture_output=True, text=True)
    assert refused.returncode == 2


def test_only_stdlib_dependencies_and_no_production_launcher():
    import ast
    tree = ast.parse((ROOT/"src/trotterlib/targeted_validation_plan.py").read_text())
    modules = []
    for item in ast.walk(tree):
        if isinstance(item, ast.Import):
            modules.extend(a.name.split(".")[0] for a in item.names)
        elif isinstance(item, ast.ImportFrom):
            modules.append(item.module.split(".")[0])
    assert set(modules) <= {"copy", "graphlib", "hashlib", "json"}
