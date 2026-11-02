"""Track A PM-1 preparation. Stdlib only; molecular paths remain literals.

Preparation is NOT execution permission. A separate, reviewed authorization
must bind an actual source commit and a sealed plan before the private boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

EVIDENCE_COMMIT = "b6e65c6123475add5e620ec1064f361378bead95"
PREPARATION_STATUS = "PM1_DISCARD_PREPARED_EXECUTION_NOT_AUTHORIZED"
COMPLETE_STATUS = "PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW"
FAILURE_STATUS = "IMPLEMENTATION_GATE_FAILED"
PLAN_VERSION = "track_a_pm1_discard_plan_v1"
RESULT_VERSION = "track_a_pm1_discard_result_v1"
AUTH_VERSION = "track_a_pm1_discard_authorization_v1"
METRICS = ("rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")
AXES = ("cosine", "sine")
COMPARATORS = ("B2-rank3-q1-r4-K2", "B2-rank3-q1-r8-K2",
               "B0-rank6-q1-r0-K0", "B1-rank12-q1-r0-K0", "B3-rank0-q8-r32-K4")
INPUTS = {
    "signal": ("artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json",
               "1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086"),
    "compile": ("artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json",
                "71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4"),
}
DEVELOPMENT_LITERAL = "artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz"
SNAPSHOT_IDENTITY = {
    "snapshot_sha256": "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a",
    "hamiltonian_hash": "de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424",
    "state_hash": "31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd",
    "state_vector_hash": "c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63",
}
COMPILER = {"qiskit_version": "1.3.0", "basis_gates": ["rz", "sx", "x", "cx"],
            "optimization_level": 1, "transpiler_seed": 17, "backend_name": None,
            "coupling_map": None, "layout_method": None, "routing_method": None}
ENVIRONMENT = {"python": "3.11.0rc1", "numpy": "1.26.4", "scipy": "1.14.1", "qiskit": "1.3.0",
               "openfermion": "1.6.1", "openfermionpyscf": "0.5", "pyscf": "2.7.0"}
THREAD_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                      "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
CAPS = {"candidate_signals": 8, "full_wrappers": 16, "development_loads": 1,
        "cpu_processes": 1, "random_trajectories": 0, "held_out_access": 0,
        "gpu_operations": 0, "quantum_shots": 0, "cache_reuse": 0, "execution_run_limit": 1}
PERMISSIONS = {"development_load": True, "eight_discard_signals": True, "sixteen_wrappers": True,
               "held_out_access": False, "random_sampling": False, "candidate_search": False,
               "resume_or_retry": False, "automatic_research_decision": False, "next_stage": False}
NEW_SOURCE_PATHS = (
    "src/trottertracks/__init__.py", "src/trottertracks/resource_applicability/__init__.py",
    "src/trottertracks/resource_applicability/pm1_discard_contract.py",
    "src/trottertracks/resource_applicability/pm1_discard_execution.py",
    "scripts/resource_applicability/run_pr2_pm1_discard_contract.py",
    "scripts/resource_applicability/run_pr2_pm1_discard.py",
    "scripts/resource_applicability/run_pr2_pm1_preparation_tests.py",
    "tests/tracks/resource_applicability/test_pm1_discard.py",
    "docs/research/pr2_pm1_nearby_discard_contract_v1.md",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_text_bytes(root, relative):
    """Filter BEFORE any open/stat/resolve. Never follow paths embedded in data."""
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts and ".runtime" not in p.parts,
            "not an allowed relative text path")
    require(p.suffix in {".py", ".json", ".md"}, "not an allowed text suffix")
    return (root / p).read_bytes()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def source_inventory(root, source_commit=None):
    """Git metadata enumeration is restricted to library Python; no data crawl."""
    commit = source_commit or EVIDENCE_COMMIT
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "full source commit required")
    paths = git(root, "ls-tree", "-r", "--name-only", commit, "--", "src/trotterlib").decode().splitlines()
    paths = sorted(set(p for p in paths if p.endswith(".py")) | set(NEW_SOURCE_PATHS))
    hashes = {}
    for relative in paths:
        data = read_text_bytes(root, relative)
        if source_commit:
            require(data == git(root, "show", f"{source_commit}:{relative}"), "source differs from commit blob")
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return hashes


def load_saved_inputs(root):
    values, audit = {}, {}
    for name, (relative, expected) in INPUTS.items():
        data = read_text_bytes(root, relative)
        require(hashlib.sha256(data).hexdigest() == expected, "saved input SHA mismatch")
        require(data == git(root, "show", f"{EVIDENCE_COMMIT}:{relative}"), "saved input differs from evidence blob")
        values[name] = json.loads(data)
        audit[name] = {"path": relative, "sha256": expected, "bytes": len(data), "commit_blob_identical": True}
    return values, audit


def candidates():
    answer = []
    for rank in (4, 5):
        for q in (1, 2, 4, 8):
            identity = {**SNAPSHOT_IDENTITY, "scope": "track_a_pm1_nearby_discard_v1",
                        "method": "B0", "mode": "discard", "rank": rank, "q": q, "r": 0, "K": 0,
                        "T": 0.8, "T_hex": float(0.8).hex(), "delta": 0.8 / q, "delta_hex": (0.8 / q).hex(),
                        "identity_policy": "extract_identity_phase", "coefficient_atol": 0.0,
                        "outer_formula": "symmetric_second_order_product_formula",
                        "construction_policy": "boundary_optimized",
                        "wrapper_semantics": "full_measured_hadamard_wrapper_without_state_preparation",
                        "compiler_identity": COMPILER}
            answer.append({"candidate_id": f"PM1-B0-rank{rank}-q{q}-r0-K0",
                           "candidate_fingerprint": fingerprint(identity), **identity})
    return answer


def frozen_references(values):
    signals = {r["candidate_id"]: r for r in values["signal"]["signal_records"]}
    costs = {r["candidate"]["candidate_id"]: r for r in values["compile"]["compile_map"]}
    targets = {canonical(s["exact_target"]) for s in signals.values()}
    require(len(targets) == 1, "full-H saved target is not common")
    refs = []
    for candidate_id in COMPARATORS:
        s, c = signals[candidate_id], costs[candidate_id]
        require(c["signal_record_fingerprint"] == fingerprint(s), "signal/cost fingerprint mismatch")
        require(s["candidate"] == c["candidate"] and s["accuracy_eligible"] is True, "saved reference candidate mismatch")
        require(all(s["candidate"][k] == v for k, v in SNAPSHOT_IDENTITY.items()), "saved reference snapshot identity mismatch")
        require(c["axis_shots"] == s["axis_shots"], "saved shots mismatch")
        refs.append({"candidate_id": candidate_id, "candidate_fingerprint": s["candidate_fingerprint"],
                     "signal_record_fingerprint": fingerprint(s), "compile_record_fingerprint": fingerprint(c),
                     "total_shots": s["total_shots"], "work": c["matched_accuracy_compiled_work_no_state_preparation"],
                     "cost_kind": "saved_32_trajectory_point_estimate" if s["candidate"]["method"] in {"B2", "B3"} else "saved_exact_deterministic"})
    return refs, json.loads(next(iter(targets)))


def make_plan(references, target, source_hashes, input_audit, *, source_commit=None):
    require(len(references) == 5 and tuple(r["candidate_id"] for r in references) == COMPARATORS,
            "exact five saved comparators required")
    if source_commit is not None:
        require(bool(re.fullmatch(r"[0-9a-f]{40}", source_commit)), "full source commit required")
    plan = {"schema_version": PLAN_VERSION, "status": PREPARATION_STATUS,
            "execution_authorized": False, "source_commit": source_commit,
            "source_binding": "COMMIT_BLOB_BOUND" if source_commit else "LOCAL_CONTENT_HASH_BOUND_AWAITING_SOURCE_COMMIT",
            "source_sha256": dict(source_hashes), "evidence_commit": EVIDENCE_COMMIT,
            "saved_input_audit": input_audit, "development_path_literal": DEVELOPMENT_LITERAL,
            "model": {"molecule": "H4 linear", "geometry_angstrom": 1.0, "basis": "STO-3G", "df_rank": 12, "system_qubits": 8},
            "accuracy": {"complex_error": 0.05, "axis_error": 0.05 / math.sqrt(2.0), "axis_alpha": 0.025},
            "saved_full_H_target": target, "candidates": candidates(), "saved_comparators": references,
            "comparison_scope": "eight_new_discard_cells_plus_five_frozen_development_comparators_only",
            "compiler": COMPILER, "environment": ENVIRONMENT, "thread_environment": THREAD_ENVIRONMENT,
            "resource_caps": CAPS, "future_permissions_required": PERMISSIONS,
            "primary": "N_real*C_cosine_RZ+N_imag*C_sine_RZ", "secondary_metrics": list(METRICS),
            "bias_semantics": "discard_plus_PF_total_against_saved_full_H_target; pure_discard/PF_not_separated",
            "compile_ineligible_for_completeness": True, "cache_policy": "no_reuse_no_resume_no_retry",
            "terminal_statuses": [COMPLETE_STATUS, FAILURE_STATUS], "mandatory_stop": True,
            "next_stage_authorized": False, "research_decision": None,
            "comparison_is_point_only_not_precision_winner": True}
    plan["plan_fingerprint"] = fingerprint(plan)
    return plan


def validate_plan(plan):
    require(plan["plan_fingerprint"] == fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"}), "plan fingerprint mismatch")
    expected = make_plan(plan["saved_comparators"], plan["saved_full_H_target"], plan["source_sha256"],
                         plan["saved_input_audit"], source_commit=plan["source_commit"])
    require(canonical(plan) == canonical(expected), "plan differs from frozen contract")


def wrapper_key(plan, candidate, axis):
    require(candidate in plan["candidates"] and axis in AXES, "unregistered candidate/axis")
    return fingerprint({"plan": plan["plan_fingerprint"], "source_commit": plan["source_commit"],
                        "source_sha256": plan["source_sha256"], "candidate": candidate["candidate_fingerprint"],
                        "axis": axis, "trajectory_index": 0, "trajectory_seed": None,
                        "compiler": plan["compiler"], "wrapper_semantics": candidate["wrapper_semantics"]})


def plan_schema():
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "required": ["schema_version", "status", "execution_authorized", "source_commit", "source_sha256", "candidates", "resource_caps", "plan_fingerprint"],
            "properties": {"schema_version": {"const": PLAN_VERSION}, "status": {"const": PREPARATION_STATUS},
                           "execution_authorized": {"const": False}, "source_commit": {"type": ["string", "null"], "pattern": "^[0-9a-f]{40}$"},
                           "candidates": {"type": "array", "minItems": 8, "maxItems": 8}, "resource_caps": {"const": CAPS}}}


def result_schema():
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "required": ["schema_version", "status", "candidate_records", "execution_audit", "mandatory_stop_reached", "next_stage_authorized", "research_decision", "result_fingerprint"],
            "properties": {"schema_version": {"const": RESULT_VERSION}, "status": {"enum": [COMPLETE_STATUS, FAILURE_STATUS]},
                           "mandatory_stop_reached": {"const": True}, "next_stage_authorized": {"const": False}, "research_decision": {"type": "null"}},
            "allOf": [{"if": {"properties": {"status": {"const": COMPLETE_STATUS}}}, "then": {"properties": {"candidate_records": {"minItems": 8, "maxItems": 8}}}}]}
