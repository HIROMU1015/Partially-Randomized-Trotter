from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py"
RUNNER = ROOT / "scripts/run_pr2_matched_accuracy_m1_b1_contract.py"
DOC = ROOT / "docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md"
ARTIFACT_ROOT = ROOT / "artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30"
AUTHORIZATION = ARTIFACT_ROOT / "pr2_matched_accuracy_m1_b1_implementation_authorization_v1.json"
PLAN = ARTIFACT_ROOT / "pr2_matched_accuracy_m1_b1_zero_compute_plan_v1.json"
PLAN_SCHEMA = ARTIFACT_ROOT / "pr2_matched_accuracy_m1_b1_plan_schema_v1.json"
RESULT_SCHEMA = ARTIFACT_ROOT / "pr2_matched_accuracy_m1_b1_result_schema_v1.json"


def _load_contract():
    spec = importlib.util.spec_from_file_location("pr2_m1_b1_contract_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _m1_result() -> dict:
    path = ROOT / contract.M1_A_RESULT_RELATIVE
    return _json(path)


def _build_plan() -> dict:
    return contract.build_plan(
        m1_a_result=_m1_result(),
        m1_a_result_sha256=contract.M1_A_RESULT_SHA256,
        authorization_sha256="a" * 64,
        source_hashes={"source": "b" * 64},
        source_commit="c" * 40,
    )


def test_frozen_result_produces_exact_194_plus_16_cells() -> None:
    result = _m1_result()
    contract.validate_m1_a_result(result, sha256=contract.M1_A_RESULT_SHA256)
    random_cells, baseline_cells = contract.frozen_cells(result)
    assert len(random_cells) == 194
    assert len(baseline_cells) == 16
    assert sum(cell["method"] == "B2" for cell in random_cells) == 145
    assert sum(cell["method"] == "B3" for cell in random_cells) == 49
    assert sum(cell["method"] == "B0" for cell in baseline_cells) == 12
    assert sum(cell["method"] == "B1" for cell in baseline_cells) == 4
    assert sum(cell["accuracy_eligible"] for cell in baseline_cells) == 12


def test_plan_has_exact_bounded_compile_counts_and_no_science() -> None:
    plan = _build_plan()
    contract.validate_plan(plan)
    assert plan["resource_caps"] == {
        "random_cells": 194,
        "trajectories_per_random_cell": 32,
        "random_trajectories": 6208,
        "random_full_wrappers": 12416,
        "deterministic_or_discard_cells": 16,
        "deterministic_or_discard_full_wrappers": 32,
        "total_full_wrappers": 12448,
        "axes_per_cell_or_trajectory": 2,
        "maximum_process_workers": 6,
        "blas_threads_per_worker": 1,
        "extension_trajectories": 0,
        "maximum_trajectories_per_random_cell": 32,
    }
    assert all(value == 0 for value in plan["zero_compute_counters"].values())
    assert plan["decision"]["science_execution_authorized"] is False
    assert plan["decision"]["automatic_next_stage"] is None


def test_cosine_and_sine_share_trajectory_but_not_wrapper_cache_key() -> None:
    candidate = contract.frozen_cells(_m1_result())[0][0]
    fp = candidate["candidate_fingerprint"]
    cosine = contract.wrapper_identity(
        source_commit="d" * 40,
        candidate_fingerprint=fp,
        axis="cosine",
        trajectory_index=7,
    )
    sine = contract.wrapper_identity(
        source_commit="d" * 40,
        candidate_fingerprint=fp,
        axis="sine",
        trajectory_index=7,
    )
    assert cosine["trajectory_seed"] == sine["trajectory_seed"]
    assert contract.fingerprint(cosine) != contract.fingerprint(sine)
    assert contract.trajectory_seed(fp, 7) != contract.trajectory_seed(fp, 8)
    assert contract.occurrence_seed(
        fp, 7, outer_step=0, tail_occurrence=0, rte_step=0
    ) != contract.occurrence_seed(
        fp, 7, outer_step=0, tail_occurrence=1, rte_step=0
    )


def test_cache_identity_separates_source_candidate_axis_and_trajectory() -> None:
    cells = contract.frozen_cells(_m1_result())[0]
    first = contract.wrapper_identity(
        source_commit="1" * 40,
        candidate_fingerprint=cells[0]["candidate_fingerprint"],
        axis="cosine",
        trajectory_index=0,
    )
    variants = []
    changed_source = copy.deepcopy(first)
    changed_source["source_commit"] = "2" * 40
    variants.append(changed_source)
    changed_candidate = copy.deepcopy(first)
    changed_candidate["candidate_fingerprint"] = cells[1]["candidate_fingerprint"]
    variants.append(changed_candidate)
    changed_axis = copy.deepcopy(first)
    changed_axis["axis"] = "sine"
    variants.append(changed_axis)
    changed_trajectory = copy.deepcopy(first)
    changed_trajectory["trajectory_index"] = 1
    changed_trajectory["trajectory_seed"] = contract.trajectory_seed(
        cells[0]["candidate_fingerprint"], 1
    )
    variants.append(changed_trajectory)
    changed_compiler = copy.deepcopy(first)
    changed_compiler["compiler_identity"]["optimization_level"] = 2
    variants.append(changed_compiler)
    base = contract.fingerprint(first)
    assert all(contract.fingerprint(item) != base for item in variants)


def test_validator_rejects_cross_cell_cache_reuse_and_count_mutation() -> None:
    plan = _build_plan()
    duplicate = copy.deepcopy(plan)
    duplicate["random_cells"][1]["wrapper_cache_keys"]["cosine"][0] = (
        duplicate["random_cells"][0]["wrapper_cache_keys"]["cosine"][0]
    )
    duplicate_body = {k: v for k, v in duplicate.items() if k != "plan_fingerprint"}
    duplicate["plan_fingerprint"] = contract.fingerprint(duplicate_body)
    with pytest.raises(ValueError, match="missing or duplicated"):
        contract.validate_plan(duplicate)

    bad_count = copy.deepcopy(plan)
    bad_count["resource_caps"]["total_full_wrappers"] = 12447
    bad_count_body = {k: v for k, v in bad_count.items() if k != "plan_fingerprint"}
    bad_count["plan_fingerprint"] = contract.fingerprint(bad_count_body)
    with pytest.raises(ValueError, match="resource cap mismatch"):
        contract.validate_plan(bad_count)


def test_authorization_freezes_sources_and_forbids_execution() -> None:
    authorization = _json(AUTHORIZATION)
    assert authorization["schema_version"] == contract.AUTHORIZATION_SCHEMA_VERSION
    assert authorization["status"] == contract.STATUS
    permissions = authorization["permissions"]
    assert permissions["zero_compute_plan_generation_authorized"] is True
    for name in (
        "m1_b1_scientific_execution_authorized",
        "development_npz_load_authorized",
        "signal_reevaluation_authorized",
        "trajectory_sampling_authorized",
        "circuit_build_authorized",
        "direct_compile_authorized",
        "held_out_access_authorized",
        "additional_96_trajectories_authorized",
        "transfer_authorized",
        "s3_authorized",
    ):
        assert permissions[name] is False
    assert permissions["automatic_next_stage"] is None
    for relative, expected in authorization["required_source_hashes"].items():
        assert contract.file_sha256(ROOT / relative) == expected


def test_schemas_freeze_counts_and_post_b1_decisions() -> None:
    plan_schema = _json(PLAN_SCHEMA)
    caps = plan_schema["properties"]["resource_caps"]["properties"]
    assert caps["random_cells"]["const"] == 194
    assert caps["random_full_wrappers"]["const"] == 12416
    assert caps["total_full_wrappers"]["const"] == 12448
    result_schema = _json(RESULT_SCHEMA)
    statuses = set(result_schema["properties"]["status"]["enum"])
    assert statuses == set(contract.POST_B1_DECISIONS) | {"IMPLEMENTATION_GATE_FAILED"}
    assert result_schema["properties"]["additional_96_trajectories_executed"]["const"] is False
    assert result_schema["properties"]["held_out_accessed"]["const"] is False


def test_contract_source_and_runner_are_standard_library_only() -> None:
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "importlib",
        "json",
        "pathlib",
        "subprocess",
        "typing",
    }
    for path in (SOURCE, RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert imported <= allowed


def test_committed_zero_compute_plan_is_valid_when_present() -> None:
    if not PLAN.exists():
        pytest.skip("zero-compute plan is generated only after the source commit")
    plan = _json(PLAN)
    contract.validate_plan(plan)
    assert plan["authorization_sha256"] == contract.file_sha256(AUTHORIZATION)
    assert plan["resource_caps"]["total_full_wrappers"] == 12448
    assert all(value == 0 for value in plan["zero_compute_counters"].values())
