from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m1_contract.py"
RUNNER = ROOT / "scripts/run_pr2_matched_accuracy_m1_contract.py"
ARTIFACT_ROOT = (
    ROOT / "artifacts/pr2_matched_accuracy_m1_contract/2026-09-29"
)
AUTHORIZATION = (
    ARTIFACT_ROOT
    / "pr2_matched_accuracy_m1_implementation_authorization_v1.json"
)
DRY_RUN = ARTIFACT_ROOT / "pr2_matched_accuracy_m1_dry_run_v1.json"
DRY_RUN_SCHEMA = (
    ARTIFACT_ROOT / "pr2_matched_accuracy_m1_dry_run_schema_v1.json"
)
RESULT_SCHEMA = ARTIFACT_ROOT / "pr2_matched_accuracy_m1_result_schema_v1.json"


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "pr2_matched_accuracy_m1_contract_test", SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_base_candidate_ledger_is_exact_unique_and_q_correct() -> None:
    candidates = contract.enumerate_base_candidates()
    assert contract.candidate_counts(candidates) == {
        "B0": 12,
        "B1": 4,
        "B2": 144,
        "B3": 48,
        "deterministic_or_discard": 16,
        "random_base": 192,
        "base_total": 208,
        "maximum_boundary_additions": 4,
        "maximum_signal_candidates": 212,
        "maximum_random_direct_compile_cells": 16,
    }
    assert len({item["candidate_id"] for item in candidates}) == 208
    assert len({item["candidate_fingerprint"] for item in candidates}) == 208
    assert all(item["q"] * item["delta"] == pytest.approx(0.8) for item in candidates)
    assert all(item["state_hash"] == contract.DEVELOPMENT_STATE_HASH for item in candidates)
    assert all(
        item["state_vector_hash"] == contract.DEVELOPMENT_STATE_VECTOR_HASH
        for item in candidates
    )


def test_occurrence_seed_is_unique_across_every_required_coordinate() -> None:
    candidate = contract.make_candidate("B2", 6, 8, 32, 4)
    seeds = {
        contract.occurrence_seed(
            candidate["candidate_fingerprint"],
            axis=axis,
            trajectory=trajectory,
            outer_step=outer_step,
            tail_occurrence=tail_occurrence,
            rte_step=rte_step,
            master_seed=contract.MASTER_SEED_INITIAL,
        )
        for axis in ("cosine", "sine")
        for trajectory in range(2)
        for outer_step in range(candidate["q"])
        for tail_occurrence in range(2)
        for rte_step in range(candidate["r"])
    }
    expected = 2 * 2 * 8 * 2 * 32
    assert len(seeds) == expected


def test_synthetic_selector_exercises_boundaries_cap_and_stop_rule() -> None:
    fixture = contract.selector_fixture(contract.enumerate_base_candidates())
    selection = fixture["selection"]
    assert fixture["fixture_only"] is True
    assert fixture["scientific_values"] is False
    assert fixture["base_proxy_record_count"] == 192
    assert fixture["boundary_request_count"] == 4
    assert fixture["all_proxy_record_count"] == 196
    assert selection["selected_count"] == 16
    assert selection["tier_counts"] == {
        "split_anchor": 4,
        "q_anchor": 4,
        "boundary_check": 4,
        "tail_challenger": 4,
        "proxy_frontier_round_robin": 0,
        "W_action_fill": 0,
    }
    assert selection["selection_limited"] is True
    assert "unselected_proxy_nondominated_candidates" in selection[
        "selection_limited_reasons"
    ]


def test_authorization_freezes_sources_and_forbids_scientific_execution() -> None:
    authorization = _json(AUTHORIZATION)
    assert authorization["schema_version"] == contract.AUTHORIZATION_SCHEMA_VERSION
    assert authorization["status"] == contract.STATUS
    permissions = authorization["permissions"]
    assert permissions["implementation_contract_and_dry_run_authorized"] is True
    assert permissions["m1_scientific_execution_authorized"] is False
    assert permissions["development_npz_load_authorized"] is False
    assert permissions["held_out_access_authorized"] is False
    assert permissions["signal_evaluation_authorized"] is False
    assert permissions["trajectory_sampling_authorized"] is False
    assert permissions["circuit_build_authorized"] is False
    assert permissions["direct_compile_authorized"] is False
    assert permissions["s3_authorized"] is False
    assert permissions["automatic_next_stage"] is None
    for relative, expected in authorization["required_source_hashes"].items():
        assert contract.file_sha256(ROOT / relative) == expected


def test_committed_dry_run_and_schemas_are_consistent() -> None:
    payload = _json(DRY_RUN)
    contract.validate_dry_run(payload)
    assert payload["authorization_sha256"] == contract.file_sha256(AUTHORIZATION)
    assert all(value == 0 for value in payload["zero_compute_counters"].values())
    assert payload["access_audit"]["development_snapshot_reads"] == 0
    assert payload["access_audit"]["held_out_path_stat_calls"] == 0
    assert payload["access_audit"]["held_out_hash_reads"] == 0
    assert payload["access_audit"]["held_out_npz_loads"] == 0

    dry_schema = _json(DRY_RUN_SCHEMA)
    assert dry_schema["properties"]["schema_version"]["const"] == payload[
        "schema_version"
    ]
    assert dry_schema["properties"]["candidate_counts"]["properties"][
        "base_total"
    ]["const"] == 208
    result_schema = _json(RESULT_SCHEMA)
    assert result_schema["properties"]["schema_version"]["const"] == (
        "pr2_matched_accuracy_m1_result_v1"
    )
    assert set(result_schema["properties"]["status"]["enum"]) == {
        "CONTINUE_TO_FROZEN_TRANSFER_REVIEW",
        "NARROW_TO_TECHNICAL_NOTE",
        "STOP_DUPLICATIVE",
        "SELECTION_LIMITED",
        "IMPLEMENTATION_GATE_FAILED",
    }


def test_contract_source_and_runner_have_standard_library_imports_only() -> None:
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "importlib",
        "json",
        "math",
        "pathlib",
        "sys",
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


def test_runner_writes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "dry-run.json"
    command = [
        sys.executable,
        str(RUNNER),
        "--authorization",
        str(AUTHORIZATION),
        "--artifact",
        str(output),
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(first.stdout)["mandatory_stop_reached"] is True
    contract.validate_dry_run(_json(output))
    second = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode != 0
    assert "Refusing to overwrite artifact" in second.stderr
