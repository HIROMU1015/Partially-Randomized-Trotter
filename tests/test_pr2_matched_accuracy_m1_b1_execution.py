from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
import trotterlib.pr2_matched_accuracy_m1_b1_execution as execution
from trotterlib.pr2_matched_accuracy_m1_b1_contract import (
    M1_A_RESULT_RELATIVE,
    M1_A_RESULT_SHA256,
    file_sha256,
    load_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _m1_result() -> dict:
    return load_json(ROOT / M1_A_RESULT_RELATIVE)


def _plan(source_commit: str = "a" * 40) -> dict:
    return execution.build_execution_plan(
        m1_a_result=_m1_result(),
        m1_a_result_sha256=M1_A_RESULT_SHA256,
        source_commit=source_commit,
        source_hashes={"execution.py": "b" * 64},
    )


def test_source_bound_plan_freezes_actual_benchmark_seeds_and_12448_keys() -> None:
    plan = _plan()
    execution.validate_execution_plan(plan)
    assert len(plan["random_cells"]) == 194
    assert len(plan["baseline_cells"]) == 16
    assert plan["resource_caps"]["total_full_wrappers"] == 12_448
    assert plan["resource_caps"]["maximum_process_workers"] == 6
    cell = plan["random_cells"][0]
    assert cell["sampled_trajectory_seeds"] == execution.expected_trajectory_seeds(
        cell["request_master_seed"], cell["q"]
    )
    assert cell["wrapper_cache_keys"]["cosine"] != cell["wrapper_cache_keys"]["sine"]
    assert plan["execution_authorized"] is False
    assert plan["research_decision_automation_authorized"] is False


def test_actual_execution_source_commit_changes_every_task_identity() -> None:
    first = _plan("1" * 40)
    second = _plan("2" * 40)
    assert first["plan_fingerprint"] != second["plan_fingerprint"]
    assert first["random_cells"][0]["task_fingerprint"] != second["random_cells"][0]["task_fingerprint"]
    assert first["random_cells"][0]["wrapper_cache_keys"] != second["random_cells"][0]["wrapper_cache_keys"]


def test_plan_rejects_cross_cell_cache_identity_reuse() -> None:
    plan = _plan()
    corrupted = copy.deepcopy(plan)
    corrupted["random_cells"][1]["wrapper_cache_keys"]["cosine"][0] = (
        corrupted["random_cells"][0]["wrapper_cache_keys"]["cosine"][0]
    )
    body = {key: value for key, value in corrupted.items() if key != "plan_fingerprint"}
    corrupted["plan_fingerprint"] = execution.fingerprint(body)
    with pytest.raises(ValueError, match="missing or duplicated"):
        execution.validate_execution_plan(corrupted)


def test_checkpoint_requires_exact_task_fingerprint(tmp_path: Path) -> None:
    cell = _plan()["random_cells"][0]
    job = execution.M1B1CompileJob(0, cell, object(), str(tmp_path / "cache.sqlite3"))  # type: ignore[arg-type]
    result = {
        "candidate_fingerprint": cell["candidate_fingerprint"],
        "task_fingerprint": cell["task_fingerprint"],
    }
    path = tmp_path / "checkpoint.json"
    execution._checkpoint(path, job, result)
    assert execution._read_checkpoint(path, cell) == result
    changed = copy.deepcopy(cell)
    changed["task_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="task identity mismatch"):
        execution._read_checkpoint(path, changed)


def test_result_contract_stops_at_compile_map_review() -> None:
    schema = json.loads(
        (ROOT / "artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/"
         "pr2_matched_accuracy_m1_b1_result_schema_v2.json").read_text(encoding="utf-8")
    )
    assert set(schema["properties"]["status"]["enum"]) == {
        execution.COMPLETE_STATUS,
        execution.FAILURE_STATUS,
    }
    assert schema["properties"]["research_decision"] == {"type": "null"}
    serialized = json.dumps(schema, sort_keys=True)
    assert all(decision not in serialized for decision in execution.RESEARCH_DECISIONS)


def test_authorization_gate_requires_exact_permissions_caps_and_source_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    schema_path = "result_schema_v2.json"
    schema_sha = "c" * 64
    authorization = {
        "schema_version": execution.AUTHORIZATION_SCHEMA_VERSION,
        "status": "M1_B1_EXECUTION_AUTHORIZED_ONCE",
        "source_commit": plan["source_commit"],
        "source_hashes": plan["source_hashes"],
        "execution_plan_sha256": "d" * 64,
        "execution_plan_fingerprint": plan["plan_fingerprint"],
        "permissions": execution.EXPECTED_EXECUTION_PERMISSIONS,
        "resource_caps": plan["resource_caps"],
        "execution_run_limit": 1,
        "result_schema": {
            "path": schema_path,
            "sha256": schema_sha,
            "schema_version": execution.RESULT_SCHEMA_VERSION,
        },
        "result_terminal_statuses": [
            execution.COMPLETE_STATUS,
            execution.FAILURE_STATUS,
        ],
    }
    monkeypatch.setattr(execution, "_git_head", lambda _root: "e" * 40)
    monkeypatch.setattr(execution, "_is_ancestor", lambda _root, _a, _d: True)
    monkeypatch.setattr(
        execution,
        "file_sha256",
        lambda path: schema_sha if str(path).endswith(schema_path) else "b" * 64,
    )
    execution.validate_execution_inputs(
        ROOT,
        authorization,
        "f" * 64,
        plan,
        "d" * 64,
    )
    unauthorized = copy.deepcopy(authorization)
    unauthorized["permissions"]["held_out_access_authorized"] = True
    with pytest.raises(ValueError, match="permissions"):
        execution.validate_execution_inputs(
            ROOT,
            unauthorized,
            "f" * 64,
            plan,
            "d" * 64,
        )


def test_synthetic_random_cell_uses_dynamic_delta_and_paired_seed_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution, "TRAJECTORIES_PER_RANDOM_CELL", 2)
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "m1-b1-synthetic"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 0),
        identity_policy="extract_identity_phase",
    )
    master_seed = 12345
    cell = {
        "candidate_fingerprint": "c" * 64,
        "task_fingerprint": "d" * 64,
        "method": "B3",
        "rank": 0,
        "q": 2,
        "r": 1,
        "K": 2,
        "accuracy_eligible": True,
        "request_master_seed": master_seed,
        "sampled_trajectory_seeds": execution.expected_trajectory_seeds(master_seed, 2),
    }
    result = execution.compile_cell(
        execution.M1B1CompileJob(
            ordinal=0,
            plan_cell=cell,
            preparation=preparation,
            cache_path=str(tmp_path / "cell.sqlite3"),
        )
    )
    assert result["delta"] == pytest.approx(0.4)
    cosine = result["axes"]["cosine"]
    sine = result["axes"]["sine"]
    assert cosine["sampled_trajectory_seeds"] == sine["sampled_trajectory_seeds"]
    assert cosine["sampled_trajectory_seeds"] == cell["sampled_trajectory_seeds"]
    assert cosine["sample_count"] == sine["sample_count"] == 2
    assert cosine["state_preparation_included"] is False
    assert cosine["measurement_included"] is True


def test_frozen_m1_a_sha_is_still_exact() -> None:
    assert file_sha256(ROOT / M1_A_RESULT_RELATIVE) == M1_A_RESULT_SHA256
