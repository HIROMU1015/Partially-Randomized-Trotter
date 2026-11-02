from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trotterlib.research_direction_full_opt2_completion import (
    build_completion_audit,
    finalize_completion_audit,
    validate_completion_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path("artifacts/research_direction_full_opt2/2026-09-24")
AUDIT_PATH = BASE / "wp11_all_r_opt2_completion_audit_20260924_230342.json"
RAW_OUTPUT = BASE / "wp11_all_r_opt2_initial_20260924_023012"
RAW_EVIDENCE_AVAILABLE = all(
    (REPO_ROOT / RAW_OUTPUT / directory).is_dir()
    for directory in ("checkpoints", "tasks", "worker_results")
)


def _artifact() -> dict[str, object]:
    return json.loads((REPO_ROOT / AUDIT_PATH).read_text())


def _body() -> dict[str, object]:
    return build_completion_audit(
        project_root=REPO_ROOT,
        initial_manifest_path=REPO_ROOT
        / BASE
        / "wp11_all_r_opt2_initial_20260924_023012.manifest.json",
        compute_output_dir=REPO_ROOT
        / BASE
        / "wp11_all_r_opt2_initial_20260924_023012",
        initial_analysis_path=REPO_ROOT
        / BASE
        / "wp11_all_r_opt2_analysis_20260924_222811.json",
        extension_manifest_path=REPO_ROOT
        / BASE
        / "wp11_all_r_opt2_extension_20260924_222811.manifest.json",
        runtime_observation={
            "observed_at_utc": "2026-09-24T00:00:00+00:00",
            "tmux_session": "wp11_all_r_opt2",
            "tmux_session_present": False,
            "matching_processes": [],
        },
        original_main_observation={
            "path": "/home/AbeHiromu/projects/partially-randomized-trotter",
            "git_status": [
                "## main...origin/main",
                "?? docs/gpu_execution_environment.md",
            ],
            "status_matches_expected": True,
            "gpu_execution_environment_sha256": "test-observation",
        },
    )


def test_completion_audit_validates_initial_batch_and_gates_reoptimization() -> None:
    artifact = _artifact()
    validate_completion_audit(artifact)
    assert artifact["initial_compute_integrity_pass"] is True
    assert artifact["execution"]["initial_batch"]["expected"] == 36
    assert artifact["execution"]["initial_batch"]["completed"] == 36
    assert artifact["execution"]["preregistered_workflow"]["expected"] == 51
    assert artifact["execution"]["preregistered_workflow"]["completed"] == 36
    assert artifact["extension_gate"]["task_count"] == 15
    assert artifact["extension_gate"]["completed_task_count"] == 0
    assert len(artifact["extension_gate"]["failing_initial_cells"]) == 5
    assert artifact["analysis_items"]["3_beta_alpha_shots_schedule_by_r"] == (
        "blocked_by_fresh32_gate"
    )


def test_completion_audit_records_direct_scope_and_holdout_accuracy() -> None:
    body = _artifact()
    selected = [
        row
        for row in body["direct_compiled_rz_measurements"]
        if row["ld"] == 3 and row["policy"] == "support_run_le_1"
    ]
    assert {row["r"] for row in selected} == {1, 2, 4, 8, 16, 32}
    assert {row["delta"] for row in selected} == {0.01, 0.02}
    assert body["calibration_holdout_diagnostics"]["failing_cell_count"] == 5
    assert body["calibration_holdout_diagnostics"][
        "maximum_selected_rz_holdout_error"
    ] <= 0.05


def test_completion_audit_fingerprint_detects_tampering() -> None:
    artifact = _artifact()
    tampered = deepcopy(artifact)
    tampered["status"] = "coherent_opt2_reoptimization_complete"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_completion_audit(tampered)


@pytest.mark.skipif(
    not RAW_EVIDENCE_AVAILABLE,
    reason="raw M06-F task/checkpoint/worker evidence is not tracked by Git",
)
def test_raw_completion_evidence_rebuilds_when_available() -> None:
    artifact = finalize_completion_audit(
        _body(),
        provenance={"evidence_status": "optional_raw_integration_test"},
    )
    validate_completion_audit(artifact)
    assert artifact["initial_compute_integrity_pass"] is True
