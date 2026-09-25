from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trotterlib.research_direction_proxy_lineage_reconciliation import (
    evaluate_proxy_lineage_reconciliation,
    validate_proxy_lineage_reconciliation_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
LATEST = (
    ROOT
    / "artifacts/research_direction_full_opt2/2026-09-25/"
    "wp11_all_r_opt2_coherent_analysis_20260925_065827.json"
)
LEGACY = (
    ROOT
    / "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_same_trajectory_compute_v1.json"
)
ARTIFACT = (
    ROOT
    / "artifacts/research_direction_full_opt2/2026-09-25/"
    "m06f_a0_proxy_lineage_reconciliation_v1.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a0_reconciles_selected_policy_q16_q32_without_refitting() -> None:
    artifact = _read(ARTIFACT)
    validate_proxy_lineage_reconciliation_artifact(artifact)
    assert artifact["overall_pass"] is True
    assert artifact["scope"]["new_compilation_performed"] is False
    assert artifact["scope"]["legacy_holdouts_refit"] is False
    summary = artifact["summary"]
    assert summary["selected_policy_q16_q32_pass_5_percent"] is True
    assert summary["maximum_selected_policy_absolute_relative_error"][
        "absolute_relative_error"
    ] == pytest.approx(0.04911236093712892)
    assert summary["maximum_selected_policy_absolute_relative_error"]["q_m"] == 32


def test_a0_preserves_full_basis_q32_failure_and_scope_limits() -> None:
    artifact = _read(ARTIFACT)
    assert artifact["summary"]["full_basis_q16_q32_pass_5_percent"] is False
    maximum = artifact["summary"]["maximum_full_basis_absolute_relative_error"]
    assert maximum["q_m"] == 32
    assert maximum["absolute_relative_error"] == pytest.approx(
        0.05597865674912205
    )
    domain = artifact["applicable_domain"]
    assert domain["full_basis_compiled_rz"]["maximum_q_passing_individually"] == 16
    assert domain["all_metrics_latest_fresh_holdout_max_q"] == 8
    assert domain["q_above_32_validated"] is False


def test_a0_rebuilds_and_rejects_tampering() -> None:
    artifact = _read(ARTIFACT)
    rebuilt = evaluate_proxy_lineage_reconciliation(_read(LATEST), _read(LEGACY))
    assert rebuilt["summary"] == artifact["summary"]
    assert rebuilt["applicable_domain"] == artifact["applicable_domain"]

    tampered = deepcopy(artifact)
    tampered["summary"]["selected_policy_q16_q32_pass_5_percent"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_proxy_lineage_reconciliation_artifact(tampered)
