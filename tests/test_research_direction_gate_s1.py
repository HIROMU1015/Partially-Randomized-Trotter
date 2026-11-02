from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trotterlib.research_direction_gate_s1 import (
    build_gate_s1_body,
    finalize_gate_s1_artifact,
    validate_gate_s1_artifact,
)


ROOT = Path("artifacts")
PATHS = {
    "wp00": ROOT
    / "research_direction_prevalidation/2026-09-21/wp00_comparison_contract_v1.json",
    "wp02": ROOT
    / "research_direction_prevalidation/2026-09-21/wp02_round_horizon_coverage_v1.json",
    "wp01s": ROOT
    / "research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json",
    "wp04": ROOT
    / "research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json",
    "wp03": ROOT
    / "research_direction_pf_sensitivity/2026-09-22/wp03_pf_coefficient_selection_sensitivity_v1.json",
}


def _inputs() -> dict[str, dict]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in PATHS.items()
    }


def test_gate_s1_artifact_fingerprint_is_tamper_evident() -> None:
    inputs = _inputs()
    body = build_gate_s1_body(**inputs)
    payload = finalize_gate_s1_artifact(body, provenance={"test": True})
    validate_gate_s1_artifact(payload)

    tampered = deepcopy(payload)
    tampered["summary"]["next_action"] = "WP05"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_gate_s1_artifact(tampered)


def test_gate_s1_preserves_uncertainty_and_selects_one_next_action() -> None:
    body = build_gate_s1_body(**_inputs())

    assert body["overall_pass"]
    assert body["summary"]["candidate_cost_judgement"] == (
        "undetermined_not_tied"
    )
    assert not body["summary"]["partial_specific_advantage_demonstrated"]
    assert not body["summary"]["pf_coefficient_choice_changes_shortlist"]
    assert body["next_action"]["work_package"] == "WP06-a"
    assert body["next_action"]["preregistered_decision_rule"][
        "eta_decision_relative_rz"
    ] == pytest.approx(0.05)
    assert body["next_action"]["following_action"].startswith("WP05")


def test_gate_s1_direction_table_has_all_seven_topics() -> None:
    body = build_gate_s1_body(**_inputs())
    decisions = {
        row["direction_id"]: row["decision"]
        for row in body["direction_decisions"]
    }

    assert set(decisions) == {f"T{index}" for index in range(1, 8)}
    assert decisions["T3"] == "defer"
    assert decisions["T4"] == "continue_high_priority"
    assert decisions["T7"] == "continue_primary_framing"


def test_gate_s1_rejects_unbound_upstream_chain() -> None:
    inputs = _inputs()
    broken = deepcopy(inputs["wp04"])
    broken["source_evidence"]["wp01_screening"]["content_fingerprint"] = "0" * 64
    inputs["wp04"] = broken

    with pytest.raises(ValueError, match="content_fingerprint mismatch"):
        build_gate_s1_body(**inputs)
