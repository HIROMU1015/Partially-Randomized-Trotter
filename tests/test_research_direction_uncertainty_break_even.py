from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_uncertainty_break_even import (
    evaluate_uncertainty_break_even,
    finalize_uncertainty_break_even_artifact,
    validate_uncertainty_break_even_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_n07_p03_separates_uncertainty_and_parameterizes_preparation() -> None:
    body = evaluate_uncertainty_break_even(
        _load(
            "artifacts/research_direction_decision_cost/2026-09-22/"
            "wp01d_c07_full_scope_optimization_compute_v2.json"
        ),
        _load(
            "artifacts/research_direction_proxy_precision/2026-09-22/"
            "wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json"
        ),
        _load(
            "artifacts/research_direction_compiler_transfer/2026-09-23/"
            "m06_l08_opt2_focused_analysis_reaggregation_v1.json"
        ),
    )
    assert body["overall_pass"]
    assert body["configuration"]["ld3_total_shots"] == 13_538
    assert body["configuration"]["ld12_total_shots"] == 11_162
    assert body["summary"]["opt1_local_ld3_guaranteed_better_until"] > 0.0
    assert body["summary"]["opt2_focused_ld3_guaranteed_better_until"] is None
    assert body["decision"]["robust_directional_result"] == (
        "undetermined_under_compiler_transfer_and_preparation_sensitivity"
    )
    assert not body["scope"]["state_preparation_cost_measured"]
    assert not body["scope"]["scientific_superiority_claimed"]


def test_n07_p03_artifact_is_tamper_evident() -> None:
    body = {
        "checks": {"test": True},
        "overall_pass": True,
        "scope": {
            "state_preparation_cost_measured": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }
    payload = finalize_uncertainty_break_even_artifact(
        body, provenance={"test": True}
    )
    validate_uncertainty_break_even_artifact(payload)
    tampered = deepcopy(payload)
    tampered["scope"]["state_preparation_cost_measured"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_uncertainty_break_even_artifact(tampered)
