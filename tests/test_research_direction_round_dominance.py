from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_round_dominance import (
    evaluate_g08_round_dominance,
    finalize_g08_artifact,
    validate_g08_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
COMPUTE = ROOT / (
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_full_scope_optimization_compute_v2.json"
)
SYNTHESIS = ROOT / (
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_conditional_interval_synthesis_v1.json"
)


def test_g08_routes_dominant_late_rounds_to_m08() -> None:
    compute = json.loads(COMPUTE.read_text())
    synthesis = json.loads(SYNTHESIS.read_text())
    body = evaluate_g08_round_dominance(compute, synthesis)
    assert body["overall_pass"]
    assert body["m08_target"]["rte_steps"] == [32]
    assert body["m08_target"]["direct_holdout_q_values"] == [16, 32]
    assert body["candidates"]["3"][
        "last_three_round_cost_fraction"
    ] > 0.9
    assert body["candidates"]["3"][
        "dominant_cost_round"
    ] != body["candidates"]["3"]["dominant_rte_risk_round"]


def test_g08_artifact_is_tamper_evident() -> None:
    body = {
        "scope": {
            "new_circuit_compilation_performed": False,
            "final_total_cost_evaluation_performed": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_g08_artifact(body, provenance={"test": True})
    validate_g08_artifact(payload)
    tampered = deepcopy(payload)
    tampered["scope"]["new_circuit_compilation_performed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_g08_artifact(tampered)
