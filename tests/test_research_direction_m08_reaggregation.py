from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_m08_reaggregation import (
    evaluate_m08_reaggregation,
    finalize_m08_reaggregation_artifact,
    validate_m08_reaggregation_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_m08_reaggregation_preserves_local_and_robust_claim_scopes() -> None:
    body = evaluate_m08_reaggregation(
        _load(
            "artifacts/research_direction_decision_cost/2026-09-22/"
            "wp01d_c07_full_scope_optimization_compute_v2.json"
        ),
        _load(
            "artifacts/research_direction_decision_cost/2026-09-22/"
            "wp01d_c07_conditional_interval_synthesis_v1.json"
        ),
        _load(
            "artifacts/research_direction_round_dominance/2026-09-22/"
            "g08_round_cost_risk_proxy_dominance_v1.json"
        ),
        _load(
            "artifacts/research_direction_proxy_precision/2026-09-22/"
            "m08_late_round_q16_q32_proxy_precision_v1.json"
        ),
    )
    assert body["overall_pass"]
    assert not body["scenarios"]["m08_selected_policy_rz_q_le_32"][
        "intervals_overlap"
    ]
    assert not body["scenarios"]["local_5_percent"]["intervals_overlap"]
    assert body["scenarios"]["transfer_25_percent"]["intervals_overlap"]
    assert not body["decision"][
        "partial_randomization_scientific_superiority_established"
    ]


def test_m08_reaggregation_artifact_is_tamper_evident() -> None:
    body = {
        "decision": {
            "partial_randomization_scientific_superiority_established": False
        },
        "scope": {
            "q_above_32_directly_validated": False,
            "final_total_cost_evaluation_performed": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_m08_reaggregation_artifact(
        body, provenance={"test": True}
    )
    validate_m08_reaggregation_artifact(payload)
    tampered = deepcopy(payload)
    tampered["scope"]["q_above_32_directly_validated"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_m08_reaggregation_artifact(tampered)
