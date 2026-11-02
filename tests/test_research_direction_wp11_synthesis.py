from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_wp11_synthesis import (
    evaluate_wp11_synthesis,
    finalize_wp11_artifact,
    validate_wp11_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "gate_s1": "artifacts/research_direction_gate_s1/2026-09-22/gate_s1_research_direction_decision_v1.json",
    "wp06a": "artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json",
    "wp06b": "artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json",
    "wp05a": "artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json",
    "wp05b": "artifacts/research_direction_full_scope_extension/2026-09-22/wp05b_q8_delta_0p01_full_scope_extension_v1.json",
    "wp05br": "artifacts/research_direction_full_scope_replication/2026-09-22/wp05br_r32_32trajectory_replication_v1.json",
    "wp01d": "artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v2.json",
    "g08": "artifacts/research_direction_round_dominance/2026-09-22/g08_round_cost_risk_proxy_dominance_v1.json",
    "m08_reaggregation": "artifacts/research_direction_proxy_precision/2026-09-22/wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json",
    "compiler_transfer": "artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_focused_analysis_reaggregation_v1.json",
    "n07_p03": "artifacts/research_direction_uncertainty_break_even/2026-09-23/n07_p03_uncertainty_break_even_v1.json",
}


def _inputs() -> dict[str, dict]:
    return {
        name: json.loads((ROOT / path).read_text(encoding="utf-8"))
        for name, path in PATHS.items()
    }


def test_wp11_selects_one_coherent_opt2_followup() -> None:
    body = evaluate_wp11_synthesis(**_inputs())

    assert body["overall_pass"]
    assert body["summary"]["status"] == "WP11_scoped_direction_synthesis_complete"
    assert body["summary"]["selected_next_followup"] == (
        "all_r_coherent_opt2_reoptimization"
    )
    assert body["summary"]["new_opt2_ld3_r_values_required"] == [1, 2, 4, 8, 16]
    assert not body["summary"]["gpu_required_for_selected_followup"]
    assert body["scope"]["full_opt2_reoptimization_performed"] is False
    assert body["scope"]["scientific_superiority_claimed"] is False


def test_wp11_has_all_direction_decisions_and_preserves_failed_pilot() -> None:
    body = evaluate_wp11_synthesis(**_inputs())
    decisions = {
        row["direction_id"]: row["wp11_decision"]
        for row in body["direction_decisions"]
    }

    assert set(decisions) == {f"T{i}" for i in range(1, 8)}
    assert decisions["T1"] == "rescope"
    assert decisions["T3"] == "hold"
    assert decisions["T4"] == "continue_primary"
    assert decisions["T7"] == "continue_primary"
    assert body["checks"]["initial_wp05b_failure_is_preserved"]
    assert body["checks"]["independent_wp05br_followup_passes"]


def test_wp11_rejects_unbound_chain_and_tampering() -> None:
    inputs = _inputs()
    broken = deepcopy(inputs["n07_p03"])
    broken["input_fingerprints"]["wp01d"] = "0" * 64
    inputs["n07_p03"] = broken
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_wp11_synthesis(**inputs)

    body = {
        "candidate_followups": [{"id": "test", "status": "selected"}],
        "scope": {
            "new_physical_simulation_performed": False,
            "new_circuit_compilation_performed": False,
            "full_opt2_reoptimization_performed": False,
            "external_instance_validated": False,
            "state_preparation_cost_measured": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    artifact = finalize_wp11_artifact(body, provenance={"test": True})
    validate_wp11_artifact(artifact)
    tampered = deepcopy(artifact)
    tampered["scope"]["scientific_superiority_claimed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp11_artifact(tampered)
