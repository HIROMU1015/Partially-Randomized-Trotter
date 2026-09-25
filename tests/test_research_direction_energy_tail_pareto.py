from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_energy_tail_pareto import (
    BLIND_HOLDOUT_LD,
    DEVELOPMENT_LD,
    FORMULA_LABELS,
    composition_tail_weights,
    evaluate_formula_order_registry,
    evaluate_nested_vs_global_order,
    validate_energy_tail_pareto_artifact,
    validate_expected_task_manifest,
)
from trotterlib.research_direction_full_scope import fingerprint


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "research_direction_energy_tail_pareto"
    / "2026-09-25"
)
EXPECTED = ARTIFACT_ROOT / "pd_energy_tail_expected_tasks_v1.json"
FINAL = ARTIFACT_ROOT / "pd_energy_tail_pareto_v1.json"
EXPECTED_FINGERPRINT = "f302dafce37fb90f3acfe32aa83edf563dd1609015a1b3d50972880e13407c7d"
FINAL_FINGERPRINT = "846362ab808e9b26e5648f7f9d12d541dd01a6f9c2954e45d9194b4dfef8d835"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_formula_registry_has_normalized_palindromic_tail_sequences() -> None:
    for label in FORMULA_LABELS:
        coefficients = composition_tail_weights(label)
        assert sum(coefficients) == pytest.approx(1.0, abs=1.0e-12)
        assert coefficients == pytest.approx(tuple(reversed(coefficients)), abs=1.0e-12)


def test_toy_order_and_nested_global_distinction_pass() -> None:
    order = evaluate_formula_order_registry()
    assert order["commutator_spectral_norm"] > 0.1
    assert order["all_formulae_eligible"]
    assert {row["label"] for row in order["formulae"]} == set(FORMULA_LABELS)
    nested = evaluate_nested_vs_global_order()
    assert nested["inner_commutator_spectral_norm"] > 0.1
    assert nested["outer_commutator_spectral_norm"] > 0.1
    assert nested["overall_pass"]


def test_frozen_expected_and_final_artifacts() -> None:
    expected = _load(EXPECTED)
    validate_expected_task_manifest(expected)
    assert expected["content_fingerprint"] == EXPECTED_FINGERPRINT
    assert expected["task_count"] == 2 * len(FORMULA_LABELS)
    assert expected["configuration"]["development_ld"] == DEVELOPMENT_LD
    assert expected["configuration"]["blind_holdout_ld"] == BLIND_HOLDOUT_LD
    assert expected["exploration_disclosure"][
        "development_ld3_candidate_grid_inspected_before_freeze"
    ]
    assert not expected["exploration_disclosure"][
        "blind_ld4_candidate_grid_inspected_before_freeze"
    ]

    final = _load(FINAL)
    assert final["content_fingerprint"] == FINAL_FINGERPRINT
    validate_energy_tail_pareto_artifact(final)
    assert final["expected_task_fingerprint"] == expected["content_fingerprint"]
    assert [split["ld"] for split in final["splits"]] == [
        DEVELOPMENT_LD,
        BLIND_HOLDOUT_LD,
    ]
    assert final["overall_pass"]
    assert all(final["gates"].values())
    assert final["decision"] == {
        "current_primary_theme": "P-D-conditional-candidate",
        "next_required_gate": "signed_negative_time_rte_oracle_and_real_internal_hd_error",
        "status": "advance_pd_as_conditional_candidate_pending_signed_time_and_inner_hd_validation",
        "thresholds_changed_after_results": False,
    }

    for split in final["splits"]:
        decision = split["decision"]
        assert decision["energy_only_label"] == "8th(Morales)"
        assert decision["tail_aware_label"] == "4th(new_2)"
        assert decision["gamma_reduction"] == pytest.approx(0.5670189565972156)
        assert decision["selection_gate_pass"]


def test_artifacts_reject_tampering_and_scope_overstatement() -> None:
    expected = _load(EXPECTED)
    expected["tasks"][0]["ld"] = 99
    with pytest.raises(ValueError, match="fingerprint"):
        validate_expected_task_manifest(expected)

    final = _load(FINAL)
    tampered = copy.deepcopy(final)
    tampered["decision"]["status"] = "arbitrary"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_energy_tail_pareto_artifact(tampered)

    overstated = copy.deepcopy(final)
    overstated.pop("content_fingerprint")
    overstated["scope"]["compiled_circuit_cost_evaluated"] = True
    overstated["content_fingerprint"] = fingerprint(overstated)
    with pytest.raises(ValueError, match="overstates scope"):
        validate_energy_tail_pareto_artifact(overstated)
