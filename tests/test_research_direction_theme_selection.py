from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_theme_selection import (
    evaluate_theme_selection,
    finalize_theme_selection_artifact,
    validate_theme_selection_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "pb": ROOT
    / "artifacts/research_direction_signal_weight_pilot/2026-09-25/"
    "pb_h4_s2_prefix_grid_v1.json",
    "pc": ROOT
    / "artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/"
    "pc_h4_geometry_signed_error_v1.json",
    "pa": ROOT
    / "artifacts/research_direction_joint_synthesis_pilot/2026-09-25/"
    "pa_h4_interval_union_joint_synthesis_v1.json",
}


def _inputs() -> dict[str, dict]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in PATHS.items()
    }


def test_theme_selection_is_pa_provisional_pc_secondary_pb_stopped() -> None:
    body = evaluate_theme_selection(**_inputs())

    assert body["overall_pass"]
    assert body["decision"]["primary_theme"] == "P-A_joint_sequence_synthesis"
    assert body["decision"]["secondary_theme"] == "P-C_geometry_energy_difference"
    assert body["decision"]["stopped_theme_in_current_scope"] == "P-B_signal_weight"
    assert body["decision"]["h12_or_long_rpe_required_next"] is False
    assert body["scope"]["literature_novelty_established"] is False


def test_theme_selection_preserves_quantitative_pilot_basis() -> None:
    body = evaluate_theme_selection(**_inputs())

    assert body["quantitative_basis"]["pa"][
        "pooled_rz_relative_change_vs_current"
    ] == pytest.approx(-0.07193862496311598)
    assert body["quantitative_basis"]["pc"][
        "maximum_geometry_coefficient_holdout_relative_error"
    ] == pytest.approx(0.04408565409572596)
    assert body["quantitative_basis"]["pb"]["selection_disagreement_count"] == 0


def test_theme_selection_rejects_upstream_and_own_tampering() -> None:
    inputs = _inputs()
    inputs["pa"] = deepcopy(inputs["pa"])
    inputs["pa"]["summary"]["pa_hypothesis_supported_in_scope"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_theme_selection(**inputs)

    artifact = finalize_theme_selection_artifact(
        evaluate_theme_selection(**_inputs()), provenance={"test": True}
    )
    validate_theme_selection_artifact(artifact)
    tampered = deepcopy(artifact)
    tampered["scope"]["literature_novelty_established"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_theme_selection_artifact(tampered)
