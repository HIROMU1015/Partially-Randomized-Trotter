from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_full_scope import fingerprint
from trotterlib.research_direction_geometry_energy_difference_pilot import (
    EXPECTED_GEOMETRIES,
    evaluate_geometry_energy_difference_pilot,
    finalize_geometry_energy_difference_pilot_artifact,
    validate_geometry_energy_difference_pilot_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "research_direction_geometry_energy_difference_pilot"
    / "2026-09-25"
)
INPUT_DIRECTORY = ARTIFACT_DIRECTORY / "inputs"
ARTIFACT = ARTIFACT_DIRECTORY / "pc_h4_geometry_signed_error_v1.json"


def _distance_label(distance: float) -> str:
    return f"{int(round(100 * distance)):03d}"


def _inputs() -> list[dict]:
    return [
        json.loads(
            (
                INPUT_DIRECTORY
                / f"h4_sto3g_d{_distance_label(distance)}_rank12_ld3_v5.json"
            ).read_text(encoding="utf-8")
        )
        for distance in EXPECTED_GEOMETRIES
    ]


def test_pc_geometry_holdouts_and_difference_prediction_are_evaluated() -> None:
    body = evaluate_geometry_energy_difference_pilot(_inputs())

    assert body["overall_pass"]
    assert len(body["geometry_rows"]) == 5
    assert len(body["geometry_coefficient_holdouts"]) == 2
    assert len(body["adjacent_pair_results_at_delta_0p1"]) == 4
    assert body["summary"][
        "maximum_geometry_holdout_coefficient_relative_error"
    ] >= 0.0
    assert body["summary"][
        "combined_pair_prediction_error_normalized_by_endpoint_bias"
    ] >= 0.0
    assert body["scope"]["exact_energy_means_df_rank12_sector_reference"]
    assert not body["scope"]["final_total_cost_evaluation_performed"]


def test_pc_artifact_is_tamper_evident_and_scope_limited() -> None:
    artifact = finalize_geometry_energy_difference_pilot_artifact(
        evaluate_geometry_energy_difference_pilot(_inputs()),
        provenance={"test": True},
    )
    validate_geometry_energy_difference_pilot_artifact(artifact)

    tampered = deepcopy(artifact)
    tampered["summary"]["coefficient_span_fraction"] = -1.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_geometry_energy_difference_pilot_artifact(tampered)

    overstated = deepcopy(artifact)
    overstated["scope"]["potential_energy_surface_claimed"] = True
    unsigned = dict(overstated)
    unsigned.pop("content_fingerprint")
    overstated["content_fingerprint"] = fingerprint(unsigned)
    with pytest.raises(ValueError, match="overstates scope"):
        validate_geometry_energy_difference_pilot_artifact(overstated)


def test_pc_committed_artifact_validates() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_geometry_energy_difference_pilot_artifact(payload)
    assert len(payload["source_evidence"]) == 5
    assert payload["decision"]["next_pilot"] == (
        "P-A-joint-circuit-sequence-synthesis"
    )
