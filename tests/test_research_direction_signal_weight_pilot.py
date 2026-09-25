from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_signal_weight_pilot import (
    EXPECTED_LD_VALUES,
    evaluate_signal_weight_pilot,
    finalize_signal_weight_pilot_artifact,
    validate_signal_weight_pilot_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = ROOT / "artifacts" / "pf_delta_validation"
ARTIFACT = (
    ROOT
    / "artifacts"
    / "research_direction_signal_weight_pilot"
    / "2026-09-25"
    / "pb_h4_s2_prefix_grid_v1.json"
)


def _inputs() -> list[dict]:
    return [
        json.loads(
            (
                INPUT_DIRECTORY
                / f"h4_sto3g_d100_rank12_ld{ld}_v5.json"
            ).read_text(encoding="utf-8")
        )
        for ld in EXPECTED_LD_VALUES
    ]


def test_pb_reanalysis_finds_no_meaningful_selection_disagreement() -> None:
    body = evaluate_signal_weight_pilot(_inputs())

    assert body["overall_pass"]
    assert body["configuration"]["candidate_count"] == 72
    assert body["summary"]["minimum_target_weight"]["target_weight"] == (
        pytest.approx(0.9999803238781496)
    )
    assert body["summary"]["minimum_q_signal_radius"][
        "minimum_q_signal_radius"
    ] == pytest.approx(0.999974490356076)
    assert body["summary"]["energy_signal_selection_disagreement_count"] == 0
    assert body["summary"]["pairwise_ordering_inversion_count"] > 0
    assert body["summary"]["meaningful_pairwise_ordering_inversion_count"] == 0
    assert body["summary"]["meaningful_near_tie_signal_advantage_count"] == 0
    assert body["summary"]["pb_hypothesis_supported_in_scope"] is False
    assert body["decision"]["pb_primary_theme_recommended"] is False
    assert body["decision"]["broader_signal_weight_research_rejected"] is False
    assert body["decision"]["next_pilot"] == "P-C-geometry-signed-error"


def test_pb_artifact_is_tamper_evident_and_scope_limited() -> None:
    body = evaluate_signal_weight_pilot(_inputs())
    artifact = finalize_signal_weight_pilot_artifact(
        body, provenance={"test": True}
    )
    validate_signal_weight_pilot_artifact(artifact)

    tampered = deepcopy(artifact)
    tampered["summary"]["energy_signal_selection_disagreement_count"] = 1
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_signal_weight_pilot_artifact(tampered)

    overstated = deepcopy(artifact)
    overstated["scope"]["p_b_general_hypothesis_rejected"] = True
    unsigned = dict(overstated)
    unsigned.pop("content_fingerprint")
    from trotterlib.research_direction_full_scope import fingerprint

    overstated["content_fingerprint"] = fingerprint(unsigned)
    with pytest.raises(ValueError, match="general hypothesis"):
        validate_signal_weight_pilot_artifact(overstated)


def test_committed_pb_artifact_validates() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_signal_weight_pilot_artifact(payload)
    assert len(payload["source_evidence"]) == 12
    assert payload["decision"]["status"] == (
        "do_not_advance_pb_on_current_h4_s2_prefix_grid"
    )
