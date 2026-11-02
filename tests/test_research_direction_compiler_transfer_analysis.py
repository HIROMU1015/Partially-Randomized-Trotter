import copy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_compiler_transfer_analysis import (
    validate_compiler_transfer_analysis_artifact,
)


ARTIFACT = Path(
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_focused_analysis_reaggregation_v1.json"
)


def _load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compiler_transfer_analysis_records_focused_nonfinal_decision() -> None:
    payload = _load()
    validate_compiler_transfer_analysis_artifact(payload)
    assert payload["overall_pass"]
    assert payload["decision"]["compiler_invariant_local_separation"] == (
        "not_established"
    )
    assert payload["summary"]["focused_measured_discrepancy_intervals_overlap"]
    assert not payload["scope"]["schedule_shots_alpha_beta_reoptimized_under_opt2"]
    assert not payload["scope"]["scientific_superiority_claimed"]


def test_compiler_transfer_analysis_rejects_tampering() -> None:
    payload = _load()
    tampered = copy.deepcopy(payload)
    tampered["summary"]["focused_ld3_point_estimate"] += 1.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_compiler_transfer_analysis_artifact(tampered)
