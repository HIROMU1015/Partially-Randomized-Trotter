from copy import deepcopy

import pytest

from trotterlib.research_direction_decision_cost import (
    finalize_wp01d_compute_artifact,
    validate_wp01d_compute_artifact,
)


def test_wp01d_compute_artifact_is_tamper_evident_and_scope_guarded() -> None:
    body = {
        "scope": {
            "calculation_stage_only": True,
            "final_scientific_superiority_claimed": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_wp01d_compute_artifact(
        body, provenance={"test": True}
    )
    validate_wp01d_compute_artifact(payload)
    tampered = deepcopy(payload)
    tampered["scope"]["final_scientific_superiority_claimed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp01d_compute_artifact(tampered)
