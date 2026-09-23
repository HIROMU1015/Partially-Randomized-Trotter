from copy import deepcopy

import pytest

from trotterlib.research_direction_decision_synthesis import (
    finalize_wp01d_synthesis_artifact,
    validate_wp01d_synthesis_artifact,
)


def test_wp01d_synthesis_artifact_is_tamper_evident() -> None:
    body = {
        "decision": {
            "partial_randomization_scientific_superiority_established": False
        },
        "scope": {"final_total_cost_evaluation_performed": False},
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_wp01d_synthesis_artifact(
        body, provenance={"test": True}
    )
    validate_wp01d_synthesis_artifact(payload)
    tampered = deepcopy(payload)
    tampered["decision"][
        "partial_randomization_scientific_superiority_established"
    ] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp01d_synthesis_artifact(tampered)
