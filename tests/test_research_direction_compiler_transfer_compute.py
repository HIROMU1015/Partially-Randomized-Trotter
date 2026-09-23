from copy import deepcopy

import pytest

from trotterlib.research_direction_compiler_transfer_compute import (
    finalize_compiler_transfer_compute_artifact,
    validate_compiler_transfer_compute_artifact,
)


def test_compiler_transfer_compute_artifact_defers_scientific_decision() -> None:
    body = {
        "checks": {"raw_compute_complete": True},
        "scope": {
            "decision_reaggregation_performed": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }
    artifact = finalize_compiler_transfer_compute_artifact(
        body, provenance={"test": True}
    )
    validate_compiler_transfer_compute_artifact(artifact)
    tampered = deepcopy(artifact)
    tampered["scope"]["scientific_superiority_claimed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_compiler_transfer_compute_artifact(tampered)
