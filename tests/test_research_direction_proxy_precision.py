from copy import deepcopy

import pytest

from trotterlib.research_direction_proxy_precision import (
    finalize_m08_artifact,
    validate_m08_artifact,
)


def test_m08_artifact_is_tamper_evident_and_scope_guarded() -> None:
    body = {
        "scope": {
            "q_above_32_directly_validated": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_m08_artifact(body, provenance={"test": True})
    validate_m08_artifact(payload)
    tampered = deepcopy(payload)
    tampered["scope"]["scientific_superiority_claimed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_m08_artifact(tampered)
