from copy import deepcopy

import pytest

from trotterlib.research_direction_full_scope_replication import (
    finalize_wp05br_artifact,
    validate_wp05br_artifact,
)


def test_wp05br_artifact_is_tamper_evident_and_scope_guarded() -> None:
    body = {
        "scope": {
            "fresh_trajectories_only": True,
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_wp05br_artifact(body, provenance={"test": True})
    validate_wp05br_artifact(payload)
    tampered = deepcopy(payload)
    tampered["scope"]["decision_grade"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp05br_artifact(tampered)
