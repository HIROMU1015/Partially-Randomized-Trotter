from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from trotterlib.research_direction_joint_synthesis_pilot import (
    finalize_joint_synthesis_pilot_artifact,
    preserved_columns_unitary_completion,
    validate_joint_synthesis_pilot_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts"
    / "research_direction_joint_synthesis_pilot"
    / "2026-09-25"
    / "pa_h4_interval_union_joint_synthesis_v1.json"
)


def test_preserved_columns_completion_supports_interval_unions() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(5, 5))
    unitary, _ = np.linalg.qr(matrix)
    completed = preserved_columns_unitary_completion(unitary, (0, 2, 4))

    assert np.max(
        np.abs(completed[:, (0, 2, 4)] - unitary[:, (0, 2, 4)])
    ) < 1e-12
    assert np.max(np.abs(completed.T @ completed - np.eye(5))) < 1e-12


def test_pa_artifact_is_tamper_evident_and_scope_limited() -> None:
    body = {
        "scope": {
            "full_partial_s2_or_hadamard_wrapper_compiled": False,
            "production_default_changed": False,
            "backend_or_noise_evaluated": False,
            "rpe_or_final_total_cost_evaluated": False,
            "scientific_superiority_claimed": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    artifact = finalize_joint_synthesis_pilot_artifact(
        body, provenance={"test": True}
    )
    validate_joint_synthesis_pilot_artifact(artifact)

    tampered = deepcopy(artifact)
    tampered["overall_pass"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_joint_synthesis_pilot_artifact(tampered)


def test_pa_committed_artifact_validates() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_joint_synthesis_pilot_artifact(payload)
    assert payload["decision"]["next_action"] == (
        "compare_PB_PC_PA_and_select_primary_theme"
    )
    assert payload["baseline_contract"][
        "adjacent_basis_cancellation_enabled_for_all_policies"
    ]
