from __future__ import annotations

from copy import deepcopy

import pytest

from trotterlib.research_direction_full_scope import AXES, METRICS
from trotterlib.research_direction_full_scope_extension import (
    finalize_wp05b_artifact,
    fit_affine_holdouts,
    validate_wp05b_artifact,
)


def _point(q_m: int) -> dict:
    axes = {}
    for axis in AXES:
        policies = {}
        for policy, offset in (
            ("full_basis_shared", 3.0),
            ("support_run_le_1", 2.0),
        ):
            policies[policy] = {
                metric: {
                    "mean": 10.0 * q_m + offset,
                    "standard_error": 0.25,
                    "minimum": 10.0 * q_m + offset,
                    "maximum": 10.0 * q_m + offset,
                }
                for metric in METRICS
            }
        axes[axis] = {"policies": policies}
    return {"axes": axes}


def test_affine_models_score_q4_and_q8_holdouts() -> None:
    models = fit_affine_holdouts(
        {q: _point(q) for q in (1, 2, 4, 8)},
        policy="support_run_le_1",
    )
    for axis in AXES:
        for metric in METRICS:
            assert models[axis][metric]["slope"] == 10.0
            assert (
                models[axis][metric]["holdouts"]["4"][
                    "absolute_relative_error"
                ]
                == 0.0
            )
            assert (
                models[axis][metric]["holdouts"]["8"][
                    "absolute_relative_error"
                ]
                == 0.0
            )


def test_affine_models_require_calibration_and_holdouts() -> None:
    with pytest.raises(ValueError, match="q=1 and q=2"):
        fit_affine_holdouts({2: _point(2), 4: _point(4)}, policy=None)
    with pytest.raises(ValueError, match="Missing direct q=8"):
        fit_affine_holdouts(
            {q: _point(q) for q in (1, 2, 4)},
            policy="full_basis_shared",
        )


def test_wp05b_artifact_is_tamper_evident_and_scope_guarded() -> None:
    body = {
        "policy_input": {"production_default_changed": False},
        "scope": {
            "q8_directly_transpiled": True,
            "delta_0p01_directly_transpiled": True,
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_wp05b_artifact(body, provenance={"test": True})
    validate_wp05b_artifact(payload)

    tampered = deepcopy(payload)
    tampered["scope"]["decision_grade"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp05b_artifact(tampered)
