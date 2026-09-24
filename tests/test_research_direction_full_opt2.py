from __future__ import annotations

from copy import deepcopy

import pytest

from trotterlib.parallel_validation_executor import _THREAD_ENVIRONMENT
from trotterlib.research_direction_full_opt2 import (
    EXPECTED_COMPILER,
    build_full_opt2_task_definitions,
    evaluate_proxy_cell,
    finalize_full_opt2_analysis_artifact,
    initial_randomized_cells,
    load_and_validate_full_opt2_evidence,
    validate_full_opt2_analysis_artifact,
)


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)


def _point(q_m: int, *, holdout_scale: float = 1.0) -> dict[str, object]:
    axes = {}
    for axis_index, axis in enumerate(("cosine", "sine")):
        policies = {}
        for policy_index, policy in enumerate(
            ("full_basis_shared", "support_run_le_1")
        ):
            metrics = {}
            for metric_index, metric in enumerate(METRICS):
                mean = float(
                    100
                    + axis_index
                    + policy_index
                    + metric_index
                    + 10 * q_m
                )
                if q_m == 8 and policy == "support_run_le_1":
                    mean *= holdout_scale
                metrics[metric] = {
                    "mean": mean,
                    "standard_error": mean * 0.005,
                    "minimum": mean,
                    "maximum": mean,
                }
            policies[policy] = metrics
        axes[axis] = {"policies": policies}
    return {"sample_count": 8, "q_m": q_m, "axes": axes}


def test_full_opt2_task_scope_and_compiler_context_are_fixed() -> None:
    evidence = load_and_validate_full_opt2_evidence(REPO_ROOT)
    tasks = build_full_opt2_task_definitions(evidence)
    randomized = [task for task in tasks if task["ld"] == 3]
    deterministic = [task for task in tasks if task["ld"] == 12]
    assert len(initial_randomized_cells()) == 33
    assert len(randomized) == 33
    assert len(deterministic) == 3
    assert all(task["compiler_settings"] == EXPECTED_COMPILER for task in tasks)
    assert all(task["resource"] == "cpu" for task in tasks)
    assert {(task["delta"], task["r"]) for task in randomized} == {
        *((0.01, r) for r in (1, 2, 4, 8, 16, 32)),
        *((0.02, r) for r in (1, 2, 4, 8, 16)),
    }
    assert {task["q"] for task in randomized} == {1, 2, 8}
    assert all(task["parameters"]["sample_count"] == 8 for task in randomized)
    assert _THREAD_ENVIRONMENT["NUMBA_NUM_THREADS"] == "1"


def test_fresh_extension_uses_only_requested_cells_and_new_32_stream() -> None:
    evidence = load_and_validate_full_opt2_evidence(REPO_ROOT)
    initial = build_full_opt2_task_definitions(
        evidence,
        initial_cells=((0.01, 1, 8),),
        include_deterministic=False,
    )[0]
    extension = build_full_opt2_task_definitions(
        evidence,
        extension_cells=((0.01, 1, 8),),
        include_deterministic=False,
    )[0]
    assert extension["parameters"]["sample_count"] == 32
    assert extension["parameters"]["extension"] is True
    assert extension["seed"] != initial["seed"]


def test_proxy_acceptance_and_holdout_failure_are_explicit() -> None:
    passing = evaluate_proxy_cell(
        {q_m: _point(q_m) for q_m in (1, 2, 8)}, holdout_q=(8,)
    )
    assert passing["pass"] is True
    assert passing["maximum_direct_rz_relative_standard_error"] <= 0.02
    failing = evaluate_proxy_cell(
        {
            1: _point(1),
            2: _point(2),
            8: _point(8, holdout_scale=1.2),
        },
        holdout_q=(8,),
    )
    assert failing["pass"] is False
    assert failing["checks"]["selected_rz_holdout_within_5_percent"] is False


def test_analysis_artifact_rejects_tampering_and_final_claims() -> None:
    body = {
        "checks": {"complete": True},
        "overall_pass": True,
        "scope": {
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }
    artifact = finalize_full_opt2_analysis_artifact(
        body, provenance={"test": True}
    )
    validate_full_opt2_analysis_artifact(artifact)
    tampered = deepcopy(artifact)
    tampered["scope"]["scientific_superiority_claimed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_full_opt2_analysis_artifact(tampered)
