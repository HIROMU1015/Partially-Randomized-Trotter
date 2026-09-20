from __future__ import annotations

import copy
import math

import pytest

from trotterlib.rpe_delta_compiled_cost_validation import (
    METHOD,
    SCHEMA_VERSION,
    _fingerprint,
    _k4_iid_form,
    iid_local_window_form,
    validate_rpe_delta_compiled_cost_payload,
)


def test_iid_local_window_form_has_expected_multiplicities() -> None:
    form = iid_local_window_form(4, {0: 0.9, 2: 0.1})
    assert math.isclose(form["k1:0"], 3.6)
    assert math.isclose(form["k1:2"], 0.4)
    assert math.isclose(form["k2:0,0"], 3 * 0.9**2)
    assert math.isclose(form["k2:0,2"], 3 * 0.9 * 0.1)
    assert math.isclose(form["k3:2,2,2"], 2 * 0.1**3)
    assert math.isclose(
        sum(value for key, value in form.items() if key.startswith("k1:")),
        4.0,
    )
    assert math.isclose(
        sum(value for key, value in form.items() if key.startswith("k2:")),
        3.0,
    )
    assert math.isclose(
        sum(value for key, value in form.items() if key.startswith("k3:")),
        2.0,
    )


def test_k4_form_retains_zero_or_one_rare_event_and_reports_omitted_mass() -> None:
    form, omitted = _k4_iid_form(32, {0: 0.9, 2: 0.1})
    assert set(form) == {
        "k4:0,0,0,0",
        "k4:0,0,0,2",
        "k4:0,0,2,0",
        "k4:0,2,0,0",
        "k4:2,0,0,0",
    }
    retained_probability = 0.9**4 + 4 * 0.1 * 0.9**3
    assert math.isclose(sum(form.values()), 29 * retained_probability)
    assert math.isclose(omitted, 29 * (1.0 - retained_probability))


def _minimal_payload() -> dict:
    metrics = (
        "rz_count",
        "rz_depth",
        "cx_count",
        "cx_depth",
        "total_depth",
        "circuit_size",
    )
    rankings = {
        metric: {
            "best_delta": 0.02,
            "rows": [
                {
                    "delta_time": 0.02,
                    "cost": 1.0,
                    "relative_overhead_to_best": 0.0,
                }
            ],
        }
        for metric in metrics
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scope": {
            "central_rte_blocks_only": True,
            "deterministic_df_sweeps_included": False,
            "deterministic_rte_boundary_cost_included": False,
            "controlled_evolution_overhead_included": False,
            "ancilla_hadamard_axis_measurement_wrapper_included": False,
            "state_preparation_included": False,
            "q_greater_than_8_full_circuit_compilation_performed": False,
            "final_one_shot_compiled_cost_evaluated": False,
            "final_total_cost_evaluation_performed": False,
        },
        "candidate_projections": [{"delta_time": 0.02}],
        "rankings": rankings,
        "summary": {"angle_invariance_passed": True},
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    return payload


def test_payload_validator_accepts_scoped_projection() -> None:
    validate_rpe_delta_compiled_cost_payload(_minimal_payload())


def test_payload_validator_rejects_tampering_and_scope_overclaim() -> None:
    tampered = copy.deepcopy(_minimal_payload())
    tampered["rankings"]["rz_count"]["rows"][0]["cost"] = 2.0
    with pytest.raises(ValueError, match="fingerprint"):
        validate_rpe_delta_compiled_cost_payload(tampered)

    overclaim = copy.deepcopy(_minimal_payload())
    overclaim["scope"]["final_one_shot_compiled_cost_evaluated"] = True
    overclaim["content_fingerprint"] = _fingerprint(
        {key: value for key, value in overclaim.items() if key != "content_fingerprint"}
    )
    with pytest.raises(ValueError, match="overclaims"):
        validate_rpe_delta_compiled_cost_payload(overclaim)
