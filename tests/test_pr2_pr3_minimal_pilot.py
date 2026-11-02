from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_rte_tail import (
    dense_extracted_df_tail,
    extract_df_tail_from_hamiltonian,
)
from trotterlib.pr2_pr3_minimal_pilot import (
    PREREGISTRATION_SHA256,
    SCHEMA_VERSION,
    _dense_df_hamiltonian,
    _json_fingerprint,
    _residual_sample_cost,
    _theme_scores,
    run_pr3_pilot,
    validate_pr2_pr3_payload,
)


def _complex(record: dict[str, float]) -> complex:
    return complex(record["real"], record["imag"])


def test_pr3_fixed_extrapolation_obeys_preregistered_stop_gate() -> None:
    result = run_pr3_pilot()
    levels = result["partial_levels"]
    extrapolated = 2.0 * _complex(levels[1]["mean"]) - _complex(levels[0]["mean"])

    assert result["correctness"]["implementation_valid"] is True
    assert result["correctness"]["monotone_window"] is True
    assert result["bias_condition_pass"] is True
    assert _complex(result["partial_tail_extrapolated"]["estimate"]) == pytest.approx(
        extrapolated
    )
    assert result["partial_tail_extrapolated_bias"] <= 0.75 * result["bias_8"]
    assert result["work_ratio_to_best_ordinary"] >= 2.0
    assert result["decision"] == "STOP_PR3_VARIANCE_BACKBONE_DOMINATES"


def test_synthetic_df_residual_reconstructs_and_compiles_sample_cost() -> None:
    reference = DFHamiltonian(
        constant=0.17,
        one_body=np.asarray([[0.2, 0.03], [0.03, -0.11]], dtype=np.complex128),
        lambdas=np.asarray([0.42, -0.31], dtype=np.float64),
        g_matrices=(
            np.asarray([[0.7, 0.12], [0.12, -0.2]], dtype=np.complex128),
            np.asarray([[0.1, -0.24], [-0.24, 0.5]], dtype=np.complex128),
        ),
        metadata={"source": "synthetic-pr2-test"},
    )
    compressed = reference.select_blocks((0,))
    extraction = extract_df_tail_from_hamiltonian(
        "synthetic-rank2-minus-rank1",
        reference,
        (1,),
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
    )
    residual = dense_extracted_df_tail(extraction)
    np.testing.assert_allclose(
        _dense_df_hamiltonian(reference) - _dense_df_hamiltonian(compressed),
        residual,
        atol=1e-12,
    )

    sample_cost = _residual_sample_cost(extraction, angle=0.017)
    assert sample_cost["component_count"] > 0
    assert sample_cost["probability_sum"] == pytest.approx(1.0, abs=1e-13)
    assert sample_cost["expected_rz_count"] > 0.0
    assert sample_cost["minimum_rz_count"] >= 0


def test_theme_selection_enforces_stop_without_authorizing_more_numerics() -> None:
    selection = _theme_scores(
        {"decision": "STOP_PR2_NO_COMPETITIVE_TRADEOFF"},
        {"decision": "STOP_PR3_VARIANCE_BACKBONE_DOMINATES"},
    )
    assert selection["selection"] == "RETURN_TO_PR4_ESTIMATOR_COST_AUDIT"
    assert selection["mandatory_stop"] == "STOP_AFTER_PR2_PR3_MINIMAL_PILOTS"
    assert selection["additional_numerics_authorized"] is False


def test_payload_fingerprint_and_stop_guard_reject_tampering() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "pr2": {"pilot_fingerprint": "pr2"},
        "pr3": {"pilot_fingerprint": "pr3"},
        "theme_selection": {
            "mandatory_stop": "STOP_AFTER_PR2_PR3_MINIMAL_PILOTS",
            "additional_numerics_authorized": False,
        },
    }
    payload["result_fingerprint"] = _json_fingerprint(payload)
    validate_pr2_pr3_payload(payload)

    altered = copy.deepcopy(payload)
    altered["theme_selection"]["additional_numerics_authorized"] = True
    with pytest.raises(ValueError, match="must not authorize"):
        validate_pr2_pr3_payload(altered)

    altered = copy.deepcopy(payload)
    altered["pr2"]["decision"] = "GO_PR2"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_pr2_pr3_payload(altered)


def test_registered_artifact_has_frozen_hash_and_valid_payload() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "artifacts/pr2_pr3_minimal_pilot/2026-09-27/"
        "pr2_pr3_minimal_pilot_v1.json"
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "65d5c12a129cd50f521b069b63be140f7bbee1eddf7ea8b859b18737dbb8d302"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    validate_pr2_pr3_payload(payload)
    assert payload["pr2"]["decision"] == "GO_PR2"
    assert payload["pr3"]["decision"] == (
        "STOP_PR3_VARIANCE_BACKBONE_DOMINATES"
    )
    assert payload["theme_selection"]["selection"] == (
        "SELECT_PR2_PRIMARY_CANDIDATE"
    )
