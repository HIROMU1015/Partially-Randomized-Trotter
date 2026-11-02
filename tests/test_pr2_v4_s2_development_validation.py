from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import trotterlib.pr2_v4_s2_development_validation as validation
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte_compiled_cost import TranspiledCircuitCostCache


ROOT = Path(__file__).resolve().parents[1]


def _toy_hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.13,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.4, -0.3]),
        g_matrices=(
            np.asarray([[1.0]], dtype=np.complex128),
            np.asarray([[0.7]], dtype=np.complex128),
        ),
        metadata={"name": "pr2-v4-s2-test"},
    )


def _candidate(
    method: str,
    work: float,
    interval: tuple[float, float],
    *,
    rank: int,
    eligible: bool = True,
    r: int = 1,
    k: int = 2,
    shots: int = 100,
) -> dict:
    return {
        "method": method,
        "rank": rank,
        "r": r,
        "K": k,
        "signal": {
            "normalization_multiplier": 1.1,
            "exact_rte_lambda_r": 0.5,
            "component_count": 3,
            "corrected_raw_reconstruction_abs_error": 0.0,
        },
        "resource": {
            "accuracy_eligible": eligible,
            "total_work_no_preparation": work if eligible else None,
            "work_interval": list(interval) if eligible else None,
            "total_shots": shots if eligible else None,
            "rz_relative_standard_error_max": 0.01 if eligible else None,
        },
    }


def _common_payload(schema: str, status: str) -> dict:
    return {
        "schema_version": schema,
        "series_id": validation.SERIES_ID,
        "status": status,
        "authorization_commit": validation.AUTHORIZATION_COMMIT,
        "held_out": {
            "npz_loaded": False,
            "signal_cost_ranking_evaluated": False,
        },
        "counters": {
            "held_out_npz_loads": 0,
            "molecular_calculations": 0,
        },
        "S3_authorized": False,
        "automatic_next_stage": None,
        "quantum_shots_executed": 0,
    }


def _finish(payload: dict) -> dict:
    payload.pop("result_fingerprint", None)
    payload["result_fingerprint"] = validation._fingerprint(payload)
    return payload


def test_authorization_and_prior_artifact_hashes_are_frozen() -> None:
    authorization = (
        ROOT
        / "artifacts/pr2_v4_s2_development/2026-09-28/"
        "pr2_v4_s2_authorization_v1.json"
    )
    specification = ROOT / "docs/research/pr2_v4_s2_development_authorization_v5.md"
    assert validation.file_sha256(authorization) == validation.AUTHORIZATION_SHA256
    assert validation.file_sha256(specification) == validation.SPECIFICATION_SHA256
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    assert payload["authorization"]["v4_authorized"] is True
    assert payload["authorization"]["s3_authorized"] is False
    assert payload["mandatory_stop"]["held_out_npz_load_authorized"] is False
    prior = validation.validate_v1_v3_artifact(ROOT)
    assert prior["result_fingerprint"] == validation.EXPECTED_V1_V3_FINGERPRINT


def test_pool_statistics_combines_independent_32_and_96_batches() -> None:
    first = {
        "mean": 10.0,
        "unbiased_sample_variance": 4.0,
        "standard_error": 0.0,
        "minimum": 5.0,
        "maximum": 15.0,
    }
    second = {
        "mean": 12.0,
        "unbiased_sample_variance": 9.0,
        "standard_error": 0.0,
        "minimum": 4.0,
        "maximum": 18.0,
    }
    pooled = validation._pool_statistic_batches(((32, first), (96, second)))
    expected_mean = (32 * 10.0 + 96 * 12.0) / 128
    expected_m2 = 31 * 4.0 + 95 * 9.0
    expected_m2 += 32 * (10.0 - expected_mean) ** 2
    expected_m2 += 96 * (12.0 - expected_mean) ** 2
    expected_variance = expected_m2 / 127
    assert pooled["mean"] == pytest.approx(expected_mean)
    assert pooled["unbiased_sample_variance"] == pytest.approx(expected_variance)
    assert pooled["standard_error"] == pytest.approx(
        np.sqrt(expected_variance / 128)
    )
    assert pooled["minimum"] == 4.0
    assert pooled["maximum"] == 18.0


def test_resource_record_uses_axis_shots_and_mean_plus_minus_two_se() -> None:
    signal = {
        "accuracy_eligible": True,
        "axis_shots": {"real": 10, "imag": 20},
        "total_shots": 30,
    }
    pooled = {
        "sample_count": 32,
        "axes": {
            "cosine": {
                "rz_count": {"mean": 100.0, "standard_error": 2.0}
            },
            "sine": {
                "rz_count": {"mean": 120.0, "standard_error": 3.0}
            },
        },
    }
    resource = validation._resource_record(signal, pooled)
    assert resource["total_work_no_preparation"] == 10 * 100 + 20 * 120
    assert resource["work_interval"] == [10 * 96 + 20 * 114, 10 * 104 + 20 * 126]
    assert resource["rz_relative_standard_error_max"] == pytest.approx(0.025)


def test_decision_rules_distinguish_transfer_conditional_and_negative() -> None:
    b0 = _candidate("B0", 130.0, (130.0, 130.0), rank=6)
    b1 = _candidate("B1", 100.0, (100.0, 100.0), rank=12)
    b2 = _candidate("B2", 80.0, (75.0, 85.0), rank=6)
    b3 = _candidate("B3", 95.0, (90.0, 100.0), rank=0)
    transfer = validation._decision([b2], [b3], [b0, b1])
    assert transfer["status"] == "S2_TRANSFER_CANDIDATE_AWAITING_REVIEW"
    assert transfer["transfer_conditions"]["b2_over_b1_ratio_upper_below_0p9"]

    conditional_b2 = _candidate("B2", 94.0, (91.0, 97.0), rank=6)
    conditional = validation._decision([conditional_b2], [b3], [b0, b1])
    assert conditional["status"] == "S2_CONDITIONAL_RESOURCE_MAP_AWAITING_REVIEW"

    cheap_b1 = _candidate("B1", 50.0, (50.0, 50.0), rank=12)
    negative = validation._decision([b2], [b3], [b0, cheap_b1])
    assert negative["status"] == "COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN"
    assert any(key.startswith("B1-") for key in negative["materially_dominating_endpoints"])


def test_random_signal_and_wrapper_semantics_on_toy_case() -> None:
    hamiltonian = _toy_hamiltonian()
    preparation = validation._prepare_random(hamiltonian, "B2", 1)
    state = np.asarray([np.sqrt(0.3), 1j * np.sqrt(0.7)], dtype=np.complex128)
    state /= np.linalg.norm(state)
    energy, target = validation._target_signal(hamiltonian, state, 0.2)
    assert np.isfinite(energy)
    point = validation._random_signal_point(
        preparation,
        state,
        target,
        stage="test",
        method="B2",
        rank=1,
        q=2,
        rte_steps=1,
        cutoff=2,
    )
    assert point["corrected_raw_reconstruction_abs_error"] <= 1e-12
    assert point["normalization_attenuation_inverse_error"] <= 1e-12
    probe = validation._wrapper_semantics_probe(preparation, state)
    assert probe["overall_pass"]
    assert probe["axes"]["cosine"]["signal_component"] == "real"
    assert probe["axes"]["sine"]["signal_component"] == "imaginary"


def test_compiled_cost_path_covers_random_and_deterministic_full_wrappers() -> None:
    hamiltonian = _toy_hamiltonian()
    cache = TranspiledCircuitCostCache(maximum_entries=64)
    random = validation._compile_cost_batch(
        validation._prepare_random(hamiltonian, "B2", 1),
        stage="test",
        stream="canonical",
        method="B2",
        rank=1,
        q=1,
        rte_steps=1,
        cutoff=2,
        sample_count=1,
        master_seed=7,
        cache=cache,
    )
    deterministic = validation._compile_cost_batch(
        validation._prepare_deterministic(hamiltonian, 2),
        stage="test",
        stream="deterministic",
        method="B1",
        rank=2,
        q=1,
        rte_steps=0,
        cutoff=0,
        sample_count=None,
        master_seed=0,
        cache=cache,
    )
    for batch in (random, deterministic):
        assert set(batch["axes"]) == {"cosine", "sine"}
        assert all(
            axis["measurement_included"]
            and not axis["state_preparation_included"]
            and axis["wrapped_evolution_already_controlled"]
            and not axis["additional_control_applied"]
            for axis in batch["axes"].values()
        )


def test_v4_validator_enforces_stage_stop_and_no_resource_winner() -> None:
    payload = _common_payload(validation.V4_SCHEMA_VERSION, validation.V4_PASS_STATUS)
    payload.update(
        {
            "summary": {
                "resource_winner_selected": False,
                "expected_cost_monte_carlo_32_or_128_performed": False,
                "corrected_raw_normalization_pass": True,
                "re_im_hadamard_semantics_pass": True,
                "controlled_full_wrapper_compile_pass": True,
            },
            "deviations": [],
            "S2_development_authorized": True,
        }
    )
    validation.validate_v4_payload(_finish(payload))
    tampered = copy.deepcopy(payload)
    tampered["S3_authorized"] = True
    with pytest.raises(ValueError, match="S3"):
        validation.validate_v4_payload(_finish(tampered))


def test_s2_validator_enforces_32_or_128_and_mandatory_stop() -> None:
    payload = _common_payload(
        validation.S2_SCHEMA_VERSION,
        "S2_CONDITIONAL_RESOURCE_MAP_AWAITING_REVIEW",
    )
    payload.update(
        {
            "B2_candidates": [{"pooled_cost": {"sample_count": 32}}],
            "B3_candidates": [{"pooled_cost": {"sample_count": 128}}],
            "sampling": {"additional_sampling_authorized": False},
            "mandatory_stop_reached": True,
            "held_out_npz_loaded": False,
            "held_out_signal_cost_ranking_evaluated": False,
        }
    )
    validation.validate_s2_payload(_finish(payload))
    invalid = copy.deepcopy(payload)
    invalid["B2_candidates"][0]["pooled_cost"]["sample_count"] = 64
    with pytest.raises(ValueError, match="sample count"):
        validation.validate_s2_payload(_finish(invalid))


def test_write_json_artifact_refuses_overwrite(tmp_path: Path) -> None:
    payload = _common_payload(validation.V4_SCHEMA_VERSION, validation.V4_PASS_STATUS)
    payload.update(
        {
            "summary": {
                "resource_winner_selected": False,
                "expected_cost_monte_carlo_32_or_128_performed": False,
                "corrected_raw_normalization_pass": True,
                "re_im_hadamard_semantics_pass": True,
                "controlled_full_wrapper_compile_pass": True,
            },
            "deviations": [],
            "S2_development_authorized": True,
        }
    )
    payload = _finish(payload)
    target = tmp_path / "result.json"
    validation.write_json_artifact(
        payload,
        target,
        validator=validation.validate_v4_payload,
    )
    with pytest.raises(FileExistsError):
        validation.write_json_artifact(
            payload,
            target,
            validator=validation.validate_v4_payload,
        )
