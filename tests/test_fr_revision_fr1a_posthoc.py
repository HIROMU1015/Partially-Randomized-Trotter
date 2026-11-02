from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trotterlib.fr_revision_fr1a_posthoc import (
    METHOD_ORDER,
    SOURCE_DECISION,
    build_fr1a_expected,
    run_fr_revision_fr1a_posthoc,
    validate_fr1a_expected,
    validate_fr1a_payload,
    write_fr1a_expected,
    write_fr1a_payload,
)


SOURCE = Path(
    "artifacts/finite_rte_phase_amplitude/2026-09-26/"
    "finite_rte_phase_amplitude_fr1_v1.json"
)
FROZEN_PREREGISTRATION = Path(
    "artifacts/finite_rte_phase_amplitude/2026-09-26/"
    "fr1_preregistration_frozen.md"
)
PLAN = Path("docs/research/fr_revision_fr1a_posthoc_plan.md")


@pytest.fixture(scope="module")
def source() -> dict:
    return json.loads(SOURCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload(source: dict) -> dict:
    return run_fr_revision_fr1a_posthoc(
        source,
        source_result_path=SOURCE,
        frozen_preregistration_path=FROZEN_PREREGISTRATION,
        posthoc_plan_path=PLAN,
        provenance={"test": True},
    )


def test_expected_specification_is_frozen_and_scope_limited(source: dict) -> None:
    expected = build_fr1a_expected(source)
    validate_fr1a_expected(expected)
    assert expected["posthoc"] is True
    assert expected["expected_condition_count"] == 33
    assert expected["expected_state_count"] == 99
    assert len(expected["expected_condition_ids"]) == 33
    assert expected["method_order"] == list(METHOD_ORDER)
    assert expected["old_decision_must_remain"] == SOURCE_DECISION
    assert expected["fr1b_started"] is False
    assert expected["h4_or_h12_evaluation_performed"] is False


def test_reconstruction_scalar_reanalysis_and_stop_contract(payload: dict) -> None:
    validate_fr1a_payload(payload)
    assert payload["source_audit"]["all_pass"]
    assert payload["reconstruction_audit"]["all_pass"]
    assert payload["reconstruction_audit"]["all_state_fingerprints_match"]
    assert payload["reconstruction_audit"]["maximum_numeric_abs_difference"] <= 2e-12
    assert payload["summary"]["condition_count"] == 33
    assert payload["summary"]["state_record_count"] == 99
    assert payload["summary"]["main_record_count"] == 5
    assert payload["summary"]["method_record_count"] == 99 * len(METHOD_ORDER)
    assert payload["summary"]["soundness_failure_count"] == 0
    assert payload["summary"]["execution_valid"]
    assert payload["summary"]["old_decision"] == SOURCE_DECISION
    assert payload["summary"]["old_decision_changed"] is False
    assert payload["summary"]["research_go_authorized"] is False
    assert payload["fr1b_started"] is False


def test_involution_scalar_explains_old_gain_on_frozen_main_rows(payload: dict) -> None:
    assert payload["summary"]["classification"] == "POSTHOC_SCALAR_EXPLAINS_OLD_GAIN"
    assert payload["summary"]["scalar_explains_old_gain"]
    assert payload["summary"]["residual_fr_increment"] is False
    assert payload["summary"]["oracle_only_increment"] is False
    assert payload["summary"]["common_one_sided_certifications"] == []
    assert payload["summary"]["optimized_one_sided_certifications"] == []

    main_rows = [
        state
        for condition in payload["conditions"]
        for state in condition["state_records"]
        if state["scope"] == "primary"
        and state["state_label"] == "analytic_mixture_state"
    ]
    assert len(main_rows) == 5
    for row in main_rows:
        assert list(row["methods"]) == list(METHOD_ORDER)
        assert row["all_applicable_bounds_pass"]
        assert row["methods"]["DENSE_ORACLE"]["information_level"] == "I2"
        assert row["methods"]["DENSE_ORACLE"]["oracle"] is True
        assert row["methods"]["OPT_SCALAR_NORM"]["oracle"] is False

    for condition in payload["conditions"]:
        scalar = condition["positive_scalar"]
        assert scalar["gamma_mid"] > 0.0
        assert scalar["centered_hermitian_spectral_width"] == pytest.approx(
            0.0, abs=2e-15
        )
        assert scalar["gamma_opt_norm_certification"]["finite_grid_maximum_used"] is False


def test_payloads_are_tamper_evident_and_non_overwriting(
    source: dict, payload: dict, tmp_path: Path
) -> None:
    expected = build_fr1a_expected(source)
    expected_path = tmp_path / "expected.json"
    result_path = tmp_path / "result.json"
    write_fr1a_expected(expected, expected_path)
    write_fr1a_payload(payload, result_path)
    assert expected_path.exists()
    assert result_path.exists()

    tampered = deepcopy(payload)
    tampered["summary"]["old_decision_changed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_fr1a_payload(tampered)

    result_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_fr1a_payload(payload, result_path)
