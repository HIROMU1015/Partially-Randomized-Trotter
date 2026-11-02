from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from trotterlib import research_direction_pd_s1_posthoc as posthoc


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "artifacts/research_direction_pd_fair_comparison/2026-09-26/"
    "pd_s1_fair_comparison_v2.json"
)


def _source() -> dict[str, object]:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def test_fixed_source_identity_and_formal_classification() -> None:
    assert hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() == (
        posthoc.SOURCE_RESULT_FILE_SHA256
    )
    source = _source()
    posthoc.validate_source_result(source)
    changed = copy.deepcopy(source)
    changed["classification"]["primary_case"] = "A"
    with pytest.raises(ValueError):
        posthoc.validate_source_result(changed)


def test_posthoc_result_preserves_case_and_separates_primary_baselines() -> None:
    body = posthoc.reanalyze_s1(_source())
    assert body["source_result"]["formal_classification_preserved"][
        "primary_case"
    ] == "B"
    assert body["posthoc_interpretation"]["selection_label"] == (
        "A_equivalent_on_frozen_candidate_set_under_B4"
    )
    assert body["posthoc_interpretation"][
        "formal_case_b_and_boundary_are_caused_by_b1a_ablation_not_b1b"
    ] is True
    assert body["decision"]["additional_pd_s2_computation_authorized"] is False
    for scope in posthoc.SCOPES:
        assert body["scope_reanalysis"][scope][
            "primary_baseline_selection_consensus"
        ] is True


def test_b1a_and_construction_diagnostics_match_frozen_rows() -> None:
    body = posthoc.reanalyze_s1(_source())
    b1a = body["b1a_inner_substep_diagnostic"]
    assert b1a["b1a_objective_outer_stage_values"] == [22]
    assert [row["inner_hd_substeps"] for row in b1a["records"]] == [
        8,
        16,
        32,
        64,
        128,
    ]
    assert b1a[
        "same_tail_setting_can_be_made_b4_feasible_by_increasing_only_m_d"
    ] is False
    decomposition = body["construction_decomposition"]
    assert decomposition["same_formula_label"] is True
    assert decomposition["nested_over_native_ratios"][
        "b4_objective"
    ] == pytest.approx(13.229689510687095)
    assert decomposition["original_case_d_classifier_dimension"] == (
        "selected_formula_label_only"
    )


def test_finalized_posthoc_result_detects_mutation() -> None:
    artifact = posthoc.finalize_result(
        posthoc.reanalyze_s1(_source()),
        provenance={"test": True},
        source_evidence=[],
    )
    posthoc.validate_result(artifact)
    changed = copy.deepcopy(artifact)
    changed["decision"]["additional_pd_s2_computation_authorized"] = True
    with pytest.raises(ValueError):
        posthoc.validate_result(changed)
