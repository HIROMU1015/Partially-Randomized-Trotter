from __future__ import annotations

import json
from pathlib import Path

import pytest

from trotterlib.pr2_matched_accuracy_m1_b1_result_validation import (
    METRICS,
    RESULT_FINGERPRINT,
    _lower_envelope,
    _pareto_frontier,
    _spearman,
    fingerprint,
    validate_and_analyze,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/"
    "pr2_matched_accuracy_m1_b1_compile_map_result_v2.json"
)


def _record(name: str, values: tuple[float, ...], shots: int) -> dict:
    return {
        "candidate": {
            "candidate_id": name,
            "candidate_fingerprint": name * 64,
            "method": "B2",
            "rank": 3,
            "q": 1,
            "delta": 0.8,
            "r": 1,
            "K": 2,
        },
        "axis_shots": {"real": shots, "imag": 0},
        "matched_accuracy_compiled_work_no_state_preparation": dict(zip(METRICS, values, strict=True)),
    }


def test_result_fingerprint_is_still_exact() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "result_fingerprint"}
    assert payload["result_fingerprint"] == RESULT_FINGERPRINT
    assert fingerprint(body) == RESULT_FINGERPRINT


def test_pareto_and_state_preparation_envelope_are_independent() -> None:
    low_intercept = _record("a", (1, 3, 1, 3, 3, 1), 10)
    low_slope = _record("b", (3, 1, 3, 1, 1, 3), 1)
    dominated = _record("c", (4, 4, 4, 4, 4, 4), 4)
    assert {item["candidate"]["candidate_id"] for item in _pareto_frontier([low_intercept, low_slope, dominated])} == {"a", "b"}
    envelope = _lower_envelope([low_intercept, low_slope, dominated])
    assert [item["candidate"]["candidate_id"] for item in envelope] == ["a", "b"]
    assert envelope[1]["start_P_inclusive"] == pytest.approx(2.0 / 9.0)


def test_spearman_handles_ties() -> None:
    assert _spearman([1, 1, 3, 4], [2, 2, 6, 8]) == pytest.approx(1.0)


def test_full_completed_result_validation() -> None:
    payload = validate_and_analyze(ROOT)
    assert payload["integrity"]["all_gates_passed"] is True
    assert payload["integrity"]["wrapper_records"] == 12_448
    assert payload["integrity"]["checkpoint_files"] == 210
    assert payload["analysis"]["accuracy_eligible_cells"] == 206
    assert len(payload["analysis"]["actual_six_metric_pareto"]) == 2
    assert payload["external_research_review"]["decision"] == "CONTINUE_RESOURCE_STUDY"
