from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py"
RUNNER = ROOT / "scripts/run_pr2_matched_accuracy_m2_transfer_contract.py"
ARTIFACT_ROOT = ROOT / "artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04"
PLAN = ARTIFACT_ROOT / "pr2_matched_accuracy_m2_transfer_zero_compute_plan_v1.json"
PLAN_SCHEMA = ARTIFACT_ROOT / "pr2_matched_accuracy_m2_transfer_plan_schema_v1.json"
RESULT_SCHEMA = ARTIFACT_ROOT / "pr2_matched_accuracy_m2_transfer_result_schema_v1.json"


def _load_contract():
    spec = importlib.util.spec_from_file_location("pr2_m2_transfer_contract_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _build_plan() -> dict:
    result_path = ROOT / contract.M1_B1_RESULT_RELATIVE
    validation_path = ROOT / contract.M1_B1_VALIDATION_RELATIVE
    return contract.build_plan(
        result=_json(result_path),
        validation=_json(validation_path),
        result_sha256=contract.M1_B1_RESULT_SHA256,
        validation_sha256=contract.M1_B1_VALIDATION_SHA256,
        source_commit="a" * 40,
        source_hashes={"source": "b" * 64},
    )


def test_exact_five_candidates_are_frozen_from_validated_result() -> None:
    plan = _build_plan()
    assert [item["candidate_id"] for item in plan["frozen_candidates"]] == [
        "B2-rank3-q1-r4-K2",
        "B2-rank3-q1-r8-K2",
        "B0-rank6-q1-r0-K0",
        "B1-rank12-q1-r0-K0",
        "B3-rank0-q8-r32-K4",
    ]
    assert [item["role"] for item in plan["frozen_candidates"][:2]] == [
        "development_actual_pareto_partial",
        "development_actual_pareto_partial",
    ]
    assert all(item["development_prediction"]["accuracy_eligible"] for item in plan["frozen_candidates"])
    assert plan["provisional_claim"]["general_optimum_claimed"] is False
    assert plan["provisional_claim"]["r4_vs_r8_exact_winner_claimed"] is False


def test_plan_freezes_196_wrapper_cap_and_zero_science() -> None:
    plan = _build_plan()
    contract.validate_plan(plan)
    assert plan["resource_caps"] == {
        "candidate_count": 5,
        "random_candidate_count": 3,
        "deterministic_or_discard_candidate_count": 2,
        "trajectories_per_random_candidate": 32,
        "random_full_wrappers": 192,
        "deterministic_or_discard_full_wrappers": 4,
        "total_full_wrappers": 196,
        "maximum_process_workers": 5,
        "blas_threads_per_worker": 1,
        "additional_trajectories": 0,
        "candidate_searches": 0,
        "held_out_snapshot_loads": 1,
    }
    assert all(value == 0 for value in plan["zero_science_counters"].values())
    assert plan["authorization"]["held_out_access_authorized"] is False
    assert plan["authorization"]["transfer_execution_authorized"] is False


def test_random_axes_share_seed_schedule_and_all_seeds_are_unique() -> None:
    plan = _build_plan()
    random = [item for item in plan["frozen_candidates"] if item["method"] in {"B2", "B3"}]
    assert all(len(item["future_trajectory_seeds"]) == 32 for item in random)
    all_seeds = [seed for item in random for seed in item["future_trajectory_seeds"]]
    assert len(all_seeds) == 96
    assert len(set(all_seeds)) == 96
    first = random[0]
    assert first["future_trajectory_seeds"][7] == contract.trajectory_seed(
        first["transfer_configuration_fingerprint"], 7
    )
    assert plan["seed_policy"]["same_random_trajectory_shared_by_cosine_and_sine"] is True


def test_prediction_and_major_underestimate_are_machine_decidable() -> None:
    item = _build_plan()["frozen_candidates"][0]
    prediction = contract.predicted_work(
        item["development_prediction"]["axis_one_shot_compiled_cost"],
        {"real": 100, "imag": 50},
        "rz_count",
    )
    assert prediction > 0
    assert contract.underestimate_fraction(predicted=prediction, actual=prediction * 1.10) == pytest.approx(0.10)
    assert contract.underestimate_fraction(predicted=prediction, actual=prediction * 1.10001) > 0.10


def _synthetic_results(*, b2_ratio: float, b2_se: float = 0.0, pareto: bool = False, major: bool = False):
    values = []
    for expected in contract.FROZEN_CANDIDATES:
        primary = b2_ratio * 100.0 if expected["method"] == "B2" else 100.0
        values.append(
            {
                "candidate_id": expected["candidate_id"],
                "method": expected["method"],
                "accuracy_eligible": True,
                "major_cost_underestimate": major if expected["method"] == "B2" else False,
                "point_six_metric_pareto": pareto if expected["method"] == "B2" else False,
                "primary_work": primary,
                "primary_standard_error": b2_se if expected["method"] == "B2" else 0.0,
            }
        )
    return values


def test_terminal_status_rules_are_fixed() -> None:
    assert contract.classify_transfer(
        _synthetic_results(b2_ratio=1.05)
    )["status"] == "TRANSFER_SUPPORTED"
    assert contract.classify_transfer(
        _synthetic_results(b2_ratio=1.20)
    )["status"] == "TRANSFER_NOT_SUPPORTED"
    assert contract.classify_transfer(
        _synthetic_results(b2_ratio=1.10, b2_se=2.0)
    )["status"] == "TRANSFER_INCONCLUSIVE"
    assert contract.classify_transfer(
        _synthetic_results(b2_ratio=1.30, pareto=True)
    )["status"] == "TRANSFER_SUPPORTED"
    assert contract.classify_transfer(
        _synthetic_results(b2_ratio=1.05, major=True)
    )["status"] == "TRANSFER_NOT_SUPPORTED"
    assert contract.classify_transfer(
        _synthetic_results(b2_ratio=1.05), implementation_gate_passed=False
    )["status"] == "IMPLEMENTATION_GATE_FAILED"


def test_validator_rejects_candidate_or_authorization_mutation() -> None:
    plan = _build_plan()
    changed = copy.deepcopy(plan)
    changed["frozen_candidates"][0]["q"] = 8
    changed["plan_fingerprint"] = contract.fingerprint(
        {key: value for key, value in changed.items() if key != "plan_fingerprint"}
    )
    with pytest.raises(ValueError, match="candidate identity or order"):
        contract.validate_plan(changed)
    enabled = copy.deepcopy(plan)
    enabled["authorization"]["held_out_access_authorized"] = True
    enabled["plan_fingerprint"] = contract.fingerprint(
        {key: value for key, value in enabled.items() if key != "plan_fingerprint"}
    )
    with pytest.raises(ValueError, match="held-out access authorized"):
        contract.validate_plan(enabled)


def test_schemas_freeze_statuses_metrics_and_no_next_stage() -> None:
    plan_schema = _json(PLAN_SCHEMA)
    assert plan_schema["properties"]["status"]["const"] == contract.STATUS
    assert plan_schema["properties"]["resource_caps"]["properties"]["total_full_wrappers"]["const"] == 196
    result_schema = _json(RESULT_SCHEMA)
    assert set(result_schema["properties"]["status"]["enum"]) == set(contract.TRANSFER_STATUSES)
    assert result_schema["properties"]["primary_metric"]["const"] == "rz_count"
    assert result_schema["properties"]["next_stage_authorized"]["const"] is False


def test_contract_and_runner_are_standard_library_only() -> None:
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "importlib",
        "json",
        "math",
        "pathlib",
        "subprocess",
        "typing",
    }
    for path in (SOURCE, RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert imported <= allowed


def test_committed_zero_compute_plan_is_valid_when_present() -> None:
    if not PLAN.exists():
        pytest.skip("plan is generated only after the source commit")
    plan = _json(PLAN)
    contract.validate_plan(plan)
    assert plan["resource_caps"]["total_full_wrappers"] == 196
    assert all(value == 0 for value in plan["zero_science_counters"].values())
