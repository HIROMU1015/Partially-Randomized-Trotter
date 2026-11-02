from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py"
RUNNER = ROOT / "scripts/run_pr2_matched_accuracy_m2_transfer_contract.py"
ARTIFACT_ROOT = ROOT / "artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04"
PLAN = ARTIFACT_ROOT / "pr2_matched_accuracy_m2_transfer_zero_compute_plan_v1.json"
PLAN_SCHEMA = ARTIFACT_ROOT / "pr2_matched_accuracy_m2_transfer_plan_schema_v2.json"
RESULT_SCHEMA = ARTIFACT_ROOT / "pr2_matched_accuracy_m2_transfer_result_schema_v2.json"


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


@pytest.mark.parametrize("pareto", [False, True])
def test_twenty_percent_underestimate_cannot_support_good_actual_ratio(pareto) -> None:
    values = _synthetic_results(b2_ratio=1.05, pareto=pareto)
    for item in values:
        if item["method"] == "B2":
            actual = item["primary_work"]
            fraction = contract.underestimate_fraction(predicted=actual / 1.20, actual=actual)
            assert fraction == pytest.approx(0.20)
            item["major_cost_underestimate"] = fraction > contract.MAJOR_UNDERESTIMATE_FRACTION
    decision = contract.classify_transfer(values)
    assert decision["status"] == "TRANSFER_NOT_SUPPORTED"
    assert decision["primary_ratio"] is None
    assert decision["usable_b2_candidate_ids"] == []


@pytest.mark.parametrize("usable_ratio,se,status", [
    (1.05, 0.0, "TRANSFER_SUPPORTED"),
    (1.20, 0.0, "TRANSFER_NOT_SUPPORTED"),
    (1.10, 2.0, "TRANSFER_INCONCLUSIVE"),
])
def test_all_ratio_statuses_ignore_cheaper_unusable_b2(usable_ratio, se, status) -> None:
    values = _synthetic_results(b2_ratio=usable_ratio, b2_se=se)
    values[0].update(primary_work=50.0, major_cost_underestimate=True, point_six_metric_pareto=True)
    decision = contract.classify_transfer(values)
    assert decision["status"] == status
    assert decision["primary_ratio"]["point"] == pytest.approx(usable_ratio)
    assert decision["best_b2_candidate_id"] == values[1]["candidate_id"]
    assert decision["usable_b2_candidate_ids"] == [values[1]["candidate_id"]]


@pytest.mark.parametrize("empty_usable,status", [
    (True, "TRANSFER_NOT_SUPPORTED"), (False, "TRANSFER_INCONCLUSIVE"),
])
def test_empty_set_precedence_is_explicit(empty_usable, status) -> None:
    values = _synthetic_results(b2_ratio=1.05, major=empty_usable)
    for item in values:
        if item["method"] != "B2":
            item["accuracy_eligible"] = False
    decision = contract.classify_transfer(values)
    assert decision["status"] == status
    assert decision["primary_ratio"] is None
    assert decision["eligible_endpoint_candidate_ids"] == []


def test_ineligible_b2_cannot_supply_pareto_or_ratio_support() -> None:
    values = _synthetic_results(b2_ratio=1.20)
    values[0].update(primary_work=50.0, accuracy_eligible=False, point_six_metric_pareto=True)
    decision = contract.classify_transfer(values)
    assert decision["status"] == "TRANSFER_NOT_SUPPORTED"
    assert decision["primary_ratio"]["point"] == pytest.approx(1.20)


@pytest.mark.parametrize("field,replacement", [
    ("method", "B3"), ("major_cost_underestimate", None), ("accuracy_eligible", 1),
])
def test_classifier_rejects_ambiguous_usable_inputs(field, replacement) -> None:
    values = _synthetic_results(b2_ratio=1.05)
    values[0][field] = replacement
    with pytest.raises(ValueError, match="classification"):
        contract.classify_transfer(values)


def test_classifier_rejects_duplicate_candidate() -> None:
    values = _synthetic_results(b2_ratio=1.05)
    with pytest.raises(ValueError, match="exactly the five"):
        contract.classify_transfer(values + [copy.deepcopy(values[0])])


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
    assert contract.STATUS in plan_schema["properties"]["status"]["enum"]
    assert plan_schema["properties"]["resource_caps"]["properties"]["total_full_wrappers"]["const"] == 196
    result_schema = _json(RESULT_SCHEMA)
    assert set(result_schema["properties"]["status"]["enum"]) == set(contract.TRANSFER_STATUSES)
    assert result_schema["properties"]["primary_metric"]["const"] == "rz_count"
    assert result_schema["properties"]["next_stage_authorized"]["const"] is False
    assert plan_schema["properties"]["terminal_decision_rule"]["const"] == contract.TERMINAL_DECISION_RULE
    assert result_schema["properties"]["terminal_decision_rule"]["const"] == contract.TERMINAL_DECISION_RULE
    assert plan_schema["properties"]["materiality_and_uncertainty_rule"]["const"] == contract.MATERIALITY_RULE
    Draft202012Validator.check_schema(plan_schema)
    Draft202012Validator.check_schema(result_schema)
    Draft202012Validator(plan_schema).validate(_build_plan())


@pytest.mark.parametrize("rule_name", ["terminal_decision_rule", "materiality_and_uncertainty_rule"])
def test_plan_validator_and_schema_reject_eligible_only_ratio_even_when_rehashed(rule_name) -> None:
    plan = _build_plan()
    plan[rule_name]["ratio_numerator_candidate_set" if rule_name == "terminal_decision_rule"
        else "numerator_candidate_set"] = "accuracy_eligible_B2"
    plan["plan_fingerprint"] = contract.fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    )
    with pytest.raises(ValueError, match="usable B2"):
        contract.validate_plan(plan)
    assert not Draft202012Validator(_json(PLAN_SCHEMA)).is_valid(plan)


def _synthetic_result_payload() -> dict:
    values = _synthetic_results(b2_ratio=1.05)
    for item in values:
        item["transfer_support_usable"] = (
            item["method"] == "B2" and item["accuracy_eligible"]
            and not item["major_cost_underestimate"]
        )
    return {
        "schema_version": "pr2_matched_accuracy_m2_transfer_result_v2",
        "status": "TRANSFER_SUPPORTED",
        "transfer_plan_fingerprint": "a" * 64,
        "candidate_results": values,
        "primary_metric": contract.PRIMARY_METRIC,
        "point_pareto_metrics": list(contract.METRICS),
        "held_out_reoptimized": False,
        "held_out_candidate_searches": 0,
        "additional_trajectories_executed": 0,
        "next_stage_authorized": False,
        "automatic_next_stage": None,
        "terminal_decision_rule": contract.TERMINAL_DECISION_RULE,
    }


def test_reserved_result_schema_requires_usable_b2_for_support() -> None:
    validator = Draft202012Validator(_json(RESULT_SCHEMA))
    result = _synthetic_result_payload()
    validator.validate(result)
    for item in result["candidate_results"]:
        if item["method"] == "B2":
            item["major_cost_underestimate"] = True
            item["transfer_support_usable"] = False
    assert not validator.is_valid(result)
    result["status"] = "TRANSFER_NOT_SUPPORTED"
    validator.validate(result)
    result["candidate_results"][0]["transfer_support_usable"] = True
    assert not validator.is_valid(result)


def test_draft_plan_is_explicitly_not_commit_bound() -> None:
    plan = contract.build_plan(
        result=_json(ROOT / contract.M1_B1_RESULT_RELATIVE),
        validation=_json(ROOT / contract.M1_B1_VALIDATION_RELATIVE),
        result_sha256=contract.M1_B1_RESULT_SHA256,
        validation_sha256=contract.M1_B1_VALIDATION_SHA256,
        source_commit="a" * 40, source_hashes={"source": "b" * 64},
        source_committed=False,
    )
    assert plan["status"] == contract.DRAFT_STATUS
    assert plan["source_binding_status"] == "WORKTREE_DRAFT"
    contract.validate_plan(plan)
    Draft202012Validator(_json(PLAN_SCHEMA)).validate(plan)
    plan["status"] = contract.STATUS
    with pytest.raises(ValueError, match="source binding"):
        contract.validate_plan(plan)


def test_plan_build_does_not_resolve_stat_hash_or_load_held_out(monkeypatch) -> None:
    def guard(original):
        def wrapped(path, *args, **kwargs):
            assert "held_out" not in str(path), f"forbidden held-out access: {path}"
            return original(path, *args, **kwargs)
        return wrapped
    for name in ("open", "stat", "resolve"):
        monkeypatch.setattr(Path, name, guard(getattr(Path, name)))
    plan = _build_plan()
    assert plan["held_out_target"]["npz_loaded_during_planning"] is False
    assert all(value == 0 for value in plan["zero_science_counters"].values())


def test_revised_plan_preserves_five_configurations_seeds_and_resource_caps() -> None:
    old_plan = _json(PLAN)
    new_plan = _build_plan()
    for field in ("frozen_candidates", "resource_caps", "accuracy_and_shot_rule", "seed_policy",
                  "compiled_cost_rule", "prediction_and_underestimate_rule", "held_out_target",
                  "input_identity", "prohibitions", "authorization", "zero_science_counters"):
        assert new_plan[field] == old_plan[field]


def test_official_plan_runner_requires_byte_identical_commit_sources(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("pr2_m2_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import subprocess
    def fake_run(command, **_kwargs):
        assert command == ["git", "show", f"{'a' * 40}:synthetic_source.py"]
        return subprocess.CompletedProcess(command, 0, stdout=b"committed source", stderr=b"")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    digest = contract.hashlib.sha256(b"committed source").hexdigest()
    runner._require_committed_sources("a" * 40, {"synthetic_source.py": digest})
    with pytest.raises(ValueError, match="source differs from committed blob"):
        runner._require_committed_sources("a" * 40, {"synthetic_source.py": "b" * 64})


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


def test_original_v1_plan_and_contract_are_preserved_as_review_history() -> None:
    assert contract.file_sha256(PLAN) == "7b9a2c4e9fbf4e9c58b6a5e5402239b9527c49d27ccf8e8f0f1a5ae080097132"
    original_contract = ROOT / "docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md"
    assert contract.file_sha256(original_contract) == "a78d70c493125ef5321a989468dc41961ee5d3e053e6879a75db4b68d73e2f67"
    plan = _json(PLAN)
    assert plan["schema_version"] == "pr2_matched_accuracy_m2_transfer_zero_compute_plan_v1"
    assert plan["plan_fingerprint"] == contract.fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    )
    with pytest.raises(ValueError, match="unexpected plan schema"):
        contract.validate_plan(plan)
    assert plan["resource_caps"]["total_full_wrappers"] == 196
    assert all(value == 0 for value in plan["zero_science_counters"].values())
