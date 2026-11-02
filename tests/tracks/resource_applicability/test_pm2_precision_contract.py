"""Contract/inventory tests only; no precision sweep or scientific data access."""
import ast
import copy
import hashlib
from pathlib import Path

import jsonschema
import pytest

from trottertracks.resource_applicability import pm2_precision_contract as c


@pytest.fixture(scope="module")
def saved():
    root = Path(__file__).parents[3]
    values, audit = c.load_inputs(root)
    return values, audit, c.candidate_inventory(values)


def test_exact_saved_domain_including_original_ineligible(saved):
    _, audit, inventory = saved
    assert len(audit) == 4
    assert all(a["commit_blob_identical"] for a in audit.values())
    development = inventory["development"]
    assert len(development) == 218
    assert sum(r["reference_accuracy_eligible"] for r in development) == 214
    assert len({r["candidate_fingerprint"] for r in development}) == 218
    assert {r["parameters"]["rank"] for r in development if r["signal_source"] == "pm1"} == {4, 5}


def test_transfer_stays_original_five(saved):
    transfer = saved[2]["transfer_fixed_five"]
    assert {r["candidate_id"] for r in transfer} == set(c.COMMON_FIVE)
    assert all(r["dataset"] == "transfer_fixed_five" for r in transfer)
    assert not any(r["parameters"]["rank"] in (4, 5) for r in transfer)


def test_field_coverage_and_paired_saved_samples(saved):
    for rows in saved[2].values():
        for row in rows:
            assert row["saved_field_coverage"]["axis_metric_means"]
            assert not row["saved_field_coverage"]["source_values_copied_or_reevaluated"]
            assert row["cost_sample_inventory"]["paired_axis_identities_identical"]
            assert row["cost_sample_inventory"]["complete_saved_samples"]
            assert row["cost_sample_inventory"]["sample_count"] == (32 if row["parameters"]["method"] in ("B2", "B3") else 1)


def test_separate_signal_cost_indices_resolve_exact_records(saved):
    values, _, inventory = saved
    for row in inventory["development"]:
        if row["signal_source"] == "m1_signal":
            signal = values["m1_signal"]["signal_records"][row["signal_row_index"]]
            cost = values["m1_compile"]["compile_map"][row["compile_row_index"]]
            assert signal["candidate"] == cost["candidate"]
            assert signal["candidate_id"] == row["candidate_id"]


@pytest.mark.parametrize("field,value", [("minimum", 0.01), ("maximum", 0.2), ("adaptive_precision_sampling", True)])
def test_precision_design_mutation_rejected(field, value):
    settings = c.contract_settings()
    settings["epsilon_design"][field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(settings, c.plan_schema())


def test_valid_contract_schema_and_exact_reference():
    settings = c.contract_settings()
    jsonschema.Draft202012Validator.check_schema(c.plan_schema())
    jsonschema.validate(settings, c.plan_schema())
    assert settings["epsilon_design"]["reference"] == 0.05
    assert settings["accounting"]["alpha_axis"] == 0.025
    assert settings["epsilon_design"]["equality_boundary_eligible"] is False
    assert settings["accounting"]["fixed_C_eff_allowed"] is False
    assert settings["P_design"]["maximum"] is None
    assert settings["P_design"]["P_grid"] is False


def test_preparation_is_not_analysis_permission():
    settings = c.contract_settings()
    assert settings["status"] == c.STATUS
    assert settings["analysis_label"] == "POSTHOC_SAVED_VALUES_ONLY"
    assert settings["permissions"]["precision_analysis"] is False
    assert settings["mandatory_stop"] is True
    assert settings["research_decision"] is None
    assert settings["uncertainty"]["formal_CI"] is False
    assert not any(hasattr(c, name) for name in ("analyze", "precision_sweep", "evaluate_shots", "lower_envelope", "run"))


@pytest.mark.parametrize("status", ["CONTINUE_RESOURCE_STUDY", "NARROW_TO_TECHNICAL_NOTE", "TRANSFER_SUPPORTED"])
def test_reserved_result_has_no_research_decision_status(status):
    result = result_fixture()
    result["status"] = status
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, c.reserved_result_schema())


def result_fixture():
    return {"schema_version": "track_a_pm2_precision_result_v1", "status": c.COMPLETE,
        "analysis_label": "POSTHOC_SAVED_VALUES_ONLY", "preparation_manifest_sha256": "a" * 64,
        "input_identity": {}, "domain_counts": {"development": 218, "transfer_fixed_five": 5},
        "reference_reproduction": {"passed": True}, "output_files": [], "failure_reason": None,
        "new_science_counts": {k: 0 for k in c.ZERO_ACTIONS[4:]},
        "mandatory_stop": True, "next_stage_authorized": False, "research_decision": None}


def test_reserved_success_and_failure_schema():
    jsonschema.Draft202012Validator.check_schema(c.reserved_result_schema())
    result = result_fixture()
    jsonschema.validate(result, c.reserved_result_schema())
    result.update(status=c.FAILURE, failure_reason="input mismatch", reference_reproduction=None)
    jsonschema.validate(result, c.reserved_result_schema())


def test_reserved_complete_requires_reference_gate_pass():
    result = result_fixture()
    result["reference_reproduction"]["passed"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, c.reserved_result_schema())


@pytest.mark.parametrize("field,value", [("next_stage_authorized", True), ("mandatory_stop", False), ("research_decision", "GO")])
def test_reserved_result_stop_cannot_be_relaxed(field, value):
    result = result_fixture()
    result[field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, c.reserved_result_schema())


def test_reserved_result_forbids_new_science():
    result = result_fixture()
    result["new_science_counts"]["compile"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, c.reserved_result_schema())


def test_input_bytes_and_blob_mismatch_rejected():
    with pytest.raises(ValueError, match="SHA"):
        c.verified_bytes(b"changed", b"original", hashlib.sha256(b"original").hexdigest())
    with pytest.raises(ValueError, match="blob"):
        c.verified_bytes(b"changed", b"original", hashlib.sha256(b"changed").hexdigest())


def test_pair_mismatch_and_missing_samples_rejected(saved):
    axis = next(r["compiled_axes"] for r in saved[0]["m1_compile"]["compile_map"] if r["candidate"]["method"] == "B2")
    changed = copy.deepcopy(axis)
    changed["sine"]["retained_trajectory_records"][0]["trajectory_seed"] += 1
    with pytest.raises(ValueError, match="pairing"):
        c.pair_inventory(changed, True)
    changed = copy.deepcopy(axis)
    changed["cosine"]["trajectory_records_truncated"] = True
    with pytest.raises(ValueError, match="missing"):
        c.pair_inventory(changed, True)


def test_stdlib_only_and_explicit_text_allowlist():
    tree = ast.parse(Path(c.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(n and n.startswith(("numpy", "scipy", "qiskit", "trotterlib", "cupy")) for n in imports)
    assert len(c.INPUTS) == 4
    assert all(p.endswith(".json") and ".runtime" not in p for p, _ in c.INPUTS.values())
