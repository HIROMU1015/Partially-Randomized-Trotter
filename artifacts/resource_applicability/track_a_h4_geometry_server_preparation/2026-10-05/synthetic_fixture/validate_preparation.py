"""Contract/schema checks only, with explicitly artificial checkpoint examples."""
import copy
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
from jsonschema import Draft202012Validator


def main():
    root, output = map(Path, sys.argv[1:])
    plan = json.loads((output / "zero_compute_plan_draft_v0.json").read_text())
    schema = json.loads((output / "zero_compute_plan_draft_schema_v0.json").read_text())
    checkpoints = json.loads((output / "future_checkpoint_schema_draft_v0.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator.check_schema(checkpoints)
    validator = Draft202012Validator(schema)
    validator.validate(plan)
    cases = [{"name": "current exact draft validates", "pass": True}]
    for key, value in [("science_execution_authorized", True), ("geometry_frozen", True), ("sealed", True),
                       ("production_source_commit", "f" * 40), ("master_seed", 20261005),
                       ("next_stage_authorized", True), ("mandatory_stop", False), ("research_decision", "GO")]:
        changed = copy.deepcopy(plan)
        changed[key] = value
        assert list(validator.iter_errors(changed)), key
        cases.append({"name": f"reject unauthorized/fabricated {key}", "pass": True})
    changed = copy.deepcopy(plan)
    changed["candidate_templates"].pop()
    assert list(validator.iter_errors(changed))
    cases.append({"name": "reject altered template domain", "pass": True})
    # Artificial schema fixtures below are never used as molecular/source identities.
    sample = {key: "SYNTHETIC_SCHEMA_EXAMPLE" for key in checkpoints["required"]}
    for key in ["hamiltonian_sha256", "df_sha256", "state_sha256", "candidate_fingerprint", "compiler_fingerprint", "environment_fingerprint"]:
        sample[key] = "a" * 64
    sample.update(source_commit="a" * 40, axis="cosine", trajectory_seed=None, trajectory_index=0,
                  status="COMPLETE", metrics={key: 1 for key in plan["precision_postprocessing_proposal"]["metrics"]},
                  cache_reuse=False, actual_transpile_invocation_id="SYNTHETIC_INVOCATION", cache_owner_wrapper_key=None, mandatory_stop=True)
    checkpoint_validator = Draft202012Validator(checkpoints)
    checkpoint_validator.validate(sample)
    cases.append({"name": "artificial complete noncached checkpoint validates", "pass": True})
    for key, value in [("metrics", None), ("actual_transpile_invocation_id", None), ("axis", "INVALID"),
                       ("hamiltonian_sha256", "not-a-hash"), ("source_commit", "not-a-source-commit")]:
        changed = copy.deepcopy(sample)
        changed[key] = value
        assert list(checkpoint_validator.iter_errors(changed)), key
        cases.append({"name": f"reject malformed/incomplete checkpoint {key}", "pass": True})
    changed = copy.deepcopy(sample)
    changed["cache_reuse"] = True
    assert list(checkpoint_validator.iter_errors(changed))
    changed["cache_owner_wrapper_key"] = "SYNTHETIC_COMPLETED_OWNER"
    checkpoint_validator.validate(changed)
    cases.append({"name": "cached completion requires explicit completed cache owner", "pass": True})
    fingerprint = plan.pop("draft_fingerprint")
    actual = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert fingerprint == actual
    cases.append({"name": "draft fingerprint matches content", "pass": True})
    audit = json.loads((output / "static_audit_v0.json").read_text())
    for source in audit["source_hashes"]:
        assert hashlib.sha256((root / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    for record in audit["allowed_json_identity"]:
        assert hashlib.sha256((root / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    cases.append({"name": "all 247 old sources and 6 allowed evidence files unchanged", "pass": True})
    environment = json.loads((output / "environment_inventory_v0.json").read_text())
    for name, record in environment["dependencies"].items():
        assert md.version(name) == record["version"]
        installed_record = md.distribution(name).read_text("RECORD")
        assert hashlib.sha256(installed_record.encode()).hexdigest() == record["installed_record_sha256"]
    cases.append({"name": "existing dependency versions and RECORD metadata unchanged", "pass": True})
    fixture = output / "synthetic_fixture"
    frozen = json.loads((fixture / "task_definitions_v0.json").read_text())
    assert hashlib.sha256((fixture / "synthetic_fixture.py").read_bytes()).hexdigest() == frozen["source_sha256"]
    benchmark = json.loads((fixture / "benchmark_result_v0.json").read_text())
    tests = json.loads((fixture / "synthetic_tests_v0.json").read_text())
    assert benchmark["actual_transpile_calls"] == 120 and tests["actual_transpiles"] == 4
    assert benchmark["wall_s"] < 1800
    assert all(c["same_gate_metrics_as_one_worker"] and not c["failures"] for c in benchmark["conditions"])
    cases.append({"name": "frozen fixture and 124-transpile/30-minute budgets valid", "pass": True})
    result = {"status": "PREPARATION_STATIC_CONTRACT_CHECKS_PASS", "checks": cases, "passed": len(cases),
              "failed": 0, "skipped": 0, "jsonschema_version": md.version("jsonschema"),
              "new_transpile_calls": 0, "science_tests_run": 0, "molecular_or_runtime_access": 0,
              "gpu_access": 0, "shared_environment_mutations": 0}
    with (output / "preparation_checks_v0.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ["status", "passed", "failed", "skipped", "new_transpile_calls"]}))


if __name__ == "__main__":
    main()
