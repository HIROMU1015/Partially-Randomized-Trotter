"""Accept independently valid owners, then refuse cross-scope cache links."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator
import contract_validator_v1 as v
from run_contract_tests import install_guards, protected_attempts, import_attempts

HERE = Path(__file__).parent


def main():
    install_guards()
    examples = json.loads((HERE / "synthetic_records_v1.json").read_text())
    schema = Draft202012Validator(json.loads((HERE / "checkpoint_schema_v1.json").read_text()))
    owner, reuse, baseline = examples["records"]
    original_records = {r["wrapper_key"]: r for r in [owner, reuse, baseline]}
    cases = []
    fields = ["geometry", "candidate_template", "axis", "hamiltonian_sha256",
              "df_sha256", "state_sha256", "source_commit", "compiler_fingerprint", "environment_fingerprint"]
    for field in fields:
        foreign = copy.deepcopy(owner)
        circuit = copy.deepcopy(examples["declarative_circuit"])
        if field == "geometry":
            foreign[field] = "0.80"
        elif field == "candidate_template":
            foreign[field].update(L_D=6, template_id="B2-rank6-q1-r4-K2")
        elif field == "axis":
            foreign[field] = circuit["axis"] = "sine"
        else:
            foreign[field] = "9"*(40 if field == "source_commit" else 64)
        foreign["candidate_fingerprint"] = v.candidate_fingerprint(foreign)
        foreign["trajectory_seed"] = v.trajectory_seed(foreign, examples["master_seed_is_artificial"])
        foreign["wrapper_key"] = v.wrapper_key(foreign)
        foreign["numerical_circuit_fingerprint"] = v.circuit_fingerprint(circuit)
        foreign["actual_transpile_invocation_id"] = "ARTIFICIAL_FOREIGN_" + field
        records = dict(original_records); records[foreign["wrapper_key"]] = foreign
        ledger = copy.deepcopy(examples["external_ledger"])
        ledger["entries"][foreign["wrapper_key"]] = {"status": "COMPLETE", "record_sha256": v.record_digest(foreign)}
        ledger["reservations"][foreign["actual_transpile_invocation_id"]] = {"status": "COMPLETE", "wrapper_key": foreign["wrapper_key"]}
        registry = dict(examples["independent_numerical_registry"]); registry[foreign["wrapper_key"]] = foreign["numerical_circuit_fingerprint"]
        v.validate_record(foreign, {k: foreign[k] for k in v.IDENTITY_FIELDS}, circuit, records,
                          ledger, registry, examples["master_seed_is_artificial"], schema)
        cases.append({"name": "independently valid owner with different " + field, "expected": "accept", "passed": True})
        linked = copy.deepcopy(reuse); linked["cache_owner_wrapper_key"] = foreign["wrapper_key"]
        ledger["entries"][linked["wrapper_key"]]["record_sha256"] = v.record_digest(linked)
        try:
            v.validate_record(linked, {k: linked[k] for k in v.IDENTITY_FIELDS},
                              examples["declarative_circuit"], records, ledger, registry,
                              examples["master_seed_is_artificial"], schema)
        except ValueError as error:
            assert str(error) == "cross-identity reuse", str(error)
        else:
            raise AssertionError("Cross-identity reuse accepted: " + field)
        cases.append({"name": "valid foreign owner cannot be reused across " + field, "expected": "reject", "passed": True})
    assert not protected_attempts and not import_attempts
    result = {"status": "VALID_FOREIGN_OWNER_CACHE_REFUSAL_PASS", "cases": cases,
              "passed": len(cases), "failed": 0, "skipped": 0,
              "science_build_compile_transpile": 0, "molecular_runtime_gpu_access": 0,
              "protected_attempts": protected_attempts, "scientific_import_attempts": import_attempts}
    with (HERE / "additional_contract_checks_v1.json").open("x") as handle:
        json.dump(result, handle, indent=2); handle.write("\n")
    print(json.dumps({"status": result["status"], "passed": len(cases), "failed": 0, "skipped": 0}))


if __name__ == "__main__":
    main()
