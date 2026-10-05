"""Artificial JSON tests with protected input/import guards; no science execution."""
import builtins
import copy
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from jsonschema import Draft202012Validator, ValidationError
import contract_validator_v1 as v

HERE = pathlib.Path(__file__).parent
MASTER = 7  # Artificial test seed, never the scientific campaign seed.
SEMANTICS = "ordinary_controlled_diag(I,U);T=0.8;DF-prefix-S2;canonical-finite-RTE;no-state-prep;bit0=+1;bit1=-1;real=cosine;imag=sine"
read_events = []
protected_attempts = []
import_attempts = []


def install_guards():
    original_import = builtins.__import__
    forbidden = {"numpy", "scipy", "qiskit", "openfermion", "openfermionpyscf",
                 "pyscf", "trotterlib", "trottertracks", "torch", "cupy"}
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            import_attempts.append(name)
            raise RuntimeError("Scientific import forbidden in contract tests")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    def guard(event, args):
        if event in {"subprocess.Popen", "os.system", "os.kill", "os.killpg"}:
            protected_attempts.append(event)
            raise RuntimeError("Process execution/job mutation forbidden")
        if event not in {"open", "os.listdir", "os.scandir"} or not args:
            return
        path = args[0]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        path = pathlib.Path(os.fsdecode(path))
        if path.suffix.lower() in {".npz", ".npy", ".pickle", ".pkl", ".sqlite", ".sqlite3", ".db"} or any(
                part in {".runtime", "runtime", "checkpoint", "checkpoints", "cache", "caches"}
                for part in path.parts):
            protected_attempts.append(str(path))
            raise RuntimeError("Protected scientific/runtime input forbidden")
        if event == "open":
            read_events.append(str(path))
    sys.addaudithook(guard)


def load(name):
    return json.loads((HERE / name).read_text())


def instruction(name, qargs, cargs=None, params=None):
    return {"name": name, "operation_identity": "DECLARATIVE_STANDARD_EXAMPLE:" + name,
            "qargs": qargs, "cargs": cargs or [], "params": params or [],
            "definition": None, "condition": None, "control_state": None}


def artificial_circuit(axis="cosine"):
    # An ordered JSON declaration, never a Qiskit circuit or scientific builder.
    return {"format": "ordered-numerical-full-circuit-v1", "axis": axis,
            "qubits": [f"system:{i}" for i in range(8)] + ["ancilla:8"], "clbits": ["measurement:0"],
            "global_phase": v.real64(0.173),
            "instructions": [instruction("h", [8]), instruction("rz", [0], params=[v.real64(0.125)]),
                             instruction("cx", [8, 0]), instruction("measure", [8], [0])]}


def artificial_record(index, circuit, baseline=False):
    template = {"template_id": "B1-rank12-q1-r0-K0" if baseline else "B2-rank3-q1-r4-K2",
                "method": "B1" if baseline else "B2", "L_D": 12 if baseline else 3,
                "q": 1, "r": 0 if baseline else 4, "K": 0 if baseline else 2}
    record = {"schema_version": "h4-wrapper-record-v1", "artifact_scope": "SYNTHETIC_CONTRACT_TEST_ONLY",
        "geometry": "0.70", "hamiltonian_sha256": "a"*64, "df_sha256": "b"*64, "state_sha256": "c"*64,
        "candidate_template": template, "candidate_fingerprint": None, "axis": circuit["axis"],
        "trajectory_seed": None, "trajectory_index": None if baseline else index,
        "compiler_fingerprint": "d"*64, "environment_fingerprint": "e"*64, "source_commit": "f"*40,
        "wrapper_semantics": SEMANTICS, "wrapper_key": None,
        "numerical_circuit_fingerprint": v.circuit_fingerprint(circuit), "status": "COMPLETE",
        "metrics": dict(zip(v.METRICS, [3, 2, 1, 1, 4, 6])),
        "cache_reuse": False, "cache_owner_wrapper_key": None,
        "actual_transpile_invocation_id": "ARTIFICIAL_COMPILE_" + ("BASELINE" if baseline else str(index)),
        "sample_weight": {"numerator": 1, "denominator": 1 if baseline else 32}, "mandatory_stop": True}
    record["candidate_fingerprint"] = v.candidate_fingerprint(record)
    if not baseline:
        record["trajectory_seed"] = v.trajectory_seed(record, MASTER)
    record["wrapper_key"] = v.wrapper_key(record)
    return record


def main():
    started = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    os.nice(19)
    install_guards()
    schemas = {name: load(name) for name in [
        "scope_schema_v1.json", "plan_schema_v1.json", "checkpoint_schema_v1.json",
        "completion_ledger_schema_v1.json", "numerical_circuit_schema_v1.json", "result_schema_v1.json"]}
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schemas["checkpoint_schema_v1.json"])
    cases = []
    def accepts(name, fn):
        fn()
        cases.append({"name": name, "expected": "accept", "passed": True})
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, ValidationError, KeyError, TypeError):
            cases.append({"name": name, "expected": "reject", "passed": True})
            return
        raise AssertionError("Unexpected acceptance: " + name)
    plan = load("zero_compute_plan_v1.json")
    scope = load("scope_v1.json")
    accepts("exact scope", lambda: Draft202012Validator(schemas["scope_schema_v1.json"]).validate(scope))
    accepts("exact zero-compute plan", lambda: Draft202012Validator(schemas["plan_schema_v1.json"]).validate(plan))
    content = {k: val for k, val in plan.items() if k != "plan_fingerprint"}
    assert hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == plan["plan_fingerprint"]
    assert len(plan["templates"]) == 218 and sum(s["logical_wrapper_slots"] for s in plan["symbolic_slots"]) == 74784
    for key, changed in [("science_execution_authorized", True), ("execution_plan_sealed", True),
                         ("actual_science_source_commit", "a"*40), ("master_seed", 20261006),
                         ("next_stage_authorized", True), ("research_decision", "GO"),
                         ("distances_angstrom", ["0.70"]), ("future_resource_caps", {}),
                         ("science_memory_cap", 4), ("science_output_root", "/tmp/fake-run")]:
        altered = copy.deepcopy(plan); altered[key] = changed
        rejects("plan rejects " + key, lambda p=altered: Draft202012Validator(schemas["plan_schema_v1.json"]).validate(p))

    circuit = artificial_circuit()
    owner = artificial_record(0, circuit)
    reuse = artificial_record(1, circuit)
    reuse.update(cache_reuse=True, cache_owner_wrapper_key=owner["wrapper_key"], actual_transpile_invocation_id=None)
    baseline = artificial_record(None, circuit, baseline=True)
    records = {r["wrapper_key"]: r for r in [owner, reuse, baseline]}
    ledger = {"schema_version": "h4-external-completion-ledger-v1", "mandatory_stop": True,
        "entries": {k: {"status": "COMPLETE", "record_sha256": v.record_digest(r)} for k, r in records.items()},
        "reservations": {r["actual_transpile_invocation_id"]: {"status": "COMPLETE", "wrapper_key": r["wrapper_key"]}
                         for r in [owner, baseline]}}
    numerical = {k: r["numerical_circuit_fingerprint"] for k, r in records.items()}
    expected = {k: {f: r[f] for f in v.IDENTITY_FIELDS} for k, r in records.items()}
    def check(r=owner, c=circuit, rs=records, led=ledger, registry=numerical, exp=None):
        exp = expected[owner["wrapper_key"]] if exp is None else exp
        v.validate_record(r, exp, c, rs, led, registry, MASTER, validator)
    accepts("ledger structural schema", lambda: Draft202012Validator(schemas["completion_ledger_schema_v1.json"]).validate(ledger))
    accepts("circuit declaration structural schema", lambda: Draft202012Validator(schemas["numerical_circuit_schema_v1.json"]).validate(circuit))
    accepts("complete actual owner", check)
    accepts("identical circuit reuse different seed/index retains weight", lambda: check(reuse, exp=expected[reuse["wrapper_key"]]))
    accepts("baseline null seed AND null index", lambda: check(baseline, exp=expected[baseline["wrapper_key"]]))
    registered = copy.deepcopy(owner); registered.update(status="REGISTERED", metrics=None,
        numerical_circuit_fingerprint=None, actual_transpile_invocation_id=None)
    accepts("REGISTERED not-built fingerprint null", lambda: check(registered))
    reserved = copy.deepcopy(registered); reserved.update(status="RESERVED", actual_transpile_invocation_id="ARTIFICIAL_RESERVED")
    accepts("RESERVED schema is not completion", lambda: check(reserved))
    accepts("logical records and compile budget accounting", lambda: v.validate_accounting(list(records.values()), ledger))
    accepts("distinct trajectory seeds", lambda: v.validate_seed_uniqueness([owner, reuse]))
    other_axis = copy.deepcopy(owner); other_axis["axis"] = "sine"
    accepts("paired axes share trajectory seed", lambda: v.validate_seed_uniqueness([owner, other_axis]))

    for field in validator.schema["required"]:
        altered = copy.deepcopy(owner); altered.pop(field)
        rejects("missing record " + field, lambda r=altered: check(r))
    mutations = {
        "geometry": "0.80", "hamiltonian_sha256": "1"*64, "df_sha256": "2"*64,
        "state_sha256": "3"*64, "candidate_fingerprint": "4"*64, "axis": "sine",
        "trajectory_seed": owner["trajectory_seed"] + 1, "trajectory_index": 2,
        "source_commit": "5"*40, "compiler_fingerprint": "6"*64,
        "environment_fingerprint": "7"*64, "wrapper_key": "8"*64,
        "numerical_circuit_fingerprint": None, "wrapper_semantics": "phase ignored",
        "sample_weight": {"numerator": 1, "denominator": 1}, "metrics": None,
    }
    for field, value in mutations.items():
        altered = copy.deepcopy(owner); altered[field] = value
        rejects("mutated record " + field, lambda r=altered: check(r))
    baseline_bad = copy.deepcopy(baseline); baseline_bad["trajectory_index"] = 0
    rejects("baseline null-index mismatch", lambda: check(baseline_bad, exp=expected[baseline["wrapper_key"]]))
    for field, value in [("geometry", "0.80"), ("source_commit", "5"*40), ("axis", "sine")]:
        altered = copy.deepcopy(owner); altered[field] = value
        altered["candidate_fingerprint"] = v.candidate_fingerprint(altered)
        altered["trajectory_seed"] = v.trajectory_seed(altered, MASTER)
        altered["wrapper_key"] = v.wrapper_key(altered)
        rejects("rehash cannot replace independent expected " + field, lambda r=altered: check(r))
    for field in ["angle", "global_phase", "qubit_order", "instruction_order", "measurement_axis", "classical_bit_identity", "operation_identity"]:
        altered = copy.deepcopy(circuit)
        if field == "angle": altered["instructions"][1]["params"] = [v.real64(0.25)]
        elif field == "global_phase": altered["global_phase"] = v.real64(0.0)
        elif field == "qubit_order": altered["qubits"][0], altered["qubits"][1] = altered["qubits"][1], altered["qubits"][0]
        elif field == "instruction_order": altered["instructions"][0], altered["instructions"][1] = altered["instructions"][1], altered["instructions"][0]
        elif field == "measurement_axis": altered["axis"] = "sine"
        elif field == "classical_bit_identity": altered["clbits"] = ["changed:0"]
        else: altered["instructions"][1]["operation_identity"] = "changed"
        rejects("full circuit detects " + field, lambda c=altered: check(c=c))
        changed_record = copy.deepcopy(owner); changed_record["numerical_circuit_fingerprint"] = v.circuit_fingerprint(altered)
        changed_ledger = copy.deepcopy(ledger); changed_ledger["entries"][owner["wrapper_key"]]["record_sha256"] = v.record_digest(changed_record)
        rejects("independent numerical registry rejects rehash " + field,
                lambda r=changed_record, c=altered, led=changed_ledger: check(r, c, led=led))
    for text in ["nan", "inf", "-inf", "0X1.0000000000000P-3"]:
        changed = copy.deepcopy(circuit); changed["instructions"][1]["params"] = [{"real64_hex": text}]
        rejects("nonfinite/noncanonical parameter " + text, lambda c=changed: v.circuit_fingerprint(c))
    for number in [float("nan"), float("inf"), float("-inf")]:
        rejects("nonfinite direct float", lambda n=number: v.real64(n))
    accepts("finite complex exact encoding", lambda: v.check_numeric(v.complex128(0.25+0.5j)))
    assert v.real64(0.0) != v.real64(-0.0)
    changed = copy.deepcopy(circuit); changed["instructions"][1]["params"] = ["theta"]
    rejects("symbolic skeleton parameter", lambda: v.circuit_fingerprint(changed))

    no_owner = copy.deepcopy(records); no_owner.pop(owner["wrapper_key"])
    rejects("reuse owner absent", lambda: check(reuse, rs=no_owner, exp=expected[reuse["wrapper_key"]]))
    owner_reserved = copy.deepcopy(records); owner_reserved[owner["wrapper_key"]].update(
        status="RESERVED", metrics=None, numerical_circuit_fingerprint=None)
    rejects("reuse of RESERVED owner", lambda: check(reuse, rs=owner_reserved, exp=expected[reuse["wrapper_key"]]))
    changed = copy.deepcopy(reuse); changed["status"] = "RESERVED"
    rejects("RESERVED record cannot reuse", lambda: check(changed, exp=expected[reuse["wrapper_key"]]))
    for field in ["geometry", "candidate_fingerprint", "axis", "source_commit", "compiler_fingerprint", "environment_fingerprint"]:
        changed_records = copy.deepcopy(records)
        changed_records[owner["wrapper_key"]][field] = "0.80" if field == "geometry" else "sine" if field == "axis" else "9"*(40 if field == "source_commit" else 64)
        rejects("reuse cross-owner " + field, lambda rs=changed_records: check(reuse, rs=rs, exp=expected[reuse["wrapper_key"]]))
    for key in [owner["wrapper_key"], reuse["wrapper_key"]]:
        changed = copy.deepcopy(ledger); changed["entries"].pop(key)
        rejects("external ledger missing " + key[:8], lambda led=changed: check(reuse, led=led, exp=expected[reuse["wrapper_key"]]))
    changed = copy.deepcopy(ledger); changed["entries"][owner["wrapper_key"]]["record_sha256"] = "0"*64
    rejects("external digest altered", lambda: check(led=changed))
    changed = copy.deepcopy(owner); changed["metrics"]["rz_count"] += 1
    rejects("whole-record digest detects metric tamper", lambda: check(changed))
    changed = copy.deepcopy(owner); changed["record_sha256"] = "0"*64
    rejects("self-referential record digest field forbidden", lambda: check(changed))
    changed = copy.deepcopy(ledger); changed["reservations"].clear()
    rejects("actual compile reservation missing", lambda: check(led=changed))
    changed = copy.deepcopy(ledger); changed["reservations"]["ARTIFICIAL_UNRESOLVED"] = {"status": "AMBIGUOUS_AWAITING_REVIEW", "wrapper_key": owner["wrapper_key"]}
    rejects("ambiguous consumed reservation STOP", lambda: v.validate_accounting(list(records.values()), changed))
    rejects("duplicate logical wrapper", lambda: v.validate_accounting([owner, owner], ledger))
    changed = copy.deepcopy(baseline); changed["actual_transpile_invocation_id"] = owner["actual_transpile_invocation_id"]
    rejects("duplicate actual invocation", lambda: v.validate_accounting([owner, changed], ledger))
    changed = copy.deepcopy(reuse); changed["trajectory_seed"] = owner["trajectory_seed"]
    rejects("duplicate seed across different sample slots", lambda: v.validate_seed_uniqueness([owner, changed]))
    changed = copy.deepcopy(reuse); changed["cache_owner_wrapper_key"] = changed["wrapper_key"]
    changed_ledger = copy.deepcopy(ledger); changed_ledger["entries"][changed["wrapper_key"]]["record_sha256"] = v.record_digest(changed)
    changed_records = dict(records); changed_records[changed["wrapper_key"]] = changed
    rejects("reuse self/owner chain", lambda: check(changed, rs=changed_records, led=changed_ledger, exp=expected[reuse["wrapper_key"]]))
    summary = {"schema_version": "h4-geometry-result-v1", "status": "IMPLEMENTATION_GATE_FAILED",
        "source_commit": "f"*40, "plan_fingerprint": "a"*64, "completion_ledger_sha256": "b"*64,
        "signal_records": 0, "logical_wrapper_records": 0, "actual_transpile_invocations": 0,
        "research_decision": None, "next_stage_authorized": False,
        "automatic_research_decision_authorized": False, "mandatory_stop": True}
    accepts("reserved failure result shape (artificial)", lambda: Draft202012Validator(schemas["result_schema_v1.json"]).validate(summary))
    changed = dict(summary, status="GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW")
    rejects("incomplete cannot claim map completion", lambda: Draft202012Validator(schemas["result_schema_v1.json"]).validate(changed))
    assert not protected_attempts and not import_attempts
    examples = {"artifact_scope": "SYNTHETIC_CONTRACT_TEST_ONLY", "not_scientific_evidence": True,
        "master_seed_is_artificial": MASTER, "declarative_circuit": circuit,
        "records": list(records.values()), "external_ledger": ledger, "independent_numerical_registry": numerical}
    with (HERE / "synthetic_records_v1.json").open("x") as handle:
        json.dump(examples, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    result = {"status": "SYNTHETIC_CONTRACT_MUTATION_TESTS_PASS", "started_jst": started,
        "completed_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), "cases": cases,
        "passed": len(cases), "failed": 0, "skipped": 0,
        "schema_checks_and_semantic_checks_separate": True, "source_path": str(HERE / "contract_validator_v1.py"),
        "jsonschema_version": importlib.metadata.version("jsonschema"),
        "execution_command": "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/venvs/trotter-common/bin/python " + str(HERE / "run_contract_tests.py"),
        "protected_access_attempts": protected_attempts, "science_import_attempts": import_attempts,
        "guard_scope": "diagnostic Python import/audit guards, not an OS sandbox or future science validator",
        "file_read_events": read_events, "science_build_compile_transpile_calls": 0,
        "molecular_access": 0, "runtime_checkpoint_cache_access": 0, "GPU": 0,
        "shared_environment_changes": 0, "other_job_changes": 0, "commit_push": 0}
    with (HERE / "contract_tests_result_v1.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "passed": len(cases), "failed": 0, "skipped": 0,
                      "scientific_build_compile_transpile": 0, "protected_access": 0, "science_imports": 0}))


if __name__ == "__main__":
    main()
