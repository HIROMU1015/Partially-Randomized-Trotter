"""L-only authority, fixed budget and durable owned-process audit.

Reuse frozen C fsync saves, hashes, locked nonrefundable budget and subreaper.
No numerical backend import before worker authorization. No implicit grant.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("L_reused_C_control", Path(__file__).with_name("h4_certificate_control.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
save, read, sha, digest, require, utc = C.save, C.read, C.sha, C.digest, C.require, C.utc
LIMITS = dict(forward=148, inverse=148, basis=11, support=324)
ZERO = ("SCF", "FCI", "DF", "state_solve", "reference", "law_mean", "bias", "sampling", "RNG_replay", "compile", "N_G")


def validate_contract(contract, *, active=False):
    require(contract["kind"] == "H4_L_PRODUCTION_CONTRACT_V1", "L_CONTRACT_KIND")
    require(contract["domain"] in ("SAVED_H4", "ARTIFICIAL"), "L_INPUT_DOMAIN")
    require(contract["precision_bits"] == 128 and type(contract["precision_bits"]) is int, "L_FIXED_PRECISION")
    require(contract["limits"] == LIMITS and all(type(x) is int for x in contract["limits"].values()), "L_FIXED_CALL_BUDGET")
    require(contract["forbidden_calls"] == {k: 0 for k in ZERO}, "L_FORBIDDEN_BUDGET")
    require(contract["expected_counts"] == dict(gates=148, bases=11, supports=324), "L_TARGET_COUNTS")
    require(contract["automatic_retry"] is False and contract["mandatory_STOP"] is True, "L_RETRY_OR_STOP")
    require(contract["formal_certificate_accepted"] is False and contract["N_G_authorized"] is False, "L_NO_ACCEPTANCE_OR_NG")
    require(contract["historical_compiler_target"] == "ORIGINAL_PROGRAM_UNCHANGED_MISSING_EVIDENCE", "L_K_TARGET_CHANGED")
    if active:
        require(all(contract[k] is True for k in ("execution_authorized", "grant_valid", "dispatch_enabled")), "L_EXECUTION_NOT_AUTHORIZED")
        require(contract.get("approval_file") and contract.get("approval_sha256"), "L_EXPLICIT_APPROVAL_REQUIRED")
        require(sha(contract["approval_file"]) == contract["approval_sha256"], "L_APPROVAL_HASH")
        approval = read(contract["approval_file"])
        require(approval == dict(kind="L_EXPLICIT_NUMERICAL_EXECUTION_APPROVAL", domain=contract["domain"],
            technical_seal_sha256=contract["technical_seal_sha256"], namespace=contract["output_namespace"],
            input_sha256=contract["input_sha256"], scope="DF_LOCAL_ONLY_FIXED128_NO_COMPILE_ACCEPTANCE_OR_NG"), "L_APPROVAL_SCOPE")
    return contract


def verify_seal(seal):
    require(seal["kind"] == "H4_L_TECHNICAL_SOURCE_NOT_EXECUTION_GRANT", "L_SEAL_KIND")
    require(seal["domain"] in ("SAVED_H4", "ARTIFICIAL"), "L_SEAL_DOMAIN")
    for group in ("code_files", "runtime_files", "input_files"):
        C.hashes(seal[group])
    require(sha(sys.executable) == seal["python_executable_sha256"], "L_EXECUTABLE_HASH")
    for path, expected in seal["code_git_files"].items():
        raw = subprocess.check_output(["git", "show", seal["SOURCE"] + ":" + path], cwd=ROOT)
        import hashlib
        require(hashlib.sha256(raw).hexdigest() == expected, "L_SOURCE_COMMIT")


def routing(contract, seal):
    require(contract["domain"] == seal["domain"], "L_ARTIFICIAL_REAL_DOMAIN_MISMATCH")
    require(contract["technical_SOURCE"] == seal["SOURCE"], "L_SOURCE_BINDING")
    require(contract["input_sha256"] == seal["input_files"], "L_INPUT_HASH_BINDING")
    require(contract["model_path"] == seal["model_path"] and contract["registry_path"] == seal["registry_path"], "L_INPUT_PATH_BINDING")


def issue_grant(contract_path, seal_path, namespace):
    contract = validate_contract(read(contract_path), active=True)
    seal = read(seal_path)
    verify_seal(seal)
    routing(contract, seal)
    require(sha(seal_path) == contract["technical_seal_sha256"], "L_SEAL_BINDING")
    ns = Path(namespace).resolve()
    require(str(ns) == contract["output_namespace"] and not ns.exists(), "L_FRESH_NAMESPACE_REQUIRED")
    ns.mkdir(parents=True)
    grant = dict(kind="L_SCIENCE_ONE_SHOT" if contract["domain"] == "SAVED_H4" else "L_ARTIFICIAL_ONE_SHOT",
        namespace=str(ns), contract_sha256=sha(contract_path), seal_sha256=sha(seal_path),
        approval_sha256=contract["approval_sha256"], one_shot=True)
    grant["grant_id"] = digest(grant)
    save(ns / "contract.json", contract)
    save(ns / "seal.json", seal)
    save(ns / "grant.json", grant)
    return grant


def authority(namespace, *, worker=False):
    ns = Path(namespace).resolve()
    contract = validate_contract(read(ns / "contract.json"), active=True)
    grant, seal = read(ns / "grant.json"), read(ns / "seal.json")
    kind = "L_SCIENCE_ONE_SHOT" if contract["domain"] == "SAVED_H4" else "L_ARTIFICIAL_ONE_SHOT"
    require(grant["kind"] == kind and grant["namespace"] == contract["output_namespace"] == str(ns), "L_GRANT_NAMESPACE_OR_KIND")
    require(grant["grant_id"] == digest({k: v for k, v in grant.items() if k != "grant_id"}), "L_GRANT_DIGEST")
    require(sha(ns / "contract.json") == grant["contract_sha256"] and sha(ns / "seal.json") == grant["seal_sha256"] == contract["technical_seal_sha256"], "L_GRANT_BINDING")
    require(grant["approval_sha256"] == contract["approval_sha256"], "L_APPROVAL_BINDING")
    require(not (ns / "STOP.json").exists(), "L_ALREADY_STOPPED")
    verify_seal(seal)
    routing(contract, seal)
    if worker:
        require(read(ns / "CONSUMED.json")["grant_id"] == grant["grant_id"] and (ns / "LAUNCH_CLAIM.json").is_file(), "L_WORKER_NOT_DISPATCHED")
    else:
        require(not (ns / "CONSUMED.json").exists(), "L_GRANT_ALREADY_USED")
    return contract, seal


def consume(namespace):
    contract, seal = authority(namespace)
    ns = Path(namespace).resolve()
    save(ns / "CONSUMED.json", dict(grant_id=read(ns / "grant.json")["grant_id"], UTC=utc(), nonrefundable=True))
    save(ns / "LAUNCH_CLAIM.json", dict(pid=os.getpid(), UTC=utc()))
    return contract, seal


def witness(directory, command):
    # The C subreaper implementation is reused in an isolated witness process.
    # Adapt only the scope label before durable creation of our NEW receipt.
    original_save = C.save
    def scoped_save(path, value, **kwargs):
        if type(value) is dict and value.get("scope") == "NEW_C_WORKER_ONLY":
            value = dict(value, scope="NEW_L_WORKER_ONLY")
        return original_save(path, value, **kwargs)
    C.save = scoped_save
    try:
        return C.witness(directory, command)
    finally:
        C.save = original_save


def launch(namespace):
    ns = Path(namespace).resolve()
    consume(ns)
    command = [sys.executable, "-B", str(ROOT / "scripts/run_h4_local_enclosure.py"), "--worker", str(ns)]
    witness_command = [sys.executable, "-B", str(ROOT / "scripts/run_h4_local_enclosure.py"),
                       "--witness", str(ns / "native"), "--command-json", json.dumps(command)]
    process = None
    try:
        process = subprocess.Popen(witness_command, start_new_session=True)
        code = process.wait()
        save(ns / "witness_exit.json", dict(exit_code=code, reaped=True, UTC=utc()))
        require(code == 0, "L_NATIVE_WORKER_FAILED")
        contract, seal = authority(ns, worker=True)
        require((ns / "worker_result.json").is_file(), "L_RESULT_MISSING")
        native = read(ns / "native" / "worker_native_exit.json")
        native_stop = read(ns / "native" / "STOP.json")
        require(native["exit_code"] == 0 and native["reaped"] is True
                and native_stop["all_owned_reaped"] is True, "L_NATIVE_RECEIPT_INCOMPLETE")
        result = read(ns / "worker_result.json")
        require(result["domain"] == contract["domain"] and result["precision_bits"] == 128
                and result["formal_certificate_accepted"] is False, "L_RESULT_SCOPE")
        verify_seal(seal)
        save(ns / "STOP.json", dict(status="L_LOCAL_EVIDENCE_SAVED_COMBINATION_BLOCKED", mandatory_STOP=True,
            native_exit=code, witness_reaped=True, formal_certificate_accepted=False,
            execution_authorized=False, grant_valid=False, dispatch_enabled=False, N_G_authorized=False, UTC=utc()))
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if not (ns / "STOP.json").exists():
            save(ns / "STOP.json", dict(status="FAILED_NO_RETRY", error=repr(error), mandatory_STOP=True, UTC=utc()))
        raise


def worker(namespace):
    ns = Path(namespace).resolve()
    contract, seal = authority(ns, worker=True)
    save(ns / "WORKER_CLAIM.json", dict(pid=os.getpid(), UTC=utc()))
    try:
        spec = importlib.util.spec_from_file_location("authorized_L_numeric", ROOT / "src/trotterlib/h4_local_enclosure.py")
        numeric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(numeric)
        model, registry = read(contract["model_path"]), read(contract["registry_path"])
        artificial = contract["domain"] == "ARTIFICIAL"
        if artificial:
            require(model.get("test_fixture_domain") == "MANUALLY_SUPPLIED_ARTIFICIAL"
                    and type(model.get("n_qubits")) is int and 0 < model["n_qubits"] <= 2,
                    "L_ARTIFICIAL_FIXTURE_ONLY_NO_H4")
        plan = numeric.plan_from_saved(model, registry, artificial=artificial)
        budget = C.CallBudget(ns, LIMITS)
        # Recheck authority before and after the numerical stage. Before every
        # reservation, cheap flags/STOP/approval binding are also checked.
        def before_call():
            current = validate_contract(read(ns / "contract.json"), active=True)
            require(digest(current) == digest(contract) and not (ns / "STOP.json").exists(), "L_AUTHORITY_CHANGED_DURING_RUN")
        result = numeric.diagnose(plan, budget, emit=lambda name, record: save(ns / name, record),
                                  authorize_before_call=before_call)
        authority(ns, worker=True)
        save(ns / "worker_result.json", result)
        save(ns / "worker_STOP.json", dict(status=result["status"], formal_certificate_accepted=False, mandatory_STOP=True, UTC=utc()))
    except BaseException as error:
        if not (ns / "worker_STOP.json").exists():
            save(ns / "worker_STOP.json", dict(status="FAILED_PARTIALS_PRESERVED_NO_RETRY", error=repr(error), mandatory_STOP=True, UTC=utc()))
        raise
