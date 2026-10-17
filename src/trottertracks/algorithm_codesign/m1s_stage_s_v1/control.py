"""Source-bound, fail-closed authorization and one-shot preservation.

No authorization/grant is created by this module. Future execution consumes an
exclusive new marker only after separately committed user execution approval.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from jsonschema import Draft202012Validator

from .spec import (BASE, INPUTS, RESOURCES, PROPOSED_SCIENCE, SOURCE_FILES, ART,
                   AUTH_PREFIX, OUTPUT_PREFIX, EVIDENCE, contrast_keys, old_status)

_TOKEN = object()


class ExecutionClosed(RuntimeError):
    pass


class _Permit:
    def __init__(self, token, contract, provenance):
        if token is not _TOKEN:
            raise ExecutionClosed("invalid execution permit")
        self.token, self.contract, self.provenance = token, contract, provenance


def require_permit(permit):
    if type(permit) is not _Permit or permit.token is not _TOKEN:
        raise ExecutionClosed("separate source-bound execution permit required")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))


def inside(root, relative, prefix=None):
    if (not isinstance(relative, str) or Path(relative).is_absolute()
            or ".." in Path(relative).parts):
        raise ExecutionClosed("relative repository path required")
    q = (root / relative).resolve()
    if (not q.is_relative_to(root.resolve()) or (prefix and
            (not relative.startswith(prefix) or not q.is_relative_to((root / prefix).resolve())))):
        raise ExecutionClosed("path outside dedicated namespace")
    return q


def read_json(path, cap):
    if path.stat().st_size > cap:
        raise ExecutionClosed("input/metadata byte cap")
    return json.loads(path.read_bytes())


def validate_contract(c, adopted=False):
    if c.get("schema") != "m1s_stage_s_v1_contract" or c.get("base_commit") != BASE:
        raise ExecutionClosed("contract schema/base")
    if c.get("contract_grants_execution") is not False or c.get("mandatory_STOP") is not True:
        raise ExecutionClosed("contract cannot grant execution or waive STOP")
    if c.get("one_shot") is not True or c.get("retry") is not False:
        raise ExecutionClosed("one-shot/no-retry required")
    if c.get("inputs") != INPUTS or c.get("resources") != RESOURCES:
        raise ExecutionClosed("input/resource proposals changed; new SOURCE version needed")
    if c.get("proposed_science") != PROPOSED_SCIENCE:
        raise ExecutionClosed("unsupported science proposal")
    if adopted:
        if (c.get("adopted") is not True or c.get("coverage_model_adopted") is not True
                or c.get("selected_science") != PROPOSED_SCIENCE
                or c.get("status") != "ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL"):
            raise ExecutionClosed("candidate not formally adopted; no scientific input read")
    elif (c.get("adopted") is not False or c.get("coverage_model_adopted") is not False
          or c.get("selected_science") is not None):
        raise ExecutionClosed("preparation candidate must remain unadopted")


def authorize(root, contract_path, contract_commit, authorization_path, authorization_commit):
    # Read design/contract metadata only. This check precedes Git, authority, input and engine.
    c = read_json(inside(root, contract_path), RESOURCES["input_bytes_per_file"])
    validate_contract(c, adopted=True)
    if not all(isinstance(x, str) and len(x) == 40
               for x in [contract_commit, authorization_commit]):
        raise ExecutionClosed("fixed contract and authority commits required")
    a_path = inside(root, authorization_path, AUTH_PREFIX)
    a = read_json(a_path, RESOURCES["input_bytes_per_file"])
    if (a.get("authorized") is not True or a.get("one_shot") is not True
            or a.get("retry") is not False or a.get("scope") != "M1S_STAGE_S_CAP_EDGE_ONLY"
            or a.get("contract_commit") != contract_commit
            or a.get("contract_path") != contract_path
            or a.get("contract_sha256") != sha256(inside(root, contract_path).read_bytes())
            or not isinstance(a.get("user_explicit_execution_approval"), str)
            or not a["user_explicit_execution_approval"].strip()):
        raise ExecutionClosed("separate explicit user execution approval absent or mismatched")
    if git(root, "rev-parse", "HEAD").decode().strip() != authorization_commit:
        raise ExecutionClosed("run only fixed authority HEAD")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ExecutionClosed("tracked checkout dirty")
    for commit, path in [(contract_commit, contract_path), (authorization_commit, authorization_path)]:
        if git(root, "show", commit + ":" + path) != inside(root, path).read_bytes():
            raise ExecutionClosed("contract/authority not committed byte-identically")
    identity = c.get("source_identity")
    if not isinstance(identity, dict) or set(identity.get("files", {})) != set(SOURCE_FILES):
        raise ExecutionClosed("fixed source inventory missing")
    S = identity.get("commit")
    if not isinstance(S, str) or len(S) != 40 or a.get("source_commit") != S:
        raise ExecutionClosed("source binding missing")
    for earlier, later in [(BASE, S), (S, contract_commit), (contract_commit, authorization_commit)]:
        git(root, "merge-base", "--is-ancestor", earlier, later)
    for path, expected in identity["files"].items():
        raw = inside(root, path).read_bytes()
        if sha256(raw) != expected["sha256"] or len(raw) != expected["bytes"] or git(root, "show", S + ":" + path) != raw:
            raise ExecutionClosed("SOURCE bytes changed: " + path)
    for expected in INPUTS.values():
        raw = inside(root, expected["path"]).read_bytes()  # identity only, not decoded
        if (len(raw) != expected["bytes"] or sha256(raw) != expected["sha256"]
                or git(root, "show", expected["commit"] + ":" + expected["path"]) != raw):
            raise ExecutionClosed("saved input identity mismatch")
    if c.get("interpreter_sha256") != sha256(Path(sys.executable).read_bytes()):
        raise ExecutionClosed("interpreter identity mismatch")
    output = a.get("output_path")
    if inside(root, output, OUTPUT_PREFIX).exists():
        raise ExecutionClosed("output already exists; one-shot cannot retry")
    return _Permit(_TOKEN, c, {"S": S, "C": contract_commit, "A": authorization_commit,
                              "contract_sha256": a["contract_sha256"],
                              "authorization_path": authorization_path,
                              "output_path": output,
                              "inputs": INPUTS})


def seed_result(permit):
    require_permit(permit)
    return {"schema": "m1s_stage_s_v1_result", "status": "TECHNICAL_STOP",
            "evidence": EVIDENCE, "coverage_scope": "SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY",
            "provenance": permit.provenance, "failure": None,
            "scientific_GO": None, "RPE_success": None, "native_costs": None,
            "total_resources": None, "global_integer_impossibility": False,
            "one_shot_consumed": True, "retry_count": 0, "mandatory_STOP": True,
            "comparisons": [dict(zip(["split", "R", "delta", "T", "comparator"], k),
                                 old_pi3_gate_status=old_status(k),
                                 old_absolute_contrast_status=None,
                                 new_stage_s_status="NOT_EVALUATED") for k in contrast_keys()]}


def bounded_save(path, result, cap):
    raw = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    if len(raw) > cap:
        raise RuntimeError("result byte cap")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def validate_result(result, schema):
    """Schema and integer-proof metadata invariants; no saved-signal evaluation."""
    Draft202012Validator(schema).validate(result)
    keys = [tuple(row[k] for k in ["split", "R", "delta", "T", "comparator"])
            for row in result["comparisons"]]
    if len(keys) != len(set(keys)) or set(keys) != set(contrast_keys()):
        raise ValueError("result registry duplicates/missing")
    for row, key in zip(result["comparisons"], keys):
        if row["old_pi3_gate_status"] != old_status(key):
            raise ValueError("result changed old gate classification")
        status = row["new_stage_s_status"]
        if status in ["CAP_EDGE_STRICT_INTEGER_WITNESS", "CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED"]:
            if old_status(key) != "SATISFIED":
                raise ValueError("numerical proof added to carry-only comparison")
            ns, nc = row["n_shared_cap_left"], row["n_comparator_cap_left"]
            for value in [ns, nc]:
                if value["interval"] != [value["value"], value["value"]]:
                    raise ValueError("resolved integer not a singleton")
            if (ns["value"] < nc["value"]) != (status == "CAP_EDGE_STRICT_INTEGER_WITNESS"):
                raise ValueError("witness label inconsistent with integer ordering")
        if result["status"] == "STAGE_S_COMPLETE_AWAITING_USER_JUDGMENT":
            if old_status(key) == "NOT_SATISFIED" and status != "CARRIED_PI3_NOT_SATISFIED_NOT_EVALUATED":
                raise ValueError("missing pi3-negative carry")
            if key[0] == "G9" and status != "G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED":
                raise ValueError("G9 absolute unresolved not preserved")


def run(root, permit):
    require_permit(permit)
    output = inside(root, permit.provenance["output_path"], OUTPUT_PREFIX)
    output.mkdir(parents=True, exist_ok=False)
    # Exclusive one-shot consumption before numerical decode or any numerical helper import.
    with (output / "one_shot_consumed.json").open("x", encoding="utf-8") as f:
        json.dump({"consumed": True, "retry": False, "provenance": permit.provenance}, f)
    result = seed_result(permit)
    started = time.monotonic()
    schema = read_json(inside(root, ART + "/result_schema.json"), RESOURCES["input_bytes_per_file"])

    def save():
        validate_result(result, schema)
        bounded_save(output / "result.json", result, RESOURCES["result_bytes"])

    save()
    try:
        from .analysis import evaluate
        data = {name: read_json(inside(root, item["path"]), RESOURCES["input_bytes_per_file"])
                for name, item in INPUTS.items()}
        evaluate(data["V1"], data["PI3"], permit, result, save)
        result["status"] = "STAGE_S_COMPLETE_AWAITING_USER_JUDGMENT"
    except Exception as exc:
        result["status"] = "TECHNICAL_STOP"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
    finally:
        result["wall_seconds_observed"] = time.monotonic() - started
        try:
            save()
        except Exception as exc:
            # Keep the last valid full prefix and separately preserve the failure.
            result["status"] = "TECHNICAL_STOP"
            bounded_save(output / "failure.json", {
                "status": "TECHNICAL_STOP", "reason": str(exc)[:4000],
                "prior_failure": result.get("failure"),
                "last_valid_prefix": "result.json", "mandatory_STOP": True,
                "provenance": permit.provenance}, RESOURCES["log_bytes"])
    return result["status"]
