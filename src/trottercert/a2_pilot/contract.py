"""Fail-closed fixed contract and separately pinned user authorization checks."""

from hashlib import sha256
import re
import subprocess

from ..a2_spec01_03.common import (
    ContractError, canonical_bytes, content_id, digest, integer, keys_exact, loads_strict, require,
)
from ..a2_spec04_07.common import policy_validate, root_precision, trial
from .identity import source_identity

CONTRACT_SCHEMA = "A2_FIXED_DF_PILOT_CONTRACT_V1"
AUTH_SCHEMA = "A2_FIXED_DF_PILOT_EXECUTION_AUTHORIZATION_V1"
TEST_AUTH_SCHEMA = "ARTIFICIAL_AUTH_TEST_VECTOR_NOT_SCIENTIFIC_AUTHORIZATION"
CALLS = {name: "1" for name in ("adapter", "SPEC01_03_generator", "SPEC01_03_checker", "SPEC04_07_generator", "SPEC04_07_checker")}
CALLS["internal_SPEC01_03_rechecks"] = "2"
TASK = "FIXED_DF_EXACT_GROUND_CONDITIONAL_SUFFICIENT_STEP_ONLY"


def _unset(value):
    if type(value) is dict:
        return any(_unset(v) for v in value.values())
    if type(value) is list:
        return any(_unset(v) for v in value)
    return value == "UNSET"


def preflight(contract_bytes, auth_bytes, pins, *, artificial_test=False):
    """Must run before input IO, marker creation or numerical calls.

    Pins are independently supplied by the authorized launcher/user. JSON is not
    a cryptographic attestation of user intent; provenance still needs review.
    """
    keys_exact(pins, ("contract_sha256", "authorization_sha256", "source_commit", "input_manifest_id"), "pins")
    for k in ("contract_sha256", "authorization_sha256", "input_manifest_id"):
        digest(pins[k], k)
    require(type(contract_bytes) is bytes and type(auth_bytes) is bytes,
            "AUTHORIZATION", "documents", "Explicit pinned document bytes required")
    require(sha256(contract_bytes).hexdigest() == pins["contract_sha256"]
            and sha256(auth_bytes).hexdigest() == pins["authorization_sha256"],
            "AUTHORIZATION", "pins", "Externally intended document hashes differ")
    contract, auth = loads_strict(contract_bytes.decode()), loads_strict(auth_bytes.decode())
    fields = ("schema", "adopted", "contract_grants_execution", "scope", "source", "input", "task", "limits", "failure_policy", "state", "excluded", "execution_store")
    keys_exact(contract, fields, "contract")
    require(contract["schema"] == CONTRACT_SCHEMA and contract["scope"] == TASK,
            "CONTRACT", "schema/scope", "Wrong contract kind")
    require(contract["adopted"] is True and contract["contract_grants_execution"] is False,
            "NOT_ADOPTED", "contract", "Adoption and execution approval must remain separate")
    require(not _unset(contract), "UNSET_CONDITION", "contract", "All required scientific/technical choices need adoption")
    expected_auth = TEST_AUTH_SCHEMA if artificial_test else AUTH_SCHEMA
    keys_exact(auth, ("schema", "execution_authorized", "user_statement", "user_statement_sha256", "contract_sha256",
                      "source_commit", "input_manifest_id", "run_id", "one_shot", "retry"), "authorization")
    require(auth["schema"] == expected_auth and auth["execution_authorized"] is True
            and auth["one_shot"] is True and auth["retry"] == "NO_AUTOMATIC_RETRY",
            "AUTHORIZATION", "scope", "Separate one-shot execution approval required")
    require(type(auth["user_statement"]) is str and bool(auth["user_statement"].strip())
            and sha256(auth["user_statement"].encode()).hexdigest() == auth["user_statement_sha256"],
            "AUTHORIZATION", "user_statement", "Missing/corrupt approval provenance")
    require(type(auth["run_id"]) is str and re.fullmatch(r"[A-Za-z0-9_-]+", auth["run_id"]) is not None,
            "AUTHORIZATION", "run_id", "Safe explicit run identifier required")
    source = contract["source"]
    keys_exact(source, ("commit", "identity"), "source")
    require(re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is not None
            and source["commit"] == pins["source_commit"] == auth["source_commit"]
            and source["identity"] == source_identity(),
            "SOURCE_IDENTITY", "source", "Frozen SOURCE identity differs")
    inp = contract["input"]
    keys_exact(inp, ("commit", "repository_path", "manifest", "mapping", "expected_model_id", "provenance_basis"), "input")
    require(type(inp["repository_path"]) is str and not inp["repository_path"].startswith("/")
            and all(p not in ("", ".", "..") for p in inp["repository_path"].split("/")),
            "INPUT_PATH", "input", "Explicit repository-relative input path required")
    require(re.fullmatch(r"[0-9a-f]{40}", inp["commit"]) is not None, "INPUT_COMMIT", "input", "Fixed input commit required")
    if inp["expected_model_id"] != "DERIVE_EXACT_FROM_PINNED_RAW_AND_MAPPING":
        digest(inp["expected_model_id"], "expected_model_id")
    manifest_id = content_id("a2_input_manifest_v1", inp["manifest"])
    require(manifest_id == pins["input_manifest_id"] == auth["input_manifest_id"]
            and auth["contract_sha256"] == pins["contract_sha256"],
            "IDENTITY_MISMATCH", "authorization", "Approval does not bind the intended contract/input")
    task = contract["task"]
    keys_exact(task, ("occupied_modes", "fragment_order", "sqrt_bits", "comparison_policy", "cap_policy", "guarantees", "information_policy"), "task")
    root_precision(task["sqrt_bits"])
    mapping = inp["mapping"]
    n = integer(mapping["n_modes"], natural=True)
    trial(task["occupied_modes"], n, mapping["sector_N"])
    require(len(mapping["mode_order"]) == n and len(set(mapping["mode_order"])) == n,
            "MAPPING", "mode_order", "Complete distinct declared modes required")
    inventory = ["one_body"] + ["df:" + k for k in mapping["block_inventory"]]
    require(type(task["fragment_order"]) is list and len(task["fragment_order"]) == len(inventory)
            and set(task["fragment_order"]) == set(inventory),
            "MAPPING", "fragment_order", "Full fixed inventory including zero fragments")
    require(policy_validate(task["comparison_policy"]) is not None, "POLICY", "task", "Explicit comparison policy required")
    require(task["cap_policy"] == "ENTRYWISE_RE_IM_ABS_SUM_V1"
            and task["guarantees"] == ["RF0", "RF1", "EW2", "CENTERED_CBAR2_WITH_GROUPED_STRUCTURE"]
            and task["information_policy"] == "SAME_ALL_VERIFIED_CERTIFICATES_NO_FREE_ACQUISITION",
            "POLICY", "task", "Fixed guarantees/cap/strong baseline required")
    limits = contract["limits"]
    keys_exact(limits, ("calls", "output_bytes", "log_bytes", "cpu_cap", "wall_cap", "memory_cap", "worker_policy"), "limits")
    require(limits["calls"] == CALLS, "CALL_SCOPE", "limits", "One authorized pipeline; no new inputs/retries")
    require(integer(limits["output_bytes"], natural=True) > 0 and integer(limits["log_bytes"], natural=True) > 0,
            "OUTPUT_LIMIT", "limits", "Explicit positive output/log bounds required")
    require(all(limits[k] is None for k in ("cpu_cap", "wall_cap", "memory_cap"))
            and limits["worker_policy"] == "AVAILABLE_RESOURCES_NO_SHARED_CHANGES",
            "RESOURCE_POLICY", "limits", "No artificial CPU/RAM/wall caps")
    require(contract["failure_policy"] == {"one_shot": True, "retry": "NO_AUTOMATIC_RETRY", "stop": "MANDATORY_STOP",
                                           "claim_before_input_IO": True, "preserve_partial": True},
            "STOP_POLICY", "failure_policy", "Must consume attempt and preserve failures")
    require(contract["state"] == {"A2": "NOT_ADOPTED", "A1": "CLOSED", "K1_F": "UNAPPROVED"},
            "STATE", "state", "Research state must not change")
    require(contract["excluded"] == ["SPEC08", "QPE", "NATIVE_COMPILE", "STATE_PREPARATION", "ORIGINAL_MOLECULAR_CERTIFICATION", "SCIENTIFIC_GO_NO_GO"],
            "SCOPE", "excluded", "Fixed scope exclusions required")
    keys_exact(contract["execution_store"], ("canonical_repository", "ledger_relative"), "execution_store")
    require(type(contract["execution_store"]["canonical_repository"]) is str
            and contract["execution_store"]["canonical_repository"].startswith("/")
            and contract["execution_store"]["ledger_relative"] == "artifacts/qc_a2_pilot_execution",
            "STORE", "execution_store", "Single canonical execution repository/ledger must be adopted")
    return contract, auth


def verify_git_sources(repository, contract):
    """Read-only Git blob checks, no checkout/fetch; no numerical data decode."""
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=repository)
    require(git("remote", "get-url", "origin").decode().strip() ==
            "git@github.com:HIROMU1015/Partially-Randomized-Trotter.git",
            "REPOSITORY", "origin", "Only intended HIROMU1015 repository")
    source = contract["source"]
    for name, expected in source["identity"]["modules"].items():
        blob = git("show", source["commit"] + ":" + name)
        require(sha256(blob).hexdigest() == expected, "SOURCE_COMMIT", name, "SOURCE hash not present at fixed commit")
