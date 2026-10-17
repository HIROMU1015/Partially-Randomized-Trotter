"""Representation, context validation and provenance shared by both paths.

No expectation, Pauli, BCH, commutator or step-bound arithmetic is shared.
The SPEC-01--03 checker is an explicitly shared trusted prerequisite.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from ..a2_spec01_03.checker import check_spec01_03
from ..a2_spec01_03.common import (
    ContractError, canonical_bytes, content_id, integer, keys_exact, loads_strict,
    node, rat, read_complex, read_rat, require, submission_identity,
)

VERSION = "A2_SPEC04_07_V1"
SCOPE = "SPEC04_07_SYNTHETIC_TECHNICAL_ONLY"
NAMESPACE = "spec04_07_evidence_v1"
MODULES = ("__init__.py", "common.py", "generator.py", "checker.py",
           "producer_algebra.py", "verifier_algebra.py", "producer_steps.py",
           "verifier_steps.py")
LIMITS = {
    "original_molecular_guarantee": "NOT_ESTABLISHED",
    "other_baseline_envelope": "NOT_EVALUATED",
    "total_resource_improvement": "NOT_EVALUATED",
    "end_to_end_execution_step": "NOT_ESTABLISHED",
    "ground_state_preparation": "NOT_IMPLEMENTED",
    "platform_certificate": "NOT_CERTIFIED",
    "A2_status": "NOT_ADOPTED", "A1_status": "CLOSED", "K1_F_status": "UNAPPROVED",
    "execution_authorization_issued": False,
}


def source_identity():
    directory = Path(__file__).parent
    return {"version": VERSION,
            "modules": {name: sha256((directory / name).read_bytes()).hexdigest()
                        for name in MODULES}}


def detached(value):
    return loads_strict(value) if type(value) is str else loads_strict(canonical_bytes(value).decode())


def prerequisite(raw, base, number, order, model_id):
    report = check_spec01_03(raw, base, expected_particle_number=number,
                            expected_fragment_order=order, expected_model_id=model_id)
    if report["verdict"] == "IDENTITY_MISMATCH_INCOMPARABLE":
        raise ContractError("CONTEXT_MISMATCH", "SPEC01_03", "Prerequisite context differs")
    require(report["verdict"] == "SPEC01_03_VERIFIED_EXACT_SYNTHETIC",
            "BASE_NOT_VERIFIED", "SPEC01_03", "Prerequisite rejected: " + report["verdict"])
    body = detached(base)["payload"]
    model = body["intake"]["model"]["payload"]
    return body, model, report


def trial(occupied, n, number):
    require(type(occupied) is list, "TRIAL", "occupied_modes", "List of canonical mode indices required")
    indices = [integer(a, "occupied_modes", natural=True) for a in occupied]
    require(indices == sorted(set(indices)) and all(i < n for i in indices)
            and len(indices) == integer(number, natural=True),
            "TRIAL", "occupied_modes", "Require sorted unique in-range indices, count N")
    return {"kind": "SINGLE_DETERMINANT", "occupied_modes": list(occupied),
            "n_modes": str(n), "N": number,
            "norm_squared": rat(1), "basis": "CREATION_ASCENDING_MODE0_LSB"}


def root_precision(bits):
    p = integer(bits, "sqrt_bits", natural=True)
    require(p > 0, "PRECISION", "sqrt_bits", "Explicit positive dyadic grid resolution required")
    return p


def policy_validate(policy):
    if policy is None:
        return None
    policy = detached(policy)
    keys_exact(policy, ("d", "alpha_spec", "root_width", "max_bisections"), "comparison_policy")
    d, alpha, width = (read_rat(policy[k]) for k in ("d", "alpha_spec", "root_width"))
    count = integer(policy["max_bisections"], "max_bisections", natural=True)
    require(d > 0 and 0 < alpha < 1 and width > 0 and count > 0,
            "INVALID_TASK_OR_CERTIFICATE", "comparison_policy", "Require d,width,count>0 and 0<alpha<1")
    return policy


def unpack(model, base):
    n = integer(model["n_modes"], natural=True)
    q = [[read_complex(c) for c in row] for row in model["one_body"]]
    squares = [(s["block_id"], read_rat(s["lambda"]),
                [[read_complex(c) for c in row] for row in s["g"]]) for s in model["squares"]]
    fragments = base["partition"]["payload"]["fragments"]
    caps = [read_rat(f["f"]) for f in fragments]
    return n, read_rat(model["constant"]), q, squares, fragments, caps


def failure(exc, stage, partial):
    verdict = ("IDENTITY_MISMATCH_INCOMPARABLE" if exc.code == "CONTEXT_MISMATCH"
               else "INCOMPLETE_CERTIFICATE" if exc.code == "INCOMPLETE_CERTIFICATE"
               else "TECHNICAL_ERROR_OR_INTERRUPTED" if exc.code == "SOURCE_IO"
               else "INVALID_TASK_OR_CERTIFICATE" if exc.code in ("INVALID_TASK_OR_CERTIFICATE", "TRIAL", "PRECISION")
               else "INVALID_CERTIFICATE")
    return {"schema": VERSION, "scope": SCOPE, "verdict": verdict,
            "stage": stage, "error": {"code": exc.code, "path": exc.path, "message": str(exc)},
            "partial": partial, "verified_quantities": [], "limitations": dict(LIMITS)}


def partial_identity(raw, base, evidence=None):
    result = {"submitted_input": submission_identity(raw), "base_evidence": submission_identity(base)}
    if evidence is not None:
        result["submitted_evidence"] = submission_identity(evidence)
    return result
